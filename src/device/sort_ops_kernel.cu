#include <tuple>

#include "tensor_iterator.h"
#include "scalar_type.h"
#include "sorting_radix_sort.h"
#include "accumulate_type.h"
#include "device_allocator.h"
#include "tensor_loops.h"

// ======================= block sort =======================

template <typename method_t, typename key_t, typename value_t>
struct SegmentedBlockRadixSortPairsFunctor {
    DEVICE void operator()(ITEM &item) const {
        int seg_idx = item.block_idx_x();
        int seg_offset = seg_idx * num_elements_;
        auto method = method_t(item);
        method.load_keys(keys_in_ + seg_offset, num_elements_);
        method.load_values(
            values_in_ == nullptr ? nullptr : values_in_ + seg_offset,
            num_elements_);
        int begin_bit = 0;
        int end_bit = KeyTraits<key_t>::endbit();
        while (true) {
            method.rank_keys(begin_bit, end_bit);
            method.exchange_keys();
            method.exchange_values();
            begin_bit += method_t::RADIX_BITS;
            if (begin_bit >= end_bit)
                break;
        }
        method.store_keys(keys_out_ + seg_offset, num_elements_);
        method.store_values(values_out_ + seg_offset, num_elements_);
    }
    SegmentedBlockRadixSortPairsFunctor(
        const key_t *keys_in,
        key_t *keys_out,
        const value_t *values_in,
        value_t *values_out,
        int num_elements) :
        keys_in_(keys_in),
        keys_out_(keys_out),
        values_in_(values_in),
        values_out_(values_out),
        num_elements_(num_elements) {
    }

private:
    const key_t *keys_in_;
    key_t *keys_out_;
    const value_t *values_in_;
    value_t *values_out_;
    int num_elements_;
};

template <
    typename key_t,
    typename value_t,
    bool IS_DESCENDING,
    int KEYS_PER_ITEM,
    int BLOCK_SIZE,
    int WARP_SIZE>
void segmented_block_radix_sort_pairs_kernel(
    const key_t *keys_in,
    key_t *keys_out,
    const value_t *values_in,
    value_t *values_out,
    int num_segments,
    int num_elements) {
    using method_t = BlockRadixSort<
        key_t,
        BLOCK_SIZE,
        WARP_SIZE,
        KEYS_PER_ITEM,
        IS_DESCENDING,
        value_t>;
    auto kernel = SegmentedBlockRadixSortPairsFunctor<method_t, key_t, value_t>(
        keys_in, keys_out, values_in, values_out, num_elements);
    Launcher::GetInstance()->submit(method_t::LocalMemorySize(),
                                    {num_segments}, {BLOCK_SIZE}, kernel);
}

// ======================= upsweep =======================

template <typename method_t, typename key_t, typename value_t>
struct SegmentedRadixSortPairsUpsweepFunctor {
    DEVICE void operator()(ITEM &item) const {
        int num_tiles = (num_elements_ + method_t::PROCESSING_LENGTH - 1) / method_t::PROCESSING_LENGTH;
        int seg_idx = item.block_idx_x() / num_tiles;
        int tile_idx = item.block_idx_x() % num_tiles;
        auto keys_in_seg = keys_in_ + seg_idx * num_elements_;
        auto counts_seg = counts_ + seg_idx * method_t::RADIX_BUCKETS * num_tiles;
        int tile_offset = tile_idx * method_t::PROCESSING_LENGTH;
        int tile_end = tile_offset + method_t::PROCESSING_LENGTH;
        tile_end = tile_end > num_elements_ ? num_elements_ : tile_end;
        auto method = method_t(
            item,
            keys_in_seg,
            tile_idx,
            begin_bit_,
            end_bit_,
            num_tiles,
            counts_seg);
        method.run(tile_offset, tile_end);
    }
    SegmentedRadixSortPairsUpsweepFunctor(
        const key_t *keys_in,
        int *counts,
        int num_elements,
        int begin_bit,
        int end_bit) :
        keys_in_(keys_in),
        counts_(counts),
        num_elements_(num_elements),
        begin_bit_(begin_bit),
        end_bit_(end_bit) {
    }

private:
    const key_t *keys_in_;
    int *counts_;
    int num_elements_;
    int begin_bit_;
    int end_bit_;
};

template <
    typename key_t,
    typename value_t,
    bool IS_DESCENDING,
    int KEYS_PER_ITEM,
    int BLOCK_SIZE,
    int WARP_SIZE>
void segmented_radix_sort_pairs_upsweep_kernel(
    const key_t *keys_in,
    int *counts,
    int num_segments,
    int num_elements,
    int begin_bit,
    int end_bit) {
    using method_t = RadixSortUpsweep<
        key_t,
        BLOCK_SIZE,
        WARP_SIZE,
        KEYS_PER_ITEM,
        IS_DESCENDING,
        value_t>;
    int num_tiles = (num_elements + method_t::PROCESSING_LENGTH - 1) / method_t::PROCESSING_LENGTH;
    auto kernel = SegmentedRadixSortPairsUpsweepFunctor<method_t, key_t, value_t>(
        keys_in, counts, num_elements, begin_bit, end_bit);
    Launcher::GetInstance()->submit(method_t::LocalMemorySize(),
                                    {num_segments * num_tiles},
                                    {BLOCK_SIZE}, kernel);
}

// ======================= scan bins =======================

template <typename method_t>
struct SegmentedRadixSortPairsScanFunctor {
    DEVICE void operator()(ITEM &item) const {
        constexpr int RADIX_BUCKETS = 16;
        int seg_idx = item.block_idx_x();
        auto counts_seg = counts_ + seg_idx * RADIX_BUCKETS * num_tiles_;
        auto method = method_t(item, counts_seg);
        method.run(num_tiles_ * RADIX_BUCKETS);
    }
    SegmentedRadixSortPairsScanFunctor(int *counts, int num_tiles) :
        counts_(counts), num_tiles_(num_tiles) {
    }

private:
    int *counts_;
    int num_tiles_;
};

template <int KEYS_PER_ITEM, int BLOCK_SIZE, int WARP_SIZE>
void segmented_radix_sort_pairs_scan_kernel(
    int *counts,
    int num_tiles,
    int num_segments) {
    using method_t = RadixSortScanBins<BLOCK_SIZE, KEYS_PER_ITEM, WARP_SIZE>;
    auto kernel = SegmentedRadixSortPairsScanFunctor<method_t>(counts, num_tiles);
    Launcher::GetInstance()->submit(method_t::LocalMemorySize(),
                                    {num_segments}, {BLOCK_SIZE}, kernel);
}

// ======================= downsweep =======================

template <typename method_t, typename key_t, typename value_t>
struct SegmentedRadixSortPairsDownsweepFunctor {
    DEVICE void operator()(ITEM &item) const {
        int num_tiles = (num_elements_ + method_t::PROCESSING_LENGTH - 1) / method_t::PROCESSING_LENGTH;
        int seg_idx = item.block_idx_x() / num_tiles;
        int tile_idx = item.block_idx_x() % num_tiles;
        int seg_offset = seg_idx * num_elements_;
        int tile_offset = tile_idx * method_t::PROCESSING_LENGTH;
        auto counts_seg = counts_ + seg_idx * method_t::RADIX_BUCKETS * num_tiles;
        auto method = method_t(item);
        method.load_keys(keys_in_ + seg_offset, num_elements_, tile_offset);
        method.load_values(
            values_in_ == nullptr ? nullptr : values_in_ + seg_offset,
            num_elements_,
            tile_offset);
        method.load_bin_offsets(counts_seg, tile_idx, num_tiles);
        method.rank_keys(begin_bit_, end_bit_);
        method.exchange_and_store_keys(keys_out_ + seg_offset, num_elements_);
        method.exchange_and_store_values(values_out_ + seg_offset, num_elements_);
    }
    SegmentedRadixSortPairsDownsweepFunctor(
        const key_t *keys_in,
        key_t *keys_out,
        const value_t *values_in,
        value_t *values_out,
        int num_elements,
        int begin_bit,
        int end_bit,
        int *counts) :
        keys_in_(keys_in),
        keys_out_(keys_out),
        values_in_(values_in),
        values_out_(values_out),
        num_elements_(num_elements),
        begin_bit_(begin_bit),
        end_bit_(end_bit),
        counts_(counts) {
    }

private:
    const key_t *keys_in_;
    key_t *keys_out_;
    const value_t *values_in_;
    value_t *values_out_;
    int num_elements_;
    int begin_bit_;
    int end_bit_;
    int *counts_;
};

template <
    typename key_t,
    typename value_t,
    bool IS_DESCENDING,
    int KEYS_PER_ITEM,
    int BLOCK_SIZE,
    int WARP_SIZE>
void segmented_radix_sort_pairs_downsweep_kernel(
    const key_t *keys_in,
    key_t *keys_out,
    const value_t *values_in,
    value_t *values_out,
    int num_segments,
    int num_elements,
    int begin_bit,
    int end_bit,
    int *count) {
    using method_t = BlockRadixSort<
        key_t,
        BLOCK_SIZE,
        WARP_SIZE,
        KEYS_PER_ITEM,
        IS_DESCENDING,
        value_t>;
    int num_tiles = (num_elements + method_t::PROCESSING_LENGTH - 1) / method_t::PROCESSING_LENGTH;
    auto kernel =
        SegmentedRadixSortPairsDownsweepFunctor<method_t, key_t, value_t>(
            keys_in,
            keys_out,
            values_in,
            values_out,
            num_elements,
            begin_bit,
            end_bit,
            count);
    Launcher::GetInstance()->submit(method_t::LocalMemorySize(),
                                    {num_segments * num_tiles}, {BLOCK_SIZE}, kernel);
}

// ======================= large sort =======================

template <typename scalar_t>
struct ABBufferCopyFunctor {
    scalar_t operator()(scalar_t x) const {
        return x;
    }
};

template <
    typename key_t,
    typename value_t,
    bool IS_DESCENDING,
    int KEYS_PER_ITEM,
    int BLOCK_SIZE,
    int WARP_SIZE>
void segmented_radix_sort_pairs_kernel(
    const key_t *keys_in,
    key_t *keys_out,
    const value_t *values_in,
    value_t *values_out,
    int num_segments,
    int num_elements) {
    constexpr int TILE_PROCESSING_LENGTH = BLOCK_SIZE * KEYS_PER_ITEM;
    int num_tiles =
        (num_elements + TILE_PROCESSING_LENGTH - 1) / TILE_PROCESSING_LENGTH;
    constexpr int RADIX_BITS = 4;
    constexpr int RADIX_BUCKETS = 16;
    int begin_bit = 0;
    int end_bit = KeyTraits<key_t>::endbit();
    int *counts;
    key_t *keys_temp;
    value_t *values_temp;

    int device = Launcher::GetInstance()->device();

    DataPtr counts_data = DeviceAllocator::GetInstance()->allocate(
        num_segments * RADIX_BUCKETS * num_tiles * sizeof(int), device);
    DataPtr keys_temp_data = DeviceAllocator::GetInstance()->allocate(
        num_segments * num_elements * sizeof(key_t), device);
    DataPtr values_temp_data = DeviceAllocator::GetInstance()->allocate(
        num_segments * num_elements * sizeof(value_t), device);

    counts = (int *)counts_data.get();
    keys_temp = (key_t *)keys_temp_data.get();
    values_temp = (value_t *)values_temp_data.get();

    key_t *keys_in_ = const_cast<key_t *>(keys_in);
    key_t *keys_out_ = keys_temp;
    value_t *values_in_ = const_cast<value_t *>(values_in);
    value_t *values_out_ = values_temp;

    while (true) {
        segmented_radix_sort_pairs_upsweep_kernel<
            key_t,
            value_t,
            IS_DESCENDING,
            KEYS_PER_ITEM,
            BLOCK_SIZE,
            WARP_SIZE>(
            keys_in_, counts, num_segments, num_elements, begin_bit, end_bit);

        segmented_radix_sort_pairs_scan_kernel<
            KEYS_PER_ITEM,
            BLOCK_SIZE,
            WARP_SIZE>(counts, num_tiles, num_segments);

        segmented_radix_sort_pairs_downsweep_kernel<
            key_t,
            value_t,
            IS_DESCENDING,
            KEYS_PER_ITEM,
            BLOCK_SIZE,
            WARP_SIZE>(
            keys_in_,
            keys_out_,
            values_in_,
            values_out_,
            num_segments,
            num_elements,
            begin_bit,
            end_bit,
            counts);

        if (begin_bit == 0) {
            keys_in_ = keys_temp;
            keys_out_ = keys_out;
            values_in_ = values_temp;
            values_out_ = values_out;
        } else {
            std::swap(keys_in_, keys_out_);
            std::swap(values_in_, values_out_);
        }
        begin_bit += RADIX_BITS;
        if (begin_bit >= end_bit)
            break;
    }

    // Among basic types, the bit size of bool is not an even multiple of 4. AB
    // buffer switching is required.
    if constexpr (std::is_same<key_t, bool>::value) {
        auto input_calc = TrivialOffsetCalculator<2>();
        memory::array<char *, 2> data;
        if (keys_out) {
            data[0] = (char *)keys_out;
            data[1] = (char *)keys_temp;
            auto fn = ABBufferCopyFunctor<key_t>();
            auto vec_size = memory_access::can_vectorize_up_to<decltype(fn)>(data);
            launch_vectorized_kernel(
                num_segments * num_elements, fn, data, input_calc, vec_size);
        }
        if (values_out) {
            data[0] = (char *)values_out;
            data[1] = (char *)values_temp;
            auto fn = ABBufferCopyFunctor<value_t>();
            auto vec_size = memory_access::can_vectorize_up_to<decltype(fn)>(data);
            launch_vectorized_kernel(
                num_segments * num_elements, fn, data, input_calc, vec_size);
        }
    }
}

// ======================= interface =======================

template <
    typename key_t,
    typename value_t,
    bool IS_DESCENDING,
    int WARP_SIZE = GPU_WARP_SIZE>
void segmented_sort_pairs_(
    const key_t *keys_in,
    key_t *keys_out,
    const value_t *values_in,
    value_t *values_out,
    int num_segments,
    int num_elements) {
    constexpr int scaling_coef = sizeof(key_t) * sizeof(value_t) >= 64 ? 2 : 1; // Attempt to reduce register pressure for performance.
    if (num_elements > 4096 / scaling_coef) {
        // Considering register pressure, we use a problem size of 4096 to delineate
        // the boundary between single tile sort and group sort.
        segmented_radix_sort_pairs_kernel<
            key_t,
            value_t,
            IS_DESCENDING,
            4 / scaling_coef,
            512,
            WARP_SIZE>(
            keys_in, keys_out, values_in, values_out, num_segments, num_elements);
    } else if (num_elements > 2048 / scaling_coef) {
        segmented_block_radix_sort_pairs_kernel<
            key_t,
            value_t,
            IS_DESCENDING,
            4 / scaling_coef,
            1024,
            WARP_SIZE>(
            keys_in, keys_out, values_in, values_out, num_segments, num_elements);
    } else if (num_elements > 1024 / scaling_coef) {
        segmented_block_radix_sort_pairs_kernel<
            key_t,
            value_t,
            IS_DESCENDING,
            4 / scaling_coef,
            512,
            WARP_SIZE>(
            keys_in, keys_out, values_in, values_out, num_segments, num_elements);
    } else if (num_elements > 512 / scaling_coef) {
        segmented_block_radix_sort_pairs_kernel<
            key_t,
            value_t,
            IS_DESCENDING,
            4 / scaling_coef,
            256,
            WARP_SIZE>(
            keys_in, keys_out, values_in, values_out, num_segments, num_elements);
    } else if (num_elements > 256 / scaling_coef) {
        segmented_block_radix_sort_pairs_kernel<
            key_t,
            value_t,
            IS_DESCENDING,
            4 / scaling_coef,
            128,
            WARP_SIZE>(
            keys_in, keys_out, values_in, values_out, num_segments, num_elements);
    } else {
        segmented_block_radix_sort_pairs_kernel<
            key_t,
            value_t,
            IS_DESCENDING,
            4 / scaling_coef,
            64,
            WARP_SIZE>(
            keys_in, keys_out, values_in, values_out, num_segments, num_elements);
    }
}

template <typename key_t, typename value_t>
void segmented_sort_pairs(
    const key_t *keys_in,
    key_t *keys_out,
    const value_t *values_in,
    value_t *values_out,
    int num_segments,
    int num_elements,
    bool descending) {
    if (descending)
        segmented_sort_pairs_<key_t, value_t, true>(
            keys_in, keys_out, values_in, values_out, num_segments, num_elements);
    else
        segmented_sort_pairs_<key_t, value_t, false>(
            keys_in, keys_out, values_in, values_out, num_segments, num_elements);
}

template <typename key_t, typename value_t>
void sort_pairs(
    const key_t *keys_in,
    key_t *keys_out,
    const value_t *values_in,
    value_t *values_out,
    int num_elements,
    bool descending) {
    segmented_sort_pairs<key_t, value_t>(
        keys_in, keys_out, values_in, values_out, 1, num_elements, descending);
}

// ======================= Dispatch =======================

template <typename key_t, typename value_t, typename func_t>
inline void host_kvsort(
    key_t *kbegin,
    key_t *kend,
    value_t *vbegin,
    const func_t &fn) {
    for (auto kit = kbegin, vit = vbegin; kit != kend; kit++, vit++) {
        for (auto kit_ = kit, vit_ = vit; kit_ != kend; kit_++, vit_++) {
            if (!fn(*kit, *kit_)) {
                std::swap(*kit, *kit_);
                std::swap(*vit, *vit_);
            }
        }
    }
}

std::vector<int64_t> infer_dense_strides_dim_last(
    const Tensor &self,
    int64_t dim) {
    int64_t ndim = self.dim();
    // sort the strides in descending order according to its value,
    // keeping dim the last.
    std::vector<int64_t> strides = self.strides();
    strides[dim] = -1;
    std::vector<int64_t> original_dim(ndim);
    for (int64_t i = 0; i < ndim; i++) {
        original_dim[i] = i;
    }
    host_kvsort(
        strides.data(),
        strides.data() + ndim,
        original_dim.data(),
        std::greater<int64_t>());
    // generate contiguous strides on permuted dims
    std::vector<int64_t> new_strides(ndim);
    std::vector<int64_t> new_strides_unsort(ndim);
    int64_t cumprod = 1;
    for (int64_t i = 0; i < ndim; i++) {
        new_strides[ndim - 1 - i] = cumprod;
        cumprod *= self.sizes()[original_dim[ndim - 1 - i]];
    }
    // unsort new strides
    for (int64_t i = 0; i < ndim; i++) {
        new_strides_unsort[original_dim[i]] = new_strides[i];
    }
    return new_strides_unsort;
}

std::tuple<Tensor, Tensor> sort_stable_kernel(
    const Tensor &self,
    int64_t dim,
    bool descending) {
    dim = maybe_wrap_dim(dim, self.dim());
    int64_t numel = self.numel();
    int64_t nsort = self.sizes()[dim];

    CHECK_FAIL(
        nsort <= std::numeric_limits<int>::max(),
        "The dimension being sorted can not have more than INT_MAX elements.");
    const auto self_dtype = self.dtype();
    CHECK_FAIL(self_dtype != ScalarType::Bool,
               "Sort currently does not support bool dtypes.");

    Tensor self_;
    bool newself = false;
    if (self.is_contiguous() && self.stride(dim) == 1) {
        self_ = self;
    } else {
        auto new_strides_unsort = infer_dense_strides_dim_last(self, dim);
        self_ = empty_strided(self.sizes(), new_strides_unsort, self.dtype(), self.device());
        self_.copy_(self);
        newself = true;
    }

    Tensor values_tensor, indices_tensor;
    if (newself) {
        values_tensor = empty_strided(self_.sizes(), self_.strides(), self_.dtype(), self_.device());
        indices_tensor = empty_strided(self_.sizes(), self_.strides(), ScalarType::Long, self_.device());
    } else {
        values_tensor = empty_like(self_);
        indices_tensor = empty(self_.sizes(), ScalarType::Long, self_.device());
    }

    if (numel > 0) {
        DISPATCH_BASIC_TYPES(
            self.dtype(),
            "sort_stable_kernel",
            [&]() {
                auto self_ptr = self_.data_ptr<scalar_t>();
                int nsegments = numel / nsort;
                segmented_sort_pairs<scalar_t, int64_t>(
                    self_ptr,
                    values_tensor.data_ptr<scalar_t>(),
                    nullptr,
                    indices_tensor.data_ptr<int64_t>(),
                    nsegments,
                    nsort,
                    descending);
            });
    }

    if (newself) {
        Tensor values_tensor_ = empty_like(self);
        Tensor indices_tensor_ = empty(self.sizes(), ScalarType::Long, self.device());
        values_tensor_.copy_(values_tensor);
        indices_tensor_.copy_(indices_tensor);
        return std::make_tuple(values_tensor_, indices_tensor_);
    } else {
        return std::make_tuple(values_tensor, indices_tensor);
    }
}
