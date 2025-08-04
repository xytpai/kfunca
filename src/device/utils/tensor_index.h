#pragma once

#include <limits>
#include <tuple>
#include <utility>
#include <iostream>
#include <vector>
#include <array>

#include "function_traits.h"
#include "tensor_iterator.h"
#include "scalar_type.h"
#include "tensor_offset_calculator.h"
#include "tensor_memory_access.h"
#include "exception.h"
#include "array.h"
#include "launcher.h"

static constexpr int launch_bound2 = 4;
static constexpr int launch_size_nd = 128;

template <int nt, int vt, typename func_t>
struct IndexElementwiseKernel {
    DEVICE void operator()(ITEM &item) const {
        const auto tid = item.thread_idx_x();
        const auto nv = nt * vt;
        auto idx = nv * item.block_idx_x() + tid;
#pragma unroll
        for (int i = 0; i < vt; i++) {
            if (idx < N_) {
                f_(idx);
                idx += nt;
            }
        }
    }
    IndexElementwiseKernel(int64_t N, func_t f) :
        N_(N), f_(f) {
    }

private:
    int64_t N_;
    func_t f_;
};

template <int nt, int vt, typename func_t>
static void launch_index_kernel(const int64_t N, const func_t f) {
    CHECK_FAIL(N >= 0 && N <= std::numeric_limits<int32_t>::max());
    if (N == 0) {
        return;
    }
    auto kernel = IndexElementwiseKernel<nt, vt, func_t>(N, f);
    [[maybe_unused]] int nb = (N + nt * vt - 1) / (nt * vt);
    Launcher::GetInstance()->submit(0, {nb}, {nt}, kernel);
}

template <typename offset_fn_t, typename func_t>
struct IndexingOffsetFunctor {
    DEVICE void operator()(int idx) const {
        const auto offsets = offset_fn_.get(idx);
        auto out_data = out_ptr_ + offsets[0];
        auto in_data = in_ptr_ + offsets[1];

        int64_t offset = 0;
#pragma unroll
        for (int i = 0; i < num_indices_; i++) {
            int64_t index = *reinterpret_cast<int64_t *>(index_ptrs_[i] + offsets[2]);
            // assert(-sizes_[i] <= index && index < sizes_[i] && "index out of bounds");
            if (index < 0) {
                index += sizes_[i];
            }
            offset += index * strides_[i];
        }

        f_(out_data, in_data, offset);
    }
    IndexingOffsetFunctor(
        char *out_ptr,
        const char *in_ptr,
        int num_indices,
        memory::array<int64_t, MAX_TENSOR_DIMS> sizes,
        memory::array<int64_t, MAX_TENSOR_DIMS> strides,
        memory::array<char *, MAX_TENSOR_DIMS> index_ptrs,
        offset_fn_t offset_fn,
        func_t f) :
        out_ptr_(out_ptr),
        in_ptr_(in_ptr),
        num_indices_(num_indices),
        sizes_(sizes),
        strides_(strides),
        index_ptrs_(index_ptrs),
        offset_fn_(offset_fn),
        f_(f) {
    }

private:
    char *out_ptr_;
    const char *in_ptr_;
    int num_indices_;
    memory::array<int64_t, MAX_TENSOR_DIMS> sizes_;
    memory::array<int64_t, MAX_TENSOR_DIMS> strides_;
    memory::array<char *, MAX_TENSOR_DIMS> index_ptrs_;
    offset_fn_t offset_fn_;
    func_t f_;
};

template <typename func_t>
void gpu_index_kernel(
    TensorIterator &iter,
    const std::vector<int64_t> index_size,
    const std::vector<int64_t> index_stride,
    const func_t f) {
    const auto num_indices = index_size.size();
    CHECK_FAIL(num_indices == index_stride.size());
    CHECK_FAIL(static_cast<int64_t>(num_indices) == iter.ntensors() - 2);

    if (iter.numel() == 0) {
        return;
    }

    if (!iter.can_use_32bit_indexing()) {
        for (auto &sub_iter : iter.with_32bit_indexing()) {
            gpu_index_kernel<func_t>(sub_iter, index_size, index_stride, f);
        }
        return;
    }

    auto sizes = memory::array<int64_t, MAX_TENSOR_DIMS>{};
    auto strides = memory::array<int64_t, MAX_TENSOR_DIMS>{};
    auto index_ptrs = memory::array<char *, MAX_TENSOR_DIMS>{};
    for (unsigned i = 0; i < num_indices; i++) {
        sizes[i] = index_size[i];
        strides[i] = index_stride[i];
        index_ptrs[i] = (char *)iter.data_ptr(i + 2);
    }

    char *out_ptr = static_cast<char *>(iter.data_ptr(0));
    char *in_ptr = static_cast<char *>(iter.data_ptr(1));

    auto offset_calc = make_offset_calculator<3>(iter);
    auto offset_fn = IndexingOffsetFunctor<decltype(offset_calc), func_t>(
        out_ptr, in_ptr, num_indices, sizes, strides, index_ptrs, offset_calc, f);
    launch_index_kernel<launch_size_nd, launch_bound2>(iter.numel(), offset_fn);
}
