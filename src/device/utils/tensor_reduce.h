#pragma once

#include <optional>

#include "launcher.h"
#include "device_common.h"
#include "function_traits.h"
#include "tensor_iterator.h"
#include "tensor_offset_calculator.h"
#include "scalar_type.h"
#include "exception.h"
#include "data_ptr.h"
#include "intrusive_ptr.h"
#include "device_allocator.h"

// returns reduced fraction numerator & denominator
HOST_DEVICE static void reduce_fraction(size_t &numerator, size_t &denominator) {
    // get GCD of num and denom using Euclid's algorithm.
    // Can replace this with std::gcd if we ever support c++17.
    size_t a = denominator;
    size_t b = numerator;
    while (b != 0) {
        a %= b;
        // swap(a,b)
        size_t tmp = a;
        a = b;
        b = tmp;
    }

    // a is now the GCD
    numerator /= a;
    denominator /= a;
}

template <typename scalar_t>
int get_output_vec_size(TensorIterator &iter) {
    int vec_size = 4;
    auto update_vec_size = [&vec_size](uint64_t n) {
        while (n % vec_size != 0) {
            vec_size /= 2;
        }
    };

    uint64_t base_address = reinterpret_cast<uint64_t>(iter.data_ptr(iter.noutputs())) / sizeof(scalar_t);
    update_vec_size(base_address);

    const int output_index = iter.num_reduce_dims();
    update_vec_size(iter.shape(output_index));

    int j = 0;
    for (int c = 0; c < iter.ndim(); c++) {
        auto i = iter.strides(iter.noutputs())[c];
        if (j != output_index) {
            update_vec_size(i / sizeof(scalar_t));
        }
        j++;
    }
    return vec_size;
}

template <typename index_t>
static OffsetCalculator<2, index_t> make_output_calculator(TensorIterator &iter) {
    int num_reduce_dims = iter.num_reduce_dims();
    int num_output_dims = iter.ndim() - num_reduce_dims;
    int input_index = iter.ntensors() - 1;
    int output_index = 0;
    std::array<const int64_t *, 2> strides = {
        iter.strides(output_index) + num_reduce_dims,
        iter.strides(input_index) + num_reduce_dims,
    };
    auto shape = iter.shape() + num_reduce_dims;
    return OffsetCalculator<2, index_t>(num_output_dims, shape, strides.data());
}

template <typename index_t>
static OffsetCalculator<1, index_t> make_input_calculator(TensorIterator &iter) {
    int num_reduce_dims = iter.num_reduce_dims();
    int input_index = iter.ntensors() - 1;
    std::array<const int64_t *, 1> strides = {
        iter.strides(input_index),
    };
    return OffsetCalculator<1, index_t>(num_reduce_dims, iter.shape(), strides.data());
}

template <typename out_scalar_t, typename func_t>
struct func_wrapper_t {
    using arg_t = typename binary_function_traits<func_t>::arg1_t;
    using scalar_t = typename binary_function_traits<func_t>::arg2_t;

    func_t combine;
    static DEVICE_INLINE out_scalar_t project(arg_t arg) {
        return (out_scalar_t)arg;
    }
    static DEVICE_INLINE arg_t gpu_shfl_down(ITEM &item, arg_t arg, int offset) {
        return GPU_SHFL_DOWN(item, arg, offset);
    }

    static DEVICE arg_t translate_idx(arg_t acc, int64_t /*idx*/) {
        return acc;
    }

    func_wrapper_t(const func_t &op) :
        combine(op) {
    }

    // wrap a normal reduction that ignores the index
    DEVICE arg_t reduce(arg_t acc, scalar_t val, int64_t idx) const {
        return combine(acc, val);
    }
};

template <typename scalar_t, typename func_t>
func_wrapper_t<scalar_t, func_t> func_wrapper(const func_t &op) {
    return func_wrapper_t<scalar_t, func_t>{op};
}

template <typename T>
struct mnt_wrapper {
    static constexpr int MAX_NUM_THREADS = 512;
};

struct ReduceConfig {
    static constexpr int BLOCK_X = 0;
    static constexpr int BLOCK_Y = 1;
    static constexpr int CTA = 2;

    ReduceConfig(int element_size_bytes, int num_outputs, int num_inputs) :
        element_size_bytes(element_size_bytes), num_outputs(num_outputs), num_inputs(num_inputs) {
    }
    int element_size_bytes;
    int num_outputs;
    int num_inputs;
    int step_input = 1;
    int step_output = 1;
    int ctas_per_output = 1;
    int input_mult[3] = {0, 0, 0};
    int output_mult[2] = {0, 0};

    int block_width;
    int block_height;
    int num_threads;

    bool vectorize_input = false;
    int output_vec_size = 1;

    template <typename T>
    void set_block_dimension(int64_t dim_x, int64_t dim_y) {
        const int max_num_threads = mnt_wrapper<T>::MAX_NUM_THREADS / output_vec_size;
        int dim_x_pow2 = dim_x < max_num_threads ? static_cast<int>(last_pow2(dim_x)) : max_num_threads;
        int dim_y_pow2 = dim_y < max_num_threads ? static_cast<int>(last_pow2(dim_y)) : max_num_threads;
        block_width = std::min(dim_x_pow2, int(GPU_WARP_SIZE));
        block_height = std::min(dim_y_pow2, int(max_num_threads / block_width));
        block_width = std::min(dim_x_pow2, int(max_num_threads / block_height));
        num_threads = block_width * block_height;
    }

    int split_input(int parallelism) {
        int step = step_input;
        step_input *= parallelism;
        return step;
    }

    int split_output(int parallelism) {
        int step = step_output;
        step_output *= parallelism;
        return step;
    }

    std::vector<int> block() const {
        std::vector<int> v = {block_width, block_height};
        return v;
    }

    std::vector<int> grid() const {
        std::vector<int> v = {div_up(num_outputs / output_vec_size, step_output), ctas_per_output};
        return v;
    }

    HOST_DEVICE bool should_block_x_reduce() const {
        return input_mult[BLOCK_X] != 0;
    }

    HOST_DEVICE bool should_block_y_reduce() const {
        return input_mult[BLOCK_Y] != 0;
    }

    HOST_DEVICE bool should_global_reduce() const {
        return input_mult[CTA] != 0;
    }

    DEVICE bool should_store(ITEM &item, int output_idx) const {
        return output_idx < num_outputs && (!should_block_x_reduce() || item.thread_idx_x() == 0) && (!should_block_y_reduce() || item.thread_idx_y() == 0);
    }

    DEVICE bool should_reduce_tail(ITEM &item) const {
        return (!should_block_y_reduce() || item.thread_idx_y() == 0) && (!should_global_reduce() || item.block_idx_y() == 0);
    }

    DEVICE int input_idx(ITEM &item) const {
        int lane = item.thread_idx_x();
        int warp = item.thread_idx_y();
        int cta2 = item.block_idx_y();
        return (lane * input_mult[BLOCK_X] + warp * input_mult[BLOCK_Y] + cta2 * input_mult[CTA]);
    }

    template <int output_vec_size>
    DEVICE int output_idx(ITEM &item) const {
        int lane = item.thread_idx_x();
        int warp = item.thread_idx_y();
        int cta1 = item.block_idx_x();
        return (lane * output_mult[BLOCK_X] + warp * output_mult[BLOCK_Y] + cta1 * step_output) * output_vec_size;
    }

    DEVICE int shared_memory_offset(ITEM &item, int offset) const {
        return item.thread_idx_x() + (item.thread_idx_y() + offset) * item.thread_range_x();
    }

    DEVICE int staging_memory_offset(ITEM &item, int cta2) const {
        int offset = cta2 + item.block_idx_x() * item.block_range_y();
        if (!should_block_x_reduce()) {
            offset = item.thread_idx_x() + offset * item.thread_range_x();
        }
        return offset;
    }

    int shared_memory_size() const {
        if (!should_block_y_reduce() && (!should_block_x_reduce() || block_width <= GPU_WARP_SIZE)) {
            return 0;
        }
        return element_size_bytes * num_threads * output_vec_size;
    }

    int64_t global_memory_size() const {
        if (!should_global_reduce()) {
            return 0;
        }
        auto size = (int64_t)element_size_bytes * num_outputs * ctas_per_output;
        if (!should_block_x_reduce()) {
            size *= block_width * output_vec_size;
        }
        return size;
    }

    int semaphore_size() const {
        if (!should_global_reduce()) {
            return 0;
        }
        return sizeof(int) * grid()[0];
    }

    int values_per_thread() const {
        return div_up(num_inputs, step_input);
    }
};

template <typename arg_t, typename scalar_t, int vt0, int input_vec_size = vt0>
ReduceConfig set_reduce_config(TensorIterator &iter) {
    // Start by assuming that each thread handles a single output and all
    // the inputs for that output.
    int64_t num_outputs = iter.num_output_elements();
    int64_t inputs_per_output = iter.numel() / num_outputs;
    int input_index = iter.ntensors() - 1;

    auto config = ReduceConfig(sizeof(arg_t), num_outputs, inputs_per_output);

    int64_t dim0;
    int64_t dim1;
    int64_t fastest_moving_stride;
    bool reduction_on_fastest_striding_dimension;

    if (iter.ndim() > 0) {
        // Adjust block size to map block width to fastest changing dimension of input
        // tensor. This grants the best possible memory accessing pattern, given that
        // for non-contiguous tensor with space in between, we cannot have perfect
        // memory coalescing.
        reduction_on_fastest_striding_dimension =
            (iter.num_reduce_dims() == iter.ndim()) || (iter.strides(/*arg=*/input_index)[0] < iter.strides(/*arg=*/input_index)[iter.num_reduce_dims()]);
        // Notice that dim0 & dim1 does NOT guarantee any launch configuration here!
        // dim0 & dim1 are more like the upper bound of the block dimension. The
        // actual launch config and reduction scheme is determined by setting values
        // to `config.input_mult` and `config.output_mult`.
        // We try to max out dim1 so that we have enough threads per CTA to deliver
        // performance for larger problem size.
        if (reduction_on_fastest_striding_dimension) {
            // Map block.x to the fastest reducing dimension. It implies:
            //   1. block_x_reduce is required.
            //   2. block.y now max out to num_outputs.
            dim0 = inputs_per_output;
            dim1 = num_outputs;
            fastest_moving_stride = iter.strides(/*arg=*/input_index)[0];
        } else {
            // Map block.x to the fastest non reducing dimension. It implies:
            //   1. block_x_reduce is turned off.
            //   2. block.y now max out to inputs_per_output.
            dim0 = num_outputs;
            dim1 = inputs_per_output;
            fastest_moving_stride = iter.strides(/*arg=*/input_index)[iter.num_reduce_dims()];
        }
    } else {
        reduction_on_fastest_striding_dimension = true;
        fastest_moving_stride = sizeof(scalar_t);
        dim0 = 1;
        dim1 = 1;
    }

    // We do vectorization to gain better memory access, there are two cases which we call
    // "vectorize along input" and "vectorize along output". Note that the "input/output"
    // here does not mean we are vectorizing load/store instructions. We always only vectorize
    // load instructions.
    //
    // Case 1: "vectorize along input"
    // This case happens when we are reducing along fastest moving dimesion. In such case, threads
    // with the same threadIdx_y works on the same reduction cooperatively and will produce results
    // for the same output. In such case, values in each loaded vector always correspond to the same output.
    //
    // Case 2: "vectorize along output"
    // This case happens when the fastest moving dimesion is not the dimension of reduction. In such case,
    // threads with different threadIdx_x are independent and will produce results for different outputs.
    // In such case, values in each loaded vector always correspond to different outputs.
    if (fastest_moving_stride == sizeof(scalar_t)) {
        if (reduction_on_fastest_striding_dimension && dim0 > 128 && iter.num_reduce_dims() == 1 && vt0 >= input_vec_size) {
            // Case 1: "vectorize along input"
            // Note that if vt0 < ReduceConfig::vec_size, then this means the register pressure could be high, in such case,
            // we should avoid vectorization.
            config.vectorize_input = true;
            dim0 /= input_vec_size;
        } else if (!reduction_on_fastest_striding_dimension) {
            // Case 2: "vectorize along output"
            config.output_vec_size = get_output_vec_size<scalar_t>(iter);
            dim0 /= config.output_vec_size;
        }
    }

    // Adjust block_width and block_height
    config.set_block_dimension<scalar_t>(dim0, dim1);

    int block_width = config.block_width;
    int block_height = config.block_height;

    if (iter.ndim() == 0 || reduction_on_fastest_striding_dimension) {
        // Split the input across lanes if the input is contiguous in the reduced
        // dimension. This will require reduction between threads using warp
        // shuffle instructions and shared memory (if block_width > GPU_WARP_SIZE).
        config.input_mult[0] = config.split_input(block_width);
    } else {
        // Otherwise split the output across lanes in a warp.
        config.output_mult[0] = config.split_output(block_width);
    }

    constexpr int min_values_per_thread = 16;
    constexpr int max_values_per_thread = 256;

    const int warp_split_threshold =
        std::min<int>(block_height * 16, max_values_per_thread);
    bool split_across_warps = config.values_per_thread() >= warp_split_threshold;
    const int num_mp = Launcher::GetInstance()->multi_processor_count();

    if (split_across_warps) {
        // Divide the input across warps in a thread-block, if that leaves at least
        // 16 elements to be summed by each thread. This will require inter-warp
        // reduction using shared memory.
        config.input_mult[1] = config.split_input(block_height);
    } else {
        // Otherwise, each warp handles a separate output.
        config.output_mult[1] = config.split_output(block_height);
    }

    int max_threads_per_mp = Launcher::GetInstance()->max_threads_per_multi_processor();

    const int blocks_per_sm = max_threads_per_mp / config.num_threads;
    const int target_grid_size = num_mp * blocks_per_sm;
    int grid = config.grid()[0];
    if (config.input_mult[1] != 0 && config.values_per_thread() >= max_values_per_thread && grid <= target_grid_size) {
        // Divide the input across thread-blocks if the amount of work per-thread
        // is large enough and the size of the output is small enough. This will
        // require a reduction using global memory.
        // If we decide to split input across blocks, as long as we can get enough
        // number of blocks (`target_grid_size`) to balance SM, we should still
        // make the number of values per thread large for best performance.
        int ctas_per_output1 = div_up(target_grid_size, grid);
        int ctas_per_output2 = div_up(config.values_per_thread(), min_values_per_thread);
        int ctas_per_output3 = div_up(config.values_per_thread(), max_values_per_thread);
        // We want the minimum of ctas_per_output1 and ctas_per_output2, so that each thread can have
        // a large number of values to deal with. But we don't want values_per_thread to be larger than
        // max_values_per_thread
        config.ctas_per_output = std::max(std::min<int>(ctas_per_output1, ctas_per_output2), ctas_per_output3);
        if (config.ctas_per_output > 1) {
            config.input_mult[2] = config.split_input(config.ctas_per_output);
        }
    }
    return config;
};

template <typename scalar_t, typename ops_t, typename index_t, typename out_scalar_t = scalar_t, int vt0 = 4, int input_vec_size = vt0>
struct ReduceOp {
    using traits = function_traits<decltype(&ops_t::reduce)>;
    using arg_t = typename std::decay<typename traits::template arg<0>::type>::type;

    using InputCalculator = OffsetCalculator<1, index_t>;
    using OutputCalculator = OffsetCalculator<2, index_t>;

    static constexpr bool can_accumulate_in_output =
        std::is_convertible_v<arg_t, out_scalar_t> && std::is_convertible_v<out_scalar_t, arg_t>;

    ops_t ops;
    arg_t ident;
    ReduceConfig config;
    InputCalculator input_calc;
    OutputCalculator output_calc;
    const void *src;
    const char *dst[2]; // it accepts at most two destinations
    // acc_buf used for accumulation among sub Tensor Iterator when accumulation on
    // output is not permissible
    void *acc_buf;
    // cta_buf used for accumulation between blocks during global reduction
    void *cta_buf;
    int *semaphores;
    int64_t base_idx;
    bool accumulate;
    bool final_output;
    int noutputs;

    ReduceOp(
        ops_t ops,
        ReduceConfig config,
        InputCalculator input_calc,
        OutputCalculator output_calc,
        const void *src,
        char *dst0,
        std::optional<char *> dst1,
        void *acc_buf,
        void *cta_buf,
        int *semaphores,
        arg_t ident,
        int noutputs,
        int64_t base_idx) :
        ops(ops),
        ident(ident),
        config(config),
        input_calc(input_calc),
        output_calc(output_calc),
        src(src),
        acc_buf(acc_buf),
        cta_buf(cta_buf),
        semaphores(semaphores),
        base_idx(base_idx),
        noutputs(noutputs) {
        dst[0] = dst0;
        if (dst1.has_value()) {
            dst[1] = dst1.value();
        }
    }

    template <int output_vec_size>
    DEVICE void run(ITEM &item) const {
        auto shared_memory = item.shared_ptr();
        index_t output_idx = config.output_idx<output_vec_size>(item);
        index_t input_idx = config.input_idx(item);
        auto base_offsets1 = output_calc.get(output_idx)[1];

        using arg_vec_t = std::array<arg_t, output_vec_size>;
        arg_vec_t value;

        if (output_idx < config.num_outputs && input_idx < config.num_inputs) {
            const scalar_t *input_slice = (const scalar_t *)((const char *)src + base_offsets1);
            value = thread_reduce<output_vec_size>(item, input_slice);
        }

        if (config.should_block_y_reduce()) {
            value = block_y_reduce<output_vec_size>(item, value, shared_memory);
        }
        if (config.should_block_x_reduce()) {
            value = block_x_reduce<output_vec_size>(item, value, shared_memory);
        }

        using out_ptr_vec_t = std::array<out_scalar_t *, output_vec_size>;
        using offset_vec_t = std::array<index_t, output_vec_size>;
        offset_vec_t base_offsets;
        out_ptr_vec_t out;

#pragma unroll
        for (int i = 0; i < output_vec_size; i++) {
            base_offsets[i] = output_calc.get(output_idx + i)[0];
            out[i] = (out_scalar_t *)((char *)dst[0] + base_offsets[i]);
        }

        arg_vec_t *acc = nullptr;
        if (acc_buf != nullptr) {
            size_t numerator = sizeof(arg_t);
            size_t denominator = sizeof(out_scalar_t);
            reduce_fraction(numerator, denominator);
            acc = (arg_vec_t *)((char *)acc_buf + (base_offsets[0] * numerator / denominator));
        }

        if (config.should_global_reduce()) {
            value = global_reduce<output_vec_size>(item, value, acc, shared_memory);
        } else if (config.should_store(item, output_idx)) {
            if (accumulate) {
#pragma unroll
                for (int i = 0; i < output_vec_size; i++) {
                    value[i] = ops.translate_idx(value[i], base_idx);
                }
            }

            if (acc == nullptr) {
                if (accumulate) {
                    value = accumulate_in_output<output_vec_size, can_accumulate_in_output>(out, value);
                }
                if (final_output) {
                    set_results_to_output<output_vec_size>(value, base_offsets);
                } else {
#pragma unroll
                    for (int i = 0; i < output_vec_size; i++) {
                        *(out[i]) = get_accumulated_output<can_accumulate_in_output>(out[i], value[i]);
                    }
                }
            } else {
                if (accumulate) {
#pragma unroll
                    for (int i = 0; i < output_vec_size; i++) {
                        value[i] = ops.combine((*acc)[i], value[i]);
                    }
                }
                if (final_output) {
                    set_results_to_output<output_vec_size>(value, base_offsets);
                } else {
                    *acc = value;
                }
            }
        }
    }

    template <int output_vec_size>
    DEVICE std::array<arg_t, output_vec_size> thread_reduce(ITEM &item, const scalar_t *data) const {
        if (config.vectorize_input) {
            // reduce at the header of input_slice where memory is not aligned,
            // so that thread_reduce will have an aligned memory to work on.
            return {input_vectorized_thread_reduce_impl(item, data)};
        } else {
            index_t element_stride = input_calc.strides_[0][0] / sizeof(scalar_t);
            bool is_contiguous = (input_calc.dims == 1 && element_stride == 1);
            if (is_contiguous) {
                return thread_reduce_impl<output_vec_size>(item, data, [](index_t idx) { return idx; });
            } else if (input_calc.dims == 1) {
                return thread_reduce_impl<output_vec_size>(item, data, [&](index_t idx) { return idx * element_stride; });
            } else {
                return thread_reduce_impl<output_vec_size>(item, data, [&](index_t idx) { return input_calc.get(idx)[0] / sizeof(scalar_t); });
            }
        }
    }

    DEVICE arg_t input_vectorized_thread_reduce_impl(ITEM &item, const scalar_t *data) const {
        index_t end = config.num_inputs;

        // Handle the head of input slice where data is not aligned
        arg_t value = ident;
        constexpr int align_bytes = alignof(memory::aligned_array<scalar_t, input_vec_size>);
        constexpr int align_elements = align_bytes / sizeof(scalar_t);
        int shift = ((uint64_t)data) % align_bytes / sizeof(scalar_t);
        auto thread_idx_x = item.thread_idx_x();
        if (shift > 0) {
            data -= shift;
            end += shift;
            if (thread_idx_x >= shift && thread_idx_x < align_elements && config.should_reduce_tail(item)) {
                value = ops.reduce(value, *(data + thread_idx_x), thread_idx_x - shift);
            }
            end -= align_elements;
            data += align_elements;
            shift = align_elements - shift;
        }

        // Do the vectorized reduction
        using load_t = memory::aligned_array<scalar_t, input_vec_size>;

        index_t idx = config.input_idx(item);
        const index_t stride = config.step_input;

        // Multiple accumulators to remove dependency between unrolled loops.
        arg_t value_list[input_vec_size];
        value_list[0] = value;

#pragma unroll
        for (int i = 1; i < input_vec_size; i++) {
            value_list[i] = ident;
        }

        while (idx * input_vec_size + input_vec_size - 1 < end) {
            const auto values_vec = memory::load_vector<input_vec_size>(data, idx);
#pragma unroll
            for (index_t i = 0; i < input_vec_size; i++) {
                value_list[i] = ops.reduce(value_list[i], values_vec.val[i], shift + idx * input_vec_size + i);
            }
            idx += stride;
        }

        // tail
        index_t tail_start = end - end % input_vec_size;
        if (config.should_reduce_tail(item)) {
            int idx = tail_start + thread_idx_x;
            if (idx < end) {
                const auto value = *(data + idx);
                value_list[0] = ops.reduce(value_list[0], value, idx + shift);
            }
        }

// combine accumulators
#pragma unroll
        for (int i = 1; i < input_vec_size; i++) {
            value_list[0] = ops.combine(value_list[0], value_list[i]);
        }
        return value_list[0];
    }

    template <int output_vec_size, typename offset_calc_t>
    DEVICE std::array<arg_t, output_vec_size> thread_reduce_impl(ITEM &item, const scalar_t *data_, offset_calc_t calc) const {
        index_t idx = config.input_idx(item);
        const index_t end = config.num_inputs;
        const index_t stride = config.step_input;

        using arg_vec_t = std::array<arg_t, output_vec_size>;
        using load_t = memory::aligned_array<scalar_t, output_vec_size>;

        // Multiple accumulators to remove dependency between unrolled loops.
        arg_vec_t value_list[vt0];

#pragma unroll
        for (int i = 0; i < vt0; i++) {
#pragma unroll
            for (int j = 0; j < output_vec_size; j++) {
                value_list[i][j] = ident;
            }
        }

        load_t values[vt0];

        while (idx + (vt0 - 1) * stride < end) {
#pragma unroll
            for (index_t i = 0; i < vt0; i++) {
                const auto offset = calc(idx + i * stride) / output_vec_size;
                values[i] = memory::load_vector<output_vec_size>(data_, offset);
            }
#pragma unroll
            for (index_t i = 0; i < vt0; i++) {
#pragma unroll
                for (index_t j = 0; j < output_vec_size; j++) {
                    value_list[i][j] = ops.reduce(value_list[i][j], values[i].val[j], idx + i * stride);
                }
            }
            idx += stride * vt0;
        }

        // tail
        int idx_ = idx;
#pragma unroll
        for (index_t i = 0; i < vt0; i++) {
            if (idx >= end) {
                break;
            }
            const auto offset = calc(idx) / output_vec_size;
            values[i] = memory::load_vector<output_vec_size>(data_, offset);
            idx += stride;
        }
        idx = idx_;
#pragma unroll
        for (index_t i = 0; i < vt0; i++) {
            if (idx >= end) {
                break;
            }
#pragma unroll
            for (index_t j = 0; j < output_vec_size; j++) {
                value_list[i][j] = ops.reduce(value_list[i][j], values[i].val[j], idx);
            }
            idx += stride;
        }

// combine accumulators
#pragma unroll
        for (int i = 1; i < vt0; i++) {
#pragma unroll
            for (index_t j = 0; j < output_vec_size; j++) {
                value_list[0][j] = ops.combine(value_list[0][j], value_list[i][j]);
            }
        }
        return value_list[0];
    }

    template <int output_vec_size>
    DEVICE std::array<arg_t, output_vec_size> block_x_reduce(ITEM &item, std::array<arg_t, output_vec_size> value, char *shared_memory) const {
        using args_vec_t = std::array<arg_t, output_vec_size>;
        int dim_x = item.thread_range_x();
        args_vec_t *shared = (args_vec_t *)shared_memory;
        auto thread_idx_x = item.thread_idx_x();
        if (dim_x > GPU_WARP_SIZE) {
            int address_base = thread_idx_x + item.thread_idx_y() * item.thread_range_x();
            shared[address_base] = value;
            for (int offset = dim_x / 2; offset >= GPU_WARP_SIZE; offset >>= 1) {
                item.barrier();
                if (thread_idx_x < offset && thread_idx_x + offset < item.thread_range_x()) {
                    args_vec_t other = shared[address_base + offset];
#pragma unroll
                    for (int i = 0; i < output_vec_size; i++) {
                        value[i] = ops.combine(value[i], other[i]);
                    }
                    shared[address_base] = value;
                }
            }
            dim_x = GPU_WARP_SIZE;
        }

        item.barrier();

        for (int offset = 1; offset < dim_x; offset <<= 1) {
#pragma unroll
            for (int i = 0; i < output_vec_size; i++) {
                arg_t other = ops.gpu_shfl_down(item, value[i], offset);
                value[i] = ops.combine(value[i], other);
            }
        }
        return value;
    }

    template <int output_vec_size>
    DEVICE std::array<arg_t, output_vec_size> block_y_reduce(ITEM &item, std::array<arg_t, output_vec_size> value, char *shared_memory) const {
        using args_vec_t = std::array<arg_t, output_vec_size>;
        args_vec_t *shared = (args_vec_t *)shared_memory;
        shared[config.shared_memory_offset(item, 0)] = value;
        auto thread_idx_y = item.thread_idx_y();
        for (int offset = item.thread_range_y() / 2; offset > 0; offset >>= 1) {
            item.barrier();
            if (thread_idx_y < offset && thread_idx_y + offset < item.thread_range_y()) {
                args_vec_t other = shared[config.shared_memory_offset(item, offset)];
#pragma unroll
                for (int i = 0; i < output_vec_size; i++) {
                    value[i] = ops.combine(value[i], other[i]);
                }
                shared[config.shared_memory_offset(item, 0)] = value;
            }
        }
        return value;
    }

    DEVICE bool mark_block_finished(ITEM &item, char *shared_memory) const {
        auto is_last_block_done_shared = reinterpret_cast<bool *>(shared_memory);

        item.barrier();

        if (item.thread_idx_x() == 0 && item.thread_idx_y() == 0) {
            int prev_blocks_finished = atomicAdd(&semaphores[item.block_idx_x()], 1);
            *is_last_block_done_shared = (prev_blocks_finished == item.block_range_y() - 1);
        }

        item.barrier();

        return *is_last_block_done_shared;
    }

    template <int output_vec_size, bool can_acc>
    DEVICE std::array<arg_t, output_vec_size> accumulate_in_output(
        std::array<out_scalar_t *, output_vec_size> out,
        std::array<arg_t, output_vec_size> value,
        typename std::enable_if_t<can_acc> * = nullptr) const {
        std::array<arg_t, output_vec_size> ret;
#pragma unroll
        for (int i = 0; i < output_vec_size; i++) {
            ret[i] = ops.combine(*(out[i]), value[i]);
        }
        return ret;
    }

    template <bool can_acc>
    DEVICE out_scalar_t get_accumulated_output(
        out_scalar_t *out, arg_t value,
        typename std::enable_if_t<can_acc> * = nullptr) const {
        return (out_scalar_t)value;
    }

    // This function should never be called --
    // it's the version of `accumulate_in_output`
    // when accumulation in the output is not possible.
    template <int output_vec_size, bool can_acc>
    DEVICE std::array<arg_t, output_vec_size> accumulate_in_output(
        std::array<out_scalar_t *, output_vec_size>,
        std::array<arg_t, output_vec_size>,
        typename std::enable_if_t<!can_acc> * = nullptr) const {
        return {arg_t{}};
    }

    // This function should never be called --
    // it's the version of `get_accumulated_output`
    // when accumulation in the output is not possible.
    template <bool can_acc>
    DEVICE out_scalar_t get_accumulated_output(
        out_scalar_t *out, arg_t value,
        typename std::enable_if_t<!can_acc> * = nullptr) const {
        return *out;
    }

    template <class T>
    DEVICE void set_results(const T x, const index_t base_offset) const {
        auto res = (out_scalar_t *)((char *)dst[0] + base_offset);
        *res = x;
    }

    // Currently implemented for max of two outputs
    template <class T1, class T2>
    DEVICE void set_results(const std::pair<T1, T2> x, const index_t base_offset) const {
        if (noutputs >= 1) {
            auto res0 = (T1 *)((char *)dst[0] + base_offset);
            *res0 = x.first;
        }
        if (noutputs >= 2) {
            // base offset is computed assuming element size being sizeof(T1), so we need to make a
            // correction to obtain the correct base offset
            auto res1 = (T2 *)((char *)dst[1] + base_offset / sizeof(T1) * sizeof(T2));
            *res1 = x.second;
        }
    }

    template <int output_vec_size>
    DEVICE void set_results_to_output(std::array<arg_t, output_vec_size> value, std::array<index_t, output_vec_size> base_offset) const {
#pragma unroll
        for (int i = 0; i < output_vec_size; i++) {
            set_results(ops.project(value[i]), base_offset[i]);
        }
    }

    template <int output_vec_size>
    DEVICE std::array<arg_t, output_vec_size> global_reduce(ITEM &item, std::array<arg_t, output_vec_size> value, std::array<arg_t, output_vec_size> *acc, char *shared_memory) const {
        using arg_vec_t = std::array<arg_t, output_vec_size>;
        using out_ptr_vec_t = std::array<out_scalar_t *, output_vec_size>;
        using offset_vec_t = std::array<index_t, output_vec_size>;

        arg_vec_t *reduce_buffer = (arg_vec_t *)cta_buf;
        index_t output_idx = config.output_idx<output_vec_size>(item);
        offset_vec_t base_offsets;
        out_ptr_vec_t out;

#pragma unroll
        for (int i = 0; i < output_vec_size; i++) {
            base_offsets[i] = output_calc.get(output_idx + i)[0];
            out[i] = (out_scalar_t *)((char *)dst[0] + base_offsets[i]);
        }

        bool should_store = config.should_store(item, output_idx);
        if (should_store) {
            index_t offset = config.staging_memory_offset(item, item.block_idx_y());
            reduce_buffer[offset] = value;
        }

        item.memory_order_fence(); // make sure writes are globally visible
        item.barrier();            // if multiple warps in this block wrote to staging, make sure they're all done
        bool is_last_block_done = mark_block_finished(item, shared_memory);

        if (is_last_block_done) {
            item.memory_order_fence(); // complete the acquire pattern after atomic
            for (auto &v : value) {
                v = ident;
            }
            if (config.should_block_x_reduce()) {
                index_t input_offset = item.thread_idx_x() + item.thread_idx_y() * item.thread_range_x();
                index_t step = item.thread_range_x() * item.thread_range_y();
                for (; input_offset < config.ctas_per_output; input_offset += step) {
                    index_t idx = config.staging_memory_offset(item, input_offset);
                    arg_vec_t next = reduce_buffer[idx];
#pragma unroll
                    for (int i = 0; i < output_vec_size; i++) {
                        value[i] = ops.combine(value[i], next[i]);
                    }
                }
            } else {
                index_t input_offset = item.thread_idx_y();
                index_t step = item.thread_range_y();
                for (; input_offset < config.ctas_per_output; input_offset += step) {
                    index_t idx = config.staging_memory_offset(item, input_offset);
                    arg_vec_t next = reduce_buffer[idx];
#pragma unroll
                    for (int i = 0; i < output_vec_size; i++) {
                        value[i] = ops.combine(value[i], next[i]);
                    }
                }
            }
            value = block_y_reduce<output_vec_size>(item, value, shared_memory);
            if (config.should_block_x_reduce()) {
                value = block_x_reduce<output_vec_size>(item, value, shared_memory);
            }
            if (should_store) {
                if (accumulate) {
#pragma unroll
                    for (int i = 0; i < output_vec_size; i++) {
                        value[i] = ops.translate_idx(value[i], base_idx);
                    }
                }

                if (acc == nullptr) {
                    if (accumulate) {
                        value = accumulate_in_output<output_vec_size, can_accumulate_in_output>(out, value);
                    }
                    if (final_output) {
                        set_results_to_output<output_vec_size>(value, base_offsets);
                    } else {
#pragma unroll
                        for (int i = 0; i < output_vec_size; i++) {
                            *(out[i]) = get_accumulated_output<can_accumulate_in_output>(out[i], value[i]);
                        }
                    }
                } else {
                    if (accumulate) {
#pragma unroll
                        for (int i = 0; i < output_vec_size; i++) {
                            value[i] = ops.combine((*acc)[i], value[i]);
                        }
                    }
                    if (final_output) {
                        set_results_to_output<output_vec_size>(value, base_offsets);
                    } else {
                        *acc = value;
                    }
                }
            }
        }

        return value;
    }
};

template <int nt, int output_vec_size, typename R>
struct ReduceKernel {
    DEVICE void operator()(ITEM &item) const {
        reduction_.template run<output_vec_size>(item);
    }
    ReduceKernel(R reduction) :
        reduction_(reduction) {
    }

private:
    R reduction_;
};

template <int max_threads, typename R>
static void launch_reduce_kernel(const ReduceConfig &config, const R &reduction) {
    auto block = config.block();
    auto grid = config.grid();
    int shared_memory = config.shared_memory_size();
    auto l = Launcher::GetInstance();

    switch (config.output_vec_size) {
    case 4:
        l->submit(shared_memory, grid, block, ReduceKernel<max_threads / 4, 4, R>(reduction));
        break;
    case 2:
        l->submit(shared_memory, grid, block, ReduceKernel<max_threads / 2, 2, R>(reduction));
        break;
    default:
        l->submit(shared_memory, grid, block, ReduceKernel<max_threads / 1, 1, R>(reduction));
    }
}

class AccumulationBuffer {
public:
    AccumulationBuffer() {
    }

    AccumulationBuffer(size_t acc_t_size, size_t out_t_size, char *out_ptr, int64_t size, int device) {
        out_ptr_ = (char *)out_ptr;
        if (out_t_size >= acc_t_size) {
            // reusing output buffer for accumulation.
            acc_ptr_ = (char *)out_ptr;
            numerator_ = 1;
            denominator_ = 1;
        } else {
            buffer_ = DeviceAllocator::GetInstance()->allocate(size, device);
            acc_ptr_ = (char *)buffer_.get();
            numerator_ = acc_t_size;
            denominator_ = out_t_size;
            reduce_fraction(numerator_, denominator_);
        }
    }

    char *get_acc_slice(char *out_ptr) {
        if (acc_ptr_ == nullptr) {
            return nullptr;
        }
        return acc_ptr_ + ((out_ptr - out_ptr_) * numerator_ / denominator_);
    }

private:
    char *acc_ptr_ = nullptr;
    char *out_ptr_ = nullptr;
    size_t numerator_;
    size_t denominator_;
    DataPtr buffer_;
};

template <typename scalar_t, typename out_scalar_t, int vt0 = 4, int input_vec_size = vt0, typename ops_t, typename ident_t = double>
inline void gpu_reduce_kernel(TensorIterator &iter, const ops_t &ops, ident_t ident = 0,
                              AccumulationBuffer *acc_buf_ptr = nullptr, int64_t base_idx = 0) {
    CHECK_FAIL(iter.numel() > 0 && iter.ntensors() - iter.noutputs() == 1 && iter.noutputs() >= 1);

    using traits = function_traits<decltype(&ops_t::reduce)>;
    using arg_t = typename traits::template arg<0>::type;

    static constexpr bool is_inp_out_type_half =
        (std::is_same_v<dtype::Half, scalar_t> && std::is_same_v<dtype::Half, out_scalar_t>);

    static constexpr bool can_accumulate_in_output =
        std::is_convertible_v<arg_t, out_scalar_t> && (!is_inp_out_type_half);

    bool can_use_32bit_indexing = iter.can_use_32bit_indexing();
    std::unique_ptr<AccumulationBuffer> owned_buf_ptr;
    // The acc_buf_ptr is a shared pointer. It is create at the first entrance and
    // reused by all recursive function calls.
    if (acc_buf_ptr == NULL) {
        // acc_buf_ptr holds buffer used for accumulation among multiple sub_iter
        // when accumulation in output is not possible.
        if (!can_accumulate_in_output && !can_use_32bit_indexing) {
            int64_t output_memory_size = iter.element_size_in_bytes(0);
            for (int dim = 0; dim < iter.ndim(); dim++) {
                output_memory_size = std::max(output_memory_size, iter.shape()[dim] * iter.strides(0)[dim]);
            }
            output_memory_size /= iter.element_size_in_bytes(0); // iter.strides is in bytes
            owned_buf_ptr.reset(new AccumulationBuffer(sizeof(arg_t),
                                                       sizeof(out_scalar_t),
                                                       (char *)iter.data_ptr(0),
                                                       output_memory_size * sizeof(arg_t),
                                                       iter.device()));
        } else {
            owned_buf_ptr.reset(new AccumulationBuffer());
        }
        acc_buf_ptr = owned_buf_ptr.get();
    }

    if (!can_use_32bit_indexing) {
        for (auto &sub_iter : iter.with_32bit_indexing()) {
            int64_t sub_iter_base_idx = sub_iter.view_offsets(0);
            gpu_reduce_kernel<scalar_t, out_scalar_t, vt0, input_vec_size>(
                sub_iter, ops, ident, acc_buf_ptr, sub_iter_base_idx);
        }
        return;
    }

    const char *in_data = (char *)iter.data_ptr(iter.ntensors() - 1);
    char *out_data = (char *)iter.data_ptr(0);
    const auto noutputs = iter.noutputs();
    std::optional<char *> out_data_extra;
    if (noutputs > 1) {
        out_data_extra = (char *)iter.data_ptr(1);
    } else {
        out_data_extra = std::nullopt;
    }
    char *acc_data = acc_buf_ptr->get_acc_slice(out_data);

    ReduceConfig config = set_reduce_config<arg_t, scalar_t, vt0, input_vec_size>(iter);

    DataPtr buffer;
    DataPtr semaphores;
    if (config.should_global_reduce()) {
        buffer = DeviceAllocator::GetInstance()->allocate(config.global_memory_size(), iter.device());
        semaphores = DeviceAllocator::GetInstance()->allocate(config.semaphore_size(), iter.device());
    }

    CHECK_FAIL(can_use_32bit_indexing);

    auto output_calc = make_output_calculator<uint32_t>(iter);
    auto input_calc = make_input_calculator<uint32_t>(iter);
    auto reduce = ReduceOp<scalar_t, ops_t, uint32_t, out_scalar_t, vt0, input_vec_size>(
        ops,
        config,
        input_calc,
        output_calc,
        in_data,
        out_data,
        out_data_extra,
        acc_data,
        buffer.get(),
        (int *)semaphores.get(),
        ident,
        noutputs,
        base_idx);

    reduce.accumulate = iter.should_accumulate();
    reduce.final_output = iter.is_final_output();
    launch_reduce_kernel<mnt_wrapper<scalar_t>::MAX_NUM_THREADS>(config, reduce);
}
