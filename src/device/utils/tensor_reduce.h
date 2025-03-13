#pragma once

#include "launcher.h"
#include "device_common.h"
#include "function_traits.h"
#include "tensor_iterator.h"
#include "exception.h"

template <typename out_scalar_t, typename func_t>
struct func_wrapper_t {
    using arg_t = typename binary_function_traits<func_t>::arg1_t;
    using scalar_t = typename binary_function_traits<func_t>::arg2_t;

    func_t combine;
    static DEVICE_INLINE out_scalar_t project(arg_t arg) {
        return (out_scalar_t)arg;
    }
    static DEVICE_INLINE arg_t gpu_shfl_down(arg_t arg, int offset) {
        return GPU_SHFL_DOWN(arg, offset);
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

// template <typename scalar_t, typename ops_t, typename index_t, typename out_scalar_t = scalar_t, int vt0 = 4, int input_vec_size = vt0>
// struct ReduceOp {
//     using traits = function_traits<decltype(&ops_t::reduce)>;
//     using arg_t = typename std::decay<typename traits::template arg<0>::type>::type;

//     using InputCalculator = OffsetCalculator<1, index_t>;
//     using OutputCalculator = OffsetCalculator<2, index_t>;

//     static constexpr bool can_accumulate_in_output =
//         std::is_convertible_v<arg_t, out_scalar_t> && std::is_convertible_v<out_scalar_t, arg_t>;

//     ops_t ops;
//     arg_t ident;
//     ReduceConfig config;
//     InputCalculator input_calc;
//     OutputCalculator output_calc;
//     const void *src;
//     const char *dst[2]; // it accepts at most two destinations
//     // acc_buf used for accumulation among sub Tensor Iterator when accumulation on
//     // output is not permissible
//     void *acc_buf;
//     // cta_buf used for accumulation between blocks during global reduction
//     void *cta_buf;
//     int *semaphores;
//     int64_t base_idx;
//     bool accumulate;
//     bool final_output;
//     int noutputs;

//     ReduceOp(
//         ops_t ops,
//         ReduceConfig config,
//         InputCalculator input_calc,
//         OutputCalculator output_calc,
//         const void *src,
//         char *dst0,
//         std::optional<char *> dst1,
//         void *acc_buf,
//         void *cta_buf,
//         int *semaphores,
//         arg_t ident,
//         int noutputs,
//         int64_t base_idx) :
//         ops(ops),
//         ident(ident),
//         config(config),
//         input_calc(input_calc),
//         output_calc(output_calc),
//         src(src),
//         acc_buf(acc_buf),
//         cta_buf(cta_buf),
//         semaphores(semaphores),
//         base_idx(base_idx),
//         noutputs(noutputs) {
//         dst[0] = dst0;
//         if (dst1.has_value()) {
//             dst[1] = dst1.value();
//         }
//     }

//     template <int output_vec_size>
//     C10_DEVICE void run() const {
//         extern __shared__ char shared_memory[];
//         index_t output_idx = config.output_idx<output_vec_size>();
//         index_t input_idx = config.input_idx();
//         auto base_offsets1 = output_calc.get(output_idx)[1];

//         using arg_vec_t = std::array<arg_t, output_vec_size>;
//         arg_vec_t value;

//         if (output_idx < config.num_outputs && input_idx < config.num_inputs) {
//             const scalar_t *input_slice = (const scalar_t *)((const char *)src + base_offsets1);
//             value = thread_reduce<output_vec_size>(input_slice);
//         }

//         if (config.should_block_y_reduce()) {
//             value = block_y_reduce<output_vec_size>(value, shared_memory);
//         }
//         if (config.should_block_x_reduce()) {
//             value = block_x_reduce<output_vec_size>(value, shared_memory);
//         }

//         using out_ptr_vec_t = std::array<out_scalar_t *, output_vec_size>;
//         using offset_vec_t = std::array<index_t, output_vec_size>;
//         offset_vec_t base_offsets;
//         out_ptr_vec_t out;

// #pragma unroll
//         for (int i = 0; i < output_vec_size; i++) {
//             base_offsets[i] = output_calc.get(output_idx + i)[0];
//             out[i] = (out_scalar_t *)((char *)dst[0] + base_offsets[i]);
//         }

//         arg_vec_t *acc = nullptr;
//         if (acc_buf != nullptr) {
//             size_t numerator = sizeof(arg_t);
//             size_t denominator = sizeof(out_scalar_t);
//             reduce_fraction(numerator, denominator);
//             acc = (arg_vec_t *)((char *)acc_buf + (base_offsets[0] * numerator / denominator));
//         }

//         if (config.should_global_reduce()) {
//             value = global_reduce<output_vec_size>(value, acc, shared_memory);
//         } else if (config.should_store(output_idx)) {
//             if (accumulate) {
// #pragma unroll
//                 for (int i = 0; i < output_vec_size; i++) {
//                     value[i] = ops.translate_idx(value[i], base_idx);
//                 }
//             }

//             if (acc == nullptr) {
//                 if (accumulate) {
//                     value = accumulate_in_output<output_vec_size, can_accumulate_in_output>(out, value);
//                 }
//                 if (final_output) {
//                     set_results_to_output<output_vec_size>(value, base_offsets);
//                 } else {
// #pragma unroll
//                     for (int i = 0; i < output_vec_size; i++) {
//                         *(out[i]) = get_accumulated_output<can_accumulate_in_output>(out[i], value[i]);
//                     }
//                 }
//             } else {
//                 if (accumulate) {
// #pragma unroll
//                     for (int i = 0; i < output_vec_size; i++) {
//                         value[i] = ops.combine((*acc)[i], value[i]);
//                     }
//                 }
//                 if (final_output) {
//                     set_results_to_output<output_vec_size>(value, base_offsets);
//                 } else {
//                     *acc = value;
//                 }
//             }
//         }
//     }

//     template <int output_vec_size>
//     C10_DEVICE std::array<arg_t, output_vec_size> thread_reduce(const scalar_t *data) const {
//         if (config.vectorize_input) {
//             CUDA_KERNEL_ASSERT(output_vec_size == 1);
//             // reduce at the header of input_slice where memory is not aligned,
//             // so that thread_reduce will have an aligned memory to work on.
//             return {input_vectorized_thread_reduce_impl(data)};
//         } else {
//             index_t element_stride = input_calc.strides_[0][0] / sizeof(scalar_t);
//             bool is_contiguous = (input_calc.dims == 1 && element_stride == 1);
//             if (is_contiguous) {
//                 return thread_reduce_impl<output_vec_size>(data, [](index_t idx) { return idx; });
//             } else if (input_calc.dims == 1) {
//                 return thread_reduce_impl<output_vec_size>(data, [&](index_t idx) { return idx * element_stride; });
//             } else {
//                 return thread_reduce_impl<output_vec_size>(data, [&](index_t idx) { return input_calc.get(idx)[0] / sizeof(scalar_t); });
//             }
//         }
//     }

//     C10_DEVICE arg_t input_vectorized_thread_reduce_impl(const scalar_t *data) const {
//         index_t end = config.num_inputs;

//         // Handle the head of input slice where data is not aligned
//         arg_t value = ident;
//         constexpr int align_bytes = alignof(at::native::memory::aligned_vector<scalar_t, input_vec_size>);
//         constexpr int align_elements = align_bytes / sizeof(scalar_t);
//         int shift = ((uint64_t)data) % align_bytes / sizeof(scalar_t);
//         if (shift > 0) {
//             data -= shift;
//             end += shift;
//             if (threadIdx.x >= shift && threadIdx.x < align_elements && config.should_reduce_tail()) {
//                 value = ops.reduce(value, c10::load(data + threadIdx.x), threadIdx.x - shift);
//             }
//             end -= align_elements;
//             data += align_elements;
//             shift = align_elements - shift;
//         }

//         // Do the vectorized reduction
//         using load_t = at::native::memory::aligned_vector<scalar_t, input_vec_size>;

//         index_t idx = config.input_idx();
//         const index_t stride = config.step_input;

//         // Multiple accumulators to remove dependency between unrolled loops.
//         arg_t value_list[input_vec_size];
//         value_list[0] = value;

// #pragma unroll
//         for (int i = 1; i < input_vec_size; i++) {
//             value_list[i] = ident;
//         }

//         while (idx * input_vec_size + input_vec_size - 1 < end) {
//             const auto values_vec = memory::load_vector<input_vec_size>(data, idx);
// #pragma unroll
//             for (index_t i = 0; i < input_vec_size; i++) {
//                 value_list[i] = ops.reduce(value_list[i], values_vec.val[i], shift + idx * input_vec_size + i);
//             }
//             idx += stride;
//         }

//         // tail
//         index_t tail_start = end - end % input_vec_size;
//         if (config.should_reduce_tail()) {
//             int idx = tail_start + threadIdx.x;
//             if (idx < end) {
//                 const auto value = c10::load(data + idx);
//                 value_list[0] = ops.reduce(value_list[0], value, idx + shift);
//             }
//         }

// // combine accumulators
// #pragma unroll
//         for (int i = 1; i < input_vec_size; i++) {
//             value_list[0] = ops.combine(value_list[0], value_list[i]);
//         }
//         return value_list[0];
//     }

//     template <int output_vec_size, typename offset_calc_t>
//     C10_DEVICE std::array<arg_t, output_vec_size> thread_reduce_impl(const scalar_t *data_, offset_calc_t calc) const {
//         index_t idx = config.input_idx();
//         const index_t end = config.num_inputs;
//         const index_t stride = config.step_input;

//         using arg_vec_t = std::array<arg_t, output_vec_size>;
//         using load_t = at::native::memory::aligned_vector<scalar_t, output_vec_size>;

//         // Multiple accumulators to remove dependency between unrolled loops.
//         arg_vec_t value_list[vt0];

// #pragma unroll
//         for (int i = 0; i < vt0; i++) {
// #pragma unroll
//             for (int j = 0; j < output_vec_size; j++) {
//                 value_list[i][j] = ident;
//             }
//         }

//         load_t values[vt0];

//         while (idx + (vt0 - 1) * stride < end) {
// #pragma unroll
//             for (index_t i = 0; i < vt0; i++) {
//                 const auto offset = calc(idx + i * stride) / output_vec_size;
//                 values[i] = memory::load_vector<output_vec_size>(data_, offset);
//             }
// #pragma unroll
//             for (index_t i = 0; i < vt0; i++) {
// #pragma unroll
//                 for (index_t j = 0; j < output_vec_size; j++) {
//                     value_list[i][j] = ops.reduce(value_list[i][j], values[i].val[j], idx + i * stride);
//                 }
//             }
//             idx += stride * vt0;
//         }

//         // tail
//         int idx_ = idx;
// #pragma unroll
//         for (index_t i = 0; i < vt0; i++) {
//             if (idx >= end) {
//                 break;
//             }
//             const auto offset = calc(idx) / output_vec_size;
//             values[i] = memory::load_vector<output_vec_size>(data_, offset);
//             idx += stride;
//         }
//         idx = idx_;
// #pragma unroll
//         for (index_t i = 0; i < vt0; i++) {
//             if (idx >= end) {
//                 break;
//             }
// #pragma unroll
//             for (index_t j = 0; j < output_vec_size; j++) {
//                 value_list[i][j] = ops.reduce(value_list[i][j], values[i].val[j], idx);
//             }
//             idx += stride;
//         }

// // combine accumulators
// #pragma unroll
//         for (int i = 1; i < vt0; i++) {
// #pragma unroll
//             for (index_t j = 0; j < output_vec_size; j++) {
//                 value_list[0][j] = ops.combine(value_list[0][j], value_list[i][j]);
//             }
//         }
//         return value_list[0];
//     }

//     template <int output_vec_size>
//     C10_DEVICE std::array<arg_t, output_vec_size> block_x_reduce(std::array<arg_t, output_vec_size> value, char *shared_memory) const {
//         using args_vec_t = std::array<arg_t, output_vec_size>;
//         int dim_x = blockDim.x;
//         args_vec_t *shared = (args_vec_t *)shared_memory;
//         if (dim_x > warpSize) {
//             int address_base = threadIdx.x + threadIdx.y * blockDim.x;
//             shared[address_base] = value;
//             for (int offset = dim_x / 2; offset >= warpSize; offset >>= 1) {
//                 __syncthreads();
//                 if (threadIdx.x < offset && threadIdx.x + offset < blockDim.x) {
//                     args_vec_t other = shared[address_base + offset];
// #pragma unroll
//                     for (int i = 0; i < output_vec_size; i++) {
//                         value[i] = ops.combine(value[i], other[i]);
//                     }
//                     shared[address_base] = value;
//                 }
//             }
//             dim_x = warpSize;
//         }

//         __syncthreads();

//         for (int offset = 1; offset < dim_x; offset <<= 1) {
// #pragma unroll
//             for (int i = 0; i < output_vec_size; i++) {
//                 arg_t other = ops.warp_shfl_down(value[i], offset);
//                 value[i] = ops.combine(value[i], other);
//             }
//         }
//         return value;
//     }

//     template <int output_vec_size>
//     C10_DEVICE std::array<arg_t, output_vec_size> block_y_reduce(std::array<arg_t, output_vec_size> value, char *shared_memory) const {
//         using args_vec_t = std::array<arg_t, output_vec_size>;
//         args_vec_t *shared = (args_vec_t *)shared_memory;
//         shared[config.shared_memory_offset(0)] = value;
//         for (int offset = blockDim.y / 2; offset > 0; offset >>= 1) {
//             __syncthreads();
//             if (threadIdx.y < offset && threadIdx.y + offset < blockDim.y) {
//                 args_vec_t other = shared[config.shared_memory_offset(offset)];
// #pragma unroll
//                 for (int i = 0; i < output_vec_size; i++) {
//                     value[i] = ops.combine(value[i], other[i]);
//                 }
//                 shared[config.shared_memory_offset(0)] = value;
//             }
//         }
//         return value;
//     }

//     C10_DEVICE bool mark_block_finished() const {
//         __shared__ bool is_last_block_done_shared;

//         __syncthreads();
//         if (threadIdx.x == 0 && threadIdx.y == 0) {
//             int prev_blocks_finished = atomicAdd(&semaphores[blockIdx.x], 1);
//             is_last_block_done_shared = (prev_blocks_finished == gridDim.y - 1);
//         }

//         __syncthreads();

//         return is_last_block_done_shared;
//     }

//     template <int output_vec_size, bool can_acc>
//     C10_DEVICE std::array<arg_t, output_vec_size> accumulate_in_output(
//         std::array<out_scalar_t *, output_vec_size> out,
//         std::array<arg_t, output_vec_size> value,
//         typename std::enable_if_t<can_acc> * = nullptr) const {
//         std::array<arg_t, output_vec_size> ret;
// #pragma unroll
//         for (int i = 0; i < output_vec_size; i++) {
//             ret[i] = ops.combine(*(out[i]), value[i]);
//         }
//         return ret;
//     }

//     template <bool can_acc>
//     C10_DEVICE out_scalar_t get_accumulated_output(
//         out_scalar_t *out, arg_t value,
//         typename std::enable_if_t<can_acc> * = nullptr) const {
//         CUDA_KERNEL_ASSERT(!final_output);
//         return (out_scalar_t)value;
//     }

//     // This function should never be called --
//     // it's the version of `accumulate_in_output`
//     // when accumulation in the output is not possible.
//     template <int output_vec_size, bool can_acc>
//     C10_DEVICE std::array<arg_t, output_vec_size> accumulate_in_output(
//         std::array<out_scalar_t *, output_vec_size>,
//         std::array<arg_t, output_vec_size>,
//         typename std::enable_if_t<!can_acc> * = nullptr) const {
//         CUDA_KERNEL_ASSERT(false);
//         return {arg_t{}};
//     }

//     // This function should never be called --
//     // it's the version of `get_accumulated_output`
//     // when accumulation in the output is not possible.
//     template <bool can_acc>
//     C10_DEVICE out_scalar_t get_accumulated_output(
//         out_scalar_t *out, arg_t value,
//         typename std::enable_if_t<!can_acc> * = nullptr) const {
//         CUDA_KERNEL_ASSERT(false);
//         return *out;
//     }

//     template <class T>
//     C10_DEVICE void set_results(const T x, const index_t base_offset) const {
//         CUDA_KERNEL_ASSERT(noutputs == 1);
//         auto res = (out_scalar_t *)((char *)dst[0] + base_offset);
//         *res = x;
//     }

//     // Currently implemented for max of two outputs
//     template <class T1, class T2>
//     C10_DEVICE void set_results(const thrust::pair<T1, T2> x, const index_t base_offset) const {
//         if (noutputs >= 1) {
//             auto res0 = (T1 *)((char *)dst[0] + base_offset);
//             *res0 = x.first;
//         }
//         if (noutputs >= 2) {
//             // base offset is computed assuming element size being sizeof(T1), so we need to make a
//             // correction to obtain the correct base offset
//             auto res1 = (T2 *)((char *)dst[1] + base_offset / sizeof(T1) * sizeof(T2));
//             *res1 = x.second;
//         }
//     }

//     template <int output_vec_size>
//     C10_DEVICE void set_results_to_output(std::array<arg_t, output_vec_size> value, std::array<index_t, output_vec_size> base_offset) const {
//         CUDA_KERNEL_ASSERT(final_output);
// #pragma unroll
//         for (int i = 0; i < output_vec_size; i++) {
//             set_results(ops.project(value[i]), base_offset[i]);
//         }
//     }

//     template <int output_vec_size>
//     C10_DEVICE std::array<arg_t, output_vec_size> global_reduce(std::array<arg_t, output_vec_size> value, std::array<arg_t, output_vec_size> *acc, char *shared_memory) const {
//         using arg_vec_t = std::array<arg_t, output_vec_size>;
//         using out_ptr_vec_t = std::array<out_scalar_t *, output_vec_size>;
//         using offset_vec_t = std::array<index_t, output_vec_size>;

//         arg_vec_t *reduce_buffer = (arg_vec_t *)cta_buf;
//         index_t output_idx = config.output_idx<output_vec_size>();
//         offset_vec_t base_offsets;
//         out_ptr_vec_t out;

// #pragma unroll
//         for (int i = 0; i < output_vec_size; i++) {
//             base_offsets[i] = output_calc.get(output_idx + i)[0];
//             out[i] = (out_scalar_t *)((char *)dst[0] + base_offsets[i]);
//         }

//         bool should_store = config.should_store(output_idx);
//         if (should_store) {
//             index_t offset = config.staging_memory_offset(blockIdx.y);
//             reduce_buffer[offset] = value;
//         }

//         __threadfence(); // make sure writes are globally visible
//         __syncthreads(); // if multiple warps in this block wrote to staging, make sure they're all done
//         bool is_last_block_done = mark_block_finished();

//         if (is_last_block_done) {
//             __threadfence(); // complete the acquire pattern after atomic
//             for (auto &v : value) {
//                 v = ident;
//             }
//             if (config.should_block_x_reduce()) {
//                 index_t input_offset = threadIdx.x + threadIdx.y * blockDim.x;
//                 index_t step = blockDim.x * blockDim.y;
//                 for (; input_offset < config.ctas_per_output; input_offset += step) {
//                     index_t idx = config.staging_memory_offset(input_offset);
//                     arg_vec_t next = reduce_buffer[idx];
// #pragma unroll
//                     for (int i = 0; i < output_vec_size; i++) {
//                         value[i] = ops.combine(value[i], next[i]);
//                     }
//                 }
//             } else {
//                 index_t input_offset = threadIdx.y;
//                 index_t step = blockDim.y;
//                 for (; input_offset < config.ctas_per_output; input_offset += step) {
//                     index_t idx = config.staging_memory_offset(input_offset);
//                     arg_vec_t next = reduce_buffer[idx];
// #pragma unroll
//                     for (int i = 0; i < output_vec_size; i++) {
//                         value[i] = ops.combine(value[i], next[i]);
//                     }
//                 }
//             }
//             value = block_y_reduce<output_vec_size>(value, shared_memory);
//             if (config.should_block_x_reduce()) {
//                 value = block_x_reduce<output_vec_size>(value, shared_memory);
//             }
//             if (should_store) {
//                 if (accumulate) {
// #pragma unroll
//                     for (int i = 0; i < output_vec_size; i++) {
//                         value[i] = ops.translate_idx(value[i], base_idx);
//                     }
//                 }

//                 if (acc == nullptr) {
//                     if (accumulate) {
//                         value = accumulate_in_output<output_vec_size, can_accumulate_in_output>(out, value);
//                     }
//                     if (final_output) {
//                         set_results_to_output<output_vec_size>(value, base_offsets);
//                     } else {
// #pragma unroll
//                         for (int i = 0; i < output_vec_size; i++) {
//                             *(out[i]) = get_accumulated_output<can_accumulate_in_output>(out[i], value[i]);
//                         }
//                     }
//                 } else {
//                     if (accumulate) {
// #pragma unroll
//                         for (int i = 0; i < output_vec_size; i++) {
//                             value[i] = ops.combine((*acc)[i], value[i]);
//                         }
//                     }
//                     if (final_output) {
//                         set_results_to_output<output_vec_size>(value, base_offsets);
//                     } else {
//                         *acc = value;
//                     }
//                 }
//             }
//         }

//         return value;
//     }
// };

template <typename scalar_t, typename out_scalar_t, int vt0 = 4, int input_vec_size = vt0, typename ops_t, typename ident_t = double>
inline void gpu_reduce_kernel(TensorIterator &iter, const ops_t &ops, ident_t ident = 0, int64_t base_idx = 0) {
    CHECK_FAIL(iter.numel() > 0 && iter.ntensors() - iter.noutputs() == 1 && iter.noutputs() >= 1);

    using traits = function_traits<decltype(&ops_t::reduce)>;
    using arg_t = typename traits::template arg<0>::type;

    bool can_use_32bit_indexing = iter.can_use_32bit_indexing();

    if (!can_use_32bit_indexing) {
        for (auto &sub_iter : iter.with_32bit_indexing()) {
            int64_t sub_iter_base_idx = sub_iter.view_offsets(0);
            gpu_reduce_kernel<scalar_t, out_scalar_t, vt0, input_vec_size>(
                sub_iter, ops, ident, sub_iter_base_idx);
        }
        return;
    }
}
