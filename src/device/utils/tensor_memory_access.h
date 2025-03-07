#pragma once

#include <tuple>
#include <type_traits>
#include <algorithm>

#include "launcher.h"
#include "scalar_type.h"
#include "memory.h"

using namespace utils;

#define FETCH_AND_CAST_CASE(type, scalartype, ...) \
    case ScalarType::scalartype:                   \
        return static_cast<type>(*(const type *)ptr);
template <typename dest_t>
DEVICE_INLINE dest_t fetch_and_cast(const ScalarType src_type, const void *ptr) {
    switch (src_type) {
        FORALL_BASIC_SCALAR_TYPES(FETCH_AND_CAST_CASE)
    default:;
    }
    return dest_t(0);
}
#undef FETCH_AND_CAST_CASE

#define CAST_AND_STORE_CASE(type, scalartype, ...) \
    case ScalarType::scalartype:                   \
        *(type *)ptr = static_cast<type>(value);   \
        return;
template <typename src_t>
DEVICE_INLINE void cast_and_store(const ScalarType dest_type, void *ptr, src_t value) {
    switch (dest_type) {
        FORALL_BASIC_SCALAR_TYPES(CAST_AND_STORE_CASE)
    default:;
    }
}
#undef CAST_AND_STORE_CASE

namespace memory_access {

template <template <int i> typename func, int end, int current = 0>
struct static_unroll {
    template <typename... Args>
    static HOST_DEVICE_INLINE void with_args(Args &&...args) {
        func<current>::apply(std::forward<Args>(args)...);
        static_unroll<func, end, current + 1>::with_args(args...);
    }
};

template <template <int i> typename func, int end>
struct static_unroll<func, end, end> {
    template <typename... Args>
    static HOST_DEVICE_INLINE void with_args(Args... args) {
    }
};

struct LoadWithoutCast {
    template <typename scalar_t>
    DEVICE scalar_t load(char *base_ptr, uint32_t offset, int arg) {
        return *(reinterpret_cast<scalar_t *>(base_ptr) + offset);
    }
};

struct StoreWithoutCast {
    template <typename scalar_t>
    DEVICE void store(scalar_t value, char *base_ptr, uint32_t offset) {
        *(reinterpret_cast<scalar_t *>(base_ptr) + offset) = value;
    }
};

template <int N>
struct LoadWithCast {
    using array_t = memory::array<ScalarType, std::max<int>(N, 1)>;
    using size_array_t = memory::array<uint32_t, std::max<int>(N, 1)>;

    array_t dtypes;
    size_array_t element_sizes;

    template <typename array_t_>
    LoadWithCast(array_t_ dtypes) {
#pragma unroll
        for (int i = 0; i < N; i++) {
            this->dtypes[i] = dtypes[i];
            element_sizes[i] = element_size(dtypes[i]);
        }
    }

    template <typename scalar_t>
    DEVICE scalar_t load(char *base_ptr, uint32_t offset, int arg) {
        void *ptr = base_ptr + element_sizes[arg] * offset;
        return fetch_and_cast<scalar_t>(dtypes[arg], ptr);
    }
};

struct StoreWithCast {
    ScalarType dtype;
    uint32_t element_size_;
    StoreWithCast(ScalarType dtype) :
        dtype(dtype), element_size_(element_size(dtype)) {
    }
    template <typename scalar_t>
    DEVICE void store(scalar_t value, char *base_ptr, uint32_t offset) {
        void *ptr = base_ptr + element_size_ * offset;
        cast_and_store<scalar_t>(dtype, ptr, value);
    }
};

namespace policies {

template <int arg_index>
struct unroll_load_helper {
    template <typename args_t, typename policy_t, typename offset_t, typename loader_t>
    static DEVICE void apply(policy_t &self, args_t *args, offset_t offset, loader_t loader, int j, int num_outputs) {
        using arg_t = std::tuple_element_t<arg_index, args_t>;
        std::get<arg_index>(args[j]) = loader.template load<arg_t>(self.data[arg_index + num_outputs], offset[arg_index], arg_index);
    }
};

template <int THREAD_WORK_SIZE, typename data_t, typename inp_calc_t, typename out_calc_t, typename loader_t, typename storer_t, int num_outputs = 1>
struct unroll {
    data_t data;
    int remaining;
    inp_calc_t input_offset_calculator;
    out_calc_t output_offset_calculator;
    loader_t loader;
    storer_t storer;
    int tid;
    int bid;
    int block_size;
    int block_work_size;

    DEVICE unroll(data_t data, int remaining, inp_calc_t ic, out_calc_t oc, loader_t l, storer_t s, int tid, int bid, int block_size) :
        data(data), remaining(remaining), input_offset_calculator(ic), output_offset_calculator(oc), loader(l), storer(s),
        tid(tid), bid(bid), block_size(block_size), block_work_size(THREAD_WORK_SIZE * block_size) {
    }

    DEVICE_INLINE bool check_inbounds(int thread_work_elem) const {
        return ((tid + thread_work_elem * block_size) < remaining);
    }

    template <typename args_t>
    DEVICE_INLINE void load(args_t *args) {
        constexpr int arity = std::tuple_size<args_t>::value;
        int thread_idx = tid;
#pragma unroll
        for (int i = 0; i < THREAD_WORK_SIZE; i++) {
            if (thread_idx >= remaining) {
                return;
            }
            int linear_idx = thread_idx + block_work_size * bid;
            auto offset = input_offset_calculator.get(linear_idx);
            static_unroll<unroll_load_helper, arity>::with_args(*this, args, offset, loader, i, num_outputs);
            thread_idx += block_size;
        }
    }

    template <typename scalar_t>
    DEVICE_INLINE void store(scalar_t *from) {
        int thread_idx = tid;
#pragma unroll
        for (int i = 0; i < THREAD_WORK_SIZE; i++) {
            if (thread_idx >= remaining) {
                return;
            }
            int linear_idx = thread_idx + block_work_size * bid;
            int offset = output_offset_calculator.get(linear_idx)[0];
            storer.store(from[i], data[0], offset);
            thread_idx += block_size;
        }
    }
};

template <int arg_index, typename args_t, typename arg_t>
struct ArgsAccessorFunctor {
    DEVICE arg_t &operator()(int thread_unroll_idx) const {
        return std::get<arg_index>(args_[args_offset_ + thread_unroll_idx]);
    }
    DEVICE ArgsAccessorFunctor(args_t *args, int args_offset) :
        args_(args), args_offset_(args_offset) {
    }

private:
    args_t *args_;
    int args_offset_;
};

template <int arg_index>
struct vectorized_load_helper {
    template <typename args_t, typename policy_t, typename offset_t>
    static DEVICE void apply(policy_t &self, args_t *args, offset_t offset, int args_offset) {
        using arg_t = std::tuple_element_t<arg_index, args_t>;
        auto ptr = reinterpret_cast<arg_t *>(self.data[arg_index + 1]) + offset[arg_index];
        auto args_accessor = ArgsAccessorFunctor<arg_index, args_t, arg_t>(args, args_offset);
        self.load_single_arg(args_accessor, ptr);
    }
};

template <int THREAD_WORK_SIZE, int vec_size, typename data_t, typename inp_calc_t>
struct vectorized {
    static_assert(THREAD_WORK_SIZE % vec_size == 0, "The workload per thread must be a multiple of vec_size");
    static constexpr int loop_size = THREAD_WORK_SIZE / vec_size;
    data_t data;
    inp_calc_t input_offset_calculator;
    int tid;
    int bid;
    int block_size;
    int block_work_size;

    DEVICE vectorized(data_t data, inp_calc_t ic, int tid, int bid, int block_size) :
        data(data), input_offset_calculator(ic), tid(tid), bid(bid),
        block_size(block_size), block_work_size(THREAD_WORK_SIZE * block_size) {
    }

    DEVICE_INLINE constexpr bool check_inbounds(int thread_work_elem) const {
        return true;
    }

    template <typename accessor_t, typename scalar_t>
    DEVICE_INLINE void load_single_arg(accessor_t to, scalar_t *from) {
        using vec_t = memory::aligned_array<scalar_t, vec_size>;
        vec_t *from_ = reinterpret_cast<vec_t *>(from);
#pragma unroll
        for (int j = 0; j < vec_size; j++) {
            to(j) = from_->val[j];
        }
    }

    template <typename args_t>
    DEVICE_INLINE void load(args_t *args) {
        constexpr int arity = std::tuple_size<args_t>::value;
        int block_offset = block_work_size * bid;
#pragma unroll
        for (int i = 0; i < loop_size; i++) {
            auto index = block_offset + (tid + i * block_size) * vec_size;
            auto offset = input_offset_calculator.get(index);
            static_unroll<vectorized_load_helper, arity>::with_args(*this, args, offset, vec_size * i);
        }
    }

    template <typename scalar_t>
    DEVICE_INLINE void store(scalar_t *from) {
        using vec_t = memory::aligned_array<scalar_t, vec_size>;
        scalar_t *to = reinterpret_cast<scalar_t *>(data[0]) + block_work_size * bid;
        vec_t *to_ = reinterpret_cast<vec_t *>(to);
#pragma unroll
        for (int i = 0; i < loop_size; i++) {
            int index = tid + i * block_size;
            vec_t v;
#pragma unroll
            for (int j = 0; j < vec_size; j++) {
                v.val[j] = from[vec_size * i + j];
            }
            to_[index] = v;
        }
    }
};

template <int current>
struct multi_outputs_store_helper {
    template <int ntensors, int num_outputs, typename... Args>
    HOST_DEVICE static void apply(
        memory::array<char *, ntensors> data,
        memory::array<uint32_t, num_outputs> offsets,
        std::tuple<Args...> ret) {
        using T = typename std::tuple_element<current, std::tuple<Args...>>::type;
        T *to = reinterpret_cast<T *>(data[current]) + offsets[current];
        *to = std::get<current>(ret);
    }
};

template <int THREAD_WORK_SIZE, typename data_t, typename inp_calc_t, typename out_calc_t, int num_outputs>
struct multi_outputs_unroll {
    data_t data;
    int remaining;
    inp_calc_t input_offset_calculator;
    out_calc_t output_offset_calculator;
    LoadWithoutCast loader;
    StoreWithoutCast storer;
    int tid;
    int bid;
    int block_size;
    int block_work_size;

    DEVICE multi_outputs_unroll(data_t data, int remaining, inp_calc_t ic, out_calc_t oc, int tid, int bid, int block_size) :
        data(data), remaining(remaining), input_offset_calculator(ic), output_offset_calculator(oc),
        tid(tid), bid(bid), block_size(block_size), block_work_size(THREAD_WORK_SIZE * block_size) {
    }

    DEVICE_INLINE bool check_inbounds(int thread_work_elem) const {
        return ((tid + thread_work_elem * block_size) < remaining);
    }

    template <typename args_t>
    DEVICE_INLINE void load(args_t *args) {
        constexpr int arity = std::tuple_size<args_t>::value;
        int thread_idx = tid;
#pragma unroll
        for (int i = 0; i < THREAD_WORK_SIZE; i++) {
            if (thread_idx >= remaining) {
                return;
            }
            int linear_idx = thread_idx + block_work_size * bid;
            auto offset = input_offset_calculator.get(linear_idx);
            static_unroll<unroll_load_helper, arity>::with_args(*this, args, offset, loader, i, num_outputs);
            thread_idx += block_size;
        }
    }

    template <typename return_t>
    DEVICE_INLINE void store(return_t *from) {
        int thread_idx = tid;
#pragma unroll
        for (int i = 0; i < THREAD_WORK_SIZE; i++) {
            if (thread_idx >= this->remaining) {
                return;
            }
            int linear_idx = thread_idx + block_work_size * bid;
            auto offsets = this->output_offset_calculator.get(linear_idx);
            static_unroll<multi_outputs_store_helper, num_outputs>::with_args(this->data, offsets, from[i]);
            thread_idx += block_size;
        }
    }
};

} // namespace policies

template <typename scalar_t>
HOST_DEVICE_INLINE int can_vectorize_up_to(char *pointer) {
    uint64_t address = reinterpret_cast<uint64_t>(pointer);
    constexpr int vec2_alignment = std::alignment_of<memory::aligned_array<scalar_t, 2>>::value;
    constexpr int vec4_alignment = std::alignment_of<memory::aligned_array<scalar_t, 4>>::value;
    constexpr int vec8_alignment = std::alignment_of<memory::aligned_array<scalar_t, 8>>::value;
    if (address % vec8_alignment == 0) {
        return 8;
    } else if (address % vec4_alignment == 0) {
        return 4;
    } else if (address % vec2_alignment == 0) {
        return 2;
    }
    return 1;
}

template <int i>
struct can_vectorize_up_to_helper {
    template <typename array_t, typename traits>
    static HOST_DEVICE void apply(int &result, array_t pointers, traits _) {
        using arg_t = typename traits::template arg<i>::type;
        result = std::min<int>(result, can_vectorize_up_to<arg_t>(pointers[i + 1]));
    }
};

template <typename func_t, typename array_t>
inline int can_vectorize_up_to(array_t pointers) {
    using traits = function_traits<func_t>;
    using return_t = typename traits::result_type;
    constexpr int arity = traits::arity;
    int result = can_vectorize_up_to<return_t>(pointers[0]);
    static_unroll<can_vectorize_up_to_helper, arity>::with_args(result, pointers, traits());
    return result;
}

} // namespace memory_access
