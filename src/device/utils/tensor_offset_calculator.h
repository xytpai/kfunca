#pragma once

#include <iostream>
#include <algorithm>
#include <type_traits>
#include <array>

#include "launcher.h"
#include "tensor.h"
#include "tensor_iterator.h"
#include "array.h"
#include "exception.h"

template <typename Value>
struct DivMod {
    Value div, mod;
    HOST_DEVICE_INLINE DivMod(Value div, Value mod) :
        div(div), mod(mod) {
    }
};

template <typename Value>
struct IntDivider {
    IntDivider() {
    } // Dummy constructor for arrays.
    IntDivider(Value d) :
        divisor(d) {
    }
    HOST_DEVICE_INLINE Value div(Value n) const {
        return n / divisor;
    }
    HOST_DEVICE_INLINE Value mod(Value n) const {
        return n % divisor;
    }
    HOST_DEVICE_INLINE DivMod<Value> divmod(Value n) const {
        return DivMod<Value>(n / divisor, n % divisor);
    }
    Value divisor;
};

template <>
struct IntDivider<unsigned int> {
    static_assert(sizeof(unsigned int) == 4, "Assumes 32-bit unsigned int.");
    IntDivider() {
    } // Dummy constructor for arrays.
    IntDivider(unsigned int d) :
        divisor(d) {
        // assert(divisor >= 1 && divisor <= INT32_MAX);
        // TODO: gcc/clang has __builtin_clz() but it's not portable.
        for (shift = 0; shift < 32; shift++)
            if ((1U << shift) >= divisor) break;
        uint64_t one = 1;
        uint64_t magic = ((one << 32) * ((one << shift) - divisor)) / divisor + 1;
        m1 = magic;
        // assert(m1 > 0 && m1 == magic);  // m1 must fit in 32 bits.
    }

    DEVICE_INLINE unsigned int div(unsigned int n) const {
#if defined(__CUDACC__)
        unsigned int t = __umulhi(n, m1);
        return (t + n) >> shift;
#else
        uint64_t t = ((uint64_t)n * m1) >> 32;
        return (t + n) >> shift;
#endif
    }

    DEVICE_INLINE unsigned int mod(unsigned int n) const {
        return n - div(n) * divisor;
    }

    DEVICE_INLINE DivMod<unsigned int> divmod(unsigned int n) const {
        unsigned int q = div(n);
        return DivMod<unsigned int>(q, n - q * divisor);
    }

    unsigned int divisor; // d above.
    unsigned int m1;      // Magic number: m' above.
    unsigned int shift;   // Shift amounts.
};

template <int NARGS, typename index_t = uint32_t, bool signed_strides = false>
struct OffsetCalculator {
    using stride_t = std::conditional_t<signed_strides,
                                        std::make_signed_t<index_t>,
                                        index_t>;
    using offset_type = memory::array<stride_t, std::max<int>(NARGS, 1)>;

    // if element_sizes is nullptr, then the strides will be in bytes, otherwise
    // the strides will be in # of elements.
    OffsetCalculator(
        int dims,
        const int64_t *sizes,
        const int64_t *const *strides,
        const int64_t *element_sizes = nullptr) :
        dims(dims) {
        CHECK_FAIL(dims <= MAX_TENSOR_DIMS, "tensor has too many (>", MAX_TENSOR_DIMS, ") dims");
        for (int i = 0; i < dims; i++) {
            sizes_[i] = IntDivider<index_t>(sizes[i]);
            for (int arg = 0; arg < NARGS; arg++) {
                int64_t element_size = (element_sizes == nullptr ? 1LL : element_sizes[arg]);
                strides_[i][arg] = strides[arg][i] / element_size;
            }
        }
    }

    HOST_DEVICE offset_type get(index_t linear_idx) const {
        offset_type offsets;
#pragma unroll
        for (int arg = 0; arg < NARGS; arg++) {
            offsets[arg] = 0;
        }

#pragma unroll
        for (int dim = 0; dim < MAX_TENSOR_DIMS; ++dim) {
            if (dim == dims) {
                break;
            }
            auto divmod = sizes_[dim].divmod(linear_idx);
            linear_idx = divmod.div;

#pragma unroll
            for (int arg = 0; arg < NARGS; arg++) {
                offsets[arg] += divmod.mod * strides_[dim][arg];
            }
        }
        return offsets;
    }

    int dims;
    IntDivider<index_t> sizes_[MAX_TENSOR_DIMS];
    stride_t strides_[MAX_TENSOR_DIMS][std::max<int>(NARGS, 1)];
};

template <int NARGS, typename index_t = uint32_t>
struct TrivialOffsetCalculator {
    using offset_type = memory::array<index_t, std::max<int>(NARGS, 1)>;
    HOST_DEVICE_INLINE offset_type get(index_t linear_idx) const {
        offset_type offsets;
#pragma unroll
        for (int arg = 0; arg < NARGS; arg++) {
            offsets[arg] = linear_idx;
        }
        return offsets;
    }
};

// Make an OffsetCalculator with byte offsets
template <int N, bool signed_strides = false>
static OffsetCalculator<N, uint32_t, signed_strides> make_offset_calculator(TensorIterator &iter) {
    CHECK_FAIL(N <= iter.ntensors());
    std::array<const int64_t *, N> strides;
    for (int i = 0; i < N; i++) {
        strides[i] = iter.strides(i);
    }
    return OffsetCalculator<N, uint32_t, signed_strides>(iter.ndim(), iter.shape(), strides.data());
}

// Make an OffsetCalculator with element offsets
template <int N, bool signed_strides = false>
static OffsetCalculator<N, uint32_t, signed_strides> make_element_offset_calculator(
    TensorIterator &iter) {
    CHECK_FAIL(N <= iter.ntensors());
    std::array<const int64_t *, N> strides;
    std::array<int64_t, N> element_sizes;
    for (int i = 0; i < N; i++) {
        strides[i] = iter.strides(i);
        element_sizes[i] = iter.element_size_in_bytes(i);
    }
    return OffsetCalculator<N, uint32_t, signed_strides>(
        iter.ndim(), iter.shape(), strides.data(), element_sizes.data());
}

template <int N, bool signed_strides = false>
static OffsetCalculator<N, uint32_t, signed_strides> make_input_offset_calculator(TensorIterator &iter) {
    constexpr int array_size = std::max<int>(N, 1);
    CHECK_FAIL(N == iter.ntensors() - iter.noutputs());
    std::array<const int64_t *, array_size> strides;
    int64_t element_sizes[array_size];
    for (int i = 0; i < N; i++) {
        strides[i] = iter.strides(i + iter.noutputs());
        element_sizes[i] = iter.element_size_in_bytes(i + iter.noutputs());
    }
    return OffsetCalculator<N, uint32_t, signed_strides>(iter.ndim(), iter.shape(), strides.data(), element_sizes);
}

template <int num_outputs = 1, bool signed_strides = false>
static OffsetCalculator<num_outputs, uint32_t, signed_strides> make_output_offset_calculator(TensorIterator &iter) {
    CHECK_FAIL(num_outputs == iter.noutputs());
    std::array<const int64_t *, num_outputs> strides;
    int64_t element_sizes[num_outputs];
    for (int i = 0; i < num_outputs; i++) {
        strides[i] = iter.strides(i);
        element_sizes[i] = iter.element_size_in_bytes(i);
    }
    return OffsetCalculator<num_outputs, uint32_t, signed_strides>(iter.ndim(), iter.shape(), strides.data(), element_sizes);
}
