#pragma once

#include <iostream>

#include "scalar_type.h"
#include "launcher.h"

struct NullType {
    using value_type = NullType;
    template <typename T>
    inline NullType &operator=(const T &) {
        return *this;
    }
    inline bool operator==(const NullType &) {
        return true;
    }
    inline bool operator!=(const NullType &) {
        return false;
    }
};

template <typename T>
struct KeyTraits {};

template <>
struct KeyTraits<NullType> {
    using Type = uint32_t;
    HOST_DEVICE static inline Type convert(float v) {
        return 0;
    }
    HOST_DEVICE static inline NullType deconvert(Type v) {
        return NullType();
    }
    HOST_DEVICE static inline unsigned int endbit() {
        return 0;
    }
};

template <>
struct KeyTraits<float> {
    using Type = uint32_t;
    HOST_DEVICE static inline Type convert(float v) {
        Type x = *((Type *)&v);
        Type mask = (x & 0x80000000) ? 0xffffffff : 0x80000000;
        return (x ^ mask);
    }
    HOST_DEVICE static inline float deconvert(Type v) {
        Type mask = (v & 0x80000000) ? 0x80000000 : 0xffffffff;
        auto v_de = v ^ mask;
        return *((float *)&v_de);
    }
    HOST_DEVICE static inline int endbit() {
        return sizeof(Type) << 3;
    }
};

template <>
struct KeyTraits<bool> {
    using Type = bool;
    HOST_DEVICE static inline Type convert(bool v) {
        return v;
    }
    HOST_DEVICE static inline bool deconvert(Type v) {
        return v;
    }
    HOST_DEVICE static inline int endbit() {
        return 1;
    }
};

template <>
struct KeyTraits<uint8_t> {
    using Type = uint8_t;
    HOST_DEVICE static inline Type convert(uint8_t v) {
        return v;
    }
    HOST_DEVICE static inline uint8_t deconvert(Type v) {
        return v;
    }
    HOST_DEVICE static inline int endbit() {
        return sizeof(Type) << 3;
    }
};

template <>
struct KeyTraits<uint16_t> {
    using Type = uint16_t;
    using SrcType = uint16_t;
    HOST_DEVICE static inline Type convert(SrcType v) {
        return v;
    }
    HOST_DEVICE static inline SrcType deconvert(Type v) {
        return v;
    }
    HOST_DEVICE static inline int endbit() {
        return sizeof(Type) << 3;
    }
};

template <>
struct KeyTraits<uint32_t> {
    using Type = uint32_t;
    using SrcType = uint32_t;
    HOST_DEVICE static inline Type convert(SrcType v) {
        return v;
    }
    HOST_DEVICE static inline SrcType deconvert(Type v) {
        return v;
    }
    HOST_DEVICE static inline int endbit() {
        return sizeof(Type) << 3;
    }
};

template <>
struct KeyTraits<uint64_t> {
    using Type = uint64_t;
    using SrcType = uint64_t;
    HOST_DEVICE static inline Type convert(SrcType v) {
        return v;
    }
    HOST_DEVICE static inline SrcType deconvert(Type v) {
        return v;
    }
    HOST_DEVICE static inline int endbit() {
        return sizeof(Type) << 3;
    }
};

template <>
struct KeyTraits<int8_t> {
    using Type = uint8_t;
    HOST_DEVICE static inline Type convert(int8_t v) {
        return 128u + v;
    }
    HOST_DEVICE static inline int8_t deconvert(Type v) {
        return v - 128;
    }
    HOST_DEVICE static inline int endbit() {
        return sizeof(Type) << 3;
    }
};

template <>
struct KeyTraits<int16_t> {
    using Type = uint16_t;
    HOST_DEVICE static inline Type convert(int16_t v) {
        return 32768u + v;
    }
    HOST_DEVICE static inline int16_t deconvert(Type v) {
        return v - 32768;
    }
    HOST_DEVICE static inline int endbit() {
        return sizeof(Type) << 3;
    }
};

template <>
struct KeyTraits<int32_t> {
    using Type = uint32_t;
    HOST_DEVICE static inline Type convert(int32_t v) {
        return 2147483648u + v;
    }
    HOST_DEVICE static inline int32_t deconvert(Type v) {
        return v - 2147483648u;
    }
    HOST_DEVICE static inline int endbit() {
        return sizeof(Type) << 3;
    }
};

template <>
struct KeyTraits<int64_t> {
    using Type = uint64_t;
    HOST_DEVICE static inline Type convert(int64_t v) {
        return 9223372036854775808ull + v;
    }
    HOST_DEVICE static inline int64_t deconvert(Type v) {
        return v - 9223372036854775808ull;
    }
    HOST_DEVICE static inline int endbit() {
        return sizeof(Type) << 3;
    }
};

template <>
struct KeyTraits<double> {
    using Type = uint64_t;
    HOST_DEVICE static inline Type convert(double v) {
        Type x = *((Type *)&v);
        Type mask = -((x >> 63)) | 0x8000000000000000;
        return (x ^ mask);
    }
    HOST_DEVICE static inline double deconvert(Type v) {
        Type mask = ((v >> 63) - 1) | 0x8000000000000000;
        auto v_de = v ^ mask;
        return *((double *)&v_de);
    }
    HOST_DEVICE static inline int endbit() {
        return sizeof(Type) << 3;
    }
};

template <>
struct KeyTraits<dtype::Half> {
    using Type = uint16_t;
    HOST_DEVICE static inline Type convert(dtype::Half v) {
        Type x = *((Type *)&v);
        Type mask = -((x >> 15)) | 0x8000;
        return (x ^ mask);
    }
    HOST_DEVICE static inline dtype::Half deconvert(Type v) {
        Type mask = ((v >> 15) - 1) | 0x8000;
        Type v_de = v ^ mask;
        return reinterpret_cast<dtype::Half &>(v_de);
    }
    HOST_DEVICE static inline int endbit() {
        return sizeof(Type) << 3;
    }
};

template <>
struct KeyTraits<dtype::BFloat16> {
    using Type = uint16_t;
    HOST_DEVICE static inline Type convert(dtype::BFloat16 v) {
        Type x = *((Type *)&v);
        Type mask = -((x >> 15)) | 0x8000;
        return (x ^ mask);
    }
    HOST_DEVICE static inline dtype::BFloat16 deconvert(Type v) {
        Type mask = ((v >> 15) - 1) | 0x8000;
        Type v_de = v ^ mask;
        return reinterpret_cast<dtype::BFloat16 &>(v_de);
    }
    HOST_DEVICE static inline int endbit() {
        return sizeof(Type) << 3;
    }
};

template <int N, int CURRENT_VAL = N, int COUNT = 0>
struct Log2 {
    enum { VALUE = Log2<N, (CURRENT_VAL >> 1), COUNT + 1>::VALUE };
};

template <int N, int COUNT>
struct Log2<N, 0, COUNT> {
    enum { VALUE = (1 << (COUNT - 1) < N) ? COUNT : COUNT - 1 };
};

template <typename T, int STEPS>
DEVICE inline void warp_cumsum(
    ITEM &item,
    const int wid,
    const T input,
    T &inclusive_sum,
    T &exclusive_sum) {
    inclusive_sum = input;
#pragma unroll
    for (int i = 0, offset = 1; i < STEPS; ++i, offset <<= 1) {
        T temp = GPU_SHFL_UP(item, inclusive_sum, offset);
        if (wid >= offset)
            inclusive_sum += temp;
    }
    exclusive_sum = inclusive_sum - input;
}

template <
    typename T,
    int COUNTER_LANES,
    int BLOCK_SIZE,
    int WARP_SIZE,
    bool EXCLUSIVE = true>
DEVICE inline T block_cumsum(T *storage, ITEM &item) {
    static_assert(
        BLOCK_SIZE % WARP_SIZE == 0,
        "BLOCK_SIZE should be n * WARP_SIZE. (n = 1, 2, 3, ...)");

    const int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
    const int WARP_SCAN_STEPS = Log2<WARP_SIZE>::VALUE;

    int lid = item.thread_idx_x();

    int warp_local_id = lid % WARP_SIZE;
    int warp_id = lid / WARP_SIZE;
    int lane_temp_values[COUNTER_LANES];

    // Read input lane sum
    auto storage_lanes = storage + lid * COUNTER_LANES;
    T lane_all_sum = 0;

    if (EXCLUSIVE) {
#pragma unroll
        for (int lane = 0; lane < COUNTER_LANES; ++lane) {
            lane_temp_values[lane] = lane_all_sum;
            lane_all_sum += storage_lanes[lane];
        }
    } else {
#pragma unroll
        for (int lane = 0; lane < COUNTER_LANES; ++lane) {
            lane_all_sum += storage_lanes[lane];
            lane_temp_values[lane] = lane_all_sum;
        }
    }

    // Get warp level exclusive sum
    T warp_inclusive_sum, warp_exclusive_sum;
    warp_cumsum<T, WARP_SCAN_STEPS>(
        item,
        warp_local_id,
        lane_all_sum,
        warp_inclusive_sum,
        warp_exclusive_sum);
    item.barrier();

    // Write to storage
    if (warp_local_id == (WARP_SIZE - 1))
        storage[warp_id] = warp_inclusive_sum;
    item.barrier();

    // Get group prefix
    T group_all_sum = 0, group_exclusive_sum;
#pragma unroll
    for (int i = 0; i < NUM_WARPS; ++i) {
        if (warp_id == i)
            group_exclusive_sum = group_all_sum;
        group_all_sum += storage[i];
    }
    item.barrier();

    // Write to storage
    warp_exclusive_sum += group_exclusive_sum;
#pragma unroll
    for (int lane = 0; lane < COUNTER_LANES; ++lane) {
        storage_lanes[lane] = warp_exclusive_sum + lane_temp_values[lane];
    }
    item.barrier();

    return group_all_sum;
}

template <typename T, int COUNTER_LANES, int BLOCK_SIZE, int WARP_SIZE>
DEVICE inline T block_exclusive_cumsum(T *slm_storage, ITEM &item) {
    return block_cumsum<T, COUNTER_LANES, BLOCK_SIZE, WARP_SIZE, true>(
        slm_storage, item);
}

template <typename T, int COUNTER_LANES, int BLOCK_SIZE, int WARP_SIZE>
DEVICE inline T block_inclusive_cumsum(T *slm_storage, ITEM &item) {
    return block_cumsum<T, COUNTER_LANES, BLOCK_SIZE, WARP_SIZE, false>(
        slm_storage, item);
}
