#pragma once

#include <limits>
#include <algorithm>

#include "launcher.h"

namespace block {

template <int WARP_SIZE, typename func_t>
DEVICE_INLINE void warp_reduce_f32(ITEM &item, float *x, int wid, int warp_tid, float *out, func_t fn) {
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        *x = fn(*x, GPU_SHFL_DOWN(item, *x, offset, WARP_SIZE));
    }
    if (warp_tid == 0) out[wid] = *x;
}

struct WarpReduceSumFN {
    DEVICE_INLINE float operator()(float a, float b) const {
        return a + b;
    }
    DEVICE_INLINE float identity_element() const {
        return (float)0;
    }
};

struct WarpReduceMaxFN {
    DEVICE_INLINE float operator()(float a, float b) const {
        return std::max(a, b);
    }
    DEVICE_INLINE float identity_element() const {
        return -std::numeric_limits<float>::infinity();
    }
};

template <
    typename input_t,
    typename output_t,
    int M,
    int N,
    int K,
    bool IS_A_ROW_MAJOR,
    bool IS_B_ROW_MAJOR,
    bool ACCUMULATE>
DEVICE_INLINE void fma_dot_ref(
    ITEM &item,
    output_t *out_row_major,
    const input_t *a,
    const input_t *b,
    float alpha,
    float beta) {
    for (int idx = item.thread_idx_x(); idx < M * N; idx += item.thread_range_x()) {
        int mi = idx / N;
        int ni = idx % N;
        float sum = 0.0;
        for (int ki = 0; ki < K; ++ki) {
            int a_offset, b_offset;
            if constexpr (IS_A_ROW_MAJOR) {
                a_offset = mi * K + ki;
            } else {
                a_offset = ki * M + mi;
            }
            if constexpr (IS_B_ROW_MAJOR) {
                b_offset = ki * N + ni;
            } else {
                b_offset = ni * K + ki;
            }
            sum += (float)a[a_offset] * (float)b[b_offset];
        }
        float val = 0.0;
        if constexpr (ACCUMULATE) {
            val = out_row_major[mi * N + ni];
        }
        out_row_major[mi * N + ni] = alpha * sum + beta * val;
    }
}

template <
    int NUM_THREADS,
    typename input_t,
    typename output_t,
    int M,
    int N,
    int K,
    bool IS_A_ROW_MAJOR,
    bool IS_B_ROW_MAJOR,
    bool ACCUMULATE>
DEVICE void block_gemm(
    ITEM &item,
    output_t *out_row_major,
    const input_t *a,
    const input_t *b,
    float alpha,
    float beta) {
    static_assert(NUM_THREADS == 256);
    if constexpr (std::is_same_v<input_t, float>) {
        fma_dot_ref<input_t, output_t, M, N, K, IS_A_ROW_MAJOR, IS_B_ROW_MAJOR, ACCUMULATE>(
            item, out_row_major, a, b, alpha, beta);
    } else {
        constexpr int aspr = N / M;
        constexpr int WMMA_M = aspr >= 2 ? 8 : 16;
        constexpr int WMMA_N = aspr >= 2 ? 32 : 16;
        block_gemm_asic<
            input_t,
            output_t,
            M, N, K,
            2,      /* NUM_LANS_PER_WARP_M */
            2,      /* NUM_LANS_PER_WARP_N */
            2,      /* NUM_WARPS_M */
            4,      /* NUM_WARPS_N */
            WMMA_M, /* WMMA_M */
            WMMA_N, /* WMMA_N */
            16,     /* WMMA_K */
            GPU_WARP_SIZE, IS_A_ROW_MAJOR, IS_B_ROW_MAJOR, ACCUMULATE>(
            item, out_row_major, a, b, alpha, beta);
    }
}

} // namespace block
