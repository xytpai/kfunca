#pragma once

#include <limits>
#include <algorithm>

#include "array.h"
#include "tensor_memory_access.h"

#define FMHA_FOR_LOOP_SYNC(IDX, N, FN)                                                 \
    {                                                                                  \
        for (int IDX = item.thread_idx_x(); IDX < N; IDX += item.thread_range_x()) FN; \
        item.barrier();                                                                \
    }

namespace block {

template <typename scalar_t, int WARP_SIZE, typename item_t>
DEVICE_INLINE void warp_reduce_sum(item_t &item, scalar_t *x, int wid, int warp_tid, scalar_t *out) {
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        *x = *x + GPU_SHFL_DOWN(item, *x, offset, WARP_SIZE);
    }
    if (warp_tid == 0) out[wid] = *x;
}

template <typename scalar_t, int WARP_SIZE, typename item_t>
DEVICE_INLINE void warp_reduce_max(item_t &item, scalar_t *x, int wid, int warp_tid, scalar_t *out) {
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        *x = std::max(*x, GPU_SHFL_DOWN(item, *x, offset, WARP_SIZE));
    }
    if (warp_tid == 0) out[wid] = *x;
}

template <typename scalar_t, int SEQ_N, int C, int NUM_THREADS, int WARP_SIZE, typename item_t>
DEVICE_INLINE void fmha_dot(item_t &item, scalar_t *out, const scalar_t *a, const scalar_t *b, scalar_t *temp, scalar_t scale) {
    constexpr int NUM_WARPS_PER_CNL = C / WARP_SIZE;
    constexpr int SEQ_BLOCK_SIZE = NUM_THREADS / WARP_SIZE / NUM_WARPS_PER_CNL;

    int tid = item.thread_idx_x();
    int wid = tid / WARP_SIZE;
    int warp_tid = tid % WARP_SIZE;

    int wid_s = wid / SEQ_BLOCK_SIZE;
    int wid_c = wid % SEQ_BLOCK_SIZE;

#pragma unroll
    for (int m = 0; m < SEQ_N; m++) {
        for (int n = 0; n < SEQ_N; n += SEQ_BLOCK_SIZE) {
            auto a_offset = m * C + wid_c * WARP_SIZE + warp_tid;
            auto b_offset = (n + wid_s) * C + wid_c * WARP_SIZE + warp_tid;
            scalar_t sum = a[a_offset] * b[b_offset];
            warp_reduce_sum<scalar_t, WARP_SIZE>(item, &sum, wid, warp_tid, temp);
            item.barrier();
            if (wid_c == 0 && warp_tid == 0) {
                scalar_t wsum = 0;
#pragma unroll
                for (int i = 0; i < NUM_WARPS_PER_CNL; i++)
                    wsum += temp[wid_s * NUM_WARPS_PER_CNL + i];
                out[m * SEQ_N + n + wid_s] = wsum * scale;
            }
            item.barrier();
        }
    }
}

template <typename scalar_t, int SEQ_N, int C, int NUM_THREADS, int WARP_SIZE, typename item_t>
DEVICE_INLINE void fmha_dot_acc(item_t &item, scalar_t *out, const scalar_t *a, const scalar_t *b, scalar_t *temp) {
    constexpr int CNL_BLOCK_SIZE = NUM_THREADS / WARP_SIZE;

    int tid = item.thread_idx_x();
    int wid = tid / WARP_SIZE;
    int warp_tid = tid % WARP_SIZE;

#pragma unroll
    for (int m = 0; m < SEQ_N; m++) {
        for (int n = 0; n < C; n += CNL_BLOCK_SIZE) {
            auto a_offset = m * SEQ_N + warp_tid;
            auto b_offset = warp_tid * C + n + wid;
            scalar_t sum = a[a_offset] * b[b_offset];
            warp_reduce_sum<scalar_t, WARP_SIZE>(item, &sum, wid, warp_tid, temp);
            item.barrier();
            if (warp_tid == 0) {
                scalar_t wsum = 0;
                out[m * C + n + wid] += temp[wid];
            }
            item.barrier();
        }
    }
}

template <typename scalar_t, int SEQ_N, int NUM_THREADS, int WARP_SIZE, typename item_t>
DEVICE_INLINE void fmha_max(item_t &item, scalar_t *out, const scalar_t *curr_mat, const scalar_t *prev, scalar_t *temp) {
    constexpr int NUM_WARPS = SEQ_N / WARP_SIZE;
    int tid = item.thread_idx(0);
    int wid = tid / WARP_SIZE;
    int warp_tid = tid % WARP_SIZE;
    const scalar_t neg_inf = -std::numeric_limits<scalar_t>::infinity();
#pragma unroll
    for (int m = 0; m < SEQ_N; m++) {
        scalar_t x = tid < SEQ_N ? curr_mat[m * SEQ_N + tid] : neg_inf;
        item.barrier();
        warp_reduce_max<scalar_t, WARP_SIZE>(item, &x, wid, warp_tid, temp);
        item.barrier();
        if (tid == 0) {
            scalar_t wmax = neg_inf;
#pragma unroll
            for (int k = 0; k < NUM_WARPS; k++) {
                wmax = std::max(temp[k], wmax);
            }
            out[m] = std::max(prev[m], wmax);
        }
        item.barrier();
    }
}

template <typename scalar_t, int SEQ_N, int NUM_THREADS, int WARP_SIZE, typename item_t>
DEVICE_INLINE void fmha_sum(item_t &item, scalar_t *out, const scalar_t *curr_mat, const scalar_t *prev, scalar_t *temp) {
    constexpr int NUM_WARPS = SEQ_N / WARP_SIZE;
    int tid = item.thread_idx_x();
    int wid = tid / WARP_SIZE;
    int warp_tid = tid % WARP_SIZE;
#pragma unroll
    for (int m = 0; m < SEQ_N; m++) {
        scalar_t x = tid < SEQ_N ? curr_mat[m * SEQ_N + tid] : 0;
        item.barrier();
        warp_reduce_sum<scalar_t, WARP_SIZE>(item, &x, wid, warp_tid, temp);
        item.barrier();
        if (tid == 0) {
            scalar_t wsum = 0;
#pragma unroll
            for (int k = 0; k < NUM_WARPS; k++) {
                wsum += temp[k];
            }
            out[m] = wsum + prev[m];
        }
        item.barrier();
    }
}

} // namespace block

template <typename scalar_t, int BLOCK_M, int HIDDEN_SIZE, int BLOCK_THREADS, int WARP_SIZE, typename item_t>
DEVICE void causal_attention_forward_kernel(
    item_t &item, scalar_t *out, const scalar_t *q,
    const scalar_t *k, const scalar_t *v, const int seq_length,
    scalar_t *out_m, scalar_t *out_l) {
    constexpr int BSIZE = BLOCK_M * HIDDEN_SIZE;

    auto acc = reinterpret_cast<float *>(item.shared_ptr());
    auto q_temp = acc + BSIZE;
    auto k_temp = q_temp + BSIZE;
    auto v_temp = k_temp + BSIZE;
    auto qk_temp = v_temp + BSIZE;
    auto p = qk_temp + BLOCK_M * BLOCK_M;
    auto m_prev = p + BLOCK_M * BLOCK_M;
    auto l_prev = m_prev + BLOCK_M;
    auto m_curr = l_prev + BLOCK_M;
    auto l_curr = m_curr + BLOCK_M;
    auto l_rcp = l_curr + BLOCK_M;
    auto warp_reduce_temp = l_rcp + BLOCK_M;

    auto offset_b = item.block_idx_x() * seq_length * HIDDEN_SIZE;
    auto offset_seq = item.block_idx_y() * BLOCK_M * HIDDEN_SIZE;
    auto o_bs = out + offset_b + offset_seq;
    auto q_bs = q + offset_b + offset_seq;
    auto k_b = k + offset_b;
    auto v_b = v + offset_b;

    offset_b = item.block_idx_x() * seq_length;
    auto o_ms = out_m + offset_b + item.block_idx_y() * BLOCK_M;
    auto o_ls = out_l + offset_b + item.block_idx_y() * BLOCK_M;

    FMHA_FOR_LOOP_SYNC(i, BLOCK_M, {
        m_prev[i] = -1e20;
        l_prev[i] = 0;
    });

    FMHA_FOR_LOOP_SYNC(i, BSIZE, {
        acc[i] = 0;
        q_temp[i] = q_bs[i];
    });

    for (int start_s = 0; start_s < seq_length; start_s += BLOCK_M) {
        FMHA_FOR_LOOP_SYNC(i, BSIZE, {
            k_temp[i] = k_b[start_s * HIDDEN_SIZE + i];
        });

        block::fmha_dot<float, BLOCK_M, HIDDEN_SIZE, BLOCK_THREADS, WARP_SIZE>(
            item, qk_temp, q_temp, k_temp, warp_reduce_temp, 1.0f / std::sqrt((float)HIDDEN_SIZE));

        FMHA_FOR_LOOP_SYNC(i, BLOCK_M * BLOCK_M, {
            int row = item.block_idx(1) * BLOCK_M + i / BLOCK_M;
            int col = start_s + i % BLOCK_M;
            qk_temp[i] = row >= col ? qk_temp[i] : -1e20;
        });

        block::fmha_max<scalar_t, BLOCK_M, BLOCK_THREADS, WARP_SIZE>(
            item, m_curr, qk_temp, m_prev, warp_reduce_temp);

        FMHA_FOR_LOOP_SYNC(i, BLOCK_M, {
            l_prev[i] *= std::exp(m_prev[i] - m_curr[i]);
        });

        FMHA_FOR_LOOP_SYNC(i, BLOCK_M * BLOCK_M, {
            int j = i / BLOCK_M;
            p[i] = std::exp(qk_temp[i] - m_curr[j]);
        });

        block::fmha_sum<scalar_t, BLOCK_M, BLOCK_THREADS, WARP_SIZE>(
            item, l_curr, p, l_prev, warp_reduce_temp);

        FMHA_FOR_LOOP_SYNC(i, BLOCK_M, {
            l_rcp[i] = 1.0f / l_curr[i];
        });

        FMHA_FOR_LOOP_SYNC(i, BLOCK_M * BLOCK_M, {
            int j = i / BLOCK_M;
            p[i] *= l_rcp[j];
        });

        FMHA_FOR_LOOP_SYNC(i, BSIZE, {
            int j = i / HIDDEN_SIZE;
            acc[i] *= l_prev[j] * l_rcp[j];
        });

        FMHA_FOR_LOOP_SYNC(i, BSIZE, {
            v_temp[i] = v_b[start_s * HIDDEN_SIZE + i];
        });

        block::fmha_dot_acc<float, BLOCK_M, HIDDEN_SIZE, BLOCK_THREADS, WARP_SIZE>(
            item, acc, p, v_temp, warp_reduce_temp);

        FMHA_FOR_LOOP_SYNC(i, BLOCK_M, {
            l_prev[i] = l_curr[i];
        });

        FMHA_FOR_LOOP_SYNC(i, BLOCK_M, {
            m_prev[i] = m_curr[i];
        });
    }

    FMHA_FOR_LOOP_SYNC(i, BSIZE, {
        o_bs[i] = acc[i];
    });

    FMHA_FOR_LOOP_SYNC(i, BLOCK_M, {
        o_ms[i] = m_prev[i];
    });

    FMHA_FOR_LOOP_SYNC(i, BLOCK_M, {
        o_ls[i] = l_prev[i];
    });
}

template <typename scalar_t, int BLOCK_M, int HIDDEN_SIZE, int BLOCK_THREADS, int WARP_SIZE>
struct CausalAttentionForwardFN {
    DEVICE void operator()(ITEM &item) const {
        causal_attention_forward_kernel<scalar_t, BLOCK_M, HIDDEN_SIZE, BLOCK_THREADS, WARP_SIZE>(
            item, out_, q_, k_, v_, seq_length_, out_m_, out_l_);
    }
    CausalAttentionForwardFN(scalar_t *out, const scalar_t *q, const scalar_t *k, const scalar_t *v,
                             const int batch_size, const int num_heads, const int seq_length,
                             scalar_t *out_m, scalar_t *out_l) :
        out_(out),
        q_(q), k_(k), v_(v),
        batch_size_(batch_size), num_heads_(num_heads), seq_length_(seq_length),
        out_m_(out_m), out_l_(out_l) {
    }

private:
    scalar_t *out_;
    const scalar_t *q_;
    const scalar_t *k_;
    const scalar_t *v_;
    const int batch_size_;
    const int num_heads_;
    const int seq_length_;
    scalar_t *out_m_;
    scalar_t *out_l_;
};

template <typename scalar_t, int BLOCK_M = 32, int HIDDEN_SIZE = 64>
void causal_attention_forward(
    scalar_t *out, const scalar_t *q, const scalar_t *k, const scalar_t *v,
    const int batch_size, const int num_heads, const int seq_length,
    scalar_t *out_m, scalar_t *out_l) {
    constexpr int BLOCK_THREADS = 128;
    constexpr int WARP_SIZE = 32;
    constexpr int NUM_WARPS = BLOCK_THREADS / WARP_SIZE;

    CHECK_FAIL(seq_length % BLOCK_M == 0);
    CHECK_FAIL(BLOCK_M == WARP_SIZE); // TODO: for experimental

    if (batch_size * num_heads * seq_length == 0) return;

    auto slm_size = (4 * BLOCK_M * HIDDEN_SIZE
                     + 2 * BLOCK_M * BLOCK_M
                     + 5 * BLOCK_M
                     + NUM_WARPS)
                    * sizeof(float);

    auto l = Launcher::GetInstance();
    CHECK_FAIL(slm_size <= l->shared_local_memory_size());
    auto kernel = CausalAttentionForwardFN<scalar_t, BLOCK_M, HIDDEN_SIZE, BLOCK_THREADS, WARP_SIZE>(
        out, q, k, v, batch_size, num_heads, seq_length, out_m, out_l);
    l->submit(
        slm_size,
        {batch_size * num_heads, seq_length / BLOCK_M},
        {BLOCK_THREADS},
        kernel);
}
