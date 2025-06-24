#pragma once

#include <limits>
#include <algorithm>

#include "array.h"
#include "tensor_memory_access.h"

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

template <typename scalar_t, int SEQ_Q, int SEQ_KV, int NUM_THREADS, int WARP_SIZE, typename func_t>
DEVICE_INLINE void fmha_reduce(
    ITEM &item,
    scalar_t *out,
    const scalar_t *curr_mat,
    const scalar_t *prev,
    float *warp_reduce_temp,
    func_t func) {
    constexpr int NUM_WARPS = SEQ_KV / WARP_SIZE;
    int tid = item.thread_idx_x();
    int wid = tid / WARP_SIZE;
    int warp_tid = tid % WARP_SIZE;
#pragma unroll
    for (int m = 0; m < SEQ_Q; m++) {
        float x = tid < SEQ_KV ? (float)curr_mat[m * SEQ_KV + tid] : func.identity_element();
        item.barrier();
        warp_reduce_f32<WARP_SIZE>(item, &x, wid, warp_tid, warp_reduce_temp, func);
        item.barrier();
        if (tid == 0) {
            float value = warp_reduce_temp[0];
#pragma unroll
            for (int k = 1; k < NUM_WARPS; k++) {
                value = func(warp_reduce_temp[k], value);
            }
            out[m] = (scalar_t)func((float)prev[m], value);
        }
        item.barrier();
    }
}

template <
    typename input_t,
    typename output_t,
    int M,
    int N,
    int K,
    int NUM_THREADS,
    int WARP_SIZE,
    bool IS_B_ROW_MAJOR,
    bool ACCUMULATE>
DEVICE_INLINE void fma_dot_ref(
    ITEM &item,
    output_t *out_row_major,
    const input_t *a_row_major,
    const input_t *b,
    float *warp_reduce_temp,
    float scale) {
    constexpr int TOTAL_WARPS = NUM_THREADS / WARP_SIZE;
    constexpr int NUM_WARPS_PER_K = K / WARP_SIZE;
    static_assert(TOTAL_WARPS >= NUM_WARPS_PER_K);

    constexpr int SPLIT_N = TOTAL_WARPS / NUM_WARPS_PER_K;

    WarpReduceSumFN warp_reduce_sum_fn;

    int tid = item.thread_idx_x();
    int wid = tid / WARP_SIZE;
    int warp_tid = tid % WARP_SIZE;

    int wid_n = wid / NUM_WARPS_PER_K;
    int wid_k = wid % NUM_WARPS_PER_K;

    for (int mi = 0; mi < M; mi++) {
        for (int ni = 0; ni < N; ni += SPLIT_N) {
            int a_offset = mi * K + wid_k * WARP_SIZE + warp_tid;
            int b_offset;
            if constexpr (IS_B_ROW_MAJOR) {
                b_offset = (wid_k * WARP_SIZE + warp_tid) * N + ni + wid_n;
            } else {
                b_offset = (ni + wid_n) * K + wid_k * WARP_SIZE + warp_tid;
            }
            float sum = a_row_major[a_offset] * b[b_offset];
            warp_reduce_f32<WARP_SIZE>(item, &sum, wid, warp_tid, warp_reduce_temp, warp_reduce_sum_fn);
            item.barrier();
            if (wid_k == 0 && warp_tid == 0) {
                float wsum = 0;
#pragma unroll
                for (int i = 0; i < NUM_WARPS_PER_K; i++)
                    wsum += warp_reduce_temp[wid_n * NUM_WARPS_PER_K + i];
                if constexpr (ACCUMULATE) {
                    out_row_major[mi * N + ni + wid_n] += (output_t)(wsum * scale);
                } else {
                    out_row_major[mi * N + ni + wid_n] = (output_t)(wsum * scale);
                }
            }
            item.barrier();
        }
    }
}

} // namespace block

template <typename scalar_t, int BLOCK_Q, int BLOCK_KV, int HIDDEN_SIZE, int NUM_WARPS>
struct AttentionSLMBlock {
    float o[BLOCK_Q * HIDDEN_SIZE];
    scalar_t q[BLOCK_Q * HIDDEN_SIZE];
    scalar_t k[BLOCK_KV * HIDDEN_SIZE];
    scalar_t v[BLOCK_KV * HIDDEN_SIZE];
    scalar_t qk[BLOCK_Q * BLOCK_KV];
    scalar_t p[BLOCK_Q * BLOCK_KV];
    scalar_t m_curr[BLOCK_Q];
    scalar_t l_curr[BLOCK_Q];
    scalar_t m_prev[BLOCK_Q];
    scalar_t l_prev[BLOCK_Q];
    scalar_t l_rcp[BLOCK_Q];
    float warp_reduce_temp[NUM_WARPS];
};

#define FMHA_FOR_LOOP_SYNC(IDX, N, VEC_SIZE, FN)                                                     \
    {                                                                                                \
        for (int IDX = item.thread_idx_x() * VEC_SIZE; IDX < N; IDX += BLOCK_THREADS * VEC_SIZE) FN; \
        item.barrier();                                                                              \
    }

template <typename scalar_t, int VEC_SIZE, int BLOCK_Q, int BLOCK_KV, int HIDDEN_SIZE, int BLOCK_THREADS, int WARP_SIZE = GPU_WARP_SIZE>
struct CausalAttentionForwardFN {
    static_assert(HIDDEN_SIZE % VEC_SIZE == 0);
    using AttentionSLMBlockT = AttentionSLMBlock<scalar_t, BLOCK_Q, BLOCK_KV, HIDDEN_SIZE, BLOCK_THREADS / WARP_SIZE>;
    using vec_t = memory::aligned_array<scalar_t, VEC_SIZE>;
    using vec_acc_t = memory::aligned_array<float, VEC_SIZE>;

    DEVICE void operator()(ITEM &item) const {
        auto shared = reinterpret_cast<AttentionSLMBlockT *>(item.shared_ptr());
        auto bx = item.block_idx_x();
        auto by = item.block_idx_y();
        auto offset_q_batch = bx * q_seq_length_ * HIDDEN_SIZE;
        auto offset_kv_batch = bx * kv_seq_length_ * HIDDEN_SIZE;
        auto offset_q_seq = by * BLOCK_Q * HIDDEN_SIZE;
        auto o_bs = out_ + offset_q_batch + offset_q_seq;
        auto q_bs = const_cast<scalar_t *>(q_) + offset_q_batch + offset_q_seq;
        auto k_b = const_cast<scalar_t *>(k_) + offset_kv_batch;
        auto v_b = const_cast<scalar_t *>(v_) + offset_kv_batch;
        auto o_ms = out_m_ + bx * q_seq_length_ + by * BLOCK_Q;
        auto o_ls = out_l_ + bx * q_seq_length_ + by * BLOCK_Q;

        FMHA_FOR_LOOP_SYNC(i, BLOCK_Q, 1, {
            shared->m_prev[i] = -1e20;
            shared->l_prev[i] = 0;
        });

        vec_acc_t acc_zero;
#pragma unroll
        for (int ii = 0; ii < VEC_SIZE; ++ii) {
            acc_zero[ii] = 0;
        }

        FMHA_FOR_LOOP_SYNC(i, (BLOCK_Q * HIDDEN_SIZE), VEC_SIZE, {
            *reinterpret_cast<vec_acc_t *>(&shared->o[i]) = acc_zero;
            *reinterpret_cast<vec_t *>(&shared->q[i]) = *reinterpret_cast<vec_t *>(&q_bs[i]);
        });

        block::WarpReduceMaxFN warp_reduce_max_fn;
        block::WarpReduceSumFN warp_reduce_sum_fn;

        for (int start_kv_s = 0; start_kv_s < kv_seq_length_; start_kv_s += BLOCK_KV) {
            FMHA_FOR_LOOP_SYNC(i, (BLOCK_KV * HIDDEN_SIZE), VEC_SIZE, {
                *reinterpret_cast<vec_t *>(&shared->k[i]) =
                    *reinterpret_cast<vec_t *>(&k_b[start_kv_s * HIDDEN_SIZE + i]);
            });

            block::fma_dot_ref<scalar_t, scalar_t, BLOCK_Q, BLOCK_KV, HIDDEN_SIZE, BLOCK_THREADS, WARP_SIZE, false, false>(
                item, shared->qk, shared->q, shared->k, shared->warp_reduce_temp,
                1.0f / std::sqrt((float)HIDDEN_SIZE));

            FMHA_FOR_LOOP_SYNC(i, BLOCK_Q * BLOCK_KV, 1, {
                int row = by * BLOCK_Q + i / BLOCK_KV;
                int col = start_kv_s + i % BLOCK_KV;
                shared->qk[i] = row >= col ? shared->qk[i] : -1e20;
            });

            block::fmha_reduce<scalar_t, BLOCK_Q, BLOCK_KV, BLOCK_THREADS, WARP_SIZE>(
                item, shared->m_curr, shared->qk, shared->m_prev, shared->warp_reduce_temp, warp_reduce_max_fn);

            FMHA_FOR_LOOP_SYNC(i, BLOCK_Q, 1, {
                shared->l_prev[i] *= std::exp(shared->m_prev[i] - shared->m_curr[i]);
            });

            FMHA_FOR_LOOP_SYNC(i, BLOCK_Q * BLOCK_KV, 1, {
                int j = i / BLOCK_KV;
                shared->p[i] = std::exp(shared->qk[i] - shared->m_curr[j]);
            });

            block::fmha_reduce<scalar_t, BLOCK_Q, BLOCK_KV, BLOCK_THREADS, WARP_SIZE>(
                item, shared->l_curr, shared->p, shared->l_prev, shared->warp_reduce_temp, warp_reduce_sum_fn);

            FMHA_FOR_LOOP_SYNC(i, BLOCK_Q, 1, {
                shared->l_rcp[i] = 1.0f / shared->l_curr[i];
            });

            FMHA_FOR_LOOP_SYNC(i, (BLOCK_Q * BLOCK_KV), 1, {
                int j = i / BLOCK_KV;
                shared->p[i] *= shared->l_rcp[j];
            });

            FMHA_FOR_LOOP_SYNC(i, (BLOCK_Q * HIDDEN_SIZE), 1, {
                int j = i / HIDDEN_SIZE;
                shared->o[i] *= shared->l_prev[j] * shared->l_rcp[j];
            });

            FMHA_FOR_LOOP_SYNC(i, (BLOCK_KV * HIDDEN_SIZE), VEC_SIZE, {
                *reinterpret_cast<vec_t *>(&shared->v[i]) =
                    *reinterpret_cast<vec_t *>(&v_b[start_kv_s * HIDDEN_SIZE + i]);
            });

            block::fma_dot_ref<scalar_t, float, BLOCK_Q, HIDDEN_SIZE, BLOCK_KV,
                               BLOCK_THREADS, WARP_SIZE, true, true>(
                item, shared->o, shared->p, shared->v, shared->warp_reduce_temp, 1.0f);

            FMHA_FOR_LOOP_SYNC(i, BLOCK_Q, 1, {
                shared->l_prev[i] = shared->l_curr[i];
            });

            FMHA_FOR_LOOP_SYNC(i, BLOCK_Q, 1, {
                shared->m_prev[i] = shared->m_curr[i];
            });
        }

        FMHA_FOR_LOOP_SYNC(i, (BLOCK_Q * HIDDEN_SIZE), VEC_SIZE, {
            *reinterpret_cast<vec_t *>(&o_bs[i]) = *reinterpret_cast<vec_t *>(&shared->o[i]);
        });

        FMHA_FOR_LOOP_SYNC(i, BLOCK_Q, VEC_SIZE, {
            *reinterpret_cast<vec_t *>(&o_ms[i]) = *reinterpret_cast<vec_t *>(&shared->m_prev[i]);
        });

        FMHA_FOR_LOOP_SYNC(i, BLOCK_Q, VEC_SIZE, {
            *reinterpret_cast<vec_t *>(&o_ls[i]) = *reinterpret_cast<vec_t *>(&shared->l_prev[i]);
        });
    }
    CausalAttentionForwardFN(
        scalar_t *out, const scalar_t *q, const scalar_t *k, const scalar_t *v,
        const int q_seq_length, const int kv_seq_length, scalar_t *out_m, scalar_t *out_l) :
        out_(out),
        q_(q), k_(k), v_(v),
        q_seq_length_(q_seq_length), kv_seq_length_(kv_seq_length),
        out_m_(out_m), out_l_(out_l) {
    }

    int shared_size() const {
        return sizeof(AttentionSLMBlockT);
    }

private:
    scalar_t *out_;
    const scalar_t *q_;
    const scalar_t *k_;
    const scalar_t *v_;
    const int q_seq_length_;
    const int kv_seq_length_;
    scalar_t *out_m_;
    scalar_t *out_l_;
};

template <typename scalar_t, int VEC_SIZE, int BLOCK_Q, int BLOCK_KV, int HIDDEN_SIZE>
int causal_attention_forward(
    scalar_t *out, const scalar_t *q, const scalar_t *k, const scalar_t *v,
    const int batch_size, const int num_heads, const int q_seq_length, const int kv_seq_length,
    scalar_t *out_m, scalar_t *out_l) {
    if ((q_seq_length % BLOCK_Q != 0) || (kv_seq_length % BLOCK_KV != 0))
        return -1;
    constexpr int BLOCK_THREADS = 256;
    constexpr int WARP_SIZE = GPU_WARP_SIZE;
    if (batch_size * num_heads * q_seq_length == 0) return 0;

    auto l = Launcher::GetInstance();
    auto kernel = CausalAttentionForwardFN<scalar_t, VEC_SIZE, BLOCK_Q, BLOCK_KV, HIDDEN_SIZE,
                                           BLOCK_THREADS, WARP_SIZE>(
        out, q, k, v, q_seq_length, kv_seq_length, out_m, out_l);

    auto slm_size = kernel.shared_size();
    CHECK_FAIL(slm_size <= l->shared_local_memory_size());
    l->submit(
        slm_size,
        {batch_size * num_heads, q_seq_length / BLOCK_Q},
        {BLOCK_THREADS},
        kernel);
    return 0;
}

template <typename scalar_t, int HIDDEN_SIZE>
int causal_attention_forward_fma_ref(
    scalar_t *out, const scalar_t *q, const scalar_t *k, const scalar_t *v,
    const int batch_size, const int num_heads, const int q_seq_length, const int kv_seq_length,
    scalar_t *out_m, scalar_t *out_l) {
    int vec_size = 4;
    vec_size = std::min(vec_size, memory_access::can_vectorize_up_to<scalar_t>((char *)out));
    vec_size = std::min(vec_size, memory_access::can_vectorize_up_to<scalar_t>((char *)const_cast<scalar_t *>(q)));
    vec_size = std::min(vec_size, memory_access::can_vectorize_up_to<scalar_t>((char *)const_cast<scalar_t *>(k)));
    vec_size = std::min(vec_size, memory_access::can_vectorize_up_to<scalar_t>((char *)const_cast<scalar_t *>(v)));
    int ret;
    switch (vec_size) {
    case 4:
        ret = causal_attention_forward<scalar_t, 4, 8, 32, HIDDEN_SIZE>(
            out, q, k, v, batch_size, num_heads, q_seq_length, kv_seq_length, out_m, out_l);
        break;
    case 2:
        ret = causal_attention_forward<scalar_t, 2, 8, 32, HIDDEN_SIZE>(
            out, q, k, v, batch_size, num_heads, q_seq_length, kv_seq_length, out_m, out_l);
        break;
    default:
        ret = causal_attention_forward<scalar_t, 1, 8, 32, HIDDEN_SIZE>(
            out, q, k, v, batch_size, num_heads, q_seq_length, kv_seq_length, out_m, out_l);
        break;
    }
    return ret;
}
