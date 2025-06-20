#pragma once

#include <limits>
#include <algorithm>

#include "array.h"
#include "tensor_memory_access.h"

template <typename scalar_t, bool IS_CAUSAL = true>
struct CausalAttentionRefForwardFN {
    DEVICE void operator()(ITEM &item) const {
        auto b = item.block_idx_x();
        if (item.thread_idx_x() != 0) {
            return;
        }
        auto offset_b_q = b * q_seq_length_ * hidden_size_;
        auto offset_b_kv = b * kv_seq_length_ * hidden_size_;
        auto o_b = out_ + offset_b_q;
        auto q_b = q_ + offset_b_q;
        auto k_b = k_ + offset_b_kv;
        auto v_b = v_ + offset_b_kv;
        auto out_m_b = out_m_ + b * q_seq_length_;
        auto out_l_b = out_l_ + b * q_seq_length_;
        auto seq2 = qk_temp_ + b * q_seq_length_ * kv_seq_length_;
        for (int m = 0; m < q_seq_length_; m++) {
            for (int n = 0; n < kv_seq_length_; n++) {
                auto q_s = q_b + m * hidden_size_;
                auto k_s = k_b + n * hidden_size_;
                float sum = 0;
                for (int kk = 0; kk < hidden_size_; kk++)
                    sum += (float)(q_s[kk] * k_s[kk]);
                seq2[m * kv_seq_length_ + n] = sum * (1.0f / std::sqrt((float)hidden_size_));
            }
        }
        if constexpr (IS_CAUSAL) {
            for (int m = 0; m < q_seq_length_; m++)
                for (int n = 0; n < kv_seq_length_; n++)
                    seq2[m * kv_seq_length_ + n] =
                        m >= n ? seq2[m * kv_seq_length_ + n] : neg_inf_;
        }
        for (int m = 0; m < q_seq_length_; m++) {
            scalar_t max_value = neg_inf_;
            for (int n = 0; n < kv_seq_length_; n++) {
                max_value = std::max(seq2[m * kv_seq_length_ + n], max_value);
            }
            out_m_b[m] = max_value;
            float e2sum = 0;
            for (int n = 0; n < kv_seq_length_; n++) {
                e2sum += std::exp((float)seq2[m * kv_seq_length_ + n] - max_value);
            }
            out_l_b[m] = e2sum;
            for (int n = 0; n < kv_seq_length_; n++) {
                seq2[m * kv_seq_length_ + n] = std::exp(seq2[m * kv_seq_length_ + n] - max_value) / e2sum;
            }
        }
        for (int m = 0; m < q_seq_length_; m++) {
            for (int kk = 0; kk < hidden_size_; kk++) {
                float sum = 0;
                for (int n = 0; n < kv_seq_length_; n++) {
                    sum += (float)(seq2[m * kv_seq_length_ + n] * v_b[n * hidden_size_ + kk]);
                }
                o_b[m * hidden_size_ + kk] = sum;
            }
        }
    }
    CausalAttentionRefForwardFN(
        scalar_t *out,
        const scalar_t *q,
        const scalar_t *k,
        const scalar_t *v,
        const int batch_size,
        const int num_heads,
        const int q_seq_length,
        const int kv_seq_length,
        const int hidden_size,
        scalar_t *qk_temp, scalar_t *out_m, scalar_t *out_l,
        scalar_t neg_inf) :
        out_(out),
        q_(q), k_(k), v_(v),
        batch_size_(batch_size), num_heads_(num_heads),
        q_seq_length_(q_seq_length), kv_seq_length_(kv_seq_length),
        hidden_size_(hidden_size),
        qk_temp_(qk_temp), out_m_(out_m), out_l_(out_l),
        neg_inf_(neg_inf) {
    }

private:
    scalar_t *out_;
    const scalar_t *q_;
    const scalar_t *k_;
    const scalar_t *v_;
    const int batch_size_;
    const int num_heads_;
    const int q_seq_length_;
    const int kv_seq_length_;
    const int hidden_size_;
    scalar_t *qk_temp_;
    scalar_t *out_m_;
    scalar_t *out_l_;
    scalar_t neg_inf_;
};

template <typename scalar_t>
void causal_attention_ref_forward(
    scalar_t *out, const scalar_t *q, const scalar_t *k, const scalar_t *v,
    const int batch_size, const int num_heads, const int q_seq_length, const int kv_seq_length,
    const int hidden_size, scalar_t *qk_temp, scalar_t *out_m, scalar_t *out_l) {
    std::cout << "You're now entering the ref path, which will result in a significant performance drop.\n";
    auto l = Launcher::GetInstance();
    scalar_t neg_inf = -std::numeric_limits<scalar_t>::infinity();
    auto kernel = CausalAttentionRefForwardFN<scalar_t>(
        out, q, k, v, batch_size, num_heads, q_seq_length, kv_seq_length, hidden_size, qk_temp, out_m, out_l, neg_inf);
    l->submit(
        0,
        {batch_size * num_heads},
        {32},
        kernel);
}
