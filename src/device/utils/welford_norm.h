#pragma once

#include <tuple>
#include <algorithm>

#include "array.h"
#include "function_traits.h"
#include "tensor_memory_access.h"

// returns floor(log2(n))
inline int last_pow2(int n) {
    n |= (n >> 1);
    n |= (n >> 2);
    n |= (n >> 4);
    n |= (n >> 8);
    n |= (n >> 16);
    return std::max(1, n - (n >> 1));
}

template <typename T>
inline T divup(T a, T b) {
    return (a + b - 1) / b;
}

std::tuple<int, int, int, int> get_adaptive_config(
    const int problem_size,
    const int batch_size,
    const int vec_size,
    int max_wg_size,
    int loops_per_item = 8) {
    loops_per_item /= vec_size;
    int block_size_x = std::min(last_pow2(batch_size / vec_size), 32);
    int block_size_y = std::min(
        last_pow2(divup(problem_size, loops_per_item)), max_wg_size / block_size_x);
    if (block_size_x * block_size_y != max_wg_size) {
        block_size_x =
            std::min(last_pow2(batch_size / vec_size), max_wg_size / block_size_y);
    }

    int max_threads_gpu = 4 * GPU_WARP_SIZE * Launcher::GetInstance()->multi_processor_count();
    int nblock_x = divup(batch_size, block_size_x * vec_size);
    int nblock_y = std::min(
        divup(problem_size, block_size_y * loops_per_item),
        max_threads_gpu / (nblock_x * block_size_x) / (block_size_y));
    nblock_y = std::max(nblock_y, 1);

    // it's not worth having reduction between blocks if the reduction
    // dimension is not big enough
    // nblock_y = nblock_y < 4 ? 1 : nblock_y;
    nblock_y = 1;

    return std::make_tuple(block_size_y, block_size_x, nblock_y, nblock_x);
}

template <typename T, typename C>
DEVICE_INLINE void welford_merge(
    C &count,
    T &mean,
    T &m2n,
    const C &count_new,
    const T &mean_new,
    const T &m2n_new) {
    T factor = T(1.0) / std::max(1, (count + count_new));
    T delta0 = mean - mean_new;
    mean = (mean_new * count_new + mean * count) * factor;
    m2n += m2n_new + delta0 * delta0 * count_new * count * factor;
    count += count_new;
}

template <typename scalar_t, typename acc_t, int VEC_SIZE>
struct WelfordNormPFKernel {
    using vec_t = memory::aligned_array<scalar_t, VEC_SIZE>;
    using acc_vec_t = memory::aligned_array<acc_t, VEC_SIZE>;
    using int_vec_t = memory::aligned_array<int, VEC_SIZE>;

    DEVICE void operator()(ITEM &item) const {
        // init welford counters
        acc_vec_t mean;
        acc_vec_t m2n;
        int_vec_t count;
#pragma unroll
        for (int v = 0; v < VEC_SIZE; ++v) {
            mean[v] = acc_t(0);
            m2n[v] = acc_t(0);
            count[v] = int(0);
        }

        int bx = item.block_idx_x(); // along batch dim
        int by = item.block_idx_y(); // along problem dim
        int batch_vec_offset = (bx * item.thread_range_x() + item.thread_idx_x()) * VEC_SIZE;
        int num_cooperative_blocks = item.block_range_y();
        int inner_loop_stride = item.thread_range_y() * num_cooperative_blocks;

        if (batch_vec_offset < batch_size_) {
            for (int p_offset = (by * item.thread_range_y() + item.thread_idx_y()); p_offset < problem_size_;
                 p_offset += inner_loop_stride) {
                int address_vec_base = p_offset * batch_size_ + batch_vec_offset;
                auto input_vec = *reinterpret_cast<vec_t *>(
                    const_cast<scalar_t *>(&input_[address_vec_base]));
#pragma unroll
                for (int v = 0; v < VEC_SIZE; ++v) {
                    auto x = input_vec[v];
                    count[v]++;
                    acc_t delta0 = x - mean[v];
                    mean[v] += delta0 / count[v];
                    acc_t delta1 = x - mean[v];
                    m2n[v] += delta0 * delta1;
                }
            }
        }

        auto block_size = item.thread_range_x() * item.thread_range_y();
        auto shmem_mean = reinterpret_cast<acc_vec_t *>(item.shared_ptr());
        auto shmem_m2n = shmem_mean + block_size;
        auto shmem_count = reinterpret_cast<int_vec_t *>(shmem_m2n + block_size);
        welford_vertical_merge(item, count, mean, m2n, shmem_count, shmem_mean, shmem_m2n);
        // if (num_cooperative_blocks > 1) {
        // }
        if (item.thread_idx_y() == 0 && batch_vec_offset < batch_size_) {
            acc_vec_t invstd_vec;
#pragma unroll
            for (int v = 0; v < VEC_SIZE; ++v) {
                invstd_vec[v] = m2n[v] / count[v];
            }
            *reinterpret_cast<acc_vec_t *>(&save_mean_[batch_vec_offset]) = mean;
            *reinterpret_cast<acc_vec_t *>(&save_invstd_[batch_vec_offset]) = invstd_vec;
        }
    }

    WelfordNormPFKernel(const scalar_t *input,
                        acc_t *save_mean,
                        acc_t *save_invstd,
                        int batch_size,
                        int problem_size,
                        acc_t *staging_data = nullptr,
                        int *semaphores = nullptr) :
        input_(input),
        save_mean_(save_mean),
        save_invstd_(save_invstd), batch_size_(batch_size),
        problem_size_(problem_size), staging_data_(staging_data),
        semaphores_(semaphores) {
    }

    void init() {
        auto max_block_size = Launcher::GetInstance()->max_thread_per_block() / 2;
        std::tie(block_size_y_, block_size_x_, nblocks_y_, nblocks_x_) =
            get_adaptive_config(
                problem_size_, batch_size_, VEC_SIZE, max_block_size);
    }

    size_t slm_size() const {
        size_t local_size = block_size_x_ * block_size_y_;
        return local_size * (2 * sizeof(acc_vec_t) + sizeof(int_vec_t)) + 1;
    }

    static bool valid(
        int batch_size,
        int problem_size,
        const scalar_t *input,
        acc_t *save_mean,
        acc_t *save_invstd) {
        if (VEC_SIZE <= 1) return true;
        bool valid = sizeof(scalar_t) <= 4;
        valid = valid && (batch_size % VEC_SIZE == 0);
        valid = valid && (memory_access::can_vectorize_up_to<scalar_t>((char *)input) >= VEC_SIZE);
        valid = valid && (memory_access::can_vectorize_up_to<acc_t>((char *)save_mean) >= VEC_SIZE);
        valid = valid && (memory_access::can_vectorize_up_to<acc_t>((char *)save_invstd) >= VEC_SIZE);
        return valid;
    }

    std::tuple<int, int> get_block_size_yx() const {
        return std::make_tuple(block_size_y_, block_size_x_);
    }

    std::tuple<int, int> get_block_range_yx() const {
        return std::make_tuple(nblocks_y_, nblocks_x_);
    }

    int staging_size() const {
        return nblocks_y_ * batch_size_ * 4;
    }

    int semaphores_size() const {
        return nblocks_x_;
    }

    bool set_staging_data_check(acc_t *staging_data) {
        staging_data_ = staging_data;
        return (
            (staging_data == nullptr) || (memory_access::can_vectorize_up_to<acc_t>((char *)staging_data) >= VEC_SIZE));
    }

    void set_semaphores(int *semaphores) {
        semaphores_ = semaphores;
    }

    int num_cooperative_groups() const {
        return nblocks_y_;
    }

    DEVICE_INLINE void welford_vertical_merge(
        ITEM &item,
        int_vec_t &count,
        acc_vec_t &mean,
        acc_vec_t &m2n,
        int_vec_t *shmem_count,
        acc_vec_t *shmem_mean,
        acc_vec_t *shmem_m2n) const {
        // write to shared memory
        auto address_base = item.thread_idx_y() * item.thread_range_x() + item.thread_idx_x();
#pragma unroll
        for (int offset = item.thread_range_y() / 2; offset > 0; offset >>= 1) {
            if (item.thread_idx_y() < offset * 2) {
                shmem_mean[address_base] = mean;
                shmem_m2n[address_base] = m2n;
                shmem_count[address_base] = count;
            }
            item.barrier();
            if (item.thread_idx_y() < offset && item.thread_idx_y() + offset < item.thread_range_y()) {
                auto address = address_base + offset * item.thread_range_x();
                // read shared memory back to register for reduction
                auto count_new = shmem_count[address];
                auto mean_new = shmem_mean[address];
                auto m2n_new = shmem_m2n[address];
#pragma unroll
                for (int v = 0; v < VEC_SIZE; ++v) {
                    welford_merge(
                        count[v], mean[v], m2n[v], count_new[v], mean_new[v], m2n_new[v]);
                }
            }
        }
    }

private:
    const scalar_t *input_;
    acc_t *save_mean_;
    acc_t *save_invstd_;
    int batch_size_;
    int problem_size_;
    acc_t *staging_data_;
    int *semaphores_;

    int block_size_y_;
    int block_size_x_;
    int nblocks_y_;
    int nblocks_x_;
};
