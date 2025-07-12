#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <mma.h>
#include <cuda_fp16.h>

#include <iostream>
#include <string>
#include <vector>
#include <ctime>

#include "exception.h"

#ifndef HOST_DEVICE
#define HOST_DEVICE __host__ __device__
#endif

#ifndef HOST
#define HOST __host__
#endif

#ifndef DEVICE
#define DEVICE __device__
#endif

#ifndef HOST_DEVICE_INLINE
#define HOST_DEVICE_INLINE __host__ __device__ __forceinline__
#endif

#ifndef DEVICE_INLINE
#define DEVICE_INLINE __device__ __forceinline__
#endif

#define GPU_WARP_SIZE 32

using namespace utils;

struct ITEM {
    using index_t = const unsigned int; // May change on some device
    DEVICE_INLINE ITEM(char *smem) :
        smem_(smem) {
    }
    DEVICE_INLINE index_t thread_idx_x() const {
        return threadIdx.x;
    }
    DEVICE_INLINE index_t thread_idx_y() const {
        return threadIdx.y;
    }
    DEVICE_INLINE index_t thread_idx_z() const {
        return threadIdx.z;
    }
    DEVICE_INLINE index_t thread_range_x() const {
        return blockDim.x;
    }
    DEVICE_INLINE index_t thread_range_y() const {
        return blockDim.y;
    }
    DEVICE_INLINE index_t thread_range_z() const {
        return blockDim.z;
    }
    DEVICE_INLINE index_t block_idx_x() const {
        return blockIdx.x;
    }
    DEVICE_INLINE index_t block_idx_y() const {
        return blockIdx.y;
    }
    DEVICE_INLINE index_t block_idx_z() const {
        return blockIdx.z;
    }
    DEVICE_INLINE index_t block_range_x() const {
        return gridDim.x;
    }
    DEVICE_INLINE index_t block_range_y() const {
        return gridDim.y;
    }
    DEVICE_INLINE index_t block_range_z() const {
        return gridDim.z;
    }
    DEVICE_INLINE void barrier() {
        __syncthreads();
    }
    DEVICE_INLINE void memory_order_fence() {
        __threadfence();
    }
    DEVICE_INLINE char *shared_ptr() {
        return smem_;
    };
    DEVICE_INLINE int fetch_atomic_add(int *addr, int val) {
        return atomicAdd(addr, val);
    }

private:
    char *smem_;
};

template <typename func_t, typename... args_t>
__global__ void kernel_wrapper(func_t fn, args_t &&...args) {
    extern __shared__ char smem[];
    auto it = ITEM(smem);
    fn(it, std::forward<args_t>(args)...);
}

class Launcher {
public:
    enum COPY {
        D2H = 0,
        H2D,
        D2D
    };

    static Launcher *
    GetInstance() {
        return m_pInstance;
    }

    // Intrinsic API for device control

    void stream_begin() {
        if (stream_) stream_end();
        CHECK_FAIL(cudaStreamCreate((cudaStream_t *)&stream_) == 0);
    }

    void stream_sync() {
        if (stream_) CHECK_FAIL(cudaStreamSynchronize((cudaStream_t)stream_) == 0);
    }

    void stream_end() {
        stream_sync();
        CHECK_FAIL(cudaStreamDestroy((cudaStream_t)stream_) == 0);
        stream_ = 0;
    }

    void reset_device() {
        CHECK_FAIL(cudaDeviceReset() == 0);
    }

    void set_device(int d, bool reset = true) {
        if (current_device_ != d) {
            CHECK_FAIL(d >= 0 && d < device_count_);
            if (stream_) stream_end();
            current_device_ = d;
            CHECK_FAIL(cudaSetDevice(current_device_) == 0);
        }
        if (reset) reset_device();
    }

    int device() const {
        return current_device_;
    }

    template <typename T>
    T *malloc(size_t len) {
        T *ptr = nullptr;
        CHECK_FAIL(cudaMalloc((void **)&ptr, len * sizeof(T)) == 0);
        return ptr;
    }

    void *malloc_(size_t size) {
        void *ptr = nullptr;
        CHECK_FAIL(cudaMalloc(&ptr, size) == 0);
        return ptr;
    }

    void free(void *ptr) {
        if (ptr) CHECK_FAIL(cudaFree(ptr) == 0);
    }

    void memcpy(void *dst, const void *src, size_t size, COPY dir, bool sync = true) {
        bool need_new_stream = stream_ != 0 ? false : true;
        if (need_new_stream) stream_begin();
        switch (dir) {
        case COPY::H2D:
            CHECK_FAIL(cudaMemcpyAsync(dst, src, size,
                                       cudaMemcpyHostToDevice, (cudaStream_t)stream_)
                       == 0);
            break;
        case COPY::D2H:
            CHECK_FAIL(cudaMemcpyAsync(dst, src, size,
                                       cudaMemcpyDeviceToHost, (cudaStream_t)stream_)
                       == 0);
            break;
        case COPY::D2D:
            CHECK_FAIL(cudaMemcpyAsync(dst, src, size,
                                       cudaMemcpyDeviceToDevice, (cudaStream_t)stream_)
                       == 0);
            break;
        default:
            CHECK_FAIL(false, "invalid copy direction");
        }
        if (sync) stream_sync();
        if (need_new_stream) stream_end();
    }

    void memset(void *ptr, int value, size_t count, bool sync = true) {
        bool need_new_stream = stream_ != 0 ? false : true;
        if (need_new_stream) stream_begin();
        cudaMemsetAsync(ptr, value, count, (cudaStream_t)stream_);
        if (sync) stream_sync();
        if (need_new_stream) stream_end();
    }

    // For property

    int device_count() const {
        return device_count_;
    }
    size_t stream() const {
        return stream_;
    }

    std::string device_names() const {
        return device_names_[current_device_];
    }

    int max_thread_per_block() const {
        return device_max_thread_per_block_[current_device_];
    }

    int multi_processor_count() const {
        return device_multi_processor_count_[current_device_];
    }

    int max_threads_per_multi_processor() const {
        return device_max_threads_per_multi_processor_[current_device_];
    }

    int max_thread_per_group() const {
        return max_thread_per_block();
    }

    int work_size_for_loops() const {
        return max_thread_per_block() / 4;
    }

    size_t shared_local_memory_size() const {
        return device_shared_memory_[current_device_];
    }

    size_t global_memory_size() const {
        return device_global_memory_[current_device_];
    }

    void set_sync_mode(bool is_sync) {
        sync_mode_ = is_sync;
    }

    bool is_sync_mode() const {
        return sync_mode_;
    }

    void set_profiling_mode(bool status) {
        profiling_mode_ = status;
    }

private:
    Launcher() {
        current_device_ = -1;
        // Need intrinsic API
        srand((unsigned)time(0));
        stream_ = 0;
        CHECK_FAIL(cudaGetDeviceCount(&device_count_) == 0);
        for (int i = 0; i < device_count_; i++) {
            set_device(i, false);
            int dev;
            cudaDeviceProp prop;
            CHECK_FAIL(cudaGetDevice(&dev) == 0);
            CHECK_FAIL(cudaGetDeviceProperties(&prop, dev) == 0);
            device_names_.push_back(prop.name);
            device_max_thread_per_block_.push_back(prop.maxThreadsPerBlock);
            device_multi_processor_count_.push_back(prop.multiProcessorCount);
            device_max_threads_per_multi_processor_.push_back(prop.maxThreadsPerMultiProcessor);
            device_shared_memory_.push_back(prop.sharedMemPerBlock);
            device_global_memory_.push_back(prop.totalGlobalMem);
        }
#ifdef DEBUG
        std::cout << "Device count: " << device_count_ << std::endl
                  << std::endl;
        for (int i = 0; i < device_count_; i++) {
            std::cout << "[" << i << "]Name: " << device_names_[i] << std::endl;
            std::cout << "MaxThreadPerBlock: " << device_max_thread_per_block_[i] << std::endl;
            std::cout << "SharedMemory(KB): " << device_shared_memory_[i] / 1024 << std::endl;
            std::cout << "GlobalMemory(MB): " << device_global_memory_[i] / 1024 / 1024 << std::endl;
        }
        std::cout << std::endl;
#endif
        current_device_ = -1;
        set_device(0);
        sync_mode_ = true;
    }

    ~Launcher() {
        set_device(0); // reset state
    }

    Launcher(const Launcher &) = delete;
    Launcher &operator=(const Launcher &) = delete;

    static Launcher *m_pInstance;

    int device_count_;
    std::vector<std::string> device_names_;
    std::vector<int> device_max_thread_per_block_;
    std::vector<int> device_multi_processor_count_;
    std::vector<int> device_max_threads_per_multi_processor_;
    std::vector<size_t> device_shared_memory_;
    std::vector<size_t> device_global_memory_;
    int current_device_;
    size_t stream_;
    bool sync_mode_;
    bool profiling_mode_ = false;

public:
    template <typename func_t, typename... args_t>
    float submit(
        size_t slm_size,
        std::vector<int> grid_size,
        std::vector<int> block_size,
        func_t fn, args_t &&...args) {
        float milliseconds = 0;
        dim3 grid, block;
        if (grid_size.size() == 1)
            grid = dim3(grid_size[0]);
        else if (grid_size.size() == 2)
            grid = dim3(grid_size[0], grid_size[1]);
        else if (grid_size.size() == 3)
            grid = dim3(grid_size[0], grid_size[1], grid_size[2]);
        if (block_size.size() == 1)
            block = dim3(block_size[0]);
        else if (block_size.size() == 2)
            block = dim3(block_size[0], block_size[1]);
        else if (block_size.size() == 3)
            block = dim3(block_size[0], block_size[1], block_size[2]);

        if (profiling_mode_) {
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start);
            kernel_wrapper<<<grid, block, slm_size, (cudaStream_t)stream_>>>(fn,
                                                                             std::forward<args_t>(args)...);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&milliseconds, start, stop);
        } else {
            kernel_wrapper<<<grid, block, slm_size, (cudaStream_t)stream_>>>(fn,
                                                                             std::forward<args_t>(args)...);
        }

        if (is_sync_mode()) stream_sync();
        return milliseconds;
    }
};

template <typename T, typename item_t>
DEVICE_INLINE T GPU_SHFL_UP(item_t &item, T value, unsigned int delta, int width = GPU_WARP_SIZE, unsigned int mask = 0xffffffff) {
    return __shfl_up_sync(mask, value, delta, width);
}

template <typename T, typename item_t>
DEVICE_INLINE T GPU_SHFL_DOWN(item_t &item, T value, unsigned int delta, int width = GPU_WARP_SIZE, unsigned int mask = 0xffffffff) {
    return __shfl_down_sync(mask, value, delta, width);
}

template <typename T, typename item_t>
DEVICE_INLINE T GPU_SHFL_XOR(item_t &item, T value, int laneMask, int width = GPU_WARP_SIZE, unsigned int mask = 0xffffffff) {
    return __shfl_xor_sync(mask, value, laneMask, width);
}

/**
 * block_gemm:
 * out = alpha * a @ b + beta * out
 */
template <
    typename input_t,
    typename output_t,
    int M_DIM,
    int N_DIM,
    int K_DIM,
    int NUM_LANS_PER_WARP_M,
    int NUM_LANS_PER_WARP_N,
    int NUM_WARPS_M,
    int NUM_WARPS_N,
    int WMMA_M,
    int WMMA_N,
    int WMMA_K,
    int WARP_SIZE,
    bool IS_A_ROW_MAJOR,
    bool IS_B_ROW_MAJOR,
    bool ACCUMULATE>
DEVICE_INLINE void block_gemm_asic(
    ITEM &item,
    output_t *out_row_major,
    const input_t *a,
    const input_t *b,
    float alpha = 1.0,
    float beta = 1.0) {
    constexpr int M = NUM_LANS_PER_WARP_M * NUM_WARPS_M * WMMA_M;
    constexpr int N = NUM_LANS_PER_WARP_N * NUM_WARPS_N * WMMA_N;
    constexpr int STEP_K = WMMA_K * 2;

    static_assert(M_DIM <= M && N_DIM <= N);
    static_assert(M_DIM % WMMA_M == 0 && N_DIM % WMMA_N == 0);
    static_assert(K_DIM % STEP_K == 0);
    static_assert(NUM_WARPS_M * NUM_WARPS_N == 8); // 256th

    int wid = item.thread_idx_x() / WARP_SIZE;
    int wid_m = wid / NUM_WARPS_N;
    int wid_n = wid % NUM_WARPS_N;

    using a_frag_t = std::conditional_t<
        IS_A_ROW_MAJOR,
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, input_t, nvcuda::wmma::row_major>,
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, input_t, nvcuda::wmma::col_major>>;
    a_frag_t a_frag[2][NUM_LANS_PER_WARP_M];

    using b_frag_t = std::conditional_t<
        IS_B_ROW_MAJOR,
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, input_t, nvcuda::wmma::row_major>,
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, input_t, nvcuda::wmma::col_major>>;
    b_frag_t b_frag[2][NUM_LANS_PER_WARP_N];

    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K,
                           float>
        o_frag[NUM_LANS_PER_WARP_M][NUM_LANS_PER_WARP_N];

#pragma unroll
    for (int i = 0; i < NUM_LANS_PER_WARP_M; i++) {
#pragma unroll
        for (int j = 0; j < NUM_LANS_PER_WARP_N; j++) {
            nvcuda::wmma::fill_fragment(o_frag[i][j], 0.0f);
        }
    }

    for (int ki = 0; ki < K_DIM; ki += STEP_K) {
#pragma unroll
        for (int i = 0; i < NUM_LANS_PER_WARP_M; i++) {
            int mi = (wid_m + i * NUM_WARPS_M) * WMMA_M;
            if (mi < M_DIM) {
                if constexpr (IS_A_ROW_MAJOR) {
                    nvcuda::wmma::load_matrix_sync(a_frag[0][i], a + mi * K_DIM + ki, K_DIM);
                    nvcuda::wmma::load_matrix_sync(a_frag[1][i], a + mi * K_DIM + ki + WMMA_K, K_DIM);
                } else {
                    nvcuda::wmma::load_matrix_sync(a_frag[0][i], a + mi + ki * N_DIM, N_DIM);
                    nvcuda::wmma::load_matrix_sync(a_frag[1][i], a + mi + (ki + WMMA_K) * N_DIM, N_DIM);
                }
            }
        }
#pragma unroll
        for (int i = 0; i < NUM_LANS_PER_WARP_N; i++) {
            int ni = (wid_n + i * NUM_WARPS_N) * WMMA_N;
            if (ni < N_DIM) {
                if constexpr (IS_B_ROW_MAJOR) {
                    nvcuda::wmma::load_matrix_sync(b_frag[0][i], b + ni + ki * N_DIM, N_DIM);
                    nvcuda::wmma::load_matrix_sync(b_frag[1][i], b + ni + (ki + WMMA_K) * N_DIM, N_DIM);
                } else {
                    nvcuda::wmma::load_matrix_sync(b_frag[0][i], b + ni * K_DIM + ki, K_DIM);
                    nvcuda::wmma::load_matrix_sync(b_frag[1][i], b + ni * K_DIM + ki + WMMA_K, K_DIM);
                }
            }
        }
#pragma unroll
        for (int i = 0; i < NUM_LANS_PER_WARP_M; i++) {
#pragma unroll
            for (int j = 0; j < NUM_LANS_PER_WARP_N; j++) {
                nvcuda::wmma::mma_sync(o_frag[i][j], a_frag[0][i], b_frag[0][j], o_frag[i][j]);
                nvcuda::wmma::mma_sync(o_frag[i][j], a_frag[1][i], b_frag[1][j], o_frag[i][j]);
            }
        }
    }

    // write back
    for (int i = 0; i < NUM_LANS_PER_WARP_M; i++) {
        for (int j = 0; j < NUM_LANS_PER_WARP_N; j++) {
            for (int k = 0; k < o_frag[i][j].num_elements; k++) {
                o_frag[i][j].x[k] *= alpha;
            }
            int mi = (wid_m + i * NUM_WARPS_M) * WMMA_M;
            int ni = (wid_n + j * NUM_WARPS_N) * WMMA_N;
            if (mi < M_DIM && ni < N_DIM) {
                int out_offset = mi * N_DIM + ni;
                if constexpr (ACCUMULATE) {
                    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, output_t> c_frag;
                    nvcuda::wmma::load_matrix_sync(c_frag, out_row_major + out_offset, N_DIM, nvcuda::wmma::mem_row_major);
                    for (int k = 0; k < c_frag.num_elements; k++) {
                        c_frag.x[k] = (output_t)((float)c_frag.x[k] * beta + o_frag[i][j].x[k]);
                    }
                    nvcuda::wmma::store_matrix_sync(out_row_major + out_offset, c_frag, N_DIM, nvcuda::wmma::mem_row_major);
                } else {
                    if constexpr (std::is_same<output_t, float>::value) {
                        nvcuda::wmma::store_matrix_sync(
                            out_row_major + out_offset, o_frag[i][j], N_DIM, nvcuda::wmma::mem_row_major);
                    } else {
                        nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, output_t> c_frag;
                        for (int k = 0; k < c_frag.num_elements; k++) {
                            c_frag.x[k] = (output_t)(o_frag[i][j].x[k]);
                        }
                        nvcuda::wmma::store_matrix_sync(out_row_major + out_offset, c_frag, N_DIM, nvcuda::wmma::mem_row_major);
                    }
                }
            }
            item.barrier();
        }
    }
}
