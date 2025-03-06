#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>
#include <string>
#include <vector>
#include <ctime>

#include "utils.h"

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
    DEVICE_INLINE char *shared_ptr() {
        return smem_;
    };
private:
    char *smem_;
};

template <typename func_t, typename... args_t>
__global__ void kernel_wrapper(func_t fn, args_t &&...args) {
    extern __shared__ char smem[];
    auto info = ITEM(smem);
    fn(info, std::forward<args_t>(args)...);
}

class Launcher {
public:
    enum Direction {
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
        CHECK_FAIL(d >= 0 && d < device_count_);
        if (stream_) stream_end();
        current_device_ = d;
        CHECK_FAIL(cudaSetDevice(current_device_) == 0);
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

    void free(void *ptr) {
        if (ptr) CHECK_FAIL(cudaFree(ptr) == 0);
    }

    void memcpy(void *dst, const void *src, unsigned int len, Direction dir, bool sync = true) {
        bool need_new_stream = stream_ != 0 ? false : true;
        if (need_new_stream) stream_begin();
        switch (dir) {
        case Direction::H2D:
            CHECK_FAIL(cudaMemcpyAsync(dst, src, len * sizeof(char),
                                       cudaMemcpyHostToDevice, (cudaStream_t)stream_)
                       == 0);
            break;
        case Direction::D2H:
            CHECK_FAIL(cudaMemcpyAsync(dst, src, len * sizeof(char),
                                       cudaMemcpyDeviceToHost, (cudaStream_t)stream_)
                       == 0);
            break;
        case Direction::D2D:
            CHECK_FAIL(cudaMemcpyAsync(dst, src, len * sizeof(char),
                                       cudaMemcpyDeviceToDevice, (cudaStream_t)stream_)
                       == 0);
            break;
        default:
            CHECK_FAIL(false, "invalid direction");
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
        // Need intrinsic API
        srand((unsigned)time(0));
        CHECK_FAIL(cudaGetDeviceCount(&device_count_) == 0);
        for (int i = 0; i < device_count_; i++) {
            set_device(i, false);
            int dev;
            cudaDeviceProp prop;
            CHECK_FAIL(cudaGetDevice(&dev) == 0);
            CHECK_FAIL(cudaGetDeviceProperties(&prop, dev) == 0);
            device_names_.push_back(prop.name);
            device_max_thread_per_block_.push_back(prop.maxThreadsPerBlock);
            device_shared_memory_.push_back(prop.sharedMemPerBlock);
            device_global_memory_.push_back(prop.totalGlobalMem);
        }
// #ifdef DEBUG
        std::cout << "Device count: " << device_count_ << std::endl
                  << std::endl;
        for (int i = 0; i < device_count_; i++) {
            std::cout << "[" << i << "]Name: " << device_names_[i] << std::endl;
            std::cout << "MaxThreadPerBlock: " << device_max_thread_per_block_[i] << std::endl;
            std::cout << "SharedMemory(KB): " << device_shared_memory_[i] / 1024 << std::endl;
            std::cout << "GlobalMemory(MB): " << device_global_memory_[i] / 1024 / 1024 << std::endl;
        }
        std::cout << std::endl;
// #endif
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
    std::vector<size_t> device_shared_memory_;
    std::vector<size_t> device_global_memory_;
    int current_device_;
    size_t stream_;
    bool sync_mode_;
    bool profiling_mode_ = false;

public:
    template <typename func_t, typename... args_t>
    void submit(
        size_t slm_size,
        std::vector<int> grid_size,
        std::vector<int> block_size,
        func_t fn, args_t &&...args) {
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
            float milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);
            std::cout << milliseconds << " ms" << std::endl;
        } else {
            kernel_wrapper<<<grid, block, slm_size, (cudaStream_t)stream_>>>(fn,
                                                                             std::forward<args_t>(args)...);
        }

        if (is_sync_mode()) stream_sync();
    }
};
Launcher *Launcher::m_pInstance = new Launcher();

template <typename T, typename info_t>
DEVICE_INLINE T GPU_SHFL_UP(info_t &info, T value, unsigned int delta, int width = GPU_WARP_SIZE, unsigned int mask = 0xffffffff) {
    return __shfl_up_sync(mask, value, delta, width);
}

template <typename T, typename info_t>
DEVICE_INLINE T GPU_SHFL_DOWN(info_t &info, T value, unsigned int delta, int width = GPU_WARP_SIZE, unsigned int mask = 0xffffffff) {
    return __shfl_down_sync(mask, value, delta, width);
}

template <typename T, typename info_t>
DEVICE_INLINE T GPU_SHFL_XOR(info_t &info, T value, int laneMask, int width = GPU_WARP_SIZE, unsigned int mask = 0xffffffff) {
    return __shfl_xor_sync(mask, value, laneMask, width);
}
