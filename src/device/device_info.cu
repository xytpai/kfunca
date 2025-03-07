#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>

#include "device_info.h"

#include "array.h"
#include "launcher.h"

template <typename T, int vec_size>
struct ThreadCopyKernel {
    DEVICE void operator()(ITEM &item) const {
        const int block_work_size = item.thread_range_x() * vec_size;
        auto index = item.block_idx_x() * block_work_size + item.thread_idx_x() * vec_size;
        auto remaining = n_ - index;
        if (remaining < vec_size) {
            for (auto i = index; i < n_; i++) {
                out_[i] = in_[i];
            }
        } else {
            using vec_t = memory::aligned_array<T, vec_size>;
            auto in_vec = reinterpret_cast<vec_t *>(const_cast<T *>(&in_[index]));
            auto out_vec = reinterpret_cast<vec_t *>(&out_[index]);
            *out_vec = *in_vec;
        }
    }
    ThreadCopyKernel(const T *in, T *out, const size_t n) :
        in_(in), out_(out), n_(n) {
    }

private:
    const T *in_;
    T *out_;
    const size_t n_;
};

template <typename T, int vec_size>
float threads_copy(const T *in, T *out, size_t n) {
    const int block_size = 1024;
    const int block_work_size = block_size * vec_size;
    auto l = Launcher::GetInstance();
    l->set_profiling_mode(true);
    l->stream_begin();
    auto kernel = ThreadCopyKernel<T, vec_size>(in, out, n);
    auto ms = l->submit(0, {((int)n + block_work_size - 1) / block_work_size}, {block_size}, kernel);
    l->stream_sync();
    l->stream_end();
    return ms;
}

template <int vec_size>
void test_threads_copy(size_t n) {
    auto l = Launcher::GetInstance();
    auto in_cpu = new float[n];
    auto out_cpu = new float[n];
    for (int i = 0; i < n; i++)
        in_cpu[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);

    auto in_device = l->malloc<float>(n);
    auto out_device = l->malloc<float>(n);
    l->memcpy((void *)in_device, (void *)in_cpu, n * sizeof(float), Launcher::COPY::H2D);

    float timems;
    for (int i = 0; i < 300; i++)
        timems = threads_copy<float, vec_size>(in_device, out_device, n);

    float total_GBytes = (n + n) * sizeof(float) / 1000.0 / 1000.0;
    std::cout << total_GBytes / (timems) << " GBPS ... ";

    l->memcpy((void *)out_cpu, (void *)out_device, n * sizeof(float), Launcher::COPY::D2H);

    for (int i = 0; i < n; i++) {
        auto diff = out_cpu[i] - in_cpu[i];
        diff = diff > 0 ? diff : -diff;
        if (diff > 0.01) {
            std::cout << "error\n";
            return;
        }
    }
    std::cout << "ok\n";

    l->free(in_device);
    l->free(out_device);
    delete in_cpu;
    delete out_cpu;
}

template <int LOOP>
struct FMADLoopKernel {
    DEVICE void operator()(ITEM &item) const {
        int index = item.thread_idx_x() + item.block_idx_x() * item.thread_range_x();
        float a = x_[index], b = -1.0f;
        for (int i = 0; i < LOOP; i++) {
            for (int j = 0; j < LOOP; j++) {
                a = a * b + b;
            }
        }
        x_[index] = a;
    }
    FMADLoopKernel(float *x) :
        x_(x) {
    }

private:
    float *x_;
};

template <int LOOP, int block_size, int num_blocks>
float fmad_loop() {
    auto l = Launcher::GetInstance();
    l->set_profiling_mode(true);
    constexpr int n = block_size * num_blocks;

    auto x = new float[n];
    auto dx = l->malloc<float>(n);
    l->memcpy((void *)dx, (void *)x, n * sizeof(float), Launcher::COPY::H2D);

    l->stream_begin();
    auto kernel = FMADLoopKernel<LOOP>(dx);
    auto ms = l->submit(0, {num_blocks}, {block_size}, kernel);
    l->stream_sync();
    l->stream_end();

    l->memcpy((void *)x, (void *)dx, n * sizeof(float), Launcher::COPY::D2H);

    l->free(dx);
    delete[] x;
    return ms;
}

#define PRINT_PROP(PARAM) std::cout << #PARAM << ": " << prop.PARAM << std::endl;
#define ENDL_ std::cout << std::endl;
#define PRINT_(PARAM) std::cout << #PARAM << ": " << PARAM << std::endl;

void device_property() {
    int dev;
    cudaDeviceProp prop;
    cudaGetDevice(&dev);
    cudaGetDeviceProperties(&prop, dev);

    PRINT_PROP(name)
    PRINT_PROP(clockRate)
    PRINT_PROP(warpSize)
    PRINT_PROP(multiProcessorCount)
    ENDL_

    PRINT_PROP(totalConstMem)
    PRINT_PROP(totalGlobalMem)
    PRINT_PROP(memoryBusWidth)
    PRINT_PROP(memPitch)
    PRINT_PROP(unifiedAddressing)
    PRINT_PROP(unifiedFunctionPointers)
    PRINT_PROP(ECCEnabled)
    PRINT_PROP(l2CacheSize)
    PRINT_PROP(persistingL2CacheMaxSize)
    ENDL_

    PRINT_PROP(sharedMemPerBlock)
    PRINT_PROP(sharedMemPerBlockOptin)
    PRINT_PROP(sharedMemPerMultiprocessor)
    PRINT_PROP(localL1CacheSupported)
    PRINT_PROP(globalL1CacheSupported)
    ENDL_

    PRINT_PROP(maxThreadsPerBlock)
    PRINT_PROP(maxThreadsPerMultiProcessor)
    PRINT_PROP(maxBlocksPerMultiProcessor)
    ENDL_

    PRINT_PROP(regsPerMultiprocessor)
    PRINT_PROP(regsPerBlock)
    ENDL_

    PRINT_PROP(concurrentKernels)
    PRINT_PROP(directManagedMemAccessFromHost)
    PRINT_PROP(hostNativeAtomicSupported)
    ENDL_

    uint64_t clock_freq_khz = prop.clockRate;
    uint64_t cuda_cores = prop.multiProcessorCount * prop.warpSize * 4;
    PRINT_(cuda_cores)

    float fma_tflops = (2 * clock_freq_khz * cuda_cores) / 1e9f;
    PRINT_(fma_tflops)
}

#undef PRINT_PROP
#undef PRINT_PROP
#undef PRINT_

void device_info() {
    device_property();

    std::cout << "\n1GB threads copy test ...\n";
    std::cout << "float1: ";
    test_threads_copy<1>(1024 * 1024 * 256 + 2);
    std::cout << "float2: ";
    test_threads_copy<2>(1024 * 1024 * 256 + 2);
    std::cout << "float4: ";
    test_threads_copy<4>(1024 * 1024 * 256 + 2);
    std::cout << "float8: ";
    test_threads_copy<8>(1024 * 1024 * 256 + 2);
    std::cout << "float16: ";
    test_threads_copy<16>(1024 * 1024 * 256 + 2);

    std::cout << "\nFP32 MAD loop test ...\n";
    constexpr int LOOP = 10000;
    constexpr int block_size = 256;
    constexpr int num_blocks = 2048;
    for (int i = 0; i < 3; i++) {
        auto timems = fmad_loop<LOOP, block_size, num_blocks>();
        auto tflops =
            2.0 * LOOP * LOOP * num_blocks * block_size / (timems / 1000) * 1e-12;
        std::cout << tflops << " TFLOPS" << std::endl;
    }
}
