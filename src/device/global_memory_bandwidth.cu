#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <random>

#include "device_info.h"
#include "utils.h"

using namespace std;

template <typename T, int vec_size>
__global__ void threads_copy_kernel(const T *in, T *out, const size_t n) {
    const int block_work_size = blockDim.x * vec_size;
    auto index = blockIdx.x * block_work_size + threadIdx.x * vec_size;
    auto remaining = n - index;
    if (remaining < vec_size) {
        for (auto i = index; i < n; i++) {
            out[i] = in[i];
        }
    } else {
        using vec_t = aligned_array<T, vec_size>;
        auto in_vec = reinterpret_cast<vec_t *>(const_cast<T *>(&in[index]));
        auto out_vec = reinterpret_cast<vec_t *>(&out[index]);
        *out_vec = *in_vec;
    }
}

template <typename T, int vec_size>
float threads_copy(const T *in, T *out, size_t n) {
    const int block_size = 1024;
    const int block_work_size = block_size * vec_size;

    dim3 threadsPerBlock(block_size);
    dim3 numBlocks((n + block_work_size - 1) / block_work_size);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    threads_copy_kernel<T, vec_size><<<numBlocks, threadsPerBlock>>>(in, out, n);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    return ms;
}

template <int vec_size>
void test_threads_copy(size_t n) {
    auto in_cpu = new float[n];
    auto out_cpu = new float[n];
    for (int i = 0; i < n; i++)
        in_cpu[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);

    float *in_cuda, *out_cuda;
    cudaMalloc(&in_cuda, n * sizeof(float));
    cudaMalloc(&out_cuda, n * sizeof(float));

    cudaMemcpy(in_cuda, in_cpu, n * sizeof(float), cudaMemcpyHostToDevice);

    float timems;
    for (int i = 0; i < 300; i++)
        timems = threads_copy<float, vec_size>(in_cuda, out_cuda, n);

    float total_GBytes = (n + n) * sizeof(float) / 1000.0 / 1000.0;
    std::cout << total_GBytes / (timems) << " GBPS ... ";

    cudaMemcpy(out_cpu, out_cuda, n * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; i++) {
        auto diff = out_cpu[i] - in_cpu[i];
        diff = diff > 0 ? diff : -diff;
        if (diff > 0.01) {
            std::cout << "error\n";
            return;
        }
    }
    std::cout << "ok\n";

    cudaFree(in_cuda);
    cudaFree(out_cuda);
    delete in_cpu;
    delete out_cpu;
}

void global_memory_bandwidth() {
    std::cout << "1GB threads copy test ...\n";
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
}
