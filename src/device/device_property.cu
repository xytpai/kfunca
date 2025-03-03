#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>

#include "device_info.h"

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
