cmake_minimum_required(VERSION 3.12)
project(kfunca)

set(ROOTDIR ${PROJECT_SOURCE_DIR})

include_directories(${ROOTDIR}/src/device)
include_directories(${ROOTDIR}/src/device/utils)

find_package(CUDA REQUIRED)
if(CUDA_FOUND)
    enable_language(CUDA)
    add_definitions(-DUSE_CUDA)
    list(APPEND CUDA_NVCC_FLAGS "--expt-relaxed-constexpr")
    list(APPEND CUDA_NVCC_FLAGS "--std=c++17")
    file(GLOB KERNEL_SRCS "src/device/*.cu")
    add_library(device_code SHARED ${KERNEL_SRCS})
endif(CUDA_FOUND)

message(pybind11_DIR)
find_package(pybind11 REQUIRED)

pybind11_add_module(kfunca src/register.cpp)
target_link_libraries(kfunca PRIVATE device_code)
