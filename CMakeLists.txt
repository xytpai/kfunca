cmake_minimum_required(VERSION 3.12)
project(kfunca)

set(ROOTDIR ${PROJECT_SOURCE_DIR})

include_directories(${ROOTDIR}/src/device)
include_directories(${ROOTDIR}/src/device/include)
include_directories(${ROOTDIR}/src/device/utils)
include_directories(${ROOTDIR}/src/device/utils/memory)

find_package(CUDA REQUIRED)
if(CUDA_FOUND)
    enable_language(CUDA)
    add_definitions(-DUSE_CUDA)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --std=c++17")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
    file(GLOB KERNEL_SRCS "src/device/*.cu")
    add_library(kfunca_device SHARED ${KERNEL_SRCS})
endif(CUDA_FOUND)

include_directories(${ROOTDIR}/src/core)
include_directories(${ROOTDIR}/src/core/include)
file(GLOB CORE_SRCS "src/core/*.cpp")
add_library(kfunca_core SHARED ${CORE_SRCS})

message(pybind11_DIR)
find_package(pybind11 REQUIRED)

pybind11_add_module(kfunca src/register.cpp)
target_link_libraries(kfunca PRIVATE kfunca_device kfunca_core)
