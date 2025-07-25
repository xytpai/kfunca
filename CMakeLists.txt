cmake_minimum_required(VERSION 3.12)
project(kfunca)

set(ROOTDIR ${PROJECT_SOURCE_DIR})

include_directories(${ROOTDIR}/src/device)
include_directories(${ROOTDIR}/src/device/include)
include_directories(${ROOTDIR}/src/device/utils)
include_directories(${ROOTDIR}/src/device/utils/memory)
include_directories(${ROOTDIR}/third_party/cutlass/include)

set(CMAKE_CXX_STANDARD 20)

find_package(CUDA REQUIRED)
if(CUDA_FOUND)
    enable_language(CUDA)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --std=c++20")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_80")
    # set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")
    file(GLOB KERNEL_SRCS "src/device/*.cu")
    add_library(kfunca_device SHARED ${KERNEL_SRCS})
endif(CUDA_FOUND)

include_directories(${ROOTDIR}/src/core)
include_directories(${ROOTDIR}/src/core/include)
include_directories(${ROOTDIR}/src/core/utils)
include_directories(${ROOTDIR}/src/core/utils/memory)
file(GLOB CORE_SRCS "src/core/*.cpp")
add_library(kfunca_core SHARED ${CORE_SRCS})

message(pybind11_DIR)
find_package(pybind11 REQUIRED)

pybind11_add_module(kfunca src/register.cpp)
target_link_libraries(kfunca PRIVATE kfunca_core kfunca_device)

file(GLOB TEST_SRCS "test/core/*.cpp")
foreach(TEST_SRC ${TEST_SRCS})
    get_filename_component(TEST_BASE_NAME ${TEST_SRC} NAME_WE)
    add_executable(${TEST_BASE_NAME}.out ${TEST_SRC})
    target_link_libraries(${TEST_BASE_NAME}.out PRIVATE kfunca_core kfunca_device)
endforeach()
