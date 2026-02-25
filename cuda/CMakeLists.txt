cmake_minimum_required(VERSION 3.18)

# Detect ROCm/HIP vs CUDA build
if(SECP256K1_BUILD_ROCM AND hip_FOUND)
    project(Secp256k1Cuda LANGUAGES CXX HIP)
    set(_GPU_LANG HIP)
    message(STATUS "secp256k1-gpu: Building for ROCm/HIP (AMD)")
else()
    # CMAKE_CUDA_ARCHITECTURES is set by parent CMakeLists.txt
    project(Secp256k1Cuda LANGUAGES CXX CUDA)
    set(_GPU_LANG CUDA)
    message(STATUS "secp256k1-gpu: Building for CUDA (NVIDIA)")
endif()

option(SECP256K1_CUDA_USE_MONTGOMERY "Use Montgomery field arithmetic backend in CUDA" OFF)
option(SECP256K1_CUDA_LIMBS_32 "Use 8x32-bit limbs for field arithmetic" OFF)

if(_GPU_LANG STREQUAL "CUDA")
    set(CMAKE_CUDA_STANDARD 17)
    # Set CUDA architecture if not already set by parent (fallback to 89 = Ada Lovelace)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "" OR CMAKE_CUDA_ARCHITECTURES STREQUAL "OFF")
        set(CMAKE_CUDA_ARCHITECTURES 89)
    endif()
endif()
set(CMAKE_CXX_STANDARD 17)

include_directories(include ${CMAKE_CURRENT_SOURCE_DIR}/../include)

# Source files -- .cu extension works with both nvcc and hipcc
set(_GPU_SOURCES src/secp256k1.cu)

# Library target
if(_GPU_LANG STREQUAL "HIP")
    # HIP: set source language explicitly
    set_source_files_properties(${_GPU_SOURCES} PROPERTIES LANGUAGE HIP)
    add_library(secp256k1_cuda_lib STATIC ${_GPU_SOURCES})
    target_compile_options(secp256k1_cuda_lib PRIVATE
        $<$<COMPILE_LANGUAGE:HIP>:-O3>
        $<$<COMPILE_LANGUAGE:HIP>:-ffast-math>
    )
else()
    add_library(secp256k1_cuda_lib STATIC ${_GPU_SOURCES})
    target_compile_options(secp256k1_cuda_lib PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
        $<$<COMPILE_LANGUAGE:CUDA>:-O3>
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-O3>
    )
    set_target_properties(secp256k1_cuda_lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

target_include_directories(secp256k1_cuda_lib PUBLIC
    include
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

if(SECP256K1_CUDA_USE_MONTGOMERY)
    target_compile_definitions(secp256k1_cuda_lib PUBLIC SECP256K1_CUDA_USE_MONTGOMERY=1)
endif()

if(SECP256K1_CUDA_LIMBS_32)
    target_compile_definitions(secp256k1_cuda_lib PUBLIC SECP256K1_CUDA_LIMBS_32=1)
endif()

# Test suite
set(_TEST_SOURCES src/test_suite.cu)
if(_GPU_LANG STREQUAL "HIP")
    set_source_files_properties(${_TEST_SOURCES} PROPERTIES LANGUAGE HIP)
endif()
add_executable(secp256k1_cuda_test ${_TEST_SOURCES})
target_link_libraries(secp256k1_cuda_test PRIVATE secp256k1_cuda_lib)

# Benchmark executable
set(_BENCH_SOURCES src/bench_cuda.cu)
if(_GPU_LANG STREQUAL "HIP")
    set_source_files_properties(${_BENCH_SOURCES} PROPERTIES LANGUAGE HIP)
    add_executable(secp256k1_cuda_bench ${_BENCH_SOURCES})
    target_compile_options(secp256k1_cuda_bench PRIVATE
        $<$<COMPILE_LANGUAGE:HIP>:-O3>
        $<$<COMPILE_LANGUAGE:HIP>:-ffast-math>
    )
else()
    add_executable(secp256k1_cuda_bench ${_BENCH_SOURCES})
    target_compile_options(secp256k1_cuda_bench PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
        $<$<COMPILE_LANGUAGE:CUDA>:-O3>
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-O3>
    )
endif()
target_link_libraries(secp256k1_cuda_bench PRIVATE secp256k1_cuda_lib)

# CTest integration
enable_testing()
add_test(NAME cuda_selftest COMMAND secp256k1_cuda_test)

