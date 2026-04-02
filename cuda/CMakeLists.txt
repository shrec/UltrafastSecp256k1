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
    # Required for device-code linking when secp256k1_cuda_lib uses separable
    # compilation. Without this, executables that link the static lib get
    # undefined references to __cudaRegisterLinkedBinary_* at link time
    # (CUDA 12+ / Blackwell / Ada and later architectures).
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
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
    set_target_properties(secp256k1_cuda_lib PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON)
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

# ===== GPU Unified Audit Runner =====
set(_AUDIT_SOURCES src/gpu_audit_runner.cu)
if(_GPU_LANG STREQUAL "HIP")
    set_source_files_properties(${_AUDIT_SOURCES} PROPERTIES LANGUAGE HIP)
    add_executable(gpu_audit_runner ${_AUDIT_SOURCES})
    target_compile_options(gpu_audit_runner PRIVATE
        $<$<COMPILE_LANGUAGE:HIP>:-O3>
        $<$<COMPILE_LANGUAGE:HIP>:-ffast-math>
    )
else()
    add_executable(gpu_audit_runner ${_AUDIT_SOURCES})
    target_compile_options(gpu_audit_runner PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
        $<$<COMPILE_LANGUAGE:CUDA>:-O3>
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-O3>
    )
endif()
target_link_libraries(gpu_audit_runner PRIVATE secp256k1_cuda_lib)
# Link CPU library for CPU-GPU differential cross-verification
if(TARGET fastsecp256k1)
    target_include_directories(gpu_audit_runner PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../cpu/include
    )
    target_link_libraries(gpu_audit_runner PRIVATE
        $<TARGET_FILE:fastsecp256k1>
    )
    target_compile_definitions(gpu_audit_runner PRIVATE HAVE_CPU_LIB=1)
    set_target_properties(gpu_audit_runner PROPERTIES INTERPROCEDURAL_OPTIMIZATION FALSE)
    target_link_options(gpu_audit_runner PRIVATE -fno-lto)
endif()
add_test(NAME gpu_audit COMMAND gpu_audit_runner --json-only)

# ===== GPU Unified Benchmark =====
set(_BENCH_UNIFIED_SOURCES src/gpu_bench_unified.cu)
if(_GPU_LANG STREQUAL "HIP")
    set_source_files_properties(${_BENCH_UNIFIED_SOURCES} PROPERTIES LANGUAGE HIP)
    add_executable(gpu_bench_unified ${_BENCH_UNIFIED_SOURCES})
    target_compile_options(gpu_bench_unified PRIVATE
        $<$<COMPILE_LANGUAGE:HIP>:-O3>
        $<$<COMPILE_LANGUAGE:HIP>:-ffast-math>
    )
else()
    add_executable(gpu_bench_unified ${_BENCH_UNIFIED_SOURCES})
    target_compile_options(gpu_bench_unified PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
        $<$<COMPILE_LANGUAGE:CUDA>:-O3>
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-O3>
    )
endif()
target_link_libraries(gpu_bench_unified PRIVATE secp256k1_cuda_lib)

# ===== BIP-324 Transport Benchmark (CUDA batch parallel) =====
set(_BENCH_BIP324_SOURCES src/bench_bip324_transport.cu)
if(_GPU_LANG STREQUAL "HIP")
    set_source_files_properties(${_BENCH_BIP324_SOURCES} PROPERTIES LANGUAGE HIP)
    add_executable(bench_bip324_cuda ${_BENCH_BIP324_SOURCES})
    target_compile_options(bench_bip324_cuda PRIVATE
        $<$<COMPILE_LANGUAGE:HIP>:-O3>
        $<$<COMPILE_LANGUAGE:HIP>:-ffast-math>
    )
else()
    add_executable(bench_bip324_cuda ${_BENCH_BIP324_SOURCES})
    target_compile_options(bench_bip324_cuda PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
        $<$<COMPILE_LANGUAGE:CUDA>:-O3>
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-O3>
    )
    set_target_properties(bench_bip324_cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF)
endif()

# ===== CPU vs GPU Comparison Benchmark =====
set(_BENCH_COMPARE_SOURCES src/bench_compare.cu)
if(_GPU_LANG STREQUAL "HIP")
    set_source_files_properties(${_BENCH_COMPARE_SOURCES} PROPERTIES LANGUAGE HIP)
    add_executable(bench_compare ${_BENCH_COMPARE_SOURCES})
    target_compile_options(bench_compare PRIVATE
        $<$<COMPILE_LANGUAGE:HIP>:-O3>
        $<$<COMPILE_LANGUAGE:HIP>:-ffast-math>
    )
else()
    add_executable(bench_compare ${_BENCH_COMPARE_SOURCES})
    target_compile_options(bench_compare PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
        $<$<COMPILE_LANGUAGE:CUDA>:-O3>
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-O3>
    )
endif()
# Link both GPU and CPU libraries for side-by-side comparison.
# The CPU target suppresses INTERFACE LTO propagation when CUDA is enabled, so
# linking the target directly keeps build dependencies correct and still lets
# nvcc's host linker consume the fat archive with -fno-lto.
target_link_libraries(bench_compare PRIVATE secp256k1_cuda_lib)
target_include_directories(bench_compare PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../cpu/include
)
target_link_libraries(bench_compare PRIVATE
    fastsecp256k1
)
set_target_properties(bench_compare PROPERTIES INTERPROCEDURAL_OPTIMIZATION FALSE)
target_link_options(bench_compare PRIVATE -fno-lto)

# ===== BIP-352 Silent Payments Pipeline Benchmark =====
set(_BENCH_BIP352_SOURCES src/bench_bip352.cu)
if(_GPU_LANG STREQUAL "HIP")
    set_source_files_properties(${_BENCH_BIP352_SOURCES} PROPERTIES LANGUAGE HIP)
    add_executable(bench_bip352 ${_BENCH_BIP352_SOURCES})
    target_compile_options(bench_bip352 PRIVATE
        $<$<COMPILE_LANGUAGE:HIP>:-O3>
        $<$<COMPILE_LANGUAGE:HIP>:-ffast-math>
    )
else()
    add_executable(bench_bip352 ${_BENCH_BIP352_SOURCES})
    target_compile_options(bench_bip352 PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
        $<$<COMPILE_LANGUAGE:CUDA>:-O3>
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-O3>
    )
endif()
target_link_libraries(bench_bip352 PRIVATE secp256k1_cuda_lib)
target_include_directories(bench_bip352 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../cpu/include
)
target_link_libraries(bench_bip352 PRIVATE
    fastsecp256k1
)
set_target_properties(bench_bip352 PROPERTIES
    INTERPROCEDURAL_OPTIMIZATION FALSE
    # bench_bip352 is self-contained: all device code is defined inline in
    # bench_bip352.cu + secp256k1.cuh headers; it does not call __device__
    # functions from secp256k1_cuda_lib's .cu TU.  Disabling separable
    # compilation for this target only restores whole-program nvcc register
    # allocation, allowing tpb >= 256 and recovering ~15% throughput vs the
    # global CMAKE_CUDA_SEPARABLE_COMPILATION=ON default.
    CUDA_SEPARABLE_COMPILATION OFF
)
target_link_options(bench_bip352 PRIVATE -fno-lto)

# ===== GPU CT Layer Smoke Test =====
set(_CT_TEST_SOURCES src/test_ct_smoke.cu)
if(_GPU_LANG STREQUAL "HIP")
    set_source_files_properties(${_CT_TEST_SOURCES} PROPERTIES LANGUAGE HIP)
    add_executable(test_ct_smoke ${_CT_TEST_SOURCES})
    target_compile_options(test_ct_smoke PRIVATE
        $<$<COMPILE_LANGUAGE:HIP>:-O3>
        $<$<COMPILE_LANGUAGE:HIP>:-ffast-math>
    )
else()
    add_executable(test_ct_smoke ${_CT_TEST_SOURCES})
    target_compile_options(test_ct_smoke PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
        $<$<COMPILE_LANGUAGE:CUDA>:-O3>
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-O3>
    )
endif()
target_link_libraries(test_ct_smoke PRIVATE secp256k1_cuda_lib)
add_test(NAME gpu_ct_smoke COMMAND test_ct_smoke)

# ===== GPU CT Leakage Probe =====
set(_CT_LEAKAGE_SOURCES src/gpu_ct_leakage_probe.cu)
if(_GPU_LANG STREQUAL "HIP")
    set_source_files_properties(${_CT_LEAKAGE_SOURCES} PROPERTIES LANGUAGE HIP)
    add_executable(gpu_ct_leakage_probe ${_CT_LEAKAGE_SOURCES})
    target_compile_options(gpu_ct_leakage_probe PRIVATE
        $<$<COMPILE_LANGUAGE:HIP>:-O3>
        $<$<COMPILE_LANGUAGE:HIP>:-ffast-math>
    )
else()
    add_executable(gpu_ct_leakage_probe ${_CT_LEAKAGE_SOURCES})
    target_compile_options(gpu_ct_leakage_probe PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
        $<$<COMPILE_LANGUAGE:CUDA>:-O3>
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-O3>
    )
endif()
target_link_libraries(gpu_ct_leakage_probe PRIVATE secp256k1_cuda_lib)
add_test(NAME gpu_ct_leakage_probe COMMAND gpu_ct_leakage_probe --json-only)

# ===== GPU ZK Benchmark =====
set(_BENCH_ZK_SOURCES src/bench_zk.cu)
if(_GPU_LANG STREQUAL "HIP")
    set_source_files_properties(${_BENCH_ZK_SOURCES} PROPERTIES LANGUAGE HIP)
    add_executable(bench_zk ${_BENCH_ZK_SOURCES})
    target_compile_options(bench_zk PRIVATE
        $<$<COMPILE_LANGUAGE:HIP>:-O3>
        $<$<COMPILE_LANGUAGE:HIP>:-ffast-math>
    )
else()
    add_executable(bench_zk ${_BENCH_ZK_SOURCES})
    target_compile_options(bench_zk PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
        $<$<COMPILE_LANGUAGE:CUDA>:-O3>
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-O3>
    )
endif()
target_link_libraries(bench_zk PRIVATE secp256k1_cuda_lib)
