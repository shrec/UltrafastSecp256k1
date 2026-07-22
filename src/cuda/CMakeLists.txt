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
    if(NOT DEFINED CMAKE_CUDA_RUNTIME_LIBRARY OR CMAKE_CUDA_RUNTIME_LIBRARY STREQUAL "")
        set(CMAKE_CUDA_RUNTIME_LIBRARY Static CACHE STRING
            "CUDA runtime linkage (Static by default so engine binaries do not depend on libcudart.so/cudart64*.dll)"
            FORCE)
    endif()
    # CMAKE_CUDA_ARCHITECTURES is set by the parent CMakeLists.txt before this
    # subdirectory is entered (explicit caller value, CUDAARCHS, or a named
    # SECP256K1_CUDA_ARCH_PROFILE -- see cmake/CudaArchPolicy.cmake). There is
    # intentionally no local fallback here: a fallback at this scope would be
    # a second, redundant place that could silently pick an architecture list,
    # defeating the fail-fast policy enforced at the parent scope.
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "" OR CMAKE_CUDA_ARCHITECTURES STREQUAL "OFF")
        message(FATAL_ERROR
            "src/cuda/CMakeLists.txt: CMAKE_CUDA_ARCHITECTURES is unset. This "
            "subdirectory must only be reached after the parent CMakeLists.txt "
            "has resolved an explicit value, a CUDAARCHS env value, or a named "
            "SECP256K1_CUDA_ARCH_PROFILE. This is a build-graph invariant "
            "violation, not a normal user-facing configuration error.")
    endif()
    # Required for device-code linking when secp256k1_cuda_lib uses separable
    # compilation. Without this, executables that link the static lib get
    # undefined references to __cudaRegisterLinkedBinary_* at link time
    # (CUDA 12+ / Blackwell / Ada and later architectures).
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
endif()
set(CMAKE_CXX_STANDARD 17)

include_directories(include ${CMAKE_CURRENT_SOURCE_DIR}/../../include)

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

# Issue #335 acceptance repair (round 11): these were PUBLIC, unwrapped
# paths ("include" relative, and an absolute parent path) -- CMake refuses
# to export a target whose PUBLIC/INTERFACE include dirs are not
# BUILD_INTERFACE/INSTALL_INTERFACE-wrapped, which blocked
# SECP256K1_INSTALL_CABI=ON + SECP256K1_BUILD_CUDA=ON from configuring at
# all (secp256k1_gpu_host PRIVATE-links this target; ufsecp_static
# PRIVATE-links secp256k1_gpu_host; CMake's install(EXPORT ufsecpTargets ...)
# requires the whole chain to be exportable). Demoted to PRIVATE after
# confirming no consumer needs the PUBLIC propagation:
#   - the sibling test/bench executables defined later in THIS file
#     (secp256k1_cuda_test, gpu_audit_runner, bench_compare, etc.) already
#     get both paths from the directory-scoped `include_directories(include
#     ${CMAKE_CURRENT_SOURCE_DIR}/../../include)` call above (a legacy,
#     non-target-specific command that applies regardless of this target's
#     own PUBLIC/PRIVATE keyword), so nothing in this file depends on the
#     propagation;
#   - src/gpu/CMakeLists.txt (the one out-of-file consumer that PRIVATE-links
#     secp256k1_cuda_lib) already explicitly adds
#     "${CMAKE_CURRENT_SOURCE_DIR}/../cuda/include" itself, PRIVATE, one
#     level up ("CUDA-specific settings" block) -- this PUBLIC copy was
#     redundant for it too.
# PRIVATE also avoids installing ~600KB of internal .cuh device-implementation
# headers (secp256k1.cuh, zk.cuh, etc.) into the public package tree for no
# real consumer -- the stable public GPU surface is ufsecp_gpu.h, installed
# separately by the root CMakeLists.txt `if(TARGET secp256k1_gpu_host)`.
target_include_directories(secp256k1_cuda_lib PRIVATE
    include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

# Join the ufsecp C ABI package's export set (see src/gpu/CMakeLists.txt for
# the full issue #335 round-11 rationale). ARCHIVE-only: this is always a
# STATIC library (HIP and CUDA branches above both add_library(... STATIC)).
include(GNUInstallDirs)
install(TARGETS secp256k1_cuda_lib
    EXPORT ufsecpTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
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

# Issue #335 acceptance repair (round 11 coordinator follow-up): every
# executable named bench_*/secp256k1_cuda_bench below is now gated behind
# SECP256K1_BUILD_BENCH, matching src/opencl/CMakeLists.txt's own existing
# convention (its benchmark executables are already wrapped the same way).
# Previously these built UNCONDITIONALLY whenever CUDA was enabled at all --
# CMake's generated "install" target depends on "all" by default, so ANY
# broken CUDA benchmark (e.g. bench_decompress_ab's pre-existing undefined
# reference to secp256k1::cuda::ecdsa_verify_batch_kernel -- a separate,
# unrelated bug, NOT fixed here, out of this round's scope) blocked
# `cmake --build --target install` even when the caller never asked for
# benchmarks (SECP256K1_BUILD_BENCH=OFF, e.g.
# ci/check_release_package_contents.py --gpu-export-closure's cuda/
# opencl_cuda combos). Test/audit/probe targets in this file
# (secp256k1_cuda_test, test_bip324_kat, gpu_audit_runner, test_ct_smoke,
# gpu_ct_leakage_probe) are deliberately left unconditional, mirroring
# src/opencl/CMakeLists.txt leaving opencl_test/opencl_audit_runner
# unconditional too.
if(SECP256K1_BUILD_BENCH)

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

endif() # SECP256K1_BUILD_BENCH

# CTest integration
enable_testing()
add_test(NAME cuda_selftest COMMAND secp256k1_cuda_test)

# ===== BIP-324 / ChaCha20 GPU Known-Answer Test (RFC 8439) =====
# Tests CUDA kernels at the lowest primitive layer against RFC 8439 vectors.
# Catches GPU-specific bugs (e.g. wrong __byte_perm selector, issue #256)
# that CPU tests cannot detect.
add_executable(test_bip324_kat src/test_bip324_kat.cu)
target_link_libraries(test_bip324_kat PRIVATE secp256k1_cuda_lib)
target_include_directories(test_bip324_kat PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)
set_target_properties(test_bip324_kat PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON)
add_test(NAME bip324_kat_cuda COMMAND test_bip324_kat)
set_tests_properties(bip324_kat_cuda PROPERTIES
    TIMEOUT 60
    LABELS "gpu;cuda;kat;bip324;chacha20;rfc8439")

# ===== Windows macro compatibility compile fixture =====
# Deterministic regression for the rpcndr.h `small` macro collision: compiles
# secp256k1.cuh with the macro pre-defined (see test_windows_macro_compat.cu).
# The compile itself is the test; the executable is a trivial exit.
add_executable(test_windows_macro_compat src/test_windows_macro_compat.cu)
target_include_directories(test_windows_macro_compat PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)
add_test(NAME cuda_windows_macro_compat COMMAND test_windows_macro_compat)
set_tests_properties(cuda_windows_macro_compat PROPERTIES
    TIMEOUT 60
    LABELS "gpu;cuda;compat;windows")

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
    # OBJECT-library mode has no TARGET_FILE — splice the objects in directly
    # (same effect: raw objects, no INTERFACE compile-flag propagation to nvcc).
    if(SECP256K1_BUILD_AS_OBJECT)
        target_sources(gpu_audit_runner PRIVATE $<TARGET_OBJECTS:fastsecp256k1>)
    else()
        target_link_libraries(gpu_audit_runner PRIVATE
            $<TARGET_FILE:fastsecp256k1>
        )
    endif()
    if(TARGET OpenMP::OpenMP_CXX)
        target_link_libraries(gpu_audit_runner PRIVATE OpenMP::OpenMP_CXX)
    elseif(OpenMP_CXX_FLAGS OR OpenMP_CXX_LIB_NAMES)
        target_link_options(gpu_audit_runner PRIVATE ${OpenMP_CXX_FLAGS})
        target_link_libraries(gpu_audit_runner PRIVATE ${OpenMP_CXX_LIB_NAMES})
    endif()
    target_compile_definitions(gpu_audit_runner PRIVATE HAVE_CPU_LIB=1)
    set_target_properties(gpu_audit_runner PROPERTIES INTERPROCEDURAL_OPTIMIZATION FALSE)
    target_link_options(gpu_audit_runner PRIVATE -fno-lto)
endif()
add_test(NAME gpu_audit COMMAND gpu_audit_runner --json-only)

# Issue #335 acceptance repair (round 11 coordinator follow-up): gated
# behind SECP256K1_BUILD_BENCH -- see the comment ahead of
# secp256k1_cuda_bench above for the full rationale.
if(SECP256K1_BUILD_BENCH)

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

# ===== BCH RPA Scanner Benchmark =====
set(_BENCH_BCH_SCAN_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/../bch/cuda/bch_scan.cu)
if(_GPU_LANG STREQUAL "HIP")
    set_source_files_properties(${_BENCH_BCH_SCAN_SOURCES} PROPERTIES LANGUAGE HIP)
    add_executable(bench_bch_scan ${_BENCH_BCH_SCAN_SOURCES})
    target_compile_options(bench_bch_scan PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-O3>)
else()
    add_executable(bench_bch_scan ${_BENCH_BCH_SCAN_SOURCES})
    target_compile_options(bench_bch_scan PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
        $<$<COMPILE_LANGUAGE:CUDA>:-O3>
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-O3>
    )
endif()
target_link_libraries(bench_bch_scan PRIVATE secp256k1_cuda_lib)
target_include_directories(bench_bch_scan PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../cpu/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(bench_bch_scan PRIVATE fastsecp256k1)
set_target_properties(bench_bch_scan PROPERTIES INTERPROCEDURAL_OPTIMIZATION FALSE CUDA_SEPARABLE_COMPILATION OFF)
target_link_options(bench_bch_scan PRIVATE -fno-lto)

# ===== LTC-SP Scanner Benchmark =====
set(_BENCH_LTCSP_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/../ltc/cuda/bench_ltcsp.cu)
if(_GPU_LANG STREQUAL "HIP")
    set_source_files_properties(${_BENCH_LTCSP_SOURCES} PROPERTIES LANGUAGE HIP)
    add_executable(bench_ltcsp ${_BENCH_LTCSP_SOURCES})
    target_compile_options(bench_ltcsp PRIVATE
        $<$<COMPILE_LANGUAGE:HIP>:-O3>
        $<$<COMPILE_LANGUAGE:HIP>:-ffast-math>
    )
else()
    add_executable(bench_ltcsp ${_BENCH_LTCSP_SOURCES})
    target_compile_options(bench_ltcsp PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
        $<$<COMPILE_LANGUAGE:CUDA>:-O3>
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-O3>
    )
endif()
target_link_libraries(bench_ltcsp PRIVATE secp256k1_cuda_lib)
target_include_directories(bench_ltcsp PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../cpu/include
)
target_link_libraries(bench_ltcsp PRIVATE fastsecp256k1)
set_target_properties(bench_ltcsp PROPERTIES
    INTERPROCEDURAL_OPTIMIZATION FALSE
    CUDA_SEPARABLE_COMPILATION OFF
)
target_link_options(bench_ltcsp PRIVATE -fno-lto)

# ===== ECDSA SNARK Witness Benchmark (CPU vs GPU, C ABI) =====
set(_BENCH_SNARK_SOURCES src/bench_snark_witness.cu)
if(_GPU_LANG STREQUAL "HIP")
    set_source_files_properties(${_BENCH_SNARK_SOURCES} PROPERTIES LANGUAGE HIP)
    add_executable(bench_snark_witness ${_BENCH_SNARK_SOURCES})
    target_compile_options(bench_snark_witness PRIVATE
        $<$<COMPILE_LANGUAGE:HIP>:-O3>
        $<$<COMPILE_LANGUAGE:HIP>:-ffast-math>
    )
else()
    add_executable(bench_snark_witness ${_BENCH_SNARK_SOURCES})
    target_compile_options(bench_snark_witness PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
        $<$<COMPILE_LANGUAGE:CUDA>:-O3>
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-O3>
    )
endif()
target_link_libraries(bench_snark_witness PRIVATE secp256k1_cuda_lib)
target_include_directories(bench_snark_witness PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../cpu/include
)
target_link_libraries(bench_snark_witness PRIVATE
    fastsecp256k1
)
set_target_properties(bench_snark_witness PROPERTIES
    INTERPROCEDURAL_OPTIMIZATION FALSE
    CUDA_SEPARABLE_COMPILATION OFF
)
target_link_options(bench_snark_witness PRIVATE -fno-lto)

endif() # SECP256K1_BUILD_BENCH

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

# Issue #335 acceptance repair (round 11 coordinator follow-up): gated
# behind SECP256K1_BUILD_BENCH -- see the comment ahead of
# secp256k1_cuda_bench above for the full rationale. This block in
# particular contains bench_decompress_ab, the target with the pre-existing
# undefined-reference link bug that used to block
# `cmake --build --target install` unconditionally.
if(SECP256K1_BUILD_BENCH)

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

# ===== GPU Pubkey Decompress A/B Benchmark =====
set(_BENCH_DECOMPRESS_AB_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/../../benchmarks/gpu/cuda/decompress_ab_bench.cu)
if(_GPU_LANG STREQUAL "HIP")
    set_source_files_properties(${_BENCH_DECOMPRESS_AB_SOURCES} PROPERTIES LANGUAGE HIP)
    add_executable(bench_decompress_ab ${_BENCH_DECOMPRESS_AB_SOURCES})
    target_compile_options(bench_decompress_ab PRIVATE
        $<$<COMPILE_LANGUAGE:HIP>:-O3>
        $<$<COMPILE_LANGUAGE:HIP>:-ffast-math>
    )
else()
    add_executable(bench_decompress_ab ${_BENCH_DECOMPRESS_AB_SOURCES})
    target_compile_options(bench_decompress_ab PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
        $<$<COMPILE_LANGUAGE:CUDA>:-O3>
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-O3>
    )
endif()
target_link_libraries(bench_decompress_ab PRIVATE secp256k1_cuda_lib)
set_target_properties(bench_decompress_ab PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON)

# ===== GPU Raw Entry ECDSA Verify Benchmark =====
set(_BENCH_RAW_ENTRY_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/../../benchmarks/gpu/cuda/raw_entry_ab_bench.cu)
if(_GPU_LANG STREQUAL "HIP")
    set_source_files_properties(${_BENCH_RAW_ENTRY_SOURCES} PROPERTIES LANGUAGE HIP)
    add_executable(bench_raw_entry ${_BENCH_RAW_ENTRY_SOURCES})
    target_compile_options(bench_raw_entry PRIVATE
        $<$<COMPILE_LANGUAGE:HIP>:-O3>
        $<$<COMPILE_LANGUAGE:HIP>:-ffast-math>
    )
else()
    add_executable(bench_raw_entry ${_BENCH_RAW_ENTRY_SOURCES})
    target_compile_options(bench_raw_entry PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
        $<$<COMPILE_LANGUAGE:CUDA>:-O3>
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-O3>
    )
endif()
target_link_libraries(bench_raw_entry PRIVATE secp256k1_cuda_lib)
set_target_properties(bench_raw_entry PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON)

endif() # SECP256K1_BUILD_BENCH

# Applies to every target in this directory regardless of
# SECP256K1_BUILD_BENCH -- deliberately left outside the gate above so
# secp256k1_cuda_lib and the always-on test/audit targets still get the
# correct CUDA_RUNTIME_LIBRARY when benchmarks are OFF.
if(_GPU_LANG STREQUAL "CUDA")
    get_property(_secp256k1_cuda_targets DIRECTORY PROPERTY BUILDSYSTEM_TARGETS)
    foreach(_secp256k1_cuda_target IN LISTS _secp256k1_cuda_targets)
        set_property(TARGET ${_secp256k1_cuda_target}
            PROPERTY CUDA_RUNTIME_LIBRARY "${CMAKE_CUDA_RUNTIME_LIBRARY}")
    endforeach()
endif()
