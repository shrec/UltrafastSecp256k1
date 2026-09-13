cmake_minimum_required(VERSION 3.21)

# -------------------------------------------------------------
# Metal backend for UltrafastSecp256k1
# - Host-side type tests: cross-platform (Windows, Linux, macOS)
# - GPU runtime + kernels: Apple only (M1/M2/M3/M4+)
# -------------------------------------------------------------

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# -------------------------------------------------------------
# Cross-Platform Host Test (pure C++, no GPU required)
# Validates types, hex conversion, bridges with types.hpp
# -------------------------------------------------------------

if(SECP256K1_BUILD_TESTS)
    add_executable(metal_host_test
        tests/test_metal_host.cpp
    )

    target_include_directories(metal_host_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include
        ${CMAKE_CURRENT_SOURCE_DIR}/..
    )

    target_compile_features(metal_host_test PRIVATE cxx_std_20)

    enable_testing()
    add_test(NAME metal_host_test COMMAND metal_host_test)
endif()

# -------------------------------------------------------------
# GPU backend (Apple only from this point)
# -------------------------------------------------------------

if(NOT APPLE)
    message(STATUS "secp256k1-metal: Host tests configured; GPU backend skipped (not Apple)")
    return()
endif()

project(Secp256k1Metal LANGUAGES CXX OBJCXX)

# Check for Metal framework
find_library(METAL_FRAMEWORK Metal)
find_library(FOUNDATION_FRAMEWORK Foundation)

if(NOT METAL_FRAMEWORK OR NOT FOUNDATION_FRAMEWORK)
    message(WARNING "secp256k1-metal: Metal or Foundation framework not found, skipping GPU backend")
    return()
endif()

message(STATUS "secp256k1-metal: Building for Apple Metal (Apple Silicon GPU)")

set(CMAKE_OBJCXX_STANDARD 20)

# -------------------------------------------------------------
# Metal Runtime Library (Objective-C++)
# -------------------------------------------------------------

add_library(secp256k1_metal_lib STATIC
    src/metal_runtime.mm
)

# Issue #335 acceptance repair (round 11): structural parity with the
# OpenCL/CUDA export-set closure fix (src/opencl/CMakeLists.txt,
# src/cuda/CMakeLists.txt, src/gpu/CMakeLists.txt) -- NOT runtime-verified,
# METAL_RUNTIME_CONFIRMATION_PENDING (this file only reaches this point
# under APPLE; see the `if(NOT APPLE) ... return()` guard above, so a Linux
# configure/build never exercises any line below this comment). These paths
# were PUBLIC and unwrapped, the same defect class that blocked
# SECP256K1_INSTALL_CABI=ON + SECP256K1_BUILD_OPENCL/CUDA=ON on this host.
# Demoted to PRIVATE: the local consumers in this same file
# (metal_audit_runner) already explicitly add
# "${CMAKE_CURRENT_SOURCE_DIR}/include" themselves, PRIVATE, so nothing here
# depends on the PUBLIC propagation; there is no known out-of-file consumer
# (src/gpu/CMakeLists.txt adds "${CMAKE_CURRENT_SOURCE_DIR}/../metal/include"
# itself, PRIVATE, when SECP256K1_BUILD_METAL links secp256k1_metal_lib).
target_include_directories(secp256k1_metal_lib PRIVATE
    include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

target_link_libraries(secp256k1_metal_lib PRIVATE
    ${METAL_FRAMEWORK}
    ${FOUNDATION_FRAMEWORK}
)

# Join the ufsecp C ABI package's export set (see src/gpu/CMakeLists.txt for
# the full issue #335 round-11 rationale). Configure-level parity only --
# this code path is unreachable on non-Apple hosts (see the `if(NOT APPLE)`
# guard above), so it cannot be exercised by this round's real
# cmake --install verification (Linux-only machine).
include(GNUInstallDirs)
install(TARGETS secp256k1_metal_lib
    EXPORT ufsecpTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

# Enable ARC for Objective-C++
target_compile_options(secp256k1_metal_lib PRIVATE
    $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>
    $<$<COMPILE_LANGUAGE:OBJCXX>:-fmodules>
)

# -------------------------------------------------------------
# Test/Benchmark Application
# -------------------------------------------------------------

add_executable(metal_secp256k1_test
    app/metal_test.mm
)

target_include_directories(metal_secp256k1_test PRIVATE
    include
    ${CMAKE_CURRENT_SOURCE_DIR}/../cpu/include
)

target_link_libraries(metal_secp256k1_test PRIVATE
    secp256k1_metal_lib
    ${METAL_FRAMEWORK}
    ${FOUNDATION_FRAMEWORK}
)

target_compile_options(metal_secp256k1_test PRIVATE
    $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>
)

# -------------------------------------------------------------
# Comprehensive Benchmark (matching CUDA benchmark format)
# -------------------------------------------------------------

add_executable(metal_secp256k1_bench_full
    app/bench_metal.mm
)

target_include_directories(metal_secp256k1_bench_full PRIVATE
    include
    ${CMAKE_CURRENT_SOURCE_DIR}/../cpu/include
)

target_link_libraries(metal_secp256k1_bench_full PRIVATE
    secp256k1_metal_lib
    ${METAL_FRAMEWORK}
    ${FOUNDATION_FRAMEWORK}
)

target_compile_options(metal_secp256k1_bench_full PRIVATE
    $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>
)

# -------------------------------------------------------------
# Metal Unified Audit Runner (27 modules, 8 sections)
# Mirrors OpenCL/CUDA audit runner format
# -------------------------------------------------------------

add_executable(metal_audit_runner
    src/metal_audit_runner.mm
)

target_include_directories(metal_audit_runner PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(metal_audit_runner PRIVATE
    ${METAL_FRAMEWORK}
    ${FOUNDATION_FRAMEWORK}
)

target_compile_options(metal_audit_runner PRIVATE
    $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>
)

target_compile_features(metal_audit_runner PRIVATE cxx_std_20)

# -------------------------------------------------------------
# Compile Metal Shaders -> .metallib
# -------------------------------------------------------------

# Find Metal compiler (xcrun metal)
find_program(METAL_COMPILER xcrun)

# The shader SOURCE directory, which exists whether or not a Metal compiler
# does. It used to be set only inside if(METAL_COMPILER), while the
# SHADER_FILES copy list below sits outside that guard -- so on any build
# without xcrun (a non-Apple host with SECP256K1_BUILD_METAL=ON, or an Apple
# host without the Xcode command-line tools) every entry expanded to a bare
# "/secp256k1_*.h" at the filesystem root.
set(SHADER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)

if(METAL_COMPILER)
    set(METAL_SOURCE ${SHADER_DIR}/secp256k1_kernels.metal)
    set(METAL_AIR ${CMAKE_CURRENT_BINARY_DIR}/secp256k1_kernels.air)
    set(METAL_LIB ${CMAKE_CURRENT_BINARY_DIR}/secp256k1_kernels.metallib)

    # Step 1: .metal -> .air (intermediate representation)
    add_custom_command(
        OUTPUT ${METAL_AIR}
        COMMAND ${METAL_COMPILER} -sdk macosx metal
            -I ${SHADER_DIR}
            -O2
            -std=macos-metal2.4
            -o ${METAL_AIR}
            -c ${METAL_SOURCE}
        DEPENDS
            ${METAL_SOURCE}
            ${SHADER_DIR}/secp256k1_field.h
            ${SHADER_DIR}/secp256k1_point.h
            ${SHADER_DIR}/secp256k1_extended.h
            ${SHADER_DIR}/secp256k1_hash160.h
            ${SHADER_DIR}/secp256k1_affine.h
            ${SHADER_DIR}/secp256k1_zk.h
        COMMENT "[Metal] Compiling secp256k1_kernels.metal -> .air"
    )

    # Step 2: .air -> .metallib (final library)
    add_custom_command(
        OUTPUT ${METAL_LIB}
        COMMAND ${METAL_COMPILER} -sdk macosx metallib
            -o ${METAL_LIB}
            ${METAL_AIR}
        DEPENDS ${METAL_AIR}
        COMMENT "[Metal] Linking secp256k1_kernels.metallib"
    )

    # Custom target for shader compilation
    add_custom_target(metal_shaders ALL DEPENDS ${METAL_LIB})
    add_dependencies(metal_secp256k1_test metal_shaders)
    add_dependencies(metal_secp256k1_bench_full metal_shaders)
    add_dependencies(metal_audit_runner metal_shaders)

    # Copy metallib to test binary directory
    add_custom_command(TARGET metal_secp256k1_test POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${METAL_LIB}
            $<TARGET_FILE_DIR:metal_secp256k1_test>/secp256k1_kernels.metallib
        COMMENT "[Metal] Copying metallib to test binary directory"
    )
    add_custom_command(TARGET metal_secp256k1_bench_full POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${METAL_LIB}
            $<TARGET_FILE_DIR:metal_secp256k1_bench_full>/secp256k1_kernels.metallib
        COMMENT "[Metal] Copying metallib to bench binary directory"
    )
    add_custom_command(TARGET metal_audit_runner POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${METAL_LIB}
            $<TARGET_FILE_DIR:metal_audit_runner>/secp256k1_kernels.metallib
        COMMENT "[Metal] Copying metallib to audit runner directory"
    )

    # The CPU audit binaries -- unified_audit_runner and ~500 standalone CTest
    # targets -- live in ${CMAKE_BINARY_DIR}/audit, and ctest runs them from
    # that directory. The three copies above all land in
    # ${CMAKE_BINARY_DIR}/src/metal (that is where the metal_* targets are), and
    # none of gpu_backend_metal.mm's CWD-relative candidates
    # ("secp256k1_kernels.metallib", "../…", "../metal/…", "../../metal/…")
    # reaches across from audit/ to src/metal/. So every Metal-backed audit
    # module failed with "[Metal] ERROR: Failed to load metallib: library not
    # found" -- on CI / macos (Release) that is gpu_abi_gate, unified_audit and
    # regression_bip352_ct_varbase, all three reporting GPU failures for a
    # missing file rather than for anything about the GPU.
    #
    # Attached to metal_shaders rather than to an audit target because
    # add_custom_command(TARGET) requires the target to be defined in this
    # directory, and audit/ is a sibling added later (CMakeLists.txt:772).
    # make_directory first: this runs whether or not audit/ was configured
    # (it is gated on SECP256K1_BUILD_CPU AND BUILD_TESTING).
    add_custom_command(TARGET metal_shaders POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/audit
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${METAL_LIB}
            ${CMAKE_BINARY_DIR}/audit/secp256k1_kernels.metallib
        COMMENT "[Metal] Copying metallib to the CPU audit binary directory"
    )
else()
    message(WARNING "secp256k1-metal: 'xcrun' not found; shaders must be compiled at runtime")
endif()

# -------------------------------------------------------------
# Copy shader source files (for runtime compilation fallback)
# -------------------------------------------------------------

# The COMPLETE #include closure of secp256k1_kernels.metal, not a subset.
# gpu_backend_metal.mm's runtime-compile fallback expands the entry file's own
# includes, so a copied shaders/ directory missing even one of them yields no
# translation unit at all -- which is what the old four-header list did.
# regression_metal_shader_closure pins this list against the actual includes.
set(SHADER_FILES
    ${SHADER_DIR}/secp256k1_field.h
    ${SHADER_DIR}/secp256k1_point.h
    ${SHADER_DIR}/secp256k1_extended.h
    ${SHADER_DIR}/secp256k1_hash160.h
    ${SHADER_DIR}/secp256k1_zk.h
    ${SHADER_DIR}/secp256k1_bip324.h
    ${SHADER_DIR}/secp256k1_ct_ops.h
    ${SHADER_DIR}/secp256k1_ct_field.h
    ${SHADER_DIR}/secp256k1_ct_scalar.h
    ${SHADER_DIR}/secp256k1_ct_point.h
    ${SHADER_DIR}/secp256k1_ct_sign.h
    ${SHADER_DIR}/secp256k1_kernels.metal
)

add_custom_command(TARGET metal_secp256k1_test POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory
        $<TARGET_FILE_DIR:metal_secp256k1_test>/shaders
)

foreach(SHADER ${SHADER_FILES})
    get_filename_component(SHADER_NAME ${SHADER} NAME)
    add_custom_command(TARGET metal_secp256k1_test POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${SHADER}
            $<TARGET_FILE_DIR:metal_secp256k1_test>/shaders/${SHADER_NAME}
    )
endforeach()

# Also copy shader sources for bench_full (runtime compilation fallback)
add_custom_command(TARGET metal_secp256k1_bench_full POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory
        $<TARGET_FILE_DIR:metal_secp256k1_bench_full>/shaders
)

foreach(SHADER ${SHADER_FILES})
    get_filename_component(SHADER_NAME ${SHADER} NAME)
    add_custom_command(TARGET metal_secp256k1_bench_full POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${SHADER}
            $<TARGET_FILE_DIR:metal_secp256k1_bench_full>/shaders/${SHADER_NAME}
    )
endforeach()

# Also copy shader sources for audit runner (runtime compilation fallback)
add_custom_command(TARGET metal_audit_runner POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory
        $<TARGET_FILE_DIR:metal_audit_runner>/shaders
)

foreach(SHADER ${SHADER_FILES})
    get_filename_component(SHADER_NAME ${SHADER} NAME)
    add_custom_command(TARGET metal_audit_runner POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${SHADER}
            $<TARGET_FILE_DIR:metal_audit_runner>/shaders/${SHADER_NAME}
    )
endforeach()

# -------------------------------------------------------------
# CTest integration
# -------------------------------------------------------------

enable_testing()
add_test(NAME secp256k1_metal_test
         COMMAND metal_secp256k1_test)

add_test(NAME secp256k1_metal_bench
         COMMAND metal_secp256k1_test --bench)

add_test(NAME secp256k1_metal_bench_full
         COMMAND metal_secp256k1_bench_full)

add_test(NAME secp256k1_metal_audit
         COMMAND metal_audit_runner --report-dir ${CMAKE_CURRENT_BINARY_DIR})
