# =============================================================================
# UltrafastSecp256k1 OpenCL Implementation
# =============================================================================
# Cross-platform GPU acceleration for Intel, AMD, NVIDIA GPUs
# Zero external dependencies - uses only OpenCL headers
# =============================================================================

cmake_minimum_required(VERSION 3.16)

# Always define project for standalone builds
project(secp256k1_opencl
    VERSION 1.0.0
    DESCRIPTION "secp256k1 OpenCL GPU acceleration"
    LANGUAGES C CXX
)

# Option to enable/disable OpenCL
option(SECP256K1_USE_OPENCL "Enable OpenCL GPU acceleration" ON)

if(NOT SECP256K1_USE_OPENCL)
    message(STATUS "OpenCL support disabled")
    return()
endif()

# Find OpenCL
find_package(OpenCL QUIET)

if(NOT OpenCL_FOUND)
    # Try to find OpenCL manually on Windows
    if(WIN32)
        # Intel OpenCL SDK paths
        set(INTEL_OCL_PATHS
            "$ENV{INTELOCLSDKROOT}"
            "$ENV{PROGRAMFILES}/Intel/OpenCL SDK"
            "$ENV{PROGRAMFILES\(x86\)}/Intel/OpenCL SDK"
            "C:/Program Files/Intel/OpenCL SDK"
            "C:/Program Files (x86)/Intel/OpenCL SDK"
        )

        foreach(PATH ${INTEL_OCL_PATHS})
            if(EXISTS "${PATH}/include/CL/cl.h")
                set(OpenCL_INCLUDE_DIR "${PATH}/include")
                if(CMAKE_SIZEOF_VOID_P EQUAL 8)
                    set(OpenCL_LIBRARY "${PATH}/lib/x64/OpenCL.lib")
                else()
                    set(OpenCL_LIBRARY "${PATH}/lib/x86/OpenCL.lib")
                endif()
                if(EXISTS "${OpenCL_LIBRARY}")
                    set(OpenCL_FOUND TRUE)
                    break()
                endif()
            endif()
        endforeach()

        # Fallback: System OpenCL.lib (usually works with Intel drivers)
        if(NOT OpenCL_FOUND)
            find_library(OpenCL_LIBRARY OpenCL
                PATHS
                    "$ENV{SYSTEMROOT}/System32"
                    "C:/Windows/System32"
            )
            if(OpenCL_LIBRARY)
                # Use bundled OpenCL headers
                set(OpenCL_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include/external")
                set(OpenCL_FOUND TRUE)
            endif()
        endif()
    endif()

    if(NOT OpenCL_FOUND)
        message(WARNING "OpenCL not found - OpenCL acceleration disabled")
        message(STATUS "  Install Intel OpenCL SDK or use system OpenCL")
        return()
    endif()
endif()

message(STATUS "OpenCL found:")
message(STATUS "  Include: ${OpenCL_INCLUDE_DIR}")
message(STATUS "  Library: ${OpenCL_LIBRARY}")

# =============================================================================
# OpenCL Secp256k1 Library
# =============================================================================

add_library(secp256k1_opencl STATIC
    src/opencl_context.cpp
    src/opencl_field.cpp
    src/opencl_point.cpp
    src/opencl_batch.cpp
    src/opencl_selftest.cpp
)

set_target_properties(secp256k1_opencl PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(secp256k1_opencl
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${OpenCL_INCLUDE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../cpu/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

target_link_libraries(secp256k1_opencl
    PRIVATE
        ${OpenCL_LIBRARY}
)

# Compiler flags (global -Wall/-Wextra/-Wpedantic inherited from root CMakeLists.txt)
# Only add per-target overrides that differ from global policy.
if(MSVC)
    target_compile_options(secp256k1_opencl PRIVATE
        /wd4100  # unreferenced formal parameter (matches -Wno-unused-parameter)
        /wd4244  # conversion warnings (OpenCL host API types)
    )
endif()

# C++20 standard
target_compile_features(secp256k1_opencl PUBLIC cxx_std_20)

# Platform defines
if(WIN32)
    target_compile_definitions(secp256k1_opencl PRIVATE
        _CRT_SECURE_NO_WARNINGS
        NOMINMAX
        WIN32_LEAN_AND_MEAN
    )
endif()

# OpenCL 1.2+ compatibility (wider driver support than 2.0)
target_compile_definitions(secp256k1_opencl PRIVATE
    CL_TARGET_OPENCL_VERSION=120
    CL_USE_DEPRECATED_OPENCL_1_2_APIS
)

# =============================================================================
# Embed OpenCL Kernels as Strings
# =============================================================================

# Generate header with embedded kernel source
set(KERNEL_FILE_1 "${CMAKE_CURRENT_SOURCE_DIR}/kernels/secp256k1_field.cl")
set(KERNEL_FILE_2 "${CMAKE_CURRENT_SOURCE_DIR}/kernels/secp256k1_point.cl")
set(KERNEL_FILE_3 "${CMAKE_CURRENT_SOURCE_DIR}/kernels/secp256k1_batch.cl")
set(KERNEL_FILE_4 "${CMAKE_CURRENT_SOURCE_DIR}/kernels/secp256k1_affine.cl")
set(KERNEL_FILE_5 "${CMAKE_CURRENT_SOURCE_DIR}/kernels/secp256k1_extended.cl")
set(KERNEL_FILE_6 "${CMAKE_CURRENT_SOURCE_DIR}/kernels/secp256k1_hash160.cl")
set(KERNEL_FILE_7 "${CMAKE_CURRENT_SOURCE_DIR}/kernels/secp256k1_ecdh.cl")
set(KERNEL_FILE_8 "${CMAKE_CURRENT_SOURCE_DIR}/kernels/secp256k1_bip352.cl")
set(KERNEL_FILE_9 "${CMAKE_CURRENT_SOURCE_DIR}/kernels/secp256k1_bip324.cl")

set(KERNEL_FILES_LIST
    ${KERNEL_FILE_1}
    ${KERNEL_FILE_2}
    ${KERNEL_FILE_3}
    ${KERNEL_FILE_4}
    ${KERNEL_FILE_5}
    ${KERNEL_FILE_6}
    ${KERNEL_FILE_7}
    ${KERNEL_FILE_8}
    ${KERNEL_FILE_9}
)

set(KERNEL_HEADER "${CMAKE_CURRENT_BINARY_DIR}/include/secp256k1_kernels_embedded.hpp")

# Convert list to escaped string for passing to script
string(REPLACE ";" "|" KERNEL_FILES_STR "${KERNEL_FILES_LIST}")

add_custom_command(
    OUTPUT ${KERNEL_HEADER}
    COMMAND ${CMAKE_COMMAND}
        -DKERNEL_FILES="${KERNEL_FILES_STR}"
        -DOUTPUT_FILE="${KERNEL_HEADER}"
        -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_kernels.cmake
    DEPENDS ${KERNEL_FILES_LIST}
    COMMENT "Embedding OpenCL kernels into header"
)

add_custom_target(embed_opencl_kernels DEPENDS ${KERNEL_HEADER})
add_dependencies(secp256k1_opencl embed_opencl_kernels)

target_include_directories(secp256k1_opencl PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}/include
)

# =============================================================================
# Benchmark Executable
# =============================================================================

if(SECP256K1_BUILD_BENCH)
    # Check if CPU library is available
    if(TARGET fastsecp256k1)
        set(HAVE_CPU_LIB TRUE)
    else()
        # Try to find it in parent build
        find_library(FASTSECP256K1_LIB fastsecp256k1
            PATHS
                ${CMAKE_BINARY_DIR}/../cpu
                ${CMAKE_CURRENT_SOURCE_DIR}/../cpu/build
                ${CMAKE_CURRENT_SOURCE_DIR}/../cpu/build_rel
        )
        if(FASTSECP256K1_LIB)
            set(HAVE_CPU_LIB TRUE)
        else()
            set(HAVE_CPU_LIB FALSE)
            message(STATUS "CPU library not found - benchmark will use OpenCL only")
        endif()
    endif()

    add_executable(opencl_benchmark
        benchmarks/bench_opencl.cpp
    )

    if(HAVE_CPU_LIB)
        target_link_libraries(opencl_benchmark PRIVATE
            secp256k1_opencl
            $<TARGET_NAME_IF_EXISTS:fastsecp256k1>
            ${FASTSECP256K1_LIB}
        )
        target_compile_definitions(opencl_benchmark PRIVATE HAVE_CPU_LIB=1)
    else()
        target_link_libraries(opencl_benchmark PRIVATE
            secp256k1_opencl
        )
    endif()

    add_executable(opencl_bip352_benchmark
        benchmarks/bench_bip352_opencl.cpp
    )

    target_link_libraries(opencl_bip352_benchmark PRIVATE
        secp256k1_opencl
        $<TARGET_NAME_IF_EXISTS:fastsecp256k1>
        ${FASTSECP256K1_LIB}
    )

    target_compile_definitions(opencl_bip352_benchmark PRIVATE
        SECP256K1_OPENCL_KERNEL_DIR="${CMAKE_CURRENT_SOURCE_DIR}/kernels"
    )
endif()

# =============================================================================
# Test Executable
# =============================================================================

add_executable(opencl_test
    tests/test_opencl.cpp
)

target_link_libraries(opencl_test PRIVATE
    secp256k1_opencl
)

if(HAVE_CPU_LIB AND TARGET fastsecp256k1)
    target_link_libraries(opencl_test PRIVATE fastsecp256k1)
    target_compile_definitions(opencl_test PRIVATE HAVE_CPU_LIB=1)
endif()

# =============================================================================
# Audit Runner Executable
# =============================================================================

add_executable(opencl_audit_runner
    src/opencl_audit_runner.cpp
)

target_link_libraries(opencl_audit_runner PRIVATE
    secp256k1_opencl
    ${OpenCL_LIBRARY}
)

target_include_directories(opencl_audit_runner PRIVATE
    ${OpenCL_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../cpu/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

# Copy kernel files next to the audit runner binary for runtime loading
add_custom_command(TARGET opencl_audit_runner POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:opencl_audit_runner>/kernels
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/kernels/secp256k1_field.cl
        ${CMAKE_CURRENT_SOURCE_DIR}/kernels/secp256k1_point.cl
        ${CMAKE_CURRENT_SOURCE_DIR}/kernels/secp256k1_batch.cl
        ${CMAKE_CURRENT_SOURCE_DIR}/kernels/secp256k1_affine.cl
        ${CMAKE_CURRENT_SOURCE_DIR}/kernels/secp256k1_extended.cl
        ${CMAKE_CURRENT_SOURCE_DIR}/kernels/secp256k1_hash160.cl
        $<TARGET_FILE_DIR:opencl_audit_runner>/kernels/
    COMMENT "Copying OpenCL kernel files for audit runner"
)

# CTest integration
enable_testing()
add_test(NAME opencl_selftest COMMAND opencl_test)
add_test(NAME opencl_audit COMMAND opencl_audit_runner --kernel-dir ${CMAKE_CURRENT_SOURCE_DIR}/kernels)

# =============================================================================
# Installation
# =============================================================================

install(TARGETS secp256k1_opencl
    EXPORT secp256k1_opencl-targets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.hpp" PATTERN "*.h"
)

install(DIRECTORY kernels/
    DESTINATION share/secp256k1/opencl
    FILES_MATCHING PATTERN "*.cl"
)
