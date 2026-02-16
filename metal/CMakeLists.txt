cmake_minimum_required(VERSION 3.21)

# ─────────────────────────────────────────────────────────────
# Metal backend for UltrafastSecp256k1
# - Host-side type tests: cross-platform (Windows, Linux, macOS)
# - GPU runtime + kernels: Apple only (M1/M2/M3/M4+)
# ─────────────────────────────────────────────────────────────

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ─────────────────────────────────────────────────────────────
# Cross-Platform Host Test (pure C++, no GPU required)
# Validates types, hex conversion, bridges with types.hpp
# ─────────────────────────────────────────────────────────────

if(SECP256K1_BUILD_TESTS)
    add_executable(metal_host_test
        tests/test_metal_host.cpp
    )

    target_include_directories(metal_host_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
        ${CMAKE_CURRENT_SOURCE_DIR}/..
    )

    target_compile_features(metal_host_test PRIVATE cxx_std_20)

    enable_testing()
    add_test(NAME metal_host_test COMMAND metal_host_test)
endif()

# ─────────────────────────────────────────────────────────────
# GPU backend (Apple only from this point)
# ─────────────────────────────────────────────────────────────

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

# ─────────────────────────────────────────────────────────────
# Metal Runtime Library (Objective-C++)
# ─────────────────────────────────────────────────────────────

add_library(secp256k1_metal_lib STATIC
    src/metal_runtime.mm
)

target_include_directories(secp256k1_metal_lib PUBLIC
    include
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

target_link_libraries(secp256k1_metal_lib PRIVATE
    ${METAL_FRAMEWORK}
    ${FOUNDATION_FRAMEWORK}
)

# Enable ARC for Objective-C++
target_compile_options(secp256k1_metal_lib PRIVATE
    $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>
    $<$<COMPILE_LANGUAGE:OBJCXX>:-fmodules>
)

# ─────────────────────────────────────────────────────────────
# Test/Benchmark Application
# ─────────────────────────────────────────────────────────────

add_executable(metal_secp256k1_test
    app/metal_test.mm
)

target_link_libraries(metal_secp256k1_test PRIVATE
    secp256k1_metal_lib
    ${METAL_FRAMEWORK}
    ${FOUNDATION_FRAMEWORK}
)

target_compile_options(metal_secp256k1_test PRIVATE
    $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>
)

# ─────────────────────────────────────────────────────────────
# Comprehensive Benchmark (matching CUDA benchmark format)
# ─────────────────────────────────────────────────────────────

add_executable(metal_secp256k1_bench_full
    app/bench_metal.mm
)

target_link_libraries(metal_secp256k1_bench_full PRIVATE
    secp256k1_metal_lib
    ${METAL_FRAMEWORK}
    ${FOUNDATION_FRAMEWORK}
)

target_compile_options(metal_secp256k1_bench_full PRIVATE
    $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>
)

# ─────────────────────────────────────────────────────────────
# Compile Metal Shaders → .metallib
# ─────────────────────────────────────────────────────────────

# Find Metal compiler (xcrun metal)
find_program(METAL_COMPILER xcrun)

if(METAL_COMPILER)
    set(SHADER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)
    set(METAL_SOURCE ${SHADER_DIR}/secp256k1_kernels.metal)
    set(METAL_AIR ${CMAKE_CURRENT_BINARY_DIR}/secp256k1_kernels.air)
    set(METAL_LIB ${CMAKE_CURRENT_BINARY_DIR}/secp256k1_kernels.metallib)

    # Step 1: .metal → .air (intermediate representation)
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
            ${SHADER_DIR}/secp256k1_bloom.h
        COMMENT "[Metal] Compiling secp256k1_kernels.metal → .air"
    )

    # Step 2: .air → .metallib (final library)
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
else()
    message(WARNING "secp256k1-metal: 'xcrun' not found; shaders must be compiled at runtime")
endif()

# ─────────────────────────────────────────────────────────────
# Copy shader source files (for runtime compilation fallback)
# ─────────────────────────────────────────────────────────────

set(SHADER_FILES
    ${SHADER_DIR}/secp256k1_field.h
    ${SHADER_DIR}/secp256k1_point.h
    ${SHADER_DIR}/secp256k1_bloom.h
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

# ─────────────────────────────────────────────────────────────
# CTest integration
# ─────────────────────────────────────────────────────────────

enable_testing()
add_test(NAME secp256k1_metal_test
         COMMAND metal_secp256k1_test)

add_test(NAME secp256k1_metal_bench
         COMMAND metal_secp256k1_test --bench)

add_test(NAME secp256k1_metal_bench_full
         COMMAND metal_secp256k1_bench_full)
