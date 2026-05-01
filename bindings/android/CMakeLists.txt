# ============================================================================
# UltrafastSecp256k1 -- Android Native Library Build
# ============================================================================
# Usage (from this directory):
#   cmake -S . -B build-android-arm64 \
#       -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
#       -DANDROID_ABI=arm64-v8a \
#       -DANDROID_PLATFORM=android-24 \
#       -DCMAKE_BUILD_TYPE=Release \
#       -G Ninja
#   cmake --build build-android-arm64 -j
#
# Or use the provided build script: ./build_android.sh
# ============================================================================

cmake_minimum_required(VERSION 3.18)

# Read version from VERSION.txt (single source of truth)
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/../../VERSION.txt" _version_raw)
string(STRIP "${_version_raw}" UFSECP_VERSION)

project(UltrafastSecp256k1Android
    VERSION ${UFSECP_VERSION}
    LANGUAGES C CXX
    DESCRIPTION "UltrafastSecp256k1 for Android"
)

# Verify we're cross-compiling for Android
if(NOT ANDROID)
    message(FATAL_ERROR
        "This CMakeLists.txt is for Android cross-compilation only.\n"
        "Use -DCMAKE_TOOLCHAIN_FILE=\$ANDROID_NDK/build/cmake/android.toolchain.cmake"
    )
endif()

message(STATUS "======================================")
message(STATUS "UltrafastSecp256k1 Android Build")
message(STATUS "  ABI:        ${ANDROID_ABI}")
message(STATUS "  Platform:   ${ANDROID_PLATFORM}")
message(STATUS "  NDK:        ${ANDROID_NDK}")
message(STATUS "  STL:        ${ANDROID_STL}")
message(STATUS "  Compiler:   ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "======================================")

# Include the CPU library (it handles Android detection internally)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../src/cpu ${CMAKE_CURRENT_BINARY_DIR}/cpu)

# ============================================================================
# JNI Shared Library (for Java/Kotlin apps)
# ============================================================================
add_library(secp256k1_jni SHARED
    jni/secp256k1_jni.cpp
)

target_link_libraries(secp256k1_jni PRIVATE
    fastsecp256k1
    log     # Android log library
)

target_include_directories(secp256k1_jni PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/cpu/include
)

target_compile_features(secp256k1_jni PRIVATE cxx_std_20)
target_compile_options(secp256k1_jni PRIVATE -O3 -fvisibility=hidden)

# Only export JNI functions
set_target_properties(secp256k1_jni PROPERTIES
    # Strip debug info in release
    LINK_FLAGS_RELEASE "-s"
    # Library name without "lib" prefix collision
    OUTPUT_NAME "secp256k1_jni"
)

# ============================================================================
# Static library for NDK-only C++ apps (no JNI)
# ============================================================================
# Already available as 'fastsecp256k1' from cpu subdirectory

# ============================================================================
# On-device test executable (push via adb, run in shell)
# ============================================================================
add_executable(android_test test/android_test.cpp)
target_link_libraries(android_test PRIVATE fastsecp256k1)
target_include_directories(android_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/cpu/include
)
target_compile_features(android_test PRIVATE cxx_std_20)
target_compile_options(android_test PRIVATE -O3)
# Static link for standalone execution on device
target_link_options(android_test PRIVATE -static-libstdc++)

# ============================================================================
# On-device focused CPU hotspot benchmarks
# ============================================================================
add_executable(bench_kP ../../src/cpu/tests/bench_kP.cpp)
target_link_libraries(bench_kP PRIVATE fastsecp256k1)
target_include_directories(bench_kP PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/cpu/include
)
target_compile_features(bench_kP PRIVATE cxx_std_20)
target_compile_options(bench_kP PRIVATE -O3)
target_link_options(bench_kP PRIVATE -static-libstdc++)

add_executable(bench_bip324 ../../src/cpu/bench/bench_bip324.cpp)
target_link_libraries(bench_bip324 PRIVATE fastsecp256k1)
target_include_directories(bench_bip324 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/cpu/include
)
target_compile_features(bench_bip324 PRIVATE cxx_std_20)
target_compile_options(bench_bip324 PRIVATE -O3)
target_link_options(bench_bip324 PRIVATE -static-libstdc++)

# ============================================================================
# Bench Hornet (Bitcoin Consensus full benchmark with libsecp256k1 apple-to-apple)
# Requires libsecp256k1 source tree -- skip gracefully when absent (e.g. CI)
# ============================================================================
set(LIBSECP_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../_research_repos/secp256k1/src")

if(EXISTS "${LIBSECP_SRC_DIR}/secp256k1.c")
    add_executable(bench_hornet
        test/bench_hornet_android.cpp
        test/libsecp_bench.c
        ${LIBSECP_SRC_DIR}/precomputed_ecmult.c
        ${LIBSECP_SRC_DIR}/precomputed_ecmult_gen.c
    )
    target_link_libraries(bench_hornet PRIVATE fastsecp256k1)
    target_include_directories(bench_hornet PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../src/cpu/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../_research_repos/secp256k1/include
        ${LIBSECP_SRC_DIR}
    )
    target_compile_features(bench_hornet PRIVATE cxx_std_20)
    target_compile_options(bench_hornet PRIVATE -O3)
    target_link_options(bench_hornet PRIVATE -static-libstdc++)
    set(_bench_hornet_status "bench_hornet       - Bitcoin Consensus bench + libsecp256k1 apple-to-apple")
else()
    message(STATUS "bench_hornet: skipped (libsecp256k1 source not found at ${LIBSECP_SRC_DIR})")
    set(_bench_hornet_status "bench_hornet       - SKIPPED (libsecp256k1 source not found)")
endif()

message(STATUS "Targets available:")
message(STATUS "  fastsecp256k1      - Static library (link into your native code)")
message(STATUS "  secp256k1_jni      - Shared JNI library (for Java/Kotlin apps)")
message(STATUS "  android_test       - On-device test binary (adb push + adb shell)")
message(STATUS "  bench_kP           - On-device BIP-352 k*P hotspot benchmark")
message(STATUS "  bench_bip324       - On-device BIP-324 protocol benchmark")
message(STATUS "  ${_bench_hornet_status}")
