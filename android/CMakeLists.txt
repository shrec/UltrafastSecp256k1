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
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/../VERSION.txt" _version_raw)
string(STRIP "${_version_raw}" UFSECP_VERSION)

project(UltrafastSecp256k1Android
    VERSION ${UFSECP_VERSION}
    LANGUAGES CXX
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
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../cpu ${CMAKE_CURRENT_BINARY_DIR}/cpu)

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
    ${CMAKE_CURRENT_SOURCE_DIR}/../cpu/include
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
    ${CMAKE_CURRENT_SOURCE_DIR}/../cpu/include
)
target_compile_features(android_test PRIVATE cxx_std_20)
target_compile_options(android_test PRIVATE -O3)
# Static link for standalone execution on device
target_link_options(android_test PRIVATE -static-libstdc++)

# ============================================================================
# Comprehensive benchmark executable (same as desktop, for cross-platform comparison)
# ============================================================================
add_executable(bench_comprehensive
    ${CMAKE_CURRENT_SOURCE_DIR}/../cpu/bench/bench_comprehensive_riscv.cpp
)
target_link_libraries(bench_comprehensive PRIVATE fastsecp256k1)
target_include_directories(bench_comprehensive PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../cpu/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)
target_compile_features(bench_comprehensive PRIVATE cxx_std_20)
target_compile_options(bench_comprehensive PRIVATE -O3)
target_link_options(bench_comprehensive PRIVATE -static-libstdc++)

message(STATUS "Targets available:")
message(STATUS "  fastsecp256k1      - Static library (link into your native code)")
message(STATUS "  secp256k1_jni      - Shared JNI library (for Java/Kotlin apps)")
message(STATUS "  android_test       - On-device test binary (adb push + adb shell)")
message(STATUS "  bench_comprehensive - Cross-platform benchmark (adb push + adb shell)")
