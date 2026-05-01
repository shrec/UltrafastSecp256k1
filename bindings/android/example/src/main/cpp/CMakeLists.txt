# ============================================================================
# Example: CMakeLists.txt for Android app integration
# ============================================================================
# Place this at: app/src/main/cpp/CMakeLists.txt
# It wraps the UltrafastSecp256k1 android build.
# ============================================================================

cmake_minimum_required(VERSION 3.18)
project(Secp256k1Example LANGUAGES CXX)

# Path to the UltrafastSecp256k1 android directory
# Adjust this path based on where you placed the library
set(SECP256K1_ANDROID_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../")

# Include the android build (builds both static lib and JNI shared lib)
add_subdirectory(${SECP256K1_ANDROID_DIR} ${CMAKE_CURRENT_BINARY_DIR}/secp256k1)
