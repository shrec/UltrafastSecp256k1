# ============================================================================
# iOS Cross-Compilation Toolchain for UltrafastSecp256k1
# ============================================================================
#
# Usage (device):
#   cmake -S . -B build-ios \
#     -G Xcode \
#     -DCMAKE_TOOLCHAIN_FILE=cmake/ios.toolchain.cmake
#
# Usage (simulator):
#   cmake -S . -B build-ios-sim \
#     -G Xcode \
#     -DCMAKE_TOOLCHAIN_FILE=cmake/ios.toolchain.cmake \
#     -DIOS_PLATFORM=SIMULATOR
#
# ============================================================================

set(CMAKE_SYSTEM_NAME iOS)
set(CMAKE_OSX_DEPLOYMENT_TARGET "17.0" CACHE STRING "Minimum iOS deployment target")

# Platform selection: OS (device arm64) or SIMULATOR (sim arm64)
if(NOT DEFINED IOS_PLATFORM)
    set(IOS_PLATFORM "OS" CACHE STRING "iOS platform: OS (device) or SIMULATOR")
endif()

if(IOS_PLATFORM STREQUAL "SIMULATOR")
    set(CMAKE_OSX_SYSROOT iphonesimulator CACHE STRING "iOS Simulator SDK")
else()
    set(CMAKE_OSX_SYSROOT iphoneos CACHE STRING "iOS Device SDK")
endif()

# ARM64 only (iOS 17+ dropped 32-bit and x86_64 simulator)
set(CMAKE_OSX_ARCHITECTURES "arm64" CACHE STRING "Target architecture")

# Disable code signing for library builds
set(CMAKE_XCODE_ATTRIBUTE_CODE_SIGNING_ALLOWED "NO")
set(CMAKE_XCODE_ATTRIBUTE_CODE_SIGN_IDENTITY "")

# Build static library only
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

# C++20
set(CMAKE_CXX_STANDARD 20 CACHE STRING "C++ standard")
set(CMAKE_CXX_STANDARD_REQUIRED ON CACHE BOOL "Require C++ standard")

# Disable components not needed for iOS library
set(SECP256K1_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(SECP256K1_BUILD_BENCH OFF CACHE BOOL "" FORCE)
set(SECP256K1_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(SECP256K1_BUILD_CUDA OFF CACHE BOOL "" FORCE)
set(SECP256K1_BUILD_OPENCL OFF CACHE BOOL "" FORCE)
set(SECP256K1_INSTALL OFF CACHE BOOL "" FORCE)

# Enable ARM64 assembly (MUL/UMULH inline asm)
set(SECP256K1_USE_ASM ON CACHE BOOL "" FORCE)
