#!/bin/bash
# ============================================================================
# UltrafastSecp256k1 -- Android Build Script
# ============================================================================
# Builds native libraries for all Android ABIs using the Android NDK.
#
# Prerequisites:
#   - Android NDK installed (r25+ recommended)
#   - Set ANDROID_NDK_HOME or ANDROID_NDK environment variable
#   - CMake 3.18+ and Ninja
#
# Usage:
#   ./build_android.sh                    # Build all ABIs (arm64, armv7, x86_64, x86)
#   ./build_android.sh arm64-v8a          # Build ARM64 only
#   ./build_android.sh armeabi-v7a        # Build ARMv7 only
#   ./build_android.sh x86_64             # Build x86_64 (emulator)
#   ANDROID_MIN_SDK=21 ./build_android.sh # Override minimum SDK
# ============================================================================

set -euo pipefail

# Resolve NDK path
NDK="${ANDROID_NDK_HOME:-${ANDROID_NDK:-${ANDROID_HOME:-}/ndk-bundle}}"
if [ ! -d "$NDK" ]; then
    # Try common locations
    for candidate in \
        "$HOME/Android/Sdk/ndk"/* \
        "$HOME/Library/Android/sdk/ndk"/* \
        "/usr/local/lib/android/sdk/ndk"/* \
        "C:/Users/$USER/AppData/Local/Android/Sdk/ndk"/*; do
        if [ -d "$candidate" ] && [ -f "$candidate/build/cmake/android.toolchain.cmake" ]; then
            NDK="$candidate"
            break
        fi
    done
fi

if [ ! -f "$NDK/build/cmake/android.toolchain.cmake" ]; then
    echo "ERROR: Android NDK not found."
    echo "Set ANDROID_NDK_HOME to your NDK installation directory."
    echo "  export ANDROID_NDK_HOME=/path/to/android-ndk-r26c"
    exit 1
fi

echo "Using NDK: $NDK"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MIN_SDK="${ANDROID_MIN_SDK:-24}"
BUILD_TYPE="${BUILD_TYPE:-Release}"

# ABIs to build
if [ $# -ge 1 ]; then
    ABIS=("$@")
else
    ABIS=("arm64-v8a" "armeabi-v7a" "x86_64" "x86")
fi

TOOLCHAIN="$NDK/build/cmake/android.toolchain.cmake"

for ABI in "${ABIS[@]}"; do
    BUILD_DIR="$SCRIPT_DIR/build-android-$ABI"
    echo ""
    echo "======================================"
    echo "Building: $ABI (API $MIN_SDK, $BUILD_TYPE)"
    echo "  Output: $BUILD_DIR"
    echo "======================================"

    cmake -S "$SCRIPT_DIR" -B "$BUILD_DIR" \
        -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN" \
        -DANDROID_ABI="$ABI" \
        -DANDROID_PLATFORM="android-$MIN_SDK" \
        -DANDROID_STL=c++_static \
        -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
        -G Ninja

    cmake --build "$BUILD_DIR" -j "$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)"

    # Report output
    echo ""
    echo "Libraries for $ABI:"
    find "$BUILD_DIR" -name "*.so" -o -name "*.a" | head -10
done

# Create unified output directory (for AAR packaging)
OUTPUT_DIR="$SCRIPT_DIR/output/jniLibs"
echo ""
echo "======================================"
echo "Collecting libraries to: $OUTPUT_DIR"
echo "======================================"

for ABI in "${ABIS[@]}"; do
    BUILD_DIR="$SCRIPT_DIR/build-android-$ABI"
    ABI_OUT="$OUTPUT_DIR/$ABI"
    mkdir -p "$ABI_OUT"

    # Copy JNI shared library
    JNI_SO=$(find "$BUILD_DIR" -name "libsecp256k1_jni.so" -print -quit)
    if [ -n "$JNI_SO" ]; then
        cp "$JNI_SO" "$ABI_OUT/"
        echo "  $ABI: $(du -h "$ABI_OUT/libsecp256k1_jni.so" | cut -f1)"
    else
        echo "  $ABI: WARNING -- libsecp256k1_jni.so not found"
    fi
done

echo ""
echo "Done! Copy output/jniLibs/ into your Android project's app/src/main/ directory."
echo ""
echo "Or use Gradle CMake integration (see ANDROID_BUILD.md)."
