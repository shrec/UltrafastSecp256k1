#!/bin/bash
# ============================================================================
# Build UltrafastSecp256k1.xcframework for iOS
# ============================================================================
# Requirements: macOS with Xcode 15+ and CMake 3.18+
# Output:       build-xcframework/UltrafastSecp256k1.xcframework
#
# Usage:
#   ./scripts/build_xcframework.sh
#   ./scripts/build_xcframework.sh --release    (default)
#   ./scripts/build_xcframework.sh --debug
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$ROOT_DIR/build-xcframework"
OUTPUT_DIR="$BUILD_DIR/output"
BUILD_TYPE="${1:---release}"

case "$BUILD_TYPE" in
    --debug)  CONFIG="Debug" ;;
    *)        CONFIG="Release" ;;
esac

echo ""
echo "================================================================"
echo "  Building UltrafastSecp256k1.xcframework ($CONFIG)"
echo "================================================================"
echo ""

# Clean previous build
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

# ── 1. Build for iOS Device (arm64) ─────────────────────────────────────────
echo "── [1/3] iOS Device (arm64) ──"
cmake -S "$ROOT_DIR" -B "$BUILD_DIR/ios-device" \
    -G Xcode \
    -DCMAKE_TOOLCHAIN_FILE="$ROOT_DIR/cmake/ios.toolchain.cmake" \
    -DIOS_PLATFORM=OS \
    -DCMAKE_BUILD_TYPE="$CONFIG" \
    2>&1 | tail -5

cmake --build "$BUILD_DIR/ios-device" \
    --config "$CONFIG" \
    --target fastsecp256k1 \
    -- -quiet
echo "  [OK] Device library built"

# ── 2. Build for iOS Simulator (arm64 — Apple Silicon) ──────────────────────
echo ""
echo "── [2/3] iOS Simulator (arm64) ──"
cmake -S "$ROOT_DIR" -B "$BUILD_DIR/ios-simulator" \
    -G Xcode \
    -DCMAKE_TOOLCHAIN_FILE="$ROOT_DIR/cmake/ios.toolchain.cmake" \
    -DIOS_PLATFORM=SIMULATOR \
    -DCMAKE_BUILD_TYPE="$CONFIG" \
    2>&1 | tail -5

cmake --build "$BUILD_DIR/ios-simulator" \
    --config "$CONFIG" \
    --target fastsecp256k1 \
    -- -quiet
echo "  [OK] Simulator library built"

# ── 3. Create XCFramework ───────────────────────────────────────────────────
echo ""
echo "── [3/3] Creating XCFramework ──"
mkdir -p "$OUTPUT_DIR"

# Locate built static libraries
DEVICE_LIB=$(find "$BUILD_DIR/ios-device" -name "libfastsecp256k1.a" -path "*/$CONFIG*" | head -1)
SIM_LIB=$(find "$BUILD_DIR/ios-simulator" -name "libfastsecp256k1.a" -path "*/$CONFIG*" | head -1)

if [ -z "$DEVICE_LIB" ]; then
    echo "ERROR: Device library not found"
    exit 1
fi
if [ -z "$SIM_LIB" ]; then
    echo "ERROR: Simulator library not found"
    exit 1
fi

echo "  Device:    $DEVICE_LIB"
echo "  Simulator: $SIM_LIB"

# Prepare public headers for xcframework
HEADERS_DIR="$BUILD_DIR/headers"
mkdir -p "$HEADERS_DIR"
cp -R "$ROOT_DIR/cpu/include/" "$HEADERS_DIR/"

# Also copy shared types header
mkdir -p "$HEADERS_DIR/secp256k1"
if [ -f "$ROOT_DIR/include/secp256k1/types.hpp" ]; then
    cp "$ROOT_DIR/include/secp256k1/types.hpp" "$HEADERS_DIR/secp256k1/"
fi

# Copy generated version header (from either build)
for D in "$BUILD_DIR/ios-device" "$BUILD_DIR/ios-simulator"; do
    VER=$(find "$D" -name "version.hpp" -path "*/secp256k1/*" 2>/dev/null | head -1)
    if [ -n "$VER" ]; then
        cp "$VER" "$HEADERS_DIR/secp256k1/"
        break
    fi
done

# Create .xcframework
xcodebuild -create-xcframework \
    -library "$DEVICE_LIB" -headers "$HEADERS_DIR" \
    -library "$SIM_LIB"    -headers "$HEADERS_DIR" \
    -output "$OUTPUT_DIR/UltrafastSecp256k1.xcframework"

echo ""
echo "================================================================"
echo "  XCFramework built successfully"
echo "================================================================"
echo "  Output: $OUTPUT_DIR/UltrafastSecp256k1.xcframework"
echo ""
du -sh "$OUTPUT_DIR/UltrafastSecp256k1.xcframework"
echo ""
echo "  Integration:"
echo "    Xcode: Drag .xcframework into project"
echo "    SPM:   Use Package.swift (recommended)"
echo ""
