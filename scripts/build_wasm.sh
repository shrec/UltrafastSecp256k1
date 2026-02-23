#!/usr/bin/env bash
# ============================================================================
# Build UltrafastSecp256k1 WASM module via Emscripten
# ============================================================================
# Prerequisites: emsdk installed and activated (source emsdk_env.sh)
#
# Usage:
#   ./scripts/build_wasm.sh            # Release build
#   ./scripts/build_wasm.sh debug      # Debug build with assertions
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

BUILD_TYPE="${1:-Release}"
BUILD_DIR="$PROJECT_ROOT/build/wasm"

echo "╔══════════════════════════════════════════════════════╗"
echo "║  UltrafastSecp256k1 — WebAssembly Build             ║"
echo "╠══════════════════════════════════════════════════════╣"
echo "║  Build type: ${BUILD_TYPE}"
echo "║  Output:     ${BUILD_DIR}/dist/"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

# Verify emcc is available
if ! command -v emcc &> /dev/null; then
    echo "ERROR: emcc not found. Install Emscripten SDK:"
    echo "  git clone https://github.com/emscripten-core/emsdk.git"
    echo "  cd emsdk && ./emsdk install latest && ./emsdk activate latest"
    echo "  source emsdk_env.sh"
    exit 1
fi

echo "Emscripten: $(emcc --version | head -1)"

# Configure
emcmake cmake -S "$PROJECT_ROOT/wasm" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -G Ninja

# Build
cmake --build "$BUILD_DIR" -j"$(nproc 2>/dev/null || sysctl -n hw.logicalcpu 2>/dev/null || echo 4)"

echo ""
echo "✅ Build complete!"
echo ""
echo "Output files:"
ls -lh "$BUILD_DIR/dist/"
echo ""

WASM_SIZE=$(du -h "$BUILD_DIR/dist/secp256k1_wasm.wasm" | cut -f1)
echo "WASM binary size: ${WASM_SIZE}"
echo ""
echo "Usage (Node.js):"
echo "  import { Secp256k1 } from '${BUILD_DIR}/dist/secp256k1.mjs';"
echo "  const lib = await Secp256k1.create();"
echo "  console.log(lib.version());"
