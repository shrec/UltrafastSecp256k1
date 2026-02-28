#!/usr/bin/env bash
# bench/scripts/build.sh -- Configure + build bench_compare (Linux)
# Usage: bash bench/scripts/build.sh [Release|Debug]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_TYPE="${1:-Release}"
BUILD_DIR="$REPO_ROOT/build/bench-compare"

echo "=== bench_compare build ==="
echo "  Repo root  : $REPO_ROOT"
echo "  Build dir  : $BUILD_DIR"
echo "  Build type : $BUILD_TYPE"
echo ""

# Detect compiler
if command -v clang++-17 >/dev/null 2>&1; then
    export CC=clang-17 CXX=clang++-17
    echo "  Compiler   : clang-17"
elif command -v clang++ >/dev/null 2>&1; then
    export CC=clang CXX=clang++
    echo "  Compiler   : clang (default)"
elif command -v g++-13 >/dev/null 2>&1; then
    export CC=gcc-13 CXX=g++-13
    echo "  Compiler   : gcc-13"
else
    echo "  Compiler   : system default"
fi

echo ""

cmake -S "$REPO_ROOT" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DSECP256K1_BUILD_BENCH_COMPARE=ON \
    -DSECP256K1_BUILD_TESTS=OFF \
    -DSECP256K1_BUILD_EXAMPLES=OFF \
    -DSECP256K1_BUILD_BENCH=OFF

cmake --build "$BUILD_DIR" --target bench_compare -j"$(nproc)"

echo ""
echo "[OK] bench_compare built: $BUILD_DIR/bench/bench_compare"
echo "     Run: $BUILD_DIR/bench/bench_compare --help"
