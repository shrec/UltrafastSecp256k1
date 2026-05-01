#!/bin/bash
# ============================================================================
# PGO (Profile-Guided Optimization) Build Script -- x86_64 / AArch64
# ============================================================================
# Three-phase build:
#   1. Instrument: compile with profiling hooks
#   2. Profile:    run benchmarks/tests to collect hot-path data
#   3. Optimize:   rebuild using collected profiles
#
# Expected improvement: 10-25% on scalar multiplication hot paths.
#
# Requirements: Clang + llvm-profdata (recommended) OR GCC
# Usage:        ./build_pgo.sh [--compiler clang|gcc] [--jobs N]
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build/pgo"
PGO_DIR="${BUILD_DIR}/pgo_profiles"
COMPILER="${1:---compiler}"
JOBS="$(nproc 2>/dev/null || echo 4)"

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --compiler) COMPILER="$2"; shift 2 ;;
        --jobs)     JOBS="$2"; shift 2 ;;
        *)          shift ;;
    esac
done

# Default compiler
if [[ "${COMPILER}" == "--compiler" ]]; then
    if command -v clang++ &>/dev/null; then
        COMPILER="clang"
    else
        COMPILER="gcc"
    fi
fi

case "${COMPILER}" in
    clang)
        CC=clang
        CXX=clang++
        ;;
    gcc)
        CC=gcc
        CXX=g++
        ;;
    *)
        echo "Error: Unknown compiler '${COMPILER}'. Use 'clang' or 'gcc'."
        exit 1
        ;;
esac

echo "=============================================="
echo "  PGO Build -- Phase 1: Instrumentation"
echo "  Compiler: ${CXX}"
echo "=============================================="

rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}" "${PGO_DIR}"

cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER="${CC}" \
    -DCMAKE_CXX_COMPILER="${CXX}" \
    -DBUILD_TESTING=ON \
    -DSECP256K1_USE_PGO_GEN=ON \
    -DSECP256K1_PGO_PROFILE_DIR="${PGO_DIR}" \
    -G Ninja

cmake --build "${BUILD_DIR}" -j"${JOBS}"

echo ""
echo "=============================================="
echo "  PGO Build -- Phase 2: Profiling"
echo "=============================================="

# Run all available tests and benchmarks to exercise hot paths
PROFILED=0

# Run CTest (exercises scalar mul, field ops, signing, verification)
if ctest --test-dir "${BUILD_DIR}" --output-on-failure 2>/dev/null; then
    echo "  [OK] CTest suite completed"
    PROFILED=1
fi

# Run any benchmark executables found
while IFS= read -r bench; do
    echo "  Running: ${bench}"
    "${bench}" || true
    PROFILED=1
done < <(find "${BUILD_DIR}" -name "*bench*" -type f -executable 2>/dev/null)

if [[ ${PROFILED} -eq 0 ]]; then
    echo "  Warning: No tests or benchmarks found. Profile data may be sparse."
fi

echo ""
echo "=============================================="
echo "  PGO Build -- Phase 3: Merge & Optimize"
echo "=============================================="

if [[ "${COMPILER}" == "clang" ]]; then
    PROFRAW_COUNT=$(find "${PGO_DIR}" -name "*.profraw" 2>/dev/null | wc -l)
    if [[ ${PROFRAW_COUNT} -gt 0 ]]; then
        echo "  Merging ${PROFRAW_COUNT} profile(s)..."
        llvm-profdata merge -o "${PGO_DIR}/default.profdata" "${PGO_DIR}"/*.profraw
    else
        echo "  Warning: No .profraw files found in ${PGO_DIR}"
    fi
fi

# Reconfigure with PGO-USE
rm -f "${BUILD_DIR}/CMakeCache.txt"
cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER="${CC}" \
    -DCMAKE_CXX_COMPILER="${CXX}" \
    -DBUILD_TESTING=ON \
    -DSECP256K1_USE_PGO_GEN=OFF \
    -DSECP256K1_USE_PGO_USE=ON \
    -DSECP256K1_USE_LTO=ON \
    -DSECP256K1_PGO_PROFILE_DIR="${PGO_DIR}" \
    -G Ninja

cmake --build "${BUILD_DIR}" -j"${JOBS}"

echo ""
echo "=============================================="
echo "  PGO Build -- Verification"
echo "=============================================="

FAILURES=0
if ctest --test-dir "${BUILD_DIR}" --output-on-failure 2>/dev/null; then
    echo "  [OK] All tests pass with PGO build"
else
    echo "  [WARN] Some tests failed -- check output above"
    FAILURES=1
fi

echo ""
echo "=============================================="
echo "  PGO Build -- Complete!"
echo "=============================================="
echo ""
echo "  Library: ${BUILD_DIR}/libs/UltrafastSecp256k1/cpu/libfastsecp256k1.a"
echo "  Profile: ${PGO_DIR}"
echo ""
echo "  Expected improvements on hot paths:"
echo "    - Scalar multiplication: 10-20%"
echo "    - Point addition:         5-15%"
echo "    - Schnorr/ECDSA sign:    10-15%"
echo ""

exit ${FAILURES}
