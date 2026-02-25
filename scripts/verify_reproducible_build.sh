#!/usr/bin/env bash
# ===========================================================================
# verify_reproducible_build.sh -- Local reproducible build check
# ===========================================================================
# Runs two clean Release builds and compares library checksums.
# Usage: ./scripts/verify_reproducible_build.sh
# Requires: cmake, ninja, g++ (or set CC/CXX)
# ===========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

CC="${CC:-gcc}"
CXX="${CXX:-g++}"
GENERATOR="${GENERATOR:-Ninja}"

# Force deterministic environment
export SOURCE_DATE_EPOCH=1700000000
export TZ=UTC
export LC_ALL=C

echo "=== Reproducible Build Verification ==="
echo "Source:    ${ROOT_DIR}"
echo "Compiler:  ${CXX}"
echo "Generator: ${GENERATOR}"
echo ""

build_and_hash() {
    local label="$1"
    local build_dir="${ROOT_DIR}/build-repro-${label}"

    rm -rf "${build_dir}"
    cmake -S "${ROOT_DIR}" -B "${build_dir}" -G "${GENERATOR}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER="${CC}" \
        -DCMAKE_CXX_COMPILER="${CXX}" \
        -DSECP256K1_BUILD_TESTS=OFF \
        -DSECP256K1_BUILD_BENCH=OFF \
        -DSECP256K1_BUILD_EXAMPLES=OFF \
        -DSECP256K1_BUILD_SHARED=ON \
        -DSECP256K1_USE_ASM=ON \
        > /dev/null 2>&1

    cmake --build "${build_dir}" -j"$(nproc)" > /dev/null 2>&1

    find "${build_dir}" -maxdepth 3 \( -name '*.a' -o -name '*.so' -o -name '*.so.*' \) \
        ! -path '*/CMakeFiles/*' \
        -exec sha256sum {} \; \
        | sed "s|${build_dir}/|build/|g" \
        | sort
}

echo "[1/2] Building (pass A)..."
HASH_A=$(build_and_hash "A")

echo "[2/2] Building (pass B)..."
HASH_B=$(build_and_hash "B")

echo ""
echo "Build A checksums:"
echo "${HASH_A}"
echo ""
echo "Build B checksums:"
echo "${HASH_B}"
echo ""

if [ "${HASH_A}" = "${HASH_B}" ]; then
    echo "[OK] PASS: Builds are byte-identical (reproducible)"
    rm -rf "${ROOT_DIR}/build-repro-A" "${ROOT_DIR}/build-repro-B"
    exit 0
else
    echo "[FAIL] FAIL: Builds differ"
    diff <(echo "${HASH_A}") <(echo "${HASH_B}") || true
    rm -rf "${ROOT_DIR}/build-repro-A" "${ROOT_DIR}/build-repro-B"
    exit 1
fi
