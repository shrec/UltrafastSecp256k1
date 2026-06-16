#!/usr/bin/env bash
# =============================================================================
# check_fuzz_smoke.sh — proof-the-fuzzers-still-run smoke gate (heavier; advisory).
# =============================================================================
# The wiring gate (ci/check_fuzz_harness_wiring.py) proves every harness is wired
# into a build — statically. This gate proves the libFuzzer layer is actually ALIVE:
# it builds ONE coverage-guided harness with `clang -fsanitize=fuzzer,address` against
# the current tree and runs it for a bounded number of iterations against the seed
# corpus. If the harness no longer compiles (library API drift) or crashes on the
# seed corpus, the fuzzing layer is broken and this gate goes red.
#
# It is intentionally NOT in run_fast_gates.sh (it builds the static library with a
# sanitizer — minutes, not seconds). Run it from ci_local.sh --full or a dedicated
# CI fuzz-smoke job. ClusterFuzzLite runs the full campaign on its own schedule; this
# is the fast "did we bit-rot the harnesses?" check that runs per-tree.
#
# Exit:  0 = harness built + ran clean ;  77 = advisory skip (no clang / no fuzzer) ;
#        1 = harness failed to build or crashed on the seed corpus.
# =============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
HARNESS="${FUZZ_SMOKE_HARNESS:-fuzz_scalar}"   # minimal-dependency default
RUNS="${FUZZ_SMOKE_RUNS:-20000}"
BUILD_DIR="${ROOT}/out/fuzz-smoke"

CXX="${CXX:-clang++}"
if ! command -v "${CXX}" >/dev/null 2>&1; then
    echo "[fuzz-smoke] ADVISORY SKIP: ${CXX} not found (no clang -> no libFuzzer)"
    exit 77
fi
if ! echo 'int LLVMFuzzerTestOneInput(const unsigned char*x,unsigned long n){return 0;}' \
        | "${CXX}" -x c++ -fsanitize=fuzzer -c -o /dev/null - >/dev/null 2>&1; then
    echo "[fuzz-smoke] ADVISORY SKIP: ${CXX} has no -fsanitize=fuzzer runtime"
    exit 77
fi

HARNESS_SRC="${ROOT}/src/cpu/fuzz/${HARNESS}.cpp"
CORPUS="${ROOT}/src/cpu/fuzz/corpus/${HARNESS}"
if [ ! -f "${HARNESS_SRC}" ]; then
    echo "[fuzz-smoke] FAIL: harness source not found: ${HARNESS_SRC}"
    exit 1
fi

echo "[fuzz-smoke] building libfastsecp256k1 (clang + fuzzer,address) — this is the slow step..."
cmake -S "${ROOT}" -B "${BUILD_DIR}" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER="${CC:-clang}" \
    -DCMAKE_CXX_COMPILER="${CXX}" \
    -DCMAKE_C_FLAGS="-fsanitize=fuzzer-no-link,address" \
    -DCMAKE_CXX_FLAGS="-fsanitize=fuzzer-no-link,address" \
    -DCMAKE_C_FLAGS_RELEASE="-DNDEBUG" \
    -DCMAKE_CXX_FLAGS_RELEASE="-DNDEBUG" \
    -DSECP256K1_BUILD_TESTS=OFF \
    -DSECP256K1_BUILD_BENCH=OFF \
    -DSECP256K1_BUILD_EXAMPLES=OFF \
    -DSECP256K1_BUILD_METAL=OFF \
    -DSECP256K1_USE_ASM=OFF \
    -DSECP256K1_USE_LTO=OFF >/dev/null 2>&1 || {
        echo "[fuzz-smoke] FAIL: cmake configure failed"; exit 1; }

cmake --build "${BUILD_DIR}" -j"$(nproc)" --target fastsecp256k1 >/dev/null 2>&1 || {
    echo "[fuzz-smoke] FAIL: library build failed"; exit 1; }

LIB="${BUILD_DIR}/src/cpu/libfastsecp256k1.a"
INC="${ROOT}/src/cpu/include"
BIN="${BUILD_DIR}/${HARNESS}"

echo "[fuzz-smoke] compiling + linking ${HARNESS} harness..."
"${CXX}" -std=c++20 -O1 -g -fsanitize=fuzzer,address -I "${INC}" \
    "${HARNESS_SRC}" "${LIB}" -o "${BIN}" || {
        echo "[fuzz-smoke] FAIL: ${HARNESS} did not compile/link (library API drift?)"; exit 1; }

echo "[fuzz-smoke] running ${HARNESS} for ${RUNS} iterations..."
# Run against a COPY of the seed corpus in a temp dir so libFuzzer's newly-discovered
# coverage inputs land in the temp dir, not the committed seed corpus (no repo churn).
WORK_CORPUS="$(mktemp -d)"
trap 'rm -rf "${WORK_CORPUS}"' EXIT
if [ -d "${CORPUS}" ]; then cp "${CORPUS}"/* "${WORK_CORPUS}/" 2>/dev/null || true; fi
if "${BIN}" -runs="${RUNS}" -timeout=25 -error_exitcode=99 "${WORK_CORPUS}" >/dev/null 2>&1; then
    echo "[fuzz-smoke] OK: ${HARNESS} built and ran ${RUNS} iterations clean (fuzz layer alive)"
    exit 0
else
    rc=$?
    echo "[fuzz-smoke] FAIL: ${HARNESS} crashed/errored (rc=${rc}) — a finding or a broken harness"
    exit 1
fi
