#!/usr/bin/env bash
# ============================================================================
# Code Coverage Report Generator
# Phase IV, Task 4.6.3 -- LLVM source-based coverage + lcov export
# ============================================================================
# Builds with instrumentation, runs all tests, generates:
#   - HTML coverage report (viewable in browser)
#   - LCOV file (for Codecov upload)
#   - JSON summary (for CI badges)
#
# Usage:
#   ./scripts/generate_coverage.sh [--html] [--json] [--target 95]
#
# Prerequisites:
#   clang-17+ (or any version with -fprofile-instr-generate support)
#   llvm-profdata, llvm-cov
#
# Exit codes:
#   0 = coverage generated (and above threshold if --target specified)
#   1 = coverage below threshold
#   2 = build/tool error
# ============================================================================

set -euo pipefail

SRC_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="$SRC_DIR/build/coverage"
REPORT_DIR="$BUILD_DIR/report"
COVERAGE_TARGET=0
GEN_HTML=false
GEN_JSON=false

# -- Parse args ------------------------------------------------------------

while [[ $# -gt 0 ]]; do
    case "$1" in
        --html)   GEN_HTML=true; shift ;;
        --json)   GEN_JSON=true; shift ;;
        --target) COVERAGE_TARGET="$2"; shift 2 ;;
        *)        echo "Unknown arg: $1"; exit 2 ;;
    esac
done

# -- Detect LLVM tools ----------------------------------------------------

LLVM_VER=""
for v in 21 19 18 17 16 15; do
    if command -v "clang++-$v" &>/dev/null; then
        LLVM_VER="$v"
        break
    fi
done

if [[ -z "$LLVM_VER" ]]; then
    if command -v clang++ &>/dev/null; then
        LLVM_VER=""
        CXX="clang++"
        PROFDATA="llvm-profdata"
        LLVM_COV="llvm-cov"
    else
        echo "ERROR: clang++ not found. Install clang-17+."
        exit 2
    fi
else
    CXX="clang++-$LLVM_VER"
    PROFDATA="llvm-profdata-$LLVM_VER"
    LLVM_COV="llvm-cov-$LLVM_VER"
fi

echo "==========================================================="
echo "  Code Coverage Report"
echo "==========================================================="
echo "  Compiler:  $CXX"
echo "  Profdata:  $PROFDATA"
echo "  llvm-cov:  $LLVM_COV"
echo "  Target:    ${COVERAGE_TARGET}%"
echo ""

# -- Build with coverage instrumentation -----------------------------------

echo "[1/5] Configuring with coverage instrumentation..."
cmake -S "$SRC_DIR" -B "$BUILD_DIR" -G Ninja \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_CXX_COMPILER="$CXX" \
    -DSECP256K1_BUILD_TESTS=ON \
    -DSECP256K1_BUILD_BENCH=OFF \
    -DSECP256K1_USE_ASM=OFF \
    -DCMAKE_C_FLAGS="-fprofile-instr-generate -fcoverage-mapping" \
    -DCMAKE_CXX_FLAGS="-fprofile-instr-generate -fcoverage-mapping" \
    -DCMAKE_EXE_LINKER_FLAGS="-fprofile-instr-generate" \
    2>&1 | tail -5

echo "[2/5] Building..."
cmake --build "$BUILD_DIR" -j"$(nproc)" 2>&1 | tail -5

# -- Run tests -------------------------------------------------------------

echo "[3/5] Running all tests (collecting profiles)..."
export LLVM_PROFILE_FILE="$BUILD_DIR/%p-%m.profraw"
ctest --test-dir "$BUILD_DIR" --output-on-failure -j"$(nproc)" -E "^ct_sidechannel$" || true

# -- Merge profiles --------------------------------------------------------

echo "[4/5] Merging coverage profiles..."
find "$BUILD_DIR" -name '*.profraw' -print0 | xargs -0 "$PROFDATA" merge -sparse -o "$BUILD_DIR/coverage.profdata"

# Find all instrumented object files
OBJECTS=""
for bin in $(find "$BUILD_DIR" -type f -executable); do
    if "$LLVM_COV" show --instr-profile="$BUILD_DIR/coverage.profdata" "$bin" >/dev/null 2>&1; then
        OBJECTS="$OBJECTS -object=$bin"
    fi
done

if [[ -z "$OBJECTS" ]]; then
    echo "ERROR: No instrumented objects found"
    exit 2
fi

# -- Generate reports -----------------------------------------------------

echo "[5/5] Generating reports..."
mkdir -p "$REPORT_DIR"

# LCOV export (always)
# shellcheck disable=SC2086
"$LLVM_COV" export \
    --format=lcov \
    --instr-profile="$BUILD_DIR/coverage.profdata" \
    $OBJECTS \
    --ignore-filename-regex='(tests/|bench/|examples/|/usr/|secp256k1/)' \
    > "$REPORT_DIR/coverage.lcov"

echo "  LCOV: $REPORT_DIR/coverage.lcov"

# Summary report
SUMMARY=$("$LLVM_COV" report \
    --instr-profile="$BUILD_DIR/coverage.profdata" \
    $OBJECTS \
    --ignore-filename-regex='(tests/|bench/|examples/|/usr/|secp256k1/)' 2>/dev/null)

echo ""
echo "$SUMMARY" | tail -10
echo ""

# Extract line coverage percentage
LINE_COV=$(echo "$SUMMARY" | tail -1 | awk '{
    for(i=1;i<=NF;i++) {
        if($i ~ /[0-9]+\.[0-9]+%/) {
            gsub(/%/,"",$i);
            print $i;
            exit;
        }
    }
}')

if [[ -z "$LINE_COV" ]]; then
    LINE_COV="0.0"
fi

echo "  Line coverage: ${LINE_COV}%"

# HTML report (optional)
if $GEN_HTML; then
    # shellcheck disable=SC2086
    "$LLVM_COV" show \
        --format=html \
        --instr-profile="$BUILD_DIR/coverage.profdata" \
        $OBJECTS \
        --ignore-filename-regex='(tests/|bench/|examples/|/usr/|secp256k1/)' \
        --output-dir="$REPORT_DIR/html"
    echo "  HTML: $REPORT_DIR/html/index.html"
fi

# JSON summary (optional)
if $GEN_JSON; then
    # shellcheck disable=SC2086
    "$LLVM_COV" export \
        --format=text \
        --instr-profile="$BUILD_DIR/coverage.profdata" \
        $OBJECTS \
        --ignore-filename-regex='(tests/|bench/|examples/|/usr/|secp256k1/)' \
        --summary-only \
        > "$REPORT_DIR/coverage_summary.json"
    echo "  JSON: $REPORT_DIR/coverage_summary.json"
fi

# -- Threshold check ------------------------------------------------------

echo ""
if [[ "$COVERAGE_TARGET" -gt 0 ]]; then
    # Compare floating point
    PASS=$(awk "BEGIN { print ($LINE_COV >= $COVERAGE_TARGET) ? 1 : 0 }")
    if [[ "$PASS" -eq 1 ]]; then
        echo "  OK Coverage ${LINE_COV}% >= ${COVERAGE_TARGET}% target"
        exit 0
    else
        echo "  X Coverage ${LINE_COV}% < ${COVERAGE_TARGET}% target"
        exit 1
    fi
else
    echo "  OK Coverage report generated (no threshold check)"
    exit 0
fi
