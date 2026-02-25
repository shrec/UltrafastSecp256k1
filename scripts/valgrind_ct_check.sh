#!/usr/bin/env bash
# ============================================================================
# Valgrind Memcheck CT Analysis
# Phase IV, Task 4.3.3 -- Detect secret-dependent branches via uninit tracking
# ============================================================================
# Uses Valgrind's --track-origins=yes to detect control flow that depends on
# uninitialized / "secret-tainted" memory. We mark secret key material as
# undefined, then check if any branches depend on it.
#
# Approach: Build with special VALGRIND_CT_CHECK define that:
#   1. Marks secret scalars as UNDEFINED via VALGRIND_MAKE_MEM_UNDEFINED
#   2. Runs CT operations (scalar_mul, ecdsa_sign, schnorr_sign)
#   3. Valgrind reports any "Conditional jump depends on uninitialised value"
#   4. Zero reports = CT proven at binary level
#
# Usage:
#   ./scripts/valgrind_ct_check.sh [build_dir]
#
# Prerequisites:
#   apt-get install valgrind
#   Build with: -DCMAKE_BUILD_TYPE=Debug (no optimizations strip the checks)
#
# Exit codes:
#   0 = no secret-dependent branches detected
#   1 = potential CT violation found
#   2 = build/setup error
# ============================================================================

set -euo pipefail

BUILD_DIR="${1:-build/valgrind-ct}"
SRC_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REPORT_DIR="$BUILD_DIR/valgrind_reports"
VALGRIND_LOG="$REPORT_DIR/valgrind_ct.log"
VALGRIND_XML="$REPORT_DIR/valgrind_ct.xml"

echo "==========================================================="
echo "  Valgrind CT Analysis"
echo "==========================================================="
echo "  Source:  $SRC_DIR"
echo "  Build:   $BUILD_DIR"
echo "  Reports: $REPORT_DIR"
echo ""

# -- Check prerequisites ---------------------------------------------------

if ! command -v valgrind &>/dev/null; then
    echo "ERROR: valgrind not found. Install with: apt-get install valgrind"
    exit 2
fi

VALGRIND_VERSION=$(valgrind --version 2>/dev/null || echo "unknown")
echo "  Valgrind: $VALGRIND_VERSION"
echo ""

# -- Build test binary with Valgrind CT checks ----------------------------

echo "[1/4] Configuring (Debug + Valgrind CT markers)..."
cmake -S "$SRC_DIR" -B "$BUILD_DIR" -G Ninja \
    -DCMAKE_BUILD_TYPE=Debug \
    -DSECP256K1_BUILD_TESTS=ON \
    -DSECP256K1_USE_ASM=OFF \
    -DCMAKE_CXX_FLAGS="-DVALGRIND_CT_CHECK=1 -g -O0" \
    2>&1 | tail -5

echo "[2/4] Building test binary..."
cmake --build "$BUILD_DIR" --target test_ct_sidechannel_standalone -j"$(nproc)" 2>&1 | tail -3

TEST_BIN="$BUILD_DIR/cpu/test_ct_sidechannel_standalone"
if [[ ! -x "$TEST_BIN" ]]; then
    echo "ERROR: Test binary not found: $TEST_BIN"
    exit 2
fi

# -- Run under Valgrind ----------------------------------------------------

mkdir -p "$REPORT_DIR"

echo "[3/4] Running under Valgrind (this takes several minutes)..."
echo "      Tracking: secret-dependent branches + memory errors"
echo ""

set +e
valgrind \
    --tool=memcheck \
    --track-origins=yes \
    --leak-check=no \
    --show-reachable=no \
    --error-exitcode=42 \
    --xml=yes \
    --xml-file="$VALGRIND_XML" \
    --log-file="$VALGRIND_LOG" \
    --num-callers=20 \
    --suppressions=/dev/null \
    "$TEST_BIN" 2>&1 | tee "$REPORT_DIR/stdout.log"

VG_EXIT=$?
set -e

# -- Analyze results ------------------------------------------------------

echo ""
echo "[4/4] Analyzing Valgrind output..."
echo ""

# Count "Conditional jump or move depends on uninitialised value(s)"
CT_ERRORS=$(grep -c "Conditional jump or move depends on uninitialised" "$VALGRIND_LOG" 2>/dev/null || echo "0")

# Count "Use of uninitialised value of size"
UNINIT_ERRORS=$(grep -c "Use of uninitialised value" "$VALGRIND_LOG" 2>/dev/null || echo "0")

# Count total errors
TOTAL_ERRORS=$(grep -c "ERROR SUMMARY:" "$VALGRIND_LOG" 2>/dev/null || echo "0")
ERROR_SUMMARY=$(grep "ERROR SUMMARY:" "$VALGRIND_LOG" 2>/dev/null | tail -1 || echo "N/A")

echo "-----------------------------------------------------------"
echo "  Valgrind CT Analysis Results"
echo "-----------------------------------------------------------"
echo "  Conditional branch on uninit:  $CT_ERRORS"
echo "  Use of uninit value:           $UNINIT_ERRORS"
echo "  $ERROR_SUMMARY"
echo ""
echo "  Full log:   $VALGRIND_LOG"
echo "  XML report: $VALGRIND_XML"
echo "-----------------------------------------------------------"

# -- Generate JSON report -------------------------------------------------

cat > "$REPORT_DIR/valgrind_ct_report.json" <<EOF
{
  "tool": "valgrind_ct_check",
  "valgrind_version": "$VALGRIND_VERSION",
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "binary": "$TEST_BIN",
  "ct_branch_errors": $CT_ERRORS,
  "uninit_value_errors": $UNINIT_ERRORS,
  "valgrind_exit_code": $VG_EXIT,
  "verdict": "$([ "$CT_ERRORS" -eq 0 ] && echo "PASS" || echo "FAIL")"
}
EOF

echo "  JSON report: $REPORT_DIR/valgrind_ct_report.json"
echo ""

# -- Verdict --------------------------------------------------------------

if [[ "$CT_ERRORS" -eq 0 ]]; then
    echo "  OK No secret-dependent branches detected by Valgrind"
    exit 0
else
    echo "  X POTENTIAL CT VIOLATION: $CT_ERRORS conditional branches on uninit data"
    echo ""
    echo "  Offending locations:"
    grep -A5 "Conditional jump or move depends on uninitialised" "$VALGRIND_LOG" | head -30
    exit 1
fi
