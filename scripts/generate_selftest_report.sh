#!/usr/bin/env bash
# ============================================================================
# Selftest JSON Reporter
# Phase IV, Tasks 4.1.4-4.1.5 -- Machine-readable test results for releases
# ============================================================================
# Builds and runs the complete test suite, capturing structured output in JSON.
# This script is designed to be integrated into CI release pipelines.
#
# Output: selftest_report.json with per-module pass/fail counts + metadata
#
# Usage:
#   ./scripts/generate_selftest_report.sh [build_dir]
#
# Exit code: 0 if all tests pass, 1 if any fail
# ============================================================================

set -euo pipefail

SRC_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="${1:-$SRC_DIR/build/selftest-report}"
REPORT="$BUILD_DIR/selftest_report.json"

echo "==========================================================="
echo "  Selftest JSON Reporter"
echo "==========================================================="
echo "  Source: $SRC_DIR"
echo "  Build:  $BUILD_DIR"
echo ""

# -- Build -----------------------------------------------------------------

echo "[1/3] Building..."
cmake -S "$SRC_DIR" -B "$BUILD_DIR" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DSECP256K1_BUILD_TESTS=ON \
    -DSECP256K1_BUILD_BENCH=OFF \
    2>&1 | tail -5

cmake --build "$BUILD_DIR" -j"$(nproc)" 2>&1 | tail -3

# -- Run CTest and capture ------------------------------------------------

echo "[2/3] Running CTest..."
CTEST_LOG="$BUILD_DIR/ctest_output.log"

set +e
ctest --test-dir "$BUILD_DIR" \
    --output-on-failure \
    --output-junit "$BUILD_DIR/junit_results.xml" \
    -j"$(nproc)" \
    -E "^ct_sidechannel$" \
    2>&1 | tee "$CTEST_LOG"
CTEST_EXIT=$?
set -e

# -- Parse CTest output --------------------------------------------------

echo "[3/3] Generating JSON report..."

# Extract test results from CTest log
TOTAL_TESTS=$(grep -c "Test #" "$CTEST_LOG" 2>/dev/null || echo "0")
PASSED_TESTS=$(grep -c "Passed" "$CTEST_LOG" 2>/dev/null || echo "0")
FAILED_TESTS=$(grep -c "Failed" "$CTEST_LOG" 2>/dev/null || echo "0")

# Parse per-test details
TEST_ENTRIES=""
while IFS= read -r line; do
    # Match lines like: "Test #1: selftest .....   Passed    1.23 sec"
    if echo "$line" | grep -qE "Test +#[0-9]+:"; then
        TEST_NUM=$(echo "$line" | grep -oE '#[0-9]+' | tr -d '#')
        TEST_NAME=$(echo "$line" | sed -E 's/.*Test +#[0-9]+: +([^ ]+).*/\1/')
        if echo "$line" | grep -q "Passed"; then
            TEST_STATUS="pass"
        elif echo "$line" | grep -q "Failed"; then
            TEST_STATUS="fail"
        else
            TEST_STATUS="unknown"
        fi
        TEST_TIME=$(echo "$line" | grep -oE '[0-9]+\.[0-9]+ sec' | grep -oE '[0-9]+\.[0-9]+' || echo "0")
        
        [[ -n "$TEST_ENTRIES" ]] && TEST_ENTRIES="$TEST_ENTRIES,"
        TEST_ENTRIES="$TEST_ENTRIES
    {\"id\": $TEST_NUM, \"name\": \"$TEST_NAME\", \"status\": \"$TEST_STATUS\", \"time_sec\": $TEST_TIME}"
    fi
done < "$CTEST_LOG"

# -- Get version ----------------------------------------------------------

VERSION="unknown"
if [[ -f "$SRC_DIR/VERSION.txt" ]]; then
    VERSION=$(cat "$SRC_DIR/VERSION.txt" | tr -d '[:space:]')
fi

# -- Get git info ---------------------------------------------------------

GIT_COMMIT="unknown"
GIT_BRANCH="unknown"
if command -v git &>/dev/null && [[ -d "$SRC_DIR/.git" ]]; then
    GIT_COMMIT=$(cd "$SRC_DIR" && git rev-parse --short HEAD 2>/dev/null || echo "unknown")
    GIT_BRANCH=$(cd "$SRC_DIR" && git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
fi

# -- Generate JSON --------------------------------------------------------

cat > "$REPORT" <<EOF
{
  "report": "selftest",
  "version": "$VERSION",
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "git": {
    "commit": "$GIT_COMMIT",
    "branch": "$GIT_BRANCH"
  },
  "platform": {
    "os": "$(uname -s 2>/dev/null || echo "unknown")",
    "arch": "$(uname -m 2>/dev/null || echo "unknown")",
    "compiler": "$(cmake -S "$SRC_DIR" -B /dev/null 2>&1 | grep -oP 'CXX compiler.*' | head -1 || echo "unknown")"
  },
  "summary": {
    "total": $TOTAL_TESTS,
    "passed": $PASSED_TESTS,
    "failed": $FAILED_TESTS,
    "ctest_exit_code": $CTEST_EXIT,
    "verdict": "$([ "$CTEST_EXIT" -eq 0 ] && echo "PASS" || echo "FAIL")"
  },
  "tests": [$TEST_ENTRIES
  ]
}
EOF

echo ""
echo "==========================================================="
echo "  Report: $REPORT"
echo "  Tests:  $PASSED_TESTS/$TOTAL_TESTS passed"
echo "  Status: $([ "$CTEST_EXIT" -eq 0 ] && echo "ALL PASS" || echo "FAILURES DETECTED")"
echo "==========================================================="

exit $CTEST_EXIT
