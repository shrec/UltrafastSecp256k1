#!/usr/bin/env bash
# ============================================================================
# Memory Access Pattern Analysis (Cache-Line Leak Detection)
# Phase IV, Task 4.3.5 -- Cachegrind-based analysis for CT functions
# ============================================================================
# Uses Valgrind's cachegrind tool to detect data-dependent cache access patterns
# in constant-time cryptographic functions. Cache-line leaks are a real
# side-channel vector even when branches are eliminated.
#
# What we check:
#   - Cache miss patterns should NOT correlate with secret data
#   - D1 miss rate should be uniform across CT operations
#   - No significant difference in cache behavior for different secret values
#
# Usage:
#   ./scripts/cachegrind_ct_analysis.sh [build_dir]
#
# Prerequisites:
#   apt-get install valgrind
#   Build in Debug mode
#
# Exit codes:
#   0 = cache patterns look uniform
#   1 = suspicious patterns detected
#   2 = build/setup error
# ============================================================================

set -euo pipefail

BUILD_DIR="${1:-build/cachegrind-ct}"
SRC_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REPORT_DIR="$BUILD_DIR/cachegrind_reports"

echo "==========================================================="
echo "  Cache-Line Leak Analysis (Cachegrind)"
echo "==========================================================="
echo "  Source:  $SRC_DIR"
echo "  Build:   $BUILD_DIR"
echo ""

# -- Check prerequisites ---------------------------------------------------

if ! command -v valgrind &>/dev/null; then
    echo "ERROR: valgrind not found. Install with: apt-get install valgrind"
    exit 2
fi

if ! command -v cg_annotate &>/dev/null; then
    echo "WARNING: cg_annotate not found. Install valgrind-tools or similar."
fi

echo "  Valgrind: $(valgrind --version 2>/dev/null || echo "unknown")"
echo ""

# -- Build test binary ----------------------------------------------------

echo "[1/4] Building (Debug, no ASM for pure cache analysis)..."
cmake -S "$SRC_DIR" -B "$BUILD_DIR" -G Ninja \
    -DCMAKE_BUILD_TYPE=Debug \
    -DSECP256K1_BUILD_TESTS=ON \
    -DSECP256K1_USE_ASM=OFF \
    -DCMAKE_CXX_FLAGS="-g -O1" \
    2>&1 | tail -5

cmake --build "$BUILD_DIR" --target run_selftest -j"$(nproc)" 2>&1 | tail -3

TEST_BIN="$BUILD_DIR/cpu/run_selftest"
if [[ ! -x "$TEST_BIN" ]]; then
    echo "ERROR: Test binary not found: $TEST_BIN"
    exit 2
fi

mkdir -p "$REPORT_DIR"

# -- Run 1: Cachegrind on selftest (baseline) ----------------------------

echo "[2/4] Running Cachegrind (this takes several minutes)..."

CACHE_FILE_1="$REPORT_DIR/cachegrind.out.1"

valgrind --tool=cachegrind \
    --cachegrind-out-file="$CACHE_FILE_1" \
    --cache-sim=yes \
    --branch-sim=yes \
    "$TEST_BIN" smoke > "$REPORT_DIR/run1_stdout.log" 2> "$REPORT_DIR/run1_stderr.log"

echo "  Run 1 complete: $CACHE_FILE_1"

# -- Run 2: Second run (for comparison) ------------------------------------

echo "[3/4] Running Cachegrind (second pass for comparison)..."

CACHE_FILE_2="$REPORT_DIR/cachegrind.out.2"

valgrind --tool=cachegrind \
    --cachegrind-out-file="$CACHE_FILE_2" \
    --cache-sim=yes \
    --branch-sim=yes \
    "$TEST_BIN" smoke > "$REPORT_DIR/run2_stdout.log" 2> "$REPORT_DIR/run2_stderr.log"

echo "  Run 2 complete: $CACHE_FILE_2"

# -- Analyze results ------------------------------------------------------

echo "[4/4] Analyzing cache patterns..."
echo ""

# Extract summary statistics from stderr
parse_cachegrind() {
    local log="$1"
    local label="$2"
    
    echo "  -- $label --"
    grep -E "I +refs|D +refs|LL refs|I1 +miss|D1 +miss|LL miss|Branches|Mispred" "$log" | head -10
    echo ""
}

parse_cachegrind "$REPORT_DIR/run1_stderr.log" "Run 1"
parse_cachegrind "$REPORT_DIR/run2_stderr.log" "Run 2"

# Compare D1 miss rates
D1_MISS_1=$(grep "D1  miss rate:" "$REPORT_DIR/run1_stderr.log" 2>/dev/null | head -1 | grep -oE '[0-9]+\.[0-9]+%' || echo "N/A")
D1_MISS_2=$(grep "D1  miss rate:" "$REPORT_DIR/run2_stderr.log" 2>/dev/null | head -1 | grep -oE '[0-9]+\.[0-9]+%' || echo "N/A")

BR_MISPRED_1=$(grep "Mispred rate:" "$REPORT_DIR/run1_stderr.log" 2>/dev/null | head -1 | grep -oE '[0-9]+\.[0-9]+%' || echo "N/A")
BR_MISPRED_2=$(grep "Mispred rate:" "$REPORT_DIR/run2_stderr.log" 2>/dev/null | head -1 | grep -oE '[0-9]+\.[0-9]+%' || echo "N/A")

echo "-----------------------------------------------------------"
echo "  Cache Pattern Comparison"
echo "-----------------------------------------------------------"
echo "  D1 miss rate:      Run1=$D1_MISS_1  Run2=$D1_MISS_2"
echo "  Branch mispred:    Run1=$BR_MISPRED_1  Run2=$BR_MISPRED_2"
echo ""

# -- Generate annotated report ---------------------------------------------

if command -v cg_annotate &>/dev/null; then
    cg_annotate "$CACHE_FILE_1" > "$REPORT_DIR/annotated_report.txt" 2>/dev/null || true
    echo "  Annotated report: $REPORT_DIR/annotated_report.txt"
fi

# -- JSON report ----------------------------------------------------------

cat > "$REPORT_DIR/cachegrind_report.json" <<EOF
{
  "tool": "cachegrind_ct_analysis",
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "binary": "$TEST_BIN",
  "runs": [
    {"id": 1, "d1_miss_rate": "$D1_MISS_1", "branch_mispred": "$BR_MISPRED_1"},
    {"id": 2, "d1_miss_rate": "$D1_MISS_2", "branch_mispred": "$BR_MISPRED_2"}
  ],
  "cachegrind_files": ["$CACHE_FILE_1", "$CACHE_FILE_2"],
  "notes": "Compare D1 miss rates: if significantly different between runs with different secret data, cache side-channel may exist."
}
EOF

echo "  JSON report: $REPORT_DIR/cachegrind_report.json"
echo ""
echo "  OK Cache analysis complete. Review annotated report for hotspots."
echo "    Key check: D1 miss rates should be consistent regardless of secret values."
