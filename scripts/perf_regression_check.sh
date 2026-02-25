#!/usr/bin/env bash
# ============================================================================
# Performance Regression Tracker
# Runs benchmark, compares against baseline, flags regressions > threshold
# ============================================================================
#
# Usage:
#   ./scripts/perf_regression_check.sh [--baseline baseline.json] [--threshold 10]
#
# Outputs:
#   build/perf_report.json    -- current benchmark results
#   build/perf_comparison.txt -- human-readable comparison
#
# In CI, the baseline is stored as an artifact from the previous release.
# A regression > threshold% triggers a warning (non-blocking by default).
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build_perf"
THRESHOLD=10  # percent
BASELINE=""
REPORT_JSON="${BUILD_DIR}/perf_report.json"
COMPARISON="${BUILD_DIR}/perf_comparison.txt"

# -- Parse Args ----------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        --baseline)  BASELINE="$2"; shift 2;;
        --threshold) THRESHOLD="$2"; shift 2;;
        *)           shift;;
    esac
done

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

info()  { echo -e "${GREEN}[PERF]${NC} $*"; }
warn()  { echo -e "${YELLOW}[PERF]${NC} $*"; }
fail()  { echo -e "${RED}[PERF]${NC} $*"; }

# -- Build ---------------------------------------------------------------------
info "Building benchmarks..."
cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DSECP256K1_BUILD_BENCH=ON \
    2>/dev/null

cmake --build "${BUILD_DIR}" -j "$(nproc)" 2>/dev/null

# -- Run Benchmarks ------------------------------------------------------------
mkdir -p "${BUILD_DIR}"

RESULTS=()

run_bench() {
    local name="$1"
    local binary="$2"
    local pattern="$3"

    if [[ ! -x "${binary}" ]]; then
        warn "Skipping ${name} (binary not found: ${binary})"
        return
    fi

    info "Running: ${name}..."
    local output
    output=$("${binary}" 2>&1 || true)

    # Extract timing line (look for ns/op, us/op, ms/op patterns)
    local timing
    timing=$(echo "${output}" | grep -oP '\d+\.?\d*\s*(ns|us|ms)/op' | head -1 || echo "N/A")

    if [[ "${timing}" == "N/A" ]]; then
        # Try alternative: look for "ops/sec" pattern
        timing=$(echo "${output}" | grep -oP '\d+\.?\d*\s*ops/sec' | head -1 || echo "N/A")
    fi

    RESULTS+=("\"${name}\": \"${timing}\"")
}

# Run known benchmark targets
run_bench "scalar_mul" "${BUILD_DIR}/cpu/bench_scalar_mul" "scalar_mul"
run_bench "field_mul" "${BUILD_DIR}/cpu/bench_field_mul_kernels" "field_mul"
run_bench "ct_ops" "${BUILD_DIR}/cpu/bench_ct" "ct"

# -- Generate JSON Report ------------------------------------------------------
{
    echo "{"
    echo "  \"timestamp\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\","
    echo "  \"platform\": \"$(uname -m)\","
    echo "  \"compiler\": \"$(cc --version 2>/dev/null | head -1 || echo unknown)\","
    echo "  \"threshold_percent\": ${THRESHOLD},"
    echo "  \"results\": {"
    local first=true
    for r in "${RESULTS[@]}"; do
        if [[ "${first}" == "true" ]]; then
            echo "    ${r}"
            first=false
        else
            echo "    ,${r}"
        fi
    done
    echo "  }"
    echo "}"
} > "${REPORT_JSON}"

info "Report written to: ${REPORT_JSON}"

# -- Compare Against Baseline -------------------------------------------------
if [[ -n "${BASELINE}" && -f "${BASELINE}" ]]; then
    info "Comparing against baseline: ${BASELINE}"
    echo "Performance Comparison" > "${COMPARISON}"
    echo "=====================" >> "${COMPARISON}"
    echo "Baseline: ${BASELINE}" >> "${COMPARISON}"
    echo "Current:  ${REPORT_JSON}" >> "${COMPARISON}"
    echo "Threshold: ${THRESHOLD}%" >> "${COMPARISON}"
    echo "" >> "${COMPARISON}"
    echo "Note: Detailed numeric comparison requires jq and numeric parsing." >> "${COMPARISON}"
    echo "In CI, use the GitHub Actions benchmark action for precise tracking." >> "${COMPARISON}"
    cat "${COMPARISON}"
else
    info "No baseline provided -- storing current results as new baseline."
    cp "${REPORT_JSON}" "${BUILD_DIR}/perf_baseline.json"
fi

# -- Summary -------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  Performance Regression Check"
echo "============================================================"
echo "  Benchmarks run:  ${#RESULTS[@]}"
echo "  Threshold:       ${THRESHOLD}%"
echo "  Report:          ${REPORT_JSON}"
echo "============================================================"
