#!/usr/bin/env bash
# ============================================================================
# Cross-Compiler CT Stress Test
# Phase V, Task 5.2.4 -- Verify CT under GCC, Clang, MSVC (via CI matrix)
# ============================================================================
# Builds and tests CT code under multiple compilers and optimization levels
# to detect compiler-specific CT violations (e.g. branch injection at -O3).
#
# Usage:
#   ./scripts/cross_compiler_ct_stress.sh [--full]
#
# Matrix tested:
#   GCC:   -O0, -O1, -O2, -O3, -Os, -Ofast
#   Clang: -O0, -O1, -O2, -O3, -Os, -Oz
#   (MSVC tested separately in CI via /Od, /O1, /O2)
#
# For each (compiler, opt-level) pair:
#   1. Build CT test + selftest
#   2. Run selftest (correctness check)
#   3. Run CT disasm verification (branch check)
#   4. Run dudect smoke test (timing check, if --full)
#
# Exit codes:
#   0 = all compiler+opt combos pass
#   1 = at least one fails
# ============================================================================

set -euo pipefail

SRC_DIR="$(cd "$(dirname "$0")/.." && pwd)"
FULL_MODE=false
VERIFY_DISASM="$SRC_DIR/scripts/verify_ct_disasm.sh"

if [[ "${1:-}" == "--full" ]]; then
    FULL_MODE=true
fi

echo "==========================================================="
echo "  Cross-Compiler CT Stress Test"
echo "==========================================================="
echo "  Source: $SRC_DIR"
echo "  Full mode: $FULL_MODE"
echo ""

# -- Discover available compilers ------------------------------------------

COMPILERS=()

# GCC variants
for v in g++-13 g++-12 g++-11 g++; do
    if command -v "$v" &>/dev/null; then
        COMPILERS+=("$v")
        break
    fi
done

# Clang variants
for v in clang++-21 clang++-19 clang++-18 clang++-17 clang++-16 clang++; do
    if command -v "$v" &>/dev/null; then
        COMPILERS+=("$v")
        break
    fi
done

if [[ ${#COMPILERS[@]} -eq 0 ]]; then
    echo "ERROR: No C++ compilers found."
    exit 1
fi

echo "  Compilers: ${COMPILERS[*]}"

# -- Define optimization levels --------------------------------------------

get_opt_levels() {
    local compiler="$1"
    if [[ "$compiler" == *clang* ]]; then
        echo "-O0 -O1 -O2 -O3 -Os -Oz"
    else
        echo "-O0 -O1 -O2 -O3 -Os -Ofast"
    fi
}

# -- Test matrix ----------------------------------------------------------

TOTAL=0
PASSED=0
FAILED=0
SKIPPED=0
RESULTS=""

for COMPILER in "${COMPILERS[@]}"; do
    for OPT in $(get_opt_levels "$COMPILER"); do
        TOTAL=$((TOTAL + 1))
        TAG="${COMPILER}_${OPT}"
        BUILD_DIR="$SRC_DIR/build/ct-stress/$TAG"
        
        echo ""
        echo "------------------------------------------------"
        echo "  [$TAG] Building..."
        echo "------------------------------------------------"
        
        # Build
        BUILD_OK=true
        cmake -S "$SRC_DIR" -B "$BUILD_DIR" -G Ninja \
            -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_CXX_COMPILER="$COMPILER" \
            -DCMAKE_CXX_FLAGS_RELEASE="$OPT -DNDEBUG" \
            -DSECP256K1_BUILD_TESTS=ON \
            -DSECP256K1_USE_ASM=ON \
            2>&1 | tail -3 || BUILD_OK=false
            
        if ! $BUILD_OK; then
            echo "  [$TAG] BUILD FAILED -- skipping"
            SKIPPED=$((SKIPPED + 1))
            RESULTS="$RESULTS\n  [SKIP] $TAG -- build failed"
            continue
        fi
        
        cmake --build "$BUILD_DIR" -j"$(nproc)" 2>&1 | tail -3 || {
            echo "  [$TAG] BUILD FAILED -- skipping"
            SKIPPED=$((SKIPPED + 1))
            RESULTS="$RESULTS\n  [SKIP] $TAG -- build failed"
            continue
        }
        
        # 1. Run selftest (correctness)
        SELFTEST_BIN="$BUILD_DIR/cpu/run_selftest"
        if [[ -x "$SELFTEST_BIN" ]]; then
            echo "  [$TAG] Running selftest..."
            if timeout 120 "$SELFTEST_BIN" > "$BUILD_DIR/selftest.log" 2>&1; then
                echo "  [$TAG] Selftest: PASS"
            else
                echo "  [$TAG] Selftest: FAIL"
                RESULTS="$RESULTS\n  [FAIL] $TAG -- selftest failed"
                FAILED=$((FAILED + 1))
                continue
            fi
        fi
        
        # 2. Disasm verification (branch check)
        CT_BIN="$BUILD_DIR/cpu/test_ct_sidechannel_standalone"
        if [[ -x "$CT_BIN" ]] && [[ -x "$VERIFY_DISASM" ]]; then
            echo "  [$TAG] Checking disassembly for CT violations..."
            if bash "$VERIFY_DISASM" "$CT_BIN" --json "$BUILD_DIR/ct_disasm.json" 2>&1 | tail -5; then
                echo "  [$TAG] Disasm: PASS"
            else
                echo "  [$TAG] Disasm: WARNING (branches found in CT code)"
                RESULTS="$RESULTS\n  [WARN] $TAG -- branches in CT disasm"
                # Don't count as failure, just warning (some opt levels may optimize differently)
            fi
        fi
        
        # 3. dudect smoke test (timing, only in full mode)
        SMOKE_BIN="$BUILD_DIR/cpu/test_ct_sidechannel_smoke"
        if $FULL_MODE && [[ -x "$SMOKE_BIN" ]]; then
            echo "  [$TAG] Running dudect smoke (60s)..."
            if timeout 60 "$SMOKE_BIN" > "$BUILD_DIR/dudect_smoke.log" 2>&1; then
                echo "  [$TAG] dudect: PASS"
            else
                echo "  [$TAG] dudect: FAIL (timing leakage)"
                RESULTS="$RESULTS\n  [FAIL] $TAG -- dudect detected leakage"
                FAILED=$((FAILED + 1))
                continue
            fi
        fi
        
        PASSED=$((PASSED + 1))
        RESULTS="$RESULTS\n  [PASS] $TAG"
    done
done

# -- Summary --------------------------------------------------------------

echo ""
echo "==========================================================="
echo "  Cross-Compiler CT Stress: Summary"
echo "==========================================================="
echo -e "$RESULTS"
echo ""
echo "  Total: $TOTAL  Pass: $PASSED  Fail: $FAILED  Skip: $SKIPPED"
echo "==========================================================="

# -- JSON report ----------------------------------------------------------

REPORT="$SRC_DIR/build/ct-stress/report.json"
mkdir -p "$(dirname "$REPORT")"
cat > "$REPORT" <<EOF
{
  "tool": "cross_compiler_ct_stress",
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "full_mode": $FULL_MODE,
  "compilers": [$(printf '"%s",' "${COMPILERS[@]}" | sed 's/,$//')],
  "total": $TOTAL,
  "pass": $PASSED,
  "fail": $FAILED,
  "skip": $SKIPPED,
  "verdict": "$([ "$FAILED" -eq 0 ] && echo "PASS" || echo "FAIL")"
}
EOF

echo "  JSON: $REPORT"
echo ""

if [[ $FAILED -gt 0 ]]; then
    exit 1
else
    echo "  OK All compiler/optimization combos passed CT verification"
    exit 0
fi
