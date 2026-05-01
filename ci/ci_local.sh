#!/usr/bin/env bash
# ============================================================================
# ci_local.sh — Run CI gates locally before pushing
# ============================================================================
# Usage:
#   ./ci/ci_local.sh           # quick gates only (~30s)
#   ./ci/ci_local.sh --full    # quick + build + tests (~10min)
#   ./ci/ci_local.sh --msan    # also runs no-ASM build (MSan path)
#
# Install as pre-push hook:
#   ln -sf ../../ci/ci_local.sh .git/hooks/pre-push
# ============================================================================
set -uo pipefail
# NOTE: no -e intentionally — we handle exit codes manually in run_check.
# bash arithmetic ((n++)) returns 1 when n=0, which would kill the script
# with -e even on success.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BOLD='\033[1m'; NC='\033[0m'

FULL=0; MSAN=0
for arg in "$@"; do
  [[ "$arg" == "--full" ]] && FULL=1
  [[ "$arg" == "--msan" ]] && MSAN=1 && FULL=1
done

pass=0; fail=0

run_check() {
  local label="$1"; shift
  printf "  %-52s" "$label..."
  if output=$("$@" 2>&1); then
    echo -e "${GREEN}OK${NC}"
    ((pass++))
  else
    echo -e "${RED}FAIL${NC}"
    echo "$output" | tail -10 | sed 's/^/    /'
    ((fail++))
  fi
}

echo -e "${BOLD}━━━ CI Local Gates ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# ── Fast gates (~5s) ────────────────────────────────────────────────────────
echo -e "${BOLD}[1] Doc & Wiring Gates${NC}"
run_check "Exploit wiring parity"   python3 ci/check_exploit_wiring.py
run_check "Canonical data sync"     python3 ci/build_canonical_data.py --dry-run
run_check "Docs from canonical"     python3 ci/sync_docs_from_canonical.py --dry-run
run_check "Module count sync"       python3 ci/sync_module_count.py --dry-run
echo ""

# ── Code quality scanners (~15s) ─────────────────────────────────────────────
echo -e "${BOLD}[2] Code Quality Scanners${NC}"
run_check "hot_path_alloc regression"  python3 ci/run_code_quality.py --fail-on-regression --json > /dev/null
  # CUDA checker lives in parent repo tools/ (or local tools/)
  _cuda_checker=""
  for _p in tools/cuda_intrinsic_checker.py ../tools/cuda_intrinsic_checker.py; do
    [[ -f "$ROOT/$_p" ]] && _cuda_checker="$ROOT/$_p" && break
  done
  if [[ -n "$_cuda_checker" ]]; then
    run_check "CUDA intrinsic selectors" python3 "$_cuda_checker" --check-only
  else
    echo -e "  CUDA intrinsic checker not found (skipped)              ${YELLOW}SKIP${NC}"
  fi
echo ""

# ── Preflight full scan (~20s) ─────────────────────────────────────────────
echo -e "${BOLD}[3] Preflight Scan (--bug-scan)${NC}"
run_check "Preflight --bug-scan"    python3 ci/preflight.py --bug-scan
echo ""

if [[ $FULL -eq 0 ]]; then
  echo -e "${YELLOW}Tip: run with --full to also build + test (catches MSan-class issues)${NC}"
  echo ""
fi

# ── Full build + no-ASM test (~5min) ─────────────────────────────────────────
if [[ $FULL -eq 1 ]]; then
  echo -e "${BOLD}[4] Build (Release, no-ASM — mirrors sanitizer CI path)${NC}"
  BUILD_DIR="/tmp/ci_local_build_$$"
  if cmake -S . -B "$BUILD_DIR" -G Ninja \
      -DCMAKE_BUILD_TYPE=Debug \
      -DSECP256K1_USE_ASM=OFF \
      -DSECP256K1_BUILD_TESTS=ON \
      -DSECP256K1_BUILD_BENCH=OFF \
      -DSECP256K1_BUILD_EXAMPLES=OFF \
      -DSECP256K1_BUILD_CUDA=OFF \
      -DSECP256K1_BUILD_OPENCL=OFF \
      -DSECP256K1_BUILD_METAL=OFF \
      > /tmp/ci_local_cmake.log 2>&1; then
    echo -e "  cmake configure                                      ${GREEN}OK${NC}"
    ((pass++))
  else
    echo -e "  cmake configure                                      ${RED}FAIL${NC}"
    tail -20 /tmp/ci_local_cmake.log | sed 's/^/    /'
    ((fail++))
  fi

  if [[ $fail -eq 0 ]]; then
    printf "  %-52s" "cmake build (no-ASM)..."
    if cmake --build "$BUILD_DIR" -j"$(nproc)" > /tmp/ci_local_build.log 2>&1; then
      echo -e "${GREEN}OK${NC}"; ((pass++))
    else
      echo -e "${RED}FAIL${NC}"
      tail -20 /tmp/ci_local_build.log | sed 's/^/    /'
      ((fail++))
    fi
  fi

  if [[ $fail -eq 0 ]]; then
    echo -e "${BOLD}[5] Tests (no-ASM — mul_wide portable path)${NC}"
    run_check "field_26 cross-check"    "$BUILD_DIR/cpu/test_field_26_standalone"
    run_check "field_52 cross-check"    "$BUILD_DIR/cpu/test_field_52_standalone"
    run_check "Cross-platform KAT"      "$BUILD_DIR/audit/test_cross_platform_kat_standalone" 2>/dev/null || true
    run_check "BIP-39 NFKD"            "$BUILD_DIR/audit/test_exploit_bip39_nfkd_standalone"
  fi

  rm -rf "$BUILD_DIR" 2>/dev/null || true
  echo ""
fi

# ── MSan-specific checks (optional) ─────────────────────────────────────────
if [[ $MSAN -eq 1 ]]; then
  echo -e "${BOLD}[6] MSan quick smoke (clang -fsanitize=memory)${NC}"
  if ! command -v clang-17 &>/dev/null; then
    echo -e "  ${YELLOW}clang-17 not found — skipping MSan smoke test${NC}"
  else
    MSAN_DIR="/tmp/ci_local_msan_$$"
    if CC=clang-17 CXX=clang++-17 cmake -S . -B "$MSAN_DIR" -G Ninja \
        -DCMAKE_BUILD_TYPE=Debug \
        -DCMAKE_C_FLAGS="-fsanitize=memory -fno-omit-frame-pointer" \
        -DCMAKE_CXX_FLAGS="-fsanitize=memory -fno-omit-frame-pointer -stdlib=libc++" \
        -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=memory -stdlib=libc++ -lc++abi" \
        -DSECP256K1_USE_ASM=OFF \
        -DSECP256K1_BUILD_TESTS=ON \
        -DSECP256K1_BUILD_BENCH=OFF \
        -DSECP256K1_BUILD_CUDA=OFF \
        -DSECP256K1_BUILD_OPENCL=OFF \
        > /dev/null 2>&1 && \
       cmake --build "$MSAN_DIR" --target test_field_26_standalone -j"$(nproc)" > /dev/null 2>&1; then
      run_check "MSan field_26 (no false positives)" \
        "$MSAN_DIR/cpu/test_field_26_standalone"
    fi
    rm -rf "$MSAN_DIR" 2>/dev/null || true
  fi
  echo ""
fi

# ── Summary ──────────────────────────────────────────────────────────────────
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
total=$((pass+fail))
if [[ $fail -eq 0 ]]; then
  echo -e "${GREEN}${BOLD}  ALL $total CHECKS PASSED${NC} — safe to push"
else
  echo -e "${RED}${BOLD}  $fail/$total CHECKS FAILED${NC} — fix before pushing"
  exit 1
fi
