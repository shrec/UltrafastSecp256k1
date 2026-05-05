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
# NOTE: no -e intentionally — we handle exit codes manually in run_check/run_caas_check.
# bash arithmetic ((n++)) returns 1 when n=0, which would kill the script with -e.
# Consequence: commands OUTSIDE run_check blocks (e.g. the JSON artifact write, mkdir)
# do not abort automatically on failure. Those paths must handle errors explicitly.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BOLD='\033[1m'; NC='\033[0m'

FULL=0; MSAN=0
for arg in "$@"; do
  [[ "$arg" == "--full" ]] && FULL=1
  [[ "$arg" == "--msan" ]] && MSAN=1 && FULL=1
done

# F-23 fix: ensure temp build dirs are cleaned up even on SIGINT/SIGTERM/ERR.
# Without a trap, Ctrl-C or a timeout kill leaves /tmp/ci_local_build_<PID>
# and /tmp/ci_local_msan_<PID> accumulating across multiple interrupted runs.
BUILD_DIR=""
MSAN_DIR=""
_ci_local_cleanup() {
  [[ -n "${BUILD_DIR:-}" ]] && rm -rf "$BUILD_DIR" 2>/dev/null
  [[ -n "${MSAN_DIR:-}"  ]] && rm -rf "$MSAN_DIR"  2>/dev/null
}
trap '_ci_local_cleanup' EXIT INT TERM

pass=0; fail=0; adv_skip=0
declare -A gate_results  # gate label → pass|fail|adv-skip, for local artifact

run_check() {
  local label="$1"; shift
  printf "  %-52s" "$label..."
  if output=$("$@" 2>&1); then
    echo -e "${GREEN}OK${NC}"
    ((pass++))
    gate_results["$label"]="pass"
  else
    echo -e "${RED}FAIL${NC}"
    echo "$output" | tail -10 | sed 's/^/    /'
    ((fail++))
    gate_results["$label"]="fail"
  fi
}

echo -e "${BOLD}━━━ CI Local Gates ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# ── Path validator — prevents "file not found" on GitHub ─────────────────────
echo -e "${BOLD}[0] CI Path Integrity${NC}"
run_check "CI paths exist"          python3 ci/check_ci_paths.py
echo ""

# ── Fast gates — SAME script that gate.yml runs ──────────────────────────────
# Single source of truth: ci/run_fast_gates.sh
# local passes == GitHub passes. No divergence possible.
echo -e "${BOLD}[1] Fast Gates (= gate.yml Block 1)${NC}"
run_check "Fast gates"              bash ci/run_fast_gates.sh
echo ""

# ── Code quality scanners (~15s) ─────────────────────────────────────────────
echo -e "${BOLD}[2] Code Quality Scanners${NC}"
run_check "hot_path_alloc regression"  python3 ci/run_code_quality.py --fail-on-regression
  # L-2: ../tools/ is in the parent Secp256K1fast repo. On standalone
  # UltrafastSecp256k1 checkouts this path is absent and CUDA checks are skipped.
  _cuda_checker=""
  for _p in tools/cuda_intrinsic_checker.py ../tools/cuda_intrinsic_checker.py; do
    [[ -f "$ROOT/$_p" ]] && _cuda_checker="$ROOT/$_p" && break
  done
  if [[ -n "$_cuda_checker" ]]; then
    run_check "CUDA intrinsic selectors" python3 "$_cuda_checker" --check-only
  else
    printf "  %-52s" "CUDA intrinsic selectors..."
    echo -e "${YELLOW}ADV-SKIP${NC} (checker not found)"
    ((adv_skip++))
    gate_results["CUDA intrinsic selectors"]="adv-skip"
  fi
echo ""

# ── Preflight full scan (~20s) ─────────────────────────────────────────────
echo -e "${BOLD}[3] Preflight Scan${NC}"
run_check "Preflight --bug-scan"    python3 ci/preflight.py --bug-scan
# R-6 fix: --security, --abi, --freshness are hard-fail gates in preflight.yml
# (run on every push/PR). Running only --bug-scan locally means these failures
# are only discovered on CI, causing 2+ day fix cycles.
run_check "Preflight --security"    python3 ci/preflight.py --security
run_check "Preflight --abi"         python3 ci/preflight.py --abi
run_check "Preflight --freshness"   python3 ci/preflight.py --freshness
run_check "Exploit traceability join" python3 ci/exploit_traceability_join.py
echo ""

# ── CAAS security gates (~10s) ──────────────────────────────────────────────
# Mirrors caas.yml Stage 1 (scanner, via Section [1] fast-gates above) +
# Stage 2 (audit_gate) + Stage 3 (security_autonomy). Stage 1 scanner runs
# inside run_fast_gates.sh (Section [1]) via validate_assurance.py.
# exploit_traceability_join.py (Stage 1b) runs in Section [3] Preflight above.
echo -e "${BOLD}[4] CAAS Security Gates (mirrors CI caas.yml stages 1b+2+3)${NC}"
_caas_fail=0

run_caas_check() {
  local label="$1"; shift
  printf "  %-52s" "$label..."
  local out rc
  out=$("$@" 2>&1); rc=$?
  if [[ $rc -eq 0 ]]; then
    echo -e "${GREEN}OK${NC}"; ((pass++)); gate_results["$label"]="pass"
  elif [[ $rc -eq 77 ]]; then
    # F-06 fix: count advisory-skips separately so the final summary is accurate.
    echo -e "${YELLOW}ADV-SKIP${NC} (no required infrastructure)"; ((adv_skip++)); gate_results["$label"]="adv-skip"
  else
    echo -e "${RED}FAIL${NC}"; echo "$out" | tail -6 | sed 's/^/    /'
    ((fail++)); ((_caas_fail++)); gate_results["$label"]="fail"
  fi
}

# R-5 fix: audit_gate.py requires the project graph DB. Without rebuilding it
# first, the gate fails with "Graph DB not found" on a fresh checkout.
# This matches what gate.yml Block 3 / caas.yml do before running audit_gate.py.
printf "  %-52s" "Build project graph..."
if _graph_out=$(python3 ci/build_project_graph.py --rebuild 2>&1); then
  echo -e "${GREEN}OK${NC}"
  ((pass++))
  gate_results["Build project graph"]="pass"
else
  echo -e "${RED}FAIL${NC}"
  echo "$_graph_out" | tail -6 | sed 's/^/    /'
  ((fail++)); ((_caas_fail++))
  gate_results["Build project graph"]="fail"
fi

run_caas_check "Audit gate"          python3 ci/audit_gate.py
run_caas_check "Security autonomy"   python3 ci/security_autonomy_check.py
run_caas_check "Shim parity"         python3 ci/check_libsecp_shim_parity.py
echo ""

if [[ $FULL -eq 0 ]]; then
  echo -e "${YELLOW}Tip: run with --full to also build + test (catches MSan-class issues)${NC}"
  echo ""
fi

# ── Full build + no-ASM test (~5min) ─────────────────────────────────────────
if [[ $FULL -eq 1 ]]; then
  echo -e "${BOLD}[4] Build (Release, no-ASM — mirrors sanitizer CI path)${NC}"
  BUILD_DIR="${TMPDIR:-/tmp}/ci_local_build_$$"
  _cmake_log="${TMPDIR:-/tmp}/ci_local_cmake_$$.log"
  _build_log="${TMPDIR:-/tmp}/ci_local_build_$$.log"
  if cmake -S . -B "$BUILD_DIR" -G Ninja \
      -DCMAKE_BUILD_TYPE=Debug \
      -DSECP256K1_USE_ASM=OFF \
      -DSECP256K1_BUILD_TESTS=ON \
      -DSECP256K1_BUILD_BENCH=OFF \
      -DSECP256K1_BUILD_EXAMPLES=OFF \
      -DSECP256K1_BUILD_CUDA=OFF \
      -DSECP256K1_BUILD_OPENCL=OFF \
      -DSECP256K1_BUILD_METAL=OFF \
      > "$_cmake_log" 2>&1; then
    echo -e "  cmake configure                                      ${GREEN}OK${NC}"
    ((pass++))
    gate_results["build:cmake_config"]="pass"
  else
    echo -e "  cmake configure                                      ${RED}FAIL${NC}"
    tail -20 "$_cmake_log" | sed 's/^/    /'
    ((fail++))
    gate_results["build:cmake_config"]="fail"
  fi

  if [[ $fail -eq 0 ]]; then
    printf "  %-52s" "cmake build (no-ASM)..."
    if cmake --build "$BUILD_DIR" -j"$(nproc)" > "$_build_log" 2>&1; then
      echo -e "${GREEN}OK${NC}"; ((pass++))
      gate_results["build:cmake_build"]="pass"
    else
      echo -e "${RED}FAIL${NC}"
      tail -20 "$_build_log" | sed 's/^/    /'
      ((fail++))
      gate_results["build:cmake_build"]="fail"
    fi
  else
    # F-16 fix: announce explicitly that the build was skipped due to prior failures,
    # so the developer is not left wondering why no build output appeared.
    echo -e "  cmake build (no-ASM)                                 ${YELLOW}SKIP${NC} (prior checks failed)"
  fi

  if [[ $fail -eq 0 ]]; then
    echo -e "${BOLD}[5] Tests (no-ASM — mul_wide portable path)${NC}"
    run_check "field_26 cross-check"    "$BUILD_DIR/cpu/test_field_26_standalone"
    run_check "field_52 cross-check"    "$BUILD_DIR/cpu/test_field_52_standalone"
    # Cross-platform KAT binary may not exist in every cmake mode;
    # explicitly skip rather than masking failures with `|| true`.
    if [[ -x "$BUILD_DIR/audit/test_cross_platform_kat_standalone" ]]; then
      run_check "Cross-platform KAT"    "$BUILD_DIR/audit/test_cross_platform_kat_standalone"
    else
      printf "  %-52s" "Cross-platform KAT..."
      echo -e "${YELLOW}SKIP${NC} (binary not built)"
    fi
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
    MSAN_DIR="${TMPDIR:-/tmp}/ci_local_msan_$$"
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

# ── Local artifact ───────────────────────────────────────────────────────────
# Write a JSON summary for post-run inspection without re-running (MEDIUM-4/LOW-6).
# F-25 fix: use Python to produce properly escaped JSON instead of raw printf,
# which could produce malformed JSON for gate labels with quotes or backslashes.
_artifact="${SCRIPT_DIR}/../out/ci_local_last_run.json"
mkdir -p "$(dirname "$_artifact")"
{
  python3 - <<PYEOF
import json, subprocess, sys
gates = {}
$(for key in "${!gate_results[@]}"; do
    printf "gates[%s] = %s\n" "$(python3 -c "import json,sys; sys.stdout.write(json.dumps(sys.argv[1]))" "$key")" "$(python3 -c "import json,sys; sys.stdout.write(json.dumps(sys.argv[1]))" "${gate_results[$key]}")"
  done)
# F-03 persistence fix: record the git SHA so two runs on the same commit are
# distinguishable and the artifact can be traced back to a specific commit.
_sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip() or "unknown"
doc = {
    "generated_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "git_sha": _sha,
    "overall_pass": $([ $fail -eq 0 ] && echo True || echo False),
    "pass": $pass,
    "fail": $fail,
    "adv_skip": $adv_skip,
    "gates": gates,
}
print(json.dumps(doc, indent=2))
PYEOF
} > "$_artifact"
if [ $? -ne 0 ]; then
  echo -e "  ${YELLOW}Warning: failed to write local artifact${NC} $_artifact" >&2
fi
echo -e "  ${YELLOW}Local artifact:${NC} $_artifact"
echo ""

# ── Summary ──────────────────────────────────────────────────────────────────
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
total=$((pass+fail))
# F-06 fix: show advisory-skip count separately so the total is accurate.
_adv_suffix=""
[[ $adv_skip -gt 0 ]] && _adv_suffix=" (${adv_skip} advisory-skipped)"
if [[ $fail -eq 0 ]]; then
  echo -e "${GREEN}${BOLD}  ALL $total CHECKS PASSED${NC}${_adv_suffix} — safe to push"
else
  echo -e "${RED}${BOLD}  $fail/$total CHECKS FAILED${NC}${_adv_suffix} — fix before pushing"
  exit 1
fi
