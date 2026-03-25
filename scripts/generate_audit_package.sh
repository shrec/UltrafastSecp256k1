#!/usr/bin/env bash
# ============================================================================
# generate_audit_package.sh -- Comprehensive Audit Evidence Generator
# ============================================================================
#
# Builds the unified_audit_runner, runs it, and aggregates all evidence
# into a single dated directory ready for auditor review.
#
# Usage:
#   bash scripts/generate_audit_package.sh
#   bash scripts/generate_audit_package.sh --build-dir build-audit
#   bash scripts/generate_audit_package.sh --section math_invariants
#   bash scripts/generate_audit_package.sh --skip-build
# ============================================================================
set -euo pipefail

BUILD_DIR="build-audit"
SECTION=""
SKIP_BUILD=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --build-dir)  BUILD_DIR="$2"; shift 2 ;;
        --section)    SECTION="$2";   shift 2 ;;
        --skip-build) SKIP_BUILD=1;   shift ;;
        --help|-h)
            cat <<'EOF'
generate_audit_package.sh -- Comprehensive Audit Evidence Generator

OPTIONS:
  --build-dir <dir>     Build directory (default: build-audit)
  --section <id>        Run only one section (default: all 8)
  --skip-build          Skip cmake configure + build
  --help                Show this message

SECTIONS:
  math_invariants     Mathematical Invariants (Fp, Zn, Group Laws)
  ct_analysis         Constant-Time & Side-Channel Analysis
  differential        Differential & Cross-Library Testing
  standard_vectors    Standard Test Vectors (BIP-340, RFC-6979, BIP-32)
  fuzzing             Fuzzing & Adversarial Attack Resilience
  protocol_security   Protocol Security (ECDSA, Schnorr, MuSig2, FROST)
  memory_safety       ABI & Memory Safety (zeroization, hardening)
  performance         Performance Validation & Regression
EOF
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

# -- Locate project root ---------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
if [ ! -f "$PROJECT_ROOT/CMakeLists.txt" ]; then
    echo "ERROR: Cannot find CMakeLists.txt in $PROJECT_ROOT" >&2
    exit 1
fi

# -- Timestamp & output directory -------------------------------------------
TS="$(date +%Y%m%d-%H%M%S)"
OUTPUT_DIR="$PROJECT_ROOT/audit-evidence-$TS"
mkdir -p "$OUTPUT_DIR/tool_evidence"
mkdir -p "$OUTPUT_DIR/ct_evidence"

echo "================================================================"
echo "  Audit Evidence Package Generator"
echo "  Output: $OUTPUT_DIR"
echo "  Timestamp: $TS"
echo "================================================================"
echo ""

# -- Step 1: Build ----------------------------------------------------------
FULL_BUILD_DIR="$PROJECT_ROOT/$BUILD_DIR"

if [ "$SKIP_BUILD" -eq 0 ]; then
    echo "[1/4] Configuring + building..."

    CMAKE_ARGS=(
        -S "$PROJECT_ROOT"
        -B "$FULL_BUILD_DIR"
        -DCMAKE_BUILD_TYPE=Release
        -DBUILD_TESTING=ON
        -DSECP256K1_BUILD_PROTOCOL_TESTS=ON
        -DSECP256K1_BUILD_FUZZ_TESTS=ON
    )

    # Prefer Ninja if available
    if command -v ninja &>/dev/null; then
        CMAKE_ARGS+=(-G Ninja)
    fi

    cmake "${CMAKE_ARGS[@]}"
    cmake --build "$FULL_BUILD_DIR" --config Release -j "$(nproc)"
else
    echo "[1/4] Build skipped (--skip-build)"
    if [ ! -d "$FULL_BUILD_DIR" ]; then
        echo "ERROR: Build directory $FULL_BUILD_DIR does not exist" >&2
        exit 1
    fi
fi

# -- Step 2: Find the runner binary -----------------------------------------
RUNNER=""
for candidate in \
    "$FULL_BUILD_DIR/audit/unified_audit_runner" \
    "$FULL_BUILD_DIR/audit/Release/unified_audit_runner"; do
    if [ -x "$candidate" ]; then
        RUNNER="$candidate"
        break
    fi
done

if [ -z "$RUNNER" ]; then
    echo "ERROR: Cannot find unified_audit_runner in $FULL_BUILD_DIR" >&2
    exit 1
fi
echo "[2/4] Found runner: $RUNNER"

# -- Step 3: Run unified audit runner ---------------------------------------
echo "[3/4] Running unified audit runner..."

RUNNER_ARGS=(--report-dir "$OUTPUT_DIR")
if [ -n "$SECTION" ]; then
    RUNNER_ARGS+=(--section "$SECTION")
fi

EXIT_CODE=0
"$RUNNER" "${RUNNER_ARGS[@]}" || EXIT_CODE=$?

echo ""
if [ "$EXIT_CODE" -eq 0 ]; then
    echo "[3/4] Audit runner: ALL PASSED"
else
    echo "[3/4] Audit runner: FAILURES DETECTED (exit code $EXIT_CODE)"
fi

# -- Step 4: Collect additional evidence ------------------------------------
echo "[4/4] Collecting tool evidence..."

# 4a. Build info JSON
GIT_HASH="$(git -C "$PROJECT_ROOT" rev-parse --short=8 HEAD 2>/dev/null || echo unknown)"
GIT_BRANCH="$(git -C "$PROJECT_ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
OS_INFO="$(uname -srm 2>/dev/null || echo unknown)"
cat > "$OUTPUT_DIR/build_info.json" <<EOJSON
{
  "timestamp": "$TS",
  "project_root": "$PROJECT_ROOT",
  "build_dir": "$FULL_BUILD_DIR",
  "runner_path": "$RUNNER",
  "runner_exit_code": $EXIT_CODE,
  "section_filter": "${SECTION:-all}",
  "os": "$OS_INFO",
  "git_hash": "$GIT_HASH",
  "git_branch": "$GIT_BRANCH"
}
EOJSON

# 4b. Compiler version
{
    for cc in g++-13 g++ clang++-17 clang++ c++; do
        if command -v "$cc" &>/dev/null; then
            echo "=== $cc ==="
            "$cc" --version 2>&1 | head -3
            echo ""
        fi
    done
} > "$OUTPUT_DIR/tool_evidence/compiler_version.txt" 2>/dev/null || true

# 4c. CMake cache summary
CACHE_FILE="$FULL_BUILD_DIR/CMakeCache.txt"
if [ -f "$CACHE_FILE" ]; then
    grep -E '^(CMAKE_BUILD_TYPE|CMAKE_CXX_COMPILER|CMAKE_CXX_STANDARD|SECP256K1_BUILD_|BUILD_TESTING|CMAKE_SYSTEM)' \
        "$CACHE_FILE" > "$OUTPUT_DIR/tool_evidence/cmake_cache_summary.txt" 2>/dev/null || true
fi

# 4d. CTest results
CTEST_XML="$(find "$FULL_BUILD_DIR/Testing" -name 'Test.xml' 2>/dev/null | tail -1)"
if [ -n "$CTEST_XML" ] && [ -f "$CTEST_XML" ]; then
    cp "$CTEST_XML" "$OUTPUT_DIR/tool_evidence/ctest_results.xml"
fi

# 4e. Git info
echo "Branch: $GIT_BRANCH" > "$OUTPUT_DIR/tool_evidence/git_info.txt"
echo "Commit: $GIT_HASH" >> "$OUTPUT_DIR/tool_evidence/git_info.txt"

# 4f. CT evidence collection
echo "  Collecting CT evidence artifacts..."
CT_EVIDENCE_DIR="$OUTPUT_DIR/ct_evidence"
if python3 "$PROJECT_ROOT/scripts/collect_ct_evidence.py" \
    --repo-root "$PROJECT_ROOT" \
    --build-dir "$FULL_BUILD_DIR" \
    --output-dir "$CT_EVIDENCE_DIR" \
    --runner-binary "$RUNNER"; then
    CT_COUNT=$(find "$CT_EVIDENCE_DIR" -type f 2>/dev/null | wc -l)
    echo "  CT evidence: $CT_COUNT artifact(s) normalized"
else
    echo "  [!] CT evidence normalization failed"
fi

# 4g. Auditor README
cat > "$OUTPUT_DIR/README.txt" <<'EOREADME'
================================================================
  UltrafastSecp256k1 -- Audit Evidence Package
================================================================

This directory contains the complete self-audit evidence for the
UltrafastSecp256k1 cryptographic library.

CONTENTS:
  audit_report.json          Machine-readable test results (8 sections)
  audit_report.txt           Human-readable audit summary
  build_info.json            Build environment metadata
  tool_evidence/
    compiler_version.txt     Compiler version used
    cmake_cache_summary.txt  CMake build options
    ctest_results.xml        CTest XML results (if run)
    git_info.txt             Git branch + commit
  ct_evidence/
    ct_verif.log             Deterministic CT verification output (ct-verif)
    ct_verif_summary.json    CT verification structured results
    valgrind_ct.log          Valgrind CT memcheck output (ctgrind mode)
    valgrind_ct_report.json  Valgrind CT structured results
    disasm_branch_scan.json  Disassembly branch analysis of CT functions
    dudect_smoke.log         Statistical timing test (dudect, smoke run)
    dudect_full.log          Statistical timing test (dudect, full run)
        ct_evidence_summary.json Aggregate CT evidence posture and residual gaps
        ct_evidence_summary.txt  Human-readable CT evidence summary

Note: ct_evidence/ files are present when available from local or CI runs.
If empty, check CI artifacts at the URLs listed below.

AUDIT SECTIONS (8 categories):
  1. Mathematical Invariants    -- Fp, Zn, group laws (13 modules)
  2. CT & Side-Channel          -- dudect, FAST==CT, timing (5 modules)
  3. Differential Testing       -- cross-library, Fiat-Crypto (3 modules)
  4. Standard Test Vectors      -- BIP-340, RFC-6979, BIP-32, FROST (4 modules)
  5. Fuzzing & Adversarial      -- parser fuzz, fault injection (4 modules)
  6. Protocol Security          -- ECDSA, Schnorr, MuSig2, FROST (9 modules)
  7. ABI & Memory Safety        -- zeroization, ABI gate (3 modules)
  8. Performance Validation     -- SIMD, hash accel, multi-scalar (4 modules)

TOTAL: 47 test modules + 1 library selftest = 48 checks

CT VERIFICATION STACK:
  Layer 1: Statistical timing leakage testing (dudect)
    - Welch t-test, |t| > 4.5 = leak. Advisory on shared runners.
  Layer 2: Deterministic CT verification (ct-verif + valgrind-ct)
    - ct-verif: LLVM-based taint tracking, blocking in CI
    - valgrind-ct: ctgrind-mode memcheck, blocking in CI
  Layer 3: Disassembly branch scan
    - Checks compiled CT functions for conditional branches on secrets
  Layer 4: Machine-checked proof frameworks
    - Not currently applied (Vale/Jasmin/Coq/Fiat-Crypto)

HOW TO VERIFY:
  1. Review audit_report.json for structured pass/fail data
  2. Confirm all 8 sections show "status": "PASS"
  3. Check ct_evidence/ for deterministic CT verification results
  4. Verify platform/compiler info matches expected target
  5. Check build_info.json for reproducible build parameters

EXTERNAL TOOL EVIDENCE (collected separately by CI):
  - ct-verif:    .github/workflows/ct-verif.yml    -> Actions artifacts
  - valgrind-ct: .github/workflows/ct-verif.yml    -> Actions artifacts
  - CodeQL:      .github/workflows/codeql.yml      -> Security tab
  - Cppcheck:    .github/workflows/cppcheck.yml    -> Security tab
  - Scorecard:   .github/workflows/scorecard.yml   -> Security tab
  - SonarCloud:  .github/workflows/sonarcloud.yml  -> sonarcloud.io
  - ClusterFuzz: .github/workflows/cflite.yml      -> PR checks
  - Mutation:    .github/workflows/mutation.yml     -> Weekly run
  - Clang-Tidy:  .github/workflows/clang-tidy.yml  -> Security tab
  - GPU audit:   .github/workflows/gpu-selfhosted.yml -> Self-hosted runner
================================================================
EOREADME

# -- Summary ----------------------------------------------------------------
echo ""
echo "================================================================"
echo "  Audit Evidence Package: COMPLETE"
echo "  Location: $OUTPUT_DIR"
echo ""
echo "  Files:"
find "$OUTPUT_DIR" -type f -printf "    %-45P %8s bytes\n" 2>/dev/null || \
    find "$OUTPUT_DIR" -type f -exec ls -l {} \;
echo ""
if [ "$EXIT_CODE" -eq 0 ]; then
    echo "  VERDICT: AUDIT-READY (all modules passed)"
else
    echo "  VERDICT: AUDIT-BLOCKED (failures detected)"
fi
echo "================================================================"

exit "$EXIT_CODE"
