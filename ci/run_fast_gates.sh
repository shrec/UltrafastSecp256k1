#!/usr/bin/env bash
# =============================================================================
# run_fast_gates.sh — Fast deterministic CI gates (~30s)
# =============================================================================
# SINGLE SOURCE OF TRUTH for fast gates.
# Called by:  ci_local.sh  AND  gate.yml's "Fast deterministic gates" step.
# If this script passes locally → it passes on GitHub. No divergence.
#
# Usage:
#   bash ci/run_fast_gates.sh            # exit 1 on first failure
#   bash ci/run_fast_gates.sh --summary  # print summary (for GitHub Step Summary)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

# Validate caas_runner.py: syntax + import-time correctness before running any gates
python3 -c "
import importlib.util, sys
spec = importlib.util.spec_from_file_location('caas_runner', 'ci/caas_runner.py')
mod  = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(mod)
except Exception as e:
    print(f'ERROR: ci/caas_runner.py failed at import time: {e}', file=sys.stderr)
    sys.exit(1)
" 2>&1 || {
    echo "ERROR: ci/caas_runner.py has a SyntaxError or import-time error — fix before running gates"
    exit 1
}

FAILED=0
SUMMARY=0
[[ "${1:-}" == "--summary" ]] && SUMMARY=1

# Gates where rc=77 (SKIP) must be treated as FAIL.
# These scripts must never return 77 in a healthy repo — canonical JSON is
# always present, wiring checks always run, etc.
MANDATORY_GATES=(
    "ci/check_exploit_wiring.py"
    "ci/check_security_fix_has_test.py"
    "ci/check_version_sync.py"
    "ci/build_canonical_data.py"
    "ci/sync_docs_from_canonical.py"
    "ci/sync_module_count.py"
    "ci/sync_canonical_numbers.py"
    "ci/check_bench_doc_consistency.py"
    "ci/check_backend_parity.py"
    "ci/check_secret_parse_strictness.py"
    "ci/check_protocol_invariants.py"
    "ci/check_nonce_erase_coverage.py"
    "ci/check_doc_drift.py"
    "tools/render_repo_map.py"
    "ci/validate_assurance.py"
)

is_mandatory() {
    local script="$1"
    local g
    for g in "${MANDATORY_GATES[@]}"; do
        if [[ "$script" == *"$g"* ]]; then
            return 0
        fi
    done
    return 1
}

run() {
    local label="$1"; shift
    printf "  %-48s" "${label}..."
    local out rc
    out=$(python3 "$@" 2>&1); rc=$?
    if [ "$rc" -eq 0 ]; then
        printf " \033[0;32mOK\033[0m\n"
    elif [ "$rc" -eq 77 ]; then
        # rc=77 on a mandatory gate is a FAIL (missing artifact / broken gate)
        if is_mandatory "$*"; then
            printf " \033[0;31mFAIL\033[0m (rc=77 on mandatory gate — missing artifact or broken skip logic)\n"
            printf "%s\n" "$out" | sed 's/^/    /'
            FAILED=$((FAILED + 1))
        else
            printf " \033[0;33mSKIP\033[0m\n"
        fi
    else
        printf " \033[0;31mFAIL\033[0m\n"
        printf "%s\n" "$out" | sed 's/^/    /'
        FAILED=$((FAILED + 1))
    fi
}

run_sh() {
    local label="$1"; shift
    printf "  %-48s" "${label}..."
    local out rc
    rc=0
    out=$(bash "$@" 2>&1) || rc=$?  # || prevents set -e from firing on non-zero exit
    if [ "$rc" -eq 0 ]; then
        printf " \033[0;32mOK\033[0m\n"
    elif [ "$rc" -eq 77 ]; then
        # Advisory shell gates: rc=77 = SKIP (e.g. GPU not present)
        printf " \033[0;33mSKIP\033[0m\n"
    else
        printf " \033[0;31mFAIL\033[0m\n"
        printf "%s\n" "$out" | sed 's/^/    /'
        FAILED=$((FAILED + 1))
    fi
}

run "Repo map check"          tools/render_repo_map.py --check
run "Exploit wiring parity"  ci/check_exploit_wiring.py
run "Security fix has test"   ci/check_security_fix_has_test.py --commits 10
run "Version + count sync"   ci/check_version_sync.py
run "Canonical data sync"    ci/build_canonical_data.py --dry-run
run "Docs from canonical"    ci/sync_docs_from_canonical.py --dry-run
run "Module count sync"      ci/sync_module_count.py --dry-run
run "Canonical numbers sync" ci/sync_canonical_numbers.py --dry-run
run "Audit scripts"          ci/test_audit_scripts.py --quick
run "Assurance validation"   ci/validate_assurance.py

# Protocol & Backend Parity Gates — catch copy-paste divergence and
# protocol invariant violations (root cause of confirmed red-team bugs C1–C8).
run "Backend parity"                           ci/check_backend_parity.py
run "Secret parse strictness (Rule 11)"        ci/check_secret_parse_strictness.py
run "Protocol invariants (FROST threshold)"    ci/check_protocol_invariants.py
run "Nonce erase coverage (BIP-327)"           ci/check_nonce_erase_coverage.py
run "Doc drift (badges, removed files)"        ci/check_doc_drift.py
run "Bench/doc consistency (banned patterns)" ci/check_bench_doc_consistency.py

# Advisory gates — rc=77 means infrastructure absent (GPU etc.), not a failure
run_sh "Advisory skip codes"  ci/check_advisory_skip_returns.sh

if [[ "${FAILED}" -gt 0 ]]; then
    echo ""
    echo "  \033[0;31m${FAILED} gate(s) FAILED\033[0m"
    exit 1
fi

echo "  \033[0;32mAll fast gates passed\033[0m"
