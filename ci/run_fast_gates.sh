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

run() {
    local label="$1"; shift
    printf "  %-48s" "${label}..."
    local out rc
    out=$(python3 "$@" 2>&1); rc=$?
    if [ "$rc" -eq 0 ]; then
        printf " \033[0;32mOK\033[0m\n"
    elif [ "$rc" -eq 77 ]; then
        printf " \033[0;33mSKIP\033[0m\n"
    else
        printf " \033[0;31mFAIL\033[0m\n"
        printf "%s\n" "$out" | sed 's/^/    /'
        FAILED=$((FAILED + 1))
    fi
}

run "Repo map check"          tools/render_repo_map.py --check
run "Exploit wiring parity"  ci/check_exploit_wiring.py
run "Canonical data sync"    ci/build_canonical_data.py --dry-run
run "Docs from canonical"    ci/sync_docs_from_canonical.py --dry-run
run "Module count sync"      ci/sync_module_count.py --dry-run
run "Audit scripts"          ci/test_audit_scripts.py --quick
run "Assurance validation"   ci/validate_assurance.py

if [[ "${FAILED}" -gt 0 ]]; then
    echo ""
    echo "  \033[0;31m${FAILED} gate(s) FAILED\033[0m"
    exit 1
fi

echo "  \033[0;32mAll fast gates passed\033[0m"
