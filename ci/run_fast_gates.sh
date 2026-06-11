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
    # check_advisory_skip_returns.sh is NOT listed here: in fast_gates (pre-build)
    # it legitimately returns 77 (no binaries). It is mandatory only post-build.
    "ci/check_section_ids.py"            # section IDs in ALL_MODULES must be declared in SECTIONS[]
    "ci/test_audit_scripts.py"           # P3-PR-011: audit framework self-test is mandatory
    "ci/check_version_sync.py"
    "ci/check_abi_version_sync.py"       # REL-ABI: binding EXPECTED_ABI must equal library ABI (== MAJOR)
    "ci/check_randomize_claim_consistency.py"  # REVIEWER-FRICTION-001: no doc may call context_randomize a no-op
    "ci/check_required_checks_match_jobs.py"   # CAAS-CI-001: branch-protection contexts must resolve to PR-triggered jobs
    "ci/check_sanitizer_result_assertions.py"  # CAAS6-01: required memcheck jobs must fail closed on "no logs produced"
    "ci/test_check_sanitizer_result_assertions.py"  # self-test: the sanitizer-assertion gate must flag a fail-open block
    "ci/check_doc_module_counts.py"            # CLAIMS-AUDIT-001: reviewer-doc module/workflow counts must match canonical
    "ci/gen_build_options.py"                  # docs/BUILD_OPTIONS.md must match CMake option() declarations
    "ci/test_gen_build_options.py"             # self-test: the build-options drift gate parser + render
    "ci/build_canonical_data.py"
    "ci/sync_docs_from_canonical.py"
    "ci/sync_module_count.py"
    "ci/sync_canonical_numbers.py"
    "ci/check_bench_doc_consistency.py"
    "ci/check_backend_parity.py"
    "ci/check_zk_tag_conformance.py"
    "ci/check_tag_conformance.py"
    "ci/check_secret_parse_strictness.py"
    "ci/check_ct_branches.py"             # GPU-CT-001: forbid secret-dependent branches in CT arithmetic primitives
    "ci/test_check_ct_branches.py"        # self-test: the CT-branch gate must catch the leak pattern
    "ci/check_protocol_invariants.py"
    "ci/check_nonce_erase_coverage.py"
    "ci/check_doc_drift.py"
    "ci/check_advisory_skip_ceiling.py"
    "ci/check_test_assertions.py"        # TEST-001/§12: forbid non-asserting "documented open" probes (MSI-4 anti-pattern)
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
        # CI-011: rc=77 on a mandatory gate is a FAIL — same logic as run().
        # Non-mandatory shell gates: rc=77 = SKIP (e.g. GPU not present).
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

run "Repo map check"          tools/render_repo_map.py --check
run "Exploit wiring parity"  ci/check_exploit_wiring.py
run "Advisory blocking twin (CAAS-FG-01)" ci/check_advisory_has_blocking_test.py
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
run "ZK Fiat-Shamir tag conformance"           ci/check_zk_tag_conformance.py
run "Tagged-hash tag conformance (all tags)"   ci/check_tag_conformance.py
run "Secret parse strictness (Rule 11)"        ci/check_secret_parse_strictness.py
run "GPU/CPU CT-branch lint (GPU-CT-001)"      ci/check_ct_branches.py
run "CT-branch lint self-test"                 ci/test_check_ct_branches.py
run "Protocol invariants (FROST threshold)"    ci/check_protocol_invariants.py
run "Soundness coverage (negative-test ledger)" ci/check_soundness_coverage.py
run "Soundness-gate self-test (proof-it-blocks)" ci/test_check_soundness_coverage.py
run "Threat-gate coverage (don't trust verify)" ci/check_threat_gate_coverage.py
run "Threat-gate self-test (proof-it-blocks)"  ci/test_check_threat_gate_coverage.py
run "Metamorphic coverage (positive-invariant ledger)" ci/check_metamorphic_coverage.py
run "Metamorphic-gate self-test (proof-it-blocks)" ci/test_check_metamorphic_coverage.py
run "Fuzz-harness wiring/liveness" ci/check_fuzz_harness_wiring.py
run "Fuzz-wiring self-test (proof-it-blocks)" ci/test_check_fuzz_harness_wiring.py
run "dudect binary-CT detector self-test" ci/test_dudect_ct_probe.py
run "Backend value-differential coverage" ci/check_backend_value_differential.py
run "Backend value-diff self-test (proof-it-blocks)" ci/test_check_backend_value_differential.py
run "Fault-countermeasure coverage (sign-then-verify)" ci/check_fault_countermeasure_coverage.py
run "Fault-countermeasure self-test (proof-it-blocks)" ci/test_check_fault_countermeasure_coverage.py
run "Locked-map handle-escape (UAF class)" ci/check_locked_map_handle_escape.py
run "Handle-escape self-test (proof-it-blocks)" ci/test_check_locked_map_handle_escape.py
run "Entropy-source integrity (single-source CSPRNG)" ci/check_entropy_source_integrity.py
run "Entropy-source self-test (proof-it-blocks)" ci/test_check_entropy_source_integrity.py
run "Nonce erase coverage (BIP-327)"           ci/check_nonce_erase_coverage.py
run "Doc drift (badges, removed files)"        ci/check_doc_drift.py
run "Bench/doc consistency (banned patterns)" ci/check_bench_doc_consistency.py

# Regression gates for the 2026-05-29 read-only-review fixes:
#  - ABI count + names: docs/nuspec must match the real UFSECP_API surface (REL-04)
#  - Workflow trigger claims: docs must not call a workflow_dispatch-only workflow CI-enforced (CLAIM-07)
#  - Secret-erase coverage: every seckey-parsing shim fn must secure_erase (CT-04/RT-05)
run "ABI count + names (REL-04)"               ci/check_abi_count.py
run "ABI version sync (REL-ABI-MISMATCH)"      ci/check_abi_version_sync.py
run "Randomize claim consistency (RF-001)"     ci/check_randomize_claim_consistency.py
run "Required-checks match jobs (CAAS-CI-001)"  ci/check_required_checks_match_jobs.py
run "Sanitizer result assertions (CAAS6-01)"    ci/check_sanitizer_result_assertions.py
run "Sanitizer-assertion gate self-test"        ci/test_check_sanitizer_result_assertions.py
run "Reviewer-doc module counts (CLAIMS-001)"   ci/check_doc_module_counts.py
run "Build options doc sync"                    ci/gen_build_options.py --check
run "Build options gate self-test"              ci/test_gen_build_options.py
run "Workflow trigger claims (CLAIM-07)"       ci/check_workflow_trigger_claims.py
run "Secret-erase coverage (CT-04/RT-05)"      ci/check_secret_erase_coverage.py
run "Secret-erase self-test (proof-it-blocks)" ci/test_check_secret_erase_coverage.py

# Profile manifest cross-check: ci/profiles.json -> CMakePresets.json -> ci/caas_runner.py.
# Fast (<1s) — catches the class of bug where a chain-specific preset disables an
# optional module but a dependent module remains ON (e.g. LTC_SP needs BIP352)
# before the actual cmake --preset invocation fails at configure time.
run "Profile manifest consistency" ci/profile_manifest.py --quiet

run_sh "Advisory skip returns (Rule 16)" ci/check_advisory_skip_returns.sh
run "Advisory skip ceiling (TEST-004)"  ci/check_advisory_skip_ceiling.py
run "Test assertions (non-asserting probe scan)" ci/check_test_assertions.py
run "Section IDs consistency"        ci/check_section_ids.py

if [[ "${FAILED}" -gt 0 ]]; then
    echo ""
    echo "  \033[0;31m${FAILED} gate(s) FAILED\033[0m"
    exit 1
fi

echo "  \033[0;32mAll fast gates passed\033[0m"
