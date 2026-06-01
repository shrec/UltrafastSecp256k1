#!/usr/bin/env bash
# Rule 16 enforcement wrapper used by preflight.yml, gate.yml (shim-gate), and
# release.yml. Invokes ci/check_advisory_skip_returns.sh and treats:
#   rc=0  → notice  : all advisory modules return 77 (enforcement passed)
#   rc=77 → warning : no standalone advisory binaries built in this configuration
#                     (legitimate skip — the build only produces unified_audit_runner
#                     and not the per-module standalone executables that the
#                     script enumerates). Enforcement is exercised in build
#                     configurations that DO produce standalone targets.
#   rc=*  → error   : at least one advisory module returned a non-77 non-zero
#                     value (Rule 16 violation: rc=0 = silent false PASS).
#
# Extracted 2026-05-24 to remove the identical inline wrapper across three
# workflows (SonarCloud Quality Gate flagged "9.1% duplication on new code").
#
# CAAS-CI-002 fix: the standalone-binary path (check_advisory_skip_returns.sh)
# permanently soft-skips at every CI choke point because those jobs build only
# ufsecp_shared / unified_audit_runner — the per-module *_standalone executables
# it enumerates are never compiled. Previously rc=77 emitted a benign ::warning,
# so the gate enforced NOTHING. Now, when no standalone binaries exist, we fall
# back to ci/check_advisory_json_rule16.py, which enforces Rule 16 directly from
# the unified_audit_runner JSON report (produced at every choke point). Only if
# BOTH paths cannot run do we soft-skip — and that is escalated to ::error when
# RULE16_STRICT=1 (set in release-grade jobs) so a true no-op is never green.
SCRIPT_DIR="$(dirname "$0")"
set +e
bash "${SCRIPT_DIR}/check_advisory_skip_returns.sh"
rc=$?
set -e
if [ "$rc" -eq 0 ]; then
    echo "::notice::Rule 16: all advisory standalone binaries return 77 (enforcement passed)"
elif [ "$rc" -eq 77 ]; then
    echo "Rule 16: no standalone advisory binaries built — falling back to JSON report enforcement"
    set +e
    python3 "${SCRIPT_DIR}/check_advisory_json_rule16.py" "${RULE16_AUDIT_REPORT:-}"
    jrc=$?
    set -e
    if [ "$jrc" -eq 0 ]; then
        echo "::notice::Rule 16: enforced via unified_audit_runner JSON report (no standalone binaries needed)"
    elif [ "$jrc" -eq 77 ]; then
        if [ "${RULE16_STRICT:-0}" = "1" ]; then
            echo "::error::Rule 16: neither standalone binaries NOR a JSON report were available in a release-grade job — enforcement could not run (RULE16_STRICT=1)"
            exit 1
        fi
        echo "::warning::Rule 16: no standalone binaries and no JSON report in this configuration; gate soft-skipped"
    else
        echo "::error::Rule 16: JSON report enforcement found a violation (advisory false-PASS or real failure)"
        exit "$jrc"
    fi
else
    echo "::error::Rule 16: at least one advisory standalone binary returned a non-77 non-zero value (0 = silent false PASS)"
    exit "$rc"
fi
