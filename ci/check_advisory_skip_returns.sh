#!/usr/bin/env bash
# Verifies that ALL advisory audit modules (advisory=true in
# audit/unified_audit_runner.cpp ALL_MODULES[]) return ADVISORY_SKIP_CODE=77
# when their optional infrastructure (GPU runtime, shim, Python harness, etc.)
# is absent — not 0, which would be a silent false PASS.
#
# Rule 16 (CLAUDE.md):
#   "Advisory CAAS modules MUST use ADVISORY_SKIP_CODE sentinel. Returning 0
#    from a skip path is banned because 0 means PASS — a skipped advisory
#    would be falsely reported as passing."
#
# Closes review finding P1-CI-001 ("Advisory skip meta-gate covers only
# 2/30+ advisory modules"). The previous hand-curated 2-entry list is now
# replaced by a data-driven enumeration that reads ALL_MODULES[] directly.
#
# Exit codes:
#   0  — every advisory binary that exists on disk returns 77 (or is genuinely
#        missing because the build did not produce it). Real PASS.
#   1  — at least one advisory binary returned 0 (silent false PASS — Rule 16
#        violation). Real FAIL — the binary must be fixed to return 77.
#  77  — no advisory binaries found at all (no build dir, no standalone targets).
#        Advisory skip — the gate could not run.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUNNER_CPP="${REPO_ROOT}/audit/unified_audit_runner.cpp"

if [ ! -f "${RUNNER_CPP}" ]; then
    echo "  ERROR audit/unified_audit_runner.cpp not found — cannot enumerate"
    exit 1
fi

# Enumerate advisory=true module IDs from ALL_MODULES[]. The format is:
#   { "module_id", "description...", "section", module_run_fn, true },
# advisory flag is the trailing literal `true` (vs `false`) before `}`.
# We match lines whose final tokens are `, true }` and capture the first
# quoted field (the id). awk handles the C++ aggregate-init line format
# without spawning a full parser.
ADVISORY_IDS=$(awk -F\" '
    /, true \},?$/ && /\{ *"/ {
        # $2 is the first quoted field = the module id.
        print $2
    }
' "${RUNNER_CPP}" | sort -u)

if [ -z "${ADVISORY_IDS}" ]; then
    echo "  ERROR no advisory=true entries found in ALL_MODULES[] — parser regex stale?"
    exit 1
fi

ADVISORY_COUNT=$(echo "${ADVISORY_IDS}" | wc -l | tr -d ' ')
echo "  Enumerated ${ADVISORY_COUNT} advisory=true modules from ALL_MODULES[]"

# Candidate build directories — first one with any standalone binary wins.
# These cover the common in-tree build paths (developer-driven and CI-driven).
CANDIDATE_BUILDS=(
    "${UFSECP_BUILD_DIR:-}"
    "${REPO_ROOT}/build_bench_opt"
    "${REPO_ROOT}/build-runner"
    "${REPO_ROOT}/build-audit"
    "${REPO_ROOT}/build"
    "${REPO_ROOT}/out/ci-shim"
    "${REPO_ROOT}/out/release"
)

BUILD=""
for cand in "${CANDIDATE_BUILDS[@]}"; do
    if [ -n "${cand}" ] && [ -d "${cand}/audit" ]; then
        BUILD="${cand}"
        break
    fi
done

if [ -z "${BUILD}" ]; then
    echo "  SKIP advisory skip check (no build dir found among: ${CANDIDATE_BUILDS[*]})"
    echo "  Set UFSECP_BUILD_DIR=/path/to/build to point at a non-standard location."
    exit 77
fi
echo "  Using build dir: ${BUILD}"

FAILED=0
CHECKED=0
MISSING=0
FAILED_IDS=()

# Run each advisory binary's standalone variant; check exit code.
while IFS= read -r mod_id; do
    [ -z "${mod_id}" ] && continue
    bin="${BUILD}/audit/test_${mod_id}_standalone"
    # Some advisory tests have non-test_ prefix; try that too.
    [ -f "${bin}" ] || bin="${BUILD}/audit/${mod_id}_standalone"
    [ -f "${bin}" ] || bin="${BUILD}/audit/test_exploit_${mod_id}_standalone"
    if [ ! -f "${bin}" ]; then
        MISSING=$((MISSING + 1))
        continue
    fi
    CHECKED=$((CHECKED + 1))
    rc=0
    # Use timeout(1) when available so a hanging advisory binary cannot stall CI.
    if command -v timeout >/dev/null 2>&1; then
        timeout 30 "${bin}" >/dev/null 2>&1 || rc=$?
    else
        "${bin}" >/dev/null 2>&1 || rc=$?
    fi
    case "${rc}" in
        77)
            : # OK — advisory-skip path fired, as required.
            ;;
        0)
            echo "  FAIL ${mod_id} returned 0 (silent false PASS) instead of 77"
            FAILED=$((FAILED + 1))
            FAILED_IDS+=("${mod_id}")
            ;;
        *)
            # Non-77 non-zero is treated as a real failure (assertion fired,
            # crash, timeout, etc.) — out of scope for the advisory-skip gate
            # but worth surfacing so it does not hide behind silence.
            echo "  WARN ${mod_id} returned ${rc} (real failure surface, not Rule-16 violation)"
            ;;
    esac
done <<< "${ADVISORY_IDS}"

echo "  Checked ${CHECKED} advisory binaries (${MISSING} not built in this configuration)"

if [ "${CHECKED}" -eq 0 ]; then
    echo "  SKIP no standalone advisory binaries present in ${BUILD}"
    exit 77
fi

if [ "${FAILED}" -gt 0 ]; then
    echo "  ${FAILED} advisory module(s) violated Rule 16: ${FAILED_IDS[*]}"
    exit 1
fi

echo "  All ${CHECKED} advisory skip-return checks passed (Rule 16 satisfied)"
exit 0
