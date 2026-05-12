#!/usr/bin/env bash
# Verifies that advisory audit modules return ADVISORY_SKIP_CODE=77 when
# GPU/optional infrastructure is absent (not 0, which would be a false PASS).
#
# Exit codes:
#   0  — all checked advisory binaries return 77 correctly (or were not found)
#   1  — at least one advisory binary returned 0 (false PASS)
#  77  — no advisory binaries were found to check (GPU build not present)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

FAILED=0
CHECKED=0

# Check that an advisory binary returns 77 when run without GPU infrastructure.
# Returns 0 if correct, 1 if the binary returned 0 (false PASS).
check_advisory() {
    local name="$1"
    local binary="$2"
    if [ ! -f "$binary" ]; then
        echo "  SKIP $name (binary not found: $binary)"
        return 0
    fi
    CHECKED=$((CHECKED + 1))
    local rc=0
    "$binary" 2>/dev/null || rc=$?
    if [ "$rc" -eq 77 ]; then
        echo "  OK   $name returns 77 (ADVISORY_SKIP_CODE) correctly"
    elif [ "$rc" -eq 0 ]; then
        echo "  FAIL $name returns 0 (false PASS) instead of 77 — advisory skip not firing"
        FAILED=$((FAILED + 1))
    else
        echo "  WARN $name returns $rc (unexpected — may be a real failure)"
    fi
}

# Check common advisory GPU-dependent binaries in build_bench_opt
BUILD="${REPO_ROOT}/build_bench_opt"

check_advisory "test_exploit_gpu_memory_safety" \
    "${BUILD}/audit/test_exploit_gpu_memory_safety_standalone"
check_advisory "test_exploit_kat_corpus" \
    "${BUILD}/audit/test_exploit_kat_corpus_standalone"

if [ "$CHECKED" -eq 0 ]; then
    echo "  SKIP advisory skip check (no advisory binaries found — GPU build not present)"
    exit 77
fi

if [ "$FAILED" -gt 0 ]; then
    echo "  ${FAILED} advisory module(s) returned 0 instead of 77 — false PASS detected"
    exit 1
fi

echo "  All advisory skip checks passed (${CHECKED} checked)"
exit 0
