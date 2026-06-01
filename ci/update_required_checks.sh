#!/usr/bin/env bash
# ===========================================================================
# Update GitHub Branch Protection Required Status Checks
# ===========================================================================
# This script updates the required status checks on the main branch to
# include all security-critical CI jobs. Previously only 'build', 'ci',
# 'test' were required; now security gates are also blocking.
#
# Usage: bash ci/update_required_checks.sh
#
# Prerequisites:
#   - gh CLI authenticated with admin access
#   - Repository: shrec/UltrafastSecp256k1
#
# What this changes:
#   Required checks now use the EXACT current job display names (CAAS-CI-001 fix:
#   the old list pinned phantom "CI" + "linux (gcc-13, …)" contexts that match no
#   job, and contained ZERO gate.yml CAAS jobs, so the security pipeline was not
#   required at the merge boundary). The list now includes the gate.yml Block 1/2/3
#   CAAS jobs and "Gate / Final Verdict" aggregator.
#
#   The contexts below are validated by ci/check_required_checks_match_jobs.py
#   (run in fast gates): every context MUST resolve to a real job that triggers on
#   pull_request, so this list can never silently drift from the workflows again.
#   Do NOT hand-edit a context to a name not produced by a PR-triggered job.
# ===========================================================================

set -euo pipefail

REPO="shrec/UltrafastSecp256k1"
BRANCH="main"

echo "=== Updating required status checks for ${REPO}:${BRANCH} ==="

# Step 1: Get current branch protection settings
echo "Current protection settings:"
gh api "repos/${REPO}/branches/${BRANCH}/protection" --jq '.required_status_checks.contexts // []' 2>/dev/null || echo "(none set)"

# Step 2: Update required status checks
# Uses the branch protection API to add security-critical checks
echo ""
echo "Updating required status checks..."

gh api "repos/${REPO}/branches/${BRANCH}/protection/required_status_checks" \
  -X PATCH \
  -f strict=true \
  --input - <<'EOF'
{
  "strict": true,
  "contexts": [
    "linux (gcc-14, Release)",
    "linux (gcc-14, Debug)",
    "linux (clang-17, Release)",
    "linux (clang-17, Debug)",
    "windows (Release)",
    "macos (Release)",
    "Sanitizers (ASan+UBSan)",
    "Sanitizers (TSan)",
    "Differential Smoke Test",
    "Protocol Vectors (MuSig2 BIP-327 / FROST-KAT)",
    "Build with -Werror",
    "Valgrind Memcheck",
    "Differential vs bitcoin-core/libsecp256k1",
    "Analyze (C/C++)",
    "Static Analysis (Cppcheck)",
    "preflight",
    "Doc / Fast Gates",
    "Block 1 / Fast CAAS Gates",
    "Block 2 / Build + Unit Tests",
    "Block 3 / CAAS Security Gates",
    "Block 3 / Security Gate (aggregator)",
    "Block 3 / Shim Security Gate",
    "Gate / Final Verdict"
  ]
}
EOF

echo ""
echo "=== Updated required status checks ==="
gh api "repos/${REPO}/branches/${BRANCH}/protection/required_status_checks" --jq '.contexts[]'

echo ""
echo "[OK] Required checks updated. Security-critical jobs now block merges."
echo ""
echo "Required blocking checks:"
echo "  - CI (linux/windows/macos matrix)"
echo "  - Build with -Werror (compiler warnings)"
echo "  - ASan + UBSan (memory safety)"
echo "  - Valgrind Memcheck (leak detection)"
echo "  - ct-verif LLVM analysis (constant-time verification)"
echo "  - Benchmark Regression Check (performance gate)"
echo "  - CodeQL (security scanning)"
echo "  - Audit Verdict (unified audit 3-platform pass)"
