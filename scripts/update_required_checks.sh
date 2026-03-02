#!/usr/bin/env bash
# ===========================================================================
# Update GitHub Branch Protection Required Status Checks
# ===========================================================================
# This script updates the required status checks on the main branch to
# include all security-critical CI jobs. Previously only 'build', 'ci',
# 'test' were required; now security gates are also blocking.
#
# Usage: bash scripts/update_required_checks.sh
#
# Prerequisites:
#   - gh CLI authenticated with admin access
#   - Repository: shrec/UltrafastSecp256k1
#
# What this changes:
#   BEFORE: required checks = [build, ci, test]
#   AFTER:  required checks = [build, ci, test, security-audit, codeql, ct-verif, bench-gate, audit-verdict]
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
    "CI",
    "linux (gcc-13, Release)",
    "linux (gcc-13, Debug)",
    "linux (clang-17, Release)",
    "linux (clang-17, Debug)",
    "windows (Release)",
    "macos (Release)",
    "Sanitizers (ASan+UBSan)",
    "Sanitizers (TSan)",
    "Build with -Werror",
    "ASan + UBSan",
    "Valgrind Memcheck",
    "ct-verif LLVM analysis",
    "Benchmark Regression Check",
    "CodeQL",
    "Audit Verdict"
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
