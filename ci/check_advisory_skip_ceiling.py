#!/usr/bin/env python3
"""
check_advisory_skip_ceiling.py — CI gate: advisory=true module count must not exceed ceiling.

Counts ALL_MODULES[] entries with advisory=true in audit/unified_audit_runner.cpp.
Each AuditModule struct uses positional initialization; the last field (bool advisory)
is `true` or `false`. This script counts entries where the last positional arg is `true`,
regardless of surrounding #if guards (reports the maximum possible count when all
optional features are compiled in).

Exit 0  = count is within the ceiling
Exit 1  = count exceeded the ceiling (new advisory modules added without review)
Exit 77 = unified_audit_runner.cpp not found (advisory skip)

Purpose (TEST-004):
  advisory=true modules skip silently in CI when their dependency (GPU, shim, etc.)
  is unavailable. A growing advisory count means growing silent coverage gaps.
  This gate enforces a reviewed ceiling — adding an advisory module requires
  incrementing ADVISORY_CEILING here.

When incrementing the ceiling:
  1. Add the new advisory=true module to ALL_MODULES[] in unified_audit_runner.cpp.
  2. Increment ADVISORY_CEILING below and update the comment.
  3. Document why the module cannot be made mandatory (shim absent, GPU absent, etc.)
     in the module's inline comment in unified_audit_runner.cpp.
"""
from __future__ import annotations
import re
import sys
from pathlib import Path

# Ceiling: maximum allowed advisory=true entries in ALL_MODULES[].
# Kept TIGHT (== the actual count) so a new advisory module — which skips silently in CI
# when its dependency (GPU/shim/etc.) is absent, i.e. a silent coverage gap — cannot be
# added without a deliberate, reviewed ceiling bump in the same commit.
# Re-tightened 2026-06-08 from 62 -> 58 (the prior 4-slot slack let 4 advisory modules
# be added unreviewed). Counted from unified_audit_runner.cpp.
# 2026-06-08: 58 -> 59 — added "gpu_zk_prove_verify_differential" (CPU range-proof →
# GPU Bulletproof poly-check consistency). Cannot be mandatory: needs a GPU with the ZK
# module compiled in; skips on no-GPU CI runners and on backends without SECP256K1_GPU_HAS_ZK.
# 2026-07-17: 59 -> 60 — registered "regression_schnorr_r_zero_ct" (SEC-006/SEC-007:
# shim secp256k1_schnorrsig_sign32/sign_custom R-zero check now uses a CT OR-accumulator,
# r_nonzero|=byte, instead of a variable-time for+break loop). The module was
# forward-declared 2026-05-12 but never wired into ALL_MODULES[] until the 2026-07-15
# acceptance-repair round; the ceiling was not bumped when it was wired in, so this
# closes that drift. Cannot be mandatory: depends on the libsecp256k1 shim being linked
# in (SHIM_WEAK symbols resolve to null / ctx_create returns null when the shim is
# absent, e.g. shim-less CI legs); the test self-detects that at runtime and returns
# ADVISORY_SKIP_CODE(77) — same runtime-detection pattern as the existing
# "regression_ecdh_xy64_erase" advisory module. Its non-shim twin,
# "regression_schnorr_ct_arithmetic" (HIGH-03/HIGH-06: native ufsecp_schnorr_sign r==0
# check, no shim dependency), remains advisory=false/mandatory — verified unchanged.
# 2026-07-18 (issue #335 round 6): 60 -> 61 — NOT a new module. Cross-verified this
# gate's own count against the independent structural parser in
# ci/check_audit_cwd_independence.py (built for the issue #335 CWD-independence sweep)
# and check_exploit_wiring.py's parse_all_modules(): both agree ALL_MODULES[] has 61
# advisory=true rows, not 60. count_advisory_modules()'s old regex (`,\s*true\s*\}`) did
# NOT strip comments first, so "fiat_crypto_link" -- whose row is
# `..., test_fiat_crypto_linkage_run, true  /* advisory=true: requires __int128 (MSVC
# skips with code 77) */ },` -- was silently undercounted: the inline block comment
# between `true` and `}` is not whitespace, so `\s*` never bridged it. The comment right
# above the old regex even said "skip optional trailing whitespace or comments" while
# the code only skipped whitespace -- a real bug, not a stale ceiling. Fixed by stripping
# comments before counting (see count_advisory_modules() below); ceiling raised to match
# the now-correct count, not because a new advisory module was added.
ADVISORY_CEILING: int = 61

# Frozen twin (CAAS meta-gate): bumping the ceiling requires updating BOTH constants,
# turning a silent loosening into a deliberate, diff-visible change. Mirrors
# RETROACTIVELY_COVERED_FROZEN_COUNT in ci/check_security_fix_has_test.py.
ADVISORY_CEILING_FROZEN: int = 61
assert ADVISORY_CEILING == ADVISORY_CEILING_FROZEN, (
    f"ADVISORY_CEILING ({ADVISORY_CEILING}) != ADVISORY_CEILING_FROZEN "
    f"({ADVISORY_CEILING_FROZEN}). Bump BOTH (and add advisory module(s) + docs) "
    "in the same commit — see ci/test_check_advisory_skip_ceiling.py."
)

RUNNER_PATH = Path("audit/unified_audit_runner.cpp")


def count_advisory_modules(runner: Path) -> int:
    """Count AuditModule entries with `true` as the last positional field.

    Comments are stripped BEFORE matching (2026-07-18 fix): a row can carry
    an inline `/* ... */` comment between `true` and the closing `}` (e.g.
    "fiat_crypto_link": `..., true  /* advisory=true: requires __int128 */ },`)
    and a comment is not whitespace, so a bare `,\\s*true\\s*\\}` pattern
    silently skips such rows -- undercounting by exactly the number of rows
    with a trailing inline comment. Matches the comment-stripping already
    used by check_exploit_wiring.py's parse_all_modules() and
    check_audit_cwd_independence.py's parse_all_modules_ordered(), so all
    three ALL_MODULES[] parsers in this repo now agree on the same count.
    """
    text = runner.read_text()
    stripped = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
    stripped = re.sub(r'//[^\n]*', '', stripped)

    # Locate the ALL_MODULES[] block
    start = stripped.find("static const AuditModule ALL_MODULES[]")
    end = stripped.find("static constexpr int NUM_MODULES", start)
    if start == -1 or end == -1:
        return -1
    block = stripped[start:end]

    # Each advisory entry ends with: ..., func_name, true }
    # Non-advisory entries have `false` in the same position — we do NOT count those.
    count = len(re.findall(r',\s*true\s*\}', block))
    return count


def main() -> int:
    if not RUNNER_PATH.exists():
        print(f"::notice::check_advisory_skip_ceiling: {RUNNER_PATH} not found — skip")
        return 77

    count = count_advisory_modules(RUNNER_PATH)
    if count == -1:
        print(f"::error::Could not locate ALL_MODULES[] block in {RUNNER_PATH}")
        return 1

    if count > ADVISORY_CEILING:
        print(f"::error::Advisory module count {count} exceeds ceiling {ADVISORY_CEILING}.")
        print(f"  New advisory=true modules were added without updating the ceiling.")
        print(f"  If intentional: increment ADVISORY_CEILING in ci/check_advisory_skip_ceiling.py")
        print(f"  and document why the module cannot be made mandatory.")
        return 1

    if count < ADVISORY_CEILING:
        print(f"Advisory modules: {count} (ceiling={ADVISORY_CEILING}).")
        print(f"  Tip: consider lowering ADVISORY_CEILING to {count} to keep the guard tight.")
        return 0

    print(f"Advisory module ceiling OK: {count}/{ADVISORY_CEILING}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
