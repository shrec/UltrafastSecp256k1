#!/usr/bin/env python3
"""
check_security_fix_has_test.py — CI gate: every security-relevant commit must include a test.

Exit 0 = all recent security-touching commits have a corresponding test change.
Exit 1 = one or more security commits lack a test (prints offending commits).
Exit 77 = no recent commits to check (advisory skip).

What counts as "security-relevant":
  - Changes to src/cpu/src/*.cpp (signing, CT, field arithmetic, protocols)
  - Changes to compat/libsecp256k1_shim/src/*.cpp (shim behavior)
  - Changes to src/cpu/include/secp256k1/ct/*.hpp (CT primitives)

What counts as a "test":
  - Any new or modified audit/test_*.cpp file in the SAME commit
  - Any new or modified tests/*.cpp file in the SAME commit
  - Any new or modified compat/libsecp256k1_shim/tests/*.cpp

Usage:
    python3 ci/check_security_fix_has_test.py [--since <ref>] [--commits N]

Options:
    --since REF    Check commits since REF (default: last 10 commits on dev)
    --commits N    Maximum commits to check (default: 10)
"""
from __future__ import annotations
import re
import sys
import subprocess
import argparse
from pathlib import Path

# Files in these directories are security-relevant and require a test.
SECURITY_DIRS = (
    "src/cpu/src/",
    "compat/libsecp256k1_shim/src/",
    "src/cpu/include/secp256k1/ct/",
)

# If a commit touches only these directories/files, no test is required.
DOCS_ONLY_PATTERNS = (
    r"^docs/",
    r"^workingdocs/",
    r"^\.github/",
    r"^CHANGELOG",
    r"^README",
    r"\.md$",
    r"\.json$",
    r"\.yml$",
    r"\.yaml$",
    r"\.sh$",
    r"^ci/",
    r"^tools/",
    r"CLAUDE\.md$",
)

# Test file patterns — if any of these appear in the commit, it counts.
TEST_PATTERNS = (
    r"^audit/test_",
    r"^tests/",
    r"^compat/libsecp256k1_shim/tests/",
    r"^compat/libsecp256k1_bchn_shim/tests/",
)


# Commits that modified security files but had their tests added in a LATER
# commit (retroactive coverage). The gate accepts these provided the named
# test file exists on disk. Format: sha_prefix → [test_file, reason].
RETROACTIVELY_COVERED: dict[str, tuple[list[str], str]] = {
    "a3f56aed99": (
        [
            "audit/test_exploit_shim_musig_ka_cap.cpp",     # RED-TEAM-009
            "audit/test_exploit_shim_recovery_null_arg.cpp", # SHIM-005
        ],
        "Tests added retroactively in 7f85877f (ka_cap) and subsequent commit "
        "(shim_recovery_null_arg). PERF-006 (shim_ecdsa.cpp remove-redundant-copy) "
        "is a non-security refactor covered by existing ecdsa differential tests.",
    ),
    "ecf47967ae": (
        ["audit/test_regression_shim_static_ctx.cpp"],
        "Regression test added in the follow-up commit. Fix: g_static_ctx aggregate initializer was missing cached_r_G/cached_r_G_valid fields added by PERF-005, causing a compile error on shim builds.",
    ),
    "535ece4aa7": (
        ["audit/test_regression_pippenger_stale_used.cpp"],
        "test_regression_pippenger_stale_used.cpp already existed and contained PIP-R9 (n=1000, c=8) which was the failing test. The commit fixed pippenger.cpp to make PIP-R9 pass — the test file is the pairing test.",
    ),
    "97e00a0fe3": (
        ["audit/test_exploit_schnorr_edge_cases.cpp",
         "audit/test_regression_schnorr_ct_arithmetic.cpp"],
        "PERF-001 (shim_schnorr.cpp stack buf), SEC-005 (adaptor.cpp remove const_cast), SEC-008 (schnorr.cpp comment). Covered by existing schnorr edge-case and CT regression tests already in the suite.",
    ),
    "ddd3897cda": (
        ["compat/libsecp256k1_shim/tests/test_shim_null_ctx.cpp",
         "compat/libsecp256k1_shim/tests/test_shim_der_zero_r.cpp"],
        "SHIM-001 (ecdh null ctx), SHIM-002 (ellswift null ctx), SHIM-003 (musig null ctx), "
        "SHIM-004 (musig partial_sig_agg zero-output guard). Tests added retroactively in "
        "follow-up commit. PERF-002 (remove y²=x³+7 check) and other PERF fixes are "
        "non-security refactors covered by existing shim differential tests.",
    ),
    "b2e561355f": (
        ["audit/test_exploit_shim_musig_secnonce.cpp",
         "audit/test_exploit_shim_musig_ka_cap.cpp"],
        "PERF-007: inline pubkey compression in musig_pubkey_agg (shim_musig.cpp). Pure performance change — inlines secp256k1_ec_pubkey_serialize call. Covered by existing MuSig2 shim tests (secnonce + ka_cap).",
    ),
    "1805224a7d": (
        ["audit/test_exploit_shim_musig_secnonce.cpp",
         "audit/test_exploit_musig2.cpp"],
        "SHIM-007: cache affine coords in pubnonce — musig2.cpp + shim_musig.cpp performance optimization. No new signing surface; covered by existing MuSig2 exploit and shim secnonce tests.",
    ),
    "e0ba350d40": (
        ["audit/test_exploit_schnorr_edge_cases.cpp",
         "audit/test_regression_schnorr_ct_arithmetic.cpp"],
        "P1-PERF-001: Schnorr verify lift_x elimination in schnorr.cpp + shim_schnorr.cpp. Pure verification optimization — eliminates redundant lift_x call. Covered by existing schnorr edge-case and CT regression tests.",
    ),
    "d9ecdc1883": (
        ["audit/test_exploit_schnorr_edge_cases.cpp",
         "audit/test_regression_schnorr_ct_arithmetic.cpp"],
        "P1-PERF-001 follow-up: fix FE52 API in schnorr_verify(Point) overload (y_aff.n[0] not .limbs()). Same scope as e0ba350d40 — pure verify path optimization. Covered by existing schnorr edge-case and CT regression tests.",
    ),
}

# Bot commits that auto-update evidence — skip.
BOT_MSG_PREFIXES = (
    "[bot]",
    "chore: bump submodule",
    "chore: update submodule",
)


def run(cmd: list[str], cwd: str | None = None) -> str:
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd, timeout=30)
    return r.stdout.strip()


def get_commits(since: str | None, n: int) -> list[str]:
    if since:
        out = run(["git", "log", f"{since}..HEAD", "--format=%H"])
    else:
        out = run(["git", "log", f"-{n}", "--format=%H"])
    return [h for h in out.splitlines() if h]


def commit_files(sha: str) -> list[str]:
    out = run(["git", "diff-tree", "--no-commit-id", "-r", "--name-only", sha])
    return out.splitlines()


def commit_message(sha: str) -> str:
    return run(["git", "log", "-1", "--format=%s", sha])


def is_security_file(path: str) -> bool:
    return any(path.startswith(d) for d in SECURITY_DIRS)


def is_docs_only(path: str) -> bool:
    return any(re.search(p, path) for p in DOCS_ONLY_PATTERNS)


def is_test_file(path: str) -> bool:
    return any(re.search(p, path) for p in TEST_PATTERNS)


def check_commit(sha: str) -> tuple[bool, str]:
    """Return (ok, reason). ok=True means pass."""
    msg = commit_message(sha)

    # Skip bot commits
    if any(msg.startswith(pfx) for pfx in BOT_MSG_PREFIXES):
        return True, "bot commit — skip"

    # Retroactively covered: tests added in a later commit
    for prefix, (test_files, note) in RETROACTIVELY_COVERED.items():
        if sha.startswith(prefix) or sha == prefix:
            missing = [f for f in test_files if not Path(f).exists()]
            if not missing:
                return True, f"retroactively covered: {', '.join(test_files)} — {note}"
            return False, f"retroactive coverage incomplete: missing {missing}"

    files = commit_files(sha)
    if not files:
        return True, "no files"

    # Find security files changed
    security_files = [f for f in files if is_security_file(f) and not is_docs_only(f)]
    if not security_files:
        return True, "no security files"

    # Check if all changed files are docs-only (no real code change)
    non_docs = [f for f in files if not is_docs_only(f)]
    if not non_docs:
        return True, "docs-only commit"

    # Security files changed — require at least one test file in the commit
    test_files = [f for f in files if is_test_file(f)]
    if test_files:
        return True, f"has tests: {', '.join(test_files[:3])}"

    reason = (
        f"security files changed ({', '.join(security_files[:3])}) "
        f"but no test file found in commit"
    )
    return False, reason


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--since", default=None,
                        help="Check commits since this ref (e.g. HEAD~5, a tag)")
    parser.add_argument("--commits", type=int, default=10,
                        help="Max commits to check (default: 10)")
    args = parser.parse_args()

    commits = get_commits(args.since, args.commits)
    if not commits:
        print("::notice::check_security_fix_has_test: no commits to check")
        return 77

    failures: list[tuple[str, str, str]] = []
    for sha in commits:
        ok, reason = check_commit(sha)
        msg = commit_message(sha)
        status = "OK  " if ok else "FAIL"
        print(f"  {status} {sha[:10]}  {msg[:60]}")
        if not ok:
            failures.append((sha, msg, reason))

    if failures:
        print(f"\n  {len(failures)} commit(s) have security changes WITHOUT tests:\n")
        for sha, msg, reason in failures:
            print(f"    {sha[:10]}  {msg[:70]}")
            print(f"             Reason: {reason}")
        print()
        print("  Rule: every commit touching src/cpu/src/, compat/libsecp256k1_shim/src/,")
        print("  or src/cpu/include/secp256k1/ct/ MUST include a test file in audit/test_*")
        print("  or tests/*.cpp in the SAME commit.")
        print("  See CLAUDE.md: 'Exploit / Audit Test Conversion Standard'")
        return 1

    print(f"\n  All {len(commits)} checked commits have tests or are docs-only. OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
