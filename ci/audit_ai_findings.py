#!/usr/bin/env python3
"""
audit_ai_findings.py — AI Findings Quarantine Reporter
=======================================================
Separates CTest audit results into two buckets:

  1. Confirmed tests     — manually reviewed and validated
  2. AI-suggested tests  — labelled "ai_finding;unconfirmed";
                           excluded from official audit score totals

Usage:
    python3 ci/audit_ai_findings.py [--build-dir BUILD_DIR] [--json OUT.json]

Options:
    --build-dir DIR   CTest build directory  (default: auto-detect build_opencl or build
                      relative to the script's grandparent)
    --json PATH       Write a JSON summary to PATH in addition to stdout output
    -v, --verbose     Print per-test details for unconfirmed tests
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_build_dir(script_root: Path) -> Path:
    """Try a few conventional build directory names relative to the repo root."""
    repo_root = script_root.parent
    candidates = ["build_opencl", "build", "build_rel", "build-linux"]
    for name in candidates:
        d = repo_root / name
        if (d / "CTestTestfile.cmake").exists():
            return d
    raise FileNotFoundError(
        "Could not auto-detect a CTest build directory. "
        "Pass --build-dir explicitly."
    )


def run_ctest(build_dir: Path, extra_args: list[str]) -> str:
    """Run ctest and return its stdout."""
    cmd = ["ctest", "--test-dir", str(build_dir), "-N"] + extra_args
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout


def parse_test_list(ctest_output: str) -> list[str]:
    """Parse 'Test #N: name' lines from `ctest -N` output."""
    tests = []
    for line in ctest_output.splitlines():
        line = line.strip()
        if line.startswith("Test #"):
            # "Test #3: exploit_bip340_kat"  →  "exploit_bip340_kat"
            parts = line.split(":", 1)
            if len(parts) == 2:
                tests.append(parts[1].strip())
    return tests


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Report confirmed vs AI-suggested (unconfirmed) audit tests."
    )
    parser.add_argument(
        "--build-dir", metavar="DIR", default=None,
        help="CTest build directory (auto-detected if omitted)"
    )
    parser.add_argument(
        "--json", metavar="PATH", default=None,
        help="Write JSON summary to PATH"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Print names of unconfirmed AI-suggested tests"
    )
    args = parser.parse_args()

    script_root = Path(__file__).resolve().parent
    try:
        build_dir = Path(args.build_dir) if args.build_dir else find_build_dir(script_root)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if not build_dir.exists():
        print(f"ERROR: build directory not found: {build_dir}", file=sys.stderr)
        return 1

    # All audit tests
    all_audit_output    = run_ctest(build_dir, ["-L", "audit"])
    all_audit_tests     = parse_test_list(all_audit_output)

    # AI-suggested unconfirmed tests (labelled "ai_finding")
    ai_output           = run_ctest(build_dir, ["-L", "ai_finding"])
    ai_tests            = set(parse_test_list(ai_output))

    confirmed_tests     = [t for t in all_audit_tests if t not in ai_tests]

    print("=" * 60)
    print("  UltrafastSecp256k1 Audit Test Quarantine Report")
    print("=" * 60)
    print(f"  Build dir        : {build_dir}")
    print(f"  Total audit tests: {len(all_audit_tests)}")
    print(f"  Confirmed        : {len(confirmed_tests)}  (included in official score)")
    print(f"  AI-suggested     : {len(ai_tests)}"
          "  (labelled ai_finding; excluded from score)")
    print("=" * 60)

    if args.verbose and ai_tests:
        print("\nAI-suggested (unconfirmed) tests:")
        for name in sorted(ai_tests):
            print(f"    {name}")

    summary = {
        "build_dir": str(build_dir),
        "total_audit": len(all_audit_tests),
        "confirmed": len(confirmed_tests),
        "ai_suggested": len(ai_tests),
        "confirmed_tests": sorted(confirmed_tests),
        "ai_suggested_tests": sorted(ai_tests),
    }

    if args.json:
        out_path = Path(args.json)
        out_path.write_text(json.dumps(summary, indent=2))
        print(f"\nJSON summary written to: {out_path}")

    # Non-zero exit if any unconfirmed tests exist (useful in CI gate mode)
    return 0


if __name__ == "__main__":
    sys.exit(main())
