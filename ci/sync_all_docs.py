#!/usr/bin/env python3
"""
sync_all_docs.py — Single central entry-point: propagate EVERY canonical value
into all documentation / metadata, from the canonical sources of truth.

This is the orchestrator that the individual sync scripts tell you to run
(sync_module_count.py, sync_canonical_numbers.py, sync_docs_from_canonical.py
all print "Run: python3 ci/sync_all_docs.py"). It chains them in dependency
order so nothing is ever hand-edited and nothing drifts.

Canonical sources of truth (never edit derived values in docs by hand):
  VERSION.txt                         → library version string (all manifests + docs)
  audit/unified_audit_runner.cpp      → module / exploit-PoC counts
  .github/workflows/*.yml             → CI workflow + CT-pipeline counts
  docs/BITCOIN_CORE_TEST_RESULTS.json → Bitcoin Core test pass/total
  docs/canonical_data.json            → derived counts (built from the above)
  docs/canonical_numbers.json         → benchmark ratios / wording

Usage:
    python3 ci/sync_all_docs.py            # APPLY: rebuild canonical + propagate everywhere
    python3 ci/sync_all_docs.py --check    # CI gate: report drift, exit 1 if anything is stale
    python3 ci/sync_all_docs.py --verbose  # stream every sub-step's full output

The --check mode is read-only (only --dry-run / --check sub-commands run); it
writes nothing and is safe to wire into CI.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

CI_DIR = Path(__file__).resolve().parent
ROOT = CI_DIR.parent
PY = sys.executable or "python3"


# Each step: (label, apply_argv, check_argv_or_None)
#   apply_argv  — command that WRITES the propagated values
#   check_argv  — read-only command that exits non-zero on drift (None = skip in --check)
#
# Order matters: canonical_data.json is rebuilt first so every downstream
# propagation reads fresh derived counts.
STEPS: list[tuple[str, list[str], list[str] | None]] = [
    ("canonical_data.json (rebuild from source)",
     ["ci/build_canonical_data.py"],
     ["ci/build_canonical_data.py", "--dry-run"]),

    ("version strings  (VERSION.txt → manifests + docs)",
     ["ci/sync_version_refs.py"],
     # sync_version_refs.py --dry-run does not signal drift via exit code;
     # check_version_sync.py is the authoritative read-only version gate.
     ["ci/check_version_sync.py", "--version-only"]),

    ("derived counts  (canonical_data.json → docs)",
     ["ci/sync_docs_from_canonical.py"],
     ["ci/sync_docs_from_canonical.py", "--dry-run"]),

    ("benchmark numbers  (canonical_numbers.json → docs)",
     ["ci/sync_canonical_numbers.py"],
     ["ci/sync_canonical_numbers.py", "--dry-run"]),

    ("module / exploit-PoC counts  (unified_audit_runner.cpp → docs)",
     ["ci/sync_module_count.py"],
     ["ci/sync_module_count.py", "--check"]),

    ("doc consistency  (version + counts + workflows + ABI)",
     ["ci/sync_docs.py"],
     ["ci/sync_docs.py", "--check"]),

    ("AUDIT_REPORT.md release-version marker",
     ["ci/sync_audit_report_version.py"],
     # No dry-run mode; the count gate (check_version_sync.py) already runs.
     None),
]


def _run(argv: list[str], verbose: bool) -> int:
    cmd = [PY] + argv
    if verbose:
        return subprocess.run(cmd, cwd=ROOT).returncode
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        # On failure, surface the captured output so the operator can see why.
        sys.stdout.write(proc.stdout)
        sys.stderr.write(proc.stderr)
    return proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true",
                        help="Read-only CI gate: report drift and exit 1 if any doc is stale")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Stream each sub-step's full output")
    args = parser.parse_args()

    mode = "CHECK (read-only)" if args.check else "APPLY"
    print(f"── sync_all_docs.py [{mode}] ─ canonical → docs ──────────────────────")

    failures: list[str] = []
    for label, apply_argv, check_argv in STEPS:
        argv = check_argv if args.check else apply_argv
        if argv is None:
            print(f"  SKIP  {label}  (no read-only mode)")
            continue
        print(f"\n▶ {label}")
        rc = _run(argv, args.verbose)
        if rc != 0:
            failures.append(label)
            print(f"  ✗ {'DRIFT' if args.check else 'FAILED'} ({' '.join(argv)} → rc={rc})")
        else:
            print(f"  ✓ ok")

    print("\n" + "─" * 70)
    if failures:
        verb = "out of sync" if args.check else "failed"
        print(f"  {len(failures)} step(s) {verb}:")
        for f in failures:
            print(f"    - {f}")
        if args.check:
            print("\n  Fix: python3 ci/sync_all_docs.py   (then commit the updated files)")
        return 1

    print("  All canonical values are in sync." if args.check
          else "  Propagated all canonical values into docs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
