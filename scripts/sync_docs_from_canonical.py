#!/usr/bin/env python3
"""
sync_docs_from_canonical.py
============================
Reads docs/canonical_data.json and propagates every value into all
documentation files that reference it.

Usage:
    python3 scripts/sync_docs_from_canonical.py            # apply updates
    python3 scripts/sync_docs_from_canonical.py --dry-run  # report drift, exit 1 if any

Never edit doc numbers manually — run this script instead.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT       = SCRIPT_DIR.parent
CANON_FILE = ROOT / "docs" / "canonical_data.json"

# ---------------------------------------------------------------------------
# Replacement rule registry
#
# Each entry: (file_path_relative, search_regex, replacement_template)
# The template may reference named groups from the regex or {KEY} placeholders
# from canonical_data.json.
# ---------------------------------------------------------------------------

_RULES: list[tuple[str, str, str]] = [

    # ── VERSION ────────────────────────────────────────────────────────────
    ("VERSION.txt",
     r"\d+\.\d+\.\d+",
     "{version}"),

    # ── README.md ──────────────────────────────────────────────────────────
    ("README.md",
     r"\d+ exploit-PoC test files \(all \d+ registered",
     "{exploit_poc_count} exploit-PoC test files (all {exploit_poc_count} registered"),

    ("README.md",
     r"\d+ CI workflows, 5 CT verification pipelines",
     "{ci_workflow_count} CI workflows, {ct_pipeline_count} CT verification pipelines"),

    ("README.md",
     r"1M\+ assertions, \d+ CI workflows",
     "1M+ assertions, {ci_workflow_count} CI workflows"),

    ("README.md",
     r"\d+ independent \(LLVM ct-verif.*?ARM64 native\)",
     "{ct_pipeline_count} independent (LLVM ct-verif, Valgrind taint, ct-prover, dudect, ARM64 native)"),

    # ── WHY_ULTRAFASTSECP256K1.md ──────────────────────────────────────────
    ("WHY_ULTRAFASTSECP256K1.md",
     r"\d+ dedicated adversarial PoC modules across 20\+ coverage categories \(`audit/test_exploit_\*\.cpp`\) \| \d+ test files",
     "{exploit_poc_count} dedicated adversarial PoC modules across 20+ coverage categories (`audit/test_exploit_*.cpp`) | {exploit_poc_count} test files"),

    ("WHY_ULTRAFASTSECP256K1.md",
     r"\*\*unified_audit_runner\*\* across \d+ non-exploit modules \+ \d+ exploit-PoC modules \(\d+ total\)",
     "**unified_audit_runner** across {non_exploit_modules} non-exploit modules + {exploit_poc_count} exploit-PoC modules ({total_modules} total)"),

    ("WHY_ULTRAFASTSECP256K1.md",
     r"\*\*\d+ exploit-style PoC modules\*\* across 20\+ coverage categories.*?\| \*\*\d+ modules, 0 failures\*\*",
     "**{exploit_poc_count} exploit-style PoC modules** across 20+ coverage categories, all in `audit/test_exploit_*.cpp` | **{exploit_poc_count} modules, 0 failures**"),

    ("WHY_ULTRAFASTSECP256K1.md",
     r"\d+ workflows, \d+ platform combinations",
     "{ci_workflow_count} workflows, 16 platform combinations"),

    ("WHY_ULTRAFASTSECP256K1.md",
     r"\d+ independent pipelines \(LLVM ct-verif.*?ARM64 native\)",
     "{ct_pipeline_count} independent pipelines (LLVM ct-verif, Valgrind taint, ct-prover, dudect, ARM64 native)"),

    # ── AUDIT_COVERAGE.md (root) ────────────────────────────────────────────
    ("AUDIT_COVERAGE.md",
     r"\d+ modules, \d+\+ attack vectors",
     "{total_modules} modules, 200+ attack vectors"),

    ("AUDIT_COVERAGE.md",
     r"\d+ tests across \d+ attack categories",
     "{exploit_poc_count} tests across 20+ attack categories"),

    # ── AUDIT_GUIDE.md ──────────────────────────────────────────────────────
    ("AUDIT_GUIDE.md",
     r"\d+/\d+ total modules",
     "{total_modules}/{total_modules} total modules"),

    # ── docs/BITCOIN_CORE_BACKEND_EVIDENCE.md ──────────────────────────────
    ("docs/BITCOIN_CORE_BACKEND_EVIDENCE.md",
     r"\d+/\d+ test_bitcoin",
     "{bitcoin_core_tests_pass}/{bitcoin_core_tests_total} test_bitcoin"),

    # ── docs/BITCOIN_CORE_PR_DESCRIPTION.md ────────────────────────────────
    ("docs/BITCOIN_CORE_PR_DESCRIPTION.md",
     r"\*\*\d+/\d+ test_bitcoin, 0 failures\*\*",
     "**{bitcoin_core_tests_pass}/{bitcoin_core_tests_total} test_bitcoin, 0 failures**"),

    # ── BITCOIN_CORE_PR_BLOCKERS.md ─────────────────────────────────────────
    ("BITCOIN_CORE_PR_BLOCKERS.md",
     r"✅ \d+/\d+.*test_bitcoin.*JSON",
     "✅ {bitcoin_core_tests_pass}/{bitcoin_core_tests_total} | `docs/BITCOIN_CORE_TEST_RESULTS.json`"),

    # ── docs/EXPLOIT_TEST_CATALOG.md ───────────────────────────────────────
    # (counts managed by AUDIT_CHANGELOG — catalog has per-file detail,
    #  not a total line; skip to avoid over-matching)

    # ── docs/API_REFERENCE.md version line ─────────────────────────────────
    ("docs/API_REFERENCE.md",
     r"UltrafastSecp256k1 v\d+\.\d+\.\d+",
     "UltrafastSecp256k1 v{version}"),

    # ── docs/BACKEND_ASSURANCE_MATRIX.md ───────────────────────────────────
    # (no numeric counts to sync here — skip)
]


# ---------------------------------------------------------------------------
def _apply_rule(content: str, pattern: str, template: str, data: dict) -> str:
    """Replace first occurrence of `pattern` with `template` expanded by `data`."""
    replacement = template
    for k, v in data.items():
        replacement = replacement.replace("{" + k + "}", str(v))
    return re.sub(pattern, replacement, content, count=1)


def sync_file(path: Path, rules: list[tuple[str, str]], data: dict,
              dry_run: bool) -> list[str]:
    """Apply all matching rules to a single file. Returns list of changed patterns."""
    if not path.exists():
        return []

    original = path.read_text(errors="replace")
    updated  = original
    changed  = []

    for pattern, template in rules:
        new = _apply_rule(updated, pattern, template, data)
        if new != updated:
            changed.append(pattern[:60])
            updated = new

    if updated != original:
        if not dry_run:
            path.write_text(updated)

    return changed


def main() -> int:
    dry_run = "--dry-run" in sys.argv

    if not CANON_FILE.exists():
        print(f"ERROR: {CANON_FILE} not found.")
        print("Run: python3 scripts/build_canonical_data.py")
        return 1

    data = json.loads(CANON_FILE.read_text())

    # Group rules by file
    from collections import defaultdict
    file_rules: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for rel_path, pattern, template in _RULES:
        file_rules[rel_path].append((pattern, template))

    total_changed = 0
    drift_files   = []

    for rel_path, rules in file_rules.items():
        path = ROOT / rel_path
        changed = sync_file(path, rules, data, dry_run=dry_run)
        if changed:
            total_changed += len(changed)
            drift_files.append(rel_path)
            verb = "would update" if dry_run else "updated"
            print(f"  {verb}: {rel_path} ({len(changed)} substitution(s))")

    if dry_run:
        if drift_files:
            print(f"\n[DRIFT] {len(drift_files)} file(s) out of sync with canonical_data.json:")
            for f in drift_files:
                print(f"    {f}")
            print("Run: python3 scripts/sync_docs_from_canonical.py")
            return 1
        else:
            print("[OK] All docs are in sync with canonical_data.json")
            return 0
    else:
        print(f"\nUpdated {total_changed} substitution(s) across {len(drift_files)} file(s).")
        return 0


if __name__ == "__main__":
    sys.exit(main())
