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
     r"\d+ non-exploit modules \+ \d+ exploit PoCs",
     "{non_exploit_modules} non-exploit modules + {exploit_poc_count} exploit PoCs"),

    ("README.md",
     r"\d+ non-exploit audit modules and \d+ exploit PoCs",
     "{non_exploit_modules} non-exploit audit modules and {exploit_poc_count} exploit PoCs"),

    ("README.md",
     r"\d+ non-exploit modules \+ \d+ exploit-PoC modules",
     "{non_exploit_modules} non-exploit modules + {exploit_poc_count} exploit-PoC modules"),

    ("README.md",
     r"\*\*\d+ non-exploit modules \+ \d+ exploit PoCs across \d+ sections, 0 failures\*\*",
     "**{non_exploit_modules} non-exploit modules + {exploit_poc_count} exploit PoCs across 9 sections, 0 failures**"),

    ("README.md",
     r"\d+ non-exploit modules \+ \d+ exploit PoCs, ~1M\+ assertions",
     "{non_exploit_modules} non-exploit modules + {exploit_poc_count} exploit PoCs, ~1M+ assertions"),

    ("README.md",
     r"\*\*\d+ non-exploit audit modules \+ \d+ exploit PoCs\*\*",
     "**{non_exploit_modules} non-exploit audit modules + {exploit_poc_count} exploit PoCs**"),

    ("README.md",
     r"\d+ non-exploit modules \+ \d+ exploit PoCs and platform verdicts",
     "{non_exploit_modules} non-exploit modules + {exploit_poc_count} exploit PoCs and platform verdicts"),

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

    # ── WHY_ULTRAFASTSECP256K1.md section headers ──────────────────────────
    ("WHY_ULTRAFASTSECP256K1.md",
     r"## 2\. CI/CD Pipeline — \d+ Automated Workflows",
     "## 2. CI/CD Pipeline — {ci_workflow_count} Automated Workflows"),

    ("WHY_ULTRAFASTSECP256K1.md",
     r"quality enforcement system with \d+ GitHub Actions workflows",
     "quality enforcement system with {ci_workflow_count} GitHub Actions workflows"),

    # ── WHY_ULTRAFASTSECP256K1.md security-audit.yml row ──────────────────
    ("WHY_ULTRAFASTSECP256K1.md",
     r"\d+ non-exploit \+ \d+ exploit-PoC modules, ~1M assertions",
     "{non_exploit_modules} non-exploit + {exploit_poc_count} exploit-PoC modules, ~1M assertions"),

    # ── AUDIT_REPORT.md ────────────────────────────────────────────────────
    ("AUDIT_REPORT.md",
     r"\d+ modules, ~1,000,000\+ checks, \d+ exploit PoC tests, 0 failures",
     "{total_modules} modules, ~1,000,000+ checks, {exploit_poc_count} exploit PoC tests, 0 failures"),

    ("AUDIT_REPORT.md",
     r"Exploit PoC tests \| — \| \d+ tests, \d+ attack vectors",
     "Exploit PoC tests | — | {exploit_poc_count} tests, {exploit_poc_count} attack vectors"),

    # ── docs/AUDIT_READINESS_REPORT_v1.md ──────────────────────────────────
    ("docs/AUDIT_READINESS_REPORT_v1.md",
     r"# Exploit PoC security probes \(\d+ probes\)",
     "# Exploit PoC security probes ({exploit_poc_count} probes)"),

    # ── docs/AUDIT_SCOPE.md ────────────────────────────────────────────────
    ("docs/AUDIT_SCOPE.md",
     r"# Exploit PoC security probes \(\d+ probes\)",
     "# Exploit PoC security probes ({exploit_poc_count} probes)"),

    # ── THREAT_MODEL.md version refs ──────────────────────────────────────
    ("THREAT_MODEL.md",
     r"\*\*Added in v\d+\.\d+\.\d+\.\*\* Provides non-interactive ZK",
     "**Added in v{version}.** Provides non-interactive ZK"),

    ("THREAT_MODEL.md",
     r"\*\*⚠️ Status \(v\d+\.\d+\.\d+\):\*\*",
     "**⚠️ Status (v{version}):**"),

    ("THREAT_MODEL.md",
     r"\*\*Added in v\d+\.\d+\.\d+\.\*\* Provides Ethereum-specific",
     "**Added in v{version}.** Provides Ethereum-specific"),

    ("THREAT_MODEL.md",
     r"## Automated Security Measures \(v\d+\.\d+\.\d+\)",
     "## Automated Security Measures (v{version})"),

    ("THREAT_MODEL.md",
     r"Internal audit report \(v\d+\.\d+\.\d+ baseline",
     "Internal audit report (v{version} baseline"),

    # ── SECURITY.md version ref ────────────────────────────────────────────
    ("SECURITY.md",
     r"Internal audit report \(v\d+\.\d+\.\d+ baseline",
     "Internal audit report (v{version} baseline"),

    # ── AUDIT_GUIDE.md version ref ─────────────────────────────────────────
    ("AUDIT_GUIDE.md",
     r"\(v\d+\.\d+\.\d+\):",
     "(v{version}):"),

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
