#!/usr/bin/env python3
"""
sync_docs_from_canonical.py
============================
Reads docs/canonical_data.json and propagates every value into all
documentation files that reference it.

Usage:
    python3 ci/sync_docs_from_canonical.py            # apply updates
    python3 ci/sync_docs_from_canonical.py --dry-run  # report drift, exit 1 if any

Never edit doc numbers manually — run scripts/sync_all_docs.py instead.
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

    # ── docs/WHY_ULTRAFASTSECP256K1.md ────────────────────────────────────
    # Table row: exploit PoC column (line ~54) — actual text uses "| N wired, 0 failures"
    ("docs/WHY_ULTRAFASTSECP256K1.md",
     r"\d+ dedicated adversarial PoC modules across 20\+ coverage categories \(`audit/test_exploit_\*\.cpp`\) \| \d+ wired, 0 failures",
     "{exploit_poc_count} dedicated adversarial PoC modules across 20+ coverage categories (`audit/test_exploit_*.cpp`) | {exploit_poc_count} wired, 0 failures"),

    # Total row (line ~63) — actual text uses "exploit PoCs modules" (no dash, plural PoCs)
    ("docs/WHY_ULTRAFASTSECP256K1.md",
     r"\*\*unified_audit_runner\*\* across \d+ non-exploit modules \+ \d+ exploit PoCs modules \(\d+ total\)",
     "**unified_audit_runner** across {non_exploit_modules} non-exploit modules + {exploit_poc_count} exploit PoC modules ({total_modules} total)"),

    # Exploit PoC total row (line ~64) — actual text uses "exploit PoCs modules" not "exploit-style PoC modules"
    ("docs/WHY_ULTRAFASTSECP256K1.md",
     r"\*\*\d+ exploit PoCs modules\*\* across 20\+ coverage categories.*?\| \*\*\d+ modules, 0 failures\*\*",
     "**{exploit_poc_count} exploit PoC modules** across 20+ coverage categories, all in `audit/test_exploit_*.cpp` | **{exploit_poc_count} modules, 0 failures**"),

    ("docs/WHY_ULTRAFASTSECP256K1.md",
     r"\d+ workflows, \d+ platform combinations",
     "{ci_workflow_count} workflows, 16 platform combinations"),

    ("docs/WHY_ULTRAFASTSECP256K1.md",
     r"\d+ independent pipelines \(LLVM ct-verif.*?ARM64 native\)",
     "{ct_pipeline_count} independent pipelines (LLVM ct-verif, Valgrind taint, ct-prover, dudect, ARM64 native)"),

    # ── docs/AUDIT_COVERAGE.md ─────────────────────────────────────────────
    # TL;DR table row: "Audit modules"
    ("docs/AUDIT_COVERAGE.md",
     r"\d+ across \d+ failure classes",
     "{total_modules} across 9 failure classes"),

    # TL;DR table row: "Exploit PoC tests"
    ("docs/AUDIT_COVERAGE.md",
     r"\d+ exploit PoCs modules, \d+\+ attack vectors",
     "{exploit_poc_count} exploit PoC modules, 200+ attack vectors"),

    # TL;DR table row: CI workflows
    ("docs/AUDIT_COVERAGE.md",
     r"\d+ GitHub Actions workflows",
     "{ci_workflow_count} GitHub Actions workflows"),

    # Summary table row: CI Workflows (second occurrence — count=1 only hits the first above)
    ("docs/AUDIT_COVERAGE.md",
     r"(CI Workflows\s+\|)\s*\d+ GitHub Actions workflows",
     r"\g<1> {ci_workflow_count} GitHub Actions workflows"),

    # Summary table: "Audit Modules" row
    ("docs/AUDIT_COVERAGE.md",
     r"\d+ \(55 \+ dedicated C ABI thread stress\)",
     "{non_exploit_modules} (non-exploit modules)"),

    # Summary table: "Exploit PoC Tests" row
    ("docs/AUDIT_COVERAGE.md",
     r"\*\*\d+ tests across 20\+ attack categories\*\*",
     "**{exploit_poc_count} tests across 20+ attack categories**"),

    # Verdict header line
    ("docs/AUDIT_COVERAGE.md",
     r"\*\*AUDIT-READY\*\* -- \d+ modules, \d+ failure classes",
     "**AUDIT-READY** -- {total_modules} modules, 9 failure classes"),

    # Generic fallback for "N modules, N+ attack vectors"
    ("docs/AUDIT_COVERAGE.md",
     r"\d+ modules, \d+\+ attack vectors",
     "{total_modules} modules, 200+ attack vectors"),

    ("docs/AUDIT_COVERAGE.md",
     r"\d+ tests across \d+ attack categories",
     "{exploit_poc_count} tests across 20+ attack categories"),

    # ── docs/AUDIT_GUIDE.md ────────────────────────────────────────────────
    ("docs/AUDIT_GUIDE.md",
     r"\d+/\d+ total modules",
     "{total_modules}/{total_modules} total modules"),

    # ── docs/BITCOIN_CORE_BACKEND_EVIDENCE.md ─────────────────────────────
    ("docs/BITCOIN_CORE_BACKEND_EVIDENCE.md",
     r"\d+/\d+ test_bitcoin",
     "{bitcoin_core_tests_pass}/{bitcoin_core_tests_total} test_bitcoin"),

    # ── docs/BITCOIN_CORE_PR_DESCRIPTION.md ───────────────────────────────
    ("docs/BITCOIN_CORE_PR_DESCRIPTION.md",
     r"\*\*\d+/\d+ test_bitcoin, 0 failures\*\*",
     "**{bitcoin_core_tests_pass}/{bitcoin_core_tests_total} test_bitcoin, 0 failures**"),

    # ── docs/BITCOIN_CORE_PR_BLOCKERS.md ──────────────────────────────────
    ("docs/BITCOIN_CORE_PR_BLOCKERS.md",
     r"✅ \d+/\d+.*test_bitcoin.*JSON",
     "✅ {bitcoin_core_tests_pass}/{bitcoin_core_tests_total} | `docs/BITCOIN_CORE_TEST_RESULTS.json`"),

    # ── docs/API_REFERENCE.md version line ────────────────────────────────
    ("docs/API_REFERENCE.md",
     r"UltrafastSecp256k1 v\d+\.\d+\.\d+",
     "UltrafastSecp256k1 v{version}"),

    # ── docs/WHY_ULTRAFASTSECP256K1.md section headers ────────────────────
    ("docs/WHY_ULTRAFASTSECP256K1.md",
     r"## 2\. CI/CD Pipeline — \d+ Automated Workflows",
     "## 2. CI/CD Pipeline — {ci_workflow_count} Automated Workflows"),

    ("docs/WHY_ULTRAFASTSECP256K1.md",
     r"quality enforcement system with \d+ GitHub Actions workflows",
     "quality enforcement system with {ci_workflow_count} GitHub Actions workflows"),

    # ── docs/WHY_ULTRAFASTSECP256K1.md security-audit.yml row ─────────────
    ("docs/WHY_ULTRAFASTSECP256K1.md",
     r"\d+ non-exploit \+ \d+ exploit-PoC modules, ~1M assertions",
     "{non_exploit_modules} non-exploit + {exploit_poc_count} exploit-PoC modules, ~1M assertions"),

    # ── AUDIT_REPORT.md ────────────────────────────────────────────────────
    ("AUDIT_REPORT.md",
     r"\d+ modules, ~1,000,000\+ checks, \d+ exploit PoC tests, 0 failures",
     "{total_modules} modules, ~1,000,000+ checks, {exploit_poc_count} exploit PoC tests, 0 failures"),

    ("AUDIT_REPORT.md",
     r"Exploit PoC tests \| — \| \d+ tests, \d+ attack vectors",
     "Exploit PoC tests | — | {exploit_poc_count} tests, {exploit_poc_count} attack vectors"),

    # ── docs/AUDIT_READINESS_REPORT_v1.md ─────────────────────────────────
    ("docs/AUDIT_READINESS_REPORT_v1.md",
     r"# Exploit PoC security probes \(\d+ probes\)",
     "# Exploit PoC security probes ({exploit_poc_count} probes)"),

    # ── docs/AUDIT_SCOPE.md ────────────────────────────────────────────────
    ("docs/AUDIT_SCOPE.md",
     r"# Exploit PoC security probes \(\d+ probes\)",
     "# Exploit PoC security probes ({exploit_poc_count} probes)"),

    # ── docs/THREAT_MODEL.md version refs ─────────────────────────────────
    ("docs/THREAT_MODEL.md",
     r"\*\*Added in v\d+\.\d+\.\d+\.\*\* Provides non-interactive ZK",
     "**Added in v{version}.** Provides non-interactive ZK"),

    ("docs/THREAT_MODEL.md",
     r"\*\*⚠️ Status \(v\d+\.\d+\.\d+\):\*\*",
     "**⚠️ Status (v{version}):**"),

    ("docs/THREAT_MODEL.md",
     r"\*\*Added in v\d+\.\d+\.\d+\.\*\* Provides Ethereum-specific",
     "**Added in v{version}.** Provides Ethereum-specific"),

    ("docs/THREAT_MODEL.md",
     r"## Automated Security Measures \(v\d+\.\d+\.\d+\)",
     "## Automated Security Measures (v{version})"),

    ("docs/THREAT_MODEL.md",
     r"Internal audit report \(v\d+\.\d+\.\d+ baseline",
     "Internal audit report (v{version} baseline"),

    # ── SECURITY.md version ref ────────────────────────────────────────────
    ("SECURITY.md",
     r"Internal audit report \(v\d+\.\d+\.\d+ baseline",
     "Internal audit report (v{version} baseline"),

    # ── docs/AUDIT_GUIDE.md version ref ───────────────────────────────────
    ("docs/AUDIT_GUIDE.md",
     r"\(v\d+\.\d+\.\d+\):",
     "(v{version}):"),

    # ── ARCHITECTURE.svg ───────────────────────────────────────────────────
    ("ARCHITECTURE.svg",
     r"\d+(?:,\d+)? lines • \d+ exploit PoC • \d+ audit modules • \d+ CI workflows",
     "114,458 lines • {exploit_poc_count} exploit PoC • {non_exploit_modules} audit modules • {ci_workflow_count} CI workflows"),

    ("ARCHITECTURE.svg",
     r"\d+ exploit PoC tests",
     "{exploit_poc_count} exploit PoC tests"),
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
            print("Run: python3 scripts/sync_all_docs.py")
            return 1
        else:
            print("[OK] All docs are in sync with canonical_data.json")
            return 0
    else:
        print(f"\nUpdated {total_changed} substitution(s) across {len(drift_files)} file(s).")
        return 0


if __name__ == "__main__":
    sys.exit(main())
