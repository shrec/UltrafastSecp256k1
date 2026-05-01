#!/usr/bin/env python3
"""
sync_docs.py — Single entry-point for all documentation consistency fixes.

Sources of truth (never edit these counts manually in docs):
  VERSION.txt                         → library version string
  audit/unified_audit_runner.cpp      → module counts (ALL_MODULES[])
  audit/test_exploit_*.cpp            → exploit PoC file count
  .github/workflows/*.yml             → CI workflow count
  include/ufsecp/ufsecp.h             → CPU ABI function count
  include/ufsecp/ufsecp_gpu.h         → GPU ABI function count

Usage:
    python3 ci/sync_docs.py [--dry-run] [--verbose] [--check]

  --check    Exit with code 1 if any doc is stale (CI gate mode).
  --dry-run  Show what would change without writing files.
  --verbose  Print every file scanned, even if unchanged.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ─── Source-of-truth readers ──────────────────────────────────────────────────

def read_version() -> str:
    return (ROOT / "VERSION.txt").read_text().strip().lstrip("v")


def read_module_counts() -> dict[str, int]:
    runner = ROOT / "audit" / "unified_audit_runner.cpp"
    text = runner.read_text()
    m = re.search(r"static const AuditModule ALL_MODULES\[\]\s*=\s*\{(.+?)^\};",
                  text, re.DOTALL | re.MULTILINE)
    if not m:
        raise RuntimeError("ALL_MODULES[] not found in unified_audit_runner.cpp")
    block = m.group(1)
    total = len(re.findall(r"^\s*\{\s*\"[a-z]", block, re.MULTILINE))
    exploit = len(re.findall(r"^\s*\{\s*\"exploit_", block, re.MULTILINE))
    poc_files = len(list((ROOT / "audit").glob("test_exploit_*.cpp")))
    return {
        "total": total,
        "exploit_mods": exploit,
        "non_exploit_mods": total - exploit,
        "exploit_files": poc_files,
    }


def read_workflow_count() -> int:
    wf_dir = ROOT / ".github" / "workflows"
    return len(list(wf_dir.glob("*.yml")))


def read_abi_counts() -> dict[str, int]:
    def _count(header_name: str) -> int:
        p = ROOT / "include" / "ufsecp" / header_name
        if not p.exists():
            return 0
        return len(re.findall(r"^UFSECP_API\b", p.read_text(), re.MULTILINE))
    return {
        "cpu": _count("ufsecp.h"),
        "gpu": _count("ufsecp_gpu.h"),
    }


# ─── Replacement rule sets ────────────────────────────────────────────────────

def _sub(pattern: str, replacement: str, text: str) -> tuple[str, int]:
    new = re.sub(pattern, replacement, text)
    return new, text.count(re.findall(pattern, text)[0]) if text != new and re.findall(pattern, text) else (new, 0)


def apply_version_rules(text: str, version: str) -> tuple[str, int]:
    """Version string replacements (subset; full coverage in sync_version_refs.py)."""
    rules = [
        (r"(\*\*Version\*\*:\s*)[\d]+\.[\d]+\.[\d]+", r"\g<1>" + version),
        (r"(\*\*UltrafastSecp256k1\s+v)[\d]+\.[\d]+\.[\d]+(\*\*)", r"\g<1>" + version + r"\g<2>"),
        (r"(\*UltrafastSecp256k1\s+v)[\d]+\.[\d]+\.[\d]+(\s+--)", r"\g<1>" + version + r"\g<2>"),
        (r"(\|\s*\*\*Library Version\*\*\s*\|\s*)[\d]+\.[\d]+\.[\d]+(\s*\|)", r"\g<1>" + version + r"\g<2>"),
        (r"(\|\s*Version\s*\|\s*)[\d]+\.[\d]+\.[\d]+(\s*\|)", r"\g<1>" + version + r"\g<2>"),
        (r"(UltrafastSecp256k1\s+v)[\d]+\.[\d]+\.[\d]+", r"\g<1>" + version),
        (r"(ultrafastsecp256k1/)[\d]+\.[\d]+\.[\d]+(@)", r"\g<1>" + version + r"\g<2>"),
        (r"(GIT_TAG\s+v)[\d]+\.[\d]+\.[\d]+", r"\g<1>" + version),
        (r"(ufsecp@v)[\d]+\.[\d]+\.[\d]+", r"\g<1>" + version),
        (r"(<version>)[\d]+\.[\d]+\.[\d]+(</version>)", r"\g<1>" + version + r"\g<2>"),
        (r'(ufsecp\s*=\s*\{\s*version\s*=\s*")[\d]+\.[\d]+\.[\d]+(")', r"\g<1>" + version + r"\g<2>"),
        (r'("library_version":\s*")[\d]+\.[\d]+\.[\d]+(")', r"\g<1>" + version + r"\g<2>"),
    ]
    changes = 0
    for pattern, repl in rules:
        new = re.sub(pattern, repl, text)
        if new != text:
            changes += 1
            text = new
    return text, changes


def apply_module_rules(text: str, counts: dict[str, int]) -> tuple[str, int]:
    em = counts["exploit_mods"]
    ne = counts["non_exploit_mods"]
    ef = counts["exploit_files"]
    total = counts["total"]

    rules = [
        # "157 PoC files, 146 registered modules"
        (r"\b\d{1,3} PoC files, \d{1,3} registered modules\b",
         f"{ef} PoC files, {em} registered modules"),
        # "146 PoC files" standalone
        (r"\b\d{1,3} PoC files?\b",
         f"{ef} PoC files"),
        # "146 exploit-PoC modules" / "146 exploit PoC modules"
        (r"\b\d{1,3} exploit[- ]?PoC modules?\b",
         f"{em} exploit-PoC modules"),
        # "146 exploit-style PoC modules/tests"
        (r"\b\d{1,3} exploit-style PoC (?:tests?|modules?)\b",
         f"{em} exploit-style PoC modules"),
        # "146 dedicated adversarial PoC modules"
        (r"\b\d{1,3} dedicated adversarial PoC modules?\b",
         f"{em} dedicated adversarial PoC modules"),
        # "74 non-exploit modules + 146 exploit-PoC modules (220 total)"
        (r"\b\d{1,3} non-exploit modules? \+ \d{1,3} exploit-PoC modules? \(\d{1,3} total\)",
         f"{ne} non-exploit modules + {em} exploit-PoC modules ({total} total)"),
        # "73 non-exploit + 146 exploit-PoC modules"
        (r"\b\d{1,3} non-exploit \+ \d{1,3} exploit-PoC modules?",
         f"{ne} non-exploit + {em} exploit-PoC modules"),
        # "All 146 exploit PoC modules pass"
        (r"\bAll \d{1,3} exploit PoC modules? pass\b",
         f"All {em} exploit PoC modules pass"),
        # "All 74 non-exploit audit modules"
        (r"\bAll \d{1,3} non-exploit audit modules?\b",
         f"All {ne} non-exploit audit modules"),
        # "across 220 modules: 74 core + 146 exploit-PoC"  (in workflow YAML)
        (r"\(\d{1,3} modules: \d{1,3} core \+ \d{1,3} exploit-PoC\)",
         f"({total} modules: {ne} core + {em} exploit-PoC)"),
        # "(73 non-exploit modules, ~1M checks)" style
        (r"\(\d{1,3} non-exploit modules? \+ \d{1,3} exploit-PoC modules?, ~\d+M [a-z]+\)",
         f"({ne} non-exploit + {em} exploit-PoC modules, ~1M assertions)"),
    ]
    changes = 0
    for pattern, repl in rules:
        new = re.sub(pattern, repl, text, flags=re.IGNORECASE)
        if new != text:
            changes += 1
            text = new
    return text, changes


def apply_workflow_rules(text: str, wf_count: int) -> tuple[str, int]:
    rules = [
        # "33 workflows, 16 platform combinations"
        (r"\b\d{1,3} workflows?, \d{1,3} platform combinations?\b",
         f"{wf_count} workflows, \\2 platform combinations"),  # keep platform count
        # "33 CI workflows"
        (r"\b\d{1,3} CI(?:/CD)? workflows?\b",
         f"{wf_count} CI workflows"),
        # "33 GitHub Actions workflows"
        (r"\b\d{1,3} GitHub Actions workflows?\b",
         f"{wf_count} GitHub Actions workflows"),
        # "31 automated workflows" in section heading / paragraph
        (r"\b\d{1,3} Automated Workflows\b",
         f"{wf_count} Automated Workflows"),
        # "33 workflows" standalone in table
        (r"\b(\d{1,3}) workflows\b",
         f"{wf_count} workflows"),
    ]
    changes = 0
    for pattern, repl in rules:
        # For the platform-combinations rule, use a real back-ref
        if r"\2" in repl:
            full = re.sub(
                r"(\b\d{1,3} workflows?, )(\d{1,3} platform combinations?\b)",
                lambda m: f"{wf_count} workflows, {m.group(2)}",
                text
            )
        else:
            full = re.sub(pattern, repl, text, flags=re.IGNORECASE)
        if full != text:
            changes += 1
            text = full
    return text, changes


def apply_abi_rules(text: str, abi: dict[str, int]) -> tuple[str, int]:
    gpu = abi["gpu"]
    rules = [
        # "(17 FFI functions, incl. FROST)" → actual GPU count
        (r"\(\d{1,3} FFI functions, incl\. FROST\)",
         f"({gpu} functions, incl. FROST)"),
        # "GPU C ABI header (18 functions, 8 ops, 3 backends)"
        (r"GPU C ABI header \(\d{1,3} functions,",
         f"GPU C ABI header ({gpu} functions,"),
    ]
    changes = 0
    for pattern, repl in rules:
        new = re.sub(pattern, repl, text)
        if new != text:
            changes += 1
            text = new
    return text, changes


# ─── File list ────────────────────────────────────────────────────────────────

# (path relative to ROOT, set of rule groups to apply)
DOC_FILES: list[tuple[str, set[str]]] = [
    ("README.md",                         {"version", "modules", "workflows", "abi"}),
    ("WHY_ULTRAFASTSECP256K1.md",         {"version", "modules", "workflows"}),
    ("AUDIT_GUIDE.md",                    {"version", "modules", "workflows"}),
    ("AUDIT_REPORT.md",                   {"version"}),
    ("docs/AUDIT_CHANGELOG.md",           {"version", "modules"}),
    ("docs/AUDIT_TRACEABILITY.md",        {"version", "modules"}),
    ("docs/ASSURANCE_LEDGER.md",          {"version", "modules"}),
    ("docs/CT_VERIFICATION.md",           {"version"}),
    (".github/workflows/security-audit.yml", {"modules"}),
    (".github/workflows/audit-report.yml",   {"modules"}),
]

# Files that are intentional historical records — version patches skipped
VERSION_SKIP_FILES = {
    "CHANGELOG.md", "ROADMAP.md", "AUDIT_REPORT.md",
}


# ─── Main ─────────────────────────────────────────────────────────────────────

def process_file(
    rel: str,
    rule_groups: set[str],
    version: str,
    counts: dict[str, int],
    wf_count: int,
    abi: dict[str, int],
    dry_run: bool,
    verbose: bool,
) -> int:
    path = ROOT / rel
    if not path.exists():
        if verbose:
            print(f"  SKIP (not found): {rel}")
        return 0

    original = path.read_text(encoding="utf-8")
    text = original
    total_changes = 0

    if "version" in rule_groups and path.name not in VERSION_SKIP_FILES:
        text, n = apply_version_rules(text, version)
        total_changes += n

    if "modules" in rule_groups:
        text, n = apply_module_rules(text, counts)
        total_changes += n

    if "workflows" in rule_groups:
        text, n = apply_workflow_rules(text, wf_count)
        total_changes += n

    if "abi" in rule_groups:
        text, n = apply_abi_rules(text, abi)
        total_changes += n

    if total_changes == 0:
        if verbose:
            print(f"  OK   (no changes): {rel}")
        return 0

    print(f"  {'[dry-run] ' if dry_run else ''}UPDATED ({total_changes} changes): {rel}")
    if not dry_run:
        path.write_text(text, encoding="utf-8")
    return total_changes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would change without writing files")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print every file scanned")
    parser.add_argument("--check", action="store_true",
                        help="Exit 1 if any doc is stale (CI gate mode)")
    args = parser.parse_args()

    dry_run = args.dry_run or args.check

    # Read canonical values
    version   = read_version()
    counts    = read_module_counts()
    wf_count  = read_workflow_count()
    abi       = read_abi_counts()

    print("Sources of truth:")
    print(f"  VERSION.txt        : {version}")
    print(f"  Module counts      : {counts['total']} total "
          f"({counts['non_exploit_mods']} core + {counts['exploit_mods']} exploit-PoC), "
          f"{counts['exploit_files']} .cpp files")
    print(f"  CI workflows       : {wf_count}")
    print(f"  CPU ABI functions  : {abi['cpu']}")
    print(f"  GPU ABI functions  : {abi['gpu']}")
    print()

    total_stale = 0
    for rel, groups in DOC_FILES:
        n = process_file(rel, groups, version, counts, wf_count, abi,
                         dry_run, args.verbose)
        total_stale += n

    print()
    if args.check:
        if total_stale:
            print(f"STALE: {total_stale} occurrence(s) out of sync with sources of truth.")
            print("Run:  python3 ci/sync_docs.py  to fix.")
            return 1
        else:
            print("OK: all docs are in sync with sources of truth.")
            return 0
    elif dry_run:
        print(f"Dry run: {total_stale} occurrence(s) would be updated.")
    elif total_stale == 0:
        print("All docs already up to date.")
    else:
        print(f"Updated {total_stale} occurrence(s) across all docs.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
