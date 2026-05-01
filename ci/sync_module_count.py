#!/usr/bin/env python3
"""
sync_module_count.py — Auto-sync module/exploit-PoC counts across all docs.

Reads the actual counts from:
  - audit/unified_audit_runner.cpp  (ALL_MODULES[] array — canonical)
  - audit/test_exploit_*.cpp        (exploit PoC file count)

Then updates every doc file that contains stale numeric references.

Usage:
    python3 ci/sync_module_count.py [--dry-run] [--verbose]
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent

# ── Patterns that identify stale counts in doc files ─────────────────────────

# Matches e.g. "55 modules", "58 modules", "215 modules", "220 modules"
MODULE_COUNT_RE = re.compile(r'\b\d{1,3} modules?\b')

# Matches e.g. "86 exploit PoC", "135 exploit PoC", "147 exploit"
EXPLOIT_MODULE_RE = re.compile(r'\b\d{1,3} exploit[- ]?PoC modules?\b', re.IGNORECASE)

# Matches e.g. "86 exploit-style PoC tests", "147 exploit-style PoC modules"
EXPLOIT_STYLE_RE = re.compile(r'\b\d{1,3} exploit-style PoC (?:tests?|modules?)\b', re.IGNORECASE)

# Matches e.g. "135 PoC files", "157 PoC files"
POC_FILES_RE = re.compile(r'\b\d{1,3} PoC files?\b', re.IGNORECASE)

# Matches "N PoC files, M registered modules" in TL;DR table
POC_FILES_AND_MODULES_RE = re.compile(
    r'\b\d{1,3} PoC files, \d{1,3} registered modules\b'
)

# Matches "(N modules," or "(N modules)" or "across N modules"
PAREN_MODULES_RE = re.compile(r'(?:across|\() *\d{1,3} (?:non-exploit )?modules?\b')

# Matches lines like "All N exploit PoC modules pass."
ALL_EXPLOIT_RE = re.compile(r'\bAll \d{1,3} exploit PoC modules? pass\b', re.IGNORECASE)

# Matches "All N non-exploit audit modules"
ALL_NONEXPLOIT_RE = re.compile(r'\bAll \d{1,3} non-exploit audit modules?\b', re.IGNORECASE)


# ── Count sources ─────────────────────────────────────────────────────────────

def count_all_modules(runner: Path) -> tuple[int, int, int]:
    """Parse ALL_MODULES[] in unified_audit_runner.cpp.

    Returns (total, exploit_poc_count, non_exploit_count).
    """
    text = runner.read_text()
    # Extract the ALL_MODULES[] block
    m = re.search(r'static const AuditModule ALL_MODULES\[\]\s*=\s*\{(.+?)^\};',
                  text, re.DOTALL | re.MULTILINE)
    if not m:
        raise RuntimeError("Could not find ALL_MODULES[] in unified_audit_runner.cpp")
    block = m.group(1)
    # Each entry starts with: { "name",
    entries = re.findall(r'^\s*\{\s*"[a-z]', block, re.MULTILINE)
    total = len(entries)
    exploit_entries = re.findall(r'^\s*\{\s*"exploit_', block, re.MULTILINE)
    exploit = len(exploit_entries)
    non_exploit = total - exploit
    return total, exploit, non_exploit


def count_exploit_files(audit_dir: Path) -> int:
    return len(list(audit_dir.glob('test_exploit_*.cpp')))


# ── Replacement helpers ───────────────────────────────────────────────────────

def make_replacements(content: str,
                      total: int,
                      exploit_mods: int,
                      non_exploit: int,
                      exploit_files: int) -> tuple[str, int]:
    """Apply all substitutions; return (new_content, n_changes)."""
    changes = 0
    new = content

    # "N PoC files, M registered modules" → exact combined form
    def _replace_poc_and_mods(m: re.Match) -> str:
        nonlocal changes
        replacement = f'{exploit_files} PoC files, {exploit_mods} registered modules'
        if m.group(0) != replacement:
            changes += 1
        return replacement

    new = POC_FILES_AND_MODULES_RE.sub(_replace_poc_and_mods, new)

    # "N PoC files" standalone
    def _replace_poc_files(m: re.Match) -> str:
        nonlocal changes
        replacement = f'{exploit_files} PoC files'
        if m.group(0) != replacement:
            changes += 1
        return replacement

    new = POC_FILES_RE.sub(_replace_poc_files, new)

    # "N exploit PoC modules" / "N exploit-PoC modules"
    def _replace_exploit_mods(m: re.Match) -> str:
        nonlocal changes
        replacement = f'{exploit_mods} exploit-PoC modules'
        if m.group(0).lower().replace(' ', '-') != replacement.lower().replace(' ', '-'):
            changes += 1
        return replacement

    new = EXPLOIT_MODULE_RE.sub(_replace_exploit_mods, new)

    # "N exploit-style PoC tests/modules"
    def _replace_exploit_style(m: re.Match) -> str:
        nonlocal changes
        replacement = f'{exploit_mods} exploit-style PoC modules'
        if m.group(0) != replacement:
            changes += 1
        return replacement

    new = EXPLOIT_STYLE_RE.sub(_replace_exploit_style, new)

    # "All N exploit PoC modules pass"
    def _replace_all_exploit(m: re.Match) -> str:
        nonlocal changes
        replacement = f'All {exploit_mods} exploit PoC modules pass'
        if m.group(0) != replacement:
            changes += 1
        return replacement

    new = ALL_EXPLOIT_RE.sub(_replace_all_exploit, new)

    # "All N non-exploit audit modules"
    def _replace_all_nonexploit(m: re.Match) -> str:
        nonlocal changes
        replacement = f'All {non_exploit} non-exploit audit modules'
        if m.group(0) != replacement:
            changes += 1
        return replacement

    new = ALL_NONEXPLOIT_RE.sub(_replace_all_nonexploit, new)

    return new, changes


# ── Files to update ───────────────────────────────────────────────────────────

DOC_FILES = [
    'WHY_ULTRAFASTSECP256K1.md',
    'README.md',
    'AUDIT_GUIDE.md',
    'AUDIT_REPORT.md',
    'docs/AUDIT_CHANGELOG.md',
    'docs/AUDIT_TRACEABILITY.md',
    'docs/ASSURANCE_LEDGER.md',
    '.github/workflows/security-audit.yml',
    '.github/workflows/audit-report.yml',
]


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would change without writing files')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Print each file processed')
    args = parser.parse_args()

    runner = BASE / 'audit' / 'unified_audit_runner.cpp'
    audit_dir = BASE / 'audit'

    if not runner.exists():
        print(f'ERROR: {runner} not found', file=sys.stderr)
        return 1

    total, exploit_mods, non_exploit = count_all_modules(runner)
    exploit_files = count_exploit_files(audit_dir)

    print(f'Counts from audit/unified_audit_runner.cpp:')
    print(f'  ALL_MODULES total : {total}')
    print(f'  exploit-PoC mods  : {exploit_mods}')
    print(f'  non-exploit mods  : {non_exploit}')
    print(f'  exploit .cpp files: {exploit_files}')
    print()

    total_changed_files = 0
    total_changed_lines = 0

    for rel in DOC_FILES:
        path = BASE / rel
        if not path.exists():
            if args.verbose:
                print(f'  SKIP (not found): {rel}')
            continue

        original = path.read_text()
        updated, n = make_replacements(original, total, exploit_mods, non_exploit, exploit_files)

        if n == 0:
            if args.verbose:
                print(f'  OK   (no changes): {rel}')
            continue

        total_changed_files += 1
        total_changed_lines += n
        print(f'  UPDATED ({n} changes): {rel}')

        if not args.dry_run:
            path.write_text(updated)

    print()
    if args.dry_run:
        print(f'Dry run: would update {total_changed_files} file(s), {total_changed_lines} occurrence(s).')
    elif total_changed_files == 0:
        print('All docs already up to date.')
    else:
        print(f'Updated {total_changed_files} file(s), {total_changed_lines} occurrence(s).')

    return 0


if __name__ == '__main__':
    sys.exit(main())
