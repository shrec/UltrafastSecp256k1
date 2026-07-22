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
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent

# ── Patterns that identify stale counts in doc files ─────────────────────────

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

# Matches "All N non-exploit audit modules and all N exploit PoCs" (README variant)
ALL_BOTH_RE = re.compile(r'All \d+ non-exploit audit modules and all \d+ exploit PoCs')

# Matches "N non-exploit modules + N exploit PoCs across N sections, N failures"
TABLE_EXPLOIT_ROW_RE = re.compile(
    r'\d+ non-exploit modules \+ \d+ exploit PoCs across \d+ sections, \d+ failures'
)

# Matches "N non-exploit modules + N exploit-PoC modules (N total)" — WHY doc table row
TABLE_WHY_DOC_RE = re.compile(
    r'\d+ non-exploit modules \+ \d+ exploit-PoC modules \(\d+ total\)'
)

# Matches "All N exploit PoCs modules pass" (plural variant — different from ALL_EXPLOIT_RE)
ALL_EXPLOIT_PLURAL_RE = re.compile(r'\bAll \d{1,3} exploit PoCs modules pass\b', re.IGNORECASE)

# Matches "N-module unified_audit_runner" (plain and backtick-wrapped variants)
RUNNER_MODULE_COUNT_RE = re.compile(r'\b\d{3}-module unified_audit_runner\b')
RUNNER_MODULE_COUNT_BT_RE = re.compile(r'\b\d{3}-module `unified_audit_runner`')

# Matches "264 exploit PoCs" / "264 exploit-PoCs" without "modules" suffix.
# Uses 3+ digit requirement to avoid false positives from incremental changelog entries.
EXPLOIT_POCS_SIMPLE_RE = re.compile(r'\b(\d{3,})\s+exploit[ -]PoC[s]?\b', re.IGNORECASE)

# Matches paired headline/readme claims such as "262 exploit PoCs / 367 modules".
# This must run before EXPLOIT_POCS_SIMPLE_RE, otherwise only the exploit count
# is updated and the total module count silently drifts.
EXPLOIT_POCS_SLASH_MODULES_RE = re.compile(
    r'\b\d{3,}\s+exploit[ -]PoC[s]?\s*/\s*\d{3,}\s+(?:registered\s+)?modules?\b',
    re.IGNORECASE,
)

# Matches "N registered exploit-style PoC test modules (M test files)"
REGISTERED_STYLE_RE = re.compile(
    r'\d+ registered exploit-style PoC test modules \(\d+ test files\)'
)

# Matches "N exploit PoC test files (M registered as runner modules"
EXPLOIT_FILES_REGISTERED_RE = re.compile(
    r'\d+ exploit PoC test files \(\d+ registered as runner modules'
)

# Matches "Exploit PoC Test Suite (N Tests,"
SUITE_HEADER_RE = re.compile(r'Exploit PoC Test Suite \(\d+ Tests,')

# Matches "N exploit PoCs test files (all N registered as runner modules"
EXPLOIT_POCS_ALL_RE = re.compile(
    r'\d+ exploit PoCs? test files? \(all \d+ registered as runner modules'
)

# Matches "N tests, 20+ coverage areas, N failures" (README By-the-Numbers table)
EXPLOIT_TEST_COUNT_TABLE_RE = re.compile(
    r'\d+ exploit PoC modules \(\d+ source files\), \d+\+ coverage areas, \d+ failures'
)

# Matches "(N source files" — WHY doc table footnote "(261 source files — N modules share...)"
SOURCE_FILES_PAREN_RE = re.compile(r'\(\d{1,3} source files')

# Matches "**N modules, 0 failures**" — WHY doc table cell bold module count
BOLD_MODULE_FAILURES_RE = re.compile(r'\*\*\d{1,3} modules, 0 failures\*\*')

# Matches "All N registered exploit-test entries" or "All N registered exploit-PoC modules"
ALL_REGISTERED_ENTRIES_RE = re.compile(
    r'All \d+ registered exploit-(?:test entries|PoC modules) live in'
)

# Matches "-- N modules, N failure classes" (AUDIT_COVERAGE.md Verdict line)
VERDICT_MODULES_RE = re.compile(r'(-- )\d+( modules, \d+ failure classes)')

# Matches "| Audit Modules        | N (non-exploit modules) |" (AUDIT_COVERAGE.md Summary table)
SUMMARY_NONEXPLOIT_RE = re.compile(r'(\| Audit Modules\s+\| )\d+( \(non-exploit modules\) \|)')

# Matches "N-module `unified_audit_runner`" (backtick-wrapped variant used in README)
# Handled by RUNNER_MODULE_COUNT_BT_RE defined above; applied in make_replacements.

# Matches "N exploit PoCs modules + N non-exploit modules = N total"
# Used in docs/ATTACK_GUIDE.md:7 (Current assurance state line). Added 2026-05-24
# after the multi-pass review found this format was the only ATTACK_GUIDE count
# expression and no existing pattern covered it (caused 371-vs-400 drift).
EXPLOIT_PLUS_TOTAL_RE = re.compile(
    r'\b\d+ exploit PoCs? modules? \+ \d+ non-exploit modules? = \d+ total\b'
)

# Matches "across N sections" in isolation (not caught by TABLE_EXPLOIT_ROW_RE).
# Used to update standalone section-count references in AUDIT_COVERAGE.md and
# other docs that mention the section count outside the full table row context.
ACROSS_SECTIONS_RE = re.compile(r'\bacross \d+ sections\b', re.IGNORECASE)


# ── Count sources ─────────────────────────────────────────────────────────────

def count_all_modules(runner: Path) -> tuple[int, int, int]:
    """Parse ALL_MODULES[] in unified_audit_runner.cpp.

    Returns (total, exploit_poc_count, non_exploit_count).
    Counts by section name (third quoted field), not by key prefix.
    """
    text = runner.read_text()
    start = text.find("static const AuditModule ALL_MODULES")
    if start == -1:
        start = text.find("ALL_MODULES[] =")
    end = text.find("};\n\nstatic constexpr int NUM_MODULES", start)
    if start == -1 or end == -1:
        raise RuntimeError("Could not find ALL_MODULES[] in unified_audit_runner.cpp")
    block = text[start:end]
    total = len(re.findall(r'^\s+\{[^/]', block, re.MULTILINE))
    exploit = len(re.findall(r'"exploit_poc"', block))
    non_exploit = total - exploit
    return total, exploit, non_exploit


def count_sections(runner: Path) -> int:
    """Count entries in SECTIONS[] in unified_audit_runner.cpp."""
    text = runner.read_text()
    start = text.find("static const SectionInfo SECTIONS[]")
    if start == -1:
        raise RuntimeError("Could not find SECTIONS[] in unified_audit_runner.cpp")
    end = text.find("static constexpr int NUM_SECTIONS", start)
    if end == -1:
        raise RuntimeError("Could not find end of SECTIONS[] in unified_audit_runner.cpp")
    block = text[start:end]
    return len(re.findall(r'^\s+\{', block, re.MULTILINE))


def count_exploit_files(audit_dir: Path) -> int:
    return len(list(audit_dir.glob('test_exploit_*.cpp')))


# ── Replacement helpers ───────────────────────────────────────────────────────

def make_replacements(content: str,
                      total: int,
                      exploit_mods: int,
                      non_exploit: int,
                      exploit_files: int,
                      n_sections: int = 0) -> tuple[str, int]:
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

    # "N exploit PoCs modules + N non-exploit modules = N total" (ATTACK_GUIDE format)
    def _replace_exploit_plus_total(m: re.Match) -> str:
        nonlocal changes
        replacement = f'{exploit_mods} exploit PoCs modules + {non_exploit} non-exploit modules = {total} total'
        if m.group(0) != replacement:
            changes += 1
        return replacement

    new = EXPLOIT_PLUS_TOTAL_RE.sub(_replace_exploit_plus_total, new)

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

    def _sub(pattern: re.Pattern, repl: str) -> None:
        nonlocal new, changes
        result = pattern.sub(repl, new)
        if result != new:
            changes += 1
        new = result

    # PAREN_MODULES_RE: replaces ALL occurrences (count=0 is the default for re.sub)
    # e.g. "(N modules," or "across N modules" → updated counts
    def _replace_paren_modules(m: re.Match) -> str:
        nonlocal changes
        prefix = 'across' if m.group(0).lstrip().startswith('across') else '('
        sep = ' ' if prefix == 'across' else ' '
        inner = m.group(0)
        if 'non-exploit' in inner:
            replacement = f'{prefix} {non_exploit} non-exploit modules'
        else:
            replacement = f'{prefix} {total} modules'
        if inner != replacement:
            changes += 1
        return replacement

    new = PAREN_MODULES_RE.sub(_replace_paren_modules, new)

    _sub(ALL_BOTH_RE,
         f'All {non_exploit} non-exploit audit modules and all {exploit_mods} exploit PoCs')
    _sub(TABLE_EXPLOIT_ROW_RE,
         f'{non_exploit} non-exploit modules + {exploit_mods} exploit PoCs across {n_sections} sections, 0 failures')
    # Standalone "across N sections" references (outside the full table-row pattern).
    if n_sections > 0:
        _sub(ACROSS_SECTIONS_RE, f'across {n_sections} sections')
    _sub(RUNNER_MODULE_COUNT_RE, f'{total}-module unified_audit_runner')
    _sub(RUNNER_MODULE_COUNT_BT_RE, f'{total}-module `unified_audit_runner`')
    _sub(REGISTERED_STYLE_RE,
         f'{exploit_mods} registered exploit-style PoC test modules ({exploit_files} test files)')
    _sub(EXPLOIT_FILES_REGISTERED_RE,
         f'{exploit_files} exploit PoC test files ({exploit_mods} registered as runner modules')
    _sub(SUITE_HEADER_RE, f'Exploit PoC Test Suite ({exploit_mods} Tests,')
    _sub(EXPLOIT_POCS_ALL_RE,
         f'{exploit_files} exploit PoC test files (all {exploit_mods} registered as runner modules')
    _sub(EXPLOIT_TEST_COUNT_TABLE_RE,
         f'{exploit_mods} exploit PoC modules ({exploit_files} source files), 20+ coverage areas, 0 failures')
    _sub(ALL_REGISTERED_ENTRIES_RE,
         f'All {exploit_mods} registered exploit-PoC modules live in')

    # "N non-exploit modules + N exploit-PoC modules (N total)" — WHY doc table
    _sub(TABLE_WHY_DOC_RE,
         f'{non_exploit} non-exploit modules + {exploit_mods} exploit-PoC modules ({total} total)')

    # "All N exploit PoCs modules pass" (plural variant)
    _sub(ALL_EXPLOIT_PLURAL_RE, f'All {exploit_mods} exploit PoCs modules pass')

    # "(N source files" — WHY doc table footnote
    def _replace_source_files(m: re.Match) -> str:
        nonlocal changes
        replacement = f'({exploit_files} source files'
        if m.group(0) != replacement:
            changes += 1
        return replacement

    new = SOURCE_FILES_PAREN_RE.sub(_replace_source_files, new)

    # "**N modules, 0 failures**" — WHY doc table cell
    def _replace_bold_module_failures(m: re.Match) -> str:
        nonlocal changes
        replacement = f'**{exploit_mods} modules, 0 failures**'
        if m.group(0) != replacement:
            changes += 1
        return replacement

    new = BOLD_MODULE_FAILURES_RE.sub(_replace_bold_module_failures, new)

    # "-- N modules, N failure classes" (AUDIT_COVERAGE.md Verdict line)
    def _replace_verdict_modules(m: re.Match) -> str:
        nonlocal changes
        replacement = m.group(1) + str(total) + m.group(2)
        if m.group(0) != replacement:
            changes += 1
        return replacement

    new = VERDICT_MODULES_RE.sub(_replace_verdict_modules, new)

    # "| Audit Modules | N (non-exploit modules) |" (AUDIT_COVERAGE.md Summary table)
    def _replace_summary_nonexploit(m: re.Match) -> str:
        nonlocal changes
        replacement = m.group(1) + str(non_exploit) + m.group(2)
        if m.group(0) != replacement:
            changes += 1
        return replacement

    new = SUMMARY_NONEXPLOIT_RE.sub(_replace_summary_nonexploit, new)

    # "N exploit PoCs / M modules" — compact README/GitHub summary form.
    def _replace_exploit_pocs_slash_modules(m: re.Match) -> str:
        nonlocal changes
        replacement = f'{exploit_mods} exploit PoCs / {total} modules'
        if m.group(0) != replacement:
            changes += 1
        return replacement

    new = EXPLOIT_POCS_SLASH_MODULES_RE.sub(_replace_exploit_pocs_slash_modules, new)

    # "N exploit PoCs" / "N exploit-PoCs" — short form without "modules" (3+ digit only)
    def _replace_exploit_pocs_simple(m: re.Match) -> str:
        nonlocal changes
        # Preserve the spacing/hyphen style from the match
        full = m.group(0)
        replacement = re.sub(r'\d{3,}', str(exploit_mods), full, count=1)
        if full != replacement:
            changes += 1
        return replacement

    new = EXPLOIT_POCS_SIMPLE_RE.sub(_replace_exploit_pocs_simple, new)

    return new, changes


# ── Files to update ───────────────────────────────────────────────────────────

DOC_FILES = [
    'docs/WHY_ULTRAFASTSECP256K1.md',
    'README.md',
    'docs/AUDIT_GUIDE.md',
    'docs/AUDIT_REPORT.md',
    'docs/AUDIT_TRACEABILITY.md',
    'docs/ASSURANCE_LEDGER.md',
    '.github/workflows/security-audit.yml',
    '.github/workflows/audit-report.yml',
    # Additional docs that contain canonical exploit PoC count references
    # and are checked by check_version_sync.py. Must be kept in sync.
    'docs/ATTACK_GUIDE.md',
    'docs/AUDIT_COVERAGE.md',
    'docs/AUDIT_PHILOSOPHY.md',
    'docs/AUDIT_READINESS_REPORT_v1.md',
    'docs/CAAS_REVIEWER_QUICKSTART.md',
    'docs/BITCOIN_CORE_BACKEND_EVIDENCE.md',
    'docs/EXTERNAL_AUDIT_BUNDLE.json',
    'docs/AI_AUDIT_PROTOCOL.md',
    'docs/FORTRESS_ROADMAP.md',
    'docs/BACKEND_ASSURANCE_MATRIX.md',
    # Citation metadata: .zenodo.json description carries hardcoded counts
    # (e.g. "270 exploit proof-of-concept regression tests") which previously
    # drifted. Scan and substitute via the same regex set.
    '.zenodo.json',
    # CHANGELOG.md and docs/AUDIT_CHANGELOG.md: NOT in this list, on purpose.
    # The regex set targets generic patterns ("\d+ modules skipped",
    # "\d+ exploit PoC modules", etc.) which match historical/dated section
    # text and over-rewrite it as if it were a live claim. CHANGELOG.md is
    # checked by ci/check_changelog_canonical_consistency.py instead, which
    # only validates the topmost ([Unreleased] or most recent [X.Y.Z])
    # section; docs/AUDIT_CHANGELOG.md is a pure historical log and is never
    # count-checked (each entry describes a past point-in-time count, not the
    # current state — see the SKIP_DYNAMIC set below, which keeps it, and
    # docs/EXPLOIT_TEST_CATALOG.md's dated growth-log, out of the dynamic
    # docs/*.md sweep too).
]


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dry-run', '--check', action='store_true',
                        help='Show what would change without writing files (--check is an alias)')
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
    n_sections = count_sections(runner)

    print(f'Counts from audit/unified_audit_runner.cpp:')
    print(f'  ALL_MODULES total : {total}')
    print(f'  exploit-PoC mods  : {exploit_mods}')
    print(f'  non-exploit mods  : {non_exploit}')
    print(f'  exploit .cpp files: {exploit_files}')
    print(f'  SECTIONS count    : {n_sections}')
    print()

    total_changed_files = 0
    total_changed_lines = 0

    # Combine fixed DOC_FILES with dynamic scan of all docs/*.md files.
    # This ensures no file is missed when new docs are added.
    # Skip changelog/catalog files that contain historical (non-current, dated)
    # counts — the regex set has no date-awareness and will over-rewrite a
    # dated "as of that point in time" number as if it were a live claim
    # (see docs/AUDIT_CHANGELOG.md incident: seven dated entries had their
    # original 272-exploit-PoC counts silently rewritten to the current count
    # before this file was excluded from DOC_FILES; docs/EXPLOIT_TEST_CATALOG.md's
    # 2026-04-11 dated growth-log row had its "442 modules" similarly rewritten
    # to the current total before being added here).
    SKIP_DYNAMIC = {
        'CHANGELOG.md', 'AUDIT_CHANGELOG.md', 'ROADMAP.md',
        'AUDIT_REPORT.md', 'RELEASE_NOTES.md', 'EXPLOIT_TEST_CATALOG.md',
    }
    dynamic_paths = []
    docs_dir = BASE / 'docs'
    if docs_dir.exists():
        for p in sorted(docs_dir.glob('*.md')):
            if p.name not in SKIP_DYNAMIC:
                rel = str(p.relative_to(BASE))
                if rel not in DOC_FILES:
                    dynamic_paths.append(rel)

    all_files = list(DOC_FILES) + dynamic_paths

    for rel in all_files:
        path = BASE / rel
        if not path.exists():
            if args.verbose:
                print(f'  SKIP (not found): {rel}')
            continue

        original = path.read_text()
        updated, n = make_replacements(original, total, exploit_mods, non_exploit, exploit_files, n_sections)

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
        if total_changed_files > 0:
            print(f'[DRIFT] would update {total_changed_files} file(s), {total_changed_lines} occurrence(s).')
            print('Run: python3 ci/sync_all_docs.py')
            return 1
        print('[OK] All docs already up to date.')
    elif total_changed_files == 0:
        print('All docs already up to date.')
    else:
        print(f'Updated {total_changed_files} file(s), {total_changed_lines} occurrence(s).')

    return 0


if __name__ == '__main__':
    sys.exit(main())
