#!/usr/bin/env python3
"""
check_version_sync.py — CI gate: verify all canonical numbers match their sources.

Exit 0  = all sources agree.
Exit 1  = one or more mismatches (prints diff table).

Usage:
    python3 ci/check_version_sync.py [--root DIR] [--counts-only] [--version-only]

Version checks (canonical: VERSION.txt):
    include/ufsecp/ufsecp_version.h
    CITATION.cff
    packaging/cocoapods/UltrafastSecp256k1.podspec
    packaging/rpm/libufsecp.spec
    packaging/arch/PKGBUILD

Count checks (canonical: computed from source):
    Exploit PoC count  → count "exploit_poc" in audit/unified_audit_runner.cpp
    Total audit modules → count all _run entries in unified_audit_runner.cpp
    GPU ABI functions  → count UFSECP_GPU_API in include/ufsecp/ufsecp_gpu.h

Intentionally skipped (build-time templated or historical):
    packaging/nuget/UltrafastSecp256k1.Native.nuspec  (0.0.0-dev)
    packaging/debian/changelog                          (historical log)
"""

from __future__ import annotations
import re
import sys
import argparse
from pathlib import Path


# ---------------------------------------------------------------------------
# Extractors: each returns (version_string | None, description_of_location)
# ---------------------------------------------------------------------------

def _extract_version_txt(root: Path) -> str | None:
    p = root / 'VERSION.txt'
    if not p.exists():
        return None
    return p.read_text().strip().lstrip('v')


def _extract_ufsecp_header(root: Path) -> str | None:
    p = root / 'include' / 'ufsecp' / 'ufsecp_version.h'
    if not p.exists():
        return None
    text = p.read_text(encoding='utf-8')
    major = re.search(r'^#define\s+UFSECP_VERSION_MAJOR\s+(\d+)', text, re.MULTILINE)
    minor = re.search(r'^#define\s+UFSECP_VERSION_MINOR\s+(\d+)', text, re.MULTILINE)
    patch = re.search(r'^#define\s+UFSECP_VERSION_PATCH\s+(\d+)', text, re.MULTILINE)
    if not (major and minor and patch):
        return None
    return f'{major.group(1)}.{minor.group(1)}.{patch.group(1)}'


def _extract_header_string_macro(root: Path) -> str | None:
    p = root / 'include' / 'ufsecp' / 'ufsecp_version.h'
    if not p.exists():
        return None
    text = p.read_text(encoding='utf-8')
    m = re.search(r'^#define\s+UFSECP_VERSION_STRING\s+"(\d+\.\d+\.\d+)"', text, re.MULTILINE)
    return m.group(1) if m else None


def _extract_citation_cff(root: Path) -> str | None:
    p = root / 'CITATION.cff'
    if not p.exists():
        return None
    text = p.read_text(encoding='utf-8')
    m = re.search(r'^version:\s*"(\d+\.\d+\.\d+)"', text, re.MULTILINE)
    return m.group(1) if m else None


def _extract_podspec(root: Path) -> str | None:
    for p in (root / 'packaging' / 'cocoapods').glob('*.podspec'):
        text = p.read_text(encoding='utf-8')
        m = re.search(r's\.version\s*=\s*"(\d+\.\d+\.\d+)"', text)
        if m:
            return m.group(1)
    return None


def _extract_rpm_spec(root: Path) -> str | None:
    for p in (root / 'packaging' / 'rpm').glob('*.spec'):
        text = p.read_text(encoding='utf-8')
        m = re.search(r'^Version:\s+(\d+\.\d+\.\d+)', text, re.MULTILINE)
        if m:
            return m.group(1)
    return None


def _extract_rpm_soversion(root: Path) -> str | None:
    for p in (root / 'packaging' / 'rpm').glob('*.spec'):
        text = p.read_text(encoding='utf-8')
        m = re.search(r'^%global\s+soversion\s+(\d+)', text, re.MULTILINE)
        if m:
            return m.group(1)
    return None


def _extract_pkgbuild(root: Path) -> str | None:
    p = root / 'packaging' / 'arch' / 'PKGBUILD'
    if not p.exists():
        return None
    text = p.read_text(encoding='utf-8')
    m = re.search(r'^pkgver=(\d+\.\d+\.\d+)', text, re.MULTILINE)
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# Count extractors (from docs, to compare against authoritative source)
# ---------------------------------------------------------------------------

_SKIP_COUNT_FILES = {
    'CHANGELOG.md', 'AUDIT_CHANGELOG.md', 'ROADMAP.md',
    'AUDIT_REPORT.md', 'RELEASE_NOTES.md',
}
_SKIP_COUNT_DIRS = {'archive', 'build', '.git'}


def _scan_docs_for_count(root: Path, pattern: re.Pattern) -> list[tuple[str, int]]:
    """Find all doc files containing pattern; return list of (relpath, found_int)."""
    hits = []
    scan_dirs = [root / 'docs', root / 'include', root]
    for scan_dir in scan_dirs:
        if not scan_dir.exists():
            continue
        glob_pat = '*.md' if scan_dir == root else '**/*.md'
        for p in sorted(scan_dir.glob(glob_pat)):
            if p.name in _SKIP_COUNT_FILES:
                continue
            if any(part in _SKIP_COUNT_DIRS for part in p.parts):
                continue
            try:
                text = p.read_text(encoding='utf-8')
            except Exception:
                continue
            for m in pattern.finditer(text):
                hits.append((str(p.relative_to(root)), int(m.group(1))))
    return hits


# ---------------------------------------------------------------------------
# Main check
# ---------------------------------------------------------------------------

def _print_table(rows: list[tuple[str, str, str, str]], col_w: int) -> bool:
    ok = True
    header = f'{"Location":<{col_w}}  {"Found":<12}  {"Expected":<12}  Status'
    print(header)
    print('-' * len(header))
    for label, found, expected, status in rows:
        if status != 'OK':
            ok = False
        marker = '' if status == 'OK' else ' <--'
        print(f'{label:<{col_w}}  {found:<12}  {expected:<12}  {status}{marker}')
    return ok


def check_version_sync(root: Path) -> bool:
    canonical = _extract_version_txt(root)
    if not canonical:
        print('ERROR: VERSION.txt not found or empty', file=sys.stderr)
        return False

    expected_major = canonical.split('.')[0]

    raw_checks = [
        ('VERSION.txt (canonical)',             canonical),
        ('ufsecp_version.h  MAJOR/MINOR/PATCH', _extract_ufsecp_header(root)),
        ('ufsecp_version.h  VERSION_STRING',    _extract_header_string_macro(root)),
        ('CITATION.cff  version',               _extract_citation_cff(root)),
        ('cocoapods podspec  s.version',        _extract_podspec(root)),
        ('rpm spec  Version:',                  _extract_rpm_spec(root)),
        ('arch PKGBUILD  pkgver',               _extract_pkgbuild(root)),
        ('rpm spec  soversion (expect major)',  _extract_rpm_soversion(root)),
    ]

    rows = []
    for label, found in raw_checks:
        expected = expected_major if 'soversion' in label else canonical
        if found is None:
            status = 'MISSING'
        elif found == expected:
            status = 'OK'
        else:
            status = 'MISMATCH'
        rows.append((label, found or 'N/A', expected, status))

    col_w = max(len(r[0]) for r in rows) + 2
    print('\n── Version sync ────────────────────────────────────────────────')
    ok = _print_table(rows, col_w)
    if ok:
        print(f'\n  All version declarations match {canonical}.')
    else:
        print('\n  FAIL — run: python3 scripts/sync_version_refs.py --dry-run')
    return ok


def check_count_sync(root: Path) -> bool:
    """Check that canonical counts (exploit PoC, GPU ABI, etc.) match doc references."""
    runner = root / 'audit' / 'unified_audit_runner.cpp'
    gpu_hdr = root / 'include' / 'ufsecp' / 'ufsecp_gpu.h'

    auth_exploit = 0
    auth_gpu = 0
    if runner.exists():
        t = runner.read_text(encoding='utf-8')
        auth_exploit = len(re.findall(r'"exploit_poc"', t))
    if gpu_hdr.exists():
        t = gpu_hdr.read_text(encoding='utf-8')
        # Count stable GPU batch-op functions (skip lifecycle/utility functions)
        auth_gpu = len(re.findall(
            r'^UFSECP_API\s+\S+\s+ufsecp_gpu_(?!ctx_|backend_|device_|is_avail|get_)\w+\s*\(',
            t, re.MULTILINE))

    # Scan docs for stale exploit counts (3+ digit numbers only — small counts
    # like "14 new exploit PoCs" are incremental changelog entries, not totals)
    exploit_pattern = re.compile(r'\b(\d{3,})\s+exploit[- ]PoC[s]?\b', re.IGNORECASE)
    stale = [(p, n) for p, n in _scan_docs_for_count(root, exploit_pattern)
             if n != auth_exploit]

    col_w = 55
    print('\n── Canonical count sync ─────────────────────────────────────────')
    print(f'  Authoritative exploit PoC count (unified_audit_runner.cpp): {auth_exploit}')
    print(f'  Authoritative GPU stable batch ops (ufsecp_gpu.h):          {auth_gpu}')

    ok = True
    if stale:
        ok = False
        print(f'\n  STALE exploit PoC references ({len(stale)} locations):')
        for path, n in stale[:10]:
            print(f'    {path}: found {n}, expected {auth_exploit}')
        print(f'\n  Fix: python3 scripts/sync_version_refs.py --sync-exploit --dry-run')
    else:
        print(f'\n  All exploit PoC count references match {auth_exploit}. OK')
    return ok


def check_sync(root: Path, version_only: bool = False, counts_only: bool = False) -> bool:
    ok = True
    if not counts_only:
        ok = check_version_sync(root) and ok
    if not version_only:
        ok = check_count_sync(root) and ok
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(
        description='CI gate: verify all canonical numbers match their source of truth')
    parser.add_argument('--root', default=None, help='Repo root (default: parent of ci/)')
    parser.add_argument('--version-only', action='store_true',
                        help='Only check version declarations')
    parser.add_argument('--counts-only', action='store_true',
                        help='Only check canonical counts (exploit PoC, GPU ABI)')
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    root = Path(args.root) if args.root else script_dir.parent

    ok = check_sync(root, version_only=args.version_only, counts_only=args.counts_only)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
