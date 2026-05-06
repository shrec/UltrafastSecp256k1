#!/usr/bin/env python3
"""
check_version_sync.py — CI gate: verify all version declarations match VERSION.txt.

Exit 0  = all sources agree.
Exit 1  = one or more mismatches (prints diff table).

Usage:
    python3 ci/check_version_sync.py [--root DIR]

Checked locations:
    VERSION.txt                                (canonical)
    include/ufsecp/ufsecp_version.h            (C header static copy)
    CITATION.cff                               (GitHub citation)
    packaging/cocoapods/UltrafastSecp256k1.podspec
    packaging/rpm/libufsecp.spec
    packaging/arch/PKGBUILD

Intentionally skipped (build-time templated or historical):
    packaging/nuget/UltrafastSecp256k1.Native.nuspec  (version = 0.0.0-dev)
    packaging/debian/changelog                          (historical log)
    include/ufsecp/ufsecp_version.h.in                 (CMake template)
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
# Main check
# ---------------------------------------------------------------------------

def check_sync(root: Path) -> bool:
    canonical = _extract_version_txt(root)
    if not canonical:
        print('ERROR: VERSION.txt not found or empty', file=sys.stderr)
        return False

    expected_major = canonical.split('.')[0]

    checks: list[tuple[str, str | None, bool]] = [
        # (label, extracted_value, should_equal_canonical)
        ('VERSION.txt (canonical)',             canonical,                        True),
        ('ufsecp_version.h  MAJOR/MINOR/PATCH', _extract_ufsecp_header(root),    True),
        ('ufsecp_version.h  VERSION_STRING',    _extract_header_string_macro(root), True),
        ('CITATION.cff  version',               _extract_citation_cff(root),     True),
        ('cocoapods podspec  s.version',        _extract_podspec(root),           True),
        ('rpm spec  Version:',                  _extract_rpm_spec(root),          True),
        ('arch PKGBUILD  pkgver',               _extract_pkgbuild(root),          True),
    ]

    # soversion should match major
    soversion = _extract_rpm_soversion(root)
    checks.append(('rpm spec  %global soversion (expect major)',
                   soversion, False))  # handled separately below

    col_w = max(len(c[0]) for c in checks) + 2
    header = f'{"Location":<{col_w}}  {"Found":<12}  {"Expected":<12}  Status'
    print(header)
    print('-' * len(header))

    ok = True
    for label, found, _compare_canonical in checks:
        if label.startswith('rpm spec  %global soversion'):
            expected = expected_major
            status = 'OK' if found == expected else 'MISMATCH'
        else:
            expected = canonical
            if found is None:
                status = 'MISSING'
            elif found == expected:
                status = 'OK'
            else:
                status = 'MISMATCH'

        if status != 'OK':
            ok = False

        marker = '' if status == 'OK' else ' <--'
        print(f'{label:<{col_w}}  {(found or "N/A"):<12}  {expected:<12}  {status}{marker}')

    print()
    if ok:
        print(f'All version declarations match {canonical}.')
    else:
        print('FAIL: version mismatch(es) detected. Run:')
        print('  python3 scripts/sync_version_refs.py --dry-run')
        print('to preview fixes, then drop --dry-run to apply.')
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description='CI gate: verify all version refs match VERSION.txt')
    parser.add_argument('--root', default=None, help='Repo root (default: parent of ci/)')
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    root = Path(args.root) if args.root else script_dir.parent

    ok = check_sync(root)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
