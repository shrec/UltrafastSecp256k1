#!/usr/bin/env python3
"""
verify_build_hardening.py — Binary hardening audit for UltrafastSecp256k1
==========================================================================

Checks the compiled shared library (or executable) for the following
security-relevant binary properties:

  Canary       -fstack-protector-strong symbols present  (GCC/Clang)
  FORTIFY      _FORTIFY_SOURCE wrappers present          (glibc)
  PIE          Position-independent executable           (readelf/checksec)
  RELRO        Read-only relocations (partial or full)   (readelf)
  NX           Non-executable stack                      (GNU_STACK RW-only)
  Bindnow      Immediate binding (DT_BIND_NOW or BIND_NOW in flags)

The script auto-locates the built library under build/ or build_opencl/
but accepts an explicit --library path.

Exit codes:
  0  All critical checks pass
  1  One or more CRITICAL checks failed
  2  Tool dependency missing (readelf not found)
  3  Library target not found
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
LIB_ROOT   = SCRIPT_DIR.parent

# Candidate build output dirs, searched in order
BUILD_SEARCH_DIRS = [
    LIB_ROOT / 'build_opencl',
    LIB_ROOT / 'build',
    LIB_ROOT / 'build_rel',
]
LIB_PATTERNS = [
    'libufsecp.so*',
    'libs/libufsecp.so*',
    'libsecp256k1-fast.so*',
    '**/libufsecp.so*',
    '**/libsecp256k1*.so*',
]


# ---------------------------------------------------------------------------
# Library discovery
# ---------------------------------------------------------------------------

def find_library(explicit: Path | None) -> Path:
    if explicit:
        if not explicit.exists():
            raise FileNotFoundError(f'Explicit library path not found: {explicit}')
        return explicit

    for build_dir in BUILD_SEARCH_DIRS:
        if not build_dir.is_dir():
            continue
        for pattern in LIB_PATTERNS:
            matches = sorted(build_dir.glob(pattern))
            if matches:
                # Prefer the real file over symlinks
                real = [m for m in matches if not m.is_symlink()]
                return real[0] if real else matches[0]

    raise FileNotFoundError(
        'No built library found. Run cmake --build first, or pass --library.'
    )


# ---------------------------------------------------------------------------
# readelf helpers
# ---------------------------------------------------------------------------

def run_readelf(flag: str, path: Path) -> str:
    cmd = ['readelf', flag, str(path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout


def check_nx(path: Path) -> tuple[bool, str]:
    """NX: GNU_STACK segment must be RW (not RWE/RWX)."""
    stdout = run_readelf('-W -l', path)
    for line in stdout.splitlines():
        if 'GNU_STACK' in line:
            # Fields: Type Offset VirtAddr PhysAddr FileSiz MemSiz Flg Align
            # Flg column: RW = non-exec (good), RWE = executable (bad)
            parts = line.split()
            flags_col = next((p for p in parts if re.fullmatch(r'[RWE]+', p)), '')
            is_nx = 'E' not in flags_col
            return is_nx, f'GNU_STACK flags={flags_col!r}'
    return False, 'GNU_STACK segment not found (static binary or stripped)'


def check_pie(path: Path) -> tuple[bool, str]:
    """PIE: ELF type must be DYN (shared object / PIE executable)."""
    stdout = run_readelf('-h', path)
    for line in stdout.splitlines():
        if 'Type:' in line:
            is_pie = 'DYN' in line
            return is_pie, line.strip()
    return False, 'ELF Type line not found'


def check_relro(path: Path) -> tuple[bool, str]:
    """RELRO: look for GNU_RELRO program header."""
    stdout = run_readelf('-W -l', path)
    full = stdout
    partial = 'GNU_RELRO' in full
    # Full RELRO additionally requires BIND_NOW
    bindnow_dyn = check_bindnow_dynamic(path)
    if partial and bindnow_dyn:
        return True, 'Full RELRO (GNU_RELRO + BIND_NOW)'
    if partial:
        return True, 'Partial RELRO (GNU_RELRO present)'
    return False, 'No GNU_RELRO segment found'


def check_bindnow_dynamic(path: Path) -> bool:
    stdout = run_readelf('-W -d', path)
    return '(BIND_NOW)' in stdout or 'BIND_NOW' in stdout


def check_canary(path: Path) -> tuple[bool, str]:
    """Stack canary: __stack_chk_fail or __stack_chk_guard symbol present."""
    stdout = run_readelf('-s', path)
    has_chk = '__stack_chk_fail' in stdout or '__stack_chk_guard' in stdout
    return has_chk, '__stack_chk_fail present' if has_chk else '__stack_chk_fail NOT found'


def check_fortify(path: Path) -> tuple[bool, str]:
    """FORTIFY_SOURCE: __*_chk symbols (e.g. __memcpy_chk) present."""
    stdout = run_readelf('-s', path)
    chk_syms = [line for line in stdout.splitlines() if '_chk@' in line or '_chk@@' in line]
    has_fortify = len(chk_syms) > 0
    detail = f'{len(chk_syms)} _chk symbol(s) found' if has_fortify else 'no _chk symbols found'
    return has_fortify, detail


# ---------------------------------------------------------------------------
# Severity model
# ---------------------------------------------------------------------------

CHECKS = [
    # (id,        label,      function,       critical)
    ('NX',        'Non-executable stack (NX/GNU_STACK)',      check_nx,     True),
    ('PIE',       'Position-independent executable (PIE/DYN)', check_pie,   True),
    ('RELRO',     'Read-only relocations (RELRO)',            check_relro,  True),
    ('CANARY',    'Stack canary (-fstack-protector-strong)',   check_canary, True),
    ('FORTIFY',   'FORTIFY_SOURCE (_chk symbols)',            check_fortify, False),
]


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------

def render_text(results: list[dict]) -> str:
    lines = ['Build Hardening Audit — UltrafastSecp256k1', '=' * 50]
    for r in results:
        status = 'PASS' if r['pass'] else ('FAIL' if r['critical'] else 'WARN')
        lines.append(f"  [{status:4s}] {r['label']}")
        lines.append(f"         {r['detail']}")
    critical_fails = [r for r in results if not r['pass'] and r['critical']]
    lines.append('')
    if critical_fails:
        lines.append(f'CRITICAL FAILURES: {len(critical_fails)}')
        for r in critical_fails:
            lines.append(f"  - {r['id']}: {r['label']}")
    else:
        lines.append('All critical hardening checks PASSED.')
    return '\n'.join(lines) + '\n'


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--library', type=Path, default=None,
                   help='Path to the .so to audit (auto-detected if omitted).')
    p.add_argument('--output', type=Path, default=None,
                   help='Write JSON report to this file.')
    p.add_argument('--warn-only', action='store_true',
                   help='Exit 0 even when critical checks fail (CI advisory mode).')
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if not shutil.which('readelf'):
        print('error: readelf not found — install binutils', file=sys.stderr)
        return 2

    try:
        lib = find_library(args.library)
    except FileNotFoundError as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 3

    print(f'Auditing: {lib}')

    results = []
    for check_id, label, fn, critical in CHECKS:
        passed, detail = fn(lib)
        results.append({
            'id':       check_id,
            'label':    label,
            'pass':     passed,
            'detail':   detail,
            'critical': critical,
        })

    print(render_text(results))

    if args.output:
        report = {
            'library': str(lib),
            'checks':  results,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        print(f'JSON report written to {args.output}')

    any_critical_fail = any(not r['pass'] and r['critical'] for r in results)
    if any_critical_fail and not args.warn_only:
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
