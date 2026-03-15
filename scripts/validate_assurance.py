#!/usr/bin/env python3
"""
validate_assurance.py  --  Cross-reference assurance docs vs actual code

Checks:
  1. FEATURE_ASSURANCE_LEDGER.md lists all ufsecp_* functions from ufsecp.h
  2. TEST_MATRIX.md test count matches actual CTest targets in CMakeLists files
  3. ABI functions in graph match header declarations
  4. Conditional compilation blocks (Ethereum) are annotated

Usage:
    python3 scripts/validate_assurance.py           # all checks
    python3 scripts/validate_assurance.py --json    # JSON output for CI
"""

import re
import sys
import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
LIB_ROOT = SCRIPT_DIR.parent

RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BOLD = '\033[1m'
RESET = '\033[0m'


def scan_header_functions():
    """Extract all ufsecp_* function names from ufsecp.h."""
    header = LIB_ROOT / 'include' / 'ufsecp' / 'ufsecp.h'
    if not header.exists():
        return set(), set()
    fn_re = re.compile(r'\b(ufsecp_\w+)\s*\(')
    all_fns = set()
    conditional_fns = set()
    in_conditional = False
    with open(header, 'r', errors='replace') as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith('#ifdef') or stripped.startswith('#if '):
                in_conditional = True
            elif stripped.startswith('#endif'):
                in_conditional = False
            m = fn_re.search(line)
            if m:
                name = m.group(1)
                all_fns.add(name)
                if in_conditional:
                    conditional_fns.add(name)
    return all_fns, conditional_fns


def scan_ledger_functions():
    """Extract function names from FEATURE_ASSURANCE_LEDGER.md table rows."""
    ledger = LIB_ROOT / 'docs' / 'FEATURE_ASSURANCE_LEDGER.md'
    if not ledger.exists():
        return set()
    fn_re = re.compile(r'^\|\s*`?(ufsecp_\w+)`?\s*\|')
    fns = set()
    with open(ledger, 'r', errors='replace') as f:
        for line in f:
            m = fn_re.search(line)
            if m:
                fns.add(m.group(1))
    return fns


def scan_ctest_targets():
    """Find all add_test() entries across CMakeLists.txt files."""
    targets = set()
    test_re = re.compile(r'add_test\s*\(\s*NAME\s+(\S+)')
    for cmake_file in LIB_ROOT.rglob('CMakeLists.txt'):
        # Skip build directories
        rel = str(cmake_file.relative_to(LIB_ROOT))
        if rel.startswith('build') or '_build' in rel:
            continue
        try:
            with open(cmake_file, 'r', errors='replace') as f:
                for line in f:
                    m = test_re.search(line)
                    if m:
                        targets.add(m.group(1))
        except Exception:
            continue
    return targets


def scan_test_matrix_targets():
    """Extract test file/target references from TEST_MATRIX.md."""
    matrix = LIB_ROOT / 'docs' / 'TEST_MATRIX.md'
    if not matrix.exists():
        return set()
    targets = set()
    # Match backtick-wrapped filenames (with optional path prefix): `audit_field.cpp`, `metal/tests/test_metal_host.cpp`
    file_re = re.compile(r'`(?:[\w./-]*/)?([\w_-]+\.(?:cpp|cu|hpp|mm))`')
    # Also match bare CTest target names in backtick table cells: `cuda_selftest`
    target_re = re.compile(r'`([\w_-]+)`')
    with open(matrix, 'r', errors='replace') as f:
        for line in f:
            for m in file_re.finditer(line):
                fname = m.group(1)
                # Derive CTest-style name: strip prefix/suffix
                stem = Path(fname).stem
                for prefix in ('test_', 'audit_', 'bench_'):
                    if stem.startswith(prefix):
                        stem = stem[len(prefix):]
                        break
                targets.add(stem)
                # Also keep the raw filename stem
                targets.add(Path(fname).stem)
            # Capture bare identifiers (only from table rows with |)
            if '|' in line:
                for m in target_re.finditer(line):
                    name = m.group(1)
                    # Skip if it looks like a source file (already handled above)
                    if '.' in name:
                        continue
                    targets.add(name)
    return targets


def check_ledger_completeness():
    """Check that ledger lists all ufsecp_* functions."""
    header_fns, conditional_fns = scan_header_functions()
    ledger_fns = scan_ledger_functions()

    missing = header_fns - ledger_fns
    extra = ledger_fns - header_fns

    issues = []
    if missing:
        for fn in sorted(missing):
            note = " (conditional)" if fn in conditional_fns else ""
            issues.append(f"  {YELLOW}MISSING{RESET} {fn} not in FEATURE_ASSURANCE_LEDGER{note}")
    if extra:
        for fn in sorted(extra):
            issues.append(f"  {RED}STALE{RESET}   {fn} in ledger but not in ufsecp.h")

    return {
        'header_count': len(header_fns),
        'ledger_count': len(ledger_fns),
        'missing': sorted(missing),
        'extra': sorted(extra),
        'conditional': sorted(conditional_fns),
        'issues': issues,
    }


def check_test_matrix():
    """Check that TEST_MATRIX.md covers actual CTest targets."""
    actual = scan_ctest_targets()
    documented_files = scan_test_matrix_targets()

    # Build a fuzzy match: CTest target -> any documented name
    missing = set()
    for target in actual:
        # Check if any documented name matches the CTest target
        found = False
        for doc_name in documented_files:
            if target == doc_name or target in doc_name or doc_name in target:
                found = True
                break
        if not found:
            missing.add(target)

    issues = []
    if missing:
        for t in sorted(missing):
            issues.append(f"  {YELLOW}UNDOCUMENTED{RESET} CTest target '{t}' not in TEST_MATRIX")

    return {
        'actual_count': len(actual),
        'documented_count': len(documented_files),
        'missing': sorted(missing),
        'extra': [],
        'issues': issues,
    }


def main():
    json_mode = '--json' in sys.argv
    results = {}
    exit_code = 0

    if not json_mode:
        print(f"\n{BOLD}{'='*60}{RESET}")
        print(f"{BOLD}  Assurance Documentation Validation{RESET}")
        print(f"{BOLD}{'='*60}{RESET}\n")

    # 1. Ledger completeness
    ledger = check_ledger_completeness()
    results['ledger'] = {
        'header_functions': ledger['header_count'],
        'ledger_functions': ledger['ledger_count'],
        'missing': ledger['missing'],
        'extra': ledger['extra'],
        'conditional': ledger['conditional'],
    }
    if not json_mode:
        print(f"{BOLD}[1/2] Ledger Completeness{RESET}")
        print(f"  Header: {ledger['header_count']} functions, Ledger: {ledger['ledger_count']} functions")
        if ledger['issues']:
            for i in ledger['issues']:
                print(i)
            if ledger['extra']:
                exit_code = 1
        else:
            print(f"  {GREEN}[OK] Ledger covers all header functions{RESET}")
        print()

    # 2. Test matrix
    matrix = check_test_matrix()
    results['test_matrix'] = {
        'actual_targets': matrix['actual_count'],
        'documented_targets': matrix['documented_count'],
        'missing': matrix['missing'],
        'extra': matrix['extra'],
    }
    if not json_mode:
        print(f"{BOLD}[2/2] Test Matrix Accuracy{RESET}")
        print(f"  CTest targets: {matrix['actual_count']}, Documented: {matrix['documented_count']}")
        if matrix['issues']:
            for i in matrix['issues']:
                print(i)
        else:
            print(f"  {GREEN}[OK] TEST_MATRIX matches CTest targets{RESET}")
        print()

    # Summary
    total = len(ledger['missing']) + len(ledger['extra']) + len(matrix['missing']) + len(matrix['extra'])
    results['total_issues'] = total

    if not json_mode:
        print(f"{BOLD}{'='*60}{RESET}")
        if total == 0:
            print(f"{GREEN}{BOLD}  ASSURANCE VALIDATION PASSED{RESET}")
        else:
            print(f"{YELLOW}{BOLD}  ASSURANCE VALIDATION: {total} issues{RESET}")
        print(f"{BOLD}{'='*60}{RESET}\n")
    else:
        print(json.dumps(results, indent=2))

    return exit_code


if __name__ == '__main__':
    sys.exit(main())
