#!/usr/bin/env python3
"""
release_diff.py -- Generate release diff summary between two tags/commits.

Produces a structured report of:
  - New/removed ABI functions (ufsecp_*)
  - Changed test targets
  - Security pattern changes
  - Doc changes
  - Protocol/feature changes

Usage:
    python3 ci/release_diff.py v3.21.0 v3.22.0
    python3 ci/release_diff.py HEAD~10 HEAD
    python3 ci/release_diff.py v3.22.0 HEAD --json
"""

import re
import subprocess
import sys
import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
LIB_ROOT = SCRIPT_DIR.parent

RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
CYAN = '\033[96m'
BOLD = '\033[1m'
RESET = '\033[0m'


def git_diff_files(ref_from, ref_to):
    """Get list of changed files between two refs."""
    result = subprocess.run(
        ['git', 'diff', '--name-status', ref_from, ref_to],
        capture_output=True, text=True, cwd=str(LIB_ROOT)
    )
    files = {'A': [], 'M': [], 'D': [], 'R': []}
    for line in result.stdout.strip().split('\n'):
        if not line.strip():
            continue
        parts = line.split('\t')
        status = parts[0][0]  # A/M/D/R
        fname = parts[-1]
        files.setdefault(status, []).append(fname)
    return files


def git_diff_content(ref_from, ref_to, path_filter):
    """Get diff content for a specific path pattern."""
    result = subprocess.run(
        ['git', 'diff', ref_from, ref_to, '--', path_filter],
        capture_output=True, text=True, cwd=str(LIB_ROOT)
    )
    return result.stdout


def extract_abi_changes(ref_from, ref_to):
    """Find added/removed ufsecp_* function declarations."""
    diff = git_diff_content(ref_from, ref_to, 'include/ufsecp/ufsecp.h')
    fn_re = re.compile(r'(ufsecp_\w+)\s*\(')
    added, removed = [], []
    for line in diff.split('\n'):
        if line.startswith('+') and not line.startswith('+++'):
            for m in fn_re.finditer(line):
                added.append(m.group(1))
        elif line.startswith('-') and not line.startswith('---'):
            for m in fn_re.finditer(line):
                removed.append(m.group(1))
    return sorted(set(added) - set(removed)), sorted(set(removed) - set(added))


def categorize_changes(changed_files):
    """Categorize changed files by area."""
    categories = {
        'abi': [], 'ct_layer': [], 'protocol': [], 'gpu': [],
        'tests': [], 'docs': [], 'ci': [], 'build': [], 'other': [],
    }
    for status, files in changed_files.items():
        for f in files:
            entry = f"{status}\t{f}"
            if 'include/ufsecp/' in f:
                categories['abi'].append(entry)
            elif 'ct_' in f or '/ct/' in f:
                categories['ct_layer'].append(entry)
            elif any(p in f for p in ['musig', 'frost', 'adaptor', 'silent_pay', 'ecies', 'dleq']):
                categories['protocol'].append(entry)
            elif any(p in f for p in ['cuda/', 'opencl/', 'metal/']):
                categories['gpu'].append(entry)
            elif any(p in f for p in ['test', 'audit/', 'fuzz']):
                categories['tests'].append(entry)
            elif f.startswith('docs/') or f.endswith('.md'):
                categories['docs'].append(entry)
            elif '.github/' in f:
                categories['ci'].append(entry)
            elif 'CMakeLists' in f or f.endswith('.cmake'):
                categories['build'].append(entry)
            else:
                categories['other'].append(entry)
    return categories


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <from-ref> <to-ref> [--json]")
        sys.exit(1)

    ref_from = sys.argv[1]
    ref_to = sys.argv[2]
    json_mode = '--json' in sys.argv

    changed = git_diff_files(ref_from, ref_to)
    abi_added, abi_removed = extract_abi_changes(ref_from, ref_to)
    categories = categorize_changes(changed)

    total_files = sum(len(v) for v in changed.values())

    if json_mode:
        report = {
            'from': ref_from,
            'to': ref_to,
            'total_files_changed': total_files,
            'abi_added': abi_added,
            'abi_removed': abi_removed,
            'categories': {k: v for k, v in categories.items() if v},
        }
        print(json.dumps(report, indent=2))
        return

    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}  Release Diff: {ref_from} -> {ref_to}{RESET}")
    print(f"{BOLD}{'='*60}{RESET}\n")

    print(f"  Total files changed: {total_files}")
    print(f"  Added: {len(changed.get('A', []))}, Modified: {len(changed.get('M', []))}, "
          f"Deleted: {len(changed.get('D', []))}\n")

    # ABI changes
    if abi_added or abi_removed:
        print(f"{BOLD}ABI Surface Changes{RESET}")
        for fn in abi_added:
            print(f"  {GREEN}+ {fn}{RESET}")
        for fn in abi_removed:
            print(f"  {RED}- {fn}{RESET}")
        print()

    # By category
    for cat, entries in categories.items():
        if not entries:
            continue
        print(f"{BOLD}{cat.upper().replace('_', ' ')} ({len(entries)}){RESET}")
        for e in entries[:15]:
            print(f"  {e}")
        if len(entries) > 15:
            print(f"  ... +{len(entries) - 15} more")
        print()

    # Checklist
    print(f"{BOLD}Release Checklist{RESET}")
    checks = [
        ('ABI functions added/removed', bool(abi_added or abi_removed)),
        ('CT layer files changed', bool(categories['ct_layer'])),
        ('Protocol files changed', bool(categories['protocol'])),
        ('GPU backends changed', bool(categories['gpu'])),
        ('Test files changed', bool(categories['tests'])),
        ('CI workflows changed', bool(categories['ci'])),
        ('Build system changed', bool(categories['build'])),
    ]
    for desc, triggered in checks:
        marker = f"{YELLOW}[!]{RESET}" if triggered else f"{GREEN}[ ]{RESET}"
        print(f"  {marker} {desc}")

    if abi_added or abi_removed:
        print(f"\n  {YELLOW}ACTION: Update binding READMEs for ABI changes{RESET}")
    if categories['ct_layer']:
        print(f"  {YELLOW}ACTION: Verify CT security patterns preserved{RESET}")
    if categories['protocol']:
        print(f"  {YELLOW}ACTION: Review protocol adversarial test coverage{RESET}")
    print()


if __name__ == '__main__':
    main()
