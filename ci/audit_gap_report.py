#!/usr/bin/env python3
"""
audit_gap_report.py -- Execute the self-audit failure-class matrix

Reads docs/SELF_AUDIT_FAILURE_MATRIX.md and turns it into a machine-checkable
gap report.

Current checks:
  1. Every failure-class row has a non-empty deterministic audit surface field
  2. Partial / deferred rows have a non-empty residual-risk note
  3. Referenced file-like evidence paths resolve inside the repository
  4. Emits a summary of covered / partial / deferred classes

Exit code:
  0  all rows are structurally valid
  1  one or more structural problems were found

Usage:
  python3 ci/audit_gap_report.py
  python3 ci/audit_gap_report.py --json
  python3 ci/audit_gap_report.py --strict
  python3 ci/audit_gap_report.py --json -o audit_gap_report.json
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
LIB_ROOT = SCRIPT_DIR.parent
MATRIX_PATH = LIB_ROOT / 'docs' / 'SELF_AUDIT_FAILURE_MATRIX.md'

RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
CYAN = '\033[96m'
BOLD = '\033[1m'
RESET = '\033[0m'


ROW_STATUS_MAP = {
    'covered': 'covered',
    'partial': 'partial',
    'intentionally deferred': 'deferred',
    'deferred': 'deferred',
}


@dataclass
class MatrixRow:
    failure_class: str
    primary_risk: str
    deterministic_surfaces: str
    current_status: str
    residual_risk: str

    @property
    def normalized_status(self) -> str:
        return ROW_STATUS_MAP.get(self.current_status.strip().lower(), 'unknown')


def _clean_cell(text: str) -> str:
    return text.strip().strip('`').strip()


def _split_table_line(line: str) -> list[str]:
    parts = [part.strip() for part in line.strip().strip('|').split('|')]
    return parts


def load_matrix_rows() -> list[MatrixRow]:
    if not MATRIX_PATH.exists():
        raise FileNotFoundError(f'missing matrix: {MATRIX_PATH}')

    rows: list[MatrixRow] = []
    lines = MATRIX_PATH.read_text(encoding='utf-8').splitlines()
    in_matrix = False

    for line in lines:
        if line.startswith('| Failure Class |'):
            in_matrix = True
            continue
        if in_matrix and line.startswith('|---'):
            continue
        if in_matrix:
            if not line.startswith('|'):
                break
            parts = _split_table_line(line)
            if len(parts) != 5:
                continue
            rows.append(
                MatrixRow(
                    failure_class=_clean_cell(parts[0]),
                    primary_risk=_clean_cell(parts[1]),
                    deterministic_surfaces=_clean_cell(parts[2]),
                    current_status=_clean_cell(parts[3]),
                    residual_risk=_clean_cell(parts[4]),
                )
            )

    return rows


def _candidate_surface_paths(token: str) -> list[Path]:
    normalized = token.strip().strip('`').strip()
    if not normalized:
        return []
    candidates = [LIB_ROOT / normalized]
    if '/' not in normalized:
        candidates.append(LIB_ROOT / 'docs' / normalized)
        candidates.append(LIB_ROOT / 'audit' / normalized)
        candidates.append(LIB_ROOT / 'scripts' / normalized)
    return candidates


def _normalize_surface_token(token: str) -> str:
    normalized = token.strip().strip('`').strip()
    if not normalized:
        return ''
    normalized = normalized.split()[0].rstrip(',:;)')
    return normalized


def _looks_like_repo_path(token: str) -> bool:
    normalized = _normalize_surface_token(token)
    if not normalized:
        return False
    if '/' not in normalized and '.' not in normalized:
        return False
    if re.search(r'\.[A-Za-z0-9_+-]+$', normalized):
        return True
    if normalized.startswith(('docs/', 'audit/', 'ci/', 'tools/', 'include/', 'cpu/', 'gpu/', 'opencl/', 'metal/')):
        return True
    return False


def _extract_surface_tokens(text: str) -> list[str]:
    tokens = []
    for token in re.findall(r'`([^`]+)`', text):
        normalized = _normalize_surface_token(token)
        if _looks_like_repo_path(normalized):
            tokens.append(normalized)
    if tokens:
        return sorted(dict.fromkeys(tokens))
    fallback = []
    for chunk in [part.strip() for part in text.split(',')]:
        normalized = _normalize_surface_token(chunk)
        if not normalized:
            continue
        if _looks_like_repo_path(normalized):
            fallback.append(normalized)
    return sorted(dict.fromkeys(fallback))


def _resolve_surface_tokens(text: str) -> tuple[list[str], list[str]]:
    resolved = []
    missing = []
    for token in _extract_surface_tokens(text):
        found = False
        for candidate in _candidate_surface_paths(token):
            if candidate.exists():
                try:
                    resolved.append(str(candidate.relative_to(LIB_ROOT)).replace('\\', '/'))
                except ValueError:
                    resolved.append(str(candidate))
                found = True
                break
        if not found:
            missing.append(token)
    return sorted(dict.fromkeys(resolved)), sorted(dict.fromkeys(missing))


def build_report(strict: bool = False) -> tuple[dict, bool]:
    rows = load_matrix_rows()

    issues = []
    row_reports = []
    counts = {'covered': 0, 'partial': 0, 'deferred': 0, 'unknown': 0}
    strict_fail = False

    for row in rows:
        status = row.normalized_status
        counts[status] = counts.get(status, 0) + 1

        resolved_surfaces, missing_surfaces = _resolve_surface_tokens(row.deterministic_surfaces)
        row_issues = []

        if not row.deterministic_surfaces:
            row_issues.append('missing deterministic audit surfaces')
        if not resolved_surfaces and not missing_surfaces:
            row_issues.append('no file-like deterministic surfaces were detected')
        if missing_surfaces:
            row_issues.append(f'unresolved surfaces: {", ".join(missing_surfaces)}')
        if status in ('partial', 'deferred') and not row.residual_risk:
            row_issues.append('partial/deferred row missing residual-risk note')
        if status == 'unknown':
            row_issues.append(f'unknown status: {row.current_status}')
        if strict and status in ('partial', 'unknown'):
            strict_fail = True

        if row_issues:
            issues.append({'failure_class': row.failure_class, 'issues': row_issues})

        row_reports.append({
            'failure_class': row.failure_class,
            'status': status,
            'resolved_surfaces': resolved_surfaces,
            'missing_surfaces': missing_surfaces,
            'residual_risk': row.residual_risk,
            'issues': row_issues,
        })

    summary = {
        'matrix_path': str(MATRIX_PATH.relative_to(LIB_ROOT)),
        'total_rows': len(rows),
        'counts': counts,
        'issues': issues,
        'rows': row_reports,
        'strict_mode': strict,
    }
    has_fail = bool(issues) or strict_fail
    return summary, has_fail


def print_report(report: dict, has_fail: bool) -> None:
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}  Self-Audit Gap Report{RESET}")
    print(f"{BOLD}{'='*60}{RESET}\n")

    counts = report['counts']
    print(f"{GREEN}Covered{RESET}: {counts.get('covered', 0)}")
    print(f"{YELLOW}Partial{RESET}: {counts.get('partial', 0)}")
    print(f"{CYAN}Deferred{RESET}: {counts.get('deferred', 0)}")
    if counts.get('unknown', 0):
        print(f"{RED}Unknown{RESET}: {counts.get('unknown', 0)}")
    print()

    if report['issues']:
        print(f"{BOLD}Structural Issues{RESET}")
        for issue in report['issues']:
            print(f"  {RED}[FAIL]{RESET} {issue['failure_class']}")
            for item in issue['issues']:
                print(f"    - {item}")
        print()

    print(f"{BOLD}Risk Posture{RESET}")
    for row in report['rows']:
        status = row['status']
        if status == 'covered':
            icon = f"{GREEN}[OK]{RESET}"
        elif status == 'partial':
            icon = f"{YELLOW}[PARTIAL]{RESET}"
        elif status == 'deferred':
            icon = f"{CYAN}[DEFERRED]{RESET}"
        else:
            icon = f"{RED}[UNKNOWN]{RESET}"
        print(f"  {icon} {row['failure_class']}")
        if row['missing_surfaces']:
            print(f"    unresolved: {', '.join(row['missing_surfaces'])}")
        if row['residual_risk']:
            print(f"    residual: {row['residual_risk']}")
    print()

    print(f"{BOLD}{'='*60}{RESET}")
    if has_fail:
        print(f"{RED}{BOLD}  GAP REPORT: FAIL{RESET}")
    else:
        print(f"{GREEN}{BOLD}  GAP REPORT: PASS{RESET}")
    print(f"{BOLD}{'='*60}{RESET}\n")


def main() -> int:
    args = sys.argv[1:]
    json_mode = '--json' in args
    strict = '--strict' in args
    out_file = None
    if '-o' in args:
        idx = args.index('-o')
        if idx + 1 < len(args):
            out_file = args[idx + 1]

    report, has_fail = build_report(strict=strict)
    if json_mode:
        payload = json.dumps(report, indent=2)
        if out_file:
            Path(out_file).write_text(payload, encoding='utf-8')
        else:
            print(payload)
    else:
        print_report(report, has_fail)
    return 1 if has_fail else 0


if __name__ == '__main__':
    sys.exit(main())