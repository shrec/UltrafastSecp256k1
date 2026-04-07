#!/usr/bin/env python3
"""
audit_gate.py  --  Automated audit gate for UltrafastSecp256k1

Implements the 10 audit principles defined in docs/AUDIT_MANIFEST.md.
Every principle maps to a check that produces FAIL/WARN/INFO findings.

Exit code:
    0  —  all checks passed (may have WARN/INFO)
    1  —  at least one FAIL

Usage:
    python3 scripts/audit_gate.py                          # full gate
    python3 scripts/audit_gate.py --abi-completeness       # single check
    python3 scripts/audit_gate.py --json                   # JSON output
    python3 scripts/audit_gate.py --json -o report.json    # write JSON to file
"""

import json
import os
import re
import sqlite3
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from audit_gap_report import build_report as build_audit_gap_report
from check_secret_path_changes import build_report as build_secret_path_report
from generate_abi_negative_tests import build_manifest as build_abi_negative_manifest

SCRIPT_DIR = Path(__file__).resolve().parent
LIB_ROOT = SCRIPT_DIR.parent
DB_PATH = LIB_ROOT / ".project_graph.db"

RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
CYAN = '\033[96m'
BOLD = '\033[1m'
RESET = '\033[0m'


def get_conn():
    if not DB_PATH.exists():
        print(f"{RED}ERROR: Graph DB not found at {DB_PATH}{RESET}", file=sys.stderr)
        print(f"Run: python3 scripts/build_project_graph.py --rebuild", file=sys.stderr)
        sys.exit(1)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def _env_int(name, default):
    value = os.environ.get(name)
    if not value:
        return default
    try:
        return max(1, int(value))
    except ValueError:
        return default


def _env_float(name, default):
    value = os.environ.get(name)
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def run_script_json(script_name, extra_args=None, timeout=300):
    extra_args = extra_args or []
    script_path = SCRIPT_DIR / script_name
    if not script_path.exists():
        return 127, None, '', f'missing script: {script_path.name}'

    try:
        with tempfile.TemporaryDirectory(prefix='ufsecp_audit_gate_') as tmpdir:
            report_path = Path(tmpdir) / f'{script_path.stem}_report.json'
            cmd = [sys.executable, str(script_path), *extra_args, '-o', str(report_path)]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(LIB_ROOT),
                timeout=timeout,
            )
            stdout = (result.stdout or '')[-2000:]
            stderr = (result.stderr or '')[-2000:]
            if not report_path.exists():
                details = stderr or stdout or 'script did not emit a JSON report'
                return result.returncode, None, stdout, details
            try:
                report = json.loads(report_path.read_text(errors='replace'))
            except Exception as exc:
                return result.returncode, None, stdout, f'failed to parse {report_path.name}: {exc}'
    except subprocess.TimeoutExpired:
        return 124, None, '', f'timed out after {timeout}s'

    return result.returncode, report, stdout, stderr


def scan_header_functions():
    """Extract all ufsecp_* from public headers."""
    headers = [
        LIB_ROOT / 'include' / 'ufsecp' / 'ufsecp.h',
        LIB_ROOT / 'include' / 'ufsecp' / 'ufsecp_gpu.h',
        LIB_ROOT / 'include' / 'ufsecp' / 'ufsecp_version.h',
    ]
    api_re = re.compile(r'UFSECP_API\s+.*?(ufsecp_\w+)\s*\(')
    fns = set()
    for h in headers:
        if not h.exists():
            continue
        with open(h, 'r', errors='replace') as f:
            for line in f:
                m = api_re.search(line)
                if m:
                    fns.add(m.group(1))
    return fns


def scan_ledger_functions():
    """Extract function names from FEATURE_ASSURANCE_LEDGER.md."""
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


# ---------------------------------------------------------------------------
# P1 — ABI Completeness
# ---------------------------------------------------------------------------
def check_abi_completeness(conn):
    findings = []

    header_fns = scan_header_functions()
    graph_fns = {r['name'] for r in conn.execute("SELECT name FROM c_abi_functions").fetchall()}
    ledger_fns = scan_ledger_functions()

    # Header vs graph
    missing_graph = sorted(header_fns - graph_fns)
    extra_graph = sorted(graph_fns - header_fns)
    if missing_graph:
        findings.append(('FAIL', f'{len(missing_graph)} header functions not in graph: {", ".join(missing_graph[:5])}'))
    if extra_graph:
        findings.append(('WARN', f'{len(extra_graph)} graph functions not in headers: {", ".join(extra_graph[:5])}'))

    # Header vs ledger
    missing_ledger = sorted(header_fns - ledger_fns)
    extra_ledger = sorted(ledger_fns - header_fns)
    if missing_ledger:
        findings.append(('WARN', f'{len(missing_ledger)} header functions not in FEATURE_ASSURANCE_LEDGER'))
    if extra_ledger:
        findings.append(('WARN', f'{len(extra_ledger)} ledger functions removed from headers'))

    if not findings:
        findings.append(('PASS', f'All {len(header_fns)} header functions in graph and ledger'))

    return 'P1: ABI Completeness', findings


# ---------------------------------------------------------------------------
# P0 — Failure-Class Matrix
# ---------------------------------------------------------------------------
def check_failure_class_matrix(conn):
    findings = []
    report, has_fail = build_audit_gap_report(strict=True)
    counts = report.get('counts', {})

    findings.append((
        'INFO',
        'Failure-class matrix counts: '
        f"covered={counts.get('covered', 0)}, "
        f"partial={counts.get('partial', 0)}, "
        f"deferred={counts.get('deferred', 0)}, "
        f"unknown={counts.get('unknown', 0)}"
    ))

    for issue in report.get('issues', []):
        findings.append(('FAIL', f"{issue['failure_class']}: {'; '.join(issue['issues'])}"))

    owner_grade_residuals = [
        row for row in report.get('rows', [])
        if row.get('status') in ('partial', 'unknown')
    ]
    if owner_grade_residuals:
        names = ', '.join(row['failure_class'] for row in owner_grade_residuals[:5])
        findings.append(('FAIL', f'{len(owner_grade_residuals)} owner-grade residual failure classes remain: {names}'))
    elif has_fail:
        findings.append(('FAIL', 'Failure-class matrix gate reported blocking issues'))
    else:
        findings.append(('PASS', 'Failure-class matrix is structurally valid and clear of owner-grade residual blockers'))

    return 'P0: Failure-Class Matrix', findings


# ---------------------------------------------------------------------------
# P0 — ABI Hostile-Caller Manifest
# ---------------------------------------------------------------------------
def check_abi_negative_tests(conn):
    findings = []
    report, has_fail = build_abi_negative_manifest()

    findings.append((
        'INFO',
        'ABI negative-test manifest counts: '
        f"exports={report.get('header_count', 0)}, "
        f"blocking={report.get('blocking_function_count', 0)}, "
        f"null={report.get('coverage_counts', {}).get('null_rejection', 0)}, "
        f"zero={report.get('coverage_counts', {}).get('zero_edge', 0)}, "
        f"invalid={report.get('coverage_counts', {}).get('invalid_content', 0)}, "
        f"smoke={report.get('coverage_counts', {}).get('success_smoke', 0)}"
    ))

    blockers = [item for item in report.get('functions', []) if item.get('blocking')]
    if blockers:
        preview = ', '.join(item['function'] for item in blockers[:5])
        findings.append(('FAIL', f"{len(blockers)} exported functions missing hostile-caller quartet coverage: {preview}"))
    elif has_fail:
        findings.append(('FAIL', 'ABI hostile-caller manifest reported blocking coverage gaps'))
    else:
        findings.append(('PASS', 'All exported ABI functions satisfy the hostile-caller coverage quartet'))

    return 'P0: ABI Hostile-Caller Manifest', findings


# ---------------------------------------------------------------------------
# P0 — Invalid-Input Grammar
# ---------------------------------------------------------------------------
def check_invalid_input_grammar(conn):
    findings = []
    timeout = _env_int('UFSECP_AUDIT_INVALID_INPUT_TIMEOUT', 300)
    rc, report, _stdout, details = run_script_json(
        'invalid_input_grammar.py',
        timeout=timeout,
    )

    if report is None:
        findings.append(('FAIL', f'Invalid-input grammar harness did not produce JSON: {details}'))
        return 'P0: Invalid-Input Grammar', findings

    findings.append((
        'INFO',
        'Invalid-input grammar counts: '
        f"cases={report.get('total_cases', 0)}, "
        f"passed={report.get('passed', 0)}, "
        f"failed={report.get('failed', 0)}"
    ))

    if rc != 0 or report.get('overall') != 'PASS':
        preview = '; '.join(item.get('name', 'unknown') for item in report.get('findings', [])[:3])
        findings.append(('FAIL', f'Invalid-input grammar harness failed: {preview or details or "see JSON report"}'))
    else:
        findings.append(('PASS', f"Structured invalid inputs correctly rejected ({report.get('total_cases', 0)} cases)"))

    return 'P0: Invalid-Input Grammar', findings


# ---------------------------------------------------------------------------
# P0 — Stateful Sequence Integrity
# ---------------------------------------------------------------------------
def check_stateful_sequences(conn):
    findings = []
    count = _env_int('UFSECP_AUDIT_STATEFUL_COUNT', 24)
    timeout = _env_int('UFSECP_AUDIT_STATEFUL_TIMEOUT', 600)
    rc, report, _stdout, details = run_script_json(
        'stateful_sequences.py',
        ['--count', str(count)],
        timeout=timeout,
    )

    if report is None:
        findings.append(('FAIL', f'Stateful sequence harness did not produce JSON: {details}'))
        return 'P0: Stateful Sequence Integrity', findings

    findings.append((
        'INFO',
        'Stateful sequence counts: '
        f"passed={report.get('passed', 0)}, "
        f"failed={report.get('failed', 0)}, "
        f"count={count}"
    ))

    if rc != 0 or report.get('overall') != 'PASS':
        preview = '; '.join(
            f"{item.get('sequence', '?')} step={item.get('step', '?')}: {item.get('detail', '')}"
            for item in report.get('findings', [])[:3]
        )
        findings.append(('FAIL', f'Stateful sequence harness failed: {preview or details or "see JSON report"}'))
    else:
        findings.append(('PASS', f"Stateful sequence harness passed ({report.get('passed', 0)} checks)"))

    return 'P0: Stateful Sequence Integrity', findings


# ---------------------------------------------------------------------------
# P0 — Secret-Path Change Gate
# ---------------------------------------------------------------------------
def check_secret_path_gate(conn):
    findings = []
    report, has_fail = build_secret_path_report()

    if not report.get('changed_files'):
        findings.append(('PASS', 'No uncommitted changes to evaluate'))
        return 'P0: Secret-Path Change Gate', findings

    if not report.get('triggered_rules'):
        findings.append(('PASS', 'No changed secret-bearing paths detected'))
        return 'P0: Secret-Path Change Gate', findings

    for rule in report.get('triggered_rules', []):
        findings.append(('INFO', f"{rule['name']}: {', '.join(rule['matches'])}"))
        if rule.get('missing_docs'):
            findings.append(('FAIL', f"{rule['name']} missing paired docs: {', '.join(rule['missing_docs'])}"))

    if not has_fail:
        findings.append(('PASS', 'Secret-bearing changes have paired documentation updates'))

    return 'P0: Secret-Path Change Gate', findings


# ---------------------------------------------------------------------------
# P2 — Test Coverage Mapping
# ---------------------------------------------------------------------------
def check_test_coverage(conn):
    findings = []

    # Coverage gaps view
    gaps = conn.execute("SELECT * FROM v_coverage_gaps").fetchall()
    if gaps:
        names = [r['name'] if 'name' in r.keys() else str(dict(r)) for r in gaps[:5]]
        findings.append(('FAIL', f'{len(gaps)} coverage gaps: {", ".join(names)}'))

    # ABI functions without test mapping
    all_abi = {r['name'] for r in conn.execute("SELECT name FROM c_abi_functions").fetchall()}
    mapped = {r[0] for r in conn.execute(
        "SELECT DISTINCT function_name FROM function_test_map WHERE function_name LIKE 'ufsecp_%'"
    ).fetchall()}
    direct_call_mapped = {r[0] for r in conn.execute(
        """SELECT DISTINCT callee_func FROM call_edges
           WHERE callee_func LIKE 'ufsecp_%'
             AND (
                 caller_file LIKE 'audit/%' OR
                 caller_file LIKE 'cpu/tests/%' OR
                 caller_file LIKE 'tests/%'
             )"""
    ).fetchall()}

    mapped |= direct_call_mapped

    unmapped = sorted(all_abi - mapped)
    if unmapped:
        # GPU functions may not have direct test mappings
        gpu_unmapped = [f for f in unmapped if 'gpu' in f]
        non_gpu_unmapped = [f for f in unmapped if 'gpu' not in f]
        if non_gpu_unmapped:
            findings.append(('FAIL', f'{len(non_gpu_unmapped)} non-GPU ABI functions without test mapping: {", ".join(non_gpu_unmapped[:5])}'))
        if gpu_unmapped:
            findings.append(('WARN', f'{len(gpu_unmapped)} GPU ABI functions without test mapping (GPU tests map differently)'))

    if not findings:
        findings.append(('PASS', f'All {len(all_abi)} ABI functions have test coverage'))

    return 'P2: Test Coverage', findings


# ---------------------------------------------------------------------------
# P2 — Audit Test Quality
# ---------------------------------------------------------------------------
def check_audit_test_quality(conn):
    findings = []
    timeout = _env_int('UFSECP_AUDIT_TEST_QUALITY_TIMEOUT', 300)
    rc, report, _stdout, details = run_script_json(
        'audit_test_quality_scanner.py',
        ['--audit-dir', str(LIB_ROOT / 'audit'), '--min-severity', 'low'],
        timeout=timeout,
    )

    if report is None:
        findings.append(('FAIL', f'Audit test quality scanner did not produce JSON: {details}'))
        return 'P2: Audit Test Quality', findings

    severity_counts = {sev: 0 for sev in ('critical', 'high', 'medium', 'low', 'info')}
    for item in report.get('findings', []):
        severity = item.get('severity', 'info')
        severity_counts[severity] = severity_counts.get(severity, 0) + 1

    findings.append((
        'INFO',
        'Audit test-quality findings: '
        f"files={report.get('total_files', 0)}, "
        f"critical={severity_counts.get('critical', 0)}, "
        f"high={severity_counts.get('high', 0)}, "
        f"medium={severity_counts.get('medium', 0)}, "
        f"low={severity_counts.get('low', 0)}"
    ))

    preview_items = report.get('findings', [])[:3]
    for item in preview_items:
        findings.append((
            'INFO',
            f"{item.get('severity', 'info')} {item.get('file', '?')}:{item.get('line', '?')} {item.get('label', 'finding')}"
        ))

    if rc != 0 or severity_counts.get('critical', 0) or severity_counts.get('high', 0):
        findings.append((
            'FAIL',
            f"Audit test-quality scanner found blocking findings: critical={severity_counts.get('critical', 0)}, high={severity_counts.get('high', 0)}"
        ))
    elif severity_counts.get('medium', 0):
        findings.append(('WARN', f"{severity_counts.get('medium', 0)} medium-severity audit-test-quality findings remain"))
    elif severity_counts.get('low', 0):
        findings.append(('WARN', f"{severity_counts.get('low', 0)} low-severity audit-test-quality findings remain"))
    else:
        findings.append(('PASS', 'Audit test quality is clear of critical/high findings'))

    return 'P2: Audit Test Quality', findings


# ---------------------------------------------------------------------------
# P3 — Security Pattern Preservation
# ---------------------------------------------------------------------------
def check_security_patterns(conn):
    findings = []
    lost_count = 0
    new_count = 0

    expected = conn.execute("""
        SELECT source_file, pattern, COUNT(*) as cnt
        FROM security_patterns GROUP BY source_file, pattern
    """).fetchall()

    patterns_re = {
        'secure_erase': re.compile(r'secure_erase\s*\('),
        'value_barrier': re.compile(r'value_barrier\s*\('),
        'CLASSIFY': re.compile(r'SECP256K1_CLASSIFY\s*\('),
        'DECLASSIFY': re.compile(r'SECP256K1_DECLASSIFY\s*\('),
    }

    for r in expected:
        src_file, pat_name, exp_cnt = r['source_file'], r['pattern'], r['cnt']
        filepath = LIB_ROOT / src_file
        if not filepath.exists():
            findings.append(('FAIL', f'MISSING {src_file} (expected {exp_cnt} {pat_name})'))
            lost_count += 1
            continue
        pat_re = patterns_re.get(pat_name)
        if not pat_re:
            continue
        count = 0
        try:
            with open(filepath, 'r', errors='replace') as f:
                for line in f:
                    stripped = line.strip()
                    if pat_name in ('secure_erase', 'value_barrier'):
                        if stripped.startswith('//') or stripped.startswith('#include'):
                            continue
                    if pat_re.search(line):
                        count += 1
        except Exception:
            findings.append(('FAIL', f'UNREADABLE {src_file}'))
            continue

        if count < exp_cnt:
            findings.append(('FAIL', f'LOST {src_file}: {pat_name} {exp_cnt}->{count} ({exp_cnt - count} removed)'))
            lost_count += 1
        elif count > exp_cnt:
            findings.append(('INFO', f'NEW {src_file}: {pat_name} {exp_cnt}->{count} (+{count - exp_cnt}, rebuild graph)'))
            new_count += 1

    if lost_count == 0 and not any(f[0] == 'FAIL' for f in findings):
        total = sum(r['cnt'] for r in expected)
        findings.insert(0, ('PASS', f'All {total} security patterns preserved'))

    return 'P3: Security Patterns', findings


# ---------------------------------------------------------------------------
# P4 — CT Layer Integrity
# ---------------------------------------------------------------------------
def check_ct_integrity(conn):
    findings = []

    ct_funcs = conn.execute("""
        SELECT abi_function FROM abi_routing WHERE layer='ct'
    """).fetchall()
    ct_names = [r['abi_function'] for r in ct_funcs]

    ct_doc = LIB_ROOT / 'docs' / 'CT_VERIFICATION.md'
    if ct_doc.exists():
        content = ct_doc.read_text(errors='replace')
        missing_docs = [fn for fn in ct_names if fn not in content]
        if missing_docs:
            findings.append(('WARN', f'{len(missing_docs)} CT functions not in CT_VERIFICATION.md: {", ".join(missing_docs[:5])}'))
    else:
        findings.append(('WARN', 'CT_VERIFICATION.md not found'))

    if not findings:
        findings.append(('PASS', f'All {len(ct_names)} CT-routed functions documented'))

    return 'P4: CT Integrity', findings


# ---------------------------------------------------------------------------
# P5 — Narrative Consistency
# ---------------------------------------------------------------------------
STALE_PHRASES = [
    (r'(?i)\bno\s+formal\s+(ct\s+)?verification\b',
     'Claims no formal CT verification'),
    (r'(?i)\btool\s+integration\s+not\s+yet\s+done\b',
     'Claims tool integration not done'),
    (r'(?i)\bno\s+formal\s+verification\s+applied\b',
     'Claims no formal verification applied'),
]

NARRATIVE_FILES = [
    'docs/AUDIT_READINESS_REPORT_v1.md',
    'audit/AUDIT_TEST_PLAN.md',
    'docs/TEST_MATRIX.md',
    'audit/run_full_audit.sh',
    'audit/run_full_audit.ps1',
]

HISTORICAL_MARKER = re.compile(r'(?i)(historical\s+report|superseded\s+by|snapshot\s+from\s+v\d)')


def check_narrative(conn):
    findings = []

    for pat_str, description in STALE_PHRASES:
        pat = re.compile(pat_str)
        for rel_path in NARRATIVE_FILES:
            filepath = LIB_ROOT / rel_path
            if not filepath.exists():
                continue
            try:
                content = filepath.read_text(errors='replace')
            except Exception:
                continue
            if HISTORICAL_MARKER.search(content[:500]):
                continue
            for i, line in enumerate(content.splitlines(), 1):
                if pat.search(line):
                    findings.append(('WARN', f'DRIFT {rel_path}:{i} — {description}'))

    if not findings:
        findings.append(('PASS', 'No stale narrative detected'))

    return 'P5: Narrative Consistency', findings


# ---------------------------------------------------------------------------
# P6 — Graph Freshness
# ---------------------------------------------------------------------------
def check_freshness(conn):
    findings = []

    built_str = conn.execute("SELECT value FROM meta WHERE key='built_at'").fetchone()['value']
    built_dt = datetime.fromisoformat(built_str)

    stale = []
    rows = conn.execute("""
        SELECT path FROM source_files WHERE layer IN ('fast','ct','abi')
    """).fetchall()

    for r in rows:
        filepath = LIB_ROOT / r['path']
        if not filepath.exists():
            stale.append(('DELETED', r['path']))
            continue
        mtime = datetime.fromtimestamp(filepath.stat().st_mtime, tz=timezone.utc)
        if mtime > built_dt:
            stale.append(('MODIFIED', r['path']))

    if stale:
        findings.append(('WARN', f'{len(stale)} files modified/deleted since graph build ({built_str[:19]})'))
        for kind, path in stale[:5]:
            findings.append(('INFO', f'{kind}: {path}'))
    else:
        findings.append(('PASS', f'Graph is fresh (built: {built_str[:19]})'))

    return 'P6: Graph Freshness', findings


# ---------------------------------------------------------------------------
# P7 — GPU Backend Parity
# ---------------------------------------------------------------------------
def check_gpu_parity(conn):
    findings = []

    # Check all GPU ABI functions are in graph
    header_gpu = {fn for fn in scan_header_functions() if 'gpu' in fn}
    graph_gpu = {r['name'] for r in conn.execute(
        "SELECT name FROM c_abi_functions WHERE category='gpu'"
    ).fetchall()}

    missing = sorted(header_gpu - graph_gpu)
    if missing:
        findings.append(('FAIL', f'{len(missing)} GPU header functions not in graph: {", ".join(missing)}'))

    # Scan for undocumented Unsupported returns
    unsupported_dirs = [
        LIB_ROOT / 'gpu' / 'src',
        LIB_ROOT / 'opencl',
        LIB_ROOT / 'metal',
    ]
    unsup_re = re.compile(r'Unsupported')
    todo_re = re.compile(r'TODO\(parity\)|PARITY-EXCEPTION')

    undocumented = []
    for scan_dir in unsupported_dirs:
        if not scan_dir.exists():
            continue
        for root, dirs, files in os.walk(scan_dir):
            for fname in files:
                fpath = Path(root) / fname
                try:
                    lines = fpath.read_text(errors='replace').splitlines()
                except Exception:
                    continue
                for i, line in enumerate(lines):
                    if unsup_re.search(line):
                        # Check preceding 3 lines for TODO or PARITY-EXCEPTION
                        context = '\n'.join(lines[max(0, i-3):i+1])
                        if not todo_re.search(context):
                            rel = str(fpath.relative_to(LIB_ROOT))
                            undocumented.append(f'{rel}:{i+1}')

    if undocumented:
        findings.append(('WARN', f'{len(undocumented)} undocumented Unsupported returns'))
        for loc in undocumented[:5]:
            findings.append(('INFO', f'  {loc}'))

    if not findings:
        findings.append(('PASS', f'GPU parity OK ({len(header_gpu)} GPU ABI functions)'))

    return 'P7: GPU Parity', findings


# ---------------------------------------------------------------------------
# P8 — Test Target Documentation
# ---------------------------------------------------------------------------
def check_test_docs(conn):
    findings = []

    # Actual CTest targets
    actual = set()
    test_re = re.compile(r'add_test\s*\(\s*NAME\s+(\S+)')
    for cmake_file in LIB_ROOT.rglob('CMakeLists.txt'):
        rel = str(cmake_file.relative_to(LIB_ROOT))
        if rel.startswith('build') or '_build' in rel:
            continue
        try:
            for line in open(cmake_file, 'r', errors='replace'):
                m = test_re.search(line)
                if m:
                    actual.add(m.group(1))
        except Exception:
            continue

    # Documented in TEST_MATRIX.md
    matrix = LIB_ROOT / 'docs' / 'TEST_MATRIX.md'
    documented = set()
    if matrix.exists():
        content = matrix.read_text(errors='replace')
        for target in actual:
            if target in content:
                documented.add(target)

    missing = sorted(actual - documented)
    if missing:
        findings.append(('WARN', f'{len(missing)} CTest targets not in TEST_MATRIX.md: {", ".join(missing[:5])}'))

    if not findings:
        findings.append(('PASS', f'All {len(actual)} CTest targets documented'))

    return 'P8: Test Documentation', findings


# ---------------------------------------------------------------------------
# P9 — ABI Routing Consistency
# ---------------------------------------------------------------------------
def check_routing(conn):
    findings = []

    ct_funcs = conn.execute("""
        SELECT abi_function, internal_call FROM abi_routing WHERE layer='ct'
    """).fetchall()

    for r in ct_funcs:
        call = r['internal_call'] or ''
        if call and not any(kw in call.lower() for kw in ['ct_', 'constant_time', 'ct::']):
            findings.append(('INFO', f'{r["abi_function"]} routed CT but calls {call}'))

    if not any(f[0] in ('FAIL', 'WARN') for f in findings):
        findings.insert(0, ('PASS', f'ABI routing consistent ({len(ct_funcs)} CT functions)'))

    return 'P9: Routing Consistency', findings


# ---------------------------------------------------------------------------
# P10 — Doc-Code Pairing
# ---------------------------------------------------------------------------
DOC_PAIRS = {
    'include/ufsecp/ufsecp.h': ['docs/API_REFERENCE.md', 'docs/USER_GUIDE.md'],
    'include/ufsecp/ufsecp_impl.cpp': ['docs/API_REFERENCE.md'],
    'CMakeLists.txt': ['docs/BUILDING.md'],
    'cpu/src/musig2.cpp': ['docs/API_REFERENCE.md'],
    'cpu/src/frost.cpp': ['docs/API_REFERENCE.md'],
    'cpu/src/ct_sign.cpp': ['docs/CT_VERIFICATION.md', 'docs/SECURITY_CLAIMS.md'],
    'cpu/src/ct_field.cpp': ['docs/CT_VERIFICATION.md'],
    'cpu/src/ct_scalar.cpp': ['docs/CT_VERIFICATION.md'],
    'cpu/src/ct_point.cpp': ['docs/CT_VERIFICATION.md'],
}


def check_doc_pairing(conn):
    findings = []

    # Get changed files
    try:
        result = subprocess.run(
            ['git', 'diff', '--name-only', 'HEAD'],
            capture_output=True, text=True, cwd=str(LIB_ROOT)
        )
        changed = {f.strip() for f in result.stdout.strip().split('\n') if f.strip()}
        result2 = subprocess.run(
            ['git', 'diff', '--cached', '--name-only'],
            capture_output=True, text=True, cwd=str(LIB_ROOT)
        )
        changed |= {f.strip() for f in result2.stdout.strip().split('\n') if f.strip()}
    except Exception:
        changed = set()

    if not changed:
        findings.append(('PASS', 'No uncommitted changes'))
        return 'P10: Doc-Code Pairing', findings

    for code_file, expected_docs in DOC_PAIRS.items():
        if any(code_file in cf for cf in changed):
            for doc in expected_docs:
                if not any(doc in cf for cf in changed):
                    findings.append(('WARN', f'{code_file} changed but {doc} not updated'))

    if not findings:
        findings.append(('PASS', f'{len(changed)} changed files have matching docs'))

    return 'P10: Doc-Code Pairing', findings


# ---------------------------------------------------------------------------
# P2 — Mutation Kill Rate (explicit heavy lane)
# ---------------------------------------------------------------------------
def check_mutation_kill_rate(conn):
    findings = []
    timeout = _env_int('UFSECP_AUDIT_MUTATION_TIMEOUT', 3600)
    threshold = _env_float('UFSECP_AUDIT_MUTATION_THRESHOLD', 75.0)
    build_dir = os.environ.get('UFSECP_AUDIT_MUTATION_BUILD_DIR', 'build_opencl')
    rc, report, _stdout, details = run_script_json(
        'mutation_kill_rate.py',
        ['--build-dir', build_dir, '--ctest-mode', '--threshold', str(threshold)],
        timeout=timeout,
    )

    if report is None:
        findings.append(('FAIL', f'Mutation kill-rate runner did not produce JSON: {details}'))
        return 'P2: Mutation Kill Rate', findings

    findings.append((
        'INFO',
        'Mutation kill-rate summary: '
        f"total={report.get('total', 0)}, "
        f"killed={report.get('killed', 0)}, "
        f"survived={report.get('survived', 0)}, "
        f"kill_rate={report.get('kill_rate_pct', 0.0)}%"
    ))

    if rc != 0 or not report.get('passed', False):
        findings.append(('FAIL', f"Mutation kill rate below threshold ({report.get('kill_rate_pct', 0.0)}% < {threshold:.1f}%)"))
    else:
        findings.append(('PASS', f"Mutation kill rate meets threshold ({report.get('kill_rate_pct', 0.0)}%)"))

    return 'P2: Mutation Kill Rate', findings


# ---------------------------------------------------------------------------
# P3 — Crash Risk Analysis (source graph)
# ---------------------------------------------------------------------------
SOURCE_GRAPH_DB = LIB_ROOT / "tools" / "source_graph_kit" / "source_graph.db"


def check_crash_risks(_conn):
    findings = []

    if not SOURCE_GRAPH_DB.exists():
        findings.append(('WARN', 'Source graph DB not found — skipping crash risk analysis'))
        return 'P3: Crash Risks', findings

    sg_conn = sqlite3.connect(str(SOURCE_GRAPH_DB))
    sg_conn.row_factory = sqlite3.Row
    try:
        rows = sg_conn.execute("""
            SELECT risk_type, COUNT(*) as cnt FROM crash_risks GROUP BY risk_type
        """).fetchall()
        total = sum(r['cnt'] for r in rows)

        # Check crash risks in CT-sensitive files
        ct_crash = sg_conn.execute("""
            SELECT COUNT(*) as cnt FROM crash_risks cr
            JOIN symbol_metadata sm ON cr.file = sm.file_path
            WHERE sm.ct_sensitive = 1
        """).fetchone()
        ct_crash_count = ct_crash['cnt'] if ct_crash else 0
    finally:
        sg_conn.close()

    breakdown = ", ".join(f'{r["risk_type"]}={r["cnt"]}' for r in rows)
    findings.append(('INFO', f'Total crash risks: {total} ({breakdown})'))

    if ct_crash_count > 0:
        findings.append(('WARN', f'{ct_crash_count} crash risks in CT-sensitive functions'))
    else:
        findings.append(('PASS', f'No crash risks in CT-sensitive code ({total} total, none in CT paths)'))

    return 'P3: Crash Risks', findings


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
CHECK_MAP = {
    '--failure-matrix': check_failure_class_matrix,
    '--abi-negative-tests': check_abi_negative_tests,
    '--invalid-inputs': check_invalid_input_grammar,
    '--stateful-sequences': check_stateful_sequences,
    '--secret-paths': check_secret_path_gate,
    '--abi-completeness': check_abi_completeness,
    '--test-coverage': check_test_coverage,
    '--audit-test-quality': check_audit_test_quality,
    '--security-patterns': check_security_patterns,
    '--ct-integrity': check_ct_integrity,
    '--narrative': check_narrative,
    '--freshness': check_freshness,
    '--gpu-parity': check_gpu_parity,
    '--test-docs': check_test_docs,
    '--routing': check_routing,
    '--doc-pairing': check_doc_pairing,
    '--mutation-kill': check_mutation_kill_rate,
    '--crash-risks': check_crash_risks,
}

ALL_CHECKS = [
    check_failure_class_matrix,
    check_abi_negative_tests,
    check_invalid_input_grammar,
    check_stateful_sequences,
    check_secret_path_gate,
    check_abi_completeness,
    check_test_coverage,
    check_audit_test_quality,
    check_security_patterns,
    check_ct_integrity,
    check_narrative,
    check_freshness,
    check_gpu_parity,
    check_test_docs,
    check_routing,
    check_doc_pairing,
    check_crash_risks,
]


def main():
    args = sys.argv[1:]
    json_mode = '--json' in args
    out_file = None
    if '-o' in args:
        idx = args.index('-o')
        if idx + 1 < len(args):
            out_file = args[idx + 1]

    # Select checks
    selected = []
    for arg in args:
        if arg in CHECK_MAP:
            selected.append(CHECK_MAP[arg])
    if not selected:
        selected = ALL_CHECKS

    conn = get_conn()
    results = []
    has_fail = False

    for check_fn in selected:
        title, findings = check_fn(conn)
        results.append({'check': title, 'findings': findings})
        if any(f[0] == 'FAIL' for f in findings):
            has_fail = True

    conn.close()

    if json_mode:
        report = {
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'status': 'FAIL' if has_fail else 'PASS',
            'checks': [
                {
                    'name': r['check'],
                    'status': 'FAIL' if any(f[0] == 'FAIL' for f in r['findings']) else 'PASS',
                    'findings': [{'severity': f[0], 'message': f[1]} for f in r['findings']],
                }
                for r in results
            ],
        }
        output = json.dumps(report, indent=2)
        if out_file:
            Path(out_file).write_text(output)
            print(f"Report written to {out_file}", file=sys.stderr)
        else:
            print(output)
    else:
        print(f"\n{BOLD}{'='*60}{RESET}")
        print(f"{BOLD}  UltrafastSecp256k1 Audit Gate{RESET}")
        print(f"{BOLD}{'='*60}{RESET}\n")

        for r in results:
            title = r['check']
            findings = r['findings']
            check_fail = any(f[0] == 'FAIL' for f in findings)
            check_warn = any(f[0] == 'WARN' for f in findings)

            if check_fail:
                icon = f"{RED}[FAIL]{RESET}"
            elif check_warn:
                icon = f"{YELLOW}[WARN]{RESET}"
            else:
                icon = f"{GREEN}[PASS]{RESET}"

            print(f"{BOLD}{title}{RESET}  {icon}")
            for sev, msg in findings:
                if sev == 'FAIL':
                    print(f"  {RED}{sev}{RESET}  {msg}")
                elif sev == 'WARN':
                    print(f"  {YELLOW}{sev}{RESET}  {msg}")
                elif sev == 'INFO':
                    print(f"  {CYAN}{sev}{RESET}  {msg}")
                elif sev == 'PASS':
                    print(f"  {GREEN}{sev}{RESET}  {msg}")
            print()

        print(f"{BOLD}{'='*60}{RESET}")
        if has_fail:
            fail_count = sum(1 for r in results for f in r['findings'] if f[0] == 'FAIL')
            print(f"{RED}{BOLD}  AUDIT GATE: FAILED ({fail_count} blocking findings){RESET}")
        else:
            warn_count = sum(1 for r in results for f in r['findings'] if f[0] == 'WARN')
            if warn_count:
                print(f"{YELLOW}{BOLD}  AUDIT GATE: PASSED with {warn_count} warnings{RESET}")
            else:
                print(f"{GREEN}{BOLD}  AUDIT GATE: PASSED{RESET}")
        print(f"{BOLD}{'='*60}{RESET}\n")

    return 1 if has_fail else 0


if __name__ == '__main__':
    sys.exit(main())
