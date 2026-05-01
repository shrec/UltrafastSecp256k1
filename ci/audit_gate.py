#!/usr/bin/env python3
"""
audit_gate.py  --  Automated audit gate for UltrafastSecp256k1

Implements the audit principles (P0–P11 + P21 + G-10..G-12) defined in docs/AUDIT_MANIFEST.md.
Every principle maps to a check that produces FAIL/WARN/INFO findings.
P12–P18 (Security Autonomy) are run via standalone scripts; see the manifest.

Exit code:
    0  —  all checks passed (may have WARN/INFO)
    1  —  at least one FAIL

Usage:
    python3 ci/audit_gate.py                          # full gate
    python3 ci/audit_gate.py --abi-completeness       # single check
    python3 ci/audit_gate.py --json                   # JSON output
    python3 ci/audit_gate.py --json -o report.json    # write JSON to file
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

try:
    from report_provenance import collect_provenance
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from report_provenance import collect_provenance

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
        print(f"Run: python3 ci/build_project_graph.py --rebuild", file=sys.stderr)
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
    # CAAS-21 fix: read whole file and use re.DOTALL so the pattern crosses
    # newlines. The old line-by-line approach silently missed multi-line
    # declarations like `UFSECP_API\nufsecp_error_t ufsecp_foo(...)`.
    api_re = re.compile(r'UFSECP_API\s+.*?(ufsecp_\w+)\s*\(', re.DOTALL)
    fns = set()
    for h in headers:
        if not h.exists():
            continue
        text = h.read_text(errors='replace')
        for m in api_re.finditer(text):
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
        # secp256k1_* shim functions are tested via standalone shim tests (not
        # routed through the unified runner), so they won't appear in call_edges.
        # They are covered by check_libsecp_shim_parity.py + dedicated shim tests.
        shim_unmapped = [f for f in unmapped if f.startswith('secp256k1_') and 'gpu' not in f]
        non_gpu_unmapped = [f for f in unmapped
                            if 'gpu' not in f and not f.startswith('secp256k1_')]
        if non_gpu_unmapped:
            findings.append(('FAIL', f'{len(non_gpu_unmapped)} non-GPU ABI functions without test mapping: {", ".join(non_gpu_unmapped[:5])}'))
        if gpu_unmapped:
            findings.append(('WARN', f'{len(gpu_unmapped)} GPU ABI functions without test mapping (GPU tests map differently)'))
        if shim_unmapped:
            findings.append(('WARN', f'{len(shim_unmapped)} secp256k1_* shim ABI functions covered by standalone shim tests (not in unified runner call graph)'))

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

    if rc != 0:
        findings.append((
            'WARN',
            f"Audit test-quality scanner returned exit code {rc}. Results may be incomplete."
        ))
    if severity_counts.get('critical', 0) or severity_counts.get('high', 0):
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

HISTORICAL_MARKER = re.compile(r'^(?:#+\s*)?\[?(?:historical\s+report|superseded\s+by|snapshot\s+from\s+v\d+)\]?', re.IGNORECASE | re.MULTILINE)


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
            if HISTORICAL_MARKER.search(content):
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

    _built_row = conn.execute("SELECT value FROM meta WHERE key='built_at'").fetchone()
    built_str = _built_row['value'] if _built_row is not None else "unknown"
    if built_str == "unknown":
        findings.append(('WARN', "Graph meta key 'built_at' not found — graph may need rebuild"))
        return 'P6: Graph Freshness', findings
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
# P2 — Mutation Evidence Freshness
# ---------------------------------------------------------------------------
_DEFAULT_MUTATION_REPORT = "mutation_kill_report.json"
_DEFAULT_MUTATION_STALENESS_DAYS = 14


def check_mutation_freshness(_conn):
    """Verify that a mutation-kill report exists and is not stale."""
    findings = []
    report_name = os.environ.get('UFSECP_MUTATION_REPORT', _DEFAULT_MUTATION_REPORT)
    staleness_days = _env_int('UFSECP_MUTATION_STALENESS_DAYS', _DEFAULT_MUTATION_STALENESS_DAYS)
    report_path = LIB_ROOT / report_name

    if not report_path.exists():
        findings.append(('WARN', f'Mutation kill report not found: {report_name} — run: '
                         'python3 ci/mutation_kill_rate.py --ctest-mode --json -o mutation_kill_report.json'))
        return 'P2: Mutation Evidence Freshness', findings

    try:
        with open(report_path, 'r') as f:
            report = json.load(f)
    except Exception as e:
        findings.append(('WARN', f'Could not parse mutation report: {e}'))
        return 'P2: Mutation Evidence Freshness', findings

    ts_str = report.get('timestamp') or report.get('generated_at', '')
    if ts_str:
        try:
            ts = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
            age = datetime.now(timezone.utc) - ts
            age_days = age.total_seconds() / 86400
            findings.append(('INFO', f'Mutation report age: {age_days:.1f} days (threshold: {staleness_days})'))
            if age_days > staleness_days:
                findings.append(('WARN', f'Mutation report is stale ({age_days:.1f} > {staleness_days} days) — rerun recommended'))
            else:
                kill_rate = report.get('kill_rate_pct', report.get('kill_rate', 'N/A'))
                findings.append(('PASS', f'Mutation evidence is current (age {age_days:.1f}d, rate {kill_rate}%)'))
        except (ValueError, TypeError):
            findings.append(('WARN', f'Could not parse mutation report timestamp: {ts_str!r}'))
    else:
        # Fall back to file mtime
        mtime = os.path.getmtime(report_path)
        age_days = (datetime.now(timezone.utc).timestamp() - mtime) / 86400
        findings.append(('INFO', f'Mutation report mtime age: {age_days:.1f} days (no timestamp field)'))
        if age_days > staleness_days:
            findings.append(('WARN', f'Mutation report is stale by mtime ({age_days:.1f} > {staleness_days} days)'))
        else:
            findings.append(('PASS', f'Mutation evidence is current by mtime ({age_days:.1f}d)'))

    return 'P2: Mutation Evidence Freshness', findings


# ---------------------------------------------------------------------------
# P3 — Crash Risk Analysis (source graph)
# ---------------------------------------------------------------------------
SOURCE_GRAPH_DB = LIB_ROOT / "tools" / "source_graph_kit" / "source_graph.db"


def check_crash_risks(_conn):
    findings = []

    if not SOURCE_GRAPH_DB.exists():
        findings.append(('WARN', 'Source graph DB not found — skipping crash risk analysis'))
        return 'P3: Crash Risks', findings

    # Patterns that identify test/bench/audit harness files (not production code)
    _TEST_FILE_PATTERNS = ('test_', 'bench_', 'bench/', 'audit_', 'fuzz_', 'harness')

    sg_conn = sqlite3.connect(str(SOURCE_GRAPH_DB))
    sg_conn.row_factory = sqlite3.Row
    try:
        rows = sg_conn.execute("""
            SELECT risk_type, COUNT(*) as cnt FROM crash_risks GROUP BY risk_type
        """).fetchall()
        total = sum(r['cnt'] for r in rows)

        # Check crash risks in CT-sensitive files, excluding test/bench/audit harnesses
        # Use DISTINCT to avoid JOIN fan-out when a file has multiple CT symbols
        ct_rows = sg_conn.execute("""
            SELECT DISTINCT cr.file, cr.line, cr.risk_type FROM crash_risks cr
            JOIN symbol_metadata sm ON cr.file = sm.file_path
            WHERE sm.ct_sensitive = 1
        """).fetchall()
        ct_production = [r for r in ct_rows
                         if not any(pat in os.path.basename(r['file']) for pat in _TEST_FILE_PATTERNS)
                         and '/bench/' not in r['file']]
        ct_prod_count = len(ct_production)
        ct_total_count = len(ct_rows)
    finally:
        sg_conn.close()

    breakdown = ", ".join(f'{r["risk_type"]}={r["cnt"]}' for r in rows)
    findings.append(('INFO', f'Total crash risks: {total} ({breakdown})'))

    if ct_prod_count > 0:
        findings.append(('WARN', f'{ct_prod_count} crash risks in CT-sensitive production code'))
    elif ct_total_count > 0:
        findings.append(('INFO', f'{ct_total_count} crash risks in CT-sensitive test/bench files (not production)'))
        findings.append(('PASS', f'No crash risks in CT-sensitive production code ({total} total)'))
    else:
        findings.append(('PASS', f'No crash risks in CT-sensitive code ({total} total, none in CT paths)'))

    return 'P3: Crash Risks', findings


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# G-1..G-10 CAAS roadmap sub-gates (lightweight doc cross-reference checks)
# ---------------------------------------------------------------------------

DOCS_DIR = LIB_ROOT / "docs"


def check_threat_model(_conn):
    """G-1 gate: verify THREAT_MODEL.md is well-formed and cross-references
    every RR-* citation back to RESIDUAL_RISK_REGISTER.md and every AM-N back
    to its own §3 attacker-model section."""
    findings = []
    tm_path = DOCS_DIR / "THREAT_MODEL.md"
    rr_path = DOCS_DIR / "RESIDUAL_RISK_REGISTER.md"
    if not tm_path.is_file():
        findings.append(('FAIL', 'docs/THREAT_MODEL.md missing (G-1 not closed)'))
        return 'G-1: Threat Model', findings
    if not rr_path.is_file():
        findings.append(('FAIL', 'docs/RESIDUAL_RISK_REGISTER.md missing'))
        return 'G-1: Threat Model', findings

    tm_text = tm_path.read_text(encoding='utf-8', errors='replace')
    rr_text = rr_path.read_text(encoding='utf-8', errors='replace')

    # 1. STRIDE-per-ABI coverage: require all 6 STRIDE categories named
    #    (accept "Info-disclosure" / "Info Disclosure" / "Information disclosure"
    #    and "DoS" / "Denial-of-service" / "Denial of service" as variants)
    stride_patterns = [
        ('Spoofing',     r'\bSpoofing\b'),
        ('Tampering',    r'\bTampering\b'),
        ('Repudiation',  r'\bRepudiation\b'),
        ('Info-disclosure', r'\bInfo[- ]?disclosure\b|\bInformation[- ]disclosure\b'),
        ('DoS',          r'\bDoS\b|\bDenial[- ]of[- ]service\b|\bDenial\b'),
        ('Elevation',    r'\bElevation\b'),
    ]
    missing_stride = [name for name, pat in stride_patterns
                      if not re.search(pat, tm_text, flags=re.IGNORECASE)]
    if missing_stride:
        findings.append(('FAIL', f'THREAT_MODEL.md missing STRIDE term(s): {missing_stride}'))
    else:
        findings.append(('PASS', 'THREAT_MODEL.md covers all 6 STRIDE categories'))

    # 2. AM-N cross-refs: every AM-<n> cited must be defined somewhere in the
    #    file (section header OR a table row beginning with "| AM-N |").
    am_cited = set(re.findall(r'\bAM-(\d+)\b', tm_text))
    am_defined = set(re.findall(r'^\|\s*AM-(\d+)\s*\|', tm_text, flags=re.MULTILINE))
    am_defined |= set(re.findall(r'^\s*#+\s*AM-(\d+)\b', tm_text, flags=re.MULTILINE))
    am_dangling = sorted(am_cited - am_defined, key=int)
    if am_dangling:
        findings.append(('FAIL', f'THREAT_MODEL.md cites undefined AM-N: {am_dangling}'))
    else:
        findings.append(('PASS', f'{len(am_cited)} AM-N references, all defined'))

    # 3. RR-NNN cross-refs: every RR-NNN cited must be defined in RR register
    rr_cited = set(re.findall(r'\bRR-(\d{3,})\b', tm_text))
    rr_defined = set(re.findall(r'\bRR-(\d{3,})\b', rr_text))
    rr_dangling = sorted(rr_cited - rr_defined)
    if rr_dangling:
        findings.append(('FAIL', f'THREAT_MODEL.md cites undefined RR-NNN: {rr_dangling[:10]}'))
    else:
        findings.append(('PASS', f'{len(rr_cited)} RR-NNN references, all defined in register'))

    return 'G-1: Threat Model', findings


def check_residual_risk_register(_conn):
    """Verify every RR-NNN entry in RESIDUAL_RISK_REGISTER.md is a table row
    with at least Disposition and Scope populated. The register stores entries
    as pipe-delimited rows: `| RR-NNN | Risk | Disposition | Scope | Details |`."""
    findings = []
    rr_path = DOCS_DIR / "RESIDUAL_RISK_REGISTER.md"
    if not rr_path.is_file():
        findings.append(('FAIL', 'docs/RESIDUAL_RISK_REGISTER.md missing'))
        return 'G-1b: Residual Risk Register', findings

    text = rr_path.read_text(encoding='utf-8', errors='replace')
    # Parse table rows: | RR-NNN | col2 | col3 | col4 | col5 |
    row_re = re.compile(
        r'^\|\s*(RR-\d{3,})\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|',
        flags=re.MULTILINE,
    )
    rows = row_re.findall(text)
    if not rows:
        findings.append(('FAIL', 'RESIDUAL_RISK_REGISTER.md has no RR-NNN table rows'))
        return 'G-1b: Residual Risk Register', findings

    bad = []
    for rr_id, risk, disposition, scope, details in rows:
        if not disposition.strip() or disposition.strip() == '-':
            bad.append(f'{rr_id} has blank Disposition')
        if not scope.strip() or scope.strip() == '-':
            bad.append(f'{rr_id} has blank Scope')
        if not details.strip() or len(details.strip()) < 20:
            bad.append(f'{rr_id} has thin Details (<20 chars)')
    if bad:
        findings.append(('FAIL', f'RR entries with missing fields: {bad[:5]} (total {len(bad)})'))
    else:
        findings.append(('PASS',
                         f'{len(rows)} RR-NNN table rows, all with Risk/Disposition/Scope/Details'))

    return 'G-1b: Residual Risk Register', findings


def check_disclosure_sla(_conn):
    """G-10 gate: verify SECURITY.md has tiered SLA and .well-known/security.txt
    has RFC 9116 required fields (Contact, Expires)."""
    findings = []
    sec_md = LIB_ROOT / "SECURITY.md"
    wk = LIB_ROOT / ".well-known" / "security.txt"

    if not sec_md.is_file():
        findings.append(('FAIL', 'SECURITY.md missing (G-10 not closed)'))
    else:
        sec_text = sec_md.read_text(encoding='utf-8', errors='replace')
        required_phrases = ['Critical', 'High', 'Medium', 'Low']
        missing = [p for p in required_phrases if p not in sec_text]
        if missing:
            findings.append(('FAIL', f'SECURITY.md SLA missing tier(s): {missing}'))
        else:
            findings.append(('PASS', 'SECURITY.md has tiered SLA (Critical/High/Medium/Low)'))

    if not wk.is_file():
        findings.append(('FAIL', '.well-known/security.txt missing (RFC 9116)'))
    else:
        wk_text = wk.read_text(encoding='utf-8', errors='replace')
        # RFC 9116 required: Contact, Expires
        for field in ('Contact:', 'Expires:'):
            if field not in wk_text:
                findings.append(('FAIL', f'.well-known/security.txt missing "{field}"'))
        # Expires must be in the future (best-effort ISO 8601 parse)
        m = re.search(r'Expires:\s*(\S+)', wk_text)
        if m:
            try:
                from datetime import datetime as _dt
                exp = _dt.fromisoformat(m.group(1).replace('Z', '+00:00'))
                now = _dt.now(exp.tzinfo) if exp.tzinfo else _dt.utcnow()
                if exp < now:
                    findings.append(('FAIL',
                                     f'.well-known/security.txt Expires={m.group(1)} is in the past'))
                else:
                    findings.append(('PASS',
                                     f'.well-known/security.txt Expires={m.group(1)} future'))
            except ValueError:
                findings.append(('WARN',
                                 f'.well-known/security.txt Expires={m.group(1)} unparseable'))

    return 'G-10: Disclosure SLA + RFC 9116', findings


def check_ct_tool_agreement(_conn):
    """G-8 gate: verify CT_TOOL_INDEPENDENCE.md §6 coverage table has all three
    primary tools reporting a non-blank verdict for every listed function."""
    findings = []
    ct_path = DOCS_DIR / "CT_TOOL_INDEPENDENCE.md"
    if not ct_path.is_file():
        findings.append(('FAIL', 'docs/CT_TOOL_INDEPENDENCE.md missing (G-8 not closed)'))
        return 'G-8: CT Tool Independence', findings

    text = ct_path.read_text(encoding='utf-8', errors='replace')

    # Find the §6 coverage table header row and data rows
    m = re.search(r'\|\s*Function\s*\|\s*dudect\s*\|\s*Valgrind CT\s*\|\s*ct-verif\s*\|\s*Verdict\s*\|',
                  text, flags=re.IGNORECASE)
    if not m:
        findings.append(('FAIL', 'CT_TOOL_INDEPENDENCE.md §6 coverage table not found '
                                  '(columns Function | dudect | Valgrind CT | ct-verif | Verdict)'))
        return 'G-8: CT Tool Independence', findings

    # Read subsequent lines that look like data rows (start with "| `")
    start = m.end()
    tail = text[start:]
    rows = re.findall(r'^\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*$',
                      tail, flags=re.MULTILINE)
    # Skip the separator row (---)
    data_rows = [r for r in rows if '---' not in r[0] and r[0].strip()]
    if not data_rows:
        findings.append(('FAIL', 'CT_TOOL_INDEPENDENCE.md §6 coverage table has no data rows'))
        return 'G-8: CT Tool Independence', findings

    bad = []
    for func, dudect, valgrind, ctverif, verdict in data_rows:
        # Stop if we walked past the table
        if func.startswith('#') or func.lower().startswith('what'):
            break
        for tool_name, tool_val in (('dudect', dudect), ('Valgrind', valgrind), ('ct-verif', ctverif)):
            if not tool_val.strip() or tool_val.strip().lower() in ('-', 'n/a', 'missing', 'tbd', '?'):
                bad.append(f'{func.strip()}: {tool_name}={tool_val.strip() or "<blank>"}')
        if not verdict.strip() or 'verified' not in verdict.lower():
            bad.append(f'{func.strip()}: verdict={verdict.strip() or "<blank>"}')

    if bad:
        findings.append(('FAIL', f'CT coverage gaps ({len(bad)}): {bad[:3]}'))
    else:
        findings.append(('PASS', f'{len(data_rows)} CT-claimed functions, all 3 tools agree'))

    return 'G-8: CT Tool Independence', findings


# ---------------------------------------------------------------------------
# P21 — External-Audit Replacement Gate
# ---------------------------------------------------------------------------
def check_external_audit_replacement(conn):
    """P21: verify all required CAAS docs are present to replace verifiable
    parts of a traditional external audit."""
    findings = []

    required = [
        'docs/CAAS_PROTOCOL.md',
        'docs/AUDIT_PHILOSOPHY.md',
        'docs/RESIDUAL_RISK_REGISTER.md',
        'docs/SECURITY_CLAIMS.md',
        'docs/EXPLOIT_TEST_CATALOG.md',
        'docs/EXTERNAL_AUDIT_BUNDLE.json',
        'docs/EXTERNAL_AUDIT_BUNDLE.sha256',
        'docs/CAAS_REVIEWER_QUICKSTART.md',
        'docs/CAAS_FAQ.md',
        'docs/CAAS_THREAT_MODEL.md',
        'docs/NEGATIVE_RESULTS_LEDGER.md',
        'docs/THREAD_SAFETY.md',
        'docs/ABI_VERSIONING.md',
        'docs/SECURITY_INCIDENT_TIMELINE.md',
        'ci/verify_external_audit_bundle.py',
    ]

    missing = []
    for rel in required:
        path = LIB_ROOT / rel
        if path.exists():
            findings.append(('PASS', f'{rel} present'))
        else:
            findings.append(('FAIL', f'{rel} missing'))
            missing.append(rel)

    # Summarise
    n_total = len(required)
    n_missing = len(missing)
    if n_missing == 0:
        findings.insert(0, ('PASS', f'P21: all required CAAS docs present ({n_total} files)'))
    else:
        findings.insert(0, ('FAIL', f'P21: missing {n_missing} required docs: {missing}'))

    return 'P21: External-Audit Replacement Gate', findings


# ---------------------------------------------------------------------------
# G-10 — Spec Traceability
# ---------------------------------------------------------------------------
def check_spec_traceability(conn):
    """G-10: verify key specification clauses are mapped to tests in
    docs/SPEC_TRACEABILITY_MATRIX.md."""
    findings = []

    matrix_path = LIB_ROOT / 'docs' / 'SPEC_TRACEABILITY_MATRIX.md'
    if not matrix_path.exists():
        findings.append(('FAIL', 'docs/SPEC_TRACEABILITY_MATRIX.md missing'))
        return 'G-10: Spec Traceability', findings

    try:
        content = matrix_path.read_text(errors='replace')
    except Exception as exc:
        findings.append(('FAIL', f'Cannot read SPEC_TRACEABILITY_MATRIX.md: {exc}'))
        return 'G-10: Spec Traceability', findings

    if not content.strip():
        findings.append(('FAIL', 'SPEC_TRACEABILITY_MATRIX.md is empty'))
        return 'G-10: Spec Traceability', findings

    # Count data rows (lines starting with | that are not separator rows)
    data_rows = [
        ln for ln in content.splitlines()
        if ln.strip().startswith('|') and '---' not in ln
    ]
    # Skip header row (first pipe row)
    n_rows = max(0, len(data_rows) - 1)
    if n_rows >= 10:
        findings.append(('PASS', f'traceability matrix has {n_rows} rows'))
    else:
        findings.append(('WARN', f'traceability matrix has only {n_rows} rows'))

    # Spec clause references to look for
    spec_clauses = [
        ('BIP-340 / BIP340', ['BIP-340', 'BIP340']),
        ('RFC 6979 / RFC6979', ['RFC 6979', 'RFC6979']),
        ('libsecp / compat', ['libsecp', 'compat']),
        ('DER', ['DER']),
        ('Taproot / taproot', ['Taproot', 'taproot']),
        ('x-only / xonly', ['x-only', 'xonly']),
    ]
    for label, variants in spec_clauses:
        found = any(v in content for v in variants)
        if found:
            findings.append(('PASS', f'spec clause covered: {label}'))
        else:
            findings.append(('WARN', f'spec clause not in traceability matrix: {label}'))

    return 'G-10: Spec Traceability', findings


# ---------------------------------------------------------------------------
# G-11 — Exploit Traceability
# ---------------------------------------------------------------------------
def check_exploit_traceability(conn):
    """G-11: verify every test_exploit_*.cpp is wired into
    unified_audit_runner.cpp AND listed in EXPLOIT_TEST_CATALOG.md."""
    findings = []

    audit_dir = LIB_ROOT / 'audit'
    exploit_files = sorted(audit_dir.glob('test_exploit_*.cpp'))
    findings.append(('INFO', f'Total exploit files found: {len(exploit_files)}'))

    if not exploit_files:
        findings.append(('WARN', 'No test_exploit_*.cpp files found in audit/'))
        return 'G-11: Exploit Traceability', findings

    # Read unified_audit_runner.cpp
    runner_path = audit_dir / 'unified_audit_runner.cpp'
    if runner_path.exists():
        try:
            runner_text = runner_path.read_text(errors='replace')
        except Exception as exc:
            runner_text = ''
            findings.append(('WARN', f'Cannot read unified_audit_runner.cpp: {exc}'))
    else:
        runner_text = ''
        findings.append(('FAIL', 'audit/unified_audit_runner.cpp not found'))

    # Read EXPLOIT_TEST_CATALOG.md
    catalog_path = LIB_ROOT / 'docs' / 'EXPLOIT_TEST_CATALOG.md'
    if catalog_path.exists():
        try:
            catalog_text = catalog_path.read_text(errors='replace')
        except Exception as exc:
            catalog_text = ''
            findings.append(('WARN', f'Cannot read EXPLOIT_TEST_CATALOG.md: {exc}'))
    else:
        catalog_text = ''
        findings.append(('WARN', 'docs/EXPLOIT_TEST_CATALOG.md not found'))

    un_wired = []
    wired_count = 0
    in_catalog_count = 0

    for ef in exploit_files:
        stem = ef.stem  # e.g. "test_exploit_ecdsa_nonce_reuse"

        # Check wired in runner: look for stem_run() or stem in runner text
        run_sym = stem + '_run'
        if runner_text and run_sym in runner_text:
            wired_count += 1
        else:
            un_wired.append(stem)

        # Check in catalog (less strict)
        if catalog_text and stem in catalog_text:
            in_catalog_count += 1
        else:
            if catalog_text:
                findings.append(('WARN', f'not in EXPLOIT_TEST_CATALOG.md: {stem}'))

    n_total = len(exploit_files)
    if un_wired:
        findings.append(('FAIL',
                         f'un-wired exploit tests ({len(un_wired)}/{n_total}): '
                         f'{", ".join(un_wired[:10])}'
                         f'{" …" if len(un_wired) > 10 else ""}'))
    else:
        findings.append(('PASS', f'wired in runner: {wired_count}/{n_total}'))

    if catalog_text:
        if in_catalog_count == n_total:
            findings.append(('PASS', f'in catalog: {in_catalog_count}/{n_total}'))
        else:
            findings.append(('WARN',
                             f'in catalog: {in_catalog_count}/{n_total} '
                             f'({n_total - in_catalog_count} missing)'))

    return 'G-11: Exploit Traceability', findings


# ---------------------------------------------------------------------------
# G-12 — Source Graph Quality
# ---------------------------------------------------------------------------
def check_source_graph_quality(conn):
    """G-12: check source graph DB quality (existence, freshness, row count).

    NOTE: this check inspects SOURCE_GRAPH_DB (tools/source_graph_kit/source_graph.db),
    NOT the project graph DB (.project_graph.db used by the main conn parameter).
    The two databases are distinct:
      - .project_graph.db  — ABI routing, CT coverage, audit metadata (project graph)
      - source_graph.db    — symbol bodies, call edges, crash risks (semantic source graph)
    """
    findings = []

    # G-12 targets the semantic source graph, not the project graph.
    sg_db = SOURCE_GRAPH_DB

    if not sg_db.exists():
        findings.append(('FAIL',
                         f'Source graph DB not found: {sg_db} '
                         f'(run: python3 tools/source_graph_kit/source_graph.py build -i)'))
        return 'G-12: Source Graph Quality', findings

    findings.append(('PASS', f'Source graph DB exists: {sg_db.relative_to(LIB_ROOT)}'))

    # Check DB staleness vs any .cpp file under cpu/
    cpu_dir = LIB_ROOT / 'cpu'
    db_mtime = sg_db.stat().st_mtime
    stale_files = []
    if cpu_dir.exists():
        for cpp_file in cpu_dir.rglob('*.cpp'):
            try:
                if cpp_file.stat().st_mtime > db_mtime:
                    stale_files.append(str(cpp_file.relative_to(LIB_ROOT)))
            except Exception:
                pass
    if stale_files:
        findings.append(('WARN',
                         f'{len(stale_files)} cpu/*.cpp file(s) newer than source graph DB '
                         f'(rebuild recommended): {stale_files[:3]}'))
    else:
        findings.append(('PASS', 'Source graph DB is up-to-date relative to cpu/ sources'))

    # Count rows in a well-known source-graph table
    try:
        sg_conn = sqlite3.connect(str(sg_db))
        sg_conn.row_factory = sqlite3.Row
        try:
            tables = [
                row[0]
                for row in sg_conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
                ).fetchall()
            ]
            findings.append(('INFO', f'Source graph tables: {", ".join(tables[:10])}'
                             f'{"…" if len(tables) > 10 else ""}'))

            # Pick a candidate table to count rows — source_graph.db uses
            # 'symbols', 'files', or 'entries'; project_graph uses 'c_abi_functions'.
            count_table = None
            for candidate in ('symbols', 'files', 'entries', 'source_files', 'c_abi_functions'):
                if candidate in tables:
                    count_table = candidate
                    break

            if count_table is None and tables:
                count_table = tables[0]

            if count_table:
                row_count = sg_conn.execute(
                    f'SELECT COUNT(*) FROM "{count_table}"'
                ).fetchone()[0]
                if row_count >= 200:
                    findings.append(('PASS',
                                     f'Source graph "{count_table}" has {row_count} rows '
                                     f'(threshold: 200)'))
                else:
                    findings.append(('FAIL',
                                     f'Source graph "{count_table}" has only {row_count} rows '
                                     f'(threshold: 200) — graph may be incomplete'))
            else:
                findings.append(('WARN', 'Source graph DB has no tables — rebuild required'))
        finally:
            sg_conn.close()
    except Exception as exc:
        findings.append(('WARN', f'Could not query source graph DB: {exc}'))

    return 'G-12: Source Graph Quality', findings


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
    '--mutation-freshness': check_mutation_freshness,
    '--crash-risks': check_crash_risks,
    '--threat-model': check_threat_model,
    '--residual-risk-register': check_residual_risk_register,
    '--disclosure-sla': check_disclosure_sla,
    '--ct-tool-agreement': check_ct_tool_agreement,
    '--external-audit-replacement': check_external_audit_replacement,
    '--spec-traceability': check_spec_traceability,
    '--exploit-traceability': check_exploit_traceability,
    '--source-graph-quality': check_source_graph_quality,
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
    check_mutation_freshness,
    check_crash_risks,
    check_threat_model,
    check_residual_risk_register,
    check_disclosure_sla,
    check_ct_tool_agreement,
    check_external_audit_replacement,
    check_spec_traceability,
    check_exploit_traceability,
    check_source_graph_quality,
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
        provenance = collect_provenance()
        warn_count = sum(1 for r in results for f in r['findings'] if f[0] == 'WARN')
        advisory_count = sum(1 for r in results for f in r['findings'] if f[0] in ('WARN', 'INFO'))
        if has_fail:
            verdict = 'FAIL'
        elif warn_count:
            verdict = 'PASS with advisory'
        else:
            verdict = 'PASS'

        report = {
            'schema_version': '1.0.0',
            'run_id': f"audit_gate-{provenance.get('git', {}).get('short', 'noprov')}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}",
            'runner': 'audit_gate',
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'commit': {
                'value': provenance.get('git', {}) if provenance.get('git', {}).get('sha') else None,
                'status': 'available' if provenance.get('git', {}).get('sha') else 'unavailable',
                'reason': None if provenance.get('git', {}).get('sha') else 'git not available',
            },
            'platform': provenance.get('platform', {}).get('system'),
            'provenance': provenance,
            'verdict': verdict,
            'summary': {
                'total_findings': sum(len(r['findings']) for r in results),
                'blocking': sum(1 for r in results for f in r['findings'] if f[0] == 'FAIL'),
                'advisory': advisory_count,
                'skipped_sections': 0,
                'sections': len(results),
            },
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
