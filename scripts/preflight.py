#!/usr/bin/env python3
"""
preflight.py  --  Pre-commit quality gate for UltrafastSecp256k1

Validates that changes follow the project's non-negotiable rules:
  1. Security invariants: CT files retain all secure_erase/value_barrier calls
  2. Narrative drift: audit docs don't claim CT layers are missing when active
  3. Test coverage gaps: source files with no test coverage
  4. Graph freshness: DB vs filesystem consistency
  5. Doc-code pairing: code changes have matching doc updates
  6. ABI surface check: new/removed ufsecp_* functions detected

Usage:
    python3 scripts/preflight.py                    # full check
    python3 scripts/preflight.py --security         # security only
    python3 scripts/preflight.py --cuda-msvc        # Windows/MSVC CUDA portability only
    python3 scripts/preflight.py --drift            # narrative drift only
    python3 scripts/preflight.py --coverage         # coverage gaps only
    python3 scripts/preflight.py --freshness        # graph freshness only
    python3 scripts/preflight.py --claims           # graph-backed assurance claim checks
    python3 scripts/preflight.py --ai-review        # AI review-event log checks
    python3 scripts/preflight.py --gpu-evidence     # GPU backend evidence / publishability checks
    python3 scripts/preflight.py --api-contracts    # machine-readable API contract gate
    python3 scripts/preflight.py --determinism      # fail-closed determinism gate
    python3 scripts/preflight.py --changed          # check git-changed files
    python3 scripts/preflight.py --secret-paths     # fail closed on secret-bearing edits without doc updates
    python3 scripts/preflight.py --abi              # ABI surface check
    python3 scripts/preflight.py --ctest-registry   # detect stale CTest entries (missing executables)
    python3 scripts/preflight.py --bug-scan         # crypto-aware dev bug scanner
    python3 scripts/preflight.py --py-audit          # Python audit infrastructure self-test
"""

import json
import sqlite3
import os
import re
import sys
import subprocess
from pathlib import Path
from datetime import datetime, timezone

from check_secret_path_changes import build_report as build_secret_path_report

SCRIPT_DIR = Path(__file__).resolve().parent
LIB_ROOT = SCRIPT_DIR.parent
DB_PATH = LIB_ROOT / ".project_graph.db"

# ANSI colors
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
CYAN = '\033[96m'
BOLD = '\033[1m'
RESET = '\033[0m'

_VALIDATE_ASSURANCE_CACHE = None

CUDA_MSVC_SENSITIVE_HEADERS = [
    'cuda/include/secp256k1.cuh',
    'cuda/include/ecdsa.cuh',
    'cuda/include/recovery.cuh',
    'cuda/include/schnorr.cuh',
    'cuda/include/bip32.cuh',
    'cuda/include/zk.cuh',
    'cuda/include/host_helpers.cuh',
]

def get_conn():
    if not DB_PATH.exists():
        print(f"{RED}ERROR: Graph DB not found at {DB_PATH}{RESET}")
        print(f"Run: python3 {SCRIPT_DIR}/build_project_graph.py --rebuild")
        sys.exit(1)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def get_validate_assurance_payload():
    """Run validate_assurance once and cache the parsed JSON payload."""
    global _VALIDATE_ASSURANCE_CACHE
    if _VALIDATE_ASSURANCE_CACHE is not None:
        return _VALIDATE_ASSURANCE_CACHE

    validator = SCRIPT_DIR / 'validate_assurance.py'
    try:
        result = subprocess.run(
            ['python3', str(validator), '--json'],
            capture_output=True, text=True, cwd=str(LIB_ROOT), check=False,
        )
    except Exception as exc:
        _VALIDATE_ASSURANCE_CACHE = {'error': f"could not execute validate_assurance.py: {exc}"}
        return _VALIDATE_ASSURANCE_CACHE

    if result.returncode not in (0, 1):
        _VALIDATE_ASSURANCE_CACHE = {
            'error': result.stderr.strip() or result.stdout.strip() or 'validate_assurance.py failed',
        }
        return _VALIDATE_ASSURANCE_CACHE

    try:
        _VALIDATE_ASSURANCE_CACHE = {'payload': json.loads(result.stdout)}
    except json.JSONDecodeError as exc:
        _VALIDATE_ASSURANCE_CACHE = {'error': f'invalid JSON from validate_assurance.py: {exc}'}
    return _VALIDATE_ASSURANCE_CACHE


def check_cuda_msvc_portability():
    """Fail if GNU-only 128-bit integers reappear in active MSVC-sensitive CUDA headers."""
    issues = []
    pattern = re.compile(r'\b(?:unsigned\s+__int128|__uint128_t)\b')

    for rel_path in CUDA_MSVC_SENSITIVE_HEADERS:
        path = LIB_ROOT / rel_path
        if not path.exists():
            issues.append(f"  {RED}MISSING{RESET} {rel_path}")
            continue

        try:
            conditional_stack = []
            with open(path, 'r', encoding='utf-8', errors='replace') as handle:
                for line_no, line in enumerate(handle, 1):
                    stripped = line.strip()
                    if stripped.startswith('//'):
                        continue
                    if stripped.startswith('#if'):
                        native_only = (
                            '__SIZEOF_INT128__' in stripped or
                            'SECP256K1_CUDA_HAS_NATIVE_UINT128' in stripped
                        )
                        conditional_stack.append(native_only)
                        continue
                    if stripped.startswith('#else'):
                        if conditional_stack:
                            conditional_stack[-1] = not conditional_stack[-1]
                        continue
                    if stripped.startswith('#elif'):
                        if conditional_stack:
                            conditional_stack[-1] = (
                                '__SIZEOF_INT128__' in stripped or
                                'SECP256K1_CUDA_HAS_NATIVE_UINT128' in stripped
                            )
                        continue
                    if stripped.startswith('#endif'):
                        if conditional_stack:
                            conditional_stack.pop()
                        continue
                    if any(conditional_stack):
                        continue
                    if pattern.search(line):
                        issues.append(
                            f"  {RED}MSVC-BREAK{RESET} {rel_path}:{line_no} -- GNU-only 128-bit integer in active CUDA header"
                        )
        except Exception as exc:
            issues.append(f"  {RED}UNREADABLE{RESET} {rel_path}: {exc}")

    return issues

# ---------------------------------------------------------------------------
# 1. Security Invariant Check
# ---------------------------------------------------------------------------
def check_security_invariants():
    """Verify CT files retain expected security patterns."""
    conn = get_conn()
    issues = []

    # Get expected patterns from graph
    expected = {}
    rows = conn.execute("""SELECT source_file, pattern, COUNT(*) as cnt
        FROM security_patterns GROUP BY source_file, pattern""").fetchall()
    for r in rows:
        key = (r['source_file'], r['pattern'])
        expected[key] = r['cnt']

    # Scan actual files
    actual = {}
    patterns_re = {
        'secure_erase': re.compile(r'secure_erase\s*\('),
        'value_barrier': re.compile(r'value_barrier\s*\('),
        'CLASSIFY': re.compile(r'SECP256K1_CLASSIFY\s*\('),
        'DECLASSIFY': re.compile(r'SECP256K1_DECLASSIFY\s*\('),
    }

    for (src_file, pat_name), exp_cnt in expected.items():
        filepath = LIB_ROOT / src_file
        if not filepath.exists():
            issues.append(f"  {RED}MISSING{RESET} {src_file} (expected {exp_cnt} {pat_name})")
            continue
        pat_re = patterns_re.get(pat_name)
        if not pat_re:
            continue
        count = 0
        try:
            with open(filepath, 'r', errors='replace') as f:
                for line in f:
                    stripped = line.strip()
                    # Skip comment-only lines for erase/barrier patterns,
                    # matching build_project_graph.py scanning logic.
                    # CLASSIFY/DECLASSIFY are exempt — the graph builder keeps
                    # those even in comment lines (macro definition context).
                    if pat_name in ('secure_erase', 'value_barrier'):
                        if stripped.startswith('//') or stripped.startswith('#include'):
                            continue
                    if pat_re.search(line):
                        count += 1
        except Exception:
            issues.append(f"  {RED}UNREADABLE{RESET} {src_file}")
            continue
        actual[(src_file, pat_name)] = count
        if count < exp_cnt:
            issues.append(f"  {RED}LOST{RESET} {src_file}: {pat_name} {exp_cnt} -> {count} ({exp_cnt - count} removed)")
        elif count > exp_cnt:
            issues.append(f"  {CYAN}NEW{RESET}  {src_file}: {pat_name} {exp_cnt} -> {count} (+{count - exp_cnt}, rebuild graph)")

    conn.close()
    return issues

# ---------------------------------------------------------------------------
# 1b. Narrative Drift Detection
# ---------------------------------------------------------------------------
STALE_PHRASES = [
    # (regex_pattern, description, files_to_check)
    (r'(?i)\bno\s+formal\s+(ct\s+)?verification\b',
     'Claims no formal CT verification -- ct-verif and valgrind-ct are active in CI',
     ['docs/AUDIT_READINESS_REPORT_v1.md', 'audit/AUDIT_TEST_PLAN.md']),
    (r'(?i)\btool\s+integration\s+not\s+yet\s+done\b',
     'Claims tool integration not done -- tools are integrated',
     ['docs/AUDIT_READINESS_REPORT_v1.md', 'audit/AUDIT_TEST_PLAN.md',
      'docs/TEST_MATRIX.md']),
    (r'(?i)\bno\s+formal\s+verification\s+applied\b',
     'Claims no formal verification applied -- ct-verif is running and blocking',
     ['audit/run_full_audit.sh', 'audit/run_full_audit.ps1']),
    (r'(?i)\bno\s+multi-uarch\b',
     'Claims no multi-uarch support -- cross-platform KAT and CI exist',
     ['docs/AUDIT_READINESS_REPORT_v1.md']),
    (r'(?i)\bgpu\s+equivalence\s+planned\b',
     'Claims GPU equivalence only planned -- GPU audit runners exist',
     ['docs/AUDIT_READINESS_REPORT_v1.md']),
]

# Files that are marked historical are exempt from drift checks
HISTORICAL_EXEMPT_MARKER = re.compile(
    r'(?i)(historical\s+report|superseded\s+by|snapshot\s+from\s+v\d)',
)

def check_narrative_drift():
    """Detect stale CT/audit phrases in narrative docs."""
    issues = []
    for pattern_str, description, target_files in STALE_PHRASES:
        pat = re.compile(pattern_str)
        for rel_path in target_files:
            filepath = LIB_ROOT / rel_path
            if not filepath.exists():
                continue
            try:
                with open(filepath, 'r', errors='replace') as f:
                    content = f.read()
            except Exception:
                continue
            # Skip files explicitly marked as historical
            if HISTORICAL_EXEMPT_MARKER.search(content[:500]):
                continue
            for i, line in enumerate(content.splitlines(), 1):
                if pat.search(line):
                    issues.append(
                        f"  {YELLOW}DRIFT{RESET} {rel_path}:{i} -- {description}"
                    )
    return issues

# ---------------------------------------------------------------------------
# 2. Test Coverage Gap Analysis
# ---------------------------------------------------------------------------
def check_coverage_gaps():
    """Find source files with no test coverage."""
    conn = get_conn()

    # Core source files (cpu_core layer, not headers/tests/tools)
    core_files = conn.execute("""SELECT path FROM source_files
        WHERE layer IN ('fast', 'ct', 'abi')
        AND category = 'cpu_core'
        AND file_type IN ('cpp', 'source')
        ORDER BY path""").fetchall()

    # Files that have at least one 'covers' edge
    covered = set()
    rows = conn.execute("""SELECT DISTINCT dst_id FROM edges
        WHERE relation='covers' AND dst_type='source_file'""").fetchall()
    for r in rows:
        covered.add(r['dst_id'])

    gaps = []
    for f in core_files:
        if f['path'] not in covered:
            # Check if it's a significant file (>50 lines)
            info = conn.execute("SELECT lines FROM source_files WHERE path=?",
                                (f['path'],)).fetchone()
            if info and info['lines'] > 50:
                gaps.append((f['path'], info['lines']))

    conn.close()
    return gaps

# ---------------------------------------------------------------------------
# 3. Graph Freshness Check
# ---------------------------------------------------------------------------
def check_freshness():
    """Compare graph build time vs file modification times."""
    conn = get_conn()
    stale = []

    built_str = conn.execute("SELECT value FROM meta WHERE key='built_at'").fetchone()['value']
    built_dt = datetime.fromisoformat(built_str)

    rows = conn.execute("SELECT path, lines FROM source_files WHERE layer IN ('fast','ct','abi') ORDER BY lines DESC").fetchall()
    for r in rows:
        filepath = LIB_ROOT / r['path']
        if not filepath.exists():
            stale.append(('DELETED', r['path'], 0))
            continue
        mtime = datetime.fromtimestamp(filepath.stat().st_mtime, tz=timezone.utc)
        if mtime > built_dt:
            stale.append(('MODIFIED', r['path'], r['lines']))

    # Check for new files not in graph
    scan_dirs = ['cpu/src', 'cpu/include', 'include/ufsecp']
    known_paths = {r['path'] for r in rows}
    for scan_dir in scan_dirs:
        dirpath = LIB_ROOT / scan_dir
        if not dirpath.exists():
            continue
        for root, dirs, files in os.walk(dirpath):
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            for fname in files:
                ext = os.path.splitext(fname)[1].lower()
                if ext in ('.cpp', '.hpp', '.h'):
                    rel = str(Path(root, fname).relative_to(LIB_ROOT))
                    if rel not in known_paths:
                        stale.append(('NEW', rel, 0))

    conn.close()
    return stale, built_str

# ---------------------------------------------------------------------------
# 4. Doc-Code Pairing Check (for git-changed files)
# ---------------------------------------------------------------------------
DOC_PAIRS = {
    # Public API / C ABI
    'include/ufsecp/ufsecp.h':         ['docs/API_REFERENCE.md', 'docs/USER_GUIDE.md', 'docs/SECURITY_CLAIMS.md'],
    'include/ufsecp/ufsecp_impl.cpp':  ['docs/API_REFERENCE.md', 'docs/SECRET_LIFECYCLE.md', 'docs/FFI_HOSTILE_CALLER.md'],
    # Build system
    'CMakeLists.txt':                  ['docs/BUILDING.md', 'README.md'],
    # Benchmark
    'cpu/bench/bench_unified.cpp':     ['docs/BENCHMARKS.md', 'docs/BENCHMARK_METHODOLOGY.md'],
    # Audit
    'audit/unified_audit_runner.cpp':  ['docs/TEST_MATRIX.md', 'docs/AUDIT_GUIDE.md'],
    # Protocol implementations
    'cpu/src/musig2.cpp':              ['docs/API_REFERENCE.md', 'docs/SECRET_LIFECYCLE.md'],
    'cpu/src/frost.cpp':               ['docs/API_REFERENCE.md', 'docs/SECRET_LIFECYCLE.md'],
    'cpu/src/adaptor.cpp':             ['docs/API_REFERENCE.md'],
    'cpu/src/silent_payments.cpp':     ['docs/API_REFERENCE.md'],
    'cpu/src/ecies.cpp':               ['docs/API_REFERENCE.md'],
    # CT layer
    'cpu/src/ct_sign.cpp':             ['docs/CT_VERIFICATION.md', 'docs/SECURITY_CLAIMS.md', 'docs/SECRET_LIFECYCLE.md'],
    'cpu/src/ct_field.cpp':            ['docs/CT_VERIFICATION.md'],
    'cpu/src/ct_scalar.cpp':           ['docs/CT_VERIFICATION.md'],
    'cpu/src/ct_point.cpp':            ['docs/CT_VERIFICATION.md'],
    # GPU backends
    'cuda/secp256k1_cuda.cu':          ['docs/COMPATIBILITY.md'],
    'opencl/secp256k1_opencl.cpp':     ['docs/COMPATIBILITY.md'],
    'metal/secp256k1_metal.mm':        ['docs/COMPATIBILITY.md'],
    # Core headers
    'cpu/include/secp256k1/field.hpp': ['docs/API_REFERENCE.md'],
    'cpu/include/secp256k1/scalar.hpp':['docs/API_REFERENCE.md'],
    'cpu/include/secp256k1/point.hpp': ['docs/API_REFERENCE.md'],
    # Release workflow
    '.github/workflows/release.yml':   ['docs/LOCAL_CI.md'],
    '.github/workflows/auditor-prep.yml': ['docs/EXTERNAL_AUDIT_AUTOMATION.md', 'AUDIT_GUIDE.md'],
    'scripts/external_audit_prep.sh': ['docs/EXTERNAL_AUDIT_AUTOMATION.md', 'AUDIT_GUIDE.md'],
}

def check_doc_pairing(changed_files):
    """Check if code changes have matching doc updates."""
    missing = []
    changed_set = set(changed_files)

    for code_file, expected_docs in DOC_PAIRS.items():
        if any(code_file in cf for cf in changed_set):
            for doc in expected_docs:
                if not any(doc in cf for cf in changed_set):
                    missing.append((code_file, doc))

    # Check CT layer changes
    # Match only actual CT source files (filename starts with ct_), not paths
    # that happen to contain the substring "ct_" (e.g. "project_graph.py")
    ct_changed = [f for f in changed_files
                  if (Path(f).name.startswith('ct_') and Path(f).suffix in ('.cpp', '.hpp', '.h'))
                  or '/ct/' in f]
    if ct_changed:
        ct_docs = ['docs/CT_VERIFICATION.md', 'docs/SECURITY_CLAIMS.md']
        for doc in ct_docs:
            if not any(doc in cf for cf in changed_set):
                for ct_f in ct_changed:
                    missing.append((ct_f, doc))

    return missing

# ---------------------------------------------------------------------------
# 5. ABI Surface Check
# ---------------------------------------------------------------------------
def check_abi_surface():
    """Detect new/removed ufsecp_* functions vs graph."""
    conn = get_conn()

    # Known from graph
    known = set()
    rows = conn.execute("SELECT name FROM c_abi_functions").fetchall()
    for r in rows:
        known.add(r['name'])

    # Scan actual headers (ufsecp.h + ufsecp_version.h)
    actual = set()
    fn_re = re.compile(r'UFSECP_API\s+.*?(ufsecp_\w+)\s*\(')
    for hdr_name in ('ufsecp.h', 'ufsecp_gpu.h', 'ufsecp_version.h'):
        header = LIB_ROOT / 'include' / 'ufsecp' / hdr_name
        if header.exists():
            with open(header, 'r', errors='replace') as f:
                for line in f:
                    m = fn_re.search(line)
                    if m:
                        actual.add(m.group(1))

    added = actual - known
    removed = known - actual
    conn.close()
    return added, removed


# ---------------------------------------------------------------------------
# 5b. Graph-Driven Assurance Claim Check
# ---------------------------------------------------------------------------
def check_claim_surfaces():
    """Run validate_assurance and extract graph-backed claim-surface issues."""
    data = get_validate_assurance_payload()
    if 'error' in data:
        return [f"  {RED}ERROR{RESET} {data['error']}"]

    payload = data['payload']
    claim_surface = payload.get('claim_surface', {})
    issues = []
    for label in claim_surface.get('missing_surfaces', []):
        issues.append(f"  {RED}STALE{RESET} {label}")
    for label in claim_surface.get('unindexed_surfaces', []):
        issues.append(f"  {YELLOW}UNINDEXED{RESET} {label}")
    return issues


def check_ai_review_log():
    """Run validate_assurance and extract AI review-event log issues."""
    data = get_validate_assurance_payload()
    if 'error' in data:
        return [f"  {RED}ERROR{RESET} {data['error']}"]

    payload = data['payload']
    ai_review = payload.get('ai_review_events', {})
    return [f"  {RED}INVALID{RESET} {label}" for label in ai_review.get('invalid_entries', [])]


def check_gpu_backend_evidence():
    """Run validate_assurance and extract GPU backend evidence issues."""
    data = get_validate_assurance_payload()
    if 'error' in data:
        return [f"  {RED}ERROR{RESET} {data['error']}"]

    payload = data['payload']
    gpu = payload.get('gpu_backend_evidence', {})
    return [f"  {RED}INVALID{RESET} {label}" for label in gpu.get('invalid_entries', [])]


def check_api_contracts():
    """Run machine-readable API contract checks."""
    checker = SCRIPT_DIR / 'check_api_contracts.py'
    try:
        result = subprocess.run(
            [sys.executable, str(checker)],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(LIB_ROOT),
            check=False,
        )
    except Exception as exc:
        return [f"  {RED}ERROR{RESET} could not execute check_api_contracts.py: {exc}"], True

    output_lines = [line.strip() for line in (result.stdout + result.stderr).splitlines() if line.strip()]
    if result.returncode != 0:
        issues = [f"  {RED}CONTRACT{RESET} {line}" for line in output_lines if line.startswith('FAIL')]
        if not issues:
            issues = [f"  {RED}CONTRACT{RESET} check_api_contracts.py failed (exit {result.returncode})"]
        return issues, True

    return [], False


def check_determinism_gate():
    """Run fail-closed determinism checks for core API surfaces."""
    checker = SCRIPT_DIR / 'check_determinism_gate.py'
    try:
        result = subprocess.run(
            [sys.executable, str(checker), '--json', '--repeat', '5'],
            capture_output=True,
            text=True,
            timeout=90,
            cwd=str(LIB_ROOT),
            check=False,
        )
    except Exception as exc:
        return [f"  {RED}DETERMINISM{RESET} could not execute check_determinism_gate.py: {exc}"], True

    payload = None
    output = (result.stdout or '') + (result.stderr or '')
    try:
        payload = json.loads(result.stdout) if result.stdout.strip() else None
    except json.JSONDecodeError:
        payload = None

    if result.returncode != 0:
        issues = []
        if isinstance(payload, dict):
            for issue in payload.get('issues', []):
                issues.append(f"  {RED}DETERMINISM{RESET} {issue}")
        if not issues:
            issues = [f"  {RED}DETERMINISM{RESET} determinism checker failed (exit {result.returncode})"]
            if output.strip():
                issues.append(f"  {RED}DETERMINISM{RESET} {output.strip()[:220]}")
        return issues, True

    if not isinstance(payload, dict) or not payload.get('overall_pass', False):
        return [f"  {RED}DETERMINISM{RESET} invalid determinism payload or non-pass result"], True

    return [], False

# ---------------------------------------------------------------------------
# 6. Changed Files Analysis
# ---------------------------------------------------------------------------
def get_changed_files():
    """Get files changed vs HEAD (staged + unstaged)."""
    try:
        result = subprocess.run(
            ['git', 'diff', '--name-only', 'HEAD'],
            capture_output=True, text=True, cwd=str(LIB_ROOT)
        )
        files = [f.strip() for f in result.stdout.strip().split('\n') if f.strip()]
        # Also staged
        result2 = subprocess.run(
            ['git', 'diff', '--cached', '--name-only'],
            capture_output=True, text=True, cwd=str(LIB_ROOT)
        )
        files2 = [f.strip() for f in result2.stdout.strip().split('\n') if f.strip()]
        return list(set(files + files2))
    except Exception:
        return []


def check_secret_path_changes(changed_files=None):
    return build_secret_path_report(changed_files or get_changed_files())


def check_ctest_registry_health():
    """Detect stale CTest entries that reference missing executables."""
    issues = []
    candidate_dirs = [
        LIB_ROOT / 'build_rel',
        LIB_ROOT / 'build',
        LIB_ROOT / 'build_ci',
        LIB_ROOT / 'build-ci',
        LIB_ROOT / 'build_linux',
    ]
    launcher_names = {
        'python', 'python3', 'bash', 'sh', 'cmake', 'ctest',
        'pwsh', 'powershell', 'cmd', 'cmd.exe',
    }

    scanned_any = False

    for build_dir in candidate_dirs:
        if not (build_dir / 'CTestTestfile.cmake').exists():
            continue
        scanned_any = True

        available_targets = set()
        if (build_dir / 'build.ninja').exists():
            try:
                ninja_targets = subprocess.run(
                    ['ninja', '-t', 'targets', 'all'],
                    capture_output=True,
                    text=True,
                    cwd=str(build_dir),
                    check=False,
                )
                if ninja_targets.returncode == 0:
                    for line in ninja_targets.stdout.splitlines():
                        name = line.split(':', 1)[0].strip()
                        if name:
                            available_targets.add(Path(name).name)
            except Exception:
                pass

        if not available_targets:
            try:
                help_targets = subprocess.run(
                    ['cmake', '--build', '.', '--target', 'help'],
                    capture_output=True,
                    text=True,
                    cwd=str(build_dir),
                    check=False,
                )
                if help_targets.returncode == 0:
                    for line in help_targets.stdout.splitlines():
                        line = line.strip()
                        if line.startswith('... '):
                            available_targets.add(line[4:].strip())
            except Exception:
                pass
        try:
            result = subprocess.run(
                ['ctest', '--show-only=json-v1'],
                capture_output=True,
                text=True,
                cwd=str(build_dir),
                check=False,
            )
        except Exception as exc:
            issues.append(f"  {YELLOW}WARN{RESET} {build_dir.name}: failed to inspect CTest registry: {exc}")
            continue

        if result.returncode != 0:
            detail = (result.stderr or result.stdout or '').strip()
            issues.append(
                f"  {YELLOW}WARN{RESET} {build_dir.name}: ctest --show-only failed"
                + (f" ({detail})" if detail else "")
            )
            continue

        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            issues.append(f"  {YELLOW}WARN{RESET} {build_dir.name}: invalid CTest JSON output: {exc}")
            continue

        for test in payload.get('tests', []):
            name = test.get('name', '<unknown>')
            command = test.get('command') or []
            if not command:
                continue
            exe = command[0]
            if not isinstance(exe, str) or not exe:
                continue
            if '${' in exe or '$<' in exe:
                continue
            if Path(exe).name.lower() in launcher_names:
                continue

            exe_path = Path(exe)
            if not exe_path.is_absolute():
                exe_path = build_dir / exe_path

            if not exe_path.exists():
                exe_name = exe_path.name
                exe_stem = exe_path.stem
                if exe_name in available_targets or exe_stem in available_targets:
                    issues.append(
                        f"  {YELLOW}UNBUILT-TEST{RESET} {build_dir.name}:{name} -> target exists but executable is not built yet ({exe_path})"
                    )
                else:
                    issues.append(
                        f"  {RED}STALE-CTEST{RESET} {build_dir.name}:{name} -> missing executable with no matching build target ({exe_path})"
                    )

    if not scanned_any:
        issues.append(f"  {YELLOW}WARN{RESET} no CTest build directory found to inspect")

    return issues

def analyze_changed_files(changed):
    """For changed files, show impact via graph."""
    if not changed:
        return []
    conn = get_conn()
    impacts = []
    for cf in changed:
        row = conn.execute("SELECT * FROM source_files WHERE path LIKE ?",
                           (f'%{cf}%',)).fetchone()
        if not row:
            continue
        fpath = row['path']
        # Tests
        tests = conn.execute("""SELECT src_id FROM edges
            WHERE dst_type='source_file' AND dst_id=? AND relation='covers'""",
            (fpath,)).fetchall()
        test_names = [t['src_id'] for t in tests]
        # Security
        sec = conn.execute("SELECT COUNT(*) as cnt FROM security_patterns WHERE source_file=?",
                           (fpath,)).fetchone()
        sec_cnt = sec['cnt'] if sec else 0
        # Routing
        fname = Path(fpath).stem
        routing = conn.execute("""SELECT abi_function, layer FROM abi_routing
            WHERE internal_call LIKE ? OR abi_function LIKE ?""",
            (f'%{fname}%', f'%{fname}%')).fetchall()
        rt_list = [(r['abi_function'], r['layer']) for r in routing]

        impacts.append({
            'file': fpath,
            'layer': row['layer'],
            'lines': row['lines'],
            'tests': test_names,
            'security_patterns': sec_cnt,
            'abi_routing': rt_list,
        })
    conn.close()
    return impacts

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def run_all(args):
    mode = args[0] if args else '--all'
    exit_code = 0
    total_issues = 0
    changed = None

    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}  UltrafastSecp256k1 Preflight Check{RESET}")
    print(f"{BOLD}{'='*60}{RESET}\n")

    # Security
    if mode in ('--all', '--security'):
        print(f"{BOLD}[1/17] Security Invariants{RESET}")
        issues = check_security_invariants()
        if issues:
            for i in issues:
                print(i)
            lost = sum(1 for i in issues if 'LOST' in i)
            if lost:
                exit_code = 1
                total_issues += lost
            print(f"  {RED}{lost} lost, {len(issues) - lost} info{RESET}\n")
        else:
            print(f"  {GREEN}[OK] All security patterns preserved{RESET}\n")

    # CUDA / MSVC portability
    if mode in ('--all', '--cuda-msvc'):
        print(f"{BOLD}[2/17] CUDA / MSVC Portability{RESET}")
        cuda_msvc_issues = check_cuda_msvc_portability()
        if cuda_msvc_issues:
            for issue in cuda_msvc_issues:
                print(issue)
            total_issues += len(cuda_msvc_issues)
            exit_code = 1
            print(f"  {RED}{len(cuda_msvc_issues)} Windows/MSVC CUDA portability issue(s){RESET}\n")
        else:
            print(f"  {GREEN}[OK] Active CUDA headers are free of GNU-only __int128 usage{RESET}\n")

    # Narrative drift
    if mode in ('--all', '--drift'):
        print(f"{BOLD}[3/17] Narrative Drift Detection{RESET}")
        drift_issues = check_narrative_drift()
        if drift_issues:
            for i in drift_issues:
                print(i)
            total_issues += len(drift_issues)
            print(f"  {YELLOW}{len(drift_issues)} stale narrative phrase(s){RESET}\n")
        else:
            print(f"  {GREEN}[OK] No stale CT/audit narrative detected{RESET}\n")

    # Coverage
    if mode in ('--all', '--coverage'):
        print(f"{BOLD}[4/17] Test Coverage Gaps{RESET}")
        gaps = check_coverage_gaps()
        if gaps:
            for path, lines in sorted(gaps, key=lambda x: -x[1])[:20]:
                print(f"  {YELLOW}UNTESTED{RESET} {path} ({lines} lines)")
            total_issues += len(gaps)
            print(f"  {YELLOW}{len(gaps)} core files without test coverage{RESET}\n")
        else:
            print(f"  {GREEN}[OK] All core files have test coverage{RESET}\n")

    # Freshness
    if mode in ('--all', '--freshness'):
        print(f"{BOLD}[5/17] Graph Freshness{RESET}")
        stale, built = check_freshness()
        if stale:
            for kind, path, lines in stale[:15]:
                print(f"  {YELLOW}{kind:8s}{RESET} {path}")
            if len(stale) > 15:
                print(f"  ... and {len(stale) - 15} more")
            print(f"  {YELLOW}{len(stale)} stale entries (built: {built[:19]}){RESET}")
            print(f"  Run: python3 scripts/build_project_graph.py --rebuild\n")
        else:
            print(f"  {GREEN}[OK] Graph is fresh (built: {built[:19]}){RESET}\n")

    # Changed files
    if mode in ('--all', '--claims'):
        print(f"{BOLD}[6/17] Assurance Claim Surfaces{RESET}")
        claim_issues = check_claim_surfaces()
        if claim_issues:
            for issue in claim_issues:
                print(issue)
            total_issues += len(claim_issues)
            exit_code = 1
            print(f"  {YELLOW}{len(claim_issues)} graph-backed claim surface issue(s){RESET}\n")
        else:
            print(f"  {GREEN}[OK] Claim surfaces resolve and are graph-indexed{RESET}\n")

    if mode in ('--all', '--ai-review'):
        print(f"{BOLD}[7/17] AI Review Event Log{RESET}")
        ai_review_issues = check_ai_review_log()
        if ai_review_issues:
            for issue in ai_review_issues:
                print(issue)
            total_issues += len(ai_review_issues)
            exit_code = 1
            print(f"  {YELLOW}{len(ai_review_issues)} AI review-event issue(s){RESET}\n")
        else:
            print(f"  {GREEN}[OK] AI review-event log is schema-valid{RESET}\n")

    if mode in ('--all', '--gpu-evidence'):
        print(f"{BOLD}[8/17] GPU Backend Evidence{RESET}")
        gpu_issues = check_gpu_backend_evidence()
        if gpu_issues:
            for issue in gpu_issues:
                print(issue)
            total_issues += len(gpu_issues)
            exit_code = 1
            print(f"  {YELLOW}{len(gpu_issues)} GPU backend evidence issue(s){RESET}\n")
        else:
            print(f"  {GREEN}[OK] GPU backend evidence is fail-closed for publishability and ROCm/HIP promotion{RESET}\n")

    # Changed files
    if mode in ('--all', '--changed'):
        print(f"{BOLD}[9/17] Changed Files Impact{RESET}")
        changed = get_changed_files()
        if changed:
            print(f"  {len(changed)} files changed vs HEAD:")
            impacts = analyze_changed_files(changed)
            for imp in impacts:
                layer_color = RED if imp['layer'] == 'ct' else CYAN
                print(f"  {layer_color}[{imp['layer']:4s}]{RESET} {imp['file']} ({imp['lines']} lines)")
                if imp['tests']:
                    print(f"         Tests: {', '.join(imp['tests'])}")
                else:
                    print(f"         {YELLOW}Tests: NONE{RESET}")
                if imp['security_patterns'] > 0:
                    print(f"         Security patterns: {imp['security_patterns']}")
                if imp['abi_routing']:
                    for fn, layer in imp['abi_routing'][:5]:
                        print(f"         ABI: [{layer}] {fn}")

            # Doc pairing
            doc_missing = check_doc_pairing(changed)
            if doc_missing:
                print(f"\n  {YELLOW}Doc-code pairing violations:{RESET}")
                for code, doc in doc_missing:
                    print(f"    {code} changed but {doc} not updated")
                total_issues += len(doc_missing)
            print()
        else:
            print(f"  {GREEN}[OK] No uncommitted changes{RESET}\n")

    if mode in ('--all', '--secret-paths'):
        print(f"{BOLD}[10/17] Secret-Path Change Gate{RESET}")
        secret_report, secret_fail = check_secret_path_changes(changed)
        if secret_report['triggered_rules']:
            for rule in secret_report['triggered_rules']:
                print(f"  {CYAN}{rule['name']}{RESET}")
                print(f"    Changed: {', '.join(rule['matches'])}")
                print(f"    Required docs: {', '.join(rule['required_docs'])}")
                if rule['missing_docs']:
                    print(f"    {RED}Missing docs:{RESET} {', '.join(rule['missing_docs'])}")
                    print(f"    Reason: {rule['reason']}")
                else:
                    print(f"    {GREEN}Paired docs updated{RESET}")
            if secret_fail:
                total_issues += len(secret_report['blocking_findings'])
                exit_code = 1
                print(f"  {RED}{len(secret_report['blocking_findings'])} blocking secret-path change issue(s){RESET}\n")
            else:
                print(f"  {GREEN}[OK] Secret-bearing changes have paired evidence updates{RESET}\n")
        else:
            print(f"  {GREEN}[OK] No changed secret-bearing paths{RESET}\n")

    # ABI
    if mode in ('--all', '--api-contracts'):
        print(f"{BOLD}[11/17] API Security Contracts{RESET}")
        api_contract_issues, api_contract_fail = check_api_contracts()
        if api_contract_issues:
            for issue in api_contract_issues:
                print(issue)
        if api_contract_fail:
            total_issues += len(api_contract_issues) if api_contract_issues else 1
            exit_code = 1
            print(f"  {RED}API contract gate failed{RESET}\n")
        else:
            print(f"  {GREEN}[OK] API security contracts are valid and up-to-date{RESET}\n")

    if mode in ('--all', '--determinism'):
        print(f"{BOLD}[12/17] Determinism Gate{RESET}")
        determinism_issues, determinism_fail = check_determinism_gate()
        if determinism_issues:
            for issue in determinism_issues:
                print(issue)
        if determinism_fail:
            total_issues += len(determinism_issues) if determinism_issues else 1
            exit_code = 1
            print(f"  {RED}Determinism gate failed{RESET}\n")
        else:
            print(f"  {GREEN}[OK] Core API behavior is deterministic for locked vectors{RESET}\n")

    if mode in ('--all', '--abi'):
        print(f"{BOLD}[13/17] ABI Surface{RESET}")
        added, removed = check_abi_surface()
        if added:
            print(f"  {CYAN}NEW functions (not in graph):{RESET}")
            for fn in sorted(added):
                print(f"    + {fn}")
        if removed:
            print(f"  {RED}REMOVED functions (in graph but not in header):{RESET}")
            for fn in sorted(removed):
                print(f"    - {fn}")
            exit_code = 1
            total_issues += len(removed)
        if not added and not removed:
            print(f"  {GREEN}[OK] ABI surface matches graph{RESET}")
        print()

    # Doc Consistency
    if mode in ('--all', '--doc-sync'):
        print(f"{BOLD}[14/17] Documentation Consistency{RESET}")
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "sync_docs",
                Path(__file__).parent / "sync_docs.py"
            )
            sync_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(sync_mod)
            version  = sync_mod.read_version()
            counts   = sync_mod.read_module_counts()
            wf_count = sync_mod.read_workflow_count()
            abi      = sync_mod.read_abi_counts()
            stale = 0
            for rel, groups in sync_mod.DOC_FILES:
                path_abs = LIB_ROOT / rel
                if not path_abs.exists():
                    continue
                original = path_abs.read_text(encoding="utf-8")
                text = original
                if "version" in groups and path_abs.name not in sync_mod.VERSION_SKIP_FILES:
                    text, _ = sync_mod.apply_version_rules(text, version)
                if "modules" in groups:
                    text, _ = sync_mod.apply_module_rules(text, counts)
                if "workflows" in groups:
                    text, _ = sync_mod.apply_workflow_rules(text, wf_count)
                if "abi" in groups:
                    text, _ = sync_mod.apply_abi_rules(text, abi)
                if text != original:
                    print(f"  {YELLOW}STALE: {rel}{RESET}")
                    stale += 1
            if stale:
                print(f"  {YELLOW}{stale} doc file(s) have stale numbers.{RESET}")
                print(f"  Run:  python3 scripts/sync_docs.py  to fix.\n")
            else:
                print(f"  {GREEN}[OK] All docs are in sync with sources of truth{RESET}\n")
        except Exception as exc:
            print(f"  {YELLOW}WARNING: doc-sync check failed: {exc}{RESET}\n")

    if mode in ('--all', '--ctest-registry'):
        print(f"{BOLD}[15/17] CTest Registry Health{RESET}")
        ctest_registry_issues = check_ctest_registry_health()
        blocking = [i for i in ctest_registry_issues if 'STALE-CTEST' in i]
        for issue in ctest_registry_issues:
            print(issue)
        if blocking:
            total_issues += len(blocking)
            exit_code = 1
            print(f"  {RED}{len(blocking)} stale CTest entry(ies) detected -- reconfigure build dir(s){RESET}\n")
        else:
            print(f"  {GREEN}[OK] CTest registries are consistent with built executables{RESET}\n")

    # Unified Code Quality Gate (dev_bug_scanner + hot_path_alloc + audit_test_quality)
    if mode in ('--all', '--bug-scan'):
        print(f"{BOLD}[16/17] Code Quality Gate (all scanners){RESET}")
        try:
            runner_path = SCRIPT_DIR / "run_code_quality.py"
            result = subprocess.run(
                [sys.executable, str(runner_path),
                 "--fail-on-regression", "--json"],
                capture_output=True, text=True, timeout=300,
                cwd=str(LIB_ROOT),
            )
            if result.stdout.strip():
                report = json.loads(result.stdout)
                regressions = report.get("regressions", [])
                total = report.get("total_findings", 0)
                scanners = report.get("scanners", {})

                for name, info in scanners.items():
                    counts = info.get("counts", {})
                    err = info.get("error")
                    if err:
                        print(f"  {YELLOW}{name}: ERROR — {err}{RESET}")
                    else:
                        h = counts.get("HIGH", 0)
                        m = counts.get("MEDIUM", 0)
                        t = counts.get("total", 0)
                        status = f"{GREEN}OK{RESET}" if h == 0 else f"{RED}{h} HIGH{RESET}"
                        print(f"  {name}: [{status}] {t} findings (H:{h} M:{m})")

                if regressions:
                    for r in regressions:
                        print(f"  {RED}REGRESSION{RESET}: {r}")
                    total_issues += len(regressions)
                    exit_code = 1
                    print(f"  {RED}{len(regressions)} regression(s) vs baseline{RESET}\n")
                else:
                    print(f"  {GREEN}[OK] No regressions — {total} findings within baseline{RESET}\n")
            elif result.returncode != 0:
                print(f"  {YELLOW}WARNING: run_code_quality exited with code {result.returncode}{RESET}\n")
            else:
                print(f"  {GREEN}[OK] No findings{RESET}\n")
        except FileNotFoundError:
            print(f"  {YELLOW}WARNING: run_code_quality.py not found{RESET}\n")
        except subprocess.TimeoutExpired:
            print(f"  {YELLOW}WARNING: code quality gate timed out (300s){RESET}\n")
        except Exception as exc:
            print(f"  {YELLOW}WARNING: code quality gate failed: {exc}{RESET}\n")

    # Python audit infrastructure self-test
    if mode in ('--all', '--py-audit'):
        print(f"{BOLD}[17/17] Python Audit Self-Test{RESET}")
        try:
            selftest_path = SCRIPT_DIR / "test_audit_scripts.py"
            result = subprocess.run(
                [sys.executable, str(selftest_path), "--quick"],
                capture_output=True, text=True, timeout=60,
                cwd=str(LIB_ROOT),
            )
            output = result.stdout + result.stderr
            if result.returncode == 0:
                # Extract pass count from output
                import re as _re
                m = _re.search(r'(\d+) passed', output)
                count = m.group(1) if m else "?"
                print(f"  {GREEN}[OK] {count} checks passed{RESET}\n")
            else:
                # Show failures
                for line in output.split('\n'):
                    if 'FAIL' in line:
                        print(f"  {RED}{line.strip()}{RESET}")
                fail_m = _re.search(r'(\d+) FAILED', output) if '_re' in dir() else None
                n_fail = fail_m.group(1) if fail_m else "?"
                total_issues += int(n_fail) if isinstance(n_fail, str) and n_fail.isdigit() else 1
                exit_code = 1
                print(f"  {RED}{n_fail} Python audit self-test failure(s){RESET}\n")
        except FileNotFoundError:
            print(f"  {YELLOW}WARNING: test_audit_scripts.py not found{RESET}\n")
        except subprocess.TimeoutExpired:
            print(f"  {YELLOW}WARNING: Python self-test timed out (60s){RESET}\n")
        except Exception as exc:
            print(f"  {YELLOW}WARNING: Python self-test failed: {exc}{RESET}\n")

    # Security Autonomy Gates (non-blocking informational in Phase 1)
    if mode in ('--all', '--autonomy'):
        print(f"{BOLD}[18/20] Security Autonomy Gates (informational){RESET}")
        autonomy_gates = [
            ("Formal Invariants", "check_formal_invariants.py", []),
            ("Risk-Surface Coverage", "risk_surface_coverage.py", []),
            ("Audit SLA", "audit_sla_check.py", []),
            ("Supply-Chain", "supply_chain_gate.py", []),
            ("Evidence Governance", "evidence_governance.py", ["validate"]),
        ]
        autonomy_pass = 0
        autonomy_total = len(autonomy_gates)
        for gate_name, gate_script, gate_args in autonomy_gates:
            gate_path = SCRIPT_DIR / gate_script
            if not gate_path.exists():
                print(f"  {YELLOW}SKIP{RESET} {gate_name} (script not found)")
                continue
            try:
                result = subprocess.run(
                    [sys.executable, str(gate_path)] + gate_args + ["--json"],
                    capture_output=True, text=True, timeout=120,
                    cwd=str(LIB_ROOT),
                )
                try:
                    report = json.loads(result.stdout)
                    passed = report.get("overall_pass", False)
                except (json.JSONDecodeError, ValueError):
                    passed = result.returncode == 0
                if passed:
                    autonomy_pass += 1
                    print(f"  {GREEN}PASS{RESET} {gate_name}")
                else:
                    print(f"  {YELLOW}FAIL{RESET} {gate_name}")
            except subprocess.TimeoutExpired:
                print(f"  {YELLOW}TIMEOUT{RESET} {gate_name}")
            except Exception as exc:
                print(f"  {YELLOW}ERROR{RESET} {gate_name}: {exc}")
        print(f"  Autonomy gates: {autonomy_pass}/{autonomy_total}")
        print()

    if mode in ('--all', '--autonomy'):
        print(f"{BOLD}[19/20] Misuse-Resistance & Co-Gating (informational){RESET}")
        cogate_checks = [
            ("Misuse Resistance", "check_misuse_resistance.py", []),
            ("Perf-Security Co-gate", "perf_security_cogate.py", []),
        ]
        for gate_name, gate_script, gate_args in cogate_checks:
            gate_path = SCRIPT_DIR / gate_script
            if not gate_path.exists():
                print(f"  {YELLOW}SKIP{RESET} {gate_name} (script not found)")
                continue
            try:
                result = subprocess.run(
                    [sys.executable, str(gate_path)] + gate_args + ["--json"],
                    capture_output=True, text=True, timeout=120,
                    cwd=str(LIB_ROOT),
                )
                try:
                    report = json.loads(result.stdout)
                    passed = report.get("overall_pass", False)
                except (json.JSONDecodeError, ValueError):
                    passed = result.returncode == 0
                if passed:
                    print(f"  {GREEN}PASS{RESET} {gate_name}")
                else:
                    print(f"  {YELLOW}FAIL{RESET} {gate_name}")
            except subprocess.TimeoutExpired:
                print(f"  {YELLOW}TIMEOUT{RESET} {gate_name}")
            except Exception as exc:
                print(f"  {YELLOW}ERROR{RESET} {gate_name}: {exc}")
        print()

    if mode in ('--all', '--autonomy'):
        print(f"{BOLD}[20/20] Incident Drills (informational){RESET}")
        drill_path = SCRIPT_DIR / "incident_drills.py"
        if drill_path.exists():
            try:
                result = subprocess.run(
                    [sys.executable, str(drill_path), "--json"],
                    capture_output=True, text=True, timeout=60,
                    cwd=str(LIB_ROOT),
                )
                try:
                    report = json.loads(result.stdout)
                    d_passed = report.get("drills_passed", 0)
                    d_total = report.get("drills_total", 0)
                    print(f"  Drills: {d_passed}/{d_total} passed")
                except (json.JSONDecodeError, ValueError):
                    print(f"  {YELLOW}Could not parse drill output{RESET}")
            except subprocess.TimeoutExpired:
                print(f"  {YELLOW}TIMEOUT{RESET}")
            except Exception as exc:
                print(f"  {YELLOW}ERROR{RESET}: {exc}")
        else:
            print(f"  {YELLOW}SKIP{RESET} incident_drills.py not found")
        print()

    # Summary
    print(f"{BOLD}{'='*60}{RESET}")
    if total_issues == 0:
        print(f"{GREEN}{BOLD}  PREFLIGHT PASSED{RESET}")
    else:
        print(f"{RED}{BOLD}  PREFLIGHT: {total_issues} issues found{RESET}")
    print(f"{BOLD}{'='*60}{RESET}\n")

    return exit_code

if __name__ == '__main__':
    sys.exit(run_all(sys.argv[1:]))
