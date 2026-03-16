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
    python3 scripts/preflight.py --drift            # narrative drift only
    python3 scripts/preflight.py --coverage         # coverage gaps only
    python3 scripts/preflight.py --freshness        # graph freshness only
    python3 scripts/preflight.py --changed          # check git-changed files
    python3 scripts/preflight.py --abi              # ABI surface check
"""

import sqlite3
import os
import re
import sys
import subprocess
from pathlib import Path
from datetime import datetime, timezone

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

def get_conn():
    if not DB_PATH.exists():
        print(f"{RED}ERROR: Graph DB not found at {DB_PATH}{RESET}")
        print(f"Run: python3 {SCRIPT_DIR}/build_project_graph.py --rebuild")
        sys.exit(1)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn

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
    'include/ufsecp/ufsecp.h':         ['docs/API_REFERENCE.md', 'docs/USER_GUIDE.md'],
    'include/ufsecp/ufsecp_impl.cpp':  ['docs/API_REFERENCE.md'],
    # Build system
    'CMakeLists.txt':                  ['docs/BUILDING.md', 'README.md'],
    # Benchmark
    'cpu/bench/bench_unified.cpp':     ['docs/BENCHMARKS.md', 'docs/BENCHMARK_METHODOLOGY.md'],
    # Audit
    'audit/unified_audit_runner.cpp':  ['docs/TEST_MATRIX.md', 'docs/AUDIT_GUIDE.md'],
    # Protocol implementations
    'cpu/src/musig2.cpp':              ['docs/API_REFERENCE.md'],
    'cpu/src/frost.cpp':               ['docs/API_REFERENCE.md'],
    'cpu/src/adaptor.cpp':             ['docs/API_REFERENCE.md'],
    'cpu/src/silent_payments.cpp':     ['docs/API_REFERENCE.md'],
    'cpu/src/ecies.cpp':               ['docs/API_REFERENCE.md'],
    # CT layer
    'cpu/src/ct_sign.cpp':             ['docs/CT_VERIFICATION.md', 'docs/SECURITY_CLAIMS.md'],
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
    ct_changed = [f for f in changed_files if 'ct_' in f or '/ct/' in f]
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
    for hdr_name in ('ufsecp.h', 'ufsecp_version.h'):
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

    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}  UltrafastSecp256k1 Preflight Check{RESET}")
    print(f"{BOLD}{'='*60}{RESET}\n")

    # Security
    if mode in ('--all', '--security'):
        print(f"{BOLD}[1/6] Security Invariants{RESET}")
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

    # Narrative drift
    if mode in ('--all', '--drift'):
        print(f"{BOLD}[2/6] Narrative Drift Detection{RESET}")
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
        print(f"{BOLD}[3/6] Test Coverage Gaps{RESET}")
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
        print(f"{BOLD}[4/6] Graph Freshness{RESET}")
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
    if mode in ('--all', '--changed'):
        print(f"{BOLD}[5/6] Changed Files Impact{RESET}")
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

    # ABI
    if mode in ('--all', '--abi'):
        print(f"{BOLD}[6/6] ABI Surface{RESET}")
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
