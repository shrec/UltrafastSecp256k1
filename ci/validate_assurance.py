#!/usr/bin/env python3
"""
validate_assurance.py  --  Cross-reference assurance docs vs actual code

Checks:
  1. FEATURE_ASSURANCE_LEDGER.md lists all ufsecp_* functions from ufsecp.h
  2. TEST_MATRIX.md test count matches actual CTest targets in CMakeLists files
  3. ABI functions in graph match header declarations
  4. Conditional compilation blocks (Ethereum) are annotated

Usage:
    python3 ci/validate_assurance.py           # all checks
    python3 ci/validate_assurance.py --json    # JSON output for CI
"""

import json
import re
import sqlite3
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
LIB_ROOT = SCRIPT_DIR.parent
SOURCE_GRAPH_DB = LIB_ROOT / 'tools' / 'source_graph_kit' / 'source_graph.db'
SOURCE_GRAPH_TOOL = LIB_ROOT / 'tools' / 'source_graph_kit' / 'source_graph.py'

RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BOLD = '\033[1m'
RESET = '\033[0m'


def scan_header_functions():
    """Extract all ufsecp_* function names from public headers."""
    headers = [
        LIB_ROOT / 'include' / 'ufsecp' / 'ufsecp.h',
        LIB_ROOT / 'include' / 'ufsecp' / 'ufsecp_gpu.h',
        LIB_ROOT / 'include' / 'ufsecp' / 'ufsecp_version.h',
    ]
    api_re = re.compile(r'UFSECP_API\s+.*?(ufsecp_\w+)\s*\(')
    all_fns = set()
    conditional_fns = set()
    for header in headers:
        if not header.exists():
            continue
        in_conditional = False
        with open(header, 'r', errors='replace') as f:
            for line in f:
                stripped = line.strip()
                if stripped.startswith('#ifdef') or stripped.startswith('#if '):
                    in_conditional = True
                elif stripped.startswith('#endif'):
                    in_conditional = False
                m = api_re.search(line)
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
                        name = m.group(1)
                        # Skip unresolved CMake variables like ${_harness}
                        if '${' in name:
                            continue
                        targets.add(name)
        except Exception:
            continue
    return targets


def scan_test_matrix_targets():
    """Extract test file/target references from TEST_MATRIX.md."""
    matrix = LIB_ROOT / 'docs' / 'TEST_MATRIX.md'
    if not matrix.exists():
        return set()
    targets = set()
    # Match backtick-wrapped filenames (with optional path prefix): `audit_field.cpp`, `src/metal/tests/test_metal_host.cpp`
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
            # Capture bare identifiers from table rows (|) and list items (-)
            stripped = line.strip()
            if '|' in line or stripped.startswith('-'):
                for m in target_re.finditer(line):
                    name = m.group(1)
                    # Skip if it looks like a source file (already handled above)
                    if '.' in name:
                        continue
                    targets.add(name)
    return targets


def scan_assurance_ledger_ids():
    """Extract claim IDs from ASSURANCE_LEDGER.md table rows."""
    ledger = LIB_ROOT / 'docs' / 'ASSURANCE_LEDGER.md'
    if not ledger.exists():
        return set()
    claim_re = re.compile(r'^\|\s*(A-\d{3})\s*\|')
    claim_ids = set()
    with open(ledger, 'r', errors='replace') as f:
        for line in f:
            m = claim_re.search(line)
            if m:
                claim_ids.add(m.group(1))
    return claim_ids


def _load_claims_json():
    claims_path = LIB_ROOT / 'docs' / 'ASSURANCE_CLAIMS.json'
    if not claims_path.exists():
        raise FileNotFoundError(f'missing file: {claims_path}')
    return json.loads(claims_path.read_text(encoding='utf-8'))


def _path_like_surface(value: str) -> bool:
    if not value:
        return False
    if value.endswith('/') or '*' in value or '?' in value:
        return True
    return '/' in value or '.' in value


def _graph_candidate_paths(fs_path: Path):
    try:
        rel = fs_path.resolve().relative_to(LIB_ROOT.resolve())
    except ValueError:
        return []
    rel_text = str(rel).replace('\\', '/')
    candidates = [rel_text]
    parts = rel.parts
    if len(parts) == 1:
        return candidates
    graph_root_dirs = {
        'cpu', 'include', 'audit', 'benchmarks', 'cuda', 'examples',
        'gpu', 'metal', 'opencl', 'tests', 'docs', 'scripts', 'tools',
        'bindings', '.github',
        # src/ is the new canonical prefix (src/cpu, src/cuda, etc.)
        'src',
    }
    if parts[0] in graph_root_dirs and len(parts) > 1:
        candidates.append(str(Path(*parts[1:])).replace('\\', '/'))
    # Handle src/<backend>/... → strip src/ AND src/<backend>/ for graph lookup
    # e.g. src/cpu/src/ct_sign.cpp → cpu/src/ct_sign.cpp AND src/ct_sign.cpp
    if parts[0] == 'src' and len(parts) > 2:
        # strip just 'src/' prefix
        candidates.append(str(Path(*parts[1:])).replace('\\', '/'))
        # strip 'src/<backend>/' prefix (project-relative, what DB stores)
        if len(parts) > 2:
            candidates.append(str(Path(*parts[2:])).replace('\\', '/'))
    return list(dict.fromkeys(candidates))


def _expand_surface(surface: str):
    rel = surface.strip()
    base = LIB_ROOT / rel.rstrip('/')
    if '*' in rel or '?' in rel:
        return sorted(path for path in LIB_ROOT.glob(rel) if path.is_file())
    if base.is_file():
        return [base]
    if base.is_dir():
        return sorted(path for path in base.rglob('*') if path.is_file())
    return []


def ensure_source_graph_db():
    if SOURCE_GRAPH_DB.exists():
        return None
    if not SOURCE_GRAPH_TOOL.exists():
        return f"source graph DB not found at {SOURCE_GRAPH_DB}"
    try:
        subprocess.run(
            ['python3', str(SOURCE_GRAPH_TOOL), 'build', '-i'],
            cwd=str(LIB_ROOT),
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or '').strip()
        if detail:
            return f"failed to build source graph DB via {SOURCE_GRAPH_TOOL}: {detail}"
        return f"failed to build source graph DB via {SOURCE_GRAPH_TOOL}"
    if not SOURCE_GRAPH_DB.exists():
        return f"source graph DB not found at {SOURCE_GRAPH_DB} after build"
    return None


def check_claim_surface_indexing():
    """Check that path-like claim surfaces resolve to real indexed graph files."""
    issues = []
    checked = []

    ensure_error = ensure_source_graph_db()
    if ensure_error:
        issues.append(f"  {RED}MISSING{RESET} {ensure_error}")
        return {
            'checked_surfaces': [],
            'missing_surfaces': ['source_graph.db'],
            'unindexed_surfaces': [],
            'issues': issues,
        }

    data = _load_claims_json()
    claims = data.get('claims', []) if isinstance(data, dict) else []
    con = sqlite3.connect(str(SOURCE_GRAPH_DB))
    con.row_factory = sqlite3.Row

    missing_surfaces = []
    unindexed_surfaces = []

    for claim in claims:
        claim_id = claim.get('claim_id', 'unknown')
        for field_name in ('primary_evidence', 'owner_surface'):
            values = claim.get(field_name, [])
            if not isinstance(values, list):
                continue
            for surface in values:
                if not isinstance(surface, str) or not _path_like_surface(surface):
                    continue
                matched_files = _expand_surface(surface)
                label = f'{claim_id}:{field_name}:{surface}'
                if not matched_files:
                    missing_surfaces.append(label)
                    issues.append(f"  {RED}MISSING{RESET} {claim_id} {field_name} surface '{surface}' does not resolve to a repository file")
                    continue
                graph_paths = []
                for path in matched_files:
                    graph_paths.extend(_graph_candidate_paths(path))
                graph_paths = list(dict.fromkeys(graph_paths))
                indexed_count = 0
                if graph_paths:
                    placeholders = ','.join('?' for _ in graph_paths)
                    indexed_rows = con.execute(
                        f"SELECT path FROM files WHERE path IN ({placeholders})",
                        graph_paths,
                    ).fetchall()
                    indexed_count = len(indexed_rows)
                checked.append({
                    'claim_id': claim_id,
                    'field': field_name,
                    'surface': surface,
                    'matched_files': len(matched_files),
                    'graph_paths': len(graph_paths),
                    'indexed_files': indexed_count,
                })
                if graph_paths and indexed_count == 0:
                    unindexed_surfaces.append(label)
                    issues.append(f"  {YELLOW}UNINDEXED{RESET} {claim_id} {field_name} surface '{surface}' resolves on disk but not in source_graph.db")

    con.close()
    return {
        'checked_surfaces': checked,
        'missing_surfaces': missing_surfaces,
        'unindexed_surfaces': unindexed_surfaces,
        'issues': issues,
    }


def check_assurance_claims_companion():
    """Check that ASSURANCE_CLAIMS.json matches ASSURANCE_LEDGER.md claim IDs."""
    ledger_ids = scan_assurance_ledger_ids()
    claims_path = LIB_ROOT / 'docs' / 'ASSURANCE_CLAIMS.json'

    if not claims_path.exists():
        return {
            'ledger_ids': sorted(ledger_ids),
            'json_ids': [],
            'missing_in_json': sorted(ledger_ids),
            'extra_in_json': [],
            'invalid_entries': ['missing_file'],
            'issues': [f"  {RED}MISSING{RESET} docs/ASSURANCE_CLAIMS.json not found"],
        }

    try:
        data = json.loads(claims_path.read_text(encoding='utf-8'))
    except Exception as exc:
        return {
            'ledger_ids': sorted(ledger_ids),
            'json_ids': [],
            'missing_in_json': sorted(ledger_ids),
            'extra_in_json': [],
            'invalid_entries': [f'invalid_json:{exc}'],
            'issues': [f"  {RED}INVALID{RESET} docs/ASSURANCE_CLAIMS.json could not be parsed: {exc}"],
        }

    claims = data.get('claims')
    if not isinstance(claims, list):
        return {
            'ledger_ids': sorted(ledger_ids),
            'json_ids': [],
            'missing_in_json': sorted(ledger_ids),
            'extra_in_json': [],
            'invalid_entries': ['claims_not_list'],
            'issues': [f"  {RED}INVALID{RESET} docs/ASSURANCE_CLAIMS.json must contain a top-level 'claims' list"],
        }

    required_fields = {
        'claim_id', 'area', 'claim', 'scope', 'primary_evidence',
        'enforcing_workflows', 'artifacts', 'verification_cadence',
        'current_status', 'stale_risk', 'owner_surface'
    }
    json_ids = []
    invalid_entries = []
    issues = []

    for idx, claim in enumerate(claims):
        if not isinstance(claim, dict):
            invalid_entries.append(f'entry_{idx}_not_object')
            issues.append(f"  {RED}INVALID{RESET} ASSURANCE_CLAIMS entry {idx} is not an object")
            continue
        missing_fields = sorted(required_fields - set(claim.keys()))
        claim_id = claim.get('claim_id', f'entry_{idx}')
        if missing_fields:
            invalid_entries.append(f'{claim_id}:missing_fields')
            issues.append(f"  {RED}INVALID{RESET} {claim_id} missing fields: {', '.join(missing_fields)}")
        if claim.get('claim_id'):
            json_ids.append(claim['claim_id'])

    json_id_set = set(json_ids)
    missing_in_json = sorted(ledger_ids - json_id_set)
    extra_in_json = sorted(json_id_set - ledger_ids)

    for claim_id in missing_in_json:
        issues.append(f"  {YELLOW}MISSING{RESET} {claim_id} present in ASSURANCE_LEDGER.md but not in ASSURANCE_CLAIMS.json")
    for claim_id in extra_in_json:
        issues.append(f"  {RED}STALE{RESET}   {claim_id} present in ASSURANCE_CLAIMS.json but not in ASSURANCE_LEDGER.md")

    return {
        'ledger_ids': sorted(ledger_ids),
        'json_ids': sorted(json_id_set),
        'missing_in_json': missing_in_json,
        'extra_in_json': extra_in_json,
        'invalid_entries': invalid_entries,
        'issues': issues,
    }


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


def check_ai_review_events():
    """Validate the machine-readable AI review-event log schema."""
    path = LIB_ROOT / 'docs' / 'AI_REVIEW_EVENTS.json'
    if not path.exists():
        return {
            'event_count': 0,
            'invalid_entries': ['missing_file'],
            'issues': [f"  {RED}MISSING{RESET} docs/AI_REVIEW_EVENTS.json not found"],
        }

    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except Exception as exc:
        return {
            'event_count': 0,
            'invalid_entries': [f'invalid_json:{exc}'],
            'issues': [f"  {RED}INVALID{RESET} docs/AI_REVIEW_EVENTS.json could not be parsed: {exc}"],
        }

    events = payload.get('events')
    if not isinstance(events, list):
        return {
            'event_count': 0,
            'invalid_entries': ['events_not_list'],
            'issues': [f"  {RED}INVALID{RESET} docs/AI_REVIEW_EVENTS.json must contain a top-level 'events' list"],
        }

    allowed_modes = {'auditor', 'attacker', 'bug-bounty', 'performance-skeptic', 'documentation-skeptic'}
    allowed_classes = {'bug', 'security', 'performance', 'docs-drift', 'coverage-gap', 'graph-gap', 'false-positive'}
    allowed_status = {'accepted', 'rejected', 'unconfirmed'}
    required_fields = {
        'event_id', 'reviewed_at', 'review_mode', 'finding_class', 'status',
        'target', 'summary', 'reproduced', 'repository_evidence', 'resulting_changes'
    }

    invalid_entries = []
    issues = []
    for idx, event in enumerate(events):
        if not isinstance(event, dict):
            invalid_entries.append(f'entry_{idx}_not_object')
            issues.append(f"  {RED}INVALID{RESET} AI review event {idx} is not an object")
            continue
        event_id = event.get('event_id', f'entry_{idx}')
        missing = sorted(required_fields - set(event.keys()))
        if missing:
            invalid_entries.append(f'{event_id}:missing_fields')
            issues.append(f"  {RED}INVALID{RESET} {event_id} missing fields: {', '.join(missing)}")
            continue
        if event['review_mode'] not in allowed_modes:
            invalid_entries.append(f'{event_id}:review_mode')
            issues.append(f"  {RED}INVALID{RESET} {event_id} has invalid review_mode '{event['review_mode']}'")
        if event['finding_class'] not in allowed_classes:
            invalid_entries.append(f'{event_id}:finding_class')
            issues.append(f"  {RED}INVALID{RESET} {event_id} has invalid finding_class '{event['finding_class']}'")
        if event['status'] not in allowed_status:
            invalid_entries.append(f'{event_id}:status')
            issues.append(f"  {RED}INVALID{RESET} {event_id} has invalid status '{event['status']}'")
        if not isinstance(event['reproduced'], bool):
            invalid_entries.append(f'{event_id}:reproduced')
            issues.append(f"  {RED}INVALID{RESET} {event_id} reproduced must be boolean")
        for field in ('repository_evidence', 'resulting_changes'):
            if not isinstance(event[field], list):
                invalid_entries.append(f'{event_id}:{field}')
                issues.append(f"  {RED}INVALID{RESET} {event_id} {field} must be a list")
        if event['status'] == 'accepted':
            if not event['repository_evidence']:
                invalid_entries.append(f'{event_id}:accepted_evidence')
                issues.append(f"  {RED}INVALID{RESET} {event_id} is accepted but has no repository_evidence")
            if not event['resulting_changes']:
                invalid_entries.append(f'{event_id}:accepted_changes')
                issues.append(f"  {RED}INVALID{RESET} {event_id} is accepted but has no resulting_changes")

    return {
        'event_count': len(events),
        'invalid_entries': invalid_entries,
        'issues': issues,
    }


def check_gpu_backend_evidence():
    """Validate machine-readable GPU backend evidence and ROCm/HIP promotion rules."""
    path = LIB_ROOT / 'docs' / 'GPU_BACKEND_EVIDENCE.json'
    if not path.exists():
        return {
            'backend_count': 0,
            'invalid_entries': ['missing_file'],
            'issues': [f"  {RED}MISSING{RESET} docs/GPU_BACKEND_EVIDENCE.json not found"],
        }

    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except Exception as exc:
        return {
            'backend_count': 0,
            'invalid_entries': [f'invalid_json:{exc}'],
            'issues': [f"  {RED}INVALID{RESET} docs/GPU_BACKEND_EVIDENCE.json could not be parsed: {exc}"],
        }

    backends = payload.get('backends')
    if not isinstance(backends, list):
        return {
            'backend_count': 0,
            'invalid_entries': ['backends_not_list'],
            'issues': [f"  {RED}INVALID{RESET} docs/GPU_BACKEND_EVIDENCE.json must contain a top-level 'backends' list"],
        }

    required_names = {'cuda', 'opencl', 'metal', 'rocm-hip'}
    allowed_status = {'validated', 'partial', 'planned', 'experimental'}
    required_fields = {
        'backend', 'status', 'hardware_backed', 'publishable', 'device_class',
        'required_artifacts', 'artifact_notes'
    }
    seen = set()
    invalid_entries = []
    issues = []

    for idx, backend in enumerate(backends):
        if not isinstance(backend, dict):
            invalid_entries.append(f'entry_{idx}_not_object')
            issues.append(f"  {RED}INVALID{RESET} GPU backend evidence entry {idx} is not an object")
            continue
        name = backend.get('backend', f'entry_{idx}')
        seen.add(name)
        missing = sorted(required_fields - set(backend.keys()))
        if missing:
            invalid_entries.append(f'{name}:missing_fields')
            issues.append(f"  {RED}INVALID{RESET} {name} missing fields: {', '.join(missing)}")
            continue
        if backend['status'] not in allowed_status:
            invalid_entries.append(f'{name}:status')
            issues.append(f"  {RED}INVALID{RESET} {name} has invalid status '{backend['status']}'")
        for field in ('hardware_backed', 'publishable'):
            if not isinstance(backend[field], bool):
                invalid_entries.append(f'{name}:{field}')
                issues.append(f"  {RED}INVALID{RESET} {name} {field} must be boolean")
        for field in ('required_artifacts', 'artifact_notes'):
            if not isinstance(backend[field], list) or not backend[field]:
                invalid_entries.append(f'{name}:{field}')
                issues.append(f"  {RED}INVALID{RESET} {name} {field} must be a non-empty list")
        if backend['publishable'] and not backend['hardware_backed']:
            invalid_entries.append(f'{name}:publishable_without_hardware')
            issues.append(f"  {RED}INVALID{RESET} {name} cannot be publishable without hardware-backed validation")
        if name == 'rocm-hip':
            if backend['hardware_backed']:
                if backend['status'] == 'planned':
                    invalid_entries.append('rocm-hip:planned_with_hardware')
                    issues.append(f"  {RED}INVALID{RESET} rocm-hip cannot remain 'planned' once hardware-backed")
            else:
                if backend['publishable']:
                    invalid_entries.append('rocm-hip:publishable_without_amd')
                    issues.append(f"  {RED}INVALID{RESET} rocm-hip must remain non-publishable without AMD hardware evidence")
                if backend['status'] not in {'planned', 'experimental'}:
                    invalid_entries.append('rocm-hip:status_without_amd')
                    issues.append(f"  {RED}INVALID{RESET} rocm-hip without AMD hardware must stay planned or experimental")

    missing_backends = sorted(required_names - seen)
    for name in missing_backends:
        invalid_entries.append(f'{name}:missing_backend')
        issues.append(f"  {RED}MISSING{RESET} GPU backend evidence missing required backend '{name}'")

    return {
        'backend_count': len(backends),
        'invalid_entries': invalid_entries,
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
            # H-3 fix: trigger exit_code on ANY issues, not only when
            # extra/missing are truthy. issues can be non-empty without
            # extra/missing being set (different code path).
            exit_code = 1
        else:
            print(f"  {GREEN}[OK] Ledger covers all header functions{RESET}")
        print()

    # 2. Assurance claims JSON companion
    claims = check_assurance_claims_companion()
    results['assurance_claims'] = {
        'ledger_ids': claims['ledger_ids'],
        'json_ids': claims['json_ids'],
        'missing_in_json': claims['missing_in_json'],
        'extra_in_json': claims['extra_in_json'],
        'invalid_entries': claims['invalid_entries'],
    }
    if not json_mode:
        print(f"{BOLD}[2/3] Assurance Claims Companion{RESET}")
        print(f"  Ledger claims: {len(claims['ledger_ids'])}, JSON claims: {len(claims['json_ids'])}")
        if claims['issues']:
            for i in claims['issues']:
                print(i)
            if claims['extra_in_json'] or claims['missing_in_json'] or claims['invalid_entries']:
                exit_code = 1
        else:
            print(f"  {GREEN}[OK] ASSURANCE_CLAIMS.json matches ASSURANCE_LEDGER.md claim IDs{RESET}")
        print()

    # 3. Graph-backed claim surfaces
    claim_surface = check_claim_surface_indexing()
    results['claim_surface'] = {
        'checked_surfaces': claim_surface['checked_surfaces'],
        'missing_surfaces': claim_surface['missing_surfaces'],
        'unindexed_surfaces': claim_surface['unindexed_surfaces'],
    }
    if not json_mode:
        print(f"{BOLD}[3/4] Claim Surface Graph Coverage{RESET}")
        print(f"  Checked surfaces: {len(claim_surface['checked_surfaces'])}")
        if claim_surface['issues']:
            for i in claim_surface['issues']:
                print(i)
            if claim_surface['missing_surfaces'] or claim_surface['unindexed_surfaces']:
                exit_code = 1
        else:
            print(f"  {GREEN}[OK] Path-like claim surfaces resolve and are indexed in source_graph.db{RESET}")
        print()

    # 4. AI review-event log
    ai_review = check_ai_review_events()
    results['ai_review_events'] = {
        'event_count': ai_review['event_count'],
        'invalid_entries': ai_review['invalid_entries'],
    }
    if not json_mode:
        print(f"{BOLD}[4/5] AI Review Event Log{RESET}")
        print(f"  Logged events: {ai_review['event_count']}")
        if ai_review['issues']:
            for i in ai_review['issues']:
                print(i)
            exit_code = 1
        else:
            print(f"  {GREEN}[OK] AI_REVIEW_EVENTS.json is present and schema-valid{RESET}")
        print()

    # 5. GPU backend evidence
    gpu_evidence = check_gpu_backend_evidence()
    results['gpu_backend_evidence'] = {
        'backend_count': gpu_evidence['backend_count'],
        'invalid_entries': gpu_evidence['invalid_entries'],
    }
    if not json_mode:
        print(f"{BOLD}[5/6] GPU Backend Evidence{RESET}")
        print(f"  Tracked backends: {gpu_evidence['backend_count']}")
        if gpu_evidence['issues']:
            for i in gpu_evidence['issues']:
                print(i)
            exit_code = 1
        else:
            print(f"  {GREEN}[OK] GPU_BACKEND_EVIDENCE.json is present and fail-closed for ROCm/HIP promotion{RESET}")
        print()

    # 6. Test matrix
    matrix = check_test_matrix()
    results['test_matrix'] = {
        'actual_targets': matrix['actual_count'],
        'documented_targets': matrix['documented_count'],
        'missing': matrix['missing'],
        'extra': matrix['extra'],
    }
    if not json_mode:
        print(f"{BOLD}[6/6] Test Matrix Accuracy{RESET}")
        print(f"  CTest targets: {matrix['actual_count']}, Documented: {matrix['documented_count']}")
        if matrix['issues']:
            for i in matrix['issues']:
                print(i)
            if matrix['missing']:
                exit_code = 1
        else:
            print(f"  {GREEN}[OK] TEST_MATRIX matches CTest targets{RESET}")
        print()

    # Summary
    total = (
        len(ledger['missing'])
        + len(ledger['extra'])
        + len(claims['missing_in_json'])
        + len(claims['extra_in_json'])
        + len(claims['invalid_entries'])
        + len(claim_surface['missing_surfaces'])
        + len(claim_surface['unindexed_surfaces'])
        + len(ai_review['invalid_entries'])
        + len(gpu_evidence['invalid_entries'])
        + len(matrix['missing'])
        + len(matrix['extra'])
    )
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
