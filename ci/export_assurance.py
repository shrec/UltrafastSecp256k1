#!/usr/bin/env python3
"""
export_assurance.py  --  Machine-readable assurance status export

Queries the SQLite project graph and produces a JSON report covering:
  - Feature maturity by subsystem
  - API coverage (unit/fuzz/adversarial/CT/GPU/vectors)
  - Test target inventory with categories
  - Security pattern density per file
  - Protocol coverage status
  - ABI routing summary (CT vs fast)

Usage:
    python3 ci/export_assurance.py                     # stdout JSON
    python3 ci/export_assurance.py -o assurance.json   # write to file
"""

import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
LIB_ROOT = SCRIPT_DIR.parent
DB_PATH = LIB_ROOT / ".project_graph.db"

try:
    from report_provenance import collect_provenance
except ImportError:
    sys.path.insert(0, str(SCRIPT_DIR))
    from report_provenance import collect_provenance


def get_conn():
    if not DB_PATH.exists():
        print(f"ERROR: Graph DB not found at {DB_PATH}", file=sys.stderr)
        sys.exit(1)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def export_subsystem_summary(conn):
    """Per-subsystem file/line counts and layers."""
    rows = conn.execute("""
        SELECT subsystem, layer, COUNT(*) as files, SUM(lines) as total_lines
        FROM source_files
        WHERE subsystem IS NOT NULL AND subsystem != ''
        GROUP BY subsystem, layer
        ORDER BY total_lines DESC
    """).fetchall()
    result = {}
    for r in rows:
        ss = r['subsystem']
        if ss not in result:
            result[ss] = []
        result[ss].append({
            'layer': r['layer'],
            'files': r['files'],
            'lines': r['total_lines'],
        })
    return result


def export_api_coverage(conn):
    """ABI functions with routing and test coverage from function_test_map."""
    fns = conn.execute("""
        SELECT a.name, a.category, a.layer,
               r.internal_call, r.layer as route_layer
        FROM c_abi_functions a
        LEFT JOIN abi_routing r ON r.abi_function = a.name
        ORDER BY a.category, a.name
    """).fetchall()

    # Pre-fetch all test mappings in one query
    ftm_rows = conn.execute("""
        SELECT function_name, test_target, coverage_type
        FROM function_test_map
        ORDER BY function_name, test_target
    """).fetchall()
    ftm = {}
    for r in ftm_rows:
        fn = r['function_name']
        if fn not in ftm:
            ftm[fn] = []
        ftm[fn].append({
            'test': r['test_target'],
            'type': r['coverage_type'],
        })

    result = []
    for f in fns:
        name = f['name']
        coverage = ftm.get(name, [])

        result.append({
            'function': name,
            'category': f['category'],
            'layer': f['layer'],
            'route_layer': f['route_layer'],
            'internal_call': f['internal_call'],
            'test_coverage': coverage,
        })
    return result


def export_test_targets(conn):
    """All CTest targets with categories and timeouts."""
    rows = conn.execute("""
        SELECT name, category, timeout, labels
        FROM test_targets
        ORDER BY category, name
    """).fetchall()
    return [dict(r) for r in rows]


def export_security_density(conn):
    """Security pattern counts per file."""
    rows = conn.execute("""
        SELECT source_file, pattern, COUNT(*) as count
        FROM security_patterns
        GROUP BY source_file, pattern
        ORDER BY source_file
    """).fetchall()
    result = {}
    for r in rows:
        sf = r['source_file']
        if sf not in result:
            result[sf] = {}
        result[sf][r['pattern']] = r['count']
    return result


def export_protocol_status(conn):
    """Protocol-related subsystems with their file counts and test edges."""
    protocols = ['musig2', 'frost', 'adaptor', 'silent_payments', 'ecies', 'dleq']
    result = {}
    for proto in protocols:
        files = conn.execute("""
            SELECT path, lines, layer FROM source_files
            WHERE subsystem = ? OR path LIKE ?
            ORDER BY layer, path
        """, (proto, f'%{proto}%')).fetchall()
        tests = conn.execute("""
            SELECT name, category FROM test_targets
            WHERE name LIKE ? OR category LIKE ?
        """, (f'%{proto}%', f'%{proto}%')).fetchall()
        result[proto] = {
            'files': [{'path': f['path'], 'lines': f['lines'], 'layer': f['layer']} for f in files],
            'tests': [{'name': t['name'], 'category': t['category']} for t in tests],
        }
    return result


def export_routing_summary(conn):
    """Summary of CT vs fast ABI routing."""
    rows = conn.execute("""
        SELECT layer, COUNT(*) as count
        FROM abi_routing
        GROUP BY layer
    """).fetchall()
    return {r['layer']: r['count'] for r in rows}


def export_semantic_tags(conn):
    """Semantic tag inventory and densest entities."""
    tags = conn.execute("""
        SELECT st.tag, st.domain, st.description, COUNT(et.id) AS entities
        FROM semantic_tags st
        LEFT JOIN entity_tags et ON et.tag = st.tag
        GROUP BY st.tag, st.domain, st.description
        ORDER BY entities DESC, st.tag
    """).fetchall()
    top_entities = conn.execute("""
        SELECT entity_type, entity_id, COUNT(*) AS tag_count,
               GROUP_CONCAT(tag, ', ') AS tags
        FROM entity_tags
        GROUP BY entity_type, entity_id
        ORDER BY tag_count DESC, entity_type, entity_id
        LIMIT 25
    """).fetchall()
    return {
        'inventory': [dict(r) for r in tags],
        'top_entities': [dict(r) for r in top_entities],
    }


def export_symbol_reasoning(conn):
    """Reasoning-oriented symbol inventory for optimization and audit workflows."""
    try:
        conn.execute("SELECT 1 FROM v_symbol_reasoning LIMIT 1")
    except Exception:
        print("WARNING: v_symbol_reasoning view missing (old graph schema?); symbol reasoning unavailable.", file=sys.stderr)
        return {'summary': [], 'optimization_candidates': [], 'risk_hotspots': []}
    summary = conn.execute("""
        SELECT category, backend, COUNT(*) AS symbols,
               AVG(risk_score) AS avg_risk,
               AVG(gain_score) AS avg_gain
        FROM v_symbol_reasoning
        GROUP BY category, backend
        ORDER BY symbols DESC, category, backend
    """).fetchall()
    optimize = conn.execute("""
        SELECT symbol_name, file_path, category, backend, risk_score, gain_score, optimization_priority
        FROM v_symbol_reasoning
        ORDER BY optimization_priority DESC, gain_score DESC
        LIMIT 25
    """).fetchall()
    risk = conn.execute("""
        SELECT symbol_name, file_path, category, secret_class, risk_score, gain_score
        FROM v_symbol_reasoning
        ORDER BY risk_score DESC, gain_score DESC
        LIMIT 25
    """).fetchall()
    return {
        'summary': [dict(r) for r in summary],
        'optimization_candidates': [dict(r) for r in optimize],
        'risk_hotspots': [dict(r) for r in risk],
    }


def export_graph_meta(conn):
    """Graph metadata."""
    meta = {}
    rows = conn.execute("SELECT key, value FROM meta").fetchall()
    for r in rows:
        meta[r['key']] = r['value']
    # Table counts
    tables = conn.execute("""
        SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'
    """).fetchall()
    counts = {}
    for t in tables:
        cnt = conn.execute(f"SELECT COUNT(*) as c FROM [{t['name']}]").fetchone()
        counts[t['name']] = cnt['c']
    meta['table_counts'] = counts
    return meta


def main():
    out_file = None
    if '-o' in sys.argv:
        idx = sys.argv.index('-o')
        if idx + 1 < len(sys.argv):
            out_file = sys.argv[idx + 1]

    conn = get_conn()

    try:
        provenance = collect_provenance()

        report = {
            'schema_version': '1.0.0',
            'runner': 'export_assurance',
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'provenance': provenance,
            'graph_meta': export_graph_meta(conn),
            'subsystem_summary': export_subsystem_summary(conn),
            'api_coverage': export_api_coverage(conn),
            'test_targets': export_test_targets(conn),
            'security_density': export_security_density(conn),
            'protocol_status': export_protocol_status(conn),
            'routing_summary': export_routing_summary(conn),
            'semantic_tags': export_semantic_tags(conn),
            'symbol_reasoning': export_symbol_reasoning(conn),
        }
    except Exception as e:
        print(f"ERROR: export_assurance failed during data collection: {e}", file=sys.stderr)
        conn.close()
        sys.exit(1)

    conn.close()

    output = json.dumps(report, indent=2, ensure_ascii=False)
    if out_file:
        Path(out_file).write_text(output)
        print(f"Assurance report written to {out_file}", file=sys.stderr)
    else:
        print(output)
    return 0


if __name__ == '__main__':
    # H-5 fix: propagate return value to process exit code.
    # Previously main() had no return statement (implicitly None) and the
    # __main__ block did not call sys.exit(), so partial/failed exports
    # always exited 0.
    sys.exit(main())
