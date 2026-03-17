#!/usr/bin/env python3
"""
query_graph.py  --  Query the UltrafastSecp256k1 Project Knowledge Graph

Lightweight CLI for AI agents and developers to query the SQLite project graph.

Usage Examples:
    python3 scripts/query_graph.py search "schnorr sign"
    python3 scripts/query_graph.py file cpu/src/ecdsa.cpp
    python3 scripts/query_graph.py subsystem ecdsa
    python3 scripts/query_graph.py deps cpu/src/musig2.cpp
    python3 scripts/query_graph.py rdeps secp256k1/schnorr.hpp
    python3 scripts/query_graph.py abi ecdsa
    python3 scripts/query_graph.py test ethereum
    python3 scripts/query_graph.py layer ct
    python3 scripts/query_graph.py function ufsecp_ecdsa_sign
    python3 scripts/query_graph.py audit protocol
    python3 scripts/query_graph.py platform x86_64
    python3 scripts/query_graph.py methods FieldElement
    python3 scripts/query_graph.py security ct_sign.cpp
    python3 scripts/query_graph.py routing ecdsa_sign
    python3 scripts/query_graph.py bindings rust
    python3 scripts/query_graph.py macros size_constant
    python3 scripts/query_graph.py impact cpu/src/ct_sign.cpp
    python3 scripts/query_graph.py context cpu/src/ecdsa.cpp
    python3 scripts/query_graph.py gaps
    python3 scripts/query_graph.py summary
    python3 scripts/query_graph.py sql "SELECT * FROM error_codes"
"""

import sqlite3
import sys
import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
LIB_ROOT = SCRIPT_DIR.parent
DB_PATH = LIB_ROOT / ".project_graph.db"

def get_conn():
    if not DB_PATH.exists():
        print(f"ERROR: Database not found at {DB_PATH}")
        print(f"Run: python3 {SCRIPT_DIR}/build_project_graph.py")
        sys.exit(1)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def parse_limit_arg(arg: str, default: int = 15):
    try:
        return max(1, int(arg.strip()))
    except Exception:
        return default


def find_symbol_rows(conn, name: str):
    exact = conn.execute(
        """SELECT * FROM symbol_metadata
           WHERE symbol_name = ? OR symbol_name = ?
           ORDER BY review_priority DESC, file_path, start_line""",
        (name, name.split('::')[-1])
    ).fetchall()
    if exact:
        return exact
    return conn.execute(
        """SELECT * FROM symbol_metadata
           WHERE symbol_name LIKE ? OR file_path LIKE ? OR summary LIKE ?
           ORDER BY review_priority DESC, file_path, start_line""",
        (f'%{name}%', f'%{name}%', f'%{name}%')
    ).fetchall()

def cmd_search(query: str):
    """Full-text search across files, functions, docs, methods, and routing."""
    conn = get_conn()
    # Convert space-separated words to OR for FTS5
    fts_query = ' OR '.join(query.split())
    print(f"=== Search: {query} ===\n")
    
    # Files
    rows = conn.execute("SELECT path, category, subsystem, layer FROM fts_files WHERE fts_files MATCH ?", (fts_query,)).fetchall()
    if rows:
        print(f"FILES ({len(rows)}):")
        for r in rows:
            print(f"  [{r['layer'] or '?':4s}] {r['path']}  ({r['category']}, {r['subsystem'] or '-'})")
    
    # Functions
    rows = conn.execute("SELECT name, category, layer, signature FROM fts_functions WHERE fts_functions MATCH ?", (fts_query,)).fetchall()
    if rows:
        print(f"\nC ABI FUNCTIONS ({len(rows)}):")
        for r in rows:
            print(f"  [{r['layer']:4s}] {r['name']}  ({r['category']})")
    
    # C++ Methods
    try:
        rows = conn.execute("SELECT class_name, method, signature, layer FROM fts_methods WHERE fts_methods MATCH ?", (fts_query,)).fetchall()
        if rows:
            print(f"\nC++ METHODS ({len(rows)}):")
            for r in rows:
                cls = r['class_name'] or '(free)'
                print(f"  [{r['layer']:4s}] {cls}::{r['method']}")
    except Exception:
        pass

    # ABI Routing
    try:
        rows = conn.execute("SELECT abi_function, internal_call, layer FROM fts_routing WHERE fts_routing MATCH ?", (fts_query,)).fetchall()
        if rows:
            print(f"\nABI ROUTING ({len(rows)}):")
            for r in rows:
                print(f"  [{r['layer']:4s}] {r['abi_function']} -> {r['internal_call']}")
    except Exception:
        pass
    
    # Docs
    rows = conn.execute("SELECT path, title, category FROM fts_docs WHERE fts_docs MATCH ?", (fts_query,)).fetchall()
    if rows:
        print(f"\nDOCS ({len(rows)}):")
        for r in rows:
            print(f"  {r['path']}  ({r['category']})")
    
    conn.close()

def cmd_file(path: str):
    """Show everything known about a file."""
    conn = get_conn()
    r = conn.execute("SELECT * FROM source_files WHERE path LIKE ?", (f'%{path}%',)).fetchone()
    if not r:
        print(f"File not found: {path}")
        return
    
    print(f"FILE: {r['path']}")
    print(f"  Category:  {r['category']}")
    print(f"  Subsystem: {r['subsystem'] or '-'}")
    print(f"  Layer:     {r['layer']}")
    print(f"  Lines:     {r['lines']}")
    print(f"  Type:      {r['file_type']}")
    
    # Dependencies
    deps = conn.execute("SELECT included_file FROM include_deps WHERE source_file=?", (r['path'],)).fetchall()
    if deps:
        print(f"\n  INCLUDES ({len(deps)}):")
        for d in deps:
            print(f"    {d['included_file']}")
    
    # Reverse deps
    rdeps = conn.execute("SELECT source_file FROM include_deps WHERE included_file LIKE ?", (f'%{Path(r["path"]).name}%',)).fetchall()
    if rdeps:
        print(f"\n  INCLUDED BY ({len(rdeps)}):")
        for d in rdeps:
            print(f"    {d['source_file']}")
    
    # Tests covering this file
    tests = conn.execute("""SELECT src_id FROM edges 
        WHERE dst_type='source_file' AND dst_id=? AND relation='covers'""", (r['path'],)).fetchall()
    if tests:
        print(f"\n  TESTED BY ({len(tests)}):")
        for t in tests:
            print(f"    {t['src_id']}")
    
    # ABI functions implementing through this file
    funcs = conn.execute("""SELECT src_id FROM edges 
        WHERE dst_type='source_file' AND dst_id=? AND relation='implements'""", (r['path'],)).fetchall()
    if funcs:
        print(f"\n  ABI FUNCTIONS ({len(funcs)}):")
        for f in funcs:
            print(f"    {f['src_id']}")
    
    conn.close()

def cmd_subsystem(name: str):
    """List all files and functions in a subsystem."""
    conn = get_conn()
    rows = conn.execute("SELECT path, category, layer, lines FROM source_files WHERE subsystem=? ORDER BY lines DESC", (name,)).fetchall()
    print(f"=== Subsystem: {name} ({len(rows)} files) ===\n")
    total_lines = 0
    for r in rows:
        print(f"  [{r['layer']:4s}] {r['path']:60s} {r['lines']:5d} lines")
        total_lines += r['lines']
    print(f"\n  Total: {total_lines} lines")
    
    funcs = conn.execute("SELECT name, layer FROM c_abi_functions WHERE category=?", (name,)).fetchall()
    if funcs:
        print(f"\n  C ABI FUNCTIONS ({len(funcs)}):")
        for f in funcs:
            print(f"    [{f['layer']:4s}] {f['name']}")
    conn.close()

def cmd_deps(path: str):
    """Show include dependencies for a source file."""
    conn = get_conn()
    rows = conn.execute("SELECT included_file, is_local FROM include_deps WHERE source_file LIKE ? ORDER BY is_local DESC, included_file",
                        (f'%{path}%',)).fetchall()
    print(f"=== Dependencies of {path} ({len(rows)}) ===\n")
    for r in rows:
        kind = 'local' if r['is_local'] else 'system'
        print(f"  [{kind:6s}] {r['included_file']}")
    conn.close()

def cmd_rdeps(header: str):
    """Show reverse dependencies (who includes this header)."""
    conn = get_conn()
    rows = conn.execute("SELECT source_file FROM include_deps WHERE included_file LIKE ? ORDER BY source_file",
                        (f'%{header}%',)).fetchall()
    print(f"=== Reverse deps of {header} ({len(rows)} files include it) ===\n")
    for r in rows:
        print(f"  {r['source_file']}")
    conn.close()

def cmd_abi(category: str = None):
    """List C ABI functions, optionally filtered by category."""
    conn = get_conn()
    if category:
        rows = conn.execute("SELECT name, category, layer, line_no FROM c_abi_functions WHERE category=? ORDER BY line_no", (category,)).fetchall()
    else:
        rows = conn.execute("SELECT name, category, layer, line_no FROM c_abi_functions ORDER BY category, line_no").fetchall()
    print(f"=== C ABI Functions ({len(rows)}) ===\n")
    for r in rows:
        print(f"  [{r['layer']:4s}] {r['name']:50s} ({r['category']}, L{r['line_no']})")
    conn.close()

def cmd_test(filter_str: str = None):
    """List test targets, optionally filtered."""
    conn = get_conn()
    if filter_str:
        rows = conn.execute("""SELECT name, category, timeout, labels FROM test_targets 
            WHERE name LIKE ? OR category LIKE ? OR labels LIKE ? ORDER BY name""",
            (f'%{filter_str}%', f'%{filter_str}%', f'%{filter_str}%')).fetchall()
    else:
        rows = conn.execute("SELECT name, category, timeout, labels FROM test_targets ORDER BY category, name").fetchall()
    print(f"=== Test Targets ({len(rows)}) ===\n")
    for r in rows:
        print(f"  {r['name']:40s} [{r['category']:20s}] timeout={r['timeout']}s  {r['labels']}")
    conn.close()

def cmd_layer(layer: str):
    """List all files in a specific layer."""
    conn = get_conn()
    rows = conn.execute("SELECT path, category, subsystem, lines FROM source_files WHERE layer=? ORDER BY lines DESC",
                        (layer,)).fetchall()
    total = sum(r['lines'] for r in rows)
    print(f"=== Layer: {layer} ({len(rows)} files, {total} lines) ===\n")
    for r in rows[:30]:
        print(f"  {r['path']:60s} {r['lines']:5d}  ({r['category']}, {r['subsystem'] or '-'})")
    if len(rows) > 30:
        print(f"  ... and {len(rows) - 30} more files")
    conn.close()

def cmd_function(name: str):
    """Show details of a specific C ABI function."""
    conn = get_conn()
    r = conn.execute("SELECT * FROM c_abi_functions WHERE name LIKE ?", (f'%{name}%',)).fetchone()
    if not r:
        print(f"Function not found: {name}")
        return
    print(f"FUNCTION: {r['name']}")
    print(f"  Category:  {r['category']}")
    print(f"  Layer:     {r['layer']}")
    print(f"  Line:      {r['line_no']}")
    print(f"  Signature: {r['signature']}")
    
    # Implementation file
    impl = conn.execute("""SELECT dst_id FROM edges 
        WHERE src_type='c_abi_function' AND src_id=? AND relation='implements'""", (r['name'],)).fetchone()
    if impl:
        print(f"  Impl:      {impl['dst_id']}")
    conn.close()

def cmd_audit(section: str = None):
    """List audit modules, optionally filtered by section."""
    conn = get_conn()
    if section:
        rows = conn.execute("SELECT * FROM audit_modules WHERE section LIKE ? ORDER BY section_no, module_id",
                            (f'%{section}%',)).fetchall()
    else:
        rows = conn.execute("SELECT * FROM audit_modules ORDER BY section_no, module_id").fetchall()
    print(f"=== Audit Modules ({len(rows)}) ===\n")
    cur_section = None
    for r in rows:
        if r['section'] != cur_section:
            cur_section = r['section']
            print(f"\n  [{r['section_no']}] {cur_section.upper()}:")
        print(f"    {r['module_id']:25s} {r['name']}")
    conn.close()

def cmd_platform(platform: str):
    """Show platform-specific dispatch info."""
    conn = get_conn()
    rows = conn.execute("SELECT * FROM platform_dispatch WHERE platform LIKE ? ORDER BY source_file",
                        (f'%{platform}%',)).fetchall()
    print(f"=== Platform: {platform} ({len(rows)} dispatch points) ===\n")
    for r in rows:
        print(f"  {r['source_file']:40s} [{r['mechanism']:12s}] {r['description']}")
    conn.close()

def cmd_summary():
    """Show database summary statistics."""
    conn = get_conn()
    stats = json.loads(conn.execute("SELECT value FROM meta WHERE key='stats'").fetchone()['value'])
    built = conn.execute("SELECT value FROM meta WHERE key='built_at'").fetchone()['value']
    version = conn.execute("SELECT value FROM meta WHERE key='version'").fetchone()['value']
    schema_version = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()['value']
    extractor_version = conn.execute("SELECT value FROM meta WHERE key='extractor_version'").fetchone()['value']
    revision = conn.execute("SELECT value FROM meta WHERE key='graph_build_revision'").fetchone()['value']
    
    print(f"=== Project Knowledge Graph v{version} ===")
    print(f"  Built: {built}")
    print(f"  Schema: {schema_version}  Extractor: {extractor_version}  Revision: {revision}")
    print(f"  Database: {DB_PATH} ({DB_PATH.stat().st_size / 1024:.0f} KB)\n")
    
    total = sum(stats.values())
    print(f"TABLES ({total} total records):")
    for table, count in sorted(stats.items()):
        print(f"  {table:25s} {count:5d}")
    
    print(f"\nLAYER SUMMARY:")
    for r in conn.execute("SELECT * FROM v_layer_summary").fetchall():
        print(f"  {r['layer']:5s} {r['file_count']:4d} files  {r['total_lines']:6d} lines")
    
    print(f"\nTOP 10 SUBSYSTEMS:")
    for r in conn.execute("SELECT * FROM v_subsystem_files LIMIT 10").fetchall():
        print(f"  {r['subsystem']:15s} {r['file_count']:3d} files  {r['total_lines']:6d} lines")
    if 'analysis_scores' in stats:
        top = conn.execute("SELECT COUNT(*) AS cnt FROM analysis_scores WHERE overall_priority >= 30").fetchone()['cnt']
        tasks = conn.execute("SELECT COUNT(*) AS cnt FROM ai_tasks WHERE status='pending'").fetchone()['cnt']
        print(f"\nANALYSIS: {top} high-priority symbols, {tasks} pending AI tasks")
    conn.close()

def cmd_sql(query: str):
    """Execute raw SQL query."""
    conn = get_conn()
    try:
        rows = conn.execute(query).fetchall()
        if rows:
            # Print header
            cols = rows[0].keys()
            print('|'.join(cols))
            print('-' * 80)
            for r in rows:
                print('|'.join(str(r[c]) for c in cols))
        else:
            print("(no results)")
    except Exception as e:
        print(f"SQL ERROR: {e}")
    conn.close()

def cmd_methods(class_name: str = None):
    """List C++ methods, optionally filtered by class."""
    conn = get_conn()
    if class_name:
        rows = conn.execute("""SELECT class_name, method, signature, layer, header_path, line_no
            FROM cpp_methods WHERE class_name LIKE ? OR method LIKE ?
            ORDER BY class_name, line_no""",
            (f'%{class_name}%', f'%{class_name}%')).fetchall()
    else:
        rows = conn.execute("""SELECT class_name, method, signature, layer, header_path, line_no
            FROM cpp_methods ORDER BY class_name, line_no""").fetchall()
    print(f"=== C++ Methods ({len(rows)}) ===\n")
    cur_cls = None
    for r in rows:
        cls = r['class_name'] or '(free)'
        if cls != cur_cls:
            cur_cls = cls
            print(f"\n  {cur_cls}:")
        print(f"    [{r['layer']:4s}] {r['method']:30s} {r['signature']}")
    conn.close()

def cmd_security(file_filter: str = None):
    """Show security-critical patterns (secure_erase, value_barrier, CLASSIFY)."""
    conn = get_conn()
    if file_filter:
        rows = conn.execute("""SELECT pattern, source_file, line_no, context
            FROM security_patterns WHERE source_file LIKE ?
            ORDER BY source_file, line_no""",
            (f'%{file_filter}%',)).fetchall()
    else:
        rows = conn.execute("""SELECT source_file, pattern, COUNT(*) as cnt,
            GROUP_CONCAT(line_no) as lines
            FROM security_patterns GROUP BY source_file, pattern
            ORDER BY source_file""").fetchall()
    if file_filter:
        print(f"=== Security Patterns in *{file_filter}* ({len(rows)}) ===\n")
        for r in rows:
            print(f"  L{r['line_no']:4d} [{r['pattern']:15s}] {r['context'][:80]}")
    else:
        print(f"=== Security Pattern Hotspots ({len(rows)} file-pattern groups) ===\n")
        cur_file = None
        for r in rows:
            if r['source_file'] != cur_file:
                cur_file = r['source_file']
                print(f"\n  {cur_file}:")
            print(f"    {r['pattern']:15s} x{r['cnt']:2d}  lines: {r['lines']}")
    conn.close()

def cmd_routing(fn_filter: str = None):
    """Show ABI routing: which ufsecp_* maps to CT vs fast."""
    conn = get_conn()
    if fn_filter:
        rows = conn.execute("""SELECT abi_function, internal_call, layer, impl_line
            FROM abi_routing WHERE abi_function LIKE ? OR internal_call LIKE ?
            ORDER BY impl_line""",
            (f'%{fn_filter}%', f'%{fn_filter}%')).fetchall()
    else:
        rows = conn.execute("""SELECT abi_function, internal_call, layer, impl_line
            FROM abi_routing ORDER BY impl_line""").fetchall()
    print(f"=== ABI Routing ({len(rows)}) ===\n")
    for r in rows:
        line_str = f"L{r['impl_line']}" if r['impl_line'] else '   ?'
        print(f"  [{r['layer']:4s}] {r['abi_function']:45s} -> {r['internal_call']:35s} ({line_str})")
    conn.close()

def cmd_bindings(lang: str = None):
    """Show binding language info."""
    conn = get_conn()
    if lang:
        rows = conn.execute("SELECT * FROM binding_languages WHERE language LIKE ?",
            (f'%{lang}%',)).fetchall()
    else:
        rows = conn.execute("SELECT * FROM binding_languages ORDER BY status, language").fetchall()
    print(f"=== Binding Languages ({len(rows)}) ===\n")
    for r in rows:
        print(f"  {r['language']:15s} [{r['status']:12s}] {r['directory']:30s} {r['file_count']:3d} files  FFI: {r['ffi_method']:12s} pkg: {r['package_name']}")
    conn.close()

def cmd_macros(category: str = None):
    """Show compile-time macros and defines."""
    conn = get_conn()
    if category:
        rows = conn.execute("SELECT * FROM macros WHERE category LIKE ? ORDER BY name",
            (f'%{category}%',)).fetchall()
    else:
        rows = conn.execute("SELECT * FROM macros ORDER BY category, name").fetchall()
    print(f"=== Macros ({len(rows)}) ===\n")
    cur_cat = None
    for r in rows:
        if r['category'] != cur_cat:
            cur_cat = r['category']
            print(f"\n  [{cur_cat}]:")
        val = f"= {r['value']}" if r['value'] else ''
        print(f"    {r['name']:40s} {val:20s} ({r['file_path']}:{r['line_no']})")
    conn.close()

def cmd_impact(path: str):
    """Show full impact analysis for a file: deps, rdeps, tests, ABI, security."""
    conn = get_conn()
    # Find the file
    r = conn.execute("SELECT * FROM source_files WHERE path LIKE ?", (f'%{path}%',)).fetchone()
    if not r:
        print(f"File not found: {path}")
        return
    fpath = r['path']
    print(f"=== IMPACT ANALYSIS: {fpath} ===")
    print(f"  Layer: {r['layer']}, Subsystem: {r['subsystem']}, Lines: {r['lines']}\n")

    # Direct includes
    deps = conn.execute("SELECT included_file FROM include_deps WHERE source_file=?", (fpath,)).fetchall()
    print(f"  DEPENDS ON ({len(deps)}):")
    for d in deps:
        print(f"    {d['included_file']}")

    # Reverse deps
    fname = Path(fpath).name
    rdeps = conn.execute("SELECT source_file FROM include_deps WHERE included_file LIKE ?",
                         (f'%{fname}%',)).fetchall()
    print(f"\n  DEPENDED ON BY ({len(rdeps)}):")
    for d in rdeps:
        print(f"    {d['source_file']}")

    # Tests
    tests = conn.execute("""SELECT src_id FROM edges 
        WHERE dst_type='source_file' AND dst_id=? AND relation='covers'""", (fpath,)).fetchall()
    print(f"\n  TESTED BY ({len(tests)}):")
    for t in tests:
        print(f"    {t['src_id']}")

    # ABI functions
    funcs = conn.execute("""SELECT src_id FROM edges 
        WHERE dst_type='source_file' AND dst_id=? AND relation='implements'""", (fpath,)).fetchall()
    print(f"\n  ABI FUNCTIONS ({len(funcs)}):")
    for f in funcs:
        print(f"    {f['src_id']}")

    # Security patterns
    secs = conn.execute("SELECT pattern, line_no FROM security_patterns WHERE source_file=? ORDER BY line_no",
                        (fpath,)).fetchall()
    if secs:
        print(f"\n  SECURITY PATTERNS ({len(secs)}):")
        for s in secs:
            print(f"    L{s['line_no']:4d} {s['pattern']}")

    # ABI routing through this file
    routings = conn.execute("""SELECT abi_function, layer FROM abi_routing 
        WHERE internal_call LIKE ? OR abi_function LIKE ?""",
        (f'%{fname.replace(".cpp","").replace(".hpp","")}%',
         f'%{fname.replace(".cpp","").replace(".hpp","")}%')).fetchall()
    if routings:
        print(f"\n  ABI ROUTING ({len(routings)}):")
        for rt in routings:
            print(f"    [{rt['layer']:4s}] {rt['abi_function']}")

    conn.close()

def cmd_gaps():
    """Show test coverage gaps -- core files with no test coverage."""
    conn = get_conn()
    rows = conn.execute("SELECT * FROM v_coverage_gaps").fetchall()
    total_lines = sum(r['lines'] for r in rows)
    print(f"=== Test Coverage Gaps ({len(rows)} untested core files, {total_lines} total lines) ===\n")
    for r in rows:
        sec = f"  SEC:{r['security_patterns']}" if r['security_patterns'] else ''
        abi = f"  ABI:{r['abi_functions']}" if r['abi_functions'] else ''
        print(f"  [{r['layer']:4s}] {r['path']:50s} {r['lines']:5d} lines  ({r['subsystem'] or '-'}){sec}{abi}")
    conn.close()

def cmd_context(path: str):
    """One-shot context dump: summary + deps + rdeps + tests + security + routing + functions.
    Replaces 5-6 separate queries. Designed for maximum token efficiency."""
    conn = get_conn()
    # Find file
    r = conn.execute("SELECT * FROM source_files WHERE path LIKE ?", (f'%{path}%',)).fetchone()
    if not r:
        print(f"File not found: {path}")
        conn.close()
        return
    fpath = r['path']

    # Summary
    summary = conn.execute("SELECT summary FROM file_summaries WHERE path=?", (fpath,)).fetchone()
    desc = summary['summary'] if summary else '(no summary)'
    print(f"FILE: {fpath}  [{r['layer']}] {r['lines']} lines")
    print(f"  {desc}")
    print(f"  category={r['category']}  subsystem={r['subsystem'] or '-'}")

    # Dependencies (compact)
    deps = conn.execute("SELECT included_file FROM include_deps WHERE source_file=?", (fpath,)).fetchall()
    if deps:
        print(f"\nINCLUDES ({len(deps)}): {', '.join(d['included_file'] for d in deps)}")

    # Reverse deps (compact)
    fname = Path(fpath).name
    rdeps = conn.execute("SELECT source_file FROM include_deps WHERE included_file LIKE ?",
                         (f'%{fname}%',)).fetchall()
    if rdeps:
        print(f"\nINCLUDED BY ({len(rdeps)}): {', '.join(d['source_file'] for d in rdeps)}")

    # Tests
    tests = conn.execute("""SELECT src_id FROM edges 
        WHERE dst_type='source_file' AND dst_id=? AND relation='covers'""", (fpath,)).fetchall()
    if tests:
        print(f"\nTESTS ({len(tests)}): {', '.join(t['src_id'] for t in tests)}")
    else:
        print(f"\nTESTS: NONE (coverage gap!)")

    # ABI functions
    funcs = conn.execute("""SELECT src_id FROM edges 
        WHERE dst_type='source_file' AND dst_id=? AND relation='implements'""", (fpath,)).fetchall()
    if funcs:
        print(f"\nABI ({len(funcs)}): {', '.join(f['src_id'] for f in funcs)}")

    # Security patterns (compact counts)
    secs = conn.execute("""SELECT pattern, COUNT(*) as cnt FROM security_patterns 
        WHERE source_file=? GROUP BY pattern""", (fpath,)).fetchall()
    if secs:
        sec_str = ', '.join(f"{s['pattern']}:{s['cnt']}" for s in secs)
        print(f"\nSECURITY: {sec_str}")

    # ABI routing
    base = fname.replace('.cpp', '').replace('.hpp', '')
    routings = conn.execute("""SELECT abi_function, internal_call, layer FROM abi_routing 
        WHERE internal_call LIKE ? OR abi_function LIKE ?""",
        (f'%{base}%', f'%{base}%')).fetchall()
    if routings:
        print(f"\nROUTING ({len(routings)}):")
        for rt in routings:
            print(f"  [{rt['layer']:4s}] {rt['abi_function']} -> {rt['internal_call']}")

    # Function index (key section for token savings)
    fidx = conn.execute("""SELECT name, start_line, end_line, kind, class_name 
        FROM function_index WHERE file_path=? ORDER BY start_line""", (fpath,)).fetchall()
    if fidx:
        print(f"\nFUNCTIONS ({len(fidx)}):")
        for f in fidx:
            cls = f"{f['class_name']}::" if f['class_name'] else ''
            span = f['end_line'] - f['start_line'] + 1
            print(f"  L{f['start_line']:4d}-{f['end_line']:4d} ({span:3d}) {cls}{f['name']}")

    conn.close()

def cmd_preflight(mode: str = None):
    """Run preflight quality checks (delegates to scripts/preflight.py)."""
    import subprocess
    script = Path(__file__).resolve().parent / 'preflight.py'
    args = ['python3', str(script)]
    if mode:
        args.append(mode)
    subprocess.run(args)


def cmd_semantic(name: str):
    """Metadata-first symbol summary without raw code."""
    conn = get_conn()
    rows = find_symbol_rows(conn, name)
    if not rows:
        print(f"Symbol not found: {name}")
        conn.close()
        return
    row = rows[0]
    tags = json.loads(row['semantic_tags']) if row['semantic_tags'] else []
    print(f"SYMBOL: {row['symbol_name']}")
    print(f"  file: {row['file_path']}:{row['start_line']}-{row['end_line']}")
    print(f"  class: {row['class_name'] or '-'}")
    print(f"  risk: {row['risk_level']}  priority={row['review_priority']}")
    print(f"  hot_path={row['hot_path']}  ct_sensitive={row['ct_sensitive']}  batchable={row['batchable']}  gpu_candidate={row['gpu_candidate']}")
    print(f"  summary: {row['summary']}")
    print(f"  tags: {', '.join(tags) if tags else '-'}")
    call_ctx = conn.execute(
        "SELECT callers, callees, caller_count, callee_count FROM v_symbol_call_context WHERE symbol_name=? AND file_path=?",
        (row['symbol_name'], row['file_path'])
    ).fetchone()
    if call_ctx:
        print(f"  callers({call_ctx['caller_count']}): {call_ctx['callers'] or '-'}")
        print(f"  callees({call_ctx['callee_count']}): {call_ctx['callees'] or '-'}")
    conn.close()


def cmd_slice(name: str):
    """Signature + exact line slice + call context for token-efficient retrieval."""
    conn = get_conn()
    rows = find_symbol_rows(conn, name)
    if not rows:
        print(f"Symbol not found: {name}")
        conn.close()
        return
    row = rows[0]
    span = row['end_line'] - row['start_line'] + 1
    print(f"SLICE: {row['symbol_name']}")
    print(f"  file: {row['file_path']}")
    print(f"  lines: {row['start_line']}-{row['end_line']} ({span} lines)")
    print(f"  signature: {row['signature'] or '(signature unavailable)'}")
    print(f"  summary: {row['summary']}")
    call_ctx = conn.execute(
        "SELECT callers, callees FROM v_symbol_call_context WHERE symbol_name=? AND file_path=?",
        (row['symbol_name'], row['file_path'])
    ).fetchone()
    if call_ctx:
        print(f"  callers: {call_ctx['callers'] or '-'}")
        print(f"  callees: {call_ctx['callees'] or '-'}")
    print("  retrieval_hint: read only the slice lines first; open dependent symbols only if needed")
    conn.close()


def cmd_hot(limit_arg: str = None):
    """Top hot-path symbols for performance-focused reasoning."""
    conn = get_conn()
    limit = parse_limit_arg(limit_arg or '15')
    rows = conn.execute(
        """SELECT symbol_name, file_path, summary, review_priority, risk_level
           FROM symbol_metadata
           WHERE hot_path=1
           ORDER BY review_priority DESC, file_path, start_line
           LIMIT ?""",
        (limit,)
    ).fetchall()
    print(f"=== Hot Symbols ({len(rows)}) ===\n")
    for row in rows:
        print(f"  {row['symbol_name']:30s} [{row['risk_level']:6s}] p={row['review_priority']:2d}  {row['file_path']}")
        print(f"    {row['summary']}")
    conn.close()


def cmd_risk(limit_arg: str = None):
    """High-risk symbols first, with semantic context."""
    conn = get_conn()
    limit = parse_limit_arg(limit_arg or '15')
    rows = conn.execute(
        """SELECT symbol_name, file_path, summary, risk_level, review_priority, ct_sensitive
           FROM symbol_metadata
           ORDER BY CASE risk_level WHEN 'high' THEN 0 WHEN 'medium' THEN 1 ELSE 2 END,
                    review_priority DESC, file_path, start_line
           LIMIT ?""",
        (limit,)
    ).fetchall()
    print(f"=== Risk Queue ({len(rows)}) ===\n")
    for row in rows:
        ct = ' ct' if row['ct_sensitive'] else ''
        print(f"  {row['symbol_name']:30s} [{row['risk_level']}{ct}] p={row['review_priority']:2d}  {row['file_path']}")
        print(f"    {row['summary']}")
    conn.close()


def cmd_candidates(limit_arg: str = None):
    """Optimization candidates: hot + batchable/GPU-friendly symbols."""
    conn = get_conn()
    limit = parse_limit_arg(limit_arg or '15')
    rows = conn.execute(
        """SELECT symbol_name, file_path, summary, risk_level, review_priority,
                  batchable, gpu_candidate, ct_sensitive
           FROM v_optimization_candidates
           LIMIT ?""",
        (limit,)
    ).fetchall()
    print(f"=== Optimization Candidates ({len(rows)}) ===\n")
    for row in rows:
        flags = []
        if row['batchable']:
            flags.append('batchable')
        if row['gpu_candidate']:
            flags.append('gpu')
        if row['ct_sensitive']:
            flags.append('ct')
        flag_str = ', '.join(flags) if flags else '-'
        print(f"  {row['symbol_name']:30s} [{row['risk_level']:6s}] p={row['review_priority']:2d}  {flag_str}")
        print(f"    {row['file_path']} :: {row['summary']}")
    conn.close()


def cmd_bottlenecks(limit_arg: str = None):
    """Top proactive bottleneck candidates ranked from derived analysis."""
    conn = get_conn()
    limit = parse_limit_arg(limit_arg or '15')
    rows = conn.execute(
        """SELECT symbol_name, file_path, start_line, hotness_score, complexity_score, fanin_score,
                  gpu_score, ct_risk_score, audit_gap_score, overall_priority, summary, reasons
           FROM v_bottleneck_queue
           LIMIT ?""",
        (limit,)
    ).fetchall()
    print(f"=== Bottleneck Queue ({len(rows)}) ===\n")
    for row in rows:
        reasons = json.loads(row['reasons']) if row['reasons'] else []
        print(
            f"  {row['symbol_name']:30s} p={row['overall_priority']:3d} hot={row['hotness_score']:2d} "
            f"cx={row['complexity_score']:2d} in={row['fanin_score']:2d} gpu={row['gpu_score']:2d} "
            f"ct={row['ct_risk_score']:2d} gap={row['audit_gap_score']:2d}"
        )
        print(f"    {row['file_path']}:{row['start_line']} :: {row['summary']}")
        print(f"    reasons: {', '.join(reasons) if reasons else '-'}")
    conn.close()


def cmd_tasks(limit_arg: str = None):
    """Show the prebuilt AI task queue derived from graph analysis."""
    conn = get_conn()
    limit = parse_limit_arg(limit_arg or '15')
    rows = conn.execute(
          """SELECT task_type, symbol_name, file_path, start_line, priority, status, reasons
           FROM v_ai_task_queue
           LIMIT ?""",
        (limit,)
    ).fetchall()
    print(f"=== AI Task Queue ({len(rows)}) ===\n")
    for row in rows:
        reasons = json.loads(row['reasons']) if row['reasons'] else []
        print(f"  {row['task_type']:12s} {row['symbol_name']:28s} p={row['priority']:3d} [{row['status']}]")
        print(f"    {row['file_path']}:{row['start_line']}")
        print(f"    reasons: {', '.join(reasons) if reasons else '-'}")
    conn.close()


def cmd_score(name: str):
    """Show full derived analysis scores for a symbol."""
    conn = get_conn()
    rows = find_symbol_rows(conn, name)
    if not rows:
        print(f"Symbol not found: {name}")
        conn.close()
        return
    row = rows[0]
    score = conn.execute(
        """SELECT * FROM analysis_scores WHERE symbol_name=? AND file_path=? AND start_line=?""",
        (row['symbol_name'], row['file_path'], row['start_line'])
    ).fetchone()
    if not score:
        print(f"No analysis scores found for: {row['symbol_name']}")
        conn.close()
        return
    reasons = json.loads(score['reasons']) if score['reasons'] else []
    print(f"SCORE: {row['symbol_name']}")
    print(f"  file: {row['file_path']}:{row['start_line']}-{row['end_line']}")
    print(f"  hotness={score['hotness_score']} complexity={score['complexity_score']} fanin={score['fanin_score']} fanout={score['fanout_score']}")
    print(f"  optimization={score['optimization_score']} gpu={score['gpu_score']} ct_risk={score['ct_risk_score']} audit_gap={score['audit_gap_score']}")
    print(f"  perf_priority={score['perf_priority']} safe_priority={score['safe_priority']} overall={score['overall_priority']}")
    print(f"  reasons: {', '.join(reasons) if reasons else '-'}")
    tasks = conn.execute(
        """SELECT task_type, priority, status FROM ai_tasks WHERE symbol_name=? AND file_path=? AND start_line=? ORDER BY priority DESC""",
        (row['symbol_name'], row['file_path'], row['start_line'])
    ).fetchall()
    if tasks:
        print("  tasks:")
        for task in tasks:
            print(f"    {task['task_type']} p={task['priority']} [{task['status']}]")
    conn.close()

COMMANDS = {
    'search': ('search <query>', cmd_search),
    'file': ('file <path>', cmd_file),
    'subsystem': ('subsystem <name>', cmd_subsystem),
    'deps': ('deps <source_file>', cmd_deps),
    'rdeps': ('rdeps <header>', cmd_rdeps),
    'abi': ('abi [category]', cmd_abi),
    'test': ('test [filter]', cmd_test),
    'layer': ('layer <name>', cmd_layer),
    'function': ('function <name>', cmd_function),
    'audit': ('audit [section]', cmd_audit),
    'platform': ('platform <name>', cmd_platform),
    'summary': ('summary', cmd_summary),
    'sql': ('sql "<query>"', cmd_sql),
    'methods': ('methods [class]', cmd_methods),
    'security': ('security [file]', cmd_security),
    'routing': ('routing [function]', cmd_routing),
    'bindings': ('bindings [language]', cmd_bindings),
    'macros': ('macros [category]', cmd_macros),
    'impact': ('impact <file>', cmd_impact),
    'gaps': ('gaps', cmd_gaps),
    'context': ('context <file>', cmd_context),
    'slice': ('slice <symbol>', cmd_slice),
    'semantic': ('semantic <symbol>', cmd_semantic),
    'hot': ('hot [limit]', cmd_hot),
    'risk': ('risk [limit]', cmd_risk),
    'candidates': ('candidates [limit]', cmd_candidates),
    'bottlenecks': ('bottlenecks [limit]', cmd_bottlenecks),
    'tasks': ('tasks [limit]', cmd_tasks),
    'score': ('score <symbol>', cmd_score),
    'preflight': ('preflight [--security|--coverage|--abi]', cmd_preflight),
}

if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print("Usage: query_graph.py <command> [args]\n")
        print("Commands:")
        for name, (usage, _) in sorted(COMMANDS.items()):
            print(f"  {usage}")
        sys.exit(1)
    
    cmd_name = sys.argv[1]
    args = ' '.join(sys.argv[2:]) if len(sys.argv) > 2 else None
    handler = COMMANDS[cmd_name][1]

    optional_arg_cmds = {'summary', 'gaps', 'hot', 'risk', 'candidates', 'bottlenecks', 'tasks'}

    if args:
        handler(args)
    elif cmd_name in optional_arg_cmds:
        handler()
    else:
        handler()
