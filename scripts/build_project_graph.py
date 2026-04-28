#!/usr/bin/env python3
"""
build_project_graph.py  --  UltrafastSecp256k1 Project Knowledge Graph Builder

Creates a SQLite database (.project_graph.db) at the library root containing
the full project knowledge graph: source files, C ABI functions, include
dependencies, CMake targets, CI workflows, audit modules, constants, namespaces,
GPU backends, error codes, and rich cross-reference edges.

Usage:
    python3 scripts/build_project_graph.py          # build from library root
    python3 scripts/build_project_graph.py --rebuild # drop & rebuild

The resulting DB is used by AI agents for instant context retrieval.
"""

import sqlite3
import os
import re
import sys
import json
import hashlib
import subprocess
from pathlib import Path
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
LIB_ROOT = SCRIPT_DIR.parent          # libs/UltrafastSecp256k1/
DB_PATH = LIB_ROOT / ".project_graph.db"

SOURCE_EXTS = {'.cpp', '.hpp', '.h', '.cu', '.cuh', '.cl', '.metal', '.mm',
               '.S', '.asm', '.py', '.sh', '.ps1', '.cmake', '.md', '.json',
               '.yml', '.yaml', '.toml', '.txt'}
SKIP_DIRS = {'build-linux', 'build_rel', 'build_opencl', 'build-cuda',
             'build-riscv-rel', '_research_repos', 'node_modules', '.git',
             'build', 'build_bench', 'build_rel'}

# ---------------------------------------------------------------------------
# SCHEMA
# ---------------------------------------------------------------------------
SCHEMA_SQL = """
-- ============================================================
--  UltrafastSecp256k1 Project Knowledge Graph  --  SQLite
-- ============================================================

-- Metadata
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);

-- Source files
CREATE TABLE IF NOT EXISTS source_files (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    path      TEXT UNIQUE NOT NULL,       -- relative to lib root
    category  TEXT,                        -- cpu_core, cpu_header, audit, cuda, opencl, metal, binding, example, script, test, abi, compat
    subsystem TEXT,                        -- field, scalar, point, ecdsa, schnorr, ct, musig2, frost, ethereum, bip32, ...
    file_type TEXT,                        -- cpp, hpp, h, cu, cuh, cl, metal, mm, S, asm
    lines     INTEGER DEFAULT 0,
    sha256    TEXT,
    layer     TEXT                         -- fast, ct, both, abi, gpu, tool
);

-- Namespaces
CREATE TABLE IF NOT EXISTS namespaces (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    name     TEXT UNIQUE NOT NULL,         -- e.g. secp256k1::fast, secp256k1::ct
    purpose  TEXT
);

-- C++ types (classes, structs, enums) in public headers
CREATE TABLE IF NOT EXISTS cpp_types (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    kind        TEXT,                       -- class, struct, enum, using
    namespace   TEXT,
    header_path TEXT,
    description TEXT
);

-- C ABI functions (ufsecp_*)
CREATE TABLE IF NOT EXISTS c_abi_functions (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    name       TEXT UNIQUE NOT NULL,
    category   TEXT,                        -- context, seckey, pubkey, ecdsa, schnorr, ecdh, hash, ...
    signature  TEXT,
    line_no    INTEGER,
    layer      TEXT,                         -- fast, ct, both
    inputs     TEXT,                         -- JSON array of param names
    outputs    TEXT                          -- JSON array of output params
);

-- Include dependencies (source -> header)
CREATE TABLE IF NOT EXISTS include_deps (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    source_file   TEXT NOT NULL,            -- relative path of .cpp
    included_file TEXT NOT NULL,            -- the #included path
    is_local      BOOLEAN DEFAULT 1,       -- 1 = "..." local, 0 = <...> system
    UNIQUE(source_file, included_file)
);

-- CMake targets
CREATE TABLE IF NOT EXISTS cmake_targets (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    name     TEXT UNIQUE NOT NULL,
    type     TEXT,                          -- executable, static_library, shared_library, interface_library, test
    category TEXT,                          -- core, bench, audit, gpu, example, binding, tool
    timeout  INTEGER DEFAULT 0
);

-- CTest test targets
CREATE TABLE IF NOT EXISTS test_targets (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT UNIQUE NOT NULL,
    executable  TEXT,
    category    TEXT,                       -- cpu_core, audit_always, audit_conditional, gpu
    timeout     INTEGER DEFAULT 300,
    labels      TEXT                        -- JSON array of CTest labels
);

-- CI workflows
CREATE TABLE IF NOT EXISTS ci_workflows (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    filename   TEXT UNIQUE NOT NULL,
    name       TEXT,
    triggers   TEXT,                        -- JSON: {push: [...], pull_request: [...], schedule: ...}
    category   TEXT,                        -- merge_blocking, advisory, release
    jobs       TEXT                         -- JSON array of job names
);

-- Audit modules (from unified_audit_runner)
CREATE TABLE IF NOT EXISTS audit_modules (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    module_id  TEXT UNIQUE NOT NULL,
    name       TEXT,
    section    TEXT,                        -- math_invariants, ct_analysis, differential, vectors, fuzzing, protocol, memory, performance
    section_no INTEGER,
    runner     TEXT DEFAULT 'unified_audit_runner'  -- or gpu_audit_runner, opencl_audit_runner, metal_audit_runner
);

-- Project constants
CREATE TABLE IF NOT EXISTS constants (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    name     TEXT NOT NULL,
    value    TEXT NOT NULL,
    category TEXT,                          -- curve, precomp, abi_size, error_code, cmake_option
    context  TEXT                           -- where defined or used
);

-- Error codes
CREATE TABLE IF NOT EXISTS error_codes (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    code    INTEGER UNIQUE NOT NULL,
    name    TEXT NOT NULL,
    symbol  TEXT,                           -- UFSECP_ERR_*
    meaning TEXT
);

-- GPU backends
CREATE TABLE IF NOT EXISTS gpu_backends (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    backend  TEXT UNIQUE NOT NULL,          -- cuda, opencl, metal
    sources  TEXT,                          -- JSON array of source files
    kernels  TEXT,                          -- JSON array of kernel files
    targets  TEXT,                          -- JSON array of CMake targets
    features TEXT                           -- JSON: {ecdsa: true, schnorr: true, ct: true, ...}
);

-- Cross-reference edges (generic relation graph)
CREATE TABLE IF NOT EXISTS edges (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    src_type  TEXT NOT NULL,                -- source_file, c_abi_function, test_target, audit_module, ...
    src_id    TEXT NOT NULL,                -- name or path
    dst_type  TEXT NOT NULL,
    dst_id    TEXT NOT NULL,
    relation  TEXT NOT NULL,               -- includes, tests, implements, depends_on, calls, covers
    weight    REAL DEFAULT 1.0
);

-- Build configurations
CREATE TABLE IF NOT EXISTS build_configs (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    name    TEXT UNIQUE NOT NULL,           -- cmake option name
    default_value TEXT,
    type    TEXT,                           -- BOOL, STRING, INTEGER
    effect  TEXT
);

-- Platform dispatch
CREATE TABLE IF NOT EXISTS platform_dispatch (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    source_file TEXT NOT NULL,
    platform    TEXT NOT NULL,              -- x86_64, arm64, riscv64, esp32, wasm, msvc
    mechanism   TEXT,                       -- ifdef, constexpr_if, asm_file
    description TEXT
);

-- Documentation index
CREATE TABLE IF NOT EXISTS docs (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    path     TEXT UNIQUE NOT NULL,
    title    TEXT,
    category TEXT,                          -- architecture, api, build, security, audit, testing, benchmark, binding, release, optimization
    topics   TEXT                           -- JSON array of topic tags
);

-- Semantic tags for richer project navigation and AI retrieval
CREATE TABLE IF NOT EXISTS semantic_tags (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    tag         TEXT UNIQUE NOT NULL,
    domain      TEXT,
    description TEXT
);

CREATE TABLE IF NOT EXISTS entity_tags (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_type TEXT NOT NULL,              -- source_file, c_abi_function, doc
    entity_id   TEXT NOT NULL,              -- path / symbol / doc path
    tag         TEXT NOT NULL,
    confidence  REAL DEFAULT 1.0,
    origin      TEXT DEFAULT 'derived',
    UNIQUE(entity_type, entity_id, tag)
);

-- C++ public methods (extracted from headers)
CREATE TABLE IF NOT EXISTS cpp_methods (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    class_name TEXT NOT NULL,               -- FieldElement, Scalar, Point, etc.
    method     TEXT NOT NULL,               -- method name
    signature  TEXT,                        -- return_type method(params)
    is_static  BOOLEAN DEFAULT 0,
    is_const   BOOLEAN DEFAULT 0,
    is_noexcept BOOLEAN DEFAULT 0,
    header_path TEXT,
    line_no    INTEGER,
    layer      TEXT                         -- fast, ct
);

-- Security-critical pattern locations
CREATE TABLE IF NOT EXISTS security_patterns (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    pattern    TEXT NOT NULL,               -- secure_erase, value_barrier, CLASSIFY, DECLASSIFY
    source_file TEXT NOT NULL,
    line_no    INTEGER NOT NULL,
    context    TEXT                         -- brief snippet or purpose
);

-- ABI routing map (which internal function each ufsecp_* calls)
CREATE TABLE IF NOT EXISTS abi_routing (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    abi_function TEXT NOT NULL,             -- ufsecp_ecdsa_sign
    internal_call TEXT NOT NULL,            -- ct::ecdsa_sign or fast::ecdsa_verify
    layer        TEXT NOT NULL,             -- ct, fast
    impl_line    INTEGER                   -- line in ufsecp_impl.cpp
);

-- Binding language metadata
CREATE TABLE IF NOT EXISTS binding_languages (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    language     TEXT UNIQUE NOT NULL,
    directory    TEXT NOT NULL,
    file_count   INTEGER DEFAULT 0,
    status       TEXT,                     -- stable, active, supported, community, experimental
    package_name TEXT,                     -- pip name, crate name, pod name, etc.
    ffi_method   TEXT                      -- cffi, ctypes, cgo, JNI, N-API, P/Invoke, FFI
);

-- Compile-time macros and defines
CREATE TABLE IF NOT EXISTS macros (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    name      TEXT NOT NULL,
    value     TEXT,
    file_path TEXT NOT NULL,
    line_no   INTEGER,
    category  TEXT                          -- platform_guard, size_constant, feature_flag, ct_marker, debug
);

-- Per-file one-line summaries so agents never need to open a file just to understand its role
CREATE TABLE IF NOT EXISTS file_summaries (
    path    TEXT PRIMARY KEY,
    summary TEXT NOT NULL                   -- max ~120 chars, enough for instant context
);

-- Function/method line ranges: agents can read_file with exact ranges, no guessing
CREATE TABLE IF NOT EXISTS function_index (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path  TEXT NOT NULL,
    name       TEXT NOT NULL,               -- function/method name
    start_line INTEGER NOT NULL,
    end_line   INTEGER NOT NULL,
    kind       TEXT DEFAULT 'function',     -- function, method, constructor, operator
    class_name TEXT,                        -- NULL for free functions
    UNIQUE(file_path, name, start_line)
);

-- FTS5 full-text search on key tables
CREATE VIRTUAL TABLE IF NOT EXISTS fts_files USING fts5(
    path, category, subsystem, layer, content=source_files, content_rowid=id
);

CREATE VIRTUAL TABLE IF NOT EXISTS fts_functions USING fts5(
    name, category, signature, layer, content=c_abi_functions, content_rowid=id
);

CREATE VIRTUAL TABLE IF NOT EXISTS fts_docs USING fts5(
    path, title, category, topics, content=docs, content_rowid=id
);

CREATE VIRTUAL TABLE IF NOT EXISTS fts_tags USING fts5(
    entity_type, entity_id, tag, domain, description
);

-- Triggers to keep FTS in sync
CREATE TRIGGER IF NOT EXISTS source_files_ai AFTER INSERT ON source_files BEGIN
    INSERT INTO fts_files(rowid, path, category, subsystem, layer) VALUES (new.id, new.path, new.category, new.subsystem, new.layer);
END;

CREATE TRIGGER IF NOT EXISTS c_abi_functions_ai AFTER INSERT ON c_abi_functions BEGIN
    INSERT INTO fts_functions(rowid, name, category, signature, layer) VALUES (new.id, new.name, new.category, new.signature, new.layer);
END;

CREATE TRIGGER IF NOT EXISTS docs_ai AFTER INSERT ON docs BEGIN
    INSERT INTO fts_docs(rowid, path, title, category, topics) VALUES (new.id, new.path, new.title, new.category, new.topics);
END;

-- FTS5 on methods (search by class, method name, signature)
CREATE VIRTUAL TABLE IF NOT EXISTS fts_methods USING fts5(
    class_name, method, signature, layer, content=cpp_methods, content_rowid=id
);
CREATE TRIGGER IF NOT EXISTS cpp_methods_ai AFTER INSERT ON cpp_methods BEGIN
    INSERT INTO fts_methods(rowid, class_name, method, signature, layer) VALUES (new.id, new.class_name, new.method, new.signature, new.layer);
END;

-- FTS5 on ABI routing (search by function name, internal call, layer)
CREATE VIRTUAL TABLE IF NOT EXISTS fts_routing USING fts5(
    abi_function, internal_call, layer, content=abi_routing, content_rowid=id
);
CREATE TRIGGER IF NOT EXISTS abi_routing_ai AFTER INSERT ON abi_routing BEGIN
    INSERT INTO fts_routing(rowid, abi_function, internal_call, layer) VALUES (new.id, new.abi_function, new.internal_call, new.layer);
END;

-- Useful views
CREATE VIEW IF NOT EXISTS v_file_deps AS
  SELECT sf.path AS source, d.included_file AS dependency, d.is_local
  FROM include_deps d
  JOIN source_files sf ON sf.path = d.source_file;

CREATE VIEW IF NOT EXISTS v_subsystem_files AS
  SELECT subsystem, COUNT(*) AS file_count, SUM(lines) AS total_lines
  FROM source_files
  WHERE subsystem IS NOT NULL
  GROUP BY subsystem
  ORDER BY total_lines DESC;

CREATE VIEW IF NOT EXISTS v_layer_summary AS
  SELECT layer, COUNT(*) AS file_count, SUM(lines) AS total_lines
  FROM source_files
  WHERE layer IS NOT NULL
  GROUP BY layer;

CREATE VIEW IF NOT EXISTS v_test_coverage AS
  SELECT t.name AS test, t.category,
         GROUP_CONCAT(e.dst_id) AS covers_files
  FROM test_targets t
  LEFT JOIN edges e ON e.src_type='test_target' AND e.src_id=t.name AND e.relation='covers'
  GROUP BY t.name;

CREATE VIEW IF NOT EXISTS v_abi_routing AS
  SELECT ar.abi_function, ar.internal_call, ar.layer,
         e.dst_id AS impl_file
  FROM abi_routing ar
  LEFT JOIN edges e ON e.src_type='c_abi_function' AND e.src_id=ar.abi_function AND e.relation='implements';

CREATE VIEW IF NOT EXISTS v_security_hotspots AS
  SELECT source_file, pattern, COUNT(*) AS count,
         GROUP_CONCAT(line_no) AS lines
  FROM security_patterns
  GROUP BY source_file, pattern
  ORDER BY count DESC;

CREATE VIEW IF NOT EXISTS v_class_methods AS
  SELECT class_name, layer, COUNT(*) AS method_count,
         GROUP_CONCAT(method, ', ') AS methods
  FROM cpp_methods
  GROUP BY class_name, layer;

CREATE VIEW IF NOT EXISTS v_impact_analysis AS
  SELECT sf.path, sf.subsystem, sf.lines,
         (SELECT COUNT(*) FROM include_deps WHERE source_file=sf.path) AS dep_count,
         (SELECT COUNT(*) FROM include_deps WHERE included_file LIKE '%'||REPLACE(sf.path,'/','%')) AS rdep_count,
         (SELECT COUNT(*) FROM edges WHERE dst_id=sf.path AND relation='covers') AS test_count,
         (SELECT COUNT(*) FROM edges WHERE dst_id=sf.path AND relation='implements') AS abi_func_count,
         (SELECT COUNT(*) FROM security_patterns WHERE source_file=sf.path) AS security_pattern_count
  FROM source_files sf
  WHERE sf.category IN ('cpu_core', 'abi')
  ORDER BY sf.lines DESC;

-- Coverage gaps: core files with no test coverage, sorted by risk (lines DESC)
CREATE VIEW IF NOT EXISTS v_coverage_gaps AS
  SELECT sf.path, sf.subsystem, sf.layer, sf.lines,
         (SELECT COUNT(*) FROM security_patterns WHERE source_file=sf.path) AS security_patterns,
         (SELECT COUNT(*) FROM edges WHERE dst_id=sf.path AND relation='implements') AS abi_functions
  FROM source_files sf
  WHERE sf.category = 'cpu_core'
    AND sf.file_type IN ('cpp', 'source')
    AND sf.lines > 50
    AND NOT EXISTS (
      SELECT 1 FROM edges WHERE dst_id=sf.path AND relation='covers'
    )
  ORDER BY sf.lines DESC;

-- Edge type summary
CREATE VIEW IF NOT EXISTS v_edge_summary AS
  SELECT relation, COUNT(*) AS cnt,
         COUNT(DISTINCT src_id) AS unique_sources,
         COUNT(DISTINCT dst_id) AS unique_targets
  FROM edges GROUP BY relation ORDER BY cnt DESC;

-- ABI security map: which ABI functions touch CT code
CREATE VIEW IF NOT EXISTS v_abi_security AS
  SELECT ar.abi_function, ar.layer, ar.internal_call,
         (SELECT GROUP_CONCAT(DISTINCT sp.pattern)
          FROM security_patterns sp
          JOIN edges e ON e.dst_id = sp.source_file
          WHERE e.src_id = ar.abi_function AND e.relation = 'routes_through'
         ) AS security_patterns_in_impl
  FROM abi_routing ar
  WHERE ar.layer = 'ct'
  ORDER BY ar.abi_function;

-- ---------------------------------------------------------------------------
-- PHASE 4 ADDITIONS: call graph, config bindings, symbol aliases, hotspot
--                    scores, reachability, runtime entrypoints, function-test
-- ---------------------------------------------------------------------------

-- IMPROVEMENT 1: Function-level call graph (who calls whom)
CREATE TABLE IF NOT EXISTS call_edges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    caller_file TEXT NOT NULL,
    caller_func TEXT NOT NULL,
    callee_func TEXT NOT NULL,
    callee_file TEXT,
    call_line INTEGER,
    UNIQUE(caller_file, caller_func, callee_func, call_line)
);

-- IMPROVEMENT 2: Config / CMake option -> code symbol binding
CREATE TABLE IF NOT EXISTS config_bindings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    config_file TEXT NOT NULL,      -- CMakeLists.txt, config.json, etc.
    config_key TEXT NOT NULL,       -- option name / JSON field
    code_symbol TEXT NOT NULL,      -- #define / function / macro that handles it
    code_file TEXT,
    binding_type TEXT,              -- cmake_option, json_field, build_flag, project_constant
    description TEXT,
    UNIQUE(config_file, config_key, code_symbol)
);

-- IMPROVEMENT 3: Symbol aliases and typo/variant detection
CREATE TABLE IF NOT EXISTS symbol_aliases (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    canonical TEXT NOT NULL,
    alias TEXT NOT NULL,
    similarity REAL,                -- 0.0-1.0 SequenceMatcher ratio
    kind TEXT,                      -- typo, variant, abbreviation
    source_file TEXT,
    UNIQUE(canonical, alias)
);

-- IMPROVEMENT 4: Per-file hotspot risk scores
CREATE TABLE IF NOT EXISTS hotspot_scores (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path TEXT UNIQUE NOT NULL,
    coupling_score REAL DEFAULT 0,      -- fan-in + fan-out from include_deps
    security_density REAL DEFAULT 0,    -- security_patterns / lines * 100
    null_risk_score REAL DEFAULT 0,     -- raw pointer / reinterpret patterns
    test_coverage_gap REAL DEFAULT 0,   -- 1 = no tests, 0 = covered
    hotspot_score REAL DEFAULT 0,       -- weighted composite (0-10)
    reasons TEXT                        -- JSON array of contributing factors
);

-- IMPROVEMENT 5: Dead-code / reachability analysis
CREATE TABLE IF NOT EXISTS reachability (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    file_path TEXT NOT NULL,
    is_reachable INTEGER DEFAULT 1,
    reach_via TEXT,                 -- caller that makes it reachable
    dead_reason TEXT,               -- no_caller_in_call_graph, etc.
    UNIQUE(symbol, file_path)
);

-- IMPROVEMENT 6: Runtime entrypoints and startup file loaders
CREATE TABLE IF NOT EXISTS runtime_entrypoints (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    binary TEXT NOT NULL,
    entrypoint_func TEXT NOT NULL,
    loads_file TEXT,
    load_mechanism TEXT,            -- fopen, compiled-in, cmake-option, string_ref
    source_file TEXT,
    line_no INTEGER
);

-- IMPROVEMENT 7: Function -> test target mapping (function-level coverage)
CREATE TABLE IF NOT EXISTS function_test_map (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    function_name TEXT NOT NULL,
    function_file TEXT NOT NULL,
    test_target TEXT NOT NULL,
    coverage_type TEXT,             -- indirect, kat, fuzzing
    UNIQUE(function_name, function_file, test_target)
);

-- Phase 5: crypto reasoning layers for symbol-level analysis
CREATE TABLE IF NOT EXISTS symbol_semantics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol_name TEXT NOT NULL,
    file_path TEXT NOT NULL,
    category TEXT,
    math_core TEXT,
    backend TEXT,
    coordinate_model TEXT,
    secret_class TEXT,
    abi_surface INTEGER DEFAULT 0,
    generator_path INTEGER DEFAULT 0,
    varpoint_path INTEGER DEFAULT 0,
    bip340_related INTEGER DEFAULT 0,
    bip352_related INTEGER DEFAULT 0,
    UNIQUE(symbol_name, file_path)
);

CREATE TABLE IF NOT EXISTS symbol_security (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol_name TEXT NOT NULL,
    file_path TEXT NOT NULL,
    uses_secret_input INTEGER DEFAULT 0,
    must_be_constant_time INTEGER DEFAULT 0,
    public_data_only INTEGER DEFAULT 0,
    device_secret_upload INTEGER DEFAULT 0,
    requires_zeroization INTEGER DEFAULT 0,
    invalid_input_sensitive INTEGER DEFAULT 0,
    notes TEXT,
    UNIQUE(symbol_name, file_path)
);

CREATE TABLE IF NOT EXISTS symbol_performance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol_name TEXT NOT NULL,
    file_path TEXT NOT NULL,
    hotness_score REAL DEFAULT 0,
    estimated_cost REAL DEFAULT 0,
    batchable INTEGER DEFAULT 0,
    vectorizable INTEGER DEFAULT 0,
    gpu_candidate INTEGER DEFAULT 0,
    memory_bound INTEGER DEFAULT 0,
    compute_bound INTEGER DEFAULT 0,
    duplicated_backends INTEGER DEFAULT 0,
    UNIQUE(symbol_name, file_path)
);

CREATE TABLE IF NOT EXISTS symbol_audit_coverage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol_name TEXT NOT NULL,
    file_path TEXT NOT NULL,
    covered_by_unit_test INTEGER DEFAULT 0,
    covered_by_fuzz INTEGER DEFAULT 0,
    covered_by_invalid_vectors INTEGER DEFAULT 0,
    covered_by_ct_test INTEGER DEFAULT 0,
    covered_by_cross_impl_diff INTEGER DEFAULT 0,
    covered_by_gpu_equivalence INTEGER DEFAULT 0,
    covered_by_regression_test INTEGER DEFAULT 0,
    last_audit_result TEXT DEFAULT 'unknown',
    times_failed_historically INTEGER DEFAULT 0,
    known_fragile INTEGER DEFAULT 0,
    UNIQUE(symbol_name, file_path)
);

CREATE TABLE IF NOT EXISTS symbol_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol_name TEXT NOT NULL,
    file_path TEXT NOT NULL,
    times_modified INTEGER DEFAULT 0,
    recently_modified INTEGER DEFAULT 0,
    bug_fix_count INTEGER DEFAULT 0,
    performance_tuning_count INTEGER DEFAULT 0,
    audit_related_changes INTEGER DEFAULT 0,
    last_modified TEXT,
    UNIQUE(symbol_name, file_path)
);

CREATE TABLE IF NOT EXISTS symbol_scores (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol_name TEXT NOT NULL,
    file_path TEXT NOT NULL,
    risk_score REAL DEFAULT 0,
    gain_score REAL DEFAULT 0,
    optimization_priority REAL DEFAULT 0,
    risk_reasons TEXT,
    gain_reasons TEXT,
    UNIQUE(symbol_name, file_path)
);

-- Views for new tables
CREATE VIEW IF NOT EXISTS v_call_graph AS
    SELECT caller_file, caller_func, callee_func, callee_file, call_line
    FROM call_edges ORDER BY caller_file, caller_func, call_line;

CREATE VIEW IF NOT EXISTS v_hotspot_top AS
    SELECT file_path, hotspot_score, coupling_score, security_density,
           test_coverage_gap, null_risk_score, reasons
    FROM hotspot_scores ORDER BY hotspot_score DESC LIMIT 30;

CREATE VIEW IF NOT EXISTS v_dead_code AS
    SELECT symbol, file_path, dead_reason
    FROM reachability WHERE is_reachable = 0
    ORDER BY file_path, symbol;

CREATE VIEW IF NOT EXISTS v_function_coverage AS
    SELECT ftm.function_name, ftm.function_file, ftm.test_target, ftm.coverage_type,
           fi.start_line, fi.end_line
    FROM function_test_map ftm
    LEFT JOIN function_index fi ON fi.file_path = ftm.function_file
        AND fi.name = ftm.function_name
    ORDER BY ftm.function_file, ftm.function_name;

CREATE VIEW IF NOT EXISTS v_symbol_reasoning AS
    SELECT ss.symbol_name, ss.file_path, ss.category, ss.math_core, ss.backend,
           ss.coordinate_model, ss.secret_class, ss.abi_surface,
           sec.uses_secret_input, sec.must_be_constant_time, sec.public_data_only,
           perf.hotness_score, perf.gpu_candidate, perf.batchable,
           perf.compute_bound, perf.memory_bound,
           cov.covered_by_unit_test, cov.covered_by_fuzz, cov.covered_by_ct_test,
           hist.times_modified, hist.recently_modified,
           score.risk_score, score.gain_score, score.optimization_priority
    FROM symbol_semantics ss
    LEFT JOIN symbol_security sec
      ON sec.symbol_name = ss.symbol_name AND sec.file_path = ss.file_path
    LEFT JOIN symbol_performance perf
      ON perf.symbol_name = ss.symbol_name AND perf.file_path = ss.file_path
    LEFT JOIN symbol_audit_coverage cov
      ON cov.symbol_name = ss.symbol_name AND cov.file_path = ss.file_path
    LEFT JOIN symbol_history hist
      ON hist.symbol_name = ss.symbol_name AND hist.file_path = ss.file_path
    LEFT JOIN symbol_scores score
      ON score.symbol_name = ss.symbol_name AND score.file_path = ss.file_path;
"""

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def file_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    try:
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                h.update(chunk)
        return h.hexdigest()[:16]  # short hash
    except Exception:
        return ""

def count_lines(filepath: Path) -> int:
    try:
        with open(filepath, 'rb') as f:
            return sum(1 for _ in f)
    except Exception:
        return 0


def should_skip_dir(dirname: str) -> bool:
    """Filter generated or irrelevant directories during graph traversal."""
    return dirname.startswith('.') or dirname in SKIP_DIRS or dirname.startswith('build')

def classify_file(rel_path: str):
    """Return (category, subsystem, layer) for a relative path."""
    p = rel_path.lower()
    
    # Category
    if p.startswith('cpu/src/'):
        cat = 'cpu_core'
    elif p.startswith('cpu/include/'):
        cat = 'cpu_header'
    elif p.startswith('cpu/tests/') or p.startswith('cpu/test'):
        cat = 'cpu_test'
    elif p.startswith('cpu/bench/'):
        cat = 'benchmark'
    elif p.startswith('cpu/fuzz/'):
        cat = 'fuzz'
    elif p.startswith('audit/'):
        cat = 'audit'
    elif p.startswith('cuda/'):
        cat = 'cuda'
    elif p.startswith('opencl/'):
        cat = 'opencl'
    elif p.startswith('metal/'):
        cat = 'metal'
    elif p.startswith('include/ufsecp/'):
        cat = 'abi'
    elif p.startswith('bindings/'):
        cat = 'binding'
    elif p.startswith('examples/'):
        cat = 'example'
    elif p.startswith('scripts/'):
        cat = 'script'
    elif p.startswith('compat/'):
        cat = 'compat'
    elif p.startswith('wasm/'):
        cat = 'wasm'
    elif p.startswith('android/'):
        cat = 'android'
    elif p.startswith('tests/'):
        cat = 'test_integration'
    else:
        cat = 'other'
    
    # Subsystem
    basename = os.path.basename(p)
    sub = None
    subsystem_map = {
        'field': ['field'],
        'scalar': ['scalar'],
        'point': ['point'],
        'ecdsa': ['ecdsa'],
        'schnorr': ['schnorr'],
        'ecdh': ['ecdh'],
        'musig2': ['musig2'],
        'frost': ['frost'],
        'adaptor': ['adaptor'],
        'taproot': ['taproot'],
        'ecies': ['ecies'],
        'bip32': ['bip32'],
        'bip39': ['bip39'],
        'bip340': ['bip340'],
        'ethereum': ['ethereum', 'eth_signing', 'keccak'],
        'hash': ['sha256', 'sha512', 'hash_accel', 'hash160', 'keccak256', 'tagged_hash'],
        'pedersen': ['pedersen'],
        'zk': ['zk', 'range_proof'],
        'glv': ['glv'],
        'precompute': ['precompute', 'ecmult_gen_comb'],
        'recovery': ['recovery'],
        'multiscalar': ['multiscalar', 'pippenger'],
        'batch': ['batch_verify', 'batch_add_affine'],
        'wallet': ['wallet', 'coin_address', 'coin_hd', 'coin_params', 'address', 'message_signing', 'wif'],
        'selftest': ['selftest'],
        'ct': ['ct_field', 'ct_scalar', 'ct_point', 'ct_sign', 'ct_ops', 'ct_utils', 'ct_zk'],
    }
    for subsys, keywords in subsystem_map.items():
        for kw in keywords:
            if kw in basename:
                sub = subsys
                break
        if sub:
            break
    
    # Layer
    # Use startswith to avoid false matches like "project_" containing "ct_"
    if 'ct/' in p or basename.startswith('ct_'):
        layer = 'ct'
    elif cat in ('cuda', 'opencl', 'metal'):
        layer = 'gpu'
    elif cat == 'abi':
        layer = 'abi'
    elif cat in ('script', 'benchmark'):
        layer = 'tool'
    elif cat in ('audit', 'cpu_test', 'fuzz', 'test_integration'):
        layer = 'test'
    else:
        layer = 'fast'
    
    return cat, sub, layer

def extract_includes(filepath: Path):
    """Extract #include directives from a source file."""
    includes = []
    try:
        with open(filepath, 'r', errors='replace') as f:
            for line in f:
                m = re.match(r'\s*#include\s+([<"])(.*?)[>"]', line)
                if m:
                    is_local = m.group(1) == '"'
                    includes.append((m.group(2), is_local))
    except Exception:
        pass
    return includes

def categorize_abi_func(name: str):
    """Categorize a ufsecp_* function name."""
    n = name.replace('ufsecp_', '')
    categories = {
        'gpu': 'gpu', 'ctx': 'context', 'seckey': 'seckey', 'pubkey': 'pubkey',
        'ecdsa': 'ecdsa', 'schnorr': 'schnorr', 'ecdh': 'ecdh',
        'sha256': 'hash', 'sha512': 'hash', 'hash160': 'hash',
        'tagged_hash': 'hash', 'keccak256': 'ethereum',
        'addr': 'address', 'wif': 'wif',
        'bip32': 'bip32', 'bip39': 'bip39',
        'taproot': 'taproot', 'musig2': 'musig2', 'frost': 'frost',
        'adaptor': 'adaptor', 'pedersen': 'pedersen', 'zk': 'zk',
        'coin': 'wallet', 'btc_message': 'wallet',
        'silent_payment': 'silent_payments',
        'ecies': 'ecies', 'eth': 'ethereum',
        'shamir': 'multiscalar', 'multi_scalar': 'multiscalar',
        'last_error': 'error', 'error': 'error',
    }
    for prefix, cat in categories.items():
        if n.startswith(prefix):
            return cat
    return 'other'

def abi_layer(name: str):
    """Which layer does a C ABI function route to?"""
    ct_funcs = {'ecdsa_sign', 'ecdsa_sign_verified', 'schnorr_sign',
                'schnorr_sign_verified', 'ecdh', 'ecdh_xonly', 'ecdh_raw',
                'bip32_master', 'bip32_derive', 'bip32_derive_path',
                'seckey_negate', 'seckey_tweak_add', 'seckey_tweak_mul',
                'musig2_partial_sign', 'frost_sign',
                'schnorr_adaptor_sign', 'ecdsa_adaptor_sign',
                'ecies_encrypt', 'ecies_decrypt',
                'eth_sign', 'silent_payment_create_output',
                'taproot_tweak_seckey'}
    n = name.replace('ufsecp_', '')
    if n in ct_funcs:
        return 'ct'
    elif 'verify' in n or 'batch' in n or 'parse' in n or 'create' in n:
        return 'fast'
    return 'both'


SEMANTIC_TAGS = {
    'constant_time': ('security', 'Touches secret-bearing constant-time paths or CT verification logic.'),
    'verification': ('crypto', 'Signature, proof, or validity verification logic.'),
    'signing': ('crypto', 'Secret-key signing, nonce generation, or signing session flow.'),
    'parser_boundary': ('security', 'Strict parsing, decoding, or malformed-input boundary handling.'),
    'wallet_flow': ('protocol', 'Wallet, address, mnemonic, or HD derivation user-facing flow.'),
    'protocol_multisig': ('protocol', 'MuSig2, FROST, adaptor, or multi-party signing logic.'),
    'privacy_protocol': ('protocol', 'Silent payments, ECIES, Pedersen, or privacy-preserving flow.'),
    'zk_proof': ('protocol', 'Zero-knowledge proof generation or verification.'),
    'gpu_acceleration': ('platform', 'GPU backend, kernels, host API, or backend dispatch logic.'),
    'cross_platform': ('platform', 'Platform dispatch, compatibility, or architecture-specific implementation.'),
    'ffi_surface': ('ecosystem', 'Public C ABI, bindings, or host-language interop surface.'),
    'audit_evidence': ('tooling', 'Audit runner, assurance export, CI evidence, or self-audit infrastructure.'),
    'benchmarking': ('tooling', 'Benchmark harness, performance measurement, or regression gate.'),
    'build_release': ('tooling', 'Build, release, packaging, or reproducibility flow.'),
    'hashing': ('crypto', 'SHA-2, Keccak, tagged hash, hash160, or hash acceleration paths.'),
    'ethereum_stack': ('protocol', 'Ethereum signing, recovery, addressing, or Keccak-backed APIs.'),
    'taproot_stack': ('protocol', 'Taproot, BIP-340, x-only pubkeys, or taproot tweaking logic.'),
    'differential_testing': ('tooling', 'Cross-lib differential, KAT, vector, or equivalence testing.'),
    'bitcoin_core': ('ecosystem', 'Bitcoin Core integration, consensus parity, or libsecp256k1 compatibility.'),
    'compat_shim': ('ecosystem', 'libsecp256k1-compatible shim surface and API semantics.'),
    'caas_evidence': ('tooling', 'Continuous Audit as a Service pipeline, gate, bundle, or reproducible evidence.'),
}


def derive_semantic_tags_for_source(path: str, category: str, subsystem: str, layer: str):
    """Derive semantic tags for a source/docs artifact from path/category/layer hints."""
    p = path.lower()
    tags = {}

    def add(tag: str, confidence: float = 1.0):
        if tag in SEMANTIC_TAGS:
            tags[tag] = max(tags.get(tag, 0.0), confidence)

    if layer == 'ct' or '/ct/' in p or os.path.basename(p).startswith('ct_'):
        add('constant_time', 1.0)
    if category in ('abi', 'binding') or p.startswith('bindings/') or 'ufsecp' in p:
        add('ffi_surface', 0.95)
    if category in ('audit', 'cpu_test', 'fuzz', 'test_integration') or 'audit' in p:
        add('audit_evidence', 0.95)
    if category == 'benchmark' or 'bench' in p:
        add('benchmarking', 0.95)
    if category in ('cuda', 'opencl', 'metal') or p.startswith('gpu/'):
        add('gpu_acceleration', 1.0)
    if category in ('compat', 'android', 'wasm') or any(x in p for x in ('arm64', 'riscv', 'esp32', 'wasm', 'msvc', 'cross_platform')):
        add('cross_platform', 0.9)
    if category == 'compat' or 'libsecp256k1_shim' in p or 'secp256k1_shim' in p:
        add('compat_shim', 1.0)
        add('bitcoin_core', 0.9)
        add('ffi_surface', 0.9)
    if category in ('script',) or any(x in p for x in ('release', 'package', 'build', 'cmake', 'preset', 'reproducible')):
        add('build_release', 0.8)
    if any(x in p for x in ('caas', 'audit_gate', 'security_autonomy', 'external_audit_bundle', 'evidence_governance')):
        add('caas_evidence', 1.0)
        add('audit_evidence', 0.95)
    if any(x in p for x in ('bitcoin_core', 'bitcoin-core', 'libsecp256k1', 'libsecp', 'consensus')):
        add('bitcoin_core', 1.0)

    if subsystem in ('ecdsa', 'schnorr'):
        add('signing', 0.9)
        add('verification', 0.85)
    if subsystem in ('wallet', 'bip32', 'bip39'):
        add('wallet_flow', 1.0)
    if subsystem in ('musig2', 'frost', 'adaptor'):
        add('protocol_multisig', 1.0)
    if subsystem in ('ecies', 'pedersen'):
        add('privacy_protocol', 0.95)
    if subsystem == 'zk':
        add('zk_proof', 1.0)
    if subsystem == 'ethereum':
        add('ethereum_stack', 1.0)
    if subsystem == 'taproot' or 'bip340' in p or 'taproot' in p:
        add('taproot_stack', 0.95)
    if subsystem == 'hash' or any(x in p for x in ('sha256', 'sha512', 'hash160', 'keccak', 'tagged_hash', 'hash_accel')):
        add('hashing', 1.0)

    if any(x in p for x in ('parse', 'parser', 'der', 'ffi_round_trip', 'boundary', 'normalize', 'strict')):
        add('parser_boundary', 0.9)
    if any(x in p for x in ('wycheproof', 'differential', 'fiat_crypto', 'equivalence', 'cross_platform_kat', 'vectors')):
        add('differential_testing', 0.9)

    return tags


def derive_semantic_tags_for_abi(name: str, category: str, layer: str):
    """Derive semantic tags for a C ABI function."""
    n = name.lower()
    tags = {'ffi_surface': 1.0}

    def add(tag: str, confidence: float = 1.0):
        if tag in SEMANTIC_TAGS:
            tags[tag] = max(tags.get(tag, 0.0), confidence)

    if layer == 'ct':
        add('constant_time', 1.0)
    if category in ('ecdsa', 'schnorr'):
        if 'sign' in n:
            add('signing', 1.0)
        if 'verify' in n:
            add('verification', 1.0)
    if category in ('wallet', 'bip32', 'bip39', 'address', 'wif'):
        add('wallet_flow', 1.0)
    if category in ('musig2', 'frost', 'adaptor'):
        add('protocol_multisig', 1.0)
    if category in ('ecies', 'silent_payments', 'pedersen'):
        add('privacy_protocol', 0.95)
    if category == 'zk':
        add('zk_proof', 1.0)
    if category == 'ethereum':
        add('ethereum_stack', 1.0)
    if category == 'taproot':
        add('taproot_stack', 1.0)
    if category == 'hash':
        add('hashing', 1.0)
    if 'parse' in n or 'from_der' in n or 'validate' in n:
        add('parser_boundary', 0.9)
    return tags


def infer_backend_from_path(path: str) -> str:
    p = path.lower()
    if p.startswith('cuda/') or p.startswith('gpu/'):
        return 'cuda'
    if p.startswith('opencl/'):
        return 'opencl'
    if p.startswith('metal/'):
        return 'metal'
    if p.startswith('wasm/'):
        return 'wasm'
    if p.startswith('scripts/'):
        return 'script'
    return 'cpu'


def infer_coordinate_model(symbol_name: str, file_path: str) -> str:
    s = f"{symbol_name} {file_path}".lower()
    if 'jacobian' in s:
        return 'jacobian'
    if 'affine' in s:
        return 'affine'
    if 'point' in s or 'scalar_mul' in s or 'ecmult' in s:
        return 'mixed'
    return 'n/a'


def classify_symbol_category(symbol_name: str, file_path: str):
    s = f"{symbol_name} {file_path}".lower()
    if any(x in s for x in ('libsecp256k1_shim', 'secp256k1_context', 'secp256k1_ecdsa', 'secp256k1_schnorr')):
        return 'compat_shim', 'abi'
    if any(x in s for x in ('caas', 'audit_gate', 'security_autonomy', 'external_audit_bundle', 'evidence_governance')):
        return 'caas', 'tool'
    if any(x in s for x in ('bitcoin_core', 'bitcoin-core', 'libsecp256k1', 'libsecp', 'consensus')):
        return 'bitcoin_core', 'abi'
    if any(x in s for x in ('field', 'fe52', 'fe_')):
        return 'field_arithmetic', 'field'
    if any(x in s for x in ('scalar', 'safegcd', 'wnaf')):
        return 'scalar_arithmetic', 'scalar'
    if any(x in s for x in ('point', 'ecmult', 'generator_mul', 'scalar_mul', 'multiscalar', 'pippenger', 'shamir')):
        return 'point_arithmetic', 'point'
    if any(x in s for x in ('inverse', 'modinv', 'safegcd')):
        return 'modinv', 'scalar'
    if 'batch_inversion' in s:
        return 'batch_inversion', 'field'
    if 'ecdsa' in s:
        return 'ecdsa', 'protocol'
    if 'schnorr' in s or 'bip340' in s:
        return 'schnorr', 'protocol'
    if 'ecdh' in s:
        return 'ecdh', 'protocol'
    if 'bip352' in s or 'silent_payment' in s:
        return 'bip352', 'protocol'
    if any(x in s for x in ('sha256', 'sha512', 'keccak', 'hash160', 'tagged_hash', 'hash_accel')):
        return 'hashing', 'hash'
    if any(x in s for x in ('kernel', '.cl', '.metal', '.cu')):
        return 'gpu_kernel', 'gpu'
    if any(x in s for x in ('ufsecp_', 'ffi', 'binding')):
        return 'ffi_abi', 'abi'
    if any(x in s for x in ('test_', '/tests/', '/audit/')):
        return 'test', 'tool'
    if '/audit/' in s or 'audit_' in s:
        return 'audit', 'tool'
    if '/fuzz/' in s or 'fuzz' in s:
        return 'fuzz', 'tool'
    if any(x in s for x in ('bip32', 'bip39', 'wallet', 'address', 'coin_', 'wif')):
        return 'wallet', 'wallet'
    return 'general', 'tool'


def derive_symbol_semantics(symbol_name: str, file_path: str):
    category, math_core = classify_symbol_category(symbol_name, file_path)
    s = f"{symbol_name} {file_path}".lower()
    secret_class = 'public_only'
    # Verify/validate functions use only public data — never ct_sensitive.
    # Must check before the broad ct_sensitive keywords (e.g. 'adaptor', 'frost', 'musig2')
    # which would otherwise match 'ecdsa_adaptor_verify', 'frost_verify_partial', etc.
    is_verify = any(x in s for x in ('verify', 'validate', 'check_sig', 'batch_verify'))
    if not is_verify and any(x in s for x in ('sign', 'nonce', 'seckey', 'ecdh', 'decrypt', 'derive', 'blind', 'adaptor', 'silent_payment', 'frost', 'musig2')):
        secret_class = 'ct_sensitive'
    elif any(x in s for x in ('aggregate', 'combine', 'session', 'tweak', 'commit')):
        secret_class = 'mixed'
    return {
        'category': category,
        'math_core': math_core,
        'backend': infer_backend_from_path(file_path),
        'coordinate_model': infer_coordinate_model(symbol_name, file_path),
        'secret_class': secret_class,
        'abi_surface': 1 if symbol_name.startswith('ufsecp_') or '/ufsecp_' in file_path else 0,
        'generator_path': 1 if any(x in s for x in ('generator_mul', 'gen_mul', '*g', 'output_key')) else 0,
        'varpoint_path': 1 if any(x in s for x in ('scalar_mul', 'varpoint', 'pippenger', 'shamir', 'point_mul')) else 0,
        'bip340_related': 1 if any(x in s for x in ('bip340', 'schnorr', 'taproot')) else 0,
        'bip352_related': 1 if any(x in s for x in ('bip352', 'silent_payment')) else 0,
    }


def derive_symbol_security(symbol_name: str, file_path: str, semantics: dict):
    s = f"{symbol_name} {file_path}".lower()
    uses_secret = 1 if semantics['secret_class'] in ('ct_sensitive', 'mixed') else 0
    must_ct = 1 if semantics['secret_class'] == 'ct_sensitive' or 'ct_' in s or '/ct/' in s else 0
    public_only = 1 if semantics['secret_class'] == 'public_only' else 0
    device_secret = 1 if semantics['backend'] in ('cuda', 'opencl', 'metal') and uses_secret else 0
    zeroize = 1 if any(x in s for x in ('sign', 'nonce', 'ecdh', 'decrypt', 'blind', 'derive', 'seed', 'mnemonic')) else 0
    invalid_input = 1 if any(x in s for x in ('parse', 'verify', 'validate', 'from_der', 'pubkey', 'sig', 'address', 'decode', 'ecies', 'seckey', 'context_randomize')) else 0
    notes = []
    if device_secret:
        notes.append('secret-bearing symbol reaches GPU/backend memory')
    if invalid_input:
        notes.append('parser or validation boundary')
    if must_ct:
        notes.append('constant-time sensitive')
    return {
        'uses_secret_input': uses_secret,
        'must_be_constant_time': must_ct,
        'public_data_only': public_only,
        'device_secret_upload': device_secret,
        'requires_zeroization': zeroize,
        'invalid_input_sensitive': invalid_input,
        'notes': '; '.join(notes),
    }


def derive_symbol_performance(symbol_name: str, file_path: str, semantics: dict):
    s = f"{symbol_name} {file_path}".lower()
    hotness = 20.0
    if semantics['math_core'] in ('field', 'scalar', 'point'):
        hotness += 25.0
    if any(x in s for x in ('mul', 'verify', 'scan', 'msm', 'pippenger', 'batch', 'kernel')):
        hotness += 20.0
    if any(x in s for x in ('generator_mul', 'scalar_mul', 'ecdh', 'bip352', 'silent_payment')):
        hotness += 15.0
    estimated_cost = min(100.0, hotness + (15.0 if any(x in s for x in ('msm', 'pippenger', 'verify', 'batch')) else 0.0))
    batchable = 1 if any(x in s for x in ('batch', 'msm', 'verify', 'scan', 'aggregate', 'combine')) else 0
    vectorizable = 1 if semantics['math_core'] in ('field', 'scalar', 'point', 'hash') else 0
    gpu_candidate = 1 if semantics['backend'] == 'cpu' and (
        semantics['math_core'] in ('field', 'scalar', 'point', 'hash', 'protocol') and batchable
        or any(x in s for x in ('msm', 'batch', 'verify', 'scan', 'hash160', 'keccak', 'ecdh'))
    ) else 0
    memory_bound = 1 if any(x in s for x in ('parse', 'serialize', 'decode', 'encode', 'address', 'wallet', 'seed')) else 0
    compute_bound = 1 if any(x in s for x in ('mul', 'verify', 'msm', 'inverse', 'kernel', 'hash', 'scan')) else 0
    duplicated_backends = 1 if semantics['backend'] in ('cuda', 'opencl', 'metal') or any(x in s for x in ('gpu_', 'opencl', 'metal', 'cuda')) else 0
    return {
        'hotness_score': min(100.0, hotness),
        'estimated_cost': estimated_cost,
        'batchable': batchable,
        'vectorizable': vectorizable,
        'gpu_candidate': gpu_candidate,
        'memory_bound': memory_bound,
        'compute_bound': compute_bound,
        'duplicated_backends': duplicated_backends,
    }


def get_file_history(file_path: str, cache: dict):
    if file_path in cache:
        return cache[file_path]
    try:
        proc = subprocess.run(
            ['git', '-C', str(LIB_ROOT), 'log', '--follow', '--format=%H%x09%ct%x09%s', '--', file_path],
            capture_output=True,
            text=True,
            check=False,
        )
        lines = [line for line in proc.stdout.splitlines() if line.strip()]
    except Exception:
        lines = []
    history = {
        'times_modified': len(lines),
        'recently_modified': 0,
        'bug_fix_count': 0,
        'performance_tuning_count': 0,
        'audit_related_changes': 0,
        'last_modified': None,
    }
    if lines:
        now_ts = int(datetime.now(timezone.utc).timestamp())
        first = lines[0].split('\t', 2)
        if len(first) >= 2:
            history['last_modified'] = first[1]
        for entry in lines:
            parts = entry.split('\t', 2)
            if len(parts) < 3:
                continue
            _, ts, subject = parts
            try:
                if now_ts - int(ts) <= 30 * 24 * 3600:
                    history['recently_modified'] = 1
            except Exception:
                pass
            lowered = subject.lower()
            if any(x in lowered for x in ('fix', 'bug', 'regression', 'correct', 'repair')):
                history['bug_fix_count'] += 1
            if any(x in lowered for x in ('perf', 'opt', 'speed', 'fast', 'vector', 'batch', 'gpu')):
                history['performance_tuning_count'] += 1
            if any(x in lowered for x in ('audit', 'security', 'ct', 'fuzz', 'wycheproof', 'sanitizer')):
                history['audit_related_changes'] += 1
    cache[file_path] = history
    return history

# ---------------------------------------------------------------------------
# POPULATION FUNCTIONS
# ---------------------------------------------------------------------------

def populate_source_files(cur: sqlite3.Cursor):
    """Walk the source tree and insert all source files."""
    count = 0
    for root, dirs, files in os.walk(LIB_ROOT):
        # Skip build dirs
        dirs[:] = [d for d in dirs if not should_skip_dir(d)]
        for fname in sorted(files):
            ext = os.path.splitext(fname)[1].lower()
            if ext not in SOURCE_EXTS:
                continue
            full = Path(root) / fname
            rel = str(full.relative_to(LIB_ROOT))
            if any(skip in rel for skip in ['build/', 'CMakeFiles/', 'CMakeCUDACompilerId', 'CMakeCXXCompilerId',
                                             'x509_crt_bundle', 'sdkconfig.h', 'kernels_embedded']):
                continue
            cat, sub, layer = classify_file(rel)
            lines = count_lines(full)
            sha = file_sha256(full)
            ftype = ext.lstrip('.')
            cur.execute("""INSERT OR IGNORE INTO source_files
                (path, category, subsystem, file_type, lines, sha256, layer)
                VALUES (?,?,?,?,?,?,?)""",
                (rel, cat, sub, ftype, lines, sha, layer))
            count += 1
    return count

def populate_abi_functions(cur: sqlite3.Cursor):
    """Parse ufsecp.h, ufsecp_gpu.h, and ufsecp_version.h for all C ABI function declarations."""
    headers = [
        LIB_ROOT / 'include' / 'ufsecp' / 'ufsecp.h',
        LIB_ROOT / 'include' / 'ufsecp' / 'ufsecp_gpu.h',
        LIB_ROOT / 'include' / 'ufsecp' / 'ufsecp_version.h',
    ]
    count = 0
    for header in headers:
        if not header.exists():
            continue
        with open(header, 'r') as f:
            lines = f.readlines()
        
        i = 0
        while i < len(lines):
            line = lines[i]
            # Match UFSECP_API function declarations
            m = re.match(r'UFSECP_API\s+(\w[\w\s*]*?)\s+(ufsecp_\w+)\s*\(', line)
            if m:
                ret_type = m.group(1).strip()
                func_name = m.group(2)
                # Collect full signature
                sig = line.strip()
                j = i + 1
                while j < len(lines) and ';' not in sig:
                    sig += ' ' + lines[j].strip()
                    j += 1
                sig = re.sub(r'\s+', ' ', sig).strip().rstrip(';')
                
                cat = categorize_abi_func(func_name)
                layer = abi_layer(func_name)
                
                cur.execute("""INSERT OR IGNORE INTO c_abi_functions
                    (name, category, signature, line_no, layer)
                    VALUES (?,?,?,?,?)""",
                    (func_name, cat, sig, i+1, layer))
                count += 1
            i += 1
    return count

def populate_include_deps(cur: sqlite3.Cursor):
    """Extract include dependencies from all .cpp/.cu/.mm files."""
    count = 0
    for root, dirs, files in os.walk(LIB_ROOT):
        dirs[:] = [d for d in dirs if not should_skip_dir(d)]
        for fname in sorted(files):
            ext = os.path.splitext(fname)[1].lower()
            if ext not in ('.cpp', '.cu', '.mm', '.c'):
                continue
            full = Path(root) / fname
            rel = str(full.relative_to(LIB_ROOT))
            if any(skip in rel for skip in ['build/', 'CMakeFiles/', 'CMakeCUDACompilerId',
                                             'CMakeCXXCompilerId', 'kernels_embedded']):
                continue
            for inc, is_local in extract_includes(full):
                cur.execute("""INSERT OR IGNORE INTO include_deps
                    (source_file, included_file, is_local)
                    VALUES (?,?,?)""", (rel, inc, is_local))
                count += 1
    return count

def populate_test_targets(cur: sqlite3.Cursor):
    """Insert known CTest targets with metadata."""
    tests = [
        # CPU Core (18)
        ('selftest', 'run_selftest', 'cpu_core', 300, '["core"]'),
        ('batch_add_affine', 'test_batch_add_affine_standalone', 'cpu_core', 300, '["core"]'),
        ('hash_accel', 'test_hash_accel_standalone', 'cpu_core', 300, '["core"]'),
        ('field_52', 'test_field_52_standalone', 'cpu_core', 300, '["core"]'),
        ('field_26', 'test_field_26_standalone', 'cpu_core', 300, '["core"]'),
        ('exhaustive', 'test_exhaustive_standalone', 'cpu_core', 300, '["core"]'),
        ('comprehensive', 'test_comprehensive_standalone', 'cpu_core', 300, '["core"]'),
        ('bip340_vectors', 'test_bip340_vectors_standalone', 'cpu_core', 300, '["core","bip340"]'),
        ('bip340_strict', 'test_bip340_strict_standalone', 'cpu_core', 300, '["core","bip340"]'),
        ('bip32_vectors', 'test_bip32_vectors_standalone', 'cpu_core', 300, '["core","bip32"]'),
        ('bip39', 'test_bip39_standalone', 'cpu_core', 300, '["core","bip39"]'),
        ('rfc6979_vectors', 'test_rfc6979_vectors_standalone', 'cpu_core', 300, '["core","rfc6979"]'),
        ('ecc_properties', 'test_ecc_properties_standalone', 'cpu_core', 300, '["core"]'),
        ('point_edge_cases', 'test_point_edge_cases_standalone', 'cpu_core', 300, '["core"]'),
        ('edge_cases', 'test_edge_cases_standalone', 'cpu_core', 300, '["core"]'),
        ('ethereum', 'test_ethereum_standalone', 'cpu_core', 300, '["core","ethereum"]'),
        ('zk_proofs', 'test_zk_standalone', 'cpu_core', 300, '["core","zk"]'),
        ('wallet', 'test_wallet_standalone', 'cpu_core', 300, '["core","wallet"]'),
        # Audit always-on (20)
        ('ct_sidechannel', 'test_ct_sidechannel_standalone', 'audit_always', 600, '["audit","ct"]'),
        ('ct_sidechannel_smoke', 'test_ct_sidechannel_standalone', 'audit_always', 120, '["audit","ct"]'),
        ('differential', 'differential_test_standalone', 'audit_always', 600, '["audit"]'),
        ('ct_equivalence', 'test_ct_equivalence_standalone', 'audit_always', 300, '["audit","ct"]'),
        ('fault_injection', 'test_fault_injection_standalone', 'audit_always', 900, '["audit"]'),
        ('debug_invariants', 'test_debug_invariants_standalone', 'audit_always', 600, '["audit"]'),
        ('fiat_crypto_vectors', 'test_fiat_crypto_vectors_standalone', 'audit_always', 900, '["audit"]'),
        ('carry_propagation', 'test_carry_propagation_standalone', 'audit_always', 900, '["audit"]'),
        ('wycheproof_ecdsa', 'test_wycheproof_ecdsa_standalone', 'audit_always', 1200, '["audit","ecdsa"]'),
        ('wycheproof_ecdh', 'test_wycheproof_ecdh_standalone', 'audit_always', 1200, '["audit","ecdh"]'),
        ('batch_randomness', 'test_batch_randomness_standalone', 'audit_always', 600, '["audit"]'),
        ('cross_platform_kat', 'test_cross_platform_kat_standalone', 'audit_always', 900, '["audit"]'),
        ('abi_gate', 'test_abi_gate', 'audit_always', 30, '["audit","abi"]'),
        ('ct_verif_formal', 'test_ct_verif_formal_standalone', 'audit_always', 1200, '["audit","ct"]'),
        ('fiat_crypto_linkage', 'test_fiat_crypto_linkage_standalone', 'audit_always', 900, '["audit"]'),
        ('audit_fuzz', 'audit_fuzz_standalone', 'audit_always', 600, '["audit","fuzz"]'),
        ('adversarial_protocol', 'test_adversarial_protocol_standalone', 'audit_always', 600, '["audit","protocol"]'),
        ('ecies_regression', 'test_ecies_regression_standalone', 'audit_always', 120, '["audit","ecies"]'),
        ('diag_scalar_mul', 'diag_scalar_mul', 'audit_always', 300, '["audit"]'),
        ('unified_audit', 'unified_audit_runner', 'audit_always', 1200, '["audit"]'),
        # Audit conditional (7)
        ('cross_libsecp256k1', 'test_cross_libsecp256k1_standalone', 'audit_conditional', 300, '["audit"]'),
        ('fuzz_parsers', 'test_fuzz_parsers_standalone', 'audit_conditional', 300, '["audit","fuzz"]'),
        ('fuzz_address_bip32_ffi', 'test_fuzz_address_bip32_ffi_standalone', 'audit_conditional', 300, '["audit","fuzz"]'),
        ('musig2_frost', 'test_musig2_frost_standalone', 'audit_conditional', 300, '["audit","musig2","frost"]'),
        ('musig2_frost_advanced', 'test_musig2_frost_advanced_standalone', 'audit_conditional', 300, '["audit","musig2","frost"]'),
        ('frost_kat', 'test_frost_kat_standalone', 'audit_conditional', 300, '["audit","frost"]'),
        ('musig2_bip327_vectors', 'test_musig2_bip327_vectors_standalone', 'audit_conditional', 300, '["audit","musig2"]'),
        ('gpu_bip352_scan', 'test_gpu_bip352_scan', 'audit_conditional', 300, '["audit","gpu","bip352"]'),
        # GPU (10)
        ('cuda_selftest', 'secp256k1_cuda_test', 'gpu', 300, '["gpu","cuda"]'),
        ('gpu_audit', 'gpu_audit_runner', 'gpu', 1200, '["gpu","cuda","audit"]'),
        ('gpu_ct_smoke', 'test_ct_smoke', 'gpu', 300, '["gpu","cuda","ct"]'),
        ('opencl_selftest', 'opencl_test', 'gpu', 300, '["gpu","opencl"]'),
        ('opencl_audit', 'opencl_audit_runner', 'gpu', 1200, '["gpu","opencl","audit"]'),
        ('metal_host_test', 'metal_host_test', 'gpu', 300, '["gpu","metal"]'),
        ('secp256k1_metal_test', 'metal_secp256k1_test', 'gpu', 300, '["gpu","metal"]'),
        ('secp256k1_metal_bench', 'metal_secp256k1_bench', 'gpu', 300, '["gpu","metal"]'),
        ('secp256k1_metal_bench_full', 'metal_secp256k1_bench_full', 'gpu', 600, '["gpu","metal"]'),
        ('secp256k1_metal_audit', 'metal_audit_runner', 'gpu', 1200, '["gpu","metal","audit"]'),
    ]
    for name, exe, cat, timeout, labels in tests:
        cur.execute("""INSERT OR IGNORE INTO test_targets
            (name, executable, category, timeout, labels)
            VALUES (?,?,?,?,?)""", (name, exe, cat, timeout, labels))
    return len(tests)

def populate_ci_workflows(cur: sqlite3.Cursor):
    """Insert CI workflow metadata."""
    workflows = [
        ('ci.yml', 'CI', '{"push":["main","dev"],"pull_request":["main","dev"]}', 'merge_blocking',
         '["linux-gcc13","linux-clang17","windows","macos"]'),
        ('clang-tidy.yml', 'Static Analysis (clang-tidy)', '{"push":["main","dev"],"pull_request":["main","dev"]}', 'merge_blocking',
         '["clang-tidy"]'),
        ('cppcheck.yml', 'Static Analysis (Cppcheck)', '{"push":["main","dev"],"pull_request":["main","dev"]}', 'merge_blocking',
         '["cppcheck"]'),
        ('ct-verif.yml', 'CT-Verif LLVM', '{"push":["main","dev"],"pull_request":["main","dev"]}', 'merge_blocking',
         '["ct-verif"]'),
        ('codeql.yml', 'CodeQL', '{"push":["main","dev"],"pull_request":["main"],"schedule":"weekly"}', 'merge_blocking',
         '["analyze"]'),
        ('security-audit.yml', 'Security Audit', '{"push":["main","dev"],"pull_request":["main","dev"],"schedule":"weekly"}', 'merge_blocking',
         '["werror","asan","audit"]'),
        ('bench-regression.yml', 'Benchmark Regression', '{"push":["main","dev"]}', 'merge_blocking',
         '["bench-regression"]'),
        ('nightly.yml', 'Nightly Extended', '{"schedule":"0 3 * * *"}', 'advisory',
         '["differential-extended","dudect-30min"]'),
        ('ct-arm64.yml', 'CT ARM64', '{"push":["main","dev"],"schedule":"0 4 * * *"}', 'advisory',
         '["ct-arm64-dudect"]'),
        ('valgrind-ct.yml', 'Valgrind CT', '{"push":["main","dev"]}', 'advisory',
         '["valgrind-taint"]'),
        ('cflite.yml', 'ClusterFuzzLite', '{"push":[],"pull_request":[],"schedule":"daily"}', 'advisory',
         '["fuzz"]'),
        ('mutation.yml', 'Mutation Testing', '{"schedule":"0 5 * * 0"}', 'advisory',
         '["mutation"]'),
        ('sonarcloud.yml', 'SonarCloud', '{"push":["main","dev"],"pull_request":["main","dev"]}', 'advisory',
         '["sonar"]'),
        ('scorecard.yml', 'OpenSSF Scorecard', '{"push":["main"],"schedule":"weekly"}', 'advisory',
         '["scorecard"]'),
        ('dependency-review.yml', 'Dependency Review', '{"pull_request":[]}', 'advisory',
         '["dependency-review"]'),
        ('release.yml', 'Release', '{"tags":["v*"],"workflow_dispatch":{}}', 'release',
         '["build-artifacts"]'),
        ('packaging.yml', 'Packaging', '{"tags":["v*"]}', 'release',
         '["deb","rpm"]'),
        ('audit-report.yml', 'Audit Report', '{"schedule":"weekly"}', 'release',
         '["unified-audit-artifacts"]'),
        ('benchmark.yml', 'Benchmark', '{"push":["main"]}', 'release',
         '["gh-pages-perf"]'),
        ('docs.yml', 'Documentation', '{"push":["main"]}', 'release',
         '["doxygen-gh-pages"]'),
        ('bindings.yml', 'Bindings', '{"push":["main","dev"],"pull_request":[]}', 'release',
         '["12-language-compile"]'),
        ('discord-commits.yml', 'Discord Notify', '{"push":["main","dev"]}', 'release',
         '["webhook"]'),
    ]
    for fname, name, triggers, cat, jobs in workflows:
        cur.execute("""INSERT OR IGNORE INTO ci_workflows
            (filename, name, triggers, category, jobs)
            VALUES (?,?,?,?,?)""", (fname, name, triggers, cat, jobs))
    return len(workflows)

def populate_error_codes(cur: sqlite3.Cursor):
    codes = [
        (0, 'OK', 'UFSECP_OK', 'Success'),
        (1, 'NULL_ARG', 'UFSECP_ERR_NULL_ARG', 'Null pointer argument'),
        (2, 'BAD_KEY', 'UFSECP_ERR_BAD_KEY', 'Invalid private key (zero or >= n)'),
        (3, 'BAD_PUBKEY', 'UFSECP_ERR_BAD_PUBKEY', 'Invalid or unparseable public key'),
        (4, 'BAD_SIG', 'UFSECP_ERR_BAD_SIG', 'Invalid or unparseable signature'),
        (5, 'BAD_INPUT', 'UFSECP_ERR_BAD_INPUT', 'Generic invalid input'),
        (6, 'VERIFY_FAIL', 'UFSECP_ERR_VERIFY_FAIL', 'Signature verification failed'),
        (7, 'ARITH', 'UFSECP_ERR_ARITH', 'Arithmetic error (point at infinity, etc.)'),
        (8, 'SELFTEST', 'UFSECP_ERR_SELFTEST', 'Self-test failure on init'),
        (9, 'INTERNAL', 'UFSECP_ERR_INTERNAL', 'Internal / unexpected error'),
        (10, 'BUF_TOO_SMALL', 'UFSECP_ERR_BUF_TOO_SMALL', 'Output buffer too small'),
    ]
    for code, name, symbol, meaning in codes:
        cur.execute("""INSERT OR IGNORE INTO error_codes
            (code, name, symbol, meaning) VALUES (?,?,?,?)""",
            (code, name, symbol, meaning))
    return len(codes)

def populate_constants(cur: sqlite3.Cursor):
    constants = [
        # Curve
        ('p', '0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F', 'curve', 'Field prime'),
        ('n', '0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141', 'curve', 'Group order'),
        ('G.x', '0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798', 'curve', 'Generator X'),
        ('G.y', '0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8', 'curve', 'Generator Y'),
        ('h', '1', 'curve', 'Cofactor'),
        ('K_MONTGOMERY', '0x1000003D1', 'curve', 'Montgomery reduction constant'),
        # Precomputation
        ('WINDOW_G', '15', 'precomp', '8192-entry table for k*G'),
        ('WINDOW_P', '5', 'precomp', '8-entry table for k*P'),
        ('GLV_WINDOW_DEFAULT_x86', '5', 'precomp', 'GLV window width x86/ARM/RISC-V'),
        ('GLV_WINDOW_DEFAULT_ESP32', '4', 'precomp', 'GLV window width ESP32/WASM'),
        ('COMB_TEETH', '6', 'precomp', 'CT comb multiplication teeth'),
        ('COMB_BLOCKS', '11', 'precomp', 'CT comb multiplication blocks'),
        ('CT_SCALAR_MUL_TABLE', '16', 'precomp', 'CT scalar_mul table size'),
        # ABI sizes
        ('PRIVKEY_SIZE', '32', 'abi_size', 'Private key bytes'),
        ('PUBKEY_COMPRESSED', '33', 'abi_size', 'Compressed public key'),
        ('PUBKEY_UNCOMPRESSED', '65', 'abi_size', 'Uncompressed public key'),
        ('PUBKEY_XONLY', '32', 'abi_size', 'X-only public key'),
        ('SIG_COMPACT', '64', 'abi_size', 'Compact ECDSA/Schnorr signature'),
        ('SIG_DER_MAX', '72', 'abi_size', 'Maximum DER-encoded signature'),
        ('BIP32_SERIALIZED', '78', 'abi_size', 'BIP-32 extended key serialized'),
        ('HASH256_SIZE', '32', 'abi_size', 'SHA-256 / Keccak-256 output'),
        ('HASH512_SIZE', '64', 'abi_size', 'SHA-512 output'),
    ]
    for name, value, cat, ctx in constants:
        cur.execute("""INSERT OR IGNORE INTO constants
            (name, value, category, context) VALUES (?,?,?,?)""",
            (name, value, cat, ctx))
    return len(constants)

def populate_gpu_backends(cur: sqlite3.Cursor):
    backends = [
        ('cuda',
         json.dumps(['cuda/src/secp256k1.cu','cuda/src/test_suite.cu','cuda/src/gpu_audit_runner.cu',
                      'cuda/src/gpu_bench_unified.cu','cuda/src/bench_compare.cu','cuda/src/bench_bip352.cu',
                      'cuda/src/bench_zk.cu','cuda/src/test_ct_smoke.cu','cuda/src/bench_cuda.cu']),
         json.dumps([]),  # CUDA uses header-only kernels
         json.dumps(['secp256k1_cuda_lib','secp256k1_cuda_test','gpu_audit_runner','gpu_bench_unified',
                      'bench_compare','test_ct_smoke','bench_zk','bench_bip352']),
         json.dumps({'ecdsa_sign':True,'ecdsa_verify':True,'schnorr_sign':True,'schnorr_verify':True,
                      'batch_verify':True,'bip32':True,'ecdh':True,'hash160':True,'keccak256':True,
                      'pedersen':True,'zk':True,'msm':True,'ct_field':True,'ct_scalar':True,
                      'ct_point':True,'ct_sign':True,'ct_zk':True,'bloom':True,'recovery':True})),
        ('opencl',
         json.dumps(['opencl/src/opencl_context.cpp','opencl/src/opencl_field.cpp',
                      'opencl/src/opencl_point.cpp','opencl/src/opencl_batch.cpp',
                      'opencl/src/opencl_selftest.cpp','opencl/src/opencl_audit_runner.cpp']),
         json.dumps(['secp256k1_field.cl','secp256k1_point.cl','secp256k1_batch.cl',
                      'secp256k1_affine.cl','secp256k1_extended.cl','secp256k1_hash160.cl',
                      'secp256k1_keccak256.cl','secp256k1_msm.cl','secp256k1_pedersen.cl',
                      'secp256k1_recovery.cl','secp256k1_zk.cl','secp256k1_bloom.cl',
                      'secp256k1_bip32.cl','secp256k1_ecdh.cl','secp256k1_gen_table_w8.cl',
                      'secp256k1_bp_gen_table.cl','secp256k1_ct_field.cl','secp256k1_ct_scalar.cl',
                      'secp256k1_ct_ops.cl','secp256k1_ct_point.cl','secp256k1_ct_sign.cl',
                      'secp256k1_ct_zk.cl']),
         json.dumps(['secp256k1_opencl','opencl_test','opencl_benchmark','opencl_audit_runner']),
         json.dumps({'ecdsa_sign':True,'ecdsa_verify':True,'schnorr_sign':True,'schnorr_verify':True,
                      'bip32':True,'ecdh':True,'hash160':True,'keccak256':True,
                      'pedersen':True,'zk':True,'msm':True,'ct_field':True,'ct_scalar':True,
                      'ct_point':True,'ct_sign':True,'ct_zk':True,'bloom':True,'recovery':True})),
        ('metal',
         json.dumps(['metal/src/metal_runtime.mm','metal/src/metal_audit_runner.mm',
                      'metal/app/bench_metal.mm','metal/app/metal_test.mm',
                      'metal/tests/test_metal_host.cpp','metal/tests/metal_extended_test.mm']),
         json.dumps(['secp256k1_field.h','secp256k1_point.h','secp256k1_affine.h',
                      'secp256k1_extended.h','secp256k1_hash160.h','secp256k1_keccak256.h',
                      'secp256k1_msm.h','secp256k1_pedersen.h','secp256k1_recovery.h',
                      'secp256k1_zk.h','secp256k1_bloom.h','secp256k1_bip32.h',
                      'secp256k1_ecdh.h','secp256k1_gen_table_w8.h','secp256k1_bp_gen_table.h',
                      'secp256k1_ct_field.h','secp256k1_ct_scalar.h','secp256k1_ct_ops.h',
                      'secp256k1_ct_point.h','secp256k1_ct_sign.h','secp256k1_ct_zk.h',
                      'secp256k1_kernels.metal']),
         json.dumps(['secp256k1_metal_lib','metal_secp256k1_test','metal_secp256k1_bench_full',
                      'metal_audit_runner','metal_host_test']),
         json.dumps({'ecdsa_sign':True,'ecdsa_verify':True,'schnorr_sign':True,'schnorr_verify':True,
                      'batch_verify':True,'bip32':True,'ecdh':True,'hash160':True,'keccak256':True,
                      'pedersen':True,'zk':True,'msm':True,'ct_field':True,'ct_scalar':True,
                      'ct_point':True,'ct_sign':True,'ct_zk':True,'bloom':True,'recovery':True})),
    ]
    for backend, sources, kernels, targets, features in backends:
        cur.execute("""INSERT OR IGNORE INTO gpu_backends
            (backend, sources, kernels, targets, features)
            VALUES (?,?,?,?,?)""", (backend, sources, kernels, targets, features))
    return len(backends)

def populate_build_configs(cur: sqlite3.Cursor):
    configs = [
        ('SECP256K1_BUILD_CPU', 'ON', 'BOOL', 'Build CPU backend'),
        ('SECP256K1_BUILD_CUDA', 'OFF', 'BOOL', 'Build CUDA GPU backend'),
        ('SECP256K1_BUILD_ROCM', 'OFF', 'BOOL', 'Build AMD ROCm/HIP GPU backend'),
        ('SECP256K1_BUILD_OPENCL', 'OFF', 'BOOL', 'Build OpenCL GPU backend'),
        ('SECP256K1_BUILD_METAL', 'OFF', 'BOOL', 'Build Apple Metal GPU backend'),
        ('SECP256K1_USE_ASM', 'ON', 'BOOL', 'Enable inline assembly (x64/ARM64/RISC-V)'),
        ('SECP256K1_BUILD_ETHEREUM', 'ON', 'BOOL', 'Build Ethereum module (Keccak-256, EIP-155)'),
        ('SECP256K1_SPEED_FIRST', 'OFF', 'BOOL', '-Ofast, no stack protector'),
        ('SECP256K1_GLV_WINDOW_WIDTH', 'platform', 'INTEGER', '5 on x86/ARM/RISC-V, 4 on ESP32/WASM'),
        ('UFSECP_BITCOIN_STRICT', 'ON', 'BOOL', 'BIP-340 strict encoding enforcement'),
        ('SECP256K1_USE_LTO', 'OFF', 'BOOL', 'Link-time optimization'),
        ('SECP256K1_BUILD_FUZZ_TESTS', 'OFF', 'BOOL', 'Build fuzz test binaries'),
        ('CMAKE_CUDA_ARCHITECTURES', '89', 'STRING', 'GPU compute capabilities'),
        ('CMAKE_BUILD_TYPE', 'Release', 'STRING', 'Build type'),
    ]
    for name, default, typ, effect in configs:
        cur.execute("""INSERT OR IGNORE INTO build_configs
            (name, default_value, type, effect) VALUES (?,?,?,?)""",
            (name, default, typ, effect))
    return len(configs)

def populate_namespaces(cur: sqlite3.Cursor):
    nss = [
        ('secp256k1', 'Top-level namespace'),
        ('secp256k1::fast', 'Variable-time maximum-throughput operations'),
        ('secp256k1::ct', 'Constant-time side-channel-resistant operations'),
        ('secp256k1::fast::debug', 'Debug invariant checking'),
        ('secp256k1::detail', 'Internal implementation utilities'),
        ('secp256k1::coins', 'Multi-coin wallet layer (BTC, ETH, LTC, etc.)'),
        ('secp256k1::hash', 'Accelerated hashing (SHA-NI, ARMv8)'),
        ('secp256k1::fast::fe52_constants', 'Field element 5x52 limb constants'),
    ]
    for name, purpose in nss:
        cur.execute("""INSERT OR IGNORE INTO namespaces
            (name, purpose) VALUES (?,?)""", (name, purpose))
    return len(nss)

def populate_cpp_types(cur: sqlite3.Cursor):
    types = [
        ('FieldElement', 'class', 'secp256k1::fast', 'cpu/include/secp256k1/field.hpp', 'Prime field element mod p (4x64 or 5x52 limbs)'),
        ('Scalar', 'class', 'secp256k1::fast', 'cpu/include/secp256k1/scalar.hpp', 'Scalar mod n (4x64 limbs)'),
        ('Point', 'class', 'secp256k1::fast', 'cpu/include/secp256k1/point.hpp', 'Affine/Jacobian point on secp256k1'),
        ('ECDSASignature', 'struct', 'secp256k1::fast', 'cpu/include/secp256k1/ecdsa.hpp', 'ECDSA (r,s) signature'),
        ('SchnorrSignature', 'struct', 'secp256k1::fast', 'cpu/include/secp256k1/schnorr.hpp', 'Schnorr/BIP-340 (R,s) signature'),
        ('ExtendedKey', 'class', 'secp256k1::fast', 'cpu/include/secp256k1/bip32.hpp', 'BIP-32 extended key (xpub/xprv)'),
        ('RecoverableSignature', 'struct', 'secp256k1::fast', 'cpu/include/secp256k1/recovery.hpp', 'ECDSA signature with recovery id'),
        ('AdaptorSignature', 'struct', 'secp256k1::fast', 'cpu/include/secp256k1/adaptor.hpp', 'Adaptor (pre-)signature'),
        ('PedersenCommitment', 'struct', 'secp256k1::fast', 'cpu/include/secp256k1/pedersen.hpp', 'Pedersen commitment (point)'),
        ('RangeProof', 'struct', 'secp256k1::fast', 'cpu/include/secp256k1/zk.hpp', 'Zero-knowledge range proof'),
        ('DLEQProof', 'struct', 'secp256k1::fast', 'cpu/include/secp256k1/zk.hpp', 'DLEQ proof'),
        ('KnowledgeProof', 'struct', 'secp256k1::fast', 'cpu/include/secp256k1/zk.hpp', 'ZK knowledge proof'),
        ('MuSig2KeyAggContext', 'struct', 'secp256k1::fast', 'cpu/include/secp256k1/musig2.hpp', 'MuSig2 key aggregation context'),
        ('MuSig2Session', 'struct', 'secp256k1::fast', 'cpu/include/secp256k1/musig2.hpp', 'MuSig2 signing session'),
        ('FROSTKeygenResult', 'struct', 'secp256k1::fast', 'cpu/include/secp256k1/frost.hpp', 'FROST DKG result'),
        ('CoinParams', 'struct', 'secp256k1::coins', 'cpu/include/secp256k1/coins/coin_params.hpp', 'Per-coin address parameters'),
        ('FieldElement', 'class', 'secp256k1::ct', 'cpu/include/secp256k1/ct/field.hpp', 'CT field element (constant-time ops)'),
        ('Scalar', 'class', 'secp256k1::ct', 'cpu/include/secp256k1/ct/scalar.hpp', 'CT scalar (constant-time ops)'),
        ('SHA256', 'class', 'secp256k1::fast', 'cpu/include/secp256k1/sha256.hpp', 'SHA-256 hasher (hardware-accelerated)'),
        ('SHA512', 'class', 'secp256k1::fast', 'cpu/include/secp256k1/sha512.hpp', 'SHA-512 hasher'),
    ]
    for name, kind, ns, header, desc in types:
        cur.execute("""INSERT OR IGNORE INTO cpp_types
            (name, kind, namespace, header_path, description) VALUES (?,?,?,?,?)""",
            (name, kind, ns, header, desc))
    return len(types)

def populate_platform_dispatch(cur: sqlite3.Cursor):
    dispatches = [
        ('cpu/src/field.cpp', 'x86_64', 'ifdef', '__SIZEOF_INT128__ -> FE52 5x52 inline mul/sqr'),
        ('cpu/src/field.cpp', 'arm64', 'ifdef', '__SIZEOF_INT128__ -> FE52 5x52 inline mul/sqr'),
        ('cpu/src/field.cpp', 'riscv64', 'ifdef', '__SIZEOF_INT128__ + optional asm mul/sqr'),
        ('cpu/src/field.cpp', 'msvc', 'ifdef', 'No __int128 -> 4x64 Comba fallback'),
        ('cpu/src/field.cpp', 'esp32', 'ifdef', 'No __int128 -> 4x64 Comba fallback'),
        ('cpu/src/ct_field.cpp', 'x86_64', 'ifdef', 'FE52 constant-time field ops w/ 5x52'),
        ('cpu/src/ct_field.cpp', 'msvc', 'ifdef', 'SafeGCD30 25x30 (no __int128)'),
        ('cpu/src/ct_scalar.cpp', 'x86_64', 'ifdef', 'SafeGCD 10x59 limbs for inversion'),
        ('cpu/src/ct_scalar.cpp', 'msvc', 'ifdef', 'SafeGCD30 25x30 limbs for inversion'),
        ('cpu/src/field_asm.cpp', 'x86_64', 'asm_file', 'ADX/MULX inline assembly mul/sqr'),
        ('cpu/src/field_asm_arm64.cpp', 'arm64', 'asm_file', 'NEON/MUL inline assembly'),
        ('cpu/src/field_asm_riscv64.S', 'riscv64', 'asm_file', 'RV64 assembly field mul/sqr'),
        ('cpu/src/field_asm52_riscv64.S', 'riscv64', 'asm_file', 'RV64 5x52 assembly'),
        ('cpu/src/field_asm_x64.asm', 'x86_64', 'asm_file', 'MASM x64 assembly (Windows)'),
        ('cpu/src/field_asm_x64_gas.S', 'x86_64', 'asm_file', 'GAS x64 assembly (Linux)'),
        ('cpu/src/hash_accel.cpp', 'x86_64', 'ifdef', 'SHA-NI intrinsics'),
        ('cpu/src/hash_accel.cpp', 'arm64', 'ifdef', 'ARMv8 CE SHA intrinsics'),
        ('cpu/src/ct_point.cpp', 'x86_64', 'ifdef', 'SSE2/AVX2 constant-time conditional move'),
        ('cpu/src/glv.cpp', 'all', 'constexpr_if', 'GLV endomorphism beta/lambda decomposition'),
    ]
    for src, plat, mech, desc in dispatches:
        cur.execute("""INSERT OR IGNORE INTO platform_dispatch
            (source_file, platform, mechanism, description) VALUES (?,?,?,?)""",
            (src, plat, mech, desc))
    return len(dispatches)

def populate_docs(cur: sqlite3.Cursor):
    """Scan docs/ directory for markdown files."""
    docs_dir = LIB_ROOT / 'docs'
    count = 0
    if not docs_dir.exists():
        return 0
    
    doc_categories = {
        'ARCHITECTURE': 'architecture', 'CODING_STANDARDS': 'architecture', 'THREAD_SAFETY': 'architecture',
        'API_REFERENCE': 'api', 'USER_GUIDE': 'api', 'ABI_VERSIONING': 'api', 'DEPRECATION': 'api',
        'BUILDING': 'build', 'COMPATIBILITY': 'build', 'CROSS_PLATFORM': 'build', 'ESP32': 'build', 'REPRODUCIBLE': 'build',
        'SECURITY': 'security', 'CT_': 'security', 'SAFE_DEFAULTS': 'security', 'BUG_BOUNTY': 'security',
        'INCIDENT': 'security', 'CRYPTO_INVARIANTS': 'security', 'INVARIANTS': 'security',
        'AUDIT': 'audit', 'FEATURE_ASSURANCE': 'audit', 'INTERNAL_AUDIT': 'audit', 'GPU_VALIDATION': 'audit',
        'DIFFERENTIAL': 'audit', 'TEST_MATRIX': 'testing', 'NORMALIZATION': 'testing', 'ENGINEERING': 'testing',
        'BENCHMARK': 'benchmark', 'PERFORMANCE': 'benchmark', 'ARM64_AUDIT': 'benchmark',
        'BINDINGS': 'binding', 'RELEASE': 'release', 'PRE_RELEASE': 'release', 'LTS': 'release', 'LOCAL_CI': 'release',
        'OPTIMIZATION': 'optimization', 'FROST': 'protocol', 'FAQ': 'general', 'README': 'general',
    }
    
    for root, dirs, files in os.walk(docs_dir):
        for fname in sorted(files):
            if not fname.endswith('.md'):
                continue
            rel = str((Path(root) / fname).relative_to(LIB_ROOT))
            title = fname.replace('.md', '').replace('_', ' ')
            
            cat = 'general'
            for prefix, c in doc_categories.items():
                if fname.upper().startswith(prefix):
                    cat = c
                    break
            
            cur.execute("""INSERT OR IGNORE INTO docs
                (path, title, category, topics) VALUES (?,?,?,?)""",
                (rel, title, cat, '[]'))
            count += 1
    return count


def populate_semantic_tags(cur: sqlite3.Cursor):
    """Populate semantic tags and entity mappings for files, ABI functions, and docs."""
    cur.execute("DELETE FROM fts_tags")
    cur.execute("DELETE FROM entity_tags")
    cur.execute("DELETE FROM semantic_tags")
    count = 0

    for tag, (domain, description) in SEMANTIC_TAGS.items():
        cur.execute("""INSERT OR IGNORE INTO semantic_tags (tag, domain, description)
                       VALUES (?,?,?)""", (tag, domain, description))
        count += 1

    cur.execute("SELECT path, category, subsystem, layer FROM source_files")
    for path, category, subsystem, layer in cur.fetchall():
        for tag, confidence in derive_semantic_tags_for_source(path, category, subsystem, layer).items():
            cur.execute("""INSERT OR IGNORE INTO entity_tags
                           (entity_type, entity_id, tag, confidence, origin)
                           VALUES (?,?,?,?,?)""",
                        ('source_file', path, tag, confidence, 'derived'))
            cur.execute("""INSERT INTO fts_tags(entity_type, entity_id, tag, domain, description)
                           VALUES (?,?,?,?,?)""",
                        ('source_file', path, tag, SEMANTIC_TAGS[tag][0], SEMANTIC_TAGS[tag][1]))
            count += 1

    cur.execute("SELECT name, category, layer FROM c_abi_functions")
    for name, category, layer in cur.fetchall():
        for tag, confidence in derive_semantic_tags_for_abi(name, category, layer).items():
            cur.execute("""INSERT OR IGNORE INTO entity_tags
                           (entity_type, entity_id, tag, confidence, origin)
                           VALUES (?,?,?,?,?)""",
                        ('c_abi_function', name, tag, confidence, 'derived'))
            cur.execute("""INSERT INTO fts_tags(entity_type, entity_id, tag, domain, description)
                           VALUES (?,?,?,?,?)""",
                        ('c_abi_function', name, tag, SEMANTIC_TAGS[tag][0], SEMANTIC_TAGS[tag][1]))
            count += 1

    cur.execute("SELECT path, category FROM docs")
    for path, category in cur.fetchall():
        for tag, confidence in derive_semantic_tags_for_source(path, category, None, 'tool').items():
            cur.execute("""INSERT OR IGNORE INTO entity_tags
                           (entity_type, entity_id, tag, confidence, origin)
                           VALUES (?,?,?,?,?)""",
                        ('doc', path, tag, confidence, 'derived'))
            cur.execute("""INSERT INTO fts_tags(entity_type, entity_id, tag, domain, description)
                           VALUES (?,?,?,?,?)""",
                        ('doc', path, tag, SEMANTIC_TAGS[tag][0], SEMANTIC_TAGS[tag][1]))
            count += 1

    return count

def populate_edges(cur: sqlite3.Cursor):
    """Build cross-reference edges between entities."""
    count = 0
    
    # Test -> source file coverage edges
    test_coverage = {
        'field_52': ['cpu/src/field_52.cpp', 'cpu/src/field.cpp'],
        'field_26': ['cpu/src/field_26.cpp'],
        'selftest': ['cpu/src/selftest.cpp', 'cpu/src/precompute.cpp'],
        'comprehensive': [
            'cpu/src/ecdsa.cpp', 'cpu/src/schnorr.cpp', 'cpu/src/point.cpp',
            'cpu/src/glv.cpp', 'cpu/src/ecmult_gen_comb.cpp', 'cpu/src/precompute.cpp',
        ],
        'exhaustive': [
            'cpu/src/field.cpp', 'cpu/src/scalar.cpp', 'cpu/src/point.cpp',
            'cpu/src/field_asm.cpp', 'cpu/src/field_asm_arm64.cpp',
            'cpu/src/field_asm_riscv64.cpp',
        ],
        'bip340_vectors': ['cpu/src/schnorr.cpp'],
        'bip340_strict': ['cpu/src/schnorr.cpp'],
        'bip32_vectors': ['cpu/src/bip32.cpp'],
        'bip39': ['cpu/src/bip39.cpp'],
        'rfc6979_vectors': ['cpu/src/ecdsa.cpp'],
        'ecc_properties': ['cpu/src/point.cpp', 'cpu/src/scalar.cpp'],
        'ethereum': [
            'cpu/src/ethereum.cpp', 'cpu/src/eth_signing.cpp',
            'cpu/src/keccak256.cpp', 'cpu/src/message_signing.cpp',
        ],
        'zk_proofs': ['cpu/src/zk.cpp', 'cpu/src/pedersen.cpp'],
        'wallet': ['cpu/src/wallet.cpp', 'cpu/src/coin_address.cpp', 'cpu/src/coin_hd.cpp', 'cpu/src/address.cpp'],
        'ct_equivalence': ['cpu/src/ct_field.cpp', 'cpu/src/ct_scalar.cpp', 'cpu/src/ct_point.cpp', 'cpu/src/ct_sign.cpp'],
        'ct_sidechannel': ['cpu/src/ct_sign.cpp', 'cpu/src/ct_point.cpp'],
        'adversarial_protocol': [
            'cpu/src/musig2.cpp', 'cpu/src/frost.cpp', 'cpu/src/adaptor.cpp',
            'cpu/src/ecdsa.cpp', 'cpu/src/ecdh.cpp', 'cpu/src/recovery.cpp',
            # H.2: chacha20-poly1305 AEAD
            'cpu/src/chacha20_poly1305.cpp',
            # H.4: ElligatorSwift / EllSwift
            'cpu/src/ellswift.cpp',
            # H.9: BIP-143 sighash
            'cpu/src/bip143.cpp',
            # H.10: BIP-144 txid / wtxid / witness commitment
            'cpu/src/bip144.cpp',
            # H.11: SegWit program helpers
            'cpu/src/segwit.cpp',
            # H.12 + K.4-K.5: Taproot keypath / tapscript sighash
            'cpu/src/taproot.cpp',
            # K.1-K.3: BIP-324 session protocol
            'cpu/src/bip324.cpp',
            # BIP-324 / EllSwift uses HKDF internally
            'cpu/src/hkdf.cpp',
            # I.5: batch verify paths
            'cpu/src/batch_verify.cpp',
        ],
        'ecies_regression': ['cpu/src/ecies.cpp'],
        'musig2_frost': ['cpu/src/musig2.cpp', 'cpu/src/frost.cpp'],
        'abi_gate': ['cpu/src/ufsecp_impl.cpp'],
        'gpu_bip352_scan': ['cpu/src/ufsecp_gpu_impl.cpp'],
        'gpu_abi_gate': ['cpu/src/ufsecp_gpu_impl.cpp'],
        'gpu_api_negative': ['cpu/src/ufsecp_gpu_impl.cpp'],
        # B-04 monolith split: monolith_split exercises one function from each domain wrapper
        'monolith_split': [
            'cpu/src/ufsecp_impl.cpp',
            'cpu/src/impl/ufsecp_address.cpp',
            'cpu/src/impl/ufsecp_bip322.cpp',
            'cpu/src/impl/ufsecp_coins.cpp',
            'cpu/src/impl/ufsecp_core.cpp',
            'cpu/src/impl/ufsecp_ecdsa.cpp',
            'cpu/src/impl/ufsecp_musig2.cpp',
            'cpu/src/impl/ufsecp_taproot.cpp',
            'cpu/src/impl/ufsecp_zk.cpp',
        ],
        'wycheproof_ecdsa': ['cpu/src/ecdsa.cpp'],
        'wycheproof_ecdh': ['cpu/src/ecdh.cpp'],
        'fault_injection': ['cpu/src/ecdsa.cpp', 'cpu/src/schnorr.cpp'],
        'batch_add_affine': [
            'cpu/src/batch_add_affine.cpp',
            'cpu/src/multiscalar.cpp',
            'cpu/src/pippenger.cpp',
            'cpu/src/batch_verify.cpp',
        ],
        'hash_accel': ['cpu/src/hash_accel.cpp'],
        'message_signing': ['cpu/src/message_signing.cpp'],
    }
    for test, files in test_coverage.items():
        for f in files:
            cur.execute("""INSERT OR IGNORE INTO edges
                (src_type, src_id, dst_type, dst_id, relation)
                VALUES (?,?,?,?,?)""",
                ('test_target', test, 'source_file', f, 'covers'))
            count += 1
    
    # C ABI -> source implementation edges
    # Note: after B-04 monolith split, ABI wrapper layer lives in cpu/src/impl/ufsecp_*.cpp.
    # Map both the deep implementation file and the wrapper file so neither is flagged as uncovered.
    abi_impl = {
        'ecdsa': 'cpu/src/ecdsa.cpp', 'schnorr': 'cpu/src/schnorr.cpp',
        'ecdh': 'cpu/src/ecdh.cpp', 'bip32': 'cpu/src/bip32.cpp',
        'bip39': 'cpu/src/bip39.cpp', 'musig2': 'cpu/src/musig2.cpp',
        'frost': 'cpu/src/frost.cpp', 'adaptor': 'cpu/src/adaptor.cpp',
        'taproot': 'cpu/src/taproot.cpp', 'pedersen': 'cpu/src/pedersen.cpp',
        'zk': 'cpu/src/zk.cpp', 'ecies': 'cpu/src/ecies.cpp',
        'ethereum': 'cpu/src/ethereum.cpp', 'hash': 'cpu/src/hash_accel.cpp',
        'wallet': 'cpu/src/wallet.cpp', 'address': 'cpu/src/address.cpp',
        'multiscalar': 'cpu/src/multiscalar.cpp',
        # B-04 wrapper layer (cpu/src/impl/): categories that don't map to a deep impl file
        'coins': 'cpu/src/impl/ufsecp_coins.cpp',
        'pubkey': 'cpu/src/impl/ufsecp_core.cpp',
        'seckey': 'cpu/src/impl/ufsecp_core.cpp',
        'context': 'cpu/src/impl/ufsecp_core.cpp',
        'other': 'cpu/src/impl/ufsecp_core.cpp',
        'silent_payments': 'cpu/src/impl/ufsecp_core.cpp',
        'error': 'cpu/src/impl/ufsecp_core.cpp',
        'wif': 'cpu/src/impl/ufsecp_core.cpp',
    }
    cur.execute("SELECT name, category FROM c_abi_functions")
    for fname, cat in cur.fetchall():
        target_file = None
        if fname.startswith('ufsecp_gpu_') or fname == 'ufsecp_bip352_prepare_scan_plan':
            target_file = 'cpu/src/ufsecp_gpu_impl.cpp'
        elif cat in abi_impl:
            target_file = abi_impl[cat]

        if target_file:
            cur.execute("""INSERT OR IGNORE INTO edges
                (src_type, src_id, dst_type, dst_id, relation)
                VALUES (?,?,?,?,?)""",
                ('c_abi_function', fname, 'source_file', target_file, 'implements'))
            count += 1
    
    # Include dependency edges
    cur.execute("SELECT source_file, included_file FROM include_deps WHERE is_local=1")
    for src, inc in cur.fetchall():
        cur.execute("""INSERT OR IGNORE INTO edges
            (src_type, src_id, dst_type, dst_id, relation)
            VALUES (?,?,?,?,?)""",
            ('source_file', src, 'source_file', inc, 'includes'))
        count += 1
    
    # ABI routing -> source file edges (routes_through)
    ct_impl_map = {
        'ct::ecdsa_sign': 'cpu/src/ct_sign.cpp',
        'ct::schnorr_sign': 'cpu/src/ct_sign.cpp',
        'ct::generator_mul': 'cpu/src/ct_point.cpp',
        'ct::scalar_mul': 'cpu/src/ct_point.cpp',
        'ct::musig2_partial_sign': 'cpu/src/musig2.cpp',
        'ct::frost_sign': 'cpu/src/frost.cpp',
        'ct::ecdsa_adaptor_sign': 'cpu/src/adaptor.cpp',
        'ct::adaptor_sign': 'cpu/src/adaptor.cpp',
    }
    cur.execute("SELECT abi_function, internal_call, layer FROM abi_routing")
    for abi_fn, internal, layer in cur.fetchall():
        # Link to CT implementation
        for prefix, impl_file in ct_impl_map.items():
            if prefix in (internal or ''):
                cur.execute("""INSERT OR IGNORE INTO edges
                    (src_type, src_id, dst_type, dst_id, relation)
                    VALUES (?,?,?,?,?)""",
                    ('abi_routing', abi_fn, 'source_file', impl_file, 'routes_through'))
                count += 1
                break

    # Binding -> C ABI function edges (wraps)
    cur.execute("SELECT language FROM binding_languages")
    for (lang,) in cur.fetchall():
        cur.execute("SELECT name FROM c_abi_functions")
        for (fn_name,) in cur.fetchall():
            cur.execute("""INSERT OR IGNORE INTO edges
                (src_type, src_id, dst_type, dst_id, relation)
                VALUES (?,?,?,?,?)""",
                ('binding', lang, 'c_abi_function', fn_name, 'wraps'))
            count += 1

    # Security pattern -> source file edges (protects)
    cur.execute("""SELECT DISTINCT source_file FROM security_patterns""")
    for (src_file,) in cur.fetchall():
        cur.execute("""INSERT OR IGNORE INTO edges
            (src_type, src_id, dst_type, dst_id, relation)
            VALUES (?,?,?,?,?)""",
            ('security_pattern', 'ct_protection', 'source_file', src_file, 'protects'))
        count += 1

    return count

def populate_audit_modules(cur: sqlite3.Cursor):
    modules = [
        # Section 1: Mathematical Invariants (13)
        ('audit_field', 'Field Arithmetic Invariants', 'math_invariants', 1),
        ('audit_scalar', 'Scalar Arithmetic Invariants', 'math_invariants', 1),
        ('audit_point', 'Point/Group Law Invariants', 'math_invariants', 1),
        ('mul', 'Multiplication Correctness', 'math_invariants', 1),
        ('arith_correct', 'Arithmetic Correctness', 'math_invariants', 1),
        ('scalar_mul', 'Scalar Multiplication', 'math_invariants', 1),
        ('exhaustive', 'Exhaustive 6-bit Group', 'math_invariants', 1),
        ('comprehensive', 'Comprehensive Operations', 'math_invariants', 1),
        ('ecc_properties', 'ECC Group Properties', 'math_invariants', 1),
        ('batch_add', 'Batch Addition', 'math_invariants', 1),
        ('carry_propagation', 'Carry Propagation', 'math_invariants', 1),
        ('field_52', 'Field 5x52 Representation', 'math_invariants', 1),
        ('field_26', 'Field 10x26 Representation', 'math_invariants', 1),
        # Section 2: CT Analysis (6)
        ('audit_ct', 'CT Layer Coverage', 'ct_analysis', 2),
        ('ct', 'CT Equivalence', 'ct_analysis', 2),
        ('ct_equivalence', 'CT Fast-vs-CT Parity', 'ct_analysis', 2),
        ('ct_sidechannel', 'Dudect Timing Analysis', 'ct_analysis', 2),
        ('ct_verif_formal', 'LLVM Formal CT Verification', 'ct_analysis', 2),
        ('diag_scalar_mul', 'Scalar Mul Diagnostics', 'ct_analysis', 2),
        # Section 3: Differential (4)
        ('differential', 'libsecp256k1 Differential', 'differential', 3),
        ('fiat_crypto', 'Independent Reference Vectors', 'differential', 3),
        ('fiat_crypto_link', 'Independent Reference Linkage', 'differential', 3),
        ('cross_platform_kat', 'Cross-Platform KAT', 'differential', 3),
        # Section 4: Standard Vectors (8)
        ('bip340_vectors', 'BIP-340 Official Vectors', 'vectors', 4),
        ('bip340_strict', 'BIP-340 Strict Encoding', 'vectors', 4),
        ('bip32_vectors', 'BIP-32 HD Key Vectors', 'vectors', 4),
        ('rfc6979_vectors', 'RFC-6979 Deterministic k', 'vectors', 4),
        ('frost_kat', 'FROST Key/Sign Vectors', 'vectors', 4),
        ('musig2_bip327', 'MuSig2 BIP-327 Vectors', 'vectors', 4),
        ('wycheproof_ecdsa', 'Wycheproof ECDSA', 'vectors', 4),
        ('wycheproof_ecdh', 'Wycheproof ECDH', 'vectors', 4),
        # Section 5: Fuzzing (4+)
        ('audit_fuzz', 'Core Fuzzing', 'fuzzing', 5),
        ('fuzz_parsers', 'Parser Fuzzing', 'fuzzing', 5),
        ('fuzz_addr_bip32', 'Address/BIP32/FFI Fuzz', 'fuzzing', 5),
        ('fault_injection', 'Bit-Flip Fault Injection', 'fuzzing', 5),
        # Section 6: Protocol Security (12+)
        ('ecdsa_schnorr', 'ECDSA/Schnorr Roundtrip', 'protocol', 6),
        ('bip32', 'BIP-32 HD Derivation', 'protocol', 6),
        ('bip39', 'BIP-39 Mnemonic', 'protocol', 6),
        ('musig2', 'MuSig2 Protocol', 'protocol', 6),
        ('ecdh_recovery', 'ECDH + Recovery', 'protocol', 6),
        ('v4_features', 'v4 Feature Coverage', 'protocol', 6),
        ('coins', 'Multi-Coin Wallet', 'protocol', 6),
        ('musig2_frost', 'MuSig2+FROST Combined', 'protocol', 6),
        ('musig2_frost_adv', 'MuSig2+FROST Advanced', 'protocol', 6),
        ('audit_integration', 'Integration / Adaptor', 'protocol', 6),
        ('batch_randomness', 'Batch Randomness Quality', 'protocol', 6),
        ('ethereum', 'Ethereum Module', 'protocol', 6),
        # Section 7: Memory Safety (6+)
        ('audit_security', 'Security Hardening', 'memory_safety', 7),
        ('debug_invariants', 'Debug Invariant Checks', 'memory_safety', 7),
        ('abi_gate', 'ABI Gate (symbol check)', 'memory_safety', 7),
        ('ffi_round_trip', 'FFI Round-Trip', 'memory_safety', 7),
        ('adversarial_proto', 'Adversarial Protocol', 'memory_safety', 7),
        ('ecies_regression', 'ECIES Regression', 'memory_safety', 7),
        # Section 8: Performance (4)
        ('hash_accel', 'Hash Acceleration', 'performance', 8),
        ('edge_cases', 'Edge Case Performance', 'performance', 8),
        ('multiscalar', 'Multi-Scalar Ops', 'performance', 8),
        ('audit_perf', 'Performance Validation', 'performance', 8),
    ]
    for mid, name, section, section_no in modules:
        cur.execute("""INSERT OR IGNORE INTO audit_modules
            (module_id, name, section, section_no) VALUES (?,?,?,?)""",
            (mid, name, section, section_no))
    return len(modules)

def populate_cpp_methods(cur: sqlite3.Cursor):
    """Extract public C++ methods from core headers."""
    methods = [
        # FieldElement (fast)
        ('FieldElement', 'zero', 'static FieldElement zero()', 1, 0, 0, 'cpu/include/secp256k1/field.hpp', 35, 'fast'),
        ('FieldElement', 'one', 'static FieldElement one()', 1, 0, 0, 'cpu/include/secp256k1/field.hpp', 36, 'fast'),
        ('FieldElement', 'from_uint64', 'static FieldElement from_uint64(uint64_t)', 1, 0, 0, 'cpu/include/secp256k1/field.hpp', 37, 'fast'),
        ('FieldElement', 'from_limbs', 'static FieldElement from_limbs(const limbs_type&)', 1, 0, 0, 'cpu/include/secp256k1/field.hpp', 38, 'fast'),
        ('FieldElement', 'from_bytes', 'static FieldElement from_bytes(const array<uint8_t,32>&)', 1, 0, 0, 'cpu/include/secp256k1/field.hpp', 39, 'fast'),
        ('FieldElement', 'parse_bytes_strict', 'static bool parse_bytes_strict(const uint8_t*, FieldElement&) noexcept', 1, 0, 1, 'cpu/include/secp256k1/field.hpp', 45, 'fast'),
        ('FieldElement', 'from_mont', 'static FieldElement from_mont(const FieldElement&)', 1, 0, 0, 'cpu/include/secp256k1/field.hpp', 50, 'fast'),
        ('FieldElement', 'from_hex', 'static FieldElement from_hex(const string&)', 1, 0, 0, 'cpu/include/secp256k1/field.hpp', 54, 'fast'),
        ('FieldElement', 'to_bytes', 'array<uint8_t,32> to_bytes() const', 0, 1, 0, 'cpu/include/secp256k1/field.hpp', 58, 'fast'),
        ('FieldElement', 'to_bytes_into', 'void to_bytes_into(uint8_t*) const noexcept', 0, 1, 1, 'cpu/include/secp256k1/field.hpp', 60, 'fast'),
        ('FieldElement', 'to_hex', 'string to_hex() const', 0, 1, 0, 'cpu/include/secp256k1/field.hpp', 61, 'fast'),
        ('FieldElement', 'limbs', 'const limbs_type& limbs() const noexcept', 0, 1, 1, 'cpu/include/secp256k1/field.hpp', 62, 'fast'),
        ('FieldElement', 'limbs_mut', 'limbs_type& limbs_mut() noexcept', 0, 0, 1, 'cpu/include/secp256k1/field.hpp', 66, 'fast'),
        ('FieldElement', 'operator+', 'FieldElement operator+(const FieldElement&) const', 0, 1, 0, 'cpu/include/secp256k1/field.hpp', 74, 'fast'),
        ('FieldElement', 'operator-', 'FieldElement operator-(const FieldElement&) const', 0, 1, 0, 'cpu/include/secp256k1/field.hpp', 75, 'fast'),
        ('FieldElement', 'operator*', 'FieldElement operator*(const FieldElement&) const', 0, 1, 0, 'cpu/include/secp256k1/field.hpp', 76, 'fast'),
        ('FieldElement', 'square', 'FieldElement square() const', 0, 1, 0, 'cpu/include/secp256k1/field.hpp', 77, 'fast'),
        ('FieldElement', 'inverse', 'FieldElement inverse() const', 0, 1, 0, 'cpu/include/secp256k1/field.hpp', 78, 'fast'),
        ('FieldElement', 'sqrt', 'FieldElement sqrt() const', 0, 1, 0, 'cpu/include/secp256k1/field.hpp', 84, 'fast'),
        ('FieldElement', 'negate', 'FieldElement negate(unsigned=1) const', 0, 1, 0, 'cpu/include/secp256k1/field.hpp', 95, 'fast'),
        ('FieldElement', 'square_inplace', 'void square_inplace()', 0, 0, 0, 'cpu/include/secp256k1/field.hpp', 99, 'fast'),
        ('FieldElement', 'inverse_inplace', 'void inverse_inplace()', 0, 0, 0, 'cpu/include/secp256k1/field.hpp', 100, 'fast'),
        ('FieldElement', 'operator==', 'bool operator==(const FieldElement&) const noexcept', 0, 1, 1, 'cpu/include/secp256k1/field.hpp', 102, 'fast'),
        # Scalar (fast)
        ('Scalar', 'zero', 'static Scalar zero()', 1, 0, 0, 'cpu/include/secp256k1/scalar.hpp', 18, 'fast'),
        ('Scalar', 'one', 'static Scalar one()', 1, 0, 0, 'cpu/include/secp256k1/scalar.hpp', 19, 'fast'),
        ('Scalar', 'from_uint64', 'static Scalar from_uint64(uint64_t)', 1, 0, 0, 'cpu/include/secp256k1/scalar.hpp', 20, 'fast'),
        ('Scalar', 'from_limbs', 'static Scalar from_limbs(const limbs_type&)', 1, 0, 0, 'cpu/include/secp256k1/scalar.hpp', 21, 'fast'),
        ('Scalar', 'from_bytes', 'static Scalar from_bytes(const array<uint8_t,32>&)', 1, 0, 0, 'cpu/include/secp256k1/scalar.hpp', 22, 'fast'),
        ('Scalar', 'parse_bytes_strict', 'static bool parse_bytes_strict(const uint8_t*, Scalar&) noexcept', 1, 0, 1, 'cpu/include/secp256k1/scalar.hpp', 27, 'fast'),
        ('Scalar', 'parse_bytes_strict_nonzero', 'static bool parse_bytes_strict_nonzero(const uint8_t*, Scalar&) noexcept', 1, 0, 1, 'cpu/include/secp256k1/scalar.hpp', 31, 'fast'),
        ('Scalar', 'from_hex', 'static Scalar from_hex(const string&)', 1, 0, 0, 'cpu/include/secp256k1/scalar.hpp', 37, 'fast'),
        ('Scalar', 'to_bytes', 'array<uint8_t,32> to_bytes() const', 0, 1, 0, 'cpu/include/secp256k1/scalar.hpp', 39, 'fast'),
        ('Scalar', 'to_hex', 'string to_hex() const', 0, 1, 0, 'cpu/include/secp256k1/scalar.hpp', 40, 'fast'),
        ('Scalar', 'limbs', 'const limbs_type& limbs() const noexcept', 0, 1, 1, 'cpu/include/secp256k1/scalar.hpp', 41, 'fast'),
        ('Scalar', 'operator+', 'Scalar operator+(const Scalar&) const', 0, 1, 0, 'cpu/include/secp256k1/scalar.hpp', 43, 'fast'),
        ('Scalar', 'operator-', 'Scalar operator-(const Scalar&) const', 0, 1, 0, 'cpu/include/secp256k1/scalar.hpp', 44, 'fast'),
        ('Scalar', 'operator*', 'Scalar operator*(const Scalar&) const', 0, 1, 0, 'cpu/include/secp256k1/scalar.hpp', 45, 'fast'),
        ('Scalar', 'is_zero', 'bool is_zero() const noexcept', 0, 1, 1, 'cpu/include/secp256k1/scalar.hpp', 51, 'fast'),
        ('Scalar', 'operator==', 'bool operator==(const Scalar&) const noexcept', 0, 1, 1, 'cpu/include/secp256k1/scalar.hpp', 52, 'fast'),
        ('Scalar', 'inverse', 'Scalar inverse() const', 0, 1, 0, 'cpu/include/secp256k1/scalar.hpp', 56, 'fast'),
        ('Scalar', 'negate', 'Scalar negate() const', 0, 1, 0, 'cpu/include/secp256k1/scalar.hpp', 60, 'fast'),
        ('Scalar', 'is_even', 'bool is_even() const noexcept', 0, 1, 1, 'cpu/include/secp256k1/scalar.hpp', 63, 'fast'),
        ('Scalar', 'bit', 'uint8_t bit(size_t) const', 0, 1, 0, 'cpu/include/secp256k1/scalar.hpp', 76, 'fast'),
        ('Scalar', 'to_wnaf', 'vector<int8_t> to_wnaf(unsigned) const', 0, 1, 0, 'cpu/include/secp256k1/scalar.hpp', 95, 'fast'),
        # Point (fast)
        ('Point', 'generator', 'static Point generator()', 1, 0, 0, 'cpu/include/secp256k1/point.hpp', 84, 'fast'),
        ('Point', 'infinity', 'static Point infinity()', 1, 0, 0, 'cpu/include/secp256k1/point.hpp', 85, 'fast'),
        ('Point', 'from_affine', 'static Point from_affine(const FieldElement&, const FieldElement&)', 1, 0, 0, 'cpu/include/secp256k1/point.hpp', 86, 'fast'),
        ('Point', 'from_hex', 'static Point from_hex(const string&, const string&)', 1, 0, 0, 'cpu/include/secp256k1/point.hpp', 90, 'fast'),
        ('Point', 'x', 'FieldElement x() const', 0, 1, 0, 'cpu/include/secp256k1/point.hpp', 93, 'fast'),
        ('Point', 'y', 'FieldElement y() const', 0, 1, 0, 'cpu/include/secp256k1/point.hpp', 94, 'fast'),
        ('Point', 'is_infinity', 'bool is_infinity() const noexcept', 0, 1, 1, 'cpu/include/secp256k1/point.hpp', 95, 'fast'),
        ('Point', 'add', 'Point add(const Point&) const', 0, 1, 0, 'cpu/include/secp256k1/point.hpp', 120, 'fast'),
        ('Point', 'dbl', 'Point dbl() const', 0, 1, 0, 'cpu/include/secp256k1/point.hpp', 121, 'fast'),
        ('Point', 'scalar_mul', 'Point scalar_mul(const Scalar&) const', 0, 1, 0, 'cpu/include/secp256k1/point.hpp', 122, 'fast'),
        ('Point', 'scalar_mul_precomputed_k', 'Point scalar_mul_precomputed_k(const Scalar&) const', 0, 1, 0, 'cpu/include/secp256k1/point.hpp', 126, 'fast'),
        ('Point', 'negate', 'Point negate() const', 0, 1, 0, 'cpu/include/secp256k1/point.hpp', 145, 'fast'),
        ('Point', 'next', 'Point next() const', 0, 1, 0, 'cpu/include/secp256k1/point.hpp', 148, 'fast'),
        ('Point', 'prev', 'Point prev() const', 0, 1, 0, 'cpu/include/secp256k1/point.hpp', 149, 'fast'),
        ('Point', 'add_inplace', 'void add_inplace(const Point&)', 0, 0, 0, 'cpu/include/secp256k1/point.hpp', 154, 'fast'),
        # ECDSASignature
        ('ECDSASignature', 'to_der', 'pair<array<uint8_t,72>,size_t> to_der() const', 0, 1, 0, 'cpu/include/secp256k1/ecdsa.hpp', 32, 'fast'),
        ('ECDSASignature', 'to_compact', 'array<uint8_t,64> to_compact() const', 0, 1, 0, 'cpu/include/secp256k1/ecdsa.hpp', 35, 'fast'),
        ('ECDSASignature', 'from_compact', 'static ECDSASignature from_compact(const uint8_t*)', 1, 0, 0, 'cpu/include/secp256k1/ecdsa.hpp', 38, 'fast'),
        ('ECDSASignature', 'parse_compact_strict', 'static bool parse_compact_strict(const uint8_t*, ECDSASignature&) noexcept', 1, 0, 1, 'cpu/include/secp256k1/ecdsa.hpp', 42, 'fast'),
        ('ECDSASignature', 'normalize', 'ECDSASignature normalize() const', 0, 1, 0, 'cpu/include/secp256k1/ecdsa.hpp', 46, 'fast'),
        ('ECDSASignature', 'is_low_s', 'bool is_low_s() const', 0, 1, 0, 'cpu/include/secp256k1/ecdsa.hpp', 49, 'fast'),
        # Free functions
        ('', 'ecdsa_sign', 'ECDSASignature ecdsa_sign(const array<uint8_t,32>&, const fast::Scalar&)', 0, 0, 0, 'cpu/include/secp256k1/ecdsa.hpp', 57, 'fast'),
        ('', 'ecdsa_sign_verified', 'ECDSASignature ecdsa_sign_verified(const array<uint8_t,32>&, const fast::Scalar&)', 0, 0, 0, 'cpu/include/secp256k1/ecdsa.hpp', 62, 'fast'),
        ('', 'ecdsa_verify', 'bool ecdsa_verify(const uint8_t*, const fast::Point&, const ECDSASignature&)', 0, 0, 0, 'cpu/include/secp256k1/ecdsa.hpp', 81, 'fast'),
        ('', 'rfc6979_nonce', 'fast::Scalar rfc6979_nonce(const fast::Scalar&, const array<uint8_t,32>&)', 0, 0, 0, 'cpu/include/secp256k1/ecdsa.hpp', 95, 'fast'),
        ('', 'schnorr_sign', 'SchnorrSignature schnorr_sign(const array<uint8_t,32>&, const fast::Scalar&)', 0, 0, 0, 'cpu/include/secp256k1/schnorr.hpp', 77, 'fast'),
        ('', 'schnorr_verify', 'bool schnorr_verify(const array<uint8_t,32>&, const array<uint8_t,32>&, const SchnorrSignature&)', 0, 0, 0, 'cpu/include/secp256k1/schnorr.hpp', 82, 'fast'),
        ('', 'generate_schnorr_keypair', 'SchnorrKeypair generate_schnorr_keypair(const fast::Scalar&)', 0, 0, 0, 'cpu/include/secp256k1/schnorr.hpp', 63, 'fast'),
        # CT layer  
        ('', 'ct::ecdsa_sign', 'ECDSASignature ct::ecdsa_sign(const array<uint8_t,32>&, const Scalar&)', 0, 0, 0, 'cpu/include/secp256k1/ct/sign.hpp', 0, 'ct'),
        ('', 'ct::schnorr_sign', 'SchnorrSignature ct::schnorr_sign(const array<uint8_t,32>&, const Scalar&)', 0, 0, 0, 'cpu/include/secp256k1/ct/sign.hpp', 0, 'ct'),
        ('', 'ct::generator_mul', 'Point ct::generator_mul(const Scalar&)', 0, 0, 0, 'cpu/include/secp256k1/ct/point.hpp', 0, 'ct'),
        ('', 'ct::scalar_mul', 'Point ct::scalar_mul(const Point&, const Scalar&)', 0, 0, 0, 'cpu/include/secp256k1/ct/point.hpp', 0, 'ct'),
        # SHA256
        ('SHA256', 'update', 'void update(const uint8_t*, size_t)', 0, 0, 0, 'cpu/include/secp256k1/sha256.hpp', 0, 'fast'),
        ('SHA256', 'finalize', 'array<uint8_t,32> finalize()', 0, 0, 0, 'cpu/include/secp256k1/sha256.hpp', 0, 'fast'),
        ('SHA256', 'digest', 'static array<uint8_t,32> digest(const uint8_t*, size_t)', 1, 0, 0, 'cpu/include/secp256k1/sha256.hpp', 0, 'fast'),
    ]
    for cls, meth, sig, is_s, is_c, is_n, hdr, ln, layer in methods:
        cur.execute("""INSERT OR IGNORE INTO cpp_methods
            (class_name, method, signature, is_static, is_const, is_noexcept, header_path, line_no, layer)
            VALUES (?,?,?,?,?,?,?,?,?)""",
            (cls, meth, sig, is_s, is_c, is_n, hdr, ln, layer))
    return len(methods)

def populate_security_patterns(cur: sqlite3.Cursor):
    """Scan source for secure_erase, value_barrier, CLASSIFY/DECLASSIFY."""
    cur.execute("DELETE FROM security_patterns")
    patterns_to_scan = [
        ('secure_erase', re.compile(r'secure_erase\s*\(')),
        ('value_barrier', re.compile(r'value_barrier\s*\(')),
        ('CLASSIFY', re.compile(r'SECP256K1_CLASSIFY\s*\(')),
        ('DECLASSIFY', re.compile(r'SECP256K1_DECLASSIFY\s*\(')),
    ]
    scan_dirs = ['cpu/src', 'cpu/include', 'include/ufsecp', 'audit']
    count = 0
    for scan_dir in scan_dirs:
        dirpath = LIB_ROOT / scan_dir
        if not dirpath.exists():
            continue
        for root, dirs, files in os.walk(dirpath):
            dirs[:] = [d for d in dirs if not should_skip_dir(d)]
            for fname in files:
                ext = os.path.splitext(fname)[1].lower()
                if ext not in ('.cpp', '.hpp', '.h'):
                    continue
                filepath = Path(root) / fname
                rel = str(filepath.relative_to(LIB_ROOT))
                try:
                    with open(filepath, 'r', errors='replace') as f:
                        for i, line in enumerate(f, 1):
                            # Skip #include and comment-only lines
                            stripped = line.strip()
                            if stripped.startswith('#include') or stripped.startswith('//'):
                                # Allow CLASSIFY/DECLASSIFY defines though
                                if 'CLASSIFY' not in stripped and 'DECLASSIFY' not in stripped:
                                    continue
                            for pat_name, pat_re in patterns_to_scan:
                                if pat_re.search(line):
                                    ctx = stripped[:120]
                                    cur.execute("""INSERT INTO security_patterns
                                        (pattern, source_file, line_no, context)
                                        VALUES (?,?,?,?)""",
                                        (pat_name, rel, i, ctx))
                                    count += 1
                except Exception:
                    pass
    return count

def populate_abi_routing(cur: sqlite3.Cursor):
    """Map each ufsecp_* function to the internal call it makes."""
    routing = [
        # Context
        ('ufsecp_ctx_create', 'run_selftest + alloc', 'both', 223),
        ('ufsecp_ctx_clone', 'memcpy ctx', 'fast', 244),
        ('ufsecp_last_error', 'ctx->last_err', 'fast', 262),
        # Seckey (all CT)
        ('ufsecp_seckey_verify', 'Scalar::parse_bytes_strict_nonzero', 'ct', 279),
        ('ufsecp_seckey_negate', 'Scalar::negate', 'ct', 292),
        ('ufsecp_seckey_tweak_add', 'ct scalar add + validate', 'ct', 307),
        ('ufsecp_seckey_tweak_mul', 'ct scalar mul + validate', 'ct', 331),
        # Pubkey (CT for create, fast for parse/verify)
        ('ufsecp_pubkey_create', 'ct::generator_mul(sk)', 'ct', 375),
        ('ufsecp_pubkey_create_uncompressed', 'ct::generator_mul(sk)', 'ct', 387),
        ('ufsecp_pubkey_parse', 'point_from_compressed + on_curve', 'fast', 400),
        ('ufsecp_pubkey_xonly', 'schnorr_pubkey(sk)', 'ct', 440),
        ('ufsecp_pubkey_add', 'Point::add', 'fast', 1232),
        ('ufsecp_pubkey_negate', 'Point::negate', 'fast', 1251),
        ('ufsecp_pubkey_tweak_add', 'Point::add(gen_mul(tweak))', 'fast', 1264),
        ('ufsecp_pubkey_tweak_mul', 'Point::scalar_mul(tweak)', 'fast', 1284),
        ('ufsecp_pubkey_combine', 'multi-point add', 'fast', 1303),
        # ECDSA (CT sign, fast verify)
        ('ufsecp_ecdsa_sign', 'ct::ecdsa_sign(msg, sk)', 'ct', 461),
        ('ufsecp_ecdsa_sign_verified', 'ct::ecdsa_sign + ecdsa_verify', 'ct', 483),
        ('ufsecp_ecdsa_verify', 'ecdsa_verify(msg, pubkey, sig)', 'fast', 504),
        ('ufsecp_ecdsa_sig_to_der', 'ECDSASignature::to_der', 'fast', 531),
        ('ufsecp_ecdsa_sig_from_der', 'DER parse', 'fast', 555),
        ('ufsecp_ecdsa_sign_recoverable', 'ct::ecdsa_sign + recovery_id', 'ct', 675),
        ('ufsecp_ecdsa_recover', 'ecrecover(msg, sig, v)', 'fast', 704),
        # Schnorr (CT sign, fast verify)
        ('ufsecp_schnorr_sign', 'ct::schnorr_sign(msg, sk)', 'ct', 739),
        ('ufsecp_schnorr_sign_verified', 'ct::schnorr_sign + schnorr_verify', 'ct', 767),
        ('ufsecp_schnorr_verify', 'schnorr_verify(msg, xpub, sig)', 'fast', 795),
        ('ufsecp_schnorr_keypair', 'generate_schnorr_keypair(sk)', 'ct', 0),
        # ECDH (CT)
        ('ufsecp_ecdh', 'ct::scalar_mul(pubkey, sk)', 'ct', 843),
        ('ufsecp_ecdh_xonly', 'ct::scalar_mul + x-only output', 'ct', 859),
        ('ufsecp_ecdh_raw', 'ct::scalar_mul + raw output', 'ct', 875),
        # Hash (fast, no secrets)
        ('ufsecp_sha256', 'SHA256::digest(data, len)', 'fast', 895),
        ('ufsecp_hash160', 'SHA256+RIPEMD160', 'fast', 905),
        ('ufsecp_tagged_hash', 'tagged_hash(tag, msg)', 'fast', 913),
        ('ufsecp_sha512', 'SHA512::digest(data, len)', 'fast', 1488),
        # Address (fast, public)
        ('ufsecp_addr_p2pkh', 'hash160 + base58check', 'fast', 926),
        ('ufsecp_addr_p2wpkh', 'hash160 + bech32', 'fast', 947),
        ('ufsecp_addr_p2tr', 'taproot_output_key + bech32m', 'fast', 968),
        ('ufsecp_wif_encode', 'base58check(privkey)', 'fast', 992),
        ('ufsecp_wif_decode', 'base58check_decode + validate', 'fast', 1016),
        # BIP-32 (CT for secret derivation)
        ('ufsecp_bip32_master', 'HMAC-SHA512(seed)', 'ct', 1067),
        ('ufsecp_bip32_derive', 'CKD_priv or CKD_pub', 'ct', 1088),
        ('ufsecp_bip32_derive_path', 'multi-level CKD', 'ct', 1109),
        ('ufsecp_bip32_privkey', 'ExtendedKey::privkey()', 'ct', 1130),
        ('ufsecp_bip32_pubkey', 'ExtendedKey::pubkey()', 'fast', 1149),
        # Taproot
        ('ufsecp_taproot_output_key', 'taproot_output_key(xpub, merkle)', 'fast', 1165),
        ('ufsecp_taproot_tweak_seckey', 'taproot_tweak_seckey(sk, merkle)', 'ct', 1185),
        ('ufsecp_taproot_verify', 'taproot_verify(xpub, merkle, output)', 'fast', 1209),
        # BIP-39
        ('ufsecp_bip39_generate', 'bip39_generate(strength)', 'ct', 1329),
        ('ufsecp_bip39_validate', 'bip39_validate(mnemonic)', 'fast', 1349),
        ('ufsecp_bip39_to_seed', 'PBKDF2-SHA512(mnemonic, passphrase)', 'ct', 1357),
        ('ufsecp_bip39_to_entropy', 'bip39_to_entropy(mnemonic)', 'fast', 1371),
        # Batch (fast)
        ('ufsecp_schnorr_batch_verify', 'batch_schnorr_verify(entries)', 'fast', 1391),
        ('ufsecp_ecdsa_batch_verify', 'batch_ecdsa_verify(entries)', 'fast', 1414),
        ('ufsecp_schnorr_batch_identify_invalid', 'bisection invalid finder', 'fast', 1437),
        ('ufsecp_ecdsa_batch_identify_invalid', 'bisection invalid finder', 'fast', 1460),
        # Multi-scalar
        ('ufsecp_shamir_trick', 'shamir_trick(a*G + b*P)', 'fast', 1500),
        ('ufsecp_multi_scalar_mul', 'pippenger_msm(points, scalars)', 'fast', 1524),
        # MuSig2 (mixed)
        ('ufsecp_musig2_key_agg', 'musig2_key_agg(pubkeys)', 'fast', 1552),
        ('ufsecp_musig2_nonce_gen', 'musig2_nonce_gen(sk)', 'ct', 1575),
        ('ufsecp_musig2_nonce_agg', 'musig2_nonce_agg(nonces)', 'fast', 1607),
        ('ufsecp_musig2_partial_sign', 'ct::musig2_partial_sign(sk)', 'ct', 0),
        ('ufsecp_musig2_partial_verify', 'musig2_partial_verify', 'fast', 0),
        ('ufsecp_musig2_aggregate', 'musig2_aggregate(partials)', 'fast', 0),
        ('ufsecp_musig2_start_sign_session', 'musig2_session_init', 'ct', 1628),
        # FROST (mixed)
        ('ufsecp_frost_keygen_begin', 'frost_keygen_begin', 'ct', 0),
        ('ufsecp_frost_keygen_finalize', 'frost_keygen_finalize', 'fast', 0),
        ('ufsecp_frost_sign_nonce_gen', 'frost_sign_nonce_gen', 'ct', 0),
        ('ufsecp_frost_sign', 'ct::frost_sign(sk, nonce)', 'ct', 0),
        ('ufsecp_frost_verify_partial', 'frost_verify_partial', 'fast', 0),
        ('ufsecp_frost_aggregate', 'frost_aggregate(partials)', 'fast', 0),
        # Adaptor
        ('ufsecp_schnorr_adaptor_sign', 'ct::adaptor_sign(sk)', 'ct', 0),
        ('ufsecp_schnorr_adaptor_verify', 'adaptor_verify', 'fast', 0),
        ('ufsecp_schnorr_adaptor_adapt', 'adaptor_adapt(pre_sig, secret)', 'ct', 0),
        ('ufsecp_schnorr_adaptor_extract', 'adaptor_extract(sig, pre_sig)', 'fast', 0),
        ('ufsecp_ecdsa_adaptor_sign', 'ct::ecdsa_adaptor_sign(sk)', 'ct', 0),
        ('ufsecp_ecdsa_adaptor_verify', 'ecdsa_adaptor_verify', 'fast', 0),
        ('ufsecp_ecdsa_adaptor_adapt', 'ecdsa_adaptor_adapt', 'ct', 0),
        ('ufsecp_ecdsa_adaptor_extract', 'ecdsa_adaptor_extract', 'fast', 0),
        # Pedersen + ZK
        ('ufsecp_pedersen_commit', 'pedersen_commit(v, r)', 'fast', 0),
        ('ufsecp_pedersen_verify', 'pedersen_verify(C, v, r)', 'fast', 0),
        ('ufsecp_pedersen_blind_sum', 'blind factor sum', 'ct', 0),
        ('ufsecp_pedersen_verify_tally', 'verify_tally(inputs, outputs)', 'fast', 0),
        ('ufsecp_pedersen_switch_commit', 'switch_commit(v, r)', 'fast', 0),
        ('ufsecp_zk_range_proof_create', 'create_range_proof', 'ct', 0),
        ('ufsecp_zk_range_proof_verify', 'verify_range_proof', 'fast', 0),
        ('ufsecp_zk_knowledge_prove', 'prove_knowledge(sk)', 'ct', 0),
        ('ufsecp_zk_knowledge_verify', 'verify_knowledge(proof, P)', 'fast', 0),
        ('ufsecp_zk_dleq_prove', 'dleq_prove(sk)', 'ct', 0),
        ('ufsecp_zk_dleq_verify', 'dleq_verify(proof)', 'fast', 0),
        # Wallet / Coins
        ('ufsecp_coin_address', 'coin_address(coin, pubkey)', 'fast', 0),
        ('ufsecp_coin_address_validate', 'coin_address_validate(coin, addr)', 'fast', 0),
        ('ufsecp_coin_hd_derive', 'coin_hd_derive(coin, xprv, path)', 'ct', 0),
        ('ufsecp_btc_message_sign', 'btc_message_sign(msg, sk)', 'ct', 0),
        ('ufsecp_btc_message_verify', 'btc_message_verify(msg, sig, addr)', 'fast', 0),
        ('ufsecp_coin_params', 'get coin_params(coin_id)', 'fast', 0),
        # Silent Payments
        ('ufsecp_silent_payment_create_output', 'silent_payment_create_output', 'ct', 0),
        ('ufsecp_silent_payment_scan', 'silent_payment_scan', 'ct', 0),
        ('ufsecp_silent_payment_verify_label', 'silent_payment_verify_label', 'fast', 0),
        # ECIES
        ('ufsecp_ecies_encrypt', 'ecies_encrypt(pubkey, msg)', 'ct', 0),
        ('ufsecp_ecies_decrypt', 'ecies_decrypt(sk, ciphertext)', 'ct', 0),
        # Ethereum
        ('ufsecp_keccak256', 'keccak256(data)', 'fast', 0),
        ('ufsecp_eth_sign', 'ct::ecdsa_sign(keccak(msg), sk) + v', 'ct', 0),
        ('ufsecp_eth_recover', 'ecrecover(keccak(msg), sig, v)', 'fast', 0),
        ('ufsecp_eth_address', 'keccak256(pubkey)[12:]', 'fast', 0),
        ('ufsecp_eth_eip55_checksum', 'eip55_checksum(addr)', 'fast', 0),
        ('ufsecp_eth_typed_data_hash', 'eip712_hash(domain, msg)', 'fast', 0),
    ]
    for abi_fn, internal, layer, line in routing:
        cur.execute("""INSERT OR IGNORE INTO abi_routing
            (abi_function, internal_call, layer, impl_line)
            VALUES (?,?,?,?)""",
            (abi_fn, internal, layer, line if line else None))
    return len(routing)

def populate_binding_languages(cur: sqlite3.Cursor):
    """Insert binding language metadata."""
    bindings = [
        ('C', 'bindings/c_api', 5, 'stable', 'ufsecp', 'direct'),
        ('Python', 'bindings/python', 7, 'active', 'ufsecp', 'cffi'),
        ('Rust', 'bindings/rust', 13, 'active', 'ufsecp-sys', 'FFI'),
        ('Swift', 'bindings/swift', 8, 'active', 'UltrafastSecp256k1', 'C interop'),
        ('Go', 'bindings/go', 5, 'active', 'ufsecp', 'cgo'),
        ('Java', 'bindings/java', 12, 'supported', 'com.ultrafast.secp256k1', 'JNI'),
        ('C#', 'bindings/csharp', 9, 'supported', 'UltrafastSecp256k1', 'P/Invoke'),
        ('Node.js', 'bindings/nodejs', 7, 'supported', '@ultrafast/secp256k1', 'N-API'),
        ('Dart', 'bindings/dart', 9, 'supported', 'ufsecp', 'dart:ffi'),
        ('PHP', 'bindings/php', 6, 'community', 'ufsecp', 'FFI'),
        ('Ruby', 'bindings/ruby', 6, 'community', 'ufsecp', 'FFI'),
        ('React Native', 'bindings/react-native', 15, 'experimental', '@ultrafast/react-native-secp256k1', 'JSI'),
        ('WASM', 'wasm', 0, 'experimental', 'ufsecp-wasm', 'Emscripten'),
    ]
    for lang, directory, fc, status, pkg, ffi in bindings:
        cur.execute("""INSERT OR IGNORE INTO binding_languages
            (language, directory, file_count, status, package_name, ffi_method)
            VALUES (?,?,?,?,?,?)""",
            (lang, directory, fc, status, pkg, ffi))
    return len(bindings)

def populate_macros(cur: sqlite3.Cursor):
    """Insert known compile-time macros and defines."""
    macros = [
        # Platform guards
        ('SECP256K1_FAST_52BIT', '1', 'cpu/include/secp256k1/point.hpp', 10, 'platform_guard'),
        ('SECP256K1_PLATFORM_X86_64', None, 'cpu/include/secp256k1/config.hpp', 0, 'platform_guard'),
        ('__SIZEOF_INT128__', None, 'cpu/src/field.cpp', 0, 'platform_guard'),
        # Size constants
        ('UFSECP_PRIVKEY_LEN', '32', 'include/ufsecp/ufsecp.h', 45, 'size_constant'),
        ('UFSECP_PUBKEY_COMPRESSED_LEN', '33', 'include/ufsecp/ufsecp.h', 46, 'size_constant'),
        ('UFSECP_PUBKEY_UNCOMPRESSED_LEN', '65', 'include/ufsecp/ufsecp.h', 47, 'size_constant'),
        ('UFSECP_PUBKEY_XONLY_LEN', '32', 'include/ufsecp/ufsecp.h', 48, 'size_constant'),
        ('UFSECP_SIG_COMPACT_LEN', '64', 'include/ufsecp/ufsecp.h', 49, 'size_constant'),
        ('UFSECP_SIG_DER_MAX_LEN', '72', 'include/ufsecp/ufsecp.h', 50, 'size_constant'),
        ('UFSECP_HASH_LEN', '32', 'include/ufsecp/ufsecp.h', 51, 'size_constant'),
        ('UFSECP_HASH160_LEN', '20', 'include/ufsecp/ufsecp.h', 52, 'size_constant'),
        ('UFSECP_SHARED_SECRET_LEN', '32', 'include/ufsecp/ufsecp.h', 53, 'size_constant'),
        ('UFSECP_BIP32_SERIALIZED_LEN', '78', 'include/ufsecp/ufsecp.h', 54, 'size_constant'),
        ('UFSECP_NET_MAINNET', '0', 'include/ufsecp/ufsecp.h', 57, 'size_constant'),
        ('UFSECP_NET_TESTNET', '1', 'include/ufsecp/ufsecp.h', 58, 'size_constant'),
        # Feature flags
        ('SECP256K1_CT_VALGRIND', '1', 'cpu/include/secp256k1/ct/ops.hpp', 0, 'feature_flag'),
        ('SECP256K1_BUILD_ETHEREUM', None, 'CMakeLists.txt', 0, 'feature_flag'),
        ('UFSECP_BITCOIN_STRICT', None, 'CMakeLists.txt', 0, 'feature_flag'),
        ('SECP256K1_USE_ASM', None, 'CMakeLists.txt', 0, 'feature_flag'),
        # CT markers
        ('SECP256K1_CLASSIFY', 'VALGRIND_MAKE_MEM_UNDEFINED', 'cpu/include/secp256k1/ct/ops.hpp', 49, 'ct_marker'),
        ('SECP256K1_DECLASSIFY', 'VALGRIND_MAKE_MEM_DEFINED', 'cpu/include/secp256k1/ct/ops.hpp', 50, 'ct_marker'),
        # Debug
        ('SCALED', 'ternary(normal,reduced)', 'cpu/include/secp256k1/sanitizer_scale.hpp', 37, 'debug'),
        ('SECP256K1_INIT', 'init_macro', 'cpu/include/secp256k1/init.hpp', 42, 'debug'),
        ('SECP256K1_INIT_VERBOSE', 'verbose_init_macro', 'cpu/include/secp256k1/init.hpp', 45, 'debug'),
    ]
    for name, val, fp, ln, cat in macros:
        cur.execute("""INSERT OR IGNORE INTO macros
            (name, value, file_path, line_no, category)
            VALUES (?,?,?,?,?)""",
            (name, val, fp, ln, cat))
    return len(macros)


# ---------------------------------------------------------------------------
# FILE SUMMARIES  --  one-line descriptions for token-efficient agent context
# ---------------------------------------------------------------------------
def populate_file_summaries(cur):
    """Hardcoded one-line descriptions for core source files."""
    summaries = [
        # ---- cpu/src core implementations ----
        ('cpu/src/field.cpp', 'Fast-layer 4-limb/5-limb field element arithmetic (mul, sqr, inv, add, reduce) mod secp256k1 prime p'),
        ('cpu/src/field_52.cpp', 'FE52 5x52-bit field representation: mul/sqr via 128-bit intermediates'),
        ('cpu/src/field_26.cpp', 'FE26 10x26-bit field representation for platforms without __int128'),
        ('cpu/src/field_asm.cpp', 'x86-64 inline assembly field_mul/field_sqr using mulx/adcx/adox'),
        ('cpu/src/field_asm_arm64.cpp', 'ARM64 NEON inline assembly for field arithmetic'),
        ('cpu/src/field_asm_riscv64.cpp', 'RISC-V 64-bit inline assembly for field multiplication'),
        ('cpu/src/field_asm_x64.asm', 'Standalone x86-64 NASM assembly for field_mul_asm/field_sqr_asm'),
        ('cpu/src/scalar.cpp', 'Scalar arithmetic mod group order n: mul, inv (modinv64), add, negate, split_lambda (GLV)'),
        ('cpu/src/point.cpp', 'Jacobian point operations: add, double, mul (GLV+comb), to_affine, batch_normalize'),
        ('cpu/src/ecdsa.cpp', 'Fast-layer ECDSA: sign (RFC-6979 nonce), verify (Shamir multi-scalar), DER/compact parse'),
        ('cpu/src/schnorr.cpp', 'Fast-layer BIP-340 Schnorr: keypair_create, sign, verify, batch_verify'),
        ('cpu/src/precompute.cpp', 'Comb table precomputation for k*G: build/serialize 16K-entry generator table'),
        ('cpu/src/ecmult_gen_comb.cpp', 'Comb-based ecmult_gen: fixed-base scalar multiplication using precomputed table'),
        ('cpu/src/glv.cpp', 'GLV endomorphism: lambda/beta constants, scalar decomposition, w-NAF recoding'),
        ('cpu/src/selftest.cpp', 'Runtime self-test suite (21 modules): field, scalar, point, ECDSA, Schnorr, BIP-32/39'),
        ('cpu/src/hash_accel.cpp', 'Hardware-accelerated SHA-256: SHA-NI (x86), SHA2 (ARM), Zba (RISC-V) dispatch'),
        ('cpu/src/recovery.cpp', 'ECDSA public key recovery from signature + recovery id (ecrecover)'),
        ('cpu/src/adaptor.cpp', 'Adaptor signatures: pre-sign, adapt, extract for atomic swaps'),
        ('cpu/src/musig2.cpp', 'MuSig2 multi-signature protocol: key aggregation, nonce gen, partial sign, combine'),
        ('cpu/src/frost.cpp', 'FROST threshold signatures: key generation, signing shares, aggregation'),
        ('cpu/src/taproot.cpp', 'BIP-341 Taproot: tweak_pubkey, check_output, script path spending'),
        ('cpu/src/bip32.cpp', 'BIP-32 HD key derivation: master_key_generate, ckd_priv, ckd_pub, path parsing'),
        ('cpu/src/bip39.cpp', 'BIP-39 mnemonic: entropy_to_mnemonic, mnemonic_to_seed, wordlist validation'),
        ('cpu/src/ecdh.cpp', 'ECDH key exchange: shared_secret = SHA-256(x-only of k*P)'),
        ('cpu/src/ecies.cpp', 'ECIES encryption/decryption: ephemeral ECDH + AES-256-GCM'),
        ('cpu/src/pedersen.cpp', 'Pedersen commitments: commit(v,r) = v*H + r*G, verify, add/sub homomorphic'),
        ('cpu/src/multiscalar.cpp', 'Multi-scalar multiplication: Strauss/Shamir for a*G + b*P and vectorized sums'),
        ('cpu/src/pippenger.cpp', 'Pippenger bucket method: multi-scalar multiplication for large input vectors'),
        ('cpu/src/zk.cpp', 'Zero-knowledge proofs: range proofs, Schnorr DLEQ, Pedersen opening proofs'),
        ('cpu/src/address.cpp', 'Address generation: P2PKH, P2SH, Bech32/Bech32m encoding, RIPEMD-160'),
        ('cpu/src/wallet.cpp', 'High-level wallet: derive path, sign tx, serialize privkey/pubkey'),
        ('cpu/src/coin_address.cpp', 'Multi-coin address derivation with coin-specific parameters'),
        ('cpu/src/coin_hd.cpp', 'Multi-coin HD key derivation with slip44 coin types'),
        ('cpu/src/message_signing.cpp', 'Bitcoin message signing/verification (BIP-137 format)'),
        ('cpu/src/eth_signing.cpp', 'Ethereum transaction signing: EIP-155 chain ID, v/r/s encoding'),
        ('cpu/src/ethereum.cpp', 'Ethereum address derivation: Keccak-256 of uncompressed pubkey'),
        ('cpu/src/keccak256.cpp', 'Keccak-256 hash implementation for Ethereum compatibility'),
        ('cpu/src/batch_add_affine.cpp', 'Batch affine addition: Montgomery batch inversion for parallel point adds'),
        ('cpu/src/batch_verify.cpp', 'Batch ECDSA/Schnorr verification: random-linear-combination method'),
        # ---- CT (constant-time) layer ----
        ('cpu/src/ct_field.cpp', 'CT field arithmetic: constant-time mul/sqr/inv with value_barrier, no branching'),
        ('cpu/src/ct_scalar.cpp', 'CT scalar arithmetic: constant-time inverse (SafeGCD), negate, cmov operations'),
        ('cpu/src/ct_point.cpp', 'CT point operations: constant-time mul, add, double with uniform memory access'),
        ('cpu/src/ct_sign.cpp', 'CT ECDSA+Schnorr signing: constant-time nonce gen, signing, secure_erase cleanup'),
        # ---- Headers (key subset) ----
        ('cpu/include/secp256k1/field.hpp', 'FieldElement class: 4x64 limb storage, normalize, is_zero, to_bytes, from_bytes'),
        ('cpu/include/secp256k1/scalar.hpp', 'Scalar class: mod-n arithmetic, parse/serialize, is_zero, negate, inverse'),
        ('cpu/include/secp256k1/point.hpp', 'Point/JacobianPoint classes: affine/jacobian coords, infinity, compress/decompress'),
        ('cpu/include/secp256k1/ecdsa.hpp', 'ECDSA public API: sign, verify, parse/serialize DER and compact signatures'),
        ('cpu/include/secp256k1/schnorr.hpp', 'BIP-340 Schnorr API: keypair, sign, verify, batch_verify declarations'),
        ('cpu/include/secp256k1/precompute.hpp', 'Comb precomputation table layout, serialize/deserialize declarations'),
        ('cpu/include/secp256k1/ct/ops.hpp', 'CT operations: ct_select, ct_equal, value_barrier, CLASSIFY/DECLASSIFY macros'),
        ('cpu/include/secp256k1/ct/sign.hpp', 'CT signing API: ct_ecdsa_sign, ct_schnorr_sign, ct_nonce_function declarations'),
        ('cpu/include/secp256k1/ct/field.hpp', 'CT FieldElement class: same API as fast::FieldElement, constant-time guarantees'),
        ('cpu/include/secp256k1/ct/scalar.hpp', 'CT Scalar class: constant-time inverse, negate, cmov declarations'),
        ('cpu/include/secp256k1/ct/point.hpp', 'CT Point class: constant-time scalar multiplication declarations'),
        ('cpu/include/secp256k1/ct_utils.hpp', 'CT utility functions: secure_erase, ct_memcmp, ct_cswap, timing annotations'),
        ('cpu/include/secp256k1/glv.hpp', 'GLV decomposition header: split_scalar, beta constant, window width config'),
        ('cpu/include/secp256k1/sha256.hpp', 'SHA-256 class: init, update, finalize, HMAC-SHA256, double-SHA256'),
        ('cpu/include/secp256k1/sha512.hpp', 'SHA-512 class: init, update, finalize, HMAC-SHA512 for PBKDF2'),
        ('cpu/include/secp256k1/hash_accel.hpp', 'Hash acceleration dispatch: SHA-NI, ARM SHA2, software fallback detection'),
        ('cpu/include/secp256k1/bip32.hpp', 'BIP-32 HD derivation API: ExtendedKey struct, ckd_priv, ckd_pub, from_seed'),
        ('cpu/include/secp256k1/bip39.hpp', 'BIP-39 mnemonic API: generate, validate, to_seed declarations'),
        ('cpu/include/secp256k1/musig2.hpp', 'MuSig2 API: key aggregation, nonce commitment, partial sign, aggregate'),
        ('cpu/include/secp256k1/frost.hpp', 'FROST threshold API: keygen, sign_share, aggregate, verify_share'),
        ('cpu/include/secp256k1/taproot.hpp', 'Taproot API: tweak_pubkey, check_output, control_block verification'),
        ('cpu/include/secp256k1/recovery.hpp', 'ECDSA recovery API: recover_pubkey from signature + recid'),
        ('cpu/include/secp256k1/pedersen.hpp', 'Pedersen commitment API: commit, verify, add, sub declarations'),
        ('cpu/include/secp256k1/zk.hpp', 'ZK proof API: range_proof, dleq_prove, dleq_verify, opening_proof'),
        ('cpu/include/secp256k1/ecdh.hpp', 'ECDH API: shared_secret computation declaration'),
        ('cpu/include/secp256k1/ecies.hpp', 'ECIES API: encrypt, decrypt with ephemeral ECDH + AES-GCM'),
        ('cpu/include/secp256k1/config.hpp', 'Compile-time platform detection: __int128, ASM, SIMD, cache sizes'),
        ('cpu/include/secp256k1/context.hpp', 'Library context: init, destroy, precomp table pointer, allocator config'),
        ('cpu/include/secp256k1/field_52.hpp', 'FE52 representation header: 5x52 limb layout, mul/sqr/inv declarations'),
        ('cpu/include/secp256k1/field_26.hpp', 'FE26 representation header: 10x26 limb layout for 32-bit platforms'),
        # ---- C ABI ----
        ('include/ufsecp/ufsecp.h', 'Stable C ABI header: all ufsecp_* function declarations, opaque context, error codes'),
        ('include/ufsecp/ufsecp_impl.cpp', 'C ABI implementation: routes 108 ufsecp_* functions to fast/CT internal calls'),
        ('include/ufsecp/ufsecp_error.h', 'Error code enum: ufsecp_error_t with 11 error codes and string conversion'),
        ('include/ufsecp/ufsecp_version.h', 'Library version macros: UFSECP_VERSION_MAJOR/MINOR/PATCH, version string'),
    ]
    for path, summary in summaries:
        cur.execute("INSERT OR IGNORE INTO file_summaries (path, summary) VALUES (?,?)",
                    (path, summary))
    return len(summaries)


# ---------------------------------------------------------------------------
# FUNCTION INDEX  --  exact line ranges for token-efficient file reading
# ---------------------------------------------------------------------------
def populate_function_index(cur):
    """Scan source files for function/method definitions with line ranges."""
    import re
    
    count = 0
    # Patterns for C/C++ function definitions (not declarations)
    # Match: ReturnType [Class::]func_name( ... ) [const] [noexcept] {
    fn_pattern = re.compile(
        r'^(?:static\s+)?(?:inline\s+)?(?:constexpr\s+)?'
        r'(?:[\w:*&<>, ]+?)\s+'            # return type
        r'((?:\w+::)*(\w+))\s*\(',          # [Class::]function_name(
        re.MULTILINE
    )
    
    # Files to scan (core .cpp files where agents most often need line ranges)
    scan_paths = [
        'cpu/src/field.cpp', 'cpu/src/scalar.cpp', 'cpu/src/point.cpp',
        'cpu/src/ecdsa.cpp', 'cpu/src/schnorr.cpp', 'cpu/src/precompute.cpp',
        'cpu/src/ct_field.cpp', 'cpu/src/ct_scalar.cpp', 'cpu/src/ct_point.cpp',
        'cpu/src/ct_sign.cpp', 'cpu/src/glv.cpp', 'cpu/src/selftest.cpp',
        'cpu/src/hash_accel.cpp', 'cpu/src/recovery.cpp', 'cpu/src/musig2.cpp',
        'cpu/src/frost.cpp', 'cpu/src/taproot.cpp', 'cpu/src/bip32.cpp',
        'cpu/src/bip39.cpp', 'cpu/src/ecdh.cpp', 'cpu/src/ecies.cpp',
        'cpu/src/pedersen.cpp', 'cpu/src/batch_add_affine.cpp',
        'cpu/src/batch_verify.cpp', 'cpu/src/zk.cpp', 'cpu/src/adaptor.cpp',
        'cpu/src/address.cpp', 'cpu/src/wallet.cpp',
        'cpu/src/multiscalar.cpp', 'cpu/src/pippenger.cpp',
        'cpu/src/ecmult_gen_comb.cpp',
        'cpu/src/eth_signing.cpp', 'cpu/src/ethereum.cpp',
        'cpu/src/keccak256.cpp', 'cpu/src/message_signing.cpp',
        'include/ufsecp/ufsecp_impl.cpp',
    ]
    
    for rel_path in scan_paths:
        full_path = LIB_ROOT / rel_path
        if not full_path.exists():
            continue
        
        try:
            lines = full_path.read_text(encoding='utf-8', errors='replace').split('\n')
        except Exception:
            continue
        
        # Find function definitions by looking for lines that:
        # 1) Start a function (has a return type and opening paren)
        # 2) Are followed eventually by an opening brace at appropriate nesting
        func_starts = []
        for i, line in enumerate(lines):
            stripped = line.strip()
            # Skip comments, preprocessor, blank lines
            if not stripped or stripped.startswith('//') or stripped.startswith('#') or stripped.startswith('/*'):
                continue
            # Skip lines that are clearly not function defs
            if stripped.startswith('return') or stripped.startswith('if') or stripped.startswith('for'):
                continue
            if stripped.startswith('else') or stripped.startswith('while') or stripped.startswith('switch'):
                continue
            if stripped.startswith('case') or stripped.startswith('using') or stripped.startswith('typedef'):
                continue
            
            m = fn_pattern.match(stripped)
            if m:
                qualified = m.group(1)  # e.g., FieldElement::mul or ecdsa_sign
                func_name = m.group(2)  # e.g., mul or ecdsa_sign
                
                # Skip common false positives
                if func_name in ('if', 'for', 'while', 'switch', 'return', 'sizeof',
                                 'static_assert', 'alignas', 'alignof', 'throw',
                                 'catch', 'try', 'delete', 'new'):
                    continue
                
                # Determine class name
                class_name = None
                if '::' in qualified:
                    class_name = qualified.rsplit('::', 1)[0]
                
                # Determine kind
                kind = 'method' if class_name else 'function'
                
                func_starts.append((i, func_name, class_name, kind))
        
        # Compute end lines: each function ends at the line before the next function starts,
        # or at EOF for the last function. Then refine by finding the matching closing brace.
        for idx, (start_line, func_name, class_name, kind) in enumerate(func_starts):
            # End boundary: next function start or EOF
            if idx + 1 < len(func_starts):
                boundary = func_starts[idx + 1][0]
            else:
                boundary = len(lines)
            
            # Find the actual end by tracking brace depth
            depth = 0
            found_open = False
            end_line = boundary - 1
            for j in range(start_line, boundary):
                for ch in lines[j]:
                    if ch == '{':
                        depth += 1
                        found_open = True
                    elif ch == '}':
                        depth -= 1
                        if found_open and depth == 0:
                            end_line = j
                            break
                if found_open and depth == 0:
                    break
            
            cur.execute("""INSERT OR IGNORE INTO function_index
                (file_path, name, start_line, end_line, kind, class_name)
                VALUES (?,?,?,?,?,?)""",
                (rel_path, func_name, start_line + 1, end_line + 1, kind, class_name))
            count += 1
    
    return count


# ---------------------------------------------------------------------------
# PHASE 4 BUILDERS
# ---------------------------------------------------------------------------

def populate_call_edges(cur: sqlite3.Cursor):
    """Build function-level call graph by scanning .cpp source files."""
    # Load all known function names -> canonical file mappings
    known: dict = {}
    cur.execute("SELECT file_path, name FROM function_index")
    for fpath, name in cur.fetchall():
        if name not in known:
            known[name] = fpath
    cur.execute("SELECT name, category FROM c_abi_functions")
    for name, category in cur.fetchall():
        if name not in known:
            if category == 'gpu' or name.startswith('ufsecp_gpu_') or name == 'ufsecp_bip352_prepare_scan_plan':
                known[name] = 'cpu/src/ufsecp_gpu_impl.cpp'
            else:
                known[name] = 'cpu/src/ufsecp_impl.cpp'

    # Build file -> sorted list of (start_line, end_line, func_name)
    ranges: dict = {}
    cur.execute("SELECT file_path, name, start_line, end_line FROM function_index ORDER BY file_path, start_line")
    for fpath, name, sl, el in cur.fetchall():
        ranges.setdefault(fpath, []).append((sl, el, name))

    SKIP_WORDS = frozenset([
        'if', 'for', 'while', 'switch', 'return', 'sizeof', 'static_assert',
        'alignas', 'alignof', 'throw', 'catch', 'try', 'delete', 'new',
        'else', 'case', 'do', 'goto', 'break', 'continue', 'decltype',
        'static_cast', 'reinterpret_cast', 'const_cast', 'dynamic_cast',
        'assert', 'ASSERT', 'CHECK', 'memset', 'memcpy', 'memmove', 'memcmp',
        'printf', 'fprintf', 'puts', 'fopen', 'fclose', 'fread', 'strlen',
        'std', 'secp256k1', 'ct', 'fast', 'detail', 'hash', 'coins',
    ])

    call_re = re.compile(r'\b([a-zA-Z_]\w*)\s*\(')
    count = 0

    for rel_path, func_list in sorted(ranges.items()):
        full = LIB_ROOT / rel_path
        if not full.exists():
            continue
        try:
            lines = full.read_text(encoding='utf-8', errors='replace').split('\n')
        except Exception:
            continue

        for lineno, line in enumerate(lines, 1):
            s = line.strip()
            if not s or s.startswith('//') or s.startswith('#') or \
               s.startswith('/*') or s.startswith('*'):
                continue

            # Find which function this line belongs to
            caller = None
            for sl, el, fname in func_list:
                if sl <= lineno <= el:
                    caller = fname
                    break
            if not caller:
                continue

            for m in call_re.finditer(line):
                callee = m.group(1)
                if callee in SKIP_WORDS or callee == caller:
                    continue
                if callee not in known:
                    continue
                callee_file = known[callee]
                try:
                    cur.execute("""INSERT OR IGNORE INTO call_edges
                        (caller_file, caller_func, callee_func, callee_file, call_line)
                        VALUES (?,?,?,?,?)""",
                        (rel_path, caller, callee, callee_file, lineno))
                    count += 1
                except Exception:
                    pass

    # Also scan audit/test directories for ufsecp_* calls (these files are not in
    # function_index because they are test binaries, not library source). This
    # populates call_edges so that function_test_map can be derived for P2 coverage.
    test_scan_dirs = [
        LIB_ROOT / 'audit',
        LIB_ROOT / 'cpu' / 'tests',
        LIB_ROOT / 'tests',
    ]
    for test_dir in test_scan_dirs:
        if not test_dir.exists():
            continue
        for filepath in sorted(test_dir.rglob('*.cpp')):
            rel_path = str(filepath.relative_to(LIB_ROOT))
            try:
                lines = filepath.read_text(encoding='utf-8', errors='replace').split('\n')
            except Exception:
                continue
            for lineno, line in enumerate(lines, 1):
                s = line.strip()
                if not s or s.startswith('//') or s.startswith('#') or \
                   s.startswith('/*') or s.startswith('*'):
                    continue
                for m in call_re.finditer(line):
                    callee = m.group(1)
                    if not callee.startswith('ufsecp_'):
                        continue
                    if callee in SKIP_WORDS:
                        continue
                    callee_file = known.get(callee, 'cpu/src/ufsecp_impl.cpp')
                    try:
                        cur.execute("""INSERT OR IGNORE INTO call_edges
                            (caller_file, caller_func, callee_func, callee_file, call_line)
                            VALUES (?,?,?,?,?)""",
                            (rel_path, 'test_body', callee, callee_file, lineno))
                        count += 1
                    except Exception:
                        pass

    return count


def populate_config_bindings(cur: sqlite3.Cursor):
    """Map CMake options and project constants to the code symbols that implement them."""
    count = 0

    # 1. Scan CMakeLists.txt for option() declarations
    cmake_files = [LIB_ROOT / 'CMakeLists.txt']
    opt_re = re.compile(r'option\s*\(\s*(\w+)\s+"([^"]*)"')
    for cmake_f in cmake_files:
        if not cmake_f.exists():
            continue
        text = cmake_f.read_text(errors='replace')
        for m in opt_re.finditer(text):
            opt_name, opt_desc = m.group(1), m.group(2)
            cur.execute("""INSERT OR IGNORE INTO config_bindings
                (config_file, config_key, code_symbol, code_file, binding_type, description)
                VALUES (?,?,?,?,?,?)""",
                ('CMakeLists.txt', opt_name, opt_name, 'CMakeLists.txt',
                 'cmake_option', opt_desc))
            count += 1
            # Find which source files reference this option symbol
            for root, dirs, files in os.walk(LIB_ROOT / 'cpu'):
                dirs[:] = [d for d in dirs if not should_skip_dir(d)]
                for fname in files:
                    if os.path.splitext(fname)[1].lower() not in ('.cpp', '.hpp', '.h'):
                        continue
                    fpath = Path(root) / fname
                    try:
                        if opt_name in fpath.read_text(errors='replace'):
                            rel = str(fpath.relative_to(LIB_ROOT))
                            cur.execute("""INSERT OR IGNORE INTO config_bindings
                                (config_file, config_key, code_symbol, code_file, binding_type, description)
                                VALUES (?,?,?,?,?,?)""",
                                ('CMakeLists.txt', opt_name, opt_name, rel,
                                 'build_flag', f'Referenced in {rel}'))
                            count += 1
                    except Exception:
                        pass

    # 2. Insert project constants as config bindings
    cur.execute("SELECT name, value, category, context FROM constants")
    for cname, cval, ccat, cctx in cur.fetchall():
        cur.execute("""INSERT OR IGNORE INTO config_bindings
            (config_file, config_key, code_symbol, code_file, binding_type, description)
            VALUES (?,?,?,?,?,?)""",
            ('constants', cname, cname, None, 'project_constant', cctx or ''))
        count += 1

    # 3. Map error codes as config bindings (value -> symbol)
    cur.execute("SELECT code, symbol, meaning FROM error_codes")
    for code, symbol, meaning in cur.fetchall():
        cur.execute("""INSERT OR IGNORE INTO config_bindings
            (config_file, config_key, code_symbol, code_file, binding_type, description)
            VALUES (?,?,?,?,?,?)""",
            ('error_codes', str(code), symbol, 'include/ufsecp/ufsecp_error.h',
             'error_code', meaning or ''))
        count += 1

    return count


def populate_symbol_aliases(cur: sqlite3.Cursor):
    """Find similar symbol names (variant/typo detection) using string similarity."""
    from difflib import SequenceMatcher
    from collections import defaultdict

    # Collect all function names from function_index + c_abi_functions
    names: set = set()
    cur.execute("SELECT DISTINCT name FROM function_index WHERE kind IN ('function','method')")
    for (n,) in cur.fetchall():
        if len(n) > 4:  # skip very short identifiers
            names.add(n)
    cur.execute("SELECT name FROM c_abi_functions")
    for (n,) in cur.fetchall():
        names.add(n)

    # Group by leading prefix (first 2 underscore segments) for scalability
    groups: dict = defaultdict(list)
    for n in sorted(names):
        parts = n.split('_')
        key = '_'.join(parts[:2]) if len(parts) >= 2 else parts[0]
        groups[key].append(n)

    count = 0
    for key, group in groups.items():
        if len(group) < 2:
            continue
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, b = group[i], group[j]
                ratio = SequenceMatcher(None, a, b).ratio()
                if 0.72 <= ratio < 1.0:
                    kind = 'typo' if ratio >= 0.90 else 'variant' if ratio >= 0.80 else 'abbreviation'
                    # canonical = shorter name (assumed more primitive)
                    canonical, alias = (a, b) if len(a) <= len(b) else (b, a)
                    try:
                        cur.execute("""INSERT OR IGNORE INTO symbol_aliases
                            (canonical, alias, similarity, kind)
                            VALUES (?,?,?,?)""",
                            (canonical, alias, round(ratio, 3), kind))
                        count += 1
                    except Exception:
                        pass

    return count


def populate_hotspot_scores(cur: sqlite3.Cursor):
    """Compute per-file hotspot risk scores from existing graph data."""
    cur.execute("""SELECT path, lines, layer, subsystem FROM source_files
                   WHERE category IN ('cpu_core', 'abi', 'audit')
                   AND file_type IN ('cpp', 'source')
                   AND lines > 0 ORDER BY path""")
    files = cur.fetchall()

    count = 0
    for fpath, lines, layer, subsystem in files:
        # Coupling: fan-in (rdeps) + fan-out (deps)
        cur.execute("SELECT COUNT(*) FROM include_deps WHERE source_file=?", (fpath,))
        fan_out = cur.fetchone()[0]
        fname_only = os.path.basename(fpath)
        cur.execute("SELECT COUNT(*) FROM include_deps WHERE included_file LIKE ?",
                    (f'%{fname_only}%',))
        fan_in = cur.fetchone()[0]
        coupling = (fan_in + fan_out) / 10.0

        # Security density: patterns per 100 lines
        cur.execute("SELECT COUNT(*) FROM security_patterns WHERE source_file=?", (fpath,))
        sec_count = cur.fetchone()[0]
        security_density = (sec_count / lines) * 100.0

        # Test coverage gap: 0 = covered, 1 = not covered
        cur.execute("""SELECT COUNT(*) FROM edges
                       WHERE dst_type='source_file' AND dst_id=? AND relation='covers'""",
                    (fpath,))
        test_gap = 0.0 if cur.fetchone()[0] > 0 else 1.0

        # Pointer/null risk: scan source for risky patterns
        null_risk = 0.0
        full_path = LIB_ROOT / fpath
        if full_path.exists():
            try:
                text = full_path.read_text(errors='replace')
                null_count = len(re.findall(r'\*\s*\w+\s*(?:==|!=)\s*nullptr', text))
                null_count += len(re.findall(r'reinterpret_cast<', text))
                null_count += len(re.findall(r'\bfree\s*\(', text))
                null_risk = min(null_count / 10.0, 1.0)
            except Exception:
                pass

        # CT layer is more critical
        ct_mult = 1.5 if layer == 'ct' else 1.0
        score = ct_mult * (coupling * 0.25 + security_density * 4.0 +
                           test_gap * 3.0 + null_risk * 0.5)
        score = min(score, 10.0)

        reasons = []
        if fan_in + fan_out > 20:
            reasons.append(f'high_coupling:{fan_in}in+{fan_out}out')
        if sec_count > 5:
            reasons.append(f'security_critical:{sec_count}_patterns')
        if test_gap > 0:
            reasons.append('no_test_coverage')
        if null_risk > 0.3:
            reasons.append('pointer_risk')
        if layer == 'ct':
            reasons.append('ct_layer')

        cur.execute("""INSERT OR REPLACE INTO hotspot_scores
            (file_path, coupling_score, security_density, null_risk_score,
             test_coverage_gap, hotspot_score, reasons)
            VALUES (?,?,?,?,?,?,?)""",
            (fpath, round(coupling, 3), round(security_density, 3),
             round(null_risk, 3), test_gap, round(score, 3),
             json.dumps(reasons)))
        count += 1

    return count


def populate_reachability(cur: sqlite3.Cursor):
    """Mark functions as reachable/dead using BFS from ABI entry points."""
    # Seed: all public ABI entry points
    reachable: set = set()
    cur.execute("SELECT name FROM c_abi_functions")
    for (name,) in cur.fetchall():
        reachable.add(name)

    # Also seed from test executables (they independently exercise code)
    cur.execute("SELECT name, executable FROM test_targets")
    for name, exe in cur.fetchall():
        reachable.add(name)
        if exe:
            reachable.add(exe)

    # BFS through call_edges
    queue = list(reachable)
    visited = set(reachable)
    reach_via: dict = {}

    while queue:
        caller = queue.pop(0)
        cur.execute("SELECT callee_func FROM call_edges WHERE caller_func=?", (caller,))
        for (callee,) in cur.fetchall():
            if callee not in visited:
                visited.add(callee)
                reachable.add(callee)
                reach_via[callee] = caller
                queue.append(callee)

    # Insert reachability for all indexed functions
    cur.execute("SELECT file_path, name FROM function_index")
    all_funcs = cur.fetchall()

    count = 0
    for fpath, name in all_funcs:
        is_r = 1 if name in reachable else 0
        via = reach_via.get(name)
        dead_reason = None if is_r else 'no_caller_in_call_graph'
        try:
            cur.execute("""INSERT OR IGNORE INTO reachability
                (symbol, file_path, is_reachable, reach_via, dead_reason)
                VALUES (?,?,?,?,?)""",
                (name, fpath, is_r, via, dead_reason))
            count += 1
        except Exception:
            pass

    return count


def populate_runtime_entrypoints(cur: sqlite3.Cursor):
    """Find main() entrypoints and config-loading patterns in app source files."""
    count = 0
    app_dirs = ['apps', 'cpu/tests', 'cpu/bench', 'audit', 'examples']
    main_re = re.compile(r'\bint\s+main\s*\(')
    config_re = re.compile(r'"config\.json"|"config_path"|"database"|fopen\s*\(')

    for app_dir in app_dirs:
        dirpath = LIB_ROOT / app_dir
        if not dirpath.exists():
            continue
        for root, dirs, files in os.walk(dirpath):
            dirs[:] = [d for d in dirs if not should_skip_dir(d)]
            for fname in sorted(files):
                if os.path.splitext(fname)[1].lower() not in ('.cpp', '.c', '.cu', '.mm'):
                    continue
                fpath = Path(root) / fname
                rel = str(fpath.relative_to(LIB_ROOT))
                binary = os.path.splitext(fname)[0]
                try:
                    lines = fpath.read_text(errors='replace').split('\n')
                except Exception:
                    continue
                for i, line in enumerate(lines, 1):
                    if main_re.search(line):
                        cur.execute("""INSERT OR IGNORE INTO runtime_entrypoints
                            (binary, entrypoint_func, loads_file, load_mechanism, source_file, line_no)
                            VALUES (?,?,?,?,?,?)""",
                            (binary, 'main', None, 'compiled-in', rel, i))
                        count += 1
                    elif config_re.search(line):
                        loads = 'config.json' if 'config.json' in line else None
                        mech = 'fopen' if 'fopen' in line else 'string_ref'
                        cur.execute("""INSERT OR IGNORE INTO runtime_entrypoints
                            (binary, entrypoint_func, loads_file, load_mechanism, source_file, line_no)
                            VALUES (?,?,?,?,?,?)""",
                            (binary, 'config_load', loads, mech, rel, i))
                        count += 1

    # ufsecp_ctx_create always runs selftest on startup
    cur.execute("""INSERT OR IGNORE INTO runtime_entrypoints
        (binary, entrypoint_func, loads_file, load_mechanism, source_file, line_no)
        VALUES (?,?,?,?,?,?)""",
        ('libufsecp', 'ufsecp_ctx_create', None, 'compiled-in',
         'include/ufsecp/ufsecp_impl.cpp', 223))
    count += 1

    return count


def populate_function_test_map(cur: sqlite3.Cursor):
    """Map functions to test targets at function level (derived from test-covers-file edges)."""
    count = 0

    # Derive from existing test -> source_file edges
    cur.execute("""SELECT src_id, dst_id FROM edges
                   WHERE src_type='test_target' AND dst_type='source_file'
                   AND relation='covers'""")
    pairs = cur.fetchall()

    for test_name, source_file in pairs:
        cur.execute("SELECT name FROM function_index WHERE file_path=?", (source_file,))
        for (func_name,) in cur.fetchall():
            try:
                cur.execute("""INSERT OR IGNORE INTO function_test_map
                    (function_name, function_file, test_target, coverage_type)
                    VALUES (?,?,?,?)""",
                    (func_name, source_file, test_name, 'indirect'))
                count += 1
            except Exception:
                pass

        cur.execute("""SELECT src_id FROM edges
                       WHERE src_type='c_abi_function' AND dst_type='source_file'
                         AND relation='implements' AND dst_id=?""", (source_file,))
        for (abi_name,) in cur.fetchall():
            try:
                cur.execute("""INSERT OR IGNORE INTO function_test_map
                    (function_name, function_file, test_target, coverage_type)
                    VALUES (?,?,?,?)""",
                    (abi_name, source_file, test_name, 'abi_indirect'))
                count += 1
            except Exception:
                pass

    # Additional KAT linkage: test -> abi_routing -> impl file -> functions
    cur.execute("""SELECT name, category FROM test_targets
                   WHERE category IN ('cpu_core', 'audit_always')""")
    for test_name, cat in cur.fetchall():
        parts = test_name.split('_')
        pattern = f'%{parts[0]}%' if parts else '%'
        cur.execute("SELECT abi_function FROM abi_routing WHERE abi_function LIKE ?",
                    (pattern,))
        for (abi_fn,) in cur.fetchall():
            impl = cur.execute("""SELECT dst_id FROM edges WHERE src_id=?
                                  AND relation='implements'""", (abi_fn,)).fetchone()
            if not impl:
                continue
            impl_file = impl[0]
            cur.execute("SELECT name FROM function_index WHERE file_path=?", (impl_file,))
            for (func_name,) in cur.fetchall():
                try:
                    cur.execute("""INSERT OR IGNORE INTO function_test_map
                        (function_name, function_file, test_target, coverage_type)
                        VALUES (?,?,?,?)""",
                        (func_name, impl_file, test_name, 'kat'))
                    count += 1
                except Exception:
                    pass

    # Direct audit/test caller linkage: map explicit ufsecp_* calls in tests back
    # to the ABI function itself so file-level coverage gaps do not hide real callers.
    cur.execute("""SELECT DISTINCT caller_file, callee_func, COALESCE(callee_file, '')
                   FROM call_edges
                   WHERE callee_func LIKE 'ufsecp_%'
                     AND (
                         caller_file LIKE 'audit/%' OR
                         caller_file LIKE 'cpu/tests/%' OR
                         caller_file LIKE 'tests/%'
                     )""")
    for caller_file, callee_func, callee_file in cur.fetchall():
        test_name = Path(caller_file).stem
        if test_name.startswith('test_'):
            test_name = test_name[len('test_'):]
        function_file = callee_file or 'include/ufsecp/ufsecp_impl.cpp'
        try:
            cur.execute("""INSERT OR IGNORE INTO function_test_map
                (function_name, function_file, test_target, coverage_type)
                VALUES (?,?,?,?)""",
                (callee_func, function_file, test_name, 'direct_call'))
            count += 1
        except Exception:
            pass

    return count


def populate_symbol_reasoning(cur: sqlite3.Cursor):
    """Populate symbol-level semantic, security, performance, audit, history, and score layers."""
    for table in (
        'symbol_semantics',
        'symbol_security',
        'symbol_performance',
        'symbol_audit_coverage',
        'symbol_history',
        'symbol_scores',
    ):
        cur.execute(f"DELETE FROM {table}")

    history_cache = {}
    count = 0

    rows = cur.execute("""
        SELECT DISTINCT file_path, name
        FROM function_index
        ORDER BY file_path, name
    """).fetchall()

    for file_path, symbol_name in rows:
        semantics = derive_symbol_semantics(symbol_name, file_path)
        security = derive_symbol_security(symbol_name, file_path, semantics)
        performance = derive_symbol_performance(symbol_name, file_path, semantics)
        history = get_file_history(file_path, history_cache)

        test_rows = cur.execute("""
            SELECT DISTINCT test_target, coverage_type
            FROM function_test_map
            WHERE function_name=? AND function_file=?
        """, (symbol_name, file_path)).fetchall()
        test_names = [r[0] for r in test_rows]
        test_blob = ' '.join(test_names).lower()
        coverage = {
            'covered_by_unit_test': 1 if test_names else 0,
            'covered_by_fuzz': 1 if any('fuzz' in t for t in test_names) else 0,
            'covered_by_invalid_vectors': 1 if any(x in test_blob for x in ('wycheproof', 'vectors', 'strict', 'bip340', 'frost_kat', 'bip327')) else 0,
            'covered_by_ct_test': 1 if any(x in test_blob for x in ('ct_', 'ct-', 'ctsidechannel', 'ct_sidechannel', 'ct_equivalence', 'ct_verif')) else 0,
            'covered_by_cross_impl_diff': 1 if any(x in test_blob for x in ('differential', 'cross_', 'fiat_crypto', 'equivalence')) else 0,
            'covered_by_gpu_equivalence': 1 if any(x in test_blob for x in ('gpu_', 'opencl_', 'metal_')) else 0,
            'covered_by_regression_test': 1 if any(x in test_blob for x in ('regression', 'fault_injection', 'adversarial')) else 0,
            'last_audit_result': 'covered' if test_names else 'uncovered',
            'times_failed_historically': history['bug_fix_count'],
            'known_fragile': 1 if any(x in test_blob for x in ('fault_injection', 'regression', 'adversarial')) or history['bug_fix_count'] >= 3 else 0,
        }

        low_audit = 1 if (coverage['covered_by_unit_test'] + coverage['covered_by_fuzz'] + coverage['covered_by_ct_test']) <= 1 else 0
        risk_reasons = []
        gain_reasons = []
        risk = 0.0
        gain = 0.0

        if security['must_be_constant_time']:
            risk += 5.0
            risk_reasons.append('ct_sensitive')
        if security['invalid_input_sensitive']:
            risk += 4.0
            risk_reasons.append('invalid_input_sensitive')
        if low_audit:
            risk += 3.0
            risk_reasons.append('low_audit_coverage')
        if history['recently_modified']:
            risk += 2.0
            risk_reasons.append('recently_modified')
        if coverage['times_failed_historically']:
            # Cap per-symbol contribution from file-level bug_fix_count.
            # The history is file-scoped (e.g. ufsecp_impl.cpp 182 funcs): raw count
            # is shared by every function, inflating risk.  Dampen with log + cap.
            import math
            hist_contribution = min(6.0, math.log2(1 + coverage['times_failed_historically']) * 1.5)
            risk += hist_contribution
            risk_reasons.append('historical_failures')

        gain += performance['hotness_score'] * 0.4
        if performance['batchable']:
            gain += 12.0
            gain_reasons.append('batchable')
        if performance['gpu_candidate']:
            gain += 18.0
            gain_reasons.append('gpu_candidate')
        if performance['duplicated_backends']:
            gain += 8.0
            gain_reasons.append('duplicated_backends')
        if performance['compute_bound']:
            gain += 6.0
            gain_reasons.append('compute_bound')

        optimization_priority = max(0.0, min(100.0, gain - (risk * 0.35)))

        cur.execute("""
            INSERT INTO symbol_semantics
            (symbol_name, file_path, category, math_core, backend, coordinate_model,
             secret_class, abi_surface, generator_path, varpoint_path, bip340_related, bip352_related)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            symbol_name, file_path, semantics['category'], semantics['math_core'], semantics['backend'],
            semantics['coordinate_model'], semantics['secret_class'], semantics['abi_surface'],
            semantics['generator_path'], semantics['varpoint_path'], semantics['bip340_related'],
            semantics['bip352_related'],
        ))

        cur.execute("""
            INSERT INTO symbol_security
            (symbol_name, file_path, uses_secret_input, must_be_constant_time, public_data_only,
             device_secret_upload, requires_zeroization, invalid_input_sensitive, notes)
            VALUES (?,?,?,?,?,?,?,?,?)
        """, (
            symbol_name, file_path, security['uses_secret_input'], security['must_be_constant_time'],
            security['public_data_only'], security['device_secret_upload'], security['requires_zeroization'],
            security['invalid_input_sensitive'], security['notes'],
        ))

        cur.execute("""
            INSERT INTO symbol_performance
            (symbol_name, file_path, hotness_score, estimated_cost, batchable, vectorizable,
             gpu_candidate, memory_bound, compute_bound, duplicated_backends)
            VALUES (?,?,?,?,?,?,?,?,?,?)
        """, (
            symbol_name, file_path, performance['hotness_score'], performance['estimated_cost'],
            performance['batchable'], performance['vectorizable'], performance['gpu_candidate'],
            performance['memory_bound'], performance['compute_bound'], performance['duplicated_backends'],
        ))

        cur.execute("""
            INSERT INTO symbol_audit_coverage
            (symbol_name, file_path, covered_by_unit_test, covered_by_fuzz, covered_by_invalid_vectors,
             covered_by_ct_test, covered_by_cross_impl_diff, covered_by_gpu_equivalence,
             covered_by_regression_test, last_audit_result, times_failed_historically, known_fragile)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            symbol_name, file_path, coverage['covered_by_unit_test'], coverage['covered_by_fuzz'],
            coverage['covered_by_invalid_vectors'], coverage['covered_by_ct_test'],
            coverage['covered_by_cross_impl_diff'], coverage['covered_by_gpu_equivalence'],
            coverage['covered_by_regression_test'], coverage['last_audit_result'],
            coverage['times_failed_historically'], coverage['known_fragile'],
        ))

        cur.execute("""
            INSERT INTO symbol_history
            (symbol_name, file_path, times_modified, recently_modified, bug_fix_count,
             performance_tuning_count, audit_related_changes, last_modified)
            VALUES (?,?,?,?,?,?,?,?)
        """, (
            symbol_name, file_path, history['times_modified'], history['recently_modified'],
            history['bug_fix_count'], history['performance_tuning_count'],
            history['audit_related_changes'], history['last_modified'],
        ))

        cur.execute("""
            INSERT INTO symbol_scores
            (symbol_name, file_path, risk_score, gain_score, optimization_priority, risk_reasons, gain_reasons)
            VALUES (?,?,?,?,?,?,?)
        """, (
            symbol_name, file_path, round(min(100.0, risk * 3.5), 2), round(min(100.0, gain), 2),
            round(optimization_priority, 2), json.dumps(risk_reasons), json.dumps(gain_reasons),
        ))
        count += 1

    return count


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def reset_database_files():
    for path in (DB_PATH, Path(f"{DB_PATH}-wal"), Path(f"{DB_PATH}-shm")):
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def build_graph(rebuild=False):
    # This builder repopulates the entire graph, so always recreate the DB.
    # Reusing an existing file risks stale rows and FTS shadow-table corruption.
    reset_database_files()
    
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    cur = conn.cursor()
    
    # Create schema
    cur.executescript(SCHEMA_SQL)
    
    # Populate
    now = datetime.now(timezone.utc).isoformat()
    cur.execute("INSERT OR REPLACE INTO meta VALUES ('built_at', ?)", (now,))
    cur.execute("INSERT OR REPLACE INTO meta VALUES ('lib_root', ?)", (str(LIB_ROOT),))
    cur.execute("INSERT OR REPLACE INTO meta VALUES ('version', '4.29.0')", ())
    cur.execute("INSERT OR REPLACE INTO meta VALUES ('schema_version', '4')", ())
    
    stats = {}
    stats['source_files'] = populate_source_files(cur)
    stats['namespaces'] = populate_namespaces(cur)
    stats['cpp_types'] = populate_cpp_types(cur)
    stats['c_abi_functions'] = populate_abi_functions(cur)
    stats['include_deps'] = populate_include_deps(cur)
    stats['test_targets'] = populate_test_targets(cur)
    stats['ci_workflows'] = populate_ci_workflows(cur)
    stats['audit_modules'] = populate_audit_modules(cur)
    stats['error_codes'] = populate_error_codes(cur)
    stats['constants'] = populate_constants(cur)
    stats['gpu_backends'] = populate_gpu_backends(cur)
    stats['build_configs'] = populate_build_configs(cur)
    stats['platform_dispatch'] = populate_platform_dispatch(cur)
    stats['docs'] = populate_docs(cur)
    stats['semantic_tags'] = populate_semantic_tags(cur)
    stats['cpp_methods'] = populate_cpp_methods(cur)
    stats['security_patterns'] = populate_security_patterns(cur)
    stats['abi_routing'] = populate_abi_routing(cur)
    stats['binding_languages'] = populate_binding_languages(cur)
    stats['macros'] = populate_macros(cur)
    stats['file_summaries'] = populate_file_summaries(cur)
    stats['function_index'] = populate_function_index(cur)
    stats['edges'] = populate_edges(cur)
    # Phase 4: new intelligence layers
    stats['call_edges'] = populate_call_edges(cur)
    stats['config_bindings'] = populate_config_bindings(cur)
    stats['symbol_aliases'] = populate_symbol_aliases(cur)
    stats['hotspot_scores'] = populate_hotspot_scores(cur)
    stats['reachability'] = populate_reachability(cur)
    stats['runtime_entrypoints'] = populate_runtime_entrypoints(cur)
    stats['function_test_map'] = populate_function_test_map(cur)
    stats['symbol_reasoning'] = populate_symbol_reasoning(cur)
    
    cur.execute("INSERT OR REPLACE INTO meta VALUES ('stats', ?)", (json.dumps(stats),))
    
    conn.commit()
    conn.close()
    
    return stats

if __name__ == '__main__':
    rebuild = '--rebuild' in sys.argv
    print(f"Building project graph: {DB_PATH}")
    stats = build_graph(rebuild=rebuild)
    total = sum(stats.values())
    print(f"\nPopulated {total} records across {len(stats)} tables:")
    for table, count in sorted(stats.items()):
        print(f"  {table:25s} {count:5d}")
    print(f"\nDatabase: {DB_PATH} ({DB_PATH.stat().st_size / 1024:.0f} KB)")
