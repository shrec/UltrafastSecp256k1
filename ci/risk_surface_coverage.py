#!/usr/bin/env python3
"""Risk-surface coverage matrix — measures coverage by risk class, not by code lines.

Risk classes:
  ct_paths        — constant-time sensitive functions (sign, nonce, scalar_inv, ecdh)
  parser_boundary — DER/BIP/serialization parse entry points
  abi_boundary    — ufsecp_* C ABI surface
  gpu_parity      — GPU backend method parity (CUDA/OpenCL/Metal)
  secret_lifecycle— paths that touch secret material (create, use, zeroize)
  determinism     — operations that must be bitwise identical across runs/arches
  fuzz_corpus     — fuzz target coverage of risky input surfaces

Each class has a minimum coverage threshold. Fail-closed: if any critical class
is below threshold, the gate fails.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
LIB_ROOT = SCRIPT_DIR.parent
GRAPH_DB = LIB_ROOT / "tools" / "source_graph_kit" / "source_graph.db"

# Risk classes: name → (symbol patterns, min threshold, critical flag)
RISK_CLASSES: dict[str, dict] = {
    "ct_paths": {
        "description": "Constant-time sensitive functions",
        "symbol_patterns": [
            "%ct_%", "%constant_time%", "%ct_sign%", "%ct_verify%",
            "%ct_scalar%", "%ct_field%", "%scalar_inverse%",
            "%nonce_function%", "%rfc6979%",
        ],
        "file_patterns": [
            "%ct_%.cpp", "%ct_%.hpp", "%ct_sign%", "%ct_verify%",
        ],
        "min_coverage": 80,
        "critical": True,
    },
    "parser_boundary": {
        "description": "DER/BIP/serialization parse entry points",
        "symbol_patterns": [
            "%parse%", "%deserialize%", "%decode%", "%from_bytes%",
            "%der_%", "%bip32_parse%", "%pubkey_parse%",
        ],
        "file_patterns": [
            "%parse%.cpp", "%der_%", "%fuzz_der%", "%fuzz_pubkey%",
        ],
        "min_coverage": 80,
        "critical": True,
    },
    "abi_boundary": {
        "description": "ufsecp_* C ABI surface",
        "symbol_patterns": [
            "ufsecp_%",
        ],
        "file_patterns": [
            "%ufsecp_%.h", "%ufsecp_%.cpp", "%test_c_abi%",
        ],
        "min_coverage": 85,
        "critical": True,
    },
    "gpu_parity": {
        "description": "GPU backend method parity",
        "symbol_patterns": [
            "%GpuBackend%", "%gpu_%", "%cuda_%", "%opencl_%", "%metal_%",
        ],
        "file_patterns": [
            "%gpu_backend%", "%cuda_backend%", "%opencl_backend%", "%metal_backend%",
        ],
        "min_coverage": 70,
        "critical": True,
    },
    "secret_lifecycle": {
        "description": "Paths touching secret material",
        "symbol_patterns": [
            "%secure_erase%", "%zeroize%", "%secret%", "%private_key%",
            "%signing_key%", "%nonce%", "%scalar_inverse%",
        ],
        "file_patterns": [
            "%secure_erase%", "%secret_%", "%ct_sign%",
        ],
        "min_coverage": 85,
        "critical": True,
    },
    "determinism": {
        "description": "Operations requiring bitwise determinism",
        "symbol_patterns": [
            "%determinism%", "%deterministic%", "%rfc6979%",
        ],
        "file_patterns": [
            "%determinism%", "%rfc6979%", "%check_determinism%",
        ],
        "min_coverage": 90,
        "critical": True,
    },
    "fuzz_corpus": {
        "description": "Fuzz target coverage of risky inputs",
        "symbol_patterns": [
            "%fuzz_%", "%LLVMFuzzerTestOneInput%",
        ],
        "file_patterns": [
            "%fuzz_%.cpp", "%audit_fuzz%", "%test_fuzz%", "%test_libfuzzer%",
        ],
        "min_coverage": 60,
        "critical": False,
    },
}


def connect_graph() -> sqlite3.Connection | None:
    """Connect to the source graph database."""
    if not GRAPH_DB.exists():
        return None
    conn = sqlite3.connect(str(GRAPH_DB))
    conn.row_factory = sqlite3.Row
    return conn


def _count_like(conn: sqlite3.Connection, table: str, col: str, patterns: list[str]) -> int:
    """Count distinct rows matching any LIKE pattern."""
    if not patterns:
        return 0
    clauses = " OR ".join(f"{col} LIKE ?" for _ in patterns)
    sql = f"SELECT COUNT(DISTINCT {col}) FROM {table} WHERE {clauses}"
    try:
        row = conn.execute(sql, patterns).fetchone()
        return row[0] if row else 0
    except sqlite3.OperationalError:
        return 0


def _count_tested(conn: sqlite3.Connection, symbol_patterns: list[str]) -> tuple[int, int]:
    """Return (symbols_with_tests, total_symbols) matching patterns."""
    if not symbol_patterns:
        return 0, 0
    clauses = " OR ".join("function_name LIKE ?" for _ in symbol_patterns)

    # total symbols from function_index
    sql_total = f"SELECT COUNT(DISTINCT function_name) FROM function_index WHERE ({clauses})"
    try:
        total = conn.execute(sql_total, symbol_patterns).fetchone()[0]
    except sqlite3.OperationalError:
        total = 0

    if total == 0:
        return 0, 0

    # symbols with test mappings via test_function_map
    sql_tested = f"""
        SELECT COUNT(DISTINCT fi.function_name)
        FROM function_index fi
        INNER JOIN test_function_map t ON fi.function_name = t.function_name
        WHERE ({clauses})
    """
    try:
        tested = conn.execute(sql_tested, symbol_patterns).fetchone()[0]
    except sqlite3.OperationalError:
        # fallback: try symbol_audit_coverage
        clauses_sac = " OR ".join("symbol_name LIKE ?" for _ in symbol_patterns)
        sql_alt = f"""
            SELECT COUNT(DISTINCT symbol_name)
            FROM symbol_audit_coverage
            WHERE ({clauses_sac})
            AND coverage_score > 30
        """
        try:
            tested = conn.execute(sql_alt, symbol_patterns).fetchone()[0]
        except sqlite3.OperationalError:
            tested = 0

    return tested, total


def _count_files_tested(conn: sqlite3.Connection, file_patterns: list[str]) -> tuple[int, int]:
    """Return (files_with_tests, total_files) matching patterns."""
    if not file_patterns:
        return 0, 0
    clauses = " OR ".join("path LIKE ?" for _ in file_patterns)

    sql_total = f"SELECT COUNT(DISTINCT path) FROM files WHERE ({clauses})"
    try:
        total = conn.execute(sql_total, file_patterns).fetchone()[0]
    except sqlite3.OperationalError:
        total = 0

    if total == 0:
        return 0, 0

    # files with audit coverage > threshold
    clauses_ac = " OR ".join("file LIKE ?" for _ in file_patterns)
    sql_covered = f"""
        SELECT COUNT(DISTINCT file)
        FROM audit_coverage
        WHERE ({clauses_ac}) AND coverage_score > 30
    """
    try:
        covered = conn.execute(sql_covered, file_patterns).fetchone()[0]
    except sqlite3.OperationalError:
        covered = 0

    return covered, total


def evaluate_class(conn: sqlite3.Connection, name: str, spec: dict) -> dict:
    """Evaluate a single risk class and return metrics."""
    sym_tested, sym_total = _count_tested(conn, spec["symbol_patterns"])
    file_tested, file_total = _count_files_tested(conn, spec["file_patterns"])

    # combined coverage: weighted average (symbols 60%, files 40%)
    if sym_total + file_total == 0:
        coverage = 0.0
    else:
        sym_pct = (sym_tested / sym_total * 100) if sym_total > 0 else 0.0
        file_pct = (file_tested / file_total * 100) if file_total > 0 else 0.0
        if sym_total > 0 and file_total > 0:
            coverage = sym_pct * 0.6 + file_pct * 0.4
        elif sym_total > 0:
            coverage = sym_pct
        else:
            coverage = file_pct

    passing = coverage >= spec["min_coverage"]

    return {
        "class": name,
        "description": spec["description"],
        "critical": spec["critical"],
        "min_coverage": spec["min_coverage"],
        "coverage_percent": round(coverage, 1),
        "symbols_tested": sym_tested,
        "symbols_total": sym_total,
        "files_tested": file_tested,
        "files_total": file_total,
        "passing": passing,
    }


def run(json_mode: bool, out_file: str | None) -> int:
    conn = connect_graph()
    if conn is None:
        report = {
            "overall_pass": False,
            "error": f"Graph DB not found at {GRAPH_DB}",
            "classes": [],
        }
        rendered = json.dumps(report, indent=2)
        if out_file:
            Path(out_file).write_text(rendered, encoding="utf-8")
        print(rendered if json_mode else f"FAIL graph DB not found: {GRAPH_DB}")
        return 1

    results = []
    for name, spec in RISK_CLASSES.items():
        results.append(evaluate_class(conn, name, spec))

    conn.close()

    critical_failing = [r for r in results if r["critical"] and not r["passing"]]
    overall_pass = len(critical_failing) == 0

    report = {
        "overall_pass": overall_pass,
        "classes_total": len(results),
        "classes_passing": sum(1 for r in results if r["passing"]),
        "critical_classes_total": sum(1 for r in results if r["critical"]),
        "critical_classes_passing": sum(1 for r in results if r["critical"] and r["passing"]),
        "critical_failures": [r["class"] for r in critical_failing],
        "classes": results,
    }

    rendered = json.dumps(report, indent=2)
    if out_file:
        Path(out_file).write_text(rendered, encoding="utf-8")

    if json_mode:
        print(rendered)
    else:
        for r in results:
            status = "PASS" if r["passing"] else "FAIL"
            crit = " [CRITICAL]" if r["critical"] and not r["passing"] else ""
            print(f"  {status} {r['class']}: {r['coverage_percent']}% "
                  f"(min {r['min_coverage']}%) "
                  f"syms={r['symbols_tested']}/{r['symbols_total']} "
                  f"files={r['files_tested']}/{r['files_total']}{crit}")
        print()
        if overall_pass:
            print("PASS risk-surface coverage gate")
        else:
            print(f"FAIL {len(critical_failing)} critical class(es) below threshold")

    return 0 if overall_pass else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("-o", dest="out_file", help="Write report to file")
    args = parser.parse_args()
    return run(args.json, args.out_file)


if __name__ == "__main__":
    raise SystemExit(main())
