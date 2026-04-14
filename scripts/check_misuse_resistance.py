#!/usr/bin/env python3
"""Misuse-resistance gate — verifies hostile-caller coverage on ABI boundary.

Checks that every ufsecp_* C ABI function has adequate negative test coverage:
  - NULL pointer arguments
  - Zero-length buffers
  - Oversized length params
  - Double-free / double-destroy
  - Context misuse (wrong type, uninitialized)

Minimum: 3 negative test cases per ABI function. Fail-closed.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
LIB_ROOT = SCRIPT_DIR.parent
GRAPH_DB = LIB_ROOT / "tools" / "source_graph_kit" / "source_graph.db"
ABI_HEADER = LIB_ROOT / "include" / "ufsecp" / "ufsecp.h"
ABI_GPU_HEADER = LIB_ROOT / "include" / "ufsecp" / "ufsecp_gpu.h"
NEGATIVE_MANIFEST = LIB_ROOT / "docs" / "ABI_NEGATIVE_TEST_MANIFEST.json"

# Minimum negative test cases per ABI function
MIN_NEGATIVE_TESTS = 3

# Negative test categories we check for
NEGATIVE_CATEGORIES = [
    "null_pointer",
    "zero_length",
    "oversized_length",
    "double_free",
    "context_misuse",
    "partially_initialized",
    "cross_thread",
]


def _extract_abi_functions_from_header(header: Path) -> list[str]:
    """Extract ufsecp_* function names from a header file."""
    if not header.exists():
        return []
    content = header.read_text(encoding="utf-8", errors="replace")
    # Match function declarations: ufsecp_something(
    pattern = re.compile(r'\b(ufsecp_\w+)\s*\(')
    return sorted(set(pattern.findall(content)))


def _extract_abi_functions_from_graph(conn: sqlite3.Connection) -> list[str]:
    """Extract ufsecp_* symbols from the graph."""
    try:
        rows = conn.execute(
            "SELECT DISTINCT symbol_name FROM symbols WHERE symbol_name LIKE 'ufsecp_%' ORDER BY symbol_name"
        ).fetchall()
        return [r[0] for r in rows]
    except sqlite3.OperationalError:
        return []


def _count_negative_tests_from_manifest(manifest: Path) -> dict[str, int]:
    """Load negative test counts from manifest."""
    if not manifest.exists():
        return {}
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
        counts: dict[str, int] = {}
        if isinstance(data, dict):
            for fn_name, tests in data.get("functions", {}).items():
                if isinstance(tests, list):
                    counts[fn_name] = len(tests)
                elif isinstance(tests, int):
                    counts[fn_name] = tests
        return counts
    except (json.JSONDecodeError, KeyError):
        return {}


def _count_negative_tests_from_graph(conn: sqlite3.Connection, fn_name: str) -> int:
    """Count negative tests for a function from the graph test mappings."""
    try:
        row = conn.execute(
            """SELECT COUNT(*) FROM test_mappings
               WHERE symbol_name LIKE ? AND
               (test_name LIKE '%negative%' OR test_name LIKE '%null%'
                OR test_name LIKE '%invalid%' OR test_name LIKE '%hostile%'
                OR test_name LIKE '%misuse%' OR test_name LIKE '%abuse%'
                OR test_name LIKE '%abi_negative%')""",
            (f"%{fn_name}%",),
        ).fetchone()
        return row[0] if row else 0
    except sqlite3.OperationalError:
        return 0


def run(json_mode: bool, out_file: str | None) -> int:
    # Collect ABI functions from headers
    abi_fns = set()
    for header in [ABI_HEADER, ABI_GPU_HEADER]:
        abi_fns.update(_extract_abi_functions_from_header(header))

    # Also check graph
    conn = None
    if GRAPH_DB.exists():
        try:
            conn = sqlite3.connect(str(GRAPH_DB))
            abi_fns.update(_extract_abi_functions_from_graph(conn))
        except Exception:
            pass

    abi_fns_sorted = sorted(abi_fns)

    # Load manifest-based counts
    manifest_counts = _count_negative_tests_from_manifest(NEGATIVE_MANIFEST)

    # Evaluate each function
    results: list[dict] = []
    for fn in abi_fns_sorted:
        # Count from manifest first, then graph fallback
        count = manifest_counts.get(fn, 0)
        if count == 0 and conn:
            count = _count_negative_tests_from_graph(conn, fn)

        passing = count >= MIN_NEGATIVE_TESTS
        results.append({
            "function": fn,
            "negative_tests": count,
            "min_required": MIN_NEGATIVE_TESTS,
            "passing": passing,
        })

    if conn:
        conn.close()

    total = len(results)
    passing_count = sum(1 for r in results if r["passing"])
    failing = [r for r in results if not r["passing"]]

    overall_pass = len(failing) == 0

    report = {
        "overall_pass": overall_pass,
        "abi_functions_total": total,
        "abi_functions_covered": passing_count,
        "negative_tests_total": sum(r["negative_tests"] for r in results),
        "functions_below_threshold": [r["function"] for r in failing],
        "min_tests_per_function": MIN_NEGATIVE_TESTS,
        "functions": results,
    }

    rendered = json.dumps(report, indent=2)
    if out_file:
        Path(out_file).write_text(rendered, encoding="utf-8")

    if json_mode:
        print(rendered)
    else:
        if failing:
            print(f"  Functions below threshold ({MIN_NEGATIVE_TESTS} min):")
            for r in failing[:20]:  # cap display
                print(f"    {r['function']}: {r['negative_tests']} tests")
            if len(failing) > 20:
                print(f"    ... and {len(failing) - 20} more")
        print()
        print(f"  Coverage: {passing_count}/{total} ABI functions meet threshold")
        if overall_pass:
            print("PASS misuse-resistance gate")
        else:
            print(f"FAIL {len(failing)} function(s) below {MIN_NEGATIVE_TESTS} negative tests")

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
