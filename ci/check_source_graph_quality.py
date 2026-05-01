#!/usr/bin/env python3
"""check_source_graph_quality.py  --  Source graph quality gate for UltrafastSecp256k1

Verifies that the project knowledge graph (.project_graph.db) exists, is fresh,
and covers the key source directories and symbols that CAAS depends on.

Checks
------
1. DB exists at .project_graph.db  -- FAIL if missing
2. DB is not stale  -- FAIL if DB mtime is older than any source file in the
   watched directories (configurable staleness threshold: 1 hour by default)
3. Key directories indexed  -- FAIL if entry count is below per-directory floor
4. Key symbols indexed  -- FAIL if a mandatory symbol is absent from the graph
5. CT metadata present  -- WARN if no CT-tagged functions found in the graph
6. Graph size sanity  -- FAIL if total entry count is below 200

Exit codes
----------
  0  all checks pass (may have WARN)
  1  one or more FAIL

Usage
-----
    python3 ci/check_source_graph_quality.py
    python3 ci/check_source_graph_quality.py --json
    python3 ci/check_source_graph_quality.py --db /path/to/.project_graph.db
    python3 ci/check_source_graph_quality.py --stale-hours 2
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
LIB_ROOT = SCRIPT_DIR.parent

DEFAULT_DB = LIB_ROOT / ".project_graph.db"
DEFAULT_STALE_HOURS = 1

# Directories to watch for staleness (relative to LIB_ROOT)
WATCHED_DIRS = ["cpu", "include", "compat", "audit"]

# Per-directory minimum entry counts
DIR_FLOORS: dict[str, int] = {
    "cpu":     50,
    "include": 6,
    "audit":  100,
    "compat":  10,
    "docs":    30,
    "scripts": 20,
}

# Symbols that must exist in the graph (checked against c_abi_functions,
# function_index, and cpp_methods depending on which tables exist)
REQUIRED_SYMBOLS = [
    "generator_mul",
    "ecdsa_sign",
    "schnorr_sign",
    "secp256k1_ecdsa_sign",
    "secp256k1_keypair_create",
]

# Minimum total rows across all tables
TOTAL_ROW_FLOOR = 200

# ANSI colours (suppressed when stdout is not a tty)
_USE_COLOR = sys.stdout.isatty()
RED    = '\033[91m' if _USE_COLOR else ''
GREEN  = '\033[92m' if _USE_COLOR else ''
YELLOW = '\033[93m' if _USE_COLOR else ''
CYAN   = '\033[96m' if _USE_COLOR else ''
BOLD   = '\033[1m'  if _USE_COLOR else ''
RESET  = '\033[0m'  if _USE_COLOR else ''


# ---------------------------------------------------------------------------
# Result helpers
# ---------------------------------------------------------------------------

class Check:
    """A single quality check result."""

    __slots__ = ("name", "status", "detail")

    def __init__(self, name: str, status: str, detail: str = "") -> None:
        self.name   = name
        self.status = status   # "PASS", "WARN", "FAIL"
        self.detail = detail

    def color(self) -> str:
        return {
            "PASS": GREEN,
            "WARN": YELLOW,
            "FAIL": RED,
        }.get(self.status, RESET)

    def as_dict(self) -> dict:
        return {"name": self.name, "status": self.status, "detail": self.detail}


def _pass(name: str, detail: str = "") -> Check:
    return Check(name, "PASS", detail)


def _warn(name: str, detail: str) -> Check:
    return Check(name, "WARN", detail)


def _fail(name: str, detail: str) -> Check:
    return Check(name, "FAIL", detail)


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _open_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def _list_tables(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    return {r["name"] for r in rows}


def _table_count(conn: sqlite3.Connection, table: str) -> int:
    try:
        return conn.execute(f"SELECT COUNT(*) AS cnt FROM {table}").fetchone()["cnt"]
    except sqlite3.OperationalError:
        return 0


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_db_exists(db_path: Path) -> Check:
    name = "db_exists"
    if db_path.exists():
        size_kb = db_path.stat().st_size // 1024
        return _pass(name, f"{db_path}  ({size_kb} KB)")
    return _fail(name, f"Database not found at {db_path}. "
                       f"Run: python3 ci/build_project_graph.py --rebuild")


def check_db_freshness(db_path: Path, stale_hours: float) -> Check:
    name = "db_freshness"
    threshold_secs = stale_hours * 3600
    db_mtime = db_path.stat().st_mtime

    newest_src: tuple[float, str] | None = None
    for rel_dir in WATCHED_DIRS:
        src_dir = LIB_ROOT / rel_dir
        if not src_dir.is_dir():
            continue
        for src_file in src_dir.rglob("*"):
            if not src_file.is_file():
                continue
            mtime = src_file.stat().st_mtime
            if newest_src is None or mtime > newest_src[0]:
                newest_src = (mtime, str(src_file.relative_to(LIB_ROOT)))

    if newest_src is None:
        return _warn(name, "No source files found in watched dirs; cannot assess freshness")

    src_mtime, src_path = newest_src
    age_secs = src_mtime - db_mtime

    if age_secs > threshold_secs:
        age_min = age_secs / 60
        return _fail(
            name,
            f"DB is stale by {age_min:.1f} min. "
            f"Newest source: {src_path}. "
            f"Run: python3 ci/build_project_graph.py --rebuild"
        )

    db_age_min = (time.time() - db_mtime) / 60
    return _pass(name, f"DB is current (built {db_age_min:.1f} min ago, "
                       f"threshold {stale_hours}h)")


def check_directory_coverage(conn: sqlite3.Connection, tables: set[str]) -> list[Check]:
    """Check that each key directory has enough indexed entries."""
    results: list[Check] = []

    # source_files is the canonical per-file table; fall back to a general count
    if "source_files" in tables:
        for rel_dir, floor in DIR_FLOORS.items():
            name = f"dir_coverage_{rel_dir}"
            count = conn.execute(
                "SELECT COUNT(*) AS cnt FROM source_files WHERE path LIKE ?",
                (f"{rel_dir}/%",),
            ).fetchone()["cnt"]

            if count == 0 and rel_dir not in ("docs", "scripts"):
                results.append(_fail(name,
                    f"{rel_dir}/: 0 entries indexed (floor {floor}). "
                    "Rebuild or expand source_graph.toml coverage."))
            elif count == 0:
                # docs/scripts are optional — warn only
                results.append(_warn(name,
                    f"{rel_dir}/: 0 entries indexed (optional; floor {floor})"))
            elif count < floor:
                results.append(_fail(name,
                    f"{rel_dir}/: only {count} entries (floor {floor})"))
            else:
                results.append(_pass(name, f"{rel_dir}/: {count} entries (floor {floor})"))
    else:
        results.append(_warn("dir_coverage",
            "Table 'source_files' not found; skipping directory coverage checks"))

    return results


def check_key_symbols(conn: sqlite3.Connection, tables: set[str]) -> list[Check]:
    """Check that mandatory symbols exist somewhere in the graph."""
    results: list[Check] = []

    # Tables that might store function/symbol names, in priority order
    candidate_tables: list[tuple[str, str]] = []
    if "c_abi_functions" in tables:
        candidate_tables.append(("c_abi_functions", "name"))
    if "function_index" in tables:
        candidate_tables.append(("function_index", "name"))
    if "cpp_methods" in tables:
        candidate_tables.append(("cpp_methods", "method"))

    if not candidate_tables:
        results.append(_warn("key_symbols",
            "No symbol tables found (c_abi_functions / function_index / cpp_methods). "
            "Skipping symbol presence checks."))
        return results

    for sym in REQUIRED_SYMBOLS:
        name = f"symbol_{sym}"
        found = False
        found_in = ""
        for table, col in candidate_tables:
            try:
                row = conn.execute(
                    f"SELECT 1 FROM {table} WHERE {col} LIKE ? LIMIT 1",
                    (f"%{sym}%",),
                ).fetchone()
                if row:
                    found = True
                    found_in = table
                    break
            except sqlite3.OperationalError:
                continue

        if found:
            results.append(_pass(name, f"'{sym}' found in {found_in}"))
        else:
            results.append(_fail(name,
                f"Symbol '{sym}' not found in any of "
                f"{[t for t, _ in candidate_tables]}. "
                "Rebuild the graph or check source_graph.toml coverage."))

    return results


def check_ct_metadata(conn: sqlite3.Connection, tables: set[str]) -> Check:
    """Verify that at least some CT-related metadata is indexed in the graph."""
    name = "ct_metadata"

    # Strategy 1: entity_tags with a CT-related tag
    if "entity_tags" in tables:
        try:
            count = conn.execute(
                "SELECT COUNT(*) AS cnt FROM entity_tags WHERE tag LIKE '%ct%' OR tag LIKE '%constant_time%'"
            ).fetchone()["cnt"]
            if count > 0:
                return _pass(name, f"{count} entity_tag rows with CT-related tags")
        except sqlite3.OperationalError:
            pass

    # Strategy 2: security_patterns with value_barrier / CLASSIFY / secure_erase
    if "security_patterns" in tables:
        try:
            count = conn.execute(
                "SELECT COUNT(*) AS cnt FROM security_patterns "
                "WHERE pattern IN ('value_barrier', 'CLASSIFY', 'secure_erase', 'ct_select')"
            ).fetchone()["cnt"]
            if count > 0:
                return _pass(name, f"{count} security_pattern rows with CT primitives")
        except sqlite3.OperationalError:
            pass

    # Strategy 3: symbol_security with must_be_constant_time=1
    if "symbol_security" in tables:
        try:
            count = conn.execute(
                "SELECT COUNT(*) AS cnt FROM symbol_security WHERE must_be_constant_time=1"
            ).fetchone()["cnt"]
            if count > 0:
                return _pass(name, f"{count} symbol_security rows marked must_be_constant_time")
        except sqlite3.OperationalError:
            pass

    # Strategy 4: metadata table with a CT key
    if "metadata" in tables:
        try:
            row = conn.execute(
                "SELECT 1 FROM metadata WHERE key LIKE '%ct%' OR key LIKE '%constant_time%' LIMIT 1"
            ).fetchone()
            if row:
                return _pass(name, "CT metadata key present in metadata table")
        except sqlite3.OperationalError:
            pass

    # Strategy 5: source_files containing ct_ in path
    if "source_files" in tables:
        try:
            count = conn.execute(
                "SELECT COUNT(*) AS cnt FROM source_files WHERE path LIKE '%ct_%' OR path LIKE '%/ct/%'"
            ).fetchone()["cnt"]
            if count > 0:
                return _warn(name,
                    f"{count} CT-named source files indexed but no CT metadata tags found. "
                    "Consider rebuilding graph with CT metadata extraction enabled.")
        except sqlite3.OperationalError:
            pass

    return _warn(name,
        "No CT metadata found in any known table "
        "(entity_tags / security_patterns / symbol_security / metadata). "
        "CT coverage evidence may be missing from the graph.")


def check_graph_size(conn: sqlite3.Connection, tables: set[str]) -> Check:
    """Fail if the graph looks suspiciously small (suggests a failed rebuild)."""
    name = "graph_size_sanity"

    total = 0
    per_table: dict[str, int] = {}
    for table in sorted(tables):
        cnt = _table_count(conn, table)
        per_table[table] = cnt
        total += cnt

    if total < TOTAL_ROW_FLOOR:
        top = sorted(per_table.items(), key=lambda x: x[1], reverse=True)[:5]
        top_str = ", ".join(f"{t}:{c}" for t, c in top)
        return _fail(name,
            f"Total row count {total} is below floor {TOTAL_ROW_FLOOR}. "
            f"Top tables: {top_str}. "
            "The graph may be empty or partially built. "
            "Run: python3 ci/build_project_graph.py --rebuild")

    return _pass(name, f"Total rows across {len(tables)} tables: {total}")


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _print_checks(checks: list[Check], json_mode: bool, stale_hours: float,
                  db_path: Path, elapsed_ms: float) -> int:
    """Print results and return exit code (0 = all pass/warn, 1 = any fail)."""
    fails = [c for c in checks if c.status == "FAIL"]
    warns = [c for c in checks if c.status == "WARN"]
    passed = [c for c in checks if c.status == "PASS"]
    overall = "FAIL" if fails else "PASS"

    if json_mode:
        payload = {
            "tool": "check_source_graph_quality",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "db_path": str(db_path),
            "stale_threshold_hours": stale_hours,
            "overall": overall,
            "summary": {
                "pass": len(passed),
                "warn": len(warns),
                "fail": len(fails),
                "total": len(checks),
            },
            "checks": [c.as_dict() for c in checks],
            "elapsed_ms": round(elapsed_ms, 1),
        }
        print(json.dumps(payload, indent=2))
    else:
        width = 64
        print("=" * width)
        print("Source Graph Quality Check")
        print("=" * width)
        print(f"  DB      : {db_path}")
        print(f"  Stale   : >{stale_hours}h")
        print(f"  Checks  : {len(checks)}")
        print()

        for c in checks:
            label = f"{c.color()}{c.status}{RESET}"
            print(f"  [{label}] {c.name}")
            if c.detail:
                # Indent continuation lines
                for line in c.detail.splitlines():
                    print(f"         {line}")

        print()
        print(f"  PASS:{len(passed)}  WARN:{len(warns)}  FAIL:{len(fails)}  "
              f"({elapsed_ms:.0f} ms)")
        print()

        color = RED if fails else GREEN
        print(f"  {BOLD}OVERALL: {color}{overall}{RESET}")

    return 1 if fails else 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify UltrafastSecp256k1 source graph quality."
    )
    parser.add_argument(
        "--db",
        metavar="PATH",
        default=str(DEFAULT_DB),
        help=f"Path to the graph database (default: {DEFAULT_DB})",
    )
    parser.add_argument(
        "--stale-hours",
        metavar="HOURS",
        type=float,
        default=DEFAULT_STALE_HOURS,
        help=f"Staleness threshold in hours (default: {DEFAULT_STALE_HOURS})",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON to stdout",
    )
    args = parser.parse_args(argv)

    db_path = Path(args.db).resolve()
    t0 = time.monotonic()

    checks: list[Check] = []

    # --- Check 1: DB existence ---
    existence = check_db_exists(db_path)
    checks.append(existence)
    if existence.status == "FAIL":
        # Nothing else can run without the DB
        elapsed = (time.monotonic() - t0) * 1000
        return _print_checks(checks, args.json, args.stale_hours, db_path, elapsed)

    # --- Check 2: Freshness ---
    checks.append(check_db_freshness(db_path, args.stale_hours))

    # --- Open connection ---
    try:
        conn = _open_db(db_path)
        tables = _list_tables(conn)
    except Exception as exc:
        checks.append(_fail("db_open", f"Cannot open DB: {exc}"))
        elapsed = (time.monotonic() - t0) * 1000
        return _print_checks(checks, args.json, args.stale_hours, db_path, elapsed)

    # --- Check 3: Directory coverage ---
    checks.extend(check_directory_coverage(conn, tables))

    # --- Check 4: Key symbols ---
    checks.extend(check_key_symbols(conn, tables))

    # --- Check 5: CT metadata ---
    checks.append(check_ct_metadata(conn, tables))

    # --- Check 6: Graph size sanity ---
    checks.append(check_graph_size(conn, tables))

    conn.close()

    elapsed = (time.monotonic() - t0) * 1000
    return _print_checks(checks, args.json, args.stale_hours, db_path, elapsed)


if __name__ == "__main__":
    sys.exit(main())
