#!/usr/bin/env python3
"""check_source_graph_quality.py  --  Source graph quality gate for UltrafastSecp256k1

Verifies that the canonical source_graph_kit database exists, is fresh, matches
HEAD, and covers the key source, CI, workflow, docs, and CT/security symbols
that CAAS depends on.

Checks
------
1. DB exists at tools/source_graph_kit/source_graph.db  -- FAIL if missing
2. DB is not stale  -- FAIL if DB mtime is older than any source file in the
   watched directories (configurable staleness threshold: 1 hour by default)
3. DB build revision matches current HEAD
4. Required source graph tables exist
5. Key projects/path classes are indexed above minimum floors
6. Mandatory CAAS files and crypto symbols are present
7. CT metadata present
8. Focus routing smoke tests keep graph queries auditable
9. Graph size sanity  -- FAIL if total entry count is below 200

Exit codes
----------
  0  all checks pass (may have WARN)
  1  one or more FAIL

Usage
-----
    python3 ci/check_source_graph_quality.py
    python3 ci/check_source_graph_quality.py --json
    python3 ci/check_source_graph_quality.py --db /path/to/source_graph.db
    python3 ci/check_source_graph_quality.py --stale-hours 2
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
LIB_ROOT = SCRIPT_DIR.parent

DEFAULT_DB = LIB_ROOT / "tools" / "source_graph_kit" / "source_graph.db"
DEFAULT_STALE_HOURS = 1

# Directories to watch for staleness (relative to LIB_ROOT)
# NOTE: cpu sources live under src/cpu/ after 2026-05 repo reorganization.
WATCHED_DIRS = ["src/cpu", "include", "compat", "audit", "ci", "docs", ".github", "tools/source_graph_kit"]

# Per-directory minimum entry counts for the legacy .project_graph.db
# compatibility path (keys are path prefixes in source_files.path).
DIR_FLOORS: dict[str, int] = {
    "src/cpu": 50,
    "include": 6,
    "audit":  100,
    "compat":  10,
    "docs":    30,
    "scripts": 2,   # post script→ci/ migration: clean_local_artifacts.sh + sync_all_docs.py
}

# Per-project minimum file counts for source_graph_kit/source_graph.db.
PROJECT_FLOORS: dict[str, int] = {
    "cpu": 150,
    "audit": 300,
    "ci": 120,
    "docs": 150,
    "github": 40,
    "compat": 30,
    "tools": 4,
}

REQUIRED_TABLES = [
    "files",
    "function_index",
    "symbol_metadata",
    "audit_coverage",
    "call_edges",
    "graph_metadata",
]

REQUIRED_FILES = [
    "check_source_graph_quality.py",
    "test_caas_integrity.py",
    "verify_external_audit_bundle.py",
    "external_audit_bundle.py",
    "workflows/caas.yml",
    "workflows/gate.yml",
    "workflows/preflight.yml",
    "CAAS_PROTOCOL.md",
]

# Symbols that must exist in the graph (checked against c_abi_functions,
# function_index, and cpp_methods depending on which tables exist)
REQUIRED_SYMBOLS = [
    "generator_mul",
    "ecdsa_sign",
    "schnorr_sign",
    "secp256k1_ecdsa_sign",
    "secp256k1_keypair_create",
]

FOCUS_GOLDENS: list[tuple[str, str]] = [
    ("source_graph_quality", "check_source_graph_quality.py"),
    ("caas", "workflows/caas.yml"),
    ("external_audit_bundle", "external_audit_bundle.py"),
    ("test_caas_integrity", "test_caas_integrity.py"),
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


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    try:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    except sqlite3.OperationalError:
        return set()
    return {str(row["name"]) for row in rows}


def _git_head() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(LIB_ROOT),
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except Exception:
        return ""
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_db_exists(db_path: Path) -> Check:
    name = "db_exists"
    if db_path.exists():
        size_kb = db_path.stat().st_size // 1024
        return _pass(name, f"{db_path}  ({size_kb} KB)")
    return _fail(name, f"Database not found at {db_path}. "
                       f"Run: python3 tools/source_graph_kit/source_graph.py build -i")


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
            f"Run: python3 tools/source_graph_kit/source_graph.py build -i"
        )

    db_age_min = (time.time() - db_mtime) / 60
    return _pass(name, f"DB is current (built {db_age_min:.1f} min ago, "
                       f"threshold {stale_hours}h)")


def check_required_tables(tables: set[str]) -> list[Check]:
    """Check that source_graph_kit's canonical schema is present."""
    results: list[Check] = []
    for table in REQUIRED_TABLES:
        if table in tables:
            results.append(_pass(f"table_{table}", "present"))
        else:
            results.append(_fail(
                f"table_{table}",
                f"required source_graph_kit table '{table}' missing; rebuild with "
                "python3 tools/source_graph_kit/source_graph.py build -i",
            ))
    return results


def check_build_revision(conn: sqlite3.Connection, tables: set[str]) -> Check:
    """Fail if source_graph.db was built for a different commit."""
    name = "graph_head_match"
    if "graph_metadata" not in tables:
        return _fail(name, "graph_metadata table missing; cannot verify commit binding")
    try:
        row = conn.execute(
            "SELECT graph_build_revision, built_at FROM graph_metadata ORDER BY id DESC LIMIT 1"
        ).fetchone()
    except sqlite3.OperationalError as exc:
        return _fail(name, f"cannot read graph_metadata: {exc}")
    if row is None:
        return _fail(name, "graph_metadata is empty; cannot verify commit binding")
    graph_rev = str(row["graph_build_revision"] or "")
    head = _git_head()
    if not head:
        return _fail(name, "cannot resolve git HEAD")
    if graph_rev != head:
        return _fail(
            name,
            f"source graph built for {graph_rev or '<empty>'}, current HEAD is {head}. "
            "Run: python3 tools/source_graph_kit/source_graph.py build -i",
        )
    return _pass(name, f"source graph revision matches HEAD {head[:12]} (built_at={row['built_at']})")


def check_directory_coverage(conn: sqlite3.Connection, tables: set[str]) -> list[Check]:
    """Check that each key directory has enough indexed entries."""
    results: list[Check] = []

    if "files" in tables:
        for project, floor in PROJECT_FLOORS.items():
            name = f"project_coverage_{project}"
            count = conn.execute(
                "SELECT COUNT(*) AS cnt FROM files WHERE project = ?",
                (project,),
            ).fetchone()["cnt"]
            if count < floor:
                results.append(_fail(
                    name,
                    f"{project}: only {count} files indexed (floor {floor}). "
                    "Update source_graph.toml or rebuild source_graph.db.",
                ))
            else:
                results.append(_pass(name, f"{project}: {count} files indexed (floor {floor})"))
        return results

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
    if "function_index" in tables and "function_name" in _table_columns(conn, "function_index"):
        candidate_tables.append(("function_index", "function_name"))
    if "symbol_metadata" in tables and "symbol_name" in _table_columns(conn, "symbol_metadata"):
        candidate_tables.append(("symbol_metadata", "symbol_name"))
    if "c_abi_functions" in tables and "name" in _table_columns(conn, "c_abi_functions"):
        candidate_tables.append(("c_abi_functions", "name"))
    if "cpp_methods" in tables and "method" in _table_columns(conn, "cpp_methods"):
        candidate_tables.append(("cpp_methods", "method"))
    if "symbols" in tables and "symbol_name" in _table_columns(conn, "symbols"):
        candidate_tables.append(("symbols", "symbol_name"))

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

    if "symbol_metadata" in tables:
        cols = _table_columns(conn, "symbol_metadata")
        if "ct_sensitive" in cols:
            try:
                count = conn.execute(
                    "SELECT COUNT(*) AS cnt FROM symbol_metadata WHERE ct_sensitive = 1"
                ).fetchone()["cnt"]
                if count > 0:
                    return _pass(name, f"{count} symbol_metadata rows marked ct_sensitive")
            except sqlite3.OperationalError:
                pass
        if "semantic_tags" in cols:
            try:
                count = conn.execute(
                    "SELECT COUNT(*) AS cnt FROM symbol_metadata "
                    "WHERE semantic_tags LIKE '%ct%' OR semantic_tags LIKE '%constant-time%'"
                ).fetchone()["cnt"]
                if count > 0:
                    return _pass(name, f"{count} symbol_metadata rows with CT semantic tags")
            except sqlite3.OperationalError:
                pass

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


def check_required_files(conn: sqlite3.Connection, tables: set[str]) -> list[Check]:
    """Ensure CAAS-critical scripts, docs, and workflows are indexed."""
    results: list[Check] = []
    if "files" not in tables:
        return [_fail("required_files", "files table missing; cannot verify CAAS file coverage")]
    for rel in REQUIRED_FILES:
        row = conn.execute("SELECT 1 FROM files WHERE path = ? LIMIT 1", (rel,)).fetchone()
        if row:
            results.append(_pass(f"file_{rel}", "indexed"))
        else:
            results.append(_fail(
                f"file_{rel}",
                f"{rel} is not indexed; update tools/source_graph_kit/source_graph.toml and rebuild",
            ))
    return results


def check_focus_routing() -> list[Check]:
    """Run low-cost focus query goldens for CAAS auditability."""
    script = LIB_ROOT / "tools" / "source_graph_kit" / "source_graph.py"
    if not script.exists():
        return [_fail("focus_routing", f"source graph CLI missing: {script}")]

    results: list[Check] = []
    for term, expected in FOCUS_GOLDENS:
        name = f"focus_{term}"
        try:
            result = subprocess.run(
                [sys.executable, str(script), "focus", term, "20", "--core"],
                cwd=str(LIB_ROOT),
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
            )
        except subprocess.TimeoutExpired:
            results.append(_fail(name, f"focus query timed out for {term!r}"))
            continue
        except Exception as exc:
            results.append(_fail(name, f"focus query failed to launch: {exc}"))
            continue
        output = result.stdout + result.stderr
        if result.returncode != 0:
            results.append(_fail(name, f"focus query exited {result.returncode}: {output[:240]}"))
        elif expected not in output:
            results.append(_fail(
                name,
                f"expected {expected!r} in focus output for {term!r}; got: {output[:240]}",
            ))
        else:
            results.append(_pass(name, f"{term!r} routes to {expected}"))
    return results


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
            "schema_version": "1.1.0",
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
    parser.add_argument(
        "--skip-focus-smoke",
        action="store_true",
        help="Skip source_graph.py focus routing smoke tests",
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

    # --- Check 3: Required canonical tables ---
    checks.extend(check_required_tables(tables))

    # --- Check 4: Commit binding ---
    checks.append(check_build_revision(conn, tables))

    # --- Check 5: Directory/project coverage ---
    checks.extend(check_directory_coverage(conn, tables))

    # --- Check 6: Required CAAS file coverage ---
    checks.extend(check_required_files(conn, tables))

    # --- Check 7: Key symbols ---
    checks.extend(check_key_symbols(conn, tables))

    # --- Check 8: CT metadata ---
    checks.append(check_ct_metadata(conn, tables))

    # --- Check 9: Graph size sanity ---
    checks.append(check_graph_size(conn, tables))

    conn.close()

    # --- Check 10: Focus routing smoke ---
    if not args.skip_focus_smoke:
        checks.extend(check_focus_routing())

    elapsed = (time.monotonic() - t0) * 1000
    return _print_checks(checks, args.json, args.stale_hours, db_path, elapsed)


if __name__ == "__main__":
    sys.exit(main())
