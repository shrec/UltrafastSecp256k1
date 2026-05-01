#!/usr/bin/env python3
"""Review-queue aging check.

Reads the source-graph review_queue table, classifies rows by age, and
reports any that exceed the AUDIT_SLA review_queue_max_open_days threshold
without an explicit deferral marker.

Age source: tracked separately in `docs/REVIEW_QUEUE_AGES.json`. When a row
is first seen the script stores its first_seen ISO date there. On every run
it computes age = now - first_seen for rows still present in the graph, and
removes entries that no longer appear in the graph (i.e. were addressed).

Exit code: 0 always (warning-only SLO). Use --strict to exit 1 on violations.

Usage:
    python3 ci/review_queue_age_check.py
    python3 ci/review_queue_age_check.py --json -o review_queue_age.json
    python3 ci/review_queue_age_check.py --strict
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
LIB_ROOT = SCRIPT_DIR.parent
GRAPH_DB = LIB_ROOT / "tools" / "source_graph_kit" / "source_graph.db"
SLA_DEF = LIB_ROOT / "docs" / "AUDIT_SLA.json"
AGES_FILE = LIB_ROOT / "docs" / "REVIEW_QUEUE_AGES.json"


def _load_threshold() -> int:
    if not SLA_DEF.exists():
        return 90
    with SLA_DEF.open() as f:
        sla = json.load(f)
    return int(
        sla.get("slos", {})
           .get("review_queue_max_open_days", {})
           .get("threshold", 90)
    )


def _load_queue() -> list[dict]:
    if not GRAPH_DB.exists():
        return []
    conn = sqlite3.connect(str(GRAPH_DB))
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.execute(
            "SELECT file, queue_type, priority_score, rationale "
            "FROM review_queue ORDER BY file, queue_type"
        )
        return [dict(r) for r in cur.fetchall()]
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()


def _row_key(r: dict) -> str:
    return f"{r['file']}::{r['queue_type']}"


def _load_ages() -> dict:
    if not AGES_FILE.exists():
        return {}
    try:
        with AGES_FILE.open() as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_ages(ages: dict) -> None:
    AGES_FILE.parent.mkdir(parents=True, exist_ok=True)
    with AGES_FILE.open("w") as f:
        json.dump(ages, f, indent=2, sort_keys=True)
        f.write("\n")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--json", action="store_true", help="machine-readable output")
    p.add_argument("-o", "--output", type=Path, help="write JSON to file")
    p.add_argument("--strict", action="store_true", help="exit 1 on any violation")
    p.add_argument("--no-update", action="store_true", help="do not update REVIEW_QUEUE_AGES.json")
    args = p.parse_args()

    threshold_days = _load_threshold()
    now = datetime.now(timezone.utc)
    now_iso = now.isoformat()

    queue = _load_queue()
    ages = _load_ages()

    current_keys = set()
    findings = []
    for row in queue:
        key = _row_key(row)
        current_keys.add(key)
        if key not in ages:
            ages[key] = {"first_seen": now_iso}
            continue
        first_seen_str = ages[key].get("first_seen", now_iso)
        try:
            first_seen = datetime.fromisoformat(first_seen_str)
        except ValueError:
            first_seen = now
            ages[key]["first_seen"] = now_iso
        age_days = (now - first_seen).total_seconds() / 86400.0
        deferred = ages[key].get("deferred", False)
        if age_days > threshold_days and not deferred:
            findings.append({
                "file": row["file"],
                "queue_type": row["queue_type"],
                "priority_score": row["priority_score"],
                "rationale": row["rationale"],
                "first_seen": first_seen_str,
                "age_days": round(age_days, 1),
                "threshold_days": threshold_days,
            })

    # Drop rows that no longer appear in the graph (addressed).
    stale_keys = [k for k in ages.keys() if k not in current_keys]
    for k in stale_keys:
        del ages[k]

    if not args.no_update:
        _save_ages(ages)

    report = {
        "timestamp": now_iso,
        "threshold_days": threshold_days,
        "queue_total": len(queue),
        "violations": len(findings),
        "findings": findings,
    }

    out = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(out + "\n")
    elif args.json:
        print(out)
    else:
        print(f"Review-queue rows: {len(queue)}")
        print(f"Threshold: {threshold_days} days")
        print(f"Violations: {len(findings)}")
        for f in findings:
            print(f"  STALE [{f['queue_type']}] {f['file']} (age {f['age_days']}d > {threshold_days}d)")

    if args.strict and findings:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
