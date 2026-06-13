#!/usr/bin/env python3
"""Fuzz campaign evidence freshness + crash->regression gate (Bastion B15).

Reads docs/FUZZ_CAMPAIGN_STATUS.json and decides, per fuzz surface, whether the
corpus/harness/regression evidence is current (pass), stale, missing, has an
UNCONVERTED crash artifact (crash_unconverted), or is owner_gated. This makes
fuzzing evidence Bastion-grade: CAAS knows whether fuzz evidence is fresh and
whether every crash artifact was converted into a permanent regression.

This is an EVIDENCE-STATUS gate -- it does NOT run long fuzz campaigns. It is
cheap (filesystem + date checks) and safe on every push.

Verdict rules (blocking row):
  - fails if corpus_path is missing
  - fails if last_verified is stale/malformed
  - fails if crash_path contains crash artifacts without matching regression
    evidence (crash_unconverted)
A `warning` row is advisory on missing/stale (still fails on an unconverted crash,
because an unconverted crash is a correctness gap regardless of severity).
`owner_gated` rows are surfaced explicitly and are NEVER counted as current.

Exit code:
  0  no blocking failures
  1  a blocking row is missing / stale / has an unconverted crash
  2  the manifest is missing or malformed

Usage:
  python3 ci/check_fuzz_campaign_status.py [--json] [-o report.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
LIB_ROOT = SCRIPT_DIR.parent
MANIFEST_PATH = LIB_ROOT / "docs" / "FUZZ_CAMPAIGN_STATUS.json"

DEFAULT_PRE_ALERT_BUFFER_DAYS = 14
DEFAULT_CRASH_GLOBS = ["crash-*", "leak-*", "timeout-*", "oom-*", "slow-unit-*", "poc-*", "*.crash"]
VALID_SEVERITY = {"blocking", "warning", "owner_gated"}


def _resolve(p: str) -> Path:
    """Resolve a manifest path: absolute as-is, else relative to LIB_ROOT."""
    pp = Path(p)
    return pp if pp.is_absolute() else (LIB_ROOT / pp)


def _crash_artifacts(crash_path: str, globs: list) -> list:
    """Return crash artifact file names present under crash_path (empty if the
    directory is absent or holds no crash files)."""
    if not crash_path:
        return []
    d = _resolve(crash_path)
    if not d.is_dir():
        return []
    found = []
    for g in globs:
        for f in d.glob(g):
            if f.is_file():
                found.append(f.name)
    return sorted(set(found))


def _classify_row(row: dict, today, default_buffer: int, crash_globs: list) -> dict:
    rid = row.get("id", "<no-id>")
    severity = row.get("severity", "")
    corpus_path = row.get("corpus_path", "") or ""
    crash_path = row.get("crash_path", "") or ""
    regression_path = row.get("regression_path", "") or ""
    freshness = row.get("freshness_days")
    buffer = int(row.get("pre_alert_buffer_days", default_buffer))

    out = {
        "id": rid,
        "target": row.get("target", ""),
        "severity": severity,
        "corpus_path": corpus_path,
        "crash_path": crash_path,
        "regression_path": regression_path,
        "freshness_days": freshness,
        "last_verified": row.get("last_verified", ""),
        "corpus_present": bool(corpus_path) and _resolve(corpus_path).exists(),
        "regression_present": bool(regression_path) and _resolve(regression_path).exists(),
        "crash_artifacts": _crash_artifacts(crash_path, crash_globs),
        "computed_status": None,
        "age_days": None,
        "days_until_block": None,
        "pre_alert": False,
        "blocking_failure": False,
        "detail": "",
    }

    if severity not in VALID_SEVERITY:
        out["computed_status"] = "missing"
        out["blocking_failure"] = True
        out["detail"] = f"invalid severity {severity!r}"
        return out

    age = None
    try:
        lv = datetime.strptime(str(row.get("last_verified", "")), "%Y-%m-%d").date()
        age = (today - lv).days
        out["age_days"] = age
        if isinstance(freshness, (int, float)):
            out["days_until_block"] = round(freshness - age, 1)
    except (ValueError, TypeError):
        age = None

    # An unconverted crash (crash artifacts present but no regression evidence) is a
    # correctness gap for ANY severity except owner_gated (which is never current).
    crash_unconverted = bool(out["crash_artifacts"]) and not out["regression_present"]

    if severity == "owner_gated":
        out["computed_status"] = "owner_gated"
        stale = age is not None and isinstance(freshness, (int, float)) and age > freshness
        out["owner_gated_stale"] = bool(stale)
        out["detail"] = ("owner-gated: heavy/host-only fuzz; never current evidence"
                         + (f" (last_verified {age}d ago, past {freshness}d)" if stale else "")
                         + (f"; UNCONVERTED CRASHES: {out['crash_artifacts']}" if crash_unconverted else ""))
        # Even owner_gated must surface an unconverted crash, but it does not block.
        return out

    # blocking / warning
    if not out["corpus_present"]:
        out["computed_status"] = "missing"
        out["detail"] = f"corpus/harness path missing: {corpus_path}"
    elif age is None or not isinstance(freshness, (int, float)):
        out["computed_status"] = "missing"
        out["detail"] = f"last_verified {row.get('last_verified')!r} not a valid YYYY-MM-DD date"
    elif crash_unconverted:
        out["computed_status"] = "crash_unconverted"
        out["detail"] = (f"crash artifact(s) without a matching regression ({regression_path or '<none>'}): "
                         f"{out['crash_artifacts'][:5]}")
    elif age > freshness:
        out["computed_status"] = "stale"
        out["detail"] = f"campaign evidence last_verified {age}d ago (> {freshness}d SLO)"
    else:
        out["computed_status"] = "pass"
        out["detail"] = "current"
        if age > (freshness - buffer):
            out["pre_alert"] = True
            out["detail"] = f"PRE-ALERT: {round(freshness - age, 1)}d until campaign evidence stale — re-run"

    # blocking rows block on missing/stale/crash; warning rows block ONLY on an
    # unconverted crash (missing/stale are advisory for warning).
    if severity == "blocking":
        out["blocking_failure"] = out["computed_status"] in ("missing", "stale", "crash_unconverted")
    else:  # warning
        out["blocking_failure"] = out["computed_status"] == "crash_unconverted"
    return out


def evaluate(manifest: dict, today=None) -> dict:
    if today is None:
        today = datetime.now(timezone.utc).date()
    default_buffer = int(manifest.get("default_pre_alert_buffer_days", DEFAULT_PRE_ALERT_BUFFER_DAYS))
    crash_globs = manifest.get("crash_artifact_globs") or DEFAULT_CRASH_GLOBS
    rows_in = manifest.get("rows")
    if not isinstance(rows_in, list) or not rows_in:
        return {"overall_pass": False, "error": "manifest has no rows[]",
                "rows": [], "missing_rows": [], "stale_rows": [], "crash_unconverted_rows": [],
                "owner_gated_rows": [], "pre_alerts": [], "min_days_until_block": None}

    rows = [_classify_row(r, today, default_buffer, crash_globs) for r in rows_in]
    blocking_failures = [r for r in rows if r["blocking_failure"]]
    measurable = [r["days_until_block"] for r in rows
                  if r["severity"] != "owner_gated" and r["days_until_block"] is not None]

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "overall_pass": len(blocking_failures) == 0,
        "rows_total": len(rows),
        "blocking_total": sum(1 for r in rows if r["severity"] == "blocking"),
        "warning_total": sum(1 for r in rows if r["severity"] == "warning"),
        "owner_gated_total": sum(1 for r in rows if r["severity"] == "owner_gated"),
        "blocking_failures": [r["id"] for r in blocking_failures],
        "missing_rows": [r["id"] for r in rows if r["computed_status"] == "missing"],
        "stale_rows": [r["id"] for r in rows if r["computed_status"] == "stale"],
        "crash_unconverted_rows": [r["id"] for r in rows if r["computed_status"] == "crash_unconverted"],
        "owner_gated_rows": [r["id"] for r in rows if r["computed_status"] == "owner_gated"],
        "pre_alerts": [r["id"] for r in rows if r["pre_alert"]],
        "min_days_until_block": min(measurable) if measurable else None,
        "rows": rows,
    }


def load_and_evaluate(path: Path, today=None) -> tuple[dict, int]:
    if not path.exists():
        return ({"overall_pass": False, "error": f"manifest missing: {path}"}, 2)
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        return ({"overall_pass": False, "error": f"manifest malformed JSON: {exc}"}, 2)
    if not isinstance(manifest, dict):
        return ({"overall_pass": False, "error": "manifest root is not a JSON object"}, 2)
    report = evaluate(manifest, today=today)
    if "error" in report:
        return (report, 2)
    return (report, 0 if report["overall_pass"] else 1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("-o", dest="out_file", help="Write report to file")
    args = parser.parse_args()

    report, code = load_and_evaluate(MANIFEST_PATH)
    rendered = json.dumps(report, indent=2)
    if args.out_file:
        Path(args.out_file).write_text(rendered, encoding="utf-8")

    if args.json:
        print(rendered)
    else:
        if "error" in report:
            print(f"FAIL fuzz campaign: {report['error']}")
            return code
        for r in report["rows"]:
            tag = {"pass": "OK", "stale": "STALE", "missing": "MISSING",
                   "crash_unconverted": "CRASH!", "owner_gated": "OWNER"}.get(r["computed_status"], "?")
            if r["pre_alert"]:
                tag = "ALERT"
            dub = r["days_until_block"]
            runway = "" if dub is None else f"  ({dub}d to block)"
            print(f"  [{tag:8}] {r['severity']:11} {r['id']}{runway}")
            if r["computed_status"] in ("stale", "missing", "crash_unconverted") or r["pre_alert"]:
                print(f"            → {r['detail']}")
        print()
        if report["crash_unconverted_rows"]:
            print(f"  UNCONVERTED CRASHES (crash artifact without a regression): "
                  f"{', '.join(report['crash_unconverted_rows'])}")
        if report["owner_gated_rows"]:
            print(f"  owner-gated (not current evidence): {', '.join(report['owner_gated_rows'])}")
        if report["overall_pass"]:
            print(f"PASS fuzz campaign evidence ({len(report['pre_alerts'])} pre-alert(s), "
                  f"{report['owner_gated_total']} owner-gated)")
        else:
            print(f"FAIL fuzz campaign evidence: blocking failures: {report['blocking_failures']}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
