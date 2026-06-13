#!/usr/bin/env python3
"""External integration evidence freshness gate (Bastion B13).

Reads docs/INTEGRATION_EVIDENCE_STATUS.json and decides, per external-integration
surface (libsecp shim / cross-libsecp / libbitcoin bridge / Bitcoin Core), whether
the evidence is current (pass), stale, missing, or owner_gated. This turns
docs/INTEGRATION_EVIDENCE_TABLE.md from a human replay index into a freshness-gated
surface so CAAS knows the integration posture.

Verdict rules:
  - A `blocking` row fails the gate if its evidence_path is missing OR its
    last_verified is older than freshness_days (or the date is malformed).
  - A `warning` row in the same condition is advisory (does not fail the gate).
  - A `blocking`/`warning` row within pre_alert_buffer_days of expiry emits a
    PRE-ALERT (advisory) so refresh is signalled before the block.
  - `owner_gated` rows are reported explicitly and are NEVER counted as current
    evidence; staleness is surfaced as informational, not a block. They require an
    explicit owner re-verification / artifact import.

Exit code:
  0  no blocking failures (may have warnings / pre-alerts / owner-gated rows)
  1  at least one blocking row is missing or stale
  2  the manifest is missing or malformed

Usage:
  python3 ci/check_integration_evidence.py [--json] [-o report.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
LIB_ROOT = SCRIPT_DIR.parent
MANIFEST_PATH = LIB_ROOT / "docs" / "INTEGRATION_EVIDENCE_STATUS.json"

DEFAULT_PRE_ALERT_BUFFER_DAYS = 7
VALID_SEVERITY = {"blocking", "warning", "owner_gated"}


def _classify_row(row: dict, today, default_buffer: int) -> dict:
    """Compute the LIVE status of a row from evidence_path existence + last_verified.

    The manifest's stored `status` is documentation; the verdict is recomputed
    here so a row cannot be green merely because the JSON says so."""
    rid = row.get("id", "<no-id>")
    severity = row.get("severity", "")
    evidence_path = row.get("evidence_path", "") or ""
    freshness = row.get("freshness_days")
    buffer = int(row.get("pre_alert_buffer_days", default_buffer))

    out = {
        "id": rid,
        "surface": row.get("surface", ""),
        "severity": severity,
        "evidence_path": evidence_path,
        "freshness_days": freshness,
        "last_verified": row.get("last_verified", ""),
        "evidence_present": bool(evidence_path) and (LIB_ROOT / evidence_path).exists(),
        "computed_status": None,
        "age_days": None,
        "days_until_block": None,
        "pre_alert": False,
        "blocking_failure": False,
        "detail": "",
    }

    if severity not in VALID_SEVERITY:
        out["computed_status"] = "error"
        out["blocking_failure"] = True
        out["detail"] = f"invalid severity {severity!r}"
        return out

    # Age from last_verified (None when malformed/absent).
    age = None
    try:
        lv = datetime.strptime(str(row.get("last_verified", "")), "%Y-%m-%d").date()
        age = (today - lv).days
        out["age_days"] = age
        if isinstance(freshness, (int, float)):
            out["days_until_block"] = round(freshness - age, 1)
    except (ValueError, TypeError):
        age = None

    if severity == "owner_gated":
        # Explicit, never counted as current evidence; surface staleness as info.
        out["computed_status"] = "owner_gated"
        stale = age is not None and isinstance(freshness, (int, float)) and age > freshness
        out["owner_gated_stale"] = bool(stale)
        out["detail"] = (
            "owner-gated: requires explicit owner re-verification / artifact import"
            + (f" (last_verified {age}d ago, past {freshness}d window)" if stale else "")
        )
        return out

    # blocking / warning: recompute live status.
    if not out["evidence_present"]:
        out["computed_status"] = "missing"
        out["detail"] = f"evidence_path missing: {evidence_path}"
    elif age is None or not isinstance(freshness, (int, float)):
        out["computed_status"] = "missing"
        out["detail"] = f"last_verified {row.get('last_verified')!r} not a valid YYYY-MM-DD date"
    elif age > freshness:
        out["computed_status"] = "stale"
        out["detail"] = f"last_verified {age}d ago (> {freshness}d freshness SLO)"
    elif age > (freshness - buffer):
        out["computed_status"] = "pass"
        out["pre_alert"] = True
        out["detail"] = f"PRE-ALERT: {round(freshness - age, 1)}d until stale — re-verify soon"
    else:
        out["computed_status"] = "pass"
        out["detail"] = "current"

    out["blocking_failure"] = (
        severity == "blocking" and out["computed_status"] in ("missing", "stale", "error")
    )
    return out


def evaluate(manifest: dict, today=None) -> dict:
    """Evaluate a parsed manifest dict and return the report (pure / testable)."""
    if today is None:
        today = datetime.now(timezone.utc).date()
    default_buffer = int(manifest.get("default_pre_alert_buffer_days", DEFAULT_PRE_ALERT_BUFFER_DAYS))
    rows_in = manifest.get("rows")
    if not isinstance(rows_in, list) or not rows_in:
        return {
            "overall_pass": False,
            "error": "manifest has no rows[]",
            "rows": [], "stale_rows": [], "missing_rows": [],
            "owner_gated_rows": [], "pre_alerts": [], "min_days_until_block": None,
        }

    rows = [_classify_row(r, today, default_buffer) for r in rows_in]

    blocking_failures = [r for r in rows if r["blocking_failure"]]
    stale_rows = [r["id"] for r in rows if r["computed_status"] == "stale"]
    missing_rows = [r["id"] for r in rows if r["computed_status"] == "missing"]
    owner_gated_rows = [r["id"] for r in rows if r["computed_status"] == "owner_gated"]
    pre_alerts = [r["id"] for r in rows if r["pre_alert"]]

    measurable = [
        r["days_until_block"] for r in rows
        if r["severity"] != "owner_gated" and r["days_until_block"] is not None
    ]
    min_dub = min(measurable) if measurable else None

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "overall_pass": len(blocking_failures) == 0,
        "rows_total": len(rows),
        "blocking_total": sum(1 for r in rows if r["severity"] == "blocking"),
        "warning_total": sum(1 for r in rows if r["severity"] == "warning"),
        "owner_gated_total": len(owner_gated_rows),
        "blocking_failures": [r["id"] for r in blocking_failures],
        "stale_rows": stale_rows,
        "missing_rows": missing_rows,
        "owner_gated_rows": owner_gated_rows,
        "pre_alerts": pre_alerts,
        "min_days_until_block": min_dub,
        "rows": rows,
    }


def load_and_evaluate(path: Path, today=None) -> tuple[dict, int]:
    """Load the manifest file and evaluate it. Returns (report, exit_code)."""
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
            print(f"FAIL integration evidence: {report['error']}")
            return code
        for r in report["rows"]:
            tag = {"pass": "OK", "stale": "STALE", "missing": "MISSING",
                   "owner_gated": "OWNER", "error": "ERROR"}.get(r["computed_status"], "?")
            if r["pre_alert"]:
                tag = "ALERT"
            dub = r["days_until_block"]
            runway = "" if dub is None else f"  ({dub}d to block)"
            print(f"  [{tag:7}] {r['severity']:11} {r['id']}{runway}")
            if r["detail"] and r["computed_status"] in ("stale", "missing", "error") or r["pre_alert"]:
                print(f"            → {r['detail']}")
        print()
        if report["owner_gated_rows"]:
            print(f"  owner-gated (NOT current evidence — owner must re-verify): "
                  f"{', '.join(report['owner_gated_rows'])}")
        if report["overall_pass"]:
            print(f"PASS integration evidence ({len(report['pre_alerts'])} pre-alert(s), "
                  f"{report['owner_gated_total']} owner-gated)")
        else:
            print(f"FAIL integration evidence: blocking failures: {report['blocking_failures']}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
