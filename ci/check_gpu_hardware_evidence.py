#!/usr/bin/env python3
"""GPU / hardware evidence status gate (Bastion B16).

Reads docs/GPU_HARDWARE_EVIDENCE_STATUS.json and makes the GPU/hardware claim
surface explicit and freshness-gated: which backend claims are backed by
real-device evidence (owner-run; no GitHub GPU runners), which are host-side
fallback correctness (committed, runs without a GPU), and which are owner-gated /
documented residuals.

Integrity + verdict rules:
  - a `blocking` row fails if evidence_path is missing or last_verified is
    stale/malformed;
  - a `warning` row is advisory on missing/stale (pre-alerted);
  - `owner_gated` rows (real-device / host-only) are explicit and NEVER counted
    as current evidence;
  - `documented_residual` rows MUST resolve to a docs/RESIDUAL_RISK_REGISTER.md
    RR-id (an unresolved id fails);
  - a `fallback_correctness` row MUST name an existing fallback_path and is never
    counted as native GPU performance evidence; a `performance` row that names a
    fallback_path is a mislabel and fails.

Exit code:
  0  no blocking failures
  1  a blocking row / integrity failure / unresolved residual
  2  the manifest is missing or malformed

Usage:
  python3 ci/check_gpu_hardware_evidence.py [--json] [-o report.json]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
LIB_ROOT = SCRIPT_DIR.parent
MANIFEST_PATH = LIB_ROOT / "docs" / "GPU_HARDWARE_EVIDENCE_STATUS.json"
RESIDUAL_REGISTER = LIB_ROOT / "docs" / "RESIDUAL_RISK_REGISTER.md"

DEFAULT_PRE_ALERT_BUFFER_DAYS = 14
VALID_SEVERITY = {"blocking", "warning", "owner_gated", "documented_residual"}
VALID_CLAIM = {"correctness", "performance", "fallback_correctness", "hardware_ct", "out_of_scope"}


def _resolve(p: str) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (LIB_ROOT / pp)


def _load_rr_ids(register_path: Path) -> set:
    """Set of RR-* ids defined in RESIDUAL_RISK_REGISTER.md."""
    if not register_path.exists():
        return set()
    text = register_path.read_text(encoding="utf-8", errors="replace")
    return set(re.findall(r"\bRR-[A-Z0-9-]+\b", text))


def _classify_row(row: dict, today, default_buffer: int, rr_ids: set) -> dict:
    rid = row.get("id", "<no-id>")
    claim_type = row.get("claim_type", "")
    severity = row.get("severity", "")
    evidence_path = row.get("evidence_path", "") or ""
    fallback_path = row.get("fallback_path", "") or ""
    residual_id = row.get("residual_risk_id", "") or ""
    freshness = row.get("freshness_days")
    buffer = int(row.get("pre_alert_buffer_days", default_buffer))

    out = {
        "id": rid,
        "backend": row.get("backend", ""),
        "operation": row.get("operation", ""),
        "claim_type": claim_type,
        "severity": severity,
        "evidence_path": evidence_path,
        "fallback_path": fallback_path,
        "residual_risk_id": residual_id,
        "hardware_required": bool(row.get("hardware_required", False)),
        "freshness_days": freshness,
        "last_verified": row.get("last_verified", ""),
        "evidence_present": bool(evidence_path) and _resolve(evidence_path).exists(),
        "computed_status": None,
        "age_days": None,
        "days_until_block": None,
        "pre_alert": False,
        "residual_unresolved": False,
        "blocking_failure": False,
        "detail": "",
    }

    if severity not in VALID_SEVERITY:
        out["computed_status"] = "missing"
        out["blocking_failure"] = True
        out["detail"] = f"invalid severity {severity!r}"
        return out
    if claim_type not in VALID_CLAIM:
        out["computed_status"] = "missing"
        out["blocking_failure"] = True
        out["detail"] = f"invalid claim_type {claim_type!r}"
        return out

    # age
    age = None
    try:
        lv = datetime.strptime(str(row.get("last_verified", "")), "%Y-%m-%d").date()
        age = (today - lv).days
        out["age_days"] = age
        if isinstance(freshness, (int, float)):
            out["days_until_block"] = round(freshness - age, 1)
    except (ValueError, TypeError):
        age = None

    # (1) claim_type integrity (applies to every severity).
    if claim_type == "fallback_correctness":
        if not fallback_path or not _resolve(fallback_path).exists():
            out["computed_status"] = "missing"
            out["blocking_failure"] = True
            out["detail"] = f"fallback_correctness requires an existing fallback_path (got {fallback_path!r})"
            return out
    elif claim_type == "performance" and fallback_path:
        out["computed_status"] = "missing"
        out["blocking_failure"] = True
        out["detail"] = "performance claim must not name a fallback_path (fallback mislabeled as native performance)"
        return out

    # (2) documented_residual must resolve to an RR id.
    if severity == "documented_residual":
        if not residual_id or residual_id not in rr_ids:
            out["computed_status"] = "missing"
            out["residual_unresolved"] = True
            out["blocking_failure"] = True
            out["detail"] = f"documented_residual must resolve to a RESIDUAL_RISK_REGISTER.md id (got {residual_id!r})"
            return out
        out["computed_status"] = "documented_residual"
        out["detail"] = f"documented residual {residual_id} (RESIDUAL_RISK_REGISTER.md)"
        if age is not None and isinstance(freshness, (int, float)) and age > freshness:
            out["detail"] += f" — last_verified {age}d ago (past {freshness}d)"
        return out

    # (3) owner_gated: explicit, never current.
    if severity == "owner_gated":
        out["computed_status"] = "owner_gated"
        stale = age is not None and isinstance(freshness, (int, float)) and age > freshness
        out["owner_gated_stale"] = bool(stale)
        out["detail"] = ("owner-gated: real-device / host-only evidence; never current on push"
                         + (f" (last_verified {age}d ago, past {freshness}d)" if stale else "")
                         + (f"; residual {residual_id}" if residual_id else ""))
        return out

    # (4) blocking / warning: committed evidence + freshness.
    if not out["evidence_present"]:
        out["computed_status"] = "missing"
        out["detail"] = f"evidence_path missing: {evidence_path}"
    elif age is None or not isinstance(freshness, (int, float)):
        out["computed_status"] = "missing"
        out["detail"] = f"last_verified {row.get('last_verified')!r} not a valid YYYY-MM-DD date"
    elif age > freshness:
        out["computed_status"] = "stale"
        out["detail"] = f"last_verified {age}d ago (> {freshness}d SLO)"
    else:
        out["computed_status"] = "pass"
        out["detail"] = "current"
        if age > (freshness - buffer):
            out["pre_alert"] = True
            out["detail"] = f"PRE-ALERT: {round(freshness - age, 1)}d until stale — re-verify"

    out["blocking_failure"] = severity == "blocking" and out["computed_status"] in ("missing", "stale")
    return out


def evaluate(manifest: dict, today=None, rr_ids=None) -> dict:
    if today is None:
        today = datetime.now(timezone.utc).date()
    if rr_ids is None:
        rr_ids = _load_rr_ids(RESIDUAL_REGISTER)
    default_buffer = int(manifest.get("default_pre_alert_buffer_days", DEFAULT_PRE_ALERT_BUFFER_DAYS))
    rows_in = manifest.get("rows")
    if not isinstance(rows_in, list) or not rows_in:
        return {"overall_pass": False, "error": "manifest has no rows[]",
                "rows": [], "missing_rows": [], "stale_rows": [], "owner_gated_rows": [],
                "documented_residual_rows": [], "unresolved_residual_rows": [], "pre_alerts": [],
                "min_days_until_block": None}

    rows = [_classify_row(r, today, default_buffer, rr_ids) for r in rows_in]
    blocking_failures = [r for r in rows if r["blocking_failure"]]
    measurable = [r["days_until_block"] for r in rows
                  if r["severity"] in ("blocking", "warning") and r["days_until_block"] is not None]

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "overall_pass": len(blocking_failures) == 0,
        "rows_total": len(rows),
        "blocking_total": sum(1 for r in rows if r["severity"] == "blocking"),
        "warning_total": sum(1 for r in rows if r["severity"] == "warning"),
        "owner_gated_total": sum(1 for r in rows if r["severity"] == "owner_gated"),
        "documented_residual_total": sum(1 for r in rows if r["severity"] == "documented_residual"),
        "fallback_correctness_rows": [r["id"] for r in rows if r["claim_type"] == "fallback_correctness"],
        "blocking_failures": [r["id"] for r in blocking_failures],
        "missing_rows": [r["id"] for r in rows if r["computed_status"] == "missing"],
        "stale_rows": [r["id"] for r in rows if r["computed_status"] == "stale"],
        "owner_gated_rows": [r["id"] for r in rows if r["computed_status"] == "owner_gated"],
        "documented_residual_rows": [r["id"] for r in rows if r["computed_status"] == "documented_residual"],
        "unresolved_residual_rows": [r["id"] for r in rows if r["residual_unresolved"]],
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
            print(f"FAIL GPU/hardware evidence: {report['error']}")
            return code
        for r in report["rows"]:
            tag = {"pass": "OK", "stale": "STALE", "missing": "MISSING",
                   "owner_gated": "OWNER", "documented_residual": "RESIDUAL"}.get(r["computed_status"], "?")
            if r["pre_alert"]:
                tag = "ALERT"
            dub = r["days_until_block"]
            runway = "" if dub is None else f"  ({dub}d to block)"
            print(f"  [{tag:8}] {r['severity']:18} {r['claim_type']:20} {r['id']}{runway}")
            if r["computed_status"] in ("stale", "missing") or r["pre_alert"]:
                print(f"            → {r['detail']}")
        print()
        if report["fallback_correctness_rows"]:
            print(f"  fallback-correctness (NOT native-performance evidence): "
                  f"{', '.join(report['fallback_correctness_rows'])}")
        if report["owner_gated_rows"]:
            print(f"  owner-gated (not current evidence): {', '.join(report['owner_gated_rows'])}")
        if report["documented_residual_rows"]:
            print(f"  documented residuals: {', '.join(report['documented_residual_rows'])}")
        if report["overall_pass"]:
            print(f"PASS GPU/hardware evidence ({report['owner_gated_total']} owner-gated, "
                  f"{report['documented_residual_total']} documented-residual)")
        else:
            print(f"FAIL GPU/hardware evidence: blocking failures: {report['blocking_failures']}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
