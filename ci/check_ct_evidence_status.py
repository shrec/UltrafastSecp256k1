#!/usr/bin/env python3
"""Constant-time (CT) evidence freshness + verdict-binding gate (Bastion B14).

Reads docs/CT_EVIDENCE_STATUS.json and decides, per CT-sensitive surface, whether
the evidence is current (pass), stale, missing, inconclusive, or owner_gated. This
turns CT claims from doc/table confidence into freshness-gated, tool-bound evidence.

Two dimensions:
  - committed evidence (CT primitive source + audit regression tests): checked on
    EVERY push (cheap). A blocking row fails if a path is missing or last_verified
    is stale/malformed.
  - tool verdicts (ct-verif / valgrind-ct / dudect, produced by the heavy CT CI
    workflows): evaluated ONLY when a verdict directory is supplied (--verdict-dir).
    When evaluated, a required-tool FAIL or a single PASS + SKIP is INCONCLUSIVE
    (never silently PASS); a blocking row fails if its required_tools do not all
    PASS. When no verdicts are present (push CI), the tool dimension is reported as
    not_evaluated and only the committed-evidence + freshness dimension gates.

owner_gated rows (e.g. host-only --gpu CT-uniformity) are surfaced explicitly and
are NEVER counted as current evidence.

Exit code:
  0  no blocking failures
  1  a blocking row is missing / stale / has a failing-or-inconclusive required tool
  2  the manifest is missing or malformed

Usage:
  python3 ci/check_ct_evidence_status.py [--json] [-o report.json] [--verdict-dir DIR]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
LIB_ROOT = SCRIPT_DIR.parent
MANIFEST_PATH = LIB_ROOT / "docs" / "CT_EVIDENCE_STATUS.json"

DEFAULT_PRE_ALERT_BUFFER_DAYS = 14
VALID_SEVERITY = {"blocking", "warning", "owner_gated"}


def _tool_status(required_tools: list, verdicts: dict) -> str:
    """Classify the required-tool verdict set: pass | fail | inconclusive.

    Reuses the ct-independence rule: any FAIL -> fail; all required PASS -> pass;
    a required tool that is SKIP / missing while others PASS -> inconclusive
    (a single PASS + SKIP must stay inconclusive, never PASS)."""
    if not required_tools:
        return "pass"  # no required tool to satisfy
    vs = [str(verdicts.get(t, "MISSING")).upper() for t in required_tools]
    if any(v == "FAIL" for v in vs):
        return "fail"
    if all(v == "PASS" for v in vs):
        return "pass"
    return "inconclusive"  # PASS + SKIP / missing -> inconclusive


def _classify_row(row: dict, today, default_buffer: int, verdicts) -> dict:
    rid = row.get("id", "<no-id>")
    severity = row.get("severity", "")
    evidence_paths = row.get("evidence_paths") or []
    freshness = row.get("freshness_days")
    buffer = int(row.get("pre_alert_buffer_days", default_buffer))
    required_tools = row.get("required_tools") or []

    out = {
        "id": rid,
        "surface": row.get("surface", ""),
        "ct_claim": row.get("ct_claim", ""),
        "severity": severity,
        "evidence_paths": evidence_paths,
        "required_tools": required_tools,
        "freshness_days": freshness,
        "last_verified": row.get("last_verified", ""),
        "evidence_present": bool(evidence_paths) and all((LIB_ROOT / p).exists() for p in evidence_paths),
        "computed_status": None,
        "tool_status": "not_evaluated",
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

    if severity == "owner_gated":
        out["computed_status"] = "owner_gated"
        stale = age is not None and isinstance(freshness, (int, float)) and age > freshness
        out["owner_gated_stale"] = bool(stale)
        out["detail"] = ("owner-gated: host-only / owner-run verdicts; never current evidence"
                         + (f" (last_verified {age}d ago, past {freshness}d)" if stale else ""))
        return out

    # blocking / warning: committed-evidence dimension first.
    missing_paths = [p for p in evidence_paths if not (LIB_ROOT / p).exists()]
    if not evidence_paths or missing_paths:
        out["computed_status"] = "missing"
        out["detail"] = f"evidence path(s) missing: {missing_paths or '<none declared>'}"
    elif age is None or not isinstance(freshness, (int, float)):
        out["computed_status"] = "missing"
        out["detail"] = f"last_verified {row.get('last_verified')!r} not a valid YYYY-MM-DD date"
    elif age > freshness:
        out["computed_status"] = "stale"
        out["detail"] = f"committed evidence last_verified {age}d ago (> {freshness}d SLO)"
    else:
        # present + fresh; now the tool-verdict dimension (only if verdicts supplied).
        if verdicts is None:
            out["computed_status"] = "pass"
            out["tool_status"] = "not_evaluated"
            out["detail"] = "committed evidence current; tool verdicts not present this run (CI-produced)"
            if age > (freshness - buffer):
                out["pre_alert"] = True
                out["detail"] = f"PRE-ALERT: {round(freshness - age, 1)}d until evidence stale — re-verify"
        else:
            ts = _tool_status(required_tools, verdicts)
            out["tool_status"] = ts
            if ts == "pass":
                out["computed_status"] = "pass"
                out["detail"] = f"required tools PASS: {required_tools}"
            elif ts == "fail":
                out["computed_status"] = "inconclusive"
                out["detail"] = f"a required CT tool reported FAIL (leakage): {required_tools}"
            else:
                out["computed_status"] = "inconclusive"
                out["detail"] = f"required tools not all PASS (PASS+SKIP/missing => inconclusive): {required_tools}"

    out["blocking_failure"] = severity == "blocking" and out["computed_status"] in ("missing", "stale", "inconclusive")
    return out


def evaluate(manifest: dict, today=None, verdicts=None) -> dict:
    """Evaluate a parsed manifest. `verdicts` is None (tool dimension not evaluated)
    or a {tool: verdict} dict. Pure / testable."""
    if today is None:
        today = datetime.now(timezone.utc).date()
    default_buffer = int(manifest.get("default_pre_alert_buffer_days", DEFAULT_PRE_ALERT_BUFFER_DAYS))
    rows_in = manifest.get("rows")
    if not isinstance(rows_in, list) or not rows_in:
        return {"overall_pass": False, "error": "manifest has no rows[]",
                "rows": [], "missing_rows": [], "stale_rows": [], "inconclusive_rows": [],
                "owner_gated_rows": [], "pre_alerts": [], "min_days_until_block": None,
                "verdicts_evaluated": verdicts is not None}

    rows = [_classify_row(r, today, default_buffer, verdicts) for r in rows_in]
    blocking_failures = [r for r in rows if r["blocking_failure"]]
    measurable = [r["days_until_block"] for r in rows
                  if r["severity"] != "owner_gated" and r["days_until_block"] is not None]

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "overall_pass": len(blocking_failures) == 0,
        "verdicts_evaluated": verdicts is not None,
        "rows_total": len(rows),
        "blocking_total": sum(1 for r in rows if r["severity"] == "blocking"),
        "warning_total": sum(1 for r in rows if r["severity"] == "warning"),
        "owner_gated_total": sum(1 for r in rows if r["severity"] == "owner_gated"),
        "blocking_failures": [r["id"] for r in blocking_failures],
        "missing_rows": [r["id"] for r in rows if r["computed_status"] == "missing"],
        "stale_rows": [r["id"] for r in rows if r["computed_status"] == "stale"],
        "inconclusive_rows": [r["id"] for r in rows if r["computed_status"] == "inconclusive"],
        "owner_gated_rows": [r["id"] for r in rows if r["computed_status"] == "owner_gated"],
        "pre_alerts": [r["id"] for r in rows if r["pre_alert"]],
        "min_days_until_block": min(measurable) if measurable else None,
        "rows": rows,
    }


def _load_verdicts(verdict_dir: Path) -> dict:
    """Load {tool: verdict} from ct-verdict-*.json files in a directory."""
    verdicts: dict = {}
    if not verdict_dir.is_dir():
        return verdicts
    for p in sorted(verdict_dir.glob("ct-verdict-*.json")):
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        tool = obj.get("tool")
        verdict = obj.get("verdict")
        if tool and verdict:
            verdicts[str(tool)] = str(verdict)
    return verdicts


def load_and_evaluate(path: Path, today=None, verdict_dir=None) -> tuple[dict, int]:
    if not path.exists():
        return ({"overall_pass": False, "error": f"manifest missing: {path}"}, 2)
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        return ({"overall_pass": False, "error": f"manifest malformed JSON: {exc}"}, 2)
    if not isinstance(manifest, dict):
        return ({"overall_pass": False, "error": "manifest root is not a JSON object"}, 2)
    verdicts = _load_verdicts(Path(verdict_dir)) if verdict_dir else None
    report = evaluate(manifest, today=today, verdicts=verdicts)
    if "error" in report:
        return (report, 2)
    return (report, 0 if report["overall_pass"] else 1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("-o", dest="out_file", help="Write report to file")
    parser.add_argument("--verdict-dir", default=None,
                        help="Directory of ct-verdict-*.json files to evaluate the tool dimension "
                             "(CI-produced; absent on push CI -> tool dimension not evaluated)")
    args = parser.parse_args()

    report, code = load_and_evaluate(MANIFEST_PATH, verdict_dir=args.verdict_dir)
    rendered = json.dumps(report, indent=2)
    if args.out_file:
        Path(args.out_file).write_text(rendered, encoding="utf-8")

    if args.json:
        print(rendered)
    else:
        if "error" in report:
            print(f"FAIL CT evidence: {report['error']}")
            return code
        for r in report["rows"]:
            tag = {"pass": "OK", "stale": "STALE", "missing": "MISSING",
                   "inconclusive": "INCONCL", "owner_gated": "OWNER"}.get(r["computed_status"], "?")
            if r["pre_alert"]:
                tag = "ALERT"
            dub = r["days_until_block"]
            runway = "" if dub is None else f"  ({dub}d to block)"
            print(f"  [{tag:8}] {r['severity']:11} {r['id']}{runway}")
            if r["computed_status"] in ("stale", "missing", "inconclusive") or r["pre_alert"]:
                print(f"            → {r['detail']}")
        print()
        print(f"  verdicts_evaluated: {report['verdicts_evaluated']}")
        if report["owner_gated_rows"]:
            print(f"  owner-gated (not current evidence): {', '.join(report['owner_gated_rows'])}")
        if report["overall_pass"]:
            print(f"PASS CT evidence ({len(report['pre_alerts'])} pre-alert(s), "
                  f"{len(report['inconclusive_rows'])} inconclusive, {report['owner_gated_total']} owner-gated)")
        else:
            print(f"FAIL CT evidence: blocking failures: {report['blocking_failures']}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
