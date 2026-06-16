#!/usr/bin/env python3
"""Evidence-refresh coverage gate (Bastion B19, hardens RR-BAS-01 / RR-BAS-02).

Every freshness artifact tracked by ci/audit_sla_check.py must have an explicit
refresh disposition in docs/AUDIT_SLA.json `freshness_artifacts`, so a blocking
freshness SLO can never depend on an artifact that is silently neither
auto-refreshed nor documented as a residual:

  - the manifest must cover EVERY artifact audit_sla_check.py measures
    (no tracked artifact may be missing a disposition);
  - `auto` entries are cross-checked against the named workflow's actual commit
    list — the manifest cannot claim nightly coverage the workflow does not give;
  - `residual` entries must resolve to a docs/RESIDUAL_RISK_REGISTER.md id
    (authored specs, stable goldens, build-only artifacts, owner-chosen manual
    chores — never auto-regenerated because that would fake freshness);
  - a BLOCKING artifact with neither a verifiable producer nor a resolvable
    residual fails the gate (warning artifacts are advisory).

The `incident_drill_log` disposition is surfaced explicitly for RR-BAS-02: the
gate confirms the drill log is genuinely auto-refreshed by the lane (the
precondition for promoting incident_drill_freshness_days warning -> blocking).

Exit code:
  0  every tracked artifact is covered (auto producer verified or residual resolves)
  1  an uncovered/blocking artifact, an unverifiable auto-producer, or an unresolved residual
  2  the manifest is missing or malformed

Usage:
  python3 ci/check_evidence_refresh_coverage.py [--json] [-o report.json]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
LIB_ROOT = SCRIPT_DIR.parent

SLA_FILE = LIB_ROOT / "docs" / "AUDIT_SLA.json"
REGISTER_FILE = LIB_ROOT / "docs" / "RESIDUAL_RISK_REGISTER.md"

VALID_MODES = {"auto", "residual"}

# Canonical fallback: the artifact set ci/audit_sla_check.py measures, used only
# if that module cannot be imported. Kept in sync with CRITICAL_EVIDENCE there
# plus the incident-drill log (tracked by check_incident_drill_freshness).
_FALLBACK_TRACKED = {
    "assurance_report", "ct_evidence", "determinism_golden", "api_contracts",
    "assurance_claims", "risk_coverage_report", "incident_drill_log",
}


def _tracked_artifacts() -> set:
    """Authoritative set of freshness artifacts ci/audit_sla_check.py tracks.

    Derived from audit_sla_check.CRITICAL_EVIDENCE (+ the incident-drill log,
    which the SLA checker tracks via a dedicated check, not CRITICAL_EVIDENCE)
    so the manifest cannot silently drift from what actually gates."""
    try:
        if str(SCRIPT_DIR) not in sys.path:
            sys.path.insert(0, str(SCRIPT_DIR))
        import audit_sla_check  # stdlib-only imports; safe
        names = {ev["name"] for ev in audit_sla_check.CRITICAL_EVIDENCE}
        names.add("incident_drill_log")
        return names
    except Exception:
        return set(_FALLBACK_TRACKED)


def _workflow_committed_paths(workflow_text: str) -> set:
    """Paths the workflow actually stages (`for f in ...; do ... git add "$f"`
    loops plus literal `git add <path>` lines)."""
    paths: set = set()
    # `for f in A \ B \ ...; do` blocks (line continuations included via DOTALL).
    for body in re.findall(r"for\s+f\s+in\s+(.*?);\s*do", workflow_text, re.S):
        for tok in re.split(r"[\s\\]+", body):
            tok = tok.strip().strip('"').strip("'")
            if tok and ("/" in tok or "." in tok):
                paths.add(tok)
    # Literal `git add <path>` with a non-variable argument.
    for m in re.findall(r"git\s+add\s+(?!\"?\$)([^\s\"';]+)", workflow_text):
        m = m.strip().strip('"').strip("'")
        if m and m not in (".", "-A", "--all"):
            paths.add(m)
    return paths


def _residual_ids(register_text: str) -> set:
    """Residual ids defined in RESIDUAL_RISK_REGISTER.md (table rows + headings)."""
    ids: set = set()
    for line in register_text.splitlines():
        m = re.match(r"\s*\|\s*(RR-[A-Z0-9][A-Z0-9-]*)\s*\|", line)
        if m:
            ids.add(m.group(1))
        m = re.match(r"\s*#{1,6}\s+(RR-[A-Z0-9][A-Z0-9-]*)\b", line)
        if m:
            ids.add(m.group(1))
    return ids


def evaluate(artifacts: list, tracked: set, committed: set,
             residual_ids: set, existing_workflows: set) -> dict:
    """Evaluate the refresh-disposition manifest (pure / testable)."""
    rows = []
    by_name = {}
    for a in artifacts:
        name = a.get("name", "<no-name>")
        by_name[name] = a
        severity = a.get("severity", "blocking")
        refresh = a.get("refresh") or {}
        mode = refresh.get("mode")
        problems = []

        if mode not in VALID_MODES:
            problems.append(f"malformed:bad-mode:{mode!r}")
        elif mode == "auto":
            wf = refresh.get("producer_workflow", "")
            if wf not in existing_workflows:
                problems.append(f"missing_producer:workflow-not-found:{wf}")
            cp = refresh.get("committed_paths") or []
            if not cp:
                problems.append("malformed:auto-without-committed_paths")
            for p in cp:
                if p not in committed:
                    problems.append(f"missing_producer:not-committed-by-workflow:{p}")
        elif mode == "residual":
            rid = refresh.get("residual_id", "")
            if rid not in residual_ids:
                problems.append(f"unresolved_residual:{rid or '<none>'}")
            if not (refresh.get("reason") or "").strip():
                problems.append("malformed:residual-without-reason")

        rows.append({
            "name": name, "severity": severity, "mode": mode,
            "blocking": severity == "blocking", "problems": problems,
        })

    covered_names = set(by_name)
    uncovered_tracked = sorted(tracked - covered_names)
    phantom = sorted(covered_names - tracked)  # manifest entry not actually tracked

    def collect(prefix, blocking_only):
        out = []
        for r in rows:
            if blocking_only and not r["blocking"]:
                continue
            if any(p.split(":")[0] == prefix for p in r["problems"]):
                out.append(r["name"])
        return out

    malformed = collect("malformed", blocking_only=False)
    missing_producer = collect("missing_producer", blocking_only=True)
    unresolved_residual = collect("unresolved_residual", blocking_only=True)

    drill = by_name.get("incident_drill_log") or {}
    drill_refresh = drill.get("refresh") or {}
    incident_drill_autorefresh = (
        drill_refresh.get("mode") == "auto"
        and all(p in committed for p in (drill_refresh.get("committed_paths") or []))
        and bool(drill_refresh.get("committed_paths"))
    )

    overall = not (uncovered_tracked or malformed or missing_producer
                   or unresolved_residual or phantom)

    return {
        "overall_pass": overall,
        "artifacts_total": len(rows),
        "tracked_total": len(tracked),
        "auto_count": sum(1 for r in rows if r["mode"] == "auto"),
        "residual_count": sum(1 for r in rows if r["mode"] == "residual"),
        "uncovered_tracked": uncovered_tracked,
        "phantom_entries": phantom,
        "malformed": malformed,
        "missing_producer": missing_producer,
        "unresolved_residual": unresolved_residual,
        "incident_drill_autorefresh": incident_drill_autorefresh,
        "rows": rows,
    }


def load_and_evaluate() -> tuple[dict, int]:
    if not SLA_FILE.exists():
        return ({"overall_pass": False, "error": f"SLA file missing: {SLA_FILE}"}, 2)
    try:
        sla = json.loads(SLA_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        return ({"overall_pass": False, "error": f"AUDIT_SLA.json malformed: {exc}"}, 2)

    fa = sla.get("freshness_artifacts")
    if not isinstance(fa, dict) or not isinstance(fa.get("artifacts"), list) or not fa["artifacts"]:
        return ({"overall_pass": False,
                 "error": "AUDIT_SLA.json has no freshness_artifacts.artifacts[]"}, 2)

    artifacts = fa["artifacts"]
    tracked = _tracked_artifacts()

    # Producer-workflow commit lists + which producer workflows actually exist.
    existing_workflows = set()
    committed: set = set()
    for a in artifacts:
        wf = (a.get("refresh") or {}).get("producer_workflow")
        if wf:
            wf_path = LIB_ROOT / wf
            if wf_path.exists():
                existing_workflows.add(wf)
                committed |= _workflow_committed_paths(
                    wf_path.read_text(encoding="utf-8", errors="replace"))

    residual_ids = _residual_ids(
        REGISTER_FILE.read_text(encoding="utf-8", errors="replace")
        if REGISTER_FILE.exists() else "")

    report = evaluate(artifacts, tracked, committed, residual_ids, existing_workflows)
    return (report, 0 if report["overall_pass"] else 1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("-o", dest="out_file", help="Write report to file")
    args = parser.parse_args()

    report, code = load_and_evaluate()
    rendered = json.dumps(report, indent=2)
    if args.out_file:
        Path(args.out_file).write_text(rendered, encoding="utf-8")

    if args.json:
        print(rendered)
    else:
        if "error" in report:
            print(f"FAIL evidence-refresh coverage: {report['error']}")
            return code
        for r in report["rows"]:
            if r["problems"]:
                tag = "FAIL" if r["blocking"] else "WARN"
                print(f"  [{tag}] {r['name']} ({r['mode']}): {r['problems']}")
        for n in report["uncovered_tracked"]:
            print(f"  [FAIL] tracked artifact has no disposition in freshness_artifacts: {n}")
        for n in report["phantom_entries"]:
            print(f"  [FAIL] manifest entry is not a tracked freshness artifact: {n}")
        print()
        print(f"  incident_drill_log auto-refreshed by lane (RR-BAS-02 precondition): "
              f"{report['incident_drill_autorefresh']}")
        print()
        if report["overall_pass"]:
            print(f"PASS evidence-refresh coverage "
                  f"({report['auto_count']} auto, {report['residual_count']} residual, "
                  f"{report['tracked_total']} tracked artifacts all covered)")
        else:
            print(f"FAIL evidence-refresh coverage: uncovered={report['uncovered_tracked']} "
                  f"malformed={report['malformed']} missing_producer={report['missing_producer']} "
                  f"unresolved_residual={report['unresolved_residual']} phantom={report['phantom_entries']}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
