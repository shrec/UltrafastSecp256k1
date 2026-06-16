#!/usr/bin/env python3
"""Audit SLA/SLO compliance checker — fail-closed release gate.

Checks measurable SLO thresholds:
  - Evidence staleness (max age of critical evidence artifacts)
  - Critical-path evidence freshness (stricter threshold)
  - Determinism golden reference freshness

For every tracked artifact the checker also reports `days_until_block` and emits
a non-blocking PRE-ALERT warning while the artifact is within `pre_alert_buffer_days`
of its blocking threshold. This prevents the SLA from silently jumping straight
from green to blocked (Bastion B3): operators get advance warning, and the
`evidence_status` array gives every artifact's remaining runway.

If any blocking SLO is violated, release_ready = false.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
LIB_ROOT = SCRIPT_DIR.parent

SLA_DEF_FILE = LIB_ROOT / "docs" / "AUDIT_SLA.json"

# Critical evidence artifacts and their expected locations
CRITICAL_EVIDENCE: list[dict] = [
    {"name": "assurance_report", "path": "out/reports/assurance_report.json", "scope": "suite"},
    {"name": "ct_evidence", "path": "audit/ci-evidence", "scope": "lib", "is_dir": True},
    {"name": "determinism_golden", "path": "docs/DETERMINISM_GOLDEN.json", "scope": "lib"},
    {"name": "api_contracts", "path": "docs/API_SECURITY_CONTRACTS.json", "scope": "lib"},
    {"name": "assurance_claims", "path": "docs/ASSURANCE_CLAIMS.json", "scope": "lib"},
    {"name": "risk_coverage_report", "path": "out/reports/risk_surface_report.json", "scope": "build"},
]

SUITE_ROOT = LIB_ROOT.parent.parent  # workspace root

# Standalone mode: true when running inside the library repo directly (e.g. CI
# checkout of UltrafastSecp256k1 alone, without the surrounding suite tree).
# In standalone mode, suite-scoped evidence files are simply not present, and
# missing them should be a warning rather than a blocking violation.
_STANDALONE = not (SUITE_ROOT / "libs").is_dir()

# Pre-alert: warn this many days BEFORE an artifact would cross its blocking
# freshness threshold, so the SLA never silently jumps from green to blocked.
# Overridable per-SLO via AUDIT_SLA.json "pre_alert_buffer_days".
DEFAULT_PRE_ALERT_BUFFER_DAYS = 5


def _load_sla_defs() -> dict:
    """Load SLA definitions."""
    if not SLA_DEF_FILE.exists():
        return {}
    return json.loads(SLA_DEF_FILE.read_text(encoding="utf-8"))


def _pre_alert_buffer(sla_defs: dict, slo: str) -> int:
    """Per-SLO pre-alert buffer (days), falling back to the global default."""
    try:
        return int(sla_defs.get("slos", {}).get(slo, {}).get(
            "pre_alert_buffer_days", DEFAULT_PRE_ALERT_BUFFER_DAYS))
    except (TypeError, ValueError):
        return DEFAULT_PRE_ALERT_BUFFER_DAYS


def _classify(age: float | None, threshold: float, buffer: int) -> str:
    """ok | pre_alert | stale | missing."""
    if age is None:
        return "missing"
    if age > threshold:
        return "stale"
    if age > (threshold - buffer):
        return "pre_alert"
    return "ok"


def _status_row(name: str, slo: str, age: float | None, threshold: float, buffer: int) -> dict:
    """Per-artifact freshness status with days_until_block for ops visibility."""
    return {
        "evidence": name,
        "slo": slo,
        "state": _classify(age, threshold, buffer),
        "age_days": None if age is None else round(age, 1),
        "threshold_days": threshold,
        "days_until_block": None if age is None else round(threshold - age, 1),
    }


def _file_age_days(path: Path) -> float | None:
    """Return age of file in days based on git commit date, or None if missing/untracked.

    Uses git log rather than mtime because CI checkouts reset mtime to "now",
    making filesystem timestamps useless for staleness detection.
    Falls back to mtime only when the file is not tracked by git (e.g. generated files
    outside the repo, or fresh local writes not yet committed).
    """
    if not path.exists():
        return None
    try:
        result = subprocess.run(
            ["git", "log", "-1", "--format=%ct", "--", str(path)],
            capture_output=True,
            text=True,
            cwd=str(LIB_ROOT),
            timeout=15,
        )
        ts_str = result.stdout.strip()
        if ts_str:
            age_secs = time.time() - float(ts_str)
            return age_secs / 86400.0
    except (subprocess.SubprocessError, ValueError, OSError):
        pass
    # Fallback to mtime for untracked/local-only files
    mtime = os.path.getmtime(path)
    age_secs = time.time() - mtime
    return age_secs / 86400.0


def _dir_newest_age_days(dirpath: Path) -> float | None:
    """Return age (days) of the newest file in a directory."""
    if not dirpath.is_dir():
        return None
    newest = None
    for f in dirpath.rglob("*"):
        if f.is_file():
            age = _file_age_days(f)
            if age is not None:
                if newest is None or age < newest:
                    newest = age
    return newest


def check_evidence_staleness(sla_defs: dict) -> tuple[list[dict], list[dict]]:
    """Check that no critical evidence is staler than the SLO threshold.

    Returns (findings, statuses). Findings carry blocking 'stale' violations,
    missing-artifact entries, and non-blocking 'pre_alert' warnings. Statuses
    carry every artifact's days_until_block for the report's evidence_status."""
    slo = "max_stale_evidence_days"
    threshold = sla_defs.get("slos", {}).get(slo, {}).get("threshold", 30)
    severity = sla_defs.get("slos", {}).get(slo, {}).get("severity", "blocking")
    buffer = _pre_alert_buffer(sla_defs, slo)
    findings: list[dict] = []
    statuses: list[dict] = []

    for ev in CRITICAL_EVIDENCE:
        if ev["scope"] == "suite":
            base = SUITE_ROOT
        else:
            base = LIB_ROOT  # lib + build fallback

        full_path = base / ev["path"]
        is_dir = ev.get("is_dir", False)
        age = _dir_newest_age_days(full_path) if is_dir else _file_age_days(full_path)

        statuses.append(_status_row(ev["name"], slo, age, threshold, buffer))

        if age is None:
            # In standalone mode OR outside CI (local dev), suite- and
            # build-scoped artifacts are structurally absent (suite: no
            # surrounding repo tree; build: out/ is gitignored and only
            # exists after a local or full-CI build).  Downgrade to warning
            # so the SLA gate does not block on files that cannot exist in a
            # clean checkout.
            _in_ci = os.environ.get("GITHUB_ACTIONS") == "true"
            effective_severity = (
                "warning"
                if ((_STANDALONE or not _in_ci) and ev["scope"] in ("suite", "build"))
                else severity
            )
            findings.append({
                "slo": slo,
                "evidence": ev["name"],
                "status": "missing",
                "severity": effective_severity,
                "detail": f"Evidence artifact not found: {full_path}",
            })
        elif age > threshold:
            findings.append({
                "slo": slo,
                "evidence": ev["name"],
                "status": "stale",
                "age_days": round(age, 1),
                "threshold_days": threshold,
                "days_until_block": round(threshold - age, 1),
                "severity": severity,
                "detail": f"{ev['name']} is {round(age, 1)} days old (max {threshold})",
            })
        elif age > (threshold - buffer):
            findings.append({
                "slo": slo,
                "evidence": ev["name"],
                "status": "pre_alert",
                "age_days": round(age, 1),
                "threshold_days": threshold,
                "days_until_block": round(threshold - age, 1),
                "severity": "warning",
                "detail": (f"PRE-ALERT: {ev['name']} is {round(age, 1)} days old; "
                           f"blocks at {threshold} ({round(threshold - age, 1)} days left) — refresh now"),
            })

    return findings, statuses


def check_critical_freshness(sla_defs: dict) -> tuple[list[dict], list[dict]]:
    """Check critical-path evidence freshness (stricter threshold)."""
    slo = "critical_evidence_freshness_days"
    threshold = sla_defs.get("slos", {}).get(slo, {}).get("threshold", 14)
    severity = sla_defs.get("slos", {}).get(slo, {}).get("severity", "blocking")
    buffer = _pre_alert_buffer(sla_defs, slo)
    findings: list[dict] = []
    statuses: list[dict] = []

    critical_files = [
        ("ct_evidence_dir", LIB_ROOT / "audit" / "ci-evidence", True),
        ("api_contracts", LIB_ROOT / "docs" / "API_SECURITY_CONTRACTS.json", False),
    ]

    for name, path, is_dir in critical_files:
        age = _dir_newest_age_days(path) if is_dir else _file_age_days(path)
        statuses.append(_status_row(name, slo, age, threshold, buffer))

        if age is None:
            continue
        if age > threshold:
            findings.append({
                "slo": slo,
                "evidence": name,
                "status": "stale",
                "age_days": round(age, 1),
                "threshold_days": threshold,
                "days_until_block": round(threshold - age, 1),
                "severity": severity,
                "detail": f"Critical evidence {name} is {round(age, 1)} days old (max {threshold})",
            })
        elif age > (threshold - buffer):
            findings.append({
                "slo": slo,
                "evidence": name,
                "status": "pre_alert",
                "age_days": round(age, 1),
                "threshold_days": threshold,
                "days_until_block": round(threshold - age, 1),
                "severity": "warning",
                "detail": (f"PRE-ALERT: critical evidence {name} is {round(age, 1)} days old; "
                           f"blocks at {threshold} ({round(threshold - age, 1)} days left) — refresh now"),
            })

    return findings, statuses


def check_determinism_golden_freshness(sla_defs: dict) -> tuple[list[dict], list[dict]]:
    """Check determinism golden reference freshness."""
    slo = "determinism_golden_staleness_days"
    threshold = sla_defs.get("slos", {}).get(slo, {}).get("threshold", 30)
    severity = sla_defs.get("slos", {}).get(slo, {}).get("severity", "blocking")
    buffer = _pre_alert_buffer(sla_defs, slo)
    golden = LIB_ROOT / "docs" / "DETERMINISM_GOLDEN.json"
    age = _file_age_days(golden)
    statuses = [_status_row("determinism_golden", slo, age, threshold, buffer)]

    if age is None:
        return [{
            "slo": slo,
            "evidence": "determinism_golden",
            "status": "missing",
            "severity": severity,
            "detail": "DETERMINISM_GOLDEN.json not yet created — will be generated by determinism gate",
        }], statuses
    if age > threshold:
        return [{
            "slo": slo,
            "evidence": "determinism_golden",
            "status": "stale",
            "age_days": round(age, 1),
            "threshold_days": threshold,
            "days_until_block": round(threshold - age, 1),
            "severity": severity,
            "detail": f"Determinism golden is {round(age, 1)} days old (max {threshold})",
        }], statuses
    if age > (threshold - buffer):
        return [{
            "slo": slo,
            "evidence": "determinism_golden",
            "status": "pre_alert",
            "age_days": round(age, 1),
            "threshold_days": threshold,
            "days_until_block": round(threshold - age, 1),
            "severity": "warning",
            "detail": (f"PRE-ALERT: determinism golden is {round(age, 1)} days old; "
                       f"blocks at {threshold} ({round(threshold - age, 1)} days left) — re-verify now"),
        }], statuses
    return [], statuses


def check_incident_drill_freshness(sla_defs: dict) -> tuple[list[dict], list[dict]]:
    """Check that incident drills are still running (Bastion B9).

    Reads the git commit date of docs/INCIDENT_DRILL_LOG.json (written every
    incident_drills.py run, refreshed nightly). If drills stop running the log
    goes stale and this surfaces it. Advisory (warning) for now — see the SLO
    note in AUDIT_SLA.json."""
    slo = "incident_drill_freshness_days"
    threshold = sla_defs.get("slos", {}).get(slo, {}).get("threshold", 14)
    severity = sla_defs.get("slos", {}).get(slo, {}).get("severity", "warning")
    buffer = _pre_alert_buffer(sla_defs, slo)
    log = LIB_ROOT / "docs" / "INCIDENT_DRILL_LOG.json"
    age = _file_age_days(log)
    statuses = [_status_row("incident_drill_log", slo, age, threshold, buffer)]

    if age is None:
        return [], statuses  # not yet committed; absence is not a violation here
    if age > threshold:
        return [{
            "slo": slo, "evidence": "incident_drill_log", "status": "stale",
            "age_days": round(age, 1), "threshold_days": threshold,
            "days_until_block": round(threshold - age, 1), "severity": severity,
            "detail": f"Incident-drill log is {round(age, 1)} days old (max {threshold}) — drills may have stopped running",
        }], statuses
    if age > (threshold - buffer):
        return [{
            "slo": slo, "evidence": "incident_drill_log", "status": "pre_alert",
            "age_days": round(age, 1), "threshold_days": threshold,
            "days_until_block": round(threshold - age, 1), "severity": "warning",
            "detail": f"PRE-ALERT: incident-drill log is {round(age, 1)} days old; stales at {threshold} — re-run drills",
        }], statuses
    return [], statuses


def run(json_mode: bool, out_file: str | None) -> int:
    sla_defs = _load_sla_defs()

    all_findings: list[dict] = []
    evidence_status: list[dict] = []
    for check in (check_evidence_staleness, check_critical_freshness,
                  check_determinism_golden_freshness, check_incident_drill_freshness):
        findings, statuses = check(sla_defs)
        all_findings.extend(findings)
        evidence_status.extend(statuses)

    blocking = [f for f in all_findings if f.get("severity") == "blocking"]
    warnings = [f for f in all_findings if f.get("severity") == "warning"]
    pre_alerts = [f for f in all_findings if f.get("status") == "pre_alert"]

    # Soonest blocking deadline across all tracked artifacts (None when nothing
    # has a measurable age, e.g. clean checkout with no built evidence).
    measurable = [s["days_until_block"] for s in evidence_status
                  if s.get("days_until_block") is not None]
    min_days_until_block = min(measurable) if measurable else None

    release_ready = len(blocking) == 0

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "overall_pass": release_ready,
        "release_ready": release_ready,
        "blocking_violations": len(blocking),
        "warning_violations": len(warnings),
        "pre_alerts": len(pre_alerts),
        "min_days_until_block": min_days_until_block,
        "total_findings": len(all_findings),
        "evidence_status": evidence_status,
        "findings": all_findings,
    }

    rendered = json.dumps(report, indent=2)
    if out_file:
        Path(out_file).write_text(rendered, encoding="utf-8")

    if json_mode:
        print(rendered)
    else:
        for f in all_findings:
            sev = f.get("severity")
            tag = "BLOCK" if sev == "blocking" else ("ALERT" if f.get("status") == "pre_alert" else "WARN")
            print(f"  [{tag}] {f['detail']}")
        # Runway summary for every tracked artifact (days_until_block).
        print()
        for s in sorted(evidence_status, key=lambda r: (r["days_until_block"] is None, r["days_until_block"])):
            dub = s["days_until_block"]
            runway = "n/a (absent)" if dub is None else f"{dub}d to block"
            print(f"  - {s['evidence']:<22} {s['state']:<9} {runway}")
        print()
        if release_ready:
            print(f"PASS audit SLA check ({len(warnings)} warning(s), {len(pre_alerts)} pre-alert(s))")
        else:
            print(f"FAIL {len(blocking)} blocking SLA violation(s)")

    return 0 if release_ready else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("-o", dest="out_file", help="Write report to file")
    args = parser.parse_args()
    return run(args.json, args.out_file)


if __name__ == "__main__":
    raise SystemExit(main())
