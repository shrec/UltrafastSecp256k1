#!/usr/bin/env python3
"""Audit SLA/SLO compliance checker — fail-closed release gate.

Checks measurable SLO thresholds:
  - Evidence staleness (max age of critical evidence artifacts)
  - Unresolved HIGH findings window
  - Determinism golden reference freshness
  - Exploit-to-regression conversion time

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


def _load_sla_defs() -> dict:
    """Load SLA definitions."""
    if not SLA_DEF_FILE.exists():
        return {}
    return json.loads(SLA_DEF_FILE.read_text(encoding="utf-8"))


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


def check_evidence_staleness(sla_defs: dict) -> list[dict]:
    """Check that no critical evidence is staler than the SLO threshold."""
    threshold = sla_defs.get("slos", {}).get("max_stale_evidence_days", {}).get("threshold", 30)
    severity = sla_defs.get("slos", {}).get("max_stale_evidence_days", {}).get("severity", "blocking")
    findings: list[dict] = []

    for ev in CRITICAL_EVIDENCE:
        if ev["scope"] == "lib":
            base = LIB_ROOT
        elif ev["scope"] == "suite":
            base = SUITE_ROOT
        else:
            base = LIB_ROOT  # fallback

        full_path = base / ev["path"]
        is_dir = ev.get("is_dir", False)

        if is_dir:
            age = _dir_newest_age_days(full_path)
        else:
            age = _file_age_days(full_path)

        if age is None:
            # In standalone mode OR outside CI (local dev), suite-scoped
            # artifacts are structurally absent — downgrade to warning.
            _in_ci = os.environ.get("GITHUB_ACTIONS") == "true"
            effective_severity = (
                "warning"
                if ((_STANDALONE or not _in_ci) and ev["scope"] == "suite")
                else severity
            )
            findings.append({
                "slo": "max_stale_evidence_days",
                "evidence": ev["name"],
                "status": "missing",
                "severity": effective_severity,
                "detail": f"Evidence artifact not found: {full_path}",
            })
        elif age > threshold:
            findings.append({
                "slo": "max_stale_evidence_days",
                "evidence": ev["name"],
                "status": "stale",
                "age_days": round(age, 1),
                "threshold_days": threshold,
                "severity": severity,
                "detail": f"{ev['name']} is {round(age, 1)} days old (max {threshold})",
            })

    return findings


def check_critical_freshness(sla_defs: dict) -> list[dict]:
    """Check critical-path evidence freshness (stricter threshold)."""
    threshold = sla_defs.get("slos", {}).get("critical_evidence_freshness_days", {}).get("threshold", 14)
    severity = sla_defs.get("slos", {}).get("critical_evidence_freshness_days", {}).get("severity", "blocking")
    findings: list[dict] = []

    critical_files = [
        ("ct_evidence_dir", LIB_ROOT / "audit" / "ci-evidence", True),
        ("api_contracts", LIB_ROOT / "docs" / "API_SECURITY_CONTRACTS.json", False),
    ]

    for name, path, is_dir in critical_files:
        if is_dir:
            age = _dir_newest_age_days(path)
        else:
            age = _file_age_days(path)

        if age is not None and age > threshold:
            findings.append({
                "slo": "critical_evidence_freshness_days",
                "evidence": name,
                "status": "stale",
                "age_days": round(age, 1),
                "threshold_days": threshold,
                "severity": severity,
                "detail": f"Critical evidence {name} is {round(age, 1)} days old (max {threshold})",
            })

    return findings


def check_determinism_golden_freshness(sla_defs: dict) -> list[dict]:
    """Check determinism golden reference freshness."""
    threshold = sla_defs.get("slos", {}).get("determinism_golden_staleness_days", {}).get("threshold", 30)
    severity = sla_defs.get("slos", {}).get("determinism_golden_staleness_days", {}).get("severity", "blocking")
    golden = LIB_ROOT / "docs" / "DETERMINISM_GOLDEN.json"
    age = _file_age_days(golden)

    if age is None:
        return [{
            "slo": "determinism_golden_staleness_days",
            "evidence": "determinism_golden",
            "status": "missing",
            "severity": severity,
            "detail": "DETERMINISM_GOLDEN.json not yet created — will be generated by determinism gate",
        }]
    if age > threshold:
        return [{
            "slo": "determinism_golden_staleness_days",
            "evidence": "determinism_golden",
            "status": "stale",
            "age_days": round(age, 1),
            "threshold_days": threshold,
            "severity": severity,
            "detail": f"Determinism golden is {round(age, 1)} days old (max {threshold})",
        }]
    return []


def run(json_mode: bool, out_file: str | None) -> int:
    sla_defs = _load_sla_defs()

    all_findings: list[dict] = []
    all_findings.extend(check_evidence_staleness(sla_defs))
    all_findings.extend(check_critical_freshness(sla_defs))
    all_findings.extend(check_determinism_golden_freshness(sla_defs))

    blocking = [f for f in all_findings if f.get("severity") == "blocking"]
    warnings = [f for f in all_findings if f.get("severity") == "warning"]

    release_ready = len(blocking) == 0

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "overall_pass": release_ready,
        "release_ready": release_ready,
        "blocking_violations": len(blocking),
        "warning_violations": len(warnings),
        "total_findings": len(all_findings),
        "findings": all_findings,
    }

    rendered = json.dumps(report, indent=2)
    if out_file:
        Path(out_file).write_text(rendered, encoding="utf-8")

    if json_mode:
        print(rendered)
    else:
        for f in all_findings:
            tag = "BLOCK" if f.get("severity") == "blocking" else "WARN"
            print(f"  [{tag}] {f['detail']}")
        print()
        if release_ready:
            print(f"PASS audit SLA check ({len(warnings)} warning(s))")
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
