#!/usr/bin/env python3
"""Package / release provenance binding gate (Bastion B20).

A package or release artifact is only trustworthy when it is bound to the exact
audited commit, the committed CAAS evidence bundle, the audit_gate verdict, and
its own artifact hash. This gate validates docs/PACKAGE_PROVENANCE_STATUS.json:

  - every surface declares the full binding CONTRACT (artifact, producer_workflow,
    source_commit, source_branch, artifact_sha256, caas_bundle_sha256,
    audit_gate_verdict, workflow_run_id, status, severity) — a missing field fails;
  - `producer_workflow` must reference a real .github/workflows/*.yml;
  - `template` surfaces (dev contract, nothing built) must hold the recognized
    sentinels and null artifact_sha256 / workflow_run_id — no fake current values;
  - `bound` surfaces (a real built artifact) must MATCH: source_commit == HEAD,
    caas_bundle_sha256 == the committed bundle digest, artifact_sha256 present,
    audit_gate_verdict == 'pass', workflow_run_id present;
  - `owner_gated` release surfaces must NEVER be current in the dev tree — a real
    artifact_sha256 or workflow_run_id marks a release artifact current without an
    owner release and fails.

THIS IS PROVENANCE BINDING, NOT RELEASE AUTHORIZATION. The gate never publishes,
tags, or authorizes a release; it only refuses to call an artifact 'audited'
unless the binding holds. See docs/PACKAGE_PROVENANCE.md.

Exit code:
  0  every surface's binding contract holds (and every bound surface matches)
  1  a missing/contradictory binding, a mismatch, or a release artifact marked current
  2  the manifest is missing or malformed

Usage:
  python3 ci/check_package_provenance_binding.py [--json] [-o report.json]
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
LIB_ROOT = SCRIPT_DIR.parent
WORKFLOWS = LIB_ROOT / ".github" / "workflows"

MANIFEST = LIB_ROOT / "docs" / "PACKAGE_PROVENANCE_STATUS.json"
BUNDLE_DIGEST = LIB_ROOT / "docs" / "EXTERNAL_AUDIT_BUNDLE.sha256"

REQUIRED_FIELDS = ("artifact", "producer_workflow", "source_commit", "source_branch",
                   "artifact_sha256", "caas_bundle_sha256", "audit_gate_verdict",
                   "workflow_run_id", "status", "severity")

STATUS_VALUES = {"template", "bound", "owner_gated"}
SEVERITY_VALUES = {"blocking", "warning", "owner_gated"}

HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")

# Problems that always fail (binding integrity), regardless of severity.
STRUCTURAL = {"missing_binding", "bad_status", "bad_severity", "unknown_workflow",
              "malformed_template", "malformed_owner_gated", "release_marked_current"}
# Problems that fail only for blocking surfaces (a bound artifact mismatch); a
# warning surface downgrades these to advisory.
MATCH = {"commit_mismatch", "bundle_mismatch", "missing_artifact_hash",
         "bad_verdict", "missing_run_id"}


def evaluate(surfaces: list, head_commit: str, committed_bundle_sha: str,
             existing_workflows: set) -> dict:
    """Validate package provenance binding (pure / testable)."""
    rows = []
    for s in surfaces:
        name = s.get("artifact", "<no-artifact>")
        severity = s.get("severity", "blocking")
        status = s.get("status")
        problems = []

        for field in REQUIRED_FIELDS:
            if field not in s:
                problems.append(f"missing_binding:{field}")
        if status not in STATUS_VALUES:
            problems.append(f"bad_status:{status!r}")
        if severity not in SEVERITY_VALUES:
            problems.append(f"bad_severity:{severity!r}")
        if s.get("producer_workflow") not in existing_workflows:
            problems.append(f"unknown_workflow:{s.get('producer_workflow')}")

        sha = s.get("artifact_sha256")
        run_id = s.get("workflow_run_id")
        commit = s.get("source_commit")
        bundle = s.get("caas_bundle_sha256")
        verdict = s.get("audit_gate_verdict")

        if status == "template":
            if commit != "@HEAD":
                problems.append("malformed_template:source_commit-not-@HEAD")
            if bundle != "@committed-bundle":
                problems.append("malformed_template:caas_bundle-not-@committed-bundle")
            if verdict != "@audit-gate":
                problems.append("malformed_template:audit_gate_verdict-not-@audit-gate")
            if sha is not None:
                problems.append("malformed_template:artifact_sha256-set-should-be-bound")
            if run_id is not None:
                problems.append("malformed_template:workflow_run_id-set-should-be-bound")

        elif status == "owner_gated":
            if severity != "owner_gated":
                problems.append("malformed_owner_gated:severity-not-owner_gated")
            # A release artifact is NEVER current in the dev tree.
            if (isinstance(sha, str) and HEX64.match(sha)) or run_id:
                problems.append("release_marked_current:has-real-hash-or-run-id")
            if commit != "@release":
                problems.append("malformed_owner_gated:source_commit-not-@release")
            if bundle not in ("@release-bundle", "@committed-bundle"):
                problems.append("malformed_owner_gated:caas_bundle-not-@release-bundle")

        elif status == "bound":
            if not (isinstance(commit, str) and HEX40.match(commit)) or commit != head_commit:
                problems.append("commit_mismatch:bound-source_commit-!=-HEAD")
            if not (isinstance(sha, str) and HEX64.match(sha)):
                problems.append("missing_artifact_hash:bound-artifact_sha256-not-64hex")
            if bundle != committed_bundle_sha:
                problems.append("bundle_mismatch:bound-caas_bundle-!=-committed")
            if verdict != "pass":
                problems.append("bad_verdict:bound-audit_gate_verdict-!=-pass")
            if not run_id:
                problems.append("missing_run_id:bound-workflow_run_id-empty")

        # A surface fails if it has any STRUCTURAL problem, or a MATCH problem
        # while it is a blocking surface.
        fails = any(p.split(":")[0] in STRUCTURAL for p in problems) or (
            severity == "blocking" and any(p.split(":")[0] in MATCH for p in problems))

        rows.append({"artifact": name, "status": status, "severity": severity,
                     "problems": problems, "fails": fails})

    def collect(prefix):
        return [r["artifact"] for r in rows if any(p.split(":")[0] == prefix for p in r["problems"])]

    buckets = {p: collect(p) for p in sorted(STRUCTURAL | MATCH)}
    overall = not any(r["fails"] for r in rows)

    return {
        "overall_pass": overall,
        "surfaces_total": len(rows),
        "template": sum(1 for r in rows if r["status"] == "template"),
        "bound": sum(1 for r in rows if r["status"] == "bound"),
        "owner_gated": sum(1 for r in rows if r["status"] == "owner_gated"),
        "head_commit": head_commit,
        "committed_bundle_sha256": committed_bundle_sha,
        "problems": {k: v for k, v in buckets.items() if v},
        "rows": rows,
    }


def _git_head() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(LIB_ROOT),
                           capture_output=True, text=True, timeout=15, check=False)
        return r.stdout.strip() if r.returncode == 0 else ""
    except Exception:
        return ""


def _committed_bundle_sha() -> str:
    if not BUNDLE_DIGEST.exists():
        return ""
    tokens = BUNDLE_DIGEST.read_text(encoding="utf-8", errors="replace").strip().split()
    return tokens[0] if tokens else ""


def _existing_workflows() -> set:
    if not WORKFLOWS.is_dir():
        return set()
    return {f".github/workflows/{p.name}" for p in WORKFLOWS.glob("*.yml")}


def load_and_evaluate() -> tuple[dict, int]:
    if not MANIFEST.exists():
        return ({"overall_pass": False, "error": f"manifest missing: {MANIFEST}"}, 2)
    try:
        data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        return ({"overall_pass": False, "error": f"manifest malformed JSON: {exc}"}, 2)
    surfaces = data.get("surfaces")
    if not isinstance(surfaces, list) or not surfaces:
        return ({"overall_pass": False, "error": "manifest has no surfaces[]"}, 2)
    report = evaluate(surfaces, _git_head(), _committed_bundle_sha(), _existing_workflows())
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
            print(f"FAIL package provenance binding: {report['error']}")
            return code
        for r in report["rows"]:
            if r["problems"]:
                tag = "FAIL" if r["fails"] else "WARN"
                print(f"  [{tag}] {r['artifact'][:48]} ({r['status']}): {r['problems']}")
        print()
        if report["overall_pass"]:
            print(f"PASS package provenance binding ({report['surfaces_total']} surfaces: "
                  f"{report['template']} template, {report['bound']} bound, "
                  f"{report['owner_gated']} owner_gated — all bound to commit+bundle+verdict+hash or sentineled)")
        else:
            print(f"FAIL package provenance binding: {report['problems']}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
