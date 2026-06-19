#!/usr/bin/env python3
"""Libbitcoin performance/integration matrix gate (Bastion B21).

The libbitcoin bridge has several performance-sensitive surfaces: C++ default
controller use, row/column table formats, CUDA row staging, MSVC builds, and the
benchmark artifact itself. This gate keeps those claims honest:

  - every required surface is present;
  - each row declares target_context == "libbitcoin";
  - evidence paths and reproduce commands exist;
  - native-hardware performance is not claimed from docs-only or owner-gated rows;
  - the benchmark JSON contract is present before numbers are treated as evidence.

Exit code:
  0  manifest is complete and non-overclaiming
  1  missing/contradictory surface evidence
  2  manifest missing or malformed

Usage:
  python3 ci/check_libbitcoin_perf_matrix.py [--json] [-o report.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
LIB_ROOT = SCRIPT_DIR.parent
MANIFEST = LIB_ROOT / "docs" / "LIBBITCOIN_PERF_MATRIX_STATUS.json"

REQUIRED_IDS = {
    "lbtc_cpp_default_controller",
    "lbtc_benchmark_json_artifact",
    "lbtc_cuda_row_persistent_scratch",
    "lbtc_msvc_windows_profile",
    "lbtc_vertical_opaque_contract",
    "lbtc_caas_perf_matrix_gate",
}

REQUIRED_FIELDS = {
    "id",
    "surface",
    "target_context",
    "status",
    "severity",
    "claim_scope",
    "evidence_paths",
    "reproduce_command",
    "native_hardware_claim",
    "copy_policy",
}

STATUS_VALUES = {
    "implemented",
    "measured_external",
    "documented_current",
    "owner_gated",
}

SEVERITY_VALUES = {"blocking", "warning", "owner_gated"}


def evaluate(surfaces: list) -> dict:
    """Validate the libbitcoin perf matrix manifest (pure/testable)."""
    rows = []
    seen = set()

    for row in surfaces:
        sid = str(row.get("id", "<missing-id>"))
        seen.add(sid)
        severity = row.get("severity", "blocking")
        status = row.get("status")
        problems = []

        for field in sorted(REQUIRED_FIELDS):
            if field not in row:
                problems.append(f"missing_field:{field}")

        if row.get("target_context") != "libbitcoin":
            problems.append(f"bad_context:{row.get('target_context')!r}")
        if status not in STATUS_VALUES:
            problems.append(f"bad_status:{status!r}")
        if severity not in SEVERITY_VALUES:
            problems.append(f"bad_severity:{severity!r}")
        if status == "owner_gated" and severity != "owner_gated":
            problems.append("owner_gated_severity:not-owner_gated")

        evidence_paths = row.get("evidence_paths")
        if not isinstance(evidence_paths, list) or not evidence_paths:
            problems.append("missing_evidence_paths")
        elif not all(isinstance(p, str) and p.strip() for p in evidence_paths):
            problems.append("malformed_evidence_path")
        else:
            missing = [p for p in evidence_paths if not (LIB_ROOT / p).exists()]
            for path in missing:
                problems.append(f"missing_evidence:{path}")

        if not isinstance(row.get("reproduce_command"), str) or not row.get("reproduce_command", "").strip():
            problems.append("missing_command")
        if not isinstance(row.get("claim_scope"), str) or not row.get("claim_scope", "").strip():
            problems.append("missing_claim_scope")
        if not isinstance(row.get("copy_policy"), str) or not row.get("copy_policy", "").strip():
            problems.append("missing_copy_policy")

        native_claim = row.get("native_hardware_claim")
        if not isinstance(native_claim, bool):
            problems.append("bad_native_claim_type")
        if native_claim and status not in {"implemented", "measured_external"}:
            problems.append("native_overclaim:status-not-implemented-or-measured")
        if native_claim and not row.get("benchmark_artifact_contract"):
            problems.append("native_overclaim:no-benchmark-artifact-contract")

        if sid == "lbtc_benchmark_json_artifact" and not row.get("benchmark_artifact_contract"):
            problems.append("missing_benchmark_artifact_contract")

        fails = severity == "blocking" and bool(problems)
        rows.append({
            "id": sid,
            "status": status,
            "severity": severity,
            "problems": problems,
            "fails": fails,
        })

    missing_required = sorted(REQUIRED_IDS - seen)
    for sid in missing_required:
        rows.append({
            "id": sid,
            "status": "missing",
            "severity": "blocking",
            "problems": ["missing_required_surface"],
            "fails": True,
        })

    problem_buckets = {}
    for row in rows:
        for problem in row["problems"]:
            key = problem.split(":", 1)[0]
            problem_buckets.setdefault(key, []).append(row["id"])

    return {
        "overall_pass": not any(row["fails"] for row in rows),
        "surface_total": len(rows),
        "required_total": len(REQUIRED_IDS),
        "implemented": sum(1 for row in rows if row["status"] == "implemented"),
        "measured_external": sum(1 for row in rows if row["status"] == "measured_external"),
        "documented_current": sum(1 for row in rows if row["status"] == "documented_current"),
        "owner_gated": sum(1 for row in rows if row["status"] == "owner_gated"),
        "problems": problem_buckets,
        "rows": rows,
    }


def load_and_evaluate() -> tuple[dict, int]:
    if not MANIFEST.exists():
        return ({"overall_pass": False, "error": f"manifest missing: {MANIFEST}"}, 2)
    try:
        payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        return ({"overall_pass": False, "error": f"manifest malformed JSON: {exc}"}, 2)
    surfaces = payload.get("surfaces")
    if not isinstance(surfaces, list) or not surfaces:
        return ({"overall_pass": False, "error": "manifest has no surfaces[]"}, 2)
    report = evaluate(surfaces)
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
        Path(args.out_file).write_text(rendered + "\n", encoding="utf-8")

    if args.json:
        print(rendered)
    else:
        if "error" in report:
            print(f"FAIL libbitcoin perf matrix: {report['error']}")
            return code
        for row in report["rows"]:
            if row["problems"]:
                tag = "FAIL" if row["fails"] else "WARN"
                print(f"  [{tag}] {row['id']}: {row['problems']}")
        if report["overall_pass"]:
            print(f"PASS libbitcoin perf matrix ({report['surface_total']} surfaces: "
                  f"{report['implemented']} implemented, "
                  f"{report['measured_external']} measured_external, "
                  f"{report['documented_current']} documented_current, "
                  f"{report['owner_gated']} owner_gated)")
        else:
            print(f"FAIL libbitcoin perf matrix: {report['problems']}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
