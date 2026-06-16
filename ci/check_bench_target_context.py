#!/usr/bin/env python3
"""Benchmark target-context taxonomy + claim-scope gate (Bastion B17, closes RR-BAS-03).

Every canonical benchmark artifact must declare its `target_context` (from the
enum in docs/BENCH_TARGET_CONTEXT_SCHEMA.json) and a `claim_scope`, so:
  - a microbenchmark cannot be mistaken for Bitcoin Core / libbitcoin node throughput;
  - a GPU public-data / CPU-fallback benchmark cannot be presented as native
    GPU-hardware (or general native-engine) performance;
  - a bitcoin_core / libbitcoin claim must reference integration evidence;
  - unknown_owner_gated context is explicit and never counted as current proof.

This gate is folded into ci/check_bench_doc_consistency.py (so `--json` there
includes it) and co-gated by ci/perf_security_cogate.py. Corrupt-artifact checks
(zero/negative/non-finite/impossible timings) remain in check_bench_doc_consistency
(Bastion B8) and stay blocking.

Exit code:
  0  every benchmark artifact declares a valid context + scope
  1  a missing/invalid context or a scope mismatch
  2  the schema is missing or malformed

Usage:
  python3 ci/check_bench_target_context.py [--json] [-o report.json]
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
LIB_ROOT = SCRIPT_DIR.parent
SCHEMA_PATH = LIB_ROOT / "docs" / "BENCH_TARGET_CONTEXT_SCHEMA.json"

GPU_NONNATIVE_CONTEXTS = {"gpu_public_data"}
INTEGRATION_CONTEXTS = {"bitcoin_core", "libbitcoin"}


def _load_enum() -> list:
    if not SCHEMA_PATH.exists():
        return []
    try:
        return json.loads(SCHEMA_PATH.read_text(encoding="utf-8")).get("target_context_enum", [])
    except Exception:
        return []


def _row_from_artifact(path: str, obj: dict) -> dict:
    """Extract the context block from a benchmark JSON (top-level or under metadata)."""
    meta = obj.get("metadata") if isinstance(obj.get("metadata"), dict) else {}

    def pick(key):
        if key in obj:
            return obj.get(key)
        return meta.get(key)

    results = obj.get("results")
    has_timings = isinstance(results, list) and len(results) > 0
    if not has_timings:
        # Bitcoin Core results are split across results_lto/results_nolto etc.
        has_timings = any(k.startswith("results") and obj.get(k) for k in obj)
    return {
        "id": path,
        "target_context": pick("target_context"),
        "operation": pick("operation"),
        "claim_scope": pick("claim_scope"),
        "evidence_path": pick("evidence_path") or path,
        "security_gate_dependency": pick("security_gate_dependency"),
        "integration_evidence": pick("integration_evidence"),
        "native_hardware_claim": bool(pick("native_hardware_claim")),
        "notes": str(pick("notes") or ""),
        "has_timings": has_timings,
    }


def evaluate(rows: list, enum=None) -> dict:
    """Evaluate a list of context rows (pure / testable)."""
    if enum is None:
        enum = _load_enum()
    enum_set = set(enum)

    out_rows = []
    for r in rows:
        rid = r.get("id", "<no-id>")
        tc = r.get("target_context")
        problems = []
        if not tc:
            problems.append("missing_context")
        elif tc not in enum_set:
            problems.append(f"invalid_context:{tc}")
        else:
            if r.get("has_timings") and not r.get("claim_scope"):
                problems.append("scope_mismatch:claim_scope missing for a timed artifact")
            if not r.get("security_gate_dependency"):
                problems.append("scope_mismatch:security_gate_dependency missing")
            if tc in GPU_NONNATIVE_CONTEXTS and r.get("native_hardware_claim"):
                problems.append("scope_mismatch:gpu_public_data presented as native hardware performance")
            if tc in INTEGRATION_CONTEXTS and not r.get("integration_evidence"):
                problems.append("scope_mismatch:bitcoin_core/libbitcoin claim lacks integration_evidence reference")
            if tc == "unknown_owner_gated" and "owner_gated" not in r.get("notes", "").lower():
                problems.append("scope_mismatch:unknown_owner_gated without an explicit owner_gated note")
        out_rows.append({**r, "problems": problems})

    missing = [r["id"] for r in out_rows if any(p == "missing_context" for p in r["problems"])]
    invalid = [r["id"] for r in out_rows if any(p.startswith("invalid_context") for p in r["problems"])]
    scope = [r["id"] for r in out_rows if any(p.startswith("scope_mismatch") for p in r["problems"])]
    owner_gated = [r["id"] for r in out_rows if r.get("target_context") == "unknown_owner_gated"]

    return {
        "overall_pass": not (missing or invalid or scope),
        "rows_total": len(out_rows),
        "missing_context_rows": missing,
        "invalid_context_rows": invalid,
        "scope_mismatch_rows": scope,
        "owner_gated_rows": owner_gated,
        "rows": out_rows,
    }


def scan_canonical() -> list:
    """Build context rows from the canonical benchmark artifacts."""
    rows = []
    for f in sorted(glob.glob(str(LIB_ROOT / "docs" / "bench_unified_*.json"))):
        try:
            obj = json.loads(Path(f).read_text(encoding="utf-8"))
        except Exception as exc:
            rows.append({"id": str(Path(f).relative_to(LIB_ROOT)), "target_context": None,
                         "has_timings": True, "notes": f"unreadable: {exc}"})
            continue
        rows.append(_row_from_artifact(str(Path(f).relative_to(LIB_ROOT)), obj))
    bc = LIB_ROOT / "docs" / "BITCOIN_CORE_BENCH_RESULTS.json"
    if bc.exists():
        try:
            rows.append(_row_from_artifact("docs/BITCOIN_CORE_BENCH_RESULTS.json",
                                           json.loads(bc.read_text(encoding="utf-8"))))
        except Exception as exc:
            rows.append({"id": "docs/BITCOIN_CORE_BENCH_RESULTS.json", "target_context": None,
                         "has_timings": True, "notes": f"unreadable: {exc}"})
    return rows


def load_and_evaluate() -> tuple[dict, int]:
    if not SCHEMA_PATH.exists():
        return ({"overall_pass": False, "error": f"schema missing: {SCHEMA_PATH}"}, 2)
    enum = _load_enum()
    if not enum:
        return ({"overall_pass": False, "error": "schema has no target_context_enum"}, 2)
    report = evaluate(scan_canonical(), enum=enum)
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
            print(f"FAIL bench target-context: {report['error']}")
            return code
        for r in report["rows"]:
            tag = "OK" if not r["problems"] else "FAIL"
            print(f"  [{tag:4}] {str(r.get('target_context')):20} {r['id']}")
            for p in r["problems"]:
                print(f"          → {p}")
        print()
        if report["owner_gated_rows"]:
            print(f"  owner-gated context (not current proof): {', '.join(report['owner_gated_rows'])}")
        if report["overall_pass"]:
            print(f"PASS bench target-context ({report['rows_total']} artifact(s))")
        else:
            print(f"FAIL bench target-context: missing={report['missing_context_rows']} "
                  f"invalid={report['invalid_context_rows']} scope={report['scope_mismatch_rows']}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
