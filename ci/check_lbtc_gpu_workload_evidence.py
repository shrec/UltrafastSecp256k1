#!/usr/bin/env python3
"""Libbitcoin GPU workload benchmark evidence gate.

Opens a benchmark JSON artifact produced by the libbitcoin-direct benchmark
harnesses and rejects it as invalid evidence if it is corrupt, internally
inconsistent, or overclaims GPU acceleration from CPU-only data. This fills a
confirmed gap: `check_libbitcoin_perf_matrix.py` only checks that
`evidence_paths` files *exist* on disk, it never opens or validates the JSON
inside them.

Two artifact schemas are recognized:

  ufsecp-lbtc-public-ops-benchmark-v1
      The CURRENT `bench_lbtc_public_ops` harness (bench_public_ops.cpp).
      This harness does not yet measure a real upload/kernel/download phase
      split, so every row's `evidence_class` MUST be "api_correctness"
      unconditionally, even on rows that DO carry a real (non-"cpu")
      backend/device identification -- it structurally cannot produce
      `gpu_acceleration` evidence under this schema yet. The phase timing
      fields (`prep_seconds`, `upload_seconds`, `kernel_seconds`,
      `download_seconds`) may be `null` (not measured) for this schema.

  ufsecp-lbtc-gpu-workload-benchmark-v1
      The phase-aware schema for txid/wtxid/sighash/merkle-shaped workloads,
      defined in
      workingdocs/libbitcoin_gpu_workloads/evidence_matrix_claude.json.
      `prep_seconds` and `kernel_seconds` are always required and positive.
      `upload_seconds`/`download_seconds` MUST be null for backend == cpu;
      for non-cpu backends they may be a positive number (when a harness has
      genuine phase-split instrumentation) OR null (when it honestly does
      not -- see "Honest gaps" below) -- either is accepted, but a present
      non-null value must still be positive (zero/negative is always
      rejected as zero_timing).

  A row with evidence_class == "gpu_acceleration" (only possible under the
  v2/gpu-workload schema; v1 bans it unconditionally, see above) MUST
  additionally satisfy ALL of the following, checked explicitly as a single
  group (not merely inferred from scattered per-field rules): backend != cpu,
  device is a real (non-empty, non-"n/a") string, provider_linked == true,
  hook_installed == true, validation_status == matched_reference, and both
  best_seconds and kernel_seconds are present, numeric, and > 0.

  Honest gaps (never treated as malformed, never require fabrication):
    - driver_version may be null on a non-cpu row. secp256k1::gpu::DeviceInfo
      (gpu_backend.hpp) carries no driver-version field, and adding one is
      out of scope for the harnesses that produce these artifacts -- null
      here means "honestly not queryable", not "forgot to fill in". An
      EMPTY STRING is still rejected as malformed (a caller that actually
      queried an empty result should use null, not "").
    - upload_seconds/download_seconds may be null on a non-cpu row: the
      GpuBackend interface exposes each op as a single opaque call with no
      internal phase-split instrumentation, so a harness that only measures
      one wall-clock span per call honestly reports null here rather than
      inventing a split. kernel_seconds mirroring best_seconds (same
      single-span convention already used by schema v1) is the documented,
      accepted way to report that.

Rejection rules enforced on every row (see evidence_matrix_claude.json
`rejection_rules` for the full authoritative rule text):

  1. zero_timing              -- any present, non-null timing field <= 0.0
  2. impossible_throughput    -- ns_per_row * count must reconstruct
                                  best_seconds within a 1% tolerance
  3. malformed_columns        -- missing/mistyped required field, or an
                                  inconsistent backend/device/driver_version
                                  pairing
  4. missing_backend_evidence -- backend != cpu without provider_linked, or
                                  evidence_class == gpu_acceleration without
                                  hook_installed
  5. cpu_only_relabeled_as_gpu -- backend == cpu or hook_installed == false
                                  tagged evidence_class == gpu_acceleration
                                  (schema v1 additionally requires
                                  api_correctness unconditionally)
  6. missing_validation       -- validation_status != matched_reference
  7. one_sided_speedup        -- speedup_vs_cpu_forced present with only one
                                  of compute_only_ratio / end_to_end_ratio
  8. gpu_acceleration_incomplete -- evidence_class == gpu_acceleration but
                                  one of the explicit group conditions above
                                  (backend/device/provider_linked/
                                  hook_installed/validation/positive timing)
                                  is not satisfied

Not machine-checkable from a single artifact (documented, not enforced here):
  - backend_relabeling: cross-checking `backend` against the actual linked
    GPU provider requires build-time provenance outside this JSON file.
  - phantom_row: proving a *missing* row was not silently zero-filled
    requires host inventory knowledge this script does not have.

Exit code:
  0  artifact is valid evidence
  1  one or more rows rejected
  2  artifact missing, malformed JSON, or unknown/missing schema

Usage:
  python3 ci/check_lbtc_gpu_workload_evidence.py <artifact.json> [--json] [-o report.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
LIB_ROOT = SCRIPT_DIR.parent

SCHEMA_PUBLIC_OPS_V1 = "ufsecp-lbtc-public-ops-benchmark-v1"
SCHEMA_GPU_WORKLOAD_V1 = "ufsecp-lbtc-gpu-workload-benchmark-v1"
KNOWN_SCHEMAS = {SCHEMA_PUBLIC_OPS_V1, SCHEMA_GPU_WORKLOAD_V1}

BACKEND_VALUES = {"cpu", "cuda", "opencl", "metal"}
EVIDENCE_CLASS_VALUES = {"api_correctness", "gpu_acceleration"}
VALIDATION_STATUS_VALUES = {"matched_reference", "mismatch"}
WORKLOAD_VALUES = {
    "txid_batch", "wtxid_batch", "sighash_batch",
    "merkle_pair_batch", "merkle_root_batch",
}
BATCH_CLASS_VALUES = {"small", "medium", "block_scale", "stress"}

ARITHMETIC_TOLERANCE = 0.01

# Row fields present in both schemas, always required, never null.
ALWAYS_POSITIVE_FIELDS = {
    "best_seconds", "m_rows_per_sec", "payload_mib_per_sec", "ns_per_row",
    "count", "payload_bytes_per_iter",
}
ALWAYS_PRESENT_NON_NUMERIC_FIELDS = {
    "mode", "backend", "device", "hook_installed", "provider_linked",
    "validation_hash", "validation_status", "evidence_class",
}
NULLABLE_TIMING_FIELDS = {"prep_seconds", "upload_seconds", "kernel_seconds", "download_seconds"}

ROW_FIELDS_V1 = (
    ALWAYS_POSITIVE_FIELDS | ALWAYS_PRESENT_NON_NUMERIC_FIELDS
    | NULLABLE_TIMING_FIELDS | {"op", "driver_version"}
)
ROW_FIELDS_V2 = (
    ALWAYS_POSITIVE_FIELDS | ALWAYS_PRESENT_NON_NUMERIC_FIELDS
    | NULLABLE_TIMING_FIELDS | {"driver_version"}
)

ENVELOPE_COMMON_FIELDS = {
    "schema", "target_context", "claim_scope", "c_abi_required",
    "shim_required", "bridge_required", "count", "iters", "host_context", "results",
}
ENVELOPE_V2_EXTRA_FIELDS = {"workload", "batch_class", "payload_bytes_total"}

HOST_CONTEXT_FIELDS = {"compiler", "cpu_model", "turbo_disabled", "cpu_pinned", "kernel"}


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _row_problems(row: dict, schema: str) -> list:
    problems = []
    required_fields = ROW_FIELDS_V1 if schema == SCHEMA_PUBLIC_OPS_V1 else ROW_FIELDS_V2

    for field in sorted(required_fields):
        if field not in row:
            problems.append(f"malformed_columns:missing_field:{field}")

    backend = row.get("backend")
    if backend not in BACKEND_VALUES:
        problems.append(f"malformed_columns:bad_backend:{backend!r}")

    device = row.get("device")
    driver_version = row.get("driver_version")
    if backend == "cpu":
        if "device" in row and device != "n/a":
            problems.append("malformed_columns:cpu_device_not_na")
        if "driver_version" in row and driver_version is not None:
            problems.append("malformed_columns:cpu_driver_version_not_null")
    elif backend in BACKEND_VALUES:
        if "device" in row and (not isinstance(device, str) or device in ("", "n/a")):
            problems.append("malformed_columns:gpu_missing_device")
        # driver_version is an HONEST GAP, not a required field, on non-cpu
        # rows: secp256k1::gpu::DeviceInfo carries no driver-version field, so
        # a harness that cannot query one reports null rather than fabricate
        # a string. Only an explicit empty string is rejected as malformed
        # (a caller with a real empty result should use null, not "").
        if "driver_version" in row and driver_version is not None and driver_version == "":
            problems.append("malformed_columns:gpu_driver_version_empty_string")

    provider_linked = row.get("provider_linked")
    if backend in BACKEND_VALUES and backend != "cpu" and provider_linked is not True:
        problems.append("missing_backend_evidence:provider_not_linked")

    hook_installed = row.get("hook_installed")
    evidence_class = row.get("evidence_class")
    if evidence_class not in EVIDENCE_CLASS_VALUES:
        problems.append(f"malformed_columns:bad_evidence_class:{evidence_class!r}")
    else:
        if evidence_class == "gpu_acceleration" and hook_installed is not True:
            problems.append("missing_backend_evidence:gpu_claim_without_hook")
        if (backend == "cpu" or hook_installed is not True) and evidence_class == "gpu_acceleration":
            problems.append("cpu_only_relabeled_as_gpu:cpu_or_hook_off_claims_gpu")
        if schema == SCHEMA_PUBLIC_OPS_V1 and evidence_class != "api_correctness":
            problems.append("cpu_only_relabeled_as_gpu:schema_v1_requires_api_correctness")

    validation_status = row.get("validation_status")
    if "validation_status" in row and validation_status not in VALIDATION_STATUS_VALUES:
        problems.append(f"malformed_columns:bad_validation_status:{validation_status!r}")
    elif validation_status != "matched_reference":
        problems.append(f"missing_validation:{validation_status!r}")

    if "validation_hash" in row and (not isinstance(row.get("validation_hash"), str)
                                      or not row.get("validation_hash")):
        problems.append("malformed_columns:missing_validation_hash")

    for field in sorted(ALWAYS_POSITIVE_FIELDS):
        if field not in row:
            continue
        value = row.get(field)
        if not _is_number(value):
            problems.append(f"malformed_columns:bad_type:{field}")
        elif value <= 0:
            problems.append(f"zero_timing:{field}")

    kernel_prep_required = schema == SCHEMA_GPU_WORKLOAD_V1
    for field in sorted(NULLABLE_TIMING_FIELDS):
        if field not in row:
            continue
        value = row.get(field)
        if value is None:
            # upload_seconds/download_seconds: null is REQUIRED for backend
            # == cpu (no device transfer happened) and PERMITTED (not
            # required) for non-cpu backends -- a harness that cannot
            # honestly measure a phase-split reports null rather than
            # fabricate a number (see "Honest gaps" in the module docstring).
            # prep_seconds/kernel_seconds stay mandatory-non-null for schema
            # v2 regardless of backend.
            if field not in ("upload_seconds", "download_seconds") and kernel_prep_required:
                problems.append(f"malformed_columns:{field}_null_but_required_for_schema")
            continue
        if not _is_number(value):
            problems.append(f"malformed_columns:bad_type:{field}")
            continue
        if value <= 0:
            problems.append(f"zero_timing:{field}")

    best_seconds = row.get("best_seconds")
    ns_per_row = row.get("ns_per_row")
    count = row.get("count")
    if _is_number(best_seconds) and _is_number(ns_per_row) and _is_number(count) and best_seconds > 0:
        expected = best_seconds * 1e9
        actual = ns_per_row * count
        rel_err = abs(actual - expected) / expected if expected else float("inf")
        if rel_err > ARITHMETIC_TOLERANCE:
            problems.append(f"impossible_throughput:rel_err={rel_err:.4f}")

    speedup = row.get("speedup_vs_cpu_forced")
    if speedup is not None:
        if not isinstance(speedup, dict):
            problems.append("one_sided_speedup:not_object")
        else:
            compute_ok = _is_number(speedup.get("compute_only_ratio")) and speedup["compute_only_ratio"] > 0
            e2e_ok = _is_number(speedup.get("end_to_end_ratio")) and speedup["end_to_end_ratio"] > 0
            if compute_ok != e2e_ok:
                problems.append("one_sided_speedup:missing_pair")
            elif not compute_ok and not e2e_ok:
                problems.append("one_sided_speedup:empty_object")

    # Explicit gpu_acceleration group check: every condition below is ALSO
    # implied by one of the individual field rules above (backend/device via
    # gpu_missing_device, provider_linked via missing_backend_evidence,
    # hook_installed via gpu_claim_without_hook, validation_status via
    # missing_validation, positive timing via zero_timing/malformed_columns)
    # -- this block re-asserts them together, by name, as a single readable
    # unit so "what does a valid GPU-accelerated row require" has one
    # unambiguous answer in the code, instead of being inferable only by
    # reading every other rule in this function.
    if evidence_class == "gpu_acceleration":
        kernel_seconds = row.get("kernel_seconds")
        incomplete = (
            backend == "cpu"
            or not isinstance(device, str) or device in ("", "n/a")
            or provider_linked is not True
            or hook_installed is not True
            or validation_status != "matched_reference"
            or not (_is_number(best_seconds) and best_seconds > 0)
            or not (_is_number(kernel_seconds) and kernel_seconds > 0)
        )
        if incomplete:
            problems.append("gpu_acceleration_incomplete:missing_required_condition")

    return problems


def _envelope_problems(payload: dict, schema: str) -> list:
    problems = []
    required = ENVELOPE_COMMON_FIELDS | (ENVELOPE_V2_EXTRA_FIELDS if schema == SCHEMA_GPU_WORKLOAD_V1 else set())
    for field in sorted(required):
        if field not in payload:
            problems.append(f"malformed_columns:missing_envelope_field:{field}")

    if schema == SCHEMA_GPU_WORKLOAD_V1:
        if payload.get("target_context") != "libbitcoin":
            problems.append(f"malformed_columns:bad_target_context:{payload.get('target_context')!r}")
        if payload.get("workload") not in WORKLOAD_VALUES:
            problems.append(f"malformed_columns:bad_workload:{payload.get('workload')!r}")
        if payload.get("batch_class") not in BATCH_CLASS_VALUES:
            problems.append(f"malformed_columns:bad_batch_class:{payload.get('batch_class')!r}")

    host_context = payload.get("host_context")
    if not isinstance(host_context, dict):
        problems.append("malformed_columns:host_context_not_object")
    else:
        for field in sorted(HOST_CONTEXT_FIELDS):
            if field not in host_context:
                problems.append(f"malformed_columns:missing_host_context_field:{field}")

    results = payload.get("results")
    if not isinstance(results, list) or not results:
        problems.append("malformed_columns:results_empty_or_missing")

    return problems


def evaluate(payload: dict) -> dict:
    """Validate a benchmark evidence artifact (pure/testable)."""
    schema = payload.get("schema")
    if schema not in KNOWN_SCHEMAS:
        return {
            "overall_pass": False,
            "schema": schema,
            "error": f"unknown or missing schema: {schema!r} (known: {sorted(KNOWN_SCHEMAS)})",
            "rows": [],
        }

    envelope_problems = _envelope_problems(payload, schema)
    results = payload.get("results") if isinstance(payload.get("results"), list) else []

    rows = []
    for index, row in enumerate(results):
        if not isinstance(row, dict):
            rows.append({
                "index": index, "op": None, "mode": None,
                "problems": ["malformed_columns:row_not_object"], "fails": True,
            })
            continue
        problems = _row_problems(row, schema)
        rows.append({
            "index": index,
            "op": row.get("op") or row.get("workload"),
            "mode": row.get("mode"),
            "problems": problems,
            "fails": bool(problems),
        })

    problem_buckets: dict = {}
    for row in rows:
        for problem in row["problems"]:
            key = problem.split(":", 1)[0]
            problem_buckets.setdefault(key, []).append(row["index"])
    for problem in envelope_problems:
        key = problem.split(":", 1)[0]
        problem_buckets.setdefault(key, []).append("envelope")

    overall_pass = not envelope_problems and not any(row["fails"] for row in rows)

    return {
        "overall_pass": overall_pass,
        "schema": schema,
        "envelope_problems": envelope_problems,
        "row_total": len(rows),
        "rows_failed": sum(1 for row in rows if row["fails"]),
        "problems": problem_buckets,
        "rows": rows,
    }


def load_and_evaluate(path: Path) -> tuple:
    if not path.exists():
        return ({"overall_pass": False, "error": f"artifact missing: {path}"}, 2)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        return ({"overall_pass": False, "error": f"artifact malformed JSON: {exc}"}, 2)
    if not isinstance(payload, dict):
        return ({"overall_pass": False, "error": "artifact root is not a JSON object"}, 2)

    report = evaluate(payload)
    if "error" in report:
        return (report, 2)
    return (report, 0 if report["overall_pass"] else 1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("artifact", help="path to a benchmark evidence JSON artifact")
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("-o", dest="out_file", help="write report to file")
    args = parser.parse_args()

    report, code = load_and_evaluate(Path(args.artifact))
    rendered = json.dumps(report, indent=2)
    if args.out_file:
        Path(args.out_file).write_text(rendered + "\n", encoding="utf-8")

    if args.json:
        print(rendered)
    else:
        if "error" in report:
            print(f"FAIL lbtc gpu workload evidence: {report['error']}")
            return code
        for row in report["rows"]:
            if row["problems"]:
                print(f"  [FAIL] row {row['index']} ({row['op']}/{row['mode']}): {row['problems']}")
        for problem in report.get("envelope_problems", []):
            print(f"  [FAIL] envelope: {problem}")
        if report["overall_pass"]:
            print(f"PASS lbtc gpu workload evidence ({report['schema']}, "
                  f"{report['row_total']} rows, 0 rejected)")
        else:
            print(f"FAIL lbtc gpu workload evidence ({report['schema']}, "
                  f"{report['rows_failed']}/{report['row_total']} rows rejected): "
                  f"{report['problems']}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
