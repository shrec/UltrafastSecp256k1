#!/usr/bin/env python3
"""
test_lbtc_gpu_workload_evidence.py — unit test for ci/check_lbtc_gpu_workload_evidence.py.

Guards the evidence gate's fail-closed property: a valid artifact must pass, and
each documented corruption class (zero timing, CPU-as-GPU relabeling, malformed/
missing fields, arithmetic-inconsistent throughput, missing validation, one-sided
speedup, GPU evidence without provider/hook) must independently flip the gate to
FAIL. Covers both recognized schemas (current bench_lbtc_public_ops harness and
the future phase-aware libbitcoin GPU workload schema).

Self-contained. Exit 0 = pass, 1 = fail.
"""
import copy
import importlib.util
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GATE = os.path.join(ROOT, "ci", "check_lbtc_gpu_workload_evidence.py")

failures = []


def check(cond, msg):
    print(("  ok  : " if cond else "  FAIL: ") + msg)
    if not cond:
        failures.append(msg)


def load_gate():
    spec = importlib.util.spec_from_file_location("lbtc_evidence_gate", GATE)
    chk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(chk)
    return chk


def valid_host_context():
    return {
        "compiler": "gcc 14.2.0",
        "cpu_model": "Intel Core i5-14400F",
        "turbo_disabled": True,
        "cpu_pinned": True,
        "kernel": "Linux 6.8.0-134-generic",
    }


def valid_v1_row():
    return {
        "op": "hash256",
        "mode": "direct-cpu-forced",
        "hook_installed": False,
        "provider_linked": False,
        "backend": "cpu",
        "device": "n/a",
        "driver_version": None,
        "payload_bytes_per_iter": 64,
        "count": 1000,
        "prep_seconds": None,
        "upload_seconds": None,
        "kernel_seconds": 0.5,
        "download_seconds": None,
        "best_seconds": 0.5,
        "m_rows_per_sec": 1000 / 0.5 / 1e6,
        "payload_mib_per_sec": 1.0,
        "ns_per_row": 0.5 * 1e9 / 1000,
        "validation_hash": "ab" * 32,
        "validation_status": "matched_reference",
        "evidence_class": "api_correctness",
    }


def valid_v1_artifact():
    return {
        "schema": "ufsecp-lbtc-public-ops-benchmark-v1",
        "target_context": "libbitcoin-direct-cpp",
        "claim_scope": "test fixture",
        "c_abi_required": False,
        "shim_required": False,
        "bridge_required": False,
        "count": 1000,
        "iters": 3,
        "host_context": valid_host_context(),
        "results": [valid_v1_row()],
    }


def valid_v2_row(backend="cuda", driver_version="570.00", phase_split=True):
    return {
        "mode": "direct-cuda-forced",
        "backend": backend,
        "device": "n/a" if backend == "cpu" else "RTX 5060 Ti",
        "driver_version": None if backend == "cpu" else driver_version,
        "provider_linked": backend == "cpu" or True,
        "hook_installed": backend != "cpu",
        "count": 2048,
        "payload_bytes_per_iter": 128,
        "prep_seconds": 0.001,
        "upload_seconds": None if (backend == "cpu" or not phase_split) else 0.0005,
        "kernel_seconds": 0.01,
        "download_seconds": None if (backend == "cpu" or not phase_split) else 0.0003,
        "best_seconds": 0.02,
        "m_rows_per_sec": 2048 / 0.02 / 1e6,
        "payload_mib_per_sec": 5.0,
        "ns_per_row": 0.02 * 1e9 / 2048,
        "validation_hash": "cd" * 32,
        "validation_status": "matched_reference",
        "evidence_class": "gpu_acceleration" if backend != "cpu" else "api_correctness",
    }


def valid_v2_artifact():
    return {
        "schema": "ufsecp-lbtc-gpu-workload-benchmark-v1",
        "target_context": "libbitcoin",
        "workload": "merkle_pair_batch",
        "batch_class": "medium",
        "claim_scope": "test fixture",
        "c_abi_required": False,
        "shim_required": False,
        "bridge_required": False,
        "count": 2048,
        "iters": 5,
        "payload_bytes_total": 262144,
        "host_context": valid_host_context(),
        "results": [valid_v2_row()],
    }


def main() -> int:
    chk = load_gate()

    # 1. Valid v1 (current harness) artifact passes.
    rep = chk.evaluate(valid_v1_artifact())
    check(rep["overall_pass"] is True, "valid v1 (public-ops) artifact passes")
    check(rep["rows_failed"] == 0, "valid v1 artifact has zero rejected rows")

    # 2. Valid v2 (future phase-aware) artifact passes.
    rep = chk.evaluate(valid_v2_artifact())
    check(rep["overall_pass"] is True, "valid v2 (gpu-workload) artifact passes")

    # 3. Zero timing on a measured phase is rejected.
    art = valid_v1_artifact()
    art["results"][0]["best_seconds"] = 0.0
    rep = chk.evaluate(art)
    check(rep["overall_pass"] is False, "zero best_seconds rejected")
    check("zero_timing" in rep["problems"], "zero_timing bucket populated for zero best_seconds")

    art2 = valid_v2_artifact()
    art2["results"][0]["kernel_seconds"] = -0.001
    rep = chk.evaluate(art2)
    check(rep["overall_pass"] is False, "negative kernel_seconds rejected")
    check("zero_timing" in rep["problems"], "zero_timing bucket populated for negative kernel_seconds")

    # 4. CPU-only row relabeled as gpu_acceleration is rejected (both schemas).
    art = valid_v1_artifact()
    art["results"][0]["evidence_class"] = "gpu_acceleration"
    rep = chk.evaluate(art)
    check(rep["overall_pass"] is False, "v1 CPU row claiming gpu_acceleration rejected")
    check("cpu_only_relabeled_as_gpu" in rep["problems"],
          "cpu_only_relabeled_as_gpu bucket populated (v1 schema-wide ban)")

    art2 = valid_v2_artifact()
    art2["results"][0] = valid_v2_row(backend="cpu")
    art2["results"][0]["evidence_class"] = "gpu_acceleration"
    rep = chk.evaluate(art2)
    check(rep["overall_pass"] is False, "v2 CPU-backend row claiming gpu_acceleration rejected")
    check("cpu_only_relabeled_as_gpu" in rep["problems"],
          "cpu_only_relabeled_as_gpu bucket populated (v2 cpu backend)")

    # 5. GPU evidence without hook_installed is rejected.
    art2 = valid_v2_artifact()
    art2["results"][0]["hook_installed"] = False
    rep = chk.evaluate(art2)
    check(rep["overall_pass"] is False, "gpu_acceleration claim without hook_installed rejected")
    check("missing_backend_evidence" in rep["problems"] or "cpu_only_relabeled_as_gpu" in rep["problems"],
          "missing hook_installed flagged")

    # 6. GPU backend without provider_linked is rejected.
    art2 = valid_v2_artifact()
    art2["results"][0]["provider_linked"] = False
    rep = chk.evaluate(art2)
    check(rep["overall_pass"] is False, "non-cpu backend without provider_linked rejected")
    check("missing_backend_evidence" in rep["problems"], "missing_backend_evidence bucket populated")

    # 7. Missing required field is rejected (malformed columns).
    art = valid_v1_artifact()
    del art["results"][0]["best_seconds"]
    rep = chk.evaluate(art)
    check(rep["overall_pass"] is False, "missing best_seconds field rejected")
    check("malformed_columns" in rep["problems"], "malformed_columns bucket populated for missing field")

    # 8. Inconsistent backend/device pairing is rejected.
    art = valid_v1_artifact()
    art["results"][0]["backend"] = "cuda"
    art["results"][0]["device"] = "n/a"
    rep = chk.evaluate(art)
    check(rep["overall_pass"] is False, "backend=cuda with device=n/a rejected")
    check("malformed_columns" in rep["problems"] or "missing_backend_evidence" in rep["problems"],
          "cuda/n-a device mismatch flagged")

    # 9. Arithmetic-inconsistent throughput (ns_per_row doesn't reconstruct best_seconds).
    art = valid_v1_artifact()
    art["results"][0]["ns_per_row"] = art["results"][0]["ns_per_row"] * 5.0
    rep = chk.evaluate(art)
    check(rep["overall_pass"] is False, "ns_per_row/count/best_seconds mismatch rejected")
    check("impossible_throughput" in rep["problems"], "impossible_throughput bucket populated")

    # 10. Missing validation (validation_status != matched_reference) is rejected.
    art = valid_v1_artifact()
    art["results"][0]["validation_status"] = "mismatch"
    rep = chk.evaluate(art)
    check(rep["overall_pass"] is False, "validation_status=mismatch rejected")
    check("missing_validation" in rep["problems"], "missing_validation bucket populated")

    # 11. One-sided speedup claim (only one of the pair present) is rejected.
    art2 = valid_v2_artifact()
    art2["results"][0]["speedup_vs_cpu_forced"] = {"compute_only_ratio": 3.2}
    rep = chk.evaluate(art2)
    check(rep["overall_pass"] is False, "one-sided speedup_vs_cpu_forced rejected")
    check("one_sided_speedup" in rep["problems"], "one_sided_speedup bucket populated")

    # 12. Paired speedup claim (both ratios present) passes.
    art2 = valid_v2_artifact()
    art2["results"][0]["speedup_vs_cpu_forced"] = {
        "compute_only_ratio": 3.2, "end_to_end_ratio": 2.1,
    }
    rep = chk.evaluate(art2)
    check(rep["overall_pass"] is True, "paired speedup_vs_cpu_forced (both ratios) passes")

    # 13. Unknown schema is rejected at load time (exit code 2 path via evaluate()).
    art = valid_v1_artifact()
    art["schema"] = "not-a-real-schema"
    rep = chk.evaluate(art)
    check(rep["overall_pass"] is False, "unknown schema rejected")
    check("error" in rep, "unknown schema reports an error, not a silent pass")

    # 14. Missing host_context is rejected (envelope-level).
    art = valid_v1_artifact()
    del art["host_context"]
    rep = chk.evaluate(art)
    check(rep["overall_pass"] is False, "missing host_context envelope rejected")

    # 15. deepcopy sanity: mutating a fixture copy must not affect the original helper output.
    base = valid_v1_artifact()
    mutated = copy.deepcopy(base)
    mutated["results"][0]["best_seconds"] = 0.0
    rep_base = chk.evaluate(base)
    check(rep_base["overall_pass"] is True, "original fixture untouched by mutated copy")

    # 16. Honest gap: driver_version == null on a valid GPU row PASSES (the
    # GpuBackend interface has no driver-version field; forcing a fabricated
    # string here would itself be dishonest -- see check_lbtc_gpu_workload_evidence.py
    # module docstring "Honest gaps").
    art2 = valid_v2_artifact()
    art2["results"][0] = valid_v2_row(driver_version=None)
    rep = chk.evaluate(art2)
    check(rep["overall_pass"] is True, "driver_version=null on a valid GPU row passes (honest gap, not fabricated)")

    # 17. driver_version == "" (empty string) on a GPU row is still rejected
    # as malformed -- distinguishable from an honest null.
    art2 = valid_v2_artifact()
    art2["results"][0] = valid_v2_row(driver_version="")
    rep = chk.evaluate(art2)
    check(rep["overall_pass"] is False, "driver_version='' (empty string) on a GPU row rejected")
    check("malformed_columns" in rep["problems"], "malformed_columns bucket populated for empty driver_version")

    # 18. Honest gap: upload_seconds/download_seconds == null on a valid GPU
    # row PASSES (no upload/kernel/download phase-split instrumentation
    # exists on the caller side for every current harness).
    art2 = valid_v2_artifact()
    art2["results"][0] = valid_v2_row(phase_split=False)
    rep = chk.evaluate(art2)
    check(rep["overall_pass"] is True,
          "upload_seconds/download_seconds=null on a valid GPU row passes (honest gap, no fabricated split)")

    # 19. But a present, non-null upload_seconds/download_seconds must still
    # be positive -- zero/negative is always rejected.
    art2 = valid_v2_artifact()
    art2["results"][0] = valid_v2_row(phase_split=True)
    art2["results"][0]["upload_seconds"] = 0.0
    rep = chk.evaluate(art2)
    check(rep["overall_pass"] is False, "zero upload_seconds on a GPU row rejected")
    check("zero_timing" in rep["problems"], "zero_timing bucket populated for zero upload_seconds")

    # 20. Explicit gpu_acceleration group check: device == 'n/a' on a
    # gpu_acceleration row is rejected even if every other field looks fine.
    art2 = valid_v2_artifact()
    art2["results"][0] = valid_v2_row()
    art2["results"][0]["device"] = "n/a"
    rep = chk.evaluate(art2)
    check(rep["overall_pass"] is False, "gpu_acceleration row with device='n/a' rejected")
    check("gpu_acceleration_incomplete" in rep["problems"] or "malformed_columns" in rep["problems"],
          "device='n/a' on a gpu_acceleration row flagged")

    # 21. Explicit gpu_acceleration group check: validation_status mismatch
    # on an otherwise-valid GPU row is rejected via the dedicated group rule
    # too (in addition to the pre-existing missing_validation rule).
    art2 = valid_v2_artifact()
    art2["results"][0] = valid_v2_row()
    art2["results"][0]["validation_status"] = "mismatch"
    rep = chk.evaluate(art2)
    check(rep["overall_pass"] is False, "gpu_acceleration row with validation_status=mismatch rejected")
    check("gpu_acceleration_incomplete" in rep["problems"],
          "gpu_acceleration_incomplete bucket populated for mismatched validation on a GPU row")

    # 22. Explicit gpu_acceleration group check: kernel_seconds missing/null
    # on a gpu_acceleration row is rejected by the dedicated group rule.
    art2 = valid_v2_artifact()
    art2["results"][0] = valid_v2_row()
    art2["results"][0]["kernel_seconds"] = None
    rep = chk.evaluate(art2)
    check(rep["overall_pass"] is False, "gpu_acceleration row with kernel_seconds=null rejected")
    check("gpu_acceleration_incomplete" in rep["problems"] or "malformed_columns" in rep["problems"],
          "null kernel_seconds on a gpu_acceleration row flagged")

    # 23. A GPU-backend row under schema v1 (public-ops) with a real
    # backend/device but evidence_class=api_correctness still passes --
    # schema v1 permits honest backend identification, it only forbids
    # claiming gpu_acceleration under that schema (see rule 5 above).
    art = valid_v1_artifact()
    art["results"][0]["backend"] = "opencl"
    art["results"][0]["device"] = "RTX 5060 Ti"
    art["results"][0]["driver_version"] = None
    art["results"][0]["provider_linked"] = True
    art["results"][0]["hook_installed"] = True
    art["results"][0]["evidence_class"] = "api_correctness"
    rep = chk.evaluate(art)
    check(rep["overall_pass"] is True,
          "schema v1 row with real backend/device but api_correctness evidence_class passes")

    print("\n" + ("ALL PASS" if not failures else f"FAILURES: {len(failures)}"))
    return 1 if failures else 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
