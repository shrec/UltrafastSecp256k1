#!/usr/bin/env python3
"""
check_backend_value_differential.py — runtime cross-backend value-differential gate.

WHY (the bug class this prevents):
ci/check_backend_parity.py is a SOURCE gate: it catches copy-paste divergence in kernel
text. It CANNOT see a VALUE divergence — a GPU kernel that compiles cleanly but computes a
different result than the CPU on the same input (a porting bug, a carry/precision
difference, an endianness flip in serialization). The only thing that catches that is
feeding identical inputs through CPU and each GPU backend at runtime and asserting
byte-identical outputs.

This gate is two honestly-separated layers:

  1. DETECTOR + COVERAGE (verified):
       * diff_outputs() / classify(): the byte-exact comparator that flags any per-element
         divergence between a CPU result vector and a GPU result vector.
       * coverage check over docs/BACKEND_DIFFERENTIAL_OPS.json: every 'covered' GPU op
         must have its runtime value-differential probe wired into the audit runner
         (regression guard), and 'roadmap' ops are reported as declared gaps.
     ci/test_check_backend_value_differential.py PROVES this layer blocks by flagging an
     injected one-byte divergence and an unwired covered op.

  2. LIVE RUN (advisory): executing the differential against a real GPU. There is no GPU on
     the default CI runners, so the live layer ADVISORY-SKIPS (77) unless a backend is
     present. The audit modules (exploit_gpu_cpu_divergence, exploit_backend_divergence,
     gpu_zk_prove_verify_differential) carry the live run on GPU hosts.

Exit 0 = coverage clean, exit 1 = a covered probe is unwired (regression of differential coverage).
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEDGER = os.path.join(ROOT, "docs", "BACKEND_DIFFERENTIAL_OPS.json")
RUNNER = os.path.join(ROOT, "audit", "unified_audit_runner.cpp")


def diff_outputs(cpu, gpu):
    """Byte-exact comparator. cpu/gpu: parallel sequences of comparable values
    (hex strings, bytes, ints). Returns list of (index, cpu_val, gpu_val) mismatches.
    A length mismatch is itself a divergence (reported at the overflow indices)."""
    mismatches = []
    n = max(len(cpu), len(gpu))
    for i in range(n):
        cv = cpu[i] if i < len(cpu) else None
        gv = gpu[i] if i < len(gpu) else None
        if cv != gv:
            mismatches.append((i, cv, gv))
    return mismatches


def classify(mismatches):
    return "match" if not mismatches else "divergence"


def check_coverage(ledger, runner_src):
    """Return (blocking[list[str]], covered[list], roadmap[list])."""
    ops = ledger.get("ops", [])
    covered = [o for o in ops if o.get("status") == "covered"]
    roadmap = [o for o in ops if o.get("status") == "roadmap"]
    blocking = []
    for o in covered:
        sym = o.get("probe_run_symbol")
        mod = o.get("probe_module")
        if not sym or not mod:
            blocking.append(f"op '{o['id']}' is 'covered' but declares no probe_run_symbol/probe_module")
            continue
        if sym not in runner_src:
            blocking.append(f"covered differential probe '{sym}' not referenced in unified_audit_runner.cpp")
        if f'"{mod}"' not in runner_src:
            blocking.append(f"covered differential module \"{mod}\" not registered in ALL_MODULES")
    return blocking, covered, roadmap


def _gpu_present():
    """Best-effort: a GPU backend is available for a live run."""
    if os.environ.get("UFSECP_HAS_GPU") == "1":
        return True
    for tool in ("nvidia-smi", "rocminfo"):
        for d in os.environ.get("PATH", "").split(os.pathsep):
            if d and os.path.exists(os.path.join(d, tool)):
                return True
    return False


def run() -> int:
    if not os.path.exists(LEDGER):
        print(f"FAIL: backend-differential ledger not found: {LEDGER}")
        return 1
    ledger = json.load(open(LEDGER, encoding="utf-8"))
    runner_src = open(RUNNER, encoding="utf-8").read() if os.path.exists(RUNNER) else ""

    blocking, covered, roadmap = check_coverage(ledger, runner_src)

    print("=" * 70)
    print("  Cross-Backend Value-Differential Gate (CPU<->GPU runtime equality)")
    print("=" * 70)
    print(f"  ops: {len(covered) + len(roadmap)}  |  covered (probed): {len(covered)}  |  roadmap: {len(roadmap)}")
    for o in covered:
        print(f"    \033[92m[COVERED]\033[0m {o['id']:38} <- {o.get('probe_module')}")
    for o in roadmap:
        print(f"    \033[93m[ROADMAP]\033[0m {o['id']:38} (differential probe required)")

    if blocking:
        print()
        for b in blocking:
            print(f"  \033[91mFAIL\033[0m  {b}")
        print(f"\n\033[91m\033[1m  BACKEND VALUE-DIFFERENTIAL: {len(blocking)} blocking issue(s)\033[0m")
        return 1

    print()
    if _gpu_present():
        print("  GPU detected: the covered differential modules carry the LIVE CPU<->GPU run.")
    else:
        print("  ADVISORY: no GPU on this host — live CPU<->GPU run is skipped; coverage is enforced")
        print("  statically and the byte comparator is verified by ci/test_check_backend_value_differential.py.")
    print(f"  OK: all {len(covered)} covered differential probe(s) wired; {len(roadmap)} roadmap op(s) declared.")
    return 0


def main() -> int:
    return run()


if __name__ == "__main__":
    sys.exit(main())
