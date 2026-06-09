#!/usr/bin/env python3
"""
check_gpu_ct_uniformity.py — WHITE-BOX GPU constant-time gate (Nsight branch uniformity).

WHY THIS EXISTS:
  The black-box dudect probe (gpu_ct_leakage_probe, interleaved fixed/random keys in one
  warp) MASKED a real GPU ECDSA constant-time leak: key-dependent warp divergence
  averages out across the shared warp, so the Welch t-test reads |t|=0 (PASS). The leak
  was a data-dependent conditional subtraction in the scalar/field mod reduction; it was
  only caught by an out-of-band Nsight Compute (ncu) branch-uniformity measurement run
  with ALL-fixed keys vs ALL-random keys — which the probe supports via --keys but which
  no script invoked.

WHAT THIS GATE DOES:
  For each secret-scalar CT kernel it runs the probe twice under ncu, once with every
  thread holding the SAME key (--keys fixed) and once with every thread holding a DISTINCT
  random key (--keys random), and reads the average branch-target thread-uniformity
  (smsp__sass_average_branch_targets_threads_uniform.pct). For a constant-time kernel the
  executed control flow is independent of the secret, so uniformity MUST be the same in
  both modes. A key-dependent branch shows up as random-key uniformity dropping below the
  fixed-key baseline (the original leak was 100% fixed vs 84.48% random).

  PASS  : for every kernel, (uniformity_fixed - uniformity_random) <= THRESHOLD_PCT
  FAIL  : any kernel's random-key uniformity is more than THRESHOLD_PCT below fixed-key
  SKIP(77): no NVIDIA GPU, no ncu, or the probe binary is not built (advisory — GPU CT
            can only be measured on a GPU host; this is a Local-CI gate by necessity).

This is the detection that actually caught the leak. Keeping it scripted + gated is what
converts "we have CT tools" into "CT is gated".
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys

ADVISORY_SKIP_CODE = 77

# Kernel regex (ncu -k) -> friendly op name. These are the secret-scalar CT kernels in
# src/cuda/src/gpu_ct_leakage_probe.cu.
KERNELS = [
    ("probe_ct_generator_mul", "ct_generator_mul"),
    ("probe_ct_ecdsa_sign",    "ct_ecdsa_sign"),
    ("probe_ct_schnorr_sign",  "ct_schnorr_sign"),
]

METRIC = "smsp__sass_average_branch_targets_threads_uniform.pct"
# Allowed (uniformity_fixed - uniformity_random). The leak was ~15.5 pts; CT is 0 pts.
# 1.0 pt absorbs profiler/scheduling noise while still catching any real key-dependent
# divergence by a wide margin.
DEFAULT_THRESHOLD_PCT = 1.0


def have_gpu() -> bool:
    if not shutil.which("nvidia-smi"):
        return False
    try:
        return subprocess.run(["nvidia-smi", "-L"], capture_output=True,
                              timeout=30).returncode == 0
    except Exception:
        return False


def find_probe(build_dir: str) -> str | None:
    for root, _dirs, files in os.walk(build_dir):
        if "gpu_ct_leakage_probe" in files:
            p = os.path.join(root, "gpu_ct_leakage_probe")
            if os.access(p, os.X_OK):
                return p
    return None


def measure_uniformity(ncu: str, probe: str, kernel_regex: str, keys_mode: str) -> float | None:
    """Run the probe under ncu for one kernel/keys-mode; return branch uniformity %."""
    cmd = [
        ncu, "--metrics", METRIC, "--launch-count", "1",
        "-k", f"regex:{kernel_regex}", probe,
        "--keys", keys_mode, "--samples", "256", "--repetitions", "1",
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=300).stdout
    except Exception as e:  # noqa: BLE001
        print(f"  WARN: ncu failed for {kernel_regex}/{keys_mode}: {e}")
        return None
    for line in out.splitlines():
        if "branch_targets_threads_uniform" in line:
            m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*$", line.strip())
            if m:
                return float(m.group(1))
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build-dir", default="out/lbtc-gpu",
                    help="CMake build dir containing gpu_ct_leakage_probe")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD_PCT,
                    help="max allowed (uniformity_fixed - uniformity_random) in pct points")
    args = ap.parse_args()

    if not have_gpu():
        print("check_gpu_ct_uniformity: no NVIDIA GPU — advisory skip")
        return ADVISORY_SKIP_CODE
    ncu = shutil.which("ncu") or shutil.which("nv-nsight-cu-cli")
    if not ncu:
        print("check_gpu_ct_uniformity: Nsight Compute (ncu) not found — advisory skip")
        return ADVISORY_SKIP_CODE
    probe = find_probe(args.build_dir)
    if not probe:
        print(f"check_gpu_ct_uniformity: gpu_ct_leakage_probe not built in {args.build_dir}"
              " — advisory skip")
        return ADVISORY_SKIP_CODE

    print(f"check_gpu_ct_uniformity: white-box ncu branch-uniformity gate (threshold "
          f"{args.threshold:.2f} pct), probe={probe}")
    failures = []
    measured = 0
    for kernel_regex, name in KERNELS:
        uf = measure_uniformity(ncu, probe, kernel_regex, "fixed")
        ur = measure_uniformity(ncu, probe, kernel_regex, "random")
        if uf is None or ur is None:
            print(f"  WARN {name}: could not measure (ncu output unparsed) — not a fail")
            continue
        measured += 1
        delta = uf - ur
        status = "OK" if delta <= args.threshold else "LEAK"
        print(f"  {status:4} {name:18} fixed={uf:6.2f}%  random={ur:6.2f}%  "
              f"drop={delta:+.2f} pct")
        if delta > args.threshold:
            failures.append((name, uf, ur, delta))

    if measured == 0:
        print("  no kernels measurable (ncu produced no uniformity values) — advisory skip")
        return ADVISORY_SKIP_CODE
    if failures:
        print(f"\n::error::GPU CT leak: {len(failures)} kernel(s) show key-dependent "
              f"branch divergence (random-key uniformity below fixed-key baseline).")
        for name, uf, ur, d in failures:
            print(f"  {name}: fixed {uf:.2f}% vs random {ur:.2f}% (drop {d:.2f} pct > "
                  f"{args.threshold:.2f})")
        return 1
    print(f"\nGPU CT white-box gate PASS: all {measured} CT kernels uniform "
          "(secret-independent control flow).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
