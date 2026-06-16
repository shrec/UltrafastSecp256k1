#!/usr/bin/env python3
"""
dudect_ct_probe.py — binary-level constant-time probe (dudect-style Welch t-test).

WHY (the bug class this prevents):
ci/check_ct_branches.py is a SOURCE lint: it catches secret-dependent branches written
in the source. It CANNOT see a timing leak the compiler introduces (a cmov lowered to a
branch, a table lookup that became data-dependent, a div the backend specialised). The
only thing that catches those is measuring the COMPILED binary's timing on two input
classes and testing for a statistically significant difference — the dudect method
(Reparaz/Balasch/Verbauwhede, ePrint 2016/1123).

This probe is two layers, kept honestly separate:

  1. DETECTOR (verified): the Welch t-test + leak classifier. A leak is a timing
     distribution that differs by input class. classify() returns 'leak' when |t| is
     large. ci/test_dudect_ct_probe.py PROVES this layer blocks by feeding it a known
     leaky distribution (must flag) and a known-uniform one (must pass). This is the
     DON'T-TRUST-VERIFY guarantee: the detector is proven to detect a leak.

  2. LIVE MEASUREMENT (advisory): running it against a real CT-timing binary that emits
     cycle samples for two classes. This is inherently noisy on shared/CI hardware
     (DVFS, SMT, scheduler), so the live layer is ADVISORY — it reports, and only a
     |t| past the hard leak threshold is treated as a finding; the borderline band
     advisory-skips. (The audit suite's exploit_eucleak_inversion_timing covers the
     ct::scalar_inverse path the same way.)

Exit:  0 = clean / detector ok ;  77 = advisory skip (no live target binary) ;
       1 = a live measurement crossed the hard leak threshold.
"""
import math
import os
import sys

# dudect thresholds on |t|. dudect's own guidance: t>10 is a definite leak; t<5 is fine;
# 5..10 is borderline (needs more samples / advisory). We treat >=HARD as a finding.
T_HARD_LEAK = 10.0
T_BORDERLINE = 5.0


def welch_t(a, b):
    """Welch's t-statistic for two unequal-variance samples. 0.0 if undefined."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0
    ma, mb = sum(a) / na, sum(b) / nb
    va = sum((x - ma) ** 2 for x in a) / (na - 1)
    vb = sum((x - mb) ** 2 for x in b) / (nb - 1)
    denom = math.sqrt(va / na + vb / nb)
    if denom == 0.0:
        return 0.0
    return (ma - mb) / denom


def classify(t):
    """Map a t-statistic to a verdict: 'leak' | 'borderline' | 'uniform'."""
    at = abs(t)
    if at >= T_HARD_LEAK:
        return "leak"
    if at >= T_BORDERLINE:
        return "borderline"
    return "uniform"


def _percentile_crop(samples, pct):
    """dudect crops the upper tail (cache misses / preemptions) before testing."""
    if not samples:
        return samples
    s = sorted(samples)
    cut = max(1, int(len(s) * pct))
    return s[:cut]


def measure_pair(class0, class1, crop_pct=0.95):
    """Crop tails then Welch-t. Returns (t, verdict)."""
    a = _percentile_crop(class0, crop_pct)
    b = _percentile_crop(class1, crop_pct)
    t = welch_t(a, b)
    return t, classify(t)


def _read_samples(path):
    """Parse a target binary's output: two whitespace-separated columns 'class cycles'."""
    class0, class1 = [], []
    for line in open(path, encoding="utf-8", errors="replace"):
        parts = line.split()
        if len(parts) != 2:
            continue
        try:
            cls, cyc = int(parts[0]), float(parts[1])
        except ValueError:
            continue
        (class0 if cls == 0 else class1).append(cyc)
    return class0, class1


def main() -> int:
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--samples", help="path to a 'class cycles' samples file from a CT-timing target")
    args = p.parse_args()

    samples = args.samples or os.environ.get("DUDECT_SAMPLES")
    print("=" * 70)
    print("  dudect binary-CT probe (Welch t-test leak detector)")
    print("=" * 70)
    if not samples or not os.path.exists(samples):
        print("  ADVISORY SKIP: no live CT-timing samples file provided "
              "(set --samples / $DUDECT_SAMPLES to a 'class cycles' file).")
        print("  The DETECTOR is verified by ci/test_dudect_ct_probe.py on synthetic data;")
        print("  live measurement is advisory (CI timing noise).")
        return 77

    class0, class1 = _read_samples(samples)
    t, verdict = measure_pair(class0, class1)
    print(f"  samples: class0={len(class0)}  class1={len(class1)}  |t|={abs(t):.2f}  -> {verdict}")
    if verdict == "leak":
        print(f"\n  \033[91mFAIL\033[0m  |t|={abs(t):.2f} >= {T_HARD_LEAK}: statistically significant "
              f"timing difference by input class — binary-level CT leak.")
        return 1
    if verdict == "borderline":
        print(f"  ADVISORY: |t|={abs(t):.2f} in borderline band [{T_BORDERLINE},{T_HARD_LEAK}) — "
              f"inconclusive on this hardware; collect more samples.")
        return 77
    print(f"  OK: |t|={abs(t):.2f} < {T_BORDERLINE} — no measurable timing difference.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
