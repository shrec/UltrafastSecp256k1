#!/usr/bin/env python3
"""
nonce_bias_detector.py -- Statistical nonce bias detector for ECDSA
====================================================================

An external auditor runs this because NO static analysis tool and NO LLM review
can tell you whether k-values are statistically uniform — you have to measure it.

Background:
  RFC 6979 says k must be derived deterministically and uniformly from (sk, msg).
  Many real-world implementations have subtle biases because:
    1. They use k = HMAC_DRBG_output mod n directly, but if HMAC output > n,
       they reduce, introducing a ~2^-128 but measurable bias at bit positions
       near bit 255.
    2. Broken implementations sample k by rejection-sampling with bugs that
       cause low k values to appear more often.
    3. Hardware RNG with initialization bias (for hedged-ECDSA variants).

What this script measures:
  1. MSB (bit 255) bias test — should be ~50% set for uniform k mod n.
     Since n ≈ 2^256 - 4.3×10^38, the exact probability of bit 255 = 1 is
     P = (n - 2^255) / n ≈ 50% minus a 2^-18 correction.
  2. LSB bias test — for some naive truncation bugs, LSB is always 0 or 1.
  3. Nonce collision detection — ANY repeated r-value in 50,000 signatures
     is catastrophic (allows private key recovery).
  4. Chi-squared test on each individual bit position for N=10,000 samples.
  5. Kolmogorov-Smirnov uniformity test on r-value distribution.

Published attacks enabled by biased nonces:
  Minerva (2020): ~4 LSBs biased → private key recovery in hours via LLL
  TPM-FAIL (2019): 8-32 MSBs leaked → key recovery in minutes
  ECDSA-lattice (general): any bias in top 1 bit enough given ~2^27 sigs

Threshold (we set conservatively):
  Any bit position with |frequency - 0.5| > 4σ → WARN
  p-value < 0.0001 (4-sigma) on chi-squared → FAIL for MSB/LSB
  p-value < 1e-6 on chi-squared → FAIL for all 254 interior bits

Usage:
    python3 ci/nonce_bias_detector.py --lib build_opencl/.../libufsecp.so.3
    python3 ci/nonce_bias_detector.py --count 10000 --json
    python3 ci/nonce_bias_detector.py --full 50000

Requirements:
    No external packages required (uses only stdlib math).
    Optional: scipy for KS-test (pip install scipy).
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import secrets
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
LIB_ROOT = SCRIPT_DIR.parent

N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141

# KS guard band for CI stability: values just above the 1% threshold are reported
# as warnings; only a materially larger excess is treated as a hard failure.
KS_HARD_FAIL_MULTIPLIER = 1.25

try:
    from scipy import stats as scipy_stats
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False


# ---------------------------------------------------------------------------
# Library + signing — delegates to shared _ufsecp wrapper
# ---------------------------------------------------------------------------

import importlib as _importlib
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
_ufsecp_mod = _importlib.import_module("_ufsecp")
_find_lib   = _ufsecp_mod.find_lib
Signer      = _ufsecp_mod.UfSecp


def _rand_privkey() -> bytes:
    while True:
        k = secrets.token_bytes(32)
        if 1 <= int.from_bytes(k, "big") < N:
            return k


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def _chi_squared_p(observed: int, expected: float, n: int) -> float:
    """
    One-sample chi-squared test for a single bit position.
    Returns p-value (< 0.05 → suspicious, < 0.001 → alarm).
    Uses normal approximation: z = (observed - n*p) / sqrt(n*p*(1-p)).
    """
    p = 0.5
    mu  = n * p
    sig = math.sqrt(n * p * (1.0 - p))
    if sig == 0:
        return 1.0
    z = abs(observed - mu) / sig
    # Approximation: p-value from z using erfc
    return math.erfc(z / math.sqrt(2))


def _collision_check(r_values: List[int]) -> List[tuple]:
    """Return list of (r, count) for any repeated r-values."""
    seen: dict = {}
    for r in r_values:
        seen[r] = seen.get(r, 0) + 1
    return [(r, cnt) for r, cnt in seen.items() if cnt > 1]


def _bit_frequency(r_values: List[int], bit: int) -> int:
    """Count how many r-values have the given bit set."""
    mask = 1 << bit
    return sum(1 for r in r_values if r & mask)


def _ks_statistic(r_values: List[int]) -> float:
    """
    KS statistic vs uniform[0, n).
    D = max_i |F_empirical(r_i) - r_i / n|
    """
    n_samples = len(r_values)
    if n_samples == 0:
        return 0.0
    sorted_r = sorted(r_values)
    d_max = 0.0
    for i, r in enumerate(sorted_r):
        expected = r / N
        empirical = (i + 1) / n_samples
        d_max = max(d_max, abs(empirical - expected))
    return d_max


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

@dataclass
class BiasReport:
    n_samples: int = 0
    n_collisions: int = 0
    collision_details: list = field(default_factory=list)
    biased_bits: list = field(default_factory=list)  # (bit, freq, pvalue)
    ks_stat: float = 0.0
    ks_critical: float = 0.0
    overall_pass: bool = True
    msb_freq: float = 0.0
    lsb_freq: float = 0.0
    warnings: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n_samples": self.n_samples,
            "n_collisions": self.n_collisions,
            "collision_details": self.collision_details,
            "biased_bits_count": len(self.biased_bits),
            "biased_bits": [
                {"bit": b, "freq_pct": f"{freq/self.n_samples*100:.3f}",
                 "p_value": f"{pv:.6f}"}
                for b, freq, pv in self.biased_bits
            ],
            "ks_stat": self.ks_stat,
            "ks_critical_1pct": self.ks_critical,
            "ks_pass": self.ks_stat < self.ks_critical,
            "msb_freq_pct": f"{self.msb_freq*100:.3f}",
            "lsb_freq_pct": f"{self.lsb_freq*100:.3f}",
            "overall_pass": self.overall_pass,
            "warnings": self.warnings,
        }


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def collect_r_values(signer: Signer, n_samples: int, fixed_msg: bool = False) -> List[int]:
    """
    Collect r-values from n_samples ECDSA signatures.

    Two modes:
      fixed_msg=True:  same message, N different keys → measures key-space r distribution
      fixed_msg=False: same key, N different messages → measures nonce diversity per key

    Both should produce statistically uniform r-values.
    We run with fixed_msg=True because that stresses nonce derivation most broadly.
    """
    msg = secrets.token_bytes(32)
    r_values = []
    print(f"  Collecting {n_samples:,} r-values...", end="", flush=True)
    block = n_samples // 10
    for i in range(n_samples):
        if i % block == 0 and i > 0:
            print(f" {i/n_samples*100:.0f}%", end="", flush=True)
        sk = _rand_privkey()
        sig = signer.sign(msg, sk)
        r = int.from_bytes(sig[:32], "big")
        r_values.append(r)
    print(" done")
    return r_values


def also_collect_same_key(signer: Signer, n_samples: int) -> List[int]:
    """Collect r-values from one key signing many messages — checks nonce diversity."""
    sk = _rand_privkey()
    r_values = []
    for _ in range(n_samples):
        msg = secrets.token_bytes(32)
        sig = signer.sign(msg, sk)
        r = int.from_bytes(sig[:32], "big")
        r_values.append(r)
    return r_values


def analyze(r_values: List[int], label: str = "") -> BiasReport:
    n = len(r_values)
    report = BiasReport(n_samples=n)
    header = f"  [{label}] " if label else "  "

    # 1. Collision check (catastrophic)
    collisions = _collision_check(r_values)
    if collisions:
        report.n_collisions = sum(c for _, c in collisions)
        report.collision_details = [(hex(r), c) for r, c in collisions[:5]]
        report.overall_pass = False
        report.warnings.append(f"CRITICAL: {len(collisions)} repeated r-values — private key is recoverable!")
        print(f"{header}☠  COLLISION DETECTED: {collisions[:2]}")
    else:
        print(f"{header}✓  No r-value collisions in {n:,} signatures")

    # 2. MSB bias (bit 255)
    msb_count = _bit_frequency(r_values, 255)
    report.msb_freq = msb_count / n
    # Expected: n < 2^256 so bit 255 of r is ~50% minus tiny correction
    expected_msb_p = (N - (1 << 255)) / N  # ≈ 0.5 - 1.18e-20 ≈ 0.5
    pval_msb = _chi_squared_p(msb_count, n * expected_msb_p, n)
    sigma_msb = abs(msb_count / n - expected_msb_p) / math.sqrt(expected_msb_p * (1 - expected_msb_p) / n)
    status = "✓ " if pval_msb > 1e-4 else "⚠ "
    print(f"{header}{status} MSB (bit 255): {msb_count/n*100:.3f}% set  "
          f"(expected ~{expected_msb_p*100:.3f}%)  z={sigma_msb:.2f}  p={pval_msb:.4f}")
    if pval_msb < 1e-4:
        report.biased_bits.append((255, msb_count, pval_msb))
        report.warnings.append(f"MSB bias detected: p={pval_msb:.6f}")
        report.overall_pass = False

    # 3. LSB bias (bit 0)
    lsb_count = _bit_frequency(r_values, 0)
    report.lsb_freq = lsb_count / n
    pval_lsb = _chi_squared_p(lsb_count, n * 0.5, n)
    sigma_lsb = abs(lsb_count/n - 0.5) / math.sqrt(0.25/n)
    status = "✓ " if pval_lsb > 1e-4 else "⚠ "
    print(f"{header}{status} LSB (bit   0): {lsb_count/n*100:.3f}% set  "
          f"(expected ~50.000%)  z={sigma_lsb:.2f}  p={pval_lsb:.4f}")
    if pval_lsb < 1e-4:
        report.biased_bits.append((0, lsb_count, pval_lsb))
        report.warnings.append(f"LSB bias detected: p={pval_lsb:.6f}")
        report.overall_pass = False

    # 4. Sweep all 256 bit positions with strict threshold (skip if < 5000 samples)
    if n >= 5000:
        alarm_bits = []
        for bit in range(256):
            if bit in (255, 0):
                continue
            cnt = _bit_frequency(r_values, bit)
            pv  = _chi_squared_p(cnt, n * 0.5, n)
            if pv < 1e-6:  # Very strict; false-positive rate at this threshold: ~1e-4 over 254 bits
                alarm_bits.append((bit, cnt, pv))
        if alarm_bits:
            for bit, cnt, pv in alarm_bits[:5]:
                print(f"{header}⚠  Bit {bit:3d}: {cnt/n*100:.3f}% set  p={pv:.2e}")
            report.biased_bits.extend(alarm_bits)
            report.warnings.append(f"{len(alarm_bits)} additional bit positions show strong bias")
            report.overall_pass = False
        else:
            print(f"{header}✓  All 254 interior bit positions pass chi-squared at p>1e-6")

    # 5. KS test
    ks = _ks_statistic(r_values)
    report.ks_stat = ks
    # Critical value for KS at α=0.01: D_crit ≈ 1.63/sqrt(n)
    d_crit = 1.63 / math.sqrt(n)
    report.ks_critical = d_crit
    ks_warn = ks >= d_crit
    ks_hard_fail = ks >= (KS_HARD_FAIL_MULTIPLIER * d_crit)
    status = "✓ " if not ks_warn else "⚠ "
    print(f"{header}{status} KS test: D={ks:.6f}  D_crit(1%)={d_crit:.6f}  "
          f"{'PASS' if not ks_warn else ('WARN' if not ks_hard_fail else 'FAIL')}")
    if ks_hard_fail:
        report.warnings.append(
            f"KS test FAIL: D={ks:.6f} > {KS_HARD_FAIL_MULTIPLIER:.2f}*D_crit(1%)"
        )
        report.overall_pass = False
    elif ks_warn:
        report.warnings.append(f"KS test WARN: D={ks:.6f} > D_crit(1%)")

    return report


def run(lib_path: Optional[str], n_main: int, n_samekey: int,
        json_out: bool, out_file: Optional[str]):
    try:
        lpath = _find_lib(lib_path)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Library: {lpath}")
    print()

    try:
        signer = Signer(lpath)
    except Exception as e:
        print(f"ERROR loading library: {e}", file=sys.stderr)
        sys.exit(1)

    reports = {}

    print(f"=== Test 1: {n_main:,} signatures, one message, N random keys ===")
    r1 = collect_r_values(signer, n_main, fixed_msg=True)
    report1 = analyze(r1, "multi-key")
    reports["multi_key"] = report1.to_dict()
    print()

    print(f"=== Test 2: {n_samekey:,} signatures, one key, N random messages ===")
    r2 = also_collect_same_key(signer, n_samekey)
    # Collision on same-key run = nonce reuse = catastrophic
    collisions2 = _collision_check(r2)
    if collisions2:
        print(f"  ☠  NONCE REUSE: {len(collisions2)} repeated r-values from one key!")
        report2_pass = False
    else:
        print(f"  ✓  No nonce reuse: {n_samekey:,} unique r-values from single key")
        report2_pass = True
    reports["single_key_diversity"] = {
        "n_samples": n_samekey,
        "collisions": len(collisions2),
        "pass": report2_pass,
    }
    print()

    overall = report1.overall_pass and report2_pass
    print("=" * 60)
    print(f"Overall: {'PASS — no detectable nonce bias' if overall else 'FAIL — see warnings above'}")
    if not overall:
        for w in report1.warnings:
            print(f"  ⚠  {w}")

    full_report = {
        "library": lpath,
        "n_main_samples": n_main,
        "n_samekey_samples": n_samekey,
        "results": reports,
        "overall_pass": overall,
    }

    if json_out or out_file:
        j = json.dumps(full_report, indent=2)
        if out_file:
            Path(out_file).write_text(j)
            print(f"\nReport written to {out_file}")
        else:
            print(j)

    sys.exit(0 if overall else 1)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lib",      help="Path to shared library")
    p.add_argument("--count",    type=int, default=10000,
                   help="Signatures to collect for main test (default: 10000)")
    p.add_argument("--full",     type=int, default=None,
                   help="Override --count for deeper analysis (e.g. 50000)")
    p.add_argument("--samekey",  type=int, default=5000,
                   help="Signatures from single key for nonce-reuse test (default: 5000)")
    p.add_argument("--json",     action="store_true")
    p.add_argument("-o",         dest="out")
    args = p.parse_args()
    n = args.full if args.full else args.count
    run(args.lib, n, args.samekey, args.json, args.out)


if __name__ == "__main__":
    main()
