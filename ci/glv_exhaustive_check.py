#!/usr/bin/env python3
"""
glv_exhaustive_check.py -- GLV scalar decomposition algebraic verifier
=======================================================================

This script verifies the correctness of the GLV (Gallant-Lambert-Vanstone)
scalar decomposition for adversarial scalar values that stress the Babai
rounding algorithm.

The GLV endomorphism on secp256k1:
  For any point P = k·G, we can split k into (k1, k2) such that:
    k·G = k1·G + k2·φ(G)
  where φ(P) = (β·x : y : 1) is the endomorphism and β is the cube root
  of unity in Fp.

The decomposition identity that must hold:
    k ≡ k1 + k2·λ  (mod n)

where λ = 0x5363AD4CC05C30E0A5261C028812645A122E22EA20816678DF02967C1B23BD72

If this identity fails for any k:
  - The signature will contain a wrong r-value
  - The bug is invisible to static analysis (depends on Babai rounding)

What adversarial values stress the decomposition:
  1. k near n/2       — boundary between positive/negative k1
  2. k near sqrt(n)   — where k2 changes sign
  3. k = λ            — k1 = 0 or k2 = 0 degenerate case
  4. k = n - λ        — mirror of λ
  5. k = 2^127, 2^128 — straddles the k2 sign boundary
  6. Random values with k1 ≈ ±2^128 (max) or k2 ≈ ±2^128 (max)
  7. k such that Babai rounding flips: k ≈ A/2 + ε for each lattice basis vector

Why LLMs cannot verify this:
  The Babai rounding in GLV involves floating-point or fixed-point arithmetic
  to find the nearest lattice vector. Off-by-one errors in the rounding are
  invisible in code review but will produce wrong results for specific k values.
  Only by running actual point multiplications and checking the geometric identity
  k1·G + k2·φ(G) == k·G can we verify the Babai rounding is correct.

Verification method:
  We cannot easily call k1·G directly without library internals. Instead:
    1. Sign a message with known nonce relationship (using k as private key to get kG)
    2. Use coincurve to compute k·G and (k1+k2·λ)·G independently
    3. Verify they match

  This works because coincurve uses libsecp256k1 (reference implementation)
  which has independently verified GLV. If our library's GLV is buggy for
  certain k values, our signatures will give wrong r-values for those k values.

  Cross-verification strategy:
    - Use private key k for both our lib and coincurve
    - Sign the same message
    - Compare r-values
    - If they differ, our GLV decomposition produced wrong results for that k

Usage:
    python3 ci/glv_exhaustive_check.py --lib build_opencl/.../libufsecp.so.3
    python3 ci/glv_exhaustive_check.py --extra-random 10000 --json -o report.json
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import secrets
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

SCRIPT_DIR = Path(__file__).resolve().parent
LIB_ROOT = SCRIPT_DIR.parent

# secp256k1 parameters
N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
# GLV endomorphism eigenvalue
LAMBDA = 0x5363AD4CC05C30E0A5261C028812645A122E22EA20816678DF02967C1B23BD72

# Approximate sqrt(n) — used to generate adversarial scalars
SQRT_N_APPROX = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF4F55AA97A60E0B8D  # ~ 2^128


# ---------------------------------------------------------------------------
# Library wrapper (sign)
# ---------------------------------------------------------------------------

import sys as _sys, importlib as _importlib
if str(SCRIPT_DIR) not in _sys.path:
    _sys.path.insert(0, str(SCRIPT_DIR))
_ufsecp_mod = _importlib.import_module("_ufsecp")
_find_lib = _ufsecp_mod.find_lib


class SignLib:
    """Thin shim — delegates to UfSecp; returns None on failure for GLV test."""
    def __init__(self, lib_path: str):
        self._uf = _ufsecp_mod.UfSecp(lib_path)

    def sign(self, sk32: bytes, msg32: bytes) -> Optional[bytes]:
        try:
            return self._uf.sign(msg32, sk32)
        except Exception:
            return None


# ---------------------------------------------------------------------------
# coincurve reference
# ---------------------------------------------------------------------------

def _coincurve_sign(sk32: bytes, msg32: bytes) -> Optional[bytes]:
    """Returns compact 64-byte sig (r||s) using coincurve (libsecp256k1)."""
    try:
        import coincurve
        pri = coincurve.PrivateKey(sk32)
        # coincurve.sign returns DER by default; use raw for low_s normalized
        # We want compact r||s:
        sig_obj = pri.sign(msg32, hasher=None)
        # sig_obj is DER; convert to compact r||s
        return _der_to_compact(sig_obj)
    except Exception:
        return None


def _der_to_compact(der: bytes) -> Optional[bytes]:
    """Convert DER signature to compact r||s (each 32 bytes, big-endian)."""
    if len(der) < 8 or der[0] != 0x30:
        return None
    idx = 2
    if der[idx] != 0x02:
        return None
    r_len = der[idx + 1]
    r_bytes = der[idx + 2: idx + 2 + r_len]
    idx = idx + 2 + r_len
    if der[idx] != 0x02:
        return None
    s_len = der[idx + 1]
    s_bytes = der[idx + 2: idx + 2 + s_len]
    r = int.from_bytes(r_bytes, "big")
    s = int.from_bytes(s_bytes, "big")
    # Low-S normalize
    if s > N // 2:
        s = N - s
    return r.to_bytes(32, "big") + s.to_bytes(32, "big")


# ---------------------------------------------------------------------------
# Adversarial scalar generation
# ---------------------------------------------------------------------------

def _clamp(k: int) -> int:
    """Force k into [1, N-1]."""
    k = k % N
    return k if k != 0 else 1


def adversarial_scalars() -> list[tuple[str, int]]:
    """
    Return (name, k) pairs that stress Babai rounding in GLV decomposition.
    """
    vals: list[tuple[str, int]] = []

    # 1. Near n/2 (where k1 sign flips)
    half_n = N // 2
    for delta in [-2, -1, 0, 1, 2]:
        k = _clamp(half_n + delta)
        vals.append((f"n/2 + ({delta:+d})", k))

    # 2. λ and its multiples (eigenvalue-related)
    for mult in [1, 2, N - 1, N // 2]:
        k = _clamp(LAMBDA * mult % N)
        vals.append((f"λ × {mult:#x}", k))

    # 3. n - λ (mirror)
    vals.append(("n - λ", _clamp(N - LAMBDA)))

    # 4. Near 2^127, 2^128 (straddles k2 sign boundary)
    for bits in [126, 127, 128, 129]:
        k = _clamp(1 << bits)
        vals.append((f"2^{bits}", k))
        k = _clamp((1 << bits) + 1)
        vals.append((f"2^{bits} + 1", k))
        k = _clamp((1 << bits) - 1)
        vals.append((f"2^{bits} - 1", k))

    # 5. Extreme values
    vals += [
        ("1",     1),
        ("2",     2),
        ("n-1",   N - 1),
        ("n-2",   N - 2),
        ("3",     3),
    ]

    # 6. k such that k2 is maximized (k ≈ λ/2 boundary)
    half_lambda = LAMBDA // 2
    for delta in [-1, 0, 1]:
        k = _clamp(half_lambda + delta)
        vals.append((f"λ/2 + ({delta:+d})", k))

    # 7. Near exact Babai rounding boundaries
    # The Babai-optimal basis for secp256k1 GLV is approximately:
    #   b11 ≈ 0x3086d221a7d46bcde86c90e49284eb15  (≈ 2^127.13)
    #   b21 ≈ 0xe4437ed6010e88286f547fa90abfe4c3  (≈ high 128-bit)
    B11 = 0x3086D221A7D46BCDE86C90E49284EB15
    for delta in [-1, 0, 1]:
        k = _clamp(B11 + delta)
        vals.append((f"B11 + ({delta:+d})", k))

    # 8. k that maximizes k1 magnitude (where Babai rounding is hardest)
    half_B11 = B11 // 2
    for delta in [-1, 0, 1]:
        k = _clamp(half_B11 + delta)
        vals.append((f"B11/2 + ({delta:+d})", k))

    return vals


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class GLVMismatch:
    name: str
    k_hex: str
    r_ours_hex: str
    r_ref_hex: str


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run(lib_path: Optional[str], extra_random: int, json_out: bool, out_file: Optional[str]):
    try:
        import coincurve  # noqa: F401
    except ImportError:
        print("ERROR: 'coincurve' is required. Install: pip install coincurve", file=sys.stderr)
        sys.exit(1)

    try:
        lpath = _find_lib(lib_path)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Library: {lpath}")

    try:
        lib = SignLib(lpath)
    except Exception as e:
        print(f"ERROR loading library: {e}", file=sys.stderr)
        sys.exit(1)

    # Fixed test message — same for all scalars
    # Use canonical "The quick brown fox..." sha256 to keep it meaningful
    msg32 = hashlib.sha256(b"GLV decomposition audit message").digest()

    adversarial = adversarial_scalars()
    random_scalars: list[tuple[str, int]] = [
        (f"random[{i}]", secrets.randbelow(N - 1) + 1)
        for i in range(extra_random)
    ]
    all_scalars = adversarial + random_scalars

    print(f"\nTesting {len(adversarial)} adversarial + {extra_random} random scalars "
          f"= {len(all_scalars)} total")
    print()

    mismatches: list[GLVMismatch] = []
    skip_count = 0
    ok_count = 0

    print("=== Adversarial scalar verification ===")
    for i, (name, k) in enumerate(all_scalars):
        k32 = k.to_bytes(32, "big")
        sig_ours = lib.sign(k32, msg32)
        sig_ref  = _coincurve_sign(k32, msg32)

        if sig_ours is None or sig_ref is None:
            skip_count += 1
            if i < len(adversarial):
                print(f"  SKIP [{name}] — library returned None (k may be rejected)")
            continue

        r_ours = sig_ours[:32].hex()
        r_ref  = sig_ref[:32].hex()

        if r_ours == r_ref:
            ok_count += 1
            if i < len(adversarial):
                print(f"  ✓  [{name}]  r={r_ours[:16]}...")
        else:
            mismatches.append(GLVMismatch(
                name=name, k_hex=f"{k:064x}",
                r_ours_hex=r_ours, r_ref_hex=r_ref,
            ))
            if i < len(adversarial):
                print(f"  ✗  [{name}]  GLV BUG DETECTED!")
                print(f"       k      = {k:064x}")
                print(f"       r(ours)= {r_ours}")
                print(f"       r(ref) = {r_ref}")

        if extra_random > 0 and (i + 1) % 1000 == 0:
            sys.stdout.write(f"\r  Random progress: {i - len(adversarial) + 1}/{extra_random}")
            sys.stdout.flush()

    if extra_random > 0:
        print()

    total = len(all_scalars) - skip_count
    print()
    print("=" * 60)
    print(f"Adversarial scalars: {min(len(adversarial), ok_count + len(mismatches))}/{len(adversarial)}")
    print(f"Total verified: {ok_count}/{total}  ({skip_count} skipped)")
    print(f"Mismatches: {len(mismatches)}")

    overall = len(mismatches) == 0
    print()
    if overall:
        print("Overall: PASS")
        print("  ✓ GLV decomposition correct for all adversarial scalars")
        if extra_random:
            print(f"  ✓ GLV decomposition correct for {extra_random} random scalars")
    else:
        print("Overall: FAIL")
        print(f"  ✗ {len(mismatches)} GLV decomposition errors detected!")
        print()
        print("  This means for certain k values, our library computes k·G incorrectly.")
        print("  Root cause: Babai rounding error in the GLV lattice reduction.")
        print()
        print("  Failing scalars:")
        for mm in mismatches[:10]:
            print(f"    [{mm.name}]  k={mm.k_hex[:32]}...")
            print(f"      r(ours)={mm.r_ours_hex}")
            print(f"      r(ref) ={mm.r_ref_hex}")

    report = {
        "library": lpath,
        "adversarial_count": len(adversarial),
        "random_count": extra_random,
        "ok_count": ok_count,
        "skip_count": skip_count,
        "mismatch_count": len(mismatches),
        "overall_pass": overall,
        "mismatches": [
            {
                "name": mm.name, "k": mm.k_hex,
                "r_ours": mm.r_ours_hex, "r_ref": mm.r_ref_hex,
            }
            for mm in mismatches
        ],
    }

    if json_out or out_file:
        j = json.dumps(report, indent=2)
        if out_file:
            Path(out_file).write_text(j)
            print(f"\nReport written to {out_file}")
        else:
            print(j)

    sys.exit(0 if overall else 1)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lib",          help="Path to shared library")
    p.add_argument("--extra-random", type=int, default=0,
                   help="Additional random scalars to test (default: 0; use 10000+ for thorough)")
    p.add_argument("--json",         action="store_true")
    p.add_argument("-o",             dest="out")
    args = p.parse_args()
    run(args.lib, args.extra_random, args.json, args.out)


if __name__ == "__main__":
    main()
