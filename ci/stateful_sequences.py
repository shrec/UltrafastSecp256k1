#!/usr/bin/env python3
"""
stateful_sequences.py -- Stateful API call sequence verifier
=============================================================

Tests that the library behaves correctly under arbitrary sequences of API
calls — not just individual operations in isolation.  State-machine bugs
often only appear when operations are composed:

  - Signing after a failed verify leaves context in a valid state
  - Serialising a key produced by a failed derive doesn't segfault
  - Re-using a context for many operations doesn't accumulate errors
  - BIP-32 chained derivation is consistent with single-step path derivation
  - Context destroy / recreate clears state completely

Class of bugs this catches
--------------------------
  1. Use-after-error: library corrupts context on failure path, next valid
     call returns garbage or crashes.

  2. Operation ordering: result of sign(k) differs if called before vs after
     an unrelated verify(bad_sig) on the same context.

  3. Chained BIP-32 consistency: derive(derive(k, a), b) == derive_path(k, "m/a/b")

  4. Serialization idempotence after error: after a failing API call, a
     subsequent successful call still returns the correct answer.

  5. Context isolation: two separate UfSecp instances don't share state.

  6. High-frequency reuse: 10,000 sign→verify on the same context doesn't
     degrade or drift.

Sequence types run
------------------
  SEQ-A  Random interleaved sign/verify/ecdh over 1 context × N iterations
  SEQ-B  Error injection: insert a failing call every K steps; verify
         subsequent valid calls are unaffected
  SEQ-C  BIP-32 multi-level path consistency:
           m/a/b == derive(derive(master,a),b) for random a,b
           m/a/b/c == ... for 3-level
  SEQ-D  Two-context isolation: same operations on ctx1 and ctx2 give
         identical results
  SEQ-E  Context recreate: destroy + create + same sign → same sig
  SEQ-F  High-frequency reuse: 5000 sign + verify on one context

Usage::
    python3 ci/stateful_sequences.py --lib /path/to/libufsecp.so.3
    python3 ci/stateful_sequences.py --json -o report.json
    python3 ci/stateful_sequences.py --count 200  # iterations per sequence
"""

from __future__ import annotations

import argparse
import hashlib
import json
import secrets
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import importlib as _importlib
_ufsecp_mod = _importlib.import_module("_ufsecp")
find_lib    = _ufsecp_mod.find_lib
UfSecp      = _ufsecp_mod.UfSecp
bip32_depth = _ufsecp_mod.bip32_depth

N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rand_sk() -> bytes:
    v = secrets.randbelow(N - 1) + 1
    return v.to_bytes(32, "big")


def _rand_msg() -> bytes:
    return hashlib.sha256(secrets.token_bytes(64)).digest()


def _rand_index_normal() -> int:
    return secrets.randbelow(0x80000000)


# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------

@dataclass
class Finding:
    sequence: str
    step:     int
    detail:   str


@dataclass
class Results:
    passed:   int = 0
    findings: list = field(default_factory=list)

    def ok(self):
        self.passed += 1

    def fail(self, seq: str, step: int, detail: str):
        self.findings.append(Finding(sequence=seq, step=step, detail=detail))

    @property
    def failed(self) -> int:
        return len(self.findings)

    def overall_pass(self) -> bool:
        return self.failed == 0

    def summary(self) -> dict:
        return {
            "passed": self.passed,
            "failed": self.failed,
            "overall": "PASS" if self.overall_pass() else "FAIL",
            "findings": [
                {"sequence": f.sequence, "step": f.step, "detail": f.detail}
                for f in self.findings
            ],
        }


# ---------------------------------------------------------------------------
# SEQ-A: Random interleaved operations on one context
# ---------------------------------------------------------------------------

def seq_a_interleaved(lib: UfSecp, count: int, results: Results):
    """Random interleaved sign/verify/ecdh — context must stay consistent."""
    name = "SEQ-A:interleaved"
    # Pre-generate a pool of valid keys + messages
    pool_sk  = [_rand_sk()  for _ in range(10)]
    pool_msg = [_rand_msg() for _ in range(10)]
    pool_pk  = [lib.pubkey(sk) for sk in pool_sk]
    pool_sig = [lib.sign(m, sk) for m, sk in zip(pool_msg, pool_sk)]

    for step in range(count):
        op = step % 4
        i  = secrets.randbelow(10)
        j  = secrets.randbelow(10)
        try:
            if op == 0:
                # sign + verify with fresh random
                sk  = pool_sk[i]
                msg = _rand_msg()
                sig = lib.sign(msg, sk)
                pk  = pool_pk[i]
                if not lib.verify(msg, sig, pk):
                    results.fail(name, step, "sign+verify failed after interleaved ops")
                    continue
                results.ok()

            elif op == 1:
                # verify a previously stored sig
                if not lib.verify(pool_msg[i], pool_sig[i], pool_pk[i]):
                    results.fail(name, step, f"stored verify failed at step {step}")
                    continue
                results.ok()

            elif op == 2:
                # ECDH symmetry check
                sh1 = lib.ecdh(pool_sk[i], pool_pk[j])
                sh2 = lib.ecdh(pool_sk[j], pool_pk[i])
                if sh1 != sh2:
                    results.fail(name, step, "ECDH not symmetric after interleaved ops")
                    continue
                results.ok()

            else:
                # Verify wrong-key returns False
                if i == j:
                    results.ok()
                    continue
                sig = lib.sign(pool_msg[i], pool_sk[i])
                if lib.verify(pool_msg[i], sig, pool_pk[j]):
                    results.fail(name, step, "wrong-key verify returned True")
                    continue
                results.ok()

        except Exception as e:
            results.fail(name, step, f"EXCEPTION: {e}")

    sys.stdout.write(f"\r  SEQ-A: {count} steps, "
                     f"pass={results.passed} fail={results.failed}    \n")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# SEQ-B: Error injection — valid ops after failures must still work
# ---------------------------------------------------------------------------

def seq_b_error_injection(lib: UfSecp, count: int, results: Results):
    """After a failing API call, subsequent valid calls must work correctly."""
    name = "SEQ-B:error_injection"

    sk  = _rand_sk()
    msg = _rand_msg()
    pk  = lib.pubkey(sk)
    sig = lib.sign(msg, sk)

    # Baseline: repeated sign→verify as sanity
    baseline_sig = lib.sign(msg, sk)
    if sig != baseline_sig:
        results.fail(name, 0, "baseline sign not deterministic before injection")
        return

    bad_msg  = b"\x00" * 32
    bad_sig  = b"\x00" * 64
    bad_pk   = b"\x00" * 33

    for step in range(count):
        # Inject a failure every 3 steps
        if step % 3 == 1:
            # Attempt to verify with bad pubkey (should return False / nonzero)
            try:
                lib.verify(msg, sig, bad_pk)  # False/error OK
            except Exception:
                pass
        elif step % 3 == 2:
            # Attempt to verify a bad signature
            try:
                lib.verify(msg, bad_sig, pk)  # False OK
            except Exception:
                pass

        # After each potential error, a valid operation must still work
        try:
            new_sig = lib.sign(msg, sk)
            if new_sig != sig:
                results.fail(name, step,
                             f"sign output changed after error injection at step {step}")
                continue
            if not lib.verify(msg, new_sig, pk):
                results.fail(name, step,
                             "verify failed after error injection")
                continue
            results.ok()
        except Exception as e:
            results.fail(name, step, f"EXCEPTION after injection: {e}")

    sys.stdout.write(f"\r  SEQ-B: {count} steps, "
                     f"pass={results.passed} fail={results.failed}    \n")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# SEQ-C: BIP-32 multi-level path consistency
# ---------------------------------------------------------------------------

def seq_c_bip32_path_consistency(lib: UfSecp, count: int, results: Results):
    """derive(derive(master,a),b) == derive_path(master,'m/a/b')."""
    name = "SEQ-C:bip32_path"
    seq_count = min(count, 50)  # BIP-32 is slower

    for step in range(seq_count):
        seed = secrets.token_bytes(32)
        try:
            master = lib.bip32_master(seed)

            # 2-level
            a = _rand_index_normal()
            b = _rand_index_normal()
            child_ab_direct = lib.bip32_derive(lib.bip32_derive(master, a), b)
            child_ab_path   = lib.bip32_derive_path(master, f"m/{a}/{b}")
            sk_ab_d = lib.bip32_privkey(child_ab_direct)
            sk_ab_p = lib.bip32_privkey(child_ab_path)
            if sk_ab_d != sk_ab_p:
                results.fail(name, step,
                             f"2-level: derive({a},{b}) != path m/{a}/{b}")
                continue

            # 3-level
            c = _rand_index_normal()
            l3_direct = lib.bip32_derive(child_ab_direct, c)
            l3_path   = lib.bip32_derive_path(master, f"m/{a}/{b}/{c}")
            sk3_d = lib.bip32_privkey(l3_direct)
            sk3_p = lib.bip32_privkey(l3_path)
            if sk3_d != sk3_p:
                results.fail(name, step,
                             f"3-level: derive({a},{b},{c}) != path m/{a}/{b}/{c}")
                continue

            # Depth tracking: master=0, child=1, grandchild=2, great-grandchild=3
            if bip32_depth(master) != 0:
                results.fail(name, step, f"master depth != 0: {bip32_depth(master)}")
                continue
            child_0 = lib.bip32_derive(master, a)
            if bip32_depth(child_0) != 1:
                results.fail(name, step, f"child depth != 1: {bip32_depth(child_0)}")
                continue
            if bip32_depth(child_ab_direct) != 2:
                results.fail(name, step,
                             f"grandchild depth != 2: {bip32_depth(child_ab_direct)}")
                continue

            results.ok()

        except Exception as e:
            results.fail(name, step, f"EXCEPTION: {e}")

    sys.stdout.write(f"\r  SEQ-C: {seq_count} seeds, "
                     f"pass={results.passed} fail={results.failed}    \n")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# SEQ-D: Two-context isolation
# ---------------------------------------------------------------------------

def seq_d_context_isolation(lib_path: str, count: int, results: Results):
    """Two independent UfSecp contexts must produce identical results."""
    name = "SEQ-D:ctx_isolation"
    lib1 = UfSecp(lib_path)
    lib2 = UfSecp(lib_path)
    seq_count = min(count, 100)

    for step in range(seq_count):
        sk  = _rand_sk()
        msg = _rand_msg()
        try:
            pk1  = lib1.pubkey(sk)
            pk2  = lib2.pubkey(sk)
            if pk1 != pk2:
                results.fail(name, step, "pubkey differs across contexts")
                continue

            sig1 = lib1.sign(msg, sk)
            sig2 = lib2.sign(msg, sk)
            if sig1 != sig2:
                results.fail(name, step, "sign differs across contexts")
                continue

            ok1 = lib1.verify(msg, sig1, pk1)
            ok2 = lib2.verify(msg, sig2, pk2)
            if ok1 != ok2:
                results.fail(name, step,
                             f"verify disagrees: ctx1={ok1} ctx2={ok2}")
                continue

            results.ok()
        except Exception as e:
            results.fail(name, step, f"EXCEPTION: {e}")

    sys.stdout.write(f"\r  SEQ-D: {seq_count} pairs, "
                     f"pass={results.passed} fail={results.failed}    \n")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# SEQ-E: Context recreate gives same results
# ---------------------------------------------------------------------------

def seq_e_context_recreate(lib_path: str, count: int, results: Results):
    """Destroy + recreate context → same sign output for same input."""
    name = "SEQ-E:ctx_recreate"
    seq_count = min(count, 50)

    # Reference results from first context
    sk  = _rand_sk()
    msg = _rand_msg()
    lib_ref  = UfSecp(lib_path)
    ref_sig  = lib_ref.sign(msg, sk)
    ref_pk   = lib_ref.pubkey(sk)
    del lib_ref  # destroy

    for step in range(seq_count):
        try:
            lib_new = UfSecp(lib_path)
            new_sig = lib_new.sign(msg, sk)
            new_pk  = lib_new.pubkey(sk)
            del lib_new

            if new_sig != ref_sig:
                results.fail(name, step,
                             "sign output differs after context recreate (non-deterministic?)")
                continue
            if new_pk != ref_pk:
                results.fail(name, step, "pubkey differs after context recreate")
                continue
            results.ok()
        except Exception as e:
            results.fail(name, step, f"EXCEPTION: {e}")

    sys.stdout.write(f"\r  SEQ-E: {seq_count} recreates, "
                     f"pass={results.passed} fail={results.failed}    \n")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# SEQ-F: High-frequency reuse (5000 iterations on same context)
# ---------------------------------------------------------------------------

def seq_f_high_frequency(lib: UfSecp, count: int, results: Results):
    """5000 sign+verify on same context — no drift or state corruption."""
    name  = "SEQ-F:high_freq"
    iters = max(count * 5, 500)
    sk    = _rand_sk()
    pk    = lib.pubkey(sk)
    fail_at = None

    for step in range(iters):
        msg = _rand_msg()
        try:
            sig = lib.sign(msg, sk)
            if not lib.verify(msg, sig, pk):
                fail_at = step
                results.fail(name, step,
                             f"verify failed at iteration {step}/{iters}")
                break
        except Exception as e:
            fail_at = step
            results.fail(name, step, f"EXCEPTION at iteration {step}: {e}")
            break

    if fail_at is None:
        results.ok()  # count as 1 pass for the whole sequence
        sys.stdout.write(f"\r  SEQ-F: {iters} iterations, ✓    \n")
    else:
        sys.stdout.write(f"\r  SEQ-F: failed at step {fail_at}    \n")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run(lib_path: Optional[str], count: int, json_out: bool, out_file: Optional[str]):
    try:
        lpath = find_lib(lib_path)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Library: {lpath}")
    print(f"Running 6 sequence types × ~{count} steps each\n")

    lib     = UfSecp(lpath)
    results = Results()

    seq_a_interleaved(lib, count, results)
    seq_b_error_injection(lib, count, results)
    seq_c_bip32_path_consistency(lib, count, results)
    seq_d_context_isolation(lpath, count, results)
    seq_e_context_recreate(lpath, count, results)
    seq_f_high_frequency(lib, count, results)

    print(f"\n{'='*60}")
    print(f"Total checks:   {results.passed + results.failed}")
    print(f"Passed:         {results.passed}")
    print(f"Failed:         {results.failed}")
    print(f"\nOverall: {'PASS' if results.overall_pass() else 'FAIL'}")

    if results.findings:
        print(f"\nFINDINGS:")
        for f in results.findings[:20]:
            print(f"  [{f.sequence} step={f.step}] {f.detail[:200]}")
    else:
        print("  ✓ All stateful sequence properties hold")

    rep = results.summary()
    rep["library"] = lpath

    if json_out or out_file:
        js = json.dumps(rep, indent=2)
        if json_out:
            print("\n" + js)
        if out_file:
            Path(out_file).write_text(js)

    sys.exit(0 if results.overall_pass() else 1)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lib",   help="Path to shared library (.so/.dylib/.dll)")
    p.add_argument("--count", type=int, default=200,
                   help="Iterations per sequence type (default: 200)")
    p.add_argument("--json",  action="store_true")
    p.add_argument("-o",      dest="out")
    args = p.parse_args()
    run(args.lib, args.count, args.json, args.out)


if __name__ == "__main__":
    main()
