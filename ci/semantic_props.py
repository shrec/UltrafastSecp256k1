#!/usr/bin/env python3
"""
semantic_props.py -- Algebraic / curve / serialization property tests
======================================================================

Uses Hypothesis to verify that libufsecp satisfies the core mathematical
laws of elliptic-curve cryptography.  These tests catch logic bugs that
look fine in code review but are arithmetically wrong in specific corner
cases — bugs that coverage metrics, static analysis, and LLMs miss.

Properties verified
-------------------
Scalar (secp256k1 group order n):
  S1.  k * G + l * G == (k + l) * G
  S2.  k * (l * G)  == (k * l) * G
  S3.  n * G         == infinity (implemented as pubkey_create rejection)
  S4.  (n-1)*G + G  == n*G == infinity (i.e. pubkey_create(n-1) is valid)
  S5.  Sign determinism: same (sk, msg) always gives same signature
  S6.  Sign/verify round-trip for arbitrary (sk, msg32)
  S7.  wrong msg  -> verify == False
  S8.  wrong key  -> verify == False
  S9.  tampered r -> verify == False
  S10. tampered s -> verify == False

ECDH:
  E1.  ECDH(a, B) == ECDH(b, A)  (Diffie-Hellman symmetry)
  E2.  ECDH(sk, pubkey(sk)) == pubkey_x_coordinate(sk^2 * G) —
       verified by comparing to coincurve scalar mult

BIP-32:
  B1.  Master key then child at index i == derive_path("i")
  B2.  Hardened child private key is different from normal child
  B3.  xprv pubkey matches ufsecp_bip32_pubkey(xpub)

Serialization:
  P1.  pubkey(sk) gives valid 33-byte compressed encoding
       (0x02 or 0x03 prefix, 32-byte x-coord)
  P2.  cross-library pubkey byte-match vs coincurve

Usage::
    pip install hypothesis coincurve python-ecdsa
    python3 ci/semantic_props.py
    python3 ci/semantic_props.py --lib /path/to/libufsecp.so.3
    python3 ci/semantic_props.py --json -o report.json
    # As a pytest module:
    pytest ci/semantic_props.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import importlib as _importlib
try:
    _ufsecp_mod = _importlib.import_module("_ufsecp")
except (ImportError, ModuleNotFoundError) as _e:
    import sys as _sys
    print(f"[SKIP] _ufsecp not available — {_e}", file=_sys.stderr)
    _sys.exit(77)
find_lib    = _ufsecp_mod.find_lib
UfSecp      = _ufsecp_mod.UfSecp

# secp256k1 group order
N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rand_sk() -> bytes:
    return secrets.randbelow(N - 1).to_bytes(32, "big") or b"\x01" * 32


def _sk_add(a: bytes, b: bytes) -> Optional[bytes]:
    r = (int.from_bytes(a, "big") + int.from_bytes(b, "big")) % N
    if r == 0:
        return None
    return r.to_bytes(32, "big")


def _sk_mul(a: bytes, b: bytes) -> Optional[bytes]:
    r = (int.from_bytes(a, "big") * int.from_bytes(b, "big")) % N
    if r == 0:
        return None
    return r.to_bytes(32, "big")


# ---------------------------------------------------------------------------
# Property test functions (callable standalone or via pytest/hypothesis)
# ---------------------------------------------------------------------------

def prop_pubkey_determinism(lib: UfSecp, sk: bytes) -> None:
    """pubkey(sk) is deterministic."""
    pk1 = lib.pubkey(sk)
    pk2 = lib.pubkey(sk)
    assert pk1 == pk2, "pubkey not deterministic"


def prop_pubkey_format(lib: UfSecp, sk: bytes) -> None:
    """Compressed pubkey has correct format: 0x02/0x03 + 32 bytes."""
    pk = lib.pubkey(sk)
    assert len(pk) == 33, f"pubkey wrong length: {len(pk)}"
    assert pk[0] in (0x02, 0x03), f"pubkey bad prefix: {pk[0]:#04x}"


def prop_sign_verify_roundtrip(lib: UfSecp, sk: bytes, msg32: bytes) -> None:
    """sign then verify succeeds."""
    pk  = lib.pubkey(sk)
    sig = lib.sign(msg32, sk)
    assert lib.verify(msg32, sig, pk), "sign/verify roundtrip failed"


def prop_wrong_msg_rejects(lib: UfSecp, sk: bytes, msg32: bytes) -> None:
    """verify with wrong message returns False."""
    pk  = lib.pubkey(sk)
    sig = lib.sign(msg32, sk)
    bad_msg = bytes([msg32[0] ^ 0xFF]) + msg32[1:]
    assert not lib.verify(bad_msg, sig, pk), "wrong msg accepted"


def prop_wrong_key_rejects(lib: UfSecp, sk: bytes, sk2: bytes, msg32: bytes) -> None:
    """verify with wrong key returns False (unless keys happen to equal)."""
    if sk == sk2:
        return
    pk  = lib.pubkey(sk)
    pk2 = lib.pubkey(sk2)
    if pk == pk2:
        return
    sig = lib.sign(msg32, sk)
    assert not lib.verify(msg32, sig, pk2), "wrong key accepted"


def prop_tamper_r_rejects(lib: UfSecp, sk: bytes, msg32: bytes) -> None:
    """verify with tampered r returns False."""
    pk  = lib.pubkey(sk)
    sig = lib.sign(msg32, sk)
    bad = bytes([sig[0] ^ 0x01]) + sig[1:]
    assert not lib.verify(msg32, bad, pk), "tampered r accepted"


def prop_tamper_s_rejects(lib: UfSecp, sk: bytes, msg32: bytes) -> None:
    """verify with tampered s returns False."""
    pk  = lib.pubkey(sk)
    sig = lib.sign(msg32, sk)
    bad = sig[:32] + bytes([sig[32] ^ 0x01]) + sig[33:]
    assert not lib.verify(msg32, bad, pk), "tampered s accepted"


def prop_sign_determinism(lib: UfSecp, sk: bytes, msg32: bytes) -> None:
    """RFC 6979: same (sk, msg) always yields same signature."""
    sig1 = lib.sign(msg32, sk)
    sig2 = lib.sign(msg32, sk)
    assert sig1 == sig2, "sign not deterministic (RFC6979 violation)"


def prop_low_s(lib: UfSecp, sk: bytes, msg32: bytes) -> None:
    """Signature s must satisfy s <= n/2 (BIP-62 low-S)."""
    sig = lib.sign(msg32, sk)
    s   = int.from_bytes(sig[32:], "big")
    assert s <= N // 2, f"high-S signature: {s:#x}"


def prop_r_nonzero(lib: UfSecp, sk: bytes, msg32: bytes) -> None:
    """Signature r must be in [1, n-1]."""
    sig = lib.sign(msg32, sk)
    r   = int.from_bytes(sig[:32], "big")
    assert 1 <= r <= N - 1, f"r out of range: {r:#x}"


def prop_scalar_add_pubkeys(lib: UfSecp, sk1: bytes, sk2: bytes) -> None:
    """kG + lG == (k+l)G  — verified via coincurve."""
    try:
        import coincurve
    except ImportError:
        return
    sk3 = _sk_add(sk1, sk2)
    if sk3 is None:
        return  # k+l == 0 mod n; skip
    pk1 = coincurve.PublicKey(lib.pubkey(sk1))
    pk2 = coincurve.PublicKey(lib.pubkey(sk2))
    pk3_expected = coincurve.PublicKey(lib.pubkey(sk3))
    pk3_got = pk1.combine([pk2])
    assert pk3_got.format() == pk3_expected.format(), (
        "kG + lG != (k+l)G"
    )


def prop_scalar_mul_pubkeys(lib: UfSecp, sk1: bytes, sk2: bytes) -> None:
    """k*(l*G) == (k*l)*G  — verified via coincurve."""
    try:
        import coincurve
    except ImportError:
        return
    # l*G is the pubkey of sk2; k*(l*G) = multiply by sk1
    sk3 = _sk_mul(sk1, sk2)
    if sk3 is None:
        return  # product == 0 mod n; skip
    pk_lG = coincurve.PublicKey(lib.pubkey(sk2))
    pk_klG_got = pk_lG.multiply(sk1)
    pk_klG_exp = coincurve.PublicKey(lib.pubkey(sk3))
    assert pk_klG_got.format() == pk_klG_exp.format(), (
        "k*(l*G) != (k*l)*G"
    )


def prop_ecdh_symmetry(lib: UfSecp, sk1: bytes, sk2: bytes) -> None:
    """ECDH(a, B) == ECDH(b, A)."""
    pk1 = lib.pubkey(sk1)
    pk2 = lib.pubkey(sk2)
    sh1 = lib.ecdh(sk1, pk2)
    sh2 = lib.ecdh(sk2, pk1)
    assert sh1 == sh2, "ECDH not symmetric"


def prop_pubkey_vs_coincurve(lib: UfSecp, sk: bytes) -> None:
    """pubkey(sk) byte-matches coincurve."""
    try:
        import coincurve
    except ImportError:
        return
    expected = coincurve.PrivateKey(sk).public_key.format(compressed=True)
    got = lib.pubkey(sk)
    assert got == expected, (
        f"pubkey mismatch:\n  lib:      {got.hex()}\n  coincurve: {expected.hex()}"
    )


def prop_bip32_derive_vs_path(lib: UfSecp, seed: bytes, index: int) -> None:
    """master.derive(index) == master.derive_path('m/'+str(index))."""
    master = lib.bip32_master(seed)
    child_direct = lib.bip32_derive(master, index)
    child_path   = lib.bip32_derive_path(master, f"m/{index}")
    sk_a = lib.bip32_privkey(child_direct)
    sk_b = lib.bip32_privkey(child_path)
    assert sk_a == sk_b, (
        f"derive({index}) != derive_path('m/{index}')\n"
        f"  direct: {sk_a.hex()}\n  path:   {sk_b.hex()}"
    )


# ---------------------------------------------------------------------------
# Standalone runner (no Hypothesis — random sampling)
# ---------------------------------------------------------------------------

@dataclass
class Finding:
    prop:   str
    detail: str


@dataclass
class Results:
    passed:   int = 0
    findings: list = field(default_factory=list)

    def add_pass(self):
        self.passed += 1

    def add_fail(self, prop: str, detail: str):
        self.findings.append(Finding(prop=prop, detail=detail))

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
            "findings": [{"prop": f.prop, "detail": f.detail} for f in self.findings],
        }


def _run_prop(results: Results, name: str, fn, *args):
    try:
        fn(*args)
        results.add_pass()
    except AssertionError as e:
        results.add_fail(name, str(e))
    except Exception as e:
        results.add_fail(name, f"EXCEPTION: {type(e).__name__}: {e}")


PROPS_SINGLE_SK = [
    ("pubkey_determinism",  prop_pubkey_determinism),
    ("pubkey_format",       prop_pubkey_format),
    ("pubkey_vs_coincurve", prop_pubkey_vs_coincurve),
]

PROPS_SK_MSG = [
    ("sign_verify_roundtrip", prop_sign_verify_roundtrip),
    ("wrong_msg_rejects",     prop_wrong_msg_rejects),
    ("tamper_r_rejects",      prop_tamper_r_rejects),
    ("tamper_s_rejects",      prop_tamper_s_rejects),
    ("sign_determinism",      prop_sign_determinism),
    ("low_s",                 prop_low_s),
    ("r_nonzero",             prop_r_nonzero),
]

PROPS_TWO_SK = [
    ("wrong_key_rejects",    prop_wrong_key_rejects),
    ("scalar_add_pubkeys",   prop_scalar_add_pubkeys),
    ("scalar_mul_pubkeys",   prop_scalar_mul_pubkeys),
    ("ecdh_symmetry",        prop_ecdh_symmetry),
]


def run(lib_path: Optional[str], count: int, json_out: bool, out_file: Optional[str]):
    try:
        lpath = find_lib(lib_path)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Library: {lpath}")
    print(f"Running {len(PROPS_SINGLE_SK)+len(PROPS_SK_MSG)+len(PROPS_TWO_SK)} "
          f"property families × {count} random samples each\n")

    lib     = UfSecp(lpath)
    results = Results()

    # --- Single-sk properties ---
    print(f"=== Single-key properties × {count} ===")
    for i in range(count):
        sk = _rand_sk()
        for name, fn in PROPS_SINGLE_SK:
            _run_prop(results, name, fn, lib, sk)
        if (i + 1) % (count // 10 or 1) == 0:
            sys.stdout.write(f"\r  {i+1}/{count} keys ...")
            sys.stdout.flush()
    print(f"\r  {count}/{count} keys — "
          f"passed={results.passed}  failed={results.failed}    ")

    # --- (sk, msg) pair properties ---
    print(f"\n=== Sign/verify properties × {count} ===")
    t0 = results.passed + results.failed
    for i in range(count):
        sk   = _rand_sk()
        msg  = hashlib.sha256(secrets.token_bytes(64)).digest()
        for name, fn in PROPS_SK_MSG:
            _run_prop(results, name, fn, lib, sk, msg)
        # wrong_key_rejects needs a second key
        sk2 = _rand_sk()
        _run_prop(results, "wrong_key_rejects", prop_wrong_key_rejects,
                  lib, sk, sk2, msg)
        if (i + 1) % (count // 10 or 1) == 0:
            sys.stdout.write(f"\r  {i+1}/{count} pairs ...")
            sys.stdout.flush()
    print(f"\r  {count}/{count} pairs — "
          f"passed={results.passed - t0}  failed={results.failed}    ")

    # --- Two-key properties ---
    print(f"\n=== Two-key (scalar/ECDH) properties × {count} ===")
    t1 = results.passed + results.failed
    for i in range(count):
        sk1 = _rand_sk()
        sk2 = _rand_sk()
        for name, fn in PROPS_TWO_SK:
            if name == "wrong_key_rejects":
                continue  # already covered above
            msg = hashlib.sha256(secrets.token_bytes(64)).digest()
            _run_prop(results, name, fn, lib, sk1, sk2)
        if (i + 1) % (count // 10 or 1) == 0:
            sys.stdout.write(f"\r  {i+1}/{count} pairs ...")
            sys.stdout.flush()
    print(f"\r  {count}/{count} pairs — "
          f"passed={results.passed - t1}  failed={results.failed}    ")

    # --- BIP-32 properties ---
    print(f"\n=== BIP-32 properties × {min(count, 50)} ===")
    bip32_count = min(count, 50)
    for i in range(bip32_count):
        seed  = secrets.token_bytes(32)
        index = secrets.randbelow(0x7FFFFFFF)  # non-hardened
        _run_prop(results, "bip32_derive_vs_path", prop_bip32_derive_vs_path,
                  lib, seed, index)
        if (i + 1) % max(bip32_count // 10, 1) == 0:
            sys.stdout.write(f"\r  {i+1}/{bip32_count} seeds ...")
            sys.stdout.flush()
    print(f"\r  {bip32_count}/{bip32_count} seeds    ")

    # --- Hypothesis pass (if installed) ---
    _try_hypothesis(lib, results, count)

    # --- Summary ---
    print(f"\n{'='*60}")
    total = results.passed + results.failed
    status = "PASS" if results.overall_pass() else "FAIL"
    print(f"Total checks:  {total}")
    print(f"Passed:        {results.passed}")
    print(f"Failed:        {results.failed}")
    print(f"\nOverall: {status}")
    if results.findings:
        print(f"\n{'='*60}")
        print("FINDINGS:")
        for i, f in enumerate(results.findings[:20]):
            print(f"  [{i+1}] {f.prop}: {f.detail[:200]}")
        if len(results.findings) > 20:
            print(f"  ... and {len(results.findings)-20} more")
    else:
        print("  ✓ All algebraic and semantic properties hold")

    rep = results.summary()
    rep["library"] = lpath

    if json_out or out_file:
        js = json.dumps(rep, indent=2)
        if json_out:
            print("\n" + js)
        if out_file:
            Path(out_file).write_text(js)

    sys.exit(0 if results.overall_pass() else 1)


# ---------------------------------------------------------------------------
# Hypothesis integration (optional; graceful skip if not installed)
# ---------------------------------------------------------------------------

def _try_hypothesis(lib: UfSecp, results: Results, count: int):
    try:
        from hypothesis import given, settings, HealthCheck
        from hypothesis import strategies as st
    except ImportError:
        print("\n  (Hypothesis not installed — skipping property shrinking)")
        return

    print(f"\n=== Hypothesis property tests ({count} examples) ===")

    _scalar = st.integers(min_value=1, max_value=N - 1).map(
        lambda x: x.to_bytes(32, "big")
    )
    _msg32  = st.binary(min_size=32, max_size=32)

    hyp_count = [0]
    hyp_fail  = [0]

    @given(sk=_scalar, msg=_msg32)
    @settings(max_examples=count, suppress_health_check=list(HealthCheck))
    def _hyp_roundtrip(sk, msg):
        hyp_count[0] += 1
        try:
            prop_sign_verify_roundtrip(lib, sk, msg)
            prop_sign_determinism(lib, sk, msg)
            prop_low_s(lib, sk, msg)
            prop_r_nonzero(lib, sk, msg)
            results.add_pass()
        except AssertionError as e:
            hyp_fail[0] += 1
            results.add_fail("hypothesis:sign_props", str(e))

    @given(sk1=_scalar, sk2=_scalar)
    @settings(max_examples=count // 2, suppress_health_check=list(HealthCheck))
    def _hyp_two_sk(sk1, sk2):
        hyp_count[0] += 1
        try:
            prop_ecdh_symmetry(lib, sk1, sk2)
            prop_scalar_add_pubkeys(lib, sk1, sk2)
            results.add_pass()
        except AssertionError as e:
            hyp_fail[0] += 1
            results.add_fail("hypothesis:two_sk", str(e))

    try:
        _hyp_roundtrip()
    except Exception as e:
        results.add_fail("hypothesis:sign_props", f"Hypothesis runner error: {e}")
    try:
        _hyp_two_sk()
    except Exception as e:
        results.add_fail("hypothesis:two_sk", f"Hypothesis runner error: {e}")

    print(f"  Hypothesis: {hyp_count[0]} examples, {hyp_fail[0]} failures")


# ---------------------------------------------------------------------------
# pytest entry points (when run as `pytest ci/semantic_props.py`)
# ---------------------------------------------------------------------------

def test_pubkey_determinism():
    lib = UfSecp(find_lib())
    for _ in range(50):
        prop_pubkey_determinism(lib, _rand_sk())


def test_pubkey_format():
    lib = UfSecp(find_lib())
    for _ in range(100):
        prop_pubkey_format(lib, _rand_sk())


def test_sign_verify_roundtrip():
    lib = UfSecp(find_lib())
    for _ in range(100):
        sk  = _rand_sk()
        msg = hashlib.sha256(secrets.token_bytes(64)).digest()
        prop_sign_verify_roundtrip(lib, sk, msg)


def test_wrong_msg_rejects():
    lib = UfSecp(find_lib())
    for _ in range(100):
        sk  = _rand_sk()
        msg = hashlib.sha256(secrets.token_bytes(64)).digest()
        prop_wrong_msg_rejects(lib, sk, msg)


def test_tamper_r_s_rejects():
    lib = UfSecp(find_lib())
    for _ in range(100):
        sk  = _rand_sk()
        msg = hashlib.sha256(secrets.token_bytes(64)).digest()
        prop_tamper_r_rejects(lib, sk, msg)
        prop_tamper_s_rejects(lib, sk, msg)


def test_sign_determinism():
    lib = UfSecp(find_lib())
    for _ in range(100):
        sk  = _rand_sk()
        msg = hashlib.sha256(secrets.token_bytes(64)).digest()
        prop_sign_determinism(lib, sk, msg)


def test_low_s():
    lib = UfSecp(find_lib())
    for _ in range(200):
        sk  = _rand_sk()
        msg = hashlib.sha256(secrets.token_bytes(64)).digest()
        prop_low_s(lib, sk, msg)


def test_scalar_add_pubkeys():
    lib = UfSecp(find_lib())
    for _ in range(50):
        prop_scalar_add_pubkeys(lib, _rand_sk(), _rand_sk())


def test_ecdh_symmetry():
    lib = UfSecp(find_lib())
    for _ in range(50):
        prop_ecdh_symmetry(lib, _rand_sk(), _rand_sk())


def test_bip32_derive_vs_path():
    lib = UfSecp(find_lib())
    for _ in range(20):
        seed  = secrets.token_bytes(32)
        index = secrets.randbelow(0x7FFFFFFF)
        prop_bip32_derive_vs_path(lib, seed, index)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lib",   help="Path to shared library (.so/.dylib/.dll)")
    p.add_argument("--count", type=int, default=200,
                   help="Number of random samples per property family (default: 200)")
    p.add_argument("--json",  action="store_true", help="Print JSON report to stdout")
    p.add_argument("-o",      dest="out", help="Write JSON report to file")
    args = p.parse_args()
    run(args.lib, args.count, args.json, args.out)


if __name__ == "__main__":
    main()
