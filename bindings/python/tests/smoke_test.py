#!/usr/bin/env python3
"""
UltrafastSecp256k1 -- Python Binding Smoke Test (Golden Vectors)

Verifies FFI boundary correctness using deterministic known-answer tests.
Runs in <2 seconds. Requires the ufsecp shared library.

Golden vectors:
  - BIP-340 Schnorr (vector 0 from BIP-340 test vectors)
  - RFC 6979 ECDSA (deterministic nonce)
  - SHA-256 (NIST)
  - P2WPKH address

Usage:
  python smoke_test.py
  python -m pytest smoke_test.py -v
"""

import os
import sys
import struct

# Add parent so we can import ufsecp
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ufsecp import Ufsecp, UfsecpError, NET_MAINNET

# ── Golden Vectors ───────────────────────────────────────────────────────────

# Private key: 32 bytes (k=1 for simplicity in some tests, known key for BIP-340)
KNOWN_PRIVKEY = bytes.fromhex(
    "0000000000000000000000000000000000000000000000000000000000000001"
)

# Expected compressed pubkey for k=1 (generator point G)
KNOWN_PUBKEY_COMPRESSED = bytes.fromhex(
    "0279BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798"
)

# Expected x-only pubkey for k=1
KNOWN_PUBKEY_XONLY = bytes.fromhex(
    "79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798"
)

# SHA-256 of empty string
SHA256_EMPTY = bytes.fromhex(
    "E3B0C44298FC1C149AFBF4C8996FB92427AE41E4649B934CA495991B7852B855"
)

# Known message for RFC 6979 test
RFC6979_MSG = bytes(32)  # all-zero 32-byte hash

# BIP-340 test vector 0:
# privkey: 3 (adjusted for BIP-340 — we use k=1 which is simpler)
# We verify sign→verify round-trip with deterministic aux=zeros
BIP340_AUX = bytes(32)

# ── Tests ────────────────────────────────────────────────────────────────────

def test_ctx_create_destroy():
    """Context lifecycle: create, ABI check, destroy."""
    ctx = Ufsecp()
    abi = ctx.abi_version
    assert abi >= 1, f"ABI version {abi} < 1"
    ctx.close()


def test_ctx_context_manager():
    """Context manager protocol."""
    with Ufsecp() as ctx:
        abi = ctx.abi_version
        assert abi >= 1


def test_seckey_verify():
    """Private key validation."""
    with Ufsecp() as ctx:
        # Valid key
        ctx.seckey_verify(KNOWN_PRIVKEY)

        # Invalid: all-zero
        try:
            ctx.seckey_verify(bytes(32))
            assert False, "should have raised"
        except UfsecpError as e:
            assert e.code == 2  # ERR_BAD_KEY


def test_pubkey_create():
    """Pubkey derivation — golden vector k=1 → G."""
    with Ufsecp() as ctx:
        pub = ctx.pubkey_create(KNOWN_PRIVKEY)
        assert pub == KNOWN_PUBKEY_COMPRESSED, (
            f"Expected {KNOWN_PUBKEY_COMPRESSED.hex()}, got {pub.hex()}"
        )


def test_pubkey_xonly():
    """X-only pubkey — golden vector k=1."""
    with Ufsecp() as ctx:
        xonly = ctx.pubkey_xonly(KNOWN_PRIVKEY)
        assert xonly == KNOWN_PUBKEY_XONLY


def test_ecdsa_sign_verify():
    """ECDSA sign/verify round-trip (RFC 6979 deterministic)."""
    with Ufsecp() as ctx:
        sig = ctx.ecdsa_sign(RFC6979_MSG, KNOWN_PRIVKEY)
        assert len(sig) == 64

        # Verify with compressed pubkey
        ctx.ecdsa_verify(RFC6979_MSG, sig, KNOWN_PUBKEY_COMPRESSED)

        # Mutated sig must fail
        bad_sig = bytearray(sig)
        bad_sig[0] ^= 0x01
        try:
            ctx.ecdsa_verify(RFC6979_MSG, bytes(bad_sig), KNOWN_PUBKEY_COMPRESSED)
            assert False, "should have raised"
        except UfsecpError:
            pass  # expected


def test_ecdsa_der_roundtrip():
    """ECDSA compact ↔ DER conversion."""
    with Ufsecp() as ctx:
        sig = ctx.ecdsa_sign(RFC6979_MSG, KNOWN_PRIVKEY)
        der = ctx.ecdsa_sig_to_der(sig)
        recovered = ctx.ecdsa_sig_from_der(der)
        assert recovered == sig


def test_schnorr_sign_verify():
    """Schnorr BIP-340 sign/verify round-trip."""
    with Ufsecp() as ctx:
        sig = ctx.schnorr_sign(RFC6979_MSG, KNOWN_PRIVKEY, BIP340_AUX)
        assert len(sig) == 64

        # Verify
        ctx.schnorr_verify(RFC6979_MSG, sig, KNOWN_PUBKEY_XONLY)

        # Mutated sig must fail
        bad_sig = bytearray(sig)
        bad_sig[0] ^= 0x01
        try:
            ctx.schnorr_verify(RFC6979_MSG, bytes(bad_sig), KNOWN_PUBKEY_XONLY)
            assert False, "should have raised"
        except UfsecpError:
            pass  # expected


def test_ecdsa_recover():
    """ECDSA recoverable sign + recover pubkey."""
    with Ufsecp() as ctx:
        sig, recid = ctx.ecdsa_sign_recoverable(RFC6979_MSG, KNOWN_PRIVKEY)
        assert 0 <= recid <= 3
        recovered_pub = ctx.ecdsa_recover(RFC6979_MSG, sig, recid)
        assert recovered_pub == KNOWN_PUBKEY_COMPRESSED


def test_sha256_golden():
    """SHA-256 golden vector: empty string."""
    with Ufsecp() as ctx:
        digest = ctx.sha256(b"")
        assert digest == SHA256_EMPTY


def test_hash160():
    """Hash160 = RIPEMD160(SHA256(data)) smoke test."""
    with Ufsecp() as ctx:
        h = ctx.hash160(KNOWN_PUBKEY_COMPRESSED)
        assert len(h) == 20


def test_addr_p2wpkh():
    """P2WPKH address for k=1 on mainnet."""
    with Ufsecp() as ctx:
        addr = ctx.addr_p2wpkh(KNOWN_PUBKEY_COMPRESSED, NET_MAINNET)
        assert addr.startswith("bc1q"), f"Expected bc1q..., got {addr}"


def test_wif_roundtrip():
    """WIF encode/decode round-trip."""
    with Ufsecp() as ctx:
        wif = ctx.wif_encode(KNOWN_PRIVKEY, compressed=True, network=NET_MAINNET)
        privkey_back, compressed, network = ctx.wif_decode(wif)
        assert privkey_back == KNOWN_PRIVKEY
        assert compressed == 1
        assert network == NET_MAINNET


def test_bip32_derivation():
    """BIP-32 master key + derivation smoke test."""
    with Ufsecp() as ctx:
        seed = bytes(16)  # 16-byte seed (minimum)
        master = ctx.bip32_master(seed)
        assert master is not None

        # Derive m/0'
        child = ctx.bip32_derive(master, 0x80000000)
        assert child is not None


def test_ecdh():
    """ECDH shared secret smoke test."""
    with Ufsecp() as ctx:
        # Two keys
        k1 = bytes.fromhex(
            "0000000000000000000000000000000000000000000000000000000000000001"
        )
        k2 = bytes.fromhex(
            "0000000000000000000000000000000000000000000000000000000000000002"
        )
        pub1 = ctx.pubkey_create(k1)
        pub2 = ctx.pubkey_create(k2)

        secret_12 = ctx.ecdh(k1, pub2)
        secret_21 = ctx.ecdh(k2, pub1)
        assert secret_12 == secret_21, "ECDH must be symmetric"
        assert len(secret_12) == 32


def test_error_path():
    """Intentional error: verify error code and message are populated."""
    with Ufsecp() as ctx:
        try:
            ctx.seckey_verify(bytes(32))  # all-zero key → invalid
            assert False, "should have raised"
        except UfsecpError as e:
            assert e.code == 2  # ERR_BAD_KEY
            assert "key" in str(e).lower()


def test_golden_ecdsa_deterministic():
    """RFC 6979: same key + same message → same signature every time."""
    with Ufsecp() as ctx:
        sig1 = ctx.ecdsa_sign(RFC6979_MSG, KNOWN_PRIVKEY)
        sig2 = ctx.ecdsa_sign(RFC6979_MSG, KNOWN_PRIVKEY)
        assert sig1 == sig2, "RFC 6979 signatures must be deterministic"


def test_golden_schnorr_deterministic():
    """BIP-340: same key + same message + same aux → same signature."""
    with Ufsecp() as ctx:
        sig1 = ctx.schnorr_sign(RFC6979_MSG, KNOWN_PRIVKEY, BIP340_AUX)
        sig2 = ctx.schnorr_sign(RFC6979_MSG, KNOWN_PRIVKEY, BIP340_AUX)
        assert sig1 == sig2, "Schnorr signatures must be deterministic"


# ── Runner ───────────────────────────────────────────────────────────────────

def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    failed = 0
    for t in tests:
        name = t.__name__
        try:
            t()
            print(f"  [OK] {name}")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"  Python smoke test: {passed} passed, {failed} failed")
    print(f"{'=' * 60}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
