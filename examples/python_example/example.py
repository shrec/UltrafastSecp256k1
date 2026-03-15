#!/usr/bin/env python3
"""
UltrafastSecp256k1 -- Python Example (CPU + GPU)

Demonstrates the full Python ctypes binding: key ops, ECDSA, Schnorr,
ECDH, hashing, Bitcoin addresses, BIP-32, Taproot, and GPU batch ops.

Usage:
    UFSECP_LIB=../../build-linux/include/ufsecp/libufsecp.so python3 example.py

    # Or if LD_LIBRARY_PATH is set:
    LD_LIBRARY_PATH=../../build-linux/include/ufsecp python3 example.py
"""

import ctypes
import os
import sys
from ctypes import (
    POINTER, byref, c_char_p, c_int, c_size_t, c_uint8, c_uint32, c_void_p,
)

# Add the bindings directory to the path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BINDINGS_DIR = os.path.join(SCRIPT_DIR, '..', '..', 'bindings', 'python')
sys.path.insert(0, BINDINGS_DIR)

# Point to the shared library
if 'UFSECP_LIB' not in os.environ:
    candidate = os.path.join(SCRIPT_DIR, '..', '..', 'build-linux', 'include', 'ufsecp', 'libufsecp.so')
    if os.path.exists(candidate):
        os.environ['UFSECP_LIB'] = os.path.abspath(candidate)

from ufsecp import Ufsecp, NET_MAINNET

# ── Helper ─────────────────────────────────────────────────────────────────

def hexs(data: bytes) -> str:
    return data.hex()

def section(num: int, title: str):
    print(f"\n[{num}] {title}")

# ── CPU Examples ──────────────────────────────────────────────────────────

def demo_cpu():
    print("=== CPU Operations ===")

    with Ufsecp() as ctx:
        privkey = bytes(31) + b'\x01'
        privkey2 = bytes(31) + b'\x02'

        # 1. Key Generation
        section(1, "Key Generation")
        pub = ctx.pubkey_create(privkey)
        pub_uncompressed = ctx.pubkey_create_uncompressed(privkey)
        xonly = ctx.pubkey_xonly(privkey)
        print(f"  Private key:        {hexs(privkey)}")
        print(f"  Compressed (33B):   {hexs(pub)}")
        print(f"  Uncompressed (65B): {hexs(pub_uncompressed)}")
        print(f"  X-only (32B):       {hexs(xonly)}")

        # 2. ECDSA
        section(2, "ECDSA Sign / Verify (RFC 6979)")
        msg = ctx.sha256(b"Hello UltrafastSecp256k1!")
        print(f"  Message hash:       {hexs(msg)}")

        sig = ctx.ecdsa_sign(msg, privkey)
        print(f"  ECDSA signature:    {hexs(sig)}")

        ok = ctx.ecdsa_verify(msg, sig, pub)
        print(f"  Verify:             {'VALID' if ok else 'INVALID'}")

        # DER encoding
        der = ctx.ecdsa_sig_to_der(sig)
        print(f"  DER length:         {len(der)} bytes")
        sig_back = ctx.ecdsa_sig_from_der(der)
        print(f"  DER roundtrip:      {'match' if sig == sig_back else 'MISMATCH'}")

        # Recovery
        rsig = ctx.ecdsa_sign_recoverable(msg, privkey)
        recovered = ctx.ecdsa_recover(msg, rsig.signature, rsig.recovery_id)
        print(f"  Recovery:           recid={rsig.recovery_id}, match={'YES' if recovered == pub else 'NO'}")

        # 3. Schnorr
        section(3, "Schnorr Sign / Verify (BIP-340)")
        aux = bytes(32)
        schnorr_sig = ctx.schnorr_sign(msg, privkey, aux)
        print(f"  Schnorr signature:  {hexs(schnorr_sig)}")

        ok = ctx.schnorr_verify(msg, schnorr_sig, xonly)
        print(f"  Verify:             {'VALID' if ok else 'INVALID'}")

        # 4. ECDH
        section(4, "ECDH Key Agreement")
        pub2 = ctx.pubkey_create(privkey2)
        secret_a = ctx.ecdh(privkey, pub2)
        secret_b = ctx.ecdh(privkey2, pub)
        print(f"  Secret (A->B):      {hexs(secret_a)}")
        print(f"  Secret (B->A):      {hexs(secret_b)}")
        print(f"  Match:              {'YES' if secret_a == secret_b else 'NO'}")

        # 5. Hashing
        section(5, "Hashing")
        sha = ctx.sha256(pub)
        h160 = ctx.hash160(pub)
        tagged = ctx.tagged_hash("BIP0340/challenge", msg)
        print(f"  SHA-256(pubkey):    {hexs(sha)}")
        print(f"  Hash160(pubkey):    {hexs(h160)}")
        print(f"  Tagged hash:        {hexs(tagged)}")

        # 6. Bitcoin Addresses
        section(6, "Bitcoin Addresses")
        print(f"  P2PKH:              {ctx.addr_p2pkh(pub)}")
        print(f"  P2WPKH:             {ctx.addr_p2wpkh(pub)}")
        print(f"  P2TR:               {ctx.addr_p2tr(xonly)}")

        # 7. WIF
        section(7, "WIF Encoding")
        wif = ctx.wif_encode(privkey)
        print(f"  WIF:                {wif}")
        decoded = ctx.wif_decode(wif)
        print(f"  Decode roundtrip:   match={'YES' if decoded.privkey == privkey else 'NO'}")

        # 8. BIP-32
        section(8, "BIP-32 HD Key Derivation")
        seed = bytes([0x42] * 64)
        master = ctx.bip32_master(seed)
        child_key = ctx.bip32_derive_path(master, "m/44'/0'/0'/0/0")
        child_priv = ctx.bip32_privkey(child_key)
        child_pub = ctx.bip32_pubkey(child_key)
        print(f"  BIP-32 child priv:  {hexs(child_priv)}")
        print(f"  BIP-32 child pub:   {hexs(child_pub)}")

        # 9. Taproot
        section(9, "Taproot (BIP-341)")
        tap = ctx.taproot_output_key(xonly)
        print(f"  Output key:         {hexs(tap.output_key_x)}")
        print(f"  Parity:             {tap.parity}")
        ok = ctx.taproot_verify(tap.output_key_x, tap.parity, xonly)
        print(f"  Verify:             {'VALID' if ok else 'INVALID'}")

    # 10. Pedersen Commitment
    section(10, "Pedersen Commitment")
    lib_path = os.environ.get('UFSECP_LIB')
    if not lib_path:
        lib_path = os.path.join(SCRIPT_DIR, '..', '..', 'build-linux',
                                'include', 'ufsecp', 'libufsecp.so')
    _lib = ctypes.CDLL(lib_path)
    _lib.ufsecp_pedersen_commit.argtypes = [c_void_p, POINTER(c_uint8), POINTER(c_uint8), POINTER(c_uint8)]
    _lib.ufsecp_pedersen_commit.restype = c_int
    _lib.ufsecp_pedersen_verify.argtypes = [c_void_p, POINTER(c_uint8), POINTER(c_uint8), POINTER(c_uint8)]
    _lib.ufsecp_pedersen_verify.restype = c_int

    with Ufsecp() as ctx:
        value = (c_uint8 * 32)(*([0] * 31 + [42]))
        blinding = (c_uint8 * 32)(*([0] * 31 + [7]))
        commitment = (c_uint8 * 33)()
        rc = _lib.ufsecp_pedersen_commit(ctx._ctx, value, blinding, commitment)
        assert rc == 0, f"pedersen_commit failed: {rc}"
        print(f"  Commitment:         {hexs(bytes(commitment))}")
        rc = _lib.ufsecp_pedersen_verify(ctx._ctx, commitment, value, blinding)
        print(f"  Verify:             {'VALID' if rc == 0 else 'INVALID'}")

    print()

# ── GPU Examples ──────────────────────────────────────────────────────────

def demo_gpu():
    print("=== GPU Operations ===")

    # Load GPU functions directly from the C library
    lib_path = os.environ.get('UFSECP_LIB')
    if not lib_path:
        lib_path = os.path.join(SCRIPT_DIR, '..', '..', 'build-linux',
                                'include', 'ufsecp', 'libufsecp.so')
    lib = ctypes.CDLL(lib_path)

    # Bind GPU functions
    lib.ufsecp_gpu_backend_count.argtypes = [POINTER(c_uint32), c_uint32]
    lib.ufsecp_gpu_backend_count.restype = c_uint32
    lib.ufsecp_gpu_backend_name.argtypes = [c_uint32]
    lib.ufsecp_gpu_backend_name.restype = c_char_p
    lib.ufsecp_gpu_is_available.argtypes = [c_uint32]
    lib.ufsecp_gpu_is_available.restype = c_int
    lib.ufsecp_gpu_device_count.argtypes = [c_uint32]
    lib.ufsecp_gpu_device_count.restype = c_uint32
    lib.ufsecp_gpu_ctx_create.argtypes = [POINTER(c_void_p), c_uint32, c_uint32]
    lib.ufsecp_gpu_ctx_create.restype = c_int
    lib.ufsecp_gpu_ctx_destroy.argtypes = [c_void_p]
    lib.ufsecp_gpu_ctx_destroy.restype = None
    lib.ufsecp_gpu_generator_mul_batch.restype = c_int
    lib.ufsecp_gpu_ecdsa_verify_batch.restype = c_int
    lib.ufsecp_gpu_hash160_pubkey_batch.restype = c_int
    lib.ufsecp_gpu_msm.restype = c_int
    lib.ufsecp_gpu_error_str.argtypes = [c_int]
    lib.ufsecp_gpu_error_str.restype = c_char_p

    # 10. Backend Discovery
    section(10, "GPU Backend Discovery")
    backend_ids = (c_uint32 * 4)()
    n_backends = lib.ufsecp_gpu_backend_count(backend_ids, 4)
    print(f"  Backends compiled:  {n_backends}")

    use_backend = 0
    for i in range(n_backends):
        bid = backend_ids[i]
        name = lib.ufsecp_gpu_backend_name(bid).decode()
        avail = lib.ufsecp_gpu_is_available(bid)
        devs = lib.ufsecp_gpu_device_count(bid)
        print(f"  Backend {bid}: {name:<8s} available={avail} devices={devs}")
        if avail and not use_backend:
            use_backend = bid

    if not use_backend:
        print("  No GPU backends available -- skipping GPU demos.")
        return

    # Create GPU context
    gpu = c_void_p()
    rc = lib.ufsecp_gpu_ctx_create(byref(gpu), use_backend, 0)
    if rc != 0:
        print(f"  GPU context creation failed: {lib.ufsecp_gpu_error_str(rc).decode()}")
        return

    # 11. Batch Key Generation
    section(11, "GPU Batch Key Generation (4 keys)")
    N = 4
    scalars = (c_uint8 * (N * 32))(*([0] * (N * 32)))
    for i in range(N):
        scalars[i * 32 + 31] = i + 1

    pubkeys = (c_uint8 * (N * 33))()
    rc = lib.ufsecp_gpu_generator_mul_batch(gpu, scalars, N, pubkeys)
    if rc == 0:
        for i in range(N):
            pk = bytes(pubkeys[i*33:(i+1)*33])
            print(f"  GPU pubkey[{i}]:      {hexs(pk)}")
    else:
        print(f"  gpu_generator_mul_batch: {lib.ufsecp_gpu_error_str(rc).decode()}")

    # 12. ECDSA Batch Verify
    section(12, "GPU ECDSA Batch Verify")

    # Sign on CPU, verify on GPU
    with Ufsecp() as ctx:
        msgs = (c_uint8 * (N * 32))()
        sigs = (c_uint8 * (N * 64))()
        pubs = (c_uint8 * (N * 33))()

        for i in range(N):
            priv = bytes(31) + bytes([i + 1])
            msg_hash = ctx.sha256(bytes([i]))
            sig = ctx.ecdsa_sign(msg_hash, priv)
            pub = bytes(pubkeys[i*33:(i+1)*33])

            for j in range(32):
                msgs[i*32+j] = msg_hash[j]
            for j in range(64):
                sigs[i*64+j] = sig[j]
            for j in range(33):
                pubs[i*33+j] = pub[j]

    results = (c_uint8 * N)()
    rc = lib.ufsecp_gpu_ecdsa_verify_batch(gpu, msgs, pubs, sigs, N, results)
    if rc == 0:
        result_str = " ".join(f"[{i}]={'VALID' if results[i] else 'INVALID'}" for i in range(N))
        print(f"  Results: {result_str}")
    else:
        print(f"  gpu_ecdsa_verify_batch: {lib.ufsecp_gpu_error_str(rc).decode()}")

    # 13. Hash160 Batch
    section(13, "GPU Hash160 Batch")
    hashes = (c_uint8 * (N * 20))()
    rc = lib.ufsecp_gpu_hash160_pubkey_batch(gpu, pubkeys, N, hashes)
    if rc == 0:
        for i in range(N):
            h = bytes(hashes[i*20:(i+1)*20])
            print(f"  Hash160[{i}]:         {hexs(h)}")
    else:
        print(f"  gpu_hash160_pubkey_batch: {lib.ufsecp_gpu_error_str(rc).decode()}")

    # 14. MSM
    section(14, "GPU Multi-Scalar Multiplication")
    msm_result = (c_uint8 * 33)()
    rc = lib.ufsecp_gpu_msm(gpu, scalars, pubkeys, N, msm_result)
    if rc == 0:
        print(f"  MSM result:         {hexs(bytes(msm_result))}")
    else:
        print(f"  gpu_msm: {lib.ufsecp_gpu_error_str(rc).decode()}")

    lib.ufsecp_gpu_ctx_destroy(gpu)
    print()

# ── Main ──────────────────────────────────────────────────────────────────

def main():
    print("UltrafastSecp256k1 -- Python Example")

    with Ufsecp() as ctx:
        print(f"ABI version: {ctx.abi_version}")
        print(f"Library:     {ctx.version_string()}")

    demo_cpu()
    demo_gpu()
    print("All examples completed successfully.")

if __name__ == '__main__':
    main()
