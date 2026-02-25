# Supported Guarantees -- `ufsecp` C ABI

> **Version**: 3.3.0 (ABI 1)
> **Date**: 2026-02-19

This document defines what the `ufsecp_*` C ABI promises and where the
boundary lies.

---

## Tier 1 -- Stable (ABI >= 1)

These are covered by the ABI version contract.  Breaking changes require
a new `UFSECP_ABI_VERSION`.

| Category | Guarantee |
|---|---|
| **Opaque context** | `ufsecp_ctx` is always heap-allocated via `ufsecp_ctx_create()` and freed via `ufsecp_ctx_destroy()`.  No layout is exposed. |
| **Error model** | Every function returns `ufsecp_error_t` (0 = OK).  New error codes may be added (minor bump). |
| **Key sizes** | Private: 32 bytes, Compressed pubkey: 33 bytes, Uncompressed: 65 bytes, x-only: 32 bytes. |
| **Signature sizes** | Compact ECDSA/Schnorr: 64 bytes, DER ECDSA: <=72 bytes. |
| **Deterministic nonce** | ECDSA uses RFC 6979 (HMAC-DRBG / SHA-256). |
| **Low-S** | `ufsecp_ecdsa_sign()` always normalises to low-S (BIP-62). |
| **BIP-340** | Schnorr follows BIP-340 byte-for-byte. |
| **SHA-256** | FIPS 180-4 correct; hardware acceleration (SHA-NI, ARMv8-SHA2) used when detected at runtime. |
| **ECDH** | Three modes: compressed-hash, x-only-hash, raw x-coordinate. |
| **BIP-32** | Full HD derivation: master from seed, normal/hardened child, full path string. |
| **Taproot** | BIP-341 output key, key-path tweak, commitment verification. |
| **Addresses** | P2PKH, P2WPKH (Bech32), P2TR (Bech32m). |
| **WIF** | Encode/decode, mainnet/testnet, compressed/uncompressed. |

### Thread safety

Each `ufsecp_ctx` is single-threaded.  Create one per thread, or protect
with external synchronisation.  Stateless functions (`ufsecp_sha256`,
`ufsecp_hash160`, `ufsecp_tagged_hash`) are thread-safe.

### Memory

All output buffers are caller-owned.  The library never allocates on
behalf of the caller except during `ufsecp_ctx_create` /
`ufsecp_ctx_clone`.

---

## Tier 2 -- Experimental (no ABI promise)

| Feature | Status |
|---|---|
| FROST threshold signatures | API may change (not exposed in `ufsecp.h` yet) |
| MuSig2 multi-signatures | API may change |
| Adaptor signatures | API may change |
| Pedersen commitments | API may change |
| Multi-coin address derivation | API may change |
| GPU batch operations | Separate library, no C ABI promise |

These will graduate to Tier 1 once their API surface is frozen and a
test harness covers all edge cases.

---

## Tier 3 -- Internal (never exposed)

- Field element / scalar / point internals
- Precompution table format
- Montgomery / Barrett reduction details
- Assembly code layout

---

## Constant-Time Architecture (Dual-Layer Model)

Unlike libraries that expose a flag or mode switch for constant-time safety,
UltrafastSecp256k1 uses a **dual-layer architecture** where both layers are
**always active simultaneously**.  There is no opt-in, no opt-out,
no flag -- the human factor is eliminated by design.

```
+---------------------------------------------------------------+
|  Layer 1 -- FAST:  public operations (verify, point arith)     |
|  Layer 2 -- CT  :  secret operations (sign, nonce, tweak)      |
|  Both layers are ALWAYS ACTIVE.  No flag.  No user choice.    |
+---------------------------------------------------------------+
```

| Layer | Namespace | What runs here | Guarantee |
|---|---|---|---|
| **Fast** | `secp256k1::fast` | Verification, public key serialisation, point arithmetic for non-secret operands | Maximum speed.  No timing guarantee needed -- operands are public. |
| **CT** | `secp256k1::ct` | Signing, nonce generation, key tweak, scalar multiplication with secret keys, ECDH | Side-channel resistant.  No secret-dependent branches or memory accesses.  Complete addition formula (branchless, 12M+2S).  Fixed-trace scalar multiplication.  CT table lookup (scans all entries). |

### Verification tools

The CT layer supports Valgrind/MSAN verification via compile-time markers:
- `SECP256K1_CLASSIFY(ptr, len)` -- mark memory as secret (undefined)
- `SECP256K1_DECLASSIFY(ptr, len)` -- mark memory as public (defined)

Build with `-DSECP256K1_CT_VALGRIND=1` to activate.  Any
"conditional jump depends on uninitialised value" between
classify and declassify indicates a CT violation.

> **Why not a flag?**  A flag-based model (`FLAG_CT` / `FLAG_FAST`)
> requires the developer to make the right choice for every call site.
> A single mistake (forgetting CT for a signing operation) silently opens
> a side-channel.  In UltrafastSecp256k1, secret-dependent operations
> **always** take the CT path -- correctness is architectural, not optional.

---

## Versioning Rules

| Bump | Meaning | ABI Impact |
|---|---|---|
| Patch (3.3.x) | Bug fixes only | Compatible |
| Minor (3.x.0) | New functions added | Compatible (additions only) |
| Major (x.0.0) | Breaking changes | ABI version incremented |

Clients should guard with:

```c
if (ufsecp_abi_version() != UFSECP_ABI_VERSION) {
    /* linked library is incompatible */
}
```

---

## What This Library Does NOT Do

- No random number generation.  Callers provide entropy / aux randomness.
- No key storage or wallet management.
- No network communication.
- No consensus validation beyond signature math.

---

*"Correctness is absolute; performance is earned."*
