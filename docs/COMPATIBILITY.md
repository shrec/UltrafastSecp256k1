# Compatibility Targets

UltrafastSecp256k1 supports two operational modes for input validation.
This document describes the precise semantics and how each mode is engaged.

---

## General mode (internal)

Internal helpers such as `Scalar::from_bytes()` and `FieldElement::from_bytes()`
apply **modular reduction** to out-of-range inputs:

- `Scalar::from_bytes(bytes32)` -- reduces mod n (curve order)
- `FieldElement::from_bytes(bytes32)` -- reduces mod p (field prime)

These are used for intermediate computations (e.g. hash-to-scalar in RFC 6979
nonce generation) where reduction is mathematically correct and required.

**General-mode functions must NOT be used on public/untrusted inputs.**

---

## Bitcoin strict mode (public API)

All public-facing parsing and verification functions enforce **canonical encoding**
as required by BIP-340, BIP-32, and standard Bitcoin consensus rules:

| Rule | BIP reference | Enforcement |
|------|--------------|-------------|
| Private key: 1 <= sk < n | BIP-340 signing | `Scalar::parse_bytes_strict_nonzero` |
| Schnorr r: r < p | BIP-340 verify | `FieldElement::parse_bytes_strict` |
| Schnorr s: 1 <= s < n | BIP-340 verify | `Scalar::parse_bytes_strict_nonzero` |
| Pubkey x: x < p | BIP-340 / compressed | `FieldElement::parse_bytes_strict` |

Values at or above the modulus are **rejected immediately** -- never reduced.

### Strict parsing API

```cpp
// Returns false if bytes >= n (no reduction, no mutation of out)
bool Scalar::parse_bytes_strict(const uint8_t* bytes32, Scalar& out);

// Returns false if bytes >= n OR bytes == 0
bool Scalar::parse_bytes_strict_nonzero(const uint8_t* bytes32, Scalar& out);

// Returns false if bytes >= p (no reduction, no mutation of out)
bool FieldElement::parse_bytes_strict(const uint8_t* bytes32, FieldElement& out);

// Returns false if r >= p, s >= n, or s == 0
bool SchnorrSignature::parse_strict(const uint8_t* sig64, SchnorrSignature& out);
```

### C ABI strict behavior

The C ABI (`ufsecp_*` functions) defaults to **Bitcoin strict** for all
public inputs:

| Function | Strict check | Error on violation |
|----------|-------------|-------------------|
| `ufsecp_seckey_verify` | 1 <= sk < n | `UFSECP_ERR_BAD_KEY` |
| `ufsecp_schnorr_verify` | r < p, s < n, s != 0, pk.x < p | `UFSECP_ERR_BAD_SIG` / `UFSECP_ERR_BAD_KEY` |
| `ufsecp_ec_pubkey_parse` | x < p (compressed/x-only) | `UFSECP_ERR_BAD_PUBKEY` |

Downstream integrators can distinguish between **encoding rejection**
(`UFSECP_ERR_BAD_SIG`, `UFSECP_ERR_BAD_KEY`) and **cryptographic verification
failure** (`UFSECP_ERR_VERIFY_FAIL`) via the returned error code.

---

## Build flag: `-DUFSECP_BITCOIN_STRICT=ON`

When this CMake flag is enabled (default: ON), the compile-time define
`UFSECP_BITCOIN_STRICT` is set globally. This ensures:

1. All public API paths use strict parsing (already the default behavior).
2. A compile-time assertion documents that the library is built in
   Bitcoin-compatible strict mode.
3. CI includes a dedicated job that builds and tests with this flag.

When set to OFF, the library still uses strict parsing in its public API
but the compile-time marker is absent. This is intended only for
non-Bitcoin use cases that may add custom reduction policies in the future.

---

## Interoperability with libsecp256k1

UltrafastSecp256k1 aims for **accept/reject parity** with
[libsecp256k1](https://github.com/bitcoin-core/secp256k1) on all standard
inputs:

- Same BIP-340 test vectors pass/fail
- Same edge-case rejection (r=p, s=n, pk.x=p, etc.)
- Same ECDSA low-S normalization behavior

The `test_bip340_strict` test suite validates 31 non-canonical rejection
scenarios. The `test_bip340_vectors` suite validates all 15 official
BIP-340 test vectors (0-14).

---

## Self-audit vs independent audit

The unified audit runner (`unified_audit_runner`) is a **self-assessment**
tool. It does NOT replace independent third-party cryptographic audit.
See [SECURITY.md](../SECURITY.md) for the project's security posture and
audit status.
