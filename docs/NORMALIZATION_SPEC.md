# Signature Normalization Specification

**UltrafastSecp256k1 v3.13.0** -- Canonical Form & Strictness Rules

---

## 1. Overview

This document specifies the **canonical signature forms** enforced by UltrafastSecp256k1.
Bitcoin consensus and relay rules (BIP-62, BIP-66, BIP-146) require specific normalization
to prevent transaction malleability. This library enforces these rules by default.

---

## 2. ECDSA Signature Normalization

### 2.1 Low-S Rule (BIP-62 / BIP-146)

**Rule**: A valid ECDSA signature `(r, s)` MUST satisfy `s <= n/2`, where:

$$n = \texttt{0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141}$$

$$\frac{n}{2} = \texttt{0x7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF5D576E7357A4501DDFE92F46681B20A0}$$

**Why**: Given valid `(r, s)`, the pair `(r, n - s)` is also a valid signature for
the same message and key. Without low-S enforcement, either form satisfies verification,
enabling **transaction malleability** -- a third party can flip `s` without invalidating
the signature, changing the transaction hash (txid).

**Enforcement in this library**:

| Function | Behavior |
|----------|----------|
| `ecdsa_sign()` | Always returns low-S (`sig.normalize()` called internally) |
| `ct::ecdsa_sign()` | Always returns low-S (constant-time normalize) |
| `ecdsa_verify()` | Accepts **both** low-S and high-S (permissive verify) |
| `ECDSASignature::normalize()` | If `s > n/2`, replaces with `n - s` |
| `ECDSASignature::is_low_s()` | Returns `true` iff `s <= n/2` |

**Implementation** ([ecdsa.cpp](../cpu/src/ecdsa.cpp)):

```cpp
// HALF_ORDER = n/2
static const Scalar HALF_ORDER = Scalar::from_hex(
    "7fffffffffffffffffffffffffffffff5d576e7357a4501ddfe92f46681b20a0");

ECDSASignature ECDSASignature::normalize() const {
    if (is_low_s()) return *this;
    return {r, s.negate()};   // s' = n - s
}

bool ECDSASignature::is_low_s() const {
    auto s_bytes = s.to_bytes();
    auto half_bytes = HALF_ORDER.to_bytes();
    for (size_t i = 0; i < 32; ++i) {
        if (s_bytes[i] < half_bytes[i]) return true;
        if (s_bytes[i] > half_bytes[i]) return false;
    }
    return true; // s == n/2
}
```

**Test coverage**: `audit_security.cpp`, `audit_fuzz.cpp`, `differential_test.cpp`
all assert `sig.is_low_s()` after signing and `normalize().is_low_s()` round-trip.

### 2.2 Scalar Range

| Component | Valid Range | Check |
|-----------|------------|-------|
| `r` | `[1, n-1]` | `ecdsa_sign` returns zero-sig if `r == 0`; `ecdsa_verify` rejects `r == 0` |
| `s` | `[1, n-1]` | `ecdsa_sign` returns zero-sig if `s == 0`; `ecdsa_verify` rejects `s == 0` |
| `s` (normalized) | `[1, n/2]` | Enforced by `normalize()` in `ecdsa_sign` |

---

## 3. DER Encoding (BIP-66)

### 3.1 Encoding Rules

ECDSA signatures are DER-encoded as:

```
0x30 <total_len> 0x02 <r_len> <r_bytes> 0x02 <s_len> <s_bytes>
```

**Strict DER** (BIP-66, enforced since Bitcoin soft fork at block 363,724):

| Rule | Description |
|------|-------------|
| Tag bytes | `0x30` (SEQUENCE), `0x02` (INTEGER) -- no alternatives |
| Length | Single-byte length only (max 72 bytes total, fits in 1 byte) |
| No leading zeros | `r` and `s` MUST NOT have unnecessary leading `0x00` bytes |
| Negative pad | If high bit of `r` or `s` value byte is set, prepend `0x00` |
| Minimal encoding | `r` and `s` MUST NOT be empty; MUST NOT be negative |
| No trailing data | Encoded length MUST match actual data length exactly |

### 3.2 Implementation

`ECDSASignature::to_der()` ([ecdsa.cpp](../cpu/src/ecdsa.cpp)):

- Strips leading zeros from `r` and `s` byte arrays
- Adds `0x00` padding when high bit is set (prevents DER negative interpretation)
- Returns `{buffer, actual_length}` -- max 72 bytes
- Output is always strict BIP-66 compliant

### 3.3 Compact Encoding

`ECDSASignature::to_compact()` produces a fixed 64-byte concatenation `r(32) || s(32)`.
This is **not** DER and is used for internal serialization, test vectors, and protocols
that specify raw `(r, s)` pairs (e.g., BIP-340 Schnorr).

---

## 4. Schnorr Signatures (BIP-340)

BIP-340 Schnorr signatures are **inherently non-malleable** by design:

| Property | BIP-340 Rule |
|----------|-------------|
| Format | 64 bytes: `R.x (32) \|\| s (32)` |
| `R` | Always even-Y (x-only); no normalization needed |
| `s` | Full range `[0, n-1]`; no low-S rule (malleability prevented by `e` binding) |
| Nonce `k` | Deterministic from `(d', aux) -> t -> rand -> k` (BIP-340 Sdefault signing) |
| Public key | x-only (32 bytes); always even-Y internally |

**Why no low-S for Schnorr?** The verification equation `s*G = R + e*P` binds `s`
uniquely to `(R, e, P)`. Flipping `s` would require a different `R`, which changes the
challenge `e = H(R.x || P.x || m)`, making the forged signature invalid.

---

## 5. RFC 6979 Deterministic Nonce

### 5.1 Specification Compliance

The library implements RFC 6979 with HMAC-SHA256 for deterministic ECDSA nonce generation.

| Property | Value |
|----------|-------|
| Hash | HMAC-SHA256 |
| Private key encoding | 32 bytes, big-endian |
| Message hash | 32 bytes (pre-hashed) |
| Loop | Retry until `k ∈ [1, n-1]` |
| Extra data | None (standard mode) |

### 5.2 Implementation Notes

- Optimized: precomputed HMAC `ipad`/`opad` midstates reuse
- Direct `sha256_compress_dispatch()` calls (no SHA256 object overhead)
- Saves ~4 compress calls vs naive implementation
- Tested against RFC 6979 Appendix A.2.5 (secp256k1) test vectors

---

## 6. Verification Strictness

### 6.1 What `ecdsa_verify` Accepts

| Input | Acceptance |
|-------|-----------|
| Low-S signature | [OK] Accepted |
| High-S signature | [OK] Accepted (permissive) |
| `r == 0` or `s == 0` | [FAIL] Rejected |
| `r >= n` or `s >= n` | [FAIL] Rejected (Scalar constructor reduces mod n) |
| Infinity result | [FAIL] Rejected |

> **Note**: `ecdsa_verify` is intentionally permissive on S-normalization. This matches
> Bitcoin Core's `secp256k1_ecdsa_verify()` behavior. Consensus-level low-S enforcement
> is the responsibility of the transaction validation layer, not the signature primitive.

### 6.2 Schnorr Verify Strictness (BIP-340)

| Input | Acceptance |
|-------|-----------|
| Valid `(R.x, s)` with `s < n` | [OK] Accepted |
| `s >= n` | [FAIL] Rejected |
| `R` not on curve | [FAIL] Rejected (lift_x fails) |
| Public key not on curve | [FAIL] Rejected |

---

## 7. Cross-Backend Consistency

All backends (CPU, CUDA, OpenCL, Metal) enforce identical normalization:

- Sign always produces low-S (ECDSA)
- DER encoding is identical byte-for-byte
- Compact encoding is identical
- Schnorr signatures are bit-identical across backends (deterministic nonce)

Verified by: `test_ct_equivalence.cpp` (CT==FAST on CPU), multi-backend equivalence tests.

---

## 8. Summary Table

| Feature | ECDSA | Schnorr (BIP-340) |
|---------|-------|--------------------|
| Low-S normalization | [OK] Always on sign | N/A (not needed) |
| DER encoding | [OK] Strict BIP-66 | N/A (64-byte fixed) |
| Nonce generation | RFC 6979 (HMAC-SHA256) | BIP-340 Sdefault (tagged hash) |
| Malleability protection | Low-S + deterministic k | Inherent (challenge binding) |
| Verify accepts high-S | Yes (permissive) | N/A |
| CT variant | `ct::ecdsa_sign` | `ct::schnorr_sign` |

---

## References

- [BIP-62](https://github.com/bitcoin/bips/blob/master/bip-0062.mediawiki) -- Dealing with malleability
- [BIP-66](https://github.com/bitcoin/bips/blob/master/bip-0066.mediawiki) -- Strict DER signatures
- [BIP-146](https://github.com/bitcoin/bips/blob/master/bip-0146.mediawiki) -- Low-S enforcement
- [BIP-340](https://github.com/bitcoin/bips/blob/master/bip-0340.mediawiki) -- Schnorr Signatures
- [RFC 6979](https://www.rfc-editor.org/rfc/rfc6979) -- Deterministic DSA/ECDSA
- [Bitcoin Core consensus](https://github.com/bitcoin/bitcoin/blob/master/src/script/interpreter.cpp) -- `SCRIPT_VERIFY_LOW_S`

---

*UltrafastSecp256k1 v3.13.0 -- Normalization Specification*
