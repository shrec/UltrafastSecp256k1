# Signature & Encoding Normalization Specification

> This document specifies the normalization rules enforced by UltrafastSecp256k1
> for ECDSA signatures, DER encoding, and Schnorr signatures.

---

## 1. Low-S Normalization (BIP-62 Rule 5)

### Background

An ECDSA signature `(r, s)` for secp256k1 has a **malleability** property:
both `(r, s)` and `(r, n - s)` are valid signatures for the same message.
This allows third parties to mutate transaction signatures without
invalidating them, which caused real issues in early Bitcoin.

**BIP-62 Rule 5** mitigates this by requiring `s ≤ n/2`, where:

```
n   = FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
n/2 = 7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF5D576E7357A4501DDFE92F46681B20A0
```

### Implementation

| Function | Location | Behavior |
|----------|----------|----------|
| `ECDSASignature::is_low_s()` | `cpu/src/ecdsa.cpp:75` | Returns `true` iff `s ≤ n/2` (byte-wise comparison) |
| `ECDSASignature::normalize()` | `cpu/src/ecdsa.cpp:70` | If `s > n/2`, returns `{r, n - s}`; otherwise returns `*this` |
| `ecdsa_sign()` | `cpu/src/ecdsa.cpp:305` | **Always** normalizes output: `return sig.normalize()` |

### Guarantees

- **Sign**: `ecdsa_sign()` always produces low-S signatures. No caller action needed.
- **Verify**: `ecdsa_verify()` accepts **both** low-S and high-S signatures (permissive).
  This matches Bitcoin Core's verification behavior.
- **Manual normalization**: Callers receiving external signatures can call
  `sig.normalize()` before storage/relay to enforce low-S.

### Constant-Time Note

`is_low_s()` performs a byte-wise comparison that is **not** constant-time.
This is acceptable because `s` is a public value in a signature — there is
no secret to leak. The `ct::` namespace should be used for any operation
involving private keys.

---

## 2. DER Encoding (ITU-T X.690)

### Encoding: `to_der()`

Produces **strict DER** encoding per X.690 Distinguished Encoding Rules:

```
SEQUENCE {
  INTEGER r   -- minimal encoding, no leading zeros (except 0x00 pad for high-bit)
  INTEGER s   -- minimal encoding, no leading zeros (except 0x00 pad for high-bit)
}
```

Format:
```
30 <total_len> 02 <r_len> <r_bytes> 02 <s_len> <s_bytes>
```

**Rules enforced during encoding:**
1. Leading zero bytes in `r`/`s` are stripped (minimal encoding)
2. If the most-significant bit of the first remaining byte is set, a `0x00`
   padding byte is prepended (integers are signed in DER)
3. Maximum encoded length: 72 bytes (2 + 2 + 33 + 2 + 33)

### Decoding: `from_der()`

**Not currently implemented.** External DER signatures should be decoded with
a strict parser that rejects:
- Non-minimal length encodings
- Negative integers (high bit set without `0x00` pad)
- Trailing garbage bytes
- Length > 72

> **TODO (roadmap 2.3.1):** Add `ECDSASignature::from_der()` with strict
> parsing + fuzzing coverage.

### Compact Encoding

| Function | Format | Length |
|----------|--------|--------|
| `to_compact()` | `r[32] \|\| s[32]` (big-endian, fixed-width) | 64 bytes |
| `from_compact()` | Parses 64-byte array back to `{r, s}` | — |

Compact encoding is always 64 bytes with no ambiguity or parsing risk.

---

## 3. Schnorr Signatures (BIP-340)

### Normalization Rules

BIP-340 signatures use a different convention:

- **Public key**: x-only (32 bytes). The y-coordinate is implicitly even.
- **Nonce point R**: x-only. If the computed R has odd y, negate the nonce `k`.
- **Signature format**: `R.x[32] || s[32]` = 64 bytes (fixed-width, no DER).

### Even-Y Convention

```
R = k·G
if R.y is odd:
    k = n - k   (negate nonce)
    R = -R       (now R.y is even)
```

This is enforced in `schnorr_sign()` (cpu/src/schnorr.cpp). There is no
malleability in BIP-340 because the nonce commitment binds to `R.x` and the
even-y convention is deterministic.

### Verification

`schnorr_verify()` lifts `R.x` to the point with even y and checks the
equation. No normalization choices exist for the verifier.

---

## 4. Field Element Normalization

Field arithmetic in UltrafastSecp256k1 uses a 5×52-bit limb representation.
Results may exceed the prime `p` before normalization.

| Function | Purpose |
|----------|---------|
| `normalize()` | Full reduction mod p: result in `[0, p-1]` |
| `normalize_weak()` | Partial reduction: magnitude ≤ 1, but may equal `p` |

**Rule:** Any field element used for comparison, serialization, or output
**must** be fully normalized first. Internal arithmetic chains may defer
normalization for performance.

---

## 5. Scalar Normalization

Scalars are always reduced mod `n` (group order) upon construction.
`Scalar::from_bytes()` and `Scalar::from_hex()` both reduce the input mod `n`.

There is no separate normalization step — scalars are always canonical.

---

## 6. Summary of Guarantees

| Property | Status | Notes |
|----------|--------|-------|
| Sign produces low-S | ✅ Enforced | `ecdsa_sign()` calls `normalize()` |
| Verify accepts high-S | ✅ Permissive | Matches Bitcoin Core |
| DER encoding is strict | ✅ Enforced | Minimal-length integers |
| DER decoding (parsing) | ❌ Not implemented | Roadmap 2.3.1 |
| Schnorr even-y nonce | ✅ Enforced | BIP-340 compliant |
| Schnorr 64-byte format | ✅ Fixed | No ambiguity |
| Field elements normalized for output | ✅ Required | `normalize()` before compare/serialize |
| Scalars always canonical | ✅ Enforced | Reduced mod n at construction |

---

## References

- [BIP-62](https://github.com/bitcoin/bips/blob/master/bip-0062.mediawiki) — Dealing with malleability
- [BIP-340](https://github.com/bitcoin/bips/blob/master/bip-0340.mediawiki) — Schnorr signatures
- [RFC 6979](https://datatracker.ietf.org/doc/html/rfc6979) — Deterministic ECDSA nonce
- [ITU-T X.690](https://www.itu.int/rec/T-REC-X.690) — DER encoding rules
