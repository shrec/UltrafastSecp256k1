# PROTOCOL_SPEC.md — UltrafastSecp256k1 (publishable)

> Version: 1.0 — 2026-04-21
> Closes CAAS gap **G-9**.
>
> This document is the citation-ready specification of the library's
> public ABI surface. External auditors and downstream implementers
> can read this without reading source. Source-level evidence for each
> clause is in [SPEC_TRACEABILITY_MATRIX.md](SPEC_TRACEABILITY_MATRIX.md).
>
> SPDX-License-Identifier: MIT
> Stable identifier: `urn:ufsecp:spec:1.0:2026-04-21`

## 0. Scope and conventions

This specification describes the externally-observable behaviour of
UltrafastSecp256k1. It does not describe internal data structures,
performance properties, or non-ABI helpers.

| Convention | Meaning |
|------------|---------|
| MUST / MUST NOT / SHALL | Strict requirement; violation is a defect |
| SHOULD / SHOULD NOT | Recommended; deviation requires written justification in the doc string |
| MAY | Permitted variation |
| `ufsecp_*` | Public C ABI symbol; see `include/ufsecp/` |
| `secp256k1_*` | Internal C++ namespace; not part of the public spec |
| Byte order | All multi-byte integers are big-endian on the wire unless explicitly stated |
| Strings | All identifiers are ASCII, NUL-terminated, length-bounded |

## 1. Domain parameters

UltrafastSecp256k1 implements operations over **secp256k1** as defined
by SEC 2 v2.0 §2.4.1. Constants:

```
p = 0xFFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFE FFFFFC2F
n = 0xFFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFE BAAEDCE6 AF48A03B BFD25E8C D0364141
a = 0
b = 7
G_x = 0x79BE667E F9DCBBAC 55A06295 CE870B07 029BFCDB 2DCE28D9 59F2815B 16F81798
G_y = 0x483ADA77 26A3C465 5DA4FBFC 0E1108A8 FD17B448 A6855419 9C47D08F FB10D4B8
h = 1
```

The library does not support any other curve.

## 2. Encoding rules

### 2.1 Scalars (32 bytes, big-endian)

A scalar `s` is serialised as 32 bytes big-endian. The library MUST
reject any scalar in a secret-bearing API where `s = 0` or `s ≥ n`.

### 2.2 Field elements (32 bytes, big-endian)

A field element is serialised as 32 bytes big-endian. The library
MUST reject any input in a parser API where `x ≥ p`.

### 2.3 Public keys

| Encoding | Length | First byte | Body |
|----------|--------|-----------|------|
| Compressed | 33 | `0x02` (Y even) or `0x03` (Y odd) | `X` |
| Uncompressed | 65 | `0x04` | `X ‖ Y` |
| X-only (BIP-340) | 32 | (none) | `X` |

The library MUST:

1. Reject the point at infinity in any encode/parse API.
2. Reject a parsed point that does not satisfy `Y² ≡ X³ + 7 (mod p)`.
3. Reject any first-byte value not listed above.
4. Reject hybrid encodings (`0x06`/`0x07`) that some legacy libraries
   accept.

### 2.4 ECDSA signatures (DER and compact)

DER per SEC 1: `30 LL 02 LR R 02 LS S` with strict canonical form:

- No unnecessary leading zeros in `R` or `S`.
- High bit of `R` and `S` set if and only if a leading zero is present.
- `LL` matches the body length exactly.
- `R`, `S` ∈ [1, n−1].

Compact form: `R ‖ S` as 64 bytes. Same `R`, `S` constraints.

The library MUST reject signatures that do not satisfy these
constraints. Low-S enforcement is **not** automatic; callers can
enforce via `ufsecp_ecdsa_signature_normalize()` or `is_low_s()`.

### 2.5 Schnorr signatures (BIP-340)

`R_x ‖ S` as 64 bytes. The library MUST reject `R_x ≥ p`, `S ≥ n`,
and any non-conforming length.

## 3. Algorithms

### 3.1 ECDSA sign (default — RFC 6979 deterministic)

Given private key `d ∈ [1, n−1]` and message hash `z` (32 bytes):

1. Derive `k` per RFC 6979 §3.2 with `int2octets(d) ‖ bits2octets(z)`.
2. Compute `R = kG`; let `r = R.x mod n`. If `r = 0`, regenerate `k`.
3. Compute `s = k⁻¹(z + r·d) mod n`. If `s = 0`, regenerate `k`.
4. Output `(r, s)`. The library MAY return the low-S form via the
   `_low_s` variant.

The library MUST NOT use any randomness in this path. Optional
`extra_data` (32 bytes) is mixed into the RFC 6979 K-update per §3.6.

### 3.2 ECDSA verify

Given public key `Q`, message hash `z`, and signature `(r, s)`:

1. MUST reject if `r ∉ [1, n−1]` or `s ∉ [1, n−1]`.
2. Compute `u₁ = z·s⁻¹ mod n`, `u₂ = r·s⁻¹ mod n`.
3. Compute `R' = u₁·G + u₂·Q`. MUST reject if `R'` is the point at
   infinity.
4. Accept if `R'.x mod n == r` (computed in $\mathbb{F}_p$, reduced
   to $\mathbb{Z}_n$).

When `R'.x ∈ [n, p−1]`, the comparison `R'.x mod n == r` is performed
after reduction (RR-004 closed 2026-04-03).

### 3.3 Schnorr sign (BIP-340)

Implements BIP-340 verbatim. Optional `aux_rand` (32 bytes) per spec.
Without `aux_rand` the operation is deterministic.

### 3.4 Schnorr verify (single + batch)

Single: implements BIP-340 verify verbatim.

Batch: given $\{(P_i, m_i, \sigma_i)\}_{i=1}^N$, the library samples
$a_1 = 1$ and $a_2,\dots,a_N$ uniformly from $[1, n-1]$, and accepts
if and only if

$$
\Bigl(\sum_i a_i s_i\Bigr) G = \sum_i a_i R_i + \sum_i a_i e_i P_i
$$

with $e_i = H_{\text{BIP0340/challenge}}(R_i \mid\mid P_i \mid\mid m_i)$.
Any single failed signature MUST cause the batch to reject; the
library MUST NOT skip a failure to optimise throughput.

### 3.5 ECDH (X-only and X9.63)

X-only (libsecp256k1 default): output `SHA256(0x02|0x03 ‖ X)` of the
shared point. X9.63: caller-selectable hash; library implements
SHA-256 by default.

The library MUST reject a peer pubkey that is the point at infinity
or off-curve before any scalar multiplication.

### 3.6 EC recovery (`ecrecover`)

Given message hash `z`, signature `(r, s)`, and recovery id `v ∈ {0,1,2,3}`:

1. Reconstruct `R` from `r` (and possibly `r + n` if `v ≥ 2`) and
   parity from `v`.
2. Compute `Q = r⁻¹(s·R − z·G)`.
3. Output the compressed encoding of `Q`.

The library MUST reject `v ∉ {0,1,2,3}` and any reconstruction that
yields an off-curve point.

### 3.7 BIP-32 HD derivation

Implements BIP-32 verbatim. Hardened indices use `0x00 ‖ k_par ‖
ser32(i)`; non-hardened use `ser_P(K_par) ‖ ser32(i)`. Depth limit
255 is enforced.

### 3.8 BIP-340/341/342 Taproot

Implements key-path and script-path tweaks per BIP-341. Control block
structure validated per BIP-341 §control block.

### 3.9 BIP-324 v2 transport

Handshake: ECDH on x-only pubkeys; HKDF labels per BIP-324; ChaCha20-
Poly1305 AEAD over framed messages; rekey on counter rollover.

### 3.10 BIP-352 silent payments

Per-output tweak `t_k = hash_BIP0352("SharedSecret", ecdh ‖ ser32(k))`.
Optional label tweak supported.

### 3.11 MuSig2 (BIP-327)

Implements BIP-327 verbatim. Round state machine enforced; out-of-
order calls MUST be rejected.

### 3.12 FROST (RFC 9591)

Implements RFC 9591 verbatim. Identifiable abort surfaces the
participant ID that caused failure.

## 4. Constant-time guarantees

The library guarantees that the following functions execute with no
secret-dependent branch and no secret-dependent memory address on the
tested platforms (x86-64, ARM64, RISC-V, Apple M-series; verified by
dudect + Valgrind CT + ct-verif agreement):

- `ufsecp_ecdsa_sign*`, `ufsecp_schnorr_sign*`
- `ufsecp_ecdh*`
- `ufsecp_keygen_*`
- `ufsecp_bip32_*` (private-key path)
- `ufsecp_bip324_*` (handshake key derivation)
- `ufsecp_musig2_*` (partial sign)
- `ufsecp_frost_*` (signing path)
- All scalar / field / group primitives in CT layer

Verify-only and parser-only functions are not required to be
constant-time, but MUST NOT branch on secret data when secret data
is incidentally present (e.g. a verifier that also knows the signer's
key for testing).

See [HARDWARE_SIDE_CHANNEL_METHODOLOGY.md](HARDWARE_SIDE_CHANNEL_METHODOLOGY.md)
for what this CT claim does NOT cover.

## 5. Failure model

| Failure | ABI behaviour |
|---------|--------------|
| Invalid input bytes | Return `UFSECP_ERROR_INVALID_INPUT`; do not write outputs |
| OS RNG failure | Return `UFSECP_ERROR_RNG`; do not zero-fill, do not fall back to userland PRNG |
| Internal contract violation (assert) | Build configuration dependent; release builds return error code, debug builds abort |
| Out-of-memory | Return `UFSECP_ERROR_NOMEM`; library is otherwise heap-free in hot paths |

All error returns MUST leave caller-provided output buffers
unmodified.

## 6. Versioning and stability

The library follows semantic versioning. The C ABI under
`include/ufsecp/` is stable across PATCH releases. MINOR releases may
add ABI symbols; they MUST NOT remove or change existing symbols.
MAJOR releases may break ABI; the changelog MUST list every break.

## 7. Out of scope

- Other curves (P-256, ed25519, sm2, etc.)
- General-purpose hash, AEAD, KDF API surface (only BIP-324's
  ChaCha20-Poly1305 is exposed and only inside BIP-324)
- Post-quantum algorithms
- Hardware-side-channel resistance (see G-3)
- Compliance certification (see G-6)

## 8. References

- SEC 2 v2.0 — Recommended Elliptic Curve Domain Parameters
- SEC 1 v2.0 — Elliptic Curve Cryptography
- RFC 6979 — Deterministic ECDSA Nonce
- RFC 9591 — FROST
- BIP-32, BIP-39, BIP-324, BIP-327, BIP-340, BIP-341, BIP-342, BIP-352
- Wycheproof project (`google/wycheproof`)
- NIST SP 800-90A/B (informative; library does not implement a DRBG)

## 9. Citation

> UltrafastSecp256k1 Project. *UltrafastSecp256k1 Protocol
> Specification, version 1.0.* 2026-04-21. Stable identifier
> `urn:ufsecp:spec:1.0:2026-04-21`. Available at:
> https://github.com/shrec/UltrafastSecp256k1/blob/dev/docs/PROTOCOL_SPEC.md
