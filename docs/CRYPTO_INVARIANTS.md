# Cryptographic Invariants

> **Audience:** Integrators, binding authors, auditors.
> Violating any invariant below can lead to **key leakage**, **forged signatures**, or **consensus failure**.

---

## 1. Private Keys

| Invariant | Detail |
|-----------|--------|
| Range | `0 < k < n` where `n = FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141` |
| Rejection | `k == 0` and `k >= n` are always rejected (`UFSECP_ERR_BAD_KEY`) |
| Constant-time | All private key operations use the CT layer -- no opt-in required |
| Zeroisation | Callers must `memset(seckey, 0, 32)` after use; the library zeroes internal copies |

### Common Misuse

```
// WRONG: using raw random bytes without validation
getrandom(key, 32);
ufsecp_ecdsa_sign(ctx, key, ...);  // may fail if key >= n

// CORRECT: validate first
if (ufsecp_seckey_verify(ctx, key) != UFSECP_OK) {
    // retry or abort
}
```

---

## 2. Nonce Safety (RFC 6979)

| Invariant | Detail |
|-----------|--------|
| Deterministic | ECDSA nonces are always RFC 6979 -- no caller-supplied nonce API |
| Domain separation | `sign(key, hash1) != sign(key, hash2)` always (different message = different nonce) |
| No nonce reuse | Two signatures with the same nonce reveal the private key |

### Why No Custom Nonce API

Exposing a nonce parameter is the #1 cause of key leakage in ECDSA libraries
(PlayStation 3, Android SecureRandom, many others). This library provides
**no way** to supply a custom nonce. RFC 6979 is hardwired.

---

## 3. Public Key Validation

| Invariant | Detail |
|-----------|--------|
| On-curve | `ufsecp_pubkey_parse()` rejects points not on `y^2 = x^3 + 7` |
| Not infinity | The point at infinity is always rejected |
| Canonical encoding | Compressed (33 bytes, prefix `02`/`03`) or uncompressed (65 bytes, prefix `04`) |
| No hybrid | Prefix `06`/`07` (hybrid encoding) is rejected |

### Invalid Point Attacks

Accepting an invalid public key in ECDH can leak the private key
via invalid-curve attacks. `ufsecp_ecdh()` validates the peer key internally.

---

## 4. Signature Encoding

### ECDSA

| Invariant | Detail |
|-----------|--------|
| Range | `0 < r < n` and `0 < s < n` |
| Low-S | `ufsecp_ecdsa_sign()` always produces low-S (`s <= n/2`) per BIP-62/BIP-146 |
| DER | `ufsecp_ecdsa_sig_to_der()` produces strict DER; `ufsecp_ecdsa_sig_from_der()` rejects non-strict |
| Compact | `ufsecp_ecdsa_sign()` output is `R || S` (64 bytes); `ufsecp_ecdsa_verify()` accepts only this |

### Schnorr / BIP-340

| Invariant | Detail |
|-----------|--------|
| x-only pubkey | 32 bytes, even-Y convention (BIP-340 sec 3) |
| Tagged hash | `hash = SHA256(SHA256("BIP0340/challenge") || SHA256("BIP0340/challenge") || R || P || m)` |
| No malleability | Single valid signature per `(key, message)` pair (deterministic nonce per BIP-340) |
| Strict mode | With `UFSECP_BITCOIN_STRICT=ON` (default): rejects `r >= p` and `s >= n` |

---

## 5. Hash Domain Separation

| Context | Tag / Prefix |
|---------|-------------|
| BIP-340 challenge | `"BIP0340/challenge"` |
| BIP-340 aux | `"BIP0340/aux"` |
| BIP-340 nonce | `"BIP0340/nonce"` |
| BIP-32 HMAC | `"Bitcoin seed"` |
| Taproot tweak | `"TapTweak"` |
| ECDSA (RFC 6979) | No tag -- uses raw SHA-256 of message hash |

### Misuse: Wrong Hash

```
// WRONG: passing a message directly to sign (expects 32-byte hash)
ufsecp_ecdsa_sign(ctx, key, message, sig);  // silent misbehavior if len(message) != 32

// CORRECT: hash first, then sign
ufsecp_sha256(ctx, message, msg_len, hash);
ufsecp_ecdsa_sign(ctx, key, hash, sig);
```

---

## 6. Point Arithmetic

| Invariant | Detail |
|-----------|--------|
| Curve equation | `y^2 = x^3 + 7 (mod p)` where `p = 2^256 - 2^32 - 977` |
| Group order | `n = FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141` |
| Cofactor | `h = 1` (no cofactor pitfalls unlike Ed25519) |
| Generator | Fixed `G` point per SEC 2 / BIP-340 |
| Infinity | `k*G = O` iff `k = 0 (mod n)` |
| Negation | `-(x, y) = (x, p - y)` |
| No small subgroups | Cofactor 1 means every non-infinity point generates the full group |

---

## 7. Key Tweaking (Taproot / BIP-32)

| Invariant | Detail |
|-----------|--------|
| Tweak range | `0 <= t < n`; tweak `t >= n` is rejected |
| Result validation | `key + t*G` must not be infinity; if so, the tweak is rejected |
| BIP-32 hardened | Index `>= 0x80000000` uses private key in HMAC input |
| BIP-32 normal | Index `< 0x80000000` uses public key in HMAC input |
| Neutered keys | `xpub` derivation cannot derive hardened children |

### Misuse: Tweak Without Validation

```
// WRONG: assuming tweak always succeeds
ufsecp_seckey_tweak_add(ctx, key, tweak);

// CORRECT: check return value
if (ufsecp_seckey_tweak_add(ctx, key, tweak) != UFSECP_OK) {
    // tweak resulted in invalid key (extremely rare but possible)
}
```

---

## 8. Serialization Byte Order

| Context | Endianness |
|---------|-----------|
| Public API (`uint8_t[]`) | **Big-endian** (network byte order, per SEC 1 / BIP-340) |
| Internal FieldElement limbs | **Little-endian** (host-native 64-bit limbs) |
| Precompute cache file | **Little-endian** (raw struct write, platform-specific) |
| Test vectors (hex strings) | **Big-endian** (standard crypto convention) |

Cache files are **not portable** across different endianness architectures.
The library detects mismatched cache files via magic number validation and
rebuilds automatically.

---

## 9. Thread Safety

| Rule | Detail |
|------|--------|
| Context isolation | Each `ufsecp_ctx` must be used by exactly one thread |
| No global state | All state is in the context (except the precompute table, which is read-only after init) |
| Precompute table | Built once at first use (thread-safe via internal mutex), then immutable |
| Cloning | `ufsecp_ctx_clone()` creates an independent copy for another thread |

---

## 10. Constant-Time Guarantees

| Operation | Layer | CT Property |
|-----------|-------|-------------|
| `ufsecp_ecdsa_sign` | CT | No secret-dependent branches or memory access |
| `ufsecp_schnorr_sign` | CT | No secret-dependent branches or memory access |
| `ufsecp_ecdh` | CT | No secret-dependent branches or memory access |
| `ufsecp_seckey_tweak_add` | CT | No secret-dependent branches or memory access |
| `ufsecp_ecdsa_verify` | Fast | Variable-time (public inputs only) |
| `ufsecp_schnorr_verify` | Fast | Variable-time (public inputs only) |
| `ufsecp_pubkey_create` | CT | Secret key used in CT scalar_mul |

### Verification Methods

1. **Compile-time:** LLVM ct-verif pass (deterministic proof)
2. **Runtime:** Valgrind taint tracking (`-DVALGRIND_CT_CHECK=1`)
3. **Statistical:** dudect (Welch t-test, |t| > 4.5 = leak)
4. **CI enforcement:** All three run on every PR to `main`
