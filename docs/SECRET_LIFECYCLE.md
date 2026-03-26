# Secret Lifecycle Review

**Last updated**: 2026-03-15 | **Version**: 3.22.0

Documents how secret material (private keys, nonces, session state) is handled throughout its lifecycle: creation, use, and destruction.

Secret-lifecycle and zeroization changes are under stricter change control:
`scripts/check_secret_path_changes.py` requires paired updates to this document
and `docs/SECURITY_CLAIMS.md` whenever those surfaces change.

---

## Zeroization Infrastructure

### `detail::secure_erase(void*, size_t)`

Located in `cpu/include/secp256k1/detail/secure_erase.hpp`. Three-tier implementation:

| Platform | Method | Barrier |
|----------|--------|---------|
| MSVC | Volatile write loop (SecureZeroMemory pattern) | `atomic_signal_fence(seq_cst)` |
| glibc 2.25+ / BSD | `explicit_bzero()` | `atomic_signal_fence(seq_cst)` |
| Fallback | Volatile function-pointer to `memset` | `atomic_signal_fence(seq_cst)` |

The `atomic_signal_fence` prevents LTO/IPO from eliding the write.

### `PrivateKey` RAII wrapper

`cpu/include/secp256k1/private_key.hpp` -- destructor calls `secure_erase()` on the key material. Prevents key leaks from forgotten cleanup.

---

## Coverage by Module

### C ABI Layer (`ufsecp_impl.cpp`) -- ~75 secure_erase calls

Every function that touches secrets erases them on all exit paths:

| Category | What's erased | Functions |
|----------|--------------|-----------|
| ECDSA/Schnorr sign | `sk` (parsed secret key) | `ufsecp_ecdsa_sign`, `_sign_verified`, `_sign_recoverable`, `ufsecp_schnorr_sign`, `_sign_verified` |
| BIP-32 derivation | `ek.key`, `ek.chain_code`, child keys | `ufsecp_bip32_master`, `_derive`, `_derive_path`, `_privkey` |
| ECDH | `sk`, shared secret | `ufsecp_ecdh`, `_xonly`, `_raw` |
| Key tweaks | `sk`, `tw`, `result` | `ufsecp_seckey_tweak_add`, `_tweak_mul`, `_negate` |
| MuSig2 | `sk`, `sn` (sec nonce), secnonce buffer | `ufsecp_musig2_nonce_gen`, `_partial_sign` |
| FROST | `signing_share`, `hiding_nonce`, `binding_nonce`, `seed_arr`, `h`, `b` | `ufsecp_frost_keygen_begin`, `_keygen_finalize`, `_sign_nonce_gen`, `_sign` |
| Silent Payments | `scan_sk`, `spend_sk` | `ufsecp_silent_payment_*` |
| ECIES | `privkey` | `ufsecp_ecies_decrypt` |
| Ethereum | `sk` | `ufsecp_eth_sign` |

### ECDSA Fast Path (`cpu/src/ecdsa.cpp`) -- ~18 calls

Erases: `V`, `K` (HMAC-DRBG state), `x_bytes`, `buf97`/`buf129` (RFC 6979 intermediates), `k` (nonce), `k_inv`, `z` (message scalar).

### CT ECDSA Sign (`cpu/src/ct_sign.cpp`) -- 9+8 calls

`schnorr_sign`: Exemplary -- erases `d_bytes`, `t_hash`, `t`, `nonce_input`, `rand_hash`, `challenge_input`, `k_prime`, `k` (9 calls).

`ecdsa_sign` / `ecdsa_sign_hedged`: Erases `k`, `k_inv`, `z`, `s` before return (fixed 2026-03-15).

### MuSig2 (`cpu/src/musig2.cpp`) -- 3+2 calls

`musig2_nonce_gen`: Erases `sk_bytes`, `aux_hash`, `t`.

`musig2_partial_sign`: Erases `k` (effective nonce) and `d` (adjusted signing key) before return (fixed 2026-03-15).

### FROST (`cpu/src/frost.cpp`) -- 4 calls (added 2026-03-15)

`frost_keygen_begin`: Erases polynomial coefficients vector after share generation.

`frost_sign`: Erases `d` (hiding nonce), `ei` (binding nonce), `s_i` (signing share) before return.

### ECIES (`cpu/src/ecies.cpp`) -- 13 calls

Erases: `shared_x` (ECDH raw), `kdf` (64B enc+mac keys), `eph_privkey`, `eph_bytes`, AES key schedule `W[240]`, AES CTR `keystream[16]`, HMAC pads (`k_pad`, `ipad`, `opad`).

### BIP-39 (`cpu/src/bip39.cpp`) -- 4 calls

Erases: entropy buffers after mnemonic generation and seed derivation.

### ECDH (`cpu/src/ecdh.cpp`) -- 2 calls

Erases: compressed point representation, `x_bytes` after shared secret derivation.

---

## Secret Classification

| Secret type | Lifetime | Cleanup location | CT path? |
|-------------|----------|-----------------|----------|
| Private key (32B) | Caller-owned | C ABI wrapper + internal | CT sign |
| ECDSA nonce k | Function-local | `ecdsa.cpp` / `ct_sign.cpp` | CT ECDSA |
| Schnorr nonce k | Function-local | `ct_sign.cpp` | CT Schnorr |
| MuSig2 sec nonce (k1, k2) | Session-scoped | C ABI wrapper | CT partial_sign |
| MuSig2 effective nonce k | Function-local | `musig2.cpp` | CT path |
| FROST polynomial coeffs | Function-local | `frost.cpp` | CT gen_mul |
| FROST nonces (d, ei) | Function-local | `frost.cpp` | Cleared on return |
| FROST signing share | Key pkg member | C ABI wrapper | Cleared on return |
| ECDH shared secret | Function-local | `ecdh.cpp` + C ABI | CT mul |
| ECIES derived keys | Function-local | `ecies.cpp` | AES-CBC key schedule |
| BIP-32 chain code | Derived state | C ABI wrapper | HMAC-SHA512 |
| BIP-39 entropy | Function-local | `bip39.cpp` | Zeroized after use |
| RFC 6979 HMAC state | Function-local | `ecdsa.cpp` | V, K buffers |

---

## Design Principles

1. **Defense in depth**: Both the internal function AND the C ABI wrapper erase secrets. Double erase is intentional -- the internal function erases its locals, and the wrapper erases its parsed copies.

2. **All exit paths**: Early returns (validation failures) happen before secret material is computed, so no cleanup is needed on those paths.

3. **Stack vs heap**: Stack secrets use `secure_erase(&var, sizeof(var))`. Heap secrets (FROST coefficients vector) iterate and erase each element.

4. **No secret in return value**: Functions return public values (signatures, commitments). Secret intermediates are never returned.
