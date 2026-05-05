# Secret Lifecycle Review

**Last updated**: 2026-05-05 | **Version**: 3.70.0

### 2026-05-05 Secret Lifecycle Changes (A-06 perf fix)

- `ecdsa.cpp` (`ecdsa_sign`): Replaced `R.x_only_bytes()` + `Scalar::from_bytes(r_bytes)`
  with `Scalar::from_limbs(R.x().limbs())`. The value `r = R.x mod n` is the public
  x-coordinate of nonce point R = k·G. It is NOT secret material — no zeroization is
  required or performed for `r`. Secret nonce `k` and its inverse `k_inv` continue to be
  erased via `secure_erase` before return. No change to secret zeroization paths.
- `ct_sign.cpp` (`ct::ecdsa_sign`): Same optimisation for the CT path. Secret nonce `k`
  and `k_inv` zeroization is unchanged.

- `musig2.cpp`: Clarified comment in `musig2_partial_sign` for out-of-range
  `signer_index` case — returns `Scalar::zero()` (caller ABI returns
  `UFSECP_ERR_BAD_INPUT`). No functional change.
- CUDA bench: Private key VRAM (`d_priv`) is now zeroed via `cudaMemset`
  before `cudaFree`. Host-side `h_privkeys` vector zeroed via volatile loop.
- OpenCL bench: Same pattern applied to `bench_compare.cu` sign functions.

Documents how secret material (private keys, nonces, session state) is handled throughout its lifecycle: creation, use, and destruction.

Secret-lifecycle and zeroization changes are under stricter change control:
`ci/check_secret_path_changes.py` requires paired updates to this document
and `docs/SECURITY_CLAIMS.md` whenever those surfaces change.

---

## 2026-04-07 CI Hardening Notes

1. **MSan added to CI sanitizer matrix.** MemorySanitizer (`-fsanitize=memory -fsanitize-memory-track-origins=2`) now runs alongside ASan and TSan. This catches use-of-uninitialized-memory on secret-bearing paths — a direct complement to zeroization enforcement.
2. **Crash-risk analysis** is now a first-class audit-gate check (`check_crash_risks`). Division-by-zero and other crash risks in CT-sensitive functions are flagged automatically.
3. **dudect PR gate** runs a 60 s smoke test on every pull request; the full 30 min statistical analysis runs on push to `dev`/`main`.
4. **Coverage upload** now fails CI on error (`fail_ci_if_error: true`).
5. **CT scalar_inverse(0) zero guard.** Both the SafeGCD and Fermat fallback constant-time scalar inverse paths in `src/cpu/src/ct_scalar.cpp` now return `Scalar::zero()` for zero input. Previously, only the FAST-path `Scalar::inverse()` had this guard. This is a defense-in-depth fix: `ct::scalar_inverse(0)` was undefined behavior, theoretically reachable if a caller passed a zero scalar to a CT signing path. Verified by `test_exploit_boundary_sentinels` BS-1 and BS-10.

## 2026-04-14 FROST Secret-Path Notes

1. `src/cpu/src/frost.cpp` signing, partial verification, and aggregation no longer materialize temporary signer-ID or binding-factor heap vectors. Group commitment and Lagrange derivation now traverse `FrostNonceCommitment` entries directly.
2. This change reduces transient secret-adjacent heap residency during FROST signing without expanding the secret surface. The same secret values remain in scope: `signing_share`, `hiding_nonce`, `binding_nonce`, and the signer-local derived scalars used for the final partial signature.
3. Cleanup guarantees are unchanged: C ABI wrappers still erase parsed secret-bearing state on return, and `frost_sign()` still zeroizes the local copies of hiding nonce, binding nonce, and signing share before exit.

---

## 2026-04-06 Change-Control Notes

Recent secret-path evidence hardening relevant to this document:

1. `src/cpu/src/schnorr.cpp` now keeps verify-side x-only cache entries normalized before storage. Those cached points are derived only from public verification inputs and do not persist secret scalars, nonce material, or secret-derived intermediates.
2. `audit/test_ct_sidechannel.cpp` and `audit/test_ct_verif_formal.cpp` remain paired CT-evidence surfaces for secret-bearing regressions. Current interpretation explicitly treats allocator/address-layout bias as a harness artifact, not as proof of secret dependence, unless the signal survives layout de-correlation.

---

## Zeroization Infrastructure

### `detail::secure_erase(void*, size_t)`

Located in `src/cpu/include/secp256k1/detail/secure_erase.hpp`. Three-tier implementation:

| Platform | Method | Barrier |
|----------|--------|---------|
| MSVC | Volatile write loop (SecureZeroMemory pattern) | `atomic_signal_fence(seq_cst)` |
| glibc 2.25+ / BSD | `explicit_bzero()` | `atomic_signal_fence(seq_cst)` |
| Fallback | Volatile function-pointer to `memset` | `atomic_signal_fence(seq_cst)` |

The `atomic_signal_fence` prevents LTO/IPO from eliding the write.

### `PrivateKey` RAII wrapper

`src/cpu/include/secp256k1/private_key.hpp` -- destructor calls `secure_erase()` on the key material. Prevents key leaks from forgotten cleanup.

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

### ECDSA Fast Path (`src/cpu/src/ecdsa.cpp`) -- ~18 calls

Erases: `V`, `K` (HMAC-DRBG state), `x_bytes`, `buf97`/`buf129` (RFC 6979 intermediates), `k` (nonce), `k_inv`, `z` (message scalar).

### CT ECDSA Sign (`src/cpu/src/ct_sign.cpp`) -- 9+8 calls

`schnorr_sign`: Exemplary -- erases `d_bytes`, `t_hash`, `t`, `nonce_input`, `rand_hash`, `challenge_input`, `k_prime`, `k` (9 calls).

`ecdsa_sign` / `ecdsa_sign_hedged`: Erases `k`, `k_inv`, `z`, `s` before return (fixed 2026-03-15).

### MuSig2 (`src/cpu/src/musig2.cpp`) -- 3+2 calls

`musig2_nonce_gen`: Erases `sk_bytes`, `aux_hash`, `t`.

`musig2_partial_sign`: Erases `k` (effective nonce) and `d` (adjusted signing key) before return (fixed 2026-03-15).

### FROST (`src/cpu/src/frost.cpp`) -- 4 calls (added 2026-03-15)

`frost_keygen_begin`: Erases polynomial coefficients vector after share generation.

`frost_sign`: Erases `d` (hiding nonce), `ei` (binding nonce), `s_i` (signing share) before return.

2026-04-14 note: signer-set validation and binding-factor derivation no longer allocate
temporary signer-ID / binding-factor vectors, reducing transient heap copies on
secret-bearing signing flows.

### ECIES (`src/cpu/src/ecies.cpp`) -- 13 calls

Erases: `shared_x` (ECDH raw), `kdf` (64B enc+mac keys), `eph_privkey`, `eph_bytes`, AES key schedule `W[240]`, AES CTR `keystream[16]`, HMAC pads (`k_pad`, `ipad`, `opad`).

### BIP-39 (`src/cpu/src/bip39.cpp`) -- 4 calls

Erases: entropy buffers after mnemonic generation and seed derivation.

### ECDH (`src/cpu/src/ecdh.cpp`) -- 2 calls

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
| FROST signer-set scratch | Derived/public transcript data | `frost.cpp` | Reduced in 2026-04-14 refactor |
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
