# Secret Lifecycle Review

**Last updated**: 2026-05-11 | **Version**: 4.0.0

### 2026-05-11 SHIM-007 — musig2_nonce_agg_points (PUBLIC-DATA path, no secret material)

- **`src/cpu/src/musig2.cpp` (`musig2_nonce_agg_points`)**: New function. Sums pre-decompressed
  (R1, R2) Point pairs — these are **public nonces only**. No private key, secret scalar, or secret
  nonce material flows through this function. The Points are affine-cached representations of
  signer public nonces that are broadcast to all participants. Zero secret lifecycle changes.
  Erase requirement: N/A — no secrets present.

### 2026-05-11 Secret Lifecycle Changes — 10-pass review P1/P2 fixes

- **`src/cpu/src/ct_sign.cpp` (`ct::ecdsa_sign_hedged_recoverable`)**: New function.
  Secret material erased before all return paths: nonce `k`, `k_inv`, `z`, `s`, `pre_sig`.
  Pattern mirrors `ct::ecdsa_sign_recoverable` (line 374–378) which was audited 2026-05-07.
- **`src/cpu/src/frost.cpp` (`derive_scalar`, `derive_scalar_pair`)**: Helper function
  `derive_scalar_from_hash` added — erases the hash copy in the loop before return via
  `secure_erase(hash.data(), hash.size())`. Replaces `Scalar::from_bytes` (silent mod-n
  reduction) with `parse_bytes_strict_nonzero` + counter retry to prevent zero polynomial
  coefficients and nonces from being silently accepted.
- **`src/cpu/src/bip324.cpp` (`Bip324Session::complete_handshake`)**: ECDH shared-secret
  all-zero check replaced with constant-time byte accumulator. Not a secret lifecycle change
  per se, but a CT boundary fix on the ECDH path. The `secure_erase(&sk, sizeof(sk))` before
  early return was already present.

### 2026-05-07 Secret Lifecycle Changes — multi-agent ultrareview zeroization audit

- **`recovery.cpp` (`secp256k1::ecdsa_sign_recoverable`, public C++ API)**: Added
  `detail::secure_erase` calls for `k`, `k_inv`, `r_times_d`, `z_plus_rd`, and `s` before
  all return paths. The CT ABI path (`ct::ecdsa_sign_recoverable` in `ct_sign.cpp`) was
  already clean. Direct C++ callers of the lower API were exposed.
  Erased variables: nonce `k` (SECRET), k inverse `k_inv` (SECRET), r*d product (SECRET),
  z+r*d sum (SECRET), signature scalar `s` (SECRET before publication).
- **`bip32.cpp` (`ExtendedKey::derive_child`)**: `child_scalar` (child private key) was
  serialized into `child.key` but never erased from the local variable. Added
  `detail::secure_erase(&child_scalar, sizeof(child_scalar))` after `to_bytes()`. Also
  added erasure on the `is_zero` early-return path that previously skipped it entirely.
- **`adaptor.cpp` (`adaptor_nonce`)**: Nonce domain separation hardened to BIP-340 tagged-hash
  prefix pattern. This affects nonce derivation (not zeroization), but it is a secret-path
  security boundary change. Domain-first prevents cross-protocol nonce collision.
- **`ecdsa.cpp` (`HMAC_Ctx::compute_short`)**: Runtime guard + `assert(msg_len <= 55)` added.
  Without the guard, `msg_len > 55` causes `size_t` unsigned wrap in
  `std::memset(block + msg_len + 1, 0, 55 - msg_len)` — potential stack smash that could
  corrupt secret material on the stack.

### 2026-05-06 Secret Lifecycle Changes (CA-build-fix — ecdsa.cpp always_inline removed)

- **`ecdsa.cpp` (`ECDSASignature::is_low_s_ct`)**: Removed `__attribute__((always_inline))`
  which GCC-13 rejected with `-Werror=attributes` when the function body was too complex to
  inline. No change to cryptographic logic, constant-time properties, or secret paths. The
  function body is unchanged; only the compiler hint is removed.

### 2026-05-06 Secret Lifecycle Changes (ultrareview TASK-04/05 — is_zero_ct + adaptor const_cast UB)

- **`recovery.cpp` (`ecdsa_sign_recoverable`)**: `private_key.is_zero()` and `k.is_zero()`
  replaced with `private_key.is_zero_ct()` and `k.is_zero_ct()`. The VT `is_zero()` method
  returns early on the first non-zero limb, leaking timing information about the magnitude of
  the secret scalar. `is_zero_ct()` reads all 4 limbs unconditionally, eliminating this oracle.
  No change to zeroization paths (nonce `k` erased after use via caller chain).
- **`ecdh.cpp` (`ecdh_compute`, `ecdh_compute_xonly`, `ecdh_compute_raw`)**: All three ECDH
  variants had `private_key.is_zero()` → `private_key.is_zero_ct()`. Same rationale: the
  early-exit path in VT `is_zero()` reveals whether the private key is small (many leading
  zero limbs). The CT variant reads all limbs before deciding. `shared_point` erased on all
  paths after `ct::scalar_mul` (unchanged).
- **`adaptor.cpp` (`ecdsa_adaptor_sign`)**: `Scalar const k` and `Scalar const binding`
  declarations changed to `Scalar k` / `Scalar binding`. Calling `secure_erase` on a
  `const`-qualified object via `const_cast<Scalar*>` is C++ UB — LTO / `-O3` may observe
  that the object is "never modified" and elide the zeroization entirely, leaving the secret
  nonce `k` live in memory after the function returns. The fix removes `const`; zeroization
  calls now operate on non-const lvalues and cannot be optimized away.

### 2026-05-06 Secret Lifecycle Changes (ultrareview P1/P2 — BIP32 strict parse + FROST CT inverse)

- **`bip32.cpp` (`ExtendedKey::derive_child`)**: Replaced `Scalar::from_bytes(key)` (reducing,
  accepts key ≥ n silently) with `Scalar::parse_bytes_strict_nonzero(key, parent_scalar)`.
  Returns `{ExtendedKey{}, false}` for key ≥ n or key == 0 instead of silently corrupting the
  derivation. Added `secure_erase(&parent_scalar, sizeof(parent_scalar))` on both exit paths
  (early failure and after `ct::scalar_add`) — parent private key scalar fully erased before
  function returns. `il_scalar` and `I`/`IL` buffers erased on all paths (unchanged).
- **`frost.cpp` (`frost_lagrange_coefficient`)**: `den.inverse()` (variable-time GCD) replaced
  with `ct::scalar_inverse(den)`. `num * den.inverse()` replaced with
  `ct::scalar_mul(num, ct::scalar_inverse(den))`. Zero-denominator guard added before inverse.
  Same fix applied to `frost_lagrange_coefficient_from_commitments`. Lambda_i is public but
  the operation now runs in constant time, eliminating a correlated timing oracle on total
  signing time when adversary controls participant ID sets (CT-002).

### 2026-05-06 Secret Lifecycle Changes (ultrareview P2 batch — FROST O(n²) fix)

- **`frost.cpp` (`compute_group_commitment_inline_binding`)**: Refactored to
  precompute all `to_compressed()` serializations once before the binding-factor
  loop. This reduces field inversions from O(n²) to O(n). The precomputed
  `FrostCommitmentSerialized` struct holds only compressed public nonce points
  (hiding_point, binding_point) — these are NOT secret material. Secret keys and
  signing shares are not involved in this function. No change to zeroization paths.

### 2026-05-06 Secret Lifecycle Changes (ultrareview P1/P2 batch)

- **`musig2.cpp` (`musig_nonce_gen` k2 fix)**: k2 HMAC now uses
  `cached_tagged_hash(g_musig_nonce_midstate, ...)` matching k1. Both k1/k2 now derive
  under the same domain separator — eliminates tag-mismatch risk. Secret nonce material
  (k1, k2) already erased via `secure_erase` on all exit paths; no change to zeroization.
- **`musig2.cpp` (`musig2_partial_sig_agg`)**: Added `s.is_zero_ct() || R.is_infinity()`
  fail-closed check before serializing the final signature. Degenerate aggregated s=0
  returns all-zero array instead of silently serializing an invalid signature. No secret
  material involved — s is the sum of public partial signatures.
- **`frost.cpp` (`frost_aggregate`)**: Same s==0 fail-closed check added. Accumulation now
  uses `ct::scalar_add` (was `operator+`) for consistency with CT discipline even though
  partial sig accumulation is not secret-bearing in the aggregator role.
- **`frost.cpp` (DKG signing_share accumulation)**: `signing_share += share.value` changed
  to `ct::scalar_add` — signing_share is secret key material. Zeroization via caller
  (keypkg struct) remains unchanged.
- **`bip324.cpp` (constructor, ephemeral key)**: Replaced `Scalar::from_bytes(privkey)` with
  `Scalar::parse_bytes_strict_nonzero` in both CSPRNG and caller-supplied paths. Retry loop
  added for CSPRNG path (probability ≈ 2^-128). `secure_erase(&sk)` on all paths unchanged.
- **`bip324.cpp` (`complete_handshake`)**: Same strict parsing for stored privkey. Returns
  false immediately if stored key fails strict check (invariant: should not occur with
  correct constructor usage). `sk` stack variable erased after ECDH (unchanged).

### 2026-05-05 Secret Lifecycle Changes (perf audit P1/P2 batch)

- `ecdsa.cpp` (`ecdsa_sign_hedged`): Replaced `R.x_only_bytes()` + `Scalar::from_bytes(r_bytes)`
  with `Scalar::from_limbs(R.x().limbs())` — same fix as `ecdsa_sign` (A-06 above). `r` is the
  public x-coordinate of R = k·G. NOT secret material. Secret nonce `k` and `k_inv` continue to
  be erased via `secure_erase`. No change to secret zeroization paths.
- `ecdsa.cpp` (`rfc6979_nonce_hedged`): Hoisted `std::array<uint8_t, 32> t` and `uint8_t buf33[33]`
  before the retry loop. Added `secure_erase(t.data(), t.size())` on the early-return path to
  match the pattern already used in `rfc6979_nonce`. `t` is a temporary holding HMAC-DRBG output
  before parsing as a candidate scalar — classified as secret-adjacent material; zeroization is
  maintained on all exit paths (early return and loop-exhaustion fallthrough).

### 2026-05-05 Secret Lifecycle Changes (A-06, A-15 perf fixes)

- `ecdsa.cpp` (pubkey parse, A-15): `y.to_bytes()[31] & 1` → `y.limbs()[0] & 1u` for
  Y-parity check in compressed pubkey parsing. `y` is the result of `sqrt()` on a
  public x-coordinate — it is NOT secret material. No zeroization path is affected.

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
