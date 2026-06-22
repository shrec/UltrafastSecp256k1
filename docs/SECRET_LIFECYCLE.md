# Secret Lifecycle Review

**Last updated**: 2026-06-17 | **Version**: 4.4.0

### 2026-06-22 - CPU batch-verify pool + decompress: verify-path only, no secret lifecycle change

The persistent worker pool (`detail::batch_worker_pool`), the fused parse+verify in
`ufsecp_ecdsa_verify_opaque_rows_mt`, and the FE52/point-only pubkey decompress
(`pubkey33_to_point`) added on this date touch the unity root
(`src/cpu/src/ufsecp_impl.cpp`) and the ECDSA/Schnorr **batch-verify** paths only. These
process exclusively PUBLIC data (public keys, signatures, message hashes) — no private key,
nonce, or signing share flows through them. No secret zeroization surface, key-erasure
sequence, or `secure_erase` call site was added, removed, or reordered. The pool worker
`thread_local` scratch holds only `ECDSABatchEntry` (public points/signatures), never secrets.

### 2026-06-18 - Bech32 address encoder allocation reduction: public-data only

The `bech32_encode` allocation-reduction change in `src/cpu/src/address.cpp`
uses a stack-backed 5-bit conversion buffer and streams the Bech32 checksum for
Bitcoin witness address encoding. This path receives only PUBLIC address data:
the network HRP, witness version, and witness program/hash. It does not process
private keys, nonces, signing scalars, seed material, or caller-owned secret
buffers.

No secret lifecycle or zeroization contract changed. `secure_erase` sites were
not added, removed, or reordered; signing, BIP-32/BIP-39, MuSig2, FROST, ECIES,
and wallet secret cleanup paths remain unchanged. The stack buffer contains only
public 5-bit address groups and requires no secret scrubbing.

### 2026-06-17 - Batch verify MT + cache-dir API: no new secret surface

The bridge-free batch-verify work (`schnorr_batch_verify_mt`, the smart shim
batch path, and the `secp256k1_*_verify_batch_{mt,results}` symbols) and the
programmatic cache-directory API (`ufsecp_set_cache_dir` /
`secp256k1::fast::set_cache_directory`) introduce **no new secret material and no
new zeroization surface**. Batch verification operates only on PUBLIC data
(pubkey/message/signature); multi-threading is a pure throughput change with no
secret-dependent branch (same class as `ecdsa_batch_verify_mt`). The
`ufsecp_set_cache_dir` ABI takes a directory string (public) and stores it; it
never touches keys. The only change in `src/cpu/src/ufsecp_impl.cpp` for this
work is an additional unity-build `#include` to expose `set_cache_directory` — no
secret-lifecycle logic was added or altered. No `secure_erase` sites were added
or removed; existing seckey/nonce erasure paths (`ecdsa_sign`, `schnorr`,
`musig2`, `frost`, BIP-32) are unchanged.

### 2026-06-11 - MuSig2 partial-sign secret-derived stack residue erasure

`musig2_partial_sign` (`src/cpu/src/musig2.cpp`) now `secure_erase`s three
secret-derived intermediates that previously lingered on the stack: `neg_k = -k`
(secret nonce), `neg_d = -d` (secret signing key), and `ead = ea*d` (carries the
secret key; `ea = e*a_i` itself is public). Previously only `k`, `d`, and the
caller's `sec_nonce.k1/k2` were erased. Same class as `FROST-SIGN-RESIDUE`; no
timing/CT change (all branchless `ct::`), purely stack-scrubbing hygiene. **Found
by the improved `ci/dev_bug_scanner.py` secret-derived-unerased check** (which now
scans `frost.cpp`/`musig2.cpp` and handles `Scalar const` decls — the exact gaps
that hid the FROST residue). Regression: `regression_secret_scalar_residue_erase`.

### 2026-06-10 - FROST/keypair secret-derived stack residue erasure (FROST-SIGN-RESIDUE)

`frost_sign` (`src/cpu/src/frost.cpp`) now `secure_erase`s the secret-derived intermediate
products `rho_ei = my_binding·ei` and `lambda_s_e = lambda_i·s_i·e` (both changed from
`const` to non-const so they can be scrubbed). Both carry the **secret** FROST binding nonce
`ei` and signing share `s_i`; previously only `d`, `ei`, `s_i`, and the caller's nonce halves
were erased, leaving these two products as stack residue after return. `z_i` (the partial-sig
output) is public and is correctly not erased. In addition, `schnorr_keypair_create`
(`src/cpu/src/ct_sign.cpp` and `src/cpu/src/schnorr.cpp`) now `secure_erase`s the private-key
copy `d_prime` after `kp.d`/`kp.px` are derived. No timing/CT behavior changes — all
arithmetic is branchless `ct::`; this is secret-erasure hygiene of the same class as
`T08-SCALAR-ERASE` (`ecdsa_sign`/`musig2_partial_sig_agg`). Exploitation would require a
separate stack-memory-disclosure primitive. Regression test
`regression_secret_scalar_residue_erase` source-scans all three sites and round-trips
Schnorr `keypair_create`+`sign`+`verify` to confirm the `d_prime` erase does not corrupt the
returned keypair.

### 2026-06-08 - MuSig2 infinity aggregate-nonce conformance — secret handling unchanged

`musig2_start_sign_session` now accepts an infinity aggregate-nonce half and substitutes
`R = G` when the effective nonce is infinity (BIP-327 conformance). All affected values are
PUBLIC: the aggregate nonce, the binding factor `b`, the session nonce `R`, and the
challenge `e` are computed from public inputs only. No secret material enters this path and
no `secure_erase`/lifecycle behavior changes — `musig2_partial_sign`'s signing-key and
nonce erasure are untouched.

### 2026-06-08 - MuSig2 binding-factor tag fix — secret handling unchanged

`musig2_start_sign_session` now derives the nonce binding factor `b` with the BIP-327
tag `"MuSig/noncecoef"` (was `"MuSig/nonceblinding"`). `b` is a PUBLIC value computed from
the public aggregate nonce, the public aggregate pubkey `Q_x`, and the public message;
no secret material enters the binding hash. The change is a tagged-hash tag string only —
the signing-key and nonce `secure_erase` paths in `musig2_partial_sign` are untouched.
No lifecycle impact.

### 2026-06-08 - MuSig2 BIP-327 tweak fix (gacc/tacc) — secret handling unchanged

The MuSig2 tweaked-signing fix added BIP-327 `gacc`/`tacc` accumulators to the keyagg
cache and folds them into signing. Secret lifecycle is **unaffected**: `gacc`/`tacc` are
PUBLIC (derived from public pubkeys + public tweaks). In `musig2_partial_sign` the signing
key `d` is still `secure_erase`d after use — the new `d = ct::scalar_mul(d, gacc)` multiply
by a public ±1 occurs before the existing erase of `d`, `k`, and the caller's secret nonce.
No secret is added to the keyagg cache or the session; `tweak_s = e·g·tacc` is public.

### 2026-06-06 - CPU ABI/shim fail-closed refresh and BIP32/BIP39 cleanup

`src/cpu/src/bip32.cpp` now erases the temporary strict-parsed private
scalar in `ExtendedKey::public_key()` on both invalid-key and success paths.
`bip32_master_key()` erases `I`, `IL`, `IR`, the parsed `master_key`, and the
temporary serialized `master_bytes` after copying the final master key and
chain code into the returned `ExtendedKey`; the returned key remains caller
owned and is not scrubbed. On strict-parse failure the same HMAC output buffers
and scalar are erased before returning `{ExtendedKey{}, false}`.

`src/cpu/src/bip39.cpp` `pbkdf2_hmac_sha512()` now erases each per-block
`salt_block` after the block output is copied. The existing `result` and `u`
erasure remains unchanged, so password-derived intermediate material no longer
survives across PBKDF2 block iterations.

The CPU ABI hardening in `src/cpu/src/impl/ufsecp_musig2.cpp`,
`src/cpu/src/impl/ufsecp_taproot.cpp`, `src/cpu/src/impl/ufsecp_coins.cpp`,
`src/cpu/src/eth_signing.cpp`, and the libsecp256k1 shim refresh is
fail-closed: result buffers are zero-initialized before secret-bearing work or
before parser dispatch, and internal zero/invalid signing results now return a
non-OK error instead of serializing a zero signature/key as success. Unsupported
custom nonce callbacks in the shim are rejected before signing dispatch and
before output clearing, matching upstream-style pre-dispatch argument rejection.
FROST signing continues to consume and erase caller nonce material on every
non-NULL exit path; FROST aggregate rejects zero partial scalars and all-zero
aggregate results before exposing a signature.

Validation: `unified_audit_runner --section protocol_security`,
`--section standard_vectors`, `--section memory_safety`, `--section
ct_analysis`, `--section fuzzing`, `--section exploit_poc`, shim CTest targets,
and `ci/check_exploit_wiring.py`.

### 2026-06-06 - GPU ABI fail-closed outputs and secret-buffer erasure parity

`src/cpu/src/ufsecp_gpu_impl.cpp` now clears result-bearing GPU C ABI output
buffers before backend dispatch and clears them again after non-OK backend
returns. This applies to generator multiplication, batch verification, ECDH,
Hash160, MSM, FROST partial verification, ecrecover, ZK verification,
Bulletproof verification, BIP-324 AEAD, SNARK witness helpers, and BIP-352
scan. The in-place collect APIs are intentionally excluded because their
`key_buffer` is caller-owned marker state used for CPU fallback.

`ufsecp_bip352_prepare_scan_plan()` and `ufsecp_gpu_bip352_scan_batch()` now
strict-parse the scan private key with `parse_bytes_strict_nonzero`; zero and
order-or-larger keys fail with `UFSECP_ERR_BAD_KEY` / `GpuError::BadKey` and
leave the plan or prefix outputs all-zero. CUDA, OpenCL, and Metal BIP-352 scan
paths now reject invalid scan keys consistently. CUDA validates decompressed
spend/tweak points before kernel use and guards device buffers so early returns
zero key-derived device memory. OpenCL wraps the host `Bip352ScanPlan` and
device `d_plan` in erasing guards, covering success and every error path. Metal
wraps BIP-352 scan scalar buffers, ECDH scratch scalars, and BIP-324 AEAD shared
key buffers with erase-on-exit guards.

Validation: `audit/test_gpu_bip352_scan.cpp`,
`audit/test_gpu_host_api_negative.cpp`,
`audit/test_exploit_gpu_bip352_key_erase.cpp`, `unified_audit_runner --section
memory_safety`, and `unified_audit_runner --section exploit_poc`.

### 2026-06-04 - P2-CT-001: bip32_derive_path erases intermediate child keys

`bip32_derive_path` (`src/cpu/src/bip32.cpp`) walks the path with `current = child`
each iteration but did not scrub the redundant `child` copy of the intermediate
private key / chain code, leaving secret material on the stack across iterations
(`derive_child` already scrubs its own internals — `child_scalar`, `parent_scalar`,
`I`, `IL`, `il_scalar` — but the path-walk loop did not). Now `detail::secure_erase`
the `child` key + chain_code after each assignment, and the working `current` on the
failure path. `current` is a LOCAL copy of the caller's master (never the caller's
object); the FINAL returned key is never scrubbed. No behavioral change — verified by
`src/cpu/tests/test_bip32.cpp` (derive_path == manual derive_child chain, deterministic).

**2026-06-04 follow-up:** the on-failure intermediate scrub was made *unconditional*
(a harmless erase of the public x-coordinate when the material is not secret) so the
on-failure path is reachable and is exercised by an xpub-hardened-derivation regression
test in `test_bip32.cpp` — restoring codecov patch coverage of the new lines to 100%.

### 2026-05-31 — MuSig2 signer-index check made mandatory (fail-closed); no new secret residue

`src/cpu/src/musig2.cpp` — `musig2_partial_sign` now treats the Rule-13 signer-index
cross-check as mandatory: when `individual_pubkeys` cannot validate `signer_index` (empty
or too short) it returns `Scalar::zero()` **before** loading any secret-derived intermediate
(`k`, adjusted `d`), so the new fail-close path introduces no additional secret residue and
performs no secret-dependent branch (it branches only on the public `signer_index` and the
container size). The signing-path erasure is unchanged: `k`, `d`, and `sec_nonce.k1/k2` are
still `secure_erase`d after a successful sign, and the degenerate `e == 0` path still erases
the secnonce. The empty-pubkeys rejection mirrors the pre-existing out-of-range `signer_index`
bounds-check, which likewise returns without consuming the caller's secnonce; the v2 ABI wrapper
(`ufsecp_musig2_partial_sign_v2`) additionally guards the secnonce buffer with `ScopeSecureErase`
on every exit path. Closes MED-3 — see `RESIDUAL_RISK_REGISTER.md` RR-010 and the
`AUDIT_CHANGELOG.md` 2026-05-31 entry.

### 2026-05-28 — defense-in-depth: erase nonce-derived r in ecdsa_sign_hedged

`src/cpu/src/ecdsa.cpp` — `ecdsa_sign_hedged` now erases `r` (the x-coordinate
of `kG` reduced mod n) via `secure_erase(&r, sizeof(r))` immediately after the
inner block that uses it. `r` is nonce-derived and while it is not directly a
secret scalar, defense-in-depth requires erasing all nonce-derived intermediates
to prevent residue accumulation on the stack across repeated calls.

### 2026-05-28 — batch parallel dispatch: private key read-only, output zeroed on error

`src/cpu/src/impl/ufsecp_ecdsa.cpp` — `ufsecp_ecdsa_sign_batch` and
`ufsecp_schnorr_sign_batch` now dispatch slots in parallel via `std::thread`.
Private key material (`privkeys32`) is accessed read-only by each thread
(non-overlapping slot indices, no aliasing). Output buffers (`sigs64_out`) are
fail-closed: on any slot error all output bytes are zeroed before the function
returns. Threads do not retain references to private key data beyond their
signing call; each slot routes through the existing CT signing primitive
(`ecdsa_sign_ct` / `schnorr_sign_ct`), so the per-slot secret lifecycle is
unchanged. `ctx->last_err` is `std::atomic<int>` — thread-safe.

### 2026-05-28 — SEC-002/003/004: FROST/MuSig2 degenerate input guards + nonce seed erasure

`src/cpu/src/frost.cpp` — `frost_sign_nonce_gen` now takes `nonce_seed` by value
and calls `secure_erase(nonce_seed.data(), nonce_seed.size())` before returning,
preventing the caller's seed copy from being the only place the secret resides
after the function returns. `frost_keygen_finalize` now rejects a group public key
that accumulates to the point at infinity (adversarial commitment cancellation).

`src/cpu/src/musig2.cpp` — `musig2_partial_sign` erases `k1` and `k2` (the two
nonce scalars from `sec_nonce`) before returning when `session.e == 0` (degenerate
challenge), ensuring no nonce residue remains on the stack in the error path.

### 2026-05-27 — SEC-006: FROST compute_challenge — e_hash and challenge_data erasure

`src/cpu/src/frost.cpp` — `compute_challenge()` now calls
`secure_erase(e_hash.data(), e_hash.size())` and
`secure_erase(challenge_data, sizeof(challenge_data))` before returning.
`e_hash` is the output of `cached_tagged_hash(g_challenge_midstate, ...)` —
a 32-byte digest seeded in part by the nonce commitment; `challenge_data`
(96 bytes) holds the concatenated nonce R + pubkey X + message, which
includes the nonce. Both are nonce-adjacent secrets and must be erased.
Previously neither was erased, leaving residue on the FROST signing stack.

### 2026-05-27 — SEC-005: ECDH off-curve pubkey rejection (invalid-curve attack)

`src/cpu/src/ecdh.cpp` — all three `ecdh_compute`, `ecdh_compute_xonly`, and
`ecdh_compute_raw` functions now validate the public key before
`ct::scalar_mul`:
1. `is_infinity()` check — rejects the point at infinity.
2. `y² = x³ + 7` field-element equality check — rejects points on twist curves
   (invalid-curve / small-subgroup attack).
If either check fails the function returns an empty array (no ECDH output).
The private key is never passed to `ct::scalar_mul` on an invalid input.

### 2026-05-26 — SHIM-NONCEGEN-001: musig2_nonce_gen nonce_extra parameter

### 2026-05-26 — SHIM-NONCEGEN-001: musig2_nonce_gen nonce_extra parameter

`src/cpu/src/musig2.cpp` — `musig2_nonce_gen` now accepts a `nonce_extra`
parameter (BIP-327 `extra_input32`). When non-NULL, the 32 bytes are appended
to the `nonce_input` buffer before the counter byte, expanding the hash input
from 129 → 161 bytes for both k1 and k2 derivations. Both code paths call
`secure_erase(nonce_input, sizeof(nonce_input))` unconditionally before return
(existing lifecycle; unchanged). The `nonce_extra` buffer is caller-owned
(no erase required on our side). Backward-compatible: NULL → 129-byte path,
identical k1/k2 as before.

### 2026-05-24 — v9 RT-006/-007/-014/-015 / TASK-022: stack-residue hardening bundle

Four small lifecycle fixes — all on the CPU signing surface — that close
remaining stack-residue gaps surfaced in the v9 adversarial review:

- **`src/cpu/src/schnorr.cpp:502,511`** — `schnorr_sign(const Scalar&)`
  and `schnorr_sign_verified(const Scalar&)` raw-key convenience
  overloads materialise a temporary `SchnorrKeypair` on the local stack
  (containing the negated signing scalar `kp.d`). The new lifecycle:
  call `schnorr_keypair_create`, run the sub-call, then
  `detail::secure_erase(&kp.d, sizeof(kp.d))` BEFORE returning. The
  returned `SchnorrSignature` is the public sig (no secret to erase).
- **`src/cpu/src/bip32.cpp:414`** — `derive_child` now uses
  `child_scalar.is_zero_ct()` (was `is_zero()`). `child_scalar` is the
  secret-derived child private key; using the CT predicate removes a
  data-dependent branch that would leak the (extremely unlikely,
  ~2^-256) wraparound case across many sessions.
- **`src/cpu/src/frost.cpp:60-106`** — `derive_scalar` and
  `derive_scalar_pair` now `secure_erase` the local `SHA256` object
  `h`, the per-tag `tag_hash`, and the finalized `hash` array before
  returning. All three incorporate the seed (secret material) into a
  stack-resident state. The output `Scalar` is moved to the return
  value via NRVO, so no secret residue remains in the caller frame.
- **`src/cpu/src/adaptor.cpp:311` (ecdsa_adaptor_sign)** — the
  degenerate-r early-return path now pre-erases `k`, `binding`, and
  `R_x_bytes` before returning the zero sentinel. The success-path
  erase block (later in the function) is unreachable from this branch.

Regression coverage: `audit/test_regression_secret_stack_residue_v9.cpp`
(wired in `unified_audit_runner` as `regression_secret_stack_residue_v9`,
advisory=false, `differential` section). 16/16 source-scan +
functional-roundtrip checks pass on the patched code.

### 2026-05-24 — v9 RT-002 / TASK-002: adaptor signing — DPA-blinded generator_mul

`src/cpu/src/adaptor.cpp` now routes every secret-scalar generator
multiplication through `ct::generator_mul_blinded(...)`. This applies
the same DPA-blinding discipline already active on ECDSA/Schnorr/MuSig2
sign paths to the adaptor variants:
- `schnorr_adaptor_sign`: long-term key `P = sk*G` (was unblinded).
- `ecdsa_adaptor_sign`: secret nonce `base_nonce = k*G` (was unblinded).
The `binding` scalar in `ecdsa_adaptor_sign` is derived from the PUBLIC
`adaptor_point` and stays on the unblinded primitive (correct + cheaper).

Regression coverage:
`audit/test_regression_adaptor_blinded_nonce.cpp` —
`test_adaptor_blinded_all_secret_sites_source_scan` asserts both blinded
calls are present AND no bare `ct::generator_mul(k)` or
`ct::generator_mul(private_key)` remains in `adaptor.cpp`.

### 2026-05-24 — v9 RT-001 / TASK-001: ufsecp_musig2_partial_sign (v1) DISABLED

`src/cpu/src/impl/ufsecp_musig2.cpp:203` — v1 ABI now hard-fails on
every call with `UFSECP_ERR_DEPRECATED_API`. Fail-closed lifecycle:
- Output buffer is `memset`-zeroed before any other work.
- `ScopeSecureErase` guards the caller-provided `secnonce`: the BIP-327
  consume-once invariant is preserved even on the reject path, so a
  caller that ignores the new error code cannot accidentally reuse the
  nonce in another call.
- NULL-arg validation still preempts (returns `UFSECP_ERR_NULL_ARG`).

Rationale: v1 cannot enforce the Rule-13 (privkey↔signer_index)
cross-check because the keyagg blob carries no per-signer pubkeys; v2
is the only supported entry. See `docs/SECURITY_CLAIMS.md` and
`docs/FFI_HOSTILE_CALLER.md` for full closure rationale.

### 2026-05-23 musig2.cpp + ufsecp_musig2.cpp — P1-SEC-01 / MED-3 partial mitigation (SUPERSEDED 2026-05-24)

- **`src/cpu/src/musig2.cpp` `musig2_partial_sign`**: Rule-13
  cross-validation block remains conditional on `individual_pubkeys`
  being non-empty AND `signer_index` in range. When both conditions
  hold (the standard C++ caller path via `musig2_key_agg`), the CT
  `ct::generator_mul(secret_key)` → `to_compressed()` comparison fires
  as defense-in-depth. When the field is empty (an ABI-deserialized
  caller that did not populate it), the check is silently skipped and
  the v1 ABI surface remains protected only by the caller getting the
  signer_index right.
  **Lifecycle**: the `secret_key` parameter (Scalar reference) is read
  for the CT byte comparison via `ct::generator_mul(secret_key)` →
  `to_compressed()`. No additional stack copies are introduced; the
  derived 33-byte compressed pubkey is a stack array that exits scope
  when the validation block returns. No new `secure_erase` required —
  the compressed pubkey is public-domain material.
- **`src/cpu/src/impl/ufsecp_musig2.cpp` v1 + v2 share a core helper**:
  Both ABI entries now use a new internal static helper
  `musig2_partial_sign_core` that handles secnonce parsing, session
  parsing, and the C++ signing call. v1 calls core directly (no pubkey
  cross-validation — deprecated path, known limitation). v2 populates
  `kagg_check.individual_pubkeys` from its caller-supplied pubkeys array
  before calling core, which lets the C++ Rule-13 check fire. Lifecycle
  of `sk`, `k1`, `k2`, and `MuSig2SecNonce sn` is unchanged: all four
  are ScopeSecureErase-guarded and zeroized on every exit path; the
  `secnonce` byte buffer is zeroized by an outer guard at the ABI entry.
- **No CT boundary change.** All comparisons and arithmetic on
  `secret_key` remain constant-time.

### 2026-05-22 address.cpp — silent-payments t_k path adopts ct::generator_mul (merged from main e4e17305)

- **`src/cpu/src/address.cpp` `silent_payment_create` / `silent_payment_scan` (P-SP-CT-001)**:
  The expected output point `P = B_spend + t_k · G` previously used
  `Point::generator().scalar_mul(t_k)` (variable-time GLV) on `t_k` derived from
  `tagged_hash("BIP0352/SharedSecret", S || ser32(k))` where `S = b_scan · A_sum`.
  Because `S` (and therefore `t_k`) depends on the scan private key, `t_k` is
  secret-adjacent and a variable-time generator multiply on it leaks timing
  information about the scan key. Both call sites now use
  `ct::generator_mul(t_k)` which (a) is constant-time and (b) uses the
  precomputed comb table (~33 µs vs ~826 µs cold FAST path).
  **Lifecycle**: `t_k` is a stack-local `Scalar` consumed by `ct::generator_mul`
  and `expected_x` computation; it is implicitly destroyed at end of loop scope.
  The existing `secure_erase(&a_sum, ...)`, `secure_erase(S_comp.data(), ...)`,
  `secure_erase(t_hash.data(), t_hash.size())` calls in `silent_payment_create`
  cover the upstream secret-bearing material; no new erase is required for
  `t_k` because no copy escapes the local stack frame.
  Originated in main commit `e4e17305` (perf+CT). No corresponding change
  needed in the scan loop's erase sequence.

### 2026-05-21 ecdsa.cpp / musig2.cpp / frost.cpp — P2-CT-001/002/003/007 nonce candidate scalar zeroization

- **`src/cpu/src/ecdsa.cpp` `rfc6979_nonce` (P2-CT-001)**: `cand1` and `cand2` are
  intermediate nonce candidate scalars derived from HMAC-SHA256 output keyed on the
  private key. After `ct::scalar_select(cand1, cand2, mask1)` selects the valid candidate
  into `result`, both candidates now receive `secure_erase(&cand1, sizeof(cand1))` and
  `secure_erase(&cand2, sizeof(cand2))` before the remaining erase block and `return result`.
  **Lifecycle**: created on stack, used once for CT select, erased immediately after.
  No other reference to cand1/cand2 exists after the erase.

- **`src/cpu/src/ecdsa.cpp` `rfc6979_nonce_hedged` (P2-CT-002)**: Identical fix applied
  to the hedged variant's candidate scalars. Same erase pattern, same lifecycle.

- **`src/cpu/src/musig2.cpp` `musig2_nonce_gen` k1 block (P2-CT-003)**: `cand1` and
  `cand2` are scoped to the k1 `{}` block. After `sec.k1 = ct::scalar_select(cand1, cand2, mask)`,
  both are now erased before `secure_erase(nonce_input, ...)`.

- **`src/cpu/src/musig2.cpp` `musig2_nonce_gen` k2 block (P2-CT-003)**: Same fix in the
  k2 scoped block after `sec.k2 = ct::scalar_select(cand1, cand2, mask)`.

- **`src/cpu/src/frost.cpp` `derive_scalar_from_hash` (P2-CT-007)**: `cand1` and `cand2`
  are derived from secret polynomial coefficient hashes. Both are now erased after
  `ct::scalar_select` and before `secure_erase(hash.data(), ...)`.

- **Impact**: Stack residue of unselected nonce candidates eliminated in all four
  locations. The selected candidate is returned via value copy (`Scalar const result`);
  cand1/cand2 themselves are zeroed and go out of scope.

### 2026-05-21 shim_*.cpp — SHIM-A01..A08 NULL-arg illegal_callback (no secret material)

- **`compat/libsecp256k1_shim/src/shim_ecdsa.cpp`** (`secp256k1_ecdsa_signature_normalize`):
  `sigin=NULL` now fires `secp256k1_shim_call_illegal_cb` before returning 0. No secret
  material is touched on the NULL path — the callback fires and the function exits before
  any signature bytes are read.
- **`compat/libsecp256k1_shim/src/shim_pubkey.cpp`** (`secp256k1_ec_pubkey_sort`,
  `secp256k1_ec_pubkey_negate`, `secp256k1_ec_pubkey_tweak_add`,
  `secp256k1_ec_pubkey_tweak_mul`, `secp256k1_ec_pubkey_combine`): NULL ctx or NULL
  pubkey/tweak/ins/out now fires the illegal callback. All affected functions operate on
  public key material only (no private key, nonce, or scalar secret is touched on these
  error paths).
- **`compat/libsecp256k1_shim/src/shim_tagged_hash.cpp`** (`secp256k1_tagged_sha256`):
  `msg=NULL` with `msglen=0` is now allowed (matches libsecp256k1 zero-length message
  semantics). `msg=NULL` with `msglen>0` fires the illegal callback. No secret material
  involved — this is a hashing utility, not a signing path.
- **Secret lifecycle impact**: None. All early-exit paths on these NULL-arg checks exit
  before any private key, nonce, or signing scalar is read. Erase requirement: N/A.

### 2026-05-21 musig2.cpp -- SEC-005 infinity nonce guard in musig2_start_sign_session

- **`src/cpu/src/musig2.cpp` (`musig2_start_sign_session`)**: Added `agg_nonce.R1.is_infinity() || agg_nonce.R2.is_infinity()` guard before computing the nonce-blinding factor. Returns a default-constructed `MuSig2Session{}` (all-zero scalars, infinity R) on failure.
  **Secret lifecycle**: No secret material is present in `agg_nonce` — R1 and R2 are public nonce points. The early return happens before any secrets are consumed. No erase requirement on the guard path.

### 2026-05-21 musig2.cpp -- SEC-009 empty-vector guard in musig2_nonce_agg

- **`src/cpu/src/musig2.cpp` (`musig2_nonce_agg`)**: Added early return for empty `pub_nonces` vector, returning an all-infinity `MuSig2AggNonce`. All inputs to `musig2_nonce_agg` are public nonces (broadcast to all participants). No secret material flows through this function. Erase requirement: N/A.

### 2026-05-12 ecdsa.cpp -- SEC-004 compute_three_block bounds guard (no secret material change)

- **`src/cpu/src/ecdsa.cpp` (`compute_three_block`)**: Added a `msg_len` bounds guard
  `if (msg_len < 128 || msg_len > 183) return;` at function entry.
  **No secret lifecycle change**: guard prevents `size_t` underflow on invalid input.
  Secrets written in normal path are still erased in all normal completion paths.

### 2026-05-12 frost.cpp -- SEC-010 threshold==0 guard (nonce erasure on early exit)

- **`src/cpu/src/frost.cpp` (`frost_sign`)**: Added explicit `key_pkg.threshold == 0` guard.
  **Secret material erased on early exit**: `nonce.hiding_nonce` and `nonce.binding_nonce`
  erased via `secure_erase` before returning a zero partial sig, matching the existing
  erase pattern for the quorum-failure path below.

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

<!-- 2026-05-21: ecdsa.cpp — ECDSASignature::from_compact deprecated (SEC-003); both overloads now carry [[deprecated]] attribute directing callers to parse_compact_strict. The array overload was inlined to avoid -Werror=deprecated-declarations from a deprecated function calling another deprecated function. No behavioral change; RFC 6979 HMAC state lifecycle unchanged. -->
<!-- 2026-05-24: bip39.cpp — extracted decode_bip39_words() static helper from bip39_validate / bip39_mnemonic_to_entropy (was a duplicated 32-line bit-reconstruction + SHA256 checksum block). The same `entropy[32]` stack buffer + `indices` vector are still zeroized via `detail::secure_erase` at every exit path on the caller side; the helper itself holds no persistent state and returns the entropy bytes via an out-pointer the caller already owns and erases. Pure code-shape refactor: BIP-39 entropy lifecycle unchanged. -->
<!-- 2026-05-24: sp_scanner.cpp + ltc/ltc_sp.cpp — extracted scan_batch pipeline into sp_scan_batch_impl.hpp (templated helper parameterised by SHA256 domain-separation tag midstate). Both BTC BIP-352 and LTC-SP scanners now call the shared impl. The scan path operates only on PUBLIC inputs (output pubkeys, A_sum, S_comp) plus the scan_privkey via batch GLV (timing-leaks-nothing argument unchanged: no oracle, batch input). spend_privkey is added to t_k only AFTER an x-only match against public output bytes. No secret-bearing state newly persisted by the shared helper. -->
<!-- 2026-05-24: types.hpp (include/secp256k1/types.hpp + src/cpu/include/secp256k1/types.hpp) — collapsed four inline {fe,sc}_to_data static_cast bodies into a single detail::to_data_cast<T>() template (one const + one non-const overload). The public wrappers retain their names, signatures, return types, and noexcept; -O3 inlines the wrapper to the original cast so the resulting object code is identical. types.hpp itself defines only struct layouts (FieldElementData, ScalarData, AffinePointData, JacobianPointData) and these zero-cost cast helpers — it touches no secret material directly. Pure code-shape refactor; no zeroization, lifetime, or CT semantics change. -->
<!-- 2026-05-25: ecdsa.cpp + recovery.cpp — SEC-CPU-004/005: r.is_zero() and s.is_zero() replaced with r.is_zero_ct() and s.is_zero_ct() in ecdsa_sign(), ecdsa_sign_hedged(), and ecdsa_sign_recoverable(). r and s are derived from the secret nonce k (r = kG.x mod n, s = k^{-1}*(z+r*d)); the VT branch on these scalars is a timing side-channel. CT variant eliminates the branch. Probability of triggering either guard: ~2^{-128}. Secret-erase coverage unchanged: k, k_inv, z, r_times_d, z_plus_rd, s are all zeroed on every exit path (early-return and normal return). -->
<!-- 2026-05-25: shim_ellswift.cpp — SHIM-006: general XDH path now erases x32_arr (ECDH shared secret x-coord, 32 bytes) after hashfp() via secure_erase. BIP-324 path and general path both now erase sk (parsed Scalar, sizeof(Scalar) bytes) and kb (raw byte copy, 32 bytes) before return. Previously these lingered on the stack after the function returned. Behavioral output (output[] written before erase) unchanged. -->
<!-- 2026-05-26: ecdsa.cpp — rfc6979_nonce, rfc6979_nonce_hedged, rfc6979_nonce_libsecp_compat: iteration-2 parse call changed from (void)Scalar::parse_bytes_strict_nonzero(t.data(), cand2) to bool const ok2 = Scalar::parse_bytes_strict_nonzero(t.data(), cand2); (void)ok2;. Pure code-quality fix (scanner satisfaction); CT select logic and secure_erase coverage unchanged. ok2 failure probability is ~2^{-128}; caller detects zero r/s. No new secret material introduced; existing zeroization of cand1, cand2, V, K, x_bytes, buf*, hmac on all exit paths is unaffected. -->
<!-- 2026-05-24: ellswift.cpp — three deduplications. (1) Hoisted XSWIFTEC_C1, XSWIFTEC_C2, FE_ZERO, FE_ONE, FE_THREE, FE_FOUR from per-function statics in xswiftec_frac / xswiftec_fwd / xswiftec_fwd_point to anonymous-namespace file scope (~75 lines of constant-declaration duplication removed). (2) Extracted the SHA256(tag||tag||sk||0×32||auxrnd||cnt) → u → ellswift_try_u retry loop into the static helper ellswift_create_retry_loop. Three callers (ellswift_create(privkey, auxrnd32), ellswift_create_fast(privkey), ellswift_create_fast(privkey, auxrnd32)) now share the loop body. The deterministic variant passes kEllswiftZero32 as auxrnd, preserving the original "two zero blocks" hash schedule. (3) Existing secure_erase of rand_bytes inside the loop is now done once in the shared helper (was previously duplicated three times) — same coverage, same exit paths. No change to CT routing of the pubkey derivation: each caller still chooses ct::generator_mul vs scalar_mul_generator as before; only the post-pubkey retry mechanics are shared. -->
<!-- 2026-05-28: shim_ecdsa.cpp + shim_recovery.cpp + shim_ellswift.cpp + bip32.cpp — secret-key stack-residue hardening (read-only-review CT-01/SHIM-01/02/CT-02). (CT-01) secp256k1_ecdsa_sign now secure_erase's the parsed private-key scalar `k` on every return path incl. parse-fail, plus the hedging `aux`; secp256k1_ecdsa_sign_recoverable does the same for `privkey_scalar` across all #ifdef branches (compat / hedged / plain). (SHIM-01) secp256k1_ellswift_create erases `sk`+`kb` on the success return AND the strict-parse early return (previously neither). (SHIM-02) secp256k1_ellswift_xdh general path now erases `sk`+`kb` on the parse-fail and all three error returns (sqrt-fail, two is_infinity) via an erase_secrets() scope helper — previously only the success path erased; this completes SHIM-006. (CT-02) ExtendedKey::derive_child now secure_erase's the secret-derived `il_scalar` (HMAC image of the parent private key on hardened paths: data = 0x00||privkey||index) on every one of its 7 return paths, and also erases I/IL on the two early returns (parse-fail, is_zero_ct) that previously skipped them. Output bytes are written before erase; no behavioral change — pure secret-lifecycle hardening. Regression guard: audit/test_regression_shim_seckey_erase.cpp (source-scan + functional round-trips). -->
