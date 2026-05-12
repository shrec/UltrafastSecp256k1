# Security Claims & API Contract

**UltrafastSecp256k1 v4.0.0** -- FAST / CT Dual-Layer Architecture (CPU + GPU)

### 2026-05-12 ecdsa.cpp -- SEC-004 compute_three_block bounds guard

- **`src/cpu/src/ecdsa.cpp` (`compute_three_block`)**: Bounds guard `msg_len < 128 || > 183`
  added. Prevents `size_t` underflow in `rem = msg_len - 128`. CT contract unchanged.
  No bypass via public API: `compute_three_block` is a `static` internal function.

### 2026-05-12 frost.cpp -- SEC-010 threshold==0 quorum bypass prevention

- **`src/cpu/src/frost.cpp` (`frost_sign`)**: `threshold == 0` now explicitly rejected.
  Prevents unsigned-comparison bypass of quorum enforcement. Nonces erased on new exit path.
  ABI layer was already guarded; this fix closes the C++ layer gap.

### 2026-05-11 SHIM-007 — musig2_nonce_agg_points performance overload

- **`src/cpu/src/musig2.cpp` (`musig2_nonce_agg_points`)**: Adds a fast overload for nonce
  aggregation that accepts pre-decompressed Point pairs instead of compressed byte arrays.
  **No change to security contract**: nonce aggregation operates exclusively on public
  broadcast nonces (R1_i, R2_i for each signer). No secret material is involved. CT invariant
  unaffected. Signing paths unchanged. The function eliminates redundant `decompress_point`
  (sqrt) calls that occurred when nonces had already been validated via `pubnonce_parse`.

### 2026-05-11 ct_sign.cpp — ecdsa_sign_hedged_recoverable + OpenCL low-S fix

- **`src/cpu/src/ct_sign.cpp`**: `ct::ecdsa_sign_hedged_recoverable()` added.
  Hedged RFC 6979 + aux_rand nonce with recid from K's y-parity during signing.
  Eliminates the 4× ecdsa_recover loop in `secp256k1_ecdsa_sign_recoverable` (ndata path).
  CT contract: recid derivation and low-S normalization are branchless (bitmask operations).
  All secret nonce material (`k`, `k_inv`, `z`, `s`, `pre_sig`) erased via `secure_erase`.
- **`src/opencl/kernels/secp256k1_extended.cl` (`ecdsa_sign_impl`)**: Low-S normalization
  replaced from `if (!scalar_is_low_s) scalar_negate(s)` to branchless bitmask conditional
  negate. The non-CT kernel is not a production path (ct_ecdsa_sign_impl is used instead)
  but the VT branch on the secret nonce-derived `s` was architecturally incorrect.
- **CT contract unchanged** for all production signing paths (ct_sign.cpp, ct_ecdsa_sign_impl).

### 2026-05-11 frost.cpp — threshold enforcement + derive_scalar strict parsing

- **`src/cpu/src/frost.cpp` (`frost_sign`)**: Threshold enforcement added at C++ layer.
  Direct C++ callers can no longer bypass the sub-quorum guard (was only at ABI layer).
  Returns zero partial sig on sub-quorum input; nonces erased before return.
- **`src/cpu/src/frost.cpp` (`derive_scalar`, `derive_scalar_pair`)**: `Scalar::from_bytes`
  (silent mod-n reduction) replaced with `parse_bytes_strict_nonzero` + counter retry via
  new helper `derive_scalar_from_hash`. Hash material is erased inside the helper.
  CT contract: nonce and polynomial coefficient scalars now guaranteed non-zero and in [1,n-1].



### 2026-05-11 scalar_mul_jac_fe52_z1 — HAMBURG=true, degenerate-case check removed

- **`ct_point.cpp`**: Enabled HAMBURG mode for `scalar_mul_jac_fe52_z1`.
  The function is exclusively called from `ecmult_const_xonly` (ellswift_xdh)
  where K_CONST encoding ensures `m_val ≠ 0` throughout, making the 18 ns
  degenerate-case guard unnecessary. CT security unchanged; this is a ~18 ns
  per-call optimization on the xdh path.

### 2026-05-11 ct_field::field_mul/field_sqr asm barriers — disabled under sanitizers

- **`ct_field.cpp`**: FE52 limb barriers in field_mul and field_sqr also disabled
  under TSan/ASan/MSan. Same root cause as field_add: TSan shadow-memory
  misinterpretation caused wrong field multiplication/squaring results under
  sanitizers. Manifested as `point_is_on_curve(kG) == false` for k=8..16.

### 2026-05-11 ct_field::field_add asm barrier — disabled under sanitizers

- **`ct_field.cpp`**: compiler barrier `asm volatile("" : "+r"(ptr) : : "memory")`
  in `field_add` now excluded under TSan/ASan/MSan. The barrier was confusing
  TSan's shadow-memory tracking, causing `ct::field_add` to appear different
  from `fast::operator+`. No CT security impact: sanitizer builds have LTO
  disabled by default, so the constant-propagation the barrier guards against
  cannot occur.

### 2026-05-10 ct_field::sub256 — ARM64/TSan borrow-chain correctness

- **`ct_field.cpp` (`sub256`)**: `__builtin_subcll` now restricted to
  `__x86_64__` and excluded under sanitizers. On ARM64 and under TSan,
  the borrow-flag chain was broken, causing `ct::field_sub`, `ct::field_add`
  (the p-conditional-subtraction path), and `ct::field_neg` to return wrong
  values — detected as 84 comprehensive-test failures under TSan and as
  "CT field_add == fast +" failures on macOS ARM64 CI runners.
- **CT contract unchanged.** Portable borrow chain is branchless and gives
  the same result as hardware SBB on all platforms.

### 2026-05-10 ct_field::add256 — ARM64 carry-chain correctness

- **`ct_field.cpp`**: `__builtin_addcll` carry-chain path restricted to
  `__x86_64__`. On Apple M-series ARM64 with Clang, the carry flag was not
  correctly propagated across chained `__builtin_addcll` calls, causing
  `ct::field_add` to produce results different from `fast::operator+` for
  certain inputs. The portable 64-bit carry fallback now applies to all
  non-x86-64 platforms.
- **CT contract unchanged.** Portable fallback is branchless on all platforms.
  The numerical result is identical to the portable chain on x86-64.

### 2026-05-10 ct_field::add256 — sanitizer/coverage build correctness

- **`ct_field.cpp`**: Disabled the `__builtin_addcll` (ADCX/ADOX) Clang
  fast-path under AddressSanitizer, ThreadSanitizer, MemorySanitizer, and
  coverage builds. Sanitizer/coverage instrumentation inserted between the
  paired addcll calls clobbers the x86 carry flag, which silently produces
  wrong addition results. The portable carry-chain fallback is now used in
  those configurations.
- **No security claim change.** The CT contract for `add256` is preserved:
  both the ADCX/ADOX path and the portable fallback are constant-time and
  free of secret-dependent branches. See `CT_VERIFICATION.md` 2026-05-10 for
  the CT-side analysis.
- **Why this matters for CI claims**: prior to this fix, sanitizer and
  coverage CI runs could surface false-positive arithmetic failures in
  field-element addition that masked or distorted real signal from those
  build configurations. The fix restores trustworthy ASan/TSan/MSan
  evidence.

### 2026-05-07 GPU batch signing CT audit — confirmation + annotation (SEC-004/SEC-005)

- **Metal batch signing (`secp256k1_kernels.metal`)**: Audit confirmed `ecdsa_sign_batch` and
  `schnorr_sign_batch` kernels already route through `ct_ecdsa_sign_metal()` and
  `ct_schnorr_sign_metal()` from `secp256k1_ct_sign.h`. Variable-time wrappers were replaced
  in a prior commit (a29196b). Added explicit `// GPU Guardrail 8: CT signing mandatory`
  annotations at both kernel entry points.
- **CUDA batch signing (`secp256k1.cu`)**: Same confirmation — `ecdsa_sign_batch_kernel` and
  `schnorr_sign_batch_kernel` both call `ct::ct_ecdsa_sign()` / `ct::ct_schnorr_sign()` from
  `include/ct/ct_sign.cuh`. Both kernels zero the output buffer before signing and check the CT
  return value, satisfying GPU Guardrails 8 and 9. Guardrail 8 annotations added.
- **GPU CT boundary rule**: All GPU batch signing kernels (Metal, CUDA, OpenCL) are audited to
  use CT signing primitives. Variable-time `ecdsa_sign()` / `schnorr_sign()` are banned from
  kernels that handle secret nonces or private keys.

### 2026-05-07 Multi-agent ultrareview — secret lifecycle zeroization fixes

- **`recovery.cpp` (`secp256k1::ecdsa_sign_recoverable`, public C++ API)**: Added
  `detail::secure_erase` for `k`, `k_inv`, `r_times_d`, `z_plus_rd`, and `s` before return
  on all paths (degenerate s==0 and normal path). The production ABI (`ct::ecdsa_sign_recoverable`)
  was already clean; this closes the gap for direct C++ API callers.
- **`bip32.cpp` (`ExtendedKey::derive_child`)**: Added `detail::secure_erase(&child_scalar, ...)`
  after `child.key = child_scalar.to_bytes()`, and on the `is_zero` early-return path which
  previously skipped erasure. Child private key no longer lingers on stack after serialization.
- **`adaptor.cpp` (`adaptor_nonce`)**: Restructured domain separation from tag-appended-last
  to BIP-340 tagged hash prefix pattern — `H = SHA256(SHA256(tag)||SHA256(tag)||data)`.
  Domain-first prevents cross-protocol nonce collision (tag appended last allows crafted
  prefix to collide with BIP-340 or RFC 6979 hash inputs).
- **`musig2.cpp` / `ufsecp_musig2.cpp` (`ufsecp_musig2_partial_sig_agg`)**: ABI wrapper now
  checks for all-zero aggregated signature (degenerate `sum(partial_sigs) = 0 mod n`) and
  returns `UFSECP_ERR_INTERNAL` instead of `UFSECP_OK`. Enforces Security Guardrail Rule 4.
- **`ecdsa.cpp` (`HMAC_Ctx::compute_short`)**: Added `assert(msg_len <= 55)` and runtime
  guard before `std::memset(block + msg_len + 1, 0, 55 - msg_len)`. Without the guard,
  `msg_len > 55` causes `size_t` unsigned wrap to ~0ULL — potential stack smash.

### 2026-05-07 libsecp256k1 shim — security hardening

- **`shim_batch_verify.cpp` (`secp256k1_ecdsa_verify_batch`)**: Added `y.square() == x*x*x + 7`
  curve membership check before `Point::from_affine` in both small-batch fallback path and
  large-batch Pippenger build loop. An off-curve pubkey propagated to Pippenger could produce
  false-positive batch verify results (soundness failure).
- **`shim_tagged_hash.cpp` (`secp256k1_tagged_sha256`)**: Fixed null-byte truncation for tags
  with embedded null bytes; replaced heap-allocating `std::string` with stack buffer.
- **`shim_schnorr.cpp` / `shim_ecdsa.cpp`**: Two-phase pubkey cache — first encounter writes
  fingerprint only (~8 bytes) to avoid 1.5 KB/1.4 KB slot writes for every unique pubkey
  (ConnectBlock pattern: ~19K unique pubkeys × 1.5 KB = 27 MB cache pollution per block).

### 2026-05-06 Build Fix — ecdsa.cpp always_inline (no security boundary change)

- **`ecdsa.cpp` (`ECDSASignature::is_low_s_ct`)**: Removed `__attribute__((always_inline))`
  that caused `-Werror=attributes` with GCC-13 (`always_inline function might not be inlinable`).
  No change to CT properties, secret paths, or API semantics. The function body, branching
  behaviour, and memory access patterns are unchanged.

### 2026-05-06 Security Fixes — ultrareview TASK-04/05 — is_zero_ct + adaptor const_cast UB

- **`recovery.cpp` / `ecdh.cpp` (TASK-04 / RT-05 / CT-12)**: `is_zero()` (variable-time,
  early-return on first non-zero limb) replaced with `is_zero_ct()` for all zero-checks on
  secret scalars (private key, ECDH key). CT claim for `ecdsa_sign_recoverable` and the three
  ECDH variants now holds at the public C++ layer.
- **`adaptor.cpp` (TASK-05 / RT-04)**: `const_cast<Scalar*>` on `const`-qualified `k` and
  `binding` removed by dropping the `const` qualifiers. C++ UB eliminated: `secure_erase`
  now operates on non-const lvalues and cannot be optimized away by LTO. Zeroization of
  secret nonce `k` after `ecdsa_adaptor_sign` is now guaranteed at the language level.

### 2026-05-06 Performance Fix — FROST O(n²) serialization (no security-boundary change)

- **`frost.cpp` (`compute_group_commitment_inline_binding`)**: Precompute all
  `to_compressed()` calls once before the binding-factor loop. Reduces field
  inversions from O(n²) to O(n). Nonce commitment points (hiding, binding) are
  public — no secret material involved. No CT or zeroization change.

### 2026-05-06 Security Fixes — ultrareview P1/P2 TASK-001..012 batch

- **`shim_schnorr.cpp` `sign_custom` (RT-001/CT-001)**: `s = k + e * kp.d` using
  `fast::operator*` on secret nonce and key replaced with `ct::scalar_add(k, ct::scalar_mul(e, kp.d))`.
  CT claim for `secp256k1_schnorrsig_sign_custom` (variable-length path) now holds.
- **`shim_extrakeys.cpp` `keypair_xonly_tweak_add` (CT-006)**: `new_sk = sk + t` using
  `fast::operator+` on secret key replaced with `ct::scalar_add(sk, t)`.
  CT claim for Taproot keypair tweak now holds (overflow branch no longer leaks).
- **`bip32.cpp` `derive_child` (RT-007)**: `Scalar::from_bytes(key)` replaced with
  `parse_bytes_strict_nonzero` — rejects parent keys ≥ n or == 0 with error return instead
  of silent mod-n reduction. `parent_scalar` erased after use on all paths.
- **`frost.cpp` `frost_lagrange_coefficient` (CT-002)**: `den.inverse()` (variable-time GCD)
  replaced with `ct::scalar_inverse(den)`. Lagrange computation now runs in constant time.
- **`shim_pubkey.cpp` `pubkey_combine` (SC-010)**: Per-element NULL check prevents null
  dereference (was UB); returns 0 on NULL element.

### 2026-05-06 Security Fixes — ultrareview P1/P2 batch

- **`musig2.cpp` nonce k2**: k2 domain separator aligned to k1 via `cached_tagged_hash(g_musig_nonce_midstate)`.
  Both nonces now derive under identical tag — eliminates cross-nonce tag divergence.
- **`musig2.cpp` `musig2_partial_sig_agg`**: Fail-closed for degenerate s=0 aggregate signature.
  Returns all-zero (failure indicator) rather than serializing an invalid s=0 Schnorr signature.
- **`frost.cpp` `frost_aggregate`**: Same fail-closed for s=0 aggregated FROST signature.
- **`bip324.cpp` private key parsing**: `Scalar::from_bytes` replaced with
  `parse_bytes_strict_nonzero` — rejects keys in [n, 2^256) and k=0 without silent reduction.
  CT claim: BIP-324 ephemeral key is secret; strict parsing ensures valid scalar throughout.

### 2026-05-05 Performance Audit — perf batch (no security-boundary change)

- **`ecdsa.cpp` (`ecdsa_sign_hedged`)**: `r = R.x_only_bytes()` → `Scalar::from_limbs(R.x().limbs())`.
  `r` is public nonce-point x-coordinate, not secret. No CT or zeroization impact.
- **`ecdsa.cpp` (`rfc6979_nonce_hedged`)**: Stack buffers `t` and `buf33` hoisted before retry loop.
  `secure_erase(t)` added to early-return path — zeroization parity with `rfc6979_nonce`. Secret
  nonce candidate material remains erased on all exit paths. No CT boundary change.
- **`schnorr.cpp`**: r-zero check replaced with 4-word OR — mechanical, no secret handling change.
- **`point.cpp`**: `alignas(64)` on hot table arrays; `kMaxZr` assertion; `SECP256K1_UNLIKELY`
  annotation removed from 50%-density branch. No secret-path changes.
- **`pippenger.cpp`**: Mixed-add optimization in aggregate phase for affine buckets. No secret paths.
- **OpenCL `secp256k1_field.cl`**: `field_sqr_n_impl` copy removed. No secret paths.
- **CUDA `schnorr.cuh`**: `memcpy` + branchless rx compare in `schnorr_verify`. No secret paths.

### 2026-05-05 Performance Audit — A-12/A-13/A-15 P3 minor fixes (no security-boundary change)

- **`scalar.cpp` A-12**: `k.bit(0) == 1` → `k.limbs_[0] & 1u` in `to_naf()`/`to_wnaf()`
  loops. Purely mechanical — no algorithm or security change.
- **`point.cpp` A-13**: Removed trailing-zero trim loops after `compute_wnaf_into()` calls.
  Those loops were no-ops (function already sets `out_len = last_set_bit + 1`).
- **`ecdsa.cpp` A-15**: `y.to_bytes()[31] & 1` → `y.limbs()[0] & 1u` in compressed pubkey
  parse. `y` is sqrt of a public x-coordinate — not secret. No CT/zeroization impact.

### 2026-05-05 Performance Audit — A-04/A-06 (no security-boundary change)

- **`ct_sign.cpp` / `ecdsa.cpp` A-06**: `r = R.x mod n` conversion path changed from
  `to_bytes()` + `from_bytes()` (8 byte-swaps) to `from_limbs()` (direct limb copy +
  `ge(ORDER)` check). `r` is the public x-coordinate of nonce point R = k·G. It is NOT
  a secret scalar. CT properties of `k` and the private key are unaffected. Claimed
  speedup: ecdsa_sign −11%, ct::ecdsa_sign −2% (measured, 11-pass IQR median).
- **`point.cpp` A-04**: `derive_phi52_table` now uses file-scope `kBeta52_pt` instead of
  a function-local `static const beta52`. Same value (GLV β constant), no algorithm change.

### 2026-05-05 Performance Audit — CT Refactoring (no security-boundary change)

- **`ct_point.cpp` `ct_glv_make_v` extraction**: The GLV half-scalar CT-negate/increment
  routine was duplicated in four scalar-mul functions. Extracted to a single
  `SECP256K1_INLINE static ct_glv_make_v` with identical logic. No CT semantics changed;
  the refactor reduces the risk of copy-paste divergence in a CT secret path.
- **`schnorr_verify(SchnorrXonlyPubkey)` overloads**: Marked `[[nodiscard]]` and `noexcept`.
  No algorithm change.

### 2026-05-05 Security Claim Updates

Following the full red-team / bug-bounty audit, 17 findings were fixed (4 Critical, 6 High, 7 Medium):

- **All GPU backends** (CUDA, OpenCL, Metal) now route all secret-bearing signing through
  CT paths. Variable-time `scalar_mul_generator_windowed` / `scalar_mul_generator_impl`
  on secret nonces is eliminated from all signing kernels.
- **Fault countermeasures** in OpenCL `ct_ecdsa_sign_verified_impl` and
  `ct_schnorr_sign_verified_impl` now actually call the verify step (previously stubs).
- **Metal `ecdsa_sign_recoverable_metal`** low-S normalization was using parity (`& 1`)
  instead of half-order comparison. Fixed — recid and low-S are now always correct.
- **Private key erasure** in CUDA/bench code: device buffers zeroed before free.
- **`secp256k1::ecdsa_sign`** (fast-path, not for production) marked `[[deprecated]]`
  to warn downstream C++ users.
- **`schnorr_sign_verified`** (CPU) now checks both `s==0` AND `r==all-zeros` (Rule 14).

## Operating Assumption

This document is written for owner-grade deployment standards, not just
outside evaluation.

The working question is not "will outsiders trust this?" The working question
is "if this engine had to satisfy the same standard we would demand for
protecting large assets, what claims would we require before accepting that
risk?"

That means:

1. Claims should be conservative.
2. Residual risks should be explicit.
3. Secret-bearing paths should be judged more harshly than public-data paths.
4. Experimental features should be treated as opt-in risk, not silently upgraded to production trust.
5. Secret-bearing code changes must be paired with updates to the matching evidence docs, enforced by `ci/check_secret_path_changes.py`.

## Fail-Closed Assurance Perimeter

The default `ci/audit_gate.py` path now treats the following as first-class
enforcement surfaces rather than optional tooling:

1. Failure-class matrix execution
2. ABI hostile-caller quartet coverage
3. Structured invalid-input grammar rejection
4. Stateful multi-call sequence integrity
5. Audit test-quality scanning

Two explicit non-claims remain important:

1. The library does **not** claim total hostile-caller quartet closure while the live ABI blocker set is non-zero.
2. Mutation kill-rate evidence is part of the assurance perimeter, but remains the heavier batch / owner-grade lane rather than the default per-commit runtime path.

Additional enforcement surfaces added in the 2026-04-07 CI hardening:

6. Crash-risk analysis: division-by-zero and other crash vectors in CT-sensitive paths
7. MemorySanitizer (MSan): detects use-of-uninitialized-memory, complements zeroization
8. Coverage upload failure gating: `fail_ci_if_error: true`
9. CT scalar_inverse(0) zero guard: both SafeGCD and Fermat fallback CT paths now return
   `Scalar::zero()` for zero input, matching the FAST path behavior (defense-in-depth)
10. Boundary sentinel test suite: 18 exploit-class checks covering inverse(0), empty batch
    verify, half-order low-S boundary, MuSig2 duplicate keys, aux_rand edge values,
    has_even_y(infinity), and CT inverse round-trips (`test_exploit_boundary_sentinels`)

---

## 1. Semantic Equivalence Contract

> **FAST and CT functions return identical results for all valid inputs.**

Both layers implement the same secp256k1 elliptic curve operations with the same
mathematical semantics. They differ **only** in execution profile:

| Property | FAST (`secp256k1::fast::`) | Public-namespace signing (`secp256k1::`) | CT (`secp256k1::ct::`) |
|----------|-----------------------------|------------------------------------------|------------------------|
| **Throughput** | Maximum | ~same as CT for signing | ~1.8-3.2x slower than fast:: |
| **Signing timing** | Data-dependent (variable-time) | **CT** — routes through `ct::generator_mul_blinded` + `ct::scalar_inverse` | Data-independent (constant-time) |
| **Branching** | May short-circuit on identity/zero | No secret-dependent branches in signing | Never branches on secret data |
| **Table Lookup** | Direct index | Blinded comb (same as ct::) for sign | Scans all entries via cmov |
| **Nonce Erasure** | Not erased | Stack nonces erased (k, k_inv, z) | Intermediate nonces erased (volatile fn-ptr) |
| **Side-Channel** | Not resistant | Resistant for signing paths | Resistant (CPU backend) |
| **Audit/ABI backing** | No | Use `ct::` or ABI for explicit audit trail | Canonical CT-validated path |

> **Note (2026-05-01):** `secp256k1::ecdsa_sign`, `secp256k1::schnorr_sign`, and
> `secp256k1::ecdsa_sign_recoverable` were found to internally call
> `ct::generator_mul_blinded` and `ct::scalar_inverse`, making them CT-equivalent
> in practice. The previous documentation describing them as "variable-time" was
> incorrect. See `src/cpu/src/ecdsa.cpp:signing_generator_mul` and
> `src/cpu/src/recovery.cpp:signing_generator_mul` (both alias `ct::generator_mul_blinded`),
> and `src/cpu/src/schnorr.cpp` (calls `ct::generator_mul_blinded` directly).
> The canonical explicit CT path (`secp256k1::ct::ecdsa_sign`) remains preferred
> for audits and new code because it carries explicit CT intent.

### CT Overhead by Platform (v3.4.0)

Measured with `bench_unified` / `gpu_bench_unified` (signing operations; verify uses public inputs -- CT not needed):

| Platform | ECDSA Sign CT/FAST | Schnorr Sign CT/FAST |
|---|---|---|
| x86-64 (i5-14400F, GCC 14.2) | **1.93x** | **2.13x** |
| ARM64 Cortex-A55 (Clang 18) | 2.57x | 3.18x |
| RISC-V U74 @ 1.5 GHz (GCC 13) | 1.96x | 2.37x |
| ESP32-S3 Xtensa LX7 @ 240 MHz | 1.05x | 1.06x |
| **GPU RTX 5060 Ti (CUDA 12.0)** | **2.06x** | **2.51x** |

ESP32 has near-zero CT overhead: in-order core, no speculative execution. x86 overhead
improved in v3.16.0 (was 1.94x ECDSA) following the GLV decomposition correctness fix.

**2026-05-04 (add_affine_fast_ct magnitude fix):** Corrected FE52 negate()
magnitude parameters in the incomplete mixed-add used by `ct::generator_mul`
and `ct::scalar_mul_prebuilt_fast`. The 5x52 lazy-reduction scheme requires
`negate(m)` to be called with m ≥ actual element magnitude to avoid uint64
underflow. Affected parameters: negate(8)→negate(18) for X1 (mag≤18 after
point_dbl_n_core), negate(4)→negate(10) for Y1 (mag≤10), negate(1)→negate(7)
for X3 (mag=7 after two-add accumulation). This was the root cause of all
signing correctness failures introduced with the incomplete-add optimization.
CT timing invariant unaffected.

**2026-05-04 (AVX2 comb_lookup):** `ct::generator_mul` reduced from 13951 ns to ~8700 ns
(-37.7%) on x86-64-v3 (AVX2) via vectorized 32-entry comb table scan in `comb_lookup`.
The AVX2 path uses the same XOR/AND CT blend idiom as `table_lookup_core` (already used
by `ct::scalar_mul`), processing each 80-byte CTAffinePoint as 2.5 ymm registers.
CT invariant preserved: no data-dependent branches; all 32 entries touched per lookup.
Post-AVX2 CT-vs-libsecp comparison (x86-64-v3, bench_unified):
  ct::generator_mul:  8700 ns  (was 13951 ns)
  CT Schnorr Sign:   10864 ns  vs libsecp 13635 ns  → 1.26x  (was 0.87x)
  CT ECDSA Sign:     12897 ns  vs libsecp 17809 ns  → 1.38x  (was 0.91x)

**2026-05-04 (new CT primitive: `ecmult_const_xonly` for BIP-324 ECDH):**
Added `ct::ecmult_const_xonly(xn, xd, q)` to `ct_point.cpp`/`ct/point.hpp`.
Computes x-coordinate of `q * P` (BIP-324 X-only ECDH) without sqrt. Secret
input is `q`; `xn`/`xd` are public. Internally delegates to `scalar_mul_jac`
(same GLV + signed-digit CT path as `ct::scalar_mul`) and recovers x via one
combined field inversion. CT invariant: fixed iteration count, no branches on `q`.
A 4x64 fallback using sqrt is provided for non-x86 platforms (slower but
functionally equivalent). Also refactored `scalar_mul` to extract `scalar_mul_jac`
as a static Jacobian-output helper; `scalar_mul` behaviour and CT properties
are unchanged.

**2026-05-04 (CT primitive internal optimizations — perf review B-1, B-8, B-10):**
Three internal changes to `ct_scalar.cpp` and `ct_field.cpp` that improve performance
without altering CT invariants. No new security claims; existing claims unaffected.

1. `divsteps_59`: `volatile uint64_t c1/c2` → plain `uint64_t`. The CT property is
   algorithmic (fixed 59 iterations, branchless bitmasks), not from `volatile`.
   Saves ~118 memory round-trips per `scalar_inverse` call (~100–200 ns).

2. `scalar_cswap`: Full-Scalar temporaries → XOR-swap with mask (identical to
   `field_cswap`). Same CT semantics; eliminates 64-byte copy overhead.

3. `add256`/`sub256`: `__builtin_addcll`/`__builtin_subcll` guarded Clang-only
   (`#if defined(__clang__)`). Original guard `defined(__GNUC__) || defined(__clang__)`
   was incorrect: GCC-13 defines `__GNUC__` but lacks these Clang-extension builtins,
   causing a build failure. GCC now falls through to the portable carry loop.
   CT invariant unaffected — both paths produce identical results and have
   data-independent latency on all x86-64 targets (2026-05-04 compiler compat fix).

Existing `audit_ct`, `ct_sidechannel`, and `test_ct_equivalence` CAAS modules
continue to cover these primitives. Regression: `test_regression_perf_review_sec_2026_05_04`.

### Where Results May Differ

Both layers are tested for bit-exact equivalence. Possible divergences:

- **Error handling**: Both return zero/infinity for invalid inputs, but CT may
  take longer to return on error (it completes the full execution trace).
- **Timing**: By design — FAST is faster, CT is constant-time.
- **Input validation**: Identical. Both reject zero scalars, out-of-range values.

### Verified by CI

FAST == CT equivalence is verified in every CI run:
- `test_ct` — arithmetic, scalar mul, generator mul, ECDSA sign, Schnorr sign
- `test_ct_equivalence` — property-based (random + edge vectors)

Preflight coverage reporting is graph-driven. The project graph builder records
coverage from direct CTest targets and from selected unified-audit modules for
public-data code paths such as batch verification, recovery, taproot, address,
and multiscalar logic.

---

## 2. Developer Guidance: When to Use FAST vs CT

### CT Is REQUIRED For:

| Operation | Why | Function |
|-----------|-----|----------|
| **ECDSA signing** | Private key enters scalar multiplication | `ct::ecdsa_sign()` (canonical); `secp256k1::ecdsa_sign()` also CT as of 2026-05-01 |
| **Schnorr signing** | Private key + nonce in scalar mul | `ct::schnorr_sign()` (canonical); `secp256k1::schnorr_sign()` also CT as of 2026-05-01 |
| **Key generation / derivation** | Secret scalar x G | `ct::generator_mul()` |
| **Keypair creation** | Private key enters point mul | `ct::schnorr_keypair_create()` |
| **X-only pubkey from privkey** | Secret scalar x G | `ct::schnorr_pubkey()` |
| **Any scalar mul with secret scalar** | Timing leaks scalar bits | `ct::scalar_mul()` |
| **Nonce generation** | k must remain secret | RFC 6979 (used internally) |
| **Secret-dependent selection** | Branch on secret data | `ct::scalar_cmov/cswap/select` |

### FAST Is OK For:

| Operation | Why | Function |
|-----------|-----|----------|
| **ECDSA verification** | All inputs are public | `ecdsa_verify()` |
| **Schnorr verification** | All inputs are public | `schnorr_verify()` |
| **Batch verification** | Public signatures + public keys | `schnorr_verify()` in loop |
| **Public key arithmetic** | No secret data involved | `Point::scalar_mul()` on public key |
| **Parsing / serialization** | No secret data | `from_bytes()`, `to_bytes()` |
| **Hash operations** | BIP-340 tagged hash on public data | `tagged_hash()` |
| **Address generation from public key** | No secret data | All coin-dispatch functions |

### If You Are Unsure: Use CT

When in doubt about whether an input is secret, **always use the CT variant**.
The performance cost is bounded (1.8-3.2x depending on platform) and eliminates
timing side-channel risk.

```cpp
// [OK] CORRECT: explicit CT namespace for signing — canonical, auditable path
#include <secp256k1/ct/sign.hpp>
auto sig = secp256k1::ct::ecdsa_sign(msg_hash, private_key);

// [OK] ALSO CORRECT: secp256k1::ecdsa_sign routes through ct::generator_mul_blinded
// and ct::scalar_inverse (see src/cpu/src/ecdsa.cpp:signing_generator_mul).
// Use this only when you have confirmed CT coverage via the source graph.
// For new code, prefer the explicit ct:: namespace for audit clarity.
#include <secp256k1/ecdsa.hpp>
auto sig2 = secp256k1::ecdsa_sign(msg_hash, private_key);

// [OK] CORRECT: FAST for verification (all inputs public)
#include <secp256k1/ecdsa.hpp>
bool ok = secp256k1::ecdsa_verify(msg_hash, pubkey, sig);

// [FAIL] WRONG: direct fast:: scalar_mul on generator with secret scalar —
// variable-time, leaks private key timing.
// Point::generator().scalar_mul(private_key)  <-- never do this for secrets
```

### Compile-Time Guardrail

Define `SECP256K1_REQUIRE_CT=1` to get deprecation warnings on non-CT sign
functions. This helps catch accidental use of the FAST path for secret operations:

```bash
cmake -DCMAKE_CXX_FLAGS="-DSECP256K1_REQUIRE_CT=1" ...
```

---

## 3. BIP-340 Strict Parsing (v3.16.0)

> **All cryptographic parsing now enforces strict encoding by default.**

v3.16.0 adds strict parsing APIs that reject all malformed inputs at parse time,
preventing degenerate or out-of-range values from entering the cryptographic pipeline.

### Strict APIs

| API | Rejects |
|-----|---------|
| `Scalar::parse_bytes_strict(bytes)` | zero scalar, value >= group order n |
| `FieldElement::parse_bytes_strict(bytes)` | zero element, value >= field prime p |
| `SchnorrSignature::parse_strict(bytes)` | r >= p, s >= n |

### C ABI Strict Enforcement

The following C ABI functions use strict parsing internally (v3.16.0):
- `ufsecp_schnorr_verify` — rejects malformed signatures before any computation
- `ufsecp_schnorr_sign` — validates keypair before signing
- `ufsecp_pubkey_xonly` — rejects x-coordinate >= p when lifting to curve point

### CMake Option

```cmake
# Enforce strict parsing library-wide (replaces all lenient parse_bytes calls)
-DUFSECP_BITCOIN_STRICT=ON
```

### Test Coverage

31-test BIP-340 strict suite (`test_bip340_strict_parsing`):
- reject-zero scalar, reject-zero field element
- reject overflow (r == n, s == p, r == p+1)
- accept all valid boundary values (r == 1, r == n-1)

---

## 4. CT Nonce Erasure (v3.16.0)

> **Intermediate nonces are erased from the stack after signing.**

`ct::schnorr_sign` and `ct::ecdsa_sign` now erase intermediate RFC 6979 nonces
immediately after use via the **volatile function-pointer trick**, matching the
approach used in bitcoin-core/libsecp256k1:

```cpp
// Pattern used internally in ct::ecdsa_sign and ct::schnorr_sign:
static void (*volatile wipe_fn)(void*, size_t) = memset;
wipe_fn(&nonce_k, 0, sizeof(nonce_k));
```

This is a best-effort mitigation. Complete nonce erasure cannot be guaranteed
due to compiler stack reuse and register allocation — this is true for all
cryptographic implementations, including libsecp256k1.

---

## 5. FROST / MuSig2 Protocol CT Status (v3.16.0)

### MuSig2 (BIP-327)

- **Scalar multiplications in signing**: use `ct::` namespace — CT-protected
- **Nonce generation**: RFC 6979-based — CT-protected
- **Protocol-level timing**: added to dudect in v3.16.0
- **Status**: Early implementation. API may change while CAAS protocol evidence matures.

### FROST (RFC 9591)

- **DKG scalar operations**: use `ct::` namespace
- **Signing round scalar mul**: CT-protected
- **Protocol-level timing**: added to dudect in v3.16.0 (sample counts lower)
- **Secret scratch reduction**: 2026-04-14 refactor removed temporary signer-ID and
  binding-factor heap vectors from signing/verification/aggregation paths; this lowers
  transient secret-adjacent allocation pressure without changing the public API.
- **Status**: Early implementation. secp256k1 ciphersuite not in RFC 9591.

> **Explicit claim**: Neither MuSig2 nor FROST have been subjected to a
> protocol-level side-channel analysis by a third party. Use in production
> at your own risk.

---

## 6. API Mapping: FAST <-> CT

### CPU API

| Operation | FAST (public data) | CT (secret data) |
|-----------|--------------------|-------------------|
| Scalar x G | `Point::generator().scalar_mul(k)` | `ct::generator_mul(k)` |
| Scalar x P | `P.scalar_mul(k)` | `ct::scalar_mul(P, k)` |
| Point add | `Point::add(P, Q)` | `ct::point_add_complete(P, Q)` |
| Point double | `Point::double_point(P)` | `ct::point_dbl(P)` |
| ECDSA sign | `secp256k1::ecdsa_sign(...)` | `ct::ecdsa_sign(...)` |
| Schnorr sign | `secp256k1::schnorr_sign(...)` | `ct::schnorr_sign(...)` |
| Schnorr pubkey | `secp256k1::schnorr_pubkey(k)` | `ct::schnorr_pubkey(k)` |
| Keypair create | `schnorr_keypair_create(k)` | `ct::schnorr_keypair_create(k)` |
| Knowledge prove | N/A | `zk::knowledge_prove()` (uses CT internally) |
| DLEQ prove | N/A | `zk::dleq_prove()` (uses CT internally) |
| Range prove | N/A | `zk::range_prove()` (uses CT internally) |
| Knowledge verify | `zk::knowledge_verify()` | N/A (public data) |
| DLEQ verify | `zk::dleq_verify()` | N/A (public data) |
| Range verify | `zk::range_verify()` | N/A (public data) |
| Scalar cond. move | N/A (use if/else) | `ct::scalar_cmov(r, a, mask)` |
| Scalar cond. swap | N/A (use std::swap) | `ct::scalar_cswap(a, b, mask)` |
| Scalar cond. negate | `s.negate()` with if | `ct::scalar_cneg(a, mask)` |

### GPU (CUDA/OpenCL/Metal) API

All GPU CT functions are in the `secp256k1::cuda::ct::` namespace (CUDA),
with equivalent kernels in OpenCL (`secp256k1_ct_sign.cl`, `secp256k1_ct_zk.cl`)
and Metal (`secp256k1_ct_sign.metal`, `secp256k1_ct_zk.metal`).
All three backends implement identical CT algorithms.

| Operation | FAST (`secp256k1::cuda::`) | CT (`secp256k1::cuda::ct::`) |
|-----------|---------------------------|------------------------------|
| Scalar x G | `scalar_mul_generator_const(k, &r)` | `ct_generator_mul(k, &r)` |
| Scalar x P | `scalar_mul(&P, k, &r)` | `ct_scalar_mul(&P, k, &r)` |
| Point add | `jacobian_add(&P, &Q, &r)` | `ct_point_add(&P, &Q, &r)` |
| Point double | `jacobian_double(&P, &r)` | `ct_point_dbl(&P, &r)` |
| Mixed add | N/A | `ct_point_add_mixed(&P, &Q, &r)` |
| ECDSA sign | `ecdsa_sign(msg, key, &sig)` | `ct_ecdsa_sign(msg, key, &sig)` |
| Schnorr sign | `schnorr_sign(key, msg, aux, &sig)` | `ct_schnorr_sign(key, msg, aux, &sig)` |
| Keypair create | N/A | `ct_schnorr_keypair_create(key, &kp)` |
| Knowledge prove | N/A | `ct_knowledge_prove_device(sec, pk, base, msg, aux, &pf)` |
| DLEQ prove | N/A | `ct_dleq_prove_device(sec, G, H, P, Q, aux, &pf)` |
| Knowledge verify | `knowledge_verify_device(...)` | N/A (public data) |
| DLEQ verify | `dleq_verify_device(...)` | N/A (public data) |
| Field cmov | N/A | `field_cmov(&r, &a, mask)` |
| Scalar cmov | N/A | `scalar_cmov(&r, &a, mask)` |
| Scalar inverse | `scalar_inverse(a, &r)` | `scalar_inverse(a, &r)` (CT Fermat) |

#### GPU CT Throughput (RTX 5060 Ti)

| Operation | ns/op | Throughput | CT/FAST |
|-----------|-------|------------|---------|
| ct::k*G | 341.9 | 2.92 M/s | 2.65x |
| ct::k*P | 347.2 | 2.88 M/s | -- |
| ct::ecdsa_sign | 433.9 | **2.30 M/s** | 2.06x |
| ct::schnorr_sign | 715.8 | **1.40 M/s** | 2.51x |

---

## 7. CT Timing Verification

CT claims are verified empirically using the **dudect** methodology
(Reparaz, Balasch, Verbauwhede, 2017):

- **Per-PR**: dudect smoke test (`|t| < 25.0`, ~60s) in the Extended Tests workflow
- **Per-push** (dev/main): full statistical analysis (`|t| < 4.5`, ~30 min) in the Extended Tests workflow
- **Native ARM64**: Apple Silicon M1 (macos-14): smoke per-PR + full per-push in `ct-arm64.yml`
- **Valgrind taint**: `MAKE_MEM_UNDEFINED` on all secret inputs, every CI run
- **ct-verif LLVM pass**: compile-time CT verification (no secret-dependent branches at IR level)
- **MuSig2/FROST**: protocol-level timing tests added in v3.16.0

### Functions Under dudect Coverage

`ct::field_mul`, `ct::field_inv`, `ct::field_square`, `ct::scalar_mul`,
`ct::generator_mul`, `ct::point_add_complete`, `field_select`, ECDSA sign,
Schnorr sign, MuSig2 sign (protocol-level), FROST sign (protocol-level).

See [docs/CT_EMPIRICAL_REPORT.md](CT_EMPIRICAL_REPORT.md) for full methodology.

### CT Claim Scope

> The CT guarantee applies to:
> - **CPU**: `secp256k1::ct::` under `g++-13` / `clang-17+` at `-O2`, on **x86-64** and **ARM64**
> - **CUDA GPU**: `secp256k1::cuda::ct::` under CUDA 12.0+ / nvcc, on **SM 7.5+** (Turing through Blackwell)
> - **OpenCL GPU**: CT kernels in `secp256k1_ct_sign.cl` / `secp256k1_ct_zk.cl`
> - **Metal GPU**: CT shaders in `secp256k1_ct_sign.metal` / `secp256k1_ct_zk.metal`

All GPU CT layers provide **algorithmic** constant-time guarantees (no secret-dependent
branches or memory access patterns). Hardware-level side-channel resistance on GPUs
is limited by the SIMT/SIMD execution model.

**Explicitly NOT covered:**
- Protocol internals of FROST and MuSig2 -- partial coverage only
- Compilers or optimization levels not tested in CI
- Microarchitectures not in the CI matrix
- Hardware-level electromagnetic/power analysis on any platform

---

## 8. ZK Proof Security Properties

### Schnorr Knowledge Proof

- **Soundness**: Prover cannot forge proof without knowing discrete log (Fiat-Shamir in ROM)
- **Zero-Knowledge**: Proof reveals no information about secret beyond the public key
- **Binding**: Challenge derived via tagged SHA-256 ("ZK/knowledge"), bound to R, P, and msg
- **CT**: Proving uses `ct::generator_mul` for nonce commitment; nonce erased after use

### DLEQ Proof (Discrete Log Equality)

- **Soundness**: Both discrete logs must be identical or attack succeeds with negligible probability
- **Binding**: Challenge bound to full tuple (G, H, P, Q, R1, R2) via tagged SHA-256
- **Zero-Knowledge**: Proof reveals no information about the shared secret
- **CT**: Proving uses CT scalar multiplications for both bases

### Bulletproof Range Proof

- **Completeness**: Valid proofs always verify
- **Soundness**: Prover cannot create proof for value outside [0, 2^64) except with negligible probability
- **Zero-Knowledge**: Proof leaks no information about value or blinding factor
- **Logarithmic Size**: O(log n) group elements for n-bit range (12 group elements for 64-bit)
- **No Trusted Setup**: Nothing-up-my-sleeve generators derived from tagged hashes
- **CT**: Prover uses CT layer for all secret-dependent operations (blinding, nonce generation)
- **Verification**: Uses FAST layer with MSM optimization (public data only)

---

## 9. Release CT Scope Tracking

Every release must answer: **"Did the CT scope change?"**

| Release | CT Scope Changed? | Details |
|---------|-------------------|---------|
| dev (2026-05-04) | **Perf, no scope change** | `ct::generator_mul` comb inner loop: complete unified add (12M+2S) → incomplete mixed Jacobian+affine add (7M+3S, `add_affine_fast_ct`). CT invariants (fixed iteration count, branchless cmov lookup) unchanged. Safety: all table entries are fixed G multiples; degenerate probability ~2^-128. Both 52-bit and 4x64 paths updated. |
| v3.22.0 | **Yes** | OpenCL CT layer (secp256k1_ct_sign.cl, secp256k1_ct_zk.cl); Metal CT layer (secp256k1_ct_sign.metal, secp256k1_ct_zk.metal); full C ABI with 80+ functions; BIP-39, Ethereum, Pedersen, ZK, Adaptor, MuSig2, FROST |
| v3.21.0 | **Yes** | GPU CT layer (5 headers); GPU CT audit modules in gpu_audit_runner; GPU CT benchmarks in gpu_bench_unified |
| v3.16.0 | **Yes** | CT nonce erasure (volatile fn-ptr trick); MuSig2/FROST dudect added; ct-arm64 ARM64 native CI |
| v3.15.0 | **Yes** | Branchless `scalar_window` on RISC-V; `value_barrier` after mask; RISC-V `is_zero_mask` asm |
| v3.13.1 | **Yes (fix)** | GLV decomposition correctness fix; CT scalar_mul overhead reduced to 1.05x |
| v3.13.0 | **Yes** | Added `ct::ecdsa_sign`, `ct::schnorr_sign`, `ct::schnorr_pubkey`, `ct::schnorr_keypair_create` |
| v3.12.x | No | CT layer existed (scalar/field/point), no high-level sign API |

---

## 9. Equivalence Test Coverage

### Automated in CI (`test_ct` + `test_ct_equivalence`)

| Category | Tests | Edge Vectors |
|----------|-------|--------------|
| Field arithmetic | add, sub, mul, sqr, neg, inv, normalize | 0, 1, p-1 |
| Scalar arithmetic | add, sub, neg, half | 0, 1, n-1 |
| Conditional ops | cmov, cswap, select, cneg, is_zero, eq | all-zero, all-ones |
| Point addition | general, doubling, identity, inverse | O+O, P+O, O+P, P+(-P) |
| Scalar mul | k=0,1,2, known vectors, large k, random | 0, 1, 2, n-1, n-2, random |
| Generator mul | fast vs CT equivalence | 1, 2, random 256-bit |
| ECDSA sign | CT vs FAST identical output | Key=1, key=3, random keys |
| Schnorr sign | CT vs FAST identical output | Key=1, key=3, random keys |
| Schnorr pubkey | CT vs FAST identical output | Key=1, random keys |

### Property-Based (`test_ct_equivalence`)

- 64 random 256-bit scalars → `ct::generator_mul(k) == fast::scalar_mul(G, k)`
- 64 random scalars → `ct::scalar_mul(P, k) == fast::scalar_mul(P, k)`
- 32 random key+msg pairs → `ct::ecdsa_sign == fast::ecdsa_sign` + verify
- 32 random key+msg pairs → `ct::schnorr_sign == fast::schnorr_sign` + verify
- Boundary scalars: 0, 1, 2, n-1, n-2, (n+1)/2

---

## References

- [SECURITY.md](../SECURITY.md) — Vulnerability reporting
- [THREAT_MODEL.md](THREAT_MODEL.md) — Attack surface analysis
- [docs/CT_VERIFICATION.md](CT_VERIFICATION.md) — Technical CT methodology, dudect details
- [docs/CT_EMPIRICAL_REPORT.md](CT_EMPIRICAL_REPORT.md) — Full empirical proof report
- [AUDIT_GUIDE.md](AUDIT_GUIDE.md) — Auditor navigation
- [dudect paper](https://eprint.iacr.org/2016/1123) — Reparaz et al., 2017

---

<!-- 2026-04-28: ufsecp_gpu.h docstring corrected — ufsecp_gpu_context_create → ufsecp_gpu_ctx_create (phantom export removal, misuse_resistance gate fix). No behavioral change; GPU ABI secret-bearing claims unchanged. -->
<!-- 2026-05-04: ct_point.cpp — generator_mul comb inner loop switched to incomplete mixed add (add_affine_fast_ct). CT scope unchanged; fixed iteration count and branchless cmov table lookup preserved. Performance improvement only. -->

*UltrafastSecp256k1 v4.0.0 -- Security Claims*
