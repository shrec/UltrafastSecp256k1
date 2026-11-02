# Constant-Time Verification

> **Freshness-gated (Bastion B14).** The CT claims in this document are bound to
> machine-readable evidence in [`docs/CT_EVIDENCE_STATUS.json`](CT_EVIDENCE_STATUS.json)
> and gated by [`ci/check_ct_evidence_status.py`](../ci/check_ct_evidence_status.py)
> (also `audit_gate.py --ct-evidence-status`, principle **G-14**). Each CT surface
> (ECDSA / Schnorr / recoverable signing, keypair/secret-key, scalar inverse,
> RFC 6979 nonce, GPU public-data boundary) binds to committed evidence (CT
> primitive headers + audit regression tests) plus the verdict tools that
> establish the claim. On every push the committed-evidence + freshness dimension
> is checked (cheap); the heavy tool verdicts (ct-verif / valgrind-ct / dudect)
> are evaluated when supplied via `--verdict-dir` (CI workflows), where a
> required-tool FAIL or a single PASS + SKIP is **inconclusive, never a pass**.
> Run: `python3 ci/check_ct_evidence_status.py --json`.

### 2026-09-24 — ECDH point-IO and public x-only validation boundaries

The three C++ ECDH variants and the libsecp256k1 ECDH shim now consume
`ct::detail::scalar_mul_jacobian` directly. The secret-derived shared point
stays in `CTJacobianPoint`; `point_to_x32`, `point_to_compressed33`, and
`point_to_xy64` serialize it without the public `Point` conversion and its
declassification. The fixed-size serializers compute and mask their output
before the caller declassifies the result-valid mask and branches on it.
Public peer validation may vary with the public point; the private scalar
remains on the existing CT multiplication path. Custom shim hash callbacks
consume secret bytes and are outside the library's CT claim.

Public `ct::ecmult_const_xonly` validates public `xn`/`xd`, including zero
denominator and invalid-lift rejection. Its native Jacobi check is explicitly
variable-time on those public coordinates, never on the secret scalar. The
private `ct::detail::ecmult_const_xonly_trusted` route is used only by BIP-324
and ElligatorSwift decoders that establish a valid lift and nonzero denominator;
it adds no public API or permission to skip validation for arbitrary callers.

Focused byte/rejection coverage is in `ct_point_io`, `selftest`
(`tests/test_ecdh_recovery_taproot.cpp`), `secp256k1_shim_test`,
`regression_ecdh_xy64_erase`, and `xonly_invalid_contract`. It includes
non-affine peers, fixed reference bytes, invalid inputs, and custom callback
bytes/return values. The `test_ct_point_io_standalone` target provides
`SECP256K1_CT_SERIALIZER_PROBE` and its old-conversion negative control.
`test_regression_ecdh_xy64_erase_standalone` provides
`SECP256K1_ECDH_CPP_TAINT_PROBE` and `SECP256K1_ECDH_SHIM_TAINT_PROBE`;
re-tainting the new internal boundary also requires the library build define
`SECP256K1_CT_ECDH_TAINT_PROBE` and runtime `SECP256K1_ECDH_NEW_PATH_PROBE`.
All taint builds need active `SECP256K1_CT_VALGRIND` markers.

These are scoped regression tests and probe entry points, not a newly completed
Valgrind/dudect or all-platform campaign. The generic portable field-kernel
limitations and earlier ladder-taint findings are not closed by this update.
No performance guarantee or end-to-end BIP-352 CT claim is added.

### 2026-09-23 — libbitcoin direct BIP-352 column scan boundary (#431)

`ufsecp::lbtc::bip352_scan_columns` accepts the receiver's secret
`scan_privkey32`. Its CPU fallback strictly parses the scalar and computes
`scan * tweak_pubkey` with `ct::scalar_mul`, then computes the shared-secret
tweak times G with `ct::generator_mul`. Public spend and tweak points are
decompressed before use. Adjacent equal public correlates are grouped, and
the candidate prefixes are compared with caller-supplied prefixes on the host.

The new finite-point serializers normalize the secret-derived Jacobian shared
point and candidate x coordinate using `ct::field_*` operations. An isolated
native 5x52 coordinate-taint run of `test_lbtc_direct_verify --bip352-ct-only`
reported **0 Valgrind errors** for those serializers. That measurement does
not cover the complete scan or establish the same result for every field
backend.

This is **not an end-to-end constant-time claim for the column scan**: the
CPU computes secret-derived candidates with `fast::Point::add` in
`spend.add(offset)`; that addition remains a timing review blocker. The match
loops also stop on a match or an invalid derived candidate, and the
observable match result depends on the receiver's scan key. The operation
also uses the batch worker pool for CPU groups; `max_threads == 1` requests
serial execution. Neither the public-data verification timing evidence nor
the CT evidence for the two scalar-multiplication primitives measures this
complete adapter.

The optional GPU hook calls the existing
`GpuBackend::bip352_scan_batch_multispend` on CUDA, OpenCL, or Metal and
compares its returned candidate prefixes on the host. It requires the
documented trusted single-tenant GPU environment. This change adds no GPU
kernel and supplies no new GPU timing evidence; it does not establish
constant-time behavior for the whole GPU scan or its host matching loop.

### 2026-09-21 v4.6.0 addendum — the cache-directory fix moves no CT boundary

The fixed-base cache relocation and the Pippenger shift guard (see
`AUDIT_CHANGELOG.md` under the same date) both sit outside the CT surface, and
this records why rather than leaving it to be re-derived.

**`secure_temp_cache_dir()` runs once, before any secret exists.** It is reached
from `ensure_built_locked()` during first-touch table construction, its inputs
are environment variables and the effective uid, and it never sees a private
key, a nonce or a signing share. The branches it introduces (`mkdir` result,
`lstat` result, ownership, permission bits) are all on filesystem state, not on
secret data, so none of them is a CT-relevant branch.

**The fixed-base table itself is public data.** It is precomputed multiples of
G. What the fix protects is its INTEGRITY -- a poisoned table yields wrong
public keys -- not its confidentiality, and the CT claims in the table below
concern neither.

**`pippenger.cpp` gains one comparison on `bit_idx`.** `bit_idx` is a window
position derived from the loop counter, not from the scalar's value, so the
added `bit_idx != 0` test is data-independent and does not introduce a
secret-dependent branch. Pippenger MSM is a verification-side, public-data path
in any case.

No CT surface in the table below changes, and no CT evidence is invalidated.

### 2026-09-21 v4.6.0 — the field defects were correctness, not CT

Two field bugs fixed in this release returned **wrong answers for legal inputs**,
and it is worth being precise about what that does and does not say about the CT
claims in this document.

`reduce()` lost a carry two ways and `FieldElement::sqrt()` returned a non-root
for ~18% of inputs. Neither is a timing or a secrecy defect: no branch or memory
access became data-dependent, no secret was exposed through a side channel, and
no CT boundary in the table below moved. They are correctness defects, and the CT
claims here were never claims of correctness.

What they do say is that **a green KAT suite is not evidence for a surface the
KATs do not reach**. Both defects live on the 4x64 chain; the BIP-340 hot paths
run on `FieldElement52`, so every standard vector stayed green while
`ellswift`, `adaptor`, `zk`, `pedersen` and `address` were calling a `sqrt()`
that returned a non-root almost one time in five. The affected callers that
re-validate `y^2 == x^3+7` turned it into a false negative; the ones that do not
propagated an off-curve y. The fix for `reduce()` is pinned by an exact-limb
regression over the trigger family plus 200,000 randomised rows compared at
32-byte granularity against big-integer ground truth — never through normalised
`operator==`, which is exactly what hid it.

CT surfaces in the table below are unaffected by both, and their evidence is
unchanged.

### 2026-09-21 v4.6.0 — OpenCL scan-only guard and a dead co-Z helper (no CT boundary moved)

Two changes reach the gate-watched surface (`CHANGELOG.md`); neither moves a CT
boundary, and this records why rather than waiving the classification.

**`src/opencl/kernels/secp256k1_extended.cl` — preprocessor only.**
`SECP256K1_OPENCL_SCAN_ONLY` (GitHub issue #415) excludes the four
`secp256k1_ct_*.cl` includes and every function and kernel that reaches them, so
a consumer embedding the six scan-only kernel files can compile them with the
`#include` lines stripped. No statement inside any CT routine was edited, no
scalar-mul or inverse was substituted, and the CT sign/ECDH paths are unchanged
in every build that defines nothing. Verified as text identity rather than
asserted: preprocessing the file with no macro defined, before (`17fceb76`) and
after, yields 2169 identical lines and an empty diff. A scan-only build does not
contain the CT sign paths at all — it therefore makes no CT claim about them.

**`src/cpu/src/point.cpp` — dead code removed.** `jac52_add_mixed_inplace_zr`
lost its only caller in `49925a1a` (the `#else` arm superseded by the co-Z table
build) and was deleted. A static function with no callers emits no code, so no
CT surface, timing profile or instruction sequence changes. The co-Z path that
replaced its call site keeps its existing coverage under
`regression_scalar_decomposition_and_comb`.

**CT evidence:** the residual CT artifacts in `audit/ci-evidence/` were refreshed
from real runs on this tree the same day — `regression_ct_ops` 42/42,
`adversarial_protocol` 793/793, `ecies_regression` 92/92, `fuzz_parsers`
580019/580019, `fuzz_address_bip32_ffi` 82976/82976, 0 crashes.

### 2026-09-15 legacy c_api renamed off libsecp256k1's namespace (no CT boundary moved)

The 36 functions in `bindings/c_api` are `ultrafast_secp256k1_*` now instead of
`secp256k1_*`. Recorded here because the secret-path gate classifies `CHANGELOG.md`
as a CT secret-bearing surface.

**No constant-time property changes and no cryptographic code was edited.** The
change is C symbol names, the export macro (`ULTRAFAST_SECP256K1_API`) and the
linker version script. Each renamed function's body is byte-for-byte what it was
and still delegates to the same `secp256k1::ct::*` / `secp256k1::fast::*` primitive.
No branch, table index or memory access became dependent on secret data, and none
stopped being.

The defect repaired is an ABI-safety one, not a timing one -- see
[`SECURITY_CLAIMS.md`](SECURITY_CLAIMS.md) and
[`SECRET_LIFECYCLE.md`](SECRET_LIFECYCLE.md) under the same date, and KB
`BCH-SCHNORR-CAPI-SYMBOL-COLLISION`.

### 2026-09-14 ecdsa.cpp — RFC 6979 algo16 became a parameter (no CT boundary moved)

`rfc6979_nonce_libsecp_compat` gained an optional `algo16` argument so the BCH
Schnorr shim can pass the tag its specification requires. `src/cpu/src/ecdsa.cpp`
is a CT secret-bearing surface, so the classification is answered here.

**The HMAC-DRBG is untouched.** The keydata layout, the V/K ladder, the fixed
two-iteration candidate loop (CT-001) and every `secure_erase` are exactly as
they were. The only change is which 16 bytes get appended: `nullptr` selects the
`"ECDSA\0..."` constant the function always used, so every existing caller --
`ct::ecdsa_sign_libsecp_compat` and its recoverable variant -- produces
byte-identical nonces to before. Verified by construction and by the 8/8
cross-check against an independent RFC 6979 implementation described in
`regression_bch_schnorr_spec`.

**What the tag fixes is a CT-adjacent secret-lifetime defect, not a timing one.**
The BCH shim previously called `rfc6979_nonce(d, msg)`, the same call
`ct::ecdsa_sign` makes, so one nonce could serve two different signature
equations and the private key is recoverable from that pair (16/16 recovered in
measurement). Domain separation removes it. See
[`SECRET_LIFECYCLE.md`](SECRET_LIFECYCLE.md) and
[`SECURITY_CLAIMS.md`](SECURITY_CLAIMS.md) under the same date, and KB
`BCH-SCHNORR-NONCE-REUSE-ECDSA`.

**On the shim side**, the quadratic-residue normalisation added to
`secp256k1_schnorr_sign` negates the nonce through `ct::scalar_cneg` with an
arithmetic mask (`(uint64_t)is_qr - 1`), not a branch. The bit being tested is
public -- `R.x` is the signature's `r`, and anyone can `lift_x(r)` and recompute
it -- but `k` is secret and the test sits inside its live range, so the negation
is constant time regardless. The matching verifier check is variable-time, which
is correct: verify is a public-data path (CT-VERIFY).

### 2026-09-14 ct_point.cpp / point.cpp — co-Z table build and CT SafeGCD inverse became the default (no CT boundary moved)

Two representation-search wins stopped being macro-guarded (`REPSEARCH_COZ_TABLE`,
`REPSEARCH_CT_SAFEGCD_INV`) and became the default build; the superseded branches
were deleted. The secret-path gate classifies `src/cpu/src/ct_point.cpp` as a CT
secret-bearing surface, so the classification is answered here rather than waived.

**Nothing became variable-time and nothing became constant-time.** Both changes
replace one implementation with another *inside* the existing boundary.

**1. Field inverse — `ct::ecmult_const_xonly`, `Point::batch_scalar_mul_fixed_k`.**
Was `FieldElement52::inverse()`: the Fermat chain, 255 squarings + 15 multiplies,
a fixed operation sequence and therefore constant-time. Is now `ct::field_inv`:
Bernstein-Yang SafeGCD, 10 x 59 = 590 divsteps on `__int128` platforms and
25 x 30 = 750 on the generic (MSVC / 32-bit) arm, both fixed iteration counts.
`ct_divsteps_59` (`src/cpu/src/ct_field.cpp:262`) carries no branch and no memory
access dependent on `f`, `g` or `zeta` — every conditional is a mask built from
`zeta >> 63` and `-(g & 1)`. Unlike `ct::scalar_inverse`, whose multiply chain
falls back to `fast::` when `__int128` is absent (SEC-001-INCOMPLETE), `field_inv`
is constant-time on **both** dispatch arms.

The inverted value is secret-derived at both sites, so a CT inverse is required at
both: the ElligatorSwift XDH denominator is `R.z^2 * g * xd` where `R = q*P_eff`
and `q` is the secret ECDH scalar; the batch site inverts a Montgomery prefix
product of Z-coordinates derived from the `KPlan`'s scalar, which callers populate
with the BIP-352 scan key (`bch_scan.cpp`, `sp_scan_batch_impl.hpp`, `address.cpp`)
— a CT-mandatory input per CLAUDE.md. A CT inverse is what runs, before and after.
Only the algorithm changed. See KB `FE52-INVERSE-CT-DOMINATED`.

**2. Odd-multiple table build — four sites.** Was a mixed-add chain on an
isomorphic curve (`jac_add_ge_var_zr`, `jac52_add_mixed_inplace_zr`); is now a
co-Z DBLU/ZADDU chain (`jac52_dblu_ct` / `jac52_zaddu_ct` on the ct:: track,
`jac52_dblu` / `jac52_zaddu_zr` on the fast:: track). **The table is a function of
the base point only, never of the scalar** — before and after. `jac52_zaddu_ct`
does hold one variable-time test, `dx.normalizes_to_zero_var()`, guarding the
`X1 == X2` case ZADDU cannot represent; its timing is therefore a function of the
base point, exactly as the `_var` mixed add it replaces already was. No branch,
table index or memory access in either chain reads the secret scalar. The scalar
is consumed afterwards by `table_lookup_core`, which is untouched.

Evidence: `regression_table_build_invariants` gained section (5)
`ecmult_const_xonly(x_P, 1, q) == x(q*P)` over 16 random cases and section (6)
`batch_scalar_mul_fixed_k(k) == scalar_mul(k)` over 64 points — the two sites
newly on the default path that sections (1)-(3) could not reach. 12/12 checks
pass. Equivalence to the previously-measured macro-on build was established as
code identity: `point.cpp` and `ct_point.cpp` compiled from the default tree emit
assembly identical to the prior tree compiled with both macros set.

### 2026-09-07 precompute.cpp / CMakeLists.txt — fixed-base cache location and no-op reconfigure (no CT surface moved)

Two changes touch files the secret-path gate classifies as CT secret-bearing
surfaces. **Neither moves a CT boundary**, and both are recorded here so the
classification is answered rather than waived.

- **Fixed-base table cached in the per-user cache directory.** The table is
  precomputed multiples of the generator `G` — public data any observer can
  derive independently. Nothing secret has ever been written to
  `cache_w*.bin`, so no advisory is warranted for the earlier revisions that
  left the file in the caller's working directory; it was a filesystem
  side-effect defect, not a key-material disclosure. What changed is the
  location (per-user cache directory, created and kept; never the CWD) and the
  fact that the table is built once and loaded thereafter.
- **`configure_fixed_base()` no longer invalidates on a no-op.** Re-applying an
  identical configuration keeps the built context instead of recomputing it. The
  comparison is over the whole configuration, so the context and `g_config` can
  never describe different settings. This changes *when* a public table is
  rebuilt, not what any constant-time primitive does: `ct::scalar_mul`,
  `ct::generator_mul_blinded`, `ct::scalar_inverse` and the RFC 6979 nonce path
  are untouched, and no branch or memory access anywhere became dependent on
  secret data.

Tests: `regression_fixed_base_cache_lifecycle` (FBC-1..4, location and
lifetime, both cache modes) and `regression_precompute_noop_reconfigure`
(PNR-1..4, with a negative control proving a real config change still
invalidates).

### 2026-09-10 GPU compact-ECDSA strict-range guard — public-data boundary, no CT boundary moved

Three files the secret-path gate classifies as CT secret-bearing surfaces gained
a guard for public compact-signature scalars; this section answers that
classification rather than waiving it.

- **`src/cuda/include/ecdsa.cuh`**, **`src/metal/shaders/secp256k1_extended.h`**,
  **`src/opencl/kernels/secp256k1_extended.cl`**: `ecdsa_verify()` on each GPU
  backend now rejects a compact signature with `r >= n` or `s >= n` up front,
  before the `scalar_inverse()` call, so every verify route (single, batch,
  collect, sign-and-verify) shares one strict compact contract. The CUDA batch
  path keeps its strict `ecdsa_sig_parse_compact_strict` parse.
- **CT status: unchanged.** The guard compares the two public scalars `r`, `s`
  of the already-parsed 64-byte compact signature against the group order `n`.
  Both are public data; the comparisons are public-data (variable-time is
  permitted for them), and no secret value reaches a new branch or memory
  access. No CT primitive changed, no constant-time code path was made
  variable-time, and the failure exits return before the verify math that
  touches secret-path state.
- Pinned by `gpu_ecdsa_compact_range` — `[A]` CPU-only source gate asserts the
  guard sits inside each backend's `ecdsa_verify()` body before
  `scalar_inverse()` (kills any helper-stranding regression), `[B]` on-device
  small-`(r,s)` boundary differential through batch and collect vs the CPU
  strict oracle. Identity with the entry in `docs/SECURITY_CLAIMS.md`.

### 2026-06-01 ct_sign.cpp — variable-length Schnorr CT sign overload (SHIM-001 restore)

- **`src/cpu/src/ct_sign.cpp`**: added a variable-length overload
  `secp256k1::ct::schnorr_sign(const SchnorrKeypair& kp, const uint8_t* msg, size_t msglen,
  const std::array<uint8_t,32>& aux_rand)` to support BIP-340 `sign_custom` for arbitrary
  message lengths (matching upstream libsecp256k1). It is an **exact mirror of the audited
  fixed-32 CT path** in the same file — only the message length fed into the two tagged
  hashes differs:
  - nonce point uses `ct::generator_mul_blinded(k')` (DPA-blinded), identical to the 32-byte path;
  - the even-Y conditional negate uses `ct::scalar_cneg(k', bool_to_mask(r_y_odd))` — branchless,
    no secret-dependent branch on the nonce parity;
  - `s = ct::scalar_add(k, ct::scalar_mul(e, kp.d))` uses the branchless CT scalar ops
    (not `fast::` operators), so the secret nonce `k` and key `kp.d` never hit a data-dependent
    reduction branch (V-01 discipline);
  - every secret-derived buffer is erased on return: `d_bytes`, `t_hash`, `t`, `nonce_input`,
    `rand_hash`, `challenge_input`, `e_hash`, `e`, `k_prime`, `k`. The `nonce_input` /
    `challenge_input` `secure_erase` covers the full `64 + msglen` span (SBO stack buffer for
    `msglen ≤ 512`, heap allocation beyond), so the message-bearing hash inputs are wiped
    regardless of length.
- **CT classification:** secret inputs are `kp.d` (signing key) and the derived nonce `k'/k`;
  the message and `R.x` are public. No new variable-time primitive is introduced — the overload
  reuses the same `ct::*` primitives already proven on the fixed-32 path. For `msglen == 32` the
  output is byte-identical to `schnorr_sign(kp, msg32, aux)` (verified by micro-test), so the CT
  guarantees of the established path carry over unchanged.
- The shim `secp256k1_schnorrsig_sign_custom` keeps forwarding `msglen == 32` to `sign32` and
  routes `msglen != 32` through this overload; the 32-byte hot path is untouched.

### 2026-05-27 ct_sign.cpp — is_zero_ct on r in ecdsa_sign_verified paths (CT-008/SEC-004)

- **`src/cpu/src/ct_sign.cpp`**: `secp256k1::ct::ecdsa_sign_verified` and
  `ecdsa_sign_hedged_verified` used `sig.r.is_zero()` (variable-time branch)
  to guard the fault-attack verification step. `sig.r` is derived from the
  nonce point `R.x mod n` — a nonce-adjacent value. Changed to `sig.r.is_zero_ct()`
  (branchless `cmov`-based check) in both functions.
- **`src/cpu/src/ecdsa.cpp`**: Same fix applied to the fast-path
  `secp256k1::ecdsa_sign_hedged_verified` (benchmarking path, not production ABI,
  but consistent defence-in-depth).
- **CT status**: Both CT signing paths now uniformly use `is_zero_ct()` on all
  nonce-derived values. No algorithm change.

### 2026-05-14 ct_field.cpp — drop dead add256 + add_carry_u64 (build-only cleanup)

- **`src/cpu/src/ct_field.cpp`**: After delegating field_add/sub/neg
  to the fast layer, `static add256(...)` and its helper
  `add_carry_u64(...)` had no callers. Kept them under `#if 0` initially,
  but `-Werror=unused-function` (Security Audit / Build with -Werror)
  and MSVC's lack of `__attribute__((noinline))` made the leftover
  declarations into build errors. Deleted both. The corresponding
  `__int128` carry-chain I had added to `sub256` is also reverted —
  the `__int128` literal trips `-Werror=pedantic` and `sub256`'s
  portable branchless path was correct already.
- **CT status**: No behavior change.

### 2026-05-14 ct_field.cpp — delegate field_add/sub/neg to fast::operator

- **`src/cpu/src/ct_field.cpp`**: `ct::field_add`, `ct::field_sub`, and
  `ct::field_neg` previously had hand-written parallel implementations
  (custom `add256` / `sub256` / `cmov256` chains, asm-volatile barriers,
  `__builtin_addcll` Clang intrinsics) intended to defeat LTO constant
  propagation. Under Clang ThinLTO at -O3 (the actual flags every CI
  Clang job uses — `-O3 -flto=thin` is forced onto the library target in
  `src/cpu/CMakeLists.txt`), Clang miscompiled the chain: `r[0..3]`
  received zeros while the sum-of-low-limb ended up in the carry-return
  slot. Reproduced locally with Clang 18; CI / linux (clang-17, Debug+Release)
  / Sanitizers (TSan/MSan/ASan) hit the same code path.
- **Fix**: Delegate to the fast layer (`a + b`, `a - b`, `zero - a`).
  `fast::add_impl` / `sub_impl` / unary minus in `src/cpu/src/field.cpp`
  are themselves branchless: 4 × `add64` followed by an XOR-mask
  conditional reduce — no data-dependent branches, no `__builtin_addcll`,
  no LTO-sensitive asm barriers. The CT guarantee is preserved by
  construction.
- **CT status**: Equivalent to the previous CT path (same branchless
  primitives); the LTO-defeating barriers were redundant because the
  fast path already has the constant-time property. Verified locally:
  `test_comprehensive` 12023/0/10 pass under Clang -O3+ThinLTO Debug,
  `test_field_52` 270/270 under ASM=OFF Debug.

### 2026-05-14 ct_field.cpp — Clang sanitizer detection fix (build-only, no semantics change)

- **`src/cpu/src/ct_field.cpp`**: Five preprocessor sites that opted out of
  the `__builtin_addcll/_subcll` ADCX path and the LTO-defeating `asm volatile("" :::"memory")`
  barriers under sanitizers used the GCC-only spellings
  `__SANITIZE_THREAD__ / __SANITIZE_ADDRESS__ / __SANITIZE_MEMORY__`. Clang does
  **not** define these macros (Clang exposes sanitizer state via `__has_feature`),
  so under Clang TSan/MSan/ASan the barriers still ran and produced false
  positives like `FAIL: ct field_add` / `FAIL: ct add #1..#64` in
  `test_comprehensive`. A new `SECP256K1_HAS_SANITIZER` macro consolidates
  both detection forms.
- **CT status**: Pure build-time detection change. No CT algorithm changed:
  `field_add / field_sub / field_mul / field_sqr / field_inv` still use the
  same constant-time primitives (`add256`, `sub256`, FE52 `mul_inner`).
  The asm-memory-barrier intent (defeat LTO constant propagation that would
  shorten the carry chain) is preserved on production Release builds.
- **Audit method**: Compiler-defined-macro audit (cross-checked GCC manual
  §3.16 + Clang `__has_feature` docs); local `clang++ -fsanitize=thread`
  build of `ct_field.cpp` confirms no warnings on the new guard.

### 2026-05-11 ct_sign.cpp — ecdsa_sign_hedged_recoverable added

- **`src/cpu/src/ct_sign.cpp`**: Added `ct::ecdsa_sign_hedged_recoverable()` that returns a
  `RecoverableSignature` with recid derived from `R.y` parity and `R.x >= n` overflow during
  signing. This eliminates the 4× `ecdsa_recover` loop in `secp256k1_ecdsa_sign_recoverable`
  when `ndata != NULL` (Bitcoin Core CKey::Sign path).
- **CT status**: Recovery ID derivation is branchless throughout:
  - Bit 0: `R.y().limbs()[0] & 1u` — no branch on nonce
  - Bit 1: byte-cascade comparison of `r_bytes` vs `ORDER_BYTES` — no early exit
  - Low-S flip: `recid ^= static_cast<int>(high_mask & 1)` — branchless XOR
  - All secret nonce material erased via `secure_erase` before return
- **Audit method**: Code review; same implementation pattern as `ct::ecdsa_sign_recoverable`
  (line 302), which passed CT review in 2026-05-07 ultrareview.



**UltrafastSecp256k1 v4.6.2** -- CT Layer Methodology & Audit Status

### 2026-05-11 ct_point::scalar_mul_jac_fe52_z1 — HAMBURG=true (xdh-dedicated path)

- **`ct_point.cpp` / `ellswift.cpp`**: `scalar_mul_jac_fe52_z1` is exclusively
  called from `ecmult_const_xonly` (→ ellswift_xdh). In that path, K_CONST
  encoding guarantees `m_val = S1+S2 ≠ 0` for all 52 intermediate
  `unified_add_core` calls, making the degenerate-case check
  (`fe52_normalizes_to_zero` + cmovs, ~18 ns/call) provably unnecessary.
  Setting `HAMBURG=true` removes the check, saving ~18 ns per unified-add.
- **CT impact: none.** The degenerate-case branch handles `m_val==0` (P=-Q),
  which is the point at infinity result. Skipping it is safe here because the
  K_CONST encoding prohibits m_val=0; the proof is in the commit message for
  `9f524c1f`. The CT execution trace of the scalar multiplication is unchanged.

### 2026-05-11 ct_field::field_mul/field_sqr — same sanitizer guard on asm barriers

- **`ct_field.cpp` (`field_mul`, `field_sqr`)**: The `+r` limb barriers
  `asm volatile("" : "+r"(fa.n[i]) ...)` were unconditional. Under TSan the
  asm barriers in field_mul/field_sqr cause incorrect shadow-memory tracking
  of FE52 limbs, producing wrong multiplication/squaring results. This manifested
  as `point_is_on_curve(kG) == false` for k=8..16 under TSan comprehensive tests,
  because `ctp::point_is_on_curve` uses field_sqr and field_mul internally.
  Added the same `!__SANITIZE_{THREAD,ADDRESS,MEMORY}__` guard to all FE52 limb
  barriers, consistent with the fix applied to field_add.
- **CT contract unchanged.** Under sanitizers, LTO and constant-propagation are
  disabled by the sanitizer build flags, making the barriers redundant.

### 2026-05-11 ct_field::field_add — disable asm barriers under sanitizers

- **`ct_field.cpp` (`field_add`, `field_sub`, `field_neg`)**: The compiler
  barrier `asm volatile("" : "+r"(ptr) : : "memory")` was applied
  unconditionally in `field_add`. This barrier prevents LTO constant-propagation
  of known operands into the carry chain (CT timing guarantee), but under TSan
  the `"memory"` clobber causes TSan to apply incorrect shadow-memory analysis
  to the subsequent reads through `a_ptr`/`b_ptr`, producing field values that
  differ from `fast::operator+`. Added the same `!__SANITIZE_{THREAD,ADDRESS,
  MEMORY}__` guard that already protects `__builtin_addcll` / `__builtin_subcll`.
  The barrier is still active for non-sanitized Release builds where CT timing
  matters.
- **CT impact: none.** Under sanitizers, compiler optimizations that could
  break constant-time (LTO constant-propagation) are disabled by the sanitizer
  itself, making the barrier redundant.

### 2026-05-10 ct_field::sub256 — same x86-64+no-sanitizer guard as add256

- **`ct_field.cpp` (`sub256`)**: `__builtin_subcll` (SBB borrow-chain instruction)
  now has the same guards as `__builtin_addcll`: restricted to x86-64 and
  excluded under ASan/TSan/MSan. Previously `sub256` used `__builtin_subcll`
  unconditionally for all Clang targets, causing two categories of failure:
  (a) On ARM64 (macOS M-series): same carry/borrow-flag reliability issue as
  `__builtin_addcll` — Clang doesn't always emit a pure SUBS/SBCS chain.
  (b) Under TSan: sanitizer instrumentation inserted between calls corrupts the
  borrow flag, causing `ct::field_sub` / `ct::field_add` (which calls sub256
  for the conditional p-subtraction) / `ct::field_neg` to return wrong values.
  This produced 84 comprehensive-test failures under TSan and matching failures
  on the macOS ARM64 runner.
- **CT impact: none.** Portable carry-borrow chain produces the same results
  on all platforms and is equally branchless.

### 2026-05-10 ct_field::add256 — x86-64-only __builtin_addcll (ARM64 fix)

- **`ct_field.cpp` (`add256`)**: The Clang `__builtin_addcll` carry-chain path
  is now restricted to `__x86_64__`. On ARM64/AArch64 (Apple Silicon, macOS),
  Clang does not always emit a pure ADDS/ADCS carry sequence — the carry flag
  can be broken between chained `__builtin_addcll` calls, producing wrong
  256-bit sums. Removing the ARM64 path causes the portable carry-chain fallback
  to be used instead, which is correct on all platforms.
- **CT impact: none.** The portable fallback is branchless and produces the
  same numerical result on x86-64 and ARM64. The x86-64 ADCX/ADOX path remains
  unchanged for performance.
- **Evidence**: `test_ct_field_differential` in `audit_ct.cpp` confirmed
  `ct::field_add != fast::operator+` on Apple M-series runners before this fix.
  After the fix, both paths produce identical results on all tested platforms.

### 2026-05-10 ct_field::add256 — instrumentation-safe carry chain

- **`ct_field.cpp` (`add256`)**: The Clang fast-path that uses `__builtin_addcll`
  (which lowers to ADCX/ADOX) is now disabled under AddressSanitizer,
  ThreadSanitizer, MemorySanitizer, and coverage instrumentation
  (`LLVM_PROFILE_ENABLED`). The instrumentation inserts code *between* the
  back-to-back addcll calls, and that code can clobber the x86 carry flag,
  producing wrong sums. The build now falls through to the portable
  `add_carry_u64` chain in those configurations.
- **CT impact: none.** Both paths are constant-time:
  - The `__builtin_addcll` chain has no secret-dependent branches; the carry
    is propagated through ADCX/ADOX which take constant time on every Intel
    Broadwell+ / AMD Zen+ part we support.
  - The portable `add_carry_u64` fallback uses unsigned arithmetic and a
    constant-time bit operation (`(sum < a)`) for carry detection — no
    branches, no table lookups, no secret-dependent memory access.
- **Why this is a CT-relevant note even though semantics don't change**:
  `add256` runs on secret 256-bit field elements (private keys, nonces) in
  every CT scalar multiplication step. The Documentation Discipline rule
  requires every change to a CT secret-bearing surface to be documented even
  when the change is a portability fix rather than a CT semantics change,
  so reviewers can confirm CT properties remain intact.
- **Unrelated regression context**: This is paired with the ongoing FE52/FE26
  mul-impl repair (see commits ae606aac, d152c883). The two are independent
  but were investigated together because Clang+coverage builds were failing
  test_field_52 and test_field_26 simultaneously.

### 2026-05-07 Multi-agent ultrareview CT Fixes (CA-007 hardening + verify/sign boundary)

- **`scalar.cpp` (`Scalar::negate`)**: CT-hardened via CA-007 — replaces early-return on
  `is_zero()` (variable-time) with unconditional `ORDER − s` + mask-to-zero via `uint64_t`
  arithmetic. Eliminates timing leak about whether a secret scalar is zero (nonce/key negation).
- **`scalar.cpp` (`Scalar::negate_var`)**: NEW — variable-time negate for PUBLIC scalars only
  (challenge hash `e`, GLV sub-scalars). Branches on `is_zero()` (safe: public data cannot
  leak secrets). Used in `schnorr_verify` for `neg_e = e.negate_var()`. Design rule: every
  CT primitive that protects secret paths must have a paired `_var` counterpart for public paths.
- **`taproot.cpp` (`taproot_tweak_privkey`)**: Replaced `d + t` (fast::Scalar::operator+,
  secret-dependent branch in modular reduction) with `ct::scalar_add(d, t)`. `d` is the
  conditionally-negated private key — a secret scalar.
- **`frost.cpp` (`frost_sign`)**: Replaced `if (negate_R) { d.negate(); ei.negate(); }` and
  `if (negate_key) { s_i.negate(); }` branches on secret nonces/signing share with
  `ct::bool_to_mask` + `ct::scalar_cneg`. Branch condition `negate_R`/`negate_key` is public
  (R.y parity), but branching on a public condition with secret operands may produce
  conditional branch instructions that expose the nonce value to branch-predictor side-channels.
- **`point.cpp` (`dual_scalar_mul_gen_point` MSVC path)**: Removed `ct::generator_mul(a)` from
  the MSVC 4×64 verify path where `a = sig.s` (public). CT on public data adds overhead with
  zero security benefit. Variable-time verify is correct when all inputs are public.
- **Verify paths (`schnorr_verify` both overloads)**: Use `e.negate_var()` (not `e.negate()`)
  for the challenge scalar — public data.

CT/verify boundary rule established:
  CT mandatory: private key, nonce, signing share, ECDH scalar, BIP-352 scan key.
  Variable-time correct: all verify inputs (pubkey, sig.r, sig.s, msg_hash, challenge e).

### 2026-05-06 Performance Review CT Fix (SEC-01)

- `frost.cpp` (`frost_keygen_finalize`): Replaced
  `Point::generator().scalar_mul(share.value)` with `ct::generator_mul(share.value)`.
  `share.value` is a secret polynomial evaluation (private key material in DKG context).
  The previous code used the variable-time GLV/Strauss wNAF path on a secret scalar,
  violating guardrail 1 (CT mandate for secret-bearing signing paths) and guardrail 12
  (`Point::generator().scalar_mul` banned for private keys). The fix routes through the
  Hamburg precomputed comb (`ct::generator_mul`) which has no secret-dependent branches
  or data-dependent timing. Regression PoC: `test_exploit_frost_secret_share_ct.cpp`.

### 2026-05-05 Performance Audit CT Refactoring (A-04, A-06)

- `ct_sign.cpp` (`ct::ecdsa_sign`): Replaced `r_fe.to_bytes()` + `Scalar::from_bytes(r_bytes)`
  with `Scalar::from_limbs(R.x().limbs())`. **No CT impact**: `r = R.x mod n` is the
  x-coordinate of the public nonce point R = k·G — public data, not a secret scalar. The CT
  properties of the secret nonce `k` and private key `d` are unchanged. The `ge(ORDER)`
  reduction check in `from_limbs()` is still performed (secp256k1 p > n).
- `ct_sign.cpp` (`ct::ecdsa_sign_hedged`): Applied the same `from_limbs()` optimization
  (replacing the redundant `to_bytes()+from_bytes()` round-trip) to the hedged ECDSA sign
  variant. **No CT impact**: identical reasoning applies — `R.x` is public curve-point data.
  The hedged path was not updated in A-04; this closes the inconsistency with `ct::ecdsa_sign`.
- `point.cpp` (`derive_phi52_table`): Function-local `static const FieldElement52 beta52`
  replaced with the existing file-scope `kBeta52_pt`. Both hold the public GLV constant β.
  No CT impact — β is never secret-derived.

- `ct_point.cpp`: Extracted the `make_v` GLV half-scalar CT-negate/increment lambda from four
  separate function-local copies (`scalar_mul_jac_fe52_z1`, `scalar_mul_jac`, `scalar_mul_prebuilt`,
  `scalar_mul_prebuilt_fast`) into a single file-scoped `SECP256K1_INLINE static ct_glv_make_v`
  helper. **No logic change** — all four copies were verified identical in behavior. The refactor
  eliminates copy-paste divergence risk in a security-critical CT path: a future bug fix would
  previously need to be applied to all four copies independently.
- `schnorr.hpp` / `schnorr.cpp`: Added `[[nodiscard]]` and `noexcept` to the two
  `schnorr_verify(SchnorrXonlyPubkey, ...)` overloads. No functional change; `noexcept` is
  accurate since neither overload allocates or throws.

### 2026-05-05 Red-Team Audit CT Changes

- `ct_sign.cpp`: Comments added to `Scalar::from_bytes` callsites to clarify
  they parse public data (message hash, curve-point x-coord), not secret scalars.
  No functional change to CT signing paths.
- CUDA: added `ct::ct_ecdsa_sign_recoverable` (CT recovery ID computation, branchless
  overflow check). All GPU recoverable signing now CT.
- OpenCL: `ct_ecdsa_sign_verified_impl` now calls `ct_generator_mul_impl` (CT pubkey)
  + `ecdsa_verify_impl` (fault countermeasure no longer a stub).
- OpenCL: `ct_schnorr_sign_verified_impl` now calls `schnorr_verify_impl`.
- Metal: added `ct_ecdsa_sign_recoverable_metal` with branchless recid computation.
  All Metal signing functions now route through CT equivalents.

---

## Overview

The constant-time (CT) layer provides side-channel resistant operations for secret key material. It is available on **all backends**:

- **CPU**: `secp256k1::ct::` namespace (headers in `src/cpu/include/secp256k1/ct/`)
- **CUDA GPU**: `secp256k1::cuda::ct::` namespace (headers in `src/cuda/include/ct/`)
- **OpenCL GPU**: CT kernels in `src/opencl/kernels/` (`secp256k1_ct_sign.cl`, `secp256k1_ct_zk.cl`)
- **Metal GPU**: CT shaders in `src/metal/shaders/` (`secp256k1_ct_sign.metal`, `secp256k1_ct_zk.metal`)

The FAST layer (`secp256k1::fast::` on CPU, `secp256k1::cuda::` on GPU) is explicitly variable-time for maximum throughput.

**Principle**: Any operation that touches secret data (private keys, nonces, intermediate scalars) MUST use `ct::` functions on CPU. GPU operations that accept secret keys (`ecdh_batch`, `bip352_scan_batch`, `bip324_aead_*_batch`) require a trusted single-tenant environment. The default `fast::` namespace is allowed only when all inputs are public.

The repository preflight also consumes graph-linked coverage metadata from
`ci/build_project_graph.py`. That metadata records both standalone CTest
coverage and selected unified-audit module coverage for core files; it is used
for coverage-gap reporting and does not replace the executable CT tests.

CT secret-bearing implementation changes are under stricter change control:
`ci/check_secret_path_changes.py` requires paired updates to this document
and `docs/SECURITY_CLAIMS.md` whenever CT-layer secret surfaces change.

---

## CT-Routed C ABI Surface

The audit gate treats the graph's `abi_routing` table as the routing source of
truth for secret-bearing C ABI entry points. The functions below are the
currently documented CT-routed ABI surface together with the routed internal
operation recorded by the graph.

| C ABI Function | Routed Internal Operation |
|----------------|---------------------------|
| `ufsecp_bip32_derive` | `CKD_priv or CKD_pub` |
| `ufsecp_bip32_derive_path` | `multi-level CKD` |
| `ufsecp_bip32_master` | `HMAC-SHA512(seed)` |
| `ufsecp_bip32_privkey` | `ExtendedKey::privkey()` |
| `ufsecp_bip39_generate` | `bip39_generate(strength)` |
| `ufsecp_bip39_to_seed` | `PBKDF2-SHA512(mnemonic, passphrase)` |
| `ufsecp_btc_message_sign` | `btc_message_sign(msg, sk)` |
| `ufsecp_coin_hd_derive` | `coin_hd_derive(coin, xprv, path)` |
| `ufsecp_ecdh` | `ct::detail::scalar_mul_jacobian + point_to_compressed33 + SHA256` |
| `ufsecp_ecdh_raw` | `ct::detail::scalar_mul_jacobian + point_to_x32` |
| `ufsecp_ecdh_xonly` | `ct::detail::scalar_mul_jacobian + point_to_x32 + SHA256` |
| `ufsecp_ecdsa_adaptor_adapt` | `ecdsa_adaptor_adapt` |
| `ufsecp_ecdsa_adaptor_sign` | `ct::ecdsa_adaptor_sign(sk)` |
| `ufsecp_ecdsa_sign` | `ct::ecdsa_sign(msg, sk)` |
| `ufsecp_ecdsa_sign_recoverable` | `ct::ecdsa_sign + recovery_id` |
| `ufsecp_ecdsa_sign_verified` | `ct::ecdsa_sign + ecdsa_verify` |
| `ufsecp_ecies_decrypt` | `ecies_decrypt(sk, ciphertext)` |
| `ufsecp_ecies_encrypt` | `ecies_encrypt(pubkey, msg)` |
| `ufsecp_eth_sign` | `ct::ecdsa_sign(keccak(msg), sk) + v` |
| `ufsecp_frost_keygen_begin` | `frost_keygen_begin` |
| `ufsecp_frost_sign` | `ct::frost_sign(sk, nonce)` |
| `ufsecp_frost_sign_nonce_gen` | `frost_sign_nonce_gen` |
| `ufsecp_musig2_nonce_gen` | `musig2_nonce_gen(sk)` |
| `ufsecp_musig2_partial_sign` | `ct::musig2_partial_sign(sk)` |
| `ufsecp_musig2_start_sign_session` | `musig2_session_init` |
| `ufsecp_pedersen_blind_sum` | `blind factor sum` |
| `ufsecp_pubkey_create` | `ct::generator_mul(sk)` |
| `ufsecp_pubkey_create_uncompressed` | `ct::generator_mul(sk)` |
| `ufsecp_pubkey_xonly` | `schnorr_pubkey(sk)` |
| `ufsecp_schnorr_adaptor_adapt` | `adaptor_adapt(pre_sig, secret)` |
| `ufsecp_schnorr_adaptor_sign` | `ct::adaptor_sign(sk)` |
| `ufsecp_schnorr_keypair` | `generate_schnorr_keypair(sk)` |
| `ufsecp_schnorr_sign` | `ct::schnorr_sign(msg, sk)` |
| `ufsecp_schnorr_sign_verified` | `ct::schnorr_sign + schnorr_verify` |
| `ufsecp_seckey_negate` | `Scalar::negate` |
| `ufsecp_seckey_tweak_add` | `ct scalar add + validate` |
| `ufsecp_seckey_tweak_mul` | `ct scalar mul + validate` |
| `ufsecp_seckey_verify` | `Scalar::parse_bytes_strict_nonzero` |
| `ufsecp_silent_payment_create_output` | `silent_payment_create_output` |
| `ufsecp_silent_payment_scan` | `silent_payment_scan` |
| `ufsecp_taproot_tweak_seckey` | `taproot_tweak_seckey(sk, merkle)` |
| `ufsecp_zk_dleq_prove` | `dleq_prove(sk)` |
| `ufsecp_zk_knowledge_prove` | `prove_knowledge(sk)` |
| `ufsecp_zk_range_proof_create` | `create_range_proof` |

These entries are routing and review anchors, not stand-alone proof claims.
They document which exported secret-bearing APIs are expected to stay on the CT
path and which internal CT-sensitive primitive or wrapper currently implements
that route.

---

## CT Layer Architecture

### CPU CT Layer

```
secp256k1::ct::
+-- ops.hpp          -- Low-level CT primitives (cmov, select, cswap)
+-- field.hpp        -- CT field multiplication, inversion, square
+-- scalar.hpp       -- CT scalar multiplication, addition
+-- point.hpp        -- CT point operations (scalar_mul, generator_mul)
+-- ct_utils.hpp     -- Utility: timing barriers, constant-time comparison

secp256k1::fast::
+-- field_branchless.hpp  -- Branchless field_select (bitwise cmov)
+-- ...                   -- Variable-time (NOT for secrets)
```

### GPU CT Layer

```
secp256k1::cuda::ct::
+-- ct_ops.cuh       -- CT primitives: value_barrier (PTX asm), masks, cmov, cswap
+-- ct_field.cuh     -- CT field: add, sub, neg, mul, sqr, inv, half, cmov, cswap
+-- ct_scalar.cuh    -- CT scalar: add, sub, neg, half, mul, inverse (Fermat), GLV
+-- ct_point.cuh     -- CT point: dbl, add_mixed (Brier-Joye 7M+5S), add (11M+6S),
|                       scalar_mul (GLV + bit-by-bit), generator_mul
+-- ct_sign.cuh      -- CT signing: ct_ecdsa_sign, ct_schnorr_sign, ct_schnorr_keypair
```

The GPU CT layer mirrors the CPU CT layer with identical algorithms adapted for CUDA:
- `value_barrier()` uses PTX `asm volatile` to prevent compiler optimization
- All mask operations are 64-bit (matching GPU's native word size)
- No branch divergence on secret data (critical for SIMT warp execution)
- Field/scalar heavy arithmetic delegates to fast-path (same cost) with CT
  control flow wrapping

#### GPU CT Usage

```cuda
#include "ct/ct_sign.cuh"

__global__ void sign_kernel(const uint8_t* msg, const Scalar* privkey,
                            ECDSASignatureGPU* sig, bool* ok) {
    // CT ECDSA sign -- constant-time k*G, k^-1, scalar ops
    *ok = secp256k1::cuda::ct::ct_ecdsa_sign(msg, privkey, sig);
}

__global__ void schnorr_kernel(const Scalar* privkey, const uint8_t* msg,
                               const uint8_t* aux, SchnorrSignatureGPU* sig, bool* ok) {
    // CT Schnorr sign -- constant-time nonce generation + signing
    *ok = secp256k1::cuda::ct::ct_schnorr_sign(privkey, msg, aux, sig);
}
```

#### GPU CT Benchmark Results (RTX 5060 Ti, SM 12.0)

| Operation | FAST | CT | CT/FAST Overhead |
|-----------|------|-----|------------------|
| k*G (generator) | 129.1 ns | 341.9 ns | 2.65x |
| k*P (scalar mul) | -- | 347.2 ns | -- |
| ECDSA sign | 211.1 ns | 433.9 ns | 2.06x |
| Schnorr sign | 284.9 ns | 715.8 ns | 2.51x |

GPU CT throughput: **2.30M ECDSA sign/sec**, **1.40M Schnorr sign/sec**.

#### GPU CT ZK Layer

```
secp256k1::cuda::ct::
+-- ct_zk.cuh        -- CT ZK proving: knowledge proof (Schnorr sigma), DLEQ proof
                        Uses ct_scalar_mul for secret nonce operations, ct_jacobian_to_affine,
                        scalar_cneg for BIP-340 Y-parity normalization.
                        Deterministic nonce: SHA-256 tagged hash with XOR hedging.
```

The GPU CT ZK layer ensures that all proving operations (which handle secret keys
and nonces) use constant-time scalar multiplication and arithmetic. Verification
operations use the fast path since all inputs are public.

| CT ZK Operation | Approach | Secret Data Protected |
|-----------------|----------|----------------------|
| `ct_knowledge_prove_device` | CT `ct_scalar_mul` for k*B | Nonce k, secret key |
| `ct_knowledge_prove_generator_device` | CT `ct_scalar_mul` for k*G | Nonce k, secret key |
| `ct_dleq_prove_device` | 2x CT `ct_scalar_mul` for k*G, k*H | Nonce k, secret key |
| `knowledge_verify_device` | Fast-path `scalar_mul` | N/A (public data) |
| `dleq_verify_device` | Fast-path `scalar_mul` | N/A (public data) |

**Test coverage:** `test_ct_smoke.cu` tests 8-9 verify CT knowledge prove + verify and
CT DLEQ prove + verify round-trips on GPU. All 9/9 tests pass.

### OpenCL CT Layer

```
src/opencl/kernels/
+-- secp256k1_ct_sign.cl    -- CT ECDSA sign, CT Schnorr sign, CT keypair create
+-- secp256k1_ct_zk.cl      -- CT ZK proving: knowledge proof, DLEQ proof
```

The OpenCL CT layer mirrors the CUDA CT implementation with OpenCL-native barriers:
- `value_barrier()` via inline OpenCL `asm volatile` or volatile loads
- Branchless masks and conditional moves on all secret-dependent paths
- CT scalar multiplication with fixed iteration count (GLV + signed-digit)
- Audited via `opencl_audit_runner` ( 482 modules including CT sections)

### Metal CT Layer

```
src/metal/shaders/
+-- secp256k1_ct_sign.metal -- CT ECDSA sign, CT Schnorr sign, CT keypair create
+-- secp256k1_ct_zk.metal   -- CT ZK proving: knowledge proof, DLEQ proof
```

The Metal CT layer uses Metal Shading Language (MSL) with:
- `value_barrier()` via threadgroup memory fence pattern
- Identical algorithms to CUDA/OpenCL CT layers
- Audited via `metal_audit_runner` ( 482 modules including CT sections)

---

## CT Guarantees

### What IS Constant-Time

| Operation | Implementation | Guarantee Level |
|-----------|---------------|-----------------|
| `ct::scalar_mul(P, k)` | GLV + signed-digit, fixed iteration count | Strong |
| `ct::generator_mul(k)` | Hamburg comb, precomputed table | Strong |
| `ct::ecmult_const_xonly(xn, xd, q)` | Variable-time validation of public xn/xd, then existing secret-q CT ladder | Profile-scoped; see 2026-09-24 addendum and retained taint limitations |
| `ct::field_mul` | Same arithmetic as FAST, no early-exit | Strong |
| `ct::field_inv` | Fixed iteration SafeGCD or exponentiation chain | Strong |
| `ct::point_add_complete` | Complete addition formula (handles all cases) | Strong |
| `ct::point_dbl` | No identity check branching | Strong |
| `field_select(a, b, flag)` | Bitwise masking: `(a & mask) \| (b & ~mask)` | Strong |
| ECDSA nonce (RFC 6979) | Deterministic, CT HMAC-DRBG | Strong |

### What Is NOT Constant-Time

| Operation | Why | Risk |
|-----------|-----|------|
| `fast::scalar_mul` | Window-NAF with variable-length representation | Timing leak on scalar bits |
| `fast::field_inverse` | Variable-time SafeGCD (divsteps exit early) | Leak on field element value |
| `fast::point_add` | Short-circuits on infinity | Leak on point identity |
| GPU kernels (all) | SIMT execution model, shared memory | Observable via GPU profiling |
| Some FROST / MuSig2 helpers | Session setup, public-index math, and broader protocol orchestration are not all elevated to full secret-path CT claims | Experimental surface; secret-bearing `musig2_partial_sign` and `frost_sign` do have protocol-level dudect coverage |

---

## CT Primitive: Constant-Time Select

The fundamental building block of CT operations:

```cpp
// src/cpu/include/secp256k1/field_branchless.hpp

inline FieldElement field_select(const FieldElement& a,
                                  const FieldElement& b,
                                  bool flag) noexcept {
    // Convert bool to all-1s or all-0s mask (branchless)
    std::uint64_t mask = -static_cast<std::uint64_t>(flag);

    const auto& a_limbs = a.limbs();
    const auto& b_limbs = b.limbs();

    return FieldElement::from_limbs({
        (a_limbs[0] & mask) | (b_limbs[0] & ~mask),
        (a_limbs[1] & mask) | (b_limbs[1] & ~mask),
        (a_limbs[2] & mask) | (b_limbs[2] & ~mask),
        (a_limbs[3] & mask) | (b_limbs[3] & ~mask)
    });
}
```

**Audit points**:
1. `bool -> uint64_t mask` must not be compiled to a branch
2. Both paths of `from_limbs` must execute (no short-circuit)
3. Compiler must not optimize away the unused path

---

## CT Scalar Multiplication Details

### `ct::scalar_mul(P, k)` -- Arbitrary Point

```
Algorithm: GLV + 5-bit signed encoding

1. Transform: s = (k + K) / 2  (K = group order bias)
2. GLV split: s -> v1, v2 (each ~129 bits)
3. Recode v1, v2 into 26 groups of 5-bit signed odd digits
   -> every digit is guaranteed non-zero and odd
4. Precompute table: 16 odd multiples of P and lambdaP
   T = [1P, 3P, 5P, ..., 31P, 1lambdaP, 3lambdaP, ..., 31lambdaP]
5. Fixed iteration: for i = 25 downto 0:
   a. 5 x point_double (CT)
   b. lookup T[|v1[i]|] with CT table scan (touch all entries)
   c. conditional negate based on sign bit (CT)
   d. unified_add (CT complete formula)
   e. repeat for v2[i]

Cost: 125 dbl + 52 unified_add + 52 signed_lookups(16)
All iterations execute regardless of scalar value.
```

### `ct::generator_mul(k)` -- Generator Point

```
Algorithm: Hamburg signed-digit comb

1. v = (k + 2^256 - 1) / 2 mod n
2. Every 4-bit window yields guaranteed odd digit
3. Precomputed table: fixed G multiples per window (generated at init)
4. COMB_SPACING outer iterations x COMB_BLOCKS inner iterations:
   a. CT table lookup -- scan all 32 entries (AVX2 vectorized on x86-64-v3)
   b. incomplete mixed Jacobian+affine add (7M+3S)
   c. point_double between outer iterations
5. Correction point added at end

Inner-loop add formula (2026-05-04): switched from complete unified
formula (12M+2S) to incomplete mixed Jacobian+affine (7M+3S, see
add_affine_fast_ct / add_affine_fast_ct_4x64 in ct_point.cpp).
SAFETY: all table entries are fixed precomputed G multiples; the
probability that the running accumulator R equals any table entry
is ~2^-128 for a random scalar, making the degenerate case of the
incomplete formula cryptographically negligible. The correction point
uses the same incomplete formula for the same reason.
This is identical reasoning to scalar_mul_prebuilt_fast.
Savings: ~5M per add x ~43 inner additions ≈ 215M ≈ 2800 ns.

Magnitude fix (2026-05-04): add_affine_fast_ct used incorrect negate()
magnitude parameters, causing uint64 underflow in the 5x52 lazy-reduction
scheme when called after point_dbl_n_core (X mag=18, Y mag=10) or after
a prior add (X3 mag=7). Fixed: negate(8)→negate(18) for X1, negate(4)→
negate(10) for Y1, negate(1)→negate(7) for X3. Root cause of schnorr_adaptor
and ECDSA self-verify failures. CT invariant unaffected (no branches added).

AVX2 comb_lookup (2026-05-04): the 32-entry CT table scan in comb_lookup
now uses AVX2 when SECP256K1_CT_AVX2 is defined (x86-64 + __AVX2__).
Each CTAffinePoint (80 bytes = x.n[5] + y.n[5]) is processed as 2.5 ymm
registers (r0: x.n[0..3], r1: x.n[4]+y.n[0..2], r2_128: y.n[3..4]).
The CT blend idiom is: r = (r ^ s) & mask ^ r (identical to table_lookup_core).
Exactly one mask fires per call (i == index), so OR-accumulation is equivalent
to selection. The Y-negate step remains scalar (runs once, not in the scan loop).
CT invariant preserved: no data-dependent branches; all 32 entries touched.
Scalar fallback path unchanged for non-AVX2 targets.
```

---

## Timing Verification: dudect Methodology

### Implementation

File: `audit/test_ct_sidechannel.cpp` (1300+ lines)

Uses the dudect approach (Reparaz, Balasch, Verbauwhede, 2017):

```
1. Two classes of inputs:
   - Class 0: Edge-case values (zero, one, identity, max)
   - Class 1: Random pre-generated values

2. For each function under test:
   a. Pre-generate N input pairs (class 0 and class 1)
   b. Random class assignment per measurement
   c. Array-based class selection (constant-cost lookup)
   d. rdtsc/cntvct timing of the operation
   e. Collect timing distributions

3. Statistical test: Welch's t-test
   - |t| < 4.5 -> no detectable timing difference (PASS)
   - |t| >= 4.5 -> timing leak detected (FAIL, 99.999% confidence)

4. Timing barriers: asm volatile prevents reordering
```

### Functions Tested

| Function | Class 0 (edge) | Class 1 (random) |
|----------|----------------|-------------------|
| `ct::field_mul` | Zero, One | Random field elements |
| `ct::field_inv` | One | Random field elements |
| `ct::field_square` | Zero | Random field elements |
| `ct::scalar_mul` | Small scalars | Random 256-bit scalars |
| `ct::generator_mul` | One, Two | Random 256-bit scalars |
| `ct::point_add` | Identity + P | Random points |
| `field_select` | flag=0, flag=1 | Random flags |
| ECDSA sign | Known keys | Random keys |
| Schnorr sign | Known keys | Random keys |
| `musig2_partial_sign` | Fixed secret key | Random secret keys |
| `frost_sign` | Low-Hamming-weight signing share | High-Hamming-weight signing share |
| `frost_lagrange_coefficient` | Signer set `{1,2}` | Signer set `{1,3}` (advisory, public indices) |

### Running the Test

```bash
# Direct execution (recommended)
./build/tests/test_ct_sidechannel

# Under Valgrind (checks memory access patterns)
valgrind ./build/tests/test_ct_sidechannel_vg

# Interpretation:
# |t| < 4.5 for all operations -> PASS
# Current result: timing variance ratio 1.035 (well below 1.2 concern threshold)
```

---

## Known Limitations

### 1. Formal Verification (Expanded, Still Not End-to-End)

The CT layer is verified using:
- **ct-verif LLVM pass** -- deterministic compile-time CT check of `ct_field.cpp`, `ct_scalar.cpp`, `ct_point.cpp`, and `ct_sign.cpp` (`.github/workflows/ct-verif.yml`). If the LLVM pass is unavailable, a fallback IR branch analysis runs.
- **Valgrind CT taint analysis** -- `ci/valgrind_ct_check.sh` marks private-key bytes as secret via `MAKE_MEM_UNDEFINED` / `--track-origins=yes` and runs signing + ECDH operations, failing on any secret-derived branch or memory access. Integrated in `.github/workflows/valgrind-ct.yml`.

The normalized evidence collector (`ci/collect_ct_evidence.py --strict`) now treats the expected ct-verif module set as owner-grade blocking: `ct_field`, `ct_scalar`, `ct_point`, and `ct_sign`. A deterministic artifact that omits one of those modules is downgraded from usable proof material to a configured-only gap.

Not yet integrated:
- **Vale** (F\* verified assembly)
- **Fiat-Crypto** (formally verified field arithmetic)
- **Cryptol/SAW** (symbolic analysis)

Additional CT guarantees come from:
- Manual code review
- Compiler discipline (`-O2` specifically)
- dudect empirical testing (x86-64 + ARM64 native)
- ASan/UBSan runtime checks

Protocol-level nuance:
- `musig2_partial_sign` and `frost_sign` are exercised by the dudect suite in `audit/test_ct_sidechannel.cpp`
- `frost_lagrange_coefficient` is tracked as advisory timing only because it operates on public participant indices
- This timing evidence improves confidence for secret-bearing signing steps, but does not by itself promote the whole MuSig2/FROST protocol stack out of experimental status

### 2. Compiler Risk

Compilers may break CT properties by:
- Converting bitwise cmov to branches for "optimization"
- Eliminating "dead" computation paths
- Auto-vectorizing with data-dependent masking
- Different behavior at `-O3` vs `-O2`

**Mitigation**: The project uses `asm volatile` barriers and recommends `-O2` for production CT builds. Higher optimization levels should be validated with dudect.

### 3. Microarchitecture Variability

CT properties verified on one CPU may not hold on another:
- Intel vs AMD vs ARM have different timing behaviors
- Variable-latency multipliers on some uarch
- Cache hierarchy differences

**Status**: Tested on x86-64 (Intel/AMD) and ARM64 (Apple M1 native). Multi-uarch dudect coverage:
- x86-64: CI runners (ubuntu-24.04) -- every push/PR
- ARM64: Apple Silicon M1 (macos-14) -- smoke per-PR, full nightly (`.github/workflows/ct-arm64.yml`)
- ARM64: cross-compiled via aarch64-linux-gnu-g++-13 (compile check only)

### 4. GPU CT Guarantees

The GPU CT layers (CUDA `secp256k1::cuda::ct::`, OpenCL `secp256k1_ct_sign.cl`/`secp256k1_ct_zk.cl`,
Metal `secp256k1_ct_sign.metal`/`secp256k1_ct_zk.metal`) provide **algorithmic** constant-time
guarantees: no secret-dependent branches, no secret-dependent memory access patterns,
fixed iteration counts. All three GPU backends implement identical CT algorithms.

**What GPU CT protects against:**
- Software-level timing attacks from co-located GPU workloads
- Branch divergence leaking scalar bits within a warp/wavefront/threadgroup
- Memory access pattern analysis via GPU profiling tools

**What GPU CT does NOT protect against:**
- Hardware-level electromagnetic or power analysis
- GPU shared memory bank conflict timing (microarchitectural)
- Driver-level scheduling observation
- Physical side-channels requiring oscilloscope-level measurements

The GPU CT layers are tested via:
- **CUDA**: `test_ct_smoke` (9 functional tests) + GPU audit runner (Section S6: CT Analysis)
- **OpenCL**: `opencl_audit_runner` ( 482 modules including CT signing + CT ZK sections)
- **Metal**: `metal_audit_runner` ( 482 modules including CT signing + CT ZK sections)

### 5. Experimental Protocols

FROST and MuSig2 remain broader experimental protocol surfaces, and the repo tracks their protocol evidence explicitly:
- Secret-bearing `musig2_partial_sign` and `frost_sign` have protocol-level dudect coverage in `audit/test_ct_sidechannel.cpp`
- `frost_lagrange_coefficient` is tracked as advisory timing-only because it operates on public participant indices
- Multi-party orchestration, session setup, and misuse-boundary analysis need more CAAS protocol-grade evidence before the full stacks can be promoted to strong CT claims

---

## CT Audit Checklist for Reviewers

- [ ] **field_select**: Verify `-static_cast<uint64_t>(flag)` produces all-1s/all-0s
- [ ] **field_select**: Confirm compiler emits no branch (inspect assembly)
- [ ] **ct::scalar_mul**: Fixed iteration count (26 groups x 5 doublings + 52 adds)
- [ ] **ct::scalar_mul**: Table lookup scans ALL entries (no early-exit)
- [ ] **ct::ecmult_const_xonly**: validate public `xn`/`xd` before the existing secret-`q` ladder; audit the combined field inverse and generated ladder separately. Decoder-only trusted calls require a valid lift and nonzero denominator; see the 2026-09-24 scope limits.
- [ ] **ct::generator_mul**: Fixed COMB_SPACING × COMB_BLOCKS iterations, no conditional skip
- [ ] **ct::generator_mul table add**: uses incomplete mixed-add (add_affine_fast_ct) — safe because all table entries are fixed G multiples (degenerate probability ~2^-128)
- [ ] **ct::point_add_complete**: Handles P+P, P+O, O+P, P+(-P) without branching
- [ ] **ct::field_inv**: Fixed exponentiation chain length (no variable-time SafeGCD)
- [ ] **ECDSA nonce**: RFC 6979 HMAC-DRBG is CT (no secret-dependent branches)
- [ ] **Schnorr nonce**: BIP-340 tagged hash is CT
- [ ] **No early return**: grep for `if (is_zero())` or `if (is_infinity())` in CT path
- [ ] **ct::scalar_inverse(0) guard**: both SafeGCD and Fermat CT inverse paths return `Scalar::zero()` for zero input (defense-in-depth, not a timing exit — the zero check is on the input, not on secret-derived data; verified by `test_exploit_boundary_sentinels` BS-1, BS-10)
- [ ] **No array indexing by secret**: all lookups use linear scan + cmov
- [ ] **asm volatile barriers**: present around timing-sensitive sections
- [ ] **dudect passes**: |t| < 4.5 for all tested functions

---

## CT Primitive Change Log (Secret-Path Surface)

### 2026-05-04 — New CT Primitive: `ecmult_const_xonly` (BIP-324 ECDH)

Added `ct::ecmult_const_xonly(xn, xd, q)` to `src/cpu/src/ct_point.cpp` and
`src/cpu/include/secp256k1/ct/point.hpp`. Port of libsecp256k1
`secp256k1_ecmult_const_xonly`. Computes the x-coordinate of `q * P` where P
has x-coordinate `xn/xd`, without any sqrt. Secret input is `q`; `xn`/`xd`
are public (from BIP-324 ELL64 encoding).

Implementation: builds `P_eff = (g·xn, g²)` on the isomorphic curve, then
calls `scalar_mul_jac(P_eff, q)` (same CT path as `scalar_mul`), and recovers
the x-coordinate via one combined field inversion (`R.z² * g * xd`), avoiding
a separate point normalization. The 4x64 fallback path uses sqrt since it lacks
FE52 Jacobian-output optimisation (slower, non-x86 only).

CT invariant: `scalar_mul_jac` has fixed iteration count and branchless table
lookups. The combined inversion is data-independent. No branches on `q`.

Also refactored `scalar_mul` to extract `scalar_mul_jac` as a static helper
returning raw `CTJacobianPoint` (Jacobian coords before Z-normalization).
`scalar_mul` now delegates to `scalar_mul_jac` + `to_point()`. Behaviour
and CT properties of `scalar_mul` are unchanged.

### 2026-05-04 — Performance Review: CT Primitive Internal Optimizations

Three internal optimizations to `ct_scalar.cpp` and `ct_field.cpp`. All changes
preserve the CT invariant — no secret-dependent branches or memory access patterns
were added. The changes are in the *implementation* of CT primitives, not their
*contract*.

**`ct_scalar.cpp:divsteps_59` — volatile removal (B-1)**

- **Before:** `volatile uint64_t c1`, `volatile uint64_t c2` inside the 59-iteration divstep loop.
- **After:** `uint64_t c1`, `uint64_t c2` (plain, compiler keeps in registers).
- **CT impact:** None. The CT property of `divsteps_59` is algorithmic: fixed 59 iterations, all-branchless bitmask conditionals. The `volatile` keyword forced memory round-trips (store+reload), wasting ~118 memory ops per `scalar_inverse` call, but did not *provide* the CT property. The equivalent `ct_field.cpp:ct_divsteps_59` never had `volatile` and passes the same dudect suite.
- **Verification:** `test_prf3_ct_scalar_inverse` (PRF-3) in the regression suite; existing `audit_ct` and `ct_sidechannel` CAAS modules.

**`ct_scalar.cpp:scalar_cswap` — XOR-swap (B-8)**

- **Before:** Two full `Scalar` temporaries (64 bytes) + two `scalar_select` calls.
- **After:** 4×uint64 XOR-swap with mask (identical to `ct_field.cpp:field_cswap` at line 761).
- **CT impact:** None. The XOR-swap with mask is a textbook CT swap. The `scalar_select` calls used the same `ct_select` primitive; the XOR-swap is semantically equivalent and avoids 64-byte copy temporaries.
- **Verification:** `test_prf2_scalar_cswap` (PRF-2); `audit_ct` module.

**`ct_field.cpp:add256/sub256` — `__builtin_addcll` for ADX emission (B-10)**

- **Before:** Portable carry loop using `add_carry_u64` helper (data-independent but no ADCX hint).
- **After:** `__builtin_addcll` / `__builtin_subcll` on Clang only (`#if defined(__clang__)`); portable carry loop on all other compilers including GCC.
- **CT impact:** None. The `__builtin_addcll` intrinsic is a carry-chain hint that instructs the compiler to emit `ADCX`/`ADOX` instructions — these have data-independent latency on all x86-64 targets. The CT contract is unchanged: no branches, no secret-dependent memory access.
- **2026-05-04 compiler compat fix:** The original guard `#if defined(__GNUC__) || defined(__clang__)` was incorrect — GCC-13 defines `__GNUC__` but lacks these Clang-extension builtins, causing a build failure on GCC-13. Guard narrowed to `#if defined(__clang__)`. GCC now falls through to the portable carry loop (identical CT correctness; no ADCX emission on GCC targets).
- **Verification:** `test_prf6_ct_field_carry` (PRF-6); existing `ct_verif` LLVM pass.

---

## Planned Improvements

- [ ] **Formal verification** with Fiat-Crypto for field arithmetic
- [x] **ct-verif** LLVM pass integration for CT verification (`.github/workflows/ct-verif.yml`)
- [x] **ct-verif: ct_point.cpp** -- point operations are included in `.github/workflows/ct-verif.yml`, and `ci/collect_ct_evidence.py --strict` treats missing point-module evidence as owner-grade blocking
- [x] **Multi-uarch dudect** -- x86-64 CI + ARM64 Apple M1 native (`.github/workflows/ct-arm64.yml`)
- [x] **dudect expansion** to cover FROST/MuSig2 -- `musig2_partial_sign`, `frost_sign`, `frost_lagrange_coefficient`
- [x] **Valgrind CT taint** in CI -- MAKE_MEM_UNDEFINED + --track-origins (`ci/valgrind_ct_check.sh`, `.github/workflows/valgrind-ct.yml`)
- [ ] **Hardware timing analysis** with oscilloscope-level measurements
- [ ] **Compiler output audit** for every release at `-O2` and `-O3`

---

## References

- [dudect: dude, is my code constant time?](https://eprint.iacr.org/2016/1123) -- Reparaz et al., 2017
- [Timing-safe code: A guide for the rest of us](https://www.chosenplaintext.ca/open-source/dudect/) -- Aumasson
- [ct-verif: A Tool for Constant-Time Verification](https://github.com/imdea-software/verifying-constant-time) -- IMDEA
- [Fiat-Crypto: Proofs of Correctness of ECC](https://github.com/mit-plv/fiat-crypto) -- MIT
- [bitcoin-core/secp256k1](https://github.com/bitcoin-core/secp256k1) -- Reference CT implementation

---

<!-- 2026-05-04: ct::generator_mul comb inner loop switched from complete unified
add (unified_add_core<false>, 12M+2S) to incomplete mixed Jacobian+affine add
(add_affine_fast_ct / add_affine_fast_ct_4x64, 7M+3S). All table entries are
fixed precomputed G multiples; degenerate probability ~2^-128. CT properties
(fixed iteration count, branchless table lookup via cmov) unchanged. -->

*UltrafastSecp256k1 v4.6.2 -- CT Verification*
