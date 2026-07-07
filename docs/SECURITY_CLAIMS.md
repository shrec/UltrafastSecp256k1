# Security Claims & API Contract

**UltrafastSecp256k1 v4.5.0** -- FAST / CT Dual-Layer Architecture (CPU + GPU)

### 2026-07-06 - `ufsecp_gpu_hash256_var` added: batch variable-length HASH256, no new secret-bearing surface

Added `GpuBackend::hash256_var` / C ABI `ufsecp_gpu_hash256_var` (native CUDA, OpenCL,
and Metal kernels) and the bridge-free libbitcoin-direct wrapper `hash256_var_batch`.
Row `i` is `inputs[i*stride .. i*stride+input_lens[i])` with no BIP-340 tag prefix and
no GPU-side transaction parsing — this is the primitive a future libbitcoin
`txid_batch`/`wtxid_batch` wrapper will compose with CPU-side BIP141 serialization.
**Security claim: PUBLIC-DATA / variable-time only.** Every input (transaction/merkle
preimage bytes) is public on-chain data; no private key, nonce, signing share, or ECDH
scalar is ever passed through this path, so no `ct::*` boundary applies (see the
CT-vs-VT boundary rule). `include/ufsecp/ufsecp_gpu.h` is touched only because it is
the shared header for both secret-bearing GPU ops (e.g. `ufsecp_gpu_ecdh_batch`) and
this public-data op — this change adds no new secret-bearing ABI boundary. Fail-closed
contract: `ctx==NULL`/null buffer with `n>0` -> `UFSECP_ERR_NULL_ARG`; `n==0` -> no-op
`UFSECP_OK`; `stride==0`/`>kMaxHash256VarStride` (4 MiB) or any `input_lens[i]==0`/
`>stride` -> `UFSECP_ERR_BAD_INPUT`; `out32` is never left holding a partial or stale
digest on any rejected call. Covered by `audit/test_regression_hash256_var_batch.cpp`,
`audit/test_regression_hash256_var_parity.cpp` (cross-backend byte-identical output),
and `audit/test_exploit_hash256_var_bounds.cpp` (hostile-input bounds).

### 2026-07-06 - `ufsecp_gpu.h` C ABI banner corrected for the six lbtc-batch ops (doc-only, no claim change)

The `include/ufsecp/ufsecp_gpu.h` section banner above `ufsecp_gpu_xonly_validate`,
`ufsecp_gpu_commitment_verify`, `ufsecp_gpu_tagged_hash`, `ufsecp_gpu_pubkey_validate`,
`ufsecp_gpu_tagged_hash_var`, and `ufsecp_gpu_hash256` carried a stale "CUDA
implemented; OpenCL/Metal return `UFSECP_ERR_GPU_UNSUPPORTED`" claim. It has been
rewritten to state CUDA, OpenCL, and Metal all dispatch native on-device kernels for
these six ops (no host-CPU fallback), matching the same correction already applied to
`src/gpu/include/gpu_backend.hpp` and `docs/BACKEND_ASSURANCE_MATRIX.md`. The stale
"OpenCL/Metal `Unsupported` -> CPU fallback" wording in the "GPU per-item batch ABI"
claim below (unchanged since 2026-06-08) is corrected in the same pass. This is
comment/doc-only: no C ABI signature, enum, macro, dispatch logic, or the PUBLIC-DATA
/ no-CT-boundary claim for these six ops changed. See `docs/AUDIT_CHANGELOG.md`
(2026-07-06, "doc-only follow-up: `ufsecp_gpu.h` C ABI banner corrected for the same
six ops") for full detail.

### 2026-06-22 - CPU batch-verify throughput: persistent pool + FE52 decompress (verify-path only)

Reworked the CPU multi-threaded batch-verify paths (`ecdsa_batch_verify_mt`,
`schnorr_batch_verify_mt`, `ufsecp_ecdsa_verify_opaque_rows_mt`) to run on a persistent
worker pool and to use a faster FE52/point-only pubkey decompress. These are **verify**
paths over PUBLIC data; the security contract is unchanged and no signing/secret-bearing ABI
was added or altered. The decompress keeps the same prefix / x-range / on-curve (QR) /
parity validation as the tested `ecdsa_pubkey_parse`, so off-curve and out-of-range public
keys are still rejected (fail-closed). The multi-threaded verdict is bit-identical to the
serial path (verification is over public data and variable-time by design).

### 2026-06-18 - Bech32 address encoder allocation reduction

`bech32_encode` now avoids heap-backed intermediate vectors for normal Bitcoin
witness address payloads by converting 8-bit witness data into a stack-backed
5-bit buffer and computing the checksum as a streaming polymod. This is a
public-data address formatting optimization only. It does not touch signing,
verification, secret-key parsing, nonce generation, CT arithmetic, or ABI output
clearing.

**Claim:** P2WPKH/P2WSH/P2TR Bech32/Bech32m address outputs remain byte-identical
while normal witness address encoding performs less transient allocation work.
Secret lifecycle and constant-time signing guarantees are unchanged because the
encoder operates only on public HRP/version/witness-program bytes.

Validation: `run_selftest smoke`, `ci/check_source_graph_quality.py`, and local
`bench_hotpaths` runs showing `address_p2wpkh` improvement from the captured
baseline `522.11 ns` to `471.46 ns` / `470.74 ns`.

### 2026-06-13 - Opaque ECDSA verify compatibility for libsecp-style consumers

The public CPU C ABI now supports copied libsecp-compatible
`secp256k1_ecdsa_signature` scalar storage as an explicit ECDSA input format.
Compact `ufsecp_ecdsa_verify` remains strict compact `r||s`; the new opaque
verify APIs parse the opaque scalar limbs, reject zero/out-of-range scalars,
low-S normalize internally, and then verify. Batch/row variants return per-row
verdict bytes, so malformed public rows fail closed without aborting unrelated
rows.

The GPU C ABI adds `ufsecp_gpu_ecdsa_verify_opaque_rows` with
`ufsecp_gpu_ecdsa_verify_lbtc_rows` as a compatibility alias. CUDA, OpenCL, and
Metal parse the same strided public rows on device and apply low-S normalization
before ECDSA verify. These are PUBLIC-DATA verification paths: they do not carry
private keys, nonces, or secret scalars.

**Claim:** libbitcoin-style consumers can keep their existing opaque
`ec_signature` storage and still receive libsecp-compatible
normalize-then-verify behavior without bridge-side compact row repacking. Caller
owned signatures and rows are not mutated.

Validation: `audit/test_ffi_round_trip.cpp`,
`audit/test_c_abi_negative.cpp`, `audit/test_gpu_abi_gate.cpp`, and
`compat/libbitcoin_bridge/tests/test_lbtc_bridge.cpp`.

### 2026-06-22 - MuSig2 partial-verify accepts both pubkey Y-parities (by design)

`musig2_partial_verify` (`src/musig2.cpp`, exposed as `ufsecp_musig2_partial_verify`)
verifies a partial signature against BOTH Y-parities of the signer's public key
(`sG == R_eff + ea*P_i` OR `sG == R_eff + ea*(-P_i)`). This is intentional and
required for self-consistency: this library's `musig2_partial_sign` takes a RAW
32-byte secret key (its original parity), not an x-only key, so the signer's `P_i`
may have either Y parity (BIP-327 signing uses `d_i` directly, with no per-signer
parity flip). Tightening verify to a single parity would reject valid partial
signatures produced by this library's own signer for odd-Y keys.

**Scope / impact:** partial-verify is a coordinator diagnostic to locate a
misbehaving signer before aggregation — it is NOT the consensus verifier. The
both-parity acceptance is a bounded verify-laxity (a partial sig valid under the
opposite signer-key parity is also accepted); it does **not** enable
aggregate-signature forgery — the aggregate is still checked by the standard
x-only Schnorr verifier, which fixes the parity. The divergence from strict
BIP-327 `PartialSigVerify` (which consumes the x-only individual key) is therefore
intentional and safe.

**Claim:** the both-parity acceptance is a deliberate consequence of the raw-seckey
partial-sign API and does not weaken aggregate unforgeability. kb
`MUSIG2-PVERIFY-PARITY`. Validation: MuSig2 sign->partial_verify roundtrip tests
(`audit/test_fuzz_musig2_frost.cpp::test_musig2_partial_verify_random`).

### 2026-06-11 - MuSig2 partial-sign secret-erasure hardening

`musig2_partial_sign` now scrubs every secret-derived stack local: `neg_k` (-k),
`neg_d` (-d), and `ead` (ea*d). Hardening only -- no timing behavior change
(branchless `ct::`); exploitation needs a separate stack-disclosure primitive.
Discovered by the improved `dev_bug_scanner.py` secret-derived-unerased check
(scans frost/musig + `Scalar const`). Regression test
`regression_secret_scalar_residue_erase`.

**Claim:** MuSig2 partial signing leaves no secret-derived scalar residue
(`neg_k`/`neg_d`/`ead`, plus `k`/`d`/`sec_nonce`) on the stack after return.

### 2026-06-10 - FROST/keypair secret-erasure hardening (FROST-SIGN-RESIDUE)

`frost_sign` and both `schnorr_keypair_create` variants now scrub every secret-derived
stack local: `rho_ei` (`my_binding·ei`) and `lambda_s_e` (`lambda_i·s_i·e`) in `frost_sign`
— which carry the secret binding nonce `ei` and signing share `s_i` — and the `d_prime`
private-key copy in `schnorr_keypair_create` (`src/cpu/src/ct_sign.cpp`,
`src/cpu/src/schnorr.cpp`). This is a secret-erasure (stack-scrubbing) hardening of the same
class as `T08-SCALAR-ERASE`; there is no timing/CT behavior change (all arithmetic is
branchless `ct::`), and exploitation would require a separate stack-disclosure primitive.
Regression test `regression_secret_scalar_residue_erase` (source-scan of all three sites +
Schnorr `keypair_create`+`sign`+`verify` round-trip).

**Claim:** FROST partial signing (`frost_sign`) and Schnorr keypair creation
(`schnorr_keypair_create`) leave no secret-derived scalar residue on the stack after return;
the secret binding nonce, signing share, and private-key copies are all `secure_erase`d.

### 2026-06-08 - MuSig2 infinity aggregate-nonce BIP-327 conformance (R=G)

`musig2_start_sign_session` now handles an infinity aggregate nonce per BIP-327
§GetSessionValues: a 33-zero ("ext") aggregate-nonce half is the point at infinity and is
accepted, and when the effective nonce `R = R1 + b·R2` is infinity the session nonce is set
to the generator `G`. Previously the engine rejected infinity halves (a non-conformant
divergence based on a BIP-327 misreading), so the official `sign_verify_vectors.json`
"both halves at infinity" valid case could not be processed. The shim
(`secp256k1_musig_nonce_process`) and native ABI (`ufsecp_musig2_start_sign_session`) now
accept a 33-zero half while still rejecting a non-zero half that fails to decompress;
individual pubnonces are still rejected at `pubnonce_parse` (BIP-327 `cpoint`). All affected
values are PUBLIC; no secret handling changes. Verified against `reference.py` and
libsecp256k1.

**Claim:** MuSig2 aggregate-nonce processing (including the infinity / `R=G` case) is now
interoperable with BIP-327 / libsecp256k1. The `SHIM-MUSIG-INF` divergence is removed.

### 2026-06-08 - MuSig2 BIP-327 binding-factor tag (interoperability)

Fixed a P1 interoperability bug: the MuSig2 nonce binding factor `b` in
`musig2_start_sign_session` was derived with the tag `"MuSig/nonceblinding"` instead of
BIP-327's `"MuSig/noncecoef"` (the hashed input was otherwise identical). This was
self-consistent for a pure-engine signing session but incompatible with any BIP-327
implementation (e.g. libsecp256k1): the engine rejected every externally-produced partial
signature, and a mixed-implementation session could never complete. `b` is PUBLIC, so this
is a conformance/interop fix, not a secret-handling change, and no previously-valid
pure-engine aggregate signature is affected. Confirmed against the bip-0327 `reference.py`
oracle and upstream libsecp256k1; regression test
`compat/libsecp256k1_shim/tests/test_bip327_sign_verify_vectors.cpp` (CTest
`bip327_sign_verify_vectors`) verifies external BIP-327 partial signatures for even-Y and
odd-Y aggregates.

**Claim:** MuSig2 partial signatures and aggregate nonces are now interoperable with
BIP-327 / libsecp256k1. (The infinity aggregate-nonce path remains a documented
fail-closed divergence — SHIM-MUSIG-INF in `docs/SHIM_KNOWN_DIVERGENCES.md`.)

### 2026-06-08 - MuSig2 BIP-327 tweak correctness (gacc/tacc)

Fixed a P1 correctness bug: tweaked MuSig2 signing. The keyagg cache lacked the BIP-327
`gacc`/`tacc` accumulators, so `secp256k1_musig_pubkey_{ec,xonly}_tweak_add` baked the
tweak into the aggregate but signing never applied the additive/sign correction — the
EC-tweaked aggregate key was wrong for odd-Y aggregates and the aggregated signature
failed to verify against any tweaked key. The fix threads `gacc`/`tacc` through the
tweak functions, `musig2_partial_sign` (`d = g·gacc·d'`), `musig2_partial_sig_agg`
(`s += e·g·tacc`), `musig2_start_sign_session`, and `musig2_partial_verify`.
`gacc`/`tacc` are PUBLIC; the untweaked path is byte-identical (defaults `gacc=1`,
`tacc=0`). Verified against the bip-0327 `reference.py` oracle (tweaked-output KATs) and
sign+verify+partial-verify roundtrips (regression test
`compat/libsecp256k1_shim/tests/test_bip327_tweak_sign.cpp`, CTest `bip327_tweak_sign`).
Taproot+MuSig2 key/script-path tweaked signing is now correct.

### 2026-06-06 - CPU ABI/shim fail-closed outputs and secret lifecycle refresh

`src/cpu/src/bip32.cpp` and `src/cpu/src/bip39.cpp` received a secret-lifecycle
cleanup pass. BIP32 public-key derivation now erases its parsed private scalar
after constant-time generator multiplication, and BIP32 master-key derivation
erases HMAC output halves, the parsed master scalar, and the temporary serialized
private-key bytes after copying the final key material into the caller-owned
`ExtendedKey`. BIP39 PBKDF2 now erases the per-block `salt_block` as soon as the
block output is copied; the existing `result` and `u` erasure remains in place.

**Claim:** these changes are lifecycle hardening only. They do not change BIP32
or BIP39 derived outputs, and they do not introduce variable-time secret
branches. Invalid strict-parse paths erase local secret-derived buffers before
returning failure.

The CPU ABI and shim boundary now treats malformed or internally invalid
secret-bearing results as fail-closed. Taproot, BIP144, coin helpers, FROST
sign/aggregate, Ethereum recoverable signing, and libsecp256k1 shim parser/sign
entry points zero their output buffers before processing or before returning
non-OK once they enter parser/signing dispatch. Unsupported custom nonce
callbacks in the shim are rejected before signing dispatch and leave the caller's
output signature untouched; this is a pre-dispatch argument/configuration reject,
not an internally invalid signing result. Zero partial signatures, all-zero
aggregate signatures, invalid recoverable signatures, and bad serialized keys are
returned as errors rather than success-with-zero-output.

**Contract:** callers may rely on a non-OK CPU ABI/shim return leaving the
documented output buffer in a zero/invalid default state for these hardened
surfaces. Caller-owned secret inputs remain caller-owned; when an ABI consumes a
nonce by contract, that nonce is still erased on every non-NULL exit path.

Validation: `unified_audit_runner --section protocol_security`,
`--section standard_vectors`, `--section memory_safety`, `--section
ct_analysis`, `--section fuzzing`, `--section exploit_poc`, shim CTest targets,
and `ci/check_exploit_wiring.py`.

### 2026-06-06 - GPU C ABI fail-closed outputs and secret-buffer erasure parity

The stable GPU C ABI now has an explicit fail-closed output contract for
result-bearing calls in `include/ufsecp/ufsecp_gpu.h` and
`src/cpu/src/ufsecp_gpu_impl.cpp`: outputs are cleared before backend dispatch
and are cleared again if the backend returns non-OK. This covers batch
verification, ECDH, Hash160, MSM, ecrecover, FROST partial verification, ZK
verification, Bulletproof verification, BIP-324 AEAD, SNARK witness helpers,
and BIP-352 scan. The collect APIs are excluded because their `key_buffer` is
in-place caller-owned marker state used for fallback.

**Claim:** the secret-bearing GPU operations are ECDH, BIP-352 scan, and
BIP-324 AEAD encrypt/decrypt. BIP-352 scan keys are now strict-parsed on the CPU
ABI path and in CUDA/OpenCL/Metal backends; zero and order-or-larger scan keys
fail with a bad-key error and leave outputs zeroed. OpenCL erases both the host
GLV wNAF scan plan and device plan buffer on every return path. CUDA validates
decompressed spend/tweak points before kernel use and zeroes scan/device buffers
through erase-on-exit guards. Metal erases BIP-352 scan scalar buffers, ECDH
scratch scalar buffers, and BIP-324 AEAD shared key buffers before release.

**Contract:** GPU callers must still treat any non-OK return as failure and
discard outputs, but the ABI additionally zeroes result buffers so hostile or
buggy callers do not observe partial successful outputs after backend errors.
Backend parity remains required across CUDA, OpenCL, and Metal for every
`GpuBackend` operation.

Validation: `audit/test_gpu_bip352_scan.cpp`,
`audit/test_gpu_host_api_negative.cpp`,
`audit/test_exploit_gpu_bip352_key_erase.cpp`, `unified_audit_runner --section
memory_safety`, and `unified_audit_runner --section exploit_poc`.

### 2026-06-04 - P2-CT-001: bip32_derive_path intermediate-key erasure (stack residue)

`src/cpu/src/bip32.cpp` `bip32_derive_path` now `secure_erase`s the redundant `child`
copy of each intermediate extended private key (key + chain code) between derivation
steps, and the working `current` on the failure path. Closes a stack secret-residue
gap: the path-walk loop previously left intermediate child private keys un-erased
across iterations. Pure hardening — the derived key is unchanged (`test_bip32.cpp`
asserts derive_path equals a manual `derive_child` chain byte-for-byte and is
deterministic). No API or semantic change.

**2026-06-04 follow-up:** the on-failure intermediate scrub is now *unconditional* (the
public x-coordinate erase is a harmless write when the material is not secret), making
the on-failure path reachable and covered by an xpub-hardened-derivation regression test
in `test_bip32.cpp`.

### 2026-06-01 — SHIM-001: variable-length Schnorr `sign_custom` restored (CT, matches upstream)

`secp256k1_schnorrsig_sign_custom` previously rejected `msglen != 32` (`return 0`), diverging
from upstream libsecp256k1 (whose `sign_custom` signs any length). The rejection was a left-over
from a period when the shim's *verify* was 32-byte-only; verify is now variable-length, so the
asymmetry is gone and varlen signing is restored.

**Claim:** variable-length BIP-340 signing through the shim is **constant-time in the signing key
and nonce**, with the same guarantees as the 32-byte path. The new
`secp256k1::ct::schnorr_sign(kp, msg, msglen, aux)` overload (`src/cpu/src/ct_sign.cpp`) is a
line-for-line mirror of the audited fixed-32 CT path: DPA-blinded `generator_mul_blinded` nonce
point, branchless `scalar_cneg` parity negate, branchless `ct::scalar_add/scalar_mul` for
`s = k + e·d`, and `secure_erase` of every secret-derived buffer (including the full
`64 + msglen` nonce/challenge hash inputs) on every return path. For `msglen == 32` the output is
byte-identical to the fixed-32 overload.

**Contract:** the shim forwards `msglen == 32` to `sign32` (unchanged fast path) and `msglen != 32`
to the new overload; `extraparams->ndata` is the 32-byte aux entropy (NULL → 32 zero bytes,
deterministic). Custom nonce functions remain rejected (fail-closed, unchanged). A signature
produced for a given message length verifies through `secp256k1_schnorrsig_verify` for the same
length (sign/verify symmetric). Regression: `audit/test_regression_schnorr_varlen_ct_fixes.cpp`
VCS-1..7 (round-trip + tamper-rejection across 1/31/33/64/100/256/300-byte messages).

### 2026-05-28 — defense-in-depth: ecdsa_sign_hedged now erases r (nonce-derived)

`ecdsa_sign_hedged` in `src/cpu/src/ecdsa.cpp` now erases `r = kG.x mod n` via
`secure_erase` after the inner signing block. No security claim change — `r` is not
a private key or nonce scalar. This is defense-in-depth stack hygiene.

### 2026-05-28 — batch parallel signing: no new secret exposure

`ufsecp_ecdsa_sign_batch` / `ufsecp_schnorr_sign_batch`: parallel slot dispatch
via `std::thread` does not weaken CT guarantees. Each slot calls the same CT
signing primitive as the serial path; private keys are accessed read-only;
output is zeroed on any error (fail-closed). No new attack surface vs serial.

### 2026-05-28 — Security fix bundle: SEC-001..005 (ct_sign/FROST/MuSig2), SHIM-NULL-CB, TEST-003, CAAS-007/008

**SEC-001 (P1):** All 6 `r.is_zero()` calls in `ct_sign.cpp` converted to `r.is_zero_ct()`
(SIZ-1..7 — includes recoverable + hedged-recoverable sign variants).

**SEC-002 (P1):** `frost_keygen_finalize` rejects group public key that accumulates to infinity
(adversarial commitments that cancel; guard: `group_key.is_infinity()` before share verification).

**SEC-003 (P1):** `musig2_partial_sign` fail-closed on degenerate session where `e == 0`
(returns `Scalar::zero()` and erases `k1`/`k2` before returning).

**SEC-004 (P2):** `frost_sign_nonce_gen` takes `nonce_seed` by value and securely erases it
(was const-ref — caller's copy could be inspected after call).

**SEC-005 (P2):** `musig2_partial_sign` signer-index validation uses `ct::generator_mul_blinded`
(was `ct::generator_mul` — DPA-vulnerable under power analysis).

**SHIM-NULL-CB (P2):** `secp256k1_xonly_pubkey_tweak_add`, `secp256k1_xonly_pubkey_tweak_add_check`,
`secp256k1_keypair_xonly_tweak_add`, and `secp256k1_ecdsa_recoverable_signature_convert` now fire
`secp256k1_shim_call_illegal_cb` on NULL non-ctx args (was silent return 0).

### 2026-05-27 — Security fix bundle: SEC-001/004/005/006, CT-007/008, COMPAT-001/004/006/010, VER-006

**SEC-006 (P1):** `compute_challenge` in `frost.cpp` now erases `e_hash` (32 B,
nonce-adjacent) and `challenge_data` (96 B, nonce + pubkey + msg) before return.

**SEC-005 (P2):** All three `ecdh_compute*` functions reject off-curve public keys
before `ct::scalar_mul`. Guard: `is_infinity()` + `y²=x³+7` field check.

**CT-008 / SEC-004 (P1):** `is_zero_ct()` now used on `sig.r` in
`ct::ecdsa_sign_verified`, `ct::ecdsa_sign_hedged_verified`, and
`secp256k1::ecdsa_sign_hedged_verified` (was `is_zero()` — VT branch on nonce-derived value).

**SEC-001 / COMPAT-010 (P1):** BCHN shim `secp256k1_schnorr_sign`: `s.is_zero_ct()` (was VT
`is_zero()`); `secp256k1_schnorr_verify`: `parse_bytes_strict` rejects `s >= n`.

**CT-007 (P2):** CUDA `ecdsa_sign_recoverable` overflow compare loop replaced with branchless
mask-accumulation (no early `break` on nonce-derived `rx_bytes`).

**COMPAT-001 (P2):** `secp256k1_context_create` and `secp256k1_context_preallocated_create` now
reject unknown flag bits above `BIT_CONTEXT_SIGN` (bits 10+).

**COMPAT-004 (P2):** `secp256k1_schnorrsig_sign_custom` returns 0 for `msglen != 32` without
firing the illegal callback (firing was wrong — not a programming error per upstream).

**COMPAT-006 (P2):** `secp256k1_xonly_pubkey_from_pubkey` NULL `xonly_pubkey` / `pubkey` args
now fire the illegal callback (was silent return 0).

**VER-006 (P2):** CocoaPods podspec no longer defines `SECP256K1_FAST_NO_SECURITY_CHECKS=1`
or `SECP256K1_ULTRA_SPEED=1` — shipping with security checks disabled was a distribution error.

### 2026-05-26 — SHIM-NONCEGEN-001 fixed: secp256k1_musig_nonce_gen now mixes extra_input32

**Claim update:** `secp256k1_musig_nonce_gen(extra_input32 ≠ NULL)` now produces
nonces distinct from `extra_input32 = NULL` and from any other non-NULL
`extra_input32` value. The BIP-327 defense-in-depth property is now satisfied:
callers supplying session-specific entropy in `extra_input32` receive uniquely
derived nonces. Verified by `audit/test_regression_musig_noncegen_extra_input.cpp`
NCI-2 and NCI-3 (bug-fixed mode assertions).

### 2026-05-24 — v9 RT-002 / TASK-002: Adaptor signing — DPA-blinded generator_mul on every secret

`src/cpu/src/adaptor.cpp` now routes every secret-scalar generator
multiplication through `ct::generator_mul_blinded(...)`:
- `schnorr_adaptor_sign`: long-term key `P = sk*G` (was unblinded).
- `ecdsa_adaptor_sign`: secret nonce `base_nonce = k*G` (was unblinded).
The `binding` scalar in `ecdsa_adaptor_sign` is derived from the PUBLIC
`adaptor_point` and stays on the unblinded primitive (correct + cheaper).

Constraint enforced: `GENERATOR-MUL-CT` (knowledge_base). The same DPA
blinding discipline now active on `ecdsa_sign`, `schnorr_sign`, and the
MuSig2 sign paths is applied to adaptor signing.

Regression coverage:
`audit/test_regression_adaptor_blinded_nonce.cpp` adds
`test_adaptor_blinded_all_secret_sites_source_scan` (counts
`generator_mul_blinded(k)` ≥ 2 across both adaptor variants; asserts no
bare `ct::generator_mul(k)` or `ct::generator_mul(private_key)` remains).

### 2026-05-24 — v9 RT-006/-007/-014/-015 / TASK-022: secret stack residue bundle

Four small fixes hardening stack-residue lifecycle on CPU signing paths:

- **RT-006 (`src/cpu/src/schnorr.cpp:502,511`)** —
  `schnorr_sign(const Scalar&)` and `schnorr_sign_verified(const Scalar&)`
  raw-key convenience overloads now `secure_erase(&kp.d, sizeof(kp.d))`
  after the sub-call returns. Previously the locally-constructed
  `SchnorrKeypair`'s negated signing scalar lingered in the stack frame.
- **RT-007 (`src/cpu/src/bip32.cpp:414`)** — `derive_child` uses
  `child_scalar.is_zero_ct()` (was `is_zero()`). Removes data-dependent
  branch on the secret-derived child scalar.
- **RT-014 (`src/cpu/src/frost.cpp:60-106`)** — `derive_scalar` and
  `derive_scalar_pair` now `secure_erase` the local SHA256 state `h`,
  the per-tag `tag_hash`, and the finalized `hash` array before
  returning. All three incorporate the seed (secret material).
- **RT-015 (`src/cpu/src/adaptor.cpp:311`)** — `ecdsa_adaptor_sign`
  degenerate-r early-return path now erases `k`, `binding`, and
  `R_x_bytes` before returning the zero sentinel. The success-path
  erase block is unreachable from this branch.

Regression coverage:
`audit/test_regression_secret_stack_residue_v9.cpp` (wired in
`unified_audit_runner` as `regression_secret_stack_residue_v9`,
advisory=false, `differential` section) source-scans for the required
`secure_erase` / `is_zero_ct` calls AND round-trips the schnorr raw-key
overloads to confirm no functional regression. 16/16 checks pass.

### 2026-05-31 — MED-3: C++-layer defense-in-depth Rule-13 check now fail-closed (final closure)

- **MED-3 status:** the last piece. The 2026-05-24 closure (below) hard-failed the v1 ABI and
  routed all callers through v2, but explicitly left the C++ `musig2_partial_sign` Rule-13 block
  gated on a non-empty `individual_pubkeys` (skipped when empty). `musig2_partial_sign`
  (`src/cpu/src/musig2.cpp`) now treats the cross-check as **mandatory**: it fail-closes
  (returns `Scalar::zero()`) when `individual_pubkeys` cannot validate `signer_index`, so a C++
  caller that supplies an unvalidatable context (or manually clears the field) can no longer sign
  blind. `signer_index` and the container size are public, so the guard adds no secret-dependent
  branch.
- The v2 ABI is unaffected (it populates `individual_pubkeys` before signing) and production never
  reached the gap. Regression: `audit/test_regression_musig2_signer_index_validation.cpp` MSI-4
  now asserts the fail-close (`advisory=false`); `RESIDUAL_RISK_REGISTER.md` RR-010 → CLOSED.

### 2026-05-24 — v9 RT-001 / TASK-001: MuSig2 v1 partial_sign DISABLED (full closure)

- **MED-3 / P1-SEC-002 status:** CLOSED. Prior revisions left
  `ufsecp_musig2_partial_sign` (v1) functional for non-migrated callers
  and relied on the `[[deprecated]]` compile warning to nudge migration.
  Adversarial review v9 RT-001 confirmed this kept the C++-layer Rule-13
  check (`musig2_partial_sign` at `src/cpu/src/musig2.cpp`) silently
  disarmed because `parse_musig2_keyagg` does not populate
  `MuSig2KeyAggCtx::individual_pubkeys`, so the entire validation block
  was skipped. v1 therefore produced valid partial signatures for
  arbitrary `(privkey, signer_index)` pairs.
- **Closure (v9 TASK-001):** `ufsecp_musig2_partial_sign`
  (`src/cpu/src/impl/ufsecp_musig2.cpp:203`) now hard-fails on every call
  with the new error code `UFSECP_ERR_DEPRECATED_API` (= 12). The output
  buffer is zeroed and the `secnonce` is securely erased on the reject
  path so callers that ignore the return code cannot reuse the nonce.
  `ufsecp_musig2_partial_sign_v2` is the only supported entry; it carries
  the `pubkeys[]` array, performs constant-time
  privkey↔pubkeys[signer_index] comparison, and populates
  `kagg_check.individual_pubkeys` before invoking
  `musig2_partial_sign_core` so the C++-layer Rule-13 check engages too.
- **Regression coverage:**
  `audit/test_regression_musig2_v1_partial_sign_deprecated.cpp`
  (wired in `unified_audit_runner` as
  `regression_musig2_v1_partial_sign_deprecated`, advisory=false) asserts:
  (1) v1 returns `UFSECP_ERR_DEPRECATED_API` on valid inputs,
  (2) output buffer all-zero on the reject path,
  (3) `secnonce` securely erased on the reject path,
  (4) NULL ctx still preempts with `UFSECP_ERR_NULL_ARG`,
  (5) v2 still produces a valid partial signature.
- **Breaking change:** any third-party caller still on v1 starts
  receiving `UFSECP_ERR_DEPRECATED_API` after this change. This is
  intentional. Migration to v2 takes one extra argument (the `pubkeys`
  array) and is documented in `include/ufsecp/ufsecp.h:817`.

### 2026-05-23 — P1-SEC-01 / MED-3 partial mitigation: v2 ABI as the secure path (SUPERSEDED 2026-05-24)

- **P1-SEC-01 / MED-3 status:** PARTIAL CLOSURE. The realistic attack
  surface is the ABI boundary (an external coordinator passing a wrong
  signer_index). `ufsecp_musig2_partial_sign_v2`
  (`src/cpu/src/impl/ufsecp_musig2.cpp`) closes this by cross-validating
  privkey ↔ pubkeys[signer_index] in CT before invoking the signing
  core. v2 has been the recommended path since SEC-001.
- **v2 refactor (2026-05-23):** v2 no longer delegates to v1. After the
  ABI-level pubkey check passes, v2 populates `kagg_check.individual_pubkeys`
  from its caller-supplied `pubkeys` array and invokes a new internal
  helper `musig2_partial_sign_core` that runs the secnonce parse, session
  parse, and the C++ `musig2_partial_sign` call. This keeps the v1 entry
  point free of v2's pubkey-array logic and allows the C++ Rule-13 check
  to engage as a defense-in-depth (it now sees populated
  `individual_pubkeys` on the v2 path).
- **v1 ABI (`ufsecp_musig2_partial_sign`)** remains functional with the
  long-standing `[[deprecated]]` attribute. A malicious wrong-index
  caller using v1 is NOT protected at the C++ layer (the keyagg blob
  does not carry pubkeys, so `individual_pubkeys` is empty and the
  Rule-13 check is silently skipped). Callers MUST migrate to v2 for
  protection. The internal v1 body now factors into `musig2_partial_sign_core`
  for code-share with v2 — semantics are unchanged from the prior v1.
- **MSI-4 regression test remains advisory-skipped** (rc=77) until the
  C++-layer bypass is closed (which would require either making
  `individual_pubkeys` non-empty mandatory on every C++ call site or
  adding a separate enforce-flag). The `ALL_MODULES` entry stays
  `advisory=true`; `RESIDUAL_RISK_REGISTER.md` should continue to track
  MED-3 as an open partial-closure item until the C++-layer fix lands.

### 2026-05-22 — TASK-007: ufsecp_musig2_partial_sign v1 ABI deprecation

- **P1-SEC-002 / MED-3 / RED-002** (`include/ufsecp/ufsecp.h`,
  `src/cpu/src/impl/ufsecp_musig2.cpp`): the v1 partial-sign export now
  carries `UFSECP_DEPRECATED("...")` again — external callers receive a
  compile-time warning directing them to
  `ufsecp_musig2_partial_sign_v2()`, which validates signer_index against
  privkey via a constant-time pubkey comparison at the ABI boundary. v1
  remains functional for backwards compatibility (forcing immediate
  breakage has too much blast radius for a stabilisation push); the
  compile-time warning makes the security gap impossible to miss for any
  caller that recompiles. The v2 export and the seven internal audit
  tests that intentionally exercise v1 to verify ABI contract coverage
  retain access via `#pragma diagnostic` suppression / per-source
  `-Wno-deprecated-declarations` so the rest of the codebase stays
  buildable under -Werror.
- **No API change** for callers already on v2. Callers still on v1 are
  encouraged to migrate; the wrong-signer-index forgery vector (a
  malicious coordinator who places the victim's pubkey at the wrong
  position in the keyagg blob) remains exploitable against v1 callers
  until they migrate. Tracking IDs P1-SEC-002 / MED-3 / RED-002.

### 2026-05-22 address.cpp — silent-payments t_k generator-mul is now CT (merged from main e4e17305)

- **P-SP-CT-001** (`silent_payment_create` and `silent_payment_scan` in
  `src/cpu/src/address.cpp`): The output-point computation `P = B_spend + t_k · G`
  now uses `ct::generator_mul(t_k)` in both `silent_payment_create` (output key
  derivation) and the per-output check inside `silent_payment_scan`. Previously
  used the variable-time GLV path via `Point::generator().scalar_mul(t_k)`.
  `t_k = SHA-256("BIP0352/SharedSecret" || S || ser32(k))` is derived from the
  shared secret `S = b_scan · A_sum`, so it depends on the scan private key;
  variable-time generator multiplication on `t_k` therefore leaks timing
  information about the scan key. The CT path closes that side channel and
  also adopts the precomputed comb table (~25× speedup on cold path).
- **No API change**: Same `SilentPaymentOutput` / scan-result shape; same
  produced output bytes for honest inputs (verified by the
  `test_v4_features` silent-payment encode/decode/scan pipeline and the
  in-tree LtcSpScanner parity test).
- **Originated**: main commit `e4e17305 perf(silent-payments): ct::generator_mul
  replaces generator().scalar_mul — 3× create, 2× scan`. Merged into dev
  on 2026-05-22.

### 2026-05-21 ecdsa.cpp / musig2.cpp / frost.cpp — P2-CT-001/002/003/007 nonce candidate scalar erase

- **P2-CT-001** (`rfc6979_nonce`): Both `cand1` and `cand2` nonce candidate scalars are
  now `secure_erase`d immediately after `ct::scalar_select` picks the selected candidate.
  Previously both remained as stack residue until the function frame was reclaimed.
- **P2-CT-002** (`rfc6979_nonce_hedged`): Identical fix in the hedged variant.
- **P2-CT-003** (`musig2_nonce_gen`): `cand1`/`cand2` erased in both the k1 and k2
  scoped blocks after their respective `ct::scalar_select` calls.
- **P2-CT-007** (`derive_scalar_from_hash` in frost.cpp): `cand1`/`cand2` derived from
  secret polynomial coefficient hashes are now erased after `ct::scalar_select`.
- **Security impact**: Eliminates nonce-derived secret material from stack residue in all
  four nonce generation paths. CT selection property is unchanged — `ct::scalar_select`
  still performs a branchless pick. Only the post-select cleanup is improved.
- **No signing output change**: `Scalar const result` is returned by value copy; the
  erased candidates do not affect the returned nonce.

### 2026-05-21 shim_*.cpp — SHIM-A01..A08 NULL-arg illegal_callback parity

- **SHIM-A01** (`secp256k1_ecdsa_signature_normalize`): `sigin=NULL` now fires the illegal
  callback matching libsecp256k1 behavior. Previously silently returned 0.
- **SHIM-A02** (`secp256k1_ec_pubkey_sort`): `ctx=NULL` now fires the illegal callback
  before any pubkey processing. The existing `pubkeys=NULL` and `pubkeys[i]=NULL` guards
  already fired the callback; the ctx guard was missing.
- **SHIM-A03** (`secp256k1_tagged_sha256`): `msg=NULL` with `msglen=0` is now valid
  (zero-length message — matches libsecp256k1 semantics). `msg=NULL` with `msglen>0`
  fires the illegal callback.
- **SHIM-A07/A08** (`secp256k1_ec_pubkey_negate`, `secp256k1_ec_pubkey_tweak_add`,
  `secp256k1_ec_pubkey_tweak_mul`, `secp256k1_ec_pubkey_combine`): NULL pubkey, tweak32,
  out, or ins pointer now fires the illegal callback before returning 0.
- **Security impact**: These are defensive-programming fixes at the ABI boundary. They
  eliminate silent-return-0 on provably-wrong inputs, matching libsecp256k1's fail-loud
  contract. No signing, CT arithmetic, or private key path is affected.

### 2026-05-21 musig2.cpp -- SEC-005/SEC-009 BIP-327 infinity nonce enforcement

- **SEC-005:** `musig2_start_sign_session` now enforces BIP-327 §GetSessionValues step 2: aborts with an invalid session if `agg_nonce.R1` or `agg_nonce.R2` is the point at infinity. Previously, infinity inputs would cause `to_compressed()` to be called on an invalid point.
- **SEC-009:** `musig2_nonce_agg` now guards the empty-vector case, returning an all-infinity `MuSig2AggNonce` that is subsequently rejected by SEC-005.
- **Security impact:** Removes a class of degenerate-input attack where a cancellation nonce (R1+…+Rn = ∞) or empty aggregation could produce a session with an all-zero challenge, silently signing under a broken protocol state.
- **No CT boundary change:** Both guard paths involve only public nonce data. The CT signing paths (`musig2_partial_sign` via `ct::scalar_mul/add`) are unchanged.

### 2026-05-14 ct_field.cpp -- dead-code cleanup (Werror + MSVC fix)

- Deleted unused `static add256()` and `add_carry_u64()` helpers left
  over from the prior delegate refactor. Reverted my `__int128`
  addition to `sub256` (tripped `-Werror=pedantic`). No semantics change.

### 2026-05-14 ct_field.cpp -- delegate field_add/sub/neg to fast

- **`src/cpu/src/ct_field.cpp`**: Removed the hand-written parallel
  add256/sub256/cmov256 chains and let `ct::field_add` / `field_sub` /
  `field_neg` call `fast::operator+ / - / unary minus` directly. The
  fast layer is already branchless (4 × `add64` + XOR-mask reduce —
  see `field.cpp::add_impl`).
- **Security impact**: None. Same constant-time guarantee, expressed
  via the fast path. Removed two miscompile classes: Clang ThinLTO at
  -O3 (CI / linux clang-17 + Sanitizers) and Clang -O0 with sanitizer
  shadow memory.
- **No public-API change.**

### 2026-05-14 ct_field.cpp -- Clang sanitizer detection (build-only)

- **`src/cpu/src/ct_field.cpp`**: Sanitizer-detection guards replaced with a
  unified `SECP256K1_HAS_SANITIZER` macro that covers both GCC
  (`__SANITIZE_THREAD__` / `_ADDRESS_` / `_MEMORY_`) and Clang
  (`__has_feature(thread_sanitizer)` etc.). Clang TSan/MSan/ASan were
  previously not detected, causing the LTO-barrier asm to run under
  sanitizers and produce false positives.
- **Security impact**: None — CT primitives (`add256`, `sub256`, FE52 multiply,
  SafeGCD inverse) and their constant-time guarantee are unchanged. Production
  Release builds still use the LTO-barrier asm. Sanitizer builds correctly
  fall through to the portable path without the barrier (which was the
  intent from day one — the bug was the guard didn't catch Clang).
- **No public-API change.**

### 2026-05-12 ufsecp_musig2.cpp -- SEC-001 MuSig2 ABI signer-index cross-validation

- **`src/cpu/src/impl/ufsecp_musig2.cpp` (`ufsecp_musig2_partial_sign_v2`)**: New ABI
  function that validates `privkey ↔ signer_index` before consuming any secret material.
  The original `ufsecp_musig2_partial_sign()` cannot perform this check because the 165-byte
  keyagg blob does not store individual public keys.
- **Validation mechanism**: Derives `pubkey = secp256k1::ct::generator_mul(privkey)` (constant-time)
  and compares it against `pubkeys[signer_index * 33 .. +33]` using a constant-time byte loop
  (`diff |= derived[i] ^ expected[i]` for all 33 bytes). Returns `UFSECP_ERR_BAD_KEY` on mismatch.
- **Nonce erasure**: `secnonce` is zeroed via `ScopeSecureErase` on ALL exit paths, including
  validation failure, to prevent nonce reuse. Matches BIP-327 nonce-erasure requirements.
- **ABI compatibility**: Original `ufsecp_musig2_partial_sign()` preserved unchanged for backward
  compatibility; security warning added to its header doc and implementation directing callers to v2.
- **CT contract**: The validation path (ct::generator_mul) is constant-time on the privkey input.
  The comparison loop is constant-time (no early-exit). Validation failure does not reveal timing
  information about the private key to an attacker controlling signer_index.
- **Test**: `audit/test_regression_musig2_abi_signer_index.cpp` (SIV-1..7): wrong-index
  rejected, correct-index accepted, null-pubkeys rejected, out-of-range rejected, 3-of-3
  correct/wrong indices, full 2-of-2 round-trip.
- **Hostile-caller coverage**: `docs/FFI_HOSTILE_CALLER.md §Section L`.

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

### GPU collect-verify ABI (libbitcoin bridge, 2026-06-02 — PUBLIC-DATA)

`ufsecp_gpu_ecdsa_verify_collect` / `ufsecp_gpu_schnorr_verify_collect` are a
libbitcoin-bridge specialization of batch verification. They are **public-data
operations**: inputs are messages, public keys and signatures (all on-chain), and
`key_buffer` carries only opaque caller verdict/correlation markers — **no private
key, nonce, or secret material is processed**, so the CT-vs-variable-time boundary
does not apply (variable-time verify is correct, per the project's verify-path
rule). The verdict is written in place into a 1-byte-per-row cell (valid → 0,
invalid → left). The dedicated CUDA kernel is a verbatim copy of the audited verify
kernel with only the output store changed, so its accept/reject verdict is
bit-identical to `ufsecp_gpu_*_verify_batch` (proven GPU==CPU==libsecp by
`test_lbtc_consensus_diff`). OpenCL/Metal return `Unsupported` and the bridge falls
back to the host-collapse path. Hostile-caller quartet: see
[FFI_HOSTILE_CALLER.md](FFI_HOSTILE_CALLER.md) Section J.

### CPU multi-threaded batch verify (`ecdsa_batch_verify_mt`, 2026-06-17 — PUBLIC-DATA)

`secp256k1::ecdsa_batch_verify_mt` (engine) and its thin ABI wrapper
`ufsecp_ecdsa_batch_verify_mt` add **CPU parallelism as a first-class engine
feature**: a large ECDSA batch is split into fixed-size chunks pulled from an
atomic work queue and verified across up to `max_threads` threads (0 = auto =
`hardware_concurrency()`, 1 = serial). The worker count is the caller's to own:
an explicit `max_threads` is honoured and reduced only to what the hardware can
run (and to the chunk count) — there is **no arbitrary upper cap** (the former
hard-coded 64-thread limit and fixed `std::array<std::thread,64>` pool were
removed 2026-06-17 in favour of a dynamic `std::vector<std::thread>`); workers
inherit the calling process's thread priority. Like all verification, inputs are
**public data** (message hashes,
public keys, signatures — all on-chain), so **no private key, nonce, or secret
material is processed** and the CT-vs-variable-time boundary does not apply
(variable-time verify is correct per the verify-path rule). Threading therefore
has **zero side-channel relevance**: each chunk runs the audited serial
`ecdsa_batch_verify`, and the boolean accept/reject result is bit-identical to
the single-threaded path for any thread count (proven by
`regression_ecdsa_batch_verify_mt`: parity across thread counts {0,1,2,4,8,64}
and above the old cap {65,128,256,1024}, a source-scan asserting no `kMaxThreads`
cap remains, single-sig corruption detection, and corruption propagation across
multi-chunk batches). Thread-safety rests on the GLV/generator precompute being a C++11
function-local magic static and `ecdsa_batch_verify`'s `thread_local` scratch —
the same guarantees the existing parallel sign batch relies on. The serial
`ufsecp_ecdsa_batch_verify` is unchanged; integrators choose. Hostile-caller
quartet: see [FFI_HOSTILE_CALLER.md](FFI_HOSTILE_CALLER.md) Section I.5.

The same public-data MT model extends (2026-06-22) to **`ufsecp_schnorr_batch_verify_mt`**
and **`ufsecp_ecdsa_verify_opaque_rows_mt`**, and to the libbitcoin-bridge packed-row
verify twins `ufsecp_lbtc_verify_{ecdsa[_opaque|_compact],schnorr}_mt`. These reuse the
audited serial marshalling and only swap the all-valid fast check to
`secp256k1::{ecdsa,schnorr}_batch_verify_mt`; the per-row locate fallback stays serial.
All inputs remain **public data** (message hashes, x-only/compressed public keys,
signatures), so no secret material is processed, the CT-vs-variable-time boundary does
not apply, and threading has **zero side-channel relevance**. In the bridge the GPU path
is unaffected (`max_threads` governs the CPU fallback only) and the cancellation token is
still polled between chunks. Per-row verdicts are bit-identical to serial for any thread
count (parity across {0,1,2,8} plus a 4096-chunk-boundary corruption case in
`test_lbtc_bridge`; hostile-caller negatives in `test_c_abi_negative`, Section I.5b). The
single-threaded functions are unchanged; integrators that shard across their own thread
pool keep using them. Hostile-caller quartet: see
[FFI_HOSTILE_CALLER.md](FFI_HOSTILE_CALLER.md) Section I.5b.

### GPU per-item batch ABI (libbitcoin bridge, 2026-06-08 — PUBLIC-DATA)

`ufsecp_gpu_xonly_validate`, `ufsecp_gpu_commitment_verify` and
`ufsecp_gpu_tagged_hash` are libbitcoin-bridge per-item batch kernels. They are
**public-data operations**: inputs are x-only keys, BIP-341 tweaks/outputs, and
script-tree messages (all on-chain / caller-public) — **no private key, nonce, or
secret material is processed**, so the CT-vs-variable-time boundary does not apply
(variable-time is correct, per the verify-path rule). CUDA, OpenCL, and Metal each
dispatch a native one-thread-per-item on-device kernel (no host-CPU fallback).
Correctness is anchored bit-for-bit on
the libsecp shim: `ufsecp_gpu_xonly_validate` == `secp256k1_xonly_pubkey_parse`
(including the `x ≥ p` reject), `ufsecp_gpu_commitment_verify` ==
`secp256k1_xonly_pubkey_tweak_add_check` per row, and `ufsecp_gpu_tagged_hash` ==
`secp256k1_tagged_sha256` per item — proven in `tests/test_lbtc_commitment.cpp`
(section 9). Outputs are fail-closed (cleared on any error). Hostile-caller quartet:
see [FFI_HOSTILE_CALLER.md](FFI_HOSTILE_CALLER.md) Section J (J.lbtc-batch).

Three further libbitcoin-bridge batch kernels follow the same model (native CUDA,
OpenCL, and Metal on-device kernels, no host-CPU fallback, fail-closed, all
PUBLIC-DATA):
`ufsecp_gpu_pubkey_validate` (full compressed-pubkey on-curve check, == shim
`ec_pubkey_parse`), `ufsecp_gpu_tagged_hash_var` (TapLeaf per-item-length tagged
hash, == shim `tagged_sha256`), and `ufsecp_gpu_hash256` (double-SHA-256 / merkle
node hashing, == SHA256d reference) — all proven per item in
`tests/test_lbtc_commitment.cpp` (sections 10–12).

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
<!-- 2026-05-21: ecdsa.cpp SEC-003 — ECDSASignature::from_compact deprecated via [[deprecated]] attribute in both overloads; callers directed to parse_compact_strict which enforces r,s in (0, n). Array overload inlined (was calling the pointer overload, triggering -Werror=deprecated-declarations). No security-relevant behavioral change. -->
<!-- 2026-05-24: bip39.cpp — extracted decode_bip39_words() static helper from bip39_validate / bip39_mnemonic_to_entropy (duplicated 32-line bit-reconstruction + SHA256 checksum block). Helper signature passes entropy via an out-pointer the caller already secure_erases on every exit path. Pure code-shape refactor; BIP-39 entropy claims (zeroization on all exit paths) unchanged. -->
<!-- 2026-05-24: sp_scanner.cpp + ltc/ltc_sp.cpp — extracted scan_batch pipeline into sp_scan_batch_impl.hpp (templated helper parameterised by SHA256 domain-separation tag midstate). BTC BIP-352 and LTC-SP scanners now share the impl; the scan path operates only on public inputs (output pubkeys, A_sum, S_comp) plus scan_privkey via batch GLV (no oracle, batch input — timing-leaks-nothing argument unchanged). No secret-bearing state newly persisted; security claims unchanged. -->
<!-- 2026-05-24: types.hpp (include/secp256k1/types.hpp + src/cpu/include/secp256k1/types.hpp) — collapsed four inline {fe,sc}_to_data static_cast bodies into a single detail::to_data_cast<T>() template. Public wrappers retain their names/signatures/noexcept; -O3 inlines them to the original cast. types.hpp defines struct layouts + zero-cost cast helpers only — no secret material is touched. Pure code-shape refactor; CT/zeroization/lifetime claims unchanged. -->
<!-- 2026-05-24: ellswift.cpp — hoisted XSWIFTEC constants to file scope and extracted ellswift_create_retry_loop helper. All three ellswift_create / ellswift_create_fast callers now share one retry loop. CT routing of pubkey derivation is unchanged (each caller still chooses ct::generator_mul vs scalar_mul_generator); secure_erase of the random-bytes scratch is preserved (now done once in the helper instead of duplicated three times). BIP-324 ellswift security claims unchanged. -->
<!-- 2026-05-25: ecdsa.cpp + recovery.cpp + shim_ellswift.cpp — three CT hardening fixes. (1) SEC-CPU-004: ecdsa_sign() and ecdsa_sign_hedged() — r.is_zero() guard on nonce-derived scalar r replaced with r.is_zero_ct(). (2) SEC-CPU-005: ecdsa_sign_recoverable() — r.is_zero() and s.is_zero() guards replaced with r.is_zero_ct() and s.is_zero_ct(). Both r and s are derived from the secret nonce k; branching VT is a timing leak. Probability of triggering: ~2^{-128}. CT invariant for ECDSA/recoverable signing paths: all guards on nonce-derived scalars now use the CT comparison variant. (3) SHIM-006: secp256k1_ellswift_xdh() general XDH path — x32_arr (ECDH shared secret x-coord) now erased via secure_erase after hashfp(); sk and kb erased on both BIP-324 and general exit paths. Claim: BIP-324 ECDH shared-secret material does not persist on the stack after return. -->

<!-- 2026-05-26: ecdsa.cpp — rfc6979_nonce, rfc6979_nonce_hedged, rfc6979_nonce_libsecp_compat iter-2 parse changed from (void) discard to bool const ok2 = ...; (void)ok2; (scanner quality fix). CT select logic, zeroization coverage, and RFC 6979 security claims are unaffected. -->

<!-- 2026-05-28: shim_ecdsa.cpp + shim_recovery.cpp + shim_ellswift.cpp + bip32.cpp — secret-key stack-residue hardening (CT-01/SHIM-01/02/CT-02). Claim: parsed private-key scalars and BIP-324 handshake key material do not persist on the stack after the call returns. (CT-01) shim ECDSA sign / sign_recoverable secure_erase the parsed key scalar (`k` / `privkey_scalar`) on every return path; (SHIM-01/02) ellswift_create and ellswift_xdh erase `sk`+`kb` on all returns (success, parse-fail, and the three xdh error branches), completing SHIM-006; (CT-02) BIP-32 hardened derive_child erases the HMAC-derived `il_scalar` on all 7 return paths. Also RT-02: secp256k1_ecdsa_signature_parse_der now requires exact SEQUENCE consumption (`p == end`), rejecting trailing bytes inside the SEQUENCE — matching upstream + the native C ABI parser. Output bytes written before erase; no behavioral change. Regression guard: audit/test_regression_shim_seckey_erase.cpp. -->

*UltrafastSecp256k1 v4.5.0 -- Security Claims*
