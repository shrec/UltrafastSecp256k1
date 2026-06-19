# FFI Hostile-Caller Coverage

**Last updated**: 2026-06-13 | **Version**: 4.4.0

### 2026-06-13 - Opaque ECDSA ABI compatibility

`include/ufsecp/ufsecp.h` now exposes opaque ECDSA helpers for consumers whose
public signature type stores copied libsecp-compatible
`secp256k1_ecdsa_signature` scalar storage: compact↔opaque conversion,
opaque low-S normalization, single verify, column batch verify, and strided-row
verify. These APIs reject NULL pointers, zero/invalid scalar limbs, malformed
compressed public keys, oversized batch counts, and undersized row strides. The
verify APIs return per-row verdict bytes for batch/row calls; malformed rows are
reported as invalid verdicts rather than aborting the whole batch.

`include/ufsecp/ufsecp_gpu.h` also exposes
`ufsecp_gpu_ecdsa_verify_opaque_rows`; `ufsecp_gpu_ecdsa_verify_lbtc_rows` is a
compatibility alias. Hostile-caller quartet coverage is in
`audit/test_gpu_abi_gate.cpp`: NULL context/buffer rejection, `count=0`
zero-edge, `stride < 129` invalid-content rejection, and valid opaque-row smoke
when GPU hardware is present.

Coverage: `audit/test_ffi_round_trip.cpp` checks compact↔opaque conversion,
high-S normalization, single verify, column batch, and strided-row batch;
`audit/test_c_abi_negative.cpp` checks NULL/zero/invalid argument contracts;
`compat/libbitcoin_bridge/tests/test_lbtc_bridge.cpp` checks libbitcoin opaque
rows and columns are accepted without rewriting caller-owned signatures.

### 2026-06-06 - GPU C ABI fail-closed output contract

`include/ufsecp/ufsecp_gpu.h` and `src/cpu/src/ufsecp_gpu_impl.cpp` now
document and enforce a hostile-caller fail-closed contract for result-bearing
GPU C ABI calls:

- Output buffers are cleared to zero/invalid defaults before backend dispatch.
- If a backend returns non-OK, the same output buffers are cleared again before
  the ABI returns to the caller.
- The in-place collect APIs are excluded because `key_buffer` is caller-owned
  verdict-marker state used for CPU fallback; it is not a secret buffer.

This prevents a hostile or buggy caller from observing stale or partial GPU
results after failures in generator multiplication, batch verification, ECDH,
Hash160, MSM, FROST partial verification, ecrecover, ZK verification,
Bulletproof verification, BIP-324 AEAD, SNARK witness helpers, or BIP-352 scan.

BIP-352 scan is now explicitly hostile-input hardened at the C ABI and backend
boundary: zero or order-or-larger scan keys fail with `UFSECP_ERR_BAD_KEY` /
`GpuError::BadKey`, invalid spend/tweak public keys fail before kernel use, and
`prefix64_out` remains zero on failure. CUDA, OpenCL, and Metal additionally
erase uploaded scan-key or scan-plan buffers before device memory release.

Coverage: `audit/test_gpu_host_api_negative.cpp` checks NULL/invalid content
and fail-closed output behavior; `audit/test_gpu_bip352_scan.cpp` checks
BIP-352 zero/order scan key rejection and prefix clearing; and
`audit/test_exploit_gpu_bip352_key_erase.cpp` source-scans CUDA/OpenCL
scan-key/plan erasure.

### 2026-06-02 - failure-path output zeroing + recover overflow guard (review fixes)

ABI/shim hardening from the 2026-06-01 review tightens the hostile-caller
contract at the C boundary (`include/ufsecp/ufsecp.h` and the libsecp256k1 shim):

- **Failure-path output zeroing (PASS-COMPAT-001/002/004).** On any FAILED
  operation the output buffer is zeroed before return, matching upstream:
  `secp256k1_ec_seckey_negate/_tweak_add/_tweak_mul` zero the caller's seckey
  (`shim_seckey.cpp`); `secp256k1_ecdsa_recover` zeros the output pubkey
  (`shim_recovery.cpp`); `secp256k1_xonly_pubkey_tweak_add` zeros the output
  pubkey (`shim_extrakeys.cpp`). A hostile caller cannot read a stale/partial
  key or pubkey out of a failed call.
- **ECDSA recover overflow guard (bbhunt-001).** `ecdsa_recover` (CPU + shim +
  ABI + CUDA/OpenCL kernels) rejects any `r >= (p - n)` in the recid&2 branch,
  matching upstream `secp256k1_ecdsa_sig_recover`. A hostile caller crafting
  `recid&2` with `r ∈ [p-n, n)` can no longer obtain a bogus public key returned
  as success — recovery returns failure (0), as upstream does.
- **Constant-time variable-base scalar mul (CT-CRYPTO-001).**
  `secp256k1::ct::scalar_mul(Point,Scalar)` (reached from ECDH /
  `ec_seckey_tweak_mul`) no longer branches on the secret GLV sign bit; the GLV
  "v" half-scalars are built with a branchless masked select.
- **ABI version contract.** `ufsecp_abi_version()` returns `4` (== `PROJECT_VERSION_MAJOR`);
  all bindings pin `EXPECTED_ABI = 4` and fail closed at context creation on
  mismatch — a mismatched/hostile binding cannot construct a context.

Coverage: `audit/test_regression_recover_rplus_n_overflow` (REC-1..4),
`audit/test_regression_ct_glv_make_v_branchless` (CT-GLV-1..3),
`compat/libsecp256k1_shim/tests/shim_test.cpp`
(`test_seckey_failure_clear` + `test_passcompat_failure_clear`),
`ci/check_abi_version_sync.py`.

### 2026-05-28 — batch parallel dispatch: hostile-caller contract

`ufsecp_ecdsa_sign_batch` / `ufsecp_schnorr_sign_batch` (ABI layer in
`src/cpu/src/ufsecp_impl.cpp`): a hostile caller supplying overlapping or
aliased `sigs64_out` / `privkeys32` / `msgs32` buffers triggers undefined
behaviour at the C layer (same as any other buffer-overlap violation). The
parallel path does not create new aliasing vulnerabilities: threads operate on
non-overlapping slot-index ranges computed at entry. On first error the entire
`sigs64_out` is zeroed before return — a hostile caller cannot read a partial
valid signature out of a failed batch call.

### 2026-05-24 — v9 RT-002 / TASK-002: Adaptor signing DPA blinding

`src/cpu/src/adaptor.cpp`: every secret-scalar generator multiplication
now uses `ct::generator_mul_blinded(...)` (was `ct::generator_mul(...)`):
- `schnorr_adaptor_sign` long-term key derivation `P = sk*G`.
- `ecdsa_adaptor_sign` secret nonce derivation `base_nonce = k*G`.
A hostile caller observing power/EM traces cannot recover `sk` or the
adaptor `k` from a single trace; the blinding mask is applied when
`secp256k1_context_randomize()` has been called. Verifier-facing ABI
behaviour is unchanged — these are micro-architectural side-channel
hardening only.

Hostile-caller coverage: `audit/test_regression_adaptor_blinded_nonce.cpp`
adds `test_adaptor_blinded_all_secret_sites_source_scan` enforcing the
presence of `generator_mul_blinded` and the absence of any bare
`ct::generator_mul(k)` / `ct::generator_mul(private_key)` in the adaptor
source.

### 2026-05-24 — v9 RT-006/-007/-014/-015 / TASK-022: secret stack residue bundle

CPU signing paths hardened against secret residue on the stack frame
after the function returns:

- `schnorr_sign(const Scalar&)` / `schnorr_sign_verified(const Scalar&)`
  (`src/cpu/src/schnorr.cpp:502,511`): raw-key overloads now
  `secure_erase(&kp.d, ...)` after the sub-call returns. A hostile
  caller that scrapes its caller's stack frame after the call returns
  finds only zeros where the negated signing scalar used to live.
- `bip32::derive_child` (`src/cpu/src/bip32.cpp:414`): uses CT
  `child_scalar.is_zero_ct()` on the secret-derived child scalar —
  closes a probabilistic timing side-channel on the
  (extremely unlikely) wraparound case.
- FROST `derive_scalar` / `derive_scalar_pair`
  (`src/cpu/src/frost.cpp:60-106`): SHA256 state, tag hash, and the
  finalized hash buffer are all securely erased before return; they
  incorporate the seed (secret material) into a stack-resident state.
- `ecdsa_adaptor_sign` degenerate-r early return
  (`src/cpu/src/adaptor.cpp:311`): pre-erases `k`, `binding`, and
  `R_x_bytes` before returning the zero sentinel — the success-path
  erase block is unreachable from this branch.

Hostile-caller coverage:
`audit/test_regression_secret_stack_residue_v9.cpp` source-scans every
affected function for the required erase / CT-predicate calls and
round-trips the schnorr raw-key overloads. 16/16 checks pass on the
patched code; pre-fix code would fail the four source-scan assertions.

### 2026-05-24 — v9 RT-001 / TASK-001: MuSig2 v1 partial_sign hard-fail

`ufsecp_musig2_partial_sign` (v1) at
`src/cpu/src/impl/ufsecp_musig2.cpp:203` now returns
`UFSECP_ERR_DEPRECATED_API` (new error code 12) on every call. The output
buffer is zeroed and the `secnonce` is securely erased on the reject path
to protect callers that ignore the return code from accidental nonce
reuse. NULL-arg validation still takes precedence — a NULL `ctx` /
`secnonce` / `privkey` / `keyagg` / `session` / `partial_sig32_out` still
returns `UFSECP_ERR_NULL_ARG`, preserving argument-discipline coverage.

Rationale (v9 RT-001): the v1 ABI cannot enforce the Rule-13
privkey↔signer_index cross-check because the keyagg blob does not carry
per-signer pubkeys; the prior revision left this check silently disarmed.
Callers must migrate to `ufsecp_musig2_partial_sign_v2`, which carries
the `pubkeys[]` array and validates the binding in constant time at the
ABI boundary.

Hostile-caller coverage added:
`audit/test_regression_musig2_v1_partial_sign_deprecated.cpp` —
verifies the reject path, the output-zero and secnonce-erase invariants,
the NULL-arg precedence, and confirms v2 is unaffected.

### 2026-05-22 — TASK-007: ufsecp_musig2_partial_sign v1 ABI deprecation restored (SUPERSEDED 2026-05-24)

`include/ufsecp/ufsecp.h` `ufsecp_musig2_partial_sign` once again carries
`UFSECP_DEPRECATED(...)` directing callers to
`ufsecp_musig2_partial_sign_v2()` for the safe path. The attribute was
previously dropped to keep -Werror builds green; this commit's compromise:
* v2's internal delegation to v1 (already validated) is wrapped in a
  `#pragma GCC diagnostic ignored "-Wdeprecated-declarations"` push/pop;
* seven audit-test files that intentionally call v1 to verify the ABI
  contract get per-source `-Wno-deprecated-declarations` via
  `set_source_files_properties` in `audit/CMakeLists.txt`
  (test_adversarial_protocol, test_ffi_round_trip,
  test_regression_musig2_abi_signer_index, test_regression_musig2_zero_psig,
  test_exploit_musig2_nonce_erasure_le32_ecdh,
  test_exploit_musig2_parallel_session_cross,
  test_exploit_musig2_partial_forgery).
External callers get the compile-time warning that points them at v2 —
loud, unambiguous, and impossible to miss; internal call sites stay
buildable under -Werror.

Documents the hostile-caller test coverage for the C ABI (`ufsecp_*` functions). All tests are in `audit/test_adversarial_protocol.cpp`:
- Section G (FFI Hostile-Caller) — original 97-function coverage
- Section H (New ABI Surface Edge Cases) — 26 additional functions added in v3.22+

Secret-bearing ABI boundary changes are under stricter change control:
`ci/check_secret_path_changes.py` requires paired updates to this document
and `docs/SECURITY_CLAIMS.md` whenever secret-bearing ABI surfaces change.

---

## Attack Vectors Tested

| Vector | Test ID | Description | Functions covered |
|--------|---------|-------------|-------------------|
| **Null context** | G.1 | Pass `ctx = NULL` | All 97 functions |
| **Null output pointer** | G.2 | Valid inputs but `out = NULL` | sign, pubkey_create, ecdh, bip32, addresses |
| **Null input pointer** | G.3 | `privkey = NULL`, `sig = NULL`, `pubkey = NULL` | sign, verify, ecdh, parse |
| **All-zero private key** | G.4 | `privkey = {0}` (scalar == 0) | seckey_verify, sign, pubkey_create |
| **All-0xFF private key** | G.5 | `privkey = {0xFF..FF}` (scalar > n) | seckey_verify, sign, pubkey_create |
| **Invalid pubkey prefix** | G.6 | Prefix byte `0x00`, `0x01`, `0x05`, `0xFF` | pubkey_parse, ecdsa_verify, ecdh |
| **Off-curve pubkey** | G.7 | Valid prefix `0x02` + random x not on curve | pubkey_parse, ecdsa_verify |
| **Zero signature (r=0, s=0)** | G.8 | 64 zero bytes as compact sig | ecdsa_verify, ecdsa_recover |
| **Max scalar signature** | G.9 | r = n, s = n (>= order) | ecdsa_verify, schnorr_verify |
| **Malformed DER** | G.10 | Truncated, wrong tags, long-form length, trailing bytes | ecdsa_sig_from_der |
| **Empty batch arrays** | G.11 | `count = 0` with valid pointers | ecdsa_batch_verify, schnorr_batch_verify |
| **Single-element batch** | G.12 | `count = 1` (edge case) | ecdsa_batch_verify, schnorr_batch_verify |
| **Oversized batch count** | G.13 | `count = UINT32_MAX` with small buffer | ecdsa_batch_verify, schnorr_batch_verify |
| **Undersized pubkey buffer** | G.14 | 32 bytes instead of 33 for compressed | pubkey_parse |
| **Undersized sig buffer** | G.15 | 63 bytes instead of 64 for compact | ecdsa_verify |
| **Overlapping input/output** | G.16 | `privkey == out_pubkey` (aliased pointers) | pubkey_create |
| **Invalid WIF string** | G.17 | Non-base58 chars, wrong checksum, truncated | wif_decode |
| **Invalid mnemonic** | G.18 | Wrong checksum, non-wordlist words, empty | bip39_validate |
| **Invalid BIP-32 path** | G.19 | Empty, missing `m/`, negative index, overflow | bip32_derive_path |
| **ECIES hostile inputs** | G.20 | Zero-length plaintext, truncated envelope (<82B), wrong HMAC, corrupted ciphertext, oversized (1MB) | ecies_encrypt, ecies_decrypt |

---

## Coverage Matrix by Error Code

| Error code | What triggers it | Hostile-caller test? |
|------------|-----------------|---------------------|
| `ERR_NULL_ARG` (1) | Any NULL pointer argument | G.1, G.2, G.3 |
| `ERR_BAD_KEY` (2) | privkey == 0 or >= n | G.4, G.5 |
| `ERR_BAD_PUBKEY` (3) | Bad prefix, off-curve, x >= p, infinity | G.6, G.7, G.14 |
| `ERR_BAD_SIG` (4) | r/s == 0 or >= n, malformed DER | G.8, G.9, G.10, G.15 |
| `ERR_BAD_INPUT` (5) | Wrong length, invalid format, bad count | G.11-G.14, G.17-G.19 |
| `ERR_VERIFY_FAIL` (6) | Signature verification failed (valid format, wrong key) | Standard verify tests |
| `ERR_ARITH` (7) | Scalar overflow during tweak | Tweak tests |
| `ERR_BUF_TOO_SMALL` (10) | Output buffer insufficient | G.14, ECIES |

---

## Additional Coverage (beyond section G)

| Source | Hostile patterns | Check count |
|--------|-----------------|-------------|
| `test_fuzz_parsers.cpp` | 580K+ random malformed pubkeys + DER sigs | ~580,000 |
| `test_fuzz_address_bip32_ffi.cpp` | Random invalid addresses, WIF, BIP-32 paths, BIP-39 | ~83,000 |
| `test_wycheproof_ecdsa.cpp` | Boundary r/s, bit-flipped sigs, invalid DER | 500+ vectors |
| `test_wycheproof_ecdh.cpp` | Infinity, twist, off-curve, zero key | 200+ vectors |
| `test_ecies_regression.cpp` | Wrong key, truncated, empty, 1MB, overlapping | 85 checks |
| `test_ffi_round_trip.cpp` | Full round-trip all 97 functions with valid inputs | 286 calls |
| `test_fault_injection.cpp` | Bit-flip in scalar/point/signature mid-computation | 50+ checks |

---

## Section H: New ABI Surface Edge Cases (v3.22+)

A gap analysis found 26 `ufsecp_*` functions with no dedicated edge-case tests.
All gaps are closed by `test_h1_*`–`test_h12_*` in `test_adversarial_protocol.cpp`.

| Test ID | Functions | Coverage |
|---------|-----------|----------|
| H.1 | `ufsecp_ctx_size` | positive-size smoke |
| H.2 | `ufsecp_aead_chacha20_poly1305_encrypt`, `ufsecp_aead_chacha20_poly1305_decrypt` | NULL guards, bad-tag, wrong-nonce, zero-length roundtrip |
| H.3 | `ufsecp_ecies_encrypt/decrypt` | NULL guards, off-curve pubkey, tampered envelope |
| H.4 | `ufsecp_ellswift_create/xdh` | NULL guards, zero privkey, symmetric shared secret |
| H.5 | `ufsecp_eth_address_checksummed`, `ufsecp_eth_personal_hash` | NULL guards, undersized buffer |
| H.6 | `ufsecp_pedersen_switch_commit` | NULL guards, prefix byte validation |
| H.7 | `ufsecp_schnorr_adaptor_extract` | NULL guards, zero inputs |
| H.8 | `ufsecp_ecdsa_sign_batch`, `ufsecp_schnorr_sign_batch` | NULL ctx/msgs/keys/output, count=0 |
| H.9 | `ufsecp_bip143_sighash`, `ufsecp_bip143_p2wpkh_script_code` | NULL guards, OP_DUP OP_HASH160 PUSH20 format |
| H.10 | `ufsecp_bip144_txid/wtxid/witness_commitment` | NULL guards, determinism |
| H.11 | `ufsecp_segwit_is_witness_program`, `ufsecp_segwit_parse_program`, `ufsecp_segwit_p2wpkh_spk`, `ufsecp_segwit_p2wsh_spk`, `ufsecp_segwit_p2tr_spk`, `ufsecp_segwit_witness_script_hash` | NULL guards, format correctness, non-witness rejection |
| H.12 | `ufsecp_taproot_keypath_sighash`, `ufsecp_tapscript_sighash` | NULL guards, count=0, OOB index, determinism |

---

## Section I: Remaining ABI Surface (v3.23+)

A second gap analysis found 8 `ufsecp_*` functions with zero edge-case coverage, plus
shallow batch-verify paths. All gaps are closed by `test_i1_*`–`test_i5_*` in
`test_adversarial_protocol.cpp`.

| Test ID | Functions | Coverage |
|---------|-----------|----------|
| I.1 | `ufsecp_ctx_clone`, `ufsecp_last_error_msg` | NULL guards, independent clone (results match), error state propagation |
| I.2 | `ufsecp_pubkey_parse`, `ufsecp_pubkey_create_uncompressed` | NULL guards, bad prefix/length, 0x04 output format, compressed round-trip |
| I.3 | `ufsecp_ecdsa_sign_recoverable`, `ufsecp_ecdsa_recover` | NULL guards (all 4 args), recid in [0,3], recovery round-trip, invalid recid rejection |
| I.4 | `ufsecp_ecdsa_sign_verified`, `ufsecp_schnorr_sign_verified` | NULL guards, zero privkey, output verified via ecdsa_verify / schnorr_verify |
| I.5 | `ufsecp_schnorr_batch_verify`, `ufsecp_ecdsa_batch_verify`, `ufsecp_batch_identify_invalid` | Valid entry passes, tampered sig fails, identify_invalid returns correct index, count=0 vacuously OK |
| I.5a | `ufsecp_ecdsa_batch_verify_mt` | NULL ctx/entries rejected; valid batch passes (smoke) and result is identical to serial for every thread count, including counts above the former 64-thread cap (no arbitrary upper limit; dynamic `std::vector<std::thread>` pool, reduced only to `hardware_concurrency`); tampered/invalid sig fails; count=0 (n=0) vacuously OK; max_threads=0 auto, 1 serial — public-data variable-time verify, threading has no side-channel relevance |
| I.6 | `ufsecp_context_randomize` | NULL ctx rejected; zero-seed (all-zero 32 bytes) accepted — clears/resets blinding; valid random seed succeeds (smoke); signing after randomize produces valid signature |
| I.7 | `ufsecp_ecdsa_sign` with non-NULL `noncefp` | NULL noncefp treated as RFC 6979 default; custom noncefp returning valid scalar succeeds; custom noncefp returning zero scalar triggers retry; output valid after custom nonce |

---

## Section J: GPU C ABI (v3.24+)

`test_gpu_host_api_negative.cpp` and `test_gpu_abi_gate.cpp` cover all 22
`ufsecp_gpu_*` functions without requiring GPU hardware (GPU-only smoke paths run
when a device is present). Both files are integrated into the unified audit runner
(modules `gpu_api_negative` and `gpu_abi_gate`).

| Test File | Checks | Coverage |
|-----------|--------|----------|
| `test_gpu_host_api_negative` | 38 | NULL ctx for all batch ops; NULL ctx_out / info_out; ctx_create with backend 0/99/255; is_available/device_count for invalid backend; count=0 no-ops; NULL buffers + count>0; invalid device index; GPU error strings (7 codes); backend name edge cases (0, 99, 0xFFFFFFFF) |
| `test_gpu_abi_gate` | 39 | Backend count/ids/names (CUDA/OpenCL/Metal/none/invalid); device_info null guard + invalid backend + available device; ctx_create null/invalid/valid lifecycle; ctx_destroy(nullptr) no-crash; last_error/last_error_msg(nullptr); NULL buffer batch ops; error_str(OK/UNAVAILABLE/UNSUPPORTED/999); GPU ops if available (1*G smoke, count=0, NULL-scalar failure); `ufsecp_gpu_is_ready` NULL guard (returns 0) + valid ctx succeeds (smoke, returns 1) |

**J.collect — `ufsecp_gpu_ecdsa_verify_collect` / `ufsecp_gpu_schnorr_verify_collect`**
(libbitcoin bridge, PUBLIC-DATA; `key_buffer` carries only opaque verdict markers,
never secret material). Hostile-caller quartet in `test_gpu_abi_gate.cpp`:

| Function | null | zero | invalid | smoke |
|----------|------|------|---------|-------|
| `ufsecp_gpu_ecdsa_verify_collect`   | NULL ctx and NULL buffer → `UFSECP_ERR_NULL_ARG` | count=0 → no-op OK (zero-edge) | oversized count > cap → `UFSECP_ERR_BAD_INPUT` (invalid/reject) | valid signature zeroes its 1-byte verdict (success, when GPU present) |
| `ufsecp_gpu_schnorr_verify_collect` | NULL ctx → `UFSECP_ERR_NULL_ARG` | count=0 → no-op OK (zero-edge) | oversized count > cap → `UFSECP_ERR_BAD_INPUT` (invalid/reject) | valid Schnorr signature zeroes its verdict (success smoke) |

**J.opaque-ecdsa — `ufsecp_gpu_ecdsa_verify_opaque_rows` /
`ufsecp_gpu_ecdsa_verify_lbtc_rows`** (PUBLIC-DATA; direct strided rows carrying
copied libsecp-compatible ECDSA opaque scalar storage). Hostile-caller quartet in
`test_gpu_abi_gate.cpp`:

| Function | null | zero | invalid | smoke |
|----------|------|------|---------|-------|
| `ufsecp_gpu_ecdsa_verify_opaque_rows` | NULL ctx / NULL rows / NULL output → `UFSECP_ERR_NULL_ARG` | count=0 → no-op OK (zero-edge) | stride < 129 → `UFSECP_ERR_BAD_INPUT` (invalid/reject) | valid opaque row returns result 1 (success, when GPU present) |
| `ufsecp_gpu_ecdsa_verify_lbtc_rows` | NULL ctx / NULL rows / NULL output → `UFSECP_ERR_NULL_ARG` | count=0 → no-op OK (zero-edge) | stride < 129 → `UFSECP_ERR_BAD_INPUT` (invalid/reject) | alias of `ufsecp_gpu_ecdsa_verify_opaque_rows`; same valid-row smoke path |

For the collect APIs, backends without a native collect kernel (OpenCL/Metal)
return `Unsupported`; the libbitcoin bridge then falls back to the host-collapse
path (consensus-identical).

**J.lbtc-batch — `ufsecp_gpu_xonly_validate` / `ufsecp_gpu_commitment_verify` /
`ufsecp_gpu_tagged_hash`** (libbitcoin bridge, PUBLIC-DATA; no secret material on
these paths). CUDA-native per-item kernels; OpenCL/Metal return `Unsupported` and the
bridge falls back to its threaded CPU path (consensus-identical). Hostile-caller
quartet cross-checked against the libsecp shim in `tests/test_lbtc_commitment.cpp`
(section 9: GPU verdict == shim per row/key):

| Function | null | zero | invalid | smoke |
|----------|------|------|---------|-------|
| `ufsecp_gpu_xonly_validate`    | NULL ctx / NULL buffer → `UFSECP_ERR_NULL_ARG` | n=0 → no-op OK (zero-edge) | x ≥ p / off-curve key → result 0 (invalid/reject), matches shim xonly_parse | valid x-only key succeeds (smoke, GPU present) |
| `ufsecp_gpu_commitment_verify` | NULL ctx / any NULL column → `UFSECP_ERR_NULL_ARG` | n=0 → no-op OK (zero-edge) | corrupted tweaked_x / wrong parity / internal x ≥ p → result 0 (invalid/reject) | valid BIP-341 commitment succeeds (smoke); matches shim tweak_add_check |
| `ufsecp_gpu_tagged_hash`       | NULL ctx / NULL msgs → `UFSECP_ERR_NULL_ARG` | n=0 → no-op OK (zero-edge) | msg_len 0 or > 256 → `UFSECP_ERR_BAD_INPUT` (invalid/reject) | valid TapBranch digest succeeds (smoke); matches shim tagged_sha256 |
| `ufsecp_gpu_pubkey_validate`   | NULL ctx / NULL buffer → `UFSECP_ERR_NULL_ARG` | n=0 → no-op OK (zero-edge) | bad prefix / x ≥ p / off-curve → result 0 (invalid/reject), matches shim ec_pubkey_parse | valid compressed pubkey succeeds (smoke, GPU present) |
| `ufsecp_gpu_tagged_hash_var`   | NULL ctx / NULL msgs / NULL lens → `UFSECP_ERR_NULL_ARG` | n=0 → no-op OK (zero-edge) | stride 0 or > 256 → `UFSECP_ERR_BAD_INPUT` (invalid/reject) | valid TapLeaf per-item digest succeeds (smoke); matches shim tagged_sha256 |
| `ufsecp_gpu_hash256`           | NULL ctx / NULL inputs → `UFSECP_ERR_NULL_ARG` | n=0 → no-op OK (zero-edge) | input_len 0 or > 320 → `UFSECP_ERR_BAD_INPUT` (invalid/reject) | valid SHA256d digest succeeds (smoke); matches SHA256d reference |

---

## Section K: Deep Session Security (v3.4+)

`audit/test_adversarial_protocol.cpp` (K.1-K.6) covers BIP324 session protocol
security and scalar arithmetic edge cases.  K.1-K.3 are conditionally compiled
under `SECP256K1_BIP324`; K.4-K.6 are always-on.

| Test ID | Functions | Coverage |
|---------|-----------|----------|
| K.1 | `ufsecp_bip324_create`, `ufsecp_bip324_handshake`, `ufsecp_bip324_encrypt`, `ufsecp_bip324_decrypt` | 10-packet round-trip with counter integrity; tampered ciphertext rejected |
| K.2 | same | Cross-session isolation: session B cannot decrypt session A's ciphertext |
| K.3 | `ufsecp_bip324_handshake` | Double-handshake rejection (calling handshake twice on the same session object) |
| K.4 | `ufsecp_seckey_tweak_add` | Arithmetic overflow: k+t≡0 (mod n) must fail; identity tweak t=0 is valid |
| K.5 | `ufsecp_seckey_tweak_add`, `ufsecp_seckey_tweak_mul` | Out-of-range tweaks (≥ n) rejected; zero tweak for mul rejected; valid n-1 tweak succeeds |
| K.6 | `ufsecp_ecdh`, `ufsecp_ecdh_raw`, `ufsecp_ecdh_xonly` | Semantic differentiation (all three produce distinct encodings); ECDH commutativity; bad pubkey rejection |

---

## Guarantee

Every `ufsecp_*` function is tested with at least:
1. Valid inputs (FFI round-trip)
2. NULL context (G.1)
3. NULL critical pointers (G.2, G.3)
4. Malformed domain-specific input (G.4-G.20 / H.1-H.12 / I.1-I.5 / J.1-J.2, per function category)

**Mandatory edge-case rule for new ABI functions** (enforced since v3.22):
Every new `ufsecp_*` function MUST be covered by all four checks below before
an audit release commits it to the coverage matrix:
1. NULL rejection for every pointer parameter
2. Zero-count / zero-length / zero-key rejection where the contract requires it
3. Invalid-content rejection (bad prefix, off-curve, truncated, wrong tag, OOB index)
4. A success smoke test with valid inputs

No function can crash, hang, or leak memory on any hostile input. All reject with the appropriate `ufsecp_error_t` and leave output buffers untouched.

---

## Section L: MuSig2 ABI v2 — Signer-Index Cross-Validation (2026-05-12)

`ufsecp_musig2_partial_sign_v2` is a new secret-bearing ABI function that performs
`privkey ↔ signer_index` cross-validation before signing. Hostile-caller coverage
is provided by `audit/test_regression_musig2_abi_signer_index.cpp` (SIV-1..7).

| Test ID | Function | Coverage |
|---------|----------|----------|
| SIV-1 | `ufsecp_musig2_partial_sign_v2` | NULL pointer rejection (ctx=NULL, pubkeys=NULL) → `UFSECP_ERR_NULL_ARG` |
| SIV-2 | `ufsecp_musig2_partial_sign_v2` | Zero-value privkey → `UFSECP_ERR_BAD_KEY` (via parse_bytes_strict_nonzero) |
| SIV-3 | `ufsecp_musig2_partial_sign_v2` | signer_index out-of-range (99 for 2-signer keyagg) → `UFSECP_ERR_BAD_INPUT` |
| SIV-4 | `ufsecp_musig2_partial_sign_v2` | Wrong signer_index (SK0 claiming index 1) → `UFSECP_ERR_BAD_KEY` |
| SIV-5 | `ufsecp_musig2_partial_sign_v2` | 3-of-3: all signers succeed with correct index → `UFSECP_OK` |
| SIV-6 | `ufsecp_musig2_partial_sign_v2` | 3-of-3: all signers fail with neighbour index → `UFSECP_ERR_BAD_KEY` |
| SIV-7 | `ufsecp_musig2_partial_sign_v2` | Full 2-of-2 round-trip → valid Schnorr signature (success smoke) |

**Nonce erase guarantee:** `secnonce` is zeroed via `ScopeSecureErase` on ALL exit
paths — including validation failure — to prevent nonce reuse even when the wrong
`signer_index` is supplied. This matches BIP-327 nonce-erasure requirements.

<!-- 2026-04-28: ufsecp_gpu.h docstring corrected — ufsecp_gpu_context_create → ufsecp_gpu_ctx_create. Phantom export removed from misuse_resistance gate. No hostile-caller contract changes. -->
<!-- 2026-05-12: SEC-001 fix — ufsecp_musig2_partial_sign_v2 added (Section L). Hostile-caller quartet documented above. -->
