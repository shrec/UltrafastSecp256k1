# FFI Hostile-Caller Coverage

**Last updated**: 2026-03-23 | **Version**: 4.0.0

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
| I.6 | `ufsecp_context_randomize` | NULL ctx rejected; zero-seed (all-zero 32 bytes) accepted — clears/resets blinding; valid random seed succeeds (smoke); signing after randomize produces valid signature |
| I.7 | `ufsecp_ecdsa_sign` with non-NULL `noncefp` | NULL noncefp treated as RFC 6979 default; custom noncefp returning valid scalar succeeds; custom noncefp returning zero scalar triggers retry; output valid after custom nonce |

---

## Section J: GPU C ABI (v3.24+)

`test_gpu_host_api_negative.cpp` and `test_gpu_abi_gate.cpp` cover all 18
`ufsecp_gpu_*` functions without requiring GPU hardware. Both files are integrated
into the unified audit runner (modules `gpu_api_negative` and `gpu_abi_gate`).

| Test File | Checks | Coverage |
|-----------|--------|----------|
| `test_gpu_host_api_negative` | 38 | NULL ctx for all batch ops; NULL ctx_out / info_out; ctx_create with backend 0/99/255; is_available/device_count for invalid backend; count=0 no-ops; NULL buffers + count>0; invalid device index; GPU error strings (7 codes); backend name edge cases (0, 99, 0xFFFFFFFF) |
| `test_gpu_abi_gate` | 30 | Backend count/ids/names (CUDA/OpenCL/Metal/none/invalid); device_info null guard + invalid backend + available device; ctx_create null/invalid/valid lifecycle; ctx_destroy(nullptr) no-crash; last_error/last_error_msg(nullptr); NULL buffer batch ops; error_str(OK/UNAVAILABLE/UNSUPPORTED/999); GPU ops if available (1*G smoke, count=0, NULL-scalar failure); `ufsecp_gpu_is_ready` NULL guard (returns 0) + valid ctx succeeds (smoke, returns 1) |

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

<!-- 2026-04-28: ufsecp_gpu.h docstring corrected — ufsecp_gpu_context_create → ufsecp_gpu_ctx_create. Phantom export removed from misuse_resistance gate. No hostile-caller contract changes. -->
