# FFI Hostile-Caller Coverage

**Last updated**: 2026-06-12 | **Version**: 3.22.0

Documents the hostile-caller test coverage for the C ABI (`ufsecp_*` functions). All tests are in `audit/test_adversarial_protocol.cpp`:
- Section G (FFI Hostile-Caller) — original 97-function coverage
- Section H (New ABI Surface Edge Cases) — 26 additional functions added in v3.22+

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
| H.2 | `ufsecp_aead_chacha20_encrypt/decrypt` | NULL guards, bad-tag, wrong-nonce, zero-length roundtrip |
| H.3 | `ufsecp_ecies_encrypt/decrypt` | NULL guards, off-curve pubkey, tampered envelope |
| H.4 | `ufsecp_ellswift_create/xdh` | NULL guards, zero privkey, symmetric shared secret |
| H.5 | `ufsecp_eth_address_checksummed`, `ufsecp_eth_personal_hash` | NULL guards, undersized buffer |
| H.6 | `ufsecp_pedersen_switch_commit` | NULL guards, prefix byte validation |
| H.7 | `ufsecp_schnorr_adaptor_extract` | NULL guards, zero inputs |
| H.8 | `ufsecp_ecdsa_sign_batch`, `ufsecp_schnorr_sign_batch` | NULL ctx/msgs/keys/output, count=0 |
| H.9 | `ufsecp_bip143_sighash`, `ufsecp_bip143_p2wpkh_script_code` | NULL guards, OP_DUP OP_HASH160 PUSH20 format |
| H.10 | `ufsecp_bip144_txid/wtxid/witness_commitment` | NULL guards, determinism |
| H.11 | `ufsecp_is_witness_program`, `ufsecp_parse_witness_program`, `ufsecp_p2wpkh/p2wsh/p2tr_spk`, `ufsecp_witness_script_hash` | NULL guards, format correctness, non-witness rejection |
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

---

## Guarantee

Every `ufsecp_*` function is tested with at least:
1. Valid inputs (FFI round-trip)
2. NULL context (G.1)
3. NULL critical pointers (G.2, G.3)
4. Malformed domain-specific input (G.4-G.20 / H.1-H.12 / I.1-I.5, per function category)

**Mandatory edge-case rule for new ABI functions** (enforced since v3.22):
Every new `ufsecp_*` function MUST be covered by all four checks below before
an audit release commits it to the coverage matrix:
1. NULL rejection for every pointer parameter
2. Zero-count / zero-length / zero-key rejection where the contract requires it
3. Invalid-content rejection (bad prefix, off-curve, truncated, wrong tag, OOB index)
4. A success smoke test with valid inputs

No function can crash, hang, or leak memory on any hostile input. All reject with the appropriate `ufsecp_error_t` and leave output buffers untouched.
