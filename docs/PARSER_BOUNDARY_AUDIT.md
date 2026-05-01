# Parser Boundary Audit

**Last updated**: 2026-03-15 | **Version**: 3.68.0

This document maps every public C ABI entry point to its input parsing behavior: what gets validated, what gets rejected, and with which error code. Every `ufsecp_*` function that accepts external bytes is listed.

---

## Parsing Primitives

| Primitive | Reject zero? | Reject >= bound? | Error code | Used for |
|-----------|-------------|-----------------|------------|----------|
| `scalar_parse_strict_nonzero` | Yes | Yes (>= n) | `ERR_BAD_KEY` | Secret keys, adaptor secrets, nonce scalars |
| `scalar_parse_strict` | No | Yes (>= n) | `ERR_BAD_INPUT` / `ERR_ARITH` | Tweaks, blindings, partial sigs, MSM scalars |
| `FE::parse_bytes_strict` | No | Yes (>= p) | `ERR_BAD_PUBKEY` | x-only pubkeys (32B), Schnorr sig r-component |
| `point_from_compressed` | N/A | Prefix 0x02/0x03, x < p, on-curve, not infinity | `ERR_BAD_PUBKEY` | Compressed pubkeys (33B) |
| `parse_compact_strict` | Yes (r,s) | Yes (>= n) | `ERR_BAD_SIG` | ECDSA compact signatures (64B) |
| `SchnorrSignature::parse_strict` | N/A | r < p, s < n | `ERR_BAD_SIG` | BIP-340 signatures (64B) |
| Raw memcpy | No | No | N/A | Message hashes (32B), aux_rand, seeds |

---

## All Null-Pointer Checks

Every `ufsecp_*` function checks all pointer arguments for NULL before any parsing logic. Returns `UFSECP_ERR_NULL_ARG` immediately. This is enforced by the FFI round-trip test (286 calls) and adversarial protocol tests (G.1-G.20).

---

## Private Key Inputs (32 bytes, `scalar_parse_strict_nonzero`)

Rejects: `== 0` AND `>= n`. Error: `ERR_BAD_KEY`.

| Function | Parameter(s) |
|----------|-------------|
| `ufsecp_seckey_verify` | `privkey[32]` |
| `ufsecp_seckey_negate` | `privkey[32]` |
| `ufsecp_seckey_tweak_add` | `privkey[32]` |
| `ufsecp_seckey_tweak_mul` | `privkey[32]` |
| `ufsecp_pubkey_create` | `privkey[32]` |
| `ufsecp_pubkey_create_uncompressed` | `privkey[32]` |
| `ufsecp_pubkey_xonly` | `privkey[32]` |
| `ufsecp_ecdsa_sign` / `_sign_verified` | `privkey[32]` |
| `ufsecp_ecdsa_sign_recoverable` | `privkey[32]` |
| `ufsecp_schnorr_sign` / `_sign_verified` | `privkey[32]` |
| `ufsecp_ecdh` / `_xonly` / `_raw` | `privkey[32]` |
| `ufsecp_wif_encode` | `privkey[32]` |
| `ufsecp_taproot_tweak_seckey` | `privkey[32]` |
| `ufsecp_musig2_nonce_gen` | `privkey[32]` |
| `ufsecp_musig2_partial_sign` | `privkey[32]` |
| `ufsecp_schnorr_adaptor_sign` | `privkey[32]` |
| `ufsecp_ecdsa_adaptor_sign` | `privkey[32]` |
| `ufsecp_schnorr_adaptor_adapt` | `adaptor_secret[32]` |
| `ufsecp_ecdsa_adaptor_adapt` | `adaptor_secret[32]` |
| `ufsecp_zk_knowledge_prove` | `secret[32]` |
| `ufsecp_zk_dleq_prove` | `secret[32]` |
| `ufsecp_coin_wif_encode` | `privkey[32]` |
| `ufsecp_btc_message_sign` | `privkey[32]` |
| `ufsecp_silent_payment_address` | `scan_privkey[32]`, `spend_privkey[32]` |
| `ufsecp_silent_payment_create_output` | `input_privkeys[i*32]` (each) |
| `ufsecp_silent_payment_scan` | `scan_privkey[32]`, `spend_privkey[32]` |
| `ufsecp_ecies_decrypt` | `privkey[32]` |
| `ufsecp_eth_sign` | `privkey[32]` |

---

## Compressed Public Key Inputs (33 bytes, `point_from_compressed`)

Rejects: bad prefix (not 0x02/0x03), x >= p, not on curve, infinity. Error: `ERR_BAD_PUBKEY`.

| Function | Parameter(s) |
|----------|-------------|
| `ufsecp_pubkey_parse` (33B) | `input[33]` |
| `ufsecp_ecdsa_verify` | `pubkey33[33]` |
| `ufsecp_ecdh` / `_xonly` / `_raw` | `pubkey33[33]` |
| `ufsecp_addr_p2pkh` / `_p2wpkh` | `pubkey33[33]` |
| `ufsecp_pubkey_add` | `a33[33]`, `b33[33]` |
| `ufsecp_pubkey_negate` | `pubkey33[33]` |
| `ufsecp_pubkey_tweak_add` / `_tweak_mul` | `pubkey33[33]` |
| `ufsecp_pubkey_combine` | `pubkeys[i*33]` (each) |
| `ufsecp_shamir_trick` | `P33[33]`, `Q33[33]` |
| `ufsecp_multi_scalar_mul` | `points[i*33]` (each) |
| `ufsecp_coin_address` | `pubkey33[33]` |
| `ufsecp_btc_message_verify` | `pubkey33[33]` |
| `ufsecp_ecies_encrypt` | `recipient_pubkey33[33]` |
| `ufsecp_eth_address` / `_checksummed` | `pubkey33[33]` |
| `ufsecp_silent_payment_create_output` | `scan_pubkey33`, `spend_pubkey33` |
| `ufsecp_silent_payment_scan` | `input_pubkeys33[i*33]` (each) |

Uncompressed (65B): Only via `ufsecp_pubkey_parse(65)` -- prefix 0x04, x < p, y < p, y^2 == x^3+7, not infinity.

---

## X-Only Public Key Inputs (32 bytes, `FE::parse_bytes_strict`)

Rejects: x >= p. Error: `ERR_BAD_PUBKEY`.

| Function | Parameter(s) |
|----------|-------------|
| `ufsecp_schnorr_verify` | `pubkey_x[32]` |
| `ufsecp_schnorr_batch_verify` | each `pubkey_x` |
| `ufsecp_schnorr_batch_identify_invalid` | each `pubkey_x` |
| `ufsecp_schnorr_adaptor_verify` | `pubkey_x[32]` |
| `ufsecp_musig2_key_agg` | `pubkeys[n*32]` (x-only) |
| `ufsecp_taproot_output_key` | `internal_key_x[32]` |
| `ufsecp_taproot_verify` | `output_key_x[32]` |
| `ufsecp_addr_p2tr` | `xonly_pubkey[32]` |

---

## ECDSA Signature Inputs (64 bytes compact, `parse_compact_strict`)

Rejects: r == 0, s == 0, r >= n, s >= n. Error: `ERR_BAD_SIG`.

| Function | Parameter(s) |
|----------|-------------|
| `ufsecp_ecdsa_verify` | `sig64[64]` |
| `ufsecp_ecdsa_sig_to_der` | `sig64[64]` |
| `ufsecp_ecdsa_recover` | `sig64[64]` + recid 0..3 |
| `ufsecp_ecdsa_batch_verify` | each 64-byte sig |
| `ufsecp_ecdsa_batch_identify_invalid` | each 64-byte sig |
| `ufsecp_ecdsa_adaptor_extract` | `sig64[64]` |

---

## DER Signature Input (variable length, strict DER parser)

Validates: length 8-72, starts 0x30, single-byte length encoding (no 0x80 long-form), `seq_len + 2 == der_len`, each INTEGER tag 0x02, no negative high bit, no leading zeros, component <= 32 bytes, no trailing bytes, final r,s via `scalar_parse_strict_nonzero`.

| Function | Parameter(s) |
|----------|-------------|
| `ufsecp_ecdsa_sig_from_der` | `der[der_len]` |

---

## Schnorr/BIP-340 Signature Inputs (64 bytes, `SchnorrSignature::parse_strict`)

Validates: r (bytes 0-31) as field element (reject >= p), s (bytes 32-63) as scalar (reject >= n). Error: `ERR_BAD_SIG`.

| Function | Parameter(s) |
|----------|-------------|
| `ufsecp_schnorr_verify` | `sig64[64]` |
| `ufsecp_schnorr_batch_verify` | each 64-byte sig |
| `ufsecp_schnorr_batch_identify_invalid` | each 64-byte sig |
| `ufsecp_schnorr_adaptor_extract` | `sig64[64]` |

---

## Tweak/Scalar Inputs (32 bytes, `scalar_parse_strict`)

Rejects: >= n. Allows zero (except `*_tweak_mul` which uses `scalar_parse_strict_nonzero`).

| Function | Parameter | Allows zero? |
|----------|-----------|-------------|
| `ufsecp_seckey_tweak_add` | `tweak[32]` | Yes |
| `ufsecp_seckey_tweak_mul` | `tweak[32]` | No (would zero the key) |
| `ufsecp_pubkey_tweak_add` | `tweak[32]` | Yes |
| `ufsecp_pubkey_tweak_mul` | `tweak[32]` | No (would produce infinity) |
| `ufsecp_shamir_trick` | `a[32]`, `b[32]` | Yes |
| `ufsecp_multi_scalar_mul` | `scalars[i*32]` | Yes |

---

## Message Hashes (32 bytes, raw copy -- NO validation)

Treated as opaque 32-byte blobs. Not parsed as scalars, not reduced mod n.

Functions: `ufsecp_ecdsa_sign*`, `ufsecp_ecdsa_verify`, `ufsecp_ecdsa_recover`, `ufsecp_schnorr_sign*`, `ufsecp_schnorr_verify`, `ufsecp_eth_sign`, `ufsecp_eth_ecrecover`

---

## Seeds & Entropy

| Function | Parameter | Validation |
|----------|-----------|-----------|
| `ufsecp_bip32_master` | `seed[seed_len]` | `16 <= seed_len <= 64` |
| `ufsecp_bip39_generate` | `entropy_in[entropy_bytes]` | `entropy_bytes in {16, 20, 24, 28, 32}` |
| `ufsecp_frost_keygen_begin` | `seed[32]` | Raw 32-byte, no scalar check |
| `ufsecp_frost_sign_nonce_gen` | `nonce_seed[32]` | Raw 32-byte, no scalar check |

---

## String Inputs

| Function | Parameter | Validation |
|----------|-----------|-----------|
| `ufsecp_wif_decode` | `wif` (NUL-terminated) | Base58Check decode, prefix, checksum |
| `ufsecp_bip39_validate` | `mnemonic` (NUL-terminated) | Wordlist lookup + checksum |
| `ufsecp_bip39_to_seed` | `mnemonic` (NUL-terminated) | PBKDF2(mnemonic, "mnemonic"+passphrase) |
| `ufsecp_bip39_to_entropy` | `mnemonic` (NUL-terminated) | Reverse wordlist lookup + checksum |
| `ufsecp_bip32_derive_path` | `path` (NUL-terminated, e.g. `"m/44'/0'/0'"`) | Parsed by `bip32_derive_path` |
| `ufsecp_btc_message_verify` | `base64_sig` (NUL-terminated) | Base64 decode (reject invalid) |

---

## ECIES Envelope (variable length)

| Function | Parameter | Validation |
|----------|-----------|-----------|
| `ufsecp_ecies_encrypt` | `plaintext[plaintext_len]` | `plaintext_len > 0`, overflow check (`> SIZE_MAX - 81`) |
| `ufsecp_ecies_decrypt` | `envelope[envelope_len]` | `envelope_len >= 82` (33 ephemeral + 16 IV + 1 min CT + 32 HMAC), ephemeral pubkey decompressed, HMAC verified before decrypt |

---

## Test Coverage

| Parser behavior | Test file | Coverage |
|----------------|-----------|----------|
| All null pointers rejected | `test_adversarial_protocol.cpp` (G.1-G.20) | 97 functions |
| Undersized/truncated buffers | `test_adversarial_protocol.cpp` (G.14-G.18) | Batch, combine, MSM |
| Overlapping in/out buffers | `test_adversarial_protocol.cpp` (G.19) | ECDSA sign |
| Random malformed pubkeys | `test_fuzz_parsers.cpp` | 580K+ iterations |
| Random malformed DER | `test_fuzz_parsers.cpp` | 580K+ iterations |
| Wycheproof boundary sigs | `test_wycheproof_ecdsa.cpp` | 500+ vectors |
| Wycheproof ECDH edge cases | `test_wycheproof_ecdh.cpp` | Infinity, twist, off-curve |
| BIP-340 strict encoding | `test_bip340_strict.cpp` | Official BIP-340 edge vectors |
| Address/WIF/BIP-32/BIP-39 | `test_fuzz_address_bip32_ffi.cpp` | 82K+ iterations |
| FFI round-trip (all 97 funcs) | `test_ffi_round_trip.cpp` | 286 calls covering full API |
