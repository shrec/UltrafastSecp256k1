# Feature Assurance Ledger -- UltrafastSecp256k1

**Generated:** 2026-03-25
**Scope:** All `UFSECP_API` exported functions + internal library capabilities
**Total API functions:** 158

## Legend

| Symbol | Meaning |
|--------|---------|
| Y | Yes -- fully covered |
| P | Partial -- some coverage, not exhaustive |
| - | No coverage |
| N/A | Not applicable for this category |

---

## 1. Context Management (9 functions)

| Function | Unit Test | Fuzz | Adversarial | Differential | CT Path | GPU | Ext. Vectors | Zeroization |
|----------|-----------|------|-------------|--------------|---------|-----|-------------|-------------|
| `ufsecp_abi_version` | Y | - | - | N/A | N/A | - | N/A | N/A |
| `ufsecp_version` | Y | - | - | N/A | N/A | - | N/A | N/A |
| `ufsecp_version_string` | Y | - | - | N/A | N/A | - | N/A | N/A |
| `ufsecp_ctx_create` | Y | Y | Y (null) | N/A | N/A | - | N/A | N/A |
| `ufsecp_ctx_clone` | Y | - | Y (null) | N/A | N/A | - | N/A | N/A |
| `ufsecp_ctx_destroy` | Y | Y | Y (null-safe) | N/A | N/A | - | N/A | N/A |
| `ufsecp_last_error` | Y | - | - | N/A | N/A | - | N/A | N/A |
| `ufsecp_last_error_msg` | Y | - | - | N/A | N/A | - | N/A | N/A |
| `ufsecp_ctx_size` | Y | - | - | N/A | N/A | - | N/A | N/A |
| `ufsecp_ctx_clone` | Y | - | Y (null) | N/A | N/A | - | N/A | N/A |
| `ufsecp_ctx_destroy` | Y | Y | Y (null-safe) | N/A | N/A | - | N/A | N/A |
| `ufsecp_last_error` | Y | - | - | N/A | N/A | - | N/A | N/A |
| `ufsecp_last_error_msg` | Y | - | - | N/A | N/A | - | N/A | N/A |
| `ufsecp_ctx_size` | Y | - | - | N/A | N/A | - | N/A | N/A |

**Test files:** `audit/test_ffi_round_trip.cpp`, `audit/test_adversarial_protocol.cpp`

---

## 2. Private Key Utilities (4 functions)

| Function | Unit Test | Fuzz | Adversarial | Differential | CT Path | GPU | Ext. Vectors | Zeroization |
|----------|-----------|------|-------------|--------------|---------|-----|-------------|-------------|
| `ufsecp_seckey_verify` | Y | Y | Y | Y | Y (CT scalar) | - | Y (Wycheproof) | N/A |
| `ufsecp_seckey_negate` | Y | Y | Y | Y | Y | - | - | Y |
| `ufsecp_seckey_tweak_add` | Y | Y | Y | Y | Y | - | - | Y |
| `ufsecp_seckey_tweak_mul` | Y | Y | Y | Y | Y | - | - | Y |

**Test files:** `audit/test_ffi_round_trip.cpp`, `audit/test_adversarial_protocol.cpp`, `audit/audit_fuzz.cpp`
**CT:** All secret-key ops wired through CT layer in `ufsecp_impl.cpp`
**Zeroization:** `secure_erase` in `ufsecp_impl.cpp`

---

## 3. Public Key (4 functions)

| Function | Unit Test | Fuzz | Adversarial | Differential | CT Path | GPU | Ext. Vectors | Zeroization |
|----------|-----------|------|-------------|--------------|---------|-----|-------------|-------------|
| `ufsecp_pubkey_create` | Y | Y | Y | Y | Y (k*G via CT) | Y (CUDA/OCL/Metal) | Y | N/A |
| `ufsecp_pubkey_create_uncompressed` | Y | Y | - | Y | Y | Y | - | N/A |
| `ufsecp_pubkey_parse` | Y | Y | Y (malformed) | - | N/A (public) | - | Y (Wycheproof) | N/A |
| `ufsecp_pubkey_xonly` | Y | Y | Y | Y | Y | - | Y (BIP-340) | N/A |

**Test files:** `audit/test_ffi_round_trip.cpp`, `audit/audit_fuzz.cpp`, `audit/differential_test.cpp`
**GPU:** Scalar multiplication (k*G) on CUDA (`secp256k1.cuh`), OpenCL (`secp256k1_point.cl`), Metal (`secp256k1_point.h`)

---

## 4. ECDSA (8 functions)

| Function | Unit Test | Fuzz | Adversarial | Differential | CT Path | GPU | Ext. Vectors | Zeroization |
|----------|-----------|------|-------------|--------------|---------|-----|-------------|-------------|
| `ufsecp_ecdsa_sign` | Y | Y | Y | Y | Y (CT sign) | Y (CUDA) | Y (RFC 6979) | Y |
| `ufsecp_ecdsa_sign_verified` | Y | Y | - | - | Y (CT sign + verify) | - | - | Y |
| `ufsecp_ecdsa_verify` | Y | Y | Y (r=0,s=0,>=n) | Y | N/A (public) | Y (CUDA) | Y (Wycheproof) | N/A |
| `ufsecp_ecdsa_sig_to_der` | Y | Y | Y | - | N/A | - | - | N/A |
| `ufsecp_ecdsa_sig_from_der` | Y | Y | Y (malformed DER) | - | N/A | - | Y (Wycheproof) | N/A |
| `ufsecp_ecdsa_sign_recoverable` | Y | Y | Y (edge recids) | Y | Y (CT sign) | Y (CUDA `recovery.cuh`) | - | Y |
| `ufsecp_ecdsa_recover` | Y | Y | Y (recid=4, wrong) | Y | N/A (public) | Y (CUDA `recovery.cuh`) | - | N/A |

**Test files:**
- Unit: `audit/test_ffi_round_trip.cpp`, `audit/differential_test.cpp`
- Fuzz: `audit/audit_fuzz.cpp` (malformed pubkeys, invalid sigs, DER round-trip, normalization)
- Adversarial: `audit/test_adversarial_protocol.cpp`
- Wycheproof: `audit/test_wycheproof_ecdsa.cpp` (r=0,s=0,r>=n,s>=n, bit-flips, boundary values, wrong key/msg, infinity pk, High-S)
- Fault injection: `audit/test_fault_injection.cpp` (signature bit-flip, message bit-flip)
- CT sidechannel: `audit/test_ct_sidechannel.cpp` (dudect timing on ECDSA sign)
- Batch randomness: `audit/test_batch_randomness.cpp`
- Cross-lib: `audit/test_cross_libsecp256k1.cpp`

**CT:** Signing via `ct_sign.cpp` with `secure_erase` of nonce/key/intermediate. Verification uses fast path (public data).
**GPU:** CUDA sign+verify (`cuda/include/ecdsa.cuh`), batch verify (`cuda/include/batch_verify.cuh`), recovery (`cuda/include/recovery.cuh`)
**Zeroization:** 10 `secure_erase` calls in `ct_sign.cpp` covering nonce (k), private key bytes, challenge hash, aux rand XOR

---

## 5. Schnorr / BIP-340 (3 functions)

| Function | Unit Test | Fuzz | Adversarial | Differential | CT Path | GPU | Ext. Vectors | Zeroization |
|----------|-----------|------|-------------|--------------|---------|-----|-------------|-------------|
| `ufsecp_schnorr_sign` | Y | Y | Y | Y | Y (CT sign) | Y (CUDA) | Y (BIP-340) | Y |
| `ufsecp_schnorr_sign_verified` | Y | - | - | - | Y (CT sign + verify) | - | - | Y |
| `ufsecp_schnorr_verify` | Y | Y | Y (zero pk, wrong msg) | Y | N/A (public) | Y (CUDA) | Y (BIP-340) | N/A |

**Test files:**
- Unit: `audit/test_ffi_round_trip.cpp`, `audit/differential_test.cpp`
- Fuzz: `audit/audit_fuzz.cpp` (corrupted r, zero pk, wrong msg, byte round-trip)
- BIP-340 vectors: `audit/differential_test.cpp` (test_bip340_vectors)
- Fault injection: `audit/test_fault_injection.cpp` (schnorr sig bit-flip)
- CT sidechannel: `audit/test_ct_sidechannel.cpp` (dudect timing on Schnorr sign)

**GPU:** CUDA sign+verify (`cuda/include/schnorr.cuh`) with BIP-340 midstate optimization. Batch verify (`cuda/include/batch_verify.cuh`).

---

## 6. ECDH (3 functions)

| Function | Unit Test | Fuzz | Adversarial | Differential | CT Path | GPU | Ext. Vectors | Zeroization |
|----------|-----------|------|-------------|--------------|---------|-----|-------------|-------------|
| `ufsecp_ecdh` | Y | - | Y (infinity, off-curve, zero key) | Y | Y (CT scalar_mul) | Y (CUDA/OCL/Metal) | Y (Wycheproof) | Y |
| `ufsecp_ecdh_xonly` | Y | - | Y | Y | Y | Y | Y (Wycheproof) | Y |
| `ufsecp_ecdh_raw` | Y | - | Y | Y | Y | Y | Y (Wycheproof) | Y |

**Test files:**
- Wycheproof: `audit/test_wycheproof_ecdh.cpp` (infinity, off-curve, twist attack, zero key, commutativity, point validation, variant consistency)
- Adversarial: `audit/test_adversarial_protocol.cpp`
- FFI: `audit/test_ffi_round_trip.cpp`

**GPU:** CUDA (`cuda/include/ecdh.cuh`), OpenCL (`opencl/kernels/secp256k1_ecdh.cl`), Metal (`metal/shaders/secp256k1_ecdh.h`)
**Zeroization:** `secure_erase` in `cpu/src/ecdh.cpp`

---

## 7. Hashing (4 functions)

| Function | Unit Test | Fuzz | Adversarial | Differential | CT Path | GPU | Ext. Vectors | Zeroization |
|----------|-----------|------|-------------|--------------|---------|-----|-------------|-------------|
| `ufsecp_sha256` | Y | - | - | - | N/A (deterministic) | Y (all backends) | Y (NIST) | N/A |
| `ufsecp_sha512` | Y | - | - | - | N/A | - | - | N/A |
| `ufsecp_hash160` | Y | - | - | - | N/A | Y (all backends) | - | N/A |
| `ufsecp_tagged_hash` | Y | - | - | - | N/A | Y (CUDA midstate) | Y (BIP-340) | N/A |

**GPU:** SHA-256 on CUDA/OCL/Metal, Hash160 on all backends (`hash160.cuh`, `secp256k1_hash160.cl`, `secp256k1_hash160.h`). Keccak-256 on all backends.

---

## 8. Bitcoin Addresses (3 functions)

| Function | Unit Test | Fuzz | Adversarial | Differential | CT Path | GPU | Ext. Vectors | Zeroization |
|----------|-----------|------|-------------|--------------|---------|-----|-------------|-------------|
| `ufsecp_addr_p2pkh` | Y | Y | Y | - | N/A (public) | - | - | N/A |
| `ufsecp_addr_p2wpkh` | Y | Y | Y | - | N/A | - | - | N/A |
| `ufsecp_addr_p2tr` | Y | Y | Y | - | N/A | - | Y (BIP-341) | N/A |

**Test files:** `audit/test_ffi_round_trip.cpp`, `audit/test_fuzz_address_bip32_ffi.cpp`, `audit/test_adversarial_protocol.cpp`

---

## 9. WIF (2 functions)

| Function | Unit Test | Fuzz | Adversarial | Differential | CT Path | GPU | Ext. Vectors | Zeroization |
|----------|-----------|------|-------------|--------------|---------|-----|-------------|-------------|
| `ufsecp_wif_encode` | Y | Y | Y | - | N/A | - | - | N/A |
| `ufsecp_wif_decode` | Y | Y | Y | - | N/A | - | - | N/A |

**Test files:** `audit/test_ffi_round_trip.cpp`, `audit/test_fuzz_address_bip32_ffi.cpp`, `audit/test_adversarial_protocol.cpp`

---

## 10. BIP-32 HD Key Derivation (5 functions)

| Function | Unit Test | Fuzz | Adversarial | Differential | CT Path | GPU | Ext. Vectors | Zeroization |
|----------|-----------|------|-------------|--------------|---------|-----|-------------|-------------|
| `ufsecp_bip32_master` | Y | Y | Y (bad seed) | - | Y | Y (CUDA/OCL/Metal) | Y (BIP-32) | Y |
| `ufsecp_bip32_derive` | Y | Y | Y (depth overflow) | - | Y | Y | Y (BIP-32) | Y |
| `ufsecp_bip32_derive_path` | Y | Y | Y (bad path) | - | Y | Y | Y (BIP-32) | Y |
| `ufsecp_bip32_privkey` | Y | Y | Y (xpub rejection) | - | N/A | - | - | N/A |
| `ufsecp_bip32_pubkey` | Y | Y | - | - | N/A (public) | - | - | N/A |

**Test files:** `audit/test_ffi_round_trip.cpp`, `audit/test_adversarial_protocol.cpp`, `audit/test_fuzz_address_bip32_ffi.cpp`
**GPU:** CUDA (`cuda/include/bip32.cuh`), OpenCL (`opencl/kernels/secp256k1_bip32.cl`), Metal (`metal/shaders/secp256k1_bip32.h`)

---

## 11. Taproot / BIP-341 (3 functions)

| Function | Unit Test | Fuzz | Adversarial | Differential | CT Path | GPU | Ext. Vectors | Zeroization |
|----------|-----------|------|-------------|--------------|---------|-----|-------------|-------------|
| `ufsecp_taproot_output_key` | Y | - | Y | - | Y (tweak via CT) | - | Y (BIP-341) | N/A |
| `ufsecp_taproot_tweak_seckey` | Y | - | Y | - | Y | - | - | Y |
| `ufsecp_taproot_verify` | Y | - | Y | - | N/A (public) | - | Y (BIP-341) | N/A |

**Test files:** `audit/test_ffi_round_trip.cpp`, `audit/test_adversarial_protocol.cpp`

---

## 12. Public Key Arithmetic (5 functions)

| Function | Unit Test | Fuzz | Adversarial | Differential | CT Path | GPU | Ext. Vectors | Zeroization |
|----------|-----------|------|-------------|--------------|---------|-----|-------------|-------------|
| `ufsecp_pubkey_add` | Y | - | Y | Y | N/A (public) | Y (all backends) | - | N/A |
| `ufsecp_pubkey_negate` | Y | - | Y | Y | N/A | - | - | N/A |
| `ufsecp_pubkey_tweak_add` | Y | - | Y | - | N/A | - | - | N/A |
| `ufsecp_pubkey_tweak_mul` | Y | - | Y | - | Y (CT scalar_mul) | - | - | N/A |
| `ufsecp_pubkey_combine` | Y | - | Y | - | N/A | - | - | N/A |

**Test files:** `audit/test_ffi_round_trip.cpp`, `audit/test_adversarial_protocol.cpp`, `audit/differential_test.cpp`

---

## 13. BIP-39 Mnemonic (4 functions)

| Function | Unit Test | Fuzz | Adversarial | Differential | CT Path | GPU | Ext. Vectors | Zeroization |
|----------|-----------|------|-------------|--------------|---------|-----|-------------|-------------|
| `ufsecp_bip39_generate` | Y | Y | Y | - | N/A | - | Y (BIP-39) | Y |
| `ufsecp_bip39_validate` | Y | Y | Y | - | N/A | - | Y (BIP-39) | N/A |
| `ufsecp_bip39_to_seed` | Y | Y | Y | - | N/A | - | Y (BIP-39) | Y |
| `ufsecp_bip39_to_entropy` | Y | Y | Y | - | N/A | - | Y (BIP-39) | Y |

**Test files:** `audit/test_ffi_round_trip.cpp`, `audit/test_adversarial_protocol.cpp`, `audit/test_fuzz_address_bip32_ffi.cpp`

---

## 14. Batch Verification (4 functions)

| Function | Unit Test | Fuzz | Adversarial | Differential | CT Path | GPU | Ext. Vectors | Zeroization |
|----------|-----------|------|-------------|--------------|---------|-----|-------------|-------------|
| `ufsecp_schnorr_batch_verify` | Y | - | Y | - | N/A (public) | Y (CUDA) | - | N/A |
| `ufsecp_ecdsa_batch_verify` | Y | - | Y | - | N/A | Y (CUDA) | - | N/A |
| `ufsecp_schnorr_batch_identify_invalid` | Y | - | Y | - | N/A | Y (CUDA) | - | N/A |
| `ufsecp_ecdsa_batch_identify_invalid` | Y | - | Y | - | N/A | Y (CUDA) | - | N/A |

**Test files:** `audit/test_adversarial_protocol.cpp`, `audit/test_batch_randomness.cpp`
**GPU:** CUDA batch verify kernels (`cuda/include/batch_verify.cuh`) -- parallel per-thread verification

---

## 15. Multi-Scalar Multiplication (2 functions)

| Function | Unit Test | Fuzz | Adversarial | Differential | CT Path | GPU | Ext. Vectors | Zeroization |
|----------|-----------|------|-------------|--------------|---------|-----|-------------|-------------|
| `ufsecp_shamir_trick` | Y | - | Y | - | N/A (public) | Y (CUDA/OCL/Metal) | - | N/A |
| `ufsecp_multi_scalar_mul` | Y | - | Y | - | N/A | Y (CUDA/OCL/Metal) | - | N/A |

**Test files:** `audit/test_ffi_round_trip.cpp`, `audit/test_adversarial_protocol.cpp`
**GPU:** MSM (Pippenger) on CUDA (`cuda/include/msm.cuh`), OpenCL (`opencl/kernels/secp256k1_msm.cl`), Metal (`metal/shaders/secp256k1_msm.h`)

---

## 16. MuSig2 / BIP-327 (7 functions)

| Function | Unit Test | Fuzz | Adversarial | Differential | CT Path | GPU | Ext. Vectors | Zeroization |
|----------|-----------|------|-------------|--------------|---------|-----|-------------|-------------|
| `ufsecp_musig2_key_agg` | Y | - | Y (null, n=0) | - | N/A (public agg) | - | Y (BIP-327) | N/A |
| `ufsecp_musig2_nonce_gen` | Y | - | Y (null ctx) | - | Y (CT nonce gen) | - | Y (BIP-327) | Y |
| `ufsecp_musig2_nonce_agg` | Y | - | Y (null ctx) | - | N/A | - | - | N/A |
| `ufsecp_musig2_start_sign_session` | Y | - | Y (null ctx) | - | N/A | - | - | N/A |
| `ufsecp_musig2_partial_sign` | Y | - | Y (nonce reuse) | - | Y (CT sign) | - | Y (BIP-327) | Y (nonce consumed) |
| `ufsecp_musig2_partial_verify` | Y | - | Y (cross-session replay) | - | N/A (public) | - | Y (BIP-327) | N/A |
| `ufsecp_musig2_partial_sig_agg` | Y | - | Y (replayed partial) | - | N/A | - | Y (BIP-327) | N/A |

**Test files:**
- Unit: `audit/test_musig2_frost.cpp`, `audit/test_ffi_round_trip.cpp`
- Advanced: `audit/test_musig2_frost_advanced.cpp`
- Adversarial: `audit/test_adversarial_protocol.cpp` (nonce reuse, partial sig replay, hostile null/junk args)
- External vectors: `audit/test_musig2_bip327_vectors.cpp` (BIP-327 official test vectors)
- CT sidechannel: `audit/test_ct_sidechannel.cpp` (MuSig2 timing)

**Nonce safety:** `secnonce` is zeroed after `partial_sign` to prevent reuse. Second call fails.
**Zeroization:** `secure_erase` in `cpu/src/musig2.cpp`

---

## 17. FROST Threshold Signatures (6 functions)

| Function | Unit Test | Fuzz | Adversarial | Differential | CT Path | GPU | Ext. Vectors | Zeroization |
|----------|-----------|------|-------------|--------------|---------|-----|-------------|-------------|
| `ufsecp_frost_keygen_begin` | Y | - | Y | - | Y | - | Y (FROST KAT) | Y |
| `ufsecp_frost_keygen_finalize` | Y | - | Y | - | Y | - | Y (FROST KAT) | Y |
| `ufsecp_frost_sign_nonce_gen` | Y | - | Y | - | Y | - | - | Y |
| `ufsecp_frost_sign` | Y | - | Y (below threshold) | - | Y (CT sign) | - | Y (FROST KAT) | Y |
| `ufsecp_frost_verify_partial` | Y | - | Y (malformed commit) | - | N/A (public) | - | Y (FROST KAT) | N/A |
| `ufsecp_frost_aggregate` | Y | - | Y (below threshold) | - | N/A | - | Y (FROST KAT) | N/A |

**Test files:**
- Unit: `audit/test_musig2_frost.cpp`, `audit/test_ffi_round_trip.cpp`
- Advanced: `audit/test_musig2_frost_advanced.cpp`
- Adversarial: `audit/test_adversarial_protocol.cpp` (below-threshold, malformed commitment)
- KAT: `audit/test_frost_kat.cpp`
- CT sidechannel: `audit/test_ct_sidechannel.cpp` (FROST timing)

---

## 18. Adaptor Signatures (8 functions)

### Schnorr Adaptor (4 functions)

| Function | Unit Test | Fuzz | Adversarial | Differential | CT Path | GPU | Ext. Vectors | Zeroization |
|----------|-----------|------|-------------|--------------|---------|-----|-------------|-------------|
| `ufsecp_schnorr_adaptor_sign` | Y | - | Y (invalid/wrong point) | - | Y | - | - | Y |
| `ufsecp_schnorr_adaptor_verify` | Y | - | Y | - | N/A (public) | - | - | N/A |
| `ufsecp_schnorr_adaptor_adapt` | Y | - | Y | - | N/A | - | - | N/A |
| `ufsecp_schnorr_adaptor_extract` | Y | - | Y (transcript check) | - | N/A | - | - | N/A |

### ECDSA Adaptor (4 functions)

| Function | Unit Test | Fuzz | Adversarial | Differential | CT Path | GPU | Ext. Vectors | Zeroization |
|----------|-----------|------|-------------|--------------|---------|-----|-------------|-------------|
| `ufsecp_ecdsa_adaptor_sign` | Y | - | Y (full round-trip) | - | Y | - | - | Y |
| `ufsecp_ecdsa_adaptor_verify` | Y | - | Y | - | N/A (public) | - | - | N/A |
| `ufsecp_ecdsa_adaptor_adapt` | Y | - | Y | - | N/A | - | - | N/A |
| `ufsecp_ecdsa_adaptor_extract` | Y | - | Y | - | N/A | - | - | N/A |

**Test files:** `audit/test_adversarial_protocol.cpp` (ECDSA adaptor full round-trip, Schnorr adaptor adversarial), `audit/test_ffi_round_trip.cpp`

---

## 19. Pedersen Commitments (5 functions)

| Function | Unit Test | Fuzz | Adversarial | Differential | CT Path | GPU | Ext. Vectors | Zeroization |
|----------|-----------|------|-------------|--------------|---------|-----|-------------|-------------|
| `ufsecp_pedersen_commit` | Y | - | Y | - | Y | Y (CUDA/OCL/Metal) | - | N/A |
| `ufsecp_pedersen_verify` | Y | - | Y | - | N/A (public) | Y | - | N/A |
| `ufsecp_pedersen_verify_sum` | Y | - | Y | - | N/A | Y | - | N/A |
| `ufsecp_pedersen_blind_sum` | Y | - | Y | - | Y | - | - | Y |
| `ufsecp_pedersen_switch_commit` | Y | - | - | - | Y | - | - | N/A |

**Test files:** `audit/test_ffi_round_trip.cpp`, `audit/test_adversarial_protocol.cpp`
**GPU:** CUDA (`cuda/include/pedersen.cuh`), OpenCL (`opencl/kernels/secp256k1_pedersen.cl`), Metal (`metal/shaders/secp256k1_pedersen.h`)

---

## 20. Zero-Knowledge Proofs (6 functions)

| Function | Unit Test | Fuzz | Adversarial | Differential | CT Path | GPU | Ext. Vectors | Zeroization |
|----------|-----------|------|-------------|--------------|---------|-----|-------------|-------------|
| `ufsecp_zk_knowledge_prove` | Y | - | Y | - | Y | Y (CUDA/OCL/Metal) | - | Y |
| `ufsecp_zk_knowledge_verify` | Y | - | Y | - | N/A (public) | Y | - | N/A |
| `ufsecp_zk_dleq_prove` | Y | - | Y | - | Y | Y | - | Y |
| `ufsecp_zk_dleq_verify` | Y | - | Y | - | N/A | Y | - | N/A |
| `ufsecp_zk_range_prove` | Y | - | - | - | Y | Y | - | Y |
| `ufsecp_zk_range_verify` | Y | - | - | - | N/A | Y | - | N/A |

**Test files:** `audit/test_ffi_round_trip.cpp`, `audit/test_adversarial_protocol.cpp`
**GPU:** CUDA (`cuda/include/zk.cuh`), OpenCL (`opencl/kernels/secp256k1_zk.cl`, `secp256k1_ct_zk.cl`), Metal (`metal/shaders/secp256k1_zk.h`, `secp256k1_ct_zk.h`)
**Bulletproof generator table:** CUDA (`bp_gen_table.cuh`), OpenCL (`secp256k1_bp_gen_table.cl`), Metal (`secp256k1_bp_gen_table.h`)

---

## 21. Multi-Coin Wallet (3 functions)

| Function | Unit Test | Fuzz | Adversarial | Differential | CT Path | GPU | Ext. Vectors | Zeroization |
|----------|-----------|------|-------------|--------------|---------|-----|-------------|-------------|
| `ufsecp_coin_address` | Y | Y | Y | - | N/A (public) | - | - | N/A |
| `ufsecp_coin_derive_from_seed` | Y | Y | Y | - | Y (secret derivation) | - | - | Y |
| `ufsecp_coin_wif_encode` | Y | Y | Y | - | N/A | - | - | N/A |

**Supported coins:** Bitcoin (0), Litecoin (2), Dogecoin (3), Dash (5), Ethereum (60), Bitcoin Cash (145), Tron (195)
**Test files:** `audit/test_ffi_round_trip.cpp`, `audit/test_adversarial_protocol.cpp`, `audit/test_fuzz_address_bip32_ffi.cpp`

---

## 22. Bitcoin Message Signing (3 functions)

| Function | Unit Test | Fuzz | Adversarial | Differential | CT Path | GPU | Ext. Vectors | Zeroization |
|----------|-----------|------|-------------|--------------|---------|-----|-------------|-------------|
| `ufsecp_btc_message_sign` | Y | - | Y | - | Y (CT sign) | - | - | Y |
| `ufsecp_btc_message_verify` | Y | - | Y | - | N/A (public) | - | - | N/A |
| `ufsecp_btc_message_hash` | Y | - | Y | - | N/A | - | - | N/A |

**Test files:** `audit/test_ffi_round_trip.cpp`, `audit/test_adversarial_protocol.cpp`

---

## 23. BIP-352 Silent Payments (3 functions)

| Function | Unit Test | Fuzz | Adversarial | Differential | CT Path | GPU | Ext. Vectors | Zeroization |
|----------|-----------|------|-------------|--------------|---------|-----|-------------|-------------|
| `ufsecp_silent_payment_address` | Y | - | Y (wrong ordering, dup keys) | - | Y | - | - | Y |
| `ufsecp_silent_payment_create_output` | Y | - | Y (bad keys) | - | Y (CT scalar_mul) | - | - | Y |
| `ufsecp_silent_payment_scan` | Y | - | Y | - | Y | - | - | Y |

**Test files:** `audit/test_ffi_round_trip.cpp`, `audit/test_adversarial_protocol.cpp`

---

## 24. ECIES Encryption (2 functions)

| Function | Unit Test | Fuzz | Adversarial | Differential | CT Path | GPU | Ext. Vectors | Zeroization |
|----------|-----------|------|-------------|--------------|---------|-----|-------------|-------------|
| `ufsecp_ecies_encrypt` | Y | Y | Y | - | Y (CT ECDH) | - | - | Y |
| `ufsecp_ecies_decrypt` | Y | Y | Y | - | Y (CT scalar_mul) | - | - | Y |

**Test files:** `audit/test_ffi_round_trip.cpp`, `audit/test_ecies_regression.cpp`
**Regression suite (85 tests):**
- (A) Parity tamper: flip 0x02/0x03 on ephemeral pubkey -> decrypt must fail
- (B) Invalid prefix: bad prefixes 0x00, 0x04, 0xFF -> decrypt must fail
- (C) Truncated envelope: 6 truncated sizes (0, 1, 32, 33, 49, 81 bytes) -> clean failure
- (D) Tamper matrix: flip 1 bit in each field (ephemeral pubkey, IV, ciphertext, HMAC tag)
- (E) Round-trip KAT: 3 plaintext sizes (1, 13, 32 bytes), envelope structure, wrong-key rejection
- (F) ABI prefix rejection: 6 bad prefixes x 5 ABI endpoints = 30 checks
- (G) Pubkey parser consistency: 3 malformed x-coords -> consistent `BAD_PUBKEY` across `pubkey_parse`, `ecdh`, `ecies_encrypt`
- (H) RNG fail-closed: fork + seccomp blocks `getrandom` -> process must SIGABRT (Linux x86-64 only)
**Zeroization:** Extensive -- 14+ `secure_erase` calls in `cpu/src/ecies.cpp` covering ephemeral key, shared secret, KDF output, AES keystream, HMAC pads

---

## 25. Ethereum (6 functions, conditional: `SECP256K1_BUILD_ETHEREUM`)

| Function | Unit Test | Fuzz | Adversarial | Differential | CT Path | GPU | Ext. Vectors | Zeroization |
|----------|-----------|------|-------------|--------------|---------|-----|-------------|-------------|
| `ufsecp_keccak256` | Y | - | - | - | N/A | Y (CUDA/OCL/Metal) | - | N/A |
| `ufsecp_eth_address` | Y | - | Y | - | N/A (public) | Y (CUDA/OCL/Metal) | - | N/A |
| `ufsecp_eth_address_checksummed` | Y | - | Y | - | N/A | Y (CUDA/OCL/Metal) | - | N/A |
| `ufsecp_eth_personal_hash` | Y | - | - | - | N/A | - | - | N/A |
| `ufsecp_eth_sign` | Y | - | Y | - | Y (CT sign) | - | - | Y |
| `ufsecp_eth_ecrecover` | Y | - | Y | - | N/A (public) | - | - | N/A |

**Test files:** `audit/test_ffi_round_trip.cpp`, `audit/test_adversarial_protocol.cpp`
**GPU:** Keccak-256 on all 3 backends (`keccak256.cuh`, `secp256k1_keccak256.cl`, `secp256k1_keccak256.h`). ETH address derivation on all backends. EIP-55 checksum on OpenCL/Metal.

---

## GPU Operation Matrix

| Operation | CUDA | OpenCL | Metal |
|-----------|------|--------|-------|
| Field arithmetic (mul, sqr, inv, add, sub) | Y | Y | Y |
| Scalar arithmetic | Y | Y | Y |
| Point arithmetic (add, dbl, mixed add) | Y | Y | Y |
| Scalar multiplication (k*G, k*P) | Y | Y | Y |
| GLV endomorphism | Y | Y | Y |
| Generator table (w8 precomp) | Y | Y | Y |
| MSM / Pippenger | Y | Y | Y |
| Batch Montgomery inversion | Y | Y | Y |
| Batch Jacobian-to-affine | Y | Y | Y |
| Affine batch add | Y | Y | Y |
| ECDSA sign (RFC 6979) | Y | Y | Y |
| ECDSA verify | Y | Y | Y |
| ECDSA recovery | Y | Y | Y |
| Schnorr sign (BIP-340) | Y | Y | Y |
| Schnorr verify (BIP-340) | Y | Y | Y |
| BIP-340 midstate optimization | Y | Y | Y |
| Batch verify (ECDSA + Schnorr) | Y | - | Y |
| SHA-256 | Y | Y | Y |
| Hash160 (RIPEMD160(SHA256)) | Y | Y | Y |
| Keccak-256 (Ethereum) | Y | Y | Y |
| ETH address + EIP-55 | Y | Y | Y |
| BIP-32 HD derivation | Y | Y | Y |
| ECDH (x-only + raw) | Y | Y | Y |
| Pedersen commitment | Y | Y | Y |
| ZK proofs (knowledge, DLEQ) | Y | Y | Y |
| Bulletproof range proof verify | Y | Y | Y |
| Bulletproof generator table | Y | Y | Y |
| CT field ops | Y | Y | Y |
| CT scalar ops | Y | Y | Y |
| CT point ops | Y | Y | Y |
| CT sign (ECDSA + Schnorr) | Y | Y | Y |
| CT ZK proofs | Y | Y | Y |
| Bloom filter lookup | Y | Y | Y |

---

## 26. AEAD / ChaCha20-Poly1305 (2 functions, conditional: `SECP256K1_BUILD_BIP324`)

| Function | Unit Test | Fuzz | Adversarial | Differential | CT Path | GPU | Ext. Vectors | Zeroization |
|----------|-----------|------|-------------|--------------|---------|-----|-------------|-------------|
| `ufsecp_aead_chacha20_poly1305_encrypt` | Y | - | Y | - | N/A | - | N/A | N/A |
| `ufsecp_aead_chacha20_poly1305_decrypt` | Y | - | Y | - | N/A | - | N/A | N/A |

**Test files:** `tests/test_ffi_coverage.cpp`, `audit/test_adversarial_protocol.cpp`
**Implementation:** `ufsecp/ufsecp_impl.cpp`, `src/chacha20_poly1305.cpp`

---

## 27. BIP-324 / EllSwift (7 functions, conditional: `SECP256K1_BUILD_BIP324`)

| Function | Unit Test | Fuzz | Adversarial | Differential | CT Path | GPU | Ext. Vectors | Zeroization |
|----------|-----------|------|-------------|--------------|---------|-----|-------------|-------------|
| `ufsecp_bip324_create` | Y | - | Y | - | Y | - | N/A | Y |
| `ufsecp_bip324_destroy` | Y | - | Y (null-safe) | - | Y | - | N/A | Y |
| `ufsecp_bip324_handshake` | Y | - | Y | - | Y | - | N/A | Y |
| `ufsecp_bip324_encrypt` | Y | - | Y | - | Y | - | N/A | Y |
| `ufsecp_bip324_decrypt` | Y | - | Y | - | Y | - | N/A | Y |
| `ufsecp_ellswift_create` | P | - | Y | - | Y | - | N/A | Y |
| `ufsecp_ellswift_xdh` | P | - | Y | - | Y | - | N/A | Y |

**Test files:** `tests/test_ffi_coverage.cpp`, `audit/test_adversarial_protocol.cpp`, `audit/test_exploit_bip324_transcript_splice.cpp`, `audit/test_exploit_bip324_counter_desync.cpp`
**Implementation:** `ufsecp/ufsecp_impl.cpp`, `include/secp256k1/bip324.hpp`

---

## 28. Bitcoin Serialization / SegWit / Sighash (13 functions)

| Function | Unit Test | Fuzz | Adversarial | Differential | CT Path | GPU | Ext. Vectors | Zeroization |
|----------|-----------|------|-------------|--------------|---------|-----|-------------|-------------|
| `ufsecp_bip143_p2wpkh_script_code` | Y | - | Y | - | N/A | - | N/A | N/A |
| `ufsecp_bip143_sighash` | Y | - | Y | - | N/A | - | N/A | N/A |
| `ufsecp_bip144_txid` | Y | - | Y | - | N/A | - | N/A | N/A |
| `ufsecp_bip144_witness_commitment` | Y | - | Y | - | N/A | - | N/A | N/A |
| `ufsecp_bip144_wtxid` | Y | - | Y | - | N/A | - | N/A | N/A |
| `ufsecp_segwit_is_witness_program` | Y | - | Y | - | N/A | - | N/A | N/A |
| `ufsecp_segwit_p2tr_spk` | Y | - | Y | - | N/A | - | N/A | N/A |
| `ufsecp_segwit_p2wpkh_spk` | Y | - | Y | - | N/A | - | N/A | N/A |
| `ufsecp_segwit_p2wsh_spk` | Y | - | Y | - | N/A | - | N/A | N/A |
| `ufsecp_segwit_parse_program` | Y | - | Y | - | N/A | - | N/A | N/A |
| `ufsecp_segwit_witness_script_hash` | Y | - | Y | - | N/A | - | N/A | N/A |
| `ufsecp_taproot_keypath_sighash` | P | - | Y | - | N/A | - | Y (BIP-341) | N/A |
| `ufsecp_tapscript_sighash` | P | - | Y | - | N/A | - | Y (BIP-341) | N/A |

**Test files:** `audit/test_adversarial_protocol.cpp`, `tests/test_ffi_coverage.cpp`
**Implementation:** `ufsecp/ufsecp_impl.cpp`, `src/bip143.cpp`

---

## 29. Batch Signing (2 functions)

| Function | Unit Test | Fuzz | Adversarial | Differential | CT Path | GPU | Ext. Vectors | Zeroization |
|----------|-----------|------|-------------|--------------|---------|-----|-------------|-------------|
| `ufsecp_ecdsa_sign_batch` | P | - | Y | - | Y | - | N/A | Y |
| `ufsecp_schnorr_sign_batch` | P | - | Y | - | Y | - | N/A | Y |

**Test files:** `audit/test_adversarial_protocol.cpp`
**Implementation:** `ufsecp/ufsecp_impl.cpp`

---

## 30. GPU C ABI (`ufsecp_gpu_*`) -- 23 functions

Backend-neutral GPU acceleration surface (`ufsecp_gpu.h`). Separate opaque context (`ufsecp_gpu_ctx*`).

### Discovery & Lifecycle

| Function | Unit Test | Negative/NULL | Error Strings | Notes |
|----------|-----------|---------------|---------------|-------|
| `ufsecp_gpu_backend_count` | Y | Y (empty output) | N/A | Returns compiled backend IDs |
| `ufsecp_gpu_backend_name` | Y | Y (invalid ID → "none") | N/A | CUDA/OpenCL/Metal/none |
| `ufsecp_gpu_is_available` | Y | Y (invalid ID → 0) | N/A | Runtime probe |
| `ufsecp_gpu_device_count` | Y | Y (invalid ID → 0) | N/A | Per-backend device count |
| `ufsecp_gpu_device_info` | Y | Y (NULL info, invalid dev) | N/A | Name, memory, CUs, clock |
| `ufsecp_gpu_ctx_create` | Y | Y (NULL ctx_out, invalid bid, bad dev) | N/A | Returns ERR_GPU_UNAVAILABLE |
| `ufsecp_gpu_ctx_destroy` | Y | Y (NULL safe) | N/A | delete + shutdown |
| `ufsecp_gpu_last_error` | Y | Y (NULL → ERR_NULL_ARG) | N/A | Last op result |
| `ufsecp_gpu_last_error_msg` | Y | Y (NULL → fixed msg) | N/A | Human-readable |
| `ufsecp_gpu_error_str` | Y | Y (unknown code → "unknown error") | Y | CPU + GPU codes |

### Batch Operations (First Wave)

| Function | OpenCL | CUDA | Metal | Equivalence Test | Notes |
|----------|--------|------|-------|-----------------|-------|
| `ufsecp_gpu_generator_mul_batch` | Y | Y | Y | Y (1*G == G) | Scalar→compressed pubkey |
| `ufsecp_gpu_ecdsa_verify_batch` | Y | Y | Y | - | Batch ECDSA verify |
| `ufsecp_gpu_schnorr_verify_batch` | Y | Y | Y | - | BIP-340 batch verify |
| `ufsecp_gpu_ecdh_batch` | Y | Y | Y | - | SECRET-BEARING |
| `ufsecp_gpu_hash160_pubkey_batch` | Y | Y | Y | - | SHA-256+RIPEMD-160 |
| `ufsecp_gpu_msm` | Y | Y | Y | - | Multi-scalar multiplication |
| `ufsecp_gpu_frost_verify_partial_batch` | Y | Y | Y | - | Batch FROST partial verification |
| `ufsecp_gpu_ecrecover_batch` | Y | Y | Y | - | Recover compressed pubkeys from recoverable ECDSA sigs |
| `ufsecp_gpu_zk_knowledge_verify_batch` | - | - | - | CUDA only | Batch ZK knowledge proof verification |
| `ufsecp_gpu_zk_dleq_verify_batch` | - | - | - | CUDA only | Batch DLEQ proof verification |
| `ufsecp_gpu_bulletproof_verify_batch` | - | - | - | CUDA only | Batch Bulletproof range proof verification |
| `ufsecp_gpu_bip324_aead_encrypt_batch` | - | - | - | CUDA only | Batch BIP-324 AEAD encrypt |
| `ufsecp_gpu_bip324_aead_decrypt_batch` | - | - | - | CUDA only | Batch BIP-324 AEAD decrypt |

**Test file:** `audit/test_gpu_abi_gate.cpp` (39 assertions)

---

## Audit & Testing Methodology Matrix

| Test Methodology | Files | Features Covered |
|-----------------|-------|-----------------|
| **Unit / FFI round-trip** | `test_ffi_round_trip.cpp` (286 ufsecp_ calls) | All 96 API functions |
| **Fuzzing (random input)** | `audit_fuzz.cpp` | ECDSA, Schnorr, scalars, field, DER, recovery, state fuzzing |
| **Parser fuzzing** | `test_fuzz_parsers.cpp` | Pubkey parse, DER parse |
| **Address/BIP32/FFI fuzzing** | `test_fuzz_address_bip32_ffi.cpp` | Addresses, WIF, BIP-32, BIP-39, coin deriv |
| **Adversarial protocol** | `test_adversarial_protocol.cpp` (89 unique ufsecp_ functions, 186 checks) | MuSig2 (nonce reuse/replay, rogue-key, transcript mutation, signer ordering, malicious aggregator), FROST (below-threshold, malformed commitment, malicious coordinator, duplicate nonce), Silent Payments, ECDSA adaptor (round-trip, invalid/wrong point, transcript mismatch, extraction misuse), Schnorr adaptor, DLEQ (malformed proof, wrong generators), BIP-32, FFI hostile-caller (null args, undersized buffers, overlapping buffers, malformed counts) |
| **Wycheproof ECDSA** | `test_wycheproof_ecdsa.cpp` | r/s validation, boundary scalars, bit-flip, DER, High-S |
| **Wycheproof ECDH** | `test_wycheproof_ecdh.cpp` | Infinity, off-curve, twist, zero key, commutativity |
| **Differential (fast vs CT)** | `audit_ct.cpp`, `differential_test.cpp` | Field, scalar, point ops, ECDSA, Schnorr |
| **CT sidechannel (dudect)** | `test_ct_sidechannel.cpp` | CT primitives, field, scalar, point, ECDSA sign, Schnorr sign, MuSig2, FROST |
| **Fault injection** | `test_fault_injection.cpp` | Scalar bit-flip, point coord flip, ECDSA/Schnorr sig flip, CT compare, cascading faults |
| **Carry propagation** | `test_carry_propagation.cpp` | Field/scalar boundary arithmetic correctness |
| **Cross-platform KAT** | `test_cross_platform_kat.cpp` | Known-answer tests across x86/ARM/RISC-V |
| **ABI gate** | `test_abi_gate.cpp` | ABI version, struct sizes, symbol visibility |
| **Debug invariants** | `test_debug_invariants.cpp` | Internal assertion coverage |
| **Fiat-Crypto vectors** | `test_fiat_crypto_vectors.cpp` | Reference field arithmetic from Fiat-Crypto project |
| **Fiat-Crypto linkage** | `test_fiat_crypto_linkage.cpp` | Formal verification linkage |
| **CT formal verification** | `test_ct_verif_formal.cpp` | Formal CT property checking |
| **BIP-327 vectors** | `test_musig2_bip327_vectors.cpp` | Official BIP-327 MuSig2 test vectors |
| **FROST KAT** | `test_frost_kat.cpp` | FROST known-answer tests |
| **Batch randomness** | `test_batch_randomness.cpp` | Random-linear-combination batch verify integrity |
| **Cross-libsecp256k1** | `test_cross_libsecp256k1.cpp` | Differential against upstream libsecp256k1 |

---

## Zeroization Coverage

Files with `secure_erase` for secret data cleanup:

| File | # Erase calls | Secrets covered |
|------|--------------|-----------------|
| `cpu/src/ct_sign.cpp` | 10 | Private key bytes, nonce (k, k'), challenge hash, aux XOR, tag hash |
| `cpu/src/ecies.cpp` | 14+ | Ephemeral privkey, shared secret X, KDF output, AES keystream, HMAC ipad/opad |
| `cpu/src/ecdh.cpp` | Multiple | Shared secret intermediate values |
| `cpu/src/ecdsa.cpp` | Multiple | RFC 6979 nonce intermediates |
| `cpu/src/musig2.cpp` | Multiple | Secret nonce (consumed after sign), partial sign intermediates |
| `include/ufsecp/ufsecp_impl.cpp` | Multiple | ABI boundary cleanup of parsed secrets |

**Implementation:** `secp256k1::detail::secure_erase` (compiler-barrier-protected memset that cannot be optimized away).

---

## Summary Statistics

| Metric | Count |
|--------|-------|
| Total `UFSECP_API` functions | 112 (96 CPU + 16 GPU) |
| Functions with unit test coverage | 112 (100%) |
| Functions tested in adversarial protocol | 89 (93%), 186 individual checks |
| Functions with fuzzing | ~40 (42%) |
| Functions with external test vectors | ~35 (36%) |
| Functions using CT signing path | ~25 (all secret-dependent ops) |
| Functions with GPU support | ~50+ (point/field/scalar/hash + derived ops) |
| Audit source files | 32 (.cpp files in `audit/`) |
| GPU backends | 3 (CUDA, OpenCL, Metal) |
| `secure_erase` call sites | 141 across 6 files |
| CTest targets | 42 |

### Coverage Gaps (items for future work)

1. ~~**ECIES:** No fuzz or adversarial testing (only FFI round-trip)~~ **RESOLVED** -- `test_ecies_regression.cpp` (85 tests: parity tamper, invalid prefix, truncated envelope, tamper matrix, KAT, ABI prefix rejection, pubkey parser consistency, RNG fail-closed)
2. ~~**ZK range proofs:** No adversarial/malformed proof testing~~ **RESOLVED** -- `test_exploit_zk_adversarial.cpp` (14 tests: garbage bytes, all-zero proof, scalar overflow, truncated data, identity pubkey, identity generator, degenerate G==H DLEQ, wrong commitment, overflow e, 64-byte-flip sensitivity)
3. ~~**Pedersen switch commit:** No adversarial testing~~ **RESOLVED** -- `test_exploit_pedersen_adversarial.cpp` (12 tests: switch roundtrip, zero-blind equivalence, switch binding, zero-commit identity, negation cancellation, imbalanced verify_sum, blind_sum subtraction, switch-as-normal rejection, double-spend detection, generator J independence)
4. ~~**Ethereum functions:** No differential testing against reference (e.g., ethers.js)~~ **RESOLVED** -- `audit/test_exploit_ethereum_differential.cpp` (10 tests, 15 sub-checks: address derivation go-ethereum KAT, privkey=1 canonical address, ecrecover vs ADDR_GOETH with go-ethereum test msg, EIP-191 hash vs web3.py, sign+ecrecover roundtrip, EIP-155 v encoding, eth_personal_sign roundtrip, tamper detection, keccak256("abc") go-ethereum KAT, anti-collision)
5. ~~**GPU sign (CUDA-only):** ECDSA/Schnorr signing only on CUDA, not on OpenCL/Metal~~ **PARTIALLY RESOLVED** -- OpenCL: wired `zk_knowledge_verify_batch`, `zk_dleq_verify_batch`, `bip324_aead_encrypt_batch`, `bip324_aead_decrypt_batch` (4 new kernels); `bulletproof_verify_batch` has PARITY-EXCEPTION (no OpenCL WNAF multi-scalar). Metal: stubs documented with PARITY-EXCEPTION/TODO markers pointing to OpenCL path. See `docs/BACKEND_ASSURANCE_MATRIX.md`.
6. ~~**Batch verify GPU:** Only CUDA has batch verify kernels; OpenCL/Metal missing~~ **PARTIALLY RESOLVED** -- see Gap #5 above; CUDA+OpenCL now have 4 matching ZK/BIP-324 batch ops. Bulletproof batch on OpenCL/Metal remains PARITY-EXCEPTION.
7. ~~**Parser fuzzing for advanced protocols:** MuSig2/FROST/adaptor have null-arg testing but no random-byte fuzz~~ **RESOLVED** -- `audit/test_fuzz_musig2_frost.cpp` (15 tests, 16 sub-checks: musig2 key_agg/nonce_agg/partial_verify/partial_sig_agg random inputs, FROST keygen_finalize/sign/verify_partial/aggregate random inputs, schnorr+ecdsa adaptor random inputs, boundary n_signers=0 → must error). Also added ClusterFuzzLite harnesses: `cpu/fuzz/fuzz_ecdsa.cpp` (ECDSA sign/verify invariants) and `cpu/fuzz/fuzz_schnorr.cpp` (BIP-340 Schnorr invariants) — total ClusterFuzzLite targets now 5.
