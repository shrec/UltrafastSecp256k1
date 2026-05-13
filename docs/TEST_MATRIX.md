# Test Coverage Matrix

**UltrafastSecp256k1 current repository state** -- Comprehensive Test Map for Auditors

The owner-grade audit bundle and `ci/validate_assurance.py` are the
authoritative source for live target counts. When a narrative summary below
lags behind the generated validation surfaces, prefer the generated counts.

---

## Summary

| Category | Tests | Status |
|----------|-------|--------|
| **CTest targets** | 261 active targets | [OK] Documented in current assurance validation |
| **Audit suite checks** | 641,194+ | [OK] 0 failures |
| **Exploit PoC test files** | **197 tests, 20+ categories** | [OK] 0 failures |
| **Fuzz harnesses** | 11 | [OK] Active (5 cpu/fuzz + 6 audit/; `libfuzzer_unified` CI-blocking) |
| **ECIES regression** | 85 | [OK] All passing |
| **Adversarial protocol** | 114 functions, 360+ checks | [OK] Active |
| **Side-channel (dudect)** | 1 | [OK] Active |
| **Benchmark suites** | 4+ | [OK] Active |
| **Platform-specific** | 5+ | [OK] Per-platform |

---

## Test File Index

### Audit Suites (`tests/`)

| File | Checks | Focus Area |
|------|--------|------------|
| `audit_field.cpp` | 264,484 | Field arithmetic: identity, commutativity, associativity, distributivity, inverse, boundary, Fermat, special values |
| `audit_scalar.cpp` | 93,847 | Scalar arithmetic: ring properties, overflow, negate, inverse, boundary-near-order |
| `audit_point.cpp` | 116,312 | Point operations: on-curve, group law, add, dbl, scalar_mul, compress/decompress, infinity |
| `audit_ct.cpp` | 120,128 | CT layer: FAST-vs-CT equivalence, complete formulas, no-branch verification |
| `audit_fuzz.cpp` | 15,423 | Fuzz-derived: random inputs through all operation paths |
| `audit_perf.cpp` | -- | Performance benchmarks (throughput, latency) |
| `audit_security.cpp` | 17,856 | Security: nonce uniqueness, invalid input rejection, edge-case handling |
| `audit_integration.cpp` | 13,144 | End-to-end: sign->verify, derive->use, full protocol flows |
| `test_ct_sidechannel.cpp` | -- | dudect timing: Welch t-test for side-channel leakage |
| `differential_test.cpp` | -- | Cross-implementation comparison |
| `test_ecies_regression.cpp` | 85 | ECIES hardening: parity tamper, invalid prefix, truncated envelope, tamper matrix, KAT, ABI prefix rejection, pubkey parser consistency, RNG fail-closed |
| `test_adversarial_protocol.cpp` | 114 functions, 360+ checks | Adversarial protocol: MuSig2 (nonce reuse/replay, rogue-key, transcript mutation, signer ordering, malicious aggregator), FROST (below-threshold, malformed commitment, malicious coordinator, duplicate nonce), Silent Payments, ECDSA adaptor (round-trip, transcript mismatch, extraction misuse), Schnorr adaptor, DLEQ (malformed proof, wrong generators), BIP-32, FFI hostile-caller (null args, undersized buffers, overlapping buffers, malformed counts), **New ABI edge cases H.1-H.12** (ctx_size, AEAD, ECIES, EllSwift, ETH address, Pedersen switch, Schnorr adaptor extract, batch sign, BIP-143, BIP-144, SegWit, Taproot sighash), **Remaining ABI surface I.1-I.5** (ctx_clone, last_error_msg, pubkey_parse, pubkey_create_uncompressed, ecdsa_sign_recoverable, ecdsa_recover, ecdsa_sign_verified, schnorr_sign_verified, batch verify deep) |
| `test_fuzz_parsers.cpp` | 10K/suite | Parser fuzz: DER, Schnorr sig, compressed/uncompressed pubkey round-trip |
| `test_fuzz_address_bip32_ffi.cpp` | 10K/suite | Address/BIP-32/FFI fuzz: P2PKH/P2WPKH/P2TR/WIF, BIP-32 paths, BIP-39, coin derivation, FFI boundaries |
| `bench_ct_vs_libsecp.cpp` | -- | Performance comparison with libsecp256k1 |
| `bench_field_ops.cpp` | -- | Field operation microbenchmarks |
| `test_abi_gate.cpp` | -- | ABI compatibility gate: version checks, symbol presence, struct sizes |
| `test_batch_randomness.cpp` | -- | Batch randomness: nonce independence, distribution, uniqueness |
| `test_carry_propagation.cpp` | -- | Carry propagation: field arithmetic edge cases across limb boundaries |
| `test_cross_libsecp256k1.cpp` | -- | Cross-implementation: differential test against bitcoin-core/secp256k1 |
| `test_cross_platform_kat.cpp` | -- | Cross-platform known-answer tests: deterministic outputs across architectures |
| `test_debug_invariants.cpp` | -- | Debug invariants: internal consistency checks under debug mode |
| `test_fiat_crypto_linkage.cpp` | -- | Independent reference linkage: field arithmetic cross-check against schoolbook oracle |
| `test_frost_kat.cpp` | -- | FROST t-of-n threshold signing known-answer tests |
| `test_wycheproof_ecdsa.cpp` | -- | Wycheproof ECDSA: Google Project Wycheproof test vectors |
| `test_wycheproof_ecdh.cpp` | -- | Wycheproof ECDH: Google Project Wycheproof test vectors |
| `unified_audit_runner.cpp` | 249 modules (60 non-exploit + 260 exploit PoCs) | Unified audit: all current modules in single binary (includes GPU null-guard paths) |

### CPU Unit Tests (`src/cpu/tests/`)

| File | Focus Area | Status |
|------|------------|--------|
| `test_comprehensive.cpp` | 25+ categories: field, scalar, point, ECDSA, Schnorr, GLV, SHA, batch, etc. | [OK] |
| `test_arithmetic_correctness.cpp` | Arithmetic correctness: field/scalar edge cases | [OK] |
| `test_ct.cpp` | CT layer correctness (FAST vs CT equivalence) | [OK] |
| `test_ecdsa_schnorr.cpp` | ECDSA (RFC 6979) + Schnorr (BIP-340) vectors | [OK] |
| `test_ecdh_recovery_taproot.cpp` | ECDH, key recovery, Taproot | [OK] |
| `test_bip32.cpp` | BIP-32 HD key derivation | [OK] |
| `test_bip39.cpp` | BIP-39 mnemonic: PBKDF2, wordlist, entropy, validation, seed derivation (57 tests) | [OK] |
| `test_coins.cpp` | 28-coin address dispatch + P2SH/P2SH-P2WPKH/CashAddr | [OK] |
| `test_wallet.cpp` | Wallet API: key management, signing, address formats, recovery | [OK] |
| `test_ethereum.cpp` | Ethereum signing: EIP-155, EIP-191, ecrecover, personal_sign | [OK] |
| `test_musig2.cpp` | MuSig2 protocol tests | [OK] |
| `test_batch_add_affine.cpp` | Batch affine addition | [OK] |
| `test_multiscalar_batch.cpp` | Multi-scalar multiplication | [OK] |
| `test_simd_batch.cpp` | SIMD batch operations | [OK] |
| `test_mul.cpp` | Multiplication correctness | [OK] |
| `test_large_scalar_multiplication.cpp` | Large scalar multiplication | [OK] |
| `test_field_52.cpp` | 52-bit limb representation | [OK] |
| `test_field_26.cpp` | 26-bit limb representation | [OK] |
| `test_zk.cpp` | ZK proofs: knowledge, DLEQ, Bulletproof, batch (24 tests) | [OK] |
| `test_hash_accel.cpp` | SHA-256 acceleration tests | [OK] |
| `test_exhaustive.cpp` | Exhaustive tests (small curves) | [OK] |
| `test_v4_features.cpp` | v4 feature tests | [OK] |
| `run_selftest.cpp` | Selftest runner (smoke/ci/stress) | [OK] |
| `test_ecc_properties.cpp` | ECC algebraic properties: associativity, commutativity, distributivity | [OK] |
| `test_edge_cases.cpp` | Edge cases: scalar zero, infinity arithmetic, BIP-32 IL>=n, cache corruption | [OK] |
| `test_point_edge_cases.cpp` | Point edge cases: infinity, Z=0 guards, roundtrip encoding | [OK] |

### Fuzz Harnesses (`src/cpu/fuzz/`)

| File | Operations Fuzzed | Input |
|------|-------------------|-------|
| `fuzz_field.cpp` | add/sub round-trip, mul identity, square, inverse | 32-byte field element |
| `fuzz_scalar.cpp` | add/sub, mul identity, distributive law | 32-byte scalar |
| `fuzz_point.cpp` | on-curve check, negate, compress round-trip, dbl vs add | 32-byte x-coordinate seed |

### GPU Tests

| File | Backend | Focus |
|------|---------|-------|
| `opencl/tests/test_opencl.cpp` | OpenCL | Kernel correctness |
| `opencl/tests/opencl_extended_test.cpp` | OpenCL | Extended operations |
| `opencl/src/opencl_audit_runner.cpp` | OpenCL | Unified GPU audit (27 modules, 8 sections) |
| `metal/tests/test_metal_host.cpp` | Metal | Metal shader correctness |
| `metal/src/metal_audit_runner.mm` | Metal | `secp256k1_metal_audit`: unified GPU audit (27 modules, 8 sections) |
| `src/cuda/src/test_ct_smoke.cu` | CUDA | CT smoke tests incl. ZK knowledge + DLEQ prove/verify (9 tests) |
| `src/cuda/src/gpu_ct_leakage_probe.cu` | CUDA | Fixed-vs-random device-cycle Welch t-test for CT generator/signing kernels with JSON evidence output |
| `src/cuda/src/test_suite.cu` | CUDA | `cuda_selftest`: kernel correctness, field + scalar + point ops |
| `src/cuda/src/gpu_audit_runner.cu` | CUDA | `gpu_audit`: unified GPU audit (27 modules, 8 sections) |
| `metal/app/metal_test.mm` | Metal | `secp256k1_metal_test`: shader correctness, compute pipeline |
| `metal/app/bench_metal.mm` | Metal | `secp256k1_metal_bench_full`: comprehensive Metal benchmark |
| `compat/libsecp256k1_shim/tests/shim_test.cpp` | CPU | `secp256k1_shim_test`: libsecp256k1 API compatibility shim |
| `audit/test_gpu_abi_gate.cpp` | GPU (all) | `gpu_abi_gate`: GPU C ABI surface test -- discovery, lifecycle, NULL safety, error strings, generator_mul equivalence |
| `audit/test_gpu_ops_equivalence.cpp` | GPU (all) | `gpu_ops_equivalence`: GPU vs CPU reference for all 6 first-wave ops (skips UNSUPPORTED) |
| `audit/test_gpu_host_api_negative.cpp` | GPU (all) | `gpu_host_api_negative`: NULL ptrs, count=0 no-ops, invalid backend/device, error strings |
| `audit/test_gpu_backend_matrix.cpp` | GPU (all) | `gpu_backend_matrix`: backend enumeration, device info sanity, per-backend op probing |

### Additional CTest Targets

These standalone CTest entries are part of the active validation surface and are tracked explicitly here so the documented matrix matches the real runner surface.

| CTest target | Scope | Notes |
|-------------|-------|-------|
| `audit_invariants` | CPU audit | Algebraic and serialization invariant checks across core arithmetic and point paths |
| `audit_secure_erase` | Security audit | Verifies zeroization / secure erase behavior remains wired and callable |
| `bip141_143_144` | Bitcoin protocol | SegWit serialization and BIP-141/BIP-143/BIP-144 correctness coverage |
| `bip342` | Bitcoin protocol | Taproot / Tapscript BIP-342 correctness coverage |
| `c_abi_negative` | C ABI hostile-caller | Negative-path checks for invalid inputs, malformed pointers, and fail-closed behavior |
| `c_abi_thread_stress` | C ABI concurrency | Threaded stress coverage for public ABI entry points |
| `gpu_ct_leakage_probe` | GPU | CUDA advisory side-channel probe using fixed-vs-random `clock64()` Welch t-test on CT generator and signing kernels |
| `exploit_ecdsa_der_confusion` | Exploit PoC | Ensures DER parser behavior rejects non-canonical or ambiguous encodings |
| `exploit_batch_verify_poison` | Exploit PoC | Regression coverage for adversarial batch-verify poisoning and accumulator corruption attempts |
| `exploit_der_parsing_differential` | Exploit PoC | Differential strictness coverage for ambiguous or non-canonical DER parser behavior |
| `exploit_ecies_envelope_confusion` | Exploit PoC | Hostile envelope parsing and domain-confusion regression coverage for ECIES inputs |
| `exploit_ecrecover_confusion` | Exploit PoC | Recovery-path confusion and invalid-recid regression coverage for Ethereum/compact recovery flows |
| `exploit_frost_commitment_reuse` | Exploit PoC | Adversarial FROST nonce/commitment reuse regression coverage |
| `exploit_gpu_cpu_divergence` | Exploit PoC | Detects backend divergence between GPU results and CPU reference behavior |
| `exploit_gpu_host_api_shape` | Exploit PoC | Validates hostile shape/count/path misuse against the public GPU host API |
| `exploit_hedged_nonce_bias` | Exploit PoC | Regression coverage for biased or malformed hedged-nonce construction |
| `exploit_invalid_curve_twist` | Exploit PoC | Rejects twist/off-curve inputs that could poison public-key or shared-secret flows |
| `exploit_pedersen_adversarial` | Exploit PoC | Adversarial Pedersen commitment misuse/regression coverage |
| `exploit_pedersen_switch_misuse` | Exploit PoC | Switch-commitment misuse and binding-confusion regression coverage |
| `exploit_schnorr_xonly_parity_confusion` | Exploit PoC | X-only/parity ambiguity regression coverage for Schnorr verification flows |
| `exploit_seckey_tweak_cancel` | Exploit PoC | Regression coverage for tweak-cancellation edge cases on secret-key arithmetic |
| `exploit_silent_payment_confusion` | Exploit PoC | Silent payment transcript and domain-confusion regression coverage |
| `exploit_taproot_merkle_path_alias` | Exploit PoC | Detects aliasing and malformed-merkle-path edge cases in Taproot proof handling |
| `ffi_coverage` | FFI surface | Coverage-oriented validation for public foreign-function interface paths |
| `kat_all_operations` | Known-answer tests | Broad deterministic vectors across exposed operations |
| `nonce_uniqueness` | Security audit | Nonce uniqueness and replay-resistance regression coverage |
| `secp256k1_spec` | Spec conformance | Specification-oriented secp256k1 behavior checks |

### Supplemental Active CTest Targets

The following active CTest targets are also part of the documented validation
surface and are named here explicitly so the matrix matches the generated
CTest inventory exactly:

- `[=[metal_host_test]=]`
- `audit_ct_namespace`
- `bip324_transport`
- `bip32_vectors`
- `bip340_strict`
- `bip340_vectors`
- `ct_equivalence`
- `ct_sidechannel_smoke`
- `ct_verif_formal`
- `diag_scalar_mul`
- `exploit_bip324_counter_desync`
- `exploit_bip324_transcript_splice`
- `exploit_batch_verify_poison`
- `exploit_der_parsing_differential`
- `exploit_ecies_envelope_confusion`
- `exploit_ecrecover_confusion`
- `exploit_ethereum_differential`
- `exploit_gpu_cpu_divergence`
- `exploit_gpu_host_api_shape`
- `exploit_hedged_nonce_bias`
- `exploit_invalid_curve_twist`
- `exploit_musig2_transcript_fork`
- `exploit_pedersen_switch_misuse`
- `exploit_schnorr_xonly_parity_confusion`
- `exploit_seckey_tweak_cancel`
- `exploit_silent_payment_confusion`
- `exploit_taproot_merkle_path_alias`
- `exploit_zk_adversarial`
- `fault_injection`
- `fiat_crypto_vectors`
- `fuzz_musig2_frost`
- `gpu_ct_smoke`
- `musig2_bip327_vectors`
- `musig2_frost`
- `musig2_frost_advanced`
- `mutation_artifact_scan`
- `opencl_selftest`
- `parse_strictness`
- `rfc6979_vectors`
- `secp256k1_ecdh_example`
- `secp256k1_ecdsa_example`
- `secp256k1_ellswift_example`
- `secp256k1_musig_example`
- `secp256k1_schnorr_example`

### Generated Inventory Sync (2026-04-06)

The following active CTest targets were added to the explicit inventory during
the 2026-04-06 fortress sync so `TEST_MATRIX.md` matches the live generated
CTest surface exactly:

- `${_harness}`
- `bip352_kat`
- `exploit_adaptor_extraction_soundness`
- `exploit_address_prefix_collision`
- `exploit_batch_sign`
- `exploit_biased_nonce_chain_scan`
- `exploit_binding_adversarial_api`
- `exploit_binding_invalid_curve`
- `exploit_bip322_type_confusion`
- `exploit_bip32_child_key_attack`
- `exploit_bip32_parent_fingerprint_confusion`
- `exploit_bip352_parity_confusion`
- `exploit_bip352_scan_dos`
- `exploit_bip85_path_collision`
- `exploit_buff_kr_ecdsa`
- `exploit_buffer_type_confusion`
- `exploit_cache_sidechannel_amplification`
- `exploit_cross_scheme_pubkey`
- `exploit_ct_fast_equivalence`
- `exploit_ctx_clone`
- `exploit_ctx_lifecycle_hostile`
- `exploit_descriptor_injection`
- `exploit_differential_libsecp`
- `exploit_ecdh_zvp_glv_static`
- `exploit_ecdsa_fault_injection`
- `exploit_ecdsa_nonce_reuse`
- `exploit_ecdsa_r_overflow`
- `exploit_ecdsa_pmn_wraparound`
- `exploit_ecdsa_sign_sentinels`
- `exploit_boundary_sentinels`
- `exploit_ecies_ephemeral_reuse`
- `exploit_eip712_kat`
- `exploit_ellswift_bad_scalar_ecdh`
- `exploit_ellswift_xdh_overflow`
- `exploit_fe_set_b32_limit_uninit`
- `exploit_field_boundary_exhaustive`
- `exploit_foreign_field_plonk`
- `exploit_frost_adaptive_corruption`
- `exploit_frost_binding_factor_mismatch`
- `exploit_frost_ct_nonce`
- `exploit_frost_identifiable_abort`
- `exploit_frost_participant_set_malleability`
- `exploit_gcs_false_positive`
- `exploit_hash_algo_sig_isolation`
- `exploit_hertzbleed_dvfs_timing`
- `exploit_kr_ecdsa_buff_binding`
- `exploit_ladderleak_subbit_nonce`
- `exploit_metal_field_reduce`
- `exploit_minerva_cve_2024_23342`
- `exploit_minerva_noisy_hnp`
- `exploit_mutation_residue`
- `exploit_musig2_byzantine_multiparty`
- `exploit_musig2_parallel_session_cross`
- `exploit_network_validation_bypass`
- `exploit_p2sh_address_confusion`
- `exploit_psbt_input_confusion`
- `exploit_pubkey_arith`
- `exploit_quantum_exposure`
- `exploit_rfc6979_minerva_amplified`
- `exploit_rfc6979_truncation_bias`
- `exploit_scalar_mul`
- `exploit_schnorr_forgery_vectors`
- `exploit_schnorr_msg_length_confusion`
- `exploit_schnorr_nonce_reuse`
- `exploit_seckey_arith`
- `exploit_taproot_commitment_adversarial`
- `exploit_wallet_cross_domain_replay`
- `exploit_wif_security`
- `exploit_zk_new_schemes`
- `gpu_bip352_scan`
- `gpu_ecdsa_snark_witness`
- `infinity_edge_cases`
- `py_bip32_cka`
- `py_dev_bug_scan`
- `py_differential_crossimpl`
- `py_glv_exhaustive`
- `py_hot_path_alloc_scan`
- `py_invalid_input_grammar`
- `py_nonce_bias`
- `py_rfc6979_spec`
- `py_semantic_props`
- `py_stateful_sequences`
- `wycheproof_chacha20_poly1305`
- `wycheproof_ecdsa_bitcoin`
- `wycheproof_ecdsa_extended`
- `wycheproof_ecdsa_sha256`
- `wycheproof_ecdsa_sha256_p1363`
- `wycheproof_ecdsa_sha512`
- `wycheproof_ecdsa_sha512_p1363`
- `wycheproof_hkdf_sha256`
- `wycheproof_hmac_sha256`

### Generated Inventory Sync (2026-04-09)

The following active CTest targets were added during the 2026-04-09 audit
gate reconciliation:

- `exploit_bip324_aead_forgery`
- `exploit_blind_spa_cmov_leak`
- `exploit_deterministic_sig_dfa`
- `exploit_ecdh_twist_injection`
- `exploit_ecdsa_affine_nonce_relation`
- `exploit_ecdsa_batch_verify_rand`
- `exploit_ecdsa_cross_key_nonce_reuse`
- `exploit_ecdsa_differential_fault`
- `exploit_ecdsa_half_half_nonce`
- `exploit_ecdsa_nonce_modular_bias`
- `exploit_ectester_point_validation`
- `exploit_eucleak_inversion_timing`
- `exploit_frost_rogue_key`
- `exploit_frost_weak_binding`
- `exploit_lattice_sieve_hnp`
- `exploit_musig2_partial_forgery`
- `exploit_ros_concurrent_schnorr`
- `exploit_ros_dimensional_erosion`
- `exploit_schnorr_batch_inflation`
- `exploit_schnorr_hash_order`
- `exploit_sign_type_confusion_kreuse`
- `exploit_zvp_glv_dcp_multiscalar`

### Generated Inventory Sync (2026-05-06)

The following active CTest targets were added during the 2026-05-06 performance
review so `TEST_MATRIX.md` matches the live generated CTest surface:

- `exploit_frost_secret_share_ct`
- `regression_comb_gen_lockfree`
- `regression_pippenger_stale_used`

### Generated Inventory Sync (2026-05-03)

The following active CTest targets were added during the 2026-05-03 audit
validation sync so `TEST_MATRIX.md` matches the live generated CTest surface:

- `exploit_batch_verify_malleability`
- `exploit_blinding_recovery_hnp`
- `exploit_bug004_batch_failclosed`
- `exploit_cross_protocol_kreuse`
- `exploit_ecdsa_fast_path_isolation`
- `exploit_encoding_memory_corruption`
- `exploit_gpu_memory_safety`
- `exploit_hedged_return_value`
- `exploit_kat_corpus`
- `exploit_primitive_kat`
- `exploit_rs_zero_check`
- `exploit_shim_der_bip66`
- `exploit_shim_musig_ka_cap`
- `exploit_shim_musig_secnonce`
- `exploit_shim_noncefp_bypass`
- `exploit_shim_recovery_null_arg`
- `exploit_tagged_hash_ext`
- `exploit_thread_local_blinding`
- `regression_gpu_key_erase_raii`
- `regression_shim_per_context_blinding`
- `regression_shim_static_ctx`
- `shim_der_zero_r`
- `regression_shim_pubkey_sort`
- `secp256k1_noverify_tests`
- `secp256k1_tests`
- `zeroization`

### Generated Inventory Sync (2026-05-12)

The following active CTest targets were added during the 2026-05-12 doc+config
fix pass so `TEST_MATRIX.md` matches the live generated CTest surface:

- `regression_frost_threshold_zero`
- `regression_hash_three_block_bounds`
- `regression_shim_high_s_verify`
- `regression_shim_perf_correctness`

---

## API Function -> Test Coverage Map

### Field Arithmetic (`FieldElement`)

| Function | audit_field | test_comprehensive | fuzz_field | CT check |
|----------|:-----------:|:-----------------:|:----------:|:--------:|
| `add` / `operator+` | [OK] | [OK] | [OK] | [OK] |
| `sub` / `operator-` | [OK] | [OK] | [OK] | [OK] |
| `mul` / `operator*` | [OK] | [OK] | [OK] | [OK] |
| `square()` | [OK] | [OK] | [OK] | [OK] |
| `inverse()` | [OK] | [OK] | [OK] | [OK] |
| `negate()` | [OK] | [OK] | -- | [OK] |
| `from_limbs()` | [OK] | [OK] | -- | -- |
| `from_bytes()` | [OK] | [OK] | -- | -- |
| `to_bytes()` | [OK] | [OK] | -- | -- |
| `from_hex()` / `to_hex()` | [OK] | [OK] | -- | -- |
| `normalize()` | [OK] | [OK] | [OK] | -- |
| `field_select()` | -- | -- | -- | [OK] |
| `square_inplace()` | [OK] | -- | -- | -- |
| `inverse_inplace()` | [OK] | -- | -- | -- |
| `fe_batch_inverse()` | [OK] | [OK] | -- | -- |

### Scalar Arithmetic (`Scalar`)

| Function | audit_scalar | test_comprehensive | fuzz_scalar | CT check |
|----------|:------------:|:-----------------:|:-----------:|:--------:|
| `add` / `operator+` | [OK] | [OK] | [OK] | [OK] |
| `sub` / `operator-` | [OK] | [OK] | [OK] | [OK] |
| `mul` / `operator*` | [OK] | [OK] | [OK] | [OK] |
| `inverse()` | [OK] | [OK] | -- | [OK] |
| `negate()` | [OK] | [OK] | -- | [OK] |
| `from_uint64()` | [OK] | [OK] | -- | -- |
| `from_bytes()` | [OK] | [OK] | -- | -- |
| `from_hex()` | [OK] | [OK] | -- | -- |
| `is_zero()` | [OK] | [OK] | -- | -- |

### Point Operations (`Point`)

| Function | audit_point | test_comprehensive | fuzz_point | CT check |
|----------|:-----------:|:-----------------:|:----------:|:--------:|
| `add()` | [OK] | [OK] | [OK] | [OK] |
| `dbl()` / `double_point()` | [OK] | [OK] | [OK] | [OK] |
| `scalar_mul()` | [OK] | [OK] | -- | [OK] |
| `is_on_curve()` | [OK] | [OK] | [OK] | -- |
| `is_infinity()` | [OK] | [OK] | -- | -- |
| `compress()` / `decompress()` | [OK] | [OK] | [OK] | -- |
| `to_affine()` | [OK] | [OK] | -- | -- |
| `generator()` | [OK] | [OK] | -- | -- |
| `negate()` | [OK] | [OK] | [OK] | -- |

### GLV Endomorphism

| Function | audit_point | test_comprehensive | CT check |
|----------|:-----------:|:-----------------:|:--------:|
| `apply_endomorphism()` | [OK] | [OK] | [OK] |
| `verify_endomorphism()` | -- | [OK] | -- |
| `glv_decompose()` | [OK] | [OK] | [OK] |
| `ct::point_endomorphism()` | -- | -- | [OK] |

### Signatures

| Function | audit_security | audit_integration | test_ecdsa_schnorr | dudect |
|----------|:-------------:|:-----------------:|:------------------:|:------:|
| `ecdsa::sign()` | [OK] | [OK] | [OK] | [OK] |
| `ecdsa::verify()` | [OK] | [OK] | [OK] | -- |
| `schnorr::sign()` | [OK] | [OK] | [OK] | [OK] |
| `schnorr::verify()` | [OK] | [OK] | [OK] | -- |

### CT Layer

CT functions are verified by a layered approach: equivalence tests (`audit_ct`, `test_ct`),
statistical timing tests (`dudect`), and deterministic CT verification (`ct-verif` + `valgrind-ct` in CI).
Machine-checked proofs (Fiat-Crypto/Vale/Jasmin) are not yet applied.

| Function | audit_ct | test_ct | dudect | ct-verif | Machine-Checked Proof |
|----------|:--------:|:-------:|:------:|:--------:|:---------------------:|
| `ct::field_mul` | [OK] | [OK] | [OK] | [OK] | -- |
| `ct::field_inv` | [OK] | [OK] | [OK] | [OK] | -- |
| `ct::scalar_mul` | [OK] | [OK] | [OK] | [OK] | -- |
| `ct::generator_mul` | [OK] | [OK] | [OK] | [OK] | -- |
| `ct::point_add_complete` | [OK] | [OK] | [OK] | [OK] | -- |
| `ct::point_dbl` | [OK] | [OK] | -- | [OK] | -- |

### Protocols

| Function | Test File | Coverage | Notes |
|----------|-----------|----------|-------|
| MuSig2 key aggregation | `test_musig2.cpp` | [OK] Basic | No extended vectors |
| MuSig2 2-round sign | `test_musig2.cpp` | [OK] Full | Rogue-key, transcript mutation, signer ordering, malicious aggregator adversarial tests added |
| FROST t-of-n | `test_v4_features.cpp` | [OK] Basic | Keygen, sign, aggregate, verify |
| Adaptor signatures | `test_v4_features.cpp` | [OK] Full | Transcript mismatch, extraction misuse, DLEQ malformed proof, wrong generators adversarial tests added |
| Pedersen commitments | `test_v4_features.cpp` | [OK] Basic | Limited vectors |
| ZK Knowledge proof | `test_zk.cpp` | [OK] | Prove/verify, arbitrary base, serialization |
| ZK DLEQ proof | `test_zk.cpp` | [OK] | Prove/verify, cross-basis equality |
| ZK Bulletproof range | `test_zk.cpp` | [OK] | Prove/verify, boundary values, inner product |
| ZK batch range verify | `test_zk.cpp` | [OK] | Multi-proof batch verification |
| GPU ZK Knowledge proof | `test_ct_smoke.cu` | [OK] | CT prove + fast-path verify on CUDA |
| GPU ZK DLEQ proof | `test_ct_smoke.cu` | [OK] | CT prove + fast-path verify on CUDA |
| Taproot (BIP-341) | `test_ecdh_recovery_taproot.cpp` | [OK] Basic | -- |
| BIP-32 HD derivation | `test_bip32.cpp` | [OK] | Standard vectors |
| 28-coin dispatch | `test_coins.cpp` | [OK] | Per-coin address format (P2PKH, P2WPKH, P2TR, P2SH-P2WPKH, CashAddr, EIP-55, TRON_BASE58) |
| Wallet API | `test_wallet.cpp` | [OK] | Chain-agnostic key mgmt, signing, recovery |
| Ethereum signing | `test_ethereum.cpp` | [OK] | EIP-155/-191, ecrecover, multi-chain |
| ECDH | `test_ecdh_recovery_taproot.cpp` | [OK] | -- |
| Key recovery | `test_ecdh_recovery_taproot.cpp` | [OK] | -- |

---

## New ABI Surface Edge-Case Coverage (v3.22+ §N)

> Gap analysis found 26 `ufsecp_*` functions with no dedicated edge-case tests.
> All gaps are closed by `test_h1_*`–`test_h12_*` in
> `audit/test_adversarial_protocol.cpp`.

| Test ID | ABI functions | NULL | Zero-count/len | Invalid content | Smoke |
|---------|---------------|:----:|:--------------:|:---------------:|:-----:|
| H.1 | `ufsecp_ctx_size` | -- | -- | -- | [OK] |
| H.2 | `ufsecp_aead_chacha20_encrypt`, `ufsecp_aead_chacha20_decrypt` | [OK] | [OK] | [OK] (bad-tag, wrong-nonce) | [OK] |
| H.3 | `ufsecp_ecies_encrypt`, `ufsecp_ecies_decrypt` | [OK] | -- | [OK] (off-curve, tampered) | [OK] |
| H.4 | `ufsecp_ellswift_create`, `ufsecp_ellswift_xdh` | [OK] | -- | [OK] (zero key) | [OK] |
| H.5 | `ufsecp_eth_address_checksummed`, `ufsecp_eth_personal_hash` | [OK] | [OK] | -- | [OK] |
| H.6 | `ufsecp_pedersen_switch_commit` | [OK] | -- | -- | [OK] |
| H.7 | `ufsecp_schnorr_adaptor_extract` | [OK] | -- | [OK] (zero inputs) | -- |
| H.8 | `ufsecp_ecdsa_sign_batch`, `ufsecp_schnorr_sign_batch` | [OK] | [OK] | -- | -- |
| H.9 | `ufsecp_bip143_sighash`, `ufsecp_bip143_p2wpkh_script_code` | [OK] | -- | -- | [OK] |
| H.10 | `ufsecp_bip144_txid`, `ufsecp_bip144_wtxid`, `ufsecp_bip144_witness_commitment` | [OK] | -- | -- | [OK] |
| H.11 | `ufsecp_is_witness_program`, `ufsecp_parse_witness_program`, `ufsecp_p2wpkh_spk`, `ufsecp_p2wsh_spk`, `ufsecp_p2tr_spk`, `ufsecp_witness_script_hash` | [OK] | -- | [OK] (non-witness) | [OK] |
| H.12 | `ufsecp_taproot_keypath_sighash`, `ufsecp_tapscript_sighash` | [OK] | [OK] | [OK] (OOB index) | [OK] |

---

## Coverage Gaps (Transparency)

### High Priority

| Gap | Impact | Blocked By |
|-----|--------|------------|
| **Machine-checked proofs** | CT/math properties not proven in Coq/Jasmin/Vale-style frameworks | Separate proof-bearing core or generated arithmetic path needed |
| **Cross-ABI tests** | Cannot verify FFI correctness across calling conventions | Need multi-compiler test matrix |

### Medium Priority

| Gap | Impact | Status |
|-----|--------|--------|
| MuSig2 extended test vectors | Full adversarial coverage (A.4-A.7) | Reference impl vectors available via BIP-327 |
| Multi-uarch timing tests | CT may break on specific CPUs | Need hardware test farm |
| GPU vs CPU differential | GPU arithmetic may diverge | Covered by gpu_ops_equivalence (6 ops) + OpenCL/CUDA tests |

### Low Priority

| Gap | Impact | Status |
|-----|--------|--------|
| WASM-specific tests | WASM arithmetic may diverge | Build-tested, limited runtime tests |
| ESP32/STM32 hardware tests | Embedded correctness | Requires physical devices |
| Adaptor signature extended vectors | Full adversarial coverage (D.1-D.6, E.1-E.5) | Transcript mismatch and extraction misuse covered |

---

## Continuous Integration Test Matrix

| Platform | Compiler | Sanitizers | Tests |
|----------|----------|------------|-------|
| Linux x86-64 | GCC 12+ | ASan, UBSan, TSan | Full suite |
| Linux x86-64 | Clang 15+ | ASan, UBSan | Full suite |
| Linux ARM64 | aarch64-linux-gnu + QEMU | -- | Cross-build + `run_selftest smoke` + `test_bip324_standalone` + `bench_kP` + `bench_bip324` |
| Linux RISC-V 64 | riscv64-linux-gnu + QEMU | -- | Cross-build + `run_selftest smoke` + `test_bip324_standalone` + `bench_kP` + `bench_bip324` |
| Windows x86-64 | MSVC 2022 | -- | Full suite |
| macOS ARM64 | AppleClang | -- | Full suite |
| macOS x86-64 | AppleClang | -- | Full suite |
| iOS ARM64 | Xcode toolchain | -- | Build only |
| Android ARM64 | NDK | -- | Build only |
| WASM | Emscripten | -- | Build + smoke |
| CUDA | nvcc + host compiler | -- | GPU-specific |
| Valgrind | GCC/Clang | Memcheck | Weekly |

---

## Running Tests

```bash
# All CTest targets
ctest --test-dir build --output-on-failure

# Specific audit suite
./build/tests/audit_field
./build/tests/audit_scalar
./build/tests/audit_point
./build/tests/audit_ct

# Side-channel test
./build/tests/test_ct_sidechannel

# Fuzzing (clang required)
clang++ -fsanitize=fuzzer,address -O2 -std=c++20 \
  -I src/cpu/include src/cpu/fuzz/fuzz_field.cpp src/cpu/src/field.cpp \
  -o fuzz_field
./fuzz_field -max_len=64 -runs=10000000

# Selftest (smoke/ci/stress modes)
./build/src/cpu/tests/run_selftest

# Linux ARM64 smoke under QEMU (cross-compiled)
bash ./ci/run-qemu-smoke.sh arm64

# Or run the commands manually
qemu-aarch64 -L /usr/aarch64-linux-gnu ./build-arm64/cpu/run_selftest smoke
qemu-aarch64 -L /usr/aarch64-linux-gnu ./build-arm64/src/cpu/test_bip324_standalone
qemu-aarch64 -L /usr/aarch64-linux-gnu ./build-arm64/cpu/bench_kP
qemu-aarch64 -L /usr/aarch64-linux-gnu ./build-arm64/cpu/bench_bip324

# Linux RISC-V smoke under QEMU (cross-compiled)
bash ./ci/run-qemu-smoke.sh riscv64

# Or run the commands manually
qemu-riscv64 -L /usr/riscv64-linux-gnu ./build-riscv64/cpu/run_selftest smoke
qemu-riscv64 -L /usr/riscv64-linux-gnu ./build-riscv64/src/cpu/test_bip324_standalone
qemu-riscv64 -L /usr/riscv64-linux-gnu ./build-riscv64/cpu/bench_kP
qemu-riscv64 -L /usr/riscv64-linux-gnu ./build-riscv64/cpu/bench_bip324
```

---

## Exploit PoC Test Suite (`audit/test_exploit_*.cpp`)

197 standalone exploit-style tests that actively try to break the library.
Each test compiles as a separate binary and verifies that attacks fail, edge cases are handled, and security invariants hold under adversarial inputs.

| Category | File(s) | Attack / Property Verified |
|----------|---------|---------------------------|
| ECDSA / Signature | `test_exploit_ecdsa_malleability` | BIP-62 low-s enforcement, high-s rejection, `normalize()`, strict parser |
| ECDSA / Signature | `test_exploit_ecdsa_edge_cases` | Zero and boundary inputs |
| ECDSA / Signature | `test_exploit_ecdsa_recovery` | Key recovery edge cases |
| ECDSA / Signature | `test_exploit_ecdsa_rfc6979_kat` | RFC 6979 deterministic nonce KAT |
| ECDH | `test_exploit_ecdh` | ECDH correctness |
| ECDH | `test_exploit_ecdh_degenerate` | Degenerate ECDH inputs |
| ECDH | `test_exploit_ecdh_variants` | ECDH variants |
| Schnorr / BIP-340 | `test_exploit_schnorr_edge_cases` | Schnorr edge cases |
| Schnorr / BIP-340 | `test_exploit_schnorr_bip340_kat` | BIP-340 known-answer tests |
| Batch Schnorr | `test_exploit_batch_schnorr` | Basic batch Schnorr verification |
| Batch Schnorr | `test_exploit_batch_schnorr_forge` | Forge detection, `identify_invalid` accuracy |
| Batch Schnorr | `test_exploit_batch_soundness` | Batch soundness properties |
| GLV / Math | `test_exploit_glv_endomorphism` | Endomorphism properties |
| GLV / Math | `test_exploit_glv_kat` | GLV ±k₁±k₂λ≡k, φ(G)=λG, φ²+φ+1=0 decomposition KAT |
| GLV / Math | `test_exploit_field_arithmetic` | Field element arithmetic |
| GLV / Math | `test_exploit_scalar_group_order` | Scalar group-order properties |
| GLV / Math | `test_exploit_scalar_invariants` | Scalar invariants |
| GLV / Math | `test_exploit_scalar_systematic` | Systematic scalar coverage |
| GLV / Math | `test_exploit_point_group_law` | Point group law |
| GLV / Math | `test_exploit_point_serialization` | Point serialization |
| GLV / Math | `test_exploit_multiscalar` | Multi-scalar multiplication |
| GLV / Math | `test_exploit_pippenger_msm` | Pippenger MSM |
| Batch Verify | `test_exploit_batch_verify_correctness` | Batch verify math |
| BIP-32 / HD | `test_exploit_bip32_depth` | Depth overflow |
| BIP-32 / HD | `test_exploit_bip32_derivation` | Derivation correctness |
| BIP-32 / HD | `test_exploit_bip32_path_overflow` | Path overflow attack |
| BIP-32 / HD | `test_exploit_bip32_ckd_hardened` | Hardened isolation, xpub guard, fingerprint |
| BIP-39 | `test_exploit_bip39_entropy` | Entropy edge cases |
| BIP-39 | `test_exploit_bip39_mnemonic` | Mnemonic generation and parsing |
| HD Derivation | `test_exploit_coin_hd_derivation` | HD derivation paths per coin type |
| MuSig2 | `test_exploit_musig2` | MuSig2 protocol |
| MuSig2 | `test_exploit_musig2_key_agg` | Key aggregation |
| MuSig2 | `test_exploit_musig2_nonce_reuse` | Nonce reuse attack |
| MuSig2 | `test_exploit_musig2_ordering` | Key ordering independence |
| FROST | `test_exploit_frost_byzantine` | Byzantine participant |
| FROST | `test_exploit_frost_dkg` | Distributed key generation |
| FROST | `test_exploit_frost_index` | Participant index handling |
| FROST | `test_exploit_frost_lagrange_duplicate` | Duplicate Lagrange coefficients |
| FROST | `test_exploit_frost_participant_zero` | Index-zero participant |
| FROST | `test_exploit_frost_signing` | FROST signing protocol |
| FROST | `test_exploit_frost_threshold_degenerate` | Degenerate threshold |
| Adaptor / ZK | `test_exploit_adaptor_extended` | Extended adaptor attacks |
| Adaptor / ZK | `test_exploit_adaptor_parity` | Adaptor parity handling |
| Adaptor / ZK | `test_exploit_zk_proofs` | ZK proof properties |
| Adaptor / ZK | `test_exploit_pedersen_homomorphism` | Pedersen commitment homomorphism |
| AEAD / ChaCha20 | `test_exploit_aead_integrity` | ChaCha20-Poly1305 MAC bypass, nonce reuse, zeroed output on failure |
| AEAD / ChaCha20 | `test_exploit_chacha20_kat` | ChaCha20 known-answer tests |
| AEAD / ChaCha20 | `test_exploit_chacha20_nonce_reuse` | Nonce reuse hazard |
| AEAD / ChaCha20 | `test_exploit_chacha20_poly1305` | AEAD roundtrip |
| HKDF | `test_exploit_hkdf_kat` | HKDF known-answer tests |
| HKDF | `test_exploit_hkdf_security` | HKDF security properties |
| Hash primitives | `test_exploit_keccak256_kat` | Keccak-256 KAT |
| Hash primitives | `test_exploit_ripemd160_kat` | RIPEMD-160 KAT |
| Hash primitives | `test_exploit_sha256_kat` | SHA-256 KAT |
| Hash primitives | `test_exploit_sha512_kat` | SHA-512 KAT |
| Hash primitives | `test_exploit_sha_kat` | SHA family KAT |
| ECIES | `test_exploit_ecies_auth` | ECIES authentication |
| ECIES | `test_exploit_ecies_encryption` | ECIES encryption |
| ECIES | `test_exploit_ecies_roundtrip` | ECIES roundtrip |
| Protocol BIPs | `test_exploit_bip143_sighash` | BIP-143 sighash |
| Protocol BIPs | `test_exploit_bip144_serialization` | BIP-144 serialization |
| Protocol BIPs | `test_exploit_bip324_session` | BIP-324 encrypted P2P session |
| Protocol BIPs | `test_exploit_segwit_encoding` | SegWit address encoding |
| Protocol BIPs | `test_exploit_taproot_scripts` | Taproot script path |
| Protocol BIPs | `test_exploit_taproot_tweak` | Taproot key tweak |
| Address / Wallet | `test_exploit_address_encoding` | Address encoding |
| Address / Wallet | `test_exploit_address_generation` | Address generation |
| Address / Wallet | `test_exploit_wallet_api` | Wallet API |
| Address / Wallet | `test_exploit_private_key` | Private key handling |
| Address / Wallet | `test_exploit_eth_signing` | Ethereum signing |
| Address / Wallet | `test_exploit_bitcoin_message_signing` | Bitcoin message signing |
| Constant-Time | `test_exploit_ct_recov` | CT key recovery |
| Constant-Time | `test_exploit_ct_systematic` | Systematic CT verification |
| Constant-Time | `test_exploit_backend_divergence` | Backend divergence detection |
| ElligatorSwift | `test_exploit_ellswift` | ElligatorSwift encoding correctness |
| ElligatorSwift | `test_exploit_ellswift_ecdh` | ElligatorSwift ECDH |
| Self-Test / API | `test_exploit_selftest_api` | Self-test API |
| Recovery | `test_exploit_recovery_extended` | Extended recovery edge cases |
| ECDSA / Signature | `test_exploit_ecdsa_nonce_reuse` | ECDSA nonce-reuse key extraction: verifies RFC 6979 prevents k reuse |
| ECDSA / Signature | `test_exploit_ecdsa_r_overflow` | ECDSA r-overflow: r ≥ n, r = 0, DER parse edge cases, PMN constants |
| ECDSA / Signature | `test_exploit_ecdsa_sign_sentinels` | ECDSA sign sentinels: zero-sk/r/s rejection; mass sign no zero-components |
| ECDSA / Signature | `test_exploit_rfc6979_truncation_bias` | RFC 6979 truncation bias: nonce truncation correctness for message sizes |
| ECDSA / Signature | `test_exploit_binding_invalid_curve` | Invalid curve point injection into ECDH/Schnorr verify rejected |
| Schnorr / BIP-340 | `test_exploit_batch_sign` | Verifies Schnorr batch-sign cannot produce invalid/forgeable signatures |
| Schnorr / BIP-340 | `test_exploit_schnorr_msg_length_confusion` | Schnorr message-length confusion: empty/short/prefix/large isolation |
| Schnorr / BIP-340 | `test_exploit_schnorr_nonce_reuse` | Schnorr nonce-reuse key extraction: tagged-hash nonce prevents reuse |
| GLV / Math | `test_exploit_ecdh_zvp_glv_static` | ECDH zero-value-point w/ GLV static key: all-zero output rejection |
| GLV / Math | `test_exploit_fe_set_b32_limit_uninit` | Field element set_b32 limit: uninit/poisoned-memory parsing safety |
| GLV / Math | `test_exploit_pubkey_arith` | Public-key arithmetic: add/tweak/negate/combine edge cases |
| GLV / Math | `test_exploit_seckey_arith` | Secret-key arithmetic: add/tweak/negate/verify edge cases on scalars |
| BIP-32 / HD | `test_exploit_bip85_path_collision` | BIP-85 path collision: different paths must yield different entropy |
| BIP-352 / Silent Payments | `test_exploit_bip352_parity_confusion` | BIP-352 parity confusion: pubkey negation vs ECDH output consistency |
| BIP-352 / Silent Payments | `test_exploit_bip352_scan_dos` | BIP-352 scan DoS resistance: timing-bounded large input sets |
| FROST | `test_exploit_frost_adaptive_corruption` | FROST adaptive corruption: corrupted partial sigs detected after DKG |
| FROST | `test_exploit_frost_identifiable_abort` | FROST identifiable abort: bad partials correctly identified in 2-of-3 |
| FROST | `test_exploit_frost_participant_set_malleability` | FROST participant-set malleability: reordered/swapped nonces detected |
| Constant-Time | `test_exploit_biased_nonce_chain_scan` | Nonce bias detection via chain-scanning statistical analysis |
| Constant-Time | `test_exploit_cache_sidechannel_amplification` | Cache side-channel amplification: timing noise vs ECDSA sign leakage |
| Constant-Time | `test_exploit_hertzbleed_dvfs_timing` | Hertzbleed/DVFS timing: ECDSA sign timing variance under freq scaling |
| Constant-Time | `test_exploit_minerva_cve_2024_23342` | Minerva CVE-2024-23342 timing attack: nanosecond median/MAD analysis |
| Constant-Time | `test_exploit_minerva_noisy_hnp` | Minerva noisy HNP: nonce-bias detection under realistic noise |
| Constant-Time | `test_exploit_rfc6979_minerva_amplified` | RFC 6979 Minerva amplified: repeated ECDSA sign nanosecond timing |
| ECIES | `test_exploit_ecies_ephemeral_reuse` | ECIES ephemeral key reuse to different recipients must be rejected |
| ElligatorSwift | `test_exploit_ellswift_xdh_overflow` | ElligatorSwift XDH overflow: boundary/malformed 64-byte inputs |
| Protocol BIPs | `test_exploit_eip712_kat` | EIP-712 structured data signing known-answer tests |
| Protocol BIPs | `test_exploit_taproot_commitment_adversarial` | Taproot commitment adversarial: output-key manipulation attempts |
| Address / Wallet | `test_exploit_psbt_input_confusion` | PSBT input confusion: segwit encoding, key isolation, taproot sighash |
| Recovery | `test_exploit_buff_kr_ecdsa` | Key-recovery ECDSA to Ethereum address binding roundtrip |
| Recovery | `test_exploit_kr_ecdsa_buff_binding` | Key-recovery ECDSA buffer binding: verify-after-recover consistency |
| Self-Test / API | `test_exploit_binding_adversarial_api` | Adversarial API misuse: ctx lifecycle, double destroy, bad ctx |
| Self-Test / API | `test_exploit_buffer_type_confusion` | Type confusion: passing wrong-type buffers to API functions |
| Self-Test / API | `test_exploit_cross_scheme_pubkey` | Cross-scheme pubkey reuse: same key in ECDH/Schnorr/ECIES isolation |
| Self-Test / API | `test_exploit_differential_libsecp` | Differential testing vs libsecp256k1: sign/verify/ECDH mismatch |
| Adaptor / ZK | `test_exploit_quantum_exposure` | Quantum exposure: pubkey creation under adversarial key guessing |
| Boundary sentinels | `test_exploit_boundary_sentinels` | Zero, max, order-boundary sentinel values across all API entry points |
| Hash | `test_exploit_hash_algo_sig_isolation` | Hash-algorithm signature isolation: SHA-256 vs alt-hash no cross-verify |
| Misc | `test_exploit_gcs_false_positive` | GCS filter: false-positive rate, determinism, null handling |
| Metal | `test_exploit_metal_field_reduce` | Metal field_reduce_512 truncation regression: acc[8] > 32-bit carry chains |
| Mutation | `test_exploit_mutation_residue` | Mutation residue exploit: inverse sweep, ECDSA roundtrip, CT scalar_inverse(0) |
| Mutation | `test_mutation_artifact_scan` | Source artifact scanner: stale mutation markers, vestigial test debris |
| ABI / Network | `test_exploit_network_validation_bypass` | Network selector validation bypass: OOB enum, INT_MAX, negative values |
| Scalar Mul | `test_exploit_scalar_mul` | Point::scalar_mul edge-case PoC: zero/identity/order-n/GLV/Shamir/MSM/overflow/doubling-chain |
| SafeGCD | `test_exploit_safegcd_divsteps` | SafeGCD / Bernstein-Yang divsteps correctness: modular inverse boundaries, reduction verification |
| Nonce | `test_exploit_custom_nonce_injection` | Nonce function edge cases: custom nonce injection, zero nonce, boundary nonce handling |
| ZK / Adaptor | `test_fault_zk_adaptor` | ZK/Pedersen/Adaptor fault-injection: bit-flip proofs, corrupted commitments, adaptor tampering |
| Field / Scalar | `test_field_scalar_edge` | Field & Scalar boundary conditions: carry propagation, reduction edge cases, overflow handling |
| Secret Lifecycle | `test_secret_lifecycle` | Secret lifecycle audit: sk creation → sign → verify → zeroize, cross-function secret flow |

Build and run all exploit tests:
```bash
cmake -S . -B build-audit -G Ninja -DCMAKE_BUILD_TYPE=Release -DSECP256K1_BUILD_TESTS=ON
cmake --build build-audit -j
ctest --test-dir build-audit -R "exploit" --output-on-failure
```

---

## BIP-352 Known-Answer Tests

| Target | Source | Description |
|--------|--------|-------------|
| `bip352_kat` | `tests/test_bip352_kat.cpp` | BIP-352 known-answer tests: output creation, roundtrip, rejection, index isolation |

---

## GPU Integration Tests

| Target | Source | Description |
|--------|--------|-------------|
| `gpu_bip352_scan` | `tests/test_gpu_bip352_scan.cpp` | GPU BIP-352 silent-payment scan: CPU plan + GPU batch scan round-trip |
| `gpu_ecdsa_snark_witness` | `tests/test_gpu_ecdsa_snark_witness.cpp` | GPU ECDSA SNARK witness generation: limb roundtrip, bounds, batch macro |

---

## Python Audit Harnesses (`ci/`)

| Target | Source | Description |
|--------|--------|-------------|
| `py_dev_bug_scan` | `ci/dev_bug_scanner.py` | Static analysis for common API misuse patterns |
| `py_differential_crossimpl` | `ci/differential_cross_impl.py` | Differential cross-implementation testing via Python FFI |
| `py_hot_path_alloc_scan` | `ci/hot_path_alloc_scanner.py` | Hot-path allocation scanner: heap allocations in critical paths |
| `py_invalid_input_grammar` | `ci/invalid_input_grammar.py` | Grammar-guided malformed input generation |
| `py_nonce_bias` | `ci/nonce_bias_detector.py` | Statistical analysis of ECDSA nonce distribution |
| `py_rfc6979_spec` | `ci/rfc6979_spec_verifier.py` | RFC 6979 deterministic nonce KAT compliance |
| `py_semantic_props` | `ci/semantic_props.py` | Algebraic invariant property-based checking |
| `py_stateful_sequences` | `ci/stateful_sequences.py` | Multi-call API state machine exploration |

---

## Wycheproof Test Vectors

| Target | Source | Description |
|--------|--------|-------------|
| `wycheproof_chacha20_poly1305` | `tests/test_wycheproof_chacha20_poly1305.cpp` | Wycheproof ChaCha20-Poly1305 AEAD test vectors |
| `wycheproof_hkdf_sha256` | `tests/test_wycheproof_hkdf_sha256.cpp` | Wycheproof HKDF-SHA256 KDF test vectors |
| `wycheproof_hmac_sha256` | `tests/test_wycheproof_hmac_sha256.cpp` | Wycheproof HMAC-SHA256 MAC test vectors |

---

## Remaining ABI Surface Edge-Case Coverage (v3.23+ §I/§O)

| ID  | Functions | NULL args | Invalid inputs | Valid round-trip |
|-----|-----------|-----------|----------------|------------------|
| I.1 | `ufsecp_ctx_clone`, `ufsecp_last_error_msg`, `ufsecp_last_error` | [OK] | [OK] (error state) | [OK] (independent clone) |
| I.2 | `ufsecp_pubkey_parse`, `ufsecp_pubkey_create_uncompressed` | [OK] | [OK] (bad len, bad prefix, zero key) | [OK] (uncompressed→compressed normalisation) |
| I.3 | `ufsecp_ecdsa_sign_recoverable`, `ufsecp_ecdsa_recover` | [OK] | [OK] (zero key, bad recid) | [OK] (recovered pubkey matches original) |
| I.4 | `ufsecp_ecdsa_sign_verified`, `ufsecp_schnorr_sign_verified` | [OK] | [OK] (zero key) | [OK] (outputs verify via _verify counterpart) |
| I.5 | `ufsecp_schnorr_batch_verify`, `ufsecp_ecdsa_batch_verify`, `ufsecp_schnorr_batch_identify_invalid`, `ufsecp_ecdsa_batch_identify_invalid` | [OK] | [OK] (tampered sig) | [OK] (valid entry verifies; identify_invalid returns correct index) |

---

## Legend

| Symbol | Meaning |
|--------|---------|
| [OK] | Tested with passing checks |
| [!] | Partial or no coverage |
| [FAIL] | Not implemented |
| -- | Not applicable |

---

*UltrafastSecp256k1 v4.0.0 -- Test Coverage Matrix*


---

## Additional Exploit PoC Tests (batch 2 — v3.68+)

| Target | Source | Description |
|--------|--------|-------------|
| `exploit_batch_verify_low_s` | `audit/test_exploit_batch_verify_low_s.cpp` | Batch verify rejects high-S signatures (BIP-62 low-S enforcement) |
| `exploit_bech32_underflow` | `audit/test_exploit_bech32_underflow.cpp` | Bech32 decoder integer-underflow PoC (CVE class) |
| `exploit_binding_retval` | `audit/test_exploit_binding_retval.cpp` | FFI bindings must not silently ignore error return values |
| `exploit_bip352_batch_correctness` | `audit/test_exploit_bip352_batch_correctness.cpp` | BIP-352 scan plan batch correctness vs CPU reference |
| `exploit_bitcoin_core_rgrinding` | `audit/test_exploit_bitcoin_core_rgrinding.cpp` | R-grinding loop correctness (Bitcoin Core ndata/grind interaction) |
| `exploit_dark_skippy_exfil` | `audit/test_exploit_dark_skippy_exfil.cpp` | Dark Skippy nonce exfiltration — signing path must not leak key bits |
| `exploit_differential_openssl` | `audit/test_exploit_differential_openssl.cpp` | Differential ECDSA parity vs OpenSSL reference |
| `exploit_eth_signing_ct` | `audit/test_exploit_eth_signing_ct.cpp` | Ethereum signing path constant-time verification |
| `exploit_fiat_shamir_frozen_heart` | `audit/test_exploit_fiat_shamir_frozen_heart.cpp` | Frozen Heart (CVE-2023-33242): Fiat-Shamir transcript must commit to all prover messages |
| `exploit_gpu_secret_erase` | `audit/test_exploit_gpu_secret_erase.cpp` | GPU device-side secret erase on error paths |
| `exploit_hertzbleed_scalar_blind` | `audit/test_exploit_hertzbleed_scalar_blind.cpp` | Hertzbleed DVFS timing — blinded scalar must not vary with CPU frequency |
| `exploit_jni_retval_ignored` | `audit/test_exploit_jni_retval_ignored.cpp` | JNI/Java bindings: error return value must propagate, not be silently dropped |
| `exploit_libsecp_eckey_api` | `audit/test_exploit_libsecp_eckey_api.cpp` | Libsecp256k1 eckey API parity: all 17 ECKEY-* edge cases |
| `exploit_monolith_split` | `audit/test_exploit_monolith_split.cpp` | Monolith split (B-04): impl domain files must match monolith behavior |
| `exploit_pippenger_batch_regression` | `audit/test_exploit_pippenger_batch_regression.cpp` | Pippenger MSM regression: batch result must match naive multi-scalar |
| `exploit_recoverable_sign_ct` | `audit/test_exploit_recoverable_sign_ct.cpp` | Recoverable ECDSA signing constant-time on secret key and nonce |
| `exploit_thread_unsafe_lazy_init` | `audit/test_exploit_thread_unsafe_lazy_init.cpp` | Lazy initialisation race — table build must be thread-safe |
| `exploit_wallet_sign_ct` | `audit/test_exploit_wallet_sign_ct.cpp` | Wallet ECDSA/Schnorr signing CT path on secret key |
| `regression_bip324_session` | `audit/test_regression_bip324_session.cpp` | BIP-324 transport session key agreement regression |
| `regression_cuda_pool_cap` | `audit/test_regression_cuda_pool_cap.cpp` | CUDA memory pool capacity regression (RTX-series OOM) |
| `regression_musig2_verify` | `audit/test_regression_musig2_verify.cpp` | MuSig2 partial signature verification regression |
| `regression_z_fe_nonzero` | `audit/test_regression_z_fe_nonzero.cpp` | Jacobian Z-coordinate non-zero invariant regression |
| `regression_adaptor_binding_domain` | `audit/test_regression_adaptor_binding_domain.cpp` | SEC-010: ecdsa_adaptor_binding BIP-340 tagged hash domain separation (ADB-1..6) |

## Upstream Libsecp256k1 Parity Tests (batch 3 — v3.69+)

| Target | Source | Description |
|--------|--------|-------------|
| `exploit_pubkey_cmp` | `audit/test_exploit_pubkey_cmp.cpp` | Pubkey comparison ordering (GAP-3) — ports upstream run_pubkey_comparison; lexicographic ordering correctness for MuSig2 key aggregation |
| `exploit_pubkey_sort` | `audit/test_exploit_pubkey_sort.cpp` | Pubkey sort + MuSig2 BIP-327 ordering (GAP-4) — ports upstream run_pubkey_sort; all 120 permutations must yield same sorted order |
| `exploit_alloc_bounds` | `audit/test_exploit_alloc_bounds.cpp` | Allocation boundary batch verify (GAP-1) — ports upstream run_scratch_tests; count=0/1/64/128/192 edge cases and fail-closed on tampered sig |
| `exploit_hsort` | `audit/test_exploit_hsort.cpp` | Heap sort / batch ordering (GAP-2) — ports upstream run_hsort_tests; batch verify must be order-independent (forward/reversed/shuffled inputs) |
| `exploit_wnaf` | `audit/test_exploit_wnaf.cpp` | wNAF window decomposition boundaries (GAP-5) — ports upstream run_wnaf_tests; sk=1→G, sk=n-1→-G, GLV split, alternating-bit and FF scalars |
| `exploit_int128` | `audit/test_exploit_int128.cpp` | 128-bit field arithmetic boundaries (GAP-7) — ports upstream run_int128_tests; field identity, p-1 edge, 2^64 carry boundaries, commutativity |

## Upstream Libsecp256k1 Parity Tests (batch 4 — v3.70+)

| Target | Source | Description |
|--------|--------|-------------|
| `exploit_scratch` | `audit/test_exploit_scratch.cpp` | Scratch allocator risk surface — upstream run_scratch_tests; scratch_space API not in shim (confirmed), tests via batch lifecycle + context stress |
| `exploit_xoshiro` | `audit/test_exploit_xoshiro.cpp` | xoshiro256** PRNG context randomization — upstream run_xoshiro256pp_tests; KAT seed from upstream, behavioral properties via context_randomize |
| `exploit_bugbounty_20260505` | `audit/test_exploit_bugbounty_20260505.cpp` | 2026-05-05 bug bounty red-team round 2: BB-01 FROST n_signers<threshold, BB-02 zero signing share, BB-03 shim ctx_can_sign, BB-04 low-S≠even proof, BB-05 BIP32 depth guard, BB-06 ABI low-S invariant |
| `exploit_redteam_round3_20260505` | `audit/test_exploit_redteam_round3_20260505.cpp` | 2026-05-05 red-team round 3: RR3-01/02 MuSig2 secnonce not zeroed on error paths (BUG-1), RR3-03/04 FROST nonce not zeroed on early exits (BUG-2), RR3-05 last_error_msg thread_local path (BUG-4), RR3-06 MuSig2 keyagg LE32 round-trip (BUG-6) |
