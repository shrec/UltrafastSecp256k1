// ============================================================================
// Unified Audit Runner -- UltrafastSecp256k1
// ============================================================================
//
// Unified self-audit application. Single binary for all platforms.
// Build, run, validate all tests, save report.
//
// Single binary that runs ALL library tests and produces a structured
// JSON + text audit report. Build once, run on any platform.
//
// Usage:
//   unified_audit_runner              # run all tests, write report
//   unified_audit_runner --json-only  # suppress console, write JSON only
//   unified_audit_runner --report-dir <dir>  # write reports to <dir>
//
// Generates:
//   audit_report.json   -- machine-readable structured result
//   audit_report.txt    -- human-readable summary
// ============================================================================

#ifndef UNIFIED_AUDIT_RUNNER
#define UNIFIED_AUDIT_RUNNER  // Guard standalone main() in test modules
#endif

#include "secp256k1/selftest.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <chrono>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#include <sys/utsname.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#endif

// Library version (injected by CMake from VERSION.txt)
#if __has_include("secp256k1/version.hpp")
#include "secp256k1/version.hpp"
#endif
#include "secp256k1/precompute.hpp"
#ifndef SECP256K1_VERSION_STRING
#define SECP256K1_VERSION_STRING "unknown"
#endif

// Git hash (injected at compile time via -DGIT_HASH=...)
#ifndef GIT_HASH
#define GIT_HASH "unknown"
#endif

// Audit framework version (bump when report schema changes)
static constexpr const char* AUDIT_FRAMEWORK_VERSION = "2.0.0";

using namespace secp256k1::fast;

static void reset_fixed_base_state_for_module() {
    FixedBaseConfig cfg{};
    cfg.window_bits = 15U;
    cfg.enable_glv = false;
    cfg.use_jsf = false;
    cfg.adaptive_glv = false;
    cfg.use_cache = false;
    cfg.cache_path.clear();
    cfg.cache_dir.clear();
    cfg.max_windows_to_load = 0U;
    configure_fixed_base(cfg);
}

// ============================================================================
// Forward declarations -- selftest modules (from run_selftest.cpp sources)
// ============================================================================
int test_large_scalar_multiplication_run();
int test_mul_run();
int test_arithmetic_correctness_run();
int test_ct_run();
int test_ct_equivalence_run();
int test_ecdsa_schnorr_run();
int test_multiscalar_batch_run();
int test_bip32_run();
int test_bip32_vectors_run();
int test_bip39_run();
int test_musig2_run();
int test_ecdh_recovery_taproot_run();
int test_edge_cases_run();
int test_v4_features_run();
int test_coins_run();
int test_batch_add_affine_run();
int test_hash_accel_run();
int run_exhaustive_tests();
int test_comprehensive_run();
int test_bip340_vectors_run();
int test_rfc6979_vectors_run();
int test_ecc_properties_run();

// ============================================================================
// Forward declarations -- additional standalone test _run() functions
// ============================================================================
int test_carry_propagation_run();
int test_fault_injection_run();
int test_fiat_crypto_vectors_run();
int test_cross_platform_kat_run();
int test_debug_invariants_run();
int test_abi_gate_run();
int test_ct_sidechannel_smoke_run();
int test_differential_run();
int test_bip340_strict_run();

// ============================================================================
// Forward declarations -- MuSig2 / FROST protocol tests
// ============================================================================
int test_musig2_frost_protocol_run();
int test_musig2_frost_advanced_run();
int test_frost_kat_run();
int test_musig2_bip327_vectors_run();

// ============================================================================
// Forward declarations -- Cross-ABI / FFI round-trip tests
// ============================================================================
int test_ffi_round_trip_run();
int test_adversarial_protocol_run();
int test_ecies_regression_run();

// ============================================================================
// Forward declarations -- GPU ABI tests (no hardware required for null-guard paths)
// ============================================================================
int test_gpu_host_api_negative_run(); // NULL guards, invalid backend/device, error strings
int test_gpu_abi_gate_run();          // Discovery, lifecycle, ops-if-available

// ============================================================================
// Forward declarations -- adversarial / fuzz tests
// ============================================================================
int test_audit_fuzz_run();
int test_fuzz_parsers_run();
int test_fuzz_address_bip32_ffi_run();
int test_fuzz_musig2_frost_run();
int test_libfuzzer_unified_run();  // deterministic LibFuzzer harness regression suite
int test_mutation_kill_rate_run(); // mutation kill-rate tracker (advisory: needs Python)
int test_exploit_mutation_residue_run(); // mutation residue exploit vectors (wNAF OOB + divsteps mask)
int test_mutation_artifact_scan_run();   // source-file integrity scanner for mutation artifacts
int test_exploit_metal_field_reduce_run(); // Metal field_reduce_512 acc[8] truncation (issue #226)
int test_exploit_network_validation_bypass_run(); // Network selector validation bypass (address/WIF ABI)
int test_exploit_ecdsa_half_half_nonce_run();     // ePrint 2023/841 half-half nonce key recovery
int test_exploit_ecdsa_nonce_modular_bias_run();  // CVE-2024-31497 nonce modular reduction bias
int test_exploit_ecdsa_differential_fault_run();  // ePrint 2017/975 differential fault on RFC 6979
int test_exploit_eucleak_inversion_timing_run();   // ePrint 2024/1380 EUCLEAK non-CT inversion
int test_exploit_ecdsa_cross_key_nonce_reuse_run(); // ePrint 2025/654 cross-key nonce reuse cascade
int test_exploit_schnorr_hash_order_run();          // ePrint 2025/1846 Fiat-Shamir hash order
int test_exploit_zvp_glv_dcp_multiscalar_run();     // ePrint 2025/076 ZVP-DCP on GLV multiscalar
int test_exploit_lattice_sieve_hnp_run();            // ePrint 2024/296 lattice sieve HNP sub-bit
int test_exploit_deterministic_sig_dfa_run();        // ePrint 2017/975 DFA on RFC 6979 det. sigs
int test_exploit_sign_type_confusion_kreuse_run();   // CVE-2024-49364/49365/2022-41340 type confusion k-reuse
int test_exploit_ros_concurrent_schnorr_run();       // ePrint 2020/945 ROS concurrent Schnorr
int test_exploit_cross_protocol_kreuse_run();         // Original: Cross-Protocol Key Reuse (CPK-1..5)
int test_exploit_tagged_hash_ext_run();               // Original: BIP-340 Tagged SHA-256 Length Extension (TAGEXT-1..5)
int test_exploit_wnaf_cache_ampl_run();               // Original: wNAF Window-18 Cache Amplification (WCACHE-1..5)
int test_exploit_musig2_fingerprint_collision_run();  // Original: MuSig2 KeyAgg Fingerprint Collision (FPC-1..6)
int test_exploit_blinding_recovery_hnp_run();         // Original: Context Blinding Recovery via HNP (BLIND-1..7)
int test_exploit_shim_noncefp_bypass_run();           // N6: Shim noncefp Callback Bypass (NONCEFP-1..5)
int test_exploit_encoding_memory_corruption_run();    // N7: Encoding Memory Corruption (ENCORR-1..6)
int test_exploit_batch_verify_malleability_run();     // N8: Batch Verify Malleability (BVM-1..6)
int test_exploit_thread_local_blinding_run();         // N9: Thread-Local Blinding State Race (TLB-1..4)
int test_exploit_hedged_return_value_run();           // N10: Hedged Sign Return-Value Silence (HEDGED-1..4)
int test_exploit_gpu_memory_safety_run();             // N11: GPU Kernel Memory Safety (GPU-1..5)
int test_exploit_rs_zero_check_run();                 // N12: ECDSA r,s Zero Check Gap (RZERO-1..5)
int test_exploit_bip352_address_collision_run();      // N13: BIP-352 Address Collision (SP-1..4)
int test_exploit_frost_weak_binding_run();            // ePrint 2026/075, 2025/1001 FROST weak binding
int test_exploit_blind_spa_cmov_leak_run();           // ePrint 2024/589, 2025/935 Blind SPA + cmov leak
int test_exploit_ectester_point_validation_run();     // ePrint 2025/1293 ECTester point validation
int test_exploit_ros_dimensional_erosion_run();       // ePrint 2025/306, 2025/1353 eROS + BZ blind sig
int test_exploit_ecdsa_batch_verify_rand_run();       // ePrint 2026/663 modified ECDSA batch verify
int test_exploit_bip324_aead_forgery_run();           // BIP-324 AEAD forgery boundary attacks
int test_exploit_frost_rogue_key_run();               // ePrint 2020/852 FROST rogue-key
int test_exploit_musig2_partial_forgery_run();        // ePrint 2020/1261 MuSig2 partial sig forgery
int test_exploit_adaptor_extraction_soundness_run();  // ePrint 2020/476 adaptor extraction soundness
int test_exploit_ecdh_twist_injection_run();          // ePrint 2015/1233 ECDH twist injection
int test_exploit_schnorr_batch_inflation_run();       // BIP-340 Schnorr batch inflation
int test_cryptol_specs_run();      // Cryptol formal spec property check (blocking: apt-get install cryptol)
int test_exploit_safegcd_divsteps_run();              // Bernstein-Yang SafeGCD divstep count + correctness
int test_exploit_ecdsa_pmn_wraparound_run();          // ECDSA PMN wraparound: r ∈ [n,p) constant + logic (2026-05-05)
int test_exploit_custom_nonce_injection_run();        // RFC 6979 nonce edge cases (null/zero/n/n-1)

// ============================================================================
// Forward declarations -- Wycheproof & batch-randomness (Track I3, I6-3)
// ============================================================================
int test_wycheproof_ecdsa_run();
int test_wycheproof_ecdh_run();
int test_batch_randomness_run();

// ============================================================================
// Forward declarations -- CT formal verification & independent reference linkage (I5)
// ============================================================================
int test_ct_verif_formal_run();
int test_fiat_crypto_linkage_run();

// ============================================================================
// Forward declarations -- deep audit modules
// ============================================================================
int audit_field_run();       // Section I.1: Field Fp correctness
int audit_scalar_run();      // Section I.2: Scalar Zn correctness
int audit_point_run();       // Section I.3: Point & signature correctness
int audit_ct_run();          // Section II:  CT & side-channel deep audit
int audit_integration_run(); // Section VI:  Integration & cross-protocol
int audit_security_run();    // Section V:   Security hardening
int audit_perf_run();        // Section IV:  Performance validation

// ============================================================================
// Forward declarations -- field representation tests
// ============================================================================
#ifdef __SIZEOF_INT128__
int test_field_52_main();   // 5x52 lazy-reduction (requires __uint128_t)
#endif
int test_field_26_main();   // 10x26 lazy-reduction

// ============================================================================
// Forward declarations -- diagnostics
// ============================================================================
int diag_scalar_mul_run();

// ============================================================================
// Forward declarations -- ZK proof layer
// ============================================================================
int audit_zk_run();

// ============================================================================
// Forward declarations -- Specification oracle & invariant monitor
// ============================================================================
int test_secp256k1_spec_run();    // SEC2 v2.0 curve constant oracle
int audit_invariants_run();       // Post-operation on-curve invariant checker
int test_c_abi_negative_run();    // C ABI null/bad-input contract tests
int test_c_abi_thread_stress_run(); // C ABI thread stress using one ctx per thread

// ============================================================================
// Forward declarations -- Security audit modules (new)
// ============================================================================
int audit_secure_erase_run();       // Secure memory erasure verification
int audit_ct_namespace_run();       // CT namespace discipline (source-level)
int test_kat_all_operations_run();  // KAT for ops not in standard vector suite
int test_nonce_uniqueness_run();    // RFC 6979 nonce determinism + uniqueness
int test_parse_strictness_run();    // Public parse path strictness audit

// ============================================================================
// Forward declarations -- Ethereum (conditional)
// ============================================================================
#ifdef SECP256K1_BUILD_ETHEREUM
int test_ethereum_run();
#endif

// ============================================================================
// Forward declarations -- Exploit PoC tests
// ============================================================================
int test_exploit_adaptor_extended_run();
int test_exploit_adaptor_parity_run();
int test_exploit_address_encoding_run();
int test_exploit_address_generation_run();
int test_exploit_bech32_underflow_run();
int test_exploit_aead_integrity_run();
int test_exploit_backend_divergence_run();
int test_exploit_batch_schnorr_run();
int test_exploit_batch_schnorr_forge_run();
int test_exploit_batch_soundness_run();
int test_exploit_batch_verify_correctness_run();
int test_exploit_batch_verify_poison_run();
int test_exploit_bip143_sighash_run();
int test_exploit_bip144_serialization_run();
int test_exploit_bip324_counter_desync_run();
int test_exploit_bip324_session_run();
int test_exploit_bip324_transcript_splice_run();
int test_exploit_bip32_ckd_hardened_run();
int test_exploit_bip32_depth_run();
int test_exploit_bip32_derivation_run();
int test_exploit_bip32_parent_fingerprint_confusion_run();
int test_exploit_bip32_path_overflow_run();
int test_exploit_bip39_entropy_run();
int test_exploit_bip39_mnemonic_run();
int test_exploit_bip39_nfkd_run();
int test_exploit_bitcoin_message_signing_run();
int test_exploit_chacha20_kat_run();
int test_exploit_kat_corpus_run();
int test_exploit_primitive_kat_run();
int test_exploit_chacha20_nonce_reuse_run();
int test_exploit_chacha20_poly1305_run();
int test_exploit_coin_hd_derivation_run();
int test_exploit_ct_fast_equivalence_run();
int test_exploit_ct_recov_run();
int test_exploit_ct_systematic_run();
int test_exploit_ctx_clone_run();
int test_exploit_ctx_lifecycle_hostile_run();
int test_exploit_der_parsing_differential_run();
int test_exploit_ecdh_run();
int test_exploit_ecdh_degenerate_run();
int test_exploit_ecdh_variants_run();
int test_exploit_ecdsa_der_confusion_run();
int test_exploit_ecdsa_edge_cases_run();
int test_exploit_ecdsa_malleability_run();
int test_exploit_batch_verify_low_s_run();  // batch low-S regression (2026-04-27)
int test_exploit_binding_retval_run();       // ABI return-value coverage (2026-04-27)
int test_exploit_ecdsa_recovery_run();
int test_exploit_ecdsa_rfc6979_kat_run();
int test_exploit_ecies_auth_run();
int test_exploit_ecies_encryption_run();
int test_exploit_ecies_envelope_confusion_run();
int test_exploit_ecies_roundtrip_run();
int test_exploit_ecrecover_confusion_run();
int test_exploit_ellswift_run();
int test_exploit_ellswift_ecdh_run();
int test_exploit_ethereum_differential_run();
int test_exploit_eth_signing_run();
int test_exploit_field_arithmetic_run();
int test_exploit_field_boundary_exhaustive_run();
int test_exploit_frost_binding_factor_mismatch_run();
int test_exploit_frost_byzantine_run();
int test_exploit_frost_commitment_reuse_run();
int test_exploit_frost_dkg_run();
int test_exploit_frost_index_run();
int test_exploit_frost_lagrange_duplicate_run();
int test_exploit_frost_participant_zero_run();
int test_exploit_frost_signing_run();
int test_exploit_frost_threshold_degenerate_run();
int test_exploit_glv_endomorphism_run();
int test_exploit_glv_kat_run();
int test_exploit_gpu_cpu_divergence_run();
int test_exploit_gpu_host_api_shape_run();
int test_exploit_hedged_nonce_bias_run();
int test_exploit_hkdf_kat_run();
int test_exploit_hkdf_security_run();
int test_infinity_edge_cases_run();
int test_exploit_invalid_curve_twist_run();
int test_exploit_keccak256_kat_run();
int test_exploit_multiscalar_run();
int test_exploit_musig2_run();
int test_exploit_musig2_key_agg_run();
int test_exploit_musig2_nonce_reuse_run();
int test_exploit_musig2_ordering_run();
int test_exploit_musig2_transcript_fork_run();
int test_exploit_pedersen_adversarial_run();
int test_exploit_pedersen_homomorphism_run();
int test_exploit_pedersen_switch_misuse_run();
int test_exploit_pippenger_msm_run();
int test_exploit_point_group_law_run();
int test_exploit_point_serialization_run();
int test_exploit_private_key_run();
int test_exploit_recovery_extended_run();
int test_exploit_ripemd160_kat_run();
int test_exploit_scalar_group_order_run();
int test_exploit_scalar_invariants_run();
int test_exploit_scalar_systematic_run();
int test_exploit_schnorr_bip340_kat_run();
int test_exploit_schnorr_edge_cases_run();
int test_exploit_schnorr_forgery_vectors_run();
int test_exploit_schnorr_xonly_parity_confusion_run();
int test_exploit_seckey_arith_run();
int test_exploit_seckey_tweak_cancel_run();
int test_exploit_segwit_encoding_run();
int test_exploit_selftest_api_run();
int test_exploit_sha256_kat_run();
int test_exploit_sha512_kat_run();
int test_exploit_sha_kat_run();
int test_exploit_silent_payment_confusion_run();
int test_exploit_taproot_merkle_path_alias_run();
int test_exploit_taproot_scripts_run();
int test_exploit_taproot_tweak_run();
int test_exploit_wallet_api_run();
int test_exploit_wallet_cross_domain_replay_run();
int test_exploit_zk_adversarial_run();
int test_exploit_zk_proofs_run();
int test_exploit_fe_set_b32_limit_uninit_run();
int test_exploit_zk_new_schemes_run();
int test_exploit_foreign_field_plonk_run();

// ============================================================================
// Forward declarations -- Feature Exploit PoC tests (P2SH, BIP-85, BIP-340 var,
//                          BIP-322, BIP-158 GCS, PSBT, Descriptors)
// ============================================================================
int test_exploit_p2sh_address_confusion_run();
int test_exploit_bip85_path_collision_run();
int test_exploit_schnorr_msg_length_confusion_run();
int test_exploit_bip322_type_confusion_run();
int test_exploit_gcs_false_positive_run();
int test_exploit_psbt_input_confusion_run();
int test_exploit_descriptor_injection_run();
int test_exploit_pubkey_arith_run();
int test_exploit_batch_sign_run();
int test_exploit_ecdsa_nonce_reuse_run();
int test_exploit_cross_scheme_pubkey_run();
int test_exploit_wif_security_run();
int test_exploit_buffer_type_confusion_run();
int test_exploit_differential_libsecp_run();
int test_exploit_differential_openssl_run();
int test_exploit_bip352_scan_dos_run();
int test_exploit_taproot_commitment_adversarial_run();
int test_exploit_bip352_parity_confusion_run();
int test_exploit_bip352_batch_correctness_run();
int test_exploit_rfc6979_truncation_bias_run();
int test_exploit_binding_adversarial_api_run();
int test_exploit_quantum_exposure_run();
int test_exploit_ecies_ephemeral_reuse_run();
int test_exploit_address_prefix_collision_run();
int test_exploit_binding_invalid_curve_run();
int test_exploit_frost_ct_nonce_run();
int test_exploit_frost_participant_set_malleability_run();
int test_exploit_musig2_parallel_session_cross_run();
int test_exploit_ecdh_zvp_glv_static_run();
int test_exploit_frost_adaptive_corruption_run();
int test_exploit_ecdsa_fault_injection_run();
int test_exploit_cache_sidechannel_amplification_run();
int test_exploit_ladderleak_subbit_nonce_run();
int test_exploit_minerva_noisy_hnp_run();
int test_exploit_hertzbleed_dvfs_timing_run();
int test_exploit_biased_nonce_chain_scan_run();
int test_exploit_kr_ecdsa_buff_binding_run();
int test_exploit_jni_retval_ignored_run();     // JNI return-value ignored (RVI-1..8) — 2026-04-23
int test_exploit_dark_skippy_exfil_run();      // Dark Skippy nonce exfiltration (DS-1..8) — ePrint 2024/1225
int test_exploit_fiat_shamir_frozen_heart_run(); // Frozen Heart ZK Fiat-Shamir incomplete binding (FH-1..10) — ePrint 2022/411
int test_exploit_hertzbleed_scalar_blind_run(); // Hertzbleed scalar blinding Hamming-weight defence (SB-1..9) — ePrint 2022/823
int test_exploit_thread_unsafe_lazy_init_run(); // Thread-safe gen_fb_table init: std::once_flag regression (TIR-1..5) — 2026-04-27
int test_regression_z_fe_nonzero_run();         // z_fe_nonzero | vs & correctness regression (ZFN-1..6) — 2026-04-27
int test_regression_cuda_pool_cap_run();         // CUDA pool minimum-capacity rounding regression (CAP-1..4) — 2026-04-27
int test_regression_musig2_verify_run();        // musig2_partial_verify OOB + infinity-nonce regression (MVV-1..6) — 2026-04-23
int test_regression_bip324_session_run();       // Bip324Session sk-leak + destructor secure-erase regression (BPS-1..8) — 2026-04-23
int test_regression_exception_erase_run();      // Exception-path sk/ek/entropy leakage in C ABI layer (EPE-RAII + EPE-1..12) — 2026-04-23
int test_exploit_recoverable_sign_ct_run();     // C ABI recovery signing CT path (RCTX-1..8) — C-1 fix — 2026-04-27
int test_exploit_pippenger_batch_regression_run(); // Pippenger batch verify regression guard (PIPBATCH-1..8) — M-6 — 2026-04-27
int test_exploit_eth_signing_ct_run();          // Ethereum signing CT path (ETHCT-1..8) — B-01 fix — 2026-04-27
int test_exploit_wallet_sign_ct_run();          // Wallet sign_hash CT path (WALCT-1..8) — B-02 fix — 2026-04-27
int test_exploit_monolith_split_run();          // Monolith split integrity (MONO-1..12) — B-04 fix — 2026-04-27
int test_exploit_gpu_secret_erase_run();        // GPU secret erase + batch sign partial (B08-1..4, B09-1..5) — B-08/B-09 — 2026-04-27
int test_exploit_libsecp_eckey_api_run();       // libsecp256k1 EC key API compat (ECKEY-1..17) — L-01 — 2026-04-27
int test_exploit_bitcoin_core_rgrinding_run();  // Bitcoin Core R-grinding nonce pattern (RGRIND-1..8) — BC-01 — 2026-04-27
int test_exploit_pubkey_cmp_run();              // Pubkey comparison ordering (GAP-3) — upstream run_pubkey_comparison — 2026-04-28
int test_exploit_pubkey_sort_run();             // Pubkey sort + MuSig2 ordering (GAP-4) — upstream run_pubkey_sort — 2026-04-28
int test_exploit_alloc_bounds_run();            // Allocation boundary batch verify (GAP-1) — upstream run_scratch_tests — 2026-04-28
int test_exploit_hsort_run();                   // Heap sort / batch ordering (GAP-2) — upstream run_hsort_tests — 2026-04-28
int test_exploit_wnaf_run();                    // wNAF window decomposition boundaries (GAP-5) — upstream run_wnaf_tests — 2026-04-28
int test_exploit_int128_run();                  // 128-bit arithmetic field boundaries (GAP-7) — upstream run_int128_tests — 2026-04-28
int test_exploit_scratch_run();                 // Scratch allocator risk surface — upstream run_scratch_tests — 2026-04-28
int test_exploit_xoshiro_run();                 // xoshiro256** PRNG / context randomization — upstream run_xoshiro256pp_tests — 2026-04-28

// ============================================================================
// Forward declarations -- orphan PoCs previously built as CTest targets but
// not registered in unified_audit_runner. Wired in 2026-04-17 to close the
// Conversion Standard gap (docs/EXPLOIT_BACKLOG.md).
// ============================================================================
int test_exploit_bip32_child_key_attack_run();
int test_exploit_boundary_sentinels_run();
int test_exploit_buff_kr_ecdsa_run();
int test_exploit_ecdsa_affine_nonce_relation_run();
int test_exploit_ecdsa_r_overflow_run();
int test_exploit_ecdsa_sign_sentinels_run();
int test_exploit_ellswift_bad_scalar_ecdh_run();
int test_exploit_ellswift_xdh_overflow_run();
int test_exploit_frost_identifiable_abort_run();
int test_exploit_hash_algo_sig_isolation_run();
int test_exploit_minerva_cve_2024_23342_run();
int test_exploit_musig2_byzantine_multiparty_run();
int test_exploit_rfc6979_minerva_amplified_run();
int test_exploit_scalar_mul_run();
int test_exploit_schnorr_nonce_reuse_run();
int test_exploit_eip712_kat_run();

// ============================================================================
// Forward declarations -- BUG-001..008 fixes (2026-04-28 full audit)
// ============================================================================
int test_exploit_bug001_addr_overflow_run();  // BUG-001/007: addr off-by-one + p2sh null
int test_exploit_bug002_recovery_ct_run();    // BUG-002: recovery.cpp CT recid overflow
int test_exploit_bug003_normalize_ct_run();   // BUG-003/008: ECDSASignature::normalize() CT
int test_exploit_bug004_batch_failclosed_run(); // BUG-004: schnorr batch sign fail-closed (BFC-1..9)
int test_exploit_shim_der_bip66_run();        // HIGH-2/3: shim DER parser BIP-66 (DER66-1..8)
int test_exploit_shim_musig_secnonce_run();   // CRIT-1: shim MuSig2 secnonce reuse (MSN-1..6)
int test_exploit_shim_musig_ka_cap_run();     // RED-TEAM-009: ka_put DoS-cap fail-closed
int test_exploit_shim_recovery_null_arg_run(); // SHIM-005: NULL arg fail-closed (a3f56ae)

// ============================================================================
// Forward declarations -- 2026-05-01 Red Team Audit Fixes
// ============================================================================
int test_exploit_legacy_capi_key_parsing_run();
int test_exploit_legacy_capi_degenerate_sig_run();
int test_exploit_musig_unknown_signer_run();
int test_exploit_bchn_schnorr_strict_parsing_run();
int test_exploit_context_flag_bypass_run();
int test_exploit_metal_schnorr_aux_rand_run();
int test_exploit_metal_batch_failclosed_run();
int test_exploit_gpu_bip352_key_erase_run();
int test_exploit_metal_ecdh_key_erase_run();
int test_exploit_shim_pubkey_ct_run();   // P-01/02/03: shim pubkey CT generator mul (SPC-1..10)
int test_exploit_opencl_runner_key_erase_run();  // Q-01/02/03: OpenCL audit runner key erase (OCR-1..8)
int test_exploit_ecdsa_fast_path_isolation_run();  // FPI-1..10: ecdsa fast path not used in production

// ============================================================================
// Forward declarations -- ABI/CT/memory-safety regression guards
// ============================================================================
int test_exploit_abi_recoverable_schnorr_ct_regression_run(); // ABI recid+recov+schnorr_verified CT regressions
int test_exploit_frost_ocl_shim_bip32_ct_regression_run();   // FROST+OCL+shim+BIP32 CT regression guards
int test_exploit_musig2_nonce_erasure_le32_ecdh_run();        // secnonce/nonce erasure + LE32 + ECDH Y-parity

// ============================================================================
// Forward declarations -- 2026-05-02 Bitcoin Core PR security audit fixes
// ============================================================================
int test_regression_schnorr_ct_arithmetic_run();  // HIGH-03+HIGH-06: schnorr_sign CT arithmetic + r==0 check
int test_regression_musig2_zero_psig_run();       // CRIT-03: musig2 zero partial-sig → UFSECP_ERR_INTERNAL

// ============================================================================
// Forward declarations -- 2026-05-11 shim regression tests (SHIM-001/002/003 + DER r=0)
// ============================================================================
int test_shim_der_zero_r_run();   // DER parse r=0 rejection (shim vs libsecp divergence)
int test_shim_null_ctx_run();     // NULL context illegal-callback enforcement (SHIM-001/002/003)

// ============================================================================
// Forward declarations -- 2026-05-06 performance review fixes
// ============================================================================
int test_regression_pippenger_stale_used_run();  // BUG-01: Pippenger used[] stale across windows
int test_exploit_frost_secret_share_ct_run();    // SEC-01: FROST DKG non-CT generator mul on secret share
int test_regression_comb_gen_lockfree_run();     // CRIT-01: comb_gen_mul lock-free thread safety
// GPU RAII test requires GPU symbols; stub returns advisory-skip when GPU not built.
#if defined(SECP256K1_BUILD_GPU_AUDIT)
int test_regression_gpu_key_erase_raii_run();
#else
static inline int test_regression_gpu_key_erase_raii_run() { return 77; }
#endif
int test_regression_bip352_ct_varbase_run();      // CRIT-02: BIP-352 CT variable-base scalar mul
int test_regression_signing_ct_scalar_correctness_run(); // CT gen-mul, inv, cswap, Pippenger, BatchVerify
int test_regression_ct_fast_scalar_v01_run();            // V-01: fast::Scalar operator* timing guard (advisory)
int test_regression_schnorr_abi_edge_cases_run();        // TQ-005: Schnorr r==0/r>=p/s==0/s>=n ABI rejection
int test_regression_ct_mixed_add_magnitude_run();        // CA-mixed-add: point_add_mixed_complete magnitude contract
int test_regression_shim_static_ctx_run();              // ecf47967: g_static_ctx PERF-005 field alignment fix
int test_regression_ellswift_ct_path_run();              // CT-001: ellswift_create CT path + XDH round-trip
int test_regression_musig2_nonce_strict_run();           // CT-006: MuSig2 k1/k2 strict nonce parsing
int test_regression_bip32_private_key_strict_run();      // RT-011: BIP-32 private_key()/public_key() strict parse
int test_regression_shim_pubkey_sort_run();              // SHIM-012: pubkey_sort no-crash + correct order
int test_regression_shim_per_context_blinding_run();     // SHIM-001: per-context blinding semantics
int test_regression_musig2_session_token_run();          // SHIM-010: MuSig2 token-keyed map
int test_regression_musig2_signer_index_validation_run(); // SEC-007: signer_index cross-check (Rule 13)

// ============================================================================
// Report section IDs -- 9 audit categories
// ============================================================================
//   1. math_invariants   -- Mathematical Invariants (Fp, Zn, Group Laws)
//   2. ct_analysis       -- Constant-Time / Side-Channel Analysis
//   3. differential      -- Differential & Cross-Library Testing
//   4. standard_vectors  -- Standard Test Vectors (BIP-340, RFC-6979, BIP-32)
//   5. fuzzing           -- Fuzzing & Adversarial Attack Resilience
//   6. protocol_security -- Protocol Security (ECDSA, Schnorr, MuSig2, FROST)
//   7. memory_safety     -- ABI & Memory Safety (sanitizer, zeroization)
//   8. performance       -- Performance Validation & Regression
//   9. exploit_poc       -- Exploit PoC Security Probes
// ============================================================================

struct AuditModule {
    const char* id;           // short ID for JSON
    const char* name;         // human-readable name
    const char* section;      // one of 8 report sections
    int (*run)();             // returns 0=PASS, non-zero=FAIL
    bool advisory;            // if true, failure does not block audit verdict
};

// Section display names (Georgian + English)
struct SectionInfo {
    const char* id;
    const char* title_ka;     // Georgian
    const char* title_en;     // English
};

static const SectionInfo SECTIONS[] = {
    { "math_invariants",   "\xe1\x83\x9b\xe1\x83\x90\xe1\x83\x97\xe1\x83\x94\xe1\x83\x9b\xe1\x83\x90\xe1\x83\xa2\xe1\x83\x98\xe1\x83\x99\xe1\x83\xa3\xe1\x83\xa0\xe1\x83\x98 \xe1\x83\x98\xe1\x83\x9c\xe1\x83\x95\xe1\x83\x90\xe1\x83\xa0\xe1\x83\x98\xe1\x83\x90\xe1\x83\x9c\xe1\x83\xa2\xe1\x83\x94\xe1\x83\x91\xe1\x83\x98",
                           "Mathematical Invariants (Fp, Zn, Group Laws)" },
    { "ct_analysis",       "Constant-Time \xe1\x83\x90\xe1\x83\x9c\xe1\x83\x90\xe1\x83\x9a\xe1\x83\x98\xe1\x83\x96\xe1\x83\x98",
                           "Constant-Time & Side-Channel Analysis" },
    { "differential",      "\xe1\x83\x93\xe1\x83\x98\xe1\x83\xa4\xe1\x83\x94\xe1\x83\xa0\xe1\x83\x94\xe1\x83\x9c\xe1\x83\xaa\xe1\x83\x98\xe1\x83\x90\xe1\x83\x9a\xe1\x83\xa3\xe1\x83\xa0\xe1\x83\x98 \xe1\x83\xa2\xe1\x83\x94\xe1\x83\xa1\xe1\x83\xa2\xe1\x83\x98\xe1\x83\xa0\xe1\x83\x94\xe1\x83\x91\xe1\x83\x90",
                           "Differential & Cross-Library Testing" },
    { "standard_vectors",  "\xe1\x83\xa1\xe1\x83\xa2\xe1\x83\x90\xe1\x83\x9c\xe1\x83\x93\xe1\x83\x90\xe1\x83\xa0\xe1\x83\xa2\xe1\x83\xa3\xe1\x83\x9a\xe1\x83\x98 \xe1\x83\x95\xe1\x83\x94\xe1\x83\xa5\xe1\x83\xa2\xe1\x83\x9d\xe1\x83\xa0\xe1\x83\x94\xe1\x83\x91\xe1\x83\x98",
                           "Standard Test Vectors (BIP-340, RFC-6979, BIP-32)" },
    { "fuzzing",           "\xe1\x83\xa4\xe1\x83\x90\xe1\x83\x96\xe1\x83\x98\xe1\x83\x9c\xe1\x83\x92\xe1\x83\x98 & \xe1\x83\x90\xe1\x83\x93\xe1\x83\x95\xe1\x83\x94\xe1\x83\xa0\xe1\x83\xa1\xe1\x83\x90\xe1\x83\xa0\xe1\x83\x98",
                           "Fuzzing & Adversarial Attack Resilience" },
    { "protocol_security", "\xe1\x83\x9e\xe1\x83\xa0\xe1\x83\x9d\xe1\x83\xa2\xe1\x83\x9d\xe1\x83\x99\xe1\x83\x9d\xe1\x83\x9a\xe1\x83\x94\xe1\x83\x91\xe1\x83\x98\xe1\x83\xa1 \xe1\x83\xa3\xe1\x83\xa1\xe1\x83\x90\xe1\x83\xa4\xe1\x83\xa0\xe1\x83\x97\xe1\x83\xae\xe1\x83\x9d\xe1\x83\x94\xe1\x83\x91\xe1\x83\x90",
                           "Protocol Security (ECDSA, Schnorr, MuSig2, FROST)" },
    { "memory_safety",     "ABI & \xe1\x83\x9b\xe1\x83\x94\xe1\x83\xae\xe1\x83\xa1\xe1\x83\x98\xe1\x83\x94\xe1\x83\xa0\xe1\x83\x94\xe1\x83\x91\xe1\x83\x98\xe1\x83\xa1 \xe1\x83\xa3\xe1\x83\xa1\xe1\x83\x90\xe1\x83\xa4\xe1\x83\xa0\xe1\x83\x97\xe1\x83\xae\xe1\x83\x9d\xe1\x83\x94\xe1\x83\x91\xe1\x83\x90",
                           "ABI & Memory Safety (zeroization, hardening)" },
    { "performance",       "\xe1\x83\x9e\xe1\x83\x94\xe1\x83\xa0\xe1\x83\xa4\xe1\x83\x9d\xe1\x83\xa0\xe1\x83\x9b\xe1\x83\x90\xe1\x83\x9c\xe1\x83\xa1\xe1\x83\x98\xe1\x83\xa1 \xe1\x83\x95\xe1\x83\x90\xe1\x83\x9a\xe1\x83\x98\xe1\x83\x93\xe1\x83\x90\xe1\x83\xaa\xe1\x83\x98\xe1\x83\x90",
                           "Performance Validation & Regression" },
    { "exploit_poc",       "\xe1\x83\x94\xe1\x83\xa5\xe1\x83\xa1\xe1\x83\x9e\xe1\x83\x9a\xe1\x83\x9d\xe1\x83\x98\xe1\x83\xa2 PoC \xe1\x83\xa2\xe1\x83\x94\xe1\x83\xa1\xe1\x83\xa2\xe1\x83\x94\xe1\x83\x91\xe1\x83\x98",
                           "Exploit PoC Security Probes" },
};
static constexpr int NUM_SECTIONS = sizeof(SECTIONS) / sizeof(SECTIONS[0]);

static const AuditModule ALL_MODULES[] = {
    // ===================================================================
    // Section 1: Mathematical Invariants (Fp, Zn, Group Laws)
    // ===================================================================
    { "secp256k1_spec",    "SEC2 v2.0 curve constant oracle",              "math_invariants", test_secp256k1_spec_run, false },
    { "audit_invariants",  "Post-op invariant monitor (on-curve/normalized)","math_invariants", audit_invariants_run, false },
    { "audit_field",       "Field Fp deep audit (add/mul/inv/sqrt/batch)", "math_invariants", audit_field_run, false },
    { "audit_scalar",      "Scalar Zn deep audit (mod/GLV/edge/inv)",      "math_invariants", audit_scalar_run, false },
    { "audit_point",       "Point ops deep audit (Jac/affine/sigs)",       "math_invariants", audit_point_run, false },
    { "mul",               "Field & scalar arithmetic",                    "math_invariants", test_mul_run, false },
    { "arith_correct",     "Arithmetic correctness",                       "math_invariants", test_arithmetic_correctness_run, false },
    { "scalar_mul",        "Scalar multiplication",                        "math_invariants", test_large_scalar_multiplication_run, false },
    { "exhaustive",        "Exhaustive algebraic verification",            "math_invariants", run_exhaustive_tests, false },
    { "comprehensive",     "Comprehensive 500+ suite",                     "math_invariants", test_comprehensive_run, false },
    { "ecc_properties",    "ECC property-based invariants",                "math_invariants", test_ecc_properties_run, false },
    { "batch_add",         "Affine batch addition",                        "math_invariants", test_batch_add_affine_run, false },
    { "carry_propagation", "Carry chain stress (limb boundary)",           "math_invariants", test_carry_propagation_run, false },
#ifdef __SIZEOF_INT128__
    { "field_52",          "FieldElement52 (5x52) vs 4x64",               "math_invariants", test_field_52_main, false },
#endif
    { "field_26",          "FieldElement26 (10x26) vs 4x64",              "math_invariants", test_field_26_main, false },

    // ===================================================================
    // Section 2: Constant-Time / Side-Channel Analysis
    // ===================================================================
    { "audit_ct",          "CT deep audit (masks/cmov/cswap/timing)",      "ct_analysis",    audit_ct_run, false },
    { "ct",                "Constant-time layer",                          "ct_analysis",    test_ct_run, false },
    { "ct_equivalence",    "FAST == CT equivalence",                       "ct_analysis",    test_ct_equivalence_run, false },
    { "ct_sidechannel",    "Side-channel dudect (smoke)",                  "ct_analysis",    test_ct_sidechannel_smoke_run, true },
    { "ct_verif_formal",   "Formal CT verification (ctgrind/MSAN)",       "ct_analysis",    test_ct_verif_formal_run, true },
    { "diag_scalar_mul",   "CT scalar_mul vs fast (diagnostic)",           "ct_analysis",    diag_scalar_mul_run, false },

    // ===================================================================
    // Section 3: Differential & Cross-Library Testing
    // ===================================================================
    { "differential",      "Differential correctness",                     "differential",   test_differential_run, false },
    { "fiat_crypto",       "Independent reference golden vectors",         "differential",   test_fiat_crypto_vectors_run, false },
    { "fiat_crypto_link",  "Independent reference linkage (100%% parity)","differential",   test_fiat_crypto_linkage_run, false },
    { "cross_platform_kat","Cross-platform KAT",                          "differential",   test_cross_platform_kat_run, false },

    // ===================================================================
    // Section 4: Standard Test Vectors (BIP-340, RFC-6979, BIP-32)
    // ===================================================================
    { "bip340_vectors",    "BIP-340 official vectors",                     "standard_vectors", test_bip340_vectors_run, false },
    { "bip340_strict",     "BIP-340 strict encoding (non-canonical)",      "standard_vectors", test_bip340_strict_run, false },
    { "bip32_vectors",     "BIP-32 official vectors TV1-5",               "standard_vectors", test_bip32_vectors_run, false },
    { "rfc6979_vectors",   "RFC 6979 ECDSA vectors",                      "standard_vectors", test_rfc6979_vectors_run, false },
    { "frost_kat",         "FROST reference KAT vectors",                 "standard_vectors", test_frost_kat_run, false },
    { "musig2_bip327",     "MuSig2 BIP-327 reference vectors",            "standard_vectors", test_musig2_bip327_vectors_run, false },
    { "wycheproof_ecdsa",  "Wycheproof ECDSA secp256k1 vectors",          "standard_vectors", test_wycheproof_ecdsa_run, false },
    { "wycheproof_ecdh",   "Wycheproof ECDH secp256k1 vectors",           "standard_vectors", test_wycheproof_ecdh_run, false },

    // ===================================================================
    // Section 5: Fuzzing & Adversarial Attack Resilience
    // ===================================================================
    { "audit_fuzz",        "Adversarial fuzz (malform/edge)",              "fuzzing",        test_audit_fuzz_run, false },
    { "fuzz_parsers",      "Parser fuzz (DER/Schnorr/Pubkey)",            "fuzzing",        test_fuzz_parsers_run, false },
    { "fuzz_addr_bip32",   "Address/BIP32/FFI boundary fuzz",             "fuzzing",        test_fuzz_address_bip32_ffi_run, false },
    { "fuzz_musig2_frost", "Parser fuzz: MuSig2/FROST/Adaptor (15 probes)","fuzzing",        test_fuzz_musig2_frost_run, false },
    { "libfuzzer_unified", "LibFuzzer deterministic regression (6 parsers)","fuzzing",        test_libfuzzer_unified_run, false },
    { "mutation_kill_rate","Mutation kill-rate audit (advisory)",          "fuzzing",        test_mutation_kill_rate_run, true  },
    { "cryptol_specs",     "Cryptol formal spec — arithmetic primitives",  "formal_proof",   test_cryptol_specs_run, true  },
    { "fault_injection",   "Fault injection simulation",                   "fuzzing",        test_fault_injection_run, false },

    // ===================================================================
    // Section 6: Protocol Security (ECDSA, Schnorr, MuSig2, FROST)
    // ===================================================================
    { "ecdsa_schnorr",     "ECDSA + Schnorr",                             "protocol_security", test_ecdsa_schnorr_run, false },
    { "bip32",             "BIP-32 HD derivation",                        "protocol_security", test_bip32_run, false },
    { "bip39",             "BIP-39 mnemonic seed phrases",                "protocol_security", test_bip39_run, false },
    { "musig2",            "MuSig2",                                       "protocol_security", test_musig2_run, false },
    { "ecdh_recovery",     "ECDH + recovery + taproot",                   "protocol_security", test_ecdh_recovery_taproot_run, false },
    { "v4_features",       "v4 (Pedersen/FROST/etc)",                     "protocol_security", test_v4_features_run, false },
    { "coins",             "Coins layer",                                  "protocol_security", test_coins_run, false },
    { "musig2_frost",      "MuSig2 + FROST protocol suite",              "protocol_security", test_musig2_frost_protocol_run, false },
    { "musig2_frost_adv",  "MuSig2 + FROST advanced/adversar",           "protocol_security", test_musig2_frost_advanced_run, false },
    { "audit_integration", "Integration (ECDH/batch/cross-proto)",        "protocol_security", audit_integration_run, false },
    { "batch_randomness",  "Batch verify weight randomness audit",        "protocol_security", test_batch_randomness_run, false },
    { "audit_zk",          "ZK proofs (knowledge/DLEQ/Bulletproof range)","protocol_security", audit_zk_run, false },
#ifdef SECP256K1_BUILD_ETHEREUM
    { "ethereum",          "Ethereum signing layer (EIP-191/155/ecrecover)","protocol_security", test_ethereum_run, false },
#endif

    // ===================================================================
    // Section 7: ABI & Memory Safety (zeroization, hardening)
    // ===================================================================
    { "audit_security",    "Security hardening (zero/bitflip/nonce)",      "memory_safety",  audit_security_run, false },
    { "debug_invariants",  "Debug invariant assertions",                   "memory_safety",  test_debug_invariants_run, false },
    { "abi_gate",          "ABI version gate (compile-time)",              "memory_safety",  test_abi_gate_run, false },
    { "ffi_round_trip",    "Cross-ABI/FFI round-trip (ufsecp C API)",     "memory_safety",  test_ffi_round_trip_run, false },
    { "c_abi_negative",    "C ABI null/bad-key/bad-sig contract tests",   "memory_safety",  test_c_abi_negative_run, false },
    { "c_abi_thread_stress", "C ABI thread stress (one ctx per thread)",  "memory_safety",  test_c_abi_thread_stress_run, false },
    { "secure_erase",      "Secure memory erasure (volatile readback)",   "memory_safety",  audit_secure_erase_run, false },
    // advisory=true: returns ADVISORY_SKIP_CODE (77) when source tree is absent (e.g. out-of-tree build).
    { "ct_namespace",      "CT namespace discipline (source-level scan)", "memory_safety",  audit_ct_namespace_run, true },
    { "kat_all_ops",       "KAT: ECDH/WIF/P2PKH/P2WPKH/P2TR/hash/arith","standard_vectors", test_kat_all_operations_run, false },
    { "nonce_uniqueness",  "RFC 6979 nonce determinism + uniqueness",     "memory_safety",  test_nonce_uniqueness_run, false },
    { "parse_strictness",  "Public parse path strictness (malformed inputs)","memory_safety", test_parse_strictness_run, false },
    { "adversarial_proto", "Adversarial protocol & FFI hostile-caller",   "fuzzing",         test_adversarial_protocol_run, false },
    { "ecies_regression",  "ECIES regression + C ABI prefix enforce",   "fuzzing",         test_ecies_regression_run, false },
    { "gpu_api_negative",  "GPU C ABI null/invalid-backend/error paths", "memory_safety",   test_gpu_host_api_negative_run, false },
    { "gpu_abi_gate",      "GPU ABI discovery, lifecycle, ops-if-avail", "memory_safety",   test_gpu_abi_gate_run, false },

    // ===================================================================
    // Section 8: Performance Validation & Regression
    // ===================================================================
    { "hash_accel",        "Accelerated hashing",                          "performance",    test_hash_accel_run, false },
    { "edge_cases",         "Edge cases & coverage gaps",                  "math_invariants",test_edge_cases_run, false },
    { "multiscalar",       "Multi-scalar & batch verify",                  "performance",    test_multiscalar_batch_run, false },
    { "audit_perf",        "Performance smoke (sign/verify roundtrip)",    "performance",    audit_perf_run, false },

    // ===================================================================
    // Section 9: Exploit PoC Security Probes
    // ===================================================================
    { "exploit_adaptor_extended",       "Adaptor Signature Extended Security",          "exploit_poc", test_exploit_adaptor_extended_run, false },
    { "exploit_adaptor_parity",         "Adaptor Signature R.y Parity",                "exploit_poc", test_exploit_adaptor_parity_run, false },
    { "exploit_address_encoding",       "Bitcoin Address Encoding Security",            "exploit_poc", test_exploit_address_encoding_run, false },
    { "exploit_address_generation",     "Bitcoin Address Generation Security",          "exploit_poc", test_exploit_address_generation_run, false },
    { "exploit_bech32_underflow",       "bech32_decode SIZE_MAX Underflow Regression",  "exploit_poc", test_exploit_bech32_underflow_run, false },
    { "exploit_aead_integrity",         "AEAD ChaCha20-Poly1305 Integrity",            "exploit_poc", test_exploit_aead_integrity_run, false },
    { "exploit_backend_divergence",     "Backend Divergence Detection",                "exploit_poc", test_exploit_backend_divergence_run, false },
    { "exploit_batch_schnorr",          "Schnorr Batch Verification Soundness",        "exploit_poc", test_exploit_batch_schnorr_run, false },
    { "exploit_batch_schnorr_forge",    "Schnorr Batch Forge Detection",               "exploit_poc", test_exploit_batch_schnorr_forge_run, false },
    { "exploit_batch_soundness",        "Batch Signature Verification Soundness",      "exploit_poc", test_exploit_batch_soundness_run, false },
    { "exploit_batch_verify_correct",   "Schnorr Batch Verify Correctness",            "exploit_poc", test_exploit_batch_verify_correctness_run, false },
    { "exploit_batch_verify_poison",    "Batch Verification Poisoning",                "exploit_poc", test_exploit_batch_verify_poison_run, false },
    { "exploit_bip143_sighash",         "BIP-143 SegWit v0 Signature Hash",            "exploit_poc", test_exploit_bip143_sighash_run, false },
    { "exploit_bip144_serialization",   "BIP-144 Witness Tx Serialization",            "exploit_poc", test_exploit_bip144_serialization_run, false },
    { "exploit_bip324_counter_desync",  "BIP-324 Counter Desync via Forged Packet",    "exploit_poc", test_exploit_bip324_counter_desync_run, false },
    { "exploit_bip324_session",         "BIP-324 Encrypted Transport Session",         "exploit_poc", test_exploit_bip324_session_run, false },
    { "exploit_bip324_transcript_splice","BIP-324 Transcript Splice / Packet Mix",     "exploit_poc", test_exploit_bip324_transcript_splice_run, false },
    { "exploit_bip32_ckd_hardened",     "BIP-32 Hardened Derivation Edge Cases",       "exploit_poc", test_exploit_bip32_ckd_hardened_run, false },
    { "exploit_bip32_depth",            "BIP-32 Depth uint8_t Silent Overflow",        "exploit_poc", test_exploit_bip32_depth_run, false },
    { "exploit_bip32_derivation",       "BIP-32 HD Key Derivation",                    "exploit_poc", test_exploit_bip32_derivation_run, false },
    { "exploit_bip32_fingerprint",      "BIP-32 Parent Fingerprint Confusion",         "exploit_poc", test_exploit_bip32_parent_fingerprint_confusion_run, false },
    { "exploit_bip32_path_overflow",    "BIP-32 Path Parser Integer Overflow",         "exploit_poc", test_exploit_bip32_path_overflow_run, false },
    { "exploit_bip39_entropy",          "BIP-39 Mnemonic Security Properties",         "exploit_poc", test_exploit_bip39_entropy_run, false },
    { "exploit_bip39_mnemonic",         "BIP-39 Mnemonic Security Properties",         "exploit_poc", test_exploit_bip39_mnemonic_run, false },
    { "exploit_bip39_nfkd",             "BIP-39 NFKD Normalization Correctness",       "exploit_poc", test_exploit_bip39_nfkd_run, false },
    { "exploit_btc_msg_signing",        "Bitcoin Message Signing Security",             "exploit_poc", test_exploit_bitcoin_message_signing_run, false },
    { "exploit_chacha20_kat",           "ChaCha20-Poly1305 RFC 8439 KAT",              "exploit_poc", test_exploit_chacha20_kat_run, false },
    { "exploit_kat_corpus",             "KAT Corpus — runtime JSON loader (all layers)", "exploit_poc", test_exploit_kat_corpus_run, true  },
    { "exploit_primitive_kat",           "Primitive-layer KAT (bitops/QR/block/AEAD)",   "exploit_poc", test_exploit_primitive_kat_run, false },
    { "exploit_chacha20_nonce_reuse",   "ChaCha20-Poly1305 Nonce Reuse",               "exploit_poc", test_exploit_chacha20_nonce_reuse_run, false },
    { "exploit_chacha20_poly1305",      "ChaCha20-Poly1305 AEAD (RFC 8439)",           "exploit_poc", test_exploit_chacha20_poly1305_run, false },
    { "exploit_coin_hd_derivation",     "Multi-Coin BIP-44 HD Derivation",             "exploit_poc", test_exploit_coin_hd_derivation_run, false },
    { "exploit_ct_fast_equiv",          "CT vs Fast Signing Equivalence",              "exploit_poc", test_exploit_ct_fast_equivalence_run, false },
    { "exploit_ct_recov",              "CT ct::ecdsa_sign_recoverable",               "exploit_poc", test_exploit_ct_recov_run, false },
    { "exploit_ct_systematic",          "CT vs FAST Output Divergence",                "exploit_poc", test_exploit_ct_systematic_run, false },
    { "exploit_ctx_clone",              "Context Clone Isolation Security",            "exploit_poc", test_exploit_ctx_clone_run, false },
    { "exploit_ctx_lifecycle",          "Context / Precompute Lifecycle Abuse",         "exploit_poc", test_exploit_ctx_lifecycle_hostile_run, false },
    { "exploit_der_parse_diff",         "DER Parsing Differential",                    "exploit_poc", test_exploit_der_parsing_differential_run, false },
    { "exploit_ecdh",                   "ECDH Key Exchange Security",                  "exploit_poc", test_exploit_ecdh_run, false },
    { "exploit_ecdh_degenerate",        "ECDH Degenerate Input Handling",              "exploit_poc", test_exploit_ecdh_degenerate_run, false },
    { "exploit_ecdh_variants",          "ECDH Variants Security",                      "exploit_poc", test_exploit_ecdh_variants_run, false },
    { "exploit_ecdsa_der_confusion",    "ECDSA DER Encoding Confusion",                "exploit_poc", test_exploit_ecdsa_der_confusion_run, false },
    { "exploit_ecdsa_edge_cases",       "ECDSA Edge Cases and Normalization",          "exploit_poc", test_exploit_ecdsa_edge_cases_run, false },
    { "exploit_ecdsa_malleability",     "ECDSA Signature Malleability",                "exploit_poc", test_exploit_ecdsa_malleability_run, false },
    { "exploit_batch_verify_low_s",     "ECDSA Batch Verify Low-S Enforcement",        "exploit_poc", test_exploit_batch_verify_low_s_run, false },
    { "exploit_ecdsa_pmn_wraparound",  "ECDSA PMN Wraparound r∈[n,p) (2026-05-05)", "exploit_poc", test_exploit_ecdsa_pmn_wraparound_run, false },
    { "exploit_binding_retval",         "ABI Return-Value Coverage (binding guard)",   "exploit_poc", test_exploit_binding_retval_run, false },
    { "exploit_ecdsa_recovery",         "ECDSA Key Recovery Edge Cases",               "exploit_poc", test_exploit_ecdsa_recovery_run, false },
    { "exploit_ecdsa_rfc6979_kat",      "ECDSA RFC 6979 Nonce KAT",                   "exploit_poc", test_exploit_ecdsa_rfc6979_kat_run, false },
    { "exploit_ecies_auth",             "ECIES Authentication Bypass",                 "exploit_poc", test_exploit_ecies_auth_run, false },
    { "exploit_ecies_encryption",       "ECIES Encryption Security",                   "exploit_poc", test_exploit_ecies_encryption_run, false },
    { "exploit_ecies_envelope",         "ECIES Envelope Confusion",                    "exploit_poc", test_exploit_ecies_envelope_confusion_run, false },
    { "exploit_ecies_roundtrip",        "ECIES End-to-End Security",                   "exploit_poc", test_exploit_ecies_roundtrip_run, false },
    { "exploit_ecrecover_confusion",    "ECDSA ecrecover Key Confusion",               "exploit_poc", test_exploit_ecrecover_confusion_run, false },
    { "exploit_ellswift",               "ElligatorSwift (BIP-324) Security",           "exploit_poc", test_exploit_ellswift_run, false },
    { "exploit_ellswift_ecdh",          "ElligatorSwift BIP-324 ECDH",                 "exploit_poc", test_exploit_ellswift_ecdh_run, false },
    { "exploit_eth_differential",       "Ethereum Differential Test",                  "exploit_poc", test_exploit_ethereum_differential_run, false },
    { "exploit_eth_signing",            "Ethereum Signing / ecrecover",                "exploit_poc", test_exploit_eth_signing_run, false },
    { "exploit_field_arithmetic",       "Field Arithmetic Invariants",                 "exploit_poc", test_exploit_field_arithmetic_run, false },
    { "exploit_field_boundary",         "Field Boundary / Carry Exhaustive",           "exploit_poc", test_exploit_field_boundary_exhaustive_run, false },
    { "exploit_frost_binding",          "FROST Binding Factor Mismatch",               "exploit_poc", test_exploit_frost_binding_factor_mismatch_run, false },
    { "exploit_frost_byzantine",        "FROST Byzantine Signer Detection",            "exploit_poc", test_exploit_frost_byzantine_run, false },
    { "exploit_frost_commit_reuse",     "FROST Commitment Reuse",                      "exploit_poc", test_exploit_frost_commitment_reuse_run, false },
    { "exploit_frost_dkg",              "FROST DKG and Threshold Signing",             "exploit_poc", test_exploit_frost_dkg_run, false },
    { "exploit_frost_index",            "FROST Participant Index Edge Cases",           "exploit_poc", test_exploit_frost_index_run, false },
    { "exploit_frost_lagrange_dup",     "FROST Lagrange Duplicate",                    "exploit_poc", test_exploit_frost_lagrange_duplicate_run, false },
    { "exploit_frost_part_zero",        "FROST Participant Zero",                      "exploit_poc", test_exploit_frost_participant_zero_run, false },
    { "exploit_frost_signing",          "FROST Threshold Signing E2E",                 "exploit_poc", test_exploit_frost_signing_run, false },
    { "exploit_frost_threshold_degen",  "FROST Degenerate Threshold",                  "exploit_poc", test_exploit_frost_threshold_degenerate_run, false },
    { "exploit_glv_endomorphism",       "GLV Endomorphism Correctness",                "exploit_poc", test_exploit_glv_endomorphism_run, false },
    { "exploit_glv_kat",                "GLV Decomposition KAT",                       "exploit_poc", test_exploit_glv_kat_run, false },
    { "exploit_gpu_cpu_divergence",     "GPU/CPU Algebraic Consistency",               "exploit_poc", test_exploit_gpu_cpu_divergence_run, false },
    { "exploit_gpu_host_api_shape",     "GPU Host API Hostile Caller",                 "exploit_poc", test_exploit_gpu_host_api_shape_run, false },
    { "exploit_hedged_nonce_bias",      "Hedged Signature Nonce Bias",                 "exploit_poc", test_exploit_hedged_nonce_bias_run, false },
    { "exploit_hkdf_kat",               "HKDF-SHA256 KAT (RFC 5869)",                 "exploit_poc", test_exploit_hkdf_kat_run, false },
    { "exploit_hkdf_security",          "HKDF-SHA256 Security (RFC 5869)",             "exploit_poc", test_exploit_hkdf_security_run, false },
    { "infinity_edge_cases",            "Point-at-Infinity Edge Cases (INF-1..28)",    "exploit_poc", test_infinity_edge_cases_run, false },
    { "exploit_invalid_curve_twist",    "Invalid Curve / Twist Point Injection",       "exploit_poc", test_exploit_invalid_curve_twist_run, false },
    { "exploit_keccak256_kat",          "Keccak-256 KAT Vectors",                      "exploit_poc", test_exploit_keccak256_kat_run, false },
    { "exploit_multiscalar",            "Multi-Scalar Multiplication",                 "exploit_poc", test_exploit_multiscalar_run, false },
    { "exploit_musig2",                 "MuSig2 Multi-Signature Security",             "exploit_poc", test_exploit_musig2_run, false },
    { "exploit_musig2_key_agg",         "MuSig2 Key Aggregation (BIP-327)",            "exploit_poc", test_exploit_musig2_key_agg_run, false },
    { "exploit_musig2_nonce_reuse",     "MuSig2 Nonce Reuse Attack",                   "exploit_poc", test_exploit_musig2_nonce_reuse_run, false },
    { "exploit_musig2_ordering",        "MuSig2 Key Aggregation Order",                "exploit_poc", test_exploit_musig2_ordering_run, false },
    { "exploit_musig2_transcript_fork", "MuSig2 Transcript Fork",                     "exploit_poc", test_exploit_musig2_transcript_fork_run, false },
    { "exploit_pedersen_adversarial",   "Pedersen Adversarial / Switch-Commit",        "exploit_poc", test_exploit_pedersen_adversarial_run, false },
    { "exploit_pedersen_homomorphism",  "Pedersen Commitment Homomorphic",             "exploit_poc", test_exploit_pedersen_homomorphism_run, false },
    { "exploit_pedersen_switch_misuse", "Pedersen Switch Commitment Misuse",           "exploit_poc", test_exploit_pedersen_switch_misuse_run, false },
    { "exploit_pippenger_msm",          "Pippenger MSM Security",                      "exploit_poc", test_exploit_pippenger_msm_run, false },
    { "exploit_point_group_law",        "Point Group Law Correctness",                 "exploit_poc", test_exploit_point_group_law_run, false },
    { "exploit_point_serialization",    "Point Serialization Edge Cases",              "exploit_poc", test_exploit_point_serialization_run, false },
    { "exploit_private_key",            "PrivateKey Strong Type Security",              "exploit_poc", test_exploit_private_key_run, false },
    { "exploit_recovery_extended",      "ECDSA Recovery Extended Security",            "exploit_poc", test_exploit_recovery_extended_run, false },
    { "exploit_ripemd160_kat",          "RIPEMD-160 KAT + hash160",                    "exploit_poc", test_exploit_ripemd160_kat_run, false },
    { "exploit_scalar_group_order",     "Scalar Group Order Invariants",               "exploit_poc", test_exploit_scalar_group_order_run, false },
    { "exploit_scalar_invariants",      "Scalar Arithmetic Edge Cases",                "exploit_poc", test_exploit_scalar_invariants_run, false },
    { "exploit_scalar_systematic",      "Scalar Arithmetic Systematic",                "exploit_poc", test_exploit_scalar_systematic_run, false },
    { "exploit_schnorr_bip340_kat",     "Schnorr BIP-340 Official Vectors",            "exploit_poc", test_exploit_schnorr_bip340_kat_run, false },
    { "exploit_schnorr_edge_cases",     "Schnorr Signature Edge Cases",                "exploit_poc", test_exploit_schnorr_edge_cases_run, false },
    { "exploit_schnorr_forgery",        "Schnorr BIP-340 Forgery Rejection",           "exploit_poc", test_exploit_schnorr_forgery_vectors_run, false },
    { "exploit_schnorr_xonly_parity",   "Schnorr X-Only Parity Confusion",             "exploit_poc", test_exploit_schnorr_xonly_parity_confusion_run, false },
    { "exploit_seckey_arith",           "Secret Key Arithmetic (tweak/negate/cross)",  "exploit_poc", test_exploit_seckey_arith_run, false },
    { "exploit_seckey_tweak_cancel",    "Secret Key Tweak Cancellation",               "exploit_poc", test_exploit_seckey_tweak_cancel_run, false },
    { "exploit_segwit_encoding",        "SegWit Script Encoding Security",             "exploit_poc", test_exploit_segwit_encoding_run, false },
    { "exploit_selftest_api",           "Selftest API Security",                       "exploit_poc", test_exploit_selftest_api_run, false },
    { "exploit_sha256_kat",             "SHA-256 KAT + BIP-340 Tagged Hash",           "exploit_poc", test_exploit_sha256_kat_run, false },
    { "exploit_sha512_kat",             "SHA-512 KAT Vectors",                         "exploit_poc", test_exploit_sha512_kat_run, false },
    { "exploit_sha_kat",                "SHA-256/512 NIST FIPS 180-4 KAT",             "exploit_poc", test_exploit_sha_kat_run, false },
    { "exploit_silent_payment",         "Silent Payment BIP-352 Confusion",            "exploit_poc", test_exploit_silent_payment_confusion_run, false },
    { "exploit_taproot_merkle",         "Taproot Merkle Path Alias",                   "exploit_poc", test_exploit_taproot_merkle_path_alias_run, false },
    { "exploit_taproot_scripts",        "Taproot Script Tree Security",                "exploit_poc", test_exploit_taproot_scripts_run, false },
    { "exploit_taproot_tweak",          "Taproot Key Tweak / Commitment",              "exploit_poc", test_exploit_taproot_tweak_run, false },
    { "exploit_wallet_api",             "Unified Wallet API Security",                 "exploit_poc", test_exploit_wallet_api_run, false },
    { "exploit_wallet_cross_domain",    "Wallet Cross-Domain Replay",                  "exploit_poc", test_exploit_wallet_cross_domain_replay_run, false },
    { "exploit_zk_adversarial",         "ZK Proof Adversarial / Malformed",            "exploit_poc", test_exploit_zk_adversarial_run, false },
    { "exploit_zk_proofs",              "ZK Proof Soundness",                          "exploit_poc", test_exploit_zk_proofs_run, false },
    { "exploit_fe_set_b32_limit_uninit","Fe set_b32 Limit / Uninit Overflow Flag (libsecp PR #1839 bug class)", "exploit_poc", test_exploit_fe_set_b32_limit_uninit_run, false },
    { "exploit_zk_new_schemes",         "ZK New Schemes — Bulletproof + Anon Cred (eprint 2024/2010)", "exploit_poc", test_exploit_zk_new_schemes_run, false },
    { "exploit_foreign_field_plonk",    "Foreign-Field PLONK secp256k1 (eprint 2025/695)",   "exploit_poc", test_exploit_foreign_field_plonk_run, false },
    // Feature exploit PoC tests (P2SH, BIP-85, BIP-340 var, BIP-322, GCS, PSBT, Desc)
    { "exploit_p2sh_addr_confusion",    "P2SH / P2SH-P2WPKH Address Type Confusion",  "exploit_poc", test_exploit_p2sh_address_confusion_run, false },
    { "exploit_bip85_path_collision",   "BIP-85 Path Collision and Entropy Security",  "exploit_poc", test_exploit_bip85_path_collision_run, false },
    { "exploit_schnorr_msg_len",        "Schnorr Variable-Length Message Confusion",   "exploit_poc", test_exploit_schnorr_msg_length_confusion_run, false },
    { "exploit_bip322_type_confusion",  "BIP-322 Generic Message Signing Type Confusion", "exploit_poc", test_exploit_bip322_type_confusion_run, false },
    { "exploit_gcs_false_positive",     "BIP-158 GCS Filter False Positive Correctness", "exploit_poc", test_exploit_gcs_false_positive_run, false },
    { "exploit_psbt_input_confusion",   "PSBT Input Type Confusion (BIP-174/370)",     "exploit_poc", test_exploit_psbt_input_confusion_run, false },
    { "exploit_descriptor_injection",   "Output Descriptor Injection / Type Confusion","exploit_poc", test_exploit_descriptor_injection_run, false },
    { "exploit_pubkey_arith",           "Public Key Arithmetic Security (PUB-1..9)",   "exploit_poc", test_exploit_pubkey_arith_run, false },
    { "exploit_batch_sign",             "Batch Signing Edge-Case Security (BSG-1..9)", "exploit_poc", test_exploit_batch_sign_run, false },
    { "exploit_ecdsa_nonce_reuse",      "ECDSA Nonce Reuse Key Recovery (NRR-1..8)",   "exploit_poc", test_exploit_ecdsa_nonce_reuse_run, false },
    { "exploit_cross_scheme_pubkey",    "Cross-Scheme Pubkey Consistency (CSP-1..8)",  "exploit_poc", test_exploit_cross_scheme_pubkey_run, false },
    { "exploit_wif_security",           "WIF Key Format Security (WIF-1..10)",         "exploit_poc", test_exploit_wif_security_run, false },
    { "exploit_buffer_type_confusion",  "Buffer/Type-Confusion (BTC-0..12)",           "exploit_poc", test_exploit_buffer_type_confusion_run, false },
    { "exploit_differential_libsecp",   "Differential Correctness (DIF-1..10)",        "exploit_poc", test_exploit_differential_libsecp_run, false },
    { "exploit_differential_openssl",   "INTEROP: Differential vs OpenSSL libcrypto",  "differential", test_exploit_differential_openssl_run, true },
    { "exploit_bip352_scan_dos",        "BIP-352 Scan DoS Prevention (DOS-0..3)",      "exploit_poc", test_exploit_bip352_scan_dos_run, false },
    { "exploit_taproot_commit_adv",     "Taproot Commitment Adversarial (TCA-1..5)",   "exploit_poc", test_exploit_taproot_commitment_adversarial_run, false },
    { "exploit_bip352_parity",          "BIP-352 Parity Confusion (PAR-1..6)",         "exploit_poc", test_exploit_bip352_parity_confusion_run, false },
    { "exploit_bip352_batch_correct",   "BIP-352 Batch ScalarMul + Input Agg (BSM/IAG-1..10)", "math_invariants", test_exploit_bip352_batch_correctness_run, false },
    { "exploit_rfc6979_trunc_bias",     "RFC6979 Nonce Truncation Bias (NTB-1..5)",    "exploit_poc", test_exploit_rfc6979_truncation_bias_run, false },
    { "exploit_binding_adv_api",        "Binding Adversarial API (BAT-1..11)",         "exploit_poc", test_exploit_binding_adversarial_api_run, false },
    { "exploit_quantum_exposure",       "Quantum Exposure Surface (QEX-1..6)",         "exploit_poc", test_exploit_quantum_exposure_run, false },
    { "exploit_ecies_ephemeral_reuse",  "ECIES Ephemeral Reuse (EKR-1..4)",            "exploit_poc", test_exploit_ecies_ephemeral_reuse_run, false },
    { "exploit_addr_prefix_collision",  "Address Prefix Collision (APC-1..6)",         "exploit_poc", test_exploit_address_prefix_collision_run, false },
    { "exploit_binding_invalid_curve",  "Binding Invalid Curve (BIC)",                 "exploit_poc", test_exploit_binding_invalid_curve_run, false },
    { "exploit_frost_ct_nonce",         "FROST CT Nonce (FCN)",                        "exploit_poc", test_exploit_frost_ct_nonce_run, false },
    { "exploit_frost_part_set_mall",    "FROST Participant Set Malleability (FPS)",    "exploit_poc", test_exploit_frost_participant_set_malleability_run, false },
    { "exploit_musig2_par_session",     "MuSig2 Parallel Session Cross (MPS)",         "exploit_poc", test_exploit_musig2_parallel_session_cross_run, false },
    { "exploit_ecdh_zvp_glv_static",    "ECDH ZVP/GLV Static-Key Abuse",               "exploit_poc", test_exploit_ecdh_zvp_glv_static_run, false },
    { "exploit_frost_adaptive_corr",    "FROST Adaptive Corruption Misuse",            "exploit_poc", test_exploit_frost_adaptive_corruption_run, false },
    { "exploit_ecdsa_fault_injection",  "Deterministic ECDSA Fault-Injection Surface", "exploit_poc", test_exploit_ecdsa_fault_injection_run, false },
    { "exploit_cache_sidechannel_amp",  "Cache Side-Channel Amplification Surface",    "exploit_poc", test_exploit_cache_sidechannel_amplification_run, false },
    { "exploit_ladderleak_subbit_nonce", "LadderLeak-Style Sub-Bit Nonce Leakage",     "exploit_poc", test_exploit_ladderleak_subbit_nonce_run, false },
    { "exploit_minerva_noisy_hnp",       "Minerva-Style Noisy HNP Leakage Surface",    "exploit_poc", test_exploit_minerva_noisy_hnp_run, false },
    { "exploit_hertzbleed_dvfs_timing",  "Hertzbleed-Style DVFS Timing Surface",       "exploit_poc", test_exploit_hertzbleed_dvfs_timing_run, false },
    { "exploit_biased_nonce_chain_scan", "Biased-Nonce Chain-Scale Scan Surface",      "exploit_poc", test_exploit_biased_nonce_chain_scan_run, false },
    { "exploit_kr_ecdsa_buff_binding",   "KR-ECDSA/BUFF Binding Regression Surface",   "exploit_poc", test_exploit_kr_ecdsa_buff_binding_run, false },
    { "exploit_mutation_residue",        "Mutation Residue Detection (MR-1..MR-7)",    "exploit_poc", test_exploit_mutation_residue_run, false },
    { "mutation_artifact_scan",          "Source Integrity Scanner (MA-1..MA-4)",       "exploit_poc", test_mutation_artifact_scan_run, false },
    { "exploit_metal_field_reduce",       "Metal field_reduce_512 Regression (#226)",    "exploit_poc", test_exploit_metal_field_reduce_run, false },
    { "exploit_network_validation_bypass", "Network Selector Bypass (NVB-1..NVB-8)",     "exploit_poc", test_exploit_network_validation_bypass_run, false },
    { "exploit_half_half_nonce",         "Half-Half Nonce Key Recovery (HH-1..HH-10)", "exploit_poc", test_exploit_ecdsa_half_half_nonce_run, false },
    { "exploit_nonce_modular_bias",      "Nonce Modular Reduction Bias (NMB-1..NMB-6)","exploit_poc", test_exploit_ecdsa_nonce_modular_bias_run, false },
    { "exploit_differential_fault",      "Differential Fault RFC 6979 (DF-1..DF-8)",   "exploit_poc", test_exploit_ecdsa_differential_fault_run, false },
    { "exploit_eucleak_inversion",        "EUCLEAK Inversion Timing (EUC-1..EUC-12)",   "exploit_poc", test_exploit_eucleak_inversion_timing_run, false },
    { "exploit_cross_key_nonce_reuse",    "Cross-Key Nonce Reuse (CKN-1..CKN-10)",      "exploit_poc", test_exploit_ecdsa_cross_key_nonce_reuse_run, false },
    { "exploit_schnorr_hash_order",       "Schnorr Hash Order (SHO-1..SHO-10)",         "exploit_poc", test_exploit_schnorr_hash_order_run, false },
    { "exploit_zvp_glv_dcp_multiscalar",   "ZVP-DCP GLV Multiscalar (ZVPDCP-1..8)",      "exploit_poc", test_exploit_zvp_glv_dcp_multiscalar_run, false },
    { "exploit_lattice_sieve_hnp",         "Lattice Sieve HNP Sub-bit (LSHNP-1..8)",    "exploit_poc", test_exploit_lattice_sieve_hnp_run, false },
    { "exploit_deterministic_sig_dfa",     "Det. Sig DFA RFC 6979 (DSDFA-1..8)",         "exploit_poc", test_exploit_deterministic_sig_dfa_run, false },
    { "exploit_sign_type_confusion",       "Type Confusion k-Reuse (STCK-1..10)",        "exploit_poc", test_exploit_sign_type_confusion_kreuse_run, false },
    { "exploit_ros_concurrent_schnorr",    "ROS Concurrent Schnorr (ROS-1..10)",         "exploit_poc", test_exploit_ros_concurrent_schnorr_run, false },
    { "exploit_frost_weak_binding",        "FROST Weak Binding (FWB-1..8)",              "exploit_poc", test_exploit_frost_weak_binding_run, false },
    { "exploit_blind_spa_cmov_leak",       "Blind SPA & cmov Leak (BSPA-1..12)",        "exploit_poc", test_exploit_blind_spa_cmov_leak_run, false },
    { "exploit_ectester_point_validation", "ECTester Point Validation (ECT-1..18)",     "exploit_poc", test_exploit_ectester_point_validation_run, false },
    { "exploit_ros_dimensional_erosion",   "ROS Dimensional eROS (RDE-1..12)",          "exploit_poc", test_exploit_ros_dimensional_erosion_run, false },
    { "exploit_ecdsa_batch_verify_rand",   "ECDSA Batch Verify Rand (BVR-1..16)",      "exploit_poc", test_exploit_ecdsa_batch_verify_rand_run, false },
    { "exploit_bip324_aead_forgery",       "BIP-324 AEAD Forgery (BAF-1..15)",          "exploit_poc", test_exploit_bip324_aead_forgery_run, false },
    { "exploit_frost_rogue_key",           "FROST Rogue-Key Attack (FRK-1..12)",        "exploit_poc", test_exploit_frost_rogue_key_run, false },
    { "exploit_musig2_partial_forgery",    "MuSig2 Partial Forgery (MPF-1..10)",        "exploit_poc", test_exploit_musig2_partial_forgery_run, false },
    { "exploit_adaptor_extraction",        "Adaptor Extraction Soundness (ASE-1..12)",  "exploit_poc", test_exploit_adaptor_extraction_soundness_run, false },
    { "exploit_ecdh_twist_injection",      "ECDH Twist Injection (ETP-1..12)",          "exploit_poc", test_exploit_ecdh_twist_injection_run, false },
    { "exploit_schnorr_batch_inflation",   "Schnorr Batch Inflation (SBI-1..12)",       "exploit_poc", test_exploit_schnorr_batch_inflation_run, false },

    // ===================================================================
    // Section 9 (continued): Orphan PoCs wired 2026-04-17 — close the gap
    // between on-disk audit/test_exploit_*.cpp files and registered modules
    // per the Conversion Standard in docs/EXPLOIT_BACKLOG.md.
    // ===================================================================
    { "exploit_bip32_child_key_attack",    "BIP-32 Child Key Attack",                   "exploit_poc", test_exploit_bip32_child_key_attack_run, false },
    { "exploit_boundary_sentinels",        "ABI Boundary Sentinel / OOB Detection",     "exploit_poc", test_exploit_boundary_sentinels_run, false },
    { "exploit_buff_kr_ecdsa",             "BUFF KR-ECDSA Binding (ePrint 2021/1514)",  "exploit_poc", test_exploit_buff_kr_ecdsa_run, false },
    { "exploit_ecdsa_affine_nonce_rel",    "ECDSA Affine Nonce Relation Attack",        "exploit_poc", test_exploit_ecdsa_affine_nonce_relation_run, false },
    { "exploit_ecdsa_r_overflow",          "ECDSA Large-r Verify Bug (Stark Bank class)", "exploit_poc", test_exploit_ecdsa_r_overflow_run, false },
    { "exploit_ecdsa_sign_sentinels",      "ECDSA Sign Sentinel / Boundary Inputs",     "exploit_poc", test_exploit_ecdsa_sign_sentinels_run, false },
    { "exploit_eip712_kat",                "EIP-712 Typed Structured Data KAT",         "exploit_poc", test_exploit_eip712_kat_run, false },
    { "exploit_ellswift_bad_scalar_ecdh",  "ElligatorSwift Bad-Scalar ECDH",            "exploit_poc", test_exploit_ellswift_bad_scalar_ecdh_run, false },
    { "exploit_ellswift_xdh_overflow",     "ElligatorSwift xDH Overflow",               "exploit_poc", test_exploit_ellswift_xdh_overflow_run, false },
    { "exploit_frost_identifiable_abort",  "FROST Identifiable Abort (ePrint 2022/550)", "exploit_poc", test_exploit_frost_identifiable_abort_run, false },
    { "exploit_hash_algo_sig_isolation",   "Hash-Algorithm vs Signature Scheme Isolation", "exploit_poc", test_exploit_hash_algo_sig_isolation_run, false },
    { "exploit_minerva_cve_2024_23342",    "Minerva CVE-2024-23342 Regression",         "exploit_poc", test_exploit_minerva_cve_2024_23342_run, false },
    { "exploit_musig2_byzantine_multi",    "MuSig2 Byzantine Multi-Party",              "exploit_poc", test_exploit_musig2_byzantine_multiparty_run, false },
    { "exploit_rfc6979_minerva_amplified", "RFC 6979 Minerva-Amplified Bias",           "exploit_poc", test_exploit_rfc6979_minerva_amplified_run, false },
    { "exploit_scalar_mul",                "Point::scalar_mul Edge Cases (SM-1..SM-12)", "exploit_poc", test_exploit_scalar_mul_run, false },
    { "exploit_schnorr_nonce_reuse",       "Schnorr Nonce Reuse Key Recovery",          "exploit_poc", test_exploit_schnorr_nonce_reuse_run, false },

    // ===================================================================
    // Section 10: Math Invariants — SafeGCD & Nonce Edge Cases
    // ===================================================================
    { "exploit_safegcd_divsteps",  "SafeGCD/Bernstein-Yang divstep count + correctness (SGD-1..11)", "math_invariants", test_exploit_safegcd_divsteps_run, false },
    { "exploit_nonce_injection",   "RFC 6979 nonce edge cases: null/zero/n/n-1/det (NIN-1..15)",    "protocol_security", test_exploit_custom_nonce_injection_run, false },
    { "exploit_jni_retval_ignored", "JNI Return-Value Ignored: sha256/hash160/tagged_hash/negate (RVI-1..8)", "exploit_poc", test_exploit_jni_retval_ignored_run, false },
    { "exploit_dark_skippy_exfil",         "Dark Skippy Nonce Exfiltration: RFC6979/aux_rand anti-grinding (DS-1..8) — ePrint 2024/1225",         "exploit_poc", test_exploit_dark_skippy_exfil_run, false },
    { "exploit_fiat_shamir_frozen_heart",  "Frozen Heart: ZK Fiat-Shamir incomplete binding (FH-1..10) — ePrint 2022/411",                       "exploit_poc", test_exploit_fiat_shamir_frozen_heart_run, false },
    { "exploit_hertzbleed_scalar_blind",   "Hertzbleed Scalar Blinding: Hamming-weight oracle + aux_rand mitigation (SB-1..9) — ePrint 2022/823",  "exploit_poc", test_exploit_hertzbleed_scalar_blind_run, false },
    { "exploit_thread_unsafe_lazy_init",   "Thread-safe gen_fb_table lazy init: std::once_flag regression (TIR-1..5)",                             "exploit_poc", test_exploit_thread_unsafe_lazy_init_run, false },
    { "regression_z_fe_nonzero",           "z_fe_nonzero | vs & correctness regression (ZFN-1..6)",                                                 "math_invariants", test_regression_z_fe_nonzero_run, false },
    { "regression_cuda_pool_cap",          "CUDA pool minimum-capacity rounding regression (CAP-1..4)",                                              "math_invariants", test_regression_cuda_pool_cap_run, false },
    { "regression_musig2_verify",          "musig2_partial_verify OOB + infinity-nonce regression (MVV-1..6)",                                      "math_invariants", test_regression_musig2_verify_run, false },
    { "regression_bip324_session",         "Bip324Session sk-leak + destructor secure-erase regression (BPS-1..8)",                                   "memory_safety",  test_regression_bip324_session_run, false },
    { "regression_exception_erase",        "Exception-path sk/ek/entropy leakage in C ABI layer (EPE-RAII+EPE-1..12)",                               "memory_safety",  test_regression_exception_erase_run, false },
    // ===================================================================
    // Section 11: Quality Audit Fixes (2026-04-27 Comprehensive Report)
    // ===================================================================
    { "exploit_recoverable_sign_ct",         "C ABI recovery signing CT path (RCTX-1..8) — C-1 fix 2026-04-27",              "exploit_poc",   test_exploit_recoverable_sign_ct_run, false },
    { "exploit_pippenger_batch_regression",  "Pippenger batch verify regression guard (PIPBATCH-1..8) — M-6 2026-04-27",     "exploit_poc",   test_exploit_pippenger_batch_regression_run, false },
    { "exploit_eth_signing_ct",              "Ethereum signing CT path (ETHCT-1..8) — B-01 fix 2026-04-27",                  "exploit_poc",   test_exploit_eth_signing_ct_run, false },
    { "exploit_wallet_sign_ct",              "Wallet sign_hash CT path (WALCT-1..8) — B-02 fix 2026-04-27",                  "exploit_poc",   test_exploit_wallet_sign_ct_run, false },
    { "exploit_monolith_split",              "ufsecp_impl.cpp unity-build split integrity (MONO-1..12) — B-04 2026-04-27",   "exploit_poc",   test_exploit_monolith_split_run, false },
    { "exploit_gpu_secret_erase",            "GPU secret erase + batch sign partial (B08-1..4, B09-1..5) — B-08/B-09 2026-04-27", "exploit_poc", test_exploit_gpu_secret_erase_run, false },
    { "exploit_libsecp_eckey_api",           "libsecp256k1 EC key API compat (ECKEY-1..17) — L-01 2026-04-27",                   "differential", test_exploit_libsecp_eckey_api_run, false },
    { "exploit_bitcoin_core_rgrinding",      "Bitcoin Core R-grinding nonce pattern (RGRIND-1..8) — BC-01 2026-04-27",           "differential", test_exploit_bitcoin_core_rgrinding_run, false },
    // Section 12: Upstream libsecp256k1 test parity (2026-04-28 gap closure)
    // ===================================================================
    { "exploit_pubkey_cmp",                  "Pubkey comparison ordering (GAP-3) — upstream run_pubkey_comparison 2026-04-28",   "exploit_poc",  test_exploit_pubkey_cmp_run, false },
    { "exploit_pubkey_sort",                 "Pubkey sort + MuSig2 ordering (GAP-4) — upstream run_pubkey_sort 2026-04-28",      "exploit_poc",  test_exploit_pubkey_sort_run, false },
    { "exploit_alloc_bounds",                "Allocation boundary batch verify (GAP-1) — upstream run_scratch_tests 2026-04-28", "exploit_poc",  test_exploit_alloc_bounds_run, false },
    { "exploit_hsort",                       "Heap sort / batch ordering (GAP-2) — upstream run_hsort_tests 2026-04-28",         "exploit_poc",  test_exploit_hsort_run, false },
    { "exploit_wnaf",                        "wNAF window decomposition boundaries (GAP-5) — upstream run_wnaf_tests 2026-04-28","exploit_poc",  test_exploit_wnaf_run, false },
    { "exploit_int128",                      "128-bit arithmetic field boundaries (GAP-7) — upstream run_int128_tests 2026-04-28","exploit_poc", test_exploit_int128_run, false },
    { "exploit_scratch",                     "Scratch allocator risk surface — upstream run_scratch_tests 2026-04-28",            "exploit_poc",  test_exploit_scratch_run, false },
    { "exploit_xoshiro",                     "xoshiro256** PRNG context randomization — upstream run_xoshiro256pp_tests 2026-04-28","exploit_poc",test_exploit_xoshiro_run, false },
    { "exploit_cross_protocol_kreuse",       "Cross-Protocol Key Reuse: MuSig2+FROST+ECDSA same sk (CPK-1..5) — Original 2026-04-28", "exploit_poc", test_exploit_cross_protocol_kreuse_run, false },
    { "exploit_tagged_hash_ext",             "BIP-340 Tagged SHA-256 Length Extension (TAGEXT-1..5) — Original 2026-04-28",           "exploit_poc", test_exploit_tagged_hash_ext_run, false },
    { "exploit_wnaf_cache_ampl",             "wNAF Window-18 Cache Amplification 8× (WCACHE-1..5) — Original 2026-04-28",            "exploit_poc", test_exploit_wnaf_cache_ampl_run, false },
    { "exploit_musig2_fingerprint_collision","MuSig2 KeyAgg Fingerprint Collision (FPC-1..6) — Original 2026-04-28",                  "exploit_poc", test_exploit_musig2_fingerprint_collision_run, false },
    { "exploit_blinding_recovery_hnp",       "Context Blinding Recovery via HNP (BLIND-1..7) — Original 2026-04-28",                  "exploit_poc", test_exploit_blinding_recovery_hnp_run, false },
    { "exploit_shim_noncefp_bypass",         "Shim noncefp Callback Bypass: callback silently ignored (NONCEFP-1..5) — Original 2026-04-28", "exploit_poc", test_exploit_shim_noncefp_bypass_run, false },
    { "exploit_encoding_memory_corruption",  "Encoding Memory Corruption: DER/field/pubkey adversarial inputs (ENCORR-1..6) — Original 2026-04-28", "exploit_poc", test_exploit_encoding_memory_corruption_run, false },
    { "exploit_batch_verify_malleability",   "Batch Verify Malleability: order/dup/poison/ECDSA correctness (BVM-1..6) — Original 2026-04-28", "exploit_poc", test_exploit_batch_verify_malleability_run, false },
    { "exploit_thread_local_blinding",       "Thread-Local Blinding Race: shared ctx concurrent randomize+sign (TLB-1..4) — Original 2026-04-28", "exploit_poc", test_exploit_thread_local_blinding_run, false },
    { "exploit_hedged_return_value",         "Hedged Sign Return Silence: fail-closed zero-sig invariant (HEDGED-1..4) — Original 2026-04-28", "exploit_poc", test_exploit_hedged_return_value_run, false },
    { "exploit_gpu_memory_safety",           "GPU Kernel Memory Safety: NULL/invalid API boundary (GPU-1..5) — Original 2026-04-28", "exploit_poc", test_exploit_gpu_memory_safety_run, false },
    { "exploit_rs_zero_check",               "ECDSA r,s Zero Check Gap: CVE-2022-39272-class rejection (RZERO-1..5) — Original 2026-04-28", "exploit_poc", test_exploit_rs_zero_check_run, false },
    { "exploit_bip352_address_collision",    "BIP-352 Address Collision: domain separation collision resistance (SP-1..4) — Original 2026-04-28", "exploit_poc", test_exploit_bip352_address_collision_run, false },
    // Section 13: BUG-001..008 fixes (2026-04-28 full audit)
    // ===================================================================
    { "exploit_bug001_addr_overflow",    "BUG-001/007: ufsecp_addr_* off-by-one overflow + p2sh null (AOF-1..15) — 2026-04-28", "exploit_poc", test_exploit_bug001_addr_overflow_run, false },
    { "exploit_bug002_recovery_ct",      "BUG-002: recovery.cpp CT recid overflow check (RCT-1..8) — 2026-04-28",               "exploit_poc", test_exploit_bug002_recovery_ct_run, false },
    { "exploit_bug003_normalize_ct",     "BUG-003/008: ECDSASignature::normalize() CT path (NCT-1..8) — 2026-04-28",            "exploit_poc", test_exploit_bug003_normalize_ct_run, false },
    { "exploit_bug004_batch_failclosed", "BUG-004 + Guardrail #4: batch fail-closed + zero-sig ABI checks (BFC-1..9) — 2026-05-01", "exploit_poc", test_exploit_bug004_batch_failclosed_run, false },
    // Section 14: 2026-05-01 security audit fixes (shim layer)
    // ===================================================================
    { "exploit_shim_der_bip66",         "HIGH-2/3: shim DER parser BIP-66 negative-int + trailing-bytes (DER66-1..8) — 2026-05-01", "exploit_poc", test_exploit_shim_der_bip66_run, true },
    { "exploit_shim_musig_secnonce",    "CRIT-1: shim MuSig2 secnonce reuse key-leak prevention (MSN-1..6) — 2026-05-01",           "exploit_poc", test_exploit_shim_musig_secnonce_run, true },
    { "exploit_shim_musig_ka_cap",      "RED-TEAM-009: ka_put DoS-cap fail-closed — pubkey_agg returns 0 at session limit",            "exploit_poc", test_exploit_shim_musig_ka_cap_run,    false },
    { "exploit_shim_recovery_null_arg", "SHIM-005: secp256k1_ecdsa_sign_recoverable NULL arg fail-closed (REC-NULL-1..4)",                "exploit_poc", test_exploit_shim_recovery_null_arg_run, true },
    // === 2026-05-01 Red Team Audit Fixes — advisory=true: shim not linked in unified runner ===
    { "test_exploit_legacy_capi_key_parsing",    "Legacy C API invalid private key rejection (KP-1..14) — 2026-05-01",                          "exploit_poc", test_exploit_legacy_capi_key_parsing_run,    true },
    { "test_exploit_legacy_capi_degenerate_sig", "Legacy C API degenerate zero-sig output guard (DSG-1..7) — 2026-05-01",                       "exploit_poc", test_exploit_legacy_capi_degenerate_sig_run, true },
    { "test_exploit_musig_unknown_signer",       "MuSig2 partial_sign with unknown signer key (MUS-1..5) — 2026-05-01",                         "exploit_poc", test_exploit_musig_unknown_signer_run,       true },
    { "test_exploit_bchn_schnorr_strict_parsing","BCHN Schnorr shim strict private key parsing (BCH-1..9) — 2026-05-01",                        "exploit_poc", test_exploit_bchn_schnorr_strict_parsing_run, true },
    { "test_exploit_context_flag_bypass",        "libsecp256k1 shim context flag enforcement bypass (CFB-1..9) — 2026-05-01",                   "exploit_poc", test_exploit_context_flag_bypass_run,        true },
    { "test_exploit_metal_schnorr_aux_rand",     "Metal Schnorr batch aux_rand uses private key CRITICAL-1 (MA-1..4) — 2026-05-01",             "exploit_poc", test_exploit_metal_schnorr_aux_rand_run,     false },
    { "test_exploit_metal_batch_failclosed",     "Metal batch sign ignored return + non-CT path CRITICAL-2+HIGH-1 (MB-1..6) — 2026-05-01",     "exploit_poc", test_exploit_metal_batch_failclosed_run,     false },
    { "test_exploit_gpu_bip352_key_erase",       "GPU BIP-352 scan key not zeroed before device memory free HIGH-3 (BK-1..8) — 2026-05-01",   "exploit_poc", test_exploit_gpu_bip352_key_erase_run,       false },
    { "test_exploit_metal_ecdh_key_erase",       "Metal ECDH batch private key not erased from shared buffer HIGH-2+LOW-5 (ME-1..5) — 2026-05-01", "exploit_poc", test_exploit_metal_ecdh_key_erase_run,  false },
    { "test_exploit_opencl_runner_key_erase",    "OpenCL audit runner d_priv/d_scalar not zeroed before release Q-01/02/03 (OCR-1..8) — 2026-05-02", "exploit_poc", test_exploit_opencl_runner_key_erase_run, false },
    { "test_exploit_ecdsa_fast_path_isolation",  "ecdsa.cpp fast path isolation: public APIs must use ct::ecdsa_sign (FPI-1..10) — 2026-05-02",       "exploit_poc", test_exploit_ecdsa_fast_path_isolation_run, false },
    // === 2026-05-01 Red Team Audit Round 3 (P-01..P-09) ===
    // Shim not linked in unified runner — _run() returns ADVISORY_SKIP_CODE (77).
    // Actual SPC-1..10 checks run via test_exploit_shim_pubkey_ct_standalone CTest target.
    { "exploit_shim_pubkey_ct",                 "P-01/02/03: shim pubkey_create/keypair CT generator mul + key rejection (SPC-1..10) — 2026-05-01", "exploit_poc", test_exploit_shim_pubkey_ct_run, true },
    // === 2026-05-02 Bitcoin Core PR Security Audit Fixes ===
    // (forward declarations at top of file)
    { "regression_schnorr_ct_arithmetic",  "Schnorr s=k+e*d uses ct:: arithmetic, r==all-zeros rejected (HIGH-03, HIGH-06) — 2026-05-02", "exploit_poc", test_regression_schnorr_ct_arithmetic_run,  false },
    { "regression_musig2_zero_psig",       "musig2_partial_sign degenerate zero psig → UFSECP_ERR_INTERNAL (CRIT-03) — 2026-05-02",       "exploit_poc", test_regression_musig2_zero_psig_run,         false },
    { "regression_gpu_key_erase_raii",     "GPU key material erased on all exit paths: CUDA RAII + OpenCL pubkey-first + scalar buffer zero (CRIT-01, HIGH-01, HIGH-02, HIGH-04) — 2026-05-02", "memory_safety", test_regression_gpu_key_erase_raii_run, false },
    { "regression_bip352_ct_varbase",      "BIP-352 scan kernel uses CT variable-base scalar mul for scan_k (CRIT-02) — 2026-05-02",       "ct_analysis",  test_regression_bip352_ct_varbase_run,        false },
    // === 2026-05-04 Performance Review Security + Correctness Fixes ===
    { "signing_ct_scalar_correctness_regression", "CT signing scalar correctness: gen-mul, inv, cswap, Pippenger, BatchVerify (PRF-1..8)", "exploit_poc", test_regression_signing_ct_scalar_correctness_run, false },
    // === 2026-05-05 Full Red-Team Audit Regression Guards ===
    // CPU-testable regressions: P1-5 (double-normalize removed), P2-3 (r==0 check),
    // P0-3 (ct_ecdsa_sign_verified actually verifies), P0-2 (low-S correctness).
    { "abi_recoverable_schnorr_ct_regression", "ABI recid+recov+schnorr_verified CT regression guards", "exploit_poc", test_exploit_abi_recoverable_schnorr_ct_regression_run, false },
    // === 2026-05-05 Bug Bounty Red-Team Regression Guards ===
    // C1: OCL recovery scalar_is_even→scalar_is_low_s  C2: non-CT nonce k*G
    // C3: FROST n_signers<threshold  C4: FROST signing share strict_nonzero
    // H1: GPU BIP32 depth overflow  H3: shim secp256k1_ecdsa_sign_recoverable ctx_can_sign
    { "frost_ocl_shim_bip32_ct_regression", "FROST+OCL+shim+BIP32 CT regression guards", "exploit_poc", test_exploit_frost_ocl_shim_bip32_ct_regression_run, false },
    { "musig2_nonce_erasure_le32_ecdh", "MuSig2 secnonce/nonce erasure + LE32 round-trip + ECDH Y-parity prefix", "memory_safety", test_exploit_musig2_nonce_erasure_le32_ecdh_run, false },
    { "regression_pippenger_stale_used", "Pippenger used[] not cleared per-window — stale bits corrupt MSM for n>=48 (BUG-01, PIP-R1..R7)", "math_invariants", test_regression_pippenger_stale_used_run, false },
    { "exploit_frost_secret_share_ct",   "FROST DKG share.value processed with ct::generator_mul not variable-time scalar_mul (SEC-01, FROST-CT1..5)", "ct_analysis",    test_exploit_frost_secret_share_ct_run,    false },
    { "regression_comb_gen_lockfree",    "comb_gen_mul/ct lock-free after once_flag init: no mutex on read path (CRIT-01, COMB-LF1..6)", "math_invariants", test_regression_comb_gen_lockfree_run,     false },
    // === CT timing regression: V-01 fast::Scalar operator* on secrets ===
    { "ct_fast_scalar_v01_timing", "V-01: fast::Scalar operator* banned on secret material — Welch t-test on ECDSA sign with HW=1 vs HW=80 keys", "ct_analysis", test_regression_ct_fast_scalar_v01_run, true },
    { "schnorr_abi_edge_cases", "TQ-005: Schnorr BIP-340 ABI edge cases — r==0, r>=p, s==0, s>=n, wrong-msg, NULL args all rejected", "exploit_poc", test_regression_schnorr_abi_edge_cases_run, false },
    { "regression_ct_mixed_add_magnitude", "CA-mixed-add: point_add_mixed_complete FE52 magnitude contract — normalize_weak X1/Y1 guard, 2G+3G=5G, blinded==unblinded, ct vs fast recover", "math_invariants", test_regression_ct_mixed_add_magnitude_run, false },
    { "regression_shim_static_ctx", "ecf47967: g_static_ctx field alignment after PERF-005 cached_r_G addition", "math_invariants", test_regression_shim_static_ctx_run, true },
    // === 2026-05-11 Security audit regression guards (CT-001, CT-006, RT-011, SHIM-001, SHIM-010, SHIM-012) ===
    // advisory=true: these modules depend on the libsecp256k1 shim being linked.
    // On GitHub CI (shim absent) the stubs return ADVISORY_SKIP_CODE (77).
    // The advisory flag MUST match the stub behaviour — false would falsely claim mandatory pass.
    { "regression_ellswift_ct_path",           "CT-001: ellswift_create routes through ct::generator_mul — deterministic encoding, XDH round-trip, zero/null key rejection", "ct_analysis",    test_regression_ellswift_ct_path_run,           true },
    { "regression_musig2_nonce_strict",        "CT-006: MuSig2 k1/k2 nonces via parse_bytes_strict_nonzero — non-zero R1/R2 commitments, distinct nonces per extra_input",  "ct_analysis",    test_regression_musig2_nonce_strict_run,        true },
    { "regression_bip32_private_key_strict",   "RT-011: BIP-32 private_key() strict parsing — key==n → zero, key==0 → zero, valid key round-trips, HD child derivation",    "protocol_security", test_regression_bip32_private_key_strict_run, true },
    { "regression_shim_pubkey_sort",           "SHIM-012: secp256k1_ec_pubkey_sort no longer crashes via nullptr ctx — lexicographic order correctness (PST-1..4)",          "memory_safety",  test_regression_shim_pubkey_sort_run,           true },
    { "regression_shim_per_context_blinding",  "SHIM-001: per-context blinding — two contexts on same thread sign independently, unblinded ctx works, NULL seed clears",     "ct_analysis",    test_regression_shim_per_context_blinding_run,  true },
    { "regression_musig2_session_token",       "SHIM-010: MuSig2 token-keyed session map — non-zero token after agg, distinct tokens, reuse gets fresh token, 2-of-2 sign", "memory_safety",  test_regression_musig2_session_token_run,       true },
    // SEC-007: MuSig2 signer_index cross-check — uses C++ API directly (no shim required)
    { "regression_musig2_signer_index",        "SEC-007: musig2_partial_sign validates secret_key<->signer_index (Rule 13) — wrong index returns zero, correct index signs", "protocol_security", test_regression_musig2_signer_index_validation_run, false },
    // === 2026-05-11 Shim regression tests (Agent 5) ===
    // advisory=true: depend on the libsecp256k1 shim being linked.
    // On GitHub CI (shim absent) the stubs return ADVISORY_SKIP_CODE (77).
    { "exploit_shim_der_zero_r", "DER parse r=0 rejection (shim vs libsecp divergence)", "exploit_poc", test_shim_der_zero_r_run, true },
    { "exploit_shim_null_ctx",   "NULL context illegal-callback enforcement (SHIM-001/002/003)", "exploit_poc", test_shim_null_ctx_run, true },
};

static constexpr int NUM_MODULES = sizeof(ALL_MODULES) / sizeof(ALL_MODULES[0]);

// ============================================================================
// Platform detection
// ============================================================================
struct PlatformInfo {
    std::string os;
    std::string arch;
    std::string compiler;
    std::string build_type;
    std::string timestamp;
    std::string library_version;
    std::string git_hash;
    std::string framework_version;
};

static PlatformInfo detect_platform() {
    PlatformInfo info;

    // -- OS --
#if defined(_WIN32)
    info.os = "Windows";
#elif defined(__APPLE__)
    info.os = "macOS";
#elif defined(__linux__)
    info.os = "Linux";
#elif defined(__FreeBSD__)
    info.os = "FreeBSD";
#else
    info.os = "Unknown";
#endif

    // -- Architecture --
#if defined(__x86_64__) || defined(_M_X64)
    info.arch = "x86-64";
#elif defined(__aarch64__) || defined(_M_ARM64)
    info.arch = "ARM64";
#elif defined(__riscv) && (__riscv_xlen == 64)
    info.arch = "RISC-V 64";
#elif defined(__riscv)
    info.arch = "RISC-V 32";
#elif defined(__EMSCRIPTEN__)
    info.arch = "WASM";
#elif defined(__arm__) || defined(_M_ARM)
    info.arch = "ARM32";
#else
    info.arch = "Unknown";
#endif

    // -- Compiler --
#if defined(__clang__)
    char buf[128];
    (void)std::snprintf(buf, sizeof(buf), "Clang %d.%d.%d", __clang_major__, __clang_minor__, __clang_patchlevel__);
    info.compiler = buf;
#elif defined(__GNUC__)
    char buf[128];
    (void)std::snprintf(buf, sizeof(buf), "GCC %d.%d.%d", __GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__);
    info.compiler = buf;
#elif defined(_MSC_VER)
    char buf[128];
    (void)std::snprintf(buf, sizeof(buf), "MSVC %d", _MSC_VER);
    info.compiler = buf;
#else
    info.compiler = "Unknown";
#endif

    // -- Build type --
#if defined(NDEBUG)
    info.build_type = "Release";
#else
    info.build_type = "Debug";
#endif

    // -- Timestamp --
    auto now = std::chrono::system_clock::now();
    auto t = std::chrono::system_clock::to_time_t(now);
    char timebuf[64];
    struct tm tm_buf{};
#ifdef _WIN32
    (void)localtime_s(&tm_buf, &t);
#else
    (void)localtime_r(&t, &tm_buf);
#endif
    (void)std::strftime(timebuf, sizeof(timebuf), "%Y-%m-%dT%H:%M:%S", &tm_buf);
    info.timestamp = timebuf;

    // -- Version / git / framework --
    info.library_version  = SECP256K1_VERSION_STRING;
    info.git_hash         = GIT_HASH;
    info.framework_version = AUDIT_FRAMEWORK_VERSION;

    return info;
}

// ============================================================================
// JSON escaping
// ============================================================================
static std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char const c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            case '\b': out += "\\b";  break;
            case '\f': out += "\\f";  break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8]; std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned char>(c));
                    out += buf;
                } else {
                    out += c;
                }
                break;
        }
    }
    return out;
}

// ============================================================================
// Module result
// ============================================================================
// MEDIUM-5: advisory skip/fail classification uses return_code (sentinel 77)
// as the primary signal; elapsed_ms < 1.0 is kept as a backward-compat fallback.
static constexpr int ADVISORY_SKIP_CODE = 77;  // mirrors audit_check.hpp

struct ModuleResult {
    const char* id;
    const char* name;
    const char* section;
    bool        passed;
    bool        advisory;
    double      elapsed_ms;
    int         return_code;  // raw return value from m.run() (MEDIUM-5 fix)
};

// ============================================================================
// Section summary helper
// ============================================================================
struct SectionSummary {
    const char* section_id;
    const char* title_en;
    int total;
    int passed;
    int failed;
    int advisory_skipped;
    int advisory_failed;
    double time_ms;
};

static std::vector<SectionSummary> compute_section_summaries(
    const std::vector<ModuleResult>& results)
{
    std::vector<SectionSummary> out;
    for (int s = 0; s < NUM_SECTIONS; ++s) {
        SectionSummary ss{};
        ss.section_id       = SECTIONS[s].id;
        ss.title_en         = SECTIONS[s].title_en;
        ss.total = ss.passed = ss.failed = 0;
        ss.advisory_skipped = ss.advisory_failed = 0;
        ss.time_ms = 0;
        for (auto& r : results) {
            if (std::strcmp(r.section, SECTIONS[s].id) == 0) {
                ++ss.total;
                if (r.passed) {
                    ++ss.passed;
                } else if (r.advisory) {
                    if (r.return_code == ADVISORY_SKIP_CODE) {
                        ++ss.advisory_skipped;
                    } else {
                        ++ss.advisory_failed;
                    }
                } else {
                    ++ss.failed;
                }
                ss.time_ms += r.elapsed_ms;
            }
        }
        out.push_back(ss);
    }
    return out;
}

// ============================================================================
// Report writer -- JSON (structured by 8 sections)
// ============================================================================
static void write_json_report(const char* path,
                               const PlatformInfo& plat,
                               const std::vector<ModuleResult>& results,
                               bool selftest_passed,
                               double selftest_ms,
                               double total_ms) {
#ifdef _WIN32
    FILE* f = std::fopen(path, "w");
#else
    int const fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    FILE* f = (fd >= 0) ? fdopen(fd, "w") : nullptr;
#endif
    if (!f) {
        (void)std::fprintf(stderr, "WARNING: Cannot open %s for writing\n", path);
        return;
    }

    int total_pass = 0, total_fail = 0, total_advisory = 0;
    int total_advisory_skipped = 0, total_advisory_failed = 0;
    for (auto& r : results) {
        if (r.passed) {
            ++total_pass;
        } else if (r.advisory) {
            ++total_advisory;
            // MEDIUM-5 fix: use return_code sentinel as primary classifier.
            // elapsed_ms heuristic removed — it caused fast non-advisory tests
            // to be mis-classified as advisory_skipped.
            if (r.return_code == ADVISORY_SKIP_CODE) {
                ++total_advisory_skipped;
            } else {
                ++total_advisory_failed;
            }
        } else {
            ++total_fail;
        }
    }
    if (selftest_passed) { ++total_pass; } else { ++total_fail; }

    auto sections = compute_section_summaries(results);

    (void)std::fprintf(f, "{\n");
    (void)std::fprintf(f, "  \"report_type\": \"industrial_self_audit\",\n");
    (void)std::fprintf(f, "  \"library\": \"UltrafastSecp256k1\",\n");
    (void)std::fprintf(f, "  \"library_version\": \"%s\",\n", json_escape(plat.library_version).c_str());
    (void)std::fprintf(f, "  \"git_hash\": \"%s\",\n", json_escape(plat.git_hash).c_str());
    (void)std::fprintf(f, "  \"audit_framework_version\": \"%s\",\n", json_escape(plat.framework_version).c_str());
    (void)std::fprintf(f, "  \"timestamp\": \"%s\",\n", json_escape(plat.timestamp).c_str());
    (void)std::fprintf(f, "  \"platform\": {\n");
    (void)std::fprintf(f, "    \"os\": \"%s\",\n", json_escape(plat.os).c_str());
    (void)std::fprintf(f, "    \"arch\": \"%s\",\n", json_escape(plat.arch).c_str());
    (void)std::fprintf(f, "    \"compiler\": \"%s\",\n", json_escape(plat.compiler).c_str());
    (void)std::fprintf(f, "    \"build_type\": \"%s\"\n", json_escape(plat.build_type).c_str());
    (void)std::fprintf(f, "  },\n");
    (void)std::fprintf(f, "  \"summary\": {\n");
    (void)std::fprintf(f, "    \"total_modules\": %d,\n", (int)results.size() + 1);
    (void)std::fprintf(f, "    \"passed\": %d,\n", total_pass);
    (void)std::fprintf(f, "    \"failed\": %d,\n", total_fail);
    (void)std::fprintf(f, "    \"advisory_warnings\": %d,\n", total_advisory);
    (void)std::fprintf(f, "    \"advisory_skipped\": %d,\n", total_advisory_skipped);
    (void)std::fprintf(f, "    \"advisory_failed\": %d,\n", total_advisory_failed);
    (void)std::fprintf(f, "    \"all_passed\": %s,\n", (total_fail == 0) ? "true" : "false");
    (void)std::fprintf(f, "    \"total_time_ms\": %.1f,\n", total_ms);
    // Verdict rules (2026-04-17):
    //   AUDIT-BLOCKED       -- any non-advisory module failed
    //   AUDIT-READY         -- everything mandatory passed, no advisory failures
    //   AUDIT-READY-DEGRADED -- mandatory modules passed, but one or more
    //                          advisory modules ran and failed; the report is
    //                          usable but no longer a clean pass.
    const char* verdict;
    if (total_fail != 0) {
        verdict = "AUDIT-BLOCKED";
    } else if (total_advisory_failed != 0) {
        verdict = "AUDIT-READY-DEGRADED";
    } else {
        verdict = "AUDIT-READY";
    }
    (void)std::fprintf(f, "    \"audit_verdict\": \"%s\"\n", verdict);
    (void)std::fprintf(f, "  },\n");

    // Selftest
    (void)std::fprintf(f, "  \"selftest\": {\n");
    (void)std::fprintf(f, "    \"passed\": %s,\n", selftest_passed ? "true" : "false");
    (void)std::fprintf(f, "    \"time_ms\": %.1f\n", selftest_ms);
    (void)std::fprintf(f, "  },\n");

    // Sections summary
    (void)std::fprintf(f, "  \"sections\": [\n");
    for (int s = 0; s < (int)sections.size(); ++s) {
        auto& sec = sections[s];
        (void)std::fprintf(f, "    {\n");
        (void)std::fprintf(f, "      \"id\": \"%s\",\n", sec.section_id);
        (void)std::fprintf(f, "      \"title\": \"%s\",\n", json_escape(sec.title_en).c_str());
        (void)std::fprintf(f, "      \"total\": %d,\n", sec.total);
        (void)std::fprintf(f, "      \"passed\": %d,\n", sec.passed);
        (void)std::fprintf(f, "      \"failed\": %d,\n", sec.failed);
        (void)std::fprintf(f, "      \"advisory_skipped\": %d,\n", sec.advisory_skipped);
        (void)std::fprintf(f, "      \"advisory_failed\": %d,\n", sec.advisory_failed);
        (void)std::fprintf(f, "      \"time_ms\": %.1f,\n", sec.time_ms);
        (void)std::fprintf(f, "      \"status\": \"%s\",\n", (sec.failed == 0) ? "PASS" : "FAIL");

        // Nested modules for this section
        (void)std::fprintf(f, "      \"modules\": [\n");
        bool first = true;
        for (auto& r : results) {
            if (std::strcmp(r.section, sec.section_id) != 0) continue;
            if (!first) (void)std::fprintf(f, ",\n");
            first = false;
            (void)std::fprintf(f, "        { \"id\": \"%s\", \"name\": \"%s\", \"passed\": %s, \"advisory\": %s, \"return_code\": %d, \"time_ms\": %.1f }",
                         r.id, json_escape(r.name).c_str(),
                         r.passed ? "true" : "false",
                         r.advisory ? "true" : "false", r.return_code, r.elapsed_ms);
        }
        (void)std::fprintf(f, "\n      ]\n");
        (void)std::fprintf(f, "    }%s\n", (s + 1 < (int)sections.size()) ? "," : "");
    }
    (void)std::fprintf(f, "  ]\n");
    (void)std::fprintf(f, "}\n");

    (void)std::fclose(f);
}

// ============================================================================
// Report writer -- Text (structured by 8 sections)
// ============================================================================
static void write_text_report(const char* path,
                               const PlatformInfo& plat,
                               const std::vector<ModuleResult>& results,
                               bool selftest_passed,
                               double selftest_ms,
                               double total_ms) {
#ifdef _WIN32
    FILE* f = std::fopen(path, "w");
#else
    int const fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    FILE* f = (fd >= 0) ? fdopen(fd, "w") : nullptr;
#endif
    if (!f) {
        (void)std::fprintf(stderr, "WARNING: Cannot open %s for writing\n", path);
        return;
    }

    int total_pass = 0, total_fail = 0, total_advisory = 0, total_advisory_failed = 0;
    for (auto& r : results) {
        if (r.passed) {
            ++total_pass;
        } else if (r.advisory) {
            ++total_advisory;
            if (r.return_code != ADVISORY_SKIP_CODE) {
                ++total_advisory_failed;
            }
        } else {
            ++total_fail;
        }
    }
    if (selftest_passed) { ++total_pass; } else { ++total_fail; }

    auto sections = compute_section_summaries(results);

    (void)std::fprintf(f, "================================================================\n");
    (void)std::fprintf(f, "  UltrafastSecp256k1 -- Industrial Self-Audit Report\n");
    (void)std::fprintf(f, "================================================================\n\n");
    (void)std::fprintf(f, "Library:    UltrafastSecp256k1 v%s\n", plat.library_version.c_str());
    (void)std::fprintf(f, "Git Hash:   %s\n", plat.git_hash.c_str());
    (void)std::fprintf(f, "Framework:  Audit Framework v%s\n", plat.framework_version.c_str());
    (void)std::fprintf(f, "Timestamp:  %s\n", plat.timestamp.c_str());
    (void)std::fprintf(f, "OS:         %s\n", plat.os.c_str());
    (void)std::fprintf(f, "Arch:       %s\n", plat.arch.c_str());
    (void)std::fprintf(f, "Compiler:   %s\n", plat.compiler.c_str());
    (void)std::fprintf(f, "Build:      %s\n", plat.build_type.c_str());
    (void)std::fprintf(f, "\n");

    // -- Library selftest ---
    (void)std::fprintf(f, "----------------------------------------------------------------\n");
    (void)std::fprintf(f, "  [0] Library Selftest (core KAT)          %s  (%.0f ms)\n",
                 selftest_passed ? "PASS" : "FAIL", selftest_ms);
    (void)std::fprintf(f, "----------------------------------------------------------------\n\n");

    // -- 8 Sections ---
    int module_idx = 1;
    for (int s = 0; s < (int)sections.size(); ++s) {
        auto& sec = sections[s];
        (void)std::fprintf(f, "================================================================\n");
        (void)std::fprintf(f, "  Section %d/%d: %s\n", s + 1, NUM_SECTIONS, sec.title_en);
        (void)std::fprintf(f, "================================================================\n");

        for (auto& r : results) {
            if (std::strcmp(r.section, sec.section_id) != 0) continue;
            const char* status = r.passed ? "PASS" : (r.advisory ? "WARN" : "FAIL");
            (void)std::fprintf(f, "  [%2d] %-45s %s  (%.0f ms)\n",
                         module_idx++, r.name,
                         status, r.elapsed_ms);
        }

        (void)std::fprintf(f, "  -------- Section Result: %d/%d passed", sec.passed, sec.total);
        if (sec.failed > 0) (void)std::fprintf(f, " (%d FAILED)", sec.failed);
        (void)std::fprintf(f, " (%.0f ms)\n\n", sec.time_ms);
    }

    // -- Grand total ---
    int const total_count = total_pass + total_fail + total_advisory;
    const char* text_verdict;
    if (total_fail != 0) {
        text_verdict = "AUDIT-BLOCKED (FAILURES DETECTED)";
    } else if (total_advisory_failed != 0) {
        text_verdict = "AUDIT-READY-DEGRADED";
    } else {
        text_verdict = "AUDIT-READY";
    }
    (void)std::fprintf(f, "================================================================\n");
    (void)std::fprintf(f, "  AUDIT VERDICT: %s\n", text_verdict);
    (void)std::fprintf(f, "  TOTAL: %d/%d modules passed", total_pass, total_count);
    if (total_advisory > 0) {
        (void)std::fprintf(f, "  (%d advisory warnings)", total_advisory);
    }
    (void)std::fprintf(f, "  (%.1f s)\n", total_ms / 1000.0);
    (void)std::fprintf(f, "  Platform: %s %s | %s | %s\n",
                 plat.os.c_str(), plat.arch.c_str(),
                 plat.compiler.c_str(), plat.build_type.c_str());
    (void)std::fprintf(f, "================================================================\n");

    (void)std::fclose(f);
}

// ============================================================================
// Report writer -- SARIF v2.1.0 (for GitHub Code Scanning integration)
// ============================================================================
// SARIF (Static Analysis Results Interchange Format) output enables
// GitHub Advanced Security code scanning alerts from audit failures.
// Upload with: github/codeql-action/upload-sarif@v3
// ============================================================================
static void write_sarif_report(const char* path,
                                const PlatformInfo& plat,
                                const std::vector<ModuleResult>& results,
                                bool selftest_passed,
                                double /* selftest_ms */,
                                double /* total_ms */) {
#ifdef _WIN32
    FILE* f = std::fopen(path, "w");
#else
    int const fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    FILE* f = (fd >= 0) ? fdopen(fd, "w") : nullptr;
#endif
    if (!f) {
        (void)std::fprintf(stderr, "WARNING: Cannot open %s for SARIF writing\n", path);
        return;
    }

    // Collect failed modules (non-advisory) as SARIF results
    // Advisory warnings become "warning" level; hard failures become "error"
    int result_count = 0;

    (void)std::fprintf(f, "{\n");
    (void)std::fprintf(f, "  \"$schema\": \"https://raw.githubusercontent.com/oasis-tcs/sarif-spec/main/sarif-2.1/schema/sarif-schema-2.1.0.json\",\n");
    (void)std::fprintf(f, "  \"version\": \"2.1.0\",\n");
    (void)std::fprintf(f, "  \"runs\": [\n");
    (void)std::fprintf(f, "    {\n");
    (void)std::fprintf(f, "      \"tool\": {\n");
    (void)std::fprintf(f, "        \"driver\": {\n");
    (void)std::fprintf(f, "          \"name\": \"UltrafastSecp256k1 Audit Runner\",\n");
    (void)std::fprintf(f, "          \"version\": \"%s\",\n", json_escape(plat.library_version).c_str());
    (void)std::fprintf(f, "          \"semanticVersion\": \"%s\",\n", json_escape(plat.framework_version).c_str());
    (void)std::fprintf(f, "          \"informationUri\": \"https://github.com/shrec/UltrafastSecp256k1\",\n");
    (void)std::fprintf(f, "          \"rules\": [\n");

    (void)std::fprintf(f, "            {\n");
    (void)std::fprintf(f, "              \"id\": \"AUDIT/selftest\",\n");
    (void)std::fprintf(f, "              \"name\": \"Library selftest (core KAT)\",\n");
    (void)std::fprintf(f, "              \"shortDescription\": { \"text\": \"Library selftest (core KAT)\" },\n");
    (void)std::fprintf(f, "              \"defaultConfiguration\": { \"level\": \"error\" },\n");
    (void)std::fprintf(f, "              \"properties\": { \"section\": \"selftest\" }\n");
    (void)std::fprintf(f, "            },\n");

    // Emit rule definitions for all modules
    for (int i = 0; i < NUM_MODULES; ++i) {
        auto& m = ALL_MODULES[i];
        (void)std::fprintf(f, "            {\n");
        (void)std::fprintf(f, "              \"id\": \"AUDIT/%s\",\n", m.id);
        (void)std::fprintf(f, "              \"name\": \"%s\",\n", json_escape(m.name).c_str());
        (void)std::fprintf(f, "              \"shortDescription\": { \"text\": \"%s\" },\n", json_escape(m.name).c_str());
        (void)std::fprintf(f, "              \"defaultConfiguration\": { \"level\": \"%s\" },\n",
                     m.advisory ? "warning" : "error");
        (void)std::fprintf(f, "              \"properties\": { \"section\": \"%s\" }\n", m.section);
        (void)std::fprintf(f, "            }%s\n", (i + 1 < NUM_MODULES) ? "," : "");
    }
    (void)std::fprintf(f, "          ]\n");
    (void)std::fprintf(f, "        }\n");
    (void)std::fprintf(f, "      },\n");

    // Results array: only failed modules produce SARIF results
    (void)std::fprintf(f, "      \"results\": [\n");
    bool first_result = true;

    // Selftest failure
    if (!selftest_passed) {
        (void)std::fprintf(f, "        {\n");
        (void)std::fprintf(f, "          \"ruleId\": \"AUDIT/selftest\",\n");
        (void)std::fprintf(f, "          \"level\": \"error\",\n");
        (void)std::fprintf(f, "          \"message\": { \"text\": \"Library selftest (core KAT) FAILED\" },\n");
        (void)std::fprintf(f, "          \"locations\": [{ \"physicalLocation\": { \"artifactLocation\": { \"uri\": \"src/cpu/include/secp256k1/selftest.hpp\" } } }]\n");
        (void)std::fprintf(f, "        }");
        first_result = false;
        ++result_count;
    }

    for (auto& r : results) {
        if (r.passed) {
            continue;
        }
        if (!first_result) {
            (void)std::fprintf(f, ",\n");
        } else {
            (void)std::fprintf(f, "\n");
        }
        first_result = false;

        const char* level = r.advisory ? "warning" : "error";
        // Map section to a representative source file
        const char* uri = "audit/unified_audit_runner.cpp";
        if (std::strcmp(r.section, "math_invariants") == 0) {
            uri = "src/cpu/src/field.cpp";
        } else if (std::strcmp(r.section, "ct_analysis") == 0) {
            uri = "src/cpu/include/secp256k1/ct/ops.hpp";
        } else if (std::strcmp(r.section, "standard_vectors") == 0) {
            uri = "audit/test_cross_platform_kat.cpp";
        } else if (std::strcmp(r.section, "protocol_security") == 0) {
            uri = "src/cpu/src/musig2.cpp";
        } else if (std::strcmp(r.section, "fuzzing") == 0) {
            uri = "audit/audit_fuzz.cpp";
        } else if (std::strcmp(r.section, "memory_safety") == 0) {
            uri = "audit/test_abi_gate.cpp";
        } else if (std::strcmp(r.section, "performance") == 0) {
            uri = "src/cpu/bench/bench_unified.cpp";
        } else if (std::strcmp(r.section, "differential") == 0) {
            uri = "audit/test_fiat_crypto_vectors.cpp";
        } else if (std::strcmp(r.section, "exploit_poc") == 0) {
            uri = "audit/test_exploit_ecdsa_malleability.cpp";
        }

        (void)std::fprintf(f, "        {\n");
        (void)std::fprintf(f, "          \"ruleId\": \"AUDIT/%s\",\n", r.id);
        (void)std::fprintf(f, "          \"level\": \"%s\",\n", level);
        (void)std::fprintf(f, "          \"message\": { \"text\": \"Audit module '%s' FAILED (section: %s, %.0f ms)\" },\n",
                     json_escape(r.name).c_str(), r.section, r.elapsed_ms);
        (void)std::fprintf(f, "          \"locations\": [{ \"physicalLocation\": { \"artifactLocation\": { \"uri\": \"%s\" } } }]\n", uri);
        (void)std::fprintf(f, "        }");
        ++result_count;
    }

    (void)std::fprintf(f, "\n      ],\n");

    // Invocation properties
    (void)std::fprintf(f, "      \"invocations\": [\n");
    (void)std::fprintf(f, "        {\n");
    (void)std::fprintf(f, "          \"executionSuccessful\": true,\n");
    (void)std::fprintf(f, "          \"toolExecutionNotifications\": []\n");
    (void)std::fprintf(f, "        }\n");
    (void)std::fprintf(f, "      ],\n");

    // Properties
    (void)std::fprintf(f, "      \"properties\": {\n");
    (void)std::fprintf(f, "        \"platform\": \"%s %s\",\n", plat.os.c_str(), plat.arch.c_str());
    (void)std::fprintf(f, "        \"compiler\": \"%s\",\n", json_escape(plat.compiler).c_str());
    (void)std::fprintf(f, "        \"gitHash\": \"%s\"\n", json_escape(plat.git_hash).c_str());
    (void)std::fprintf(f, "      }\n");
    (void)std::fprintf(f, "    }\n");
    (void)std::fprintf(f, "  ]\n");
    (void)std::fprintf(f, "}\n");

    (void)std::fclose(f);
}

// ============================================================================
// Resolve output directory (executable dir by default)
// ============================================================================
static std::string get_exe_dir() {
#ifdef _WIN32
    char buf[MAX_PATH] = {};
    GetModuleFileNameA(nullptr, buf, MAX_PATH);
    std::string const path(buf);
    auto pos = path.find_last_of("\\/");
    return (pos != std::string::npos) ? path.substr(0, pos) : ".";
#else
    char buf[4096] = {};
    ssize_t const len = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (len <= 0) return ".";
    buf[len] = '\0';
    std::string const path(buf);
    auto pos = path.find_last_of('/');
    return (pos != std::string::npos) ? path.substr(0, pos) : ".";
#endif
}

// ============================================================================
// Main
// ============================================================================
static void print_usage() {
    std::printf("Usage: unified_audit_runner [OPTIONS]\n\n");
    std::printf("Options:\n");
    std::printf("  --json-only            Suppress console output; write JSON only\n");
    std::printf("  --sarif                Also generate SARIF v2.1.0 report (for GitHub Code Scanning)\n");
    std::printf("  --report-dir <dir>     Write reports to <dir> (default: exe dir)\n");
    std::printf("  --section <id>         Run only modules in section <id>\n");
    std::printf("  --list-sections        Print available sections and exit\n");
    std::printf("  --help                 Show this message\n\n");
    std::printf("Sections:\n");
    for (int s = 0; s < NUM_SECTIONS; ++s) {
        std::printf("  %-20s %s\n", SECTIONS[s].id, SECTIONS[s].title_en);
    }
}

int main(int argc, char* argv[]) {
    // Disable full-buffering so sub-test progress appears in real-time
    // (CTest / Docker / CI runners buffer stdout when it is not a TTY)
#ifdef _WIN32
    (void)std::setvbuf(stdout, nullptr, _IONBF, 0);  // Windows: unbuffered
#else
    (void)std::setvbuf(stdout, nullptr, _IOLBF, 0);  // POSIX: line-buffered
#endif

    // Parse args
    bool json_only = false;
    bool sarif_enabled = false;
    std::string report_dir = "";
    std::string section_filter = "";  // empty = run all
    {
        int i = 1;
        while (i < argc) {
            if (std::strcmp(argv[i], "--json-only") == 0) {
                json_only = true;
                ++i;
            } else if (std::strcmp(argv[i], "--sarif") == 0) {
                sarif_enabled = true;
                ++i;
            } else if (std::strcmp(argv[i], "--report-dir") == 0 && i + 1 < argc) {
                report_dir = argv[i + 1];
                i += 2;
            } else if (std::strcmp(argv[i], "--section") == 0 && i + 1 < argc) {
                section_filter = argv[i + 1];
                i += 2;
            } else if (std::strcmp(argv[i], "--list-sections") == 0) {
                for (int s = 0; s < NUM_SECTIONS; ++s) {
                    std::printf("%s\n", SECTIONS[s].id);
                }
                return 0;
            } else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
                print_usage();
                return 0;
            } else {
                ++i;
            }
        }
    }
    if (report_dir.empty()) {
        report_dir = get_exe_dir();
    }

    // Validate section filter
    if (!section_filter.empty()) {
        bool found = false;
        for (int s = 0; s < NUM_SECTIONS; ++s) {
            if (section_filter == SECTIONS[s].id) { found = true; break; }
        }
        if (!found) {
            (void)std::fprintf(stderr, "ERROR: unknown section '%s'\n", section_filter.c_str());
            print_usage();
            return 1;
        }
    }

    auto plat = detect_platform();

    auto total_start = std::chrono::steady_clock::now();

    if (!json_only) {
        std::printf("================================================================\n");
        std::printf("  UltrafastSecp256k1 -- Unified Audit Runner\n");
        std::printf("  Library v%s  |  Git: %.8s  |  Framework v%s\n",
                    plat.library_version.c_str(), plat.git_hash.c_str(),
                    plat.framework_version.c_str());
        std::printf("  %s | %s | %s | %s\n",
                    plat.os.c_str(), plat.arch.c_str(),
                    plat.compiler.c_str(), plat.build_type.c_str());
        std::printf("  %s\n", plat.timestamp.c_str());
        if (!section_filter.empty()) {
            std::printf("  Filter: section=%s\n", section_filter.c_str());
}
        std::printf("================================================================\n\n");
    }

    // -- Phase 1: Library selftest ----------------------------------------
    if (!json_only) std::printf("[Phase 1/3] Library selftest (ci mode)...\n");
    auto st_start = std::chrono::steady_clock::now();
    bool const selftest_passed = Selftest(false, SelftestMode::ci, 0);
    auto st_end = std::chrono::steady_clock::now();
    double const selftest_ms = std::chrono::duration<double, std::milli>(st_end - st_start).count();

    if (!json_only) {
        if (selftest_passed) {
            std::printf("[Phase 1/3] Selftest PASSED (%.0f ms)\n\n", selftest_ms);
        } else {
            std::printf("[Phase 1/3] *** Selftest FAILED *** (%.0f ms)\n\n", selftest_ms);
        }
    }

    // -- Phase 2: All test modules (grouped by 8 sections) ----------------
    // Count modules to run (with filter)
    int modules_to_run = 0;
    for (int i = 0; i < NUM_MODULES; ++i) {
        if (section_filter.empty() || section_filter == ALL_MODULES[i].section) {
            ++modules_to_run;
}
    }

    if (!json_only) {
        std::printf("[Phase 2/3] Running %d test modules across %d audit sections...\n\n",
                    modules_to_run, NUM_SECTIONS);
    }

    std::vector<ModuleResult> results;
    results.reserve(NUM_MODULES);

    int modules_passed = 0;
    int modules_failed = 0;
    int modules_advisory_warned = 0;  // skips + advisory failures combined
    int modules_advisory_failed = 0;  // advisory actual failures (rc != SKIP_CODE, rc != 0)

    // Track which section we're in for console grouping
    const char* current_section = "";
    int section_num = 0;
    int run_idx = 0;

    for (int i = 0; i < NUM_MODULES; ++i) {
        auto& m = ALL_MODULES[i];

        // Apply section filter
        if (!section_filter.empty() && section_filter != m.section) {
            continue;
}

        // Print section header on transition
        if (!json_only && std::strcmp(m.section, current_section) != 0) {
            current_section = m.section;
            ++section_num;
            // Find the section title
            for (int s = 0; s < NUM_SECTIONS; ++s) {
                if (std::strcmp(SECTIONS[s].id, current_section) == 0) {
                    std::printf("  ----------------------------------------------------------\n");
                    std::printf("  Section %d/%d: %s\n", section_num, NUM_SECTIONS, SECTIONS[s].title_en);
                    std::printf("  ----------------------------------------------------------\n");
                    break;
                }
            }
        }

        ++run_idx;
        if (!json_only) {
            std::printf("  [%2d/%d] %-45s ", run_idx, modules_to_run, m.name);
            (void)std::fflush(stdout);
        }

        reset_fixed_base_state_for_module();

        auto t0 = std::chrono::steady_clock::now();
        int const rc = m.run();
        auto t1 = std::chrono::steady_clock::now();
        double const ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        bool const ok = (rc == 0);
        if (ok) {
            ++modules_passed;
            if (!json_only) std::printf("PASS  (%.0f ms)\n", ms);
        } else if (m.advisory) {
            ++modules_advisory_warned;
            if (rc == ADVISORY_SKIP_CODE) {
                if (!json_only) std::printf("SKIP  (%.0f ms) [advisory — infrastructure absent]\n", ms);
            } else {
                ++modules_advisory_failed;  // actual advisory failure, not a skip
                if (!json_only) std::printf("WARN  (%.0f ms) [advisory]\n", ms);
            }
        } else {
            ++modules_failed;
            if (!json_only) std::printf("FAIL  (%.0f ms)\n", ms);
        }

        results.push_back({ m.id, m.name, m.section, ok, m.advisory, ms, rc });
    }

    auto total_end = std::chrono::steady_clock::now();
    double const total_ms = std::chrono::duration<double, std::milli>(total_end - total_start).count();

    // -- Phase 3: Generate reports ---------------------------------------
    if (!json_only) std::printf("\n[Phase 3/3] Generating audit reports...\n");

    std::string const json_path = report_dir + "/audit_report.json";
    std::string const text_path = report_dir + "/audit_report.txt";

    write_json_report(json_path.c_str(), plat, results, selftest_passed, selftest_ms, total_ms);
    if (!json_only) {
        write_text_report(text_path.c_str(), plat, results, selftest_passed, selftest_ms, total_ms);
    }

    // SARIF report (for GitHub Code Scanning)
    std::string sarif_path;
    if (sarif_enabled) {
        sarif_path = report_dir + "/audit_report.sarif";
        write_sarif_report(sarif_path.c_str(), plat, results, selftest_passed, selftest_ms, total_ms);
    }

    if (!json_only) {
        std::printf("  JSON:  %s\n", json_path.c_str());
        std::printf("  Text:  %s\n", text_path.c_str());
        if (sarif_enabled) {
            std::printf("  SARIF: %s\n", sarif_path.c_str());
        }
    }

    // -- Section Summary Table -------------------------------------------
    auto sections = compute_section_summaries(results);

    if (!json_only) {
        std::printf("\n================================================================\n");
        std::printf("  %-4s %-50s %s\n", "#", "Audit Section", "Result");
        std::printf("  ---- -------------------------------------------------- ------\n");
        for (int s = 0; s < (int)sections.size(); ++s) {
            auto& sec = sections[s];
            if (sec.total == 0) continue;  // skip empty sections (filtered)
            std::printf("  %-4d %-50s %d/%d %s\n",
                        s + 1, sec.title_en, sec.passed, sec.total,
                        sec.failed == 0 ? "PASS" : "FAIL");
        }
    }

    // -- Final Summary ---------------------------------------------------
    int const total_pass = modules_passed + (selftest_passed ? 1 : 0);
    int const total_fail = modules_failed + (selftest_passed ? 0 : 1);
    int const total_count = total_pass + total_fail + modules_advisory_warned;

    if (!json_only) {
        std::printf("\n================================================================\n");
        // F-13 fix: match the three-way verdict that the JSON report emits.
        // Previously the console always said "AUDIT-READY" when total_fail == 0,
        // even when advisory modules had failed (AUDIT-READY-DEGRADED).
        // Use modules_advisory_failed (not modules_advisory_warned) so that
        // pure advisory skips (infrastructure absent) do not cause DEGRADED.
        const char* console_verdict =
            (total_fail > 0)                ? "AUDIT-BLOCKED"        :
            (modules_advisory_failed > 0)   ? "AUDIT-READY-DEGRADED" :
                                              "AUDIT-READY";
        std::printf("  AUDIT VERDICT: %s\n", console_verdict);
        std::printf("  TOTAL: %d/%d modules passed", total_pass, total_count);
        if (total_fail == 0) {
            std::printf("  --  ALL PASSED");
        } else {
            std::printf("  --  %d FAILED", total_fail);
        }
        if (modules_advisory_warned > 0) {
            std::printf("  (%d advisory warnings)", modules_advisory_warned);
        }
        std::printf("  (%.1f s)\n", total_ms / 1000.0);
        std::printf("  Platform: %s %s | %s | %s\n",
                    plat.os.c_str(), plat.arch.c_str(),
                    plat.compiler.c_str(), plat.build_type.c_str());
        std::printf("================================================================\n");
    }

    return total_fail > 0 ? 1 : 0;
}
