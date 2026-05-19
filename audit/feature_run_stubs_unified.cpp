// feature_run_stubs_unified.cpp
// Stub _run() functions for optional-module test functions.
// When a module is disabled (SECP256K1_BUILD_FROST=OFF etc.) the real
// .cpp files are excluded from unified_audit_runner and these stubs are
// included instead.  Each stub returns ADVISORY_SKIP_CODE (77) so the
// runner reports them as advisory-skipped rather than link errors.

// Pull in the generated feature flags (SECP256K1_HAS_FROST etc.)
#if __has_include("secp256k1/secp256k1_features.h")
#include "secp256k1/secp256k1_features.h"
#endif

#ifndef SECP256K1_HAS_FROST
#define SECP256K1_HAS_FROST 0
#endif
#ifndef SECP256K1_HAS_ZK
#define SECP256K1_HAS_ZK 0
#endif
#ifndef SECP256K1_HAS_ECIES
#define SECP256K1_HAS_ECIES 0
#endif
#ifndef SECP256K1_HAS_BIP352
#define SECP256K1_HAS_BIP352 0
#endif
#ifndef SECP256K1_HAS_ADAPTOR
#define SECP256K1_HAS_ADAPTOR 0
#endif
#ifndef SECP256K1_HAS_WALLET
#define SECP256K1_HAS_WALLET 0
#endif

#ifndef ADVISORY_SKIP_CODE
#define ADVISORY_SKIP_CODE 77
#endif

// ============================================================================
// FROST stubs — active when SECP256K1_BUILD_FROST=OFF
// ============================================================================
#if !SECP256K1_HAS_FROST
// test_ct_sidechannel.cpp includes FROST for its FROST CT timing test
int test_ct_sidechannel_smoke_run()             { return ADVISORY_SKIP_CODE; }
// selftest stubs
int test_musig2_frost_protocol_run()        { return ADVISORY_SKIP_CODE; }
int test_musig2_frost_advanced_run()        { return ADVISORY_SKIP_CODE; }
int test_frost_kat_run()                    { return ADVISORY_SKIP_CODE; }
int test_musig2_bip327_vectors_run()        { return ADVISORY_SKIP_CODE; }
// fuzz stubs
int test_fuzz_musig2_frost_run()            { return ADVISORY_SKIP_CODE; }
// exploit stubs
int test_exploit_frost_binding_factor_mismatch_run()    { return ADVISORY_SKIP_CODE; }
int test_exploit_frost_byzantine_run()                  { return ADVISORY_SKIP_CODE; }
int test_exploit_frost_commitment_reuse_run()           { return ADVISORY_SKIP_CODE; }
int test_exploit_frost_dkg_run()                        { return ADVISORY_SKIP_CODE; }
int test_exploit_frost_index_run()                      { return ADVISORY_SKIP_CODE; }
int test_exploit_frost_lagrange_duplicate_run()         { return ADVISORY_SKIP_CODE; }
int test_exploit_frost_participant_zero_run()           { return ADVISORY_SKIP_CODE; }
int test_exploit_frost_signing_run()                    { return ADVISORY_SKIP_CODE; }
int test_exploit_frost_threshold_degenerate_run()       { return ADVISORY_SKIP_CODE; }
int test_exploit_frost_ct_nonce_run()                   { return ADVISORY_SKIP_CODE; }
int test_exploit_frost_participant_set_malleability_run(){ return ADVISORY_SKIP_CODE; }
int test_exploit_frost_adaptive_corruption_run()        { return ADVISORY_SKIP_CODE; }
int test_exploit_frost_weak_binding_run()               { return ADVISORY_SKIP_CODE; }
int test_exploit_frost_rogue_key_run()                  { return ADVISORY_SKIP_CODE; }
int test_exploit_frost_identifiable_abort_run()         { return ADVISORY_SKIP_CODE; }
// regression stubs
int test_exploit_frost_ocl_shim_bip32_ct_regression_run() { return ADVISORY_SKIP_CODE; }
int test_exploit_frost_secret_share_ct_run()            { return ADVISORY_SKIP_CODE; }
int test_regression_frost_threshold_zero_run()          { return ADVISORY_SKIP_CODE; }
#endif // !SECP256K1_HAS_FROST

// ============================================================================
// ZK stubs — active when SECP256K1_BUILD_ZK=OFF
// ============================================================================
#if !SECP256K1_HAS_ZK
int audit_zk_run()                              { return ADVISORY_SKIP_CODE; }
int test_exploit_pedersen_adversarial_run()     { return ADVISORY_SKIP_CODE; }
int test_exploit_pedersen_homomorphism_run()    { return ADVISORY_SKIP_CODE; }
int test_exploit_pedersen_switch_misuse_run()   { return ADVISORY_SKIP_CODE; }
int test_exploit_zk_adversarial_run()           { return ADVISORY_SKIP_CODE; }
int test_exploit_zk_proofs_run()                { return ADVISORY_SKIP_CODE; }
int test_exploit_zk_new_schemes_run()           { return ADVISORY_SKIP_CODE; }
int test_exploit_fiat_shamir_frozen_heart_run() { return ADVISORY_SKIP_CODE; }
#endif // !SECP256K1_HAS_ZK

// ============================================================================
// FROST+ZK combined stub (v4_features requires both)
// ============================================================================
#if !SECP256K1_HAS_FROST || !SECP256K1_HAS_ZK
int test_v4_features_run()                      { return ADVISORY_SKIP_CODE; }
#endif

// ============================================================================
// ECIES stubs — active when SECP256K1_BUILD_ECIES=OFF
// ============================================================================
#if !SECP256K1_HAS_ECIES
int test_ecies_regression_run()                 { return ADVISORY_SKIP_CODE; }
int test_exploit_ecies_auth_run()               { return ADVISORY_SKIP_CODE; }
int test_exploit_ecies_encryption_run()         { return ADVISORY_SKIP_CODE; }
int test_exploit_ecies_envelope_confusion_run() { return ADVISORY_SKIP_CODE; }
int test_exploit_ecies_roundtrip_run()          { return ADVISORY_SKIP_CODE; }
int test_exploit_ecies_ephemeral_reuse_run()    { return ADVISORY_SKIP_CODE; }
#endif // !SECP256K1_HAS_ECIES

// ============================================================================
// BIP-352 stubs — active when SECP256K1_BUILD_BIP352=OFF
// ============================================================================
#if !SECP256K1_HAS_BIP352
int test_exploit_silent_payment_confusion_run()  { return ADVISORY_SKIP_CODE; }
int test_exploit_bip352_scan_dos_run()           { return ADVISORY_SKIP_CODE; }
int test_exploit_bip352_parity_confusion_run()   { return ADVISORY_SKIP_CODE; }
int test_exploit_bip352_batch_correctness_run()  { return ADVISORY_SKIP_CODE; }
int test_exploit_bip352_address_collision_run()  { return ADVISORY_SKIP_CODE; }
int test_exploit_gpu_bip352_key_erase_run()      { return ADVISORY_SKIP_CODE; }
int test_regression_bip352_ct_varbase_run()      { return ADVISORY_SKIP_CODE; }
int test_regression_opencl_bip352_scan_key_boundary_run() { return ADVISORY_SKIP_CODE; }
// bip352_kat is selftest only, handled in cpu/tests
#endif // !SECP256K1_HAS_BIP352

// ============================================================================
// ADAPTOR stubs — active when SECP256K1_BUILD_ADAPTOR=OFF
// ============================================================================
#if !SECP256K1_HAS_ADAPTOR
int test_exploit_adaptor_extended_run()             { return ADVISORY_SKIP_CODE; }
int test_exploit_adaptor_parity_run()               { return ADVISORY_SKIP_CODE; }
int test_exploit_adaptor_extraction_soundness_run() { return ADVISORY_SKIP_CODE; }
int test_regression_adaptor_binding_domain_run()    { return ADVISORY_SKIP_CODE; }
int test_regression_adaptor_degenerate_v7_run()     { return ADVISORY_SKIP_CODE; }
#endif // !SECP256K1_HAS_ADAPTOR

// ============================================================================
// WALLET stubs — active when SECP256K1_BUILD_WALLET=OFF
// ============================================================================
#if !SECP256K1_HAS_WALLET
// selftest stubs
int test_bip32_run()                                    { return ADVISORY_SKIP_CODE; }
int test_bip39_run()                                    { return ADVISORY_SKIP_CODE; }
int test_bip32_vectors_run()                            { return ADVISORY_SKIP_CODE; }
// edge_cases and coins use wallet/BIP-32 symbols
int test_edge_cases_run()                               { return ADVISORY_SKIP_CODE; }
int test_coins_run()                                    { return ADVISORY_SKIP_CODE; }
// audit_security uses wallet symbols
int audit_security_run()                                { return ADVISORY_SKIP_CODE; }
// exploit stubs
int test_exploit_bip32_ckd_hardened_run()               { return ADVISORY_SKIP_CODE; }
int test_exploit_bip32_depth_run()                      { return ADVISORY_SKIP_CODE; }
int test_exploit_bip32_derivation_run()                 { return ADVISORY_SKIP_CODE; }
int test_exploit_bip32_parent_fingerprint_confusion_run(){ return ADVISORY_SKIP_CODE; }
int test_exploit_bip32_path_overflow_run()              { return ADVISORY_SKIP_CODE; }
int test_exploit_bip39_entropy_run()                    { return ADVISORY_SKIP_CODE; }
int test_exploit_bip39_mnemonic_run()                   { return ADVISORY_SKIP_CODE; }
int test_exploit_bip39_nfkd_run()                       { return ADVISORY_SKIP_CODE; }
int test_exploit_bitcoin_message_signing_run()          { return ADVISORY_SKIP_CODE; }
int test_exploit_coin_hd_derivation_run()               { return ADVISORY_SKIP_CODE; }
int test_exploit_wallet_api_run()                       { return ADVISORY_SKIP_CODE; }
int test_exploit_wallet_cross_domain_replay_run()       { return ADVISORY_SKIP_CODE; }
int test_exploit_bip32_child_key_attack_run()           { return ADVISORY_SKIP_CODE; }
int test_exploit_wallet_sign_ct_run()                   { return ADVISORY_SKIP_CODE; }
// test_regression_bip32_private_key_strict_run() is in shim_run_stubs_unified.cpp
#endif // !SECP256K1_HAS_WALLET
