// shim_run_stubs_unified.cpp
// Stub _run() functions for shim-dependent regression tests.
// Used when secp256k1_shim is NOT linked into unified_audit_runner (CI
// without SECP256K1_BUILD_SHIM=OFF).  Returns ADVISORY_SKIP_CODE (77)
// so the unified runner reports them as advisory-skipped, not failures.
//
// When secp256k1_shim IS available, the real .cpp files are compiled
// instead (see audit/CMakeLists.txt — if(TARGET secp256k1_shim) block).

#ifndef ADVISORY_SKIP_CODE
#define ADVISORY_SKIP_CODE 77
#endif

int test_shim_der_zero_r_run()               { return ADVISORY_SKIP_CODE; }
int test_regression_schnorr_r_zero_ct_run()   { return ADVISORY_SKIP_CODE; }
int test_shim_null_ctx_run()                 { return ADVISORY_SKIP_CODE; }
int test_regression_shim_high_s_verify_run() { return ADVISORY_SKIP_CODE; }
int test_regression_shim_perf_correctness_run() { return ADVISORY_SKIP_CODE; }
int test_regression_bip32_private_key_strict_run()  { return ADVISORY_SKIP_CODE; }
int test_regression_ellswift_ct_path_run()          { return ADVISORY_SKIP_CODE; }
int test_regression_musig2_nonce_strict_run()        { return ADVISORY_SKIP_CODE; }
int test_regression_shim_pubkey_sort_run()           { return ADVISORY_SKIP_CODE; }
int test_regression_shim_per_context_blinding_run()  { return ADVISORY_SKIP_CODE; }
int test_regression_musig2_session_token_run()       { return ADVISORY_SKIP_CODE; }
int test_regression_shim_security_v7_run()           { return ADVISORY_SKIP_CODE; }
int test_regression_shim_security_v8_run()           { return ADVISORY_SKIP_CODE; }
int test_regression_shim_preallocated_ctx_run()      { return ADVISORY_SKIP_CODE; }
int test_regression_shim_rgrind_functional_run()     { return ADVISORY_SKIP_CODE; }
int test_regression_ecdsa_batch_curve_check_run()    { return ADVISORY_SKIP_CODE; }
int test_shim_security_edge_cases_run()              { return ADVISORY_SKIP_CODE; }
int test_exploit_musig2_infinity_pubnonce_run()      { return ADVISORY_SKIP_CODE; }
int test_regression_musig2_nonce_gen_seckey_run()    { return ADVISORY_SKIP_CODE; }
int test_regression_shim_ndata_rgrind_run()          { return ADVISORY_SKIP_CODE; }
int test_regression_schnorr_varlen_ct_fixes_run()    { return ADVISORY_SKIP_CODE; }
// PASS3-001/002: shim recovery parse compat + custom-noncefp illegal_callback
int test_shim_recovery_and_noncefp_run()             { return ADVISORY_SKIP_CODE; }
// SHIM-013: ecdsa_verify cache consistency (CVC-1..3) — real test in
// audit/test_regression_ecdsa_verify_cache_consistency.cpp links into the
// runner only when SECP256K1_BUILD_COMPAT_SHIM is defined.
int test_regression_ecdsa_verify_cache_consistency_run() { return ADVISORY_SKIP_CODE; }
// SHIM-NEW-012/015: serialize NULL callbacks + seckey NULL callbacks
// 2026-05-24: weak attribute removed (MSVC ignores it, causing duplicate-symbol
// link errors). Strong symbol. The real test_regression_shim_security_v9.cpp's
// __else branch deliberately defines NO body when __has_include fails, so this
// stub is the sole definer in that case. When __has_include succeeds (shim
// path reachable), the test's real body is in target_sources but shim_run_stubs
// is NOT (CMakeLists if/else is mutually exclusive). No conflict either way.
int test_regression_shim_security_v9_run()           { return ADVISORY_SKIP_CODE; }
// test_regression_p2_ct_shim_fixes_run is always provided by test_regression_p2_ct_shim_fixes.cpp
// SHIM-NONCEGEN-001: extra_input32 behavioral freeze (secp256k1_musig_nonce_gen ignores extra_input32)
int test_regression_musig_noncegen_extra_input_run() { return ADVISORY_SKIP_CODE; }
// T-09/10: NULL-arg illegal_callback for keypair_create/sec/pub/xonly_pub + sig parse (2026-05-25)
int test_regression_shim_keypair_null_cb_run() { return ADVISORY_SKIP_CODE; }
// T-11/12: memset→secure_erase + is_zero→is_zero_ct in shim_context.cpp (2026-05-25)
int test_regression_shim_context_erase_run() { return ADVISORY_SKIP_CODE; }
// ILLCB-001/002, DER-STRICT, keypair_sec BIP-340 normalization (2026-05-26)
int test_regression_shim_divergence_fixes_run() { return ADVISORY_SKIP_CODE; }
// SHIM-NEW-004: NULL ctx rejected by pubnonce_serialize/parse, nonce_agg, nonce_process (2026-05-26)
int test_regression_shim_musig_null_ctx_run()   { return ADVISORY_SKIP_CODE; }
// SHIM-NEW-001/002/003: NULL non-ctx args fire illegal_callback — pubkey_create/serialize,
// xonly_pubkey_parse/serialize, recoverable_sig_parse/serialize (2026-05-26)
int test_regression_shim_null_arg_cb_run()      { return ADVISORY_SKIP_CODE; }
// TRNC-1..4: NULL non-ctx args fire illegal_callback in xonly_tweak_add, tweak_add_check,
// keypair_xonly_tweak_add, recoverable_sig_convert (2026-05-28, SHIM-NULL-CB-2026)
int test_regression_shim_tweak_recover_null_cb_run() { return ADVISORY_SKIP_CODE; }
