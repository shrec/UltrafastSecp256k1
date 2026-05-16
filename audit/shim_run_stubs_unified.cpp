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
int test_regression_ecdsa_batch_curve_check_run()    { return ADVISORY_SKIP_CODE; }
