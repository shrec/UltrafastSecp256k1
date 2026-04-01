# ============================================================================
# audit/CMakeLists.txt -- Audit Infrastructure
# ============================================================================
#
# This directory contains everything needed for the library audit:
#   - Unified Audit Runner (unified execution + JSON/TXT report)
#   - Standalone CTest targets (CT, differential, fault injection, ...)
#   - Protocol tests (MuSig2, FROST, KAT)
#   - Fuzz / adversarial tests
#   - Cross-library differential tests (vs bitcoin-core/libsecp256k1)
#
# Core library tests (run_selftest) remain in cpu/tests/.
# ============================================================================

if(NOT BUILD_TESTING)
    return()
endif()

find_package(Threads REQUIRED)

# Shorthand for cpu/tests/ -- core library test sources reused by unified runner
set(CPU_TESTS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../cpu/tests)

# -- Helper: common link + stack options ------------------------------------
macro(audit_target_defaults target_name)
    target_link_libraries(${target_name} PRIVATE fastsecp256k1)
    if(MSVC OR (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND WIN32))
        target_link_options(${target_name} PRIVATE "LINKER:/STACK:8388608")
    endif()
endmacro()

# ===========================================================================
# Standalone CTest targets
# ===========================================================================

# -- dudect side-channel timing test ---------------------------------------
add_executable(test_ct_sidechannel_standalone test_ct_sidechannel.cpp)
audit_target_defaults(test_ct_sidechannel_standalone)
target_compile_definitions(test_ct_sidechannel_standalone PRIVATE STANDALONE_TEST)
add_test(NAME ct_sidechannel COMMAND test_ct_sidechannel_standalone)
set_tests_properties(ct_sidechannel PROPERTIES TIMEOUT 600)

# Smoke version of dudect (short run, relaxed threshold -- safe for CI)
add_executable(test_ct_sidechannel_smoke test_ct_sidechannel.cpp)
audit_target_defaults(test_ct_sidechannel_smoke)
target_compile_definitions(test_ct_sidechannel_smoke PRIVATE STANDALONE_TEST DUDECT_SMOKE)
add_test(NAME ct_sidechannel_smoke COMMAND test_ct_sidechannel_smoke)
set_tests_properties(ct_sidechannel_smoke PROPERTIES TIMEOUT 120)

# -- Differential/self-consistency test ------------------------------------
add_executable(test_differential_standalone differential_test.cpp)
audit_target_defaults(test_differential_standalone)
add_test(NAME differential COMMAND test_differential_standalone)
set_tests_properties(differential PROPERTIES TIMEOUT 600)

# -- FAST==CT equivalence test ---------------------------------------------
add_executable(test_ct_equivalence_standalone ${CPU_TESTS_DIR}/test_ct_equivalence.cpp)
audit_target_defaults(test_ct_equivalence_standalone)
target_compile_definitions(test_ct_equivalence_standalone PRIVATE STANDALONE_TEST)
add_test(NAME ct_equivalence COMMAND test_ct_equivalence_standalone)

# -- Fault injection simulation --------------------------------------------
add_executable(test_fault_injection test_fault_injection.cpp)
audit_target_defaults(test_fault_injection)
target_compile_definitions(test_fault_injection PRIVATE STANDALONE_TEST)
add_test(NAME fault_injection COMMAND test_fault_injection)
set_tests_properties(fault_injection PROPERTIES TIMEOUT 900)

# -- Debug invariant assertions --------------------------------------------
add_executable(test_debug_invariants test_debug_invariants.cpp)
audit_target_defaults(test_debug_invariants)
add_test(NAME debug_invariants COMMAND test_debug_invariants)
set_tests_properties(debug_invariants PROPERTIES TIMEOUT 600)

# -- Fiat-Crypto comparison vectors ----------------------------------------
add_executable(test_fiat_crypto_vectors test_fiat_crypto_vectors.cpp)
audit_target_defaults(test_fiat_crypto_vectors)
target_compile_definitions(test_fiat_crypto_vectors PRIVATE STANDALONE_TEST)
add_test(NAME fiat_crypto_vectors COMMAND test_fiat_crypto_vectors)
set_tests_properties(fiat_crypto_vectors PROPERTIES TIMEOUT 900)

# -- Carry propagation stress test -----------------------------------------
add_executable(test_carry_propagation test_carry_propagation.cpp)
audit_target_defaults(test_carry_propagation)
target_compile_definitions(test_carry_propagation PRIVATE STANDALONE_TEST)
add_test(NAME carry_propagation COMMAND test_carry_propagation)
set_tests_properties(carry_propagation PROPERTIES TIMEOUT 900)

# -- Wycheproof ECDSA secp256k1 vectors (Track I3-1) ----------------------
add_executable(test_wycheproof_ecdsa test_wycheproof_ecdsa.cpp)
audit_target_defaults(test_wycheproof_ecdsa)
target_compile_definitions(test_wycheproof_ecdsa PRIVATE STANDALONE_TEST)
add_test(NAME wycheproof_ecdsa COMMAND test_wycheproof_ecdsa)
set_tests_properties(wycheproof_ecdsa PROPERTIES TIMEOUT 1200)

# -- Wycheproof ECDH secp256k1 vectors (Track I3-2) -----------------------
add_executable(test_wycheproof_ecdh test_wycheproof_ecdh.cpp)
audit_target_defaults(test_wycheproof_ecdh)
target_compile_definitions(test_wycheproof_ecdh PRIVATE STANDALONE_TEST)
add_test(NAME wycheproof_ecdh COMMAND test_wycheproof_ecdh)
set_tests_properties(wycheproof_ecdh PROPERTIES TIMEOUT 1200)

# -- Batch verify randomness audit (Track I6-3) ---------------------------
add_executable(test_batch_randomness test_batch_randomness.cpp)
audit_target_defaults(test_batch_randomness)
target_compile_definitions(test_batch_randomness PRIVATE STANDALONE_TEST)
add_test(NAME batch_randomness COMMAND test_batch_randomness)
set_tests_properties(batch_randomness PROPERTIES TIMEOUT 600)

# -- Cross-platform KAT equivalence ---------------------------------------
add_executable(test_cross_platform_kat test_cross_platform_kat.cpp)
audit_target_defaults(test_cross_platform_kat)
target_compile_definitions(test_cross_platform_kat PRIVATE STANDALONE_TEST)
add_test(NAME cross_platform_kat COMMAND test_cross_platform_kat)
set_tests_properties(cross_platform_kat PROPERTIES TIMEOUT 900)

# -- ABI version gate (compile-time check) ---------------------------------
add_executable(test_abi_gate test_abi_gate.cpp)
target_include_directories(test_abi_gate PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)
target_compile_definitions(test_abi_gate PRIVATE STANDALONE_TEST)
add_test(NAME abi_gate COMMAND test_abi_gate)
set_tests_properties(abi_gate PROPERTIES TIMEOUT 30)

# -- GPU ABI gate (requires GPU host layer + ufsecp with GPU impl) ----------
if(TARGET secp256k1_gpu_host AND (TARGET ufsecp_shared OR TARGET ufsecp_static))
    # Prefer the already device-linked shared C ABI when present.
    # This avoids duplicate CUDA static archive/device-link ownership in the
    # standalone GPU audit executables while keeping a static fallback.
    if(TARGET ufsecp_shared)
        set(UFSECP_GPU_TEST_LIB ufsecp_shared)
    else()
        set(UFSECP_GPU_TEST_LIB ufsecp_static)
    endif()

    add_executable(test_gpu_abi_gate test_gpu_abi_gate.cpp)
    target_link_libraries(test_gpu_abi_gate PRIVATE ${UFSECP_GPU_TEST_LIB})
    target_include_directories(test_gpu_abi_gate PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
        ${CMAKE_BINARY_DIR}/include/ufsecp    # ufsecp_version.h
    )
    target_compile_definitions(test_gpu_abi_gate PRIVATE STANDALONE_TEST)
    add_test(NAME gpu_abi_gate COMMAND test_gpu_abi_gate)
    set_tests_properties(gpu_abi_gate PROPERTIES TIMEOUT 60)

    # -- GPU ops equivalence (GPU vs CPU reference) --
    add_executable(test_gpu_ops_equivalence test_gpu_ops_equivalence.cpp)
    target_link_libraries(test_gpu_ops_equivalence PRIVATE ${UFSECP_GPU_TEST_LIB})
    target_include_directories(test_gpu_ops_equivalence PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
        ${CMAKE_BINARY_DIR}/include/ufsecp
    )
    target_compile_definitions(test_gpu_ops_equivalence PRIVATE STANDALONE_TEST)
    add_test(NAME gpu_ops_equivalence COMMAND test_gpu_ops_equivalence)
    set_tests_properties(gpu_ops_equivalence PROPERTIES TIMEOUT 120)

    # -- GPU host API negative tests --
    add_executable(test_gpu_host_api_negative test_gpu_host_api_negative.cpp)
    target_link_libraries(test_gpu_host_api_negative PRIVATE ${UFSECP_GPU_TEST_LIB})
    target_include_directories(test_gpu_host_api_negative PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
        ${CMAKE_BINARY_DIR}/include/ufsecp
    )
    target_compile_definitions(test_gpu_host_api_negative PRIVATE STANDALONE_TEST)
    add_test(NAME gpu_host_api_negative COMMAND test_gpu_host_api_negative)
    set_tests_properties(gpu_host_api_negative PROPERTIES TIMEOUT 60)

    # -- GPU backend matrix (enumerate + probe) --
    add_executable(test_gpu_backend_matrix test_gpu_backend_matrix.cpp)
    target_link_libraries(test_gpu_backend_matrix PRIVATE ${UFSECP_GPU_TEST_LIB})
    target_include_directories(test_gpu_backend_matrix PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
        ${CMAKE_BINARY_DIR}/include/ufsecp
    )
    target_compile_definitions(test_gpu_backend_matrix PRIVATE STANDALONE_TEST)
    add_test(NAME gpu_backend_matrix COMMAND test_gpu_backend_matrix)
    set_tests_properties(gpu_backend_matrix PROPERTIES TIMEOUT 60)

    # Disable LTO for GPU-linked targets (nvcc LTO version differs from host compiler)
    if(SECP256K1_BUILD_CUDA)
        set_target_properties(
            test_gpu_abi_gate test_gpu_ops_equivalence
            test_gpu_host_api_negative test_gpu_backend_matrix
            PROPERTIES INTERPROCEDURAL_OPTIMIZATION FALSE
        )
    endif()
endif()

# -- SEC2 curve constant oracle -------------------------------------------
add_executable(test_secp256k1_spec_standalone test_secp256k1_spec.cpp)
audit_target_defaults(test_secp256k1_spec_standalone)
target_compile_definitions(test_secp256k1_spec_standalone PRIVATE STANDALONE_TEST)
add_test(NAME secp256k1_spec COMMAND test_secp256k1_spec_standalone)
set_tests_properties(secp256k1_spec PROPERTIES TIMEOUT 60)

# -- Post-operation invariant monitor -------------------------------------
add_executable(test_audit_invariants_standalone audit_invariants.cpp)
audit_target_defaults(test_audit_invariants_standalone)
target_compile_definitions(test_audit_invariants_standalone PRIVATE STANDALONE_TEST)
add_test(NAME audit_invariants COMMAND test_audit_invariants_standalone)
set_tests_properties(audit_invariants PROPERTIES TIMEOUT 300)

# -- C ABI negative contract tests ----------------------------------------
add_executable(test_c_abi_negative_standalone
    test_c_abi_negative.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/ufsecp/ufsecp_impl.cpp)
audit_target_defaults(test_c_abi_negative_standalone)
target_include_directories(test_c_abi_negative_standalone PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_BINARY_DIR}/include
)
target_compile_definitions(test_c_abi_negative_standalone PRIVATE UFSECP_BUILDING)
add_test(NAME c_abi_negative COMMAND test_c_abi_negative_standalone)
set_tests_properties(c_abi_negative PROPERTIES TIMEOUT 120)

# -- Secure memory erasure verification -----------------------------------
add_executable(test_audit_secure_erase_standalone
    audit_secure_erase.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/ufsecp/ufsecp_impl.cpp)
audit_target_defaults(test_audit_secure_erase_standalone)
target_include_directories(test_audit_secure_erase_standalone PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_BINARY_DIR}/include
)
target_compile_definitions(test_audit_secure_erase_standalone PRIVATE UFSECP_BUILDING)
add_test(NAME audit_secure_erase COMMAND test_audit_secure_erase_standalone)
set_tests_properties(audit_secure_erase PROPERTIES TIMEOUT 120)

# -- CT namespace discipline (source-level scan) ---------------------------
add_executable(test_audit_ct_namespace_standalone audit_ct_namespace.cpp)
audit_target_defaults(test_audit_ct_namespace_standalone)
add_test(NAME audit_ct_namespace COMMAND test_audit_ct_namespace_standalone)
set_tests_properties(audit_ct_namespace PROPERTIES TIMEOUT 60)

# -- KAT for all operations (ECDH/WIF/addresses/hash/arithmetic) ----------
add_executable(test_kat_all_operations_standalone
    test_kat_all_operations.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/ufsecp/ufsecp_impl.cpp)
audit_target_defaults(test_kat_all_operations_standalone)
target_include_directories(test_kat_all_operations_standalone PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_BINARY_DIR}/include
)
target_compile_definitions(test_kat_all_operations_standalone PRIVATE UFSECP_BUILDING)
add_test(NAME kat_all_operations COMMAND test_kat_all_operations_standalone)
set_tests_properties(kat_all_operations PROPERTIES TIMEOUT 120)

# -- Public parse path strictness audit -----------------------------------
add_executable(test_parse_strictness_standalone
    test_parse_strictness.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/ufsecp/ufsecp_impl.cpp)
audit_target_defaults(test_parse_strictness_standalone)
target_include_directories(test_parse_strictness_standalone PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_BINARY_DIR}/include
)
target_compile_definitions(test_parse_strictness_standalone PRIVATE UFSECP_BUILDING)
add_test(NAME parse_strictness COMMAND test_parse_strictness_standalone)
set_tests_properties(parse_strictness PROPERTIES TIMEOUT 120)

# -- RFC 6979 nonce determinism + uniqueness monitor ----------------------
add_executable(test_nonce_uniqueness_standalone
    test_nonce_uniqueness.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/ufsecp/ufsecp_impl.cpp)
audit_target_defaults(test_nonce_uniqueness_standalone)
target_include_directories(test_nonce_uniqueness_standalone PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_BINARY_DIR}/include
)
target_compile_definitions(test_nonce_uniqueness_standalone PRIVATE UFSECP_BUILDING)
add_test(NAME nonce_uniqueness COMMAND test_nonce_uniqueness_standalone)
set_tests_properties(nonce_uniqueness PROPERTIES TIMEOUT 120)

# -- C ABI thread stress audit ---------------------------------------------
add_executable(test_c_abi_thread_stress_standalone
    test_c_abi_thread_stress.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/ufsecp/ufsecp_impl.cpp)
audit_target_defaults(test_c_abi_thread_stress_standalone)
target_link_libraries(test_c_abi_thread_stress_standalone PRIVATE Threads::Threads)
target_include_directories(test_c_abi_thread_stress_standalone PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_BINARY_DIR}/include
)
target_compile_definitions(test_c_abi_thread_stress_standalone PRIVATE UFSECP_BUILDING)
add_test(NAME c_abi_thread_stress COMMAND test_c_abi_thread_stress_standalone)
set_tests_properties(c_abi_thread_stress PROPERTIES TIMEOUT 300)

# -- Formal CT verification via Valgrind/MSAN (Track I5-1) ----------------
add_executable(test_ct_verif_formal_standalone test_ct_verif_formal.cpp)
audit_target_defaults(test_ct_verif_formal_standalone)
target_compile_definitions(test_ct_verif_formal_standalone PRIVATE STANDALONE_TEST)
add_test(NAME ct_verif_formal COMMAND test_ct_verif_formal_standalone)
set_tests_properties(ct_verif_formal PROPERTIES TIMEOUT 1200)

# -- Independent reference linkage cross-check (Track I5-2) ----------------
add_executable(test_fiat_crypto_linkage_standalone test_fiat_crypto_linkage.cpp)
audit_target_defaults(test_fiat_crypto_linkage_standalone)
target_compile_definitions(test_fiat_crypto_linkage_standalone PRIVATE STANDALONE_TEST)
add_test(NAME fiat_crypto_linkage COMMAND test_fiat_crypto_linkage_standalone)
set_tests_properties(fiat_crypto_linkage PROPERTIES TIMEOUT 900)

# -- Standalone audit_fuzz test --------------------------------------------
add_executable(test_audit_fuzz_standalone audit_fuzz.cpp)
audit_target_defaults(test_audit_fuzz_standalone)
add_test(NAME audit_fuzz COMMAND test_audit_fuzz_standalone)
set_tests_properties(audit_fuzz PROPERTIES TIMEOUT 600)

# -- Adversarial protocol & FFI hostile-caller tests -----------------------
add_executable(test_adversarial_protocol_standalone test_adversarial_protocol.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/ufsecp/ufsecp_impl.cpp)
audit_target_defaults(test_adversarial_protocol_standalone)
target_include_directories(test_adversarial_protocol_standalone PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_BINARY_DIR}/include
)
target_compile_definitions(test_adversarial_protocol_standalone PRIVATE UFSECP_BUILDING)
add_test(NAME adversarial_protocol COMMAND test_adversarial_protocol_standalone)
set_tests_properties(adversarial_protocol PROPERTIES TIMEOUT 600)

# ===========================================================================
# Security Exploit PoC Tests (regression gates for discovered vulnerabilities)
# ===========================================================================
# Each test FAILS on the buggy code and PASSES after the fix is applied.
# They serve as living documentation of vulnerabilities found in audit round 5.

# -- PoC: CT violation in ct::ecdsa_sign_recoverable (secret-dependent branches)
add_executable(test_exploit_ct_recov_standalone test_exploit_ct_recov.cpp)
audit_target_defaults(test_exploit_ct_recov_standalone)
target_compile_definitions(test_exploit_ct_recov_standalone PRIVATE
    STANDALONE_TEST DUDECT_SMOKE)
add_test(NAME exploit_ct_recov COMMAND test_exploit_ct_recov_standalone)
set_tests_properties(exploit_ct_recov PROPERTIES TIMEOUT 300 LABELS "audit;exploit;ct")

# -- PoC: BIP-32 depth uint8_t silent overflow (255 -> 0 wrap)
add_executable(test_exploit_bip32_depth_standalone test_exploit_bip32_depth.cpp)
audit_target_defaults(test_exploit_bip32_depth_standalone)
target_compile_definitions(test_exploit_bip32_depth_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_bip32_depth COMMAND test_exploit_bip32_depth_standalone)
set_tests_properties(exploit_bip32_depth PROPERTIES TIMEOUT 60 LABELS "audit;exploit;bip32")

# -- PoC: MuSig2 key aggregation order dependence (ordering attack)
add_executable(test_exploit_musig2_ordering_standalone test_exploit_musig2_ordering.cpp)
audit_target_defaults(test_exploit_musig2_ordering_standalone)
target_compile_definitions(test_exploit_musig2_ordering_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_musig2_ordering COMMAND test_exploit_musig2_ordering_standalone)
set_tests_properties(exploit_musig2_ordering PROPERTIES
    TIMEOUT 120 LABELS "audit;exploit;musig2")

# -- PoC: FROST polynomial index collision (participant_id * 1000 + j formula)
add_executable(test_exploit_frost_index_standalone test_exploit_frost_index.cpp)
audit_target_defaults(test_exploit_frost_index_standalone)
target_compile_definitions(test_exploit_frost_index_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_frost_index COMMAND test_exploit_frost_index_standalone)
set_tests_properties(exploit_frost_index PROPERTIES TIMEOUT 60 LABELS "audit;exploit;frost")

# -- PoC: FROST participant_id=0 secret exposure (x=0 is the secret evaluation point)
add_executable(test_exploit_frost_participant_zero_standalone test_exploit_frost_participant_zero.cpp)
audit_target_defaults(test_exploit_frost_participant_zero_standalone)
target_compile_definitions(test_exploit_frost_participant_zero_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_frost_participant_zero COMMAND test_exploit_frost_participant_zero_standalone)
set_tests_properties(exploit_frost_participant_zero PROPERTIES TIMEOUT 60 LABELS "audit;exploit;frost")

# -- PoC: FROST threshold=0 degenerate case (empty polynomial, all-zero shares)
add_executable(test_exploit_frost_threshold_degenerate_standalone test_exploit_frost_threshold_degenerate.cpp)
audit_target_defaults(test_exploit_frost_threshold_degenerate_standalone)
target_compile_definitions(test_exploit_frost_threshold_degenerate_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_frost_threshold_degenerate COMMAND test_exploit_frost_threshold_degenerate_standalone)
set_tests_properties(exploit_frost_threshold_degenerate PROPERTIES TIMEOUT 60 LABELS "audit;exploit;frost")

# -- PoC: FROST Lagrange coefficient corruption via duplicate signer IDs
add_executable(test_exploit_frost_lagrange_duplicate_standalone test_exploit_frost_lagrange_duplicate.cpp)
audit_target_defaults(test_exploit_frost_lagrange_duplicate_standalone)
target_compile_definitions(test_exploit_frost_lagrange_duplicate_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_frost_lagrange_duplicate COMMAND test_exploit_frost_lagrange_duplicate_standalone)
set_tests_properties(exploit_frost_lagrange_duplicate PROPERTIES TIMEOUT 60 LABELS "audit;exploit;frost")

# -- PoC: Schnorr adaptor signature R.y parity not verified (DoS on atomic swaps)
add_executable(test_exploit_adaptor_parity_standalone test_exploit_adaptor_parity.cpp)
audit_target_defaults(test_exploit_adaptor_parity_standalone)
target_compile_definitions(test_exploit_adaptor_parity_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_adaptor_parity COMMAND test_exploit_adaptor_parity_standalone)
set_tests_properties(exploit_adaptor_parity PROPERTIES TIMEOUT 60 LABELS "audit;exploit;adaptor")

# -- PoC: Scalar arithmetic edge cases and invariant violations
add_executable(test_exploit_scalar_invariants_standalone test_exploit_scalar_invariants.cpp)
audit_target_defaults(test_exploit_scalar_invariants_standalone)
target_compile_definitions(test_exploit_scalar_invariants_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_scalar_invariants COMMAND test_exploit_scalar_invariants_standalone)
set_tests_properties(exploit_scalar_invariants PROPERTIES TIMEOUT 60 LABELS "audit;exploit;scalar")

# -- PoC: ECDH degenerate input handling (zero key, infinity point, protocol confusion)
add_executable(test_exploit_ecdh_degenerate_standalone test_exploit_ecdh_degenerate.cpp)
audit_target_defaults(test_exploit_ecdh_degenerate_standalone)
target_compile_definitions(test_exploit_ecdh_degenerate_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_ecdh_degenerate COMMAND test_exploit_ecdh_degenerate_standalone)
set_tests_properties(exploit_ecdh_degenerate PROPERTIES TIMEOUT 60 LABELS "audit;exploit;ecdh")

# -- PoC: Schnorr signature BIP-340 edge cases (s=0, r>=p, zero msg, determinism)
add_executable(test_exploit_schnorr_edge_cases_standalone test_exploit_schnorr_edge_cases.cpp)
audit_target_defaults(test_exploit_schnorr_edge_cases_standalone)
target_compile_definitions(test_exploit_schnorr_edge_cases_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_schnorr_edge_cases COMMAND test_exploit_schnorr_edge_cases_standalone)
set_tests_properties(exploit_schnorr_edge_cases PROPERTIES TIMEOUT 60 LABELS "audit;exploit;schnorr")

# -- PoC: ECIES authentication bypass and ciphertext tampering tests
add_executable(test_exploit_ecies_auth_standalone test_exploit_ecies_auth.cpp)
audit_target_defaults(test_exploit_ecies_auth_standalone)
target_compile_definitions(test_exploit_ecies_auth_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_ecies_auth COMMAND test_exploit_ecies_auth_standalone)
set_tests_properties(exploit_ecies_auth PROPERTIES TIMEOUT 120 LABELS "audit;exploit;ecies")

# -- PoC: Schnorr batch verification soundness (invalid sig detection, a_0=1 weight)
add_executable(test_exploit_batch_soundness_standalone test_exploit_batch_soundness.cpp)
audit_target_defaults(test_exploit_batch_soundness_standalone)
target_compile_definitions(test_exploit_batch_soundness_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_batch_soundness COMMAND test_exploit_batch_soundness_standalone)
set_tests_properties(exploit_batch_soundness PROPERTIES TIMEOUT 120 LABELS "audit;exploit;batch")

# -- PoC: BIP-32 path parser overflow and boundary conditions
add_executable(test_exploit_bip32_path_overflow_standalone test_exploit_bip32_path_overflow.cpp)
audit_target_defaults(test_exploit_bip32_path_overflow_standalone)
target_compile_definitions(test_exploit_bip32_path_overflow_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_bip32_path_overflow COMMAND test_exploit_bip32_path_overflow_standalone)
set_tests_properties(exploit_bip32_path_overflow PROPERTIES TIMEOUT 60 LABELS "audit;exploit;bip32")

# -- PoC: MuSig2 nonce reuse attack (private key recovery, single-use enforcement)
add_executable(test_exploit_musig2_nonce_reuse_standalone test_exploit_musig2_nonce_reuse.cpp)
audit_target_defaults(test_exploit_musig2_nonce_reuse_standalone)
target_compile_definitions(test_exploit_musig2_nonce_reuse_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_musig2_nonce_reuse COMMAND test_exploit_musig2_nonce_reuse_standalone)
set_tests_properties(exploit_musig2_nonce_reuse PROPERTIES TIMEOUT 120 LABELS "audit;exploit;musig2")

# -- PoC: MuSig2 transcript fork / coordinator equivocation
add_executable(test_exploit_musig2_transcript_fork_standalone test_exploit_musig2_transcript_fork.cpp)
audit_target_defaults(test_exploit_musig2_transcript_fork_standalone)
target_compile_definitions(test_exploit_musig2_transcript_fork_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_musig2_transcript_fork COMMAND test_exploit_musig2_transcript_fork_standalone)
set_tests_properties(exploit_musig2_transcript_fork PROPERTIES TIMEOUT 120 LABELS "audit;exploit;musig2")

# -- PoC: Elliptic curve point group law correctness (commutativity, associativity, etc.)
add_executable(test_exploit_point_group_law_standalone test_exploit_point_group_law.cpp)
audit_target_defaults(test_exploit_point_group_law_standalone)
target_compile_definitions(test_exploit_point_group_law_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_point_group_law COMMAND test_exploit_point_group_law_standalone)
set_tests_properties(exploit_point_group_law PROPERTIES TIMEOUT 60 LABELS "audit;exploit;ecc")

# -- PoC: ECDSA key recovery edge cases
add_executable(test_exploit_ecdsa_recovery_standalone test_exploit_ecdsa_recovery.cpp)
audit_target_defaults(test_exploit_ecdsa_recovery_standalone)
target_compile_definitions(test_exploit_ecdsa_recovery_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_ecdsa_recovery COMMAND test_exploit_ecdsa_recovery_standalone)
set_tests_properties(exploit_ecdsa_recovery PROPERTIES TIMEOUT 60 LABELS "audit;exploit;ecdsa")

# -- PoC: ECDSA edge cases and normalization
add_executable(test_exploit_ecdsa_edge_cases_standalone test_exploit_ecdsa_edge_cases.cpp)
audit_target_defaults(test_exploit_ecdsa_edge_cases_standalone)
target_compile_definitions(test_exploit_ecdsa_edge_cases_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_ecdsa_edge_cases COMMAND test_exploit_ecdsa_edge_cases_standalone)
set_tests_properties(exploit_ecdsa_edge_cases PROPERTIES TIMEOUT 60 LABELS "audit;exploit;ecdsa")

# -- PoC: ECDSA DER canonicality / alternate-encoding confusion
add_executable(test_exploit_ecdsa_der_confusion_standalone
    test_exploit_ecdsa_der_confusion.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/ufsecp/ufsecp_impl.cpp)
audit_target_defaults(test_exploit_ecdsa_der_confusion_standalone)
target_include_directories(test_exploit_ecdsa_der_confusion_standalone PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_BINARY_DIR}/include
)
target_compile_definitions(test_exploit_ecdsa_der_confusion_standalone PRIVATE
    UFSECP_BUILDING
    STANDALONE_TEST
)
add_test(NAME exploit_ecdsa_der_confusion COMMAND test_exploit_ecdsa_der_confusion_standalone)
set_tests_properties(exploit_ecdsa_der_confusion PROPERTIES TIMEOUT 60 LABELS "audit;exploit;ecdsa")

# -- PoC: Taproot key tweak and commitment security
add_executable(test_exploit_taproot_tweak_standalone test_exploit_taproot_tweak.cpp)
audit_target_defaults(test_exploit_taproot_tweak_standalone)
target_compile_definitions(test_exploit_taproot_tweak_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_taproot_tweak COMMAND test_exploit_taproot_tweak_standalone)
set_tests_properties(exploit_taproot_tweak PROPERTIES TIMEOUT 60 LABELS "audit;exploit;taproot")

# -- PoC: ChaCha20-Poly1305 nonce reuse and AEAD properties
add_executable(test_exploit_chacha20_nonce_reuse_standalone test_exploit_chacha20_nonce_reuse.cpp)
audit_target_defaults(test_exploit_chacha20_nonce_reuse_standalone)
target_compile_definitions(test_exploit_chacha20_nonce_reuse_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_chacha20_nonce_reuse COMMAND test_exploit_chacha20_nonce_reuse_standalone)
set_tests_properties(exploit_chacha20_nonce_reuse PROPERTIES TIMEOUT 60 LABELS "audit;exploit;aead")

# -- PoC: ZK proof soundness (knowledge, DLEQ, range proofs)
add_executable(test_exploit_zk_proofs_standalone test_exploit_zk_proofs.cpp)
audit_target_defaults(test_exploit_zk_proofs_standalone)
target_compile_definitions(test_exploit_zk_proofs_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_zk_proofs COMMAND test_exploit_zk_proofs_standalone)
set_tests_properties(exploit_zk_proofs PROPERTIES TIMEOUT 120 LABELS "audit;exploit;zk")

# -- PoC: FROST threshold signing end-to-end (DKG + sign + verify)
add_executable(test_exploit_frost_signing_standalone test_exploit_frost_signing.cpp)
audit_target_defaults(test_exploit_frost_signing_standalone)
target_compile_definitions(test_exploit_frost_signing_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_frost_signing COMMAND test_exploit_frost_signing_standalone)
set_tests_properties(exploit_frost_signing PROPERTIES TIMEOUT 120 LABELS "audit;exploit;frost")

# -- PoC: FROST stale commitment replay / subset fork
add_executable(test_exploit_frost_commitment_reuse_standalone test_exploit_frost_commitment_reuse.cpp)
audit_target_defaults(test_exploit_frost_commitment_reuse_standalone)
target_compile_definitions(test_exploit_frost_commitment_reuse_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_frost_commitment_reuse COMMAND test_exploit_frost_commitment_reuse_standalone)
set_tests_properties(exploit_frost_commitment_reuse PROPERTIES TIMEOUT 120 LABELS "audit;exploit;frost")

# -- PoC: BIP-39 mnemonic entropy and checksum security
add_executable(test_exploit_bip39_entropy_standalone test_exploit_bip39_entropy.cpp)
audit_target_defaults(test_exploit_bip39_entropy_standalone)
target_compile_definitions(test_exploit_bip39_entropy_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_bip39_entropy COMMAND test_exploit_bip39_entropy_standalone)
set_tests_properties(exploit_bip39_entropy PROPERTIES TIMEOUT 60 LABELS "audit;exploit;bip39")

# -- PoC: ElligatorSwift (BIP-324) ECDH security
add_executable(test_exploit_ellswift_standalone test_exploit_ellswift.cpp)
audit_target_defaults(test_exploit_ellswift_standalone)
target_compile_definitions(test_exploit_ellswift_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_ellswift COMMAND test_exploit_ellswift_standalone)
set_tests_properties(exploit_ellswift PROPERTIES TIMEOUT 60 LABELS "audit;exploit;bip324")

add_executable(test_exploit_address_generation_standalone test_exploit_address_generation.cpp)
audit_target_defaults(test_exploit_address_generation_standalone)
target_compile_definitions(test_exploit_address_generation_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_address_generation COMMAND test_exploit_address_generation_standalone)
set_tests_properties(exploit_address_generation PROPERTIES TIMEOUT 60 LABELS "audit;exploit;address")

add_executable(test_exploit_bip324_session_standalone test_exploit_bip324_session.cpp)
audit_target_defaults(test_exploit_bip324_session_standalone)
target_compile_definitions(test_exploit_bip324_session_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_bip324_session COMMAND test_exploit_bip324_session_standalone)
set_tests_properties(exploit_bip324_session PROPERTIES TIMEOUT 60 LABELS "audit;exploit;bip324")

add_executable(test_exploit_bip324_transcript_splice_standalone test_exploit_bip324_transcript_splice.cpp)
audit_target_defaults(test_exploit_bip324_transcript_splice_standalone)
target_compile_definitions(test_exploit_bip324_transcript_splice_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_bip324_transcript_splice COMMAND test_exploit_bip324_transcript_splice_standalone)
set_tests_properties(exploit_bip324_transcript_splice PROPERTIES TIMEOUT 60 LABELS "audit;exploit;bip324")

add_executable(test_exploit_bip324_counter_desync_standalone test_exploit_bip324_counter_desync.cpp)
audit_target_defaults(test_exploit_bip324_counter_desync_standalone)
target_compile_definitions(test_exploit_bip324_counter_desync_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_bip324_counter_desync COMMAND test_exploit_bip324_counter_desync_standalone)
set_tests_properties(exploit_bip324_counter_desync PROPERTIES TIMEOUT 60 LABELS "audit;exploit;bip324")

add_executable(test_exploit_multiscalar_standalone test_exploit_multiscalar.cpp)
audit_target_defaults(test_exploit_multiscalar_standalone)
target_compile_definitions(test_exploit_multiscalar_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_multiscalar COMMAND test_exploit_multiscalar_standalone)
set_tests_properties(exploit_multiscalar PROPERTIES TIMEOUT 60 LABELS "audit;exploit;msm")

add_executable(test_exploit_pedersen_homomorphism_standalone test_exploit_pedersen_homomorphism.cpp)
audit_target_defaults(test_exploit_pedersen_homomorphism_standalone)
target_compile_definitions(test_exploit_pedersen_homomorphism_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_pedersen_homomorphism COMMAND test_exploit_pedersen_homomorphism_standalone)
set_tests_properties(exploit_pedersen_homomorphism PROPERTIES TIMEOUT 60 LABELS "audit;exploit;pedersen")

add_executable(test_exploit_hkdf_security_standalone test_exploit_hkdf_security.cpp)
audit_target_defaults(test_exploit_hkdf_security_standalone)
target_compile_definitions(test_exploit_hkdf_security_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_hkdf_security COMMAND test_exploit_hkdf_security_standalone)
set_tests_properties(exploit_hkdf_security PROPERTIES TIMEOUT 60 LABELS "audit;exploit;hkdf")

add_executable(test_exploit_batch_schnorr_standalone test_exploit_batch_schnorr.cpp)
audit_target_defaults(test_exploit_batch_schnorr_standalone)
target_compile_definitions(test_exploit_batch_schnorr_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_batch_schnorr COMMAND test_exploit_batch_schnorr_standalone)
set_tests_properties(exploit_batch_schnorr PROPERTIES TIMEOUT 60 LABELS "audit;exploit;schnorr")

add_executable(test_exploit_segwit_encoding_standalone test_exploit_segwit_encoding.cpp)
audit_target_defaults(test_exploit_segwit_encoding_standalone)
target_compile_definitions(test_exploit_segwit_encoding_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_segwit_encoding COMMAND test_exploit_segwit_encoding_standalone)
set_tests_properties(exploit_segwit_encoding PROPERTIES TIMEOUT 60 LABELS "audit;exploit;segwit")

add_executable(test_exploit_private_key_standalone test_exploit_private_key.cpp)
audit_target_defaults(test_exploit_private_key_standalone)
target_compile_definitions(test_exploit_private_key_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_private_key COMMAND test_exploit_private_key_standalone)
set_tests_properties(exploit_private_key PROPERTIES TIMEOUT 60 LABELS "audit;exploit;privkey")

add_executable(test_exploit_bip143_sighash_standalone test_exploit_bip143_sighash.cpp)
audit_target_defaults(test_exploit_bip143_sighash_standalone)
target_compile_definitions(test_exploit_bip143_sighash_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_bip143_sighash COMMAND test_exploit_bip143_sighash_standalone)
set_tests_properties(exploit_bip143_sighash PROPERTIES TIMEOUT 60 LABELS "audit;exploit;bip143")

add_executable(test_exploit_ecies_roundtrip_standalone test_exploit_ecies_roundtrip.cpp)
audit_target_defaults(test_exploit_ecies_roundtrip_standalone)
target_compile_definitions(test_exploit_ecies_roundtrip_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_ecies_roundtrip COMMAND test_exploit_ecies_roundtrip_standalone)
set_tests_properties(exploit_ecies_roundtrip PROPERTIES TIMEOUT 60 LABELS "audit;exploit;ecies")

add_executable(test_exploit_ecdh_variants_standalone test_exploit_ecdh_variants.cpp)
audit_target_defaults(test_exploit_ecdh_variants_standalone)
target_compile_definitions(test_exploit_ecdh_variants_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_ecdh_variants COMMAND test_exploit_ecdh_variants_standalone)
set_tests_properties(exploit_ecdh_variants PROPERTIES TIMEOUT 60 LABELS "audit;exploit;ecdh")

add_executable(test_exploit_glv_endomorphism_standalone test_exploit_glv_endomorphism.cpp)
audit_target_defaults(test_exploit_glv_endomorphism_standalone)
target_compile_definitions(test_exploit_glv_endomorphism_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_glv_endomorphism COMMAND test_exploit_glv_endomorphism_standalone)
set_tests_properties(exploit_glv_endomorphism PROPERTIES TIMEOUT 60 LABELS "audit;exploit;glv")

add_executable(test_exploit_sha_kat_standalone test_exploit_sha_kat.cpp)
audit_target_defaults(test_exploit_sha_kat_standalone)
target_compile_definitions(test_exploit_sha_kat_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_sha_kat COMMAND test_exploit_sha_kat_standalone)
set_tests_properties(exploit_sha_kat PROPERTIES TIMEOUT 60 LABELS "audit;exploit;hash")

add_executable(test_exploit_bip144_serialization_standalone test_exploit_bip144_serialization.cpp)
audit_target_defaults(test_exploit_bip144_serialization_standalone)
target_compile_definitions(test_exploit_bip144_serialization_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_bip144_serialization COMMAND test_exploit_bip144_serialization_standalone)
set_tests_properties(exploit_bip144_serialization PROPERTIES TIMEOUT 60 LABELS "audit;exploit;bip144")

add_executable(test_exploit_musig2_key_agg_standalone test_exploit_musig2_key_agg.cpp)
audit_target_defaults(test_exploit_musig2_key_agg_standalone)
target_compile_definitions(test_exploit_musig2_key_agg_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_musig2_key_agg COMMAND test_exploit_musig2_key_agg_standalone)
set_tests_properties(exploit_musig2_key_agg PROPERTIES TIMEOUT 60 LABELS "audit;exploit;musig2")

add_executable(test_exploit_bip32_derivation_standalone test_exploit_bip32_derivation.cpp)
audit_target_defaults(test_exploit_bip32_derivation_standalone)
target_compile_definitions(test_exploit_bip32_derivation_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_bip32_derivation COMMAND test_exploit_bip32_derivation_standalone)
set_tests_properties(exploit_bip32_derivation PROPERTIES TIMEOUT 60 LABELS "audit;exploit;bip32")

add_executable(test_exploit_adaptor_extended_standalone test_exploit_adaptor_extended.cpp)
audit_target_defaults(test_exploit_adaptor_extended_standalone)
target_compile_definitions(test_exploit_adaptor_extended_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_adaptor_extended COMMAND test_exploit_adaptor_extended_standalone)
set_tests_properties(exploit_adaptor_extended PROPERTIES TIMEOUT 60 LABELS "audit;exploit;adaptor")

add_executable(test_exploit_chacha20_poly1305_standalone test_exploit_chacha20_poly1305.cpp)
audit_target_defaults(test_exploit_chacha20_poly1305_standalone)
target_compile_definitions(test_exploit_chacha20_poly1305_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_chacha20_poly1305 COMMAND test_exploit_chacha20_poly1305_standalone)
set_tests_properties(exploit_chacha20_poly1305 PROPERTIES TIMEOUT 60 LABELS "audit;exploit;aead")

add_executable(test_exploit_field_arithmetic_standalone test_exploit_field_arithmetic.cpp)
audit_target_defaults(test_exploit_field_arithmetic_standalone)
target_compile_definitions(test_exploit_field_arithmetic_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_field_arithmetic COMMAND test_exploit_field_arithmetic_standalone)
set_tests_properties(exploit_field_arithmetic PROPERTIES TIMEOUT 60 LABELS "audit;exploit;field")

add_executable(test_exploit_selftest_api_standalone test_exploit_selftest_api.cpp)
audit_target_defaults(test_exploit_selftest_api_standalone)
target_compile_definitions(test_exploit_selftest_api_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_selftest_api COMMAND test_exploit_selftest_api_standalone)
set_tests_properties(exploit_selftest_api PROPERTIES TIMEOUT 120 LABELS "audit;exploit;selftest")

add_executable(test_exploit_scalar_group_order_standalone test_exploit_scalar_group_order.cpp)
audit_target_defaults(test_exploit_scalar_group_order_standalone)
target_compile_definitions(test_exploit_scalar_group_order_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_scalar_group_order COMMAND test_exploit_scalar_group_order_standalone)
set_tests_properties(exploit_scalar_group_order PROPERTIES TIMEOUT 60 LABELS "audit;exploit;scalar")

add_executable(test_exploit_pippenger_msm_standalone test_exploit_pippenger_msm.cpp)
audit_target_defaults(test_exploit_pippenger_msm_standalone)
target_compile_definitions(test_exploit_pippenger_msm_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_pippenger_msm COMMAND test_exploit_pippenger_msm_standalone)
set_tests_properties(exploit_pippenger_msm PROPERTIES TIMEOUT 120 LABELS "audit;exploit;msm")

add_executable(test_exploit_recovery_extended_standalone test_exploit_recovery_extended.cpp)
audit_target_defaults(test_exploit_recovery_extended_standalone)
target_compile_definitions(test_exploit_recovery_extended_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_recovery_extended COMMAND test_exploit_recovery_extended_standalone)
set_tests_properties(exploit_recovery_extended PROPERTIES TIMEOUT 60 LABELS "audit;exploit;ecdsa")

add_executable(test_exploit_batch_verify_correctness_standalone test_exploit_batch_verify_correctness.cpp)
audit_target_defaults(test_exploit_batch_verify_correctness_standalone)
target_compile_definitions(test_exploit_batch_verify_correctness_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_batch_verify_correctness COMMAND test_exploit_batch_verify_correctness_standalone)
set_tests_properties(exploit_batch_verify_correctness PROPERTIES TIMEOUT 120 LABELS "audit;exploit;batch")

add_executable(test_exploit_eth_signing_standalone test_exploit_eth_signing.cpp)
audit_target_defaults(test_exploit_eth_signing_standalone)
target_compile_definitions(test_exploit_eth_signing_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_eth_signing COMMAND test_exploit_eth_signing_standalone)
set_tests_properties(exploit_eth_signing PROPERTIES TIMEOUT 60 LABELS "audit;exploit;ethereum")

add_executable(test_exploit_bitcoin_message_signing_standalone test_exploit_bitcoin_message_signing.cpp)
audit_target_defaults(test_exploit_bitcoin_message_signing_standalone)
target_compile_definitions(test_exploit_bitcoin_message_signing_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_bitcoin_message_signing COMMAND test_exploit_bitcoin_message_signing_standalone)
set_tests_properties(exploit_bitcoin_message_signing PROPERTIES TIMEOUT 60 LABELS "audit;exploit;bitcoin")

add_executable(test_exploit_coin_hd_derivation_standalone test_exploit_coin_hd_derivation.cpp)
audit_target_defaults(test_exploit_coin_hd_derivation_standalone)
target_compile_definitions(test_exploit_coin_hd_derivation_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_coin_hd_derivation COMMAND test_exploit_coin_hd_derivation_standalone)
set_tests_properties(exploit_coin_hd_derivation PROPERTIES TIMEOUT 60 LABELS "audit;exploit;bip44")

add_executable(test_exploit_sha512_kat_standalone test_exploit_sha512_kat.cpp)
audit_target_defaults(test_exploit_sha512_kat_standalone)
target_compile_definitions(test_exploit_sha512_kat_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_sha512_kat COMMAND test_exploit_sha512_kat_standalone)
set_tests_properties(exploit_sha512_kat PROPERTIES TIMEOUT 60 LABELS "audit;exploit;hash")

add_executable(test_exploit_sha256_kat_standalone test_exploit_sha256_kat.cpp)
audit_target_defaults(test_exploit_sha256_kat_standalone)
target_compile_definitions(test_exploit_sha256_kat_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_sha256_kat COMMAND test_exploit_sha256_kat_standalone)
set_tests_properties(exploit_sha256_kat PROPERTIES TIMEOUT 60 LABELS "audit;exploit;hash")

add_executable(test_exploit_keccak256_kat_standalone test_exploit_keccak256_kat.cpp)
audit_target_defaults(test_exploit_keccak256_kat_standalone)
target_compile_definitions(test_exploit_keccak256_kat_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_keccak256_kat COMMAND test_exploit_keccak256_kat_standalone)
set_tests_properties(exploit_keccak256_kat PROPERTIES TIMEOUT 60 LABELS "audit;exploit;hash")

add_executable(test_exploit_ecdh_standalone test_exploit_ecdh.cpp)
audit_target_defaults(test_exploit_ecdh_standalone)
target_compile_definitions(test_exploit_ecdh_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_ecdh COMMAND test_exploit_ecdh_standalone)
set_tests_properties(exploit_ecdh PROPERTIES TIMEOUT 60 LABELS "audit;exploit;ecdh")

add_executable(test_exploit_bip39_mnemonic_standalone test_exploit_bip39_mnemonic.cpp)
audit_target_defaults(test_exploit_bip39_mnemonic_standalone)
target_compile_definitions(test_exploit_bip39_mnemonic_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_bip39_mnemonic COMMAND test_exploit_bip39_mnemonic_standalone)
set_tests_properties(exploit_bip39_mnemonic PROPERTIES TIMEOUT 120 LABELS "audit;exploit;bip39")

add_executable(test_exploit_musig2_standalone test_exploit_musig2.cpp)
audit_target_defaults(test_exploit_musig2_standalone)
target_compile_definitions(test_exploit_musig2_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_musig2 COMMAND test_exploit_musig2_standalone)
set_tests_properties(exploit_musig2 PROPERTIES TIMEOUT 120 LABELS "audit;exploit;musig2")

add_executable(test_exploit_ecies_encryption_standalone test_exploit_ecies_encryption.cpp)
audit_target_defaults(test_exploit_ecies_encryption_standalone)
target_compile_definitions(test_exploit_ecies_encryption_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_ecies_encryption COMMAND test_exploit_ecies_encryption_standalone)
set_tests_properties(exploit_ecies_encryption PROPERTIES TIMEOUT 60 LABELS "audit;exploit;ecies")

add_executable(test_exploit_wallet_api_standalone test_exploit_wallet_api.cpp)
audit_target_defaults(test_exploit_wallet_api_standalone)
target_compile_definitions(test_exploit_wallet_api_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_wallet_api COMMAND test_exploit_wallet_api_standalone)
set_tests_properties(exploit_wallet_api PROPERTIES TIMEOUT 60 LABELS "audit;exploit;wallet")

add_executable(test_exploit_frost_dkg_standalone test_exploit_frost_dkg.cpp)
audit_target_defaults(test_exploit_frost_dkg_standalone)
target_compile_definitions(test_exploit_frost_dkg_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_frost_dkg COMMAND test_exploit_frost_dkg_standalone)
set_tests_properties(exploit_frost_dkg PROPERTIES TIMEOUT 120 LABELS "audit;exploit;frost")

add_executable(test_exploit_taproot_scripts_standalone test_exploit_taproot_scripts.cpp)
audit_target_defaults(test_exploit_taproot_scripts_standalone)
target_compile_definitions(test_exploit_taproot_scripts_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_taproot_scripts COMMAND test_exploit_taproot_scripts_standalone)
set_tests_properties(exploit_taproot_scripts PROPERTIES TIMEOUT 60 LABELS "audit;exploit;taproot")

add_executable(test_exploit_hkdf_kat_standalone test_exploit_hkdf_kat.cpp)
audit_target_defaults(test_exploit_hkdf_kat_standalone)
target_compile_definitions(test_exploit_hkdf_kat_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_hkdf_kat COMMAND test_exploit_hkdf_kat_standalone)
set_tests_properties(exploit_hkdf_kat PROPERTIES TIMEOUT 60 LABELS "audit;exploit;hash;hkdf")

add_executable(test_exploit_ellswift_ecdh_standalone test_exploit_ellswift_ecdh.cpp)
audit_target_defaults(test_exploit_ellswift_ecdh_standalone)
target_compile_definitions(test_exploit_ellswift_ecdh_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_ellswift_ecdh COMMAND test_exploit_ellswift_ecdh_standalone)
set_tests_properties(exploit_ellswift_ecdh PROPERTIES TIMEOUT 60 LABELS "audit;exploit;bip324;ellswift")

add_executable(test_exploit_address_encoding_standalone test_exploit_address_encoding.cpp)
audit_target_defaults(test_exploit_address_encoding_standalone)
target_compile_definitions(test_exploit_address_encoding_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_address_encoding COMMAND test_exploit_address_encoding_standalone)
set_tests_properties(exploit_address_encoding PROPERTIES TIMEOUT 60 LABELS "audit;exploit;address;bip352")

# -- CT vs FAST layer output divergence (fills audit gap from 2026-03-24) ---
add_executable(test_exploit_ct_systematic_standalone test_exploit_ct_systematic.cpp)
audit_target_defaults(test_exploit_ct_systematic_standalone)
target_compile_definitions(test_exploit_ct_systematic_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_ct_systematic COMMAND test_exploit_ct_systematic_standalone)
set_tests_properties(exploit_ct_systematic PROPERTIES TIMEOUT 60 LABELS "audit;exploit;ct")

# -- Backend divergence: recoverable sig, tamper detection, batch mixing ----
add_executable(test_exploit_backend_divergence_standalone test_exploit_backend_divergence.cpp)
audit_target_defaults(test_exploit_backend_divergence_standalone)
target_compile_definitions(test_exploit_backend_divergence_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_backend_divergence COMMAND test_exploit_backend_divergence_standalone)
set_tests_properties(exploit_backend_divergence PROPERTIES TIMEOUT 60 LABELS "audit;exploit;ct;backend")

# -- BIP-340 Schnorr official KAT (sign indices 0-3, invalid verify 4-12) ----
add_executable(test_exploit_schnorr_bip340_kat_standalone test_exploit_schnorr_bip340_kat.cpp)
audit_target_defaults(test_exploit_schnorr_bip340_kat_standalone)
target_compile_definitions(test_exploit_schnorr_bip340_kat_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_schnorr_bip340_kat COMMAND test_exploit_schnorr_bip340_kat_standalone)
set_tests_properties(exploit_schnorr_bip340_kat PROPERTIES TIMEOUT 60 LABELS "audit;exploit;schnorr;bip340")

# -- RIPEMD-160 KAT (NIST vectors, hash160, ripemd160_32 fast path) ----------
add_executable(test_exploit_ripemd160_kat_standalone test_exploit_ripemd160_kat.cpp)
audit_target_defaults(test_exploit_ripemd160_kat_standalone)
target_compile_definitions(test_exploit_ripemd160_kat_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_ripemd160_kat COMMAND test_exploit_ripemd160_kat_standalone)
set_tests_properties(exploit_ripemd160_kat PROPERTIES TIMEOUT 60 LABELS "audit;exploit;hash;ripemd160")

# -- ChaCha20-Poly1305 RFC 8439 KAT (block, stream, Poly1305 MAC, AEAD) ------
add_executable(test_exploit_chacha20_kat_standalone test_exploit_chacha20_kat.cpp)
audit_target_defaults(test_exploit_chacha20_kat_standalone)
target_compile_definitions(test_exploit_chacha20_kat_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_chacha20_kat COMMAND test_exploit_chacha20_kat_standalone)
set_tests_properties(exploit_chacha20_kat PROPERTIES TIMEOUT 60 LABELS "audit;exploit;chacha20;rfc8439")

# -- EC point serialization (G coords, 2*G, infinity, parity, associativity) -
add_executable(test_exploit_point_serialization_standalone test_exploit_point_serialization.cpp)
audit_target_defaults(test_exploit_point_serialization_standalone)
target_compile_definitions(test_exploit_point_serialization_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_point_serialization COMMAND test_exploit_point_serialization_standalone)
set_tests_properties(exploit_point_serialization PROPERTIES TIMEOUT 60 LABELS "audit;exploit;ec;point")

# -- Point-at-Infinity edge cases (INF-1..INF-28: cancellation, serial, tweak) -
add_executable(test_infinity_edge_cases_standalone test_infinity_edge_cases.cpp)
audit_target_defaults(test_infinity_edge_cases_standalone)
target_compile_definitions(test_infinity_edge_cases_standalone PRIVATE STANDALONE_TEST)
add_test(NAME infinity_edge_cases COMMAND test_infinity_edge_cases_standalone)
set_tests_properties(infinity_edge_cases PROPERTIES TIMEOUT 60 LABELS "audit;exploit;ec;point;infinity")

# -- ECDSA RFC 6979 KAT (determinism, low-S, wrong-key/msg rejection) --------
add_executable(test_exploit_ecdsa_rfc6979_kat_standalone test_exploit_ecdsa_rfc6979_kat.cpp)
audit_target_defaults(test_exploit_ecdsa_rfc6979_kat_standalone)
target_compile_definitions(test_exploit_ecdsa_rfc6979_kat_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_ecdsa_rfc6979_kat COMMAND test_exploit_ecdsa_rfc6979_kat_standalone)
set_tests_properties(exploit_ecdsa_rfc6979_kat PROPERTIES TIMEOUT 60 LABELS "audit;exploit;ecdsa;rfc6979")

# -- FROST byzantine fault injection (bad partial, forged aggregate) ---------
add_executable(test_exploit_frost_byzantine_standalone test_exploit_frost_byzantine.cpp)
audit_target_defaults(test_exploit_frost_byzantine_standalone)
target_compile_definitions(test_exploit_frost_byzantine_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_frost_byzantine COMMAND test_exploit_frost_byzantine_standalone)
set_tests_properties(exploit_frost_byzantine PROPERTIES TIMEOUT 60 LABELS "audit;exploit;frost;byzantine")

# -- Scalar group law systematic (inverses, identities, overflow, roundtrip) --
add_executable(test_exploit_scalar_systematic_standalone test_exploit_scalar_systematic.cpp)
audit_target_defaults(test_exploit_scalar_systematic_standalone)
target_compile_definitions(test_exploit_scalar_systematic_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_scalar_systematic COMMAND test_exploit_scalar_systematic_standalone)
set_tests_properties(exploit_scalar_systematic PROPERTIES TIMEOUT 60 LABELS "audit;exploit;scalar;group")

# -- PoC: ECDSA malleability — low-s canonicalization, BIP-62/BIP-146
add_executable(test_exploit_ecdsa_malleability_standalone test_exploit_ecdsa_malleability.cpp)
audit_target_defaults(test_exploit_ecdsa_malleability_standalone)
target_compile_definitions(test_exploit_ecdsa_malleability_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_ecdsa_malleability COMMAND test_exploit_ecdsa_malleability_standalone)
set_tests_properties(exploit_ecdsa_malleability PROPERTIES TIMEOUT 60 LABELS "audit;exploit;ecdsa;malleability")

# -- PoC: GLV decomposition KAT — endomorphism and scalar-mul parity
add_executable(test_exploit_glv_kat_standalone test_exploit_glv_kat.cpp)
audit_target_defaults(test_exploit_glv_kat_standalone)
target_compile_definitions(test_exploit_glv_kat_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_glv_kat COMMAND test_exploit_glv_kat_standalone)
set_tests_properties(exploit_glv_kat PROPERTIES TIMEOUT 60 LABELS "audit;exploit;glv;kat")

# -- PoC: AEAD ChaCha20-Poly1305 integrity — MAC bypass and nonce reuse
add_executable(test_exploit_aead_integrity_standalone test_exploit_aead_integrity.cpp)
audit_target_defaults(test_exploit_aead_integrity_standalone)
target_compile_definitions(test_exploit_aead_integrity_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_aead_integrity COMMAND test_exploit_aead_integrity_standalone)
set_tests_properties(exploit_aead_integrity PROPERTIES TIMEOUT 60 LABELS "audit;exploit;aead;integrity")

# -- PoC: Schnorr batch forge — single-bad-sig detection and identify_invalid
add_executable(test_exploit_batch_schnorr_forge_standalone test_exploit_batch_schnorr_forge.cpp)
audit_target_defaults(test_exploit_batch_schnorr_forge_standalone)
target_compile_definitions(test_exploit_batch_schnorr_forge_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_batch_schnorr_forge COMMAND test_exploit_batch_schnorr_forge_standalone)
set_tests_properties(exploit_batch_schnorr_forge PROPERTIES TIMEOUT 60 LABELS "audit;exploit;schnorr;batch")

# -- PoC: BIP-32 hardened derivation edge cases — isolation and xpub guard
add_executable(test_exploit_bip32_ckd_hardened_standalone test_exploit_bip32_ckd_hardened.cpp)
audit_target_defaults(test_exploit_bip32_ckd_hardened_standalone)
target_compile_definitions(test_exploit_bip32_ckd_hardened_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_bip32_ckd_hardened COMMAND test_exploit_bip32_ckd_hardened_standalone)
set_tests_properties(exploit_bip32_ckd_hardened PROPERTIES TIMEOUT 60 LABELS "audit;exploit;bip32;hardened")

# -- PoC: ZK adversarial / malformed inputs — forged proofs, overflow, identity
add_executable(test_exploit_zk_adversarial_standalone test_exploit_zk_adversarial.cpp)
audit_target_defaults(test_exploit_zk_adversarial_standalone)
target_compile_definitions(test_exploit_zk_adversarial_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_zk_adversarial COMMAND test_exploit_zk_adversarial_standalone)
set_tests_properties(exploit_zk_adversarial PROPERTIES TIMEOUT 120 LABELS "audit;exploit;zk;adversarial")

# -- PoC: Pedersen adversarial / switch-commit — binding, negation, double-spend
add_executable(test_exploit_pedersen_adversarial_standalone test_exploit_pedersen_adversarial.cpp)
audit_target_defaults(test_exploit_pedersen_adversarial_standalone)
target_compile_definitions(test_exploit_pedersen_adversarial_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_pedersen_adversarial COMMAND test_exploit_pedersen_adversarial_standalone)
set_tests_properties(exploit_pedersen_adversarial PROPERTIES TIMEOUT 60 LABELS "audit;exploit;pedersen;adversarial")

# -- Ethereum differential: our output vs. go-ethereum/web3.py/ethers.js KATs
add_executable(test_exploit_ethereum_differential_standalone test_exploit_ethereum_differential.cpp)
audit_target_defaults(test_exploit_ethereum_differential_standalone)
target_compile_definitions(test_exploit_ethereum_differential_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_ethereum_differential COMMAND test_exploit_ethereum_differential_standalone)
set_tests_properties(exploit_ethereum_differential PROPERTIES TIMEOUT 60 LABELS "audit;exploit;ethereum;differential")

# -- MuSig2/FROST/adaptor parser fuzz (deterministic, no libFuzzer required) --
add_executable(test_fuzz_musig2_frost_standalone test_fuzz_musig2_frost.cpp)
audit_target_defaults(test_fuzz_musig2_frost_standalone)
target_link_libraries(test_fuzz_musig2_frost_standalone PRIVATE ufsecp_static)
target_include_directories(test_fuzz_musig2_frost_standalone PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_BINARY_DIR}/include/ufsecp
    ${CMAKE_CURRENT_SOURCE_DIR}/../cpu/include
)
target_compile_definitions(test_fuzz_musig2_frost_standalone PRIVATE
    STANDALONE_TEST "UFSECP_API=")
add_test(NAME fuzz_musig2_frost COMMAND test_fuzz_musig2_frost_standalone)
set_tests_properties(fuzz_musig2_frost PROPERTIES TIMEOUT 120 LABELS "audit;fuzz;musig2;frost;adaptor")

# -- PoC: Invalid curve / twist point injection (Tier-1 audit probe) ---------
add_executable(test_exploit_invalid_curve_twist_standalone test_exploit_invalid_curve_twist.cpp)
audit_target_defaults(test_exploit_invalid_curve_twist_standalone)
target_compile_definitions(test_exploit_invalid_curve_twist_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_invalid_curve_twist COMMAND test_exploit_invalid_curve_twist_standalone)
set_tests_properties(exploit_invalid_curve_twist PROPERTIES TIMEOUT 120 LABELS "audit;exploit;twist;ecdh")

# -- PoC: Schnorr BIP-340 forgery / out-of-range rejection -------------------
add_executable(test_exploit_schnorr_forgery_vectors_standalone test_exploit_schnorr_forgery_vectors.cpp)
audit_target_defaults(test_exploit_schnorr_forgery_vectors_standalone)
target_compile_definitions(test_exploit_schnorr_forgery_vectors_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_schnorr_forgery_vectors COMMAND test_exploit_schnorr_forgery_vectors_standalone)
set_tests_properties(exploit_schnorr_forgery_vectors PROPERTIES TIMEOUT 120 LABELS "audit;exploit;schnorr;bip340")

# -- PoC: Secret key tweak overflow / cancellation (BIP-32, Taproot) ---------
add_executable(test_exploit_seckey_tweak_cancel_standalone test_exploit_seckey_tweak_cancel.cpp)
audit_target_defaults(test_exploit_seckey_tweak_cancel_standalone)
target_compile_definitions(test_exploit_seckey_tweak_cancel_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_seckey_tweak_cancel COMMAND test_exploit_seckey_tweak_cancel_standalone)
set_tests_properties(exploit_seckey_tweak_cancel PROPERTIES TIMEOUT 60 LABELS "audit;exploit;tweak;bip32")

# -- PoC: Hedged nonce bias detection (lattice attack surface) ---------------
add_executable(test_exploit_hedged_nonce_bias_standalone test_exploit_hedged_nonce_bias.cpp)
audit_target_defaults(test_exploit_hedged_nonce_bias_standalone)
target_compile_definitions(test_exploit_hedged_nonce_bias_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_hedged_nonce_bias COMMAND test_exploit_hedged_nonce_bias_standalone)
set_tests_properties(exploit_hedged_nonce_bias PROPERTIES TIMEOUT 120 LABELS "audit;exploit;nonce;lattice")

# -- PoC: ECDSA ecrecover / key recovery confusion ---------------------------
add_executable(test_exploit_ecrecover_confusion_standalone test_exploit_ecrecover_confusion.cpp)
audit_target_defaults(test_exploit_ecrecover_confusion_standalone)
target_compile_definitions(test_exploit_ecrecover_confusion_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_ecrecover_confusion COMMAND test_exploit_ecrecover_confusion_standalone)
set_tests_properties(exploit_ecrecover_confusion PROPERTIES TIMEOUT 120 LABELS "audit;exploit;ecdsa;recovery")

# -- PoC: Batch verification poisoning (single bad sig detection) ------------
add_executable(test_exploit_batch_verify_poison_standalone test_exploit_batch_verify_poison.cpp)
audit_target_defaults(test_exploit_batch_verify_poison_standalone)
target_compile_definitions(test_exploit_batch_verify_poison_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_batch_verify_poison COMMAND test_exploit_batch_verify_poison_standalone)
set_tests_properties(exploit_batch_verify_poison PROPERTIES TIMEOUT 120 LABELS "audit;exploit;batch;schnorr")

# -- PoC: Context / precompute lifecycle hostile caller patterns --------------
add_executable(test_exploit_ctx_lifecycle_hostile_standalone test_exploit_ctx_lifecycle_hostile.cpp)
audit_target_defaults(test_exploit_ctx_lifecycle_hostile_standalone)
target_compile_definitions(test_exploit_ctx_lifecycle_hostile_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_ctx_lifecycle_hostile COMMAND test_exploit_ctx_lifecycle_hostile_standalone)
set_tests_properties(exploit_ctx_lifecycle_hostile PROPERTIES TIMEOUT 60 LABELS "audit;exploit;context;lifecycle")

# -- PoC: DER / signature encoding differential (consensus-critical) ---------
add_executable(test_exploit_der_parsing_differential_standalone test_exploit_der_parsing_differential.cpp)
audit_target_defaults(test_exploit_der_parsing_differential_standalone)
target_compile_definitions(test_exploit_der_parsing_differential_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_der_parsing_differential COMMAND test_exploit_der_parsing_differential_standalone)
set_tests_properties(exploit_der_parsing_differential PROPERTIES TIMEOUT 60 LABELS "audit;exploit;ecdsa;der")

# -- PoC: GPU/CPU divergence (algebraic consistency cross-check) -------------
add_executable(test_exploit_gpu_cpu_divergence_standalone test_exploit_gpu_cpu_divergence.cpp)
audit_target_defaults(test_exploit_gpu_cpu_divergence_standalone)
target_compile_definitions(test_exploit_gpu_cpu_divergence_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_gpu_cpu_divergence COMMAND test_exploit_gpu_cpu_divergence_standalone)
set_tests_properties(exploit_gpu_cpu_divergence PROPERTIES TIMEOUT 120 LABELS "audit;exploit;divergence;algebraic")

# -- PoC: Field boundary / carry propagation exhaustive ----------------------
add_executable(test_exploit_field_boundary_exhaustive_standalone test_exploit_field_boundary_exhaustive.cpp)
audit_target_defaults(test_exploit_field_boundary_exhaustive_standalone)
target_compile_definitions(test_exploit_field_boundary_exhaustive_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_field_boundary_exhaustive COMMAND test_exploit_field_boundary_exhaustive_standalone)
set_tests_properties(exploit_field_boundary_exhaustive PROPERTIES TIMEOUT 120 LABELS "audit;exploit;field;boundary")

# -- PoC: Schnorr x-only parity confusion -----------------------------------
add_executable(test_exploit_schnorr_xonly_parity_confusion_standalone test_exploit_schnorr_xonly_parity_confusion.cpp)
audit_target_defaults(test_exploit_schnorr_xonly_parity_confusion_standalone)
target_compile_definitions(test_exploit_schnorr_xonly_parity_confusion_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_schnorr_xonly_parity_confusion COMMAND test_exploit_schnorr_xonly_parity_confusion_standalone)
set_tests_properties(exploit_schnorr_xonly_parity_confusion PROPERTIES TIMEOUT 60 LABELS "audit;exploit;schnorr;taproot;parity")

# -- PoC: Taproot Merkle path alias -----------------------------------------
add_executable(test_exploit_taproot_merkle_path_alias_standalone test_exploit_taproot_merkle_path_alias.cpp)
audit_target_defaults(test_exploit_taproot_merkle_path_alias_standalone)
target_compile_definitions(test_exploit_taproot_merkle_path_alias_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_taproot_merkle_path_alias COMMAND test_exploit_taproot_merkle_path_alias_standalone)
set_tests_properties(exploit_taproot_merkle_path_alias PROPERTIES TIMEOUT 60 LABELS "audit;exploit;taproot;merkle")

# -- PoC: FROST binding factor mismatch -------------------------------------
add_executable(test_exploit_frost_binding_factor_mismatch_standalone test_exploit_frost_binding_factor_mismatch.cpp)
audit_target_defaults(test_exploit_frost_binding_factor_mismatch_standalone)
target_compile_definitions(test_exploit_frost_binding_factor_mismatch_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_frost_binding_factor_mismatch COMMAND test_exploit_frost_binding_factor_mismatch_standalone)
set_tests_properties(exploit_frost_binding_factor_mismatch PROPERTIES TIMEOUT 120 LABELS "audit;exploit;frost;binding")

# -- PoC: BIP-32 parent fingerprint confusion --------------------------------
add_executable(test_exploit_bip32_parent_fingerprint_confusion_standalone test_exploit_bip32_parent_fingerprint_confusion.cpp)
audit_target_defaults(test_exploit_bip32_parent_fingerprint_confusion_standalone)
target_compile_definitions(test_exploit_bip32_parent_fingerprint_confusion_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_bip32_parent_fingerprint_confusion COMMAND test_exploit_bip32_parent_fingerprint_confusion_standalone)
set_tests_properties(exploit_bip32_parent_fingerprint_confusion PROPERTIES TIMEOUT 60 LABELS "audit;exploit;bip32;fingerprint")

# -- PoC: ECIES envelope confusion ------------------------------------------
add_executable(test_exploit_ecies_envelope_confusion_standalone test_exploit_ecies_envelope_confusion.cpp)
audit_target_defaults(test_exploit_ecies_envelope_confusion_standalone)
target_compile_definitions(test_exploit_ecies_envelope_confusion_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_ecies_envelope_confusion COMMAND test_exploit_ecies_envelope_confusion_standalone)
set_tests_properties(exploit_ecies_envelope_confusion PROPERTIES TIMEOUT 60 LABELS "audit;exploit;ecies;envelope")

# -- PoC: Wallet cross-domain replay ----------------------------------------
add_executable(test_exploit_wallet_cross_domain_replay_standalone test_exploit_wallet_cross_domain_replay.cpp)
audit_target_defaults(test_exploit_wallet_cross_domain_replay_standalone)
target_compile_definitions(test_exploit_wallet_cross_domain_replay_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_wallet_cross_domain_replay COMMAND test_exploit_wallet_cross_domain_replay_standalone)
set_tests_properties(exploit_wallet_cross_domain_replay PROPERTIES TIMEOUT 60 LABELS "audit;exploit;wallet;replay;domain")

# -- PoC: GPU host API shape (hostile caller) --------------------------------
add_executable(test_exploit_gpu_host_api_shape_standalone test_exploit_gpu_host_api_shape.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/ufsecp/ufsecp_impl.cpp)
audit_target_defaults(test_exploit_gpu_host_api_shape_standalone)
target_include_directories(test_exploit_gpu_host_api_shape_standalone PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_BINARY_DIR}/include
)
target_compile_definitions(test_exploit_gpu_host_api_shape_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_gpu_host_api_shape COMMAND test_exploit_gpu_host_api_shape_standalone)
set_tests_properties(exploit_gpu_host_api_shape PROPERTIES TIMEOUT 60 LABELS "audit;exploit;gpu;ffi;hostile")

# -- PoC: Silent Payment confusion (BIP-352 key role / output isolation) -----
add_executable(test_exploit_silent_payment_confusion_standalone test_exploit_silent_payment_confusion.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/ufsecp/ufsecp_impl.cpp)
audit_target_defaults(test_exploit_silent_payment_confusion_standalone)
target_include_directories(test_exploit_silent_payment_confusion_standalone PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_BINARY_DIR}/include
)
target_compile_definitions(test_exploit_silent_payment_confusion_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_silent_payment_confusion COMMAND test_exploit_silent_payment_confusion_standalone)
set_tests_properties(exploit_silent_payment_confusion PROPERTIES TIMEOUT 60 LABELS "audit;exploit;bip352;silent_payment")

# -- PoC: CT vs Fast implementation equivalence ------------------------------
add_executable(test_exploit_ct_fast_equivalence_standalone test_exploit_ct_fast_equivalence.cpp)
audit_target_defaults(test_exploit_ct_fast_equivalence_standalone)
target_compile_definitions(test_exploit_ct_fast_equivalence_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_ct_fast_equivalence COMMAND test_exploit_ct_fast_equivalence_standalone)
set_tests_properties(exploit_ct_fast_equivalence PROPERTIES TIMEOUT 120 LABELS "audit;exploit;ct;equivalence")

# -- PoC: Pedersen switch commitment misuse ----------------------------------
add_executable(test_exploit_pedersen_switch_misuse_standalone test_exploit_pedersen_switch_misuse.cpp)
audit_target_defaults(test_exploit_pedersen_switch_misuse_standalone)
target_compile_definitions(test_exploit_pedersen_switch_misuse_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_pedersen_switch_misuse COMMAND test_exploit_pedersen_switch_misuse_standalone)
set_tests_properties(exploit_pedersen_switch_misuse PROPERTIES TIMEOUT 60 LABELS "audit;exploit;pedersen;switch")

# -- ECIES regression + C ABI prefix enforcement ----------------------------
add_executable(test_ecies_regression_standalone test_ecies_regression.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/ufsecp/ufsecp_impl.cpp)
audit_target_defaults(test_ecies_regression_standalone)
target_include_directories(test_ecies_regression_standalone PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_BINARY_DIR}/include
)
target_compile_definitions(test_ecies_regression_standalone PRIVATE
    UFSECP_BUILDING STANDALONE_ECIES_REGRESSION)
add_test(NAME ecies_regression COMMAND test_ecies_regression_standalone)
set_tests_properties(ecies_regression PROPERTIES TIMEOUT 600 LABELS "audit")

# ===========================================================================
# Feature Exploit PoC Standalone Tests (P2SH, BIP-85, BIP-340, BIP-322, GCS, PSBT, Desc)
# ===========================================================================

# -- PoC: P2SH / P2SH-P2WPKH address type confusion -------------------------
add_executable(test_exploit_p2sh_address_confusion_standalone
    test_exploit_p2sh_address_confusion.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/ufsecp/ufsecp_impl.cpp)
audit_target_defaults(test_exploit_p2sh_address_confusion_standalone)
target_include_directories(test_exploit_p2sh_address_confusion_standalone PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_BINARY_DIR}/include
)
target_compile_definitions(test_exploit_p2sh_address_confusion_standalone PRIVATE
    UFSECP_BUILDING STANDALONE_TEST)
add_test(NAME exploit_p2sh_address_confusion
    COMMAND test_exploit_p2sh_address_confusion_standalone)
set_tests_properties(exploit_p2sh_address_confusion PROPERTIES
    TIMEOUT 60 LABELS "audit;exploit;address;p2sh")

# -- PoC: BIP-85 path collision and entropy security -------------------------
add_executable(test_exploit_bip85_path_collision_standalone
    test_exploit_bip85_path_collision.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/ufsecp/ufsecp_impl.cpp)
audit_target_defaults(test_exploit_bip85_path_collision_standalone)
target_include_directories(test_exploit_bip85_path_collision_standalone PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_BINARY_DIR}/include
)
target_compile_definitions(test_exploit_bip85_path_collision_standalone PRIVATE
    UFSECP_BUILDING STANDALONE_TEST)
add_test(NAME exploit_bip85_path_collision
    COMMAND test_exploit_bip85_path_collision_standalone)
set_tests_properties(exploit_bip85_path_collision PROPERTIES
    TIMEOUT 60 LABELS "audit;exploit;bip85;entropy")

# -- PoC: Schnorr variable-length message length confusion -------------------
add_executable(test_exploit_schnorr_msg_length_confusion_standalone
    test_exploit_schnorr_msg_length_confusion.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/ufsecp/ufsecp_impl.cpp)
audit_target_defaults(test_exploit_schnorr_msg_length_confusion_standalone)
target_include_directories(test_exploit_schnorr_msg_length_confusion_standalone PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_BINARY_DIR}/include
)
target_compile_definitions(test_exploit_schnorr_msg_length_confusion_standalone PRIVATE
    UFSECP_BUILDING STANDALONE_TEST)
add_test(NAME exploit_schnorr_msg_length_confusion
    COMMAND test_exploit_schnorr_msg_length_confusion_standalone)
set_tests_properties(exploit_schnorr_msg_length_confusion PROPERTIES
    TIMEOUT 60 LABELS "audit;exploit;schnorr;bip340")

# -- PoC: BIP-322 generic message signing type confusion ---------------------
add_executable(test_exploit_bip322_type_confusion_standalone
    test_exploit_bip322_type_confusion.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/ufsecp/ufsecp_impl.cpp)
audit_target_defaults(test_exploit_bip322_type_confusion_standalone)
target_include_directories(test_exploit_bip322_type_confusion_standalone PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_BINARY_DIR}/include
)
target_compile_definitions(test_exploit_bip322_type_confusion_standalone PRIVATE
    UFSECP_BUILDING STANDALONE_TEST)
add_test(NAME exploit_bip322_type_confusion
    COMMAND test_exploit_bip322_type_confusion_standalone)
set_tests_properties(exploit_bip322_type_confusion PROPERTIES
    TIMEOUT 60 LABELS "audit;exploit;bip322;signing")

# -- PoC: BIP-158 GCS filter false positive and correctness ------------------
add_executable(test_exploit_gcs_false_positive_standalone
    test_exploit_gcs_false_positive.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/ufsecp/ufsecp_impl.cpp)
audit_target_defaults(test_exploit_gcs_false_positive_standalone)
target_include_directories(test_exploit_gcs_false_positive_standalone PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_BINARY_DIR}/include
)
target_compile_definitions(test_exploit_gcs_false_positive_standalone PRIVATE
    UFSECP_BUILDING STANDALONE_TEST)
add_test(NAME exploit_gcs_false_positive
    COMMAND test_exploit_gcs_false_positive_standalone)
set_tests_properties(exploit_gcs_false_positive PROPERTIES
    TIMEOUT 60 LABELS "audit;exploit;bip158;gcs")

# -- PoC: PSBT input type confusion (BIP-174/370) ----------------------------
add_executable(test_exploit_psbt_input_confusion_standalone
    test_exploit_psbt_input_confusion.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/ufsecp/ufsecp_impl.cpp)
audit_target_defaults(test_exploit_psbt_input_confusion_standalone)
target_include_directories(test_exploit_psbt_input_confusion_standalone PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_BINARY_DIR}/include
)
target_compile_definitions(test_exploit_psbt_input_confusion_standalone PRIVATE
    UFSECP_BUILDING STANDALONE_TEST)
add_test(NAME exploit_psbt_input_confusion
    COMMAND test_exploit_psbt_input_confusion_standalone)
set_tests_properties(exploit_psbt_input_confusion PROPERTIES
    TIMEOUT 60 LABELS "audit;exploit;psbt;bip174")

# -- PoC: Output descriptor injection and type confusion (BIP-380-386) -------
add_executable(test_exploit_descriptor_injection_standalone
    test_exploit_descriptor_injection.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/ufsecp/ufsecp_impl.cpp)
audit_target_defaults(test_exploit_descriptor_injection_standalone)
target_include_directories(test_exploit_descriptor_injection_standalone PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_BINARY_DIR}/include
)
target_compile_definitions(test_exploit_descriptor_injection_standalone PRIVATE
    UFSECP_BUILDING STANDALONE_TEST)
add_test(NAME exploit_descriptor_injection
    COMMAND test_exploit_descriptor_injection_standalone)
set_tests_properties(exploit_descriptor_injection PROPERTIES
    TIMEOUT 60 LABELS "audit;exploit;descriptor;bip380")

# -- PoC: Secret key arithmetic (tweak_add/mul, negate, cross-check) --------
add_executable(test_exploit_seckey_arith_standalone test_exploit_seckey_arith.cpp)
audit_target_defaults(test_exploit_seckey_arith_standalone)
target_compile_definitions(test_exploit_seckey_arith_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_seckey_arith COMMAND test_exploit_seckey_arith_standalone)
set_tests_properties(exploit_seckey_arith PROPERTIES TIMEOUT 60 LABELS "audit;exploit;seckey;tweak")

# -- PoC: Context clone isolation (CLN-1..CLN-6) ----------------------------
add_executable(test_exploit_ctx_clone_standalone test_exploit_ctx_clone.cpp)
audit_target_defaults(test_exploit_ctx_clone_standalone)
target_link_libraries(test_exploit_ctx_clone_standalone PRIVATE Threads::Threads)
target_compile_definitions(test_exploit_ctx_clone_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_ctx_clone COMMAND test_exploit_ctx_clone_standalone)
set_tests_properties(exploit_ctx_clone PROPERTIES TIMEOUT 60 LABELS "audit;exploit;context;clone")

# -- Diagnostic: ct::scalar_mul step-by-step comparison --------------------
add_executable(diag_scalar_mul ${CPU_TESTS_DIR}/diag_scalar_mul.cpp)
audit_target_defaults(diag_scalar_mul)
target_compile_definitions(diag_scalar_mul PRIVATE STANDALONE_TEST)
add_test(NAME diag_scalar_mul COMMAND diag_scalar_mul)

# ===========================================================================
# Cross-library differential test (vs bitcoin-core/libsecp256k1)
# ===========================================================================
option(SECP256K1_BUILD_CROSS_TESTS
       "Build in-process differential tests against bitcoin-core/libsecp256k1" OFF)

if(SECP256K1_BUILD_CROSS_TESTS)
    include(FetchContent)
    FetchContent_Declare(
        libsecp256k1_ref
        GIT_REPOSITORY https://github.com/bitcoin-core/secp256k1.git
        GIT_TAG        v0.6.0
        GIT_SHALLOW    TRUE
    )
    FetchContent_GetProperties(libsecp256k1_ref)
    if(NOT libsecp256k1_ref_POPULATED)
        FetchContent_Populate(libsecp256k1_ref)
        set(BUILD_SHARED_LIBS                  OFF CACHE BOOL "" FORCE)
        set(SECP256K1_BUILD_SHARED             OFF CACHE BOOL "" FORCE)
        set(SECP256K1_ENABLE_MODULE_SCHNORRSIG ON  CACHE BOOL "" FORCE)
        set(SECP256K1_ENABLE_MODULE_EXTRAKEYS  ON  CACHE BOOL "" FORCE)
        set(SECP256K1_ENABLE_MODULE_RECOVERY   ON  CACHE BOOL "" FORCE)
        set(SECP256K1_ENABLE_MODULE_ECDH       ON  CACHE BOOL "" FORCE)
        set(SECP256K1_BUILD_BENCHMARK          OFF CACHE BOOL "" FORCE)
        set(SECP256K1_BUILD_EXAMPLES           OFF CACHE BOOL "" FORCE)
        set(SECP256K1_BUILD_PROGRAMS           OFF CACHE BOOL "" FORCE)
        set(SECP256K1_BUILD_TESTS              OFF CACHE BOOL "" FORCE)
        set(SECP256K1_BUILD_EXHAUSTIVE_TESTS   OFF CACHE BOOL "" FORCE)
        add_subdirectory(
            ${libsecp256k1_ref_SOURCE_DIR}
            ${libsecp256k1_ref_BINARY_DIR}
            EXCLUDE_FROM_ALL
        )
    endif()

    add_executable(test_cross_libsecp256k1 test_cross_libsecp256k1.cpp)
    target_link_libraries(test_cross_libsecp256k1 PRIVATE fastsecp256k1 secp256k1)
    target_include_directories(test_cross_libsecp256k1
        PRIVATE ${libsecp256k1_ref_SOURCE_DIR}/include
    )
    if(MSVC OR (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND WIN32))
        target_link_options(test_cross_libsecp256k1 PRIVATE "LINKER:/STACK:8388608")
    endif()
    add_test(NAME cross_libsecp256k1 COMMAND test_cross_libsecp256k1)
    set_tests_properties(cross_libsecp256k1 PROPERTIES TIMEOUT 900 LABELS "audit;exploit;differential;libsecp")
    message(STATUS "  Cross-test vs libsecp256k1: ON (ref: v0.6.0)")

    # NOTE: CT vs libsecp benchmark removed -- use cpu/bench/bench_unified instead
    #       (bench_unified Section 8 has full CT-vs-CT apple-to-apple ratios)
endif()

# ===========================================================================
# Parser fuzz tests (deterministic pseudo-fuzz)
# ===========================================================================
option(SECP256K1_BUILD_FUZZ_TESTS
       "Build deterministic fuzz tests for parsers (DER, Schnorr, Pubkey)" OFF)

if(SECP256K1_BUILD_FUZZ_TESTS AND TARGET ufsecp_static)
    add_executable(test_fuzz_parsers test_fuzz_parsers.cpp)
    target_link_libraries(test_fuzz_parsers PRIVATE ufsecp_static fastsecp256k1)
    target_include_directories(test_fuzz_parsers PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
        ${CMAKE_BINARY_DIR}/include/ufsecp
        ${CMAKE_CURRENT_SOURCE_DIR}/../cpu/include
    )
    target_compile_definitions(test_fuzz_parsers PRIVATE "UFSECP_API=")
    if(MSVC OR (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND WIN32))
        target_link_options(test_fuzz_parsers PRIVATE "LINKER:/STACK:8388608")
    endif()
    add_test(NAME fuzz_parsers COMMAND test_fuzz_parsers)
    set_tests_properties(fuzz_parsers PROPERTIES TIMEOUT 900)
    message(STATUS "  Parser fuzz tests: ON")

    # Address + BIP32 + FFI boundary fuzz
    add_executable(test_fuzz_address_bip32_ffi test_fuzz_address_bip32_ffi.cpp)
    target_link_libraries(test_fuzz_address_bip32_ffi PRIVATE ufsecp_static fastsecp256k1)
    target_include_directories(test_fuzz_address_bip32_ffi PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
        ${CMAKE_BINARY_DIR}/include/ufsecp
        ${CMAKE_CURRENT_SOURCE_DIR}/../cpu/include
    )
    target_compile_definitions(test_fuzz_address_bip32_ffi PRIVATE "UFSECP_API=")
    if(MSVC OR (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND WIN32))
        target_link_options(test_fuzz_address_bip32_ffi PRIVATE "LINKER:/STACK:8388608")
    endif()
    add_test(NAME fuzz_address_bip32_ffi COMMAND test_fuzz_address_bip32_ffi)
    set_tests_properties(fuzz_address_bip32_ffi PROPERTIES TIMEOUT 900)
    message(STATUS "  Address + BIP32 + FFI fuzz tests: ON")

    # Disable LTO when CUDA is linked (nvcc LTO version differs from host compiler)
    if(SECP256K1_BUILD_CUDA)
        set_target_properties(
            test_fuzz_parsers test_fuzz_address_bip32_ffi
            PROPERTIES INTERPROCEDURAL_OPTIMIZATION FALSE
        )
    endif()
endif()

# ===========================================================================
# MuSig2 + FROST protocol tests
# ===========================================================================
option(SECP256K1_BUILD_PROTOCOL_TESTS
       "Build MuSig2 + FROST protocol tests" OFF)

if(SECP256K1_BUILD_PROTOCOL_TESTS)
    add_executable(test_musig2_frost test_musig2_frost.cpp)
    target_link_libraries(test_musig2_frost PRIVATE fastsecp256k1)
    target_include_directories(test_musig2_frost PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../cpu/include
    )
    if(MSVC OR (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND WIN32))
        target_link_options(test_musig2_frost PRIVATE "LINKER:/STACK:8388608")
    endif()
    add_test(NAME musig2_frost COMMAND test_musig2_frost)
    set_tests_properties(musig2_frost PROPERTIES TIMEOUT 600)
    message(STATUS "  MuSig2 + FROST protocol tests: ON")

    # Advanced protocol tests
    add_executable(test_musig2_frost_advanced test_musig2_frost_advanced.cpp)
    target_link_libraries(test_musig2_frost_advanced PRIVATE fastsecp256k1)
    target_include_directories(test_musig2_frost_advanced PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../cpu/include
    )
    if(MSVC OR (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND WIN32))
        target_link_options(test_musig2_frost_advanced PRIVATE "LINKER:/STACK:8388608")
    endif()
    add_test(NAME musig2_frost_advanced COMMAND test_musig2_frost_advanced)
    set_tests_properties(musig2_frost_advanced PROPERTIES TIMEOUT 600)
    message(STATUS "  MuSig2 + FROST advanced tests: ON")

    # FROST reference KAT vectors
    add_executable(test_frost_kat test_frost_kat.cpp)
    target_link_libraries(test_frost_kat PRIVATE fastsecp256k1)
    target_include_directories(test_frost_kat PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../cpu/include
    )
    if(MSVC OR (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND WIN32))
        target_link_options(test_frost_kat PRIVATE "LINKER:/STACK:8388608")
    endif()
    add_test(NAME frost_kat COMMAND test_frost_kat)
    set_tests_properties(frost_kat PROPERTIES TIMEOUT 600)
    message(STATUS "  FROST reference KAT vectors: ON")

    # MuSig2 BIP-327 reference vectors
    add_executable(test_musig2_bip327_vectors test_musig2_bip327_vectors.cpp)
    target_link_libraries(test_musig2_bip327_vectors PRIVATE fastsecp256k1)
    target_include_directories(test_musig2_bip327_vectors PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../cpu/include
    )
    if(MSVC OR (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND WIN32))
        target_link_options(test_musig2_bip327_vectors PRIVATE "LINKER:/STACK:8388608")
    endif()
    add_test(NAME musig2_bip327_vectors COMMAND test_musig2_bip327_vectors)
    set_tests_properties(musig2_bip327_vectors PROPERTIES TIMEOUT 600)
    message(STATUS "  MuSig2 BIP-327 reference vectors: ON")
endif()

# ===========================================================================
# Unified Audit Runner -- Unified Audit Binary
# ===========================================================================
# Single binary that runs ALL test modules + generates JSON/TXT reports.
# Build once, run on any platform. Self-audit artifact.
add_executable(unified_audit_runner
    unified_audit_runner.cpp
    # -- selftest modules (from cpu/tests/) --
    ${CPU_TESTS_DIR}/test_large_scalar_multiplication.cpp
    ${CPU_TESTS_DIR}/test_mul.cpp
    ${CPU_TESTS_DIR}/test_arithmetic_correctness.cpp
    ${CPU_TESTS_DIR}/test_ct.cpp
    ${CPU_TESTS_DIR}/test_ct_equivalence.cpp
    ${CPU_TESTS_DIR}/test_ecdsa_schnorr.cpp
    ${CPU_TESTS_DIR}/test_multiscalar_batch.cpp
    ${CPU_TESTS_DIR}/test_bip32.cpp
    ${CPU_TESTS_DIR}/test_bip32_vectors.cpp
    ${CPU_TESTS_DIR}/test_bip39.cpp
    ${CPU_TESTS_DIR}/test_musig2.cpp
    ${CPU_TESTS_DIR}/test_ecdh_recovery_taproot.cpp
    ${CPU_TESTS_DIR}/test_edge_cases.cpp
    ${CPU_TESTS_DIR}/test_v4_features.cpp
    ${CPU_TESTS_DIR}/test_coins.cpp
    ${CPU_TESTS_DIR}/test_batch_add_affine.cpp
    ${CPU_TESTS_DIR}/test_hash_accel.cpp
    ${CPU_TESTS_DIR}/test_exhaustive.cpp
    ${CPU_TESTS_DIR}/test_comprehensive.cpp
    ${CPU_TESTS_DIR}/test_bip340_vectors.cpp
    ${CPU_TESTS_DIR}/test_bip340_strict.cpp
    ${CPU_TESTS_DIR}/test_rfc6979_vectors.cpp
    ${CPU_TESTS_DIR}/test_ecc_properties.cpp
    # -- standalone audit modules (in this directory) --
    test_carry_propagation.cpp
    test_fault_injection.cpp
    test_fiat_crypto_vectors.cpp
    test_cross_platform_kat.cpp
    test_debug_invariants.cpp
    test_abi_gate.cpp
    test_ct_sidechannel.cpp
    differential_test.cpp
    # -- Wycheproof (Track I3) --
    test_wycheproof_ecdsa.cpp
    test_wycheproof_ecdh.cpp
    # -- Batch verify randomness (Track I6-3) --
    test_batch_randomness.cpp
    # -- CT formal verification (Track I5-1) --
    test_ct_verif_formal.cpp
    # -- Independent reference linkage (Track I5-2) --
    test_fiat_crypto_linkage.cpp
    # -- MuSig2 / FROST / adversarial / fuzz --
    test_musig2_frost.cpp
    test_musig2_frost_advanced.cpp
    test_frost_kat.cpp
    test_musig2_bip327_vectors.cpp
    test_ffi_round_trip.cpp
    test_adversarial_protocol.cpp
    test_ecies_regression.cpp
    audit_fuzz.cpp
    test_fuzz_parsers.cpp
    test_fuzz_address_bip32_ffi.cpp
    test_fuzz_musig2_frost.cpp
    # -- Deep audit modules --
    audit_field.cpp
    audit_scalar.cpp
    audit_point.cpp
    audit_ct.cpp
    audit_integration.cpp
    audit_security.cpp
    audit_perf.cpp
    audit_zk.cpp
    # -- Specification oracle & invariant monitor --
    test_secp256k1_spec.cpp
    audit_invariants.cpp
    # -- C ABI negative contract tests --
    test_c_abi_negative.cpp
    test_c_abi_thread_stress.cpp
    # -- Security audit modules --
    audit_secure_erase.cpp
    audit_ct_namespace.cpp
    test_kat_all_operations.cpp
    test_nonce_uniqueness.cpp
    test_parse_strictness.cpp
    # -- ufsecp FFI implementation (needed by fuzz_parsers + fuzz_address) --
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/ufsecp/ufsecp_impl.cpp
    # -- ufsecp GPU ABI implementation + tests (null-guard paths work without hardware) --
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/ufsecp/ufsecp_gpu_impl.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../gpu/src/gpu_registry.cpp
    test_gpu_host_api_negative.cpp
    test_gpu_abi_gate.cpp
    # -- field representation tests --
    ${CPU_TESTS_DIR}/test_field_26.cpp
    # -- diagnostics --
    ${CPU_TESTS_DIR}/diag_scalar_mul.cpp
    # -- Exploit PoC tests (110 attack vectors, Section 9) --
    test_exploit_adaptor_extended.cpp
    test_exploit_adaptor_parity.cpp
    test_exploit_address_encoding.cpp
    test_exploit_address_generation.cpp
    test_exploit_aead_integrity.cpp
    test_exploit_backend_divergence.cpp
    test_exploit_batch_schnorr.cpp
    test_exploit_batch_schnorr_forge.cpp
    test_exploit_batch_soundness.cpp
    test_exploit_batch_verify_correctness.cpp
    test_exploit_batch_verify_poison.cpp
    test_exploit_bip143_sighash.cpp
    test_exploit_bip144_serialization.cpp
    test_exploit_bip324_counter_desync.cpp
    test_exploit_bip324_session.cpp
    test_exploit_bip324_transcript_splice.cpp
    test_exploit_bip32_ckd_hardened.cpp
    test_exploit_bip32_depth.cpp
    test_exploit_bip32_derivation.cpp
    test_exploit_bip32_parent_fingerprint_confusion.cpp
    test_exploit_bip32_path_overflow.cpp
    test_exploit_bip39_entropy.cpp
    test_exploit_bip39_mnemonic.cpp
    test_exploit_bitcoin_message_signing.cpp
    test_exploit_chacha20_kat.cpp
    test_exploit_chacha20_nonce_reuse.cpp
    test_exploit_chacha20_poly1305.cpp
    test_exploit_coin_hd_derivation.cpp
    test_exploit_ct_fast_equivalence.cpp
    test_exploit_ct_recov.cpp
    test_exploit_ct_systematic.cpp
    test_exploit_ctx_lifecycle_hostile.cpp
    test_exploit_der_parsing_differential.cpp
    test_exploit_ecdh.cpp
    test_exploit_ecdh_degenerate.cpp
    test_exploit_ecdh_variants.cpp
    test_exploit_ecdsa_der_confusion.cpp
    test_exploit_ecdsa_edge_cases.cpp
    test_exploit_ecdsa_malleability.cpp
    test_exploit_ecdsa_recovery.cpp
    test_exploit_ecdsa_rfc6979_kat.cpp
    test_exploit_ecies_auth.cpp
    test_exploit_ecies_encryption.cpp
    test_exploit_ecies_envelope_confusion.cpp
    test_exploit_ecies_roundtrip.cpp
    test_exploit_ecrecover_confusion.cpp
    test_exploit_ellswift.cpp
    test_exploit_ellswift_ecdh.cpp
    test_exploit_ethereum_differential.cpp
    test_exploit_eth_signing.cpp
    test_exploit_field_arithmetic.cpp
    test_exploit_field_boundary_exhaustive.cpp
    test_exploit_frost_binding_factor_mismatch.cpp
    test_exploit_frost_byzantine.cpp
    test_exploit_frost_commitment_reuse.cpp
    test_exploit_frost_dkg.cpp
    test_exploit_frost_index.cpp
    test_exploit_frost_lagrange_duplicate.cpp
    test_exploit_frost_participant_zero.cpp
    test_exploit_frost_signing.cpp
    test_exploit_frost_threshold_degenerate.cpp
    test_exploit_glv_endomorphism.cpp
    test_exploit_glv_kat.cpp
    test_exploit_gpu_cpu_divergence.cpp
    test_exploit_gpu_host_api_shape.cpp
    test_exploit_hedged_nonce_bias.cpp
    test_exploit_hkdf_kat.cpp
    test_exploit_hkdf_security.cpp
    test_exploit_invalid_curve_twist.cpp
    test_exploit_keccak256_kat.cpp
    test_exploit_multiscalar.cpp
    test_exploit_musig2.cpp
    test_exploit_musig2_key_agg.cpp
    test_exploit_musig2_nonce_reuse.cpp
    test_exploit_musig2_ordering.cpp
    test_exploit_musig2_transcript_fork.cpp
    test_exploit_pedersen_adversarial.cpp
    test_exploit_pedersen_homomorphism.cpp
    test_exploit_pedersen_switch_misuse.cpp
    test_exploit_pippenger_msm.cpp
    test_exploit_point_group_law.cpp
    test_exploit_point_serialization.cpp
    test_exploit_private_key.cpp
    test_exploit_recovery_extended.cpp
    test_exploit_ripemd160_kat.cpp
    test_exploit_scalar_group_order.cpp
    test_exploit_scalar_invariants.cpp
    test_exploit_scalar_systematic.cpp
    test_exploit_schnorr_bip340_kat.cpp
    test_exploit_schnorr_edge_cases.cpp
    test_exploit_schnorr_forgery_vectors.cpp
    test_exploit_schnorr_xonly_parity_confusion.cpp
    test_exploit_seckey_tweak_cancel.cpp
    test_exploit_segwit_encoding.cpp
    test_exploit_selftest_api.cpp
    test_exploit_sha256_kat.cpp
    test_exploit_sha512_kat.cpp
    test_exploit_sha_kat.cpp
    test_exploit_silent_payment_confusion.cpp
    test_exploit_taproot_merkle_path_alias.cpp
    test_exploit_taproot_scripts.cpp
    test_exploit_taproot_tweak.cpp
    test_exploit_wallet_api.cpp
    test_exploit_wallet_cross_domain_replay.cpp
    test_exploit_zk_adversarial.cpp
    test_exploit_zk_proofs.cpp
    test_infinity_edge_cases.cpp
    test_exploit_seckey_arith.cpp
    test_exploit_ctx_clone.cpp
    # -- Feature Exploit PoC tests (P2SH, BIP-85, BIP-340 var, BIP-322, GCS, PSBT, Desc)
    test_exploit_p2sh_address_confusion.cpp
    test_exploit_bip85_path_collision.cpp
    test_exploit_schnorr_msg_length_confusion.cpp
    test_exploit_bip322_type_confusion.cpp
    test_exploit_gcs_false_positive.cpp
    test_exploit_psbt_input_confusion.cpp
    test_exploit_descriptor_injection.cpp
)
# Conditionally add 5x52 field test (requires __uint128_t; skip on MSVC)
if(NOT (MSVC AND NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang"))
    target_sources(unified_audit_runner PRIVATE ${CPU_TESTS_DIR}/test_field_52.cpp)
endif()
# Conditionally add Ethereum layer tests
if(SECP256K1_BUILD_ETHEREUM)
    target_sources(unified_audit_runner PRIVATE ${CPU_TESTS_DIR}/test_ethereum.cpp)
endif()
target_link_libraries(unified_audit_runner PRIVATE fastsecp256k1)
target_include_directories(unified_audit_runner PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_BINARY_DIR}/../include   # for generated version.hpp
    ${CMAKE_BINARY_DIR}/include              # fallback for version.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../gpu/include  # for gpu_backend.hpp (gpu_registry.cpp)
)
# Inject git hash at compile time (short hash, 8 chars)
execute_process(
    COMMAND git -C "${CMAKE_CURRENT_SOURCE_DIR}" rev-parse --short=8 HEAD
    OUTPUT_VARIABLE _git_hash
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
    RESULT_VARIABLE _git_rc
)
if(_git_rc)
    set(_git_hash "unknown")
endif()
target_compile_definitions(unified_audit_runner PRIVATE
    UNIFIED_AUDIT_RUNNER
    DUDECT_SMOKE
    UFSECP_BUILDING
    GIT_HASH="${_git_hash}"
)
if(MSVC OR (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND WIN32))
    target_link_options(unified_audit_runner PRIVATE "LINKER:/STACK:8388608")
endif()
add_test(NAME unified_audit COMMAND unified_audit_runner)
set_tests_properties(unified_audit PROPERTIES TIMEOUT 1200)

# ===========================================================================
target_link_libraries(unified_audit_runner PRIVATE Threads::Threads)
# Full Audit Orchestrator (custom target)
# ===========================================================================
# "cmake --build <dir> --target run_full_audit" runs the orchestrator script.
# On Windows, runs the PowerShell version; on Linux/macOS, runs the bash version.
if(WIN32)
    add_custom_target(run_full_audit
        COMMAND ${CMAKE_COMMAND} -E echo "Running full audit orchestrator (A-M)..."
        COMMAND pwsh -NoProfile -File "${CMAKE_CURRENT_SOURCE_DIR}/run_full_audit.ps1"
            -BuildDir "${CMAKE_BINARY_DIR}"
            -OutputDir "${CMAKE_BINARY_DIR}/audit-output"
            -SkipBuild
        DEPENDS unified_audit_runner
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/.."
        COMMENT "Full audit orchestrator (categories A-M)"
        VERBATIM
    )
else()
    add_custom_target(run_full_audit
        COMMAND ${CMAKE_COMMAND} -E echo "Running full audit orchestrator (A-M)..."
        COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/run_full_audit.sh"
        DEPENDS unified_audit_runner
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/.."
        COMMENT "Full audit orchestrator (categories A-M)"
        VERBATIM
    )
    # Pass build dir via environment
    set_property(TARGET run_full_audit APPEND PROPERTY
        ENVIRONMENT "BUILD_DIR=${CMAKE_BINARY_DIR}" "SKIP_BUILD=1"
    )
endif()

# -- CTest labels for grouping ---------------------------------------------
# Label all audit tests so they can be run as a group:
#   ctest --test-dir <build> -L audit
set_tests_properties(
    ct_sidechannel ct_sidechannel_smoke
    differential ct_equivalence
    fault_injection debug_invariants
    fiat_crypto_vectors carry_propagation
    cross_platform_kat abi_gate
    audit_fuzz diag_scalar_mul
    wycheproof_ecdsa wycheproof_ecdh
    batch_randomness
    ct_verif_formal fiat_crypto_linkage
    adversarial_protocol
    secp256k1_spec audit_invariants c_abi_negative
    audit_secure_erase audit_ct_namespace kat_all_operations nonce_uniqueness
    parse_strictness
    unified_audit
    exploit_ct_recov exploit_bip32_depth exploit_musig2_ordering
    exploit_frost_index exploit_frost_participant_zero exploit_frost_threshold_degenerate
    exploit_frost_lagrange_duplicate exploit_adaptor_parity exploit_scalar_invariants
    exploit_ecdh_degenerate exploit_schnorr_edge_cases exploit_ecies_auth
    exploit_batch_soundness exploit_bip32_path_overflow exploit_musig2_nonce_reuse
    exploit_point_group_law
    exploit_ecdsa_recovery exploit_ecdsa_edge_cases exploit_taproot_tweak
    exploit_chacha20_nonce_reuse exploit_zk_proofs
    exploit_frost_signing exploit_bip39_entropy exploit_ellswift
    exploit_address_generation exploit_bip324_session exploit_multiscalar
    exploit_pedersen_homomorphism exploit_hkdf_security exploit_batch_schnorr
    exploit_segwit_encoding exploit_private_key exploit_bip143_sighash
    exploit_ecies_roundtrip exploit_ecdh_variants exploit_glv_endomorphism
    exploit_sha_kat exploit_bip144_serialization
    exploit_musig2_key_agg exploit_bip32_derivation
    exploit_ct_systematic exploit_backend_divergence
    exploit_schnorr_bip340_kat exploit_ripemd160_kat exploit_chacha20_kat
    exploit_point_serialization exploit_ecdsa_rfc6979_kat
    exploit_frost_byzantine exploit_scalar_systematic
    exploit_ecdsa_malleability exploit_glv_kat exploit_aead_integrity
    exploit_batch_schnorr_forge exploit_bip32_ckd_hardened
    exploit_zk_adversarial exploit_pedersen_adversarial
    exploit_silent_payment_confusion
    exploit_ct_fast_equivalence exploit_pedersen_switch_misuse
    PROPERTIES LABELS "audit"
)
if(SECP256K1_BUILD_PROTOCOL_TESTS)
    set_tests_properties(
        musig2_frost musig2_frost_advanced frost_kat musig2_bip327_vectors
        PROPERTIES LABELS "audit"
    )
endif()

message(STATUS "  Audit test plan: audit/AUDIT_TEST_PLAN.md")
message(STATUS "  Full audit script (Windows): audit/run_full_audit.ps1")
message(STATUS "  Full audit script (Linux):   audit/run_full_audit.sh")
