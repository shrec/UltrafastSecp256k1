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
set(CPU_TESTS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/tests)

# -- Helper: common link + stack options ------------------------------------
macro(audit_target_defaults target_name)
    target_link_libraries(${target_name} PRIVATE fastsecp256k1)
    # ufsecp_static provides the ufsecp_* C ABI symbols (ufsecp_pubkey_create etc.)
    # It is built when SECP256K1_BUILD_CABI=ON (default). Link it when available.
    if(TARGET ufsecp_static)
        target_link_libraries(${target_name} PRIVATE ufsecp_static)
    elseif(TARGET ufsecp_shared)
        target_link_libraries(${target_name} PRIVATE ufsecp_shared)
    endif()
    target_include_directories(${target_name} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../include/ufsecp)
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

# -- Zeroization audit ----------------------------------------------------
add_executable(test_zeroization test_zeroization.cpp)
audit_target_defaults(test_zeroization)
target_compile_definitions(test_zeroization PRIVATE STANDALONE_TEST)
add_test(NAME zeroization COMMAND test_zeroization)
set_tests_properties(zeroization PROPERTIES TIMEOUT 900)

# -- ZK/Pedersen/Adaptor fault injection -----------------------------------
add_executable(test_fault_zk_adaptor test_fault_zk_adaptor.cpp)
audit_target_defaults(test_fault_zk_adaptor)
target_compile_definitions(test_fault_zk_adaptor PRIVATE STANDALONE_TEST)
# macOS only: Apple ld (ThinLTO) crashes with SIGABRT when linking this target
# due to the very large ZK/adaptor template expansions (zk.hpp + adaptor.hpp).
# The corrected mul_wide (j=0) changes the LTO code-size profile enough to
# trigger Apple ld's internal OOM limit. Scope the fix to APPLE only — setting
# INTERPROCEDURAL_OPTIMIZATION OFF globally causes GCC on Linux to mix LTO and
# non-LTO objects in the same project, which emits SIGILL from the field-
# arithmetic path in other (unrelated) targets built under the same CMake run.
if(APPLE)
    set_target_properties(test_fault_zk_adaptor PROPERTIES INTERPROCEDURAL_OPTIMIZATION OFF)
endif()
add_test(NAME fault_zk_adaptor COMMAND test_fault_zk_adaptor)
set_tests_properties(fault_zk_adaptor PROPERTIES TIMEOUT 1200)

# -- Secret lifecycle audit ------------------------------------------------
add_executable(test_secret_lifecycle test_secret_lifecycle.cpp)
audit_target_defaults(test_secret_lifecycle)
target_compile_definitions(test_secret_lifecycle PRIVATE STANDALONE_TEST)
add_test(NAME secret_lifecycle COMMAND test_secret_lifecycle)
set_tests_properties(secret_lifecycle PROPERTIES TIMEOUT 1200)

# -- Field / scalar arithmetic edge cases ---------------------------------
add_executable(test_field_scalar_edge test_field_scalar_edge.cpp)
audit_target_defaults(test_field_scalar_edge)
target_compile_definitions(test_field_scalar_edge PRIVATE STANDALONE_TEST)
add_test(NAME field_scalar_edge COMMAND test_field_scalar_edge)
set_tests_properties(field_scalar_edge PROPERTIES TIMEOUT 1200)

# -- Point group law correctness -------------------------------------------
add_executable(test_point_group_law test_point_group_law.cpp)
audit_target_defaults(test_point_group_law)
target_compile_definitions(test_point_group_law PRIVATE STANDALONE_TEST)
add_test(NAME point_group_law COMMAND test_point_group_law)
set_tests_properties(point_group_law PROPERTIES TIMEOUT 1200)

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

    # -- ECDSA SNARK witness batch (eprint 2025/695) --
    add_executable(test_gpu_ecdsa_snark_witness test_gpu_ecdsa_snark_witness.cpp)
    target_link_libraries(test_gpu_ecdsa_snark_witness PRIVATE ${UFSECP_GPU_TEST_LIB})
    target_include_directories(test_gpu_ecdsa_snark_witness PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
        ${CMAKE_BINARY_DIR}/include/ufsecp
    )
    target_compile_definitions(test_gpu_ecdsa_snark_witness PRIVATE STANDALONE_TEST)
    add_test(NAME gpu_ecdsa_snark_witness COMMAND test_gpu_ecdsa_snark_witness)
    set_tests_properties(gpu_ecdsa_snark_witness PROPERTIES
        TIMEOUT 300 LABELS "audit;zk;snark;foreign-field;gpu")

    # -- BIP-352 Silent Payment GPU batch scan --
    add_executable(test_gpu_bip352_scan test_gpu_bip352_scan.cpp)
    target_link_libraries(test_gpu_bip352_scan PRIVATE ${UFSECP_GPU_TEST_LIB})
    target_include_directories(test_gpu_bip352_scan PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
        ${CMAKE_BINARY_DIR}/include/ufsecp
    )
    target_compile_definitions(test_gpu_bip352_scan PRIVATE STANDALONE_TEST)
    add_test(NAME gpu_bip352_scan COMMAND test_gpu_bip352_scan)
    set_tests_properties(gpu_bip352_scan PROPERTIES
        TIMEOUT 300 LABELS "audit;bip352;silent-payment;gpu")

    # Disable LTO for GPU-linked targets (nvcc LTO version differs from host compiler)
    if(SECP256K1_BUILD_CUDA)
        set_target_properties(
            test_gpu_abi_gate test_gpu_ops_equivalence
            test_gpu_host_api_negative test_gpu_backend_matrix
            test_gpu_ecdsa_snark_witness test_gpu_bip352_scan
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
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/src/ufsecp_impl.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/src/ufsecp_gpu_impl.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/gpu/src/gpu_registry.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/gpu/src/gpu_backend_fallback.cpp)
audit_target_defaults(test_c_abi_negative_standalone)
target_include_directories(test_c_abi_negative_standalone PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_BINARY_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/gpu/include
)
target_compile_definitions(test_c_abi_negative_standalone PRIVATE UFSECP_BUILDING)
add_test(NAME c_abi_negative COMMAND test_c_abi_negative_standalone)
set_tests_properties(c_abi_negative PROPERTIES TIMEOUT 120)

# -- Secure memory erasure verification -----------------------------------
add_executable(test_audit_secure_erase_standalone
    audit_secure_erase.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/src/ufsecp_impl.cpp)
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
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/src/ufsecp_impl.cpp)
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
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/src/ufsecp_impl.cpp)
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
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/src/ufsecp_impl.cpp)
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
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/src/ufsecp_impl.cpp)
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
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/src/ufsecp_impl.cpp)
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
set_tests_properties(exploit_frost_index PROPERTIES TIMEOUT 600 LABELS "audit;exploit;frost")

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
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/src/ufsecp_impl.cpp)
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

# -- PoC: FROST participant-set malleability (Wagner attack, 2024 research)
add_executable(test_exploit_frost_participant_set_malleability_standalone
    test_exploit_frost_participant_set_malleability.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/src/ufsecp_impl.cpp)
audit_target_defaults(test_exploit_frost_participant_set_malleability_standalone)
target_include_directories(test_exploit_frost_participant_set_malleability_standalone PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_BINARY_DIR}/include
)
target_compile_definitions(test_exploit_frost_participant_set_malleability_standalone PRIVATE STANDALONE_TEST UFSECP_BUILDING)
add_test(NAME exploit_frost_participant_set_malleability COMMAND test_exploit_frost_participant_set_malleability_standalone)
set_tests_properties(exploit_frost_participant_set_malleability PROPERTIES TIMEOUT 120 LABELS "audit;exploit;frost")

# -- PoC: FROST/MuSig2 constant-time nonce audit (C1 gap)
add_executable(test_exploit_frost_ct_nonce_standalone
    test_exploit_frost_ct_nonce.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/src/ufsecp_impl.cpp)
audit_target_defaults(test_exploit_frost_ct_nonce_standalone)
target_include_directories(test_exploit_frost_ct_nonce_standalone PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_BINARY_DIR}/include
)
target_compile_definitions(test_exploit_frost_ct_nonce_standalone PRIVATE STANDALONE_TEST UFSECP_BUILDING)
add_test(NAME exploit_frost_ct_nonce COMMAND test_exploit_frost_ct_nonce_standalone)
set_tests_properties(exploit_frost_ct_nonce PROPERTIES TIMEOUT 120 LABELS "audit;exploit;frost;ct")

# -- PoC: Binding-layer invalid-curve attack (CVE-2024-48930 class)
add_executable(test_exploit_binding_invalid_curve_standalone
    test_exploit_binding_invalid_curve.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/src/ufsecp_impl.cpp)
audit_target_defaults(test_exploit_binding_invalid_curve_standalone)
target_include_directories(test_exploit_binding_invalid_curve_standalone PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_BINARY_DIR}/include
)
target_compile_definitions(test_exploit_binding_invalid_curve_standalone PRIVATE STANDALONE_TEST UFSECP_BUILDING)
add_test(NAME exploit_binding_invalid_curve COMMAND test_exploit_binding_invalid_curve_standalone)
set_tests_properties(exploit_binding_invalid_curve PROPERTIES TIMEOUT 120 LABELS "audit;exploit;binding")

# -- PoC: MuSig2 parallel session cross-signing attack
add_executable(test_exploit_musig2_parallel_session_cross_standalone
    test_exploit_musig2_parallel_session_cross.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/src/ufsecp_impl.cpp)
audit_target_defaults(test_exploit_musig2_parallel_session_cross_standalone)
target_include_directories(test_exploit_musig2_parallel_session_cross_standalone PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_BINARY_DIR}/include
)
target_compile_definitions(test_exploit_musig2_parallel_session_cross_standalone PRIVATE STANDALONE_TEST UFSECP_BUILDING)
add_test(NAME exploit_musig2_parallel_session_cross COMMAND test_exploit_musig2_parallel_session_cross_standalone)
set_tests_properties(exploit_musig2_parallel_session_cross PROPERTIES TIMEOUT 120 LABELS "audit;exploit;musig2")

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

add_executable(test_exploit_eip712_kat_standalone test_exploit_eip712_kat.cpp)
audit_target_defaults(test_exploit_eip712_kat_standalone)
target_compile_definitions(test_exploit_eip712_kat_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_eip712_kat COMMAND test_exploit_eip712_kat_standalone)
set_tests_properties(exploit_eip712_kat PROPERTIES TIMEOUT 60 LABELS "audit;exploit;ethereum;eip712")

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

add_executable(test_exploit_bip39_nfkd_standalone test_exploit_bip39_nfkd.cpp)
audit_target_defaults(test_exploit_bip39_nfkd_standalone)
target_compile_definitions(test_exploit_bip39_nfkd_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_bip39_nfkd COMMAND test_exploit_bip39_nfkd_standalone)
set_tests_properties(exploit_bip39_nfkd PROPERTIES TIMEOUT 60 LABELS "audit;exploit;bip39;nfkd")

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

# -- bech32_decode SIZE_MAX underflow regression (CVE-class: OOB read via size_t wrap) ---
add_executable(test_exploit_bech32_underflow_standalone test_exploit_bech32_underflow.cpp)
audit_target_defaults(test_exploit_bech32_underflow_standalone)
target_compile_definitions(test_exploit_bech32_underflow_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_bech32_underflow COMMAND test_exploit_bech32_underflow_standalone)
set_tests_properties(exploit_bech32_underflow PROPERTIES TIMEOUT 30 LABELS "audit;exploit;address;regression")

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

# -- KAT Corpus: runtime JSON vector loader -----------------------------------
# Loads audit/kat_vectors/*.json at runtime. Add vectors by editing JSON.
# No recompilation needed for new test vectors.
add_executable(test_exploit_kat_corpus_standalone test_exploit_kat_corpus.cpp)
audit_target_defaults(test_exploit_kat_corpus_standalone)
target_compile_definitions(test_exploit_kat_corpus_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_kat_corpus COMMAND test_exploit_kat_corpus_standalone)
# set_tests_properties must come AFTER add_test
set_tests_properties(exploit_kat_corpus PROPERTIES
    TIMEOUT 30
    LABELS "audit;exploit;kat;corpus;chacha20;sha256;rfc8439;fips180"
    ENVIRONMENT "KAT_VECTORS_DIR=${CMAKE_CURRENT_SOURCE_DIR}/kat_vectors")

# -- Primitive-layer KAT (bottom-up: bitops → quarter-round → block → AEAD) --
# Every stream/hash/AEAD tested at the LOWEST layer against RFC/FIPS vectors.
# Round-trip tests are insufficient — both sides share the bug (issue #256).
add_executable(test_exploit_primitive_kat_standalone test_exploit_primitive_kat.cpp)
audit_target_defaults(test_exploit_primitive_kat_standalone)
target_compile_definitions(test_exploit_primitive_kat_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_primitive_kat COMMAND test_exploit_primitive_kat_standalone)
set_tests_properties(exploit_primitive_kat PROPERTIES TIMEOUT 60 LABELS "audit;exploit;kat;chacha20;sha256;rfc8439;fips180")

# -- EC point serialization (G coords, 2*G, infinity, parity, associativity) -
add_executable(test_exploit_point_serialization_standalone test_exploit_point_serialization.cpp)
audit_target_defaults(test_exploit_point_serialization_standalone)
target_compile_definitions(test_exploit_point_serialization_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_point_serialization COMMAND test_exploit_point_serialization_standalone)
set_tests_properties(exploit_point_serialization PROPERTIES TIMEOUT 60 LABELS "audit;exploit;ec;point")

# -- Point-at-Infinity edge cases (INF-1..INF-28: cancellation, serial, tweak) -
add_executable(test_infinity_edge_cases_standalone
    test_infinity_edge_cases.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/src/ufsecp_impl.cpp)
audit_target_defaults(test_infinity_edge_cases_standalone)
target_include_directories(test_infinity_edge_cases_standalone PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_BINARY_DIR}/include
)
target_compile_definitions(test_infinity_edge_cases_standalone PRIVATE STANDALONE_TEST UFSECP_BUILDING)
add_test(NAME infinity_edge_cases COMMAND test_infinity_edge_cases_standalone)
set_tests_properties(infinity_edge_cases PROPERTIES TIMEOUT 900 LABELS "audit;exploit;ec;point;infinity")

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

# -- MuSig2/FROST Byzantine 3-of-5 multi-party adversarial tests -------------
add_executable(test_exploit_musig2_byzantine_multiparty_standalone test_exploit_musig2_byzantine_multiparty.cpp)
audit_target_defaults(test_exploit_musig2_byzantine_multiparty_standalone)
target_compile_definitions(test_exploit_musig2_byzantine_multiparty_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_musig2_byzantine_multiparty COMMAND test_exploit_musig2_byzantine_multiparty_standalone)
set_tests_properties(exploit_musig2_byzantine_multiparty PROPERTIES TIMEOUT 120 LABELS "audit;exploit;musig2;frost;byzantine;multiparty")

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

# -- PoC: ECDSA batch verify low-S enforcement (regression, 2026-04-27)
add_executable(test_exploit_batch_verify_low_s_standalone
    test_exploit_batch_verify_low_s.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/src/ufsecp_impl.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/src/ufsecp_gpu_impl.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/gpu/src/gpu_registry.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/gpu/src/gpu_backend_fallback.cpp)
audit_target_defaults(test_exploit_batch_verify_low_s_standalone)
target_include_directories(test_exploit_batch_verify_low_s_standalone PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_BINARY_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/gpu/include
)
target_compile_definitions(test_exploit_batch_verify_low_s_standalone PRIVATE STANDALONE_TEST UFSECP_BUILDING)
add_test(NAME exploit_batch_verify_low_s COMMAND test_exploit_batch_verify_low_s_standalone)
set_tests_properties(exploit_batch_verify_low_s PROPERTIES TIMEOUT 60 LABELS "audit;exploit;ecdsa;batch;malleability")

# -- PoC: ABI return-value coverage — binding regression guard (2026-04-27)
add_executable(test_exploit_binding_retval_standalone
    test_exploit_binding_retval.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/src/ufsecp_impl.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/src/ufsecp_gpu_impl.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/gpu/src/gpu_registry.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/gpu/src/gpu_backend_fallback.cpp)
audit_target_defaults(test_exploit_binding_retval_standalone)
target_include_directories(test_exploit_binding_retval_standalone PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_BINARY_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/gpu/include
)
target_compile_definitions(test_exploit_binding_retval_standalone PRIVATE STANDALONE_TEST UFSECP_BUILDING)
add_test(NAME exploit_binding_retval COMMAND test_exploit_binding_retval_standalone)
set_tests_properties(exploit_binding_retval PROPERTIES TIMEOUT 60 LABELS "audit;exploit;binding;retval")

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

# -- PoC: fe_set_b32_limit uninitialized overflow flag regression (PR #1839) --
add_executable(test_exploit_fe_set_b32_limit_uninit_standalone
    test_exploit_fe_set_b32_limit_uninit.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/src/ufsecp_impl.cpp)
audit_target_defaults(test_exploit_fe_set_b32_limit_uninit_standalone)
target_include_directories(test_exploit_fe_set_b32_limit_uninit_standalone PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_BINARY_DIR}/include
)
target_compile_definitions(test_exploit_fe_set_b32_limit_uninit_standalone PRIVATE
    UFSECP_BUILDING STANDALONE_TEST)
add_test(NAME exploit_fe_set_b32_limit_uninit
    COMMAND test_exploit_fe_set_b32_limit_uninit_standalone)
set_tests_properties(exploit_fe_set_b32_limit_uninit PROPERTIES
    TIMEOUT 60 LABELS "audit;exploit;field;overflow")

# -- PoC: ZK new schemes — Bulletproof boundaries/batch/hiding/binding +
#         DLEQ anonymous credential (eprint 2024/2010) ----------------------
add_executable(test_exploit_zk_new_schemes_standalone
    test_exploit_zk_new_schemes.cpp)
audit_target_defaults(test_exploit_zk_new_schemes_standalone)
target_compile_definitions(test_exploit_zk_new_schemes_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_zk_new_schemes COMMAND test_exploit_zk_new_schemes_standalone)
set_tests_properties(exploit_zk_new_schemes PROPERTIES
    TIMEOUT 120 LABELS "audit;exploit;zk;bulletproof;range")

# -- PoC: Foreign-field arithmetic — PLONK/SNARK secp256k1 constraints (eprint 2025/695) --
add_executable(test_exploit_foreign_field_plonk_standalone
    test_exploit_foreign_field_plonk.cpp)
audit_target_defaults(test_exploit_foreign_field_plonk_standalone)
target_compile_definitions(test_exploit_foreign_field_plonk_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_foreign_field_plonk COMMAND test_exploit_foreign_field_plonk_standalone)
set_tests_properties(exploit_foreign_field_plonk PROPERTIES
    TIMEOUT 60 LABELS "audit;exploit;field;plonk;snark;foreign-field")

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
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/include
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
# Add ../include so Test 13 can resolve ufsecp.h (ufsecp_ecdsa_sig_from_der, BIP-66 leading-zero)
target_include_directories(test_exploit_der_parsing_differential_standalone PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_BINARY_DIR}/include)
target_compile_definitions(test_exploit_der_parsing_differential_standalone PRIVATE STANDALONE_TEST UFSECP_BUILDING)
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
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/src/ufsecp_impl.cpp)
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
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/src/ufsecp_impl.cpp)
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
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/src/ufsecp_impl.cpp)
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
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/src/ufsecp_impl.cpp)
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
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/src/ufsecp_impl.cpp)
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
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/src/ufsecp_impl.cpp)
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
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/src/ufsecp_impl.cpp)
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
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/src/ufsecp_impl.cpp)
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
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/src/ufsecp_impl.cpp)
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
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/src/ufsecp_impl.cpp)
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
add_executable(test_exploit_seckey_arith_standalone
    test_exploit_seckey_arith.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/src/ufsecp_impl.cpp)
audit_target_defaults(test_exploit_seckey_arith_standalone)
target_include_directories(test_exploit_seckey_arith_standalone PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_BINARY_DIR}/include
)
target_compile_definitions(test_exploit_seckey_arith_standalone PRIVATE STANDALONE_TEST UFSECP_BUILDING)
add_test(NAME exploit_seckey_arith COMMAND test_exploit_seckey_arith_standalone)
set_tests_properties(exploit_seckey_arith PROPERTIES TIMEOUT 60 LABELS "audit;exploit;seckey;tweak")

# -- PoC: Context clone isolation (CLN-1..CLN-6) ----------------------------
add_executable(test_exploit_ctx_clone_standalone
    test_exploit_ctx_clone.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/src/ufsecp_impl.cpp)
audit_target_defaults(test_exploit_ctx_clone_standalone)
target_include_directories(test_exploit_ctx_clone_standalone PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_BINARY_DIR}/include
)
target_link_libraries(test_exploit_ctx_clone_standalone PRIVATE Threads::Threads)
target_compile_definitions(test_exploit_ctx_clone_standalone PRIVATE STANDALONE_TEST UFSECP_BUILDING)
add_test(NAME exploit_ctx_clone COMMAND test_exploit_ctx_clone_standalone)
set_tests_properties(exploit_ctx_clone PROPERTIES TIMEOUT 60 LABELS "audit;exploit;context;clone")

# -- PoC: Public key arithmetic (PUB-1..PUB-9: negate, tweak_add/mul, combine) -
add_executable(test_exploit_pubkey_arith_standalone
    test_exploit_pubkey_arith.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/src/ufsecp_impl.cpp)
audit_target_defaults(test_exploit_pubkey_arith_standalone)
target_include_directories(test_exploit_pubkey_arith_standalone PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_BINARY_DIR}/include
)
target_compile_definitions(test_exploit_pubkey_arith_standalone PRIVATE STANDALONE_TEST UFSECP_BUILDING)
add_test(NAME exploit_pubkey_arith COMMAND test_exploit_pubkey_arith_standalone)
set_tests_properties(exploit_pubkey_arith PROPERTIES TIMEOUT 60 LABELS "audit;exploit;pubkey;tweak")

# -- PoC: Batch signing edge cases (BSG-1..BSG-9) ----------------------------
add_executable(test_exploit_batch_sign_standalone
    test_exploit_batch_sign.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/src/ufsecp_impl.cpp)
audit_target_defaults(test_exploit_batch_sign_standalone)
target_include_directories(test_exploit_batch_sign_standalone PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_BINARY_DIR}/include
)
target_compile_definitions(test_exploit_batch_sign_standalone PRIVATE STANDALONE_TEST UFSECP_BUILDING)
add_test(NAME exploit_batch_sign COMMAND test_exploit_batch_sign_standalone)
set_tests_properties(exploit_batch_sign PROPERTIES TIMEOUT 60 LABELS "audit;exploit;schnorr;batch")

# -- PoC: ECDSA nonce reuse key recovery (NRR-1..NRR-8) --------------------
add_executable(test_exploit_ecdsa_nonce_reuse_standalone
    test_exploit_ecdsa_nonce_reuse.cpp)
audit_target_defaults(test_exploit_ecdsa_nonce_reuse_standalone)
target_link_libraries(test_exploit_ecdsa_nonce_reuse_standalone PRIVATE fastsecp256k1)
target_compile_definitions(test_exploit_ecdsa_nonce_reuse_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_ecdsa_nonce_reuse COMMAND test_exploit_ecdsa_nonce_reuse_standalone)
set_tests_properties(exploit_ecdsa_nonce_reuse PROPERTIES TIMEOUT 60 LABELS "audit;exploit;ecdsa;nonce")

# -- PoC: Affine nonce relation key recovery (ANR-1..ANR-12) [ePrint 2025/705] --
add_executable(test_exploit_ecdsa_affine_nonce_relation_standalone
    test_exploit_ecdsa_affine_nonce_relation.cpp)
audit_target_defaults(test_exploit_ecdsa_affine_nonce_relation_standalone)
target_link_libraries(test_exploit_ecdsa_affine_nonce_relation_standalone PRIVATE fastsecp256k1)
target_compile_definitions(test_exploit_ecdsa_affine_nonce_relation_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_ecdsa_affine_nonce_relation COMMAND test_exploit_ecdsa_affine_nonce_relation_standalone)
set_tests_properties(exploit_ecdsa_affine_nonce_relation PROPERTIES TIMEOUT 120 LABELS "audit;exploit;ecdsa;nonce")

# -- PoC: Half-half nonce key recovery (HH-1..HH-10) [ePrint 2023/841] --
add_executable(test_exploit_ecdsa_half_half_nonce_standalone
    test_exploit_ecdsa_half_half_nonce.cpp)
audit_target_defaults(test_exploit_ecdsa_half_half_nonce_standalone)
target_link_libraries(test_exploit_ecdsa_half_half_nonce_standalone PRIVATE fastsecp256k1)
target_compile_definitions(test_exploit_ecdsa_half_half_nonce_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_ecdsa_half_half_nonce COMMAND test_exploit_ecdsa_half_half_nonce_standalone)
set_tests_properties(exploit_ecdsa_half_half_nonce PROPERTIES TIMEOUT 120 LABELS "audit;exploit;ecdsa;nonce")

# -- PoC: Nonce modular reduction bias (NMB-1..NMB-8) [CVE-2024-31497] --
add_executable(test_exploit_ecdsa_nonce_modular_bias_standalone
    test_exploit_ecdsa_nonce_modular_bias.cpp)
audit_target_defaults(test_exploit_ecdsa_nonce_modular_bias_standalone)
target_link_libraries(test_exploit_ecdsa_nonce_modular_bias_standalone PRIVATE fastsecp256k1)
target_compile_definitions(test_exploit_ecdsa_nonce_modular_bias_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_ecdsa_nonce_modular_bias COMMAND test_exploit_ecdsa_nonce_modular_bias_standalone)
set_tests_properties(exploit_ecdsa_nonce_modular_bias PROPERTIES TIMEOUT 120 LABELS "audit;exploit;ecdsa;nonce")

# -- PoC: Differential fault on RFC 6979 (DF-1..DF-8) [ePrint 2017/975] --
add_executable(test_exploit_ecdsa_differential_fault_standalone
    test_exploit_ecdsa_differential_fault.cpp)
audit_target_defaults(test_exploit_ecdsa_differential_fault_standalone)
target_link_libraries(test_exploit_ecdsa_differential_fault_standalone PRIVATE fastsecp256k1)
target_compile_definitions(test_exploit_ecdsa_differential_fault_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_ecdsa_differential_fault COMMAND test_exploit_ecdsa_differential_fault_standalone)
set_tests_properties(exploit_ecdsa_differential_fault PROPERTIES TIMEOUT 120 LABELS "audit;exploit;ecdsa;nonce")

# -- PoC: EUCLEAK non-CT modular inversion timing (EUC-1..EUC-12) [ePrint 2024/1380] --
add_executable(test_exploit_eucleak_inversion_timing_standalone
    test_exploit_eucleak_inversion_timing.cpp)
audit_target_defaults(test_exploit_eucleak_inversion_timing_standalone)
target_link_libraries(test_exploit_eucleak_inversion_timing_standalone PRIVATE fastsecp256k1)
target_compile_definitions(test_exploit_eucleak_inversion_timing_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_eucleak_inversion_timing COMMAND test_exploit_eucleak_inversion_timing_standalone)
set_tests_properties(exploit_eucleak_inversion_timing PROPERTIES TIMEOUT 120 LABELS "audit;exploit;sidechannel;inversion")

# -- PoC: Cross-key nonce reuse cascading compromise (CKN-1..CKN-10) [ePrint 2025/654] --
add_executable(test_exploit_ecdsa_cross_key_nonce_reuse_standalone
    test_exploit_ecdsa_cross_key_nonce_reuse.cpp)
audit_target_defaults(test_exploit_ecdsa_cross_key_nonce_reuse_standalone)
target_link_libraries(test_exploit_ecdsa_cross_key_nonce_reuse_standalone PRIVATE fastsecp256k1)
target_compile_definitions(test_exploit_ecdsa_cross_key_nonce_reuse_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_ecdsa_cross_key_nonce_reuse COMMAND test_exploit_ecdsa_cross_key_nonce_reuse_standalone)
set_tests_properties(exploit_ecdsa_cross_key_nonce_reuse PROPERTIES TIMEOUT 120 LABELS "audit;exploit;ecdsa;nonce")

# -- PoC: Fiat-Shamir hash order in BIP-340 (SHO-1..SHO-10) [ePrint 2025/1846] --
add_executable(test_exploit_schnorr_hash_order_standalone
    test_exploit_schnorr_hash_order.cpp)
audit_target_defaults(test_exploit_schnorr_hash_order_standalone)
target_link_libraries(test_exploit_schnorr_hash_order_standalone PRIVATE fastsecp256k1)
target_compile_definitions(test_exploit_schnorr_hash_order_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_schnorr_hash_order COMMAND test_exploit_schnorr_hash_order_standalone)
set_tests_properties(exploit_schnorr_hash_order PROPERTIES TIMEOUT 120 LABELS "audit;exploit;schnorr;hash")

# -- PoC: Cross-scheme pubkey consistency (CSP-1..CSP-8) -------------------
add_executable(test_exploit_cross_scheme_pubkey_standalone
    test_exploit_cross_scheme_pubkey.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/src/ufsecp_impl.cpp)
audit_target_defaults(test_exploit_cross_scheme_pubkey_standalone)
target_include_directories(test_exploit_cross_scheme_pubkey_standalone PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_BINARY_DIR}/include
)
target_compile_definitions(test_exploit_cross_scheme_pubkey_standalone PRIVATE STANDALONE_TEST UFSECP_BUILDING)
add_test(NAME exploit_cross_scheme_pubkey COMMAND test_exploit_cross_scheme_pubkey_standalone)
set_tests_properties(exploit_cross_scheme_pubkey PROPERTIES TIMEOUT 60 LABELS "audit;exploit;pubkey;ecdh;schnorr")

# -- PoC: WIF key format security (WIF-1..WIF-10) -------------------------
add_executable(test_exploit_wif_security_standalone
    test_exploit_wif_security.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/src/ufsecp_impl.cpp)
audit_target_defaults(test_exploit_wif_security_standalone)
target_include_directories(test_exploit_wif_security_standalone PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_BINARY_DIR}/include
)
target_compile_definitions(test_exploit_wif_security_standalone PRIVATE STANDALONE_TEST UFSECP_BUILDING)
add_test(NAME exploit_wif_security COMMAND test_exploit_wif_security_standalone)
set_tests_properties(exploit_wif_security PROPERTIES TIMEOUT 60 LABELS "audit;exploit;wif;key")

# -- PoC: Buffer / type-confusion attack vectors (CVE-2024-49364 class, A2 gap)
add_executable(test_exploit_buffer_type_confusion_standalone
    test_exploit_buffer_type_confusion.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/src/ufsecp_impl.cpp)
audit_target_defaults(test_exploit_buffer_type_confusion_standalone)
target_include_directories(test_exploit_buffer_type_confusion_standalone PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include ${CMAKE_BINARY_DIR}/include)
target_compile_definitions(test_exploit_buffer_type_confusion_standalone PRIVATE STANDALONE_TEST UFSECP_BUILDING)
add_test(NAME exploit_buffer_type_confusion COMMAND test_exploit_buffer_type_confusion_standalone)
set_tests_properties(exploit_buffer_type_confusion PROPERTIES TIMEOUT 60 LABELS "audit;exploit;cve")

# -- PoC: Differential correctness harness (Cryptofuzz++ style, B2 gap)
add_executable(test_exploit_differential_libsecp_standalone
    test_exploit_differential_libsecp.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/src/ufsecp_impl.cpp)
audit_target_defaults(test_exploit_differential_libsecp_standalone)
target_include_directories(test_exploit_differential_libsecp_standalone PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include ${CMAKE_BINARY_DIR}/include)
target_compile_definitions(test_exploit_differential_libsecp_standalone PRIVATE STANDALONE_TEST UFSECP_BUILDING)
add_test(NAME exploit_differential_libsecp COMMAND test_exploit_differential_libsecp_standalone)
set_tests_properties(exploit_differential_libsecp PROPERTIES TIMEOUT 120 LABELS "audit;exploit;differential")

# -- PoC: INTEROP differential vs OpenSSL libcrypto (advisory; G-9 / INTEROP §3)
add_executable(test_exploit_differential_openssl_standalone
    test_exploit_differential_openssl.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/src/ufsecp_impl.cpp)
audit_target_defaults(test_exploit_differential_openssl_standalone)
target_include_directories(test_exploit_differential_openssl_standalone PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include ${CMAKE_BINARY_DIR}/include)
target_compile_definitions(test_exploit_differential_openssl_standalone PRIVATE STANDALONE_TEST UFSECP_BUILDING)
if(NOT CMAKE_CROSSCOMPILING)
    find_package(OpenSSL QUIET COMPONENTS Crypto)
    if(OpenSSL_FOUND)
        target_link_libraries(test_exploit_differential_openssl_standalone PRIVATE OpenSSL::Crypto)
        target_compile_definitions(test_exploit_differential_openssl_standalone PRIVATE UFSECP_HAVE_OPENSSL=1)
    endif()
endif()
add_test(NAME exploit_differential_openssl COMMAND test_exploit_differential_openssl_standalone)
set_tests_properties(exploit_differential_openssl PROPERTIES TIMEOUT 120 LABELS "audit;exploit;differential;interop;advisory")

# -- PoC: BIP-352 scanning DoS prevention (A6 gap)
add_executable(test_exploit_bip352_scan_dos_standalone
    test_exploit_bip352_scan_dos.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/src/ufsecp_impl.cpp)
audit_target_defaults(test_exploit_bip352_scan_dos_standalone)
target_include_directories(test_exploit_bip352_scan_dos_standalone PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include ${CMAKE_BINARY_DIR}/include)
target_compile_definitions(test_exploit_bip352_scan_dos_standalone PRIVATE STANDALONE_TEST UFSECP_BUILDING)
add_test(NAME exploit_bip352_scan_dos COMMAND test_exploit_bip352_scan_dos_standalone)
set_tests_properties(exploit_bip352_scan_dos PROPERTIES TIMEOUT 120 LABELS "audit;exploit;bip352;dos")

# -- PoC: Taproot commitment adversarial (ePrint 2025/1307, A7 gap)
add_executable(test_exploit_taproot_commitment_adversarial_standalone
    test_exploit_taproot_commitment_adversarial.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/src/ufsecp_impl.cpp)
audit_target_defaults(test_exploit_taproot_commitment_adversarial_standalone)
target_include_directories(test_exploit_taproot_commitment_adversarial_standalone PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include ${CMAKE_BINARY_DIR}/include)
target_compile_definitions(test_exploit_taproot_commitment_adversarial_standalone PRIVATE STANDALONE_TEST UFSECP_BUILDING)
add_test(NAME exploit_taproot_commitment_adversarial COMMAND test_exploit_taproot_commitment_adversarial_standalone)
set_tests_properties(exploit_taproot_commitment_adversarial PROPERTIES TIMEOUT 60 LABELS "audit;exploit;taproot")

# -- PoC: BIP-352 parity confusion in ECDH (C2 gap)
add_executable(test_exploit_bip352_parity_confusion_standalone
    test_exploit_bip352_parity_confusion.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/src/ufsecp_impl.cpp)
audit_target_defaults(test_exploit_bip352_parity_confusion_standalone)
target_include_directories(test_exploit_bip352_parity_confusion_standalone PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include ${CMAKE_BINARY_DIR}/include)
target_compile_definitions(test_exploit_bip352_parity_confusion_standalone PRIVATE STANDALONE_TEST UFSECP_BUILDING)
add_test(NAME exploit_bip352_parity_confusion COMMAND test_exploit_bip352_parity_confusion_standalone)
set_tests_properties(exploit_bip352_parity_confusion PROPERTIES TIMEOUT 60 LABELS "audit;exploit;bip352;parity")

# -- BIP-352 Batch Scalar Mul + Input Aggregation Correctness (BSM/IAG)
add_executable(test_exploit_bip352_batch_correctness_standalone
    test_exploit_bip352_batch_correctness.cpp)
audit_target_defaults(test_exploit_bip352_batch_correctness_standalone)
target_compile_definitions(test_exploit_bip352_batch_correctness_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_bip352_batch_correctness COMMAND test_exploit_bip352_batch_correctness_standalone)
set_tests_properties(exploit_bip352_batch_correctness PROPERTIES TIMEOUT 120 LABELS "audit;exploit;bip352;batch;msm")

# -- BIP-352 Known-Answer Tests (create_output byte-exact + full round-trip)
add_executable(test_bip352_kat_standalone
    test_bip352_kat.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/src/ufsecp_impl.cpp)
audit_target_defaults(test_bip352_kat_standalone)
target_include_directories(test_bip352_kat_standalone PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include ${CMAKE_BINARY_DIR}/include)
target_compile_definitions(test_bip352_kat_standalone PRIVATE STANDALONE_TEST UFSECP_BUILDING)
add_test(NAME bip352_kat COMMAND test_bip352_kat_standalone)
set_tests_properties(bip352_kat PROPERTIES TIMEOUT 60 LABELS "audit;kat;bip352;silent-payment;regression")

# -- PoC: RFC 6979 nonce truncation bias (C5 gap)
add_executable(test_exploit_rfc6979_truncation_bias_standalone
    test_exploit_rfc6979_truncation_bias.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/src/ufsecp_impl.cpp)
audit_target_defaults(test_exploit_rfc6979_truncation_bias_standalone)
target_include_directories(test_exploit_rfc6979_truncation_bias_standalone PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include ${CMAKE_BINARY_DIR}/include)
target_compile_definitions(test_exploit_rfc6979_truncation_bias_standalone PRIVATE STANDALONE_TEST UFSECP_BUILDING)
add_test(NAME exploit_rfc6979_truncation_bias COMMAND test_exploit_rfc6979_truncation_bias_standalone)
set_tests_properties(exploit_rfc6979_truncation_bias PROPERTIES TIMEOUT 60 LABELS "audit;exploit;rfc6979;nonce")

# -- PoC: Binding adversarial API (D6 gap)
add_executable(test_exploit_binding_adversarial_api_standalone
    test_exploit_binding_adversarial_api.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/src/ufsecp_impl.cpp)
audit_target_defaults(test_exploit_binding_adversarial_api_standalone)
target_include_directories(test_exploit_binding_adversarial_api_standalone PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include ${CMAKE_BINARY_DIR}/include)
target_compile_definitions(test_exploit_binding_adversarial_api_standalone PRIVATE STANDALONE_TEST UFSECP_BUILDING)
add_test(NAME exploit_binding_adversarial_api COMMAND test_exploit_binding_adversarial_api_standalone)
set_tests_properties(exploit_binding_adversarial_api PROPERTIES TIMEOUT 60 LABELS "audit;exploit;binding;api")

# -- PoC: Quantum exposure surface audit (LOW)
add_executable(test_exploit_quantum_exposure_standalone
    test_exploit_quantum_exposure.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/src/ufsecp_impl.cpp)
audit_target_defaults(test_exploit_quantum_exposure_standalone)
target_include_directories(test_exploit_quantum_exposure_standalone PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include ${CMAKE_BINARY_DIR}/include)
target_compile_definitions(test_exploit_quantum_exposure_standalone PRIVATE STANDALONE_TEST UFSECP_BUILDING)
add_test(NAME exploit_quantum_exposure COMMAND test_exploit_quantum_exposure_standalone)
set_tests_properties(exploit_quantum_exposure PROPERTIES TIMEOUT 60 LABELS "audit;exploit;quantum")

# -- PoC: ECIES ephemeral key reuse detection (LOW)
add_executable(test_exploit_ecies_ephemeral_reuse_standalone
    test_exploit_ecies_ephemeral_reuse.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/src/ufsecp_impl.cpp)
audit_target_defaults(test_exploit_ecies_ephemeral_reuse_standalone)
target_include_directories(test_exploit_ecies_ephemeral_reuse_standalone PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include ${CMAKE_BINARY_DIR}/include)
target_compile_definitions(test_exploit_ecies_ephemeral_reuse_standalone PRIVATE STANDALONE_TEST UFSECP_BUILDING)
add_test(NAME exploit_ecies_ephemeral_reuse COMMAND test_exploit_ecies_ephemeral_reuse_standalone)
set_tests_properties(exploit_ecies_ephemeral_reuse PROPERTIES TIMEOUT 60 LABELS "audit;exploit;ecies")

# -- PoC: Address prefix collision resistance (LOW)
add_executable(test_exploit_address_prefix_collision_standalone
    test_exploit_address_prefix_collision.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/src/ufsecp_impl.cpp)
audit_target_defaults(test_exploit_address_prefix_collision_standalone)
target_include_directories(test_exploit_address_prefix_collision_standalone PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include ${CMAKE_BINARY_DIR}/include)
target_compile_definitions(test_exploit_address_prefix_collision_standalone PRIVATE STANDALONE_TEST UFSECP_BUILDING)
add_test(NAME exploit_address_prefix_collision COMMAND test_exploit_address_prefix_collision_standalone)
set_tests_properties(exploit_address_prefix_collision PROPERTIES TIMEOUT 60 LABELS "audit;exploit;address")

# -- PoC: ECDH ZVP/GLV static-key regression surface ------------------------
add_executable(test_exploit_ecdh_zvp_glv_static_standalone
    test_exploit_ecdh_zvp_glv_static.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/src/ufsecp_impl.cpp)
audit_target_defaults(test_exploit_ecdh_zvp_glv_static_standalone)
target_include_directories(test_exploit_ecdh_zvp_glv_static_standalone PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include ${CMAKE_BINARY_DIR}/include)
target_compile_definitions(test_exploit_ecdh_zvp_glv_static_standalone PRIVATE STANDALONE_TEST UFSECP_BUILDING)
add_test(NAME exploit_ecdh_zvp_glv_static COMMAND test_exploit_ecdh_zvp_glv_static_standalone)
set_tests_properties(exploit_ecdh_zvp_glv_static PROPERTIES TIMEOUT 60 LABELS "audit;exploit;ecdh;zvp;glv")

# -- PoC: FROST adaptive corruption misuse regression -----------------------
add_executable(test_exploit_frost_adaptive_corruption_standalone
    test_exploit_frost_adaptive_corruption.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/src/ufsecp_impl.cpp)
audit_target_defaults(test_exploit_frost_adaptive_corruption_standalone)
target_include_directories(test_exploit_frost_adaptive_corruption_standalone PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include ${CMAKE_BINARY_DIR}/include)
target_compile_definitions(test_exploit_frost_adaptive_corruption_standalone PRIVATE STANDALONE_TEST UFSECP_BUILDING)
add_test(NAME exploit_frost_adaptive_corruption COMMAND test_exploit_frost_adaptive_corruption_standalone)
set_tests_properties(exploit_frost_adaptive_corruption PROPERTIES TIMEOUT 60 LABELS "audit;exploit;frost;adaptive")

# -- PoC: Point::scalar_mul edge-case and correctness attack surface --------
add_executable(test_exploit_scalar_mul_standalone
    test_exploit_scalar_mul.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/src/ufsecp_impl.cpp)
audit_target_defaults(test_exploit_scalar_mul_standalone)
target_include_directories(test_exploit_scalar_mul_standalone PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include ${CMAKE_BINARY_DIR}/include)
target_compile_definitions(test_exploit_scalar_mul_standalone PRIVATE STANDALONE_TEST UFSECP_BUILDING)
add_test(NAME exploit_scalar_mul COMMAND test_exploit_scalar_mul_standalone)
set_tests_properties(exploit_scalar_mul PROPERTIES TIMEOUT 60 LABELS "audit;exploit;scalar_mul;glv")

# -- PoC: Deterministic ECDSA fault-injection regression --------------------
add_executable(test_exploit_ecdsa_fault_injection_standalone
    test_exploit_ecdsa_fault_injection.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/src/ufsecp_impl.cpp)
audit_target_defaults(test_exploit_ecdsa_fault_injection_standalone)
target_include_directories(test_exploit_ecdsa_fault_injection_standalone PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include ${CMAKE_BINARY_DIR}/include)
target_compile_definitions(test_exploit_ecdsa_fault_injection_standalone PRIVATE STANDALONE_TEST UFSECP_BUILDING)
add_test(NAME exploit_ecdsa_fault_injection COMMAND test_exploit_ecdsa_fault_injection_standalone)
set_tests_properties(exploit_ecdsa_fault_injection PROPERTIES TIMEOUT 60 LABELS "audit;exploit;ecdsa;fault")

# -- PoC: Cache side-channel amplification regression -----------------------
add_executable(test_exploit_cache_sidechannel_amplification_standalone
    test_exploit_cache_sidechannel_amplification.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/src/ufsecp_impl.cpp)
audit_target_defaults(test_exploit_cache_sidechannel_amplification_standalone)
target_include_directories(test_exploit_cache_sidechannel_amplification_standalone PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include ${CMAKE_BINARY_DIR}/include)
target_compile_definitions(test_exploit_cache_sidechannel_amplification_standalone PRIVATE STANDALONE_TEST UFSECP_BUILDING)
add_test(NAME exploit_cache_sidechannel_amplification COMMAND test_exploit_cache_sidechannel_amplification_standalone)
set_tests_properties(exploit_cache_sidechannel_amplification PROPERTIES TIMEOUT 60 LABELS "audit;exploit;ecdsa;sidechannel")

# -- PoC: LadderLeak-style sub-bit nonce leakage regression -----------------
add_executable(test_exploit_ladderleak_subbit_nonce_standalone
    test_exploit_ladderleak_subbit_nonce.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/src/ufsecp_impl.cpp)
audit_target_defaults(test_exploit_ladderleak_subbit_nonce_standalone)
target_include_directories(test_exploit_ladderleak_subbit_nonce_standalone PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include ${CMAKE_BINARY_DIR}/include)
target_compile_definitions(test_exploit_ladderleak_subbit_nonce_standalone PRIVATE STANDALONE_TEST UFSECP_BUILDING)
add_test(NAME exploit_ladderleak_subbit_nonce COMMAND test_exploit_ladderleak_subbit_nonce_standalone)
set_tests_properties(exploit_ladderleak_subbit_nonce PROPERTIES TIMEOUT 60 LABELS "audit;exploit;ecdsa;nonce;ladderleak")

# -- PoC: Minerva-style noisy HNP leakage regression ------------------------
add_executable(test_exploit_minerva_noisy_hnp_standalone
    test_exploit_minerva_noisy_hnp.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/src/ufsecp_impl.cpp)
audit_target_defaults(test_exploit_minerva_noisy_hnp_standalone)
target_include_directories(test_exploit_minerva_noisy_hnp_standalone PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include ${CMAKE_BINARY_DIR}/include)
target_compile_definitions(test_exploit_minerva_noisy_hnp_standalone PRIVATE STANDALONE_TEST UFSECP_BUILDING)
add_test(NAME exploit_minerva_noisy_hnp COMMAND test_exploit_minerva_noisy_hnp_standalone)
set_tests_properties(exploit_minerva_noisy_hnp PROPERTIES TIMEOUT 60 LABELS "audit;exploit;ecdsa;nonce;minerva")

# -- PoC: Hertzbleed-style DVFS timing regression ---------------------------
add_executable(test_exploit_hertzbleed_dvfs_timing_standalone
    test_exploit_hertzbleed_dvfs_timing.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/src/ufsecp_impl.cpp)
audit_target_defaults(test_exploit_hertzbleed_dvfs_timing_standalone)
target_include_directories(test_exploit_hertzbleed_dvfs_timing_standalone PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include ${CMAKE_BINARY_DIR}/include)
target_compile_definitions(test_exploit_hertzbleed_dvfs_timing_standalone PRIVATE STANDALONE_TEST UFSECP_BUILDING)
add_test(NAME exploit_hertzbleed_dvfs_timing COMMAND test_exploit_hertzbleed_dvfs_timing_standalone)
set_tests_properties(exploit_hertzbleed_dvfs_timing PROPERTIES TIMEOUT 60 LABELS "audit;exploit;sidechannel;timing;dvfs")

# -- PoC: Biased nonce chain-scale scan regression --------------------------
add_executable(test_exploit_biased_nonce_chain_scan_standalone
    test_exploit_biased_nonce_chain_scan.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/src/ufsecp_impl.cpp)
audit_target_defaults(test_exploit_biased_nonce_chain_scan_standalone)
target_include_directories(test_exploit_biased_nonce_chain_scan_standalone PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include ${CMAKE_BINARY_DIR}/include)
target_compile_definitions(test_exploit_biased_nonce_chain_scan_standalone PRIVATE STANDALONE_TEST UFSECP_BUILDING)
add_test(NAME exploit_biased_nonce_chain_scan COMMAND test_exploit_biased_nonce_chain_scan_standalone)
set_tests_properties(exploit_biased_nonce_chain_scan PROPERTIES TIMEOUT 60 LABELS "audit;exploit;ecdsa;nonce;bias")

# -- PoC: KR-ECDSA/BUFF binding regression ----------------------------------
add_executable(test_exploit_kr_ecdsa_buff_binding_standalone
    test_exploit_kr_ecdsa_buff_binding.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/src/ufsecp_impl.cpp)
audit_target_defaults(test_exploit_kr_ecdsa_buff_binding_standalone)
target_include_directories(test_exploit_kr_ecdsa_buff_binding_standalone PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include ${CMAKE_BINARY_DIR}/include)
target_compile_definitions(test_exploit_kr_ecdsa_buff_binding_standalone PRIVATE STANDALONE_TEST UFSECP_BUILDING)
add_test(NAME exploit_kr_ecdsa_buff_binding COMMAND test_exploit_kr_ecdsa_buff_binding_standalone)
set_tests_properties(exploit_kr_ecdsa_buff_binding PROPERTIES TIMEOUT 60 LABELS "audit;exploit;ecdsa;binding")

# -- PoC: ellswift_xdh overflow scalar rejection — upstream PR #1821 ---------
add_executable(test_exploit_ellswift_xdh_overflow_standalone
    test_exploit_ellswift_xdh_overflow.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/src/ufsecp_impl.cpp)
audit_target_defaults(test_exploit_ellswift_xdh_overflow_standalone)
target_include_directories(test_exploit_ellswift_xdh_overflow_standalone PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include ${CMAKE_BINARY_DIR}/include)
target_compile_definitions(test_exploit_ellswift_xdh_overflow_standalone PRIVATE
    STANDALONE_TEST UFSECP_BUILDING SECP256K1_BIP324)
add_test(NAME exploit_ellswift_xdh_overflow COMMAND test_exploit_ellswift_xdh_overflow_standalone)
set_tests_properties(exploit_ellswift_xdh_overflow PROPERTIES TIMEOUT 60 LABELS "audit;exploit;ellswift;ecdh;overflow")

# -- PoC: BUFF / KR-ECDSA weak non-resignability (eprint 2024/2018) ----------
add_executable(test_exploit_buff_kr_ecdsa_standalone
    test_exploit_buff_kr_ecdsa.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/src/ufsecp_impl.cpp)
audit_target_defaults(test_exploit_buff_kr_ecdsa_standalone)
target_include_directories(test_exploit_buff_kr_ecdsa_standalone PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include ${CMAKE_BINARY_DIR}/include)
target_compile_definitions(test_exploit_buff_kr_ecdsa_standalone PRIVATE
    STANDALONE_TEST UFSECP_BUILDING SECP256K1_BUILD_ETHEREUM)
add_test(NAME exploit_buff_kr_ecdsa COMMAND test_exploit_buff_kr_ecdsa_standalone)
set_tests_properties(exploit_buff_kr_ecdsa PROPERTIES TIMEOUT 120 LABELS "audit;exploit;ecdsa;buff;ethereum")

# -- PoC: CVE-2024-23342 Minerva bit-length timing regression ----------------
add_executable(test_exploit_minerva_cve_2024_23342_standalone
    test_exploit_minerva_cve_2024_23342.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/src/ufsecp_impl.cpp)
audit_target_defaults(test_exploit_minerva_cve_2024_23342_standalone)
target_include_directories(test_exploit_minerva_cve_2024_23342_standalone PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include ${CMAKE_BINARY_DIR}/include)
target_compile_definitions(test_exploit_minerva_cve_2024_23342_standalone PRIVATE STANDALONE_TEST UFSECP_BUILDING)
add_test(NAME exploit_minerva_cve_2024_23342 COMMAND test_exploit_minerva_cve_2024_23342_standalone)
set_tests_properties(exploit_minerva_cve_2024_23342 PROPERTIES TIMEOUT 120 LABELS "audit;exploit;ecdsa;nonce;minerva;cve")

# -- PoC: RFC6979 + Minerva amplified timing regression ----------------------
add_executable(test_exploit_rfc6979_minerva_amplified_standalone
    test_exploit_rfc6979_minerva_amplified.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/src/ufsecp_impl.cpp)
audit_target_defaults(test_exploit_rfc6979_minerva_amplified_standalone)
target_include_directories(test_exploit_rfc6979_minerva_amplified_standalone PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include ${CMAKE_BINARY_DIR}/include)
target_compile_definitions(test_exploit_rfc6979_minerva_amplified_standalone PRIVATE STANDALONE_TEST UFSECP_BUILDING)
add_test(NAME exploit_rfc6979_minerva_amplified COMMAND test_exploit_rfc6979_minerva_amplified_standalone)
set_tests_properties(exploit_rfc6979_minerva_amplified PROPERTIES TIMEOUT 120 LABELS "audit;exploit;rfc6979;nonce;minerva")

# -- PoC: EllSwift bad scalar — ECDH/XDH rejection consistency ---------------
add_executable(test_exploit_ellswift_bad_scalar_ecdh_standalone
    test_exploit_ellswift_bad_scalar_ecdh.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/src/ufsecp_impl.cpp)
audit_target_defaults(test_exploit_ellswift_bad_scalar_ecdh_standalone)
target_include_directories(test_exploit_ellswift_bad_scalar_ecdh_standalone PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include ${CMAKE_BINARY_DIR}/include)
target_compile_definitions(test_exploit_ellswift_bad_scalar_ecdh_standalone PRIVATE
    STANDALONE_TEST UFSECP_BUILDING SECP256K1_BIP324)
add_test(NAME exploit_ellswift_bad_scalar_ecdh COMMAND test_exploit_ellswift_bad_scalar_ecdh_standalone)
set_tests_properties(exploit_ellswift_bad_scalar_ecdh PROPERTIES TIMEOUT 60 LABELS "audit;exploit;ellswift;ecdh;scalar")

# -- ECDSA r <=> x-coordinate edge cases (Wycheproof PR #206) ---------------
add_executable(test_exploit_ecdsa_r_overflow_standalone
    ${CMAKE_CURRENT_SOURCE_DIR}/test_exploit_ecdsa_r_overflow.cpp)
audit_target_defaults(test_exploit_ecdsa_r_overflow_standalone)
target_compile_definitions(test_exploit_ecdsa_r_overflow_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_ecdsa_r_overflow COMMAND test_exploit_ecdsa_r_overflow_standalone)
set_tests_properties(exploit_ecdsa_r_overflow PROPERTIES TIMEOUT 60 LABELS "audit;exploit;ecdsa;wycheproof")

# -- ECDSA PMN wraparound: r in [n,p) constant + logic correctness (2026-05-05) --
add_executable(test_exploit_ecdsa_pmn_wraparound_standalone
    ${CMAKE_CURRENT_SOURCE_DIR}/test_exploit_ecdsa_pmn_wraparound.cpp)
audit_target_defaults(test_exploit_ecdsa_pmn_wraparound_standalone)
target_compile_definitions(test_exploit_ecdsa_pmn_wraparound_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_ecdsa_pmn_wraparound COMMAND test_exploit_ecdsa_pmn_wraparound_standalone)
set_tests_properties(exploit_ecdsa_pmn_wraparound PROPERTIES TIMEOUT 60 LABELS "audit;exploit;ecdsa;correctness")

# -- ECDSA sign sentinel paths: r=0 / s=0 / sk=0 guards (X-1, X-2, X-4) ---
add_executable(test_exploit_ecdsa_sign_sentinels_standalone
    ${CMAKE_CURRENT_SOURCE_DIR}/test_exploit_ecdsa_sign_sentinels.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/src/ufsecp_impl.cpp)
audit_target_defaults(test_exploit_ecdsa_sign_sentinels_standalone)
target_include_directories(test_exploit_ecdsa_sign_sentinels_standalone PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include ${CMAKE_BINARY_DIR}/include)
target_compile_definitions(test_exploit_ecdsa_sign_sentinels_standalone PRIVATE STANDALONE_TEST UFSECP_BUILDING)
add_test(NAME exploit_ecdsa_sign_sentinels COMMAND test_exploit_ecdsa_sign_sentinels_standalone)
set_tests_properties(exploit_ecdsa_sign_sentinels PROPERTIES TIMEOUT 60 LABELS "audit;exploit;ecdsa;sentinel")

# -- Boundary sentinel edge cases (inverse(0), empty batch, half-order, MuSig2 dup) ---
add_executable(test_exploit_boundary_sentinels_standalone
    ${CMAKE_CURRENT_SOURCE_DIR}/test_exploit_boundary_sentinels.cpp)
audit_target_defaults(test_exploit_boundary_sentinels_standalone)
target_compile_definitions(test_exploit_boundary_sentinels_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_boundary_sentinels COMMAND test_exploit_boundary_sentinels_standalone)
set_tests_properties(exploit_boundary_sentinels PROPERTIES TIMEOUT 60 LABELS "audit;exploit;boundary;sentinel")

# -- Wycheproof ECDSA Bitcoin variant vectors (BIP-62 low-S + PR #206) ------
add_executable(test_wycheproof_ecdsa_bitcoin_standalone
    ${CMAKE_CURRENT_SOURCE_DIR}/test_wycheproof_ecdsa_bitcoin.cpp)
audit_target_defaults(test_wycheproof_ecdsa_bitcoin_standalone)
target_compile_definitions(test_wycheproof_ecdsa_bitcoin_standalone PRIVATE STANDALONE_TEST)
add_test(NAME wycheproof_ecdsa_bitcoin COMMAND test_wycheproof_ecdsa_bitcoin_standalone)
set_tests_properties(wycheproof_ecdsa_bitcoin PROPERTIES TIMEOUT 60 LABELS "audit;kat;wycheproof;bitcoin")

# -- Wycheproof ECDSA secp256k1 SHA-256 DER vectors -------------------------
add_executable(test_wycheproof_ecdsa_sha256_standalone
    ${CMAKE_CURRENT_SOURCE_DIR}/test_wycheproof_ecdsa_secp256k1_sha256.cpp)
audit_target_defaults(test_wycheproof_ecdsa_sha256_standalone)
target_compile_definitions(test_wycheproof_ecdsa_sha256_standalone PRIVATE STANDALONE_TEST)
add_test(NAME wycheproof_ecdsa_sha256 COMMAND test_wycheproof_ecdsa_sha256_standalone)
set_tests_properties(wycheproof_ecdsa_sha256 PROPERTIES TIMEOUT 60 LABELS "audit;kat;wycheproof;ecdsa;der")

# -- Wycheproof ECDSA secp256k1 SHA-256 extended coverage -------------------
# Covers flag categories absent from the base sha256 test:
#   RangeCheck, InvalidTypesInSignature, EdgeCaseShamirMultiplication,
#   SmallRandS (CVE-2020-13895), PointDuplication, EdgeCasePublicKey
add_executable(test_wycheproof_ecdsa_extended_standalone
    ${CMAKE_CURRENT_SOURCE_DIR}/test_wycheproof_ecdsa_extended.cpp)
audit_target_defaults(test_wycheproof_ecdsa_extended_standalone)
target_compile_definitions(test_wycheproof_ecdsa_extended_standalone PRIVATE STANDALONE_TEST)
add_test(NAME wycheproof_ecdsa_extended COMMAND test_wycheproof_ecdsa_extended_standalone)
set_tests_properties(wycheproof_ecdsa_extended PROPERTIES TIMEOUT 60
    LABELS "audit;kat;wycheproof;ecdsa;der;edge-case")

# -- Wycheproof ECDSA secp256k1 SHA-256 P1363 vectors -----------------------
add_executable(test_wycheproof_ecdsa_sha256_p1363_standalone
    ${CMAKE_CURRENT_SOURCE_DIR}/test_wycheproof_ecdsa_secp256k1_sha256_p1363.cpp)
audit_target_defaults(test_wycheproof_ecdsa_sha256_p1363_standalone)
target_compile_definitions(test_wycheproof_ecdsa_sha256_p1363_standalone PRIVATE STANDALONE_TEST)
add_test(NAME wycheproof_ecdsa_sha256_p1363 COMMAND test_wycheproof_ecdsa_sha256_p1363_standalone)
set_tests_properties(wycheproof_ecdsa_sha256_p1363 PROPERTIES TIMEOUT 60 LABELS "audit;kat;wycheproof;ecdsa;p1363")

# -- Wycheproof HMAC-SHA256 vectors -----------------------------------------
add_executable(test_wycheproof_hmac_sha256_standalone
    ${CMAKE_CURRENT_SOURCE_DIR}/test_wycheproof_hmac_sha256.cpp)
audit_target_defaults(test_wycheproof_hmac_sha256_standalone)
target_compile_definitions(test_wycheproof_hmac_sha256_standalone PRIVATE STANDALONE_TEST)
add_test(NAME wycheproof_hmac_sha256 COMMAND test_wycheproof_hmac_sha256_standalone)
set_tests_properties(wycheproof_hmac_sha256 PROPERTIES TIMEOUT 120 LABELS "audit;kat;wycheproof;hmac")

# -- Wycheproof HKDF-SHA256 vectors -----------------------------------------
add_executable(test_wycheproof_hkdf_sha256_standalone
    ${CMAKE_CURRENT_SOURCE_DIR}/test_wycheproof_hkdf_sha256.cpp)
audit_target_defaults(test_wycheproof_hkdf_sha256_standalone)
target_compile_definitions(test_wycheproof_hkdf_sha256_standalone PRIVATE STANDALONE_TEST)
add_test(NAME wycheproof_hkdf_sha256 COMMAND test_wycheproof_hkdf_sha256_standalone)
set_tests_properties(wycheproof_hkdf_sha256 PROPERTIES TIMEOUT 120 LABELS "audit;kat;wycheproof;hkdf")

# -- Wycheproof ChaCha20-Poly1305 vectors ------------------------------------
add_executable(test_wycheproof_chacha20_poly1305_standalone
    ${CMAKE_CURRENT_SOURCE_DIR}/test_wycheproof_chacha20_poly1305.cpp)
audit_target_defaults(test_wycheproof_chacha20_poly1305_standalone)
target_compile_definitions(test_wycheproof_chacha20_poly1305_standalone PRIVATE STANDALONE_TEST)
add_test(NAME wycheproof_chacha20_poly1305 COMMAND test_wycheproof_chacha20_poly1305_standalone)
set_tests_properties(wycheproof_chacha20_poly1305 PROPERTIES TIMEOUT 120 LABELS "audit;kat;wycheproof;chacha20;aead")

# -- Wycheproof ECDSA secp256k1 SHA-512 DER vectors --------------------------
add_executable(test_wycheproof_ecdsa_sha512_standalone
    ${CMAKE_CURRENT_SOURCE_DIR}/test_wycheproof_ecdsa_secp256k1_sha512.cpp)
audit_target_defaults(test_wycheproof_ecdsa_sha512_standalone)
target_compile_definitions(test_wycheproof_ecdsa_sha512_standalone PRIVATE STANDALONE_TEST)
add_test(NAME wycheproof_ecdsa_sha512 COMMAND test_wycheproof_ecdsa_sha512_standalone)
set_tests_properties(wycheproof_ecdsa_sha512 PROPERTIES TIMEOUT 300 LABELS "audit;kat;wycheproof;ecdsa;sha512;der")

# -- Wycheproof ECDSA secp256k1 SHA-512 P1363 vectors -----------------------
add_executable(test_wycheproof_ecdsa_sha512_p1363_standalone
    ${CMAKE_CURRENT_SOURCE_DIR}/test_wycheproof_ecdsa_secp256k1_sha512_p1363.cpp)
audit_target_defaults(test_wycheproof_ecdsa_sha512_p1363_standalone)
target_compile_definitions(test_wycheproof_ecdsa_sha512_p1363_standalone PRIVATE STANDALONE_TEST)
add_test(NAME wycheproof_ecdsa_sha512_p1363 COMMAND test_wycheproof_ecdsa_sha512_p1363_standalone)
set_tests_properties(wycheproof_ecdsa_sha512_p1363 PROPERTIES TIMEOUT 300 LABELS "audit;kat;wycheproof;ecdsa;sha512;p1363")

# -- Schnorr BIP-340 nonce reuse key recovery (SNR-1..SNR-10) ---------------
add_executable(test_exploit_schnorr_nonce_reuse_standalone
    ${CMAKE_CURRENT_SOURCE_DIR}/test_exploit_schnorr_nonce_reuse.cpp)
audit_target_defaults(test_exploit_schnorr_nonce_reuse_standalone)
target_compile_definitions(test_exploit_schnorr_nonce_reuse_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_schnorr_nonce_reuse COMMAND test_exploit_schnorr_nonce_reuse_standalone)
set_tests_properties(exploit_schnorr_nonce_reuse PROPERTIES TIMEOUT 60 LABELS "audit;exploit;schnorr;nonce_reuse")

# -- BIP-32 non-hardened child key → parent private key recovery (CKA-1..CKA-8) --
add_executable(test_exploit_bip32_child_key_attack_standalone
    ${CMAKE_CURRENT_SOURCE_DIR}/test_exploit_bip32_child_key_attack.cpp)
audit_target_defaults(test_exploit_bip32_child_key_attack_standalone)
target_compile_definitions(test_exploit_bip32_child_key_attack_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_bip32_child_key_attack COMMAND test_exploit_bip32_child_key_attack_standalone)
set_tests_properties(exploit_bip32_child_key_attack PROPERTIES TIMEOUT 60 LABELS "audit;exploit;bip32;key_derivation")

# -- FROST identifiable abort / per-participant attribution (FIA-1..FIA-9) --
add_executable(test_exploit_frost_identifiable_abort_standalone
    ${CMAKE_CURRENT_SOURCE_DIR}/test_exploit_frost_identifiable_abort.cpp)
audit_target_defaults(test_exploit_frost_identifiable_abort_standalone)
target_compile_definitions(test_exploit_frost_identifiable_abort_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_frost_identifiable_abort COMMAND test_exploit_frost_identifiable_abort_standalone)
set_tests_properties(exploit_frost_identifiable_abort PROPERTIES TIMEOUT 60 LABELS "audit;exploit;frost;identifiable_abort")

# -- Hash algorithm & format isolation (HAS-1..HAS-8) ----------------------
add_executable(test_exploit_hash_algo_sig_isolation_standalone
    ${CMAKE_CURRENT_SOURCE_DIR}/test_exploit_hash_algo_sig_isolation.cpp)
audit_target_defaults(test_exploit_hash_algo_sig_isolation_standalone)
target_compile_definitions(test_exploit_hash_algo_sig_isolation_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_hash_algo_sig_isolation COMMAND test_exploit_hash_algo_sig_isolation_standalone)
set_tests_properties(exploit_hash_algo_sig_isolation PROPERTIES TIMEOUT 60 LABELS "audit;exploit;ecdsa;hash_confusion")

# -- PoC: Mutation residue detection (wNAF OOB + divsteps mask) ------------
add_executable(test_exploit_mutation_residue_standalone test_exploit_mutation_residue.cpp)
audit_target_defaults(test_exploit_mutation_residue_standalone)
target_compile_definitions(test_exploit_mutation_residue_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_mutation_residue COMMAND test_exploit_mutation_residue_standalone)
set_tests_properties(exploit_mutation_residue PROPERTIES TIMEOUT 60 LABELS "audit;exploit;mutation;scalar;point")

# -- PoC: Metal field_reduce_512 acc[8] truncation (issue #226) -----------
add_executable(test_exploit_metal_field_reduce_standalone test_exploit_metal_field_reduce.cpp)
audit_target_defaults(test_exploit_metal_field_reduce_standalone)
target_compile_definitions(test_exploit_metal_field_reduce_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_metal_field_reduce COMMAND test_exploit_metal_field_reduce_standalone)
set_tests_properties(exploit_metal_field_reduce PROPERTIES TIMEOUT 60 LABELS "audit;exploit;metal;field")

# -- PoC: Network selector validation bypass (all address/WIF ABI) ----------
add_executable(test_exploit_network_validation_bypass_standalone
    test_exploit_network_validation_bypass.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/src/ufsecp_impl.cpp)
audit_target_defaults(test_exploit_network_validation_bypass_standalone)
target_include_directories(test_exploit_network_validation_bypass_standalone PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_BINARY_DIR}/include
)
target_compile_definitions(test_exploit_network_validation_bypass_standalone PRIVATE STANDALONE_TEST UFSECP_BUILDING)
add_test(NAME exploit_network_validation_bypass COMMAND test_exploit_network_validation_bypass_standalone)
set_tests_properties(exploit_network_validation_bypass PROPERTIES TIMEOUT 60 LABELS "audit;exploit;address;abi;network")

# -- ePrint 2025/076: ZVP-DCP attack on GLV multiscalar multiplication ------
add_executable(test_exploit_zvp_glv_dcp_multiscalar_standalone
    test_exploit_zvp_glv_dcp_multiscalar.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/src/ufsecp_impl.cpp)
audit_target_defaults(test_exploit_zvp_glv_dcp_multiscalar_standalone)
target_include_directories(test_exploit_zvp_glv_dcp_multiscalar_standalone PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_BINARY_DIR}/include
)
target_compile_definitions(test_exploit_zvp_glv_dcp_multiscalar_standalone PRIVATE STANDALONE_TEST UFSECP_BUILDING)
add_test(NAME exploit_zvp_glv_dcp_multiscalar COMMAND test_exploit_zvp_glv_dcp_multiscalar_standalone)
set_tests_properties(exploit_zvp_glv_dcp_multiscalar PROPERTIES TIMEOUT 60 LABELS "audit;exploit;glv;zvp;dcp;nonce")

# -- ePrint 2024/296: Lattice sieve HNP with sub-bit nonce leakage ----------
add_executable(test_exploit_lattice_sieve_hnp_standalone
    test_exploit_lattice_sieve_hnp.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/src/ufsecp_impl.cpp)
audit_target_defaults(test_exploit_lattice_sieve_hnp_standalone)
target_include_directories(test_exploit_lattice_sieve_hnp_standalone PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_BINARY_DIR}/include
)
target_compile_definitions(test_exploit_lattice_sieve_hnp_standalone PRIVATE STANDALONE_TEST UFSECP_BUILDING)
add_test(NAME exploit_lattice_sieve_hnp COMMAND test_exploit_lattice_sieve_hnp_standalone)
set_tests_properties(exploit_lattice_sieve_hnp PROPERTIES TIMEOUT 60 LABELS "audit;exploit;hnp;lattice;nonce")

# -- ePrint 2017/975: DFA on RFC 6979 deterministic signatures ---------------
add_executable(test_exploit_deterministic_sig_dfa_standalone
    test_exploit_deterministic_sig_dfa.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/src/ufsecp_impl.cpp)
audit_target_defaults(test_exploit_deterministic_sig_dfa_standalone)
target_include_directories(test_exploit_deterministic_sig_dfa_standalone PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_BINARY_DIR}/include
)
target_compile_definitions(test_exploit_deterministic_sig_dfa_standalone PRIVATE STANDALONE_TEST UFSECP_BUILDING)
add_test(NAME exploit_deterministic_sig_dfa COMMAND test_exploit_deterministic_sig_dfa_standalone)
set_tests_properties(exploit_deterministic_sig_dfa PROPERTIES TIMEOUT 60 LABELS "audit;exploit;rfc6979;dfa;deterministic")

# -- CVE-2024-49364/49365/2022-41340: Type confusion → k-reuse & verify bypass
add_executable(test_exploit_sign_type_confusion_kreuse_standalone
    test_exploit_sign_type_confusion_kreuse.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/src/ufsecp_impl.cpp)
audit_target_defaults(test_exploit_sign_type_confusion_kreuse_standalone)
target_include_directories(test_exploit_sign_type_confusion_kreuse_standalone PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_BINARY_DIR}/include
)
target_compile_definitions(test_exploit_sign_type_confusion_kreuse_standalone PRIVATE STANDALONE_TEST UFSECP_BUILDING)
add_test(NAME exploit_sign_type_confusion_kreuse COMMAND test_exploit_sign_type_confusion_kreuse_standalone)
set_tests_properties(exploit_sign_type_confusion_kreuse PROPERTIES TIMEOUT 60 LABELS "audit;exploit;cve;kreuse;typeconfusion")

# -- ePrint 2020/945: ROS problem → concurrent blind Schnorr forgery --------
add_executable(test_exploit_ros_concurrent_schnorr_standalone
    test_exploit_ros_concurrent_schnorr.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/src/ufsecp_impl.cpp)
audit_target_defaults(test_exploit_ros_concurrent_schnorr_standalone)
target_include_directories(test_exploit_ros_concurrent_schnorr_standalone PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_BINARY_DIR}/include
)
target_compile_definitions(test_exploit_ros_concurrent_schnorr_standalone PRIVATE STANDALONE_TEST UFSECP_BUILDING)
add_test(NAME exploit_ros_concurrent_schnorr COMMAND test_exploit_ros_concurrent_schnorr_standalone)
set_tests_properties(exploit_ros_concurrent_schnorr PROPERTIES TIMEOUT 60 LABELS "audit;exploit;ros;schnorr;concurrent")

# -- ePrint 2026/075, 2025/1001: FROST weak binding & adaptive corruption ----
add_executable(test_exploit_frost_weak_binding_standalone
    test_exploit_frost_weak_binding.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/src/ufsecp_impl.cpp)
audit_target_defaults(test_exploit_frost_weak_binding_standalone)
target_include_directories(test_exploit_frost_weak_binding_standalone PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_BINARY_DIR}/include
)
target_compile_definitions(test_exploit_frost_weak_binding_standalone PRIVATE STANDALONE_TEST UFSECP_BUILDING)
add_test(NAME exploit_frost_weak_binding COMMAND test_exploit_frost_weak_binding_standalone)
set_tests_properties(exploit_frost_weak_binding PROPERTIES TIMEOUT 60 LABELS "audit;exploit;frost;binding;adaptive")

# -- ePrint 2024/589, 2025/935: Blind-Folded SPA & CT cmov/cswap leak ------
add_executable(test_exploit_blind_spa_cmov_leak_standalone
    test_exploit_blind_spa_cmov_leak.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/src/ufsecp_impl.cpp)
audit_target_defaults(test_exploit_blind_spa_cmov_leak_standalone)
target_include_directories(test_exploit_blind_spa_cmov_leak_standalone PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_BINARY_DIR}/include
)
target_compile_definitions(test_exploit_blind_spa_cmov_leak_standalone PRIVATE STANDALONE_TEST UFSECP_BUILDING)
add_test(NAME exploit_blind_spa_cmov_leak COMMAND test_exploit_blind_spa_cmov_leak_standalone)
set_tests_properties(exploit_blind_spa_cmov_leak PROPERTIES TIMEOUT 60 LABELS "audit;exploit;spa;cmov;ct")

# -- ePrint 2025/1293: ECTester point order/infinity validation -------------
add_executable(test_exploit_ectester_point_validation_standalone
    test_exploit_ectester_point_validation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/src/ufsecp_impl.cpp)
audit_target_defaults(test_exploit_ectester_point_validation_standalone)
target_include_directories(test_exploit_ectester_point_validation_standalone PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_BINARY_DIR}/include
)
target_compile_definitions(test_exploit_ectester_point_validation_standalone PRIVATE STANDALONE_TEST UFSECP_BUILDING)
add_test(NAME exploit_ectester_point_validation COMMAND test_exploit_ectester_point_validation_standalone)
set_tests_properties(exploit_ectester_point_validation PROPERTIES TIMEOUT 60 LABELS "audit;exploit;ectester;point;validation")

# -- ePrint 2025/306, 2025/1353: Dimensional eROS & BZ blind sig forgery ----
add_executable(test_exploit_ros_dimensional_erosion_standalone
    test_exploit_ros_dimensional_erosion.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/src/ufsecp_impl.cpp)
audit_target_defaults(test_exploit_ros_dimensional_erosion_standalone)
target_include_directories(test_exploit_ros_dimensional_erosion_standalone PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_BINARY_DIR}/include
)
target_compile_definitions(test_exploit_ros_dimensional_erosion_standalone PRIVATE STANDALONE_TEST UFSECP_BUILDING)
add_test(NAME exploit_ros_dimensional_erosion COMMAND test_exploit_ros_dimensional_erosion_standalone)
set_tests_properties(exploit_ros_dimensional_erosion PROPERTIES TIMEOUT 120 LABELS "audit;exploit;ros;eros;blindsig")

# -- ePrint 2026/663: Modified ECDSA batch verification randomization -------
add_executable(test_exploit_ecdsa_batch_verify_rand_standalone
    test_exploit_ecdsa_batch_verify_rand.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/src/ufsecp_impl.cpp)
audit_target_defaults(test_exploit_ecdsa_batch_verify_rand_standalone)
target_include_directories(test_exploit_ecdsa_batch_verify_rand_standalone PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_BINARY_DIR}/include
)
target_compile_definitions(test_exploit_ecdsa_batch_verify_rand_standalone PRIVATE STANDALONE_TEST UFSECP_BUILDING)
add_test(NAME exploit_ecdsa_batch_verify_rand COMMAND test_exploit_ecdsa_batch_verify_rand_standalone)
set_tests_properties(exploit_ecdsa_batch_verify_rand PROPERTIES TIMEOUT 120 LABELS "audit;exploit;ecdsa;batch;verify;rand")

# -- BIP-324 AEAD forgery & ChaCha20-Poly1305 boundary (BIP-324, RFC 8439) --
add_executable(test_exploit_bip324_aead_forgery_standalone
    test_exploit_bip324_aead_forgery.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/src/ufsecp_impl.cpp)
audit_target_defaults(test_exploit_bip324_aead_forgery_standalone)
target_include_directories(test_exploit_bip324_aead_forgery_standalone PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_BINARY_DIR}/include
)
target_compile_definitions(test_exploit_bip324_aead_forgery_standalone PRIVATE STANDALONE_TEST UFSECP_BUILDING SECP256K1_BIP324)
add_test(NAME exploit_bip324_aead_forgery COMMAND test_exploit_bip324_aead_forgery_standalone)
set_tests_properties(exploit_bip324_aead_forgery PROPERTIES TIMEOUT 120 LABELS "audit;exploit;bip324;aead")

# -- FROST rogue-key / key-cancellation (ePrint 2020/852, 2023/899) --
add_executable(test_exploit_frost_rogue_key_standalone
    test_exploit_frost_rogue_key.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/src/ufsecp_impl.cpp)
audit_target_defaults(test_exploit_frost_rogue_key_standalone)
target_include_directories(test_exploit_frost_rogue_key_standalone PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_BINARY_DIR}/include
)
target_compile_definitions(test_exploit_frost_rogue_key_standalone PRIVATE STANDALONE_TEST UFSECP_BUILDING)
add_test(NAME exploit_frost_rogue_key COMMAND test_exploit_frost_rogue_key_standalone)
set_tests_properties(exploit_frost_rogue_key PROPERTIES TIMEOUT 120 LABELS "audit;exploit;frost;rogue-key")

# -- MuSig2 partial signature forgery & session binding (ePrint 2020/1261) --
add_executable(test_exploit_musig2_partial_forgery_standalone
    test_exploit_musig2_partial_forgery.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/src/ufsecp_impl.cpp)
audit_target_defaults(test_exploit_musig2_partial_forgery_standalone)
target_include_directories(test_exploit_musig2_partial_forgery_standalone PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_BINARY_DIR}/include
)
target_compile_definitions(test_exploit_musig2_partial_forgery_standalone PRIVATE STANDALONE_TEST UFSECP_BUILDING)
add_test(NAME exploit_musig2_partial_forgery COMMAND test_exploit_musig2_partial_forgery_standalone)
set_tests_properties(exploit_musig2_partial_forgery PROPERTIES TIMEOUT 120 LABELS "audit;exploit;musig2;forgery")

# -- Schnorr adaptor extraction soundness (ePrint 2020/476, 2021/150) --
add_executable(test_exploit_adaptor_extraction_soundness_standalone
    test_exploit_adaptor_extraction_soundness.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/src/ufsecp_impl.cpp)
audit_target_defaults(test_exploit_adaptor_extraction_soundness_standalone)
target_include_directories(test_exploit_adaptor_extraction_soundness_standalone PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_BINARY_DIR}/include
)
target_compile_definitions(test_exploit_adaptor_extraction_soundness_standalone PRIVATE STANDALONE_TEST UFSECP_BUILDING)
add_test(NAME exploit_adaptor_extraction_soundness COMMAND test_exploit_adaptor_extraction_soundness_standalone)
set_tests_properties(exploit_adaptor_extraction_soundness PROPERTIES TIMEOUT 120 LABELS "audit;exploit;adaptor;soundness")

# -- ECDH twist point injection / Pohlig-Hellman (ePrint 2015/1233) --
add_executable(test_exploit_ecdh_twist_injection_standalone
    test_exploit_ecdh_twist_injection.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/src/ufsecp_impl.cpp)
audit_target_defaults(test_exploit_ecdh_twist_injection_standalone)
target_include_directories(test_exploit_ecdh_twist_injection_standalone PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_BINARY_DIR}/include
)
target_compile_definitions(test_exploit_ecdh_twist_injection_standalone PRIVATE STANDALONE_TEST UFSECP_BUILDING)
add_test(NAME exploit_ecdh_twist_injection COMMAND test_exploit_ecdh_twist_injection_standalone)
set_tests_properties(exploit_ecdh_twist_injection PROPERTIES TIMEOUT 120 LABELS "audit;exploit;ecdh;twist")

# -- Schnorr batch verify inflation / mix attack (BIP-340, ePrint 2012/549) --
add_executable(test_exploit_schnorr_batch_inflation_standalone
    test_exploit_schnorr_batch_inflation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/src/ufsecp_impl.cpp)
audit_target_defaults(test_exploit_schnorr_batch_inflation_standalone)
target_include_directories(test_exploit_schnorr_batch_inflation_standalone PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_BINARY_DIR}/include
)
target_compile_definitions(test_exploit_schnorr_batch_inflation_standalone PRIVATE STANDALONE_TEST UFSECP_BUILDING)
add_test(NAME exploit_schnorr_batch_inflation COMMAND test_exploit_schnorr_batch_inflation_standalone)
set_tests_properties(exploit_schnorr_batch_inflation PROPERTIES TIMEOUT 120 LABELS "audit;exploit;schnorr;batch")

# -- SafeGCD / Bernstein-Yang divstep correctness (SGD-1..11) ---------------
add_executable(test_exploit_safegcd_divsteps_standalone
    test_exploit_safegcd_divsteps.cpp)
audit_target_defaults(test_exploit_safegcd_divsteps_standalone)
target_include_directories(test_exploit_safegcd_divsteps_standalone PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_BINARY_DIR}/include
)
target_compile_definitions(test_exploit_safegcd_divsteps_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_safegcd_divsteps COMMAND test_exploit_safegcd_divsteps_standalone)
set_tests_properties(exploit_safegcd_divsteps PROPERTIES TIMEOUT 60 LABELS "audit;exploit;math;safegcd")

# -- Nonce edge cases: null / zero-key / n / n-1 / determinism (NIN-1..15) --
add_executable(test_exploit_custom_nonce_injection_standalone
    test_exploit_custom_nonce_injection.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/src/ufsecp_impl.cpp)
audit_target_defaults(test_exploit_custom_nonce_injection_standalone)
target_include_directories(test_exploit_custom_nonce_injection_standalone PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_BINARY_DIR}/include
)
target_compile_definitions(test_exploit_custom_nonce_injection_standalone PRIVATE STANDALONE_TEST UFSECP_BUILDING)
add_test(NAME exploit_custom_nonce_injection COMMAND test_exploit_custom_nonce_injection_standalone)
set_tests_properties(exploit_custom_nonce_injection PROPERTIES TIMEOUT 60 LABELS "audit;exploit;nonce;ecdsa")

# -- JNI return-value ignored (RVI-1..8) — 2026-04-23 ---------------------
add_executable(test_exploit_jni_retval_ignored_standalone
    test_exploit_jni_retval_ignored.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../bindings/c_api/ultrafast_secp256k1.cpp)
audit_target_defaults(test_exploit_jni_retval_ignored_standalone)
target_include_directories(test_exploit_jni_retval_ignored_standalone PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_BINARY_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../bindings/c_api
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/include
)
target_compile_definitions(test_exploit_jni_retval_ignored_standalone PRIVATE
    STANDALONE_TEST ULTRAFAST_SECP256K1_BUILDING)
add_test(NAME exploit_jni_retval_ignored COMMAND test_exploit_jni_retval_ignored_standalone)
set_tests_properties(exploit_jni_retval_ignored PROPERTIES TIMEOUT 30 LABELS "audit;exploit;jni;bindings;retval")

# -- Dark Skippy nonce exfiltration (DS-1..8) — ePrint 2024/1225 ---------
add_executable(test_exploit_dark_skippy_exfil_standalone
    test_exploit_dark_skippy_exfil.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/src/ufsecp_impl.cpp)
audit_target_defaults(test_exploit_dark_skippy_exfil_standalone)
target_include_directories(test_exploit_dark_skippy_exfil_standalone PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_BINARY_DIR}/include
)
target_compile_definitions(test_exploit_dark_skippy_exfil_standalone PRIVATE STANDALONE_TEST UFSECP_BUILDING)
add_test(NAME exploit_dark_skippy_exfil COMMAND test_exploit_dark_skippy_exfil_standalone)
set_tests_properties(exploit_dark_skippy_exfil PROPERTIES TIMEOUT 30 LABELS "audit;exploit;nonce;ecdsa;schnorr;dark_skippy")

# -- Frozen Heart ZK Fiat-Shamir incomplete binding (FH-1..10) — ePrint 2022/411 -
add_executable(test_exploit_fiat_shamir_frozen_heart_standalone
    test_exploit_fiat_shamir_frozen_heart.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/src/ufsecp_impl.cpp)
audit_target_defaults(test_exploit_fiat_shamir_frozen_heart_standalone)
target_include_directories(test_exploit_fiat_shamir_frozen_heart_standalone PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_BINARY_DIR}/include
)
target_compile_definitions(test_exploit_fiat_shamir_frozen_heart_standalone PRIVATE STANDALONE_TEST UFSECP_BUILDING)
add_test(NAME exploit_fiat_shamir_frozen_heart COMMAND test_exploit_fiat_shamir_frozen_heart_standalone)
set_tests_properties(exploit_fiat_shamir_frozen_heart PROPERTIES TIMEOUT 30 LABELS "audit;exploit;zk;fiat-shamir;frozen-heart")

# -- Hertzbleed scalar blinding Hamming-weight defence (SB-1..9) — ePrint 2022/823 -
add_executable(test_exploit_hertzbleed_scalar_blind_standalone
    test_exploit_hertzbleed_scalar_blind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/src/ufsecp_impl.cpp)
audit_target_defaults(test_exploit_hertzbleed_scalar_blind_standalone)
target_include_directories(test_exploit_hertzbleed_scalar_blind_standalone PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_BINARY_DIR}/include
)
target_compile_definitions(test_exploit_hertzbleed_scalar_blind_standalone PRIVATE STANDALONE_TEST UFSECP_BUILDING)
add_test(NAME exploit_hertzbleed_scalar_blind COMMAND test_exploit_hertzbleed_scalar_blind_standalone)
set_tests_properties(exploit_hertzbleed_scalar_blind PROPERTIES TIMEOUT 30 LABELS "audit;exploit;sidechannel;dvfs;hertzbleed")

# -- Thread-safe lazy init regression (TIR-1..5) — 2026-04-27 ----------------
add_executable(test_exploit_thread_unsafe_lazy_init_standalone
    test_exploit_thread_unsafe_lazy_init.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/src/ufsecp_impl.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/src/ufsecp_gpu_impl.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/gpu/src/gpu_registry.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/gpu/src/gpu_backend_fallback.cpp)
audit_target_defaults(test_exploit_thread_unsafe_lazy_init_standalone)
target_include_directories(test_exploit_thread_unsafe_lazy_init_standalone PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_BINARY_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/gpu/include
)
target_compile_definitions(test_exploit_thread_unsafe_lazy_init_standalone PRIVATE STANDALONE_TEST UFSECP_BUILDING)
find_package(Threads REQUIRED)
target_link_libraries(test_exploit_thread_unsafe_lazy_init_standalone PRIVATE Threads::Threads)
add_test(NAME exploit_thread_unsafe_lazy_init COMMAND test_exploit_thread_unsafe_lazy_init_standalone)
set_tests_properties(exploit_thread_unsafe_lazy_init PROPERTIES TIMEOUT 60 LABELS "audit;exploit;thread;race;concurrency")

# -- z_fe_nonzero | vs & correctness regression (ZFN-1..6) — 2026-04-27 ---
add_executable(test_regression_z_fe_nonzero_standalone
    test_regression_z_fe_nonzero.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/src/ufsecp_impl.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/src/ufsecp_gpu_impl.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/gpu/src/gpu_registry.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/gpu/src/gpu_backend_fallback.cpp)
audit_target_defaults(test_regression_z_fe_nonzero_standalone)
target_include_directories(test_regression_z_fe_nonzero_standalone PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_BINARY_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/gpu/include
)
target_compile_definitions(test_regression_z_fe_nonzero_standalone PRIVATE STANDALONE_TEST UFSECP_BUILDING)
add_test(NAME regression_z_fe_nonzero COMMAND test_regression_z_fe_nonzero_standalone)
set_tests_properties(regression_z_fe_nonzero PROPERTIES TIMEOUT 300 LABELS "audit;regression;pubkey;correctness")

# -- CUDA pool minimum-capacity rounding regression (CAP-1..4) — 2026-04-27 -
add_executable(test_regression_cuda_pool_cap_standalone
    test_regression_cuda_pool_cap.cpp)
audit_target_defaults(test_regression_cuda_pool_cap_standalone)
target_include_directories(test_regression_cuda_pool_cap_standalone PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_BINARY_DIR}/include
)
target_compile_definitions(test_regression_cuda_pool_cap_standalone PRIVATE STANDALONE_TEST)
add_test(NAME regression_cuda_pool_cap COMMAND test_regression_cuda_pool_cap_standalone)
set_tests_properties(regression_cuda_pool_cap PROPERTIES TIMEOUT 30 LABELS "audit;regression;gpu;cuda;pool")

# -- musig2_partial_verify OOB + infinity-nonce regression (MVV-1..6) — 2026-04-23 -
add_executable(test_regression_musig2_verify_standalone
    test_regression_musig2_verify.cpp)
audit_target_defaults(test_regression_musig2_verify_standalone)
target_include_directories(test_regression_musig2_verify_standalone PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_BINARY_DIR}/include
)
target_compile_definitions(test_regression_musig2_verify_standalone PRIVATE STANDALONE_TEST)
add_test(NAME regression_musig2_verify COMMAND test_regression_musig2_verify_standalone)
set_tests_properties(regression_musig2_verify PROPERTIES TIMEOUT 300 LABELS "audit;regression;musig2;bounds")

# -- Bip324Session sk-leak + destructor secure-erase regression (BPS-1..8) — 2026-04-23 --
add_executable(test_regression_bip324_session_standalone
    test_regression_bip324_session.cpp)
audit_target_defaults(test_regression_bip324_session_standalone)
target_include_directories(test_regression_bip324_session_standalone PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_BINARY_DIR}/include
)
target_compile_definitions(test_regression_bip324_session_standalone PRIVATE STANDALONE_TEST)
add_test(NAME regression_bip324_session COMMAND test_regression_bip324_session_standalone)
set_tests_properties(regression_bip324_session PROPERTIES TIMEOUT 30 LABELS "audit;regression;bip324;secure-erase")

# -- Exception-path sk/ek/entropy leakage regression (EPE-RAII + EPE-1..12) — 2026-04-23 --
# NOTE: This test is built into unified_audit_runner and run as a module.
# A standalone executable target is not feasible because the test uses C ABI functions
# (ufsecp_*) defined in ufsecp_impl.cpp, which is only included in the unified runner.
# The test validates the RAII guards in ufsecp_impl.cpp and normal-path regression
# of 12+ affected C ABI functions (EPE-1..12).
# Wired into unified_audit_runner: yes. Standalone CTest: no (would have link errors).
# Coverage: unified_audit_runner --module regression_exception_erase  (or --section memory_safety)

# -- C-1 fix: C ABI recovery signing CT path (RCTX-1..8) — 2026-04-27 -------
add_executable(test_exploit_recoverable_sign_ct_standalone
    test_exploit_recoverable_sign_ct.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/src/ufsecp_impl.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/src/ufsecp_gpu_impl.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/gpu/src/gpu_registry.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/gpu/src/gpu_backend_fallback.cpp)
audit_target_defaults(test_exploit_recoverable_sign_ct_standalone)
target_include_directories(test_exploit_recoverable_sign_ct_standalone PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_BINARY_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/gpu/include
)
target_compile_definitions(test_exploit_recoverable_sign_ct_standalone PRIVATE STANDALONE_TEST UFSECP_BUILDING)
add_test(NAME exploit_recoverable_sign_ct COMMAND test_exploit_recoverable_sign_ct_standalone)
set_tests_properties(exploit_recoverable_sign_ct PROPERTIES TIMEOUT 60 LABELS "audit;exploit;ct;recovery;ecdsa")

# -- M-6 fix: Pippenger batch verify regression guard (PIPBATCH-1..8) — 2026-04-27 ---
add_executable(test_exploit_pippenger_batch_regression_standalone
    test_exploit_pippenger_batch_regression.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/src/ufsecp_impl.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/src/ufsecp_gpu_impl.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/gpu/src/gpu_registry.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/gpu/src/gpu_backend_fallback.cpp)
audit_target_defaults(test_exploit_pippenger_batch_regression_standalone)
target_include_directories(test_exploit_pippenger_batch_regression_standalone PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_BINARY_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/gpu/include
)
target_compile_definitions(test_exploit_pippenger_batch_regression_standalone PRIVATE STANDALONE_TEST UFSECP_BUILDING)
add_test(NAME exploit_pippenger_batch_regression COMMAND test_exploit_pippenger_batch_regression_standalone)
set_tests_properties(exploit_pippenger_batch_regression PROPERTIES TIMEOUT 120 LABELS "audit;exploit;pippenger;batch;regression")

# -- B-01 fix: Ethereum signing CT path (ETHCT-1..8) — 2026-04-27 ----------
add_executable(test_exploit_eth_signing_ct_standalone
    test_exploit_eth_signing_ct.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/src/eth_signing.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/src/keccak256.cpp)
audit_target_defaults(test_exploit_eth_signing_ct_standalone)
target_include_directories(test_exploit_eth_signing_ct_standalone PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_BINARY_DIR}/include
)
target_compile_definitions(test_exploit_eth_signing_ct_standalone PRIVATE STANDALONE_TEST SECP256K1_BUILD_ETHEREUM)
add_test(NAME exploit_eth_signing_ct COMMAND test_exploit_eth_signing_ct_standalone)
set_tests_properties(exploit_eth_signing_ct PROPERTIES TIMEOUT 60 LABELS "audit;exploit;ct;ethereum;ecdsa")

# -- B-02 fix: Wallet sign_hash CT path (WALCT-1..8) — 2026-04-27 ----------
add_executable(test_exploit_wallet_sign_ct_standalone
    test_exploit_wallet_sign_ct.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/src/wallet.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/src/coin_address.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/src/message_signing.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/src/eth_signing.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/src/keccak256.cpp)
audit_target_defaults(test_exploit_wallet_sign_ct_standalone)
target_include_directories(test_exploit_wallet_sign_ct_standalone PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_BINARY_DIR}/include
)
target_compile_definitions(test_exploit_wallet_sign_ct_standalone PRIVATE STANDALONE_TEST SECP256K1_BUILD_ETHEREUM)
add_test(NAME exploit_wallet_sign_ct COMMAND test_exploit_wallet_sign_ct_standalone)
set_tests_properties(exploit_wallet_sign_ct PROPERTIES TIMEOUT 60 LABELS "audit;exploit;ct;wallet;ecdsa")

# -- B-04 fix: Monolith split integrity (MONO-1..12) — 2026-04-27 -----------
add_executable(test_exploit_monolith_split_standalone
    test_exploit_monolith_split.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/src/ufsecp_impl.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/src/ufsecp_gpu_impl.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/gpu/src/gpu_registry.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/gpu/src/gpu_backend_fallback.cpp)
audit_target_defaults(test_exploit_monolith_split_standalone)
target_include_directories(test_exploit_monolith_split_standalone PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_BINARY_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/gpu/include
)
target_compile_definitions(test_exploit_monolith_split_standalone PRIVATE STANDALONE_TEST UFSECP_BUILDING SECP256K1_BUILD_ETHEREUM)
add_test(NAME exploit_monolith_split COMMAND test_exploit_monolith_split_standalone)
set_tests_properties(exploit_monolith_split PROPERTIES TIMEOUT 60 LABELS "audit;exploit;monolith;split;regression")

# -- B-08/B-09 fix: GPU secret erase + batch sign partial — 2026-04-27 ------
add_executable(test_exploit_gpu_secret_erase_standalone
    test_exploit_gpu_secret_erase.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/src/ufsecp_impl.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/src/ufsecp_gpu_impl.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/gpu/src/gpu_registry.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/gpu/src/gpu_backend_fallback.cpp)
audit_target_defaults(test_exploit_gpu_secret_erase_standalone)
target_include_directories(test_exploit_gpu_secret_erase_standalone PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_BINARY_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/gpu/include
)
target_compile_definitions(test_exploit_gpu_secret_erase_standalone PRIVATE STANDALONE_TEST UFSECP_BUILDING SECP256K1_BUILD_ETHEREUM)
add_test(NAME exploit_gpu_secret_erase COMMAND test_exploit_gpu_secret_erase_standalone)
set_tests_properties(exploit_gpu_secret_erase PROPERTIES TIMEOUT 60 LABELS "audit;exploit;gpu;secret;erase;batch")

# -- L-01: libsecp256k1 EC key API compatibility (ECKEY-1..17) — 2026-04-27 --
add_executable(test_exploit_libsecp_eckey_api_standalone
    test_exploit_libsecp_eckey_api.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/src/ufsecp_impl.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/src/ufsecp_gpu_impl.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/gpu/src/gpu_registry.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/gpu/src/gpu_backend_fallback.cpp)
audit_target_defaults(test_exploit_libsecp_eckey_api_standalone)
target_include_directories(test_exploit_libsecp_eckey_api_standalone PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_BINARY_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/gpu/include
)
target_compile_definitions(test_exploit_libsecp_eckey_api_standalone PRIVATE STANDALONE_TEST UFSECP_BUILDING)
add_test(NAME exploit_libsecp_eckey_api COMMAND test_exploit_libsecp_eckey_api_standalone)
set_tests_properties(exploit_libsecp_eckey_api PROPERTIES TIMEOUT 60 LABELS "audit;exploit;libsecp;eckey;tweak;negate;differential")

# -- BC-01: Bitcoin Core R-grinding nonce pattern (RGRIND-1..8) — 2026-04-27 -
add_executable(test_exploit_bitcoin_core_rgrinding_standalone test_exploit_bitcoin_core_rgrinding.cpp)
audit_target_defaults(test_exploit_bitcoin_core_rgrinding_standalone)
target_compile_definitions(test_exploit_bitcoin_core_rgrinding_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_bitcoin_core_rgrinding COMMAND test_exploit_bitcoin_core_rgrinding_standalone)
set_tests_properties(exploit_bitcoin_core_rgrinding PROPERTIES TIMEOUT 60 LABELS "audit;exploit;bitcoin;core;rgrinding;nonce;differential")

# -- Upstream gap closure (GAP-1, GAP-3, GAP-4) 2026-04-28 ------------------
add_executable(test_exploit_pubkey_cmp_standalone test_exploit_pubkey_cmp.cpp)
audit_target_defaults(test_exploit_pubkey_cmp_standalone)
target_compile_definitions(test_exploit_pubkey_cmp_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_pubkey_cmp COMMAND test_exploit_pubkey_cmp_standalone)
set_tests_properties(exploit_pubkey_cmp PROPERTIES TIMEOUT 60 LABELS "audit;exploit;pubkey;comparison;ordering;musig2")

add_executable(test_exploit_pubkey_sort_standalone test_exploit_pubkey_sort.cpp)
audit_target_defaults(test_exploit_pubkey_sort_standalone)
target_compile_definitions(test_exploit_pubkey_sort_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_pubkey_sort COMMAND test_exploit_pubkey_sort_standalone)
set_tests_properties(exploit_pubkey_sort PROPERTIES TIMEOUT 60 LABELS "audit;exploit;pubkey;sort;ordering;musig2")

add_executable(test_exploit_alloc_bounds_standalone test_exploit_alloc_bounds.cpp)
audit_target_defaults(test_exploit_alloc_bounds_standalone)
target_compile_definitions(test_exploit_alloc_bounds_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_alloc_bounds COMMAND test_exploit_alloc_bounds_standalone)
set_tests_properties(exploit_alloc_bounds PROPERTIES TIMEOUT 60 LABELS "audit;exploit;allocation;batch;scratch;bounds")

# -- Upstream gap closure (GAP-2, GAP-5, GAP-7) 2026-04-28 ------------------
add_executable(test_exploit_hsort_standalone test_exploit_hsort.cpp)
audit_target_defaults(test_exploit_hsort_standalone)
target_compile_definitions(test_exploit_hsort_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_hsort COMMAND test_exploit_hsort_standalone)
set_tests_properties(exploit_hsort PROPERTIES TIMEOUT 60 LABELS "audit;exploit;heapsort;batch;ordering")

add_executable(test_exploit_wnaf_standalone test_exploit_wnaf.cpp)
audit_target_defaults(test_exploit_wnaf_standalone)
target_compile_definitions(test_exploit_wnaf_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_wnaf COMMAND test_exploit_wnaf_standalone)
set_tests_properties(exploit_wnaf PROPERTIES TIMEOUT 60 LABELS "audit;exploit;wnaf;scalar;decomposition")

add_executable(test_exploit_int128_standalone test_exploit_int128.cpp)
audit_target_defaults(test_exploit_int128_standalone)
target_compile_definitions(test_exploit_int128_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_int128 COMMAND test_exploit_int128_standalone)
set_tests_properties(exploit_int128 PROPERTIES TIMEOUT 60 LABELS "audit;exploit;int128;field;arithmetic")

# -- Upstream gap closure (scratch allocator + xoshiro256** PRNG) 2026-04-28 -
add_executable(test_exploit_scratch_standalone test_exploit_scratch.cpp)
audit_target_defaults(test_exploit_scratch_standalone)
target_compile_definitions(test_exploit_scratch_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_scratch COMMAND test_exploit_scratch_standalone)
set_tests_properties(exploit_scratch PROPERTIES TIMEOUT 60 LABELS "audit;exploit;allocator;scratch;batch")

add_executable(test_exploit_xoshiro_standalone test_exploit_xoshiro.cpp)
audit_target_defaults(test_exploit_xoshiro_standalone)
target_compile_definitions(test_exploit_xoshiro_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_xoshiro COMMAND test_exploit_xoshiro_standalone)
set_tests_properties(exploit_xoshiro PROPERTIES TIMEOUT 60 LABELS "audit;exploit;xoshiro;prng;randomize")

# -- N1: Cross-Protocol Key Reuse (CPK-1..5) ----------------------------------
add_executable(test_exploit_cross_protocol_kreuse_standalone test_exploit_cross_protocol_kreuse.cpp)
audit_target_defaults(test_exploit_cross_protocol_kreuse_standalone)
target_compile_definitions(test_exploit_cross_protocol_kreuse_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_cross_protocol_kreuse COMMAND test_exploit_cross_protocol_kreuse_standalone)
set_tests_properties(exploit_cross_protocol_kreuse PROPERTIES TIMEOUT 60 LABELS "audit;exploit;musig2;frost;ecdsa;key-reuse")

# -- N2: BIP-340 Tagged SHA-256 Length Extension (TAGEXT-1..5) ----------------
add_executable(test_exploit_tagged_hash_ext_standalone test_exploit_tagged_hash_ext.cpp)
audit_target_defaults(test_exploit_tagged_hash_ext_standalone)
target_compile_definitions(test_exploit_tagged_hash_ext_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_tagged_hash_ext COMMAND test_exploit_tagged_hash_ext_standalone)
set_tests_properties(exploit_tagged_hash_ext PROPERTIES TIMEOUT 60 LABELS "audit;exploit;sha256;tagged-hash;length-extension")

# -- N3: wNAF Window-18 Cache Amplification (WCACHE-1..5) ---------------------
add_executable(test_exploit_wnaf_cache_ampl_standalone test_exploit_wnaf_cache_ampl.cpp)
audit_target_defaults(test_exploit_wnaf_cache_ampl_standalone)
target_compile_definitions(test_exploit_wnaf_cache_ampl_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_wnaf_cache_ampl COMMAND test_exploit_wnaf_cache_ampl_standalone)
set_tests_properties(exploit_wnaf_cache_ampl PROPERTIES TIMEOUT 60 LABELS "audit;exploit;wnaf;cache;flush-reload")

# -- N4: MuSig2 KeyAgg Fingerprint Collision (FPC-1..6) -----------------------
add_executable(test_exploit_musig2_fingerprint_collision_standalone test_exploit_musig2_fingerprint_collision.cpp)
audit_target_defaults(test_exploit_musig2_fingerprint_collision_standalone)
target_compile_definitions(test_exploit_musig2_fingerprint_collision_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_musig2_fingerprint_collision COMMAND test_exploit_musig2_fingerprint_collision_standalone)
set_tests_properties(exploit_musig2_fingerprint_collision PROPERTIES TIMEOUT 60 LABELS "audit;exploit;musig2;key-agg;collision")

# -- N5: Context Blinding Recovery via HNP (BLIND-1..7) -----------------------
add_executable(test_exploit_blinding_recovery_hnp_standalone test_exploit_blinding_recovery_hnp.cpp)
audit_target_defaults(test_exploit_blinding_recovery_hnp_standalone)
target_compile_definitions(test_exploit_blinding_recovery_hnp_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_blinding_recovery_hnp COMMAND test_exploit_blinding_recovery_hnp_standalone)
set_tests_properties(exploit_blinding_recovery_hnp PROPERTIES TIMEOUT 60 LABELS "audit;exploit;blinding;hnp;nonce")

# -- N6: Shim noncefp Callback Bypass (NONCEFP-1..5) — Original 2026-04-28 --
# The libsecp256k1 shim (shim_ecdsa.cpp:210) silently ignores the noncefp
# callback. Compile shim_ecdsa.cpp directly so the test runs without
# SECP256K1_BUILD_SHIM and the unified runner uses the C++ layer path.
add_executable(test_exploit_shim_noncefp_bypass_standalone
    test_exploit_shim_noncefp_bypass.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../compat/libsecp256k1_shim/src/shim_ecdsa.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../compat/libsecp256k1_shim/src/shim_context.cpp)
audit_target_defaults(test_exploit_shim_noncefp_bypass_standalone)
target_include_directories(test_exploit_shim_noncefp_bypass_standalone PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../compat/libsecp256k1_shim/include)
target_compile_definitions(test_exploit_shim_noncefp_bypass_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_shim_noncefp_bypass COMMAND test_exploit_shim_noncefp_bypass_standalone)
set_tests_properties(exploit_shim_noncefp_bypass PROPERTIES TIMEOUT 60 LABELS "audit;exploit;shim;noncefp;ecdsa")

# -- N7: Encoding Memory Corruption (ENCORR-1..6) — Original 2026-04-28 ------
# Adversarial DER/field/pubkey blobs fed to parsers — must be rejected safely.
# Shim_ecdsa.cpp included for the DER parser path in STANDALONE mode.
add_executable(test_exploit_encoding_memory_corruption_standalone
    test_exploit_encoding_memory_corruption.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../compat/libsecp256k1_shim/src/shim_ecdsa.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../compat/libsecp256k1_shim/src/shim_context.cpp)
audit_target_defaults(test_exploit_encoding_memory_corruption_standalone)
target_include_directories(test_exploit_encoding_memory_corruption_standalone PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../compat/libsecp256k1_shim/include)
target_compile_definitions(test_exploit_encoding_memory_corruption_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_encoding_memory_corruption COMMAND test_exploit_encoding_memory_corruption_standalone)
set_tests_properties(exploit_encoding_memory_corruption PROPERTIES TIMEOUT 60 LABELS "audit;exploit;encoding;der;memory")

# -- N8: Batch Verify Weight Malleability (BVM-1..6) — Original 2026-04-28 ---
# Audits Schnorr + ECDSA batch verify correctness: order, duplicates, poison.
add_executable(test_exploit_batch_verify_malleability_standalone
    test_exploit_batch_verify_malleability.cpp)
audit_target_defaults(test_exploit_batch_verify_malleability_standalone)
target_compile_definitions(test_exploit_batch_verify_malleability_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_batch_verify_malleability COMMAND test_exploit_batch_verify_malleability_standalone)
set_tests_properties(exploit_batch_verify_malleability PROPERTIES TIMEOUT 60 LABELS "audit;exploit;batch;schnorr;malleability")

# -- N9: Thread-Local Blinding State Race (TLB-1..4) — Original 2026-04-28 --
# Validates thread-local blinding transparency: concurrent randomize+sign on
# shared ctx must produce correct, verifiable signatures for each thread.
add_executable(test_exploit_thread_local_blinding_standalone
    test_exploit_thread_local_blinding.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/src/ufsecp_impl.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/src/ufsecp_gpu_impl.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/gpu/src/gpu_registry.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/gpu/src/gpu_backend_fallback.cpp)
audit_target_defaults(test_exploit_thread_local_blinding_standalone)
target_include_directories(test_exploit_thread_local_blinding_standalone PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_BINARY_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/gpu/include)
target_compile_definitions(test_exploit_thread_local_blinding_standalone PRIVATE STANDALONE_TEST UFSECP_BUILDING)
find_package(Threads REQUIRED)
target_link_libraries(test_exploit_thread_local_blinding_standalone PRIVATE Threads::Threads)
add_test(NAME exploit_thread_local_blinding COMMAND test_exploit_thread_local_blinding_standalone)
set_tests_properties(exploit_thread_local_blinding PROPERTIES TIMEOUT 60 LABELS "audit;exploit;thread;blinding;tsan")

# -- N10: Hedged Sign Return-Value Silence (HEDGED-1..4) — Original 2026-04-28 -
# Audits fail-closed invariant: error from ecdsa_sign must leave sig zeroed,
# and hedged signing with aux produces a different sig from deterministic path.
add_executable(test_exploit_hedged_return_value_standalone
    test_exploit_hedged_return_value.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../compat/libsecp256k1_shim/src/shim_ecdsa.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../compat/libsecp256k1_shim/src/shim_context.cpp)
audit_target_defaults(test_exploit_hedged_return_value_standalone)
target_include_directories(test_exploit_hedged_return_value_standalone PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../compat/libsecp256k1_shim/include)
target_compile_definitions(test_exploit_hedged_return_value_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_hedged_return_value COMMAND test_exploit_hedged_return_value_standalone)
set_tests_properties(exploit_hedged_return_value PROPERTIES TIMEOUT 60 LABELS "audit;exploit;ecdsa;hedged;shim")

# -- N11: GPU Kernel Memory Safety (GPU-1..5) — Original 2026-04-28 ----------
# NULL/invalid input boundary tests for GPU batch APIs.
# Skips gracefully when no GPU backend is compiled or device is unavailable.
add_executable(test_exploit_gpu_memory_safety_standalone
    test_exploit_gpu_memory_safety.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/src/ufsecp_impl.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/src/ufsecp_gpu_impl.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/gpu/src/gpu_registry.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/gpu/src/gpu_backend_fallback.cpp)
audit_target_defaults(test_exploit_gpu_memory_safety_standalone)
target_include_directories(test_exploit_gpu_memory_safety_standalone PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_BINARY_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/gpu/include)
target_compile_definitions(test_exploit_gpu_memory_safety_standalone PRIVATE STANDALONE_TEST UFSECP_BUILDING)
add_test(NAME exploit_gpu_memory_safety COMMAND test_exploit_gpu_memory_safety_standalone)
set_tests_properties(exploit_gpu_memory_safety PROPERTIES
    TIMEOUT 60
    LABELS "audit;exploit;gpu;memory;boundary"
    SKIP_RETURN_CODE 77)

# -- N12: ECDSA r,s Zero Check Gap (RZERO-1..5) — Original 2026-04-28 --------
# CVE-2022-39272-class: verify must reject r=0, s=0, r=p (out-of-range).
add_executable(test_exploit_rs_zero_check_standalone
    test_exploit_rs_zero_check.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/src/ufsecp_impl.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/src/ufsecp_gpu_impl.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/gpu/src/gpu_registry.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/gpu/src/gpu_backend_fallback.cpp)
audit_target_defaults(test_exploit_rs_zero_check_standalone)
target_include_directories(test_exploit_rs_zero_check_standalone PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_BINARY_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/gpu/include)
target_compile_definitions(test_exploit_rs_zero_check_standalone PRIVATE STANDALONE_TEST UFSECP_BUILDING)
add_test(NAME exploit_rs_zero_check COMMAND test_exploit_rs_zero_check_standalone)
set_tests_properties(exploit_rs_zero_check PROPERTIES TIMEOUT 60 LABELS "audit;exploit;ecdsa;schnorr;zero;cve")

# -- N13: BIP-352 Address Collision (SP-1..4) — Original 2026-04-28 ----------
# Domain separation collision resistance: 1000 random (scan,spend) pairs
# must produce unique Silent Payment addresses.
add_executable(test_exploit_bip352_address_collision_standalone
    test_exploit_bip352_address_collision.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/src/ufsecp_impl.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/src/ufsecp_gpu_impl.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/gpu/src/gpu_registry.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/gpu/src/gpu_backend_fallback.cpp)
audit_target_defaults(test_exploit_bip352_address_collision_standalone)
target_include_directories(test_exploit_bip352_address_collision_standalone PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_BINARY_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/gpu/include)
target_compile_definitions(test_exploit_bip352_address_collision_standalone PRIVATE STANDALONE_TEST UFSECP_BUILDING)
add_test(NAME exploit_bip352_address_collision COMMAND test_exploit_bip352_address_collision_standalone)
set_tests_properties(exploit_bip352_address_collision PROPERTIES TIMEOUT 120 LABELS "audit;exploit;bip352;silent-payments;collision")

# -- BUG-001/007: addr off-by-one overflow + p2sh null ---------------------
add_executable(test_exploit_bug001_addr_overflow_standalone
    test_exploit_bug001_addr_overflow.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/src/ufsecp_impl.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/src/ufsecp_gpu_impl.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/gpu/src/gpu_registry.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/gpu/src/gpu_backend_fallback.cpp)
audit_target_defaults(test_exploit_bug001_addr_overflow_standalone)
target_include_directories(test_exploit_bug001_addr_overflow_standalone PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_BINARY_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/gpu/include)
target_compile_definitions(test_exploit_bug001_addr_overflow_standalone PRIVATE STANDALONE_TEST UFSECP_BUILDING)
add_test(NAME exploit_bug001_addr_overflow COMMAND test_exploit_bug001_addr_overflow_standalone)
set_tests_properties(exploit_bug001_addr_overflow PROPERTIES TIMEOUT 60 LABELS "audit;exploit;security;abi;buffer")

# -- BUG-002: recovery.cpp CT recid overflow check -------------------------
add_executable(test_exploit_bug002_recovery_ct_standalone test_exploit_bug002_recovery_ct.cpp)
audit_target_defaults(test_exploit_bug002_recovery_ct_standalone)
target_link_libraries(test_exploit_bug002_recovery_ct_standalone PRIVATE fastsecp256k1)
target_compile_definitions(test_exploit_bug002_recovery_ct_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_bug002_recovery_ct COMMAND test_exploit_bug002_recovery_ct_standalone)
set_tests_properties(exploit_bug002_recovery_ct PROPERTIES TIMEOUT 60 LABELS "audit;exploit;security;ct;recovery")

# -- BUG-003/008: ECDSASignature::normalize() CT path ----------------------
add_executable(test_exploit_bug003_normalize_ct_standalone test_exploit_bug003_normalize_ct.cpp)
audit_target_defaults(test_exploit_bug003_normalize_ct_standalone)
target_link_libraries(test_exploit_bug003_normalize_ct_standalone PRIVATE fastsecp256k1)
target_compile_definitions(test_exploit_bug003_normalize_ct_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_bug003_normalize_ct COMMAND test_exploit_bug003_normalize_ct_standalone)
set_tests_properties(exploit_bug003_normalize_ct PROPERTIES TIMEOUT 60 LABELS "audit;exploit;security;ct;normalize")

# -- BUG-004: Schnorr batch sign fail-closed --------------------------------
add_executable(test_exploit_bug004_batch_failclosed_standalone
    test_exploit_bug004_batch_failclosed.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/src/ufsecp_impl.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/src/ufsecp_gpu_impl.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/gpu/src/gpu_registry.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/gpu/src/gpu_backend_fallback.cpp)
audit_target_defaults(test_exploit_bug004_batch_failclosed_standalone)
target_include_directories(test_exploit_bug004_batch_failclosed_standalone PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_BINARY_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/gpu/include)
target_compile_definitions(test_exploit_bug004_batch_failclosed_standalone PRIVATE STANDALONE_TEST UFSECP_BUILDING)
add_test(NAME exploit_bug004_batch_failclosed COMMAND test_exploit_bug004_batch_failclosed_standalone)
set_tests_properties(exploit_bug004_batch_failclosed PROPERTIES TIMEOUT 60 LABELS "audit;exploit;security;batch;failclosed")

# -- HIGH-2/3: Shim DER Parser BIP-66 Compliance (DER66-1..8) --------------
# Compiles shim_ecdsa directly so shim's parse_der_int is testable.
add_executable(test_exploit_shim_der_bip66_standalone
    test_exploit_shim_der_bip66.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../compat/libsecp256k1_shim/src/shim_ecdsa.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../compat/libsecp256k1_shim/src/shim_context.cpp)
audit_target_defaults(test_exploit_shim_der_bip66_standalone)
target_include_directories(test_exploit_shim_der_bip66_standalone PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../compat/libsecp256k1_shim/include)
target_compile_definitions(test_exploit_shim_der_bip66_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_shim_der_bip66 COMMAND test_exploit_shim_der_bip66_standalone)
set_tests_properties(exploit_shim_der_bip66 PROPERTIES TIMEOUT 30 LABELS "audit;exploit;shim;der;bip66")

# -- CRIT-1: Shim MuSig2 Secnonce Reuse (MSN-1..6) -------------------------
# Compiles shim_musig + shim_ecdsa + shim_extrakeys + shim_schnorr directly.
add_executable(test_exploit_shim_musig_secnonce_standalone
    test_exploit_shim_musig_secnonce.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../compat/libsecp256k1_shim/src/shim_musig.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../compat/libsecp256k1_shim/src/shim_ecdsa.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../compat/libsecp256k1_shim/src/shim_extrakeys.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../compat/libsecp256k1_shim/src/shim_schnorr.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../compat/libsecp256k1_shim/src/shim_context.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../compat/libsecp256k1_shim/src/shim_pubkey.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../compat/libsecp256k1_shim/src/shim_seckey.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../compat/libsecp256k1_shim/src/shim_tagged_hash.cpp)
audit_target_defaults(test_exploit_shim_musig_secnonce_standalone)
target_include_directories(test_exploit_shim_musig_secnonce_standalone PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../compat/libsecp256k1_shim/include)
target_compile_definitions(test_exploit_shim_musig_secnonce_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_shim_musig_secnonce COMMAND test_exploit_shim_musig_secnonce_standalone)
set_tests_properties(exploit_shim_musig_secnonce PROPERTIES TIMEOUT 60 LABELS "audit;exploit;shim;musig;security;crit")

# -- P-01/02/03: Shim pubkey_create / keypair_create CT generator mul (SPC-1..10) --
# Tests that secp256k1_ec_pubkey_create and secp256k1_keypair_create use
# ct::generator_mul (not variable-time scalar_mul_generator) on private keys.
add_executable(test_exploit_shim_pubkey_ct_standalone
    test_exploit_shim_pubkey_ct.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../compat/libsecp256k1_shim/src/shim_pubkey.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../compat/libsecp256k1_shim/src/shim_extrakeys.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../compat/libsecp256k1_shim/src/shim_context.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../compat/libsecp256k1_shim/src/shim_seckey.cpp)
audit_target_defaults(test_exploit_shim_pubkey_ct_standalone)
target_include_directories(test_exploit_shim_pubkey_ct_standalone PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../compat/libsecp256k1_shim/include)
target_compile_definitions(test_exploit_shim_pubkey_ct_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_shim_pubkey_ct COMMAND test_exploit_shim_pubkey_ct_standalone)
set_tests_properties(exploit_shim_pubkey_ct PROPERTIES TIMEOUT 30 LABELS "audit;exploit;shim;ct;pubkey;rule12")

# -- Q-01/02/03: OpenCL audit runner key erasure (OCR-1..8) ------------------
add_executable(test_exploit_opencl_runner_key_erase_standalone
    test_exploit_opencl_runner_key_erase.cpp)
audit_target_defaults(test_exploit_opencl_runner_key_erase_standalone)
target_compile_definitions(test_exploit_opencl_runner_key_erase_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_opencl_runner_key_erase
    COMMAND test_exploit_opencl_runner_key_erase_standalone
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
set_tests_properties(exploit_opencl_runner_key_erase PROPERTIES
    TIMEOUT 30 LABELS "audit;exploit;opencl;key_erase;rule10")
# -- FPI: ecdsa fast-path isolation regression (FPI-1..10) -------------------
add_executable(test_exploit_ecdsa_fast_path_isolation_standalone
    test_exploit_ecdsa_fast_path_isolation.cpp)
audit_target_defaults(test_exploit_ecdsa_fast_path_isolation_standalone)
target_compile_definitions(test_exploit_ecdsa_fast_path_isolation_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_ecdsa_fast_path_isolation
    COMMAND test_exploit_ecdsa_fast_path_isolation_standalone
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
set_tests_properties(exploit_ecdsa_fast_path_isolation PROPERTIES
    TIMEOUT 30 LABELS "audit;exploit;ct;fast_path;regression")

# -- Source integrity scanner: known mutation artifact patterns -------------
add_executable(test_mutation_artifact_scan_standalone test_mutation_artifact_scan.cpp)
audit_target_defaults(test_mutation_artifact_scan_standalone)
target_compile_definitions(test_mutation_artifact_scan_standalone PRIVATE
    STANDALONE_TEST
    UFSECP_SOURCE_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/..")
add_test(NAME mutation_artifact_scan COMMAND test_mutation_artifact_scan_standalone)
set_tests_properties(mutation_artifact_scan PROPERTIES TIMEOUT 30 LABELS "audit;mutation;integrity")

# -- 2026-05-05 Bug Bounty regression guards (BB-01..BB-06) ----------------
# BB-01/02: FROST threshold + signing share zero key (BUG-C3/C4)
# BB-03: shim secp256k1_ecdsa_sign_recoverable ctx_can_sign (BUG-H3)
# BB-04: low-S semantics proof (BUG-C1 class)
# BB-05: CPU BIP32 depth=255 guard proxy for GPU (BUG-H1)
# BB-06: ufsecp_ecdsa_sign_recoverable always low-S (BUG-C1/C2 ABI proxy)
add_executable(test_exploit_frost_ocl_shim_bip32_ct_regression_standalone
    test_exploit_frost_ocl_shim_bip32_ct_regression.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/src/ufsecp_impl.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/src/ufsecp_gpu_impl.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/gpu/src/gpu_registry.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/gpu/src/gpu_backend_fallback.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../compat/libsecp256k1_shim/src/shim_ecdsa.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../compat/libsecp256k1_shim/src/shim_recovery.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../compat/libsecp256k1_shim/src/shim_context.cpp)
audit_target_defaults(test_exploit_frost_ocl_shim_bip32_ct_regression_standalone)
target_include_directories(test_exploit_frost_ocl_shim_bip32_ct_regression_standalone PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_BINARY_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/gpu/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../compat/libsecp256k1_shim/include)
target_compile_definitions(test_exploit_frost_ocl_shim_bip32_ct_regression_standalone PRIVATE
    STANDALONE_TEST UFSECP_BUILDING)
add_test(NAME exploit_frost_ocl_shim_bip32_ct_regression COMMAND test_exploit_frost_ocl_shim_bip32_ct_regression_standalone)
set_tests_properties(exploit_frost_ocl_shim_bip32_ct_regression PROPERTIES
    TIMEOUT 120 LABELS "audit;exploit;security;frost;shim;bip32;ct;regression")

# -- 2026-05-05 Red-Team Round 3 regression guards (RR3-01..06) ---------------
# BUG-1: musig2 secnonce not zeroed on error paths
# BUG-2: frost nonce not zeroed on early error paths
# BUG-4: ufsecp_last_error_msg thread_local path
# BUG-6: MuSig2 keyagg LE32 round-trip
add_executable(test_exploit_musig2_nonce_erasure_le32_ecdh_standalone
    test_exploit_musig2_nonce_erasure_le32_ecdh.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/src/ufsecp_impl.cpp)
audit_target_defaults(test_exploit_musig2_nonce_erasure_le32_ecdh_standalone)
target_include_directories(test_exploit_musig2_nonce_erasure_le32_ecdh_standalone PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_BINARY_DIR}/include
)
target_compile_definitions(test_exploit_musig2_nonce_erasure_le32_ecdh_standalone PRIVATE
    STANDALONE_TEST UFSECP_BUILDING)
add_test(NAME exploit_musig2_nonce_erasure_le32_ecdh
    COMMAND test_exploit_musig2_nonce_erasure_le32_ecdh_standalone)
set_tests_properties(exploit_musig2_nonce_erasure_le32_ecdh PROPERTIES
    TIMEOUT 60 LABELS "audit;exploit;security;musig2;frost;memory_safety;regression")

# -- Diagnostic: ct::scalar_mul step-by-step comparison --------------------
add_executable(diag_scalar_mul ${CPU_TESTS_DIR}/diag_scalar_mul.cpp)
audit_target_defaults(diag_scalar_mul)
target_compile_definitions(diag_scalar_mul PRIVATE STANDALONE_TEST)
add_test(NAME diag_scalar_mul COMMAND diag_scalar_mul)

# -- 2026-05-06 Perf review: Pippenger stale used[] regression (BUG-01) -----
add_executable(test_regression_pippenger_stale_used_standalone
    test_regression_pippenger_stale_used.cpp)
audit_target_defaults(test_regression_pippenger_stale_used_standalone)
target_compile_definitions(test_regression_pippenger_stale_used_standalone PRIVATE STANDALONE_TEST)
add_test(NAME regression_pippenger_stale_used
    COMMAND test_regression_pippenger_stale_used_standalone)
set_tests_properties(regression_pippenger_stale_used PROPERTIES
    TIMEOUT 60 LABELS "audit;regression;pippenger;msm;correctness")

# -- 2026-05-06 Perf review: FROST DKG non-CT secret share gen mul (SEC-01) --
add_executable(test_exploit_frost_secret_share_ct_standalone
    test_exploit_frost_secret_share_ct.cpp)
audit_target_defaults(test_exploit_frost_secret_share_ct_standalone)
target_compile_definitions(test_exploit_frost_secret_share_ct_standalone PRIVATE STANDALONE_TEST)
add_test(NAME exploit_frost_secret_share_ct
    COMMAND test_exploit_frost_secret_share_ct_standalone)
set_tests_properties(exploit_frost_secret_share_ct PROPERTIES
    TIMEOUT 60 LABELS "audit;exploit;frost;ct;dkg;secret;rule12")

# -- 2026-05-06 Perf review: comb_gen_mul lock-free thread safety (CRIT-01) --
add_executable(test_regression_comb_gen_lockfree_standalone
    test_regression_comb_gen_lockfree.cpp)
audit_target_defaults(test_regression_comb_gen_lockfree_standalone)
target_compile_definitions(test_regression_comb_gen_lockfree_standalone PRIVATE STANDALONE_TEST)
find_package(Threads REQUIRED)
target_link_libraries(test_regression_comb_gen_lockfree_standalone PRIVATE Threads::Threads)
add_test(NAME regression_comb_gen_lockfree
    COMMAND test_regression_comb_gen_lockfree_standalone)
set_tests_properties(regression_comb_gen_lockfree PROPERTIES
    TIMEOUT 60 LABELS "audit;regression;comb;ecmult_gen;thread;lockfree")

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

    # NOTE: CT vs libsecp benchmark removed -- use src/cpu/bench/bench_unified instead
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
        ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/include
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
        ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/include
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
        ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/include
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
        ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/include
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
        ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/include
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
        ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/include
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
    # -- LibFuzzer deterministic regression + mutation + formal spec (advisory) --
    test_libfuzzer_unified.cpp
    test_mutation_kill_rate.cpp
    test_cryptol_specs.cpp
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
    # -- Zeroization / ZK fault injection / secret lifecycle --
    test_zeroization.cpp
    test_fault_zk_adaptor.cpp
    test_secret_lifecycle.cpp
    # -- Field / scalar / point arithmetic edge cases --
    test_field_scalar_edge.cpp
    test_point_group_law.cpp
    # -- Security audit modules --
    audit_secure_erase.cpp
    audit_ct_namespace.cpp
    test_kat_all_operations.cpp
    test_nonce_uniqueness.cpp
    test_parse_strictness.cpp
    # -- ufsecp FFI implementation (needed by fuzz_parsers + fuzz_address) --
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/src/ufsecp_impl.cpp
    # -- ufsecp GPU ABI implementation + tests (null-guard paths work without hardware) --
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/src/ufsecp_gpu_impl.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/gpu/src/gpu_registry.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/gpu/src/gpu_backend_fallback.cpp
    test_gpu_host_api_negative.cpp
    test_gpu_abi_gate.cpp
    # -- field representation tests --
    ${CPU_TESTS_DIR}/test_field_26.cpp
    # -- diagnostics --
    ${CPU_TESTS_DIR}/diag_scalar_mul.cpp
    # -- Exploit PoC tests (Section 9) --
    test_exploit_adaptor_extended.cpp
    test_exploit_adaptor_parity.cpp
    test_exploit_address_encoding.cpp
    test_exploit_address_generation.cpp
    test_exploit_bech32_underflow.cpp
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
    test_exploit_bip39_nfkd.cpp
    test_exploit_bitcoin_message_signing.cpp
    test_exploit_chacha20_kat.cpp
    test_exploit_chacha20_nonce_reuse.cpp
    test_exploit_kat_corpus.cpp
    test_exploit_primitive_kat.cpp
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
    test_exploit_batch_verify_low_s.cpp
    test_exploit_binding_retval.cpp
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
    test_exploit_eip712_kat.cpp
    test_exploit_field_arithmetic.cpp
    test_exploit_field_boundary_exhaustive.cpp
    test_exploit_frost_binding_factor_mismatch.cpp
    test_exploit_frost_byzantine.cpp
    test_exploit_frost_commitment_reuse.cpp
    test_exploit_frost_ct_nonce.cpp
    test_exploit_frost_participant_set_malleability.cpp
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
    test_exploit_fe_set_b32_limit_uninit.cpp
    test_exploit_zk_new_schemes.cpp
    test_exploit_foreign_field_plonk.cpp
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
    test_exploit_pubkey_arith.cpp
    test_exploit_batch_sign.cpp
    test_exploit_ecdsa_nonce_reuse.cpp
    test_exploit_cross_scheme_pubkey.cpp
    test_exploit_wif_security.cpp
    test_exploit_binding_invalid_curve.cpp
    test_exploit_musig2_parallel_session_cross.cpp
    test_exploit_musig2_byzantine_multiparty.cpp
    # -- Security gap closure tests (A2, B2, A6, A7, C2, C5, D6, LOW) --
    test_exploit_buffer_type_confusion.cpp
    test_exploit_differential_libsecp.cpp
    test_exploit_differential_openssl.cpp
    test_exploit_bip352_scan_dos.cpp
    test_exploit_taproot_commitment_adversarial.cpp
    test_exploit_bip352_parity_confusion.cpp
    test_exploit_bip352_batch_correctness.cpp
    test_bip352_kat.cpp
    test_exploit_rfc6979_truncation_bias.cpp
    test_exploit_binding_adversarial_api.cpp
    test_exploit_quantum_exposure.cpp
    test_exploit_ecies_ephemeral_reuse.cpp
    test_exploit_address_prefix_collision.cpp
    test_exploit_ecdh_zvp_glv_static.cpp
    test_exploit_frost_adaptive_corruption.cpp
    test_exploit_scalar_mul.cpp
    test_exploit_ecdsa_fault_injection.cpp
    test_exploit_cache_sidechannel_amplification.cpp
    test_exploit_ladderleak_subbit_nonce.cpp
    test_exploit_minerva_noisy_hnp.cpp
    test_exploit_hertzbleed_dvfs_timing.cpp
    test_exploit_biased_nonce_chain_scan.cpp
    test_exploit_kr_ecdsa_buff_binding.cpp
    test_exploit_ellswift_xdh_overflow.cpp
    test_exploit_buff_kr_ecdsa.cpp
    test_exploit_minerva_cve_2024_23342.cpp
    test_exploit_rfc6979_minerva_amplified.cpp
    test_exploit_ellswift_bad_scalar_ecdh.cpp
    # -- Mutation residue detection + source artifact scanner --
    test_exploit_mutation_residue.cpp
    test_mutation_artifact_scan.cpp
    # -- Metal field_reduce_512 regression (issue #226) --
    test_exploit_metal_field_reduce.cpp
    # -- Network selector validation bypass --
    test_exploit_network_validation_bypass.cpp
    # -- Previously standalone-only exploit tests (unified audit gap closure) --
    test_exploit_bip32_child_key_attack.cpp
    test_exploit_boundary_sentinels.cpp
    test_exploit_ecdsa_r_overflow.cpp
    test_exploit_ecdsa_pmn_wraparound.cpp
    test_exploit_ecdsa_sign_sentinels.cpp
    test_exploit_frost_identifiable_abort.cpp
    test_exploit_hash_algo_sig_isolation.cpp
    test_exploit_schnorr_nonce_reuse.cpp
    # -- Affine nonce relation key recovery (ePrint 2025/705) --
    test_exploit_ecdsa_affine_nonce_relation.cpp
    # -- Half-half nonce key recovery (ePrint 2023/841) --
    test_exploit_ecdsa_half_half_nonce.cpp
    # -- Nonce modular reduction bias (CVE-2024-31497) --
    test_exploit_ecdsa_nonce_modular_bias.cpp
    # -- Differential fault on RFC 6979 (ePrint 2017/975) --
    test_exploit_ecdsa_differential_fault.cpp
    # -- EUCLEAK non-CT modular inversion (ePrint 2024/1380) --
    test_exploit_eucleak_inversion_timing.cpp
    # -- Cross-key nonce reuse cascade (ePrint 2025/654) --
    test_exploit_ecdsa_cross_key_nonce_reuse.cpp
    # -- Fiat-Shamir hash order in BIP-340 (ePrint 2025/1846) --
    test_exploit_schnorr_hash_order.cpp
    # -- ZVP-DCP on GLV multiscalar (ePrint 2025/076) --
    test_exploit_zvp_glv_dcp_multiscalar.cpp
    # -- Lattice sieve HNP sub-bit leakage (ePrint 2024/296) --
    test_exploit_lattice_sieve_hnp.cpp
    # -- DFA on RFC 6979 deterministic signatures (ePrint 2017/975) --
    test_exploit_deterministic_sig_dfa.cpp
    # -- Type confusion k-reuse (CVE-2024-49364/49365/2022-41340) --
    test_exploit_sign_type_confusion_kreuse.cpp
    # -- ROS concurrent blind Schnorr forgery (ePrint 2020/945) --
    test_exploit_ros_concurrent_schnorr.cpp
    # -- FROST weak binding adaptive corruption (ePrint 2026/075, 2025/1001) --
    test_exploit_frost_weak_binding.cpp
    # -- Blind-Folded SPA & cmov/cswap leak (ePrint 2024/589, 2025/935) --
    test_exploit_blind_spa_cmov_leak.cpp
    # -- ECTester point order/infinity (ePrint 2025/1293) --
    test_exploit_ectester_point_validation.cpp
    # -- Dimensional eROS & BZ blind sig forgery (ePrint 2025/306, 2025/1353) --
    test_exploit_ros_dimensional_erosion.cpp
    # -- Modified ECDSA batch verify randomization (ePrint 2026/663) --
    test_exploit_ecdsa_batch_verify_rand.cpp
    # -- BIP-324 AEAD forgery & ChaCha20-Poly1305 boundary (BIP-324, RFC 8439) --
    test_exploit_bip324_aead_forgery.cpp
    # -- FROST rogue-key attack (ePrint 2020/852, 2023/899) --
    test_exploit_frost_rogue_key.cpp
    # -- MuSig2 partial signature forgery (ePrint 2020/1261) --
    test_exploit_musig2_partial_forgery.cpp
    # -- Adaptor extraction soundness (ePrint 2020/476, 2021/150) --
    test_exploit_adaptor_extraction_soundness.cpp
    # -- ECDH twist point injection (ePrint 2015/1233) --
    test_exploit_ecdh_twist_injection.cpp
    # -- Schnorr batch verify inflation (BIP-340, ePrint 2012/549) --
    test_exploit_schnorr_batch_inflation.cpp
    # -- SafeGCD / Bernstein-Yang divstep correctness --
    test_exploit_safegcd_divsteps.cpp
    # -- Nonce edge cases: null/zero/n/n-1/determinism --
    test_exploit_custom_nonce_injection.cpp
    # -- JNI return-value ignored (RVI-1..8) — 2026-04-23 --
    # ultrafast_secp256k1.cpp provides secp256k1_sha256/hash160/tagged_hash/privkey_negate
    test_exploit_jni_retval_ignored.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../bindings/c_api/ultrafast_secp256k1.cpp
    # -- Dark Skippy nonce exfiltration (DS-1..8) — ePrint 2024/1225 --
    test_exploit_dark_skippy_exfil.cpp
    # -- Frozen Heart ZK Fiat-Shamir incomplete binding (FH-1..10) — ePrint 2022/411 --
    test_exploit_fiat_shamir_frozen_heart.cpp
    # -- Hertzbleed scalar blinding Hamming-weight defence (SB-1..9) — ePrint 2022/823 --
    test_exploit_hertzbleed_scalar_blind.cpp
    # -- Thread-safe gen_fb_table lazy init: std::once_flag regression (TIR-1..5) — 2026-04-27 --
    test_exploit_thread_unsafe_lazy_init.cpp
    # -- z_fe_nonzero | vs & correctness regression (ZFN-1..6) — 2026-04-27 --
    test_regression_z_fe_nonzero.cpp
    # -- CUDA pool minimum-capacity rounding regression (CAP-1..4) — 2026-04-27 --
    test_regression_cuda_pool_cap.cpp
    # -- musig2_partial_verify OOB + infinity-nonce regression (MVV-1..6) — 2026-04-23 --
    test_regression_musig2_verify.cpp
    # -- Bip324Session sk-leak + destructor secure-erase regression (BPS-1..8) — 2026-04-23 --
    test_regression_bip324_session.cpp
    # -- Exception-path sk/ek/entropy leakage regression (EPE-RAII + EPE-1..12) — 2026-04-23 --
    test_regression_exception_erase.cpp
    # -- Quality audit fixes (2026-04-27 Comprehensive Report) --
    # C-1: C ABI recovery signing CT path (RCTX-1..8)
    test_exploit_recoverable_sign_ct.cpp
    # M-6: Pippenger batch verify regression guard (PIPBATCH-1..8)
    test_exploit_pippenger_batch_regression.cpp
    # B-01: Ethereum signing CT path (ETHCT-1..8)
    test_exploit_eth_signing_ct.cpp
    # B-02: Wallet sign_hash CT path (WALCT-1..8)
    test_exploit_wallet_sign_ct.cpp
    # B-04: Monolith split integrity (MONO-1..12)
    test_exploit_monolith_split.cpp
    # B-08/B-09: GPU secret erase + batch sign partial
    test_exploit_gpu_secret_erase.cpp
    # L-01: libsecp256k1 EC key API compatibility
    test_exploit_libsecp_eckey_api.cpp
    # BC-01: Bitcoin Core R-grinding nonce pattern
    test_exploit_bitcoin_core_rgrinding.cpp
    # Upstream gap closure (2026-04-28) — GAP-1, GAP-3, GAP-4
    test_exploit_pubkey_cmp.cpp
    test_exploit_pubkey_sort.cpp
    test_exploit_alloc_bounds.cpp
    # Upstream gap closure (2026-04-28) — GAP-2, GAP-5, GAP-7
    test_exploit_hsort.cpp
    test_exploit_wnaf.cpp
    test_exploit_int128.cpp
    # Upstream gap closure (2026-04-28) — scratch allocator + xoshiro256**
    test_exploit_scratch.cpp
    test_exploit_xoshiro.cpp
    # Original Security Analysis (2026-04-28) — N1..N5
    test_exploit_cross_protocol_kreuse.cpp
    test_exploit_tagged_hash_ext.cpp
    test_exploit_wnaf_cache_ampl.cpp
    test_exploit_musig2_fingerprint_collision.cpp
    test_exploit_blinding_recovery_hnp.cpp
    # Original Security Analysis (2026-04-28) — N6..N8
    test_exploit_shim_noncefp_bypass.cpp
    test_exploit_encoding_memory_corruption.cpp
    test_exploit_batch_verify_malleability.cpp
    # Original Security Analysis (2026-04-28) — N9..N13
    test_exploit_thread_local_blinding.cpp
    test_exploit_hedged_return_value.cpp
    test_exploit_gpu_memory_safety.cpp
    test_exploit_rs_zero_check.cpp
    test_exploit_bip352_address_collision.cpp
    # BUG-001..008 fixes (2026-04-28 full audit)
    test_exploit_bug001_addr_overflow.cpp
    test_exploit_bug002_recovery_ct.cpp
    test_exploit_bug003_normalize_ct.cpp
    test_exploit_bug004_batch_failclosed.cpp
    # 2026-05-01 security audit fixes (shim layer — advisory-skip in unified runner)
    test_exploit_shim_der_bip66.cpp
    test_exploit_shim_musig_secnonce.cpp
    # 2026-05-01 Red Team Audit Fixes
    test_exploit_legacy_capi_key_parsing.cpp
    test_exploit_legacy_capi_degenerate_sig.cpp
    test_exploit_musig_unknown_signer.cpp
    test_exploit_bchn_schnorr_strict_parsing.cpp
    test_exploit_context_flag_bypass.cpp
    test_exploit_metal_schnorr_aux_rand.cpp
    test_exploit_metal_batch_failclosed.cpp
    test_exploit_gpu_bip352_key_erase.cpp
    test_exploit_metal_ecdh_key_erase.cpp
    # 2026-05-01 Red Team Audit Round 3 (P-01..P-09) — advisory-skip in unified runner
    test_exploit_shim_pubkey_ct.cpp
    # 2026-05-02 Red Team Audit Round 4 (Q-01..Q-03) — OpenCL audit runner key erase
    test_exploit_opencl_runner_key_erase.cpp
    # 2026-05-02 Fast-path isolation regression (FPI-1..10)
    test_exploit_ecdsa_fast_path_isolation.cpp
    # 2026-05-02 Bitcoin Core PR security audit fixes
    test_regression_schnorr_ct_arithmetic.cpp
    test_regression_musig2_zero_psig.cpp
    test_regression_bip352_ct_varbase.cpp
    # CT signing scalar correctness regression guards (PRF-1..8)
    test_regression_signing_ct_scalar_correctness.cpp
    # ABI recid+recov+schnorr_verified CT regression guards
    test_exploit_abi_recoverable_schnorr_ct_regression.cpp
    # FROST+OCL+shim+BIP32 CT regression guards
    test_exploit_frost_ocl_shim_bip32_ct_regression.cpp
    # MuSig2 secnonce/nonce erasure + LE32 + ECDH Y-parity regression guards
    test_exploit_musig2_nonce_erasure_le32_ecdh.cpp
    # 2026-05-06 Performance review: correctness + security fixes
    # BUG-01: Pippenger stale used[] across windows
    test_regression_pippenger_stale_used.cpp
    # SEC-01: FROST DKG non-CT generator mul on secret share
    test_exploit_frost_secret_share_ct.cpp
    # CRIT-01: comb_gen_mul lock-free after once_flag init
    test_regression_comb_gen_lockfree.cpp
    # V-01 timing regression: fast::Scalar operator* banned on secrets (advisory)
    test_regression_ct_fast_scalar_v01.cpp
    # TQ-005: Schnorr BIP-340 ABI edge cases (r==0, r>=p, s==0, s>=n rejection)
    test_regression_schnorr_abi_edge_cases.cpp
)
# GPU key-erase regression requires ufsecp_gpu symbols; only add when GPU is built.
if(SECP256K1_BUILD_CUDA OR SECP256K1_BUILD_OPENCL OR TARGET ufsecp_gpu_host OR TARGET secp256k1_gpu_host)
    target_sources(unified_audit_runner PRIVATE test_regression_gpu_key_erase_raii.cpp)
    target_compile_definitions(unified_audit_runner PRIVATE SECP256K1_BUILD_GPU_AUDIT)
endif()
# Conditionally add 5x52 field test (requires __uint128_t; skip on MSVC)
if(NOT (MSVC AND NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang"))
    target_sources(unified_audit_runner PRIVATE ${CPU_TESTS_DIR}/test_field_52.cpp)
endif()
# Conditionally add Ethereum layer tests
if(SECP256K1_BUILD_ETHEREUM)
    target_sources(unified_audit_runner PRIVATE ${CPU_TESTS_DIR}/test_ethereum.cpp)
endif()
target_link_libraries(unified_audit_runner PRIVATE fastsecp256k1)
find_package(Threads REQUIRED)
target_link_libraries(unified_audit_runner PRIVATE Threads::Threads)
target_include_directories(unified_audit_runner PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/ufsecp   # for ufsecp_impl.cpp "#include ufsecp.h"
    ${CMAKE_CURRENT_BINARY_DIR}/../include   # for generated version.hpp
    ${CMAKE_BINARY_DIR}/include              # fallback for version.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/gpu/include  # for gpu_backend.hpp (gpu_registry.cpp)
    ${CMAKE_CURRENT_SOURCE_DIR}/../bindings/c_api  # for ultrafast_secp256k1.h (jni_retval test)
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
    SECP256K1_BIP324
    GIT_HASH="${_git_hash}"
    UFSECP_SOURCE_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/.."
)
if(MSVC OR (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND WIN32))
    target_link_options(unified_audit_runner PRIVATE "LINKER:/STACK:8388608")
endif()
add_test(NAME unified_audit COMMAND unified_audit_runner)
set_tests_properties(unified_audit PROPERTIES TIMEOUT 1800)

# ===========================================================================
target_link_libraries(unified_audit_runner PRIVATE Threads::Threads)

# ---------------------------------------------------------------------------
# Optional INTEROP differential reference: OpenSSL libcrypto
# ---------------------------------------------------------------------------
# When OpenSSL is available on the build host, link it into the unified
# audit runner so test_exploit_differential_openssl can perform a real
# cross-implementation check. When OpenSSL is absent, the test compiles
# with __has_include guards and prints an advisory skip.
#
# Skip detection entirely when cross-compiling: find_package(OpenSSL) would
# pick up the *host* OpenSSL headers/libs which are wrong for the target
# (arm64, riscv64, etc.). The test then falls back to its advisory skip.
if(NOT CMAKE_CROSSCOMPILING)
    find_package(OpenSSL QUIET COMPONENTS Crypto)
    if(OpenSSL_FOUND)
        target_link_libraries(unified_audit_runner PRIVATE OpenSSL::Crypto)
        target_compile_definitions(unified_audit_runner PRIVATE UFSECP_HAVE_OPENSSL=1)
        message(STATUS "INTEROP: OpenSSL libcrypto wired into unified_audit_runner (advisory differential)")
    else()
        message(STATUS "INTEROP: OpenSSL libcrypto not found; differential_openssl will skip with advisory")
    endif()
else()
    message(STATUS "INTEROP: cross-compiling (target=${CMAKE_SYSTEM_PROCESSOR}); skipping OpenSSL detection, differential_openssl will skip with advisory")
endif()

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
    exploit_frost_participant_set_malleability exploit_frost_ct_nonce
    exploit_binding_invalid_curve exploit_musig2_parallel_session_cross
    exploit_musig2_byzantine_multiparty
    exploit_buffer_type_confusion exploit_differential_libsecp
    exploit_bip352_scan_dos exploit_taproot_commitment_adversarial
    exploit_bip352_parity_confusion bip352_kat exploit_rfc6979_truncation_bias
    exploit_binding_adversarial_api exploit_quantum_exposure
    exploit_ecies_ephemeral_reuse exploit_address_prefix_collision
    exploit_ecdh_zvp_glv_static exploit_frost_adaptive_corruption
    exploit_scalar_mul
    exploit_ecdsa_fault_injection exploit_cache_sidechannel_amplification
    exploit_ladderleak_subbit_nonce exploit_minerva_noisy_hnp
    exploit_hertzbleed_dvfs_timing exploit_biased_nonce_chain_scan
    exploit_kr_ecdsa_buff_binding
    exploit_ellswift_xdh_overflow exploit_buff_kr_ecdsa
    exploit_minerva_cve_2024_23342 exploit_rfc6979_minerva_amplified
    exploit_ellswift_bad_scalar_ecdh
    exploit_fe_set_b32_limit_uninit exploit_zk_new_schemes exploit_foreign_field_plonk
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

# ===========================================================================
# Python audit scripts (external-auditor-class statistical/differential tests)
# ===========================================================================
find_package(Python3 QUIET COMPONENTS Interpreter)
if(Python3_FOUND)
    set(_py_scripts_dir "${CMAKE_CURRENT_SOURCE_DIR}/../ci")
    set(_uf_lib "$<TARGET_FILE:ufsecp_shared>")
    set(_py_env "PYTHONPATH=${_py_scripts_dir}")

    add_test(NAME py_differential_crossimpl
        COMMAND ${Python3_EXECUTABLE} "${_py_scripts_dir}/differential_cross_impl.py"
                --lib "${_uf_lib}" --count 500
                --json -o "${CMAKE_BINARY_DIR}/py_differential_report.json")
    if(WIN32)
        # Windows: ctypes DLL loading + Defender scan on first load adds 3-5min overhead.
        # 360s was insufficient on CI (timed out). Raise to 600s.
        set(_py_diff_timeout 600)
    else()
        set(_py_diff_timeout 180)
    endif()
    set_tests_properties(py_differential_crossimpl PROPERTIES
        SKIP_RETURN_CODE 77
        TIMEOUT ${_py_diff_timeout}
        LABELS "audit;python;differential"
        ENVIRONMENT "${_py_env}")

    add_test(NAME py_nonce_bias
        COMMAND ${Python3_EXECUTABLE} "${_py_scripts_dir}/nonce_bias_detector.py"
                --lib "${_uf_lib}" --count 10000
                --json -o "${CMAKE_BINARY_DIR}/py_nonce_bias_report.json")
    set_tests_properties(py_nonce_bias PROPERTIES
        SKIP_RETURN_CODE 77
        TIMEOUT 300
        LABELS "audit;python;statistical"
        ENVIRONMENT "${_py_env}")

    add_test(NAME py_bip32_cka
        COMMAND ${Python3_EXECUTABLE} "${_py_scripts_dir}/bip32_cka_demo.py"
                --lib "${_uf_lib}" --paths 100
                --json -o "${CMAKE_BINARY_DIR}/py_bip32_cka_report.json")
    set_tests_properties(py_bip32_cka PROPERTIES
        SKIP_RETURN_CODE 77
        TIMEOUT 120
        LABELS "audit;python;bip32"
        ENVIRONMENT "${_py_env}")

    add_test(NAME py_rfc6979_spec
        COMMAND ${Python3_EXECUTABLE} "${_py_scripts_dir}/rfc6979_spec_verifier.py"
                --lib "${_uf_lib}" --count 200
                --json -o "${CMAKE_BINARY_DIR}/py_rfc6979_report.json")
    set_tests_properties(py_rfc6979_spec PROPERTIES
        SKIP_RETURN_CODE 77
        TIMEOUT 120
        LABELS "audit;python;rfc6979"
        ENVIRONMENT "${_py_env}")

    add_test(NAME py_glv_exhaustive
        COMMAND ${Python3_EXECUTABLE} "${_py_scripts_dir}/glv_exhaustive_check.py"
                --lib "${_uf_lib}" --extra-random 5000
                --json -o "${CMAKE_BINARY_DIR}/py_glv_report.json")
    set_tests_properties(py_glv_exhaustive PROPERTIES
        SKIP_RETURN_CODE 77
        TIMEOUT 300
        LABELS "audit;python;glv"
        ENVIRONMENT "${_py_env}")

    add_test(NAME py_semantic_props
        COMMAND ${Python3_EXECUTABLE} "${_py_scripts_dir}/semantic_props.py"
                --lib "${_uf_lib}" --count 200
                --json -o "${CMAKE_BINARY_DIR}/py_semantic_props_report.json")
    set_tests_properties(py_semantic_props PROPERTIES
        SKIP_RETURN_CODE 77
        TIMEOUT 300
        LABELS "audit;python;algebraic"
        ENVIRONMENT "${_py_env}")

    add_test(NAME py_invalid_input_grammar
        COMMAND ${Python3_EXECUTABLE} "${_py_scripts_dir}/invalid_input_grammar.py"
                --lib "${_uf_lib}"
                --json -o "${CMAKE_BINARY_DIR}/py_invalid_input_report.json")
    set_tests_properties(py_invalid_input_grammar PROPERTIES
        SKIP_RETURN_CODE 77
        TIMEOUT 60
        LABELS "audit;python;input_validation"
        ENVIRONMENT "${_py_env}")

    add_test(NAME py_stateful_sequences
        COMMAND ${Python3_EXECUTABLE} "${_py_scripts_dir}/stateful_sequences.py"
                --lib "${_uf_lib}" --count 200
                --json -o "${CMAKE_BINARY_DIR}/py_stateful_sequences_report.json")
    set_tests_properties(py_stateful_sequences PROPERTIES
        SKIP_RETURN_CODE 77
        TIMEOUT 300
        LABELS "audit;python;stateful"
        ENVIRONMENT "${_py_env}")

    # Static dev-bug scanner — no library needed, scans C++ source files for classic
    # development bugs: sizeof(ptr), off-by-one, signed/unsigned, truncation, null-after-deref, etc.
    # Runs at MEDIUM level and emits a JSON report; does NOT fail CI so it serves as a
    # review aid rather than a hard gate (until all findings are triaged and fixed).
    add_test(NAME py_dev_bug_scan
        COMMAND ${Python3_EXECUTABLE} "${_py_scripts_dir}/dev_bug_scanner.py"
                --min-severity MEDIUM
                --json -o "${CMAKE_BINARY_DIR}/py_dev_bug_report.json")
    set_tests_properties(py_dev_bug_scan PROPERTIES
        SKIP_RETURN_CODE 77
        TIMEOUT 60
        LABELS "audit;python;static_analysis"
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/..")

    # Hot-path allocation scanner — no library needed, scans C++ source files for
    # heap allocations (new, malloc, std::vector local construction, push_back, std::string)
    # inside performance-critical functions (sign, verify, ecdh, scalar_mul, derive, etc.).
    # Heap allocations on hot paths violate the zero-allocation-per-op rule and degrade
    # throughput.  Does NOT fail CI until findings are triaged (serves as a tracking tool).
    add_test(NAME py_hot_path_alloc_scan
        COMMAND ${Python3_EXECUTABLE} "${_py_scripts_dir}/hot_path_alloc_scanner.py"
                --min-severity MEDIUM
                --json -o "${CMAKE_BINARY_DIR}/py_hot_path_alloc_report.json")
    set_tests_properties(py_hot_path_alloc_scan PROPERTIES
        SKIP_RETURN_CODE 77
        TIMEOUT 60
        LABELS "audit;python;static_analysis;performance"
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/..")

    message(STATUS "  Python audit scripts: 10 tests registered (py_differential_crossimpl, py_nonce_bias, py_bip32_cka, py_rfc6979_spec, py_glv_exhaustive, py_semantic_props, py_invalid_input_grammar, py_stateful_sequences, py_dev_bug_scan, py_hot_path_alloc_scan)")
else()
    message(STATUS "  Python3 not found — skipping Python audit scripts")
endif()

# ===========================================================================
# AI Findings Label Convention
# ===========================================================================
# Tests that were suggested by AI tools and have NOT yet been manually
# validated by a human auditor must carry the "ai_finding" label.  They
# remain in the audit suite and are executed by CI, but
# scripts/audit_ai_findings.py filters them out of the official confirmed
# audit score so that unverified AI suggestions do not inflate coverage
# claims.
#
# How to mark an AI-suggested test:
#
#   set_tests_properties(<test_name> PROPERTIES
#       LABELS "audit;ai_finding;unconfirmed;<other-labels>")
#
# Once a human auditor has reviewed and validated the test, remove the
# "ai_finding" and "unconfirmed" labels, leaving only the substantive
# category labels (e.g. "audit;exploit;schnorr").
#
# Query unconfirmed tests at any time:
#   ctest -N -L ai_finding
# ===========================================================================

# ===========================================================================
# LibFuzzer harnesses (parser boundary + AEAD + BIP-32 path fuzzing)
# ===========================================================================
#
# These harnesses expose LLVMFuzzerTestOneInput() for use with LibFuzzer /
# OSS-Fuzz.  They also build in standalone mode (SECP256K1_STANDALONE_FUZZ)
# for deterministic CI runs without the LibFuzzer runtime.
#
# LibFuzzer mode (requires Clang + -fsanitize=fuzzer):
#   cmake -S . -B build-fuzz -DSECP256K1_BUILD_LIBFUZZER=ON \
#         -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
#         -DCMAKE_CXX_FLAGS="-fsanitize=fuzzer,address,undefined"
#   cmake --build build-fuzz --target fuzz_der_parse fuzz_pubkey_parse \
#         fuzz_schnorr_verify fuzz_ecdsa_verify fuzz_bip32_path fuzz_bip324_frame
#   ./build-fuzz/audit/fuzz_der_parse  corpus/
#
# Standalone / deterministic CTest mode (no LibFuzzer required):
#   cmake -S . -B build -DSECP256K1_BUILD_LIBFUZZER_STANDALONE=ON
#   ctest -R "^fuzz_"
#
# ===========================================================================

# ===========================================================================
# 2026-05-02: Bitcoin Core PR security audit regression tests (standalone)
# ===========================================================================

# HIGH-03 + HIGH-06: Schnorr CT arithmetic + r==0 rejection
add_executable(test_regression_schnorr_ct_arithmetic_standalone
    test_regression_schnorr_ct_arithmetic.cpp)
audit_target_defaults(test_regression_schnorr_ct_arithmetic_standalone)
target_compile_definitions(test_regression_schnorr_ct_arithmetic_standalone PRIVATE STANDALONE_TEST)
add_test(NAME regression_schnorr_ct_arithmetic COMMAND test_regression_schnorr_ct_arithmetic_standalone)
set_tests_properties(regression_schnorr_ct_arithmetic PROPERTIES TIMEOUT 60
    LABELS "audit;exploit;schnorr;ct;regression")

# CRIT-03: musig2 zero partial-sig → UFSECP_ERR_INTERNAL
add_executable(test_regression_musig2_zero_psig_standalone
    test_regression_musig2_zero_psig.cpp)
audit_target_defaults(test_regression_musig2_zero_psig_standalone)
target_compile_definitions(test_regression_musig2_zero_psig_standalone PRIVATE STANDALONE_TEST)
add_test(NAME regression_musig2_zero_psig COMMAND test_regression_musig2_zero_psig_standalone)
set_tests_properties(regression_musig2_zero_psig PROPERTIES TIMEOUT 60
    LABELS "audit;exploit;musig2;regression")

# CRIT-01+HIGH-01+HIGH-02+HIGH-04: GPU key erasure RAII
# Only build when GPU host library is available — the source includes ufsecp_gpu.h
# and links ufsecp_gpu_ctx_create/ufsecp_gpu_ecdh_batch/ufsecp_gpu_ctx_destroy.
# Without secp256k1_gpu_host those symbols are undefined → linker error on CPU-only CI.
if(TARGET secp256k1_gpu_host OR SECP256K1_BUILD_OPENCL OR SECP256K1_BUILD_CUDA)
    add_executable(test_regression_gpu_key_erase_raii_standalone
        test_regression_gpu_key_erase_raii.cpp)
    audit_target_defaults(test_regression_gpu_key_erase_raii_standalone)
    target_link_libraries(test_regression_gpu_key_erase_raii_standalone PRIVATE secp256k1_gpu_host)
    target_include_directories(test_regression_gpu_key_erase_raii_standalone PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../src/gpu/include)
    target_compile_definitions(test_regression_gpu_key_erase_raii_standalone PRIVATE STANDALONE_TEST)
    add_test(NAME regression_gpu_key_erase_raii COMMAND test_regression_gpu_key_erase_raii_standalone)
    set_tests_properties(regression_gpu_key_erase_raii PROPERTIES TIMEOUT 120
        SKIP_RETURN_CODE 77
        LABELS "audit;memory_safety;gpu;regression")
endif()

# CRIT-02: BIP-352 CT variable-base scalar mul
add_executable(test_regression_bip352_ct_varbase_standalone
    test_regression_bip352_ct_varbase.cpp)
audit_target_defaults(test_regression_bip352_ct_varbase_standalone)
if(TARGET secp256k1_gpu_host)
    target_link_libraries(test_regression_bip352_ct_varbase_standalone PRIVATE secp256k1_gpu_host)
    target_include_directories(test_regression_bip352_ct_varbase_standalone PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../src/gpu/include)
endif()
target_compile_definitions(test_regression_bip352_ct_varbase_standalone PRIVATE STANDALONE_TEST)
add_test(NAME regression_bip352_ct_varbase COMMAND test_regression_bip352_ct_varbase_standalone)
set_tests_properties(regression_bip352_ct_varbase PROPERTIES TIMEOUT 120
    SKIP_RETURN_CODE 77
    LABELS "audit;ct_analysis;bip352;regression")

option(SECP256K1_BUILD_LIBFUZZER
       "Build LibFuzzer harnesses (requires clang -fsanitize=fuzzer,address)" OFF)

option(SECP256K1_BUILD_LIBFUZZER_STANDALONE
       "Build LibFuzzer harnesses in standalone deterministc mode (no fuzzer runtime)" OFF)

# Helper: list of all LibFuzzer harness sources
set(_FUZZ_HARNESSES
    fuzz_der_parse
    fuzz_pubkey_parse
    fuzz_schnorr_verify
    fuzz_ecdsa_verify
    fuzz_bip32_path
    fuzz_bip324_frame
)

# Helper macro: shared include/link setup for a fuzz harness target
macro(fuzz_target_defaults target_name)
    target_link_libraries(${target_name} PRIVATE ufsecp_static fastsecp256k1)
    target_include_directories(${target_name} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
        ${CMAKE_BINARY_DIR}/include/ufsecp
        ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpu/include
    )
    target_compile_definitions(${target_name} PRIVATE "UFSECP_API=")
    if(MSVC OR (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND WIN32))
        target_link_options(${target_name} PRIVATE "LINKER:/STACK:8388608")
    endif()
    if(SECP256K1_BUILD_CUDA)
        set_target_properties(${target_name} PROPERTIES INTERPROCEDURAL_OPTIMIZATION FALSE)
    endif()
endmacro()

# ---------------------------------------------------------------------------
# A) LibFuzzer mode (-fsanitize=fuzzer) — fuzz binary only, no CTest entry
# ---------------------------------------------------------------------------
if(SECP256K1_BUILD_LIBFUZZER AND TARGET ufsecp_static)

    # Verify the compiler supports -fsanitize=fuzzer
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-fsanitize=fuzzer" _COMPILER_HAS_FUZZER)
    if(NOT _COMPILER_HAS_FUZZER)
        message(WARNING "  -fsanitize=fuzzer not supported by ${CMAKE_CXX_COMPILER} — "
                        "LibFuzzer harnesses will not be built.  Use Clang.")
    else()
        foreach(_harness ${_FUZZ_HARNESSES})
            add_executable(${_harness} ${_harness}.cpp)
            fuzz_target_defaults(${_harness})
            # Link against LibFuzzer; do NOT register a CTest entry (fuzzer runs
            # interactively against a corpus directory, not with a fixed timeout).
            target_compile_options(${_harness} PRIVATE -fsanitize=fuzzer,address,undefined)
            target_link_options(${_harness}    PRIVATE -fsanitize=fuzzer,address,undefined)
        endforeach()
        message(STATUS "  LibFuzzer harnesses: ${_FUZZ_HARNESSES}")
    endif()
endif()

# ---------------------------------------------------------------------------
# B) Standalone / deterministic mode — registered as CTest targets
# ---------------------------------------------------------------------------
if(SECP256K1_BUILD_LIBFUZZER_STANDALONE AND TARGET ufsecp_static)

    foreach(_harness ${_FUZZ_HARNESSES})
        set(_target "${_harness}_standalone")
        add_executable(${_target} ${_harness}.cpp)
        fuzz_target_defaults(${_target})
        target_compile_definitions(${_target} PRIVATE
            SECP256K1_STANDALONE_FUZZ
            LIBFUZZER_STANDALONE
        )
        add_test(NAME ${_harness} COMMAND ${_target})
        set_tests_properties(${_harness} PROPERTIES
            TIMEOUT 300
            LABELS "audit;fuzz;libfuzzer_standalone"
        )
    endforeach()

    message(STATUS "  LibFuzzer standalone harnesses: ${_FUZZ_HARNESSES} (${CMAKE_CURRENT_BINARY_DIR})")
endif()

