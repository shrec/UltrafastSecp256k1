# ============================================================================
# ESP32 Unified Audit -- CMakeLists.txt
# Compiles all 40/48 audit modules for ESP32-S3 (Xtensa LX7)
# ============================================================================

set(CPU_SRC "${CMAKE_CURRENT_SOURCE_DIR}/../../../cpu/src")
set(CPU_TESTS "${CMAKE_CURRENT_SOURCE_DIR}/../../../cpu/tests")
set(AUDIT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../audit")

idf_component_register(
    SRCS "audit_main.cpp"
         # ===============================================================
         # Library source files (32-bit safe, ESP32-compatible)
         # ===============================================================
         "${CPU_SRC}/field.cpp"
         "${CPU_SRC}/field_26.cpp"
         "${CPU_SRC}/field_52.cpp"
         "${CPU_SRC}/field_asm.cpp"
         "${CPU_SRC}/field_simd.cpp"
         "${CPU_SRC}/scalar.cpp"
         "${CPU_SRC}/point.cpp"
         "${CPU_SRC}/glv.cpp"
         "${CPU_SRC}/precompute.cpp"
         "${CPU_SRC}/ecdsa.cpp"
         "${CPU_SRC}/schnorr.cpp"
         "${CPU_SRC}/musig2.cpp"
         "${CPU_SRC}/frost.cpp"
         "${CPU_SRC}/bip32.cpp"
         "${CPU_SRC}/pedersen.cpp"
         "${CPU_SRC}/ecdh.cpp"
         "${CPU_SRC}/recovery.cpp"
         "${CPU_SRC}/taproot.cpp"
         "${CPU_SRC}/ct_field.cpp"
         "${CPU_SRC}/ct_scalar.cpp"
         "${CPU_SRC}/ct_point.cpp"
         "${CPU_SRC}/ct_sign.cpp"
         "${CPU_SRC}/hash_accel.cpp"
         "${CPU_SRC}/keccak256.cpp"
         "${CPU_SRC}/batch_add_affine.cpp"
         "${CPU_SRC}/batch_verify.cpp"
         "${CPU_SRC}/multiscalar.cpp"
         "${CPU_SRC}/pippenger.cpp"
         "${CPU_SRC}/selftest.cpp"
         "${CPU_SRC}/adaptor.cpp"
         "${CPU_SRC}/address.cpp"
         "${CPU_SRC}/ethereum.cpp"
         "${CPU_SRC}/coin_address.cpp"
         "${CPU_SRC}/coin_hd.cpp"
         # ===============================================================
         # CPU test sources (provide _run() functions)
         # ===============================================================
         "${CPU_TESTS}/test_mul.cpp"
         "${CPU_TESTS}/test_arithmetic_correctness.cpp"
         "${CPU_TESTS}/test_large_scalar_multiplication.cpp"
         "${CPU_TESTS}/test_ecc_properties.cpp"
         "${CPU_TESTS}/test_batch_add_affine.cpp"
         "${CPU_TESTS}/test_field_26.cpp"
         "${CPU_TESTS}/test_ct.cpp"
         "${CPU_TESTS}/test_ct_equivalence.cpp"
         "${CPU_TESTS}/diag_scalar_mul.cpp"
         "${CPU_TESTS}/test_bip340_vectors.cpp"
         "${CPU_TESTS}/test_bip340_strict.cpp"
         "${CPU_TESTS}/test_bip32_vectors.cpp"
         "${CPU_TESTS}/test_rfc6979_vectors.cpp"
         "${CPU_TESTS}/test_ecdsa_schnorr.cpp"
         "${CPU_TESTS}/test_bip32.cpp"
         "${CPU_TESTS}/test_musig2.cpp"
         "${CPU_TESTS}/test_ecdh_recovery_taproot.cpp"
         "${CPU_TESTS}/test_v4_features.cpp"
         "${CPU_TESTS}/test_coins.cpp"
         "${CPU_TESTS}/test_multiscalar_batch.cpp"
         # ===============================================================
         # Audit module sources (provide _run() functions)
         # ===============================================================
         "${AUDIT_DIR}/audit_field.cpp"
         "${AUDIT_DIR}/audit_scalar.cpp"
         "${AUDIT_DIR}/audit_point.cpp"
         "${AUDIT_DIR}/audit_ct.cpp"
         "${AUDIT_DIR}/audit_integration.cpp"
         "${AUDIT_DIR}/audit_security.cpp"
         "${AUDIT_DIR}/audit_perf.cpp"
         "${AUDIT_DIR}/audit_fuzz.cpp"
         "${AUDIT_DIR}/test_carry_propagation.cpp"
         "${AUDIT_DIR}/test_fault_injection.cpp"
         "${AUDIT_DIR}/test_fiat_crypto_vectors.cpp"
         "${AUDIT_DIR}/test_cross_platform_kat.cpp"
         "${AUDIT_DIR}/test_debug_invariants.cpp"
         "${AUDIT_DIR}/test_abi_gate.cpp"
         "${AUDIT_DIR}/test_ct_sidechannel.cpp"
         "${AUDIT_DIR}/differential_test.cpp"
         "${AUDIT_DIR}/test_musig2_frost.cpp"
         "${AUDIT_DIR}/test_musig2_frost_advanced.cpp"
         "${AUDIT_DIR}/test_frost_kat.cpp"
         "${AUDIT_DIR}/test_musig2_bip327_vectors.cpp"
    INCLUDE_DIRS "."
)

# Add UltrafastSecp256k1 include paths
target_include_directories(${COMPONENT_LIB} PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../cpu/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../cpu/src"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../include"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../audit"
)

# C++20 for the component
target_compile_features(${COMPONENT_LIB} PUBLIC cxx_std_20)

# ESP32 is 32-bit Xtensa: no __int128, no 64-bit ASM, no x86 SIMD
# UNIFIED_AUDIT_RUNNER suppresses standalone main() in test files
# DUDECT_SMOKE uses shorter iteration counts for side-channel test
# SECP256K1_PLATFORM_ESP32 enables ESP32-specific code paths in library
target_compile_definitions(${COMPONENT_LIB} PUBLIC
    SECP256K1_FIELD_26BIT=1
    SECP256K1_NO_INT128=1
    SECP256K1_PLATFORM_ESP32=1
    UNIFIED_AUDIT_RUNNER=1
    DUDECT_SMOKE=1
)

# Enable exceptions (needed by some audit modules, e.g. audit_security)
target_compile_options(${COMPONENT_LIB} PRIVATE -fexceptions)

