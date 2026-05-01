# ============================================================================
# ESP32-P4 Unified Audit -- CMakeLists.txt
# Target: ESP32-C6 (single-core RISC-V LP @ 160 MHz, rv32imac)
# Compiles all audit modules for cross-platform verification on RISC-V.
#
# Key differences from esp32_audit (Xtensa S3):
#   - SECP256K1_NO_ASM=1 (no inline asm on RISC-V in this port)
#   - NO field_asm.cpp (ARM/x86 asm file, not needed on RISC-V)
#   - Otherwise identical source list and audit coverage
# ============================================================================

set(CPU_SRC "${CMAKE_CURRENT_SOURCE_DIR}/../../../src/cpu/src")
set(CPU_TESTS "${CMAKE_CURRENT_SOURCE_DIR}/../../../src/cpu/tests")
set(AUDIT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../audit")

idf_component_register(
    SRCS "audit_main_c6.cpp"
         # ===============================================================
         # Library source files (32-bit safe, ESP32-P4 RISC-V compatible)
         # field_asm.cpp is intentionally omitted: RISC-V build uses only
         # portable C paths via SECP256K1_NO_ASM=1.
         # ===============================================================
         "${CPU_SRC}/field.cpp"
         "${CPU_SRC}/field_26.cpp"
         "${CPU_SRC}/field_52.cpp"

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
         "${CPU_SRC}/wallet.cpp"
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
         # Field / scalar / point arithmetic edge-case audit (added 2025-04-11)
         "${AUDIT_DIR}/test_field_scalar_edge.cpp"
         "${AUDIT_DIR}/test_point_group_law.cpp"
    INCLUDE_DIRS "."
)

# Add UltrafastSecp256k1 include paths
target_include_directories(${COMPONENT_LIB} PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../src/cpu/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../src/cpu/src"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../include"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../audit"
)

# C++20 for the component
target_compile_features(${COMPONENT_LIB} PRIVATE cxx_std_20)

# ESP32-C6 is RISC-V 32-bit: no __int128, no inline ASM, 26-bit field path.
# UNIFIED_AUDIT_RUNNER suppresses standalone main() in test files.
# DUDECT_SMOKE uses shorter iteration counts for side-channel tests.
# SECP256K1_PLATFORM_ESP32 triggers reduced counts in sanitizer_scale.hpp.
target_compile_definitions(${COMPONENT_LIB} PUBLIC
    SECP256K1_FIELD_26BIT=1
    SECP256K1_NO_INT128=1
    SECP256K1_NO_ASM=1
    SECP256K1_PLATFORM_ESP32=1
    UNIFIED_AUDIT_RUNNER=1
    DUDECT_SMOKE=1
)

# Enable exceptions (needed by some audit modules, e.g. audit_security)
target_compile_options(${COMPONENT_LIB} PRIVATE -fexceptions)
