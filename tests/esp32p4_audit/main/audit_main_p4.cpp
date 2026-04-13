/**
 * ESP32-P4 Unified Audit Runner -- Full secp256k1 verification for RISC-V target
 *
 * Adapted from audit/unified_audit_runner.cpp and tests/esp32_audit/main/audit_main.cpp
 * for the ESP32-P4 (dual-core RISC-V HP @ 400 MHz, rv32imafc).
 *
 * Runs all 42 audit modules (40 base + 2 new field/scalar/point edge-case modules).
 * Skips: SIMD-specific, exhaustive, comprehensive, FFI round-trip.
 *
 * API: v3.16.0
 * Target: ESP32-P4 (RISC-V HP 400 MHz)
 * Build: ESP-IDF v5.4.0
 */
#include <cstdio>
#include <cstdint>
#include <cstring>

#ifdef ESP_PLATFORM
#include "esp_log.h"
#include "esp_system.h"
#include "esp_timer.h"
#include "esp_heap_caps.h"
static const char* TAG = "secp256k1_audit_p4";
#define LOG(fmt, ...) ESP_LOGI(TAG, fmt, ##__VA_ARGS__)
#else
#define LOG(fmt, ...) printf("[AUDIT_P4] " fmt "\n", ##__VA_ARGS__)
#endif

// ============================================================================
// Forward declarations -- all _run() functions from audit + cpu/tests sources
// ============================================================================

// Section 1: Mathematical Invariants
int audit_field_run();
int audit_scalar_run();
int audit_point_run();
int test_mul_run();
int test_arithmetic_correctness_run();
int test_large_scalar_multiplication_run();
int test_ecc_properties_run();
int test_batch_add_affine_run();
int test_carry_propagation_run();
int test_field_26_main();

// Section 2: Constant-Time / Side-Channel
int audit_ct_run();
int test_ct_run();
int test_ct_equivalence_run();
int test_ct_sidechannel_smoke_run();
int diag_scalar_mul_run();

// Section 3: Differential & Cross-Library
int test_differential_run();
int test_fiat_crypto_vectors_run();
int test_cross_platform_kat_run();

// Section 4: Standard Test Vectors
int test_bip340_vectors_run();
int test_bip340_strict_run();
int test_bip32_vectors_run();
int test_rfc6979_vectors_run();
int test_frost_kat_run();
int test_musig2_bip327_vectors_run();

// Section 5: Fuzzing & Adversarial
int test_audit_fuzz_run();
int test_fault_injection_run();

// Section 6: Protocol Security
int test_ecdsa_schnorr_run();
int test_bip32_run();
int test_musig2_run();
int test_ecdh_recovery_taproot_run();
int test_v4_features_run();
int test_coins_run();
int test_musig2_frost_protocol_run();
int test_musig2_frost_advanced_run();
int audit_integration_run();

// Section 7: ABI & Memory Safety
int audit_security_run();
int test_debug_invariants_run();
int test_abi_gate_run();

// Section 8: Performance Validation
int test_multiscalar_batch_run();
int audit_perf_run();

// Section 9: Field / Scalar / Point arithmetic edge cases (added 2025-04-11)
int test_field_scalar_edge_run();
int test_point_group_law_run();

// ============================================================================
// Module table
// ============================================================================
struct AuditModule {
    const char* id;
    const char* name;
    const char* section;
    int (*run)();
    bool advisory;
};

static const AuditModule ALL_MODULES[] = {
    // Section 1: Mathematical Invariants
    { "audit_field",       "Field Fp deep audit",            "math",     audit_field_run, false },
    { "audit_scalar",      "Scalar Zn deep audit",           "math",     audit_scalar_run, false },
    { "audit_point",       "Point ops deep audit",           "math",     audit_point_run, false },
    { "mul",               "Field & scalar arithmetic",      "math",     test_mul_run, false },
    { "arith_correct",     "Arithmetic correctness",         "math",     test_arithmetic_correctness_run, false },
    { "scalar_mul",        "Scalar multiplication",          "math",     test_large_scalar_multiplication_run, false },
    { "ecc_properties",    "ECC property-based invariants",  "math",     test_ecc_properties_run, false },
    { "batch_add",         "Affine batch addition",          "math",     test_batch_add_affine_run, false },
    { "carry_propagation", "Carry chain stress",             "math",     test_carry_propagation_run, false },
    { "field_26",          "FieldElement26 (10x26)",         "math",     test_field_26_main, false },

    // Section 2: CT / Side-Channel
    { "audit_ct",          "CT deep audit",                  "ct",       audit_ct_run, false },
    { "ct",                "Constant-time layer",            "ct",       test_ct_run, false },
    { "ct_equivalence",    "FAST == CT equivalence",         "ct",       test_ct_equivalence_run, false },
    { "ct_sidechannel",    "Side-channel dudect (smoke)",    "ct",       test_ct_sidechannel_smoke_run, true },
    { "diag_scalar_mul",   "CT scalar_mul diagnostic",       "ct",       diag_scalar_mul_run, false },

    // Section 3: Differential
    { "differential",      "Differential correctness",       "diff",     test_differential_run, false },
    { "fiat_crypto",       "Fiat-Crypto reference vectors",  "diff",     test_fiat_crypto_vectors_run, false },
    { "cross_platform_kat","Cross-platform KAT",             "diff",     test_cross_platform_kat_run, false },

    // Section 4: Standard Vectors
    { "bip340_vectors",    "BIP-340 official vectors",       "vectors",  test_bip340_vectors_run, false },
    { "bip340_strict",     "BIP-340 strict encoding",        "vectors",  test_bip340_strict_run, false },
    { "bip32_vectors",     "BIP-32 official vectors TV1-5",  "vectors",  test_bip32_vectors_run, false },
    { "rfc6979_vectors",   "RFC 6979 ECDSA vectors",         "vectors",  test_rfc6979_vectors_run, false },
    { "frost_kat",         "FROST reference KAT vectors",    "vectors",  test_frost_kat_run, false },
    { "musig2_bip327",     "MuSig2 BIP-327 vectors",         "vectors",  test_musig2_bip327_vectors_run, false },

    // Section 5: Fuzzing & Adversarial
    { "audit_fuzz",        "Adversarial fuzz",               "fuzz",     test_audit_fuzz_run, false },
    { "fault_injection",   "Fault injection simulation",     "fuzz",     test_fault_injection_run, false },

    // Section 6: Protocol Security
    { "ecdsa_schnorr",     "ECDSA + Schnorr",                "proto",    test_ecdsa_schnorr_run, false },
    { "bip32",             "BIP-32 HD derivation",           "proto",    test_bip32_run, false },
    { "musig2",            "MuSig2",                         "proto",    test_musig2_run, false },
    { "ecdh_recovery",     "ECDH + recovery + taproot",      "proto",    test_ecdh_recovery_taproot_run, false },
    { "v4_features",       "v4 (Pedersen/FROST/adaptor)",    "proto",    test_v4_features_run, false },
    { "coins",             "Coins layer",                    "proto",    test_coins_run, false },
    { "musig2_frost",      "MuSig2 + FROST protocol",        "proto",    test_musig2_frost_protocol_run, false },
    { "musig2_frost_adv",  "MuSig2 + FROST adversarial",    "proto",    test_musig2_frost_advanced_run, false },
    { "audit_integration", "Integration (cross-proto)",      "proto",    audit_integration_run, false },

    // Section 7: Memory Safety
    { "audit_security",    "Security hardening",             "safety",   audit_security_run, false },
    { "debug_invariants",  "Debug invariant assertions",     "safety",   test_debug_invariants_run, false },
    { "abi_gate",          "ABI version gate",               "safety",   test_abi_gate_run, false },

    // Section 8: Performance
    { "multiscalar",       "Multi-scalar & batch verify",    "perf",     test_multiscalar_batch_run, false },
    { "audit_perf",        "Performance smoke",              "perf",     audit_perf_run, false },

    // Section 9: Field / Scalar / Point arithmetic edge cases
    { "field_scalar_edge", "Field/Scalar boundary & carry",  "math",     test_field_scalar_edge_run, false },
    { "point_group_law",   "Point group axioms (Jacobian)",  "math",     test_point_group_law_run, false },
};

static constexpr int NUM_MODULES = sizeof(ALL_MODULES) / sizeof(ALL_MODULES[0]);
static constexpr int TOTAL_MODULES_IN_SUITE = 48;

// ============================================================================
// ESP32-P4 entry point
// ============================================================================
#ifdef ESP_PLATFORM
extern "C" void app_main(void)
#else
int main()
#endif
{
    LOG("+==========================================================+");
    LOG("|  UltrafastSecp256k1 ESP32-P4 Audit Runner v3.16.0      |");
    LOG("|  Audit Framework v2.0.0 -- RISC-V P4 Adaptation        |");
    LOG("+==========================================================+");
    LOG("|  Target: ESP32-P4 (RISC-V HP, 400 MHz)                 |");
    LOG("|  Arch:   rv32imafc / SECP256K1_NO_ASM / 26-bit field   |");
    LOG("|  Modules: %d / %d                                   |",
        NUM_MODULES, TOTAL_MODULES_IN_SUITE);
    LOG("+==========================================================+");

#ifdef ESP_PLATFORM
    LOG("Free heap: %lu bytes", (unsigned long)esp_get_free_heap_size());
    LOG("Min free heap: %lu bytes", (unsigned long)esp_get_minimum_free_heap_size());
#endif

    int total_pass = 0;
    int total_fail = 0;
    int advisory_fail = 0;

    for (int i = 0; i < NUM_MODULES; ++i) {
        const auto& m = ALL_MODULES[i];

        LOG("");
        LOG("------------------------------------------------------------");
        LOG("[%d/%d] %s -- %s", i + 1, NUM_MODULES, m.id, m.name);
        LOG("------------------------------------------------------------");

#ifdef ESP_PLATFORM
        int64_t t0 = esp_timer_get_time();
#endif

        int result = m.run();

#ifdef ESP_PLATFORM
        int64_t t1 = esp_timer_get_time();
        int64_t elapsed_ms = (t1 - t0) / 1000;
        LOG("  Time: %lld ms | Heap: %lu bytes free",
            (long long)elapsed_ms,
            (unsigned long)esp_get_free_heap_size());
#endif

        if (result == 0) {
            total_pass++;
            LOG("  -> PASS");
        } else {
            if (m.advisory) {
                advisory_fail++;
                LOG("  -> FAIL (advisory, does not block verdict)");
            } else {
                total_fail++;
                LOG("  -> FAIL");
            }
        }
    }

    LOG("");
    LOG("============================================================");
    LOG("  ESP32-P4 AUDIT RESULTS");
    LOG("============================================================");
    LOG("  Modules tested: %d", NUM_MODULES);
    LOG("  PASSED:         %d", total_pass);
    LOG("  FAILED:         %d", total_fail);
    LOG("  Advisory FAIL:  %d", advisory_fail);
    LOG("  Skipped (N/A):  %d", TOTAL_MODULES_IN_SUITE - NUM_MODULES);
    LOG("  Verdict:        %s", total_fail == 0 ? "AUDIT-READY" : "FAIL");
    LOG("  Platform:       ESP32-P4 RISC-V HP | ESP-IDF 5.4.0");
    LOG("============================================================");

#ifdef ESP_PLATFORM
    LOG("Final free heap: %lu bytes", (unsigned long)esp_get_free_heap_size());
    LOG("Min free heap:   %lu bytes", (unsigned long)esp_get_minimum_free_heap_size());
#endif

    LOG("ESP32P4_AUDIT_COMPLETE");

#ifndef ESP_PLATFORM
    return total_fail > 0 ? 1 : 0;
#endif
}
