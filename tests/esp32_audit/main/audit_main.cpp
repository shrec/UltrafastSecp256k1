/**
 * ESP32 Unified Audit Runner -- Full secp256k1 verification for Xtensa targets
 *
 * Adapted from audit/unified_audit_runner.cpp for ESP32-S3
 * Runs 40/48 audit modules (skips: field_52, SIMD, hash_accel,
 *   exhaustive, comprehensive, FFI round-trip, fuzz_parsers, fuzz_addr_bip32)
 *
 * API: v3.16.0
 * Target: ESP32-S3 (Xtensa LX7, 240 MHz)
 * Build: ESP-IDF v5.5.1
 */
#include <cstdio>
#include <cstdint>
#include <cstring>

#ifdef ESP_PLATFORM
#include "esp_log.h"
#include "esp_system.h"
#include "esp_timer.h"
#include "esp_heap_caps.h"
static const char* TAG = "secp256k1_audit";
#define LOG(fmt, ...) ESP_LOGI(TAG, fmt, ##__VA_ARGS__)
#else
#define LOG(fmt, ...) printf("[AUDIT] " fmt "\n", ##__VA_ARGS__)
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

// ============================================================================
// Module table
// ============================================================================
struct AuditModule {
    const char* id;
    const char* name;
    const char* section;
    int (*run)();
    bool advisory;  // if true, FAIL does not block audit verdict
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
    { "musig2",            "MuSig2",                          "proto",    test_musig2_run, false },
    { "ecdh_recovery",     "ECDH + recovery + taproot",      "proto",    test_ecdh_recovery_taproot_run, false },
    { "v4_features",       "v4 (Pedersen/FROST/adaptor)",    "proto",    test_v4_features_run, false },
    { "coins",             "Coins layer",                    "proto",    test_coins_run, false },
    { "musig2_frost",      "MuSig2 + FROST protocol",        "proto",    test_musig2_frost_protocol_run, false },
    { "musig2_frost_adv",  "MuSig2 + FROST adversarial",    "proto",    test_musig2_frost_advanced_run, false },
    { "audit_integration", "Integration (cross-proto)",       "proto",    audit_integration_run, false },

    // Section 7: Memory Safety
    { "audit_security",    "Security hardening",              "safety",   audit_security_run, false },
    { "debug_invariants",  "Debug invariant assertions",      "safety",   test_debug_invariants_run, false },
    { "abi_gate",          "ABI version gate",                "safety",   test_abi_gate_run, false },

    // Section 8: Performance
    { "multiscalar",       "Multi-scalar & batch verify",     "perf",     test_multiscalar_batch_run, false },
    { "audit_perf",        "Performance smoke",               "perf",     audit_perf_run, false },
};

static constexpr int NUM_MODULES = sizeof(ALL_MODULES) / sizeof(ALL_MODULES[0]);

// ============================================================================
// ESP32 entry point
// ============================================================================
#ifdef ESP_PLATFORM
extern "C" void app_main(void)
#else
int main()
#endif
{
    LOG("+==========================================================+");
    LOG("|  UltrafastSecp256k1 ESP32 Audit Runner v3.16.0          |");
    LOG("|  Audit Framework v2.0.0 -- ESP32 Adaptation             |");
    LOG("+==========================================================+");
#ifdef CONFIG_IDF_TARGET_ESP32S3
    LOG("|  Target: ESP32-S3 (Xtensa LX7, 240 MHz)                |");
#elif defined(CONFIG_IDF_TARGET_ESP32)
    LOG("|  Target: ESP32 (Xtensa LX6)                             |");
#else
    LOG("|  Target: Generic                                        |");
#endif
    LOG("|  Modules: %d / 48 (skip: field_52/SIMD/hash_accel/     |", NUM_MODULES);
    LOG("|           exhaustive/comprehensive/FFI)                  |");
    LOG("+==========================================================+");

#ifdef ESP_PLATFORM
    // Print free heap at start
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
    LOG("  ESP32 AUDIT RESULTS");
    LOG("============================================================");
    LOG("  Modules tested: %d", NUM_MODULES);
    LOG("  PASSED:         %d", total_pass);
    LOG("  FAILED:         %d", total_fail);
    LOG("  Advisory FAIL:  %d", advisory_fail);
    LOG("  Skipped (N/A):  %d", 48 - NUM_MODULES);
    LOG("  Verdict:        %s", total_fail == 0 ? "AUDIT-READY" : "FAIL");
    LOG("  Platform:       ESP32-S3 Xtensa LX7 | ESP-IDF 5.5.1");
    LOG("============================================================");

#ifdef ESP_PLATFORM
    LOG("Final free heap: %lu bytes", (unsigned long)esp_get_free_heap_size());
    LOG("Min free heap:   %lu bytes", (unsigned long)esp_get_minimum_free_heap_size());
#endif

    LOG("ESP32_AUDIT_COMPLETE");

#ifndef ESP_PLATFORM
    return total_fail > 0 ? 1 : 0;
#endif
}
