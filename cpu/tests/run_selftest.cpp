// ============================================================================
// Unified Test Runner — UltrafastSecp256k1
// ============================================================================
// Single binary that runs the library selftest + all test modules.
// Build once, run everywhere. Registers as a single ctest target.
//
// Usage:
//   run_selftest              # ci mode (default)
//   run_selftest smoke        # fast startup check
//   run_selftest stress       # extended nightly suite
//   run_selftest ci <seed>    # explicit seed (hex)
// ============================================================================

#include "secp256k1/selftest.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>

using namespace secp256k1::fast;

// ── Forward declarations — each test module exports a run function ───────────
// Returns 0 on success, non-zero on failure.
int test_large_scalar_multiplication_run();
int test_mul_run();
int test_arithmetic_correctness_run();
int test_ct_run();
int test_ecdsa_schnorr_run();
int test_multiscalar_batch_run();
int test_bip32_run();
int test_musig2_run();
int test_ecdh_recovery_taproot_run();
int test_simd_batch_run();
int test_v4_features_run();
int test_coins_run();
int test_batch_add_affine_run();
int test_hash_accel_run();

// ── Module descriptor ────────────────────────────────────────────────────────
struct TestModule {
    const char* name;
    int (*run)();
};

static const TestModule MODULES[] = {
    { "field & scalar arithmetic",        test_mul_run },
    { "arithmetic correctness",           test_arithmetic_correctness_run },
    { "scalar multiplication",            test_large_scalar_multiplication_run },
    { "constant-time layer",              test_ct_run },
    { "ECDSA + Schnorr",                  test_ecdsa_schnorr_run },
    { "multi-scalar & batch verify",      test_multiscalar_batch_run },
    { "BIP-32 HD derivation",             test_bip32_run },
    { "MuSig2",                           test_musig2_run },
    { "ECDH + recovery + taproot",        test_ecdh_recovery_taproot_run },
    { "SIMD batch",                       test_simd_batch_run },
    { "v4 features (Pedersen/FROST/etc)", test_v4_features_run },
    { "coins layer",                      test_coins_run },
    { "affine batch addition",             test_batch_add_affine_run },
    { "accelerated hashing",                test_hash_accel_run },
};

static constexpr int NUM_MODULES = sizeof(MODULES) / sizeof(MODULES[0]);

// ── Main ─────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    // Parse mode
    SelftestMode mode = SelftestMode::ci;
    uint64_t seed = 0;

    if (argc >= 2) {
        if (std::strcmp(argv[1], "smoke") == 0) mode = SelftestMode::smoke;
        else if (std::strcmp(argv[1], "stress") == 0) mode = SelftestMode::stress;
        else if (std::strcmp(argv[1], "ci") == 0) mode = SelftestMode::ci;
    }
    if (argc >= 3) {
        seed = std::strtoull(argv[2], nullptr, 16);
    }

    const char* mode_name = (mode == SelftestMode::smoke) ? "smoke"
                          : (mode == SelftestMode::stress) ? "stress"
                          : "ci";

    std::printf("===============================================================\n");
    std::printf("  UltrafastSecp256k1 — Unified Test Runner (%s)\n", mode_name);
    std::printf("===============================================================\n\n");

    // ── Phase 1: Library selftest (core arithmetic KAT) ──────────────────────
    std::printf("[Phase 1] Library selftest (%s)...\n", mode_name);
    if (!Selftest(true, mode, seed)) {
        std::printf("\n*** SELFTEST FAILED — aborting ***\n");
        return 1;
    }
    std::printf("[Phase 1] Selftest PASSED\n\n");

    // ── Phase 2: Test modules ────────────────────────────────────────────────
    std::printf("[Phase 2] Running %d test modules...\n\n", NUM_MODULES);

    int modules_passed = 0;
    int modules_failed = 0;

    for (int i = 0; i < NUM_MODULES; ++i) {
        std::printf("── Module %d/%d: %s ──\n", i + 1, NUM_MODULES, MODULES[i].name);
        int rc = MODULES[i].run();
        if (rc == 0) {
            ++modules_passed;
            std::printf("── PASSED ──\n\n");
        } else {
            ++modules_failed;
            std::printf("── FAILED ──\n\n");
        }
    }

    // ── Summary ──────────────────────────────────────────────────────────────
    std::printf("===============================================================\n");
    std::printf("  Results: %d/%d modules passed (selftest + %d modules)\n",
                modules_passed, NUM_MODULES, NUM_MODULES);
    if (modules_failed == 0) {
        std::printf("  ALL TESTS PASSED\n");
    } else {
        std::printf("  *** %d MODULE(S) FAILED ***\n", modules_failed);
    }
    std::printf("===============================================================\n");

    return modules_failed > 0 ? 1 : 0;
}
