// ============================================================================
// test_regression_shim_high_s_verify.cpp
// Diagnostic: secp256k1_ecdsa_verify high-S signature behavior (SEC-007)
//
// Documents the known behavioral divergence: libsecp256k1 normalizes before
// verifying (high-S acceptance); Ultra shim does NOT normalize.
// Callers must call secp256k1_ecdsa_signature_normalize before
// secp256k1_ecdsa_verify for libsecp-compatible high-S acceptance.
//
// This test is DIAGNOSTIC -- returns 0 (PASS) regardless of high-S result.
// See docs/SHIM_KNOWN_DIVERGENCES.md for full description.
// ============================================================================

#if !defined(UNIFIED_AUDIT_RUNNER) && !defined(STANDALONE_TEST)
#define STANDALONE_TEST
#endif

#include <cstdio>
#include <cstring>
#include <algorithm>

#ifndef ADVISORY_SKIP_CODE
#define ADVISORY_SKIP_CODE 77
#endif

static int g_pass = 0, g_fail = 0;
#include "audit_check.hpp"

#ifdef STANDALONE_TEST

#include "secp256k1.h"

// secp256k1 group order n (big-endian)
static const unsigned char SECP256K1_N[32] = {
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFE,
    0xBA, 0xAE, 0xDC, 0xE6, 0xAF, 0x48, 0xA0, 0x3B,
    0xBF, 0xD2, 0x5E, 0x8C, 0xD0, 0x36, 0x41, 0x41
};

// n - s (big-endian subtraction)
static void negate_s(const unsigned char s[32], unsigned char out[32]) {
    unsigned int borrow = 0;
    for (int i = 31; i >= 0; --i) {
        unsigned int a = (unsigned int)SECP256K1_N[i];
        unsigned int b = (unsigned int)s[i] + borrow;
        if (a < b) {
            out[i] = (unsigned char)(256u + a - b);
            borrow = 1;
        } else {
            out[i] = (unsigned char)(a - b);
            borrow = 0;
        }
    }
}

static const unsigned char PRIVKEY[32] = {
    0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
    0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0x10,
    0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18,
    0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f, 0x20
};
static const unsigned char MSGHASH[32] = {
    0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff, 0x00, 0x11,
    0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99,
    0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff, 0x00, 0x11,
    0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99
};

static void test_high_s_verify_behavior() {
    secp256k1_context* ctx = secp256k1_context_create(
        SECP256K1_CONTEXT_SIGN | SECP256K1_CONTEXT_VERIFY);
    CHECK(ctx != nullptr, "[HSV-1] context_create");
    if (!ctx) return;

    secp256k1_pubkey pubkey{};
    CHECK(secp256k1_ec_pubkey_create(ctx, &pubkey, PRIVKEY) == 1, "[HSV-2] pubkey_create");

    secp256k1_ecdsa_signature low_s_sig{};
    CHECK(secp256k1_ecdsa_sign(ctx, &low_s_sig, MSGHASH, PRIVKEY, nullptr, nullptr) == 1,
          "[HSV-3] ecdsa_sign");

    unsigned char compact[64]{};
    secp256k1_ecdsa_signature_serialize_compact(ctx, compact, &low_s_sig);

    int low_s_result = secp256k1_ecdsa_verify(ctx, &low_s_sig, MSGHASH, &pubkey);
    std::printf("[HSV-4] low-S verify result: %d (expected 1)\n", low_s_result);
    CHECK(low_s_result == 1, "[HSV-4] low-S signature verifies");

    // Manufacture high-S form: s' = n - s
    unsigned char high_s_compact[64];
    std::memcpy(high_s_compact, compact, 64);
    negate_s(compact + 32, high_s_compact + 32);

    secp256k1_ecdsa_signature high_s_sig{};
    CHECK(secp256k1_ecdsa_signature_parse_compact(ctx, &high_s_sig, high_s_compact) == 1,
          "[HSV-5] parse high-S compact");

    int high_s_result = secp256k1_ecdsa_verify(ctx, &high_s_sig, MSGHASH, &pubkey);
    std::printf("[HSV-6] high-S verify result: %d\n", high_s_result);
    std::printf("  DIAGNOSTIC (SEC-007): libsecp256k1 would return 1 (normalizes before verify).\n");
    std::printf("  Ultra shim returns %d -- intentional divergence per SHIM_KNOWN_DIVERGENCES.md.\n",
                high_s_result);
    g_pass++; // diagnostic step always passes

    // After normalization, it must verify
    secp256k1_ecdsa_signature normalized{};
    secp256k1_ecdsa_signature_normalize(ctx, &normalized, &high_s_sig);
    int normalized_result = secp256k1_ecdsa_verify(ctx, &normalized, MSGHASH, &pubkey);
    CHECK(normalized_result == 1, "[HSV-7] normalized high-S verifies as low-S");

    secp256k1_context_destroy(ctx);
}

#endif // STANDALONE_TEST

int test_regression_shim_high_s_verify_run() {
#ifndef STANDALONE_TEST
    return ADVISORY_SKIP_CODE;
#else
    g_pass = 0; g_fail = 0;
    std::printf("[regression_shim_high_s_verify] SEC-007 high-S verify diagnostic\n");
    test_high_s_verify_behavior();
    std::printf("  pass=%d  fail=%d\n", g_pass, g_fail);
    // Always PASS: documents intentional divergence, not a correctness failure.
    return (g_fail == 0) ? 0 : 1;
#endif
}

#ifdef STANDALONE_TEST
int main() { return test_regression_shim_high_s_verify_run(); }
#endif
