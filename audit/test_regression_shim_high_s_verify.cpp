// ============================================================================
// test_regression_shim_high_s_verify.cpp
// Regression: secp256k1_ecdsa_verify must REJECT high-S signatures (SEC-007 / SHIM-008)
//
// bitcoin-core/libsecp256k1's secp256k1_ecdsa_verify rejects high-S signatures
// (s > n/2): its final return is `(!secp256k1_scalar_is_high(&s) && ...)`. This
// blocks the malleated (r, n-s) twin of a valid signature from verifying.
//
// The Ultra shim previously delegated straight to the raw-math core
// secp256k1::ecdsa_verify (which accepts both s and n-s by design) and therefore
// ACCEPTED high-S — an undocumented divergence from upstream (and from the engine's
// own ufsecp_ecdsa_verify / ecdsa_batch_verify, which both enforce BIP-62 low-S).
// The earlier diagnostic version of this test mislabeled upstream's behavior as
// "high-S acceptance"; that was incorrect.
//
// Fix: shim_ecdsa.cpp secp256k1_ecdsa_verify now rejects high-S (is_low_s() guard),
// matching upstream exactly. This test pins that behavior.
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

    // The canonical low-S signature MUST verify.
    int low_s_result = secp256k1_ecdsa_verify(ctx, &low_s_sig, MSGHASH, &pubkey);
    CHECK(low_s_result == 1, "[HSV-4] low-S signature verifies");

    // Manufacture the high-S twin: s' = n - s  (> n/2). parse_compact accepts s' < n.
    unsigned char high_s_compact[64];
    std::memcpy(high_s_compact, compact, 64);
    negate_s(compact + 32, high_s_compact + 32);

    secp256k1_ecdsa_signature high_s_sig{};
    CHECK(secp256k1_ecdsa_signature_parse_compact(ctx, &high_s_sig, high_s_compact) == 1,
          "[HSV-5] parse high-S compact");

    // libsecp parity: secp256k1_ecdsa_verify REJECTS the high-S twin (malleability).
    int high_s_result = secp256k1_ecdsa_verify(ctx, &high_s_sig, MSGHASH, &pubkey);
    CHECK(high_s_result == 0, "[HSV-6] shim REJECTS high-S (libsecp parity, BIP-62 malleability)");

    // After normalization the signature is low-S again and MUST verify.
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
    std::printf("[regression_shim_high_s_verify] SEC-007/SHIM-008 high-S rejection (libsecp parity)\n");
    test_high_s_verify_behavior();
    std::printf("  pass=%d  fail=%d\n", g_pass, g_fail);
    return (g_fail == 0) ? 0 : 1;
#endif
}

#ifdef STANDALONE_TEST
int main() { return test_regression_shim_high_s_verify_run(); }
#endif
