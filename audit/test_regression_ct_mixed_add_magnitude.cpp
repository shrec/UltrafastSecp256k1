// ============================================================================
// test_regression_ct_mixed_add_magnitude.cpp
// ============================================================================
// Regression test for the point_add_mixed_complete magnitude contract bug.
//
// Root Cause (May 2026)
//   point_add_mixed_complete(CTJacobianPoint P, CTAffinePoint Q) copied
//   P.x and P.y directly into local FE52 u1/s1 without calling
//   normalize_weak().  CTJacobianPoint coordinates can carry large
//   magnitudes after iterative scalar-mul chains.  The Brier-Joye formula
//   then calls t_val.negate(4) where t_val = u1 + u2; if the actual
//   magnitude of t_val exceeded 4 the negate was invalid, producing a
//   wrong output coordinate.  This silently broke:
//     - generator_mul_blinded (signing blinding path)
//     - Schnorr / ECDSA batch verify
//     - Any caller passing a high-magnitude Jacobian point
//
// Detection: unified_audit_runner showed 37 failing modules when a test
//   that called ufsecp_context_randomize (enabling blinding) ran before
//   test_exploit_backend_divergence.
//
// Fix: X1.normalize_weak(); Y1.normalize_weak() at the start of both
//   point_add_mixed_complete implementations (FE52 + 4x64 paths).
//
// Guard: this test must be wired to unified_audit_runner so the same bug
//   cannot regress silently.  It exercises:
//     1. point_add_mixed_complete correctness with high-magnitude inputs
//     2. generator_mul_blinded == generator_mul after context_randomize
//     3. The two invariants that directly detect the original 37 failures:
//        (a) 2G+3G=5G via mixed add
//        (b) ct::ecdsa_sign_recoverable == ecdsa_sign_recoverable after
//            blinding is active

#include "secp256k1/ct/point.hpp"
#include "secp256k1/ct/sign.hpp"
#include "secp256k1/recovery.hpp"
#include "ufsecp/ufsecp.h"
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <array>
#include <random>

static int g_pass = 0, g_fail = 0;
#include "audit_check.hpp"

using secp256k1::fast::Scalar;
using secp256k1::fast::Point;
using namespace secp256k1::ct;

static Scalar scalar_from_int(uint64_t n) {
    std::array<uint8_t,32> b{};
    for (int i = 0; i < 8; i++) b[31-i] = (n >> (8*i)) & 0xFF;
    return Scalar::from_bytes(b);
}

// ── (1) point_add_mixed_complete: nG + mG == (n+m)G ------------------
static void test_mixed_add_simple() {
    printf("[1] point_add_mixed_complete: 2G+3G=5G, 7G+11G=18G, 100G+1G=101G\n");

    auto check_sum = [&](uint64_t a, uint64_t b) {
        Point pa = generator_mul(scalar_from_int(a));
        Point pb = generator_mul(scalar_from_int(b));
        Point pe = generator_mul(scalar_from_int(a+b));

        CTJacobianPoint ja = CTJacobianPoint::from_point(pa);
        CTAffinePoint   ab = CTAffinePoint::from_point(pb);
        CTJacobianPoint js = point_add_mixed_complete(ja, ab);
        Point result = js.to_point();

        auto exp = pe.to_uncompressed();
        auto got = result.to_uncompressed();
        char msg[128];
        std::snprintf(msg, sizeof(msg),
            "%lluG + %lluG = %lluG via point_add_mixed_complete",
            (unsigned long long)a, (unsigned long long)b, (unsigned long long)(a+b));
        CHECK(std::memcmp(exp.data(), got.data(), 65) == 0, msg);
    };

    check_sum(2, 3);
    check_sum(7, 11);
    check_sum(100, 1);
    check_sum(123456, 789012);
}

// ── (2) generator_mul_blinded == generator_mul with known seed --------
static void test_blinded_equals_unblinded() {
    printf("[2] generator_mul_blinded == generator_mul (5 vectors, blinding active)\n");

    ufsecp_ctx* ctx = nullptr;
    if (ufsecp_ctx_create(&ctx) != UFSECP_OK) {
        printf("  [SKIP] ctx_create failed\n");
        return;
    }

    uint8_t seed[32] = {
        0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08,
        0x09,0x0a,0x0b,0x0c,0x0d,0x0e,0x0f,0x10,
        0x11,0x12,0x13,0x14,0x15,0x16,0x17,0x18,
        0x19,0x1a,0x1b,0x1c,0x1d,0x1e,0x1f,0x20
    };
    if (ufsecp_context_randomize(ctx, seed) != UFSECP_OK) {
        printf("  [SKIP] context_randomize failed\n");
        ufsecp_ctx_destroy(ctx);
        return;
    }

    std::mt19937_64 rng(0xBEEFCAFEDEAD1234ULL);
    for (int i = 0; i < 5; ++i) {
        std::array<uint8_t,32> kb;
        for (auto& x : kb) x = (uint8_t)rng();
        Scalar k = Scalar::from_bytes(kb);

        Point plain   = generator_mul(k);
        Point blinded = generator_mul_blinded(k);

        auto ep = plain.to_uncompressed();
        auto eg = blinded.to_uncompressed();
        char msg[128];
        std::snprintf(msg, sizeof(msg),
            "generator_mul(k) == generator_mul_blinded(k) [vector %d]", i+1);
        CHECK(std::memcmp(ep.data(), eg.data(), 65) == 0, msg);
    }
    ufsecp_ctx_destroy(ctx);
}

// ── (3) ct::ecdsa_sign_recoverable == ecdsa_sign_recoverable ----------
// (the original detector: these diverged when blinding was active)
static void test_ct_fast_recoverable_agree() {
    printf("[3] ct::ecdsa_sign_recoverable == ecdsa_sign_recoverable (10 pairs, blinding active)\n");

    ufsecp_ctx* ctx = nullptr;
    if (ufsecp_ctx_create(&ctx) != UFSECP_OK) {
        printf("  [SKIP] ctx_create failed\n");
        return;
    }

    uint8_t seed[32] = {
        0xDE,0xAD,0xBE,0xEF,0xCA,0xFE,0xBA,0xBE,
        0x01,0x23,0x45,0x67,0x89,0xAB,0xCD,0xEF,
        0xFE,0xDC,0xBA,0x98,0x76,0x54,0x32,0x10,
        0x0F,0x1E,0x2D,0x3C,0x4B,0x5A,0x69,0x78
    };
    if (ufsecp_context_randomize(ctx, seed) != UFSECP_OK) {
        printf("  [SKIP] context_randomize failed\n");
        ufsecp_ctx_destroy(ctx);
        return;
    }

    std::mt19937_64 rng(0x1234567890ABCDEFULL);
    for (int i = 0; i < 10; ++i) {
        std::array<uint8_t,32> kb, mb;
        for (auto& x : kb) x = (uint8_t)rng();
        for (auto& x : mb) x = (uint8_t)rng();
        Scalar k = Scalar::from_bytes(kb);

        auto ct_rec   = secp256k1::ct::ecdsa_sign_recoverable(mb, k);
        auto fast_rec = secp256k1::ecdsa_sign_recoverable(mb, k);

        char msg[128];
        std::snprintf(msg, sizeof(msg), "pair %d: ct vs fast recoverable sig.r identical", i+1);
        CHECK(std::memcmp(ct_rec.sig.r.to_bytes().data(),
                          fast_rec.sig.r.to_bytes().data(), 32) == 0, msg);

        std::snprintf(msg, sizeof(msg), "pair %d: ct vs fast recoverable sig.s identical", i+1);
        CHECK(std::memcmp(ct_rec.sig.s.to_bytes().data(),
                          fast_rec.sig.s.to_bytes().data(), 32) == 0, msg);

        std::snprintf(msg, sizeof(msg), "pair %d: ct vs fast recoverable recid identical", i+1);
        CHECK(ct_rec.recid == fast_rec.recid, msg);
    }
    ufsecp_ctx_destroy(ctx);
}

// ── Entry point ----------------------------------------------------------
int test_regression_ct_mixed_add_magnitude_run() {
    g_pass = 0; g_fail = 0;
    printf("==================================================================\n");
    printf("  Regression: point_add_mixed_complete magnitude contract\n");
    printf("  (fix: normalize_weak X1/Y1 before Brier-Joye formula)\n");
    printf("==================================================================\n");

    test_mixed_add_simple();
    printf("\n");
    test_blinded_equals_unblinded();
    printf("\n");
    test_ct_fast_recoverable_agree();
    printf("\n");

    printf("[regression_ct_mixed_add_magnitude] %d/%d checks passed\n",
           g_pass, g_pass + g_fail);
    return (g_fail > 0) ? 1 : 0;
}

#ifndef UNIFIED_AUDIT_RUNNER
int main() { return test_regression_ct_mixed_add_magnitude_run(); }
#endif
