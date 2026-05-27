// ============================================================================
// test_regression_ecdh_off_curve.cpp
// ============================================================================
// Regression for SEC-005: ecdh_compute / ecdh_compute_xonly / ecdh_compute_raw
// did not validate that the supplied public key lies on the secp256k1 curve
// (y² = x³ + 7 mod p).  An attacker could supply a point on a weak twist curve
// to bias private-key bits through differential analysis (ePrint 2015/1233).
//
// The fix adds an explicit on-curve check (and is_infinity check) in all three
// ecdh_compute* variants before invoking ct::scalar_mul.
//
// Tests (OCK-1..5):
//   OCK-1  off-curve pubkey rejected by ecdh_compute        (returns all-zeros)
//   OCK-2  off-curve pubkey rejected by ecdh_compute_xonly  (returns all-zeros)
//   OCK-3  off-curve pubkey rejected by ecdh_compute_raw    (returns all-zeros)
//   OCK-4  point-at-infinity rejected by ecdh_compute       (returns all-zeros)
//   OCK-5  valid on-curve pubkey still produces non-zero output (positive guard)
// ============================================================================

#ifndef UNIFIED_AUDIT_RUNNER
#include <cstdio>
#define STANDALONE_TEST
#endif

#include <cstdio>
#include <cstring>
#include <cstdint>
#include <array>

static int g_pass = 0, g_fail = 0;
#include "audit_check.hpp"

#include "secp256k1/ecdh.hpp"
#include "secp256k1/scalar.hpp"
#include "secp256k1/point.hpp"
#include "secp256k1/field.hpp"

using namespace secp256k1;
using secp256k1::fast::Scalar;
using secp256k1::fast::Point;
using secp256k1::fast::FieldElement;

// Private key = 7 (small, valid, far from zero and group order).
static const std::array<uint8_t, 32> kSkBytes = {{
    0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,7
}};

static Scalar make_sk() { return Scalar::from_bytes(kSkBytes); }

// Build an off-curve Point: take G.x() and G.y() + 1.
// For this to still be on-curve would require Gy = (p-1)/2, which is false.
static Point make_off_curve_point() {
    auto G = Point::generator();
    auto gx = G.x();
    auto gy_plus_1 = G.y() + FieldElement::from_uint64(1);
    return Point::from_affine(gx, gy_plus_1);
}

static bool all_zeros(const std::array<uint8_t, 32>& arr) {
    for (auto b : arr) if (b) return false;
    return true;
}

// ─── OCK-1: off-curve pubkey rejected by ecdh_compute ────────────────────────
static void test_ock1_off_curve_ecdh_compute() {
    auto sk = make_sk();
    auto off_curve = make_off_curve_point();
    auto result = ecdh_compute(sk, off_curve);
    CHECK(all_zeros(result),
          "[OCK-1] ecdh_compute with off-curve pubkey returns all-zeros (SEC-005)");
}

// ─── OCK-2: off-curve pubkey rejected by ecdh_compute_xonly ──────────────────
static void test_ock2_off_curve_ecdh_compute_xonly() {
    auto sk = make_sk();
    auto off_curve = make_off_curve_point();
    auto result = ecdh_compute_xonly(sk, off_curve);
    CHECK(all_zeros(result),
          "[OCK-2] ecdh_compute_xonly with off-curve pubkey returns all-zeros (SEC-005)");
}

// ─── OCK-3: off-curve pubkey rejected by ecdh_compute_raw ────────────────────
static void test_ock3_off_curve_ecdh_compute_raw() {
    auto sk = make_sk();
    auto off_curve = make_off_curve_point();
    auto result = ecdh_compute_raw(sk, off_curve);
    CHECK(all_zeros(result),
          "[OCK-3] ecdh_compute_raw with off-curve pubkey returns all-zeros (SEC-005)");
}

// ─── OCK-4: point-at-infinity rejected ───────────────────────────────────────
static void test_ock4_infinity_rejected() {
    auto sk = make_sk();
    auto inf = Point::infinity();
    auto result = ecdh_compute(sk, inf);
    CHECK(all_zeros(result),
          "[OCK-4] ecdh_compute with point-at-infinity returns all-zeros");
}

// ─── OCK-5: valid pubkey still produces non-zero output (positive guard) ─────
static void test_ock5_valid_key_works() {
    auto sk = make_sk();
    auto pk = Point::generator().scalar_mul(make_sk());  // 7*G is a valid pubkey
    auto result = ecdh_compute(sk, pk);
    CHECK(!all_zeros(result),
          "[OCK-5] ecdh_compute with valid on-curve pubkey produces non-zero output");
}

int test_regression_ecdh_off_curve_run() {
    g_pass = 0; g_fail = 0;
    std::printf("[regression_ecdh_off_curve] SEC-005: off-curve pubkey rejection in all ecdh_compute* variants\n");

    test_ock1_off_curve_ecdh_compute();
    test_ock2_off_curve_ecdh_compute_xonly();
    test_ock3_off_curve_ecdh_compute_raw();
    test_ock4_infinity_rejected();
    test_ock5_valid_key_works();

    std::printf("  pass=%d  fail=%d\n", g_pass, g_fail);
    return (g_fail == 0) ? 0 : 1;
}

#ifdef STANDALONE_TEST
int main() { return test_regression_ecdh_off_curve_run(); }
#endif
