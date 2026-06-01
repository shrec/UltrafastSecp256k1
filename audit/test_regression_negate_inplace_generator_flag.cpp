// ============================================================================
// REGRESSION: Point::negate_inplace() must clear is_generator_ (B1)
// ============================================================================
// Bug: negate_inplace() negated Y across all platform paths but never cleared
// is_generator_, while the non-inplace negate() DID clear it. So a negated
// generator kept is_gen() == true even though it is now -G, not G.
//
// Why it is a correctness bug: Point::scalar_mul() dispatches the fixed-base
// generator fast path on this->is_generator_:
//     if (is_generator_) return scalar_mul_generator(scalar);   // computes k*G
// A negated generator with a stale is_generator_==true therefore returns k*G
// instead of k*(-G) == -(k*G) — a wrong elliptic-curve result.
//
// Tests:
//   NEG-1: negate() on the generator clears is_gen()           (was already OK)
//   NEG-2: negate_inplace() on the generator clears is_gen()   (the B1 fix)
//   NEG-3: double negate_inplace() round-trips the point value
//   NEG-4: (-G).scalar_mul(k) == -(G.scalar_mul(k))            (the real bug)
//          — would return k*G (wrong) if is_generator_ stayed true
//   NEG-5: (-G).scalar_mul(k) != G.scalar_mul(k)               (sanity)
// ============================================================================

#include <cstdio>
#include <cstdint>

#include "secp256k1/scalar.hpp"
#include "secp256k1/point.hpp"

using secp256k1::fast::Scalar;
using secp256k1::fast::Point;

static int g_pass = 0, g_fail = 0;
static void check(bool cond, const char* msg) {
    if (cond) { ++g_pass; std::printf("    [OK] %s\n", msg); }
    else       { ++g_fail; std::printf("  [FAIL] %s\n", msg); }
    std::fflush(stdout);
}

static bool points_equal(const Point& A, const Point& B) {
    if (A.is_infinity() && B.is_infinity()) return true;
    if (A.is_infinity() || B.is_infinity()) return false;
    return A.to_compressed() == B.to_compressed();
}

static Scalar scalar_from_seed(uint64_t seed) {
    uint8_t buf[32]{};
    for (int i = 0; i < 8; ++i) buf[i] = static_cast<uint8_t>((seed >> (i * 8)) & 0xFF);
    buf[31] = 0x01; // ensure non-zero
    return Scalar::from_bytes(buf);
}

#ifndef UNIFIED_AUDIT_RUNNER
int main() {
#else
int test_regression_negate_inplace_generator_flag_run() {
#endif
    std::printf("====================================================================\n");
    std::printf("REGRESSION: negate_inplace() must clear is_generator_ (B1)\n");
    std::printf("====================================================================\n\n");

    // NEG-1: non-inplace negate() clears the flag (was already correct)
    {
        Point ng = Point::generator().negate();
        check(!ng.is_gen(), "NEG-1: generator().negate() -> is_gen() == false");
    }

    // NEG-2: negate_inplace() clears the flag (the B1 fix)
    {
        Point g = Point::generator();
        check(g.is_gen(), "NEG-2a: fresh generator() has is_gen() == true");
        g.negate_inplace();
        check(!g.is_gen(), "NEG-2b: generator after negate_inplace() has is_gen() == false");
    }

    // NEG-3: two negations round-trip to the original point value
    {
        Point g = Point::generator();
        Point twice = g; twice.negate_inplace(); twice.negate_inplace();
        check(points_equal(twice, g), "NEG-3: negate_inplace() twice == original");
    }

    // NEG-4: THE BUG — (-G)*k must equal -(G*k), not G*k.
    // With a stale is_generator_==true, scalar_mul() would take the fixed-base
    // generator path and return k*G (wrong).
    {
        Scalar k = scalar_from_seed(0xC0FFEE);
        Point negG = Point::generator(); negG.negate_inplace();   // -G
        Point lhs = negG.scalar_mul(k);                            // expect -(k*G)
        Point rhs = Point::generator().scalar_mul(k).negate();     // -(k*G)
        check(points_equal(lhs, rhs), "NEG-4: (-G).scalar_mul(k) == -(G.scalar_mul(k))");
    }

    // NEG-5: sanity — (-G)*k differs from G*k
    {
        Scalar k = scalar_from_seed(0xBADF00D);
        Point negG = Point::generator(); negG.negate_inplace();
        Point lhs = negG.scalar_mul(k);
        Point gk  = Point::generator().scalar_mul(k);
        check(!points_equal(lhs, gk), "NEG-5: (-G).scalar_mul(k) != G.scalar_mul(k)");
    }

    std::printf("\n--- Result: %d passed, %d failed ---\n", g_pass, g_fail);
    return (g_fail == 0) ? 0 : 1;
}
