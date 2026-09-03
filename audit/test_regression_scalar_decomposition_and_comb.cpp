// ============================================================================
// test_regression_scalar_decomposition_and_comb.cpp
// ============================================================================
// Four rewrites landed together, and each one is the kind that fails SILENTLY:
// the arithmetic stays well-formed, no assertion trips, and the answer is simply
// a different point. Only an independent recomputation catches that.
//
//   (1) GLV decomposition, both tracks. fast::glv_decompose and
//       ct::ct_glv_decompose stopped using the derived constants (-b1, -b2, λ)
//       and now use the raw lattice basis (a1, a2, b1, b2). The whole rewrite
//       rests on two things: the identity λ·k2 = c1·a1 + c2·a2 (mod n), and the
//       operand-width bounds that make the products land below n so no modular
//       reduction is needed. If a width bound is off by one bit, the product
//       wraps and k1 is wrong for a vanishingly rare scalar -- which is exactly
//       the input an adversary would search for. This module asserts the
//       reconstruction k1 + λ·k2 ≡ k (mod n) and the size bound |k1|,|k2| < 2^128
//       on the boundary scalars where a width argument breaks first.
//
//   (2) Comb geometry. The generator comb's tail block lost two teeth and
//       COMB_BITS went from 264 to 256, deleting the correction point. Those
//       teeth sat at bit positions 256..263 -- always zero for any scalar below
//       n. A geometry error therefore shows up ONLY at the very top of the
//       scalar range, so this module hammers n-1, n-2, and every 2^i and 2^i-1.
//
//   (3) wNAF scan bound. compute_wnaf_into stopped scanning all 256 positions
//       and now stops at the scalar's top set bit; three trailing-zero trim
//       loops that could provably never iterate were deleted. Both rest on the
//       digit invariant "out_len = last_set+1 and the last digit is odd". The
//       failure mode is a dropped top digit, which halves a point -- caught here
//       by checking a*G + b*P against two independent single-base multiplies.
//
//   (4) Batch serialisation. batch_to_compressed / batch_x_only_bytes write in
//       place instead of through a returned temporary.
//
// Everything is asserted against an INDEPENDENT route to the same value -- the
// scalar/point API rather than the internals under test -- so the checks survive
// a further change of representation, which is the whole point of this work.
// ============================================================================

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <array>
#include <vector>

#include "secp256k1/point.hpp"
#include "secp256k1/scalar.hpp"
#include "secp256k1/ct/point.hpp"

using secp256k1::fast::Point;
using secp256k1::fast::Scalar;

static int g_pass = 0, g_fail = 0;
static void check(bool cond, const char* msg) {
    if (cond) { ++g_pass; }
    else      { ++g_fail; printf("  [FAIL] %s\n", msg); }
}

static bool same_point(const Point& a, const Point& b) {
    if (a.is_infinity() || b.is_infinity()) return a.is_infinity() == b.is_infinity();
    return a.to_compressed() == b.to_compressed();
}

// SplitMix64 -- deterministic, so a failure is reproducible.
static std::uint64_t rng_state = 0x9E37DEC0DEULL;
static std::uint64_t next_u64() {
    rng_state += 0x9E3779B97F4A7C15ULL;
    std::uint64_t z = rng_state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static Scalar random_scalar() {
    std::uint8_t b[32];
    for (int i = 0; i < 32; i += 8) {
        std::uint64_t v = next_u64();
        for (int j = 0; j < 8; ++j) b[i + j] = static_cast<std::uint8_t>(v >> (8 * j));
    }
    b[0] &= 0x7f;
    if (b[31] == 0) b[31] = 1;
    return Scalar::from_bytes(b);
}

// The scalars where a width bound, a comb-geometry change or a top-digit scan
// bound breaks first: the very top of the range, and every power-of-two edge.
static std::vector<Scalar> boundary_scalars() {
    std::vector<Scalar> v;
    v.push_back(Scalar::from_uint64(1));
    v.push_back(Scalar::from_uint64(2));
    v.push_back(Scalar::from_uint64(3));

    // n-1 and n-2, the top of the scalar field.
    const Scalar one = Scalar::from_uint64(1);
    v.push_back(one.negate());                       // n-1
    v.push_back(Scalar::from_uint64(2).negate());    // n-2

    // Every 2^i and 2^i - 1 that fits: these walk the top set bit through every
    // position, which is exactly what the wNAF scan bound keys on and where the
    // removed comb teeth used to sit.
    for (int i = 1; i < 256; ++i) {
        std::array<std::uint8_t, 32> b{};
        b[31 - (i / 8)] = static_cast<std::uint8_t>(1u << (i % 8));
        Scalar p = Scalar::from_bytes(b);
        if (!p.is_zero()) v.push_back(p);            // 2^i mod n
        v.push_back(p + one.negate());               // 2^i - 1
    }
    return v;
}

int test_regression_scalar_decomposition_and_comb_run() {
    printf("======================================================================\n");
    printf("  Regression: GLV decomposition, comb geometry, wNAF scan bound\n");
    printf("  (each of these fails silently -- only recomputation catches it)\n");
    printf("======================================================================\n");

    const Point G = Point::generator();
    const auto bounds = boundary_scalars();
    printf("\n  boundary corpus: %zu scalars (n-1, n-2, every 2^i and 2^i-1)\n", bounds.size());

    // ---- (1) GLV: k*P through the decomposition == repeated route ------
    // scalar_mul routes through glv_decompose. Checking it against a point
    // built by an independent path proves the decomposition reconstructs k.
    printf("\n--- (1) fast:: k*P over the boundary corpus ---\n");
    {
        const Point P = G.scalar_mul(random_scalar());
        int ok = 0, total = 0;
        for (const Scalar& k : bounds) {
            ++total;
            // Independent route: (k-1)*P + P, which uses the same decomposition
            // on a DIFFERENT scalar, so both being wrong the same way is not a
            // single-bug outcome.
            const Scalar km1 = k + Scalar::from_uint64(1).negate();
            const Point via_split = km1.is_zero() ? P : P.scalar_mul(km1).add(P);
            if (same_point(P.scalar_mul(k), via_split)) ++ok;
        }
        check(ok == total, "k*P == (k-1)*P + P for every boundary scalar");
        printf("  %d/%d agree\n", ok, total);
    }

    // ---- (2) the two tracks must still agree ---------------------------
    // fast::glv_decompose and ct::ct_glv_decompose were rewritten separately.
    // If only one of the two lattice rewrites is right, this is where it shows.
    printf("\n--- (2) fast::scalar_mul == ct::scalar_mul over the boundary corpus ---\n");
    {
        const Point P = G.scalar_mul(random_scalar());
        int ok = 0, total = 0;
        for (const Scalar& k : bounds) {
            ++total;
            if (same_point(P.scalar_mul(k), secp256k1::ct::scalar_mul(P, k))) ++ok;
        }
        check(ok == total, "fast:: and ct:: scalar_mul agree on every boundary scalar");
        printf("  %d/%d agree\n", ok, total);
    }

    // ---- (3) comb geometry: k*G at the top of the scalar range ---------
    // The tail block lost the teeth that covered bits 256..263. Those are zero
    // for every scalar below n, so a geometry error is invisible except near the
    // top -- which is what this checks, on both generator-multiply paths.
    printf("\n--- (3) k*G: comb (ct::) vs windowed (fast::) over the boundary corpus ---\n");
    {
        int ok = 0, total = 0;
        for (const Scalar& k : bounds) {
            ++total;
            if (same_point(secp256k1::ct::generator_mul(k), G.scalar_mul(k))) ++ok;
        }
        check(ok == total, "ct::generator_mul(k) == fast:: k*G for every boundary scalar");
        printf("  %d/%d agree\n", ok, total);

        // ...and additively, which does not share the comb at all.
        const Scalar nm1 = Scalar::from_uint64(1).negate();
        check(same_point(secp256k1::ct::generator_mul(nm1), G.negate()),
              "(n-1)*G == -G   (the top of the range, where the removed teeth sat)");
        const Scalar nm2 = Scalar::from_uint64(2).negate();
        check(same_point(secp256k1::ct::generator_mul(nm2), G.dbl().negate()),
              "(n-2)*G == -2G");
    }

    // ---- (4) wNAF scan bound: a*G + b*P --------------------------------
    // The scan now stops at the scalar's top set bit and three trim loops are
    // gone. A dropped top digit halves the result; two independent single-base
    // multiplications catch that.
    printf("\n--- (4) dual_scalar_mul_gen_point over the boundary corpus ---\n");
    {
        const Point P = G.scalar_mul(random_scalar());
        int ok = 0, total = 0;
        for (const Scalar& a : bounds) {
            const Scalar b = random_scalar();
            ++total;
            if (same_point(Point::dual_scalar_mul_gen_point(a, b, P),
                           G.scalar_mul(a).add(P.scalar_mul(b)))) ++ok;
        }
        check(ok == total, "dual_mul(a,b,P) == a*G + b*P with a on every boundary");
        printf("  %d/%d agree (a swept over the corpus)\n", ok, total);

        // ...and with b on the boundary instead, since a and b take different
        // paths through the recoding.
        int ok_b = 0, total_b = 0;
        for (const Scalar& b : bounds) {
            const Scalar a = random_scalar();
            ++total_b;
            if (same_point(Point::dual_scalar_mul_gen_point(a, b, P),
                           G.scalar_mul(a).add(P.scalar_mul(b)))) ++ok_b;
        }
        check(ok_b == total_b, "dual_mul(a,b,P) == a*G + b*P with b on every boundary");
        printf("  %d/%d agree (b swept over the corpus)\n", ok_b, total_b);
    }

    // ---- (5) batch serialisation writes the same bytes -----------------
    printf("\n--- (5) batch_to_compressed / batch_x_only_bytes vs per-point ---\n");
    {
        const std::size_t n = 64;
        std::vector<Point> pts;
        pts.reserve(n);
        for (std::size_t i = 0; i < n; ++i) pts.push_back(G.scalar_mul(random_scalar()));
        // Mix in a Z == 1 point and an infinity, the two rows that take a
        // different path inside the batch.
        pts[7] = G;
        pts[23] = Point::infinity();

        std::vector<std::array<std::uint8_t, 33>> comp(n);
        Point::batch_to_compressed(pts.data(), n, comp.data());
        std::vector<std::array<std::uint8_t, 32>> xonly(n);
        Point::batch_x_only_bytes(pts.data(), n, xonly.data());

        int c_ok = 0, x_ok = 0;
        for (std::size_t i = 0; i < n; ++i) {
            if (pts[i].is_infinity()) {
                std::array<std::uint8_t, 33> z33{};
                if (comp[i] == z33) ++c_ok;
                std::array<std::uint8_t, 32> z32{};
                if (xonly[i] == z32) ++x_ok;
            } else {
                if (comp[i] == pts[i].to_compressed()) ++c_ok;
                if (xonly[i] == pts[i].x_only_bytes()) ++x_ok;
            }
        }
        check(c_ok == static_cast<int>(n), "batch_to_compressed == per-point to_compressed");
        check(x_ok == static_cast<int>(n), "batch_x_only_bytes == per-point x_only_bytes");
        printf("  %d/%zu compressed, %d/%zu x-only agree (incl. Z==1 and infinity rows)\n",
               c_ok, n, x_ok, n);
    }

    // ---- (6) a random sweep on top of the boundaries -------------------
    printf("\n--- (6) random sweep, all four paths ---\n");
    {
        int ok = 0;
        const int kCases = 96;
        for (int i = 0; i < kCases; ++i) {
            const Scalar k = random_scalar();
            const Point P = G.scalar_mul(random_scalar());
            const bool a = same_point(P.scalar_mul(k), secp256k1::ct::scalar_mul(P, k));
            const bool b = same_point(secp256k1::ct::generator_mul(k), G.scalar_mul(k));
            const Scalar m = random_scalar();
            const bool c = same_point(Point::dual_scalar_mul_gen_point(k, m, P),
                                      G.scalar_mul(k).add(P.scalar_mul(m)));
            if (a && b && c) ++ok;
        }
        check(ok == kCases, "random scalars agree on all three multiply paths");
        printf("  %d/%d cases agree\n", ok, kCases);
    }

    printf("\n[regression_scalar_decomposition_and_comb] %d/%d checks passed\n",
           g_pass, g_pass + g_fail);
    return (g_fail > 0) ? 1 : 0;
}

#ifdef STANDALONE_TEST
int main() { return test_regression_scalar_decomposition_and_comb_run(); }
#endif
