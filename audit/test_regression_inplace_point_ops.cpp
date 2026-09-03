// ============================================================================
// test_regression_inplace_point_ops.cpp
// ============================================================================
// 54 call sites were rewritten from `X = X.op(Y)` to `X.op_inplace(Y)` to stop
// paying for a Point-sized copy that SROA cannot eliminate. The rewrite is only
// safe if the in-place form is EXACTLY the returning form, and nothing in the
// suite asserted that — the two are separate implementations, not one wrapping
// the other, so "obviously the same" is an assumption, not a fact.
//
// This module pins the four pairs the rewrite used:
//
//     X.add_inplace(Y)     ==  X.add(Y)
//     X.sub_inplace(Y)     ==  X.add(Y.negate())     <- two sites in pedersen.cpp
//     X.dbl_inplace()      ==  X.dbl()
//     X.negate_inplace()   ==  X.negate()
//     f.negate_assign(m)   ==  f.negate(m)           <- FieldElement and FE52
//
// The cases that matter are the ones where group addition is not the generic
// formula: an infinity operand, P + (-P), P + P routed through add(), and a
// mixed affine/Jacobian pair. A sloppy in-place variant typically agrees with
// the returning one on generic inputs and diverges on exactly those.
//
// Self-aliasing (`X.add_inplace(X)`) is probed rather than asserted: no rewritten
// site does it, and the result is reported so the next reader knows whether the
// contract allows it.
// ============================================================================

#include <cstdio>
#include <cstdint>
#include <array>
#include <vector>

#include "secp256k1/point.hpp"
#include "secp256k1/scalar.hpp"
#include "secp256k1/field.hpp"

using secp256k1::fast::Point;
using secp256k1::fast::Scalar;
using secp256k1::fast::FieldElement;

static int g_pass = 0, g_fail = 0;
static void check(bool cond, const char* msg) {
    if (cond) { ++g_pass; }
    else      { ++g_fail; printf("  [FAIL] %s\n", msg); }
}

static bool same_point(const Point& a, const Point& b) {
    if (a.is_infinity() || b.is_infinity()) return a.is_infinity() == b.is_infinity();
    return a.add(b.negate()).is_infinity();
}

static std::uint64_t g_rs = 0x1AF1ACE0ULL;
static std::uint64_t nx() {
    g_rs += 0x9E3779B97F4A7C15ULL;
    std::uint64_t z = g_rs;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
static Scalar rand_scalar() {
    std::array<std::uint8_t, 32> b{};
    for (int i = 0; i < 32; i += 8) {
        std::uint64_t v = nx();
        for (int j = 0; j < 8; ++j) b[i + j] = static_cast<std::uint8_t>(v >> (8 * j));
    }
    b[0] &= 0x7f; if (b[31] == 0) b[31] = 1;
    return Scalar::from_bytes(b);
}

int test_regression_inplace_point_ops_run() {
    printf("=== in-place point ops match their returning twins ===\n");
    g_pass = g_fail = 0;

    Point const G = Point::generator();
    Point const INF = Point::infinity();

    // A pool spanning both coordinate shapes: the rewritten sites feed both,
    // and mixed adds take a different branch from generic ones.
    std::vector<Point> pool;
    for (int i = 0; i < 6; ++i) {
        Point p = G.scalar_mul(rand_scalar());
        pool.push_back(p);          // Jacobian
        p.normalize();
        pool.push_back(p);          // affine, same point
    }
    pool.push_back(G);
    pool.push_back(G.dbl());
    pool.push_back(INF);

    // ---- add / sub ------------------------------------------------------
    printf("\n--- add_inplace / sub_inplace ---\n");
    {
        int ok_add = 0, ok_sub = 0, cases = 0;
        for (const Point& a : pool) {
            for (const Point& b : pool) {
                ++cases;
                Point x = a; x.add_inplace(b);
                if (same_point(x, a.add(b))) ++ok_add;

                Point y = a; y.sub_inplace(b);
                if (same_point(y, a.add(b.negate()))) ++ok_sub;
            }
        }
        char m[96];
        std::snprintf(m, sizeof m, "add_inplace == add on all %d ordered pairs", cases);
        check(ok_add == cases, m);
        std::snprintf(m, sizeof m, "sub_inplace == add(negate) on all %d ordered pairs", cases);
        check(ok_sub == cases, m);
        printf("  add %d/%d, sub %d/%d\n", ok_add, cases, ok_sub, cases);
    }

    // ---- the non-generic cases, named individually ----------------------
    printf("\n--- the cases a generic-only implementation gets wrong ---\n");
    {
        Point const P = pool[0];
        Point const Paff = pool[1];

        { Point x = P; x.add_inplace(INF); check(same_point(x, P), "P + O == P"); }
        { Point x = INF; x.add_inplace(P); check(same_point(x, P), "O + P == P"); }
        { Point x = INF; x.add_inplace(INF); check(x.is_infinity(), "O + O == O"); }
        { Point x = P; x.add_inplace(P.negate()); check(x.is_infinity(), "P + (-P) == O"); }
        { Point x = P; x.sub_inplace(P); check(x.is_infinity(), "P - P == O"); }
        { Point x = P; x.add_inplace(Paff); check(same_point(x, P.dbl()), "P + P (affine twin) == 2P"); }
        { Point x = Paff; x.add_inplace(P); check(same_point(x, P.dbl()), "P(affine) + P == 2P"); }
        { Point x = P; x.sub_inplace(INF); check(same_point(x, P), "P - O == P"); }
        { Point x = INF; x.sub_inplace(P); check(same_point(x, P.negate()), "O - P == -P"); }
    }

    // ---- dbl / negate ---------------------------------------------------
    printf("\n--- dbl_inplace / negate_inplace ---\n");
    {
        int ok_dbl = 0, ok_neg = 0, ok_neg2 = 0;
        for (const Point& a : pool) {
            Point x = a; x.dbl_inplace();
            if (same_point(x, a.dbl())) ++ok_dbl;

            Point y = a; y.negate_inplace();
            if (same_point(y, a.negate())) ++ok_neg;

            Point z = a; z.negate_inplace(); z.negate_inplace();
            if (same_point(z, a)) ++ok_neg2;
        }
        int const n = static_cast<int>(pool.size());
        check(ok_dbl == n,  "dbl_inplace == dbl on every pool entry");
        check(ok_neg == n,  "negate_inplace == negate on every pool entry");
        check(ok_neg2 == n, "negate_inplace twice is the identity");

        Point i1 = INF; i1.dbl_inplace();
        check(i1.is_infinity(), "2*O == O");
        Point i2 = INF; i2.negate_inplace();
        check(i2.is_infinity(), "-O == O");
    }

    // ---- field negate_assign (the FieldElement half of the rewrite) -----
    printf("\n--- FieldElement negate_assign == negate ---\n");
    {
        int ok = 0, cases = 0;
        for (int t = 0; t < 8; ++t) {
            std::array<std::uint8_t, 32> b{};
            for (int i = 0; i < 32; i += 8) {
                std::uint64_t v = nx();
                for (int j = 0; j < 8; ++j) b[i + j] = static_cast<std::uint8_t>(v >> (8 * j));
            }
            b[0] &= 0x3f;
            FieldElement const f = FieldElement::from_bytes(b);
            for (unsigned m = 1; m <= 4; ++m) {
                ++cases;
                FieldElement g = f; g.negate_assign(m);
                if (g.to_bytes() == f.negate(m).to_bytes()) ++ok;
            }
        }
        char msg[96];
        std::snprintf(msg, sizeof msg, "negate_assign(m) == negate(m) for m in 1..4 (%d cases)", cases);
        check(ok == cases, msg);

        // Default argument parity: the rewrite turned `y = y.negate()` into
        // `y.negate_assign()` at sites that never passed a magnitude, so the
        // two defaults have to agree.
        std::array<std::uint8_t, 32> b{}; b[31] = 7;
        FieldElement const f = FieldElement::from_bytes(b);
        FieldElement g = f; g.negate_assign();
        check(g.to_bytes() == f.negate().to_bytes(),
              "negate_assign() and negate() share the same default magnitude");
    }

    // ---- self-aliasing: reported, not required --------------------------
    printf("\n--- self-aliasing (probe, not a requirement) ---\n");
    {
        Point const P = pool[0];
        Point x = P; x.add_inplace(x);
        bool const aliases_ok = same_point(x, P.dbl());
        printf("  X.add_inplace(X) %s 2X -- no rewritten site relies on this\n",
               aliases_ok ? "==" : "!=");
    }

    printf("\n[regression_inplace_point_ops] %d/%d checks passed\n",
           g_pass, g_pass + g_fail);
    return (g_fail > 0) ? 1 : 0;
}

#ifdef STANDALONE_TEST
int main() { return test_regression_inplace_point_ops_run(); }
#endif
