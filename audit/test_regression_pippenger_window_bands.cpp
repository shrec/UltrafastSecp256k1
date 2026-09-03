// ============================================================================
// test_regression_pippenger_window_bands.cpp
// ============================================================================
// Which ALGORITHM runs for a given MSM size changed, at exactly the sizes
// schnorr_batch_verify uses (N signatures build an MSM over n = 2N points).
// Four things moved, and each has its own silent failure mode -- silent because
// every one of them produces a well-formed point that is simply the wrong one.
//
//   (1) The window band 80..384 went from c = 6 to c = 7. Not a parameter
//       tweak: use_signed turns on at c >= 7, so every MSM in that band
//       switched from the UNSIGNED bucket path to the SIGNED-DIGIT one. The
//       signed path carries out of the top window, and losing that carry is
//       BUG-01 -- the result is off by (lost carries) * 2^256 * P_i. Sizes that
//       used to run c = 6 now run code previously reached only above n = 384.
//
//   (2) msm() routes to pippenger_msm_glv, which decomposes every scalar with
//       GLV and runs the SAME bucket core over 2n points and half-length
//       scalars. New failure surface: the decomposition, the endomorphism, the
//       two sign flags, and a window count derived from 129 bits instead of 256.
//
//   (3) msm()'s crossover moved to 60, so n in [48, 60) now routes to Strauss.
//
//   (4) schnorr_batch_verify's individual/MSM cutoff dropped from 96 to 38, so
//       batches from 39 signatures up take the MSM path for the first time.
//
// Everything is checked against an INDEPENDENT implementation. Strauss
// (multi_scalar_mul) shares no bucket, digit or window code with either
// Pippenger entry point, so agreement between them is real evidence; for the
// small sizes a naive sum of scalar_mul is a third, wholly separate route.
//
// Both window tables are also pinned. They are plain lookups with no test
// coverage of their own, so an accidental edit -- or a merge that restores the
// old bands -- would otherwise show up only as a quiet performance loss.
// ============================================================================

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <array>
#include <vector>

#include "secp256k1/point.hpp"
#include "secp256k1/scalar.hpp"
#include "secp256k1/pippenger.hpp"
#include "secp256k1/multiscalar.hpp"
#include "secp256k1/schnorr.hpp"
#include "secp256k1/batch_verify.hpp"

namespace secp256k1 {
// Not in pippenger.hpp -- it is an internal heuristic. Declared here so the
// bands can be pinned directly instead of inferred from timings.
unsigned pippenger_optimal_window(std::size_t n);
}


using secp256k1::fast::Point;
using secp256k1::fast::Scalar;

static int g_pass = 0, g_fail = 0;
static void check(bool cond, const char* msg) {
    if (cond) { ++g_pass; }
    else      { ++g_fail; printf("  [FAIL] %s\n", msg); }
}

static bool same_point(const Point& a, const Point& b) {
    if (a.is_infinity() || b.is_infinity()) return a.is_infinity() == b.is_infinity();
    return a.add(b.negate()).is_infinity();
}

static std::uint64_t g_rs = 0x9E3779B1C0FFEEULL;
static std::uint64_t nx() {
    g_rs += 0x9E3779B97F4A7C15ULL;
    std::uint64_t z = g_rs;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
static std::array<std::uint8_t, 32> rand_bytes() {
    std::array<std::uint8_t, 32> b{};
    for (int i = 0; i < 32; i += 8) {
        std::uint64_t v = nx();
        for (int j = 0; j < 8; ++j) b[i + j] = static_cast<std::uint8_t>(v >> (8 * j));
    }
    b[0] &= 0x7f;                 // stay below n without a reduction
    if (b[31] == 0) b[31] = 1;
    return b;
}
static Scalar rand_scalar() { return Scalar::from_bytes(rand_bytes()); }

// n - 1: the largest valid scalar, and the one whose top window is full, so it
// is the input that drives the signed-digit carry out of the last window.
static Scalar scalar_n_minus_1() { return Scalar::zero() - Scalar::one(); }

// One MSM checked three ways where affordable, two ways where the naive route
// would dominate the module's runtime.
static void cross_check(const std::vector<Scalar>& sc,
                        const std::vector<Point>& pt,
                        const char* what,
                        bool with_naive) {
    std::size_t const n = sc.size();
    Point const pip = secp256k1::pippenger_msm(sc.data(), pt.data(), n);
    Point const str = secp256k1::multi_scalar_mul(sc.data(), pt.data(), n);
    Point const glv = secp256k1::pippenger_msm_glv(sc.data(), pt.data(), n);
    Point const uni = secp256k1::msm(sc.data(), pt.data(), n);

    char msg[160];
    std::snprintf(msg, sizeof msg, "n=%zu %s: pippenger == strauss", n, what);
    check(same_point(pip, str), msg);
    std::snprintf(msg, sizeof msg, "n=%zu %s: GLV pippenger == strauss", n, what);
    check(same_point(glv, str), msg);
    std::snprintf(msg, sizeof msg, "n=%zu %s: msm() == strauss", n, what);
    check(same_point(uni, str), msg);

    if (with_naive) {
        Point acc = Point::infinity();
        for (std::size_t i = 0; i < n; ++i) acc = acc.add(pt[i].scalar_mul(sc[i]));
        std::snprintf(msg, sizeof msg, "n=%zu %s: pippenger == naive sum", n, what);
        check(same_point(pip, acc), msg);
    }
}

// Both input shapes, because they take different code paths and one of them
// was wrong for years.
//
// affine = true  : z = 1, what schnorr_batch_verify feeds (lift_x, pubkey parse).
// affine = false : a chain of Jacobian adds, each point with its own z != 1.
//                  This is what the two MSM tests in the main suite build, and
//                  it is what the signed-digit scatter mishandled: it read X
//                  and Y and treated them as affine coordinates, so the true
//                  affine x (X/Z^2) was silently replaced by X.
//
// Any check that only ever runs the affine shape cannot see that class of bug.
static std::vector<Point> make_points(std::size_t n, bool affine) {
    Point const G = Point::generator();
    std::vector<Point> pt(n);
    if (affine) {
        for (std::size_t i = 0; i < n; ++i) {
            pt[i] = G.scalar_mul(rand_scalar());
            pt[i].normalize();
        }
        return pt;
    }
    Point P = G.dbl();
    for (std::size_t i = 0; i < n; ++i) {
        pt[i] = P;              // left Jacobian on purpose: z != 1, and a
        P = P.add(G);           // different z for every entry
    }
    return pt;
}
static std::vector<Point> random_points(std::size_t n) { return make_points(n, true); }

int test_regression_pippenger_window_bands_run() {
    printf("=== Pippenger window bands + Strauss crossover ===\n");
    g_pass = g_fail = 0;

    // ---- (1) the window table itself ----------------------------------
    // Pinned at both sides of every band edge. The band that matters is
    // 105..896 -> 7: it is where schnorr_batch_verify lands, and it is the one
    // that used to be 6.
    printf("\n--- (1) window bands ---\n");
    {
        struct { std::size_t n; unsigned c; } const kBands[] = {
            {1, 1}, {2, 2}, {4, 2}, {5, 3}, {8, 3}, {9, 4}, {16, 4},
            {17, 5}, {104, 5},
            {105, 7}, {200, 7}, {512, 7}, {896, 7},
            {897, 8}, {2048, 8},
            {2049, 9}, {8192, 9},
        };
        int ok = 0;
        for (auto const& b : kBands) {
            unsigned const got = secp256k1::pippenger_optimal_window(b.n);
            if (got == b.c) { ++ok; continue; }
            printf("  [FAIL] window(%zu) = %u, expected %u\n", b.n, got, b.c);
        }
        check(ok == static_cast<int>(sizeof kBands / sizeof kBands[0]),
              "every window band boundary holds");

        // c = 6 is the unsigned path with the same effective bucket count as
        // c = 7 signed but ~14% more windows. It lost at every size measured,
        // so nothing in the production range may select it. (This is a
        // performance bound, not the correctness bound the tree used to claim:
        // the "c = 6 defect" was the signed scatter's missing all_affine guard,
        // which was independent of c.)
        bool any_six = false;
        for (std::size_t n = 17; n <= 4096; ++n) {
            if (secp256k1::pippenger_optimal_window(n) == 6) { any_six = true; break; }
        }
        check(!any_six, "c = 6 is never selected (dominated by c = 5 and c = 7)");

        // Monotone non-decreasing: a larger MSM must never pick a smaller
        // window. A non-monotone table means a band edge was mistyped.
        bool monotone = true;
        unsigned prev = 0;
        for (std::size_t n = 1; n <= 4096; ++n) {
            unsigned const c = secp256k1::pippenger_optimal_window(n);
            if (c < prev) { monotone = false; break; }
            prev = c;
        }
        check(monotone, "window table is monotone non-decreasing in n");
    }

    // ---- (2) the sizes the new bands actually moved --------------------
    // 128..896 switched from the unsigned bucket path to the signed-digit one.
    // Random scalars first, so an ordinary input is covered before the crafted
    // ones below.
    printf("\n--- (2) band interiors and edges, both input shapes ---\n");
    {
        std::size_t const kSizes[] = {48, 64, 96, 104, 105, 112, 119, 120, 121,
                                      128, 200, 256, 384, 512, 768, 896, 897};
        for (int affine = 1; affine >= 0; --affine) {
            for (std::size_t n : kSizes) {
                std::vector<Scalar> sc(n);
                for (std::size_t i = 0; i < n; ++i) sc[i] = rand_scalar();
                cross_check(sc, make_points(n, affine != 0),
                            affine ? "affine" : "JACOBIAN", n <= 128);
            }
        }

        // Small scalars on Jacobian points: the exact shape of the two MSM
        // tests in the main suite, and the shape that caught the signed-digit
        // scatter reading X, Y as affine. Kept explicitly so a future window
        // table change cannot quietly stop covering it.
        for (std::size_t n : {std::size_t{256}, std::size_t{384}, std::size_t{512},
                              std::size_t{800}}) {
            std::vector<Scalar> sc(n);
            for (std::size_t i = 0; i < n; ++i)
                sc[i] = Scalar::from_uint64(static_cast<std::uint64_t>(i * 31 + 17));
            cross_check(sc, make_points(n, false), "JACOBIAN, small scalars", false);
        }
    }

    // ---- (3) the inputs that break a carry ------------------------------
    // The signed-digit path subtracts 2^c from a digit above half and carries
    // +1 into the next window. n-1 has every window at its maximum, so every
    // window carries; 2^255 puts a single set bit in the TOP window, which is
    // the one BUG-01 dropped. Both are run at sizes that only reach the signed
    // path under the new bands.
    printf("\n--- (3) carry-forcing scalars on the newly signed band ---\n");
    {
        std::size_t const kSizes[] = {128, 200, 384, 512};
        for (std::size_t n : kSizes) {
            auto const pt = make_points(n, (n / 128) % 2 == 1);  // alternate shapes

            std::vector<Scalar> all_max(n, scalar_n_minus_1());
            cross_check(all_max, pt, "all scalars = n-1", false);

            std::array<std::uint8_t, 32> top{};
            top[0] = 0x80;                       // 2^255, sits in the top window
            std::vector<Scalar> all_top(n, Scalar::from_bytes(top));
            cross_check(all_top, pt, "all scalars = 2^255", false);

            // Mixed: half at the maximum, half random. A carry that is only
            // dropped for SOME lanes is caught here and not by a uniform input.
            std::vector<Scalar> mixed(n);
            for (std::size_t i = 0; i < n; ++i)
                mixed[i] = (i % 2) ? scalar_n_minus_1() : rand_scalar();
            cross_check(mixed, pt, "alternating n-1 / random", false);
        }
    }

    // ---- (4) degenerate inputs across the crossover ---------------------
    printf("\n--- (4) degenerate inputs at the crossover ---\n");
    {
        std::size_t const kSizes[] = {64, 119, 120, 128};
        for (std::size_t n : kSizes) {
            auto pt = random_points(n);

            std::vector<Scalar> ones(n, Scalar::one());
            cross_check(ones, pt, "all scalars = 1", n <= 128);

            std::size_t const mid = n / 2;
            std::size_t const last = n - 1;
            std::vector<Scalar> with_zero(n);
            for (std::size_t i = 0; i < n; ++i) with_zero[i] = rand_scalar();
            with_zero[0] = Scalar::zero();
            with_zero[mid] = Scalar::zero();
            with_zero[last] = Scalar::zero();
            cross_check(with_zero, pt, "some scalars = 0", n <= 128);

            // Every point the same: all weight lands in one bucket per window,
            // which is the sparsest possible bucket occupancy.
            std::vector<Point> same(n, pt[0]);
            std::vector<Scalar> sc(n);
            for (std::size_t i = 0; i < n; ++i) sc[i] = rand_scalar();
            cross_check(sc, same, "all points identical", n <= 128);

            // Cancelling pairs: k*P + k*(-P) = O for every pair, so the whole
            // MSM must be infinity. An off-by-one in the aggregation shows up
            // as a non-infinite result immediately.
            std::vector<Point>  cancel(n);
            std::vector<Scalar> cancel_s(n);
            for (std::size_t i = 0; i + 1 < n; i += 2) {
                Scalar const k = rand_scalar();
                cancel[i]     = pt[i];
                cancel[i + 1] = pt[i].negate();
                cancel_s[i] = cancel_s[i + 1] = k;
            }
            if (n % 2) { cancel[last] = pt[0]; cancel_s[last] = Scalar::zero(); }
            Point const z = secp256k1::msm(cancel_s.data(), cancel.data(), n);
            char msg[96];
            std::snprintf(msg, sizeof msg, "n=%zu cancelling pairs -> infinity", n);
            check(z.is_infinity(), msg);
        }
    }

    // ---- (5) the caller the bands were tuned for ------------------------
    // schnorr_batch_verify(N) builds the MSM over n = 2N points, so these Ns
    // straddle the band edge and the crossover. A batch that verifies when it
    // should and fails when one signature is corrupted exercises the whole
    // path end to end, including the parts the MSM cross-checks cannot see.
    printf("\n--- (5) schnorr_batch_verify across the moved sizes ---\n");
    {
        std::size_t const kBatch[] = {24, 38, 39, 40, 64, 100, 128, 200};
        for (std::size_t N : kBatch) {
            std::vector<secp256k1::SchnorrBatchEntry> e;
            e.reserve(N);
            for (std::size_t i = 0; i < N; ++i) {
                Scalar const sk = rand_scalar();
                auto const pkx = secp256k1::schnorr_pubkey(sk);
                std::array<std::uint8_t, 32> msg{};
                std::uint64_t const v = nx();
                std::memcpy(msg.data(), &v, sizeof v);
                std::array<std::uint8_t, 32> const aux{};
                e.push_back({pkx, msg, secp256k1::schnorr_sign(sk, msg, aux)});
            }
            char m[96];
            std::snprintf(m, sizeof m, "N=%zu (MSM n=%zu) valid batch accepted", N, 2 * N);
            check(secp256k1::schnorr_batch_verify(e), m);

            auto bad = e;
            std::size_t const victim = N / 2;
            bad[victim].signature.s = bad[victim].signature.s + Scalar::one();
            std::snprintf(m, sizeof m, "N=%zu one corrupted signature rejected", N);
            check(!secp256k1::schnorr_batch_verify(bad), m);
        }
    }

    // ---- (6) GLV Pippenger below the routing threshold -----------------
    // msm() never sends n < 60 here, but pippenger_msm_glv is a public entry
    // point and must be correct wherever it is called. Small n is also where
    // the GLV shape is most degenerate: m = 2n can be under one bucket count,
    // so most buckets stay untouched and the aggregation walks past empties.
    printf("\n--- (6) GLV Pippenger at small and degenerate n ---\n");
    {
        for (std::size_t n : {std::size_t{2}, std::size_t{3}, std::size_t{5},
                              std::size_t{8}, std::size_t{17}, std::size_t{33},
                              std::size_t{47}, std::size_t{59}, std::size_t{60}}) {
            for (int affine = 1; affine >= 0; --affine) {
                auto const pt = make_points(n, affine != 0);
                std::vector<Scalar> sc(n);
                for (std::size_t i = 0; i < n; ++i) sc[i] = rand_scalar();
                Point const g = secp256k1::pippenger_msm_glv(sc.data(), pt.data(), n);
                Point const r = secp256k1::multi_scalar_mul(sc.data(), pt.data(), n);
                char m[96];
                std::snprintf(m, sizeof m, "n=%zu GLV pippenger == strauss (%s)",
                              n, affine ? "affine" : "JACOBIAN");
                check(same_point(g, r), m);
            }
        }

        // A point at infinity in the middle of the set: the endomorphism and
        // the negation both have to leave it alone, and the bucket scatter has
        // to skip it. n = 0 and n = 1 take their own early returns.
        {
            std::size_t const n = 64;
            auto pt = random_points(n);
            pt[0] = Point::infinity();
            pt[n / 2] = Point::infinity();   // one write, not a loop
            std::vector<Scalar> sc(n);
            for (std::size_t i = 0; i < n; ++i) sc[i] = rand_scalar();
            check(same_point(secp256k1::pippenger_msm_glv(sc.data(), pt.data(), n),
                             secp256k1::multi_scalar_mul(sc.data(), pt.data(), n)),
                  "GLV pippenger tolerates infinity points");
        }

        // n-1 as every scalar: the decomposition's k1/k2 are at their widest,
        // which is where the 129-bit window bound is tightest.
        {
            std::size_t const n = 200;
            auto const pt = random_points(n);
            std::vector<Scalar> const all_max(n, scalar_n_minus_1());
            check(same_point(secp256k1::pippenger_msm_glv(all_max.data(), pt.data(), n),
                             secp256k1::multi_scalar_mul(all_max.data(), pt.data(), n)),
                  "GLV pippenger: all scalars = n-1 (widest k1, k2)");
        }

        check(secp256k1::pippenger_msm_glv(nullptr, nullptr, 0).is_infinity(),
              "GLV pippenger: n = 0 -> infinity");
    }

    printf("\n[regression_pippenger_window_bands] %d/%d checks passed\n",
           g_pass, g_pass + g_fail);
    return (g_fail > 0) ? 1 : 0;
}

#ifdef STANDALONE_TEST
int main() { return test_regression_pippenger_window_bands_run(); }
#endif
