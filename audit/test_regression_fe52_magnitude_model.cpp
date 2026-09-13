// ============================================================================
// test_regression_fe52_magnitude_model.cpp
// ============================================================================
// GitHub issue #396: FieldElement52 carries no magnitude information and there
// is no VERIFY-style mode that tracks it, so a magnitude precondition violation
// produces a silently wrong field element -- valid unsigned limbs in valid
// memory, simply the wrong number. Nothing for ASan/UBSan/MSan to report, and
// unit tests may or may not trip depending on the limb values.
//
// The bounds that hold the point formulas together live as integer literals at
// the negate() call sites in point.cpp (p.x.negate(8), p.y.negate(4)) and as
// prose in the comments beside them. Nothing checked that the two agree; #397
// was an instance of them disagreeing by 4 magnitudes.
//
// This module closes the half of #396 that does not require changing
// FieldElement52's layout: the model is written down as code
// (secp256k1/field_52_magnitude.hpp), the bounds are declared as constants
// (GEJ_{X,Y,Z}_MAGNITUDE_MAX in point.hpp), and the LIVE formulas are measured
// against them on every run. A formula swap that pushes a coordinate past its
// declared bound now fails here instead of corrupting silently.
//
// It does NOT give per-value tracking -- FieldElement52 still carries no
// magnitude, so a violation constructed inside a single expression is still
// invisible. That is the shadow-field half of the issue and stays open.
//
//   FMM-1  magnitude_of / magnitude_ok classify correctly, including the exact
//          boundary and the wrapped-limb case the model exists to catch
//   FMM-2  kernel postconditions, measured against the real kernels:
//          mul -> (1*M52, 1*M48); sqr -> (1*M52, 2*M48); normalize_weak ->
//          (1*M52, 1*M48). The sqr top limb reaching 2 is why a model derived
//          from the low limbs alone is wrong for this tree.
//   FMM-3  the declared bounds are honest: negate(m) must survive every actual
//          magnitude <= m, and the module reports where it stops surviving.
//   FMM-4  the live formulas stay inside GEJ_{X,Y,Z}_MAGNITUDE_MAX. This is the
//          assertion that catches a formula swap.
//   FMM-5  teeth: the magnitude-22 steady state of EFD's dbl-2009-l -- a
//          published, mathematically correct doubling formula, named in #396 --
//          must be classified as violating X <= 8 AND must be shown to actually
//          corrupt a subsequent product. A bookkeeping mismatch that changed no
//          arithmetic would not be worth a gate.
// ============================================================================

#include "secp256k1/field_52.hpp"
#include "secp256k1/field_52_impl.hpp"
#include "secp256k1/field_52_magnitude.hpp"
#include "secp256k1/point.hpp"
#include "secp256k1/scalar.hpp"

#include <cstdio>
#include <cstdint>
#include <array>
#include <algorithm>
#include <random>

static int g_pass = 0, g_fail = 0;
#include "audit_check.hpp"

#ifndef ADVISORY_SKIP_CODE
#define ADVISORY_SKIP_CODE 77
#endif

// FE52 COMPUTE, not FE52 STORAGE.
//
// SECP256K1_FAST_52BIT means Point STORES its coordinates as 5x52. That is the
// condition for FMM-4/FMM-5, which read live Point coordinates -- but not for
// FMM-1/2/3, which exercise the FE52 kernels themselves. MSVC x64 cl has no
// __int128, so it keeps 4x64 Point storage while still running the 5x52
// kernels as the ecmult compute path via u128_compat (see config.hpp:43-60).
// Guarding the whole module on the storage macro therefore made it return
// ADVISORY_SKIP_CODE on `CI / windows (Release)` -- where it is registered
// advisory=false, so the runner counted that as a FAIL -- while the kernels the
// first three groups test were compiled and live on that very build.
#if defined(SECP256K1_FE52_COMPUTE)

using secp256k1::fast::FieldElement52;
using secp256k1::fast::Point;
using secp256k1::fast::Scalar;
using secp256k1::fast::fe52_magnitude::magnitude_of;
using secp256k1::fast::fe52_magnitude::magnitude_ok;
using secp256k1::fast::fe52_magnitude::kM48;
using secp256k1::fast::fe52_magnitude::kM52;

namespace {

std::mt19937_64 g_rng(0x396FE52ULL);  // fixed seed: reproducible across runs/machines

FieldElement52 fe_at_magnitude(unsigned m) {
    FieldElement52 f{};
    for (int i = 0; i < 4; ++i) f.n[i] = static_cast<std::uint64_t>(m) * kM52;
    f.n[4] = static_cast<std::uint64_t>(m) * kM48;
    return f;
}

FieldElement52 fe_random_within(unsigned m) {
    FieldElement52 f{};
    for (int i = 0; i < 4; ++i) f.n[i] = g_rng() % (static_cast<std::uint64_t>(m) * kM52 + 1);
    f.n[4] = g_rng() % (static_cast<std::uint64_t>(m) * kM48 + 1);
    return f;
}

unsigned top_limb_magnitude(const FieldElement52& f) {
    return static_cast<unsigned>((f.n[4] + kM48 - 1) / kM48);
}

unsigned low_limb_magnitude(const FieldElement52& f) {
    std::uint64_t m = 0;
    for (int i = 0; i < 4; ++i) {
        std::uint64_t const mi = (f.n[i] + kM52 - 1) / kM52;
        if (mi > m) m = mi;
    }
    return static_cast<unsigned>(m);
}

// Does negate(declared) followed by a multiply still agree with the
// mathematically correct answer (normalise first, then a legal magnitude-1
// negate)? This is the corruption path #396 describes: the negated limbs wrap,
// the wrapped values overflow the 128-bit accumulators inside the multiplier,
// and the product is a well-formed wrong number.
bool negate_then_multiply_is_correct(const FieldElement52& v, unsigned declared,
                                     const FieldElement52& w) {
    FieldElement52 a = v.negate(declared);
    FieldElement52 pa{};
    secp256k1::fast::fe52_mul_inner(pa.n, a.n, w.n);
    pa.normalize();

    FieldElement52 vn = v;
    vn.normalize();
    FieldElement52 b = vn.negate(1);
    FieldElement52 pb{};
    secp256k1::fast::fe52_mul_inner(pb.n, b.n, w.n);
    pb.normalize();

    return pa == pb;
}

// Smallest actual magnitude at which negate(declared) starts producing wrong
// products, or 0 if it survived every magnitude tried.
unsigned first_corrupting_magnitude(unsigned declared, unsigned ceiling, int trials) {
    for (unsigned actual = 1; actual <= ceiling; ++actual) {
        for (int t = 0; t < trials; ++t) {
            FieldElement52 const v = fe_random_within(actual);
            FieldElement52 w{};
            for (int i = 0; i < 4; ++i) w.n[i] = g_rng() & kM52;
            w.n[4] = g_rng() & kM48;
            if (!negate_then_multiply_is_correct(v, declared, w)) return actual;
        }
    }
    return 0;
}

// The magnitudes point.cpp's negate() call sites actually pass.
//
// Literals, not secp256k1::fast::GEJ_*_MAGNITUDE_MAX, because those constants
// are declared inside point.hpp's `#if defined(SECP256K1_FAST_52BIT)` -- FE52
// STORAGE -- and this region is compiled whenever the FE52 KERNELS exist. On
// MSVC x64 cl the kernels exist and the storage does not, so referring to the
// constants here does not compile there (C2039/C2065, caught by the Windows
// benchmark build). The static_asserts pin these literals to the declared
// constants wherever the declarations do exist, so the two cannot drift.
constexpr unsigned kDeclaredY = 4;
constexpr unsigned kDeclaredX = 8;
#if defined(SECP256K1_FAST_52BIT)
static_assert(kDeclaredY == secp256k1::fast::GEJ_Y_MAGNITUDE_MAX,
              "FMM's Y bound drifted from GEJ_Y_MAGNITUDE_MAX");
static_assert(kDeclaredX == secp256k1::fast::GEJ_X_MAGNITUDE_MAX,
              "FMM's X bound drifted from GEJ_X_MAGNITUDE_MAX");
#endif

// -- FMM-1 -----------------------------------------------------------------
void test_model_classification() {
    std::printf("[FMM-1] magnitude_of / magnitude_ok classification\n");

    FieldElement52 zero{};
    CHECK(magnitude_of(zero.n) == 0, "the zero value has magnitude 0");
    CHECK(magnitude_ok(zero.n, 0), "magnitude 0 admits zero");

    for (unsigned m : {1u, 2u, 4u, 8u, 22u}) {
        FieldElement52 const exact = fe_at_magnitude(m);
        char msg[128];
        std::snprintf(msg, sizeof msg, "a value at exactly m*mask reports magnitude %u", m);
        CHECK(magnitude_of(exact.n) == m, msg);
        std::snprintf(msg, sizeof msg, "magnitude %u satisfies the magnitude-%u bound", m, m);
        CHECK(magnitude_ok(exact.n, m), msg);
        if (m > 1) {
            std::snprintf(msg, sizeof msg, "magnitude %u does NOT satisfy the magnitude-%u bound", m, m - 1);
            CHECK(!magnitude_ok(exact.n, m - 1), msg);
        }
    }

    // One over the boundary, in each limb independently.
    for (int i = 0; i < 5; ++i) {
        FieldElement52 f = fe_at_magnitude(4);
        f.n[i] += 1;
        char msg[128];
        std::snprintf(msg, sizeof msg, "limb %d one over 4*mask breaks the magnitude-4 bound", i);
        CHECK(!magnitude_ok(f.n, 4), msg);
    }

    // A wrapped limb -- the state an underflowed negate() leaves behind -- must
    // report a huge magnitude, not a small one. This is the case that makes the
    // model useful rather than decorative.
    FieldElement52 wrapped{};
    wrapped.n[0] = ~std::uint64_t{0};
    CHECK(magnitude_of(wrapped.n) > 4000,
          "a wrapped (near-2^64) limb reports a magnitude far above any legal bound");
    CHECK(!magnitude_ok(wrapped.n, kDeclaredX),
          "a wrapped limb violates the declared X bound (GEJ_X_MAGNITUDE_MAX = 8)");
}

// -- FMM-2 -----------------------------------------------------------------
void test_kernel_postconditions() {
    std::printf("[FMM-2] kernel postconditions against the real kernels\n");

    unsigned mul_low = 0, mul_top = 0, sqr_low = 0, sqr_top = 0, nw_low = 0, nw_top = 0;
    constexpr int kTrials = 20000;
    for (int t = 0; t < kTrials; ++t) {
        FieldElement52 const a = fe_random_within(1);
        FieldElement52 const b = fe_random_within(1);
        FieldElement52 r{};

        secp256k1::fast::fe52_mul_inner(r.n, a.n, b.n);
        mul_low = std::max(mul_low, low_limb_magnitude(r));
        mul_top = std::max(mul_top, top_limb_magnitude(r));

        secp256k1::fast::fe52_sqr_inner(r.n, a.n);
        sqr_low = std::max(sqr_low, low_limb_magnitude(r));
        sqr_top = std::max(sqr_top, top_limb_magnitude(r));

        FieldElement52 w = a;
        w.add_assign(b);
        w.add_assign(b);
        w.normalize_weak();
        nw_low = std::max(nw_low, low_limb_magnitude(w));
        nw_top = std::max(nw_top, top_limb_magnitude(w));
    }
    std::printf("  measured over %d trials: mul(%u,%u) sqr(%u,%u) normalize_weak(%u,%u)"
                "  [low in M52, top in M48]\n",
                kTrials, mul_low, mul_top, sqr_low, sqr_top, nw_low, nw_top);

    CHECK(mul_low <= 1, "fe52_mul_inner output: low limbs within 1*M52");
    CHECK(mul_top <= 1, "fe52_mul_inner output: top limb within 1*M48");
    CHECK(sqr_low <= 1, "fe52_sqr_inner output: low limbs within 1*M52");
    CHECK(sqr_top <= 2, "fe52_sqr_inner output: top limb within 2*M48");
    CHECK(nw_low <= 1,  "normalize_weak output: low limbs within 1*M52");
    CHECK(nw_top <= 1,  "normalize_weak output: top limb within 1*M48");
}

// -- FMM-3 -----------------------------------------------------------------
void test_declared_bounds_are_honest() {
    std::printf("[FMM-3] the declared negate() bounds survive every magnitude they admit\n");

    for (unsigned declared : {kDeclaredY, kDeclaredX}) {
        unsigned const first_bad = first_corrupting_magnitude(declared, /*ceiling=*/64, /*trials=*/400);
        char msg[192];
        std::snprintf(msg, sizeof msg,
                      "negate(%u) is correct for every actual magnitude it admits "
                      "(first corrupting magnitude %u must exceed %u)",
                      declared, first_bad, declared);
        CHECK(first_bad == 0 || first_bad > declared, msg);
        std::printf("  negate(%u): correct through magnitude %u, first wrong at %u\n",
                    declared, first_bad ? first_bad - 1 : 64u, first_bad);
    }
}

// FMM-4 and FMM-5 read live Point coordinates as FieldElement52, so they need
// FE52 STORAGE, not just FE52 compute.
#if defined(SECP256K1_FAST_52BIT)

// -- FMM-4 -----------------------------------------------------------------
void test_live_formulas_within_declared_bounds() {
    std::printf("[FMM-4] live point formulas against GEJ_{X,Y,Z}_MAGNITUDE_MAX\n");

    unsigned dx = 0, dy = 0, dz = 0, ax = 0, ay = 0, az = 0;
    Point acc = Point::generator();
    constexpr int kPoints = 96;
    constexpr int kChain = 12;
    for (int t = 0; t < kPoints; ++t) {
        std::array<std::uint8_t, 32> sb{};
        for (auto& c : sb) c = static_cast<std::uint8_t>(g_rng());
        Scalar const s = Scalar::from_bytes(sb);
        if (s.is_zero()) continue;
        Point const P = Point::generator().scalar_mul(s);

        Point const d = P.dbl();
        dx = std::max(dx, magnitude_of(d.X52().n));
        dy = std::max(dy, magnitude_of(d.Y52().n));
        dz = std::max(dz, magnitude_of(d.Z52().n));

        Point const sum = acc.add(P);
        ax = std::max(ax, magnitude_of(sum.X52().n));
        ay = std::max(ay, magnitude_of(sum.Y52().n));
        az = std::max(az, magnitude_of(sum.Z52().n));

        // Iterate the doubling chain so the coordinates reach their fixed point
        // rather than only their first-step values.
        acc = d;
        for (int k = 0; k < kChain; ++k) {
            acc = acc.dbl();
            dx = std::max(dx, magnitude_of(acc.X52().n));
            dy = std::max(dy, magnitude_of(acc.Y52().n));
            dz = std::max(dz, magnitude_of(acc.Z52().n));
        }
    }
    std::printf("  steady state over %d points x %d doublings: dbl(X %u, Y %u, Z %u)  add(X %u, Y %u, Z %u)\n",
                kPoints, kChain, dx, dy, dz, ax, ay, az);

    CHECK(dx <= secp256k1::fast::GEJ_X_MAGNITUDE_MAX, "Point::dbl X stays within GEJ_X_MAGNITUDE_MAX");
    CHECK(dy <= secp256k1::fast::GEJ_Y_MAGNITUDE_MAX, "Point::dbl Y stays within GEJ_Y_MAGNITUDE_MAX");
    CHECK(dz <= secp256k1::fast::GEJ_Z_MAGNITUDE_MAX, "Point::dbl Z stays within GEJ_Z_MAGNITUDE_MAX");
    CHECK(ax <= secp256k1::fast::GEJ_X_MAGNITUDE_MAX, "Point::add X stays within GEJ_X_MAGNITUDE_MAX");
    CHECK(ay <= secp256k1::fast::GEJ_Y_MAGNITUDE_MAX, "Point::add Y stays within GEJ_Y_MAGNITUDE_MAX");
    CHECK(az <= secp256k1::fast::GEJ_Z_MAGNITUDE_MAX, "Point::add Z stays within GEJ_Z_MAGNITUDE_MAX");
}

// -- FMM-5 -----------------------------------------------------------------
void test_the_near_miss_is_caught_and_is_real() {
    std::printf("[FMM-5] the #396 near-miss: EFD dbl-2009-l steady state (X 22, Y 10)\n");

    // Classification: the model must reject it against the declared bound.
    FieldElement52 const x22 = fe_at_magnitude(22);
    FieldElement52 const y10 = fe_at_magnitude(10);
    CHECK(!magnitude_ok(x22.n, secp256k1::fast::GEJ_X_MAGNITUDE_MAX),
          "steady-state X 22 is rejected against GEJ_X_MAGNITUDE_MAX = 8");
    CHECK(!magnitude_ok(y10.n, secp256k1::fast::GEJ_Y_MAGNITUDE_MAX),
          "steady-state Y 10 is rejected against GEJ_Y_MAGNITUDE_MAX = 4");

    // And it is a real defect, not a bookkeeping mismatch: at those magnitudes
    // negate(declared) followed by a multiply genuinely disagrees with the
    // mathematically correct answer.
    int x_wrong = 0, y_wrong = 0;
    constexpr int kTrials = 400;
    for (int t = 0; t < kTrials; ++t) {
        FieldElement52 w{};
        for (int i = 0; i < 4; ++i) w.n[i] = g_rng() & kM52;
        w.n[4] = g_rng() & kM48;
        if (!negate_then_multiply_is_correct(fe_random_within(22), secp256k1::fast::GEJ_X_MAGNITUDE_MAX, w)) ++x_wrong;
        if (!negate_then_multiply_is_correct(fe_random_within(10), secp256k1::fast::GEJ_Y_MAGNITUDE_MAX, w)) ++y_wrong;
    }
    std::printf("  products corrupted: X-path %d/%d, Y-path %d/%d\n",
                x_wrong, kTrials, y_wrong, kTrials);
    CHECK(x_wrong > 0, "magnitude-22 X through negate(8) actually corrupts a product");
    CHECK(y_wrong > 0, "magnitude-10 Y through negate(4) actually corrupts a product");

    // Control: the magnitudes the live formulas actually reach must NOT corrupt.
    int control_wrong = 0;
    for (int t = 0; t < kTrials; ++t) {
        FieldElement52 w{};
        for (int i = 0; i < 4; ++i) w.n[i] = g_rng() & kM52;
        w.n[4] = g_rng() & kM48;
        if (!negate_then_multiply_is_correct(fe_random_within(3), secp256k1::fast::GEJ_X_MAGNITUDE_MAX, w)) ++control_wrong;
    }
    CHECK(control_wrong == 0,
          "control: the magnitude-3 steady state the live formulas reach corrupts nothing");
}

#endif  // SECP256K1_FAST_52BIT (FMM-4/FMM-5)

}  // namespace

int test_regression_fe52_magnitude_model_run() {
    g_pass = g_fail = 0;
    std::printf("=== FE52 magnitude model (GitHub issue #396) ===\n");
    test_model_classification();
    test_kernel_postconditions();
    test_declared_bounds_are_honest();
#if defined(SECP256K1_FAST_52BIT)
    test_live_formulas_within_declared_bounds();
    test_the_near_miss_is_caught_and_is_real();
#else
    std::printf("[FMM-4/5] not applicable: this build computes with the 5x52 kernels "
                "but STORES Point coordinates as 4x64 (MSVC x64 cl, no __int128), so "
                "there are no live FE52 point coordinates to measure. FMM-1..3 above "
                "exercised the same kernels this build actually runs.\n");
#endif
    std::printf("[fe52_magnitude_model] %d passed, %d failed\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}

#else  // !SECP256K1_FE52_COMPUTE

// No 5x52 kernels in this build at all (ESP32 / STM32 / wasm: no __int128 and
// u128_compat is too slow there). The 4x64 FieldElement path has no magnitude
// parameter -- FieldElement::negate ignores it -- so there is nothing to model.
// A genuine inapplicability, not a skipped check.
int test_regression_fe52_magnitude_model_run() {
    std::printf("[fe52_magnitude_model] SKIP: the 5x52 kernels are not built on "
                "this platform (no __int128, no MSVC x64 u128_compat)\n");
    return ADVISORY_SKIP_CODE;
}

#endif  // SECP256K1_FE52_COMPUTE

#ifdef STANDALONE_TEST
int main() { return test_regression_fe52_magnitude_model_run(); }
#endif
