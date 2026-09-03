// ============================================================================
// test_regression_table_build_invariants.cpp
// ============================================================================
// Odd-multiple table construction: the two invariants that hold it together,
// neither of which any existing gate asserts.
//
// Both come from the CPU representation-search work (experiments/
// representation_search) and both are cases where a wrong table is
// SELF-CONSISTENT and wrong rather than obviously broken.
//
//   (1) WINDOW-CONSTANT AGREEMENT.  dual_scalar_mul_gen_point recodes its wNAF
//       digits at WINDOW_G and indexes tbl_G / tbl_H, which are SIZED from a
//       separate constant, kDualMulWindowG.  Those were independently-written
//       literals.  Setting the table constant to 12 while the recoding constant
//       stayed 15 made the lookup index past the end of the array: a SEGFAULT at
//       window 12, and at window 13 no crash at all -- just five silently wrong
//       dual_mul(a*G + b*P) results.  The constants are now derived from one
//       source with a static_assert, but a static_assert only fires if someone
//       keeps them syntactically linked.  This module asserts the BEHAVIOUR:
//       a*G + b*P computed through the windowed table must equal the same point
//       computed by two independent single-base multiplications.
//
//   (2) SHARED-Z TABLE CONTRACT.  Every odd-multiple table in the engine stores
//       pseudo-affine entries that share one implied global Z, built without a
//       single field inversion.  A table build that is internally consistent but
//       lands on the wrong points -- for instance a co-Z chain whose operands
//       drift onto different Z values -- produces a table where every entry is
//       wrong by the same factor, which a spot check of one entry cannot see.
//       This module walks the table through the public API: k*P for every odd k
//       the table represents must equal repeated addition of P.
//
// Both checks are deliberately expressed against PUBLIC behaviour, not internal
// state, so they survive a change of table representation -- which is the whole
// point, since the representation is what the experiment changes.
// ============================================================================

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <array>

#include "secp256k1/point.hpp"
#include "secp256k1/scalar.hpp"
#include "secp256k1/ct/point.hpp"
#include "secp256k1/schnorr.hpp"

using secp256k1::fast::Point;
using secp256k1::fast::Scalar;

// Point has no operator==. Compare the CANONICAL SERIALISED FORM, which is
// exactly the right level for this test: it is representation-independent, so
// these checks survive a change of internal coordinate or table representation
// -- and a change of representation is precisely what they exist to guard.
static bool same_point(const Point& a, const Point& b) {
    if (a.is_infinity() || b.is_infinity()) return a.is_infinity() == b.is_infinity();
    return a.to_compressed() == b.to_compressed();
}

static int g_pass = 0, g_fail = 0;
static void check(bool cond, const char* msg) {
    if (cond) { ++g_pass; }
    else      { ++g_fail; printf("  [FAIL] %s\n", msg); }
}

// SplitMix64 -- deterministic and seeded, so a failure is reproducible.
static std::uint64_t rng_state = 0x5EC0256B1ULL;
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
    b[0] &= 0x7f;          // comfortably below n
    if (b[31] == 0) b[31] = 1;
    return Scalar::from_bytes(b);
}

int test_regression_table_build_invariants_run() {
    printf("======================================================================\n");
    printf("  Regression: odd-multiple table build invariants\n");
    printf("  (1) window-constant agreement  (2) shared-Z table contract\n");
    printf("======================================================================\n");

    const Point G = Point::generator();

    // ---- (1) window-constant agreement --------------------------------
    // dual_scalar_mul_gen_point(a, b, P) must equal a*G + b*P computed by two
    // independent scalar multiplications. If the wNAF recoding width and the
    // table size disagree, the lookup reads outside the table and this fails --
    // or segfaults, which is also a failure and also caught here.
    printf("\n--- (1) dual_scalar_mul_gen_point vs independent a*G + b*P ---\n");
    {
        int agree = 0;
        const int kCases = 24;
        for (int i = 0; i < kCases; ++i) {
            const Scalar a = random_scalar();
            const Scalar b = random_scalar();
            const Point P = G.scalar_mul(random_scalar());

            const Point via_table = Point::dual_scalar_mul_gen_point(a, b, P);
            const Point via_parts = G.scalar_mul(a).add(P.scalar_mul(b));

            if (same_point(via_table, via_parts)) ++agree;
        }
        check(agree == kCases,
              "dual_scalar_mul_gen_point(a,b,P) == a*G + b*P for every case");
        printf("  %d/%d cases agree\n", agree, kCases);

        // Boundary scalars: a == 1 and b == 1 exercise the shortest wNAF, where
        // an off-by-one in the digit count is most likely to show.
        const Scalar one = Scalar::from_uint64(1);
        const Point P1 = G.scalar_mul(random_scalar());
        check(same_point(Point::dual_scalar_mul_gen_point(one, one, P1), G.add(P1)),
              "dual_scalar_mul_gen_point(1,1,P) == G + P");
        const Scalar big = random_scalar();
        check(same_point(Point::dual_scalar_mul_gen_point(big, one, P1), G.scalar_mul(big).add(P1)),
              "dual_scalar_mul_gen_point(k,1,P) == k*G + P");
        check(same_point(Point::dual_scalar_mul_gen_point(one, big, P1), G.add(P1.scalar_mul(big))),
              "dual_scalar_mul_gen_point(1,k,P) == G + k*P");
    }

    // ---- (2) shared-Z table contract ----------------------------------
    // Every entry a windowed table can represent must be the right point. A
    // table built onto a wrong shared Z is wrong in EVERY entry by the same
    // factor, so checking one entry proves nothing -- walk all of them.
    printf("\n--- (2) odd multiples k*P for every k a 5-bit window table holds ---\n");
    {
        const Point P = G.scalar_mul(random_scalar());

        // A 5-bit signed window stores 1P, 3P, ..., 31P.
        Point acc = P;                       // 1*P
        const Point twoP = P.dbl();
        int ok = 0, total = 0;
        for (int k = 1; k <= 31; k += 2) {
            const Point expect = acc;                       // repeated addition
            const Point actual = P.scalar_mul(Scalar::from_uint64(
                                     static_cast<std::uint64_t>(k)));
            ++total;
            if (same_point(actual, expect)) ++ok;
            acc = acc.add(twoP);                            // next odd multiple
        }
        check(ok == total, "k*P == repeated addition for every odd k in [1,31]");
        printf("  %d/%d odd multiples correct\n", ok, total);

        // The same walk through the CT path, which builds its own table.
        int ct_ok = 0, ct_total = 0;
        Point ct_acc = P;
        for (int k = 1; k <= 31; k += 2) {
            const Point expect = ct_acc;
            const Point actual = secp256k1::ct::scalar_mul(
                                     P, Scalar::from_uint64(static_cast<std::uint64_t>(k)));
            ++ct_total;
            if (same_point(actual, expect)) ++ct_ok;
            ct_acc = ct_acc.add(twoP);
        }
        check(ct_ok == ct_total,
              "ct::scalar_mul(P,k) == repeated addition for every odd k in [1,31]");
        printf("  %d/%d odd multiples correct on the ct:: path\n", ct_ok, ct_total);
    }

    // ---- (3) the two tracks must agree with each other ----------------
    // fast:: and ct:: build their tables separately. If one representation
    // changes and the other does not, this is where it shows.
    printf("\n--- (3) fast:: and ct:: scalar_mul agree ---\n");
    {
        int agree = 0;
        const int kCases = 16;
        for (int i = 0; i < kCases; ++i) {
            const Scalar k = random_scalar();
            const Point P = G.scalar_mul(random_scalar());
            if (same_point(P.scalar_mul(k), secp256k1::ct::scalar_mul(P, k))) ++agree;
        }
        check(agree == kCases, "fast::scalar_mul == ct::scalar_mul on random inputs");
        printf("  %d/%d cases agree\n", agree, kCases);
    }

    // ---- (4) x-only pubkey parse: on-curve accept, off-curve reject ----
    // Pins the behaviour of lift_x_from_limbs (schnorr.cpp) after the redundant
    // Jacobi pre-check was removed. The sqrt+verify that remains is the
    // authoritative quadratic-residue test; this asserts it still accepts every
    // real key and still rejects an x with no y on the curve. Off-curve x now
    // costs a sqrt rather than a Jacobi -- slower on that path, and matching
    // libsecp256k1's secp256k1_ge_set_xo_var and this engine's own recovery.cpp.
    printf("\n--- (4) x-only pubkey parse: on-curve accept, off-curve reject ---\n");
    {
        int accepted = 0, total = 0;
        for (int i = 0; i < 24; ++i) {
            const Point P = G.scalar_mul(random_scalar());
            const auto xb = P.x_only_bytes();
            secp256k1::SchnorrXonlyPubkey pk;
            ++total;
            if (secp256k1::schnorr_xonly_pubkey_parse(pk, xb)) ++accepted;
        }
        check(accepted == total, "x-only parse accepts every on-curve x");
        printf("  %d/%d on-curve x values accepted\n", accepted, total);

        // x = 5 has no y on secp256k1 (5^3 + 7 = 132 is a non-residue mod p);
        // x = 1 does. Both are far below 2^33 -- exactly the range the removed
        // jacobi_var was documented to misclassify, so these are the cases that
        // most need pinning now that the sqrt+verify stands alone.
        std::array<std::uint8_t, 32> x5{}; x5[31] = 5;
        std::array<std::uint8_t, 32> x1{}; x1[31] = 1;
        secp256k1::SchnorrXonlyPubkey p5, p1;
        check(!secp256k1::schnorr_xonly_pubkey_parse(p5, x5),
              "x-only parse rejects x = 5 (no y on the curve)");
        check(secp256k1::schnorr_xonly_pubkey_parse(p1, x1),
              "x-only parse accepts x = 1 (on the curve)");
    }

    printf("\n[regression_table_build_invariants] %d/%d checks passed\n",
           g_pass, g_pass + g_fail);
    return (g_fail > 0) ? 1 : 0;
}

#ifdef STANDALONE_TEST
int main() { return test_regression_table_build_invariants_run(); }
#endif
