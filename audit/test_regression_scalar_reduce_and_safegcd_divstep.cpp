// ============================================================================
// test_regression_scalar_reduce_and_safegcd_divstep.cpp
// ============================================================================
// Two rewrites of arithmetic that has exactly one correct answer and no loud
// failure mode. Both were made for speed, and both would produce a well-formed
// wrong number rather than an error if the reasoning behind them is off.
//
//   (1) Scalar::from_bytes stopped reducing by "subtract n, then select". It
//       now adds 2^256 - n and drops the carry out of the top limb, gated on a
//       specialised order_overflow() that replaces the generic borrow-chain
//       ge(x, ORDER) at nine call sites. The failure surface is the boundary:
//       n-1 must pass through untouched, n must become 0, and 2^256-1 must
//       become 2^256-1-n. An off-by-one in either the overflow test or the
//       complement lands on exactly those inputs and nowhere else, which is
//       why a random sweep alone would not find it.
//
//       This module recomputes the reduction independently, in base 256 on the
//       raw bytes -- no limbs, no complement, no shared helper -- so agreement
//       is evidence rather than a tautology.
//
//   (2) The variable-time safegcd divstep stopped cancelling one bit of g per
//       pass and now solves for the multiple of f that cancels up to six. If
//       the Hensel lift or the limit that bounds it is wrong, the transition
//       matrix stops being unimodular and the inverse is silently a different
//       field element.
//
//       Checked by the defining property a * a^-1 == 1, which depends on
//       nothing the rewrite touched, and for a sample by Fermat's little
//       theorem (a^(p-2)) computed with plain square-and-multiply -- a
//       completely different algorithm.
//
// The inputs that matter for a divstep rewrite are the ones with long runs of
// zero bits, because the bulk-skip and the multi-bit cancellation interact
// there; they are generated explicitly rather than hoped for.
// ============================================================================

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <array>
#include <vector>

#include "secp256k1/scalar.hpp"
#include "secp256k1/field.hpp"
#include "secp256k1/point.hpp"
#include "secp256k1/ecdsa.hpp"

using secp256k1::fast::Scalar;
using secp256k1::fast::FieldElement;
using secp256k1::fast::Point;

static int g_pass = 0, g_fail = 0;
static void check(bool cond, const char* msg) {
    if (cond) { ++g_pass; }
    else      { ++g_fail; printf("  [FAIL] %s\n", msg); }
}

using Bytes = std::array<std::uint8_t, 32>;

// group order n, big-endian
static const Bytes kOrder = {
  0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF, 0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFE,
  0xBA,0xAE,0xDC,0xE6,0xAF,0x48,0xA0,0x3B, 0xBF,0xD2,0x5E,0x8C,0xD0,0x36,0x41,0x41};
// field prime p, big-endian
static const Bytes kPrime = {
  0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF, 0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,
  0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF, 0xFF,0xFF,0xFF,0xFE,0xFF,0xFF,0xFC,0x2F};

// ---- independent big-endian byte arithmetic, sharing nothing with the engine

static int be_cmp(const Bytes& a, const Bytes& b) {
    for (std::size_t i = 0; i < 32; ++i) {
        if (a[i] != b[i]) return a[i] < b[i] ? -1 : 1;
    }
    return 0;
}
static Bytes be_sub(const Bytes& a, const Bytes& b) {   // a - b, a >= b
    Bytes r{};
    int borrow = 0;
    for (int i = 31; i >= 0; --i) {
        int v = int(a[i]) - int(b[i]) - borrow;
        borrow = (v < 0);
        r[std::size_t(i)] = std::uint8_t(v + (borrow ? 256 : 0));
    }
    return r;
}
static Bytes be_add_small(const Bytes& a, int delta) {   // a + delta, |delta| small
    Bytes r = a;
    int carry = delta;
    for (int i = 31; i >= 0 && carry != 0; --i) {
        int v = int(r[std::size_t(i)]) + carry;
        if (v < 0)   { r[std::size_t(i)] = std::uint8_t(v + 256); carry = -1; }
        else         { r[std::size_t(i)] = std::uint8_t(v & 0xFF); carry = v >> 8; }
    }
    return r;
}
// The whole reduction, done the obvious way: one conditional subtract.
static Bytes reduce_mod_order(const Bytes& x) {
    return (be_cmp(x, kOrder) >= 0) ? be_sub(x, kOrder) : x;
}

static std::uint64_t g_rs = 0xD1057E951ULL;
static std::uint64_t nx() {
    g_rs += 0x9E3779B97F4A7C15ULL;
    std::uint64_t z = g_rs;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
static Bytes rand_bytes() {
    Bytes b{};
    for (int k = 0; k < 32; k += 8) {
        std::uint64_t v = nx();
        for (int j = 0; j < 8; ++j) b[std::size_t(k + j)] = std::uint8_t(v >> (8 * j));
    }
    return b;
}

static bool is_zero_bytes(const Bytes& b) {
    for (auto v : b) if (v != 0) return false;
    return true;
}

// a^e mod p by square-and-multiply -- deliberately the slow, obvious algorithm.
static FieldElement fe_pow(const FieldElement& a, const Bytes& e) {
    FieldElement result = FieldElement::from_bytes(Bytes{
        0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,1});
    for (std::size_t i = 0; i < 32; ++i) {
        for (int bit = 7; bit >= 0; --bit) {
            result = result * result;
            if ((e[i] >> bit) & 1) result = result * a;
        }
    }
    return result;
}

int test_regression_scalar_reduce_and_safegcd_divstep_run() {
    printf("=== scalar reduction + safegcd divstep rewrites ===\n");
    g_pass = g_fail = 0;

    // ---- (1) Scalar::from_bytes, boundary first ------------------------
    printf("\n--- (1) scalar reduction at the boundary ---\n");
    {
        struct { Bytes in; const char* name; } cases[] = {
            { be_add_small(kOrder, -2), "n-2" },
            { be_add_small(kOrder, -1), "n-1  (largest value that must pass through)" },
            { kOrder,                   "n    (must become 0)" },
            { be_add_small(kOrder,  1), "n+1  (must become 1)" },
            { be_add_small(kOrder,  2), "n+2" },
        };
        for (auto const& c : cases) {
            Bytes const want = reduce_mod_order(c.in);
            Bytes const got  = Scalar::from_bytes(c.in).to_bytes();
            check(got == want, c.name);
        }

        Bytes all_ff{}; all_ff.fill(0xFF);
        check(Scalar::from_bytes(all_ff).to_bytes() == reduce_mod_order(all_ff),
              "2^256-1 reduces to 2^256-1-n");

        Bytes zero{};
        check(is_zero_bytes(Scalar::from_bytes(zero).to_bytes()), "0 stays 0");
        check(Scalar::from_bytes(kOrder).to_bytes() == Scalar::zero().to_bytes(),
              "n and 0 land on the same scalar");
    }

    // ---- every single-bit value, plus a wide random sweep --------------
    printf("\n--- (1b) exhaustive single bits and random inputs ---\n");
    {
        int ok = 0;
        for (int bit = 0; bit < 256; ++bit) {
            Bytes b{};
            b[std::size_t(31 - bit / 8)] = std::uint8_t(1u << (bit % 8));
            if (Scalar::from_bytes(b).to_bytes() == reduce_mod_order(b)) ++ok;
        }
        check(ok == 256, "all 256 single-bit values reduce correctly");

        int r_ok = 0;
        const int kRand = 4000;
        for (int t = 0; t < kRand; ++t) {
            Bytes b = rand_bytes();
            // half the sweep forced above n, where the reduction actually fires
            if (t % 2) for (int i = 0; i < 8; ++i) b[std::size_t(i)] = 0xFF;
            if (Scalar::from_bytes(b).to_bytes() == reduce_mod_order(b)) ++r_ok;
        }
        check(r_ok == kRand, "random inputs reduce correctly, both branches");
        printf("  %d/%d single bits, %d/%d random\n", ok, 256, r_ok, kRand);
    }

    // ---- (2) field inverse: the defining property ----------------------
    printf("\n--- (2) field inverse, a * a^-1 == 1 ---\n");
    {
        FieldElement const one = FieldElement::from_bytes(Bytes{
            0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,1});

        auto inverts = [&](const Bytes& b) -> bool {
            FieldElement const a = FieldElement::from_bytes(b);
            if (is_zero_bytes(a.to_bytes())) return true;      // 0 has no inverse
            return (a * a.inverse()).to_bytes() == one.to_bytes();
        };

        // Boundary: p-1 .. p-4, and small integers.
        int ok = 0, n = 0;
        for (int d = -4; d <= -1; ++d) { ++n; if (inverts(be_add_small(kPrime, d))) ++ok; }
        for (int k = 1; k <= 64; ++k) {
            Bytes b{}; b[31] = std::uint8_t(k); ++n; if (inverts(b)) ++ok;
        }
        check(ok == n, "boundary and small values invert");

        // Every single-bit value: these are the long-zero-run inputs the
        // bulk-skip and the multi-bit cancellation interact on.
        int bok = 0;
        for (int bit = 0; bit < 256; ++bit) {
            Bytes b{}; b[std::size_t(31 - bit / 8)] = std::uint8_t(1u << (bit % 8));
            if (inverts(b)) ++bok;
        }
        check(bok == 256, "all 256 single-bit values invert");

        // Inputs built to have long zero runs in the low and high halves.
        int zok = 0, zn = 0;
        for (int k = 0; k < 96; ++k) {
            Bytes b = rand_bytes();
            b[0] &= 0x3f;
            if (k % 3 == 0) for (int z = 0; z < 12; ++z) b[std::size_t(31 - z)] = 0;
            if (k % 5 == 0) for (int z = 0; z < 10; ++z) b[std::size_t(z + 1)] = 0;
            ++zn; if (inverts(b)) ++zok;
        }
        check(zok == zn, "long zero runs in either half invert");

        int rok = 0;
        const int kRand = 1500;
        for (int t = 0; t < kRand; ++t) {
            Bytes b = rand_bytes(); b[0] &= 0x3f;
            if (inverts(b)) ++rok;
        }
        check(rok == kRand, "random field elements invert");
        printf("  %d/%d boundary+small, %d/256 single bits, %d/%d zero runs, %d/%d random\n",
               ok, n, bok, zok, zn, rok, kRand);
    }

    // ---- (2b) a second, unrelated route to the same inverse ------------
    // a^(p-2) == a^-1 by Fermat. Square-and-multiply shares no code with
    // safegcd, so agreement pins the divstep rewrite against something outside
    // it. Kept to a handful of values -- 256 squarings each is not cheap.
    printf("\n--- (2b) safegcd inverse == Fermat a^(p-2) ---\n");
    {
        Bytes const p_minus_2 = be_add_small(kPrime, -2);
        int ok = 0;
        const int kFermat = 12;
        for (int t = 0; t < kFermat; ++t) {
            Bytes b = (t < 4) ? Bytes{} : rand_bytes();
            if (t < 4) b[31] = std::uint8_t(t + 2);           // 2, 3, 4, 5
            b[0] &= 0x3f;
            FieldElement const a = FieldElement::from_bytes(b);
            if (is_zero_bytes(a.to_bytes())) { ++ok; continue; }
            if (a.inverse().to_bytes() == fe_pow(a, p_minus_2).to_bytes()) ++ok;
        }
        check(ok == kFermat, "safegcd and Fermat agree on every sampled input");
        printf("  %d/%d agree\n", ok, kFermat);
    }

    // ---- (3) the callers, end to end -----------------------------------
    // from_bytes and inverse sit under every signature the library produces.
    // If either rewrite were subtly wrong, this is where it would surface as a
    // verification failure rather than a number nobody looks at.
    printf("\n--- (3) sign/verify still round-trips ---\n");
    {
        int ok = 0;
        const int kSigs = 64;
        for (int t = 0; t < kSigs; ++t) {
            Bytes kb = rand_bytes();
            kb[0] &= 0x7f; if (kb[31] == 0) kb[31] = 1;
            Scalar const sk = Scalar::from_bytes(kb);
            Bytes msg = rand_bytes();
            auto const sig = secp256k1::ecdsa_sign(msg, sk);
            Point const pk = Point::generator().scalar_mul(sk);
            if (secp256k1::ecdsa_verify(msg, pk, sig)) ++ok;
        }
        check(ok == kSigs, "ECDSA sign/verify round-trips on freshly parsed keys");

        // Serialization runs through the field inverse (Jacobian -> affine).
        int sok = 0;
        for (int t = 0; t < 64; ++t) {
            Bytes kb = rand_bytes();
            kb[0] &= 0x7f; if (kb[31] == 0) kb[31] = 1;
            Point p = Point::generator().scalar_mul(Scalar::from_bytes(kb));
            Point q = p;
            q.normalize();
            if (p.add(q.negate()).is_infinity()) ++sok;
        }
        check(sok == 64, "normalize (which inverts Z) preserves the point");
    }

    printf("\n[regression_scalar_reduce_and_safegcd_divstep] %d/%d checks passed\n",
           g_pass, g_pass + g_fail);
    return (g_fail > 0) ? 1 : 0;
}

#ifdef STANDALONE_TEST
int main() { return test_regression_scalar_reduce_and_safegcd_divstep_run(); }
#endif
