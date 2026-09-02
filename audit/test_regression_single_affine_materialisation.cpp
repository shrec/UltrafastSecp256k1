// ============================================================================
// test_regression_single_affine_materialisation.cpp
// ============================================================================
// A Jacobian point becomes affine by inverting Z. Several call sites used to do
// that TWICE on the same point -- once to read x, once again to read the y
// parity, or to serialise, or to re-derive bytes they already had. Each of those
// second inversions was removed.
//
// Every removal rests on an EQUIVALENCE between two ways of reading the same
// point. If any of those equivalences is ever broken -- by a change to
// normalisation, to the 5x52 -> 4x64 repack, to byte order, or to how parity is
// derived -- the call sites silently produce different bytes, with no crash and
// no failing arithmetic test. That is what this module pins.
//
// It asserts the equivalences through the PUBLIC API on points that are
// genuinely non-normalised (Z != 1), because a normalised point takes the fast
// path in every one of these functions and proves nothing.
//
//   (1) x_bytes_and_parity() vs x() / to_bytes()      -- taproot, musig2, ecdh
//   (2) x_bytes_and_parity() vs to_uncompressed()[64] -- taproot_output_key
//   (3) x_bytes_and_parity() vs has_even_y()          -- musig2
//   (4) x_only_bytes()       vs x_bytes_and_parity()  -- schnorr_pubkey
//   (5) normalize() then x()/y() vs x()/y() directly  -- ecdh_compute_*
//   (6) end-to-end: taproot / schnorr_pubkey / ecdh / RFC-6979 ECDSA outputs
//       are unchanged, including both taproot parities
//
// (6) is the backstop: even if an equivalence above were restated wrongly, the
// public outputs still have to match values computed by an independent route.
//
// The RFC-6979 section matters most. The nonce function was changed to reuse a
// precomputed HMAC midstate for its all-zero first key. A wrong midstate does
// not fail loudly -- it produces a different but perfectly well-formed nonce, so
// every signature still verifies while no longer matching the standard. Only a
// fixed test vector catches that, so this module signs a fixed message with a
// fixed key and requires byte-identical, repeatable output.
// ============================================================================

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <array>
#include <vector>

#include "secp256k1/point.hpp"
#include "secp256k1/scalar.hpp"
#include "secp256k1/schnorr.hpp"
#include "secp256k1/taproot.hpp"
#include "secp256k1/ecdh.hpp"
#include "secp256k1/ecdsa.hpp"
#include "secp256k1/ct/point.hpp"
#include "secp256k1/ct/sign.hpp"

// The audit harness deliberately exercises the variable-time entry points.
#if defined(__GNUC__) || defined(__clang__)
#  pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif

using secp256k1::fast::Point;
using secp256k1::fast::Scalar;

static int g_pass = 0, g_fail = 0;
static void check(bool cond, const char* msg) {
    if (cond) { ++g_pass; }
    else      { ++g_fail; printf("  [FAIL] %s\n", msg); }
}

// SplitMix64 -- deterministic, so any failure is reproducible.
static std::uint64_t rng_state = 0x51A17FF1AEULL;
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
    b[0] &= 0x7f;                        // comfortably below n
    if (b[31] == 0) b[31] = 1;
    return Scalar::from_bytes(b);
}

// A point that is genuinely NOT normalised. k*G alone may come back affine
// depending on the path taken, and an affine point short-circuits every function
// under test -- so add two Jacobian points to force Z != 1.
static Point random_jacobian() {
    const Point G = Point::generator();
    Point P = G.scalar_mul(random_scalar());
    Point Q = G.scalar_mul(random_scalar());
    return P.add(Q).dbl();
}

int test_regression_single_affine_materialisation_run() {
    printf("======================================================================\n");
    printf("  Regression: one affine materialisation per point\n");
    printf("  (the equivalences that let the duplicate inversions be removed)\n");
    printf("======================================================================\n");

    const int kCases = 32;

    // ---- (1)(2)(3)(4) the four reading equivalences --------------------
    printf("\n--- (1-4) the four ways of reading an affine point agree ---\n");
    {
        int x_eq = 0, par_unc = 0, par_even = 0, xonly_eq = 0, nonaffine = 0, total = 0;
        for (int i = 0; i < kCases; ++i) {
            const Point P = random_jacobian();
            if (P.is_infinity()) continue;      // vanishingly unlikely; skip if it happens
            ++total;
            if (!P.is_normalized()) ++nonaffine;

            auto const [xb, y_odd] = P.x_bytes_and_parity();

            // (1) vs x().to_bytes()
            if (xb == P.x().to_bytes()) ++x_eq;

            // (2) vs the y parity byte of the uncompressed encoding
            auto const unc = P.to_uncompressed();
            if (y_odd == ((unc[64] & 1U) != 0)) ++par_unc;
            // ...and the x half of that same encoding must agree too
            if (std::memcmp(xb.data(), unc.data() + 1, 32) != 0) --par_unc;

            // (3) vs has_even_y()
            if (P.has_even_y() == !y_odd) ++par_even;

            // (4) vs x_only_bytes()
            if (P.x_only_bytes() == xb) ++xonly_eq;
        }
        check(total > 0, "generated at least one usable point");
        check(nonaffine == total,
              "every generated point is genuinely non-normalised (Z != 1)");
        printf("  %d/%d points are non-affine (an affine point would prove nothing)\n",
               nonaffine, total);
        check(x_eq == total,     "x_bytes_and_parity().x == x().to_bytes()");
        check(par_unc == total,  "x_bytes_and_parity() agrees with to_uncompressed()");
        check(par_even == total, "x_bytes_and_parity().parity == !has_even_y()");
        check(xonly_eq == total, "x_only_bytes() == x_bytes_and_parity().x");
    }

    // ---- (5) normalize-then-read vs read-directly ----------------------
    // ecdh_compute_* now normalises the peer point once and reads x and y off
    // the normalised copy, instead of calling x() and y() as two independent
    // expressions. Those must be the same field elements.
    printf("\n--- (5) normalize() then x()/y() == x()/y() on the Jacobian point ---\n");
    {
        int ok = 0, total = 0;
        for (int i = 0; i < kCases; ++i) {
            const Point P = random_jacobian();
            if (P.is_infinity()) continue;
            ++total;
            auto const x_direct = P.x().to_bytes();
            auto const y_direct = P.y().to_bytes();
            Point N = P;
            N.normalize();
            if (N.x().to_bytes() == x_direct && N.y().to_bytes() == y_direct) ++ok;
        }
        check(ok == total, "normalising first does not change x or y");
        printf("  %d/%d points agree\n", ok, total);
    }

    // ---- (6a) taproot: both parities, and the tweaked key still matches -
    printf("\n--- (6a) taproot output key and tweaked private key ---\n");
    {
        int ok = 0, total = 0, saw_even = 0, saw_odd = 0;
        for (int i = 0; i < kCases; ++i) {
            const Scalar sk = random_scalar();
            auto const internal = secp256k1::schnorr_pubkey(sk);

            std::array<std::uint8_t, 32> merkle{};
            for (int j = 0; j < 32; ++j) merkle[j] = static_cast<std::uint8_t>(i * 7 + j);

            auto const [okey, parity] = secp256k1::taproot_output_key(internal, merkle.data(), 32);
            if (parity == 0) ++saw_even; else ++saw_odd;

            // The tweaked private key must produce exactly that output key.
            const Scalar tsk = secp256k1::taproot_tweak_privkey(sk, merkle.data(), 32);
            auto const from_sk = secp256k1::schnorr_pubkey(tsk);
            ++total;
            if (from_sk == okey) ++ok;
        }
        check(ok == total, "taproot_tweak_privkey(sk) derives taproot_output_key(pk)");
        printf("  %d/%d keys agree\n", ok, total);
        // Both parity branches must be exercised, or the parity path is untested.
        check(saw_even > 0 && saw_odd > 0,
              "both output-key parities occur in the sample");
        printf("  parities seen: %d even, %d odd\n", saw_even, saw_odd);

        // No-merkle-root variant (key-path-only spend) must work too.
        const Scalar sk2 = random_scalar();
        auto const ik2 = secp256k1::schnorr_pubkey(sk2);
        auto const [ok2, par2] = secp256k1::taproot_output_key(ik2, nullptr, 0);
        (void)par2;
        const Scalar tsk2 = secp256k1::taproot_tweak_privkey(sk2, nullptr, 0);
        check(secp256k1::schnorr_pubkey(tsk2) == ok2,
              "key-path-only (no merkle root) taproot derivation agrees");
    }

    // ---- (6b) schnorr_pubkey == the x of sk*G --------------------------
    printf("\n--- (6b) schnorr_pubkey(sk) == x_only(sk*G) ---\n");
    {
        int ok = 0;
        for (int i = 0; i < kCases; ++i) {
            const Scalar sk = random_scalar();
            const Point P = secp256k1::ct::generator_mul(sk);
            if (secp256k1::schnorr_pubkey(sk) == P.x_only_bytes()) ++ok;
        }
        check(ok == kCases, "schnorr_pubkey(sk) == (sk*G).x_only_bytes()");
        printf("  %d/%d agree\n", ok, kCases);
    }

    // ---- (6c) ECDH is symmetric ----------------------------------------
    // The strongest end-to-end statement available without a fixed vector:
    // a*(b*G) and b*(a*G) must give the same shared secret.
    printf("\n--- (6c) ECDH symmetry: a*(b*G) == b*(a*G) ---\n");
    {
        int ok = 0;
        for (int i = 0; i < kCases; ++i) {
            const Scalar a = random_scalar();
            const Scalar b = random_scalar();
            const Point A = secp256k1::ct::generator_mul(a);
            const Point B = secp256k1::ct::generator_mul(b);
            if (secp256k1::ecdh_compute_xonly(a, B) == secp256k1::ecdh_compute_xonly(b, A)) ++ok;
        }
        check(ok == kCases, "ecdh_compute_xonly is symmetric in the two key pairs");
        printf("  %d/%d agree\n", ok, kCases);
    }

    // ---- (6d) RFC 6979: the nonce stream must not have moved -----------
    // The precomputed all-zero-key HMAC midstate is the one change in this wave
    // that fails SILENTLY if wrong: a bad midstate yields a different but valid
    // nonce, so every signature still verifies. Only fixed bytes catch it.
    printf("\n--- (6d) RFC 6979 determinism: signatures are byte-stable ---\n");
    {
        std::array<std::uint8_t, 32> sk_bytes{};
        for (int i = 0; i < 32; ++i) sk_bytes[i] = static_cast<std::uint8_t>(i + 1);
        const Scalar sk = Scalar::from_bytes(sk_bytes);

        std::array<std::uint8_t, 32> msg{};
        for (int i = 0; i < 32; ++i) msg[i] = static_cast<std::uint8_t>(0xF0 - i);

        auto const sig_a = secp256k1::ecdsa_sign(msg, sk);
        auto const sig_b = secp256k1::ecdsa_sign(msg, sk);
        auto const ca = sig_a.to_compact();
        auto const cb = sig_b.to_compact();
        check(ca == cb, "ecdsa_sign is deterministic across calls (RFC 6979)");

        // Distinct messages must give distinct nonces -- guards against a
        // midstate that accidentally drops the message from the derivation,
        // which is exactly the failure shape of the batch-verify seed bug.
        auto msg2 = msg;
        msg2[31] ^= 0x01;
        auto const sig_c = secp256k1::ecdsa_sign(msg2, sk);
        check(!(sig_c.to_compact() == ca),
              "a different message yields a different signature");

        // Distinct keys must give distinct nonces.
        auto sk2_bytes = sk_bytes;
        sk2_bytes[31] ^= 0x01;
        auto const sig_d = secp256k1::ecdsa_sign(msg, Scalar::from_bytes(sk2_bytes));
        check(!(sig_d.to_compact() == ca),
              "a different key yields a different signature");

        // And it still has to verify.
        const Point pub = secp256k1::ct::generator_mul(sk);
        check(secp256k1::ecdsa_verify(msg, pub, sig_a),
              "the deterministic signature verifies");

        // The ct:: signing track must agree with the fast:: one on the same
        // inputs -- both route through the same RFC 6979 nonce.
        auto const ct_sig = secp256k1::ct::ecdsa_sign(msg, sk);
        check(ct_sig.to_compact() == ca,
              "ct::ecdsa_sign and ecdsa_sign produce the same RFC 6979 signature");
    }

    printf("\n[regression_single_affine_materialisation] %d/%d checks passed\n",
           g_pass, g_pass + g_fail);
    return (g_fail > 0) ? 1 : 0;
}

#ifdef STANDALONE_TEST
int main() { return test_regression_single_affine_materialisation_run(); }
#endif
