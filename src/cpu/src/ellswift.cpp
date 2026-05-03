// ============================================================================
// ElligatorSwift encoding for secp256k1 (BIP-324)
// ============================================================================
// Implements the XSwiftEC / ElligatorSwift algorithm as specified in BIP-324.
//
// The core idea: for any secp256k1 x-coordinate, there exist many (u, t) pairs
// such that XSwiftEC(u, t) = x. We pick a random u and solve for t.
//
// XSwiftEC(u, t):
//   Let c = u^3 + 7 (secp256k1 curve constant b=7)
//   s = (-1 - u^3 - 7) / (u^3 + u^2 * t^2 * (-3 - u^2) + 7)  ... simplified
//   ...
//
// The actual algorithm follows Bitcoin Core's libsecp256k1 implementation
// which uses the formulation from the paper by Chavez-Saab et al.
//
// References:
//   - BIP-324: https://github.com/bitcoin/bips/blob/master/bip-0324.mediawiki
//   - libsecp256k1 src/modules/ellswift/
// ============================================================================

#include "secp256k1/ellswift.hpp"
#include "secp256k1/sha256.hpp"
#include "secp256k1/hkdf.hpp"
#include "secp256k1/ecdh.hpp"
#include "secp256k1/ct/point.hpp"
#include "secp256k1/precompute.hpp"
#include "secp256k1/detail/secure_erase.hpp"
#include <cstring>
#include <stdexcept>

#include "secp256k1/detail/csprng.hpp"

namespace secp256k1 {

using fast::Scalar;
using fast::Point;
using fast::FieldElement;

namespace {

using secp256k1::detail::csprng_fill;

// secp256k1 curve constant b = 7
static const FieldElement FE_SEVEN = FieldElement::from_uint64(7);

// FieldElement from 32-byte big-endian (mod p), always succeeds
FieldElement fe_from_bytes_mod_p(const std::uint8_t bytes[32]) noexcept {
    std::array<std::uint8_t, 32> arr;
    std::memcpy(arr.data(), bytes, 32);
    return FieldElement::from_bytes(arr);
}

// Check if a field element is a quadratic residue mod p
bool fe_is_square(const FieldElement& x) noexcept {
    if (x == FieldElement::zero()) return true;
    auto s = x.sqrt();
    return (s.square() == x);
}

// XSwiftEC forward map (correct BIP-324 algorithm from libsecp256k1).
//
// Constants:
//   c1 = (sqrt(-3)-1)/2
//   c2 = (-sqrt(-3)-1)/2
//
// Algorithm:
//   If u=0: u=1.
//   If t=0: s=1, else s=t^2.
//   g = u^3+7.
//   If g+s=0: s=4*s.
//   Try x3 = (3*s*u^3-(g+s)^2)/(3*s*u^2). If on curve, return.
//   Try x2 = u*(c1*s+c2*g)/(g+s). If on curve, return.
//   Return x1 = -(x2+u).
FieldElement xswiftec_fwd(FieldElement u, FieldElement t) noexcept {
    // c1 = (sqrt(-3)-1)/2
    static const FieldElement C1 = []() {
        std::array<uint8_t, 32> b = {
            0x85,0x16,0x95,0xd4, 0x9a,0x83,0xf8,0xef,
            0x91,0x9b,0xb8,0x61, 0x53,0xcb,0xcb,0x16,
            0x63,0x0f,0xb6,0x8a, 0xed,0x0a,0x76,0x6a,
            0x3e,0xc6,0x93,0xd6, 0x8e,0x6a,0xfa,0x40
        };
        return FieldElement::from_bytes(b);
    }();
    // c2 = (-sqrt(-3)-1)/2
    static const FieldElement C2 = []() {
        std::array<uint8_t, 32> b = {
            0x7a,0xe9,0x6a,0x2b, 0x65,0x7c,0x07,0x10,
            0x6e,0x64,0x47,0x9e, 0xac,0x34,0x34,0xe9,
            0x9c,0xf0,0x49,0x75, 0x12,0xf5,0x89,0x95,
            0xc1,0x39,0x6c,0x28, 0x71,0x95,0x01,0xee
        };
        return FieldElement::from_bytes(b);
    }();
    static const FieldElement FE_ZERO  = FieldElement::zero();
    static const FieldElement FE_ONE   = FieldElement::one();
    static const FieldElement FE_THREE = FieldElement::from_uint64(3);
    static const FieldElement FE_FOUR  = FieldElement::from_uint64(4);

    if (u == FE_ZERO) u = FE_ONE;

    FieldElement s = (t == FE_ZERO) ? FE_ONE : t.square();

    auto u2 = u.square();
    auto u3 = u2 * u;
    auto g  = u3 + FE_SEVEN;   // g = u^3+7

    // p = g+s; if zero, replace s=4*s and recompute p
    auto p = g + s;
    if (p == FE_ZERO) {
        s = FE_FOUR * s;
        p = g + s;
    }

    // d = 3*s*u^2
    auto d  = FE_THREE * s * u2;

    // Try x3 = (3*s*u^3 - (g+s)^2) / (3*s*u^2) = (d*u - p^2) / d
    auto n3 = d * u - p.square();
    auto x3 = n3 * d.inverse();
    if (fe_is_square(x3 * x3 * x3 + FE_SEVEN)) return x3;

    // Try x2 = u*(c1*s + c2*g) / (g+s)
    auto n2 = (C1 * s + C2 * g) * u;
    auto x2 = n2 * p.inverse();
    if (fe_is_square(x2 * x2 * x2 + FE_SEVEN)) return x2;

    // Return x1 = -(x2+u)
    return (x2 + u).negate();
}

// XSwiftEC inverse: given x (on curve) and u (must be nonzero), find t such that
// xswiftec_fwd(u,t)=x. c (0-7) selects which of up to 8 solutions to try.
// Returns {true, t} on success; {false, _} if no solution for this c.
//
// Ported from libsecp256k1 secp256k1_ellswift_xswiftec_inv_var.
//
// c3 = (-sqrt(-3)+1)/2 = -c1 = c2+1
// c4 = ( sqrt(-3)+1)/2 = -c2 = c1+1
//
// If c&2=0 (x1/x2 case):
//   Fail if (-u-x) is on curve.
//   s = -(u^3+7)/(u^2+u*x+x^2). Fail if not square.
//   v = x.
// If c&2=2 (x3 case):
//   s = x-u. Fail if not square or zero.
//   r = sqrt(-s*(4*(u^3+7)+3*u^2*s)). Fail if doesn't exist.
//   Fail if c&1=1 and r=0.
//   v = (r/s-u)/2.
// w = sqrt(s). Negate if (c&5)==0 or (c&5)==5.
// Return w * ((c&1 ? c4 : c3)*u + v).
// Compute sqrt and check in ONE exponentiation (vs fe_is_square + sqrt = 2x).
static inline std::pair<bool, FieldElement> sqrt_check(const FieldElement& x) noexcept {
    auto r = x.sqrt();
    return {r.square() == x, r};
}

// Optimized per-u encoder: precomputes u², u³, g, neg_ux-check and s_x3-sqrt
// ONCE per u, then tries all 8 c values sharing those results.
// Savings: 6 duplicate sqrt calls (each ~5.2µs) → eliminated.
// On success: fills t_out and returns true.  y_odd drives negate-t.
static bool ellswift_try_u(
    const FieldElement& x,
    const FieldElement& u,
    bool y_odd,
    FieldElement& t_out) noexcept
{
    static const FieldElement FE_ZERO  = FieldElement::zero();
    static const FieldElement FE_THREE = FieldElement::from_uint64(3);
    static const FieldElement FE_FOUR  = FieldElement::from_uint64(4);
    static const FieldElement TWO_INV  = FieldElement::from_uint64(2).inverse();
    static const FieldElement C3 = [](){
        std::array<uint8_t,32> b={0x7a,0xe9,0x6a,0x2b,0x65,0x7c,0x07,0x10,
                                   0x6e,0x64,0x47,0x9e,0xac,0x34,0x34,0xe9,
                                   0x9c,0xf0,0x49,0x75,0x12,0xf5,0x89,0x95,
                                   0xc1,0x39,0x6c,0x28,0x71,0x95,0x01,0xef};
        return FieldElement::from_bytes(b);}();
    static const FieldElement C4 = [](){
        std::array<uint8_t,32> b={0x85,0x16,0x95,0xd4,0x9a,0x83,0xf8,0xef,
                                   0x91,0x9b,0xb8,0x61,0x53,0xcb,0xcb,0x16,
                                   0x63,0x0f,0xb6,0x8a,0xed,0x0a,0x76,0x6a,
                                   0x3e,0xc6,0x93,0xd6,0x8e,0x6a,0xfa,0x41};
        return FieldElement::from_bytes(b);}();

    // Precompute u-dependent values once.
    const auto u2 = u.square();
    const auto u3 = u2 * u;
    const auto g  = u3 + FE_SEVEN;
    const auto ux = u + x;

    // x1/x2 early-reject (shared for c={0,1,4,5}): 1 sqrt total, not 4.
    const auto neg_ux  = ux.negate();
    const bool rej_x12 = fe_is_square(neg_ux * neg_ux * neg_ux + FE_SEVEN);

    // x3: s = x-u. sqrt(s) shared for c={2,3,6,7}: 1 sqrt total, not 4.
    const auto s_x3          = x - u;
    const auto [s_x3sq, w_x3] = sqrt_check(s_x3);
    const bool s_x3ok         = s_x3sq && s_x3 != FE_ZERO;

    // --- x1/x2 case: sqrt(sp*g) shared across c={0,1,4,5} — 1 sqrt ---
    if (!rej_x12) {
        const auto sp            = ux.square().negate() + u * x;
        const auto [sq12, wc12]  = sqrt_check(sp * g);   // 1 sqrt (was 4)
        if (sq12) {
            const auto sp_inv = sp.inverse();
            const auto w_base = wc12 * sp_inv;  // sqrt(s) for x1/x2
            // Try c=0,1,4,5 — only sign of w and C3/C4 multiplier differ
            for (int c : {0, 1, 4, 5}) {
                auto w = ((c & 5) == 0 || (c & 5) == 5) ? w_base.negate() : w_base;
                auto t = w * (((c & 1) ? C4 : C3) * u + x);
                if (t == FE_ZERO) continue;
                if (((t.to_bytes()[31] & 1) != 0) != y_odd) t = t.negate();
                t_out = t;
                return true;
            }
        }
    }

    // --- x3 case: sqrt(r2) shared across c={2,3,6,7} — 1 sqrt ---
    if (s_x3ok) {
        const auto g4  = g * FE_FOUR;
        const auto r2  = (FE_THREE * u2 * s_x3 + g4).negate() * s_x3;
        const auto [rsq, r] = sqrt_check(r2);              // 1 sqrt (was 4)
        if (rsq) {
            const auto v_base = (r * s_x3.inverse() - u) * TWO_INV;
            // Try c=2,3,6,7 — only r==0 check and sign/multiplier differ
            for (int c : {2, 3, 6, 7}) {
                if ((c & 1) && r == FE_ZERO) continue;
                auto w = ((c & 5) == 0 || (c & 5) == 5) ? w_x3.negate() : w_x3;
                auto t = w * (((c & 1) ? C4 : C3) * u + v_base);
                if (t == FE_ZERO) continue;
                if (((t.to_bytes()[31] & 1) != 0) != y_odd) t = t.negate();
                t_out = t;
                return true;
            }
        }
    }
    return false;
}

std::pair<bool, FieldElement> xswiftec_inv(
    const FieldElement& x, const FieldElement& u, int c) noexcept {
    static const FieldElement C3 = []() {
        std::array<uint8_t, 32> b = {
            0x7a,0xe9,0x6a,0x2b, 0x65,0x7c,0x07,0x10,
            0x6e,0x64,0x47,0x9e, 0xac,0x34,0x34,0xe9,
            0x9c,0xf0,0x49,0x75, 0x12,0xf5,0x89,0x95,
            0xc1,0x39,0x6c,0x28, 0x71,0x95,0x01,0xef
        };
        return FieldElement::from_bytes(b);
    }();
    static const FieldElement C4 = []() {
        std::array<uint8_t, 32> b = {
            0x85,0x16,0x95,0xd4, 0x9a,0x83,0xf8,0xef,
            0x91,0x9b,0xb8,0x61, 0x53,0xcb,0xcb,0x16,
            0x63,0x0f,0xb6,0x8a, 0xed,0x0a,0x76,0x6a,
            0x3e,0xc6,0x93,0xd6, 0x8e,0x6a,0xfa,0x41
        };
        return FieldElement::from_bytes(b);
    }();
    static const FieldElement FE_ZERO  = FieldElement::zero();
    static const FieldElement FE_THREE = FieldElement::from_uint64(3);
    static const FieldElement FE_FOUR  = FieldElement::from_uint64(4);
    static const FieldElement TWO_INV  = FieldElement::from_uint64(2).inverse();

    FieldElement v, s, w;

    if (!(c & 2)) {
        // x1/x2 case
        auto neg_ux = (u + x).negate();
        if (fe_is_square(neg_ux * neg_ux * neg_ux + FE_SEVEN)) return {false, FE_ZERO};

        auto u2  = u.square();
        auto g   = u2 * u + FE_SEVEN;
        auto sp  = (u + x).square().negate() + u * x;
        // Check sp*g square AND get sqrt(s) in one shot via combined expression.
        // s = g/(-sp) = -(g/sp). w = sqrt(s). Check: w^2 == s.
        auto [sq, w_cand] = sqrt_check(sp * g);
        if (!sq) return {false, FE_ZERO};
        s = g * sp.inverse();
        // w_cand = sqrt(sp*g) = sqrt(s * sp^2) = |sp| * sqrt(s).
        // Recover sqrt(s): w = w_cand / |sp| = w_cand * sp.inverse().
        w = w_cand * sp.inverse();
        v = x;

    } else {
        // x3 case: compute both sqrt(s) and sqrt(r2) — each done once.
        s = x - u;
        auto [sq_s, w_s] = sqrt_check(s);  // sqrt(s) — reused as w below
        if (!sq_s || s == FE_ZERO) return {false, FE_ZERO};

        auto u2 = u.square();
        auto g4 = (u2 * u + FE_SEVEN) * FE_FOUR;
        auto r2 = (FE_THREE * u2 * s + g4).negate() * s;
        auto [sq_r, r] = sqrt_check(r2);   // sqrt(r2)
        if (!sq_r) return {false, FE_ZERO};
        if ((c & 1) && r == FE_ZERO) return {false, FE_ZERO};

        v = (r * s.inverse() - u) * TWO_INV;
        w = w_s;  // reuse sqrt(s) computed above — no extra sqrt call
    }

    if ((c & 5) == 0 || (c & 5) == 5) w = w.negate();
    auto t = w * (((c & 1) ? C4 : C3) * u + v);
    if (t == FE_ZERO) return {false, FE_ZERO};
    return {true, t};
}

} // anonymous namespace

// ============================================================================
// Public API
// ============================================================================

FieldElement ellswift_decode(const std::uint8_t encoding[64]) noexcept {
    auto u = fe_from_bytes_mod_p(encoding);
    auto t = fe_from_bytes_mod_p(encoding + 32);
    return xswiftec_fwd(u, t);
}

std::array<std::uint8_t, 64> ellswift_create(const Scalar& privkey) {
    // BIP-324: the encoding MUST be probabilistic so the same private key
    // produces a different 64-byte encoding on every call. Using zero auxrnd
    // makes it deterministic, which (a) leaks key identity across connections
    // and (b) causes the xdh hash (which includes the full encoding) to produce
    // the same shared secret for any two "different" encodings of the same key.
    // Fix: fill auxrnd from CSPRNG so each call is unique.
    std::uint8_t rand32[32];
    csprng_fill(rand32, 32);
    return ellswift_create(privkey, rand32);
}

std::array<std::uint8_t, 64> ellswift_create(const Scalar& privkey,
                                              const std::uint8_t* auxrnd32) {
    static constexpr std::uint8_t kZeroAux[32] = {};
    if (!auxrnd32) auxrnd32 = kZeroAux;  // null → zero auxrnd, same hash path

    // Deterministic u derivation: SHA256(tag||tag||privkey||0x00*32||auxrnd32||cnt)
    // matching libsecp256k1's secp256k1_ellswift_create semantics.
    auto pub = ct::generator_mul(privkey);
    auto x = pub.x();
    bool y_is_odd = (pub.y().to_bytes()[31] & 1) != 0;
    auto privkey_bytes = privkey.to_bytes();

    // Precompute tagged-hash prefix (tag applied twice per BIP-340 convention)
    static constexpr char kTag[] = "secp256k1_ellswift_create";
    static const auto kTagHash = SHA256::hash(
        reinterpret_cast<const std::uint8_t*>(kTag), sizeof(kTag) - 1);

    std::array<std::uint8_t, 64> result{};
    static constexpr int kMaxAttempts = 100;
    static constexpr std::uint8_t kZero32[32] = {};

    for (int attempt = 0; attempt < kMaxAttempts; ++attempt) {
        SHA256 h;
        h.update(kTagHash.data(), 32);
        h.update(kTagHash.data(), 32);
        h.update(privkey_bytes.data(), 32);
        h.update(kZero32, 32);
        h.update(auxrnd32, 32);
        auto cnt = static_cast<std::uint8_t>(attempt);
        h.update(&cnt, 1);
        auto rand_hash = h.finalize();

        std::uint8_t rand_bytes[32];
        std::memcpy(rand_bytes, rand_hash.data(), 32);

        auto u = fe_from_bytes_mod_p(rand_bytes);
        if (u == FieldElement::zero()) continue;

        FieldElement t;
        if (ellswift_try_u(x, u, y_is_odd, t)) {
            auto u_bytes = u.to_bytes();
            auto t_bytes = t.to_bytes();
            std::memcpy(result.data(),      u_bytes.data(), 32);
            std::memcpy(result.data() + 32, t_bytes.data(), 32);
            detail::secure_erase(rand_bytes, sizeof(rand_bytes));
            return result;
        }
    }
    throw std::runtime_error("ellswift_create: auxrnd32 path exhausted 100 attempts");
}

std::array<std::uint8_t, 64> ellswift_encode_x(const FieldElement& x,
                                               const std::uint8_t rnd32[32]) {
    std::array<std::uint8_t, 64> result{};

    // Derive a starting u from rnd32, then try all 8 inverse cases.
    // Expected ~1.14 u values needed.
    static constexpr int kMaxAttempts = 100;
    for (int attempt = 0; attempt < kMaxAttempts; ++attempt) {
        std::uint8_t rand_bytes[32];
        if (attempt == 0 && rnd32) {
            std::memcpy(rand_bytes, rnd32, 32);
        } else {
            csprng_fill(rand_bytes, 32);
        }

        auto u = fe_from_bytes_mod_p(rand_bytes);
        if (u == FieldElement::zero()) continue;

        for (int c = 0; c < 8; ++c) {
            auto [ok, t] = xswiftec_inv(x, u, c);
            if (!ok) continue;

            auto u_bytes = u.to_bytes();
            auto t_bytes = t.to_bytes();
            std::memcpy(result.data(),      u_bytes.data(), 32);
            std::memcpy(result.data() + 32, t_bytes.data(), 32);

            if (xswiftec_fwd(u, t) == x) {
                return result;
            }
        }
    }
    if (rnd32) {
        std::memcpy(result.data(),      rnd32, 32);
        std::memcpy(result.data() + 32, rnd32, 32);
    }
    return result;
}

std::array<std::uint8_t, 32> ellswift_xdh(
    const std::uint8_t ell_a64[64],
    const std::uint8_t ell_b64[64],
    const Scalar& our_privkey,
    bool initiating) noexcept {

    // 1. Decode their ElligatorSwift to an x-coordinate
    const std::uint8_t* their_ell = initiating ? ell_b64 : ell_a64;
    auto their_x = ellswift_decode(their_ell);

    // 2. Recover their point from x-coordinate (even y)
    auto x2 = their_x.square();
    auto x3 = x2 * their_x;
    auto y2 = x3 + FE_SEVEN;
    auto y = y2.sqrt();
    // Verify it's a valid point
    if (!(y.square() == y2)) {
        return std::array<std::uint8_t, 32>{};
    }

    // Pick even y (parity doesn't matter for x-only ECDH)
    auto y_bytes = y.to_bytes();
    if (y_bytes[31] & 1) {
        y = y.negate();
    }

    auto their_point = Point::from_affine(their_x, y);
    if (their_point.is_infinity()) {
        return std::array<std::uint8_t, 32>{};
    }

    // 3. ECDH: shared_secret = SHA256(tag || tag || ell_a || ell_b || x(privkey * their_point))
    // Constant-time: our_privkey is secret
    auto ecdh_point = ct::scalar_mul(their_point, our_privkey);
    if (ecdh_point.is_infinity()) {
        return std::array<std::uint8_t, 32>{};
    }
    auto ecdh_x = ecdh_point.x().to_bytes();

    // BIP-324 specifies: shared_secret = SHA256(
    //   SHA256("bip324_ellswift_xonly_ecdh") || SHA256("bip324_ellswift_xonly_ecdh") ||
    //   ell_a64 || ell_b64 || ecdh_x)
    //
    // This is the tagged hash: SHA256_tagged("bip324_ellswift_xonly_ecdh", ell_a || ell_b || x)

    // Compute the tag hash
    constexpr char tag_str[] = "bip324_ellswift_xonly_ecdh";
    auto tag_hash = SHA256::hash(tag_str, sizeof(tag_str) - 1);

    // Tagged hash: SHA256(tag_hash || tag_hash || ell_a || ell_b || ecdh_x)
    SHA256 hasher;
    hasher.update(tag_hash.data(), 32);
    hasher.update(tag_hash.data(), 32);
    hasher.update(ell_a64, 64);
    hasher.update(ell_b64, 64);
    hasher.update(ecdh_x.data(), 32);
    auto shared_secret = hasher.finalize();

    detail::secure_erase(ecdh_x.data(), 32);
    detail::secure_erase(&ecdh_point, sizeof(ecdh_point));

    return shared_secret;
}

std::array<std::uint8_t, 64> ellswift_create_fast(const Scalar& privkey) {
    // Non-CT: use precomputed fixed-base table + SHA256 u derivation.
    auto pub = scalar_mul_generator(privkey);
    // x_bytes_and_parity(): one field inverse instead of separate x()+y() calls.
    auto [x_bytes, y_odd] = pub.x_bytes_and_parity();
    auto x = FieldElement::from_bytes(x_bytes);
    auto privkey_bytes = privkey.to_bytes();

    static constexpr char kTag[] = "secp256k1_ellswift_create";
    static const auto kTagHash = SHA256::hash(
        reinterpret_cast<const std::uint8_t*>(kTag), sizeof(kTag) - 1);
    static constexpr std::uint8_t kZero32[32] = {};

    std::array<std::uint8_t, 64> result{};
    static constexpr int kMaxAttempts = 100;
    for (int attempt = 0; attempt < kMaxAttempts; ++attempt) {
        SHA256 h;
        h.update(kTagHash.data(), 32); h.update(kTagHash.data(), 32);
        h.update(privkey_bytes.data(), 32);
        h.update(kZero32, 32); h.update(kZero32, 32);
        auto cnt = static_cast<std::uint8_t>(attempt);
        h.update(&cnt, 1);
        auto rand_hash = h.finalize();
        std::uint8_t rand_bytes[32];
        std::memcpy(rand_bytes, rand_hash.data(), 32);

        auto u = fe_from_bytes_mod_p(rand_bytes);
        if (u == FieldElement::zero()) continue;

        FieldElement t;
        if (ellswift_try_u(x, u, y_odd, t)) {
            auto u_bytes = u.to_bytes();
            auto t_bytes = t.to_bytes();
            std::memcpy(result.data(),      u_bytes.data(), 32);
            std::memcpy(result.data() + 32, t_bytes.data(), 32);
            detail::secure_erase(rand_bytes, sizeof(rand_bytes));
            return result;
        }
    }
    throw std::runtime_error("ellswift_create_fast: RNG produced unusable values");
}

std::array<std::uint8_t, 64> ellswift_create_fast(const Scalar& privkey,
                                                    const std::uint8_t* auxrnd32) {
    if (!auxrnd32) return ellswift_create_fast(privkey);

    auto pub = scalar_mul_generator(privkey);
    auto [x_bytes_a, y_odd] = pub.x_bytes_and_parity();
    auto x = FieldElement::from_bytes(x_bytes_a);
    auto privkey_bytes = privkey.to_bytes();

    static constexpr char kTag[] = "secp256k1_ellswift_create";
    static const auto kTagHash = SHA256::hash(
        reinterpret_cast<const std::uint8_t*>(kTag), sizeof(kTag) - 1);
    static constexpr std::uint8_t kZero32[32] = {};

    std::array<std::uint8_t, 64> result{};
    for (int attempt = 0; attempt < 100; ++attempt) {
        SHA256 h;
        h.update(kTagHash.data(), 32); h.update(kTagHash.data(), 32);
        h.update(privkey_bytes.data(), 32);
        h.update(kZero32, 32);
        h.update(auxrnd32, 32);
        auto cnt = static_cast<std::uint8_t>(attempt);
        h.update(&cnt, 1);
        auto rand_hash = h.finalize();
        std::uint8_t rand_bytes[32];
        std::memcpy(rand_bytes, rand_hash.data(), 32);

        auto u = fe_from_bytes_mod_p(rand_bytes);
        if (u == FieldElement::zero()) continue;

        FieldElement t;
        if (ellswift_try_u(x, u, y_odd, t)) {
            auto u_bytes = u.to_bytes(); auto t_bytes = t.to_bytes();
            std::memcpy(result.data(),    u_bytes.data(), 32);
            std::memcpy(result.data()+32, t_bytes.data(), 32);
            detail::secure_erase(rand_bytes, sizeof(rand_bytes));
            return result;
        }
    }
    throw std::runtime_error("ellswift_create_fast: auxrnd path exhausted");
}

std::array<std::uint8_t, 32> ellswift_xdh_fast(
    const std::uint8_t ell_a64[64],
    const std::uint8_t ell_b64[64],
    const Scalar& our_privkey,
    bool initiating) noexcept {

    const std::uint8_t* their_ell = initiating ? ell_b64 : ell_a64;
    auto their_x = ellswift_decode(their_ell);

    auto x2 = their_x.square();
    auto x3 = x2 * their_x;
    auto y2 = x3 + FE_SEVEN;
    auto y = y2.sqrt();
    if (!(y.square() == y2)) return std::array<std::uint8_t, 32>{};

    auto y_bytes = y.to_bytes();
    if (y_bytes[31] & 1) y = y.negate();

    auto their_point = Point::from_affine(their_x, y);
    if (their_point.is_infinity()) return std::array<std::uint8_t, 32>{};

    // Non-CT variable-base scalar mul (~17.6 µs vs ~40 µs CT path).
    // Suitable for ephemeral BIP-324 session keys.
    auto ecdh_point = their_point.scalar_mul(our_privkey);
    if (ecdh_point.is_infinity()) return std::array<std::uint8_t, 32>{};
    auto ecdh_x = ecdh_point.x().to_bytes();

    // Precomputed midstate (static = computed once per process).
    static const auto kTagHash = SHA256::hash("bip324_ellswift_xonly_ecdh", 26);
    static const SHA256 kTagMid = [](){
        SHA256 h; h.update(kTagHash.data(), 32); h.update(kTagHash.data(), 32); return h;
    }();

    SHA256 hasher = kTagMid;
    hasher.update(ell_a64, 64);
    hasher.update(ell_b64, 64);
    hasher.update(ecdh_x.data(), 32);
    auto shared_secret = hasher.finalize();

    detail::secure_erase(ecdh_x.data(), 32);
    detail::secure_erase(&ecdh_point, sizeof(ecdh_point));

    return shared_secret;
}

} // namespace secp256k1
