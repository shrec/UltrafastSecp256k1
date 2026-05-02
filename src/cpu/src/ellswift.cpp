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
std::pair<bool, FieldElement> xswiftec_inv(
    const FieldElement& x, const FieldElement& u, int c) noexcept {
    // c3 = c2+1 = (-sqrt(-3)+1)/2
    static const FieldElement C3 = []() {
        std::array<uint8_t, 32> b = {
            0x7a,0xe9,0x6a,0x2b, 0x65,0x7c,0x07,0x10,
            0x6e,0x64,0x47,0x9e, 0xac,0x34,0x34,0xe9,
            0x9c,0xf0,0x49,0x75, 0x12,0xf5,0x89,0x95,
            0xc1,0x39,0x6c,0x28, 0x71,0x95,0x01,0xef
        };
        return FieldElement::from_bytes(b);
    }();
    // c4 = c1+1 = (sqrt(-3)+1)/2
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

    FieldElement v, s;

    if (!(c & 2)) {
        // x1/x2 case (c ∈ {0,1,4,5})
        // Fail if (-u-x) is on curve — that x would be x3 candidate, taking priority
        auto neg_ux = (u + x).negate();
        if (fe_is_square(neg_ux * neg_ux * neg_ux + FE_SEVEN)) return {false, FE_ZERO};

        // s = -(u^3+7)/(u^2+u*x+x^2)
        // sp = -(u+x)^2 + u*x = -(u^2+ux+x^2)
        auto u2  = u.square();
        auto g   = u2 * u + FE_SEVEN;
        auto sp  = (u + x).square().negate() + u * x;  // -(u^2+ux+x^2)
        // sp*g = -(u^2+ux+x^2)*(u^3+7) must be square
        if (!fe_is_square(sp * g)) return {false, FE_ZERO};
        s = g * sp.inverse();   // -(u^3+7)/(u^2+ux+x^2)
        v = x;

    } else {
        // x3 case (c ∈ {2,3,6,7})
        s = x - u;
        if (!fe_is_square(s) || s == FE_ZERO) return {false, FE_ZERO};

        // r = sqrt(-s*(4*(u^3+7)+3*u^2*s))
        auto u2 = u.square();
        auto g4 = (u2 * u + FE_SEVEN) * FE_FOUR;
        auto r2 = (FE_THREE * u2 * s + g4).negate() * s; // -s*(4*(u^3+7)+3*u^2*s)
        if (!fe_is_square(r2)) return {false, FE_ZERO};
        auto r = r2.sqrt();
        if ((c & 1) && r == FE_ZERO) return {false, FE_ZERO};

        v = (r * s.inverse() - u) * TWO_INV;
    }

    // w = sqrt(s); negate for c&5 ∈ {0,5}
    auto w = s.sqrt();
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
    auto pub = ct::generator_mul(privkey);
    auto x = pub.x();
    bool y_is_odd = (pub.y().to_bytes()[31] & 1) != 0;

    std::array<std::uint8_t, 64> result{};
    static constexpr int kMaxAttempts = 100;
    for (int attempt = 0; attempt < kMaxAttempts; ++attempt) {
        std::uint8_t rand_bytes[32];
        csprng_fill(rand_bytes, 32);

        auto u = fe_from_bytes_mod_p(rand_bytes);
        if (u == FieldElement::zero()) continue;

        for (int c = 0; c < 8; ++c) {
            auto [ok, t] = xswiftec_inv(x, u, c);
            if (!ok) continue;

            auto t_bytes = t.to_bytes();
            // XSwiftEC: decode gives y_is_odd == t_is_odd — match the pubkey's y.
            if (((t_bytes[31] & 1) != 0) != y_is_odd) continue;

            auto u_bytes = u.to_bytes();
            std::memcpy(result.data(),      u_bytes.data(), 32);
            std::memcpy(result.data() + 32, t_bytes.data(), 32);

            detail::secure_erase(rand_bytes, sizeof(rand_bytes));
            return result;
        }
    }
    throw std::runtime_error("ellswift_create: RNG produced 100 consecutive unusable values");
}

std::array<std::uint8_t, 64> ellswift_create(const Scalar& privkey,
                                              const std::uint8_t* auxrnd32) {
    if (!auxrnd32) return ellswift_create(privkey);  // fall back to CSPRNG path

    // Deterministic path: derive u candidates from
    // H("secp256k1_ellswift_create" || privkey || 0x00*32 || auxrnd32 || cnt)
    // matching libsecp256k1's secp256k1_ellswift_create auxrnd32 semantics.
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

        for (int c = 0; c < 8; ++c) {
            auto [ok, t] = xswiftec_inv(x, u, c);
            if (!ok) continue;

            auto t_bytes = t.to_bytes();
            if (((t_bytes[31] & 1) != 0) != y_is_odd) continue;

            auto u_bytes = u.to_bytes();
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
    // Non-CT: use precomputed w=18 fixed-base table (~6.7 µs vs ~27 µs CT path).
    // Suitable for ephemeral BIP-324 session keys where CT is not required.
    auto pub = scalar_mul_generator(privkey);
    auto x   = pub.x();
    bool y_odd = (pub.y().to_bytes()[31] & 1) != 0;

    std::array<std::uint8_t, 64> result{};
    static constexpr int kMaxAttempts = 200;
    for (int attempt = 0; attempt < kMaxAttempts; ++attempt) {
        std::uint8_t rand_bytes[32];
        csprng_fill(rand_bytes, 32);

        auto u = fe_from_bytes_mod_p(rand_bytes);
        if (u == FieldElement::zero()) continue;  // xswiftec_inv requires u != 0

        for (int c = 0; c < 8; ++c) {
            auto [ok, t] = xswiftec_inv(x, u, c);
            if (!ok) continue;

            // Negating t doesn't change xswiftec_fwd (only t^2 appears),
            // so we can freely adjust t parity to match pubkey y parity.
            bool t_odd = (t.to_bytes()[31] & 1) != 0;
            if (t_odd != y_odd) t = t.negate();

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

    constexpr char tag_str[] = "bip324_ellswift_xonly_ecdh";
    auto tag_hash = SHA256::hash(tag_str, sizeof(tag_str) - 1);

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

} // namespace secp256k1
