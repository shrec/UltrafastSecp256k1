#include "secp256k1/schnorr.hpp"
#include "secp256k1/sha256.hpp"
#include "secp256k1/tagged_hash.hpp"
#include "secp256k1/multiscalar.hpp"
#include "secp256k1/config.hpp"    // SECP256K1_FAST_52BIT
#include "secp256k1/field_52.hpp"
#include <cstring>
#include <string_view>

namespace secp256k1 {

using fast::Scalar;
using fast::Point;
using fast::FieldElement;
#if defined(SECP256K1_FAST_52BIT)
using FE52 = fast::FieldElement52;
#endif

// -- FE52 sqrt() and inverse() available as FieldElement52 class methods ------
// sqrt() uses FE52 ops (~4us, faster than 4x64 ~6.8us).
// inverse() uses FE52 Fermat (~4us) -- but SafeGCD (~2-3us) is faster for
// variable-time paths (point.cpp batch inverse, verify Y-parity).

// -- Shared BIP-340 tagged-hash midstates (from tagged_hash.hpp) ---------------
using detail::make_tag_midstate;
using detail::g_aux_midstate;
using detail::g_nonce_midstate;
using detail::g_challenge_midstate;
using detail::cached_tagged_hash;

// -- Tagged Hash (BIP-340) -- generic fallback ---------------------------------

std::array<uint8_t, 32> tagged_hash(const char* tag,
                                     const void* data, std::size_t len) {
    std::string_view sv(tag);
    auto tag_hash = SHA256::hash(sv.data(), sv.size());
    SHA256 ctx;
    ctx.update(tag_hash.data(), 32);
    ctx.update(tag_hash.data(), 32);
    ctx.update(data, len);
    return ctx.finalize();
}

// -- Schnorr Signature --------------------------------------------------------

std::array<uint8_t, 64> SchnorrSignature::to_bytes() const {
    std::array<uint8_t, 64> out{};
    std::memcpy(out.data(), r.data(), 32);
    auto s_bytes = s.to_bytes();
    std::memcpy(out.data() + 32, s_bytes.data(), 32);
    return out;
}

SchnorrSignature SchnorrSignature::from_bytes(const std::array<uint8_t, 64>& data) {
    SchnorrSignature sig{};
    std::memcpy(sig.r.data(), data.data(), 32);
    std::array<uint8_t, 32> s_bytes{};
    std::memcpy(s_bytes.data(), data.data() + 32, 32);
    sig.s = Scalar::from_bytes(s_bytes);
    return sig;
}

// -- X-only pubkey ------------------------------------------------------------

std::array<uint8_t, 32> schnorr_pubkey(const Scalar& private_key) {
    auto P = Point::generator().scalar_mul(private_key);
    auto [px, p_y_odd] = P.x_bytes_and_parity();
    (void)p_y_odd;
    return px;
}

// -- SchnorrKeypair Creation --------------------------------------------------

SchnorrKeypair schnorr_keypair_create(const Scalar& private_key) {
    SchnorrKeypair kp{};
    auto d_prime = private_key;
    if (d_prime.is_zero()) return kp;

    auto P = Point::generator().scalar_mul(d_prime);
    auto [px, p_y_odd] = P.x_bytes_and_parity();

    kp.d = p_y_odd ? d_prime.negate() : d_prime;
    kp.px = px;
    return kp;
}

// -- BIP-340 Sign (keypair variant, fast) -------------------------------------
// Uses pre-computed keypair: only 1 gen_mul + 1 FE52 inverse per sign.

SchnorrSignature schnorr_sign(const SchnorrKeypair& kp,
                              const std::array<uint8_t, 32>& msg,
                              const std::array<uint8_t, 32>& aux_rand) {
    if (kp.d.is_zero()) return SchnorrSignature{};

    // Step 1: t = d XOR tagged_hash("BIP0340/aux", aux_rand)
    auto t_hash = cached_tagged_hash(g_aux_midstate, aux_rand.data(), 32);
    auto d_bytes = kp.d.to_bytes();
    uint8_t t[32];
    for (std::size_t i = 0; i < 32; ++i) t[i] = d_bytes[i] ^ t_hash[i];

    // Step 2: k' = tagged_hash("BIP0340/nonce", t || pubkey_x || msg)
    uint8_t nonce_input[96];
    std::memcpy(nonce_input, t, 32);
    std::memcpy(nonce_input + 32, kp.px.data(), 32);
    std::memcpy(nonce_input + 64, msg.data(), 32);
    auto rand_hash = cached_tagged_hash(g_nonce_midstate, nonce_input, 96);
    auto k_prime = Scalar::from_bytes(rand_hash);
    if (k_prime.is_zero()) return SchnorrSignature{};

    // Step 3: R = k' * G (single gen_mul -- the only expensive point op)
    auto R = Point::generator().scalar_mul(k_prime);
    auto [rx, r_y_odd] = R.x_bytes_and_parity();

    // Step 4: k = k' if has_even_y(R), else n - k'
    auto k = r_y_odd ? k_prime.negate() : k_prime;

    // Step 5: e = tagged_hash("BIP0340/challenge", R.x || pubkey_x || msg)
    uint8_t challenge_input[96];
    std::memcpy(challenge_input, rx.data(), 32);
    std::memcpy(challenge_input + 32, kp.px.data(), 32);
    std::memcpy(challenge_input + 64, msg.data(), 32);
    auto e_hash = cached_tagged_hash(g_challenge_midstate, challenge_input, 96);
    auto e = Scalar::from_bytes(e_hash);

    // Step 6: sig = (R.x, k + e * d)
    SchnorrSignature sig{};
    sig.r = rx;
    sig.s = k + e * kp.d;
    return sig;
}

// -- BIP-340 Sign (raw key, convenience) --------------------------------------

SchnorrSignature schnorr_sign(const Scalar& private_key,
                              const std::array<uint8_t, 32>& msg,
                              const std::array<uint8_t, 32>& aux_rand) {
    auto kp = schnorr_keypair_create(private_key);
    return schnorr_sign(kp, msg, aux_rand);
}

// -- BIP-340 Verify -----------------------------------------------------------

bool schnorr_verify(const std::array<uint8_t, 32>& pubkey_x,
                    const std::array<uint8_t, 32>& msg,
                    const SchnorrSignature& sig) {
    // Step 1: Check s < n (from_bytes already reduces)
    if (sig.s.is_zero()) return false;

    // Step 2: e = tagged_hash("BIP0340/challenge", r || pubkey_x || msg) mod n
    uint8_t challenge_input[96];
    std::memcpy(challenge_input, sig.r.data(), 32);
    std::memcpy(challenge_input + 32, pubkey_x.data(), 32);
    std::memcpy(challenge_input + 64, msg.data(), 32);
    auto e_hash = cached_tagged_hash(g_challenge_midstate, challenge_input, 96);
    auto e = Scalar::from_bytes(e_hash);

    // Step 3: Lift x-only pubkey to point (all in FE52 -- ~3x faster sqrt)
#if defined(SECP256K1_FAST_52BIT)
    // Direct bytes->FE52: avoids FieldElement construction overhead
    FE52 px52 = FE52::from_bytes(pubkey_x);

    // y^2 = x^3 + 7
    FE52 x3 = px52.square() * px52;
    static const FE52 seven52 = FE52::from_fe(FieldElement::from_uint64(7));
    FE52 y2 = x3 + seven52;

    // sqrt via FE52 addition chain: a^((p+1)/4), ~253 sqr + 13 mul
    FE52 y52 = y2.sqrt();

    // Verify: y^2 == y2 (check that sqrt succeeded)
    FE52 check = y52.square();
    check.normalize();
    FE52 y2n = y2;
    y2n.normalize();
    if (!(check == y2n)) return false;

    // Ensure even Y (BIP-340 convention): check parity of normalized y
    FE52 y_norm = y52;
    y_norm.normalize();
    if (y_norm.n[0] & 1) {
        // Negate: y = p - y
        y52 = y52.negate(1);
        y52.normalize_weak();
    }

    // Zero-conversion: construct Point directly from FE52 affine coordinates
    auto P = Point::from_affine52(px52, y52);
#else
    // Fallback: 4x64 lift_x
    auto px_fe = FieldElement::from_bytes(pubkey_x);
    auto x3 = px_fe * px_fe * px_fe;
    auto y2 = x3 + FieldElement::from_uint64(7);
    auto y_fe = y2.sqrt();
    auto check = y_fe * y_fe;
    if (!(check == y2)) return false;
    auto y_bytes = y_fe.to_bytes();
    if (y_bytes[31] & 1) y_fe = y_fe.negate();
    auto P = Point::from_affine(px_fe, y_fe);
#endif

    // Step 4: R = s*G - e*P  (4-stream GLV Strauss: s*G + (-e)*P in one pass)
    auto neg_e = e.negate();
    auto R = Point::dual_scalar_mul_gen_point(sig.s, neg_e, P);

    if (R.is_infinity()) return false;

    // Steps 5+6: Combined X-check + Y-parity via single Z inverse (all FE52)
    // One SafeGCD inverse (~3us) shared between both checks.
#if defined(SECP256K1_FAST_52BIT)
    FE52 z_inv52 = R.Z52().inverse_safegcd();
    FE52 z_inv2 = z_inv52.square();         // Z^-^2

    // X-check: X * Z^-^2 == sig.r  (affine x)
    FE52 x_aff = R.X52() * z_inv2;
    x_aff.normalize();
    FE52 r52 = FE52::from_bytes(sig.r);
    r52.normalize();
    if (!(x_aff == r52)) return false;

    // Y-parity: Y * Z^-^3 must be even
    FE52 y_aff = (R.Y52() * z_inv2) * z_inv52;
    y_aff.normalize();
    return (y_aff.n[0] & 1) == 0;
#else
    auto rx_fe = R.x();
    auto r_fe = FieldElement::from_bytes(sig.r);
    if (!(r_fe == rx_fe)) return false;
    FieldElement z_inv = R.z_raw().inverse();
    FieldElement z_inv2 = z_inv;
    z_inv2.square_inplace();
    FieldElement y_aff = R.y_raw() * z_inv2 * z_inv;
    return (y_aff.limbs()[0] & 1) == 0;
#endif
}

// -- Pre-cached X-only Pubkey -------------------------------------------------

bool schnorr_xonly_pubkey_parse(SchnorrXonlyPubkey& out,
                                const std::array<uint8_t, 32>& pubkey_x) {
#if defined(SECP256K1_FAST_52BIT)
    // Direct bytes->FE52: avoids FieldElement construction overhead
    FE52 px52 = FE52::from_bytes(pubkey_x);

    FE52 x3 = px52.square() * px52;
    static const FE52 seven52 = FE52::from_fe(FieldElement::from_uint64(7));
    FE52 y2 = x3 + seven52;

    FE52 y52 = y2.sqrt();

    FE52 check = y52.square();
    check.normalize();
    FE52 y2n = y2;
    y2n.normalize();
    if (!(check == y2n)) return false;

    FE52 y_norm = y52;
    y_norm.normalize();
    if (y_norm.n[0] & 1) {
        y52 = y52.negate(1);
        y52.normalize_weak();
    }

    // Zero-conversion: construct Point directly from FE52 affine coordinates
    out.point = Point::from_affine52(px52, y52);
#else
    // Fallback: 4x64 lift_x
    auto px_fe = FieldElement::from_bytes(pubkey_x);
    auto x3 = px_fe * px_fe * px_fe;
    auto y2 = x3 + FieldElement::from_uint64(7);
    auto y_fe = y2.sqrt();
    auto check = y_fe * y_fe;
    if (!(check == y2)) return false;
    auto y_bytes_chk = y_fe.to_bytes();
    if (y_bytes_chk[31] & 1) y_fe = y_fe.negate();
    out.point = Point::from_affine(px_fe, y_fe);
#endif
    out.x_bytes = pubkey_x;
    return true;
}

SchnorrXonlyPubkey schnorr_xonly_from_keypair(const SchnorrKeypair& kp) {
    SchnorrXonlyPubkey pub{};
    auto P = Point::generator().scalar_mul(kp.d);
    auto [px, p_y_odd] = P.x_bytes_and_parity();
    if (p_y_odd) {
#if defined(SECP256K1_FAST_52BIT)
        FE52 neg_y = P.Y52().negate(1);
        neg_y.normalize_weak();
        P = Point::from_jacobian52(P.X52(), neg_y, P.Z52(), false);
#else
        auto y_neg = P.y().negate();
        P = Point::from_jacobian_coords(P.x(), y_neg, P.z(), false);
#endif
    }
    pub.point = P;
    pub.x_bytes = px;
    return pub;
}

// -- BIP-340 Verify (fast, pre-cached pubkey) ---------------------------------
// Skips lift_x sqrt (~1.6us savings). Same algorithm, just uses cached Point.

bool schnorr_verify(const SchnorrXonlyPubkey& pubkey,
                    const std::array<uint8_t, 32>& msg,
                    const SchnorrSignature& sig) {
    if (sig.s.is_zero()) return false;

    // Challenge hash uses cached x_bytes
    uint8_t challenge_input[96];
    std::memcpy(challenge_input, sig.r.data(), 32);
    std::memcpy(challenge_input + 32, pubkey.x_bytes.data(), 32);
    std::memcpy(challenge_input + 64, msg.data(), 32);
    auto e_hash = cached_tagged_hash(g_challenge_midstate, challenge_input, 96);
    auto e = Scalar::from_bytes(e_hash);

    // R = s*G - e*P  (direct Point -- no sqrt needed)
    auto neg_e = e.negate();
    auto R = Point::dual_scalar_mul_gen_point(sig.s, neg_e, pubkey.point);

    if (R.is_infinity()) return false;

    // Combined X-check + Y-parity via single Z inverse (all FE52)
#if defined(SECP256K1_FAST_52BIT)
    FE52 z_inv52 = R.Z52().inverse_safegcd();
    FE52 z_inv2 = z_inv52.square();         // Z^-^2

    // X-check: X * Z^-^2 == sig.r
    FE52 x_aff = R.X52() * z_inv2;
    x_aff.normalize();
    FE52 r52 = FE52::from_bytes(sig.r);
    r52.normalize();
    if (!(x_aff == r52)) return false;

    // Y-parity: Y * Z^-^3 must be even
    FE52 y_aff = (R.Y52() * z_inv2) * z_inv52;
    y_aff.normalize();
    return (y_aff.n[0] & 1) == 0;
#else
    auto rx_fe = R.x();
    auto r_fe = FieldElement::from_bytes(sig.r);
    if (!(r_fe == rx_fe)) return false;
    FieldElement z_inv = R.z_raw().inverse();
    FieldElement z_inv2 = z_inv;
    z_inv2.square_inplace();
    FieldElement y_aff = R.y_raw() * z_inv2 * z_inv;
    return (y_aff.limbs()[0] & 1) == 0;
#endif
}

} // namespace secp256k1
