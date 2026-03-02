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

// -- lift_x: shared BIP-340 x-only -> affine Point (no duplication) -----------
// Returns Point::infinity() on failure (x not on curve).
static Point lift_x(const uint8_t* x32) {
#if defined(SECP256K1_FAST_52BIT)
    // Direct bytes->FE52: avoids FieldElement construction overhead
    FE52 const px52 = FE52::from_bytes(x32);

    // y^2 = x^3 + 7
    FE52 const x3 = px52.square() * px52;
    static const FE52 seven52 = FE52::from_fe(FieldElement::from_uint64(7));
    FE52 const y2 = x3 + seven52;

    // sqrt via FE52 addition chain: a^((p+1)/4), ~253 sqr + 13 mul
    FE52 y52 = y2.sqrt();

    // Verify: y^2 == y2 (check that sqrt succeeded)
    FE52 check = y52.square();
    check.normalize();
    FE52 y2n = y2;
    y2n.normalize();
    if (!(check == y2n)) return Point::infinity();

    // Ensure even Y (BIP-340 convention): check parity of normalized y
    FE52 y_norm = y52;
    y_norm.normalize();
    if (y_norm.n[0] & 1) {
        // Negate: y = p - y
        y52 = y52.negate(1);
        y52.normalize_weak();
    }

    // Zero-conversion: construct Point directly from FE52 affine coordinates
    return Point::from_affine52(px52, y52);
#else
    // Fallback: 4x64 lift_x
    std::array<uint8_t, 32> px_arr;
    std::memcpy(px_arr.data(), x32, 32);
    auto px_fe = FieldElement::from_bytes(px_arr);
    auto x3 = px_fe * px_fe * px_fe;
    auto y2 = x3 + FieldElement::from_uint64(7);
    auto y_fe = y2.sqrt();
    auto chk = y_fe * y_fe;
    if (!(chk == y2)) return Point::infinity();
    // 4x64 mul_impl Barrett-reduces to [0, p), so limbs()[0] & 1 is
    // the true parity -- no serialization needed.
    if (y_fe.limbs()[0] & 1) y_fe = y_fe.negate();
    return Point::from_affine(px_fe, y_fe);
#endif
}

// -- Shared BIP-340 tagged-hash midstates (from tagged_hash.hpp) ---------------
using detail::g_aux_midstate;
using detail::g_nonce_midstate;
using detail::g_challenge_midstate;
using detail::cached_tagged_hash;

// -- Tagged Hash (BIP-340) -- generic fallback ---------------------------------

std::array<uint8_t, 32> tagged_hash(const char* tag,
                                     const void* data, std::size_t len) {
    std::string_view const sv(tag);
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

SchnorrSignature SchnorrSignature::from_bytes(const uint8_t* data64) {
    SchnorrSignature sig{};
    std::memcpy(sig.r.data(), data64, 32);
    sig.s = Scalar::from_bytes(data64 + 32);
    return sig;
}

SchnorrSignature SchnorrSignature::from_bytes(const std::array<uint8_t, 64>& data) {
    return from_bytes(data.data());
}

// -- BIP-340 strict signature parsing (r < p, 0 < s < n) ---------------------

bool SchnorrSignature::parse_strict(const uint8_t* data64, SchnorrSignature& out) noexcept {
    // BIP-340: fail if r >= p
    FieldElement r_fe;
    if (!FieldElement::parse_bytes_strict(data64, r_fe)) return false;

    // BIP-340: fail if s >= n; also reject s == 0
    Scalar s_val;
    if (!Scalar::parse_bytes_strict_nonzero(data64 + 32, s_val)) return false;

    std::memcpy(out.r.data(), data64, 32);
    out.s = s_val;
    return true;
}

bool SchnorrSignature::parse_strict(const std::array<uint8_t, 64>& data,
                                     SchnorrSignature& out) noexcept {
    return parse_strict(data.data(), out);
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

// -- BIP-340 shared verify helpers (no duplication) ---------------------------

// Compute e = tagged_hash("BIP0340/challenge", r || pubkey_x || msg) mod n
static Scalar compute_bip340_challenge(const uint8_t* r32,
                                       const uint8_t* pubkey_x32,
                                       const uint8_t* msg32) {
    SHA256 ctx = g_challenge_midstate;
    ctx.update(r32, 32);
    ctx.update(pubkey_x32, 32);
    ctx.update(msg32, 32);
    return Scalar::from_bytes(ctx.finalize());
}

// BIP-340 final checks on R: X matches sig.r, and Y is even.
// X-check: sig.r * Z^2 == R.X  (avoids Z^-2; early exit saves inverse on mismatch)
// Y-parity: affine y = Y * Z^-3 must be even (inverse unavoidable).
static bool verify_r_xcheck_yparity(const Point& R,
                                     const std::array<uint8_t, 32>& r_bytes) {
#if defined(SECP256K1_FAST_52BIT)
    FE52 const z2 = R.Z52().square();
    FE52 r52 = FE52::from_bytes(r_bytes);
    FE52 lhs = r52 * z2;
    lhs.normalize();
    FE52 rhs = R.X52();
    rhs.normalize();
    if (!(lhs == rhs)) return false;

    FE52 const z_inv = R.Z52().inverse_safegcd();
    FE52 const z_inv2 = z_inv.square();
    FE52 y_aff = (R.Y52() * z_inv2) * z_inv;
    y_aff.normalize();
    return (y_aff.n[0] & 1) == 0;
#else
    FieldElement z2 = R.z_raw();
    z2.square_inplace();
    auto r_fe = FieldElement::from_bytes(r_bytes);
    auto lhs_fe = r_fe * z2;
    if (!(lhs_fe == R.x_raw())) return false;

    FieldElement z_inv = R.z_raw().inverse();
    FieldElement z_inv2 = z_inv;
    z_inv2.square_inplace();
    FieldElement y_aff = R.y_raw() * z_inv2 * z_inv;
    return (y_aff.limbs()[0] & 1) == 0;
#endif
}

// -- BIP-340 Verify -----------------------------------------------------------

bool schnorr_verify(const uint8_t* pubkey_x32,
                    const uint8_t* msg32,
                    const SchnorrSignature& sig) {
    // Step 0: BIP-340 strict range checks
    // Check s: must be in [1, n-1] -- enforced at parse time by parse_strict,
    // but also guard here for callers using from_bytes (reducing parser).
    if (sig.s.is_zero()) return false;

    // Check r < p: if sig.r bytes represent a value >= p, reject.
    FieldElement r_fe_check;
    if (!FieldElement::parse_bytes_strict(sig.r.data(), r_fe_check)) return false;

    // Check pubkey x < p: if pubkey_x32 bytes represent a value >= p, reject.
    FieldElement pk_fe_check;
    if (!FieldElement::parse_bytes_strict(pubkey_x32, pk_fe_check)) return false;

    // Step 2: e = tagged_hash("BIP0340/challenge", r || pubkey_x || msg) mod n
    auto e = compute_bip340_challenge(sig.r.data(), pubkey_x32, msg32);

    // Step 3: Lift x-only pubkey to point
    auto P = lift_x(pubkey_x32);
    if (P.is_infinity()) return false;

    // Step 4: R = s*G - e*P  (4-stream GLV Strauss: s*G + (-e)*P in one pass)
    auto neg_e = e.negate();
    auto R = Point::dual_scalar_mul_gen_point(sig.s, neg_e, P);

    if (R.is_infinity()) return false;

    // Steps 5+6: X-check (inversion-free early exit) + Y-parity
    return verify_r_xcheck_yparity(R, sig.r);
}

// -- Pre-cached X-only Pubkey -------------------------------------------------

bool schnorr_xonly_pubkey_parse(SchnorrXonlyPubkey& out,
                                const uint8_t* pubkey_x32) {
    // BIP-340 strict: reject x >= p (no reduction)
    FieldElement x_check;
    if (!FieldElement::parse_bytes_strict(pubkey_x32, x_check)) return false;

    auto P = lift_x(pubkey_x32);
    if (P.is_infinity()) return false;
    out.point = P;
    std::memcpy(out.x_bytes.data(), pubkey_x32, 32);
    return true;
}

bool schnorr_xonly_pubkey_parse(SchnorrXonlyPubkey& out,
                                const std::array<uint8_t, 32>& pubkey_x) {
    return schnorr_xonly_pubkey_parse(out, pubkey_x.data());
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
                    const uint8_t* msg32,
                    const SchnorrSignature& sig) {
    // BIP-340 strict: s must be nonzero
    if (sig.s.is_zero()) return false;

    // BIP-340 strict: r < p
    FieldElement r_fe_check;
    if (!FieldElement::parse_bytes_strict(sig.r.data(), r_fe_check)) return false;

    // Challenge hash
    auto e = compute_bip340_challenge(sig.r.data(), pubkey.x_bytes.data(), msg32);

    // R = s*G - e*P  (direct Point -- no sqrt needed)
    auto neg_e = e.negate();
    auto R = Point::dual_scalar_mul_gen_point(sig.s, neg_e, pubkey.point);

    if (R.is_infinity()) return false;

    // X-check (inversion-free early exit) + Y-parity
    return verify_r_xcheck_yparity(R, sig.r);
}

// -- Array wrappers (delegate to raw-pointer implementations) -----------------

bool schnorr_verify(const std::array<uint8_t, 32>& pubkey_x,
                    const std::array<uint8_t, 32>& msg,
                    const SchnorrSignature& sig) {
    return schnorr_verify(pubkey_x.data(), msg.data(), sig);
}

bool schnorr_verify(const std::array<uint8_t, 32>& pubkey_x,
                    const uint8_t* msg32,
                    const SchnorrSignature& sig) {
    return schnorr_verify(pubkey_x.data(), msg32, sig);
}

bool schnorr_verify(const SchnorrXonlyPubkey& pubkey,
                    const std::array<uint8_t, 32>& msg,
                    const SchnorrSignature& sig) {
    return schnorr_verify(pubkey, msg.data(), sig);
}

} // namespace secp256k1
