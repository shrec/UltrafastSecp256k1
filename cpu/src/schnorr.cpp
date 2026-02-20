#include "secp256k1/schnorr.hpp"
#include "secp256k1/sha256.hpp"
#include "secp256k1/multiscalar.hpp"
#include "secp256k1/field_52.hpp"
#include <cstring>

namespace secp256k1 {

using fast::Scalar;
using fast::Point;
using fast::FieldElement;
using FE52 = fast::FieldElement52;

// ── FE52 sqrt() and inverse() available as FieldElement52 class methods ──────
// sqrt() uses FE52 ops (~4μs, faster than 4×64 ~6.8μs).
// inverse() uses FE52 Fermat (~4μs) — but SafeGCD (~2-3μs) is faster for
// variable-time paths (point.cpp batch inverse, verify Y-parity).

// ── Cached Tagged Hash Midstates (BIP-340) ───────────────────────────────────
// Pre-compute SHA256 midstate after processing (SHA256(tag) || SHA256(tag)).
// This is exactly 64 bytes = 1 SHA256 block, so after processing the midstate
// has buf_len_==0 and state_ captures all tag-dependent work.
// Saves 2 SHA256 block compressions per tagged_hash call.

static SHA256 make_tag_midstate(const char* tag) {
    auto tag_hash = SHA256::hash(tag, std::strlen(tag));
    SHA256 ctx;
    ctx.update(tag_hash.data(), 32);
    ctx.update(tag_hash.data(), 32);
    return ctx;
}

// BIP-340 tags used in Schnorr sign/verify
static const SHA256 g_aux_midstate       = make_tag_midstate("BIP0340/aux");
static const SHA256 g_nonce_midstate     = make_tag_midstate("BIP0340/nonce");
static const SHA256 g_challenge_midstate = make_tag_midstate("BIP0340/challenge");

// Fast tagged hash using cached midstate (avoids re-computing tag prefix)
static std::array<uint8_t, 32> cached_tagged_hash(const SHA256& midstate,
                                                    const void* data, std::size_t len) {
    SHA256 ctx = midstate;  // copy pre-computed state (trivial, ~108 bytes)
    ctx.update(data, len);
    return ctx.finalize();
}

// ── Tagged Hash (BIP-340) — generic fallback ─────────────────────────────────

std::array<uint8_t, 32> tagged_hash(const char* tag,
                                     const void* data, std::size_t len) {
    auto tag_hash = SHA256::hash(tag, std::strlen(tag));
    SHA256 ctx;
    ctx.update(tag_hash.data(), 32);
    ctx.update(tag_hash.data(), 32);
    ctx.update(data, len);
    return ctx.finalize();
}

// ── Schnorr Signature ────────────────────────────────────────────────────────

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

// ── X-only pubkey ────────────────────────────────────────────────────────────

std::array<uint8_t, 32> schnorr_pubkey(const Scalar& private_key) {
    auto P = Point::generator().scalar_mul(private_key);
    auto [px, p_y_odd] = P.x_bytes_and_parity();
    (void)p_y_odd;
    return px;
}

// ── SchnorrKeypair Creation ──────────────────────────────────────────────────

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

// ── BIP-340 Sign (keypair variant, fast) ─────────────────────────────────────
// Uses pre-computed keypair: only 1 gen_mul + 1 FE52 inverse per sign.

SchnorrSignature schnorr_sign(const SchnorrKeypair& kp,
                              const std::array<uint8_t, 32>& msg,
                              const std::array<uint8_t, 32>& aux_rand) {
    if (kp.d.is_zero()) return SchnorrSignature{};

    // Step 1: t = d XOR tagged_hash("BIP0340/aux", aux_rand)
    auto t_hash = cached_tagged_hash(g_aux_midstate, aux_rand.data(), 32);
    auto d_bytes = kp.d.to_bytes();
    uint8_t t[32];
    for (int i = 0; i < 32; ++i) t[i] = d_bytes[i] ^ t_hash[i];

    // Step 2: k' = tagged_hash("BIP0340/nonce", t || pubkey_x || msg)
    uint8_t nonce_input[96];
    std::memcpy(nonce_input, t, 32);
    std::memcpy(nonce_input + 32, kp.px.data(), 32);
    std::memcpy(nonce_input + 64, msg.data(), 32);
    auto rand_hash = cached_tagged_hash(g_nonce_midstate, nonce_input, 96);
    auto k_prime = Scalar::from_bytes(rand_hash);
    if (k_prime.is_zero()) return SchnorrSignature{};

    // Step 3: R = k' * G (single gen_mul — the only expensive point op)
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

// ── BIP-340 Sign (raw key, convenience) ──────────────────────────────────────

SchnorrSignature schnorr_sign(const Scalar& private_key,
                              const std::array<uint8_t, 32>& msg,
                              const std::array<uint8_t, 32>& aux_rand) {
    auto kp = schnorr_keypair_create(private_key);
    return schnorr_sign(kp, msg, aux_rand);
}

// ── BIP-340 Verify ───────────────────────────────────────────────────────────

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

    // Step 3: Lift x-only pubkey to point (all in FE52 — ~3x faster sqrt)
    FE52 px52 = FE52::from_fe(FieldElement::from_bytes(pubkey_x));

    // y² = x³ + 7
    FE52 x3 = px52.square() * px52;
    static const FE52 seven52 = FE52::from_fe(FieldElement::from_uint64(7));
    FE52 y2 = x3 + seven52;

    // sqrt via FE52 addition chain: a^((p+1)/4), ~253 sqr + 13 mul
    FE52 y52 = y2.sqrt();

    // Verify: y² == y2 (check that sqrt succeeded)
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

    // Convert to Point (FE52 → Point is zero-copy on FE52 path)
    auto P = Point::from_affine(px52.to_fe(), y52.to_fe());

    // Step 4: R = s*G - e*P  (4-stream GLV Strauss: s*G + (-e)*P in one pass)
    auto neg_e = e.negate();
    auto R = Point::dual_scalar_mul_gen_point(sig.s, neg_e, P);

    if (R.is_infinity()) return false;

    // Step 5: Fast Z²-based X check (no field inverse needed!)
    FE52 r52 = FE52::from_fe(FieldElement::from_bytes(sig.r));
    FE52 z2 = R.Z52().square();
    FE52 lhs = r52 * z2;       // sig.r * Z²
    lhs.normalize();

    FE52 rx = R.X52();
    rx.normalize();

    if (!(lhs == rx)) return false;

    // Step 6: Check R has even Y (SafeGCD inverse — ~2-3μs, faster than FE52 Fermat)
    FieldElement z_inv = R.z_raw().inverse();
    FieldElement z_inv2 = z_inv;
    z_inv2.square_inplace();
    FieldElement y_aff = R.y_raw() * z_inv2 * z_inv;
    auto y_bytes = y_aff.to_bytes();
    return (y_bytes[31] & 1) == 0;
}

// ── Pre-cached X-only Pubkey ─────────────────────────────────────────────────

bool schnorr_xonly_pubkey_parse(SchnorrXonlyPubkey& out,
                                const std::array<uint8_t, 32>& pubkey_x) {
    FE52 px52 = FE52::from_fe(FieldElement::from_bytes(pubkey_x));

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

    out.point = Point::from_affine(px52.to_fe(), y52.to_fe());
    out.x_bytes = pubkey_x;
    return true;
}

SchnorrXonlyPubkey schnorr_xonly_from_keypair(const SchnorrKeypair& kp) {
    SchnorrXonlyPubkey pub{};
    auto P = Point::generator().scalar_mul(kp.d);
    auto [px, p_y_odd] = P.x_bytes_and_parity();
    if (p_y_odd) {
        FE52 neg_y = P.Y52().negate(1);
        neg_y.normalize_weak();
        P = Point::from_jacobian52(P.X52(), neg_y, P.Z52(), false);
    }
    pub.point = P;
    pub.x_bytes = px;
    return pub;
}

// ── BIP-340 Verify (fast, pre-cached pubkey) ─────────────────────────────────
// Skips lift_x sqrt (~1.6μs savings). Same algorithm, just uses cached Point.

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

    // R = s*G - e*P  (direct Point — no sqrt needed)
    auto neg_e = e.negate();
    auto R = Point::dual_scalar_mul_gen_point(sig.s, neg_e, pubkey.point);

    if (R.is_infinity()) return false;

    // Z²-based X check
    FE52 r52 = FE52::from_fe(FieldElement::from_bytes(sig.r));
    FE52 z2 = R.Z52().square();
    FE52 lhs = r52 * z2;
    lhs.normalize();

    FE52 rx = R.X52();
    rx.normalize();

    if (!(lhs == rx)) return false;

    // Y parity check (SafeGCD inverse — faster than FE52 Fermat)
    FieldElement z_inv = R.z_raw().inverse();
    FieldElement z_inv2 = z_inv;
    z_inv2.square_inplace();
    FieldElement y_aff = R.y_raw() * z_inv2 * z_inv;
    auto y_bytes = y_aff.to_bytes();
    return (y_bytes[31] & 1) == 0;
}

} // namespace secp256k1
