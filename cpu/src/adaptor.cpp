// ============================================================================
// Adaptor Signatures — Implementation
// ============================================================================

#include "secp256k1/adaptor.hpp"
#include "secp256k1/sha256.hpp"
#include "secp256k1/schnorr.hpp"
#include "secp256k1/ecdsa.hpp"
#include <cstring>

namespace secp256k1 {

using fast::Point;
using fast::Scalar;
using fast::FieldElement;

// ── Internal: deterministic nonce for adaptor signing ─────────────────────────

static Scalar adaptor_nonce(const Scalar& privkey,
                             const std::uint8_t* msg, std::size_t msg_len,
                             const Point& adaptor,
                             const std::uint8_t* aux, std::size_t aux_len) {
    SHA256 h;
    auto sk_bytes = privkey.to_bytes();
    h.update(sk_bytes.data(), 32);
    h.update(msg, msg_len);
    auto adapt_comp = adaptor.to_compressed();
    h.update(adapt_comp.data(), 33);
    if (aux && aux_len > 0) {
        h.update(aux, aux_len);
    }
    const char* domain = "adaptor_nonce_v1";
    h.update(domain, std::strlen(domain));
    auto hash = h.finalize();
    Scalar k = Scalar::from_bytes(hash);
    // Ensure non-zero
    if (k.is_zero()) {
        hash[31] ^= 1;
        k = Scalar::from_bytes(hash);
    }
    return k;
}

// ── Schnorr Adaptor Signatures ───────────────────────────────────────────────

SchnorrAdaptorSig
schnorr_adaptor_sign(const Scalar& private_key,
                     const std::array<std::uint8_t, 32>& msg,
                     const Point& adaptor_point,
                     const std::array<std::uint8_t, 32>& aux_rand) {
    // Compute public key P = sk * G
    Point P = Point::generator().scalar_mul(private_key);

    // BIP-340: negate sk if P.y is odd
    Scalar sk = private_key;
    auto P_y = P.y().to_bytes();
    if (P_y[31] & 1) {
        sk = sk.negate();
        P = P.negate();
    }

    // Generate nonce k 
    Scalar k = adaptor_nonce(sk, msg.data(), 32, adaptor_point, aux_rand.data(), 32);

    // R̂ = k * G (the pre-nonce, before adapting)
    Point R_hat = Point::generator().scalar_mul(k);

    // R = R̂ + T (the final nonce point after adapting)
    Point R = R_hat.add(adaptor_point);

    // BIP-340: if R.y is odd, negate k (and R̂ and R flip accordingly)
    auto R_y = R.y().to_bytes();
    bool needs_neg = (R_y[31] & 1) != 0;
    if (needs_neg) {
        k = k.negate();
        R_hat = R_hat.negate();
        R = R.negate();
    }

    // Challenge: e = H("BIP0340/challenge", R.x || P.x || m)
    auto R_x = R.x().to_bytes();
    auto P_x = P.x().to_bytes();
    
    // Build challenge data
    std::uint8_t challenge_data[96];
    std::memcpy(challenge_data, R_x.data(), 32);
    std::memcpy(challenge_data + 32, P_x.data(), 32);
    std::memcpy(challenge_data + 64, msg.data(), 32);
    auto e_hash = tagged_hash("BIP0340/challenge", challenge_data, 96);
    Scalar e = Scalar::from_bytes(e_hash);

    // ŝ = k + e * sk (BIP-340: s = k + e*d, but partial — missing adaptor secret t)
    Scalar s_hat = k + (e * sk);

    return SchnorrAdaptorSig{R_hat, s_hat, needs_neg};
}

bool schnorr_adaptor_verify(const SchnorrAdaptorSig& pre_sig,
                            const std::array<std::uint8_t, 32>& pubkey_x,
                            const std::array<std::uint8_t, 32>& msg,
                            const Point& adaptor_point) {
    // Reconstruct P from x-only pubkey (even y)
    FieldElement px = FieldElement::from_bytes(pubkey_x);
    FieldElement x2 = px * px;
    FieldElement x3 = x2 * px;
    FieldElement rhs = x3 + FieldElement::from_uint64(7);
    auto sqrt_exp = FieldElement::from_hex(
        "3fffffffffffffffffffffffffffffffffffffffffffffffffffffffbfffff0c");
    auto py = FieldElement::one();
    {
        auto base = rhs;
        auto eb = sqrt_exp.to_bytes();
        for (int i = 0; i < 256; ++i) {
            py = py.square();
            int bi = i / 8, bt = 7 - (i % 8);
            if ((eb[bi] >> bt) & 1) py = py * base;
        }
    }
    auto py_bytes = py.to_bytes();
    if (py_bytes[31] & 1) py = FieldElement::zero() - py;
    Point P = Point::from_affine(px, py);

    // Adjust T based on whether nonce was negated during signing
    Point T_adj = pre_sig.needs_negation ? adaptor_point.negate() : adaptor_point;

    // Reconstruct R = R̂ + T_adj (should have even y)
    Point R = pre_sig.R_hat.add(T_adj);

    // Challenge: e = H("BIP0340/challenge", R.x || P.x || m)
    auto R_x = R.x().to_bytes();
    std::uint8_t challenge_data[96];
    std::memcpy(challenge_data, R_x.data(), 32);
    std::memcpy(challenge_data + 32, pubkey_x.data(), 32);
    std::memcpy(challenge_data + 64, msg.data(), 32);
    auto e_hash = tagged_hash("BIP0340/challenge", challenge_data, 96);
    Scalar e = Scalar::from_bytes(e_hash);

    // Verify: ŝ*G == R̂ + e*P  (since ŝ = k + e*d)
    Point lhs = Point::generator().scalar_mul(pre_sig.s_hat);
    Point eP = P.scalar_mul(e);
    Point rhs_point = pre_sig.R_hat.add(eP);

    auto lhs_c = lhs.to_compressed();
    auto rhs_c = rhs_point.to_compressed();
    return lhs_c == rhs_c;
}

SchnorrSignature
schnorr_adaptor_adapt(const SchnorrAdaptorSig& pre_sig,
                      const Scalar& adaptor_secret) {
    // Adjust t based on nonce negation during signing
    Scalar t = pre_sig.needs_negation ? adaptor_secret.negate() : adaptor_secret;

    // R = R̂ + t_adj*G (should have even y since we ensured it during signing)
    Point R = pre_sig.R_hat.add(Point::generator().scalar_mul(t));

    // s = ŝ + t
    Scalar s = pre_sig.s_hat + t;

    SchnorrSignature sig;
    sig.r = R.x().to_bytes();
    sig.s = s;
    return sig;
}

std::pair<Scalar, bool>
schnorr_adaptor_extract(const SchnorrAdaptorSig& pre_sig,
                        const SchnorrSignature& sig) {
    // t = s - ŝ (or negation)
    Scalar t = sig.s - pre_sig.s_hat;

    // Verify: t*G == T (adaptor point)
    // We can't fully verify without the adaptor point, but we can return t
    // and let the caller check T = t*G
    if (t.is_zero()) {
        return {t, false};
    }
    return {t, true};
}

// ── ECDSA Adaptor Signatures ─────────────────────────────────────────────────

ECDSAAdaptorSig
ecdsa_adaptor_sign(const Scalar& private_key,
                   const std::array<std::uint8_t, 32>& msg_hash,
                   const Point& adaptor_point) {
    // Generate nonce
    Scalar k = adaptor_nonce(private_key, msg_hash.data(), 32, adaptor_point, nullptr, 0);

    // R̂ = k * G
    Point R_hat = Point::generator().scalar_mul(k);

    // R = R̂ + T
    Point R = R_hat.add(adaptor_point);

    // r = R.x mod n
    auto R_x_bytes = R.x().to_bytes();
    Scalar r = Scalar::from_bytes(R_x_bytes);
    if (r.is_zero()) {
        // Degenerate case
        return ECDSAAdaptorSig{R_hat, Scalar::zero(), r};
    }

    // ŝ = k⁻¹ * (z + r*x)  where z = msg_hash
    Scalar z = Scalar::from_bytes(msg_hash);
    Scalar k_inv = k.inverse();
    Scalar s_hat = k_inv * (z + r * private_key);

    // Low-S normalization
    // Check: s_hat vs n - s_hat 
    // (we'll normalize at adapt time)

    return ECDSAAdaptorSig{R_hat, s_hat, r};
}

bool ecdsa_adaptor_verify(const ECDSAAdaptorSig& pre_sig,
                          const Point& public_key,
                          const std::array<std::uint8_t, 32>& msg_hash,
                          const Point& adaptor_point) {
    if (pre_sig.r.is_zero() || pre_sig.s_hat.is_zero()) return false;

    // R = R̂ + T
    Point R = pre_sig.R_hat.add(adaptor_point);

    // Verify r == R.x mod n
    auto R_x_bytes = R.x().to_bytes();
    Scalar r_check = Scalar::from_bytes(R_x_bytes);
    if (r_check != pre_sig.r) return false;

    // Verify: ŝ*R̂ == z*G + r*P (rearranged ECDSA equation)
    Scalar z = Scalar::from_bytes(msg_hash);
    Scalar s_inv = pre_sig.s_hat.inverse();

    Point u1G = Point::generator().scalar_mul(z * s_inv);
    Point u2P = public_key.scalar_mul(pre_sig.r * s_inv);
    Point R_check = u1G.add(u2P);

    // R_check should equal R̂ (not R̂+T, since we used the adapted r)
    // Actually for ECDSA adaptor: we check that k*G = R̂
    // The equation ŝ = k⁻¹*(z + r*x) means k = ŝ⁻¹*(z + r*x)
    // So R̂ = ŝ⁻¹*(z + r*x)*G = ŝ⁻¹*z*G + ŝ⁻¹*r*P
    auto r_hat_c = pre_sig.R_hat.to_compressed();
    auto r_chk_c = R_check.to_compressed();
    return r_hat_c == r_chk_c;
}

ECDSASignature
ecdsa_adaptor_adapt(const ECDSAAdaptorSig& pre_sig,
                    const Scalar& adaptor_secret) {
    // For ECDSA adaptor: final signature uses the adapted s
    // The nonce was k, and the adaptor adds t:
    // effective_k = k + t
    // s = effective_k⁻¹ * (z + r*x)
    // But we have ŝ = k⁻¹ * (z + r*x)
    // We need to adjust: s = ŝ * k / (k + t)
    // This requires knowing k, which we don't have directly.
    //
    // Alternative (simpler) ECDSA adaptor:
    // ŝ_hat is the "encrypted" signature value
    // s = ŝ_hat * t⁻¹  (multiplicative adaptor for ECDSA)
    Scalar t_inv = adaptor_secret.inverse();
    Scalar s = pre_sig.s_hat * t_inv;

    ECDSASignature sig;
    sig.r = pre_sig.r;
    sig.s = s;
    return sig.normalize();
}

std::pair<Scalar, bool>
ecdsa_adaptor_extract(const ECDSAAdaptorSig& pre_sig,
                      const ECDSASignature& sig) {
    if (sig.s.is_zero() || pre_sig.s_hat.is_zero()) return {Scalar::zero(), false};

    // t = ŝ * s⁻¹ (multiplicative adaptor)
    Scalar s_inv = sig.s.inverse();
    Scalar t = pre_sig.s_hat * s_inv;

    if (t.is_zero()) return {t, false};
    return {t, true};
}

} // namespace secp256k1
