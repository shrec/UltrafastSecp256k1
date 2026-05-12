// ============================================================================
// Adaptor Signatures -- Implementation
// ============================================================================

#include "secp256k1/adaptor.hpp"
#include "secp256k1/sha256.hpp"
#include "secp256k1/schnorr.hpp"
#include "secp256k1/ecdsa.hpp"
#include "secp256k1/ct/point.hpp"
#include "secp256k1/ct/scalar.hpp"
#include "secp256k1/detail/secure_erase.hpp"
#include <cstring>

namespace secp256k1 {

using fast::Point;
using fast::Scalar;
using fast::FieldElement;

// -- Internal: deterministic nonce for adaptor signing -------------------------

static Scalar adaptor_nonce(const Scalar& privkey,
                             const std::uint8_t* msg, std::size_t msg_len,
                             const Point& adaptor,
                             const std::uint8_t* aux, std::size_t aux_len) {
    // BIP-340 tagged hash: SHA256(SHA256(tag) || SHA256(tag) || data)
    // The double-prepended tag hash provides cross-protocol domain separation.
    // Previously the tag was appended LAST, which is weaker — a collision-prone
    // scheme where the hash of (data || tag) offers no isolation from untagged
    // hash functions sharing the same prefix.
    constexpr char domain[] = "adaptor_nonce_v1";
    auto tag_hash = SHA256::hash(domain, sizeof(domain) - 1);

    SHA256 h;
    h.update(tag_hash.data(), 32);  // SHA256(tag) prepended twice — BIP-340 pattern
    h.update(tag_hash.data(), 32);
    auto sk_bytes = privkey.to_bytes();
    h.update(sk_bytes.data(), 32);
    h.update(msg, msg_len);
    auto adapt_comp = adaptor.to_compressed();
    h.update(adapt_comp.data(), 33);
    if (aux && aux_len > 0) {
        h.update(aux, aux_len);
    }
    auto hash = h.finalize();
    // parse_bytes_strict_nonzero avoids the conditional mod-n branch that
    // from_bytes uses when hash >= n (prob ~2^-128) and the is_zero() branch.
    // The counter-based retry below essentially never executes in practice
    // but prevents any data-dependent branch on the secret-derived hash.
    Scalar k;
    for (std::uint8_t ctr = 0; !Scalar::parse_bytes_strict_nonzero(hash.data(), k); ++ctr) {
        hash[31] ^= static_cast<std::uint8_t>(ctr ^ 1u);
    }
    detail::secure_erase(sk_bytes.data(), sk_bytes.size());
    detail::secure_erase(hash.data(), hash.size());
    return k;
}

static Scalar ecdsa_adaptor_binding(const Point& adaptor_point) {
    SHA256 h;
    auto adaptor_bytes = adaptor_point.to_compressed();
    constexpr char domain[] = "ecdsa_adaptor_bind_v1";
    h.update(reinterpret_cast<const std::uint8_t*>(domain), sizeof(domain) - 1);
    h.update(adaptor_bytes.data(), adaptor_bytes.size());
    auto hash = h.finalize();
    Scalar binding = Scalar::from_bytes(hash);
    if (binding.is_zero()) {
        hash[31] ^= 1;
        binding = Scalar::from_bytes(hash);
    }
    detail::secure_erase(hash.data(), hash.size());
    return binding;
}

// -- Schnorr Adaptor Signatures -----------------------------------------------

SchnorrAdaptorSig
schnorr_adaptor_sign(const Scalar& private_key,
                     const std::array<std::uint8_t, 32>& msg,
                     const Point& adaptor_point,
                     const std::array<std::uint8_t, 32>& aux_rand) {
    // Compute public key P = sk * G (CT)
    Point P = ct::generator_mul(private_key);

    // BIP-340: negate sk if P.y is odd (CT branchless)
    auto [P_x_bytes, p_y_odd] = P.x_bytes_and_parity();
    std::uint64_t const sk_neg_mask = static_cast<std::uint64_t>(p_y_odd)
                                    * UINT64_C(0xFFFFFFFFFFFFFFFF);
    Scalar sk = ct::scalar_cneg(private_key, sk_neg_mask);
    // p_y_odd is the parity of the PUBLIC KEY y-coordinate (P = sk*G is public).
    // Timing leakage of p_y_odd reveals no additional secret beyond the public key.
    // LOW risk: leave as branchless-arithmetic if; the negate() call is arithmetic.
    if (p_y_odd) P = P.negate();

    // Generate nonce k
    Scalar k = adaptor_nonce(sk, msg.data(), 32, adaptor_point, aux_rand.data(), 32);

    // R^ = k * G (the pre-nonce, before adapting) — CT: k is secret
    Point R_hat = ct::generator_mul(k);

    // R = R^ + T (the final nonce point after adapting)
    Point R = R_hat.add(adaptor_point);

    // BIP-340: negate k/R^/R if R.y is odd  [CT: all three negations are branchless]
    // R.y parity is derived from the secret nonce k — a branch here leaks k.
    auto R_y = R.y().to_bytes();
    bool const needs_neg = (R_y[31] & 1u) != 0u;
    std::uint64_t const neg_mask = ct::bool_to_mask(needs_neg);
    k = ct::scalar_cneg(k, neg_mask);  // CT: k is the secret nonce
    // CT branchless conditional negate of R_hat and R via CTJacobianPoint cmov
    {
        auto ct_R_hat = ct::CTJacobianPoint::from_point(R_hat);
        auto ct_R     = ct::CTJacobianPoint::from_point(R);
        ct::point_cmov(&ct_R_hat, ct::point_neg(ct_R_hat), neg_mask);
        ct::point_cmov(&ct_R,     ct::point_neg(ct_R),     neg_mask);
        R_hat = ct_R_hat.to_point();
        R     = ct_R.to_point();
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
    Scalar const e = Scalar::from_bytes(e_hash);

    // s = k + e * sk — CT: both k (secret nonce) and sk (secret key) are secret
    Scalar const s_hat = ct::scalar_add(k, ct::scalar_mul(e, sk));

    detail::secure_erase(&k, sizeof(k));  // SEC-005: const_cast was unnecessary — k is non-const
    detail::secure_erase(&sk, sizeof(sk));
    return SchnorrAdaptorSig{R_hat, s_hat, needs_neg};
}

bool schnorr_adaptor_verify(const SchnorrAdaptorSig& pre_sig,
                            const std::array<std::uint8_t, 32>& pubkey_x,
                            const std::array<std::uint8_t, 32>& msg,
                            const Point& adaptor_point) {
    // Reconstruct P from x-only pubkey (even y)
    // Strict: reject x >= p (non-canonical encoding)
    FieldElement px;
    if (!FieldElement::parse_bytes_strict(pubkey_x.data(), px)) return false;
    FieldElement const x2 = px * px;
    FieldElement const x3 = x2 * px;
    FieldElement const rhs = x3 + FieldElement::from_uint64(7);
    auto py = rhs.sqrt();
    // Reject if px is not on the curve (py^2 must equal rhs)
    if (py * py != rhs) return false;
    if (py.limbs()[0] & 1u) py = FieldElement::zero() - py;
    Point const P = Point::from_affine(px, py);

    // Integrity check: needs_negation must be consistent with the adaptor orientation.
    // The correct value is: needs_negation == ((R_hat + T).y is odd).
    // A tampered needs_negation bit would bind the pre-sig to the wrong adaptor secret.
    {
        auto [Rcheck_x, Rcheck_y_odd] = pre_sig.R_hat.add(adaptor_point).x_bytes_and_parity();
        (void)Rcheck_x;
        if (static_cast<bool>(Rcheck_y_odd) != pre_sig.needs_negation) return false;
    }

    // Adjust T based on whether nonce was negated during signing
    Point const T_adj = pre_sig.needs_negation ? adaptor_point.negate() : adaptor_point;

    // Reconstruct R = R^ + T_adj (should have even y)
    Point const R = pre_sig.R_hat.add(T_adj);
    auto [R_x, R_y_odd] = R.x_bytes_and_parity();
    if (R_y_odd) return false;

    // Challenge: e = H("BIP0340/challenge", R.x || P.x || m)
    std::uint8_t challenge_data[96];
    std::memcpy(challenge_data, R_x.data(), 32);
    std::memcpy(challenge_data + 32, pubkey_x.data(), 32);
    std::memcpy(challenge_data + 64, msg.data(), 32);
    auto e_hash = tagged_hash("BIP0340/challenge", challenge_data, 96);
    Scalar const e = Scalar::from_bytes(e_hash);

    // Verify: s*G == R^ + e*P  (since s = k + e*d)
    Point const lhs = Point::generator().scalar_mul(pre_sig.s_hat);
    Point const eP = P.scalar_mul(e);
    Point const rhs_point = pre_sig.R_hat.add(eP);

    auto lhs_c = lhs.to_compressed();
    auto rhs_c = rhs_point.to_compressed();
    return lhs_c == rhs_c;
}

SchnorrSignature
schnorr_adaptor_adapt(const SchnorrAdaptorSig& pre_sig,
                      const Scalar& adaptor_secret) {
    // CT-004: use ct::scalar_cneg for explicit branchlessness on adaptor secret.
    // needs_negation is a public flag; adaptor_secret is a secret scalar.
    Scalar t = ct::scalar_cneg(adaptor_secret,
                               ct::bool_to_mask(pre_sig.needs_negation));

    // R = R^ + t_adj*G (should have even y since we ensured it during signing)
    Point const R = pre_sig.R_hat.add(ct::generator_mul(t));

    // s = s_hat + t — CT: t is the adaptor secret
    Scalar const s = ct::scalar_add(pre_sig.s_hat, t);

    detail::secure_erase(&t, sizeof(t));

    SchnorrSignature sig;
    sig.r = R.x().to_bytes();
    sig.s = s;
    return sig;
}

std::pair<Scalar, bool>
schnorr_adaptor_extract(const SchnorrAdaptorSig& pre_sig,
                        const SchnorrSignature& sig) {
    // adapt computed s = s_hat + t_adj where t_adj = needs_negation ? -t : t
    // so sig.s - s_hat = t_adj; reverse the negation to recover original t
    Scalar t = sig.s - pre_sig.s_hat;
    if (pre_sig.needs_negation) t = t.negate();

    if (t.is_zero()) {
        return {t, false};
    }
    return {t, true};
}

// -- ECDSA Adaptor Signatures -------------------------------------------------

ECDSAAdaptorSig
ecdsa_adaptor_sign(const Scalar& private_key,
                   const std::array<std::uint8_t, 32>& msg_hash,
                   const Point& adaptor_point) {
    // Generate nonce — must be non-const so secure_erase can zero them after use
    Scalar k = adaptor_nonce(private_key, msg_hash.data(), 32, adaptor_point, nullptr, 0);

    Scalar binding = ecdsa_adaptor_binding(adaptor_point);
    Point const base_nonce = ct::generator_mul(k);  // CT: k is secret
    Point const binding_point = ct::generator_mul(binding);

    // Bind the pre-signature to the advertised adaptor point without
    // changing the scalar k used for adaptation/extraction.
    Point const R_hat = base_nonce.add(binding_point);

    // R = k * T = (k*t) * G.  CT: k is secret nonce.
    Point const R = ct::scalar_mul(adaptor_point, k);

    // r = R.x mod n
    auto R_x_bytes = R.x().to_bytes();
    Scalar const r = Scalar::from_bytes(R_x_bytes);
    if (r.is_zero()) {
        // Degenerate case
        return ECDSAAdaptorSig{R_hat, Scalar::zero(), r};
    }

    // s = k^-^1 * (z + r*x)  where z = msg_hash
    Scalar const z = Scalar::from_bytes(msg_hash);
    if (k.is_zero()) {
        return ECDSAAdaptorSig{Point::infinity(), Scalar::zero(), Scalar::zero()};
    }
    Scalar k_inv = ct::scalar_inverse(k);  // CT: k is secret; non-const for secure_erase
    // CT: k_inv and private_key are secrets — use branchless CT arithmetic
    Scalar const s_hat = ct::scalar_mul(k_inv,
                             ct::scalar_add(z, ct::scalar_mul(r, private_key)));

    detail::secure_erase(&k, sizeof(k));
    detail::secure_erase(&k_inv, sizeof(k_inv));
    detail::secure_erase(&binding, sizeof(binding));

    return ECDSAAdaptorSig{R_hat, s_hat, r};
}

bool ecdsa_adaptor_verify(const ECDSAAdaptorSig& pre_sig,
                          const Point& public_key,
                          const std::array<std::uint8_t, 32>& msg_hash,
                          const Point& adaptor_point) {
    if (pre_sig.r.is_zero() || pre_sig.s_hat.is_zero()) return false;

    // Verify: s*R^ == z*G + r*P (rearranged ECDSA equation)
    Scalar const z = Scalar::from_bytes(msg_hash);
    Scalar const s_inv = pre_sig.s_hat.inverse();

    Point const u1G = Point::generator().scalar_mul(z * s_inv);
    Point const u2P = public_key.scalar_mul(pre_sig.r * s_inv);  // nosemgrep: secret-scalar-variable-time-mul
    Point const R_check = u1G.add(u2P);

    Scalar const binding = ecdsa_adaptor_binding(adaptor_point);
    Point const binding_point = ct::generator_mul(binding);
    Point const base_nonce = pre_sig.R_hat.add(binding_point.negate());

    auto r_hat_c = base_nonce.to_compressed();
    auto r_chk_c = R_check.to_compressed();
    return r_hat_c == r_chk_c;
}

ECDSASignature
ecdsa_adaptor_adapt(const ECDSAAdaptorSig& pre_sig,
                    const Scalar& adaptor_secret) {
    Scalar t_inv = ct::scalar_inverse(adaptor_secret); // CT: SafeGCD inverse on secret
    Scalar const s = ct::scalar_mul(pre_sig.s_hat, t_inv); // CT: t_inv derives from secret

    detail::secure_erase(&t_inv, sizeof(t_inv));

    ECDSASignature sig;
    sig.r = pre_sig.r;
    sig.s = s;
    return sig;
}

std::pair<Scalar, bool>
ecdsa_adaptor_extract(const ECDSAAdaptorSig& pre_sig,
                      const ECDSASignature& sig) {
    if (sig.s.is_zero() || pre_sig.s_hat.is_zero()) return {Scalar::zero(), false};

    // t = s * s^-^1 (multiplicative adaptor)
    Scalar const s_inv = sig.s.inverse();
    Scalar const t = pre_sig.s_hat * s_inv;

    if (t.is_zero()) return {t, false};
    return {t, true};
}

} // namespace secp256k1
