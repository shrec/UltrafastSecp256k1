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
#include <cstdint>
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
    // P2-CT-RT-004: Replace the data-dependent retry loop with a fixed
    // 2-iteration CT select pattern (same fix as rfc6979_nonce_hedged).
    // The original loop `for (ctr=0; !parse_strict_nonzero(...); ++ctr)`
    // leaks whether hash >= n or == 0 through iteration count via timing.
    // Fix: always run exactly 2 iterations unconditionally; ct::scalar_select
    // picks the first valid candidate without any secret-dependent branch.
    // Probability of needing iteration 2: ~2^-128.
    // Both failing: ~2^-256 (caller will detect k==0 and treat as sign failure).
    Scalar cand1{};
    bool const ok1 = Scalar::parse_bytes_strict_nonzero(hash.data(), cand1);

    // Advance hash state unconditionally (CT: must execute regardless of ok1).
    // XOR byte 31 with 0x01 (deterministic counter advance, same as original loop ctr=0).
    hash[31] ^= std::uint8_t{0x01u};

    Scalar cand2{};
    bool const ok2 = Scalar::parse_bytes_strict_nonzero(hash.data(), cand2);
    (void)ok2;  // cand2 used only as fallback via ct::scalar_select; ok2 implicit

    // CT select: mask = ~0ULL if ok1 (use cand1), else 0ULL (use cand2)
    std::uint64_t const mask1 = static_cast<std::uint64_t>(
        -static_cast<std::int64_t>(static_cast<int>(ok1)));
    Scalar k = ct::scalar_select(cand1, cand2, mask1);

    detail::secure_erase(sk_bytes.data(), sk_bytes.size());
    detail::secure_erase(hash.data(), hash.size());
    detail::secure_erase(&cand1, sizeof(cand1));
    detail::secure_erase(&cand2, sizeof(cand2));
    return k;
}

static Scalar ecdsa_adaptor_binding(const Point& adaptor_point) {
    // BIP-340 tagged hash: SHA256(SHA256(tag) || SHA256(tag) || data).
    // Provides cross-protocol domain separation. Changed from "SHA256(tag || data)"
    // in v2 — this is a wire-format change for ECDSA adaptor signatures.
    // Protocol version: ecdsa_adaptor_bind_v2.
    constexpr char tag[] = "ecdsa_adaptor_bind_v2";
    auto tag_hash = SHA256::hash(reinterpret_cast<const std::uint8_t*>(tag),
                                 sizeof(tag) - 1);
    auto adaptor_bytes = adaptor_point.to_compressed();

    SHA256 h;
    h.update(tag_hash.data(), 32);
    h.update(tag_hash.data(), 32);
    h.update(adaptor_bytes.data(), adaptor_bytes.size());
    auto hash = h.finalize();

    // P2-CT-RT-004 (binding): same fixed 2-iteration CT select pattern applied
    // here for consistency.  Note: adaptor_point is PUBLIC data (it is sent
    // over the wire), so this function is NOT on a secret-bearing path.
    // The CT fix is applied anyway to keep the nonce-generation pattern uniform
    // and avoid future confusion if the data classification changes.
    Scalar bcand1{};
    bool const bok1 = Scalar::parse_bytes_strict_nonzero(hash.data(), bcand1);

    hash[31] ^= std::uint8_t{0x01u};  // deterministic advance, unconditional

    Scalar bcand2{};
    bool const bok2 = Scalar::parse_bytes_strict_nonzero(hash.data(), bcand2);
    (void)bok2;  // bcand2 is the fallback; bok2 is implicit via ct::scalar_select

    std::uint64_t const bmask1 = static_cast<std::uint64_t>(
        -static_cast<std::int64_t>(static_cast<int>(bok1)));
    Scalar binding = ct::scalar_select(bcand1, bcand2, bmask1);

    detail::secure_erase(hash.data(), hash.size());
    detail::secure_erase(&bcand1, sizeof(bcand1));
    detail::secure_erase(&bcand2, sizeof(bcand2));
    return binding;
}

// -- Schnorr Adaptor Signatures -----------------------------------------------

SchnorrAdaptorSig
schnorr_adaptor_sign(const Scalar& private_key,
                     const std::array<std::uint8_t, 32>& msg,
                     const Point& adaptor_point,
                     const std::array<std::uint8_t, 32>& aux_rand) {
    // Compute public key P = sk * G (CT, DPA-blinded — v9 RT-002 / TASK-002).
    // private_key is a long-term secret; the blinded variant activates DPA
    // (differential power analysis) blinding when secp256k1_context_randomize
    // has been called. Adopts the GENERATOR-MUL-CT constraint already enforced
    // on ecdsa_sign / schnorr_sign / musig2.
    Point P = ct::generator_mul_blinded(private_key);

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

    // R^ = k * G (the pre-nonce, before adapting) — CT: k is secret, use blinded for DPA defence
    Point R_hat = ct::generator_mul_blinded(k);

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

    // Adjust T based on whether nonce was negated during signing
    Point const T_adj = pre_sig.needs_negation ? adaptor_point.negate() : adaptor_point;

    // Reconstruct R = R^ + T_adj (should have even y).
    //
    // BUG FIX (2026-05-13): the prior implementation also computed
    //   (pre_sig.R_hat + adaptor_point).y_odd == pre_sig.needs_negation
    // as an "integrity check". That check is broken when needs_negation == true:
    // pre_sig.R_hat is the POST-negation R_hat (== -original_R_hat), so
    // (pre_sig.R_hat + T).y is unrelated to original_R.y and the comparison
    // fails for valid signatures. The cryptographic binding of needs_negation
    // is already provided by:
    //   (a) the R_y_odd check below — a flipped needs_neg yields R = R_hat ± T
    //       with the wrong T sign, which generally produces an odd-y R;
    //   (b) the s*G == R_hat + e*P verification — a tampered R changes e via
    //       the challenge hash so the signature equation fails.
    // The redundant-and-wrong check is removed.
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
    // Reject degenerate signatures — sig.s == 0 means the signature is invalid
    // and would produce a wrong adaptor secret (t = -s_hat) without error.
    if (sig.s.is_zero()) return {Scalar{}, false};

    // CT: use ct::scalar_sub + ct::scalar_cneg — t is the recovered adaptor
    // secret (private material). fast::operator- and fast::negate() are VT
    // and must not operate on the secret result even though the inputs
    // (sig.s, pre_sig.s_hat) are public.
    Scalar t = ct::scalar_sub(sig.s, pre_sig.s_hat);
    t = ct::scalar_cneg(t, ct::bool_to_mask(pre_sig.needs_negation));

    if (t.is_zero_ct()) {
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
    // v9 RT-002 / TASK-002: k is the secret nonce — use DPA-blinded variant.
    // binding is derived from the PUBLIC adaptor_point only and carries no
    // secret data, so the unblinded primitive is correct (and cheaper) for it.
    Point const base_nonce = ct::generator_mul_blinded(k);
    Point const binding_point = ct::generator_mul(binding);  // PUBLIC scalar — unblinded OK

    // Bind the pre-signature to the advertised adaptor point without
    // changing the scalar k used for adaptation/extraction.
    Point const R_hat = base_nonce.add(binding_point);

    // R = k * T = (k*t) * G.  CT: k is secret nonce.
    Point const R = ct::scalar_mul(adaptor_point, k);

    // r = R.x mod n
    auto R_x_bytes = R.x().to_bytes();
    Scalar const r = Scalar::from_bytes(R_x_bytes);
    if (r.is_zero()) {
        // SEC-008: degenerate case — return fully-zero sentinel.
        // Previously returned {R_hat, Scalar::zero(), r} which left a non-zero
        // R_hat in the output struct, creating a partially-populated degenerate
        // pre-signature that could mislead callers checking only r==0.
        // v9 RT-015 / TASK-022: the success-path erase block (below, lines
        // 329-331) is unreachable on this early return; mirror it here so
        // k, binding, and R_x_bytes do not linger in the stack frame.
        detail::secure_erase(&k, sizeof(k));
        detail::secure_erase(&binding, sizeof(binding));
        detail::secure_erase(R_x_bytes.data(), R_x_bytes.size());
        return ECDSAAdaptorSig{Point::infinity(), Scalar::zero(), Scalar::zero()};
    }

    // s = k^-^1 * (z + r*x)  where z = msg_hash
    // CT: do NOT branch on k.is_zero() here — fast::is_zero() has a secret-dependent
    // early-exit. adaptor_nonce() guarantees k != 0 via strict-nonzero parsing; the
    // r.is_zero() guard above also catches the k=0 degenerate path (R=infinity→r=0).
    Scalar const z = Scalar::from_bytes(msg_hash);
    Scalar k_inv = ct::scalar_inverse(k);  // CT: k is secret; non-const for secure_erase
    // CT: k_inv and private_key are secrets — use branchless CT arithmetic
    Scalar s_hat = ct::scalar_mul(k_inv,
                       ct::scalar_add(z, ct::scalar_mul(r, private_key)));

    detail::secure_erase(&k, sizeof(k));
    detail::secure_erase(&k_inv, sizeof(k_inv));
    detail::secure_erase(&binding, sizeof(binding));

    if (s_hat.is_zero_ct()) {
        detail::secure_erase(&s_hat, sizeof(s_hat));
        return ECDSAAdaptorSig{Point::infinity(), Scalar::zero(), Scalar::zero()};
    }
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

    if (s.is_zero_ct()) return ECDSASignature{}; // SEC-003: degenerate — caller must handle

    ECDSASignature sig;
    sig.r = pre_sig.r;
    sig.s = s;
    return sig;
}

std::pair<Scalar, bool>
ecdsa_adaptor_extract(const ECDSAAdaptorSig& pre_sig,
                      const ECDSASignature& sig) {
    if (sig.s.is_zero() || pre_sig.s_hat.is_zero()) return {Scalar::zero(), false};

    // t = s_hat * sig.s^-1 (multiplicative adaptor secret extraction).
    // sig.s is public data; but t is the secret adaptor witness — use CT mul (SEC-001/CT-001).
    Scalar s_inv = sig.s.inverse();
    Scalar const t = ct::scalar_mul(pre_sig.s_hat, s_inv);
    detail::secure_erase(&s_inv, sizeof(s_inv));

    if (t.is_zero_ct()) return {Scalar::zero(), false};
    return {t, true};
}

} // namespace secp256k1
