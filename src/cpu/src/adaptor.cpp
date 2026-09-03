// ============================================================================
// Adaptor Signatures -- Implementation
// ============================================================================

#include "secp256k1/adaptor.hpp"
#include "secp256k1/sha256.hpp"
#include "secp256k1/tagged_hash.hpp"
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

// -- Tagged-hash midstates for the adaptor domain tags -------------------------
// Every tag below is a compile-time constant, so the SHA-256 state after the
// 64-byte BIP-340 prefix SHA256(tag)||SHA256(tag) is fixed for the life of the
// process. Build it once here rather than re-deriving SHA256(tag) and
// recompressing the prefix block on every call. The digest is identical by
// construction — the midstate IS the state after those 64 bytes, and the tag
// STRINGS are unchanged, so the signature bytes are unchanged. The prefix is
// exactly one whole block, so nothing is left buffered (SHA256-MIDSTATE-BOUNDARY).
// "BIP0340/challenge" is not repeated here: detail::g_challenge_midstate in
// tagged_hash.hpp already holds it.
static const SHA256 g_adaptor_nonce_midstate =
    detail::make_tag_midstate("adaptor_nonce_v1");
static const SHA256 g_dleq_challenge_midstate =
    detail::make_tag_midstate("ufsecp/ecdsa_adaptor_dleq_v1");
static const SHA256 g_dleq_nonce_midstate =
    detail::make_tag_midstate("ufsecp/ecdsa_adaptor_dleq_nonce_v1");

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
    // g_adaptor_nonce_midstate already encodes SHA256(tag)||SHA256(tag) for
    // "adaptor_nonce_v1" — the BIP-340 prefix, absorbed but not re-derived.
    SHA256 h = g_adaptor_nonce_midstate;
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

// -- ECDSA adaptor DLEQ (GHSA-c7q2-gv3g-rgxm) ---------------------------------
// The pre-signature carries a Chaum-Pedersen proof that R_hat = k*G and R = k*T
// share the same discrete log k (i.e. log_G(R_hat) == log_T(R)). This binds
// r = R.x to the adaptor point T, which the old "binding scalar" did NOT do.

// DLEQ Fiat-Shamir challenge e = H(tag, R_hat || R || T || A || B) reduced mod n.
// All inputs are PUBLIC (this runs in both sign and the variable-time verify path),
// so a non-CT reduction is correct. Binding the proof to the full statement
// (R_hat, R, T) and both commitments (A, B) makes it non-malleable.
static Scalar ecdsa_adaptor_dleq_challenge(const Point& R_hat, const Point& R,
                                           const Point& T,
                                           const Point& A, const Point& B) {
    // "ufsecp/ecdsa_adaptor_dleq_v1" prefix, precomputed at static init.
    SHA256 h = g_dleq_challenge_midstate;
    const Point* pts[] = {&R_hat, &R, &T, &A, &B};
    for (const Point* p : pts) {
        auto c = p->to_compressed();
        h.update(c.data(), c.size());
    }
    auto hash = h.finalize();
    // The challenge may be any value in [0, n); reduce mod n. (e == 0 has prob
    // ~2^-256 and is still a valid Fiat-Shamir challenge.)
    return Scalar::from_bytes(hash.data());
}

// Deterministic DLEQ proof nonce rho (SECRET). Must be non-zero, unpredictable and
// unique per pre-signature — a repeated or leaked rho across two statements would
// expose k. Derived like adaptor_nonce with a DISTINCT domain and the secret nonce
// k mixed in (so rho != k and is unique even for a fixed (x,msg,T)).
static Scalar ecdsa_adaptor_dleq_nonce(const Scalar& privkey, const Scalar& k,
                                       const Point& R_hat, const Point& R,
                                       const Point& adaptor_point,
                                       const std::uint8_t* msg, std::size_t msg_len) {
    // "ufsecp/ecdsa_adaptor_dleq_nonce_v1" prefix, precomputed at static init.
    SHA256 h = g_dleq_nonce_midstate;
    auto sk_bytes = privkey.to_bytes();
    h.update(sk_bytes.data(), 32);
    auto k_bytes = k.to_bytes();
    h.update(k_bytes.data(), 32);
    const Point* pts[] = {&R_hat, &R, &adaptor_point};
    for (const Point* p : pts) { auto c = p->to_compressed(); h.update(c.data(), c.size()); }
    h.update(msg, msg_len);
    auto hash = h.finalize();
    // Fixed 2-iteration CT-select strict-nonzero (same pattern as adaptor_nonce).
    Scalar c1{};
    bool const ok1 = Scalar::parse_bytes_strict_nonzero(hash.data(), c1);
    hash[31] ^= std::uint8_t{0x01u};
    Scalar c2{};
    (void)Scalar::parse_bytes_strict_nonzero(hash.data(), c2);
    std::uint64_t const mask1 = static_cast<std::uint64_t>(
        -static_cast<std::int64_t>(static_cast<int>(ok1)));
    Scalar rho = ct::scalar_select(c1, c2, mask1);
    detail::secure_erase(sk_bytes.data(), sk_bytes.size());
    detail::secure_erase(k_bytes.data(), k_bytes.size());
    detail::secure_erase(hash.data(), hash.size());
    detail::secure_erase(&c1, sizeof(c1));
    detail::secure_erase(&c2, sizeof(c2));
    return rho;
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
    // Same digest as tagged_hash("BIP0340/challenge", ...) — the shared midstate
    // is exactly the state after that tag's 64-byte prefix, so this only skips
    // re-deriving a constant. Matches schnorr_challenge_scalar.
    auto e_hash = detail::cached_tagged_hash(detail::g_challenge_midstate,
                                             challenge_data, 96);
    Scalar e = Scalar::from_bytes(e_hash);

    // s = k + e * sk — CT: both k (secret nonce) and sk (secret key) are secret
    Scalar const s_hat = ct::scalar_add(k, ct::scalar_mul(e, sk));

    detail::secure_erase(&k, sizeof(k));  // SEC-005: const_cast was unnecessary — k is non-const
    detail::secure_erase(&sk, sizeof(sk));
    detail::secure_erase(e_hash.data(), e_hash.size());
    detail::secure_erase(&e, sizeof(e));
    detail::secure_erase(challenge_data, sizeof(challenge_data));
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
    // Shared "BIP0340/challenge" midstate — byte-identical to the generic
    // tagged_hash() call it replaces (see schnorr_adaptor_sign).
    auto e_hash = detail::cached_tagged_hash(detail::g_challenge_midstate,
                                             challenge_data, 96);
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
    static const ECDSAAdaptorSig kZero{Point::infinity(), Point::infinity(),
                                       Scalar::zero(), Scalar::zero(),
                                       Scalar::zero(), Scalar::zero()};
    // Secret nonce k — non-const so secure_erase can zero it after use.
    Scalar k = adaptor_nonce(private_key, msg_hash.data(), 32, adaptor_point, nullptr, 0);

    // R_hat = k*G (nonce point in G), R = k*T (nonce point in the adaptor base T).
    // GHSA-c7q2-gv3g-rgxm: R_hat is now the bare k*G (no "binding" scalar — that
    // mechanism cancelled in verify and bound nothing); r is bound to T below by a
    // DLEQ proof that R_hat and R share the same k.
    // CT: k is the secret nonce — DPA-blinded generator mul; variable-base mul on T.
    Point const R_hat = ct::generator_mul_blinded(k);
    Point const R = ct::scalar_mul(adaptor_point, k);

    // r = R.x mod n
    auto R_x_bytes = R.x().to_bytes();
    Scalar const r = Scalar::from_bytes(R_x_bytes);
    if (r.is_zero()) {
        // SEC-008: degenerate — fully-zero sentinel (mirror the success-path erase).
        detail::secure_erase(&k, sizeof(k));
        detail::secure_erase(R_x_bytes.data(), R_x_bytes.size());
        return kZero;
    }

    // s_hat = k^-1 * (z + r*x)  where z = msg_hash.  CT: k, private_key are secret.
    // Do NOT branch on k.is_zero(): adaptor_nonce() returns a strict-nonzero k, and the
    // r.is_zero() guard above already catches the k=0 degenerate path (R=∞ → r=0); the
    // CT ct::scalar_inverse is itself is_zero-safe (SafeGCD, no data-dependent branch).
    Scalar const z = Scalar::from_bytes(msg_hash);
    Scalar k_inv = ct::scalar_inverse(k);  // CT: k secret; non-const for secure_erase
    Scalar s_hat = ct::scalar_mul(k_inv,
                       ct::scalar_add(z, ct::scalar_mul(r, private_key)));

    // Chaum-Pedersen DLEQ proof that log_G(R_hat) == log_T(R) == k:
    //   A = rho*G, B = rho*T ; e = H(R_hat, R, T, A, B) ; s_dleq = rho + e*k (mod n).
    // CT: rho, k are secret nonces (A, B, s_dleq become public proof material). e is public.
    Scalar rho = ecdsa_adaptor_dleq_nonce(private_key, k, R_hat, R, adaptor_point,
                                          msg_hash.data(), 32);
    Point const A = ct::generator_mul_blinded(rho);
    Point const B = ct::scalar_mul(adaptor_point, rho);
    Scalar const dleq_e = ecdsa_adaptor_dleq_challenge(R_hat, R, adaptor_point, A, B);
    Scalar dleq_s = ct::scalar_add(rho, ct::scalar_mul(dleq_e, k));

    detail::secure_erase(&k, sizeof(k));
    detail::secure_erase(&k_inv, sizeof(k_inv));
    detail::secure_erase(&rho, sizeof(rho));
    detail::secure_erase(R_x_bytes.data(), R_x_bytes.size());

    if (s_hat.is_zero_ct()) {
        detail::secure_erase(&s_hat, sizeof(s_hat));
        detail::secure_erase(&dleq_s, sizeof(dleq_s));
        return kZero;
    }
    return ECDSAAdaptorSig{R_hat, R, s_hat, r, dleq_e, dleq_s};
}

bool ecdsa_adaptor_verify(const ECDSAAdaptorSig& pre_sig,
                          const Point& public_key,
                          const std::array<std::uint8_t, 32>& msg_hash,
                          const Point& adaptor_point) {
    // All inputs are PUBLIC, so this whole path is correctly variable-time
    // (CLAUDE.md CT-VERIFY boundary). The scalar muls below operate on public
    // pre-signature / proof / message scalars, never on a secret.
    if (pre_sig.r.is_zero() || pre_sig.s_hat.is_zero()) return false;
    if (pre_sig.R_hat.is_infinity() || pre_sig.R.is_infinity()) return false;
    if (adaptor_point.is_infinity() || public_key.is_infinity()) return false;

    // (1) r must equal R.x mod n — ties the published r to the proven point R.
    auto Rx = pre_sig.R.x().to_bytes();
    Scalar const r_from_R = Scalar::from_bytes(Rx);
    if (!(r_from_R == pre_sig.r)) return false;

    // (2) DLEQ: recompute the commitments and the challenge.
    //   A' = s_dleq*G - e*R_hat ;  B' = s_dleq*T - e*R ;  e' = H(R_hat, R, T, A', B').
    //   e' == e proves R_hat = k*G and R = k*T share the same k, so r = R.x is bound to T.
    Point const A = Point::generator().scalar_mul(pre_sig.dleq_s)            // nosemgrep: secret-scalar-variable-time-mul
                        .add(pre_sig.R_hat.scalar_mul(pre_sig.dleq_e).negate());  // nosemgrep: secret-scalar-variable-time-mul
    Point const B = adaptor_point.scalar_mul(pre_sig.dleq_s)                 // nosemgrep: secret-scalar-variable-time-mul
                        .add(pre_sig.R.scalar_mul(pre_sig.dleq_e).negate());      // nosemgrep: secret-scalar-variable-time-mul
    if (A.is_infinity() || B.is_infinity()) return false;
    Scalar const e_check = ecdsa_adaptor_dleq_challenge(pre_sig.R_hat, pre_sig.R,
                                                        adaptor_point, A, B);
    if (!(e_check == pre_sig.dleq_e)) return false;

    // (3) ECDSA relation: s_hat*R_hat == z*G + r*P  (= (z + r*x)*G when s_hat = k^-1(z+rx)).
    Scalar const z = Scalar::from_bytes(msg_hash);
    Point const lhs = pre_sig.R_hat.scalar_mul(pre_sig.s_hat);               // nosemgrep: secret-scalar-variable-time-mul
    Point const rhs = Point::generator().scalar_mul(z)                       // nosemgrep: secret-scalar-variable-time-mul
                          .add(public_key.scalar_mul(pre_sig.r));            // nosemgrep: secret-scalar-variable-time-mul
    if (lhs.is_infinity() || rhs.is_infinity()) return false;
    return lhs.to_compressed() == rhs.to_compressed();
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
