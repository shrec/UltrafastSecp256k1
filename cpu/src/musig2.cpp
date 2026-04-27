// ============================================================================
// MuSig2: Two-Round Multi-Signature Scheme (BIP-327)
// ============================================================================

#include "secp256k1/musig2.hpp"
#include "secp256k1/schnorr.hpp"
#include "secp256k1/sha256.hpp"
#include "secp256k1/pippenger.hpp"
#include "secp256k1/ct/point.hpp"
#include "secp256k1/ct/scalar.hpp"
#include "secp256k1/ct/field.hpp"
#include "secp256k1/ct/ops.hpp"
#include "secp256k1/detail/secure_erase.hpp"
#include "secp256k1/detail/csprng.hpp"
#include <cstring>
#include <algorithm>

namespace {
using secp256k1::detail::secure_erase;
} // anonymous namespace

namespace secp256k1 {

using fast::Scalar;
using fast::Point;
using fast::FieldElement;

// -- Helpers ------------------------------------------------------------------

namespace {

// Decompress a 33-byte compressed point
Point decompress_point(const std::array<uint8_t, 33>& compressed) {
    if (compressed[0] != 0x02 && compressed[0] != 0x03) {
        return Point::infinity();
    }

    // Strict: reject x >= p
    FieldElement x;
    if (!FieldElement::parse_bytes_strict(compressed.data() + 1, x)) {
        return Point::infinity();
    }

    // y^2 = x^3 + 7
    auto x3 = x.square() * x;
    auto y2 = x3 + FieldElement::from_uint64(7);

    // Optimized sqrt via addition chain
    auto y = y2.sqrt();

    if (y.square() != y2) return Point::infinity();

    // Select parity (limbs()[0] LSB == big-endian byte[31] LSB for normalized FE)
    bool const y_odd = (y.limbs()[0] & 1) != 0;
    bool const want_odd = (compressed[0] == 0x03);
    if (y_odd != want_odd) {
        y = y.negate();
    }

    return Point::from_affine(x, y);
}

// Check if point has even Y
bool has_even_y(const Point& P) {
    return P.has_even_y();
}

} // anonymous namespace

// -- Key Aggregation (KeyAgg) -------------------------------------------------
// BIP-327 KeyAgg: Q = sum(a_i * P_i)
// a_i = tagged_hash("KeyAgg coefficient", L || pk_i)
// where L = tagged_hash("KeyAgg list", pk_1 || ... || pk_n)
// pk_i are 33-byte compressed pubkeys (prefix byte preserves Y parity).

MuSig2KeyAggCtx musig2_key_agg(const std::vector<std::array<uint8_t, 33>>& pubkeys) {
    MuSig2KeyAggCtx ctx{};
    std::size_t const n = pubkeys.size();
    if (n == 0) return ctx;

    // Validate ALL pubkeys upfront AND cache the lifted points.
    // BIP-327 KeyAgg uses lift_x (even-Y) for point arithmetic; the full
    // 33-byte key (with parity prefix) is used ONLY for hash inputs (L and
    // coefficient).  Using actual parity here breaks partial_sign/verify
    // consistency: musig2_partial_sign adjusts d for P_i's Y parity assuming
    // even-Y keys in Q, so key_agg MUST use even-Y lifting to match.
    static thread_local std::vector<Point> points;
    points.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        std::array<uint8_t, 33> even_key = pubkeys[i];
        even_key[0] = 0x02; // force even-Y (lift_x) for point arithmetic
        auto pt = decompress_point(even_key);
        if (pt.is_infinity()) return ctx;
        points[i] = pt;
    }

    // BIP-327: L = tagged_hash("KeyAgg list", pk_1 || ... || pk_n) — 33 bytes each
    SHA256 l_ctx;
    auto tag_hash = SHA256::hash("KeyAgg list", 11);
    l_ctx.update(tag_hash.data(), 32);
    l_ctx.update(tag_hash.data(), 32);
    for (std::size_t i = 0; i < n; ++i) {
        l_ctx.update(pubkeys[i].data(), 33);
    }
    auto L = l_ctx.finalize();

    // BIP-327: pk2 = first key in list that differs from pk_1.
    // Keys equal to pk2 get coefficient 1; all others get the hash coefficient.
    ctx.key_coefficients.resize(n);
    const std::array<uint8_t, 33>* pk2 = nullptr;
    for (std::size_t i = 1; i < n; ++i) {
        if (pubkeys[i] != pubkeys[0]) {
            pk2 = &pubkeys[i];
            break;
        }
    }

    for (std::size_t i = 0; i < n; ++i) {
        if (pk2 && pubkeys[i] == *pk2) {
            ctx.key_coefficients[i] = Scalar::one();
        } else {
            // a_i = tagged_hash("KeyAgg coefficient", L || pk_i) — pk_i is 33 bytes
            uint8_t coeff_input[65];
            std::memcpy(coeff_input, L.data(), 32);
            std::memcpy(coeff_input + 32, pubkeys[i].data(), 33);
            auto coeff_hash = tagged_hash("KeyAgg coefficient", coeff_input, 65);
            ctx.key_coefficients[i] = Scalar::from_bytes(coeff_hash);
        }
    }

    // Q = sum(a_i * P_i) via MSM (Shamir trick for n=2, Pippenger for n>=48).
    // Cached points from validation loop — no double sqrt.
    Point Q = msm(ctx.key_coefficients, points);

    ctx.Q = Q;

    // Ensure even Y for BIP-340 compatibility
    ctx.Q_negated = !has_even_y(Q);
    if (ctx.Q_negated) {
        ctx.Q = Q.negate();
    }

    ctx.Q_x = ctx.Q.x().to_bytes();
    return ctx;
}

// -- Nonce Generation ---------------------------------------------------------

std::pair<MuSig2SecNonce, MuSig2PubNonce> musig2_nonce_gen(
    const Scalar& secret_key,
    const std::array<uint8_t, 32>& pub_key,
    const std::array<uint8_t, 32>& agg_pub_key,
    const std::array<uint8_t, 32>& msg,
    const uint8_t* extra_input) {

    MuSig2SecNonce sec{};
    MuSig2PubNonce pub{};

    // t = secret_key XOR tagged_hash("MuSig/aux", extra_input or fresh_random)
    // When extra_input==NULL we fill with OS randomness so nonces are never
    // deterministic. A caller always passing NULL would otherwise produce the
    // same k1/k2 for identical (secret_key, msg) inputs — enabling nonce reuse.
    auto sk_bytes = secret_key.to_bytes();
    std::array<uint8_t, 32> aux{};
    if (extra_input) {
        std::memcpy(aux.data(), extra_input, 32);
    } else {
        secp256k1::detail::csprng_fill(aux.data(), 32);
    }
    auto aux_hash = tagged_hash("MuSig/aux", aux.data(), 32);

    uint8_t t[32];
    for (std::size_t i = 0; i < 32; ++i) t[i] = sk_bytes[i] ^ aux_hash[i];

    // k1 = tagged_hash("MuSig/nonce", t || pub_key || agg_pub_key || msg || 0x01)
    {
        uint8_t nonce_input[129];
        std::memcpy(nonce_input, t, 32);
        std::memcpy(nonce_input + 32, pub_key.data(), 32);
        std::memcpy(nonce_input + 64, agg_pub_key.data(), 32);
        std::memcpy(nonce_input + 96, msg.data(), 32);
        nonce_input[128] = 0x01;
        auto k1_hash = tagged_hash("MuSig/nonce", nonce_input, 129);
        sec.k1 = Scalar::from_bytes(k1_hash);
        secure_erase(nonce_input, sizeof(nonce_input));
        secure_erase(k1_hash.data(), k1_hash.size());
    }

    // k2 = tagged_hash("MuSig/nonce", t || pub_key || agg_pub_key || msg || 0x02)
    {
        uint8_t nonce_input[129];
        std::memcpy(nonce_input, t, 32);
        std::memcpy(nonce_input + 32, pub_key.data(), 32);
        std::memcpy(nonce_input + 64, agg_pub_key.data(), 32);
        std::memcpy(nonce_input + 96, msg.data(), 32);
        nonce_input[128] = 0x02;
        auto k2_hash = tagged_hash("MuSig/nonce", nonce_input, 129);
        sec.k2 = Scalar::from_bytes(k2_hash);
        secure_erase(nonce_input, sizeof(nonce_input));
        secure_erase(k2_hash.data(), k2_hash.size());
    }

    // Zeroize secret key material now that nonces are derived
    secure_erase(sk_bytes.data(), sk_bytes.size());
    secure_erase(aux_hash.data(), aux_hash.size());
    secure_erase(t, sizeof(t));

    // R1 = k1 * G, R2 = k2 * G (CT: nonces are secret, must use constant-time path)
    auto R1 = ct::generator_mul(sec.k1);
    auto R2 = ct::generator_mul(sec.k2);
    pub.R1 = R1.to_compressed();
    pub.R2 = R2.to_compressed();

    return {sec, pub};
}

// -- Nonce Serialization ------------------------------------------------------

std::array<uint8_t, 66> MuSig2PubNonce::serialize() const {
    std::array<uint8_t, 66> out{};
    std::memcpy(out.data(), R1.data(), 33);
    std::memcpy(out.data() + 33, R2.data(), 33);
    return out;
}

MuSig2PubNonce MuSig2PubNonce::deserialize(const std::array<uint8_t, 66>& data) {
    MuSig2PubNonce nonce{};
    std::memcpy(nonce.R1.data(), data.data(), 33);
    std::memcpy(nonce.R2.data(), data.data() + 33, 33);
    return nonce;
}

// -- Nonce Aggregation --------------------------------------------------------

MuSig2AggNonce musig2_nonce_agg(const std::vector<MuSig2PubNonce>& pub_nonces) {
    MuSig2AggNonce agg{};
    agg.R1 = Point::infinity();
    agg.R2 = Point::infinity();

    for (const auto& nonce : pub_nonces) {
        auto r1 = decompress_point(nonce.R1);
        auto r2 = decompress_point(nonce.R2);
        agg.R1 = agg.R1.add(r1);
        agg.R2 = agg.R2.add(r2);
    }

    return agg;
}

// -- Session Start ------------------------------------------------------------

MuSig2Session musig2_start_sign_session(
    const MuSig2AggNonce& agg_nonce,
    const MuSig2KeyAggCtx& key_agg_ctx,
    const std::array<uint8_t, 32>& msg) {

    MuSig2Session session{};

    // b = tagged_hash("MuSig/noncecoef", aggR1 || aggR2 || Q_x || msg)
    auto R1_comp = agg_nonce.R1.to_compressed();
    auto R2_comp = agg_nonce.R2.to_compressed();

    uint8_t b_input[130]; // 33 + 33 + 32 + 32
    std::memcpy(b_input, R1_comp.data(), 33);
    std::memcpy(b_input + 33, R2_comp.data(), 33);
    std::memcpy(b_input + 66, key_agg_ctx.Q_x.data(), 32);
    std::memcpy(b_input + 98, msg.data(), 32);
    auto b_hash = tagged_hash("MuSig/noncecoef", b_input, 130);
    session.b = Scalar::from_bytes(b_hash);

    // R = R1 + b * R2
    auto bR2 = agg_nonce.R2.scalar_mul(session.b);  // nosemgrep: secret-scalar-variable-time-mul
    session.R = agg_nonce.R1.add(bR2);

    // Negate R if needed for even Y
    session.R_negated = !has_even_y(session.R);
    if (session.R_negated) {
        session.R = session.R.negate();
    }

    // e = tagged_hash("BIP0340/challenge", R.x || Q_x || msg)
    auto R_x = session.R.x().to_bytes();
    uint8_t e_input[96];
    std::memcpy(e_input, R_x.data(), 32);
    std::memcpy(e_input + 32, key_agg_ctx.Q_x.data(), 32);
    std::memcpy(e_input + 64, msg.data(), 32);
    auto e_hash = tagged_hash("BIP0340/challenge", e_input, 96);
    session.e = Scalar::from_bytes(e_hash);

    return session;
}

// -- Partial Signing ----------------------------------------------------------

Scalar musig2_partial_sign(
    MuSig2SecNonce& sec_nonce,
    const Scalar& secret_key,
    const MuSig2KeyAggCtx& key_agg_ctx,
    const MuSig2Session& session,
    std::size_t signer_index) {

    // Bounds check: signer_index must be valid for the key_coefficients vector
    if (signer_index >= key_agg_ctx.key_coefficients.size()) {
        return Scalar::zero();
    }

    // k = k1 + b * k2
    Scalar k = sec_nonce.k1 + session.b * sec_nonce.k2;

    // CT conditional negate k if R was negated (R_negated is public,
    // but keep branchless for consistency and to avoid pipeline leaks).
    {
        std::uint64_t const mask = ct::bool_to_mask(session.R_negated);
        Scalar const neg_k = k.negate();
        k = ct::scalar_select(neg_k, k, mask);
    }

    // Adjust secret key -- fully constant-time path:
    // 1) Compute P_i = d*G using CT generator multiplication (the secret
    //    key d is the sensitive input; fast::scalar_mul is variable-time).
    Scalar d = secret_key;
    auto Pi = ct::generator_mul(d);

    // 2) CT negate d if P_i has odd Y (x-only pubkeys assume even Y).
    //    Point::has_even_y() uses fast FieldElement::inverse() (SafeGCD,
    //    variable-time). Since Pi depends on the secret key d, the Z
    //    coordinate is secret-dependent. Use CT field inverse instead.
    {
        FieldElement z_inv = ct::field_inv(Pi.z_raw());
        FieldElement z_inv2 = z_inv.square();
        FieldElement y_aff = Pi.y_raw() * z_inv2 * z_inv;
        // y_aff is fully reduced -- LSB gives parity (0 = even, 1 = odd)
        bool const odd_y = (y_aff.limbs()[0] & 1) != 0;
        std::uint64_t const mask = ct::bool_to_mask(odd_y);
        Scalar const neg_d = d.negate();
        d = ct::scalar_select(neg_d, d, mask);
    }

    // 3) CT negate d if aggregate key Q was negated for even-Y.
    {
        std::uint64_t const mask = ct::bool_to_mask(key_agg_ctx.Q_negated);
        Scalar const neg_d = d.negate();
        d = ct::scalar_select(neg_d, d, mask);
    }

    // s_i = k + e * a_i * d  (mod n)
    // Scalar +/* are fixed-iteration multi-limb arithmetic -- CT by construction.
    Scalar const result = k + session.e * key_agg_ctx.key_coefficients[signer_index] * d;

    // Erase secret nonce and adjusted signing key from stack, then consume
    // the caller's secret nonce to enforce single-use (M-03).
    secure_erase(&k, sizeof(k));
    secure_erase(&d, sizeof(d));
    secure_erase(&sec_nonce.k1, sizeof(sec_nonce.k1));
    secure_erase(&sec_nonce.k2, sizeof(sec_nonce.k2));

    return result;
}

// -- Partial Verification -----------------------------------------------------

bool musig2_partial_verify(
    const Scalar& partial_sig,
    const MuSig2PubNonce& pub_nonce,
    const std::array<uint8_t, 32>& pubkey,
    const MuSig2KeyAggCtx& key_agg_ctx,
    const MuSig2Session& session,
    std::size_t signer_index) {

    // Bounds check: signer_index must be valid for the key_coefficients vector.
    // musig2_partial_sign already has this guard; verify must be consistent.
    if (signer_index >= key_agg_ctx.key_coefficients.size()) {
        return false;
    }

    // s_i * G should equal R_i + b * R2_i + e * a_i * P_i
    // (with appropriate negation adjustments)

    auto sG = Point::generator().scalar_mul(partial_sig);

    auto R1_i = decompress_point(pub_nonce.R1);
    auto R2_i = decompress_point(pub_nonce.R2);
    // BIP-327 §4 PartialSigVerify: reject invalid (infinity) nonce points.
    if (R1_i.is_infinity() || R2_i.is_infinity()) {
        return false;
    }

    // Effective nonce: R_i = R1_i + b * R2_i
    auto R_eff = R1_i.add(R2_i.scalar_mul(session.b));
    if (session.R_negated) {
        R_eff = R_eff.negate();
    }

    // Key contribution: e * a_i * P_i
    // Lift pubkey (strict: reject x >= p)
    FieldElement px;
    if (!FieldElement::parse_bytes_strict(pubkey, px)) return false;
    auto x3 = px.square() * px;
    auto y2 = x3 + FieldElement::from_uint64(7);
    // sqrt via optimized addition chain (~253 sqr + 13 mul)
    auto y = y2.sqrt();
    // Validate sqrt: reject non-residue (x not on curve)
    if (y.square() != y2) return false;
    // BIP-340: ensure even Y
    if (y.limbs()[0] & 1) y = y.negate();

    auto Pi = Point::from_affine(px, y);
    Scalar ea = session.e * key_agg_ctx.key_coefficients[signer_index];
    if (key_agg_ctx.Q_negated) ea = ea.negate();

    auto eaP = Pi.scalar_mul(ea);

    auto expected = R_eff.add(eaP);
    // Compare: sG == expected (Jacobian cross-multiplication, avoids 2 inversions)
    // (X1,Y1,Z1) == (X2,Y2,Z2)  iff  X1*Z2^2 == X2*Z1^2  &&  Y1*Z2^3 == Y2*Z1^3
    auto const z1sq = sG.z().square();
    auto const z2sq = expected.z().square();
    if (sG.X() * z2sq != expected.X() * z1sq) return false;
    auto const z1cu = z1sq * sG.z();
    auto const z2cu = z2sq * expected.z();
    return sG.Y() * z2cu == expected.Y() * z1cu;
}

// -- Signature Aggregation ----------------------------------------------------

std::array<uint8_t, 64> musig2_partial_sig_agg(
    const std::vector<Scalar>& partial_sigs,
    const MuSig2Session& session) {

    // s = sum(s_i)
    Scalar s = Scalar::zero();
    for (const auto& si : partial_sigs) {
        s += si;
    }

    // Final signature: (R.x, s)
    auto R_x = session.R.x().to_bytes();
    auto s_bytes = s.to_bytes();

    std::array<uint8_t, 64> sig{};
    std::memcpy(sig.data(), R_x.data(), 32);
    std::memcpy(sig.data() + 32, s_bytes.data(), 32);
    return sig;
}

} // namespace secp256k1
