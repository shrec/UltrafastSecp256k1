// ============================================================================
// MuSig2: Two-Round Multi-Signature Scheme (BIP-327)
// ============================================================================

#include "secp256k1/musig2.hpp"
#include "secp256k1/schnorr.hpp"
#include "secp256k1/sha256.hpp"
#include <cstring>
#include <algorithm>

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

    std::array<uint8_t, 32> x_bytes;
    std::memcpy(x_bytes.data(), compressed.data() + 1, 32);
    auto x = FieldElement::from_bytes(x_bytes);

    // y^2 = x^3 + 7
    auto x3 = x.square() * x;
    auto y2 = x3 + FieldElement::from_uint64(7);

    // Optimized sqrt via addition chain
    auto y = y2.sqrt();

    if (y.square() != y2) return Point::infinity();

    // Select parity
    auto y_bytes = y.to_bytes();
    bool y_odd = (y_bytes[31] & 1) != 0;
    bool want_odd = (compressed[0] == 0x03);
    if (y_odd != want_odd) {
        y = FieldElement::zero() - y;
    }

    return Point::from_affine(x, y);
}

// Check if point has even Y
bool has_even_y(const Point& P) {
    auto uncomp = P.to_uncompressed();
    return (uncomp[64] & 1) == 0;
}

} // anonymous namespace

// -- Key Aggregation (KeyAgg) -------------------------------------------------
// BIP-327 KeyAgg: Q = sum(a_i * P_i)
// a_i = tagged_hash("KeyAgg coefficient", L || pk_i)
// where L = hash of all sorted pubkeys

MuSig2KeyAggCtx musig2_key_agg(const std::vector<std::array<uint8_t, 32>>& pubkeys) {
    MuSig2KeyAggCtx ctx{};
    std::size_t n = pubkeys.size();
    if (n == 0) return ctx;

    // L = tagged_hash("KeyAgg list", pk_1 || pk_2 || ... || pk_n)
    SHA256 l_ctx;
    // Use tagged hash prefix
    auto tag_hash = SHA256::hash("KeyAgg list", 11);
    l_ctx.update(tag_hash.data(), 32);
    l_ctx.update(tag_hash.data(), 32);
    for (std::size_t i = 0; i < n; ++i) {
        l_ctx.update(pubkeys[i].data(), 32);
    }
    auto L = l_ctx.finalize();

    // Compute coefficients: a_i = tagged_hash("KeyAgg coefficient", L || pk_i)
    // Exception: "second unique key" gets a_i = 1 (optimization, BIP-327)
    ctx.key_coefficients.resize(n);

    // Find second unique key (first key different from pubkeys[0])
    int second_key_idx = -1;
    for (std::size_t i = 1; i < n; ++i) {
        if (pubkeys[i] != pubkeys[0]) {
            second_key_idx = static_cast<int>(i);
            break;
        }
    }

    for (std::size_t i = 0; i < n; ++i) {
        if (static_cast<int>(i) == second_key_idx) {
            ctx.key_coefficients[i] = Scalar::one();
        } else {
            // a_i = tagged_hash("KeyAgg coefficient", L || pk_i)
            uint8_t coeff_input[64];
            std::memcpy(coeff_input, L.data(), 32);
            std::memcpy(coeff_input + 32, pubkeys[i].data(), 32);
            auto coeff_hash = tagged_hash("KeyAgg coefficient", coeff_input, 64);
            ctx.key_coefficients[i] = Scalar::from_bytes(coeff_hash);
        }
    }

    // Q = sum(a_i * P_i)
    // First, lift all x-only pubkeys to points
    Point Q = Point::infinity();
    for (std::size_t i = 0; i < n; ++i) {
        // Lift x-only to point (even Y)
        auto px = FieldElement::from_bytes(pubkeys[i]);
        auto x3 = px.square() * px;
        auto y2 = x3 + FieldElement::from_uint64(7);

        auto exp = FieldElement::from_hex(
            "3fffffffffffffffffffffffffffffffffffffffffffffffffffffffbfffff0c");
        auto y = FieldElement::one();
        auto base = y2;
        auto exp_bytes = exp.to_bytes();
        for (std::size_t b = 0; b < 256; ++b) {
            y = y.square();
            std::size_t byte_idx = b / 8;
            unsigned bit_idx = static_cast<unsigned>(7 - (b % 8));
            if ((exp_bytes[byte_idx] >> bit_idx) & 1) {
                y = y * base;
            }
        }

        auto y_bytes = y.to_bytes();
        if (y_bytes[31] & 1) {
            y = FieldElement::zero() - y;
        }

        auto Pi = Point::from_affine(px, y);
        auto aiPi = Pi.scalar_mul(ctx.key_coefficients[i]);
        Q = Q.add(aiPi);
    }

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

    // t = secret_key XOR tagged_hash("MuSig/aux", extra_input or zeros)
    auto sk_bytes = secret_key.to_bytes();
    std::array<uint8_t, 32> aux{};
    if (extra_input) std::memcpy(aux.data(), extra_input, 32);
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
    }

    // R1 = k1 * G, R2 = k2 * G
    auto R1 = Point::generator().scalar_mul(sec.k1);
    auto R2 = Point::generator().scalar_mul(sec.k2);
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
    auto bR2 = agg_nonce.R2.scalar_mul(session.b);
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
    const MuSig2SecNonce& sec_nonce,
    const Scalar& secret_key,
    const MuSig2KeyAggCtx& key_agg_ctx,
    const MuSig2Session& session,
    std::size_t signer_index) {

    // k = k1 + b * k2
    Scalar k = sec_nonce.k1 + session.b * sec_nonce.k2;

    // Adjust k if R was negated
    if (session.R_negated) {
        k = k.negate();
    }

    // Adjust secret key:
    // 1) Negate if this signer's pubkey P_i = d*G has odd Y
    //    (x-only pubkeys assume even Y, so effective d must match lift_x)
    Scalar d = secret_key;
    auto Pi = Point::generator().scalar_mul(d);
    if (!has_even_y(Pi)) {
        d = d.negate();
    }
    // 2) Negate if aggregate key Q was negated for even-Y
    if (key_agg_ctx.Q_negated) {
        d = d.negate();
    }

    // s_i = k + e * a_i * d  (mod n)
    return k + session.e * key_agg_ctx.key_coefficients[signer_index] * d;
}

// -- Partial Verification -----------------------------------------------------

bool musig2_partial_verify(
    const Scalar& partial_sig,
    const MuSig2PubNonce& pub_nonce,
    const std::array<uint8_t, 32>& pubkey,
    const MuSig2KeyAggCtx& key_agg_ctx,
    const MuSig2Session& session,
    std::size_t signer_index) {

    // s_i * G should equal R_i + b * R2_i + e * a_i * P_i
    // (with appropriate negation adjustments)

    auto sG = Point::generator().scalar_mul(partial_sig);

    auto R1_i = decompress_point(pub_nonce.R1);
    auto R2_i = decompress_point(pub_nonce.R2);

    // Effective nonce: R_i = R1_i + b * R2_i
    auto R_eff = R1_i.add(R2_i.scalar_mul(session.b));
    if (session.R_negated) {
        R_eff = R_eff.negate();
    }

    // Key contribution: e * a_i * P_i
    // Lift pubkey
    auto px = FieldElement::from_bytes(pubkey);
    auto x3 = px.square() * px;
    auto y2 = x3 + FieldElement::from_uint64(7);
    auto exp = FieldElement::from_hex(
        "3fffffffffffffffffffffffffffffffffffffffffffffffffffffffbfffff0c");
    auto y = FieldElement::one();
    auto base = y2;
    auto exp_bytes = exp.to_bytes();
    for (std::size_t b_idx = 0; b_idx < 256; ++b_idx) {
        y = y.square();
        std::size_t byte_idx = b_idx / 8;
        unsigned bit_idx = static_cast<unsigned>(7 - (b_idx % 8));
        if ((exp_bytes[byte_idx] >> bit_idx) & 1) {
            y = y * base;
        }
    }
    auto y_bytes = y.to_bytes();
    if (y_bytes[31] & 1) y = FieldElement::zero() - y;

    auto Pi = Point::from_affine(px, y);
    Scalar ea = session.e * key_agg_ctx.key_coefficients[signer_index];
    if (key_agg_ctx.Q_negated) ea = ea.negate();

    auto eaP = Pi.scalar_mul(ea);

    auto expected = R_eff.add(eaP);
    // Compare: sG == expected
    // Both are in Jacobian; convert to affine and compare x,y
    auto sG_x = sG.x().to_bytes();
    auto exp_x = expected.x().to_bytes();
    auto sG_y = sG.y().to_bytes();
    auto exp_y = expected.y().to_bytes();

    return sG_x == exp_x && sG_y == exp_y;
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
