// ============================================================================
// FROST Threshold Signatures — Implementation
// ============================================================================

#include "secp256k1/frost.hpp"
#include "secp256k1/sha256.hpp"
#include "secp256k1/schnorr.hpp"
#include <algorithm>
#include <cstring>

namespace secp256k1 {

using fast::Point;
using fast::Scalar;

// ── Internal Helpers ─────────────────────────────────────────────────────────

// Deterministic scalar from seed + context
static Scalar derive_scalar(const std::uint8_t* seed, std::size_t seed_len,
                             const char* context, std::uint32_t index) {
    SHA256 h;
    h.update(seed, seed_len);
    h.update(context, std::strlen(context));
    // Index as big-endian 4 bytes
    std::uint8_t idx_be[4] = {
        std::uint8_t(index >> 24), std::uint8_t(index >> 16),
        std::uint8_t(index >> 8), std::uint8_t(index)
    };
    h.update(idx_be, 4);
    auto hash = h.finalize();
    return Scalar::from_bytes(hash);
}

// Evaluate polynomial f(x) = a_0 + a_1*x + a_2*x^2 + ... at x
static Scalar poly_eval(const std::vector<Scalar>& coeffs, const Scalar& x) {
    // Horner's method
    Scalar result = Scalar::zero();
    for (int i = static_cast<int>(coeffs.size()) - 1; i >= 0; --i) {
        result = result * x + coeffs[static_cast<std::size_t>(i)];
    }
    return result;
}

// Compute binding factor for FROST signing
// ρ_i = H("FROST_binding", group_key || i || R_commitments || msg)
static Scalar compute_binding_factor(const Point& group_key,
                                       ParticipantId id,
                                       const std::vector<FrostNonceCommitment>& nonce_commitments,
                                       const std::array<std::uint8_t, 32>& msg) {
    SHA256 h;
    const char* tag = "FROST_binding";
    auto tag_hash = SHA256::hash(reinterpret_cast<const std::uint8_t*>(tag), std::strlen(tag));
    h.update(tag_hash.data(), 32);
    h.update(tag_hash.data(), 32);

    // Group public key (compressed)
    auto gpk = group_key.to_compressed();
    h.update(gpk.data(), 33);

    // Participant ID
    std::uint8_t id_be[4] = {
        std::uint8_t(id >> 24), std::uint8_t(id >> 16),
        std::uint8_t(id >> 8), std::uint8_t(id)
    };
    h.update(id_be, 4);

    // All nonce commitments (sorted by ID)
    for (const auto& nc : nonce_commitments) {
        auto d_comp = nc.hiding_point.to_compressed();
        auto e_comp = nc.binding_point.to_compressed();
        h.update(d_comp.data(), 33);
        h.update(e_comp.data(), 33);
    }

    // Message
    h.update(msg.data(), 32);

    auto digest = h.finalize();
    return Scalar::from_bytes(digest);
}

// Compute group commitment R = Σ(D_i + ρ_i * E_i)
static Point compute_group_commitment(
    const std::vector<FrostNonceCommitment>& nonce_commitments,
    const std::vector<Scalar>& binding_factors) {

    Point R = Point::infinity();
    for (std::size_t i = 0; i < nonce_commitments.size(); ++i) {
        Point rho_E = nonce_commitments[i].binding_point.scalar_mul(binding_factors[i]);
        Point contribution = nonce_commitments[i].hiding_point.add(rho_E);
        R = R.add(contribution);
    }
    return R;
}

// Compute challenge e = H("BIP0340/challenge", R.x || P.x || m)
static Scalar compute_challenge(const Point& R, const Point& group_key,
                                  const std::array<std::uint8_t, 32>& msg) {
    auto R_x = R.x().to_bytes();
    auto P_x = group_key.x().to_bytes();

    std::uint8_t challenge_data[96];
    std::memcpy(challenge_data, R_x.data(), 32);
    std::memcpy(challenge_data + 32, P_x.data(), 32);
    std::memcpy(challenge_data + 64, msg.data(), 32);
    auto e_hash = tagged_hash("BIP0340/challenge", challenge_data, 96);
    return Scalar::from_bytes(e_hash);
}

// ── Lagrange Coefficient ─────────────────────────────────────────────────────

Scalar frost_lagrange_coefficient(ParticipantId i,
                                   const std::vector<ParticipantId>& signer_ids) {
    Scalar num = Scalar::one();
    Scalar den = Scalar::one();

    Scalar x_i = Scalar::from_uint64(i);

    for (ParticipantId j : signer_ids) {
        if (j == i) continue;
        Scalar x_j = Scalar::from_uint64(j);
        num = num * x_j;                  // num *= j
        den = den * (x_j - x_i);          // den *= (j - i)
    }

    return num * den.inverse();
}

// ── DKG ──────────────────────────────────────────────────────────────────────

std::pair<FrostCommitment, std::vector<FrostShare>>
frost_keygen_begin(ParticipantId participant_id,
                   std::uint32_t threshold,
                   std::uint32_t num_participants,
                   const std::array<std::uint8_t, 32>& secret_seed) {
    // Generate random polynomial of degree (t-1):
    // f_i(x) = a_{i,0} + a_{i,1}*x + ... + a_{i,t-1}*x^{t-1}
    std::vector<Scalar> coeffs(threshold);
    for (std::uint32_t j = 0; j < threshold; ++j) {
        coeffs[j] = derive_scalar(secret_seed.data(), 32, "FROST_keygen_poly", 
                                   participant_id * 1000 + j);
    }

    // Commitment: A_{i,j} = a_{i,j} * G
    FrostCommitment commitment;
    commitment.from = participant_id;
    commitment.coeffs.resize(threshold);
    for (std::uint32_t j = 0; j < threshold; ++j) {
        commitment.coeffs[j] = Point::generator().scalar_mul(coeffs[j]);
    }

    // Shares: f_i(j) for j = 1..n
    std::vector<FrostShare> shares(num_participants);
    for (std::uint32_t j = 0; j < num_participants; ++j) {
        ParticipantId target = j + 1;
        Scalar x = Scalar::from_uint64(target);
        shares[j].from = participant_id;
        shares[j].id = target;
        shares[j].value = poly_eval(coeffs, x);
    }

    return {commitment, shares};
}

std::pair<FrostKeyPackage, bool>
frost_keygen_finalize(ParticipantId participant_id,
                      const std::vector<FrostCommitment>& commitments,
                      const std::vector<FrostShare>& received_shares,
                      std::uint32_t threshold,
                      std::uint32_t num_participants) {
    FrostKeyPackage pkg{};
    pkg.id = participant_id;
    pkg.threshold = threshold;
    pkg.num_participants = num_participants;

    // Verify each received share against its sender's commitment
    Scalar x_i = Scalar::from_uint64(participant_id);
    for (const auto& share : received_shares) {
        // Find commitment from the sender of this share
        const FrostCommitment* comm = nullptr;
        for (const auto& c : commitments) {
            if (c.from == share.from) {
                comm = &c;
                break;
            }
        }
        if (!comm) return {pkg, false};

        // Verify: share.value * G == Σ(A_{sender,j} * x_i^j)
        Point lhs = Point::generator().scalar_mul(share.value);
        Point rhs = Point::infinity();
        Scalar x_pow = Scalar::one();
        for (std::size_t j = 0; j < comm->coeffs.size(); ++j) {
            rhs = rhs.add(comm->coeffs[j].scalar_mul(x_pow));
            x_pow = x_pow * x_i;
        }

        auto lhs_c = lhs.to_compressed();
        auto rhs_c = rhs.to_compressed();
        if (lhs_c != rhs_c) return {pkg, false};
    }

    // Compute signing share: s_i = Σ f_j(i) for all j
    Scalar signing_share = Scalar::zero();
    for (const auto& share : received_shares) {
        signing_share = signing_share + share.value;
    }
    pkg.signing_share = signing_share;

    // Verification share: Y_i = s_i * G
    pkg.verification_share = Point::generator().scalar_mul(signing_share);

    // Group public key: Y = Σ A_{j,0} (sum of constant terms)
    Point group_key = Point::infinity();
    for (const auto& c : commitments) {
        if (!c.coeffs.empty()) {
            group_key = group_key.add(c.coeffs[0]);
        }
    }
    pkg.group_public_key = group_key;

    return {pkg, true};
}

// ── Signing ──────────────────────────────────────────────────────────────────

std::pair<FrostNonce, FrostNonceCommitment>
frost_sign_nonce_gen(ParticipantId participant_id,
                     const std::array<std::uint8_t, 32>& nonce_seed) {
    FrostNonce nonce;
    nonce.hiding_nonce = derive_scalar(nonce_seed.data(), 32, "FROST_nonce_hiding", participant_id);
    nonce.binding_nonce = derive_scalar(nonce_seed.data(), 32, "FROST_nonce_binding", participant_id);

    FrostNonceCommitment commitment;
    commitment.id = participant_id;
    commitment.hiding_point = Point::generator().scalar_mul(nonce.hiding_nonce);
    commitment.binding_point = Point::generator().scalar_mul(nonce.binding_nonce);

    return {nonce, commitment};
}

FrostPartialSig
frost_sign(const FrostKeyPackage& key_pkg,
           const FrostNonce& nonce,
           const std::array<std::uint8_t, 32>& msg,
           const std::vector<FrostNonceCommitment>& nonce_commitments) {
    // Collect signer IDs
    std::vector<ParticipantId> signer_ids;
    signer_ids.reserve(nonce_commitments.size());
    for (const auto& nc : nonce_commitments) {
        signer_ids.push_back(nc.id);
    }

    // Compute binding factors for all signers
    std::vector<Scalar> binding_factors;
    binding_factors.reserve(nonce_commitments.size());
    for (const auto& nc : nonce_commitments) {
        binding_factors.push_back(
            compute_binding_factor(key_pkg.group_public_key, nc.id,
                                   nonce_commitments, msg));
    }

    // My binding factor
    Scalar my_binding = Scalar::zero();
    for (std::size_t i = 0; i < nonce_commitments.size(); ++i) {
        if (nonce_commitments[i].id == key_pkg.id) {
            my_binding = binding_factors[i];
            break;
        }
    }

    // Group commitment R
    Point R = compute_group_commitment(nonce_commitments, binding_factors);

    // BIP-340 compatibility: negate nonces if R has odd y
    auto R_y = R.y().to_bytes();
    bool negate_R = (R_y[31] & 1) != 0;

    // Challenge e
    // For BIP-340 compat: use x-only group key
    auto gpk_y = key_pkg.group_public_key.y().to_bytes();
    bool negate_key = (gpk_y[31] & 1) != 0;

    Scalar e = compute_challenge(
        negate_R ? R.negate() : R,
        negate_key ? key_pkg.group_public_key.negate() : key_pkg.group_public_key,
        msg);

    // Lagrange coefficient
    Scalar lambda_i = frost_lagrange_coefficient(key_pkg.id, signer_ids);

    // Partial signature: z_i = d_i + ρ_i * e_i + λ_i * s_i * e
    Scalar d = nonce.hiding_nonce;
    Scalar ei = nonce.binding_nonce;
    if (negate_R) {
        d = d.negate();
        ei = ei.negate();
    }

    Scalar s_i = key_pkg.signing_share;
    if (negate_key) {
        s_i = s_i.negate();
    }

    Scalar z_i = d + (my_binding * ei) + (lambda_i * s_i * e);

    return FrostPartialSig{key_pkg.id, z_i};
}

bool frost_verify_partial(const FrostPartialSig& partial_sig,
                          const FrostNonceCommitment& signer_commitment,
                          const Point& verification_share,
                          const std::array<std::uint8_t, 32>& msg,
                          const std::vector<FrostNonceCommitment>& nonce_commitments,
                          const Point& group_public_key) {
    // Collect signer IDs
    std::vector<ParticipantId> signer_ids;
    for (const auto& nc : nonce_commitments) {
        signer_ids.push_back(nc.id);
    }

    // Compute binding factors
    std::vector<Scalar> binding_factors;
    for (const auto& nc : nonce_commitments) {
        binding_factors.push_back(
            compute_binding_factor(group_public_key, nc.id, nonce_commitments, msg));
    }

    // Find this signer's binding factor
    Scalar rho = Scalar::zero();
    for (std::size_t i = 0; i < nonce_commitments.size(); ++i) {
        if (nonce_commitments[i].id == partial_sig.id) {
            rho = binding_factors[i];
            break;
        }
    }

    // Group commitment
    Point R = compute_group_commitment(nonce_commitments, binding_factors);
    auto R_y = R.y().to_bytes();
    bool negate_R = (R_y[31] & 1) != 0;

    auto gpk_y = group_public_key.y().to_bytes();
    bool negate_key = (gpk_y[31] & 1) != 0;

    Scalar e = compute_challenge(
        negate_R ? R.negate() : R,
        negate_key ? group_public_key.negate() : group_public_key,
        msg);

    Scalar lambda_i = frost_lagrange_coefficient(partial_sig.id, signer_ids);

    // Verify: z_i * G == R_i + λ_i * e * Y_i
    // where R_i = D_i + ρ_i * E_i
    Point R_i = signer_commitment.hiding_point.add(
        signer_commitment.binding_point.scalar_mul(rho));
    if (negate_R) R_i = R_i.negate();

    Point lhs = Point::generator().scalar_mul(partial_sig.z_i);

    Point Y_i_eff = negate_key ? verification_share.negate() : verification_share;
    Point rhs = R_i.add(Y_i_eff.scalar_mul(lambda_i * e));

    auto lhs_c = lhs.to_compressed();
    auto rhs_c = rhs.to_compressed();
    return lhs_c == rhs_c;
}

SchnorrSignature
frost_aggregate(const std::vector<FrostPartialSig>& partial_sigs,
                const std::vector<FrostNonceCommitment>& nonce_commitments,
                const Point& group_public_key,
                const std::array<std::uint8_t, 32>& msg) {
    // Compute binding factors
    std::vector<Scalar> binding_factors;
    for (const auto& nc : nonce_commitments) {
        binding_factors.push_back(
            compute_binding_factor(group_public_key, nc.id, nonce_commitments, msg));
    }

    // Group commitment R
    Point R = compute_group_commitment(nonce_commitments, binding_factors);

    // BIP-340: ensure even y
    auto R_y = R.y().to_bytes();
    if (R_y[31] & 1) {
        R = R.negate();
    }

    // Aggregate: s = Σ z_i
    Scalar s = Scalar::zero();
    for (const auto& ps : partial_sigs) {
        s = s + ps.z_i;
    }

    SchnorrSignature sig;
    sig.r = R.x().to_bytes();
    sig.s = s;
    return sig;
}

} // namespace secp256k1
