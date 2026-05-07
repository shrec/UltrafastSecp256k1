// ============================================================================
// FROST Threshold Signatures -- Implementation
// ============================================================================

#include "secp256k1/frost.hpp"
#include "secp256k1/sha256.hpp"
#include "secp256k1/schnorr.hpp"
#include "secp256k1/tagged_hash.hpp"
#include "secp256k1/ct/point.hpp"
#include "secp256k1/ct/scalar.hpp"
#include "secp256k1/detail/secure_erase.hpp"
#include <algorithm>
#include <cstring>

using secp256k1::detail::secure_erase;

namespace secp256k1 {

using fast::Point;
using fast::Scalar;
using detail::g_frost_binding_midstate;
using detail::g_challenge_midstate;
using detail::cached_tagged_hash;

// -- Internal Helpers ---------------------------------------------------------

// Deterministic scalar from seed + context
template<std::size_t N>
static Scalar derive_scalar(const std::uint8_t* seed, std::size_t seed_len,
                             const char (&context)[N], std::uint32_t index) {
    // BIP-340 tagged hash: SHA256(SHA256(tag) || SHA256(tag) || msg)
    // The double-tag prefix provides domain separation and prevents cross-
    // protocol hash collisions when the same seed is used in another context.
    SHA256 h;
    auto tag_hash = SHA256::hash(context, N - 1);
    h.update(tag_hash.data(), 32);
    h.update(tag_hash.data(), 32);
    h.update(seed, seed_len);
    // Index as big-endian 4 bytes
    std::uint8_t idx_be[4] = {
        std::uint8_t(index >> 24), std::uint8_t(index >> 16),
        std::uint8_t(index >> 8), std::uint8_t(index)
    };
    h.update(idx_be, 4);
    auto hash = h.finalize();
    auto result = Scalar::from_bytes(hash);
    secure_erase(hash.data(), hash.size());
    return result;
}

template<std::size_t N>
static Scalar derive_scalar_pair(const std::uint8_t* seed, std::size_t seed_len,
                                 const char (&context)[N],
                                 std::uint32_t participant_id,
                                 std::uint32_t coeff_index) {
    SHA256 h;
    auto tag_hash = SHA256::hash(context, N - 1);
    h.update(tag_hash.data(), 32);
    h.update(tag_hash.data(), 32);
    h.update(seed, seed_len);

    std::uint8_t idx_be[8] = {
        std::uint8_t(participant_id >> 24), std::uint8_t(participant_id >> 16),
        std::uint8_t(participant_id >> 8), std::uint8_t(participant_id),
        std::uint8_t(coeff_index >> 24), std::uint8_t(coeff_index >> 16),
        std::uint8_t(coeff_index >> 8), std::uint8_t(coeff_index)
    };
    h.update(idx_be, sizeof(idx_be));

    auto hash = h.finalize();
    auto result = Scalar::from_bytes(hash);
    secure_erase(hash.data(), hash.size());
    return result;
}

static bool valid_unique_participant_ids(const std::vector<ParticipantId>& ids) {
    if (ids.empty()) return false;
    for (std::size_t i = 0; i < ids.size(); ++i) {
        if (ids[i] == 0) return false;
        for (std::size_t j = i + 1; j < ids.size(); ++j) {
            if (ids[i] == ids[j]) return false;
        }
    }
    return true;
}

static bool valid_unique_nonce_commitment_ids(
    const std::vector<FrostNonceCommitment>& nonce_commitments) {
    if (nonce_commitments.empty()) return false;
    for (std::size_t i = 0; i < nonce_commitments.size(); ++i) {
        ParticipantId const id_i = nonce_commitments[i].id;
        if (id_i == 0) return false;
        for (std::size_t j = i + 1; j < nonce_commitments.size(); ++j) {
            if (id_i == nonce_commitments[j].id) return false;
        }
    }
    return true;
}

// Evaluate polynomial f(x) = a_0 + a_1*x + a_2*x^2 + ... at x
// CT: coeffs[0] is secret key material — all arithmetic must be constant-time.
static Scalar poly_eval(const std::vector<Scalar>& coeffs, const Scalar& x) {
    Scalar result = Scalar::zero();
    for (int i = static_cast<int>(coeffs.size()) - 1; i >= 0; --i) {
        result = ct::scalar_add(ct::scalar_mul(result, x),
                                coeffs[static_cast<std::size_t>(i)]);
    }
    return result;
}

// Serialized nonce commitment pair (precomputed to avoid O(n²) inversions)
struct FrostCommitmentSerialized {
    std::array<std::uint8_t, 33> hiding;
    std::array<std::uint8_t, 33> binding;
};

// Compute binding factor for FROST signing using pre-serialized commitments.
// rho_i = H("FROST_binding", group_key || i || R_commitments || msg)
// Pre-serialized: eliminates the O(n) to_compressed() calls per binding factor.
static Scalar compute_binding_factor_precomputed(
    const std::array<std::uint8_t, 33>& gpk_comp,
    ParticipantId id,
    const std::vector<FrostCommitmentSerialized>& serialized,
    const std::array<std::uint8_t, 32>& msg) {

    SHA256 h = g_frost_binding_midstate;
    h.update(gpk_comp.data(), 33);

    std::uint8_t id_be[4] = {
        std::uint8_t(id >> 24), std::uint8_t(id >> 16),
        std::uint8_t(id >> 8), std::uint8_t(id)
    };
    h.update(id_be, 4);

    for (const auto& s : serialized) {
        h.update(s.hiding.data(), 33);
        h.update(s.binding.data(), 33);
    }
    h.update(msg.data(), 32);
    return Scalar::from_bytes(h.finalize());
}

// Compute group commitment.
// Precomputes all to_compressed() calls once — reduces field inversions from
// O(n²) to O(n): 1 (group_key) + n (hiding) + n (binding) = 2n+1 total.
static Point compute_group_commitment_inline_binding(
    const Point& group_key,
    const std::vector<FrostNonceCommitment>& nonce_commitments,
    const std::array<std::uint8_t, 32>& msg,
    ParticipantId tracked_id,
    Scalar* tracked_binding) {

    // Precompute all serializations — O(n) inversions total
    auto gpk_comp = group_key.to_compressed();
    std::vector<FrostCommitmentSerialized> serialized;
    serialized.reserve(nonce_commitments.size());
    for (const auto& nc : nonce_commitments) {
        serialized.push_back({nc.hiding_point.to_compressed(),
                              nc.binding_point.to_compressed()});
    }

    Point R = Point::infinity();
    if (tracked_binding != nullptr) {
        *tracked_binding = Scalar::zero();
    }

    for (std::size_t i = 0; i < nonce_commitments.size(); ++i) {
        const auto& nc = nonce_commitments[i];
        Scalar const rho = compute_binding_factor_precomputed(
            gpk_comp, nc.id, serialized, msg);
        if (tracked_binding != nullptr && nc.id == tracked_id) {
            *tracked_binding = rho;
        }
        Point const rho_E = nc.binding_point.scalar_mul(rho);
        Point const contribution = nc.hiding_point.add(rho_E);
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
    auto e_hash = cached_tagged_hash(g_challenge_midstate, challenge_data, 96);
    return Scalar::from_bytes(e_hash);
}

// -- Lagrange Coefficient -----------------------------------------------------

Scalar frost_lagrange_coefficient(ParticipantId i,
                                   const std::vector<ParticipantId>& signer_ids) {
    if (i == 0 || !valid_unique_participant_ids(signer_ids) ||
        std::find(signer_ids.begin(), signer_ids.end(), i) == signer_ids.end()) {
        return Scalar::zero();
    }

    Scalar num = Scalar::one();
    Scalar den = Scalar::one();

    Scalar const x_i = Scalar::from_uint64(i);

    for (ParticipantId const j : signer_ids) {
        if (j == i) continue;
        Scalar const x_j = Scalar::from_uint64(j);
        num = num * x_j;                  // num *= j
        den = den * (x_j - x_i);          // den *= (j - i)
    }

    if (secp256k1::ct::scalar_is_zero(den)) return Scalar::zero();
    return secp256k1::ct::scalar_mul(num, secp256k1::ct::scalar_inverse(den));
}

static Scalar frost_lagrange_coefficient_from_commitments(
    ParticipantId i,
    const std::vector<FrostNonceCommitment>& nonce_commitments) {
    if (i == 0) {
        return Scalar::zero();
    }

    Scalar num = Scalar::one();
    Scalar den = Scalar::one();
    Scalar const x_i = Scalar::from_uint64(i);
    bool found_i = false;

    for (const auto& nc : nonce_commitments) {
        ParticipantId const j = nc.id;
        if (j == i) {
            found_i = true;
            continue;
        }
        Scalar const x_j = Scalar::from_uint64(j);
        num = num * x_j;
        den = den * (x_j - x_i);
    }

    if (!found_i) {
        return Scalar::zero();
    }
    if (secp256k1::ct::scalar_is_zero(den)) return Scalar::zero();
    return secp256k1::ct::scalar_mul(num, secp256k1::ct::scalar_inverse(den));
}

// -- DKG ----------------------------------------------------------------------

std::pair<FrostCommitment, std::vector<FrostShare>>
frost_keygen_begin(ParticipantId participant_id,
                   std::uint32_t threshold,
                   std::uint32_t num_participants,
                   const std::array<std::uint8_t, 32>& secret_seed) {
    // Validate inputs: participant IDs are 1-based; threshold must fit.
    if (participant_id == 0 || threshold == 0 ||
        num_participants == 0 || threshold > num_participants) {
        return {{}, {}};
    }

    // Generate random polynomial of degree (t-1):
    // f_i(x) = a_{i,0} + a_{i,1}*x + ... + a_{i,t-1}*x^{t-1}
    std::vector<Scalar> coeffs(threshold);
    for (std::uint32_t j = 0; j < threshold; ++j) {
        coeffs[j] = derive_scalar_pair(secret_seed.data(), 32, "FROST_keygen_poly",
                                       participant_id, j);
    }

    // Commitment: A_{i,j} = a_{i,j} * G
    FrostCommitment commitment;
    commitment.from = participant_id;
    commitment.coeffs.resize(threshold);
    for (std::uint32_t j = 0; j < threshold; ++j) {
        commitment.coeffs[j] = ct::generator_mul(coeffs[j]);
    }

    // Shares: f_i(j) for j = 1..n
    std::vector<FrostShare> shares(num_participants);
    for (std::uint32_t j = 0; j < num_participants; ++j) {
        ParticipantId const target = j + 1;
        Scalar const x = Scalar::from_uint64(target);
        shares[j].from = participant_id;
        shares[j].id = target;
        shares[j].value = poly_eval(coeffs, x);
    }

    // Erase secret polynomial coefficients from heap.
    for (auto& c : coeffs) secure_erase(&c, sizeof(c));

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

    if (participant_id == 0 || threshold == 0 || num_participants == 0 ||
        threshold > num_participants || commitments.size() != num_participants ||
        received_shares.size() != num_participants) {
        return {pkg, false};
    }

    for (std::size_t idx = 0; idx < commitments.size(); ++idx) {
        const auto& commitment = commitments[idx];
        if (commitment.from == 0 || commitment.coeffs.size() != threshold) {
            return {pkg, false};
        }
        for (std::size_t prev = 0; prev < idx; ++prev) {
            if (commitments[prev].from == commitment.from) {
                return {pkg, false};
            }
        }
    }

    for (std::size_t idx = 0; idx < received_shares.size(); ++idx) {
        const auto& share = received_shares[idx];
        if (share.from == 0 || share.id != participant_id) {
            return {pkg, false};
        }
        for (std::size_t prev = 0; prev < idx; ++prev) {
            if (received_shares[prev].from == share.from) {
                return {pkg, false};
            }
        }
    }

    // Verify each received share against its sender's commitment
    Scalar const x_i = Scalar::from_uint64(participant_id);
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

        // Verify: share.value * G == Sum(A_{sender,j} * x_i^j)
        // share.value is a secret polynomial evaluation — must use CT generator mul.
        Point const lhs = ct::generator_mul(share.value);
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

    // Compute signing share: s_i = Sum f_j(i) for all j — CT: secret scalar
    Scalar signing_share = Scalar::zero();
    for (const auto& share : received_shares) {
        signing_share = ct::scalar_add(signing_share, share.value);
    }
    pkg.signing_share = signing_share;

    // Verification share: Y_i = s_i * G
    pkg.verification_share = ct::generator_mul(signing_share);

    // Group public key: Y = Sum A_{j,0} (sum of constant terms)
    Point group_key = Point::infinity();
    for (const auto& c : commitments) {
        if (!c.coeffs.empty()) {
            group_key = group_key.add(c.coeffs[0]);
        }
    }
    pkg.group_public_key = group_key;

    return {pkg, true};
}

// -- Signing ------------------------------------------------------------------

std::pair<FrostNonce, FrostNonceCommitment>
frost_sign_nonce_gen(ParticipantId participant_id,
                     const std::array<std::uint8_t, 32>& nonce_seed) {
    FrostNonce nonce;
    nonce.hiding_nonce = derive_scalar(nonce_seed.data(), 32, "FROST_nonce_hiding", participant_id);
    nonce.binding_nonce = derive_scalar(nonce_seed.data(), 32, "FROST_nonce_binding", participant_id);

    FrostNonceCommitment commitment;
    commitment.id = participant_id;
    commitment.hiding_point = ct::generator_mul(nonce.hiding_nonce);
    commitment.binding_point = ct::generator_mul(nonce.binding_nonce);

    return {nonce, commitment};
}

FrostPartialSig
frost_sign(const FrostKeyPackage& key_pkg,
           FrostNonce& nonce,
           const std::array<std::uint8_t, 32>& msg,
           const std::vector<FrostNonceCommitment>& nonce_commitments) {
    if (!valid_unique_nonce_commitment_ids(nonce_commitments)) {
        secure_erase(&nonce.hiding_nonce, sizeof(nonce.hiding_nonce));
        secure_erase(&nonce.binding_nonce, sizeof(nonce.binding_nonce));
        return FrostPartialSig{key_pkg.id, Scalar::zero()};
    }

    // My binding factor while computing group commitment in one pass.
    Scalar my_binding = Scalar::zero();
    Point const R = compute_group_commitment_inline_binding(
        key_pkg.group_public_key,
        nonce_commitments,
        msg,
        key_pkg.id,
        &my_binding);

    // BIP-340 compatibility: negate nonces if R has odd y
    // NOTE: R and group_public_key are public values; VT field inverse is safe here.
    auto R_y = R.y().to_bytes();
    bool const negate_R = (R_y[31] & 1) != 0;

    // Challenge e
    // For BIP-340 compat: use x-only group key
    auto gpk_y = key_pkg.group_public_key.y().to_bytes();
    bool const negate_key = (gpk_y[31] & 1) != 0;

    Scalar const e = compute_challenge(
        negate_R ? R.negate() : R,
        negate_key ? key_pkg.group_public_key.negate() : key_pkg.group_public_key,
        msg);

    // Lagrange coefficient
    Scalar const lambda_i = frost_lagrange_coefficient_from_commitments(
        key_pkg.id, nonce_commitments);

    // Partial signature: z_i = d_i + rho_i * e_i + lambda_i * s_i * e
    // CT: d, ei, s_i are secret nonces/shares — use branchless ct::scalar_cneg.
    // A conditional branch (if negate_R) on the sign of a secret scalar leaks
    // key/nonce bits via timing even though negate_R itself is derived from a
    // public value (R.y). The compiler may emit a conditional branch for the
    // negate() call; ct::scalar_cneg avoids that with a bitmask select.
    auto const negate_R_mask   = ct::bool_to_mask(negate_R);
    auto const negate_key_mask = ct::bool_to_mask(negate_key);
    Scalar d  = ct::scalar_cneg(nonce.hiding_nonce,  negate_R_mask);
    Scalar ei = ct::scalar_cneg(nonce.binding_nonce, negate_R_mask);
    Scalar s_i = ct::scalar_cneg(key_pkg.signing_share, negate_key_mask);

    // CT: d, ei, s_i are secret nonces/shares — use branchless CT arithmetic
    Scalar const rho_ei      = ct::scalar_mul(my_binding, ei);
    Scalar const lambda_s_e  = ct::scalar_mul(ct::scalar_mul(lambda_i, s_i), e);
    Scalar const z_i         = ct::scalar_add(ct::scalar_add(d, rho_ei), lambda_s_e);

    // Erase secret nonces and signing share from stack, then consume the
    // caller's nonce to enforce single-use (H-01 nonce-reuse prevention).
    secure_erase(&d,   sizeof(d));
    secure_erase(&ei,  sizeof(ei));
    secure_erase(&s_i, sizeof(s_i));
    secure_erase(&nonce.hiding_nonce,  sizeof(nonce.hiding_nonce));
    secure_erase(&nonce.binding_nonce, sizeof(nonce.binding_nonce));

    return FrostPartialSig{key_pkg.id, z_i};
}

bool frost_verify_partial(const FrostPartialSig& partial_sig,
                          const FrostNonceCommitment& signer_commitment,
                          const Point& verification_share,
                          const std::array<std::uint8_t, 32>& msg,
                          const std::vector<FrostNonceCommitment>& nonce_commitments,
                          const Point& group_public_key) {
    if (!valid_unique_nonce_commitment_ids(nonce_commitments) || partial_sig.id == 0) {
        return false;
    }

    // Find this signer's binding factor while computing group commitment.
    Scalar rho = Scalar::zero();
    Point const R = compute_group_commitment_inline_binding(
        group_public_key,
        nonce_commitments,
        msg,
        partial_sig.id,
        &rho);

    // Group commitment
    // NOTE: R and group_public_key are public values; VT field inverse is safe here.
    auto R_y = R.y().to_bytes();
    bool const negate_R = (R_y[31] & 1) != 0;

    auto gpk_y = group_public_key.y().to_bytes();
    bool const negate_key = (gpk_y[31] & 1) != 0;

    Scalar const e = compute_challenge(
        negate_R ? R.negate() : R,
        negate_key ? group_public_key.negate() : group_public_key,
        msg);

    Scalar const lambda_i = frost_lagrange_coefficient_from_commitments(
        partial_sig.id, nonce_commitments);

    // Verify: z_i * G == R_i + lambda_i * e * Y_i
    // where R_i = D_i + rho_i * E_i
    Point R_i = signer_commitment.hiding_point.add(
        signer_commitment.binding_point.scalar_mul(rho));
    if (negate_R) R_i = R_i.negate();

    Point const lhs = Point::generator().scalar_mul(partial_sig.z_i);

    Point const Y_i_eff = negate_key ? verification_share.negate() : verification_share;
    Point const rhs = R_i.add(Y_i_eff.scalar_mul(lambda_i * e));

    auto lhs_c = lhs.to_compressed();
    auto rhs_c = rhs.to_compressed();
    return lhs_c == rhs_c;
}

SchnorrSignature
frost_aggregate(const std::vector<FrostPartialSig>& partial_sigs,
                const std::vector<FrostNonceCommitment>& nonce_commitments,
                const Point& group_public_key,
                const std::array<std::uint8_t, 32>& msg) {
    if (!valid_unique_nonce_commitment_ids(nonce_commitments)) {
        return SchnorrSignature{{}, Scalar::zero()};
    }

    // Group commitment R (compute binding factors inline to avoid heap scratch).
    Point R = compute_group_commitment_inline_binding(
        group_public_key,
        nonce_commitments,
        msg,
        0,
        nullptr);

    // BIP-340: ensure even y
    // NOTE: R is a public group commitment; VT field inverse is safe here.
    auto R_y = R.y().to_bytes();
    if (R_y[31] & 1) {
        R = R.negate();
    }

    // Aggregate: s = Sum z_i
    Scalar s = Scalar::zero();
    for (const auto& ps : partial_sigs) {
        s = ct::scalar_add(s, ps.z_i);
    }

    // Fail-closed: degenerate aggregated signature is a protocol failure
    if (s.is_zero_ct() || R.is_infinity()) {
        return SchnorrSignature{{}, Scalar::zero()};
    }

    SchnorrSignature sig;
    sig.r = R.x().to_bytes();
    sig.s = s;
    return sig;
}

} // namespace secp256k1
