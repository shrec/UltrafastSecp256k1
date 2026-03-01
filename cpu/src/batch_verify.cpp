// ============================================================================
// Batch Verification: ECDSA + Schnorr (BIP-340)
// ============================================================================
// Random linear combination technique for efficient batch verification.
// Falls back to individual verification to identify invalid signature(s).
// ============================================================================

#include "secp256k1/batch_verify.hpp"
#include "secp256k1/multiscalar.hpp"
#include "secp256k1/pippenger.hpp"
#include "secp256k1/sha256.hpp"
#include <cstring>

namespace secp256k1 {

using fast::Scalar;
using fast::Point;
using fast::FieldElement;

// -- Generate batch weight ----------------------------------------------------
// Deterministic weight derived from all signatures in the batch.
// This avoids requiring a CSPRNG while remaining sound.
// Weight a_i = SHA256("batch" || i || R1 || s1 || R2 || s2 || ...)
// For simplicity, we derive: a_i = SHA256(batch_seed || i_le32)

namespace {

// Generate deterministic weights for batch verification.
// batch_seed: SHA256 over all signature data (binds to entire batch).
// Returns a_i = SHA256(batch_seed || i) interpreted as scalar.
// The first weight a_0 = 1 (optimization: skip one scalar_mul).
Scalar batch_weight(const std::array<uint8_t, 32>& batch_seed, uint32_t index) {
    if (index == 0) return Scalar::one(); // optimization

    uint8_t buf[36];
    std::memcpy(buf, batch_seed.data(), 32);
    buf[32] = static_cast<uint8_t>(index & 0xFF);
    buf[33] = static_cast<uint8_t>((index >> 8) & 0xFF);
    buf[34] = static_cast<uint8_t>((index >> 16) & 0xFF);
    buf[35] = static_cast<uint8_t>((index >> 24) & 0xFF);

    auto h = SHA256::hash(buf, 36);
    return Scalar::from_bytes(h);
}

// Lift x-only key to point (same as in schnorr_verify)
// Returns (success, point)
std::pair<bool, Point> lift_x(const std::array<uint8_t, 32>& pubkey_x) {
    auto px_fe = FieldElement::from_bytes(pubkey_x);

    // y^2 = x^3 + 7
    auto x3 = px_fe.square() * px_fe;
    auto y2 = x3 + FieldElement::from_uint64(7);

    // Optimized sqrt via addition chain
    auto y = y2.sqrt();

    // Verify: y^2 == y2
    if (y.square() != y2) return {false, Point::infinity()};

    // Ensure even Y (BIP-340) -- direct limb check avoids to_bytes() overhead
    if (y.limbs()[0] & 1) {
        y = y.negate();
    }

    return {true, Point::from_affine(px_fe, y)};
}

} // anonymous namespace

// -- Schnorr Batch Verification -----------------------------------------------
// Equation: sum(a_i * s_i) * G = sum(a_i * R_i) + sum(a_i * e_i * P_i)
// Rearranged: sum(a_i * s_i) * G - sum(a_i * e_i) * P_i - sum(a_i) * R_i = O
// We verify: (sum(a_i * s_i)) * G + sum(-a_i * e_i * P_i) + sum(-a_i * R_i) = infinity

bool schnorr_batch_verify(const SchnorrBatchEntry* entries, std::size_t n) {
    if (n == 0) return true;
    if (n == 1) {
        return schnorr_verify(entries[0].pubkey_x, entries[0].message,
                              entries[0].signature);
    }

    // Compute batch seed = SHA256(all signature data)
    SHA256 seed_ctx;
    for (std::size_t i = 0; i < n; ++i) {
        seed_ctx.update(entries[i].signature.r.data(), 32);
        auto s_bytes = entries[i].signature.s.to_bytes();
        seed_ctx.update(s_bytes.data(), 32);
        seed_ctx.update(entries[i].pubkey_x.data(), 32);
        seed_ctx.update(entries[i].message.data(), 32);
    }
    auto batch_seed = seed_ctx.finalize();

    // Collect: scalars and points for multi_scalar_mul
    // Layout: [G_coeff, P_0, ..., P_{n-1}, R_0, ..., R_{n-1}]
    // Scalars: [sum(a_i*s_i), -a_0*e_0, ..., -a_{n-1}*e_{n-1}, -a_0, ..., -a_{n-1}]
    std::vector<Scalar> scalars;
    std::vector<Point> points;

    scalars.reserve(1 + 2 * n);
    points.reserve(1 + 2 * n);

    // G coefficient: sum(a_i * s_i)
    Scalar g_coeff = Scalar::zero();

    // First pass: compute challenges, lift points, accumulate G coefficient
    std::vector<Scalar> weights(n);
    std::vector<Point> pubkeys(n);
    std::vector<Point> R_points(n);
    std::vector<Scalar> challenges(n);

    for (std::size_t i = 0; i < n; ++i) {
        weights[i] = batch_weight(batch_seed, static_cast<uint32_t>(i));

        // Lift R from x-only
        auto [r_ok, R_pt] = lift_x(entries[i].signature.r);
        if (!r_ok) return false; // invalid R, batch fails
        R_points[i] = R_pt;

        // Lift pubkey from x-only
        auto [p_ok, P_pt] = lift_x(entries[i].pubkey_x);
        if (!p_ok) return false;
        pubkeys[i] = P_pt;

        // e_i = tagged_hash("BIP0340/challenge", R.x || pubkey_x || msg)
        uint8_t challenge_input[96];
        std::memcpy(challenge_input, entries[i].signature.r.data(), 32);
        std::memcpy(challenge_input + 32, entries[i].pubkey_x.data(), 32);
        std::memcpy(challenge_input + 64, entries[i].message.data(), 32);
        auto e_hash = tagged_hash("BIP0340/challenge", challenge_input, 96);
        challenges[i] = Scalar::from_bytes(e_hash);

        // Accumulate G coefficient: sum(a_i * s_i)
        g_coeff += weights[i] * entries[i].signature.s;
    }

    // Build multi-scalar arrays
    // First: G with coefficient sum(a_i*s_i)
    scalars.push_back(g_coeff);
    points.push_back(Point::generator());

    // Then: -a_i * e_i * P_i  for each signature
    for (std::size_t i = 0; i < n; ++i) {
        scalars.push_back((weights[i] * challenges[i]).negate());
        points.push_back(pubkeys[i]);
    }

    // Then: -a_i * R_i  for each signature
    for (std::size_t i = 0; i < n; ++i) {
        scalars.push_back(weights[i].negate());
        points.push_back(R_points[i]);
    }

    // Verify: MSM should yield infinity
    // msm() auto-selects Strauss (n<=128) or Pippenger (n>128)
    // For n=500 Schnorr batch -> 1001 points -> Pippenger is 10x+ faster
    auto result = msm(scalars, points);
    return result.is_infinity();
}

bool schnorr_batch_verify(const std::vector<SchnorrBatchEntry>& entries) {
    return schnorr_batch_verify(entries.data(), entries.size());
}

// -- ECDSA Batch Verification -------------------------------------------------
// For each sig (r_i, s_i), message z_i, pubkey Q_i:
//   w_i = s_i^{-1}
//   u1_i = z_i * w_i
//   u2_i = r_i * w_i
// Verify: sum(a_i * u1_i) * G + sum(a_i * u2_i * Q_i) has x == ... 
// 
// ECDSA is harder to batch because each verification checks x-coordinate equality.
// We use the Strauss-inspired approach:
// Random linear combination + individual x-coord check fallback.
// For true batch: verify that for each i,
//   (u1_i * G + u2_i * Q_i).x mod n == r_i
// We pre-compute all R'_i = u1_i*G + u2_i*Q_i using multi_scalar_mul tricks.

bool ecdsa_batch_verify(const ECDSABatchEntry* entries, std::size_t n) {
    if (n == 0) return true;
    if (n == 1) {
        return ecdsa_verify(entries[0].msg_hash, entries[0].public_key,
                            entries[0].signature);
    }

    // For ECDSA, we can't do a single multi-scalar check like Schnorr.
    // Instead, we batch the scalar multiplications but still check each x-coord.
    // Optimization: use Shamir's trick per-signature (already 2x faster than naive).

    // Pre-compute all s_inverse values
    // Batch inversion: compute all s^{-1} with Montgomery's trick
    std::vector<Scalar> s_inv(n);
    {
        // Montgomery batch inversion: 
        // prefixes[i] = s_0 * s_1 * ... * s_i
        std::vector<Scalar> prefixes(n);
        prefixes[0] = entries[0].signature.s;
        for (std::size_t i = 1; i < n; ++i) {
            prefixes[i] = prefixes[i - 1] * entries[i].signature.s;
        }
        // One inversion
        Scalar inv = prefixes[n - 1].inverse();
        // Back-propagate
        for (std::size_t i = n - 1; i > 0; --i) {
            s_inv[i] = prefixes[i - 1] * inv;
            inv = inv * entries[i].signature.s;
        }
        s_inv[0] = inv;
    }

    // ECDSA batch: per-signature Shamir trick + Montgomery batch inversion.
    //
    // Unlike Schnorr (where batch = single MSM -> infinity check), ECDSA
    // requires per-signature x-coordinate check: R'_i.x mod n == r_i.
    // True single-MSM batch would require lifting r_i to R_i (sqrt), but
    // standard ECDSA doesn't provide the y-parity (recovery flag), so
    // ~50% of attempts would pick wrong y and force fallback.
    //
    // Shamir's trick (simultaneous u1*G + u2*Q, joint wNAF scan) gives
    // ~2x speedup over naive separate muls. Combined with Montgomery
    // batch inversion above (1 modular inverse instead of n), this is
    // near-optimal for standard ECDSA without recovery parameter.

    for (std::size_t i = 0; i < n; ++i) {
        if (entries[i].signature.r.is_zero() || entries[i].signature.s.is_zero()) {
            return false;
        }

        auto z = Scalar::from_bytes(entries[i].msg_hash);
        auto u1 = z * s_inv[i];
        auto u2 = entries[i].signature.r * s_inv[i];

        // R' = u1*G + u2*Q via Shamir's trick (joint wNAF, ~2x individual)
        auto R_prime = shamir_trick(u1, Point::generator(),
                                    u2, entries[i].public_key);
        if (R_prime.is_infinity()) return false;

        auto v_bytes = R_prime.x().to_bytes();
        auto v = Scalar::from_bytes(v_bytes);
        if (v != entries[i].signature.r) return false;
    }

    return true;
}

bool ecdsa_batch_verify(const std::vector<ECDSABatchEntry>& entries) {
    return ecdsa_batch_verify(entries.data(), entries.size());
}

// -- Identify Invalid Signatures ----------------------------------------------

std::vector<std::size_t> schnorr_batch_identify_invalid(
    const SchnorrBatchEntry* entries, std::size_t n) {
    std::vector<std::size_t> invalid;
    for (std::size_t i = 0; i < n; ++i) {
        if (!schnorr_verify(entries[i].pubkey_x, entries[i].message,
                            entries[i].signature)) {
            invalid.push_back(i);
        }
    }
    return invalid;
}

std::vector<std::size_t> ecdsa_batch_identify_invalid(
    const ECDSABatchEntry* entries, std::size_t n) {
    std::vector<std::size_t> invalid;
    for (std::size_t i = 0; i < n; ++i) {
        if (!ecdsa_verify(entries[i].msg_hash, entries[i].public_key,
                          entries[i].signature)) {
            invalid.push_back(i);
        }
    }
    return invalid;
}

} // namespace secp256k1
