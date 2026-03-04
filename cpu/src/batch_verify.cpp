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
#include "secp256k1/tagged_hash.hpp"
#if defined(__SIZEOF_INT128__) && !defined(SECP256K1_PLATFORM_ESP32) && !defined(SECP256K1_PLATFORM_STM32) && !defined(__EMSCRIPTEN__)
#include "secp256k1/field_52.hpp"
#endif
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
#if defined(SECP256K1_FAST_52BIT)
    using FE52 = fast::FieldElement52;
    // Direct bytes->FE52: avoids FieldElement construction overhead
    FE52 const px52 = FE52::from_bytes(pubkey_x.data());

    // y^2 = x^3 + 7
    FE52 const x3 = px52.square() * px52;
    static const FE52 seven52 = FE52::from_fe(FieldElement::from_uint64(7));
    FE52 const y2 = x3 + seven52;

    // sqrt via FE52 addition chain: a^((p+1)/4), ~253 sqr + 13 mul
    FE52 y52 = y2.sqrt();

    // Verify: y^2 == y2 (check that sqrt succeeded)
    FE52 check = y52.square();
    check.normalize();
    FE52 y2n = y2;
    y2n.normalize();
    if (!(check == y2n)) return {false, Point::infinity()};

    // Ensure even Y (BIP-340 convention): check parity of normalized y
    FE52 y_norm = y52;
    y_norm.normalize();
    if (y_norm.n[0] & 1) {
        y52 = y52.negate(1);
        y52.normalize_weak();
    }

    return {true, Point::from_affine52(px52, y52)};
#else
    // Fallback: 4x64 lift_x
    FieldElement px_fe;
    if (!FieldElement::parse_bytes_strict(pubkey_x, px_fe))
        return {false, Point::infinity()};

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
#endif
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

    // ---- Small-batch fast path: individual verification ----
    // For small N, individual schnorr_verify uses the highly-optimized
    // 4-stream GLV Strauss with precomputed generator tables (~30us/sig),
    // which outperforms the generic 2N-point MSM that lacks generator
    // tables and GLV.  MSM becomes profitable only for large N (>16)
    // where amortized doublings offset the per-point overhead.
    if (n <= 16) {
        for (std::size_t i = 0; i < n; ++i) {
            if (!schnorr_verify(entries[i].pubkey_x, entries[i].message,
                                entries[i].signature)) {
                return false;
            }
        }
        return true;
    }

    // ---- Large-batch path: randomized MSM ----
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

    // Collect: scalars and points for multi_scalar_mul (non-generator only)
    // Layout: [P_0, ..., P_{n-1}, R_0, ..., R_{n-1}]
    // Scalars: [-a_0*e_0, ..., -a_{n-1}*e_{n-1}, -a_0, ..., -a_{n-1}]
    std::vector<Scalar> scalars;
    std::vector<Point> points;

    scalars.reserve(2 * n);
    points.reserve(2 * n);

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
        // Uses precomputed midstate: avoids re-hashing tag on every iteration
        // (saves 2 SHA256 compress calls per signature vs generic tagged_hash)
        uint8_t challenge_input[96];
        std::memcpy(challenge_input, entries[i].signature.r.data(), 32);
        std::memcpy(challenge_input + 32, entries[i].pubkey_x.data(), 32);
        std::memcpy(challenge_input + 64, entries[i].message.data(), 32);
        auto e_hash = detail::cached_tagged_hash(
            detail::g_challenge_midstate, challenge_input, 96);
        challenges[i] = Scalar::from_bytes(e_hash);

        // Accumulate G coefficient: sum(a_i * s_i)
        g_coeff += weights[i] * entries[i].signature.s;
    }

    // ---- Optimization: separate G coefficient from MSM ----
    // G has a precomputed comb table (~6us via scalar_mul), while generic
    // points cost ~25us each in MSM.  By computing g_coeff*G separately
    // we avoid treating the generator as a generic point, gaining ~19us
    // and keeping the remaining 2*n points in the MSM range for Strauss.

    // Step 1: g_coeff * G  (uses precomputed comb table -- fast path)
    auto G_term = Point::generator().scalar_mul(g_coeff);

    // Step 2: Build MSM for non-generator points only (2*n points)
    //   scalars: [-a_0*e_0, ..., -a_{n-1}*e_{n-1}, -a_0, ..., -a_{n-1}]
    //   points:  [P_0, ..., P_{n-1}, R_0, ..., R_{n-1}]
    for (std::size_t i = 0; i < n; ++i) {
        scalars.push_back((weights[i] * challenges[i]).negate());
        points.push_back(pubkeys[i]);
    }

    for (std::size_t i = 0; i < n; ++i) {
        scalars.push_back(weights[i].negate());
        points.push_back(R_points[i]);
    }

    // Step 3: Compute MSM for P_i and R_i terms
    // msm() auto-selects Strauss (n<=128) or Pippenger (n>128)
    auto rest = msm(scalars, points);

    // Step 4: Verify g_coeff*G + rest = infinity
    auto result = G_term.add(rest);
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

    // ECDSA batch: dual_scalar_mul_gen_point + Montgomery batch inversion.
    //
    // Unlike Schnorr (where batch = single MSM -> infinity check), ECDSA
    // requires per-signature x-coordinate check: R'_i.x mod n == r_i.
    // True single-MSM batch would require lifting r_i to R_i (sqrt), but
    // standard ECDSA doesn't provide the y-parity (recovery flag), so
    // ~50% of attempts would pick wrong y and force fallback.
    //
    // dual_scalar_mul_gen_point uses 4-stream GLV Strauss with precomputed
    // generator tables (shared doublings, affine-mixed adds, W_G=15).
    // Combined with Montgomery batch inversion (1 inverse instead of n),
    // this is near-optimal for standard ECDSA without recovery parameter.

    for (std::size_t i = 0; i < n; ++i) {
        if (entries[i].signature.r.is_zero() || entries[i].signature.s.is_zero()) {
            return false;
        }

        auto z = Scalar::from_bytes(entries[i].msg_hash);
        auto u1 = z * s_inv[i];
        auto u2 = entries[i].signature.r * s_inv[i];

        // R' = u1*G + u2*Q via 4-stream GLV Strauss (precomp G tables, ~27us)
        auto R_prime = Point::dual_scalar_mul_gen_point(u1, u2,
                                                        entries[i].public_key);
        if (R_prime.is_infinity()) return false;

        // Z^2-based x-coordinate check (avoids field inverse ~940ns).
        // Check: R'.x / R'.z^2 mod n == sig.r
        // Equivalent: sig.r * R'.z^2 == R'.x (mod p)
#if defined(SECP256K1_FAST_52BIT)
        using FE52 = fast::FieldElement52;
        FE52 const r52 = FE52::from_4x64_limbs(entries[i].signature.r.limbs().data());
        FE52 const z2 = R_prime.Z52().square();
        FE52 const r_z2 = r52 * z2;

        FE52 diff = R_prime.X52();
        diff.negate_assign(23);
        diff.add_assign(r_z2);
        if (!diff.normalizes_to_zero_var()) {
            // Rare case: x_R mod p in [n, p). Probability ~2^-128.
            static constexpr std::uint64_t PMN_0 = 0x402da1732fc9bebfULL;
            static constexpr std::uint64_t PMN_1 = 0x14551231950b75fcULL;
            const auto& rl = entries[i].signature.r.limbs();
            bool r_less_than_pmn = (rl[3] == 0 && rl[2] == 0);
            if (r_less_than_pmn) {
                if (rl[1] != PMN_1) r_less_than_pmn = (rl[1] < PMN_1);
                else r_less_than_pmn = (rl[0] < PMN_0);
            }
            if (!r_less_than_pmn) return false;

            static constexpr std::uint64_t N_LIMBS[4] = {
                0xBFD25E8CD0364141ULL, 0xBAAEDCE6AF48A03BULL,
                0xFFFFFFFFFFFFFFFEULL, 0xFFFFFFFFFFFFFFFFULL
            };
            alignas(32) std::uint64_t rn[4];
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
            unsigned __int128 acc = static_cast<unsigned __int128>(rl[0]) + N_LIMBS[0];
            rn[0] = static_cast<std::uint64_t>(acc);
            acc = static_cast<unsigned __int128>(rl[1]) + N_LIMBS[1] + static_cast<std::uint64_t>(acc >> 64);
            rn[1] = static_cast<std::uint64_t>(acc);
            acc = static_cast<unsigned __int128>(rl[2]) + N_LIMBS[2] + static_cast<std::uint64_t>(acc >> 64);
            rn[2] = static_cast<std::uint64_t>(acc);
            rn[3] = rl[3] + N_LIMBS[3] + static_cast<std::uint64_t>(acc >> 64);
#pragma GCC diagnostic pop

            FE52 const r2_z2 = FE52::from_4x64_limbs(rn) * z2;
            FE52 diff2 = R_prime.X52();
            diff2.negate_assign(23);
            diff2.add_assign(r2_z2);
            if (!diff2.normalizes_to_zero_var()) return false;
        }
#else
        auto v_bytes = R_prime.x().to_bytes();
        auto v = Scalar::from_bytes(v_bytes);
        if (v != entries[i].signature.r) return false;
#endif
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
