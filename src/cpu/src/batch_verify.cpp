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
#include "secp256k1/ct/point.hpp"
#if defined(__SIZEOF_INT128__) && !defined(SECP256K1_PLATFORM_ESP32) && !defined(SECP256K1_PLATFORM_STM32) && !defined(__EMSCRIPTEN__)
#include "secp256k1/field_52.hpp"
#endif
#include <cstring>
#include <unordered_map>
#include <array>

namespace secp256k1 {

// Hash for 32-byte pubkey key in unordered_map (B-6)
struct PubkeyHash32 {
    std::size_t operator()(const std::array<uint8_t, 32>& k) const noexcept {
        // FNV-1a over the 32 bytes — cheap and collision-resistant enough for keys
        std::size_t h = 0xcbf29ce484222325ULL;
        for (unsigned char b : k) {
            h ^= b;
            h *= 0x100000001b3ULL;
        }
        return h;
    }
};

using fast::Scalar;
using fast::Point;
using fast::FieldElement;

// -- Generate batch weight ----------------------------------------------------
// Deterministic weight derived from all signatures in the batch.
// This avoids requiring a CSPRNG while remaining sound.
// Weight a_i = SHA256("batch" || i || R1 || s1 || R2 || s2 || ...)
// For simplicity, we derive: a_i = SHA256(batch_seed || i_le32)

namespace {

// Crossover (x86-64 FE52, bench_unified POOL=64 repeated entries):
// MSM path uses Pippenger(2N pts) + lift_x_cached (thread-local, hits on
// repeated pubkeys) and wins for N>=128 in bench_unified.
// lift_x_cached inside schnorr_verify(raw) is faster than a fresh pubkey_cache
// linear scan in the individual path when data is warm across bench passes.
// Empirically measured optimal cutoff: 96 (individual wins below this).
constexpr std::size_t kSchnorrBatchIndividualCutoff = 96;

// Generate deterministic weights for batch verification.
// batch_seed: SHA256 over all signature data (binds to entire batch).
// Returns a_i = SHA256(batch_seed || i) interpreted as scalar.
// The first weight a_0 = 1 (optimization: skip one scalar_mul).
Scalar batch_weight(const SHA256& batch_weight_base, uint32_t index) {
    if (index == 0) return Scalar::one(); // optimization

    uint8_t index_bytes[4];
    index_bytes[0] = static_cast<uint8_t>(index & 0xFF);
    index_bytes[1] = static_cast<uint8_t>((index >> 8) & 0xFF);
    index_bytes[2] = static_cast<uint8_t>((index >> 16) & 0xFF);
    index_bytes[3] = static_cast<uint8_t>((index >> 24) & 0xFF);

    SHA256 ctx = batch_weight_base;
    ctx.update(index_bytes, sizeof(index_bytes));
    auto h = ctx.finalize();
    return Scalar::from_bytes(h);
}

struct SchnorrBatchScratch {
    std::vector<Scalar> scalars;
    std::vector<Point> points;
};

SchnorrBatchScratch& schnorr_batch_scratch(std::size_t n) {
    static thread_local SchnorrBatchScratch scratch;
    if (scratch.scalars.size() < n) {
        scratch.scalars.resize(n);
    }
    if (scratch.points.size() < n) {
        scratch.points.resize(n);
    }
    return scratch;
}

// Lift x-only coordinate to Point.
// Delegates to schnorr_xonly_pubkey_parse so results are shared with the
// thread-local lift_x_cached table in schnorr.cpp.  On warm bench passes
// (e.g. repeated validation of the same signatures), the cache turns each
// sqrt addition-chain (~2800 ns) into a cheap table lookup (~5 ns).
// sig.r bytes are public, so caching them alongside pubkeys is safe.
std::pair<bool, Point> lift_x(const std::array<uint8_t, 32>& pubkey_x) {
    SchnorrXonlyPubkey parsed;
    if (!schnorr_xonly_pubkey_parse(parsed, pubkey_x)) {
        return {false, Point::infinity()};
    }
    return {true, std::move(parsed.point)};
}

template <typename Entry, typename VerifyOneFn, typename ResolvePubkeyFn,
          typename PubkeyBytesFn>
bool schnorr_batch_verify_impl(const Entry* entries, std::size_t n,
                               VerifyOneFn&& verify_one,
                               ResolvePubkeyFn&& resolve_pubkey,
                               PubkeyBytesFn&& pubkey_bytes) {
    if (n == 0) return true;
    if (n == 1) return verify_one(entries[0]);

    // ---- Small-batch fast path: individual verification ----
    // For small N, individual schnorr_verify uses the highly-optimized
    // 4-stream GLV Strauss with precomputed generator tables (~20us/sig)
    // still wins through moderate batch sizes on this CPU. Past that point,
    // the randomized MSM path becomes competitive enough to justify the extra
    // setup work.
    if (n <= kSchnorrBatchIndividualCutoff) {
        for (std::size_t i = 0; i < n; ++i) {
            if (!verify_one(entries[i])) {
                return false;
            }
        }
        return true;
    }

    // ---- Large-batch path: randomized MSM ----
    // Compute batch seed = SHA256(all signature data)
    SHA256 seed_ctx;
    std::uint8_t s_bytes[32];
    for (std::size_t i = 0; i < n; ++i) {
        auto const* const pubkey_x = pubkey_bytes(entries[i]);
        if (pubkey_x == nullptr) return false;

        seed_ctx.update(entries[i].signature.r.data(), 32);
        entries[i].signature.s.write_bytes(s_bytes);
        seed_ctx.update(s_bytes, 32);
        seed_ctx.update(pubkey_x->data(), 32);
        seed_ctx.update(entries[i].message.data(), 32);
    }
    auto batch_seed = seed_ctx.finalize();
    SHA256 batch_weight_base;
    batch_weight_base.update(batch_seed.data(), batch_seed.size());

    std::size_t const msm_n = 2 * n;
    auto& scratch = schnorr_batch_scratch(msm_n);
    Scalar* const scalars = scratch.scalars.data();
    Point* const points = scratch.points.data();

    Scalar g_coeff = Scalar::zero();

    for (std::size_t i = 0; i < n; ++i) {
        Scalar const weight = batch_weight(batch_weight_base,
                                           static_cast<uint32_t>(i));

        auto [r_ok, R_pt] = lift_x(entries[i].signature.r);
        if (!r_ok) return false;

        Point P_pt = Point::infinity();
        if (!resolve_pubkey(entries[i], P_pt)) return false;

        auto const* const pubkey_x = pubkey_bytes(entries[i]);
        if (pubkey_x == nullptr) return false;

        SHA256 challenge_ctx = detail::g_challenge_midstate;
        challenge_ctx.update(entries[i].signature.r.data(), 32);
        challenge_ctx.update(pubkey_x->data(), 32);
        challenge_ctx.update(entries[i].message.data(), 32);
        auto e_hash = challenge_ctx.finalize();
        Scalar const challenge = Scalar::from_bytes(e_hash);

        g_coeff += weight * entries[i].signature.s;

        scalars[i] = (weight * challenge).negate();
        points[i] = P_pt;
        scalars[n + i] = weight.negate();
        points[n + i] = R_pt;
    }

    auto G_term = ct::generator_mul(g_coeff);
    auto rest = msm(scalars, points, msm_n);
    auto result = G_term.add(rest);
    return result.is_infinity();
}

template <typename Entry, typename VerifyOneFn>
void schnorr_batch_identify_invalid_impl(
    const Entry* entries, std::size_t n,
    std::vector<std::size_t>& invalid,
    VerifyOneFn&& verify_one) {
    invalid.clear();
    invalid.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        if (!verify_one(entries[i])) {
            invalid.push_back(i);
        }
    }
}

} // anonymous namespace

// -- Schnorr Batch Verification -----------------------------------------------
// Equation: sum(a_i * s_i) * G = sum(a_i * R_i) + sum(a_i * e_i * P_i)
// Rearranged: sum(a_i * s_i) * G - sum(a_i * e_i) * P_i - sum(a_i) * R_i = O
// We verify: (sum(a_i * s_i)) * G + sum(-a_i * e_i * P_i) + sum(-a_i * R_i) = infinity

bool schnorr_batch_verify(const SchnorrBatchEntry* entries, std::size_t n) {
    // B-6: hash map for O(1) dedup instead of O(k) linear scan
    using PubkeyMap = std::unordered_map<std::array<uint8_t, 32>, Point, PubkeyHash32>;
    static thread_local PubkeyMap pubkey_index;
    pubkey_index.clear();
    // Shrink if bucket table grew far beyond current n to prevent indefinite memory growth
    // under adversarial large-then-small call patterns (clear() keeps bucket memory).
    if (pubkey_index.bucket_count() > n * 8 + 64) pubkey_index = PubkeyMap{};
    pubkey_index.reserve(n);

    // Grow-only vector reserve (no realloc when n stays the same between calls)
    static thread_local std::vector<SchnorrXonlyPubkey> pubkey_cache;
    pubkey_cache.clear();
    if (pubkey_cache.capacity() < n) pubkey_cache.reserve(n);

    auto verify_one = [](const SchnorrBatchEntry& entry) {
        return schnorr_verify(entry.pubkey_x, entry.message, entry.signature);
    };
    auto resolve_pubkey = [](const SchnorrBatchEntry& entry,
                             Point& out_point) {
        // O(1) hash map lookup instead of O(k) linear scan
        auto it = pubkey_index.find(entry.pubkey_x);
        if (it != pubkey_index.end()) {
            out_point = it->second;
            return true;
        }

        SchnorrXonlyPubkey parsed;
        if (!schnorr_xonly_pubkey_parse(parsed, entry.pubkey_x)) {
            return false;
        }
        out_point = parsed.point;
        pubkey_cache.push_back(parsed);
        pubkey_index.emplace(entry.pubkey_x, parsed.point);
        return true;
    };
    auto pubkey_bytes = [](const SchnorrBatchEntry& entry)
        -> const std::array<uint8_t, 32>* {
        return &entry.pubkey_x;
    };

    return schnorr_batch_verify_impl(entries, n, verify_one, resolve_pubkey,
                                     pubkey_bytes);
}

bool schnorr_batch_verify(const std::vector<SchnorrBatchEntry>& entries) {
    return schnorr_batch_verify(entries.data(), entries.size());
}

bool schnorr_batch_verify(const SchnorrBatchCachedEntry* entries, std::size_t n) {
    auto verify_one = [](const SchnorrBatchCachedEntry& entry) {
        return entry.pubkey != nullptr &&
               schnorr_verify(*entry.pubkey, entry.message, entry.signature);
    };
    auto resolve_pubkey = [](const SchnorrBatchCachedEntry& entry,
                             Point& out_point) {
        if (entry.pubkey == nullptr) {
            return false;
        }
        out_point = entry.pubkey->point;
        return true;
    };
    auto pubkey_bytes = [](const SchnorrBatchCachedEntry& entry)
        -> const std::array<uint8_t, 32>* {
        return (entry.pubkey == nullptr) ? nullptr : &entry.pubkey->x_bytes;
    };

    return schnorr_batch_verify_impl(entries, n, verify_one, resolve_pubkey,
                                     pubkey_bytes);
}

bool schnorr_batch_verify(const std::vector<SchnorrBatchCachedEntry>& entries) {
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

    // Pre-validate all entries before any further processing to enforce
    // BIP-62 low-S: reject any s > n/2 and reject zero r/s.
    // This must run before the n==1 shortcut to maintain consistent policy
    // with the single ecdsa_verify path.
    for (std::size_t i = 0; i < n; ++i) {
        if (entries[i].signature.r.is_zero() || entries[i].signature.s.is_zero()) {
            return false;
        }
        if (!entries[i].signature.is_low_s()) {
            return false;
        }
    }

    // Small-n fast path: individual verifies cheaper than batch inversion overhead
    constexpr std::size_t kEcdsaBatchIndividualCutoff = 8;
    if (n <= kEcdsaBatchIndividualCutoff) {
        for (std::size_t i = 0; i < n; ++i) {
            if (!ecdsa_verify(entries[i].msg_hash, entries[i].public_key,
                              entries[i].signature))
                return false;
        }
        return true;
    }

    // Pre-compute all s_inverse values
    // Batch inversion: compute all s^{-1} with Montgomery's trick
    static thread_local std::vector<Scalar> s_inv;
    s_inv.resize(n);
    // Montgomery batch inversion using s_inv as the prefix-product arena.
    s_inv[0] = entries[0].signature.s;
    for (std::size_t i = 1; i < n; ++i) {
        s_inv[i] = s_inv[i - 1] * entries[i].signature.s;
    }
    Scalar inv = s_inv[n - 1].inverse();
    for (std::size_t i = n - 1; i > 0; --i) {
        Scalar const prefix = s_inv[i - 1];
        s_inv[i] = prefix * inv;
        inv = inv * entries[i].signature.s;
    }
    s_inv[0] = inv;

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

void schnorr_batch_identify_invalid(
    const SchnorrBatchEntry* entries, std::size_t n,
    std::vector<std::size_t>& invalid_out) {
    schnorr_batch_identify_invalid_impl(
        entries, n, invalid_out, [](const SchnorrBatchEntry& entry) {
            return schnorr_verify(entry.pubkey_x, entry.message,
                                  entry.signature);
        });
}

std::vector<std::size_t> schnorr_batch_identify_invalid(
    const SchnorrBatchEntry* entries, std::size_t n) {
    std::vector<std::size_t> invalid;
    schnorr_batch_identify_invalid(entries, n, invalid);
    return invalid;
}

void schnorr_batch_identify_invalid(
    const SchnorrBatchCachedEntry* entries, std::size_t n,
    std::vector<std::size_t>& invalid_out) {
    schnorr_batch_identify_invalid_impl(
        entries, n, invalid_out, [](const SchnorrBatchCachedEntry& entry) {
            return entry.pubkey != nullptr &&
                   schnorr_verify(*entry.pubkey, entry.message,
                                  entry.signature);
        });
}

std::vector<std::size_t> schnorr_batch_identify_invalid(
    const SchnorrBatchCachedEntry* entries, std::size_t n) {
    std::vector<std::size_t> invalid;
    schnorr_batch_identify_invalid(entries, n, invalid);
    return invalid;
}

void ecdsa_batch_identify_invalid(
    const ECDSABatchEntry* entries, std::size_t n,
    std::vector<std::size_t>& invalid_out) {
    invalid_out.clear();
    invalid_out.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        if (!ecdsa_verify(entries[i].msg_hash, entries[i].public_key,
                          entries[i].signature)) {
            invalid_out.push_back(i);
        }
    }
}

std::vector<std::size_t> ecdsa_batch_identify_invalid(
    const ECDSABatchEntry* entries, std::size_t n) {
    std::vector<std::size_t> invalid;
    ecdsa_batch_identify_invalid(entries, n, invalid);
    return invalid;
}

} // namespace secp256k1
