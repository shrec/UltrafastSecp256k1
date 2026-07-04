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
#include "secp256k1/detail/csprng.hpp"
#include "secp256k1/detail/secure_erase.hpp"
#include "secp256k1/detail/batch_pool.hpp"
#if defined(__SIZEOF_INT128__) && !defined(SECP256K1_PLATFORM_ESP32) && !defined(SECP256K1_PLATFORM_STM32) && !defined(__EMSCRIPTEN__)
#include "secp256k1/field_52.hpp"
#endif
#include <cstring>
#include <unordered_map>
#include <array>
#include <random>
#include <algorithm>
#include <atomic>
#include <limits>
#include <thread>
#include <vector>

namespace secp256k1 {

// -- GPU column-verify accelerator hook ---------------------------------------
// Translation-unit-local atomic hook plus its installer. Null by default (pure
// CPU); the two *_batch_verify_*_columns entrypoints consult it internally.
//
// The engine (fastsecp256k1) deliberately carries NO gpu:: dependency: the
// default provider that bridges this hook to a real GPU backend lives in the
// GPU-host layer (src/gpu/src/gpu_engine_hook.cpp) and SELF-INSTALLS via a static
// initializer whenever that translation unit is linked (i.e. whenever the GPU
// host is built). This avoids the secp256k1_gpu_host -> fastsecp256k1 static-lib
// cycle a direct engine->backend call would create, and needs no compile-time
// macro. Default (CPU-only) builds never link the provider, so the hook stays
// null and verification runs on the CPU column path below.
namespace { std::atomic<GpuColumnsVerifyHook> g_gpu_columns_hook{nullptr}; }
GpuColumnsVerifyHook install_gpu_columns_verify_hook(GpuColumnsVerifyHook hook) noexcept {
    return g_gpu_columns_hook.exchange(hook, std::memory_order_acq_rel);
}

namespace detail {
// Single process-wide persistent worker pool, lazily created on first batch-verify _mt
// call and reused thereafter (no per-call thread spawn).
//
// INTENTIONALLY LEAKED (heap, never deleted): the destructor would join the worker
// threads at static-destruction time, which on Windows runs during DLL unload while the
// loader lock is held — joining threads there deadlocks (the workers need the loader lock
// to exit). Leaking the singleton means no destructor runs; the OS reclaims the threads
// and memory at process exit. This is the portable (MSVC + libstdc++ + libc++) choice and
// avoids static-destruction-order hazards as well.
BatchWorkerPool& batch_worker_pool() {
    static BatchWorkerPool* pool = new BatchWorkerPool();
    return *pool;
}
}  // namespace detail

// Seeded hash for 32-byte pubkey deduplication.
// Seed is randomised per batch call so adversarial pubkey inputs cannot
// pre-compute collisions and degrade lookup to O(n) (CA-010).
struct PubkeyHash32 {
    std::size_t seed{static_cast<std::size_t>(0xcbf29ce484222325ULL)};
    explicit PubkeyHash32(std::size_t s) : seed(s) {}
    std::size_t operator()(const std::array<uint8_t, 32>& k) const noexcept {
        std::size_t h = seed;
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
constexpr std::size_t kOpaqueBatchChunk = 4096;
constexpr std::size_t kOpaqueBatchMinRowsPerThread = 128;
constexpr std::size_t kOpaqueBatchStealFloor = 64;
constexpr std::size_t kEcdsaRowBytes = 32 + 33 + 64;
constexpr std::size_t kSchnorrRowBytes = 32 + 32 + 64;

bool row_layout_overflows(std::size_t count, std::size_t stride,
                          std::size_t row_bytes) noexcept {
    if (count <= 1) return false;
    const std::size_t max = std::numeric_limits<std::size_t>::max();
    return stride > (max - row_bytes) / (count - 1);
}

bool column_layout_overflows(std::size_t count, std::size_t item_bytes) noexcept {
    return count != 0 &&
           count > (std::numeric_limits<std::size_t>::max() / item_bytes);
}

void zero_results(std::uint8_t* out_results, std::size_t count) noexcept {
    if (out_results != nullptr && count != 0) {
        std::memset(out_results, 0, count);
    }
}

void one_results(std::uint8_t* out_results, std::size_t count) noexcept {
    if (out_results != nullptr && count != 0) {
        std::memset(out_results, 1, count);
    }
}

unsigned opaque_batch_thread_count(std::size_t count, std::size_t max_threads) {
    auto& pool = detail::batch_worker_pool();
    const unsigned hw = pool.size();
    const unsigned want = (max_threads == 0)
        ? hw
        : static_cast<unsigned>(std::min<std::size_t>(max_threads, hw));
    const std::size_t by_work = std::max<std::size_t>(
        1, count / kOpaqueBatchMinRowsPerThread);
    return static_cast<unsigned>(std::min<std::size_t>(
        static_cast<std::size_t>(want), by_work));
}

std::size_t opaque_batch_steal_size(std::size_t count,
                                    unsigned n_threads) noexcept {
    if (n_threads <= 1) {
        return std::min<std::size_t>(count, kOpaqueBatchChunk);
    }
    return std::clamp<std::size_t>(
        count / (static_cast<std::size_t>(n_threads) * 4),
        kOpaqueBatchStealFloor, kOpaqueBatchChunk);
}

bool decompress_compressed_pubkey(const std::uint8_t pub33[33],
                                  Point& out) noexcept {
    if (pub33[0] != 0x02 && pub33[0] != 0x03) {
        return false;
    }
    FieldElement x;
    if (!FieldElement::parse_bytes_strict(pub33 + 1, x)) {
        return false;
    }

#if defined(__SIZEOF_INT128__) && !defined(SECP256K1_PLATFORM_ESP32) && !defined(SECP256K1_PLATFORM_STM32) && !defined(__EMSCRIPTEN__)
    using fast::FieldElement52;
    const FieldElement52 x52 = FieldElement52::from_fe(x);
    static const std::uint64_t k7[4] = {7u, 0u, 0u, 0u};
    const FieldElement52 y2 = x52.square() * x52 +
                              FieldElement52::from_4x64_limbs(k7);
    const FieldElement52 y52 = y2.sqrt();
    if (!(y52.square() == y2)) {
        return false;
    }
    FieldElement y = y52.to_fe();
#else
    const FieldElement y2 = x.square() * x + FieldElement::from_uint64(7);
    FieldElement y = y2.sqrt();
    if (!(y.square() == y2)) {
        return false;
    }
#endif

    if (((y.limbs()[0] & 1u) != 0u) != (pub33[0] == 0x03)) {
        y = FieldElement::zero() - y;
    }
    out = Point::from_affine(x, y);
    return !out.is_infinity();
}

Scalar opaque_scalar_le(const std::uint8_t* p) noexcept {
    auto rd = [](const std::uint8_t* q) noexcept {
        std::uint64_t v = 0;
        for (int i = 0; i < 8; ++i) {
            v |= static_cast<std::uint64_t>(q[i]) << (i * 8);
        }
        return v;
    };
    return Scalar::from_limbs({rd(p), rd(p + 8), rd(p + 16), rd(p + 24)});
}

bool parse_ecdsa_opaque_entry(const std::uint8_t* hash32,
                              const std::uint8_t* pub33,
                              const std::uint8_t* sig64,
                              ECDSABatchEntry& out) noexcept {
    std::memcpy(out.msg_hash.data(), hash32, 32);
    if (!decompress_compressed_pubkey(pub33, out.public_key)) {
        return false;
    }
    out.signature = ECDSASignature{opaque_scalar_le(sig64),
                                   opaque_scalar_le(sig64 + 32)};
    return true;
}

bool parse_schnorr_bip340_entry(const std::uint8_t* msg32,
                                const std::uint8_t* xonly32,
                                const std::uint8_t* sig64,
                                SchnorrBatchEntry& out) noexcept {
    std::memcpy(out.message.data(), msg32, 32);
    std::memcpy(out.pubkey_x.data(), xonly32, 32);
    return SchnorrSignature::parse_strict(sig64, out.signature);
}

template <typename Entry, typename ParseEntry, typename VerifyBatch,
          typename VerifyOne>
bool verify_opaque_bounded(std::size_t count, std::uint8_t* out_results,
                           std::size_t max_threads, ParseEntry&& parse_entry,
                           VerifyBatch&& verify_batch, VerifyOne&& verify_one) {
    if (count == 0) {
        return true;
    }

    const unsigned n_threads = opaque_batch_thread_count(count, max_threads);
    const std::size_t steal = opaque_batch_steal_size(count, n_threads);
    auto& pool = detail::batch_worker_pool();
    const bool all_valid = pool.run(
        count, steal, n_threads,
        [&](std::size_t s, std::size_t e) -> bool {
            static thread_local std::vector<Entry> local;
            local.clear();
            local.resize(e - s);
            for (std::size_t i = s; i < e; ++i) {
                if (!parse_entry(i, local[i - s])) {
                    return false;
                }
            }
            return verify_batch(local.data(), e - s);
        });

    if (all_valid) {
        one_results(out_results, count);
        return true;
    }

    bool ok_all = true;
    for (std::size_t i = 0; i < count; ++i) {
        Entry one{};
        const bool ok = parse_entry(i, one) && verify_one(one);
        if (out_results != nullptr) {
            out_results[i] = ok ? 1u : 0u;
        }
        ok_all = ok_all && ok;
    }
    return ok_all;
}

// Generate deterministic weights for batch verification.
// batch_seed: SHA256 over all signature data (binds to entire batch).
// Returns a_i = SHA256(batch_seed || i) interpreted as scalar.
// All weights are derived from SHA256(batch_seed || i_le32) for every index
// including index 0. Fixing a_0 = 1 deviates from the batch-verify soundness
// proof which requires all weights to be uniform random.
// Compact midstate variant: avoids copying the full SHA256 object (104 bytes)
// per call. SHA256::Midstate holds only the 8×uint32 state + 64-bit counter
// (40 bytes). buf_[64] is always zero-filled in a midstate context.
Scalar batch_weight(const SHA256::Midstate& midstate, uint32_t index) {
    uint8_t index_bytes[4];
    index_bytes[0] = static_cast<uint8_t>(index & 0xFF);
    index_bytes[1] = static_cast<uint8_t>((index >> 8) & 0xFF);
    index_bytes[2] = static_cast<uint8_t>((index >> 16) & 0xFF);
    index_bytes[3] = static_cast<uint8_t>((index >> 24) & 0xFF);

    SHA256 ctx = SHA256::from_midstate(midstate);  // 40-byte copy, not 104
    ctx.update(index_bytes, sizeof(index_bytes));
    auto h = ctx.finalize();
    Scalar w = Scalar::from_bytes(h);
    // SEC-007: SHA256 output equal to the curve order n reduces to 0 mod n.
    // A zero weight would silently exclude this signature from the batch check,
    // turning a potentially invalid signature into an unchecked one.
    // Probability is ~2^-128 but the consequence is fail-open, so we use 1
    // as a safe non-zero fallback.
    if (w.is_zero()) w = Scalar::one();
    return w;
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
    if (n == 0) return false;
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
    // Compute batch seed = SHA256(csprng_rand || all signature data)
    // P2-SEC-002: XOR 32 CSPRNG bytes into the seed before the weight loop.
    // Without randomization, batch_weight_i = SHA256(SHA256(all_sig_data) || i)
    // is deterministic and computable by any adversary who knows the batch
    // entries.  A Wagner-style attack could craft inputs whose weights sum to
    // cancel a forged entry's contribution.  libsecp256k1 uses a CSPRNG seed
    // for exactly this reason (see secp256k1_batch_randomizer_gen in batch.h).
    // We sample 32 bytes once per batch call and XOR them into the digest
    // before use.  The XOR preserves the domain-binding property of the
    // SHA256 while making weights unpredictable to the adversary.
    std::uint8_t csprng_rand[32];
    secp256k1::detail::csprng_fill(csprng_rand, sizeof(csprng_rand));

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
    // XOR CSPRNG randomness into the batch seed (one 32-byte XOR, not per-weight).
    for (std::size_t j = 0; j < 32; ++j) {
        batch_seed[j] ^= csprng_rand[j];
    }
    detail::secure_erase(csprng_rand, sizeof(csprng_rand));

    SHA256 batch_weight_base;
    batch_weight_base.update(batch_seed.data(), batch_seed.size());
    // Capture compact midstate once (40 bytes) — avoids 104-byte SHA256 copy per weight call.
    SHA256::Midstate const bw_mid = batch_weight_base.capture_midstate();

    std::size_t const msm_n = 2 * n;
    auto& scratch = schnorr_batch_scratch(msm_n);
    Scalar* const scalars = scratch.scalars.data();
    Point* const points = scratch.points.data();

    Scalar g_coeff = Scalar::zero();

    for (std::size_t i = 0; i < n; ++i) {
        Scalar const weight = batch_weight(bw_mid, static_cast<uint32_t>(i));

        auto [r_ok, R_pt] = lift_x(entries[i].signature.r);
        if (!r_ok) return false;

        Point P_pt = Point::infinity();
        if (!resolve_pubkey(entries[i], P_pt)) return false;

        auto const* const pubkey_x = pubkey_bytes(entries[i]);
        if (pubkey_x == nullptr) return false;

        alignas(16) uint8_t challenge_input[96];
        std::memcpy(challenge_input +  0, entries[i].signature.r.data(), 32);
        std::memcpy(challenge_input + 32, pubkey_x->data(), 32);
        std::memcpy(challenge_input + 64, entries[i].message.data(), 32);
        Scalar const challenge = Scalar::from_bytes(
            detail::cached_tagged_hash(detail::g_challenge_midstate, challenge_input, 96));

        g_coeff += weight * entries[i].signature.s;

        scalars[i] = (weight * challenge).negate();
        points[i] = P_pt;
        scalars[n + i] = weight.negate();
        points[n + i] = R_pt;
    }

    // g_coeff = sum(weight_i * sig_i.s) — all public data; VT correct here.
    auto G_term = Point::generator().scalar_mul(g_coeff);
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
    // B-6 / CA-010: per-call seeded hash map for O(1) dedup.
    // New seed each call prevents adversarial pubkey inputs from pre-computing
    // hash collisions that degrade lookup to O(n) worst-case.
    //
    // Security rationale (cpp:S2245 — non-cryptographic PRNG):
    //   mt19937_64 below is NOT used for any cryptographic purpose. It only
    //   diversifies the hash-table seed across calls so an attacker who can
    //   control input pubkeys cannot pre-compute hash collisions that would
    //   flood a single bucket and turn lookup O(n²). The seed itself comes
    //   from std::random_device (OS entropy). No secret material, signing
    //   randomness, or curve operations use this stream. Suitable use case
    //   for a fast PRNG; substituting a CSPRNG here would buy nothing.
    using PubkeyMap = std::unordered_map<std::array<uint8_t, 32>, Point, PubkeyHash32>;
    static thread_local std::mt19937_64 rng{std::random_device{}()};  // NOSONAR(cpp:S2245)
    PubkeyMap pubkey_index(n, PubkeyHash32{static_cast<std::size_t>(rng())});
    pubkey_index.reserve(n);

    // Grow-only vector reserve (no realloc when n stays the same between calls)
    static thread_local std::vector<SchnorrXonlyPubkey> pubkey_cache;
    pubkey_cache.clear();
    if (pubkey_cache.capacity() < n) pubkey_cache.reserve(n);

    auto verify_one = [](const SchnorrBatchEntry& entry) {
        return schnorr_verify(entry.pubkey_x, entry.message, entry.signature);
    };
    auto resolve_pubkey = [&](const SchnorrBatchEntry& entry,
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
    if (n == 0) return false;

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
        // CA-001 (clang/arch UB fix): reject an invalid (off-curve / infinity)
        // public key explicitly. The shim maps an off-curve pubkey to
        // Point::infinity() (pubkey_data_to_point), and the single ecdsa_verify
        // path — used by the small-batch fallback below — rejects it. The
        // large-batch MSM path did not: it fed the infinity point into
        // dual_scalar_mul_gen_point + the FE52 Z^2 x-coordinate check, whose
        // behavior on an infinity operand is compiler-dependent (rejected under
        // gcc, but accepted under clang on x86-64 and arm64). Rejecting up-front
        // makes off-curve/infinity rejection identical to single verify on every
        // compiler and architecture.
        if (entries[i].public_key.is_infinity()) {
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
            // p - n in 4x64 LE: limb[0]=0x402DA1722FC9BAEE, limb[1]=0x4551231950B75FC4,
            //                   limb[2]=0x01, limb[3]=0x00
            static constexpr std::uint64_t PMN_0 = 0x402DA1722FC9BAEEULL;
            static constexpr std::uint64_t PMN_1 = 0x4551231950B75FC4ULL;
            const auto& rl = entries[i].signature.r.limbs();
            bool r_less_than_pmn = false;
            if (rl[3] != 0 || rl[2] > 1) {
                r_less_than_pmn = false;
            } else if (rl[2] == 0) {
                r_less_than_pmn = true;
            } else {
                // rl[2] == 1: compare limbs[1..0] against PMN_1..PMN_0
                if (rl[1] != PMN_1) r_less_than_pmn = (rl[1] < PMN_1);
                else                r_less_than_pmn = (rl[0] < PMN_0);
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

// ---------------------------------------------------------------------------
// ecdsa_batch_verify_mt: first-class multi-threaded ECDSA batch verification.
//
// Verification is variable-time over PUBLIC data (pubkey/sig/msg) — there is no
// secret material — so parallelism is purely a throughput win and the boolean
// result is identical to the single-threaded ecdsa_batch_verify for any thread
// count. CPU parallelism is a property of the engine, not of any caller/bridge.
//
// Rows are split into fixed-size chunks pulled from an atomic counter and run
// across `max_threads` threads (0 => hardware_concurrency; 1 => serial). The
// thread count is the caller's to own — an explicit request is honoured and is
// only reduced by the engine when it exceeds what can be used (hardware
// concurrency, or the number of chunks). There is no arbitrary upper cap; the
// caller controls thread priority via the calling process's thread priority
// (inherited by the spawned workers). Each chunk runs the serial
// ecdsa_batch_verify over its sub-range, so the Montgomery batch inversion still
// amortises within the chunk and the per-thread scratch (ecdsa_batch_verify's
// thread_local arena) stays small (~kChunk scalars) regardless of n — no O(n)
// inversion arena. Any invalid entry in any chunk makes the whole call return
// false (fail-closed), matching the serial contract. For n <= kChunk the call
// is exactly the serial path (single chunk, single thread), including its
// small-n individual-verify branch.
//
// Thread-safety: the GLV/generator precompute (get_dual_mul_gen_tables) is a
// C++11 function-local magic static (standard-guaranteed thread-safe init);
// ecdsa_batch_verify's scratch is thread_local; the point arithmetic uses no
// shared mutable state. The existing parallel sign batch relies on the same
// guarantees.
// ---------------------------------------------------------------------------
bool ecdsa_batch_verify_mt(const ECDSABatchEntry* entries, std::size_t n,
                           std::size_t max_threads) {
    if (n == 0) return false;  // identical to the serial ecdsa_batch_verify contract

    // Parallelism granularity is DECOUPLED from the batch-inversion work-steal chunk.
    // (Old bug: n_threads was capped by ceil(n/4096), so any batch < 4096 sigs ran on ONE
    // thread regardless of max_threads — block-sized batches never parallelized.) A single
    // ECDSA verify is ~25-100us, so even ~128 rows/thread dwarfs scheduling overhead. The
    // PERSISTENT pool (created once, reused) avoids a per-call std::thread spawn storm and
    // keeps worker thread_locals warm — matching libbitcoin's std::for_each(par).
    static constexpr std::size_t kMinRowsPerThread = 128;
    static constexpr std::size_t kStealFloor = 64;   // >= batch-inversion cutoff (8)

    auto& pool = detail::batch_worker_pool();
    const unsigned hw = pool.size();
    const unsigned want = (max_threads == 0)
        ? hw
        : static_cast<unsigned>(std::min<std::size_t>(max_threads, hw));
    const std::size_t by_work = std::max<std::size_t>(1, n / kMinRowsPerThread);
    const unsigned n_threads = static_cast<unsigned>(std::min<std::size_t>(
        static_cast<std::size_t>(want), by_work));
    const std::size_t steal = (n_threads <= 1)
        ? n
        : std::clamp<std::size_t>(n / (static_cast<std::size_t>(n_threads) * 4),
                                  kStealFloor, std::size_t{4096});

    return pool.run(n, steal, n_threads,
                    [entries](std::size_t s, std::size_t e) {
                        return ecdsa_batch_verify(entries + s, e - s);
                    });
}

bool ecdsa_batch_verify_mt(const std::vector<ECDSABatchEntry>& entries,
                           std::size_t max_threads) {
    return ecdsa_batch_verify_mt(entries.data(), entries.size(), max_threads);
}

// ---------------------------------------------------------------------------
// schnorr_batch_verify_mt: first-class multi-threaded Schnorr batch verify.
//
// The Schnorr twin of ecdsa_batch_verify_mt. BIP-340 verification is
// variable-time over PUBLIC data (pubkey/msg/sig) with no secret material, so
// parallelism is a pure throughput win and the boolean result is identical to
// the single-threaded schnorr_batch_verify for any thread count. Rows are
// split into fixed-size chunks pulled from an atomic counter; each chunk runs
// the serial schnorr_batch_verify over its sub-range (so the random-linear-
// combination amortises within the chunk and per-thread scratch stays
// O(kChunk), never O(n)). Any invalid entry in any chunk fails the whole call
// (fail-closed). For n <= kChunk this is exactly the serial path; for n == 0 it
// delegates to the serial contract.
// ---------------------------------------------------------------------------
bool schnorr_batch_verify_mt(const SchnorrBatchEntry* entries, std::size_t n,
                             std::size_t max_threads) {
    if (n == 0) return schnorr_batch_verify(entries, 0);  // identical serial contract

    // Same principle as ecdsa_batch_verify_mt: decouple worker count from a fixed 4096
    // chunk (so block-sized batches parallelize) and run on the PERSISTENT pool (no
    // per-call spawn). Each chunk is an independent Schnorr batch (its own randomized
    // MSM + infinity check), so chunking is correct; the steal floor keeps each chunk's
    // MSM reasonably sized.
    static constexpr std::size_t kMinRowsPerThread = 128;
    static constexpr std::size_t kStealFloor = 64;

    auto& pool = detail::batch_worker_pool();
    const unsigned hw = pool.size();
    const unsigned want = (max_threads == 0)
        ? hw
        : static_cast<unsigned>(std::min<std::size_t>(max_threads, hw));
    const std::size_t by_work = std::max<std::size_t>(1, n / kMinRowsPerThread);
    const unsigned n_threads = static_cast<unsigned>(std::min<std::size_t>(
        static_cast<std::size_t>(want), by_work));
    const std::size_t steal = (n_threads <= 1)
        ? n
        : std::clamp<std::size_t>(n / (static_cast<std::size_t>(n_threads) * 4),
                                  kStealFloor, std::size_t{4096});

    return pool.run(n, steal, n_threads,
                    [entries](std::size_t s, std::size_t e) {
                        return schnorr_batch_verify(entries + s, e - s);
                    });
}

bool schnorr_batch_verify_mt(const std::vector<SchnorrBatchEntry>& entries,
                             std::size_t max_threads) {
    return schnorr_batch_verify_mt(entries.data(), entries.size(), max_threads);
}

bool ecdsa_batch_verify_opaque_rows(const std::uint8_t* rows, std::size_t stride,
                                    std::size_t count,
                                    std::uint8_t* out_results,
                                    std::size_t max_threads) {
    if (count == 0) {
        return true;
    }
    if (rows == nullptr || stride < kEcdsaRowBytes ||
        row_layout_overflows(count, stride, kEcdsaRowBytes)) {
        zero_results(out_results, count);
        return false;
    }

    return verify_opaque_bounded<ECDSABatchEntry>(
        count, out_results, max_threads,
        [rows, stride](std::size_t i, ECDSABatchEntry& out) {
            const std::uint8_t* row = rows + i * stride;
            return parse_ecdsa_opaque_entry(row, row + 32, row + 65, out);
        },
        [](const ECDSABatchEntry* entries, std::size_t n) {
            return ecdsa_batch_verify(entries, n);
        },
        [](const ECDSABatchEntry& entry) {
            return ecdsa_batch_verify(&entry, 1);
        });
}

bool ecdsa_batch_verify_opaque_columns(const std::uint8_t* digests32,
                                       const std::uint8_t* pubkeys33,
                                       const std::uint8_t* sigs64,
                                       std::size_t count,
                                       std::uint8_t* out_results,
                                       std::size_t max_threads) {
    if (count == 0) {
        return true;
    }
    if (digests32 == nullptr || pubkeys33 == nullptr || sigs64 == nullptr ||
        column_layout_overflows(count, 32) ||
        column_layout_overflows(count, 33) ||
        column_layout_overflows(count, 64)) {
        zero_results(out_results, count);
        return false;
    }

    if (out_results != nullptr) {
        if (GpuColumnsVerifyHook hook = g_gpu_columns_hook.load(std::memory_order_acquire)) {
            const int rc = hook(0, digests32, pubkeys33, sigs64, count, out_results);
            if (rc >= 0) {
                return rc == 1;   // GPU handled the whole batch; out_results written
            }
            // rc < 0: decline -> CPU fallback below overwrites out_results fully.
        }
    }

    return verify_opaque_bounded<ECDSABatchEntry>(
        count, out_results, max_threads,
        [digests32, pubkeys33, sigs64](std::size_t i, ECDSABatchEntry& out) {
            return parse_ecdsa_opaque_entry(digests32 + i * 32,
                                            pubkeys33 + i * 33,
                                            sigs64 + i * 64, out);
        },
        [](const ECDSABatchEntry* entries, std::size_t n) {
            return ecdsa_batch_verify(entries, n);
        },
        [](const ECDSABatchEntry& entry) {
            return ecdsa_batch_verify(&entry, 1);
        });
}

bool schnorr_batch_verify_bip340_rows(const std::uint8_t* rows,
                                      std::size_t stride,
                                      std::size_t count,
                                      std::uint8_t* out_results,
                                      std::size_t max_threads) {
    if (count == 0) {
        return true;
    }
    if (rows == nullptr || stride < kSchnorrRowBytes ||
        row_layout_overflows(count, stride, kSchnorrRowBytes)) {
        zero_results(out_results, count);
        return false;
    }

    return verify_opaque_bounded<SchnorrBatchEntry>(
        count, out_results, max_threads,
        [rows, stride](std::size_t i, SchnorrBatchEntry& out) {
            const std::uint8_t* row = rows + i * stride;
            return parse_schnorr_bip340_entry(row, row + 32, row + 64, out);
        },
        [](const SchnorrBatchEntry* entries, std::size_t n) {
            return schnorr_batch_verify(entries, n);
        },
        [](const SchnorrBatchEntry& entry) {
            return schnorr_verify(entry.pubkey_x, entry.message, entry.signature);
        });
}

bool schnorr_batch_verify_bip340_columns(const std::uint8_t* digests32,
                                         const std::uint8_t* xonly32,
                                         const std::uint8_t* sigs64,
                                         std::size_t count,
                                         std::uint8_t* out_results,
                                         std::size_t max_threads) {
    if (count == 0) {
        return true;
    }
    if (digests32 == nullptr || xonly32 == nullptr || sigs64 == nullptr ||
        column_layout_overflows(count, 32) ||
        column_layout_overflows(count, 64)) {
        zero_results(out_results, count);
        return false;
    }

    if (out_results != nullptr) {
        if (GpuColumnsVerifyHook hook = g_gpu_columns_hook.load(std::memory_order_acquire)) {
            const int rc = hook(1, digests32, xonly32, sigs64, count, out_results);
            if (rc >= 0) {
                return rc == 1;   // GPU handled the whole batch; out_results written
            }
            // rc < 0: decline -> CPU fallback below overwrites out_results fully.
        }
    }

    return verify_opaque_bounded<SchnorrBatchEntry>(
        count, out_results, max_threads,
        [digests32, xonly32, sigs64](std::size_t i, SchnorrBatchEntry& out) {
            return parse_schnorr_bip340_entry(digests32 + i * 32,
                                              xonly32 + i * 32,
                                              sigs64 + i * 64, out);
        },
        [](const SchnorrBatchEntry* entries, std::size_t n) {
            return schnorr_batch_verify(entries, n);
        },
        [](const SchnorrBatchEntry& entry) {
            return schnorr_verify(entry.pubkey_x, entry.message, entry.signature);
        });
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
