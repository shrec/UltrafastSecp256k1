// ============================================================================
// REGRESSION: Schnorr batch verification with 128-bit BIP-340 randomizers
// ============================================================================
// Track:  SECURITY -- schnorr_batch_verify soundness after the short-weight MSM
//
// schnorr_batch_verify_impl (src/cpu/src/batch_verify.cpp) checks
//     (sum a_i*s_i)*G - sum (a_i*e_i)*P_i - sum a_i*R_i == O
// with the BIP-340 batch weights
//     a_0 = 1,  a_i = 128 bits of SHA256(seed || i) with bit 0 set (i >= 1)
// where seed = SHA256(r || s || pubkey_x || msg over all entries) XOR 32
// CSPRNG bytes. The R terms enter pippenger_msm_glv_short unsplit (a_i is
// already below 2^128) and only the full-width a_i*e_i are GLV-split, so the
// bucket pass runs over 3n points instead of 4n.
//
// Shortening the weights is only acceptable if it costs no soundness, so this
// module pins every property the argument in batch_verify.cpp relies on:
//
//   1. Valid batches are accepted at n = 27, 28 (either side of the
//      individual-verify cutoff), 39, 40, 64, 213, 214 and 1365, 1366 (the
//      MSM has 3n points, so these straddle the window band edges at 640 and
//      4096), 511, 512, 513 (around a power of two) and 4096, for both entry
//      types.
//   2. A batch with ONE invalid signature, at a random position, is rejected
//      in every trial -- position 0 (a_0 = 1) and position n-1 included.
//   3. Two invalid signatures whose errors cancel EXACTLY under equal weights
//      (s_i + d, s_j - d) are rejected. The test first proves the pair really
//      does cancel under equal weights, so a pass means the weights are doing
//      the work, not the corruption.
//   4. Mixed batches give the same verdict as per-signature schnorr_verify.
//   5. The engine's own weight function: weight 0 is 1, every other weight is
//      nonzero, odd and below 2^128, and depends on the seed.
//   6. The engine's own seed digest changes when any single byte of any
//      entry's r, s, pubkey or message changes, and when two entries swap.
//
// 5 and 6 call the engine through two test views exported by batch_verify.cpp
// (declared below, not in a public header), so they test the production
// functions rather than a model of them.
// ============================================================================

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <set>
#include <utility>
#include <vector>

#include "secp256k1/batch_verify.hpp"
#include "secp256k1/point.hpp"
#include "secp256k1/scalar.hpp"
#include "secp256k1/schnorr.hpp"

using namespace secp256k1;
using secp256k1::fast::Point;
using secp256k1::fast::Scalar;

namespace secp256k1::detail {
std::array<std::uint8_t, 32> schnorr_batch_seed_digest_for_test(
    const SchnorrBatchEntry* entries, std::size_t n);
fast::Scalar schnorr_batch_weight_for_test(const std::array<std::uint8_t, 32>& seed,
                                           std::uint32_t index);
} // namespace secp256k1::detail

namespace {

int g_pass = 0;
int g_fail = 0;

void check(bool cond, const char* msg) {
    if (cond) {
        ++g_pass;
    } else {
        ++g_fail;
        std::printf("  [FAIL] %s\n", msg);
        std::fflush(stdout);
    }
}

constexpr std::size_t kPoolSize = 4096;

struct Pool {
    std::vector<SchnorrXonlyPubkey> pubkeys;
    std::vector<SchnorrBatchEntry> plain;
};

Pool make_pool(std::mt19937_64& rng) {
    Pool pool;
    pool.pubkeys.resize(kPoolSize);
    pool.plain.resize(kPoolSize);
    for (std::size_t i = 0; i < kPoolSize; ++i) {
        Scalar const sk = Scalar::from_limbs(
            {0x9e3779b97f4a7c15ULL * (i + 1), rng(), rng(), 0x0123456789abcdefULL});
        SchnorrKeypair const kp = schnorr_keypair_create(sk);
        std::array<std::uint8_t, 32> msg{};
        std::array<std::uint8_t, 32> aux{};
        for (std::size_t b = 0; b < 32; ++b) {
            msg[b] = static_cast<std::uint8_t>(rng());
            aux[b] = static_cast<std::uint8_t>(rng());
        }
        pool.pubkeys[i] = schnorr_xonly_from_keypair(kp);
        pool.plain[i].pubkey_x = kp.px;
        pool.plain[i].message = msg;
        pool.plain[i].signature = schnorr_sign(kp, msg, aux);
    }
    return pool;
}

std::vector<SchnorrBatchCachedEntry> to_cached(const Pool& pool,
                                               const std::vector<SchnorrBatchEntry>& plain,
                                               std::size_t pool_offset) {
    std::vector<SchnorrBatchCachedEntry> out(plain.size());
    for (std::size_t i = 0; i < plain.size(); ++i) {
        out[i].pubkey = &pool.pubkeys[pool_offset + i];
        out[i].message = plain[i].message;
        out[i].signature = plain[i].signature;
    }
    return out;
}

bool individual_all(const std::vector<SchnorrBatchEntry>& batch) {
    bool all = true;
    for (auto const& e : batch) {
        all = schnorr_verify(e.pubkey_x, e.message, e.signature) && all;
    }
    return all;
}

Scalar scalar_one_plus(std::uint64_t v) {
    return Scalar::from_limbs({v, 0, 0, 0});
}

// Corrupt one entry so it fails verification while still parsing: lift_x and
// the pubkey must succeed, so the failure has to come out of the MSM itself.
void corrupt(SchnorrBatchEntry& e, unsigned kind, std::mt19937_64& rng) {
    switch (kind % 3) {
    case 0:
        e.signature.s = e.signature.s + scalar_one_plus(1 + (rng() & 0xffff));
        break;
    case 1:
        e.message[rng() % 32] ^= static_cast<std::uint8_t>(1u << (rng() % 8));
        break;
    default: {
        // A different but valid message hash under the same (r, s): the
        // challenge e changes, R and P stay on the curve.
        e.message[0] ^= 0x80;
        e.message[31] ^= 0x01;
        break;
    }
    }
}

// ---------------------------------------------------------------------------
// valid_batches_accepted
// ---------------------------------------------------------------------------
void test_valid_batches_accepted(const Pool& pool) {
    std::printf("[1] valid batches accepted at the boundary sizes\n");
    const std::size_t sizes[] = {27, 28, 39, 40, 64, 213, 214, 511, 512, 513, 1365, 1366, 4096};
    for (std::size_t n : sizes) {
        std::vector<SchnorrBatchEntry> batch(pool.plain.begin(),
                                             pool.plain.begin() + static_cast<std::ptrdiff_t>(n));
        auto cached = to_cached(pool, batch, 0);
        char msg[96];
        std::snprintf(msg, sizeof(msg), "valid plain batch accepted (n=%zu)", n);
        check(schnorr_batch_verify(batch), msg);
        std::snprintf(msg, sizeof(msg), "valid cached batch accepted (n=%zu)", n);
        check(schnorr_batch_verify(cached), msg);
    }
    // A slice that does not start at pool index 0, so a_0 = 1 lands on a
    // different signature.
    std::vector<SchnorrBatchEntry> shifted(pool.plain.begin() + 1000, pool.plain.begin() + 1513);
    check(schnorr_batch_verify(shifted), "valid batch at pool offset 1000 accepted (n=513)");
}

// ---------------------------------------------------------------------------
// single_invalid_rejected_at_every_position
// ---------------------------------------------------------------------------
void test_single_invalid_rejected(const Pool& pool, std::mt19937_64& rng) {
    std::printf("[2] one invalid signature at a random position is always rejected\n");
    struct Shape { std::size_t n; unsigned trials; };
    const Shape shapes[] = {{28, 28}, {40, 40}, {64, 64}, {513, 12}};
    unsigned rejected = 0;
    unsigned total = 0;
    for (auto const& sh : shapes) {
        for (unsigned t = 0; t < sh.trials + 2; ++t) {
            std::vector<SchnorrBatchEntry> batch(pool.plain.begin(),
                                                 pool.plain.begin() + static_cast<std::ptrdiff_t>(sh.n));
            std::size_t pos = (t == 0) ? 0 : (t == 1) ? sh.n - 1 : rng() % sh.n;
            corrupt(batch[pos], t, rng);
            bool const indiv = schnorr_verify(batch[pos].pubkey_x, batch[pos].message,
                                              batch[pos].signature);
            check(!indiv, "corrupted entry fails individual schnorr_verify");
            bool const plain_ok = schnorr_batch_verify(batch);
            bool const cached_ok = schnorr_batch_verify(to_cached(pool, batch, 0));
            ++total;
            if (!plain_ok && !cached_ok) {
                ++rejected;
            } else {
                std::printf("  [FAIL] accepted: n=%zu pos=%zu kind=%u\n", sh.n, pos, t % 3);
            }
        }
    }
    std::printf("    rejected %u / %u single-invalid batches\n", rejected, total);
    check(rejected == total, "every single-invalid batch rejected (plain and cached)");
}

// ---------------------------------------------------------------------------
// equal_weight_cancelling_pair_rejected
// ---------------------------------------------------------------------------
Scalar challenge_of(const SchnorrBatchEntry& e) {
    std::uint8_t buf[96];
    std::memcpy(buf, e.signature.r.data(), 32);
    std::memcpy(buf + 32, e.pubkey_x.data(), 32);
    std::memcpy(buf + 64, e.message.data(), 32);
    return Scalar::from_bytes(tagged_hash("BIP0340/challenge", buf, sizeof(buf)));
}

Point lift(const std::array<std::uint8_t, 32>& x) {
    SchnorrXonlyPubkey pk;
    if (!schnorr_xonly_pubkey_parse(pk, x)) return Point::infinity();
    return pk.point;
}

// (s_i + s_j)*G - R_i - R_j - e_i*P_i - e_j*P_j: the pair's contribution to
// the batch equation when both carry the SAME weight.
bool pair_cancels_under_equal_weights(const SchnorrBatchEntry& a, const SchnorrBatchEntry& b) {
    Point lhs = Point::generator().scalar_mul(a.signature.s + b.signature.s);
    Point rhs = lift(a.signature.r)
                    .add(lift(b.signature.r))
                    .add(lift(a.pubkey_x).scalar_mul(challenge_of(a)))
                    .add(lift(b.pubkey_x).scalar_mul(challenge_of(b)));
    return lhs.add(rhs.negate()).is_infinity();
}

void test_cancelling_pair_rejected(const Pool& pool, std::mt19937_64& rng) {
    std::printf("[3] two invalid signatures that cancel under equal weights are rejected\n");
    struct Shape { std::size_t n; unsigned trials; };
    const Shape shapes[] = {{40, 12}, {64, 16}, {512, 6}};
    unsigned rejected = 0;
    unsigned total = 0;
    for (auto const& sh : shapes) {
        for (unsigned t = 0; t < sh.trials; ++t) {
            std::vector<SchnorrBatchEntry> batch(pool.plain.begin(),
                                                 pool.plain.begin() + static_cast<std::ptrdiff_t>(sh.n));
            // Trial 0 pairs the fixed weight a_0 = 1 with a random one; trial 1
            // pairs the two adjacent indices 1 and 2; the rest are random.
            std::size_t i = (t == 0) ? 0 : (t == 1) ? 1 : rng() % sh.n;
            std::size_t j = (t == 0) ? 1 + rng() % (sh.n - 1) : (t == 1) ? 2 : rng() % sh.n;
            if (i == j) j = (j + 1) % sh.n;
            Scalar const d = Scalar::from_limbs({rng() | 1u, rng(), rng(), rng() >> 4});
            batch[i].signature.s = batch[i].signature.s + d;
            batch[j].signature.s = batch[j].signature.s - d;

            check(pair_cancels_under_equal_weights(batch[i], batch[j]),
                  "corrupted pair cancels exactly under equal weights");
            check(!schnorr_verify(batch[i].pubkey_x, batch[i].message, batch[i].signature) &&
                      !schnorr_verify(batch[j].pubkey_x, batch[j].message, batch[j].signature),
                  "both corrupted entries fail individual verify");

            bool const plain_ok = schnorr_batch_verify(batch);
            bool const cached_ok = schnorr_batch_verify(to_cached(pool, batch, 0));
            ++total;
            if (!plain_ok && !cached_ok) {
                ++rejected;
            } else {
                std::printf("  [FAIL] cancelling pair accepted: n=%zu i=%zu j=%zu\n", sh.n, i, j);
            }
        }
    }
    std::printf("    rejected %u / %u cancelling-pair batches\n", rejected, total);
    check(rejected == total, "every equal-weight cancelling pair rejected");
}

// ---------------------------------------------------------------------------
// mixed_batch_matches_individual_verify
// ---------------------------------------------------------------------------
void test_mixed_matches_individual(const Pool& pool, std::mt19937_64& rng) {
    std::printf("[4] mixed batches agree with per-signature schnorr_verify\n");
    const std::size_t sizes[] = {28, 39, 40, 64, 200, 513};
    unsigned agree = 0;
    unsigned total = 0;
    for (std::size_t n : sizes) {
        for (unsigned t = 0; t < 8; ++t) {
            std::vector<SchnorrBatchEntry> batch(pool.plain.begin() + 2000,
                                                 pool.plain.begin() + 2000 + static_cast<std::ptrdiff_t>(n));
            // t = 0: none corrupted; t = 1: all corrupted; otherwise each entry
            // with probability 1/2^t.
            for (std::size_t k = 0; k < n; ++k) {
                bool const hit = (t == 1) || (t >= 2 && (rng() & ((1ULL << t) - 1)) == 0);
                if (hit) corrupt(batch[k], static_cast<unsigned>(rng()), rng);
            }
            bool const expect = individual_all(batch);
            bool const got_plain = schnorr_batch_verify(batch);
            bool const got_cached = schnorr_batch_verify(to_cached(pool, batch, 2000));
            ++total;
            if (got_plain == expect && got_cached == expect) {
                ++agree;
            } else {
                std::printf("  [FAIL] verdict mismatch: n=%zu t=%u expect=%d plain=%d cached=%d\n",
                            n, t, expect, got_plain, got_cached);
            }
        }
    }
    std::printf("    %u / %u mixed batches agree\n", agree, total);
    check(agree == total, "batch verdict == AND of individual verdicts");
}

// ---------------------------------------------------------------------------
// weights_are_short_odd_and_seed_bound
// ---------------------------------------------------------------------------
void test_weight_shape(std::mt19937_64& rng) {
    std::printf("[5] weight 0 is 1; other weights are odd, nonzero and below 2^128\n");
    std::array<std::uint8_t, 32> seed{};
    for (auto& b : seed) b = static_cast<std::uint8_t>(rng());

    Scalar const w0 = detail::schnorr_batch_weight_for_test(seed, 0);
    check(w0 == Scalar::one(), "weight 0 == 1 (BIP-340 a_1 = 1)");

    bool all_ok = true;
    std::set<std::pair<std::uint64_t, std::uint64_t>> seen;
    for (std::uint32_t i = 1; i <= 4096; ++i) {
        Scalar const w = detail::schnorr_batch_weight_for_test(seed, i);
        auto const& l = w.limbs();
        if (w.is_zero() || (l[0] & 1u) == 0 || l[2] != 0 || l[3] != 0) all_ok = false;
        seen.insert({l[0], l[1]});
    }
    check(all_ok, "weights 1..4096 nonzero, bit 0 set, below 2^128");
    check(seen.size() == 4096, "weights 1..4096 pairwise distinct");

    // Same index, seed differing in one bit -> different weight.
    unsigned changed = 0;
    for (unsigned bit = 0; bit < 256; ++bit) {
        auto s2 = seed;
        s2[bit / 8] ^= static_cast<std::uint8_t>(1u << (bit % 8));
        if (!(detail::schnorr_batch_weight_for_test(s2, 7) ==
              detail::schnorr_batch_weight_for_test(seed, 7))) {
            ++changed;
        }
    }
    check(changed == 256, "weight 7 changes under every single-bit seed flip");
    check(detail::schnorr_batch_weight_for_test(seed, 0) ==
              detail::schnorr_batch_weight_for_test(std::array<std::uint8_t, 32>{}, 0),
          "weight 0 is the constant 1 for every seed");
}

// ---------------------------------------------------------------------------
// seed_digest_binds_every_entry_byte
// ---------------------------------------------------------------------------
void test_seed_binding(const Pool& pool) {
    std::printf("[6] the seed digest changes when any single entry byte changes\n");
    constexpr std::size_t n = 40;
    std::vector<SchnorrBatchEntry> batch(pool.plain.begin(), pool.plain.begin() + n);
    auto const base = detail::schnorr_batch_seed_digest_for_test(batch.data(), n);

    unsigned flips = 0;
    unsigned changed = 0;
    for (std::size_t i = 0; i < n; ++i) {
        for (unsigned field = 0; field < 4; ++field) {
            for (unsigned byte = 0; byte < 32; ++byte) {
                auto copy = batch;
                SchnorrBatchEntry& e = copy[i];
                if (field == 0) {
                    e.signature.r[byte] ^= 0x01;
                } else if (field == 1) {
                    std::array<std::uint8_t, 32> sb{};
                    e.signature.s.write_bytes(sb.data());
                    sb[byte] ^= 0x01;
                    e.signature.s = Scalar::from_bytes(sb);
                } else if (field == 2) {
                    e.pubkey_x[byte] ^= 0x01;
                } else {
                    e.message[byte] ^= 0x01;
                }
                ++flips;
                if (detail::schnorr_batch_seed_digest_for_test(copy.data(), n) != base) ++changed;
            }
        }
    }
    std::printf("    %u / %u single-byte changes moved the seed\n", changed, flips);
    check(changed == flips, "every single-byte change in r, s, pubkey_x or msg changes the seed");

    auto swapped = batch;
    std::swap(swapped[3], swapped[17]);
    check(detail::schnorr_batch_seed_digest_for_test(swapped.data(), n) != base,
          "swapping two entries changes the seed (order-bound)");
    check(detail::schnorr_batch_seed_digest_for_test(batch.data(), n - 1) != base,
          "dropping the last entry changes the seed");
}

} // namespace

int test_regression_schnorr_batch_short_weights_run() {
    std::printf("====================================================================\n");
    std::printf("REGRESSION: Schnorr batch verify with 128-bit BIP-340 randomizers\n");
    std::printf("====================================================================\n");
    g_pass = 0;
    g_fail = 0;
    std::mt19937_64 rng(0x5eed5c40a17ULL);
    Pool const pool = make_pool(rng);

    test_valid_batches_accepted(pool);
    test_single_invalid_rejected(pool, rng);
    test_cancelling_pair_rejected(pool, rng);
    test_mixed_matches_individual(pool, rng);
    test_weight_shape(rng);
    test_seed_binding(pool);

    std::printf("[schnorr_batch_short_weights] %d/%d checks passed\n", g_pass, g_pass + g_fail);
    return (g_fail > 0) ? 1 : 0;
}

#if defined(STANDALONE_TEST)
int main() {
    return test_regression_schnorr_batch_short_weights_run();
}
#endif
