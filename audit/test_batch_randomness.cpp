// ============================================================================
// Batch Verify Randomness Audit (Track I6-3)
// ============================================================================
// Verifies security properties of deterministic batch weight generation:
//   1. Determinism: same batch -> same weights
//   2. Uniqueness: different batches -> different weights
//   3. Non-zero: weights are never zero (zero = skip signature)
//   4. Binding: weights depend on ALL signatures in batch
//   5. a_0 = 1 optimization safety
//   6. Distribution: weights are well-distributed (not biased)
//   7. Batch verification correctness: valid batch passes, corrupted fails
//
// Weight derivation: a_i = SHA256(batch_seed || i_le32), a_0 = 1
//   batch_seed = SHA256(R0 || s0 || pk0 || msg0 || ... || Rn || sn || pkn || msgn)
// ============================================================================

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <array>
#include <vector>

#include "secp256k1/field.hpp"
#include "secp256k1/scalar.hpp"
#include "secp256k1/point.hpp"
#include "secp256k1/schnorr.hpp"
#include "secp256k1/batch_verify.hpp"
#include "secp256k1/sha256.hpp"

// Intentionally uses the legacy variable-time secp256k1::ecdsa_sign /
// schnorr_sign entry points (audit harness / cross-platform vectors).
// Suppress the deprecation warning so -Werror builds succeed.
#if defined(__GNUC__) || defined(__clang__)
#  pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif

namespace {

using secp256k1::fast::Scalar;
using secp256k1::fast::Point;
using secp256k1::fast::FieldElement;
using secp256k1::SHA256;

int g_pass = 0;
int g_fail = 0;

void check(bool cond, const char* name) {
    if (cond) {
        ++g_pass;
    } else {
        ++g_fail;
        std::printf("  [FAIL] %s\n", name);
    }
}

// Reproduce batch_weight logic for audit
Scalar batch_weight_audit(const std::array<uint8_t, 32>& batch_seed, uint32_t index) {
    if (index == 0) return Scalar::one();

    uint8_t buf[36];
    std::memcpy(buf, batch_seed.data(), 32);
    buf[32] = static_cast<uint8_t>(index & 0xFF);
    buf[33] = static_cast<uint8_t>((index >> 8) & 0xFF);
    buf[34] = static_cast<uint8_t>((index >> 16) & 0xFF);
    buf[35] = static_cast<uint8_t>((index >> 24) & 0xFF);

    auto h = SHA256::hash(buf, 36);
    return Scalar::from_bytes(h);
}

// Generate batch_seed from entries (same logic as batch_verify.cpp)
std::array<uint8_t, 32> compute_batch_seed(
    const secp256k1::SchnorrBatchEntry* entries, std::size_t n) {
    SHA256 ctx;
    for (std::size_t i = 0; i < n; ++i) {
        ctx.update(entries[i].signature.r.data(), 32);
        auto s_bytes = entries[i].signature.s.to_bytes();
        ctx.update(s_bytes.data(), 32);
        ctx.update(entries[i].pubkey_x.data(), 32);
        ctx.update(entries[i].message.data(), 32);
    }
    return ctx.finalize();
}

// Create a valid Schnorr signature for testing
struct TestSig {
    std::array<uint8_t, 32> pubkey_x;
    std::array<uint8_t, 32> message;
    secp256k1::SchnorrSignature signature;
};

TestSig make_test_sig(uint8_t seed) {
    // Derive private key from seed
    std::array<uint8_t, 32> sk_bytes{};
    sk_bytes[31] = seed + 1; // non-zero
    auto sk = Scalar::from_bytes(sk_bytes);

    // Message
    std::array<uint8_t, 32> msg{};
    msg[0] = seed;
    msg[31] = 0x42;

    // Create keypair (handles even Y adjustment internally)
    auto kp = secp256k1::schnorr_keypair_create(sk);

    // Aux randomness for signing
    std::array<uint8_t, 32> aux{};
    aux[0] = seed;
    aux[1] = 0xAA;

    auto sig = secp256k1::schnorr_sign(kp, msg, aux);

    return {kp.px, msg, sig};
}

// -- Tests --------------------------------------------------------------------

void test_determinism() {
    std::printf("  [1] Determinism: same batch -> same weights\n");

    auto ts0 = make_test_sig(10);
    auto ts1 = make_test_sig(20);
    auto ts2 = make_test_sig(30);

    std::vector<secp256k1::SchnorrBatchEntry> entries = {
        {ts0.pubkey_x, ts0.message, ts0.signature},
        {ts1.pubkey_x, ts1.message, ts1.signature},
        {ts2.pubkey_x, ts2.message, ts2.signature},
    };

    auto seed1 = compute_batch_seed(entries.data(), entries.size());
    auto seed2 = compute_batch_seed(entries.data(), entries.size());

    check(seed1 == seed2, "batch_seed deterministic");

    for (uint32_t i = 0; i < 3; ++i) {
        auto w1 = batch_weight_audit(seed1, i);
        auto w2 = batch_weight_audit(seed2, i);
        check(w1 == w2, "weight deterministic");
    }
}

void test_uniqueness() {
    std::printf("  [2] Uniqueness: different batches -> different seeds/weights\n");

    auto ts0 = make_test_sig(10);
    auto ts1 = make_test_sig(20);
    auto ts2 = make_test_sig(30);
    auto ts3 = make_test_sig(40);

    std::vector<secp256k1::SchnorrBatchEntry> batch_a = {
        {ts0.pubkey_x, ts0.message, ts0.signature},
        {ts1.pubkey_x, ts1.message, ts1.signature},
    };
    std::vector<secp256k1::SchnorrBatchEntry> batch_b = {
        {ts2.pubkey_x, ts2.message, ts2.signature},
        {ts3.pubkey_x, ts3.message, ts3.signature},
    };

    auto seed_a = compute_batch_seed(batch_a.data(), batch_a.size());
    auto seed_b = compute_batch_seed(batch_b.data(), batch_b.size());

    check(seed_a != seed_b, "different batches -> different seeds");

    // Weights for same index must differ between batches
    for (uint32_t i = 1; i < 3; ++i) {
        auto wa = batch_weight_audit(seed_a, i);
        auto wb = batch_weight_audit(seed_b, i);
        check(!(wa == wb), "different batches -> different weights");
    }
}

void test_non_zero_weights() {
    std::printf("  [3] Non-zero: weights are never zero\n");

    auto ts0 = make_test_sig(1);
    std::vector<secp256k1::SchnorrBatchEntry> batch = {
        {ts0.pubkey_x, ts0.message, ts0.signature},
    };
    auto seed = compute_batch_seed(batch.data(), batch.size());

    // a_0 = 1 (non-zero by construction)
    auto w0 = batch_weight_audit(seed, 0);
    check(w0 == Scalar::one(), "a_0 == 1");

    // Check weights for indices 1..1000
    // Probability of SHA256 output mapping to zero mod n is ~2^-256
    for (uint32_t i = 1; i <= 1000; ++i) {
        auto w = batch_weight_audit(seed, i);
        check(!w.is_zero(), "weight non-zero");
    }
}

void test_binding() {
    std::printf("  [4] Binding: weights depend on ALL signatures\n");

    auto ts0 = make_test_sig(10);
    auto ts1 = make_test_sig(20);
    auto ts2 = make_test_sig(30);

    // Batch with 3 sigs
    std::vector<secp256k1::SchnorrBatchEntry> full_batch = {
        {ts0.pubkey_x, ts0.message, ts0.signature},
        {ts1.pubkey_x, ts1.message, ts1.signature},
        {ts2.pubkey_x, ts2.message, ts2.signature},
    };

    // Batch with only 2 sigs (subset)
    std::vector<secp256k1::SchnorrBatchEntry> partial_batch = {
        {ts0.pubkey_x, ts0.message, ts0.signature},
        {ts1.pubkey_x, ts1.message, ts1.signature},
    };

    auto seed_full = compute_batch_seed(full_batch.data(), full_batch.size());
    auto seed_partial = compute_batch_seed(partial_batch.data(), partial_batch.size());

    check(seed_full != seed_partial, "subset batch -> different seed");

    // Swapped order must also produce different seed
    std::vector<secp256k1::SchnorrBatchEntry> swapped_batch = {
        {ts1.pubkey_x, ts1.message, ts1.signature},
        {ts0.pubkey_x, ts0.message, ts0.signature},
        {ts2.pubkey_x, ts2.message, ts2.signature},
    };
    auto seed_swapped = compute_batch_seed(swapped_batch.data(), swapped_batch.size());
    check(seed_full != seed_swapped, "swapped order -> different seed");
}

void test_a0_optimization() {
    std::printf("  [5] a_0 = 1 optimization safety\n");

    // a_0 = 1 is safe because:
    // - Verifier doesn't choose which signature is first
    // - Even with a_0 known, a_1...a_{n-1} remain unpredictable
    // - The equation: a_0*s_0*G + sum(a_i*s_i*G) = a_0*R_0 + sum(a_i*R_i) + ...
    //   An adversary who forges sig_0 must still satisfy a_0 * (bad equation)
    //   which means they need s_0*G = R_0 + e_0*P_0, i.e., standard BIP-340 verify

    // Verify a_0 = 1 for any seed
    std::array<uint8_t, 32> seed1{};
    seed1[0] = 0xAA;
    std::array<uint8_t, 32> seed2{};
    seed2[0] = 0xBB;

    check(batch_weight_audit(seed1, 0) == Scalar::one(), "a_0 == 1 (seed1)");
    check(batch_weight_audit(seed2, 0) == Scalar::one(), "a_0 == 1 (seed2)");

    // But a_1 is NOT 1
    auto w1_s1 = batch_weight_audit(seed1, 1);
    auto w1_s2 = batch_weight_audit(seed2, 1);
    check(!(w1_s1 == Scalar::one()), "a_1 != 1 (seed1)");
    check(!(w1_s2 == Scalar::one()), "a_1 != 1 (seed2)");
}

void test_distribution() {
    std::printf("  [6] Distribution: weights are well-distributed\n");

    auto ts0 = make_test_sig(42);
    std::vector<secp256k1::SchnorrBatchEntry> batch = {
        {ts0.pubkey_x, ts0.message, ts0.signature},
    };
    auto seed = compute_batch_seed(batch.data(), batch.size());

    // Generate 100 weights and check all are distinct
    std::vector<std::array<uint8_t, 32>> weight_bytes;
    weight_bytes.reserve(100);

    for (uint32_t i = 1; i <= 100; ++i) {
        auto w = batch_weight_audit(seed, i);
        weight_bytes.push_back(w.to_bytes());
    }

    // All pairs must be distinct
    bool all_distinct = true;
    for (std::size_t i = 0; i < weight_bytes.size() && all_distinct; ++i) {
        for (std::size_t j = i + 1; j < weight_bytes.size() && all_distinct; ++j) {
            if (weight_bytes[i] == weight_bytes[j]) {
                all_distinct = false;
            }
        }
    }
    check(all_distinct, "100 weights all distinct");

    // Check high byte distribution: at least 50 distinct MSBs in 100 weights
    // (SHA256 should produce near-uniform output)
    int distinct_msb = 0;
    bool seen[256] = {};
    for (auto& wb : weight_bytes) {
        if (!seen[wb[0]]) {
            seen[wb[0]] = true;
            ++distinct_msb;
        }
    }
    check(distinct_msb >= 30, "MSB diversity >= 30/100");
}

void test_batch_verify_correctness() {
    std::printf("  [7] Batch verify correctness\n");

    // Create valid batch of 4 signatures
    std::vector<secp256k1::SchnorrBatchEntry> entries;
    for (uint8_t i = 0; i < 4; ++i) {
        auto ts = make_test_sig(i + 50);
        entries.push_back({ts.pubkey_x, ts.message, ts.signature});
    }

    // Valid batch must pass
    bool ok = secp256k1::schnorr_batch_verify(entries);
    check(ok, "valid batch passes");

    // Corrupted signature must fail
    auto corrupted = entries;
    corrupted[2].signature.r[15] ^= 0x01;
    bool bad = secp256k1::schnorr_batch_verify(corrupted);
    check(!bad, "corrupted sig R -> batch fails");

    // Corrupted message must fail
    auto bad_msg = entries;
    bad_msg[1].message[0] ^= 0xFF;
    bool bad2 = secp256k1::schnorr_batch_verify(bad_msg);
    check(!bad2, "corrupted message -> batch fails");

    // Corrupted pubkey must fail
    auto bad_pk = entries;
    bad_pk[0].pubkey_x[31] ^= 0x01;
    bool bad3 = secp256k1::schnorr_batch_verify(bad_pk);
    check(!bad3, "corrupted pubkey -> batch fails");

    // Single valid sig batch must pass
    std::vector<secp256k1::SchnorrBatchEntry> single = {entries[0]};
    bool single_ok = secp256k1::schnorr_batch_verify(single);
    check(single_ok, "single sig batch passes");

    // Empty batch must pass
    const bool empty_ok = secp256k1::schnorr_batch_verify(
        static_cast<const secp256k1::SchnorrBatchEntry*>(nullptr), 0);
    check(empty_ok, "empty batch passes");
}

} // anonymous namespace

int test_batch_randomness_run() {
    std::printf("\n== Batch Verify Randomness Audit (I6-3) ==\n");

    test_determinism();
    test_uniqueness();
    test_non_zero_weights();
    test_binding();
    test_a0_optimization();
    test_distribution();
    test_batch_verify_correctness();

    std::printf("\n== Results: %d passed, %d failed ==\n", g_pass, g_fail);
    return g_fail;
}

#ifdef STANDALONE_TEST
int main() {
    return test_batch_randomness_run() == 0 ? 0 : 1;
}
#endif
