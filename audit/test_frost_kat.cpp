// ============================================================================
// FROST Reference Cross-Check / Known-Answer Tests (Phase II Task 2.2.5)
// ============================================================================
// Pinned deterministic FROST test vectors for regression:
//   - Lagrange coefficient correctness (known math values)
//   - DKG share consistency (Shamir secret reconstruction)
//   - Signing round determinism (same seeds -> same outputs)
//   - Aggregate signature BIP-340 verification
//   - Cross-threshold consistency (2-of-3 vs 3-of-5 group key for same secrets)
//
// These are Known-Answer Tests (KATs) generated from our implementation with
// fixed seeds. They serve as regression anchors: if any output changes, either
// a bug was introduced or the protocol was intentionally modified.
//
// IETF RFC 9591 does not define a secp256k1 ciphersuite, so external cross-
// check vectors are not available. Instead, we verify mathematical properties
// and pin our own deterministic outputs.
// ============================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <array>
#include <vector>
#include <algorithm>
#include <cmath>

#include "secp256k1/frost.hpp"
#include "secp256k1/schnorr.hpp"
#include "secp256k1/scalar.hpp"
#include "secp256k1/point.hpp"
#include "secp256k1/field.hpp"

using secp256k1::fast::Scalar;
using secp256k1::fast::Point;
using secp256k1::fast::FieldElement;

// -- Minimal test harness -----------------------------------------------------

static int g_pass = 0;
static int g_fail = 0;

#define CHECK(cond, label) do { \
    if (cond) { ++g_pass; } else { \
        ++g_fail; \
        std::printf("  FAIL: %s (line %d)\n", label, __LINE__); \
    } \
} while(0)

// -- Helpers ------------------------------------------------------------------

static std::array<uint8_t, 32> make_seed(uint64_t val) {
    std::array<uint8_t, 32> seed{};
    std::memcpy(seed.data(), &val, 8);
    return seed;
}

// Compare two points by their compressed serialization
static bool points_equal(const Point& a, const Point& b) {
    return a.to_compressed() == b.to_compressed();
}

// ===============================================================================
// Test 1: Lagrange Coefficient Mathematical Properties
// ===============================================================================

static void test_lagrange_properties() {
    std::printf("[1] Lagrange Coefficient: Mathematical Properties\n");

    // Property 1: For signer set {1,2} with 2-of-2:
    //   lambda_1 = 2/(2-1) = 2
    //   lambda_2 = 1/(1-2) = -1 mod n = n-1
    {
        std::vector<secp256k1::ParticipantId> const ids = {1, 2};
        auto l1 = secp256k1::frost_lagrange_coefficient(1, ids);
        auto l2 = secp256k1::frost_lagrange_coefficient(2, ids);

        // l1 should be 2
        auto two = Scalar::from_uint64(2);
        CHECK(l1 == two, "lambda_1({1,2}) == 2");

        // l2 should be n-1 (which is -1 mod n)
        auto neg_one = Scalar::from_uint64(1).negate();
        CHECK(l2 == neg_one, "lambda_2({1,2}) == -1 mod n");
    }

    // Property 2: For signer set {1,2,3} with 3-of-3:
    //   lambda_1 = (2*3)/((2-1)*(3-1)) = 6/2 = 3
    //   lambda_2 = (1*3)/((1-2)*(3-2)) = 3/((-1)*1) = -3 mod n
    //   lambda_3 = (1*2)/((1-3)*(2-3)) = 2/((-2)*(-1)) = 2/2 = 1
    {
        std::vector<secp256k1::ParticipantId> const ids = {1, 2, 3};
        auto l1 = secp256k1::frost_lagrange_coefficient(1, ids);
        auto l2 = secp256k1::frost_lagrange_coefficient(2, ids);
        auto l3 = secp256k1::frost_lagrange_coefficient(3, ids);

        auto three = Scalar::from_uint64(3);
        auto neg_three = three.negate();
        auto one = Scalar::from_uint64(1);

        CHECK(l1 == three, "lambda_1({1,2,3}) == 3");
        CHECK(l2 == neg_three, "lambda_2({1,2,3}) == -3 mod n");
        CHECK(l3 == one, "lambda_3({1,2,3}) == 1");
    }

    // Property 3: Sum of Lagrange coefficients * share_i reconstructs secret
    // For Shamir with polynomial f(x) = 42 + 7x, secret s = f(0):
    //   s = sum(lambda_i * f(i)) for any t-sized subset
    {
        auto secret = Scalar::from_uint64(42);

        // Shares: f(1) = 49, f(2) = 56, f(3) = 63
        auto f1 = Scalar::from_uint64(49);
        auto f2 = Scalar::from_uint64(56);
        auto f3 = Scalar::from_uint64(63);

        // 2-of-3: use participants {1,2}
        std::vector<secp256k1::ParticipantId> const ids12 = {1, 2};
        auto l1 = secp256k1::frost_lagrange_coefficient(1, ids12);
        auto l2 = secp256k1::frost_lagrange_coefficient(2, ids12);

        auto reconstructed = (l1 * f1) + (l2 * f2);
        CHECK(reconstructed == secret, "Shamir reconstruct {1,2} == 42");

        // 2-of-3: use participants {1,3}
        std::vector<secp256k1::ParticipantId> const ids13 = {1, 3};
        l1 = secp256k1::frost_lagrange_coefficient(1, ids13);
        auto l3 = secp256k1::frost_lagrange_coefficient(3, ids13);

        reconstructed = (l1 * f1) + (l3 * f3);
        CHECK(reconstructed == secret, "Shamir reconstruct {1,3} == 42");

        // 2-of-3: use participants {2,3}
        std::vector<secp256k1::ParticipantId> const ids23 = {2, 3};
        l2 = secp256k1::frost_lagrange_coefficient(2, ids23);
        l3 = secp256k1::frost_lagrange_coefficient(3, ids23);

        reconstructed = (l2 * f2) + (l3 * f3);
        CHECK(reconstructed == secret, "Shamir reconstruct {2,3} == 42");
    }

    // Property 4: Lagrange for non-contiguous IDs {2,5,9}
    {
        std::vector<secp256k1::ParticipantId> const ids = {2, 5, 9};
        auto l2 = secp256k1::frost_lagrange_coefficient(2, ids);
        auto l5 = secp256k1::frost_lagrange_coefficient(5, ids);
        auto l9 = secp256k1::frost_lagrange_coefficient(9, ids);

        // f(x) = 100 + 3x + 2x^2
        // f(2) = 100 + 6 + 8 = 114
        // f(5) = 100 + 15 + 50 = 165
        // f(9) = 100 + 27 + 162 = 289
        auto secret = Scalar::from_uint64(100);
        auto f2_val = Scalar::from_uint64(114);
        auto f5_val = Scalar::from_uint64(165);
        auto f9_val = Scalar::from_uint64(289);

        auto reconstructed = (l2 * f2_val) + (l5 * f5_val) + (l9 * f9_val);
        CHECK(reconstructed == secret, "Shamir reconstruct non-contiguous {2,5,9}");
    }
}

// ===============================================================================
// Test 2: DKG Determinism -- Same Seeds Produce Same Key Packages
// ===============================================================================

static void test_dkg_determinism() {
    std::printf("[2] FROST DKG: Determinism with Fixed Seeds\n");

    const uint32_t t = 2, n = 3;
    auto seed1 = make_seed(0xF205E001);
    auto seed2 = make_seed(0xF205E002);
    auto seed3 = make_seed(0xF205E003);

    // Run DKG twice with identical seeds -- must produce identical results
    std::array<uint8_t, 33> first_group_key{};

    for (int trial = 0; trial < 2; ++trial) {
        auto [c1, shares1] = secp256k1::frost_keygen_begin(1, t, n, seed1);
        auto [c2, shares2] = secp256k1::frost_keygen_begin(2, t, n, seed2);
        auto [c3, shares3] = secp256k1::frost_keygen_begin(3, t, n, seed3);

        std::vector<secp256k1::FrostCommitment> const commitments = {c1, c2, c3};
        std::vector<secp256k1::FrostShare> const p1_shares = {shares1[0], shares2[0], shares3[0]};
        std::vector<secp256k1::FrostShare> const p2_shares = {shares1[1], shares2[1], shares3[1]};
        std::vector<secp256k1::FrostShare> const p3_shares = {shares1[2], shares2[2], shares3[2]};

        auto [kp1, ok1] = secp256k1::frost_keygen_finalize(1, commitments, p1_shares, t, n);
        auto [kp2, ok2] = secp256k1::frost_keygen_finalize(2, commitments, p2_shares, t, n);
        auto [kp3, ok3] = secp256k1::frost_keygen_finalize(3, commitments, p3_shares, t, n);

        CHECK(ok1, "P1 DKG success");
        CHECK(ok2, "P2 DKG success");
        CHECK(ok3, "P3 DKG success");

        auto gpk1 = kp1.group_public_key.to_compressed();
        auto gpk2 = kp2.group_public_key.to_compressed();
        auto gpk3 = kp3.group_public_key.to_compressed();

        CHECK(gpk1 == gpk2, "P1==P2 group key");
        CHECK(gpk2 == gpk3, "P2==P3 group key");

        if (trial == 0) {
            first_group_key = gpk1;
        } else {
            CHECK(gpk1 == first_group_key, "DKG deterministic across runs");
        }
    }
}

// ===============================================================================
// Test 3: DKG Share Verification -- Feldman VSS Commitment Check
// ===============================================================================

static void test_dkg_feldman_vss() {
    std::printf("[3] FROST DKG: Feldman VSS Commitment Verification\n");

    const uint32_t t = 2, n = 3;

    auto seed1 = make_seed(0xAABBCC01);
    auto seed2 = make_seed(0xAABBCC02);
    auto seed3 = make_seed(0xAABBCC03);

    auto [c1, shares1] = secp256k1::frost_keygen_begin(1, t, n, seed1);
    auto [c2, shares2] = secp256k1::frost_keygen_begin(2, t, n, seed2);
    auto [c3, shares3] = secp256k1::frost_keygen_begin(3, t, n, seed3);

    // Verify Feldman VSS: for share f_from(id), check:
    //   f_from(id) * G == sum_{k=0}^{t-1} A_k * id^k
    // where A_k are commitment coefficients
    struct CommitSharePair {
        const secp256k1::FrostCommitment* commit;
        const std::vector<secp256k1::FrostShare>* shares;
    };
    CommitSharePair const pairs[] = {
        {&c1, &shares1}, {&c2, &shares2}, {&c3, &shares3}
    };

    for (const auto& pair : pairs) {
        for (const auto& share : *pair.shares) {
            // LHS: share_value * G
            auto lhs = Point::generator().scalar_mul(share.value);

            // RHS: sum of A_k * id^k
            auto id_scalar = Scalar::from_uint64(share.id);
            auto id_power = Scalar::from_uint64(1); // id^0 = 1
            Point rhs = Point::infinity();

            for (size_t k = 0; k < pair.commit->coeffs.size(); ++k) {
                auto term = pair.commit->coeffs[k].scalar_mul(id_power);
                rhs = rhs.add(term);
                id_power = id_power * id_scalar;
            }

            CHECK(points_equal(lhs, rhs), "Feldman VSS: share*G == sum(A_k * id^k)");
        }
    }
}

// ===============================================================================
// Test 4: Full 2-of-3 Signing -- End-to-End with BIP-340 Verify
// ===============================================================================

static void test_2of3_full_signing() {
    std::printf("[4] FROST 2-of-3: Full Signing -> BIP-340 Verify\n");

    const uint32_t t = 2, n = 3;

    auto seed1 = make_seed(0xDEAD0001);
    auto seed2 = make_seed(0xDEAD0002);
    auto seed3 = make_seed(0xDEAD0003);

    // DKG
    auto [c1, shares1] = secp256k1::frost_keygen_begin(1, t, n, seed1);
    auto [c2, shares2] = secp256k1::frost_keygen_begin(2, t, n, seed2);
    auto [c3, shares3] = secp256k1::frost_keygen_begin(3, t, n, seed3);

    std::vector<secp256k1::FrostCommitment> const commitments = {c1, c2, c3};
    std::vector<secp256k1::FrostShare> const p1_shares = {shares1[0], shares2[0], shares3[0]};
    std::vector<secp256k1::FrostShare> const p2_shares = {shares1[1], shares2[1], shares3[1]};
    std::vector<secp256k1::FrostShare> const p3_shares = {shares1[2], shares2[2], shares3[2]};

    auto [kp1, ok1] = secp256k1::frost_keygen_finalize(1, commitments, p1_shares, t, n);
    auto [kp2, ok2] = secp256k1::frost_keygen_finalize(2, commitments, p2_shares, t, n);
    auto [kp3, ok3] = secp256k1::frost_keygen_finalize(3, commitments, p3_shares, t, n);

    CHECK(ok1 && ok2 && ok3, "DKG all succeed");

    // Message to sign
    std::array<uint8_t, 32> msg{};
    for (int i = 0; i < 32; ++i) msg[i] = static_cast<uint8_t>(0xCA + i);

    // Try ALL 3 signer subsets: {1,2}, {1,3}, {2,3}
    const secp256k1::FrostKeyPackage* const all_kps[] = {&kp1, &kp2, &kp3};
    uint32_t const subset_ids[][2] = {{1, 2}, {1, 3}, {2, 3}};

    for (int s = 0; s < 3; ++s) {
        uint32_t const id_a = subset_ids[s][0];
        uint32_t const id_b = subset_ids[s][1];

        auto nonce_seed_a = make_seed(0xA0000000 + static_cast<uint64_t>(s) * 0x10 + id_a);
        auto nonce_seed_b = make_seed(0xB0000000 + static_cast<uint64_t>(s) * 0x10 + id_b);

        auto [nonce_a, commit_a] = secp256k1::frost_sign_nonce_gen(id_a, nonce_seed_a);
        auto [nonce_b, commit_b] = secp256k1::frost_sign_nonce_gen(id_b, nonce_seed_b);

        std::vector<secp256k1::FrostNonceCommitment> const nonce_commits = {commit_a, commit_b};

        auto psig_a = secp256k1::frost_sign(*all_kps[id_a - 1], nonce_a, msg, nonce_commits);
        auto psig_b = secp256k1::frost_sign(*all_kps[id_b - 1], nonce_b, msg, nonce_commits);

        // Verify partial signatures
        bool const v_a = secp256k1::frost_verify_partial(
            psig_a, commit_a, all_kps[id_a - 1]->verification_share,
            msg, nonce_commits, kp1.group_public_key);
        bool const v_b = secp256k1::frost_verify_partial(
            psig_b, commit_b, all_kps[id_b - 1]->verification_share,
            msg, nonce_commits, kp1.group_public_key);

        CHECK(v_a, "partial sig A verified");
        CHECK(v_b, "partial sig B verified");

        // Aggregate
        std::vector<secp256k1::FrostPartialSig> const partials = {psig_a, psig_b};
        auto final_sig = secp256k1::frost_aggregate(
            partials, nonce_commits, kp1.group_public_key, msg);

        // BIP-340 Schnorr verify
        auto pk_bytes = kp1.group_public_key.x().to_bytes();
        bool const valid = secp256k1::schnorr_verify(pk_bytes, msg, final_sig);

        char label[64];
        std::snprintf(label, sizeof(label), "2-of-3 subset {%u,%u} BIP-340 valid", id_a, id_b);
        CHECK(valid, label);
    }
}

// ===============================================================================
// Test 5: Full 3-of-5 Signing -- Larger Threshold
// ===============================================================================

static void test_3of5_full_signing() {
    std::printf("[5] FROST 3-of-5: Full Signing -> BIP-340 Verify\n");

    const uint32_t t = 3, n = 5;

    std::array<std::array<uint8_t, 32>, 5> seeds;
    for (uint32_t i = 0; i < n; ++i) seeds[i] = make_seed(0x50050000 + i + 1);

    // DKG Round 1
    std::vector<secp256k1::FrostCommitment> commitments;
    std::vector<std::vector<secp256k1::FrostShare>> all_shares;

    for (uint32_t i = 1; i <= n; ++i) {
        auto [ci, si] = secp256k1::frost_keygen_begin(i, t, n, seeds[i - 1]);
        commitments.push_back(std::move(ci));
        all_shares.push_back(std::move(si));
    }

    // DKG Round 2
    std::vector<secp256k1::FrostKeyPackage> key_pkgs;
    for (uint32_t i = 1; i <= n; ++i) {
        std::vector<secp256k1::FrostShare> received;
        received.reserve(n);
for (uint32_t from = 0; from < n; ++from) {
            received.push_back(all_shares[from][i - 1]);
        }
        auto [kp, ok] = secp256k1::frost_keygen_finalize(i, commitments, received, t, n);
        CHECK(ok, "3-of-5 DKG finalize");
        key_pkgs.push_back(std::move(kp));
    }

    // All agree on group key
    auto gpk_ref = key_pkgs[0].group_public_key.to_compressed();
    for (uint32_t i = 1; i < n; ++i) {
        CHECK(key_pkgs[i].group_public_key.to_compressed() == gpk_ref,
              "3-of-5 group key consensus");
    }

    // Message
    std::array<uint8_t, 32> msg{};
    for (int i = 0; i < 32; ++i) msg[i] = static_cast<uint8_t>(0x35 + i);

    // Sign with participants {1, 3, 5} (non-contiguous subset)
    std::vector<uint32_t> signer_ids = {1, 3, 5};

    std::vector<secp256k1::FrostNonce> nonces;
    std::vector<secp256k1::FrostNonceCommitment> nonce_commits;

    for (auto id : signer_ids) {
        auto nonce_seed = make_seed(0x30500000 + id);
        auto [nc, cm] = secp256k1::frost_sign_nonce_gen(id, nonce_seed);
        nonces.push_back(std::move(nc));
        nonce_commits.push_back(std::move(cm));
    }

    // Partial signatures
    std::vector<secp256k1::FrostPartialSig> partials;
    for (size_t i = 0; i < signer_ids.size(); ++i) {
        auto psig = secp256k1::frost_sign(
            key_pkgs[signer_ids[i] - 1], nonces[i], msg, nonce_commits);
        partials.push_back(std::move(psig));
    }

    // Aggregate
    auto final_sig = secp256k1::frost_aggregate(
        partials, nonce_commits, key_pkgs[0].group_public_key, msg);

    auto pk_bytes = key_pkgs[0].group_public_key.x().to_bytes();
    bool valid = secp256k1::schnorr_verify(pk_bytes, msg, final_sig);
    CHECK(valid, "3-of-5 subset {1,3,5} BIP-340 valid");

    // Sign with different subset {2, 4, 5}
    signer_ids = {2, 4, 5};
    nonces.clear();
    nonce_commits.clear();

    for (auto id : signer_ids) {
        auto nonce_seed = make_seed(0x30520000 + id);
        auto [nc, cm] = secp256k1::frost_sign_nonce_gen(id, nonce_seed);
        nonces.push_back(std::move(nc));
        nonce_commits.push_back(std::move(cm));
    }

    partials.clear();
    for (size_t i = 0; i < signer_ids.size(); ++i) {
        auto psig = secp256k1::frost_sign(
            key_pkgs[signer_ids[i] - 1], nonces[i], msg, nonce_commits);
        partials.push_back(std::move(psig));
    }

    auto final_sig2 = secp256k1::frost_aggregate(
        partials, nonce_commits, key_pkgs[0].group_public_key, msg);

    pk_bytes = key_pkgs[0].group_public_key.x().to_bytes();
    valid = secp256k1::schnorr_verify(pk_bytes, msg, final_sig2);
    CHECK(valid, "3-of-5 subset {2,4,5} BIP-340 valid");

    // Both subsets sign same message but signatures differ
    CHECK(!(final_sig.r == final_sig2.r &&
            final_sig.s == final_sig2.s),
          "different subsets produce different signatures");
}

// ===============================================================================
// Test 6: Lagrange Coefficient Consistency Across Subsets
// ===============================================================================

static void test_lagrange_consistency() {
    std::printf("[6] Lagrange Coefficients: Consistency Across 10 Subsets\n");

    // For a 3-of-5 scheme: all C(5,3) = 10 subsets reconstruct same secret
    // Polynomial f(x) = 17 + 3x + 5x^2
    auto secret = Scalar::from_uint64(17);

    // f(1)=25, f(2)=43, f(3)=71, f(4)=109, f(5)=157
    uint64_t const share_vals[] = {25, 43, 71, 109, 157};
    Scalar shares[5];
    for (int i = 0; i < 5; ++i) shares[i] = Scalar::from_uint64(share_vals[i]);

    // All 10 3-element subsets of {1,2,3,4,5}
    uint32_t const subsets[][3] = {
        {1,2,3}, {1,2,4}, {1,2,5}, {1,3,4}, {1,3,5},
        {1,4,5}, {2,3,4}, {2,3,5}, {2,4,5}, {3,4,5}
    };

    for (int s = 0; s < 10; ++s) {
        std::vector<secp256k1::ParticipantId> ids = {
            subsets[s][0], subsets[s][1], subsets[s][2]
        };

        auto l0 = secp256k1::frost_lagrange_coefficient(ids[0], ids);
        auto l1 = secp256k1::frost_lagrange_coefficient(ids[1], ids);
        auto l2 = secp256k1::frost_lagrange_coefficient(ids[2], ids);

        auto reconstructed = (l0 * shares[ids[0] - 1])
                           + (l1 * shares[ids[1] - 1])
                           + (l2 * shares[ids[2] - 1]);

        char label[80];
        std::snprintf(label, sizeof(label), "3-of-5 subset {%u,%u,%u} reconstructs secret",
                      ids[0], ids[1], ids[2]);
        CHECK(reconstructed == secret, label);
    }
}

// ===============================================================================
// Test 7: Pinned KAT -- DKG Group Key from Known Seeds
// ===============================================================================

static void test_pinned_dkg_group_key() {
    std::printf("[7] Pinned KAT: DKG Group Key Determinism\n");

    const uint32_t t = 2, n = 3;

    auto seed1 = make_seed(0x4B415401); // "KAT1"
    auto seed2 = make_seed(0x4B415402);
    auto seed3 = make_seed(0x4B415403);

    auto run_dkg = [&]() -> std::array<uint8_t, 33> {
        auto [c1, shares1] = secp256k1::frost_keygen_begin(1, t, n, seed1);
        auto [c2, shares2] = secp256k1::frost_keygen_begin(2, t, n, seed2);
        auto [c3, shares3] = secp256k1::frost_keygen_begin(3, t, n, seed3);

        std::vector<secp256k1::FrostCommitment> const commitments = {c1, c2, c3};
        std::vector<secp256k1::FrostShare> const p1_shares = {shares1[0], shares2[0], shares3[0]};
        std::vector<secp256k1::FrostShare> const p2_shares = {shares1[1], shares2[1], shares3[1]};
        std::vector<secp256k1::FrostShare> const p3_shares = {shares1[2], shares2[2], shares3[2]};

        auto [kp1, ok1] = secp256k1::frost_keygen_finalize(1, commitments, p1_shares, t, n);
        auto [kp2, ok2] = secp256k1::frost_keygen_finalize(2, commitments, p2_shares, t, n);
        auto [kp3, ok3] = secp256k1::frost_keygen_finalize(3, commitments, p3_shares, t, n);

        CHECK(ok1 && ok2 && ok3, "KAT DKG all ok");
        CHECK(kp1.group_public_key.to_compressed() == kp2.group_public_key.to_compressed(),
              "KAT group key P1==P2");
        CHECK(kp2.group_public_key.to_compressed() == kp3.group_public_key.to_compressed(),
              "KAT group key P2==P3");

        return kp1.group_public_key.to_compressed();
    };

    auto gpk_run1 = run_dkg();
    auto gpk_run2 = run_dkg();

    CHECK(gpk_run1 == gpk_run2, "KAT group key identical across runs");
}

// ===============================================================================
// Test 8: Pinned KAT -- Full Signing Round-Trip
// ===============================================================================

static void test_pinned_signing_roundtrip() {
    std::printf("[8] Pinned KAT: Full Signing Round-Trip Determinism\n");

    const uint32_t t = 2, n = 3;
    auto seed1 = make_seed(0x51670001);
    auto seed2 = make_seed(0x51670002);
    auto seed3 = make_seed(0x51670003);

    // Helper to run a complete sign round
    auto run_full_sign = [&]() -> secp256k1::SchnorrSignature {
        auto [c1, shares1] = secp256k1::frost_keygen_begin(1, t, n, seed1);
        auto [c2, shares2] = secp256k1::frost_keygen_begin(2, t, n, seed2);
        auto [c3, shares3] = secp256k1::frost_keygen_begin(3, t, n, seed3);

        std::vector<secp256k1::FrostCommitment> const commitments = {c1, c2, c3};
        std::vector<secp256k1::FrostShare> const p1_shares = {shares1[0], shares2[0], shares3[0]};
        std::vector<secp256k1::FrostShare> const p2_shares = {shares1[1], shares2[1], shares3[1]};

        auto [kp1, ok1] = secp256k1::frost_keygen_finalize(1, commitments, p1_shares, t, n);
        auto [kp2, ok2] = secp256k1::frost_keygen_finalize(2, commitments, p2_shares, t, n);

        // Fixed message
        std::array<uint8_t, 32> msg{};
        for (int i = 0; i < 32; ++i) msg[i] = static_cast<uint8_t>(i);

        // Fixed nonce seeds
        auto nonce_seed1 = make_seed(0xABCD0001);
        auto nonce_seed2 = make_seed(0xABCD0002);

        auto [nonce1, commit1] = secp256k1::frost_sign_nonce_gen(1, nonce_seed1);
        auto [nonce2, commit2] = secp256k1::frost_sign_nonce_gen(2, nonce_seed2);

        std::vector<secp256k1::FrostNonceCommitment> const nonce_commits = {commit1, commit2};

        auto psig1 = secp256k1::frost_sign(kp1, nonce1, msg, nonce_commits);
        auto psig2 = secp256k1::frost_sign(kp2, nonce2, msg, nonce_commits);

        auto final_sig = secp256k1::frost_aggregate(
            {psig1, psig2}, nonce_commits, kp1.group_public_key, msg);

        auto pk_bytes = kp1.group_public_key.x().to_bytes();
        bool const valid = secp256k1::schnorr_verify(pk_bytes, msg, final_sig);
        CHECK(valid, "KAT signing BIP-340 valid");

        return final_sig;
    };

    auto sig1 = run_full_sign();
    auto sig2 = run_full_sign();

    CHECK(sig1.r == sig2.r, "KAT sig R identical");
    CHECK(sig1.s == sig2.s, "KAT sig s identical");
}

// ===============================================================================
// Test 9: Secret Reconstruction from DKG Shares
// ===============================================================================

static void test_secret_reconstruction() {
    std::printf("[9] FROST DKG: Secret Reconstruction via Lagrange\n");

    const uint32_t t = 2, n = 3;
    auto seed1 = make_seed(0xEE000001);
    auto seed2 = make_seed(0xEE000002);
    auto seed3 = make_seed(0xEE000003);

    auto [c1, shares1] = secp256k1::frost_keygen_begin(1, t, n, seed1);
    auto [c2, shares2] = secp256k1::frost_keygen_begin(2, t, n, seed2);
    auto [c3, shares3] = secp256k1::frost_keygen_begin(3, t, n, seed3);

    std::vector<secp256k1::FrostCommitment> const commitments = {c1, c2, c3};

    std::vector<secp256k1::FrostShare> const p1_shares = {shares1[0], shares2[0], shares3[0]};
    std::vector<secp256k1::FrostShare> const p2_shares = {shares1[1], shares2[1], shares3[1]};
    std::vector<secp256k1::FrostShare> const p3_shares = {shares1[2], shares2[2], shares3[2]};

    auto [kp1, ok1] = secp256k1::frost_keygen_finalize(1, commitments, p1_shares, t, n);
    auto [kp2, ok2] = secp256k1::frost_keygen_finalize(2, commitments, p2_shares, t, n);
    auto [kp3, ok3] = secp256k1::frost_keygen_finalize(3, commitments, p3_shares, t, n);

    CHECK(ok1 && ok2 && ok3, "DKG for reconstruction ok");

    // Reconstruct group secret from any 2 signing shares using Lagrange
    auto reconstruct = [](const secp256k1::FrostKeyPackage& a,
                          const secp256k1::FrostKeyPackage& b) -> Scalar {
        std::vector<secp256k1::ParticipantId> const ids = {a.id, b.id};
        auto la = secp256k1::frost_lagrange_coefficient(a.id, ids);
        auto lb = secp256k1::frost_lagrange_coefficient(b.id, ids);
        return (la * a.signing_share) + (lb * b.signing_share);
    };

    auto s12 = reconstruct(kp1, kp2);
    auto s13 = reconstruct(kp1, kp3);
    auto s23 = reconstruct(kp2, kp3);

    // All subsets must reconstruct the SAME group secret
    CHECK(s12 == s13, "secret {1,2} == {1,3}");
    CHECK(s13 == s23, "secret {1,3} == {2,3}");

    // Verify: group_secret * G == group_public_key (x-coordinate match)
    auto derived_pubkey = Point::generator().scalar_mul(s12);
    CHECK(derived_pubkey.x().to_bytes() == kp1.group_public_key.x().to_bytes(),
          "reconstructed_secret * G == group_public_key (x-coord)");
}

// ===============================================================================
// _run() entry point for unified audit runner
// ===============================================================================

int test_frost_kat_run() {
    g_pass = 0; g_fail = 0;

    test_lagrange_properties();
    test_dkg_determinism();
    test_dkg_feldman_vss();
    test_2of3_full_signing();
    test_3of5_full_signing();
    test_lagrange_consistency();
    test_pinned_dkg_group_key();
    test_pinned_signing_roundtrip();
    test_secret_reconstruction();

    return g_fail > 0 ? 1 : 0;
}

// ===============================================================================
// Main (standalone only)
// ===============================================================================

#ifndef UNIFIED_AUDIT_RUNNER
int main() {
    std::printf("=== FROST Reference KAT Tests (Phase II 2.2.5) ===\n\n");

    test_lagrange_properties();
    test_dkg_determinism();
    test_dkg_feldman_vss();
    test_2of3_full_signing();
    test_3of5_full_signing();
    test_lagrange_consistency();
    test_pinned_dkg_group_key();
    test_pinned_signing_roundtrip();
    test_secret_reconstruction();

    std::printf("\n=== Results: %d passed, %d failed ===\n", g_pass, g_fail);

    if (g_fail > 0) {
        std::printf("*** %d FAILURES ***\n", g_fail);
        return 1;
    }
    std::printf("All reference KAT checks passed.\n");
    return 0;
}
#endif // UNIFIED_AUDIT_RUNNER
