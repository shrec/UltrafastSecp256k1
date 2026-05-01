// ============================================================================
// Test: Zero-Knowledge Proof Layer
// ============================================================================
// Tests for Schnorr knowledge proofs, DLEQ proofs, and Bulletproof range proofs.
// ============================================================================

#include "secp256k1/zk.hpp"
#include "secp256k1/pedersen.hpp"
#include "secp256k1/sha256.hpp"
#include "secp256k1/scalar.hpp"
#include "secp256k1/point.hpp"
#include "secp256k1/ct/point.hpp"

#include <cstdio>
#include <cstring>
#include <array>

using namespace secp256k1;
using fast::Scalar;
using fast::Point;

static int tests_run = 0;
static int tests_passed = 0;

#define CHECK(cond, msg) do { \
    ++tests_run; \
    if (cond) { ++tests_passed; std::printf("  [PASS] %s\n", msg); } \
    else { std::printf("  [FAIL] %s\n", msg); } \
} while(0)


// ============================================================================
// Knowledge Proof Tests
// ============================================================================

static void test_knowledge_proof_basic() {
    std::printf("\n=== Knowledge Proof: Basic ===\n");

    auto secret = Scalar::from_hex(
        "e8f32e723decf4051aefac8e2c93c9c5b214313817cdb01a1494b917c8436b35");
    auto pubkey = Point::generator().scalar_mul(secret);

    std::array<std::uint8_t, 32> msg{};
    msg[0] = 0x42;

    std::array<std::uint8_t, 32> aux{};
    aux[0] = 0x01;

    auto proof = zk::knowledge_prove(secret, pubkey, msg, aux);
    CHECK(zk::knowledge_verify(proof, pubkey, msg), "valid_proof_verifies");
}

static void test_knowledge_proof_wrong_key() {
    std::printf("\n=== Knowledge Proof: Wrong Key ===\n");

    auto secret = Scalar::from_uint64(42);
    auto pubkey = Point::generator().scalar_mul(secret);
    auto wrong_pubkey = Point::generator().scalar_mul(Scalar::from_uint64(43));

    const std::array<std::uint8_t, 32> msg{};
    std::array<std::uint8_t, 32> aux{};
    aux[0] = 0x02;

    auto proof = zk::knowledge_prove(secret, pubkey, msg, aux);

    CHECK(zk::knowledge_verify(proof, pubkey, msg), "correct_key_passes");
    CHECK(!zk::knowledge_verify(proof, wrong_pubkey, msg), "wrong_key_fails");
}

static void test_knowledge_proof_wrong_msg() {
    std::printf("\n=== Knowledge Proof: Wrong Message ===\n");

    auto secret = Scalar::from_uint64(12345);
    auto pubkey = Point::generator().scalar_mul(secret);

    std::array<std::uint8_t, 32> msg1{};
    msg1[0] = 0xAA;
    std::array<std::uint8_t, 32> msg2{};
    msg2[0] = 0xBB;
    std::array<std::uint8_t, 32> aux{};
    aux[0] = 0x03;

    auto proof = zk::knowledge_prove(secret, pubkey, msg1, aux);

    CHECK(zk::knowledge_verify(proof, pubkey, msg1), "correct_msg_passes");
    CHECK(!zk::knowledge_verify(proof, pubkey, msg2), "wrong_msg_fails");
}

static void test_knowledge_proof_serialization() {
    std::printf("\n=== Knowledge Proof: Serialization ===\n");

    auto secret = Scalar::from_uint64(999);
    auto pubkey = Point::generator().scalar_mul(secret);

    const std::array<std::uint8_t, 32> msg{};
    std::array<std::uint8_t, 32> aux{};
    aux[0] = 0x04;

    auto proof = zk::knowledge_prove(secret, pubkey, msg, aux);
    auto serialized = proof.serialize();

    zk::KnowledgeProof deserialized{};
    const bool ok = zk::KnowledgeProof::deserialize(serialized.data(), deserialized);
    CHECK(ok, "deserialization_succeeds");

    CHECK(zk::knowledge_verify(deserialized, pubkey, msg), "deserialized_proof_verifies");
}

static void test_knowledge_proof_custom_base() {
    std::printf("\n=== Knowledge Proof: Custom Base ===\n");

    auto secret = Scalar::from_uint64(777);
    // Use H as base instead of G
    const auto& H = pedersen_generator_H();
    auto point = H.scalar_mul(secret);

    const std::array<std::uint8_t, 32> msg{};
    std::array<std::uint8_t, 32> aux{};
    aux[0] = 0x05;

    auto proof = zk::knowledge_prove_base(secret, point, H, msg, aux);
    CHECK(zk::knowledge_verify_base(proof, point, H, msg), "custom_base_verifies");

    // Should not verify against standard generator
    CHECK(!zk::knowledge_verify(proof, point, msg), "wrong_base_fails");
}

static void test_knowledge_proof_deterministic() {
    std::printf("\n=== Knowledge Proof: Deterministic ===\n");

    auto secret = Scalar::from_uint64(42);
    auto pubkey = Point::generator().scalar_mul(secret);

    const std::array<std::uint8_t, 32> msg{};
    std::array<std::uint8_t, 32> aux{};
    aux[0] = 0x06;

    auto proof1 = zk::knowledge_prove(secret, pubkey, msg, aux);
    auto proof2 = zk::knowledge_prove(secret, pubkey, msg, aux);

    // Same inputs should produce same proof
    auto ser1 = proof1.serialize();
    auto ser2 = proof2.serialize();
    CHECK(ser1 == ser2, "deterministic_proofs_match");
}


// ============================================================================
// DLEQ Proof Tests
// ============================================================================

static void test_dleq_basic() {
    std::printf("\n=== DLEQ Proof: Basic ===\n");

    auto secret = Scalar::from_hex(
        "e8f32e723decf4051aefac8e2c93c9c5b214313817cdb01a1494b917c8436b35");
    auto G = Point::generator();
    const auto& H = pedersen_generator_H();

    auto P = G.scalar_mul(secret);   // P = x*G
    auto Q = H.scalar_mul(secret);   // Q = x*H

    std::array<std::uint8_t, 32> aux{};
    aux[0] = 0x07;

    auto proof = zk::dleq_prove(secret, G, H, P, Q, aux);
    CHECK(zk::dleq_verify(proof, G, H, P, Q), "valid_dleq_verifies");
}

static void test_dleq_wrong_secret() {
    std::printf("\n=== DLEQ Proof: Wrong Secret ===\n");

    auto x = Scalar::from_uint64(100);
    auto y = Scalar::from_uint64(200); // different secret
    auto G = Point::generator();
    const auto& H = pedersen_generator_H();

    auto P = G.scalar_mul(x);
    auto Q = H.scalar_mul(y); // Q = y*H != x*H

    std::array<std::uint8_t, 32> aux{};
    aux[0] = 0x08;

    // Prove with x (but Q was made with y, so it should still "prove" x)
    // The proof proves log_G(P) == log_H(Q), which is FALSE here
    auto proof = zk::dleq_prove(x, G, H, P, Q, aux);
    // But the prover used x, so R2 = k*H, and s = k + e*x
    // Verifier checks: s*H == R2 + e*Q => k*H + e*x*H == k*H + e*y*H
    // This fails because x != y
    CHECK(!zk::dleq_verify(proof, G, H, P, Q), "wrong_discrete_log_fails");
}

static void test_dleq_serialization() {
    std::printf("\n=== DLEQ Proof: Serialization ===\n");

    auto secret = Scalar::from_uint64(42);
    auto G = Point::generator();
    const auto& H = pedersen_generator_H();
    auto P = G.scalar_mul(secret);
    auto Q = H.scalar_mul(secret);

    std::array<std::uint8_t, 32> aux{};
    aux[0] = 0x09;

    auto proof = zk::dleq_prove(secret, G, H, P, Q, aux);
    auto serialized = proof.serialize();

    zk::DLEQProof deserialized{};
    zk::DLEQProof::deserialize(serialized.data(), deserialized);

    CHECK(zk::dleq_verify(deserialized, G, H, P, Q), "serialized_dleq_verifies");
}

static void test_dleq_deterministic() {
    std::printf("\n=== DLEQ Proof: Deterministic ===\n");

    auto secret = Scalar::from_uint64(42);
    auto G = Point::generator();
    const auto& H = pedersen_generator_H();
    auto P = G.scalar_mul(secret);
    auto Q = H.scalar_mul(secret);

    std::array<std::uint8_t, 32> aux{};
    aux[0] = 0x0A;

    auto proof1 = zk::dleq_prove(secret, G, H, P, Q, aux);
    auto proof2 = zk::dleq_prove(secret, G, H, P, Q, aux);

    auto ser1 = proof1.serialize();
    auto ser2 = proof2.serialize();
    CHECK(ser1 == ser2, "deterministic_dleq_match");
}


// ============================================================================
// Bulletproof Range Proof Tests
// ============================================================================

static void test_range_proof_generators() {
    std::printf("\n=== Range Proof: Generator Vectors ===\n");

    const auto& gens = zk::get_generator_vectors();

    // All generators should be valid non-infinity points
    bool all_valid = true;
    for (std::size_t i = 0; i < zk::RANGE_PROOF_BITS; ++i) {
        if (gens.G[i].is_infinity() || gens.H[i].is_infinity()) {
            all_valid = false;
            break;
        }
    }
    CHECK(all_valid, "all_generators_valid");

    // G[0] != H[0] (different generator series)
    auto g0 = gens.G[0].to_compressed();
    auto h0 = gens.H[0].to_compressed();
    CHECK(g0 != h0, "G_and_H_generators_differ");

    // All generators should be distinct
    auto g1 = gens.G[1].to_compressed();
    CHECK(g0 != g1, "G0_differs_from_G1");
}

static void test_range_proof_basic() {
    std::printf("\n=== Range Proof: Basic (value=42) ===\n");

    const std::uint64_t value = 42;
    auto blinding = Scalar::from_uint64(12345);
    auto commitment = pedersen_commit(Scalar::from_uint64(value), blinding);

    std::array<std::uint8_t, 32> aux{};
    aux[0] = 0x0B;

    auto proof = zk::range_prove(value, blinding, commitment, aux);
    CHECK(zk::range_verify(commitment, proof), "range_proof_42_valid");
}

static void test_range_proof_zero() {
    std::printf("\n=== Range Proof: Edge Case (value=0) ===\n");

    const std::uint64_t value = 0;
    auto blinding = Scalar::from_uint64(99999);
    auto commitment = pedersen_commit(Scalar::from_uint64(value), blinding);

    std::array<std::uint8_t, 32> aux{};
    aux[0] = 0x0C;

    auto proof = zk::range_prove(value, blinding, commitment, aux);
    CHECK(zk::range_verify(commitment, proof), "range_proof_zero_valid");
}

static void test_range_proof_max() {
    std::printf("\n=== Range Proof: Edge Case (value=2^64-1) ===\n");

    const std::uint64_t value = UINT64_MAX;
    auto blinding = Scalar::from_uint64(77777);
    auto commitment = pedersen_commit(Scalar::from_uint64(value), blinding);

    std::array<std::uint8_t, 32> aux{};
    aux[0] = 0x0D;

    auto proof = zk::range_prove(value, blinding, commitment, aux);
    CHECK(zk::range_verify(commitment, proof), "range_proof_max64_valid");
}

static void test_range_proof_wrong_commitment() {
    std::printf("\n=== Range Proof: Wrong Commitment ===\n");

    const std::uint64_t value = 100;
    auto blinding = Scalar::from_uint64(11111);
    auto commitment = pedersen_commit(Scalar::from_uint64(value), blinding);
    auto wrong_commitment = pedersen_commit(Scalar::from_uint64(200), blinding);

    std::array<std::uint8_t, 32> aux{};
    aux[0] = 0x0E;

    auto proof = zk::range_prove(value, blinding, commitment, aux);
    CHECK(zk::range_verify(commitment, proof), "correct_commitment_passes");
    CHECK(!zk::range_verify(wrong_commitment, proof), "wrong_commitment_fails");
}

static void test_range_proof_deterministic() {
    std::printf("\n=== Range Proof: Deterministic ===\n");

    const std::uint64_t value = 42;
    auto blinding = Scalar::from_uint64(12345);
    auto commitment = pedersen_commit(Scalar::from_uint64(value), blinding);

    std::array<std::uint8_t, 32> aux{};
    aux[0] = 0x0F;

    auto proof1 = zk::range_prove(value, blinding, commitment, aux);
    auto proof2 = zk::range_prove(value, blinding, commitment, aux);

    // Compare a few key fields
    auto a1 = proof1.a.to_bytes();
    auto a2 = proof2.a.to_bytes();
    CHECK(a1 == a2, "deterministic_range_proofs_match");
}


// ============================================================================
// Batch Operations Tests
// ============================================================================

static void test_batch_commit() {
    std::printf("\n=== Batch Operations: batch_commit ===\n");

    Scalar values[4] = {
        Scalar::from_uint64(10),
        Scalar::from_uint64(20),
        Scalar::from_uint64(30),
        Scalar::from_uint64(40)
    };
    Scalar blindings[4] = {
        Scalar::from_uint64(111),
        Scalar::from_uint64(222),
        Scalar::from_uint64(333),
        Scalar::from_uint64(444)
    };

    PedersenCommitment batch_results[4];
    zk::batch_commit(values, blindings, batch_results, 4);

    // Verify each individually
    bool all_valid = true;
    for (int i = 0; i < 4; ++i) {
        auto individual = pedersen_commit(values[i], blindings[i]);
        if (batch_results[i].to_compressed() != individual.to_compressed()) {
            all_valid = false;
            break;
        }
    }
    CHECK(all_valid, "batch_commit_matches_individual");
}


// ============================================================================
// Entry points
// ============================================================================

static int run_all_zk_tests() {
    std::printf("=== ZK Proof Layer Tests ===\n");

    // Knowledge Proofs
    test_knowledge_proof_basic();
    test_knowledge_proof_wrong_key();
    test_knowledge_proof_wrong_msg();
    test_knowledge_proof_serialization();
    test_knowledge_proof_custom_base();
    test_knowledge_proof_deterministic();

    // DLEQ Proofs
    test_dleq_basic();
    test_dleq_wrong_secret();
    test_dleq_serialization();
    test_dleq_deterministic();

    // Bulletproof Range Proofs
    test_range_proof_generators();
    test_range_proof_basic();
    test_range_proof_zero();
    test_range_proof_max();
    test_range_proof_wrong_commitment();
    test_range_proof_deterministic();

    // Batch Operations
    test_batch_commit();

    std::printf("\n=== Results: %d/%d passed ===\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}

int test_zk_run() { return run_all_zk_tests(); }

#ifdef STANDALONE_TEST
int main() { return run_all_zk_tests(); }
#endif
