// ============================================================================
// ZK/Pedersen/Adaptor Fault-Injection Audit Test
// ============================================================================
// Validates that ZK proofs, Pedersen commitments, and Adaptor signatures
// correctly reject corrupted / bit-flipped inputs.
//
// Existing fault injection tests cover ECDSA/Schnorr/MuSig2/FROST.
// This fills the gap for:
//   1. Pedersen commitment — bit-flip blinding / value
//   2. Knowledge proof — bit-flip proof bytes
//   3. DLEQ proof — bit-flip proof bytes
//   4. Range proof — corrupted commitment vs proof mismatch
//   5. Schnorr adaptor — bit-flip pre-signature fields
//   6. ECDSA adaptor — bit-flip pre-signature fields
//   7. Adaptor extraction — wrong signature → wrong secret
// ============================================================================

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <array>
#include <random>

#include "secp256k1/field.hpp"
#include "secp256k1/scalar.hpp"
#include "secp256k1/point.hpp"
#include "secp256k1/ecdsa.hpp"
#include "secp256k1/schnorr.hpp"
#include "secp256k1/pedersen.hpp"
#include "secp256k1/zk.hpp"
#include "secp256k1/adaptor.hpp"
#include "secp256k1/sanitizer_scale.hpp"

using namespace secp256k1;
using namespace secp256k1::fast;
using namespace secp256k1::zk;

static int g_pass = 0, g_fail = 0;
static const char* g_section = "";

#include "audit_check.hpp"

static std::mt19937_64 rng(0xFA17'1A1EULL);

static Scalar random_scalar() {
    std::array<uint8_t, 32> out{};
    for (int i = 0; i < 4; ++i) {
        uint64_t v = rng();
        std::memcpy(out.data() + static_cast<std::size_t>(i) * 8, &v, 8);
    }
    for (;;) {
        auto s = Scalar::from_bytes(out);
        if (!s.is_zero()) return s;
        out[31] ^= 0x01;
    }
}

static std::array<uint8_t, 32> random_bytes32() {
    std::array<uint8_t, 32> out{};
    for (int i = 0; i < 4; ++i) {
        uint64_t v = rng();
        std::memcpy(out.data() + static_cast<std::size_t>(i) * 8, &v, 8);
    }
    return out;
}

// ============================================================================
// 1. Pedersen commitment: bit-flip blinding → verify fails
// ============================================================================
static void test_pedersen_bitflip() {
    g_section = "pedersen_bitflip";
    printf("[1] Pedersen commitment: bit-flip blinding -> verify fails\n");

    const int TRIALS = SCALED(50, 10);
    int detected = 0;

    for (int i = 0; i < TRIALS; ++i) {
        Scalar value = random_scalar();
        Scalar blinding = random_scalar();
        auto C = pedersen_commit(value, blinding);

        // Original must verify
        CHECK(C.verify(value, blinding), "Original commitment must verify");

        // Corrupt blinding by bit-flip
        auto bl_bytes = blinding.to_bytes();
        bl_bytes[static_cast<size_t>(i % 32)] ^= static_cast<uint8_t>(1u << (i % 8));
        Scalar bad_blind = Scalar::from_bytes(bl_bytes);

        bool v = C.verify(value, bad_blind);
        CHECK(!v, "Commitment with flipped blinding must fail");
        if (!v) ++detected;
    }

    CHECK(detected == TRIALS, "All bit-flips detected in Pedersen");
    printf("    -> %d/%d detected\n\n", detected, TRIALS);
}

// ============================================================================
// 2. Pedersen commitment: wrong value → verify fails
// ============================================================================
static void test_pedersen_wrong_value() {
    g_section = "pedersen_wrong_val";
    printf("[2] Pedersen commitment: wrong value -> verify fails\n");

    const int TRIALS = SCALED(50, 10);
    int detected = 0;

    for (int i = 0; i < TRIALS; ++i) {
        Scalar value = random_scalar();
        Scalar blinding = random_scalar();
        auto C = pedersen_commit(value, blinding);

        // Different value
        Scalar wrong_val = random_scalar();
        bool v = C.verify(wrong_val, blinding);
        CHECK(!v, "Commitment with wrong value must fail");
        if (!v) ++detected;
    }

    CHECK(detected == TRIALS, "All wrong-value attempts detected");
    printf("    -> %d/%d detected\n\n", detected, TRIALS);
}

// ============================================================================
// 3. Knowledge proof: bit-flip proof → verify fails
// ============================================================================
static void test_knowledge_proof_bitflip() {
    g_section = "zk_knowledge_flip";
    printf("[3] Knowledge proof: bit-flip -> verify fails\n");

    const int TRIALS = SCALED(30, 5);
    int detected = 0;

    for (int i = 0; i < TRIALS; ++i) {
        Scalar secret = random_scalar();
        Point pubkey = Point::generator().scalar_mul(secret);
        auto msg = random_bytes32();
        auto aux = random_bytes32();

        auto proof = knowledge_prove(secret, pubkey, msg, aux);

        // Verify original
        CHECK(knowledge_verify(proof, pubkey, msg), "Original proof must verify");

        // Flip a bit in the proof's s scalar
        auto s_bytes = proof.s.to_bytes();
        s_bytes[static_cast<size_t>(i % 32)] ^= static_cast<uint8_t>(1u << (i % 8));
        KnowledgeProof bad = proof;
        bad.s = Scalar::from_bytes(s_bytes);

        bool v = knowledge_verify(bad, pubkey, msg);
        CHECK(!v, "Knowledge proof with flipped s must fail");
        if (!v) ++detected;
    }

    CHECK(detected == TRIALS, "All knowledge proof bit-flips detected");
    printf("    -> %d/%d detected\n\n", detected, TRIALS);
}

// ============================================================================
// 4. Knowledge proof: wrong pubkey → verify fails
// ============================================================================
static void test_knowledge_proof_wrong_key() {
    g_section = "zk_knowledge_wrongkey";
    printf("[4] Knowledge proof: wrong pubkey -> verify fails\n");

    const int TRIALS = SCALED(30, 5);
    int detected = 0;

    for (int i = 0; i < TRIALS; ++i) {
        Scalar secret = random_scalar();
        Point pubkey = Point::generator().scalar_mul(secret);
        auto msg = random_bytes32();
        auto aux = random_bytes32();

        auto proof = knowledge_prove(secret, pubkey, msg, aux);

        // Use a different pubkey for verification
        Scalar other = random_scalar();
        Point wrong_pk = Point::generator().scalar_mul(other);

        bool v = knowledge_verify(proof, wrong_pk, msg);
        CHECK(!v, "Knowledge proof against wrong key must fail");
        if (!v) ++detected;
    }

    CHECK(detected == TRIALS, "All wrong-key attempts detected");
    printf("    -> %d/%d detected\n\n", detected, TRIALS);
}

// ============================================================================
// 5. DLEQ proof: bit-flip proof → verify fails
// ============================================================================
static void test_dleq_bitflip() {
    g_section = "zk_dleq_flip";
    printf("[5] DLEQ proof: bit-flip -> verify fails\n");

    const int TRIALS = SCALED(20, 5);
    int detected = 0;

    for (int i = 0; i < TRIALS; ++i) {
        Scalar secret = random_scalar();
        Point G = Point::generator();
        // Create a different base H
        Scalar h_scalar = random_scalar();
        Point H = G.scalar_mul(h_scalar);

        Point P = G.scalar_mul(secret);
        Point Q = H.scalar_mul(secret);
        auto aux = random_bytes32();

        auto proof = dleq_prove(secret, G, H, P, Q, aux);

        // Verify original
        CHECK(dleq_verify(proof, G, H, P, Q), "Original DLEQ proof must verify");

        // Flip a bit in challenge e
        auto e_bytes = proof.e.to_bytes();
        e_bytes[static_cast<size_t>(i % 32)] ^= static_cast<uint8_t>(1u << (i % 8));
        DLEQProof bad = proof;
        bad.e = Scalar::from_bytes(e_bytes);

        bool v = dleq_verify(bad, G, H, P, Q);
        CHECK(!v, "DLEQ proof with flipped challenge must fail");
        if (!v) ++detected;
    }

    CHECK(detected == TRIALS, "All DLEQ bit-flips detected");
    printf("    -> %d/%d detected\n\n", detected, TRIALS);
}

// ============================================================================
// 6. DLEQ proof: mismatched points → verify fails
// ============================================================================
static void test_dleq_mismatch() {
    g_section = "zk_dleq_mismatch";
    printf("[6] DLEQ proof: mismatched points -> verify fails\n");

    const int TRIALS = SCALED(20, 5);
    int detected = 0;

    for (int i = 0; i < TRIALS; ++i) {
        Scalar secret = random_scalar();
        Point G = Point::generator();
        Scalar h_scalar = random_scalar();
        Point H = G.scalar_mul(h_scalar);

        Point P = G.scalar_mul(secret);
        Point Q = H.scalar_mul(secret);
        auto aux = random_bytes32();

        auto proof = dleq_prove(secret, G, H, P, Q, aux);

        // Use wrong Q (different secret for H)
        Scalar other = random_scalar();
        Point wrong_Q = H.scalar_mul(other);

        bool v = dleq_verify(proof, G, H, P, wrong_Q);
        CHECK(!v, "DLEQ with mismatched Q must fail");
        if (!v) ++detected;
    }

    CHECK(detected == TRIALS, "All DLEQ mismatches detected");
    printf("    -> %d/%d detected\n\n", detected, TRIALS);
}

// ============================================================================
// 7. Range proof: wrong commitment → verify fails
// ============================================================================
static void test_range_proof_wrong_commitment() {
    g_section = "zk_range_wrong";
    printf("[7] Range proof: wrong commitment -> verify fails\n");

    const int TRIALS = SCALED(5, 2);
    int detected = 0;

    for (int i = 0; i < TRIALS; ++i) {
        uint64_t value = static_cast<uint64_t>(rng() & 0xFFFFFFFF);
        Scalar blinding = random_scalar();
        Scalar val_scalar = Scalar::from_limbs({value, 0, 0, 0});
        auto C = pedersen_commit(val_scalar, blinding);
        auto aux = random_bytes32();

        auto proof = range_prove(value, blinding, C, aux);

        // Verify original
        CHECK(range_verify(C, proof), "Original range proof must verify");

        // Try with a different commitment (different blinding)
        Scalar wrong_blind = random_scalar();
        auto bad_C = pedersen_commit(val_scalar, wrong_blind);

        bool v = range_verify(bad_C, proof);
        CHECK(!v, "Range proof against wrong commitment must fail");
        if (!v) ++detected;
    }

    CHECK(detected == TRIALS, "All wrong-commitment range proofs detected");
    printf("    -> %d/%d detected\n\n", detected, TRIALS);
}

// ============================================================================
// 8. Schnorr adaptor: bit-flip pre-sig → verify fails
// ============================================================================
static void test_schnorr_adaptor_bitflip() {
    g_section = "adaptor_schnorr_flip";
    printf("[8] Schnorr adaptor: bit-flip pre-sig -> verify fails\n");

    const int TRIALS = SCALED(30, 5);
    int detected = 0;

    for (int i = 0; i < TRIALS; ++i) {
        Scalar sk = random_scalar();
        Scalar adaptor_secret = random_scalar();
        Point T = Point::generator().scalar_mul(adaptor_secret);
        auto msg = random_bytes32();
        auto aux = random_bytes32();
        auto pk_x = Point::generator().scalar_mul(sk).x().to_bytes();

        auto pre_sig = schnorr_adaptor_sign(sk, msg, T, aux);

        // Verify original
        CHECK(schnorr_adaptor_verify(pre_sig, pk_x, msg, T),
              "Original adaptor pre-sig must verify");

        // Flip bit in s_hat
        auto s_bytes = pre_sig.s_hat.to_bytes();
        s_bytes[static_cast<size_t>(i % 32)] ^= static_cast<uint8_t>(1u << (i % 8));
        SchnorrAdaptorSig bad = pre_sig;
        bad.s_hat = Scalar::from_bytes(s_bytes);

        bool v = schnorr_adaptor_verify(bad, pk_x, msg, T);
        CHECK(!v, "Schnorr adaptor with flipped s_hat must fail");
        if (!v) ++detected;
    }

    CHECK(detected == TRIALS, "All Schnorr adaptor bit-flips detected");
    printf("    -> %d/%d detected\n\n", detected, TRIALS);
}

// ============================================================================
// 9. Schnorr adaptor: wrong adaptor point → verify fails
// ============================================================================
static void test_schnorr_adaptor_wrong_T() {
    g_section = "adaptor_schnorr_wrongT";
    printf("[9] Schnorr adaptor: wrong adaptor point -> verify fails\n");

    const int TRIALS = SCALED(30, 5);
    int detected = 0;

    for (int i = 0; i < TRIALS; ++i) {
        Scalar sk = random_scalar();
        Scalar t = random_scalar();
        Point T = Point::generator().scalar_mul(t);
        auto msg = random_bytes32();
        auto aux = random_bytes32();
        auto pk_x = Point::generator().scalar_mul(sk).x().to_bytes();

        auto pre_sig = schnorr_adaptor_sign(sk, msg, T, aux);

        // Verify with wrong T
        Scalar other_t = random_scalar();
        Point wrong_T = Point::generator().scalar_mul(other_t);

        bool v = schnorr_adaptor_verify(pre_sig, pk_x, msg, wrong_T);
        CHECK(!v, "Adaptor verified with wrong T must fail");
        if (!v) ++detected;
    }

    CHECK(detected == TRIALS, "All wrong-T attempts detected");
    printf("    -> %d/%d detected\n\n", detected, TRIALS);
}

// ============================================================================
// 10. ECDSA adaptor: bit-flip pre-sig → verify fails
// ============================================================================
static void test_ecdsa_adaptor_bitflip() {
    g_section = "adaptor_ecdsa_flip";
    printf("[10] ECDSA adaptor: bit-flip pre-sig -> verify fails\n");

    const int TRIALS = SCALED(30, 5);
    int detected = 0;

    for (int i = 0; i < TRIALS; ++i) {
        Scalar sk = random_scalar();
        Point pk = Point::generator().scalar_mul(sk);
        Scalar t = random_scalar();
        Point T = Point::generator().scalar_mul(t);
        auto msg = random_bytes32();

        auto pre_sig = ecdsa_adaptor_sign(sk, msg, T);

        // Verify original
        CHECK(ecdsa_adaptor_verify(pre_sig, pk, msg, T),
              "Original ECDSA adaptor pre-sig must verify");

        // Flip bit in s_hat
        auto s_bytes = pre_sig.s_hat.to_bytes();
        s_bytes[static_cast<size_t>(i % 32)] ^= static_cast<uint8_t>(1u << (i % 8));
        ECDSAAdaptorSig bad = pre_sig;
        bad.s_hat = Scalar::from_bytes(s_bytes);

        bool v = ecdsa_adaptor_verify(bad, pk, msg, T);
        CHECK(!v, "ECDSA adaptor with flipped s_hat must fail");
        if (!v) ++detected;
    }

    CHECK(detected == TRIALS, "All ECDSA adaptor bit-flips detected");
    printf("    -> %d/%d detected\n\n", detected, TRIALS);
}

// ============================================================================
// 11. Adaptor extraction: wrong sig → extracted secret is wrong
// ============================================================================
static void test_adaptor_extract_wrong_sig() {
    g_section = "adaptor_extract_wrong";
    printf("[11] Adaptor extraction: wrong sig -> wrong secret\n");

    const int TRIALS = SCALED(20, 5);
    int ok_count = 0;

    for (int i = 0; i < TRIALS; ++i) {
        Scalar sk = random_scalar();
        Scalar t = random_scalar();
        Point T = Point::generator().scalar_mul(t);
        auto msg = random_bytes32();
        auto aux = random_bytes32();

        auto pre_sig = schnorr_adaptor_sign(sk, msg, T, aux);
        auto sig = schnorr_adaptor_adapt(pre_sig, t);

        // Correct extraction should recover t (or its negation due to BIP-340 even-y)
        auto [recovered_t, success] = schnorr_adaptor_extract(pre_sig, sig);
        CHECK(success, "Correct extraction must succeed");
        if (success) {
            // Due to BIP-340 negation, extracted may be t or -t
            auto t_bytes = t.to_bytes();
            auto neg_t = Scalar::zero() - t;
            auto neg_t_bytes = neg_t.to_bytes();
            auto rec_bytes = recovered_t.to_bytes();
            bool matches = (rec_bytes == t_bytes) || (rec_bytes == neg_t_bytes);
            CHECK(matches,
                  "Extracted secret must match original t (or -t)");
        }

        // Use a totally different signature for extraction
        Scalar other_sk = random_scalar();
        auto other_aux = random_bytes32();
        auto other_sig = schnorr_sign(other_sk, msg, other_aux);

        auto [bad_t, bad_ok] = schnorr_adaptor_extract(pre_sig, other_sig);
        // The extracted secret should NOT match our original t
        if (bad_ok) {
            bool matches = (bad_t.to_bytes() == t.to_bytes());
            CHECK(!matches, "Wrong sig must not yield correct secret");
        }
        ++ok_count;
    }

    CHECK(ok_count == TRIALS, "All extraction trials ok");
    printf("    -> %d/%d OK\n\n", ok_count, TRIALS);
}

// ============================================================================
// 12. Pedersen homomorphic property: fault in addition
// ============================================================================
static void test_pedersen_homomorphic_fault() {
    g_section = "pedersen_homo_fault";
    printf("[12] Pedersen homomorphic addition: fault detection\n");

    const int TRIALS = SCALED(20, 5);
    int ok_count = 0;

    for (int i = 0; i < TRIALS; ++i) {
        Scalar v1 = random_scalar();
        Scalar r1 = random_scalar();
        Scalar v2 = random_scalar();
        Scalar r2 = random_scalar();

        auto C1 = pedersen_commit(v1, r1);
        auto C2 = pedersen_commit(v2, r2);
        auto C_sum = C1 + C2;

        // Sum commitment should verify with summed values
        Scalar v_sum = v1 + v2;
        Scalar r_sum = r1 + r2;
        CHECK(C_sum.verify(v_sum, r_sum), "Sum commitment must verify");

        // But NOT with wrong sum (e.g., v1 + v2 + 1)
        Scalar bad_sum = v_sum + Scalar::one();
        bool v = C_sum.verify(bad_sum, r_sum);
        CHECK(!v, "Wrong sum must not verify");
        ++ok_count;
    }

    CHECK(ok_count == TRIALS, "All homomorphic fault tests pass");
    printf("    -> %d/%d OK\n\n", ok_count, TRIALS);
}

// ============================================================================
// Exportable entry
// ============================================================================
int test_fault_zk_adaptor_run() {
    g_pass = g_fail = 0;
    test_pedersen_bitflip();
    test_pedersen_wrong_value();
    test_knowledge_proof_bitflip();
    test_knowledge_proof_wrong_key();
    test_dleq_bitflip();
    test_dleq_mismatch();
    test_range_proof_wrong_commitment();
    test_schnorr_adaptor_bitflip();
    test_schnorr_adaptor_wrong_T();
    test_ecdsa_adaptor_bitflip();
    test_adaptor_extract_wrong_sig();
    test_pedersen_homomorphic_fault();
    printf("  [fault_zk_adaptor] %d passed, %d failed\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}

#ifndef UNIFIED_AUDIT_RUNNER
int main() {
    printf("============================================================\n");
    printf("  ZK / Pedersen / Adaptor Fault-Injection Audit Test\n");
    printf("============================================================\n\n");
    int rc = test_fault_zk_adaptor_run();
    printf("\n============================================================\n");
    printf("  Result: %s\n", rc == 0 ? "ALL PASSED" : "SOME FAILED");
    printf("============================================================\n");
    return rc;
}
#endif
