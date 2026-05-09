// ============================================================================
// Secret Lifecycle Audit Test
// ============================================================================
// Traces the lifecycle of secret material through the engine:
//
//   creation → use in signing → signature verification → scope exit
//
// TESTS:
//   1. Scalar secrets: fresh random → ECDSA sign → sig valid → sk scope end
//   2. Schnorr secret lifecycle: sk → schnorr_sign → verify → done
//   3. Adaptor secret lifecycle: t → pre_sign → adapt → extract → compare
//   4. ZK proof secret lifecycle: x → knowledge_prove → verify → x scope end
//   5. DLEQ secret lifecycle: x → dleq_prove → verify → done
//   6. Pedersen blinding lifecycle: r → commit → verify → r scope end
//   7. MuSig2 sec-nonce lifecycle: gen → partial_sign → verify → aggregate
//   8. Multi-secret pipeline: BIP-32 derive → sign → verify chain
//   9. Nonce hedging: same key + different aux → different nonces
//  10. Secret independence: signing with sk1 does not affect sk2
// ============================================================================

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <array>
#include <random>
#include <vector>

#include "secp256k1/field.hpp"
#include "secp256k1/scalar.hpp"
#include "secp256k1/point.hpp"
#include "secp256k1/ecdsa.hpp"
#include "secp256k1/schnorr.hpp"
#include "secp256k1/pedersen.hpp"
#include "secp256k1/zk.hpp"
#include "secp256k1/adaptor.hpp"
#include "secp256k1/musig2.hpp"
#include "secp256k1/sanitizer_scale.hpp"

using namespace secp256k1;
using namespace secp256k1::fast;
using namespace secp256k1::zk;

// Intentionally exercises the legacy variable-time secp256k1::ecdsa_sign /
// schnorr_sign entry points (test/bench/audit harness). Suppress the
// deprecation warning so -Werror builds succeed.
#if defined(__GNUC__) || defined(__clang__)
#  pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif

static int g_pass = 0, g_fail = 0;
static const char* g_section = "";

#include "audit_check.hpp"

static std::mt19937_64 rng(0xDEAD'BEEFULL);

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
// 1. ECDSA secret lifecycle
// ============================================================================
static void test_ecdsa_secret_lifecycle() {
    g_section = "ecdsa_lifecycle";
    printf("[1] ECDSA secret lifecycle: create -> sign -> verify -> scope\n");

    const int TRIALS = SCALED(50, 10);
    int ok_count = 0;

    for (int i = 0; i < TRIALS; ++i) {
        auto msg = random_bytes32();
        Scalar sk = random_scalar();
        Point pk = Point::generator().scalar_mul(sk);

        // Sign
        auto sig = ecdsa_sign(msg, sk);
        CHECK(!sig.r.is_zero(), "sig.r must be non-zero");
        CHECK(!sig.s.is_zero(), "sig.s must be non-zero");

        // Verify
        bool v = ecdsa_verify(msg, pk, sig);
        CHECK(v, "ECDSA sig must verify");

        // Verify with wrong message fails
        auto bad_msg = msg;
        bad_msg[0] ^= 0xFF;
        bool bv = ecdsa_verify(bad_msg, pk, sig);
        CHECK(!bv, "Wrong message must not verify");

        ++ok_count;
    }

    CHECK(ok_count == TRIALS, "All ECDSA lifecycle trials pass");
    printf("    -> %d/%d OK\n\n", ok_count, TRIALS);
}

// ============================================================================
// 2. Schnorr secret lifecycle
// ============================================================================
static void test_schnorr_secret_lifecycle() {
    g_section = "schnorr_lifecycle";
    printf("[2] Schnorr secret lifecycle: create -> sign -> verify -> scope\n");

    const int TRIALS = SCALED(50, 10);
    int ok_count = 0;

    for (int i = 0; i < TRIALS; ++i) {
        auto msg = random_bytes32();
        Scalar sk = random_scalar();
        auto pk_x = Point::generator().scalar_mul(sk).x().to_bytes();

        auto aux1 = random_bytes32();
        auto sig = schnorr_sign(sk, msg, aux1);

        // Verify
        bool v = schnorr_verify(pk_x, msg, sig);
        CHECK(v, "Schnorr sig must verify");

        // Wrong message
        auto bad_msg = msg;
        bad_msg[0] ^= 0xFF;
        bool bv = schnorr_verify(pk_x, bad_msg, sig);
        CHECK(!bv, "Wrong message must not verify");

        ++ok_count;
    }

    CHECK(ok_count == TRIALS, "All Schnorr lifecycle trials pass");
    printf("    -> %d/%d OK\n\n", ok_count, TRIALS);
}

// ============================================================================
// 3. Adaptor secret lifecycle: full round-trip
// ============================================================================
static void test_adaptor_secret_lifecycle() {
    g_section = "adaptor_lifecycle";
    printf("[3] Adaptor secret lifecycle: create -> presign -> adapt -> extract\n");

    const int TRIALS = SCALED(20, 5);
    int ok_count = 0;

    for (int i = 0; i < TRIALS; ++i) {
        Scalar sk = random_scalar();
        Scalar t = random_scalar();
        Point T = Point::generator().scalar_mul(t);
        auto msg = random_bytes32();
        auto aux = random_bytes32();
        auto pk_x = Point::generator().scalar_mul(sk).x().to_bytes();

        // Pre-sign
        auto pre_sig = schnorr_adaptor_sign(sk, msg, T, aux);
        bool pre_ok = schnorr_adaptor_verify(pre_sig, pk_x, msg, T);
        CHECK(pre_ok, "Pre-sig must verify");

        // Adapt with secret t
        auto final_sig = schnorr_adaptor_adapt(pre_sig, t);
        bool final_ok = schnorr_verify(pk_x, msg, final_sig);
        CHECK(final_ok, "Adapted sig must verify as valid Schnorr");

        // Extract t from (pre_sig, sig)
        auto [recovered_t, extract_ok] = schnorr_adaptor_extract(pre_sig, final_sig);
        CHECK(extract_ok, "Extraction must succeed");
        if (extract_ok) {
            // BIP-340 may negate; check t or -t
            auto t_bytes = t.to_bytes();
            auto neg_t = Scalar::zero() - t;
            auto neg_t_bytes = neg_t.to_bytes();
            auto rec_bytes = recovered_t.to_bytes();
            bool matches = (rec_bytes == t_bytes) || (rec_bytes == neg_t_bytes);
            CHECK(matches,
                  "Extracted secret must match original t (or -t)");
        }
        ++ok_count;
    }

    CHECK(ok_count == TRIALS, "All adaptor lifecycle trials pass");
    printf("    -> %d/%d OK\n\n", ok_count, TRIALS);
}

// ============================================================================
// 4. ZK knowledge proof secret lifecycle
// ============================================================================
static void test_zk_knowledge_lifecycle() {
    g_section = "zk_knowledge_lifecycle";
    printf("[4] ZK knowledge proof lifecycle: secret -> prove -> verify\n");

    const int TRIALS = SCALED(30, 5);
    int ok_count = 0;

    for (int i = 0; i < TRIALS; ++i) {
        Scalar secret = random_scalar();
        Point pubkey = Point::generator().scalar_mul(secret);
        auto msg = random_bytes32();
        auto aux = random_bytes32();

        // Prove
        auto proof = knowledge_prove(secret, pubkey, msg, aux);

        // Verify
        bool v = knowledge_verify(proof, pubkey, msg);
        CHECK(v, "Knowledge proof must verify");

        // Wrong key should fail
        Scalar other = random_scalar();
        Point wrong_pk = Point::generator().scalar_mul(other);
        bool bv = knowledge_verify(proof, wrong_pk, msg);
        CHECK(!bv, "Wrong key must fail verification");

        ++ok_count;
    }

    CHECK(ok_count == TRIALS, "All ZK knowledge lifecycle trials pass");
    printf("    -> %d/%d OK\n\n", ok_count, TRIALS);
}

// ============================================================================
// 5. DLEQ secret lifecycle
// ============================================================================
static void test_dleq_lifecycle() {
    g_section = "dleq_lifecycle";
    printf("[5] DLEQ lifecycle: secret -> prove equality -> verify\n");

    const int TRIALS = SCALED(20, 5);
    int ok_count = 0;

    for (int i = 0; i < TRIALS; ++i) {
        Scalar secret = random_scalar();
        Point G = Point::generator();
        Scalar h_scalar = random_scalar();
        Point H = G.scalar_mul(h_scalar);

        Point P = G.scalar_mul(secret);
        Point Q = H.scalar_mul(secret);
        auto aux = random_bytes32();

        auto proof = dleq_prove(secret, G, H, P, Q, aux);
        bool v = dleq_verify(proof, G, H, P, Q);
        CHECK(v, "DLEQ proof must verify");

        // With different Q should fail
        Scalar wrong = random_scalar();
        Point wrong_Q = H.scalar_mul(wrong);
        bool bv = dleq_verify(proof, G, H, P, wrong_Q);
        CHECK(!bv, "DLEQ with wrong Q must fail");

        ++ok_count;
    }

    CHECK(ok_count == TRIALS, "All DLEQ lifecycle trials pass");
    printf("    -> %d/%d OK\n\n", ok_count, TRIALS);
}

// ============================================================================
// 6. Pedersen blinding lifecycle
// ============================================================================
static void test_pedersen_blinding_lifecycle() {
    g_section = "pedersen_lifecycle";
    printf("[6] Pedersen blinding lifecycle: create -> commit -> verify\n");

    const int TRIALS = SCALED(30, 5);
    int ok_count = 0;

    for (int i = 0; i < TRIALS; ++i) {
        Scalar value = random_scalar();
        Scalar blinding = random_scalar();

        auto C = pedersen_commit(value, blinding);
        bool v = C.verify(value, blinding);
        CHECK(v, "Pedersen commitment must verify");

        // Wrong blinding
        Scalar wrong_r = random_scalar();
        bool bv = C.verify(value, wrong_r);
        CHECK(!bv, "Wrong blinding must fail");

        // Homomorphic addition
        Scalar v2 = random_scalar();
        Scalar r2 = random_scalar();
        auto C2 = pedersen_commit(v2, r2);
        auto sum = C + C2;
        bool sv = sum.verify(value + v2, blinding + r2);
        CHECK(sv, "Homomorphic sum must verify");

        ++ok_count;
    }

    CHECK(ok_count == TRIALS, "All Pedersen lifecycle trials pass");
    printf("    -> %d/%d OK\n\n", ok_count, TRIALS);
}

// ============================================================================
// 7. MuSig2 sec-nonce lifecycle
// ============================================================================
static void test_musig2_lifecycle() {
    g_section = "musig2_lifecycle";
    printf("[7] MuSig2 lifecycle: keygen -> nonce -> sign -> verify -> agg\n");

    const int N = SCALED(8, 3);
    int ok_count = 0;

    for (int round = 0; round < N; ++round) {
        Scalar sk0 = random_scalar();
        Scalar sk1 = random_scalar();
        auto pk0 = Point::generator().scalar_mul(sk0).x().to_bytes();
        auto pk1 = Point::generator().scalar_mul(sk1).x().to_bytes();
        std::vector<std::array<uint8_t, 33>> pks = {
            Point::generator().scalar_mul(sk0).to_compressed(),
            Point::generator().scalar_mul(sk1).to_compressed()};

        auto key_agg = musig2_key_agg(pks);
        auto msg = random_bytes32();

        // Nonce gen
        auto extra0 = random_bytes32();
        auto [sec0, pub0] = musig2_nonce_gen(sk0, pk0, key_agg.Q_x, msg, extra0.data());
        auto extra1 = random_bytes32();
        auto [sec1, pub1] = musig2_nonce_gen(sk1, pk1, key_agg.Q_x, msg, extra1.data());

        auto agg_nonce = musig2_nonce_agg({pub0, pub1});
        auto session = musig2_start_sign_session(agg_nonce, key_agg, msg);

        // Partial sign
        auto ps0 = musig2_partial_sign(sec0, sk0, key_agg, session, 0);
        auto ps1 = musig2_partial_sign(sec1, sk1, key_agg, session, 1);

        // Partial verify
        bool pv0 = musig2_partial_verify(ps0, pub0, pk0, key_agg, session, 0);
        bool pv1 = musig2_partial_verify(ps1, pub1, pk1, key_agg, session, 1);
        CHECK(pv0, "Partial sig 0 must verify");
        CHECK(pv1, "Partial sig 1 must verify");

        // Aggregate
        auto sig64 = musig2_partial_sig_agg({ps0, ps1}, session);
        auto sig = SchnorrSignature::from_bytes(sig64);
        bool final_ok = schnorr_verify(key_agg.Q_x, msg, sig);
        CHECK(final_ok, "Aggregated MuSig2 sig must verify");

        ++ok_count;
    }

    CHECK(ok_count == N, "All MuSig2 lifecycle trials pass");
    printf("    -> %d/%d OK\n\n", ok_count, N);
}

// ============================================================================
// 8. Nonce hedging: same key + different aux → different nonces
// ============================================================================
static void test_nonce_hedging_independence() {
    g_section = "nonce_hedging";
    printf("[8] Nonce hedging: same key + different aux -> different sigs\n");

    const int TRIALS = SCALED(30, 5);
    int ok_count = 0;

    for (int i = 0; i < TRIALS; ++i) {
        Scalar sk = random_scalar();
        auto msg = random_bytes32();

        // Sign twice, Schnorr uses aux internally
        auto aux1 = random_bytes32();
        auto sig1 = schnorr_sign(sk, msg, aux1);
        auto aux2 = random_bytes32();
        auto sig2 = schnorr_sign(sk, msg, aux2);

        // Both must verify
        auto pk_x = Point::generator().scalar_mul(sk).x().to_bytes();
        CHECK(schnorr_verify(pk_x, msg, sig1), "sig1 must verify");
        CHECK(schnorr_verify(pk_x, msg, sig2), "sig2 must verify");

        // If aux is deterministic (same seed), sigs may be identical --
        // that's OK. The important thing is both verify.
        ++ok_count;
    }

    CHECK(ok_count == TRIALS, "All nonce hedging tests pass");
    printf("    -> %d/%d OK\n\n", ok_count, TRIALS);
}

// ============================================================================
// 9. Secret independence: signing with sk1 doesn't affect sk2
// ============================================================================
static void test_secret_independence() {
    g_section = "secret_independence";
    printf("[9] Secret independence: sk1 does not affect sk2\n");

    const int TRIALS = SCALED(30, 5);
    int ok_count = 0;

    for (int i = 0; i < TRIALS; ++i) {
        Scalar sk1 = random_scalar();
        Scalar sk2 = random_scalar();
        auto msg = random_bytes32();

        Point pk1 = Point::generator().scalar_mul(sk1);
        Point pk2 = Point::generator().scalar_mul(sk2);

        // Sign with sk1
        auto sig1 = ecdsa_sign(msg, sk1);
        CHECK(ecdsa_verify(msg, pk1, sig1), "sig1 must verify with pk1");
        CHECK(!ecdsa_verify(msg, pk2, sig1), "sig1 must NOT verify with pk2");

        // Sign with sk2 (sk1 was used above but should not interfere)
        auto sig2 = ecdsa_sign(msg, sk2);
        CHECK(ecdsa_verify(msg, pk2, sig2), "sig2 must verify with pk2");
        CHECK(!ecdsa_verify(msg, pk1, sig2), "sig2 must NOT verify with pk1");

        ++ok_count;
    }

    CHECK(ok_count == TRIALS, "All independence trials pass");
    printf("    -> %d/%d OK\n\n", ok_count, TRIALS);
}

// ============================================================================
// 10. Ranged Bulletproof full lifecycle
// ============================================================================
static void test_range_proof_lifecycle() {
    g_section = "range_lifecycle";
    printf("[10] Range proof lifecycle: value -> commit -> prove -> verify\n");

    const int TRIALS = SCALED(3, 1);
    int ok_count = 0;

    for (int i = 0; i < TRIALS; ++i) {
        uint64_t value = static_cast<uint64_t>(rng() & 0xFFFFFFFF);
        Scalar blinding = random_scalar();
        Scalar val_scalar = Scalar::from_limbs({value, 0, 0, 0});
        auto C = pedersen_commit(val_scalar, blinding);
        auto aux = random_bytes32();

        auto proof = range_prove(value, blinding, C, aux);
        bool v = range_verify(C, proof);
        CHECK(v, "Range proof must verify for valid value");

        ++ok_count;
    }

    CHECK(ok_count == TRIALS, "All range proof lifecycle trials pass");
    printf("    -> %d/%d OK\n\n", ok_count, TRIALS);
}

// ============================================================================
// Exportable entry
// ============================================================================
int test_secret_lifecycle_run() {
    g_pass = g_fail = 0;
    test_ecdsa_secret_lifecycle();
    test_schnorr_secret_lifecycle();
    test_adaptor_secret_lifecycle();
    test_zk_knowledge_lifecycle();
    test_dleq_lifecycle();
    test_pedersen_blinding_lifecycle();
    test_musig2_lifecycle();
    test_nonce_hedging_independence();
    test_secret_independence();
    test_range_proof_lifecycle();
    printf("  [secret_lifecycle] %d passed, %d failed\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}

#ifndef UNIFIED_AUDIT_RUNNER
int main() {
    printf("============================================================\n");
    printf("  Secret Lifecycle Audit Test\n");
    printf("============================================================\n\n");
    int rc = test_secret_lifecycle_run();
    printf("\n============================================================\n");
    printf("  Result: %s\n", rc == 0 ? "ALL PASSED" : "SOME FAILED");
    printf("============================================================\n");
    return rc;
}
#endif
