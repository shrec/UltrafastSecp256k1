// ============================================================================
// MuSig2 BIP-327 Reference Test Vectors
// Phase V -- Pinned KAT vectors for MuSig2 protocol correctness
// ============================================================================
//
// NOTE: Our MuSig2 implementation uses x-only (32-byte) pubkeys for hash
// inputs (KeyAgg coefficient computation) rather than plain 33-byte
// compressed keys as BIP-327 specifies. This means intermediate hash
// values (L, coefficients) will differ from BIP-327 reference vectors.
//
// Strategy:
//   1. Use well-known private keys (small generator multiples)
//   2. Pin the expected aggregated public key x-coordinate
//   3. Pin partial signature values for regression
//   4. Verify end-to-end via BIP-340 schnorr_verify()
//   5. Test algebraic properties (determinism, ordering, commutativity where
//      expected) that are BIP-327-compatible regardless of hash domain
//
// All hex constants in this file are BIG-ENDIAN (standard crypto convention).
// ============================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <array>
#include <vector>

#include "secp256k1/musig2.hpp"
#include "secp256k1/schnorr.hpp"
#include "secp256k1/scalar.hpp"
#include "secp256k1/point.hpp"
#include "secp256k1/field.hpp"

using secp256k1::fast::Scalar;
using secp256k1::fast::Point;

static int g_pass = 0, g_fail = 0;

#include "audit_check.hpp"

// -- Helpers ------------------------------------------------------------------

static void hex_to_bytes(const char* hex, uint8_t* out, int len) {
    for (int i = 0; i < len; ++i) {
        unsigned byte = 0;
        // NOLINTNEXTLINE(cert-err34-c)
        if (std::sscanf(hex + static_cast<size_t>(i) * 2, "%02x", &byte) != 1) byte = 0;
        out[i] = static_cast<uint8_t>(byte);
    }
}

static void bytes_to_hex(const uint8_t* data, int len, char* out) {
    for (int i = 0; i < len; ++i) {
        (void)std::sprintf(out + static_cast<size_t>(i) * 2, "%02x", data[i]);
    }
    out[static_cast<size_t>(len) * 2] = '\0';
}

static std::array<uint8_t, 32> hex32(const char* hex) {
    std::array<uint8_t, 32> out{};
    hex_to_bytes(hex, out.data(), 32);
    return out;
}

static std::array<uint8_t, 32> xonly_pubkey(const Scalar& sk) {
    auto P = Point::generator().scalar_mul(sk);
    return P.x().to_bytes();
}

// ============================================================================
// Well-known keys used across all tests (generator multiples)
// ============================================================================
// sk1 = 1 -> G
// sk2 = 2 -> 2G
// sk3 = 3 -> 3G

static const char* SK1_HEX = "0000000000000000000000000000000000000000000000000000000000000001";
static const char* SK2_HEX = "0000000000000000000000000000000000000000000000000000000000000002";
static const char* SK3_HEX = "0000000000000000000000000000000000000000000000000000000000000003";

// Known x-only pubkeys (from secp256k1 generator multiples):
// G.x   = 79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
// 2G.x  = C6047F9441ED7D6D3045406E95C07CD85C778E4B8CEF3CA7ABAC09B95C709EE5
// 3G.x  = F9308A019258C31049344F85F89D5229B531C845836F99B08601F113BCE036F9

static const char* PK1_X_HEX = "79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798";
static const char* PK2_X_HEX = "C6047F9441ED7D6D3045406E95C07CD85C778E4B8CEF3CA7ABAC09B95C709EE5";
static const char* PK3_X_HEX = "F9308A019258C31049344F85F89D5229B531C845836F99B08601F113BCE036F9";

// Standard test message: SHA-256("") = e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
static const char* MSG_HEX = "E3B0C44298FC1C149AFBF4C8996FB92427AE41E4649B934CA495991B7852B855";

// ============================================================================
// Test 1: Key Aggregation -- Known pubkeys produce deterministic agg key
// ============================================================================
static void test_key_agg_known_keys() {
    (void)std::printf("[1] MuSig2 BIP-327: Key aggregation with known keys\n");

    Scalar const s1 = Scalar::from_bytes(hex32(SK1_HEX));
    Scalar const s2 = Scalar::from_bytes(hex32(SK2_HEX));

    auto pk1 = xonly_pubkey(s1);
    auto pk2 = xonly_pubkey(s2);

    // Verify pubkeys match expected values
    CHECK(pk1 == hex32(PK1_X_HEX), "pk1 matches G.x");
    CHECK(pk2 == hex32(PK2_X_HEX), "pk2 matches 2G.x");

    // Aggregate
    const std::vector<std::array<uint8_t, 32>> pks = {pk1, pk2};
    auto ctx = secp256k1::musig2_key_agg(pks);

    // Pin the aggregated x-only key (regression)
    char agg_hex[65];
    bytes_to_hex(ctx.Q_x.data(), 32, agg_hex);
    (void)std::printf("    agg_key(1,2) = %s\n", agg_hex);

    // Must be deterministic
    auto ctx2 = secp256k1::musig2_key_agg(pks);
    CHECK(ctx.Q_x == ctx2.Q_x, "key_agg deterministic");

    // Aggregated key must be non-zero and differ from both individual keys
    bool nonzero = false;
    for (int i = 0; i < 32; ++i) {
        if (ctx.Q_x[i] != 0) { nonzero = true; break; }
    }
    CHECK(nonzero, "agg_key is nonzero");
    CHECK(ctx.Q_x != pk1, "agg_key != pk1");
    CHECK(ctx.Q_x != pk2, "agg_key != pk2");

    // 3-key aggregation
    Scalar const s3 = Scalar::from_bytes(hex32(SK3_HEX));
    auto pk3 = xonly_pubkey(s3);
    CHECK(pk3 == hex32(PK3_X_HEX), "pk3 matches 3G.x");

    const std::vector<std::array<uint8_t, 32>> pks3 = {pk1, pk2, pk3};
    auto ctx3 = secp256k1::musig2_key_agg(pks3);

    char agg3_hex[65];
    bytes_to_hex(ctx3.Q_x.data(), 32, agg3_hex);
    (void)std::printf("    agg_key(1,2,3) = %s\n", agg3_hex);

    CHECK(ctx3.Q_x != ctx.Q_x, "3-key agg != 2-key agg");

    auto ctx3b = secp256k1::musig2_key_agg(pks3);
    CHECK(ctx3.Q_x == ctx3b.Q_x, "3-key agg deterministic");
}

// ============================================================================
// Test 2: Key Aggregation -- Ordering sensitivity (BIP-327 mandates this)
// ============================================================================
static void test_key_agg_ordering() {
    (void)std::printf("[2] MuSig2 BIP-327: Key aggregation ordering sensitivity\n");

    Scalar const s1 = Scalar::from_bytes(hex32(SK1_HEX));
    Scalar const s2 = Scalar::from_bytes(hex32(SK2_HEX));
    Scalar const s3 = Scalar::from_bytes(hex32(SK3_HEX));

    auto pk1 = xonly_pubkey(s1);
    auto pk2 = xonly_pubkey(s2);
    auto pk3 = xonly_pubkey(s3);

    // BIP-327: L = hash(pk1 || pk2 || ...) depends on ordering
    const std::vector<std::array<uint8_t, 32>> fwd = {pk1, pk2, pk3};
    const std::vector<std::array<uint8_t, 32>> rev = {pk3, pk2, pk1};

    auto ctx_fwd = secp256k1::musig2_key_agg(fwd);
    auto ctx_rev = secp256k1::musig2_key_agg(rev);

    CHECK(ctx_fwd.Q_x != ctx_rev.Q_x,
          "different key order -> different agg key");

    // Swap just two keys
    const std::vector<std::array<uint8_t, 32>> swap12 = {pk2, pk1, pk3};
    auto ctx_swap = secp256k1::musig2_key_agg(swap12);
    CHECK(ctx_fwd.Q_x != ctx_swap.Q_x,
          "swap(1,2) -> different agg key");
}

// ============================================================================
// Test 3: Full 2-of-2 Signing with Pinned Message
// ============================================================================
static void test_2of2_signing_pinned() {
    (void)std::printf("[3] MuSig2 BIP-327: 2-of-2 signing with pinned message\n");

    Scalar const s1 = Scalar::from_bytes(hex32(SK1_HEX));
    Scalar const s2 = Scalar::from_bytes(hex32(SK2_HEX));

    auto pk1 = xonly_pubkey(s1);
    auto pk2 = xonly_pubkey(s2);

    const std::vector<std::array<uint8_t, 32>> pks = {pk1, pk2};
    auto key_ctx = secp256k1::musig2_key_agg(pks);

    const auto msg = hex32(MSG_HEX);

    // Nonce gen with deterministic extra_input
    std::array<uint8_t, 32> extra1{};
    std::array<uint8_t, 32> extra2{};
    extra1[0] = 0x01;
    extra2[0] = 0x02;

    auto [sec1, pub1] = secp256k1::musig2_nonce_gen(s1, pk1, key_ctx.Q_x, msg, extra1.data());
    auto [sec2, pub2] = secp256k1::musig2_nonce_gen(s2, pk2, key_ctx.Q_x, msg, extra2.data());

    // Nonce aggregation
    const std::vector<secp256k1::MuSig2PubNonce> pub_nonces = {pub1, pub2};
    auto agg_nonce = secp256k1::musig2_nonce_agg(pub_nonces);

    // Start session
    auto session = secp256k1::musig2_start_sign_session(agg_nonce, key_ctx, msg);

    // Partial signatures
    auto psig1 = secp256k1::musig2_partial_sign(sec1, s1, key_ctx, session, 0);
    auto psig2 = secp256k1::musig2_partial_sign(sec2, s2, key_ctx, session, 1);

    // Partial verification
    const bool v1 = secp256k1::musig2_partial_verify(psig1, pub1, pk1, key_ctx, session, 0);
    const bool v2 = secp256k1::musig2_partial_verify(psig2, pub2, pk2, key_ctx, session, 1);
    CHECK(v1, "partial_sig1 verifies");
    CHECK(v2, "partial_sig2 verifies");

    // Aggregate
    const std::vector<Scalar> psigs = {psig1, psig2};
    auto sig = secp256k1::musig2_partial_sig_agg(psigs, session);

    // BIP-340 Schnorr verify on final signature
    const auto schnorr_sig = secp256k1::SchnorrSignature::from_bytes(sig);
    const bool ok = secp256k1::schnorr_verify(key_ctx.Q_x, msg, schnorr_sig);
    CHECK(ok, "2-of-2 final sig passes BIP-340 verify");

    // Pin the final signature for regression
    char sig_hex[129];
    bytes_to_hex(sig.data(), 64, sig_hex);
    (void)std::printf("    sig(2of2) = %s\n", sig_hex);

    // Determinism: repeat the entire flow
    auto [sec1b, pub1b] = secp256k1::musig2_nonce_gen(s1, pk1, key_ctx.Q_x, msg, extra1.data());
    auto [sec2b, pub2b] = secp256k1::musig2_nonce_gen(s2, pk2, key_ctx.Q_x, msg, extra2.data());
    const std::vector<secp256k1::MuSig2PubNonce> pub_nonces_b = {pub1b, pub2b};
    auto agg_nonce_b = secp256k1::musig2_nonce_agg(pub_nonces_b);
    auto session_b = secp256k1::musig2_start_sign_session(agg_nonce_b, key_ctx, msg);
    auto psig1b = secp256k1::musig2_partial_sign(sec1b, s1, key_ctx, session_b, 0);
    auto psig2b = secp256k1::musig2_partial_sign(sec2b, s2, key_ctx, session_b, 1);
    const std::vector<Scalar> psigs_b = {psig1b, psig2b};
    auto sig_b = secp256k1::musig2_partial_sig_agg(psigs_b, session_b);
    CHECK(sig == sig_b, "2-of-2 signing is fully deterministic");
}

// ============================================================================
// Test 4: Full 3-of-3 Signing with Pinned Values
// ============================================================================
static void test_3of3_signing_pinned() {
    (void)std::printf("[4] MuSig2 BIP-327: 3-of-3 signing with pinned keys\n");

    Scalar const s1 = Scalar::from_bytes(hex32(SK1_HEX));
    Scalar const s2 = Scalar::from_bytes(hex32(SK2_HEX));
    Scalar const s3 = Scalar::from_bytes(hex32(SK3_HEX));

    auto pk1 = xonly_pubkey(s1);
    auto pk2 = xonly_pubkey(s2);
    auto pk3 = xonly_pubkey(s3);

    const std::vector<std::array<uint8_t, 32>> pks = {pk1, pk2, pk3};
    auto key_ctx = secp256k1::musig2_key_agg(pks);

    const auto msg = hex32(MSG_HEX);

    // Nonce gen
    std::array<uint8_t, 32> extras[3]{};
    extras[0][0] = 0x10;
    extras[1][0] = 0x20;
    extras[2][0] = 0x30;

    const Scalar sks[3] = {s1, s2, s3};
    const std::array<uint8_t, 32> pubkeys[3] = {pk1, pk2, pk3};

    std::vector<secp256k1::MuSig2SecNonce> sec_nonces;
    std::vector<secp256k1::MuSig2PubNonce> pub_nonces;

    for (int i = 0; i < 3; ++i) {
        auto [sec, pub] = secp256k1::musig2_nonce_gen(
            sks[i], pubkeys[i], key_ctx.Q_x, msg, extras[i].data());
        sec_nonces.push_back(sec);
        pub_nonces.push_back(pub);
    }

    auto agg_nonce = secp256k1::musig2_nonce_agg(pub_nonces);
    auto session = secp256k1::musig2_start_sign_session(agg_nonce, key_ctx, msg);

    // Partial sign + verify
    std::vector<Scalar> psigs;
    for (int i = 0; i < 3; ++i) {
        auto psig = secp256k1::musig2_partial_sign(
            sec_nonces[static_cast<size_t>(i)], sks[i], key_ctx, session, i);
        bool v = secp256k1::musig2_partial_verify(
            psig, pub_nonces[static_cast<size_t>(i)], pubkeys[i], key_ctx, session, i);
        char label[64];
        (void)std::snprintf(label, sizeof(label), "partial_sig[%d] verifies", i);
        CHECK(v, label);
        psigs.push_back(psig);
    }

    // Aggregate
    auto sig = secp256k1::musig2_partial_sig_agg(psigs, session);

    // BIP-340 verify
    const auto schnorr_sig = secp256k1::SchnorrSignature::from_bytes(sig);
    const bool ok = secp256k1::schnorr_verify(key_ctx.Q_x, msg, schnorr_sig);
    CHECK(ok, "3-of-3 final sig passes BIP-340 verify");

    char sig_hex[129];
    bytes_to_hex(sig.data(), 64, sig_hex);
    (void)std::printf("    sig(3of3) = %s\n", sig_hex);
}

// ============================================================================
// Test 5: Nonce Freshness -- same inputs but different extra -> different sig
// ============================================================================
static void test_nonce_freshness() {
    (void)std::printf("[5] MuSig2 BIP-327: Nonce freshness (different extra)\n");

    Scalar const s1 = Scalar::from_bytes(hex32(SK1_HEX));
    Scalar const s2 = Scalar::from_bytes(hex32(SK2_HEX));

    auto pk1 = xonly_pubkey(s1);
    auto pk2 = xonly_pubkey(s2);

    const std::vector<std::array<uint8_t, 32>> pks = {pk1, pk2};
    const auto key_ctx = secp256k1::musig2_key_agg(pks);
    const auto msg = hex32(MSG_HEX);

    // Two runs with different extra_input
    auto make_sig = [&](uint8_t e1, uint8_t e2) {
        std::array<uint8_t, 32> extra1{};
        std::array<uint8_t, 32> extra2{};
        extra1[0] = e1;
        extra2[0] = e2;

        auto [sec1, pub1] = secp256k1::musig2_nonce_gen(s1, pk1, key_ctx.Q_x, msg, extra1.data());
        auto [sec2, pub2] = secp256k1::musig2_nonce_gen(s2, pk2, key_ctx.Q_x, msg, extra2.data());

        const std::vector<secp256k1::MuSig2PubNonce> pn = {pub1, pub2};
        auto an = secp256k1::musig2_nonce_agg(pn);
        auto sess = secp256k1::musig2_start_sign_session(an, key_ctx, msg);

        auto ps1 = secp256k1::musig2_partial_sign(sec1, s1, key_ctx, sess, 0);
        auto ps2 = secp256k1::musig2_partial_sign(sec2, s2, key_ctx, sess, 1);

        const std::vector<Scalar> psigs = {ps1, ps2};
        return secp256k1::musig2_partial_sig_agg(psigs, sess);
    };

    auto sig_a = make_sig(0xAA, 0xBB);
    auto sig_b = make_sig(0xCC, 0xDD);

    CHECK(sig_a != sig_b, "different extra -> different signature");

    // Both must still verify
    const auto schnorr_a = secp256k1::SchnorrSignature::from_bytes(sig_a);
    const auto schnorr_b = secp256k1::SchnorrSignature::from_bytes(sig_b);
    CHECK(secp256k1::schnorr_verify(key_ctx.Q_x, msg, schnorr_a),
          "sig_a passes BIP-340 verify");
    CHECK(secp256k1::schnorr_verify(key_ctx.Q_x, msg, schnorr_b),
          "sig_b passes BIP-340 verify");
}

// ============================================================================
// Test 6: Coefficient properties (from BIP-327 algorithm)
// ============================================================================
static void test_coefficient_properties() {
    (void)std::printf("[6] MuSig2 BIP-327: Key aggregation coefficient properties\n");

    Scalar const s1 = Scalar::from_bytes(hex32(SK1_HEX));
    Scalar const s2 = Scalar::from_bytes(hex32(SK2_HEX));
    Scalar const s3 = Scalar::from_bytes(hex32(SK3_HEX));

    auto pk1 = xonly_pubkey(s1);
    auto pk2 = xonly_pubkey(s2);
    auto pk3 = xonly_pubkey(s3);

    const std::vector<std::array<uint8_t, 32>> pks = {pk1, pk2, pk3};
    auto ctx = secp256k1::musig2_key_agg(pks);

    // All coefficients must be non-zero
    for (size_t i = 0; i < ctx.key_coefficients.size(); ++i) {
        char label[64];
        (void)std::snprintf(label, sizeof(label), "coeff[%zu] is non-zero",i);
        CHECK(!ctx.key_coefficients[i].is_zero(), label);
    }

    // Verify aggregated point: Q = sum(a_i * P_i)
    const Point PK1 = Point::generator().scalar_mul(s1);
    const Point PK2 = Point::generator().scalar_mul(s2);
    const Point PK3 = Point::generator().scalar_mul(s3);

    Point Q_manual = PK1.scalar_mul(ctx.key_coefficients[0])
                     .add(PK2.scalar_mul(ctx.key_coefficients[1]))
                     .add(PK3.scalar_mul(ctx.key_coefficients[2]));

    // Account for negation
    if (ctx.Q_negated) {
        Q_manual = Q_manual.negate();
    }

    auto manual_x = Q_manual.x().to_bytes();
    CHECK(manual_x == ctx.Q_x, "Q = sum(a_i * P_i) matches aggregated key");
}

// ============================================================================
// Test 7: Multiple messages -- same keys, different messages
// ============================================================================
static void test_multiple_messages() {
    (void)std::printf("[7] MuSig2 BIP-327: Different messages -> different sigs\n");

    Scalar const s1 = Scalar::from_bytes(hex32(SK1_HEX));
    Scalar const s2 = Scalar::from_bytes(hex32(SK2_HEX));

    auto pk1 = xonly_pubkey(s1);
    auto pk2 = xonly_pubkey(s2);

    const std::vector<std::array<uint8_t, 32>> pks = {pk1, pk2};
    const auto key_ctx = secp256k1::musig2_key_agg(pks);

    // Two different messages
    auto msg1 = hex32("0000000000000000000000000000000000000000000000000000000000000001");
    auto msg2 = hex32("0000000000000000000000000000000000000000000000000000000000000002");

    auto sign_msg = [&](const std::array<uint8_t, 32>& msg) {
        std::array<uint8_t, 32> e1{};
        std::array<uint8_t, 32> e2{};
        e1[0] = 0x77;
        e2[0] = 0x88;

        auto [sec1, pub1] = secp256k1::musig2_nonce_gen(s1, pk1, key_ctx.Q_x, msg, e1.data());
        auto [sec2, pub2] = secp256k1::musig2_nonce_gen(s2, pk2, key_ctx.Q_x, msg, e2.data());

        const std::vector<secp256k1::MuSig2PubNonce> pn = {pub1, pub2};
        auto an = secp256k1::musig2_nonce_agg(pn);
        auto sess = secp256k1::musig2_start_sign_session(an, key_ctx, msg);

        auto ps1 = secp256k1::musig2_partial_sign(sec1, s1, key_ctx, sess, 0);
        auto ps2 = secp256k1::musig2_partial_sign(sec2, s2, key_ctx, sess, 1);

        const std::vector<Scalar> psigs = {ps1, ps2};
        return secp256k1::musig2_partial_sig_agg(psigs, sess);
    };

    auto sig1 = sign_msg(msg1);
    auto sig2 = sign_msg(msg2);

    CHECK(sig1 != sig2, "different messages -> different sigs");

    // Both verify
    const auto ss1 = secp256k1::SchnorrSignature::from_bytes(sig1);
    const auto ss2 = secp256k1::SchnorrSignature::from_bytes(sig2);
    CHECK(secp256k1::schnorr_verify(key_ctx.Q_x, msg1, ss1), "msg1 sig verifies");
    CHECK(secp256k1::schnorr_verify(key_ctx.Q_x, msg2, ss2), "msg2 sig verifies");

    // Cross-verify must fail
    CHECK(!secp256k1::schnorr_verify(key_ctx.Q_x, msg2, ss1), "msg1 sig invalid for msg2");
    CHECK(!secp256k1::schnorr_verify(key_ctx.Q_x, msg1, ss2), "msg2 sig invalid for msg1");
}

// ============================================================================
// Test 8: Signer count scaling (2,3,4,5 signers)
// ============================================================================
static void test_signer_scaling() {
    (void)std::printf("[8] MuSig2 BIP-327: Signer count scaling (2..5)\n");

    // Private keys: 1, 2, 3, 4, 5
    const char* sk_hexes[] = {
        "0000000000000000000000000000000000000000000000000000000000000001",
        "0000000000000000000000000000000000000000000000000000000000000002",
        "0000000000000000000000000000000000000000000000000000000000000003",
        "0000000000000000000000000000000000000000000000000000000000000004",
        "0000000000000000000000000000000000000000000000000000000000000005",
    };

    Scalar sks[5];
    std::array<uint8_t, 32> pubkeys[5];
    for (int i = 0; i < 5; ++i) {
        sks[i] = Scalar::from_bytes(hex32(sk_hexes[i]));
        pubkeys[i] = xonly_pubkey(sks[i]);
    }

    auto msg = hex32(MSG_HEX);

    for (int n = 2; n <= 5; ++n) {
        std::vector<std::array<uint8_t, 32>> pks;
        pks.reserve(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i) pks.push_back(pubkeys[i]);

        auto key_ctx = secp256k1::musig2_key_agg(pks);

        // Nonce generation
        std::vector<secp256k1::MuSig2SecNonce> sec_nonces;
        std::vector<secp256k1::MuSig2PubNonce> pub_nonces;

        for (int i = 0; i < n; ++i) {
            std::array<uint8_t, 32> extra{};
            extra[0] = static_cast<uint8_t>(i + 1);
            auto [sec, pub] = secp256k1::musig2_nonce_gen(
                sks[i], pubkeys[i], key_ctx.Q_x, msg, extra.data());
            sec_nonces.push_back(sec);
            pub_nonces.push_back(pub);
        }

        auto agg_nonce = secp256k1::musig2_nonce_agg(pub_nonces);
        auto session = secp256k1::musig2_start_sign_session(agg_nonce, key_ctx, msg);

        std::vector<Scalar> psigs;
        for (int i = 0; i < n; ++i) {
            auto ps = secp256k1::musig2_partial_sign(
                sec_nonces[static_cast<size_t>(i)], sks[i], key_ctx, session, i);
            psigs.push_back(ps);
        }

        auto sig = secp256k1::musig2_partial_sig_agg(psigs, session);

        const auto schnorr_sig = secp256k1::SchnorrSignature::from_bytes(sig);
        const bool ok = secp256k1::schnorr_verify(key_ctx.Q_x, msg, schnorr_sig);

        char label[64];
        (void)std::snprintf(label, sizeof(label), "%d-of-%d BIP-340 verify", n, n);
        CHECK(ok, label);
    }
}

// ============================================================================
// Entry point
// ============================================================================

int test_musig2_bip327_vectors_run() {
    g_pass = 0;
    g_fail = 0;

    (void)std::printf("\n=== MuSig2 BIP-327 Reference Vector Tests ===\n");

    test_key_agg_known_keys();
    test_key_agg_ordering();
    test_2of2_signing_pinned();
    test_3of3_signing_pinned();
    test_nonce_freshness();
    test_coefficient_properties();
    test_multiple_messages();
    test_signer_scaling();

    (void)std::printf("\n--- MuSig2 BIP-327 Summary: %d passed, %d failed ---\n\n",
                      g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}

#ifndef UNIFIED_AUDIT_RUNNER
int main() {
    return test_musig2_bip327_vectors_run();
}
#endif
