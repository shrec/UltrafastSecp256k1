// ============================================================================
// Test: MuSig2 Multi-Signatures (BIP-327 on secp256k1)
// ============================================================================

#include "secp256k1/musig2.hpp"
#include "secp256k1/schnorr.hpp"

#include <cstdio>
#include <cstring>
#include <array>
#include <string>
#include <vector>

using namespace secp256k1;
using fast::Scalar;
using fast::Point;

static int tests_run = 0;
static int tests_passed = 0;

#define CHECK(cond, msg) do { \
    ++tests_run; \
    if (cond) { ++tests_passed; printf("  [PASS] %s\n", msg); } \
    else { printf("  [FAIL] %s\n", msg); } \
} while(0)

static void hex_to_bytes(const char* hex, uint8_t* out, size_t len) {
    for (size_t i = 0; i < len; ++i) {
        unsigned int byte = 0;
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif
        // NOLINTNEXTLINE(cert-err34-c) -- parsing 2-digit hex, overflow impossible
        if (sscanf(hex + i * 2, "%02x", &byte) != 1) byte = 0;
#ifdef __clang__
#pragma clang diagnostic pop
#endif
        out[i] = static_cast<uint8_t>(byte);
    }
}

// Helper: get x-only public key (32 bytes) from private key scalar
static std::array<uint8_t, 32> get_xonly_pubkey(const Scalar& sk) {
    auto P = Point::generator().scalar_mul(sk);
    return P.x().to_bytes();
}

// Helper: compare two points via compressed serialization
static bool points_equal(const Point& a, const Point& b) {
    if (a.is_infinity() && b.is_infinity()) return true;
    if (a.is_infinity() || b.is_infinity()) return false;
    return a.to_compressed() == b.to_compressed();
}

// -- Key Aggregation ----------------------------------------------------------

static void test_key_aggregation() {
    printf("\n--- Key Aggregation ---\n");

    // Create 2 keypairs
    std::array<uint8_t, 32> sk1_bytes{}, sk2_bytes{};
    hex_to_bytes("0000000000000000000000000000000000000000000000000000000000000001",
                 sk1_bytes.data(), 32);
    hex_to_bytes("0000000000000000000000000000000000000000000000000000000000000002",
                 sk2_bytes.data(), 32);

    Scalar const s1 = Scalar::from_bytes(sk1_bytes);
    Scalar const s2 = Scalar::from_bytes(sk2_bytes);

    auto pk1 = get_xonly_pubkey(s1);
    auto pk2 = get_xonly_pubkey(s2);

    std::vector<std::array<uint8_t, 32>> const pubkeys = {pk1, pk2};
    auto ctx = musig2_key_agg(pubkeys);

    // Aggregated key should not be infinity
    CHECK(!ctx.Q.is_infinity(), "Aggregated key is valid point");

    // Aggregated key should differ from both individual keys
    auto P1 = Point::generator().scalar_mul(s1);
    auto P2 = Point::generator().scalar_mul(s2);
    CHECK(!points_equal(ctx.Q, P1), "Agg key != P1");
    CHECK(!points_equal(ctx.Q, P2), "Agg key != P2");

    // Deterministic: same inputs -> same output
    auto ctx2 = musig2_key_agg(pubkeys);
    CHECK(points_equal(ctx.Q, ctx2.Q), "Key aggregation is deterministic");
}

// -- Nonce Generation ---------------------------------------------------------

static void test_nonce_gen() {
    printf("\n--- Nonce Generation ---\n");

    std::array<uint8_t, 32> sk_bytes{};
    hex_to_bytes("0000000000000000000000000000000000000000000000000000000000000001",
                 sk_bytes.data(), 32);
    Scalar const sk = Scalar::from_bytes(sk_bytes);

    auto pk = get_xonly_pubkey(sk);

    // Use pk as agg_pub_key for simplicity (single signer)
    std::array<uint8_t, 32> msg{};
    for (std::size_t i = 0; i < 32; ++i) msg[i] = static_cast<uint8_t>(i);

    std::array<uint8_t, 32> extra{};
    extra[0] = 0x42;

    auto [sec_nonce, pub_nonce] = musig2_nonce_gen(sk, pk, pk, msg, extra.data());

    // Both nonce scalars should be non-zero
    CHECK(!sec_nonce.k1.is_zero() && !sec_nonce.k2.is_zero(),
          "Secret nonces are non-zero");

    // Public nonces should be valid compressed points (start with 0x02 or 0x03)
    CHECK((pub_nonce.R1[0] == 0x02 || pub_nonce.R1[0] == 0x03),
          "R1 is valid compressed point");
    CHECK((pub_nonce.R2[0] == 0x02 || pub_nonce.R2[0] == 0x03),
          "R2 is valid compressed point");

    // Different extra_input -> different nonces
    std::array<uint8_t, 32> extra2{};
    extra2[0] = 0x43;
    auto [sec2, pub2] = musig2_nonce_gen(sk, pk, pk, msg, extra2.data());
    (void)sec2;
    CHECK(pub_nonce.R1 != pub2.R1, "Different extra -> different nonce");
}

// -- Full 2-of-2 Signing -----------------------------------------------------

static void test_2of2_signing() {
    printf("\n--- 2-of-2 MuSig2 Signing ---\n");

    // Setup: 2 signers
    std::array<uint8_t, 32> sk1_bytes{}, sk2_bytes{};
    hex_to_bytes("b5a2c27e2e48ed400a0e06a60f5c5e9e5fcf0a1e7e2d4a7e8f9a0b1c2d3e4f50",
                 sk1_bytes.data(), 32);
    hex_to_bytes("c4a3d38f3f59fe511b1f17b71f6d6f0f6fed1b2f8f3e5b8f9f0b1c2d3e4f5061",
                 sk2_bytes.data(), 32);

    Scalar const s1 = Scalar::from_bytes(sk1_bytes);
    Scalar const s2 = Scalar::from_bytes(sk2_bytes);

    auto pk1 = get_xonly_pubkey(s1);
    auto pk2 = get_xonly_pubkey(s2);

    // Step 1: Key aggregation
    std::vector<std::array<uint8_t, 32>> const pubkeys = {pk1, pk2};
    auto key_ctx = musig2_key_agg(pubkeys);

    // Step 2: Message
    std::array<uint8_t, 32> msg{};
    hex_to_bytes("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                 msg.data(), 32);

    // Step 3: Nonce generation
    std::array<uint8_t, 32> extra1{}, extra2{};
    extra1[0] = 1;
    extra2[0] = 2;
    auto [sec1, pub1] = musig2_nonce_gen(s1, pk1, key_ctx.Q_x, msg, extra1.data());
    auto [sec2, pub2] = musig2_nonce_gen(s2, pk2, key_ctx.Q_x, msg, extra2.data());

    // Step 4: Nonce aggregation
    std::vector<MuSig2PubNonce> const pub_nonces = {pub1, pub2};
    auto agg_nonce = musig2_nonce_agg(pub_nonces);

    // Step 5: Start signing session
    auto session = musig2_start_sign_session(agg_nonce, key_ctx, msg);

    // Step 6: Create partial signatures
    auto psig1 = musig2_partial_sign(sec1, s1, key_ctx, session, 0);
    auto psig2 = musig2_partial_sign(sec2, s2, key_ctx, session, 1);

    // Step 7: Verify partial signatures
    bool const v1 = musig2_partial_verify(psig1, pub1, pk1, key_ctx, session, 0);
    bool const v2 = musig2_partial_verify(psig2, pub2, pk2, key_ctx, session, 1);
    CHECK(v1, "Partial sig 1 verifies");
    CHECK(v2, "Partial sig 2 verifies");

    // Step 8: Aggregate into final signature
    std::vector<Scalar> const psigs = {psig1, psig2};
    auto sig = musig2_partial_sig_agg(psigs, session);

    // Step 9: Verify the aggregate signature with BIP-340 Schnorr
    auto schnorr_sig = SchnorrSignature::from_bytes(sig);
    bool const schnorr_ok = schnorr_verify(key_ctx.Q_x, msg, schnorr_sig);
    CHECK(schnorr_ok, "Final MuSig2 sig verifies as standard Schnorr");
}

// -- Full 3-of-3 Signing -----------------------------------------------------

static void test_3of3_signing() {
    printf("\n--- 3-of-3 MuSig2 Signing ---\n");

    std::array<uint8_t, 32> sk_bytes[3];
    hex_to_bytes("0101010101010101010101010101010101010101010101010101010101010101",
                 sk_bytes[0].data(), 32);
    hex_to_bytes("0202020202020202020202020202020202020202020202020202020202020202",
                 sk_bytes[1].data(), 32);
    hex_to_bytes("0303030303030303030303030303030303030303030303030303030303030303",
                 sk_bytes[2].data(), 32);

    Scalar sk[3];
    std::array<uint8_t, 32> pk[3];
    std::vector<std::array<uint8_t, 32>> pubkeys;

    for (std::size_t i = 0; i < 3; ++i) {
        sk[i] = Scalar::from_bytes(sk_bytes[i]);
        pk[i] = get_xonly_pubkey(sk[i]);
        pubkeys.push_back(pk[i]);
    }

    auto key_ctx = musig2_key_agg(pubkeys);
    CHECK(!key_ctx.Q.is_infinity(), "3-of-3 agg key valid");

    std::array<uint8_t, 32> msg{};
    hex_to_bytes("deadbeefcafebabe0000000000000000deadbeefcafebabe0000000000000000",
                 msg.data(), 32);

    // Nonce gen
    MuSig2SecNonce sec_nonces[3];
    std::vector<MuSig2PubNonce> pub_nonces;

    for (std::size_t i = 0; i < 3; ++i) {
        std::array<uint8_t, 32> extra{};
        extra[0] = static_cast<uint8_t>(i + 1);
        auto [sn, pn] = musig2_nonce_gen(sk[i], pk[i], key_ctx.Q_x, msg, extra.data());
        sec_nonces[i] = sn;
        pub_nonces.push_back(pn);
    }

    auto agg_nonce = musig2_nonce_agg(pub_nonces);
    auto session = musig2_start_sign_session(agg_nonce, key_ctx, msg);

    // Partial signs
    std::vector<Scalar> psigs;
    for (std::size_t i = 0; i < 3; ++i) {
        auto ps = musig2_partial_sign(sec_nonces[i], sk[i], key_ctx, session, i);
        psigs.push_back(ps);

        bool const pv = musig2_partial_verify(ps, pub_nonces[i], pk[i], key_ctx, session, i);
        CHECK(pv, (std::string("3-of-3 partial sig ") + std::to_string(i) + " verifies").c_str());
    }

    auto sig = musig2_partial_sig_agg(psigs, session);

    auto schnorr_sig = SchnorrSignature::from_bytes(sig);
    bool const final_ok = schnorr_verify(key_ctx.Q_x, msg, schnorr_sig);
    CHECK(final_ok, "3-of-3 MuSig2 sig verifies as Schnorr");
}

// -- Edge Cases ---------------------------------------------------------------

static void test_edge_cases() {
    printf("\n--- Edge Cases ---\n");

    // Single signer (n=1) should still work
    std::array<uint8_t, 32> sk_bytes{};
    hex_to_bytes("0000000000000000000000000000000000000000000000000000000000000007",
                 sk_bytes.data(), 32);
    Scalar const s = Scalar::from_bytes(sk_bytes);
    auto pk = get_xonly_pubkey(s);

    std::vector<std::array<uint8_t, 32>> const pubkeys = {pk};
    auto key_ctx = musig2_key_agg(pubkeys);
    CHECK(!key_ctx.Q.is_infinity(), "Single-signer agg key valid");

    std::array<uint8_t, 32> msg{};
    msg[0] = 0xAA;

    std::array<uint8_t, 32> extra{};
    extra[0] = 0xBB;
    auto [sec, pub] = musig2_nonce_gen(s, pk, key_ctx.Q_x, msg, extra.data());

    std::vector<MuSig2PubNonce> const pub_nonces = {pub};
    auto agg = musig2_nonce_agg(pub_nonces);
    auto session = musig2_start_sign_session(agg, key_ctx, msg);

    auto psig = musig2_partial_sign(sec, s, key_ctx, session, 0);
    bool const pv = musig2_partial_verify(psig, pub, pk, key_ctx, session, 0);
    CHECK(pv, "Single-signer partial verify OK");

    std::vector<Scalar> const psigs_vec = {psig};
    auto sig = musig2_partial_sig_agg(psigs_vec, session);

    auto schnorr_sig = SchnorrSignature::from_bytes(sig);
    bool const schnorr_ok = schnorr_verify(key_ctx.Q_x, msg, schnorr_sig);
    CHECK(schnorr_ok, "Single-signer MuSig2 = valid Schnorr sig");
}

// -- Main ---------------------------------------------------------------------

int test_musig2_run() {
    printf("=== MuSig2 Multi-Signature Tests ===\n");

    test_key_aggregation();
    test_nonce_gen();
    test_2of2_signing();
    test_3of3_signing();
    test_edge_cases();

    printf("\n=== Results: %d/%d passed ===\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
