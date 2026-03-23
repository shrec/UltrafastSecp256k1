// ============================================================================
// Test: C ABI FFI coverage – BIP-324, AEAD, ElligatorSwift, ZK Range, misc
// ============================================================================
// Exercises ufsecp_* C ABI functions that had zero test coverage.
// Roundtrip tests for: AEAD, BIP-324 session, ElligatorSwift, ZK range,
// Pedersen switch commit, ufsecp_ctx_size.
// ============================================================================

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <array>

// C ABI header
#include "ufsecp/ufsecp.h"

static int tests_run = 0;
static int tests_passed = 0;

#define CHECK(cond, msg) do { \
    ++tests_run; \
    if (cond) { ++tests_passed; } \
    else { std::printf("  [FAIL] %s\n", msg); } \
} while(0)

// ============================================================================
// ufsecp_ctx_size
// ============================================================================

static void test_ctx_size() {
    std::printf("\n=== FFI: ufsecp_ctx_size ===\n");
    std::size_t sz = ufsecp_ctx_size();
    CHECK(sz > 0, "ctx_size > 0");
    CHECK(sz < 1024 * 1024, "ctx_size < 1MB (sanity)");
}

// ============================================================================
// Pedersen switch commit
// ============================================================================

static void test_pedersen_switch_commit(ufsecp_ctx* ctx) {
    std::printf("\n=== FFI: ufsecp_pedersen_switch_commit ===\n");

    // value = 42, blinding and switch_blind are deterministic
    std::uint8_t value[32] = {};
    value[31] = 42;

    std::uint8_t blinding[32] = {};
    blinding[0] = 0x01;
    blinding[31] = 0x77;

    std::uint8_t switch_blind[32] = {};
    switch_blind[0] = 0x02;
    switch_blind[31] = 0x88;

    std::uint8_t commitment[33] = {};
    auto err = ufsecp_pedersen_switch_commit(ctx, value, blinding, switch_blind, commitment);
    CHECK(err == UFSECP_OK, "switch_commit returns OK");
    // Check it's a valid compressed point prefix
    CHECK(commitment[0] == 0x02 || commitment[0] == 0x03,
          "commitment has valid compressed prefix");

    // Same inputs produce same output
    std::uint8_t commitment2[33] = {};
    ufsecp_pedersen_switch_commit(ctx, value, blinding, switch_blind, commitment2);
    CHECK(std::memcmp(commitment, commitment2, 33) == 0,
          "deterministic: same inputs → same commitment");
}

// ============================================================================
// ZK Range prove/verify
// ============================================================================

static void test_zk_range(ufsecp_ctx* ctx) {
    std::printf("\n=== FFI: ufsecp_zk_range_prove / verify ===\n");

    // Create a Pedersen commitment for value 1000
    std::uint8_t blinding[32] = {};
    blinding[0] = 0x03;
    blinding[31] = 0xAA;

    std::uint8_t aux_rand[32] = {};
    aux_rand[0] = 0x04;
    aux_rand[31] = 0xBB;

    // Need commitment: C = value*H + blinding*G
    std::uint8_t value_bytes[32] = {};
    std::uint64_t val = 1000;
    // value is scalar but for Pedersen we just need blinding + value
    for (int i = 7; i >= 0; --i) {
        value_bytes[31 - i] = 0;
    }
    value_bytes[24] = static_cast<std::uint8_t>((val >> 56) & 0xFF);
    value_bytes[25] = static_cast<std::uint8_t>((val >> 48) & 0xFF);
    value_bytes[26] = static_cast<std::uint8_t>((val >> 40) & 0xFF);
    value_bytes[27] = static_cast<std::uint8_t>((val >> 32) & 0xFF);
    value_bytes[28] = static_cast<std::uint8_t>((val >> 24) & 0xFF);
    value_bytes[29] = static_cast<std::uint8_t>((val >> 16) & 0xFF);
    value_bytes[30] = static_cast<std::uint8_t>((val >>  8) & 0xFF);
    value_bytes[31] = static_cast<std::uint8_t>((val >>  0) & 0xFF);

    std::uint8_t commitment[33] = {};
    auto err = ufsecp_pedersen_commit(ctx, value_bytes, blinding, commitment);
    CHECK(err == UFSECP_OK, "pedersen_commit for range proof");

    // Prove
    std::uint8_t proof[UFSECP_ZK_RANGE_PROOF_MAX_LEN] = {};
    std::size_t proof_len = sizeof(proof);
    err = ufsecp_zk_range_prove(ctx, val, blinding, commitment, aux_rand,
                                proof, &proof_len);
    CHECK(err == UFSECP_OK, "range_prove returns OK");
    CHECK(proof_len > 0, "proof has non-zero length");
    CHECK(proof_len <= UFSECP_ZK_RANGE_PROOF_MAX_LEN, "proof <= max len");

    // Verify
    err = ufsecp_zk_range_verify(ctx, commitment, proof, proof_len);
    CHECK(err == UFSECP_OK, "range_verify accepts valid proof");

    // Tamper: flip a byte in the proof
    proof[proof_len / 2] ^= 0xFF;
    err = ufsecp_zk_range_verify(ctx, commitment, proof, proof_len);
    CHECK(err != UFSECP_OK, "range_verify rejects tampered proof");
}

#ifdef SECP256K1_BIP324

// ============================================================================
// AEAD ChaCha20-Poly1305 roundtrip
// ============================================================================

static void test_aead_roundtrip() {
    std::printf("\n=== FFI: AEAD ChaCha20-Poly1305 roundtrip ===\n");

    std::uint8_t key[32] = {};
    for (int i = 0; i < 32; ++i) key[i] = static_cast<std::uint8_t>(i + 1);

    std::uint8_t nonce[12] = {};
    nonce[0] = 0x07; nonce[11] = 0x42;

    const std::uint8_t aad[] = "additional-data";
    const std::uint8_t plaintext[] = "Hello AEAD world! Testing roundtrip.";
    constexpr std::size_t pt_len = sizeof(plaintext) - 1;

    std::uint8_t ciphertext[pt_len] = {};
    std::uint8_t tag[16] = {};

    auto err = ufsecp_aead_chacha20_poly1305_encrypt(
        key, nonce, aad, sizeof(aad) - 1,
        plaintext, pt_len, ciphertext, tag);
    CHECK(err == UFSECP_OK, "aead_encrypt ok");
    CHECK(std::memcmp(ciphertext, plaintext, pt_len) != 0, "ciphertext differs from plaintext");

    // Decrypt
    std::uint8_t recovered[pt_len] = {};
    err = ufsecp_aead_chacha20_poly1305_decrypt(
        key, nonce, aad, sizeof(aad) - 1,
        ciphertext, pt_len, tag, recovered);
    CHECK(err == UFSECP_OK, "aead_decrypt ok");
    CHECK(std::memcmp(recovered, plaintext, pt_len) == 0, "plaintext recovered");

    // Tamper ciphertext → auth failure
    ciphertext[0] ^= 0xFF;
    err = ufsecp_aead_chacha20_poly1305_decrypt(
        key, nonce, aad, sizeof(aad) - 1,
        ciphertext, pt_len, tag, recovered);
    CHECK(err == UFSECP_ERR_VERIFY_FAIL, "aead_decrypt maps tampered ciphertext to verify failure");

    ciphertext[0] ^= 0xFF;
    std::uint8_t bad_tag[16] = {};
    std::memcpy(bad_tag, tag, sizeof(tag));
    bad_tag[15] ^= 0x01;
    err = ufsecp_aead_chacha20_poly1305_decrypt(
        key, nonce, aad, sizeof(aad) - 1,
        ciphertext, pt_len, bad_tag, recovered);
    CHECK(err == UFSECP_ERR_VERIFY_FAIL, "aead_decrypt maps tampered tag to verify failure");
}

// ============================================================================
// ElligatorSwift create + XDH
// ============================================================================

static void test_ellswift(ufsecp_ctx* ctx) {
    std::printf("\n=== FFI: ElligatorSwift create + XDH ===\n");

    // Two private keys
    std::uint8_t sk_a[32] = {};
    sk_a[0] = 0x01; sk_a[31] = 0x11;
    std::uint8_t sk_b[32] = {};
    sk_b[0] = 0x02; sk_b[31] = 0x22;

    // Create encodings
    std::uint8_t ell_a[64] = {}, ell_b[64] = {};
    auto err = ufsecp_ellswift_create(ctx, sk_a, ell_a);
    CHECK(err == UFSECP_OK, "ellswift_create A ok");
    err = ufsecp_ellswift_create(ctx, sk_b, ell_b);
    CHECK(err == UFSECP_OK, "ellswift_create B ok");

    // XDH from A's perspective (initiator)
    std::uint8_t secret_a[32] = {};
    err = ufsecp_ellswift_xdh(ctx, ell_a, ell_b, sk_a, 1, secret_a);
    CHECK(err == UFSECP_OK, "ellswift_xdh A ok");

    // XDH from B's perspective (responder)
    std::uint8_t secret_b[32] = {};
    err = ufsecp_ellswift_xdh(ctx, ell_a, ell_b, sk_b, 0, secret_b);
    CHECK(err == UFSECP_OK, "ellswift_xdh B ok");

    // Both sides derive the same shared secret
    CHECK(std::memcmp(secret_a, secret_b, 32) == 0,
          "ellswift_xdh: initiator and responder derive same secret");
}

// ============================================================================
// BIP-324 session: create → handshake → encrypt → decrypt roundtrip
// ============================================================================

static void test_bip324_session(ufsecp_ctx* ctx) {
    std::printf("\n=== FFI: BIP-324 session roundtrip ===\n");

    // Create initiator session
    ufsecp_bip324_session* session_a = nullptr;
    std::uint8_t ell_a[64] = {};
    auto err = ufsecp_bip324_create(ctx, 1, &session_a, ell_a);
    CHECK(err == UFSECP_OK, "bip324_create initiator ok");
    CHECK(session_a != nullptr, "initiator session non-null");

    // Create responder session
    ufsecp_bip324_session* session_b = nullptr;
    std::uint8_t ell_b[64] = {};
    err = ufsecp_bip324_create(ctx, 0, &session_b, ell_b);
    CHECK(err == UFSECP_OK, "bip324_create responder ok");

    // Handshake: initiator receives responder's encoding
    std::uint8_t sid_a[32] = {};
    err = ufsecp_bip324_handshake(session_a, ell_b, sid_a);
    CHECK(err == UFSECP_OK, "handshake initiator ok");

    // Handshake: responder receives initiator's encoding
    std::uint8_t sid_b[32] = {};
    err = ufsecp_bip324_handshake(session_b, ell_a, sid_b);
    CHECK(err == UFSECP_OK, "handshake responder ok");

    // Session IDs should match
    CHECK(std::memcmp(sid_a, sid_b, 32) == 0, "session IDs match");

    // Encrypt a message (initiator → responder)
    const std::uint8_t msg[] = "BIP-324 test payload";
    constexpr std::size_t msg_len = sizeof(msg) - 1;

    std::vector<std::uint8_t> ct(msg_len + 64);  // generous buffer
    std::size_t ct_len = ct.size();
    err = ufsecp_bip324_encrypt(session_a, msg, msg_len, ct.data(), &ct_len);
    CHECK(err == UFSECP_OK, "bip324_encrypt ok");
    CHECK(ct_len > msg_len, "ciphertext longer than plaintext");

    // Decrypt (responder)
    std::vector<std::uint8_t> pt(msg_len + 64);
    std::size_t pt_len = pt.size();
    err = ufsecp_bip324_decrypt(session_b, ct.data(), ct_len, pt.data(), &pt_len);
    CHECK(err == UFSECP_OK, "bip324_decrypt ok");
    CHECK(pt_len == msg_len, "decrypted length matches");
    CHECK(std::memcmp(pt.data(), msg, msg_len) == 0, "decrypted content matches");

    // Cleanup
    ufsecp_bip324_destroy(session_a);
    ufsecp_bip324_destroy(session_b);
    // Double-free safety: destroy(NULL) must be safe
    ufsecp_bip324_destroy(nullptr);
    CHECK(true, "bip324_destroy(NULL) is safe");
}

#endif /* SECP256K1_BIP324 */

// ============================================================================
// Entry point
// ============================================================================

int test_ffi_coverage_run() {
    std::printf("\n========== FFI C ABI Coverage Tests ==========\n");

    ufsecp_ctx* ctx = nullptr;
    auto create_err = ufsecp_ctx_create(&ctx);
    CHECK(create_err == UFSECP_OK && ctx != nullptr, "ctx_create");

    test_ctx_size();
    test_pedersen_switch_commit(ctx);
    test_zk_range(ctx);

#ifdef SECP256K1_BIP324
    test_aead_roundtrip();
    test_ellswift(ctx);
    test_bip324_session(ctx);
#endif

    ufsecp_ctx_destroy(ctx);

    std::printf("\n  FFI coverage: %d/%d passed\n", tests_passed, tests_run);
    return tests_run - tests_passed;
}

#ifdef STANDALONE_TEST
int main() {
    return test_ffi_coverage_run();
}
#endif
