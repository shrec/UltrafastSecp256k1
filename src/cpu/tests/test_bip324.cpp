// ============================================================================
// Test: BIP-324 Encrypted Transport Protocol (v2 P2P)
// ============================================================================
// Covers: ElligatorSwift, ChaCha20-Poly1305 AEAD, HKDF-SHA256,
//         Bip324Cipher, Bip324Session — full transport stack.
// ============================================================================

#include "secp256k1/bip324.hpp"
#include "secp256k1/chacha20_poly1305.hpp"
#include "secp256k1/ellswift.hpp"
#include "secp256k1/hkdf.hpp"
#include "secp256k1/scalar.hpp"
#include "secp256k1/point.hpp"

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <array>
#include <vector>
#include <algorithm>

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

// Fixed test keys
static const std::uint8_t KEY_A[32] = {
    0xe8,0xf3,0x2e,0x72,0x3d,0xec,0xf4,0x05,
    0x1a,0xef,0xac,0x8e,0x2c,0x93,0xc9,0xc5,
    0xb2,0x14,0x31,0x38,0x17,0xcd,0xb0,0x1a,
    0x14,0x94,0xb9,0x17,0xc8,0x43,0x6b,0x35
};
static const std::uint8_t KEY_B[32] = {
    0xaa,0xbb,0xcc,0xdd,0x11,0x22,0x33,0x44,
    0x55,0x66,0x77,0x88,0x99,0x00,0xab,0xcd,
    0xef,0x01,0x23,0x45,0x67,0x89,0xab,0xcd,
    0xef,0xfe,0xdc,0xba,0x98,0x76,0x54,0x32
};

static std::vector<std::uint8_t> decrypt_or_empty(Bip324Session& session,
                                                  const std::vector<std::uint8_t>& packet) {
    std::vector<std::uint8_t> plaintext;
    if (!session.decrypt(packet.data(), packet.data() + 3, packet.size() - 3, plaintext)) {
        return {};
    }
    return plaintext;
}

static std::vector<std::uint8_t> decrypt_cipher_or_empty(Bip324Cipher& cipher,
                                                         const std::vector<std::uint8_t>& packet) {
    std::vector<std::uint8_t> plaintext;
    if (!cipher.decrypt(nullptr, 0, packet.data(), packet.data() + 3, packet.size() - 3, plaintext)) {
        return {};
    }
    return plaintext;
}

// ============================================================================
// 1. HKDF-SHA256 tests
// ============================================================================

static void test_hkdf() {
    std::printf("\n--- HKDF-SHA256 ---\n");

    // Extract with known salt and IKM
    const char* salt = "bitcoin_v2_shared_secret";
    std::uint8_t ikm[32] = {};
    for (int i = 0; i < 32; ++i) ikm[i] = static_cast<std::uint8_t>(i);

    auto prk = hkdf_sha256_extract(
        reinterpret_cast<const std::uint8_t*>(salt), std::strlen(salt),
        ikm, 32);
    CHECK(prk.size() == 32, "HKDF extract produces 32-byte PRK");

    // Extract is deterministic
    auto prk2 = hkdf_sha256_extract(
        reinterpret_cast<const std::uint8_t*>(salt), std::strlen(salt),
        ikm, 32);
    CHECK(prk == prk2, "HKDF extract is deterministic");

    // Expand with different info strings produces different keys
    std::uint8_t out_a[32], out_b[32];
    bool ok_a = hkdf_sha256_expand(prk.data(),
        reinterpret_cast<const std::uint8_t*>("initiator_L"), 11, out_a, 32);
    bool ok_b = hkdf_sha256_expand(prk.data(),
        reinterpret_cast<const std::uint8_t*>("responder_L"), 11, out_b, 32);
    CHECK(ok_a && ok_b, "HKDF expand succeeds");
    CHECK(std::memcmp(out_a, out_b, 32) != 0, "Different info → different keys");

    // Expand produces same output for same inputs
    std::uint8_t out_a2[32];
    hkdf_sha256_expand(prk.data(),
        reinterpret_cast<const std::uint8_t*>("initiator_L"), 11, out_a2, 32);
    CHECK(std::memcmp(out_a, out_a2, 32) == 0, "HKDF expand is deterministic");

    // HMAC-SHA256 basic test
    auto mac = hmac_sha256(ikm, 32, reinterpret_cast<const std::uint8_t*>("test"), 4);
    CHECK(mac.size() == 32, "HMAC-SHA256 produces 32-byte output");
    auto mac2 = hmac_sha256(ikm, 32, reinterpret_cast<const std::uint8_t*>("test"), 4);
    CHECK(mac == mac2, "HMAC-SHA256 is deterministic");
}

// ============================================================================
// 2. ChaCha20-Poly1305 AEAD tests
// ============================================================================

static void test_chacha20_poly1305() {
    std::printf("\n--- ChaCha20-Poly1305 AEAD ---\n");

    std::uint8_t key[32], nonce[12];
    for (int i = 0; i < 32; ++i) key[i] = static_cast<std::uint8_t>(i);
    std::memset(nonce, 0, 12);

    // Encrypt → decrypt roundtrip
    const char* msg = "Hello BIP-324 transport!";
    std::size_t msg_len = std::strlen(msg);

    std::vector<std::uint8_t> ct(msg_len);
    std::uint8_t tag[16];
    aead_chacha20_poly1305_encrypt(key, nonce, nullptr, 0,
        reinterpret_cast<const std::uint8_t*>(msg), msg_len,
        ct.data(), tag);

    // Ciphertext differs from plaintext
    CHECK(std::memcmp(ct.data(), msg, msg_len) != 0, "AEAD ciphertext differs from plaintext");

    // Decrypt recovers plaintext
    std::vector<std::uint8_t> pt(msg_len);
    bool ok = aead_chacha20_poly1305_decrypt(key, nonce, nullptr, 0,
        ct.data(), msg_len, tag, pt.data());
    CHECK(ok, "AEAD decrypt succeeds");
    CHECK(std::memcmp(pt.data(), msg, msg_len) == 0, "AEAD roundtrip recovers plaintext");

    // Tampered ciphertext fails auth
    ct[0] ^= 0xFF;
    ok = aead_chacha20_poly1305_decrypt(key, nonce, nullptr, 0,
        ct.data(), msg_len, tag, pt.data());
    CHECK(!ok, "AEAD tampered ciphertext fails auth");
    ct[0] ^= 0xFF;  // restore

    // Tampered tag fails auth
    tag[0] ^= 0x01;
    ok = aead_chacha20_poly1305_decrypt(key, nonce, nullptr, 0,
        ct.data(), msg_len, tag, pt.data());
    CHECK(!ok, "AEAD tampered tag fails auth");
    tag[0] ^= 0x01;

    // Wrong key fails auth
    std::uint8_t bad_key[32];
    std::memcpy(bad_key, key, 32);
    bad_key[0] ^= 0x01;
    ok = aead_chacha20_poly1305_decrypt(bad_key, nonce, nullptr, 0,
        ct.data(), msg_len, tag, pt.data());
    CHECK(!ok, "AEAD wrong key fails auth");

    // With AAD
    const std::uint8_t aad[] = {0xDE, 0xAD, 0xBE, 0xEF};
    aead_chacha20_poly1305_encrypt(key, nonce, aad, 4,
        reinterpret_cast<const std::uint8_t*>(msg), msg_len,
        ct.data(), tag);
    ok = aead_chacha20_poly1305_decrypt(key, nonce, aad, 4,
        ct.data(), msg_len, tag, pt.data());
    CHECK(ok, "AEAD with AAD roundtrip succeeds");

    // Wrong AAD fails
    std::uint8_t bad_aad[] = {0xDE, 0xAD, 0xBE, 0x00};
    ok = aead_chacha20_poly1305_decrypt(key, nonce, bad_aad, 4,
        ct.data(), msg_len, tag, pt.data());
    CHECK(!ok, "AEAD wrong AAD fails auth");

    // Empty plaintext
    std::uint8_t empty_ct[1], empty_tag[16];
    aead_chacha20_poly1305_encrypt(key, nonce, nullptr, 0,
        nullptr, 0, empty_ct, empty_tag);
    ok = aead_chacha20_poly1305_decrypt(key, nonce, nullptr, 0,
        nullptr, 0, empty_tag, nullptr);
    CHECK(ok, "AEAD empty plaintext roundtrip");

    // ChaCha20 raw block is deterministic
    std::uint8_t block1[64], block2[64];
    chacha20_block(key, nonce, 0, block1);
    chacha20_block(key, nonce, 0, block2);
    CHECK(std::memcmp(block1, block2, 64) == 0, "ChaCha20 block is deterministic");

    // Different nonces produce different blocks
    std::uint8_t nonce2[12] = {};
    nonce2[0] = 1;
    chacha20_block(key, nonce2, 0, block2);
    CHECK(std::memcmp(block1, block2, 64) != 0, "ChaCha20 different nonces → different output");
}

// ============================================================================
// 3. ElligatorSwift tests
// ============================================================================

static void test_ellswift() {
    std::printf("\n--- ElligatorSwift ---\n");

    Scalar sk = Scalar::from_bytes(KEY_A);

    // Create encoding
    auto enc = ellswift_create(sk);
    CHECK(enc.size() == 64, "ellswift_create produces 64-byte encoding");

    // Encoding is not all-zero
    bool all_zero = true;
    for (auto b : enc) { if (b != 0) { all_zero = false; break; } }
    CHECK(!all_zero, "ElligatorSwift encoding is non-trivial");

    // Decode recovers the x-coordinate
    auto x = ellswift_decode(enc.data());
    auto P = Point::generator().scalar_mul(sk);
    CHECK(x == P.x(), "ElligatorSwift decode recovers correct x-coordinate");

    // XDH: shared secret is same from both sides
    Scalar sk_b = Scalar::from_bytes(KEY_B);
    auto enc_a = ellswift_create(sk);
    auto enc_b = ellswift_create(sk_b);

    auto secret_init = ellswift_xdh(enc_a.data(), enc_b.data(), sk, true);
    auto secret_resp = ellswift_xdh(enc_a.data(), enc_b.data(), sk_b, false);
    CHECK(secret_init == secret_resp, "ElligatorSwift XDH: initiator and responder derive same secret");

    // XDH with swapped roles produces different secret
    auto secret_swapped = ellswift_xdh(enc_b.data(), enc_a.data(), sk, true);
    CHECK(secret_init != secret_swapped, "ElligatorSwift XDH: swapped encodings → different secret");
}

// ============================================================================
// 4. Bip324Cipher tests
// ============================================================================

static void test_bip324_cipher() {
    std::printf("\n--- Bip324Cipher ---\n");

    Bip324Cipher cipher;
    std::uint8_t key[32];
    for (int i = 0; i < 32; ++i) key[i] = static_cast<std::uint8_t>(i + 0x42);
    cipher.init(key);

    CHECK(cipher.packet_counter() == 0, "Cipher: initial counter is 0");

    // Encrypt a packet
    const std::uint8_t payload[] = "test payload for BIP-324";
    std::size_t payload_len = sizeof(payload) - 1;  // exclude null terminator

    auto pkt = cipher.encrypt(nullptr, 0, payload, payload_len);
    // Wire format: 3B enc length + payload + 16B tag
    CHECK(pkt.size() == 3 + payload_len + 16, "Cipher: wire packet size correct");
    CHECK(cipher.packet_counter() == 1, "Cipher: counter incremented after encrypt");

    // Decrypt the packet
    Bip324Cipher recv_cipher;
    recv_cipher.init(key);
    auto dec = decrypt_cipher_or_empty(recv_cipher, pkt);
    CHECK(!dec.empty(), "Cipher: decrypt succeeds");
    CHECK(dec.size() == payload_len, "Cipher: decrypted size matches");
    CHECK(std::memcmp(dec.data(), payload, payload_len) == 0, "Cipher: roundtrip correct");

    // Second packet uses different nonce (counter=1 vs counter=0)
    auto pkt2 = cipher.encrypt(nullptr, 0, payload, payload_len);
    CHECK(pkt != pkt2, "Cipher: different counter → different ciphertext");

    // Decrypt with wrong counter fails
    Bip324Cipher misaligned;
    misaligned.init(key);
    // Skip one decrypt to advance counter
    auto skipped = decrypt_cipher_or_empty(misaligned, pkt);
    CHECK(!skipped.empty(), "Cipher: initial aligned decrypt succeeds");
    // Now try to decrypt pkt (counter=0) with counter=1
    auto bad_dec = decrypt_cipher_or_empty(misaligned, pkt);
    // This should fail because the nonce doesn't match
    // (pkt was encrypted with counter=0, misaligned is now at counter=1, 
    //  but pkt2 was encrypted with counter=1, and we used pkt for counter=0 slot)
    // Actually the misaligned cipher just decrypted pkt at counter=0, so it's now at counter=1
    // Let's try decrypting pkt again (counter=0 data with counter=1 state)
    auto bad_dec2 = decrypt_cipher_or_empty(misaligned, pkt);
    CHECK(bad_dec.empty(), "Cipher: first misaligned decrypt fails auth");
    CHECK(bad_dec2.empty(), "Cipher: misaligned counter fails auth");
}

// ============================================================================
// 5. Bip324Session tests
// ============================================================================

static void test_bip324_session() {
    std::printf("\n--- Bip324Session ---\n");

    // Create sessions with deterministic keys
    Bip324Session initiator(true, KEY_A);
    Bip324Session responder(false, KEY_B);

    // Before handshake
    CHECK(!initiator.is_established(), "Session: initiator not established before handshake");
    CHECK(!responder.is_established(), "Session: responder not established before handshake");

    // ElligatorSwift encoding is 64 bytes
    auto& enc_init = initiator.our_ellswift_encoding();
    auto& enc_resp = responder.our_ellswift_encoding();
    CHECK(enc_init.size() == 64, "Session: initiator encoding is 64 bytes");
    CHECK(enc_resp.size() == 64, "Session: responder encoding is 64 bytes");

    // Complete handshake
    bool hs_resp = responder.complete_handshake(enc_init.data());
    bool hs_init = initiator.complete_handshake(enc_resp.data());
    CHECK(hs_init, "Session: initiator handshake succeeds");
    CHECK(hs_resp, "Session: responder handshake succeeds");
    CHECK(initiator.is_established(), "Session: initiator established after handshake");
    CHECK(responder.is_established(), "Session: responder established after handshake");

    // Session IDs match
    CHECK(initiator.session_id() == responder.session_id(),
          "Session: both sides derive same session_id");

    // Session ID is non-zero
    bool sid_zero = true;
    for (auto b : initiator.session_id()) { if (b != 0) { sid_zero = false; break; } }
    CHECK(!sid_zero, "Session: session_id is non-zero");

    // Encrypt/decrypt: initiator → responder
    const std::uint8_t msg[] = "Hello from initiator!";
    auto pkt = initiator.encrypt(msg, sizeof(msg) - 1);
    CHECK(!pkt.empty(), "Session: encrypt produces data");

    auto dec = decrypt_or_empty(responder, pkt);
    CHECK(!dec.empty(), "Session: responder decrypts successfully");
    CHECK(dec.size() == sizeof(msg) - 1, "Session: decrypted size matches");
    CHECK(std::memcmp(dec.data(), msg, sizeof(msg) - 1) == 0,
          "Session: initiator→responder roundtrip correct");

    // Encrypt/decrypt: responder → initiator
    const std::uint8_t reply[] = "Reply from responder!";
    auto rpkt = responder.encrypt(reply, sizeof(reply) - 1);
    auto rdec = decrypt_or_empty(initiator, rpkt);
    CHECK(!rdec.empty(), "Session: initiator decrypts reply");
    CHECK(std::memcmp(rdec.data(), reply, sizeof(reply) - 1) == 0,
          "Session: responder→initiator roundtrip correct");

    // Cross-direction: initiator cannot decrypt its own packets
    auto own_pkt = initiator.encrypt(msg, sizeof(msg) - 1);
    auto cross = decrypt_or_empty(initiator, own_pkt);
    CHECK(cross.empty(), "Session: cannot decrypt own packets (different key direction)");
}

// ============================================================================
// 6. Multi-packet sequence test
// ============================================================================

static void test_bip324_sequence() {
    std::printf("\n--- BIP-324 Packet Sequence ---\n");

    Bip324Session init(true, KEY_A);
    Bip324Session resp(false, KEY_B);
    resp.complete_handshake(init.our_ellswift_encoding().data());
    init.complete_handshake(resp.our_ellswift_encoding().data());

    // Send 100 packets in each direction
    bool all_ok = true;
    for (std::size_t i = 0; i < 100; ++i) {
        std::uint8_t buf[64];
        std::size_t len = (i % 60) + 1;
        for (std::size_t j = 0; j < len; ++j)
            buf[j] = static_cast<std::uint8_t>((i + j) & 0xFF);

        // init → resp
        auto pkt = init.encrypt(buf, len);
        auto dec = decrypt_or_empty(resp, pkt);
        if (dec.size() != len || std::memcmp(dec.data(), buf, len) != 0) {
            all_ok = false;
            break;
        }

        // resp → init
        auto rpkt = resp.encrypt(buf, len);
        auto rdec = decrypt_or_empty(init, rpkt);
        if (rdec.size() != len || std::memcmp(rdec.data(), buf, len) != 0) {
            all_ok = false;
            break;
        }
    }
    CHECK(all_ok, "100 bidirectional packets: all roundtrips correct");

    // Verify nonce counters advanced correctly
    // Each side encrypted 100 packets, so each send cipher is at counter=100
    // (We can't read the counter directly, but we can verify continued operation)
    const std::uint8_t final_msg[] = "final";
    auto fp = init.encrypt(final_msg, 5);
    auto fd = decrypt_or_empty(resp, fp);
    CHECK(!fd.empty() && fd.size() == 5, "Post-sequence packet #101 decrypts correctly");
}

// ============================================================================
// 7. Deterministic key derivation test
// ============================================================================

static void test_bip324_determinism() {
    std::printf("\n--- BIP-324 Determinism ---\n");

    // ellswift_create uses CSPRNG randomness, so two sessions with the same
    // private key will produce different ElligatorSwift encodings and thus
    // different ECDH shared secrets.  What IS deterministic: given the same
    // encodings and keys, the derived session keys are identical.

    // Use a single session pair: verify that re-deriving from the same
    // raw material (encodings + privkeys) is deterministic.
    Bip324Session a1(true, KEY_A);
    Bip324Session b1(false, KEY_B);
    b1.complete_handshake(a1.our_ellswift_encoding().data());
    a1.complete_handshake(b1.our_ellswift_encoding().data());

    // Verify the ECDH + HKDF path is deterministic by checking that
    // both sides agree on the session ID (already tested elsewhere, but
    // confirms the derivation path is stable).
    CHECK(a1.session_id() == b1.session_id(),
          "Deterministic derivation: both sides agree on session_id");

    // Verify encrypt→decrypt is consistent across the session
    const std::uint8_t msg[] = "determinism test";
    auto pkt1 = a1.encrypt(msg, sizeof(msg) - 1);
    auto dec1 = decrypt_or_empty(b1, pkt1);
    CHECK(!dec1.empty() && std::memcmp(dec1.data(), msg, sizeof(msg) - 1) == 0,
          "First encrypt/decrypt in session is correct");

    auto pkt2 = a1.encrypt(msg, sizeof(msg) - 1);
    auto dec2 = decrypt_or_empty(b1, pkt2);
    CHECK(!dec2.empty() && std::memcmp(dec2.data(), msg, sizeof(msg) - 1) == 0,
          "Second encrypt/decrypt (different nonce) is correct");

    // Same plaintext with different counters produces different ciphertext
    CHECK(pkt1 != pkt2, "Same plaintext + different counter → different ciphertext");
}

// ============================================================================
// 8. Variable payload sizes
// ============================================================================

static void test_bip324_sizes() {
    std::printf("\n--- BIP-324 Variable Sizes ---\n");

    Bip324Session init(true, KEY_A);
    Bip324Session resp(false, KEY_B);
    resp.complete_handshake(init.our_ellswift_encoding().data());
    init.complete_handshake(resp.our_ellswift_encoding().data());

    // Test various payload sizes including edge cases
    static const std::size_t SIZES[] = {1, 2, 3, 15, 16, 17, 31, 32, 33,
                                         63, 64, 65, 127, 128, 255, 256,
                                         512, 1024, 4096};
    bool all_ok = true;
    for (auto sz : SIZES) {
        std::vector<std::uint8_t> payload(sz);
        for (std::size_t i = 0; i < sz; ++i)
            payload[i] = static_cast<std::uint8_t>(i & 0xFF);

        auto pkt = init.encrypt(payload.data(), sz);
        if (pkt.size() != 3 + sz + 16) { all_ok = false; break; }

        auto dec = decrypt_or_empty(resp, pkt);
        if (dec.size() != sz || std::memcmp(dec.data(), payload.data(), sz) != 0) {
            all_ok = false;
            break;
        }
    }
    char buf[128];
    std::snprintf(buf, sizeof(buf), "%zu different payload sizes: all correct",
                  sizeof(SIZES) / sizeof(SIZES[0]));
    CHECK(all_ok, buf);
}

// ============================================================================
// 9. Tamper resistance
// ============================================================================

static void test_bip324_tamper() {
    std::printf("\n--- BIP-324 Tamper Resistance ---\n");

    Bip324Session init(true, KEY_A);
    Bip324Session resp(false, KEY_B);
    resp.complete_handshake(init.our_ellswift_encoding().data());
    init.complete_handshake(resp.our_ellswift_encoding().data());

    const std::uint8_t msg[] = "tamper test message payload";
    auto pkt = init.encrypt(msg, sizeof(msg) - 1);

    // Tamper with encrypted length header (first 3 bytes)
    {
        auto bad = pkt;
        bad[0] ^= 0x01;
        auto dec = decrypt_or_empty(resp, bad);
        CHECK(dec.empty(), "Tampered header byte 0 → auth failure");
    }

    // Need fresh ciphertexts since responder counter advances even on failure
    // Re-create sessions for clean counters
    Bip324Session init2(true, KEY_A);
    Bip324Session resp2(false, KEY_B);
    resp2.complete_handshake(init2.our_ellswift_encoding().data());
    init2.complete_handshake(resp2.our_ellswift_encoding().data());

    auto pkt2 = init2.encrypt(msg, sizeof(msg) - 1);

    // Tamper with payload
    {
        auto bad = pkt2;
        bad[10] ^= 0xFF;
        auto dec = decrypt_or_empty(resp2, bad);
        CHECK(dec.empty(), "Tampered payload → auth failure");
    }

    // Session 3: tamper with tag (last 16 bytes)
    Bip324Session init3(true, KEY_A);
    Bip324Session resp3(false, KEY_B);
    resp3.complete_handshake(init3.our_ellswift_encoding().data());
    init3.complete_handshake(resp3.our_ellswift_encoding().data());

    auto pkt3 = init3.encrypt(msg, sizeof(msg) - 1);
    {
        auto bad = pkt3;
        bad[bad.size() - 1] ^= 0x01;
        auto dec = decrypt_or_empty(resp3, bad);
        CHECK(dec.empty(), "Tampered tag → auth failure");
    }
}

// ============================================================================
// 10. Random-key sessions (CSPRNG constructor)
// ============================================================================

static void test_bip324_random_keys() {
    std::printf("\n--- BIP-324 Random Keys ---\n");

    // Two sessions with random keys should establish and communicate
    Bip324Session a(true);   // random key
    Bip324Session b(false);  // random key

    bool hs_b = b.complete_handshake(a.our_ellswift_encoding().data());
    bool hs_a = a.complete_handshake(b.our_ellswift_encoding().data());
    CHECK(hs_a && hs_b, "Random-key session: handshake succeeds");

    const std::uint8_t msg[] = "random key message";
    auto pkt = a.encrypt(msg, sizeof(msg) - 1);
    auto dec = decrypt_or_empty(b, pkt);
    CHECK(!dec.empty() && dec.size() == sizeof(msg) - 1,
          "Random-key session: encrypt/decrypt roundtrip");
    CHECK(std::memcmp(dec.data(), msg, sizeof(msg) - 1) == 0,
          "Random-key session: data integrity");

    // Two different random sessions produce different session IDs
    Bip324Session c(true);
    Bip324Session d(false);
    d.complete_handshake(c.our_ellswift_encoding().data());
    c.complete_handshake(d.our_ellswift_encoding().data());
    CHECK(a.session_id() != c.session_id(),
          "Different random sessions → different session_id");
}

// ============================================================================
// Entry point
// ============================================================================

int test_bip324_run() {
    std::printf("=== BIP-324 Transport Tests ===\n");

    test_hkdf();
    test_chacha20_poly1305();
    test_ellswift();
    test_bip324_cipher();
    test_bip324_session();
    test_bip324_sequence();
    test_bip324_determinism();
    test_bip324_sizes();
    test_bip324_tamper();
    test_bip324_random_keys();

    std::printf("\n=== BIP-324: %d/%d passed ===\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}

#ifdef STANDALONE_TEST
int main() { return test_bip324_run(); }
#endif
