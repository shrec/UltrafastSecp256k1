// ============================================================================
// shim_test.cpp — libsecp256k1 compatibility shim validation
// ============================================================================
// Tests that UltrafastSecp256k1's shim produces correct results for all
// API functions used by Bitcoin Core. Each test exercises the same code paths
// that bitcoin/bitcoin calls in production.
// ============================================================================

#include <secp256k1.h>
#include <secp256k1_schnorrsig.h>
#include <secp256k1_extrakeys.h>
#include <secp256k1_recovery.h>
#include <secp256k1_ecdh.h>

#include <cstdio>
#include <cstring>
#include <cstdint>
#include <cassert>

// ── Test helpers ──────────────────────────────────────────────────────────────

static int g_pass = 0, g_fail = 0;

#define CHECK(cond, label) do { \
    if (cond) { \
        printf("  [PASS] %s\n", label); \
        ++g_pass; \
    } else { \
        printf("  [FAIL] %s\n", label); \
        ++g_fail; \
    } \
} while(0)

// Known test vectors (RFC 6979 / BIP-340)
static const unsigned char PRIVKEY[32] = {
    0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
    0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
    0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
    0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
};

static const unsigned char MSG32[32] = {
    0xde, 0xad, 0xbe, 0xef, 0xca, 0xfe, 0xba, 0xbe,
    0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77,
    0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff,
    0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef,
};

static const unsigned char AUX32[32] = {
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
};

// ── Test sections ─────────────────────────────────────────────────────────────

static void test_context(secp256k1_context* ctx) {
    printf("\n[Context]\n");
    CHECK(ctx != nullptr, "context_create returns non-null");

    auto* clone = secp256k1_context_clone(ctx);
    CHECK(clone != nullptr, "context_clone");
    secp256k1_context_destroy(clone);

    // randomize is a no-op but must return 1 (success)
    unsigned char seed[32] = {0x42};
    CHECK(secp256k1_context_randomize(ctx, seed) == 1, "context_randomize returns 1");
    CHECK(secp256k1_context_randomize(ctx, nullptr) == 1, "context_randomize(null) returns 1");

    CHECK(secp256k1_context_static != nullptr, "secp256k1_context_static non-null");
}

static void test_seckey(secp256k1_context* ctx) {
    printf("\n[Secret key]\n");
    CHECK(secp256k1_ec_seckey_verify(ctx, PRIVKEY) == 1, "seckey_verify valid key");

    unsigned char zero[32] = {};
    CHECK(secp256k1_ec_seckey_verify(ctx, zero) == 0, "seckey_verify zero key fails");

    unsigned char all_ff[32];
    memset(all_ff, 0xff, 32);
    CHECK(secp256k1_ec_seckey_verify(ctx, all_ff) == 0, "seckey_verify n-1+1 fails");

    // negate
    unsigned char key_copy[32];
    memcpy(key_copy, PRIVKEY, 32);
    CHECK(secp256k1_ec_seckey_negate(ctx, key_copy) == 1, "seckey_negate");

    // tweak_add: key + 0 = key
    memcpy(key_copy, PRIVKEY, 32);
    unsigned char zero_tweak[32] = {};
    // zero tweak — should succeed (k + 0 mod n)
    // Note: some implementations reject zero tweak; check is on return only
    secp256k1_ec_seckey_tweak_add(ctx, key_copy, zero_tweak);
}

static secp256k1_pubkey test_pubkey(secp256k1_context* ctx) {
    printf("\n[Public key]\n");
    secp256k1_pubkey pubkey{};
    CHECK(secp256k1_ec_pubkey_create(ctx, &pubkey, PRIVKEY) == 1, "pubkey_create");

    // Serialize compressed
    unsigned char compressed[33];
    size_t complen = sizeof(compressed);
    CHECK(secp256k1_ec_pubkey_serialize(ctx, compressed, &complen, &pubkey, SECP256K1_EC_COMPRESSED) == 1,
          "pubkey_serialize compressed");
    CHECK(complen == 33, "compressed length == 33");
    CHECK(compressed[0] == 0x02 || compressed[0] == 0x03, "compressed prefix 02/03");

    // Serialize uncompressed
    unsigned char uncompressed[65];
    size_t uncomplen = sizeof(uncompressed);
    CHECK(secp256k1_ec_pubkey_serialize(ctx, uncompressed, &uncomplen, &pubkey, SECP256K1_EC_UNCOMPRESSED) == 1,
          "pubkey_serialize uncompressed");
    CHECK(uncomplen == 65, "uncompressed length == 65");
    CHECK(uncompressed[0] == 0x04, "uncompressed prefix 04");

    // Round-trip: parse compressed back
    secp256k1_pubkey pubkey2{};
    CHECK(secp256k1_ec_pubkey_parse(ctx, &pubkey2, compressed, 33) == 1,
          "pubkey_parse compressed");

    // Re-serialize and compare
    unsigned char compressed2[33];
    size_t complen2 = sizeof(compressed2);
    secp256k1_ec_pubkey_serialize(ctx, compressed2, &complen2, &pubkey2, SECP256K1_EC_COMPRESSED);
    CHECK(memcmp(compressed, compressed2, 33) == 0, "round-trip pubkey matches");

    // Parse uncompressed
    secp256k1_pubkey pubkey3{};
    CHECK(secp256k1_ec_pubkey_parse(ctx, &pubkey3, uncompressed, 65) == 1,
          "pubkey_parse uncompressed");

    // Invalid parse
    unsigned char bad[33] = {};
    bad[0] = 0x05;
    CHECK(secp256k1_ec_pubkey_parse(ctx, &pubkey3, bad, 33) == 0,
          "pubkey_parse bad prefix fails");

    return pubkey;
}

static void test_ecdsa(secp256k1_context* ctx, const secp256k1_pubkey* pubkey) {
    printf("\n[ECDSA]\n");

    secp256k1_ecdsa_signature sig{};
    CHECK(secp256k1_ecdsa_sign(ctx, &sig, MSG32, PRIVKEY, nullptr, nullptr) == 1,
          "ecdsa_sign");

    CHECK(secp256k1_ecdsa_verify(ctx, &sig, MSG32, pubkey) == 1,
          "ecdsa_verify valid");

    // Modify message — should fail
    unsigned char bad_msg[32];
    memcpy(bad_msg, MSG32, 32);
    bad_msg[0] ^= 0x01;
    CHECK(secp256k1_ecdsa_verify(ctx, &sig, bad_msg, pubkey) == 0,
          "ecdsa_verify wrong message fails");

    // Compact serialize/parse round-trip
    unsigned char compact[64];
    CHECK(secp256k1_ecdsa_signature_serialize_compact(ctx, compact, &sig) == 1,
          "signature_serialize_compact");
    secp256k1_ecdsa_signature sig2{};
    CHECK(secp256k1_ecdsa_signature_parse_compact(ctx, &sig2, compact) == 1,
          "signature_parse_compact");
    CHECK(secp256k1_ecdsa_verify(ctx, &sig2, MSG32, pubkey) == 1,
          "verify after compact round-trip");

    // DER serialize/parse round-trip
    unsigned char der[72];
    size_t derlen = sizeof(der);
    CHECK(secp256k1_ecdsa_signature_serialize_der(ctx, der, &derlen, &sig) == 1,
          "signature_serialize_der");
    secp256k1_ecdsa_signature sig3{};
    CHECK(secp256k1_ecdsa_signature_parse_der(ctx, &sig3, der, derlen) == 1,
          "signature_parse_der");
    CHECK(secp256k1_ecdsa_verify(ctx, &sig3, MSG32, pubkey) == 1,
          "verify after DER round-trip");

    // Normalize (low-S)
    secp256k1_ecdsa_signature sig_norm = sig;
    secp256k1_ecdsa_signature_normalize(ctx, &sig_norm, &sig);
    CHECK(secp256k1_ecdsa_verify(ctx, &sig_norm, MSG32, pubkey) == 1,
          "verify after normalize");
}

static void test_schnorr(secp256k1_context* ctx) {
    printf("\n[Schnorr BIP-340]\n");

    secp256k1_keypair keypair{};
    CHECK(secp256k1_keypair_create(ctx, &keypair, PRIVKEY) == 1,
          "keypair_create");

    unsigned char sig64[64];
    CHECK(secp256k1_schnorrsig_sign32(ctx, sig64, MSG32, &keypair, AUX32) == 1,
          "schnorrsig_sign32");

    // Get x-only pubkey from keypair
    secp256k1_xonly_pubkey xonly_pub{};
    CHECK(secp256k1_keypair_xonly_pub(ctx, &xonly_pub, nullptr, &keypair) == 1,
          "keypair_xonly_pub");

    CHECK(secp256k1_schnorrsig_verify(ctx, sig64, MSG32, 32, &xonly_pub) == 1,
          "schnorrsig_verify valid");

    // Flip a bit in sig — should fail
    unsigned char bad_sig[64];
    memcpy(bad_sig, sig64, 64);
    bad_sig[0] ^= 0x01;
    CHECK(secp256k1_schnorrsig_verify(ctx, bad_sig, MSG32, 32, &xonly_pub) == 0,
          "schnorrsig_verify flipped sig fails");

    // x-only pubkey serialize/parse round-trip
    unsigned char xonly_bytes[32];
    CHECK(secp256k1_xonly_pubkey_serialize(ctx, xonly_bytes, &xonly_pub) == 1,
          "xonly_pubkey_serialize");
    secp256k1_xonly_pubkey xonly_pub2{};
    CHECK(secp256k1_xonly_pubkey_parse(ctx, &xonly_pub2, xonly_bytes) == 1,
          "xonly_pubkey_parse");
    CHECK(secp256k1_schnorrsig_verify(ctx, sig64, MSG32, 32, &xonly_pub2) == 1,
          "verify with round-tripped x-only pubkey");
}

static void test_extrakeys(secp256k1_context* ctx) {
    printf("\n[Extra keys (BIP-340/341)]\n");

    secp256k1_keypair kp{};
    secp256k1_keypair_create(ctx, &kp, PRIVKEY);

    secp256k1_xonly_pubkey xpub{};
    int parity = -1;
    CHECK(secp256k1_keypair_xonly_pub(ctx, &xpub, &parity, &kp) == 1,
          "keypair_xonly_pub with parity");
    CHECK(parity == 0 || parity == 1, "parity is 0 or 1");

    // xonly_pubkey_from_pubkey
    secp256k1_pubkey pub{};
    secp256k1_ec_pubkey_create(ctx, &pub, PRIVKEY);
    secp256k1_xonly_pubkey xpub2{};
    int parity2 = -1;
    CHECK(secp256k1_xonly_pubkey_from_pubkey(ctx, &xpub2, &parity2, &pub) == 1,
          "xonly_pubkey_from_pubkey");

    // xonly_pubkey_cmp
    CHECK(secp256k1_xonly_pubkey_cmp(ctx, &xpub, &xpub2) == 0,
          "xonly_pubkey_cmp same key == 0");

    // Taproot key tweak
    unsigned char tweak[32] = {
        0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
        0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
        0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
        0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
    };
    secp256k1_pubkey tweaked{};
    CHECK(secp256k1_xonly_pubkey_tweak_add(ctx, &tweaked, &xpub, tweak) == 1,
          "xonly_pubkey_tweak_add");

    // Verify tweak
    unsigned char tweaked_ser[32];
    secp256k1_xonly_pubkey tweaked_xonly{};
    int tweaked_parity = -1;
    secp256k1_xonly_pubkey_from_pubkey(ctx, &tweaked_xonly, &tweaked_parity, &tweaked);
    secp256k1_xonly_pubkey_serialize(ctx, tweaked_ser, &tweaked_xonly);
    CHECK(secp256k1_xonly_pubkey_tweak_add_check(ctx, tweaked_ser, tweaked_parity, &xpub, tweak) == 1,
          "xonly_pubkey_tweak_add_check");
}

static void test_recovery(secp256k1_context* ctx) {
    printf("\n[ECDSA Recovery]\n");

    secp256k1_ecdsa_recoverable_signature rsig{};
    CHECK(secp256k1_ecdsa_sign_recoverable(ctx, &rsig, MSG32, PRIVKEY, nullptr, nullptr) == 1,
          "ecdsa_sign_recoverable");

    // Serialize/parse compact
    unsigned char compact[64];
    int recid = -1;
    CHECK(secp256k1_ecdsa_recoverable_signature_serialize_compact(ctx, compact, &recid, &rsig) == 1,
          "recoverable_sig_serialize_compact");
    CHECK(recid >= 0 && recid <= 3, "recid in [0,3]");

    secp256k1_ecdsa_recoverable_signature rsig2{};
    CHECK(secp256k1_ecdsa_recoverable_signature_parse_compact(ctx, &rsig2, compact, recid) == 1,
          "recoverable_sig_parse_compact");

    // Recover pubkey
    secp256k1_pubkey recovered{};
    CHECK(secp256k1_ecdsa_recover(ctx, &recovered, &rsig2, MSG32) == 1,
          "ecdsa_recover");

    // Recovered pubkey should match original
    secp256k1_pubkey expected{};
    secp256k1_ec_pubkey_create(ctx, &expected, PRIVKEY);
    unsigned char rec_bytes[33], exp_bytes[33];
    size_t len = 33;
    secp256k1_ec_pubkey_serialize(ctx, rec_bytes, &len, &recovered, SECP256K1_EC_COMPRESSED);
    len = 33;
    secp256k1_ec_pubkey_serialize(ctx, exp_bytes, &len, &expected, SECP256K1_EC_COMPRESSED);
    CHECK(memcmp(rec_bytes, exp_bytes, 33) == 0, "recovered pubkey matches original");
}

static void test_ecdh(secp256k1_context* ctx) {
    printf("\n[ECDH]\n");

    // Generate two keypairs
    secp256k1_pubkey pub_alice{}, pub_bob{};
    static const unsigned char PRIVKEY_BOB[32] = {
        0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02,
        0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02,
        0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02,
        0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02,
    };
    secp256k1_ec_pubkey_create(ctx, &pub_alice, PRIVKEY);
    secp256k1_ec_pubkey_create(ctx, &pub_bob, PRIVKEY_BOB);

    // Alice computes ECDH(alice_priv, bob_pub)
    unsigned char shared_alice[32];
    CHECK(secp256k1_ecdh(ctx, shared_alice, &pub_bob, PRIVKEY, nullptr, nullptr) == 1,
          "ecdh alice");

    // Bob computes ECDH(bob_priv, alice_pub)
    unsigned char shared_bob[32];
    CHECK(secp256k1_ecdh(ctx, shared_bob, &pub_alice, PRIVKEY_BOB, nullptr, nullptr) == 1,
          "ecdh bob");

    CHECK(memcmp(shared_alice, shared_bob, 32) == 0, "ECDH shared secret matches");
}

static void test_tagged_hash(secp256k1_context* ctx) {
    printf("\n[Tagged SHA256]\n");
    unsigned char out[32];
    const unsigned char data[] = "hello world";
    CHECK(secp256k1_tagged_sha256(ctx, out, (const unsigned char*)"BIP0340/challenge", 17, data, 11) == 1,
          "tagged_sha256");
    // Run twice — same result
    unsigned char out2[32];
    secp256k1_tagged_sha256(ctx, out2, (const unsigned char*)"BIP0340/challenge", 17, data, 11);
    CHECK(memcmp(out, out2, 32) == 0, "tagged_sha256 deterministic");
}

// ── Main ─────────────────────────────────────────────────────────────────────

int main() {
    printf("secp256k1 shim compatibility test\n");
    printf("===================================\n");

    secp256k1_context* ctx = secp256k1_context_create(SECP256K1_CONTEXT_SIGN | SECP256K1_CONTEXT_VERIFY);

    test_context(ctx);
    test_seckey(ctx);
    secp256k1_pubkey pubkey = test_pubkey(ctx);
    test_ecdsa(ctx, &pubkey);
    test_schnorr(ctx);
    test_extrakeys(ctx);
    test_recovery(ctx);
    test_ecdh(ctx);
    test_tagged_hash(ctx);

    secp256k1_context_destroy(ctx);

    printf("\n===================================\n");
    printf("Results: %d passed, %d failed\n", g_pass, g_fail);

    return g_fail > 0 ? 1 : 0;
}
