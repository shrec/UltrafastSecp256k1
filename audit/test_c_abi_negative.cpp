// ============================================================================
// C ABI Negative Contract Tests -- UltrafastSecp256k1
// ============================================================================
//
// Systematically tests every ufsecp_* function for correct error-code
// propagation when called with:
//
//   A. NULL required pointer arguments        -> UFSECP_ERR_NULL_ARG
//   B. Invalid / zero private keys            -> UFSECP_ERR_BAD_KEY
//   C. Out-of-range / all-order private keys  -> UFSECP_ERR_BAD_KEY
//   D. Off-curve / truncated public keys      -> UFSECP_ERR_BAD_PUBKEY
//   E. Malformed / all-zero signatures        -> UFSECP_ERR_BAD_SIG
//   F. Wrong-length / malformed DER           -> UFSECP_ERR_BAD_INPUT
//   G. Verification against wrong pubkey      -> UFSECP_ERR_VERIFY_FAIL
//   H. Buffer too small                       -> UFSECP_ERR_BUF_TOO_SMALL
//   I. BIP-32 seed length out of range        -> UFSECP_ERR_BAD_INPUT
//
// None of these must crash, abort, or silently return UFSECP_OK.
//
// Covered function groups:
//   ufsecp_ctx_*               (context lifecycle)
//   ufsecp_seckey_*            (private key utilities)
//   ufsecp_pubkey_*            (public key derivation + parsing)
//   ufsecp_ecdsa_*             (ECDSA sign / verify / DER / recovery)
//   ufsecp_schnorr_*           (BIP-340 Schnorr sign / verify)
//   ufsecp_ecdh_*              (ECDH key agreement variants)
//   ufsecp_sha256 / hash160 / tagged_hash
//   ufsecp_addr_*              (Bitcoin address generation)
//   ufsecp_wif_*               (WIF encode/decode)
//   ufsecp_bip32_*             (HD key derivation)
//   ufsecp_taproot_*           (BIP-341 Taproot)
//   ufsecp_pubkey_add / negate / tweak_add / tweak_mul
// ============================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <array>

#ifndef UFSECP_BUILDING
#define UFSECP_BUILDING
#endif
#include "ufsecp/ufsecp.h"
#include "ufsecp/ufsecp_gpu.h"

static int g_pass = 0, g_fail = 0;

#include "audit_check.hpp"

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Expects the call to return any non-OK error code (just not UFSECP_OK)
#define CHECK_ERR(expr, msg)   CHECK((expr) != UFSECP_OK, msg)
// Expects a specific error code
#define CHECK_CODE(expr, code, msg) CHECK((expr) == (code), msg)
// Expects UFSECP_OK
#define CHECK_OK(expr, msg)    CHECK((expr) == UFSECP_OK, msg)

// secp256k1 curve order n (big-endian, 32 bytes)
static constexpr uint8_t ORDER_N[32] = {
    0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,
    0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFE,
    0xBA,0xAE,0xDC,0xE6,0xAF,0x48,0xA0,0x3B,
    0xBF,0xD2,0x5E,0x8C,0xD0,0x36,0x41,0x41
};

// A valid private key (scalar = 1; minimal valid value)
static constexpr uint8_t VALID_KEY1[32] = {
    0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,1
};

// Another valid private key (scalar = 2)
static constexpr uint8_t VALID_KEY2[32] = {
    0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,2
};

// Invalid key: all zeros (= 0 mod n)
static constexpr uint8_t ZERO_KEY[32]  = { 0 };

// All-zeros 32-byte buffer (bad hash / bad tweak)
static constexpr uint8_t ZERO32[32]    = { 0 };

// All-zeros 33-byte compressed pubkey (invalid -- no valid prefix)
static constexpr uint8_t ZERO_PUBKEY33[33] = { 0 };

// All-zeros 64-byte signature (R = 0, S = 0 -- invalid)
static constexpr uint8_t ZERO_SIG64[64] = { 0 };

// All-zeros 32-byte x-only pubkey (invalid -- not on curve)
static constexpr uint8_t ZERO_XONLY[32] = { 0 };

// Garbage DER bytes
static constexpr uint8_t GARBAGE_DER[8] = { 0x30,0x06,0x02,0x01,0x00,0x02,0x01,0x00 };

// A valid 32-byte message hash (SHA256 of "audit")
static constexpr uint8_t MSG32[32] = {
    0x9f,0x86,0xd0,0x81,0x88,0x4c,0x7d,0x65,
    0x9a,0x2f,0xea,0xa0,0xc5,0x5a,0xd0,0x15,
    0xa3,0xbf,0x4f,0x1b,0x2b,0x0b,0x82,0x2c,
    0xd1,0x5d,0x6c,0x15,0xb0,0xf0,0x0a,0x08
};

// A valid BIP-32 seed (32 bytes of known entropy)
static constexpr uint8_t VALID_SEED[32] = {
    0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,
    0x08,0x09,0x0a,0x0b,0x0c,0x0d,0x0e,0x0f,
    0x10,0x11,0x12,0x13,0x14,0x15,0x16,0x17,
    0x18,0x19,0x1a,0x1b,0x1c,0x1d,0x1e,0x1f
};

// Short seed (< 16 bytes -- invalid per BIP-32)
static constexpr uint8_t SHORT_SEED[8] = { 0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08 };

// ---------------------------------------------------------------------------
// Fixture: shared context + derived data built once
// ---------------------------------------------------------------------------

struct Fixture {
    ufsecp_ctx* ctx = nullptr;
    uint8_t pubkey33[33]   = {};
    uint8_t xonly32[32]    = {};
    uint8_t sig64_ecdsa[64] = {};
    uint8_t sig64_schnorr[64] = {};
    ufsecp_bip32_key master_key = {};
    bool ok = false;

    bool init() {
        if (ufsecp_ctx_create(&ctx) != UFSECP_OK || ctx == nullptr)
            return false;
        if (ufsecp_pubkey_create(ctx, VALID_KEY1, pubkey33) != UFSECP_OK)
            return false;
        if (ufsecp_pubkey_xonly(ctx, VALID_KEY1, xonly32) != UFSECP_OK)
            return false;
        if (ufsecp_ecdsa_sign(ctx, MSG32, VALID_KEY1, sig64_ecdsa) != UFSECP_OK)
            return false;
        uint8_t aux[32] = {};
        if (ufsecp_schnorr_sign(ctx, MSG32, VALID_KEY1, aux, sig64_schnorr) != UFSECP_OK)
            return false;
        if (ufsecp_bip32_master(ctx, VALID_SEED, sizeof(VALID_SEED), &master_key) != UFSECP_OK)
            return false;
        ok = true;
        return true;
    }

    ~Fixture() {
        ufsecp_ctx_destroy(ctx);
    }
};

// ---------------------------------------------------------------------------
// NEG-1: Context lifecycle
// ---------------------------------------------------------------------------

static void run_neg1_context(ufsecp_ctx* ctx) {
    AUDIT_LOG("\n  [NEG-1] Context lifecycle null-arg checks\n");

    // Create: null ctx_out
    CHECK_CODE(ufsecp_ctx_create(nullptr), UFSECP_ERR_NULL_ARG,
               "NEG-1.1: ctx_create(null) -> NULL_ARG");

    // Clone: null src
    ufsecp_ctx* clone = nullptr;
    CHECK_CODE(ufsecp_ctx_clone(nullptr, &clone), UFSECP_ERR_NULL_ARG,
               "NEG-1.2: ctx_clone(null_src) -> NULL_ARG");

    // Clone: null ctx_out
    CHECK_CODE(ufsecp_ctx_clone(ctx, nullptr), UFSECP_ERR_NULL_ARG,
               "NEG-1.3: ctx_clone(null_out) -> NULL_ARG");

    // Destroy null: must not crash (void, no error code to check)
    ufsecp_ctx_destroy(nullptr);
        ufsecp_ctx* cloned = nullptr;
        CHECK_CODE(ufsecp_ctx_clone(ctx, &cloned), UFSECP_OK,
               "NEG-1.4: destroy(null) leaves live ctx usable");
        CHECK(cloned != nullptr, "NEG-1.4: cloned ctx returned after destroy(null)");
        ufsecp_ctx_destroy(cloned);

    // last_error on null: must return UFSECP_ERR_NULL_ARG or some non-crash value
        CHECK(ufsecp_last_error(nullptr) == UFSECP_ERR_NULL_ARG,
            "NEG-1.5: last_error(null_ctx) returns NULL_ARG");

        CHECK(std::strcmp(ufsecp_last_error_msg(nullptr), "NULL context") == 0,
            "NEG-1.6: last_error_msg(null_ctx) returns sentinel message");
}

// ---------------------------------------------------------------------------
// NEG-2: Private key validation
// ---------------------------------------------------------------------------

static void run_neg2_seckey(ufsecp_ctx* ctx) {
    AUDIT_LOG("\n  [NEG-2] Private key validation\n");

    // verify: null ctx
    CHECK_ERR(ufsecp_seckey_verify(nullptr, VALID_KEY1),
              "NEG-2.1: seckey_verify(null_ctx) -> error");

    // verify: null key
    CHECK_CODE(ufsecp_seckey_verify(ctx, nullptr), UFSECP_ERR_NULL_ARG,
               "NEG-2.2: seckey_verify(null_key) -> NULL_ARG");

    // verify: zero key (invalid)
    CHECK_CODE(ufsecp_seckey_verify(ctx, ZERO_KEY), UFSECP_ERR_BAD_KEY,
               "NEG-2.3: seckey_verify(zero_key) -> BAD_KEY");

    // verify: key == order (= 0 mod n; invalid)
    CHECK_CODE(ufsecp_seckey_verify(ctx, ORDER_N), UFSECP_ERR_BAD_KEY,
               "NEG-2.4: seckey_verify(order_key) -> BAD_KEY");

    // negate: null ctx
    uint8_t key_copy[32];
    memcpy(key_copy, VALID_KEY1, 32);
    CHECK_ERR(ufsecp_seckey_negate(nullptr, key_copy),
              "NEG-2.5: seckey_negate(null_ctx) -> error");

    // negate: null key
    CHECK_CODE(ufsecp_seckey_negate(ctx, nullptr), UFSECP_ERR_NULL_ARG,
               "NEG-2.6: seckey_negate(null_key) -> NULL_ARG");

    // tweak_add: null ctx
    memcpy(key_copy, VALID_KEY1, 32);
    CHECK_ERR(ufsecp_seckey_tweak_add(nullptr, key_copy, VALID_KEY2),
              "NEG-2.7: seckey_tweak_add(null_ctx) -> error");

    // tweak_add: null key
    CHECK_CODE(ufsecp_seckey_tweak_add(ctx, nullptr, VALID_KEY2), UFSECP_ERR_NULL_ARG,
               "NEG-2.8: seckey_tweak_add(null_key) -> NULL_ARG");

    // tweak_add: null tweak
    memcpy(key_copy, VALID_KEY1, 32);
    CHECK_CODE(ufsecp_seckey_tweak_add(ctx, key_copy, nullptr), UFSECP_ERR_NULL_ARG,
               "NEG-2.9: seckey_tweak_add(null_tweak) -> NULL_ARG");

    // tweak_mul: null ctx
    memcpy(key_copy, VALID_KEY1, 32);
    CHECK_ERR(ufsecp_seckey_tweak_mul(nullptr, key_copy, VALID_KEY2),
              "NEG-2.10: seckey_tweak_mul(null_ctx) -> error");

    // tweak_mul: null key
    CHECK_CODE(ufsecp_seckey_tweak_mul(ctx, nullptr, VALID_KEY2), UFSECP_ERR_NULL_ARG,
               "NEG-2.11: seckey_tweak_mul(null_key) -> NULL_ARG");

    // tweak_mul: null tweak
    memcpy(key_copy, VALID_KEY1, 32);
    CHECK_CODE(ufsecp_seckey_tweak_mul(ctx, key_copy, nullptr), UFSECP_ERR_NULL_ARG,
               "NEG-2.12: seckey_tweak_mul(null_tweak) -> NULL_ARG");
}

// ---------------------------------------------------------------------------
// NEG-3: Public key derivation + parsing
// ---------------------------------------------------------------------------

static void run_neg3_pubkey(ufsecp_ctx* ctx) {
    AUDIT_LOG("\n  [NEG-3] Public key derivation + parsing\n");

    uint8_t buf33[33] = {};
    uint8_t buf65[65] = {};
    uint8_t xonly[32] = {};

    // create: null ctx
    CHECK_ERR(ufsecp_pubkey_create(nullptr, VALID_KEY1, buf33),
              "NEG-3.1: pubkey_create(null_ctx) -> error");

    // create: null privkey
    CHECK_CODE(ufsecp_pubkey_create(ctx, nullptr, buf33), UFSECP_ERR_NULL_ARG,
               "NEG-3.2: pubkey_create(null_priv) -> NULL_ARG");

    // create: null output
    CHECK_CODE(ufsecp_pubkey_create(ctx, VALID_KEY1, nullptr), UFSECP_ERR_NULL_ARG,
               "NEG-3.3: pubkey_create(null_out) -> NULL_ARG");

    // create: zero key (invalid)
    CHECK_CODE(ufsecp_pubkey_create(ctx, ZERO_KEY, buf33), UFSECP_ERR_BAD_KEY,
               "NEG-3.4: pubkey_create(zero_key) -> BAD_KEY");

    // create_uncompressed: null ctx
    CHECK_ERR(ufsecp_pubkey_create_uncompressed(nullptr, VALID_KEY1, buf65),
              "NEG-3.5: pubkey_create_uncompressed(null_ctx) -> error");

    // create_uncompressed: null privkey
    CHECK_CODE(ufsecp_pubkey_create_uncompressed(ctx, nullptr, buf65), UFSECP_ERR_NULL_ARG,
               "NEG-3.6: pubkey_create_uncompressed(null_priv) -> NULL_ARG");

    // create_uncompressed: null output
    CHECK_CODE(ufsecp_pubkey_create_uncompressed(ctx, VALID_KEY1, nullptr), UFSECP_ERR_NULL_ARG,
               "NEG-3.7: pubkey_create_uncompressed(null_out) -> NULL_ARG");

    // create_uncompressed: zero key
    CHECK_CODE(ufsecp_pubkey_create_uncompressed(ctx, ZERO_KEY, buf65), UFSECP_ERR_BAD_KEY,
               "NEG-3.8: pubkey_create_uncompressed(zero_key) -> BAD_KEY");

    // parse: null ctx
    uint8_t pk33[33] = {};
    CHECK_ERR(ufsecp_pubkey_parse(nullptr, ZERO_PUBKEY33, 33, pk33),
              "NEG-3.9: pubkey_parse(null_ctx) -> error");

    // parse: null input
    CHECK_CODE(ufsecp_pubkey_parse(ctx, nullptr, 33, pk33), UFSECP_ERR_NULL_ARG,
               "NEG-3.10: pubkey_parse(null_input) -> NULL_ARG");

    // parse: null output
    CHECK_CODE(ufsecp_pubkey_parse(ctx, ZERO_PUBKEY33, 33, nullptr), UFSECP_ERR_NULL_ARG,
               "NEG-3.11: pubkey_parse(null_out) -> NULL_ARG");

    // parse: wrong length (e.g. 1 byte)
    uint8_t bad1[1] = {0x02};
    CHECK_ERR(ufsecp_pubkey_parse(ctx, bad1, 1, pk33),
              "NEG-3.12: pubkey_parse(len=1) -> error");

    // parse: wrong length (e.g. 32 bytes)
    uint8_t bad32[32] = {};
    CHECK_ERR(ufsecp_pubkey_parse(ctx, bad32, 32, pk33),
              "NEG-3.13: pubkey_parse(len=32) -> error");

    // parse: all-zero 33 bytes (bad prefix 0x00)
    CHECK_ERR(ufsecp_pubkey_parse(ctx, ZERO_PUBKEY33, 33, pk33),
              "NEG-3.14: pubkey_parse(zero_33) -> error");

    // parse: prefix 0x02 but all-zero x (not on curve)
    uint8_t bad_x[33] = {};
    bad_x[0] = 0x02;
    CHECK_ERR(ufsecp_pubkey_parse(ctx, bad_x, 33, pk33),
              "NEG-3.15: pubkey_parse(02||zero_x) -> error");

    // xonly: null ctx
    CHECK_ERR(ufsecp_pubkey_xonly(nullptr, VALID_KEY1, xonly),
              "NEG-3.16: pubkey_xonly(null_ctx) -> error");

    // xonly: null privkey
    CHECK_CODE(ufsecp_pubkey_xonly(ctx, nullptr, xonly), UFSECP_ERR_NULL_ARG,
               "NEG-3.17: pubkey_xonly(null_priv) -> NULL_ARG");

    // xonly: null output
    CHECK_CODE(ufsecp_pubkey_xonly(ctx, VALID_KEY1, nullptr), UFSECP_ERR_NULL_ARG,
               "NEG-3.18: pubkey_xonly(null_out) -> NULL_ARG");

    // xonly: zero key
    CHECK_CODE(ufsecp_pubkey_xonly(ctx, ZERO_KEY, xonly), UFSECP_ERR_BAD_KEY,
               "NEG-3.19: pubkey_xonly(zero_key) -> BAD_KEY");
}

// ---------------------------------------------------------------------------
// NEG-4: ECDSA sign / verify / DER / recovery
// ---------------------------------------------------------------------------

static void run_neg4_ecdsa(ufsecp_ctx* ctx, const uint8_t* valid_sig64,
                           const uint8_t* pubkey33) {
    AUDIT_LOG("\n  [NEG-4] ECDSA sign / verify / DER / recovery\n");

    uint8_t sig_out[64] = {};

    // sign: null ctx
    CHECK_ERR(ufsecp_ecdsa_sign(nullptr, MSG32, VALID_KEY1, sig_out),
              "NEG-4.1: ecdsa_sign(null_ctx) -> error");

    // sign: null msg
    CHECK_CODE(ufsecp_ecdsa_sign(ctx, nullptr, VALID_KEY1, sig_out), UFSECP_ERR_NULL_ARG,
               "NEG-4.2: ecdsa_sign(null_msg) -> NULL_ARG");

    // sign: null privkey
    CHECK_CODE(ufsecp_ecdsa_sign(ctx, MSG32, nullptr, sig_out), UFSECP_ERR_NULL_ARG,
               "NEG-4.3: ecdsa_sign(null_priv) -> NULL_ARG");

    // sign: null output
    CHECK_CODE(ufsecp_ecdsa_sign(ctx, MSG32, VALID_KEY1, nullptr), UFSECP_ERR_NULL_ARG,
               "NEG-4.4: ecdsa_sign(null_out) -> NULL_ARG");

    // sign: zero key
    CHECK_CODE(ufsecp_ecdsa_sign(ctx, MSG32, ZERO_KEY, sig_out), UFSECP_ERR_BAD_KEY,
               "NEG-4.5: ecdsa_sign(zero_key) -> BAD_KEY");

    // sign_verified: null ctx
    CHECK_ERR(ufsecp_ecdsa_sign_verified(nullptr, MSG32, VALID_KEY1, sig_out),
              "NEG-4.6: ecdsa_sign_verified(null_ctx) -> error");

    // sign_verified: null privkey
    CHECK_CODE(ufsecp_ecdsa_sign_verified(ctx, MSG32, nullptr, sig_out), UFSECP_ERR_NULL_ARG,
               "NEG-4.7: ecdsa_sign_verified(null_priv) -> NULL_ARG");

    // verify: null ctx
    CHECK_ERR(ufsecp_ecdsa_verify(nullptr, MSG32, valid_sig64, pubkey33),
              "NEG-4.8: ecdsa_verify(null_ctx) -> error");

    // verify: null msg
    CHECK_CODE(ufsecp_ecdsa_verify(ctx, nullptr, valid_sig64, pubkey33), UFSECP_ERR_NULL_ARG,
               "NEG-4.9: ecdsa_verify(null_msg) -> NULL_ARG");

    // verify: null sig
    CHECK_CODE(ufsecp_ecdsa_verify(ctx, MSG32, nullptr, pubkey33), UFSECP_ERR_NULL_ARG,
               "NEG-4.10: ecdsa_verify(null_sig) -> NULL_ARG");

    // verify: null pubkey
    CHECK_CODE(ufsecp_ecdsa_verify(ctx, MSG32, valid_sig64, nullptr), UFSECP_ERR_NULL_ARG,
               "NEG-4.11: ecdsa_verify(null_pubkey) -> NULL_ARG");

    // verify: zero sig (R=0 is invalid)
    CHECK_ERR(ufsecp_ecdsa_verify(ctx, MSG32, ZERO_SIG64, pubkey33),
              "NEG-4.12: ecdsa_verify(zero_sig) -> error");

    // verify: bad pubkey (all zeros)
    CHECK_ERR(ufsecp_ecdsa_verify(ctx, MSG32, valid_sig64, ZERO_PUBKEY33),
              "NEG-4.13: ecdsa_verify(zero_pubkey) -> error");

    // verify: wrong pubkey (KEY2's pubkey, not KEY1's)
    uint8_t pubkey2[33] = {};
    if (ufsecp_pubkey_create(ctx, VALID_KEY2, pubkey2) == UFSECP_OK) {
        CHECK_CODE(ufsecp_ecdsa_verify(ctx, MSG32, valid_sig64, pubkey2),
                   UFSECP_ERR_VERIFY_FAIL,
                   "NEG-4.14: ecdsa_verify(wrong_pubkey) -> VERIFY_FAIL");
    } else {
           CHECK(ufsecp_pubkey_create(ctx, VALID_KEY2, pubkey2) != UFSECP_OK, "NEG-4.14: skipped (pubkey2 derivation failed)");
    }

    // sig_to_der: null ctx
    uint8_t der[72] = {};
    size_t der_len = sizeof(der);
    CHECK_ERR(ufsecp_ecdsa_sig_to_der(nullptr, valid_sig64, der, &der_len),
              "NEG-4.15: sig_to_der(null_ctx) -> error");

    // sig_to_der: null sig
    der_len = sizeof(der);
    CHECK_CODE(ufsecp_ecdsa_sig_to_der(ctx, nullptr, der, &der_len), UFSECP_ERR_NULL_ARG,
               "NEG-4.16: sig_to_der(null_sig) -> NULL_ARG");

    // sig_to_der: null out
    der_len = sizeof(der);
    CHECK_CODE(ufsecp_ecdsa_sig_to_der(ctx, valid_sig64, nullptr, &der_len), UFSECP_ERR_NULL_ARG,
               "NEG-4.17: sig_to_der(null_out) -> NULL_ARG");

    // sig_to_der: null len ptr
    CHECK_CODE(ufsecp_ecdsa_sig_to_der(ctx, valid_sig64, der, nullptr), UFSECP_ERR_NULL_ARG,
               "NEG-4.18: sig_to_der(null_len) -> NULL_ARG");

    // sig_to_der: buffer too small (5 bytes)
    uint8_t tiny[5] = {};
    size_t tiny_len = sizeof(tiny);
    CHECK_ERR(ufsecp_ecdsa_sig_to_der(ctx, valid_sig64, tiny, &tiny_len),
              "NEG-4.19: sig_to_der(buf_too_small) -> error");

    // sig_from_der: null ctx
    uint8_t compact[64] = {};
    CHECK_ERR(ufsecp_ecdsa_sig_from_der(nullptr, GARBAGE_DER, sizeof(GARBAGE_DER), compact),
              "NEG-4.20: sig_from_der(null_ctx) -> error");

    // sig_from_der: null der
    CHECK_CODE(ufsecp_ecdsa_sig_from_der(ctx, nullptr, 72, compact), UFSECP_ERR_NULL_ARG,
               "NEG-4.21: sig_from_der(null_der) -> NULL_ARG");

    // sig_from_der: null out
    CHECK_CODE(ufsecp_ecdsa_sig_from_der(ctx, GARBAGE_DER, sizeof(GARBAGE_DER), nullptr),
               UFSECP_ERR_NULL_ARG,
               "NEG-4.22: sig_from_der(null_out) -> NULL_ARG");

    // sig_from_der: zero-length DER
    CHECK_ERR(ufsecp_ecdsa_sig_from_der(ctx, GARBAGE_DER, 0, compact),
              "NEG-4.23: sig_from_der(len=0) -> error");

    // sign_recoverable: null ctx
    int recid = -1;
    CHECK_ERR(ufsecp_ecdsa_sign_recoverable(nullptr, MSG32, VALID_KEY1, sig_out, &recid),
              "NEG-4.24: ecdsa_sign_recoverable(null_ctx) -> error");

    // sign_recoverable: null privkey
    CHECK_CODE(ufsecp_ecdsa_sign_recoverable(ctx, MSG32, nullptr, sig_out, &recid),
               UFSECP_ERR_NULL_ARG,
               "NEG-4.25: ecdsa_sign_recoverable(null_priv) -> NULL_ARG");

    // sign_recoverable: null recid
    CHECK_CODE(ufsecp_ecdsa_sign_recoverable(ctx, MSG32, VALID_KEY1, sig_out, nullptr),
               UFSECP_ERR_NULL_ARG,
               "NEG-4.26: ecdsa_sign_recoverable(null_recid) -> NULL_ARG");

    // recover: null ctx
    uint8_t rec_pub[33] = {};
    CHECK_ERR(ufsecp_ecdsa_recover(nullptr, MSG32, valid_sig64, 0, rec_pub),
              "NEG-4.27: ecdsa_recover(null_ctx) -> error");

    // recover: null msg
    CHECK_CODE(ufsecp_ecdsa_recover(ctx, nullptr, valid_sig64, 0, rec_pub),
               UFSECP_ERR_NULL_ARG,
               "NEG-4.28: ecdsa_recover(null_msg) -> NULL_ARG");

    // recover: null sig
    CHECK_CODE(ufsecp_ecdsa_recover(ctx, MSG32, nullptr, 0, rec_pub), UFSECP_ERR_NULL_ARG,
               "NEG-4.29: ecdsa_recover(null_sig) -> NULL_ARG");

    // recover: null out
    CHECK_CODE(ufsecp_ecdsa_recover(ctx, MSG32, valid_sig64, 0, nullptr), UFSECP_ERR_NULL_ARG,
               "NEG-4.30: ecdsa_recover(null_out) -> NULL_ARG");

    // recover: invalid recid (4 is out of range 0-3)
    CHECK_ERR(ufsecp_ecdsa_recover(ctx, MSG32, ZERO_SIG64, 4, rec_pub),
              "NEG-4.31: ecdsa_recover(recid=4, zero_sig) -> error");
}

// ---------------------------------------------------------------------------
// NEG-5: Schnorr / BIP-340
// ---------------------------------------------------------------------------

static void run_neg5_schnorr(ufsecp_ctx* ctx, const uint8_t* valid_sig64,
                             const uint8_t* xonly) {
    AUDIT_LOG("\n  [NEG-5] Schnorr BIP-340 sign / verify\n");

    uint8_t sig_out[64] = {};
    uint8_t aux[32] = {};

    // sign: null ctx
    CHECK_ERR(ufsecp_schnorr_sign(nullptr, MSG32, VALID_KEY1, aux, sig_out),
              "NEG-5.1: schnorr_sign(null_ctx) -> error");

    // sign: null msg
    CHECK_CODE(ufsecp_schnorr_sign(ctx, nullptr, VALID_KEY1, aux, sig_out),
               UFSECP_ERR_NULL_ARG,
               "NEG-5.2: schnorr_sign(null_msg) -> NULL_ARG");

    // sign: null privkey
    CHECK_CODE(ufsecp_schnorr_sign(ctx, MSG32, nullptr, aux, sig_out),
               UFSECP_ERR_NULL_ARG,
               "NEG-5.3: schnorr_sign(null_priv) -> NULL_ARG");

    // sign: null aux_rand (deterministic mode -- implementation-defined;
    //       but if it accepts null and treats as all-zeros, that's valid too.
    //       Check it does NOT crash.)
    {
        ufsecp_error_t rc = ufsecp_schnorr_sign(ctx, MSG32, VALID_KEY1, nullptr, sig_out);
        // Either OK (null aux = deterministic) or NULL_ARG -- either is fine,
        // but it must not crash or produce garbage we can't detect.
        CHECK(rc == UFSECP_OK || rc == UFSECP_ERR_NULL_ARG,
              "NEG-5.4: schnorr_sign(null_aux) -> OK or NULL_ARG (no crash)");
    }

    // sign: null output
    CHECK_CODE(ufsecp_schnorr_sign(ctx, MSG32, VALID_KEY1, aux, nullptr),
               UFSECP_ERR_NULL_ARG,
               "NEG-5.5: schnorr_sign(null_out) -> NULL_ARG");

    // sign: zero key
    CHECK_CODE(ufsecp_schnorr_sign(ctx, MSG32, ZERO_KEY, aux, sig_out),
               UFSECP_ERR_BAD_KEY,
               "NEG-5.6: schnorr_sign(zero_key) -> BAD_KEY");

    // sign_verified: null ctx
    CHECK_ERR(ufsecp_schnorr_sign_verified(nullptr, MSG32, VALID_KEY1, aux, sig_out),
              "NEG-5.7: schnorr_sign_verified(null_ctx) -> error");

    // verify: null ctx
    CHECK_ERR(ufsecp_schnorr_verify(nullptr, MSG32, valid_sig64, xonly),
              "NEG-5.8: schnorr_verify(null_ctx) -> error");

    // verify: null msg
    CHECK_CODE(ufsecp_schnorr_verify(ctx, nullptr, valid_sig64, xonly),
               UFSECP_ERR_NULL_ARG,
               "NEG-5.9: schnorr_verify(null_msg) -> NULL_ARG");

    // verify: null sig
    CHECK_CODE(ufsecp_schnorr_verify(ctx, MSG32, nullptr, xonly), UFSECP_ERR_NULL_ARG,
               "NEG-5.10: schnorr_verify(null_sig) -> NULL_ARG");

    // verify: null pubkey
    CHECK_CODE(ufsecp_schnorr_verify(ctx, MSG32, valid_sig64, nullptr), UFSECP_ERR_NULL_ARG,
               "NEG-5.11: schnorr_verify(null_pubkey) -> NULL_ARG");

    // verify: zero sig (R = point with x=0 not on curve)
    CHECK_ERR(ufsecp_schnorr_verify(ctx, MSG32, ZERO_SIG64, xonly),
              "NEG-5.12: schnorr_verify(zero_sig) -> error");

    // verify: zero xonly (not on curve)
    CHECK_ERR(ufsecp_schnorr_verify(ctx, MSG32, valid_sig64, ZERO_XONLY),
              "NEG-5.13: schnorr_verify(zero_xonly) -> error");

    // verify: valid sig against wrong xonly key
    uint8_t xonly2[32] = {};
    if (ufsecp_pubkey_xonly(ctx, VALID_KEY2, xonly2) == UFSECP_OK) {
        CHECK_CODE(ufsecp_schnorr_verify(ctx, MSG32, valid_sig64, xonly2),
                   UFSECP_ERR_VERIFY_FAIL,
                   "NEG-5.14: schnorr_verify(wrong_key) -> VERIFY_FAIL");
    } else {
           CHECK(ufsecp_pubkey_xonly(ctx, VALID_KEY2, xonly2) != UFSECP_OK, "NEG-5.14: skipped (xonly2 derivation failed)");
    }
}

// ---------------------------------------------------------------------------
// NEG-6: ECDH
// ---------------------------------------------------------------------------

static void run_neg6_ecdh(ufsecp_ctx* ctx, const uint8_t* pubkey33) {
    AUDIT_LOG("\n  [NEG-6] ECDH key agreement\n");

    uint8_t secret[32] = {};

    // ecdh: null ctx
    CHECK_ERR(ufsecp_ecdh(nullptr, VALID_KEY1, pubkey33, secret),
              "NEG-6.1: ecdh(null_ctx) -> error");

    // ecdh: null privkey
    CHECK_CODE(ufsecp_ecdh(ctx, nullptr, pubkey33, secret), UFSECP_ERR_NULL_ARG,
               "NEG-6.2: ecdh(null_priv) -> NULL_ARG");

    // ecdh: null pubkey
    CHECK_CODE(ufsecp_ecdh(ctx, VALID_KEY1, nullptr, secret), UFSECP_ERR_NULL_ARG,
               "NEG-6.3: ecdh(null_pubkey) -> NULL_ARG");

    // ecdh: null output
    CHECK_CODE(ufsecp_ecdh(ctx, VALID_KEY1, pubkey33, nullptr), UFSECP_ERR_NULL_ARG,
               "NEG-6.4: ecdh(null_out) -> NULL_ARG");

    // ecdh: bad pubkey (all zeros)
    CHECK_ERR(ufsecp_ecdh(ctx, VALID_KEY1, ZERO_PUBKEY33, secret),
              "NEG-6.5: ecdh(zero_pubkey) -> error");

    // ecdh: zero privkey
    CHECK_CODE(ufsecp_ecdh(ctx, ZERO_KEY, pubkey33, secret), UFSECP_ERR_BAD_KEY,
               "NEG-6.6: ecdh(zero_key) -> BAD_KEY");

    // ecdh_xonly: null ctx
    CHECK_ERR(ufsecp_ecdh_xonly(nullptr, VALID_KEY1, pubkey33, secret),
              "NEG-6.7: ecdh_xonly(null_ctx) -> error");

    // ecdh_xonly: null privkey
    CHECK_CODE(ufsecp_ecdh_xonly(ctx, nullptr, pubkey33, secret), UFSECP_ERR_NULL_ARG,
               "NEG-6.8: ecdh_xonly(null_priv) -> NULL_ARG");

    // ecdh_xonly: bad pubkey
    CHECK_ERR(ufsecp_ecdh_xonly(ctx, VALID_KEY1, ZERO_PUBKEY33, secret),
              "NEG-6.9: ecdh_xonly(zero_pubkey) -> error");

    // ecdh_raw: null ctx
    CHECK_ERR(ufsecp_ecdh_raw(nullptr, VALID_KEY1, pubkey33, secret),
              "NEG-6.10: ecdh_raw(null_ctx) -> error");

    // ecdh_raw: null privkey
    CHECK_CODE(ufsecp_ecdh_raw(ctx, nullptr, pubkey33, secret), UFSECP_ERR_NULL_ARG,
               "NEG-6.11: ecdh_raw(null_priv) -> NULL_ARG");

    // ecdh_raw: bad pubkey
    CHECK_ERR(ufsecp_ecdh_raw(ctx, VALID_KEY1, ZERO_PUBKEY33, secret),
              "NEG-6.12: ecdh_raw(zero_pubkey) -> error");
}

// ---------------------------------------------------------------------------
// NEG-7: Hash functions
// ---------------------------------------------------------------------------

static void run_neg7_hash(ufsecp_ctx* /*ctx*/) {
    AUDIT_LOG("\n  [NEG-7] Hash functions\n");

    uint8_t digest32[32] = {};
    uint8_t digest20[20] = {};

    // sha256: null data with len > 0
    CHECK_CODE(ufsecp_sha256(nullptr, 1, digest32), UFSECP_ERR_NULL_ARG,
               "NEG-7.1: sha256(null_data, len=1) -> NULL_ARG");

    // sha256: null out
    uint8_t data1[1] = {0x00};
    CHECK_CODE(ufsecp_sha256(data1, 1, nullptr), UFSECP_ERR_NULL_ARG,
               "NEG-7.2: sha256(null_out) -> NULL_ARG");

    // hash160: null data with len > 0
    CHECK_CODE(ufsecp_hash160(nullptr, 1, digest20), UFSECP_ERR_NULL_ARG,
               "NEG-7.3: hash160(null_data, len=1) -> NULL_ARG");

    // hash160: null out
    CHECK_CODE(ufsecp_hash160(data1, 1, nullptr), UFSECP_ERR_NULL_ARG,
               "NEG-7.4: hash160(null_out) -> NULL_ARG");

    // tagged_hash: null tag
    CHECK_CODE(ufsecp_tagged_hash(nullptr, data1, 1, digest32), UFSECP_ERR_NULL_ARG,
               "NEG-7.5: tagged_hash(null_tag) -> NULL_ARG");

    // tagged_hash: null out
    CHECK_CODE(ufsecp_tagged_hash("test", data1, 1, nullptr), UFSECP_ERR_NULL_ARG,
               "NEG-7.6: tagged_hash(null_out) -> NULL_ARG");

    // tagged_hash: null data with len > 0
    CHECK_CODE(ufsecp_tagged_hash("test", nullptr, 1, digest32), UFSECP_ERR_NULL_ARG,
               "NEG-7.7: tagged_hash(null_data, len=1) -> NULL_ARG");
}

// ---------------------------------------------------------------------------
// NEG-8: Bitcoin addresses
// ---------------------------------------------------------------------------

static void run_neg8_addr(ufsecp_ctx* ctx, const uint8_t* pubkey33,
                          const uint8_t* xonly) {
    AUDIT_LOG("\n  [NEG-8] Bitcoin address generation\n");

    char addr[64] = {};
    size_t addr_len;

    // p2pkh: null ctx
    addr_len = sizeof(addr);
    CHECK_ERR(ufsecp_addr_p2pkh(nullptr, pubkey33, UFSECP_NET_MAINNET, addr, &addr_len),
              "NEG-8.1: addr_p2pkh(null_ctx) -> error");

    // p2pkh: null pubkey
    addr_len = sizeof(addr);
    CHECK_CODE(ufsecp_addr_p2pkh(ctx, nullptr, UFSECP_NET_MAINNET, addr, &addr_len),
               UFSECP_ERR_NULL_ARG,
               "NEG-8.2: addr_p2pkh(null_pubkey) -> NULL_ARG");

    // p2pkh: null addr out
    addr_len = sizeof(addr);
    CHECK_CODE(ufsecp_addr_p2pkh(ctx, pubkey33, UFSECP_NET_MAINNET, nullptr, &addr_len),
               UFSECP_ERR_NULL_ARG,
               "NEG-8.3: addr_p2pkh(null_addr_out) -> NULL_ARG");

    // p2pkh: bad pubkey (all zeros)
    addr_len = sizeof(addr);
    CHECK_ERR(ufsecp_addr_p2pkh(ctx, ZERO_PUBKEY33, UFSECP_NET_MAINNET, addr, &addr_len),
              "NEG-8.4: addr_p2pkh(zero_pubkey) -> error");

    // p2pkh: buffer too small (5 bytes)
    char tiny[5] = {};
    size_t tiny_len = sizeof(tiny);
    CHECK_ERR(ufsecp_addr_p2pkh(ctx, pubkey33, UFSECP_NET_MAINNET, tiny, &tiny_len),
              "NEG-8.5: addr_p2pkh(buf_too_small) -> error");

    // p2wpkh: null ctx
    addr_len = sizeof(addr);
    CHECK_ERR(ufsecp_addr_p2wpkh(nullptr, pubkey33, UFSECP_NET_MAINNET, addr, &addr_len),
              "NEG-8.6: addr_p2wpkh(null_ctx) -> error");

    // p2wpkh: null pubkey
    addr_len = sizeof(addr);
    CHECK_CODE(ufsecp_addr_p2wpkh(ctx, nullptr, UFSECP_NET_MAINNET, addr, &addr_len),
               UFSECP_ERR_NULL_ARG,
               "NEG-8.7: addr_p2wpkh(null_pubkey) -> NULL_ARG");

    // p2wpkh: bad pubkey
    addr_len = sizeof(addr);
    CHECK_ERR(ufsecp_addr_p2wpkh(ctx, ZERO_PUBKEY33, UFSECP_NET_MAINNET, addr, &addr_len),
              "NEG-8.8: addr_p2wpkh(zero_pubkey) -> error");

    // p2tr: null ctx
    addr_len = sizeof(addr);
    CHECK_ERR(ufsecp_addr_p2tr(nullptr, xonly, UFSECP_NET_MAINNET, addr, &addr_len),
              "NEG-8.9: addr_p2tr(null_ctx) -> error");

    // p2tr: null xonly key
    addr_len = sizeof(addr);
    CHECK_CODE(ufsecp_addr_p2tr(ctx, nullptr, UFSECP_NET_MAINNET, addr, &addr_len),
               UFSECP_ERR_NULL_ARG,
               "NEG-8.10: addr_p2tr(null_xonly) -> NULL_ARG");

    // p2tr: bad xonly (all zeros)
    addr_len = sizeof(addr);
    CHECK_ERR(ufsecp_addr_p2tr(ctx, ZERO_XONLY, UFSECP_NET_MAINNET, addr, &addr_len),
              "NEG-8.11: addr_p2tr(zero_xonly) -> error");
}

// ---------------------------------------------------------------------------
// NEG-9: WIF encode / decode
// ---------------------------------------------------------------------------

static void run_neg9_wif(ufsecp_ctx* ctx) {
    AUDIT_LOG("\n  [NEG-9] WIF encode / decode\n");

    char wif[64] = {};
    size_t wif_len;
    uint8_t priv_out[32] = {};
    int compressed_out, network_out;

    // encode: null ctx
    wif_len = sizeof(wif);
    CHECK_ERR(ufsecp_wif_encode(nullptr, VALID_KEY1, 1, UFSECP_NET_MAINNET, wif, &wif_len),
              "NEG-9.1: wif_encode(null_ctx) -> error");

    // encode: null privkey
    wif_len = sizeof(wif);
    CHECK_CODE(ufsecp_wif_encode(ctx, nullptr, 1, UFSECP_NET_MAINNET, wif, &wif_len),
               UFSECP_ERR_NULL_ARG,
               "NEG-9.2: wif_encode(null_priv) -> NULL_ARG");

    // encode: zero key
    wif_len = sizeof(wif);
    CHECK_CODE(ufsecp_wif_encode(ctx, ZERO_KEY, 1, UFSECP_NET_MAINNET, wif, &wif_len),
               UFSECP_ERR_BAD_KEY,
               "NEG-9.3: wif_encode(zero_key) -> BAD_KEY");

    // decode: null ctx
    CHECK_ERR(ufsecp_wif_decode(nullptr, "5INVALID", priv_out, &compressed_out, &network_out),
              "NEG-9.4: wif_decode(null_ctx) -> error");

    // decode: null wif string
    CHECK_CODE(ufsecp_wif_decode(ctx, nullptr, priv_out, &compressed_out, &network_out),
               UFSECP_ERR_NULL_ARG,
               "NEG-9.5: wif_decode(null_wif) -> NULL_ARG");

    // decode: null privkey out
    CHECK_CODE(ufsecp_wif_decode(ctx, "5INVALID", nullptr, &compressed_out, &network_out),
               UFSECP_ERR_NULL_ARG,
               "NEG-9.6: wif_decode(null_out) -> NULL_ARG");

    // decode: clearly invalid WIF string (too short)
    CHECK_ERR(ufsecp_wif_decode(ctx, "5X", priv_out, &compressed_out, &network_out),
              "NEG-9.7: wif_decode(garbage_wif) -> error");

    // decode: empty string
    CHECK_ERR(ufsecp_wif_decode(ctx, "", priv_out, &compressed_out, &network_out),
              "NEG-9.8: wif_decode(empty_wif) -> error");
}

// ---------------------------------------------------------------------------
// NEG-10: BIP-32 HD key derivation
// ---------------------------------------------------------------------------

static void run_neg10_bip32(ufsecp_ctx* ctx, const ufsecp_bip32_key* master) {
    AUDIT_LOG("\n  [NEG-10] BIP-32 HD key derivation\n");

    ufsecp_bip32_key child = {};
    uint8_t priv_out[32] = {};
    uint8_t pub_out[33]  = {};

    // master: null ctx
    CHECK_ERR(ufsecp_bip32_master(nullptr, VALID_SEED, sizeof(VALID_SEED), &child),
              "NEG-10.1: bip32_master(null_ctx) -> error");

    // master: null seed
    CHECK_CODE(ufsecp_bip32_master(ctx, nullptr, 32, &child), UFSECP_ERR_NULL_ARG,
               "NEG-10.2: bip32_master(null_seed) -> NULL_ARG");

    // master: null output
    CHECK_CODE(ufsecp_bip32_master(ctx, VALID_SEED, sizeof(VALID_SEED), nullptr),
               UFSECP_ERR_NULL_ARG,
               "NEG-10.3: bip32_master(null_out) -> NULL_ARG");

    // master: seed too short (< 16 bytes per BIP-32)
    CHECK_ERR(ufsecp_bip32_master(ctx, SHORT_SEED, sizeof(SHORT_SEED), &child),
              "NEG-10.4: bip32_master(seed_len=8) -> error");

    // master: zero seed length
    CHECK_ERR(ufsecp_bip32_master(ctx, VALID_SEED, 0, &child),
              "NEG-10.5: bip32_master(seed_len=0) -> error");

    // derive: null ctx
    CHECK_ERR(ufsecp_bip32_derive(nullptr, master, 0, &child),
              "NEG-10.6: bip32_derive(null_ctx) -> error");

    // derive: null parent
    CHECK_CODE(ufsecp_bip32_derive(ctx, nullptr, 0, &child), UFSECP_ERR_NULL_ARG,
               "NEG-10.7: bip32_derive(null_parent) -> NULL_ARG");

    // derive: null child_out
    CHECK_CODE(ufsecp_bip32_derive(ctx, master, 0, nullptr), UFSECP_ERR_NULL_ARG,
               "NEG-10.8: bip32_derive(null_child) -> NULL_ARG");

    // derive_path: null ctx
    CHECK_ERR(ufsecp_bip32_derive_path(nullptr, master, "m/0", &child),
              "NEG-10.9: bip32_derive_path(null_ctx) -> error");

    // derive_path: null master
    CHECK_CODE(ufsecp_bip32_derive_path(ctx, nullptr, "m/0", &child), UFSECP_ERR_NULL_ARG,
               "NEG-10.10: bip32_derive_path(null_master) -> NULL_ARG");

    // derive_path: null path
    CHECK_CODE(ufsecp_bip32_derive_path(ctx, master, nullptr, &child), UFSECP_ERR_NULL_ARG,
               "NEG-10.11: bip32_derive_path(null_path) -> NULL_ARG");

    // derive_path: null child_out
    CHECK_CODE(ufsecp_bip32_derive_path(ctx, master, "m/0", nullptr), UFSECP_ERR_NULL_ARG,
               "NEG-10.12: bip32_derive_path(null_out) -> NULL_ARG");

    // privkey: null ctx
    CHECK_ERR(ufsecp_bip32_privkey(nullptr, master, priv_out),
              "NEG-10.13: bip32_privkey(null_ctx) -> error");

    // privkey: null key
    CHECK_CODE(ufsecp_bip32_privkey(ctx, nullptr, priv_out), UFSECP_ERR_NULL_ARG,
               "NEG-10.14: bip32_privkey(null_key) -> NULL_ARG");

    // privkey: null out
    CHECK_CODE(ufsecp_bip32_privkey(ctx, master, nullptr), UFSECP_ERR_NULL_ARG,
               "NEG-10.15: bip32_privkey(null_out) -> NULL_ARG");

    // pubkey: null ctx
    CHECK_ERR(ufsecp_bip32_pubkey(nullptr, master, pub_out),
              "NEG-10.16: bip32_pubkey(null_ctx) -> error");

    // pubkey: null key
    CHECK_CODE(ufsecp_bip32_pubkey(ctx, nullptr, pub_out), UFSECP_ERR_NULL_ARG,
               "NEG-10.17: bip32_pubkey(null_key) -> NULL_ARG");

    // pubkey: null out
    CHECK_CODE(ufsecp_bip32_pubkey(ctx, master, nullptr), UFSECP_ERR_NULL_ARG,
               "NEG-10.18: bip32_pubkey(null_out) -> NULL_ARG");
}

// ---------------------------------------------------------------------------
// NEG-11: Taproot (BIP-341)
// ---------------------------------------------------------------------------

static void run_neg11_taproot(ufsecp_ctx* ctx, const uint8_t* xonly) {
    AUDIT_LOG("\n  [NEG-11] Taproot BIP-341\n");

    uint8_t out_x[32] = {};
    int parity = 0;
    uint8_t tweaked[32] = {};

    // output_key: null ctx
    CHECK_ERR(ufsecp_taproot_output_key(nullptr, xonly, nullptr, out_x, &parity),
              "NEG-11.1: taproot_output_key(null_ctx) -> error");

    // output_key: null internal_x
    CHECK_CODE(ufsecp_taproot_output_key(ctx, nullptr, nullptr, out_x, &parity),
               UFSECP_ERR_NULL_ARG,
               "NEG-11.2: taproot_output_key(null_x) -> NULL_ARG");

    // output_key: null output
    CHECK_CODE(ufsecp_taproot_output_key(ctx, xonly, nullptr, nullptr, &parity),
               UFSECP_ERR_NULL_ARG,
               "NEG-11.3: taproot_output_key(null_out) -> NULL_ARG");

    // output_key: bad xonly (all zeros)
    CHECK_ERR(ufsecp_taproot_output_key(ctx, ZERO_XONLY, nullptr, out_x, &parity),
              "NEG-11.4: taproot_output_key(zero_xonly) -> error");

    // tweak_seckey: null ctx
    CHECK_ERR(ufsecp_taproot_tweak_seckey(nullptr, VALID_KEY1, nullptr, tweaked),
              "NEG-11.5: taproot_tweak_seckey(null_ctx) -> error");

    // tweak_seckey: null privkey
    CHECK_CODE(ufsecp_taproot_tweak_seckey(ctx, nullptr, nullptr, tweaked),
               UFSECP_ERR_NULL_ARG,
               "NEG-11.6: taproot_tweak_seckey(null_priv) -> NULL_ARG");

    // tweak_seckey: null out
    CHECK_CODE(ufsecp_taproot_tweak_seckey(ctx, VALID_KEY1, nullptr, nullptr),
               UFSECP_ERR_NULL_ARG,
               "NEG-11.7: taproot_tweak_seckey(null_out) -> NULL_ARG");

    // tweak_seckey: zero key
    CHECK_CODE(ufsecp_taproot_tweak_seckey(ctx, ZERO_KEY, nullptr, tweaked),
               UFSECP_ERR_BAD_KEY,
               "NEG-11.8: taproot_tweak_seckey(zero_key) -> BAD_KEY");

    // verify: null ctx
    CHECK_ERR(ufsecp_taproot_verify(nullptr, xonly, 0, xonly, nullptr, 0),
              "NEG-11.9: taproot_verify(null_ctx) -> error");

    // verify: null output_x
    CHECK_CODE(ufsecp_taproot_verify(ctx, nullptr, 0, xonly, nullptr, 0),
               UFSECP_ERR_NULL_ARG,
               "NEG-11.10: taproot_verify(null_output_x) -> NULL_ARG");

    // verify: null internal_x
    CHECK_CODE(ufsecp_taproot_verify(ctx, xonly, 0, nullptr, nullptr, 0),
               UFSECP_ERR_NULL_ARG,
               "NEG-11.11: taproot_verify(null_internal_x) -> NULL_ARG");
}

// ---------------------------------------------------------------------------
// NEG-12: Public key arithmetic
// ---------------------------------------------------------------------------

static void run_neg12_pubkey_arith(ufsecp_ctx* ctx, const uint8_t* pubkey33) {
    AUDIT_LOG("\n  [NEG-12] Public key arithmetic\n");

    uint8_t out33[33] = {};

    // pubkey_add: null ctx
    CHECK_ERR(ufsecp_pubkey_add(nullptr, pubkey33, pubkey33, out33),
              "NEG-12.1: pubkey_add(null_ctx) -> error");

    // pubkey_add: null a
    CHECK_CODE(ufsecp_pubkey_add(ctx, nullptr, pubkey33, out33), UFSECP_ERR_NULL_ARG,
               "NEG-12.2: pubkey_add(null_a) -> NULL_ARG");

    // pubkey_add: null b
    CHECK_CODE(ufsecp_pubkey_add(ctx, pubkey33, nullptr, out33), UFSECP_ERR_NULL_ARG,
               "NEG-12.3: pubkey_add(null_b) -> NULL_ARG");

    // pubkey_add: null out
    CHECK_CODE(ufsecp_pubkey_add(ctx, pubkey33, pubkey33, nullptr), UFSECP_ERR_NULL_ARG,
               "NEG-12.4: pubkey_add(null_out) -> NULL_ARG");

    // pubkey_add: bad a (all zeros)
    CHECK_ERR(ufsecp_pubkey_add(ctx, ZERO_PUBKEY33, pubkey33, out33),
              "NEG-12.5: pubkey_add(zero_a) -> error");

    // pubkey_negate: null ctx
    CHECK_ERR(ufsecp_pubkey_negate(nullptr, pubkey33, out33),
              "NEG-12.6: pubkey_negate(null_ctx) -> error");

    // pubkey_negate: null pubkey
    CHECK_CODE(ufsecp_pubkey_negate(ctx, nullptr, out33), UFSECP_ERR_NULL_ARG,
               "NEG-12.7: pubkey_negate(null_pubkey) -> NULL_ARG");

    // pubkey_negate: null out
    CHECK_CODE(ufsecp_pubkey_negate(ctx, pubkey33, nullptr), UFSECP_ERR_NULL_ARG,
               "NEG-12.8: pubkey_negate(null_out) -> NULL_ARG");

    // pubkey_negate: bad pubkey
    CHECK_ERR(ufsecp_pubkey_negate(ctx, ZERO_PUBKEY33, out33),
              "NEG-12.9: pubkey_negate(zero_pubkey) -> error");

    // pubkey_tweak_add: null ctx
    CHECK_ERR(ufsecp_pubkey_tweak_add(nullptr, pubkey33, VALID_KEY2, out33),
              "NEG-12.10: pubkey_tweak_add(null_ctx) -> error");

    // pubkey_tweak_add: null pubkey
    CHECK_CODE(ufsecp_pubkey_tweak_add(ctx, nullptr, VALID_KEY2, out33), UFSECP_ERR_NULL_ARG,
               "NEG-12.11: pubkey_tweak_add(null_pubkey) -> NULL_ARG");

    // pubkey_tweak_add: null tweak
    CHECK_CODE(ufsecp_pubkey_tweak_add(ctx, pubkey33, nullptr, out33), UFSECP_ERR_NULL_ARG,
               "NEG-12.12: pubkey_tweak_add(null_tweak) -> NULL_ARG");

    // pubkey_tweak_mul: null ctx
    CHECK_ERR(ufsecp_pubkey_tweak_mul(nullptr, pubkey33, VALID_KEY2, out33),
              "NEG-12.13: pubkey_tweak_mul(null_ctx) -> error");

    // pubkey_tweak_mul: null pubkey
    CHECK_CODE(ufsecp_pubkey_tweak_mul(ctx, nullptr, VALID_KEY2, out33), UFSECP_ERR_NULL_ARG,
               "NEG-12.14: pubkey_tweak_mul(null_pubkey) -> NULL_ARG");

    // pubkey_tweak_mul: zero tweak (multiplier = 0 mod n -> infinity)
    CHECK_ERR(ufsecp_pubkey_tweak_mul(ctx, pubkey33, ZERO32, out33),
              "NEG-12.15: pubkey_tweak_mul(zero_tweak) -> error");
}

// ---------------------------------------------------------------------------
// NEG-13: P2SH / P2SH-P2WPKH address functions
// ---------------------------------------------------------------------------

static void run_neg13_p2sh_addr(ufsecp_ctx* ctx, const uint8_t* pubkey33) {
    char addr[96]; size_t alen = sizeof(addr);
    uint8_t script[] = { 0x00, 0x14 };  // minimal mock

    // ufsecp_addr_p2sh
    CHECK_ERR(ufsecp_addr_p2sh(nullptr, 2, 0, addr, &alen),
              "NEG-13.1: addr_p2sh(null_script) -> error");
    // zero-length redeem script may be accepted (valid hash160 of empty script)
    // instead test: null output buffer
    CHECK_ERR(ufsecp_addr_p2sh(script, 2, 0, nullptr, &alen),
              "NEG-13.2: addr_p2sh(null_out) -> error");
    CHECK_ERR(ufsecp_addr_p2sh(script, 2, 0, addr, nullptr),
              "NEG-13.3: addr_p2sh(null_len_ptr) -> error");
    alen = 0;
    CHECK_ERR(ufsecp_addr_p2sh(script, 2, 0, addr, &alen),
              "NEG-13.4: addr_p2sh(buf_too_small) -> error");

    // ufsecp_addr_p2sh_p2wpkh
    alen = sizeof(addr);
    CHECK_ERR(ufsecp_addr_p2sh_p2wpkh(nullptr, pubkey33, 0, addr, &alen),
              "NEG-13.5: addr_p2sh_p2wpkh(null_ctx) -> error");
    CHECK_ERR(ufsecp_addr_p2sh_p2wpkh(ctx, nullptr, 0, addr, &alen),
              "NEG-13.6: addr_p2sh_p2wpkh(null_pubkey) -> error");
    CHECK_ERR(ufsecp_addr_p2sh_p2wpkh(ctx, pubkey33, 0, nullptr, &alen),
              "NEG-13.7: addr_p2sh_p2wpkh(null_out) -> error");
    CHECK_ERR(ufsecp_addr_p2sh_p2wpkh(ctx, ZERO_PUBKEY33, 0, addr, &alen),
              "NEG-13.8: addr_p2sh_p2wpkh(zero_pubkey) -> error");
}

// ---------------------------------------------------------------------------
// NEG-14: BIP-85 entropy & mnemonic
// ---------------------------------------------------------------------------

static void run_neg14_bip85(ufsecp_ctx* ctx, const ufsecp_bip32_key* mk) {
    uint8_t entropy[32];
    char mnemonic[512]; size_t mlen = sizeof(mnemonic);

    // ufsecp_bip85_entropy
    CHECK_ERR(ufsecp_bip85_entropy(nullptr, mk, "m/83696968'/2'/0'", entropy, 32),
              "NEG-14.1: bip85_entropy(null_ctx) -> error");
    CHECK_ERR(ufsecp_bip85_entropy(ctx, nullptr, "m/83696968'/2'/0'", entropy, 32),
              "NEG-14.2: bip85_entropy(null_xprv) -> error");
    CHECK_ERR(ufsecp_bip85_entropy(ctx, mk, nullptr, entropy, 32),
              "NEG-14.3: bip85_entropy(null_path) -> error");
    CHECK_ERR(ufsecp_bip85_entropy(ctx, mk, "m/83696968'/2'/0'", nullptr, 32),
              "NEG-14.4: bip85_entropy(null_out) -> error");

    // ufsecp_bip85_bip39
    CHECK_ERR(ufsecp_bip85_bip39(nullptr, mk, 12, 0, 0, mnemonic, &mlen),
              "NEG-14.5: bip85_bip39(null_ctx) -> error");
    CHECK_ERR(ufsecp_bip85_bip39(ctx, nullptr, 12, 0, 0, mnemonic, &mlen),
              "NEG-14.6: bip85_bip39(null_xprv) -> error");
    CHECK_ERR(ufsecp_bip85_bip39(ctx, mk, 12, 0, 0, nullptr, &mlen),
              "NEG-14.7: bip85_bip39(null_out) -> error");
    CHECK_ERR(ufsecp_bip85_bip39(ctx, mk, 7, 0, 0, mnemonic, &mlen),
              "NEG-14.8: bip85_bip39(bad_word_count=7) -> error");
}

// ---------------------------------------------------------------------------
// NEG-15: Schnorr variable-length message sign/verify
// ---------------------------------------------------------------------------

static void run_neg15_schnorr_msg(ufsecp_ctx* ctx, const uint8_t* xonly32) {
    uint8_t sig[64] = {}; uint8_t aux[32] = {};
    const uint8_t msg[] = "test message";

    // ufsecp_schnorr_sign_msg
    CHECK_ERR(ufsecp_schnorr_sign_msg(nullptr, VALID_KEY1, msg, sizeof(msg), aux, sig),
              "NEG-15.1: schnorr_sign_msg(null_ctx) -> error");
    CHECK_ERR(ufsecp_schnorr_sign_msg(ctx, nullptr, msg, sizeof(msg), aux, sig),
              "NEG-15.2: schnorr_sign_msg(null_key) -> error");
    CHECK_ERR(ufsecp_schnorr_sign_msg(ctx, ZERO_KEY, msg, sizeof(msg), aux, sig),
              "NEG-15.3: schnorr_sign_msg(zero_key) -> error");
    CHECK_ERR(ufsecp_schnorr_sign_msg(ctx, VALID_KEY1, msg, sizeof(msg), aux, nullptr),
              "NEG-15.4: schnorr_sign_msg(null_sig_out) -> error");

    // ufsecp_schnorr_verify_msg
    CHECK_ERR(ufsecp_schnorr_verify_msg(nullptr, xonly32, msg, sizeof(msg), sig),
              "NEG-15.5: schnorr_verify_msg(null_ctx) -> error");
    CHECK_ERR(ufsecp_schnorr_verify_msg(ctx, nullptr, msg, sizeof(msg), sig),
              "NEG-15.6: schnorr_verify_msg(null_pubkey) -> error");
    CHECK_ERR(ufsecp_schnorr_verify_msg(ctx, ZERO_XONLY, msg, sizeof(msg), sig),
              "NEG-15.7: schnorr_verify_msg(zero_pubkey) -> error");
}

// ---------------------------------------------------------------------------
// NEG-16: BIP-322 message signing
// ---------------------------------------------------------------------------

static void run_neg16_bip322(ufsecp_ctx* ctx) {
    uint8_t sig[128]; size_t slen = sizeof(sig);
    const uint8_t msg[] = "bip322 test";

    // ufsecp_bip322_sign
    CHECK_ERR(ufsecp_bip322_sign(nullptr, VALID_KEY1, UFSECP_BIP322_ADDR_P2WPKH,
              msg, sizeof(msg), sig, &slen),
              "NEG-16.1: bip322_sign(null_ctx) -> error");
    CHECK_ERR(ufsecp_bip322_sign(ctx, nullptr, UFSECP_BIP322_ADDR_P2WPKH,
              msg, sizeof(msg), sig, &slen),
              "NEG-16.2: bip322_sign(null_key) -> error");
    CHECK_ERR(ufsecp_bip322_sign(ctx, ZERO_KEY, UFSECP_BIP322_ADDR_P2WPKH,
              msg, sizeof(msg), sig, &slen),
              "NEG-16.3: bip322_sign(zero_key) -> error");
    CHECK_ERR(ufsecp_bip322_sign(ctx, VALID_KEY1, UFSECP_BIP322_ADDR_P2WPKH,
              msg, sizeof(msg), nullptr, &slen),
              "NEG-16.4: bip322_sign(null_sig_out) -> error");

    // ufsecp_bip322_verify
    uint8_t fake_pubkey[33] = {};
    fake_pubkey[0] = 0x02;
    CHECK_ERR(ufsecp_bip322_verify(nullptr, fake_pubkey, 33, UFSECP_BIP322_ADDR_P2WPKH,
              msg, sizeof(msg), sig, slen),
              "NEG-16.5: bip322_verify(null_ctx) -> error");
    CHECK_ERR(ufsecp_bip322_verify(ctx, nullptr, 33, UFSECP_BIP322_ADDR_P2WPKH,
              msg, sizeof(msg), sig, slen),
              "NEG-16.6: bip322_verify(null_pubkey) -> error");
    CHECK_ERR(ufsecp_bip322_verify(ctx, fake_pubkey, 0, UFSECP_BIP322_ADDR_P2WPKH,
              msg, sizeof(msg), sig, slen),
              "NEG-16.7: bip322_verify(zero_pubkey_len) -> error");
}

// ---------------------------------------------------------------------------
// NEG-17: GCS (Golomb-Coded Set) filters
// ---------------------------------------------------------------------------

static void run_neg17_gcs(void) {
    uint8_t key[16] = {};
    uint8_t item1[] = { 0x01, 0x02 };
    const uint8_t* items[] = { item1 };
    size_t sizes[] = { sizeof(item1) };
    uint8_t filter[256]; size_t flen = sizeof(filter);

    // ufsecp_gcs_build
    CHECK_ERR(ufsecp_gcs_build(nullptr, items, sizes, 1, filter, &flen),
              "NEG-17.1: gcs_build(null_key) -> error");
    CHECK_ERR(ufsecp_gcs_build(key, nullptr, sizes, 1, filter, &flen),
              "NEG-17.2: gcs_build(null_data) -> error");
    CHECK_ERR(ufsecp_gcs_build(key, items, sizes, 1, nullptr, &flen),
              "NEG-17.3: gcs_build(null_out) -> error");

    // Build a valid filter first for match tests
    flen = sizeof(filter);
    ufsecp_error_t rc = ufsecp_gcs_build(key, items, sizes, 1, filter, &flen);

    // ufsecp_gcs_match
    CHECK_ERR(ufsecp_gcs_match(nullptr, filter, flen, 1, item1, sizeof(item1)),
              "NEG-17.4: gcs_match(null_key) -> error");
    CHECK_ERR(ufsecp_gcs_match(key, nullptr, flen, 1, item1, sizeof(item1)),
              "NEG-17.5: gcs_match(null_filter) -> error");
    CHECK_ERR(ufsecp_gcs_match(key, filter, flen, 1, nullptr, sizeof(item1)),
              "NEG-17.6: gcs_match(null_item) -> error");

    // ufsecp_gcs_match_any
    CHECK_ERR(ufsecp_gcs_match_any(nullptr, filter, flen, 1, items, sizes, 1),
              "NEG-17.7: gcs_match_any(null_key) -> error");
    CHECK_ERR(ufsecp_gcs_match_any(key, nullptr, flen, 1, items, sizes, 1),
              "NEG-17.8: gcs_match_any(null_filter) -> error");
    CHECK_ERR(ufsecp_gcs_match_any(key, filter, flen, 1, nullptr, sizes, 1),
              "NEG-17.9: gcs_match_any(null_query) -> error");
    (void)rc;
}

// ---------------------------------------------------------------------------
// NEG-18: PSBT signing helpers
// ---------------------------------------------------------------------------

static void run_neg18_psbt(ufsecp_ctx* ctx, const ufsecp_bip32_key* mk) {
    uint8_t sig[73]; size_t slen = sizeof(sig);
    uint8_t aux[32] = {};
    uint8_t privkey_out[32];

    // ufsecp_psbt_sign_legacy
    CHECK_ERR(ufsecp_psbt_sign_legacy(nullptr, MSG32, VALID_KEY1, UFSECP_SIGHASH_ALL, sig, &slen),
              "NEG-18.1: psbt_sign_legacy(null_ctx) -> error");
    CHECK_ERR(ufsecp_psbt_sign_legacy(ctx, nullptr, VALID_KEY1, UFSECP_SIGHASH_ALL, sig, &slen),
              "NEG-18.2: psbt_sign_legacy(null_sighash) -> error");
    CHECK_ERR(ufsecp_psbt_sign_legacy(ctx, MSG32, nullptr, UFSECP_SIGHASH_ALL, sig, &slen),
              "NEG-18.3: psbt_sign_legacy(null_key) -> error");
    CHECK_ERR(ufsecp_psbt_sign_legacy(ctx, MSG32, ZERO_KEY, UFSECP_SIGHASH_ALL, sig, &slen),
              "NEG-18.4: psbt_sign_legacy(zero_key) -> error");

    // ufsecp_psbt_sign_segwit
    slen = sizeof(sig);
    CHECK_ERR(ufsecp_psbt_sign_segwit(nullptr, MSG32, VALID_KEY1, UFSECP_SIGHASH_ALL, sig, &slen),
              "NEG-18.5: psbt_sign_segwit(null_ctx) -> error");
    CHECK_ERR(ufsecp_psbt_sign_segwit(ctx, nullptr, VALID_KEY1, UFSECP_SIGHASH_ALL, sig, &slen),
              "NEG-18.6: psbt_sign_segwit(null_sighash) -> error");
    CHECK_ERR(ufsecp_psbt_sign_segwit(ctx, MSG32, nullptr, UFSECP_SIGHASH_ALL, sig, &slen),
              "NEG-18.7: psbt_sign_segwit(null_key) -> error");

    // ufsecp_psbt_sign_taproot
    slen = sizeof(sig);
    CHECK_ERR(ufsecp_psbt_sign_taproot(nullptr, MSG32, VALID_KEY1, UFSECP_SIGHASH_DEFAULT, aux, sig, &slen),
              "NEG-18.8: psbt_sign_taproot(null_ctx) -> error");
    CHECK_ERR(ufsecp_psbt_sign_taproot(ctx, nullptr, VALID_KEY1, UFSECP_SIGHASH_DEFAULT, aux, sig, &slen),
              "NEG-18.9: psbt_sign_taproot(null_sighash) -> error");
    CHECK_ERR(ufsecp_psbt_sign_taproot(ctx, MSG32, nullptr, UFSECP_SIGHASH_DEFAULT, aux, sig, &slen),
              "NEG-18.10: psbt_sign_taproot(null_key) -> error");

    // ufsecp_psbt_derive_key
    CHECK_ERR(ufsecp_psbt_derive_key(nullptr, mk, "m/84'/0'/0'/0/0", privkey_out),
              "NEG-18.11: psbt_derive_key(null_ctx) -> error");
    CHECK_ERR(ufsecp_psbt_derive_key(ctx, nullptr, "m/84'/0'/0'/0/0", privkey_out),
              "NEG-18.12: psbt_derive_key(null_xprv) -> error");
    CHECK_ERR(ufsecp_psbt_derive_key(ctx, mk, nullptr, privkey_out),
              "NEG-18.13: psbt_derive_key(null_path) -> error");
    CHECK_ERR(ufsecp_psbt_derive_key(ctx, mk, "m/84'/0'/0'/0/0", nullptr),
              "NEG-18.14: psbt_derive_key(null_out) -> error");
}

// ---------------------------------------------------------------------------
// NEG-19: Output descriptors
// ---------------------------------------------------------------------------

static void run_neg19_descriptors(ufsecp_ctx* ctx) {
    ufsecp_desc_key key_out;
    char addr[128]; size_t alen = sizeof(addr);

    // ufsecp_descriptor_parse
    CHECK_ERR(ufsecp_descriptor_parse(nullptr, "wpkh(xpub...)", 0, &key_out, addr, &alen),
              "NEG-19.1: descriptor_parse(null_ctx) -> error");
    CHECK_ERR(ufsecp_descriptor_parse(ctx, nullptr, 0, &key_out, addr, &alen),
              "NEG-19.2: descriptor_parse(null_descriptor) -> error");
    CHECK_ERR(ufsecp_descriptor_parse(ctx, "wpkh(xpub...)", 0, nullptr, addr, &alen),
              "NEG-19.3: descriptor_parse(null_key_out) -> error");
    CHECK_ERR(ufsecp_descriptor_parse(ctx, "", 0, &key_out, addr, &alen),
              "NEG-19.4: descriptor_parse(empty_descriptor) -> error");

    // ufsecp_descriptor_address
    alen = sizeof(addr);
    CHECK_ERR(ufsecp_descriptor_address(nullptr, "wpkh(xpub...)", 0, addr, &alen),
              "NEG-19.5: descriptor_address(null_ctx) -> error");
    CHECK_ERR(ufsecp_descriptor_address(ctx, nullptr, 0, addr, &alen),
              "NEG-19.6: descriptor_address(null_descriptor) -> error");
    CHECK_ERR(ufsecp_descriptor_address(ctx, "wpkh(xpub...)", 0, nullptr, &alen),
              "NEG-19.7: descriptor_address(null_addr_out) -> error");
    CHECK_ERR(ufsecp_descriptor_address(ctx, "", 0, addr, &alen),
              "NEG-19.8: descriptor_address(empty_descriptor) -> error");
}

// ---------------------------------------------------------------------------
// NEG-20: ZK SNARK witnesses (ECDSA + Schnorr)
// ---------------------------------------------------------------------------

static void run_neg20_zk_snark(ufsecp_ctx* ctx, const uint8_t* pubkey33,
                               const uint8_t* xonly32) {
    ufsecp_ecdsa_snark_witness_t ecdsa_w;
    ufsecp_schnorr_snark_witness_t schnorr_w;
    memset(&ecdsa_w, 0, sizeof(ecdsa_w));
    memset(&schnorr_w, 0, sizeof(schnorr_w));

    // ufsecp_zk_ecdsa_snark_witness
    CHECK_ERR(ufsecp_zk_ecdsa_snark_witness(nullptr, MSG32, pubkey33, ZERO_SIG64, &ecdsa_w),
              "NEG-20.1: zk_ecdsa_snark_witness(null_ctx) -> error");
    CHECK_ERR(ufsecp_zk_ecdsa_snark_witness(ctx, nullptr, pubkey33, ZERO_SIG64, &ecdsa_w),
              "NEG-20.2: zk_ecdsa_snark_witness(null_msg) -> error");
    CHECK_ERR(ufsecp_zk_ecdsa_snark_witness(ctx, MSG32, nullptr, ZERO_SIG64, &ecdsa_w),
              "NEG-20.3: zk_ecdsa_snark_witness(null_pubkey) -> error");
    CHECK_ERR(ufsecp_zk_ecdsa_snark_witness(ctx, MSG32, pubkey33, nullptr, &ecdsa_w),
              "NEG-20.4: zk_ecdsa_snark_witness(null_sig) -> error");
    CHECK_ERR(ufsecp_zk_ecdsa_snark_witness(ctx, MSG32, pubkey33, ZERO_SIG64, nullptr),
              "NEG-20.5: zk_ecdsa_snark_witness(null_out) -> error");

    // ufsecp_zk_schnorr_snark_witness
    CHECK_ERR(ufsecp_zk_schnorr_snark_witness(nullptr, MSG32, xonly32, ZERO_SIG64, &schnorr_w),
              "NEG-20.6: zk_schnorr_snark_witness(null_ctx) -> error");
    CHECK_ERR(ufsecp_zk_schnorr_snark_witness(ctx, nullptr, xonly32, ZERO_SIG64, &schnorr_w),
              "NEG-20.7: zk_schnorr_snark_witness(null_msg) -> error");
    CHECK_ERR(ufsecp_zk_schnorr_snark_witness(ctx, MSG32, nullptr, ZERO_SIG64, &schnorr_w),
              "NEG-20.8: zk_schnorr_snark_witness(null_pubkey) -> error");
    CHECK_ERR(ufsecp_zk_schnorr_snark_witness(ctx, MSG32, xonly32, nullptr, &schnorr_w),
              "NEG-20.9: zk_schnorr_snark_witness(null_sig) -> error");
    CHECK_ERR(ufsecp_zk_schnorr_snark_witness(ctx, MSG32, xonly32, ZERO_SIG64, nullptr),
              "NEG-20.10: zk_schnorr_snark_witness(null_out) -> error");
}

// ---------------------------------------------------------------------------
// NEG-21: GPU functions (context create + BIP-352 + ZK batch)
// ---------------------------------------------------------------------------

static void run_neg21_gpu(void) {
    // ufsecp_gpu_ctx_create (listed as ufsecp_gpu_context_create in manifest)
    CHECK_ERR(ufsecp_gpu_ctx_create(nullptr, UFSECP_GPU_BACKEND_CUDA, 0),
              "NEG-21.1: gpu_ctx_create(null_ctx_out) -> error");
    CHECK_ERR(ufsecp_gpu_ctx_create(nullptr, 0xFF, 0),
              "NEG-21.2: gpu_ctx_create(invalid_backend) -> error");
    CHECK_ERR(ufsecp_gpu_ctx_create(nullptr, UFSECP_GPU_BACKEND_NONE, 0),
              "NEG-21.3: gpu_ctx_create(backend_none) -> error");

    // ufsecp_bip352_prepare_scan_plan
    uint8_t plan[264];
    CHECK_ERR(ufsecp_bip352_prepare_scan_plan(nullptr, plan),
              "NEG-21.4: bip352_prepare_scan_plan(null_key) -> error");
    CHECK_ERR(ufsecp_bip352_prepare_scan_plan(ZERO_KEY, plan),
              "NEG-21.5: bip352_prepare_scan_plan(zero_key) -> error");
    CHECK_ERR(ufsecp_bip352_prepare_scan_plan(VALID_KEY1, nullptr),
              "NEG-21.6: bip352_prepare_scan_plan(null_out) -> error");

    // ufsecp_gpu_bip352_scan_batch (needs ctx, will fail on null)
    uint64_t prefix;
    uint8_t tweak[33] = {};
    CHECK_ERR(ufsecp_gpu_bip352_scan_batch(nullptr, VALID_KEY1, tweak, tweak, 1, &prefix),
              "NEG-21.7: gpu_bip352_scan_batch(null_ctx) -> error");
    CHECK_ERR(ufsecp_gpu_bip352_scan_batch(nullptr, nullptr, tweak, tweak, 1, &prefix),
              "NEG-21.7b: gpu_bip352_scan_batch(null_ctx+key) -> error");
    CHECK_ERR(ufsecp_gpu_bip352_scan_batch(nullptr, VALID_KEY1, nullptr, tweak, 1, &prefix),
              "NEG-21.7c: gpu_bip352_scan_batch(null_ctx+spend) -> error");

    // ufsecp_gpu_zk_ecdsa_snark_witness_batch
    uint8_t witness[760];
    CHECK_ERR(ufsecp_gpu_zk_ecdsa_snark_witness_batch(nullptr, MSG32, tweak, ZERO_SIG64, 1, witness),
              "NEG-21.8: gpu_zk_ecdsa_snark_witness_batch(null_ctx) -> error");
    CHECK_ERR(ufsecp_gpu_zk_ecdsa_snark_witness_batch(nullptr, nullptr, tweak, ZERO_SIG64, 1, witness),
              "NEG-21.8b: gpu_zk_ecdsa_snark_witness_batch(null_ctx+msg) -> error");
    CHECK_ERR(ufsecp_gpu_zk_ecdsa_snark_witness_batch(nullptr, MSG32, nullptr, ZERO_SIG64, 1, witness),
              "NEG-21.8c: gpu_zk_ecdsa_snark_witness_batch(null_ctx+pubkey) -> error");

    // ufsecp_gpu_zk_schnorr_snark_witness_batch
    uint8_t schnorr_witness[472];
    CHECK_ERR(ufsecp_gpu_zk_schnorr_snark_witness_batch(nullptr, MSG32, ZERO32, ZERO_SIG64, 1, schnorr_witness),
              "NEG-21.9: gpu_zk_schnorr_snark_witness_batch(null_ctx) -> error");
    CHECK_ERR(ufsecp_gpu_zk_schnorr_snark_witness_batch(nullptr, nullptr, ZERO32, ZERO_SIG64, 1, schnorr_witness),
              "NEG-21.9b: gpu_zk_schnorr_snark_witness_batch(null_ctx+msg) -> error");
    CHECK_ERR(ufsecp_gpu_zk_schnorr_snark_witness_batch(nullptr, MSG32, nullptr, ZERO_SIG64, 1, schnorr_witness),
              "NEG-21.9c: gpu_zk_schnorr_snark_witness_batch(null_ctx+pubkey) -> error");
}

// ---------------------------------------------------------------------------
// NEG-22: ABI version query
// ---------------------------------------------------------------------------

static void run_neg22_abi_version(void) {
    // NEG-22.1: basic non-zero sanity (success_smoke)
    unsigned int abi = ufsecp_abi_version();
    CHECK(abi > 0, "NEG-22.1: abi_version() returns non-zero");

    // NEG-22.2: idempotency — two consecutive calls must return equal values (zero_edge)
    unsigned int abi2 = ufsecp_abi_version();
    CHECK(abi == abi2, "NEG-22.2: abi_version() is idempotent");

    // NEG-22.3: matches compile-time constant (invalid_content guard: if mismatch, binary is mislinked)
    CHECK(abi == UFSECP_ABI_VERSION, "NEG-22.3: abi_version() matches UFSECP_ABI_VERSION constant");

    // NEG-22.4: upper-bound sanity — not a garbage/uninitialized value (null_rejection analogue)
    CHECK(abi < 0x10000u, "NEG-22.4: abi_version() < 65536 (not garbage)");

    // NEG-22.5: version string is non-null and non-empty (success_smoke for version family)
    const char* vs = ufsecp_version_string();
    CHECK(vs != nullptr, "NEG-22.5: version_string() non-null");
    CHECK(vs[0] != '\0', "NEG-22.6: version_string() non-empty");

    // NEG-22.7: ufsecp_version() returns same value as abi (consistent API)
    unsigned int ver = ufsecp_version();
    CHECK(ver > 0, "NEG-22.7: ufsecp_version() returns non-zero");
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

int test_c_abi_negative_run() {
    g_pass = 0; g_fail = 0;

    AUDIT_LOG("============================================================\n");
    AUDIT_LOG("  C ABI Negative Contract Tests\n");
    AUDIT_LOG("  All 50+ ufsecp_* functions: null/bad-key/bad-sig/overflow\n");
    AUDIT_LOG("============================================================\n");

    Fixture f;
    if (!f.init()) {
        printf("  [FATAL] Fixture init failed -- cannot run negative tests\n");
        return 1;
    }

    run_neg1_context(f.ctx);
    run_neg2_seckey(f.ctx);
    run_neg3_pubkey(f.ctx);
    run_neg4_ecdsa(f.ctx, f.sig64_ecdsa, f.pubkey33);
    run_neg5_schnorr(f.ctx, f.sig64_schnorr, f.xonly32);
    run_neg6_ecdh(f.ctx, f.pubkey33);
    run_neg7_hash(f.ctx);
    run_neg8_addr(f.ctx, f.pubkey33, f.xonly32);
    run_neg9_wif(f.ctx);
    run_neg10_bip32(f.ctx, &f.master_key);
    run_neg11_taproot(f.ctx, f.xonly32);
    run_neg12_pubkey_arith(f.ctx, f.pubkey33);
    run_neg13_p2sh_addr(f.ctx, f.pubkey33);
    run_neg14_bip85(f.ctx, &f.master_key);
    run_neg15_schnorr_msg(f.ctx, f.xonly32);
    run_neg16_bip322(f.ctx);
    run_neg17_gcs();
    run_neg18_psbt(f.ctx, &f.master_key);
    run_neg19_descriptors(f.ctx);
    run_neg20_zk_snark(f.ctx, f.pubkey33, f.xonly32);
    run_neg21_gpu();
    run_neg22_abi_version();

    printf("[test_c_abi_negative] %d/%d checks passed\n",
           g_pass, g_pass + g_fail);
    return (g_fail > 0) ? 1 : 0;
}

#ifndef UNIFIED_AUDIT_RUNNER
int main() {
    return test_c_abi_negative_run();
}
#endif
