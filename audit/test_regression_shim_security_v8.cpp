// ============================================================================
// test_regression_shim_security_v8.cpp — Shim security regression guards (v8)
// ============================================================================
// Covers 2026-05-13 v8 audit findings for the libsecp256k1 compatibility shim:
//   P1-SEC-NEW-001: secp256k1_ecdh rejects private key values >= curve order n
//                   (parse_bytes_strict_nonzero — Rule 11 compliance)
//   P1-SEC-RED-TEAM-008: secp256k1_ecdsa_verify rejects off-curve pubkey structs,
//                        consistent with secp256k1_ecdsa_verify_batch
//   P2-SEC-NEW-002: secp256k1_ecdh rejects off-curve public key input
//                   (invalid-curve / small-order-subgroup attack prevention)
//
// advisory=true: compiled into unified_audit_runner only when secp256k1_shim is
// linked. Otherwise shim_run_stubs_unified.cpp provides a stub returning
// ADVISORY_SKIP_CODE (77).
// ============================================================================

#ifndef UNIFIED_AUDIT_RUNNER
#define STANDALONE_TEST
#endif

#include "secp256k1.h"
#include "secp256k1_ecdh.h"
#include "secp256k1_extrakeys.h"
#include "secp256k1_schnorrsig.h"

#include <cstdio>
#include <cstring>
#include <cstdint>

namespace {

static int g_pass = 0, g_fail = 0;

#define CHECK(cond, msg) do { \
    if (cond) { ++g_pass; } \
    else { ++g_fail; std::printf("  [FAIL] %s\n", (msg)); } \
} while(0)

// secp256k1 curve order n (big-endian)
static const unsigned char CURVE_ORDER[32] = {
    0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFE,
    0xBA,0xAE,0xDC,0xE6,0xAF,0x48,0xA0,0x3B,0xBF,0xD2,0x5E,0x8C,0xD0,0x36,0x41,0x41
};

// Standard SHA-256 hashfp for ECDH
static int ecdh_hashfp_sha256(unsigned char *output, const unsigned char *x32,
                               const unsigned char * /*y32*/, void * /*data*/)
{
    (void)x32;
    // Minimal stub: write x32 directly as "output" for test purposes
    std::memcpy(output, x32, 32);
    return 1;
}

// ── P1-SEC-NEW-001: ECDH strict private key parsing (Rule 11) ───────────────

static void test_ecdh_privkey_out_of_range() {
    std::printf("  [ecdh_privkey_out_of_range] P1-SEC-NEW-001\n");
    secp256k1_context *ctx = secp256k1_context_create(
        SECP256K1_CONTEXT_SIGN | SECP256K1_CONTEXT_VERIFY);

    // Build a valid pubkey from seckey=1 to serve as the ECDH peer
    unsigned char sk_valid[32] = {0};
    sk_valid[31] = 1;
    secp256k1_pubkey peer_pub;
    secp256k1_ec_pubkey_create(ctx, &peer_pub, sk_valid);

    unsigned char output[32];

    // (a) seckey = n (ORDER) — must be rejected (== 0 mod n → parse_strict rejects)
    unsigned char sk_order[32];
    std::memcpy(sk_order, CURVE_ORDER, 32);
    int rc_order = secp256k1_ecdh(ctx, output, &peer_pub, sk_order,
                                   ecdh_hashfp_sha256, nullptr);
    CHECK(rc_order == 0, "P1-NEW-001a: ecdh(sk=ORDER) must return 0 (== 0 mod n)");

    // (b) seckey = n + 1 — must be rejected (>= n → strict parse rejects; prior
    //     behavior with from_bytes would silently reduce to scalar 1 and succeed)
    unsigned char sk_n_plus_1[32];
    std::memcpy(sk_n_plus_1, CURVE_ORDER, 32);
    // Add 1 to the last byte (little-endian add on big-endian bytes)
    for (int i = 31; i >= 0; --i) {
        if (++sk_n_plus_1[i] != 0) break;
    }
    int rc_np1 = secp256k1_ecdh(ctx, output, &peer_pub, sk_n_plus_1,
                                  ecdh_hashfp_sha256, nullptr);
    CHECK(rc_np1 == 0, "P1-NEW-001b: ecdh(sk=ORDER+1) must return 0 (>= n)");

    // (c) seckey = 0xff..ff (2^256 - 1) — must be rejected (> n)
    unsigned char sk_all_ff[32];
    std::memset(sk_all_ff, 0xFF, 32);
    int rc_ff = secp256k1_ecdh(ctx, output, &peer_pub, sk_all_ff,
                                 ecdh_hashfp_sha256, nullptr);
    CHECK(rc_ff == 0, "P1-NEW-001c: ecdh(sk=0xff..ff) must return 0 (> n)");

    // (d) seckey = 0 — must be rejected (zero scalar)
    unsigned char sk_zero[32] = {};
    int rc_zero = secp256k1_ecdh(ctx, output, &peer_pub, sk_zero,
                                   ecdh_hashfp_sha256, nullptr);
    CHECK(rc_zero == 0, "P1-NEW-001d: ecdh(sk=0) must return 0");

    // (e) seckey = n - 1 — must SUCCEED (valid scalar)
    unsigned char sk_n_minus_1[32];
    std::memcpy(sk_n_minus_1, CURVE_ORDER, 32);
    // Subtract 1 from last byte
    for (int i = 31; i >= 0; --i) {
        if (sk_n_minus_1[i]-- != 0) break;
    }
    int rc_nm1 = secp256k1_ecdh(ctx, output, &peer_pub, sk_n_minus_1,
                                  ecdh_hashfp_sha256, nullptr);
    CHECK(rc_nm1 == 1, "P1-NEW-001e: ecdh(sk=ORDER-1) must succeed (valid scalar)");

    secp256k1_context_destroy(ctx);
}

// ── P1-SEC-RED-TEAM-008: ECDSA single verify off-curve pubkey ───────────────

static void test_ecdsa_verify_off_curve_pubkey() {
    std::printf("  [ecdsa_verify_off_curve_pubkey] P1-RED-TEAM-008\n");
    secp256k1_context *ctx = secp256k1_context_create(
        SECP256K1_CONTEXT_SIGN | SECP256K1_CONTEXT_VERIFY);

    // Create a valid sig to test with
    unsigned char sk[32] = {0};
    sk[31] = 1;
    unsigned char msg[32] = {0xAB};
    secp256k1_ecdsa_signature sig;
    secp256k1_ecdsa_sign(ctx, &sig, msg, sk, nullptr, nullptr);

    // Craft a pubkey struct with off-curve coordinates by writing directly
    // to the opaque data field (bypassing ec_pubkey_parse).
    // Use X=1, Y=1 which is not on secp256k1 (would need y²=x³+7=8, y=√8 mod p, not integer)
    secp256k1_pubkey off_curve_pub;
    std::memset(&off_curve_pub, 0, sizeof(off_curve_pub));
    off_curve_pub.data[31] = 1;  // X = 1
    off_curve_pub.data[63] = 1;  // Y = 1  (1 ≠ 8 mod p — off curve)

    int rc = secp256k1_ecdsa_verify(ctx, &sig, msg, &off_curve_pub);
    CHECK(rc == 0,
          "P1-RED-TEAM-008: ecdsa_verify with off-curve pubkey struct must return 0");

    // Confirm that a legitimately parsed pubkey still works
    secp256k1_pubkey valid_pub;
    secp256k1_ec_pubkey_create(ctx, &valid_pub, sk);
    int rc_valid = secp256k1_ecdsa_verify(ctx, &sig, msg, &valid_pub);
    CHECK(rc_valid == 1, "P1-RED-TEAM-008: ecdsa_verify with valid pubkey must succeed");

    // Consistency: off-curve pub also rejected by batch verify
    const secp256k1_ecdsa_signature* sigs[1]   = {&sig};
    const unsigned char* msgs[1]               = {msg};
    const secp256k1_pubkey* pubs_off[1]        = {&off_curve_pub};
    int rc_batch_off = secp256k1_ecdsa_verify_batch(ctx, sigs, msgs, pubs_off, 1);
    CHECK(rc_batch_off == 0,
          "P1-RED-TEAM-008: ecdsa_verify_batch also rejects off-curve pubkey (consistency)");

    secp256k1_context_destroy(ctx);
}

// ── P2-SEC-NEW-002: ECDH off-curve pubkey (invalid-curve attack) ─────────────

static void test_ecdh_pubkey_off_curve() {
    std::printf("  [ecdh_pubkey_off_curve] P2-SEC-NEW-002\n");
    secp256k1_context *ctx = secp256k1_context_create(
        SECP256K1_CONTEXT_SIGN | SECP256K1_CONTEXT_VERIFY);

    unsigned char sk[32] = {0};
    sk[31] = 2;  // seckey = 2 (valid)

    unsigned char output[32];

    // Off-curve pubkey: X=1, Y=1 (not on secp256k1)
    secp256k1_pubkey off_curve_pub;
    std::memset(&off_curve_pub, 0, sizeof(off_curve_pub));
    off_curve_pub.data[31] = 1;   // X = 1
    off_curve_pub.data[63] = 1;   // Y = 1

    int rc = secp256k1_ecdh(ctx, output, &off_curve_pub, sk,
                              ecdh_hashfp_sha256, nullptr);
    CHECK(rc == 0,
          "P2-SEC-NEW-002: ecdh with off-curve pubkey must return 0 (invalid-curve guard)");

    // Valid pubkey should still work
    secp256k1_pubkey valid_pub;
    secp256k1_ec_pubkey_create(ctx, &valid_pub, sk);
    unsigned char sk2[32] = {0};
    sk2[31] = 3;
    int rc_valid = secp256k1_ecdh(ctx, output, &valid_pub, sk2,
                                    ecdh_hashfp_sha256, nullptr);
    CHECK(rc_valid == 1,
          "P2-SEC-NEW-002: ecdh with valid pubkey and valid sk must succeed");

    secp256k1_context_destroy(ctx);
}

} // namespace

int test_regression_shim_security_v8_run() {
    g_pass = 0; g_fail = 0;
    std::printf("[regression_shim_security_v8] P1-SEC-NEW-001 / RED-TEAM-008 / P2-SEC-NEW-002\n");

    test_ecdh_privkey_out_of_range();
    test_ecdsa_verify_off_curve_pubkey();
    test_ecdh_pubkey_off_curve();

    std::printf("  pass=%d  fail=%d\n", g_pass, g_fail);
    return (g_fail == 0) ? 0 : 1;
}

#ifdef STANDALONE_TEST
int main() { return test_regression_shim_security_v8_run(); }
#endif
