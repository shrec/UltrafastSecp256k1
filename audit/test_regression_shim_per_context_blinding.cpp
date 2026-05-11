// ============================================================================
// test_regression_shim_per_context_blinding.cpp
// Regression: secp256k1_context_randomize must provide per-context semantics.
//
// Bug fixed: blinding was thread-local state applied at context_randomize time.
// Two contexts on the same thread would overwrite each other's blinding.
// Fix: ContextBlindingScope — blinding applied per signing call, per context.
//
// Tests:
//   PCB-1: context_randomize returns 1 (basic success)
//   PCB-2: two contexts can be independently randomized and both sign correctly
//   PCB-3: unrandomized context signs correctly (no blinding is not a failure)
//   PCB-4: context_randomize with NULL seed clears blinding state
//   PCB-5: signing after randomize produces valid verifiable signatures
// ============================================================================

#include "audit_check.hpp"
#include <cstdio>
#include <cstring>
#include "secp256k1.h"

static const unsigned char kSk[32] = {
    0xAA,0xBB,0xCC,0xDD,0x11,0x22,0x33,0x44,
    0x55,0x66,0x77,0x88,0x99,0x00,0xAA,0xBB,
    0xCC,0xDD,0xEE,0xFF,0x01,0x02,0x03,0x04,
    0x05,0x06,0x07,0x08,0x09,0x0A,0x0B,0x0C
};
static const unsigned char kMsg[32] = {
    0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08,
    0x09,0x0A,0x0B,0x0C,0x0D,0x0E,0x0F,0x10,
    0x11,0x12,0x13,0x14,0x15,0x16,0x17,0x18,
    0x19,0x1A,0x1B,0x1C,0x1D,0x1E,0x1F,0x20
};
static const unsigned char kSeed1[32] = {0x11,0x22,0x33,0x44,0x55,0x66,0x77,0x88,0x99,0xAA,0xBB,0xCC,0xDD,0xEE,0xFF,0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08,0x09,0x0A,0x0B,0x0C,0x0D,0x0E,0x0F,0x10};
static const unsigned char kSeed2[32] = {0x99,0x88,0x77,0x66,0x55,0x44,0x33,0x22,0x11,0x00,0xFF,0xEE,0xDD,0xCC,0xBB,0xAA,0x10,0x0F,0x0E,0x0D,0x0C,0x0B,0x0A,0x09,0x08,0x07,0x06,0x05,0x04,0x03,0x02,0x01};

// ── PCB-1: context_randomize returns 1 ────────────────────────────────────
static void test_randomize_returns_one() {
    secp256k1_context* ctx = secp256k1_context_create(SECP256K1_CONTEXT_SIGN);
    int r = secp256k1_context_randomize(ctx, kSeed1);
    check(r == 1, "[PCB-1] context_randomize returns 1");
    secp256k1_context_destroy(ctx);
}

// ── PCB-2: two contexts independently randomized, both sign correctly ──────
static void test_two_contexts_independent() {
    secp256k1_context* ctx_a = secp256k1_context_create(SECP256K1_CONTEXT_SIGN | SECP256K1_CONTEXT_VERIFY);
    secp256k1_context* ctx_b = secp256k1_context_create(SECP256K1_CONTEXT_SIGN | SECP256K1_CONTEXT_VERIFY);

    check(secp256k1_context_randomize(ctx_a, kSeed1) == 1, "[PCB-2a] ctx_a randomized");
    check(secp256k1_context_randomize(ctx_b, kSeed2) == 1, "[PCB-2b] ctx_b randomized");

    secp256k1_ecdsa_signature sig_a, sig_b;
    check(secp256k1_ecdsa_sign(ctx_a, &sig_a, kMsg, kSk, nullptr, nullptr) == 1,
          "[PCB-2c] ctx_a signs successfully after ctx_b randomized on same thread");
    check(secp256k1_ecdsa_sign(ctx_b, &sig_b, kMsg, kSk, nullptr, nullptr) == 1,
          "[PCB-2d] ctx_b signs successfully");

    secp256k1_pubkey pk;
    check(secp256k1_ec_pubkey_create(ctx_a, &pk, kSk) == 1, "[PCB-2e] pubkey_create");
    check(secp256k1_ecdsa_verify(ctx_a, &sig_a, kMsg, &pk) == 1,
          "[PCB-2f] ctx_a signature verifies — blinding did not corrupt output");
    check(secp256k1_ecdsa_verify(ctx_b, &sig_b, kMsg, &pk) == 1,
          "[PCB-2g] ctx_b signature verifies — blinding did not corrupt output");

    secp256k1_context_destroy(ctx_a);
    secp256k1_context_destroy(ctx_b);
}

// ── PCB-3: unrandomized context signs correctly ───────────────────────────
static void test_unrandomized_context() {
    secp256k1_context* ctx = secp256k1_context_create(SECP256K1_CONTEXT_SIGN | SECP256K1_CONTEXT_VERIFY);
    // No randomize call
    secp256k1_ecdsa_signature sig;
    check(secp256k1_ecdsa_sign(ctx, &sig, kMsg, kSk, nullptr, nullptr) == 1,
          "[PCB-3a] signs without randomize");
    secp256k1_pubkey pk;
    secp256k1_ec_pubkey_create(ctx, &pk, kSk);
    check(secp256k1_ecdsa_verify(ctx, &sig, kMsg, &pk) == 1,
          "[PCB-3b] signature verifies (no blinding)");
    secp256k1_context_destroy(ctx);
}

// ── PCB-4: NULL seed clears blinding ──────────────────────────────────────
static void test_clear_blinding() {
    secp256k1_context* ctx = secp256k1_context_create(SECP256K1_CONTEXT_SIGN | SECP256K1_CONTEXT_VERIFY);
    check(secp256k1_context_randomize(ctx, kSeed1) == 1, "[PCB-4a] randomize");
    check(secp256k1_context_randomize(ctx, nullptr) == 1, "[PCB-4b] clear blinding (NULL seed)");
    // After clearing, signing must still work
    secp256k1_ecdsa_signature sig;
    check(secp256k1_ecdsa_sign(ctx, &sig, kMsg, kSk, nullptr, nullptr) == 1,
          "[PCB-4c] signs after clearing blinding");
    secp256k1_context_destroy(ctx);
}

// ── PCB-5: signing after randomize produces valid signatures ──────────────
static void test_sign_after_randomize() {
    secp256k1_context* ctx = secp256k1_context_create(SECP256K1_CONTEXT_SIGN | SECP256K1_CONTEXT_VERIFY);
    check(secp256k1_context_randomize(ctx, kSeed1) == 1, "[PCB-5a] randomize");

    secp256k1_pubkey pk;
    check(secp256k1_ec_pubkey_create(ctx, &pk, kSk) == 1, "[PCB-5b] pubkey");

    secp256k1_ecdsa_signature sig;
    check(secp256k1_ecdsa_sign(ctx, &sig, kMsg, kSk, nullptr, nullptr) == 1, "[PCB-5c] sign");
    check(secp256k1_ecdsa_verify(ctx, &sig, kMsg, &pk) == 1, "[PCB-5d] verify");
    secp256k1_context_destroy(ctx);
}

int test_regression_shim_per_context_blinding_run() {
    g_pass = 0; g_fail = 0;
    std::printf("[regression_shim_per_context_blinding] per-context blinding semantics\n");
    test_randomize_returns_one();
    test_two_contexts_independent();
    test_unrandomized_context();
    test_clear_blinding();
    test_sign_after_randomize();
    std::printf("  pass=%d  fail=%d\n", g_pass, g_fail);
    return (g_fail == 0) ? 0 : 1;
}

#ifdef STANDALONE_TEST
int main() { return test_regression_shim_per_context_blinding_run(); }
#endif
