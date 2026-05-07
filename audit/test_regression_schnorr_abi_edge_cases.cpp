// ============================================================================
// Regression: Schnorr BIP-340 ABI Edge-Case Rejection (TQ-005)
// ============================================================================
// Verifies that secp256k1_schnorrsig_verify (libsecp shim) correctly rejects
// non-canonical Schnorr signatures per BIP-340 spec:
//
//   r = 0               → reject (r must be in [1, p-1])
//   r = field modulus p → reject (r >= p)
//   s = 0               → reject (s must be in [1, n-1])
//   s = group order n   → reject (s >= n)
//   r,s both canonical but sig forged → reject (verify fails)
//   Valid signature      → accept
//
// Also verifies secp256k1_schnorrsig_sign32 never emits r==0 or s==0.
// ============================================================================

#ifndef UNIFIED_AUDIT_RUNNER
#include <cstdio>
#define STANDALONE_TEST
#endif

#include <cstring>
#include <cstdint>
#include <array>

#if !__has_include("secp256k1_schnorrsig.h")
// Shim headers not in include path (e.g. GPU-only builds) — stub out the test
int test_regression_schnorr_abi_edge_cases_run() { return 0; }
#ifdef STANDALONE_TEST
int main() { return 0; }
#endif
#else  // have shim headers

#include "secp256k1.h"
#include "secp256k1_schnorrsig.h"
#include "secp256k1_extrakeys.h"

static int g_pass = 0, g_fail = 0;

#define CHECK(cond, msg) do { \
    if (cond) { ++g_pass; } \
    else { ++g_fail; printf("  FAIL: %s\n", (msg)); } \
} while(0)

// secp256k1 group order n (big-endian)
static const unsigned char kGroupOrder[32] = {
    0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,
    0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFE,
    0xBA,0xAE,0xDC,0xE6,0xAF,0x48,0xA0,0x3B,
    0xBF,0xD2,0x5E,0x8C,0xD0,0x36,0x41,0x41
};

// secp256k1 field modulus p (big-endian)
static const unsigned char kFieldModulus[32] = {
    0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,
    0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,
    0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,
    0xFF,0xFF,0xFF,0xFE,0xFF,0xFF,0xFC,0x2F
};

// A valid compressed private key (value = 2)
static const unsigned char kPrivKey[32] = {
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2
};

int test_regression_schnorr_abi_edge_cases_run() {
    printf("[schnorr_abi_edge_cases] TQ-005: r==0, r>=p, s==0, s>=n rejection\n");
    g_pass = g_fail = 0;

    secp256k1_context* ctx = secp256k1_context_create(SECP256K1_CONTEXT_SIGN | SECP256K1_CONTEXT_VERIFY);
    if (!ctx) { printf("  FATAL: context_create failed\n"); return 1; }

    // Build a valid keypair
    secp256k1_keypair keypair{};
    if (!secp256k1_keypair_create(ctx, &keypair, kPrivKey)) {
        printf("  FATAL: keypair_create failed\n");
        secp256k1_context_destroy(ctx);
        return 1;
    }

    secp256k1_xonly_pubkey xonly_pk{};
    int parity = 0;
    secp256k1_keypair_xonly_pub(ctx, &xonly_pk, &parity, &keypair);

    unsigned char msg32[32]{};
    msg32[0] = 0x42;

    // Produce a valid signature for baseline
    unsigned char valid_sig[64]{};
    unsigned char aux[32]{};
    if (!secp256k1_schnorrsig_sign32(ctx, valid_sig, msg32, &keypair, aux)) {
        printf("  FATAL: sign32 failed\n");
        secp256k1_context_destroy(ctx);
        return 1;
    }

    // --- Valid signature: must verify ---
    CHECK(secp256k1_schnorrsig_verify(ctx, valid_sig, msg32, 32, &xonly_pk) == 1,
          "valid signature accepted");

    // --- r = 0 (first 32 bytes = 0) ---
    {
        unsigned char bad[64]{};
        std::memcpy(bad + 32, valid_sig + 32, 32);  // copy valid s
        // bad[0..31] = 0 (r=0)
        CHECK(secp256k1_schnorrsig_verify(ctx, bad, msg32, 32, &xonly_pk) == 0,
              "r=0 rejected");
    }

    // --- r = p (field modulus, out of range) ---
    {
        unsigned char bad[64]{};
        std::memcpy(bad, kFieldModulus, 32);  // r = p >= p → invalid
        std::memcpy(bad + 32, valid_sig + 32, 32);
        CHECK(secp256k1_schnorrsig_verify(ctx, bad, msg32, 32, &xonly_pk) == 0,
              "r=p rejected");
    }

    // --- s = 0 (last 32 bytes = 0) ---
    {
        unsigned char bad[64]{};
        std::memcpy(bad, valid_sig, 32);  // copy valid r
        // bad[32..63] = 0 (s=0)
        CHECK(secp256k1_schnorrsig_verify(ctx, bad, msg32, 32, &xonly_pk) == 0,
              "s=0 rejected");
    }

    // --- s = n (group order, out of range) ---
    {
        unsigned char bad[64]{};
        std::memcpy(bad, valid_sig, 32);  // copy valid r
        std::memcpy(bad + 32, kGroupOrder, 32);  // s = n >= n → invalid
        CHECK(secp256k1_schnorrsig_verify(ctx, bad, msg32, 32, &xonly_pk) == 0,
              "s=n rejected");
    }

    // --- r,s canonical but signature for wrong message → fails verify ---
    {
        unsigned char bad_msg[32]{};
        bad_msg[0] = 0xFF;  // different from msg32
        CHECK(secp256k1_schnorrsig_verify(ctx, valid_sig, bad_msg, 32, &xonly_pk) == 0,
              "wrong message rejected");
    }

    // --- NULL sig64 → returns 0 (illegal callback fires) ---
    CHECK(secp256k1_schnorrsig_verify(ctx, nullptr, msg32, 32, &xonly_pk) == 0,
          "NULL sig64 rejected");

    // --- NULL pubkey → returns 0 ---
    CHECK(secp256k1_schnorrsig_verify(ctx, valid_sig, msg32, 32, nullptr) == 0,
          "NULL pubkey rejected");

    // --- msglen != 32 → returns 0 ---
    CHECK(secp256k1_schnorrsig_verify(ctx, valid_sig, msg32, 31, &xonly_pk) == 0,
          "msglen=31 rejected");
    CHECK(secp256k1_schnorrsig_verify(ctx, valid_sig, msg32, 33, &xonly_pk) == 0,
          "msglen=33 rejected");

    // --- sign32 never emits r=0 or s=0 (sign 50 times with different keys) ---
    {
        int zero_r = 0, zero_s = 0;
        unsigned char sk[32] = {1};
        for (int i = 0; i < 50; ++i) {
            sk[31] = (unsigned char)(i + 1);
            secp256k1_keypair kp{};
            if (!secp256k1_keypair_create(ctx, &kp, sk)) continue;
            unsigned char sig[64]{};
            unsigned char a[32]{};
            a[0] = (unsigned char)i;
            if (!secp256k1_schnorrsig_sign32(ctx, sig, msg32, &kp, a)) continue;
            bool r_zero = true, s_zero = true;
            for (int b = 0; b < 32; ++b) { if (sig[b])    r_zero = false; }
            for (int b = 0; b < 32; ++b) { if (sig[32+b]) s_zero = false; }
            if (r_zero) ++zero_r;
            if (s_zero) ++zero_s;
        }
        CHECK(zero_r == 0, "sign32 never emits r=0");
        CHECK(zero_s == 0, "sign32 never emits s=0");
    }

    secp256k1_context_destroy(ctx);

    printf("  pass=%d  fail=%d\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}

#ifdef STANDALONE_TEST
int main() { return test_regression_schnorr_abi_edge_cases_run(); }
#endif
#endif  // have shim headers
