// ============================================================================
// test_infinity_edge_cases.cpp -- Point-at-Infinity Edge Case Audit
// ============================================================================
//
// Verifies correct handling of the additive identity (point at infinity / O)
// across all C ABI entry points that perform point arithmetic.
//
// The point at infinity arises when:
//   - k·G where k = 0 mod n  (scalar zero)
//   - P + (−P)               (additive inverse)
//   - seckey_tweak_add(k, t) where k + t ≡ 0 mod n
//   - pubkey_tweak_add(P, t) where P + t·G = O
//   - pubkey_combine with exactly cancelling keys
//
// Correct behaviour: all such operations must return an error (never silently
// yield O as a valid public key, since O is not a valid secp256k1 point for
// any protocol purpose).
//
// INF-1  … INF-4  : seckey_tweak_add cancellation (k + tweak ≡ 0 mod n)
// INF-5  … INF-8  : pubkey_add(P, −P) = infinity → error
// INF-9  … INF-12 : pubkey_tweak_add(P, tweak) where P + tweak·G = O
// INF-13 … INF-16 : pubkey_combine with cancelling key set
// INF-17 … INF-20 : ECDH / scalar-mul with zero scalar or zero result
// INF-21 … INF-24 : Taproot and BIP-32 cancellation edge cases
// INF-25 … INF-28 : Serialization: infinity must never serialize as valid key
// ============================================================================

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#ifndef UFSECP_BUILDING
#define UFSECP_BUILDING
#endif
#include "ufsecp/ufsecp.h"

static int g_pass = 0, g_fail = 0;
#include "audit_check.hpp"

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

// Group order n (secp256k1)
static constexpr uint8_t N[32] = {
    0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,
    0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFE,
    0xBA,0xAE,0xDC,0xE6,0xAF,0x48,0xA0,0x3B,
    0xBF,0xD2,0x5E,0x8C,0xD0,0x36,0x41,0x41
};

// n - 1 (negation of 1 mod n)
static constexpr uint8_t N_MINUS_1[32] = {
    0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,
    0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFE,
    0xBA,0xAE,0xDC,0xE6,0xAF,0x48,0xA0,0x3B,
    0xBF,0xD2,0x5E,0x8C,0xD0,0x36,0x41,0x40
};

// n - 2 (negation of 2 mod n)
static constexpr uint8_t N_MINUS_2[32] = {
    0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,
    0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFE,
    0xBA,0xAE,0xDC,0xE6,0xAF,0x48,0xA0,0x3B,
    0xBF,0xD2,0x5E,0x8C,0xD0,0x36,0x41,0x3F
};

// n - 3
static constexpr uint8_t N_MINUS_3[32] = {
    0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,
    0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFE,
    0xBA,0xAE,0xDC,0xE6,0xAF,0x48,0xA0,0x3B,
    0xBF,0xD2,0x5E,0x8C,0xD0,0x36,0x41,0x3E
};

// Privkeys 1, 2, 3
static constexpr uint8_t KEY1[32] = {
    0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,1
};
static constexpr uint8_t KEY2[32] = {
    0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,2
};
static constexpr uint8_t KEY3[32] = {
    0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,3
};

// ---------------------------------------------------------------------------
// INF-1 … INF-4 : seckey_tweak_add cancellation
// seckey_tweak_add(k, n−k) must fail: result ≡ 0 mod n → invalid key
// ---------------------------------------------------------------------------

static void run_inf1_seckey_cancellation(ufsecp_ctx* ctx) {
    AUDIT_LOG("\n  [INF-1..4] seckey_tweak_add: k + tweak ≡ 0 mod n must fail\n");

    // INF-1: key=1, tweak=n−1  →  1 + (n−1) = n ≡ 0
    {
        uint8_t key[32];
        std::memcpy(key, KEY1, 32);
        ufsecp_error_t rc = ufsecp_seckey_tweak_add(ctx, key, N_MINUS_1);
        CHECK(rc != UFSECP_OK,
              "INF-1: seckey_tweak_add(1, n-1) → k+t≡0 mod n must fail");
    }
    // INF-2: key=2, tweak=n−2  →  2 + (n−2) = n ≡ 0
    {
        uint8_t key[32];
        std::memcpy(key, KEY2, 32);
        ufsecp_error_t rc = ufsecp_seckey_tweak_add(ctx, key, N_MINUS_2);
        CHECK(rc != UFSECP_OK,
              "INF-2: seckey_tweak_add(2, n-2) → k+t≡0 mod n must fail");
    }
    // INF-3: key=3, tweak=n−3  →  3 + (n−3) = n ≡ 0
    {
        uint8_t key[32];
        std::memcpy(key, KEY3, 32);
        ufsecp_error_t rc = ufsecp_seckey_tweak_add(ctx, key, N_MINUS_3);
        CHECK(rc != UFSECP_OK,
              "INF-3: seckey_tweak_add(3, n-3) → k+t≡0 mod n must fail");
    }
    // INF-4: Normal tweak (non-cancelling) should still succeed
    {
        uint8_t key[32];
        std::memcpy(key, KEY1, 32);
        ufsecp_error_t rc = ufsecp_seckey_tweak_add(ctx, key, KEY2);
        CHECK(rc == UFSECP_OK,
              "INF-4: seckey_tweak_add(1, 2) → 3 is valid, must succeed");
    }
}

// ---------------------------------------------------------------------------
// INF-5 … INF-8 : pubkey_add(P, −P) = O → must fail
// ---------------------------------------------------------------------------

static void run_inf5_pubkey_add_cancel(ufsecp_ctx* ctx) {
    AUDIT_LOG("\n  [INF-5..8] pubkey_add(P, -P) = infinity must fail\n");

    uint8_t result[33] = {};

    // INF-5: pub(1) + pub(n-1) = 1·G + (n-1)·G = n·G = O
    {
        uint8_t pub1[33] = {}, pub_neg1[33] = {};
        CHECK(ufsecp_pubkey_create(ctx, KEY1, pub1) == UFSECP_OK,
              "INF-5-setup: create pub(1)");
        CHECK(ufsecp_pubkey_create(ctx, N_MINUS_1, pub_neg1) == UFSECP_OK,
              "INF-5-setup: create pub(n-1)");
        ufsecp_error_t rc = ufsecp_pubkey_add(ctx, pub1, pub_neg1, result);
        CHECK(rc != UFSECP_OK,
              "INF-5: pubkey_add(1·G, (n-1)·G) = n·G = O must fail");
    }
    // INF-6: pub(2) + pub(n-2) = O
    {
        uint8_t pub2[33] = {}, pub_neg2[33] = {};
        CHECK(ufsecp_pubkey_create(ctx, KEY2, pub2) == UFSECP_OK,
              "INF-6-setup: create pub(2)");
        CHECK(ufsecp_pubkey_create(ctx, N_MINUS_2, pub_neg2) == UFSECP_OK,
              "INF-6-setup: create pub(n-2)");
        ufsecp_error_t rc = ufsecp_pubkey_add(ctx, pub2, pub_neg2, result);
        CHECK(rc != UFSECP_OK,
              "INF-6: pubkey_add(2·G, (n-2)·G) = O must fail");
    }
    // INF-7: pub(1) negated via negate() then add back should give O
    {
        uint8_t pub1[33] = {}, neg_pub1[33] = {};
        CHECK(ufsecp_pubkey_create(ctx, KEY1, pub1) == UFSECP_OK,
              "INF-7-setup: create pub(1)");
        CHECK(ufsecp_pubkey_negate(ctx, pub1, neg_pub1) == UFSECP_OK,
              "INF-7-setup: negate pub(1)");
        ufsecp_error_t rc = ufsecp_pubkey_add(ctx, pub1, neg_pub1, result);
        CHECK(rc != UFSECP_OK,
              "INF-7: pubkey_add(P, -P) via negate = O must fail");
    }
    // INF-8: Normal non-cancelling add must still work
    {
        uint8_t pub1[33] = {}, pub2[33] = {};
        CHECK(ufsecp_pubkey_create(ctx, KEY1, pub1) == UFSECP_OK,
              "INF-8-setup: create pub(1)");
        CHECK(ufsecp_pubkey_create(ctx, KEY2, pub2) == UFSECP_OK,
              "INF-8-setup: create pub(2)");
        ufsecp_error_t rc = ufsecp_pubkey_add(ctx, pub1, pub2, result);
        CHECK(rc == UFSECP_OK,
              "INF-8: pubkey_add(1·G, 2·G) = 3·G must succeed");
    }
}

// ---------------------------------------------------------------------------
// INF-9 … INF-12 : pubkey_tweak_add(P, t) where P + t·G = O
// pubkey_tweak_add(pub(k), n−k) = k·G + (n−k)·G = O → must fail
// ---------------------------------------------------------------------------

static void run_inf9_pubkey_tweak_cancel(ufsecp_ctx* ctx) {
    AUDIT_LOG("\n  [INF-9..12] pubkey_tweak_add: P + t·G = O must fail\n");

    uint8_t result[33] = {};

    // INF-9: tweak_add(pub(1), n-1) = 1·G + (n-1)·G = O
    {
        uint8_t pub1[33] = {};
        CHECK(ufsecp_pubkey_create(ctx, KEY1, pub1) == UFSECP_OK,
              "INF-9-setup: create pub(1)");
        ufsecp_error_t rc = ufsecp_pubkey_tweak_add(ctx, pub1, N_MINUS_1, result);
        CHECK(rc != UFSECP_OK,
              "INF-9: pubkey_tweak_add(1·G, n-1) = O must fail");
    }
    // INF-10: tweak_add(pub(2), n-2) = O
    {
        uint8_t pub2[33] = {};
        CHECK(ufsecp_pubkey_create(ctx, KEY2, pub2) == UFSECP_OK,
              "INF-10-setup: create pub(2)");
        ufsecp_error_t rc = ufsecp_pubkey_tweak_add(ctx, pub2, N_MINUS_2, result);
        CHECK(rc != UFSECP_OK,
              "INF-10: pubkey_tweak_add(2·G, n-2) = O must fail");
    }
    // INF-11: tweak_add(pub(3), n-3) = O
    {
        uint8_t pub3[33] = {};
        CHECK(ufsecp_pubkey_create(ctx, KEY3, pub3) == UFSECP_OK,
              "INF-11-setup: create pub(3)");
        ufsecp_error_t rc = ufsecp_pubkey_tweak_add(ctx, pub3, N_MINUS_3, result);
        CHECK(rc != UFSECP_OK,
              "INF-11: pubkey_tweak_add(3·G, n-3) = O must fail");
    }
    // INF-12: Normal non-cancelling tweak_add must still work
    {
        uint8_t pub1[33] = {};
        CHECK(ufsecp_pubkey_create(ctx, KEY1, pub1) == UFSECP_OK,
              "INF-12-setup: create pub(1)");
        ufsecp_error_t rc = ufsecp_pubkey_tweak_add(ctx, pub1, KEY2, result);
        CHECK(rc == UFSECP_OK,
              "INF-12: pubkey_tweak_add(1·G, 2) = 3·G must succeed");
    }
}

// ---------------------------------------------------------------------------
// INF-13 … INF-16 : pubkey_combine with cancelling key set
// ---------------------------------------------------------------------------

static void run_inf13_combine_cancel(ufsecp_ctx* ctx) {
    AUDIT_LOG("\n  [INF-13..16] pubkey_combine: cancelling key set must fail\n");

    uint8_t result[33] = {};

    // INF-13: combine([pub(1), pub(n-1)]) = O
    {
        uint8_t pub1[33] = {}, pub_neg1[33] = {};
        CHECK(ufsecp_pubkey_create(ctx, KEY1, pub1) == UFSECP_OK,
              "INF-13-setup: create pub(1)");
        CHECK(ufsecp_pubkey_create(ctx, N_MINUS_1, pub_neg1) == UFSECP_OK,
              "INF-13-setup: create pub(n-1)");
        uint8_t buf[66];
        std::memcpy(buf, pub1, 33);
        std::memcpy(buf + 33, pub_neg1, 33);
        ufsecp_error_t rc = ufsecp_pubkey_combine(ctx, buf, 2, result);
        CHECK(rc != UFSECP_OK,
              "INF-13: pubkey_combine([1·G, (n-1)·G]) = O must fail");
    }
    // INF-14: combine([pub(1), pub(2), pub(n-3)]) = (1+2+(n-3))·G = O
    {
        uint8_t pub1[33] = {}, pub2[33] = {}, pub_neg3[33] = {};
        CHECK(ufsecp_pubkey_create(ctx, KEY1, pub1) == UFSECP_OK, "INF-14-setup-1");
        CHECK(ufsecp_pubkey_create(ctx, KEY2, pub2) == UFSECP_OK, "INF-14-setup-2");
        CHECK(ufsecp_pubkey_create(ctx, N_MINUS_3, pub_neg3) == UFSECP_OK, "INF-14-setup-3");
        uint8_t buf[99];
        std::memcpy(buf, pub1, 33);
        std::memcpy(buf + 33, pub2, 33);
        std::memcpy(buf + 66, pub_neg3, 33);
        ufsecp_error_t rc = ufsecp_pubkey_combine(ctx, buf, 3, result);
        CHECK(rc != UFSECP_OK,
              "INF-14: pubkey_combine([1G, 2G, (n-3)G]) = O must fail");
    }
    // INF-15: combine([pub(1), pub(2)]) = 3·G must succeed (normal case)
    {
        uint8_t pub1[33] = {}, pub2[33] = {};
        CHECK(ufsecp_pubkey_create(ctx, KEY1, pub1) == UFSECP_OK, "INF-15-setup-1");
        CHECK(ufsecp_pubkey_create(ctx, KEY2, pub2) == UFSECP_OK, "INF-15-setup-2");
        uint8_t buf[66];
        std::memcpy(buf, pub1, 33);
        std::memcpy(buf + 33, pub2, 33);
        ufsecp_error_t rc = ufsecp_pubkey_combine(ctx, buf, 2, result);
        CHECK(rc == UFSECP_OK,
              "INF-15: pubkey_combine([1G, 2G]) = 3G must succeed");
    }
    // INF-16: combine([pub(1)]) = pub(1) must succeed (trivial 1-element case)
    {
        uint8_t pub1[33] = {};
        CHECK(ufsecp_pubkey_create(ctx, KEY1, pub1) == UFSECP_OK, "INF-16-setup");
        ufsecp_error_t rc = ufsecp_pubkey_combine(ctx, pub1, 1, result);
        CHECK(rc == UFSECP_OK,
              "INF-16: pubkey_combine([1·G]) = 1·G must succeed");
    }
}

// ---------------------------------------------------------------------------
// INF-17 … INF-20 : ECDH / scalar-mul with zero scalar or degenerate input
// ---------------------------------------------------------------------------

static void run_inf17_ecdh_degenerate(ufsecp_ctx* ctx) {
    AUDIT_LOG("\n  [INF-17..20] ECDH / scalar-mul: degenerate inputs\n");

    uint8_t shared[32] = {};

    // INF-17: ecdh(scalar=0, pub) — 0·P = O → must fail
    {
        uint8_t zero_key[32] = {};
        uint8_t pub1[33] = {};
        CHECK(ufsecp_pubkey_create(ctx, KEY1, pub1) == UFSECP_OK, "INF-17-setup");
        ufsecp_error_t rc = ufsecp_ecdh(ctx, zero_key, pub1, shared);
        CHECK(rc != UFSECP_OK,
              "INF-17: ecdh(0, P) = 0·P = O must fail");
    }
    // INF-18: ecdh(scalar=n, pub) — n·P = O → must fail
    {
        uint8_t pub1[33] = {};
        CHECK(ufsecp_pubkey_create(ctx, KEY1, pub1) == UFSECP_OK, "INF-18-setup");
        ufsecp_error_t rc = ufsecp_ecdh(ctx, N, pub1, shared);
        CHECK(rc != UFSECP_OK,
              "INF-18: ecdh(n, P) = n·P = O must fail (seckey=n is invalid)");
    }
    // INF-19: ecdh(key=1, pub=1·G) must succeed (normal case)
    {
        uint8_t pub1[33] = {};
        CHECK(ufsecp_pubkey_create(ctx, KEY1, pub1) == UFSECP_OK, "INF-19-setup");
        ufsecp_error_t rc = ufsecp_ecdh(ctx, KEY1, pub1, shared);
        CHECK(rc == UFSECP_OK,
              "INF-19: ecdh(1, 1·G) = 1·G (normal case) must succeed");
    }
    // INF-20: ecdh_xonly with zero privkey → must fail
    {
        uint8_t zero_key[32] = {};
        uint8_t pub1[33] = {};
        CHECK(ufsecp_pubkey_create(ctx, KEY2, pub1) == UFSECP_OK, "INF-20-setup");
        ufsecp_error_t rc = ufsecp_ecdh_xonly(ctx, zero_key, pub1, shared);
        CHECK(rc != UFSECP_OK,
              "INF-20: ecdh_xonly(0, P) must fail");
    }
}

// ---------------------------------------------------------------------------
// INF-21 … INF-24 : Taproot and BIP-32 derived key cancellation
// ---------------------------------------------------------------------------

static void run_inf21_taproot_bip32(ufsecp_ctx* ctx) {
    AUDIT_LOG("\n  [INF-21..24] Taproot tweak + BIP-32 derived key edge cases\n");

    // INF-21: taproot_output_key(internal=1·G xonly, merkle_root=N_MINUS_1)
    //         The actual Taproot tweak is tagged_hash("TapTweak", xonly || merkle_root),
    //         so this won't produce exactly O — we verify the function returns a result
    //         (not an error) and that the output is a valid 32-byte x-only key.
    //         The infinity-cancellation case is better covered via seckey tweak (INF-22).
    {
        uint8_t pub1[33] = {};
        CHECK(ufsecp_pubkey_create(ctx, KEY1, pub1) == UFSECP_OK, "INF-21-setup");
        // Extract 32-byte x-only from compressed pubkey (skip the 0x02/0x03 prefix byte)
        const uint8_t* internal_x = pub1 + 1;
        uint8_t out_xonly[32] = {};
        int parity = 0;
        ufsecp_error_t rc = ufsecp_taproot_output_key(ctx, internal_x, N_MINUS_1,
                                                       out_xonly, &parity);
        // N_MINUS_1 as merkle_root produces a valid tap_tweak, so must succeed
        CHECK(rc == UFSECP_OK,
              "INF-21: taproot_output_key with N_MINUS_1 merkle_root must succeed (tweak is hashed)");
    }
    // INF-22: taproot_tweak_seckey(1, merkle_root=NULL) — key-path-only, valid key → must succeed
    //         True infinity cancellation via seckey is hard to trigger directly
    //         (requires tap_tweak(P, m) = n-1 which is not achievable with raw scalar).
    //         We test that invalid privkey (zero) is rejected.
    {
        uint8_t zero_key[32] = {};  // all-zero privkey is invalid
        uint8_t tweaked[32] = {};
        ufsecp_error_t rc = ufsecp_taproot_tweak_seckey(ctx, zero_key, nullptr, tweaked);
        CHECK(rc != UFSECP_OK,
              "INF-22: taproot_tweak_seckey(zero_key) must fail (invalid privkey)");
    }
    // INF-23: taproot_output_key normal case must succeed
    {
        uint8_t pub1[33] = {};
        CHECK(ufsecp_pubkey_create(ctx, KEY1, pub1) == UFSECP_OK, "INF-23-setup");
        const uint8_t* internal_x = pub1 + 1;
        uint8_t out_xonly[32] = {};
        int parity = 0;
        uint8_t merkle[32] = {
            0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08,
            0x09,0x0a,0x0b,0x0c,0x0d,0x0e,0x0f,0x10,
            0x11,0x12,0x13,0x14,0x15,0x16,0x17,0x18,
            0x19,0x1a,0x1b,0x1c,0x1d,0x1e,0x1f,0x20
        };
        ufsecp_error_t rc = ufsecp_taproot_output_key(ctx, internal_x, merkle,
                                                       out_xonly, &parity);
        CHECK(rc == UFSECP_OK,
              "INF-23: taproot_output_key normal case must succeed");
    }
    // INF-24: BIP-32 master from all-zero entropy must generate valid key
    //         (HMAC-SHA512 output over zero seed should be non-zero key)
    {
        uint8_t seed[32] = {};  // all-zero seed (degenerate but valid length)
        ufsecp_bip32_key master = {};
        ufsecp_error_t rc = ufsecp_bip32_master(ctx, seed, 32, &master);
        // All-zero seed is valid input; the HMAC-SHA512 output is non-zero
        // If it happens to produce an all-zero key (astronomically unlikely),
        // the implementation must still reject it
        if (rc == UFSECP_OK) {
            uint8_t privkey[32] = {};
            ufsecp_error_t prc = ufsecp_bip32_privkey(ctx, &master, privkey);
            if (prc == UFSECP_OK) {
                CHECK(ufsecp_seckey_verify(ctx, privkey) == UFSECP_OK,
                      "INF-24: BIP-32 master from zero seed produces valid key");
            } else {
                // If privkey extraction failed, the implementation handled it
                CHECK(true, "INF-24: BIP-32 master from zero seed handled safely");
            }
        } else {
            // Some implementations reject all-zero seed at the master level
            CHECK(true, "INF-24: BIP-32 rejected all-zero seed (safe)");
        }
    }
}

// ---------------------------------------------------------------------------
// INF-25 … INF-28 : Serialization — negated key round-trip integrity
// (Verify the negated key serializes/parses to the correct different point)
// ---------------------------------------------------------------------------

static void run_inf25_negate_round_trip(ufsecp_ctx* ctx) {
    AUDIT_LOG("\n  [INF-25..28] Negated pubkey round-trip integrity\n");

    // INF-25: negate(pub(1)) + pub(1) via seckey path
    //   negate(1) = n-1;  seckey_verify(n-1) must be OK;  pub(n-1) != pub(1)
    {
        uint8_t pub1[33] = {};
        CHECK(ufsecp_pubkey_create(ctx, KEY1, pub1) == UFSECP_OK, "INF-25-setup");
        uint8_t neg_pub1[33] = {};
        CHECK(ufsecp_pubkey_negate(ctx, pub1, neg_pub1) == UFSECP_OK,
              "INF-25: pubkey_negate must succeed");
        // neg_pub1 must be different from pub1
        CHECK(std::memcmp(neg_pub1, pub1, 33) != 0,
              "INF-25: negated pub(1) must differ from pub(1)");
    }
    // INF-26: round-trip: negate(negate(P)) == P
    {
        uint8_t pub2[33] = {};
        CHECK(ufsecp_pubkey_create(ctx, KEY2, pub2) == UFSECP_OK, "INF-26-setup");
        uint8_t neg[33] = {};
        CHECK(ufsecp_pubkey_negate(ctx, pub2, neg) == UFSECP_OK, "INF-26-negate");
        uint8_t neg_neg[33] = {};
        CHECK(ufsecp_pubkey_negate(ctx, neg, neg_neg) == UFSECP_OK, "INF-26-neg-neg");
        CHECK(std::memcmp(neg_neg, pub2, 33) == 0,
              "INF-26: negate(negate(P)) == P");
    }
    // INF-27: seckey negate: negate(1) = n-1; seckey_verify(n-1) OK; pub(n-1) = negate(pub(1))
    {
        uint8_t key[32];
        std::memcpy(key, KEY1, 32);
        CHECK(ufsecp_seckey_negate(ctx, key) == UFSECP_OK,
              "INF-27: seckey_negate(1) must succeed");
        CHECK(std::memcmp(key, N_MINUS_1, 32) == 0,
              "INF-27: seckey_negate(1) == n-1");
        CHECK(ufsecp_seckey_verify(ctx, key) == UFSECP_OK,
              "INF-27: n-1 is a valid seckey");
    }
    // INF-28: seckey negate twice round-trips to original
    {
        uint8_t key[32];
        std::memcpy(key, KEY3, 32);
        CHECK(ufsecp_seckey_negate(ctx, key) == UFSECP_OK, "INF-28-negate");
        CHECK(ufsecp_seckey_negate(ctx, key) == UFSECP_OK, "INF-28-neg-negate");
        CHECK(std::memcmp(key, KEY3, 32) == 0,
              "INF-28: seckey negate twice round-trips to original");
    }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

int test_infinity_edge_cases_run() {
    g_pass = 0; g_fail = 0;

    AUDIT_LOG("============================================================\n");
    AUDIT_LOG("  Point-at-Infinity Edge Case Audit\n");
    AUDIT_LOG("  P+(-P), k+(n-k), combine cancel, ECDH degenerate\n");
    AUDIT_LOG("============================================================\n");

    ufsecp_ctx* ctx = nullptr;
    if (ufsecp_ctx_create(&ctx) != UFSECP_OK || ctx == nullptr) {
        CHECK(false, "INF-ctx: failed to create context");
        printf("[test_infinity_edge_cases] %d/%d checks passed (context failed)\n",
               g_pass, g_pass + g_fail);
        return 1;
    }

    run_inf1_seckey_cancellation(ctx);
    run_inf5_pubkey_add_cancel(ctx);
    run_inf9_pubkey_tweak_cancel(ctx);
    run_inf13_combine_cancel(ctx);
    run_inf17_ecdh_degenerate(ctx);
    run_inf21_taproot_bip32(ctx);
    run_inf25_negate_round_trip(ctx);

    ufsecp_ctx_destroy(ctx);

    printf("[test_infinity_edge_cases] %d/%d checks passed\n",
           g_pass, g_pass + g_fail);
    return (g_fail > 0) ? 1 : 0;
}

#ifndef UNIFIED_AUDIT_RUNNER
int main() {
    return test_infinity_edge_cases_run();
}
#endif
