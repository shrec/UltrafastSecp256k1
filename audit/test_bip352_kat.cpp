/* ============================================================================
 * test_bip352_kat.cpp  --  BIP-352 Silent Payments Known-Answer Tests
 * ============================================================================
 *
 * Known-answer regression tests for the BIP-352 Silent Payments implementation.
 * Vectors were derived from the library itself using fixed small-integer private
 * keys, then hardcoded here.  Any change to the SHA256 tagged-hash ("BIP0352/
 * SharedSecret"), ECDH computation, key serialisation, or output derivation
 * will cause at least one vector to mismatch, failing this test.
 *
 * KAT ID conventions:
 *   BIP352-KAT-1 .. BIP352-KAT-4  -- create_output byte-exact regression
 *   BIP352-KAT-5 .. BIP352-KAT-8  -- full round-trip (create → scan → found)
 *   BIP352-KAT-9 .. BIP352-KAT-12 -- scan rejection (unrelated outputs not found)
 *   BIP352-KAT-13..BIP352-KAT-16  -- output index isolation (k0 ≠ k1 ≠ k2)
 *   BIP352-KAT-17..BIP352-KAT-18  -- two-input aggregation (a_sum = a1 + a2)
 *
 * Algorithm under test  (address.cpp: silent_payment_create_output):
 *   a_sum  = Σ input_privkeys
 *   S      = ct::scalar_mul(B_scan, a_sum)
 *   tag    = SHA256("BIP0352/SharedSecret")   [20-byte string]
 *   t_k    = tagged_SHA256(S_comp || ser32(k))
 *   P_out  = B_spend + t_k * G
 *
 * Compile as standalone:
 *   g++ -std=c++17 -DSTANDALONE_TEST -I../include -I../cpu/include \
 *       test_bip352_kat.cpp -lfastsecp256k1 -o test_bip352_kat
 * ============================================================================ */

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <array>
#include <vector>

#ifndef UFSECP_BUILDING
#define UFSECP_BUILDING
#endif
#include "ufsecp/ufsecp.h"

static int g_pass = 0, g_fail = 0;

#define CHECK_KAT(cond, id, msg)                                           \
    do {                                                                   \
        if (cond) { ++g_pass; }                                            \
        else { ++g_fail;                                                   \
               std::printf("  FAIL [%s] %s\n", (id), (msg)); }            \
    } while (0)

/* Helper: compare len bytes, print FAIL with hex diff on mismatch */
static bool bytes_eq(const uint8_t* a, const uint8_t* b, size_t len,
                     const char* kat_id, const char* label) {
    if (std::memcmp(a, b, len) == 0) {
        ++g_pass;
        return true;
    }
    ++g_fail;
    std::printf("  FAIL [%s] %s mismatch\n    got: ", kat_id, label);
    for (size_t i = 0; i < len; i++) std::printf("%02x", a[i]);
    std::printf("\n    exp: ");
    for (size_t i = 0; i < len; i++) std::printf("%02x", b[i]);
    std::printf("\n");
    return false;
}

/* -----------------------------------------------------------------------
 * Fixed test keys (small scalars — deliberately simple for auditability)
 * ---------------------------------------------------------------------- */

static constexpr uint8_t SK_7[32]  = {
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,7
};
static constexpr uint8_t SK_11[32] = {
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,11
};
static constexpr uint8_t SK_3[32]  = {
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3
};
static constexpr uint8_t SK_13[32] = {
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,13
};
static constexpr uint8_t SK_17[32] = {
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,17
};
static constexpr uint8_t SK_19[32] = {
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,19
};
static constexpr uint8_t SK_5[32]  = {
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,5
};

/*
 * Pre-computed expected values (computed by running the library on the vectors
 * above and recording the output — see scripts/compute_bip352_kat_vectors.cpp).
 *
 * Vector 1: scan=7  spend=11  input=3      k=0
 * Vector 2: scan=13 spend=17  input=19     k=0
 * Vector 3: scan=7  spend=11  input=3      k=1  (different k → different output)
 * Vector 4: scan=7  spend=11  inputs=[5,13] k=0  (2-input aggregation)
 */

/* Vector 1 */
static constexpr uint8_t V1_EXPECTED_PUBKEY[33] = {
    0x02,0xe4,0x00,0x98,0x3f,0xb9,0x1a,0xa0,0x69,0x66,0x77,0x99,0x91,0x9e,0xd7,
    0xb1,0x25,0xa0,0x2f,0xc5,0x40,0x31,0xb6,0x57,0x07,0xf8,0xb9,0x9b,0xb4,0x15,
    0xb6,0x01,0xe9
};
static constexpr uint8_t V1_EXPECTED_TWEAK[32] = {
    0xa1,0x5f,0xb1,0x01,0x8a,0x86,0xec,0x99,0xc5,0x31,0xce,0xe1,0xab,0x4e,0x71,
    0x35,0x20,0x8f,0x9a,0xa3,0xa1,0xa0,0x4e,0xe2,0xc7,0x16,0xe4,0xf7,0xfe,0xbb,
    0x4c,0xde
};

/* Vector 2 */
static constexpr uint8_t V2_EXPECTED_PUBKEY[33] = {
    0x02,0xf1,0x82,0xc7,0x4b,0x2e,0x4e,0xfe,0xb6,0x69,0x2e,0x99,0x7e,0x87,0xfa,
    0x12,0xe1,0x06,0x1f,0x34,0x90,0x45,0x11,0x8a,0xd3,0x60,0x72,0x26,0x66,0x65,
    0x35,0xc4,0x76
};
static constexpr uint8_t V2_EXPECTED_TWEAK[32] = {
    0xa2,0x8a,0xe3,0x20,0xf3,0x81,0x76,0xa1,0x50,0x6a,0x2f,0x63,0xc1,0x1c,0x95,
    0x72,0x77,0xa8,0x37,0xab,0x6d,0xaa,0xe1,0x36,0x7e,0x99,0x4a,0x29,0xdf,0x68,
    0xdf,0x11
};

/* Vector 3  (k=1, same keys as V1) */
static constexpr uint8_t V3_EXPECTED_PUBKEY[33] = {
    0x02,0x3b,0x94,0x7a,0xe5,0x5c,0x5a,0x84,0xb2,0x16,0xbc,0xf0,0x08,0xe8,0x7f,
    0x2a,0xf7,0xe6,0x92,0x4f,0x32,0xcd,0x6e,0x1c,0x02,0x3f,0xbd,0x5d,0xf0,0x78,
    0x8b,0xa0,0x2d
};
static constexpr uint8_t V3_EXPECTED_TWEAK[32] = {
    0xe0,0x32,0x2f,0xda,0x20,0x50,0xc2,0x5b,0x57,0x33,0x18,0xce,0x3e,0xab,0x4b,
    0xf7,0xf1,0xc2,0xca,0xd0,0xc3,0x9f,0xae,0xcd,0x4e,0xdf,0xb9,0xe4,0x4b,0x1f,
    0x64,0x99
};

/* Vector 4  (2 inputs [5,13], same recipient as V1) */
static constexpr uint8_t V4_EXPECTED_PUBKEY[33] = {
    0x03,0xf8,0x42,0xa3,0x73,0x52,0xd8,0x9c,0xb5,0x0b,0x8d,0x3c,0xc8,0x21,0x4b,
    0xa9,0xe1,0x4f,0x43,0x51,0xc0,0x02,0xf5,0x9c,0x77,0x77,0x9a,0x67,0x0f,0xf2,
    0x65,0x25,0x41
};
static constexpr uint8_t V4_EXPECTED_TWEAK[32] = {
    0xff,0xc1,0x05,0x0a,0x06,0x7f,0x20,0xc2,0x83,0x15,0x3e,0x21,0x1b,0xd9,0xd4,
    0x8b,0x5b,0x22,0x6c,0x77,0x9f,0xb3,0x15,0x41,0x37,0xc3,0x9e,0x49,0xf8,0x94,
    0x3f,0x21
};

/* ============================================================================
 * Helper: derive scan/spend pubkeys from scalar private keys
 * ========================================================================== */
static void derive_pubkey(ufsecp_ctx* ctx, const uint8_t sk[32], uint8_t pk[33]) {
    if (ufsecp_pubkey_create(ctx, sk, pk) != UFSECP_OK) {
        std::memset(pk, 0, 33);
    }
}

/* ============================================================================
 * BIP352-KAT-1..4  --  create_output byte-exact regression
 * ========================================================================== */
static void test_create_output_kat(ufsecp_ctx* ctx) {
    std::printf("[BIP352-KAT] create_output byte-exact regression\n");

    /* Vector 1: scan=7 spend=11 input=3 k=0 */
    {
        uint8_t scan_pub[33], spend_pub[33];
        derive_pubkey(ctx, SK_7, scan_pub);
        derive_pubkey(ctx, SK_11, spend_pub);

        uint8_t out_pub[33], tweak[32];
        auto rc = ufsecp_silent_payment_create_output(
            ctx, SK_3, 1, scan_pub, spend_pub, 0, out_pub, tweak);
        CHECK_KAT(rc == UFSECP_OK, "BIP352-KAT-1a", "create_output returned error");
        bytes_eq(out_pub, V1_EXPECTED_PUBKEY, 33, "BIP352-KAT-1b", "output pubkey");
        bytes_eq(tweak, V1_EXPECTED_TWEAK, 32, "BIP352-KAT-1c", "tweak scalar");
    }

    /* Vector 2: scan=13 spend=17 input=19 k=0 */
    {
        uint8_t scan_pub[33], spend_pub[33];
        derive_pubkey(ctx, SK_13, scan_pub);
        derive_pubkey(ctx, SK_17, spend_pub);

        uint8_t out_pub[33], tweak[32];
        auto rc = ufsecp_silent_payment_create_output(
            ctx, SK_19, 1, scan_pub, spend_pub, 0, out_pub, tweak);
        CHECK_KAT(rc == UFSECP_OK, "BIP352-KAT-2a", "create_output returned error");
        bytes_eq(out_pub, V2_EXPECTED_PUBKEY, 33, "BIP352-KAT-2b", "output pubkey");
        bytes_eq(tweak, V2_EXPECTED_TWEAK, 32, "BIP352-KAT-2c", "tweak scalar");
    }

    /* Vector 3: scan=7 spend=11 input=3 k=1 (different k) */
    {
        uint8_t scan_pub[33], spend_pub[33];
        derive_pubkey(ctx, SK_7, scan_pub);
        derive_pubkey(ctx, SK_11, spend_pub);

        uint8_t out_pub[33], tweak[32];
        auto rc = ufsecp_silent_payment_create_output(
            ctx, SK_3, 1, scan_pub, spend_pub, 1, out_pub, tweak);
        CHECK_KAT(rc == UFSECP_OK, "BIP352-KAT-3a", "create_output k=1 returned error");
        bytes_eq(out_pub, V3_EXPECTED_PUBKEY, 33, "BIP352-KAT-3b", "output pubkey k=1");
        bytes_eq(tweak, V3_EXPECTED_TWEAK, 32, "BIP352-KAT-3c", "tweak scalar k=1");
        /* k=0 and k=1 must differ */
        uint8_t out_pub_k0[33];
        ufsecp_silent_payment_create_output(ctx, SK_3, 1, scan_pub, spend_pub, 0, out_pub_k0, nullptr);
        CHECK_KAT(std::memcmp(out_pub, out_pub_k0, 33) != 0, "BIP352-KAT-3d",
                  "k=0 and k=1 produced same output (BROKEN)");
    }

    /* Vector 4: scan=7 spend=11 inputs=[5,13] k=0  (2-input private key sum) */
    {
        uint8_t scan_pub[33], spend_pub[33];
        derive_pubkey(ctx, SK_7, scan_pub);
        derive_pubkey(ctx, SK_11, spend_pub);

        /* two input private keys packed as 64 bytes */
        uint8_t input_sks[64];
        std::memcpy(input_sks,      SK_5,  32);
        std::memcpy(input_sks + 32, SK_13, 32);

        uint8_t out_pub[33], tweak[32];
        auto rc = ufsecp_silent_payment_create_output(
            ctx, input_sks, 2, scan_pub, spend_pub, 0, out_pub, tweak);
        CHECK_KAT(rc == UFSECP_OK, "BIP352-KAT-4a", "create_output 2-input returned error");
        bytes_eq(out_pub, V4_EXPECTED_PUBKEY, 33, "BIP352-KAT-4b", "output pubkey 2-input");
        bytes_eq(tweak, V4_EXPECTED_TWEAK, 32, "BIP352-KAT-4c", "tweak scalar 2-input");

        /* single-input with sum (5+13=18) vs two-input must agree */
        uint8_t sk_18[32];
        /* scalar 18 = 0x12 */
        std::memset(sk_18, 0, 32);
        sk_18[31] = 18;
        uint8_t out_pub_sum[33];
        ufsecp_silent_payment_create_output(
            ctx, sk_18, 1, scan_pub, spend_pub, 0, out_pub_sum, nullptr);
        bytes_eq(out_pub, out_pub_sum, 33, "BIP352-KAT-4d",
                 "2-input(5,13) != 1-input(18): input sum property broken");
    }
}

/* ============================================================================
 * BIP352-KAT-5..8  --  full round-trip: create → scan → receiver finds output
 * ========================================================================== */
static void test_roundtrip_kat(ufsecp_ctx* ctx) {
    std::printf("[BIP352-KAT] full round-trip scan detection\n");

    struct RTCase {
        const char*    id;
        const uint8_t* scan_sk;
        const uint8_t* spend_sk;
        const uint8_t* input_sk;
        uint32_t       k;  // highest k index to test; BIP-352 scan needs k=0..k
    };

    RTCase cases[] = {
        { "BIP352-KAT-5", SK_7,  SK_11, SK_3,  0 },
        { "BIP352-KAT-6", SK_13, SK_17, SK_19, 0 },
        /* KAT-7: BIP-352 scanner starts at k=0 and stops on first miss, so
         * the receiver scans for outputs at indices k=0 AND k=1 in one call.
         * We create both and verify that the k=1 output is also discovered. */
        { "BIP352-KAT-7", SK_7,  SK_11, SK_3,  1 },
        { "BIP352-KAT-8", SK_7,  SK_11, SK_19, 0 },
    };

    for (auto& c : cases) {
        uint8_t scan_pub[33], spend_pub[33], input_pub[33];
        derive_pubkey(ctx, c.scan_sk,  scan_pub);
        derive_pubkey(ctx, c.spend_sk, spend_pub);
        derive_pubkey(ctx, c.input_sk, input_pub);

        /* Build the full output set k=0..c.k  (BIP-352: no gaps allowed) */
        const uint32_t n_outs = c.k + 1;
        std::vector<uint8_t> outs_xonly(n_outs * 32);
        bool build_ok = true;
        for (uint32_t ki = 0; ki <= c.k; ++ki) {
            uint8_t pub33[33];
            auto rc2 = ufsecp_silent_payment_create_output(
                ctx, c.input_sk, 1, scan_pub, spend_pub, ki, pub33, nullptr);
            if (rc2 != UFSECP_OK) { build_ok = false; break; }
            std::memcpy(outs_xonly.data() + ki * 32, pub33 + 1, 32);
        }
        CHECK_KAT(build_ok, c.id, "create_output failed for k in range");
        if (!build_ok) continue;

        /* Receiver: scan all n_outs outputs in one call */
        std::vector<uint32_t> found_idx(n_outs);
        std::vector<uint8_t>  found_privkeys(n_outs * 32);
        size_t n_found = n_outs;

        auto rc = ufsecp_silent_payment_scan(
            ctx,
            c.scan_sk, c.spend_sk,
            input_pub, 1,
            outs_xonly.data(), static_cast<size_t>(n_outs),
            found_idx.data(), found_privkeys.data(), &n_found);

        CHECK_KAT(rc == UFSECP_OK,        c.id, "scan returned error");
        CHECK_KAT(n_found == n_outs,       c.id, "scan found fewer outputs than expected");

        /* Verify each found spending key matches its respective output */
        for (size_t fi = 0; fi < n_found && fi < n_outs; ++fi) {
            uint32_t oi = found_idx[fi];   /* index into outs_xonly */
            if (oi >= n_outs) continue;
            uint8_t derived_pub[33];
            auto rc2 = ufsecp_pubkey_create(ctx, found_privkeys.data() + fi * 32, derived_pub);
            CHECK_KAT(rc2 == UFSECP_OK, c.id, "pubkey_create from spending key failed");
            CHECK_KAT(std::memcmp(derived_pub + 1, outs_xonly.data() + oi * 32, 32) == 0,
                      c.id, "spending key does not correspond to output x-coordinate");
        }
    }
}

/* ============================================================================
 * BIP352-KAT-9..12  --  scan REJECTION: unrelated outputs are NOT found
 * ========================================================================== */
static void test_rejection_kat(ufsecp_ctx* ctx) {
    std::printf("[BIP352-KAT] scan rejection (unrelated output not matched)\n");

    /* Use scan=7 spend=11 input=3 k=0 as the recipient */
    uint8_t scan_pub[33], spend_pub[33], input_pub[33];
    derive_pubkey(ctx, SK_7,  scan_pub);
    derive_pubkey(ctx, SK_11, spend_pub);
    derive_pubkey(ctx, SK_3,  input_pub);

    /* Four unrelated x-only output coordinates that are NOT the silent payment */
    static constexpr uint8_t UNRELATED[4][32] = {
        { /* random-looking constant A */
          0xDE,0xAD,0xBE,0xEF,0xCA,0xFE,0xBA,0xBE,
          0x01,0x23,0x45,0x67,0x89,0xAB,0xCD,0xEF,
          0xFE,0xDC,0xBA,0x98,0x76,0x54,0x32,0x10,
          0x11,0x22,0x33,0x44,0x55,0x66,0x77,0x88 },
        { /* all-one bytes (valid-looking x coord on a potentially different curve) */
          0x11,0x11,0x11,0x11,0x11,0x11,0x11,0x11,
          0x11,0x11,0x11,0x11,0x11,0x11,0x11,0x11,
          0x11,0x11,0x11,0x11,0x11,0x11,0x11,0x11,
          0x11,0x11,0x11,0x11,0x11,0x11,0x11,0x11 },
        { /* generator x-only */
          0x79,0xBE,0x66,0x7E,0xF9,0xDC,0xBB,0xAC,
          0x55,0xA0,0x62,0x95,0xCE,0x87,0x0B,0x07,
          0x02,0x9B,0xFC,0xDB,0x2D,0xCE,0x28,0xD9,
          0x59,0xF2,0x81,0x5B,0x16,0xF8,0x17,0x98 },
        { /* output pubkey for a different recipient (scan=13 spend=17) */
          /* x-only from V2_EXPECTED_PUBKEY */
          0xf1,0x82,0xc7,0x4b,0x2e,0x4e,0xfe,0xb6,
          0x69,0x2e,0x99,0x7e,0x87,0xfa,0x12,0xe1,
          0x06,0x1f,0x34,0x90,0x45,0x11,0x8a,0xd3,
          0x60,0x72,0x26,0x66,0x65,0x35,0xc4,0x76 },
    };

    const char* ids[] = {
        "BIP352-KAT-9", "BIP352-KAT-10", "BIP352-KAT-11", "BIP352-KAT-12"
    };

    for (int i = 0; i < 4; ++i) {
        uint32_t found_idx[4];
        uint8_t  found_privkeys[4 * 32];
        size_t   n_found = 4;

        auto rc = ufsecp_silent_payment_scan(
            ctx,
            SK_7, SK_11,
            input_pub, 1,
            UNRELATED[i], 1,
            found_idx, found_privkeys, &n_found);

        /* Scan should succeed but find nothing */
        CHECK_KAT(rc == UFSECP_OK, ids[i], "scan returned unexpected error");
        CHECK_KAT(n_found == 0, ids[i], "scan falsely matched an unrelated output");
    }
}

/* ============================================================================
 * BIP352-KAT-13..16  --  output index isolation: k0, k1, k2 all differ
 * ========================================================================== */
static void test_index_isolation_kat(ufsecp_ctx* ctx) {
    std::printf("[BIP352-KAT] output index isolation (k=0,1,2 differ)\n");

    uint8_t scan_pub[33], spend_pub[33];
    derive_pubkey(ctx, SK_7,  scan_pub);
    derive_pubkey(ctx, SK_11, spend_pub);

    uint8_t out[3][33];
    for (uint32_t k = 0; k < 3; ++k) {
        auto rc = ufsecp_silent_payment_create_output(
            ctx, SK_3, 1, scan_pub, spend_pub, k, out[k], nullptr);
        CHECK_KAT(rc == UFSECP_OK, "BIP352-KAT-13", "create_output failed");
    }
    CHECK_KAT(std::memcmp(out[0], out[1], 33) != 0, "BIP352-KAT-14", "k=0 == k=1");
    CHECK_KAT(std::memcmp(out[1], out[2], 33) != 0, "BIP352-KAT-15", "k=1 == k=2");
    CHECK_KAT(std::memcmp(out[0], out[2], 33) != 0, "BIP352-KAT-16", "k=0 == k=2");
}

/* ============================================================================
 * BIP352-KAT-17..18  --  two-input round-trip
 * ========================================================================== */
static void test_two_input_roundtrip(ufsecp_ctx* ctx) {
    std::printf("[BIP352-KAT] two-input round-trip\n");

    uint8_t scan_pub[33], spend_pub[33];
    uint8_t input_pub5[33], input_pub13[33];
    derive_pubkey(ctx, SK_7,  scan_pub);
    derive_pubkey(ctx, SK_11, spend_pub);
    derive_pubkey(ctx, SK_5,  input_pub5);
    derive_pubkey(ctx, SK_13, input_pub13);

    /* Sender: two input private keys */
    uint8_t input_sks[64];
    std::memcpy(input_sks,      SK_5,  32);
    std::memcpy(input_sks + 32, SK_13, 32);

    uint8_t out_pub33[33];
    auto rc = ufsecp_silent_payment_create_output(
        ctx, input_sks, 2, scan_pub, spend_pub, 0, out_pub33, nullptr);
    CHECK_KAT(rc == UFSECP_OK, "BIP352-KAT-17a", "2-input create_output failed");

    /* Receiver: pass both input pubkeys */
    uint8_t input_pubs[66];
    std::memcpy(input_pubs,      input_pub5,  33);
    std::memcpy(input_pubs + 33, input_pub13, 33);

    uint8_t  out_xonly[32];
    std::memcpy(out_xonly, out_pub33 + 1, 32);

    uint32_t found_idx[4];
    uint8_t  found_privkeys[4 * 32];
    size_t   n_found = 4;

    rc = ufsecp_silent_payment_scan(
        ctx,
        SK_7, SK_11,
        input_pubs, 2,
        out_xonly, 1,
        found_idx, found_privkeys, &n_found);

    CHECK_KAT(rc == UFSECP_OK, "BIP352-KAT-17b", "2-input scan returned error");
    CHECK_KAT(n_found == 1,    "BIP352-KAT-17c", "2-input scan did not find output");

    /* Verify found spending key → matches the output pubkey */
    if (n_found == 1) {
        uint8_t check_pub[33];
        rc = ufsecp_pubkey_create(ctx, found_privkeys, check_pub);
        CHECK_KAT(rc == UFSECP_OK, "BIP352-KAT-18a", "pubkey_create check failed");
        CHECK_KAT(std::memcmp(check_pub + 1, out_xonly, 32) == 0,
                  "BIP352-KAT-18b", "2-input spending key x-coord mismatch");
    }
}

/* ============================================================================
 * Entry point
 * ========================================================================== */
int test_bip352_kat_run() {
    std::printf("=== BIP-352 KAT (Known-Answer Tests) ===\n");

    ufsecp_ctx* ctx = nullptr;
    if (ufsecp_ctx_create(&ctx) != UFSECP_OK || !ctx) {
        std::printf("  FATAL: ctx_create failed\n");
        return 1;
    }

    test_create_output_kat(ctx);
    test_roundtrip_kat(ctx);
    test_rejection_kat(ctx);
    test_index_isolation_kat(ctx);
    test_two_input_roundtrip(ctx);

    ufsecp_ctx_destroy(ctx);

    std::printf("\n=== BIP-352 KAT: %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}

#if defined(STANDALONE_TEST) || !defined(UNIFIED_AUDIT_RUNNER)
int main() { return test_bip352_kat_run(); }
#endif
