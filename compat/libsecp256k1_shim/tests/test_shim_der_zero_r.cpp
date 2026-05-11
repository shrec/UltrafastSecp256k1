// ============================================================================
// test_shim_der_zero_r.cpp — regression test for DER parse with r=0
// ============================================================================
// Verifies that secp256k1_ecdsa_signature_parse_der rejects a DER-encoded
// ECDSA signature with r=0, returning 0 (failure).
//
// Upstream libsecp256k1 accepts r=0 at parse time and rejects it later at
// verify time. The shim is stricter: it rejects r=0 at parse time using
// parse_bytes_strict_nonzero. This divergence is documented in
// docs/SHIM_KNOWN_DIVERGENCES.md ("secp256k1_ecdsa_signature_parse_der —
// rejects r=0 or s=0").
//
// DER encoding of an ECDSA signature with r=0 and a valid s:
//
//   SEQUENCE {
//     INTEGER { 0 }             <- r = 0
//     INTEGER { 0x01 * 32 }     <- s = valid non-zero scalar
//   }
//
// The canonical DER for this is:
//   30 26                        -- SEQUENCE, 38 bytes
//     02 01 00                   -- INTEGER r = 0 (1 byte)
//     02 21 00 01...01           -- INTEGER s = 0x0101...01 (33 bytes, positive)
// ============================================================================

#include <secp256k1.h>
#include <cstdio>
#include <cstring>
#include <cstdint>

// DER-encoded ECDSA signature with r=0 and s=0x0101...01 (32 bytes, all 0x01).
// Structure:
//   30 26                  -- SEQUENCE length=38
//   02 01 00               -- INTEGER r=0 (length=1, value=0x00)
//   02 21 00               -- INTEGER s (length=33; leading 0x00 because high bit of 0x01 is 0,
//                             but we add the canonical positive encoding prefix)
//     01 01 01 01 01 01 01 01
//     01 01 01 01 01 01 01 01
//     01 01 01 01 01 01 01 01
//     01 01 01 01 01 01 01 01
//
// Note: DER INTEGER for s=0x0101...01 (32 bytes, MSB=0x01 < 0x80) does NOT
// need a leading 0x00. So length is 0x20 (32), not 0x21.
// Corrected layout:
//   30 25              -- SEQUENCE length=37
//   02 01 00           -- r=0 (3 bytes total)
//   02 20              -- s length=32
//     01*32            -- s value
static const uint8_t DER_R_ZERO[] = {
    0x30, 0x25,                          // SEQUENCE, 37 bytes
    0x02, 0x01, 0x00,                    // INTEGER r = 0
    0x02, 0x20,                          // INTEGER s, 32 bytes
    0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
    0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
    0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
    0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
};

static int g_pass = 0;
static int g_fail = 0;

#define CHECK(cond, label) do { \
    if (cond) { \
        printf("  [PASS] %s\n", (label)); \
        ++g_pass; \
    } else { \
        printf("  [FAIL] %s\n", (label)); \
        ++g_fail; \
    } \
} while (0)

int test_shim_der_zero_r_run() {
    printf("\n[SHIM-005] secp256k1_ecdsa_signature_parse_der — rejects r=0\n");

    secp256k1_context* ctx = secp256k1_context_create(SECP256K1_CONTEXT_VERIFY);
    if (!ctx) {
        printf("  [FAIL] context_create returned NULL\n");
        return 1;
    }

    secp256k1_ecdsa_signature sig;
    memset(&sig, 0, sizeof(sig));

    // The shim must return 0 for r=0 at parse time.
    int ret = secp256k1_ecdsa_signature_parse_der(
        ctx,
        &sig,
        DER_R_ZERO,
        sizeof(DER_R_ZERO)
    );

    CHECK(ret == 0,
        "parse_der with r=0 returns 0 (shim rejects at parse time — diverges from upstream)");

    secp256k1_context_destroy(ctx);

    printf("\n  Results: %d passed, %d failed\n", g_pass, g_fail);
    return (g_fail == 0) ? 0 : 1;
}

#ifdef STANDALONE_TEST
int main(void) {
    return test_shim_der_zero_r_run();
}
#endif
