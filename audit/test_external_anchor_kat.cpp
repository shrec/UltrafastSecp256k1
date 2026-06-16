// ============================================================================
// test_external_anchor_kat.cpp
// ============================================================================
// EXTERNAL-ANCHOR known-answer tests — defeats the common-mode (self-anchored)
// failure mode found by the valid/invalid coverage audit (2026-06-11): many ops
// are gated only against values the SAME engine re-derives, so a shared error
// (e.g. a wrong tagged-hash or SHA wrapper) passes BOTH the valid and invalid
// gate. The only cure is a known-answer pinned to an EXTERNAL authority that did
// NOT come from this engine.
//
//   * SHA-512 at the ABI entry point (ufsecp_sha512) vs NIST FIPS 180-4 vectors —
//     the synthesis found ufsecp_sha512 was KAT-checked only against the internal
//     C++ impl, never the ABI surface. SHA-512 underlies BIP-32 (HMAC-SHA512), so
//     an external anchor here catches a shared-primitive error in HD derivation.
//   * Taproot output key (ufsecp_taproot_output_key) vs the OFFICIAL BIP-341
//     wallet-test-vectors (scriptPubKey[0], keypath-only). The native taproot path
//     was self-roundtrip only; this pins H_TapTweak(P) to the BIP-341 reference.
// ============================================================================

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <array>

#include "ufsecp/ufsecp.h"

static int g_pass = 0, g_fail = 0;
static void check(bool cond, const char* msg) {
    if (cond) { ++g_pass; }
    else      { ++g_fail; printf("  [FAIL] %s\n", msg); }
}

static int hexval(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}
static void hex2bin(const char* hex, uint8_t* out, size_t out_len) {
    for (size_t i = 0; i < out_len; ++i)
        out[i] = static_cast<uint8_t>((hexval(hex[2 * i]) << 4) | hexval(hex[2 * i + 1]));
}

int test_external_anchor_kat_run() {
    printf("======================================================================\n");
    printf("  External-anchor KAT (NIST SHA-512 + BIP-341 official vectors)\n");
    printf("======================================================================\n\n");

    // ── SHA-512 ABI vs NIST FIPS 180-4 ───────────────────────────────────────
    {
        uint8_t d[64], exp[64];
        // SHA-512("") (empty message)
        hex2bin("cf83e1357eefb8bdf1542850d66d8007d620e4050b5715dc83f4a921d36ce9ce"
                "47d0d13c5d85f2b0ff8318d2877eec2f63b931bd47417a81a538327af927da3e", exp, 64);
        check(ufsecp_sha512(reinterpret_cast<const uint8_t*>(""), 0, d) == UFSECP_OK
              && std::memcmp(d, exp, 64) == 0,
              "ufsecp_sha512(\"\") == NIST FIPS 180-4 digest");
        // SHA-512("abc")
        hex2bin("ddaf35a193617abacc417349ae20413112e6fa4e89a97ea20a9eeee64b55d39a"
                "2192992a274fc1a836ba3c23a3feebbd454d4423643ce80e2a9ac94fa54ca49f", exp, 64);
        check(ufsecp_sha512(reinterpret_cast<const uint8_t*>("abc"), 3, d) == UFSECP_OK
              && std::memcmp(d, exp, 64) == 0,
              "ufsecp_sha512(\"abc\") == NIST FIPS 180-4 digest");
    }

    // ── Taproot output key vs OFFICIAL BIP-341 wallet-test-vectors ────────────
    // scriptPubKey[0]: internalPubkey -> (no script tree) -> tweakedPubkey.
    {
        ufsecp_ctx* ctx = nullptr;
        check(ufsecp_ctx_create(&ctx) == UFSECP_OK, "ctx_create");
        uint8_t internal_x[32], expected_out[32], out[32];
        int parity = -1;
        hex2bin("d6889cb081036e0faefa3a35157ad71086b123b2b144b649798b494c300a961d", internal_x, 32);
        hex2bin("53a1f6e454df1aa2776a2814a721372d6258050de330b3c6d10ee8f4e0dda343", expected_out, 32);
        // merkle_root = NULL -> keypath-only: Q.x = (P + H_TapTweak(P)*G).x
        ufsecp_error_t e = ufsecp_taproot_output_key(ctx, internal_x, nullptr, out, &parity);
        check(e == UFSECP_OK && std::memcmp(out, expected_out, 32) == 0,
              "ufsecp_taproot_output_key(BIP-341 internal, no tree) == BIP-341 tweakedPubkey");
        ufsecp_ctx_destroy(ctx);
    }

    printf("\n[external_anchor_kat] %d/%d checks passed\n", g_pass, g_pass + g_fail);
    return (g_fail > 0) ? 1 : 0;
}

#ifdef STANDALONE_TEST
int main() { return test_external_anchor_kat_run(); }
#endif
