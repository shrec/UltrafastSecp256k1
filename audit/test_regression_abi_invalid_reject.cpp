// ============================================================================
// test_regression_abi_invalid_reject.cpp
// ============================================================================
// VALID/INVALID coverage — the "live reject branch, no invalid-gate" trap.
//
// The valid/invalid gate-coverage audit (2026-06-11) found ABI operations whose
// REJECTION branch demonstrably exists in src/cpu/src/impl but is NEVER fed an
// invalid input by the wired/blocking audit suite — so a regression that drops
// the strict check (wrong-ACCEPT) would pass every gate. This module feeds each
// such branch an invalid input and asserts the documented rejection. (The thesis:
// every operation needs BOTH a valid-accept AND an invalid-reject gate.)
//
//   * ufsecp_seckey_negate:  privkey 0 / == n / >= n  -> UFSECP_ERR_BAD_KEY
//                            (ufsecp_core.cpp:165; the buffer must be left intact)
//   * ufsecp_shamir_trick:   scalar a == n -> BAD_INPUT; invalid point -> BAD_PUBKEY
//   * ufsecp_multi_scalar_mul: scalar == n -> BAD_INPUT; invalid point -> BAD_PUBKEY;
//                            n == 0 -> BAD_INPUT
//   plus a VALID control for each so the reject path is not vacuously passing.
// ============================================================================

#include <cstdio>
#include <cstdint>
#include <cstring>

#include "ufsecp/ufsecp.h"

static int g_pass = 0, g_fail = 0;
static void check(bool cond, const char* msg) {
    if (cond) { ++g_pass; }
    else      { ++g_fail; printf("  [FAIL] %s\n", msg); }
}

// secp256k1 group order n.
static const std::uint8_t ORDER_N[32] = {
    0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF, 0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFE,
    0xBA,0xAE,0xDC,0xE6,0xAF,0x48,0xA0,0x3B, 0xBF,0xD2,0x5E,0x8C,0xD0,0x36,0x41,0x41};

int test_regression_abi_invalid_reject_run() {
    printf("======================================================================\n");
    printf("  Regression: ABI invalid-input rejection (live reject branches)\n");
    printf("======================================================================\n\n");

    ufsecp_ctx* ctx = nullptr;
    check(ufsecp_ctx_create(&ctx) == UFSECP_OK && ctx != nullptr, "ctx_create");

    // ── ufsecp_seckey_negate: zero / n / >=n must be rejected, buffer intact ──
    {
        std::uint8_t zero[32] = {0};
        check(ufsecp_seckey_negate(ctx, zero) == UFSECP_ERR_BAD_KEY,
              "seckey_negate(0) -> BAD_KEY");

        std::uint8_t n_key[32]; std::memcpy(n_key, ORDER_N, 32);
        check(ufsecp_seckey_negate(ctx, n_key) == UFSECP_ERR_BAD_KEY,
              "seckey_negate(== n) -> BAD_KEY (the >= n live reject branch)");
        check(std::memcmp(n_key, ORDER_N, 32) == 0,
              "seckey_negate reject leaves the key buffer unmodified");

        std::uint8_t ffs[32]; std::memset(ffs, 0xFF, 32);
        check(ufsecp_seckey_negate(ctx, ffs) == UFSECP_ERR_BAD_KEY,
              "seckey_negate(0xff..ff >= n) -> BAD_KEY");

        // VALID control: a small key negates successfully.
        std::uint8_t ok_key[32] = {0}; ok_key[31] = 0x07;
        check(ufsecp_seckey_negate(ctx, ok_key) == UFSECP_OK,
              "seckey_negate(valid) -> OK (reject path not vacuous)");
    }

    // ── valid pubkeys for the MSM/Shamir controls ──
    std::uint8_t k1[32] = {0}, k2[32] = {0}; k1[31] = 0x11; k2[31] = 0x22;
    std::uint8_t P[33], Q[33], out[33];
    check(ufsecp_pubkey_create(ctx, k1, P) == UFSECP_OK, "pubkey_create P");
    check(ufsecp_pubkey_create(ctx, k2, Q) == UFSECP_OK, "pubkey_create Q");
    std::uint8_t bad_point[33]; std::memset(bad_point, 0, 33); bad_point[0] = 0x01; // invalid prefix

    // ── ufsecp_shamir_trick: scalar >= n and invalid point must be rejected ──
    {
        std::uint8_t a_ok[32] = {0}, b_ok[32] = {0}; a_ok[31] = 0x03; b_ok[31] = 0x05;
        check(ufsecp_shamir_trick(ctx, a_ok, P, b_ok, Q, out) == UFSECP_OK,
              "shamir_trick(valid) -> OK (control)");
        check(ufsecp_shamir_trick(ctx, ORDER_N, P, b_ok, Q, out) == UFSECP_ERR_BAD_INPUT,
              "shamir_trick(a == n) -> BAD_INPUT (live reject branch)");
        check(ufsecp_shamir_trick(ctx, a_ok, bad_point, b_ok, Q, out) == UFSECP_ERR_BAD_PUBKEY,
              "shamir_trick(invalid point P) -> BAD_PUBKEY (live reject branch)");
    }

    // ── ufsecp_multi_scalar_mul: scalar >= n, invalid point, n==0 rejected ──
    {
        std::uint8_t scal_ok[32] = {0}; scal_ok[31] = 0x09;
        check(ufsecp_multi_scalar_mul(ctx, scal_ok, P, 1, out) == UFSECP_OK,
              "multi_scalar_mul(valid n=1) -> OK (control)");
        check(ufsecp_multi_scalar_mul(ctx, ORDER_N, P, 1, out) == UFSECP_ERR_BAD_INPUT,
              "multi_scalar_mul(scalar == n) -> BAD_INPUT (live reject branch)");
        check(ufsecp_multi_scalar_mul(ctx, scal_ok, bad_point, 1, out) == UFSECP_ERR_BAD_PUBKEY,
              "multi_scalar_mul(invalid point) -> BAD_PUBKEY (live reject branch)");
        check(ufsecp_multi_scalar_mul(ctx, scal_ok, P, 0, out) == UFSECP_ERR_BAD_INPUT,
              "multi_scalar_mul(n == 0) -> BAD_INPUT");
    }

    ufsecp_ctx_destroy(ctx);

    printf("\n[regression_abi_invalid_reject] %d/%d checks passed\n", g_pass, g_pass + g_fail);
    return (g_fail > 0) ? 1 : 0;
}

#ifdef STANDALONE_TEST
int main() { return test_regression_abi_invalid_reject_run(); }
#endif
