// ============================================================================
// test_regression_batch_dos_cap.cpp
// ============================================================================
// RESOURCE-EXHAUSTION regression (blind-zone #15): the batch sign ABI enforces a
// hard count ceiling (kMaxBatchN = 1<<20) BEFORE any allocation, so a hostile
// `count` cannot drive an unbounded malloc/DoS. The cap existed but was untested
// and ungated — one refactor from silently regressing.
//
//   * count > kMaxBatchN  -> UFSECP_ERR_BAD_INPUT (rejected before allocation)
//   * count == 0          -> UFSECP_ERR_BAD_INPUT
//   * a small valid count -> UFSECP_OK (the cap does not break normal use)
//
// The over-cap call passes tiny buffers on purpose: the ceiling check fires before
// the function ever reads `count * 32/64` bytes, so no large buffer is needed.
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

static const std::size_t kMaxBatchN = std::size_t{1} << 20;  // mirror the ABI ceiling

int test_regression_batch_dos_cap_run() {
    printf("======================================================================\n");
    printf("  Regression: batch-sign DoS count ceiling (blind-zone #15)\n");
    printf("======================================================================\n\n");

    ufsecp_ctx* ctx = nullptr;
    check(ufsecp_ctx_create(&ctx) == UFSECP_OK && ctx != nullptr, "ctx_create");

    std::uint8_t tiny[64] = {0};  // never read on the over-cap / zero-count paths

    // count > kMaxBatchN -> BAD_INPUT, BEFORE any count*size allocation.
    check(ufsecp_ecdsa_sign_batch(ctx, kMaxBatchN + 1, tiny, tiny, tiny) == UFSECP_ERR_BAD_INPUT,
          "ecdsa_sign_batch: count > kMaxBatchN rejected (BAD_INPUT, no alloc)");
    check(ufsecp_schnorr_sign_batch(ctx, kMaxBatchN + 1, tiny, tiny, tiny, tiny) == UFSECP_ERR_BAD_INPUT,
          "schnorr_sign_batch: count > kMaxBatchN rejected (BAD_INPUT, no alloc)");

    // count == 0 -> BAD_INPUT.
    check(ufsecp_ecdsa_sign_batch(ctx, 0, tiny, tiny, tiny) == UFSECP_ERR_BAD_INPUT,
          "ecdsa_sign_batch: count == 0 rejected");
    check(ufsecp_schnorr_sign_batch(ctx, 0, tiny, tiny, tiny, tiny) == UFSECP_ERR_BAD_INPUT,
          "schnorr_sign_batch: count == 0 rejected");

    // A small valid batch still works (the cap does not break normal use).
    {
        const std::size_t n = 2;
        std::uint8_t msgs[2 * 32], keys[2 * 32], sigs[2 * 64];
        std::memset(msgs, 0xA5, sizeof(msgs));
        std::memset(keys, 0, sizeof(keys));
        keys[31]  = 0x11;  // key 0 = small non-zero scalar
        keys[63]  = 0x22;  // key 1
        check(ufsecp_ecdsa_sign_batch(ctx, n, msgs, keys, sigs) == UFSECP_OK,
              "ecdsa_sign_batch: small valid batch succeeds (cap does not break normal use)");
    }

    ufsecp_ctx_destroy(ctx);

    printf("\n[regression_batch_dos_cap] %d/%d checks passed\n", g_pass, g_pass + g_fail);
    return (g_fail > 0) ? 1 : 0;
}

#ifdef STANDALONE_TEST
int main() { return test_regression_batch_dos_cap_run(); }
#endif
