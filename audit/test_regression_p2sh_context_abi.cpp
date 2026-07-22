// P2SH context-aware additive ABI regression (GitHub #348).
// The diagnostic entry point must preserve the legacy symbol and output.

#include "ufsecp.h"

#include <cstdio>
#include <cstring>

namespace {

int g_pass = 0;
int g_fail = 0;

void check(bool condition, const char* id, const char* description) {
    if (condition) {
        std::printf("[p2sh-context-abi] PASS %s: %s\n", id, description);
        ++g_pass;
    } else {
        std::printf("[p2sh-context-abi] FAIL %s: %s\n", id, description);
        ++g_fail;
    }
}

} // namespace

int test_regression_p2sh_context_abi_run() {
    g_pass = 0;
    g_fail = 0;
    std::printf("\n=== P2SH Context ABI Regression ===\n");

    ufsecp_ctx* ctx = nullptr;
    check(ufsecp_ctx_create(&ctx) == UFSECP_OK && ctx != nullptr, "PCA-0",
          "context creation succeeds");
    if (!ctx) return 1;

    uint8_t redeem_script[22] = {0x00, 0x14};
    for (size_t i = 2; i < sizeof(redeem_script); ++i)
        redeem_script[i] = static_cast<uint8_t>(i);

    char legacy[96] = {};
    char contextual[96] = {};
    size_t legacy_len = sizeof(legacy);
    size_t contextual_len = sizeof(contextual);
    check(ufsecp_addr_p2sh(redeem_script, sizeof(redeem_script),
                           UFSECP_NET_MAINNET, legacy, &legacy_len) == UFSECP_OK,
          "PCA-1", "legacy P2SH symbol remains callable");
    check(ufsecp_addr_p2sh_with_ctx(ctx, redeem_script, sizeof(redeem_script),
                                    UFSECP_NET_MAINNET, contextual,
                                    &contextual_len) == UFSECP_OK,
          "PCA-2", "context-aware P2SH succeeds");
    check(legacy_len == contextual_len && std::strcmp(legacy, contextual) == 0,
          "PCA-3", "legacy and context-aware entry points are byte-identical");

    contextual_len = sizeof(contextual);
    check(ufsecp_addr_p2sh_with_ctx(nullptr, redeem_script, sizeof(redeem_script),
                                    UFSECP_NET_MAINNET, contextual,
                                    &contextual_len) == UFSECP_ERR_NULL_ARG,
          "PCA-4", "context-aware entry point rejects a NULL context");

    contextual_len = sizeof(contextual);
    check(ufsecp_addr_p2sh_with_ctx(ctx, nullptr, 0, UFSECP_NET_MAINNET,
                                    contextual, &contextual_len) == UFSECP_ERR_NULL_ARG &&
              ufsecp_last_error(ctx) == UFSECP_ERR_NULL_ARG &&
              std::strcmp(ufsecp_last_error_msg(ctx), "redeem script is NULL") == 0,
          "PCA-5", "NULL script records a contextual diagnostic");

    contextual_len = sizeof(contextual);
    check(ufsecp_addr_p2sh_with_ctx(ctx, redeem_script, sizeof(redeem_script),
                                    2, contextual, &contextual_len) == UFSECP_ERR_BAD_INPUT &&
              ufsecp_last_error(ctx) == UFSECP_ERR_BAD_INPUT &&
              std::strcmp(ufsecp_last_error_msg(ctx), "invalid network") == 0,
          "PCA-6", "invalid network records a contextual diagnostic");

    contextual_len = 1;
    check(ufsecp_addr_p2sh_with_ctx(ctx, redeem_script, sizeof(redeem_script),
                                    UFSECP_NET_MAINNET, contextual,
                                    &contextual_len) == UFSECP_ERR_BUF_TOO_SMALL &&
              ufsecp_last_error(ctx) == UFSECP_ERR_BUF_TOO_SMALL &&
              std::strcmp(ufsecp_last_error_msg(ctx), "P2SH buffer too small") == 0,
          "PCA-7", "small output buffer records a contextual diagnostic");

    contextual_len = sizeof(contextual);
    check(ufsecp_addr_p2sh_with_ctx(ctx, redeem_script, sizeof(redeem_script),
                                    UFSECP_NET_TESTNET, contextual,
                                    &contextual_len) == UFSECP_OK &&
              ufsecp_last_error(ctx) == UFSECP_OK,
          "PCA-8", "success clears the prior context error");
    check(contextual[0] == '2', "PCA-9", "testnet P2SH keeps the expected prefix");

    ufsecp_ctx_destroy(ctx);
    std::printf("\n  %d passed  %d failed  (total %d)\n",
                g_pass, g_fail, g_pass + g_fail);
    return g_fail == 0 ? 0 : 1;
}

#ifdef STANDALONE_TEST
int main() {
    return test_regression_p2sh_context_abi_run();
}
#endif
