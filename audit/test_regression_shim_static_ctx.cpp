// ============================================================================
// test_regression_shim_static_ctx.cpp
// Regression: g_static_ctx aggregate initializer must include cached_r_G
// and cached_r_G_valid fields added by PERF-005.
//
// Bug fixed (ecf47967): shim_context.cpp's secp256k1_context_struct gained
// two fields (cached_r_G: fast::Point, cached_r_G_valid: bool) for the
// per-context blinding cache, but the g_static_ctx initializer wasn't updated.
// This caused: "cannot convert 'default_illegal_callback' to fast::Point".
//
// Tests:
//   SCS-1: secp256k1_context_static is non-null
//   SCS-2: secp256k1_context_create returns a valid context
//   SCS-3: secp256k1_context_destroy does not crash
// ============================================================================

#if !defined(UNIFIED_AUDIT_RUNNER) && !defined(STANDALONE_TEST)
#define STANDALONE_TEST
#endif

#include <cstdio>
static int g_pass = 0, g_fail = 0;
#include "audit_check.hpp"

#ifdef STANDALONE_TEST
#include "secp256k1.h"
#endif

#ifdef STANDALONE_TEST
static void test_static_ctx_nonnull() {
    CHECK(secp256k1_context_static != nullptr, "[SCS-1] secp256k1_context_static is non-null");
}

static void test_context_create_destroy() {
    secp256k1_context* ctx = secp256k1_context_create(
        SECP256K1_CONTEXT_SIGN | SECP256K1_CONTEXT_VERIFY);
    CHECK(ctx != nullptr, "[SCS-2] context_create returns non-null");
    if (ctx) {
        secp256k1_context_destroy(ctx);
        g_pass++;  // [SCS-3] destroy did not crash
    }
}
#endif

int test_regression_shim_static_ctx_run() {
#ifndef STANDALONE_TEST
    return ADVISORY_SKIP_CODE;
#else
    g_pass = 0; g_fail = 0;
    std::printf("[regression_shim_static_ctx] g_static_ctx PERF-005 field alignment\n");
    test_static_ctx_nonnull();
    test_context_create_destroy();
    std::printf("  pass=%d  fail=%d\n", g_pass, g_fail);
    return (g_fail == 0) ? 0 : 1;
#endif
}

#ifdef STANDALONE_TEST
int main() { return test_regression_shim_static_ctx_run(); }
#endif
