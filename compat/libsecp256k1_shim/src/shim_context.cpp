// ============================================================================
// shim_context.cpp -- Context lifecycle (no-op for UltrafastSecp256k1)
// ============================================================================
#include "secp256k1.h"
#include <cstdlib>
#include <cstring>
#include <mutex>

#include "secp256k1/precompute.hpp"

// UltrafastSecp256k1 is stateless -- contexts are opaque dummies.
// We allocate a small sentinel so null-checks in user code pass.

struct secp256k1_context_struct {
    unsigned int flags;
};

static secp256k1_context_struct g_static_ctx = { SECP256K1_CONTEXT_NONE };

// Auto-initialize the fixed-base precomputed table once on first context_create.
// Resolution order:
//   1. SECP256K1_CONFIG env var  -> config file path
//   2. SECP256K1_CACHE_PATH env var -> direct .bin path (bypasses config.ini)
//   3. CWD config.ini
//   4. Absolute fallback path (CI build artifact)
static void shim_ensure_fixed_base() {
    static std::once_flag once;
    std::call_once(once, []() {
        // Resolution order (first success wins):
        //   1. SECP256K1_CONFIG env var  -> config file path
        //   2. SECP256K1_CACHE_PATH env var -> direct .bin path
        //   3. CWD config.ini
        //   4. configure_fixed_base_auto() -> auto-detect best available window
        if (secp256k1::fast::configure_fixed_base_from_env()) return;

        if (const char* cp = std::getenv("SECP256K1_CACHE_PATH")) {
            secp256k1::fast::FixedBaseConfig cfg;
            cfg.cache_path = cp;
            cfg.window_bits = 18;
            cfg.enable_glv = false;
            cfg.use_cache = true;
            secp256k1::fast::configure_fixed_base(cfg);
            return;
        }

        if (secp256k1::fast::configure_fixed_base_from_file("config.ini")) return;

        // Auto-detect: uses whatever precomputed tables exist on this machine,
        // falling back to w=8 (always available, no external file required).
        secp256k1::fast::configure_fixed_base_auto();
    });
}

extern "C" {

const secp256k1_context * const secp256k1_context_static = &g_static_ctx;

secp256k1_context *secp256k1_context_create(unsigned int flags) {
    (void)flags;
    shim_ensure_fixed_base();
    auto *ctx = static_cast<secp256k1_context *>(std::malloc(sizeof(secp256k1_context)));
    if (ctx) ctx->flags = flags;
    return ctx;
}

secp256k1_context *secp256k1_context_clone(const secp256k1_context *ctx) {
    if (!ctx) return nullptr;
    auto *clone = static_cast<secp256k1_context *>(std::malloc(sizeof(secp256k1_context)));
    if (clone) std::memcpy(clone, ctx, sizeof(secp256k1_context));
    return clone;
}

void secp256k1_context_destroy(secp256k1_context *ctx) {
    if (ctx && ctx != &g_static_ctx) std::free(ctx);
}

int secp256k1_context_randomize(secp256k1_context *ctx, const unsigned char *seed32) {
    (void)ctx; (void)seed32;
    // UltrafastSecp256k1 does not use blinding -- accepted as no-op.
    return 1;
}

void secp256k1_selftest(void) {
    // The underlying library has its own selftest; this is a compatibility stub.
}

} // extern "C"
