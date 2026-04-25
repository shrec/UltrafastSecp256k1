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
// Only loads from SECP256K1_CONFIG env var (explicit path) to avoid blocking
// on expensive table computation when no cache file is available.
static void shim_ensure_fixed_base() {
    static std::once_flag once;
    std::call_once(once, []() {
        // Try env var first; if not set, try CWD config.ini.
        // configure_fixed_base_from_env() is a no-op when env is unset.
        if (!secp256k1::fast::configure_fixed_base_from_env()) {
            secp256k1::fast::configure_fixed_base_from_file("config.ini");
        }
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
