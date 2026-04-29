// ============================================================================
// shim_context.cpp -- Context lifecycle (context_randomize installs thread-local CT blinding)
// ============================================================================
#include "secp256k1.h"
#include <array>
#include <cstdlib>
#include <cstring>
#include <mutex>

#ifdef SECP256K1_SHIM_GPU
#include "shim_gpu_state.hpp"
#endif

#include "secp256k1/precompute.hpp"
#include "secp256k1/scalar.hpp"
#include "secp256k1/ct/point.hpp"
#include "secp256k1/ct/sign.hpp"

using namespace secp256k1::fast;

// UltrafastSecp256k1 is stateless -- contexts are opaque dummies.
// We allocate a small sentinel so null-checks in user code pass.

static void default_illegal_callback(const char * /*text*/, void * /*data*/) noexcept {
    std::abort();
}

struct secp256k1_context_struct {
    unsigned int flags;
    unsigned char blind[32];   // randomization seed from secp256k1_context_randomize
    bool blinded;
    secp256k1_callback_fn illegal_cb{default_illegal_callback};
    const void* illegal_cb_data{nullptr};
    secp256k1_callback_fn error_cb{default_illegal_callback};
    const void* error_cb_data{nullptr};
};

static secp256k1_context_struct g_static_ctx = {
    SECP256K1_CONTEXT_NONE, {}, false,
    default_illegal_callback, nullptr,
    default_illegal_callback, nullptr
};

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

        if (secp256k1::fast::configure_fixed_base_from_file("config.ini")) {
#ifdef SECP256K1_SHIM_GPU
            shim_gpu_init("config.ini");
            std::atexit(shim_gpu_shutdown);
#endif
            return;
        }

        // Auto-detect: uses whatever precomputed tables exist on this machine,
        // falling back to w=8 (always available, no external file required).
        secp256k1::fast::configure_fixed_base_auto();

#ifdef SECP256K1_SHIM_GPU
        // Probe GPU even when config.ini is absent (enabled=false by default)
        shim_gpu_init("config.ini");
        std::atexit(shim_gpu_shutdown);
#endif
    });
}

extern "C" {

const secp256k1_context * const secp256k1_context_static = &g_static_ctx;

secp256k1_context *secp256k1_context_create(unsigned int flags) {
    shim_ensure_fixed_base();
    void* mem = std::malloc(sizeof(secp256k1_context));
    if (!mem) return nullptr;
    auto *ctx = new(mem) secp256k1_context{};  // runs default member initializers
    ctx->flags = flags;
    return ctx;
}

secp256k1_context *secp256k1_context_clone(const secp256k1_context *ctx) {
    // libsecp256k1 calls the illegal callback (default: abort) on NULL ctx.
    if (!ctx) { std::abort(); }
    auto *clone = static_cast<secp256k1_context *>(std::malloc(sizeof(secp256k1_context)));
    if (clone) std::memcpy(clone, ctx, sizeof(secp256k1_context));
    return clone;
}

void secp256k1_context_destroy(secp256k1_context *ctx) {
    if (ctx && ctx != &g_static_ctx) std::free(ctx);
    // GPU context is process-wide; shut down on last destroy via atexit instead.
    // shim_gpu_shutdown() is registered below via std::atexit.
}

// DEVIATION FROM LIBSECP CONTRACT (documented, intentional):
// In libsecp256k1 blinding is per-context: each secp256k1_context* has
// independent blinding state. In this shim, blinding is thread-local:
// secp256k1::ct::set_blinding() installs state on the CALLING THREAD, not on ctx.
//
// Practical impact: Bitcoin Core calls context_randomize once per context,
// once per thread, and does not share contexts across threads — so this
// deviation is harmless for the primary use case. Two independent contexts
// on the same thread will overwrite each other's blinding state; contexts
// randomized on thread A are not blinded when used on thread B.
//
// For the full per-context blinding model, store the seed in ctx and apply
// it lazily inside each signing call on the calling thread (Option B in
// the audit review). Tracked as a known deviation in docs/THREAD_SAFETY.md.
int secp256k1_context_randomize(secp256k1_context *ctx, const unsigned char *seed32) {
    if (!ctx) return 0;
    if (seed32) {
        std::memcpy(ctx->blind, seed32, 32);
        ctx->blinded = true;
        // Activate additive scalar blinding on this thread's signing path.
        // r = seed32 mod n; r_G = r*G precomputed (CT). Blinding is thread-local.
        std::array<uint8_t, 32> seed_arr{};
        std::memcpy(seed_arr.data(), seed32, 32);
        Scalar r = Scalar::from_bytes(seed_arr); // reduce mod n
        if (r.is_zero()) {
            // Astronomically unlikely; fall back to unblinded rather than panic.
            secp256k1::ct::clear_blinding();
            return 1;
        }
        auto r_G = secp256k1::ct::generator_mul(r);
        secp256k1::ct::set_blinding(r, r_G);
    } else {
        std::memset(ctx->blind, 0, 32);
        ctx->blinded = false;
        secp256k1::ct::clear_blinding();
    }
    return 1;
}

void secp256k1_selftest(void) {
    // Verify that 1*G produces the known generator x-coordinate.
    // Exercises scalar_mul_generator and field/point serialization.
    // abort() on failure (matches libsecp256k1 selftest contract).
    static constexpr uint8_t kGx[32] = {
        0x79, 0xBE, 0x66, 0x7E, 0xF9, 0xDC, 0xBB, 0xAC,
        0x55, 0xA0, 0x62, 0x95, 0xCE, 0x87, 0x0B, 0x07,
        0x02, 0x9B, 0xFC, 0xDB, 0x2D, 0xCE, 0x28, 0xD9,
        0x59, 0xF2, 0x81, 0x5B, 0x16, 0xF8, 0x17, 0x98
    };
    std::array<uint8_t, 32> one_bytes{};
    one_bytes[31] = 1;
    Scalar one = Scalar::from_bytes(one_bytes);
    auto G = scalar_mul_generator(one);
    auto unc = G.to_uncompressed(); // [04][x:32][y:32]
    if (std::memcmp(unc.data() + 1, kGx, 32) != 0) { std::abort(); }
}

void secp256k1_context_set_illegal_callback(
    secp256k1_context *ctx,
    secp256k1_callback_fn fun,
    const void *data)
{
    if (!ctx || ctx == &g_static_ctx) return;
    ctx->illegal_cb      = fun ? fun : default_illegal_callback;
    ctx->illegal_cb_data = data;
}

void secp256k1_context_set_error_callback(
    secp256k1_context *ctx,
    secp256k1_callback_fn fun,
    const void *data)
{
    if (!ctx || ctx == &g_static_ctx) return;
    ctx->error_cb      = fun ? fun : default_illegal_callback;
    ctx->error_cb_data = data;
}

} // extern "C"
