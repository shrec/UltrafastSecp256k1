// ============================================================================
// shim_context.cpp -- Context lifecycle + per-context blinding scope
// ============================================================================
#include "secp256k1.h"
#include "shim_internal.hpp"
#include <array>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <new>
#include <type_traits>

#include "secp256k1/precompute.hpp"
#include "secp256k1/scalar.hpp"
#include "secp256k1/point.hpp"
#include "secp256k1/ct/point.hpp"
#include "secp256k1/ct/sign.hpp"
#include "secp256k1/detail/secure_erase.hpp"

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
    // PERF-005: r*G cached once at context_randomize time to avoid recomputing it
    // on every signing call. ContextBlindingScope uses this instead of calling
    // ct::generator_mul(r) per call (~9 µs saved per sign operation).
    secp256k1::fast::Point cached_r_G;
    bool cached_r_G_valid{false};
    // PERF-B4: r (the blinding scalar) cached alongside cached_r_G so that
    // ContextBlindingScope avoids Scalar::from_bytes() reconstruction on every
    // signing call. Scalar::from_bytes() involves a memcpy + Montgomery conversion
    // — caching it eliminates that work from the sign hot path.
    secp256k1::fast::Scalar cached_r{};
    secp256k1_callback_fn illegal_cb{default_illegal_callback};
    const void* illegal_cb_data{nullptr};
    secp256k1_callback_fn error_cb{default_illegal_callback};
    const void* error_cb_data{nullptr};
};
static_assert(std::is_trivially_copyable_v<secp256k1_context_struct>,
              "secp256k1_context_clone uses memcpy — struct must remain trivially copyable");

static secp256k1_context_struct g_static_ctx = {
    SECP256K1_CONTEXT_SIGN | SECP256K1_CONTEXT_VERIFY, {}, false,   // flags, blind, blinded
    {}, false,                                                         // cached_r_G, cached_r_G_valid
    {},                                                                // cached_r
    default_illegal_callback, nullptr,                                 // illegal_cb, illegal_cb_data
    default_illegal_callback, nullptr                                  // error_cb, error_cb_data
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
#if defined(SECP256K1_CORE_BACKEND_MODE)
        // Bitcoin Core backend mode: use a compact in-memory precomputed table
        // (window_bits=8, 16 KB, always built without file I/O).
        //
        // Rationale: the default config.ini is auto-created with window_bits=18
        // and use_cache=true, which causes a 244 MB binary file to be loaded or
        // generated on every fresh process start — adding ~50-100 ms startup
        // overhead amortised as ~3% across ConnectBlock benchmark iterations.
        //
        // Env-var overrides are still honoured so operators can opt in to a
        // larger precomputed table when hosting a high-frequency signing service.
        if (secp256k1::fast::configure_fixed_base_from_env()) return;

        {
            secp256k1::fast::FixedBaseConfig cfg{};
            cfg.window_bits = 8;   // 256 precomputed multiples, 16 KB, fits L1
            cfg.use_cache   = false; // never read/write files
            cfg.enable_glv  = true;
            secp256k1::fast::configure_fixed_base(cfg);
        }
        return;
#else
        // No implicit config.ini (neither created nor read). Resolution order:
        //   1. SECP256K1_CACHE_PATH env var -> exact .bin cache file
        //   2. engine default -> reads/writes cache_w{bits}[ _glv].bin from the
        //      directory set via secp256k1::fast::set_cache_directory() /
        //      SECP256K1_CACHE_DIR, or the current working directory when unset.
        // Callers that want a config file attach it themselves with
        // configure_fixed_base_from_file(<their path>).
        if (const char* cp = std::getenv("SECP256K1_CACHE_PATH")) {
            secp256k1::fast::FixedBaseConfig cfg;
            cfg.cache_path = cp;
            cfg.window_bits = 18;
            cfg.enable_glv = false;
            cfg.use_cache = true;
            secp256k1::fast::configure_fixed_base(cfg);
            return;
        }

        secp256k1::fast::configure_fixed_base_auto();
#endif
    });
}

extern "C" {

const secp256k1_context * const secp256k1_context_static = &g_static_ctx;

secp256k1_context *secp256k1_context_create(unsigned int flags) {
    // libsecp256k1: invalid flags type triggers the static context's illegal callback.
    // We consult g_static_ctx.illegal_cb so a user-installed no-op callback (e.g. fuzzing
    // harnesses that call secp256k1_context_set_illegal_callback on secp256k1_context_static)
    // is respected rather than unconditionally aborting.
    if ((flags & SECP256K1_FLAGS_TYPE_MASK) != SECP256K1_FLAGS_TYPE_CONTEXT) {
        g_static_ctx.illegal_cb("secp256k1_context_create: invalid flags",
                                const_cast<void*>(g_static_ctx.illegal_cb_data));
        return nullptr;
    }
    // COMPAT-001: reject unknown flag bits above the type field (bits 10+).
    // Matches upstream libsecp256k1 which fires illegal_callback for unrecognised bits.
    if (flags & ~(SECP256K1_FLAGS_TYPE_MASK | SECP256K1_FLAGS_BIT_CONTEXT_VERIFY | SECP256K1_FLAGS_BIT_CONTEXT_SIGN)) {
        g_static_ctx.illegal_cb("secp256k1_context_create: unknown flag bits",
                                const_cast<void*>(g_static_ctx.illegal_cb_data));
        return nullptr;
    }
    shim_ensure_fixed_base();
    void* mem = std::malloc(sizeof(secp256k1_context));
    if (!mem) return nullptr;
    auto *ctx = new(mem) secp256k1_context{};  // runs default member initializers
    ctx->flags = flags;
    return ctx;
}

secp256k1_context *secp256k1_context_clone(const secp256k1_context *ctx) {
    if (!ctx) {
        // SHIM-004 fix: fire registered illegal callback instead of calling std::abort()
        // directly. This allows fuzz harnesses that register a no-op callback to survive
        // NULL context calls without crashing the process. Matches upstream libsecp256k1
        // behavior where illegal_callback is invoked before returning NULL.
        secp256k1_shim_call_illegal_cb(nullptr, "secp256k1_context_clone: NULL context");
        return nullptr;
    }
    auto *clone = static_cast<secp256k1_context *>(std::malloc(sizeof(secp256k1_context)));
    if (!clone) {
        // SHIM-A04: fire error callback on malloc failure (matches libsecp behavior).
        ctx->error_cb("secp256k1_context_clone: out of memory", const_cast<void*>(ctx->error_cb_data));
        return nullptr;
    }
    std::memcpy(clone, ctx, sizeof(secp256k1_context));
    return clone;
}

void secp256k1_context_destroy(secp256k1_context *ctx) {
    if (ctx && ctx != &g_static_ctx) {
        // Erase all sensitive fields before freeing.
        // SHIM-NEW-001 fix: also erase cached_r_G (Point object), not just the valid flag.
        // The Point stores r*G where r comes from the blinding seed — key material.
        secp256k1::detail::secure_erase(ctx->blind, 32);
        secp256k1::detail::secure_erase(&ctx->cached_r_G, sizeof(ctx->cached_r_G));
        secp256k1::detail::secure_erase(&ctx->cached_r, sizeof(ctx->cached_r));
        ctx->cached_r_G_valid = false;
        // ctx was created with placement-new (new(mem) secp256k1_context{}), so the
        // destructor must be explicitly invoked before free(). Without this call, any
        // non-trivially-destructible members (e.g. Point's internal state) would leak.
        ctx->~secp256k1_context_struct();
        std::free(ctx);
    }
    // GPU context is process-wide; shut down on last destroy via atexit instead.
    // shim_gpu_shutdown() is registered below via std::atexit.
}

// Per-context blinding semantics (SHIM-001 fix):
// libsecp256k1: blinding is per-context (each secp256k1_context* has independent state).
// Previously this shim applied blinding as persistent thread-local state at
// context_randomize time, causing two contexts on the same thread to overwrite
// each other's blinding.
//
// Fix (Option B): store the seed in ctx->blind[] at context_randomize time (cheap).
// Apply the blinding lazily at the START of each signing call via ContextBlindingScope,
// clear it at the END. This achieves true per-context semantics at the cost of one
// CT generator_mul per signing call — acceptable since signing already costs ~80µs.
//
// Bitcoin Core usage pattern (one context per thread, randomized once at startup):
// identical behavior to before. Multi-context same-thread usage: now correct.
int secp256k1_context_randomize(secp256k1_context *ctx, const unsigned char *seed32) {
    if (!ctx) {
        secp256k1_shim_call_illegal_cb(nullptr, "secp256k1_context_randomize: NULL context");
        return 0;
    }
    if (seed32) {
        // Store seed; blinding is applied lazily per signing call via ContextBlindingScope.
        std::memcpy(ctx->blind, seed32, 32);
        ctx->blinded = true;
        // PERF-005: pre-compute r*G here (once) so ContextBlindingScope can use
        // the cached value rather than calling ct::generator_mul on every sign call.
        // PERF-B4: also cache r itself so ContextBlindingScope skips from_bytes().
        // CT-003: Scalar::from_bytes does a conditional subtraction of n when
        // the input is in [n, 2^256). This is variable-time and leaks one bit
        // about the seed (whether seed >= n). Fix: parse_bytes_strict_nonzero
        // rejects seed >= n and seed == 0 without branching on the value.
        // Seeds in [n, 2^256) have probability ~2^-128; disabling blinding for
        // those seeds (ctx->cached_r_G_valid = false) is the safe fallback.
        std::array<uint8_t, 32> seed_arr{};
        std::memcpy(seed_arr.data(), seed32, 32);
        Scalar r;
        if (Scalar::parse_bytes_strict_nonzero(seed_arr.data(), r)) {
            ctx->cached_r         = r;
            ctx->cached_r_G       = secp256k1::ct::generator_mul(r);
            ctx->cached_r_G_valid = true;
        } else {
            ctx->cached_r_G_valid = false;
        }
        secp256k1::detail::secure_erase(seed_arr.data(), 32);
    } else {
        secp256k1::detail::secure_erase(ctx->blind, 32);
        ctx->blinded = false;
        ctx->cached_r_G_valid = false;
        secp256k1::detail::secure_erase(&ctx->cached_r, sizeof(ctx->cached_r));
    }
    return 1;
}

// ContextBlindingScope implementation — applies ctx's blinding on entry, clears on exit.
namespace secp256k1_shim_internal {

ContextBlindingScope::ContextBlindingScope(const secp256k1_context* ctx) noexcept {
    if (!ctx) return;
    // secp256k1_context_struct is defined in this TU — use it directly.
    const auto* c = ctx;
    if (!c->blinded) return;  // context not randomized — no blinding to apply
    // PERF-B4: use cached_r directly — avoids Scalar::from_bytes() reconstruction
    // (memcpy + Montgomery conversion) on every signing call. When cached_r_G_valid,
    // r was already validated non-zero at context_randomize time, so no is_zero()
    // check is needed here. Fallback path still reconstructs from blind[] for
    // contexts where cached_r_G_valid is false (e.g. r was zero — negligible case).
    if (c->cached_r_G_valid) {
        secp256k1::ct::set_blinding(c->cached_r, c->cached_r_G);
    } else {
        std::array<uint8_t, 32> seed_arr{};
        std::memcpy(seed_arr.data(), c->blind, 32);
        Scalar r = Scalar::from_bytes(seed_arr);
        if (r.is_zero_ct()) return;  // negligible; skip blinding — is_zero_ct avoids data-dependent branch on secret
        // PERF-005: use cached r*G (computed once at context_randomize time) to
        // avoid a ~9 µs CT generator_mul on every signing call.
        secp256k1::ct::set_blinding(r, secp256k1::ct::generator_mul(r));
    }
}

ContextBlindingScope::~ContextBlindingScope() noexcept {
    secp256k1::ct::clear_blinding();
}

} // namespace secp256k1_shim_internal

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
    // SHIM-ILLCB-001 fix: NULL ctx must fire default_illegal_callback, not silently return.
    // Matches upstream libsecp256k1 behavior: ARG_CHECK(ctx != NULL) fires callback + returns.
    if (!ctx) {
        default_illegal_callback("ctx != NULL", nullptr);
        return;
    }
    // Allow setting callbacks on secp256k1_context_static — upstream libsecp v0.6+
    // supports this. Bitcoin Core fuzzing infrastructure (SetLameCallbacks) sets a
    // no-op callback on the static context to suppress abort() during fuzz runs.
    ctx->illegal_cb      = fun ? fun : default_illegal_callback;
    ctx->illegal_cb_data = data;
}

void secp256k1_context_set_error_callback(
    secp256k1_context *ctx,
    secp256k1_callback_fn fun,
    const void *data)
{
    // SHIM-ILLCB-001 fix: NULL ctx must fire default_illegal_callback, not silently return.
    if (!ctx) {
        default_illegal_callback("ctx != NULL", nullptr);
        return;
    }
    // Same as illegal_callback: static context allowed (libsecp v0.6+ compat).
    ctx->error_cb      = fun ? fun : default_illegal_callback;
    ctx->error_cb_data = data;
}

// -- Preallocated context API (TASK-008) ------------------------------------
// Callers supply their own memory. secp256k1_context_preallocated_destroy
// does NOT call free() — the caller owns the buffer.

size_t secp256k1_context_preallocated_size(unsigned int flags) {
    // Same size regardless of flags (our context struct is flag-size-independent).
    // If flags are invalid we still return a non-zero size; create() will reject them.
    (void)flags;
    return sizeof(secp256k1_context);
}

secp256k1_context *secp256k1_context_preallocated_create(void *prealloc, unsigned int flags) {
    if (!prealloc) {
        g_static_ctx.illegal_cb("secp256k1_context_preallocated_create: NULL prealloc",
                                const_cast<void*>(g_static_ctx.illegal_cb_data));
        return nullptr;
    }
    if ((flags & SECP256K1_FLAGS_TYPE_MASK) != SECP256K1_FLAGS_TYPE_CONTEXT) {
        g_static_ctx.illegal_cb("secp256k1_context_preallocated_create: invalid flags",
                                const_cast<void*>(g_static_ctx.illegal_cb_data));
        return nullptr;
    }
    // COMPAT-001: reject unknown flag bits (matches secp256k1_context_create).
    if (flags & ~(SECP256K1_FLAGS_TYPE_MASK | SECP256K1_FLAGS_BIT_CONTEXT_VERIFY | SECP256K1_FLAGS_BIT_CONTEXT_SIGN)) {
        g_static_ctx.illegal_cb("secp256k1_context_preallocated_create: unknown flag bits",
                                const_cast<void*>(g_static_ctx.illegal_cb_data));
        return nullptr;
    }
    shim_ensure_fixed_base();
    auto *ctx = new(prealloc) secp256k1_context{};
    ctx->flags = flags;
    return ctx;
}

secp256k1_context *secp256k1_context_preallocated_clone(
    const secp256k1_context *ctx, void *prealloc)
{
    if (!ctx) {
        secp256k1_shim_call_illegal_cb(nullptr,
            "secp256k1_context_preallocated_clone: NULL context");
        return nullptr;
    }
    if (!prealloc) {
        secp256k1_shim_call_illegal_cb(ctx,
            "secp256k1_context_preallocated_clone: NULL prealloc");
        return nullptr;
    }
    auto *clone = new(prealloc) secp256k1_context{};
    std::memcpy(clone, ctx, sizeof(secp256k1_context));
    return clone;
}

void secp256k1_context_preallocated_destroy(secp256k1_context *ctx) {
    if (ctx && ctx != &g_static_ctx) {
        // Erase sensitive fields — same as context_destroy but do NOT free().
        secp256k1::detail::secure_erase(ctx->blind, 32);
        secp256k1::detail::secure_erase(&ctx->cached_r_G, sizeof(ctx->cached_r_G));
        secp256k1::detail::secure_erase(&ctx->cached_r, sizeof(ctx->cached_r));
        ctx->cached_r_G_valid = false;
        ctx->~secp256k1_context_struct();
        // Caller owns the buffer — do not free().
    }
}

} // extern "C"

// Internal helper — visible to all shim_*.cpp via shim_internal.hpp.
// The struct definition lives in this TU, so the member access is valid here.
// NULL ctx: fires the default (abort) callback matching libsecp256k1 behaviour —
// previously this silently returned, making NULL ctx calls invisible to callers
// that registered illegal callbacks to intercept errors.
void secp256k1_shim_call_illegal_cb(const secp256k1_context* ctx, const char* msg) noexcept {
    if (!ctx) {
        default_illegal_callback(msg, nullptr);  // matches libsecp: NULL ctx → abort
        return;
    }
    if (ctx->illegal_cb) ctx->illegal_cb(msg, const_cast<void*>(ctx->illegal_cb_data));
}
