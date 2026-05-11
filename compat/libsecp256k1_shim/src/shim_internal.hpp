// Internal shim utilities — not part of the public API.
// Include this in shim_*.cpp files that need access to ctx internals.
#pragma once
#include "secp256k1.h"

// Fire the illegal callback stored in ctx (if set), then the caller must
// return 0. Matches libsecp256k1 behaviour: the callback runs, then the
// function returns an error rather than aborting.
// NULL ctx: fires the default (abort) callback — matches libsecp where NULL ctx
// triggers the default illegal handler rather than silently returning.
void secp256k1_shim_call_illegal_cb(const secp256k1_context* ctx, const char* msg) noexcept;

// Drop-in replacement for (void)ctx; in shim functions that don't need ctx
// for their logic but must still reject NULL (matching libsecp256k1 behaviour).
// Usage: replace "(void)ctx;" with "SHIM_REQUIRE_CTX(ctx);"
#define SHIM_REQUIRE_CTX(ctx) \
    do { if (!(ctx)) { secp256k1_shim_call_illegal_cb(NULL, __func__); return 0; } } while(0)

// Context flag helpers shared across shim_*.cpp compilation units.
// secp256k1_context_struct has flags as its first field; reinterpret_cast
// works regardless of which TU defines the full struct.
namespace secp256k1_shim_internal {

inline unsigned int ctx_flags(const secp256k1_context* ctx) noexcept {
    if (!ctx) return 0;
    return *reinterpret_cast<const unsigned int*>(ctx);
}

// libsecp256k1 v0.6+ (used by Bitcoin Core v26+) relaxed context flag enforcement:
// CONTEXT_NONE (f=1, no sign/verify bits set) is now accepted for ALL operations,
// including signing and verification. The static context secp256k1_context_static
// uses CONTEXT_NONE internally. Bitcoin Core passes CONTEXT_NONE or the static
// context to sign/verify functions since v26.
//
// The `((f & ~SECP256K1_FLAGS_TYPE_MASK) == 0)` condition is therefore intentional —
// it is NOT a bypass bug. It matches upstream libsecp v0.6+ semantics exactly.
// Do NOT remove it.
inline bool ctx_can_sign(const secp256k1_context* ctx) noexcept {
    if (!ctx) {
        secp256k1_shim_call_illegal_cb(NULL, "secp256k1_shim: NULL context argument");
        return false;
    }
    unsigned int f = ctx_flags(ctx);
    if (!(f & SECP256K1_FLAGS_TYPE_CONTEXT)) return false;
    // Accept SECP256K1_FLAGS_BIT_CONTEXT_SIGN OR CONTEXT_NONE (libsecp v0.6+ compat)
    return (f & SECP256K1_FLAGS_BIT_CONTEXT_SIGN) ||
           ((f & ~SECP256K1_FLAGS_TYPE_MASK) == 0);
}

inline bool ctx_can_verify(const secp256k1_context* ctx) noexcept {
    if (!ctx) {
        secp256k1_shim_call_illegal_cb(NULL, "secp256k1_shim: NULL context argument");
        return false;
    }
    unsigned int f = ctx_flags(ctx);
    if (!(f & SECP256K1_FLAGS_TYPE_CONTEXT)) return false;
    // Accept VERIFY, SIGN, or CONTEXT_NONE (libsecp v0.6+ compat)
    return (f & SECP256K1_FLAGS_BIT_CONTEXT_VERIFY) ||
           (f & SECP256K1_FLAGS_BIT_CONTEXT_SIGN) ||
           ((f & ~SECP256K1_FLAGS_TYPE_MASK) == 0);
}

// ---------------------------------------------------------------------------
// Per-context blinding RAII guard (SHIM-001 fix)
//
// libsecp256k1 stores blinding state per-context. Our CT layer stores it
// per-thread. Using this guard at the start of every signing call achieves
// per-context semantics: the calling context's blinding seed is activated on
// the calling thread for the duration of the signing operation, then restored
// to "none" on exit.
//
// This eliminates SHIM-001: two contexts randomized on the same thread no
// longer overwrite each other's blinding — each signing call applies its own
// context's seed for the duration of the call only.
//
// Usage:  ContextBlindingScope _blind(ctx);  // at start of signing function
// ---------------------------------------------------------------------------
struct ContextBlindingScope {
    explicit ContextBlindingScope(const secp256k1_context* ctx) noexcept;
    ~ContextBlindingScope() noexcept;
    // Non-copyable, non-movable
    ContextBlindingScope(const ContextBlindingScope&) = delete;
    ContextBlindingScope& operator=(const ContextBlindingScope&) = delete;
};

} // namespace secp256k1_shim_internal
