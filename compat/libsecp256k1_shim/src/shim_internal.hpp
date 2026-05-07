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

inline bool ctx_can_sign(const secp256k1_context* ctx) noexcept {
    if (!ctx) {
        secp256k1_shim_call_illegal_cb(NULL, "secp256k1_shim: NULL context argument");
        return false;
    }
    unsigned int f = ctx_flags(ctx);
    if (!(f & SECP256K1_FLAGS_TYPE_CONTEXT)) return false;
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
    return (f & SECP256K1_FLAGS_BIT_CONTEXT_VERIFY) ||
           (f & SECP256K1_FLAGS_BIT_CONTEXT_SIGN) ||
           ((f & ~SECP256K1_FLAGS_TYPE_MASK) == 0);
}

} // namespace secp256k1_shim_internal
