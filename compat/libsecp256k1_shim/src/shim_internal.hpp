// Internal shim utilities — not part of the public API.
// Include this in shim_*.cpp files that need access to ctx internals.
#pragma once
#include "secp256k1.h"

// Fire the illegal callback stored in ctx (if set), then the caller must
// return 0. Matches libsecp256k1 behaviour: the callback runs, then the
// function returns an error rather than aborting.
// Safe to call with ctx==nullptr: callback is silently skipped.
void secp256k1_shim_call_illegal_cb(const secp256k1_context* ctx, const char* msg) noexcept;
