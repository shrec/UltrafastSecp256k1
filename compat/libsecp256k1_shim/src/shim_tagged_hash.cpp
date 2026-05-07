// ============================================================================
// shim_tagged_hash.cpp -- BIP-340 tagged SHA256
// ============================================================================
#include "secp256k1.h"
#include "shim_internal.hpp"

#include <cstring>
#include <array>
#include <cstdint>

#include "secp256k1/sha256.hpp"

extern "C" {

int secp256k1_tagged_sha256(
    const secp256k1_context *ctx, unsigned char *hash32,
    const unsigned char *tag, size_t taglen,
    const unsigned char *msg, size_t msglen)
{
    SHIM_REQUIRE_CTX(ctx);
    if (!hash32 || !tag || !msg) return 0;

    // BIP-340 tagged hash: SHA256(SHA256(tag) || SHA256(tag) || msg)
    // Implemented directly with the length-aware SHA256 API to:
    //   1. Avoid heap allocation (no std::string construction)
    //   2. Correctly handle tags with embedded null bytes (no null truncation)
    secp256k1::SHA256 tag_ctx;
    tag_ctx.update(tag, taglen);
    auto tag_hash = tag_ctx.finalize();

    secp256k1::SHA256 ctx2;
    ctx2.update(tag_hash.data(), 32);
    ctx2.update(tag_hash.data(), 32);
    ctx2.update(msg, msglen);
    auto result = ctx2.finalize();
    std::memcpy(hash32, result.data(), 32);
    return 1;
}

} // extern "C"
