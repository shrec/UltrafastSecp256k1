// ============================================================================
// shim_tagged_hash.cpp -- BIP-340 tagged SHA256
// ============================================================================
#include "secp256k1.h"

#include <cstring>
#include <array>
#include <cstdint>

#include "secp256k1/schnorr.hpp"

extern "C" {

int secp256k1_tagged_sha256(
    const secp256k1_context *ctx, unsigned char *hash32,
    const unsigned char *tag, size_t taglen,
    const unsigned char *msg, size_t msglen)
{
    (void)ctx;
    if (!hash32 || !tag || !msg) return 0;

    // Build tag string (null-terminated for our API)
    // Our tagged_hash takes const char* tag, so we need a string version
    std::string tag_str(reinterpret_cast<const char *>(tag), taglen);

    auto result = secp256k1::tagged_hash(tag_str.c_str(), msg, msglen);
    std::memcpy(hash32, result.data(), 32);
    return 1;
}

} // extern "C"
