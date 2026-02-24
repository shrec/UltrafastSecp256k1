#ifndef SECP256K1_TAGGED_HASH_HPP
#define SECP256K1_TAGGED_HASH_HPP

// ============================================================================
// BIP-340 Tagged Hash — Shared Utilities
// ============================================================================
// Provides cached tagged-hash midstates for BIP-340 (Schnorr) operations.
// Used by both schnorr.cpp (fast path) and ct_sign.cpp (CT path).
//
// Eliminates duplication of make_tag_midstate / cached_tagged_hash / midstate
// constants between the two translation units.
// ============================================================================

#include "secp256k1/sha256.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace secp256k1::detail {

// Build a SHA256 midstate from a BIP-340 tag string.
// The midstate captures H(SHA256(tag) || SHA256(tag)) ready for further data.
inline SHA256 make_tag_midstate(std::string_view tag) {
    auto tag_hash = SHA256::hash(tag.data(), tag.size());
    SHA256 ctx;
    ctx.update(tag_hash.data(), 32);
    ctx.update(tag_hash.data(), 32);
    return ctx;
}

// Pre-computed BIP-340 midstates (constructed once, shared across TUs).
inline const SHA256 g_aux_midstate       = make_tag_midstate("BIP0340/aux");
inline const SHA256 g_nonce_midstate     = make_tag_midstate("BIP0340/nonce");
inline const SHA256 g_challenge_midstate = make_tag_midstate("BIP0340/challenge");

// Fast tagged hash using a cached midstate (avoids re-computing tag prefix).
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Warray-bounds"
#endif
inline std::array<uint8_t, 32> cached_tagged_hash(const SHA256& midstate,
                                                    const void* data,
                                                    std::size_t len) {
    SHA256 ctx = midstate;
    ctx.update(data, len);
    return ctx.finalize();
}
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

} // namespace secp256k1::detail

#endif // SECP256K1_TAGGED_HASH_HPP
