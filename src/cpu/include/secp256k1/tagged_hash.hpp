#ifndef SECP256K1_TAGGED_HASH_HPP
#define SECP256K1_TAGGED_HASH_HPP

// ============================================================================
// BIP-340 Tagged Hash -- Shared Utilities
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

// Pre-computed BIP-340 midstates.
// ESP32/bare-metal: global constructors with non-trivial init run before the
// FreeRTOS scheduler starts; ESP-IDF's static-init mutex is null → crash.
// Solution: EspLazySHA256 is constexpr-constructed (.bss, zero bytes, no ctor
// call) and computes the SHA256 midstate on first operator const SHA256&() use.
#if defined(ESP_PLATFORM) || defined(SECP256K1_PLATFORM_ESP32)
#include <new>
class EspLazySHA256 {
    const char* tag_;
    alignas(SHA256) mutable unsigned char buf_[sizeof(SHA256)];
    mutable bool ready_;
public:
    constexpr explicit EspLazySHA256(const char* tag) noexcept
        : tag_(tag), buf_{}, ready_(false) {}
    operator const SHA256&() const noexcept {
        if (!ready_) {
            ::new(static_cast<void*>(buf_)) SHA256(make_tag_midstate(tag_));
            ready_ = true;
        }
        return *reinterpret_cast<const SHA256*>(buf_);
    }
};
#  define SECP256K1_MIDSTATE(name, tag) inline EspLazySHA256 name{tag};
#else
#  define SECP256K1_MIDSTATE(name, tag) inline const SHA256 name = make_tag_midstate(tag);
#endif

SECP256K1_MIDSTATE(g_aux_midstate,                  "BIP0340/aux")
SECP256K1_MIDSTATE(g_nonce_midstate,                "BIP0340/nonce")
SECP256K1_MIDSTATE(g_challenge_midstate,            "BIP0340/challenge")
SECP256K1_MIDSTATE(g_taptweak_midstate,             "TapTweak")
SECP256K1_MIDSTATE(g_tapbranch_midstate,            "TapBranch")
SECP256K1_MIDSTATE(g_tapleaf_midstate,              "TapLeaf")
SECP256K1_MIDSTATE(g_keyagg_list_midstate,          "KeyAgg list")
SECP256K1_MIDSTATE(g_keyagg_coeff_midstate,         "KeyAgg coefficient")
SECP256K1_MIDSTATE(g_musig_nonceblinding_midstate,  "MuSig/nonceblinding")
SECP256K1_MIDSTATE(g_musig_aux_midstate,            "MuSig/aux")
SECP256K1_MIDSTATE(g_musig_nonce_midstate,          "MuSig/nonce")
SECP256K1_MIDSTATE(g_frost_binding_midstate,        "FROST_binding")
#undef SECP256K1_MIDSTATE

// Fast tagged hash using a cached midstate (avoids re-computing tag prefix).
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Warray-bounds"
#pragma GCC diagnostic ignored "-Wstringop-overflow"
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
