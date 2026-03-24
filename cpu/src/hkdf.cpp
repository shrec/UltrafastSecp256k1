// ============================================================================
// HKDF-SHA256 (RFC 5869) + HMAC-SHA256
// ============================================================================
// Built on the existing SHA256 class. Used by BIP-324 for key derivation.
// ============================================================================

#include "secp256k1/hkdf.hpp"
#include "secp256k1/sha256.hpp"
#include "secp256k1/detail/secure_erase.hpp"
#include <cstring>
#include <algorithm>

namespace secp256k1 {

// ============================================================================
// HMAC-SHA256 (RFC 2104)
// ============================================================================
// HMAC(K, m) = H((K' ^ opad) || H((K' ^ ipad) || m))
// K' = H(K) if len(K) > 64, else K zero-padded to 64 bytes
// ipad = 0x36 repeated 64 times
// opad = 0x5C repeated 64 times
// ============================================================================

std::array<std::uint8_t, 32> hmac_sha256(
    const std::uint8_t* key, std::size_t key_len,
    const std::uint8_t* data, std::size_t data_len) noexcept {

    std::uint8_t k_buf[64]{};

    if (key_len > 64) {
        auto h = SHA256::hash(key, key_len);
        std::memcpy(k_buf, h.data(), 32);
    } else {
        std::memcpy(k_buf, key, key_len);
    }

    std::uint8_t ipad[64], opad[64];
    for (int i = 0; i < 64; ++i) {
        ipad[i] = k_buf[i] ^ 0x36;
        opad[i] = k_buf[i] ^ 0x5C;
    }

    // inner = SHA256(ipad || data)
    SHA256 inner;
    inner.update(ipad, 64);
    inner.update(data, data_len);
    auto inner_hash = inner.finalize();

    // outer = SHA256(opad || inner_hash)
    SHA256 outer;
    outer.update(opad, 64);
    outer.update(inner_hash.data(), 32);
    auto result = outer.finalize();

    detail::secure_erase(k_buf, sizeof(k_buf));
    detail::secure_erase(ipad, sizeof(ipad));
    detail::secure_erase(opad, sizeof(opad));
    detail::secure_erase(inner_hash.data(), inner_hash.size());

    return result;
}

// ============================================================================
// HKDF-SHA256 Extract (RFC 5869 Section 2.2)
// ============================================================================

std::array<std::uint8_t, 32> hkdf_sha256_extract(
    const std::uint8_t* salt, std::size_t salt_len,
    const std::uint8_t* ikm, std::size_t ikm_len) noexcept {

    if (salt == nullptr || salt_len == 0) {
        std::uint8_t zero_salt[32]{};
        return hmac_sha256(zero_salt, 32, ikm, ikm_len);
    }
    return hmac_sha256(salt, salt_len, ikm, ikm_len);
}

// ============================================================================
// HKDF-SHA256 Expand (RFC 5869 Section 2.3)
// ============================================================================

bool hkdf_sha256_expand(
    const std::uint8_t prk[32],
    const std::uint8_t* info, std::size_t info_len,
    std::uint8_t* out, std::size_t out_len) noexcept {

    if (out_len > 255 * 32) return false;

    // Pre-compute HMAC ipad/opad blocks from PRK (constant across all iterations)
    std::uint8_t ipad[64], opad[64];
    for (int j = 0; j < 32; ++j) {
        ipad[j] = prk[j] ^ 0x36;
        opad[j] = prk[j] ^ 0x5C;
    }
    std::memset(ipad + 32, 0x36, 32);
    std::memset(opad + 32, 0x5C, 32);

    // Pre-hash the pad blocks — SHA256 processes exactly one 64-byte block.
    // Cloning these base states avoids re-hashing the pads on every iteration.
    SHA256 ipad_base;
    ipad_base.update(ipad, 64);
    SHA256 opad_base;
    opad_base.update(opad, 64);

    detail::secure_erase(ipad, sizeof(ipad));
    detail::secure_erase(opad, sizeof(opad));

    std::uint8_t t[32]{};
    std::size_t t_len = 0;
    std::size_t offset = 0;

    for (std::uint8_t i = 1; offset < out_len; ++i) {
        // T(i) = HMAC-SHA256(PRK, T(i-1) || info || i)
        SHA256 inner = ipad_base;  // clone pre-hashed ipad state
        if (t_len > 0) inner.update(t, t_len);
        if (info_len > 0) inner.update(info, info_len);
        inner.update(&i, 1);
        auto inner_hash = inner.finalize();

        SHA256 outer = opad_base;  // clone pre-hashed opad state
        outer.update(inner_hash.data(), 32);
        auto ti = outer.finalize();

        std::memcpy(t, ti.data(), 32);
        t_len = 32;

        std::size_t const copy = std::min<std::size_t>(32, out_len - offset);
        std::memcpy(out + offset, t, copy);
        offset += copy;

        // Erase per-iteration HMAC intermediates
        detail::secure_erase(inner_hash.data(), inner_hash.size());
        detail::secure_erase(ti.data(), ti.size());
    }

    detail::secure_erase(t, sizeof(t));
    detail::secure_erase(&ipad_base, sizeof(ipad_base));
    detail::secure_erase(&opad_base, sizeof(opad_base));
    return true;
}

} // namespace secp256k1
