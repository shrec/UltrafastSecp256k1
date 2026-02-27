#ifndef SECP256K1_SHA256_HPP
#define SECP256K1_SHA256_HPP
#pragma once

// ============================================================================
// SHA-256 implementation for ECDSA / Schnorr
// ============================================================================
// Hardware-accelerated via SHA-NI when available (runtime dispatch).
// Falls back to portable C++ on non-x86 or CPUs without SHA extensions.
// ============================================================================

#include <array>
#include <cstdint>
#include <cstddef>
#include <cstring>

namespace secp256k1 {

// Forward declare: implemented in hash_accel.cpp, dispatches to SHA-NI or scalar
namespace detail {
    void sha256_compress_dispatch(const std::uint8_t block[64],
                                  std::uint32_t state[8]) noexcept;
}

class SHA256 {
public:
    using digest_type = std::array<std::uint8_t, 32>;

    SHA256() noexcept { reset(); }

    void reset() noexcept {
        state_[0] = 0x6a09e667u; state_[1] = 0xbb67ae85u;
        state_[2] = 0x3c6ef372u; state_[3] = 0xa54ff53au;
        state_[4] = 0x510e527fu; state_[5] = 0x9b05688cu;
        state_[6] = 0x1f83d9abu; state_[7] = 0x5be0cd19u;
        total_ = 0;
        buf_len_ = 0;
    }

    void update(const void* data, std::size_t len) noexcept {
        auto ptr = static_cast<const std::uint8_t*>(data);
        total_ += len;

        if (buf_len_ > 0) {
            std::size_t const fill = 64 - buf_len_;
            if (len < fill) {
                std::memcpy(buf_ + buf_len_, ptr, len);
                buf_len_ += len;
                return;
            }
            std::memcpy(buf_ + buf_len_, ptr, fill);
            detail::sha256_compress_dispatch(buf_, state_);
            ptr += fill;
            len -= fill;
            buf_len_ = 0;
        }

        while (len >= 64) {
            detail::sha256_compress_dispatch(ptr, state_);
            ptr += 64;
            len -= 64;
        }

        if (len > 0) {
            std::memcpy(buf_, ptr, len);
            buf_len_ = len;
        }
    }

    digest_type finalize() noexcept {
        std::uint64_t const bits = total_ * 8;

        // -- Direct in-place padding (no per-byte update() calls) ---------
        // buf_len_ is invariantly [0,63] after update() processes full blocks.
        // Explicit bounds check satisfies static analysis (Sonar cpp:S3519).
        if (buf_len_ >= 64) buf_len_ = 0;
        std::size_t pos = buf_len_;   // capture index before increment
        buf_len_ = pos + 1;           // new length [1, 64]
        buf_[pos] = 0x80;             // write at [0, 63] -- always in bounds

        if (buf_len_ > 56) {
            // No room for 8-byte length -- pad, compress, start fresh block
            // buf_len_ is [57, 64]; (64 - buf_len_) is [0, 7]
            if (buf_len_ < 64) {
                std::memset(buf_ + buf_len_, 0, 64 - buf_len_);
            }
            detail::sha256_compress_dispatch(buf_, state_);
            buf_len_ = 0;
        }

        // Zero-pad to byte 56
        std::memset(buf_ + buf_len_, 0, 56 - buf_len_);

        // Append bit-length big-endian at bytes 56..63
        buf_[56] = static_cast<std::uint8_t>(bits >> 56);
        buf_[57] = static_cast<std::uint8_t>(bits >> 48);
        buf_[58] = static_cast<std::uint8_t>(bits >> 40);
        buf_[59] = static_cast<std::uint8_t>(bits >> 32);
        buf_[60] = static_cast<std::uint8_t>(bits >> 24);
        buf_[61] = static_cast<std::uint8_t>(bits >> 16);
        buf_[62] = static_cast<std::uint8_t>(bits >> 8);
        buf_[63] = static_cast<std::uint8_t>(bits);

        detail::sha256_compress_dispatch(buf_, state_);

        digest_type out{};
        for (std::size_t i = 0; i < 8; ++i) {
            out[i * 4 + 0] = static_cast<std::uint8_t>(state_[i] >> 24);
            out[i * 4 + 1] = static_cast<std::uint8_t>(state_[i] >> 16);
            out[i * 4 + 2] = static_cast<std::uint8_t>(state_[i] >> 8);
            out[i * 4 + 3] = static_cast<std::uint8_t>(state_[i]);
        }
        return out;
    }

    // One-shot convenience
    static digest_type hash(const void* data, std::size_t len) noexcept {
        SHA256 ctx;
        ctx.update(data, len);
        return ctx.finalize();
    }

    // Double-SHA256: SHA256(SHA256(data))
    static digest_type hash256(const void* data, std::size_t len) noexcept {
        auto h1 = hash(data, len);
        return hash(h1.data(), h1.size());
    }

private:
    std::uint32_t state_[8]{};
    std::uint8_t buf_[64]{};
    std::size_t buf_len_ = 0;
    std::uint64_t total_ = 0;
};

} // namespace secp256k1

#endif // SECP256K1_SHA256_HPP
