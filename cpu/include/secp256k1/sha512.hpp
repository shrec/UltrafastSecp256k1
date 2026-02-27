#ifndef SECP256K1_SHA512_HPP
#define SECP256K1_SHA512_HPP
#pragma once

// ============================================================================
// Minimal SHA-512 implementation for BIP-32 (HMAC-SHA512)
// ============================================================================
// Self-contained, no external dependencies. Used only for HD key derivation.
// ============================================================================

#include <array>
#include <cstdint>
#include <cstddef>
#include <cstring>

namespace secp256k1 {

class SHA512 {
public:
    using digest_type = std::array<std::uint8_t, 64>;

    SHA512() noexcept { reset(); }

    void reset() noexcept {
        state_[0] = 0x6a09e667f3bcc908ULL;
        state_[1] = 0xbb67ae8584caa73bULL;
        state_[2] = 0x3c6ef372fe94f82bULL;
        state_[3] = 0xa54ff53a5f1d36f1ULL;
        state_[4] = 0x510e527fade682d1ULL;
        state_[5] = 0x9b05688c2b3e6c1fULL;
        state_[6] = 0x1f83d9abfb41bd6bULL;
        state_[7] = 0x5be0cd19137e2179ULL;
        total_ = 0;
        buf_len_ = 0;
    }

    void update(const void* data, std::size_t len) noexcept {
        auto ptr = static_cast<const std::uint8_t*>(data);
        total_ += len;

        if (buf_len_ > 0) {
            std::size_t fill = 128 - buf_len_;
            if (len < fill) {
                std::memcpy(buf_ + buf_len_, ptr, len);
                buf_len_ += len;
                return;
            }
            std::memcpy(buf_ + buf_len_, ptr, fill);
            compress(buf_);
            ptr += fill;
            len -= fill;
            buf_len_ = 0;
        }

        while (len >= 128) {
            compress(ptr);
            ptr += 128;
            len -= 128;
        }

        if (len > 0) {
            std::memcpy(buf_, ptr, len);
            buf_len_ = len;
        }
    }

    digest_type finalize() noexcept {
        std::uint64_t bit_len = total_ * 8;
        std::uint8_t pad = static_cast<std::uint8_t>(0x80);
        update(&pad, 1);

        // Pad to 112 mod 128
        while (buf_len_ != 112) {
            std::uint8_t zero = 0;
            update(&zero, 1);
        }

        // Append 128-bit length (we only use lower 64 bits)
        std::uint8_t len_buf[16]{};
        for (std::size_t i = 0; i < 8; ++i) {
            len_buf[8 + 7 - i] = static_cast<std::uint8_t>(bit_len >> (i * 8));
        }
        update(len_buf, 16);

        digest_type d{};
        for (std::size_t i = 0; i < 8; ++i) {
            d[i * 8 + 0] = static_cast<std::uint8_t>(state_[i] >> 56);
            d[i * 8 + 1] = static_cast<std::uint8_t>(state_[i] >> 48);
            d[i * 8 + 2] = static_cast<std::uint8_t>(state_[i] >> 40);
            d[i * 8 + 3] = static_cast<std::uint8_t>(state_[i] >> 32);
            d[i * 8 + 4] = static_cast<std::uint8_t>(state_[i] >> 24);
            d[i * 8 + 5] = static_cast<std::uint8_t>(state_[i] >> 16);
            d[i * 8 + 6] = static_cast<std::uint8_t>(state_[i] >> 8);
            d[i * 8 + 7] = static_cast<std::uint8_t>(state_[i]);
        }
        return d;
    }

    // One-shot convenience
    static digest_type hash(const void* data, std::size_t len) noexcept {
        SHA512 ctx;
        ctx.update(data, len);
        return ctx.finalize();
    }

private:
    static std::uint64_t rotr64(std::uint64_t x, unsigned n) noexcept {
        return (x >> n) | (x << (64u - n));
    }
    static std::uint64_t ch64(std::uint64_t x, std::uint64_t y, std::uint64_t z) noexcept {
        return (x & y) ^ (~x & z);
    }
    static std::uint64_t maj64(std::uint64_t x, std::uint64_t y, std::uint64_t z) noexcept {
        return (x & y) ^ (x & z) ^ (y & z);
    }
    static std::uint64_t Sigma0(std::uint64_t x) noexcept {
        return rotr64(x, 28) ^ rotr64(x, 34) ^ rotr64(x, 39);
    }
    static std::uint64_t Sigma1(std::uint64_t x) noexcept {
        return rotr64(x, 14) ^ rotr64(x, 18) ^ rotr64(x, 41);
    }
    static std::uint64_t sigma0(std::uint64_t x) noexcept {
        return rotr64(x, 1) ^ rotr64(x, 8) ^ (x >> 7);
    }
    static std::uint64_t sigma1(std::uint64_t x) noexcept {
        return rotr64(x, 19) ^ rotr64(x, 61) ^ (x >> 6);
    }

    void compress(const std::uint8_t* block) noexcept {
        static constexpr std::uint64_t K[80] = {
            0x428a2f98d728ae22ULL, 0x7137449123ef65cdULL,
            0xb5c0fbcfec4d3b2fULL, 0xe9b5dba58189dbbcULL,
            0x3956c25bf348b538ULL, 0x59f111f1b605d019ULL,
            0x923f82a4af194f9bULL, 0xab1c5ed5da6d8118ULL,
            0xd807aa98a3030242ULL, 0x12835b0145706fbeULL,
            0x243185be4ee4b28cULL, 0x550c7dc3d5ffb4e2ULL,
            0x72be5d74f27b896fULL, 0x80deb1fe3b1696b1ULL,
            0x9bdc06a725c71235ULL, 0xc19bf174cf692694ULL,
            0xe49b69c19ef14ad2ULL, 0xefbe4786384f25e3ULL,
            0x0fc19dc68b8cd5b5ULL, 0x240ca1cc77ac9c65ULL,
            0x2de92c6f592b0275ULL, 0x4a7484aa6ea6e483ULL,
            0x5cb0a9dcbd41fbd4ULL, 0x76f988da831153b5ULL,
            0x983e5152ee66dfabULL, 0xa831c66d2db43210ULL,
            0xb00327c898fb213fULL, 0xbf597fc7beef0ee4ULL,
            0xc6e00bf33da88fc2ULL, 0xd5a79147930aa725ULL,
            0x06ca6351e003826fULL, 0x142929670a0e6e70ULL,
            0x27b70a8546d22ffcULL, 0x2e1b21385c26c926ULL,
            0x4d2c6dfc5ac42aedULL, 0x53380d139d95b3dfULL,
            0x650a73548baf63deULL, 0x766a0abb3c77b2a8ULL,
            0x81c2c92e47edaee6ULL, 0x92722c851482353bULL,
            0xa2bfe8a14cf10364ULL, 0xa81a664bbc423001ULL,
            0xc24b8b70d0f89791ULL, 0xc76c51a30654be30ULL,
            0xd192e819d6ef5218ULL, 0xd69906245565a910ULL,
            0xf40e35855771202aULL, 0x106aa07032bbd1b8ULL,
            0x19a4c116b8d2d0c8ULL, 0x1e376c085141ab53ULL,
            0x2748774cdf8eeb99ULL, 0x34b0bcb5e19b48a8ULL,
            0x391c0cb3c5c95a63ULL, 0x4ed8aa4ae3418acbULL,
            0x5b9cca4f7763e373ULL, 0x682e6ff3d6b2b8a3ULL,
            0x748f82ee5defb2fcULL, 0x78a5636f43172f60ULL,
            0x84c87814a1f0ab72ULL, 0x8cc702081a6439ecULL,
            0x90befffa23631e28ULL, 0xa4506cebde82bde9ULL,
            0xbef9a3f7b2c67915ULL, 0xc67178f2e372532bULL,
            0xca273eceea26619cULL, 0xd186b8c721c0c207ULL,
            0xeada7dd6cde0eb1eULL, 0xf57d4f7fee6ed178ULL,
            0x06f067aa72176fbaULL, 0x0a637dc5a2c898a6ULL,
            0x113f9804bef90daeULL, 0x1b710b35131c471bULL,
            0x28db77f523047d84ULL, 0x32caab7b40c72493ULL,
            0x3c9ebe0a15c9bebcULL, 0x431d67c49c100d4cULL,
            0x4cc5d4becb3e42b6ULL, 0x597f299cfc657e2aULL,
            0x5fcb6fab3ad6faecULL, 0x6c44198c4a475817ULL
        };

        std::uint64_t W[80];
        for (int i = 0; i < 16; ++i) {
            W[i] = (static_cast<std::uint64_t>(block[i * 8 + 0]) << 56) |
                   (static_cast<std::uint64_t>(block[i * 8 + 1]) << 48) |
                   (static_cast<std::uint64_t>(block[i * 8 + 2]) << 40) |
                   (static_cast<std::uint64_t>(block[i * 8 + 3]) << 32) |
                   (static_cast<std::uint64_t>(block[i * 8 + 4]) << 24) |
                   (static_cast<std::uint64_t>(block[i * 8 + 5]) << 16) |
                   (static_cast<std::uint64_t>(block[i * 8 + 6]) << 8) |
                   (static_cast<std::uint64_t>(block[i * 8 + 7]));
        }
        for (int i = 16; i < 80; ++i) {
            W[i] = sigma1(W[i - 2]) + W[i - 7] + sigma0(W[i - 15]) + W[i - 16];
        }

        std::uint64_t a = state_[0], b = state_[1], c = state_[2], d = state_[3];
        std::uint64_t e = state_[4], f = state_[5], g = state_[6], h = state_[7];

        for (int i = 0; i < 80; ++i) {
            std::uint64_t t1 = h + Sigma1(e) + ch64(e, f, g) + K[i] + W[i];
            std::uint64_t t2 = Sigma0(a) + maj64(a, b, c);
            h = g; g = f; f = e; e = d + t1;
            d = c; c = b; b = a; a = t1 + t2;
        }

        state_[0] += a; state_[1] += b; state_[2] += c; state_[3] += d;
        state_[4] += e; state_[5] += f; state_[6] += g; state_[7] += h;
    }

    std::uint64_t state_[8]{};
    std::uint64_t total_{};
    std::uint8_t buf_[128]{};
    std::size_t buf_len_{};
};

} // namespace secp256k1

#endif // SECP256K1_SHA512_HPP
