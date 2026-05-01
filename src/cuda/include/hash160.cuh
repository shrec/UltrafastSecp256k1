#pragma once
#include <cstdint>
#include <cstddef>

namespace secp256k1 {
namespace cuda {

__device__ __forceinline__ uint32_t rotr32(uint32_t x, int n) {
    return (x >> n) | (x << (32 - n));
}

__device__ __forceinline__ uint32_t rotl32(uint32_t x, int n) {
    return (x << n) | (x >> (32 - n));
}

__device__ __forceinline__ void sha256(const uint8_t* data, size_t len, uint8_t out[32]) {
    static __device__ __constant__ uint32_t k[64] = {
        0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U,
        0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
        0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
        0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
        0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
        0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
        0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
        0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
        0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
        0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
        0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U,
        0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
        0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U,
        0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
        0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
        0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U
    };

    uint32_t h0 = 0x6a09e667U;
    uint32_t h1 = 0xbb67ae85U;
    uint32_t h2 = 0x3c6ef372U;
    uint32_t h3 = 0xa54ff53aU;
    uint32_t h4 = 0x510e527fU;
    uint32_t h5 = 0x9b05688cU;
    uint32_t h6 = 0x1f83d9abU;
    uint32_t h7 = 0x5be0cd19U;

    uint8_t block[128];
    for (int i = 0; i < 128; ++i) block[i] = 0;

    for (size_t i = 0; i < len; ++i) {
        block[i] = data[i];
    }
    block[len] = 0x80;

    const uint64_t bit_len = static_cast<uint64_t>(len) * 8ULL;
    const int total_blocks = ((len + 1 + 8) <= 64) ? 1 : 2;
    const int last = total_blocks * 64 - 8;
    block[last + 0] = static_cast<uint8_t>(bit_len >> 56);
    block[last + 1] = static_cast<uint8_t>(bit_len >> 48);
    block[last + 2] = static_cast<uint8_t>(bit_len >> 40);
    block[last + 3] = static_cast<uint8_t>(bit_len >> 32);
    block[last + 4] = static_cast<uint8_t>(bit_len >> 24);
    block[last + 5] = static_cast<uint8_t>(bit_len >> 16);
    block[last + 6] = static_cast<uint8_t>(bit_len >> 8);
    block[last + 7] = static_cast<uint8_t>(bit_len);

    for (int b = 0; b < total_blocks; ++b) {
        uint32_t w[64];
        const int off = b * 64;
        for (int i = 0; i < 16; ++i) {
            const int j = off + i * 4;
            w[i] = (static_cast<uint32_t>(block[j]) << 24) |
                   (static_cast<uint32_t>(block[j + 1]) << 16) |
                   (static_cast<uint32_t>(block[j + 2]) << 8) |
                   (static_cast<uint32_t>(block[j + 3]));
        }
        for (int i = 16; i < 64; ++i) {
            const uint32_t s0 = rotr32(w[i - 15], 7) ^ rotr32(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const uint32_t s1 = rotr32(w[i - 2], 17) ^ rotr32(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }

        uint32_t a = h0;
        uint32_t b0 = h1;
        uint32_t c = h2;
        uint32_t d = h3;
        uint32_t e = h4;
        uint32_t f = h5;
        uint32_t g = h6;
        uint32_t h = h7;

        for (int i = 0; i < 64; ++i) {
            const uint32_t S1 = rotr32(e, 6) ^ rotr32(e, 11) ^ rotr32(e, 25);
            const uint32_t ch = (e & f) ^ (~e & g);
            const uint32_t temp1 = h + S1 + ch + k[i] + w[i];
            const uint32_t S0 = rotr32(a, 2) ^ rotr32(a, 13) ^ rotr32(a, 22);
            const uint32_t maj = (a & b0) ^ (a & c) ^ (b0 & c);
            const uint32_t temp2 = S0 + maj;

            h = g;
            g = f;
            f = e;
            e = d + temp1;
            d = c;
            c = b0;
            b0 = a;
            a = temp1 + temp2;
        }

        h0 += a;
        h1 += b0;
        h2 += c;
        h3 += d;
        h4 += e;
        h5 += f;
        h6 += g;
        h7 += h;
    }

    const uint32_t h[8] = {h0, h1, h2, h3, h4, h5, h6, h7};
    for (int i = 0; i < 8; ++i) {
        out[i * 4 + 0] = static_cast<uint8_t>(h[i] >> 24);
        out[i * 4 + 1] = static_cast<uint8_t>(h[i] >> 16);
        out[i * 4 + 2] = static_cast<uint8_t>(h[i] >> 8);
        out[i * 4 + 3] = static_cast<uint8_t>(h[i]);
    }
}

__device__ __forceinline__ uint32_t ripemd_f(int j, uint32_t x, uint32_t y, uint32_t z) {
    if (j <= 15) return x ^ y ^ z;
    if (j <= 31) return (x & y) | (~x & z);
    if (j <= 47) return (x | ~y) ^ z;
    if (j <= 63) return (x & z) | (y & ~z);
    return x ^ (y | ~z);
}

__device__ __forceinline__ void ripemd160_32(const uint8_t data[32], uint8_t out[20]) {
    const uint8_t r[80] = {
        0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
        7,4,13,1,10,6,15,3,12,0,9,5,2,14,11,8,
        3,10,14,4,9,15,8,1,2,7,0,6,13,11,5,12,
        1,9,11,10,0,8,12,4,13,3,7,15,14,5,6,2,
        4,0,5,9,7,12,2,10,14,1,3,8,11,6,15,13
    };
    const uint8_t r2[80] = {
        5,14,7,0,9,2,11,4,13,6,15,8,1,10,3,12,
        6,11,3,7,0,13,5,10,14,15,8,12,4,9,1,2,
        15,5,1,3,7,14,6,9,11,8,12,2,10,0,4,13,
        8,6,4,1,3,11,15,0,5,12,2,13,9,7,10,14,
        12,15,10,4,1,5,8,7,6,2,13,14,0,3,9,11
    };
    const uint8_t s[80] = {
        11,14,15,12,5,8,7,9,11,13,14,15,6,7,9,8,
        7,6,8,13,11,9,7,15,7,12,15,9,11,7,13,12,
        11,13,6,7,14,9,13,15,14,8,13,6,5,12,7,5,
        11,12,14,15,14,15,9,8,9,14,5,6,8,6,5,12,
        9,15,5,11,6,8,13,12,5,12,13,14,11,8,5,6
    };
    const uint8_t s2[80] = {
        8,9,9,11,13,15,15,5,7,7,8,11,14,14,12,6,
        9,13,15,7,12,8,9,11,7,7,12,7,6,15,13,11,
        9,7,15,11,8,6,6,14,12,13,5,14,13,13,7,5,
        15,5,8,11,14,14,6,14,6,9,12,9,12,5,15,8,
        8,5,12,9,12,5,14,6,8,13,6,5,15,13,11,11
    };
    const uint32_t K[5] = {
        0x00000000U, 0x5A827999U, 0x6ED9EBA1U, 0x8F1BBCDCU, 0xA953FD4EU
    };
    const uint32_t K2[5] = {
        0x50A28BE6U, 0x5C4DD124U, 0x6D703EF3U, 0x7A6D76E9U, 0x00000000U
    };

    uint8_t block[64];
    for (int i = 0; i < 64; ++i) block[i] = 0;
    for (int i = 0; i < 32; ++i) block[i] = data[i];
    block[32] = 0x80;
    const uint64_t bit_len = 32ULL * 8ULL;
    block[56] = static_cast<uint8_t>(bit_len);
    block[57] = static_cast<uint8_t>(bit_len >> 8);
    block[58] = static_cast<uint8_t>(bit_len >> 16);
    block[59] = static_cast<uint8_t>(bit_len >> 24);
    block[60] = static_cast<uint8_t>(bit_len >> 32);
    block[61] = static_cast<uint8_t>(bit_len >> 40);
    block[62] = static_cast<uint8_t>(bit_len >> 48);
    block[63] = static_cast<uint8_t>(bit_len >> 56);

    uint32_t X[16];
    for (int i = 0; i < 16; ++i) {
        const int j = i * 4;
        X[i] = static_cast<uint32_t>(block[j]) |
               (static_cast<uint32_t>(block[j + 1]) << 8) |
               (static_cast<uint32_t>(block[j + 2]) << 16) |
               (static_cast<uint32_t>(block[j + 3]) << 24);
    }

    uint32_t h0 = 0x67452301U;
    uint32_t h1 = 0xEFCDAB89U;
    uint32_t h2 = 0x98BADCFEU;
    uint32_t h3 = 0x10325476U;
    uint32_t h4 = 0xC3D2E1F0U;

    uint32_t a = h0, b0 = h1, c = h2, d = h3, e = h4;
    uint32_t a2 = h0, b2 = h1, c2 = h2, d2 = h3, e2 = h4;

    for (int j = 0; j < 80; ++j) {
        const uint32_t t = rotl32(a + ripemd_f(j, b0, c, d) + X[r[j]] + K[j / 16], s[j]) + e;
        a = e; e = d; d = rotl32(c, 10); c = b0; b0 = t;

        const uint32_t t2 = rotl32(a2 + ripemd_f(79 - j, b2, c2, d2) + X[r2[j]] + K2[j / 16], s2[j]) + e2;
        a2 = e2; e2 = d2; d2 = rotl32(c2, 10); c2 = b2; b2 = t2;
    }

    const uint32_t t = h1 + c + d2;
    h1 = h2 + d + e2;
    h2 = h3 + e + a2;
    h3 = h4 + a + b2;
    h4 = h0 + b0 + c2;
    h0 = t;

    const uint32_t h[5] = {h0, h1, h2, h3, h4};
    for (int i = 0; i < 5; ++i) {
        out[i * 4 + 0] = static_cast<uint8_t>(h[i]);
        out[i * 4 + 1] = static_cast<uint8_t>(h[i] >> 8);
        out[i * 4 + 2] = static_cast<uint8_t>(h[i] >> 16);
        out[i * 4 + 3] = static_cast<uint8_t>(h[i] >> 24);
    }
}

__device__ __forceinline__ void hash160_pubkey(const uint8_t* pubkey, size_t pubkey_len, uint8_t out[20]) {
    uint8_t sha[32];
    sha256(pubkey, pubkey_len, sha);
    ripemd160_32(sha, out);
}

} // namespace cuda
} // namespace secp256k1
