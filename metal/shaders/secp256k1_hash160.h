// =============================================================================
// UltrafastSecp256k1 Metal — Hash160 (SHA-256 → RIPEMD-160)
// =============================================================================
// Self-contained one-shot SHA-256 + RIPEMD-160 for public key hashing.
// Matches CUDA hash160.cuh output byte-for-byte.
//
// Functions:
//   sha256_oneshot  — One-shot SHA-256 (up to 128 bytes input)
//   ripemd_f        — RIPEMD-160 round function selector (5 variants)
//   ripemd160_32    — RIPEMD-160 optimized for exactly 32-byte input
//   hash160_pubkey  — SHA-256 then RIPEMD-160 (Bitcoin Hash160)
//
// Uses 32-bit ops only — fully compatible with Apple Silicon (no 64-bit int).
// NOTE: Uses its own one-shot SHA-256 (not the streaming SHA256Ctx from
//       secp256k1_extended.h) — more efficient for fixed-size inputs.
// =============================================================================

#pragma once

#include <metal_stdlib>
using namespace metal;

// =============================================================================
// SHA-256 Constants
// =============================================================================

constant uint HASH160_SHA256_K[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
    0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
    0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
    0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
    0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
    0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
    0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
    0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u
};

// =============================================================================
// SHA-256 Helpers
// =============================================================================

inline uint hash160_rotr32(uint x, uint n) { return (x >> n) | (x << (32u - n)); }
inline uint hash160_rotl32(uint x, uint n) { return (x << n) | (x >> (32u - n)); }

// =============================================================================
// One-shot SHA-256 (up to 128 bytes input, max 2 blocks)
// =============================================================================

inline void sha256_oneshot(thread const uchar* data, uint len, thread uchar out[32]) {
    uint h0 = 0x6a09e667u;
    uint h1 = 0xbb67ae85u;
    uint h2 = 0x3c6ef372u;
    uint h3 = 0xa54ff53au;
    uint h4 = 0x510e527fu;
    uint h5 = 0x9b05688cu;
    uint h6 = 0x1f83d9abu;
    uint h7 = 0x5be0cd19u;

    uchar block[128];
    for (int i = 0; i < 128; ++i) block[i] = 0;
    for (uint i = 0; i < len; ++i) block[i] = data[i];
    block[len] = 0x80;

    // Bit length as two 32-bit words (no 64-bit int on Metal)
    const uint bit_len_lo = len * 8u;
    const uint bit_len_hi = 0u;  // len < 128, so hi is always 0
    const int total_blocks = ((len + 1u + 8u) <= 64u) ? 1 : 2;
    const int last = total_blocks * 64 - 8;
    block[last + 0] = (uchar)(bit_len_hi >> 24);
    block[last + 1] = (uchar)(bit_len_hi >> 16);
    block[last + 2] = (uchar)(bit_len_hi >> 8);
    block[last + 3] = (uchar)(bit_len_hi);
    block[last + 4] = (uchar)(bit_len_lo >> 24);
    block[last + 5] = (uchar)(bit_len_lo >> 16);
    block[last + 6] = (uchar)(bit_len_lo >> 8);
    block[last + 7] = (uchar)(bit_len_lo);

    for (int b = 0; b < total_blocks; ++b) {
        uint w[64];
        const int off = b * 64;
        for (int i = 0; i < 16; ++i) {
            const int j = off + i * 4;
            w[i] = ((uint)block[j] << 24) |
                   ((uint)block[j + 1] << 16) |
                   ((uint)block[j + 2] << 8) |
                   ((uint)block[j + 3]);
        }
        for (int i = 16; i < 64; ++i) {
            const uint s0 = hash160_rotr32(w[i - 15], 7) ^ hash160_rotr32(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const uint s1 = hash160_rotr32(w[i - 2], 17) ^ hash160_rotr32(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }

        uint a = h0, b0 = h1, c = h2, d = h3;
        uint e = h4, f = h5, g = h6, h = h7;

        for (int i = 0; i < 64; ++i) {
            const uint S1 = hash160_rotr32(e, 6) ^ hash160_rotr32(e, 11) ^ hash160_rotr32(e, 25);
            const uint ch = (e & f) ^ (~e & g);
            const uint temp1 = h + S1 + ch + HASH160_SHA256_K[i] + w[i];
            const uint S0 = hash160_rotr32(a, 2) ^ hash160_rotr32(a, 13) ^ hash160_rotr32(a, 22);
            const uint maj = (a & b0) ^ (a & c) ^ (b0 & c);
            const uint temp2 = S0 + maj;

            h = g; g = f; f = e; e = d + temp1;
            d = c; c = b0; b0 = a; a = temp1 + temp2;
        }

        h0 += a; h1 += b0; h2 += c; h3 += d;
        h4 += e; h5 += f; h6 += g; h7 += h;
    }

    const uint hh[8] = {h0, h1, h2, h3, h4, h5, h6, h7};
    for (int i = 0; i < 8; ++i) {
        out[i * 4 + 0] = (uchar)(hh[i] >> 24);
        out[i * 4 + 1] = (uchar)(hh[i] >> 16);
        out[i * 4 + 2] = (uchar)(hh[i] >> 8);
        out[i * 4 + 3] = (uchar)(hh[i]);
    }
}

// =============================================================================
// RIPEMD-160 Constants
// =============================================================================

constant uchar RMD_R[80] = {
    0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
    7,4,13,1,10,6,15,3,12,0,9,5,2,14,11,8,
    3,10,14,4,9,15,8,1,2,7,0,6,13,11,5,12,
    1,9,11,10,0,8,12,4,13,3,7,15,14,5,6,2,
    4,0,5,9,7,12,2,10,14,1,3,8,11,6,15,13
};

constant uchar RMD_R2[80] = {
    5,14,7,0,9,2,11,4,13,6,15,8,1,10,3,12,
    6,11,3,7,0,13,5,10,14,15,8,12,4,9,1,2,
    15,5,1,3,7,14,6,9,11,8,12,2,10,0,13,4,
    8,6,4,1,3,11,15,0,5,12,2,13,9,7,10,14,
    12,15,10,4,1,5,8,7,6,2,13,14,0,3,9,11
};

constant uchar RMD_S[80] = {
    11,14,15,12,5,8,7,9,11,13,14,15,6,7,9,8,
    7,6,8,13,11,9,7,15,7,12,15,9,11,7,13,12,
    11,13,6,7,14,9,13,15,14,8,13,6,5,12,7,5,
    11,12,14,15,14,15,9,8,9,14,5,6,8,6,5,12,
    9,15,5,11,6,8,13,12,5,12,13,14,11,8,5,6
};

constant uchar RMD_S2[80] = {
    8,9,9,11,13,15,15,5,7,7,8,11,14,14,12,6,
    9,13,15,7,12,8,9,11,7,7,12,7,6,15,13,11,
    9,7,15,11,8,6,6,14,12,13,5,14,13,13,7,5,
    15,5,8,11,14,14,6,14,6,9,12,9,12,5,15,8,
    8,5,12,9,12,5,14,6,8,13,6,5,15,13,11,11
};

constant uint RMD_K[5] = {
    0x00000000u, 0x5A827999u, 0x6ED9EBA1u, 0x8F1BBCDCu, 0xA953FD4Eu
};

constant uint RMD_K2[5] = {
    0x50A28BE6u, 0x5C4DD124u, 0x6D703EF3u, 0x7A6D76E9u, 0x00000000u
};

// =============================================================================
// RIPEMD-160 Round Function Selector (5 boolean functions)
// =============================================================================

inline uint ripemd_f(int j, uint x, uint y, uint z) {
    if (j <= 15) return x ^ y ^ z;           // XOR
    if (j <= 31) return (x & y) | (~x & z);  // IF
    if (j <= 47) return (x | ~y) ^ z;        // ONX
    if (j <= 63) return (x & z) | (y & ~z);  // IFX
    return x ^ (y | ~z);                      // OXN
}

// =============================================================================
// RIPEMD-160 — Optimized for exactly 32-byte input (single block)
// =============================================================================

inline void ripemd160_32(thread const uchar data[32], thread uchar out[20]) {
    uchar block[64];
    for (int i = 0; i < 64; ++i) block[i] = 0;
    for (int i = 0; i < 32; ++i) block[i] = data[i];
    block[32] = 0x80;
    // bit_len = 256 = 0x100 → LE encoding
    const uint bit_len_lo = 32u * 8u;  // = 256
    block[56] = (uchar)(bit_len_lo);
    block[57] = (uchar)(bit_len_lo >> 8);
    block[58] = (uchar)(bit_len_lo >> 16);
    block[59] = (uchar)(bit_len_lo >> 24);
    // block[60..63] = 0 (bit_len_hi = 0)

    uint X[16];
    for (int i = 0; i < 16; ++i) {
        const int j = i * 4;
        X[i] = (uint)block[j] |
               ((uint)block[j + 1] << 8) |
               ((uint)block[j + 2] << 16) |
               ((uint)block[j + 3] << 24);
    }

    uint h0 = 0x67452301u;
    uint h1 = 0xEFCDAB89u;
    uint h2 = 0x98BADCFEu;
    uint h3 = 0x10325476u;
    uint h4 = 0xC3D2E1F0u;

    uint a = h0, b0 = h1, c = h2, d = h3, e = h4;
    uint a2 = h0, b2 = h1, c2 = h2, d2 = h3, e2 = h4;

    for (int j = 0; j < 80; ++j) {
        const uint t = hash160_rotl32(a + ripemd_f(j, b0, c, d) +
                                      X[RMD_R[j]] + RMD_K[j / 16],
                                      RMD_S[j]) + e;
        a = e; e = d; d = hash160_rotl32(c, 10); c = b0; b0 = t;

        const uint t2 = hash160_rotl32(a2 + ripemd_f(79 - j, b2, c2, d2) +
                                       X[RMD_R2[j]] + RMD_K2[j / 16],
                                       RMD_S2[j]) + e2;
        a2 = e2; e2 = d2; d2 = hash160_rotl32(c2, 10); c2 = b2; b2 = t2;
    }

    const uint t = h1 + c + d2;
    h1 = h2 + d + e2;
    h2 = h3 + e + a2;
    h3 = h4 + a + b2;
    h4 = h0 + b0 + c2;
    h0 = t;

    const uint hh[5] = {h0, h1, h2, h3, h4};
    for (int i = 0; i < 5; ++i) {
        out[i * 4 + 0] = (uchar)(hh[i]);
        out[i * 4 + 1] = (uchar)(hh[i] >> 8);
        out[i * 4 + 2] = (uchar)(hh[i] >> 16);
        out[i * 4 + 3] = (uchar)(hh[i] >> 24);
    }
}

// =============================================================================
// Hash160 — SHA-256 then RIPEMD-160 (Bitcoin public key hash)
// =============================================================================

inline void hash160_pubkey(thread const uchar* pubkey, uint pubkey_len, thread uchar out[20]) {
    uchar sha[32];
    sha256_oneshot(pubkey, pubkey_len, sha);
    ripemd160_32(sha, out);
}
