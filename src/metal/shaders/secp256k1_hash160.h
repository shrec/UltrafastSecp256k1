// =============================================================================
// UltrafastSecp256k1 Metal -- Hash160 (SHA-256 -> RIPEMD-160)
// =============================================================================
// Self-contained SHA-256 + RIPEMD-160 for public key hashing.
// Matches CUDA hash160.cuh output byte-for-byte and mirrors its structure
// (NF-2026-00033, port of commit d86d3bbf; fast path contributed by zeroxbee,
// VanityPlate, https://github.com/zeroxbee/VanityPlate).
//
// Functions:
//   sha256_oneshot  -- Generic one-shot SHA-256 (up to 128 bytes input)
//   sha256_33       -- Registers-only SHA-256 of exactly 33 bytes
//   ripemd160_32    -- Fully unrolled RIPEMD-160 of exactly 32 bytes
//   hash160_pubkey  -- SHA-256 then RIPEMD-160 (Bitcoin Hash160); 33-byte keys
//                      take sha256_33, any other length takes sha256_oneshot
//
// Uses 32-bit ops only -- fully compatible with Apple Silicon (no 64-bit int).
// NOTE: Uses its own one-shot SHA-256 (not the streaming SHA256Ctx from
//       secp256k1_extended.h) -- more efficient for fixed-size inputs.
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
// SHA-256 for exactly 33 bytes (compressed secp256k1 pubkey), registers only
// =============================================================================
// The 33-byte message fits in a single 64-byte block:
//   bytes  0-32: input data
//   byte   33  : 0x80 (padding)
//   bytes 34-55: 0x00
//   bytes 56-63: 0x00 00 00 00 00 00 01 08 (264 bits, big-endian)
// That is 16 big-endian words w[0..15]. Padding and length are baked in as
// constants and rounds 16-63 expand a rolling w[16] window, so there is no
// block[] and no w[64] array. Same structure as sha256_33 in
// src/cuda/include/hash160.cuh.

inline void sha256_33(thread const uchar* data, thread uchar out[32]) {
    uint w[16];
    w[0]  = ((uint)data[0] <<24)|((uint)data[1] <<16)|((uint)data[2] <<8)|(uint)data[3];
    w[1]  = ((uint)data[4] <<24)|((uint)data[5] <<16)|((uint)data[6] <<8)|(uint)data[7];
    w[2]  = ((uint)data[8] <<24)|((uint)data[9] <<16)|((uint)data[10]<<8)|(uint)data[11];
    w[3]  = ((uint)data[12]<<24)|((uint)data[13]<<16)|((uint)data[14]<<8)|(uint)data[15];
    w[4]  = ((uint)data[16]<<24)|((uint)data[17]<<16)|((uint)data[18]<<8)|(uint)data[19];
    w[5]  = ((uint)data[20]<<24)|((uint)data[21]<<16)|((uint)data[22]<<8)|(uint)data[23];
    w[6]  = ((uint)data[24]<<24)|((uint)data[25]<<16)|((uint)data[26]<<8)|(uint)data[27];
    w[7]  = ((uint)data[28]<<24)|((uint)data[29]<<16)|((uint)data[30]<<8)|(uint)data[31];
    // w[8]: data[32] in the top byte, 0x80 padding in the next, then zeros
    w[8]  = ((uint)data[32]<<24)|0x00800000u;
    w[9]  = 0u; w[10] = 0u; w[11] = 0u; w[12] = 0u; w[13] = 0u;
    w[14] = 0u;       // message length high word: 264 < 2^32, so 0
    w[15] = 264u;     // message length low word:  33 * 8 = 264 bits

    uint a = 0x6a09e667u, b = 0xbb67ae85u, c = 0x3c6ef372u, d = 0xa54ff53au;
    uint e = 0x510e527fu, f = 0x9b05688cu, g = 0x1f83d9abu, h = 0x5be0cd19u;

// One SHA-256 round using the rolling window
#define S256_RND(i) do { \
    uint S1 = hash160_rotr32(e,6u)^hash160_rotr32(e,11u)^hash160_rotr32(e,25u); \
    uint ch = (e&f)^(~e&g); \
    uint t1 = h + S1 + ch + HASH160_SHA256_K[i] + w[(i)&15]; \
    uint S0 = hash160_rotr32(a,2u)^hash160_rotr32(a,13u)^hash160_rotr32(a,22u); \
    uint maj = (a&b)^(a&c)^(b&c); \
    h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+S0+maj; \
} while(0)

// Expand one message schedule word in-place (rolling w[16])
#define S256_EXP(i) do { \
    uint wi15 = w[((i)-15)&15]; \
    uint wi2  = w[((i)- 2)&15]; \
    uint s0 = hash160_rotr32(wi15,7u)^hash160_rotr32(wi15,18u)^(wi15>>3); \
    uint s1 = hash160_rotr32(wi2,17u)^hash160_rotr32(wi2,19u)^(wi2>>10); \
    w[(i)&15] = w[(i)&15] + s0 + w[((i)-7)&15] + s1; \
} while(0)

    // Rounds 0-15: use the message words as-is
    S256_RND(0);  S256_RND(1);  S256_RND(2);  S256_RND(3);
    S256_RND(4);  S256_RND(5);  S256_RND(6);  S256_RND(7);
    S256_RND(8);  S256_RND(9);  S256_RND(10); S256_RND(11);
    S256_RND(12); S256_RND(13); S256_RND(14); S256_RND(15);

    // Rounds 16-63: expand then round
#define S256_ER(i) S256_EXP(i); S256_RND(i)
    S256_ER(16); S256_ER(17); S256_ER(18); S256_ER(19);
    S256_ER(20); S256_ER(21); S256_ER(22); S256_ER(23);
    S256_ER(24); S256_ER(25); S256_ER(26); S256_ER(27);
    S256_ER(28); S256_ER(29); S256_ER(30); S256_ER(31);
    S256_ER(32); S256_ER(33); S256_ER(34); S256_ER(35);
    S256_ER(36); S256_ER(37); S256_ER(38); S256_ER(39);
    S256_ER(40); S256_ER(41); S256_ER(42); S256_ER(43);
    S256_ER(44); S256_ER(45); S256_ER(46); S256_ER(47);
    S256_ER(48); S256_ER(49); S256_ER(50); S256_ER(51);
    S256_ER(52); S256_ER(53); S256_ER(54); S256_ER(55);
    S256_ER(56); S256_ER(57); S256_ER(58); S256_ER(59);
    S256_ER(60); S256_ER(61); S256_ER(62); S256_ER(63);
#undef S256_RND
#undef S256_EXP
#undef S256_ER

    a += 0x6a09e667u; b += 0xbb67ae85u; c += 0x3c6ef372u; d += 0xa54ff53au;
    e += 0x510e527fu; f += 0x9b05688cu; g += 0x1f83d9abu; h += 0x5be0cd19u;

    // Store big-endian output (8 words -> 32 bytes)
    out[ 0]=(uchar)(a>>24); out[ 1]=(uchar)(a>>16); out[ 2]=(uchar)(a>>8); out[ 3]=(uchar)a;
    out[ 4]=(uchar)(b>>24); out[ 5]=(uchar)(b>>16); out[ 6]=(uchar)(b>>8); out[ 7]=(uchar)b;
    out[ 8]=(uchar)(c>>24); out[ 9]=(uchar)(c>>16); out[10]=(uchar)(c>>8); out[11]=(uchar)c;
    out[12]=(uchar)(d>>24); out[13]=(uchar)(d>>16); out[14]=(uchar)(d>>8); out[15]=(uchar)d;
    out[16]=(uchar)(e>>24); out[17]=(uchar)(e>>16); out[18]=(uchar)(e>>8); out[19]=(uchar)e;
    out[20]=(uchar)(f>>24); out[21]=(uchar)(f>>16); out[22]=(uchar)(f>>8); out[23]=(uchar)f;
    out[24]=(uchar)(g>>24); out[25]=(uchar)(g>>16); out[26]=(uchar)(g>>8); out[27]=(uchar)g;
    out[28]=(uchar)(h>>24); out[29]=(uchar)(h>>16); out[30]=(uchar)(h>>8); out[31]=(uchar)h;
}

// =============================================================================
// RIPEMD-160 round functions and step, for the fully unrolled ripemd160_32()
// =============================================================================
// Every word index, shift amount, function and constant below is a literal:
// no constant-address-space tables, no runtime-indexed register arrays, no
// per-step branches. Same structure as src/cuda/include/hash160.cuh
// (ripemd160_32).

#define RMD_F1(x, y, z) ((x) ^ (y) ^ (z))
#define RMD_F2(x, y, z) (((x) & (y)) | (~(x) & (z)))
#define RMD_F3(x, y, z) (((x) | ~(y)) ^ (z))
#define RMD_F4(x, y, z) (((x) & (z)) | ((y) & ~(z)))
#define RMD_F5(x, y, z) ((x) ^ ((y) | ~(z)))

// One step with rotating variable roles: instead of shifting a<-e<-d<-c<-b
// every step, the caller rotates which variable plays which role. After 80
// steps (a multiple of 5) the roles line up with the original names again.
#define RMD_STEP(a, b, c, d, e, F, x, k, s) do { \
    (a) = hash160_rotl32((a) + F(b, c, d) + (x) + (k), (s)) + (e); \
    (c) = hash160_rotl32((c), 10u); \
} while (0)

// =============================================================================
// RIPEMD-160 -- fully unrolled, exactly 32-byte input (single block)
// =============================================================================
// The 32-byte message fits in one 64-byte block: bytes 0-31 data, byte 32
// 0x80, bytes 33-55 zero, bytes 56-63 the 256-bit length little-endian. That
// maps to 16 little-endian words X[0..15] built straight in registers.

inline void ripemd160_32(thread const uchar data[32], thread uchar out[20]) {
    uint X[16];
    X[ 0] = (uint)data[ 0]|((uint)data[ 1]<<8)|((uint)data[ 2]<<16)|((uint)data[ 3]<<24);
    X[ 1] = (uint)data[ 4]|((uint)data[ 5]<<8)|((uint)data[ 6]<<16)|((uint)data[ 7]<<24);
    X[ 2] = (uint)data[ 8]|((uint)data[ 9]<<8)|((uint)data[10]<<16)|((uint)data[11]<<24);
    X[ 3] = (uint)data[12]|((uint)data[13]<<8)|((uint)data[14]<<16)|((uint)data[15]<<24);
    X[ 4] = (uint)data[16]|((uint)data[17]<<8)|((uint)data[18]<<16)|((uint)data[19]<<24);
    X[ 5] = (uint)data[20]|((uint)data[21]<<8)|((uint)data[22]<<16)|((uint)data[23]<<24);
    X[ 6] = (uint)data[24]|((uint)data[25]<<8)|((uint)data[26]<<16)|((uint)data[27]<<24);
    X[ 7] = (uint)data[28]|((uint)data[29]<<8)|((uint)data[30]<<16)|((uint)data[31]<<24);
    X[ 8] = 0x00000080u;   // 0x80 padding byte
    X[ 9] = 0u; X[10] = 0u; X[11] = 0u; X[12] = 0u; X[13] = 0u;
    X[14] = 256u;          // 32 * 8 = 256 bits, little-endian low word
    X[15] = 0u;

    uint h0 = 0x67452301u, h1 = 0xEFCDAB89u, h2 = 0x98BADCFEu,
         h3 = 0x10325476u, h4 = 0xC3D2E1F0u;

    uint a = h0, b = h1, c = h2, d = h3, e = h4;
    uint a2 = h0, b2 = h1, c2 = h2, d2 = h3, e2 = h4;

    // 80 left-line and 80 right-line steps, interleaved
    RMD_STEP(a,b,c,d,e, RMD_F1, X[ 0], 0, 11);
    RMD_STEP(a2,b2,c2,d2,e2, RMD_F5, X[ 5], 0x50A28BE6U,  8);
    RMD_STEP(e,a,b,c,d, RMD_F1, X[ 1], 0, 14);
    RMD_STEP(e2,a2,b2,c2,d2, RMD_F5, X[14], 0x50A28BE6U,  9);
    RMD_STEP(d,e,a,b,c, RMD_F1, X[ 2], 0, 15);
    RMD_STEP(d2,e2,a2,b2,c2, RMD_F5, X[ 7], 0x50A28BE6U,  9);
    RMD_STEP(c,d,e,a,b, RMD_F1, X[ 3], 0, 12);
    RMD_STEP(c2,d2,e2,a2,b2, RMD_F5, X[ 0], 0x50A28BE6U, 11);
    RMD_STEP(b,c,d,e,a, RMD_F1, X[ 4], 0,  5);
    RMD_STEP(b2,c2,d2,e2,a2, RMD_F5, X[ 9], 0x50A28BE6U, 13);
    RMD_STEP(a,b,c,d,e, RMD_F1, X[ 5], 0,  8);
    RMD_STEP(a2,b2,c2,d2,e2, RMD_F5, X[ 2], 0x50A28BE6U, 15);
    RMD_STEP(e,a,b,c,d, RMD_F1, X[ 6], 0,  7);
    RMD_STEP(e2,a2,b2,c2,d2, RMD_F5, X[11], 0x50A28BE6U, 15);
    RMD_STEP(d,e,a,b,c, RMD_F1, X[ 7], 0,  9);
    RMD_STEP(d2,e2,a2,b2,c2, RMD_F5, X[ 4], 0x50A28BE6U,  5);
    RMD_STEP(c,d,e,a,b, RMD_F1, X[ 8], 0, 11);
    RMD_STEP(c2,d2,e2,a2,b2, RMD_F5, X[13], 0x50A28BE6U,  7);
    RMD_STEP(b,c,d,e,a, RMD_F1, X[ 9], 0, 13);
    RMD_STEP(b2,c2,d2,e2,a2, RMD_F5, X[ 6], 0x50A28BE6U,  7);
    RMD_STEP(a,b,c,d,e, RMD_F1, X[10], 0, 14);
    RMD_STEP(a2,b2,c2,d2,e2, RMD_F5, X[15], 0x50A28BE6U,  8);
    RMD_STEP(e,a,b,c,d, RMD_F1, X[11], 0, 15);
    RMD_STEP(e2,a2,b2,c2,d2, RMD_F5, X[ 8], 0x50A28BE6U, 11);
    RMD_STEP(d,e,a,b,c, RMD_F1, X[12], 0,  6);
    RMD_STEP(d2,e2,a2,b2,c2, RMD_F5, X[ 1], 0x50A28BE6U, 14);
    RMD_STEP(c,d,e,a,b, RMD_F1, X[13], 0,  7);
    RMD_STEP(c2,d2,e2,a2,b2, RMD_F5, X[10], 0x50A28BE6U, 14);
    RMD_STEP(b,c,d,e,a, RMD_F1, X[14], 0,  9);
    RMD_STEP(b2,c2,d2,e2,a2, RMD_F5, X[ 3], 0x50A28BE6U, 12);
    RMD_STEP(a,b,c,d,e, RMD_F1, X[15], 0,  8);
    RMD_STEP(a2,b2,c2,d2,e2, RMD_F5, X[12], 0x50A28BE6U,  6);
    RMD_STEP(e,a,b,c,d, RMD_F2, X[ 7], 0x5A827999U,  7);
    RMD_STEP(e2,a2,b2,c2,d2, RMD_F4, X[ 6], 0x5C4DD124U,  9);
    RMD_STEP(d,e,a,b,c, RMD_F2, X[ 4], 0x5A827999U,  6);
    RMD_STEP(d2,e2,a2,b2,c2, RMD_F4, X[11], 0x5C4DD124U, 13);
    RMD_STEP(c,d,e,a,b, RMD_F2, X[13], 0x5A827999U,  8);
    RMD_STEP(c2,d2,e2,a2,b2, RMD_F4, X[ 3], 0x5C4DD124U, 15);
    RMD_STEP(b,c,d,e,a, RMD_F2, X[ 1], 0x5A827999U, 13);
    RMD_STEP(b2,c2,d2,e2,a2, RMD_F4, X[ 7], 0x5C4DD124U,  7);
    RMD_STEP(a,b,c,d,e, RMD_F2, X[10], 0x5A827999U, 11);
    RMD_STEP(a2,b2,c2,d2,e2, RMD_F4, X[ 0], 0x5C4DD124U, 12);
    RMD_STEP(e,a,b,c,d, RMD_F2, X[ 6], 0x5A827999U,  9);
    RMD_STEP(e2,a2,b2,c2,d2, RMD_F4, X[13], 0x5C4DD124U,  8);
    RMD_STEP(d,e,a,b,c, RMD_F2, X[15], 0x5A827999U,  7);
    RMD_STEP(d2,e2,a2,b2,c2, RMD_F4, X[ 5], 0x5C4DD124U,  9);
    RMD_STEP(c,d,e,a,b, RMD_F2, X[ 3], 0x5A827999U, 15);
    RMD_STEP(c2,d2,e2,a2,b2, RMD_F4, X[10], 0x5C4DD124U, 11);
    RMD_STEP(b,c,d,e,a, RMD_F2, X[12], 0x5A827999U,  7);
    RMD_STEP(b2,c2,d2,e2,a2, RMD_F4, X[14], 0x5C4DD124U,  7);
    RMD_STEP(a,b,c,d,e, RMD_F2, X[ 0], 0x5A827999U, 12);
    RMD_STEP(a2,b2,c2,d2,e2, RMD_F4, X[15], 0x5C4DD124U,  7);
    RMD_STEP(e,a,b,c,d, RMD_F2, X[ 9], 0x5A827999U, 15);
    RMD_STEP(e2,a2,b2,c2,d2, RMD_F4, X[ 8], 0x5C4DD124U, 12);
    RMD_STEP(d,e,a,b,c, RMD_F2, X[ 5], 0x5A827999U,  9);
    RMD_STEP(d2,e2,a2,b2,c2, RMD_F4, X[12], 0x5C4DD124U,  7);
    RMD_STEP(c,d,e,a,b, RMD_F2, X[ 2], 0x5A827999U, 11);
    RMD_STEP(c2,d2,e2,a2,b2, RMD_F4, X[ 4], 0x5C4DD124U,  6);
    RMD_STEP(b,c,d,e,a, RMD_F2, X[14], 0x5A827999U,  7);
    RMD_STEP(b2,c2,d2,e2,a2, RMD_F4, X[ 9], 0x5C4DD124U, 15);
    RMD_STEP(a,b,c,d,e, RMD_F2, X[11], 0x5A827999U, 13);
    RMD_STEP(a2,b2,c2,d2,e2, RMD_F4, X[ 1], 0x5C4DD124U, 13);
    RMD_STEP(e,a,b,c,d, RMD_F2, X[ 8], 0x5A827999U, 12);
    RMD_STEP(e2,a2,b2,c2,d2, RMD_F4, X[ 2], 0x5C4DD124U, 11);
    RMD_STEP(d,e,a,b,c, RMD_F3, X[ 3], 0x6ED9EBA1U, 11);
    RMD_STEP(d2,e2,a2,b2,c2, RMD_F3, X[15], 0x6D703EF3U,  9);
    RMD_STEP(c,d,e,a,b, RMD_F3, X[10], 0x6ED9EBA1U, 13);
    RMD_STEP(c2,d2,e2,a2,b2, RMD_F3, X[ 5], 0x6D703EF3U,  7);
    RMD_STEP(b,c,d,e,a, RMD_F3, X[14], 0x6ED9EBA1U,  6);
    RMD_STEP(b2,c2,d2,e2,a2, RMD_F3, X[ 1], 0x6D703EF3U, 15);
    RMD_STEP(a,b,c,d,e, RMD_F3, X[ 4], 0x6ED9EBA1U,  7);
    RMD_STEP(a2,b2,c2,d2,e2, RMD_F3, X[ 3], 0x6D703EF3U, 11);
    RMD_STEP(e,a,b,c,d, RMD_F3, X[ 9], 0x6ED9EBA1U, 14);
    RMD_STEP(e2,a2,b2,c2,d2, RMD_F3, X[ 7], 0x6D703EF3U,  8);
    RMD_STEP(d,e,a,b,c, RMD_F3, X[15], 0x6ED9EBA1U,  9);
    RMD_STEP(d2,e2,a2,b2,c2, RMD_F3, X[14], 0x6D703EF3U,  6);
    RMD_STEP(c,d,e,a,b, RMD_F3, X[ 8], 0x6ED9EBA1U, 13);
    RMD_STEP(c2,d2,e2,a2,b2, RMD_F3, X[ 6], 0x6D703EF3U,  6);
    RMD_STEP(b,c,d,e,a, RMD_F3, X[ 1], 0x6ED9EBA1U, 15);
    RMD_STEP(b2,c2,d2,e2,a2, RMD_F3, X[ 9], 0x6D703EF3U, 14);
    RMD_STEP(a,b,c,d,e, RMD_F3, X[ 2], 0x6ED9EBA1U, 14);
    RMD_STEP(a2,b2,c2,d2,e2, RMD_F3, X[11], 0x6D703EF3U, 12);
    RMD_STEP(e,a,b,c,d, RMD_F3, X[ 7], 0x6ED9EBA1U,  8);
    RMD_STEP(e2,a2,b2,c2,d2, RMD_F3, X[ 8], 0x6D703EF3U, 13);
    RMD_STEP(d,e,a,b,c, RMD_F3, X[ 0], 0x6ED9EBA1U, 13);
    RMD_STEP(d2,e2,a2,b2,c2, RMD_F3, X[12], 0x6D703EF3U,  5);
    RMD_STEP(c,d,e,a,b, RMD_F3, X[ 6], 0x6ED9EBA1U,  6);
    RMD_STEP(c2,d2,e2,a2,b2, RMD_F3, X[ 2], 0x6D703EF3U, 14);
    RMD_STEP(b,c,d,e,a, RMD_F3, X[13], 0x6ED9EBA1U,  5);
    RMD_STEP(b2,c2,d2,e2,a2, RMD_F3, X[10], 0x6D703EF3U, 13);
    RMD_STEP(a,b,c,d,e, RMD_F3, X[11], 0x6ED9EBA1U, 12);
    RMD_STEP(a2,b2,c2,d2,e2, RMD_F3, X[ 0], 0x6D703EF3U, 13);
    RMD_STEP(e,a,b,c,d, RMD_F3, X[ 5], 0x6ED9EBA1U,  7);
    RMD_STEP(e2,a2,b2,c2,d2, RMD_F3, X[ 4], 0x6D703EF3U,  7);
    RMD_STEP(d,e,a,b,c, RMD_F3, X[12], 0x6ED9EBA1U,  5);
    RMD_STEP(d2,e2,a2,b2,c2, RMD_F3, X[13], 0x6D703EF3U,  5);
    RMD_STEP(c,d,e,a,b, RMD_F4, X[ 1], 0x8F1BBCDCU, 11);
    RMD_STEP(c2,d2,e2,a2,b2, RMD_F2, X[ 8], 0x7A6D76E9U, 15);
    RMD_STEP(b,c,d,e,a, RMD_F4, X[ 9], 0x8F1BBCDCU, 12);
    RMD_STEP(b2,c2,d2,e2,a2, RMD_F2, X[ 6], 0x7A6D76E9U,  5);
    RMD_STEP(a,b,c,d,e, RMD_F4, X[11], 0x8F1BBCDCU, 14);
    RMD_STEP(a2,b2,c2,d2,e2, RMD_F2, X[ 4], 0x7A6D76E9U,  8);
    RMD_STEP(e,a,b,c,d, RMD_F4, X[10], 0x8F1BBCDCU, 15);
    RMD_STEP(e2,a2,b2,c2,d2, RMD_F2, X[ 1], 0x7A6D76E9U, 11);
    RMD_STEP(d,e,a,b,c, RMD_F4, X[ 0], 0x8F1BBCDCU, 14);
    RMD_STEP(d2,e2,a2,b2,c2, RMD_F2, X[ 3], 0x7A6D76E9U, 14);
    RMD_STEP(c,d,e,a,b, RMD_F4, X[ 8], 0x8F1BBCDCU, 15);
    RMD_STEP(c2,d2,e2,a2,b2, RMD_F2, X[11], 0x7A6D76E9U, 14);
    RMD_STEP(b,c,d,e,a, RMD_F4, X[12], 0x8F1BBCDCU,  9);
    RMD_STEP(b2,c2,d2,e2,a2, RMD_F2, X[15], 0x7A6D76E9U,  6);
    RMD_STEP(a,b,c,d,e, RMD_F4, X[ 4], 0x8F1BBCDCU,  8);
    RMD_STEP(a2,b2,c2,d2,e2, RMD_F2, X[ 0], 0x7A6D76E9U, 14);
    RMD_STEP(e,a,b,c,d, RMD_F4, X[13], 0x8F1BBCDCU,  9);
    RMD_STEP(e2,a2,b2,c2,d2, RMD_F2, X[ 5], 0x7A6D76E9U,  6);
    RMD_STEP(d,e,a,b,c, RMD_F4, X[ 3], 0x8F1BBCDCU, 14);
    RMD_STEP(d2,e2,a2,b2,c2, RMD_F2, X[12], 0x7A6D76E9U,  9);
    RMD_STEP(c,d,e,a,b, RMD_F4, X[ 7], 0x8F1BBCDCU,  5);
    RMD_STEP(c2,d2,e2,a2,b2, RMD_F2, X[ 2], 0x7A6D76E9U, 12);
    RMD_STEP(b,c,d,e,a, RMD_F4, X[15], 0x8F1BBCDCU,  6);
    RMD_STEP(b2,c2,d2,e2,a2, RMD_F2, X[13], 0x7A6D76E9U,  9);
    RMD_STEP(a,b,c,d,e, RMD_F4, X[14], 0x8F1BBCDCU,  8);
    RMD_STEP(a2,b2,c2,d2,e2, RMD_F2, X[ 9], 0x7A6D76E9U, 12);
    RMD_STEP(e,a,b,c,d, RMD_F4, X[ 5], 0x8F1BBCDCU,  6);
    RMD_STEP(e2,a2,b2,c2,d2, RMD_F2, X[ 7], 0x7A6D76E9U,  5);
    RMD_STEP(d,e,a,b,c, RMD_F4, X[ 6], 0x8F1BBCDCU,  5);
    RMD_STEP(d2,e2,a2,b2,c2, RMD_F2, X[10], 0x7A6D76E9U, 15);
    RMD_STEP(c,d,e,a,b, RMD_F4, X[ 2], 0x8F1BBCDCU, 12);
    RMD_STEP(c2,d2,e2,a2,b2, RMD_F2, X[14], 0x7A6D76E9U,  8);
    RMD_STEP(b,c,d,e,a, RMD_F5, X[ 4], 0xA953FD4EU,  9);
    RMD_STEP(b2,c2,d2,e2,a2, RMD_F1, X[12], 0,  8);
    RMD_STEP(a,b,c,d,e, RMD_F5, X[ 0], 0xA953FD4EU, 15);
    RMD_STEP(a2,b2,c2,d2,e2, RMD_F1, X[15], 0,  5);
    RMD_STEP(e,a,b,c,d, RMD_F5, X[ 5], 0xA953FD4EU,  5);
    RMD_STEP(e2,a2,b2,c2,d2, RMD_F1, X[10], 0, 12);
    RMD_STEP(d,e,a,b,c, RMD_F5, X[ 9], 0xA953FD4EU, 11);
    RMD_STEP(d2,e2,a2,b2,c2, RMD_F1, X[ 4], 0,  9);
    RMD_STEP(c,d,e,a,b, RMD_F5, X[ 7], 0xA953FD4EU,  6);
    RMD_STEP(c2,d2,e2,a2,b2, RMD_F1, X[ 1], 0, 12);
    RMD_STEP(b,c,d,e,a, RMD_F5, X[12], 0xA953FD4EU,  8);
    RMD_STEP(b2,c2,d2,e2,a2, RMD_F1, X[ 5], 0,  5);
    RMD_STEP(a,b,c,d,e, RMD_F5, X[ 2], 0xA953FD4EU, 13);
    RMD_STEP(a2,b2,c2,d2,e2, RMD_F1, X[ 8], 0, 14);
    RMD_STEP(e,a,b,c,d, RMD_F5, X[10], 0xA953FD4EU, 12);
    RMD_STEP(e2,a2,b2,c2,d2, RMD_F1, X[ 7], 0,  6);
    RMD_STEP(d,e,a,b,c, RMD_F5, X[14], 0xA953FD4EU,  5);
    RMD_STEP(d2,e2,a2,b2,c2, RMD_F1, X[ 6], 0,  8);
    RMD_STEP(c,d,e,a,b, RMD_F5, X[ 1], 0xA953FD4EU, 12);
    RMD_STEP(c2,d2,e2,a2,b2, RMD_F1, X[ 2], 0, 13);
    RMD_STEP(b,c,d,e,a, RMD_F5, X[ 3], 0xA953FD4EU, 13);
    RMD_STEP(b2,c2,d2,e2,a2, RMD_F1, X[13], 0,  6);
    RMD_STEP(a,b,c,d,e, RMD_F5, X[ 8], 0xA953FD4EU, 14);
    RMD_STEP(a2,b2,c2,d2,e2, RMD_F1, X[14], 0,  5);
    RMD_STEP(e,a,b,c,d, RMD_F5, X[11], 0xA953FD4EU, 11);
    RMD_STEP(e2,a2,b2,c2,d2, RMD_F1, X[ 0], 0, 15);
    RMD_STEP(d,e,a,b,c, RMD_F5, X[ 6], 0xA953FD4EU,  8);
    RMD_STEP(d2,e2,a2,b2,c2, RMD_F1, X[ 3], 0, 13);
    RMD_STEP(c,d,e,a,b, RMD_F5, X[15], 0xA953FD4EU,  5);
    RMD_STEP(c2,d2,e2,a2,b2, RMD_F1, X[ 9], 0, 11);
    RMD_STEP(b,c,d,e,a, RMD_F5, X[13], 0xA953FD4EU,  6);
    RMD_STEP(b2,c2,d2,e2,a2, RMD_F1, X[11], 0, 11);

    const uint t = h1 + c + d2;
    h1 = h2 + d + e2;
    h2 = h3 + e + a2;
    h3 = h4 + a + b2;
    h4 = h0 + b + c2;
    h0 = t;

    // Output: 5 x 32-bit words, little-endian byte order
    out[ 0]=(uchar)h0; out[ 1]=(uchar)(h0>>8); out[ 2]=(uchar)(h0>>16); out[ 3]=(uchar)(h0>>24);
    out[ 4]=(uchar)h1; out[ 5]=(uchar)(h1>>8); out[ 6]=(uchar)(h1>>16); out[ 7]=(uchar)(h1>>24);
    out[ 8]=(uchar)h2; out[ 9]=(uchar)(h2>>8); out[10]=(uchar)(h2>>16); out[11]=(uchar)(h2>>24);
    out[12]=(uchar)h3; out[13]=(uchar)(h3>>8); out[14]=(uchar)(h3>>16); out[15]=(uchar)(h3>>24);
    out[16]=(uchar)h4; out[17]=(uchar)(h4>>8); out[18]=(uchar)(h4>>16); out[19]=(uchar)(h4>>24);
}
#undef RMD_STEP
#undef RMD_F1
#undef RMD_F2
#undef RMD_F3
#undef RMD_F4
#undef RMD_F5

// =============================================================================
// Hash160 -- SHA-256 then RIPEMD-160 (Bitcoin public key hash)
// =============================================================================

// Length dispatch: 33-byte (compressed) keys take the registers-only
// sha256_33; any other length (65-byte uncompressed keys) takes the generic
// sha256_oneshot. Callers pass a runtime length, so the dispatch must stay --
// dropping it would hash 65-byte keys wrongly.
inline void hash160_pubkey(thread const uchar* pubkey, uint pubkey_len, thread uchar out[20]) {
    uchar sha[32];
    if (pubkey_len == 33u) sha256_33(pubkey, sha);
    else sha256_oneshot(pubkey, pubkey_len, sha);
    ripemd160_32(sha, out);
}
