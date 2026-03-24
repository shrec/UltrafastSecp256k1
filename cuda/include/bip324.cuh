/* ============================================================================
 * UltrafastSecp256k1 -- BIP-324 Transport Primitives (CUDA Device)
 * ============================================================================
 * ChaCha20-Poly1305 AEAD batch kernels for GPU-accelerated BIP-324.
 * Each thread processes one independent packet (key, nonce, payload).
 *
 * Reference: BIP-324, RFC 8439
 * ============================================================================ */
#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>

namespace secp256k1 {
namespace cuda {
namespace bip324 {

static constexpr std::size_t BIP324_OVERHEAD = 3 + 16; // 3B header + 16B tag

/* ============================================================================
 * Byte helpers
 * ============================================================================ */

__device__ __forceinline__ std::uint32_t d_load32_le(const std::uint8_t* p) {
    return static_cast<std::uint32_t>(p[0])
         | (static_cast<std::uint32_t>(p[1]) << 8)
         | (static_cast<std::uint32_t>(p[2]) << 16)
         | (static_cast<std::uint32_t>(p[3]) << 24);
}

__device__ __forceinline__ void d_store32_le(std::uint8_t* p, std::uint32_t v) {
    p[0] = static_cast<std::uint8_t>(v);
    p[1] = static_cast<std::uint8_t>(v >> 8);
    p[2] = static_cast<std::uint8_t>(v >> 16);
    p[3] = static_cast<std::uint8_t>(v >> 24);
}

__device__ __forceinline__ std::uint64_t d_load64_le(const std::uint8_t* p) {
    return static_cast<std::uint64_t>(p[0])
         | (static_cast<std::uint64_t>(p[1]) << 8)
         | (static_cast<std::uint64_t>(p[2]) << 16)
         | (static_cast<std::uint64_t>(p[3]) << 24)
         | (static_cast<std::uint64_t>(p[4]) << 32)
         | (static_cast<std::uint64_t>(p[5]) << 40)
         | (static_cast<std::uint64_t>(p[6]) << 48)
         | (static_cast<std::uint64_t>(p[7]) << 56);
}

__device__ __forceinline__ void d_store64_le(std::uint8_t* p, std::uint64_t v) {
    for (int i = 0; i < 8; ++i)
        p[i] = static_cast<std::uint8_t>(v >> (i * 8));
}

/* ============================================================================
 * ChaCha20 (RFC 8439)
 * ============================================================================ */

__device__ inline void d_chacha20_quarter_round(
    std::uint32_t& a, std::uint32_t& b,
    std::uint32_t& c, std::uint32_t& d)
{
    a += b; d ^= a; d = __byte_perm(d, 0, 0x1032);       // rotl32(16)
    c += d; b ^= c; b = __funnelshift_l(b, b, 12);       // rotl32(12)
    a += b; d ^= a; d = __byte_perm(d, 0, 0x0321);       // rotl32(8)
    c += d; b ^= c; b = __funnelshift_l(b, b, 7);        // rotl32(7)
}

__device__ inline void d_chacha20_block(
    const std::uint32_t input[16], std::uint8_t output[64])
{
    std::uint32_t x[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) x[i] = input[i];

    #pragma unroll
    for (int i = 0; i < 10; ++i) {
        d_chacha20_quarter_round(x[0], x[4], x[ 8], x[12]);
        d_chacha20_quarter_round(x[1], x[5], x[ 9], x[13]);
        d_chacha20_quarter_round(x[2], x[6], x[10], x[14]);
        d_chacha20_quarter_round(x[3], x[7], x[11], x[15]);
        d_chacha20_quarter_round(x[0], x[5], x[10], x[15]);
        d_chacha20_quarter_round(x[1], x[6], x[11], x[12]);
        d_chacha20_quarter_round(x[2], x[7], x[ 8], x[13]);
        d_chacha20_quarter_round(x[3], x[4], x[ 9], x[14]);
    }

    #pragma unroll
    for (int i = 0; i < 16; ++i)
        d_store32_le(output + i * 4, x[i] + input[i]);
}

__device__ inline void d_chacha20_setup(
    std::uint32_t state[16],
    const std::uint8_t key[32],
    const std::uint8_t nonce[12],
    std::uint32_t counter)
{
    state[ 0] = 0x61707865u;
    state[ 1] = 0x3320646eu;
    state[ 2] = 0x79622d32u;
    state[ 3] = 0x6b206574u;
    for (int i = 0; i < 8; ++i) state[4 + i] = d_load32_le(key + i * 4);
    state[12] = counter;
    state[13] = d_load32_le(nonce);
    state[14] = d_load32_le(nonce + 4);
    state[15] = d_load32_le(nonce + 8);
}

/* ============================================================================
 * Poly1305 (RFC 8439) -- 5x26-bit limbs
 * ============================================================================ */

struct DevPoly1305 {
    std::uint32_t r[5], s[4], h[5];

    __device__ void init(const std::uint8_t key[32]) {
        std::uint32_t t0 = d_load32_le(key +  0) & 0x0FFFFFFFu;
        std::uint32_t t1 = d_load32_le(key +  4) & 0x0FFFFFFCu;
        std::uint32_t t2 = d_load32_le(key +  8) & 0x0FFFFFFCu;
        std::uint32_t t3 = d_load32_le(key + 12) & 0x0FFFFFFCu;

        r[0] =  t0                        & 0x3FFFFFF;
        r[1] = ((t0 >> 26) | (t1 <<  6)) & 0x3FFFFFF;
        r[2] = ((t1 >> 20) | (t2 << 12)) & 0x3FFFFFF;
        r[3] = ((t2 >> 14) | (t3 << 18)) & 0x3FFFFFF;
        r[4] =  (t3 >>  8);

        s[0] = d_load32_le(key + 16);
        s[1] = d_load32_le(key + 20);
        s[2] = d_load32_le(key + 24);
        s[3] = d_load32_le(key + 28);

        h[0] = h[1] = h[2] = h[3] = h[4] = 0;
    }

    __device__ void block(const std::uint8_t* msg, std::size_t len) {
        std::uint8_t buf[17];
        for (int i = 0; i < 17; ++i) buf[i] = 0;
        for (std::size_t i = 0; i < len; ++i) buf[i] = msg[i];
        buf[len] = 1;

        std::uint32_t t0 = d_load32_le(buf);
        std::uint32_t t1 = d_load32_le(buf + 4);
        std::uint32_t t2 = d_load32_le(buf + 8);
        std::uint32_t t3 = d_load32_le(buf + 12);
        std::uint32_t hibit = static_cast<std::uint32_t>(buf[16]);

        h[0] += t0 & 0x3FFFFFF;
        h[1] += ((t0 >> 26) | (t1 << 6)) & 0x3FFFFFF;
        h[2] += ((t1 >> 20) | (t2 << 12)) & 0x3FFFFFF;
        h[3] += ((t2 >> 14) | (t3 << 18)) & 0x3FFFFFF;
        h[4] += (t3 >> 8) | (hibit << 24);

        std::uint32_t r0 = r[0], r1 = r[1], r2 = r[2], r3 = r[3], r4 = r[4];
        std::uint32_t s1 = r1 * 5, s2 = r2 * 5, s3 = r3 * 5, s4 = r4 * 5;

        std::uint64_t d0 = (std::uint64_t)h[0]*r0 + (std::uint64_t)h[1]*s4
                         + (std::uint64_t)h[2]*s3 + (std::uint64_t)h[3]*s2
                         + (std::uint64_t)h[4]*s1;
        std::uint64_t d1 = (std::uint64_t)h[0]*r1 + (std::uint64_t)h[1]*r0
                         + (std::uint64_t)h[2]*s4 + (std::uint64_t)h[3]*s3
                         + (std::uint64_t)h[4]*s2;
        std::uint64_t d2 = (std::uint64_t)h[0]*r2 + (std::uint64_t)h[1]*r1
                         + (std::uint64_t)h[2]*r0 + (std::uint64_t)h[3]*s4
                         + (std::uint64_t)h[4]*s3;
        std::uint64_t d3 = (std::uint64_t)h[0]*r3 + (std::uint64_t)h[1]*r2
                         + (std::uint64_t)h[2]*r1 + (std::uint64_t)h[3]*r0
                         + (std::uint64_t)h[4]*s4;
        std::uint64_t d4 = (std::uint64_t)h[0]*r4 + (std::uint64_t)h[1]*r3
                         + (std::uint64_t)h[2]*r2 + (std::uint64_t)h[3]*r1
                         + (std::uint64_t)h[4]*r0;

        std::uint32_t c;
        c = (std::uint32_t)(d0 >> 26); h[0] = (std::uint32_t)d0 & 0x3FFFFFF;
        d1 += c; c = (std::uint32_t)(d1 >> 26); h[1] = (std::uint32_t)d1 & 0x3FFFFFF;
        d2 += c; c = (std::uint32_t)(d2 >> 26); h[2] = (std::uint32_t)d2 & 0x3FFFFFF;
        d3 += c; c = (std::uint32_t)(d3 >> 26); h[3] = (std::uint32_t)d3 & 0x3FFFFFF;
        d4 += c; c = (std::uint32_t)(d4 >> 26); h[4] = (std::uint32_t)d4 & 0x3FFFFFF;
        h[0] += c * 5; c = h[0] >> 26; h[0] &= 0x3FFFFFF;
        h[1] += c;
    }

    __device__ void finish(std::uint8_t tag[16]) {
        std::uint32_t c;
        c = h[1] >> 26; h[1] &= 0x3FFFFFF;
        h[2] += c; c = h[2] >> 26; h[2] &= 0x3FFFFFF;
        h[3] += c; c = h[3] >> 26; h[3] &= 0x3FFFFFF;
        h[4] += c; c = h[4] >> 26; h[4] &= 0x3FFFFFF;
        h[0] += c * 5; c = h[0] >> 26; h[0] &= 0x3FFFFFF;
        h[1] += c;

        std::uint32_t g[5];
        c = h[0] + 5; g[0] = c & 0x3FFFFFF; c >>= 26;
        c += h[1];    g[1] = c & 0x3FFFFFF; c >>= 26;
        c += h[2];    g[2] = c & 0x3FFFFFF; c >>= 26;
        c += h[3];    g[3] = c & 0x3FFFFFF; c >>= 26;
        c += h[4];    g[4] = c & 0x3FFFFFF; c >>= 26;

        std::uint32_t mask = ~(c - 1);
        for (int i = 0; i < 5; ++i)
            h[i] = (h[i] & ~mask) | (g[i] & mask);

        std::uint64_t f;
        f  = (std::uint64_t)h[0] | ((std::uint64_t)h[1] << 26);
        std::uint32_t h0 = (std::uint32_t)f;
        f  = (f >> 32) | ((std::uint64_t)h[2] << 20);
        std::uint32_t h1 = (std::uint32_t)f;
        f  = (f >> 32) | ((std::uint64_t)h[3] << 14);
        std::uint32_t h2 = (std::uint32_t)f;
        f  = (f >> 32) | ((std::uint64_t)h[4] <<  8);
        std::uint32_t h3 = (std::uint32_t)f;

        std::uint64_t t;
        t = (std::uint64_t)h0 + s[0];              h0 = (std::uint32_t)t;
        t = (std::uint64_t)h1 + s[1] + (t >> 32);  h1 = (std::uint32_t)t;
        t = (std::uint64_t)h2 + s[2] + (t >> 32);  h2 = (std::uint32_t)t;
        t = (std::uint64_t)h3 + s[3] + (t >> 32);  h3 = (std::uint32_t)t;

        d_store32_le(tag +  0, h0);
        d_store32_le(tag +  4, h1);
        d_store32_le(tag +  8, h2);
        d_store32_le(tag + 12, h3);
    }
};

/* ============================================================================
 * AEAD ChaCha20-Poly1305 (RFC 8439)
 * ============================================================================ */

__device__ inline void d_aead_encrypt(
    const std::uint8_t key[32],
    const std::uint8_t nonce[12],
    const std::uint8_t* plaintext,
    std::size_t plaintext_len,
    std::uint8_t* ciphertext,
    std::uint8_t tag[16])
{
    std::uint32_t state[16];
    d_chacha20_setup(state, key, nonce, 0);

    std::uint8_t poly_key[64];
    d_chacha20_block(state, poly_key);

    state[12] = 1;
    {
        std::uint8_t block[64];
        std::size_t offset = 0;
        while (offset < plaintext_len) {
            d_chacha20_block(state, block);
            state[12]++;
            std::size_t use = (plaintext_len - offset < 64)
                            ? (plaintext_len - offset) : 64;
            for (std::size_t j = 0; j < use; ++j)
                ciphertext[offset + j] = plaintext[offset + j] ^ block[j];
            offset += use;
        }
    }

    DevPoly1305 st;
    st.init(poly_key);

    std::size_t off = 0;
    while (off + 16 <= plaintext_len) {
        st.block(ciphertext + off, 16);
        off += 16;
    }
    if (off < plaintext_len)
        st.block(ciphertext + off, plaintext_len - off);

    std::size_t ct_pad = (16 - (plaintext_len % 16)) % 16;
    if (ct_pad > 0) {
        std::uint8_t zeros[16] = {};
        st.block(zeros, ct_pad);
    }

    std::uint8_t lens[16];
    d_store64_le(lens, 0);
    d_store64_le(lens + 8, static_cast<std::uint64_t>(plaintext_len));
    st.block(lens, 16);
    st.finish(tag);
}

__device__ inline bool d_aead_decrypt(
    const std::uint8_t key[32],
    const std::uint8_t nonce[12],
    const std::uint8_t* ciphertext,
    std::size_t ciphertext_len,
    const std::uint8_t expected_tag[16],
    std::uint8_t* plaintext)
{
    std::uint32_t state[16];
    d_chacha20_setup(state, key, nonce, 0);

    std::uint8_t poly_key[64];
    d_chacha20_block(state, poly_key);

    DevPoly1305 st;
    st.init(poly_key);

    std::size_t off = 0;
    while (off + 16 <= ciphertext_len) {
        st.block(ciphertext + off, 16);
        off += 16;
    }
    if (off < ciphertext_len)
        st.block(ciphertext + off, ciphertext_len - off);

    std::size_t ct_pad = (16 - (ciphertext_len % 16)) % 16;
    if (ct_pad > 0) {
        std::uint8_t zeros[16] = {};
        st.block(zeros, ct_pad);
    }

    std::uint8_t lens[16];
    d_store64_le(lens, 0);
    d_store64_le(lens + 8, static_cast<std::uint64_t>(ciphertext_len));
    st.block(lens, 16);

    std::uint8_t computed[16];
    st.finish(computed);

    std::uint8_t diff = 0;
    for (int i = 0; i < 16; ++i) diff |= computed[i] ^ expected_tag[i];

    if (diff != 0) return false;

    state[12] = 1;
    {
        std::uint8_t block[64];
        std::size_t offset = 0;
        while (offset < ciphertext_len) {
            d_chacha20_block(state, block);
            state[12]++;
            std::size_t use = (ciphertext_len - offset < 64)
                            ? (ciphertext_len - offset) : 64;
            for (std::size_t j = 0; j < use; ++j)
                plaintext[offset + j] = ciphertext[offset + j] ^ block[j];
            offset += use;
        }
    }
    return true;
}

/* ============================================================================
 * Batch kernels
 * ============================================================================ */

__global__ void bip324_aead_encrypt_kernel(
    const std::uint8_t* __restrict__ d_keys,       // N * 32
    const std::uint8_t* __restrict__ d_nonces,     // N * 12
    const std::uint8_t* __restrict__ d_plaintexts, // N * max_payload
    const std::uint32_t* __restrict__ d_sizes,     // N payload sizes
    std::uint8_t* __restrict__ d_wire_out,         // N * (max_payload + BIP324_OVERHEAD)
    std::uint32_t max_payload,
    int count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    std::uint32_t payload_sz = d_sizes[idx];
    const std::uint8_t* key = d_keys + idx * 32;
    const std::uint8_t* nonce = d_nonces + idx * 12;
    const std::uint8_t* pt = d_plaintexts + (std::size_t)idx * max_payload;

    std::size_t wire_stride = max_payload + BIP324_OVERHEAD;
    std::uint8_t* wire = d_wire_out + (std::size_t)idx * wire_stride;

    // 3-byte length header
    wire[0] = static_cast<std::uint8_t>(payload_sz);
    wire[1] = static_cast<std::uint8_t>(payload_sz >> 8);
    wire[2] = static_cast<std::uint8_t>(payload_sz >> 16);

    d_aead_encrypt(key, nonce, pt, payload_sz,
                   wire + 3, wire + 3 + payload_sz);
}

__global__ void bip324_aead_decrypt_kernel(
    const std::uint8_t* __restrict__ d_keys,
    const std::uint8_t* __restrict__ d_nonces,
    const std::uint8_t* __restrict__ d_wire_in,
    const std::uint32_t* __restrict__ d_sizes,
    std::uint8_t* __restrict__ d_plaintext_out,
    std::uint32_t* __restrict__ d_ok,
    std::uint32_t max_payload,
    int count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    std::uint32_t payload_sz = d_sizes[idx];
    const std::uint8_t* key = d_keys + idx * 32;
    const std::uint8_t* nonce = d_nonces + idx * 12;

    std::size_t wire_stride = max_payload + BIP324_OVERHEAD;
    const std::uint8_t* wire = d_wire_in + (std::size_t)idx * wire_stride;
    std::uint8_t* pt_out = d_plaintext_out + (std::size_t)idx * max_payload;

    const std::uint8_t* ct = wire + 3;
    const std::uint8_t* tag = wire + 3 + payload_sz;

    bool ok = d_aead_decrypt(key, nonce, ct, payload_sz, tag, pt_out);
    d_ok[idx] = ok ? 1 : 0;
}

} // namespace bip324
} // namespace cuda
} // namespace secp256k1
