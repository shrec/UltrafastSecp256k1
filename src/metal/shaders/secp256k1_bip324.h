// =============================================================================
// secp256k1_bip324.h -- BIP-324 Transport Layer for Metal
// =============================================================================
// GPU-accelerated BIP-324 v2 transport packet processing:
//   - ChaCha20 block cipher (RFC 8439)
//   - Poly1305 MAC authenticator
//   - ChaCha20-Poly1305 AEAD encrypt / decrypt
//
// Self-contained — no dependency on secp256k1 field arithmetic.
// Uses 32-bit ops only — fully compatible with Apple Silicon.
//
// Reference: BIP-324, RFC 8439
// =============================================================================

#pragma once

#include <metal_stdlib>
using namespace metal;

// ─── Little-endian load / store helpers ─────────────────────────────────────

inline uint bip324_load32_le(thread const uchar* p) {
    return (uint)p[0]
         | ((uint)p[1] << 8)
         | ((uint)p[2] << 16)
         | ((uint)p[3] << 24);
}

inline uint bip324_load32_le_dev(device const uchar* p) {
    return (uint)p[0]
         | ((uint)p[1] << 8)
         | ((uint)p[2] << 16)
         | ((uint)p[3] << 24);
}

inline void bip324_store32_le(thread uchar* p, uint v) {
    p[0] = (uchar)(v);
    p[1] = (uchar)(v >> 8);
    p[2] = (uchar)(v >> 16);
    p[3] = (uchar)(v >> 24);
}

inline void bip324_store32_le_dev(device uchar* p, uint v) {
    p[0] = (uchar)(v);
    p[1] = (uchar)(v >> 8);
    p[2] = (uchar)(v >> 16);
    p[3] = (uchar)(v >> 24);
}

inline void bip324_store64_le(thread uchar* p, ulong v) {
    for (int i = 0; i < 8; ++i)
        p[i] = (uchar)(v >> (i * 8));
}

inline uint bip324_rotl32(uint v, int n) {
    return (v << n) | (v >> (32 - n));
}

// ─── ChaCha20 (RFC 8439) ───────────────────────────────────────────────────

inline void bip324_chacha20_qr(thread uint& a, thread uint& b,
                                thread uint& c, thread uint& d) {
    a += b; d ^= a; d = bip324_rotl32(d, 16);
    c += d; b ^= c; b = bip324_rotl32(b, 12);
    a += b; d ^= a; d = bip324_rotl32(d, 8);
    c += d; b ^= c; b = bip324_rotl32(b, 7);
}

inline void bip324_chacha20_block(thread const uint input[16],
                                   thread uchar output[64]) {
    uint x[16];
    for (int i = 0; i < 16; ++i) x[i] = input[i];

    for (int i = 0; i < 10; ++i) {
        bip324_chacha20_qr(x[0], x[4], x[ 8], x[12]);
        bip324_chacha20_qr(x[1], x[5], x[ 9], x[13]);
        bip324_chacha20_qr(x[2], x[6], x[10], x[14]);
        bip324_chacha20_qr(x[3], x[7], x[11], x[15]);
        bip324_chacha20_qr(x[0], x[5], x[10], x[15]);
        bip324_chacha20_qr(x[1], x[6], x[11], x[12]);
        bip324_chacha20_qr(x[2], x[7], x[ 8], x[13]);
        bip324_chacha20_qr(x[3], x[4], x[ 9], x[14]);
    }

    for (int i = 0; i < 16; ++i)
        bip324_store32_le(output + i * 4, x[i] + input[i]);
}

inline void bip324_chacha20_setup(thread uint state[16],
                                   device const uchar* key,
                                   device const uchar* nonce,
                                   uint counter) {
    state[ 0] = 0x61707865u;
    state[ 1] = 0x3320646eu;
    state[ 2] = 0x79622d32u;
    state[ 3] = 0x6b206574u;
    for (int i = 0; i < 8; ++i)
        state[4 + i] = bip324_load32_le_dev(key + i * 4);
    state[12] = counter;
    state[13] = bip324_load32_le_dev(nonce);
    state[14] = bip324_load32_le_dev(nonce + 4);
    state[15] = bip324_load32_le_dev(nonce + 8);
}

// ─── Poly1305 (RFC 8439) — 5×26-bit limbs ──────────────────────────────────

struct Bip324Poly1305 {
    uint r[5];
    uint s[4];
    uint h[5];
};

inline Bip324Poly1305 bip324_poly1305_init(thread const uchar key[32]) {
    Bip324Poly1305 st;

    uint t0 = bip324_load32_le(key +  0) & 0x0FFFFFFFu;
    uint t1 = bip324_load32_le(key +  4) & 0x0FFFFFFCu;
    uint t2 = bip324_load32_le(key +  8) & 0x0FFFFFFCu;
    uint t3 = bip324_load32_le(key + 12) & 0x0FFFFFFCu;

    st.r[0] =  t0                        & 0x3FFFFFF;
    st.r[1] = ((t0 >> 26) | (t1 <<  6)) & 0x3FFFFFF;
    st.r[2] = ((t1 >> 20) | (t2 << 12)) & 0x3FFFFFF;
    st.r[3] = ((t2 >> 14) | (t3 << 18)) & 0x3FFFFFF;
    st.r[4] =  (t3 >>  8);

    st.s[0] = bip324_load32_le(key + 16);
    st.s[1] = bip324_load32_le(key + 20);
    st.s[2] = bip324_load32_le(key + 24);
    st.s[3] = bip324_load32_le(key + 28);

    st.h[0] = st.h[1] = st.h[2] = st.h[3] = st.h[4] = 0;
    return st;
}

inline void bip324_poly1305_block(thread Bip324Poly1305& st,
                                   thread const uchar* msg, uint len) {
    uchar buf[17];
    for (int i = 0; i < 17; ++i) buf[i] = 0;
    for (uint i = 0; i < len; ++i) buf[i] = msg[i];
    buf[len] = 1;

    uint t0 = bip324_load32_le(buf);
    uint t1 = bip324_load32_le(buf + 4);
    uint t2 = bip324_load32_le(buf + 8);
    uint t3 = bip324_load32_le(buf + 12);
    uint hibit = (uint)buf[16];

    st.h[0] += t0 & 0x3FFFFFF;
    st.h[1] += ((t0 >> 26) | (t1 << 6)) & 0x3FFFFFF;
    st.h[2] += ((t1 >> 20) | (t2 << 12)) & 0x3FFFFFF;
    st.h[3] += ((t2 >> 14) | (t3 << 18)) & 0x3FFFFFF;
    st.h[4] += (t3 >> 8) | (hibit << 24);

    uint r0 = st.r[0], r1 = st.r[1], r2 = st.r[2], r3 = st.r[3], r4 = st.r[4];
    uint s1 = r1 * 5, s2 = r2 * 5, s3 = r3 * 5, s4 = r4 * 5;

    ulong d0 = (ulong)st.h[0]*r0 + (ulong)st.h[1]*s4
             + (ulong)st.h[2]*s3 + (ulong)st.h[3]*s2
             + (ulong)st.h[4]*s1;
    ulong d1 = (ulong)st.h[0]*r1 + (ulong)st.h[1]*r0
             + (ulong)st.h[2]*s4 + (ulong)st.h[3]*s3
             + (ulong)st.h[4]*s2;
    ulong d2 = (ulong)st.h[0]*r2 + (ulong)st.h[1]*r1
             + (ulong)st.h[2]*r0 + (ulong)st.h[3]*s4
             + (ulong)st.h[4]*s3;
    ulong d3 = (ulong)st.h[0]*r3 + (ulong)st.h[1]*r2
             + (ulong)st.h[2]*r1 + (ulong)st.h[3]*r0
             + (ulong)st.h[4]*s4;
    ulong d4 = (ulong)st.h[0]*r4 + (ulong)st.h[1]*r3
             + (ulong)st.h[2]*r2 + (ulong)st.h[3]*r1
             + (ulong)st.h[4]*r0;

    uint c;
    c = (uint)(d0 >> 26); st.h[0] = (uint)d0 & 0x3FFFFFF;
    d1 += c; c = (uint)(d1 >> 26); st.h[1] = (uint)d1 & 0x3FFFFFF;
    d2 += c; c = (uint)(d2 >> 26); st.h[2] = (uint)d2 & 0x3FFFFFF;
    d3 += c; c = (uint)(d3 >> 26); st.h[3] = (uint)d3 & 0x3FFFFFF;
    d4 += c; c = (uint)(d4 >> 26); st.h[4] = (uint)d4 & 0x3FFFFFF;
    st.h[0] += c * 5; c = st.h[0] >> 26; st.h[0] &= 0x3FFFFFF;
    st.h[1] += c;
}

inline void bip324_poly1305_block_dev(thread Bip324Poly1305& st,
                                       device const uchar* msg, uint len) {
    uchar buf[17];
    for (int i = 0; i < 17; ++i) buf[i] = 0;
    for (uint i = 0; i < len; ++i) buf[i] = msg[i];
    buf[len] = 1;

    uint t0 = bip324_load32_le(buf);
    uint t1 = bip324_load32_le(buf + 4);
    uint t2 = bip324_load32_le(buf + 8);
    uint t3 = bip324_load32_le(buf + 12);
    uint hibit = (uint)buf[16];

    st.h[0] += t0 & 0x3FFFFFF;
    st.h[1] += ((t0 >> 26) | (t1 << 6)) & 0x3FFFFFF;
    st.h[2] += ((t1 >> 20) | (t2 << 12)) & 0x3FFFFFF;
    st.h[3] += ((t2 >> 14) | (t3 << 18)) & 0x3FFFFFF;
    st.h[4] += (t3 >> 8) | (hibit << 24);

    uint r0 = st.r[0], r1 = st.r[1], r2 = st.r[2], r3 = st.r[3], r4 = st.r[4];
    uint s1 = r1 * 5, s2 = r2 * 5, s3 = r3 * 5, s4 = r4 * 5;

    ulong d0 = (ulong)st.h[0]*r0 + (ulong)st.h[1]*s4
             + (ulong)st.h[2]*s3 + (ulong)st.h[3]*s2
             + (ulong)st.h[4]*s1;
    ulong d1 = (ulong)st.h[0]*r1 + (ulong)st.h[1]*r0
             + (ulong)st.h[2]*s4 + (ulong)st.h[3]*s3
             + (ulong)st.h[4]*s2;
    ulong d2 = (ulong)st.h[0]*r2 + (ulong)st.h[1]*r1
             + (ulong)st.h[2]*r0 + (ulong)st.h[3]*s4
             + (ulong)st.h[4]*s3;
    ulong d3 = (ulong)st.h[0]*r3 + (ulong)st.h[1]*r2
             + (ulong)st.h[2]*r1 + (ulong)st.h[3]*r0
             + (ulong)st.h[4]*s4;
    ulong d4 = (ulong)st.h[0]*r4 + (ulong)st.h[1]*r3
             + (ulong)st.h[2]*r2 + (ulong)st.h[3]*r1
             + (ulong)st.h[4]*r0;

    uint c;
    c = (uint)(d0 >> 26); st.h[0] = (uint)d0 & 0x3FFFFFF;
    d1 += c; c = (uint)(d1 >> 26); st.h[1] = (uint)d1 & 0x3FFFFFF;
    d2 += c; c = (uint)(d2 >> 26); st.h[2] = (uint)d2 & 0x3FFFFFF;
    d3 += c; c = (uint)(d3 >> 26); st.h[3] = (uint)d3 & 0x3FFFFFF;
    d4 += c; c = (uint)(d4 >> 26); st.h[4] = (uint)d4 & 0x3FFFFFF;
    st.h[0] += c * 5; c = st.h[0] >> 26; st.h[0] &= 0x3FFFFFF;
    st.h[1] += c;
}

inline void bip324_poly1305_finish(thread Bip324Poly1305& st, thread uchar tag[16]) {
    uint c;
    c = st.h[1] >> 26; st.h[1] &= 0x3FFFFFF;
    st.h[2] += c; c = st.h[2] >> 26; st.h[2] &= 0x3FFFFFF;
    st.h[3] += c; c = st.h[3] >> 26; st.h[3] &= 0x3FFFFFF;
    st.h[4] += c; c = st.h[4] >> 26; st.h[4] &= 0x3FFFFFF;
    st.h[0] += c * 5; c = st.h[0] >> 26; st.h[0] &= 0x3FFFFFF;
    st.h[1] += c;

    uint g[5];
    c = st.h[0] + 5; g[0] = c & 0x3FFFFFF; c >>= 26;
    c += st.h[1];     g[1] = c & 0x3FFFFFF; c >>= 26;
    c += st.h[2];     g[2] = c & 0x3FFFFFF; c >>= 26;
    c += st.h[3];     g[3] = c & 0x3FFFFFF; c >>= 26;
    c += st.h[4];     g[4] = c & 0x3FFFFFF; c >>= 26;

    uint mask = ~(c - 1);
    for (int i = 0; i < 5; ++i)
        st.h[i] = (st.h[i] & ~mask) | (g[i] & mask);

    ulong f;
    f  = (ulong)st.h[0] | ((ulong)st.h[1] << 26);
    uint h0 = (uint)f;
    f  = (f >> 32) | ((ulong)st.h[2] << 20);
    uint h1 = (uint)f;
    f  = (f >> 32) | ((ulong)st.h[3] << 14);
    uint h2 = (uint)f;
    f  = (f >> 32) | ((ulong)st.h[4] <<  8);
    uint h3 = (uint)f;

    ulong t;
    t = (ulong)h0 + st.s[0];              h0 = (uint)t;
    t = (ulong)h1 + st.s[1] + (t >> 32);  h1 = (uint)t;
    t = (ulong)h2 + st.s[2] + (t >> 32);  h2 = (uint)t;
    t = (ulong)h3 + st.s[3] + (t >> 32);  h3 = (uint)t;

    bip324_store32_le(tag +  0, h0);
    bip324_store32_le(tag +  4, h1);
    bip324_store32_le(tag +  8, h2);
    bip324_store32_le(tag + 12, h3);
}

// ─── AEAD ChaCha20-Poly1305 (RFC 8439) ─────────────────────────────────────

inline void bip324_aead_encrypt(device const uchar* key,
                                 device const uchar* nonce,
                                 device const uchar* plaintext,
                                 uint plaintext_len,
                                 device uchar* ciphertext,
                                 device uchar* tag_out) {
    // 1. Setup state
    uint state[16];
    bip324_chacha20_setup(state, key, nonce, 0);

    // 2. Poly1305 key from block 0
    uchar poly_key[64];
    bip324_chacha20_block(state, poly_key);

    // 3. Encrypt (counter=1)
    state[12] = 1;
    {
        uchar block[64];
        uint offset = 0;
        while (offset < plaintext_len) {
            bip324_chacha20_block(state, block);
            state[12]++;
            uint use = (plaintext_len - offset < 64) ? (plaintext_len - offset) : 64;
            for (uint j = 0; j < use; ++j)
                ciphertext[offset + j] = plaintext[offset + j] ^ block[j];
            offset += use;
        }
    }

    // 4. Poly1305 MAC over (empty AAD || ciphertext || lengths)
    Bip324Poly1305 st = bip324_poly1305_init(poly_key);

    uint off = 0;
    while (off + 16 <= plaintext_len) {
        bip324_poly1305_block_dev(st, ciphertext + off, 16);
        off += 16;
    }
    if (off < plaintext_len)
        bip324_poly1305_block_dev(st, ciphertext + off, plaintext_len - off);

    uint ct_pad = (16 - (plaintext_len % 16)) % 16;
    if (ct_pad > 0) {
        uchar zeros[16] = {};
        bip324_poly1305_block(st, zeros, ct_pad);
    }

    uchar lens[16];
    bip324_store64_le(lens, 0);
    bip324_store64_le(lens + 8, (ulong)plaintext_len);
    bip324_poly1305_block(st, lens, 16);

    uchar tag[16];
    bip324_poly1305_finish(st, tag);
    for (int i = 0; i < 16; ++i)
        tag_out[i] = tag[i];
}

inline bool bip324_aead_decrypt(device const uchar* key,
                                 device const uchar* nonce,
                                 device const uchar* ciphertext,
                                 uint ciphertext_len,
                                 device const uchar* expected_tag,
                                 device uchar* plaintext) {
    // 1. Setup state
    uint state[16];
    bip324_chacha20_setup(state, key, nonce, 0);

    // 2. Poly1305 key from block 0
    uchar poly_key[64];
    bip324_chacha20_block(state, poly_key);

    // 3. Verify tag
    Bip324Poly1305 st = bip324_poly1305_init(poly_key);

    uint off = 0;
    while (off + 16 <= ciphertext_len) {
        bip324_poly1305_block_dev(st, ciphertext + off, 16);
        off += 16;
    }
    if (off < ciphertext_len)
        bip324_poly1305_block_dev(st, ciphertext + off, ciphertext_len - off);

    uint ct_pad = (16 - (ciphertext_len % 16)) % 16;
    if (ct_pad > 0) {
        uchar zeros[16] = {};
        bip324_poly1305_block(st, zeros, ct_pad);
    }

    uchar lens[16];
    bip324_store64_le(lens, 0);
    bip324_store64_le(lens + 8, (ulong)ciphertext_len);
    bip324_poly1305_block(st, lens, 16);

    uchar computed[16];
    bip324_poly1305_finish(st, computed);

    uchar diff = 0;
    for (int i = 0; i < 16; ++i) diff |= computed[i] ^ expected_tag[i];

    if (diff != 0) return false;

    // 4. Decrypt (counter=1)
    state[12] = 1;
    {
        uchar block[64];
        uint offset = 0;
        while (offset < ciphertext_len) {
            bip324_chacha20_block(state, block);
            state[12]++;
            uint use = (ciphertext_len - offset < 64) ? (ciphertext_len - offset) : 64;
            for (uint j = 0; j < use; ++j)
                plaintext[offset + j] = ciphertext[offset + j] ^ block[j];
            offset += use;
        }
    }
    return true;
}

// ─── BIP-324 constants ──────────────────────────────────────────────────────

constant uint BIP324_OVERHEAD = 19;  // 3B encrypted length + 16B tag
