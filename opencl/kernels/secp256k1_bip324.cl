// =============================================================================
// secp256k1_bip324.cl -- BIP-324 Transport Layer for OpenCL
// =============================================================================
// GPU-accelerated BIP-324 v2 transport packet processing:
//   - ChaCha20 block cipher (RFC 8439)
//   - Poly1305 MAC authenticator
//   - ChaCha20-Poly1305 AEAD encrypt / decrypt
//   - Batch kernels for parallel packet processing
//
// Each work-item processes one independent packet (simulating parallel
// peer connections).
//
// Reference: BIP-324, RFC 8439
// =============================================================================

#ifndef SECP256K1_BIP324_CL
#define SECP256K1_BIP324_CL

// ─── Little-endian load / store helpers ─────────────────────────────────────

inline uint load32_le(const __global uchar* p) {
    return (uint)p[0]
         | ((uint)p[1] << 8)
         | ((uint)p[2] << 16)
         | ((uint)p[3] << 24);
}

inline uint load32_le_priv(const uchar* p) {
    return (uint)p[0]
         | ((uint)p[1] << 8)
         | ((uint)p[2] << 16)
         | ((uint)p[3] << 24);
}

inline void store32_le(__global uchar* p, uint v) {
    p[0] = (uchar)(v);
    p[1] = (uchar)(v >> 8);
    p[2] = (uchar)(v >> 16);
    p[3] = (uchar)(v >> 24);
}

inline void store32_le_priv(uchar* p, uint v) {
    p[0] = (uchar)(v);
    p[1] = (uchar)(v >> 8);
    p[2] = (uchar)(v >> 16);
    p[3] = (uchar)(v >> 24);
}

inline ulong load64_le_priv(const uchar* p) {
    return (ulong)p[0]
         | ((ulong)p[1] << 8)
         | ((ulong)p[2] << 16)
         | ((ulong)p[3] << 24)
         | ((ulong)p[4] << 32)
         | ((ulong)p[5] << 40)
         | ((ulong)p[6] << 48)
         | ((ulong)p[7] << 56);
}

inline void store64_le_priv(uchar* p, ulong v) {
    for (int i = 0; i < 8; ++i)
        p[i] = (uchar)(v >> (i * 8));
}

inline uint rotl32(uint v, int n) {
    return (v << n) | (v >> (32 - n));
}

// ─── ChaCha20 (RFC 8439) ───────────────────────────────────────────────────

inline void chacha20_quarter_round(uint* a, uint* b, uint* c, uint* d) {
    *a += *b; *d ^= *a; *d = rotl32(*d, 16);
    *c += *d; *b ^= *c; *b = rotl32(*b, 12);
    *a += *b; *d ^= *a; *d = rotl32(*d, 8);
    *c += *d; *b ^= *c; *b = rotl32(*b, 7);
}

inline void chacha20_block(const uint input[16], uchar output[64]) {
    uint x[16];
    for (int i = 0; i < 16; ++i) x[i] = input[i];

    for (int i = 0; i < 10; ++i) {
        chacha20_quarter_round(&x[0], &x[4], &x[ 8], &x[12]);
        chacha20_quarter_round(&x[1], &x[5], &x[ 9], &x[13]);
        chacha20_quarter_round(&x[2], &x[6], &x[10], &x[14]);
        chacha20_quarter_round(&x[3], &x[7], &x[11], &x[15]);
        chacha20_quarter_round(&x[0], &x[5], &x[10], &x[15]);
        chacha20_quarter_round(&x[1], &x[6], &x[11], &x[12]);
        chacha20_quarter_round(&x[2], &x[7], &x[ 8], &x[13]);
        chacha20_quarter_round(&x[3], &x[4], &x[ 9], &x[14]);
    }

    for (int i = 0; i < 16; ++i)
        store32_le_priv(output + i * 4, x[i] + input[i]);
}

inline void chacha20_setup(uint state[16],
                            const __global uchar* key,
                            const __global uchar* nonce,
                            uint counter) {
    state[ 0] = 0x61707865u;
    state[ 1] = 0x3320646eu;
    state[ 2] = 0x79622d32u;
    state[ 3] = 0x6b206574u;
    for (int i = 0; i < 8; ++i)
        state[4 + i] = load32_le(key + i * 4);
    state[12] = counter;
    state[13] = load32_le(nonce);
    state[14] = load32_le(nonce + 4);
    state[15] = load32_le(nonce + 8);
}

// ─── Poly1305 (RFC 8439) — 5×26-bit limbs ──────────────────────────────────

typedef struct {
    uint r[5];
    uint s[4];
    uint h[5];
} Poly1305State;

inline void poly1305_init(Poly1305State* st, const uchar key[32]) {
    uint t0 = load32_le_priv(key +  0) & 0x0FFFFFFFu;
    uint t1 = load32_le_priv(key +  4) & 0x0FFFFFFCu;
    uint t2 = load32_le_priv(key +  8) & 0x0FFFFFFCu;
    uint t3 = load32_le_priv(key + 12) & 0x0FFFFFFCu;

    st->r[0] =  t0                        & 0x3FFFFFF;
    st->r[1] = ((t0 >> 26) | (t1 <<  6)) & 0x3FFFFFF;
    st->r[2] = ((t1 >> 20) | (t2 << 12)) & 0x3FFFFFF;
    st->r[3] = ((t2 >> 14) | (t3 << 18)) & 0x3FFFFFF;
    st->r[4] =  (t3 >>  8);

    st->s[0] = load32_le_priv(key + 16);
    st->s[1] = load32_le_priv(key + 20);
    st->s[2] = load32_le_priv(key + 24);
    st->s[3] = load32_le_priv(key + 28);

    st->h[0] = st->h[1] = st->h[2] = st->h[3] = st->h[4] = 0;
}

inline void poly1305_block(Poly1305State* st, const uchar* msg, uint len) {
    uchar buf[17];
    for (int i = 0; i < 17; ++i) buf[i] = 0;
    for (uint i = 0; i < len; ++i) buf[i] = msg[i];
    buf[len] = 1;

    uint t0 = load32_le_priv(buf);
    uint t1 = load32_le_priv(buf + 4);
    uint t2 = load32_le_priv(buf + 8);
    uint t3 = load32_le_priv(buf + 12);
    uint hibit = (uint)buf[16];

    st->h[0] += t0 & 0x3FFFFFF;
    st->h[1] += ((t0 >> 26) | (t1 << 6)) & 0x3FFFFFF;
    st->h[2] += ((t1 >> 20) | (t2 << 12)) & 0x3FFFFFF;
    st->h[3] += ((t2 >> 14) | (t3 << 18)) & 0x3FFFFFF;
    st->h[4] += (t3 >> 8) | (hibit << 24);

    uint r0 = st->r[0], r1 = st->r[1], r2 = st->r[2], r3 = st->r[3], r4 = st->r[4];
    uint s1 = r1 * 5, s2 = r2 * 5, s3 = r3 * 5, s4 = r4 * 5;

    ulong d0 = (ulong)st->h[0]*r0 + (ulong)st->h[1]*s4
             + (ulong)st->h[2]*s3 + (ulong)st->h[3]*s2
             + (ulong)st->h[4]*s1;
    ulong d1 = (ulong)st->h[0]*r1 + (ulong)st->h[1]*r0
             + (ulong)st->h[2]*s4 + (ulong)st->h[3]*s3
             + (ulong)st->h[4]*s2;
    ulong d2 = (ulong)st->h[0]*r2 + (ulong)st->h[1]*r1
             + (ulong)st->h[2]*r0 + (ulong)st->h[3]*s4
             + (ulong)st->h[4]*s3;
    ulong d3 = (ulong)st->h[0]*r3 + (ulong)st->h[1]*r2
             + (ulong)st->h[2]*r1 + (ulong)st->h[3]*r0
             + (ulong)st->h[4]*s4;
    ulong d4 = (ulong)st->h[0]*r4 + (ulong)st->h[1]*r3
             + (ulong)st->h[2]*r2 + (ulong)st->h[3]*r1
             + (ulong)st->h[4]*r0;

    uint c;
    c = (uint)(d0 >> 26); st->h[0] = (uint)d0 & 0x3FFFFFF;
    d1 += c; c = (uint)(d1 >> 26); st->h[1] = (uint)d1 & 0x3FFFFFF;
    d2 += c; c = (uint)(d2 >> 26); st->h[2] = (uint)d2 & 0x3FFFFFF;
    d3 += c; c = (uint)(d3 >> 26); st->h[3] = (uint)d3 & 0x3FFFFFF;
    d4 += c; c = (uint)(d4 >> 26); st->h[4] = (uint)d4 & 0x3FFFFFF;
    st->h[0] += c * 5; c = st->h[0] >> 26; st->h[0] &= 0x3FFFFFF;
    st->h[1] += c;
}

inline void poly1305_block_global(Poly1305State* st, const __global uchar* msg, uint len) {
    uchar buf[17];
    for (int i = 0; i < 17; ++i) buf[i] = 0;
    for (uint i = 0; i < len; ++i) buf[i] = msg[i];
    buf[len] = 1;

    uint t0 = load32_le_priv(buf);
    uint t1 = load32_le_priv(buf + 4);
    uint t2 = load32_le_priv(buf + 8);
    uint t3 = load32_le_priv(buf + 12);
    uint hibit = (uint)buf[16];

    st->h[0] += t0 & 0x3FFFFFF;
    st->h[1] += ((t0 >> 26) | (t1 << 6)) & 0x3FFFFFF;
    st->h[2] += ((t1 >> 20) | (t2 << 12)) & 0x3FFFFFF;
    st->h[3] += ((t2 >> 14) | (t3 << 18)) & 0x3FFFFFF;
    st->h[4] += (t3 >> 8) | (hibit << 24);

    uint r0 = st->r[0], r1 = st->r[1], r2 = st->r[2], r3 = st->r[3], r4 = st->r[4];
    uint s1 = r1 * 5, s2 = r2 * 5, s3 = r3 * 5, s4 = r4 * 5;

    ulong d0 = (ulong)st->h[0]*r0 + (ulong)st->h[1]*s4
             + (ulong)st->h[2]*s3 + (ulong)st->h[3]*s2
             + (ulong)st->h[4]*s1;
    ulong d1 = (ulong)st->h[0]*r1 + (ulong)st->h[1]*r0
             + (ulong)st->h[2]*s4 + (ulong)st->h[3]*s3
             + (ulong)st->h[4]*s2;
    ulong d2 = (ulong)st->h[0]*r2 + (ulong)st->h[1]*r1
             + (ulong)st->h[2]*r0 + (ulong)st->h[3]*s4
             + (ulong)st->h[4]*s3;
    ulong d3 = (ulong)st->h[0]*r3 + (ulong)st->h[1]*r2
             + (ulong)st->h[2]*r1 + (ulong)st->h[3]*r0
             + (ulong)st->h[4]*s4;
    ulong d4 = (ulong)st->h[0]*r4 + (ulong)st->h[1]*r3
             + (ulong)st->h[2]*r2 + (ulong)st->h[3]*r1
             + (ulong)st->h[4]*r0;

    uint c;
    c = (uint)(d0 >> 26); st->h[0] = (uint)d0 & 0x3FFFFFF;
    d1 += c; c = (uint)(d1 >> 26); st->h[1] = (uint)d1 & 0x3FFFFFF;
    d2 += c; c = (uint)(d2 >> 26); st->h[2] = (uint)d2 & 0x3FFFFFF;
    d3 += c; c = (uint)(d3 >> 26); st->h[3] = (uint)d3 & 0x3FFFFFF;
    d4 += c; c = (uint)(d4 >> 26); st->h[4] = (uint)d4 & 0x3FFFFFF;
    st->h[0] += c * 5; c = st->h[0] >> 26; st->h[0] &= 0x3FFFFFF;
    st->h[1] += c;
}

inline void poly1305_finish(Poly1305State* st, uchar tag[16]) {
    uint c;
    c = st->h[1] >> 26; st->h[1] &= 0x3FFFFFF;
    st->h[2] += c; c = st->h[2] >> 26; st->h[2] &= 0x3FFFFFF;
    st->h[3] += c; c = st->h[3] >> 26; st->h[3] &= 0x3FFFFFF;
    st->h[4] += c; c = st->h[4] >> 26; st->h[4] &= 0x3FFFFFF;
    st->h[0] += c * 5; c = st->h[0] >> 26; st->h[0] &= 0x3FFFFFF;
    st->h[1] += c;

    uint g[5];
    c = st->h[0] + 5; g[0] = c & 0x3FFFFFF; c >>= 26;
    c += st->h[1];     g[1] = c & 0x3FFFFFF; c >>= 26;
    c += st->h[2];     g[2] = c & 0x3FFFFFF; c >>= 26;
    c += st->h[3];     g[3] = c & 0x3FFFFFF; c >>= 26;
    c += st->h[4];     g[4] = c & 0x3FFFFFF; c >>= 26;

    uint mask = ~(c - 1);
    for (int i = 0; i < 5; ++i)
        st->h[i] = (st->h[i] & ~mask) | (g[i] & mask);

    ulong f;
    f  = (ulong)st->h[0] | ((ulong)st->h[1] << 26);
    uint h0 = (uint)f;
    f  = (f >> 32) | ((ulong)st->h[2] << 20);
    uint h1 = (uint)f;
    f  = (f >> 32) | ((ulong)st->h[3] << 14);
    uint h2 = (uint)f;
    f  = (f >> 32) | ((ulong)st->h[4] <<  8);
    uint h3 = (uint)f;

    ulong t;
    t = (ulong)h0 + st->s[0];              h0 = (uint)t;
    t = (ulong)h1 + st->s[1] + (t >> 32);  h1 = (uint)t;
    t = (ulong)h2 + st->s[2] + (t >> 32);  h2 = (uint)t;
    t = (ulong)h3 + st->s[3] + (t >> 32);  h3 = (uint)t;

    store32_le_priv(tag +  0, h0);
    store32_le_priv(tag +  4, h1);
    store32_le_priv(tag +  8, h2);
    store32_le_priv(tag + 12, h3);
}

// ─── AEAD ChaCha20-Poly1305 (RFC 8439) ─────────────────────────────────────

inline void aead_encrypt(const __global uchar* key,
                          const __global uchar* nonce,
                          const __global uchar* plaintext,
                          uint plaintext_len,
                          __global uchar* ciphertext,
                          __global uchar* tag_out) {
    // 1. Setup state
    uint state[16];
    chacha20_setup(state, key, nonce, 0);

    // 2. Poly1305 key from block 0
    uchar poly_key[64];
    chacha20_block(state, poly_key);

    // 3. Encrypt (counter=1)
    state[12] = 1;
    {
        uchar block[64];
        uint offset = 0;
        while (offset < plaintext_len) {
            chacha20_block(state, block);
            state[12]++;
            uint use = (plaintext_len - offset < 64) ? (plaintext_len - offset) : 64;
            for (uint j = 0; j < use; ++j)
                ciphertext[offset + j] = plaintext[offset + j] ^ block[j];
            offset += use;
        }
    }

    // 4. Poly1305 MAC over (empty AAD || ciphertext || lengths)
    Poly1305State st;
    poly1305_init(&st, poly_key);

    uint off = 0;
    while (off + 16 <= plaintext_len) {
        poly1305_block_global(&st, ciphertext + off, 16);
        off += 16;
    }
    if (off < plaintext_len)
        poly1305_block_global(&st, ciphertext + off, plaintext_len - off);

    uint ct_pad = (16 - (plaintext_len % 16)) % 16;
    if (ct_pad > 0) {
        uchar zeros[16];
        for (int i = 0; i < 16; ++i) zeros[i] = 0;
        poly1305_block(&st, zeros, ct_pad);
    }

    uchar lens[16];
    store64_le_priv(lens, 0);  // AAD len = 0
    store64_le_priv(lens + 8, (ulong)plaintext_len);
    poly1305_block(&st, lens, 16);

    uchar tag[16];
    poly1305_finish(&st, tag);
    for (int i = 0; i < 16; ++i)
        tag_out[i] = tag[i];
}

inline int aead_decrypt(const __global uchar* key,
                         const __global uchar* nonce,
                         const __global uchar* ciphertext,
                         uint ciphertext_len,
                         const __global uchar* expected_tag,
                         __global uchar* plaintext) {
    // 1. Setup state
    uint state[16];
    chacha20_setup(state, key, nonce, 0);

    // 2. Poly1305 key from block 0
    uchar poly_key[64];
    chacha20_block(state, poly_key);

    // 3. Verify tag
    Poly1305State st;
    poly1305_init(&st, poly_key);

    uint off = 0;
    while (off + 16 <= ciphertext_len) {
        poly1305_block_global(&st, ciphertext + off, 16);
        off += 16;
    }
    if (off < ciphertext_len)
        poly1305_block_global(&st, ciphertext + off, ciphertext_len - off);

    uint ct_pad = (16 - (ciphertext_len % 16)) % 16;
    if (ct_pad > 0) {
        uchar zeros[16];
        for (int i = 0; i < 16; ++i) zeros[i] = 0;
        poly1305_block(&st, zeros, ct_pad);
    }

    uchar lens[16];
    store64_le_priv(lens, 0);
    store64_le_priv(lens + 8, (ulong)ciphertext_len);
    poly1305_block(&st, lens, 16);

    uchar computed[16];
    poly1305_finish(&st, computed);

    uchar diff = 0;
    for (int i = 0; i < 16; ++i) diff |= computed[i] ^ expected_tag[i];

    if (diff != 0) return 0;

    // 4. Decrypt (counter=1)
    state[12] = 1;
    {
        uchar block[64];
        uint offset = 0;
        while (offset < ciphertext_len) {
            chacha20_block(state, block);
            state[12]++;
            uint use = (ciphertext_len - offset < 64) ? (ciphertext_len - offset) : 64;
            for (uint j = 0; j < use; ++j)
                plaintext[offset + j] = ciphertext[offset + j] ^ block[j];
            offset += use;
        }
    }
    return 1;
}

// ─── BIP-324 packet constants ───────────────────────────────────────────────

#define BIP324_OVERHEAD 19  // 3B encrypted length + 16B tag


// =============================================================================
// Kernel: batch ChaCha20 block (throughput ceiling)
// =============================================================================

__kernel void kernel_bip324_chacha20_block_batch(
    __global const uchar* keys,    // N * 32
    __global const uchar* nonces,  // N * 12
    __global uchar* out,           // N * 64
    int count)
{
    int idx = get_global_id(0);
    if (idx >= count) return;

    uint state[16];
    chacha20_setup(state, keys + idx * 32, nonces + idx * 12, 0);

    uchar block[64];
    chacha20_block(state, block);

    for (int i = 0; i < 64; ++i)
        out[idx * 64 + i] = block[i];
}

// =============================================================================
// Kernel: batch AEAD encrypt
// =============================================================================
// Each work-item encrypts one BIP-324 packet.
// Input:  plaintexts (N * max_payload contiguous)
// Output: wire_out   (N * (max_payload + BIP324_OVERHEAD) contiguous)

__kernel void kernel_bip324_aead_encrypt(
    __global const uchar* keys,        // N * 32
    __global const uchar* nonces,      // N * 12
    __global const uchar* plaintexts,  // N * max_payload
    __global const uint*  sizes,       // N payload sizes
    __global uchar* wire_out,          // N * (max_payload + BIP324_OVERHEAD)
    uint max_payload,
    int count)
{
    int idx = get_global_id(0);
    if (idx >= count) return;

    uint payload_sz = sizes[idx];
    __global const uchar* key   = keys + idx * 32;
    __global const uchar* nonce = nonces + idx * 12;
    __global const uchar* pt    = plaintexts + (ulong)idx * max_payload;

    ulong wire_stride = max_payload + BIP324_OVERHEAD;
    __global uchar* wire = wire_out + (ulong)idx * wire_stride;

    // BIP-324: 3-byte LE length header (simplified for benchmark)
    wire[0] = (uchar)(payload_sz);
    wire[1] = (uchar)(payload_sz >> 8);
    wire[2] = (uchar)(payload_sz >> 16);

    // AEAD encrypt → ciphertext at wire+3, tag at wire+3+payload_sz
    aead_encrypt(key, nonce, pt, payload_sz,
                 wire + 3, wire + 3 + payload_sz);
}

// =============================================================================
// Kernel: batch AEAD decrypt
// =============================================================================

__kernel void kernel_bip324_aead_decrypt(
    __global const uchar* keys,
    __global const uchar* nonces,
    __global const uchar* wire_in,      // N * (max_payload + BIP324_OVERHEAD)
    __global const uint*  sizes,
    __global uchar* plaintext_out,      // N * max_payload
    __global uint*  ok,                 // N success flags
    uint max_payload,
    int count)
{
    int idx = get_global_id(0);
    if (idx >= count) return;

    uint payload_sz = sizes[idx];
    __global const uchar* key   = keys + idx * 32;
    __global const uchar* nonce = nonces + idx * 12;

    ulong wire_stride = max_payload + BIP324_OVERHEAD;
    __global const uchar* wire = wire_in + (ulong)idx * wire_stride;
    __global uchar* pt_out = plaintext_out + (ulong)idx * max_payload;

    __global const uchar* ct  = wire + 3;
    __global const uchar* tag = wire + 3 + payload_sz;

    ok[idx] = (uint)aead_decrypt(key, nonce, ct, payload_sz, tag, pt_out);
}

#endif // SECP256K1_BIP324_CL
