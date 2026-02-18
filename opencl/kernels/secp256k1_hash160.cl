// =============================================================================
// UltrafastSecp256k1 OpenCL — Hash160 (SHA-256 → RIPEMD-160)
// =============================================================================
// Self-contained one-shot SHA-256 + RIPEMD-160 for public key hashing.
// Matches CUDA hash160.cuh output byte-for-byte.
//
// Functions:
//   sha256_oneshot_impl  — One-shot SHA-256 (up to 128 bytes input)
//   ripemd_f_impl        — RIPEMD-160 round function selector (5 variants)
//   ripemd160_32_impl    — RIPEMD-160 optimized for exactly 32-byte input
//   hash160_pubkey_impl  — SHA-256 then RIPEMD-160 (Bitcoin Hash160)
//
// Kernel:
//   hash160_batch        — Batch Hash160 of compressed/uncompressed pubkeys
//
// NOTE: Uses its own one-shot SHA-256 (not the streaming SHA256Ctx from
//       secp256k1_extended.cl) — more efficient for fixed-size inputs
//       since it avoids state tracking overhead.
// =============================================================================

#ifndef SECP256K1_HASH160_CL
#define SECP256K1_HASH160_CL

// =============================================================================
// SHA-256 Constants
// =============================================================================

__constant uint SHA256_K[64] = {
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

// =============================================================================
// SHA-256 Helpers
// =============================================================================

inline uint hash_rotr32(uint x, uint n) { return (x >> n) | (x << (32 - n)); }
inline uint hash_rotl32(uint x, uint n) { return (x << n) | (x >> (32 - n)); }

// =============================================================================
// One-shot SHA-256 (up to 128 bytes input, max 2 blocks)
// =============================================================================
// Output: 32 bytes, big-endian (standard SHA-256 byte order)

inline void sha256_oneshot_impl(const uchar* data, uint len, uchar out[32]) {
    uint h0 = 0x6a09e667U;
    uint h1 = 0xbb67ae85U;
    uint h2 = 0x3c6ef372U;
    uint h3 = 0xa54ff53aU;
    uint h4 = 0x510e527fU;
    uint h5 = 0x9b05688cU;
    uint h6 = 0x1f83d9abU;
    uint h7 = 0x5be0cd19U;

    // Prepare padded block(s) — max 128 bytes (2 blocks)
    uchar block[128];
    for (int i = 0; i < 128; ++i) block[i] = 0;
    for (uint i = 0; i < len; ++i) block[i] = data[i];
    block[len] = 0x80;

    const ulong bit_len = (ulong)len * 8UL;
    const int total_blocks = ((len + 1 + 8) <= 64) ? 1 : 2;
    const int last = total_blocks * 64 - 8;
    block[last + 0] = (uchar)(bit_len >> 56);
    block[last + 1] = (uchar)(bit_len >> 48);
    block[last + 2] = (uchar)(bit_len >> 40);
    block[last + 3] = (uchar)(bit_len >> 32);
    block[last + 4] = (uchar)(bit_len >> 24);
    block[last + 5] = (uchar)(bit_len >> 16);
    block[last + 6] = (uchar)(bit_len >> 8);
    block[last + 7] = (uchar)(bit_len);

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
            const uint s0 = hash_rotr32(w[i - 15], 7) ^ hash_rotr32(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const uint s1 = hash_rotr32(w[i - 2], 17) ^ hash_rotr32(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }

        uint a = h0, b0 = h1, c = h2, d = h3;
        uint e = h4, f = h5, g = h6, h = h7;

        for (int i = 0; i < 64; ++i) {
            const uint S1 = hash_rotr32(e, 6) ^ hash_rotr32(e, 11) ^ hash_rotr32(e, 25);
            const uint ch = (e & f) ^ (~e & g);
            const uint temp1 = h + S1 + ch + SHA256_K[i] + w[i];
            const uint S0 = hash_rotr32(a, 2) ^ hash_rotr32(a, 13) ^ hash_rotr32(a, 22);
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

// Message schedule indices — left chain
__constant uchar RIPEMD_R[80] = {
    0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
    7,4,13,1,10,6,15,3,12,0,9,5,2,14,11,8,
    3,10,14,4,9,15,8,1,2,7,0,6,13,11,5,12,
    1,9,11,10,0,8,12,4,13,3,7,15,14,5,6,2,
    4,0,5,9,7,12,2,10,14,1,3,8,11,6,15,13
};

// Message schedule indices — right chain
__constant uchar RIPEMD_R2[80] = {
    5,14,7,0,9,2,11,4,13,6,15,8,1,10,3,12,
    6,11,3,7,0,13,5,10,14,15,8,12,4,9,1,2,
    15,5,1,3,7,14,6,9,11,8,12,2,10,0,13,4,
    8,6,4,1,3,11,15,0,5,12,2,13,9,7,10,14,
    12,15,10,4,1,5,8,7,6,2,13,14,0,3,9,11
};

// Shift amounts — left chain
__constant uchar RIPEMD_S[80] = {
    11,14,15,12,5,8,7,9,11,13,14,15,6,7,9,8,
    7,6,8,13,11,9,7,15,7,12,15,9,11,7,13,12,
    11,13,6,7,14,9,13,15,14,8,13,6,5,12,7,5,
    11,12,14,15,14,15,9,8,9,14,5,6,8,6,5,12,
    9,15,5,11,6,8,13,12,5,12,13,14,11,8,5,6
};

// Shift amounts — right chain
__constant uchar RIPEMD_S2[80] = {
    8,9,9,11,13,15,15,5,7,7,8,11,14,14,12,6,
    9,13,15,7,12,8,9,11,7,7,12,7,6,15,13,11,
    9,7,15,11,8,6,6,14,12,13,5,14,13,13,7,5,
    15,5,8,11,14,14,6,14,6,9,12,9,12,5,15,8,
    8,5,12,9,12,5,14,6,8,13,6,5,15,13,11,11
};

// Round constants — left chain
__constant uint RIPEMD_K[5] = {
    0x00000000U, 0x5A827999U, 0x6ED9EBA1U, 0x8F1BBCDCU, 0xA953FD4EU
};

// Round constants — right chain
__constant uint RIPEMD_K2[5] = {
    0x50A28BE6U, 0x5C4DD124U, 0x6D703EF3U, 0x7A6D76E9U, 0x00000000U
};

// =============================================================================
// RIPEMD-160 Round Function Selector (5 boolean functions)
// =============================================================================

inline uint ripemd_f_impl(int j, uint x, uint y, uint z) {
    if (j <= 15) return x ^ y ^ z;           // XOR
    if (j <= 31) return (x & y) | (~x & z);  // IF
    if (j <= 47) return (x | ~y) ^ z;        // ONX
    if (j <= 63) return (x & z) | (y & ~z);  // IFX
    return x ^ (y | ~z);                      // OXN
}

// =============================================================================
// RIPEMD-160 — Optimized for exactly 32-byte input (single block)
// =============================================================================
// Input: 32 bytes (SHA-256 digest)
// Output: 20 bytes, little-endian (standard RIPEMD-160 byte order)

inline void ripemd160_32_impl(const uchar data[32], uchar out[20]) {
    // Pad to 64-byte block: data ∥ 0x80 ∥ zeros ∥ LE-bitlen
    uchar block[64];
    for (int i = 0; i < 64; ++i) block[i] = 0;
    for (int i = 0; i < 32; ++i) block[i] = data[i];
    block[32] = 0x80;
    // bit_len = 256 = 0x100 → LE: block[56]=0x00, block[57]=0x01
    const ulong bit_len = 32UL * 8UL;
    block[56] = (uchar)(bit_len);
    block[57] = (uchar)(bit_len >> 8);
    block[58] = (uchar)(bit_len >> 16);
    block[59] = (uchar)(bit_len >> 24);
    block[60] = (uchar)(bit_len >> 32);
    block[61] = (uchar)(bit_len >> 40);
    block[62] = (uchar)(bit_len >> 48);
    block[63] = (uchar)(bit_len >> 56);

    // Parse block into 16 little-endian 32-bit words
    uint X[16];
    for (int i = 0; i < 16; ++i) {
        const int j = i * 4;
        X[i] = (uint)block[j] |
               ((uint)block[j + 1] << 8) |
               ((uint)block[j + 2] << 16) |
               ((uint)block[j + 3] << 24);
    }

    // Initial hash values
    uint h0 = 0x67452301U;
    uint h1 = 0xEFCDAB89U;
    uint h2 = 0x98BADCFEU;
    uint h3 = 0x10325476U;
    uint h4 = 0xC3D2E1F0U;

    // Two parallel chains: left (a,b0,c,d,e) and right (a2,b2,c2,d2,e2)
    uint a = h0, b0 = h1, c = h2, d = h3, e = h4;
    uint a2 = h0, b2 = h1, c2 = h2, d2 = h3, e2 = h4;

    // 80 rounds — left and right chains in parallel
    for (int j = 0; j < 80; ++j) {
        const uint t = hash_rotl32(a + ripemd_f_impl(j, b0, c, d) +
                                   X[RIPEMD_R[j]] + RIPEMD_K[j / 16],
                                   RIPEMD_S[j]) + e;
        a = e; e = d; d = hash_rotl32(c, 10); c = b0; b0 = t;

        const uint t2 = hash_rotl32(a2 + ripemd_f_impl(79 - j, b2, c2, d2) +
                                    X[RIPEMD_R2[j]] + RIPEMD_K2[j / 16],
                                    RIPEMD_S2[j]) + e2;
        a2 = e2; e2 = d2; d2 = hash_rotl32(c2, 10); c2 = b2; b2 = t2;
    }

    // Final combination
    const uint t = h1 + c + d2;
    h1 = h2 + d + e2;
    h2 = h3 + e + a2;
    h3 = h4 + a + b2;
    h4 = h0 + b0 + c2;
    h0 = t;

    // Output: 5 × 32-bit words, little-endian byte order
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
// Input: pubkey (33 or 65 bytes)
// Output: 20 bytes

inline void hash160_pubkey_impl(const uchar* pubkey, uint pubkey_len, uchar out[20]) {
    uchar sha[32];
    sha256_oneshot_impl(pubkey, pubkey_len, sha);
    ripemd160_32_impl(sha, out);
}

// =============================================================================
// Kernel: Batch Hash160 of public keys
// =============================================================================
// Input:  pubkeys — concatenated pubkeys (fixed stride per key)
//         stride  — bytes per pubkey (33 for compressed, 65 for uncompressed)
//         count   — number of pubkeys
// Output: hashes — 20 bytes per key (stride 20)

__kernel void hash160_batch(
    __global const uchar* pubkeys,
    __global uchar* hashes,
    const uint stride,
    const uint count
) {
    const uint tid = get_global_id(0);
    if (tid >= count) return;

    // Copy pubkey to private memory
    uchar pk[65];
    const uint pk_len = (stride <= 65) ? stride : 65;
    for (uint i = 0; i < pk_len; ++i) {
        pk[i] = pubkeys[tid * stride + i];
    }

    uchar h160[20];
    hash160_pubkey_impl(pk, pk_len, h160);

    // Write output
    for (int i = 0; i < 20; ++i) {
        hashes[tid * 20 + i] = h160[i];
    }
}

#endif // SECP256K1_HASH160_CL
