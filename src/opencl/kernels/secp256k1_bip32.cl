// =============================================================================
// UltrafastSecp256k1 OpenCL -- BIP-32 HD Key Derivation
// =============================================================================
// Provides GPU-side BIP-32 operations:
//   - SHA-512 (streaming + one-shot)
//   - HMAC-SHA512
//   - Master key from seed
//   - Child key derivation (normal + hardened)
//   - Fingerprint (Hash160 of compressed pubkey)
//   - Serialization (78 bytes)
//   - Batch derivation kernel
//
// Dependencies: secp256k1_zk.cl (scalar/point/field ops, SHA-256 streaming)
// Uses Keccak padding: N/A (this is pure Bitcoin BIP-32, not Ethereum)
//
// Reference: BIP-32 (https://github.com/bitcoin/bips/blob/master/bip-0032.mediawiki)
// =============================================================================

#ifndef SECP256K1_BIP32_CL
#define SECP256K1_BIP32_CL

#include "secp256k1_zk.cl"
// BUG-M2 FIX: CT generator mul for private key derivation in normal child keys
#include "secp256k1_ct_ops.cl"
#include "secp256k1_ct_field.cl"
#include "secp256k1_ct_scalar.cl"
#include "secp256k1_ct_point.cl"

// =============================================================================
// SHA-512 Implementation
// =============================================================================

__constant ulong SHA512_K[80] = {
    0x428a2f98d728ae22UL, 0x7137449123ef65cdUL, 0xb5c0fbcfec4d3b2fUL, 0xe9b5dba58189dbbcUL,
    0x3956c25bf348b538UL, 0x59f111f1b605d019UL, 0x923f82a4af194f9bUL, 0xab1c5ed5da6d8118UL,
    0xd807aa98a3030242UL, 0x12835b0145706fbeUL, 0x243185be4ee4b28cUL, 0x550c7dc3d5ffb4e2UL,
    0x72be5d74f27b896fUL, 0x80deb1fe3b1696b1UL, 0x9bdc06a725c71235UL, 0xc19bf174cf692694UL,
    0xe49b69c19ef14ad2UL, 0xefbe4786384f25e3UL, 0x0fc19dc68b8cd5b5UL, 0x240ca1cc77ac9c65UL,
    0x2de92c6f592b0275UL, 0x4a7484aa6ea6e483UL, 0x5cb0a9dcbd41fbd4UL, 0x76f988da831153b5UL,
    0x983e5152ee66dfabUL, 0xa831c66d2db43210UL, 0xb00327c898fb213fUL, 0xbf597fc7beef0ee4UL,
    0xc6e00bf33da88fc2UL, 0xd5a79147930aa725UL, 0x06ca6351e003826fUL, 0x142929670a0e6e70UL,
    0x27b70a8546d22ffcUL, 0x2e1b21385c26c926UL, 0x4d2c6dfc5ac42aedUL, 0x53380d139d95b3dfUL,
    0x650a73548baf63deUL, 0x766a0abb3c77b2a8UL, 0x81c2c92e47edaee6UL, 0x92722c851482353bUL,
    0xa2bfe8a14cf10364UL, 0xa81a664bbc423001UL, 0xc24b8b70d0f89791UL, 0xc76c51a30654be30UL,
    0xd192e819d6ef5218UL, 0xd69906245565a910UL, 0xf40e35855771202aUL, 0x106aa07032bbd1b8UL,
    0x19a4c116b8d2d0c8UL, 0x1e376c085141ab53UL, 0x2748774cdf8eeb99UL, 0x34b0bcb5e19b48a8UL,
    0x391c0cb3c5c95a63UL, 0x4ed8aa4ae3418acbUL, 0x5b9cca4f7763e373UL, 0x682e6ff3d6b2b8a3UL,
    0x748f82ee5defb2fcUL, 0x78a5636f43172f60UL, 0x84c87814a1f0ab72UL, 0x8cc702081a6439ecUL,
    0x90befffa23631e28UL, 0xa4506cebde82bde9UL, 0xbef9a3f7b2c67915UL, 0xc67178f2e372532bUL,
    0xca273eceea26619cUL, 0xd186b8c721c0c207UL, 0xeada7dd6cde0eb1eUL, 0xf57d4f7fee6ed178UL,
    0x06f067aa72176fbaUL, 0x0a637dc5a2c898a6UL, 0x113f9804bef90daeUL, 0x1b710b35131c471bUL,
    0x28db77f523047d84UL, 0x32caab7b40c72493UL, 0x3c9ebe0a15c9bebcUL, 0x431d67c49c100d4cUL,
    0x4cc5d4becb3e42b6UL, 0x597f299cfc657e2aUL, 0x5fcb6fab3ad6faecUL, 0x6c44198c4a475817UL,
};

inline ulong sha512_rotr(ulong x, int n) {
    return (x >> n) | (x << (64 - n));
}

typedef struct {
    ulong h[8];
    uchar buf[128];
    uint buf_len;
    ulong total;
} SHA512Ctx;

inline void sha512_compress(SHA512Ctx* ctx, const uchar block[128]) {
    ulong w[80];
    for (int i = 0; i < 16; i++) {
        w[i] = 0;
        for (int j = 0; j < 8; j++)
            w[i] = (w[i] << 8) | (ulong)block[i * 8 + j];
    }
    for (int i = 16; i < 80; i++) {
        ulong s0 = sha512_rotr(w[i-15], 1) ^ sha512_rotr(w[i-15], 8) ^ (w[i-15] >> 7);
        ulong s1 = sha512_rotr(w[i-2], 19) ^ sha512_rotr(w[i-2], 61) ^ (w[i-2] >> 6);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }

    ulong a = ctx->h[0], b = ctx->h[1], c = ctx->h[2], d = ctx->h[3];
    ulong e = ctx->h[4], f = ctx->h[5], g = ctx->h[6], hh = ctx->h[7];

    for (int i = 0; i < 80; i++) {
        ulong S1 = sha512_rotr(e, 14) ^ sha512_rotr(e, 18) ^ sha512_rotr(e, 41);
        ulong ch = (e & f) ^ (~e & g);
        ulong t1 = hh + S1 + ch + SHA512_K[i] + w[i];
        ulong S0 = sha512_rotr(a, 28) ^ sha512_rotr(a, 34) ^ sha512_rotr(a, 39);
        ulong maj = (a & b) ^ (a & c) ^ (b & c);
        ulong t2 = S0 + maj;

        hh = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }

    ctx->h[0] += a; ctx->h[1] += b; ctx->h[2] += c; ctx->h[3] += d;
    ctx->h[4] += e; ctx->h[5] += f; ctx->h[6] += g; ctx->h[7] += hh;
}

inline void sha512_init(SHA512Ctx* ctx) {
    ctx->h[0] = 0x6a09e667f3bcc908UL; ctx->h[1] = 0xbb67ae8584caa73bUL;
    ctx->h[2] = 0x3c6ef372fe94f82bUL; ctx->h[3] = 0xa54ff53a5f1d36f1UL;
    ctx->h[4] = 0x510e527fade682d1UL; ctx->h[5] = 0x9b05688c2b3e6c1fUL;
    ctx->h[6] = 0x1f83d9abfb41bd6bUL; ctx->h[7] = 0x5be0cd19137e2179UL;
    ctx->buf_len = 0;
    ctx->total = 0;
}

inline void sha512_update(SHA512Ctx* ctx, const uchar* data, uint len) {
    ctx->total += len;
    uint offset = 0;

    if (ctx->buf_len > 0) {
        uint fill = 128 - ctx->buf_len;
        uint copy = (len < fill) ? len : fill;
        for (uint i = 0; i < copy; i++) ctx->buf[ctx->buf_len + i] = data[i];
        ctx->buf_len += copy;
        offset += copy;
        if (ctx->buf_len == 128) {
            sha512_compress(ctx, ctx->buf);
            ctx->buf_len = 0;
        }
    }

    while (offset + 128 <= len) {
        sha512_compress(ctx, data + offset);
        offset += 128;
    }

    while (offset < len) {
        ctx->buf[ctx->buf_len++] = data[offset++];
    }
}

inline void sha512_final(SHA512Ctx* ctx, uchar out[64]) {
    ulong bit_len = ctx->total * 8;
    ctx->buf[ctx->buf_len++] = 0x80;

    if (ctx->buf_len > 112) {
        while (ctx->buf_len < 128) ctx->buf[ctx->buf_len++] = 0;
        sha512_compress(ctx, ctx->buf);
        ctx->buf_len = 0;
    }

    while (ctx->buf_len < 120) ctx->buf[ctx->buf_len++] = 0;

    for (int i = 0; i < 8; i++) ctx->buf[112 + i] = 0;
    for (int i = 7; i >= 0; i--)
        ctx->buf[120 + (7 - i)] = (uchar)(bit_len >> (i * 8));

    sha512_compress(ctx, ctx->buf);

    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 8; j++)
            out[i * 8 + j] = (uchar)(ctx->h[i] >> (56 - j * 8));
}

inline void sha512_impl(const uchar* data, uint len, uchar out[64]) {
    SHA512Ctx ctx;
    sha512_init(&ctx);
    sha512_update(&ctx, data, len);
    sha512_final(&ctx, out);
}

// =============================================================================
// HMAC-SHA512
// =============================================================================

inline void hmac_sha512_impl(
    const uchar* key, uint key_len,
    const uchar* msg, uint msg_len,
    uchar out[64])
{
    uchar k_buf[128];
    for (int i = 0; i < 128; i++) k_buf[i] = 0;

    if (key_len > 128) {
        sha512_impl(key, key_len, k_buf);
    } else {
        for (uint i = 0; i < key_len; i++) k_buf[i] = key[i];
    }

    uchar ipad[128], opad[128];
    for (int i = 0; i < 128; i++) {
        ipad[i] = k_buf[i] ^ 0x36;
        opad[i] = k_buf[i] ^ 0x5c;
    }

    SHA512Ctx inner;
    sha512_init(&inner);
    sha512_update(&inner, ipad, 128);
    sha512_update(&inner, msg, msg_len);
    uchar inner_hash[64];
    sha512_final(&inner, inner_hash);

    SHA512Ctx outer;
    sha512_init(&outer);
    sha512_update(&outer, opad, 128);
    sha512_update(&outer, inner_hash, 64);
    sha512_final(&outer, out);
}

// =============================================================================
// RIPEMD-160 (for BIP-32 fingerprint, inlined to avoid hash160 include conflict)
// =============================================================================

__constant uchar BIP32_RIPEMD_R[80] = {
    0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
    7,4,13,1,10,6,15,3,12,0,9,5,2,14,11,8,
    3,10,14,4,9,15,8,1,2,7,0,6,13,11,5,12,
    1,9,11,10,0,8,12,4,13,3,7,15,14,5,6,2,
    4,0,5,9,7,12,2,10,14,1,3,8,11,6,15,13
};
__constant uchar BIP32_RIPEMD_R2[80] = {
    5,14,7,0,9,2,11,4,13,6,15,8,1,10,3,12,
    6,11,3,7,0,13,5,10,14,15,8,12,4,9,1,2,
    15,5,1,3,7,14,6,9,11,8,12,2,10,0,13,4,
    8,6,4,1,3,11,15,0,5,12,2,13,9,7,10,14,
    12,15,10,4,1,5,8,7,6,2,13,14,0,3,9,11
};
__constant uchar BIP32_RIPEMD_S[80] = {
    11,14,15,12,5,8,7,9,11,13,14,15,6,7,9,8,
    7,6,8,13,11,9,7,15,7,12,15,9,11,7,13,12,
    11,13,6,7,14,9,13,15,14,8,13,6,5,12,7,5,
    11,12,14,15,14,15,9,8,9,14,5,6,8,6,5,12,
    9,15,5,11,6,8,13,12,5,12,13,14,11,8,5,6
};
__constant uchar BIP32_RIPEMD_S2[80] = {
    8,9,9,11,13,15,15,5,7,7,8,11,14,14,12,6,
    9,13,15,7,12,8,9,11,7,7,12,7,6,15,13,11,
    9,7,15,11,8,6,6,14,12,13,5,14,13,13,7,5,
    15,5,8,11,14,14,6,14,6,9,12,9,12,5,15,8,
    8,5,12,9,12,5,14,6,8,13,6,5,15,13,11,11
};
__constant uint BIP32_RIPEMD_K[5] = {
    0x00000000U, 0x5A827999U, 0x6ED9EBA1U, 0x8F1BBCDCU, 0xA953FD4EU
};
__constant uint BIP32_RIPEMD_K2[5] = {
    0x50A28BE6U, 0x5C4DD124U, 0x6D703EF3U, 0x7A6D76E9U, 0x00000000U
};

inline uint bip32_rotl32(uint x, uint n) { return (x << n) | (x >> (32 - n)); }

inline uint bip32_ripemd_f(int j, uint x, uint y, uint z) {
    if (j <= 15) return x ^ y ^ z;
    if (j <= 31) return (x & y) | (~x & z);
    if (j <= 47) return (x | ~y) ^ z;
    if (j <= 63) return (x & z) | (y & ~z);
    return x ^ (y | ~z);
}

// RIPEMD-160 of exactly 32 bytes (SHA-256 output)
inline void bip32_ripemd160_32(const uchar data[32], uchar out[20]) {
    uchar block[64];
    for (int i = 0; i < 64; ++i) block[i] = 0;
    for (int i = 0; i < 32; ++i) block[i] = data[i];
    block[32] = 0x80;
    ulong bit_len = 32UL * 8UL;
    block[56] = (uchar)(bit_len);
    block[57] = (uchar)(bit_len >> 8);

    uint X[16];
    for (int i = 0; i < 16; ++i) {
        int j = i * 4;
        X[i] = (uint)block[j] | ((uint)block[j+1]<<8) | ((uint)block[j+2]<<16) | ((uint)block[j+3]<<24);
    }

    uint h0 = 0x67452301U, h1 = 0xEFCDAB89U, h2 = 0x98BADCFEU;
    uint h3 = 0x10325476U, h4 = 0xC3D2E1F0U;

    uint a = h0, b0 = h1, c = h2, d = h3, e = h4;
    uint a2 = h0, b2 = h1, c2 = h2, d2 = h3, e2 = h4;

    for (int j = 0; j < 80; ++j) {
        uint t = bip32_rotl32(a + bip32_ripemd_f(j, b0, c, d) +
                              X[BIP32_RIPEMD_R[j]] + BIP32_RIPEMD_K[j/16],
                              BIP32_RIPEMD_S[j]) + e;
        a = e; e = d; d = bip32_rotl32(c, 10); c = b0; b0 = t;

        uint t2 = bip32_rotl32(a2 + bip32_ripemd_f(79 - j, b2, c2, d2) +
                               X[BIP32_RIPEMD_R2[j]] + BIP32_RIPEMD_K2[j/16],
                               BIP32_RIPEMD_S2[j]) + e2;
        a2 = e2; e2 = d2; d2 = bip32_rotl32(c2, 10); c2 = b2; b2 = t2;
    }

    uint t = h1 + c + d2;
    h1 = h2 + d + e2;
    h2 = h3 + e + a2;
    h3 = h4 + a + b2;
    h4 = h0 + b0 + c2;
    h0 = t;

    uint hh[5] = {h0, h1, h2, h3, h4};
    for (int i = 0; i < 5; ++i) {
        out[i*4+0] = (uchar)(hh[i]);
        out[i*4+1] = (uchar)(hh[i]>>8);
        out[i*4+2] = (uchar)(hh[i]>>16);
        out[i*4+3] = (uchar)(hh[i]>>24);
    }
}

// Hash160 = RIPEMD160(SHA256(data)), using streaming SHA-256 from extended.cl
inline void bip32_hash160(const uchar* data, uint len, uchar out[20]) {
    SHA256Ctx ctx;
    sha256_init(&ctx);
    sha256_update(&ctx, data, len);
    uchar sha[32];
    sha256_final(&ctx, sha);
    bip32_ripemd160_32(sha, out);
}

// =============================================================================
// Point decompression (33-byte compressed pubkey -> JacobianPoint)
// =============================================================================

inline int bip32_point_from_compressed_impl(const uchar compressed[33], JacobianPoint* out) {
    uchar prefix = compressed[0];
    if (prefix != 0x02 && prefix != 0x03) return 0;

    // Parse x coordinate from bytes[1..32]
    FieldElement x;
    field_from_bytes_impl(compressed + 1, &x);

    // y^2 = x^3 + 7
    FieldElement x2, x3, y2, y;
    field_sqr_impl(&x2, &x);
    field_mul_impl(&x3, &x2, &x);

    FieldElement seven;
    seven.limbs[0] = 7; seven.limbs[1] = 0; seven.limbs[2] = 0; seven.limbs[3] = 0;
    field_add_impl(&y2, &x3, &seven);

    // sqrt
    field_sqrt_impl(&y2, &y);

    // Verify: y^2 == x^3 + 7
    FieldElement y_check;
    field_sqr_impl(&y_check, &y);

    // Normalize both for comparison
    uchar y2_bytes[32], yc_bytes[32];
    field_to_bytes_impl(&y2, y2_bytes);
    field_to_bytes_impl(&y_check, yc_bytes);
    int valid = 1;
    for (int i = 0; i < 32; i++) {
        if (y2_bytes[i] != yc_bytes[i]) { valid = 0; break; }
    }
    if (!valid) return 0;

    // Fix parity: if y is odd and prefix is 0x02 (even), negate y; vice versa
    uchar y_bytes[32];
    field_to_bytes_impl(&y, y_bytes);
    int y_odd = y_bytes[31] & 1;
    int want_odd = (prefix == 0x03) ? 1 : 0;
    if (y_odd != want_odd) {
        field_neg_impl(&y, &y);
    }

    out->x = x;
    out->y = y;
    // z = 1
    out->z.limbs[0] = 1; out->z.limbs[1] = 0;
    out->z.limbs[2] = 0; out->z.limbs[3] = 0;
    out->infinity = 0;

    return 1;
}

// =============================================================================
// BIP-32 Extended Key
// =============================================================================

typedef struct {
    uchar key[33];           // Private key (32 bytes, stored at [0..31])
                             // or compressed pubkey (33 bytes, stored at [0..32])
    uchar chain_code[32];
    uchar depth;
    uint child_number;
    uchar parent_fp[4];
    int is_private;          // 1 = private, 0 = public
} ExtendedKeyOCL;

// =============================================================================
// BIP-32 Master Key from Seed
// =============================================================================

inline int bip32_master_key_impl(const uchar* seed, uint seed_len, ExtendedKeyOCL* master) {
    const uchar btc_seed[12] = {'B','i','t','c','o','i','n',' ','s','e','e','d'};
    uchar I[64];
    hmac_sha512_impl(btc_seed, 12, seed, seed_len, I);

    // IL = I[0..31] = master secret key, IR = I[32..63] = chain code
    // Validate IL: parse raw and reduced, compare to detect >= n
    Scalar raw, reduced;
    for (int i = 0; i < 4; i++) {
        ulong limb = 0;
        int base = (3 - i) * 8;
        for (int j = 0; j < 8; j++)
            limb = (limb << 8) | (ulong)I[base + j];
        raw.limbs[i] = limb;
    }
    scalar_from_bytes_impl(I, &reduced);

    // If reduced != raw, then raw >= n -> invalid
    for (int i = 0; i < 4; i++)
        if (raw.limbs[i] != reduced.limbs[i]) return 0;

    // Check for zero
    if (scalar_is_zero(&reduced)) return 0;

    for (int i = 0; i < 32; i++) master->key[i] = I[i];
    master->key[32] = 0;
    for (int i = 0; i < 32; i++) master->chain_code[i] = I[32 + i];
    master->depth = 0;
    master->child_number = 0;
    for (int i = 0; i < 4; i++) master->parent_fp[i] = 0;
    master->is_private = 1;

    return 1;
}

// =============================================================================
// BIP-32 Fingerprint
// =============================================================================

inline void bip32_fingerprint_impl(const ExtendedKeyOCL* xkey, uchar fp[4]) {
    uchar compressed[33];

    if (xkey->is_private) {
        Scalar sk;
        scalar_from_bytes_impl(xkey->key, &sk);
        JacobianPoint P;
        scalar_mul_generator_impl(&P, &sk);
        point_to_compressed_impl(&P, compressed);
    } else {
        for (int i = 0; i < 33; i++) compressed[i] = xkey->key[i];
    }

    uchar hash[20];
    bip32_hash160(compressed, 33, hash);
    for (int i = 0; i < 4; i++) fp[i] = hash[i];
}

// =============================================================================
// BIP-32 Child Key Derivation
// =============================================================================

inline int bip32_derive_child_impl(
    const ExtendedKeyOCL* parent,
    uint index,
    ExtendedKeyOCL* child)
{
    int hardened = (index >= 0x80000000U);

    // Build HMAC-SHA512 data:
    // Hardened: 0x00 || ser256(kpar) || ser32(index) = 37 bytes
    // Normal:   serP(point(kpar))   || ser32(index) = 37 bytes
    uchar data[37];

    if (hardened) {
        if (!parent->is_private) return 0;
        data[0] = 0x00;
        for (int i = 0; i < 32; i++) data[1 + i] = parent->key[i];
    } else {
        if (parent->is_private) {
            Scalar sk;
            scalar_from_bytes_impl(parent->key, &sk);
            // BUG-M2 FIX: use CT generator mul — private key is secret.
            // scalar_mul_generator_impl (windowed) leaks key bits via timing.
            CTJacobianPoint P_ct;
            ct_generator_mul_impl(&sk, &P_ct);
            JacobianPoint P = ct_point_to_jacobian(&P_ct);
            point_to_compressed_impl(&P, data);
        } else {
            for (int i = 0; i < 33; i++) data[i] = parent->key[i];
        }
    }

    // Append ser32(index) big-endian
    int idx_off = hardened ? 33 : 33;
    data[idx_off + 0] = (uchar)(index >> 24);
    data[idx_off + 1] = (uchar)(index >> 16);
    data[idx_off + 2] = (uchar)(index >> 8);
    data[idx_off + 3] = (uchar)(index);

    // I = HMAC-SHA512(Key=chain_code, Data=data)
    uchar I[64];
    hmac_sha512_impl(parent->chain_code, 32, data, 37, I);

    // IL = I[0..31], IR = I[32..63]
    // Check IL < n: parse raw and reduced, compare
    Scalar raw, IL;
    for (int i = 0; i < 4; i++) {
        ulong limb = 0;
        int base = (3 - i) * 8;
        for (int j = 0; j < 8; j++)
            limb = (limb << 8) | (ulong)I[base + j];
        raw.limbs[i] = limb;
    }
    scalar_from_bytes_impl(I, &IL);

    for (int i = 0; i < 4; i++)
        if (raw.limbs[i] != IL.limbs[i]) return 0;  // IL >= n

    if (parent->is_private) {
        // child_key = (IL + kpar) mod n
        Scalar kpar;
        scalar_from_bytes_impl(parent->key, &kpar);

        Scalar child_key;
        scalar_add_mod_n_impl(&IL, &kpar, &child_key);

        if (scalar_is_zero(&child_key)) return 0;

        scalar_to_bytes_impl(&child_key, child->key);
        child->key[32] = 0;
        child->is_private = 1;
    } else {
        // child_pubkey = point(IL) + Kpar
        JacobianPoint IL_point;
        scalar_mul_generator_impl(&IL_point, &IL);

        JacobianPoint Kpar;
        if (!bip32_point_from_compressed_impl(parent->key, &Kpar)) return 0;

        JacobianPoint child_pub;
        point_add_impl(&child_pub, &IL_point, &Kpar);

        if (point_is_infinity(&child_pub)) return 0;

        point_to_compressed_impl(&child_pub, child->key);
        child->is_private = 0;
    }

    for (int i = 0; i < 32; i++) child->chain_code[i] = I[32 + i];
    // BUG-H1 FIX: depth is uchar (uint8) — adding 1 to 255 silently wraps to 0.
    if (parent->depth == 0xFFu) return 0;
    child->depth = parent->depth + 1;
    child->child_number = index;
    bip32_fingerprint_impl(parent, child->parent_fp);

    return 1;
}

inline int bip32_derive_normal_impl(const ExtendedKeyOCL* parent, uint index,
                                     ExtendedKeyOCL* child) {
    return bip32_derive_child_impl(parent, index, child);
}

inline int bip32_derive_hardened_impl(const ExtendedKeyOCL* parent, uint index,
                                       ExtendedKeyOCL* child) {
    return bip32_derive_child_impl(parent, 0x80000000U | index, child);
}

// =============================================================================
// BIP-32 Public Key
// =============================================================================

inline int bip32_public_key_impl(const ExtendedKeyOCL* xkey, uchar compressed[33]) {
    if (xkey->is_private) {
        Scalar sk;
        scalar_from_bytes_impl(xkey->key, &sk);
        JacobianPoint P;
        scalar_mul_generator_impl(&P, &sk);
        point_to_compressed_impl(&P, compressed);
        return 1;
    }
    for (int i = 0; i < 33; i++) compressed[i] = xkey->key[i];
    return 1;
}

// =============================================================================
// BIP-32 Serialize (78 bytes)
// =============================================================================

inline void bip32_serialize_impl(const ExtendedKeyOCL* xkey, int mainnet, uchar out[78]) {
    uint version;
    if (xkey->is_private)
        version = mainnet ? 0x0488ADE4U : 0x04358394U;
    else
        version = mainnet ? 0x0488B21EU : 0x043587CFU;

    out[0] = (uchar)(version >> 24);
    out[1] = (uchar)(version >> 16);
    out[2] = (uchar)(version >> 8);
    out[3] = (uchar)(version);
    out[4] = xkey->depth;
    for (int i = 0; i < 4; i++) out[5 + i] = xkey->parent_fp[i];
    out[9]  = (uchar)(xkey->child_number >> 24);
    out[10] = (uchar)(xkey->child_number >> 16);
    out[11] = (uchar)(xkey->child_number >> 8);
    out[12] = (uchar)(xkey->child_number);
    for (int i = 0; i < 32; i++) out[13 + i] = xkey->chain_code[i];

    if (xkey->is_private) {
        out[45] = 0x00;
        for (int i = 0; i < 32; i++) out[46 + i] = xkey->key[i];
    } else {
        for (int i = 0; i < 33; i++) out[45 + i] = xkey->key[i];
    }
}

// =============================================================================
// BIP-32 to_public (private -> public)
// =============================================================================

inline int bip32_to_public_impl(const ExtendedKeyOCL* xpriv, ExtendedKeyOCL* xpub) {
    if (!xpriv->is_private) {
        *xpub = *xpriv;
        return 1;
    }

    Scalar sk;
    scalar_from_bytes_impl(xpriv->key, &sk);
    JacobianPoint P;
    scalar_mul_generator_impl(&P, &sk);
    point_to_compressed_impl(&P, xpub->key);

    for (int i = 0; i < 32; i++) xpub->chain_code[i] = xpriv->chain_code[i];
    xpub->depth = xpriv->depth;
    xpub->child_number = xpriv->child_number;
    for (int i = 0; i < 4; i++) xpub->parent_fp[i] = xpriv->parent_fp[i];
    xpub->is_private = 0;

    return 1;
}

// =============================================================================
// Batch Kernel: Derive N child keys from a single parent
// =============================================================================
// Input:  parent_key (serialized ExtendedKeyOCL fields), base_index, count
// Output: child_keys (33 bytes per child: compressed pubkey) + chain_codes (32 bytes per child)

__kernel void bip32_derive_batch(
    __global const uchar* parent_data,   // 101 bytes: key[33]+chain_code[32]+depth[1]+child_num[4]+parent_fp[4]+is_private[1]+pad[26]
    __global uchar* child_pubkeys,       // 33 bytes per output
    __global uchar* child_chain_codes,   // 32 bytes per output
    uint base_index,
    uint count
) {
    uint gid = get_global_id(0);
    if (gid >= count) return;

    // Deserialize parent
    ExtendedKeyOCL parent;
    for (int i = 0; i < 33; i++) parent.key[i] = parent_data[i];
    for (int i = 0; i < 32; i++) parent.chain_code[i] = parent_data[33 + i];
    parent.depth = parent_data[65];
    parent.child_number = ((uint)parent_data[66] << 24) | ((uint)parent_data[67] << 16) |
                          ((uint)parent_data[68] << 8)  | (uint)parent_data[69];
    for (int i = 0; i < 4; i++) parent.parent_fp[i] = parent_data[70 + i];
    parent.is_private = parent_data[74];

    uint index = base_index + gid;
    ExtendedKeyOCL child;
    int ok = bip32_derive_child_impl(&parent, index, &child);

    if (ok) {
        uchar compressed[33];
        bip32_public_key_impl(&child, compressed);
        for (int i = 0; i < 33; i++)
            child_pubkeys[gid * 33 + i] = compressed[i];
        for (int i = 0; i < 32; i++)
            child_chain_codes[gid * 32 + i] = child.chain_code[i];
    } else {
        // Mark invalid with zero prefix
        for (int i = 0; i < 33; i++)
            child_pubkeys[gid * 33 + i] = 0;
        for (int i = 0; i < 32; i++)
            child_chain_codes[gid * 32 + i] = 0;
    }
}

#endif // SECP256K1_BIP32_CL
