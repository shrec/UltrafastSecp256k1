// =============================================================================
// UltrafastSecp256k1 Metal — Extended Scalar, Crypto & MSM Operations
// =============================================================================
// This file extends the Metal shaders with all missing functionality:
//
// Layer 1: Serialization (scalar_from_bytes, scalar_to_bytes, field_to_bytes)
//          + field_sqrt (modular square root)
// Layer 2: Scalar mod-n algebra (negate, mul, inverse, add, sub, eq, ge, etc.)
//          + GLV endomorphism (decompose)
// Layer 3: SHA-256 streaming + HMAC-SHA256 + RFC 6979
// Layer 4: ECDSA sign/verify + precomputed generator mul
// Layer 5: Schnorr BIP-340 + ECDH + Key Recovery + MSM/Pippenger
//
// Depends on: secp256k1_point.h (which includes secp256k1_field.h)
// Uses 8×32-bit limbs (uint) — matching existing Metal convention.
// Apple Silicon GPU-optimized: no 64-bit int in hot loops.
// =============================================================================

#pragma once

#include "secp256k1_point.h"

// =============================================================================
// Constants — 8×32 little-endian
// =============================================================================

// secp256k1 order n
constant uint SECP256K1_N[8] = {
    0xD0364141u, 0xBFD25E8Cu, 0xAF48A03Bu, 0xBAAEDCE6u,
    0xFFFFFFFEu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu
};

// n/2 (half order for low-S)
constant uint HALF_N[8] = {
    0x681B20A0u, 0xDFE92F46u, 0x57A4501Du, 0x5D576E73u,
    0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0x7FFFFFFFu
};

// n - 2 (for inversion via Fermat)
constant uint N_MINUS_2[8] = {
    0xD036413Fu, 0xBFD25E8Cu, 0xAF48A03Bu, 0xBAAEDCE6u,
    0xFFFFFFFEu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu
};

// SHA-256 round constants
constant uint K256[64] = {
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
// LAYER 1: Serialization + field_sqrt
// =============================================================================

// Big-endian 32 bytes → Scalar256 (8×32 LE limbs) with branchless mod-n
inline Scalar256 scalar_from_bytes(thread const uchar bytes[32]) {
    Scalar256 s;
    for (int i = 0; i < 8; i++) {
        int base = (7 - i) * 4;
        s.limbs[i] = (uint(bytes[base]) << 24) | (uint(bytes[base+1]) << 16)
                    | (uint(bytes[base+2]) << 8) | uint(bytes[base+3]);
    }
    // Branchless reduction: if s >= n, subtract n
    ulong borrow = 0;
    uint tmp[8];
    for (int i = 0; i < 8; i++) {
        ulong d = ulong(s.limbs[i]) - ulong(SECP256K1_N[i]) - borrow;
        tmp[i] = uint(d);
        borrow = (d >> 63);
    }
    uint mask = -(uint(borrow == 0)); // if no borrow → s >= n → use subtracted
    uint nmask = ~mask;
    for (int i = 0; i < 8; i++)
        s.limbs[i] = (tmp[i] & mask) | (s.limbs[i] & nmask);
    return s;
}

// Scalar256 → big-endian 32 bytes
inline void scalar_to_bytes(thread const Scalar256 &s, thread uchar out[32]) {
    for (int i = 0; i < 8; i++) {
        uint limb = s.limbs[7 - i];
        out[i*4+0] = uchar(limb >> 24);
        out[i*4+1] = uchar(limb >> 16);
        out[i*4+2] = uchar(limb >> 8);
        out[i*4+3] = uchar(limb);
    }
}

// FieldElement → big-endian 32 bytes
inline void field_to_bytes(thread const FieldElement &f, thread uchar out[32]) {
    for (int i = 0; i < 8; i++) {
        uint limb = f.limbs[7 - i];
        out[i*4+0] = uchar(limb >> 24);
        out[i*4+1] = uchar(limb >> 16);
        out[i*4+2] = uchar(limb >> 8);
        out[i*4+3] = uchar(limb);
    }
}

// Scalar256 is-zero check
inline bool scalar256_is_zero(thread const Scalar256 &s) {
    uint d = 0;
    for (int i = 0; i < 8; i++) d |= s.limbs[i];
    return d == 0;
}

// Field square root: a^((p+1)/4) via optimized addition chain
inline FieldElement field_sqrt(thread const FieldElement &a) {
    FieldElement x2 = field_sqr(a);
    x2 = field_mul(x2, a);          // a^3

    FieldElement x3 = field_sqr(x2);
    x3 = field_mul(x3, a);          // a^7

    FieldElement t = field_sqr_n(x3, 3);
    FieldElement x6 = field_mul(t, x3);

    t = field_sqr_n(x6, 3);
    FieldElement x9 = field_mul(t, x3);

    t = field_sqr_n(x9, 2);
    FieldElement x11 = field_mul(t, x2);

    t = field_sqr_n(x11, 11);
    FieldElement x22 = field_mul(t, x11);

    t = field_sqr_n(x22, 22);
    FieldElement x44 = field_mul(t, x22);

    t = field_sqr_n(x44, 44);
    FieldElement x88 = field_mul(t, x44);

    t = field_sqr_n(x88, 88);
    FieldElement x176 = field_mul(t, x88);

    t = field_sqr_n(x176, 44);
    FieldElement x220 = field_mul(t, x44);

    t = field_sqr_n(x220, 2);
    FieldElement x222 = field_mul(t, x2);

    // x223
    t = field_sqr(x222);
    t = field_mul(t, a);
    t = field_sqr(t);

    t = field_sqr_n(t, 22);
    t = field_mul(t, x22);

    FieldElement a12 = field_sqr(x2);
    a12 = field_sqr(a12);

    t = field_sqr_n(t, 8);
    return field_mul(t, a12);
}

// =============================================================================
// LAYER 2: Scalar mod-n Algebra (8×32-bit limbs)
// =============================================================================

// Scalar negate: r = n - a (if a != 0)
inline Scalar256 scalar_negate(thread const Scalar256 &a) {
    Scalar256 r;
    bool iz = scalar256_is_zero(a);
    ulong borrow = 0;
    for (int i = 0; i < 8; i++) {
        ulong d = ulong(SECP256K1_N[i]) - ulong(a.limbs[i]) - borrow;
        r.limbs[i] = uint(d);
        borrow = (d >> 63);
    }
    if (iz) for (int i = 0; i < 8; i++) r.limbs[i] = 0;
    return r;
}

// Scalar add mod n: r = (a + b) mod n
inline Scalar256 scalar_add_mod_n(thread const Scalar256 &a, thread const Scalar256 &b) {
    Scalar256 r;
    ulong carry = 0;
    for (int i = 0; i < 8; i++) {
        ulong s = ulong(a.limbs[i]) + ulong(b.limbs[i]) + carry;
        r.limbs[i] = uint(s);
        carry = s >> 32;
    }
    // Reduce: if r >= n or carry, subtract n
    ulong borrow = 0;
    uint tmp[8];
    for (int i = 0; i < 8; i++) {
        ulong d = ulong(r.limbs[i]) - ulong(SECP256K1_N[i]) - borrow;
        tmp[i] = uint(d);
        borrow = (d >> 63);
    }
    uint use_sub = uint(carry) | uint(borrow == 0);
    uint mask = -use_sub;
    uint nmask = ~mask;
    for (int i = 0; i < 8; i++)
        r.limbs[i] = (tmp[i] & mask) | (r.limbs[i] & nmask);
    return r;
}

// Scalar sub mod n
inline Scalar256 scalar_sub_mod_n(thread const Scalar256 &a, thread const Scalar256 &b) {
    Scalar256 r;
    ulong borrow = 0;
    for (int i = 0; i < 8; i++) {
        ulong d = ulong(a.limbs[i]) - ulong(b.limbs[i]) - borrow;
        r.limbs[i] = uint(d);
        borrow = (d >> 63);
    }
    // If borrow, add n back
    uint mask = -(uint(borrow));
    ulong carry = 0;
    for (int i = 0; i < 8; i++) {
        ulong s = ulong(r.limbs[i]) + ulong(SECP256K1_N[i] & mask) + carry;
        r.limbs[i] = uint(s);
        carry = s >> 32;
    }
    return r;
}

// Scalar multiply mod n (256×256→512 via 8×8 schoolbook with Barrett reduction)
inline Scalar256 scalar_mul_mod_n(thread const Scalar256 &a, thread const Scalar256 &b) {
    // Full 512-bit product in 16 × 32-bit limbs
    uint prod[16];
    for (int i = 0; i < 16; i++) prod[i] = 0;

    for (int i = 0; i < 8; i++) {
        ulong carry = 0;
        for (int j = 0; j < 8; j++) {
            ulong p = ulong(a.limbs[i]) * ulong(b.limbs[j])
                    + ulong(prod[i+j]) + carry;
            prod[i+j] = uint(p);
            carry = p >> 32;
        }
        prod[i+8] = uint(carry);
    }

    // Simple reduction: subtract q*n where q ≈ prod[8..15]
    // (Barrett approximation for 32-bit limb representation)
    Scalar256 r;
    for (int i = 0; i < 8; i++) r.limbs[i] = prod[i];

    uint q[8];
    for (int i = 0; i < 8; i++) q[i] = prod[i + 8];

    // Compute q * n (low 8 limbs only)
    uint qn[8];
    for (int i = 0; i < 8; i++) qn[i] = 0;
    for (int i = 0; i < 8; i++) {
        ulong carry = 0;
        for (int j = 0; j < 8 && (i+j) < 8; j++) {
            ulong p = ulong(q[i]) * ulong(SECP256K1_N[j])
                    + ulong(qn[i+j]) + carry;
            qn[i+j] = uint(p);
            carry = p >> 32;
        }
    }

    // r = r - q*n
    ulong borrow = 0;
    for (int i = 0; i < 8; i++) {
        ulong d = ulong(r.limbs[i]) - ulong(qn[i]) - borrow;
        r.limbs[i] = uint(d);
        borrow = (d >> 63);
    }

    // Conditional subtract n at most twice
    for (int pass = 0; pass < 2; pass++) {
        borrow = 0;
        uint tmp[8];
        for (int i = 0; i < 8; i++) {
            ulong d = ulong(r.limbs[i]) - ulong(SECP256K1_N[i]) - borrow;
            tmp[i] = uint(d);
            borrow = (d >> 63);
        }
        uint mask = -(uint(borrow == 0));
        uint nmask = ~mask;
        for (int i = 0; i < 8; i++)
            r.limbs[i] = (tmp[i] & mask) | (r.limbs[i] & nmask);
    }

    return r;
}

// Scalar inverse mod n: a^(n-2) via binary exponentiation
inline Scalar256 scalar_inverse(thread const Scalar256 &a) {
    Scalar256 result;
    result.limbs[0] = 1;
    for (int i = 1; i < 8; i++) result.limbs[i] = 0;

    Scalar256 base = a;
    for (int i = 0; i < 8; i++) {
        for (int bit = 0; bit < 32; bit++) {
            if ((N_MINUS_2[i] >> bit) & 1u) {
                result = scalar_mul_mod_n(result, base);
            }
            base = scalar_mul_mod_n(base, base);
        }
    }
    return result;
}

// Scalar utilities
inline bool scalar256_is_even(thread const Scalar256 &s) { return (s.limbs[0] & 1u) == 0; }

inline bool scalar256_eq(thread const Scalar256 &a, thread const Scalar256 &b) {
    uint d = 0;
    for (int i = 0; i < 8; i++) d |= (a.limbs[i] ^ b.limbs[i]);
    return d == 0;
}

inline int scalar256_bitlen(thread const Scalar256 &s) {
    for (int i = 7; i >= 0; i--) {
        if (s.limbs[i] != 0) {
            int bits = 32;
            uint v = s.limbs[i];
            while (!(v >> 31)) { v <<= 1; bits--; }
            return i * 32 + bits;
        }
    }
    return 0;
}

inline bool scalar256_ge(thread const Scalar256 &a, thread const Scalar256 &b) {
    for (int i = 7; i >= 0; i--) {
        if (a.limbs[i] > b.limbs[i]) return true;
        if (a.limbs[i] < b.limbs[i]) return false;
    }
    return true; // equal
}

inline bool scalar_is_low_s(thread const Scalar256 &s) {
    for (int i = 7; i >= 0; i--) {
        if (s.limbs[i] > HALF_N[i]) return false;
        if (s.limbs[i] < HALF_N[i]) return true;
    }
    return true;
}

// =============================================================================
// GLV Endomorphism (Jacobian version)
// =============================================================================

inline JacobianPoint apply_endomorphism_jac(thread const JacobianPoint &p) {
    FieldElement beta;
    for (int i = 0; i < 8; i++) beta.limbs[i] = BETA_LIMBS[i];
    JacobianPoint r;
    r.x = field_mul(p.x, beta);
    r.y = p.y;
    r.z = p.z;
    r.infinity = p.infinity;
    return r;
}

// Precomputed generator multiplication (4-bit window)
inline JacobianPoint scalar_mul_generator_windowed(thread const Scalar256 &k) {
    AffinePoint G = generator_affine();

    AffinePoint table[16];
    for (int i = 0; i < 8; i++) { table[0].x.limbs[i] = 0; table[0].y.limbs[i] = 0; }
    table[1] = G;

    JacobianPoint jp;
    jp.x = G.x; jp.y = G.y; jp.z = field_one(); jp.infinity = 0;
    JacobianPoint j2 = jacobian_double(jp);
    table[2] = jacobian_to_affine(j2);

    for (int i = 3; i < 16; i++) {
        JacobianPoint prev;
        prev.x = table[i-1].x; prev.y = table[i-1].y; prev.z = field_one(); prev.infinity = 0;
        JacobianPoint sum = jacobian_add_mixed(prev, G);
        table[i] = jacobian_to_affine(sum);
    }

    JacobianPoint r = point_at_infinity();
    bool started = false;

    for (int limb = 7; limb >= 0; limb--) {
        uint w = k.limbs[limb];
        for (int nib = 7; nib >= 0; nib--) {
            uint idx = (w >> (nib * 4)) & 0xFu;
            if (started) {
                r = jacobian_double(r);
                r = jacobian_double(r);
                r = jacobian_double(r);
                r = jacobian_double(r);
            }
            if (idx != 0) {
                AffinePoint selected = affine_select(table, idx);
                if (!started) {
                    r.x = selected.x; r.y = selected.y;
                    r.z = field_one(); r.infinity = 0;
                    started = true;
                } else {
                    r = jacobian_add_mixed(r, selected);
                }
            }
        }
    }
    return r;
}

// =============================================================================
// LAYER 3: SHA-256 Streaming + HMAC + RFC 6979
// =============================================================================

struct SHA256Ctx {
    uint h[8];
    uchar buf[64];
    uint buf_len;
    uint total_len_lo;
    uint total_len_hi;
};

inline uint sha256_rotr(uint x, uint n) { return (x >> n) | (x << (32 - n)); }
inline uint sha256_ch(uint x, uint y, uint z) { return (x & y) ^ (~x & z); }
inline uint sha256_maj(uint x, uint y, uint z) { return (x & y) ^ (x & z) ^ (y & z); }
inline uint sha256_bsig0(uint x) { return sha256_rotr(x,2) ^ sha256_rotr(x,13) ^ sha256_rotr(x,22); }
inline uint sha256_bsig1(uint x) { return sha256_rotr(x,6) ^ sha256_rotr(x,11) ^ sha256_rotr(x,25); }
inline uint sha256_ssig0(uint x) { return sha256_rotr(x,7) ^ sha256_rotr(x,18) ^ (x >> 3); }
inline uint sha256_ssig1(uint x) { return sha256_rotr(x,17) ^ sha256_rotr(x,19) ^ (x >> 10); }

inline void sha256_compress(thread SHA256Ctx &ctx, thread const uchar block[64]) {
    uint w[64];
    for (int i = 0; i < 16; i++)
        w[i] = (uint(block[i*4]) << 24) | (uint(block[i*4+1]) << 16)
             | (uint(block[i*4+2]) << 8) | uint(block[i*4+3]);
    for (int i = 16; i < 64; i++)
        w[i] = sha256_ssig1(w[i-2]) + w[i-7] + sha256_ssig0(w[i-15]) + w[i-16];

    uint a=ctx.h[0], b=ctx.h[1], c=ctx.h[2], d=ctx.h[3];
    uint e=ctx.h[4], f=ctx.h[5], g=ctx.h[6], h=ctx.h[7];

    for (int i = 0; i < 64; i++) {
        uint t1 = h + sha256_bsig1(e) + sha256_ch(e,f,g) + K256[i] + w[i];
        uint t2 = sha256_bsig0(a) + sha256_maj(a,b,c);
        h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }

    ctx.h[0]+=a; ctx.h[1]+=b; ctx.h[2]+=c; ctx.h[3]+=d;
    ctx.h[4]+=e; ctx.h[5]+=f; ctx.h[6]+=g; ctx.h[7]+=h;
}

inline void sha256_init(thread SHA256Ctx &ctx) {
    ctx.h[0]=0x6a09e667u; ctx.h[1]=0xbb67ae85u;
    ctx.h[2]=0x3c6ef372u; ctx.h[3]=0xa54ff53au;
    ctx.h[4]=0x510e527fu; ctx.h[5]=0x9b05688cu;
    ctx.h[6]=0x1f83d9abu; ctx.h[7]=0x5be0cd19u;
    ctx.buf_len = 0; ctx.total_len_lo = 0; ctx.total_len_hi = 0;
}

inline void sha256_update(thread SHA256Ctx &ctx, thread const uchar* data, uint len) {
    ctx.total_len_lo += len;
    if (ctx.total_len_lo < len) ctx.total_len_hi++; // overflow
    uint i = 0;
    if (ctx.buf_len > 0) {
        while (ctx.buf_len < 64 && i < len) ctx.buf[ctx.buf_len++] = data[i++];
        if (ctx.buf_len == 64) { sha256_compress(ctx, ctx.buf); ctx.buf_len = 0; }
    }
    while (i + 64 <= len) { sha256_compress(ctx, data + i); i += 64; }
    while (i < len) ctx.buf[ctx.buf_len++] = data[i++];
}

inline void sha256_final(thread SHA256Ctx &ctx, thread uchar out[32]) {
    ulong bits = (ulong(ctx.total_len_hi) << 32) | ulong(ctx.total_len_lo);
    bits *= 8;
    uchar pad = 0x80;
    sha256_update(ctx, &pad, 1);
    uchar zero = 0;
    while (ctx.buf_len != 56) sha256_update(ctx, &zero, 1);
    uchar len_bytes[8];
    for (int i = 0; i < 8; i++) len_bytes[i] = uchar(bits >> (56 - i*8));
    sha256_update(ctx, len_bytes, 8);

    for (int i = 0; i < 8; i++) {
        out[i*4+0] = uchar(ctx.h[i] >> 24);
        out[i*4+1] = uchar(ctx.h[i] >> 16);
        out[i*4+2] = uchar(ctx.h[i] >> 8);
        out[i*4+3] = uchar(ctx.h[i]);
    }
}

inline void hmac_sha256(thread const uchar* key, uint key_len,
                         thread const uchar* msg, uint msg_len,
                         thread uchar out[32]) {
    uchar k_pad[64];
    uchar key_hash[32];
    if (key_len > 64) {
        SHA256Ctx kctx; sha256_init(kctx);
        sha256_update(kctx, key, key_len);
        sha256_final(kctx, key_hash);
        key = key_hash; key_len = 32;
    }
    for (uint i = 0; i < 64; i++) k_pad[i] = (i < key_len ? key[i] : 0) ^ 0x36;
    SHA256Ctx ictx; sha256_init(ictx);
    sha256_update(ictx, k_pad, 64);
    sha256_update(ictx, msg, msg_len);
    uchar inner[32];
    sha256_final(ictx, inner);

    for (uint i = 0; i < 64; i++) k_pad[i] = (i < key_len ? key[i] : 0) ^ 0x5c;
    SHA256Ctx octx; sha256_init(octx);
    sha256_update(octx, k_pad, 64);
    sha256_update(octx, inner, 32);
    sha256_final(octx, out);
}

inline void rfc6979_nonce(thread const Scalar256 &priv, thread const uchar msg_hash[32],
                           thread Scalar256 &k_out) {
    uchar priv_bytes[32];
    scalar_to_bytes(priv, priv_bytes);

    uchar V[32], K_[32];
    for (int i = 0; i < 32; i++) { V[i] = 0x01; K_[i] = 0x00; }

    uchar hmac_input[97];
    for (int i = 0; i < 32; i++) hmac_input[i] = V[i];
    hmac_input[32] = 0x00;
    for (int i = 0; i < 32; i++) hmac_input[33+i] = priv_bytes[i];
    for (int i = 0; i < 32; i++) hmac_input[65+i] = msg_hash[i];
    hmac_sha256(K_, 32, hmac_input, 97, K_);
    hmac_sha256(K_, 32, V, 32, V);

    for (int i = 0; i < 32; i++) hmac_input[i] = V[i];
    hmac_input[32] = 0x01;
    hmac_sha256(K_, 32, hmac_input, 97, K_);
    hmac_sha256(K_, 32, V, 32, V);

    for (int attempt = 0; attempt < 100; attempt++) {
        hmac_sha256(K_, 32, V, 32, V);
        k_out = scalar_from_bytes(V);
        if (!scalar256_is_zero(k_out)) {
            Scalar256 order;
            for (int i = 0; i < 8; i++) order.limbs[i] = SECP256K1_N[i];
            if (!scalar256_ge(k_out, order)) return;
        }
        uchar retry_input[33];
        for (int i = 0; i < 32; i++) retry_input[i] = V[i];
        retry_input[32] = 0x00;
        hmac_sha256(K_, 32, retry_input, 33, K_);
        hmac_sha256(K_, 32, V, 32, V);
    }
}

// =============================================================================
// LAYER 4: ECDSA Sign / Verify
// =============================================================================

struct ECDSASignature {
    Scalar256 r;
    Scalar256 s;
};

inline bool ecdsa_sign(thread const uchar msg_hash[32], thread const Scalar256 &priv,
                        thread ECDSASignature &sig) {
    if (scalar256_is_zero(priv)) return false;

    Scalar256 z = scalar_from_bytes(msg_hash);
    Scalar256 k;
    rfc6979_nonce(priv, msg_hash, k);
    if (scalar256_is_zero(k)) return false;

    JacobianPoint R = scalar_mul_generator_windowed(k);
    if (R.infinity != 0) return false;

    AffinePoint R_aff = jacobian_to_affine(R);
    uchar rx_bytes[32];
    field_to_bytes(R_aff.x, rx_bytes);
    sig.r = scalar_from_bytes(rx_bytes);
    if (scalar256_is_zero(sig.r)) return false;

    Scalar256 k_inv = scalar_inverse(k);
    Scalar256 rd = scalar_mul_mod_n(sig.r, priv);
    Scalar256 z_plus_rd = scalar_add_mod_n(z, rd);
    sig.s = scalar_mul_mod_n(k_inv, z_plus_rd);
    if (scalar256_is_zero(sig.s)) return false;

    if (!scalar_is_low_s(sig.s))
        sig.s = scalar_negate(sig.s);

    return true;
}

inline bool ecdsa_verify(thread const uchar msg_hash[32], thread const JacobianPoint &pubkey,
                          thread const ECDSASignature &sig) {
    if (scalar256_is_zero(sig.r) || scalar256_is_zero(sig.s)) return false;

    Scalar256 z = scalar_from_bytes(msg_hash);
    Scalar256 s_inv = scalar_inverse(sig.s);
    Scalar256 u1 = scalar_mul_mod_n(z, s_inv);
    Scalar256 u2 = scalar_mul_mod_n(sig.r, s_inv);

    AffinePoint G = generator_affine();
    JacobianPoint u1G = scalar_mul(G, u1);

    AffinePoint pub_aff = jacobian_to_affine(pubkey);
    JacobianPoint u2Q = scalar_mul(pub_aff, u2);

    JacobianPoint R = jacobian_add(u1G, u2Q);
    if (R.infinity != 0) return false;

    AffinePoint R_aff = jacobian_to_affine(R);
    uchar rx_bytes[32];
    field_to_bytes(R_aff.x, rx_bytes);
    Scalar256 rx_scalar = scalar_from_bytes(rx_bytes);

    return scalar256_eq(rx_scalar, sig.r);
}

// =============================================================================
// LAYER 5a: Tagged Hash + Schnorr BIP-340
// =============================================================================

inline void tagged_hash(thread const uchar* tag, uint tag_len,
                         thread const uchar* data, uint data_len,
                         thread uchar out[32]) {
    uchar tag_hash[32];
    SHA256Ctx ctx; sha256_init(ctx);
    sha256_update(ctx, tag, tag_len);
    sha256_final(ctx, tag_hash);

    sha256_init(ctx);
    sha256_update(ctx, tag_hash, 32);
    sha256_update(ctx, tag_hash, 32);
    sha256_update(ctx, data, data_len);
    sha256_final(ctx, out);
}

inline bool lift_x(thread const uchar x_bytes[32], thread JacobianPoint &p) {
    FieldElement x;
    for (int i = 0; i < 8; i++) {
        int base = (7 - i) * 4;
        x.limbs[i] = (uint(x_bytes[base]) << 24) | (uint(x_bytes[base+1]) << 16)
                    | (uint(x_bytes[base+2]) << 8) | uint(x_bytes[base+3]);
    }

    FieldElement x2 = field_sqr(x);
    FieldElement x3 = field_mul(x2, x);
    FieldElement seven = field_zero(); seven.limbs[0] = 7;
    FieldElement y2 = field_add(x3, seven);
    FieldElement y = field_sqrt(y2);

    FieldElement y_check = field_sqr(y);
    if (!field_eq(y_check, y2)) return false;

    // Ensure even Y
    uchar y_bytes[32];
    field_to_bytes(y, y_bytes);
    if (y_bytes[31] & 1) y = field_negate(y);

    p.x = x; p.y = y; p.z = field_one(); p.infinity = 0;
    return true;
}

struct SchnorrSignature {
    uchar r[32];
    Scalar256 s;
};

inline bool schnorr_sign(thread const Scalar256 &priv, thread const uchar msg[32],
                          thread const uchar aux_rand[32], thread SchnorrSignature &sig) {
    if (scalar256_is_zero(priv)) return false;

    JacobianPoint P = scalar_mul_generator_windowed(priv);
    if (P.infinity != 0) return false;
    AffinePoint P_aff = jacobian_to_affine(P);

    uchar py_bytes[32];
    field_to_bytes(P_aff.y, py_bytes);
    Scalar256 d = priv;
    if (py_bytes[31] & 1) d = scalar_negate(priv);

    uchar px_bytes[32];
    field_to_bytes(P_aff.x, px_bytes);

    uchar t_hash[32];
    tagged_hash((const thread uchar*)"BIP0340/aux", 12, aux_rand, 32, t_hash);

    uchar d_bytes[32];
    scalar_to_bytes(d, d_bytes);
    uchar t[32];
    for (int i = 0; i < 32; i++) t[i] = d_bytes[i] ^ t_hash[i];

    uchar nonce_input[96];
    for (int i = 0; i < 32; i++) nonce_input[i] = t[i];
    for (int i = 0; i < 32; i++) nonce_input[32+i] = px_bytes[i];
    for (int i = 0; i < 32; i++) nonce_input[64+i] = msg[i];

    uchar rand_hash[32];
    tagged_hash((const thread uchar*)"BIP0340/nonce", 13, nonce_input, 96, rand_hash);

    Scalar256 k_prime = scalar_from_bytes(rand_hash);
    if (scalar256_is_zero(k_prime)) return false;

    JacobianPoint R = scalar_mul_generator_windowed(k_prime);
    AffinePoint R_aff = jacobian_to_affine(R);

    uchar ry_bytes[32];
    field_to_bytes(R_aff.y, ry_bytes);
    Scalar256 k = k_prime;
    if (ry_bytes[31] & 1) k = scalar_negate(k_prime);

    field_to_bytes(R_aff.x, sig.r);

    uchar challenge_input[96];
    for (int i = 0; i < 32; i++) challenge_input[i] = sig.r[i];
    for (int i = 0; i < 32; i++) challenge_input[32+i] = px_bytes[i];
    for (int i = 0; i < 32; i++) challenge_input[64+i] = msg[i];

    uchar e_hash[32];
    tagged_hash((const thread uchar*)"BIP0340/challenge", 19, challenge_input, 96, e_hash);
    Scalar256 e = scalar_from_bytes(e_hash);

    Scalar256 ed = scalar_mul_mod_n(e, d);
    sig.s = scalar_add_mod_n(k, ed);
    return true;
}

inline bool schnorr_verify(thread const uchar pubkey_x[32], thread const uchar msg[32],
                             thread const SchnorrSignature &sig) {
    if (scalar256_is_zero(sig.s)) return false;

    JacobianPoint P;
    if (!lift_x(pubkey_x, P)) return false;

    uchar challenge_input[96];
    for (int i = 0; i < 32; i++) challenge_input[i] = sig.r[i];
    for (int i = 0; i < 32; i++) challenge_input[32+i] = pubkey_x[i];
    for (int i = 0; i < 32; i++) challenge_input[64+i] = msg[i];

    uchar e_hash[32];
    tagged_hash((const thread uchar*)"BIP0340/challenge", 19, challenge_input, 96, e_hash);
    Scalar256 e = scalar_from_bytes(e_hash);

    AffinePoint G = generator_affine();
    JacobianPoint sG = scalar_mul(G, sig.s);

    AffinePoint p_aff = jacobian_to_affine(P);
    JacobianPoint eP = scalar_mul(p_aff, e);

    // Negate eP
    eP.y = field_negate(eP.y);

    JacobianPoint Rpt = jacobian_add(sG, eP);
    if (Rpt.infinity != 0) return false;

    AffinePoint Rpt_aff = jacobian_to_affine(Rpt);
    uchar ry_bytes[32];
    field_to_bytes(Rpt_aff.y, ry_bytes);
    if (ry_bytes[31] & 1) return false;

    uchar rx_bytes[32];
    field_to_bytes(Rpt_aff.x, rx_bytes);
    for (int i = 0; i < 32; i++)
        if (rx_bytes[i] != sig.r[i]) return false;

    return true;
}

// =============================================================================
// LAYER 5b: ECDH
// =============================================================================

inline bool ecdh_compute_raw(thread const Scalar256 &priv, thread const AffinePoint &peer,
                              thread uchar out[32]) {
    JacobianPoint shared = scalar_mul(peer, priv);
    if (shared.infinity != 0) return false;
    AffinePoint shared_aff = jacobian_to_affine(shared);
    field_to_bytes(shared_aff.x, out);
    return true;
}

inline bool ecdh_compute_xonly(thread const Scalar256 &priv, thread const AffinePoint &peer,
                                thread uchar out[32]) {
    uchar x_bytes[32];
    if (!ecdh_compute_raw(priv, peer, x_bytes)) return false;
    SHA256Ctx ctx; sha256_init(ctx);
    sha256_update(ctx, x_bytes, 32);
    sha256_final(ctx, out);
    return true;
}

inline bool ecdh_compute(thread const Scalar256 &priv, thread const AffinePoint &peer,
                          thread uchar out[32]) {
    uchar x_bytes[32];
    if (!ecdh_compute_raw(priv, peer, x_bytes)) return false;
    SHA256Ctx ctx; sha256_init(ctx);
    uchar prefix = 0x02;
    sha256_update(ctx, &prefix, 1);
    sha256_update(ctx, x_bytes, 32);
    sha256_final(ctx, out);
    return true;
}

// =============================================================================
// LAYER 5c: Key Recovery
// =============================================================================

struct RecoverableSignature {
    ECDSASignature sig;
    int recid;
};

inline bool lift_x_field(thread const FieldElement &x_fe, int parity, thread JacobianPoint &p) {
    FieldElement x2 = field_sqr(x_fe);
    FieldElement x3 = field_mul(x2, x_fe);
    FieldElement seven = field_zero(); seven.limbs[0] = 7;
    FieldElement y2 = field_add(x3, seven);
    FieldElement y = field_sqrt(y2);

    FieldElement y_check = field_sqr(y);
    if (!field_eq(y_check, y2)) return false;

    uchar y_bytes[32];
    field_to_bytes(y, y_bytes);
    bool y_is_odd = (y_bytes[31] & 1) != 0;
    if ((parity != 0) != y_is_odd) y = field_negate(y);

    p.x = x_fe; p.y = y; p.z = field_one(); p.infinity = 0;
    return true;
}

inline bool ecdsa_sign_recoverable(thread const uchar msg_hash[32], thread const Scalar256 &priv,
                                     thread RecoverableSignature &rsig) {
    if (scalar256_is_zero(priv)) return false;

    Scalar256 z = scalar_from_bytes(msg_hash);
    Scalar256 k;
    rfc6979_nonce(priv, msg_hash, k);
    if (scalar256_is_zero(k)) return false;

    JacobianPoint R = scalar_mul_generator_windowed(k);
    if (R.infinity != 0) return false;

    AffinePoint R_aff = jacobian_to_affine(R);
    uchar rx_bytes[32], ry_bytes[32];
    field_to_bytes(R_aff.x, rx_bytes);
    field_to_bytes(R_aff.y, ry_bytes);

    rsig.sig.r = scalar_from_bytes(rx_bytes);
    if (scalar256_is_zero(rsig.sig.r)) return false;

    int recid = 0;
    if (ry_bytes[31] & 1) recid |= 1;

    // Check overflow (R.x >= n)
    Scalar256 order;
    for (int i = 0; i < 8; i++) order.limbs[i] = SECP256K1_N[i];
    // Compare rx_bytes (BE) against order (BE)
    uchar order_be[32];
    scalar_to_bytes(order, order_be);
    bool overflow = false;
    for (int i = 0; i < 32; i++) {
        if (rx_bytes[i] < order_be[i]) break;
        if (rx_bytes[i] > order_be[i]) { overflow = true; break; }
    }
    if (overflow) recid |= 2;

    Scalar256 k_inv = scalar_inverse(k);
    Scalar256 rd = scalar_mul_mod_n(rsig.sig.r, priv);
    Scalar256 z_plus_rd = scalar_add_mod_n(z, rd);
    rsig.sig.s = scalar_mul_mod_n(k_inv, z_plus_rd);
    if (scalar256_is_zero(rsig.sig.s)) return false;

    if (!scalar_is_low_s(rsig.sig.s)) {
        rsig.sig.s = scalar_negate(rsig.sig.s);
        recid ^= 1;
    }

    rsig.recid = recid;
    return true;
}

inline bool ecdsa_recover(thread const uchar msg_hash[32], thread const ECDSASignature &sig,
                            int recid, thread JacobianPoint &Q) {
    if (recid < 0 || recid > 3) return false;
    if (scalar256_is_zero(sig.r) || scalar256_is_zero(sig.s)) return false;

    // Reconstruct R.x as FieldElement
    uchar r_bytes[32];
    scalar_to_bytes(sig.r, r_bytes);
    FieldElement rx_fe;
    for (int i = 0; i < 8; i++) {
        int base = (7 - i) * 4;
        rx_fe.limbs[i] = (uint(r_bytes[base]) << 24) | (uint(r_bytes[base+1]) << 16)
                        | (uint(r_bytes[base+2]) << 8) | uint(r_bytes[base+3]);
    }

    if (recid & 2) {
        // Add n to rx_fe
        FieldElement n_fe;
        for (int i = 0; i < 8; i++) n_fe.limbs[i] = SECP256K1_N[i];
        rx_fe = field_add(rx_fe, n_fe);
    }

    JacobianPoint Rpt;
    if (!lift_x_field(rx_fe, recid & 1, Rpt)) return false;

    Scalar256 z = scalar_from_bytes(msg_hash);
    Scalar256 r_inv = scalar_inverse(sig.r);

    AffinePoint r_aff = jacobian_to_affine(Rpt);
    JacobianPoint sR = scalar_mul(r_aff, sig.s);

    AffinePoint G = generator_affine();
    JacobianPoint zG = scalar_mul(G, z);
    zG.y = field_negate(zG.y); // negate

    JacobianPoint sR_minus_zG = jacobian_add(sR, zG);

    AffinePoint diff_aff = jacobian_to_affine(sR_minus_zG);
    Q = scalar_mul(diff_aff, r_inv);

    if (Q.infinity != 0) return false;
    return true;
}

// =============================================================================
// LAYER 5d: MSM (Multi-Scalar Multiplication)
// =============================================================================

inline uint scalar_get_window(thread const Scalar256 &s, int window_idx, int c) {
    int bit_offset = window_idx * c;
    int limb_idx = bit_offset / 32;
    int bit_idx = bit_offset % 32;
    if (limb_idx >= 8) return 0;

    uint val = (s.limbs[limb_idx] >> bit_idx) & ((1u << c) - 1);
    int bits_from_first = 32 - bit_idx;
    if (bits_from_first < c && limb_idx + 1 < 8) {
        int remaining = c - bits_from_first;
        val |= (s.limbs[limb_idx+1] & ((1u << remaining) - 1)) << bits_from_first;
    }
    return val;
}

inline JacobianPoint msm_naive(thread const Scalar256* scalars, thread const AffinePoint* points,
                                int n) {
    JacobianPoint result = point_at_infinity();
    for (int i = 0; i < n; i++) {
        if (scalar256_is_zero(scalars[i])) continue;
        JacobianPoint tmp = scalar_mul(points[i], scalars[i]);
        result = jacobian_add(result, tmp);
    }
    return result;
}

inline JacobianPoint msm_pippenger(thread const Scalar256* scalars, thread const AffinePoint* points,
                                     int n, thread JacobianPoint* buckets, int c) {
    int num_buckets = 1 << c;
    int num_windows = (256 + c - 1) / c;

    JacobianPoint result = point_at_infinity();

    for (int w = num_windows - 1; w >= 0; w--) {
        if (result.infinity == 0) {
            for (int d = 0; d < c; d++)
                result = jacobian_double(result);
        }

        for (int b = 0; b < num_buckets; b++)
            buckets[b] = point_at_infinity();

        for (int i = 0; i < n; i++) {
            uint digit = scalar_get_window(scalars[i], w, c);
            if (digit == 0) continue;
            JacobianPoint jp;
            jp.x = points[i].x; jp.y = points[i].y;
            jp.z = field_one(); jp.infinity = 0;
            buckets[digit] = jacobian_add(buckets[digit], jp);
        }

        JacobianPoint running_sum = point_at_infinity();
        JacobianPoint partial_sum = point_at_infinity();

        for (int b = num_buckets - 1; b >= 1; b--) {
            running_sum = jacobian_add(running_sum, buckets[b]);
            partial_sum = jacobian_add(partial_sum, running_sum);
        }

        result = jacobian_add(result, partial_sum);
    }
    return result;
}

// =============================================================================
// Metal Compute Kernels — Extended Operations
// =============================================================================

kernel void ecdsa_sign_kernel(
    device const uchar* msg_hashes       [[buffer(0)]],
    device const Scalar256* private_keys  [[buffer(1)]],
    device ECDSASignature* signatures     [[buffer(2)]],
    device int* success_flags             [[buffer(3)]],
    constant uint& count                  [[buffer(4)]],
    uint gid                              [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    uchar msg[32];
    for (int i = 0; i < 32; i++) msg[i] = msg_hashes[gid * 32 + i];
    Scalar256 priv = private_keys[gid];
    ECDSASignature sig;
    success_flags[gid] = ecdsa_sign(msg, priv, sig) ? 1 : 0;
    signatures[gid] = sig;
}

kernel void ecdsa_verify_kernel(
    device const uchar* msg_hashes          [[buffer(0)]],
    device const JacobianPoint* pubkeys     [[buffer(1)]],
    device const ECDSASignature* signatures [[buffer(2)]],
    device int* results                     [[buffer(3)]],
    constant uint& count                    [[buffer(4)]],
    uint gid                                [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    uchar msg[32];
    for (int i = 0; i < 32; i++) msg[i] = msg_hashes[gid * 32 + i];
    JacobianPoint pub = pubkeys[gid];
    ECDSASignature sig = signatures[gid];
    results[gid] = ecdsa_verify(msg, pub, sig) ? 1 : 0;
}

kernel void schnorr_sign_kernel(
    device const uchar* messages        [[buffer(0)]],
    device const Scalar256* private_keys [[buffer(1)]],
    device const uchar* aux_rands       [[buffer(2)]],
    device SchnorrSignature* signatures [[buffer(3)]],
    device int* success_flags           [[buffer(4)]],
    constant uint& count                [[buffer(5)]],
    uint gid                            [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    uchar msg[32], aux[32];
    for (int i = 0; i < 32; i++) { msg[i] = messages[gid*32+i]; aux[i] = aux_rands[gid*32+i]; }
    Scalar256 priv = private_keys[gid];
    SchnorrSignature sig;
    success_flags[gid] = schnorr_sign(priv, msg, aux, sig) ? 1 : 0;
    signatures[gid] = sig;
}

kernel void schnorr_verify_kernel(
    device const uchar* pubkeys_x        [[buffer(0)]],
    device const uchar* messages         [[buffer(1)]],
    device const SchnorrSignature* sigs  [[buffer(2)]],
    device int* results                  [[buffer(3)]],
    constant uint& count                 [[buffer(4)]],
    uint gid                             [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    uchar pk[32], msg[32];
    for (int i = 0; i < 32; i++) { pk[i] = pubkeys_x[gid*32+i]; msg[i] = messages[gid*32+i]; }
    SchnorrSignature sig = sigs[gid];
    results[gid] = schnorr_verify(pk, msg, sig) ? 1 : 0;
}

kernel void generator_mul_windowed_kernel(
    device const Scalar256* scalars [[buffer(0)]],
    device JacobianPoint* results   [[buffer(1)]],
    constant uint& count            [[buffer(2)]],
    uint gid                        [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    Scalar256 k = scalars[gid];
    results[gid] = scalar_mul_generator_windowed(k);
}
