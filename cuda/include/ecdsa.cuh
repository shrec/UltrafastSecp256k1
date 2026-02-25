#pragma once
// ============================================================================
// ECDSA Sign / Verify for secp256k1 -- CUDA device implementation
// ============================================================================
// Provides GPU-side ECDSA operations:
//   - ecdsa_sign(msg_hash, private_key) -> ECDSASignatureGPU
//   - ecdsa_verify(msg_hash, public_key, sig) -> bool
//   - RFC 6979 deterministic nonce (HMAC-SHA256 based)
//   - Low-S normalization (BIP-62)
//
// 64-bit limb mode only (requires scalar_mul_mod_n, scalar_inverse, etc.)
// ============================================================================

#include "secp256k1.cuh"

#if !SECP256K1_CUDA_LIMBS_32

namespace secp256k1 {
namespace cuda {

// -- Byte <-> Scalar conversion (big-endian bytes <-> LE uint64_t limbs) ---------

// Convert 32 big-endian bytes to a Scalar (reduced mod n).
__device__ inline void scalar_from_bytes(const uint8_t bytes[32], Scalar* r) {
    // BE bytes -> LE uint64_t limbs
    for (int i = 0; i < 4; i++) {
        uint64_t limb = 0;
        int base = (3 - i) * 8;
        for (int j = 0; j < 8; j++) {
            limb = (limb << 8) | bytes[base + j];
        }
        r->limbs[i] = limb;
    }
    // Branchless reduction: compute r - ORDER, keep if r >= n
    uint64_t borrow = 0;
    uint64_t tmp[4];
    for (int i = 0; i < 4; i++) {
        unsigned __int128 diff = (unsigned __int128)r->limbs[i] - ORDER[i] - borrow;
        tmp[i] = (uint64_t)diff;
        borrow = (uint64_t)(-(int64_t)(diff >> 64));  // 1 if borrow, 0 otherwise
    }
    // mask = all-ones if r >= n (no borrow), all-zeros otherwise
    uint64_t mask = ~borrow + 1;   // borrow==0 -> ~0+1=0 -> wrong
    // Actually: borrow=0 means no underflow -> r >= n -> use tmp
    //           borrow=1 means underflow -> r < n -> keep r
    mask = -(uint64_t)(borrow == 0);
    for (int i = 0; i < 4; i++) {
        r->limbs[i] = (tmp[i] & mask) | (r->limbs[i] & ~mask);
    }
}

// Convert Scalar to 32 big-endian bytes.
__device__ inline void scalar_to_bytes(const Scalar* s, uint8_t bytes[32]) {
    for (int i = 0; i < 4; i++) {
        uint64_t limb = s->limbs[3 - i];
        for (int j = 0; j < 8; j++) {
            bytes[i * 8 + j] = (uint8_t)(limb >> (56 - j * 8));
        }
    }
}

// Convert FieldElement to 32 big-endian bytes.
// Normalizes (fully reduces mod p) before serialization so byte comparisons
// are always consistent, regardless of internal carry state.
__device__ inline void field_to_bytes(const FieldElement* fe, uint8_t bytes[32]) {
    // p = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
    constexpr uint64_t P[4] = {
        0xFFFFFFFEFFFFFC2FULL, 0xFFFFFFFFFFFFFFFFULL,
        0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL
    };
    // Branchless: try subtracting p; keep result only if no borrow
    uint64_t tmp[4];
    uint64_t borrow = 0;
    for (int i = 0; i < 4; i++) {
        unsigned __int128 diff = (unsigned __int128)fe->limbs[i] - P[i] - borrow;
        tmp[i] = (uint64_t)diff;
        borrow = (uint64_t)(-(int64_t)(diff >> 64));  // 1 if borrow, 0 otherwise
    }
    // If borrow==0: fe >= p -> use tmp (reduced). If borrow==1: fe < p -> use fe.
    uint64_t mask = -(uint64_t)(borrow == 0);  // all-1s if no borrow, all-0s if borrow
    uint64_t norm[4];
    for (int i = 0; i < 4; i++)
        norm[i] = (tmp[i] & mask) | (fe->limbs[i] & ~mask);

    for (int i = 0; i < 4; i++) {
        uint64_t limb = norm[3 - i];
        for (int j = 0; j < 8; j++) {
            bytes[i * 8 + j] = (uint8_t)(limb >> (56 - j * 8));
        }
    }
}

// -- SHA-256 Streaming Context ------------------------------------------------

__device__ __constant__ static const uint32_t SHA256_K[64] = {
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

struct SHA256Ctx {
    uint32_t h[8];
    uint8_t buf[64];
    uint32_t buf_len;
    uint64_t total;
};

__device__ __forceinline__ uint32_t sha256_rotr(uint32_t x, int n) {
    return (x >> n) | (x << (32 - n));
}

// Process one 64-byte block, updating state in-place.
__device__ inline void sha256_compress(uint32_t state[8], const uint8_t block[64]) {
    uint32_t w[64];
    for (int i = 0; i < 16; i++) {
        w[i] = ((uint32_t)block[i*4] << 24) | ((uint32_t)block[i*4+1] << 16)
             | ((uint32_t)block[i*4+2] << 8)  |  (uint32_t)block[i*4+3];
    }
    for (int i = 16; i < 64; i++) {
        uint32_t s0 = sha256_rotr(w[i-15],7) ^ sha256_rotr(w[i-15],18) ^ (w[i-15]>>3);
        uint32_t s1 = sha256_rotr(w[i-2],17) ^ sha256_rotr(w[i-2],19)  ^ (w[i-2]>>10);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }

    uint32_t a=state[0], b=state[1], c=state[2], d=state[3];
    uint32_t e=state[4], f=state[5], g=state[6], hh=state[7];

    for (int i = 0; i < 64; i++) {
        uint32_t S1  = sha256_rotr(e,6) ^ sha256_rotr(e,11) ^ sha256_rotr(e,25);
        uint32_t ch  = (e & f) ^ (~e & g);
        uint32_t t1  = hh + S1 + ch + SHA256_K[i] + w[i];
        uint32_t S0  = sha256_rotr(a,2) ^ sha256_rotr(a,13) ^ sha256_rotr(a,22);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t t2  = S0 + maj;

        hh = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }

    state[0]+=a; state[1]+=b; state[2]+=c; state[3]+=d;
    state[4]+=e; state[5]+=f; state[6]+=g; state[7]+=hh;
}

__device__ inline void sha256_init(SHA256Ctx* ctx) {
    ctx->h[0]=0x6a09e667U; ctx->h[1]=0xbb67ae85U;
    ctx->h[2]=0x3c6ef372U; ctx->h[3]=0xa54ff53aU;
    ctx->h[4]=0x510e527fU; ctx->h[5]=0x9b05688cU;
    ctx->h[6]=0x1f83d9abU; ctx->h[7]=0x5be0cd19U;
    ctx->buf_len = 0;
    ctx->total = 0;
}

__device__ inline void sha256_update(SHA256Ctx* ctx, const uint8_t* data, size_t len) {
    ctx->total += len;
    size_t offset = 0;

    // If we have buffered data, fill the buffer first
    if (ctx->buf_len > 0) {
        uint32_t fill = 64 - ctx->buf_len;
        uint32_t copy = (len < fill) ? (uint32_t)len : fill;
        for (uint32_t i = 0; i < copy; i++) ctx->buf[ctx->buf_len + i] = data[i];
        ctx->buf_len += copy;
        offset += copy;
        if (ctx->buf_len == 64) {
            sha256_compress(ctx->h, ctx->buf);
            ctx->buf_len = 0;
        }
    }

    // Process full blocks directly from input
    while (offset + 64 <= len) {
        sha256_compress(ctx->h, data + offset);
        offset += 64;
    }

    // Buffer remaining bytes
    while (offset < len) {
        ctx->buf[ctx->buf_len++] = data[offset++];
    }
}

__device__ inline void sha256_final(SHA256Ctx* ctx, uint8_t out[32]) {
    // Pad: append 0x80, then zeros, then 8-byte BE length
    uint64_t bit_len = ctx->total * 8;
    ctx->buf[ctx->buf_len++] = 0x80;

    // If buffer > 56 bytes, compress and start new block
    if (ctx->buf_len > 56) {
        while (ctx->buf_len < 64) ctx->buf[ctx->buf_len++] = 0;
        sha256_compress(ctx->h, ctx->buf);
        ctx->buf_len = 0;
    }

    // Pad to 56 bytes
    while (ctx->buf_len < 56) ctx->buf[ctx->buf_len++] = 0;

    // Append 8-byte big-endian length
    for (int i = 7; i >= 0; i--) {
        ctx->buf[56 + (7 - i)] = (uint8_t)(bit_len >> (i * 8));
    }

    sha256_compress(ctx->h, ctx->buf);

    // Write output as big-endian
    for (int i = 0; i < 8; i++) {
        out[i*4+0] = (uint8_t)(ctx->h[i] >> 24);
        out[i*4+1] = (uint8_t)(ctx->h[i] >> 16);
        out[i*4+2] = (uint8_t)(ctx->h[i] >> 8);
        out[i*4+3] = (uint8_t)(ctx->h[i]);
    }
}

// -- HMAC-SHA256 --------------------------------------------------------------

__device__ inline void hmac_sha256(
    const uint8_t* key, size_t key_len,
    const uint8_t* msg, size_t msg_len,
    uint8_t out[32])
{
    uint8_t k_buf[64];
    for (int i = 0; i < 64; i++) k_buf[i] = 0;

    if (key_len > 64) {
        SHA256Ctx tmp; sha256_init(&tmp);
        sha256_update(&tmp, key, key_len);
        sha256_final(&tmp, k_buf);  // k_buf[0..31]=hash, [32..63]=0
    } else {
        for (size_t i = 0; i < key_len; i++) k_buf[i] = key[i];
    }

    uint8_t ipad[64], opad[64];
    for (int i = 0; i < 64; i++) {
        ipad[i] = k_buf[i] ^ 0x36;
        opad[i] = k_buf[i] ^ 0x5c;
    }

    // inner = SHA256(ipad || msg)
    SHA256Ctx inner; sha256_init(&inner);
    sha256_update(&inner, ipad, 64);
    sha256_update(&inner, msg, msg_len);
    uint8_t inner_hash[32];
    sha256_final(&inner, inner_hash);

    // outer = SHA256(opad || inner_hash)
    SHA256Ctx outer; sha256_init(&outer);
    sha256_update(&outer, opad, 64);
    sha256_update(&outer, inner_hash, 32);
    sha256_final(&outer, out);
}

// -- RFC 6979 Deterministic Nonce ---------------------------------------------
// Generates deterministic k for ECDSA signing per RFC 6979 S3.2
// using HMAC-SHA256. Inputs: private key scalar + 32-byte message hash.

__device__ inline void rfc6979_nonce(
    const Scalar* private_key,
    const uint8_t msg_hash[32],
    Scalar* k_out)
{
    uint8_t x_bytes[32];
    scalar_to_bytes(private_key, x_bytes);

    // Step b: V = 0x01 ...01 (32 bytes)
    uint8_t V[32], K[32];
    for (int i = 0; i < 32; i++) { V[i] = 0x01; K[i] = 0x00; }

    // Step d: K = HMAC(K, V || 0x00 || x || h1)
    {
        uint8_t buf[97]; // 32 + 1 + 32 + 32
        for (int i = 0; i < 32; i++) buf[i] = V[i];
        buf[32] = 0x00;
        for (int i = 0; i < 32; i++) buf[33 + i] = x_bytes[i];
        for (int i = 0; i < 32; i++) buf[65 + i] = msg_hash[i];
        hmac_sha256(K, 32, buf, 97, K);
    }

    // Step e: V = HMAC(K, V)
    hmac_sha256(K, 32, V, 32, V);

    // Step f: K = HMAC(K, V || 0x01 || x || h1)
    {
        uint8_t buf[97];
        for (int i = 0; i < 32; i++) buf[i] = V[i];
        buf[32] = 0x01;
        for (int i = 0; i < 32; i++) buf[33 + i] = x_bytes[i];
        for (int i = 0; i < 32; i++) buf[65 + i] = msg_hash[i];
        hmac_sha256(K, 32, buf, 97, K);
    }

    // Step g: V = HMAC(K, V)
    hmac_sha256(K, 32, V, 32, V);

    // Step h: loop until valid k found
    for (int attempt = 0; attempt < 100; attempt++) {
        hmac_sha256(K, 32, V, 32, V);

        scalar_from_bytes(V, k_out);  // reduces mod n

        if (!scalar_is_zero(k_out)) return;  // valid nonce found

        // Retry: K = HMAC(K, V || 0x00), V = HMAC(K, V)
        uint8_t buf[33];
        for (int i = 0; i < 32; i++) buf[i] = V[i];
        buf[32] = 0x00;
        hmac_sha256(K, 32, buf, 33, K);
        hmac_sha256(K, 32, V, 32, V);
    }

    // Should never reach here for valid inputs
    for (int i = 0; i < 4; i++) k_out->limbs[i] = 0;
}

// -- ECDSA Types --------------------------------------------------------------

struct ECDSASignatureGPU {
    Scalar r;
    Scalar s;
};

// n/2 = 0x7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF5D576E7357A4501DDFE92F46681B20A0
__device__ __constant__ static const Scalar HALF_ORDER = {
    {0xDFE92F46681B20A0ULL, 0x5D576E7357A4501DULL,
     0xFFFFFFFFFFFFFFFFULL, 0x7FFFFFFFFFFFFFFFULL}
};

// Check if s <= n/2 (low-S per BIP-62)
__device__ __forceinline__ bool scalar_is_low_s(const Scalar* s) {
    for (int i = 3; i >= 0; i--) {
        if (s->limbs[i] < HALF_ORDER.limbs[i]) return true;
        if (s->limbs[i] > HALF_ORDER.limbs[i]) return false;
    }
    return true; // equal -> low-S
}

// -- ECDSA Sign ---------------------------------------------------------------
// Signs a 32-byte message hash with a private key.
// Uses RFC 6979 deterministic nonce.
// Returns low-S normalized signature.
// Returns false on failure (zero key, zero r, zero s).

__device__ inline bool ecdsa_sign(
    const uint8_t msg_hash[32],
    const Scalar* private_key,
    ECDSASignatureGPU* sig)
{
    if (scalar_is_zero(private_key)) return false;

    // z = message hash as scalar (reduced mod n)
    Scalar z;
    scalar_from_bytes(msg_hash, &z);

    // k = RFC 6979 nonce
    Scalar k;
    rfc6979_nonce(private_key, msg_hash, &k);
    if (scalar_is_zero(&k)) return false;

    // R = k * G  (use precomputed constant table for generator)
    JacobianPoint R;
    scalar_mul_generator_const(&k, &R);
    if (R.infinity) return false;

    // Convert R to affine x-coordinate
    FieldElement z_inv, z_inv2;
    field_inv(&R.z, &z_inv);
    field_sqr(&z_inv, &z_inv2);
    FieldElement x_affine;
    field_mul(&R.x, &z_inv2, &x_affine);

    // r = x_affine mod n
    uint8_t x_bytes[32];
    field_to_bytes(&x_affine, x_bytes);
    scalar_from_bytes(x_bytes, &sig->r);
    if (scalar_is_zero(&sig->r)) return false;

    // s = k^{-1} * (z + r * d) mod n
    Scalar k_inv;
    scalar_inverse(&k, &k_inv);

    Scalar rd;  // r * d
    scalar_mul_mod_n(&sig->r, private_key, &rd);

    Scalar z_plus_rd;  // z + r*d
    // Addition mod n: compute sum, reduce if >= n
    {
        uint64_t carry = 0;
        for (int i = 0; i < 4; i++) {
            unsigned __int128 sum = (unsigned __int128)z.limbs[i] + rd.limbs[i] + carry;
            z_plus_rd.limbs[i] = (uint64_t)sum;
            carry = (uint64_t)(sum >> 64);
        }
        // Reduce mod n
        uint64_t borrow = 0;
        uint64_t tmp[4];
        for (int i = 0; i < 4; i++) {
            unsigned __int128 diff = (unsigned __int128)z_plus_rd.limbs[i] - ORDER[i] - borrow;
            tmp[i] = (uint64_t)diff;
            borrow = (uint64_t)(-(int64_t)(diff >> 64));
        }
        uint64_t mask = -(uint64_t)(borrow == 0);
        for (int i = 0; i < 4; i++) {
            z_plus_rd.limbs[i] = (tmp[i] & mask) | (z_plus_rd.limbs[i] & ~mask);
        }
        // Handle carry from addition: if carry, the sum was >= 2^256, definitely > n
        if (carry) {
            // z_plus_rd -= n (use tmp result from above but with carry absorbed)
            borrow = 0;
            for (int i = 0; i < 4; i++) {
                unsigned __int128 diff = (unsigned __int128)z_plus_rd.limbs[i] - ORDER[i] - borrow;
                z_plus_rd.limbs[i] = (uint64_t)diff;
                borrow = (uint64_t)(-(int64_t)(diff >> 64));
            }
        }
    }

    scalar_mul_mod_n(&k_inv, &z_plus_rd, &sig->s);
    if (scalar_is_zero(&sig->s)) return false;

    // Normalize to low-S (BIP-62)
    if (!scalar_is_low_s(&sig->s)) {
        scalar_negate(&sig->s, &sig->s);
    }

    return true;
}

// -- ECDSA Verify -------------------------------------------------------------
// Verifies an ECDSA signature against a public key and message hash.
// Accepts both low-S and high-S signatures.
// public_key must be a valid Jacobian point (not infinity).

__device__ inline bool ecdsa_verify(
    const uint8_t msg_hash[32],
    const JacobianPoint* public_key,
    const ECDSASignatureGPU* sig)
{
    // Check r, s are non-zero
    if (scalar_is_zero(&sig->r) || scalar_is_zero(&sig->s)) return false;

    // z = message hash as scalar
    Scalar z;
    scalar_from_bytes(msg_hash, &z);

    // w = s^{-1} mod n
    Scalar w;
    scalar_inverse(&sig->s, &w);

    // u1 = z * w mod n
    Scalar u1;
    scalar_mul_mod_n(&z, &w, &u1);

    // u2 = r * w mod n
    Scalar u2;
    scalar_mul_mod_n(&sig->r, &w, &u2);

    // R' = u1 * G + u2 * Q  (Shamir's trick with GLV: ~128 doublings instead of 2x256)
    JacobianPoint R_prime;
    shamir_double_mul_glv(&GENERATOR_JACOBIAN, &u1, public_key, &u2, &R_prime);

    if (R_prime.infinity) return false;

    // v = R'.x mod n (convert affine x to scalar)
    FieldElement z_inv, z_inv2, x_affine;
    field_inv(&R_prime.z, &z_inv);
    field_sqr(&z_inv, &z_inv2);
    field_mul(&R_prime.x, &z_inv2, &x_affine);

    uint8_t x_bytes[32];
    field_to_bytes(&x_affine, x_bytes);

    Scalar v;
    scalar_from_bytes(x_bytes, &v);

    // Check v == r
    return scalar_eq(&v, &sig->r);
}

} // namespace cuda
} // namespace secp256k1

#endif // !SECP256K1_CUDA_LIMBS_32
