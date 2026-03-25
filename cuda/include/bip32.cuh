#pragma once
// ============================================================================
// BIP-32 Hierarchical Deterministic Key Derivation -- CUDA device
// ============================================================================
// Provides GPU-side BIP-32 operations:
//   - SHA-512 (streaming + one-shot)
//   - HMAC-SHA512
//   - Master key from seed
//   - Child key derivation (normal + hardened)
//   - Path-based derivation
//
// Reference: BIP-32 (https://github.com/bitcoin/bips/blob/master/bip-0032.mediawiki)
// ============================================================================

#include "schnorr.cuh"  // for scalar_from_bytes, scalar_mul_generator_w8, etc.
#include "hash160.cuh"  // for hash160_pubkey (BIP-32 fingerprint)

#if !SECP256K1_CUDA_LIMBS_32

namespace secp256k1 {
namespace cuda {

// ============================================================================
// SHA-512 device implementation
// ============================================================================

__device__ __constant__ static const uint64_t SHA512_K[80] = {
    0x428a2f98d728ae22ULL, 0x7137449123ef65cdULL, 0xb5c0fbcfec4d3b2fULL, 0xe9b5dba58189dbbcULL,
    0x3956c25bf348b538ULL, 0x59f111f1b605d019ULL, 0x923f82a4af194f9bULL, 0xab1c5ed5da6d8118ULL,
    0xd807aa98a3030242ULL, 0x12835b0145706fbeULL, 0x243185be4ee4b28cULL, 0x550c7dc3d5ffb4e2ULL,
    0x72be5d74f27b896fULL, 0x80deb1fe3b1696b1ULL, 0x9bdc06a725c71235ULL, 0xc19bf174cf692694ULL,
    0xe49b69c19ef14ad2ULL, 0xefbe4786384f25e3ULL, 0x0fc19dc68b8cd5b5ULL, 0x240ca1cc77ac9c65ULL,
    0x2de92c6f592b0275ULL, 0x4a7484aa6ea6e483ULL, 0x5cb0a9dcbd41fbd4ULL, 0x76f988da831153b5ULL,
    0x983e5152ee66dfabULL, 0xa831c66d2db43210ULL, 0xb00327c898fb213fULL, 0xbf597fc7beef0ee4ULL,
    0xc6e00bf33da88fc2ULL, 0xd5a79147930aa725ULL, 0x06ca6351e003826fULL, 0x142929670a0e6e70ULL,
    0x27b70a8546d22ffcULL, 0x2e1b21385c26c926ULL, 0x4d2c6dfc5ac42aedULL, 0x53380d139d95b3dfULL,
    0x650a73548baf63deULL, 0x766a0abb3c77b2a8ULL, 0x81c2c92e47edaee6ULL, 0x92722c851482353bULL,
    0xa2bfe8a14cf10364ULL, 0xa81a664bbc423001ULL, 0xc24b8b70d0f89791ULL, 0xc76c51a30654be30ULL,
    0xd192e819d6ef5218ULL, 0xd69906245565a910ULL, 0xf40e35855771202aULL, 0x106aa07032bbd1b8ULL,
    0x19a4c116b8d2d0c8ULL, 0x1e376c085141ab53ULL, 0x2748774cdf8eeb99ULL, 0x34b0bcb5e19b48a8ULL,
    0x391c0cb3c5c95a63ULL, 0x4ed8aa4ae3418acbULL, 0x5b9cca4f7763e373ULL, 0x682e6ff3d6b2b8a3ULL,
    0x748f82ee5defb2fcULL, 0x78a5636f43172f60ULL, 0x84c87814a1f0ab72ULL, 0x8cc702081a6439ecULL,
    0x90befffa23631e28ULL, 0xa4506cebde82bde9ULL, 0xbef9a3f7b2c67915ULL, 0xc67178f2e372532bULL,
    0xca273eceea26619cULL, 0xd186b8c721c0c207ULL, 0xeada7dd6cde0eb1eULL, 0xf57d4f7fee6ed178ULL,
    0x06f067aa72176fbaULL, 0x0a637dc5a2c898a6ULL, 0x113f9804bef90daeULL, 0x1b710b35131c471bULL,
    0x28db77f523047d84ULL, 0x32caab7b40c72493ULL, 0x3c9ebe0a15c9bebcULL, 0x431d67c49c100d4cULL,
    0x4cc5d4becb3e42b6ULL, 0x597f299cfc657e2aULL, 0x5fcb6fab3ad6faecULL, 0x6c44198c4a475817ULL,
};

__device__ __forceinline__ uint64_t sha512_rotr(uint64_t x, int n) {
    return (x >> n) | (x << (64 - n));
}

struct SHA512Ctx {
    uint64_t h[8];
    uint8_t buf[128];
    uint32_t buf_len;
    uint64_t total;
};

__device__ inline void sha512_compress(uint64_t state[8], const uint8_t block[128]) {
    uint64_t w[80];
    for (int i = 0; i < 16; i++) {
        w[i] = 0;
        for (int j = 0; j < 8; j++)
            w[i] = (w[i] << 8) | block[i * 8 + j];
    }
    for (int i = 16; i < 80; i++) {
        uint64_t s0 = sha512_rotr(w[i-15], 1) ^ sha512_rotr(w[i-15], 8) ^ (w[i-15] >> 7);
        uint64_t s1 = sha512_rotr(w[i-2], 19) ^ sha512_rotr(w[i-2], 61) ^ (w[i-2] >> 6);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }

    uint64_t a = state[0], b = state[1], c = state[2], d = state[3];
    uint64_t e = state[4], f = state[5], g = state[6], hh = state[7];

    for (int i = 0; i < 80; i++) {
        uint64_t S1 = sha512_rotr(e, 14) ^ sha512_rotr(e, 18) ^ sha512_rotr(e, 41);
        uint64_t ch = (e & f) ^ (~e & g);
        uint64_t t1 = hh + S1 + ch + SHA512_K[i] + w[i];
        uint64_t S0 = sha512_rotr(a, 28) ^ sha512_rotr(a, 34) ^ sha512_rotr(a, 39);
        uint64_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint64_t t2 = S0 + maj;

        hh = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }

    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += hh;
}

__device__ inline void sha512_init(SHA512Ctx* ctx) {
    ctx->h[0] = 0x6a09e667f3bcc908ULL; ctx->h[1] = 0xbb67ae8584caa73bULL;
    ctx->h[2] = 0x3c6ef372fe94f82bULL; ctx->h[3] = 0xa54ff53a5f1d36f1ULL;
    ctx->h[4] = 0x510e527fade682d1ULL; ctx->h[5] = 0x9b05688c2b3e6c1fULL;
    ctx->h[6] = 0x1f83d9abfb41bd6bULL; ctx->h[7] = 0x5be0cd19137e2179ULL;
    ctx->buf_len = 0;
    ctx->total = 0;
}

__device__ inline void sha512_update(SHA512Ctx* ctx, const uint8_t* data, size_t len) {
    ctx->total += len;
    size_t offset = 0;

    if (ctx->buf_len > 0) {
        uint32_t fill = 128 - ctx->buf_len;
        uint32_t copy = (len < fill) ? (uint32_t)len : fill;
        for (uint32_t i = 0; i < copy; i++) ctx->buf[ctx->buf_len + i] = data[i];
        ctx->buf_len += copy;
        offset += copy;
        if (ctx->buf_len == 128) {
            sha512_compress(ctx->h, ctx->buf);
            ctx->buf_len = 0;
        }
    }

    while (offset + 128 <= len) {
        sha512_compress(ctx->h, data + offset);
        offset += 128;
    }

    while (offset < len) {
        ctx->buf[ctx->buf_len++] = data[offset++];
    }
}

__device__ inline void sha512_final(SHA512Ctx* ctx, uint8_t out[64]) {
    uint64_t bit_len = ctx->total * 8;
    ctx->buf[ctx->buf_len++] = 0x80;

    if (ctx->buf_len > 112) {
        while (ctx->buf_len < 128) ctx->buf[ctx->buf_len++] = 0;
        sha512_compress(ctx->h, ctx->buf);
        ctx->buf_len = 0;
    }

    while (ctx->buf_len < 120) ctx->buf[ctx->buf_len++] = 0;

    // SHA-512 length is 128-bit big-endian; upper 64 bits are 0 for our use.
    for (int i = 0; i < 8; i++) ctx->buf[112 + i] = 0;
    for (int i = 7; i >= 0; i--)
        ctx->buf[120 + (7 - i)] = (uint8_t)(bit_len >> (i * 8));

    sha512_compress(ctx->h, ctx->buf);

    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++)
            out[i * 8 + j] = (uint8_t)(ctx->h[i] >> (56 - j * 8));
    }
}

// One-shot SHA-512
__device__ inline void sha512(const uint8_t* data, size_t len, uint8_t out[64]) {
    SHA512Ctx ctx;
    sha512_init(&ctx);
    sha512_update(&ctx, data, len);
    sha512_final(&ctx, out);
}

// ============================================================================
// HMAC-SHA512
// ============================================================================

__device__ inline void hmac_sha512(
    const uint8_t* key, size_t key_len,
    const uint8_t* msg, size_t msg_len,
    uint8_t out[64])
{
    uint8_t k_buf[128];
    for (int i = 0; i < 128; i++) k_buf[i] = 0;

    if (key_len > 128) {
        sha512(key, key_len, k_buf);  // k_buf[0..63]=hash, [64..127]=0
    } else {
        for (size_t i = 0; i < key_len; i++) k_buf[i] = key[i];
    }

    uint8_t ipad[128], opad[128];
    for (int i = 0; i < 128; i++) {
        ipad[i] = k_buf[i] ^ 0x36;
        opad[i] = k_buf[i] ^ 0x5c;
    }

    // inner = SHA512(ipad || msg)
    SHA512Ctx inner;
    sha512_init(&inner);
    sha512_update(&inner, ipad, 128);
    sha512_update(&inner, msg, msg_len);
    uint8_t inner_hash[64];
    sha512_final(&inner, inner_hash);

    // outer = SHA512(opad || inner_hash)
    SHA512Ctx outer;
    sha512_init(&outer);
    sha512_update(&outer, opad, 128);
    sha512_update(&outer, inner_hash, 64);
    sha512_final(&outer, out);
}

// ============================================================================
// BIP-32 Extended Key
// ============================================================================

struct ExtendedKeyGPU {
    uint8_t key[32];           // Private key (32 bytes) or public key (33 bytes, compressed)
    uint8_t chain_code[32];    // Chain code (32 bytes)
    uint8_t depth;             // 0 for master
    uint32_t child_number;     // 0 for master
    uint8_t parent_fp[4];      // First 4 bytes of parent's Hash160(pubkey)
    bool is_private;           // true if this is a private extended key
};

// ============================================================================
// BIP-32 Master Key from Seed
// ============================================================================

// Derives master extended key from seed bytes.
// seed_len: typically 16, 32, or 64 bytes.
// Returns false if the derived key is invalid (>= n or == 0).
__device__ inline bool bip32_master_key(
    const uint8_t* seed, size_t seed_len,
    ExtendedKeyGPU* master)
{
    // I = HMAC-SHA512(Key="Bitcoin seed", Data=seed)
    const uint8_t btc_seed[] = "Bitcoin seed";  // 12 bytes
    uint8_t I[64];
    hmac_sha512(btc_seed, 12, seed, seed_len, I);

    // IL = I[0..31] = master secret key
    // IR = I[32..63] = master chain code
    Scalar sk;
    if (!scalar_from_bytes_strict_nonzero(I, &sk)) return false;

    for (int i = 0; i < 32; i++) master->key[i] = I[i];
    for (int i = 0; i < 32; i++) master->chain_code[i] = I[32 + i];
    master->depth = 0;
    master->child_number = 0;
    for (int i = 0; i < 4; i++) master->parent_fp[i] = 0;
    master->is_private = true;

    return true;
}

// ============================================================================
// BIP-32 Fingerprint (Hash160 of compressed pubkey)
// ============================================================================

// Compute the 4-byte fingerprint of a key: first 4 bytes of Hash160(compressed pubkey).
__device__ inline void bip32_fingerprint(
    const ExtendedKeyGPU* xkey,
    uint8_t fp[4])
{
    // Compute compressed public key
    uint8_t compressed[33];
    if (xkey->is_private) {
        Scalar sk;
        scalar_from_bytes(xkey->key, &sk);
        JacobianPoint P;
        scalar_mul_generator_w8(&sk, &P);
        point_to_compressed(&P, compressed);
    } else {
        // For public keys, key[0..32] is already the compressed pubkey
        for (int i = 0; i < 33; i++) compressed[i] = xkey->key[i];
    }

    // Hash160 = RIPEMD160(SHA256(compressed))
    uint8_t hash[20];
    hash160_pubkey(compressed, 33, hash);

    for (int i = 0; i < 4; i++) fp[i] = hash[i];
}

// ============================================================================
// BIP-32 Child Key Derivation
// ============================================================================

// Derive child key from parent. index >= 0x80000000 = hardened.
// Returns false if derived key is invalid.
__device__ inline bool bip32_derive_child(
    const ExtendedKeyGPU* parent,
    uint32_t index,
    ExtendedKeyGPU* child)
{
    bool hardened = (index >= 0x80000000U);

    // Build HMAC-SHA512 data:
    // Hardened: 0x00 || ser256(kpar) || ser32(index) = 37 bytes
    // Normal:   serP(point(kpar))   || ser32(index) = 37 bytes
    uint8_t data[37];

    if (hardened) {
        if (!parent->is_private) return false;  // Can't derive hardened from public
        data[0] = 0x00;
        for (int i = 0; i < 32; i++) data[1 + i] = parent->key[i];
    } else {
        // Compressed public key
        if (parent->is_private) {
            Scalar sk;
            scalar_from_bytes(parent->key, &sk);
            JacobianPoint P;
            scalar_mul_generator_w8(&sk, &P);
            point_to_compressed(&P, data);
        } else {
            for (int i = 0; i < 33; i++) data[i] = parent->key[i];
        }
    }

    // Append ser32(index) in big-endian
    int data_len = hardened ? 37 : 37;
    int idx_off = hardened ? 33 : 33;
    data[idx_off + 0] = (uint8_t)(index >> 24);
    data[idx_off + 1] = (uint8_t)(index >> 16);
    data[idx_off + 2] = (uint8_t)(index >> 8);
    data[idx_off + 3] = (uint8_t)(index);

    // I = HMAC-SHA512(Key=cpar, Data=data)
    uint8_t I[64];
    hmac_sha512(parent->chain_code, 32, data, data_len, I);

    // IL = I[0..31], IR = I[32..63]
    Scalar IL;
    scalar_from_bytes(I, &IL);

    // Check IL < n
    {
        Scalar raw;
        for (int i = 0; i < 4; i++) {
            uint64_t limb = 0;
            int base = (3 - i) * 8;
            for (int j = 0; j < 8; j++) limb = (limb << 8) | I[base + j];
            raw.limbs[i] = limb;
        }
        if (!scalar_eq(&raw, &IL)) return false;  // IL >= n
    }

    if (parent->is_private) {
        // child_key = (IL + kpar) mod n
        Scalar kpar;
        scalar_from_bytes(parent->key, &kpar);

        Scalar child_key;
        scalar_add(&IL, &kpar, &child_key);

        // Reduce mod n
        uint64_t borrow = 0;
        uint64_t tmp[4];
        for (int i = 0; i < 4; i++) {
            const uint64_t diff = child_key.limbs[i] - ORDER[i];
            const uint64_t borrow0 = (child_key.limbs[i] < ORDER[i]) ? 1ULL : 0ULL;
            tmp[i] = diff - borrow;
            borrow = borrow0 | ((diff < borrow) ? 1ULL : 0ULL);
        }
        if (borrow == 0) {
            for (int i = 0; i < 4; i++) child_key.limbs[i] = tmp[i];
        }

        if (scalar_is_zero(&child_key)) return false;

        scalar_to_bytes(&child_key, child->key);
        child->is_private = true;
    } else {
        // child_pubkey = point(IL) + Kpar
        JacobianPoint IL_point;
        scalar_mul_generator_w8(&IL, &IL_point);

        JacobianPoint Kpar;
        if (!point_from_compressed(parent->key, &Kpar)) return false;

        JacobianPoint child_pub;
        jacobian_add(&IL_point, &Kpar, &child_pub);

        if (child_pub.infinity) return false;

        point_to_compressed(&child_pub, child->key);
        child->is_private = false;
    }

    for (int i = 0; i < 32; i++) child->chain_code[i] = I[32 + i];
    child->depth = parent->depth + 1;
    child->child_number = index;
    bip32_fingerprint(parent, child->parent_fp);

    return true;
}

// Convenience: derive normal (non-hardened) child
__device__ inline bool bip32_derive_normal(
    const ExtendedKeyGPU* parent, uint32_t index,
    ExtendedKeyGPU* child)
{
    return bip32_derive_child(parent, index, child);
}

// Convenience: derive hardened child
__device__ inline bool bip32_derive_hardened(
    const ExtendedKeyGPU* parent, uint32_t index,
    ExtendedKeyGPU* child)
{
    return bip32_derive_child(parent, 0x80000000U | index, child);
}

// ============================================================================
// BIP-32 Public Key from Extended Key
// ============================================================================

// Get the compressed public key from an extended key.
__device__ inline bool bip32_public_key(
    const ExtendedKeyGPU* xkey,
    uint8_t compressed[33])
{
    if (xkey->is_private) {
        Scalar sk;
        scalar_from_bytes(xkey->key, &sk);
        JacobianPoint P;
        scalar_mul_generator_w8(&sk, &P);
        return point_to_compressed(&P, compressed);
    } else {
        for (int i = 0; i < 33; i++) compressed[i] = xkey->key[i];
        return true;
    }
}

// ============================================================================
// BIP-32 Serialize (78 bytes, standard format)
// ============================================================================

__device__ inline void bip32_serialize(
    const ExtendedKeyGPU* xkey,
    bool mainnet,
    uint8_t out[78])
{
    // Version bytes
    uint32_t version;
    if (xkey->is_private) {
        version = mainnet ? 0x0488ADE4U : 0x04358394U;  // xprv / tprv
    } else {
        version = mainnet ? 0x0488B21EU : 0x043587CFU;  // xpub / tpub
    }
    out[0] = (uint8_t)(version >> 24);
    out[1] = (uint8_t)(version >> 16);
    out[2] = (uint8_t)(version >> 8);
    out[3] = (uint8_t)(version);

    // Depth
    out[4] = xkey->depth;

    // Parent fingerprint
    for (int i = 0; i < 4; i++) out[5 + i] = xkey->parent_fp[i];

    // Child number (big-endian)
    out[9]  = (uint8_t)(xkey->child_number >> 24);
    out[10] = (uint8_t)(xkey->child_number >> 16);
    out[11] = (uint8_t)(xkey->child_number >> 8);
    out[12] = (uint8_t)(xkey->child_number);

    // Chain code
    for (int i = 0; i < 32; i++) out[13 + i] = xkey->chain_code[i];

    // Key data
    if (xkey->is_private) {
        out[45] = 0x00;
        for (int i = 0; i < 32; i++) out[46 + i] = xkey->key[i];
    } else {
        for (int i = 0; i < 33; i++) out[45 + i] = xkey->key[i];
    }
}

// ============================================================================
// BIP-32 to_public (convert private extended key to public)
// ============================================================================

__device__ inline bool bip32_to_public(
    const ExtendedKeyGPU* xpriv,
    ExtendedKeyGPU* xpub)
{
    if (!xpriv->is_private) {
        *xpub = *xpriv;
        return true;
    }

    Scalar sk;
    scalar_from_bytes(xpriv->key, &sk);
    JacobianPoint P;
    scalar_mul_generator_w8(&sk, &P);

    if (!point_to_compressed(&P, xpub->key)) return false;

    for (int i = 0; i < 32; i++) xpub->chain_code[i] = xpriv->chain_code[i];
    xpub->depth = xpriv->depth;
    xpub->child_number = xpriv->child_number;
    for (int i = 0; i < 4; i++) xpub->parent_fp[i] = xpriv->parent_fp[i];
    xpub->is_private = false;

    return true;
}

} // namespace cuda
} // namespace secp256k1

#endif // !SECP256K1_CUDA_LIMBS_32
