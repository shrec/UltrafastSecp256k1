#pragma once
#include <cstdint>
#include <cstddef>

// ============================================================================
// hash160 = RIPEMD-160(SHA-256(pubkey)) for CUDA.
//
// Register-only fast path, contributed by zeroxbee (VanityPlate,
// https://github.com/zeroxbee/VanityPlate):
//   - sha256_33(): SHA-256 of exactly 33 bytes (compressed pubkey). Padding
//     and length are baked into w[0..15]; rounds 16-63 expand a rolling w[16]
//     window, so there is no block[] or w[64] array in local memory.
//   - ripemd160_32(): RIPEMD-160 of exactly 32 bytes, fully unrolled with
//     literal word indices, shifts and constants (no r/s tables, no runtime-
//     indexed arrays, no per-step branches).
//   - hash160_pubkey() takes the fast SHA-256 for 33-byte keys and the generic
//     sha256() for any other length (65-byte uncompressed keys).
//
// ptxas -v on sm_120 for a kernel calling hash160_pubkey with a runtime
// length: 480 -> 128 bytes stack. Kernel-only A/B on RTX 5060 Ti:
// docs/benchmark_artifacts/cuda_hash160_unrolled_ab_2026-11-02_rtx5060ti.json.
// ============================================================================

namespace secp256k1 {
namespace cuda {

// ----------------------------------------------------------------------------
// Shared helpers
// ----------------------------------------------------------------------------
__device__ __forceinline__ uint32_t rotr32(uint32_t x, int n) {
    return (x >> n) | (x << (32 - n));
}
__device__ __forceinline__ uint32_t rotl32(uint32_t x, int n) {
    return (x << n) | (x >> (32 - n));
}

// ----------------------------------------------------------------------------
// SHA-256 round constants — constant cache, not per-thread stack
// ----------------------------------------------------------------------------
static __device__ __constant__ uint32_t sha256_K[64] = {
    0x428a2f98U,0x71374491U,0xb5c0fbcfU,0xe9b5dba5U,
    0x3956c25bU,0x59f111f1U,0x923f82a4U,0xab1c5ed5U,
    0xd807aa98U,0x12835b01U,0x243185beU,0x550c7dc3U,
    0x72be5d74U,0x80deb1feU,0x9bdc06a7U,0xc19bf174U,
    0xe49b69c1U,0xefbe4786U,0x0fc19dc6U,0x240ca1ccU,
    0x2de92c6fU,0x4a7484aaU,0x5cb0a9dcU,0x76f988daU,
    0x983e5152U,0xa831c66dU,0xb00327c8U,0xbf597fc7U,
    0xc6e00bf3U,0xd5a79147U,0x06ca6351U,0x14292967U,
    0x27b70a85U,0x2e1b2138U,0x4d2c6dfcU,0x53380d13U,
    0x650a7354U,0x766a0abbU,0x81c2c92eU,0x92722c85U,
    0xa2bfe8a1U,0xa81a664bU,0xc24b8b70U,0xc76c51a3U,
    0xd192e819U,0xd6990624U,0xf40e3585U,0x106aa070U,
    0x19a4c116U,0x1e376c08U,0x2748774cU,0x34b0bcb5U,
    0x391c0cb3U,0x4ed8aa4aU,0x5b9cca4fU,0x682e6ff3U,
    0x748f82eeU,0x78a5636fU,0x84c87814U,0x8cc70208U,
    0x90befffaU,0xa4506cebU,0xbef9a3f7U,0xc67178f2U
};

// ----------------------------------------------------------------------------
// Generic SHA-256 (kept for compatibility; NOT called from the BTC hot path).
// Still allocates block[128]+w[64] — avoid inlining into tight kernels.
// ----------------------------------------------------------------------------
__device__ __forceinline__ void sha256(const uint8_t* data, size_t len, uint8_t out[32]) {
    uint32_t h0=0x6a09e667U, h1=0xbb67ae85U, h2=0x3c6ef372U, h3=0xa54ff53aU;
    uint32_t h4=0x510e527fU, h5=0x9b05688cU, h6=0x1f83d9abU, h7=0x5be0cd19U;

    uint8_t block[128];
    for (int i = 0; i < 128; ++i) block[i] = 0;
    for (size_t i = 0; i < len; ++i) block[i] = data[i];
    block[len] = 0x80;
    const uint64_t bit_len = (uint64_t)len * 8ULL;
    const int total_blocks = ((len + 1 + 8) <= 64) ? 1 : 2;
    const int last = total_blocks * 64 - 8;
    for (int i = 0; i < 8; ++i)
        block[last + i] = (uint8_t)(bit_len >> (56 - i * 8));

    for (int b = 0; b < total_blocks; ++b) {
        uint32_t w[64];
        const int off = b * 64;
        for (int i = 0; i < 16; ++i) {
            const int j = off + i * 4;
            w[i] = ((uint32_t)block[j]<<24)|((uint32_t)block[j+1]<<16)|
                   ((uint32_t)block[j+2]<<8)|(uint32_t)block[j+3];
        }
        for (int i = 16; i < 64; ++i) {
            uint32_t s0 = rotr32(w[i-15],7)^rotr32(w[i-15],18)^(w[i-15]>>3);
            uint32_t s1 = rotr32(w[i-2],17)^rotr32(w[i-2],19)^(w[i-2]>>10);
            w[i] = w[i-16]+s0+w[i-7]+s1;
        }
        uint32_t a=h0,bv=h1,c=h2,d=h3,e=h4,f=h5,g=h6,h=h7;
        for (int i = 0; i < 64; ++i) {
            uint32_t S1=rotr32(e,6)^rotr32(e,11)^rotr32(e,25);
            uint32_t ch=(e&f)^(~e&g);
            uint32_t t1=h+S1+ch+sha256_K[i]+w[i];
            uint32_t S0=rotr32(a,2)^rotr32(a,13)^rotr32(a,22);
            uint32_t maj=(a&bv)^(a&c)^(bv&c);
            h=g; g=f; f=e; e=d+t1; d=c; c=bv; bv=a; a=t1+S0+maj;
        }
        h0+=a; h1+=bv; h2+=c; h3+=d; h4+=e; h5+=f; h6+=g; h7+=h;
    }
    const uint32_t hv[8]={h0,h1,h2,h3,h4,h5,h6,h7};
    for (int i=0;i<8;++i){
        out[i*4+0]=(uint8_t)(hv[i]>>24); out[i*4+1]=(uint8_t)(hv[i]>>16);
        out[i*4+2]=(uint8_t)(hv[i]>>8);  out[i*4+3]=(uint8_t)hv[i];
    }
}

// ----------------------------------------------------------------------------
// Optimised SHA-256 for exactly 33 bytes (compressed secp256k1 pubkey).
//
// The 33-byte message fits in a single 512-bit (64-byte) block:
//   bytes  0-32: input data
//   byte   33  : 0x80  (padding)
//   bytes 34-55: 0x00  (padding)
//   bytes 56-63: 0x00 00 00 00 00 00 01 08  (264 bits, big-endian)
//
// The 64-byte block maps to exactly 16 big-endian uint32 words w[0..15].
// We load those 16 words into registers, then expand rounds 16-63 in-place
// using a rolling mod-16 index — no heap/stack allocation at all.
// ----------------------------------------------------------------------------
__device__ __forceinline__ void sha256_33(const uint8_t data[33], uint8_t out[32]) {
    // Load message block directly into 16 registers (no block[] buffer)
    uint32_t w[16];
    w[0]  = ((uint32_t)data[0] <<24)|((uint32_t)data[1] <<16)|((uint32_t)data[2] <<8)|data[3];
    w[1]  = ((uint32_t)data[4] <<24)|((uint32_t)data[5] <<16)|((uint32_t)data[6] <<8)|data[7];
    w[2]  = ((uint32_t)data[8] <<24)|((uint32_t)data[9] <<16)|((uint32_t)data[10]<<8)|data[11];
    w[3]  = ((uint32_t)data[12]<<24)|((uint32_t)data[13]<<16)|((uint32_t)data[14]<<8)|data[15];
    w[4]  = ((uint32_t)data[16]<<24)|((uint32_t)data[17]<<16)|((uint32_t)data[18]<<8)|data[19];
    w[5]  = ((uint32_t)data[20]<<24)|((uint32_t)data[21]<<16)|((uint32_t)data[22]<<8)|data[23];
    w[6]  = ((uint32_t)data[24]<<24)|((uint32_t)data[25]<<16)|((uint32_t)data[26]<<8)|data[27];
    w[7]  = ((uint32_t)data[28]<<24)|((uint32_t)data[29]<<16)|((uint32_t)data[30]<<8)|data[31];
    // w[8]: data[32] in the top byte, 0x80 padding in the next, then zeros
    w[8]  = ((uint32_t)data[32]<<24)|0x00800000U;
    w[9]  = 0U; w[10] = 0U; w[11] = 0U; w[12] = 0U; w[13] = 0U;
    w[14] = 0U;       // message length high word: 264 < 2^32, so 0
    w[15] = 264U;     // message length low word:  33 * 8 = 264 bits

    uint32_t a=0x6a09e667U, b=0xbb67ae85U, c=0x3c6ef372U, d=0xa54ff53aU;
    uint32_t e=0x510e527fU, f=0x9b05688cU, g=0x1f83d9abU, h=0x5be0cd19U;

// One SHA-256 round using the rolling window
#define S256_RND(i) do { \
    uint32_t S1 = rotr32(e,6)^rotr32(e,11)^rotr32(e,25); \
    uint32_t ch = (e&f)^(~e&g); \
    uint32_t t1 = h + S1 + ch + sha256_K[i] + w[(i)&15]; \
    uint32_t S0 = rotr32(a,2)^rotr32(a,13)^rotr32(a,22); \
    uint32_t maj = (a&b)^(a&c)^(b&c); \
    h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+S0+maj; \
} while(0)

// Expand one message schedule word in-place (rolling w[16])
#define S256_EXP(i) do { \
    uint32_t wi15 = w[((i)-15)&15]; \
    uint32_t wi2  = w[((i)- 2)&15]; \
    uint32_t s0 = rotr32(wi15,7)^rotr32(wi15,18)^(wi15>>3); \
    uint32_t s1 = rotr32(wi2,17)^rotr32(wi2,19)^(wi2>>10); \
    w[(i)&15] = w[(i)&15] + s0 + w[((i)-7)&15] + s1; \
} while(0)

    // Rounds 0-15: use the original message words as-is
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

    a+=0x6a09e667U; b+=0xbb67ae85U; c+=0x3c6ef372U; d+=0xa54ff53aU;
    e+=0x510e527fU; f+=0x9b05688cU; g+=0x1f83d9abU; h+=0x5be0cd19U;

    // Store big-endian output (8 uint32 → 32 bytes)
    out[ 0]=(uint8_t)(a>>24); out[ 1]=(uint8_t)(a>>16); out[ 2]=(uint8_t)(a>>8); out[ 3]=(uint8_t)a;
    out[ 4]=(uint8_t)(b>>24); out[ 5]=(uint8_t)(b>>16); out[ 6]=(uint8_t)(b>>8); out[ 7]=(uint8_t)b;
    out[ 8]=(uint8_t)(c>>24); out[ 9]=(uint8_t)(c>>16); out[10]=(uint8_t)(c>>8); out[11]=(uint8_t)c;
    out[12]=(uint8_t)(d>>24); out[13]=(uint8_t)(d>>16); out[14]=(uint8_t)(d>>8); out[15]=(uint8_t)d;
    out[16]=(uint8_t)(e>>24); out[17]=(uint8_t)(e>>16); out[18]=(uint8_t)(e>>8); out[19]=(uint8_t)e;
    out[20]=(uint8_t)(f>>24); out[21]=(uint8_t)(f>>16); out[22]=(uint8_t)(f>>8); out[23]=(uint8_t)f;
    out[24]=(uint8_t)(g>>24); out[25]=(uint8_t)(g>>16); out[26]=(uint8_t)(g>>8); out[27]=(uint8_t)g;
    out[28]=(uint8_t)(h>>24); out[29]=(uint8_t)(h>>16); out[30]=(uint8_t)(h>>8); out[31]=(uint8_t)h;
}

// ----------------------------------------------------------------------------
// RIPEMD-160 round functions and step, for the fully unrolled ripemd160_32().
// Every word index, shift amount, function and constant below is a literal:
// no tables, no runtime-indexed register arrays (which the compiler turned
// into ~15-way compare/select chains per step), no per-step branches.
// ----------------------------------------------------------------------------
#define RMD_F1(x, y, z) ((x) ^ (y) ^ (z))
#define RMD_F2(x, y, z) (((x) & (y)) | (~(x) & (z)))
#define RMD_F3(x, y, z) (((x) | ~(y)) ^ (z))
#define RMD_F4(x, y, z) (((x) & (z)) | ((y) & ~(z)))
#define RMD_F5(x, y, z) ((x) ^ ((y) | ~(z)))

// One step with rotating variable roles: instead of shifting a<-e<-d<-c<-b
// every step, the caller rotates which variable plays which role. After 80
// steps (a multiple of 5) the roles line up with the original names again.
#define RMD_STEP(a, b, c, d, e, F, x, k, s) do { \
    (a) = rotl32((a) + F(b, c, d) + (x) + (k), (s)) + (e); \
    (c) = rotl32((c), 10); \
} while (0)

// ----------------------------------------------------------------------------
// Optimised RIPEMD-160 for exactly 32 bytes (SHA-256 output).
//
// The 32-byte message fits in one 512-bit (64-byte) block:
//   bytes  0-31: input data
//   byte   32  : 0x80  (padding)
//   bytes 33-55: 0x00
//   bytes 56-63: 256 bits, little-endian (RIPEMD is little-endian unlike SHA)
//
// Maps to exactly 16 little-endian uint32 words X[0..15] loaded into
// registers — no block[] staging buffer needed.
// ----------------------------------------------------------------------------
__device__ __forceinline__ void ripemd160_32(const uint8_t data[32], uint8_t out[20]) {
    // Build X[16] directly in registers — no block[64] buffer
    uint32_t X[16];
    X[ 0] = (uint32_t)data[ 0]|((uint32_t)data[ 1]<<8)|((uint32_t)data[ 2]<<16)|((uint32_t)data[ 3]<<24);
    X[ 1] = (uint32_t)data[ 4]|((uint32_t)data[ 5]<<8)|((uint32_t)data[ 6]<<16)|((uint32_t)data[ 7]<<24);
    X[ 2] = (uint32_t)data[ 8]|((uint32_t)data[ 9]<<8)|((uint32_t)data[10]<<16)|((uint32_t)data[11]<<24);
    X[ 3] = (uint32_t)data[12]|((uint32_t)data[13]<<8)|((uint32_t)data[14]<<16)|((uint32_t)data[15]<<24);
    X[ 4] = (uint32_t)data[16]|((uint32_t)data[17]<<8)|((uint32_t)data[18]<<16)|((uint32_t)data[19]<<24);
    X[ 5] = (uint32_t)data[20]|((uint32_t)data[21]<<8)|((uint32_t)data[22]<<16)|((uint32_t)data[23]<<24);
    X[ 6] = (uint32_t)data[24]|((uint32_t)data[25]<<8)|((uint32_t)data[26]<<16)|((uint32_t)data[27]<<24);
    X[ 7] = (uint32_t)data[28]|((uint32_t)data[29]<<8)|((uint32_t)data[30]<<16)|((uint32_t)data[31]<<24);
    X[ 8] = 0x00000080U;   // 0x80 padding byte
    X[ 9] = 0U; X[10] = 0U; X[11] = 0U; X[12] = 0U; X[13] = 0U;
    X[14] = 256U;           // 32 * 8 = 256 bits, little-endian low word
    X[15] = 0U;

    uint32_t h0=0x67452301U, h1=0xEFCDAB89U, h2=0x98BADCFEU,
             h3=0x10325476U, h4=0xC3D2E1F0U;

    uint32_t a=h0, b=h1, c=h2, d=h3, e=h4;
    uint32_t a2=h0, b2=h1, c2=h2, d2=h3, e2=h4;

    // 80 left-line and 80 right-line steps, interleaved (generated code)
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

    uint32_t tmp = h1 + c + d2;
    h1 = h2 + d + e2;
    h2 = h3 + e + a2;
    h3 = h4 + a + b2;
    h4 = h0 + b + c2;
    h0 = tmp;

    // Store little-endian output (5 uint32 → 20 bytes)
    out[ 0]=(uint8_t)h0; out[ 1]=(uint8_t)(h0>>8); out[ 2]=(uint8_t)(h0>>16); out[ 3]=(uint8_t)(h0>>24);
    out[ 4]=(uint8_t)h1; out[ 5]=(uint8_t)(h1>>8); out[ 6]=(uint8_t)(h1>>16); out[ 7]=(uint8_t)(h1>>24);
    out[ 8]=(uint8_t)h2; out[ 9]=(uint8_t)(h2>>8); out[10]=(uint8_t)(h2>>16); out[11]=(uint8_t)(h2>>24);
    out[12]=(uint8_t)h3; out[13]=(uint8_t)(h3>>8); out[14]=(uint8_t)(h3>>16); out[15]=(uint8_t)(h3>>24);
    out[16]=(uint8_t)h4; out[17]=(uint8_t)(h4>>8); out[18]=(uint8_t)(h4>>16); out[19]=(uint8_t)(h4>>24);
}
#undef RMD_STEP
#undef RMD_F1
#undef RMD_F2
#undef RMD_F3
#undef RMD_F4
#undef RMD_F5

// ----------------------------------------------------------------------------
// hash160_pubkey: pubkey -> SHA-256 -> RIPEMD-160 -> 20 bytes.
// 33-byte (compressed) keys take sha256_33(); any other length (65-byte
// uncompressed keys) takes the generic sha256().
// ----------------------------------------------------------------------------
__device__ __forceinline__ void hash160_pubkey(const uint8_t* pubkey,
                                               size_t pubkey_len,
                                               uint8_t out[20]) {
    uint8_t sha[32];
    if (pubkey_len == 33) sha256_33(pubkey, sha);
    else sha256(pubkey, pubkey_len, sha);
    ripemd160_32(sha, out);
}

} // namespace cuda
} // namespace secp256k1
