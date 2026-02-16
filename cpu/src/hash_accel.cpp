// ============================================================================
// Accelerated Hashing — SHA-256 / RIPEMD-160 / Hash160
// ============================================================================
//
// Implementation tiers:
//   Tier 0: SCALAR   — Optimized portable C++, unrolled rounds
//   Tier 1: SHA-NI   — Intel SHA Extensions (hardware SHA-256)
//   Tier 2: AVX2     — 4-way multi-buffer SHA-256
//
// All fixed-length hot-path functions (sha256_33, ripemd160_32, hash160_33)
// use precomputed padding to eliminate branches and buffer management.
//
// SHA-256(33 bytes):
//   Block = [33 data bytes | 0x80 | 22 zeros | 0x00 0x00 0x01 0x08]
//   Padding bytes 33..63 are constant → precomputed.
//   Only 1 compression call needed (single 64-byte block).
//
// RIPEMD-160(32 bytes):
//   Block = [32 data bytes | 0x80 | 23 zeros | 0x00 0x01 0x00 0x00 ...]
//   Only 1 compression call needed (single 64-byte block).
// ============================================================================

#include "secp256k1/hash_accel.hpp"
#include "secp256k1/sha256.hpp"

#include <cstring>

// Architecture detection
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    #define SECP256K1_X86_TARGET 1
    #ifdef _MSC_VER
        #include <intrin.h>
    #else
        #include <cpuid.h>
    #endif
#endif

namespace secp256k1::hash {

// ============================================================================
// Feature Detection — cached (CPUID is expensive: ~100+ cycles per call)
// ============================================================================

#ifdef SECP256K1_X86_TARGET
static struct CpuFeatures {
    bool sha_ni;
    bool avx2;
    bool avx512;

    CpuFeatures() noexcept {
    #ifdef _MSC_VER
        int info[4];
        __cpuidex(info, 7, 0);
        sha_ni = (info[1] & (1 << 29)) != 0;
        avx2   = (info[1] & (1 << 5))  != 0;
        avx512 = (info[1] & (1 << 16)) != 0;
    #elif defined(__GNUC__) || defined(__clang__)
        unsigned int eax, ebx, ecx, edx;
        if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
            sha_ni = (ebx & (1 << 29)) != 0;
            avx2   = (ebx & (1 << 5))  != 0;
            avx512 = (ebx & (1 << 16)) != 0;
        } else {
            sha_ni = avx2 = avx512 = false;
        }
    #endif
    }
} const g_cpu_features;
#endif

bool sha_ni_available() noexcept {
#ifdef SECP256K1_X86_TARGET
    return g_cpu_features.sha_ni;
#else
    return false;
#endif
}

bool avx2_available() noexcept {
#ifdef SECP256K1_X86_TARGET
    return g_cpu_features.avx2;
#else
    return false;
#endif
}

bool avx512_available() noexcept {
#ifdef SECP256K1_X86_TARGET
    return g_cpu_features.avx512;
#else
    return false;
#endif
}

HashTier detect_hash_tier() noexcept {
    // SHA-NI usually coexists with AVX2 on modern CPUs (Zen, Ice Lake+)
    // SHA-NI single-message is often faster than multi-buffer AVX2 for
    // sequential work. For batch, AVX2 multi-buffer wins.
    if (sha_ni_available()) return HashTier::SHA_NI;
    if (avx2_available())   return HashTier::AVX2;
    return HashTier::SCALAR;
}

const char* hash_tier_name(HashTier tier) noexcept {
    switch (tier) {
        case HashTier::SHA_NI:  return "SHA-NI";
        case HashTier::AVX2:    return "AVX2";
        case HashTier::AVX512:  return "AVX-512";
        default:                return "Scalar";
    }
}

// ============================================================================
// SHA-256 Constants
// ============================================================================

static constexpr std::uint32_t SHA256_K[64] = {
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

static constexpr std::uint32_t SHA256_IV[8] = {
    0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
    0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u
};

// ============================================================================
// RIPEMD-160 Constants
// ============================================================================

static constexpr std::uint32_t RIPEMD160_IV[5] = {
    0x67452301u, 0xEFCDAB89u, 0x98BADCFEu, 0x10325476u, 0xC3D2E1F0u
};

// ============================================================================
// Utility functions
// ============================================================================

static inline std::uint32_t rotr32(std::uint32_t x, int n) noexcept {
    return (x >> n) | (x << (32 - n));
}

static inline std::uint32_t rotl32(std::uint32_t x, int n) noexcept {
    return (x << n) | (x >> (32 - n));
}

static inline std::uint32_t load_be32(const std::uint8_t* p) noexcept {
    return (std::uint32_t(p[0]) << 24) | (std::uint32_t(p[1]) << 16) |
           (std::uint32_t(p[2]) << 8)  | std::uint32_t(p[3]);
}

static inline void store_be32(std::uint8_t* p, std::uint32_t v) noexcept {
    p[0] = std::uint8_t(v >> 24);
    p[1] = std::uint8_t(v >> 16);
    p[2] = std::uint8_t(v >> 8);
    p[3] = std::uint8_t(v);
}

static inline std::uint32_t load_le32(const std::uint8_t* p) noexcept {
    return std::uint32_t(p[0]) | (std::uint32_t(p[1]) << 8) |
           (std::uint32_t(p[2]) << 16) | (std::uint32_t(p[3]) << 24);
}

static inline void store_le32(std::uint8_t* p, std::uint32_t v) noexcept {
    p[0] = std::uint8_t(v);
    p[1] = std::uint8_t(v >> 8);
    p[2] = std::uint8_t(v >> 16);
    p[3] = std::uint8_t(v >> 24);
}

// ============================================================================
// SCALAR SHA-256 — Optimized portable C++ with fully unrolled rounds
// ============================================================================

namespace scalar {

// SHA-256 round macro
#define SHA256_ROUND(a, b, c, d, e, f, g, h, ki, wi)          \
    do {                                                        \
        std::uint32_t S1 = rotr32(e, 6) ^ rotr32(e, 11) ^ rotr32(e, 25); \
        std::uint32_t ch = (e & f) ^ (~e & g);                 \
        std::uint32_t temp1 = h + S1 + ch + ki + wi;           \
        std::uint32_t S0 = rotr32(a, 2) ^ rotr32(a, 13) ^ rotr32(a, 22); \
        std::uint32_t maj = (a & b) ^ (a & c) ^ (b & c);       \
        std::uint32_t temp2 = S0 + maj;                         \
        h = g; g = f; f = e;                                    \
        e = d + temp1;                                          \
        d = c; c = b; b = a;                                    \
        a = temp1 + temp2;                                      \
    } while(0)

void sha256_compress(const std::uint8_t block[64], std::uint32_t state[8]) noexcept {
    std::uint32_t w[64];

    // Load message words (big-endian)
    for (int i = 0; i < 16; ++i) {
        w[i] = load_be32(block + i * 4);
    }

    // Message schedule expansion
    for (int i = 16; i < 64; ++i) {
        std::uint32_t s0 = rotr32(w[i-15], 7) ^ rotr32(w[i-15], 18) ^ (w[i-15] >> 3);
        std::uint32_t s1 = rotr32(w[i-2], 17) ^ rotr32(w[i-2], 19) ^ (w[i-2] >> 10);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }

    std::uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
    std::uint32_t e = state[4], f = state[5], g = state[6], h = state[7];

    // 64 rounds — let the compiler unroll
    for (int i = 0; i < 64; ++i) {
        std::uint32_t S1 = rotr32(e, 6) ^ rotr32(e, 11) ^ rotr32(e, 25);
        std::uint32_t ch = (e & f) ^ (~e & g);
        std::uint32_t temp1 = h + S1 + ch + SHA256_K[i] + w[i];
        std::uint32_t S0 = rotr32(a, 2) ^ rotr32(a, 13) ^ rotr32(a, 22);
        std::uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        std::uint32_t temp2 = S0 + maj;
        h = g; g = f; f = e; e = d + temp1;
        d = c; c = b; b = a; a = temp1 + temp2;
    }

    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}

#undef SHA256_ROUND

void sha256_33(const std::uint8_t* pubkey33, std::uint8_t* out32) noexcept {
    // SHA-256(33 bytes) = single 64-byte block
    // Layout: [33 data bytes] [0x80] [22 zeros] [64-bit big-endian length = 264 bits]
    // Length 33 * 8 = 264 = 0x108
    alignas(16) std::uint8_t block[64];
    std::memcpy(block, pubkey33, 33);
    block[33] = 0x80;
    std::memset(block + 34, 0, 22);     // bytes 34..55 = 0
    // Length in bits = 264 = 0x00000108 (big-endian at bytes 56..63)
    block[56] = 0; block[57] = 0; block[58] = 0; block[59] = 0;
    block[60] = 0; block[61] = 0; block[62] = 0x01; block[63] = 0x08;

    std::uint32_t state[8];
    std::memcpy(state, SHA256_IV, sizeof(state));
    sha256_compress(block, state);

    // Store big-endian
    for (int i = 0; i < 8; ++i) {
        store_be32(out32 + i * 4, state[i]);
    }
}

void sha256_32(const std::uint8_t* in32, std::uint8_t* out32) noexcept {
    // SHA-256(32 bytes) = single 64-byte block
    // Layout: [32 data bytes] [0x80] [23 zeros] [64-bit big-endian length = 256 bits]
    alignas(16) std::uint8_t block[64];
    std::memcpy(block, in32, 32);
    block[32] = 0x80;
    std::memset(block + 33, 0, 23);     // bytes 33..55 = 0
    block[56] = 0; block[57] = 0; block[58] = 0; block[59] = 0;
    block[60] = 0; block[61] = 0; block[62] = 0x01; block[63] = 0x00;

    std::uint32_t state[8];
    std::memcpy(state, SHA256_IV, sizeof(state));
    sha256_compress(block, state);

    for (int i = 0; i < 8; ++i) {
        store_be32(out32 + i * 4, state[i]);
    }
}

// ── RIPEMD-160 Scalar ────────────────────────────────────────────────────────

// RIPEMD-160 boolean functions
static inline std::uint32_t rmd_f(int j, std::uint32_t x, std::uint32_t y, std::uint32_t z) noexcept {
    if (j < 16) return x ^ y ^ z;
    if (j < 32) return (x & y) | (~x & z);
    if (j < 48) return (x | ~y) ^ z;
    if (j < 64) return (x & z) | (y & ~z);
    return x ^ (y | ~z);
}

void ripemd160_compress(const std::uint8_t block[64], std::uint32_t state[5]) noexcept {
    std::uint32_t X[16];
    for (int i = 0; i < 16; ++i)
        X[i] = load_le32(block + i * 4);

    static constexpr int rl[80] = {
        0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
        7,4,13,1,10,6,15,3,12,0,9,5,2,14,11,8,
        3,10,14,4,9,15,8,1,2,7,0,6,13,11,5,12,
        1,9,11,10,0,8,12,4,13,3,7,15,14,5,6,2,
        4,0,5,9,7,12,2,10,14,1,3,8,11,6,15,13
    };
    static constexpr int rr[80] = {
        5,14,7,0,9,2,11,4,13,6,15,8,1,10,3,12,
        6,11,3,7,0,13,5,10,14,15,8,12,4,9,1,2,
        15,5,1,3,7,14,6,9,11,8,12,2,10,0,4,13,
        8,6,4,1,3,11,15,0,5,12,2,13,9,7,10,14,
        12,15,10,4,1,5,8,7,6,2,13,14,0,3,9,11
    };
    static constexpr int sl[80] = {
        11,14,15,12,5,8,7,9,11,13,14,15,6,7,9,8,
        7,6,8,13,11,9,7,15,7,12,15,9,11,7,13,12,
        11,13,6,7,14,9,13,15,14,8,13,6,5,12,7,5,
        11,12,14,15,14,15,9,8,9,14,5,6,8,6,5,12,
        9,15,5,11,6,8,13,12,5,12,13,14,11,8,5,6
    };
    static constexpr int sr[80] = {
        8,9,9,11,13,15,15,5,7,7,8,11,14,14,12,6,
        9,13,15,7,12,8,9,11,7,7,12,7,6,15,13,11,
        9,7,15,11,8,6,6,14,12,13,5,14,13,13,7,5,
        15,5,8,11,14,14,6,14,6,9,12,9,12,5,15,8,
        8,5,12,9,12,5,14,6,8,13,6,5,15,13,11,11
    };
    static constexpr std::uint32_t KL[5] = {0, 0x5A827999u, 0x6ED9EBA1u, 0x8F1BBCDCu, 0xA953FD4Eu};
    static constexpr std::uint32_t KR[5] = {0x50A28BE6u, 0x5C4DD124u, 0x6D703EF3u, 0x7A6D76E9u, 0};

    std::uint32_t al = state[0], bl = state[1], cl = state[2], dl = state[3], el = state[4];
    std::uint32_t ar = state[0], br = state[1], cr = state[2], dr = state[3], er = state[4];

    for (int j = 0; j < 80; ++j) {
        std::uint32_t tl = al + rmd_f(j, bl, cl, dl) + X[rl[j]] + KL[j/16];
        tl = rotl32(tl, sl[j]) + el;
        al = el; el = dl; dl = rotl32(cl, 10); cl = bl; bl = tl;

        std::uint32_t tr = ar + rmd_f(79 - j, br, cr, dr) + X[rr[j]] + KR[j/16];
        tr = rotl32(tr, sr[j]) + er;
        ar = er; er = dr; dr = rotl32(cr, 10); cr = br; br = tr;
    }

    std::uint32_t t = state[1] + cl + dr;
    state[1] = state[2] + dl + er;
    state[2] = state[3] + el + ar;
    state[3] = state[4] + al + br;
    state[4] = state[0] + bl + cr;
    state[0] = t;
}

void ripemd160_32(const std::uint8_t* in32, std::uint8_t* out20) noexcept {
    // RIPEMD-160(32 bytes) = single 64-byte block
    // Layout: [32 data bytes] [0x80] [23 zeros] [64-bit little-endian length = 256 bits]
    // Length 32 * 8 = 256 = 0x100
    alignas(16) std::uint8_t block[64];
    std::memcpy(block, in32, 32);
    block[32] = 0x80;
    std::memset(block + 33, 0, 23);    // bytes 33..55 = 0
    // Length in bits = 256 = 0x00000100 (little-endian at bytes 56..63)
    block[56] = 0x00; block[57] = 0x01; block[58] = 0; block[59] = 0;
    block[60] = 0; block[61] = 0; block[62] = 0; block[63] = 0;

    std::uint32_t state[5];
    std::memcpy(state, RIPEMD160_IV, sizeof(state));
    ripemd160_compress(block, state);

    // Store little-endian
    for (int i = 0; i < 5; ++i) {
        store_le32(out20 + i * 4, state[i]);
    }
}

void hash160_33(const std::uint8_t* pubkey33, std::uint8_t* out20) noexcept {
    std::uint8_t sha_out[32];
    sha256_33(pubkey33, sha_out);
    ripemd160_32(sha_out, out20);
}

} // namespace scalar

// ============================================================================
// SHA-NI (Intel SHA Extensions) — Hardware-accelerated SHA-256
// ============================================================================

#ifdef SECP256K1_X86_TARGET

// SHA-NI requires SSE4.1 + SHA instructions
// MSVC: /arch:AVX2 enables; Clang/GCC: -msha -msse4.1
#if defined(_MSC_VER) || defined(__SHA__)

#include <immintrin.h>

namespace shani {

// Helper: Perform 4 SHA-256 rounds on two __m128i state registers
#define SHA256_SHANI_4ROUNDS(state0, state1, msg, k_offset)               \
    do {                                                                    \
        __m128i tmp = _mm_add_epi32(msg, _mm_loadu_si128(                  \
            reinterpret_cast<const __m128i*>(SHA256_K + k_offset)));        \
        state1 = _mm_sha256rnds2_epu32(state1, state0, tmp);              \
        tmp = _mm_shuffle_epi32(tmp, 0x0E);                                \
        state0 = _mm_sha256rnds2_epu32(state0, state1, tmp);              \
    } while(0)

#ifdef _MSC_VER
// clang-cl: supports __attribute__((target)) despite defining _MSC_VER
// Pure MSVC: doesn't support target attributes, intrinsics always available
#if defined(__clang__)
#define SHANI_FUNC_ATTR __attribute__((target("sha,sse4.1")))
#else
#define SHANI_FUNC_ATTR
#endif
#else
#define SHANI_FUNC_ATTR __attribute__((target("sha,sse4.1")))
#endif

SHANI_FUNC_ATTR
void sha256_compress(const std::uint8_t block[64], std::uint32_t state[8]) noexcept {
    // Load state into two 128-bit registers
    // state0 = [A B E F], state1 = [C D G H]  (SHA-NI layout)
    __m128i abef = _mm_loadu_si128(reinterpret_cast<const __m128i*>(state));
    __m128i cdgh = _mm_loadu_si128(reinterpret_cast<const __m128i*>(state + 4));

    // SHA-NI expects state in specific lane order
    // Rearrange from [A B C D] [E F G H] to [A B E F] [C D G H]
    __m128i tmp = _mm_shuffle_epi32(abef, 0xB1);  // [B A D C]
    cdgh = _mm_shuffle_epi32(cdgh, 0x1B);          // [H G F E]
    __m128i state0 = _mm_alignr_epi8(tmp, cdgh, 8);  // [A B E F]
    __m128i state1 = _mm_blend_epi16(cdgh, tmp, 0xF0); // [C D G H]

    // Save original state for final addition
    __m128i state0_save = state0;
    __m128i state1_save = state1;

    // Load and byte-swap message words
    const __m128i MASK = _mm_set_epi64x(0x0C0D0E0F08090A0BULL, 0x0405060700010203ULL);
    __m128i msg0 = _mm_shuffle_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i*>(block +  0)), MASK);
    __m128i msg1 = _mm_shuffle_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i*>(block + 16)), MASK);
    __m128i msg2 = _mm_shuffle_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i*>(block + 32)), MASK);
    __m128i msg3 = _mm_shuffle_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i*>(block + 48)), MASK);

    // Rounds 0-3
    SHA256_SHANI_4ROUNDS(state0, state1, msg0, 0);
    // Rounds 4-7
    SHA256_SHANI_4ROUNDS(state0, state1, msg1, 4);
    // Rounds 8-11
    SHA256_SHANI_4ROUNDS(state0, state1, msg2, 8);
    // Rounds 12-15
    SHA256_SHANI_4ROUNDS(state0, state1, msg3, 12);

    // Rounds 16-19
    msg0 = _mm_sha256msg1_epu32(msg0, msg1);
    msg0 = _mm_add_epi32(msg0, _mm_alignr_epi8(msg3, msg2, 4));
    msg0 = _mm_sha256msg2_epu32(msg0, msg3);
    SHA256_SHANI_4ROUNDS(state0, state1, msg0, 16);

    // Rounds 20-23
    msg1 = _mm_sha256msg1_epu32(msg1, msg2);
    msg1 = _mm_add_epi32(msg1, _mm_alignr_epi8(msg0, msg3, 4));
    msg1 = _mm_sha256msg2_epu32(msg1, msg0);
    SHA256_SHANI_4ROUNDS(state0, state1, msg1, 20);

    // Rounds 24-27
    msg2 = _mm_sha256msg1_epu32(msg2, msg3);
    msg2 = _mm_add_epi32(msg2, _mm_alignr_epi8(msg1, msg0, 4));
    msg2 = _mm_sha256msg2_epu32(msg2, msg1);
    SHA256_SHANI_4ROUNDS(state0, state1, msg2, 24);

    // Rounds 28-31
    msg3 = _mm_sha256msg1_epu32(msg3, msg0);
    msg3 = _mm_add_epi32(msg3, _mm_alignr_epi8(msg2, msg1, 4));
    msg3 = _mm_sha256msg2_epu32(msg3, msg2);
    SHA256_SHANI_4ROUNDS(state0, state1, msg3, 28);

    // Rounds 32-35
    msg0 = _mm_sha256msg1_epu32(msg0, msg1);
    msg0 = _mm_add_epi32(msg0, _mm_alignr_epi8(msg3, msg2, 4));
    msg0 = _mm_sha256msg2_epu32(msg0, msg3);
    SHA256_SHANI_4ROUNDS(state0, state1, msg0, 32);

    // Rounds 36-39
    msg1 = _mm_sha256msg1_epu32(msg1, msg2);
    msg1 = _mm_add_epi32(msg1, _mm_alignr_epi8(msg0, msg3, 4));
    msg1 = _mm_sha256msg2_epu32(msg1, msg0);
    SHA256_SHANI_4ROUNDS(state0, state1, msg1, 36);

    // Rounds 40-43
    msg2 = _mm_sha256msg1_epu32(msg2, msg3);
    msg2 = _mm_add_epi32(msg2, _mm_alignr_epi8(msg1, msg0, 4));
    msg2 = _mm_sha256msg2_epu32(msg2, msg1);
    SHA256_SHANI_4ROUNDS(state0, state1, msg2, 40);

    // Rounds 44-47
    msg3 = _mm_sha256msg1_epu32(msg3, msg0);
    msg3 = _mm_add_epi32(msg3, _mm_alignr_epi8(msg2, msg1, 4));
    msg3 = _mm_sha256msg2_epu32(msg3, msg2);
    SHA256_SHANI_4ROUNDS(state0, state1, msg3, 44);

    // Rounds 48-51
    msg0 = _mm_sha256msg1_epu32(msg0, msg1);
    msg0 = _mm_add_epi32(msg0, _mm_alignr_epi8(msg3, msg2, 4));
    msg0 = _mm_sha256msg2_epu32(msg0, msg3);
    SHA256_SHANI_4ROUNDS(state0, state1, msg0, 48);

    // Rounds 52-55
    msg1 = _mm_sha256msg1_epu32(msg1, msg2);
    msg1 = _mm_add_epi32(msg1, _mm_alignr_epi8(msg0, msg3, 4));
    msg1 = _mm_sha256msg2_epu32(msg1, msg0);
    SHA256_SHANI_4ROUNDS(state0, state1, msg1, 52);

    // Rounds 56-59
    msg2 = _mm_sha256msg1_epu32(msg2, msg3);
    msg2 = _mm_add_epi32(msg2, _mm_alignr_epi8(msg1, msg0, 4));
    msg2 = _mm_sha256msg2_epu32(msg2, msg1);
    SHA256_SHANI_4ROUNDS(state0, state1, msg2, 56);

    // Rounds 60-63
    msg3 = _mm_sha256msg1_epu32(msg3, msg0);
    msg3 = _mm_add_epi32(msg3, _mm_alignr_epi8(msg2, msg1, 4));
    msg3 = _mm_sha256msg2_epu32(msg3, msg2);
    SHA256_SHANI_4ROUNDS(state0, state1, msg3, 60);

    // Add saved state
    state0 = _mm_add_epi32(state0, state0_save);
    state1 = _mm_add_epi32(state1, state1_save);

    // Rearrange back from [F,E,B,A]/[H,G,D,C] to [A,B,C,D]/[E,F,G,H]
    tmp = _mm_shuffle_epi32(state0, 0x1B);                // [A,B,E,F]
    state1 = _mm_shuffle_epi32(state1, 0xB1);             // [G,H,C,D]
    __m128i abcd = _mm_blend_epi16(tmp, state1, 0xF0);    // [A,B,C,D]
    __m128i efgh = _mm_alignr_epi8(state1, tmp, 8);       // [E,F,G,H]

    _mm_storeu_si128(reinterpret_cast<__m128i*>(state), abcd);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(state + 4), efgh);
}

#undef SHA256_SHANI_4ROUNDS
#undef SHANI_FUNC_ATTR

void sha256_33(const std::uint8_t* pubkey33, std::uint8_t* out32) noexcept {
    alignas(16) std::uint8_t block[64];
    std::memcpy(block, pubkey33, 33);
    block[33] = 0x80;
    std::memset(block + 34, 0, 22);
    block[56] = 0; block[57] = 0; block[58] = 0; block[59] = 0;
    block[60] = 0; block[61] = 0; block[62] = 0x01; block[63] = 0x08;

    std::uint32_t state[8];
    std::memcpy(state, SHA256_IV, sizeof(state));
    sha256_compress(block, state);

    for (int i = 0; i < 8; ++i) {
        store_be32(out32 + i * 4, state[i]);
    }
}

void sha256_32(const std::uint8_t* in32, std::uint8_t* out32) noexcept {
    alignas(16) std::uint8_t block[64];
    std::memcpy(block, in32, 32);
    block[32] = 0x80;
    std::memset(block + 33, 0, 23);
    block[56] = 0; block[57] = 0; block[58] = 0; block[59] = 0;
    block[60] = 0; block[61] = 0; block[62] = 0x01; block[63] = 0x00;

    std::uint32_t state[8];
    std::memcpy(state, SHA256_IV, sizeof(state));
    sha256_compress(block, state);

    for (int i = 0; i < 8; ++i) {
        store_be32(out32 + i * 4, state[i]);
    }
}

void hash160_33(const std::uint8_t* pubkey33, std::uint8_t* out20) noexcept {
    std::uint8_t sha_out[32];
    sha256_33(pubkey33, sha_out);
    // RIPEMD-160 has no hardware acceleration, use scalar
    scalar::ripemd160_32(sha_out, out20);
}

} // namespace shani

#endif // _MSC_VER || __SHA__
#endif // SECP256K1_X86_TARGET

// ============================================================================
// Public API — auto-dispatch to best available tier
// ============================================================================

// Cached tier detection (initialized on first call)
static HashTier cached_tier() noexcept {
    static const HashTier tier = detect_hash_tier();
    return tier;
}

std::array<std::uint8_t, 32> sha256(const void* data, std::size_t len) noexcept {
    // For arbitrary-length input, use the existing SHA256 class
    // (this is not the hot path — hot path uses sha256_33)
    ::secp256k1::SHA256 ctx;
    ctx.update(data, len);
    return ctx.finalize();
}

void sha256_33(const std::uint8_t* pubkey33, std::uint8_t* out32) noexcept {
#ifdef SECP256K1_X86_TARGET
#if defined(_MSC_VER) || defined(__SHA__)
    if (sha_ni_available()) {
        shani::sha256_33(pubkey33, out32);
        return;
    }
#endif
#endif
    scalar::sha256_33(pubkey33, out32);
}

void sha256_32(const std::uint8_t* in32, std::uint8_t* out32) noexcept {
#ifdef SECP256K1_X86_TARGET
#if defined(_MSC_VER) || defined(__SHA__)
    if (sha_ni_available()) {
        shani::sha256_32(in32, out32);
        return;
    }
#endif
#endif
    scalar::sha256_32(in32, out32);
}

std::array<std::uint8_t, 32> sha256d(const void* data, std::size_t len) noexcept {
    auto h1 = sha256(data, len);
    std::array<std::uint8_t, 32> out;
    sha256_32(h1.data(), out.data());
    return out;
}

std::array<std::uint8_t, 20> ripemd160(const void* data, std::size_t len) noexcept {
    // For arbitrary-length, build padded block(s)
    // Simple: use SHA256 class pattern for RIPEMD160
    // This is NOT the hot path — hot path uses ripemd160_32
    if (len <= 55) {
        // Single block
        alignas(16) std::uint8_t block[64];
        std::memcpy(block, data, len);
        block[len] = 0x80;
        std::memset(block + len + 1, 0, 55 - len);
        std::uint64_t bits = len * 8;
        for (int i = 0; i < 8; ++i)
            block[56 + i] = std::uint8_t(bits >> (i * 8));

        std::uint32_t state[5];
        std::memcpy(state, RIPEMD160_IV, sizeof(state));
        scalar::ripemd160_compress(block, state);

        std::array<std::uint8_t, 20> out;
        for (int i = 0; i < 5; ++i) store_le32(out.data() + i * 4, state[i]);
        return out;
    }

    // Multi-block: handle 56..inf byte inputs
    std::uint32_t state[5];
    std::memcpy(state, RIPEMD160_IV, sizeof(state));

    auto ptr = static_cast<const std::uint8_t*>(data);
    std::size_t remaining = len;

    while (remaining >= 64) {
        scalar::ripemd160_compress(ptr, state);
        ptr += 64;
        remaining -= 64;
    }

    // Final block(s)
    alignas(16) std::uint8_t block[128]; // up to 2 blocks
    std::memcpy(block, ptr, remaining);
    block[remaining] = 0x80;

    std::size_t pad_len = (remaining < 56) ? 64 : 128;
    std::memset(block + remaining + 1, 0, pad_len - remaining - 1 - 8);

    std::uint64_t bits = len * 8;
    for (int i = 0; i < 8; ++i)
        block[pad_len - 8 + i] = std::uint8_t(bits >> (i * 8));

    scalar::ripemd160_compress(block, state);
    if (pad_len == 128) {
        scalar::ripemd160_compress(block + 64, state);
    }

    std::array<std::uint8_t, 20> out;
    for (int i = 0; i < 5; ++i) store_le32(out.data() + i * 4, state[i]);
    return out;
}

void ripemd160_32(const std::uint8_t* in32, std::uint8_t* out20) noexcept {
    scalar::ripemd160_32(in32, out20);
}

std::array<std::uint8_t, 20> hash160(const void* data, std::size_t len) noexcept {
    auto sha_out = sha256(data, len);
    std::array<std::uint8_t, 20> out;
    ripemd160_32(sha_out.data(), out.data());
    return out;
}

void hash160_33(const std::uint8_t* pubkey33, std::uint8_t* out20) noexcept {
#ifdef SECP256K1_X86_TARGET
#if defined(_MSC_VER) || defined(__SHA__)
    if (sha_ni_available()) {
        shani::hash160_33(pubkey33, out20);
        return;
    }
#endif
#endif
    scalar::hash160_33(pubkey33, out20);
}

// ============================================================================
// Batch operations
// ============================================================================

void sha256_33_batch(
    const std::uint8_t* pubkeys,
    std::uint8_t* out32s,
    std::size_t count) noexcept
{
    // Sequential dispatch per element (SHA-NI or scalar)
    // Future: AVX2 4-way multi-buffer implementation
    for (std::size_t i = 0; i < count; ++i) {
        sha256_33(pubkeys + i * 33, out32s + i * 32);
    }
}

void ripemd160_32_batch(
    const std::uint8_t* in32s,
    std::uint8_t* out20s,
    std::size_t count) noexcept
{
    for (std::size_t i = 0; i < count; ++i) {
        ripemd160_32(in32s + i * 32, out20s + i * 20);
    }
}

void hash160_33_batch(
    const std::uint8_t* pubkeys,
    std::uint8_t* out20s,
    std::size_t count) noexcept
{
    // Fused pipeline: SHA256 → RIPEMD160 per element
    // SHA-NI handles SHA-256 in hardware
    for (std::size_t i = 0; i < count; ++i) {
        hash160_33(pubkeys + i * 33, out20s + i * 20);
    }
}

} // namespace secp256k1::hash
