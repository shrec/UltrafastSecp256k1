// ============================================================================
// ChaCha20-Poly1305 AEAD (RFC 8439) — Optimized
// ============================================================================
// Features:
//   - SSE2/SSSE3 vectorized ChaCha20 quarter-round (x86-64)
//   - 64-bit Poly1305 with __int128 multiply, 3×44-bit limbs (x86-64/aarch64)
//   - 32-bit scalar fallback for embedded/MSVC targets
//   - Constant-time tag comparison via timing-safe equality check
//   - Key material is securely erased after use
// ============================================================================

#include "secp256k1/chacha20_poly1305.hpp"
#include "secp256k1/detail/secure_erase.hpp"
#include <cstring>

#if defined(__x86_64__) || defined(_M_X64)
#include <emmintrin.h>
#ifdef __SSSE3__
#include <tmmintrin.h>
#endif
#endif

namespace secp256k1 {

// ============================================================================
// ChaCha20 core (RFC 8439 Section 2.3)
// ============================================================================

namespace {

inline std::uint32_t load32_le(const std::uint8_t* p) noexcept {
    return static_cast<std::uint32_t>(p[0])
         | (static_cast<std::uint32_t>(p[1]) << 8)
         | (static_cast<std::uint32_t>(p[2]) << 16)
         | (static_cast<std::uint32_t>(p[3]) << 24);
}

#if !defined(__SSE2__) || defined(SECP256K1_NO_INT128)
inline std::uint32_t rotl32(std::uint32_t v, int n) noexcept {
    return (v << n) | (v >> (32 - n));
}

inline void store32_le(std::uint8_t* p, std::uint32_t v) noexcept {
    p[0] = static_cast<std::uint8_t>(v);
    p[1] = static_cast<std::uint8_t>(v >> 8);
    p[2] = static_cast<std::uint8_t>(v >> 16);
    p[3] = static_cast<std::uint8_t>(v >> 24);
}
#endif

inline std::uint64_t load64_le(const std::uint8_t* p) noexcept {
    return static_cast<std::uint64_t>(p[0])
         | (static_cast<std::uint64_t>(p[1]) << 8)
         | (static_cast<std::uint64_t>(p[2]) << 16)
         | (static_cast<std::uint64_t>(p[3]) << 24)
         | (static_cast<std::uint64_t>(p[4]) << 32)
         | (static_cast<std::uint64_t>(p[5]) << 40)
         | (static_cast<std::uint64_t>(p[6]) << 48)
         | (static_cast<std::uint64_t>(p[7]) << 56);
}

inline void store64_le(std::uint8_t* p, std::uint64_t v) noexcept {
    for (int i = 0; i < 8; ++i) {
        p[i] = static_cast<std::uint8_t>(v >> (i * 8));
    }
}

// ---- SSE2/SSSE3 vectorized ChaCha20 block (x86-64) ----
// Processes all 4 columns simultaneously in 128-bit XMM registers.
// Diagonal rounds use _mm_shuffle_epi32 to rotate row elements.
// SSSE3 gives _mm_shuffle_epi8 for 16-bit and 8-bit rotations.

#if defined(__SSE2__)

void chacha20_block_internal(const std::uint32_t input[16],
                              std::uint8_t output[64]) noexcept {
    __m128i a = _mm_loadu_si128(reinterpret_cast<const __m128i*>(input));
    __m128i b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(input + 4));
    __m128i c = _mm_loadu_si128(reinterpret_cast<const __m128i*>(input + 8));
    __m128i d = _mm_loadu_si128(reinterpret_cast<const __m128i*>(input + 12));
    const __m128i a0 = a, b0 = b, c0 = c, d0 = d;

#ifdef __SSSE3__
    const __m128i rot16 = _mm_set_epi8(13,12,15,14, 9,8,11,10, 5,4,7,6, 1,0,3,2);
    const __m128i rot8  = _mm_set_epi8(14,13,12,15, 10,9,8,11, 6,5,4,7, 2,1,0,3);
#endif

    for (int i = 0; i < 10; ++i) {
        // Column round
        a = _mm_add_epi32(a, b); d = _mm_xor_si128(d, a);
#ifdef __SSSE3__
        d = _mm_shuffle_epi8(d, rot16);
#else
        d = _mm_or_si128(_mm_slli_epi32(d, 16), _mm_srli_epi32(d, 16));
#endif
        c = _mm_add_epi32(c, d); b = _mm_xor_si128(b, c);
        b = _mm_or_si128(_mm_slli_epi32(b, 12), _mm_srli_epi32(b, 20));
        a = _mm_add_epi32(a, b); d = _mm_xor_si128(d, a);
#ifdef __SSSE3__
        d = _mm_shuffle_epi8(d, rot8);
#else
        d = _mm_or_si128(_mm_slli_epi32(d, 8), _mm_srli_epi32(d, 24));
#endif
        c = _mm_add_epi32(c, d); b = _mm_xor_si128(b, c);
        b = _mm_or_si128(_mm_slli_epi32(b, 7), _mm_srli_epi32(b, 25));

        // Diagonal round: shuffle rows to align diagonals
        b = _mm_shuffle_epi32(b, _MM_SHUFFLE(0, 3, 2, 1));
        c = _mm_shuffle_epi32(c, _MM_SHUFFLE(1, 0, 3, 2));
        d = _mm_shuffle_epi32(d, _MM_SHUFFLE(2, 1, 0, 3));

        a = _mm_add_epi32(a, b); d = _mm_xor_si128(d, a);
#ifdef __SSSE3__
        d = _mm_shuffle_epi8(d, rot16);
#else
        d = _mm_or_si128(_mm_slli_epi32(d, 16), _mm_srli_epi32(d, 16));
#endif
        c = _mm_add_epi32(c, d); b = _mm_xor_si128(b, c);
        b = _mm_or_si128(_mm_slli_epi32(b, 12), _mm_srli_epi32(b, 20));
        a = _mm_add_epi32(a, b); d = _mm_xor_si128(d, a);
#ifdef __SSSE3__
        d = _mm_shuffle_epi8(d, rot8);
#else
        d = _mm_or_si128(_mm_slli_epi32(d, 8), _mm_srli_epi32(d, 24));
#endif
        c = _mm_add_epi32(c, d); b = _mm_xor_si128(b, c);
        b = _mm_or_si128(_mm_slli_epi32(b, 7), _mm_srli_epi32(b, 25));

        // Undo diagonal rotation
        b = _mm_shuffle_epi32(b, _MM_SHUFFLE(2, 1, 0, 3));
        c = _mm_shuffle_epi32(c, _MM_SHUFFLE(1, 0, 3, 2));
        d = _mm_shuffle_epi32(d, _MM_SHUFFLE(0, 3, 2, 1));
    }

    _mm_storeu_si128(reinterpret_cast<__m128i*>(output),      _mm_add_epi32(a, a0));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(output + 16), _mm_add_epi32(b, b0));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(output + 32), _mm_add_epi32(c, c0));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(output + 48), _mm_add_epi32(d, d0));
}

#else // Scalar fallback

inline void quarter_round(std::uint32_t& a, std::uint32_t& b,
                           std::uint32_t& c, std::uint32_t& d) noexcept {
    a += b; d ^= a; d = rotl32(d, 16);
    c += d; b ^= c; b = rotl32(b, 12);
    a += b; d ^= a; d = rotl32(d, 8);
    c += d; b ^= c; b = rotl32(b, 7);
}

void chacha20_block_internal(const std::uint32_t input[16],
                              std::uint8_t output[64]) noexcept {
    std::uint32_t x[16];
    std::memcpy(x, input, 64);

    for (int i = 0; i < 10; ++i) {
        quarter_round(x[0], x[4], x[ 8], x[12]);
        quarter_round(x[1], x[5], x[ 9], x[13]);
        quarter_round(x[2], x[6], x[10], x[14]);
        quarter_round(x[3], x[7], x[11], x[15]);
        quarter_round(x[0], x[5], x[10], x[15]);
        quarter_round(x[1], x[6], x[11], x[12]);
        quarter_round(x[2], x[7], x[ 8], x[13]);
        quarter_round(x[3], x[4], x[ 9], x[14]);
    }

    for (int i = 0; i < 16; ++i) {
        store32_le(output + i * 4, x[i] + input[i]);
    }
}

#endif // __SSE2__

void chacha20_setup_state(std::uint32_t state[16],
                           const std::uint8_t key[32],
                           const std::uint8_t nonce[12],
                           std::uint32_t counter) noexcept {
    // "expand 32-byte k"
    state[ 0] = 0x61707865u;
    state[ 1] = 0x3320646eu;
    state[ 2] = 0x79622d32u;
    state[ 3] = 0x6b206574u;

    // Key (8 words)
    for (int i = 0; i < 8; ++i) {
        state[4 + i] = load32_le(key + static_cast<std::size_t>(i) * 4U);
    }

    // Counter
    state[12] = counter;

    // Nonce (3 words)
    state[13] = load32_le(nonce);
    state[14] = load32_le(nonce + 4);
    state[15] = load32_le(nonce + 8);
}

} // anonymous namespace

void chacha20_block(const std::uint8_t key[32],
                    const std::uint8_t nonce[12],
                    std::uint32_t counter,
                    std::uint8_t out[64]) noexcept {
    std::uint32_t state[16];
    chacha20_setup_state(state, key, nonce, counter);
    chacha20_block_internal(state, out);
    detail::secure_erase(state, sizeof(state));
}

void chacha20_crypt(const std::uint8_t key[32],
                    const std::uint8_t nonce[12],
                    std::uint32_t counter,
                    std::uint8_t* data, std::size_t len) noexcept {
    std::uint32_t state[16];
    chacha20_setup_state(state, key, nonce, counter);

    std::uint8_t block[64];
    std::size_t offset = 0;

    while (offset < len) {
        chacha20_block_internal(state, block);
        state[12]++; // increment counter

        std::size_t const use = (len - offset < 64) ? (len - offset) : 64;
        for (std::size_t i = 0; i < use; ++i) {
            data[offset + i] ^= block[i];
        }
        offset += use;
    }

    detail::secure_erase(state, sizeof(state));
    detail::secure_erase(block, sizeof(block));
}

// ============================================================================
// Poly1305 (RFC 8439 Section 2.5)
// ============================================================================
// 64-bit path: 3×44-bit limbs with unsigned __int128 multiply (9 muls vs 25).
// 32-bit fallback: original 5×26-bit limbs with uint64_t multiply.
// ============================================================================

namespace {

#if !defined(SECP256K1_NO_INT128)

struct Poly1305State {
    std::uint64_t r[3];    // clamped key r in 44/44/42-bit limbs
    std::uint64_t sr[2];   // precomputed 20*r[1], 20*r[2] for modular reduction
    std::uint64_t pad[2];  // one-time pad s (key[16..31]) as two uint64_t
    std::uint64_t h[3];    // accumulator in 44/44/42-bit limbs

    void init(const std::uint8_t key[32]) noexcept {
        std::uint64_t t0 = load64_le(key);
        std::uint64_t t1 = load64_le(key + 8);

        // RFC 8439 clamping: clear top 4 bits of bytes 3,7,11,15;
        // clear bottom 2 bits of bytes 4,8,12
        t0 &= 0x0FFFFFFC0FFFFFFFULL;
        t1 &= 0x0FFFFFFC0FFFFFFCULL;

        // Decompose into 44/44/42-bit limbs
        r[0] =  t0                        & 0xFFFFFFFFFFFULL;
        r[1] = ((t0 >> 44) | (t1 << 20)) & 0xFFFFFFFFFFFULL;
        r[2] =  (t1 >> 24)               & 0x3FFFFFFFFFFULL;

        // 2^130 ≡ 5 mod p, so 2^132 ≡ 20 mod p
        sr[0] = r[1] * 20;
        sr[1] = r[2] * 20;

        pad[0] = load64_le(key + 16);
        pad[1] = load64_le(key + 24);

        h[0] = h[1] = h[2] = 0;
    }

    void block(const std::uint8_t* msg, std::size_t len) noexcept {
        if (len > 16) len = 16; // defensive clamp
        std::uint8_t buf[17]{};
        std::memcpy(buf, msg, len);
        buf[len] = 1;

        std::uint64_t t0 = load64_le(buf);
        std::uint64_t t1 = load64_le(buf + 8);
        std::uint64_t hibit = static_cast<std::uint64_t>(buf[16]);

        h[0] +=  t0                        & 0xFFFFFFFFFFFULL;
        h[1] += ((t0 >> 44) | (t1 << 20)) & 0xFFFFFFFFFFFULL;
        h[2] += ((t1 >> 24))              | (hibit << 40);

        // h *= r mod (2^130 - 5) using __int128 multiply-accumulate
        using u128 = unsigned __int128;
        u128 d0 = (u128)h[0] * r[0] + (u128)h[1] * sr[1] + (u128)h[2] * sr[0];
        u128 d1 = (u128)h[0] * r[1] + (u128)h[1] * r[0]  + (u128)h[2] * sr[1];
        u128 d2 = (u128)h[0] * r[2] + (u128)h[1] * r[1]  + (u128)h[2] * r[0];

        // Carry propagation
        std::uint64_t c;
        c = (std::uint64_t)(d0 >> 44); h[0] = (std::uint64_t)d0 & 0xFFFFFFFFFFFULL;
        d1 += c;
        c = (std::uint64_t)(d1 >> 44); h[1] = (std::uint64_t)d1 & 0xFFFFFFFFFFFULL;
        d2 += c;
        c = (std::uint64_t)(d2 >> 42); h[2] = (std::uint64_t)d2 & 0x3FFFFFFFFFFULL;
        h[0] += c * 5;
        c = h[0] >> 44; h[0] &= 0xFFFFFFFFFFFULL;
        h[1] += c;
    }

    void finish(std::uint8_t tag[16]) noexcept {
        // Final carry propagation
        std::uint64_t c;
        c = h[1] >> 44; h[1] &= 0xFFFFFFFFFFFULL;
        h[2] += c;
        c = h[2] >> 42; h[2] &= 0x3FFFFFFFFFFULL;
        h[0] += c * 5;
        c = h[0] >> 44; h[0] &= 0xFFFFFFFFFFFULL;
        h[1] += c;

        // Conditional subtract p: compute g = h + 5 - 2^130
        std::uint64_t g[3];
        c = h[0] + 5;        g[0] = c & 0xFFFFFFFFFFFULL; c >>= 44;
        c += h[1];            g[1] = c & 0xFFFFFFFFFFFULL; c >>= 44;
        c += h[2];            g[2] = c & 0x3FFFFFFFFFFULL; c >>= 42;

        // c == 1 iff h >= p → use g; else use h
        std::uint64_t mask = ~(c - 1);
        h[0] = (h[0] & ~mask) | (g[0] & mask);
        h[1] = (h[1] & ~mask) | (g[1] & mask);
        h[2] = (h[2] & ~mask) | (g[2] & mask);

        // Convert 44/44/42-bit limbs to 128-bit and add pad
        using u128 = unsigned __int128;
        u128 h_full = (u128)h[0] | ((u128)h[1] << 44) | ((u128)h[2] << 88);
        u128 s_full = (u128)pad[0] | ((u128)pad[1] << 64);
        u128 result = h_full + s_full;

        store64_le(tag, (std::uint64_t)result);
        store64_le(tag + 8, (std::uint64_t)(result >> 64));
    }
};

#else // 32-bit fallback — 5×26-bit limbs

struct Poly1305State {
    std::uint32_t r[5];    // clamped key r in 26-bit limbs
    std::uint32_t s[4];    // key s (pad)
    std::uint32_t h[5];    // accumulator in 26-bit limbs

    void init(const std::uint8_t key[32]) noexcept {
        std::uint32_t t0 = load32_le(key +  0) & 0x0FFFFFFFU;
        std::uint32_t t1 = load32_le(key +  4) & 0x0FFFFFFCU;
        std::uint32_t t2 = load32_le(key +  8) & 0x0FFFFFFCU;
        std::uint32_t t3 = load32_le(key + 12) & 0x0FFFFFFCU;

        r[0] =  t0                        & 0x3FFFFFF;
        r[1] = ((t0 >> 26) | (t1 <<  6)) & 0x3FFFFFF;
        r[2] = ((t1 >> 20) | (t2 << 12)) & 0x3FFFFFF;
        r[3] = ((t2 >> 14) | (t3 << 18)) & 0x3FFFFFF;
        r[4] =  (t3 >>  8);

        s[0] = load32_le(key + 16);
        s[1] = load32_le(key + 20);
        s[2] = load32_le(key + 24);
        s[3] = load32_le(key + 28);

        h[0] = h[1] = h[2] = h[3] = h[4] = 0;
    }

    void block(const std::uint8_t* msg, std::size_t len) noexcept {
        if (len > 16) len = 16; // defensive clamp
        std::uint8_t buf[17]{};
        std::memcpy(buf, msg, len);
        buf[len] = 1;

        std::uint32_t t0 = load32_le(buf);
        std::uint32_t t1 = load32_le(buf + 4);
        std::uint32_t t2 = load32_le(buf + 8);
        std::uint32_t t3 = load32_le(buf + 12);
        std::uint32_t hibit = static_cast<std::uint32_t>(buf[16]);

        h[0] += t0 & 0x3FFFFFF;
        h[1] += ((t0 >> 26) | (t1 << 6)) & 0x3FFFFFF;
        h[2] += ((t1 >> 20) | (t2 << 12)) & 0x3FFFFFF;
        h[3] += ((t2 >> 14) | (t3 << 18)) & 0x3FFFFFF;
        h[4] += (t3 >> 8) | (hibit << 24);

        std::uint32_t r0 = r[0], r1 = r[1], r2 = r[2], r3 = r[3], r4 = r[4];
        std::uint32_t s1 = r1 * 5, s2 = r2 * 5, s3 = r3 * 5, s4 = r4 * 5;

        std::uint64_t d0 = static_cast<std::uint64_t>(h[0]) * r0
                         + static_cast<std::uint64_t>(h[1]) * s4
                         + static_cast<std::uint64_t>(h[2]) * s3
                         + static_cast<std::uint64_t>(h[3]) * s2
                         + static_cast<std::uint64_t>(h[4]) * s1;

        std::uint64_t d1 = static_cast<std::uint64_t>(h[0]) * r1
                         + static_cast<std::uint64_t>(h[1]) * r0
                         + static_cast<std::uint64_t>(h[2]) * s4
                         + static_cast<std::uint64_t>(h[3]) * s3
                         + static_cast<std::uint64_t>(h[4]) * s2;

        std::uint64_t d2 = static_cast<std::uint64_t>(h[0]) * r2
                         + static_cast<std::uint64_t>(h[1]) * r1
                         + static_cast<std::uint64_t>(h[2]) * r0
                         + static_cast<std::uint64_t>(h[3]) * s4
                         + static_cast<std::uint64_t>(h[4]) * s3;

        std::uint64_t d3 = static_cast<std::uint64_t>(h[0]) * r3
                         + static_cast<std::uint64_t>(h[1]) * r2
                         + static_cast<std::uint64_t>(h[2]) * r1
                         + static_cast<std::uint64_t>(h[3]) * r0
                         + static_cast<std::uint64_t>(h[4]) * s4;

        std::uint64_t d4 = static_cast<std::uint64_t>(h[0]) * r4
                         + static_cast<std::uint64_t>(h[1]) * r3
                         + static_cast<std::uint64_t>(h[2]) * r2
                         + static_cast<std::uint64_t>(h[3]) * r1
                         + static_cast<std::uint64_t>(h[4]) * r0;

        std::uint32_t c;
        c = static_cast<std::uint32_t>(d0 >> 26); h[0] = static_cast<std::uint32_t>(d0) & 0x3FFFFFF;
        d1 += c; c = static_cast<std::uint32_t>(d1 >> 26); h[1] = static_cast<std::uint32_t>(d1) & 0x3FFFFFF;
        d2 += c; c = static_cast<std::uint32_t>(d2 >> 26); h[2] = static_cast<std::uint32_t>(d2) & 0x3FFFFFF;
        d3 += c; c = static_cast<std::uint32_t>(d3 >> 26); h[3] = static_cast<std::uint32_t>(d3) & 0x3FFFFFF;
        d4 += c; c = static_cast<std::uint32_t>(d4 >> 26); h[4] = static_cast<std::uint32_t>(d4) & 0x3FFFFFF;
        h[0] += c * 5; c = h[0] >> 26; h[0] &= 0x3FFFFFF;
        h[1] += c;
    }

    void finish(std::uint8_t tag[16]) noexcept {
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
        for (int i = 0; i < 5; ++i) {
            h[i] = (h[i] & ~mask) | (g[i] & mask);
        }

        std::uint64_t f;
        f  = static_cast<std::uint64_t>(h[0])       | (static_cast<std::uint64_t>(h[1]) << 26);
        std::uint32_t h0 = static_cast<std::uint32_t>(f);
        f  = (f >> 32) | (static_cast<std::uint64_t>(h[2]) << 20);
        std::uint32_t h1 = static_cast<std::uint32_t>(f);
        f  = (f >> 32) | (static_cast<std::uint64_t>(h[3]) << 14);
        std::uint32_t h2 = static_cast<std::uint32_t>(f);
        f  = (f >> 32) | (static_cast<std::uint64_t>(h[4]) <<  8);
        std::uint32_t h3 = static_cast<std::uint32_t>(f);

        std::uint64_t t;
        t = static_cast<std::uint64_t>(h0) + s[0];              h0 = static_cast<std::uint32_t>(t);
        t = static_cast<std::uint64_t>(h1) + s[1] + (t >> 32);  h1 = static_cast<std::uint32_t>(t);
        t = static_cast<std::uint64_t>(h2) + s[2] + (t >> 32);  h2 = static_cast<std::uint32_t>(t);
        t = static_cast<std::uint64_t>(h3) + s[3] + (t >> 32);  h3 = static_cast<std::uint32_t>(t);

        store32_le(tag +  0, h0);
        store32_le(tag +  4, h1);
        store32_le(tag +  8, h2);
        store32_le(tag + 12, h3);
    }
};

#endif // SECP256K1_NO_INT128

// Constant-time tag comparison
bool poly1305_verify(const std::uint8_t a[16], const std::uint8_t b[16]) noexcept {
    std::uint8_t diff = 0;
    for (int i = 0; i < 16; ++i) {
        diff |= a[i] ^ b[i];
    }
    return diff == 0;
}

} // anonymous namespace

std::array<std::uint8_t, 16> poly1305_mac(
    const std::uint8_t key[32],
    const std::uint8_t* data, std::size_t len) noexcept {

    Poly1305State st;
    st.init(key);

    std::size_t offset = 0;
    while (offset + 16 <= len) {
        st.block(data + offset, 16);
        offset += 16;
    }
    if (offset < len) {
        st.block(data + offset, len - offset);
    }

    std::array<std::uint8_t, 16> tag{};
    st.finish(tag.data());
    detail::secure_erase(&st, sizeof(st));
    return tag;
}

// ============================================================================
// ChaCha20-Poly1305 AEAD (RFC 8439 Section 2.8)
// ============================================================================
// Construction:
//   1. Generate Poly1305 one-time key from ChaCha20 block 0
//   2. Encrypt plaintext with ChaCha20 starting at counter 1
//   3. Construct Poly1305 MAC over (AAD || padding || ciphertext || padding || lengths)
// ============================================================================

namespace {

void aead_poly1305_pad_and_mac(Poly1305State& st,
                                const std::uint8_t* aad, std::size_t aad_len,
                                const std::uint8_t* ct, std::size_t ct_len) noexcept {
    // Process AAD in full 16-byte blocks
    std::size_t off = 0;
    while (off + 16 <= aad_len) {
        st.block(aad + off, 16);
        off += 16;
    }
    // Pad last partial AAD block to 16 bytes (RFC 8439 pad16).
    // The padding zeros are part of the mac_data stream and must be
    // processed in the same Poly1305 block as the trailing AAD bytes,
    // NOT as a separate block (a separate block would get its own
    // hibit marker and produce a wrong MAC).
    if (off < aad_len) {
        std::uint8_t padded[16]{};
        std::memcpy(padded, aad + off, aad_len - off);
        st.block(padded, 16);
    }

    // Process ciphertext in full 16-byte blocks
    off = 0;
    while (off + 16 <= ct_len) {
        st.block(ct + off, 16);
        off += 16;
    }
    // Pad last partial ciphertext block to 16 bytes (RFC 8439 pad16)
    if (off < ct_len) {
        std::uint8_t padded[16]{};
        std::memcpy(padded, ct + off, ct_len - off);
        st.block(padded, 16);
    }

    // Append lengths as two 64-bit little-endian values
    std::uint8_t lens[16];
    store64_le(lens, static_cast<std::uint64_t>(aad_len));
    store64_le(lens + 8, static_cast<std::uint64_t>(ct_len));
    st.block(lens, 16);
}

} // anonymous namespace

void aead_chacha20_poly1305_encrypt(
    const std::uint8_t key[32],
    const std::uint8_t nonce[12],
    const std::uint8_t* aad, std::size_t aad_len,
    const std::uint8_t* plaintext, std::size_t plaintext_len,
    std::uint8_t* out,
    std::uint8_t tag[16]) noexcept {

    // 1. Generate Poly1305 one-time key from ChaCha20 block 0
    std::uint8_t poly_key[64];
    chacha20_block(key, nonce, 0, poly_key);

    // 2. Encrypt plaintext (counter starts at 1)
    if (plaintext_len > 0) {
        if (out != plaintext) std::memcpy(out, plaintext, plaintext_len);
        chacha20_crypt(key, nonce, 1, out, plaintext_len);
    }

    // 3. Compute Poly1305 tag over (AAD || pad || ciphertext || pad || lengths)
    Poly1305State st;
    st.init(poly_key);
    aead_poly1305_pad_and_mac(st, aad, aad_len, out, plaintext_len);
    st.finish(tag);

    detail::secure_erase(poly_key, sizeof(poly_key));
    detail::secure_erase(&st, sizeof(st));
}

bool aead_chacha20_poly1305_decrypt(
    const std::uint8_t key[32],
    const std::uint8_t nonce[12],
    const std::uint8_t* aad, std::size_t aad_len,
    const std::uint8_t* ciphertext, std::size_t ciphertext_len,
    const std::uint8_t tag[16],
    std::uint8_t* out) noexcept {

    // 1. Generate Poly1305 one-time key from ChaCha20 block 0
    std::uint8_t poly_key[64];
    chacha20_block(key, nonce, 0, poly_key);

    // 2. Verify tag first (before decrypting)
    Poly1305State st;
    st.init(poly_key);
    aead_poly1305_pad_and_mac(st, aad, aad_len, ciphertext, ciphertext_len);
    std::uint8_t computed_tag[16];
    st.finish(computed_tag);

    detail::secure_erase(poly_key, sizeof(poly_key));
    detail::secure_erase(&st, sizeof(st));

    if (!poly1305_verify(computed_tag, tag)) {
        detail::secure_erase(computed_tag, sizeof(computed_tag));
        if (ciphertext_len > 0) {
            std::memset(out, 0, ciphertext_len);
        }
        return false;
    }
    detail::secure_erase(computed_tag, sizeof(computed_tag));

    // 3. Decrypt (counter starts at 1)
    if (ciphertext_len > 0) {
        if (out != ciphertext) std::memcpy(out, ciphertext, ciphertext_len);
        chacha20_crypt(key, nonce, 1, out, ciphertext_len);
    }

    return true;
}

} // namespace secp256k1
