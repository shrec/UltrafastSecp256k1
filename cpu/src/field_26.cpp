// ============================================================================
// 10×26-bit Field Element — Implementation
// ============================================================================
// Lazy-reduction field arithmetic for secp256k1 on 32-bit platforms.
//
// Multiplication and squaring kernels adapted from bitcoin-core/secp256k1
// field_10x26_impl.h (MIT license).
//
// Key property: p = 2^256 - 0x1000003D1
//   → 2^256 ≡ 0x1000003D1 (mod p)
//   → Each upper half-column reduces with K = 0x3D1 and carry adjustments
//
// Unlike the 5×52 path, this ONLY needs uint64_t for intermediates
// (32×32→64 products), NOT uint128_t. Available on ALL platforms.
// ============================================================================

#include "secp256k1/field_26.hpp"
#include <cstring>

namespace secp256k1::fast {

using namespace fe26_constants;

// ═══════════════════════════════════════════════════════════════════════════
// Construction
// ═══════════════════════════════════════════════════════════════════════════

FieldElement26 FieldElement26::zero() noexcept {
    return FieldElement26{{0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};
}

FieldElement26 FieldElement26::one() noexcept {
    return FieldElement26{{1, 0, 0, 0, 0, 0, 0, 0, 0, 0}};
}

// ═══════════════════════════════════════════════════════════════════════════
// Conversion: 4×64 ↔ 10×26
// ═══════════════════════════════════════════════════════════════════════════
//
// 4×64 layout (256 bits total):
//   L[0] = bits[  0.. 63]   L[1] = bits[ 64..127]
//   L[2] = bits[128..191]   L[3] = bits[192..255]
//
// 10×26 layout (260 bit capacity, 256 used):
//   n[0] = bits[  0.. 25]   (26 bits from L[0])
//   n[1] = bits[ 26.. 51]   (26 bits)
//   n[2] = bits[ 52.. 77]   (12 bits from L[0] + 14 bits from L[1])
//   n[3] = bits[ 78..103]   (26 bits)
//   n[4] = bits[104..129]   (24 bits from L[1] + 2 bits from L[2])
//   n[5] = bits[130..155]   (26 bits)
//   n[6] = bits[156..181]   (26 bits)
//   n[7] = bits[182..207]   (10 bits from L[2] + 16 bits from L[3])
//   n[8] = bits[208..233]   (26 bits)
//   n[9] = bits[234..255]   (22 bits from L[3])

FieldElement26 FieldElement26::from_fe(const FieldElement& fe) noexcept {
    const auto& L = fe.limbs();
    FieldElement26 r;
    r.n[0] = (std::uint32_t)(L[0])         & M26;
    r.n[1] = (std::uint32_t)(L[0] >> 26)   & M26;
    r.n[2] = (std::uint32_t)((L[0] >> 52) | (L[1] << 12)) & M26;
    r.n[3] = (std::uint32_t)(L[1] >> 14)   & M26;
    r.n[4] = (std::uint32_t)((L[1] >> 40) | (L[2] << 24)) & M26;
    r.n[5] = (std::uint32_t)(L[2] >> 2)    & M26;
    r.n[6] = (std::uint32_t)(L[2] >> 28)   & M26;
    r.n[7] = (std::uint32_t)((L[2] >> 54) | (L[3] << 10)) & M26;
    r.n[8] = (std::uint32_t)(L[3] >> 16)   & M26;
    r.n[9] = (std::uint32_t)(L[3] >> 42)   & M22;
    return r;
}

FieldElement FieldElement26::to_fe() const noexcept {
    FieldElement26 tmp = *this;
    tmp.normalize();

    std::uint64_t L0 = (std::uint64_t)tmp.n[0]
                     | ((std::uint64_t)tmp.n[1] << 26)
                     | ((std::uint64_t)tmp.n[2] << 52);
    std::uint64_t L1 = ((std::uint64_t)tmp.n[2] >> 12)
                     | ((std::uint64_t)tmp.n[3] << 14)
                     | ((std::uint64_t)tmp.n[4] << 40);
    std::uint64_t L2 = ((std::uint64_t)tmp.n[4] >> 24)
                     | ((std::uint64_t)tmp.n[5] << 2)
                     | ((std::uint64_t)tmp.n[6] << 28)
                     | ((std::uint64_t)tmp.n[7] << 54);
    std::uint64_t L3 = ((std::uint64_t)tmp.n[7] >> 10)
                     | ((std::uint64_t)tmp.n[8] << 16)
                     | ((std::uint64_t)tmp.n[9] << 42);

    return FieldElement::from_limbs({L0, L1, L2, L3});
}

// ═══════════════════════════════════════════════════════════════════════════
// Normalization
// ═══════════════════════════════════════════════════════════════════════════

void fe26_normalize_weak(std::uint32_t* r) noexcept {
    // Single-pass carry propagation
    std::uint32_t t0 = r[0], t1 = r[1], t2 = r[2], t3 = r[3], t4 = r[4];
    std::uint32_t t5 = r[5], t6 = r[6], t7 = r[7], t8 = r[8], t9 = r[9];

    // Propagate carries
    std::uint32_t x = t0 >> 26; t0 &= M26; t1 += x;
    x = t1 >> 26; t1 &= M26; t2 += x;
    x = t2 >> 26; t2 &= M26; t3 += x;
    x = t3 >> 26; t3 &= M26; t4 += x;
    x = t4 >> 26; t4 &= M26; t5 += x;
    x = t5 >> 26; t5 &= M26; t6 += x;
    x = t6 >> 26; t6 &= M26; t7 += x;
    x = t7 >> 26; t7 &= M26; t8 += x;
    x = t8 >> 26; t8 &= M26; t9 += x;

    // t9 may exceed 22 bits; that's OK for weak normalization
    // (overflow will be handled at next full normalize or next mul)
    r[0] = t0; r[1] = t1; r[2] = t2; r[3] = t3; r[4] = t4;
    r[5] = t5; r[6] = t6; r[7] = t7; r[8] = t8; r[9] = t9;
}

void fe26_normalize(std::uint32_t* r) noexcept {
    std::uint32_t t0 = r[0], t1 = r[1], t2 = r[2], t3 = r[3], t4 = r[4];
    std::uint32_t t5 = r[5], t6 = r[6], t7 = r[7], t8 = r[8], t9 = r[9];

    // First pass: propagate carries
    std::uint32_t x = t0 >> 26; t0 &= M26; t1 += x;
    x = t1 >> 26; t1 &= M26; t2 += x;
    x = t2 >> 26; t2 &= M26; t3 += x;
    x = t3 >> 26; t3 &= M26; t4 += x;
    x = t4 >> 26; t4 &= M26; t5 += x;
    x = t5 >> 26; t5 &= M26; t6 += x;
    x = t6 >> 26; t6 &= M26; t7 += x;
    x = t7 >> 26; t7 &= M26; t8 += x;
    x = t8 >> 26; t8 &= M26; t9 += x;

    // Handle overflow from top limb: 2^256 ≡ 0x1000003D1 (mod p)
    // overflow = t9 >> 22 (bits above 256)
    // Reduce: overflow * 0x1000003D1
    // But since t9 < 2^26 after weak norm, overflow < 2^4 = 16
    // 0x1000003D1 only affects limbs 0 and 1:
    //   limb 0: += overflow * 0x3D1
    //   limb 1: += overflow * 0x40  (since 0x1000003D1 >> 26 = 0x40)
    //   Actually: 0x1000003D1 = (0x40 << 26) | 0x3D1? No.
    //   0x1000003D1 = 0x1_0000_03D1
    //     bits [0..25]  = 0x3D1
    //     bits [26..51] = 0x40 ? Let me compute: 0x1000003D1 >> 26 = 0x1000003D1/67108864
    //     = 4294968273 / 67108864 = 64.00000... So bits[26..51] = 0x40
    //     Wait: 0x40 << 26 = 0x1000000000. 0x1000003D1 - 0x1000000000 = 0x3D1. Yes.
    //   So: overflow into (limb0 += overflow*0x3D1, limb1 += overflow*0x40)
    //   Wait, let me recompute. 0x1000003D1 in 26-bit limbs:
    //     lo26 = 0x1000003D1 & 0x3FFFFFF = 0x3D1
    //     hi   = 0x1000003D1 >> 26 = 64 = 0x40
    //   So we need: t0 += m * 0x3D1; t1 += m * 0x40
    //   where m = t9 >> 22
    std::uint32_t m = t9 >> 22;
    t9 &= M22;
    t0 += m * 0x3D1U;
    t1 += m * 0x40U;

    // Second pass: propagate carries again
    x = t0 >> 26; t0 &= M26; t1 += x;
    x = t1 >> 26; t1 &= M26; t2 += x;
    x = t2 >> 26; t2 &= M26; t3 += x;
    x = t3 >> 26; t3 &= M26; t4 += x;
    x = t4 >> 26; t4 &= M26; t5 += x;
    x = t5 >> 26; t5 &= M26; t6 += x;
    x = t6 >> 26; t6 &= M26; t7 += x;
    x = t7 >> 26; t7 &= M26; t8 += x;
    x = t8 >> 26; t8 &= M26; t9 += x;

    // Handle residual overflow from second pass
    m = t9 >> 22;
    t9 &= M22;
    t0 += m * 0x3D1U;
    t1 += m * 0x40U;

    // One more carry propagation (at most one carry travels through)
    x = t0 >> 26; t0 &= M26; t1 += x;
    x = t1 >> 26; t1 &= M26; t2 += x;

    // Now value is in [0, 2p). Make canonical by subtracting p if >= p.
    // Compare with p branchlessly.
    // p = {0x3FFFC2F, 0x3FFFFBF, 0x3FFFFFF, ..., 0x3FFFFFF, 0x3FFFFF}
    // Subtract p and check for borrow
    std::uint32_t z0 = t0, z1 = t1, z2 = t2, z3 = t3, z4 = t4;
    std::uint32_t z5 = t5, z6 = t6, z7 = t7, z8 = t8, z9 = t9;

    // Trial subtraction of p
    // Carry-based: compute t - p using signed differences
    std::int64_t d;
    d = (std::int64_t)z0 - P0; z0 = (std::uint32_t)d & M26; bool borrow = (d < 0);
    d = (std::int64_t)z1 - P1 - (borrow ? 1 : 0); z1 = (std::uint32_t)d & M26; borrow = (d < 0);
    d = (std::int64_t)z2 - P2 - (borrow ? 1 : 0); z2 = (std::uint32_t)d & M26; borrow = (d < 0);
    d = (std::int64_t)z3 - P3 - (borrow ? 1 : 0); z3 = (std::uint32_t)d & M26; borrow = (d < 0);
    d = (std::int64_t)z4 - P4 - (borrow ? 1 : 0); z4 = (std::uint32_t)d & M26; borrow = (d < 0);
    d = (std::int64_t)z5 - P5 - (borrow ? 1 : 0); z5 = (std::uint32_t)d & M26; borrow = (d < 0);
    d = (std::int64_t)z6 - P6 - (borrow ? 1 : 0); z6 = (std::uint32_t)d & M26; borrow = (d < 0);
    d = (std::int64_t)z7 - P7 - (borrow ? 1 : 0); z7 = (std::uint32_t)d & M26; borrow = (d < 0);
    d = (std::int64_t)z8 - P8 - (borrow ? 1 : 0); z8 = (std::uint32_t)d & M26; borrow = (d < 0);
    d = (std::int64_t)z9 - P9 - (borrow ? 1 : 0); z9 = (std::uint32_t)d & M22; borrow = (d < 0);

    // If no borrow, t >= p → use subtracted result; else keep original
    std::uint32_t mask = borrow ? 0xFFFFFFFFU : 0U;
    r[0] = (t0 & mask) | (z0 & ~mask);
    r[1] = (t1 & mask) | (z1 & ~mask);
    r[2] = (t2 & mask) | (z2 & ~mask);
    r[3] = (t3 & mask) | (z3 & ~mask);
    r[4] = (t4 & mask) | (z4 & ~mask);
    r[5] = (t5 & mask) | (z5 & ~mask);
    r[6] = (t6 & mask) | (z6 & ~mask);
    r[7] = (t7 & mask) | (z7 & ~mask);
    r[8] = (t8 & mask) | (z8 & ~mask);
    r[9] = (t9 & mask) | (z9 & ~mask);
}

void FieldElement26::normalize_weak() noexcept { fe26_normalize_weak(n); }
void FieldElement26::normalize() noexcept { fe26_normalize(n); }

// ═══════════════════════════════════════════════════════════════════════════
// Lazy Addition (10 plain adds, NO carry propagation)
// ═══════════════════════════════════════════════════════════════════════════

FieldElement26 FieldElement26::operator+(const FieldElement26& rhs) const noexcept {
    FieldElement26 r;
    r.n[0] = n[0] + rhs.n[0];
    r.n[1] = n[1] + rhs.n[1];
    r.n[2] = n[2] + rhs.n[2];
    r.n[3] = n[3] + rhs.n[3];
    r.n[4] = n[4] + rhs.n[4];
    r.n[5] = n[5] + rhs.n[5];
    r.n[6] = n[6] + rhs.n[6];
    r.n[7] = n[7] + rhs.n[7];
    r.n[8] = n[8] + rhs.n[8];
    r.n[9] = n[9] + rhs.n[9];
    return r;
}

void FieldElement26::add_assign(const FieldElement26& rhs) noexcept {
    n[0] += rhs.n[0];
    n[1] += rhs.n[1];
    n[2] += rhs.n[2];
    n[3] += rhs.n[3];
    n[4] += rhs.n[4];
    n[5] += rhs.n[5];
    n[6] += rhs.n[6];
    n[7] += rhs.n[7];
    n[8] += rhs.n[8];
    n[9] += rhs.n[9];
}

// ═══════════════════════════════════════════════════════════════════════════
// Negate: (m+1)*p - a
// ═══════════════════════════════════════════════════════════════════════════

FieldElement26 FieldElement26::negate(unsigned magnitude) const noexcept {
    FieldElement26 r = *this;
    r.negate_assign(magnitude);
    return r;
}

void FieldElement26::negate_assign(unsigned magnitude) noexcept {
    std::uint32_t m = magnitude + 1;
    std::uint32_t mp0 = m * P0;
    std::uint32_t mp1 = m * P1;
    std::uint32_t mp2 = m * P2;
    std::uint32_t mp3 = m * P3;
    std::uint32_t mp4 = m * P4;
    std::uint32_t mp5 = m * P5;
    std::uint32_t mp6 = m * P6;
    std::uint32_t mp7 = m * P7;
    std::uint32_t mp8 = m * P8;
    std::uint32_t mp9 = m * P9;
    n[0] = mp0 - n[0];
    n[1] = mp1 - n[1];
    n[2] = mp2 - n[2];
    n[3] = mp3 - n[3];
    n[4] = mp4 - n[4];
    n[5] = mp5 - n[5];
    n[6] = mp6 - n[6];
    n[7] = mp7 - n[7];
    n[8] = mp8 - n[8];
    n[9] = mp9 - n[9];
}

// ═══════════════════════════════════════════════════════════════════════════
// Multiplication (10×26 with inline secp256k1 reduction)
// ═══════════════════════════════════════════════════════════════════════════
//
// Adapted from bitcoin-core/secp256k1 field_10x26_impl.h (MIT license).
//
// Two-accumulator algorithm:
//   c — accumulates lower-half (direct) products + reduced upper-half pieces
//   d — accumulates upper-half products; 26-bit pieces extracted & reduced
//
// Key: column 9 (direct products where i+j=9) is computed FIRST into d,
// then d carries naturally through upper columns 10..18 as we process
// output columns 0..8.
//
// Overflow analysis:
//   d: max ~10 products × 2^52 ≈ 2^55.3 per column, fits uint64_t
//   c: max ~10 products × 2^52 + u_k*R0 (2^40) ≈ 2^55.3, fits uint64_t
//   u_k: extracted 26-bit value, so u_k*R0 ≤ 2^40, u_k*R1 ≤ 2^36

void fe26_mul_inner(std::uint32_t* r, const std::uint32_t* a,
                    const std::uint32_t* b) noexcept {
    std::uint64_t c, d;
    std::uint64_t u0, u1, u2, u3, u4, u5, u6, u7, u8;
    std::uint32_t t9, t0, t1, t2, t3, t4, t5, t6, t7;
    const std::uint32_t M = 0x3FFFFFFU, R0 = 0x3D10U, R1 = 0x400U;

    // ── Column 9 (direct products) into d ───────────────────────────
    d  = (std::uint64_t)a[0] * b[9]
       + (std::uint64_t)a[1] * b[8]
       + (std::uint64_t)a[2] * b[7]
       + (std::uint64_t)a[3] * b[6]
       + (std::uint64_t)a[4] * b[5]
       + (std::uint64_t)a[5] * b[4]
       + (std::uint64_t)a[6] * b[3]
       + (std::uint64_t)a[7] * b[2]
       + (std::uint64_t)a[8] * b[1]
       + (std::uint64_t)a[9] * b[0];
    t9 = (std::uint32_t)d & M; d >>= 26;

    // ── Column 0 ────────────────────────────────────────────────────
    c  = (std::uint64_t)a[0] * b[0];
    d += (std::uint64_t)a[1] * b[9]
       + (std::uint64_t)a[2] * b[8]
       + (std::uint64_t)a[3] * b[7]
       + (std::uint64_t)a[4] * b[6]
       + (std::uint64_t)a[5] * b[5]
       + (std::uint64_t)a[6] * b[4]
       + (std::uint64_t)a[7] * b[3]
       + (std::uint64_t)a[8] * b[2]
       + (std::uint64_t)a[9] * b[1];
    u0 = d & M; d >>= 26; c += u0 * R0;
    t0 = (std::uint32_t)(c & M); c >>= 26; c += u0 * R1;

    // ── Column 1 ────────────────────────────────────────────────────
    c += (std::uint64_t)a[0] * b[1]
       + (std::uint64_t)a[1] * b[0];
    d += (std::uint64_t)a[2] * b[9]
       + (std::uint64_t)a[3] * b[8]
       + (std::uint64_t)a[4] * b[7]
       + (std::uint64_t)a[5] * b[6]
       + (std::uint64_t)a[6] * b[5]
       + (std::uint64_t)a[7] * b[4]
       + (std::uint64_t)a[8] * b[3]
       + (std::uint64_t)a[9] * b[2];
    u1 = d & M; d >>= 26; c += u1 * R0;
    t1 = (std::uint32_t)(c & M); c >>= 26; c += u1 * R1;

    // ── Column 2 ────────────────────────────────────────────────────
    c += (std::uint64_t)a[0] * b[2]
       + (std::uint64_t)a[1] * b[1]
       + (std::uint64_t)a[2] * b[0];
    d += (std::uint64_t)a[3] * b[9]
       + (std::uint64_t)a[4] * b[8]
       + (std::uint64_t)a[5] * b[7]
       + (std::uint64_t)a[6] * b[6]
       + (std::uint64_t)a[7] * b[5]
       + (std::uint64_t)a[8] * b[4]
       + (std::uint64_t)a[9] * b[3];
    u2 = d & M; d >>= 26; c += u2 * R0;
    t2 = (std::uint32_t)(c & M); c >>= 26; c += u2 * R1;

    // ── Column 3 ────────────────────────────────────────────────────
    c += (std::uint64_t)a[0] * b[3]
       + (std::uint64_t)a[1] * b[2]
       + (std::uint64_t)a[2] * b[1]
       + (std::uint64_t)a[3] * b[0];
    d += (std::uint64_t)a[4] * b[9]
       + (std::uint64_t)a[5] * b[8]
       + (std::uint64_t)a[6] * b[7]
       + (std::uint64_t)a[7] * b[6]
       + (std::uint64_t)a[8] * b[5]
       + (std::uint64_t)a[9] * b[4];
    u3 = d & M; d >>= 26; c += u3 * R0;
    t3 = (std::uint32_t)(c & M); c >>= 26; c += u3 * R1;

    // ── Column 4 ────────────────────────────────────────────────────
    c += (std::uint64_t)a[0] * b[4]
       + (std::uint64_t)a[1] * b[3]
       + (std::uint64_t)a[2] * b[2]
       + (std::uint64_t)a[3] * b[1]
       + (std::uint64_t)a[4] * b[0];
    d += (std::uint64_t)a[5] * b[9]
       + (std::uint64_t)a[6] * b[8]
       + (std::uint64_t)a[7] * b[7]
       + (std::uint64_t)a[8] * b[6]
       + (std::uint64_t)a[9] * b[5];
    u4 = d & M; d >>= 26; c += u4 * R0;
    t4 = (std::uint32_t)(c & M); c >>= 26; c += u4 * R1;

    // ── Column 5 ────────────────────────────────────────────────────
    c += (std::uint64_t)a[0] * b[5]
       + (std::uint64_t)a[1] * b[4]
       + (std::uint64_t)a[2] * b[3]
       + (std::uint64_t)a[3] * b[2]
       + (std::uint64_t)a[4] * b[1]
       + (std::uint64_t)a[5] * b[0];
    d += (std::uint64_t)a[6] * b[9]
       + (std::uint64_t)a[7] * b[8]
       + (std::uint64_t)a[8] * b[7]
       + (std::uint64_t)a[9] * b[6];
    u5 = d & M; d >>= 26; c += u5 * R0;
    t5 = (std::uint32_t)(c & M); c >>= 26; c += u5 * R1;

    // ── Column 6 ────────────────────────────────────────────────────
    c += (std::uint64_t)a[0] * b[6]
       + (std::uint64_t)a[1] * b[5]
       + (std::uint64_t)a[2] * b[4]
       + (std::uint64_t)a[3] * b[3]
       + (std::uint64_t)a[4] * b[2]
       + (std::uint64_t)a[5] * b[1]
       + (std::uint64_t)a[6] * b[0];
    d += (std::uint64_t)a[7] * b[9]
       + (std::uint64_t)a[8] * b[8]
       + (std::uint64_t)a[9] * b[7];
    u6 = d & M; d >>= 26; c += u6 * R0;
    t6 = (std::uint32_t)(c & M); c >>= 26; c += u6 * R1;

    // ── Column 7 ────────────────────────────────────────────────────
    c += (std::uint64_t)a[0] * b[7]
       + (std::uint64_t)a[1] * b[6]
       + (std::uint64_t)a[2] * b[5]
       + (std::uint64_t)a[3] * b[4]
       + (std::uint64_t)a[4] * b[3]
       + (std::uint64_t)a[5] * b[2]
       + (std::uint64_t)a[6] * b[1]
       + (std::uint64_t)a[7] * b[0];
    d += (std::uint64_t)a[8] * b[9]
       + (std::uint64_t)a[9] * b[8];
    u7 = d & M; d >>= 26; c += u7 * R0;
    t7 = (std::uint32_t)(c & M); c >>= 26; c += u7 * R1;

    // ── Column 8 (last with upper-half products) ────────────────────
    c += (std::uint64_t)a[0] * b[8]
       + (std::uint64_t)a[1] * b[7]
       + (std::uint64_t)a[2] * b[6]
       + (std::uint64_t)a[3] * b[5]
       + (std::uint64_t)a[4] * b[4]
       + (std::uint64_t)a[5] * b[3]
       + (std::uint64_t)a[6] * b[2]
       + (std::uint64_t)a[7] * b[1]
       + (std::uint64_t)a[8] * b[0];
    d += (std::uint64_t)a[9] * b[9];
    u8 = d & M; d >>= 26; c += u8 * R0;

    // ── Store r[3..7], extract r[8], prepare finalize ────────────────
    r[3] = t3;
    r[4] = t4;
    r[5] = t5;
    r[6] = t6;
    r[7] = t7;
    r[8] = (std::uint32_t)(c & M); c >>= 26; c += u8 * R1;

    // ── Finalize: fold d (upper carry) + t9 into column 9 ───────────
    // d is the remaining carry from upper-half extraction (tiny, ~0..9)
    // Reduce d: d * 2^(26*19) = d * R at column 9
    c += d * R0 + t9;
    r[9] = (std::uint32_t)(c & (M >> 4)); c >>= 22; c += d * ((std::uint64_t)R1 << 4);

    // c is now the overflow above 256 bits.
    // Reduce: c * 2^256 ≡ c * 0x1000003D1 (mod p)
    // 0x1000003D1 = (R0>>4) + (R1>>4) * 2^26 = 0x3D1 + 0x40*2^26
    d = c * (R0 >> 4) + t0;
    r[0] = (std::uint32_t)(d & M); d >>= 26;
    d += c * (R1 >> 4) + t1;
    r[1] = (std::uint32_t)(d & M); d >>= 26;
    d += t2;
    r[2] = (std::uint32_t)d;
}

// ═══════════════════════════════════════════════════════════════════════════
// Squaring (10×26 with symmetry optimization)
// ═══════════════════════════════════════════════════════════════════════════
//
// Same two-accumulator algorithm as mul, but using a[i]*a[j]=a[j]*a[i]
// symmetry: 2*a[i]*a[j] for i≠j, a[i]² for i=j.

void fe26_sqr_inner(std::uint32_t* r, const std::uint32_t* a) noexcept {
    std::uint64_t c, d;
    std::uint64_t u0, u1, u2, u3, u4, u5, u6, u7, u8;
    std::uint32_t t9, t0, t1, t2, t3, t4, t5, t6, t7;
    const std::uint32_t M = 0x3FFFFFFU, R0 = 0x3D10U, R1 = 0x400U;

    // ── Column 9 (direct) ───────────────────────────────────────────
    d  = (std::uint64_t)(a[0]*2) * a[9]
       + (std::uint64_t)(a[1]*2) * a[8]
       + (std::uint64_t)(a[2]*2) * a[7]
       + (std::uint64_t)(a[3]*2) * a[6]
       + (std::uint64_t)(a[4]*2) * a[5];
    t9 = (std::uint32_t)d & M; d >>= 26;

    // ── Column 0 ────────────────────────────────────────────────────
    c  = (std::uint64_t)a[0] * a[0];
    d += (std::uint64_t)(a[1]*2) * a[9]
       + (std::uint64_t)(a[2]*2) * a[8]
       + (std::uint64_t)(a[3]*2) * a[7]
       + (std::uint64_t)(a[4]*2) * a[6]
       + (std::uint64_t)a[5] * a[5];
    u0 = d & M; d >>= 26; c += u0 * R0;
    t0 = (std::uint32_t)(c & M); c >>= 26; c += u0 * R1;

    // ── Column 1 ────────────────────────────────────────────────────
    c += (std::uint64_t)(a[0]*2) * a[1];
    d += (std::uint64_t)(a[2]*2) * a[9]
       + (std::uint64_t)(a[3]*2) * a[8]
       + (std::uint64_t)(a[4]*2) * a[7]
       + (std::uint64_t)(a[5]*2) * a[6];
    u1 = d & M; d >>= 26; c += u1 * R0;
    t1 = (std::uint32_t)(c & M); c >>= 26; c += u1 * R1;

    // ── Column 2 ────────────────────────────────────────────────────
    c += (std::uint64_t)(a[0]*2) * a[2]
       + (std::uint64_t)a[1] * a[1];
    d += (std::uint64_t)(a[3]*2) * a[9]
       + (std::uint64_t)(a[4]*2) * a[8]
       + (std::uint64_t)(a[5]*2) * a[7]
       + (std::uint64_t)a[6] * a[6];
    u2 = d & M; d >>= 26; c += u2 * R0;
    t2 = (std::uint32_t)(c & M); c >>= 26; c += u2 * R1;

    // ── Column 3 ────────────────────────────────────────────────────
    c += (std::uint64_t)(a[0]*2) * a[3]
       + (std::uint64_t)(a[1]*2) * a[2];
    d += (std::uint64_t)(a[4]*2) * a[9]
       + (std::uint64_t)(a[5]*2) * a[8]
       + (std::uint64_t)(a[6]*2) * a[7];
    u3 = d & M; d >>= 26; c += u3 * R0;
    t3 = (std::uint32_t)(c & M); c >>= 26; c += u3 * R1;

    // ── Column 4 ────────────────────────────────────────────────────
    c += (std::uint64_t)(a[0]*2) * a[4]
       + (std::uint64_t)(a[1]*2) * a[3]
       + (std::uint64_t)a[2] * a[2];
    d += (std::uint64_t)(a[5]*2) * a[9]
       + (std::uint64_t)(a[6]*2) * a[8]
       + (std::uint64_t)a[7] * a[7];
    u4 = d & M; d >>= 26; c += u4 * R0;
    t4 = (std::uint32_t)(c & M); c >>= 26; c += u4 * R1;

    // ── Column 5 ────────────────────────────────────────────────────
    c += (std::uint64_t)(a[0]*2) * a[5]
       + (std::uint64_t)(a[1]*2) * a[4]
       + (std::uint64_t)(a[2]*2) * a[3];
    d += (std::uint64_t)(a[6]*2) * a[9]
       + (std::uint64_t)(a[7]*2) * a[8];
    u5 = d & M; d >>= 26; c += u5 * R0;
    t5 = (std::uint32_t)(c & M); c >>= 26; c += u5 * R1;

    // ── Column 6 ────────────────────────────────────────────────────
    c += (std::uint64_t)(a[0]*2) * a[6]
       + (std::uint64_t)(a[1]*2) * a[5]
       + (std::uint64_t)(a[2]*2) * a[4]
       + (std::uint64_t)a[3] * a[3];
    d += (std::uint64_t)(a[7]*2) * a[9]
       + (std::uint64_t)a[8] * a[8];
    u6 = d & M; d >>= 26; c += u6 * R0;
    t6 = (std::uint32_t)(c & M); c >>= 26; c += u6 * R1;

    // ── Column 7 ────────────────────────────────────────────────────
    c += (std::uint64_t)(a[0]*2) * a[7]
       + (std::uint64_t)(a[1]*2) * a[6]
       + (std::uint64_t)(a[2]*2) * a[5]
       + (std::uint64_t)(a[3]*2) * a[4];
    d += (std::uint64_t)(a[8]*2) * a[9];
    u7 = d & M; d >>= 26; c += u7 * R0;
    t7 = (std::uint32_t)(c & M); c >>= 26; c += u7 * R1;

    // ── Column 8 (last with upper-half products) ────────────────────
    c += (std::uint64_t)(a[0]*2) * a[8]
       + (std::uint64_t)(a[1]*2) * a[7]
       + (std::uint64_t)(a[2]*2) * a[6]
       + (std::uint64_t)(a[3]*2) * a[5]
       + (std::uint64_t)a[4] * a[4];
    d += (std::uint64_t)a[9] * a[9];
    u8 = d & M; d >>= 26; c += u8 * R0;

    // ── Store r[3..7], extract r[8], prepare finalize ────────────────
    r[3] = t3;
    r[4] = t4;
    r[5] = t5;
    r[6] = t6;
    r[7] = t7;
    r[8] = (std::uint32_t)(c & M); c >>= 26; c += u8 * R1;

    // ── Finalize ────────────────────────────────────────────────────
    c += d * R0 + t9;
    r[9] = (std::uint32_t)(c & (M >> 4)); c >>= 22; c += d * ((std::uint64_t)R1 << 4);

    d = c * (R0 >> 4) + t0;
    r[0] = (std::uint32_t)(d & M); d >>= 26;
    d += c * (R1 >> 4) + t1;
    r[1] = (std::uint32_t)(d & M); d >>= 26;
    d += t2;
    r[2] = (std::uint32_t)d;
}

// ═══════════════════════════════════════════════════════════════════════════
// Method wrappers
// ═══════════════════════════════════════════════════════════════════════════

FieldElement26 FieldElement26::operator*(const FieldElement26& rhs) const noexcept {
    FieldElement26 result;
    fe26_mul_inner(result.n, n, rhs.n);
    return result;
}

FieldElement26 FieldElement26::square() const noexcept {
    FieldElement26 result;
    fe26_sqr_inner(result.n, n);
    return result;
}

void FieldElement26::mul_assign(const FieldElement26& rhs) noexcept {
    std::uint32_t tmp[10];
    fe26_mul_inner(tmp, n, rhs.n);
    for (int i = 0; i < 10; ++i) n[i] = tmp[i];
}

void FieldElement26::square_inplace() noexcept {
    std::uint32_t tmp[10];
    fe26_sqr_inner(tmp, n);
    for (int i = 0; i < 10; ++i) n[i] = tmp[i];
}

// ═══════════════════════════════════════════════════════════════════════════
// Comparison
// ═══════════════════════════════════════════════════════════════════════════

bool FieldElement26::is_zero() const noexcept {
    FieldElement26 tmp = *this;
    tmp.normalize();
    return (tmp.n[0] | tmp.n[1] | tmp.n[2] | tmp.n[3] | tmp.n[4]
          | tmp.n[5] | tmp.n[6] | tmp.n[7] | tmp.n[8] | tmp.n[9]) == 0;
}

bool FieldElement26::operator==(const FieldElement26& rhs) const noexcept {
    FieldElement26 a = *this, b = rhs;
    a.normalize();
    b.normalize();
    return (a.n[0] == b.n[0]) & (a.n[1] == b.n[1]) & (a.n[2] == b.n[2])
         & (a.n[3] == b.n[3]) & (a.n[4] == b.n[4]) & (a.n[5] == b.n[5])
         & (a.n[6] == b.n[6]) & (a.n[7] == b.n[7]) & (a.n[8] == b.n[8])
         & (a.n[9] == b.n[9]);
}

bool FieldElement26::operator!=(const FieldElement26& rhs) const noexcept {
    return !(*this == rhs);
}

// ═══════════════════════════════════════════════════════════════════════════
// Half (a/2 mod p) — branchless
// ═══════════════════════════════════════════════════════════════════════════

FieldElement26 FieldElement26::half() const noexcept {
    FieldElement26 tmp = *this;
    tmp.normalize_weak();

    // mask = 0 if even, all-ones if odd
    std::uint32_t mask = -(tmp.n[0] & 1U);

    // Conditionally add p
    std::uint32_t t0 = tmp.n[0] + (P0 & mask);
    std::uint32_t t1 = tmp.n[1] + (P1 & mask);
    std::uint32_t t2 = tmp.n[2] + (P2 & mask);
    std::uint32_t t3 = tmp.n[3] + (P3 & mask);
    std::uint32_t t4 = tmp.n[4] + (P4 & mask);
    std::uint32_t t5 = tmp.n[5] + (P5 & mask);
    std::uint32_t t6 = tmp.n[6] + (P6 & mask);
    std::uint32_t t7 = tmp.n[7] + (P7 & mask);
    std::uint32_t t8 = tmp.n[8] + (P8 & mask);
    std::uint32_t t9 = tmp.n[9] + (P9 & mask);

    // Carry propagation
    t1 += (t0 >> 26); t0 &= M26;
    t2 += (t1 >> 26); t1 &= M26;
    t3 += (t2 >> 26); t2 &= M26;
    t4 += (t3 >> 26); t3 &= M26;
    t5 += (t4 >> 26); t4 &= M26;
    t6 += (t5 >> 26); t5 &= M26;
    t7 += (t6 >> 26); t6 &= M26;
    t8 += (t7 >> 26); t7 &= M26;
    t9 += (t8 >> 26); t8 &= M26;

    // Right shift by 1 (divide by 2)
    FieldElement26 result;
    result.n[0] = (t0 >> 1) | ((t1 & 1U) << 25);
    result.n[1] = (t1 >> 1) | ((t2 & 1U) << 25);
    result.n[2] = (t2 >> 1) | ((t3 & 1U) << 25);
    result.n[3] = (t3 >> 1) | ((t4 & 1U) << 25);
    result.n[4] = (t4 >> 1) | ((t5 & 1U) << 25);
    result.n[5] = (t5 >> 1) | ((t6 & 1U) << 25);
    result.n[6] = (t6 >> 1) | ((t7 & 1U) << 25);
    result.n[7] = (t7 >> 1) | ((t8 & 1U) << 25);
    result.n[8] = (t8 >> 1) | ((t9 & 1U) << 25);
    result.n[9] = (t9 >> 1);

    return result;
}

} // namespace secp256k1::fast
