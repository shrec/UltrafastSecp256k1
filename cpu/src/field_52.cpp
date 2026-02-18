// ============================================================================
// 5×52-bit Field Element — Implementation
// ============================================================================
// Hybrid lazy-reduction field arithmetic for secp256k1.
//
// Multiplication and squaring kernels adapted from bitcoin-core/secp256k1
// field_5x52_int128_impl.h (MIT license).
//
// Key property: p = 2^256 - 0x1000003D1
//   → 2^256 ≡ 0x1000003D1 (mod p)
//   → 2^260 ≡ 0x1000003D10 (mod p)  [since d^5 = 2^(52*5) = 2^260]
// ============================================================================

#include "secp256k1/field_52.hpp"
#include <cstring>

// Require 128-bit integer support for the mul/sqr kernels.
// clang-cl, GCC, and Clang all support this on 64-bit targets.
#if defined(__SIZEOF_INT128__) || defined(__GNUC__) || defined(__clang__)
    using uint128_t = unsigned __int128;
    #define SECP256K1_HAS_UINT128 1
#else
    // MSVC (non-clang-cl) lacks __uint128_t.
    // FieldElement52 is unavailable; this TU compiles as empty.
    // The portable FieldElement (4×64 with carry chains) is used instead.
#endif

#ifdef SECP256K1_HAS_UINT128

namespace secp256k1::fast {

using namespace fe52_constants;

// ═══════════════════════════════════════════════════════════════════════════
// Construction
// ═══════════════════════════════════════════════════════════════════════════

FieldElement52 FieldElement52::zero() noexcept {
    return FieldElement52{{0, 0, 0, 0, 0}};
}

FieldElement52 FieldElement52::one() noexcept {
    return FieldElement52{{1, 0, 0, 0, 0}};
}

// ═══════════════════════════════════════════════════════════════════════════
// Conversion: 4×64 ↔ 5×52
// ═══════════════════════════════════════════════════════════════════════════
//
// 4×64 layout (256 bits total):
//   L[0] = bits[  0.. 63]   L[1] = bits[ 64..127]
//   L[2] = bits[128..191]   L[3] = bits[192..255]
//
// 5×52 layout (260 bit capacity, 256 used):
//   n[0] = bits[  0.. 51]   (52 bits from L[0])
//   n[1] = bits[ 52..103]   (12 bits from L[0] + 40 bits from L[1])
//   n[2] = bits[104..155]   (24 bits from L[1] + 28 bits from L[2])
//   n[3] = bits[156..207]   (36 bits from L[2] + 16 bits from L[3])
//   n[4] = bits[208..255]   (48 bits from L[3])

FieldElement52 FieldElement52::from_fe(const FieldElement& fe) noexcept {
    const auto& L = fe.limbs();
    FieldElement52 r;
    r.n[0] =  L[0]                           & M52;
    r.n[1] = (L[0] >> 52) | ((L[1] & 0xFFFFFFFFFFULL) << 12);
    r.n[2] = (L[1] >> 40) | ((L[2] & 0xFFFFFFFULL)    << 24);
    r.n[3] = (L[2] >> 28) | ((L[3] & 0xFFFFULL)       << 36);
    r.n[4] =  L[3] >> 16;
    return r;
}

FieldElement FieldElement52::to_fe() const noexcept {
    // Normalize a copy first to ensure canonical form
    FieldElement52 tmp = *this;
    tmp.normalize();

    FieldElement::limbs_type L;
    L[0] =  tmp.n[0]        | (tmp.n[1] << 52);
    L[1] = (tmp.n[1] >> 12) | (tmp.n[2] << 40);
    L[2] = (tmp.n[2] >> 24) | (tmp.n[3] << 28);
    L[3] = (tmp.n[3] >> 36) | (tmp.n[4] << 16);
    return FieldElement::from_limbs(L);
}

// ═══════════════════════════════════════════════════════════════════════════
// Normalization
// ═══════════════════════════════════════════════════════════════════════════

// Weak normalization: carry-propagate so each limb ≤ 52 bits.
// Result may still be ≥ p.
void fe52_normalize_weak(std::uint64_t* r) noexcept {
    std::uint64_t t0 = r[0], t1 = r[1], t2 = r[2], t3 = r[3], t4 = r[4];

    // Propagate carries
    t1 += (t0 >> 52); t0 &= M52;
    t2 += (t1 >> 52); t1 &= M52;
    t3 += (t2 >> 52); t2 &= M52;
    t4 += (t3 >> 52); t3 &= M52;

    // Any overflow past 48 bits in t4 wraps via 2^256 ≡ 0x1000003D1 (mod p)
    std::uint64_t x = t4 >> 48;
    t4 &= M48;
    t0 += x * 0x1000003D1ULL;

    // One more carry propagation for the added value
    t1 += (t0 >> 52); t0 &= M52;
    t2 += (t1 >> 52); t1 &= M52;
    t3 += (t2 >> 52); t2 &= M52;
    t4 += (t3 >> 52); t3 &= M52;

    r[0] = t0; r[1] = t1; r[2] = t2; r[3] = t3; r[4] = t4;
}

// Full normalization: canonical result in [0, p).
void fe52_normalize(std::uint64_t* r) noexcept {
    std::uint64_t t0 = r[0], t1 = r[1], t2 = r[2], t3 = r[3], t4 = r[4];

    // First pass: carry propagation + overflow reduction
    t1 += (t0 >> 52); t0 &= M52;
    t2 += (t1 >> 52); t1 &= M52;
    t3 += (t2 >> 52); t2 &= M52;
    t4 += (t3 >> 52); t3 &= M52;

    std::uint64_t x = t4 >> 48;
    t4 &= M48;
    t0 += x * 0x1000003D1ULL;

    t1 += (t0 >> 52); t0 &= M52;
    t2 += (t1 >> 52); t1 &= M52;
    t3 += (t2 >> 52); t2 &= M52;
    t4 += (t3 >> 52); t3 &= M52;

    // Second overflow reduction (extremely rare, but possible)
    x = t4 >> 48;
    t4 &= M48;
    t0 += x * 0x1000003D1ULL;
    t1 += (t0 >> 52); t0 &= M52;
    t2 += (t1 >> 52); t1 &= M52;
    t3 += (t2 >> 52); t2 &= M52;
    t4 += (t3 >> 52); t3 &= M52;

    // Now each limb fits in its range. Check if result ≥ p.
    // p = {0xFFFFEFFFFFC2F, 0xFFFFFFFFFFFFF, 0xFFFFFFFFFFFFF, 0xFFFFFFFFFFFFF, 0xFFFFFFFFFFFF}
    //
    // Branchless: compute (t + (2^256 - p)) and check if it overflows 256 bits.
    // 2^256 - p = 0x1000003D1.
    // t ≥ p  ⟺  (t + 0x1000003D1) ≥ 2^256  ⟺  overflow from 48-bit limb
    //
    // Try adding 0x1000003D1 to the value. If it overflows limb 4 past 48 bits,
    // the original value was ≥ p, and the reduced value is the result.

    std::uint64_t u0 = t0 + 0x1000003D1ULL;
    std::uint64_t u1 = t1 + (u0 >> 52); u0 &= M52;
    std::uint64_t u2 = t2 + (u1 >> 52); u1 &= M52;
    std::uint64_t u3 = t3 + (u2 >> 52); u2 &= M52;
    std::uint64_t u4 = t4 + (u3 >> 52); u3 &= M52;

    // If u4 overflows 48 bits, the original was ≥ p → use reduced value
    std::uint64_t overflow = u4 >> 48;
    u4 &= M48;

    // Branchless select: mask = -overflow (all-ones if ≥ p, all-zeros if < p)
    std::uint64_t mask = -overflow;
    r[0] = (u0 & mask) | (t0 & ~mask);
    r[1] = (u1 & mask) | (t1 & ~mask);
    r[2] = (u2 & mask) | (t2 & ~mask);
    r[3] = (u3 & mask) | (t3 & ~mask);
    r[4] = (u4 & mask) | (t4 & ~mask);
}

void FieldElement52::normalize_weak() noexcept {
    fe52_normalize_weak(n);
}

void FieldElement52::normalize() noexcept {
    fe52_normalize(n);
}

// ═══════════════════════════════════════════════════════════════════════════
// Lazy Addition (NO carry propagation!)
// ═══════════════════════════════════════════════════════════════════════════

FieldElement52 FieldElement52::operator+(const FieldElement52& rhs) const noexcept {
    FieldElement52 r;
    r.n[0] = n[0] + rhs.n[0];
    r.n[1] = n[1] + rhs.n[1];
    r.n[2] = n[2] + rhs.n[2];
    r.n[3] = n[3] + rhs.n[3];
    r.n[4] = n[4] + rhs.n[4];
    return r;
}

void FieldElement52::add_assign(const FieldElement52& rhs) noexcept {
    n[0] += rhs.n[0];
    n[1] += rhs.n[1];
    n[2] += rhs.n[2];
    n[3] += rhs.n[3];
    n[4] += rhs.n[4];
}

// ═══════════════════════════════════════════════════════════════════════════
// Negate: computes (M+1)*p - a to guarantee positive result.
// M must be ≥ the current magnitude of the element.
// ═══════════════════════════════════════════════════════════════════════════

FieldElement52 FieldElement52::negate(unsigned magnitude) const noexcept {
    FieldElement52 r = *this;
    r.negate_assign(magnitude);
    return r;
}

void FieldElement52::negate_assign(unsigned magnitude) noexcept {
    // (M+1)*p in 5×52. Since p = {P0, P1, P2, P3, P4}:
    const std::uint64_t m1 = static_cast<std::uint64_t>(magnitude) + 1ULL;
    const std::uint64_t mp0 = m1 * P0;
    const std::uint64_t mp1 = m1 * P1;
    const std::uint64_t mp2 = m1 * P2;
    const std::uint64_t mp3 = m1 * P3;
    const std::uint64_t mp4 = m1 * P4;
    n[0] = mp0 - n[0];
    n[1] = mp1 - n[1];
    n[2] = mp2 - n[2];
    n[3] = mp3 - n[3];
    n[4] = mp4 - n[4];
}

// ═══════════════════════════════════════════════════════════════════════════
// Multiplication (5×52 with inline secp256k1 reduction)
// ═══════════════════════════════════════════════════════════════════════════
//
// Adapted from bitcoin-core/secp256k1 field_5x52_int128_impl.h.
//
// Product column layout (a_i * b_j contributes to column i+j):
//   col 0: a0*b0
//   col 1: a0*b1 + a1*b0
//   col 2: a0*b2 + a1*b1 + a2*b0
//   col 3: a0*b3 + a1*b2 + a2*b1 + a3*b0
//   col 4: a0*b4 + a1*b3 + a2*b2 + a3*b1 + a4*b0
//   col 5: a1*b4 + a2*b3 + a3*b2 + a4*b1     → reduced to col 0
//   col 6: a2*b4 + a3*b3 + a4*b2              → reduced to col 1
//   col 7: a3*b4 + a4*b3                      → reduced to col 2
//   col 8: a4*b4                              → reduced to col 3
//
// Reduction: col_k (k≥5) × d^k = col_k × d^(k-5) × d^5
//   where d^5 = 2^260 ≡ R (mod p), R = 0x1000003D10
//
// The algorithm processes columns out of order (3,4,0,1,2,3,4) to keep
// accumulator values within 128 bits.

void fe52_mul_inner(std::uint64_t* r, const std::uint64_t* a,
                    const std::uint64_t* b) noexcept {
    uint128_t c, d;
    std::uint64_t t3, t4, tx, u0;
    const std::uint64_t a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3], a4 = a[4];

    // ── Column 3 + reduced column 8 ─────────────────────────────────
    d  = (uint128_t)a0 * b[3]
       + (uint128_t)a1 * b[2]
       + (uint128_t)a2 * b[1]
       + (uint128_t)a3 * b[0];
    // Column 8: a4*b4
    c  = (uint128_t)a4 * b[4];
    // Reduce: col8 * R → add to column 3
    d += (uint128_t)R52 * (std::uint64_t)c;
    c >>= 64;
    // Extract column 3
    t3 = (std::uint64_t)d & M52;
    d >>= 52;

    // ── Column 4 + remaining col8 carry ─────────────────────────────
    d += (uint128_t)a0 * b[4]
       + (uint128_t)a1 * b[3]
       + (uint128_t)a2 * b[2]
       + (uint128_t)a3 * b[1]
       + (uint128_t)a4 * b[0];
    // Remaining high bits of col8 * R (shifted by 64-52=12 bits alignment)
    d += (uint128_t)(R52 << 12) * (std::uint64_t)c;
    // Extract column 4 (only 48 bits are "real", upper 4 bits are overflow)
    t4 = (std::uint64_t)d & M52;
    d >>= 52;
    tx = (t4 >> 48); t4 &= (M52 >> 4);  // M52>>4 = 0xFFFFFFFFFFFF = M48

    // ── Column 0 + reduced column 5 ─────────────────────────────────
    c  = (uint128_t)a0 * b[0];
    // Column 5 cross-products
    d += (uint128_t)a1 * b[4]
       + (uint128_t)a2 * b[3]
       + (uint128_t)a3 * b[2]
       + (uint128_t)a4 * b[1];
    // Extract column 5 low bits, combine with t4 overflow
    u0 = (std::uint64_t)d & M52;
    d >>= 52;
    // u0 holds col5 low 52 bits, tx holds col4 overflow (4 bits)
    // Combine: u0<<4 | tx gives the full value that needs R-reduction to col0
    u0 = (u0 << 4) | tx;
    // Reduce to column 0: u0 * (R >> 4) = u0 * 0x1000003D1
    c += (uint128_t)u0 * (R52 >> 4);
    r[0] = (std::uint64_t)c & M52;
    c >>= 52;

    // ── Column 1 + reduced column 6 ─────────────────────────────────
    c += (uint128_t)a0 * b[1]
       + (uint128_t)a1 * b[0];
    // Column 6 cross-products
    d += (uint128_t)a2 * b[4]
       + (uint128_t)a3 * b[3]
       + (uint128_t)a4 * b[2];
    // Reduce col6 → col1
    c += (uint128_t)((std::uint64_t)d & M52) * R52;
    d >>= 52;
    r[1] = (std::uint64_t)c & M52;
    c >>= 52;

    // ── Column 2 + reduced column 7 ─────────────────────────────────
    // NOTE: Column 7 uses a DIFFERENT reduction than columns 5,6!
    // Instead of extracting 52 bits and shifting by 52, we extract the
    // full low 64 bits of d, multiply by R, and shift d by 64.
    // The remaining high bits of d are then handled in the finalize step.
    c += (uint128_t)a0 * b[2]
       + (uint128_t)a1 * b[1]
       + (uint128_t)a2 * b[0];
    // Column 7 cross-products
    d += (uint128_t)a3 * b[4]
       + (uint128_t)a4 * b[3];
    // Reduce col7 → col2: use FULL low 64 bits, shift by 64
    c += (uint128_t)R52 * (std::uint64_t)d;
    d >>= 64;
    r[2] = (std::uint64_t)c & M52;
    c >>= 52;

    // ── Finalize columns 3 and 4 ────────────────────────────────────
    // d holds remaining high bits from column 7 (similar to col8's high bits).
    // These are at a 12-bit offset, so multiply by (R << 12).
    c += (uint128_t)(R52 << 12) * (std::uint64_t)d;
    c += t3;
    r[3] = (std::uint64_t)c & M52;
    c >>= 52;
    c += t4;
    r[4] = (std::uint64_t)c;

    // Output magnitude = 1 (each limb ≤ 52 bits plus small carry)
}

// ═══════════════════════════════════════════════════════════════════════════
// Squaring (5×52 with symmetry optimization)
// ═══════════════════════════════════════════════════════════════════════════
//
// Uses a[i]*a[j] == a[j]*a[i] symmetry to halve cross-product count.
// Column k: sum of a[i]*a[j] for i+j=k
//   → for i≠j, compute once and double (via a[i]*2)
//   → for i=j, compute a[i]² once

void fe52_sqr_inner(std::uint64_t* r, const std::uint64_t* a) noexcept {
    uint128_t c, d;
    std::uint64_t t3, t4, tx, u0;
    const std::uint64_t a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3], a4 = a[4];

    // ── Column 3 + reduced column 8 ─────────────────────────────────
    d  = (uint128_t)(a0 * 2) * a3
       + (uint128_t)(a1 * 2) * a2;
    c  = (uint128_t)a4 * a4;
    d += (uint128_t)R52 * (std::uint64_t)c;
    c >>= 64;
    t3 = (std::uint64_t)d & M52;
    d >>= 52;

    // ── Column 4 ────────────────────────────────────────────────────
    d += (uint128_t)(a0 * 2) * a4
       + (uint128_t)(a1 * 2) * a3
       + (uint128_t)a2 * a2;
    d += (uint128_t)(R52 << 12) * (std::uint64_t)c;
    t4 = (std::uint64_t)d & M52;
    d >>= 52;
    tx = (t4 >> 48); t4 &= (M52 >> 4);

    // ── Column 0 + reduced column 5 ─────────────────────────────────
    c  = (uint128_t)a0 * a0;
    d += (uint128_t)(a1 * 2) * a4
       + (uint128_t)(a2 * 2) * a3;
    u0 = (std::uint64_t)d & M52;
    d >>= 52;
    u0 = (u0 << 4) | tx;
    c += (uint128_t)u0 * (R52 >> 4);
    r[0] = (std::uint64_t)c & M52;
    c >>= 52;

    // ── Column 1 + reduced column 6 ─────────────────────────────────
    c += (uint128_t)(a0 * 2) * a1;
    d += (uint128_t)(a2 * 2) * a4
       + (uint128_t)a3 * a3;
    c += (uint128_t)((std::uint64_t)d & M52) * R52;
    d >>= 52;
    r[1] = (std::uint64_t)c & M52;
    c >>= 52;

    // ── Column 2 + reduced column 7 ─────────────────────────────────
    // Same col7 handling as mul: full 64-bit extract, shift by 64
    c += (uint128_t)(a0 * 2) * a2
       + (uint128_t)a1 * a1;
    d += (uint128_t)(a3 * 2) * a4;
    c += (uint128_t)R52 * (std::uint64_t)d;
    d >>= 64;
    r[2] = (std::uint64_t)c & M52;
    c >>= 52;

    // ── Finalize columns 3 and 4 ────────────────────────────────────
    c += (uint128_t)(R52 << 12) * (std::uint64_t)d;
    c += t3;
    r[3] = (std::uint64_t)c & M52;
    c >>= 52;
    c += t4;
    r[4] = (std::uint64_t)c;
}

// ═══════════════════════════════════════════════════════════════════════════
// Method wrappers
// ═══════════════════════════════════════════════════════════════════════════

FieldElement52 FieldElement52::operator*(const FieldElement52& rhs) const noexcept {
    FieldElement52 r;
    fe52_mul_inner(r.n, n, rhs.n);
    return r;
}

FieldElement52 FieldElement52::square() const noexcept {
    FieldElement52 r;
    fe52_sqr_inner(r.n, n);
    return r;
}

void FieldElement52::mul_assign(const FieldElement52& rhs) noexcept {
    std::uint64_t tmp[5];
    fe52_mul_inner(tmp, n, rhs.n);
    n[0] = tmp[0]; n[1] = tmp[1]; n[2] = tmp[2]; n[3] = tmp[3]; n[4] = tmp[4];
}

void FieldElement52::square_inplace() noexcept {
    std::uint64_t tmp[5];
    fe52_sqr_inner(tmp, n);
    n[0] = tmp[0]; n[1] = tmp[1]; n[2] = tmp[2]; n[3] = tmp[3]; n[4] = tmp[4];
}

// ═══════════════════════════════════════════════════════════════════════════
// Comparison
// ═══════════════════════════════════════════════════════════════════════════

bool FieldElement52::is_zero() const noexcept {
    // Normalize a copy and check
    FieldElement52 tmp = *this;
    tmp.normalize();
    return (tmp.n[0] | tmp.n[1] | tmp.n[2] | tmp.n[3] | tmp.n[4]) == 0;
}

bool FieldElement52::operator==(const FieldElement52& rhs) const noexcept {
    FieldElement52 a = *this, b = rhs;
    a.normalize();
    b.normalize();
    return (a.n[0] == b.n[0]) & (a.n[1] == b.n[1]) & (a.n[2] == b.n[2])
         & (a.n[3] == b.n[3]) & (a.n[4] == b.n[4]);
}

bool FieldElement52::operator!=(const FieldElement52& rhs) const noexcept {
    return !(*this == rhs);
}

// ═══════════════════════════════════════════════════════════════════════════
// Half (a/2 mod p) — branchless
// ═══════════════════════════════════════════════════════════════════════════
//
// If a is even: result = a >> 1
// If a is odd:  result = (a + p) >> 1
// Since p is odd, (a + p) is even when a is odd.

FieldElement52 FieldElement52::half() const noexcept {
    FieldElement52 tmp = *this;
    tmp.normalize_weak();

    // mask = 0 if even, all-ones if odd
    std::uint64_t mask = -(tmp.n[0] & 1ULL);

    // Conditionally add p
    std::uint64_t t0 = tmp.n[0] + (P0 & mask);
    std::uint64_t t1 = tmp.n[1] + (P1 & mask);
    std::uint64_t t2 = tmp.n[2] + (P2 & mask);
    std::uint64_t t3 = tmp.n[3] + (P3 & mask);
    std::uint64_t t4 = tmp.n[4] + (P4 & mask);

    // Carry propagation
    t1 += (t0 >> 52); t0 &= M52;
    t2 += (t1 >> 52); t1 &= M52;
    t3 += (t2 >> 52); t2 &= M52;
    t4 += (t3 >> 52); t3 &= M52;

    // Right shift by 1 (divide by 2)
    FieldElement52 r;
    r.n[0] = (t0 >> 1) | ((t1 & 1ULL) << 51);
    r.n[1] = (t1 >> 1) | ((t2 & 1ULL) << 51);
    r.n[2] = (t2 >> 1) | ((t3 & 1ULL) << 51);
    r.n[3] = (t3 >> 1) | ((t4 & 1ULL) << 51);
    r.n[4] = (t4 >> 1);

    return r;
}

} // namespace secp256k1::fast

#endif // SECP256K1_HAS_UINT128
