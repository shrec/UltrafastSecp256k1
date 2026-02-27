#include "secp256k1/point.hpp"
#if !defined(SECP256K1_PLATFORM_ESP32) && !defined(ESP_PLATFORM) && !defined(SECP256K1_PLATFORM_STM32)
#include "secp256k1/precompute.hpp"
#endif
#include "secp256k1/glv.hpp"
#if defined(__SIZEOF_INT128__) && !defined(__EMSCRIPTEN__)
#include "secp256k1/field_52.hpp"
#endif

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>

namespace secp256k1::fast {
namespace {

// ESP32/STM32 local wNAF helpers (when precompute.hpp not included)
#if defined(SECP256K1_PLATFORM_ESP32) || defined(ESP_PLATFORM) || defined(SECP256K1_PLATFORM_STM32)
// Simple wNAF computation for ESP32 (inline, no heap allocation)
static void compute_wnaf_into(const Scalar& scalar, unsigned window_width,
                               int32_t* out, std::size_t out_capacity,
                               std::size_t& out_len) {
    // Get scalar as bytes (big-endian)
    auto bytes = scalar.to_bytes();

    // Convert to limbs (little-endian)
    uint64_t limbs[4] = {0};
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 8; j++) {
            limbs[i] |= static_cast<uint64_t>(bytes[31 - i*8 - j]) << (j*8);
        }
    }

    const int64_t width = 1 << window_width;     // 2^w
    const int64_t half_width = width >> 1;       // 2^(w-1)
    const uint64_t mask = static_cast<uint64_t>(width - 1);  // 2^w - 1

    out_len = 0;

    // Process until all bits consumed
    int bit_pos = 0;
    while (bit_pos < 256 || (limbs[0] | limbs[1] | limbs[2] | limbs[3]) != 0) {
        if (limbs[0] & 1) {
            // Current bit is set - compute wNAF digit
            int64_t digit = static_cast<int64_t>(limbs[0] & mask);
            if (digit >= half_width) {
                digit -= width;
            }
            out[out_len++] = static_cast<int32_t>(digit);

            // Subtract digit from scalar (handling negative digits as addition)
            if (digit > 0) {
                // Subtract positive digit
                uint64_t borrow = static_cast<uint64_t>(digit);
                for (int i = 0; i < 4 && borrow; i++) {
                    if (limbs[i] >= borrow) {
                        limbs[i] -= borrow;
                        borrow = 0;
                    } else {
                        uint64_t old = limbs[i];
                        limbs[i] -= borrow;  // wraps
                        borrow = 1;
                    }
                }
            } else if (digit < 0) {
                // Add absolute value of negative digit
                uint64_t carry = static_cast<uint64_t>(-digit);
                for (int i = 0; i < 4 && carry; i++) {
                    uint64_t sum = limbs[i] + carry;
                    carry = (sum < limbs[i]) ? 1 : 0;
                    limbs[i] = sum;
                }
            }
        } else {
            out[out_len++] = 0;
        }

        // Right-shift scalar by 1
        limbs[0] = (limbs[0] >> 1) | (limbs[1] << 63);
        limbs[1] = (limbs[1] >> 1) | (limbs[2] << 63);
        limbs[2] = (limbs[2] >> 1) | (limbs[3] << 63);
        limbs[3] >>= 1;

        bit_pos++;
        if (out_len >= out_capacity - 1) break;
    }
}

static std::vector<int32_t> compute_wnaf(const Scalar& scalar, unsigned window_bits) {
    std::array<int32_t, 260> buf{};
    std::size_t len = 0;
    compute_wnaf_into(scalar, window_bits, buf.data(), buf.size(), len);
    return std::vector<int32_t>(buf.begin(), buf.begin() + len);
}
#endif // ESP32

inline FieldElement fe_from_uint(std::uint64_t v) {
    return FieldElement::from_uint64(v);
}

// Affine coordinates (x, y) - more compact than Jacobian
struct AffinePoint {
    FieldElement x;
    FieldElement y;
};

struct JacobianPoint {
    FieldElement x;
    FieldElement y;
    FieldElement z;
    bool infinity{true};
};

// Hot path: Point doubling - force aggressive inlining
// Optimized dbl-2007-a formula for a=0 curves (secp256k1)
// Operations: 4 squarings + 4 multiplications (vs previous 5 sqr + 3+ mul)
// Uses additions instead of multiplications for small constants (2, 3, 8)
#if defined(_MSC_VER) && !defined(__clang__)
#pragma inline_recursion(on)
#pragma inline_depth(255)
#endif
SECP256K1_HOT_FUNCTION
JacobianPoint jacobian_double(const JacobianPoint& p) {
    if (SECP256K1_UNLIKELY(p.infinity || p.y == FieldElement::zero())) {
        return {FieldElement::zero(), FieldElement::one(), FieldElement::zero(), true};
    }

    // A = X^2 (in-place)
    FieldElement A = p.x;           // Copy for in-place
    A.square_inplace();             // X^2 in-place!
    
    // B = Y^2 (in-place)
    FieldElement B = p.y;           // Copy for in-place
    B.square_inplace();             // Y^2 in-place!
    
    // C = B^2 (in-place)
    FieldElement C = B;             // Copy for in-place
    C.square_inplace();             // B^2 in-place!
    
    // D = 2*((X + B)^2 - A - C)
    FieldElement temp = p.x + B;
    temp.square_inplace();          // (X + B)^2 in-place!
    temp = temp - A;
    temp = temp - C;
    FieldElement D = temp + temp;  // *2 via addition (faster than mul)
    
    // E = 3*A
    FieldElement E = A + A;
    E = E + A;  // *3 via two additions (faster than mul)
    
    // F = E^2 (in-place)
    FieldElement F = E;             // Copy for in-place
    F.square_inplace();             // E^2 in-place!
    
    // X' = F - 2*D
    FieldElement two_D = D + D;
    FieldElement x3 = F - two_D;
    
    // Y' = E*(D - X') - 8*C
    FieldElement y3 = E * (D - x3);
    FieldElement eight_C = C + C;  // 2C
    eight_C = eight_C + eight_C;  // 4C
    eight_C = eight_C + eight_C;  // 8C (3 additions vs 1 mul)
    y3 = y3 - eight_C;
    
    // Z' = 2*Y*Z
    FieldElement z3 = p.y * p.z;
    z3 = z3 + z3;  // *2 via addition
    
    return {x3, y3, z3, false};
}

// Hot path: Mixed addition - optimize heavily
#if defined(_MSC_VER) && !defined(__clang__)
#pragma inline_recursion(on)
#pragma inline_depth(255)
#endif
// Mixed Jacobian-Affine addition: P (Jacobian) + Q (Affine) -> Result (Jacobian)
// Optimized with dbl-2007-mix formula (7M + 4S for a=0)
[[nodiscard]] [[maybe_unused]] JacobianPoint jacobian_add_mixed(const JacobianPoint& p, const AffinePoint& q) {
    if (SECP256K1_UNLIKELY(p.infinity)) {
        // Convert affine to Jacobian: (x, y) -> (x, y, 1, false)
        return {q.x, q.y, FieldElement::one(), false};
    }

    // Core formula optimized for a=0 curve
    FieldElement z1z1 = p.z;                    // Copy for in-place
    z1z1.square_inplace();                      // Z1^2 in-place! [1S]
    FieldElement u2 = q.x * z1z1;               // U2 = X2*Z1^2 [1M]
    FieldElement s2 = q.y * p.z * z1z1;         // S2 = Y2*Z1^3 [2M]
    
    if (SECP256K1_UNLIKELY(p.x == u2)) {
        if (p.y == s2) {
            return jacobian_double(p);
        }
        return {FieldElement::zero(), FieldElement::one(), FieldElement::zero(), true};
    }

    FieldElement h = u2 - p.x;                  // H = U2 - X1
    FieldElement hh = h;                        // Copy for in-place
    hh.square_inplace();                        // HH = H^2 in-place! [1S]
    FieldElement i = hh + hh + hh + hh;         // I = 4*HH (3 additions)
    FieldElement j = h * i;                     // J = H*I [1M]
    FieldElement r = (s2 - p.y) + (s2 - p.y);   // r = 2*(S2 - Y1)
    FieldElement v = p.x * i;                   // V = X1*I [1M]
    
    FieldElement x3 = r;                        // Copy for in-place
    x3.square_inplace();                        // r^2 in-place!
    x3 -= j + v + v;                            // X3 = r^2 - J - 2*V [1S]
    
    // Y3 = r*(V - X3) - 2*Y1*J -- optimized: *2 via addition
    FieldElement y1j = p.y * j;                 // [1M]
    FieldElement y3 = r * (v - x3) - (y1j + y1j); // [1M]
    
    // Z3 = (Z1+H)^2 - Z1^2 - HH -- in-place for performance
    FieldElement z3 = p.z + h;                  // Z1 + H
    z3.square_inplace();                        // (Z1+H)^2 in-place!
    z3 -= z1z1 + hh;                           // Z3 = (Z1+H)^2 - Z1^2 - HH [1S: total 7M + 4S]

    return {x3, y3, z3, false};
}

// Fast z==1 check for 4x64 FieldElement: raw limb comparison, no allocation
// Valid for z values set from FieldElement::one() or fe_from_uint(1) which are already canonical
[[maybe_unused]]
static inline bool fe_is_one_raw(const FieldElement& z) noexcept {
    const auto& l = z.limbs();
    return static_cast<bool>(static_cast<int>(l[0] == 1) & static_cast<int>(l[1] == 0) & static_cast<int>(l[2] == 0) & static_cast<int>(l[3] == 0));
}

// -- In-Place Point Doubling (4x64) --------------------------------------
// Same formula as jacobian_double but overwrites input in-place.
// Eliminates 100-byte return-value copy per call.
SECP256K1_HOT_FUNCTION
static inline void jacobian_double_inplace(JacobianPoint& p) {
    if (SECP256K1_UNLIKELY(p.infinity || p.y == FieldElement::zero())) {
        p = {FieldElement::zero(), FieldElement::one(), FieldElement::zero(), true};
        return;
    }

    FieldElement A = p.x; A.square_inplace();
    FieldElement B = p.y; B.square_inplace();
    FieldElement C = B;   C.square_inplace();

    FieldElement temp = p.x + B;
    temp.square_inplace();
    temp = temp - A - C;
    FieldElement D = temp + temp;

    FieldElement E = A + A; E = E + A;
    FieldElement F = E; F.square_inplace();

    FieldElement two_D = D + D;
    FieldElement x3 = F - two_D;

    FieldElement y3 = E * (D - x3);
    FieldElement eight_C = C + C; eight_C = eight_C + eight_C; eight_C = eight_C + eight_C;
    y3 = y3 - eight_C;

    FieldElement z3 = p.y * p.z;  // reads p.y, p.z BEFORE overwrite
    z3 = z3 + z3;

    p.x = x3; p.y = y3; p.z = z3;
    p.infinity = false;
}

// -- In-Place Mixed Addition (4x64): Jacobian + Affine -> Jacobian --------
// Same formula as jacobian_add_mixed but overwrites p in-place.
[[maybe_unused]] SECP256K1_HOT_FUNCTION
static inline void jacobian_add_mixed_inplace(JacobianPoint& p, const AffinePoint& q) {
    if (SECP256K1_UNLIKELY(p.infinity)) {
        p = {q.x, q.y, FieldElement::one(), false};
        return;
    }

    FieldElement z1z1 = p.z; z1z1.square_inplace();
    FieldElement u2 = q.x * z1z1;
    FieldElement s2 = q.y * p.z * z1z1;

    if (SECP256K1_UNLIKELY(p.x == u2)) {
        if (p.y == s2) {
            jacobian_double_inplace(p);
            return;
        }
        p = {FieldElement::zero(), FieldElement::one(), FieldElement::zero(), true};
        return;
    }

    FieldElement h = u2 - p.x;
    FieldElement hh = h; hh.square_inplace();
    FieldElement i = hh + hh + hh + hh;
    FieldElement j = h * i;
    FieldElement r = (s2 - p.y) + (s2 - p.y);
    FieldElement v = p.x * i;

    FieldElement x3 = r; x3.square_inplace();
    x3 -= j + v + v;

    FieldElement y1j = p.y * j;
    FieldElement y3 = r * (v - x3) - (y1j + y1j);

    FieldElement z3 = p.z + h; z3.square_inplace(); z3 -= z1z1 + hh;

    p.x = x3; p.y = y3; p.z = z3;
    p.infinity = false;
}

// -- In-Place Full Jacobian Addition (4x64): p += q ----------------------
// Same formula as jacobian_add but overwrites p in-place.
SECP256K1_HOT_FUNCTION
static inline void jacobian_add_inplace(JacobianPoint& p, const JacobianPoint& q) {
    if (SECP256K1_UNLIKELY(p.infinity)) { p = q; return; }
    if (SECP256K1_UNLIKELY(q.infinity)) return;

    FieldElement z1z1 = p.z; z1z1.square_inplace();
    FieldElement z2z2 = q.z; z2z2.square_inplace();
    FieldElement u1 = p.x * z2z2;
    FieldElement u2 = q.x * z1z1;
    FieldElement s1 = p.y * q.z * z2z2;
    FieldElement s2 = q.y * p.z * z1z1;

    if (u1 == u2) {
        if (s1 == s2) { jacobian_double_inplace(p); return; }
        p = {FieldElement::zero(), FieldElement::one(), FieldElement::zero(), true};
        return;
    }

    FieldElement h = u2 - u1;
    FieldElement i = h + h; i.square_inplace();
    FieldElement j = h * i;
    FieldElement r = (s2 - s1) + (s2 - s1);
    FieldElement v = u1 * i;

    FieldElement x3 = r; x3.square_inplace();
    x3 -= j + v + v;
    FieldElement s1j = s1 * j;
    FieldElement y3 = r * (v - x3) - (s1j + s1j);
    FieldElement temp_z = p.z + q.z; temp_z.square_inplace();
    FieldElement z3 = (temp_z - z1z1 - z2z2) * h;

    p.x = x3; p.y = y3; p.z = z3;
    p.infinity = false;
}

[[nodiscard]] JacobianPoint jacobian_add(const JacobianPoint& p, const JacobianPoint& q) {
    if (SECP256K1_UNLIKELY(p.infinity)) {
        return q;
    }
    if (SECP256K1_UNLIKELY(q.infinity)) {
        return p;
    }

    FieldElement z1z1 = p.z;                    // Copy for in-place
    z1z1.square_inplace();                      // z1^2 in-place!
    FieldElement z2z2 = q.z;                    // Copy for in-place
    z2z2.square_inplace();                      // z2^2 in-place!
    FieldElement u1 = p.x * z2z2;
    FieldElement u2 = q.x * z1z1;
    FieldElement s1 = p.y * q.z * z2z2;
    FieldElement s2 = q.y * p.z * z1z1;

    if (u1 == u2) {
        if (s1 == s2) {
            return jacobian_double(p);
        }
        return {FieldElement::zero(), FieldElement::one(), FieldElement::zero(), true};
    }

    FieldElement h = u2 - u1;
    FieldElement i = h + h;                     // 2*h
    i.square_inplace();                         // (2*h)^2 in-place!
    FieldElement j = h * i;
    FieldElement r = (s2 - s1) + (s2 - s1);
    FieldElement v = u1 * i;

    FieldElement x3 = r;                        // Copy for in-place
    x3.square_inplace();                        // r^2 in-place!
    x3 -= j + v + v;                           // x3 = r^2 - j - 2*v
    FieldElement s1j = s1 * j;
    FieldElement y3 = r * (v - x3) - (s1j + s1j);
    FieldElement temp_z = p.z + q.z;           // z1 + z2
    temp_z.square_inplace();                   // (z1 + z2)^2 in-place!
    FieldElement z3 = (temp_z - z1z1 - z2z2) * h;

    return {x3, y3, z3, false};
}

// ===============================================================================
//  5x52 FIELD -- FAST POINT OPERATIONS
// ===============================================================================
//
// On 64-bit platforms with __int128, use FieldElement52 (5x52-bit limbs) for
// ECC point operations. Advantages:
//   - Multiplication 2.7x faster (22 ns vs 50 ns)
//   - Squaring 2.4x faster (17 ns vs 39 ns)
//   - Addition nearly free (no carry propagation, just 5 plain adds)
//   - Subtraction via negate + add (still cheaper than 4x64 borrow chain)
//
// This gives ~2.5-3x speedup to point double/add/mixed-add.
// ===============================================================================

#if defined(__SIZEOF_INT128__) && !defined(__EMSCRIPTEN__)
#define SECP256K1_FAST_52BIT 1

struct JacobianPoint52 {
    FieldElement52 x;
    FieldElement52 y;
    FieldElement52 z;
    bool infinity{true};
};

struct AffinePoint52 {
    FieldElement52 x;
    FieldElement52 y;
};

// Convert Point -> 5x52 JacobianPoint52
// With FE52 storage: zero-cost (direct member copy)
// Without FE52 storage: 3x from_fe conversions
static inline JacobianPoint52 to_jac52(const Point& p) {
#if defined(SECP256K1_FAST_52BIT)
    return {p.X52(), p.Y52(), p.Z52(), p.is_infinity()};
#else
    return {
        FieldElement52::from_fe(p.X()),
        FieldElement52::from_fe(p.Y()),
        FieldElement52::from_fe(p.z()),
        p.is_infinity()
    };
#endif
}

// Fast z==1 check: raw limb comparison, no normalization
// Valid only for z values set directly from FieldElement52::one() (already canonical)
static inline bool fe52_is_one_raw(const FieldElement52& z) noexcept {
    return (z.n[0] == 1) & (z.n[1] == 0) & (z.n[2] == 0) & (z.n[3] == 0) & (z.n[4] == 0);
}

// Convert 5x52 JacobianPoint52 -> Point (zero conversion on FE52 path)
static inline Point from_jac52(const JacobianPoint52& j) {
#if defined(SECP256K1_FAST_52BIT)
    return Point::from_jacobian52(j.x, j.y, j.z, j.infinity);
#else
    return Point::from_jacobian_coords(j.x.to_fe(), j.y.to_fe(), j.z.to_fe(), j.infinity);
#endif
}

// -- Point Doubling (5x52) ----------------------------------------------------
// Formula: dbl-2009-l (a=0 specialization)
// Cost: 2M + 5S + ~11A (additions near-free in 5x52)
SECP256K1_HOT_FUNCTION SECP256K1_NOINLINE
static JacobianPoint52 jac52_double(const JacobianPoint52& p) {
    if (SECP256K1_UNLIKELY(p.infinity)) {
        return {FieldElement52::zero(), FieldElement52::one(), FieldElement52::zero(), true};
    }
    // Magnitudes after mul/sqr = 1.  Safe bounds for uint128 accumulation:
    //   5 x m1 x m2 x 2^104 < 2^128  ==>  m1*m2 < 3,355,443
    //   max per-limb addition: mag x 2^52 < 2^64  ==>  mag < 4096
    //
    // Output magnitudes (no normalization): x<=23, y<=10, z<=2

    const FieldElement52 A = p.x.square();                    // mag 1
    const FieldElement52 B = p.y.square();                    // mag 1
    const FieldElement52 C = B.square();                      // mag 1

    const FieldElement52 xpb = p.x + B;                      // mag <=24
    const FieldElement52 xpb_sq = xpb.square();              // mag 1  (24^2*5 = 2880 < 3.3M [ok])
    const FieldElement52 negA = A.negate(1);                  // mag 2
    const FieldElement52 negC = C.negate(1);                  // mag 2
    const FieldElement52 D_half = xpb_sq + negA + negC;       // mag 5
    const FieldElement52 D = D_half + D_half;                 // mag 10

    const FieldElement52 E = A + A + A;                       // mag 3
    const FieldElement52 F = E.square();                      // mag 1  (3^2*5 = 45 < 3.3M [ok])

    const FieldElement52 negD2 = D.negate(10);                // mag 11
    const FieldElement52 x3 = F + negD2 + negD2;              // mag 23  (23*2^52 < 2^57 < 2^64 [ok])

    const FieldElement52 negX3 = x3.negate(23);               // mag 24
    const FieldElement52 dx = D + negX3;                      // mag 34  (34*2^52 < 2^58 < 2^64 [ok])
    FieldElement52 y3 = E * dx;                               // mag 1  (3*34 = 102 < 3.3M [ok])
    const FieldElement52 C2 = C + C;                          // mag 2
    const FieldElement52 C4 = C2 + C2;                        // mag 4
    const FieldElement52 C8 = C4 + C4;                        // mag 8
    const FieldElement52 negC8 = C8.negate(8);                // mag 9
    y3 = y3 + negC8;                                          // mag 10

    FieldElement52 z3 = p.y * p.z;                            // mag 1  (10*5 = 50 < 3.3M [ok])
    z3 = z3 + z3;                                             // mag 2

    return {x3, y3, z3, false};
}

// -- In-Place Point Doubling (5x52) ----------------------------------------
// Same formula as jac52_double but overwrites the input point in-place.
// Eliminates the 128-byte return value copy on every call.
SECP256K1_HOT_FUNCTION __attribute__((always_inline))
static inline void jac52_double_inplace(JacobianPoint52& p) {
    if (SECP256K1_UNLIKELY(p.infinity)) return;

    // -- Z3 = 2*Y1*Z1 (must read p.y, p.z before they're reused as scratch) --
    FieldElement52 z3 = p.y * p.z;                            // mag 1
    z3.add_assign(z3);                                        // mag 2

    // -- A = X1^2, E = 3*A ----------------------------------------------
    FieldElement52 A = p.x.square();                          // mag 1
    FieldElement52 E = A + A + A;                             // mag 3

    // -- B = Y1^2 (reuse p.y in-place; Y1 consumed above) --------------
    p.y.square_inplace();                                     // p.y = B, mag 1

    // -- C = B^2 = Y1^4 -------------------------------------------------
    FieldElement52 C = p.y.square();                          // mag 1

    // -- D = 2*((X1+B)^2 - A - C) -- in-place on p.x -------------------
    p.x.add_assign(p.y);                                      // p.x = X1+B, mag <=24
    p.x.square_inplace();                                     // p.x = (X1+B)^2, mag 1
    A.negate_assign(1);                                       // A = -A, mag 2 (positive A consumed -> E)
    p.x.add_assign(A);                                        // p.x -= A, mag 3
    A = C; A.negate_assign(1);                                // reuse A = -C, mag 2
    p.x.add_assign(A);                                        // p.x = D_half, mag 5
    p.x.add_assign(p.x);                                     // p.x = D, mag 10

    // -- X3 = E^2 - 2*D -- reuse A for E^2, p.y for -D ------------------
    A = E.square();                                           // A = F = E^2, mag 1
    p.y = p.x; p.y.negate_assign(10);                        // p.y = -D, mag 11
    A.add_assign(p.y);                                        // A = F - D
    A.add_assign(p.y);                                        // A = X3, mag 23

    // -- Y3 = E*(D - X3) - 8*C -- in-place chain ----------------------
    p.y = A; p.y.negate_assign(23);                           // p.y = -X3, mag 24
    p.x.add_assign(p.y);                                      // p.x = D - X3, mag 34
    E.mul_assign(p.x);                                        // E = E*(D-X3), mag 1
    C.add_assign(C);                                          // 2C, mag 2
    C.add_assign(C);                                          // 4C, mag 4
    C.add_assign(C);                                          // 8C, mag 8
    C.negate_assign(8);                                       // -8C, mag 9
    E.add_assign(C);                                          // Y3 = E*(D-X3) - 8C, mag 10

    // -- Write back ---------------------------------------------------
    p.x = A;                                                  // X3
    p.y = E;                                                  // Y3
    p.z = z3;                                                 // Z3
}

// -- Mixed Addition (5x52): Jacobian + Affine -> Jacobian ----------------------
// Formula: madd-2007-bl (a=0 specialization)
// Cost: 7M + 4S + ~12A (additions near-free in 5x52)
[[maybe_unused]] SECP256K1_HOT_FUNCTION SECP256K1_NOINLINE
static JacobianPoint52 jac52_add_mixed(const JacobianPoint52& p, const AffinePoint52& q) {
    if (SECP256K1_UNLIKELY(p.infinity)) {
        return {q.x, q.y, FieldElement52::one(), false};
    }

    // Z1Z1 = Z1^2 [1S]
    const FieldElement52 z1z1 = p.z.square();

    // U2 = X2 * Z1Z1 [1M]
    const FieldElement52 u2 = q.x * z1z1;

    // S2 = Y2 * Z1 * Z1Z1 [2M]
    const FieldElement52 z1_z1z1 = p.z * z1z1;
    const FieldElement52 s2 = q.y * z1_z1z1;

    // H = U2 - X1 [sub via negate]
    // p.x magnitude: <=23 (from jac52_double) or <=7 (from add_mixed). Use 23.
    const FieldElement52 negX1 = p.x.negate(23);     // mag 24
    FieldElement52 h = u2 + negX1;                   // mag 25
    // normalize_weak before zero-check: h is at magnitude 25 (= 1 + 24).
    // Reduce limbs to canonical range so normalizes_to_zero() is reliable.
    h.normalize_weak();

    // Check for point equality/inverse (rare -- fast normalizes_to_zero)
    if (SECP256K1_UNLIKELY(h.normalizes_to_zero())) {
        const FieldElement52 negY1 = p.y.negate(10);
        const FieldElement52 diff = s2 + negY1;
        if (diff.normalizes_to_zero()) {
            return jac52_double(p);
        }
        return {FieldElement52::zero(), FieldElement52::one(), FieldElement52::zero(), true};
    }

    // HH = H^2 [1S]
    const FieldElement52 hh = h.square();            // mag 1  (25^2*5 = 3125 < 3.3M [ok])

    // I = 4*HH [3A]
    FieldElement52 I = hh + hh;                // mag 2
    I = I + I;                                  // mag 4

    // J = H*I [1M]
    const FieldElement52 j = h * I;                  // mag 1  (25*4 = 100 < 3.3M [ok])

    // r = 2*(S2 - Y1) [sub + double]
    // p.y magnitude: <=10 (from dbl) or <=4 (from add_mixed). Use 10.
    const FieldElement52 negY1 = p.y.negate(10);     // mag 11
    FieldElement52 r = s2 + negY1;             // mag 12
    r = r + r;                                 // mag 24

    // V = X1*I [1M]
    const FieldElement52 v = p.x * I;                // mag 1  (23*4 = 92 < 3.3M [ok])

    // X3 = r^2 - J - 2*V [1S + sub]
    const FieldElement52 r_sq = r.square();          // mag 1  (24^2*5 = 2880 < 3.3M [ok])
    const FieldElement52 negJ = j.negate(1);         // mag 2
    const FieldElement52 negV = v.negate(1);         // mag 2
    const FieldElement52 x3 = r_sq + negJ + negV + negV;  // mag 7

    // Y3 = r*(V - X3) - 2*Y1*J [1M + sub]
    const FieldElement52 negX3 = x3.negate(7);       // mag 8
    const FieldElement52 vx3 = v + negX3;            // mag 9
    FieldElement52 y3 = r * vx3;               // mag 1  (24*9 = 216 < 3.3M [ok])
    const FieldElement52 y1j = p.y * j;              // mag 1  (10*1 = 10 < 3.3M [ok])
    const FieldElement52 y1j2 = y1j + y1j;           // mag 2
    const FieldElement52 negY1J2 = y1j2.negate(2);   // mag 3
    y3 = y3 + negY1J2;                         // mag 4

    // Z3 = (Z1+H)^2 - Z1Z1 - HH [1S + sub]
    // p.z magnitude: <=2 (from dbl) or <=5 (from add_mixed). Use 5.
    const FieldElement52 zh = p.z + h;               // mag 30  (5+25)
    const FieldElement52 zh_sq = zh.square();        // mag 1  (30^2*5 = 4500 < 3.3M [ok])
    const FieldElement52 negZ1Z1 = z1z1.negate(1);   // mag 2
    const FieldElement52 negHH = hh.negate(1);       // mag 2
    const FieldElement52 z3 = zh_sq + negZ1Z1 + negHH;  // mag 5

    return {x3, y3, z3, false};
}

// -- In-Place Mixed Addition (5x52): Jacobian + Affine -> Jacobian -------------
// Same formula as jac52_add_mixed but overwrites p in-place.
// noinline: reduces hot-loop code size from ~5KB to ~1KB, improving I-cache.
// The 127KB dual_scalar_mul_gen_point thrashes L1 I-cache (32-48KB) when inlined.
SECP256K1_HOT_FUNCTION SECP256K1_NOINLINE
static void jac52_add_mixed_inplace(JacobianPoint52& p, const AffinePoint52& q) {
    if (SECP256K1_UNLIKELY(p.infinity)) {
        p.x = q.x; p.y = q.y; p.z = FieldElement52::one(); p.infinity = false;
        return;
    }

    FieldElement52 z1z1 = p.z.square();
    FieldElement52 u2 = q.x * z1z1;
    FieldElement52 z1_z1z1 = p.z * z1z1;
    FieldElement52 s2 = q.y * z1_z1z1;                       // z1_z1z1 dead

    FieldElement52 negX1 = p.x.negate(23);
    FieldElement52 h = u2 + negX1;                            // u2, negX1 dead

    // normalize_weak before zero-check: h is at magnitude 25 (= 1 + 24).
    // Reduce limbs to canonical range so normalizes_to_zero() is reliable.
    h.normalize_weak();
    if (SECP256K1_UNLIKELY(h.normalizes_to_zero())) {
        FieldElement52 negY1 = p.y.negate(10);
        FieldElement52 diff = s2 + negY1;
        if (diff.normalizes_to_zero()) {
            jac52_double_inplace(p);
            return;
        }
        p = {FieldElement52::zero(), FieldElement52::one(), FieldElement52::zero(), true};
        return;
    }

    FieldElement52 hh = h.square();
    FieldElement52 I = hh + hh;
    I.add_assign(I);                                          // I = 4*HH (in-place)
    FieldElement52 j = h * I;

    // Compute Y1*J before negating j (reads p.y, j intact)
    FieldElement52 y1j = p.y * j;                             // mag 1
    j.negate_assign(1);                                       // j = -J (reuse slot)

    FieldElement52 negY1 = p.y.negate(10);
    FieldElement52 r = s2 + negY1;                            // s2, negY1 dead
    r.add_assign(r);                                          // r = 2*(S2-Y1), mag 24 (in-place)
    FieldElement52 v = p.x * I;                               // I dead
    FieldElement52 r_sq = r.square();
    FieldElement52 negV = v.negate(1);
    FieldElement52 x3 = r_sq + j + negV + negV;               // j is -J, mag 7
    FieldElement52 negX3 = x3.negate(7);
    FieldElement52 vx3 = v + negX3;                           // v, negX3 dead
    r.mul_assign(vx3);                                        // r = r*(V-X3), mag 1 (in-place)
    y1j.add_assign(y1j);                                      // 2*Y1*J (in-place)
    y1j.negate_assign(2);                                     // -2*Y1*J (in-place)
    r.add_assign(y1j);                                        // Y3 = r*(V-X3) - 2*Y1*J

    // Z3 = (Z1+H)^2 - Z1Z1 - HH -- compute directly into p.z
    p.z.add_assign(h);                                        // p.z = Z1+H (h dead)
    p.z.square_inplace();                                     // (Z1+H)^2
    z1z1.negate_assign(1);                                    // -Z1Z1 (in-place)
    hh.negate_assign(1);                                      // -HH (in-place)
    p.z.add_assign(z1z1);                                     // - Z1Z1
    p.z.add_assign(hh);                                       // Z3, mag 5

    p.x = x3;
    p.y = r;                                                  // Y3
    p.infinity = false;
}

// -- In-Place Full Jacobian Addition (5x52): p += q --------------------------
// Formula: add-2007-bl (a=0)
// Cost: 12M + 5S + ~11A.  Eliminates 128-byte return copy.
SECP256K1_HOT_FUNCTION __attribute__((always_inline))
static inline void jac52_add_inplace(JacobianPoint52& p, const JacobianPoint52& q) {
    if (SECP256K1_UNLIKELY(p.infinity)) { p = q; return; }
    if (SECP256K1_UNLIKELY(q.infinity)) return;

    FieldElement52 z1z1 = p.z.square();
    FieldElement52 z2z2 = q.z.square();
    FieldElement52 u1 = p.x * z2z2;
    FieldElement52 u2 = q.x * z1z1;
    FieldElement52 s1 = p.y * q.z * z2z2;
    FieldElement52 s2 = q.y * p.z * z1z1;

    FieldElement52 negU1 = u1.negate(1);
    FieldElement52 h = u2 + negU1;

    if (SECP256K1_UNLIKELY(h.normalizes_to_zero())) {
        FieldElement52 negS1 = s1.negate(1);
        FieldElement52 diff = s2 + negS1;
        if (diff.normalizes_to_zero()) { jac52_double_inplace(p); return; }
        p = {FieldElement52::zero(), FieldElement52::one(), FieldElement52::zero(), true};
        return;
    }

    FieldElement52 h2 = h + h;
    FieldElement52 i = h2.square();
    FieldElement52 j_val = h * i;

    // Compute S1*J before negating j_val (reads s1, j_val intact)
    FieldElement52 s1j = s1 * j_val;                          // mag 1
    j_val.negate_assign(1);                                   // j_val = -J (reuse slot)

    FieldElement52 negS1 = s1.negate(1);
    FieldElement52 r = s2 + negS1;
    r.add_assign(r);                                          // r = 2*(S2-S1) (in-place)
    FieldElement52 v = u1 * i;
    FieldElement52 r_sq = r.square();
    FieldElement52 negV = v.negate(1);
    FieldElement52 x3 = r_sq + j_val + negV + negV;           // j_val is -J, mag 7
    FieldElement52 negX3 = x3.negate(7);
    FieldElement52 vx3 = v + negX3;
    r.mul_assign(vx3);                                        // r = r*(V-X3) (in-place)
    s1j.add_assign(s1j);                                      // 2*S1*J (in-place)
    s1j.negate_assign(2);                                     // -2*S1*J (in-place)
    r.add_assign(s1j);                                        // Y3

    // Z3 = ((Z1+Z2)^2 - Z1Z1 - Z2Z2) * H
    p.z.add_assign(q.z);                                      // p.z = Z1+Z2
    p.z.square_inplace();                                     // (Z1+Z2)^2
    z1z1.negate_assign(1);                                    // -Z1Z1 (in-place)
    z2z2.negate_assign(1);                                    // -Z2Z2 (in-place)
    p.z.add_assign(z1z1);
    p.z.add_assign(z2z2);
    p.z.mul_assign(h);                                        // Z3

    p.x = x3; p.y = r; p.infinity = false;
}

// -- Full Jacobian Addition (5x52): Jacobian + Jacobian -> Jacobian ------------
// Formula: add-2007-bl (a=0)
// Cost: 12M + 5S + ~11A
SECP256K1_HOT_FUNCTION SECP256K1_NOINLINE
static JacobianPoint52 jac52_add(const JacobianPoint52& p, const JacobianPoint52& q) {
    if (SECP256K1_UNLIKELY(p.infinity)) return q;
    if (SECP256K1_UNLIKELY(q.infinity)) return p;

    const FieldElement52 z1z1 = p.z.square();
    const FieldElement52 z2z2 = q.z.square();
    const FieldElement52 u1 = p.x * z2z2;
    const FieldElement52 u2 = q.x * z1z1;
    const FieldElement52 s1 = p.y * q.z * z2z2;
    const FieldElement52 s2 = q.y * p.z * z1z1;

    const FieldElement52 negU1 = u1.negate(1);              // mag 2
    const FieldElement52 h = u2 + negU1;                    // mag 3

    if (SECP256K1_UNLIKELY(h.normalizes_to_zero())) {
        const FieldElement52 negS1 = s1.negate(1);
        const FieldElement52 diff = s2 + negS1;
        if (diff.normalizes_to_zero()) return jac52_double(p);
        return {FieldElement52::zero(), FieldElement52::one(), FieldElement52::zero(), true};
    }

    const FieldElement52 h2 = h + h;                        // mag 6
    const FieldElement52 i = h2.square();                   // mag 1 (6^2*5=180 < 3.3M [ok])
    const FieldElement52 j_val = h * i;                     // mag 1 (3*1=3 < 3.3M [ok])

    const FieldElement52 negS1 = s1.negate(1);              // mag 2
    FieldElement52 r = s2 + negS1;                          // mag 3
    r = r + r;                                              // mag 6

    const FieldElement52 v = u1 * i;                        // mag 1

    const FieldElement52 r_sq = r.square();                 // mag 1 (6^2*5=180 < 3.3M [ok])
    const FieldElement52 negJ = j_val.negate(1);            // mag 2
    const FieldElement52 negV = v.negate(1);                // mag 2
    const FieldElement52 x3 = r_sq + negJ + negV + negV;    // mag 7

    const FieldElement52 negX3 = x3.negate(7);              // mag 8
    const FieldElement52 vx3 = v + negX3;                   // mag 9
    FieldElement52 y3 = r * vx3;                            // mag 1 (6*9=54 < 3.3M [ok])
    const FieldElement52 s1j = s1 * j_val;                  // mag 1
    const FieldElement52 s1j2 = s1j + s1j;                  // mag 2
    const FieldElement52 negS1J2 = s1j2.negate(2);          // mag 3
    y3 = y3 + negS1J2;                                      // mag 4

    const FieldElement52 zpz = p.z + q.z;                   // mag <=3
    const FieldElement52 zpz_sq = zpz.square();             // mag 1 (3^2*5=45 < 3.3M [ok])
    const FieldElement52 negZ1Z1 = z1z1.negate(1);          // mag 2
    const FieldElement52 negZ2Z2 = z2z2.negate(1);          // mag 2
    FieldElement52 z3 = (zpz_sq + negZ1Z1 + negZ2Z2);       // mag 5
    z3 = z3 * h;                                            // mag 1 (5*3=15 < 3.3M [ok])

    return {x3, y3, z3, false};
}

// Negate a JacobianPoint52: (X, Y, Z) -> (X, -Y, Z)
[[maybe_unused]]
static inline JacobianPoint52 jac52_negate(const JacobianPoint52& p) {
    JacobianPoint52 r = p;
    r.y.normalize_weak();
    r.y.negate_assign(1);
    r.y.normalize_weak();
    return r;
}

// -- GLV + Shamir + 5x52 scalar multiplication (isolated stack frame) -----
// Kept as a separate noinline function so the ~5 KB of local arrays
// live in their own stack frame, preventing GS-cookie corruption that
// occurs when Clang 21 inlines all jac52 helpers into Point::scalar_mul.
// The try/catch is required: it forces Clang to emit an SEH frame on
// Windows, without which the large stack frame triggers a GS-cookie fault.
SECP256K1_NOINLINE
static Point scalar_mul_glv52(const Point& base, const Scalar& scalar) {
    // Guard: infinity base or zero scalar -> result is always infinity
    if (SECP256K1_UNLIKELY(base.is_infinity() || scalar.is_zero())) {
        return Point::infinity();
    }

    try {
    // -- GLV decomposition --------------------------------------------
    GLVDecomposition decomp = glv_decompose(scalar);

    // -- Convert base point to 5x52 domain ----------------------------
    JacobianPoint52 P52 = decomp.k1_neg
        ? to_jac52(base.negate())
        : to_jac52(base);

    // -- Compute wNAF for both half-scalars ---------------------------
    constexpr unsigned glv_window = 5;
    constexpr int glv_table_size = (1 << (glv_window - 2));  // 8

    std::array<int32_t, 260> wnaf1_buf{}, wnaf2_buf{};
    std::size_t wnaf1_len = 0, wnaf2_len = 0;
    compute_wnaf_into(decomp.k1, glv_window,
                      wnaf1_buf.data(), wnaf1_buf.size(), wnaf1_len);
    compute_wnaf_into(decomp.k2, glv_window,
                      wnaf2_buf.data(), wnaf2_buf.size(), wnaf2_len);

    // Trim trailing zeros -- GLV half-scalars are ~128 bits but wNAF
    // always outputs 256+ positions. This halves the doubling count.
    while (wnaf1_len > 0 && wnaf1_buf[wnaf1_len - 1] == 0) --wnaf1_len;
    while (wnaf2_len > 0 && wnaf2_buf[wnaf2_len - 1] == 0) --wnaf2_len;

    // -- Precompute odd multiples [1P, 3P, 5P, ..., 15P] in 5x52 ------
    // Uses effective-affine technique: build table on isomorphic curve where
    // 2P is affine, so table additions use cheaper mixed add (7M+4S) instead
    // of full Jacobian add (12M+5S). Saves ~33M + 6S = ~5us on RISC-V.
    std::array<AffinePoint52, glv_table_size> tbl_P;
    std::array<AffinePoint52, glv_table_size> tbl_phiP;

    {
        // d = 2*P (Jacobian)
        JacobianPoint52 d = jac52_double(P52);
        FieldElement52 C  = d.z;              // C = d.Z
        FieldElement52 C2 = C.square();       // C^2
        FieldElement52 C3 = C2 * C;           // C^3

        // d as affine on iso curve (Z cancels in isomorphism)
        AffinePoint52 d_aff = {d.x, d.y};

        // Transform P onto iso curve: phi(P) = (P.X*C^2, P.Y*C^3, P.Z)
        std::array<JacobianPoint52, glv_table_size> iso;
        iso[0] = {P52.x * C2, P52.y * C3, P52.z, false};

        // Build rest using mixed adds on iso curve (7M+4S each, not 12M+5S)
        for (std::size_t i = 1; i < glv_table_size; i++) {
            iso[i] = iso[i - 1];
            jac52_add_mixed_inplace(iso[i], d_aff);
        }

        // Batch-invert effective Z: true Z on secp256k1 = Z_iso * C
        std::array<FieldElement52, glv_table_size> eff_z;
        for (std::size_t i = 0; i < glv_table_size; i++) {
            eff_z[i] = iso[i].z * C;
        }

        std::array<FieldElement52, glv_table_size> prods;
        prods[0] = eff_z[0];
        for (std::size_t i = 1; i < glv_table_size; i++) {
            prods[i] = prods[i - 1] * eff_z[i];
        }
        // Guard: if any eff_z is zero the cumulative product is zero
        // and batch inversion is undefined. Fall through to 4x64 path.
        if (SECP256K1_UNLIKELY(prods[glv_table_size - 1].normalizes_to_zero()))
            throw 0;  // caught by catch(...) below -> 4x64 fallback
        FieldElement52 inv = prods[glv_table_size - 1].inverse_safegcd();
        std::array<FieldElement52, glv_table_size> zs;
        for (std::size_t i = glv_table_size - 1; i > 0; --i) {
            zs[i] = prods[i - 1] * inv;
            inv = inv * eff_z[i];
        }
        zs[0] = inv;

        // Convert from iso to secp256k1 affine
        for (std::size_t i = 0; i < glv_table_size; i++) {
            FieldElement52 zinv2 = zs[i].square();
            FieldElement52 zinv3 = zinv2 * zs[i];
            tbl_P[i].x = iso[i].x * zinv2;
            tbl_P[i].y = iso[i].y * zinv3;
        }
    }

    // -- Derive phi(P) table: phi(x,y) = (beta*x, y) -----------------------
    static const FieldElement52 beta52 = FieldElement52::from_fe(
        FieldElement::from_bytes(glv_constants::BETA));

    const bool flip_phi = (decomp.k1_neg != decomp.k2_neg);
    for (std::size_t i = 0; i < glv_table_size; i++) {
        tbl_phiP[i].x = tbl_P[i].x * beta52;
        if (flip_phi) {
            tbl_phiP[i].y = tbl_P[i].y.negate(1);
            tbl_phiP[i].y.normalize_weak();
        } else {
            tbl_phiP[i].y = tbl_P[i].y;
        }
    }

    // -- Pre-compute negated tables (avoid negate+normalize_weak in hot loop)
    std::array<AffinePoint52, glv_table_size> neg_tbl_P;
    std::array<AffinePoint52, glv_table_size> neg_tbl_phiP;
    for (std::size_t i = 0; i < glv_table_size; i++) {
        neg_tbl_P[i].x = tbl_P[i].x;
        neg_tbl_P[i].y = tbl_P[i].y.negate(1);
        neg_tbl_P[i].y.normalize_weak();
        neg_tbl_phiP[i].x = tbl_phiP[i].x;
        neg_tbl_phiP[i].y = tbl_phiP[i].y.negate(1);
        neg_tbl_phiP[i].y.normalize_weak();
    }

    // -- Shamir's trick -- single doubling chain, dual lookups ---------
    JacobianPoint52 result52 = {
        FieldElement52::zero(), FieldElement52::one(),
        FieldElement52::zero(), true
    };

    const std::size_t max_len = (wnaf1_len > wnaf2_len) ? wnaf1_len : wnaf2_len;

    for (int i = static_cast<int>(max_len) - 1; i >= 0; --i) {
        jac52_double_inplace(result52);

        // k1 contribution (wnaf bufs are zero-init; d==0 is a no-op)
        {
            int32_t d = wnaf1_buf[static_cast<std::size_t>(i)];
            if (d > 0) {
                jac52_add_mixed_inplace(result52, tbl_P[static_cast<std::size_t>((d - 1) >> 1)]);
            } else if (d < 0) {
                jac52_add_mixed_inplace(result52, neg_tbl_P[static_cast<std::size_t>((-d - 1) >> 1)]);
            }
        }

        // k2 contribution
        {
            int32_t d = wnaf2_buf[static_cast<std::size_t>(i)];
            if (d > 0) {
                jac52_add_mixed_inplace(result52, tbl_phiP[static_cast<std::size_t>((d - 1) >> 1)]);
            } else if (d < 0) {
                jac52_add_mixed_inplace(result52, neg_tbl_phiP[static_cast<std::size_t>((-d - 1) >> 1)]);
            }
        }
    }

    return from_jac52(result52);
    } catch (...) {
        // 5x52 path failed -- fall through to 4x64
        constexpr unsigned window_width = 5;
        std::array<int32_t, 260> wnaf_buf{};
        std::size_t wnaf_len = 0;
        compute_wnaf_into(scalar, window_width, wnaf_buf.data(), wnaf_buf.size(), wnaf_len);
        constexpr int table_size = (1 << (window_width - 1));
        std::array<Point, table_size> precomp;
        precomp[0] = base;
        Point double_p = base;
        double_p.dbl_inplace();
        for (std::size_t i = 1; i < static_cast<std::size_t>(table_size); i++) {
            precomp[i] = precomp[i-1];
            precomp[i].add_inplace(double_p);
        }
        Point result = Point::infinity();
        for (int i = static_cast<int>(wnaf_len) - 1; i >= 0; --i) {
            result.dbl_inplace();
            int32_t digit = wnaf_buf[static_cast<std::size_t>(i)];
            if (digit > 0) {
                result.add_inplace(precomp[static_cast<std::size_t>((digit - 1) / 2)]);
            } else if (digit < 0) {
                Point neg_point = precomp[static_cast<std::size_t>((-digit - 1) / 2)];
                neg_point.negate_inplace();
                result.add_inplace(neg_point);
            }
        }
        return result;
    }
}

#endif // SECP256K1_FAST_52BIT

// Batch inversion: Convert multiple Jacobian points to Affine using Montgomery's trick
// Cost: 1 inversion + (3*n - 1) multiplications for n points
// vs n inversions for individual conversion
[[maybe_unused]] std::vector<AffinePoint> batch_to_affine(const std::vector<JacobianPoint>& jacobian_points) {
    size_t n = jacobian_points.size();
    if (n == 0) return {};
    
    std::vector<AffinePoint> affine_points;
    affine_points.reserve(n);
    
    // Special case: single point
    if (n == 1) {
        const auto& p = jacobian_points[0];
        if (p.infinity) {
            // Infinity point - use (0, 0) as representation
            affine_points.push_back({FieldElement::zero(), FieldElement::zero()});
        } else {
            FieldElement z_inv = p.z.inverse();
            FieldElement z_inv_sq = z_inv;       // Copy for in-place
            z_inv_sq.square_inplace();           // z_inv^2 in-place!
            affine_points.push_back({
                p.x * z_inv_sq,           // x/z^2
                p.y * z_inv_sq * z_inv    // y/z^3
            });
        }
        return affine_points;
    }
    
    // Montgomery's trick for batch inversion
    // Step 1: Compute prefix products
    std::vector<FieldElement> prefix(n);
    prefix[0] = jacobian_points[0].z;
    for (size_t i = 1; i < n; i++) {
        prefix[i] = prefix[i-1] * jacobian_points[i].z;
    }
    
    // Step 2: Single inversion of product
    FieldElement inv_product = prefix[n-1].inverse();
    
    // Step 3: Compute individual inverses using prefix products
    std::vector<FieldElement> z_invs(n);
    z_invs[n-1] = inv_product * prefix[n-2];
    for (std::size_t i = n - 2; i > 0; --i) {
        z_invs[i] = inv_product * prefix[i-1];
        inv_product = inv_product * jacobian_points[i+1].z;
    }
    z_invs[0] = inv_product;
    
    // Step 4: Convert to affine coordinates
    for (size_t i = 0; i < n; i++) {
        const auto& p = jacobian_points[i];
        if (p.infinity) {
            affine_points.push_back({FieldElement::zero(), FieldElement::zero()});
        } else {
            FieldElement z_inv_sq = z_invs[i];       // Copy for in-place
            z_inv_sq.square_inplace();               // z_inv^2 in-place!
            affine_points.push_back({
                p.x * z_inv_sq,              // x/z^2
                p.y * z_invs[i] * z_inv_sq   // y/z^3
            });
        }
    }
    
    return affine_points;
}

// No-alloc batch conversion: jacobian -> affine written into out[0..n)
[[maybe_unused]]
static void batch_to_affine_into(const JacobianPoint* jacobian_points,
                                 std::size_t n,
                                 AffinePoint* out) {
    if (n == 0) return;

    // Conservative upper bound for this module: w in [4,7] => tables up to 2*64 = 128
    constexpr std::size_t kMaxN = 128;
    if (n > kMaxN) {
        // Fallback to allocating version for unusually large inputs
        std::vector<JacobianPoint> tmp(jacobian_points, jacobian_points + n);
        auto v = batch_to_affine(tmp);
        for (std::size_t i = 0; i < n; ++i) out[i] = v[i];
        return;
    }

    // Special case: single point
    if (n == 1) {
        const auto& p = jacobian_points[0];
        if (p.infinity) {
            out[0] = {FieldElement::zero(), FieldElement::zero()};
        } else {
            FieldElement z_inv = p.z.inverse();
            FieldElement z_inv_sq = z_inv; z_inv_sq.square_inplace();
            out[0] = { p.x * z_inv_sq, p.y * z_inv_sq * z_inv };
        }
        return;
    }

    std::array<FieldElement, kMaxN> prefix{};
    std::array<FieldElement, kMaxN> z_invs{};

    // Step 1: prefix products
    prefix[0] = jacobian_points[0].z;
    for (std::size_t i = 1; i < n; ++i) {
        prefix[i] = prefix[i - 1] * jacobian_points[i].z;
    }

    // Step 2: invert total product
    FieldElement inv_product = prefix[n - 1].inverse();

    // Step 3: individual inverses (standard backward pass)
    // inv_product initially = (z0*z1*...*z{n-1})^{-1}
    // For i from n-1..1: z_inv[i] = inv_product * prefix[i-1]; inv_product *= z_i
    // Finally z_inv[0] = inv_product
    z_invs[n - 1] = inv_product * prefix[n - 2];
    for (std::size_t i = n - 2; i > 0; --i) {
        z_invs[i] = inv_product * prefix[i - 1];
        inv_product = inv_product * jacobian_points[i].z;
    }
    z_invs[0] = inv_product;

    // Step 4: to affine
    for (std::size_t i = 0; i < n; ++i) {
        const auto& p = jacobian_points[i];
        if (p.infinity) {
            out[i] = {FieldElement::zero(), FieldElement::zero()};
        } else {
            FieldElement z_inv_sq = z_invs[i];
            z_inv_sq.square_inplace();
            out[i] = { p.x * z_inv_sq, p.y * z_invs[i] * z_inv_sq };
        }
    }
}

} // namespace

// KPlan implementation: Cache all K-dependent work for Fixed K x Variable Q
#if !defined(SECP256K1_PLATFORM_ESP32) && !defined(ESP_PLATFORM) && !defined(SECP256K1_PLATFORM_STM32)
KPlan KPlan::from_scalar(const Scalar& k, uint8_t w) {
    // Step 1: GLV decomposition (K -> k1, k2, signs)
    auto decomp = split_scalar_glv(k);
    
    // Step 2: Compute wNAF for both scalars
    auto wnaf1 = compute_wnaf(decomp.k1, w);
    auto wnaf2 = compute_wnaf(decomp.k2, w);
    
    // Return cached plan with k1, k2 scalars preserved
    KPlan plan;
    plan.window_width = w;
    plan.k1 = decomp.k1;
    plan.k2 = decomp.k2;
    plan.wnaf1 = std::move(wnaf1);
    plan.wnaf2 = std::move(wnaf2);
    plan.neg1 = decomp.neg1;
    plan.neg2 = decomp.neg2;
    return plan;
}
#else
// ESP32: simplified implementation (no GLV, just store scalar for fallback)
KPlan KPlan::from_scalar(const Scalar& k, uint8_t w) {
    KPlan plan;
    plan.window_width = w;
    plan.k1 = k;  // Store original scalar for fallback scalar_mul
    plan.neg1 = false;
    plan.neg2 = false;
    return plan;
}
#endif

#if defined(SECP256K1_FAST_52BIT)
Point::Point() : x_(FieldElement52::zero()), y_(FieldElement52::one()), z_(FieldElement52::zero()), 
                 infinity_(true), is_generator_(false) {}

Point::Point(const FieldElement& x, const FieldElement& y, const FieldElement& z, bool infinity)
    : x_(FieldElement52::from_fe(x)), y_(FieldElement52::from_fe(y)), z_(FieldElement52::from_fe(z)), 
      infinity_(infinity), is_generator_(false) {}

// Zero-conversion FE52 constructor -- used by from_jac52 to avoid FE52->FE->FE52 round-trip
Point::Point(const FieldElement52& x, const FieldElement52& y, const FieldElement52& z, bool infinity, bool is_gen)
    : x_(x), y_(y), z_(z), 
      infinity_(infinity), is_generator_(is_gen) {}
#else
Point::Point() : x_(FieldElement::zero()), y_(FieldElement::one()), z_(FieldElement::zero()), 
                 infinity_(true), is_generator_(false) {}

Point::Point(const FieldElement& x, const FieldElement& y, const FieldElement& z, bool infinity)
    : x_(x), y_(y), z_(z), 
      infinity_(infinity), is_generator_(false) {}
#endif

Point Point::from_jacobian_coords(const FieldElement& x, const FieldElement& y, const FieldElement& z, bool infinity) {
    if (infinity || z == FieldElement::zero()) {
        return Point::infinity();
    }
    return Point(x, y, z, false);
}

#if defined(SECP256K1_FAST_52BIT)
Point Point::from_jacobian52(const FieldElement52& x, const FieldElement52& y, const FieldElement52& z, bool infinity) {
    // normalizes_to_zero() uses only ONE overflow-reduction pass and can
    // produce false negatives on high-magnitude limbs (e.g. result of
    // Shamir's trick where Z accumulated many additions).  is_zero() does
    // a full fe52_normalize_inline (TWO reduction passes + conditional
    // subtraction of p) so it is reliable at any magnitude.
    if (infinity || z.is_zero()) return Point::infinity();
    return Point(x, y, z, false, false);
}

Point Point::from_affine52(const FieldElement52& x, const FieldElement52& y) {
    return Point(x, y, FieldElement52::one(), false, false);
}
#endif

// Precomputed generator point G in affine coordinates (for fast mixed addition)
static const FieldElement kGeneratorX = FieldElement::from_bytes({
    0x79,0xBE,0x66,0x7E,0xF9,0xDC,0xBB,0xAC,
    0x55,0xA0,0x62,0x95,0xCE,0x87,0x0B,0x07,
    0x02,0x9B,0xFC,0xDB,0x2D,0xCE,0x28,0xD9,
    0x59,0xF2,0x81,0x5B,0x16,0xF8,0x17,0x98
});

static const FieldElement kGeneratorY = FieldElement::from_bytes({
    0x48,0x3A,0xDA,0x77,0x26,0xA3,0xC4,0x65,
    0x5D,0xA4,0xFB,0xFC,0x0E,0x11,0x08,0xA8,
    0xFD,0x17,0xB4,0x48,0xA6,0x85,0x54,0x19,
    0x9C,0x47,0xD0,0x8F,0xFB,0x10,0xD4,0xB8
});

// Precomputed -G (negative generator) in affine coordinates
// -G = (Gx, -Gy mod p) where p = FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
static const FieldElement kNegGeneratorY = FieldElement::from_bytes({
    0xB7,0xC5,0x25,0x88,0xD9,0x5C,0x3B,0x9A,
    0xA2,0x5B,0x04,0x03,0xF1,0xEE,0xF7,0x57,
    0x02,0xE8,0x4B,0xB7,0x59,0x7A,0xAB,0xE6,
    0x63,0xB8,0x2F,0x6F,0x04,0xEF,0x27,0x77
});

Point Point::generator() {
    Point g(kGeneratorX, kGeneratorY, fe_from_uint(1), false);
    g.is_generator_ = true;
    return g;
}

Point Point::infinity() {
    return Point(FieldElement::zero(), FieldElement::one(), FieldElement::zero(), true);
}

Point Point::from_affine(const FieldElement& x, const FieldElement& y) {
    return Point(x, y, fe_from_uint(1), false);
}

Point Point::from_hex(const std::string& x_hex, const std::string& y_hex) {
    FieldElement x = FieldElement::from_hex(x_hex);
    FieldElement y = FieldElement::from_hex(y_hex);
    return from_affine(x, y);
}

FieldElement Point::x() const {
    if (infinity_) {
        return FieldElement::zero();
    }
#if defined(SECP256K1_FAST_52BIT)
    FieldElement z_fe = z_.to_fe();  // fully normalizes
    { const auto& zL = z_fe.limbs();
      if (SECP256K1_UNLIKELY((zL[0] | zL[1] | zL[2] | zL[3]) == 0))
          return FieldElement::zero();
    }
    FieldElement z_inv = z_fe.inverse();
    FieldElement z_inv2 = z_inv;
    z_inv2.square_inplace();
    return x_.to_fe() * z_inv2;
#else
    FieldElement z_inv = z_.inverse();
    FieldElement z_inv2 = z_inv;
    z_inv2.square_inplace();
    return x_ * z_inv2;
#endif
}

FieldElement Point::y() const {
    if (infinity_) {
        return FieldElement::zero();
    }
#if defined(SECP256K1_FAST_52BIT)
    FieldElement z_fe = z_.to_fe();  // fully normalizes
    { const auto& zL = z_fe.limbs();
      if (SECP256K1_UNLIKELY((zL[0] | zL[1] | zL[2] | zL[3]) == 0))
          return FieldElement::zero();
    }
    FieldElement z_inv = z_fe.inverse();
    FieldElement z_inv3 = z_inv;
    z_inv3.square_inplace();
    z_inv3 *= z_inv;
    return y_.to_fe() * z_inv3;
#else
    FieldElement z_inv = z_.inverse();
    FieldElement z_inv3 = z_inv;
    z_inv3.square_inplace();
    z_inv3 *= z_inv;
    return y_ * z_inv3;
#endif
}

#if defined(SECP256K1_FAST_52BIT)
FieldElement Point::X() const noexcept { return x_.to_fe(); }
FieldElement Point::Y() const noexcept { return y_.to_fe(); }
FieldElement Point::z() const noexcept { return z_.to_fe(); }
FieldElement Point::x_raw() const noexcept { return x_.to_fe(); }
FieldElement Point::y_raw() const noexcept { return y_.to_fe(); }
FieldElement Point::z_raw() const noexcept { return z_.to_fe(); }
#endif

std::array<uint8_t, 16> Point::x_first_half() const {
    // Convert to affine once and reuse
    std::array<uint8_t, 32> full_x;
    x().to_bytes_into(full_x.data());
    
    std::array<uint8_t, 16> first_half;
    std::copy(full_x.begin(), full_x.begin() + 16, first_half.begin());
    return first_half;
}

std::array<uint8_t, 16> Point::x_second_half() const {
    // Convert to affine once and reuse
    std::array<uint8_t, 32> full_x;
    x().to_bytes_into(full_x.data());
    
    std::array<uint8_t, 16> second_half;
    std::copy(full_x.begin() + 16, full_x.end(), second_half.begin());
    return second_half;
}

Point Point::add(const Point& other) const {
#if defined(SECP256K1_FAST_52BIT)
    JacobianPoint52 p52{x_, y_, z_, infinity_};
    JacobianPoint52 q52{other.x_, other.y_, other.z_, other.infinity_};
    JacobianPoint52 r52 = jac52_add(p52, q52);
    Point result;
    result.x_ = r52.x; result.y_ = r52.y; result.z_ = r52.z;
    result.infinity_ = r52.infinity;
    result.is_generator_ = false;
    return result;
#else
    JacobianPoint p{x_, y_, z_, infinity_};
    JacobianPoint q{other.x_, other.y_, other.z_, other.infinity_};
    JacobianPoint r = jacobian_add(p, q);
    Point result = from_jacobian_coords(r.x, r.y, r.z, r.infinity);
    result.is_generator_ = false;
    return result;
#endif
}

Point Point::dbl() const {
#if defined(SECP256K1_FAST_52BIT)
    JacobianPoint52 p52{x_, y_, z_, infinity_};
    JacobianPoint52 r52 = jac52_double(p52);
    Point result;
    result.x_ = r52.x; result.y_ = r52.y; result.z_ = r52.z;
    result.infinity_ = r52.infinity;
    result.is_generator_ = false;
    return result;
#else
    JacobianPoint p{x_, y_, z_, infinity_};
    JacobianPoint r = jacobian_double(p);
    Point result = from_jacobian_coords(r.x, r.y, r.z, r.infinity);
    result.is_generator_ = false;
    return result;
#endif
}

Point Point::negate() const {
    if (infinity_) {
        return *this;  // Infinity is its own negation
    }
#if defined(SECP256K1_FAST_52BIT)
    Point result;
    result.x_ = x_;
    result.y_ = y_;
    result.y_.normalize_weak();
    result.y_.negate_assign(1);
    result.y_.normalize_weak();
    result.z_ = z_;
    result.infinity_ = false;
    result.is_generator_ = false;
    return result;
#else
    FieldElement neg_y = FieldElement::zero() - y_;
    Point result(x_, neg_y, z_, false);
    result.is_generator_ = false;
    return result;
#endif
}

Point Point::next() const {
#if defined(SECP256K1_FAST_52BIT)
    static const FieldElement52 kGenX52 = FieldElement52::from_fe(kGeneratorX);
    static const FieldElement52 kGenY52 = FieldElement52::from_fe(kGeneratorY);
    JacobianPoint52 p52{x_, y_, z_, infinity_};
    AffinePoint52 g52{kGenX52, kGenY52};
    jac52_add_mixed_inplace(p52, g52);
    Point result;
    result.x_ = p52.x; result.y_ = p52.y; result.z_ = p52.z;
    result.infinity_ = p52.infinity;
    result.is_generator_ = false;
    return result;
#else
    JacobianPoint p{x_, y_, z_, infinity_};
    AffinePoint g_affine{kGeneratorX, kGeneratorY};
    JacobianPoint r = jacobian_add_mixed(p, g_affine);
    Point result = from_jacobian_coords(r.x, r.y, r.z, r.infinity);
    result.is_generator_ = false;
    return result;
#endif
}

Point Point::prev() const {
#if defined(SECP256K1_FAST_52BIT)
    static const FieldElement52 kGenX52 = FieldElement52::from_fe(kGeneratorX);
    static const FieldElement52 kNegGenY52 = FieldElement52::from_fe(kNegGeneratorY);
    JacobianPoint52 p52{x_, y_, z_, infinity_};
    AffinePoint52 ng52{kGenX52, kNegGenY52};
    jac52_add_mixed_inplace(p52, ng52);
    Point result;
    result.x_ = p52.x; result.y_ = p52.y; result.z_ = p52.z;
    result.infinity_ = p52.infinity;
    result.is_generator_ = false;
    return result;
#else
    JacobianPoint p{x_, y_, z_, infinity_};
    AffinePoint neg_g_affine{kGeneratorX, kNegGeneratorY};
    JacobianPoint r = jacobian_add_mixed(p, neg_g_affine);
    Point result = from_jacobian_coords(r.x, r.y, r.z, r.infinity);
    result.is_generator_ = false;
    return result;
#endif
}

// Mutable in-place addition: *this += G (no allocation overhead)
void Point::next_inplace() {
    if (infinity_) {
        *this = Point::generator();
        return;
    }
#if defined(SECP256K1_FAST_52BIT)
    static const FieldElement52 kGenX52 = FieldElement52::from_fe(kGeneratorX);
    static const FieldElement52 kGenY52 = FieldElement52::from_fe(kGeneratorY);
    JacobianPoint52 p52{x_, y_, z_, infinity_};
    AffinePoint52 g52{kGenX52, kGenY52};
    jac52_add_mixed_inplace(p52, g52);
    x_ = p52.x; y_ = p52.y; z_ = p52.z;
    infinity_ = p52.infinity;
    is_generator_ = false;
#else
    // In-place: no return-value copy
    JacobianPoint self{x_, y_, z_, infinity_};
    AffinePoint g{kGeneratorX, kGeneratorY};
    jacobian_add_mixed_inplace(self, g);
    x_ = self.x; y_ = self.y; z_ = self.z;
    infinity_ = self.infinity;
    is_generator_ = false;
#endif
}

// Mutable in-place subtraction: *this -= G (no allocation overhead)
void Point::prev_inplace() {
    if (infinity_) {
        *this = Point::generator().negate();
        return;
    }
#if defined(SECP256K1_FAST_52BIT)
    static const FieldElement52 kGenX52 = FieldElement52::from_fe(kGeneratorX);
    static const FieldElement52 kNegGenY52 = FieldElement52::from_fe(kNegGeneratorY);
    JacobianPoint52 p52{x_, y_, z_, infinity_};
    AffinePoint52 ng52{kGenX52, kNegGenY52};
    jac52_add_mixed_inplace(p52, ng52);
    x_ = p52.x; y_ = p52.y; z_ = p52.z;
    infinity_ = p52.infinity;
    is_generator_ = false;
#else
    // In-place: no return-value copy
    JacobianPoint self{x_, y_, z_, infinity_};
    AffinePoint neg_g{kGeneratorX, kNegGeneratorY};
    jacobian_add_mixed_inplace(self, neg_g);
    x_ = self.x; y_ = self.y; z_ = self.z;
    infinity_ = self.infinity;
    is_generator_ = false;
#endif
}

// Mutable in-place addition: *this += other (no allocation overhead)
// Routes through 5x52 path on x64 for inlined field ops (zero call overhead)
void Point::add_inplace(const Point& other) {
#if defined(SECP256K1_FAST_52BIT)
    // Fast path: if other is affine (z = 1), use mixed addition
    // Raw limb check -- no normalization, z_ from affine construction is already canonical {1,0,0,0,0}
    if (!other.infinity_ && fe52_is_one_raw(other.z_)) {
        // Direct: members are already FE52
        JacobianPoint52 p52{x_, y_, z_, infinity_};
        AffinePoint52 q52{other.x_, other.y_};
        jac52_add_mixed_inplace(p52, q52);
        x_ = p52.x; y_ = p52.y; z_ = p52.z;
        infinity_ = p52.infinity;
        is_generator_ = false;
        return;
    }

    // General case: full Jacobian-Jacobian addition via 5x52 (in-place, zero copy)
    JacobianPoint52 p52{x_, y_, z_, infinity_};
    JacobianPoint52 q52{other.x_, other.y_, other.z_, other.infinity_};
    jac52_add_inplace(p52, q52);
    x_ = p52.x; y_ = p52.y; z_ = p52.z;
    infinity_ = p52.infinity;
    is_generator_ = false;
#else
    // Fast path: if other is affine (z = 1), use mixed addition
    // Raw limb check -- no temp allocation, z_ from affine construction is canonical {1,0,0,0}
    if (!other.infinity_ && fe_is_one_raw(other.z_)) {
        JacobianPoint p{x_, y_, z_, infinity_};
        AffinePoint q{other.x_, other.y_};
        jacobian_add_mixed_inplace(p, q);
        x_ = p.x; y_ = p.y; z_ = p.z;
        infinity_ = p.infinity;
        is_generator_ = false;
        return;
    }

    // General case: full Jacobian-Jacobian addition (in-place, no return copy)
    JacobianPoint p{x_, y_, z_, infinity_};
    JacobianPoint q{other.x_, other.y_, other.z_, other.infinity_};
    jacobian_add_inplace(p, q);
    x_ = p.x; y_ = p.y; z_ = p.z;
    infinity_ = p.infinity;
    is_generator_ = false;
#endif
}

// Mutable in-place subtraction: *this -= other (no allocation overhead)
void Point::sub_inplace(const Point& other) {
#if defined(SECP256K1_FAST_52BIT)
    // 5x52 path: negate other.y, then add in-place
    JacobianPoint52 p52{x_, y_, z_, infinity_};
    JacobianPoint52 q52{other.x_, other.y_, other.z_, other.infinity_};
    q52.y.normalize_weak();
    q52.y.negate_assign(1);
    q52.y.normalize_weak();
    jac52_add_inplace(p52, q52);
    x_ = p52.x; y_ = p52.y; z_ = p52.z;
    infinity_ = p52.infinity;
    is_generator_ = false;
#else
    // In-place: negate other.y, then add in-place (no return-value copy)
    JacobianPoint p{x_, y_, z_, infinity_};
    JacobianPoint q{other.x_, FieldElement::zero() - other.y_, other.z_, other.infinity_};
    jacobian_add_inplace(p, q);
    x_ = p.x; y_ = p.y; z_ = p.z;
    infinity_ = p.infinity;
    is_generator_ = false;
#endif
}

// Mutable in-place doubling: *this = 2*this (no allocation overhead)
void Point::dbl_inplace() {
#if defined(SECP256K1_PLATFORM_ESP32) || defined(__XTENSA__) || defined(SECP256K1_PLATFORM_STM32)
    // Optimized: 5S + 2M formula (saves 2S vs generic 7S + 1M)
    // Z3 = 2*Y*Z (1M) replaces (Y+Z)^2-Y^2-Z^2 (2S), operates on fields directly
    if (infinity_) return;

    FieldElement xx = x_; xx.square_inplace();           // 1S: X^2
    FieldElement yy = y_; yy.square_inplace();           // 1S: Y^2
    FieldElement yyyy = yy; yyyy.square_inplace();       // 1S: Y^4

    FieldElement s = x_ + yy;
    s.square_inplace();                                  // 1S: (X+Y^2)^2
    s -= xx + yyyy;
    s = s + s;                                           // S = 4*X*Y^2

    FieldElement m = xx + xx + xx;                        // M = 3*X^2

    // Compute Z3 BEFORE modifying x_, y_ (reads original values)
    FieldElement z3 = y_ * z_;                           // 1M: Y*Z
    z3 = z3 + z3;                                        // Z3 = 2*Y*Z

    // X3 = M^2 - 2*S
    x_ = m;
    x_.square_inplace();                                 // 1S: M^2
    x_ -= s + s;

    // Y3 = M*(S - X3) - 8*Y^4
    FieldElement yyyy8 = yyyy + yyyy;
    yyyy8 = yyyy8 + yyyy8;
    yyyy8 = yyyy8 + yyyy8;
    y_ = m * (s - x_) - yyyy8;                           // 1M

    z_ = z3;
    is_generator_ = false;
#elif defined(SECP256K1_FAST_52BIT)
    // 5x52 path: direct FE52 members, no conversion needed
    if (SECP256K1_UNLIKELY(infinity_)) return;
    JacobianPoint52 p52{x_, y_, z_, infinity_};
    jac52_double_inplace(p52);
    x_ = p52.x; y_ = p52.y; z_ = p52.z;
    is_generator_ = false;
#else
    // In-place: no return-value copy (eliminates 100B struct copy per call)
    JacobianPoint p{x_, y_, z_, infinity_};
    jacobian_double_inplace(p);
    x_ = p.x;
    y_ = p.y;
    z_ = p.z;
    infinity_ = p.infinity;
    is_generator_ = false;
#endif
}

// Mutable in-place negation: *this = -this (no allocation overhead)
void Point::negate_inplace() {
    // Negation in Jacobian: (X, Y, Z) -> (X, -Y, Z)
#if defined(SECP256K1_FAST_52BIT)
    y_.normalize_weak();
    y_.negate_assign(1);
    y_.normalize_weak();
#else
    y_ = FieldElement::zero() - y_;
#endif
}

// Explicit mixed-add with affine input: this += (ax, ay)
// Routes through 5x52 path on x64 for inlined field ops
void Point::add_mixed_inplace(const FieldElement& ax, const FieldElement& ay) {
#if defined(SECP256K1_FAST_52BIT)
    if (SECP256K1_UNLIKELY(infinity_)) {
        x_ = FieldElement52::from_fe(ax);
        y_ = FieldElement52::from_fe(ay);
        z_ = FieldElement52::one();
        infinity_ = false;
        is_generator_ = false;
        return;
    }
    JacobianPoint52 p52{x_, y_, z_, infinity_};
    AffinePoint52 q52{FieldElement52::from_fe(ax), FieldElement52::from_fe(ay)};
    jac52_add_mixed_inplace(p52, q52);
    x_ = p52.x; y_ = p52.y; z_ = p52.z;
    infinity_ = p52.infinity;
    is_generator_ = false;
#else
    if (SECP256K1_UNLIKELY(infinity_)) {
        // Convert affine to Jacobian: (x, y) -> (x, y, 1, false)
        x_ = ax;
        y_ = ay;
        z_ = FieldElement::one();
        infinity_ = false;
        is_generator_ = false;
        return;
    }

    // Core formula optimized for a=0 curve (inline to avoid struct copies)
    FieldElement z1z1 = z_;                     // Copy for in-place
    z1z1.square_inplace();                      // Z1^2 in-place! [1S]
    FieldElement u2 = ax * z1z1;                // U2 = X2*Z1^2 [1M]
    FieldElement s2 = ay * z_ * z1z1;           // S2 = Y2*Z1^3 [2M]
    
    if (SECP256K1_UNLIKELY(x_ == u2)) {
        if (y_ == s2) {
            dbl_inplace();
            return;
        }
        // Points are inverses: result is infinity
        x_ = FieldElement::zero();
        y_ = FieldElement::one();
        z_ = FieldElement::zero();
        infinity_ = true;
        is_generator_ = false;
        return;
    }

    FieldElement h = u2 - x_;                   // H = U2 - X1
    FieldElement hh = h;                        // Copy for in-place
    hh.square_inplace();                        // HH = H^2 in-place! [1S]
    FieldElement i = hh + hh + hh + hh;         // I = 4*HH (3 additions)
    FieldElement j = h * i;                     // J = H*I [1M]
    FieldElement r = (s2 - y_) + (s2 - y_);     // r = 2*(S2 - Y1)
    FieldElement v = x_ * i;                    // V = X1*I [1M]
    
    FieldElement x3 = r;                        // Copy for in-place
    x3.square_inplace();                        // r^2 in-place!
    x3 -= j + v + v;                            // X3 = r^2 - J - 2*V [1S]
    
    // Y3 = r*(V - X3) - 2*Y1*J -- optimized: *2 via addition
    FieldElement y1j = y_ * j;                  // [1M]
    FieldElement y3 = r * (v - x3) - (y1j + y1j); // [1M]
    
    // Z3 = (Z1+H)^2 - Z1^2 - HH -- in-place for performance
    FieldElement z3 = z_ + h;                   // Z1 + H
    z3.square_inplace();                        // (Z1+H)^2 in-place!
    z3 -= z1z1 + hh;                            // Z3 = (Z1+H)^2 - Z1^2 - HH [1S: total 7M + 4S]

    // Update members directly (no return copy!)
    x_ = x3;
    y_ = y3;
    z_ = z3;
    infinity_ = false;
    is_generator_ = false;
#endif
}

// Explicit mixed-sub with affine input: this -= (ax, ay) == this + (ax, -ay)
// OPTIMIZED: Inline implementation to avoid 6 FieldElement struct copies per call
void Point::sub_mixed_inplace(const FieldElement& ax, const FieldElement& ay) {
    // Negate Y coordinate: subtract is add with negated Y
    FieldElement neg_ay = FieldElement::zero() - ay;
    add_mixed_inplace(ax, neg_ay);
}

// Repeated mixed addition with a fixed affine point (ax, ay, z=1)
// Changed from CRITICAL_FUNCTION to HOT_FUNCTION - flatten breaks this!
SECP256K1_HOT_FUNCTION
void Point::add_affine_constant_inplace(const FieldElement& ax, const FieldElement& ay) {
#if defined(SECP256K1_FAST_52BIT)
    // Delegate to add_mixed_inplace which uses the FE52 path
    add_mixed_inplace(ax, ay);
#else
    if (infinity_) {
        x_ = ax;
        y_ = ay;
        z_ = FieldElement::one();
        infinity_ = false;
        is_generator_ = false;
        return;
    }

    FieldElement z1z1 = z_;          // z^2
    z1z1.square_inplace();
    FieldElement u2 = ax * z1z1;     // X2 * z^2
    FieldElement s2 = ay * z_ * z1z1; // Y2 * z^3

    if (x_ == u2) {
        if (y_ == s2) {
            dbl_inplace();
            return;
        } else {
            // Infinity
            *this = Point();
            return;
        }
    }

    FieldElement h = u2 - x_;
    FieldElement hh = h; hh.square_inplace(); // h^2
    FieldElement i = hh + hh + hh + hh;       // 4*HH
    FieldElement j = h * i;                   // H*I
    FieldElement r = (s2 - y_) + (s2 - y_);   // 2*(S2 - Y1)
    FieldElement v = x_ * i;                  // X1*I

    FieldElement x3 = r; x3.square_inplace(); // r^2
    x3 -= j + v + v;                          // r^2 - J - 2*V
    FieldElement y1j = y_ * j;                // Y1*J
    FieldElement y3 = r * (v - x3) - (y1j + y1j); // r*(V - X3) - 2*Y1*J
    FieldElement z3 = (z_ + h); z3.square_inplace(); z3 -= z1z1 + hh; // (Z1+H)^2 - Z1^2 - H^2

    x_ = x3; y_ = y3; z_ = z3; infinity_ = false; is_generator_ = false;
#endif
}

// ============================================================================
// Generator fixed-base multiplication (ESP32)
// Precomputes [1..15] x 2^(4i) x G in affine for i=0..31 (480 entries, 30KB)
// Turns k*G into ~60 mixed additions with NO doublings.
// One-time setup: ~50ms; steady-state: ~4ms per k*G
// ============================================================================
#if defined(SECP256K1_PLATFORM_ESP32) || defined(ESP_PLATFORM)
namespace {

struct GenAffine { FieldElement x, y; };

static GenAffine gen_fb_table[480];
static bool gen_fb_ready = false;

static void init_gen_fb_table() {
    if (gen_fb_ready) return;

    auto* z_orig = new FieldElement[480];
    auto* z_pfx  = new FieldElement[480];
    if (!z_orig || !z_pfx) { delete[] z_orig; delete[] z_pfx; return; }

    Point base = Point::generator();

    for (int w = 0; w < 32; w++) {
        // Compute [1..15]xbase using doubling chain: 7 dbl + 7 add
        Point pts[15];
        pts[0]  = base;
        pts[1]  = base;  pts[1].dbl_inplace();                  // 2B
        pts[2]  = pts[1]; pts[2].add_inplace(base);             // 3B
        pts[3]  = pts[1]; pts[3].dbl_inplace();                 // 4B
        pts[4]  = pts[3]; pts[4].add_inplace(base);             // 5B
        pts[5]  = pts[2]; pts[5].dbl_inplace();                 // 6B
        pts[6]  = pts[5]; pts[6].add_inplace(base);             // 7B
        pts[7]  = pts[3]; pts[7].dbl_inplace();                 // 8B
        pts[8]  = pts[7]; pts[8].add_inplace(base);             // 9B
        pts[9]  = pts[4]; pts[9].dbl_inplace();                 // 10B
        pts[10] = pts[9];  pts[10].add_inplace(base);           // 11B
        pts[11] = pts[5];  pts[11].dbl_inplace();               // 12B
        pts[12] = pts[11]; pts[12].add_inplace(base);           // 13B
        pts[13] = pts[6];  pts[13].dbl_inplace();               // 14B
        pts[14] = pts[13]; pts[14].add_inplace(base);           // 15B

        for (int d = 0; d < 15; d++) {
            int idx = w * 15 + d;
            gen_fb_table[idx].x = pts[d].x_raw();
            gen_fb_table[idx].y = pts[d].y_raw();
            z_orig[idx] = pts[d].z_raw();
        }

        if (w < 31) {
            for (int d = 0; d < 4; d++) base.dbl_inplace();
        }
    }

    // Batch-to-affine: single inversion for all 480 entries
    z_pfx[0] = z_orig[0];
    for (int i = 1; i < 480; i++) z_pfx[i] = z_pfx[i - 1] * z_orig[i];

    FieldElement inv = z_pfx[479].inverse();

    for (int i = 479; i > 0; i--) {
        FieldElement z_inv = inv * z_pfx[i - 1];
        inv = inv * z_orig[i];
        FieldElement zi2 = z_inv.square();
        FieldElement zi3 = zi2 * z_inv;
        gen_fb_table[i].x = gen_fb_table[i].x * zi2;
        gen_fb_table[i].y = gen_fb_table[i].y * zi3;
    }
    {
        FieldElement zi2 = inv.square();
        FieldElement zi3 = zi2 * inv;
        gen_fb_table[0].x = gen_fb_table[0].x * zi2;
        gen_fb_table[0].y = gen_fb_table[0].y * zi3;
    }

    delete[] z_orig;
    delete[] z_pfx;
    gen_fb_ready = true;
}

inline std::uint8_t get_nybble(const Scalar& s, int pos) {
    int limb = pos / 16;
    int shift = (pos % 16) * 4;
    return static_cast<std::uint8_t>((s.limbs()[limb] >> shift) & 0xFu);
}

static Point gen_fixed_mul(const Scalar& k) {
    if (!gen_fb_ready) init_gen_fb_table();

    auto decomp = glv_decompose(k);

    static const FieldElement beta =
        FieldElement::from_bytes(glv_constants::BETA);

    // Work directly with JacobianPoint to avoid Point wrapper overhead.
    // Eliminates 3 FieldElement copies per add (Point::from_affine + add_inplace copies).
    JacobianPoint jac = {FieldElement::zero(), FieldElement::one(), FieldElement::zero(), true};

    for (int w = 0; w < 32; w++) {
        std::uint8_t d1 = get_nybble(decomp.k1, w);
        if (d1 > 0) {
            const GenAffine& e = gen_fb_table[w * 15 + d1 - 1];
            AffinePoint q;
            q.x = e.x;
            q.y = decomp.k1_neg
                ? (FieldElement::zero() - e.y) : e.y;
            jacobian_add_mixed_inplace(jac, q);
        }

        std::uint8_t d2 = get_nybble(decomp.k2, w);
        if (d2 > 0) {
            const GenAffine& e = gen_fb_table[w * 15 + d2 - 1];
            AffinePoint q;
            q.x = e.x * beta;
            q.y = decomp.k2_neg
                ? (FieldElement::zero() - e.y) : e.y;
            jacobian_add_mixed_inplace(jac, q);
        }
    }

    return Point::from_jacobian_coords(jac.x, jac.y, jac.z, jac.infinity);
}

}  // anonymous namespace
#endif  // SECP256K1_PLATFORM_ESP32

Point Point::scalar_mul(const Scalar& scalar) const {
#if !defined(SECP256K1_PLATFORM_ESP32) && !defined(ESP_PLATFORM) && !defined(SECP256K1_PLATFORM_STM32) && !defined(__EMSCRIPTEN__)
    // WASM: precompute tables produce incorrect results under Emscripten's
    // __int128 emulation (field ops are fine, but the windowed accumulation
    // path diverges for some scalars).  Use the proven double-and-add
    // fallback instead -- correct and acceptable perf for WASM.
    if (is_generator_) {
        return scalar_mul_generator(scalar);
    }
#endif

#if defined(SECP256K1_PLATFORM_ESP32) || defined(ESP_PLATFORM) || defined(SECP256K1_PLATFORM_STM32)
    // ESP32 only: generator uses precomputed fixed-base table (~4ms vs ~12ms)
    // STM32 skips this (64KB SRAM too tight for 30KB table)
#if defined(SECP256K1_PLATFORM_ESP32) || defined(ESP_PLATFORM)
    if (is_generator_) {
        return gen_fixed_mul(scalar);
    }
#endif

    // ---------------------------------------------------------------
    // Embedded: GLV decomposition + Shamir's trick
    // Splits 256-bit scalar into two ~128-bit half-scalars and processes
    // both streams simultaneously, halving the number of doublings.
    //   k*P = sign1*|k1|*P + sign2*|k2|*phi(P)
    //   where k = k1 + k2*lambda (mod n), |k1|,|k2| ~= sqrtn
    // ---------------------------------------------------------------

    // Step 1: Decompose scalar
    GLVDecomposition decomp = glv_decompose(scalar);

    // Step 2: Handle k1 sign by negating base point before precomputation
    Point P_base = decomp.k1_neg ? this->negate() : *this;

    // Step 3: Compute wNAF for both half-scalars (stack-allocated)
    // w=5 for ~128-bit scalars: table_size=8, ~21 additions per stream
    constexpr unsigned glv_window = 5;
    constexpr int glv_table_size = (1 << (glv_window - 2));  // 8

    // Note: compute_wnaf_into always processes full 256-bit Scalar even
    // though GLV half-scalars are ~128 bits, so buffer must be 260.
    std::array<int32_t, 260> wnaf1_buf{}, wnaf2_buf{};
    std::size_t wnaf1_len = 0, wnaf2_len = 0;
    compute_wnaf_into(decomp.k1, glv_window,
                      wnaf1_buf.data(), wnaf1_buf.size(), wnaf1_len);
    compute_wnaf_into(decomp.k2, glv_window,
                      wnaf2_buf.data(), wnaf2_buf.size(), wnaf2_len);

    // Step 4: Precompute odd multiples [1, 3, 5, ..., 15] for P
    std::array<Point, glv_table_size> tbl_P, tbl_phiP;
    std::array<Point, glv_table_size> neg_tbl_P, neg_tbl_phiP;

    tbl_P[0] = P_base;
    Point dbl_P = P_base;
    dbl_P.dbl_inplace();
    for (std::size_t i = 1; i < glv_table_size; i++) {
        tbl_P[i] = tbl_P[i - 1];
        tbl_P[i].add_inplace(dbl_P);
    }

    // Pre-negate P table (eliminates copy+negate in hot loop)
    for (std::size_t i = 0; i < glv_table_size; i++) {
        neg_tbl_P[i] = tbl_P[i];
        neg_tbl_P[i].negate_inplace();
    }

    // Derive phi(P) table from P table using the endomorphism: phi(X:Y:Z) = (beta*X:Y:Z)
    // This costs only 8 field muls vs 7 additions + 1 doubling (~10x cheaper)
    // Sign adjustment: tbl_P has k1 sign baked in; flip to k2 sign if different
    bool flip_phi = (decomp.k1_neg != decomp.k2_neg);
    for (std::size_t i = 0; i < glv_table_size; i++) {
        tbl_phiP[i] = apply_endomorphism(tbl_P[i]);
        if (flip_phi) tbl_phiP[i].negate_inplace();
    }

    // Pre-negate phi(P) table
    for (std::size_t i = 0; i < glv_table_size; i++) {
        neg_tbl_phiP[i] = tbl_phiP[i];
        neg_tbl_phiP[i].negate_inplace();
    }

    // Step 5: Shamir's trick -- one doubling per iteration, two lookups
    Point result = Point::infinity();
    std::size_t max_len = (wnaf1_len > wnaf2_len) ? wnaf1_len : wnaf2_len;

    for (int i = static_cast<int>(max_len) - 1; i >= 0; --i) {
        result.dbl_inplace();

        // k1 contribution (no copy needed: pre-negated table)
        if (static_cast<std::size_t>(i) < wnaf1_len) {
            int32_t d = wnaf1_buf[static_cast<std::size_t>(i)];
            if (d > 0) {
                result.add_inplace(tbl_P[(d - 1) / 2]);
            } else if (d < 0) {
                result.add_inplace(neg_tbl_P[(-d - 1) / 2]);
            }
        }

        // k2 contribution (no copy needed: pre-negated table)
        if (static_cast<std::size_t>(i) < wnaf2_len) {
            int32_t d = wnaf2_buf[static_cast<std::size_t>(i)];
            if (d > 0) {
                result.add_inplace(tbl_phiP[(d - 1) / 2]);
            } else if (d < 0) {
                result.add_inplace(neg_tbl_phiP[(-d - 1) / 2]);
            }
        }
    }

    return result;

#else
    // -----------------------------------------------------------------
    // Non-embedded: GLV decomposition + Shamir's trick
    // -----------------------------------------------------------------
    // Splits 256-bit scalar into two ~128-bit half-scalars:
    //   k*P = sign1*|k1|*P + sign2*|k2|*phi(P)
    // Uses 5x52 FieldElement52 when available (~2.5x faster).
    // Without FE52 (WASM, MSVC): uses regular 4x64 FieldElement with
    // GLV + Shamir (same algorithm proven on ESP32/STM32).
    // -----------------------------------------------------------------

#ifdef SECP256K1_FAST_52BIT
    return scalar_mul_glv52(*this, scalar);
#else
    // -----------------------------------------------------------------
    // Non-FE52 fallback: simple right-to-left binary double-and-add.
    // No GLV, no wNAF -- just iterate over 256 scalar bits.
    // Correctness-first: uses only dbl_inplace + add_inplace.
    // Performance: ~256 doublings + ~128 additions (acceptable for WASM).
    // -----------------------------------------------------------------
    auto scalar_bytes = scalar.to_bytes();  // big-endian: [0]=MSB
    Point result = Point::infinity();
    Point base = *this;    // 2^0 * P initially

    // Scan bits from LSB (byte 31, bit 0) to MSB (byte 0, bit 7)
    for (int byte_idx = 31; byte_idx >= 0; --byte_idx) {
        uint8_t byte_val = scalar_bytes[static_cast<std::size_t>(byte_idx)];
        for (int bit = 0; bit < 8; ++bit) {
            if ((byte_val >> bit) & 1) {
                result.add_inplace(base);
            }
            base.dbl_inplace();
        }
    }
    return result;
#endif // SECP256K1_FAST_52BIT
#endif // ESP32/STM32
}

// Step 1: Use existing GLV decomposition from K*G implementation
// Q * k = Q * k1 + phi(Q) * k2
#if !defined(SECP256K1_PLATFORM_ESP32) && !defined(ESP_PLATFORM) && !defined(SECP256K1_PLATFORM_STM32)
Point Point::scalar_mul_precomputed_k(const Scalar& k) const {
    // Use the proven GLV decomposition from scalar_mul_generator
    auto decomp = split_scalar_glv(k);
    
    // Delegate to predecomposed version
    return scalar_mul_predecomposed(decomp.k1, decomp.k2, decomp.neg1, decomp.neg2);
}
#else
// ESP32: fall back to basic scalar_mul (no GLV optimization)
Point Point::scalar_mul_precomputed_k(const Scalar& k) const {
    return this->scalar_mul(k);
}
#endif

// Step 2: Runtime-only version - no decomposition overhead
// K decomposition is done once at startup, not per operation
Point Point::scalar_mul_predecomposed(const Scalar& k1, const Scalar& k2, 
                                       bool neg1, bool neg2) const {
#if !defined(SECP256K1_PLATFORM_ESP32) && !defined(ESP_PLATFORM) && !defined(SECP256K1_PLATFORM_STM32)
    // If both signs are positive, use fast Shamir's trick
    // Otherwise, fall back to separate computation (sign handling is complex)
    if (!neg1 && !neg2) {
        // Fast path: K = k1 + k2*lambda (both positive)
        constexpr unsigned window_width = 4;
        auto wnaf1 = compute_wnaf(k1, window_width);
        auto wnaf2 = compute_wnaf(k2, window_width);
        return scalar_mul_precomputed_wnaf(wnaf1, wnaf2, false, false);
    }
#endif

    // Restore original variant: compute both, negate individually, then add
    Point phi_Q = apply_endomorphism(*this);
    Point term1 = this->scalar_mul(k1);
    Point term2 = phi_Q.scalar_mul(k2);
    if (neg1) term1.negate_inplace();
    if (neg2) term2.negate_inplace();
    term1.add_inplace(term2);
    return term1;
}

// Step 3: Shamir's trick with precomputed wNAF
// All K-related work (decomposition, wNAF) is done once
// Runtime only: phi(Q), tables, interleaved double-and-add
Point Point::scalar_mul_precomputed_wnaf(const std::vector<int32_t>& wnaf1,
                                          const std::vector<int32_t>& wnaf2,
                                          bool neg1, bool neg2) const {
    // Convert this point to Jacobian for internal operations
#if defined(SECP256K1_FAST_52BIT)
    JacobianPoint p = {x_.to_fe(), y_.to_fe(), z_.to_fe(), infinity_};
#else
    JacobianPoint p = {x_, y_, z_, infinity_};
#endif
    
    // Compute phi(Q) - endomorphism (1 field multiplication)
    Point phi_Q = apply_endomorphism(*this);
#if defined(SECP256K1_FAST_52BIT)
    JacobianPoint phi_p = {phi_Q.X(), phi_Q.Y(), phi_Q.z(), phi_Q.infinity_};
#else
    JacobianPoint phi_p = {phi_Q.x_, phi_Q.y_, phi_Q.z_, phi_Q.infinity_};
#endif
    
    // Determine table size from wNAF digits
    // For window w: digits are in range [-2^w+1, 2^w-1] odd
    // Table stores positive odd multiples: [1, 3, 5, ..., 2^w-1]
    // Table size = 2^(w-1)
    int max_digit = 0;
    for (auto d : wnaf1) {
        int abs_d = (d < 0) ? -d : d;
        if (abs_d > max_digit) max_digit = abs_d;
    }
    for (auto d : wnaf2) {
        int abs_d = (d < 0) ? -d : d;
        if (abs_d > max_digit) max_digit = abs_d;
    }
    
    // Table size needed to handle max_digit
    // For w=4: max_digit=15, table_size=8 (stores 1,3,5,...,15)
    // For w=5: max_digit=31, table_size=16 (stores 1,3,5,...,31)
    int table_size = (max_digit + 1) / 2;

    // Build precomputation tables for odd multiples in Jacobian (no heap alloc)
    constexpr int kMaxWindowBits = 7;
    constexpr int kMaxTableSize = (1 << (kMaxWindowBits - 1)); // 64
    (void)sizeof(char[kMaxTableSize * 2]); // kMaxAll=128, used for static_assert sizing
    if (table_size > kMaxTableSize) {
        // Safety fallback: clamp to max supported table; digits beyond won't occur for w<=7
        table_size = kMaxTableSize;
    }

    std::array<JacobianPoint, kMaxTableSize> table_Q_jac{};
    std::array<JacobianPoint, kMaxTableSize> table_phi_Q_jac{};
    // Pre-negated tables (eliminates copy+negate in hot loop)
    std::array<JacobianPoint, kMaxTableSize> neg_table_Q_jac{};
    std::array<JacobianPoint, kMaxTableSize> neg_table_phi_Q_jac{};

    // Generate tables: [P, 3P, 5P, 7P, ...]
    table_Q_jac[0] = p;            // 1*Q
    table_phi_Q_jac[0] = phi_p;    // 1*phi(Q)

    JacobianPoint double_Q = jacobian_double(p);        // 2*Q
    JacobianPoint double_phi_Q = jacobian_double(phi_p); // 2*phi(Q)

    for (int i = 1; i < table_size; i++) {
        table_Q_jac[static_cast<std::size_t>(i)] = jacobian_add(table_Q_jac[static_cast<std::size_t>(i-1)], double_Q);
        table_phi_Q_jac[static_cast<std::size_t>(i)] = jacobian_add(table_phi_Q_jac[static_cast<std::size_t>(i-1)], double_phi_Q);
    }

    // Build pre-negated tables for sign-flipped lookups (no per-iteration copy)
    // Sign handling: neg1/neg2 flags flip global sign for k1/k2; pre-compute both orientations
    for (int i = 0; i < table_size; i++) {
        // For k1: positive lookup uses table, negative uses neg_table
        // If neg1, swap them (XOR logic: neg1 flips which table to use)
        neg_table_Q_jac[static_cast<std::size_t>(i)] = table_Q_jac[static_cast<std::size_t>(i)];
        neg_table_Q_jac[static_cast<std::size_t>(i)].y = FieldElement::zero() - neg_table_Q_jac[static_cast<std::size_t>(i)].y;
        neg_table_phi_Q_jac[static_cast<std::size_t>(i)] = table_phi_Q_jac[static_cast<std::size_t>(i)];
        neg_table_phi_Q_jac[static_cast<std::size_t>(i)].y = FieldElement::zero() - neg_table_phi_Q_jac[static_cast<std::size_t>(i)].y;
    }

    // If global sign flags are set, swap positive/negative tables
    if (neg1) { std::swap(table_Q_jac, neg_table_Q_jac); }
    if (neg2) { std::swap(table_phi_Q_jac, neg_table_phi_Q_jac); }

    // Shamir's trick: process both wNAF streams simultaneously (inplace ops)
    JacobianPoint result = {FieldElement::zero(), FieldElement::one(), FieldElement::zero(), true};
    
    size_t max_len = std::max(wnaf1.size(), wnaf2.size());
    
    for (int i = static_cast<int>(max_len) - 1; i >= 0; --i) {
        jacobian_double_inplace(result);
        
        // Add phi(Q) * k2 contribution (no copy: pre-negated tables)
        if (i < static_cast<int>(wnaf2.size())) {
            int32_t digit2 = wnaf2[static_cast<std::size_t>(i)];
            if (digit2 > 0) {
                int idx = (digit2 - 1) / 2;
                jacobian_add_inplace(result, table_phi_Q_jac[static_cast<std::size_t>(idx)]);
            } else if (digit2 < 0) {
                int idx = (-digit2 - 1) / 2;
                jacobian_add_inplace(result, neg_table_phi_Q_jac[static_cast<std::size_t>(idx)]);
            }
        }

        // Add Q * k1 contribution (no copy: pre-negated tables)
        if (i < static_cast<int>(wnaf1.size())) {
            int32_t digit1 = wnaf1[static_cast<std::size_t>(i)];
            if (digit1 > 0) {
                int idx = (digit1 - 1) / 2;
                jacobian_add_inplace(result, table_Q_jac[static_cast<std::size_t>(idx)]);
            } else if (digit1 < 0) {
                int idx = (-digit1 - 1) / 2;
                jacobian_add_inplace(result, neg_table_Q_jac[static_cast<std::size_t>(idx)]);
            }
        }
    }

    return Point(result.x, result.y, result.z, result.infinity);
}

// Fixed K x Variable Q: Optimal performance for repeated K with different Q
// All K-dependent work is cached in KPlan (GLV decomposition + wNAF computation)
// Runtime: phi(Q), tables, Shamir's trick (if signs allow) or separate computation
Point Point::scalar_mul_with_plan(const KPlan& plan) const {
#if defined(SECP256K1_PLATFORM_ESP32) || defined(ESP_PLATFORM) || defined(SECP256K1_PLATFORM_STM32)
    // Embedded: fallback to regular scalar_mul using stored k1
    return scalar_mul(plan.k1);
#else
    // Fast path: Interleaved Shamir using precomputed wNAF digits from plan
    return scalar_mul_precomputed_wnaf(plan.wnaf1, plan.wnaf2, plan.neg1, plan.neg2);
#endif
}

std::array<std::uint8_t, 33> Point::to_compressed() const {
    std::array<std::uint8_t, 33> out{};
    if (infinity_) {
        out.fill(0);
        return out;
    }
    // Compute affine coordinates with a single inversion
#if defined(SECP256K1_FAST_52BIT)
    FieldElement z_fe = z_.to_fe();  // fully normalizes
    // Defensive: if Z reduces to zero treat as infinity
    { const auto& zL = z_fe.limbs();
      if (SECP256K1_UNLIKELY((zL[0] | zL[1] | zL[2] | zL[3]) == 0)) {
          out.fill(0); return out;
      }
    }
    FieldElement z_inv = z_fe.inverse();
    FieldElement z_inv2 = z_inv;
    z_inv2.square_inplace();
    FieldElement x_aff = x_.to_fe() * z_inv2;
    FieldElement y_aff = y_.to_fe() * z_inv2 * z_inv;
#else
    FieldElement z_inv = z_.inverse();
    FieldElement z_inv2 = z_inv;
    z_inv2.square_inplace();
    FieldElement x_aff = x_ * z_inv2;
    FieldElement y_aff = y_ * z_inv2 * z_inv;
#endif
    auto x_bytes = x_aff.to_bytes();
    auto y_bytes = y_aff.to_bytes();
    out[0] = (y_bytes[31] & 1U) ? 0x03 : 0x02;
    std::copy(x_bytes.begin(), x_bytes.end(), out.begin() + 1);
    return out;
}

std::array<std::uint8_t, 65> Point::to_uncompressed() const {
    std::array<std::uint8_t, 65> out{};
    if (infinity_) {
        out.fill(0);
        return out;
    }
    // Compute affine coordinates with a single inversion
#if defined(SECP256K1_FAST_52BIT)
    FieldElement z_fe = z_.to_fe();  // fully normalizes
    // Defensive: if Z reduces to zero treat as infinity
    { const auto& zL = z_fe.limbs();
      if (SECP256K1_UNLIKELY((zL[0] | zL[1] | zL[2] | zL[3]) == 0)) {
          out.fill(0); return out;
      }
    }
    FieldElement z_inv = z_fe.inverse();
    FieldElement z_inv2 = z_inv;
    z_inv2.square_inplace();
    FieldElement x_aff = x_.to_fe() * z_inv2;
    FieldElement y_aff = y_.to_fe() * z_inv2 * z_inv;
#else
    FieldElement z_inv = z_.inverse();
    FieldElement z_inv2 = z_inv;
    z_inv2.square_inplace();
    FieldElement x_aff = x_ * z_inv2;
    FieldElement y_aff = y_ * z_inv2 * z_inv;
#endif
    auto x_bytes = x_aff.to_bytes();
    auto y_bytes = y_aff.to_bytes();
    out[0] = 0x04;
    std::copy(x_bytes.begin(), x_bytes.end(), out.begin() + 1);
    std::copy(y_bytes.begin(), y_bytes.end(), out.begin() + 33);
    return out;
}

// -- Y-parity check (single inversion) ---------------------------------------
bool Point::has_even_y() const {
    if (infinity_) return false;
#if defined(SECP256K1_FAST_52BIT)
    FieldElement z_fe = z_.to_fe();  // fully normalizes
    { const auto& zL = z_fe.limbs();
      if (SECP256K1_UNLIKELY((zL[0] | zL[1] | zL[2] | zL[3]) == 0)) return false;
    }
    FieldElement z_inv = z_fe.inverse();
    FieldElement z_inv2 = z_inv;
    z_inv2.square_inplace();
    FieldElement y_aff = y_.to_fe() * z_inv2 * z_inv;
#else
    FieldElement z_inv = z_.inverse();
    FieldElement z_inv2 = z_inv;
    z_inv2.square_inplace();
    FieldElement y_aff = y_ * z_inv2 * z_inv;
#endif
    auto y_bytes = y_aff.to_bytes();
    return (y_bytes[31] & 1) == 0;
}

// -- Combined x-bytes + Y-parity (single inversion) --------------------------
std::pair<std::array<uint8_t, 32>, bool> Point::x_bytes_and_parity() const {
    if (infinity_) return {std::array<uint8_t,32>{}, false};
#if defined(SECP256K1_FAST_52BIT)
    FieldElement z_fe = z_.to_fe();  // fully normalizes
    { const auto& zL = z_fe.limbs();
      if (SECP256K1_UNLIKELY((zL[0] | zL[1] | zL[2] | zL[3]) == 0))
          return {std::array<uint8_t,32>{}, false};
    }
    FieldElement z_inv = z_fe.inverse();
    FieldElement z_inv2 = z_inv;
    z_inv2.square_inplace();
    FieldElement x_aff = x_.to_fe() * z_inv2;
    FieldElement y_aff = y_.to_fe() * z_inv2 * z_inv;
#else
    FieldElement z_inv = z_.inverse();
    FieldElement z_inv2 = z_inv;
    z_inv2.square_inplace();
    FieldElement x_aff = x_ * z_inv2;
    FieldElement y_aff = y_ * z_inv2 * z_inv;
#endif
    auto y_bytes = y_aff.to_bytes();
    return {x_aff.to_bytes(), (y_bytes[31] & 1) != 0};
}

// -- 128-bit split Shamir: a*G + b*P -----------------------------------------
// Computes a*G + b*P using a ~128-bit doubling chain with 4 wNAF streams.
// a is split arithmetically: a = a_lo + a_hi*2^128  (no GLV for G-stream)
// b is GLV-decomposed: b = b1 + b2*lambda
// Then: a_lo*G + a_hi*H + b1*P + b2*psi(P) where H = 2^128*G
// G/H affine tables are cached statically (computed once, w=15 -> 8192 entries).
#if defined(SECP256K1_FAST_52BIT)
SECP256K1_NOINLINE
Point Point::dual_scalar_mul_gen_point(const Scalar& a, const Scalar& b, const Point& P) {
    // -- 128-bit arithmetic split of a --------------------------------
    const auto& a_limbs = a.limbs();
    Scalar a_lo = Scalar::from_limbs({a_limbs[0], a_limbs[1], 0, 0});
    Scalar a_hi = Scalar::from_limbs({a_limbs[2], a_limbs[3], 0, 0});

    // -- GLV decompose b only -----------------------------------------
    GLVDecomposition decomp_b = glv_decompose(b);

    // -- Window widths: w=15 for G (precomputed), w=5 for P (per-call) ---
    constexpr unsigned WINDOW_G = 15;    // -> 2^13 = 8192 entries per G/H table
    constexpr unsigned WINDOW_P = 5;     // -> 2^3 = 8 entries per P/psiP table
    constexpr int G_TABLE_SIZE = (1 << (WINDOW_G - 2));  // 8192
    constexpr int P_TABLE_SIZE = (1 << (WINDOW_P - 2));  // 8

    // -- Static generator tables (computed once, 8192 entries per base) --
    // Two bases: G (generator) and H = 2^128*G
    // Uses effective-affine technique for efficient table building.
    // Pre-negated tables avoid per-digit negate+normalize_weak in hot loop.
    struct GenTables {
        AffinePoint52 tbl_G[G_TABLE_SIZE];       // [G, 3G, 5G, ..., 16383G]
        AffinePoint52 tbl_H[G_TABLE_SIZE];       // [H, 3H, 5H, ..., 16383H]
        AffinePoint52 neg_tbl_G[G_TABLE_SIZE];   // negated Y for negative wNAF digits
        AffinePoint52 neg_tbl_H[G_TABLE_SIZE];   // negated Y for negative wNAF digits
    };
    static const GenTables* const gen_tables = []() -> const GenTables* {
        auto* t = new GenTables;

        // Helper: build odd-multiple table for base point B using effective-affine
        auto build_table = [](const JacobianPoint52& B,
                              AffinePoint52* out, int count) {
            // d = 2*B, work on isomorphic curve where d is affine
            JacobianPoint52 d = jac52_double(B);
            FieldElement52 C  = d.z;
            FieldElement52 C2 = C.square();
            FieldElement52 C3 = C2 * C;
            AffinePoint52 d_aff = {d.x, d.y};

            // iso[0] = phi(B) = (B.x*C^2, B.y*C^3, B.z) on iso curve
            auto* iso = new JacobianPoint52[static_cast<std::size_t>(count)];
            iso[0] = {B.x * C2, B.y * C3, B.z, false};
            for (std::size_t i = 1; i < static_cast<std::size_t>(count); i++) {
                iso[i] = iso[i - 1];
                jac52_add_mixed_inplace(iso[i], d_aff);
            }

            // Batch-invert effective Z = Z_iso * C
            auto* eff_z = new FieldElement52[static_cast<std::size_t>(count)];
            for (std::size_t i = 0; i < static_cast<std::size_t>(count); i++) {
                eff_z[i] = iso[i].z * C;
            }
            auto* prods = new FieldElement52[static_cast<std::size_t>(count)];
            prods[0] = eff_z[0];
            for (std::size_t i = 1; i < static_cast<std::size_t>(count); i++) {
                prods[i] = prods[i - 1] * eff_z[i];
            }
            FieldElement52 inv = prods[count - 1].inverse_safegcd();
            auto* zs = new FieldElement52[static_cast<std::size_t>(count)];
            for (std::size_t i = static_cast<std::size_t>(count) - 1; i > 0; --i) {
                zs[i] = prods[i - 1] * inv;
                inv = inv * eff_z[i];
            }
            zs[0] = inv;

            for (std::size_t i = 0; i < static_cast<std::size_t>(count); i++) {
                FieldElement52 zinv2 = zs[i].square();
                FieldElement52 zinv3 = zinv2 * zs[i];
                out[i].x = iso[i].x * zinv2;
                out[i].y = iso[i].y * zinv3;
            }

            delete[] zs;
            delete[] prods;
            delete[] eff_z;
            delete[] iso;
        };

        // Build tbl_G: odd multiples of G
        Point G = Point::generator();
        JacobianPoint52 G52 = to_jac52(G);
        build_table(G52, t->tbl_G, G_TABLE_SIZE);

        // Build tbl_H: odd multiples of H = 2^128*G
        JacobianPoint52 H52 = G52;
        for (std::size_t i = 0; i < 128; i++) {
            jac52_double_inplace(H52);
        }
        build_table(H52, t->tbl_H, G_TABLE_SIZE);

        // Pre-negate G/H tables: avoid per-digit negate+normalize_weak in hot loop
        for (std::size_t i = 0; i < static_cast<std::size_t>(G_TABLE_SIZE); i++) {
            t->neg_tbl_G[i].x = t->tbl_G[i].x;
            t->neg_tbl_G[i].y = t->tbl_G[i].y.negate(1);
            t->neg_tbl_G[i].y.normalize_weak();
            t->neg_tbl_H[i].x = t->tbl_H[i].x;
            t->neg_tbl_H[i].y = t->tbl_H[i].y.negate(1);
            t->neg_tbl_H[i].y.normalize_weak();
        }

        return t;
    }();

    // -- Precompute P tables (per-call, window=5, 8 entries) ---------
    // Uses effective-affine technique (same as libsecp256k1/CT path):
    //   d = 2*P (Jacobian), C = d.Z, iso curve: Y'^2 = X'^3 + C^6*7
    //   On iso curve, d is affine: (d.X, d.Y), so table adds are
    //   mixed Jac+Affine (7M+4S) instead of full Jac+Jac (12M+5S).
    //   Saves ~33M + 6S = ~1.4us per verify.
    JacobianPoint52 P52 = decomp_b.k1_neg
        ? to_jac52(P.negate())
        : to_jac52(P);

    std::array<AffinePoint52, P_TABLE_SIZE> tbl_P;
    std::array<AffinePoint52, P_TABLE_SIZE> tbl_phiP;
    std::array<AffinePoint52, P_TABLE_SIZE> neg_tbl_P;
    std::array<AffinePoint52, P_TABLE_SIZE> neg_tbl_phiP;

    // Build table via effective-affine on isomorphic curve
    {
        // d = 2*P (Jacobian)
        JacobianPoint52 d = jac52_double(P52);
        FieldElement52 C  = d.z;              // C = d.Z
        FieldElement52 C2 = C.square();       // C^2
        FieldElement52 C3 = C2 * C;           // C^3

        // d as affine on iso curve: phi(d) = (d.X, d.Y)
        // (Z = C on iso curve cancels in the isomorphism)
        AffinePoint52 d_aff = {d.x, d.y};

        // Transform P onto iso curve: phi(P) = (P.X*C^2, P.Y*C^3, P.Z)
        std::array<JacobianPoint52, P_TABLE_SIZE> iso;
        iso[0] = {P52.x * C2, P52.y * C3, P52.z, false};

        // Build rest using mixed adds on iso curve (7M+4S each, not 12M+5S)
        for (std::size_t i = 1; i < static_cast<std::size_t>(P_TABLE_SIZE); i++) {
            iso[i] = iso[i - 1];
            jac52_add_mixed_inplace(iso[i], d_aff);
        }

        // Batch-invert effective Z: true Z on secp256k1 = Z_iso * C
        std::array<FieldElement52, P_TABLE_SIZE> eff_z;
        for (std::size_t i = 0; i < static_cast<std::size_t>(P_TABLE_SIZE); i++) {
            eff_z[i] = iso[i].z * C;
        }

        std::array<FieldElement52, P_TABLE_SIZE> prods;
        prods[0] = eff_z[0];
        for (std::size_t i = 1; i < static_cast<std::size_t>(P_TABLE_SIZE); i++) {
            prods[i] = prods[i - 1] * eff_z[i];
        }
        FieldElement52 inv = prods[P_TABLE_SIZE - 1].inverse_safegcd();
        std::array<FieldElement52, P_TABLE_SIZE> zs;
        for (std::size_t i = static_cast<std::size_t>(P_TABLE_SIZE) - 1; i > 0; --i) {
            zs[i] = prods[i - 1] * inv;
            inv = inv * eff_z[i];
        }
        zs[0] = inv;

        // Convert from iso to secp256k1 affine: x = X*(Z*C)^-^2, y = Y*(Z*C)^-^3
        for (std::size_t i = 0; i < static_cast<std::size_t>(P_TABLE_SIZE); i++) {
            FieldElement52 zinv2 = zs[i].square();
            FieldElement52 zinv3 = zinv2 * zs[i];
            tbl_P[i].x = iso[i].x * zinv2;
            tbl_P[i].y = iso[i].y * zinv3;
        }
    }

    // psi(P) table
    static const FieldElement52 beta52 = FieldElement52::from_fe(
        FieldElement::from_bytes(glv_constants::BETA));

    const bool flip_phi_b = (decomp_b.k1_neg != decomp_b.k2_neg);
    for (std::size_t i = 0; i < static_cast<std::size_t>(P_TABLE_SIZE); i++) {
        tbl_phiP[i].x = tbl_P[i].x * beta52;
        if (flip_phi_b) {
            tbl_phiP[i].y = tbl_P[i].y.negate(1);
            tbl_phiP[i].y.normalize_weak();
        } else {
            tbl_phiP[i].y = tbl_P[i].y;
        }
    }

    // Pre-negate P tables
    for (std::size_t i = 0; i < static_cast<std::size_t>(P_TABLE_SIZE); i++) {
        neg_tbl_P[i].x = tbl_P[i].x;
        neg_tbl_P[i].y = tbl_P[i].y.negate(1);
        neg_tbl_P[i].y.normalize_weak();
        neg_tbl_phiP[i].x = tbl_phiP[i].x;
        neg_tbl_phiP[i].y = tbl_phiP[i].y.negate(1);
        neg_tbl_phiP[i].y.normalize_weak();
    }

    // -- Compute wNAF for all 4 half-scalars -------------------------
    std::array<int32_t, 260> wnaf_a_lo{}, wnaf_a_hi{}, wnaf_b1{}, wnaf_b2{};
    std::size_t len_a_lo = 0, len_a_hi = 0, len_b1 = 0, len_b2 = 0;

    compute_wnaf_into(a_lo,  WINDOW_G, wnaf_a_lo.data(), wnaf_a_lo.size(), len_a_lo);
    compute_wnaf_into(a_hi,  WINDOW_G, wnaf_a_hi.data(), wnaf_a_hi.size(), len_a_hi);
    compute_wnaf_into(decomp_b.k1, WINDOW_P, wnaf_b1.data(), wnaf_b1.size(), len_b1);
    compute_wnaf_into(decomp_b.k2, WINDOW_P, wnaf_b2.data(), wnaf_b2.size(), len_b2);

    // Trim trailing zeros
    while (len_a_lo > 0 && wnaf_a_lo[len_a_lo - 1] == 0) --len_a_lo;
    while (len_a_hi > 0 && wnaf_a_hi[len_a_hi - 1] == 0) --len_a_hi;
    while (len_b1 > 0 && wnaf_b1[len_b1 - 1] == 0) --len_b1;
    while (len_b2 > 0 && wnaf_b2[len_b2 - 1] == 0) --len_b2;

    // -- 4-stream Shamir interleaved scan -----------------------------
    JacobianPoint52 result52 = {
        FieldElement52::zero(), FieldElement52::one(),
        FieldElement52::zero(), true
    };

    std::size_t max_len = len_a_lo;
    if (len_a_hi > max_len) max_len = len_a_hi;
    if (len_b1 > max_len) max_len = len_b1;
    if (len_b2 > max_len) max_len = len_b2;

    for (int i = static_cast<int>(max_len) - 1; i >= 0; --i) {
        jac52_double_inplace(result52);

        // Stream 1: a_lo * G (pre-negated table for negative digits)
        {
            int32_t d = wnaf_a_lo[static_cast<std::size_t>(i)];
            if (d > 0) {
                jac52_add_mixed_inplace(result52, gen_tables->tbl_G[static_cast<std::size_t>((d - 1) >> 1)]);
            } else if (d < 0) {
                jac52_add_mixed_inplace(result52, gen_tables->neg_tbl_G[static_cast<std::size_t>((-d - 1) >> 1)]);
            }
        }

        // Stream 2: a_hi * H (pre-negated table for negative digits)
        {
            int32_t d = wnaf_a_hi[static_cast<std::size_t>(i)];
            if (d > 0) {
                jac52_add_mixed_inplace(result52, gen_tables->tbl_H[static_cast<std::size_t>((d - 1) >> 1)]);
            } else if (d < 0) {
                jac52_add_mixed_inplace(result52, gen_tables->neg_tbl_H[static_cast<std::size_t>((-d - 1) >> 1)]);
            }
        }

        // Stream 3: b1 * P
        {
            int32_t d = wnaf_b1[static_cast<std::size_t>(i)];
            if (d > 0) {
                jac52_add_mixed_inplace(result52, tbl_P[static_cast<std::size_t>((d - 1) >> 1)]);
            } else if (d < 0) {
                jac52_add_mixed_inplace(result52, neg_tbl_P[static_cast<std::size_t>((-d - 1) >> 1)]);
            }
        }

        // Stream 4: b2 * psi(P)
        {
            int32_t d = wnaf_b2[static_cast<std::size_t>(i)];
            if (d > 0) {
                jac52_add_mixed_inplace(result52, tbl_phiP[static_cast<std::size_t>((d - 1) >> 1)]);
            } else if (d < 0) {
                jac52_add_mixed_inplace(result52, neg_tbl_phiP[static_cast<std::size_t>((-d - 1) >> 1)]);
            }
        }
    }

    return from_jac52(result52);
}
#elif defined(SECP256K1_PLATFORM_ESP32) || defined(ESP_PLATFORM) || defined(SECP256K1_PLATFORM_STM32)
// -- ESP32/Embedded: 4-stream GLV Strauss (4x64 field) --------------------
// Combines a*G + b*P into a single doubling chain with 4 wNAF streams,
// halving the doublings compared to two separate scalar_mul calls.
// Expected speedup: ~25-30% on ECDSA verify.
Point Point::dual_scalar_mul_gen_point(const Scalar& a, const Scalar& b, const Point& P) {

    // -- GLV decompose both scalars ------------------------------------
    GLVDecomposition decomp_a = glv_decompose(a);
    GLVDecomposition decomp_b = glv_decompose(b);

    // -- Build wNAF w=5 for all 4 half-scalars ------------------------
    constexpr unsigned WINDOW = 5;
    constexpr int TABLE_SIZE = (1 << (WINDOW - 2));  // 8

    std::array<int32_t, 260> wnaf_a1{}, wnaf_a2{}, wnaf_b1{}, wnaf_b2{};
    std::size_t len_a1 = 0, len_a2 = 0, len_b1 = 0, len_b2 = 0;
    compute_wnaf_into(decomp_a.k1, WINDOW, wnaf_a1.data(), wnaf_a1.size(), len_a1);
    compute_wnaf_into(decomp_a.k2, WINDOW, wnaf_a2.data(), wnaf_a2.size(), len_a2);
    compute_wnaf_into(decomp_b.k1, WINDOW, wnaf_b1.data(), wnaf_b1.size(), len_b1);
    compute_wnaf_into(decomp_b.k2, WINDOW, wnaf_b2.data(), wnaf_b2.size(), len_b2);

    // -- Precompute G tables (static, computed once) ------------------
    // Odd multiples [1,3,5,...,15]xG and [1,3,...,15]xphi(G) in affine
    struct DualGenTables {
        AffinePoint tbl_G[TABLE_SIZE];
        AffinePoint tbl_phiG[TABLE_SIZE];
        AffinePoint neg_tbl_G[TABLE_SIZE];
        AffinePoint neg_tbl_phiG[TABLE_SIZE];
    };

    // C++11 magic static: thread-safe one-time initialization.
    // Replaces bare check-then-allocate pattern that had a data race.
    static const DualGenTables& gen4 = *[]() {
        auto* t = new DualGenTables;

        // Build [1,3,5,...,15]xG in Jacobian, then batch-invert to affine
        Point G = Point::generator();
        Point pts_j[TABLE_SIZE];
        pts_j[0] = G;
        Point dbl_G = G; dbl_G.dbl_inplace();
        for (int i = 1; i < TABLE_SIZE; i++) {
            pts_j[i] = pts_j[i - 1];
            pts_j[i].add_inplace(dbl_G);
        }

        // Batch invert Z coordinates to get affine
        FieldElement z_orig[TABLE_SIZE], z_pfx[TABLE_SIZE];
        for (int i = 0; i < TABLE_SIZE; i++) z_orig[i] = pts_j[i].z_raw();
        z_pfx[0] = z_orig[0];
        for (int i = 1; i < TABLE_SIZE; i++) z_pfx[i] = z_pfx[i - 1] * z_orig[i];

        FieldElement inv = z_pfx[TABLE_SIZE - 1].inverse();
        FieldElement z_inv[TABLE_SIZE];
        for (int i = TABLE_SIZE - 1; i > 0; --i) {
            z_inv[i] = inv * z_pfx[i - 1];
            inv = inv * z_orig[i];
        }
        z_inv[0] = inv;

        static const FieldElement beta = FieldElement::from_bytes(glv_constants::BETA);

        for (int i = 0; i < TABLE_SIZE; i++) {
            FieldElement zi2 = z_inv[i].square();
            FieldElement zi3 = zi2 * z_inv[i];
            FieldElement ax = pts_j[i].x_raw() * zi2;
            FieldElement ay = pts_j[i].y_raw() * zi3;

            t->tbl_G[i] = {ax, ay};
            t->neg_tbl_G[i] = {ax, FieldElement::zero() - ay};

            // phi(G): x -> beta*x, y -> y (same parity)
            FieldElement phix = ax * beta;
            t->tbl_phiG[i] = {phix, ay};
            t->neg_tbl_phiG[i] = {phix, FieldElement::zero() - ay};
        }

        return t;
    }();

    // -- Precompute P tables (per-call: 8 odd multiples + negated + endomorphism) --
    Point P_base = decomp_b.k1_neg ? P.negate() : P;
    std::array<AffinePoint, TABLE_SIZE> tbl_P, tbl_phiP, neg_tbl_P, neg_tbl_phiP;

    {
        Point pts_j[TABLE_SIZE];
        pts_j[0] = P_base;
        Point dbl_P = P_base; dbl_P.dbl_inplace();
        for (int i = 1; i < TABLE_SIZE; i++) {
            pts_j[i] = pts_j[i - 1];
            pts_j[i].add_inplace(dbl_P);
        }

        FieldElement z_orig[TABLE_SIZE], z_pfx[TABLE_SIZE];
        for (int i = 0; i < TABLE_SIZE; i++) z_orig[i] = pts_j[i].z_raw();
        z_pfx[0] = z_orig[0];
        for (int i = 1; i < TABLE_SIZE; i++) z_pfx[i] = z_pfx[i - 1] * z_orig[i];

        FieldElement inv = z_pfx[TABLE_SIZE - 1].inverse();
        FieldElement z_inv[TABLE_SIZE];
        for (int i = TABLE_SIZE - 1; i > 0; --i) {
            z_inv[i] = inv * z_pfx[i - 1];
            inv = inv * z_orig[i];
        }
        z_inv[0] = inv;

        static const FieldElement beta = FieldElement::from_bytes(glv_constants::BETA);
        bool flip_phi = (decomp_b.k1_neg != decomp_b.k2_neg);

        for (int i = 0; i < TABLE_SIZE; i++) {
            FieldElement zi2 = z_inv[i].square();
            FieldElement zi3 = zi2 * z_inv[i];
            FieldElement px = pts_j[i].x_raw() * zi2;
            FieldElement py = pts_j[i].y_raw() * zi3;

            tbl_P[i] = {px, py};
            neg_tbl_P[i] = {px, FieldElement::zero() - py};

            FieldElement phix = px * beta;
            FieldElement phiy = flip_phi ? (FieldElement::zero() - py) : py;
            tbl_phiP[i] = {phix, phiy};
            neg_tbl_phiP[i] = {phix, FieldElement::zero() - phiy};
        }
    }

    // -- Handle G sign: if a decomposed with k1_neg, use neg tables --
    const AffinePoint* g_pos  = decomp_a.k1_neg ? gen4.neg_tbl_G : gen4.tbl_G;
    const AffinePoint* g_neg  = decomp_a.k1_neg ? gen4.tbl_G : gen4.neg_tbl_G;
    // phi(G) sign: flip if k1_neg != k2_neg for a
    bool flip_a = (decomp_a.k1_neg != decomp_a.k2_neg);
    const AffinePoint* pg_pos = flip_a ? gen4.neg_tbl_phiG : gen4.tbl_phiG;
    const AffinePoint* pg_neg = flip_a ? gen4.tbl_phiG : gen4.neg_tbl_phiG;

    // -- 4-stream Shamir interleaved scan (JacobianPoint direct -- no Point wrapper) --
    std::size_t max_len = len_a1;
    if (len_a2 > max_len) max_len = len_a2;
    if (len_b1 > max_len) max_len = len_b1;
    if (len_b2 > max_len) max_len = len_b2;

    JacobianPoint jac = {FieldElement::zero(), FieldElement::one(), FieldElement::zero(), true};

    for (int i = static_cast<int>(max_len) - 1; i >= 0; --i) {
        jacobian_double_inplace(jac);

        // Stream 1: a1 * G (affine tables -> mixed add)
        {
            int32_t d = wnaf_a1[static_cast<std::size_t>(i)];
            if (d > 0) jacobian_add_mixed_inplace(jac, g_pos[(d-1)>>1]);
            else if (d < 0) jacobian_add_mixed_inplace(jac, g_neg[(-d-1)>>1]);
        }

        // Stream 2: a2 * phi(G) (affine tables -> mixed add)
        {
            int32_t d = wnaf_a2[static_cast<std::size_t>(i)];
            if (d > 0) jacobian_add_mixed_inplace(jac, pg_pos[(d-1)>>1]);
            else if (d < 0) jacobian_add_mixed_inplace(jac, pg_neg[(-d-1)>>1]);
        }

        // Stream 3: b1 * P (affine tables -> mixed add)
        {
            int32_t d = wnaf_b1[static_cast<std::size_t>(i)];
            if (d > 0) jacobian_add_mixed_inplace(jac, tbl_P[(d-1)>>1]);
            else if (d < 0) jacobian_add_mixed_inplace(jac, neg_tbl_P[(-d-1)>>1]);
        }

        // Stream 4: b2 * phi(P) (affine tables -> mixed add)
        {
            int32_t d = wnaf_b2[static_cast<std::size_t>(i)];
            if (d > 0) jacobian_add_mixed_inplace(jac, tbl_phiP[(d-1)>>1]);
            else if (d < 0) jacobian_add_mixed_inplace(jac, neg_tbl_phiP[(-d-1)>>1]);
        }
    }

    return Point(jac.x, jac.y, jac.z, jac.infinity);
}
#else
// Non-52bit / non-embedded fallback: separate multiplications
Point Point::dual_scalar_mul_gen_point(const Scalar& a, const Scalar& b, const Point& P) {
    auto aG = Point::generator().scalar_mul(a);
    aG.add_inplace(P.scalar_mul(b));
    return aG;
}
#endif

} // namespace secp256k1::fast
