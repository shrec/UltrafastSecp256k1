#include "secp256k1/point.hpp"
#if !defined(SECP256K1_PLATFORM_ESP32) && !defined(ESP_PLATFORM) && !defined(SECP256K1_PLATFORM_STM32)
#include "secp256k1/precompute.hpp"
#endif
#include "secp256k1/glv.hpp"
#if defined(__SIZEOF_INT128__) && !defined(__EMSCRIPTEN__)
#include "secp256k1/field_52.hpp"
#endif
// Inline 4x64 ADCX/ADOX field operations for hot-loop point ops
#if (defined(__x86_64__) || defined(_M_X64)) && defined(__ADX__) && defined(__BMI2__)
#include "secp256k1/field_4x64_inline.hpp"
#endif

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>

namespace secp256k1::fast {
namespace {

// ESP32/STM32 local wNAF helpers (when precompute.hpp not included)
#if defined(SECP256K1_PLATFORM_ESP32) || defined(ESP_PLATFORM) || defined(SECP256K1_PLATFORM_STM32)
// Simple wNAF computation for ESP32 (inline, no heap allocation)
// -- Optimized wNAF computation (word-at-a-time bit extraction) ---------------
// Port of libsecp256k1's secp256k1_ecmult_wnaf approach:
//   - Reads scalar limbs directly (no to_bytes serialization roundtrip)
//   - Extracts W bits at once via word-level access (no multi-word shift/sub)
//   - Skips zero-bit positions for free (continue loop)
// For W=15 on a 128-bit scalar: ~9 non-zero digits out of ~129 positions.
// Old code: 128+ iterations of 4-limb shift + multi-word arithmetic per bit.
// New code: ~129 iterations of 1 bit-test + ~9 word-extractions total.
// Saves ~800-1200ns per verify (4 wNAF computations).
static void compute_wnaf_into(const Scalar& scalar, unsigned window_width,
                               int32_t* out, std::size_t out_capacity,
                               std::size_t& out_len) {
    // Read scalar limbs directly -- 4x64-bit little-endian.
    // No need for to_bytes() -> manual big-to-little endian conversion.
    const auto& sl = scalar.limbs();

    const int w = static_cast<int>(window_width);
    const int len = static_cast<int>(out_capacity);

    // Extract `count` bits starting at bit position `pos` from 4x64 LE limbs.
    // count must be in [1, 31].  Handles cross-limb boundary reads.
    // Positions beyond 255 return 0 (scalar is 256 bits).
    auto get_bits = [&](int pos, int count) -> std::uint32_t {
        int const limb_idx = pos >> 6;          // pos / 64
        int const bit_off  = pos & 63;          // pos % 64
        std::uint64_t val = 0;
        if (limb_idx < 4) {
            val = sl[static_cast<std::size_t>(limb_idx)] >> bit_off;
            if (bit_off + count > 64 && limb_idx + 1 < 4) {
                val |= sl[static_cast<std::size_t>(limb_idx + 1)] << (64 - bit_off);
            }
        }
        return static_cast<std::uint32_t>(val) & ((1u << count) - 1);
    };

    // Zero-fill output array (only the used portion)
    std::memset(out, 0, out_capacity * sizeof(int32_t));

    int carry = 0;
    int last_set_bit = -1;
    int bit = 0;

    while (bit < len) {
        // Fast check: is current bit == carry?  If so, skip (output stays 0).
        std::uint32_t const b = get_bits(bit, 1);
        if (b == static_cast<std::uint32_t>(carry)) {
            ++bit;
            continue;
        }

        // Non-zero digit: extract W bits at once.
        int now = w;
        if (now > len - bit) now = len - bit;

        int word = static_cast<int>(get_bits(bit, now)) + carry;
        carry = word >> (w - 1);
        word -= carry << w;

        out[bit] = static_cast<int32_t>(word);
        last_set_bit = bit;
        bit += now;   // skip ahead by window width (all these positions are 0)
    }

    out_len = (last_set_bit >= 0) ? static_cast<std::size_t>(last_set_bit + 1) : 0;
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
    FieldElement const D = temp + temp;  // *2 via addition (faster than mul)
    
    // E = 3*A
    FieldElement E = A + A;
    E = E + A;  // *3 via two additions (faster than mul)
    
    // F = E^2 (in-place)
    FieldElement F = E;             // Copy for in-place
    F.square_inplace();             // E^2 in-place!
    
    // X' = F - 2*D
    FieldElement const two_D = D + D;
    FieldElement const x3 = F - two_D;
    
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
    FieldElement const u2 = q.x * z1z1;               // U2 = X2*Z1^2 [1M]
    FieldElement const s2 = q.y * p.z * z1z1;         // S2 = Y2*Z1^3 [2M]
    
    if (SECP256K1_UNLIKELY(p.x == u2)) {
        if (p.y == s2) {
            return jacobian_double(p);
        }
        return {FieldElement::zero(), FieldElement::one(), FieldElement::zero(), true};
    }

    FieldElement const h = u2 - p.x;                  // H = U2 - X1
    FieldElement hh = h;                        // Copy for in-place
    hh.square_inplace();                        // HH = H^2 in-place! [1S]
    FieldElement const i = hh + hh + hh + hh;         // I = 4*HH (3 additions)
    FieldElement const j = h * i;                     // J = H*I [1M]
    FieldElement const r = (s2 - p.y) + (s2 - p.y);   // r = 2*(S2 - Y1)
    FieldElement const v = p.x * i;                   // V = X1*I [1M]
    
    FieldElement x3 = r;                        // Copy for in-place
    x3.square_inplace();                        // r^2 in-place!
    x3 -= j + v + v;                            // X3 = r^2 - J - 2*V [1S]
    
    // Y3 = r*(V - X3) - 2*Y1*J -- optimized: *2 via addition
    FieldElement const y1j = p.y * j;                 // [1M]
    FieldElement const y3 = r * (v - x3) - (y1j + y1j); // [1M]
    
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
    FieldElement const D = temp + temp;

    FieldElement E = A + A; E = E + A;
    FieldElement F = E; F.square_inplace();

    FieldElement const two_D = D + D;
    FieldElement const x3 = F - two_D;

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
    FieldElement const u2 = q.x * z1z1;
    FieldElement const s2 = q.y * p.z * z1z1;

    if (SECP256K1_UNLIKELY(p.x == u2)) {
        if (p.y == s2) {
            jacobian_double_inplace(p);
            return;
        }
        p = {FieldElement::zero(), FieldElement::one(), FieldElement::zero(), true};
        return;
    }

    FieldElement const h = u2 - p.x;
    FieldElement hh = h; hh.square_inplace();
    FieldElement const i = hh + hh + hh + hh;
    FieldElement const j = h * i;
    FieldElement const r = (s2 - p.y) + (s2 - p.y);
    FieldElement const v = p.x * i;

    FieldElement x3 = r; x3.square_inplace();
    x3 -= j + v + v;

    FieldElement const y1j = p.y * j;
    FieldElement const y3 = r * (v - x3) - (y1j + y1j);

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
    FieldElement const u1 = p.x * z2z2;
    FieldElement const u2 = q.x * z1z1;
    FieldElement const s1 = p.y * q.z * z2z2;
    FieldElement const s2 = q.y * p.z * z1z1;

    if (u1 == u2) {
        if (s1 == s2) { jacobian_double_inplace(p); return; }
        p = {FieldElement::zero(), FieldElement::one(), FieldElement::zero(), true};
        return;
    }

    FieldElement const h = u2 - u1;
    FieldElement i = h + h; i.square_inplace();
    FieldElement const j = h * i;
    FieldElement const r = (s2 - s1) + (s2 - s1);
    FieldElement const v = u1 * i;

    FieldElement x3 = r; x3.square_inplace();
    x3 -= j + v + v;
    FieldElement const s1j = s1 * j;
    FieldElement const y3 = r * (v - x3) - (s1j + s1j);
    FieldElement temp_z = p.z + q.z; temp_z.square_inplace();
    FieldElement const z3 = (temp_z - z1z1 - z2z2) * h;

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
    FieldElement const u1 = p.x * z2z2;
    FieldElement const u2 = q.x * z1z1;
    FieldElement const s1 = p.y * q.z * z2z2;
    FieldElement const s2 = q.y * p.z * z1z1;

    if (u1 == u2) {
        if (s1 == s2) {
            return jacobian_double(p);
        }
        return {FieldElement::zero(), FieldElement::one(), FieldElement::zero(), true};
    }

    FieldElement const h = u2 - u1;
    FieldElement i = h + h;                     // 2*h
    i.square_inplace();                         // (2*h)^2 in-place!
    FieldElement const j = h * i;
    FieldElement const r = (s2 - s1) + (s2 - s1);
    FieldElement const v = u1 * i;

    FieldElement x3 = r;                        // Copy for in-place
    x3.square_inplace();                        // r^2 in-place!
    x3 -= j + v + v;                           // x3 = r^2 - j - 2*v
    FieldElement const s1j = s1 * j;
    FieldElement const y3 = r * (v - x3) - (s1j + s1j);
    FieldElement temp_z = p.z + q.z;           // z1 + z2
    temp_z.square_inplace();                   // (z1 + z2)^2 in-place!
    FieldElement const z3 = (temp_z - z1z1 - z2z2) * h;

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

// -- Compact Affine Point (4x64 = 64 bytes = 1 cache line) -------------------
// For precomputed G/H tables: stores fully normalized affine coordinates in
// 4x64-bit limb format.  64 bytes = exactly 1 cache line, halving the cache
// footprint per entry compared to AffinePoint52 (80 bytes = 2 cache lines).
// Conversion to AffinePoint52 on lookup costs ~2ns (bit-shift only, no mul).
struct alignas(64) AffinePointCompact {
    std::uint64_t x[4];
    std::uint64_t y[4];

    // Convert to AffinePoint52 for computation (4x64 -> 5x52 bit-slice)
    inline AffinePoint52 to_affine52() const noexcept {
        return {
            FieldElement52::from_4x64_limbs(x),
            FieldElement52::from_4x64_limbs(y)
        };
    }

    // Construct from AffinePoint52 (5x52 -> 4x64 via full normalize + pack)
    static inline AffinePointCompact from_affine52(const AffinePoint52& p) noexcept {
        FieldElement const fx = p.x.to_fe();
        FieldElement const fy = p.y.to_fe();
        AffinePointCompact c;
        auto const& lx = fx.limbs();
        auto const& ly = fy.limbs();
        c.x[0] = lx[0]; c.x[1] = lx[1]; c.x[2] = lx[2]; c.x[3] = lx[3];
        c.y[0] = ly[0]; c.y[1] = ly[1]; c.y[2] = ly[2]; c.y[3] = ly[3];
        return c;
    }
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
// Formula: libsecp256k1 gej_double (a=0 specialization)
// L = (3/2)*X^2, S = Y^2, T = -X*S, X3 = L^2+2T, Y3 = -(L*(X3+T)+S^2), Z3 = Y*Z
// Cost: 3M + 4S + 8 cheap ops (half, mul_int, add, negate)
// Saves ~11 cheap ops vs dbl-2009-l (2M+5S+19 cheap) per doubling.
SECP256K1_HOT_FUNCTION SECP256K1_NOINLINE
static JacobianPoint52 jac52_double(const JacobianPoint52& p) noexcept {
    if (SECP256K1_UNLIKELY(p.infinity)) {
        return {FieldElement52::zero(), FieldElement52::one(), FieldElement52::zero(), true};
    }
    // Z3 = Y1 * Z1 (1M)
    FieldElement52 const z3 = p.y * p.z;    // mag 1

    // S = Y1^2 (1S)
    FieldElement52 s = p.y.square();         // mag 1

    // L = (3/2)*X1^2 (1S + mul_int + half)
    FieldElement52 l = p.x.square();         // mag 1
    l.mul_int_assign(3);                     // mag 3
    l.half_assign();                         // mag 2 (parity correct: 3*n[0]&1 == n[0]&1)

    // T = -S * X1 (1N + 1M)
    FieldElement52 t = s.negate(1);          // mag 2
    t.mul_assign(p.x);                       // mag 1

    // X3 = L^2 + 2T (1S + 2A)
    FieldElement52 x3 = l.square();          // mag 1
    x3.add_assign(t);                        // mag 2
    x3.add_assign(t);                        // mag 3

    // S' = S^2 (1S) -- S was Y^2, so S' = Y^4
    s.square_inplace();                      // mag 1

    // T' = T + X3
    t.add_assign(x3);                        // mag 4

    // Y3 = -(L*(X3+T) + S^2) (1M + 1A + 1N)
    FieldElement52 y3 = t * l;               // mag 1
    y3.add_assign(s);                        // mag 2
    y3.negate_assign(2);                     // mag 3

    return {x3, y3, z3, false};
}

// -- Dead 4x64 point-ops block removed (lines 525-737 in previous version) --
// These functions (jac_double_4x64_inplace, jac_add_mixed_4x64_inplace) were
// scaffolding that is now superseded by the FE52 path.
// SECP256K1_HYBRID_4X64_ACTIVE remains defined for the FE52->4x64 inverse/sqrt
// boundary optimization in field_52_impl.hpp.
#if 0  // Dead code -- previously gated on SECP256K1_HYBRID_4X64_ACTIVE

// NOTE: These functions are currently unused -- they are scaffolding for an
// upcoming 4x64-only scalar_mul path. Suppress -Wunused-function.
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic ignored "-Wrestrict"
#endif
#endif

[[maybe_unused]] __attribute__((noinline))
static void jac_double_4x64_inplace(
    std::uint64_t* __restrict__ X,
    std::uint64_t* __restrict__ Y,
    std::uint64_t* __restrict__ Z) noexcept
{
    using namespace secp256k1::fast::fe4x64;

    alignas(32) std::uint64_t z3[4], s[4], l[4], t[4];

    // Z3 = Y * Z (1M)
    mul(z3, Y, Z);

    // S = Y^2 (1S)
    sqr(s, Y);

    // L = (3/2) * X^2 (1S + mul_int(3) + half)
    sqr(l, X);
    mul_int(l, l, 3);
    half(l, l);

    // T = -S * X (negate + 1M)
    negate(t, s);
    mul(t, t, X);

    // X3 = L^2 + 2T (1S + 2 add)
    sqr(X, l);
    add(X, X, t);
    add(X, X, t);

    // S' = S^2 (1S)
    sqr(s, s);

    // T' = T + X3 (add)
    add(t, t, X);

    // Y3 = -(L*(X3+T) + S^2) (1M + add + negate)
    mul(Y, t, l);
    add(Y, Y, s);
    negate(Y, Y);

    // Write Z3
    copy(Z, z3);
}

// -- 4x64 Mixed Addition: Jacobian + Affine -> Jacobian (in-place) ---------
// Formula: madd-2007-bl (a=0 specialization)
// Cost: 7M + 4S + ~12A (additions near-free with branchless mod-p)
// Qx, Qy are raw uint64_t[4] affine coordinates (e.g. AffinePointCompact.x/y).
[[maybe_unused]] __attribute__((noinline))
static void jac_add_mixed_4x64_inplace(
    std::uint64_t* __restrict__ X,
    std::uint64_t* __restrict__ Y,
    std::uint64_t* __restrict__ Z,
    bool& infinity,
    const std::uint64_t* __restrict__ Qx,
    const std::uint64_t* __restrict__ Qy) noexcept
{
    using namespace secp256k1::fast::fe4x64;

    if (SECP256K1_UNLIKELY(infinity)) {
        copy(X, Qx); copy(Y, Qy);
        Z[0] = 1; Z[1] = 0; Z[2] = 0; Z[3] = 0;
        infinity = false;
        return;
    }

    alignas(32) std::uint64_t z1z1[4], u2[4], s2[4], h[4], hh[4], i_reg[4], j[4];
    alignas(32) std::uint64_t r_val[4], v[4], x3[4], y3[4], z3[4], t1[4];

    // Z1Z1 = Z^2 (1S)
    sqr(z1z1, Z);
    // U2 = Qx * Z1Z1 (1M)
    mul(u2, Qx, z1z1);
    // S2 = Qy * Z * Z1Z1 (2M)
    mul(t1, Z, z1z1);
    mul(s2, Qy, t1);
    // H = U2 - X1
    sub(h, u2, X);
    // HH = H^2 (1S)
    sqr(hh, h);
    // I = 4*HH (2 add)
    add(i_reg, hh, hh);
    add(i_reg, i_reg, i_reg);
    // J = H * I (1M)
    mul(j, h, i_reg);
    // r = 2*(S2 - Y) (sub + add)
    sub(r_val, s2, Y);
    add(r_val, r_val, r_val);
    // V = X * I (1M)
    mul(v, X, i_reg);
    // X3 = r^2 - J - 2V (1S + subs)
    sqr(x3, r_val);
    sub(x3, x3, j);
    sub(x3, x3, v);
    sub(x3, x3, v);
    // Y3 = r*(V-X3) - 2*Y*J (2M + sub)
    sub(t1, v, x3);
    mul(y3, r_val, t1);
    mul(t1, Y, j);
    add(t1, t1, t1);
    sub(y3, y3, t1);
    // Z3 = (Z+H)^2 - Z1Z1 - HH (1S + add + subs)
    add(z3, Z, h);
    sqr(z3, z3);
    sub(z3, z3, z1z1);
    sub(z3, z3, hh);

    copy(X, x3);
    copy(Y, y3);
    copy(Z, z3);
    infinity = false;
}

// -- 4x64 Add-Zinv: Jacobian + Affine-in-Z-frame -> Jacobian (in-place) ---
// Like mixed addition but the affine point lives in a different Z-frame.
// bzinv = inverse of the shared Z, used to scale the affine coordinates.
// Cost: 9M + 3S + ~8A
[[maybe_unused]] __attribute__((noinline))
static void jac_add_zinv_4x64_inplace(
    std::uint64_t* __restrict__ X,
    std::uint64_t* __restrict__ Y,
    std::uint64_t* __restrict__ Z,
    bool& infinity,
    const std::uint64_t* __restrict__ Qx,
    const std::uint64_t* __restrict__ Qy,
    const std::uint64_t* __restrict__ bzinv) noexcept
{
    using namespace secp256k1::fast::fe4x64;

    if (SECP256K1_UNLIKELY(infinity)) {
        alignas(32) std::uint64_t zi2[4], zi3[4];
        sqr(zi2, bzinv);
        mul(zi3, zi2, bzinv);
        mul(X, Qx, zi2);
        mul(Y, Qy, zi3);
        Z[0] = 1; Z[1] = 0; Z[2] = 0; Z[3] = 0;
        infinity = false;
        return;
    }

    alignas(32) std::uint64_t az[4], az2[4], u2[4], s2[4], h[4];
    alignas(32) std::uint64_t hh[4], h3[4], v[4], r_val[4], x3[4], y3[4], z3[4], t1[4];

    // az = Z * bzinv (1M)
    mul(az, Z, bzinv);
    // az2 = az^2 (1S)
    sqr(az2, az);
    // U2 = Qx * az2 (1M)
    mul(u2, Qx, az2);
    // S2 = Qy * az^3 (2M: Qy*az2, then *az)
    mul(s2, Qy, az2);
    mul(s2, s2, az);
    // H = U2 - X
    sub(h, u2, X);
    // Z3 = Z * H (1M, uses ORIGINAL Z -- not az)
    mul(z3, Z, h);
    // HH = H^2 (1S)
    sqr(hh, h);
    // H3 = H * HH = H^3 (1M)
    mul(h3, h, hh);
    // V = X * HH (1M)
    mul(v, X, hh);
    // r = S2 - Y
    sub(r_val, s2, Y);
    // X3 = r^2 - H3 - 2V (1S + subs)
    sqr(x3, r_val);
    sub(x3, x3, h3);
    sub(x3, x3, v);
    sub(x3, x3, v);
    // Y3 = r*(V-X3) - Y*H3 (2M + sub)
    sub(t1, v, x3);
    mul(y3, r_val, t1);
    mul(t1, Y, h3);
    sub(y3, y3, t1);

    copy(X, x3);
    copy(Y, y3);
    copy(Z, z3);
    infinity = false;
}

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

#endif // Dead 4x64 point ops (was SECP256K1_HYBRID_4X64_ACTIVE)

// -- In-Place Point Doubling (5x52) ----------------------------------------
// Same formula as jac52_double but overwrites the input point in-place.
// Eliminates the 128-byte return value copy on every call.
// Formula: libsecp256k1 gej_double (3M+4S+8 cheap, a=0)
// L = (3/2)*X^2, S = Y^2, T = -X*S, X3 = L^2+2T, Y3 = -(L*(X3+T)+S^2), Z3 = Y*Z
SECP256K1_HOT_FUNCTION __attribute__((always_inline))
static inline void jac52_double_inplace(JacobianPoint52& p) noexcept {
    // Branchless infinity propagation (like libsecp's secp256k1_gej_double):
    // On secp256k1 (a=0), 2Q = infinity iff Q = infinity (no order-2 points).
    // If p.infinity == true, the formula runs on garbage X/Y/Z but p.infinity
    // stays true, so consumers correctly ignore the garbage coordinates.

    // Z3 = Y1 * Z1 (1M) -- must read p.y, p.z before reuse
    const FieldElement52 z3 = p.y * p.z;          // mag 1

    // S = Y1^2 (1S)
    FieldElement52 s = p.y.square();         // mag 1

    // L = (3/2)*X1^2 (1S + mul_int + half)
    FieldElement52 l = p.x.square();         // mag 1
    l.mul_int_assign(3);                     // mag 3
    l.half_assign();                         // mag 2 (parity correct: 3*n[0]&1 == n[0]&1)

    // T = -S * X1 (1N + 1M)
    FieldElement52 t = s.negate(1);          // mag 2
    t.mul_assign(p.x);                       // mag 1

    // X3 = L^2 + 2T (1S + 2A)
    p.x = l.square();                        // mag 1
    p.x.add_assign(t);                       // mag 2
    p.x.add_assign(t);                       // mag 3

    // S' = S^2 (1S)
    s.square_inplace();                      // mag 1

    // T' = T + X3
    t.add_assign(p.x);                       // mag 4

    // Y3 = -(L*(X3+T) + S^2) (1M + 1A + 1N)
    p.y = t * l;                             // mag 1
    p.y.add_assign(s);                       // mag 2
    p.y.negate_assign(2);                    // mag 3

    // Write Z3
    p.z = z3;
}

// -- Mixed Addition (5x52): Jacobian + Affine -> Jacobian ----------------------
// Formula: h2-negation (libsecp-style, a=0 specialization)
// Cost: 8M + 3S + ~8A
// Saves 1S vs madd-2007-bl; fewer add/sub ops; lower output magnitudes.
// Z3 = Z1*H (1M) instead of (Z1+H)^2-Z1Z1-HH (1S+2sub), freeing registers.
[[maybe_unused]] SECP256K1_HOT_FUNCTION SECP256K1_NOINLINE
static JacobianPoint52 jac52_add_mixed(const JacobianPoint52& p, const AffinePoint52& q) noexcept {
    if (SECP256K1_UNLIKELY(p.infinity)) {
        return {q.x, q.y, FieldElement52::one(), false};
    }

    // zz = Z1^2 [1S]
    FieldElement52 const zz = p.z.square();

    // U2 = X2 * zz [1M]
    FieldElement52 const u2 = q.x * zz;

    // S2 = Y2 * Z1 * zz [2M] -- order chosen for ILP (u2 parallel with s2_partial)
    FieldElement52 s2 = q.y * zz;
    s2.mul_assign(p.z);

    // H = U2 - X1
    FieldElement52 const negX1 = p.x.negate(23);     // mag 24
    FieldElement52 const h = u2 + negX1;              // mag 25

    // Variable-time zero check (prob ~2^-256)
    if (SECP256K1_UNLIKELY(h.normalizes_to_zero_var())) {
        FieldElement52 const negY1 = p.y.negate(10);
        FieldElement52 const diff = s2 + negY1;
        if (diff.normalizes_to_zero_var()) {
            return jac52_double(p);
        }
        return {FieldElement52::zero(), FieldElement52::one(), FieldElement52::zero(), true};
    }

    // i = S1 - S2 = Y1 - S2 (not doubled -- saves 1S, 2A vs madd-2007-bl)
    FieldElement52 const negS2 = s2.negate(1);        // mag 2
    FieldElement52 const i_val = p.y + negS2;         // mag 12

    // h2 = H^2 [1S], i2 = i^2 [1S] -- adjacent independent squares for ILP.
    FieldElement52 h2 = h.square();                   // mag 1  (25*25*5 = 3125 < 3.3M)
    FieldElement52 const i2 = i_val.square();         // mag 1  (12*12*5 = 720 < 3.3M)
    // h2 negation trick: h3,t carry the sign, turning subtractions into additions.
    h2.negate_assign(1);                              // h2 = -H^2, mag 2

    // h3 = H * (-H^2) = -H^3 [1M]
    FieldElement52 h3 = h2 * h;                       // mag 1  (2*25 = 50 < 3.3M)

    // t = U1 * (-H^2) = -X1*H^2 [1M]
    FieldElement52 t = p.x * h2;                      // mag 1  (23*2 = 46 < 3.3M)

    // X3 = i^2 + h3 + 2*t  (all additive; i2 ready from early square)
    FieldElement52 x3 = i2 + h3;                      // mag 2
    x3.add_assign(t);                                 // mag 3
    x3.add_assign(t);                                 // mag 4

    // Y3 = i*(t + X3) + h3*S1  [2M + 2A]
    t.add_assign(x3);                                 // t += X3, mag 5
    FieldElement52 y3 = t * i_val;                    // mag 1  (5*12 = 60 < 3.3M)
    h3.mul_assign(p.y);                               // mag 1  (1*10 = 10 < 3.3M)
    y3.add_assign(h3);                                // Y3, mag 2

    // Z3 = Z1 * H [1M] -- simpler than (Z1+H)^2-Z1Z1-HH, fewer registers
    FieldElement52 const z3 = p.z * h;                // mag 1  (5*25 = 125 < 3.3M)

    return {x3, y3, z3, false};
}

// -- In-Place Mixed Addition (5x52): Jacobian + Affine -> Jacobian -------------
// Formula: h2-negation (libsecp-style, a=0 specialization)
// Cost: 8M + 3S + ~8A.  Saves 1S vs madd-2007-bl at the cost of 1 extra M.
// Inlined into the hot loop: with the h2-negation formula, the function body
// is ~200 bytes of x86-64 code.  Two call sites at ~400 bytes added to the
// loop body keeps total I-cache footprint well within L1 (32 KB).
SECP256K1_HOT_FUNCTION
static void jac52_add_mixed_inplace(JacobianPoint52& p, const AffinePoint52& q) noexcept {
    if (SECP256K1_UNLIKELY(p.infinity)) {
        p.x = q.x; p.y = q.y; p.z = FieldElement52::one(); p.infinity = false;
        return;
    }

    // zz = Z1^2 [1S]
    FieldElement52 const zz = p.z.square();

    // U2 = X2 * zz [1M]
    FieldElement52 const u2 = q.x * zz;

    // S2 = Y2 * Z1 * zz [2M] -- ILP: u2 and s2_partial can execute in parallel
    FieldElement52 s2 = q.y * zz;
    s2.mul_assign(p.z);                                       // S2 = Y2 * Z1^3

    // H = U2 - X1
    FieldElement52 const negX1 = p.x.negate(23);              // mag 24
    FieldElement52 const h = u2 + negX1;                      // mag 25

    // Variable-time zero check (prob ~2^-256)
    if (SECP256K1_UNLIKELY(h.normalizes_to_zero_var())) {
        FieldElement52 const negY1 = p.y.negate(10);
        FieldElement52 const diff = s2 + negY1;
        if (diff.normalizes_to_zero_var()) {
            jac52_double_inplace(p);
            return;
        }
        p = {FieldElement52::zero(), FieldElement52::one(), FieldElement52::zero(), true};
        return;
    }

    // Z3 = Z1 * H [1M] -- issued early: p.z not read again after S2,
    // so compute Z3 while h2/i2/h3/t chain executes (ILP).
    p.z.mul_assign(h);                                        // Z3 = Z1*H, mag 1

    // i = S1 - S2 = Y1 - S2 (not doubled -- eliminates 1S + 2A vs madd-2007-bl)
    FieldElement52 const negS2 = s2.negate(1);                // mag 2
    FieldElement52 const i_val = p.y + negS2;                 // mag 12

    // h2 = H^2 [1S], i2 = i^2 [1S] -- adjacent independent squares:
    // OoO engine can interleave their MULX micro-ops, so i2 is ready
    // by the time h3/t finish, removing it from the critical path to X3.
    FieldElement52 h2 = h.square();                           // mag 1
    FieldElement52 const i2 = i_val.square();                 // mag 1  (12*12*5 = 720 < 3.3M)
    // h2 negation trick: h3 and t carry the sign,
    // converting X3/Y3 subtractions into pure additions.
    h2.negate_assign(1);                                      // h2 = -H^2, mag 2

    // h3 = H * (-H^2) = -H^3 [1M]
    FieldElement52 h3 = h2 * h;                               // mag 1  (2*25 = 50 < 3.3M)

    // t = U1 * (-H^2) = -X1*H^2 [1M]
    FieldElement52 t = p.x * h2;                              // mag 1  (23*2 = 46 < 3.3M)

    // X3 = i^2 + h3 + 2*t  (all additions; i2 already ready from early square)
    FieldElement52 x3 = i2 + h3;                              // mag 2
    x3.add_assign(t);                                         // mag 3
    x3.add_assign(t);                                         // mag 4

    // Y3 = i*(t + X3) + h3*S1 [2M + 2A]
    t.add_assign(x3);                                         // t = t + X3, mag 5
    FieldElement52 y3 = t * i_val;                            // mag 1  (5*12 = 60 < 3.3M)
    h3.mul_assign(p.y);                                       // h3 = (-H^3)*Y1, mag 1
    y3.add_assign(h3);                                        // Y3, mag 2

    p.x = x3;
    p.y = y3;
    p.infinity = false;
}

// -- In-place Mixed Addition with Z-ratio output (5x52) -----------------------
// Identical to jac52_add_mixed_inplace but additionally outputs the z-ratio
// (Z3/Z1 = H for the h2-negation formula) needed by table_set_globalz.
// Only used during P table construction (never in the hot main loop).
SECP256K1_NOINLINE
static void jac52_add_mixed_inplace_zr(JacobianPoint52& p,
                                        const AffinePoint52& q,
                                        FieldElement52& zr_out) noexcept {
    if (SECP256K1_UNLIKELY(p.infinity)) {
        p.x = q.x; p.y = q.y; p.z = FieldElement52::one(); p.infinity = false;
        zr_out = FieldElement52::one();
        return;
    }

    FieldElement52 const zz = p.z.square();
    FieldElement52 const u2 = q.x * zz;
    FieldElement52 s2 = q.y * zz;
    s2.mul_assign(p.z);

    FieldElement52 const negX1 = p.x.negate(23);
    FieldElement52 const h = u2 + negX1;

    if (SECP256K1_UNLIKELY(h.normalizes_to_zero_var())) {
        FieldElement52 const negY1 = p.y.negate(10);
        FieldElement52 const diff = s2 + negY1;
        if (diff.normalizes_to_zero_var()) {
            jac52_double_inplace(p);
            zr_out = FieldElement52::one();
            return;
        }
        p = {FieldElement52::zero(), FieldElement52::one(), FieldElement52::zero(), true};
        zr_out = FieldElement52::zero();
        return;
    }

    // Output z-ratio: Z3/Z1 = H for the h2-negation formula.
    zr_out = h;                                               // zr = H, mag 25

    FieldElement52 const negS2 = s2.negate(1);
    FieldElement52 const i_val = p.y + negS2;                 // mag 12

    FieldElement52 h2 = h.square();
    FieldElement52 const i2 = i_val.square();                 // ILP: adjacent independent square
    h2.negate_assign(1);                                      // h2 = -H^2, mag 2

    FieldElement52 h3 = h2 * h;                               // -H^3
    FieldElement52 t = p.x * h2;                              // -X1*H^2

    FieldElement52 x3 = i2 + h3;                              // mag 2
    x3.add_assign(t);                                         // mag 3
    x3.add_assign(t);                                         // mag 4

    t.add_assign(x3);                                         // t = t + X3, mag 5
    FieldElement52 y3 = t * i_val;                            // mag 1
    h3.mul_assign(p.y);                                       // (-H^3)*Y1
    y3.add_assign(h3);                                        // Y3, mag 2

    p.z.mul_assign(h);                                        // Z3 = Z1*H

    p.x = x3;
    p.y = y3;
    p.infinity = false;
}

// -- In-Place Zinv Addition (5x52): p += (b.x, b.y, 1/bzinv) ----------------
// Adds the affine point b as if it has effective Z = 1/bzinv.
// Equivalent to: scaling b into the bzinv frame then doing mixed addition,
// but folds the Z factor into the formula itself (no table entry modification).
//
// This is the secp256k1 analog of libsecp's secp256k1_gej_add_zinv_var.
// Used for G/H table lookups in the main ecmult loop: the precomputed G/H
// entries are true affine (Z=1), and bzinv = Z_shared (from P table's z-ratio
// construction), so the point being added is (b.x, b.y, 1/Z_shared) which
// maps G/H onto the isomorphic curve where P entries live.
//
// Cost: 9M + 3S + ~11A
// Saves 1S per G/H lookup vs the previous approach (2M scale + 7M+4S mixed add).
// Also avoids modifying the G/H table entry (no cache-line dirtying).
// NOT force-inlined: inlining both add_mixed and add_zinv causes ~1100 ns
// regression due to I-cache pressure (hot loop exceeds ~5 KB threshold).
// Only add_mixed is inlineable; add_zinv stays NOINLINE.
SECP256K1_HOT_FUNCTION SECP256K1_NOINLINE
static void jac52_add_zinv_inplace(JacobianPoint52& p,
                                    const AffinePoint52& b,
                                    const FieldElement52& bzinv) noexcept {
    // Handle infinity and edge cases
    if (SECP256K1_UNLIKELY(p.infinity)) {
        // Result = (b.x * bzinv^2, b.y * bzinv^3, 1)
        // This maps the true affine b into the bzinv frame with Z=1,
        // consistent with the final result.z *= bzinv correction.
        FieldElement52 const bzinv2 = bzinv.square();             // 1S
        FieldElement52 const bzinv3 = bzinv2 * bzinv;             // 1M
        p.x = b.x * bzinv2;                                      // 1M
        p.y = b.y * bzinv3;                                      // 1M
        p.z = FieldElement52::one();
        p.infinity = false;
        return;
    }

    // az = Z1 * bzinv  (modified Z for u2, s2 computation)
    FieldElement52 const az = p.z * bzinv;                        // 1M (extra vs mixed add)

    // az2 = az^2 (replaces z1z1 in mixed add)
    FieldElement52 const az2 = az.square();                       // 1S

    // u2 = b.x * az^2,  u1 = p.x  (implicit)
    FieldElement52 const u2 = b.x * az2;                          // 1M

    // s2 = b.y * az^3,  s1 = p.y  (implicit)
    FieldElement52 s2 = b.y * az2;                                // 1M
    s2.mul_assign(az);                                            // s2 = b.y * az^3, 1M

    // h = u2 - u1
    FieldElement52 const negX1 = p.x.negate(23);                  // mag 24
    FieldElement52 const h = u2 + negX1;                          // mag 25

    // Variable-time zero check (prob ~2^-256)
    if (SECP256K1_UNLIKELY(h.normalizes_to_zero_var())) {
        FieldElement52 const negS2 = s2.negate(1);
        FieldElement52 const diff = p.y + negS2;
        if (diff.normalizes_to_zero_var()) {
            jac52_double_inplace(p);
            return;
        }
        p = {FieldElement52::zero(), FieldElement52::one(), FieldElement52::zero(), true};
        return;
    }

    // Z3 = Z1 * h  (uses ORIGINAL p.z, NOT az!)
    // This is the key difference from mixed add: Z3 does NOT include bzinv,
    // keeping the accumulator Z in the secp256k1 domain.
    FieldElement52 const z3 = p.z * h;                            // 1M

    // i = s1 - s2 = Y1 - s2  (sign convention matching libsecp)
    FieldElement52 const negS2 = s2.negate(1);                    // mag 2
    FieldElement52 const i_val = p.y + negS2;                     // mag 12

    // h2 = h^2 [1S], i_sq = i^2 [1S] -- adjacent independent squares:
    // OoO engine can interleave their MULX chains, so i_sq is ready by the
    // time h3/t finish, removing it from the critical path to X3.
    FieldElement52 h2 = h.square();                               // 1S
    FieldElement52 const i_sq = i_val.square();                   // 1S
    h2.negate_assign(1);                                          // h2 = -h^2, mag 2

    // h3 = h * (-h^2) = -h^3
    FieldElement52 h3 = h2 * h;                                   // 1M

    // t = u1 * (-h^2) = -X1 * h^2
    FieldElement52 t = p.x * h2;                                  // 1M

    // X3 = i^2 + h3 + 2*t  (all additions -- sign carried by h2 negation;
    // i_sq already ready from the early square above)
    FieldElement52 x3 = i_sq + h3;                                // mag 2
    x3.add_assign(t);                                             // mag 4
    x3.add_assign(t);                                             // mag 5, X3 = i^2 - h^3 - 2*u1*h^2

    // Y3 = i*(t + X3) + s1*(-h^3)
    // = i*(-u1*h^2 + X3) + Y1*(-h^3)
    // = i*(X3 - u1*h^2) - Y1*h^3
    t.add_assign(x3);                                             // t = (-u1*h^2) + X3
    FieldElement52 y3 = t * i_val;                                // 1M, y3 = i*(X3 - u1*h^2)
    h3.mul_assign(p.y);                                           // h3 = (-h^3)*Y1 = -Y1*h^3, 1M
    y3.add_assign(h3);                                            // Y3 = i*(X3-u1*h^2) - Y1*h^3

    p.x = x3;
    p.y = y3;
    p.z = z3;
    p.infinity = false;
}

// -- In-Place Full Jacobian Addition (5x52): p += q --------------------------
// Formula: add-2007-bl (a=0)
// Cost: 12M + 5S + ~11A.  Eliminates 128-byte return copy.
SECP256K1_HOT_FUNCTION SECP256K1_NOINLINE
static void jac52_add_inplace(JacobianPoint52& p, const JacobianPoint52& q) noexcept {
    if (SECP256K1_UNLIKELY(p.infinity)) { p = q; return; }
    if (SECP256K1_UNLIKELY(q.infinity)) return;

    FieldElement52 z1z1 = p.z.square();
    FieldElement52 z2z2 = q.z.square();
    FieldElement52 const u1 = p.x * z2z2;
    FieldElement52 const u2 = q.x * z1z1;
    FieldElement52 const s1 = p.y * q.z * z2z2;
    FieldElement52 const s2 = q.y * p.z * z1z1;

    FieldElement52 const negU1 = u1.negate(1);
    FieldElement52 const h = u2 + negU1;

    if (SECP256K1_UNLIKELY(h.normalizes_to_zero_var())) {
        FieldElement52 const negS1 = s1.negate(1);
        FieldElement52 const diff = s2 + negS1;
        if (diff.normalizes_to_zero_var()) { jac52_double_inplace(p); return; }
        p = {FieldElement52::zero(), FieldElement52::one(), FieldElement52::zero(), true};
        return;
    }

    FieldElement52 const h2 = h + h;
    FieldElement52 const i = h2.square();
    FieldElement52 j_val = h * i;

    // Compute S1*J before negating j_val (reads s1, j_val intact)
    FieldElement52 s1j = s1 * j_val;                          // mag 1
    j_val.negate_assign(1);                                   // j_val = -J (reuse slot)

    FieldElement52 const negS1 = s1.negate(1);
    FieldElement52 r = s2 + negS1;
    r.add_assign(r);                                          // r = 2*(S2-S1) (in-place)
    FieldElement52 const v = u1 * i;
    FieldElement52 const r_sq = r.square();
    FieldElement52 const negV = v.negate(1);
    FieldElement52 const x3 = r_sq + j_val + negV + negV;           // j_val is -J, mag 7
    FieldElement52 const negX3 = x3.negate(7);
    FieldElement52 const vx3 = v + negX3;
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

    if (SECP256K1_UNLIKELY(h.normalizes_to_zero_var())) {
        const FieldElement52 negS1 = s1.negate(1);
        const FieldElement52 diff = s2 + negS1;
        if (diff.normalizes_to_zero_var()) return jac52_double(p);
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
// -- 4x64 Fallback for scalar_mul when 5x52 batch inversion fails --------
// Extracted to its own noinline function to keep the hot 5x52 path free
// of try/catch overhead.  Called only when eff_z product is zero (~never
// in practice; requires point-at-infinity in the precomputation chain).
SECP256K1_NOINLINE
static Point scalar_mul_fallback_4x64(const Point& base, const Scalar& scalar) {
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
        int32_t const digit = wnaf_buf[static_cast<std::size_t>(i)];
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

// -- Dead 4x64 GLV scalar multiplication removed --
// (jac_add_mixed_4x64_inplace_zr + scalar_mul_glv_4x64) superseded by FE52 path.
#if 0  // Dead code -- previously gated on SECP256K1_HYBRID_4X64_ACTIVE

// Mixed Add with Z-ratio output (4x64): same formula as jac_add_mixed_4x64_inplace
// but additionally outputs zr = 2*H (the Z3/Z1 ratio for madd-2007-bl).
// Used only during table construction (not in hot main loop).
SECP256K1_NOINLINE
static void jac_add_mixed_4x64_inplace_zr(
    std::uint64_t* __restrict__ X,
    std::uint64_t* __restrict__ Y,
    std::uint64_t* __restrict__ Z,
    bool& infinity,
    const std::uint64_t* __restrict__ Qx,
    const std::uint64_t* __restrict__ Qy,
    std::uint64_t* __restrict__ zr_out) noexcept
{
    using namespace secp256k1::fast::fe4x64;

    if (SECP256K1_UNLIKELY(infinity)) {
        copy(X, Qx); copy(Y, Qy);
        Z[0] = 1; Z[1] = 0; Z[2] = 0; Z[3] = 0;
        infinity = false;
        zr_out[0] = 1; zr_out[1] = 0; zr_out[2] = 0; zr_out[3] = 0;
        return;
    }

    alignas(32) std::uint64_t z1z1[4], u2[4], s2[4], h[4], hh[4], i_reg[4], j[4];
    alignas(32) std::uint64_t r_val[4], v[4], x3[4], y3[4], z3[4], t1[4];

    // Z1Z1 = Z^2 (1S)
    sqr(z1z1, Z);
    // U2 = Qx * Z1Z1 (1M)
    mul(u2, Qx, z1z1);
    // S2 = Qy * Z * Z1Z1 (2M)
    mul(t1, Z, z1z1);
    mul(s2, Qy, t1);
    // H = U2 - X1
    sub(h, u2, X);

    // Exceptional case: H == 0 means same x-coordinate
    if (SECP256K1_UNLIKELY(is_zero(h))) {
        alignas(32) std::uint64_t diff[4];
        sub(diff, s2, Y);
        if (is_zero(diff)) {
            // Same point: double
            jac_double_4x64_inplace(X, Y, Z);
            zr_out[0] = 1; zr_out[1] = 0; zr_out[2] = 0; zr_out[3] = 0;
            return;
        }
        // Inverse points: result is infinity
        X[0] = 0; X[1] = 0; X[2] = 0; X[3] = 0;
        Y[0] = 1; Y[1] = 0; Y[2] = 0; Y[3] = 0;
        Z[0] = 0; Z[1] = 0; Z[2] = 0; Z[3] = 0;
        infinity = true;
        zr_out[0] = 0; zr_out[1] = 0; zr_out[2] = 0; zr_out[3] = 0;
        return;
    }

    // Output z-ratio: Z3/Z1 = 2*H (h is const from here)
    add(zr_out, h, h);

    // HH = H^2 (1S)
    sqr(hh, h);
    // I = 4*HH (2 add)
    add(i_reg, hh, hh);
    add(i_reg, i_reg, i_reg);
    // J = H * I (1M)
    mul(j, h, i_reg);
    // r = 2*(S2 - Y) (sub + add)
    sub(r_val, s2, Y);
    add(r_val, r_val, r_val);
    // V = X * I (1M)
    mul(v, X, i_reg);
    // X3 = r^2 - J - 2V (1S + subs)
    sqr(x3, r_val);
    sub(x3, x3, j);
    sub(x3, x3, v);
    sub(x3, x3, v);
    // Y3 = r*(V-X3) - 2*Y*J (2M + sub)
    sub(t1, v, x3);
    mul(y3, r_val, t1);
    mul(t1, Y, j);
    add(t1, t1, t1);
    sub(y3, y3, t1);
    // Z3 = (Z+H)^2 - Z1Z1 - HH (1S + add + subs)
    add(z3, Z, h);
    sqr(z3, z3);
    sub(z3, z3, z1z1);
    sub(z3, z3, hh);

    copy(X, x3);
    copy(Y, y3);
    copy(Z, z3);
    infinity = false;
}

// Main 4x64 GLV scalar multiplication with z-ratio table construction.
SECP256K1_NOINLINE SECP256K1_NO_STACK_PROTECTOR
static Point scalar_mul_glv_4x64(const Point& base, const Scalar& scalar) {
    using namespace secp256k1::fast::fe4x64;

    if (SECP256K1_UNLIKELY(base.is_infinity() || scalar.is_zero()))
        return Point::infinity();

    // -- GLV decomposition ------------------------------------------------
    GLVDecomposition const decomp = glv_decompose(scalar);

    // -- Convert base point to 4x64 limbs ---------------------------------
    JacobianPoint52 const P52 = decomp.k1_neg
        ? to_jac52(base.negate())
        : to_jac52(base);

    alignas(32) std::uint64_t Px[4], Py[4], Pz[4];
    fe52_normalize_and_pack_4x64(P52.x.n, Px);
    fe52_normalize_and_pack_4x64(P52.y.n, Py);
    fe52_normalize_and_pack_4x64(P52.z.n, Pz);

    // -- Compute wNAF for both half-scalars --------------------------------
    constexpr unsigned glv_window = 5;
    constexpr int glv_table_size = (1 << (glv_window - 2));  // 8

    std::array<int32_t, 260> wnaf1_buf{}, wnaf2_buf{};
    std::size_t wnaf1_len = 0, wnaf2_len = 0;
    compute_wnaf_into(decomp.k1, glv_window,
                      wnaf1_buf.data(), wnaf1_buf.size(), wnaf1_len);
    compute_wnaf_into(decomp.k2, glv_window,
                      wnaf2_buf.data(), wnaf2_buf.size(), wnaf2_len);

    while (wnaf1_len > 0 && wnaf1_buf[wnaf1_len - 1] == 0) --wnaf1_len;
    while (wnaf2_len > 0 && wnaf2_buf[wnaf2_len - 1] == 0) --wnaf2_len;

    // -- Precompute table [1P, 3P, ..., 15P] using z-ratio propagation -----
    // Isomorphic curve technique + z-ratio backward sweep.
    // Cost: 1 dbl + 7 mixed adds + 1 backward sweep (7*3M = 21M) = ~28M + ~18S
    // vs old: 1 dbl + 7 mixed adds + 1 batch inversion (~1100 ns SafeGCD)
    struct alignas(32) Aff64 {
        std::uint64_t x[4];
        std::uint64_t y[4];
    };

    std::array<Aff64, glv_table_size> tbl_P{};
    std::array<Aff64, glv_table_size> tbl_phiP{};
    alignas(32) std::uint64_t globalz[4];

    {
        // D = 2*P (Jacobian)
        alignas(32) std::uint64_t Dx[4], Dy[4], Dz[4];
        copy(Dx, Px); copy(Dy, Py); copy(Dz, Pz);
        jac_double_4x64_inplace(Dx, Dy, Dz);

        // C = D.Z, C^2, C^3
        alignas(32) std::uint64_t C[4], C2[4], C3[4];
        copy(C, Dz);
        sqr(C2, C);
        mul(C3, C2, C);

        // pre[0] = P mapped to iso curve: (P.X * C^2, P.Y * C^3)
        mul(tbl_P[0].x, Px, C2);
        mul(tbl_P[0].y, Py, C3);

        // Accumulator on iso curve: (pre[0].x, pre[0].y, P.Z)
        alignas(32) std::uint64_t ax[4], ay[4], az[4];
        copy(ax, tbl_P[0].x);
        copy(ay, tbl_P[0].y);
        copy(az, Pz);
        bool a_inf = false;

        // d_aff on iso curve: (D.X, D.Y) -- affine since D.Z = C is the iso factor
        // Z-ratio array (zr[i] = Z_i / Z_{i-1})
        alignas(32) std::uint64_t zr[glv_table_size][4];
        copy(zr[0], C);  // first z-ratio is C (iso mapping factor)

        // Build 3P, 5P, ..., 15P via mixed adds on iso curve
        for (std::size_t i = 1; i < static_cast<std::size_t>(glv_table_size); ++i) {
            jac_add_mixed_4x64_inplace_zr(ax, ay, az, a_inf, Dx, Dy, zr[i]);
            if (SECP256K1_UNLIKELY(a_inf)) {
                // Exceptional: table entry is infinity -- use fallback
                return scalar_mul_fallback_4x64(base, scalar);
            }
            copy(tbl_P[i].x, ax);
            copy(tbl_P[i].y, ay);
        }

        // globalz = final_accumulator.Z * C (undo isomorphism: iso -> secp256k1)
        mul(globalz, az, C);

        // Backward sweep: equalize Z across all table entries.
        // After this, all entries share implied Z = Z_last on iso curve.
        // Combined with the C factor, globalz = Z_last * C maps to secp256k1.
        {
            alignas(32) std::uint64_t zs[4], zs2[4], zs3[4];
            copy(zs, zr[glv_table_size - 1]);

            std::size_t idx = static_cast<std::size_t>(glv_table_size) - 1;
            while (idx > 0) {
                if (idx != static_cast<std::size_t>(glv_table_size) - 1)
                    mul(zs, zs, zr[idx]);
                --idx;
                sqr(zs2, zs);
                mul(zs3, zs2, zs);
                mul(tbl_P[idx].x, tbl_P[idx].x, zs2);
                mul(tbl_P[idx].y, tbl_P[idx].y, zs3);
            }
        }
    }

    // -- Derive phi(P) table: phi(x,y) = (beta*x, y) or (beta*x, -y) ------
    static constexpr std::uint64_t BETA_4x64[4] = {
        0xc1396c28719501eeULL, 0x9cf0497512f58995ULL,
        0x6e64479eac3434e9ULL, 0x7ae96a2b657c0710ULL
    };

    const bool flip_phi = (decomp.k1_neg != decomp.k2_neg);
    for (std::size_t i = 0; i < static_cast<std::size_t>(glv_table_size); ++i) {
        mul(tbl_phiP[i].x, tbl_P[i].x, BETA_4x64);
        if (flip_phi) {
            negate(tbl_phiP[i].y, tbl_P[i].y);
        } else {
            copy(tbl_phiP[i].y, tbl_P[i].y);
        }
    }

    // -- Shamir's trick: single doubling chain, dual lookups ---------------
    alignas(32) std::uint64_t Rx[4]{}, Ry[4]{}, Rz[4]{};
    bool R_inf = true;

    const std::size_t max_len = (wnaf1_len > wnaf2_len) ? wnaf1_len : wnaf2_len;

    for (int i = static_cast<int>(max_len) - 1; i >= 0; --i) {
        if (!R_inf)
            jac_double_4x64_inplace(Rx, Ry, Rz);

        // k1 contribution
        {
            int32_t const d = wnaf1_buf[static_cast<std::size_t>(i)];
            if (d > 0) {
                const auto& entry = tbl_P[static_cast<std::size_t>((d - 1) >> 1)];
                jac_add_mixed_4x64_inplace(Rx, Ry, Rz, R_inf, entry.x, entry.y);
            } else if (d < 0) {
                const auto& entry = tbl_P[static_cast<std::size_t>((-d - 1) >> 1)];
                alignas(32) std::uint64_t neg_y[4];
                negate(neg_y, entry.y);
                jac_add_mixed_4x64_inplace(Rx, Ry, Rz, R_inf, entry.x, neg_y);
            }
        }

        // k2 contribution
        {
            int32_t const d = wnaf2_buf[static_cast<std::size_t>(i)];
            if (d > 0) {
                const auto& entry = tbl_phiP[static_cast<std::size_t>((d - 1) >> 1)];
                jac_add_mixed_4x64_inplace(Rx, Ry, Rz, R_inf, entry.x, entry.y);
            } else if (d < 0) {
                const auto& entry = tbl_phiP[static_cast<std::size_t>((-d - 1) >> 1)];
                alignas(32) std::uint64_t neg_y[4];
                negate(neg_y, entry.y);
                jac_add_mixed_4x64_inplace(Rx, Ry, Rz, R_inf, entry.x, neg_y);
            }
        }
    }

    // -- Final Z correction: multiply by globalz ---------------------------
    // All table entries had implied Z = globalz on secp256k1.
    // The Shamir loop treated them as affine (Z=1), so the accumulated
    // result's Z is off by a factor of globalz.
    if (!R_inf)
        mul(Rz, Rz, globalz);

    // -- Convert 4x64 result back to Point via FE52 -----------------------
    FieldElement52 const rx52 = FieldElement52::from_4x64_limbs(Rx);
    FieldElement52 const ry52 = FieldElement52::from_4x64_limbs(Ry);
    FieldElement52 const rz52 = FieldElement52::from_4x64_limbs(Rz);

    return Point::from_jacobian52(rx52, ry52, rz52, R_inf);
}

#endif // Dead 4x64 GLV scalar mul (was SECP256K1_HYBRID_4X64_ACTIVE)

// Kept as a separate noinline function so the ~5 KB of local arrays
// live in their own stack frame, preventing GS-cookie corruption that
// occurs when Clang 21 inlines all jac52 helpers into Point::scalar_mul.
// SECP256K1_NO_STACK_PROTECTOR replaces the old try/catch workaround:
// zero overhead, no SEH unwind tables, no GS-cookie check on return.
SECP256K1_NOINLINE SECP256K1_NO_STACK_PROTECTOR
static Point scalar_mul_glv52(const Point& base, const Scalar& scalar) {
    // Guard: infinity base or zero scalar -> result is always infinity
    if (SECP256K1_UNLIKELY(base.is_infinity() || scalar.is_zero())) {
        return Point::infinity();
    }

    // -- GLV decomposition --------------------------------------------
    GLVDecomposition const decomp = glv_decompose(scalar);

    // -- Convert base point to 5x52 domain ----------------------------
    JacobianPoint52 const P52 = decomp.k1_neg
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
    // Uses effective-affine technique (isomorphic curve) + z-ratio propagation.
    // Instead of batch inversion (~1100 ns SafeGCD), we track Z-ratios from
    // each mixed addition and do a backward sweep (only field multiplications).
    // All table entries share an implied Z = globalz on the secp256k1 curve.
    // The Shamir loop treats them as affine; a single Z *= globalz at the end
    // corrects the result.
    std::array<AffinePoint52, glv_table_size> tbl_P;
    std::array<AffinePoint52, glv_table_size> tbl_phiP;
    FieldElement52 globalz;

    {
        // d = 2*P (Jacobian)
        JacobianPoint52 const d = jac52_double(P52);
        FieldElement52 const C  = d.z;              // C = d.Z
        FieldElement52 const C2 = C.square();       // C^2
        FieldElement52 const C3 = C2 * C;           // C^3

        // d as affine on iso curve (Z cancels in isomorphism)
        AffinePoint52 const d_aff = {d.x, d.y};

        // Transform P onto iso curve: (P.X*C^2, P.Y*C^3, P.Z)
        JacobianPoint52 ai = {P52.x * C2, P52.y * C3, P52.z, false};
        tbl_P[0].x = ai.x;
        tbl_P[0].y = ai.y;

        // Z-ratio array: zr[i] = Z_i / Z_{i-1} for the accumulator
        std::array<FieldElement52, glv_table_size> zr;
        zr[0] = C;  // first z-ratio is C (from iso mapping)

        // Build rest using mixed adds on iso curve with z-ratio output
        for (std::size_t i = 1; i < glv_table_size; i++) {
            jac52_add_mixed_inplace_zr(ai, d_aff, zr[i]);
            if (SECP256K1_UNLIKELY(ai.infinity)) {
                return scalar_mul_fallback_4x64(base, scalar);
            }
            tbl_P[i].x = ai.x;
            tbl_P[i].y = ai.y;
        }

        // globalz = final_Z * C (maps from iso curve back to secp256k1)
        globalz = ai.z * C;

        // Backward sweep: rescale table entries so all share implied Z = Z_last.
        // zs accumulates the product zr[n-1] * ... * zr[i+1] = Z_last / Z_i.
        // Each entry is scaled: x *= zs^2, y *= zs^3.
        {
            FieldElement52 zs = zr[glv_table_size - 1];

            for (int idx = glv_table_size - 2; idx >= 0; --idx) {
                if (idx != glv_table_size - 2)
                    zs.mul_assign(zr[static_cast<std::size_t>(idx) + 1]);
                FieldElement52 const zs2 = zs.square();
                FieldElement52 const zs3 = zs2 * zs;
                tbl_P[static_cast<std::size_t>(idx)].x.mul_assign(zs2);
                tbl_P[static_cast<std::size_t>(idx)].y.mul_assign(zs3);
            }
        }

        // Normalize all entries to magnitude 1.
        // Swept entries (0..n-2) already have magnitude 1 from mul_assign,
        // but the last entry (n-1) was never swept and retains raw magnitude
        // from the mixed add (~7 for x, ~4 for y).
        // negate(1) in the phi derivation and Shamir loop requires magnitude 1.
        for (std::size_t i = 0; i < static_cast<std::size_t>(glv_table_size); ++i) {
            tbl_P[i].x.normalize_weak();
            tbl_P[i].y.normalize_weak();
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

    // -- Pre-negation eliminated: negate Y on-the-fly in hot loop.
    // P tables are small (8 entries x 80 bytes = 640 bytes) so cache impact
    // is trivial, but this removes setup cost and simplifies the code.

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
            int32_t const d = wnaf1_buf[static_cast<std::size_t>(i)];
            if (d > 0) {
                jac52_add_mixed_inplace(result52, tbl_P[static_cast<std::size_t>((d - 1) >> 1)]);
            } else if (d < 0) {
                AffinePoint52 pt = tbl_P[static_cast<std::size_t>((-d - 1) >> 1)];
                pt.y.negate_assign(1);
                jac52_add_mixed_inplace(result52, pt);
            }
        }

        // k2 contribution
        {
            int32_t const d = wnaf2_buf[static_cast<std::size_t>(i)];
            if (d > 0) {
                jac52_add_mixed_inplace(result52, tbl_phiP[static_cast<std::size_t>((d - 1) >> 1)]);
            } else if (d < 0) {
                AffinePoint52 pt = tbl_phiP[static_cast<std::size_t>((-d - 1) >> 1)];
                pt.y.negate_assign(1);
                jac52_add_mixed_inplace(result52, pt);
            }
        }
    }

    // -- Final Z correction: all table entries had implied Z = globalz.
    // The Shamir loop treated them as affine (Z=1), so the accumulated
    // result's Z is off by a factor of globalz.  One multiplication fixes it.
    if (!result52.infinity)
        result52.z.mul_assign(globalz);

    return from_jac52(result52);
}

#endif // SECP256K1_FAST_52BIT

// Batch inversion: Convert multiple Jacobian points to Affine using Montgomery's trick
// Cost: 1 inversion + (3*n - 1) multiplications for n points
// vs n inversions for individual conversion
[[maybe_unused]] std::vector<AffinePoint> batch_to_affine(const std::vector<JacobianPoint>& jacobian_points) {
    size_t const n = jacobian_points.size();
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
            FieldElement const z_inv = p.z.inverse();
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
        std::vector<JacobianPoint> const tmp(jacobian_points, jacobian_points + n);
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
            FieldElement const z_inv = p.z.inverse();
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
    Point p(x, y, FieldElement52::one(), false, false);
    p.x_.normalize();
    p.y_.normalize();
    p.z_one_ = true;
    return p;
}

bool Point::z_fe_nonzero(FieldElement& out_z_fe) const noexcept {
    out_z_fe = z_.to_fe();  // fully normalizes
    const auto& zL = out_z_fe.limbs();
    return SECP256K1_LIKELY((zL[0] | zL[1] | zL[2] | zL[3]) != 0);
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
    g.z_one_ = true;
    return g;
}

Point Point::infinity() {
    return Point(FieldElement::zero(), FieldElement::one(), FieldElement::zero(), true);
}

Point Point::from_affine(const FieldElement& x, const FieldElement& y) {
    Point p(x, y, fe_from_uint(1), false);
#if defined(SECP256K1_FAST_52BIT)
    p.x_.normalize();
    p.y_.normalize();
#endif
    p.z_one_ = true;
    return p;
}

Point Point::from_hex(const std::string& x_hex, const std::string& y_hex) {
    FieldElement const x = FieldElement::from_hex(x_hex);
    FieldElement const y = FieldElement::from_hex(y_hex);
    return from_affine(x, y);
}

FieldElement Point::x() const {
    if (infinity_) {
        return FieldElement::zero();
    }
    // Fast path: Z == 1, x_ is already affine
    if (z_one_) {
#if defined(SECP256K1_FAST_52BIT)
        return x_.to_fe();
#else
        return x_;
#endif
    }
#if defined(SECP256K1_FAST_52BIT)
    FieldElement z_fe;
    if (!z_fe_nonzero(z_fe)) return FieldElement::zero(); // LCOV_EXCL_LINE
    FieldElement const z_inv = z_fe.inverse();
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
    // Fast path: Z == 1, y_ is already affine
    if (z_one_) {
#if defined(SECP256K1_FAST_52BIT)
        return y_.to_fe();
#else
        return y_;
#endif
    }
#if defined(SECP256K1_FAST_52BIT)
    FieldElement z_fe;
    if (!z_fe_nonzero(z_fe)) return FieldElement::zero(); // LCOV_EXCL_LINE
    FieldElement const z_inv = z_fe.inverse();
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
    // Fast path: both points affine (Z=1) -- direct affine addition.
    // Returns an affine result (z_one_=true), saving ~12 field muls + the
    // field inversion that to_compressed() would otherwise need.
    // Cost: 1 inversion + 1 mul + 1 sqr  vs  ~15 muls + 4 sqr + 1 inversion.
    if (z_one_ && other.z_one_) {
        if (SECP256K1_UNLIKELY(infinity_)) return other;
        if (SECP256K1_UNLIKELY(other.infinity_)) return *this;
#if defined(SECP256K1_FAST_52BIT)
        // Check x-equality using FE52
        FieldElement52 const dx52 = other.x_ + x_.negate(1);
        if (SECP256K1_UNLIKELY(dx52.normalizes_to_zero_var())) {
            FieldElement52 const dy52 = other.y_ + y_.negate(1);
            if (dy52.normalizes_to_zero_var()) return dbl();
            return Point::infinity();
        }
        // Affine addition: lambda = (y2-y1)/(x2-x1), x3 = lam^2-x1-x2, y3 = lam*(x1-x3)-y1
        FieldElement const dx = dx52.to_fe();
        FieldElement const dx_inv = dx.inverse();
        FieldElement const x1 = x_.to_fe();
        FieldElement const y1 = y_.to_fe();
        FieldElement const x2 = other.x_.to_fe();
        FieldElement const y2 = other.y_.to_fe();
        FieldElement const lam = (y2 - y1) * dx_inv;
        FieldElement const x3 = lam * lam - x1 - x2;
        FieldElement const y3 = lam * (x1 - x3) - y1;
        return Point::from_affine(x3, y3);
#else
        FieldElement const dx = other.x_ - x_;
        if (dx == FieldElement::zero()) {
            FieldElement const dy = other.y_ - y_;
            if (dy == FieldElement::zero()) return dbl();
            return Point::infinity();
        }
        FieldElement const dx_inv = dx.inverse();
        FieldElement const lam = (other.y_ - y_) * dx_inv;
        FieldElement const x3 = lam * lam - x_ - other.x_;
        FieldElement const y3 = lam * (x_ - x3) - y_;
        return Point::from_affine(x3, y3);
#endif
    }
#if defined(SECP256K1_FAST_52BIT)
    JacobianPoint52 const p52{x_, y_, z_, infinity_};
    JacobianPoint52 const q52{other.x_, other.y_, other.z_, other.infinity_};
    JacobianPoint52 const r52 = jac52_add(p52, q52);
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
    JacobianPoint52 const p52{x_, y_, z_, infinity_};
    JacobianPoint52 const r52 = jac52_double(p52);
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
    result.z_one_ = z_one_;
    return result;
#else
    FieldElement neg_y = FieldElement::zero() - y_;
    Point result(x_, neg_y, z_, false);
    result.is_generator_ = false;
    result.z_one_ = z_one_;
    return result;
#endif
}

Point Point::next() const {
#if defined(SECP256K1_FAST_52BIT)
    static const FieldElement52 kGenX52 = FieldElement52::from_fe(kGeneratorX);
    static const FieldElement52 kGenY52 = FieldElement52::from_fe(kGeneratorY);
    JacobianPoint52 p52{x_, y_, z_, infinity_};
    AffinePoint52 const g52{kGenX52, kGenY52};
    jac52_add_mixed_inplace(p52, g52);
    Point result;
    result.x_ = p52.x; result.y_ = p52.y; result.z_ = p52.z;
    result.infinity_ = p52.infinity;
    result.is_generator_ = false;
    return result;
#else
    // Fallback: 4x64 path (non-52-bit platforms / MSVC)
    JacobianPoint p{x_, y_, z_, infinity_};
    AffinePoint const g{kGeneratorX, kGeneratorY};
    JacobianPoint const r = jacobian_add_mixed(p, g);
    Point result;
    result.x_ = r.x; result.y_ = r.y; result.z_ = r.z;
    result.infinity_ = r.infinity;
    result.is_generator_ = false;
    return result;
#endif
}

Point Point::prev() const {
#if defined(SECP256K1_FAST_52BIT)
    static const FieldElement52 kGenX52 = FieldElement52::from_fe(kGeneratorX);
    static const FieldElement52 kNegGenY52 = FieldElement52::from_fe(kNegGeneratorY);
    JacobianPoint52 p52{x_, y_, z_, infinity_};
    AffinePoint52 const ng52{kGenX52, kNegGenY52};
    jac52_add_mixed_inplace(p52, ng52);
    Point result;
    result.x_ = p52.x; result.y_ = p52.y; result.z_ = p52.z;
    result.infinity_ = p52.infinity;
    result.is_generator_ = false;
    return result;
#else
    // Fallback: 4x64 path (non-52-bit platforms / MSVC)
    JacobianPoint p{x_, y_, z_, infinity_};
    AffinePoint const ng{kGeneratorX, kNegGeneratorY};
    JacobianPoint const r = jacobian_add_mixed(p, ng);
    Point result;
    result.x_ = r.x; result.y_ = r.y; result.z_ = r.z;
    result.infinity_ = r.infinity;
    result.is_generator_ = false;
    return result;
#endif
}

// Mutable in-place addition: *this += G (no allocation overhead)
void Point::next_inplace() {
    z_one_ = false;
    if (infinity_) {
        *this = Point::generator();
        return;
    }
#if defined(SECP256K1_FAST_52BIT)
    static const FieldElement52 kGenX52 = FieldElement52::from_fe(kGeneratorX);
    static const FieldElement52 kGenY52 = FieldElement52::from_fe(kGeneratorY);
    JacobianPoint52 p52{x_, y_, z_, infinity_};
    AffinePoint52 const g52{kGenX52, kGenY52};
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
    z_one_ = false;
    if (infinity_) {
        *this = Point::generator().negate();
        return;
    }
#if defined(SECP256K1_FAST_52BIT)
    static const FieldElement52 kGenX52 = FieldElement52::from_fe(kGeneratorX);
    static const FieldElement52 kNegGenY52 = FieldElement52::from_fe(kNegGeneratorY);
    JacobianPoint52 p52{x_, y_, z_, infinity_};
    AffinePoint52 const ng52{kGenX52, kNegGenY52};
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
    z_one_ = false;
#if defined(SECP256K1_FAST_52BIT)
    // Fast path: if other is affine (z = 1), use mixed addition
    // Raw limb check -- no normalization, z_ from affine construction is already canonical {1,0,0,0,0}
    if (!other.infinity_ && fe52_is_one_raw(other.z_)) {
        // Direct: members are already FE52
        JacobianPoint52 p52{x_, y_, z_, infinity_};
        AffinePoint52 const q52{other.x_, other.y_};
        jac52_add_mixed_inplace(p52, q52);
        x_ = p52.x; y_ = p52.y; z_ = p52.z;
        infinity_ = p52.infinity;
        is_generator_ = false;
        return;
    }

    // General case: full Jacobian-Jacobian addition via 5x52 (in-place, zero copy)
    JacobianPoint52 p52{x_, y_, z_, infinity_};
    JacobianPoint52 const q52{other.x_, other.y_, other.z_, other.infinity_};
    jac52_add_inplace(p52, q52);
    x_ = p52.x; y_ = p52.y; z_ = p52.z;
    infinity_ = p52.infinity;
    is_generator_ = false;
#else
    // Fallback: 4x64 path (non-52-bit platforms)
    JacobianPoint p{x_, y_, z_, infinity_};
    JacobianPoint const q{other.x_, other.y_, other.z_, other.infinity_};
    jacobian_add_inplace(p, q);
    x_ = p.x; y_ = p.y; z_ = p.z;
    infinity_ = p.infinity;
    is_generator_ = false;
#endif
}

// Mutable in-place subtraction: *this -= other (no allocation overhead)
void Point::sub_inplace(const Point& other) {
    z_one_ = false;
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
    // Fallback: 4x64 path (non-52-bit platforms)
    JacobianPoint p{x_, y_, z_, infinity_};
    JacobianPoint q{other.x_, other.y_, other.z_, other.infinity_};
    q.y = FieldElement::zero() - q.y;
    jacobian_add_inplace(p, q);
    x_ = p.x; y_ = p.y; z_ = p.z;
    infinity_ = p.infinity;
    is_generator_ = false;
#endif
}

// Mutable in-place doubling: *this = 2*this (no allocation overhead)
void Point::dbl_inplace() {
    z_one_ = false;
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
    z_one_ = false;
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
    AffinePoint52 const q52{FieldElement52::from_fe(ax), FieldElement52::from_fe(ay)};
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

#if defined(SECP256K1_FAST_52BIT)
// FE52-native mixed-add: skips FE52->FE->FE52 roundtrip.
// Used by effective-affine Strauss MSM for zero-conversion hot-loop adds.
void Point::add_mixed52_inplace(const FieldElement52& ax, const FieldElement52& ay) {
    z_one_ = false;
    if (SECP256K1_UNLIKELY(infinity_)) {
        x_ = ax;
        y_ = ay;
        z_ = FieldElement52::one();
        infinity_ = false;
        is_generator_ = false;
        return;
    }
    JacobianPoint52 p52{x_, y_, z_, infinity_};
    AffinePoint52 const q52{ax, ay};
    jac52_add_mixed_inplace(p52, q52);
    x_ = p52.x; y_ = p52.y; z_ = p52.z;
    infinity_ = p52.infinity;
    is_generator_ = false;
}
#endif
// OPTIMIZED: Inline implementation to avoid 6 FieldElement struct copies per call
void Point::sub_mixed_inplace(const FieldElement& ax, const FieldElement& ay) {
    // Negate Y coordinate: subtract is add with negated Y
    FieldElement const neg_ay = FieldElement::zero() - ay;
    add_mixed_inplace(ax, neg_ay);
}

// Repeated mixed addition with a fixed affine point (ax, ay, z=1)
// Changed from CRITICAL_FUNCTION to HOT_FUNCTION - flatten breaks this!
SECP256K1_HOT_FUNCTION
void Point::add_affine_constant_inplace(const FieldElement& ax, const FieldElement& ay) {
    z_one_ = false;
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
        // scalar_mul_generator already normalizes via the auto-normalize path
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

    result.normalize();
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
    {
        Point r = scalar_mul_glv52(*this, scalar);
        r.normalize();
        return r;
    }
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
    result.normalize();
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
    if (!(neg1 || neg2)) {
        // Fast path: K = k1 + k2*lambda (both positive)
        constexpr unsigned window_width = 4;
        auto wnaf1 = compute_wnaf(k1, window_width);
        auto wnaf2 = compute_wnaf(k2, window_width);
        return scalar_mul_precomputed_wnaf(wnaf1, wnaf2, false, false);
    }
#endif

    // Restore original variant: compute both, negate individually, then add
    Point const phi_Q = apply_endomorphism(*this);
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
    JacobianPoint const p = {x_.to_fe(), y_.to_fe(), z_.to_fe(), infinity_};
#else
    JacobianPoint p = {x_, y_, z_, infinity_};
#endif
    
    // Compute phi(Q) - endomorphism (1 field multiplication)
    Point const phi_Q = apply_endomorphism(*this);
#if defined(SECP256K1_FAST_52BIT)
    JacobianPoint const phi_p = {phi_Q.X(), phi_Q.Y(), phi_Q.z(), phi_Q.infinity_};
#else
    JacobianPoint phi_p = {phi_Q.x_, phi_Q.y_, phi_Q.z_, phi_Q.infinity_};
#endif
    
    // Determine table size from wNAF digits
    // For window w: digits are in range [-2^w+1, 2^w-1] odd
    // Table stores positive odd multiples: [1, 3, 5, ..., 2^w-1]
    // Table size = 2^(w-1)
    int max_digit = 0;
    for (auto d : wnaf1) {
        int const abs_d = (d < 0) ? -d : d;
        if (abs_d > max_digit) max_digit = abs_d;
    }
    for (auto d : wnaf2) {
        int const abs_d = (d < 0) ? -d : d;
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

    JacobianPoint const double_Q = jacobian_double(p);        // 2*Q
    JacobianPoint const double_phi_Q = jacobian_double(phi_p); // 2*phi(Q)

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
    
    size_t const max_len = std::max(wnaf1.size(), wnaf2.size());
    
    for (int i = static_cast<int>(max_len) - 1; i >= 0; --i) {
        jacobian_double_inplace(result);
        
        // Add phi(Q) * k2 contribution (no copy: pre-negated tables)
        if (i < static_cast<int>(wnaf2.size())) {
            int32_t const digit2 = wnaf2[static_cast<std::size_t>(i)];
            if (digit2 > 0) {
                int const idx = (digit2 - 1) / 2;
                jacobian_add_inplace(result, table_phi_Q_jac[static_cast<std::size_t>(idx)]);
            } else if (digit2 < 0) {
                int const idx = (-digit2 - 1) / 2;
                jacobian_add_inplace(result, neg_table_phi_Q_jac[static_cast<std::size_t>(idx)]);
            }
        }

        // Add Q * k1 contribution (no copy: pre-negated tables)
        if (i < static_cast<int>(wnaf1.size())) {
            int32_t const digit1 = wnaf1[static_cast<std::size_t>(i)];
            if (digit1 > 0) {
                int const idx = (digit1 - 1) / 2;
                jacobian_add_inplace(result, table_Q_jac[static_cast<std::size_t>(idx)]);
            } else if (digit1 < 0) {
                int const idx = (-digit1 - 1) / 2;
                jacobian_add_inplace(result, neg_table_Q_jac[static_cast<std::size_t>(idx)]);
            }
        }
    }

    return Point(result.x, result.y, result.z, result.infinity);
}

// ============================================================================
// scalar_mul_with_plan: FE52 fast path
// ============================================================================
// Uses effective-affine technique + batch inversion + mixed additions (7M+4S)
// instead of full Jacobian additions (12M+5S) from the legacy 4x64 path.
// wNAF digits and GLV decomposition are taken from the pre-cached KPlan.
// Separate NOINLINE function to keep ~5KB of stack arrays in their own frame.
#if defined(SECP256K1_FAST_52BIT)
SECP256K1_NOINLINE SECP256K1_NO_STACK_PROTECTOR
static Point scalar_mul_with_plan_glv52(const Point& base, const KPlan& plan) {
    // Guard: infinity base -> result is always infinity
    if (SECP256K1_UNLIKELY(base.is_infinity())) {
        return Point::infinity();
    }

    // -- Convert base point to 5x52 domain ----------------------------
    JacobianPoint52 const P52 = plan.neg1
        ? to_jac52(base.negate())
        : to_jac52(base);

    // -- Table size from plan's window width ---------------------------
    const unsigned glv_window = plan.window_width;
    constexpr unsigned kMaxGlvWindow = 7;
    constexpr int kMaxGlvTableSize = 1 << (kMaxGlvWindow - 2);  // 32
    const int glv_table_size = 1 << (glv_window - 2);

    if (SECP256K1_UNLIKELY(glv_table_size > kMaxGlvTableSize || glv_window < 3)) {
        // Safety fallback for unsupported window sizes
        return base.scalar_mul_precomputed_wnaf(plan.wnaf1, plan.wnaf2,
                                                 plan.neg1, plan.neg2);
    }

    // -- Copy wNAF from plan to stack arrays and trim trailing zeros ---
    std::array<int32_t, 260> wnaf1_buf{}, wnaf2_buf{};
    std::size_t wnaf1_len = 0, wnaf2_len = 0;
    {
        const auto& w1 = plan.wnaf1;
        const auto& w2 = plan.wnaf2;
        wnaf1_len = std::min(w1.size(), wnaf1_buf.size());
        wnaf2_len = std::min(w2.size(), wnaf2_buf.size());
        std::copy_n(w1.data(), wnaf1_len, wnaf1_buf.data());
        std::copy_n(w2.data(), wnaf2_len, wnaf2_buf.data());
    }

    // Trim trailing zeros -- GLV half-scalars are ~128 bits but wNAF
    // always outputs 256+ positions.  This halves the doubling count.
    while (wnaf1_len > 0 && wnaf1_buf[wnaf1_len - 1] == 0) --wnaf1_len;
    while (wnaf2_len > 0 && wnaf2_buf[wnaf2_len - 1] == 0) --wnaf2_len;

    // -- Precompute odd multiples [1P, 3P, ..., (2T-1)P] in 5x52 -----
    // Uses z-ratio technique: no field inversion needed.
    // Build table on isomorphic curve where 2P is affine,
    // track Z-ratios from each mixed addition, then backward sweep.
    // All table entries share an implied Z = globalz on secp256k1.
    std::array<AffinePoint52, kMaxGlvTableSize> tbl_P;
    std::array<AffinePoint52, kMaxGlvTableSize> tbl_phiP;
    FieldElement52 globalz;

    {
        // d = 2*P (Jacobian)
        JacobianPoint52 const d = jac52_double(P52);
        FieldElement52 const C  = d.z;              // C = d.Z
        FieldElement52 const C2 = C.square();       // C^2
        FieldElement52 const C3 = C2 * C;           // C^3

        // d as affine on iso curve (Z cancels in isomorphism)
        AffinePoint52 const d_aff = {d.x, d.y};

        // Transform P onto iso curve: (P.X*C^2, P.Y*C^3, P.Z)
        JacobianPoint52 ai = {P52.x * C2, P52.y * C3, P52.z, false};
        tbl_P[0].x = ai.x;
        tbl_P[0].y = ai.y;

        // Z-ratio array: zr[i] = Z_i / Z_{i-1} for the accumulator
        std::array<FieldElement52, kMaxGlvTableSize> zr;
        zr[0] = C;  // first z-ratio is C (from iso mapping)

        // Build rest using mixed adds on iso curve with z-ratio output
        for (int i = 1; i < glv_table_size; i++) {
            jac52_add_mixed_inplace_zr(ai, d_aff, zr[static_cast<std::size_t>(i)]);
            if (SECP256K1_UNLIKELY(ai.infinity)) {
                return base.scalar_mul_precomputed_wnaf(plan.wnaf1, plan.wnaf2,
                                                         plan.neg1, plan.neg2);
            }
            tbl_P[static_cast<std::size_t>(i)].x = ai.x;
            tbl_P[static_cast<std::size_t>(i)].y = ai.y;
        }

        // globalz = final_Z * C (maps from iso curve back to secp256k1)
        globalz = ai.z * C;

        // Backward sweep: rescale table entries so all share implied Z = Z_last.
        {
            FieldElement52 zs = zr[static_cast<std::size_t>(glv_table_size - 1)];

            for (int idx = glv_table_size - 2; idx >= 0; --idx) {
                if (idx != glv_table_size - 2)
                    zs.mul_assign(zr[static_cast<std::size_t>(idx) + 1]);
                FieldElement52 const zs2 = zs.square();
                FieldElement52 const zs3 = zs2 * zs;
                tbl_P[static_cast<std::size_t>(idx)].x.mul_assign(zs2);
                tbl_P[static_cast<std::size_t>(idx)].y.mul_assign(zs3);
            }
        }

        // Normalize all entries to magnitude 1
        for (int i = 0; i < glv_table_size; i++) {
            tbl_P[static_cast<std::size_t>(i)].x.normalize_weak();
            tbl_P[static_cast<std::size_t>(i)].y.normalize_weak();
        }
    }

    // -- Derive phi(P) table: phi(x,y) = (beta*x, y) -----------------
    static const FieldElement52 beta52 = FieldElement52::from_fe(
        FieldElement::from_bytes(glv_constants::BETA));

    const bool flip_phi = (plan.neg1 != plan.neg2);
    for (int i = 0; i < glv_table_size; i++) {
        tbl_phiP[static_cast<std::size_t>(i)].x =
            tbl_P[static_cast<std::size_t>(i)].x * beta52;
        if (flip_phi) {
            tbl_phiP[static_cast<std::size_t>(i)].y =
                tbl_P[static_cast<std::size_t>(i)].y.negate(1);
            tbl_phiP[static_cast<std::size_t>(i)].y.normalize_weak();
        } else {
            tbl_phiP[static_cast<std::size_t>(i)].y =
                tbl_P[static_cast<std::size_t>(i)].y;
        }
    }

    // -- Shamir's trick -- single doubling chain, dual lookups ---------
    JacobianPoint52 result52 = {
        FieldElement52::zero(), FieldElement52::one(),
        FieldElement52::zero(), true
    };

    const std::size_t max_len = (wnaf1_len > wnaf2_len) ? wnaf1_len : wnaf2_len;

    for (int i = static_cast<int>(max_len) - 1; i >= 0; --i) {
        jac52_double_inplace(result52);

        // k1 contribution
        {
            int32_t const d = wnaf1_buf[static_cast<std::size_t>(i)];
            if (d > 0) {
                jac52_add_mixed_inplace(
                    result52, tbl_P[static_cast<std::size_t>((d - 1) >> 1)]);
            } else if (d < 0) {
                AffinePoint52 pt = tbl_P[static_cast<std::size_t>((-d - 1) >> 1)];
                pt.y.negate_assign(1);
                jac52_add_mixed_inplace(result52, pt);
            }
        }

        // k2 contribution
        {
            int32_t const d = wnaf2_buf[static_cast<std::size_t>(i)];
            if (d > 0) {
                jac52_add_mixed_inplace(
                    result52, tbl_phiP[static_cast<std::size_t>((d - 1) >> 1)]);
            } else if (d < 0) {
                AffinePoint52 pt = tbl_phiP[static_cast<std::size_t>((-d - 1) >> 1)];
                pt.y.negate_assign(1);
                jac52_add_mixed_inplace(result52, pt);
            }
        }
    }

    // Final Z correction: all table entries had implied Z = globalz.
    // The Shamir loop treated them as affine (Z=1), so the accumulated
    // result's Z is off by a factor of globalz.
    if (!result52.infinity)
        result52.z.mul_assign(globalz);

    return from_jac52(result52);
}
#endif // SECP256K1_FAST_52BIT

// Fixed K x Variable Q: Optimal performance for repeated K with different Q
// All K-dependent work is cached in KPlan (GLV decomposition + wNAF computation)
// Runtime: phi(Q), tables, Shamir's trick (if signs allow) or separate computation
Point Point::scalar_mul_with_plan(const KPlan& plan) const {
#if defined(SECP256K1_PLATFORM_ESP32) || defined(ESP_PLATFORM) || defined(SECP256K1_PLATFORM_STM32)
    // Embedded: fallback to regular scalar_mul using stored k1
    return scalar_mul(plan.k1);  // scalar_mul already normalizes
#elif defined(SECP256K1_FAST_52BIT)
    // FE52 fast path: effective-affine + batch inversion + mixed adds (7M+4S)
    {
        Point r = scalar_mul_with_plan_glv52(*this, plan);
        r.normalize();
        return r;
    }
#else
    // Legacy 4x64 path: full Jacobian adds (12M+5S)
    {
        Point r = scalar_mul_precomputed_wnaf(plan.wnaf1, plan.wnaf2, plan.neg1, plan.neg2);
        r.normalize();
        return r;
    }
#endif
}

std::array<std::uint8_t, 33> Point::to_compressed() const {
    std::array<std::uint8_t, 33> out{};
    if (infinity_) {
        out.fill(0);
        return out;
    }
    // Fast path: Z == 1, coordinates are already affine
    if (z_one_) {
#if defined(SECP256K1_FAST_52BIT)
        // z_one_ guarantees FE52 is pre-normalized: skip copy+normalize
        out[0] = (y_.n[0] & 1) ? 0x03 : 0x02;
        x_.to_bytes_into(out.data() + 1);
#else
        out[0] = (y_.limbs()[0] & 1) ? 0x03 : 0x02;
        x_.to_bytes_into(out.data() + 1);
#endif
        return out;
    }
    // Slow path: compute affine coordinates with a single inversion
#if defined(SECP256K1_FAST_52BIT)
    FieldElement z_fe;
    if (!z_fe_nonzero(z_fe)) { out.fill(0); return out; } // LCOV_EXCL_LINE
    FieldElement const z_inv = z_fe.inverse();
    FieldElement z_inv2 = z_inv;
    z_inv2.square_inplace();
    FieldElement const x_aff = x_.to_fe() * z_inv2;
    FieldElement const y_aff = y_.to_fe() * z_inv2 * z_inv;
#else
    FieldElement z_inv = z_.inverse();
    FieldElement z_inv2 = z_inv;
    z_inv2.square_inplace();
    FieldElement x_aff = x_ * z_inv2;
    FieldElement y_aff = y_ * z_inv2 * z_inv;
#endif
    out[0] = (y_aff.limbs()[0] & 1) ? 0x03 : 0x02;
    x_aff.to_bytes_into(out.data() + 1);
    return out;
}

std::array<std::uint8_t, 65> Point::to_uncompressed() const {
    std::array<std::uint8_t, 65> out{};
    if (infinity_) {
        out.fill(0);
        return out;
    }
    // Fast path: Z == 1, coordinates are already affine
    if (z_one_) {
#if defined(SECP256K1_FAST_52BIT)
        out[0] = 0x04;
        x_.to_bytes_into(out.data() + 1);
        y_.to_bytes_into(out.data() + 33);
#else
        out[0] = 0x04;
        x_.to_bytes_into(out.data() + 1);
        y_.to_bytes_into(out.data() + 33);
#endif
        return out;
    }
    // Slow path: compute affine coordinates with a single inversion
#if defined(SECP256K1_FAST_52BIT)
    FieldElement z_fe;
    if (!z_fe_nonzero(z_fe)) { out.fill(0); return out; } // LCOV_EXCL_LINE
    FieldElement const z_inv = z_fe.inverse();
    FieldElement z_inv2 = z_inv;
    z_inv2.square_inplace();
    FieldElement const x_aff = x_.to_fe() * z_inv2;
    FieldElement const y_aff = y_.to_fe() * z_inv2 * z_inv;
#else
    FieldElement z_inv = z_.inverse();
    FieldElement z_inv2 = z_inv;
    z_inv2.square_inplace();
    FieldElement x_aff = x_ * z_inv2;
    FieldElement y_aff = y_ * z_inv2 * z_inv;
#endif
    out[0] = 0x04;
    x_aff.to_bytes_into(out.data() + 1);
    y_aff.to_bytes_into(out.data() + 33);
    return out;
}

// -- Y-parity check (single inversion) ---------------------------------------
bool Point::has_even_y() const {
    if (infinity_) return false;
    // Fast path: Z == 1, Y is already affine
    if (z_one_) {
#if defined(SECP256K1_FAST_52BIT)
        // z_one_ guarantees FE52 is pre-normalized
        return (y_.n[0] & 1) == 0;
#else
        return (y_.limbs()[0] & 1) == 0;
#endif
    }
#if defined(SECP256K1_FAST_52BIT)
    FieldElement z_fe;
    if (!z_fe_nonzero(z_fe)) return false; // LCOV_EXCL_LINE
    FieldElement const z_inv = z_fe.inverse();
    FieldElement z_inv2 = z_inv;
    z_inv2.square_inplace();
    FieldElement const y_aff = y_.to_fe() * z_inv2 * z_inv;
#else
    FieldElement z_inv = z_.inverse();
    FieldElement z_inv2 = z_inv;
    z_inv2.square_inplace();
    FieldElement y_aff = y_ * z_inv2 * z_inv;
#endif
    // mul_impl Barrett-reduces to [0, p) -- limbs()[0] LSB is parity.
    return (y_aff.limbs()[0] & 1) == 0;
}

// -- Combined x-bytes + Y-parity (single inversion) --------------------------
std::pair<std::array<uint8_t, 32>, bool> Point::x_bytes_and_parity() const {
    if (infinity_) return {std::array<uint8_t,32>{}, false};
    // Fast path: Z == 1
    if (z_one_) {
#if defined(SECP256K1_FAST_52BIT)
        // z_one_ guarantees FE52 is pre-normalized
        bool const y_odd = (y_.n[0] & 1) != 0;
        std::array<uint8_t, 32> xb{};
        x_.to_bytes_into(xb.data());
        return {xb, y_odd};
#else
        bool const y_odd = (y_.limbs()[0] & 1) != 0;
        return {x_.to_bytes(), y_odd};
#endif
    }
#if defined(SECP256K1_FAST_52BIT)
    FieldElement z_fe;
    if (!z_fe_nonzero(z_fe)) return {std::array<uint8_t,32>{}, false}; // LCOV_EXCL_LINE
    FieldElement const z_inv = z_fe.inverse();
    FieldElement z_inv2 = z_inv;
    z_inv2.square_inplace();
    FieldElement const x_aff = x_.to_fe() * z_inv2;
    FieldElement const y_aff = y_.to_fe() * z_inv2 * z_inv;
#else
    FieldElement z_inv = z_.inverse();
    FieldElement z_inv2 = z_inv;
    z_inv2.square_inplace();
    FieldElement x_aff = x_ * z_inv2;
    FieldElement y_aff = y_ * z_inv2 * z_inv;
#endif
    // Parity from limbs directly (avoids to_bytes serialization)
    bool const y_odd = (y_aff.limbs()[0] & 1) != 0;
    return {x_aff.to_bytes(), y_odd};
}

// -- Normalize: Jacobian -> affine (Z=1) with ONE field inversion -------------
// After this call z_one_ == true and all serialization is O(1).
void Point::normalize() {
    if (z_one_ || infinity_) return;
#if defined(SECP256K1_FAST_52BIT)
    FieldElement z_fe;
    if (!z_fe_nonzero(z_fe)) return; // LCOV_EXCL_LINE
    FieldElement const z_inv = z_fe.inverse();
    FieldElement z_inv2 = z_inv;
    z_inv2.square_inplace();
    FieldElement const z_inv3 = z_inv2 * z_inv;
    x_ = FieldElement52::from_fe(x_.to_fe() * z_inv2);
    y_ = FieldElement52::from_fe(y_.to_fe() * z_inv3);
    // Pre-normalize FE52 so z_one_ fast paths skip redundant normalize
    x_.normalize();
    y_.normalize();
    z_ = FieldElement52::one();
#else
    FieldElement z_inv = z_.inverse();
    FieldElement z_inv2 = z_inv;
    z_inv2.square_inplace();
    FieldElement const z_inv3 = z_inv2 * z_inv;
    x_ *= z_inv2;
    y_ *= z_inv3;
    z_ = fe_from_uint(1);
#endif
    z_one_ = true;
}

// -- Fast x-only: 32-byte x-coordinate (no Y recovery) -----------------------
// Saves one multiply vs x_bytes_and_parity() by skipping Z^(-3)*Y.
std::array<uint8_t, 32> Point::x_only_bytes() const {
    if (infinity_) return {};
    // Fast path: Z == 1
    if (z_one_) {
#if defined(SECP256K1_FAST_52BIT)
        std::array<uint8_t, 32> xb{};
        x_.to_bytes_into(xb.data());
        return xb;
#else
        return x_.to_bytes();
#endif
    }
#if defined(SECP256K1_FAST_52BIT)
    FieldElement z_fe;
    if (!z_fe_nonzero(z_fe)) return {}; // LCOV_EXCL_LINE
    FieldElement const z_inv = z_fe.inverse();
    FieldElement z_inv2 = z_inv;
    z_inv2.square_inplace();
    FieldElement const x_aff = x_.to_fe() * z_inv2;
#else
    FieldElement z_inv = z_.inverse();
    FieldElement z_inv2 = z_inv;
    z_inv2.square_inplace();
    FieldElement const x_aff = x_ * z_inv2;
#endif
    return x_aff.to_bytes();
}

// -- Batch normalize: Montgomery trick for N points ---------------------------
// 1 inversion + 3(N-1) multiplications instead of N inversions.
void Point::batch_normalize(const Point* points, size_t n,
                            FieldElement* out_x, FieldElement* out_y) {
    if (n == 0) return;

    // Accumulate products of Z values (Montgomery's trick)
    // partials[i] = Z[0] * Z[1] * ... * Z[i]
    // We allocate on the stack for small N, heap for large N.
    constexpr size_t STACK_LIMIT = 256;
    FieldElement stack_partials[STACK_LIMIT];
    std::unique_ptr<FieldElement[]> heap_partials;
    FieldElement* partials = nullptr;
    if (n <= STACK_LIMIT) {
        partials = stack_partials;
    } else {
        heap_partials = std::make_unique<FieldElement[]>(n);
        partials = heap_partials.get();
    }

    // Forward pass: accumulate Z products
    FieldElement running = FieldElement::one();
    for (size_t i = 0; i < n; ++i) {
        if (points[i].is_infinity()) {
            partials[i] = running; // skip infinity, don't multiply
        } else {
#if defined(SECP256K1_FAST_52BIT)
            FieldElement z_fe = points[i].z_.to_fe();
#else
            FieldElement z_fe = points[i].z_;
#endif
            partials[i] = running;
            running *= z_fe;
        }
    }

    // Single inversion of the accumulated product
    FieldElement inv = running.inverse();

    // Backward pass: recover individual Z^(-1) values
    for (size_t i = n; i-- > 0; ) {
        if (points[i].is_infinity()) {
            out_x[i] = FieldElement::zero();
            out_y[i] = FieldElement::zero();
            continue;
        }
#if defined(SECP256K1_FAST_52BIT)
        FieldElement z_fe = points[i].z_.to_fe();
#else
        FieldElement z_fe = points[i].z_;
#endif
        // z_inv_i = partials[i] * inv  (partials[i] = product of Z[0..i-1])
        FieldElement const z_inv_i = partials[i] * inv;
        // Update inv for next iteration: inv *= Z[i]
        inv *= z_fe;

        // Compute affine: x_aff = X * Z^(-2), y_aff = Y * Z^(-3)
        FieldElement z_inv2 = z_inv_i;
        z_inv2.square_inplace();
#if defined(SECP256K1_FAST_52BIT)
        out_x[i] = points[i].x_.to_fe() * z_inv2;
        out_y[i] = points[i].y_.to_fe() * z_inv2 * z_inv_i;
#else
        out_x[i] = points[i].x_ * z_inv2;
        out_y[i] = points[i].y_ * z_inv2 * z_inv_i;
#endif
    }
}

// -- Batch to_compressed: serialize N points with ONE inversion ---------------
void Point::batch_to_compressed(const Point* points, size_t n,
                                std::array<uint8_t, 33>* out) {
    if (n == 0) return;

    constexpr size_t STACK_LIMIT = 256;
    FieldElement stack_x[STACK_LIMIT], stack_y[STACK_LIMIT];
    std::unique_ptr<FieldElement[]> heap_x, heap_y;
    FieldElement* aff_x = nullptr;
    FieldElement* aff_y = nullptr;
    if (n <= STACK_LIMIT) {
        aff_x = stack_x; aff_y = stack_y;
    } else {
        heap_x = std::make_unique<FieldElement[]>(n);
        heap_y = std::make_unique<FieldElement[]>(n);
        aff_x = heap_x.get(); aff_y = heap_y.get();
    }

    batch_normalize(points, n, aff_x, aff_y);

    for (size_t i = 0; i < n; ++i) {
        if (points[i].is_infinity()) {
            out[i].fill(0);
            continue;
        }
        auto x_bytes = aff_x[i].to_bytes();
        auto y_bytes = aff_y[i].to_bytes();
        out[i][0] = (y_bytes[31] & 1U) ? 0x03 : 0x02;
        std::copy(x_bytes.begin(), x_bytes.end(), out[i].begin() + 1);
    }
}

// -- Batch x_only_bytes: extract N x-coordinates with ONE inversion -----------
void Point::batch_x_only_bytes(const Point* points, size_t n,
                               std::array<uint8_t, 32>* out) {
    if (n == 0) return;

    // Montgomery batch inversion (same trick, but only need x = X*Z^(-2))
    constexpr size_t STACK_LIMIT = 256;
    FieldElement stack_partials[STACK_LIMIT];
    std::unique_ptr<FieldElement[]> heap_partials;
    FieldElement* partials = nullptr;
    if (n <= STACK_LIMIT) {
        partials = stack_partials;
    } else {
        heap_partials = std::make_unique<FieldElement[]>(n);
        partials = heap_partials.get();
    }

    FieldElement running = FieldElement::one();
    for (size_t i = 0; i < n; ++i) {
        if (points[i].is_infinity()) {
            partials[i] = running;
        } else {
#if defined(SECP256K1_FAST_52BIT)
            FieldElement z_fe = points[i].z_.to_fe();
#else
            FieldElement z_fe = points[i].z_;
#endif
            partials[i] = running;
            running *= z_fe;
        }
    }

    FieldElement inv = running.inverse();

    for (size_t i = n; i-- > 0; ) {
        if (points[i].is_infinity()) {
            out[i].fill(0);
            continue;
        }
#if defined(SECP256K1_FAST_52BIT)
        FieldElement z_fe = points[i].z_.to_fe();
#else
        FieldElement z_fe = points[i].z_;
#endif
        FieldElement const z_inv_i = partials[i] * inv;
        inv *= z_fe;

        FieldElement z_inv2 = z_inv_i;
        z_inv2.square_inplace();
#if defined(SECP256K1_FAST_52BIT)
        FieldElement const x_aff = points[i].x_.to_fe() * z_inv2;
#else
        FieldElement const x_aff = points[i].x_ * z_inv2;
#endif
        out[i] = x_aff.to_bytes();
    }
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
    Scalar const a_lo = Scalar::from_limbs({a_limbs[0], a_limbs[1], 0, 0});
    Scalar const a_hi = Scalar::from_limbs({a_limbs[2], a_limbs[3], 0, 0});

    // -- GLV decompose b only -----------------------------------------
    GLVDecomposition const decomp_b = glv_decompose(b);

    // -- Window widths: w=15 for G (precomputed), w=5 for P (per-call) ---
    constexpr unsigned WINDOW_G = 15;    // -> 2^13 = 8192 entries per G/H table
    constexpr unsigned WINDOW_P = 5;     // -> 2^3 = 8 entries per P/psiP table
    constexpr int G_TABLE_SIZE = (1 << (WINDOW_G - 2));  // 8192
    constexpr int P_TABLE_SIZE = (1 << (WINDOW_P - 2));  // 8

    // -- Static generator tables (computed once, 8192 entries per base) --
    // Two bases: G (generator) and H = 2^128*G
    // Uses effective-affine technique for efficient table building.
    // Negative wNAF digits negate Y on-the-fly (5 limb ops, data already
    // in L1 cache from the lookup) instead of pre-negated tables.
    // This halves the table memory from 2.5 MB to 1.25 MB, fitting in
    // per-core L3 cache and reducing TLB + cache pressure.
    struct GenTables {
        AffinePointCompact tbl_G[G_TABLE_SIZE];  // [G, 3G, 5G, ..., 16383G] compact 64B
        AffinePointCompact tbl_H[G_TABLE_SIZE];  // [H, 3H, 5H, ..., 16383H] compact 64B
    };
    static const GenTables* const gen_tables = []() -> const GenTables* {
        auto* t = new GenTables;

        // Helper: build odd-multiple table for base point B using effective-affine
        auto build_table = [](const JacobianPoint52& B,
                              AffinePointCompact* out, int count) {
            // d = 2*B, work on isomorphic curve where d is affine
            JacobianPoint52 const d = jac52_double(B);
            FieldElement52 const C  = d.z;
            FieldElement52 const C2 = C.square();
            FieldElement52 const C3 = C2 * C;
            AffinePoint52 const d_aff = {d.x, d.y};

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
                FieldElement52 const zinv2 = zs[i].square();
                FieldElement52 const zinv3 = zinv2 * zs[i];
                AffinePoint52 const aff = {iso[i].x * zinv2, iso[i].y * zinv3};
                out[i] = AffinePointCompact::from_affine52(aff);
            }

            delete[] zs;
            delete[] prods;
            delete[] eff_z;
            delete[] iso;
        };

        // Build tbl_G: odd multiples of G
        Point const G = Point::generator();
        JacobianPoint52 const G52 = to_jac52(G);
        build_table(G52, t->tbl_G, G_TABLE_SIZE);

        // Build tbl_H: odd multiples of H = 2^128*G
        JacobianPoint52 H52 = G52;
        for (std::size_t i = 0; i < 128; i++) {
            jac52_double_inplace(H52);
        }
        build_table(H52, t->tbl_H, G_TABLE_SIZE);

        return t;
    }();

    // -- Precompute P tables (per-call, window=5, 8 entries) ---------
    // Uses effective-affine technique (same as libsecp256k1/CT path):
    //   d = 2*P (Jacobian), C = d.Z, iso curve: Y'^2 = X'^3 + C^6*7
    //   On iso curve, d is affine: (d.X, d.Y), so table adds are
    //   mixed Jac+Affine (7M+4S) instead of full Jac+Jac (12M+5S).
    //   Saves ~33M + 6S = ~1.4us per verify.
    JacobianPoint52 const P52 = decomp_b.k1_neg
        ? to_jac52(P.negate())
        : to_jac52(P);

    std::array<AffinePoint52, P_TABLE_SIZE> tbl_P;
    std::array<AffinePoint52, P_TABLE_SIZE> tbl_phiP;

    // Z_shared: effective Z on secp256k1 shared by all pseudo-affine P entries.
    // Declared here so it is available for add_zinv and final Z correction.
    FieldElement52 Z_shared;

    // -- Z-ratio table construction (0 inversions) -------------------
    // Instead of batch-inverting all Z coordinates (~2us), collect z-ratios
    // during the iso-curve additions and use a backward sweep to express
    // all entries at a common shared Z.  The main loop works in this
    // "Z_shared frame" (G/H entries are scaled into the frame), and
    // result52.z is multiplied by Z_shared before returning.
    {
        JacobianPoint52 const d = jac52_double(P52);
        FieldElement52 const C  = d.z;
        FieldElement52 const C2 = C.square();
        FieldElement52 const C3 = C2 * C;
        AffinePoint52 const d_aff = {d.x, d.y};

        // Build table on iso curve, collecting z-ratios from each addition.
        JacobianPoint52 ai = {P52.x * C2, P52.y * C3, P52.z, false};
        tbl_P[0] = {ai.x, ai.y};  // pseudo-affine: Jacobian X,Y with Z implicit

        std::array<FieldElement52, P_TABLE_SIZE> zr;  // z-ratios [1..n-1]
        for (std::size_t i = 1; i < static_cast<std::size_t>(P_TABLE_SIZE); i++) {
            jac52_add_mixed_inplace_zr(ai, d_aff, zr[i]);
            tbl_P[i] = {ai.x, ai.y};
        }

        // Backward sweep (a la libsecp256k1 table_set_globalz):
        // Scale entries 0..n-2 so all share Z of the last point on iso curve.
        // Cost: (n-2) muls + (n-1)*(1S+2M) = 20M+7S for n=8.
        tbl_P[P_TABLE_SIZE - 1].y.normalize_weak();
        FieldElement52 zs = zr[P_TABLE_SIZE - 1];
        for (int j = static_cast<int>(P_TABLE_SIZE) - 2; j >= 0; --j) {
            if (j != static_cast<int>(P_TABLE_SIZE) - 2) {
                zs.mul_assign(zr[static_cast<std::size_t>(j) + 1]);
            }
            FieldElement52 const zs2 = zs.square();
            FieldElement52 const zs3 = zs2 * zs;
            tbl_P[static_cast<std::size_t>(j)].x.mul_assign(zs2);
            tbl_P[static_cast<std::size_t>(j)].y.mul_assign(zs3);
        }

        // Z_shared = ai.z * C: effective Z on secp256k1 for all pseudo-affine entries.
        Z_shared = ai.z * C;
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

    // neg tables removed -- negate Y on-the-fly in hot loop

    // -- Compute wNAF for all 4 half-scalars -------------------------
    // GLV half-scalars are <= 128 bits; wNAF produces at most 129 digits.
    // a_lo/a_hi are 128-bit halves of scalar a (same bound).
    // 130 entries = 129 max digits + 1 carry.  Saves ~2KB of stack zeroing.
    std::array<int32_t, 130> wnaf_a_lo{}, wnaf_a_hi{}, wnaf_b1{}, wnaf_b2{};
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
        {
            int32_t const d_g = wnaf_a_lo[static_cast<std::size_t>(i)];
            if (d_g) {
                int32_t const abs_g = (d_g ^ (d_g >> 31)) - (d_g >> 31);
                SECP256K1_PREFETCH_READ(&gen_tables->tbl_G[static_cast<std::size_t>((abs_g - 1) >> 1)]);
            }
            int32_t const d_h = wnaf_a_hi[static_cast<std::size_t>(i)];
            if (d_h) {
                int32_t const abs_h = (d_h ^ (d_h >> 31)) - (d_h >> 31);
                SECP256K1_PREFETCH_READ(&gen_tables->tbl_H[static_cast<std::size_t>((abs_h - 1) >> 1)]);
            }
        }

        jac52_double_inplace(result52);

        {
            int32_t const d = wnaf_a_lo[static_cast<std::size_t>(i)];
            if (SECP256K1_UNLIKELY(d != 0)) {
                int32_t const sign = d >> 31;
                int32_t const abs_d = (d ^ sign) - sign;
                AffinePoint52 pt = gen_tables->tbl_G[static_cast<std::size_t>((abs_d - 1) >> 1)].to_affine52();
                pt.y.conditional_negate_assign(sign);
                jac52_add_zinv_inplace(result52, pt, Z_shared);
            }
        }

        {
            int32_t const d = wnaf_a_hi[static_cast<std::size_t>(i)];
            if (SECP256K1_UNLIKELY(d != 0)) {
                int32_t const sign = d >> 31;
                int32_t const abs_d = (d ^ sign) - sign;
                AffinePoint52 pt = gen_tables->tbl_H[static_cast<std::size_t>((abs_d - 1) >> 1)].to_affine52();
                pt.y.conditional_negate_assign(sign);
                jac52_add_zinv_inplace(result52, pt, Z_shared);
            }
        }

        {
            int32_t const d = wnaf_b1[static_cast<std::size_t>(i)];
            if (SECP256K1_UNLIKELY(d != 0)) {
                int32_t const sign = d >> 31;
                int32_t const abs_d = (d ^ sign) - sign;
                AffinePoint52 pt = tbl_P[static_cast<std::size_t>((abs_d - 1) >> 1)];
                pt.y.conditional_negate_assign(sign);
                jac52_add_mixed_inplace(result52, pt);
            }
        }

        {
            int32_t const d = wnaf_b2[static_cast<std::size_t>(i)];
            if (SECP256K1_UNLIKELY(d != 0)) {
                int32_t const sign = d >> 31;
                int32_t const abs_d = (d ^ sign) - sign;
                AffinePoint52 pt = tbl_phiP[static_cast<std::size_t>((abs_d - 1) >> 1)];
                pt.y.conditional_negate_assign(sign);
                jac52_add_mixed_inplace(result52, pt);
            }
        }
    }

    if (!result52.infinity) {
        result52.z.mul_assign(Z_shared);
    }

    return from_jac52(result52);
}
// DISABLED: ESP32/Embedded 4-stream GLV Strauss produces incorrect verify results.
// Root cause: the 4-stream interleaved Shamir scan with GLV decomposition of BOTH
// scalars (a and b) computes wrong R' point, causing 100% ECDSA/Schnorr verify failure.
// The simple fallback (a*G + b*P via separate scalar_muls) uses proven code paths
// (gen_fixed_mul for G, GLV+Shamir per-point for P) and is correct.
// TODO: investigate and fix the 4-stream path, then re-enable.
#elif 0 && (defined(SECP256K1_PLATFORM_ESP32) || defined(ESP_PLATFORM) || defined(SECP256K1_PLATFORM_STM32))
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

    {
        Point r(jac.x, jac.y, jac.z, jac.infinity);
        return r;
    }
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
