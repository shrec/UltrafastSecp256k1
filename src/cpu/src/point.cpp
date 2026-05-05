#include "secp256k1/point.hpp"
#if !defined(SECP256K1_PLATFORM_ESP32) && !defined(ESP_PLATFORM) && !defined(SECP256K1_PLATFORM_STM32)
#include "secp256k1/precompute.hpp"
#endif
#include "secp256k1/glv.hpp"
#include "secp256k1/ct/point.hpp"
#if defined(__SIZEOF_INT128__) && !defined(__EMSCRIPTEN__)
#include "secp256k1/field_52.hpp"
#endif
#ifdef _OPENMP
#include <omp.h>
#endif
// Inline 4x64 ADCX/ADOX field operations for hot-loop point ops
#if (defined(__x86_64__) || defined(_M_X64)) && defined(__ADX__) && defined(__BMI2__)
#include "secp256k1/field_4x64_inline.hpp"
#endif

#include "secp256k1/debug_invariants.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#if !defined(_WIN32)
#include <unistd.h>
#endif

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
template<typename T>
static void compute_wnaf_into(const Scalar& scalar, unsigned window_width,
                               T* out, std::size_t out_capacity,
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
    std::memset(out, 0, out_capacity * sizeof(T));

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

        out[bit] = static_cast<T>(word);
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

// Forward declarations for delegation pattern (avoid duplicating formula bodies)
static inline void jacobian_double_inplace(JacobianPoint& p);
[[maybe_unused]] static inline void jacobian_add_mixed_inplace(JacobianPoint& p, const AffinePoint& q);

// Point doubling: dbl-2007-a formula (a=0). Delegates to in-place variant.
#if defined(_MSC_VER) && !defined(__clang__)
#pragma inline_recursion(on)
#pragma inline_depth(255)
#endif
SECP256K1_HOT_FUNCTION
JacobianPoint jacobian_double(const JacobianPoint& p) {
    JacobianPoint r = p;
    jacobian_double_inplace(r);
    return r;
}

// Mixed Jacobian-Affine addition: delegates to in-place variant.
#if defined(_MSC_VER) && !defined(__clang__)
#pragma inline_recursion(on)
#pragma inline_depth(255)
#endif
[[nodiscard]] [[maybe_unused]] JacobianPoint jacobian_add_mixed(const JacobianPoint& p, const AffinePoint& q) {
    JacobianPoint r = p;
    jacobian_add_mixed_inplace(r, q);
    return r;
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
    if (SECP256K1_UNLIKELY(p.infinity)) {
        // p.y == 0 is impossible for a non-infinity secp256k1 point (prime order, no 2-torsion)
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

// File-scope FE52 beta constant — avoids per-call function-static init-flag
// (atomic load on every call). Shared by batch_scalar_mul_fixed_k,
// derive_phi52_table, and any future FE52 path needing the GLV endomorphism.
static const FieldElement52 kBeta52_pt = FieldElement52::from_fe(
    FieldElement::from_bytes(glv_constants::BETA));

struct JacobianPoint52 {
    FieldElement52 x;
    FieldElement52 y;
    FieldElement52 z;
    bool infinity{true};
};

// AffinePoint52 now defined in field_52.hpp (shared with SchnorrXonlyPubkey cache).
// struct AffinePoint52 { FieldElement52 x, y; };  — removed from here.

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

// ============================================================================
// 4x64 <-> 5x52 Dual Representation Helpers
// ============================================================================
// When SECP256K1_USE_4X64_POINT_OPS is defined, point arithmetic routes
// through 4x64 FieldElement ops (ADCX/ADOX asm) instead of 5x52.
// Storage stays FE52 -- only the compute path changes.
// Conversion cost: ~1ns per field element (5 shifts), negligible vs 100+ ns per point op.
#if defined(SECP256K1_USE_4X64_POINT_OPS) && defined(SECP256K1_FAST_52BIT)

// FE52 Point members -> 4x64 JacobianPoint (normalize + pack)
static inline JacobianPoint point_to_jac4x64(const FieldElement52& x,
                                               const FieldElement52& y,
                                               const FieldElement52& z,
                                               bool inf) noexcept {
    return {x.to_fe(), y.to_fe(), z.to_fe(), inf};
}

// 4x64 JacobianPoint -> FE52 members (unpack + shift)
static inline void jac4x64_to_fe52(const JacobianPoint& j,
                                    FieldElement52& ox, FieldElement52& oy,
                                    FieldElement52& oz, bool& oinf) noexcept {
    ox = FieldElement52::from_fe(j.x);
    oy = FieldElement52::from_fe(j.y);
    oz = FieldElement52::from_fe(j.z);
    oinf = j.infinity;
}

// 4x64 AffinePoint from FE52
static inline AffinePoint fe52_to_affine4x64(const FieldElement52& x,
                                              const FieldElement52& y) noexcept {
    return {x.to_fe(), y.to_fe()};
}

#endif // SECP256K1_USE_4X64_POINT_OPS && SECP256K1_FAST_52BIT

// -- Point Doubling (5x52) ----------------------------------------------------
// Forward declaration for delegation
static inline void jac52_double_coords(FieldElement52& x, FieldElement52& y, FieldElement52& z) noexcept;

// Point doubling (5x52, return by value): delegates to the in-place coords variant.
// Formula: libsecp256k1 gej_double (a=0 specialization), 3M+4S+8 cheap.
SECP256K1_HOT_FUNCTION SECP256K1_NOINLINE
static JacobianPoint52 jac52_double(const JacobianPoint52& p) noexcept {
    if (SECP256K1_UNLIKELY(p.infinity)) {
        return {FieldElement52::zero(), FieldElement52::one(), FieldElement52::zero(), true};
    }
    JacobianPoint52 r = {p.x, p.y, p.z, false};
    jac52_double_coords(r.x, r.y, r.z);
    return r;
}

// Dead 4x64 point ops removed -- superseded by FE52 path.
// SECP256K1_HYBRID_4X64_ACTIVE preserved for FE52->4x64 inverse/sqrt boundary.

// -- In-Place Point Doubling (5x52) ----------------------------------------
// Same formula as jac52_double but overwrites the input point in-place.
// Eliminates the 128-byte return value copy on every call.
// Formula: libsecp256k1 gej_double (3M+4S+8 cheap, a=0)
// L = (3/2)*X^2, S = Y^2, T = -X*S, X3 = L^2+2T, Y3 = -(L*(X3+T)+S^2), Z3 = Y*Z
SECP256K1_HOT_FUNCTION __attribute__((always_inline))
// Core doubling on raw FE52 references (zero struct-copy overhead)
// In-place variant: reads and writes through same x/y/z references.
static inline void jac52_double_coords(FieldElement52& x, FieldElement52& y, FieldElement52& z) noexcept {
    // S = Y1^2 (1S) -- computed before y is reused
    FieldElement52 s = y.square();         // mag 1

    // L = (3/2)*X1^2 (1S + mul_int + half)
    FieldElement52 l = x.square();         // mag 1
    l.mul_int_assign(3);                     // mag 3
    l.half_assign();                         // mag 2

    // Z3 = Z1 * Y1 (1M in-place) -- y still holds Y1; z not read again after this
    z.mul_assign(y);                         // mag 1

    // T = -S * X1 (1N + 1M) -- reuse y as temp for T (y is free after Z3)
    y = s.negate(1);                         // mag 2
    y.mul_assign(x);                         // mag 1  (y = T)

    // X3 = L^2 + 2T (1S + 2A)
    x = l.square();                          // mag 1
    x.add_assign(y);                         // mag 2
    x.add_assign(y);                         // mag 3

    // S' = S^2 (1S)
    s.square_inplace();                      // mag 1

    // T' = T + X3
    y.add_assign(x);                         // mag 4  (y = T + X3)

    // Y3 = -(L*(T'+X3) + S^2) (1M in-place + 1A + 1N)
    y.mul_assign(l);                         // mag 1  (y = L*(T+X3))
    y.add_assign(s);                         // mag 2
    y.negate_assign(2);                      // mag 3
}

// Split I/O variant: reads from const in_*, writes to separate out_*.
// Eliminates the 120-byte input->output copy needed by the in-place variant
// when called from const methods (dbl, add) that return a new Point.
// __restrict__ tells the compiler that in_* and out_* never alias each other,
// enabling the same register-based optimizations as the copy-to-local approach.
static inline void jac52_double_to(
        const FieldElement52& __restrict__ in_x,
        const FieldElement52& __restrict__ in_y,
        const FieldElement52& __restrict__ in_z,
        FieldElement52& __restrict__ out_x,
        FieldElement52& __restrict__ out_y,
        FieldElement52& __restrict__ out_z) noexcept {
    // Z3 = Y1 * Z1 (1M)
    out_z = in_y * in_z;

    // S = Y1^2 (1S)
    FieldElement52 s = in_y.square();

    // L = (3/2)*X1^2 (1S + mul_int + half)
    FieldElement52 l = in_x.square();
    l.mul_int_assign(3);
    l.half_assign();

    // T = -S * X1 (1N + 1M)
    FieldElement52 t = s.negate(1);
    t.mul_assign(in_x);

    // X3 = L^2 + 2T (1S + 2A)
    out_x = l.square();
    out_x.add_assign(t);
    out_x.add_assign(t);

    // S' = S^2 (1S)
    s.square_inplace();

    // T' = T + X3
    t.add_assign(out_x);

    // Y3 = -(L*(X3+T) + S^2) (1M + 1A + 1N)
    out_y = t * l;
    out_y.add_assign(s);
    out_y.negate_assign(2);
}

// Z=1 specialization: when input Z == 1, skip Z3 = Y*Z multiplication (2M+4S).
// Saves 1M (~11ns on x86-64) for the first doubling of an affine-normalized point.
SECP256K1_HOT_FUNCTION __attribute__((always_inline))
static inline void jac52_double_z1_to(
        const FieldElement52& __restrict__ in_x,
        const FieldElement52& __restrict__ in_y,
        FieldElement52& __restrict__ out_x,
        FieldElement52& __restrict__ out_y,
        FieldElement52& __restrict__ out_z) noexcept {
    // Z3 = Y1 (Z1 == 1, skip multiplication)
    out_z = in_y;

    // S = Y1^2 (1S)
    FieldElement52 s = in_y.square();

    // L = (3/2)*X1^2 (1S + mul_int + half)
    FieldElement52 l = in_x.square();
    l.mul_int_assign(3);
    l.half_assign();

    // T = -S * X1 (1N + 1M)
    FieldElement52 t = s.negate(1);
    t.mul_assign(in_x);

    // X3 = L^2 + 2T (1S + 2A)
    out_x = l.square();
    out_x.add_assign(t);
    out_x.add_assign(t);

    // S' = S^2 (1S)
    s.square_inplace();

    // T' = T + X3
    t.add_assign(out_x);

    // Y3 = -(L*(X3+T) + S^2) (1M + 1A + 1N)
    out_y = t * l;
    out_y.add_assign(s);
    out_y.negate_assign(2);
}

SECP256K1_INLINE static void jac52_double_inplace(JacobianPoint52& p) noexcept {
    // Branchless infinity propagation (like libsecp's secp256k1_gej_double):
    // On secp256k1 (a=0), 2Q = infinity iff Q = infinity (no order-2 points).
    // If p.infinity == true, the formula runs on garbage X/Y/Z but p.infinity
    // stays true, so consumers correctly ignore the garbage coordinates.
    jac52_double_coords(p.x, p.y, p.z);
}

// Mixed Addition (5x52, return by value): delegates to in-place variant.
// Formula: h2-negation (libsecp-style, a=0 specialization), 8M+3S+~8A.
static void jac52_add_mixed_inplace(JacobianPoint52& p, const AffinePoint52& q) noexcept;  // fwd
[[maybe_unused]] SECP256K1_HOT_FUNCTION SECP256K1_NOINLINE
static JacobianPoint52 jac52_add_mixed(const JacobianPoint52& p, const AffinePoint52& q) noexcept {
    JacobianPoint52 r = {p.x, p.y, p.z, p.infinity};
    jac52_add_mixed_inplace(r, q);
    return r;
}

// -- In-Place Mixed Addition (5x52): Jacobian + Affine -> Jacobian -------------
// Formula: h2-negation (libsecp-style, a=0 specialization)
// Cost: 8M + 3S + ~8A.  Saves 1S vs madd-2007-bl at the cost of 1 extra M.
// Inlined into the hot loop: with the h2-negation formula, the function body
// is ~200 bytes of x86-64 code.  Two call sites at ~400 bytes added to the
// loop body keeps total I-cache footprint well within L1 (32 KB).
SECP256K1_HOT_FUNCTION SECP256K1_INLINE
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
    FieldElement52 const negX1 = p.x.negate(8);     // mag 9 (jac52_add x max: 7)
    FieldElement52 const h = u2 + negX1;                      // mag 6

    // Variable-time zero check (prob ~2^-256)
    if (SECP256K1_UNLIKELY(h.normalizes_to_zero_var())) {
        FieldElement52 const negY1 = p.y.negate(4);           // GEJ_Y_MAG_MAX=4
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

    // i = S1 - S2 = Y1 - S2 (not doubled -- saves 1S, 2A vs madd-2007-bl)
    FieldElement52 const negS2 = s2.negate(1);                // mag 2
    FieldElement52 const i_val = p.y + negS2;                 // mag 6

    // h2 = H^2 [1S], i2 = i^2 [1S] -- adjacent independent squares:
    // OoO engine can interleave their MULX micro-ops, so i2 is ready
    // by the time h3/t finish, removing it from the critical path to X3.
    FieldElement52 h2 = h.square();                           // mag 1
    FieldElement52 const i2 = i_val.square();                 // mag 1  (6*6*5 = 180 < 3.3M)
    // h2 negation trick: h3 and t carry the sign,
    // converting X3/Y3 subtractions into pure additions.
    h2.negate_assign(1);                                      // h2 = -H^2, mag 2

    // h3 = H * (-H^2) = -H^3 [1M]
    FieldElement52 h3 = h2 * h;                               // mag 1  (2*6 = 12 < 3.3M)

    // t = U1 * (-H^2) = -X1*H^2 [1M]  (p.x last read here)
    FieldElement52 t = p.x * h2;                              // mag 1  (4*2 = 8 < 3.3M)

    // X3 = i^2 + h3 + 2*t -- write directly to p.x (no temp)
    p.x = i2 + h3;                                            // mag 2
    p.x.add_assign(t);                                        // mag 3
    p.x.add_assign(t);                                        // mag 4

    // Y3 = i*(t + X3) + h3*S1 [2M + 2A]
    t.add_assign(p.x);                                        // t = t + X3, mag 5
    h3.mul_assign(p.y);                                       // h3 = (-H^3)*Y1, mag 1 (p.y last read)
    p.y = t * i_val;                                          // write directly to p.y
    p.y.add_assign(h3);                                       // Y3, mag 2

    p.infinity = false;
}

// Variant: adds (q.x, -q.y) — used for negative wNAF digits.
// Eliminates the 80-byte AffinePoint52 copy + negate_assign in apply_wnaf_mixed52.
// Formula identical to jac52_add_mixed_inplace; only two differences:
//   1. infinity case: result y = -q.y
//   2. i_val = Y1 + S2 (sign flipped, because adding negated Y2)
// Edge cases: h==0 with Y1 == -S2 → double; Y1 == S2 → infinity.
SECP256K1_HOT_FUNCTION SECP256K1_INLINE
static void jac52_add_mixed_neg_inplace(JacobianPoint52& p, const AffinePoint52& q) noexcept {
    if (SECP256K1_UNLIKELY(p.infinity)) {
        p.x = q.x; p.y = q.y.negate(1); p.z = FieldElement52::one(); p.infinity = false;
        return;
    }

    FieldElement52 const zz = p.z.square();
    FieldElement52 const u2 = q.x * zz;
    FieldElement52 s2 = q.y * zz;
    s2.mul_assign(p.z);                                        // S2 = q.y * Z1^3

    FieldElement52 const negX1 = p.x.negate(8);
    FieldElement52 const h = u2 + negX1;

    if (SECP256K1_UNLIKELY(h.normalizes_to_zero_var())) {
        // Doubling: p.y == -S2  (p == negated q)
        FieldElement52 const sum = p.y + s2;
        if (sum.normalizes_to_zero_var()) { jac52_double_inplace(p); return; }
        p = {FieldElement52::zero(), FieldElement52::one(), FieldElement52::zero(), true};
        return;
    }

    p.z.mul_assign(h);

    // i_val = Y1 - (-S2) = Y1 + S2  [sign flipped vs positive variant]
    FieldElement52 const i_val = p.y + s2;

    FieldElement52 h2 = h.square();
    FieldElement52 const i2 = i_val.square();
    h2.negate_assign(1);

    FieldElement52 h3 = h2 * h;
    FieldElement52 t = p.x * h2;

    p.x = i2 + h3;
    p.x.add_assign(t);
    p.x.add_assign(t);

    t.add_assign(p.x);
    h3.mul_assign(p.y);
    p.y = t * i_val;
    p.y.add_assign(h3);

    p.infinity = false;
}

// Split I/O variant: reads from const in_* and q_*, writes to out_*.
// Eliminates 320 bytes of struct-copy overhead in Point::add() const.
// __restrict__ tells the compiler that all references are non-aliasing.
static inline void jac52_add_mixed_to(
        const FieldElement52& __restrict__ in_x,
        const FieldElement52& __restrict__ in_y,
        const FieldElement52& __restrict__ in_z,
        bool in_infinity,
        const FieldElement52& __restrict__ q_x,
        const FieldElement52& __restrict__ q_y,
        FieldElement52& __restrict__ out_x,
        FieldElement52& __restrict__ out_y,
        FieldElement52& __restrict__ out_z,
        bool& out_infinity) noexcept {
    if (SECP256K1_UNLIKELY(in_infinity)) {
        out_x = q_x; out_y = q_y; out_z = FieldElement52::one();
        out_infinity = false;
        return;
    }

    FieldElement52 const zz = in_z.square();
    FieldElement52 const u2 = q_x * zz;
    FieldElement52 s2 = q_y * zz;
    s2.mul_assign(in_z);

    FieldElement52 const negX1 = in_x.negate(8);     // mag 9 (jac52_add x max: 7)
    FieldElement52 const h = u2 + negX1;                      // mag 6

    if (SECP256K1_UNLIKELY(h.normalizes_to_zero_var())) {
        FieldElement52 const negY1 = in_y.negate(4);          // GEJ_Y_MAG_MAX=4
        FieldElement52 const diff = s2 + negY1;
        if (diff.normalizes_to_zero_var()) {
            jac52_double_to(in_x, in_y, in_z, out_x, out_y, out_z);
            out_infinity = false;
            return;
        }
        out_x = FieldElement52::zero(); out_y = FieldElement52::one();
        out_z = FieldElement52::zero(); out_infinity = true;
        return;
    }

    out_z = in_z * h;

    // i = S1 - S2 = Y1 - S2
    FieldElement52 const negS2 = s2.negate(1);                // mag 2
    FieldElement52 const i_val = in_y + negS2;                // mag 6

    FieldElement52 h2 = h.square();
    FieldElement52 const i2 = i_val.square();
    h2.negate_assign(1);

    FieldElement52 h3 = h2 * h;
    FieldElement52 t = in_x * h2;

    out_x = i2 + h3;
    out_x.add_assign(t);
    out_x.add_assign(t);

    t.add_assign(out_x);
    out_y = t * i_val;                                   // i_val is i
    h3.mul_assign(in_y);
    out_y.add_assign(h3);

    out_infinity = false;
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

    FieldElement52 const negX1 = p.x.negate(8);     // mag 9 (jac52_add x max: 7)
    FieldElement52 const h = u2 + negX1;                      // mag 6

    if (SECP256K1_UNLIKELY(h.normalizes_to_zero_var())) {
        FieldElement52 const negY1 = p.y.negate(4);           // GEJ_Y_MAG_MAX=4
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
    zr_out = h;                                               // zr = H, mag 6

    // i = S1 - S2 = Y1 - S2
    FieldElement52 const negS2 = s2.negate(1);                // mag 2
    FieldElement52 const i_val = p.y + negS2;                 // mag 12

    FieldElement52 h2 = h.square();
    FieldElement52 const i2 = i_val.square();                  // ILP: adjacent independent square
    h2.negate_assign(1);                                      // h2 = -H^2, mag 2

    FieldElement52 h3 = h2 * h;                               // -H^3
    FieldElement52 t = p.x * h2;                              // -X1*H^2

    FieldElement52 x3 = i2 + h3;                              // mag 2
    x3.add_assign(t);                                         // mag 3
    x3.add_assign(t);                                         // mag 4

    t.add_assign(x3);                                         // t = t + X3, mag 5
    FieldElement52 y3 = t * i_val;                               // i_val is i
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
    FieldElement52 const negX1 = p.x.negate(8);     // mag 9 (jac52_add x max: 7)
    FieldElement52 const h = u2 + negX1;                      // mag 6

    // Variable-time zero check (prob ~2^-256)
    if (SECP256K1_UNLIKELY(h.normalizes_to_zero_var())) {
        FieldElement52 const negS2_chk = s2.negate(1);
        FieldElement52 const diff = p.y + negS2_chk;          // Y1 - S2
        if (diff.normalizes_to_zero_var()) {
            jac52_double_inplace(p);
            return;
        }
        p = {FieldElement52::zero(), FieldElement52::one(), FieldElement52::zero(), true};
        return;
    }

    // Z3 = Z1 * h  (uses ORIGINAL p.z, NOT az!)
    // In-place: p.z is not read again after this point.
    p.z.mul_assign(h);                                            // 1M

    // i = S1 - S2 = Y1 - S2
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

    // t = u1 * (-h^2) = -X1 * h^2  (p.x last read here)
    FieldElement52 t = p.x * h2;                                  // 1M

    // X3 = i^2 + h3 + 2*t  -- write directly to p.x (no temp)
    p.x = i_sq + h3;                                              // mag 2
    p.x.add_assign(t);                                            // mag 4
    p.x.add_assign(t);                                            // mag 5

    // Y3 = i*(t + X3) + s1*(-h^3)
    t.add_assign(p.x);                                            // t = (-u1*h^2) + X3
    h3.mul_assign(p.y);                                           // h3 = (-h^3)*Y1, 1M (p.y last read)
    p.y = t * i_val;                                              // write directly to p.y
    p.y.add_assign(h3);                                           // Y3 = i*(X3-u1*h^2) - Y1*h^3

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

// Full Jacobian Addition (5x52, return by value): delegates to in-place variant.
// Formula: add-2007-bl (a=0), 12M+5S+~11A.
SECP256K1_HOT_FUNCTION SECP256K1_NOINLINE
static JacobianPoint52 jac52_add(const JacobianPoint52& p, const JacobianPoint52& q) {
    JacobianPoint52 r = p;
    jac52_add_inplace(r, q);
    return r;
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

// -- GLV + Shamir helpers (shared by scalar_mul_glv52 / scalar_mul_with_plan_glv52 /
//    dual_scalar_mul_gen_point) -----------------------------------------------

// Builds odd-multiple table [1P, 3P, ..., (2T-1)P] in FE52 using the z-ratio
// technique (zero field inversions).  All table entries share an implied
// Z = globalz on the secp256k1 curve.
// Returns false if a degenerate case (infinity) occurred during table building.
static bool build_glv52_table_zr(
    const JacobianPoint52& P52,
    AffinePoint52* tbl,
    int table_size,
    FieldElement52& globalz)
{
    // d = 2*P (Jacobian)
    JacobianPoint52 const d = jac52_double(P52);
    FieldElement52 const C  = d.z;
    FieldElement52 const C2 = C.square();
    FieldElement52 const C3 = C2 * C;

    // d as affine on iso curve (Z cancels in isomorphism)
    AffinePoint52 const d_aff = {d.x, d.y};

    // Transform P onto iso curve: (P.X*C^2, P.Y*C^3, P.Z)
    JacobianPoint52 ai = {P52.x * C2, P52.y * C3, P52.z, false};
    tbl[0].x = ai.x;
    tbl[0].y = ai.y;

    // Z-ratio array: zr[i] = Z_i / Z_{i-1} for the accumulator
    constexpr int kMaxZr = 32;
    assert(table_size <= kMaxZr);
    FieldElement52 zr[kMaxZr];
    zr[0] = C;  // first z-ratio is C (from iso mapping)

    // Build rest using mixed adds on iso curve with z-ratio output
    for (int i = 1; i < table_size; i++) {
        jac52_add_mixed_inplace_zr(ai, d_aff, zr[i]);
        if (SECP256K1_UNLIKELY(ai.infinity)) {
            return false;
        }
        tbl[i].x = ai.x;
        tbl[i].y = ai.y;
    }

    // globalz = final_Z * C (maps from iso curve back to secp256k1)
    globalz = ai.z * C;

    // Backward sweep: rescale table entries so all share implied Z = Z_last.
    // zs accumulates the product zr[n-1] * ... * zr[i+1] = Z_last / Z_i.
    // Each entry is scaled: x *= zs^2, y *= zs^3.
    {
        FieldElement52 zs = zr[table_size - 1];
        for (int idx = table_size - 2; idx >= 0; --idx) {
            if (idx != table_size - 2) {
                zs.mul_assign(zr[idx + 1]);
            }
            FieldElement52 const zs2 = zs.square();
            FieldElement52 const zs3 = zs2 * zs;
            tbl[idx].x.mul_assign(zs2);
            tbl[idx].y.mul_assign(zs3);
        }
    }

    // Normalize all entries to magnitude 1.
    for (int i = 0; i < table_size; ++i) {
        tbl[i].x.normalize_weak();
        tbl[i].y.normalize_weak();
    }

    return true;
}

// Derives phi(P) table: phi(x,y) = (beta*x, y) with optional Y negation.
static void derive_phi52_table(
    const AffinePoint52* tbl_P,
    AffinePoint52* tbl_phiP,
    int table_size,
    bool flip_phi)
{
    for (int i = 0; i < table_size; i++) {
        tbl_phiP[i].x = tbl_P[i].x * kBeta52_pt;
        if (flip_phi) {
            // negate(1) gives magnitude 1 — normalize_weak() is redundant.
            tbl_phiP[i].y = tbl_P[i].y.negate(1);
        } else {
            tbl_phiP[i].y = tbl_P[i].y;
        }
    }
}

// wNAF digit application: lookup table entry, negate if negative, mixed-add.
// Called for all digit positions including d==0 (no-op). No hints needed —
// CPU branch predictor handles the 33%/67% non-zero/zero split well.
static inline void apply_wnaf_mixed52(
    JacobianPoint52& result, const AffinePoint52* table, int32_t d)
{
    if (d != 0) {
        if (d > 0) {
            jac52_add_mixed_inplace(result, table[static_cast<std::size_t>((d - 1) >> 1)]);
        } else {
            jac52_add_mixed_neg_inplace(result, table[static_cast<std::size_t>((-d - 1) >> 1)]);
        }
    }
}

// 2-stream Shamir's trick: single doubling chain with dual wNAF lookups.
// Returns the accumulated Jacobian result (caller must apply globalz correction).
static JacobianPoint52 shamir_2stream_glv52(
    const AffinePoint52* tbl_P,
    const AffinePoint52* tbl_phiP,
    const int32_t* wnaf1, std::size_t wnaf1_len,
    const int32_t* wnaf2, std::size_t wnaf2_len)
{
    JacobianPoint52 result52 = {
        FieldElement52::zero(), FieldElement52::one(),
        FieldElement52::zero(), true
    };

    const std::size_t max_len = (wnaf1_len > wnaf2_len) ? wnaf1_len : wnaf2_len;

    for (int i = static_cast<int>(max_len) - 1; i >= 0; --i) {
        jac52_double_inplace(result52);
        apply_wnaf_mixed52(result52, tbl_P, wnaf1[static_cast<std::size_t>(i)]);
        apply_wnaf_mixed52(result52, tbl_phiP, wnaf2[static_cast<std::size_t>(i)]);
    }

    return result52;
}

// 4-stream Shamir's trick: 4 independent accumulators share the same wNAF
// digit sequence. OOO execution pipelines the 4 independent add chains.
// All 4 pairs of tables must have the same size (matching glv_window).
static void shamir_4stream_glv52(
    const AffinePoint52* tbl_P0,   const AffinePoint52* tbl_phiP0,
    const AffinePoint52* tbl_P1,   const AffinePoint52* tbl_phiP1,
    const AffinePoint52* tbl_P2,   const AffinePoint52* tbl_phiP2,
    const AffinePoint52* tbl_P3,   const AffinePoint52* tbl_phiP3,
    const int32_t* wnaf1, std::size_t wnaf1_len,
    const int32_t* wnaf2, std::size_t wnaf2_len,
    JacobianPoint52 out[4])
{
    const JacobianPoint52 inf52 = {
        FieldElement52::zero(), FieldElement52::one(),
        FieldElement52::zero(), true
    };
    out[0] = out[1] = out[2] = out[3] = inf52;

    const std::size_t max_len = (wnaf1_len > wnaf2_len) ? wnaf1_len : wnaf2_len;

    for (int i = static_cast<int>(max_len) - 1; i >= 0; --i) {
        jac52_double_inplace(out[0]);
        jac52_double_inplace(out[1]);
        jac52_double_inplace(out[2]);
        jac52_double_inplace(out[3]);

        const int32_t d1 = wnaf1[static_cast<std::size_t>(i)];
        const int32_t d2 = wnaf2[static_cast<std::size_t>(i)];

        apply_wnaf_mixed52(out[0], tbl_P0, d1);
        apply_wnaf_mixed52(out[1], tbl_P1, d1);
        apply_wnaf_mixed52(out[2], tbl_P2, d1);
        apply_wnaf_mixed52(out[3], tbl_P3, d1);

        apply_wnaf_mixed52(out[0], tbl_phiP0, d2);
        apply_wnaf_mixed52(out[1], tbl_phiP1, d2);
        apply_wnaf_mixed52(out[2], tbl_phiP2, d2);
        apply_wnaf_mixed52(out[3], tbl_phiP3, d2);
    }
}

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

    // GLV half-scalars bounded by ~2^128; max wNAF output = 128+window = 133.
    // 140 gives safety margin (was 260 — 2× over-provisioned saves 480B stack).
    // No {} init: compute_wnaf_into() memsets before writing (avoids double-zero).
    std::array<int32_t, 140> wnaf1_buf, wnaf2_buf;
    std::size_t wnaf1_len = 0, wnaf2_len = 0;
    compute_wnaf_into(decomp.k1, glv_window,
                      wnaf1_buf.data(), wnaf1_buf.size(), wnaf1_len);
    compute_wnaf_into(decomp.k2, glv_window,
                      wnaf2_buf.data(), wnaf2_buf.size(), wnaf2_len);

    // Trim trailing zeros -- GLV half-scalars are ~128 bits but wNAF
    // always outputs 256+ positions. This halves the doubling count.
    // compute_wnaf_into already sets out_len = last_set_bit + 1, excluding
    // trailing zeros. The trim loops below are no-ops; removed (A-13).

    // -- Precompute odd multiples [1P, 3P, 5P, ..., 15P] in 5x52 ------
    std::array<AffinePoint52, glv_table_size> tbl_P;
    std::array<AffinePoint52, glv_table_size> tbl_phiP;
    FieldElement52 globalz;

    if (!build_glv52_table_zr(P52, tbl_P.data(), glv_table_size, globalz)) {
        return scalar_mul_fallback_4x64(base, scalar);
    }

    // -- Derive phi(P) table + Shamir's trick -------------------------
    const bool flip_phi = (decomp.k1_neg != decomp.k2_neg);
    derive_phi52_table(tbl_P.data(), tbl_phiP.data(), glv_table_size, flip_phi);

    JacobianPoint52 result52 = shamir_2stream_glv52(
        tbl_P.data(), tbl_phiP.data(),
        wnaf1_buf.data(), wnaf1_len, wnaf2_buf.data(), wnaf2_len);

    if (!result52.infinity) {
        result52.z.mul_assign(globalz);
    }

    return from_jac52(result52);
}

#endif // SECP256K1_FAST_52BIT

// NOTE: batch_to_affine / batch_to_affine_into removed — they were dead code
// ([[maybe_unused]], never called) with a Montgomery backward-pass bug.
// The actual hot path (init_gen_fb_table) does batch inversion inline correctly.

} // namespace

// KPlan implementation: Cache all K-dependent work for Fixed K x Variable Q
#if !defined(SECP256K1_PLATFORM_ESP32) && !defined(ESP_PLATFORM) && !defined(SECP256K1_PLATFORM_STM32)
KPlan KPlan::from_scalar(const Scalar& k, uint8_t w) {
    // Step 1: GLV decomposition (K -> k1, k2, signs)
    auto decomp = split_scalar_glv(k);

    // Step 2: Compute wNAF directly into the plan's fixed-size stack buffers.
    // compute_wnaf_into uses a caller-supplied array; no heap allocation.
    KPlan plan;
    plan.window_width = w;
    plan.k1 = decomp.k1;
    plan.k2 = decomp.k2;
    plan.neg1 = decomp.neg1;
    plan.neg2 = decomp.neg2;
    compute_wnaf_into(decomp.k1, w, plan.wnaf1.data(), plan.wnaf1.size(), plan.wnaf1_len);
    compute_wnaf_into(decomp.k2, w, plan.wnaf2.data(), plan.wnaf2.size(), plan.wnaf2_len);
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
    // Infinity flag is authoritative — all Jacobian operations maintain it
    // correctly.  Skipping the old z.is_zero() full-normalize saves ~15ns
    // per ecmult/scalar_mul (Z is never zero when infinity==false).
    if (infinity) return Point::infinity();
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
    SECP_ASSERT_NORMALIZED(x);
    SECP_ASSERT_NORMALIZED(y);
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
    FieldElement52 const z_inv = z_.inverse_safegcd();
    if (SECP256K1_UNLIKELY(z_inv.normalizes_to_zero_var())) return FieldElement::zero(); // LCOV_EXCL_LINE
    return (x_ * z_inv.square()).to_fe();
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
    FieldElement52 const z_inv = z_.inverse_safegcd();
    if (SECP256K1_UNLIKELY(z_inv.normalizes_to_zero_var())) return FieldElement::zero(); // LCOV_EXCL_LINE
    FieldElement52 const z_inv2 = z_inv.square();
    return (y_ * (z_inv2 * z_inv)).to_fe();
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
    SECP_ASSERT_ON_CURVE(*this);
    SECP_ASSERT_ON_CURVE(other);
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
    // Mixed-add fast path: other is affine (z=1) -> 8M+3S instead of 12M+5S
    // NRVO: write directly to result Point's members, zero intermediate copies.
    if (!other.infinity_ && (other.z_one_ || fe52_is_one_raw(other.z_))) {
        Point r;
        r.z_one_ = false;
        r.is_generator_ = false;
        jac52_add_mixed_to(x_, y_, z_, infinity_,
                           other.x_, other.y_,
                           r.x_, r.y_, r.z_, r.infinity_);
        return r;
    }
    // Symmetric: this is affine, other is Jacobian -> swap roles
    if (!infinity_ && (z_one_ || fe52_is_one_raw(z_))) {
        Point r;
        r.z_one_ = false;
        r.is_generator_ = false;
        jac52_add_mixed_to(other.x_, other.y_, other.z_, other.infinity_,
                           x_, y_,
                           r.x_, r.y_, r.z_, r.infinity_);
        return r;
    }
    // Full Jacobian + Jacobian (12M+5S)
    JacobianPoint52 const p52{x_, y_, z_, infinity_};
    JacobianPoint52 const q52{other.x_, other.y_, other.z_, other.infinity_};
    JacobianPoint52 const r52 = jac52_add(p52, q52);
    return Point(r52.x, r52.y, r52.z, r52.infinity, false);
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
    SECP_ASSERT_ON_CURVE(*this);
#if defined(SECP256K1_FAST_52BIT)
    if (SECP256K1_UNLIKELY(infinity_)) {
        return Point(x_, y_, z_, true, false);
    }
    // NRVO: r is constructed directly in the caller's return slot.
    // jac52_double writes directly to r.x_/y_/z_ -- zero intermediate copies.
    Point r;
    r.infinity_ = false;
    r.is_generator_ = false;
    r.z_one_ = false;
    if (z_one_) {
        jac52_double_z1_to(x_, y_, r.x_, r.y_, r.z_);
    } else {
        jac52_double_to(x_, y_, z_, r.x_, r.y_, r.z_);
    }
    return r;
#else
    JacobianPoint p{x_, y_, z_, infinity_};
    JacobianPoint r = jacobian_double(p);
    Point result = from_jacobian_coords(r.x, r.y, r.z, r.infinity);
    result.is_generator_ = false;
    return result;
#endif
}

Point Point::negate() const {
    SECP_ASSERT_ON_CURVE(*this);
    if (infinity_) {
        return *this;  // Infinity is its own negation
    }
#if defined(SECP256K1_FAST_52BIT)
    FieldElement52 neg_y = y_;
    neg_y.normalize_weak();
    neg_y.negate_assign(1);
    neg_y.normalize_weak();
    Point result(x_, neg_y, z_, false, false);
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

// -- Shared helper: add generator-like affine point (parameterized by Y) ------
static Point add_gen_mixed(const Point& self,
                           const FieldElement& gen_y_4x64) {
#if defined(SECP256K1_FAST_52BIT)
  #if defined(SECP256K1_USE_4X64_POINT_OPS)
    JacobianPoint p = point_to_jac4x64(self.X52(), self.Y52(), self.Z52(), self.is_infinity());
    AffinePoint const g{kGeneratorX, gen_y_4x64};
    JacobianPoint const r = jacobian_add_mixed(p, g);
    return Point::from_jacobian52(
        FieldElement52::from_fe(r.x), FieldElement52::from_fe(r.y),
        FieldElement52::from_fe(r.z), r.infinity);
  #else
    static const FieldElement52 kGenX52 = FieldElement52::from_fe(kGeneratorX);
    FieldElement52 const gen_y_52 = FieldElement52::from_fe(gen_y_4x64);
    JacobianPoint52 p52{self.X52(), self.Y52(), self.Z52(), self.is_infinity()};
    AffinePoint52 const g52{kGenX52, gen_y_52};
    jac52_add_mixed_inplace(p52, g52);
    return Point::from_jacobian52(p52.x, p52.y, p52.z, p52.infinity);
  #endif
#else
    JacobianPoint p{self.X(), self.Y(), self.z(), self.is_infinity()};
    AffinePoint const g{kGeneratorX, gen_y_4x64};
    JacobianPoint const r = jacobian_add_mixed(p, g);
    return Point::from_jacobian_coords(r.x, r.y, r.z, r.infinity);
#endif
}

Point Point::next() const {
    return add_gen_mixed(*this, kGeneratorY);
}

Point Point::prev() const {
    return add_gen_mixed(*this, kNegGeneratorY);
}

// Mutable in-place addition: *this += G (no allocation overhead)
void Point::next_inplace() {
    z_one_ = false;
    if (infinity_) {
        *this = Point::generator();
        return;
    }
    *this = add_gen_mixed(*this, kGeneratorY);
}

// Mutable in-place subtraction: *this -= G (no allocation overhead)
void Point::prev_inplace() {
    z_one_ = false;
    if (infinity_) {
        *this = Point::generator().negate();
        return;
    }
    *this = add_gen_mixed(*this, kNegGeneratorY);
}

// Mutable in-place addition: *this += other (no allocation overhead)
// Routes through 5x52 path on x64 for inlined field ops (zero call overhead)
void Point::add_inplace(const Point& other) {
    SECP_ASSERT_ON_CURVE(*this);
    SECP_ASSERT_ON_CURVE(other);
    z_one_ = false;
#if defined(SECP256K1_FAST_52BIT)
  #if defined(SECP256K1_USE_4X64_POINT_OPS)
    // Dual representation: FE52 storage -> 4x64 compute -> FE52 storage
    if (!other.infinity_ && fe52_is_one_raw(other.z_)) {
        JacobianPoint p = point_to_jac4x64(x_, y_, z_, infinity_);
        AffinePoint const q = fe52_to_affine4x64(other.x_, other.y_);
        jacobian_add_mixed_inplace(p, q);
        jac4x64_to_fe52(p, x_, y_, z_, infinity_);
        is_generator_ = false;
        return;
    }
    JacobianPoint p = point_to_jac4x64(x_, y_, z_, infinity_);
    JacobianPoint const q = point_to_jac4x64(other.x_, other.y_, other.z_, other.infinity_);
    jacobian_add_inplace(p, q);
    jac4x64_to_fe52(p, x_, y_, z_, infinity_);
    is_generator_ = false;
  #else
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
  #endif
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
  #if defined(SECP256K1_USE_4X64_POINT_OPS)
    JacobianPoint p = point_to_jac4x64(x_, y_, z_, infinity_);
    JacobianPoint q = point_to_jac4x64(other.x_, other.y_, other.z_, other.infinity_);
    q.y = FieldElement::zero() - q.y;
    jacobian_add_inplace(p, q);
    jac4x64_to_fe52(p, x_, y_, z_, infinity_);
    is_generator_ = false;
  #else
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
  #endif
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
    SECP_ASSERT_ON_CURVE(*this);
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
    // 5x52 path: operate directly on FE52 members (zero struct-copy overhead)
    if (SECP256K1_UNLIKELY(infinity_)) return;
    jac52_double_coords(x_, y_, z_);
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
    SECP_ASSERT_ON_CURVE(*this);
    // Negation in Jacobian: (X, Y, Z) -> (X, -Y, Z)
#if defined(SECP256K1_FAST_52BIT)
  #if defined(SECP256K1_USE_4X64_POINT_OPS)
    FieldElement fy = y_.to_fe();
    fy = FieldElement::zero() - fy;
    y_ = FieldElement52::from_fe(fy);
  #else
    y_.normalize_weak();
    y_.negate_assign(1);
    y_.normalize_weak();
  #endif
#else
    y_ = FieldElement::zero() - y_;
#endif
}

// Explicit mixed-add with affine input: this += (ax, ay)
// Routes through 5x52 path on x64 for inlined field ops
void Point::add_mixed_inplace(const FieldElement& ax, const FieldElement& ay) {
    SECP_ASSERT_NORMALIZED(ax);
    SECP_ASSERT_NORMALIZED(ay);
    z_one_ = false;
#if defined(SECP256K1_FAST_52BIT)
  #if defined(SECP256K1_USE_4X64_POINT_OPS)
    if (SECP256K1_UNLIKELY(infinity_)) {
        x_ = FieldElement52::from_fe(ax);
        y_ = FieldElement52::from_fe(ay);
        z_ = FieldElement52::one();
        infinity_ = false;
        is_generator_ = false;
        return;
    }
    JacobianPoint p = point_to_jac4x64(x_, y_, z_, infinity_);
    AffinePoint const q{ax, ay};
    jacobian_add_mixed_inplace(p, q);
    jac4x64_to_fe52(p, x_, y_, z_, infinity_);
    is_generator_ = false;
  #else
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
  #endif
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

    // Operate directly on Point members to eliminate 320B of struct copies.
    // Same algorithm as jac52_add_mixed_inplace but on x_/y_/z_ directly.

    FieldElement52 const zz = z_.square();                    // Z1^2 [1S]
    FieldElement52 const u2 = ax * zz;                        // X2*Z1^2 [1M]
    FieldElement52 s2 = ay * zz;                              // Y2*Z1^2
    s2.mul_assign(z_);                                        // S2 = Y2*Z1^3 [1M]

    FieldElement52 const negX1 = x_.negate(8);     // mag 9 (jac52_add x max: 7)
    FieldElement52 const h = u2 + negX1;                      // H = U2-X1, mag 6

    if (SECP256K1_UNLIKELY(h.normalizes_to_zero_var())) {
        FieldElement52 const negY1 = y_.negate(4);            // GEJ_Y_MAG_MAX=4
        FieldElement52 const diff = s2 + negY1;
        if (diff.normalizes_to_zero_var()) {
            jac52_double_coords(x_, y_, z_);
            infinity_ = false; is_generator_ = false; return;
        }
        x_ = FieldElement52::zero(); y_ = FieldElement52::one();
        z_ = FieldElement52::zero(); infinity_ = true;
        is_generator_ = false; return;
    }

    z_.mul_assign(h);                                         // Z3 = Z1*H [1M]

    FieldElement52 const negS2 = s2.negate(1);                // mag 2
    FieldElement52 const i_val = y_ + negS2;                  // i = Y1-S2, mag 6

    FieldElement52 h2 = h.square();                           // H^2 [1S]
    FieldElement52 const i2 = i_val.square();                 // i^2 [1S]
    h2.negate_assign(1);                                      // -H^2, mag 2

    FieldElement52 h3 = h2 * h;                               // -H^3 [1M]
    FieldElement52 t = x_ * h2;                               // -X1*H^2 [1M]

    x_ = i2 + h3;                                            // X3 partial
    x_.add_assign(t);
    x_.add_assign(t);                                         // X3 = i^2 - H^3 - 2*X1*H^2

    t.add_assign(x_);                                         // t = -X1*H^2 + X3
    h3.mul_assign(y_);                                        // -H^3*Y1 [1M] (y_ still Y1!)
    y_ = t * i_val;                                           // i*(t+X3) [1M]
    y_.add_assign(h3);                                        // Y3

    infinity_ = false;
    is_generator_ = false;
}
// FE52-native mixed-add with negated Y: this += (ax, -ay).
// Mirrors add_mixed52_inplace but flips the S2 sign so no copy+negate is needed
// for negative wNAF digits (saves 40-byte FE52 copy per negative digit in hot loops).
void Point::add_mixed52_neg_inplace(const FieldElement52& ax, const FieldElement52& ay) {
    z_one_ = false;
    if (SECP256K1_UNLIKELY(infinity_)) {
        x_ = ax;
        y_ = ay.negate(1);   // result y = -ay
        z_ = FieldElement52::one();
        infinity_ = false;
        is_generator_ = false;
        return;
    }

    FieldElement52 const zz = z_.square();
    FieldElement52 const u2 = ax * zz;
    FieldElement52 s2 = ay * zz;
    s2.mul_assign(z_);                                         // S2 = ay*Z1^3

    FieldElement52 const negX1 = x_.negate(8);
    FieldElement52 const h = u2 + negX1;

    if (SECP256K1_UNLIKELY(h.normalizes_to_zero_var())) {
        // Adding -ay: doubling when Y1 == -S2, i.e. Y1 + S2 == 0
        FieldElement52 const sum = y_ + s2;
        if (sum.normalizes_to_zero_var()) {
            jac52_double_coords(x_, y_, z_);
            infinity_ = false; is_generator_ = false; return;
        }
        x_ = FieldElement52::zero(); y_ = FieldElement52::one();
        z_ = FieldElement52::zero(); infinity_ = true;
        is_generator_ = false; return;
    }

    z_.mul_assign(h);

    // i_val = Y1 - (-S2) = Y1 + S2  (sign flip vs positive add)
    FieldElement52 const i_val = y_ + s2;

    FieldElement52 h2 = h.square();
    FieldElement52 const i2 = i_val.square();
    h2.negate_assign(1);

    FieldElement52 h3 = h2 * h;
    FieldElement52 t = x_ * h2;

    x_ = i2 + h3;
    x_.add_assign(t);
    x_.add_assign(t);

    t.add_assign(x_);
    h3.mul_assign(y_);
    y_ = t * i_val;
    y_.add_assign(h3);

    infinity_ = false;
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
static std::once_flag gen_fb_once;

static void init_gen_fb_table() {

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
}

inline std::uint8_t get_nybble(const Scalar& s, int pos) {
    int limb = pos / 16;
    int shift = (pos % 16) * 4;
    return static_cast<std::uint8_t>((s.limbs()[limb] >> shift) & 0xFu);
}

static Point gen_fixed_mul(const Scalar& k) {
    std::call_once(gen_fb_once, init_gen_fb_table);

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
    SECP_ASSERT_ON_CURVE(*this);
#if !defined(SECP256K1_PLATFORM_ESP32) && !defined(ESP_PLATFORM) && !defined(SECP256K1_PLATFORM_STM32) && !defined(__EMSCRIPTEN__)
    // WASM: precompute tables produce incorrect results under Emscripten's
    // __int128 emulation (field ops are fine, but the windowed accumulation
    // path diverges for some scalars).  Use the proven double-and-add
    // fallback instead -- correct and acceptable perf for WASM.
    if (is_generator_) {
        Point r = scalar_mul_generator(scalar);
        r.normalize();   // ensure affine (z=1) for O(1) serialization
        return r;
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
    // GLV half-scalars bounded by ~2^128; max wNAF output = 128+window = 133.
    // 140 gives safety margin (was 260 — 2× over-provisioned saves 480B stack).
    // No {} init: compute_wnaf_into() memsets before writing (avoids double-zero).
    std::array<int32_t, 140> wnaf1_buf, wnaf2_buf;
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

// scalar_mul_jacobian: same as scalar_mul() but skips the final normalize().
// Result stays in Jacobian (Z≠1). Feed a batch into batch_to_compressed() or
// batch_x_only_bytes() to amortise field inversion across N points.
Point Point::scalar_mul_jacobian(const Scalar& scalar) const {
    SECP_ASSERT_ON_CURVE(*this);
    if (SECP256K1_UNLIKELY(is_infinity() || scalar.is_zero()))
        return Point::infinity();
#if !defined(SECP256K1_PLATFORM_ESP32) && !defined(ESP_PLATFORM) && \
    !defined(SECP256K1_PLATFORM_STM32) && !defined(__EMSCRIPTEN__)
    if (is_generator_)
        return scalar_mul_generator(scalar);  // precomputed fixed-base → Jacobian, no normalize
#   if defined(SECP256K1_FAST_52BIT)
    return scalar_mul_glv52(*this, scalar);   // GLV Shamir → Jacobian, skip normalize
#   else
    // Non-FE52 (WASM/MSVC): binary double-and-add, skip final normalize
    auto scalar_bytes = scalar.to_bytes();
    Point result = Point::infinity();
    Point base = *this;
    for (int byte_idx = 31; byte_idx >= 0; --byte_idx) {
        uint8_t bv = scalar_bytes[static_cast<std::size_t>(byte_idx)];
        for (int bit = 0; bit < 8; ++bit) {
            if ((bv >> bit) & 1) result.add_inplace(base);
            base.dbl_inplace();
        }
    }
    return result;  // Jacobian, no normalize
#   endif
#else
    // Embedded targets: no batch-friendly path, fall back to affine output
    return scalar_mul(scalar);
#endif
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
        // Use stack buffers — no heap allocation.
        constexpr unsigned window_width = 4;
        std::array<int32_t, 140> wnaf1_stk{}, wnaf2_stk{}; // GLV half-scalars ≤128 bits
        std::size_t len1 = 0, len2 = 0;
        compute_wnaf_into(k1, window_width, wnaf1_stk.data(), wnaf1_stk.size(), len1);
        compute_wnaf_into(k2, window_width, wnaf2_stk.data(), wnaf2_stk.size(), len2);
        return scalar_mul_precomputed_wnaf(wnaf1_stk.data(), len1,
                                           wnaf2_stk.data(), len2, false, false);
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

// Vector overload — delegates to raw-pointer implementation (zero extra alloc).
Point Point::scalar_mul_precomputed_wnaf(const std::vector<int32_t>& wnaf1,
                                          const std::vector<int32_t>& wnaf2,
                                          bool neg1, bool neg2) const {
    return scalar_mul_precomputed_wnaf(wnaf1.data(), wnaf1.size(),
                                        wnaf2.data(), wnaf2.size(),
                                        neg1, neg2);
}

// Step 3: Shamir's trick with precomputed wNAF
// All K-related work (decomposition, wNAF) is done once
// Runtime only: phi(Q), tables, interleaved double-and-add
Point Point::scalar_mul_precomputed_wnaf(const int32_t* wnaf1, std::size_t len1,
                                          const int32_t* wnaf2, std::size_t len2,
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
    for (std::size_t di = 0; di < len1; ++di) {
        int const abs_d = (wnaf1[di] < 0) ? -wnaf1[di] : wnaf1[di];
        if (abs_d > max_digit) max_digit = abs_d;
    }
    for (std::size_t di = 0; di < len2; ++di) {
        int const abs_d = (wnaf2[di] < 0) ? -wnaf2[di] : wnaf2[di];
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
    if (neg1) {
        std::swap(table_Q_jac, neg_table_Q_jac);
    }
    if (neg2) {
        std::swap(table_phi_Q_jac, neg_table_phi_Q_jac);
    }

    // Shamir's trick: process both wNAF streams simultaneously (inplace ops)
    JacobianPoint result = {FieldElement::zero(), FieldElement::one(), FieldElement::zero(), true};
    
    std::size_t const max_len = std::max(len1, len2);
    
    for (int i = static_cast<int>(max_len) - 1; i >= 0; --i) {
        jacobian_double_inplace(result);
        
        // Add phi(Q) * k2 contribution (no copy: pre-negated tables)
        if (i < static_cast<int>(len2)) {
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
        if (i < static_cast<int>(len1)) {
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
        return base.scalar_mul_precomputed_wnaf(
            plan.wnaf1.data(), plan.wnaf1_len,
            plan.wnaf2.data(), plan.wnaf2_len,
            plan.neg1, plan.neg2);
    }

    // -- Use wNAF directly from plan's fixed-size arrays (no heap copy needed) ---
    // plan.wnaf1/wnaf2 are already stack arrays populated by KPlan::from_scalar.
    std::size_t wnaf1_len = plan.wnaf1_len;
    std::size_t wnaf2_len = plan.wnaf2_len;
    const int32_t* wnaf1_ptr = plan.wnaf1.data();
    const int32_t* wnaf2_ptr = plan.wnaf2.data();

    // Trim trailing zeros -- GLV half-scalars are ~128 bits but wNAF
    // always outputs 256+ positions.  This halves the doubling count.
    while (wnaf1_len > 0 && wnaf1_ptr[wnaf1_len - 1] == 0) --wnaf1_len;
    while (wnaf2_len > 0 && wnaf2_ptr[wnaf2_len - 1] == 0) --wnaf2_len;

    // -- Precompute odd multiples [1P, 3P, ..., (2T-1)P] in 5x52 -----
    std::array<AffinePoint52, kMaxGlvTableSize> tbl_P;
    std::array<AffinePoint52, kMaxGlvTableSize> tbl_phiP;
    FieldElement52 globalz;

    if (!build_glv52_table_zr(P52, tbl_P.data(), glv_table_size, globalz)) {
        return base.scalar_mul_precomputed_wnaf(
            plan.wnaf1.data(), plan.wnaf1_len,
            plan.wnaf2.data(), plan.wnaf2_len,
            plan.neg1, plan.neg2);
    }

    // -- Derive phi(P) table + Shamir's trick -------------------------
    const bool flip_phi = (plan.neg1 != plan.neg2);
    derive_phi52_table(tbl_P.data(), tbl_phiP.data(), glv_table_size, flip_phi);

    JacobianPoint52 result52 = shamir_2stream_glv52(
        tbl_P.data(), tbl_phiP.data(),
        wnaf1_ptr, wnaf1_len, wnaf2_ptr, wnaf2_len);

    if (!result52.infinity) {
        result52.z.mul_assign(globalz);
    }

    return from_jac52(result52);
}

// 4× variant: same KPlan applied to 4 base points simultaneously.
// Builds 4 sets of tables then runs a shared wNAF digit loop (shamir_4stream_glv52).
// Falls back to 4× scalar_mul_with_plan_glv52 on any degenerate input.
SECP256K1_NOINLINE SECP256K1_NO_STACK_PROTECTOR
static void scalar_mul_with_plan_glv52_4x(
    const Point bases[4], const KPlan& plan, Point results[4])
{
    const unsigned glv_window = plan.window_width;
    constexpr unsigned kMaxGlvWindow    = 7;
    constexpr int      kMaxGlvTableSize = 1 << (kMaxGlvWindow - 2);  // 32
    const int glv_table_size = 1 << (glv_window - 2);

    if (SECP256K1_UNLIKELY(glv_table_size > kMaxGlvTableSize || glv_window < 3)) {
        for (int k = 0; k < 4; ++k)
            results[k] = scalar_mul_with_plan_glv52(bases[k], plan);
        return;
    }

    std::size_t wnaf1_len = plan.wnaf1_len;
    std::size_t wnaf2_len = plan.wnaf2_len;
    const int32_t* wnaf1_ptr = plan.wnaf1.data();
    const int32_t* wnaf2_ptr = plan.wnaf2.data();
    while (wnaf1_len > 0 && wnaf1_ptr[wnaf1_len - 1] == 0) --wnaf1_len;
    while (wnaf2_len > 0 && wnaf2_ptr[wnaf2_len - 1] == 0) --wnaf2_len;

    const bool flip_phi = (plan.neg1 != plan.neg2);

    // Per-point tables: 4 × (tbl_P + tbl_phiP)
    std::array<AffinePoint52, kMaxGlvTableSize> tbl_P[4];
    std::array<AffinePoint52, kMaxGlvTableSize> tbl_phiP[4];
    FieldElement52 globalz[4];

    for (int k = 0; k < 4; ++k) {
        if (SECP256K1_UNLIKELY(bases[k].is_infinity())) {
            // Fall back entirely — rare in BIP-352 but must be correct
            for (int j = 0; j < 4; ++j)
                results[j] = scalar_mul_with_plan_glv52(bases[j], plan);
            return;
        }
        JacobianPoint52 const P52 = plan.neg1
            ? to_jac52(bases[k].negate())
            : to_jac52(bases[k]);

        if (!build_glv52_table_zr(P52, tbl_P[k].data(), glv_table_size, globalz[k])) {
            for (int j = 0; j < 4; ++j)
                results[j] = scalar_mul_with_plan_glv52(bases[j], plan);
            return;
        }
        derive_phi52_table(tbl_P[k].data(), tbl_phiP[k].data(), glv_table_size, flip_phi);
    }

    JacobianPoint52 acc[4];
    shamir_4stream_glv52(
        tbl_P[0].data(), tbl_phiP[0].data(),
        tbl_P[1].data(), tbl_phiP[1].data(),
        tbl_P[2].data(), tbl_phiP[2].data(),
        tbl_P[3].data(), tbl_phiP[3].data(),
        wnaf1_ptr, wnaf1_len, wnaf2_ptr, wnaf2_len,
        acc);

    for (int k = 0; k < 4; ++k) {
        if (!acc[k].infinity)
            acc[k].z.mul_assign(globalz[k]);
        results[k] = from_jac52(acc[k]);
    }
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
    // FE52 fast path: keep the Jacobian result lazy-affine.
    // Most callers compare points or continue arithmetic and do not require
    // eager normalization; serialization/x()/y() already normalize on demand.
    return scalar_mul_with_plan_glv52(*this, plan);
#else
    // Legacy 4x64 path: keep the Jacobian result lazy-affine here too.
    return scalar_mul_precomputed_wnaf(
        plan.wnaf1.data(), plan.wnaf1_len,
        plan.wnaf2.data(), plan.wnaf2_len,
        plan.neg1, plan.neg2);
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
        // z_one_ guarantees FE52 is pre-normalized by make_affine_inplace:
        // skip fe52_normalize_inline (saves ~5 ns per field element).
        out[0] = (y_.n[0] & 1) ? 0x03 : 0x02;
        x_.store_b32_prenorm(out.data() + 1);
#else
        out[0] = (y_.limbs()[0] & 1) ? 0x03 : 0x02;
        x_.to_bytes_into(out.data() + 1);
#endif
        return out;
    }
    // Slow path: compute affine coordinates with a single inversion
#if defined(SECP256K1_FAST_52BIT)
    FieldElement52 const z_inv = z_.inverse_safegcd();
    if (SECP256K1_UNLIKELY(z_inv.normalizes_to_zero_var())) {
        out.fill(0);
        return out;
    } // LCOV_EXCL_LINE
    FieldElement52 const z_inv2 = z_inv.square();
    FieldElement52 x_aff = x_ * z_inv2;
    FieldElement52 y_aff = y_ * (z_inv2 * z_inv);
    x_aff.normalize();
    y_aff.normalize();
    out[0] = (y_aff.n[0] & 1) ? 0x03 : 0x02;
    x_aff.store_b32_prenorm(out.data() + 1);
#else
    FieldElement z_inv = z_.inverse();
    FieldElement z_inv2 = z_inv;
    z_inv2.square_inplace();
    FieldElement x_aff = x_ * z_inv2;
    FieldElement y_aff = y_ * z_inv2 * z_inv;
    out[0] = (y_aff.limbs()[0] & 1) ? 0x03 : 0x02;
    x_aff.to_bytes_into(out.data() + 1);
#endif
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
        // z_one_ guarantees FE52 is pre-normalized by make_affine_inplace:
        // skip fe52_normalize_inline (saves ~10 ns for two field elements).
        out[0] = 0x04;
        x_.store_b32_prenorm(out.data() + 1);
        y_.store_b32_prenorm(out.data() + 33);
#else
        out[0] = 0x04;
        x_.to_bytes_into(out.data() + 1);
        y_.to_bytes_into(out.data() + 33);
#endif
        return out;
    }
    // Slow path: compute affine coordinates with a single inversion
#if defined(SECP256K1_FAST_52BIT)
    FieldElement52 const z_inv = z_.inverse_safegcd();
    if (SECP256K1_UNLIKELY(z_inv.normalizes_to_zero_var())) {
        out.fill(0);
        return out;
    } // LCOV_EXCL_LINE
    FieldElement52 const z_inv2 = z_inv.square();
    FieldElement52 x_aff = x_ * z_inv2;
    FieldElement52 y_aff = y_ * (z_inv2 * z_inv);
    x_aff.normalize();
    y_aff.normalize();
    out[0] = 0x04;
    x_aff.store_b32_prenorm(out.data() + 1);
    y_aff.store_b32_prenorm(out.data() + 33);
#else
    FieldElement z_inv = z_.inverse();
    FieldElement z_inv2 = z_inv;
    z_inv2.square_inplace();
    FieldElement x_aff = x_ * z_inv2;
    FieldElement y_aff = y_ * z_inv2 * z_inv;
    out[0] = 0x04;
    x_aff.to_bytes_into(out.data() + 1);
    y_aff.to_bytes_into(out.data() + 33);
#endif
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
    FieldElement52 const z_inv = z_.inverse_safegcd();
    if (SECP256K1_UNLIKELY(z_inv.normalizes_to_zero_var())) return false; // LCOV_EXCL_LINE
    FieldElement52 const z_inv2 = z_inv.square();
    FieldElement52 y_aff = y_ * (z_inv2 * z_inv);
    y_aff.normalize();
    return (y_aff.n[0] & 1) == 0;
#else
    FieldElement z_inv = z_.inverse();
    FieldElement z_inv2 = z_inv;
    z_inv2.square_inplace();
    FieldElement y_aff = y_ * z_inv2 * z_inv;
    // mul_impl Barrett-reduces to [0, p) -- limbs()[0] LSB is parity.
    return (y_aff.limbs()[0] & 1) == 0;
#endif
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
        x_.store_b32_prenorm(xb.data());
        return {xb, y_odd};
#else
        bool const y_odd = (y_.limbs()[0] & 1) != 0;
        return {x_.to_bytes(), y_odd};
#endif
    }
#if defined(SECP256K1_FAST_52BIT)
    FieldElement52 const z_inv = z_.inverse_safegcd();
    if (SECP256K1_UNLIKELY(z_inv.normalizes_to_zero_var())) return {std::array<uint8_t,32>{}, false}; // LCOV_EXCL_LINE
    FieldElement52 const z_inv2 = z_inv.square();
    FieldElement52 x_aff = x_ * z_inv2;
    FieldElement52 y_aff = y_ * (z_inv2 * z_inv);
    x_aff.normalize();
    y_aff.normalize();
    bool const y_odd = (y_aff.n[0] & 1) != 0;
    std::array<uint8_t, 32> xb{};
    x_aff.store_b32_prenorm(xb.data());
    return {xb, y_odd};
#else
    FieldElement z_inv = z_.inverse();
    FieldElement z_inv2 = z_inv;
    z_inv2.square_inplace();
    FieldElement x_aff = x_ * z_inv2;
    FieldElement y_aff = y_ * z_inv2 * z_inv;
    // Parity from limbs directly (avoids to_bytes serialization)
    bool const y_odd = (y_aff.limbs()[0] & 1) != 0;
    return {x_aff.to_bytes(), y_odd};
#endif
}

// -- Normalize: Jacobian -> affine (Z=1) with ONE field inversion -------------
// After this call z_one_ == true and all serialization is O(1).
void Point::normalize() {
    if (z_one_ || infinity_) return;
#if defined(SECP256K1_FAST_52BIT)
    // inverse_safegcd: FE52->4x64->SafeGCD->FE52 (returns zero if z==0)
    FieldElement52 const z_inv = z_.inverse_safegcd();
    if (SECP256K1_UNLIKELY(z_inv.normalizes_to_zero_var())) return; // LCOV_EXCL_LINE
    // All arithmetic in FE52 -- avoids 4 intermediate FE52<->4x64 conversions
    FieldElement52 const z_inv2 = z_inv.square();
    FieldElement52 const z_inv3 = z_inv2 * z_inv;
    x_ *= z_inv2;
    y_ *= z_inv3;
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
        x_.store_b32_prenorm(xb.data());
        return xb;
#else
        return x_.to_bytes();
#endif
    }
#if defined(SECP256K1_FAST_52BIT)
    FieldElement52 const z_inv = z_.inverse_safegcd();
    if (SECP256K1_UNLIKELY(z_inv.normalizes_to_zero_var())) return {}; // LCOV_EXCL_LINE
    FieldElement52 x_aff = x_ * z_inv.square();
    x_aff.normalize();
    std::array<uint8_t, 32> xb{};
    x_aff.store_b32_prenorm(xb.data());
    return xb;
#else
    FieldElement z_inv = z_.inverse();
    FieldElement z_inv2 = z_inv;
    z_inv2.square_inplace();
    FieldElement const x_aff = x_ * z_inv2;
    return x_aff.to_bytes();
#endif
}

// -- Shared Montgomery batch Z-inversion core ---------------------------------
// Computes individual Z^(-1) values for N points using Montgomery's trick:
// 1 field inversion + 2(N-1) multiplications.
// out_z_inv: caller-owned array of FieldElement, size >= n.
// For infinity points, out_z_inv[i] is left undefined (caller must check).
static void batch_z_inv(const Point* points, size_t n,
                        FieldElement* out_z_inv) {
    // Internal partials buffer (only needed within this function)
    constexpr size_t STACK_LIMIT = 256;
    constexpr size_t kMaxAllocElems =
        static_cast<size_t>(std::numeric_limits<std::ptrdiff_t>::max()) / sizeof(FieldElement);
    FieldElement stack_partials[STACK_LIMIT];
    std::unique_ptr<FieldElement[]> heap_partials;
    FieldElement* partials = stack_partials;
    if (n > STACK_LIMIT) {
        if (SECP256K1_UNLIKELY(n > kMaxAllocElems)) {
            std::abort();
        }
        heap_partials = std::make_unique<FieldElement[]>(n);
        partials = heap_partials.get();
    }

    // Forward pass: accumulate Z products
    FieldElement running = FieldElement::one();
    for (size_t i = 0; i < n; ++i) {
        if (points[i].is_infinity()) {
            partials[i] = running; // skip infinity, don't multiply
        } else {
            FieldElement const z_fe = points[i].z();
            partials[i] = running;
            running *= z_fe;
        }
    }

    // Single inversion of the accumulated product
    FieldElement inv = running.inverse();

    // Backward pass: recover individual Z^(-1) values
    for (size_t i = n; i-- > 0; ) {
        if (points[i].is_infinity()) {
            continue;
        }
        FieldElement const z_fe = points[i].z();
        out_z_inv[i] = partials[i] * inv;
        inv *= z_fe;
    }
}

// -- batch_scalar_mul_fixed_k: fixed K × N variable points -------------------
// Same wNAF for all N points (from KPlan) → one batch inversion per chunk.
// Adapted from multi_scalar_mul (multiscalar.cpp) for the fixed-K case.
void Point::batch_scalar_mul_fixed_k(const KPlan& plan,
                                     const Point* pts,
                                     size_t n,
                                     Point* results)
{
    if (n == 0) return;
    if (n == 1) { results[0] = pts[0].scalar_mul_with_plan(plan); return; }

#if !defined(SECP256K1_FAST_52BIT)
    // Non-FE52 platforms: per-point fallback
    for (size_t i = 0; i < n; ++i)
        results[i] = pts[i].scalar_mul_with_plan(plan);
    return;
#else
    using FE52 = FieldElement52;
    const FE52& beta52 = kBeta52_pt;  // file-scope constant — no per-call init-flag

    const unsigned w = plan.window_width;
    // Guard: only w ∈ [3,7] are supported by the FE52 path
    if (SECP256K1_UNLIKELY(w < 3 || w > 7)) {
        for (size_t i = 0; i < n; ++i)
            results[i] = pts[i].scalar_mul_with_plan(plan);
        return;
    }
    const size_t table_size = static_cast<size_t>(1) << (w - 2);  // 8 for w=5

    // Trim trailing zeros from shared wNAF buffers
    size_t wnaf1_len = plan.wnaf1_len;
    size_t wnaf2_len = plan.wnaf2_len;
    const int32_t* wnaf1 = plan.wnaf1.data();
    const int32_t* wnaf2 = plan.wnaf2.data();
    while (wnaf1_len > 0 && wnaf1[wnaf1_len - 1] == 0) --wnaf1_len;
    while (wnaf2_len > 0 && wnaf2[wnaf2_len - 1] == 0) --wnaf2_len;
    const size_t max_len = (wnaf1_len > wnaf2_len) ? wnaf1_len : wnaf2_len;

    // Process in chunks to keep the working set in L2/L3 (~1 MB target).
    // Each point needs table_size × 2 × FE52 (x+y) + 1 JacobianPoint52 accumulator.
    // chunk_size = 2048: 2048 × 8 × 2 × 40B ≈ 1.25 MB for tables + 400 KB accumulators.
    constexpr size_t kChunkSize = 2048;

    // Thread-local scratch buffers — reused across calls, no heap churn per tx batch.
    static thread_local std::vector<Point>  s_tables;
    static thread_local std::vector<FE52>   s_prefix;
    static thread_local std::vector<FE52>   s_tbl_P_x,  s_tbl_P_y;
    static thread_local std::vector<FE52>   s_tbl_phi_x, s_tbl_phi_y;
    static thread_local std::vector<Point>  s_acc;

    // Each chunk is fully independent: same KPlan (read-only), disjoint results[].
    // thread_local scratch guarantees zero inter-thread contention.
#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 1) if (n >= kChunkSize * 4)
#endif
    for (size_t chunk_start = 0; chunk_start < n; chunk_start += kChunkSize) {
        const size_t chunk_n = std::min(kChunkSize, n - chunk_start);
        const size_t total   = chunk_n * table_size;

        s_tables.resize(total);
        s_prefix.resize(total);
        s_tbl_P_x.resize(total);   s_tbl_P_y.resize(total);
        s_tbl_phi_x.resize(total); s_tbl_phi_y.resize(total);
        s_acc.resize(chunk_n);

        // Step 1: Build Jacobian window tables for chunk_n points.
        for (size_t i = 0; i < chunk_n; ++i) {
            const Point& pt = pts[chunk_start + i];
            if (SECP256K1_UNLIKELY(pt.is_infinity())) {
                for (size_t j = 0; j < table_size; ++j)
                    s_tables[i * table_size + j] = Point::infinity();
                continue;
            }
            Point base = plan.neg1 ? pt.negate() : pt;
            s_tables[i * table_size] = base;
            if (table_size > 1) {
                Point P2 = base.dbl();
                for (size_t j = 1; j < table_size; ++j)
                    s_tables[i * table_size + j] =
                        s_tables[i * table_size + j - 1].add(P2);
            }
        }

        // Step 2: Batch-invert all chunk_n × table_size Z-coordinates.
        //         Montgomery forward pass → single inversion → backward pass.
        bool all_affine = true;
        for (size_t k = 0; k < total && all_affine; ++k)
            if (!s_tables[k].is_infinity() && !s_tables[k].is_normalized())
                all_affine = false;

        if (all_affine) {
            for (size_t k = 0; k < total; ++k) {
                s_tbl_P_x[k] = s_tables[k].X52();
                s_tbl_P_y[k] = s_tables[k].Y52();
            }
        } else {
            // Forward accumulation
            s_prefix[0] = s_tables[0].is_infinity()
                            ? FE52::one() : s_tables[0].Z52();
            for (size_t k = 1; k < total; ++k) {
                s_prefix[k] = s_tables[k].is_infinity()
                    ? s_prefix[k - 1]
                    : s_prefix[k - 1] * s_tables[k].Z52();
            }
            if (SECP256K1_UNLIKELY(s_prefix[total - 1].normalizes_to_zero())) {
                // Degenerate: fall back per-point for this chunk
                for (size_t i = 0; i < chunk_n; ++i)
                    results[chunk_start + i] =
                        pts[chunk_start + i].scalar_mul_with_plan(plan);
                continue;
            }
            FE52 inv = s_prefix[total - 1].inverse();
            for (size_t k = total; k-- > 0; ) {
                if (s_tables[k].is_infinity()) {
                    s_tbl_P_x[k] = FE52::zero();
                    s_tbl_P_y[k] = FE52::zero();
                    continue;
                }
                FE52 const z_inv = (k > 0) ? s_prefix[k - 1] * inv : inv;
                if (k > 0) inv *= s_tables[k].Z52();
                FE52 const z2 = z_inv.square();
                FE52 const z3 = z2 * z_inv;
                s_tbl_P_x[k] = s_tables[k].X52() * z2;
                s_tbl_P_y[k] = s_tables[k].Y52() * z3;
            }
        }

        // Step 3: Derive phi tables via beta multiplication.
        const bool flip_phi = (plan.neg1 != plan.neg2);
        for (size_t k = 0; k < total; ++k) {
            s_tbl_phi_x[k] = s_tbl_P_x[k] * beta52;
            if (flip_phi) {
                s_tbl_phi_y[k] = s_tbl_P_y[k].negate(1);
                s_tbl_phi_y[k].normalize_weak();
            } else {
                s_tbl_phi_y[k] = s_tbl_P_y[k];
            }
        }

        // Step 4: Shared wNAF loop — all chunk_n accumulators advance in lockstep.
        for (size_t i = 0; i < chunk_n; ++i)
            s_acc[i] = Point::infinity();

        for (size_t bit = max_len; bit-- > 0; ) {
            // Double all accumulators
            for (size_t i = 0; i < chunk_n; ++i)
                s_acc[i].dbl_inplace();

            // k1 stream: same NAF digit for all points, different table rows
            if (bit < wnaf1_len) {
                const int32_t d1 = wnaf1[bit];
                if (d1 != 0) {
                    const size_t idx = static_cast<size_t>(
                        (d1 > 0 ? d1 - 1 : -d1 - 1) / 2);
                    for (size_t i = 0; i < chunk_n; ++i) {
                        FE52 lx = s_tbl_P_x[i * table_size + idx];
                        FE52 ly = s_tbl_P_y[i * table_size + idx];
                        if (d1 < 0) { ly.negate_assign(1); ly.normalize_weak(); }
                        s_acc[i].add_mixed52_inplace(lx, ly);
                    }
                }
            }

            // k2 stream
            if (bit < wnaf2_len) {
                const int32_t d2 = wnaf2[bit];
                if (d2 != 0) {
                    const size_t idx = static_cast<size_t>(
                        (d2 > 0 ? d2 - 1 : -d2 - 1) / 2);
                    for (size_t i = 0; i < chunk_n; ++i) {
                        FE52 lx = s_tbl_phi_x[i * table_size + idx];
                        FE52 ly = s_tbl_phi_y[i * table_size + idx];
                        if (d2 < 0) { ly.negate_assign(1); ly.normalize_weak(); }
                        s_acc[i].add_mixed52_inplace(lx, ly);
                    }
                }
            }
        }

        // Step 5: Store results (lazy Jacobian — caller uses batch_to_compressed)
        for (size_t i = 0; i < chunk_n; ++i)
            results[chunk_start + i] = s_acc[i];
    }
#endif // SECP256K1_FAST_52BIT
}

// 4× interleaved variant: groups of 4 base points share the wNAF digit loop.
// ILP from 4 independent JacobianPoint52 accumulators executing in parallel
// inside the CPU's OOO engine.  Falls back to per-point on non-FE52 platforms.
void Point::batch_scalar_mul_fixed_k_4x(const KPlan& plan,
                                         const Point* pts,
                                         size_t n,
                                         Point* results)
{
    if (n == 0) return;

#if !defined(SECP256K1_FAST_52BIT)
    for (size_t i = 0; i < n; ++i)
        results[i] = pts[i].scalar_mul_with_plan(plan);
    return;
#else
    // Process in groups of 4
    const size_t n4 = n & ~size_t(3);
    for (size_t i = 0; i < n4; i += 4)
        scalar_mul_with_plan_glv52_4x(pts + i, plan, results + i);

    // Remainder (0-3 points)
    for (size_t i = n4; i < n; ++i)
        results[i] = scalar_mul_with_plan_glv52(pts[i], plan);
#endif
}

// ============================================================================
// Precomputed scan-table: build once, run hot loop many times
// ============================================================================
// Internal storage: per-point iso-affine tables + per-point globalz.
// Exact same convention as scalar_mul_with_plan_glv52:
//   tbl entries have shared implied Z = globalz (the iso curve frame).
//   After shamir_2stream_glv52, multiply result.z by globalz to convert back.
struct ScanCacheImpl {
    unsigned window_bits;
    int table_size;
    size_t n;
#if defined(SECP256K1_FAST_52BIT)
    std::vector<AffinePoint52>  tbl_P;
    std::vector<AffinePoint52>  tbl_phiP;
    std::vector<FieldElement52> globalz;
#endif
    std::vector<uint8_t> valid;           // 0 = infinity/degenerate
};

Point::PointScanCacheHandle Point::batch_scan_precompute(
    const KPlan& plan, const Point* pts, size_t n)
{
    auto* impl = new ScanCacheImpl();
    impl->window_bits = plan.window_width;
    impl->n = n;

#if !defined(SECP256K1_FAST_52BIT)
    impl->table_size = 0;
    return PointScanCacheHandle(impl, [](void* p){ delete static_cast<ScanCacheImpl*>(p); });
#else
    const unsigned w = plan.window_width;
    const int ts = 1 << (w - 2);
    impl->table_size = ts;
    impl->tbl_P.resize(n * static_cast<size_t>(ts));
    impl->tbl_phiP.resize(n * static_cast<size_t>(ts));
    impl->globalz.resize(n);
    impl->valid.resize(n, 0);

    const bool flip_phi = (plan.neg1 != plan.neg2);

    for (size_t i = 0; i < n; ++i) {
        if (SECP256K1_UNLIKELY(pts[i].is_infinity())) continue;

        JacobianPoint52 const P52 = plan.neg1
            ? to_jac52(pts[i].negate())
            : to_jac52(pts[i]);

        AffinePoint52* tp  = impl->tbl_P.data()    + i * static_cast<size_t>(ts);
        AffinePoint52* tph = impl->tbl_phiP.data() + i * static_cast<size_t>(ts);

        if (!build_glv52_table_zr(P52, tp, ts, impl->globalz[i])) continue;
        derive_phi52_table(tp, tph, ts, flip_phi);
        impl->valid[i] = 1;
    }

    return PointScanCacheHandle(impl, [](void* p){ delete static_cast<ScanCacheImpl*>(p); });
#endif
}

void Point::batch_scan_run(const PointScanCacheHandle& cache,
                            const KPlan& plan,
                            size_t cache_offset,
                            Point* results,
                            size_t n)
{
    const auto* impl = static_cast<const ScanCacheImpl*>(cache.get());

#if !defined(SECP256K1_FAST_52BIT)
    for (size_t i = 0; i < n; ++i) results[i] = Point::infinity();
    return;
#else
    if (SECP256K1_UNLIKELY(!impl || impl->table_size == 0 ||
                            cache_offset + n > impl->n)) {
        for (size_t i = 0; i < n; ++i) results[i] = Point::infinity();
        return;
    }

    const int ts = impl->table_size;

    std::size_t wnaf1_len = plan.wnaf1_len;
    std::size_t wnaf2_len = plan.wnaf2_len;
    const int32_t* wnaf1_ptr = plan.wnaf1.data();
    const int32_t* wnaf2_ptr = plan.wnaf2.data();
    while (wnaf1_len > 0 && wnaf1_ptr[wnaf1_len - 1] == 0) --wnaf1_len;
    while (wnaf2_len > 0 && wnaf2_ptr[wnaf2_len - 1] == 0) --wnaf2_len;

    for (size_t i = 0; i < n; ++i) {
        const size_t ci = cache_offset + i;
        if (SECP256K1_UNLIKELY(!impl->valid[ci])) {
            results[i] = Point::infinity();
            continue;
        }
        const AffinePoint52* tp  = impl->tbl_P.data()    + ci * static_cast<size_t>(ts);
        const AffinePoint52* tph = impl->tbl_phiP.data() + ci * static_cast<size_t>(ts);

        JacobianPoint52 r52 = shamir_2stream_glv52(
            tp, tph, wnaf1_ptr, wnaf1_len, wnaf2_ptr, wnaf2_len);
        if (!r52.infinity)
            r52.z.mul_assign(impl->globalz[ci]);

        results[i] = from_jac52(r52);
    }
#endif
}

// N-stream lockstep: b_scan wNAF digits loop runs ONCE, inner loop over chunk_n.
// For each bit position d:
//   - evaluate digit ONCE (not per-point)
//   - skip chunk_n inner iterations when digit == 0 (~75% of positions)
//   - hoist sign-branch outside inner loop for non-zero digits
void Point::batch_scan_run_lockstep(const PointScanCacheHandle& cache,
                                     const KPlan& plan,
                                     size_t cache_offset,
                                     Point* results,
                                     size_t n,
                                     size_t chunk_size)
{
    const auto* impl = static_cast<const ScanCacheImpl*>(cache.get());

#if !defined(SECP256K1_FAST_52BIT)
    for (size_t i = 0; i < n; ++i) results[i] = Point::infinity();
    return;
#else
    if (SECP256K1_UNLIKELY(!impl || impl->table_size == 0 ||
                            cache_offset + n > impl->n)) {
        for (size_t i = 0; i < n; ++i) results[i] = Point::infinity();
        return;
    }

    const int ts = impl->table_size;

    std::size_t wnaf1_len = plan.wnaf1_len;
    std::size_t wnaf2_len = plan.wnaf2_len;
    const int32_t* wnaf1_ptr = plan.wnaf1.data();
    const int32_t* wnaf2_ptr = plan.wnaf2.data();
    while (wnaf1_len > 0 && wnaf1_ptr[wnaf1_len - 1] == 0) --wnaf1_len;
    while (wnaf2_len > 0 && wnaf2_ptr[wnaf2_len - 1] == 0) --wnaf2_len;
    const std::size_t max_len = (wnaf1_len > wnaf2_len) ? wnaf1_len : wnaf2_len;

    const JacobianPoint52 inf52 = {
        FieldElement52::zero(), FieldElement52::one(), FieldElement52::zero(), true};

    // Stack scratch for accumulators (chunk_size ≤ 256 → 30 KB, fits in L1/L2)
    constexpr std::size_t kMaxChunk = 256;
    if (chunk_size == 0 || chunk_size > kMaxChunk) chunk_size = kMaxChunk;

    JacobianPoint52 acc[kMaxChunk];

    for (std::size_t chunk_start = 0; chunk_start < n; chunk_start += chunk_size) {
        const std::size_t chunk_n = std::min(chunk_size, n - chunk_start);
        const std::size_t base_ci = cache_offset + chunk_start;

        for (std::size_t i = 0; i < chunk_n; ++i) acc[i] = inf52;

        // Outer loop: b_scan digit positions (loaded ONCE per position)
        for (int d = static_cast<int>(max_len) - 1; d >= 0; --d) {
            // Double all chunk_n accumulators
            for (std::size_t i = 0; i < chunk_n; ++i)
                jac52_double_inplace(acc[i]);

            // k1 stream — sign branch hoisted outside inner loop
            const int32_t d1 = (static_cast<std::size_t>(d) < wnaf1_len)
                                    ? wnaf1_ptr[d] : 0;
            if (d1 > 0) {
                const std::size_t idx = static_cast<std::size_t>(d1 - 1) >> 1;
                for (std::size_t i = 0; i < chunk_n; ++i) {
                    const std::size_t ci = base_ci + i;
                    if (SECP256K1_LIKELY(impl->valid[ci]))
                        jac52_add_mixed_inplace(acc[i], impl->tbl_P[ci * static_cast<std::size_t>(ts) + idx]);
                }
            } else if (d1 < 0) {
                const std::size_t idx = static_cast<std::size_t>(-d1 - 1) >> 1;
                for (std::size_t i = 0; i < chunk_n; ++i) {
                    const std::size_t ci = base_ci + i;
                    if (SECP256K1_LIKELY(impl->valid[ci])) {
                        AffinePoint52 pt = impl->tbl_P[ci * static_cast<std::size_t>(ts) + idx];
                        pt.y.negate_assign(1); pt.y.normalize_weak();
                        jac52_add_mixed_inplace(acc[i], pt);
                    }
                }
            }

            // k2 stream — sign branch hoisted outside inner loop
            const int32_t d2 = (static_cast<std::size_t>(d) < wnaf2_len)
                                    ? wnaf2_ptr[d] : 0;
            if (d2 > 0) {
                const std::size_t idx = static_cast<std::size_t>(d2 - 1) >> 1;
                for (std::size_t i = 0; i < chunk_n; ++i) {
                    const std::size_t ci = base_ci + i;
                    if (SECP256K1_LIKELY(impl->valid[ci]))
                        jac52_add_mixed_inplace(acc[i], impl->tbl_phiP[ci * static_cast<std::size_t>(ts) + idx]);
                }
            } else if (d2 < 0) {
                const std::size_t idx = static_cast<std::size_t>(-d2 - 1) >> 1;
                for (std::size_t i = 0; i < chunk_n; ++i) {
                    const std::size_t ci = base_ci + i;
                    if (SECP256K1_LIKELY(impl->valid[ci])) {
                        AffinePoint52 pt = impl->tbl_phiP[ci * static_cast<std::size_t>(ts) + idx];
                        pt.y.negate_assign(1); pt.y.normalize_weak();
                        jac52_add_mixed_inplace(acc[i], pt);
                    }
                }
            }
        }

        // Convert chunk accumulators to results
        for (std::size_t i = 0; i < chunk_n; ++i) {
            const std::size_t ci = base_ci + i;
            if (SECP256K1_UNLIKELY(!impl->valid[ci])) {
                results[chunk_start + i] = Point::infinity();
                continue;
            }
            if (!acc[i].infinity)
                acc[i].z.mul_assign(impl->globalz[ci]);
            results[chunk_start + i] = from_jac52(acc[i]);
        }
    }
#endif
}

// ============================================================================
// Scan-cache disk persistence
// ============================================================================
// Binary format:
//   [4B magic][4B version][4B window_bits][4B table_size][8B n]
//   [n × 1B valid]
//   [n × table_size × (32B x + 32B y)]   ← tbl_P
//   [n × table_size × (32B x + 32B y)]   ← tbl_phiP
//   [n × 32B]                             ← globalz
//
// Field elements are serialized as 32-byte big-endian (normalized).
// Atomic write: tmp file → rename to prevent readers seeing partial data.

static constexpr uint32_t SCAN_CACHE_MAGIC   = 0x53434E21U; // "SCN!"
static constexpr uint32_t SCAN_CACHE_VERSION = 1U;

#if defined(SECP256K1_FAST_52BIT)
static bool scan_write_fe52(std::ofstream& f, const FieldElement52& fe) {
    std::array<uint8_t, 32> buf;
    fe.to_bytes_into(buf.data());
    f.write(reinterpret_cast<const char*>(buf.data()), 32);
    return f.good();
}
static bool scan_read_fe52(std::ifstream& f, FieldElement52& fe) {
    std::array<uint8_t, 32> buf;
    f.read(reinterpret_cast<char*>(buf.data()), 32);
    if (!f.good()) return false;
    fe = FieldElement52::from_fe(FieldElement::from_bytes(buf));
    return true;
}
#endif // SECP256K1_FAST_52BIT

bool Point::batch_scan_save(const PointScanCacheHandle& cache, const std::string& path) {
    const auto* impl = static_cast<const ScanCacheImpl*>(cache.get());
    if (!impl || impl->n == 0) return false;

#if !defined(SECP256K1_FAST_52BIT)
    return false;  // No FE52 data to save
#else
    // Atomic write: tmp path → rename
#if defined(_WIN32)
    std::string tmp = path + ".tmp." + std::to_string(_getpid());
#else
    std::string tmp = path + ".tmp." + std::to_string(getpid());
#endif
    std::ofstream f(tmp, std::ios::binary);
    if (!f.is_open()) return false;

    // Header
    const uint32_t magic   = SCAN_CACHE_MAGIC;
    const uint32_t version = SCAN_CACHE_VERSION;
    const uint32_t wbits   = impl->window_bits;
    const uint32_t ts      = static_cast<uint32_t>(impl->table_size);
    const uint64_t n       = static_cast<uint64_t>(impl->n);
    f.write(reinterpret_cast<const char*>(&magic),   4);
    f.write(reinterpret_cast<const char*>(&version), 4);
    f.write(reinterpret_cast<const char*>(&wbits),   4);
    f.write(reinterpret_cast<const char*>(&ts),      4);
    f.write(reinterpret_cast<const char*>(&n),       8);
    if (!f.good()) { std::remove(tmp.c_str()); return false; }

    // valid flags
    f.write(reinterpret_cast<const char*>(impl->valid.data()),
            static_cast<std::streamsize>(impl->n));
    if (!f.good()) { std::remove(tmp.c_str()); return false; }

    // tbl_P and tbl_phiP
    const size_t total = impl->n * static_cast<size_t>(impl->table_size);
    for (size_t k = 0; k < total; ++k) {
        if (!scan_write_fe52(f, impl->tbl_P[k].x)) { std::remove(tmp.c_str()); return false; }
        if (!scan_write_fe52(f, impl->tbl_P[k].y)) { std::remove(tmp.c_str()); return false; }
    }
    for (size_t k = 0; k < total; ++k) {
        if (!scan_write_fe52(f, impl->tbl_phiP[k].x)) { std::remove(tmp.c_str()); return false; }
        if (!scan_write_fe52(f, impl->tbl_phiP[k].y)) { std::remove(tmp.c_str()); return false; }
    }

    // globalz
    for (size_t i = 0; i < impl->n; ++i) {
        if (!scan_write_fe52(f, impl->globalz[i])) { std::remove(tmp.c_str()); return false; }
    }

    f.close();
    if (!f.good()) { std::remove(tmp.c_str()); return false; }

    if (std::rename(tmp.c_str(), path.c_str()) != 0) {
        std::remove(tmp.c_str());
        return false;
    }
    return true;
#endif
}

Point::PointScanCacheHandle Point::batch_scan_load(const std::string& path) {
#if !defined(SECP256K1_FAST_52BIT)
    return nullptr;
#else
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return nullptr;

    uint32_t magic = 0, version = 0, wbits = 0, ts_u = 0;
    uint64_t n64 = 0;
    f.read(reinterpret_cast<char*>(&magic),   4);
    f.read(reinterpret_cast<char*>(&version), 4);
    f.read(reinterpret_cast<char*>(&wbits),   4);
    f.read(reinterpret_cast<char*>(&ts_u),    4);
    f.read(reinterpret_cast<char*>(&n64),     8);
    if (!f.good() || magic != SCAN_CACHE_MAGIC || version != SCAN_CACHE_VERSION) return nullptr;
    if (ts_u == 0 || n64 == 0 || ts_u > 32) return nullptr;

    const int ts = static_cast<int>(ts_u);
    const size_t n = static_cast<size_t>(n64);
    const size_t total = n * static_cast<size_t>(ts);

    auto* impl = new ScanCacheImpl();
    impl->window_bits = wbits;
    impl->table_size  = ts;
    impl->n           = n;
    impl->tbl_P.resize(total);
    impl->tbl_phiP.resize(total);
    impl->globalz.resize(n);
    impl->valid.resize(n, 0);

    f.read(reinterpret_cast<char*>(impl->valid.data()),
           static_cast<std::streamsize>(n));
    if (!f.good()) { delete impl; return nullptr; }

    for (size_t k = 0; k < total; ++k) {
        if (!scan_read_fe52(f, impl->tbl_P[k].x) ||
            !scan_read_fe52(f, impl->tbl_P[k].y)) { delete impl; return nullptr; }
    }
    for (size_t k = 0; k < total; ++k) {
        if (!scan_read_fe52(f, impl->tbl_phiP[k].x) ||
            !scan_read_fe52(f, impl->tbl_phiP[k].y)) { delete impl; return nullptr; }
    }
    for (size_t i = 0; i < n; ++i) {
        if (!scan_read_fe52(f, impl->globalz[i])) { delete impl; return nullptr; }
    }

    return PointScanCacheHandle(impl, [](void* p){ delete static_cast<ScanCacheImpl*>(p); });
#endif
}

Point::PointScanCacheHandle Point::batch_scan_precompute_or_load(
    const KPlan& plan, const Point* pts, size_t n, const std::string& cache_path)
{
    // Try to load from disk first
    if (!cache_path.empty()) {
        PointScanCacheHandle h = batch_scan_load(cache_path);
        if (h) {
            const auto* impl = static_cast<const ScanCacheImpl*>(h.get());
            // Validate: same window width and size
            if (impl->n == n && impl->window_bits == plan.window_width)
                return h;
        }
    }

    // Build from scratch
    PointScanCacheHandle h = batch_scan_precompute(plan, pts, n);

    // Persist for next run
    if (!cache_path.empty())
        batch_scan_save(h, cache_path);

    return h;
}

// -- Batch normalize: Montgomery trick for N points ---------------------------
// 1 inversion + 3(N-1) multiplications instead of N inversions.
void Point::batch_normalize(const Point* points, size_t n,
                            FieldElement* out_x, FieldElement* out_y) {
    if (n == 0) return;
    constexpr size_t kMaxAllocElems =
        static_cast<size_t>(std::numeric_limits<std::ptrdiff_t>::max()) / sizeof(FieldElement);
    if (SECP256K1_UNLIKELY(n > kMaxAllocElems)) return;

    // Fast path: all points already affine (z == 1) -- skip the batch inversion.
    bool all_affine = true;
    for (size_t i = 0; i < n; ++i) {
        if (!points[i].is_infinity() && !points[i].z_one_) {
            all_affine = false;
            break;
        }
    }
    if (all_affine) {
        for (size_t i = 0; i < n; ++i) {
            if (points[i].is_infinity()) {
                out_x[i] = FieldElement::zero();
                out_y[i] = FieldElement::zero();
                continue;
            }
#if defined(SECP256K1_FAST_52BIT)
            out_x[i] = points[i].x_.to_fe();
            out_y[i] = points[i].y_.to_fe();
#else
            out_x[i] = points[i].x_;
            out_y[i] = points[i].y_;
#endif
        }
        return;
    }

    constexpr size_t STACK_LIMIT = 256;
    FieldElement stack_z_inv[STACK_LIMIT];
    std::unique_ptr<FieldElement[]> heap_z_inv;
    FieldElement* z_inv = stack_z_inv;
    if (n > STACK_LIMIT) {
        heap_z_inv = std::make_unique<FieldElement[]>(
            static_cast<size_t>(static_cast<std::ptrdiff_t>(n)));
        z_inv = heap_z_inv.get();
    }

    batch_z_inv(points, n, z_inv);

    // Compute affine: x_aff = X * Z^(-2), y_aff = Y * Z^(-3)
    for (size_t i = 0; i < n; ++i) {
        if (points[i].is_infinity()) {
            out_x[i] = FieldElement::zero();
            out_y[i] = FieldElement::zero();
            continue;
        }
        FieldElement z_inv2 = z_inv[i];
        z_inv2.square_inplace();
#if defined(SECP256K1_FAST_52BIT)
        out_x[i] = points[i].x_.to_fe() * z_inv2;
        out_y[i] = points[i].y_.to_fe() * z_inv2 * z_inv[i];
#else
        out_x[i] = points[i].x_ * z_inv2;
        out_y[i] = points[i].y_ * z_inv2 * z_inv[i];
#endif
    }
}

// -- Batch to_compressed: serialize N points with ONE inversion ---------------
void Point::batch_to_compressed(const Point* points, size_t n,
                                std::array<uint8_t, 33>* out) {
    if (n == 0) return;
    constexpr size_t kMaxAllocElems =
        static_cast<size_t>(std::numeric_limits<std::ptrdiff_t>::max()) / sizeof(FieldElement);
    if (SECP256K1_UNLIKELY(n > kMaxAllocElems)) return;

    // Fast path: all points already affine -- no inversion needed.
    bool all_affine = true;
    for (size_t i = 0; i < n; ++i) {
        if (!points[i].is_infinity() && !points[i].z_one_) {
            all_affine = false;
            break;
        }
    }
    if (all_affine) {
        for (size_t i = 0; i < n; ++i) {
            if (points[i].is_infinity()) {
                out[i].fill(0);
                continue;
            }
#if defined(SECP256K1_FAST_52BIT)
            out[i][0] = (points[i].y_.n[0] & 1U) ? 0x03 : 0x02;
            points[i].x_.store_b32_prenorm(out[i].data() + 1);
#else
            out[i][0] = (points[i].y_.limbs()[0] & 1U) ? 0x03 : 0x02;
            auto x_bytes = points[i].x_.to_bytes();
            std::copy(x_bytes.begin(), x_bytes.end(), out[i].begin() + 1);
#endif
        }
        return;
    }

    constexpr size_t STACK_LIMIT = 256;
    FieldElement stack_x[STACK_LIMIT], stack_y[STACK_LIMIT];
    std::unique_ptr<FieldElement[]> heap_x, heap_y;
    FieldElement* aff_x = nullptr;
    FieldElement* aff_y = nullptr;
    if (n <= STACK_LIMIT) {
        aff_x = stack_x; aff_y = stack_y;
    } else {
        auto const signed_n = static_cast<std::ptrdiff_t>(n);
        heap_x = std::make_unique<FieldElement[]>(static_cast<size_t>(signed_n));
        heap_y = std::make_unique<FieldElement[]>(static_cast<size_t>(signed_n));
        aff_x = heap_x.get(); aff_y = heap_y.get();
    }

    batch_normalize(points, n, aff_x, aff_y);

    for (size_t i = 0; i < n; ++i) {
        if (points[i].is_infinity()) {
            out[i].fill(0);
            continue;
        }
        auto x_bytes = aff_x[i].to_bytes();
        out[i][0] = (aff_y[i].limbs()[0] & 1U) ? 0x03 : 0x02;
        std::copy(x_bytes.begin(), x_bytes.end(), out[i].begin() + 1);
    }
}

// -- Batch x_only_bytes: extract N x-coordinates with ONE inversion -----------
// Uses batch_z_inv for the Montgomery trick, then computes only x = X*Z^(-2)
// (skips the Y*Z^(-3) multiply that batch_normalize does -- ~33% fewer muls).
void Point::batch_x_only_bytes(const Point* points, size_t n,
                               std::array<uint8_t, 32>* out) {
    if (n == 0) return;
    constexpr size_t kMaxAllocElems =
        static_cast<size_t>(std::numeric_limits<std::ptrdiff_t>::max()) / sizeof(FieldElement);
    if (SECP256K1_UNLIKELY(n > kMaxAllocElems)) return;

    // Fast path: all points already affine.
    bool all_affine = true;
    for (size_t i = 0; i < n; ++i) {
        if (!points[i].is_infinity() && !points[i].z_one_) {
            all_affine = false;
            break;
        }
    }
    if (all_affine) {
        for (size_t i = 0; i < n; ++i) {
            if (points[i].is_infinity()) {
                out[i].fill(0);
                continue;
            }
#if defined(SECP256K1_FAST_52BIT)
            points[i].x_.store_b32_prenorm(out[i].data());
#else
            out[i] = points[i].x_.to_bytes();
#endif
        }
        return;
    }

    constexpr size_t STACK_LIMIT = 256;
    FieldElement stack_z_inv[STACK_LIMIT];
    std::unique_ptr<FieldElement[]> heap_z_inv;
    FieldElement* z_inv = stack_z_inv;
    if (n > STACK_LIMIT) {
        heap_z_inv = std::make_unique<FieldElement[]>(
            static_cast<size_t>(static_cast<std::ptrdiff_t>(n)));
        z_inv = heap_z_inv.get();
    }

    batch_z_inv(points, n, z_inv);

    for (size_t i = 0; i < n; ++i) {
        if (points[i].is_infinity()) {
            out[i].fill(0);
            continue;
        }
        FieldElement z_inv2 = z_inv[i];
        z_inv2.square_inplace();
        FieldElement const x_aff = points[i].X() * z_inv2;
        out[i] = x_aff.to_bytes();
    }
}

// -- 128-bit split Shamir: a*G + b*P -----------------------------------------
// Computes a*G + b*P using a ~128-bit doubling chain with 4 wNAF streams.
// a is split arithmetically: a = a_lo + a_hi*2^128  (no GLV for G-stream)
// b is GLV-decomposed: b = b1 + b2*lambda
// Then: a_lo*G + a_hi*H + b1*P + b2*psi(P) where H = 2^128*G
// G/H affine tables are cached statically (computed once, w=15 -> 8192 entries).
// -- File-scope generator tables (shared by dual_scalar_mul_gen_point and
//    dual_scalar_mul_gen_prebuilt) -----------------------------------------
// Initialized once on first use. NOT inside the function to allow sharing.
#if defined(SECP256K1_FAST_52BIT) && !defined(SECP256K1_USE_4X64_POINT_OPS)
namespace {
    constexpr unsigned kDualMulWindowG    = 15;
    constexpr int kDualMulGTableSize      = (1 << (kDualMulWindowG - 2)); // 8192
    constexpr unsigned kDualMulWindowP    = 5;
    constexpr int kDualMulPTableSize      = (1 << (kDualMulWindowP - 2)); // 8

    // Store directly as AffinePoint52 (5×52-bit limbs, 80 bytes/entry).
    // Previously used AffinePointCompact (4×64, 64 bytes) which required
    // a to_affine52() conversion (30 shift+mask ops, ~10-15 ns) on every
    // non-zero wNAF digit lookup in the hot verify loop. Direct FE52 storage
    // eliminates this per-lookup conversion at the cost of +131 KB per table.
    struct DualMulGenTables {
        AffinePoint52 tbl_G[kDualMulGTableSize];
        AffinePoint52 tbl_H[kDualMulGTableSize];
    };

    // Build one odd-multiple table for base point B.
    static void dual_mul_build_table(const JacobianPoint52& B,
                                     AffinePoint52* out, std::size_t count) {
        JacobianPoint52 const d = jac52_double(B);
        FieldElement52 const C = d.z, C2 = C.square(), C3 = C2 * C;
        AffinePoint52 const d_aff = {d.x, d.y};
        // Stack buffers for the common case (table sizes ≤ 32); heap fallback for larger.
        constexpr std::size_t kStackCap = 32;
        JacobianPoint52 iso_stk  [kStackCap];
        FieldElement52  eff_z_stk[kStackCap], prods_stk[kStackCap], zs_stk[kStackCap];
        std::unique_ptr<JacobianPoint52[]> iso_heap;
        std::unique_ptr<FieldElement52[]>  eff_z_heap, prods_heap, zs_heap;
        JacobianPoint52* iso   = count <= kStackCap ? iso_stk   : (iso_heap   = std::make_unique<JacobianPoint52[]>(count)).get();
        FieldElement52*  eff_z = count <= kStackCap ? eff_z_stk : (eff_z_heap = std::make_unique<FieldElement52[]>(count)).get();
        FieldElement52*  prods = count <= kStackCap ? prods_stk : (prods_heap = std::make_unique<FieldElement52[]>(count)).get();
        FieldElement52*  zs    = count <= kStackCap ? zs_stk    : (zs_heap    = std::make_unique<FieldElement52[]>(count)).get();

        iso[0] = {B.x * C2, B.y * C3, B.z, false};
        for (std::size_t i = 1; i < count; i++) {
            iso[i] = iso[i-1];
            jac52_add_mixed_inplace(iso[i], d_aff);
        }
        for (std::size_t i = 0; i < count; i++) eff_z[i] = iso[i].z * C;
        prods[0] = eff_z[0];
        for (std::size_t i = 1; i < count; i++) prods[i] = prods[i-1] * eff_z[i];
        FieldElement52 inv = prods[count-1].inverse_safegcd();
        for (std::size_t i = count-1; i > 0; --i) {
            zs[i] = prods[i-1] * inv; inv = inv * eff_z[i];
        }
        zs[0] = inv;
        for (std::size_t i = 0; i < count; i++) {
            FieldElement52 const zinv2 = zs[i].square(), zinv3 = zinv2 * zs[i];
            out[i] = {iso[i].x * zinv2, iso[i].y * zinv3};
        }
    }

    static const DualMulGenTables* get_dual_mul_gen_tables() {
        static const DualMulGenTables* const tables = []() -> const DualMulGenTables* {
            auto* t = new DualMulGenTables;
            Point const G = Point::generator();
            JacobianPoint52 const G52 = to_jac52(G);
            dual_mul_build_table(G52, t->tbl_G, static_cast<std::size_t>(kDualMulGTableSize));
            JacobianPoint52 H52 = G52;
            for (std::size_t i = 0; i < 128; i++) jac52_double_inplace(H52);
            dual_mul_build_table(H52, t->tbl_H, static_cast<std::size_t>(kDualMulGTableSize));
            return t;
        }();
        return tables;
    }
} // anonymous namespace
#endif

#if defined(SECP256K1_FAST_52BIT)
SECP256K1_HOT_FUNCTION
Point Point::dual_scalar_mul_gen_point(const Scalar& a, const Scalar& b, const Point& P) {
    SECP_ASSERT_ON_CURVE(P);
#if defined(SECP256K1_USE_4X64_POINT_OPS)
    // 4x64 path: use two separate scalar_muls (each already has GLV+Shamir 4x64)
    Point aG =
#if defined(_MSC_VER)
        // MSVC Release can diverge when runtime fixed-base tables are reconfigured.
        // Keep the generator leg of verify/recover on the proven CT path.
        ct::generator_mul(a);
#else
        Point::generator().scalar_mul(a);
#endif
    aG.add_inplace(P.scalar_mul(b));
    return aG;
#else
    // -- 128-bit arithmetic split of a --------------------------------
    const auto& a_limbs = a.limbs();
    Scalar const a_lo = Scalar::from_limbs({a_limbs[0], a_limbs[1], 0, 0});
    Scalar const a_hi = Scalar::from_limbs({a_limbs[2], a_limbs[3], 0, 0});

    // -- GLV decompose b only -----------------------------------------
    GLVDecomposition const decomp_b = glv_decompose(b);

    // -- Window widths: w=15 for G (precomputed), w=6 for P (per-call) ---
    constexpr unsigned WINDOW_G = 15;             // -> 2^13 = 8192 entries per G/H table
    constexpr unsigned WINDOW_P = kDualMulWindowP; // -> 2^4 = 16 entries per P/psiP table
    [[maybe_unused]] constexpr int G_TABLE_SIZE = (1 << (WINDOW_G - 2));  // 8192
    constexpr int P_TABLE_SIZE = kDualMulPTableSize;      // 16

    // -- Generator tables (file-scope singleton, shared with dual_scalar_mul_gen_prebuilt) --
    const DualMulGenTables* const gen_tables = get_dual_mul_gen_tables();

    // -- Precompute P tables (per-call, window=5, 8 entries) ---------
    // Uses effective-affine technique (same as libsecp256k1/CT path):
    //   d = 2*P (Jacobian), C = d.Z, iso curve: Y'^2 = X'^3 + C^6*7
    //   On iso curve, d is affine: (d.X, d.Y), so table adds are
    //   mixed Jac+Affine (7M+4S) instead of full Jac+Jac (12M+5S).
    //   Saves ~33M + 6S = ~1.4us per verify.
    JacobianPoint52 const P52 = decomp_b.k1_neg
        ? to_jac52(P.negate())
        : to_jac52(P);

    alignas(64) std::array<AffinePoint52, P_TABLE_SIZE> tbl_P;
    alignas(64) std::array<AffinePoint52, P_TABLE_SIZE> tbl_phiP;

    // Z_shared: effective Z on secp256k1 shared by all pseudo-affine P entries.
    // Declared here so it is available for add_zinv and final Z correction.
    FieldElement52 Z_shared;

    // -- Z-ratio table construction (0 inversions) -------------------
    // build_glv52_table_zr handles: iso-curve mapping, z-ratio collection,
    // backward sweep, and normalization.  All entries share Z = Z_shared.
    if (!build_glv52_table_zr(P52, tbl_P.data(), P_TABLE_SIZE, Z_shared)) {
        // Degenerate case (~2^-256): infinity during table building.
        // Fallback to separate scalar multiplications.
        auto aG = Point::generator().scalar_mul(a);
        aG.add_inplace(P.scalar_mul(b));
        return aG;
    }

    // psi(P) table
    const bool flip_phi_b = (decomp_b.k1_neg != decomp_b.k2_neg);
    derive_phi52_table(tbl_P.data(), tbl_phiP.data(), P_TABLE_SIZE, flip_phi_b);

    // neg tables removed -- negate Y on-the-fly in hot loop

    // -- Compute wNAF for all 4 half-scalars -------------------------
    // GLV half-scalars are <= 128 bits; wNAF produces at most 129 digits.
    // a_lo/a_hi are 128-bit halves of scalar a (same bound).
    // 130 entries = 129 max digits + 1 carry.
    // No {} init needed: compute_wnaf_into() always memsets before writing (avoids double-zero).
    std::array<int32_t, 130> wnaf_a_lo, wnaf_a_hi, wnaf_b1, wnaf_b2;
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
        // Look-ahead prefetch: fetch the NEXT iteration's G/H entries now,
        // so L2/L3 latency is fully hidden behind this iteration's doubling.
        // Previously prefetched current-iteration (same-iter = no hiding).
        if (i > 0) {
            int const np = i - 1;
            int const d_g = wnaf_a_lo[static_cast<std::size_t>(np)];
            if (d_g) {
                int const abs_g = d_g > 0 ? d_g : -d_g;
                SECP256K1_PREFETCH_READ(&gen_tables->tbl_G[static_cast<std::size_t>((abs_g - 1) >> 1)]);
            }
            int const d_h = wnaf_a_hi[static_cast<std::size_t>(np)];
            if (d_h) {
                int const abs_h = d_h > 0 ? d_h : -d_h;
                SECP256K1_PREFETCH_READ(&gen_tables->tbl_H[static_cast<std::size_t>((abs_h - 1) >> 1)]);
            }
            // Prefetch P/ψP tables for next iteration (mirrors dual_scalar_mul_gen_prebuilt).
            int const d_p = wnaf_b1[static_cast<std::size_t>(np)];
            if (SECP256K1_UNLIKELY(d_p != 0)) {
                int const abs_p = d_p > 0 ? d_p : -d_p;
                SECP256K1_PREFETCH_READ(&tbl_P[static_cast<std::size_t>((abs_p - 1) >> 1)]);
            }
            int const d_pp = wnaf_b2[static_cast<std::size_t>(np)];
            if (SECP256K1_UNLIKELY(d_pp != 0)) {
                int const abs_pp = d_pp > 0 ? d_pp : -d_pp;
                SECP256K1_PREFETCH_READ(&tbl_phiP[static_cast<std::size_t>((abs_pp - 1) >> 1)]);
            }
        }

        jac52_double_inplace(result52);

        // G/H table entries are now AffinePoint52 — no to_affine52() conversion needed.
        // Positive digit: pass const ref directly (zero copy). Negative: 80-byte copy + negate.
        {
            int const d = wnaf_a_lo[static_cast<std::size_t>(i)];
            if (SECP256K1_UNLIKELY(d != 0)) {
                if (d > 0) {
                    jac52_add_zinv_inplace(result52,
                        gen_tables->tbl_G[static_cast<std::size_t>((d - 1) >> 1)], Z_shared);
                } else {
                    AffinePoint52 pt = gen_tables->tbl_G[static_cast<std::size_t>((-d - 1) >> 1)];
                    pt.y.negate_assign(1);
                    jac52_add_zinv_inplace(result52, pt, Z_shared);
                }
            }
        }

        {
            int const d = wnaf_a_hi[static_cast<std::size_t>(i)];
            if (SECP256K1_UNLIKELY(d != 0)) {
                if (d > 0) {
                    jac52_add_zinv_inplace(result52,
                        gen_tables->tbl_H[static_cast<std::size_t>((d - 1) >> 1)], Z_shared);
                } else {
                    AffinePoint52 pt = gen_tables->tbl_H[static_cast<std::size_t>((-d - 1) >> 1)];
                    pt.y.negate_assign(1);
                    jac52_add_zinv_inplace(result52, pt, Z_shared);
                }
            }
        }

        apply_wnaf_mixed52(result52, tbl_P.data(), wnaf_b1[static_cast<std::size_t>(i)]);
        apply_wnaf_mixed52(result52, tbl_phiP.data(), wnaf_b2[static_cast<std::size_t>(i)]);
    }

    if (!result52.infinity) {
        result52.z.mul_assign(Z_shared);
    }

    return from_jac52(result52);
#endif // SECP256K1_USE_4X64_POINT_OPS
}

// ===========================================================================
// dual_scalar_mul_gen_prebuilt: a*G + b*P using cached P/phi(P) tables
// ===========================================================================
// Identical to dual_scalar_mul_gen_point but skips build_glv52_table_zr +
// derive_phi52_table (~1,954 ns). Used by schnorr_verify(SchnorrXonlyPubkey).
// ===========================================================================
#if defined(SECP256K1_FAST_52BIT) && !defined(SECP256K1_USE_4X64_POINT_OPS)
Point Point::dual_scalar_mul_gen_prebuilt(
    const Scalar& a, const Scalar& b,
    const std::array<AffinePoint52, kDualMulPTableSize>& tbl_P,
    const std::array<AffinePoint52, kDualMulPTableSize>& tbl_phi_base,
    const FieldElement52& Z_P)
{
    // -- 128-bit split of a ---------------------------------------------------
    const auto& a_limbs = a.limbs();
    Scalar const a_lo = Scalar::from_limbs({a_limbs[0], a_limbs[1], 0, 0});
    Scalar const a_hi = Scalar::from_limbs({a_limbs[2], a_limbs[3], 0, 0});

    // -- GLV decompose b ------------------------------------------------------
    GLVDecomposition const decomp_b = glv_decompose(b);
    // Pre-built tables are canonical (for P, not ±P). Apply signs at lookup:
    //   P table:    negate_y = k1_neg XOR (d<0)  — k1_neg from GLV(b)
    //   phi table:  negate_y = k2_neg XOR (d<0)  — k2_neg from GLV(b)
    bool const k1_neg = decomp_b.k1_neg;
    bool const k2_neg = decomp_b.k2_neg;

    // Window widths (must match dual_scalar_mul_gen_point)
    constexpr unsigned WINDOW_G = kDualMulWindowG;
    constexpr unsigned WINDOW_P = kDualMulWindowP;

    // Shared generator tables (file-scope singleton — same data as dual_scalar_mul_gen_point)
    const DualMulGenTables* const gen_tables = get_dual_mul_gen_tables();

    // -- wNAF for all 4 half-scalars ------------------------------------------
    // No {} init: compute_wnaf_into() memsets before writing (avoids double-zero).
    std::array<int32_t, 130> wnaf_a_lo, wnaf_a_hi, wnaf_b1, wnaf_b2;
    std::size_t len_a_lo = 0, len_a_hi = 0, len_b1 = 0, len_b2 = 0;
    compute_wnaf_into(a_lo,  WINDOW_G, wnaf_a_lo.data(), wnaf_a_lo.size(), len_a_lo);
    compute_wnaf_into(a_hi,  WINDOW_G, wnaf_a_hi.data(), wnaf_a_hi.size(), len_a_hi);
    compute_wnaf_into(decomp_b.k1, WINDOW_P, wnaf_b1.data(), wnaf_b1.size(), len_b1);
    compute_wnaf_into(decomp_b.k2, WINDOW_P, wnaf_b2.data(), wnaf_b2.size(), len_b2);
    while (len_a_lo > 0 && wnaf_a_lo[len_a_lo-1] == 0) --len_a_lo;
    while (len_a_hi > 0 && wnaf_a_hi[len_a_hi-1] == 0) --len_a_hi;
    while (len_b1 > 0 && wnaf_b1[len_b1-1] == 0) --len_b1;
    while (len_b2 > 0 && wnaf_b2[len_b2-1] == 0) --len_b2;

    // -- 4-stream scan (same structure as dual_scalar_mul_gen_point) ----------
    JacobianPoint52 result52 = {
        FieldElement52::zero(), FieldElement52::one(),
        FieldElement52::zero(), true
    };

    std::size_t max_len = len_a_lo;
    if (len_a_hi > max_len) max_len = len_a_hi;
    if (len_b1  > max_len) max_len = len_b1;
    if (len_b2  > max_len) max_len = len_b2;

    for (int i = static_cast<int>(max_len) - 1; i >= 0; --i) {
        // Look-ahead prefetch for G/H tables (same as original)
        if (i > 0) {
            int const np = i - 1;
            int const d_g = wnaf_a_lo[static_cast<std::size_t>(np)];
            if (d_g) {
                int const abs_g = d_g > 0 ? d_g : -d_g;
                SECP256K1_PREFETCH_READ(&gen_tables->tbl_G[static_cast<std::size_t>((abs_g-1)>>1)]);
            }
            int const d_h = wnaf_a_hi[static_cast<std::size_t>(np)];
            if (d_h) {
                int const abs_h = d_h > 0 ? d_h : -d_h;
                SECP256K1_PREFETCH_READ(&gen_tables->tbl_H[static_cast<std::size_t>((abs_h-1)>>1)]);
            }
            // Prefetch P table (50% nonzero for w=5)
            int const d_p = wnaf_b1[static_cast<std::size_t>(np)];
            if (SECP256K1_LIKELY(d_p != 0)) {
                int const abs_p = d_p > 0 ? d_p : -d_p;
                SECP256K1_PREFETCH_READ(&tbl_P[static_cast<std::size_t>((abs_p-1)>>1)]);
            }
            // Prefetch phi(P) table — same density as P stream (w=5, ~50% nonzero)
            int const d_phi = wnaf_b2[static_cast<std::size_t>(np)];
            if (SECP256K1_LIKELY(d_phi != 0)) {
                int const abs_phi = d_phi > 0 ? d_phi : -d_phi;
                SECP256K1_PREFETCH_READ(&tbl_phi_base[static_cast<std::size_t>((abs_phi-1)>>1)]);
            }
        }

        jac52_double_inplace(result52);

        // G/H table entries are AffinePoint52 — no to_affine52() conversion needed.
        {
            int const d = wnaf_a_lo[static_cast<std::size_t>(i)];
            if (SECP256K1_UNLIKELY(d != 0)) {
                if (d > 0) {
                    jac52_add_zinv_inplace(result52,
                        gen_tables->tbl_G[static_cast<std::size_t>((d-1)>>1)], Z_P);
                } else {
                    AffinePoint52 pt = gen_tables->tbl_G[static_cast<std::size_t>((-d-1)>>1)];
                    pt.y.negate_assign(1);
                    jac52_add_zinv_inplace(result52, pt, Z_P);
                }
            }
        }
        {
            int const d = wnaf_a_hi[static_cast<std::size_t>(i)];
            if (SECP256K1_UNLIKELY(d != 0)) {
                if (d > 0) {
                    jac52_add_zinv_inplace(result52,
                        gen_tables->tbl_H[static_cast<std::size_t>((d-1)>>1)], Z_P);
                } else {
                    AffinePoint52 pt = gen_tables->tbl_H[static_cast<std::size_t>((-d-1)>>1)];
                    pt.y.negate_assign(1);
                    jac52_add_zinv_inplace(result52, pt, Z_P);
                }
            }
        }

        // P digit: tbl_P is canonical (built from P). Apply k1_neg via y-negation.
        // Combined negate: k1_neg (P-sign) XOR (d<0) (digit-sign).
        {
            int const d = wnaf_b1[static_cast<std::size_t>(i)];
            if (d != 0) {
                std::size_t const idx = static_cast<std::size_t>(
                    d > 0 ? (d-1)>>1 : (-d-1)>>1);
                AffinePoint52 pt = tbl_P[idx];
                bool const negate_y = k1_neg ^ (d < 0);
                if (negate_y) pt.y.negate_assign(1);
                jac52_add_mixed_inplace(result52, pt);
            }
        }

        // phi(P) digit: tbl_phi_base has canonical y. Apply k2_neg XOR (d<0).
        {
            int const d = wnaf_b2[static_cast<std::size_t>(i)];
            if (d != 0) {
                std::size_t const idx = static_cast<std::size_t>(
                    d > 0 ? (d-1)>>1 : (-d-1)>>1);
                AffinePoint52 pt = tbl_phi_base[idx];
                bool const negate_y = k2_neg ^ (d < 0);
                if (negate_y) pt.y.negate_assign(1);
                jac52_add_mixed_inplace(result52, pt);
            }
        }
    }

    if (!result52.infinity) {
        result52.z.mul_assign(Z_P);
    }

    return from_jac52(result52);
}
#endif // SECP256K1_FAST_52BIT && !SECP256K1_USE_4X64_POINT_OPS

// Build GLV verify tables for a point P (public entry point for schnorr.cpp).
// Uses internal build_glv52_table_zr + derive_phi52_table (no-flip canonical y).
#if defined(SECP256K1_FAST_52BIT) && !defined(SECP256K1_USE_4X64_POINT_OPS)
bool Point::build_schnorr_verify_tables(
    const Point& P,
    std::array<AffinePoint52, kDualMulPTableSize>& out_tbl_P,
    std::array<AffinePoint52, kDualMulPTableSize>& out_tbl_phi_base,
    FieldElement52& out_Z_shared)
{
    if (P.is_infinity()) return false;
    JacobianPoint52 const P52 = to_jac52(P);
    if (!build_glv52_table_zr(P52, out_tbl_P.data(),
                               static_cast<int>(out_tbl_P.size()),
                               out_Z_shared))
        return false;
    // Canonical phi(P) table: y sign not flipped (k2_neg applied at lookup time).
    derive_phi52_table(out_tbl_P.data(), out_tbl_phi_base.data(),
                       static_cast<int>(out_tbl_P.size()), false);
    return true;
}
#endif

// ESP32/Embedded: 4-stream GLV Strauss (4x64 field)
// Fixed in v3.3.1: the phi(G) sign used k1_neg XOR k2_neg (flip_a) instead
// of k2_neg alone. This is correct for the P table (where k1_neg is baked
// into P_base), but wrong for the G table (precomputed once, no sign baked).
#elif (defined(SECP256K1_PLATFORM_ESP32) || defined(ESP_PLATFORM) || defined(SECP256K1_PLATFORM_STM32))
// -- ESP32/Embedded: 4-stream GLV Strauss (4x64 field) --------------------
// Combines a*G + b*P into a single doubling chain with 4 wNAF streams,
// halving the doublings compared to two separate scalar_mul calls.
// Expected speedup: ~25-30% on ECDSA verify.
SECP256K1_HOT_FUNCTION
Point Point::dual_scalar_mul_gen_point(const Scalar& a, const Scalar& b, const Point& P) {

    // -- GLV decompose both scalars ------------------------------------
    GLVDecomposition decomp_a = glv_decompose(a);
    GLVDecomposition decomp_b = glv_decompose(b);

    // -- Build wNAF w=5 for all 4 half-scalars ------------------------
    constexpr unsigned WINDOW = 5;
    constexpr int TABLE_SIZE = (1 << (WINDOW - 2));  // 8

    std::array<int32_t, 140> wnaf_a1{}, wnaf_a2{}, wnaf_b1{}, wnaf_b2{}; // all ≤128-bit
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

    // -- Precompute P tables (per-call: 8 odd multiples + endomorphism only) --
    // Negated variants removed: y is negated on-the-fly in the hot loop
    // (~5ns negate_assign vs building 2x8 extra AffinePoints = ~16 field ops).
    Point P_base = decomp_b.k1_neg ? P.negate() : P;
    std::array<AffinePoint, TABLE_SIZE> tbl_P, tbl_phiP;

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
        bool const flip_phi = (decomp_b.k1_neg != decomp_b.k2_neg);

        for (int i = 0; i < TABLE_SIZE; i++) {
            FieldElement zi2 = z_inv[i].square();
            FieldElement zi3 = zi2 * z_inv[i];
            FieldElement const px = pts_j[i].x_raw() * zi2;
            FieldElement const py = pts_j[i].y_raw() * zi3;

            tbl_P[i] = {px, py};

            FieldElement const phix = px * beta;
            FieldElement const phiy = flip_phi ? py.negate() : py;
            tbl_phiP[i] = {phix, phiy};
        }
    }

    // G tables: pre-negated variants live in the static singleton (computed once).
    const AffinePoint* g_pos  = decomp_a.k1_neg ? gen4.neg_tbl_G : gen4.tbl_G;
    const AffinePoint* g_neg  = decomp_a.k1_neg ? gen4.tbl_G : gen4.neg_tbl_G;
    const AffinePoint* pg_pos = decomp_a.k2_neg ? gen4.neg_tbl_phiG : gen4.tbl_phiG;
    const AffinePoint* pg_neg = decomp_a.k2_neg ? gen4.tbl_phiG : gen4.neg_tbl_phiG;

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
            int32_t const d = wnaf_a1[static_cast<std::size_t>(i)];
            if (d > 0) jacobian_add_mixed_inplace(jac, g_pos[(d-1)>>1]);
            else if (d < 0) jacobian_add_mixed_inplace(jac, g_neg[(-d-1)>>1]);
        }

        // Stream 2: a2 * phi(G) (affine tables -> mixed add)
        {
            int32_t const d = wnaf_a2[static_cast<std::size_t>(i)];
            if (d > 0) jacobian_add_mixed_inplace(jac, pg_pos[(d-1)>>1]);
            else if (d < 0) jacobian_add_mixed_inplace(jac, pg_neg[(-d-1)>>1]);
        }

        // Streams 3+4: P and phi(P).
        // Positive digits: pass table entry by const ref (no copy, same as G streams).
        // Negative digits: local copy + negate_assign (~5ns) then pass by ref.
        {
            int32_t const d = wnaf_b1[static_cast<std::size_t>(i)];
            if (d > 0) {
                jacobian_add_mixed_inplace(jac, tbl_P[static_cast<std::size_t>((d - 1) >> 1)]);
            } else if (d < 0) {
                AffinePoint q = tbl_P[static_cast<std::size_t>((-d - 1) >> 1)];
                q.y.negate_assign();
                jacobian_add_mixed_inplace(jac, q);
            }
        }

        {
            int32_t const d = wnaf_b2[static_cast<std::size_t>(i)];
            if (d > 0) {
                jacobian_add_mixed_inplace(jac, tbl_phiP[static_cast<std::size_t>((d - 1) >> 1)]);
            } else if (d < 0) {
                AffinePoint q = tbl_phiP[static_cast<std::size_t>((-d - 1) >> 1)];
                q.y.negate_assign();
                jacobian_add_mixed_inplace(jac, q);
            }
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
    SECP_ASSERT_ON_CURVE(P);
    auto aG = Point::generator().scalar_mul(a);
    aG.add_inplace(P.scalar_mul(b));
    return aG;
}
#endif

} // namespace secp256k1::fast
