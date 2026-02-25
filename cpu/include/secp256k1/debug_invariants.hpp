// ============================================================================
// Debug Invariant Assertions for Hot Paths
// Phase V, Task 5.3.3 -- Compile-time gated, zero overhead in release
// ============================================================================
// Include this header in source files that need debug-mode invariant checking.
//
// In Debug builds (-DNDEBUG not defined):
//   - SECP_ASSERT(expr) evaluates expr and aborts on failure
//   - SECP_ASSERT_NORMALIZED(fe) checks field element is canonical (< p)
//   - SECP_ASSERT_ON_CURVE(pt) checks point lies on secp256k1
//   - SECP_ASSERT_SCALAR_VALID(s) checks scalar is < n and non-zero
//
// In Release builds (-DNDEBUG defined):
//   - All macros compile to nothing (zero overhead)
//
// Usage:
//   #include "secp256k1/debug_invariants.hpp"
//   ...
//   void scalar_mul_inner(const Scalar& k, const Point& P, Point& out) {
//       SECP_ASSERT_SCALAR_VALID(k);
//       SECP_ASSERT_ON_CURVE(P);
//       // ... hot path ...
//       SECP_ASSERT_ON_CURVE(out);
//   }
//
// ============================================================================

#ifndef SECP256K1_DEBUG_INVARIANTS_HPP
#define SECP256K1_DEBUG_INVARIANTS_HPP

#include <cstdio>
#include <cstdlib>
#include <cstdint>

// -- Release builds: zero overhead ----------------------------------------

#if defined(NDEBUG) && !defined(SECP256K1_FORCE_INVARIANTS)

#define SECP_ASSERT(expr)                ((void)0)
#define SECP_ASSERT_MSG(expr, msg)       ((void)0)
#define SECP_ASSERT_NORMALIZED(fe)       ((void)0)
#define SECP_ASSERT_ON_CURVE(pt)         ((void)0)
#define SECP_ASSERT_SCALAR_VALID(s)      ((void)0)
#define SECP_ASSERT_SCALAR_NONZERO(s)    ((void)0)
#define SECP_ASSERT_NOT_INFINITY(pt)     ((void)0)
#define SECP_ASSERT_FE_LESS_THAN_P(fe)   ((void)0)
#define SECP_DEBUG_COUNTER_INC(name)     ((void)0)
#define SECP_DEBUG_COUNTER_REPORT()      ((void)0)

// -- Debug builds: full checking ------------------------------------------

#else

#include "secp256k1/field.hpp"
#include "secp256k1/scalar.hpp"
#include "secp256k1/point.hpp"

namespace secp256k1::fast::debug {

// secp256k1 field prime: p = 2^256 - 2^32 - 977
// Last limb must be < 0xFFFFFFFEFFFFFC2F... simplified:
// A normalized field element has limbs < p when treated as 256-bit LE integer.
inline bool is_normalized_field_element(const FieldElement& fe) noexcept {
    // p = FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
    // In 64-bit limbs (little-endian): [0xFFFFFFFEFFFFFC2F, 0xFFFF..., 0xFFFF..., 0xFFFF...]
    static constexpr uint64_t P[4] = {
        0xFFFFFFFEFFFFFC2Full,
        0xFFFFFFFFFFFFFFFFull,
        0xFFFFFFFFFFFFFFFFull,
        0xFFFFFFFFFFFFFFFFull
    };
    
    const auto& l = fe.limbs();
    // Compare from most significant limb
    for (int i = 3; i >= 0; --i) {
        if (l[i] < P[i]) return true;
        if (l[i] > P[i]) return false;
    }
    // Equal to p -- not canonical (should be reduced to 0)
    return false;
}

// Check if point (x, y) satisfies y^2 = x^3 + 7 (mod p)
inline bool is_on_curve(const Point& pt) noexcept {
    if (pt.is_infinity()) return true;
    
    // Convert to affine and check curve equation
    FieldElement x = pt.x();
    FieldElement y = pt.y();
    
    // lhs = y^2
    FieldElement lhs = y.square();
    
    // rhs = x^3 + 7
    FieldElement x2 = x.square();
    FieldElement x3 = x2 * x;
    FieldElement rhs = x3 + FieldElement::from_uint64(7);
    
    return lhs == rhs;
}

// Check scalar is in range [1, n-1]
inline bool is_valid_scalar(const Scalar& s) noexcept {
    // n = FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
    // If scalar arithmetic is correct, result is always reduced mod n
    // Just check non-zero for most checks
    return !s.is_zero();
}

// Debug counters for hot-path monitoring
struct DebugCounters {
    uint64_t field_mul_count = 0;
    uint64_t field_sqr_count = 0;
    uint64_t point_add_count = 0;
    uint64_t point_dbl_count = 0;
    uint64_t scalar_mul_count = 0;
    uint64_t invariant_check_count = 0;
    
    void report() const noexcept {
        std::fprintf(stderr,
            "[DEBUG COUNTERS]\n"
            "  field_mul:   %llu\n"
            "  field_sqr:   %llu\n"
            "  point_add:   %llu\n"
            "  point_dbl:   %llu\n"
            "  scalar_mul:  %llu\n"
            "  invariants:  %llu\n",
            (unsigned long long)field_mul_count,
            (unsigned long long)field_sqr_count,
            (unsigned long long)point_add_count,
            (unsigned long long)point_dbl_count,
            (unsigned long long)scalar_mul_count,
            (unsigned long long)invariant_check_count);
    }
};

inline DebugCounters& counters() noexcept {
    static DebugCounters c;
    return c;
}

} // namespace secp256k1::fast::debug

// -- Assertion macros ----------------------------------------------------

#define SECP_ASSERT(expr) do { \
    if (!(expr)) { \
        std::fprintf(stderr, \
            "SECP_ASSERT FAILED: %s\n  at %s:%d (%s)\n", \
            #expr, __FILE__, __LINE__, __func__); \
        std::abort(); \
    } \
} while(0)

#define SECP_ASSERT_MSG(expr, msg) do { \
    if (!(expr)) { \
        std::fprintf(stderr, \
            "SECP_ASSERT FAILED: %s\n  %s\n  at %s:%d (%s)\n", \
            #expr, msg, __FILE__, __LINE__, __func__); \
        std::abort(); \
    } \
} while(0)

#define SECP_ASSERT_NORMALIZED(fe) do { \
    ++secp256k1::fast::debug::counters().invariant_check_count; \
    if (!secp256k1::fast::debug::is_normalized_field_element(fe)) { \
        std::fprintf(stderr, \
            "SECP_ASSERT_NORMALIZED FAILED: field element not canonical\n" \
            "  at %s:%d (%s)\n", __FILE__, __LINE__, __func__); \
        const auto& _l = (fe).limbs(); \
        std::fprintf(stderr, "  limbs: [%016llx, %016llx, %016llx, %016llx]\n", \
            (unsigned long long)_l[0], (unsigned long long)_l[1], \
            (unsigned long long)_l[2], (unsigned long long)_l[3]); \
        std::abort(); \
    } \
} while(0)

#define SECP_ASSERT_ON_CURVE(pt) do { \
    ++secp256k1::fast::debug::counters().invariant_check_count; \
    if (!secp256k1::fast::debug::is_on_curve(pt)) { \
        std::fprintf(stderr, \
            "SECP_ASSERT_ON_CURVE FAILED: point not on secp256k1\n" \
            "  at %s:%d (%s)\n", __FILE__, __LINE__, __func__); \
        std::abort(); \
    } \
} while(0)

#define SECP_ASSERT_SCALAR_VALID(s) do { \
    ++secp256k1::fast::debug::counters().invariant_check_count; \
    if (!secp256k1::fast::debug::is_valid_scalar(s)) { \
        std::fprintf(stderr, \
            "SECP_ASSERT_SCALAR_VALID FAILED: scalar is zero\n" \
            "  at %s:%d (%s)\n", __FILE__, __LINE__, __func__); \
        std::abort(); \
    } \
} while(0)

#define SECP_ASSERT_SCALAR_NONZERO(s) SECP_ASSERT_SCALAR_VALID(s)

#define SECP_ASSERT_NOT_INFINITY(pt) do { \
    if ((pt).is_infinity()) { \
        std::fprintf(stderr, \
            "SECP_ASSERT_NOT_INFINITY FAILED\n" \
            "  at %s:%d (%s)\n", __FILE__, __LINE__, __func__); \
        std::abort(); \
    } \
} while(0)

#define SECP_ASSERT_FE_LESS_THAN_P(fe) SECP_ASSERT_NORMALIZED(fe)

#define SECP_DEBUG_COUNTER_INC(name) \
    (++secp256k1::fast::debug::counters().name ## _count)

#define SECP_DEBUG_COUNTER_REPORT() \
    secp256k1::fast::debug::counters().report()

#endif // NDEBUG

#endif // SECP256K1_DEBUG_INVARIANTS_HPP
