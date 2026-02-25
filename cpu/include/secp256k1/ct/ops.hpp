#ifndef SECP256K1_CT_OPS_HPP
#define SECP256K1_CT_OPS_HPP

// ============================================================================
// Constant-Time Primitives
// ============================================================================
// Low-level building blocks for side-channel resistant code.
// Every function in this header has a data-independent execution trace:
//   - No secret-dependent branches
//   - No secret-dependent memory access patterns
//   - Fixed instruction count regardless of input
//
// These primitives use volatile barriers to prevent the compiler from
// converting branchless code into branches via if-conversion.
//
// Usage:
//   #include <secp256k1/ct/ops.hpp>
//   uint64_t mask = secp256k1::ct::is_zero_mask(x);  // 0 or 0xFFFF...
//   secp256k1::ct::cmov64(&dst, &src, flag);          // if(flag) dst=src
// ============================================================================

#include <cstdint>
#include <cstddef>

// --- Declassify / Classify Markers -------------------------------------------
// For constant-time verification with Valgrind (memcheck) or MSAN.
//
// SECP256K1_CLASSIFY(ptr, len)   -- Mark memory as secret (undefined).
//                                  Call on inputs before CT operations.
// SECP256K1_DECLASSIFY(ptr, len) -- Mark memory as public (defined).
//                                  Call on outputs after CT operations.
//
// Under normal compilation these are no-ops. When compiled with:
//   -DSECP256K1_CT_VALGRIND=1 and Valgrind headers available:
//     maps to VALGRIND_MAKE_MEM_{UNDEFINED,DEFINED}
//
// Usage in tests:
//   Scalar k;
//   SECP256K1_CLASSIFY(&k, sizeof(k));   // treat k as secret
//   Point R = ct::scalar_mul(G, k);      // CT operation under test
//   SECP256K1_DECLASSIFY(&R, sizeof(R)); // declassify result for comparison
//
// If valgrind reports a "conditional jump depends on uninitialised value"
// error BETWEEN classify and declassify, the code has a CT violation.
// -----------------------------------------------------------------------------

#if defined(SECP256K1_CT_VALGRIND) && SECP256K1_CT_VALGRIND
    #include <valgrind/memcheck.h>
    #define SECP256K1_CLASSIFY(ptr, len)   VALGRIND_MAKE_MEM_UNDEFINED((ptr), (len))
    #define SECP256K1_DECLASSIFY(ptr, len) VALGRIND_MAKE_MEM_DEFINED((ptr), (len))
#else
    #define SECP256K1_CLASSIFY(ptr, len)   ((void)(ptr), (void)(len))
    #define SECP256K1_DECLASSIFY(ptr, len) ((void)(ptr), (void)(len))
#endif

namespace secp256k1::ct {

// --- Compiler barrier --------------------------------------------------------
// Prevents compiler from optimizing away branchless patterns.
// Uses inline asm (GCC/Clang) or volatile (MSVC) to create optimization barrier.

#if defined(__GNUC__) || defined(__clang__)
    // Force value through a register so compiler cannot reason about it
    inline void value_barrier(std::uint64_t& v) noexcept {
        asm volatile("" : "+r"(v) : : "memory");
    }
    inline void value_barrier(std::uint32_t& v) noexcept {
        asm volatile("" : "+r"(v) : : "memory");
    }
#else
    // MSVC: volatile prevents optimization of subsequent operations
    inline void value_barrier(std::uint64_t& v) noexcept {
        volatile std::uint64_t sink = v;
        v = sink;
    }
    inline void value_barrier(std::uint32_t& v) noexcept {
        volatile std::uint32_t sink = v;
        v = sink;
    }
#endif

// --- Mask generation ---------------------------------------------------------

// Returns 0xFFFFFFFFFFFFFFFF if v == 0, else 0x0000000000000000
inline std::uint64_t is_zero_mask(std::uint64_t v) noexcept {
#if defined(__riscv) && (__riscv_xlen == 64)
    // RISC-V: seqz + neg produces fully branchless is-zero mask.
    //   seqz tmp, v   →  tmp = (v == 0) ? 1 : 0
    //   neg  tmp, tmp →  tmp = 0 - tmp  (all-ones if was 1, zero if was 0)
    // asm volatile prevents the compiler from reasoning about the output,
    // so downstream code stays branchless.
    std::uint64_t mask;
    asm volatile(
        "seqz %0, %1\n\t"
        "neg   %0, %0"
        : "=r"(mask) : "r"(v));
    return mask;
#else
    // ~(v | -v) has MSB set iff v == 0
    value_barrier(v);   // prevent compiler from recognising v's value range
    std::uint64_t nv = -v;
    value_barrier(nv);  // prevent compiler from knowing nv == -v
    std::uint64_t mask = static_cast<std::uint64_t>(
        -static_cast<std::int64_t>((~(v | nv)) >> 63));
    value_barrier(mask); // prevent compiler from converting result into branch
    return mask;
#endif
}

// Returns 0xFFFFFFFFFFFFFFFF if v != 0, else 0x0000000000000000
inline std::uint64_t is_nonzero_mask(std::uint64_t v) noexcept {
    return ~is_zero_mask(v);
}

// Returns 0xFFFFFFFFFFFFFFFF if a == b, else 0x0000000000000000
inline std::uint64_t eq_mask(std::uint64_t a, std::uint64_t b) noexcept {
    return is_zero_mask(a ^ b);
}

// Returns 0xFFFFFFFFFFFFFFFF if flag is true (nonzero), else 0
inline std::uint64_t bool_to_mask(bool flag) noexcept {
    std::uint64_t v = static_cast<std::uint64_t>(flag);
    value_barrier(v);
    std::uint64_t mask = -v;
    value_barrier(mask); // prevent converting to branch
    return mask;
}

// Returns 0xFFFFFFFFFFFFFFFF if a < b (unsigned), else 0
// Uses the borrow bit from subtraction
inline std::uint64_t lt_mask(std::uint64_t a, std::uint64_t b) noexcept {
    // If a < b, then (a - b) borrows, so bit 64 of the extended result is 1
    // We compute this as: a < b  <=>  ~a >= b  <=>  we check borrow
    std::uint64_t diff = a - b;
    // Borrow occurred iff a < b. The borrow is in the "carry out" position.
    // For unsigned: a < b iff the subtract wraps, iff (a ^ ((a ^ b) | (diff ^ a))) has MSB set
    std::uint64_t borrow = (a ^ ((a ^ b) | (diff ^ a))) >> 63;
    value_barrier(borrow);
    return -borrow;
}

// --- Conditional move (CT) ---------------------------------------------------
// if (flag) *dst = *src;  -- constant time, no branch

inline void cmov64(std::uint64_t* dst, const std::uint64_t* src,
                   std::uint64_t mask) noexcept {
    // dst = (src & mask) | (dst & ~mask)
    *dst ^= (*dst ^ *src) & mask;
}

// Conditional move for 4-limb (256-bit) values
inline void cmov256(std::uint64_t dst[4], const std::uint64_t src[4],
                    std::uint64_t mask) noexcept {
    dst[0] ^= (dst[0] ^ src[0]) & mask;
    dst[1] ^= (dst[1] ^ src[1]) & mask;
    dst[2] ^= (dst[2] ^ src[2]) & mask;
    dst[3] ^= (dst[3] ^ src[3]) & mask;
}

// --- Conditional swap (CT) ---------------------------------------------------
// if (flag) swap(*a, *b);  -- constant time

inline void cswap256(std::uint64_t a[4], std::uint64_t b[4],
                     std::uint64_t mask) noexcept {
    for (int i = 0; i < 4; ++i) {
        std::uint64_t diff = (a[i] ^ b[i]) & mask;
        a[i] ^= diff;
        b[i] ^= diff;
    }
}

// --- Constant-time select ----------------------------------------------------
// Returns a if mask == 0xFFF...F, else b (mask must be all-zeros or all-ones)

inline std::uint64_t ct_select(std::uint64_t a, std::uint64_t b,
                                std::uint64_t mask) noexcept {
    return (a & mask) | (b & ~mask);
}

// --- Constant-time table lookup ----------------------------------------------
// Always reads ALL entries. No secret-dependent memory access pattern.
// table: pointer to N entries of 'stride' bytes each
// index: which entry to select (0-based)
// out: output buffer (stride bytes)

inline void ct_lookup(const void* table, std::size_t count,
                      std::size_t stride, std::size_t index,
                      void* out) noexcept {
    // Zero the output
    auto* dst = static_cast<std::uint8_t*>(out);
    for (std::size_t b = 0; b < stride; ++b) dst[b] = 0;

    const auto* src = static_cast<const std::uint8_t*>(table);
    for (std::size_t i = 0; i < count; ++i) {
        std::uint64_t mask = eq_mask(static_cast<std::uint64_t>(i),
                                     static_cast<std::uint64_t>(index));
        auto mask8 = static_cast<std::uint8_t>(mask & 0xFF);
        for (std::size_t b = 0; b < stride; ++b) {
            dst[b] |= src[i * stride + b] & mask8;
        }
    }
}

// Specialized: CT lookup for 4-limb (256-bit) entries
inline void ct_lookup_256(const std::uint64_t table[][4], std::size_t count,
                          std::size_t index, std::uint64_t out[4]) noexcept {
    out[0] = out[1] = out[2] = out[3] = 0;
    for (std::size_t i = 0; i < count; ++i) {
        std::uint64_t mask = eq_mask(static_cast<std::uint64_t>(i),
                                     static_cast<std::uint64_t>(index));
        out[0] |= table[i][0] & mask;
        out[1] |= table[i][1] & mask;
        out[2] |= table[i][2] & mask;
        out[3] |= table[i][3] & mask;
    }
}

} // namespace secp256k1::ct

#endif // SECP256K1_CT_OPS_HPP
