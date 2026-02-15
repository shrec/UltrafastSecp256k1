// Optimized field operations using BMI2 intrinsics (MULX/ADCX/ADOX)

#ifndef C237C0F0_BF55_4453_9221_DE161395FF08
#define C237C0F0_BF55_4453_9221_DE161395FF08
// Target: 7-8x speedup on field multiplication/squaring
// For K constant + Q variable optimization


#include "field.hpp"
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>  // BMI2 intrinsics
#endif

namespace secp256k1::fast {

// Check if BMI2 is available at runtime
bool has_bmi2_support();
// Check if ADX (ADCX/ADOX) is available at runtime
bool has_adx_support();

// ===================================================================
// BMI2-optimized field operations
// ===================================================================

// Multiply two field elements using BMI2 instructions
// Expected: ~10-15 ns (vs current ~70-80 ns)
FieldElement field_mul_bmi2(const FieldElement& a, const FieldElement& b);

// Square a field element using BMI2 instructions
// Expected: ~8-10 ns (vs current ~50-60 ns)
FieldElement field_square_bmi2(const FieldElement& a);

// ===================================================================
// ARM64 (AArch64) assembly optimizations
// ===================================================================

#if defined(__aarch64__) || defined(_M_ARM64)
#define SECP256K1_HAS_ARM64_ASM 1

// Internal ARM64 assembly functions (direct pointer interface for hot paths)
namespace arm64 {
    void field_mul_arm64(uint64_t out[4], const uint64_t a[4], const uint64_t b[4]) noexcept;
    void field_sqr_arm64(uint64_t out[4], const uint64_t a[4]) noexcept;
    void field_add_arm64(uint64_t out[4], const uint64_t a[4], const uint64_t b[4]) noexcept;
    void field_sub_arm64(uint64_t out[4], const uint64_t a[4], const uint64_t b[4]) noexcept;
    void field_neg_arm64(uint64_t out[4], const uint64_t a[4]) noexcept;
} // namespace arm64

// Multiply two field elements using ARM64 MUL/UMULH assembly
// ~50-80 cycles on Cortex-A76, ~80-120 on Cortex-A55
FieldElement field_mul_arm64(const FieldElement& a, const FieldElement& b);

// Square a field element using ARM64 assembly (10 muls vs 16)
FieldElement field_square_arm64(const FieldElement& a);

// Add/Sub with branchless normalization
FieldElement field_add_arm64(const FieldElement& a, const FieldElement& b);
FieldElement field_sub_arm64(const FieldElement& a, const FieldElement& b);

// Negate field element (branchless p - a)
FieldElement field_negate_arm64(const FieldElement& a);

#endif // __aarch64__ || _M_ARM64

// ===================================================================
// RISC-V assembly optimizations (RV64GC)
// ===================================================================

#ifdef SECP256K1_HAS_RISCV_ASM
// Multiply two field elements using RISC-V assembly
// Expected: 2-3x speedup over portable C++
FieldElement field_mul_riscv(const FieldElement& a, const FieldElement& b);

// Square a field element using RISC-V assembly
FieldElement field_square_riscv(const FieldElement& a);

// Add two field elements using RISC-V assembly
FieldElement field_add_riscv(const FieldElement& a, const FieldElement& b);

// Subtract two field elements using RISC-V assembly
FieldElement field_sub_riscv(const FieldElement& a, const FieldElement& b);

// Negate a field element using RISC-V assembly
FieldElement field_negate_riscv(const FieldElement& a);
#endif // SECP256K1_HAS_RISCV_ASM

// Square using Karatsuba algorithm (recursive decomposition)
// ~9 multiplications vs 10 in standard, may be faster for some CPUs
FieldElement field_square_karatsuba(const FieldElement& a);

// Square using Toom-Cook-3 algorithm (3-way split)
// Theoretical best complexity but high overhead for 256-bit
FieldElement field_square_toomcook(const FieldElement& a);

// Add two field elements (already fast, but optimize with ADCX)
// Expected: ~5-8 ns (vs current ~15-20 ns)
FieldElement field_add_bmi2(const FieldElement& a, const FieldElement& b);

// Negate field element (conditional subtraction)
// Expected: ~3-5 ns (vs current ~10-15 ns)
FieldElement field_negate_bmi2(const FieldElement& a);

// ===================================================================
// Internal implementation details
// ===================================================================

namespace detail {

// 64x64 → 128-bit multiplication using MULX
// MULX: multiplicand in RDX, result in two registers (no flag updates!)
inline void mulx64(uint64_t a, uint64_t b, uint64_t& lo, uint64_t& hi) {
    #if defined(_MSC_VER)
        // MSVC intrinsic
        lo = _mulx_u64(a, b, &hi);
    #elif defined(__GNUC__) || defined(__clang__)
        #if defined(__BMI2__)
            // GCC/Clang intrinsic with BMI2
            lo = _mulx_u64(a, b, (unsigned long long*)&hi);
        #else
            // GCC/Clang fallback
            #ifdef SECP256K1_NO_INT128
                // 32-bit safe implementation
                uint64_t a_lo = a & 0xFFFFFFFFULL;
                uint64_t a_hi = a >> 32;
                uint64_t b_lo = b & 0xFFFFFFFFULL;
                uint64_t b_hi = b >> 32;

                uint64_t p0 = a_lo * b_lo;
                uint64_t p1 = a_lo * b_hi;
                uint64_t p2 = a_hi * b_lo;
                uint64_t p3 = a_hi * b_hi;

                uint64_t carry = ((p0 >> 32) + (p1 & 0xFFFFFFFFULL) + (p2 & 0xFFFFFFFFULL)) >> 32;

                lo = p0 + (p1 << 32) + (p2 << 32);
                hi = p3 + (p1 >> 32) + (p2 >> 32) + carry;
            #else
                __uint128_t result = static_cast<__uint128_t>(a) * b;
                lo = static_cast<uint64_t>(result);
                hi = static_cast<uint64_t>(result >> 64);
            #endif
        #endif
    #else
        // Fallback
        // #ifdef SECP256K1_NO_INT128
        //     // 32-bit safe implementation
        //     uint64_t a_lo = a & 0xFFFFFFFFULL;
        //     uint64_t a_hi = a >> 32;
        //     uint64_t b_lo = b & 0xFFFFFFFFULL;
        //     uint64_t b_hi = b >> 32;
        //
        //     uint64_t p0 = a_lo * b_lo;
        //     uint64_t p1 = a_lo * b_hi;
        //     uint64_t p2 = a_hi * b_lo;
        //     uint64_t p3 = a_hi * b_hi;
        //
        //     uint64_t carry = ((p0 >> 32) + (p1 & 0xFFFFFFFFULL) + (p2 & 0xFFFFFFFFULL)) >> 32;
        //
        //     lo = p0 + (p1 << 32) + (p2 << 32);
        //     hi = p3 + (p1 >> 32) + (p2 >> 32) + carry;
        // #else
        //     __uint128_t result = static_cast<__uint128_t>(a) * b;
        //     lo = static_cast<uint64_t>(result);
        //     hi = static_cast<uint64_t>(result >> 64);
        // #endif
    #endif
}

// Add with carry using ADCX (carry flag chain)
// x86-only: BMI2/ADX intrinsic
#if defined(__x86_64__) || defined(_M_X64)
inline uint8_t adcx64(uint64_t a, uint64_t b, uint8_t carry, uint64_t& result) {
    #if defined(_MSC_VER)
        // MSVC intrinsic
        return _addcarry_u64(carry, a, b, reinterpret_cast<unsigned long long*>(&result));
    #else
        // GCC/Clang intrinsic (_addcarry_u64 in x86intrin.h/immintrin.h)
        return _addcarry_u64(carry, a, b, (unsigned long long*)&result);
    #endif
}
#else
// Portable fallback for non-x86 (RISC-V, ARM, ESP32, etc.)
inline uint8_t adcx64(uint64_t a, uint64_t b, uint8_t carry, uint64_t& result) {
    #ifdef SECP256K1_NO_INT128
        // 32-bit safe implementation
        result = a + b;
        uint8_t new_carry = (result < a) ? 1 : 0;
        if (carry) {
            uint64_t temp = result + 1;
            new_carry |= (temp < result) ? 1 : 0;
            result = temp;
        }
        return new_carry;
    #else
        unsigned __int128 sum = static_cast<unsigned __int128>(a) +
                                static_cast<unsigned __int128>(b) +
                                static_cast<unsigned __int128>(carry);
        result = static_cast<uint64_t>(sum);
        return static_cast<uint8_t>(sum >> 64);
    #endif
}
#endif

// Add with overflow using ADOX (overflow flag chain, independent of ADCX!)
inline uint8_t adox64(uint64_t a, uint64_t b, uint8_t overflow, uint64_t& result) {
    // Note: ADOX not directly available as intrinsic, use ADCX for now
    // In inline assembly, we'd use separate flag chains
    return adcx64(a, b, overflow, result);
}

// Full 4x4 limb multiplication (256-bit × 256-bit → 512-bit)
// Fully unrolled, using MULX/ADCX/ADOX in parallel
void mul_4x4_bmi2(
    const uint64_t a[4], 
    const uint64_t b[4], 
    uint64_t result[8]
);

// Full 4-limb squaring (optimized, fewer multiplications)
void square_4_bmi2(
    const uint64_t a[4],
    uint64_t result[8]
);

// Karatsuba squaring algorithm (recursive decomposition)
// ~9 multiplications vs 10 in standard approach
void square_4_karatsuba(
    const uint64_t a[4],
    uint64_t result[8]
);

// Toom-Cook-3 squaring algorithm (3-way split)
// Theoretical ~5 multiplications but high addition overhead
// Best for 512+ bit numbers
void square_4_toomcook(
    const uint64_t a[4],
    uint64_t result[8]
);

// Montgomery reduction modulo p (secp256k1 prime)
// Input: 512-bit value in result[8]
// Output: 256-bit reduced value
void montgomery_reduce_bmi2(uint64_t result[8]);

} // namespace detail

} // namespace secp256k1::fast


#endif /* C237C0F0_BF55_4453_9221_DE161395FF08 */
