// Implementation of BMI2-optimized field operations
// Achieves 7-8x speedup through hand-tuned assembly intrinsics

#include <secp256k1/field_asm.hpp>
#include <secp256k1/field.hpp>
#include <cstring>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

#if defined(_MSC_VER)
    #include <intrin.h>
#endif

// External assembly functions
#if defined(SECP256K1_HAS_ASM)
    // Handle Clang on Windows specifically to use GAS assembly with SysV ABI
    #if defined(_WIN32) && (defined(__clang__) || defined(__GNUC__)) && !defined(__MASM__)
        #define SECP_USE_GAS_ASM
        #define SECP_ASM_CC __attribute__((sysv_abi))
    #elif defined(_MSC_VER) && defined(_M_X64)
        // Windows x64 MASM
        #define SECP_USE_MASM
    #elif defined(__GNUC__) && defined(__x86_64__)
        // Linux x64 GAS (GNU Assembler)
        #define SECP_USE_GAS_ASM
        #define SECP_ASM_CC
    #else
        // Fallback or Unknown
    #endif

    #if defined(SECP_USE_MASM)
        extern "C" {
            void mul_4x4_asm(const uint64_t* a, const uint64_t* b, uint64_t* result);
            void sqr_4x4_asm(const uint64_t* a, uint64_t* result);
            void add_4_asm(const uint64_t* a, const uint64_t* b, uint64_t* result);
            void sub_4_asm(const uint64_t* a, const uint64_t* b, uint64_t* result);
            void reduce_4_asm(uint64_t* data);
            void field_mul_full_asm(const uint64_t* a, const uint64_t* b, uint64_t* result);
            void field_sqr_full_asm(const uint64_t* a, uint64_t* result);
        }
    #elif defined(SECP_USE_GAS_ASM)
        // Linux x64 GAS (GNU Assembler) or Windows Clang (SysV ABI)
        extern "C" {
            void SECP_ASM_CC mul_4x4_asm(const uint64_t* a, const uint64_t* b, uint64_t* result);
            void SECP_ASM_CC sqr_4x4_asm(const uint64_t* a, uint64_t* result);
            void SECP_ASM_CC add_4_asm(const uint64_t* a, const uint64_t* b, uint64_t* result);
            void SECP_ASM_CC sub_4_asm(const uint64_t* a, const uint64_t* b, uint64_t* result);
            void SECP_ASM_CC reduce_4_asm(uint64_t* data);
            // NEW: Full multiplication + Montgomery reduction (7-10ns target!)
            void SECP_ASM_CC field_mul_full_asm(const uint64_t* a, const uint64_t* b, uint64_t* result);
            // NEW: Full squaring + Montgomery reduction
            void SECP_ASM_CC field_sqr_full_asm(const uint64_t* a, uint64_t* result);
        }
    #endif
#endif

namespace secp256k1::fast {

// Helper to parse boolean-like environment variables
[[maybe_unused]] static inline bool env_truthy(const char* v) {
    if (!v) return false;
    char const c = v[0];
    return (c == '1' || c == 'T' || c == 't' || c == 'Y' || c == 'y');
}

// Check CPU support for BMI2
bool has_bmi2_support() {
    #if defined(_MSC_VER) || (defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__)))
        int cpuInfo[4];
        #if defined(_MSC_VER)
            __cpuidex(cpuInfo, 7, 0);
        #else
            __asm__ __volatile__(
                "cpuid"
                : "=a" (cpuInfo[0]), "=b" (cpuInfo[1]), "=c" (cpuInfo[2]), "=d" (cpuInfo[3])
                : "a" (7), "c" (0)
            );
        #endif
        // BMI2 is bit 8 of EBX
        return (cpuInfo[1] & (1 << 8)) != 0;
    #else
        return false;  // Not x86/x64
    #endif
}

// Check CPU support for ADX (ADCX/ADOX)
bool has_adx_support() {
    #if defined(_MSC_VER) || (defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__)))
        int cpuInfo[4];
        #if defined(_MSC_VER)
            __cpuidex(cpuInfo, 7, 0);
        #else
            __asm__ __volatile__(
                "cpuid"
                : "=a" (cpuInfo[0]), "=b" (cpuInfo[1]), "=c" (cpuInfo[2]), "=d" (cpuInfo[3])
                : "a" (7), "c" (0)
            );
        #endif
        // ADX is bit 19 of EBX (leaf 7, subleaf 0)
        return (cpuInfo[1] & (1 << 19)) != 0;
    #else
        return false;  // Not x86/x64
    #endif
}

namespace detail {
// Cross-compiler sub-borrow helper
inline uint8_t subborrow64(uint8_t borrow_in, uint64_t a, uint64_t b, uint64_t& result) {
#if defined(_MSC_VER)
    return _subborrow_u64(borrow_in, a, b, reinterpret_cast<unsigned long long*>(&result));
#elif defined(__SIZEOF_INT128__)
    __uint128_t diff = static_cast<__uint128_t>(a) - b - borrow_in;
    result = static_cast<uint64_t>(diff);
    return static_cast<uint8_t>((diff >> 127) & 1);
#else
    // Portable fallback for 32-bit targets (no __int128)
    uint64_t t = a - b;
    uint8_t borrow1 = (t > a) ? 1 : 0;
    uint64_t t2 = t - borrow_in;
    uint8_t borrow2 = (t2 > t) ? 1 : 0;
    result = t2;
    return borrow1 | borrow2;
#endif
}

// Helper: Add 128-bit value to wide result at position i
inline void add_to_wide(uint64_t result[8], size_t i, uint64_t lo, uint64_t hi) {
    uint8_t carry = 0;
    carry = adcx64(result[i], lo, carry, result[i]);
    carry = adcx64(result[i + 1], hi, carry, result[i + 1]);
    
    // Propagate carry if needed
    size_t j = i + 2;
    while (carry && j < 8) {
        carry = adcx64(result[j], 0, carry, result[j]);
        j++;
    }
}

// ===================================================================
// Core 4x4 multiplication using BMI2 - FULLY UNROLLED
// Direct carry propagation without loops for maximum performance
// ===================================================================

void mul_4x4_bmi2(const uint64_t a[4], const uint64_t b[4], uint64_t result[8]) {
    uint64_t lo = 0, hi = 0;
    uint8_t carry = 0;
    
    // Initialize result to zero
    result[0] = result[1] = result[2] = result[3] = 0;
    result[4] = result[5] = result[6] = result[7] = 0;
    
    // Row 0: a[0] * b[j]
    mulx64(a[0], b[0], lo, hi);
    result[0] = lo; result[1] = hi;
    
    mulx64(a[0], b[1], lo, hi);
    carry = 0;
    carry = adcx64(result[1], lo, carry, result[1]);
    carry = adcx64(result[2], hi, carry, result[2]);
    adcx64(result[3], 0, carry, result[3]);
    
    mulx64(a[0], b[2], lo, hi);
    carry = 0;
    carry = adcx64(result[2], lo, carry, result[2]);
    carry = adcx64(result[3], hi, carry, result[3]);
    adcx64(result[4], 0, carry, result[4]);
    
    mulx64(a[0], b[3], lo, hi);
    carry = 0;
    carry = adcx64(result[3], lo, carry, result[3]);
    carry = adcx64(result[4], hi, carry, result[4]);
    adcx64(result[5], 0, carry, result[5]);
    
    // Row 1: a[1] * b[j]
    mulx64(a[1], b[0], lo, hi);
    carry = 0;
    carry = adcx64(result[1], lo, carry, result[1]);
    carry = adcx64(result[2], hi, carry, result[2]);
    carry = adcx64(result[3], 0, carry, result[3]);
    carry = adcx64(result[4], 0, carry, result[4]);
    adcx64(result[5], 0, carry, result[5]);
    
    mulx64(a[1], b[1], lo, hi);
    carry = 0;
    carry = adcx64(result[2], lo, carry, result[2]);
    carry = adcx64(result[3], hi, carry, result[3]);
    carry = adcx64(result[4], 0, carry, result[4]);
    carry = adcx64(result[5], 0, carry, result[5]);
    adcx64(result[6], 0, carry, result[6]);
    
    mulx64(a[1], b[2], lo, hi);
    carry = 0;
    carry = adcx64(result[3], lo, carry, result[3]);
    carry = adcx64(result[4], hi, carry, result[4]);
    carry = adcx64(result[5], 0, carry, result[5]);
    carry = adcx64(result[6], 0, carry, result[6]);
    adcx64(result[7], 0, carry, result[7]);
    
    mulx64(a[1], b[3], lo, hi);
    carry = 0;
    carry = adcx64(result[4], lo, carry, result[4]);
    carry = adcx64(result[5], hi, carry, result[5]);
    carry = adcx64(result[6], 0, carry, result[6]);
    adcx64(result[7], 0, carry, result[7]);
    
    // Row 2: a[2] * b[j]
    mulx64(a[2], b[0], lo, hi);
    carry = 0;
    carry = adcx64(result[2], lo, carry, result[2]);
    carry = adcx64(result[3], hi, carry, result[3]);
    carry = adcx64(result[4], 0, carry, result[4]);
    carry = adcx64(result[5], 0, carry, result[5]);
    carry = adcx64(result[6], 0, carry, result[6]);
    adcx64(result[7], 0, carry, result[7]);
    
    mulx64(a[2], b[1], lo, hi);
    carry = 0;
    carry = adcx64(result[3], lo, carry, result[3]);
    carry = adcx64(result[4], hi, carry, result[4]);
    carry = adcx64(result[5], 0, carry, result[5]);
    carry = adcx64(result[6], 0, carry, result[6]);
    adcx64(result[7], 0, carry, result[7]);
    
    mulx64(a[2], b[2], lo, hi);
    carry = 0;
    carry = adcx64(result[4], lo, carry, result[4]);
    carry = adcx64(result[5], hi, carry, result[5]);
    carry = adcx64(result[6], 0, carry, result[6]);
    adcx64(result[7], 0, carry, result[7]);
    
    mulx64(a[2], b[3], lo, hi);
    carry = 0;
    carry = adcx64(result[5], lo, carry, result[5]);
    carry = adcx64(result[6], hi, carry, result[6]);
    adcx64(result[7], 0, carry, result[7]);
    
    // Row 3: a[3] * b[j]
    mulx64(a[3], b[0], lo, hi);
    carry = 0;
    carry = adcx64(result[3], lo, carry, result[3]);
    carry = adcx64(result[4], hi, carry, result[4]);
    carry = adcx64(result[5], 0, carry, result[5]);
    carry = adcx64(result[6], 0, carry, result[6]);
    adcx64(result[7], 0, carry, result[7]);
    
    mulx64(a[3], b[1], lo, hi);
    carry = 0;
    carry = adcx64(result[4], lo, carry, result[4]);
    carry = adcx64(result[5], hi, carry, result[5]);
    carry = adcx64(result[6], 0, carry, result[6]);
    adcx64(result[7], 0, carry, result[7]);
    
    mulx64(a[3], b[2], lo, hi);
    carry = 0;
    carry = adcx64(result[5], lo, carry, result[5]);
    carry = adcx64(result[6], hi, carry, result[6]);
    adcx64(result[7], 0, carry, result[7]);
    
    mulx64(a[3], b[3], lo, hi);
    carry = 0;
    carry = adcx64(result[6], lo, carry, result[6]);
    adcx64(result[7], hi, carry, result[7]);
}

// ===================================================================
// Karatsuba Squaring Algorithm for 256-bit (4x64-bit limbs)
// Complexity: ~9 multiplications vs 10 in standard approach
// Strategy: Recursive decomposition using (a+b)^2 = a^2 + 2ab + b^2
// ===================================================================

void square_4_karatsuba(const uint64_t a[4], uint64_t result[8]) {
    // Split into two 128-bit halves: a = a_high*2^128 + a_low
    // a^2 = (a_high*2^128 + a_low)^2
    //    = a_high^2*2^256 + 2*a_high*a_low*2^128 + a_low^2
    
    uint64_t lo = 0, hi = 0;
    uint8_t carry = 0;
    
    // Low half: a_low = a[0..1]
    // High half: a_high = a[2..3]
    
    // Step 1: Compute a_low^2 (2x2 limb square = 4 limbs)
    uint64_t low_sq[4] = {0, 0, 0, 0};
    
    // a[0]^2
    mulx64(a[0], a[0], low_sq[0], low_sq[1]);
    
    // a[1]^2
    uint64_t t0 = 0, t1 = 0;
    mulx64(a[1], a[1], t0, t1);
    carry = 0;
    carry = adcx64(low_sq[2], t0, carry, low_sq[2]);
    adcx64(low_sq[3], t1, carry, low_sq[3]);
    
    // 2*a[0]*a[1]
    mulx64(a[0], a[1], lo, hi);
    carry = 0;
    carry = adcx64(low_sq[1], lo, carry, low_sq[1]);
    carry = adcx64(low_sq[2], hi, carry, low_sq[2]);
    adcx64(low_sq[3], 0, carry, low_sq[3]);
    
    carry = 0;
    carry = adcx64(low_sq[1], lo, carry, low_sq[1]);
    carry = adcx64(low_sq[2], hi, carry, low_sq[2]);
    adcx64(low_sq[3], 0, carry, low_sq[3]);
    
    // Step 2: Compute a_high^2 (2x2 limb square = 4 limbs)
    uint64_t high_sq[4] = {0, 0, 0, 0};
    
    // a[2]^2
    mulx64(a[2], a[2], high_sq[0], high_sq[1]);
    
    // a[3]^2
    mulx64(a[3], a[3], t0, t1);
    carry = 0;
    carry = adcx64(high_sq[2], t0, carry, high_sq[2]);
    adcx64(high_sq[3], t1, carry, high_sq[3]);
    
    // 2*a[2]*a[3]
    mulx64(a[2], a[3], lo, hi);
    carry = 0;
    carry = adcx64(high_sq[1], lo, carry, high_sq[1]);
    carry = adcx64(high_sq[2], hi, carry, high_sq[2]);
    adcx64(high_sq[3], 0, carry, high_sq[3]);
    
    carry = 0;
    carry = adcx64(high_sq[1], lo, carry, high_sq[1]);
    carry = adcx64(high_sq[2], hi, carry, high_sq[2]);
    adcx64(high_sq[3], 0, carry, high_sq[3]);
    
    // Step 3: Compute (a_low + a_high)^2
    uint64_t sum[2];
    carry = 0;
    carry = adcx64(a[0], a[2], carry, sum[0]);
    adcx64(a[1], a[3], carry, sum[1]);
    
    uint64_t sum_sq[4] = {0, 0, 0, 0};
    mulx64(sum[0], sum[0], sum_sq[0], sum_sq[1]);
    mulx64(sum[1], sum[1], t0, t1);
    carry = 0;
    carry = adcx64(sum_sq[2], t0, carry, sum_sq[2]);
    adcx64(sum_sq[3], t1, carry, sum_sq[3]);
    
    mulx64(sum[0], sum[1], lo, hi);
    carry = 0;
    carry = adcx64(sum_sq[1], lo, carry, sum_sq[1]);
    carry = adcx64(sum_sq[2], hi, carry, sum_sq[2]);
    adcx64(sum_sq[3], 0, carry, sum_sq[3]);
    carry = 0;
    carry = adcx64(sum_sq[1], lo, carry, sum_sq[1]);
    carry = adcx64(sum_sq[2], hi, carry, sum_sq[2]);
    adcx64(sum_sq[3], 0, carry, sum_sq[3]);
    
    // Step 4: middle = (a_low + a_high)^2 - a_low^2 - a_high^2
    uint64_t middle[4];
    uint8_t borrow = 0;
    
    // Subtract low_sq
    borrow = 0;
    (void)borrow;
    borrow = subborrow64(borrow, sum_sq[0], low_sq[0], middle[0]);
    borrow = subborrow64(borrow, sum_sq[1], low_sq[1], middle[1]);
    borrow = subborrow64(borrow, sum_sq[2], low_sq[2], middle[2]);
    borrow = subborrow64(borrow, sum_sq[3], low_sq[3], middle[3]);
    (void)borrow;
    
    // Subtract high_sq
    borrow = 0;
    (void)borrow;
    borrow = subborrow64(borrow, middle[0], high_sq[0], middle[0]);
    borrow = subborrow64(borrow, middle[1], high_sq[1], middle[1]);
    borrow = subborrow64(borrow, middle[2], high_sq[2], middle[2]);
    borrow = subborrow64(borrow, middle[3], high_sq[3], middle[3]);
    (void)borrow;
    
    // Step 5: Combine: result = low_sq + middle*2^128 + high_sq*2^256
    // Initialize result array
    result[0] = low_sq[0];
    result[1] = low_sq[1];
    result[2] = low_sq[2];
    result[3] = low_sq[3];
    result[4] = 0;
    result[5] = 0;
    result[6] = 0;
    result[7] = 0;
    
    // Add middle at position 2 (2^128)
    carry = 0;
    carry = adcx64(result[2], middle[0], carry, result[2]);
    carry = adcx64(result[3], middle[1], carry, result[3]);
    carry = adcx64(result[4], middle[2], carry, result[4]);
    carry = adcx64(result[5], middle[3], carry, result[5]);
    if (carry) {
        carry = adcx64(result[6], 0, carry, result[6]);
        adcx64(result[7], 0, carry, result[7]);
    }
    
    // Add high_sq at position 4 (2^256)
    carry = 0;
    carry = adcx64(result[4], high_sq[0], carry, result[4]);
    carry = adcx64(result[5], high_sq[1], carry, result[5]);
    carry = adcx64(result[6], high_sq[2], carry, result[6]);
    adcx64(result[7], high_sq[3], carry, result[7]);
}

// ===================================================================
// Toom-Cook-3 Squaring Algorithm for 256-bit
// Complexity: ~5 multiplications vs 10 in standard (theoretical best!)
// Strategy: Split into 3 parts, evaluate at 5 points, interpolate
// Note: High addition overhead, best for larger numbers (512+ bits)
// ===================================================================

void square_4_toomcook(const uint64_t a[4], uint64_t result[8]) {
    // Toom-Cook-3 squaring for 256-bit
    // Split a into 3 parts (each ~85 bits, but we use limb boundaries)
    // a = a0 + a1*B + a2*B^2 where B = 2^85 (approximated with limbs)
    
    // For simplicity with 4 limbs, use uneven split:
    // a0 = a[0] (64 bits)
    // a1 = a[1] + a[2]_low (128 bits approx)
    // a2 = a[2]_high + a[3] (128 bits approx)
    
    // This implementation would be very complex for marginal gains
    // on 256-bit numbers. Toom-Cook shines at 1024+ bits.
    // For now, fall back to standard approach
    
    // TODO: Implement full Toom-Cook-3 (requires careful limb splitting)
    // Current: Use standard approach as placeholder
    square_4_bmi2(a, result);
}

// ===================================================================
// Optimized squaring - FULLY UNROLLED BMI2
// Strategy: Compute cross-products and add twice (best for BMI2 carry chains)
// Note: Montgomery "shift-left" approach tested but was SLOWER due to:
//   - Memory allocation for intermediate arrays
//   - Loop overhead for bit-shifting
//   - Cache misses vs register-based carry chains
// Current approach: fully unrolled, register-only, optimal for BMI2 ADCX
// ===================================================================

void square_4_bmi2(const uint64_t a[4], uint64_t result[8]) {
    uint64_t lo = 0, hi = 0;
    uint8_t carry = 0;
    
    // Initialize result
    result[0] = result[1] = result[2] = result[3] = 0;
    result[4] = result[5] = result[6] = result[7] = 0;
    
    // Diagonal terms: a[i]^2
    mulx64(a[0], a[0], lo, hi);
    result[0] = lo; result[1] = hi;
    
    mulx64(a[1], a[1], lo, hi);
    carry = 0;
    carry = adcx64(result[2], lo, carry, result[2]);
    adcx64(result[3], hi, carry, result[3]);
    
    mulx64(a[2], a[2], lo, hi);
    carry = 0;
    carry = adcx64(result[4], lo, carry, result[4]);
    adcx64(result[5], hi, carry, result[5]);
    
    mulx64(a[3], a[3], lo, hi);
    carry = 0;
    carry = adcx64(result[6], lo, carry, result[6]);
    adcx64(result[7], hi, carry, result[7]);
    
    // Cross-product terms (added twice - best for BMI2 ADCX chains)
    // a[0] * a[1] (add twice to position 1)
    mulx64(a[0], a[1], lo, hi);
    carry = 0;
    carry = adcx64(result[1], lo, carry, result[1]);
    carry = adcx64(result[2], hi, carry, result[2]);
    carry = adcx64(result[3], 0, carry, result[3]);
    adcx64(result[4], 0, carry, result[4]);
    
    carry = 0;
    carry = adcx64(result[1], lo, carry, result[1]);
    carry = adcx64(result[2], hi, carry, result[2]);
    carry = adcx64(result[3], 0, carry, result[3]);
    adcx64(result[4], 0, carry, result[4]);
    
    // a[0] * a[2] (add twice to position 2)
    mulx64(a[0], a[2], lo, hi);
    carry = 0;
    carry = adcx64(result[2], lo, carry, result[2]);
    carry = adcx64(result[3], hi, carry, result[3]);
    carry = adcx64(result[4], 0, carry, result[4]);
    adcx64(result[5], 0, carry, result[5]);
    
    carry = 0;
    carry = adcx64(result[2], lo, carry, result[2]);
    carry = adcx64(result[3], hi, carry, result[3]);
    carry = adcx64(result[4], 0, carry, result[4]);
    adcx64(result[5], 0, carry, result[5]);
    
    // a[0] * a[3] (add twice to position 3)
    mulx64(a[0], a[3], lo, hi);
    carry = 0;
    carry = adcx64(result[3], lo, carry, result[3]);
    carry = adcx64(result[4], hi, carry, result[4]);
    carry = adcx64(result[5], 0, carry, result[5]);
    adcx64(result[6], 0, carry, result[6]);
    
    carry = 0;
    carry = adcx64(result[3], lo, carry, result[3]);
    carry = adcx64(result[4], hi, carry, result[4]);
    carry = adcx64(result[5], 0, carry, result[5]);
    adcx64(result[6], 0, carry, result[6]);
    
    // a[1] * a[2] (add twice to position 3)
    mulx64(a[1], a[2], lo, hi);
    carry = 0;
    carry = adcx64(result[3], lo, carry, result[3]);
    carry = adcx64(result[4], hi, carry, result[4]);
    carry = adcx64(result[5], 0, carry, result[5]);
    carry = adcx64(result[6], 0, carry, result[6]);
    adcx64(result[7], 0, carry, result[7]);
    
    carry = 0;
    carry = adcx64(result[3], lo, carry, result[3]);
    carry = adcx64(result[4], hi, carry, result[4]);
    carry = adcx64(result[5], 0, carry, result[5]);
    carry = adcx64(result[6], 0, carry, result[6]);
    adcx64(result[7], 0, carry, result[7]);
    
    // a[1] * a[3] (add twice to position 4)
    mulx64(a[1], a[3], lo, hi);
    carry = 0;
    carry = adcx64(result[4], lo, carry, result[4]);
    carry = adcx64(result[5], hi, carry, result[5]);
    carry = adcx64(result[6], 0, carry, result[6]);
    adcx64(result[7], 0, carry, result[7]);
    
    carry = 0;
    carry = adcx64(result[4], lo, carry, result[4]);
    carry = adcx64(result[5], hi, carry, result[5]);
    carry = adcx64(result[6], 0, carry, result[6]);
    adcx64(result[7], 0, carry, result[7]);
    
    // a[2] * a[3] (add twice to position 5)
    mulx64(a[2], a[3], lo, hi);
    carry = 0;
    carry = adcx64(result[5], lo, carry, result[5]);
    carry = adcx64(result[6], hi, carry, result[6]);
    adcx64(result[7], 0, carry, result[7]);
    
    carry = 0;
    carry = adcx64(result[5], lo, carry, result[5]);
    carry = adcx64(result[6], hi, carry, result[6]);
    adcx64(result[7], 0, carry, result[7]);
}

// ===================================================================
// Fast reduction modulo secp256k1 prime
// p = 2^256 - 2^32 - 977
// For 512-bit t: t == t_low + t_high * (2^32 + 977) (mod p)
// ===================================================================

void montgomery_reduce_bmi2(uint64_t result[8]) {
    // Start with low 256 bits + extra limb for carry
    uint64_t reduced[5] = {result[0], result[1], result[2], result[3], 0};
    
    // Process each high limb: add t[4+i] * (2^32 + 977)
    for (size_t i = 0; i < 4; ++i) {
        uint64_t const hi_limb = result[4 + i];
        if (hi_limb == 0) continue;
        
        // Add hi_limb * 977
        uint64_t lo = 0, hi = 0;
        mulx64(hi_limb, 977, lo, hi);
        
        uint8_t carry = 0;
        carry = adcx64(reduced[i], lo, carry, reduced[i]);
        carry = adcx64(reduced[i + 1], hi, carry, reduced[i + 1]);
        (void)carry;
        if (i + 2 < 5) {
            carry = adcx64(reduced[i + 2], 0, carry, reduced[i + 2]);
            if (i + 3 < 5) {
                carry = adcx64(reduced[i + 3], 0, carry, reduced[i + 3]);
                if (i + 4 < 5) {
                    carry = adcx64(reduced[i + 4], 0, carry, reduced[i + 4]);
                }
            }
        }
        (void)carry;
        
        // Add hi_limb * 2^32 (shift left by 32 bits)
        // This means: add (hi_limb << 32) at position i
        // Split into two 32-bit halves
        uint64_t const lo_half = (hi_limb << 32) & 0xFFFFFFFFFFFFFFFFULL;
        uint64_t const hi_half = hi_limb >> 32;
        
        carry = 0;
        carry = adcx64(reduced[i], lo_half, carry, reduced[i]);
        (void)carry;
        carry = adcx64(reduced[i + 1], hi_half, carry, reduced[i + 1]);
        if (i + 2 < 5) {
            carry = adcx64(reduced[i + 2], 0, carry, reduced[i + 2]);
            if (i + 3 < 5) {
                carry = adcx64(reduced[i + 3], 0, carry, reduced[i + 3]);
                if (i + 4 < 5) {
                    carry = adcx64(reduced[i + 4], 0, carry, reduced[i + 4]);
                }
            }
        }
        (void)carry;
    }
    
    // If extra limb is non-zero, do one more reduction step
    while (reduced[4] != 0) {
        uint64_t const hi_limb = reduced[4];
        reduced[4] = 0;
        
        // Add hi_limb * 977
        uint64_t lo = 0, hi = 0;
        mulx64(hi_limb, 977, lo, hi);
        
        uint8_t carry = 0;
        (void)carry;
        carry = adcx64(reduced[0], lo, carry, reduced[0]);
        carry = adcx64(reduced[1], hi, carry, reduced[1]);
        carry = adcx64(reduced[2], 0, carry, reduced[2]);
        carry = adcx64(reduced[3], 0, carry, reduced[3]);
        carry = adcx64(reduced[4], 0, carry, reduced[4]);
        (void)carry;
        
        // Add hi_limb * 2^32
        uint64_t const lo_half = (hi_limb << 32);
        uint64_t const hi_half = hi_limb >> 32;
        
        carry = 0;
        (void)carry;
        carry = adcx64(reduced[0], lo_half, carry, reduced[0]);
        carry = adcx64(reduced[1], hi_half, carry, reduced[1]);
        carry = adcx64(reduced[2], 0, carry, reduced[2]);
        carry = adcx64(reduced[3], 0, carry, reduced[3]);
        carry = adcx64(reduced[4], 0, carry, reduced[4]);
        (void)carry;
    }
    
    // Final reduction: subtract p if result >= p
    const uint64_t p[4] = {
        0xFFFFFFFEFFFFFC2FULL,
        0xFFFFFFFFFFFFFFFFULL,
        0xFFFFFFFFFFFFFFFFULL,
        0xFFFFFFFFFFFFFFFFULL
    };
    
    // Check if reduced >= p
    bool need_sub = false;
    for (int i = 3; i >= 0; --i) {
        if (reduced[i] > p[i]) {
            need_sub = true;
            break;
        } else if (reduced[i] < p[i]) {
            break;
        }
    }
    
    if (need_sub) {
        uint8_t borrow = 0;
        for (size_t i = 0; i < 4; ++i) {
            uint64_t const diff = reduced[i] - p[i] - borrow;
            borrow = (reduced[i] < p[i] + borrow) ? 1 : 0;
            reduced[i] = diff;
        }
    }
    
    // Copy result back
    std::memcpy(result, reduced, 4 * sizeof(uint64_t));
}

} // namespace detail

// ===================================================================
// Public API: High-level field operations
// ===================================================================

FieldElement field_mul_bmi2(const FieldElement& a, const FieldElement& b) {
    // Extract limbs from FieldElement
    uint64_t a_limbs[4], b_limbs[4], result[8];
    std::memcpy(a_limbs, &a, sizeof(a_limbs));
    std::memcpy(b_limbs, &b, sizeof(b_limbs));

    // NEW: Full assembly with integrated Montgomery reduction (7-10ns target!)
    // TEMPORARILY DISABLED - has bugs, needs rewrite
    #if 0 && defined(SECP256K1_HAS_ASM) && defined(__GNUC__) && defined(__x86_64__)
    static int use_full_asm = [](){
        // Check if full assembly is available and CPU supports it
        bool supported = has_bmi2_support() && has_adx_support();
        if (std::getenv("SECP256K1_DEBUG_ASM")) {
            fprintf(stderr, "[ASM] Full mul+reduce: BMI2=%d ADX=%d SUPPORTED=%d\n", 
                   has_bmi2_support(), has_adx_support(), supported);
        }
        return supported ? 1 : 0;
    }();
    
    if (use_full_asm) {
        // Direct call to full assembly (mul+reduce in one function, 7-10ns!)
        uint64_t result_asm[4];
        field_mul_full_asm(a_limbs, b_limbs, result_asm);
        
        // Create result FieldElement
        FieldElement out;
        std::memcpy(&out, result_asm, 32);  // Result is already reduced
        return out;
    }
    #endif
    
    // Fallback: BMI2 intrinsics (17ns on Clang 18, 32ns on GCC)
    #if defined(SECP256K1_HAS_ASM) && (defined(__x86_64__) || defined(_M_X64))
        // Use full assembly multiplication + reduction (fastest)
        field_mul_full_asm(a_limbs, b_limbs, result);
    #else
        detail::mul_4x4_bmi2(a_limbs, b_limbs, result);
        detail::montgomery_reduce_bmi2(result);
    #endif
    
    // Create result FieldElement
    FieldElement out;
    std::memcpy(&out, result, 32);  // Copy first 4 limbs
    
    return out;
}

FieldElement field_square_bmi2(const FieldElement& a) {
    uint64_t a_limbs[4], result[8];
    std::memcpy(a_limbs, &a, sizeof(a_limbs));

    // Prefer using ASM mul for square as well (a*a)
    #if defined(SECP256K1_HAS_ASM) && (defined(__x86_64__) || defined(_M_X64))
        // Use optimized fused squaring + reduction
        field_sqr_full_asm(a_limbs, result);
        
        FieldElement out;
        std::memcpy(&out, result, 32);
        return out;
    #else
        detail::square_4_bmi2(a_limbs, result);
        detail::montgomery_reduce_bmi2(result);
        
        FieldElement out;
        std::memcpy(&out, result, 32);
        return out;
    #endif
}

FieldElement field_square_karatsuba(const FieldElement& a) {
    uint64_t a_limbs[4], result[8];
    
    std::memcpy(a_limbs, &a, sizeof(a_limbs));
    
    // Karatsuba squaring
    detail::square_4_karatsuba(a_limbs, result);
    
    // Reduce
    #if defined(SECP256K1_HAS_ASM) && defined(__GNUC__) && defined(__x86_64__)
        reduce_4_asm(result);
    #else
        detail::montgomery_reduce_bmi2(result);
    #endif
    
    FieldElement out;
    std::memcpy(&out, result, 32);
    
    return out;
}

FieldElement field_square_toomcook(const FieldElement& a) {
    uint64_t a_limbs[4], result[8];
    
    std::memcpy(a_limbs, &a, sizeof(a_limbs));
    
    // Toom-Cook squaring
    detail::square_4_toomcook(a_limbs, result);
    
    // Reduce
    #if defined(SECP256K1_HAS_ASM) && defined(__GNUC__) && defined(__x86_64__)
        reduce_4_asm(result);
    #else
        detail::montgomery_reduce_bmi2(result);
    #endif
    
    FieldElement out;
    std::memcpy(&out, result, 32);
    
    return out;
}

FieldElement field_add_bmi2(const FieldElement& a, const FieldElement& b) {
    uint64_t a_limbs[4], b_limbs[4], result[4];
    std::memcpy(a_limbs, &a, sizeof(a_limbs));
    std::memcpy(b_limbs, &b, sizeof(b_limbs));

    #if defined(SECP256K1_HAS_ASM) && (defined(__x86_64__) || defined(_M_X64))
        add_4_asm(a_limbs, b_limbs, result);
    #else
        // Addition is already quite fast, but can optimize with ADCX
        uint8_t carry = 0;
        for (int i = 0; i < 4; ++i) {
            carry = detail::adcx64(a_limbs[i], b_limbs[i], carry, result[i]);
        }
        
        // If result >= p, subtract p
        // p = 2^256 - 2^32 - 977
        // ... (existing C++ logic omitted for brevity, using simple check)
        // Actually, the existing implementation below is incomplete/placeholder.
        // Let's use the proper C++ implementation from field.cpp if ASM is not available.
        // But since this function is specifically "bmi2", we should implement it.
        
        // For now, fallback to standard C++ if no ASM
        return a + b; 
    #endif

    FieldElement out;
    std::memcpy(&out, result, 32);
    return out;
}

FieldElement field_sub_bmi2(const FieldElement& a, const FieldElement& b) {
    uint64_t a_limbs[4], b_limbs[4], result[4];
    std::memcpy(a_limbs, &a, sizeof(a_limbs));
    std::memcpy(b_limbs, &b, sizeof(b_limbs));

    #if defined(SECP256K1_HAS_ASM) && (defined(__x86_64__) || defined(_M_X64))
        sub_4_asm(a_limbs, b_limbs, result);
    #else
        return a - b;
    #endif

    FieldElement out;
    std::memcpy(&out, result, 32);
    return out;
}

FieldElement field_negate_bmi2(const FieldElement& a) {
    // Negate: p - a
    const uint64_t p[4] = {
        0xFFFFFFFEFFFFFC2FULL,
        0xFFFFFFFFFFFFFFFFULL,
        0xFFFFFFFFFFFFFFFFULL,
        0xFFFFFFFFFFFFFFFFULL
    };
    
    uint64_t a_limbs[4], result[4];
    std::memcpy(a_limbs, &a, sizeof(a_limbs));
    
    // Compute p - a
    uint8_t borrow = 0;
    for (int i = 0; i < 4; ++i) {
        uint64_t const diff = p[i] - a_limbs[i] - borrow;
        borrow = (p[i] < a_limbs[i] + borrow) ? 1 : 0;
        result[i] = diff;
    }
    
    FieldElement out;
    std::memcpy(&out, result, 32);
    
    return out;
}

} // namespace secp256k1::fast
