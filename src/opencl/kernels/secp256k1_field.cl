// =============================================================================
// UltrafastSecp256k1 OpenCL Kernels - Field Arithmetic
// =============================================================================
// secp256k1 field: F_p where p = 2^256 - 2^32 - 977
// Little-endian 256-bit integers using 4x64-bit limbs
// =============================================================================

// Field prime p = 2^256 - 0x1000003D1
// In 64-bit limbs (little-endian):
// p = {0xFFFFFFFEFFFFFC2F, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF}

// Constants
#define SECP256K1_P0 0xFFFFFFFEFFFFFC2FUL
#define SECP256K1_P1 0xFFFFFFFFFFFFFFFFUL
#define SECP256K1_P2 0xFFFFFFFFFFFFFFFFUL
#define SECP256K1_P3 0xFFFFFFFFFFFFFFFFUL

// K = 2^32 + 977 = 0x1000003D1 (for fast reduction)
#define SECP256K1_K 0x1000003D1UL

// =============================================================================
// Forced Inlining
// =============================================================================
// OpenCL C follows C99 inline semantics: a plain `inline` definition does NOT
// provide an external definition. When the compiler (e.g. AMD ROCm) declines
// to inline a larger helper (field_inv_impl, field_sqr_impl etc.), the call
// has no symbol to link against → "undefined hidden symbol".
//
// NVIDIA (nvoc) is unaffected because we use __attribute__((always_inline)).
// Solution: `static inline` for non-NVIDIA (TU-local definition always present).
// We also make the NVIDIA path `static inline` for consistency (no downside).
#ifdef __NV_CL_C_VERSION
  #define FORCE_INLINE __attribute__((always_inline)) static inline
  #define FORCE_INLINE_STATIC __attribute__((always_inline)) static inline
#else
  #define FORCE_INLINE static inline
  #define FORCE_INLINE_STATIC static inline
#endif

// =============================================================================
// 64-bit Multiplication Helpers
// =============================================================================

// Multiply two 64-bit numbers, get 128-bit result as (hi, lo)
FORCE_INLINE ulong2 mul64_full(ulong a, ulong b) {
    // Use OpenCL's mul_hi for high part
    ulong lo = a * b;
    ulong hi = mul_hi(a, b);
    return (ulong2)(lo, hi);
}

// Add with carry: result = a + b + carry_in, returns new carry
FORCE_INLINE ulong add_with_carry(ulong a, ulong b, ulong carry_in, ulong* carry_out) {
    ulong sum = a + b;
    ulong c1 = (sum < a) ? 1UL : 0UL;
    sum += carry_in;
    ulong c2 = (sum < carry_in) ? 1UL : 0UL;
    *carry_out = c1 + c2;
    return sum;
}

// Subtract with borrow: result = a - b - borrow_in, returns new borrow
FORCE_INLINE ulong sub_with_borrow(ulong a, ulong b, ulong borrow_in, ulong* borrow_out) {
    ulong diff = a - b;
    ulong b1 = (a < b) ? 1UL : 0UL;
    ulong temp = diff;
    diff -= borrow_in;
    ulong b2 = (temp < borrow_in) ? 1UL : 0UL;
    *borrow_out = b1 + b2;
    return diff;
}

// =============================================================================
// Field Element Type (256-bit)
// =============================================================================

typedef struct {
    ulong limbs[4];  // Little-endian: limbs[0] is LSB
} FieldElement;

// =============================================================================
// NVIDIA OpenCL PTX Acceleration (Level 1+2+3)
// =============================================================================
// On consumer NVIDIA GPUs (Turing/Ampere/Ada/Blackwell), INT32 multiply
// throughput is 32x higher than INT64. Inline PTX enables:
//   Level 1+2: mad.lo.cc.u64/madc.hi.cc.u64 carry chains (no comparison-carry)
//   Level 3:   mad.lo.cc.u32/madc.hi.cc.u32 32-bit Comba (INT32 throughput)
// Fallback (AMD, Intel, portable): mul_hi + comparison-based carry unchanged.
// Guard: __NV_CL_C_VERSION is defined only by NVIDIA's OpenCL compiler.
// =============================================================================

#ifdef __NV_CL_C_VERSION

// 32-bit MAD accumulate: (r0:r1:r2) += a * b  [3-register 96-bit accumulator]
#define OCL_MAD32(r0, r1, r2, a, b) \
    __asm volatile( \
        "mad.lo.cc.u32 %0, %3, %4, %0; \n\t" \
        "madc.hi.cc.u32 %1, %3, %4, %1; \n\t" \
        "addc.u32 %2, %2, 0; \n\t" \
        : "+r"(r0), "+r"(r1), "+r"(r2) \
        : "r"(a), "r"(b) \
    )

// 32-bit squaring diagonal: (r0:r1:r2) += a*a
#define OCL_SQR32_D(r0, r1, r2, a) \
    __asm volatile( \
        "mad.lo.cc.u32 %0, %3, %3, %0; \n\t" \
        "madc.hi.cc.u32 %1, %3, %3, %1; \n\t" \
        "addc.u32 %2, %2, 0; \n\t" \
        : "+r"(r0), "+r"(r1), "+r"(r2) \
        : "r"(a) \
    )

// 32-bit squaring off-diagonal: (r0:r1:r2) += 2 * a*b
#define OCL_SQR32_M2(r0, r1, r2, a, b) \
    do { \
        uint _lo, _hi; \
        __asm volatile( \
            "mul.lo.u32 %0, %2, %3; \n\t" \
            "mul.hi.u32 %1, %2, %3; \n\t" \
            : "=r"(_lo), "=r"(_hi) : "r"(a), "r"(b) \
        ); \
        __asm volatile( \
            "add.cc.u32 %0, %0, %3; \n\t" \
            "addc.cc.u32 %1, %1, %4; \n\t" \
            "addc.u32 %2, %2, 0; \n\t" \
            "add.cc.u32 %0, %0, %3; \n\t" \
            "addc.cc.u32 %1, %1, %4; \n\t" \
            "addc.u32 %2, %2, 0; \n\t" \
            : "+r"(r0), "+r"(r1), "+r"(r2) : "r"(_lo), "r"(_hi) \
        ); \
    } while(0)

// ----------------------------------------------------------------------------
// 32-bit Comba multiplication: 4x64 FieldElement reinterpreted as 8x32 limbs.
// Produces uint[16] raw output (little-endian 32-bit limbs of 512-bit product).
// Mirrors CUDA's mul_256_comba32 from secp256k1_32_hybrid_final.cuh.
// ----------------------------------------------------------------------------
FORCE_INLINE_STATIC void mul_256_comba32_ocl(
    const FieldElement* a, const FieldElement* b, uint t32[16]
) {
    uint a32[8], b32[8];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        a32[2*i]   = (uint)(a->limbs[i]);
        a32[2*i+1] = (uint)(a->limbs[i] >> 32);
        b32[2*i]   = (uint)(b->limbs[i]);
        b32[2*i+1] = (uint)(b->limbs[i] >> 32);
    }
    uint r0 = 0, r1 = 0, r2 = 0;

    OCL_MAD32(r0,r1,r2, a32[0],b32[0]);
    t32[0]=r0; r0=r1; r1=r2; r2=0;

    OCL_MAD32(r0,r1,r2, a32[0],b32[1]); OCL_MAD32(r0,r1,r2, a32[1],b32[0]);
    t32[1]=r0; r0=r1; r1=r2; r2=0;

    OCL_MAD32(r0,r1,r2, a32[0],b32[2]); OCL_MAD32(r0,r1,r2, a32[1],b32[1]); OCL_MAD32(r0,r1,r2, a32[2],b32[0]);
    t32[2]=r0; r0=r1; r1=r2; r2=0;

    OCL_MAD32(r0,r1,r2, a32[0],b32[3]); OCL_MAD32(r0,r1,r2, a32[1],b32[2]); OCL_MAD32(r0,r1,r2, a32[2],b32[1]); OCL_MAD32(r0,r1,r2, a32[3],b32[0]);
    t32[3]=r0; r0=r1; r1=r2; r2=0;

    OCL_MAD32(r0,r1,r2, a32[0],b32[4]); OCL_MAD32(r0,r1,r2, a32[1],b32[3]); OCL_MAD32(r0,r1,r2, a32[2],b32[2]); OCL_MAD32(r0,r1,r2, a32[3],b32[1]); OCL_MAD32(r0,r1,r2, a32[4],b32[0]);
    t32[4]=r0; r0=r1; r1=r2; r2=0;

    OCL_MAD32(r0,r1,r2, a32[0],b32[5]); OCL_MAD32(r0,r1,r2, a32[1],b32[4]); OCL_MAD32(r0,r1,r2, a32[2],b32[3]); OCL_MAD32(r0,r1,r2, a32[3],b32[2]); OCL_MAD32(r0,r1,r2, a32[4],b32[1]); OCL_MAD32(r0,r1,r2, a32[5],b32[0]);
    t32[5]=r0; r0=r1; r1=r2; r2=0;

    OCL_MAD32(r0,r1,r2, a32[0],b32[6]); OCL_MAD32(r0,r1,r2, a32[1],b32[5]); OCL_MAD32(r0,r1,r2, a32[2],b32[4]); OCL_MAD32(r0,r1,r2, a32[3],b32[3]); OCL_MAD32(r0,r1,r2, a32[4],b32[2]); OCL_MAD32(r0,r1,r2, a32[5],b32[1]); OCL_MAD32(r0,r1,r2, a32[6],b32[0]);
    t32[6]=r0; r0=r1; r1=r2; r2=0;

    OCL_MAD32(r0,r1,r2, a32[0],b32[7]); OCL_MAD32(r0,r1,r2, a32[1],b32[6]); OCL_MAD32(r0,r1,r2, a32[2],b32[5]); OCL_MAD32(r0,r1,r2, a32[3],b32[4]); OCL_MAD32(r0,r1,r2, a32[4],b32[3]); OCL_MAD32(r0,r1,r2, a32[5],b32[2]); OCL_MAD32(r0,r1,r2, a32[6],b32[1]); OCL_MAD32(r0,r1,r2, a32[7],b32[0]);
    t32[7]=r0; r0=r1; r1=r2; r2=0;

    OCL_MAD32(r0,r1,r2, a32[1],b32[7]); OCL_MAD32(r0,r1,r2, a32[2],b32[6]); OCL_MAD32(r0,r1,r2, a32[3],b32[5]); OCL_MAD32(r0,r1,r2, a32[4],b32[4]); OCL_MAD32(r0,r1,r2, a32[5],b32[3]); OCL_MAD32(r0,r1,r2, a32[6],b32[2]); OCL_MAD32(r0,r1,r2, a32[7],b32[1]);
    t32[8]=r0; r0=r1; r1=r2; r2=0;

    OCL_MAD32(r0,r1,r2, a32[2],b32[7]); OCL_MAD32(r0,r1,r2, a32[3],b32[6]); OCL_MAD32(r0,r1,r2, a32[4],b32[5]); OCL_MAD32(r0,r1,r2, a32[5],b32[4]); OCL_MAD32(r0,r1,r2, a32[6],b32[3]); OCL_MAD32(r0,r1,r2, a32[7],b32[2]);
    t32[9]=r0; r0=r1; r1=r2; r2=0;

    OCL_MAD32(r0,r1,r2, a32[3],b32[7]); OCL_MAD32(r0,r1,r2, a32[4],b32[6]); OCL_MAD32(r0,r1,r2, a32[5],b32[5]); OCL_MAD32(r0,r1,r2, a32[6],b32[4]); OCL_MAD32(r0,r1,r2, a32[7],b32[3]);
    t32[10]=r0; r0=r1; r1=r2; r2=0;

    OCL_MAD32(r0,r1,r2, a32[4],b32[7]); OCL_MAD32(r0,r1,r2, a32[5],b32[6]); OCL_MAD32(r0,r1,r2, a32[6],b32[5]); OCL_MAD32(r0,r1,r2, a32[7],b32[4]);
    t32[11]=r0; r0=r1; r1=r2; r2=0;

    OCL_MAD32(r0,r1,r2, a32[5],b32[7]); OCL_MAD32(r0,r1,r2, a32[6],b32[6]); OCL_MAD32(r0,r1,r2, a32[7],b32[5]);
    t32[12]=r0; r0=r1; r1=r2; r2=0;

    OCL_MAD32(r0,r1,r2, a32[6],b32[7]); OCL_MAD32(r0,r1,r2, a32[7],b32[6]);
    t32[13]=r0; r0=r1; r1=r2; r2=0;

    OCL_MAD32(r0,r1,r2, a32[7],b32[7]);
    t32[14]=r0; t32[15]=r1;
}

// 32-bit Comba squaring: ~40% fewer multiplications (symmetry exploitation).
// Mirrors CUDA's sqr_256_comba32 from secp256k1_32_hybrid_final.cuh.
FORCE_INLINE_STATIC void sqr_256_comba32_ocl(const FieldElement* a, uint t32[16]) {
    uint a32[8];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        a32[2*i]   = (uint)(a->limbs[i]);
        a32[2*i+1] = (uint)(a->limbs[i] >> 32);
    }
    uint r0 = 0, r1 = 0, r2 = 0;

    OCL_SQR32_D(r0,r1,r2, a32[0]);
    t32[0]=r0; r0=r1; r1=r2; r2=0;

    OCL_SQR32_M2(r0,r1,r2, a32[0],a32[1]);
    t32[1]=r0; r0=r1; r1=r2; r2=0;

    OCL_SQR32_M2(r0,r1,r2, a32[0],a32[2]); OCL_SQR32_D(r0,r1,r2, a32[1]);
    t32[2]=r0; r0=r1; r1=r2; r2=0;

    OCL_SQR32_M2(r0,r1,r2, a32[0],a32[3]); OCL_SQR32_M2(r0,r1,r2, a32[1],a32[2]);
    t32[3]=r0; r0=r1; r1=r2; r2=0;

    OCL_SQR32_M2(r0,r1,r2, a32[0],a32[4]); OCL_SQR32_M2(r0,r1,r2, a32[1],a32[3]); OCL_SQR32_D(r0,r1,r2, a32[2]);
    t32[4]=r0; r0=r1; r1=r2; r2=0;

    OCL_SQR32_M2(r0,r1,r2, a32[0],a32[5]); OCL_SQR32_M2(r0,r1,r2, a32[1],a32[4]); OCL_SQR32_M2(r0,r1,r2, a32[2],a32[3]);
    t32[5]=r0; r0=r1; r1=r2; r2=0;

    OCL_SQR32_M2(r0,r1,r2, a32[0],a32[6]); OCL_SQR32_M2(r0,r1,r2, a32[1],a32[5]); OCL_SQR32_M2(r0,r1,r2, a32[2],a32[4]); OCL_SQR32_D(r0,r1,r2, a32[3]);
    t32[6]=r0; r0=r1; r1=r2; r2=0;

    OCL_SQR32_M2(r0,r1,r2, a32[0],a32[7]); OCL_SQR32_M2(r0,r1,r2, a32[1],a32[6]); OCL_SQR32_M2(r0,r1,r2, a32[2],a32[5]); OCL_SQR32_M2(r0,r1,r2, a32[3],a32[4]);
    t32[7]=r0; r0=r1; r1=r2; r2=0;

    OCL_SQR32_M2(r0,r1,r2, a32[1],a32[7]); OCL_SQR32_M2(r0,r1,r2, a32[2],a32[6]); OCL_SQR32_M2(r0,r1,r2, a32[3],a32[5]); OCL_SQR32_D(r0,r1,r2, a32[4]);
    t32[8]=r0; r0=r1; r1=r2; r2=0;

    OCL_SQR32_M2(r0,r1,r2, a32[2],a32[7]); OCL_SQR32_M2(r0,r1,r2, a32[3],a32[6]); OCL_SQR32_M2(r0,r1,r2, a32[4],a32[5]);
    t32[9]=r0; r0=r1; r1=r2; r2=0;

    OCL_SQR32_M2(r0,r1,r2, a32[3],a32[7]); OCL_SQR32_M2(r0,r1,r2, a32[4],a32[6]); OCL_SQR32_D(r0,r1,r2, a32[5]);
    t32[10]=r0; r0=r1; r1=r2; r2=0;

    OCL_SQR32_M2(r0,r1,r2, a32[4],a32[7]); OCL_SQR32_M2(r0,r1,r2, a32[5],a32[6]);
    t32[11]=r0; r0=r1; r1=r2; r2=0;

    OCL_SQR32_M2(r0,r1,r2, a32[5],a32[7]); OCL_SQR32_D(r0,r1,r2, a32[6]);
    t32[12]=r0; r0=r1; r1=r2; r2=0;

    OCL_SQR32_M2(r0,r1,r2, a32[6],a32[7]);
    t32[13]=r0; r0=r1; r1=r2; r2=0;

    OCL_SQR32_D(r0,r1,r2, a32[7]);
    t32[14]=r0; t32[15]=r1;
}

// 32-bit reduction: T_hi x K_MOD (32-bit MAD chain) + conditional P-subtract.
// Phase 1: T_hi[8..15] x 977 (scalar, 32-bit MAD chain)
// Phase 1b: add T_hi << 32  (K_MOD = 2^32 + 977)
// Phase 2: T_lo[0..7] += result (32-bit carry chain)
// Phase 3+4: pack to 64-bit, fold overflow, conditional P-subtract (64-bit PTX)
// Mirrors CUDA's reduce_512_to_256_32 from secp256k1_32_hybrid_final.cuh.
FORCE_INLINE_STATIC void reduce_512_to_256_32_ocl(uint t32[16], FieldElement* r) {
    uint t0=t32[0], t1=t32[1], t2=t32[2], t3=t32[3];
    uint t4=t32[4], t5=t32[5], t6=t32[6], t7=t32[7];
    const uint t8 =t32[8],  t9 =t32[9],  t10=t32[10], t11=t32[11];
    const uint t12=t32[12], t13=t32[13], t14=t32[14], t15=t32[15];

    // Phase 1: A = T_hi[8..15] x 977 (32-bit scalar MAD chain -> 9 limbs)
    uint a0, a1, a2, a3, a4, a5, a6, a7, a8;
    __asm volatile(
        "mul.lo.u32 %0, %9,  977;\n\t"
        "mul.hi.u32 %1, %9,  977;\n\t"
        "mad.lo.cc.u32 %1, %10, 977, %1;\n\t"
        "madc.hi.u32 %2, %10, 977, 0;\n\t"
        "mad.lo.cc.u32 %2, %11, 977, %2;\n\t"
        "madc.hi.u32 %3, %11, 977, 0;\n\t"
        "mad.lo.cc.u32 %3, %12, 977, %3;\n\t"
        "madc.hi.u32 %4, %12, 977, 0;\n\t"
        "mad.lo.cc.u32 %4, %13, 977, %4;\n\t"
        "madc.hi.u32 %5, %13, 977, 0;\n\t"
        "mad.lo.cc.u32 %5, %14, 977, %5;\n\t"
        "madc.hi.u32 %6, %14, 977, 0;\n\t"
        "mad.lo.cc.u32 %6, %15, 977, %6;\n\t"
        "madc.hi.u32 %7, %15, 977, 0;\n\t"
        "mad.lo.cc.u32 %7, %16, 977, %7;\n\t"
        "madc.hi.u32 %8, %16, 977, 0;\n\t"
        : "=r"(a0),"=r"(a1),"=r"(a2),"=r"(a3),"=r"(a4),
          "=r"(a5),"=r"(a6),"=r"(a7),"=r"(a8)
        : "r"(t8),"r"(t9),"r"(t10),"r"(t11),
          "r"(t12),"r"(t13),"r"(t14),"r"(t15)
    );

    // Phase 1b: add T_hi << 32 (a[1..8] += T_hi[8..15], yielding a9 overflow)
    uint a9;
    __asm volatile(
        "add.cc.u32  %0, %0, %9;\n\t"
        "addc.cc.u32 %1, %1, %10;\n\t"
        "addc.cc.u32 %2, %2, %11;\n\t"
        "addc.cc.u32 %3, %3, %12;\n\t"
        "addc.cc.u32 %4, %4, %13;\n\t"
        "addc.cc.u32 %5, %5, %14;\n\t"
        "addc.cc.u32 %6, %6, %15;\n\t"
        "addc.cc.u32 %7, %7, %16;\n\t"
        "addc.u32    %8, 0, 0;\n\t"
        : "+r"(a1),"+r"(a2),"+r"(a3),"+r"(a4),
          "+r"(a5),"+r"(a6),"+r"(a7),"+r"(a8),"=r"(a9)
        : "r"(t8),"r"(t9),"r"(t10),"r"(t11),
          "r"(t12),"r"(t13),"r"(t14),"r"(t15)
    );

    // Phase 2: T_lo[0..7] += A[0..7] (32-bit carry chain)
    uint carry;
    __asm volatile(
        "add.cc.u32  %0, %0, %9;\n\t"
        "addc.cc.u32 %1, %1, %10;\n\t"
        "addc.cc.u32 %2, %2, %11;\n\t"
        "addc.cc.u32 %3, %3, %12;\n\t"
        "addc.cc.u32 %4, %4, %13;\n\t"
        "addc.cc.u32 %5, %5, %14;\n\t"
        "addc.cc.u32 %6, %6, %15;\n\t"
        "addc.cc.u32 %7, %7, %16;\n\t"
        "addc.u32    %8, 0, 0;\n\t"
        : "+r"(t0),"+r"(t1),"+r"(t2),"+r"(t3),
          "+r"(t4),"+r"(t5),"+r"(t6),"+r"(t7),"=r"(carry)
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),
          "r"(a4),"r"(a5),"r"(a6),"r"(a7)
    );

    // Phase 3: pack to 64-bit and fold overflow (extra * K)
    // Phase 3: overflow fold (fully 32-bit — no INT64 multiply)
    // extra = a8 + carry + a9*2^32, extra * K_MOD = extra*977 + extra<<32
    uint e_lo, e_carry;
    __asm volatile(
        "add.cc.u32 %0, %2, %3;\n\t"
        "addc.u32 %1, 0, 0;\n\t"
        : "=r"(e_lo), "=r"(e_carry)
        : "r"(a8), "r"(carry)
    );
    uint e_hi = a9 + e_carry;
    uint p_lo, p_hi;
    __asm volatile(
        "mul.lo.u32 %0, %2, 977;\n\t"
        "mul.hi.u32 %1, %2, 977;\n\t"
        : "=r"(p_lo), "=r"(p_hi)
        : "r"(e_lo)
    );
    uint q = e_hi * 977u;
    uint m1 = p_hi + q;
    uint ek0 = p_lo;
    uint ek1, ek1_carry;
    __asm volatile(
        "add.cc.u32 %0, %2, %3;\n\t"
        "addc.u32 %1, 0, 0;\n\t"
        : "=r"(ek1), "=r"(ek1_carry)
        : "r"(m1), "r"(e_lo)
    );
    uint ek2 = e_hi + ek1_carry;
    ulong r0 = ((ulong)t1 << 32) | t0;
    ulong r1 = ((ulong)t3 << 32) | t2;
    ulong r2 = ((ulong)t5 << 32) | t4;
    ulong r3 = ((ulong)t7 << 32) | t6;
    ulong ek_lo = ((ulong)ek1 << 32) | ek0;
    ulong ek_hi = (ulong)ek2;
    ulong c;
    __asm volatile(
        "add.cc.u64  %0, %0, %5;\n\t"
        "addc.cc.u64 %1, %1, %6;\n\t"
        "addc.cc.u64 %2, %2, 0;\n\t"
        "addc.cc.u64 %3, %3, 0;\n\t"
        "addc.u64    %4, 0, 0;\n\t"
        : "+l"(r0),"+l"(r1),"+l"(r2),"+l"(r3),"=l"(c)
        : "l"(ek_lo),"l"(ek_hi)
    );
    // CONSTANT-TIME rare-carry fold: was `if (c) { add SECP256K1_K }` — a
    // data-dependent branch on a secret-derived carry. Now always execute the add
    // with a masked addend (0 when c==0 -> no-op), so wavefront execution is
    // uniform. Mirrors the CUDA reduce_512_to_256_32 (proven CT via ncu).
    {
        ulong cmask = (ulong)0 - (ulong)(c != 0UL);
        __asm volatile("" : "+l"(cmask));   // value barrier
        ulong cfold = (ulong)SECP256K1_K & cmask;
        __asm volatile(
            "add.cc.u64  %0, %0, %4;\n\t"
            "addc.cc.u64 %1, %1, 0;\n\t"
            "addc.cc.u64 %2, %2, 0;\n\t"
            "addc.u64    %3, %3, 0;\n\t"
            : "+l"(r0),"+l"(r1),"+l"(r2),"+l"(r3)
            : "l"(cfold)
        );
    }

    // Phase 4: conditional subtraction of P (64-bit PTX sub.cc chain)
    ulong s0, s1, s2, s3, borrow;
    __asm volatile(
        "sub.cc.u64  %0, %5, %9;\n\t"
        "subc.cc.u64 %1, %6, %10;\n\t"
        "subc.cc.u64 %2, %7, %11;\n\t"
        "subc.cc.u64 %3, %8, %12;\n\t"
        "subc.u64    %4, 0, 0;\n\t"
        : "=l"(s0),"=l"(s1),"=l"(s2),"=l"(s3),"=l"(borrow)
        : "l"(r0),"l"(r1),"l"(r2),"l"(r3),
          "l"(SECP256K1_P0),"l"(SECP256K1_P1),"l"(SECP256K1_P2),"l"(SECP256K1_P3)
    );
    // CONSTANT-TIME final reduction: was `if (borrow==0) r=s; else r=r;` — a
    // data-dependent branch (borrow==0 <=> r >= P). Now branchless cmov: select
    // the subtracted limbs s iff r >= P, else keep r, via a value-barriered mask.
    // Mirrors the CUDA reduce_512_to_256_32 (proven CT via ncu).
    {
        ulong mask = (ulong)0 - (ulong)(borrow == 0UL);
        __asm volatile("" : "+l"(mask));   // value barrier
        r->limbs[0] = (s0 & mask) | (r0 & ~mask);
        r->limbs[1] = (s1 & mask) | (r1 & ~mask);
        r->limbs[2] = (s2 & mask) | (r2 & ~mask);
        r->limbs[3] = (s3 & mask) | (r3 & ~mask);
    }
}

#endif // __NV_CL_C_VERSION

// =============================================================================
// Field Reduction: r = a mod p
// Uses the fact that p = 2^256 - K where K = 0x1000003D1
// So 2^256 ≡ K (mod p), meaning we can reduce by replacing high bits with K*high
// =============================================================================

FORCE_INLINE void field_reduce(FieldElement* r, const ulong* a8) {
    // a8 is 512-bit number (8 limbs), reduce to 256-bit mod p
    // Since p = 2^256 - K, we have: a mod p = a_low + K * a_high (mod p)

    ulong carry = 0;
    ulong temp[5];

    // First reduction: fold a[4..7] into a[0..3] using K
    // temp = a[0..3] + K * a[4..7]

    // Process each high limb
    ulong2 prod;

    // limb 0: a[0] + K * a[4]
    prod = mul64_full(SECP256K1_K, a8[4]);
    temp[0] = a8[0] + prod.x;
    carry = (temp[0] < a8[0]) ? 1UL : 0UL;
    carry += prod.y;

    // limb 1: a[1] + K * a[5] + carry
    prod = mul64_full(SECP256K1_K, a8[5]);
    temp[1] = a8[1] + carry;
    ulong c1 = (temp[1] < carry) ? 1UL : 0UL;
    temp[1] += prod.x;
    c1 += (temp[1] < prod.x) ? 1UL : 0UL;
    carry = c1 + prod.y;

    // limb 2: a[2] + K * a[6] + carry
    prod = mul64_full(SECP256K1_K, a8[6]);
    temp[2] = a8[2] + carry;
    c1 = (temp[2] < carry) ? 1UL : 0UL;
    temp[2] += prod.x;
    c1 += (temp[2] < prod.x) ? 1UL : 0UL;
    carry = c1 + prod.y;

    // limb 3: a[3] + K * a[7] + carry
    prod = mul64_full(SECP256K1_K, a8[7]);
    temp[3] = a8[3] + carry;
    c1 = (temp[3] < carry) ? 1UL : 0UL;
    temp[3] += prod.x;
    c1 += (temp[3] < prod.x) ? 1UL : 0UL;
    temp[4] = c1 + prod.y;

    // Second reduction: fold temp[4]. CONSTANT-TIME: was `if (temp[4] != 0)` with a
    // nested `if (carry)` — data-dependent branches on a secret-derived overflow
    // during signing. Now ALWAYS folded (temp[4]==0 -> K*0=0 -> no-op) and the rare
    // carry fold is masked (mirrors the CUDA/Metal masked rare-carry fold; the
    // per-limb `? :` carries are branchless selects, not branches).
    {
        prod = mul64_full(SECP256K1_K, temp[4]);
        temp[0] += prod.x;
        carry = (temp[0] < prod.x) ? 1UL : 0UL;
        carry += prod.y;

        temp[1] += carry;
        carry = (temp[1] < carry) ? 1UL : 0UL;

        temp[2] += carry;
        carry = (temp[2] < carry) ? 1UL : 0UL;

        temp[3] += carry;
        carry = (temp[3] < carry) ? 1UL : 0UL;

        // Rare carry overflow (~2^-190): fold residual carry branchlessly (0 when carry==0).
        ulong cmask = (ulong)0 - (ulong)(carry != 0UL);
        ulong kfold = (ulong)SECP256K1_K & cmask;
        temp[0] += kfold;
        ulong c2 = (temp[0] < kfold) ? 1UL : 0UL;
        temp[1] += c2;
        c2 = (temp[1] < c2) ? 1UL : 0UL;
        temp[2] += c2;
        c2 = (temp[2] < c2) ? 1UL : 0UL;
        temp[3] += c2;
    }

    // Final reduction: if result >= p, subtract p
    // Check if result >= p by comparing limbs
    ulong borrow = 0;
    ulong diff[4];

    diff[0] = sub_with_borrow(temp[0], SECP256K1_P0, 0, &borrow);
    diff[1] = sub_with_borrow(temp[1], SECP256K1_P1, borrow, &borrow);
    diff[2] = sub_with_borrow(temp[2], SECP256K1_P2, borrow, &borrow);
    diff[3] = sub_with_borrow(temp[3], SECP256K1_P3, borrow, &borrow);

    // If no borrow, result >= p, use subtracted value
    // Otherwise, use original value
    // Branchless selection
    ulong mask = (borrow == 0) ? ~0UL : 0UL;

    r->limbs[0] = (diff[0] & mask) | (temp[0] & ~mask);
    r->limbs[1] = (diff[1] & mask) | (temp[1] & ~mask);
    r->limbs[2] = (diff[2] & mask) | (temp[2] & ~mask);
    r->limbs[3] = (diff[3] & mask) | (temp[3] & ~mask);
}

// =============================================================================
// Field Addition: r = (a + b) mod p
// =============================================================================

FORCE_INLINE void field_add_impl(FieldElement* r, const FieldElement* a, const FieldElement* b) {
#ifdef __NV_CL_C_VERSION
    // Level 2: native add.cc/addc carry chains (no comparison-based carry)
    ulong s0, s1, s2, s3, carry;
    __asm volatile(
        "add.cc.u64  %0, %5, %9;\n\t"
        "addc.cc.u64 %1, %6, %10;\n\t"
        "addc.cc.u64 %2, %7, %11;\n\t"
        "addc.cc.u64 %3, %8, %12;\n\t"
        "addc.u64    %4, 0, 0;\n\t"
        : "=l"(s0),"=l"(s1),"=l"(s2),"=l"(s3),"=l"(carry)
        : "l"(a->limbs[0]),"l"(a->limbs[1]),"l"(a->limbs[2]),"l"(a->limbs[3]),
          "l"(b->limbs[0]),"l"(b->limbs[1]),"l"(b->limbs[2]),"l"(b->limbs[3])
    );
    ulong d0, d1, d2, d3, borrow;
    __asm volatile(
        "sub.cc.u64  %0, %5, %9;\n\t"
        "subc.cc.u64 %1, %6, %10;\n\t"
        "subc.cc.u64 %2, %7, %11;\n\t"
        "subc.cc.u64 %3, %8, %12;\n\t"
        "subc.u64    %4, 0, 0;\n\t"
        : "=l"(d0),"=l"(d1),"=l"(d2),"=l"(d3),"=l"(borrow)
        : "l"(s0),"l"(s1),"l"(s2),"l"(s3),
          "l"(SECP256K1_P0),"l"(SECP256K1_P1),"l"(SECP256K1_P2),"l"(SECP256K1_P3)
    );
    // use diff if: no borrow (s >= P) OR carry from add (sum overflowed 2^256)
    ulong mask = ~borrow | (0UL - carry);
    r->limbs[0] = (d0 & mask) | (s0 & ~mask);
    r->limbs[1] = (d1 & mask) | (s1 & ~mask);
    r->limbs[2] = (d2 & mask) | (s2 & ~mask);
    r->limbs[3] = (d3 & mask) | (s3 & ~mask);
#else
    ulong carry = 0;
    ulong sum[4];
    sum[0] = add_with_carry(a->limbs[0], b->limbs[0], 0, &carry);
    sum[1] = add_with_carry(a->limbs[1], b->limbs[1], carry, &carry);
    sum[2] = add_with_carry(a->limbs[2], b->limbs[2], carry, &carry);
    sum[3] = add_with_carry(a->limbs[3], b->limbs[3], carry, &carry);
    ulong borrow = 0;
    ulong diff[4];
    diff[0] = sub_with_borrow(sum[0], SECP256K1_P0, 0, &borrow);
    diff[1] = sub_with_borrow(sum[1], SECP256K1_P1, borrow, &borrow);
    diff[2] = sub_with_borrow(sum[2], SECP256K1_P2, borrow, &borrow);
    diff[3] = sub_with_borrow(sum[3], SECP256K1_P3, borrow, &borrow);
    ulong use_diff = (carry != 0) | (borrow == 0);
    ulong mask = use_diff ? ~0UL : 0UL;
    r->limbs[0] = (diff[0] & mask) | (sum[0] & ~mask);
    r->limbs[1] = (diff[1] & mask) | (sum[1] & ~mask);
    r->limbs[2] = (diff[2] & mask) | (sum[2] & ~mask);
    r->limbs[3] = (diff[3] & mask) | (sum[3] & ~mask);
#endif
}

// =============================================================================
// Field Subtraction: r = (a - b) mod p
// =============================================================================

FORCE_INLINE void field_sub_impl(FieldElement* r, const FieldElement* a, const FieldElement* b) {
#ifdef __NV_CL_C_VERSION
    // Level 2: native sub.cc/subc + add.cc/addc carry chains
    ulong d0, d1, d2, d3, borrow;
    __asm volatile(
        "sub.cc.u64  %0, %5, %9;\n\t"
        "subc.cc.u64 %1, %6, %10;\n\t"
        "subc.cc.u64 %2, %7, %11;\n\t"
        "subc.cc.u64 %3, %8, %12;\n\t"
        "subc.u64    %4, 0, 0;\n\t"
        : "=l"(d0),"=l"(d1),"=l"(d2),"=l"(d3),"=l"(borrow)
        : "l"(a->limbs[0]),"l"(a->limbs[1]),"l"(a->limbs[2]),"l"(a->limbs[3]),
          "l"(b->limbs[0]),"l"(b->limbs[1]),"l"(b->limbs[2]),"l"(b->limbs[3])
    );
    // borrow = 0xFFFF...FFFF if a < b (underflow), 0 otherwise
    ulong p0 = SECP256K1_P0 & borrow;
    ulong p1 = SECP256K1_P1 & borrow;
    ulong p2 = SECP256K1_P2 & borrow;
    ulong p3 = SECP256K1_P3 & borrow;
    __asm volatile(
        "add.cc.u64  %0, %4, %8;\n\t"
        "addc.cc.u64 %1, %5, %9;\n\t"
        "addc.cc.u64 %2, %6, %10;\n\t"
        "addc.u64    %3, %7, %11;\n\t"
        : "=l"(r->limbs[0]),"=l"(r->limbs[1]),"=l"(r->limbs[2]),"=l"(r->limbs[3])
        : "l"(d0),"l"(d1),"l"(d2),"l"(d3), "l"(p0),"l"(p1),"l"(p2),"l"(p3)
    );
#else
    ulong borrow = 0;
    ulong diff[4];
    diff[0] = sub_with_borrow(a->limbs[0], b->limbs[0], 0, &borrow);
    diff[1] = sub_with_borrow(a->limbs[1], b->limbs[1], borrow, &borrow);
    diff[2] = sub_with_borrow(a->limbs[2], b->limbs[2], borrow, &borrow);
    diff[3] = sub_with_borrow(a->limbs[3], b->limbs[3], borrow, &borrow);
    ulong mask = borrow ? ~0UL : 0UL;
    ulong carry = 0;
    ulong adj[4];
    adj[0] = add_with_carry(diff[0], SECP256K1_P0 & mask, 0, &carry);
    adj[1] = add_with_carry(diff[1], SECP256K1_P1 & mask, carry, &carry);
    adj[2] = add_with_carry(diff[2], SECP256K1_P2 & mask, carry, &carry);
    adj[3] = add_with_carry(diff[3], SECP256K1_P3 & mask, carry, &carry);
    r->limbs[0] = adj[0];
    r->limbs[1] = adj[1];
    r->limbs[2] = adj[2];
    r->limbs[3] = adj[3];
#endif
}

// =============================================================================
// Field Multiplication: r = (a * b) mod p
// =============================================================================

// Helper: add 128-bit product (hi:lo) into 3-register accumulator (c2:c1:c0)
FORCE_INLINE void muladd(ulong lo, ulong hi, ulong* c0, ulong* c1, ulong* c2) {
    ulong carry;
    *c0 = add_with_carry(*c0, lo, 0, &carry);
    *c1 = add_with_carry(*c1, hi, carry, &carry);
    *c2 += carry;
}

// Helper: add 128-bit product (hi:lo) doubled into accumulator
FORCE_INLINE void muladd2(ulong lo, ulong hi, ulong* c0, ulong* c1, ulong* c2) {
    muladd(lo, hi, c0, c1, c2);
    muladd(lo, hi, c0, c1, c2);
}

FORCE_INLINE void field_mul_impl(FieldElement* r, const FieldElement* a, const FieldElement* b) {
#ifdef __NV_CL_C_VERSION
    // Level 3: 32-bit hybrid Comba + 32-bit reduction (INT32 throughput 32x > INT64)
    uint t32[16];
    mul_256_comba32_ocl(a, b, t32);
    reduce_512_to_256_32_ocl(t32, r);
#else
    ulong a0 = a->limbs[0], a1 = a->limbs[1], a2 = a->limbs[2], a3 = a->limbs[3];
    ulong b0 = b->limbs[0], b1 = b->limbs[1], b2 = b->limbs[2], b3 = b->limbs[3];
    ulong product[8];
    ulong c0, c1, c2;
    ulong2 m;

    // Column 0: a0*b0
    c0 = 0; c1 = 0; c2 = 0;
    m = mul64_full(a0, b0); muladd(m.x, m.y, &c0, &c1, &c2);
    product[0] = c0; c0 = c1; c1 = c2; c2 = 0;

    // Column 1: a0*b1 + a1*b0
    m = mul64_full(a0, b1); muladd(m.x, m.y, &c0, &c1, &c2);
    m = mul64_full(a1, b0); muladd(m.x, m.y, &c0, &c1, &c2);
    product[1] = c0; c0 = c1; c1 = c2; c2 = 0;

    // Column 2: a0*b2 + a1*b1 + a2*b0
    m = mul64_full(a0, b2); muladd(m.x, m.y, &c0, &c1, &c2);
    m = mul64_full(a1, b1); muladd(m.x, m.y, &c0, &c1, &c2);
    m = mul64_full(a2, b0); muladd(m.x, m.y, &c0, &c1, &c2);
    product[2] = c0; c0 = c1; c1 = c2; c2 = 0;

    // Column 3: a0*b3 + a1*b2 + a2*b1 + a3*b0
    m = mul64_full(a0, b3); muladd(m.x, m.y, &c0, &c1, &c2);
    m = mul64_full(a1, b2); muladd(m.x, m.y, &c0, &c1, &c2);
    m = mul64_full(a2, b1); muladd(m.x, m.y, &c0, &c1, &c2);
    m = mul64_full(a3, b0); muladd(m.x, m.y, &c0, &c1, &c2);
    product[3] = c0; c0 = c1; c1 = c2; c2 = 0;

    // Column 4: a1*b3 + a2*b2 + a3*b1
    m = mul64_full(a1, b3); muladd(m.x, m.y, &c0, &c1, &c2);
    m = mul64_full(a2, b2); muladd(m.x, m.y, &c0, &c1, &c2);
    m = mul64_full(a3, b1); muladd(m.x, m.y, &c0, &c1, &c2);
    product[4] = c0; c0 = c1; c1 = c2; c2 = 0;

    // Column 5: a2*b3 + a3*b2
    m = mul64_full(a2, b3); muladd(m.x, m.y, &c0, &c1, &c2);
    m = mul64_full(a3, b2); muladd(m.x, m.y, &c0, &c1, &c2);
    product[5] = c0; c0 = c1; c1 = c2; c2 = 0;

    // Column 6: a3*b3
    m = mul64_full(a3, b3); muladd(m.x, m.y, &c0, &c1, &c2);
    product[6] = c0;
    product[7] = c1;

    field_reduce(r, product);
#endif
}

// =============================================================================
// Field Squaring: r = a² mod p
// =============================================================================

// Forward declaration for field_sqr_n_impl
FORCE_INLINE void field_sqr_impl(FieldElement* r, const FieldElement* a);

// Repeated squaring helper: r = r^(2^n) — in-place
FORCE_INLINE void field_sqr_n_impl(FieldElement* r, int n) {
    for (int i = 0; i < n; i++) field_sqr_impl(r, r);
}

FORCE_INLINE void field_sqr_impl(FieldElement* r, const FieldElement* a) {
#ifdef __NV_CL_C_VERSION
    // Level 3: 32-bit hybrid squaring (40% fewer multiplications + INT32 throughput)
    uint t32[16];
    sqr_256_comba32_ocl(a, t32);
    reduce_512_to_256_32_ocl(t32, r);
#else
    ulong a0 = a->limbs[0], a1 = a->limbs[1], a2 = a->limbs[2], a3 = a->limbs[3];
    ulong product[8];
    ulong c0, c1, c2;
    ulong2 m;

    // Column 0: a0*a0
    c0 = 0; c1 = 0; c2 = 0;
    m = mul64_full(a0, a0); muladd(m.x, m.y, &c0, &c1, &c2);
    product[0] = c0; c0 = c1; c1 = c2; c2 = 0;

    // Column 1: 2*a0*a1
    m = mul64_full(a0, a1); muladd2(m.x, m.y, &c0, &c1, &c2);
    product[1] = c0; c0 = c1; c1 = c2; c2 = 0;

    // Column 2: 2*a0*a2 + a1*a1
    m = mul64_full(a0, a2); muladd2(m.x, m.y, &c0, &c1, &c2);
    m = mul64_full(a1, a1); muladd(m.x, m.y, &c0, &c1, &c2);
    product[2] = c0; c0 = c1; c1 = c2; c2 = 0;

    // Column 3: 2*a0*a3 + 2*a1*a2
    m = mul64_full(a0, a3); muladd2(m.x, m.y, &c0, &c1, &c2);
    m = mul64_full(a1, a2); muladd2(m.x, m.y, &c0, &c1, &c2);
    product[3] = c0; c0 = c1; c1 = c2; c2 = 0;

    // Column 4: 2*a1*a3 + a2*a2
    m = mul64_full(a1, a3); muladd2(m.x, m.y, &c0, &c1, &c2);
    m = mul64_full(a2, a2); muladd(m.x, m.y, &c0, &c1, &c2);
    product[4] = c0; c0 = c1; c1 = c2; c2 = 0;

    // Column 5: 2*a2*a3
    m = mul64_full(a2, a3); muladd2(m.x, m.y, &c0, &c1, &c2);
    product[5] = c0; c0 = c1; c1 = c2; c2 = 0;

    // Column 6: a3*a3
    m = mul64_full(a3, a3); muladd(m.x, m.y, &c0, &c1, &c2);
    product[6] = c0;
    product[7] = c1;

    field_reduce(r, product);
#endif
}

// =============================================================================
// Field Negation: r = -a mod p = p - a
// =============================================================================

FORCE_INLINE void field_neg_impl(FieldElement* r, const FieldElement* a) {
    // Check if a is zero
    ulong is_zero = ((a->limbs[0] | a->limbs[1] | a->limbs[2] | a->limbs[3]) == 0) ? 1UL : 0UL;

    ulong borrow = 0;
    r->limbs[0] = sub_with_borrow(SECP256K1_P0, a->limbs[0], 0, &borrow);
    r->limbs[1] = sub_with_borrow(SECP256K1_P1, a->limbs[1], borrow, &borrow);
    r->limbs[2] = sub_with_borrow(SECP256K1_P2, a->limbs[2], borrow, &borrow);
    r->limbs[3] = sub_with_borrow(SECP256K1_P3, a->limbs[3], borrow, &borrow);

    // If a was zero, result should be zero
    ulong mask = is_zero ? 0UL : ~0UL;
    r->limbs[0] &= mask;
    r->limbs[1] &= mask;
    r->limbs[2] &= mask;
    r->limbs[3] &= mask;
}

// =============================================================================
// Field Inversion: r = a^(-1) mod p
// Using Fermat's little theorem with optimized addition chain
// Matches CUDA's field_inv_fermat_chain for minimal mul+sqr count
// p-2 = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2D
// =============================================================================

FORCE_INLINE void field_inv_impl(FieldElement* r, const FieldElement* a) {
    FieldElement x2, x3, x6, x12, x24, x48, x96, x192, x7, x31, x223;
    FieldElement x5, x11, x22;
    FieldElement t;

    // 1. x2 = a^2 * a  (2 consecutive ones)
    field_sqr_impl(&x2, a);
    field_mul_impl(&x2, &x2, a);

    // 2. x3 = x2^2 * a  (3 consecutive ones)
    field_sqr_impl(&x3, &x2);
    field_mul_impl(&x3, &x3, a);

    // 3. x6 = x3^(2^3) * x3  (6 consecutive ones)
    field_sqr_impl(&x6, &x3);
    field_sqr_n_impl(&x6, 2);
    field_mul_impl(&x6, &x6, &x3);

    // 4. x12 = x6^(2^6) * x6  (12 consecutive ones)
    t = x6;
    field_sqr_n_impl(&t, 6);
    field_mul_impl(&x12, &t, &x6);

    // 5. x24 = x12^(2^12) * x12  (24 consecutive ones)
    t = x12;
    field_sqr_n_impl(&t, 12);
    field_mul_impl(&x24, &t, &x12);

    // 6. x48 = x24^(2^24) * x24  (48 consecutive ones)
    t = x24;
    field_sqr_n_impl(&t, 24);
    field_mul_impl(&x48, &t, &x24);

    // 7. x96 = x48^(2^48) * x48  (96 consecutive ones)
    t = x48;
    field_sqr_n_impl(&t, 48);
    field_mul_impl(&x96, &t, &x48);

    // 8. x192 = x96^(2^96) * x96  (192 consecutive ones)
    t = x96;
    field_sqr_n_impl(&t, 96);
    field_mul_impl(&x192, &t, &x96);

    // 9. x7 = x6^2 * a  (7 consecutive ones)
    field_sqr_impl(&x7, &x6);
    field_mul_impl(&x7, &x7, a);

    // 10. x31 = x24^(2^7) * x7  (31 consecutive ones)
    t = x24;
    field_sqr_n_impl(&t, 7);
    field_mul_impl(&x31, &t, &x7);

    // 11. x223 = x192^(2^31) * x31  (223 consecutive ones)
    t = x192;
    field_sqr_n_impl(&t, 31);
    field_mul_impl(&x223, &t, &x31);

    // 12. x5 = x3^(2^2) * x2  (5 consecutive ones)
    t = x3;
    field_sqr_n_impl(&t, 2);
    field_mul_impl(&x5, &t, &x2);

    // 13. x11 = x6^(2^5) * x5  (11 consecutive ones)
    t = x6;
    field_sqr_n_impl(&t, 5);
    field_mul_impl(&x11, &t, &x5);

    // 14. x22 = x11^(2^11) * x11  (22 consecutive ones)
    t = x11;
    field_sqr_n_impl(&t, 11);
    field_mul_impl(&x22, &t, &x11);

    // 15. t = x223^2  (bit 32 is 0)
    field_sqr_impl(&t, &x223);

    // 16. t = t^(2^22) * x22  (append 22 ones)
    field_sqr_n_impl(&t, 22);
    field_mul_impl(&t, &t, &x22);

    // 17. t = t^(2^4)  (bits 9,8,7,6 are 0)
    field_sqr_n_impl(&t, 4);

    // 18. Process remaining 6 bits: 101101
    // bit 5: 1
    field_sqr_impl(&t, &t);
    field_mul_impl(&t, &t, a);
    // bit 4: 0
    field_sqr_impl(&t, &t);
    // bit 3: 1
    field_sqr_impl(&t, &t);
    field_mul_impl(&t, &t, a);
    // bit 2: 1
    field_sqr_impl(&t, &t);
    field_mul_impl(&t, &t, a);
    // bit 1: 0
    field_sqr_impl(&t, &t);
    // bit 0: 1
    field_sqr_impl(&t, &t);
    field_mul_impl(r, &t, a);
}

// =============================================================================
// Variable-time modular inverse (Bernstein-Yang SafeGCD) -- PUBLIC DATA ONLY
// =============================================================================
//   field_inv_var_impl(r, a)    r = a^-1 mod p   (defined below)
//   scalar_inv_var_impl(a, r)   r = a^-1 mod n   (secp256k1_extended.cl, next
//                                                 to scalar_inverse_impl)
//   modinv30_var_impl(...)      the core both of them call
//
// !! VARIABLE TIME -- PUBLIC DATA ONLY !!
// The number of loop iterations depends on the VALUE being inverted. Never
// call these on a private key, a nonce, a signing share, an ECDH scalar, a
// BIP-32 / BIP-352 secret, or on the Z coordinate of a point produced by a
// secret scalar multiplication (k*G, sk*G, sk*P). Signing, ECDH, key
// generation, provers, every secp256k1_ct_*.cl file and the shared affine
// helpers keep field_inv_impl / scalar_inverse_impl / the ct_* inverses.
// OpenCL has no namespaces, so the rule is enforced by a source scan in
// audit/test_regression_opencl_kernel_closure.cpp: these three names may not
// appear in any secp256k1_ct_*.cl file, nor inside any function whose name
// contains sign, ecdh, keygen, derive or prove.
// Callers: signature verification, public-key recovery, the Taproot
// commitment check, the ECDSA SNARK witness, FROST partial verification and
// the Shamir helper used only by verifiers; docs/AUDIT_CHANGELOG.md has the
// per-site reasoning. A new caller needs its own argument that no secret, or
// value derived from one, reaches the inversion.
//
// Contract (same value as field_inv_impl / scalar_inverse_impl):
//   - input: any 256-bit value, reduced or not; input and output may alias
//   - output: canonical, in [0, modulus); 0 when the input is 0 mod modulus
//   - always terminates: at most MODINV30_MAX_ITERS outer iterations of at
//     most 30 inner passes each; if the cap were reached the result is 0
//     (fail closed) -- not reachable for a 256-bit input, see the cap below
//
// Algorithm: Bernstein-Yang safegcd ("Fast constant-time gcd computation and
// modular inversion", 2019), variable-time form, 30 divsteps per batch on
// 9 signed 30-bit limbs with 64-bit accumulators. Port of the CUDA code in
// src/cuda/include/modinv_var.cuh, itself a port of this library's CPU
// safegcd30 code, which is adapted from bitcoin-core/secp256k1
// secp256k1_modinv32_var (src/modinv32_impl.h):
//   Copyright (c) 2020 Peter Dettman
//   Distributed under the MIT software license, see
//   https://www.opensource.org/licenses/mit-license.php.
// As in the CUDA code: hard cap on the outer loop, table-free 6-bit inner
// step w = f*g*(f*f-2), f and g always keep 9 limbs (compile-time indices
// only, so the state stays in registers), the input is reduced first, and on
// NVIDIA each signed 32x32->64 product is one mul.wide.s32.
// =============================================================================

// 25 * 30 = 750 divsteps. 741 divsteps are proven sufficient for any 256-bit
// input with delta starting at 1 (Bernstein-Yang, Theorem 11.2); 724 by
// sipa/safegcd-bounds. bitcoin-core/secp256k1 modinv32_var uses the same 25.
#define MODINV30_MAX_ITERS 25

// acc + a*b, signed 32x32->64.
FORCE_INLINE long modinv30_mul_add(long acc, int a, int b) {
#ifdef __NV_CL_C_VERSION
    long p;
    __asm volatile("mul.wide.s32 %0, %1, %2;" : "=l"(p) : "r"(a), "r"(b));
    return acc + p;
#else
    return acc + (long)a * (long)b;
#endif
}

// 30 divsteps on the low words of f and g. Returns the new eta (= -delta) and
// writes the transition matrix (u, v, q, r), scaled by 2^30.
// Bounded for ANY f, g: every pass after the first removes at least one of
// the 30 steps (f is forced odd by the caller, and g is odd after the shift).
FORCE_INLINE int modinv30_divsteps_30_var(int eta, uint f, uint g,
                                          int* tu, int* tv, int* tq, int* tr) {
    uint u = 1u, v = 0u, q = 0u, r = 1u;
    int i = 30;
    for (;;) {
        // strip zero bits of g; the sentinel at bit i stops the count at i
        const uint x = g | (0xFFFFFFFFu << i);
        const int zeros = 31 - (int)clz(x & (0u - x));
        g >>= zeros; u <<= zeros; v <<= zeros; eta -= zeros; i -= zeros;
        if (i == 0) break;
        // eta < 0: (f, g) <- (g, -f), same for the matrix rows (selects)
        const int swap = eta < 0;
        const uint f0 = f, u0 = u, v0 = v;
        eta = swap ? -eta : eta;
        f = swap ? g : f0;   g = swap ? (0u - f0) : g;
        u = swap ? q : u0;   q = swap ? (0u - u0) : q;
        v = swap ? r : v0;   r = swap ? (0u - v0) : r;
        // cancel up to 6 low bits of g with a multiple of f (f odd):
        // g + f*w == g*(f*f-1)^2 == 0 (mod 64)
        const int limit = (eta + 1) < i ? (eta + 1) : i;
        const uint m = (0xFFFFFFFFu >> (32 - limit)) & 63u;
        const uint w = ((f * g) * (f * f - 2u)) & m;
        g += f * w; q += u * w; r += v * w;
    }
    *tu = (int)u; *tv = (int)v; *tq = (int)q; *tr = (int)r;
    return eta;
}

// r4 = (a4 mod M)^-1 mod M, canonical, where M = n when is_n != 0, else p.
// 0 if a4 == 0 mod M or if max_iters is reached. Returns the number of outer
// iterations used. a4 == r4 is allowed. Every call site passes a literal
// is_n, so after inlining the modulus constants fold to immediates.
// max_iters is a parameter only so that the OpenCL audit runner can exercise
// the fail-closed path; production callers pass MODINV30_MAX_ITERS.
// VARIABLE TIME, PUBLIC DATA ONLY (see the block comment above).
FORCE_INLINE int modinv30_var_impl(const ulong* a4, ulong* r4,
                                   const int is_n, const int max_iters) {
    const uint M30 = 0x3FFFFFFFu;

    // Modulus as 9 signed 30-bit limbs (value = sum ml[i] * 2^(30*i)), its
    // inverse mod 2^30, and the same modulus as 4 x 64-bit limbs.
    const int ml[9] = {
        is_n ? 0x10364141 : -0x3D1,  is_n ? 0x3F497A33 : -4,
        is_n ? 0x348A03BB : 0,       is_n ? 0x2BB739AB : 0,
        is_n ? -0x146 : 0,           0, 0, 0, 65536
    };
    const uint inv30 = is_n ? 0x2A774EC1u : 0x2DDACACFu;
    const ulong m0 = is_n ? 0xBFD25E8CD0364141UL : 0xFFFFFFFEFFFFFC2FUL;
    const ulong m1 = is_n ? 0xBAAEDCE6AF48A03BUL : 0xFFFFFFFFFFFFFFFFUL;
    const ulong m2 = is_n ? 0xFFFFFFFFFFFFFFFEUL : 0xFFFFFFFFFFFFFFFFUL;
    const ulong m3 = 0xFFFFFFFFFFFFFFFFUL;

    // x = a >= M ? a - M : a   (M < 2^256 < 2M, so one subtraction is enough)
    ulong x0 = a4[0], x1 = a4[1], x2 = a4[2], x3 = a4[3];
    {
        ulong borrow = 0;
        const ulong t0 = sub_with_borrow(x0, m0, 0UL, &borrow);
        const ulong t1 = sub_with_borrow(x1, m1, borrow, &borrow);
        const ulong t2 = sub_with_borrow(x2, m2, borrow, &borrow);
        const ulong t3 = sub_with_borrow(x3, m3, borrow, &borrow);
        const ulong keep = 0UL - borrow;            // all-ones when a < M
        x0 = (x0 & keep) | (t0 & ~keep);
        x1 = (x1 & keep) | (t1 & ~keep);
        x2 = (x2 & keep) | (t2 & ~keep);
        x3 = (x3 & keep) | (t3 & ~keep);
    }

    // d = 0, e = 1, f = M, g = x
    int d[9], e[9], f[9], g[9];
    #pragma unroll
    for (int i = 0; i < 9; ++i) { d[i] = 0; e[i] = 0; f[i] = ml[i]; }
    e[0] = 1;
    g[0] = (int)((uint)x0 & M30);
    g[1] = (int)((uint)(x0 >> 30) & M30);
    g[2] = (int)((uint)((x0 >> 60) | (x1 << 4)) & M30);
    g[3] = (int)((uint)(x1 >> 26) & M30);
    g[4] = (int)((uint)((x1 >> 56) | (x2 << 8)) & M30);
    g[5] = (int)((uint)(x2 >> 22) & M30);
    g[6] = (int)((uint)((x2 >> 52) | (x3 << 12)) & M30);
    g[7] = (int)((uint)(x3 >> 18) & M30);
    g[8] = (int)(x3 >> 48);

    int eta = -1;
    int done = 0;
    int it = 0;
    while (it < max_iters) {
        ++it;
        int u, v, q, r;
        // "| 1u": f is always odd, the OR only makes the inner loop bound
        // independent of that invariant
        eta = modinv30_divsteps_30_var(eta, (uint)f[0] | 1u, (uint)g[0], &u, &v, &q, &r);

        // (d, e) <- (matrix / 2^30) * (d, e) mod M
        {
            const int sd = d[8] >> 31, se = e[8] >> 31;
            int md = (u & sd) + (v & se);
            int me = (q & sd) + (r & se);
            long cd = modinv30_mul_add(modinv30_mul_add(0L, u, d[0]), v, e[0]);
            long ce = modinv30_mul_add(modinv30_mul_add(0L, q, d[0]), r, e[0]);
            md -= (int)((inv30 * (uint)cd + (uint)md) & M30);
            me -= (int)((inv30 * (uint)ce + (uint)me) & M30);
            cd += (long)ml[0] * md;
            ce += (long)ml[0] * me;
            cd >>= 30; ce >>= 30;                    // low 30 bits are zero
            #pragma unroll
            for (int i = 1; i < 9; ++i) {
                cd = modinv30_mul_add(modinv30_mul_add(cd, u, d[i]), v, e[i]);
                ce = modinv30_mul_add(modinv30_mul_add(ce, q, d[i]), r, e[i]);
                if (ml[i] != 0) {                    // constant after unrolling
                    cd += (long)ml[i] * md;
                    ce += (long)ml[i] * me;
                }
                d[i - 1] = (int)((uint)cd & M30); cd >>= 30;
                e[i - 1] = (int)((uint)ce & M30); ce >>= 30;
            }
            d[8] = (int)cd; e[8] = (int)ce;
        }

        // (f, g) <- (matrix / 2^30) * (f, g), exact; g_nonzero = OR of the new g
        int g_nonzero;
        {
            long cf = modinv30_mul_add(modinv30_mul_add(0L, u, f[0]), v, g[0]);
            long cg = modinv30_mul_add(modinv30_mul_add(0L, q, f[0]), r, g[0]);
            cf >>= 30; cg >>= 30;                    // low 30 bits are zero
            g_nonzero = 0;
            #pragma unroll
            for (int i = 1; i < 9; ++i) {
                cf = modinv30_mul_add(modinv30_mul_add(cf, u, f[i]), v, g[i]);
                cg = modinv30_mul_add(modinv30_mul_add(cg, q, f[i]), r, g[i]);
                f[i - 1] = (int)((uint)cf & M30); cf >>= 30;
                g[i - 1] = (int)((uint)cg & M30); cg >>= 30;
                g_nonzero |= g[i - 1];
            }
            f[8] = (int)cf; g[8] = (int)cg;
            g_nonzero |= g[8];
        }
        if (g_nonzero == 0) { done = 1; break; }
    }

    // normalize d to [0, M), negating first when f == -1
    {
        int cond_add = d[8] >> 31;
        #pragma unroll
        for (int i = 0; i < 9; ++i) d[i] += ml[i] & cond_add;
        const int cond_neg = f[8] >> 31;
        #pragma unroll
        for (int i = 0; i < 9; ++i) d[i] = (d[i] ^ cond_neg) - cond_neg;
        #pragma unroll
        for (int i = 0; i < 8; ++i) { d[i + 1] += d[i] >> 30; d[i] &= (int)M30; }
        cond_add = d[8] >> 31;
        #pragma unroll
        for (int i = 0; i < 9; ++i) d[i] += ml[i] & cond_add;
        #pragma unroll
        for (int i = 0; i < 8; ++i) { d[i + 1] += d[i] >> 30; d[i] &= (int)M30; }
    }

    const ulong ok = 0UL - (ulong)done;             // cap reached -> 0 (fail closed)
    const ulong d0 = (uint)d[0], d1 = (uint)d[1], d2 = (uint)d[2],
                d3 = (uint)d[3], d4 = (uint)d[4], d5 = (uint)d[5],
                d6 = (uint)d[6], d7 = (uint)d[7], d8 = (uint)d[8];
    r4[0] = ok & (d0 | (d1 << 30) | (d2 << 60));
    r4[1] = ok & ((d2 >> 4) | (d3 << 26) | (d4 << 56));
    r4[2] = ok & ((d4 >> 8) | (d5 << 22) | (d6 << 52));
    r4[3] = ok & ((d6 >> 12) | (d7 << 18) | (d8 << 48));
    return it;
}

// r = a^-1 mod p. VARIABLE TIME, PUBLIC DATA ONLY (see the block comment above).
FORCE_INLINE void field_inv_var_impl(FieldElement* r, const FieldElement* a) {
    (void)modinv30_var_impl(a->limbs, r->limbs, 0, MODINV30_MAX_ITERS);
}

// =============================================================================
// OpenCL Kernels
// =============================================================================

__kernel void field_add(
    __global const FieldElement* a,
    __global const FieldElement* b,
    __global FieldElement* result,
    const uint count
) {
    uint gid = get_global_id(0);
    if (gid >= count) return;

    // Copy from global to private memory
    FieldElement a_local = a[gid];
    FieldElement b_local = b[gid];
    FieldElement r;
    field_add_impl(&r, &a_local, &b_local);
    result[gid] = r;
}

__kernel void field_sub(
    __global const FieldElement* a,
    __global const FieldElement* b,
    __global FieldElement* result,
    const uint count
) {
    uint gid = get_global_id(0);
    if (gid >= count) return;

    // Copy from global to private memory
    FieldElement a_local = a[gid];
    FieldElement b_local = b[gid];
    FieldElement r;
    field_sub_impl(&r, &a_local, &b_local);
    result[gid] = r;
}

__kernel void field_mul(
    __global const FieldElement* a,
    __global const FieldElement* b,
    __global FieldElement* result,
    const uint count
) {
    uint gid = get_global_id(0);
    if (gid >= count) return;

    // Copy from global to private memory
    FieldElement a_local = a[gid];
    FieldElement b_local = b[gid];
    FieldElement r;
    field_mul_impl(&r, &a_local, &b_local);
    result[gid] = r;
}

__kernel void field_sqr(
    __global const FieldElement* a,
    __global FieldElement* result,
    const uint count
) {
    uint gid = get_global_id(0);
    if (gid >= count) return;

    // Copy from global to private memory
    FieldElement a_local = a[gid];
    FieldElement r;
    field_sqr_impl(&r, &a_local);
    result[gid] = r;
}

__kernel void field_inv(
    __global const FieldElement* a,
    __global FieldElement* result,
    const uint count
) {
    uint gid = get_global_id(0);
    if (gid >= count) return;

    // Copy from global to private memory
    FieldElement a_local = a[gid];
    FieldElement r;
    field_inv_impl(&r, &a_local);
    result[gid] = r;
}

