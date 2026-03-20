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
// NVIDIA's OpenCL compiler (nvoc) treats 'inline' as advisory.
// __attribute__((always_inline)) forces inlining of the entire field arithmetic
// call chain (field_mul → comba → reduce), matching CUDA's __forceinline__.
#ifdef __NV_CL_C_VERSION
  #define FORCE_INLINE __attribute__((always_inline)) inline
  #define FORCE_INLINE_STATIC __attribute__((always_inline)) static inline
#else
  #define FORCE_INLINE inline
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
    if (c) {
        __asm volatile(
            "add.cc.u64  %0, %0, %4;\n\t"
            "addc.cc.u64 %1, %1, 0;\n\t"
            "addc.cc.u64 %2, %2, 0;\n\t"
            "addc.u64    %3, %3, 0;\n\t"
            : "+l"(r0),"+l"(r1),"+l"(r2),"+l"(r3)
            : "l"((ulong)SECP256K1_K)
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
    if (borrow == 0) {
        r->limbs[0]=s0; r->limbs[1]=s1; r->limbs[2]=s2; r->limbs[3]=s3;
    } else {
        r->limbs[0]=r0; r->limbs[1]=r1; r->limbs[2]=r2; r->limbs[3]=r3;
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

    // Second reduction: if temp[4] > 0, fold it in
    if (temp[4] != 0) {
        prod = mul64_full(SECP256K1_K, temp[4]);
        temp[0] += prod.x;
        carry = (temp[0] < prod.x) ? 1UL : 0UL;
        carry += prod.y;

        temp[1] += carry;
        carry = (temp[1] < carry) ? 1UL : 0UL;

        temp[2] += carry;
        carry = (temp[2] < carry) ? 1UL : 0UL;

        temp[3] += carry;
        // At this point result fits in 256 bits (plus possible 1-bit overflow)
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

    // Add with carry chain
    sum[0] = add_with_carry(a->limbs[0], b->limbs[0], 0, &carry);
    sum[1] = add_with_carry(a->limbs[1], b->limbs[1], carry, &carry);
    sum[2] = add_with_carry(a->limbs[2], b->limbs[2], carry, &carry);
    sum[3] = add_with_carry(a->limbs[3], b->limbs[3], carry, &carry);

    // Reduce: if carry or sum >= p, subtract p
    ulong borrow = 0;
    ulong diff[4];

    diff[0] = sub_with_borrow(sum[0], SECP256K1_P0, 0, &borrow);
    diff[1] = sub_with_borrow(sum[1], SECP256K1_P1, borrow, &borrow);
    diff[2] = sub_with_borrow(sum[2], SECP256K1_P2, borrow, &borrow);
    diff[3] = sub_with_borrow(sum[3], SECP256K1_P3, borrow, &borrow);

    // If carry from addition or no borrow from subtraction, use diff
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

    // Subtract with borrow chain
    diff[0] = sub_with_borrow(a->limbs[0], b->limbs[0], 0, &borrow);
    diff[1] = sub_with_borrow(a->limbs[1], b->limbs[1], borrow, &borrow);
    diff[2] = sub_with_borrow(a->limbs[2], b->limbs[2], borrow, &borrow);
    diff[3] = sub_with_borrow(a->limbs[3], b->limbs[3], borrow, &borrow);

    // If borrow, add p (result was negative)
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
// Optimized: only need upper triangle of multiplication
// =============================================================================

// Forward declaration for field_sqr_n_impl
FORCE_INLINE void field_sqr_impl(FieldElement* r, const FieldElement* a);

// Repeated squaring helper: r = r^(2^n) — in-place
FORCE_INLINE void field_sqr_n_impl(FieldElement* r, int n) {
    for (int i = 0; i < n; i++) {
        FieldElement tmp = *r;
        field_sqr_impl(r, &tmp);
    }
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

