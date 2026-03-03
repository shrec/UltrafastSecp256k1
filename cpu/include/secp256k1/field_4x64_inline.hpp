#pragma once
// ===========================================================================
// Inline 4x64 Field Operations for secp256k1 hot paths
// ===========================================================================
// These functions operate on uint64_t[4] field elements in 4x64 representation.
// They use MULX + ADCX/ADOX inline assembly for maximum throughput.
//
// Advantages over 5x52 compiler path:
//   - 4 limbs vs 5: lower register pressure (critical in hot loops)
//   - 20 MULX per mul vs 30: fewer multiplications
//   - 14 MULX per sqr vs 20: fewer multiplications  
//   - Hand-scheduled ADCX/ADOX dual carry chains
//   - No function call overhead (inline)
//
// Reduction constant: p = 2^256 - K, where K = 0x1000003D1
// All results are fully normalized (< p).
// ===========================================================================

#include <cstdint>
#include <cstring>

#if (defined(__x86_64__) || defined(_M_X64)) && defined(__ADX__) && defined(__BMI2__)

#if defined(__GNUC__) || defined(__clang__)
  #define FE4X64_FORCE_INLINE __attribute__((always_inline)) inline
#elif defined(_MSC_VER)
  #define FE4X64_FORCE_INLINE __forceinline
#else
  #define FE4X64_FORCE_INLINE inline
#endif

namespace secp256k1::fast::fe4x64 {

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
// p = FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
static constexpr std::uint64_t P0 = 0xFFFFFFFEFFFFFC2FULL;
static constexpr std::uint64_t P1 = 0xFFFFFFFFFFFFFFFFULL;
static constexpr std::uint64_t P2 = 0xFFFFFFFFFFFFFFFFULL;
static constexpr std::uint64_t P3 = 0xFFFFFFFFFFFFFFFFULL;
static constexpr std::uint64_t K  = 0x1000003D1ULL; // 2^256 - p

// ---------------------------------------------------------------------------
// fe4x64_mul: Modular multiplication (a * b) mod p
// ---------------------------------------------------------------------------
// Uses MULX + ADCX/ADOX for 256x256 -> 512-bit product, then secp256k1
// fast reduction (high limbs * K folded into low limbs).
// Output is strictly < p (branchless mod-p normalization).
//
// Register usage:
//   r8..r15 : 512-bit accumulator (all clobbered)
//   rdx     : MULX implicit source
//   rax,rbx : MULX hi/lo outputs
//   rcx     : zero register / carry
//   [a],[b],[r] : input/output pointers (compiler-chosen registers)
// ---------------------------------------------------------------------------
FE4X64_FORCE_INLINE
void mul(std::uint64_t* __restrict__ r,
         const std::uint64_t* __restrict__ a,
         const std::uint64_t* __restrict__ b) noexcept {
    __asm__ __volatile__(
        // ---- 256x256 -> 512 multiplication (MULX + ADCX/ADOX) ----
        "xorl %%r8d, %%r8d\n\t"
        "xorl %%r9d, %%r9d\n\t"
        "xorl %%r10d, %%r10d\n\t"
        "xorl %%r11d, %%r11d\n\t"
        "xorl %%r12d, %%r12d\n\t"
        "xorl %%r13d, %%r13d\n\t"
        "xorl %%r14d, %%r14d\n\t"
        "xorl %%r15d, %%r15d\n\t"

        // Row 0: a[0] * b[0..3]
        "movq (%[a]), %%rdx\n\t"
        "xorl %%ecx, %%ecx\n\t"
        "mulxq (%[b]), %%rax, %%rbx\n\t"
        "adcxq %%rax, %%r8\n\t"
        "adoxq %%rbx, %%r9\n\t"
        "mulxq 8(%[b]), %%rax, %%rbx\n\t"
        "adcxq %%rax, %%r9\n\t"
        "adoxq %%rbx, %%r10\n\t"
        "mulxq 16(%[b]), %%rax, %%rbx\n\t"
        "adcxq %%rax, %%r10\n\t"
        "adoxq %%rbx, %%r11\n\t"
        "mulxq 24(%[b]), %%rax, %%rbx\n\t"
        "adcxq %%rax, %%r11\n\t"
        "adoxq %%rbx, %%r12\n\t"
        "adcxq %%rcx, %%r12\n\t"
        "adoxq %%rcx, %%r13\n\t"

        // Row 1: a[1] * b[0..3]
        "movq 8(%[a]), %%rdx\n\t"
        "xorl %%ecx, %%ecx\n\t"
        "mulxq (%[b]), %%rax, %%rbx\n\t"
        "adcxq %%rax, %%r9\n\t"
        "adoxq %%rbx, %%r10\n\t"
        "mulxq 8(%[b]), %%rax, %%rbx\n\t"
        "adcxq %%rax, %%r10\n\t"
        "adoxq %%rbx, %%r11\n\t"
        "mulxq 16(%[b]), %%rax, %%rbx\n\t"
        "adcxq %%rax, %%r11\n\t"
        "adoxq %%rbx, %%r12\n\t"
        "mulxq 24(%[b]), %%rax, %%rbx\n\t"
        "adcxq %%rax, %%r12\n\t"
        "adoxq %%rbx, %%r13\n\t"
        "adcxq %%rcx, %%r13\n\t"
        "adoxq %%rcx, %%r14\n\t"

        // Row 2: a[2] * b[0..3]
        "movq 16(%[a]), %%rdx\n\t"
        "xorl %%ecx, %%ecx\n\t"
        "mulxq (%[b]), %%rax, %%rbx\n\t"
        "adcxq %%rax, %%r10\n\t"
        "adoxq %%rbx, %%r11\n\t"
        "mulxq 8(%[b]), %%rax, %%rbx\n\t"
        "adcxq %%rax, %%r11\n\t"
        "adoxq %%rbx, %%r12\n\t"
        "mulxq 16(%[b]), %%rax, %%rbx\n\t"
        "adcxq %%rax, %%r12\n\t"
        "adoxq %%rbx, %%r13\n\t"
        "mulxq 24(%[b]), %%rax, %%rbx\n\t"
        "adcxq %%rax, %%r13\n\t"
        "adoxq %%rbx, %%r14\n\t"
        "adcxq %%rcx, %%r14\n\t"
        "adoxq %%rcx, %%r15\n\t"

        // Row 3: a[3] * b[0..3]
        "movq 24(%[a]), %%rdx\n\t"
        "xorl %%ecx, %%ecx\n\t"
        "mulxq (%[b]), %%rax, %%rbx\n\t"
        "adcxq %%rax, %%r11\n\t"
        "adoxq %%rbx, %%r12\n\t"
        "mulxq 8(%[b]), %%rax, %%rbx\n\t"
        "adcxq %%rax, %%r12\n\t"
        "adoxq %%rbx, %%r13\n\t"
        "mulxq 16(%[b]), %%rax, %%rbx\n\t"
        "adcxq %%rax, %%r13\n\t"
        "adoxq %%rbx, %%r14\n\t"
        "mulxq 24(%[b]), %%rax, %%rbx\n\t"
        "adcxq %%rax, %%r14\n\t"
        "adoxq %%rbx, %%r15\n\t"
        "adcxq %%rcx, %%r15\n\t"

        // ---- Reduction (512 -> 256) ----
        // K = 0x1000003D1; high[i] * K added to low[i]
        "movabsq $0x1000003D1, %%rdx\n\t"
        "xorl %%ecx, %%ecx\n\t"

        "mulxq %%r12, %%rax, %%rbx\n\t"
        "addq %%rax, %%r8\n\t"
        "adcq %%rbx, %%r9\n\t"
        "adcq $0, %%r10\n\t"
        "adcq $0, %%r11\n\t"
        "adcq $0, %%rcx\n\t"

        "mulxq %%r13, %%rax, %%rbx\n\t"
        "addq %%rax, %%r9\n\t"
        "adcq %%rbx, %%r10\n\t"
        "adcq $0, %%r11\n\t"
        "adcq $0, %%rcx\n\t"

        "mulxq %%r14, %%rax, %%rbx\n\t"
        "addq %%rax, %%r10\n\t"
        "adcq %%rbx, %%r11\n\t"
        "adcq $0, %%rcx\n\t"

        "mulxq %%r15, %%rax, %%rbx\n\t"
        "addq %%rax, %%r11\n\t"
        "adcq %%rbx, %%rcx\n\t"

        // Pass 1: reduce overflow (~34 bits max)
        "mulxq %%rcx, %%rax, %%rbx\n\t"
        "xorl %%ecx, %%ecx\n\t"
        "addq %%rax, %%r8\n\t"
        "adcq %%rbx, %%r9\n\t"
        "adcq $0, %%r10\n\t"
        "adcq $0, %%r11\n\t"
        "adcq $0, %%rcx\n\t"

        // Pass 2: reduce final 1-bit overflow (branchless)
        "movabsq $0x1000003D1, %%rax\n\t"
        "andq %%rcx, %%rax\n\t"
        "addq %%rax, %%r8\n\t"
        "adcq $0, %%r9\n\t"
        "adcq $0, %%r10\n\t"
        "adcq $0, %%r11\n\t"

        // Branchless mod-p: if result >= p, subtract p
        "movabsq $0x1000003D1, %%rcx\n\t"
        "movq %%r8, %%rax\n\t"
        "addq %%rcx, %%rax\n\t"
        "movq %%r9, %%rbx\n\t"
        "adcq $0, %%rbx\n\t"
        "movq %%r10, %%r12\n\t"
        "adcq $0, %%r12\n\t"
        "movq %%r11, %%r13\n\t"
        "adcq $0, %%r13\n\t"
        "cmovcq %%rax, %%r8\n\t"
        "cmovcq %%rbx, %%r9\n\t"
        "cmovcq %%r12, %%r10\n\t"
        "cmovcq %%r13, %%r11\n\t"

        // Store result (load output pointer from stack)
        "movq %[r_ptr], %%rax\n\t"
        "movq %%r8, (%%rax)\n\t"
        "movq %%r9, 8(%%rax)\n\t"
        "movq %%r10, 16(%%rax)\n\t"
        "movq %%r11, 24(%%rax)\n\t"


        : /* no output operands */
        : [a] "r" (a), [b] "r" (b), [r_ptr] "m" (r)
        : "rax", "rbx", "rcx", "rdx",
          "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
          "cc", "memory"
    );
}

// ---------------------------------------------------------------------------
// fe4x64_sqr: Modular squaring (a^2) mod p
// ---------------------------------------------------------------------------
// Optimized: 6 cross-products (doubled) + 4 squares = 10 MULX + 4 MULX,
// then secp256k1 fast reduction (5 MULX).
// Total: ~19 MULX (vs ~20 for 5x52 sqr via __int128)
// ---------------------------------------------------------------------------
FE4X64_FORCE_INLINE
void sqr(std::uint64_t* __restrict__ r,
         const std::uint64_t* __restrict__ a) noexcept {
    __asm__ __volatile__(
        // ---- Cross products ----
        "xorl %%r8d, %%r8d\n\t"
        "xorl %%r9d, %%r9d\n\t"
        "xorl %%r10d, %%r10d\n\t"
        "xorl %%r11d, %%r11d\n\t"
        "xorl %%r12d, %%r12d\n\t"
        "xorl %%r13d, %%r13d\n\t"
        "xorl %%r14d, %%r14d\n\t"
        "xorl %%r15d, %%r15d\n\t"

        // Pass A: a[0] * (a[1], a[2], a[3])
        "movq (%[a]), %%rdx\n\t"
        "xorl %%ecx, %%ecx\n\t"
        "mulxq 8(%[a]), %%rax, %%rbx\n\t"
        "adcxq %%rax, %%r9\n\t"
        "adoxq %%rbx, %%r10\n\t"
        "mulxq 16(%[a]), %%rax, %%rbx\n\t"
        "adcxq %%rax, %%r10\n\t"
        "adoxq %%rbx, %%r11\n\t"
        "mulxq 24(%[a]), %%rax, %%rbx\n\t"
        "adcxq %%rax, %%r11\n\t"
        "adoxq %%rbx, %%r12\n\t"
        "adcxq %%rcx, %%r12\n\t"
        "adoxq %%rcx, %%r13\n\t"

        // Pass B: a[1] * (a[2], a[3])
        "movq 8(%[a]), %%rdx\n\t"
        "xorl %%ecx, %%ecx\n\t"
        "mulxq 16(%[a]), %%rax, %%rbx\n\t"
        "adcxq %%rax, %%r11\n\t"
        "adoxq %%rbx, %%r12\n\t"
        "mulxq 24(%[a]), %%rax, %%rbx\n\t"
        "adcxq %%rax, %%r12\n\t"
        "adoxq %%rbx, %%r13\n\t"
        "adcxq %%rcx, %%r13\n\t"
        "adoxq %%rcx, %%r14\n\t"

        // Pass C: a[2] * a[3]
        "movq 16(%[a]), %%rdx\n\t"
        "xorl %%ecx, %%ecx\n\t"
        "mulxq 24(%[a]), %%rax, %%rbx\n\t"
        "adcxq %%rax, %%r13\n\t"
        "adoxq %%rbx, %%r14\n\t"
        "adcxq %%rcx, %%r14\n\t"
        "adoxq %%rcx, %%r15\n\t"

        // ---- Double the cross products ----
        "addq %%r9, %%r9\n\t"
        "adcq %%r10, %%r10\n\t"
        "adcq %%r11, %%r11\n\t"
        "adcq %%r12, %%r12\n\t"
        "adcq %%r13, %%r13\n\t"
        "adcq %%r14, %%r14\n\t"
        "adcq %%r15, %%r15\n\t"

        // ---- Add squares (ADCX chain) ----
        "xorl %%ecx, %%ecx\n\t"
        "movq (%[a]), %%rdx\n\t"
        "mulxq %%rdx, %%rax, %%rbx\n\t"
        "adcxq %%rax, %%r8\n\t"
        "adcxq %%rbx, %%r9\n\t"
        "movq 8(%[a]), %%rdx\n\t"
        "mulxq %%rdx, %%rax, %%rbx\n\t"
        "adcxq %%rax, %%r10\n\t"
        "adcxq %%rbx, %%r11\n\t"
        "movq 16(%[a]), %%rdx\n\t"
        "mulxq %%rdx, %%rax, %%rbx\n\t"
        "adcxq %%rax, %%r12\n\t"
        "adcxq %%rbx, %%r13\n\t"
        "movq 24(%[a]), %%rdx\n\t"
        "mulxq %%rdx, %%rax, %%rbx\n\t"
        "adcxq %%rax, %%r14\n\t"
        "adcxq %%rbx, %%r15\n\t"

        // ---- Reduction (512 -> 256) ----
        "movabsq $0x1000003D1, %%rdx\n\t"
        "xorl %%ecx, %%ecx\n\t"

        "mulxq %%r12, %%rax, %%rbx\n\t"
        "addq %%rax, %%r8\n\t"
        "adcq %%rbx, %%r9\n\t"
        "adcq $0, %%r10\n\t"
        "adcq $0, %%r11\n\t"
        "adcq $0, %%rcx\n\t"

        "mulxq %%r13, %%rax, %%rbx\n\t"
        "addq %%rax, %%r9\n\t"
        "adcq %%rbx, %%r10\n\t"
        "adcq $0, %%r11\n\t"
        "adcq $0, %%rcx\n\t"

        "mulxq %%r14, %%rax, %%rbx\n\t"
        "addq %%rax, %%r10\n\t"
        "adcq %%rbx, %%r11\n\t"
        "adcq $0, %%rcx\n\t"

        "mulxq %%r15, %%rax, %%rbx\n\t"
        "addq %%rax, %%r11\n\t"
        "adcq %%rbx, %%rcx\n\t"

        // Pass 1: reduce overflow
        "mulxq %%rcx, %%rax, %%rbx\n\t"
        "xorl %%ecx, %%ecx\n\t"
        "addq %%rax, %%r8\n\t"
        "adcq %%rbx, %%r9\n\t"
        "adcq $0, %%r10\n\t"
        "adcq $0, %%r11\n\t"
        "adcq $0, %%rcx\n\t"

        // Pass 2: branchless final reduction
        "movabsq $0x1000003D1, %%rax\n\t"
        "andq %%rcx, %%rax\n\t"
        "addq %%rax, %%r8\n\t"
        "adcq $0, %%r9\n\t"
        "adcq $0, %%r10\n\t"
        "adcq $0, %%r11\n\t"

        // Branchless mod-p
        "movabsq $0x1000003D1, %%rcx\n\t"
        "movq %%r8, %%rax\n\t"
        "addq %%rcx, %%rax\n\t"
        "movq %%r9, %%rbx\n\t"
        "adcq $0, %%rbx\n\t"
        "movq %%r10, %%r12\n\t"
        "adcq $0, %%r12\n\t"
        "movq %%r11, %%r13\n\t"
        "adcq $0, %%r13\n\t"
        "cmovcq %%rax, %%r8\n\t"
        "cmovcq %%rbx, %%r9\n\t"
        "cmovcq %%r12, %%r10\n\t"
        "cmovcq %%r13, %%r11\n\t"

        // Store (load output pointer from stack)
        "movq %[r_ptr], %%rax\n\t"
        "movq %%r8, (%%rax)\n\t"
        "movq %%r9, 8(%%rax)\n\t"
        "movq %%r10, 16(%%rax)\n\t"
        "movq %%r11, 24(%%rax)\n\t"

        : /* no output operands */
        : [a] "r" (a), [r_ptr] "m" (r)
        : "rax", "rbx", "rcx", "rdx",
          "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
          "cc", "memory"
    );
}

// ---------------------------------------------------------------------------
// fe4x64_add: Modular addition (a + b) mod p  (branchless)
// ---------------------------------------------------------------------------
FE4X64_FORCE_INLINE
void add(std::uint64_t* __restrict__ r,
         const std::uint64_t* __restrict__ a,
         const std::uint64_t* __restrict__ b) noexcept {
    // Sum S = a + b
    unsigned __int128 s = (unsigned __int128)a[0] + b[0];
    std::uint64_t r0 = (std::uint64_t)s; s >>= 64;
    s += (unsigned __int128)a[1] + b[1];
    std::uint64_t r1 = (std::uint64_t)s; s >>= 64;
    s += (unsigned __int128)a[2] + b[2];
    std::uint64_t r2 = (std::uint64_t)s; s >>= 64;
    s += (unsigned __int128)a[3] + b[3];
    std::uint64_t r3 = (std::uint64_t)s;
    std::uint64_t carry_s = (std::uint64_t)(s >> 64);

    // T = S + K (to check if S >= p)
    unsigned __int128 t = (unsigned __int128)r0 + K;
    std::uint64_t t0 = (std::uint64_t)t; t >>= 64;
    t += r1;
    std::uint64_t t1 = (std::uint64_t)t; t >>= 64;
    t += r2;
    std::uint64_t t2 = (std::uint64_t)t; t >>= 64;
    t += r3;
    std::uint64_t t3 = (std::uint64_t)t;
    std::uint64_t carry_t = (std::uint64_t)(t >> 64);

    // If S overflowed (carry_s) OR T overflowed (carry_t), use T
    std::uint64_t use_t = carry_s | carry_t;
    std::uint64_t mask = (std::uint64_t)0 - use_t; // all-1s if use_t, all-0s if not

    r[0] = (r0 & ~mask) | (t0 & mask);
    r[1] = (r1 & ~mask) | (t1 & mask);
    r[2] = (r2 & ~mask) | (t2 & mask);
    r[3] = (r3 & ~mask) | (t3 & mask);
}

// ---------------------------------------------------------------------------
// fe4x64_sub: Modular subtraction (a - b) mod p  (branchless)
// ---------------------------------------------------------------------------
FE4X64_FORCE_INLINE
void sub(std::uint64_t* __restrict__ r,
         const std::uint64_t* __restrict__ a,
         const std::uint64_t* __restrict__ b) noexcept {
    // D = a - b (with borrow tracking)
    unsigned __int128 d = (unsigned __int128)a[0] - b[0];
    std::uint64_t r0 = (std::uint64_t)d;
    std::uint64_t borrow = (d >> 127) & 1;  // 1 if borrow

    d = (unsigned __int128)a[1] - b[1] - borrow;
    std::uint64_t r1 = (std::uint64_t)d;
    borrow = (d >> 127) & 1;

    d = (unsigned __int128)a[2] - b[2] - borrow;
    std::uint64_t r2 = (std::uint64_t)d;
    borrow = (d >> 127) & 1;

    d = (unsigned __int128)a[3] - b[3] - borrow;
    std::uint64_t r3 = (std::uint64_t)d;
    borrow = (d >> 127) & 1;

    // If borrow, add p: result += p = (2^256 - K) means result -= K
    // Wait: a - b < 0 means we need a - b + p = a - b + 2^256 - K.
    // The 2^256 is implicit (we got the wraparound). So subtract K.
    std::uint64_t mask = (std::uint64_t)0 - borrow; // all-1s if borrow
    std::uint64_t corr = K & mask;

    d = (unsigned __int128)r0 - corr;
    r[0] = (std::uint64_t)d;
    borrow = (d >> 127) & 1;

    d = (unsigned __int128)r1 - borrow;
    r[1] = (std::uint64_t)d;
    borrow = (d >> 127) & 1;

    d = (unsigned __int128)r2 - borrow;
    r[2] = (std::uint64_t)d;
    borrow = (d >> 127) & 1;

    r[3] = r3 - borrow;
}

// ---------------------------------------------------------------------------
// fe4x64_negate: Modular negation (-a) mod p  (branchless)
// ---------------------------------------------------------------------------
FE4X64_FORCE_INLINE
void negate(std::uint64_t* __restrict__ r,
            const std::uint64_t* __restrict__ a) noexcept {
    // -a mod p = p - a (if a != 0), 0 (if a == 0)
    // Branchless: compute p - a, then mask to 0 if a was 0
    unsigned __int128 d = (unsigned __int128)P0 - a[0];
    r[0] = (std::uint64_t)d;
    std::uint64_t borrow = (d >> 127) & 1;

    d = (unsigned __int128)P1 - a[1] - borrow;
    r[1] = (std::uint64_t)d;
    borrow = (d >> 127) & 1;

    d = (unsigned __int128)P2 - a[2] - borrow;
    r[2] = (std::uint64_t)d;
    borrow = (d >> 127) & 1;

    r[3] = P3 - a[3] - borrow;

    // If a == 0, result should also be 0
    std::uint64_t nonzero = a[0] | a[1] | a[2] | a[3];
    std::uint64_t mask = (std::uint64_t)0 - (std::uint64_t)(nonzero != 0);
    r[0] &= mask;
    r[1] &= mask;
    r[2] &= mask;
    r[3] &= mask;
}

// ---------------------------------------------------------------------------
// fe4x64_mul_int: Multiply by small integer (a * k) mod p  (branchless)
// ---------------------------------------------------------------------------
FE4X64_FORCE_INLINE
void mul_int(std::uint64_t* __restrict__ r,
             const std::uint64_t* __restrict__ a,
             std::uint32_t k) noexcept {
    using u128 = unsigned __int128;
    u128 c = (u128)a[0] * k;
    r[0] = (std::uint64_t)c; c >>= 64;
    c += (u128)a[1] * k;
    r[1] = (std::uint64_t)c; c >>= 64;
    c += (u128)a[2] * k;
    r[2] = (std::uint64_t)c; c >>= 64;
    c += (u128)a[3] * k;
    r[3] = (std::uint64_t)c;
    std::uint64_t overflow = (std::uint64_t)(c >> 64);

    // Reduce overflow: overflow * K back into limb 0
    if (overflow) {
        u128 f = (u128)overflow * K + r[0];
        r[0] = (std::uint64_t)f;
        std::uint64_t carry = (std::uint64_t)(f >> 64);
        r[1] += carry;
        carry = (r[1] < carry) ? 1ULL : 0ULL;
        r[2] += carry;
        carry = (r[2] < carry) ? 1ULL : 0ULL;
        r[3] += carry;
    }
}

// ---------------------------------------------------------------------------
// fe4x64_half: Modular halving (a / 2) mod p  (branchless)
// ---------------------------------------------------------------------------
// If a is even: shift right by 1.
// If a is odd:  (a + p) / 2  (p is odd, so a+p is even).
FE4X64_FORCE_INLINE
void half(std::uint64_t* __restrict__ r,
          const std::uint64_t* __restrict__ a) noexcept {
    // Conditionally add p if odd
    std::uint64_t mask = (std::uint64_t)0 - (a[0] & 1); // all-1s if odd
    unsigned __int128 s = (unsigned __int128)a[0] + (P0 & mask);
    std::uint64_t t0 = (std::uint64_t)s; s >>= 64;
    s += (unsigned __int128)a[1] + (P1 & mask);
    std::uint64_t t1 = (std::uint64_t)s; s >>= 64;
    s += (unsigned __int128)a[2] + (P2 & mask);
    std::uint64_t t2 = (std::uint64_t)s; s >>= 64;
    s += (unsigned __int128)a[3] + (P3 & mask);
    std::uint64_t t3 = (std::uint64_t)s;
    std::uint64_t carry = (std::uint64_t)(s >> 64);

    // Shift right by 1 (divides the even result by 2)
    r[0] = (t0 >> 1) | (t1 << 63);
    r[1] = (t1 >> 1) | (t2 << 63);
    r[2] = (t2 >> 1) | (t3 << 63);
    r[3] = (t3 >> 1) | (carry << 63);
}

// ---------------------------------------------------------------------------
// fe4x64_copy: Copy field element
// ---------------------------------------------------------------------------
FE4X64_FORCE_INLINE
void copy(std::uint64_t* __restrict__ dst,
          const std::uint64_t* __restrict__ src) noexcept {
    dst[0] = src[0];
    dst[1] = src[1];
    dst[2] = src[2];
    dst[3] = src[3];
}

// ---------------------------------------------------------------------------
// fe4x64_cmov: Conditional move (branchless)
// If flag != 0: dst = src. If flag == 0: dst unchanged.
// ---------------------------------------------------------------------------
FE4X64_FORCE_INLINE
void cmov(std::uint64_t* __restrict__ dst,
          const std::uint64_t* __restrict__ src,
          std::uint64_t flag) noexcept {
    std::uint64_t mask = (std::uint64_t)0 - (std::uint64_t)(flag != 0);
    dst[0] = (dst[0] & ~mask) | (src[0] & mask);
    dst[1] = (dst[1] & ~mask) | (src[1] & mask);
    dst[2] = (dst[2] & ~mask) | (src[2] & mask);
    dst[3] = (dst[3] & ~mask) | (src[3] & mask);
}

// ---------------------------------------------------------------------------
// fe4x64_is_zero: Check if field element is zero (constant-time)
// ---------------------------------------------------------------------------
FE4X64_FORCE_INLINE
bool is_zero(const std::uint64_t* a) noexcept {
    return (a[0] | a[1] | a[2] | a[3]) == 0;
}

} // namespace secp256k1::fast::fe4x64

#endif // x86_64 + ADX + BMI2
