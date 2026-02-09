/*
 * RISC-V 64-bit Assembly Optimizations for secp256k1 Field Operations
 * 
 * Target: RV64GC (RISC-V 64-bit with General + Compressed + Multiply extensions)
 * Optional: RVV (Vector Extension) for batch operations
 * Tested on: StarFive JH7110 (Milk-V Mars)
 * 
 * Field Element: 256-bit = 4 × 64-bit limbs (little-endian)
 * Prime: 2^256 - 2^32 - 977 (secp256k1 field prime)
 * 
 * RISC-V Instructions Used:
 *   MUL    rd, rs1, rs2    - 64×64 → 64 (low bits)
 *   MULH   rd, rs1, rs2    - 64×64 → 64 (high bits, signed×signed)
 *   MULHU  rd, rs1, rs2    - 64×64 → 64 (high bits, unsigned×unsigned)
 *   ADD    rd, rs1, rs2    - 64-bit addition
 *   SLTU   rd, rs1, rs2    - Set if less than unsigned (for carry detection)
 * 
 * Build Options (CMake):
 *   -DSECP256K1_RISCV_USE_VECTOR=ON    - Enable RVV batch operations (4-8× speedup)
 *   -DSECP256K1_RISCV_FAST_REDUCTION=ON - Fast modular reduction in assembly
 *   -DSECP256K1_RISCV_USE_PREFETCH=ON  - Enable cache prefetch hints
 */

#ifndef SECP256K1_RISCV_FAST_REDUCTION
#define SECP256K1_RISCV_FAST_REDUCTION 1
#endif

    .section .text
    .align 2

/*
 * field_mul_asm_riscv64
 * 
 * Multiply two field elements: r = a × b (mod p)
 * 
 * Parameters:
 *   a0 = result pointer (uint64_t r[4])
 *   a1 = operand a pointer (uint64_t a[4])
 *   a2 = operand b pointer (uint64_t b[4])
 * 
 * Clobbers: t0-t6, a3-a7
 * 
 * Algorithm: Comba multiplication (product scanning)
 *   - Compute 4×4 → 8 limbs product
 *   - Reduce mod p using fast reduction
 */
    .globl field_mul_asm_riscv64
    .type field_mul_asm_riscv64, @function
field_mul_asm_riscv64:
    # RISC-V ABI: Save callee-saved registers used in this function
    # Used registers: s0-s11 (12 registers)
    # Stack layout: [s0-s11: 12*8 = 96 bytes] + padding = 112 bytes (aligned)
    addi    sp, sp, -112
    sd      s0, 0(sp)
    sd      s1, 8(sp)
    sd      s2, 16(sp)
    sd      s3, 24(sp)
    sd      s4, 32(sp)
    sd      s5, 40(sp)
    sd      s6, 48(sp)
    sd      s7, 56(sp)
    sd      s8, 64(sp)
    sd      s9, 72(sp)
    sd      s10, 80(sp)
    sd      s11, 88(sp)

    # Save result pointer in s11 (callee-saved)
    mv      s11, a0

    # Load operand a into s0-s3 (use callee-saved to avoid conflicts)
    ld      s0, 0(a1)       # a[0]
    ld      s1, 8(a1)       # a[1]
    ld      s2, 16(a1)      # a[2]
    ld      s3, 24(a1)      # a[3]

    # Load operand b into s4-s7
    ld      s4, 0(a2)
    ld      s5, 8(a2)
    ld      s6, 16(a2)
    ld      s7, 24(a2)

    # -------------------------------------------------------------------------
    # Comba Multiplication (512-bit result in t0-t6, s8)
    # Uses 3-register accumulator to prevent overflow:
    #   Current limb: t{i}
    #   Carry 1:      s9
    #   Carry 2:      s10
    # -------------------------------------------------------------------------

    # Column 0: r0 = a0*b0
    mul     t0, s0, s4
    mulhu   s9, s0, s4
    li      s10, 0

    # Column 1: r1 = a0*b1 + a1*b0 + c
    mv      t1, s9
    mv      s9, s10
    li      s10, 0

    # Term 0: a0*b1
    mul     a1, s0, s5
    mulhu   a2, s0, s5
    add     t1, t1, a1
    sltu    a3, t1, a1
    add     s9, s9, a2
    sltu    a4, s9, a2
    add     s9, s9, a3
    sltu    a5, s9, a3
    add     s10, s10, a4
    add     s10, s10, a5

    # Term 1: a1*b0
    mul     a1, s1, s4
    mulhu   a2, s1, s4
    add     t1, t1, a1
    sltu    a3, t1, a1
    add     s9, s9, a2
    sltu    a4, s9, a2
    add     s9, s9, a3
    sltu    a5, s9, a3
    add     s10, s10, a4
    add     s10, s10, a5

    # Column 2: r2 = a0*b2 + a1*b1 + a2*b0 + c
    mv      t2, s9
    mv      s9, s10
    li      s10, 0

    # Term 0: a0*b2
    mul     a1, s0, s6
    mulhu   a2, s0, s6
    add     t2, t2, a1
    sltu    a3, t2, a1
    add     s9, s9, a2
    sltu    a4, s9, a2
    add     s9, s9, a3
    sltu    a5, s9, a3
    add     s10, s10, a4
    add     s10, s10, a5

    # Term 1: a1*b1
    mul     a1, s1, s5
    mulhu   a2, s1, s5
    add     t2, t2, a1
    sltu    a3, t2, a1
    add     s9, s9, a2
    sltu    a4, s9, a2
    add     s9, s9, a3
    sltu    a5, s9, a3
    add     s10, s10, a4
    add     s10, s10, a5

    # Term 2: a2*b0
    mul     a1, s2, s4
    mulhu   a2, s2, s4
    add     t2, t2, a1
    sltu    a3, t2, a1
    add     s9, s9, a2
    sltu    a4, s9, a2
    add     s9, s9, a3
    sltu    a5, s9, a3
    add     s10, s10, a4
    add     s10, s10, a5

    # Column 3: r3 = a0*b3 + a1*b2 + a2*b1 + a3*b0 + c
    mv      t3, s9
    mv      s9, s10
    li      s10, 0

    # Term 0: a0*b3
    mul     a1, s0, s7
    mulhu   a2, s0, s7
    add     t3, t3, a1
    sltu    a3, t3, a1
    add     s9, s9, a2
    sltu    a4, s9, a2
    add     s9, s9, a3
    sltu    a5, s9, a3
    add     s10, s10, a4
    add     s10, s10, a5

    # Term 1: a1*b2
    mul     a1, s1, s6
    mulhu   a2, s1, s6
    add     t3, t3, a1
    sltu    a3, t3, a1
    add     s9, s9, a2
    sltu    a4, s9, a2
    add     s9, s9, a3
    sltu    a5, s9, a3
    add     s10, s10, a4
    add     s10, s10, a5

    # Term 2: a2*b1
    mul     a1, s2, s5
    mulhu   a2, s2, s5
    add     t3, t3, a1
    sltu    a3, t3, a1
    add     s9, s9, a2
    sltu    a4, s9, a2
    add     s9, s9, a3
    sltu    a5, s9, a3
    add     s10, s10, a4
    add     s10, s10, a5

    # Term 3: a3*b0
    mul     a1, s3, s4
    mulhu   a2, s3, s4
    add     t3, t3, a1
    sltu    a3, t3, a1
    add     s9, s9, a2
    sltu    a4, s9, a2
    add     s9, s9, a3
    sltu    a5, s9, a3
    add     s10, s10, a4
    add     s10, s10, a5

    # Column 4: r4 = a1*b3 + a2*b2 + a3*b1 + c
    mv      t4, s9
    mv      s9, s10
    li      s10, 0

    # Term 0: a1*b3
    mul     a1, s1, s7
    mulhu   a2, s1, s7
    add     t4, t4, a1
    sltu    a3, t4, a1
    add     s9, s9, a2
    sltu    a4, s9, a2
    add     s9, s9, a3
    sltu    a5, s9, a3
    add     s10, s10, a4
    add     s10, s10, a5

    # Term 1: a2*b2
    mul     a1, s2, s6
    mulhu   a2, s2, s6
    add     t4, t4, a1
    sltu    a3, t4, a1
    add     s9, s9, a2
    sltu    a4, s9, a2
    add     s9, s9, a3
    sltu    a5, s9, a3
    add     s10, s10, a4
    add     s10, s10, a5

    # Term 2: a3*b1
    mul     a1, s3, s5
    mulhu   a2, s3, s5
    add     t4, t4, a1
    sltu    a3, t4, a1
    add     s9, s9, a2
    sltu    a4, s9, a2
    add     s9, s9, a3
    sltu    a5, s9, a3
    add     s10, s10, a4
    add     s10, s10, a5

    # Column 5: r5 = a2*b3 + a3*b2 + c
    mv      t5, s9
    mv      s9, s10
    li      s10, 0

    # Term 0: a2*b3
    mul     a1, s2, s7
    mulhu   a2, s2, s7
    add     t5, t5, a1
    sltu    a3, t5, a1
    add     s9, s9, a2
    sltu    a4, s9, a2
    add     s9, s9, a3
    sltu    a5, s9, a3
    add     s10, s10, a4
    add     s10, s10, a5

    # Term 1: a3*b2
    mul     a1, s3, s6
    mulhu   a2, s3, s6
    add     t5, t5, a1
    sltu    a3, t5, a1
    add     s9, s9, a2
    sltu    a4, s9, a2
    add     s9, s9, a3
    sltu    a5, s9, a3
    add     s10, s10, a4
    add     s10, s10, a5

    # Column 6: r6 = a3*b3 + c
    mv      t6, s9
    mv      s9, s10
    li      s10, 0

    # Term 0: a3*b3
    mul     a1, s3, s7
    mulhu   a2, s3, s7
    add     t6, t6, a1
    sltu    a3, t6, a1
    add     s9, s9, a2
    sltu    a4, s9, a2
    add     s9, s9, a3
    sltu    a5, s9, a3
    add     s10, s10, a4
    add     s10, s10, a5

    # Column 7: r7 = carry
    # s10 should be 0 because result fits in 512 bits
    mv      s8, s9

    # -------------------------------------------------------------------------
    # Fast Modular Reduction (Operates on t0-t6, s8)
    # Result stored in s0-s3 (reusing A registers, final result)
    # -------------------------------------------------------------------------

    # s10 = 977
    li      s10, 977
    # s9 = overflow accumulator (reused)
    li      s9, 0

    # --- Process r4 (t4) ---
    # 1. Mul by 977
    mul     a1, t4, s10
    mulhu   a2, t4, s10
    add     t0, t0, a1
    sltu    a3, t0, a1
    add     t1, t1, a3
    sltu    a4, t1, a3
    add     t1, t1, a2
    sltu    a3, t1, a2

    add     t2, t2, a4
    sltu    a4, t2, a4
    add     t2, t2, a3
    sltu    a3, t2, a3

    add     t3, t3, a4
    sltu    a4, t3, a4
    add     t3, t3, a3
    sltu    a3, t3, a3
    add     s9, s9, a4
    add     s9, s9, a3

    # 2. Shift << 32
    slli    a1, t4, 32
    srli    a2, t4, 32
    add     t0, t0, a1
    sltu    a3, t0, a1
    add     t1, t1, a2
    sltu    a4, t1, a2
    add     t1, t1, a3
    sltu    a3, t1, a3

    add     t2, t2, a4
    sltu    a4, t2, a4
    add     t2, t2, a3
    sltu    a3, t2, a3

    add     t3, t3, a4
    sltu    a4, t3, a4
    add     t3, t3, a3
    sltu    a3, t3, a3
    add     s9, s9, a4
    add     s9, s9, a3

    # --- Process r5 (t5) ---
    # 1. Mul by 977
    mul     a1, t5, s10
    mulhu   a2, t5, s10
    add     t1, t1, a1
    sltu    a3, t1, a1
    add     t2, t2, a3
    sltu    a4, t2, a3
    add     t2, t2, a2
    sltu    a3, t2, a2

    add     t3, t3, a4
    sltu    a4, t3, a4
    add     t3, t3, a3
    sltu    a3, t3, a3
    add     s9, s9, a4
    add     s9, s9, a3

    # 2. Shift << 32
    slli    a1, t5, 32
    srli    a2, t5, 32
    add     t1, t1, a1
    sltu    a3, t1, a1
    add     t2, t2, a2
    sltu    a4, t2, a2
    add     t2, t2, a3
    sltu    a3, t2, a3

    add     t3, t3, a4
    sltu    a4, t3, a4
    add     t3, t3, a3
    sltu    a3, t3, a3
    add     s9, s9, a4
    add     s9, s9, a3

    # --- Process r6 (t6) ---
    # 1. Mul by 977
    mul     a1, t6, s10
    mulhu   a2, t6, s10
    add     t2, t2, a1
    sltu    a3, t2, a1
    add     t3, t3, a3
    sltu    a4, t3, a3
    add     t3, t3, a2
    sltu    a3, t3, a2
    add     s9, s9, a4
    add     s9, s9, a3

    # 2. Shift << 32
    slli    a1, t6, 32
    srli    a2, t6, 32
    add     t2, t2, a1
    sltu    a3, t2, a1
    add     t3, t3, a2
    sltu    a4, t3, a2
    add     t3, t3, a3
    sltu    a3, t3, a3
    add     s9, s9, a4
    add     s9, s9, a3

    # --- Process r7 (s8) ---
    # 1. Mul by 977
    mul     a1, s8, s10
    mulhu   a2, s8, s10
    add     t3, t3, a1
    sltu    a3, t3, a1
    add     s9, s9, a2
    add     s9, s9, a3

    # 2. Shift << 32
    slli    a1, s8, 32
    srli    a2, s8, 32
    add     t3, t3, a1
    sltu    a3, t3, a1
    add     s9, s9, a2
    add     s9, s9, a3

    # Loop for overflow reduction (s9)
    # Reuse t4 for overflow processing to avoid clobbering reduced limbs t0-t3
.Lreduce_loop:
    beqz    s9, .Lfinal_check

    mv      t4, s9
    li      s9, 0

    # Mul by 977
    mul     a1, t4, s10
    mulhu   a2, t4, s10
    add     t0, t0, a1
    sltu    a3, t0, a1
    add     t1, t1, a3
    sltu    a4, t1, a3
    add     t1, t1, a2
    sltu    a3, t1, a2

    add     t2, t2, a4
    sltu    a4, t2, a4
    add     t2, t2, a3
    sltu    a3, t2, a3

    add     t3, t3, a4
    sltu    a4, t3, a4
    add     t3, t3, a3
    sltu    a3, t3, a3
    add     s9, s9, a4
    add     s9, s9, a3

    # Shift << 32
    slli    a1, t4, 32
    srli    a2, t4, 32
    add     t0, t0, a1
    sltu    a3, t0, a1
    add     t1, t1, a2
    sltu    a4, t1, a2
    add     t1, t1, a3
    sltu    a3, t1, a3

    add     t2, t2, a4
    sltu    a4, t2, a4
    add     t2, t2, a3
    sltu    a3, t2, a3

    add     t3, t3, a4
    sltu    a4, t3, a4
    add     t3, t3, a3
    sltu    a3, t3, a3
    add     s9, s9, a4
    add     s9, s9, a3

    j       .Lreduce_loop

.Lfinal_check:
    # Final comparison magnitude check: result >= P
    # Same logic as before: Add K, check carry.
    # Result is in t0, t1, t2, t3

    li      a1, 0x3D1
    li      a2, 1
    slli    a2, a2, 32
    or      a1, a1, a2      # K

    add     a2, t0, a1      # r0 + K
    sltu    a3, a2, t0      # c (check vs original)

    add     a4, t1, a3
    sltu    a3, a4, t1

    add     a5, t2, a3
    sltu    a3, a5, t2

    add     a6, t3, a3
    sltu    a7, a6, t3      # a7 = carry out (needs reduction)

    # Condition mask: -a7
    neg     a7, a7

    # Select: if carry, result = r+K (a2, a4, a5, a6)
    # otherwise result = r (t0, t1, t2, t3)

    xor     s0, t0, a2
    and     s0, s0, a7
    xor     t0, t0, s0

    xor     s0, t1, a4
    and     s0, s0, a7
    xor     t1, t1, s0

    xor     s0, t2, a5
    and     s0, s0, a7
    xor     t2, t2, s0

    xor     s0, t3, a6
    and     s0, s0, a7
    xor     t3, t3, s0

    # Store result to (s11)
    sd      t0, 0(s11)
    sd      t1, 8(s11)
    sd      t2, 16(s11)
    sd      t3, 24(s11)

    # Restore registers
    ld      s0, 0(sp)
    ld      s1, 8(sp)
    ld      s2, 16(sp)
    ld      s3, 24(sp)
    ld      s4, 32(sp)
    ld      s5, 40(sp)
    ld      s6, 48(sp)
    ld      s7, 56(sp)
    ld      s8, 64(sp)
    ld      s9, 72(sp)
    ld      s10, 80(sp)
    ld      s11, 88(sp)
    addi    sp, sp, 112
    ret
    .size field_mul_asm_riscv64, .-field_mul_asm_riscv64
.Lfield_mul_asm_riscv64_end:

/*
 * field_sub_asm_riscv64
 * 
 * Subtract two field elements: r = a - b (mod p)
 */
    .globl field_sub_asm_riscv64
    .type field_sub_asm_riscv64, @function
field_sub_asm_riscv64:
    # Save callee-saved registers
    addi    sp, sp, -80
    sd      s0, 8(sp)
    sd      s1, 16(sp)
    sd      s2, 24(sp)
    sd      s3, 32(sp)
    sd      s4, 40(sp)
    sd      s5, 48(sp)
    sd      s6, 56(sp)
    sd      s7, 64(sp)

    # Load operands
    ld      a3, 0(a1)       # a[0]
    ld      a4, 8(a1)       # a[1]
    ld      a5, 16(a1)      # a[2]
    ld      a6, 24(a1)      # a[3]
    
    ld      t0, 0(a2)       # b[0]
    ld      t1, 8(a2)       # b[1]
    ld      t2, 16(a2)      # b[2]
    ld      t3, 24(a2)       # b[3]

    # Subtract with borrow chain: result = a - b
    # limb 0
    sltu    t5, a3, t0      # borrow0 = (a[0] < b[0])
    sub     t4, a3, t0      # result[0] = a[0] - b[0]

    # limb 1: a[1] - b[1] - borrow0
    mv      t6, a4
    sub     t6, t6, t5      # a[1] - borrow0
    sltu    s0, a4, t5      # underflow from borrow subtraction
    sltu    s1, t6, t1      # will borrow from b[1] subtraction
    sub     t6, t6, t1      # final result[1]
    or      t5, s0, s1      # combined borrow

    # limb 2: a[2] - b[2] - borrow1
    mv      s2, a5
    sub     s2, s2, t5
    sltu    s3, a5, t5
    sltu    s4, s2, t2
    sub     s2, s2, t2
    or      t5, s3, s4      # combined borrow

    # limb 3: a[3] - b[3] - borrow2
    mv      s5, a6
    sub     s5, s5, t5
    sltu    s6, a6, t5
    sltu    s7, s5, t3
    sub     s5, s5, t3
    or      t5, s6, s7      # final borrow
    
    # If borrow (t5 != 0), result underflowed
    # Need to add P, which is equivalent to subtracting K
    # Compute temp = result - K
    li      a3, 0x3D1
    li      s0, 1
    slli    s0, s0, 32
    or      a3, a3, s0      # a3 = K = 0x1000003D1

    sltu    a5, t4, a3
    sub     a4, t4, a3      # temp[0]

    mv      a6, t6
    sub     a6, a6, a5
    sltu    a5, t6, a5

    mv      a7, s2
    sub     a7, a7, a5
    sltu    a5, s2, a5

    mv      s0, s5
    sub     s0, s0, a5

    # Branchless select: if borrow (t5 != 0), use temp
    # mask = -borrow (all 1s if borrow, all 0s otherwise)
    neg     t5, t5

    xor     s1, t4, a4
    and     s1, s1, t5
    xor     t4, t4, s1

    xor     s1, t6, a6
    and     s1, s1, t5
    xor     t6, t6, s1

    xor     s1, s2, a7
    and     s1, s1, t5
    xor     s2, s2, s1

    xor     s1, s5, s0
    and     s1, s1, t5
    xor     s5, s5, s1

    # Store result
    sd      t4, 0(a0)
    sd      t6, 8(a0)
    sd      s2, 16(a0)
    sd      s5, 24(a0)

    # Restore callee-saved registers
    ld      s0, 8(sp)
    ld      s1, 16(sp)
    ld      s2, 24(sp)
    ld      s3, 32(sp)
    ld      s4, 40(sp)
    ld      s5, 48(sp)
    ld      s6, 56(sp)
    ld      s7, 64(sp)
    addi    sp, sp, 80
    ret
    .size field_sub_asm_riscv64, .-field_sub_asm_riscv64
.Lfield_sub_asm_riscv64_end:

/*
 * field_negate_asm_riscv64
 * 
 * Negate a field element: r = -a
 */
    .globl field_negate_asm_riscv64
    .type field_negate_asm_riscv64, @function
field_negate_asm_riscv64:
    # Save callee-saved registers
    addi    sp, sp, -64
    sd      s0, 8(sp)
    sd      s1, 16(sp)
    sd      s2, 24(sp)
    sd      s3, 32(sp)
    sd      s4, 40(sp)
    sd      s5, 48(sp)

    # Load operand
    ld      a3, 0(a1)       # a[0]
    ld      a4, 8(a1)       # a[1]
    ld      a5, 16(a1)      # a[2]
    ld      a6, 24(a1)       # a[3]

    # P[0] = 0xFFFFFFFEFFFFFC2F
    li      t0, 0xFFFFFC2F
    li      t1, 0xFFFFFFFE
    slli    t1, t1, 32
    or      t0, t0, t1      # t0 = P[0]
    li      t1, -1          # P[1] = P[2] = P[3] = 0xFFFFFFFFFFFFFFFF

    # r = P - a with borrow chain
    # limb 0
    sltu    t5, t0, a3      # borrow = (P[0] < a[0])
    sub     t4, t0, a3      # result[0] = P[0] - a[0]

    # limb 1: P[1] - a[1] - borrow
    mv      t6, t1
    sub     t6, t6, t5      # P[1] - borrow
    sltu    s0, t1, t5      # underflow from borrow
    sltu    s1, t6, a4      # will borrow from a[1] subtraction
    sub     t6, t6, a4      # result[1]
    or      t5, s0, s1      # combined borrow

    # limb 2: P[2] - a[2] - borrow
    mv      s0, t1
    sub     s0, s0, t5
    sltu    s2, t1, t5
    sltu    s3, s0, a5
    sub     s0, s0, a5
    or      t5, s2, s3

    # limb 3: P[3] - a[3] - borrow
    mv      s1, t1
    sub     s1, s1, t5
    sltu    s4, t1, t5
    sltu    s5, s1, a6
    sub     s1, s1, a6
    # Final borrow (s4 | s5) should be 0 for valid field element

    sd      t4, 0(a0)
    sd      t6, 8(a0)
    sd      s0, 16(a0)
    sd      s1, 24(a0)

    # Restore callee-saved registers
    ld      s0, 8(sp)
    ld      s1, 16(sp)
    ld      s2, 24(sp)
    ld      s3, 32(sp)
    ld      s4, 40(sp)
    ld      s5, 48(sp)
    addi    sp, sp, 64
    ret
    .size field_negate_asm_riscv64, .-field_negate_asm_riscv64
.Lfield_negate_asm_riscv64_end:

/*
 * field_square_asm_riscv64
 * 
 * Square a field element: r = a² (mod p)
 */
    .globl field_square_asm_riscv64
    .type field_square_asm_riscv64, @function
field_square_asm_riscv64:
    # Just call multiply with a, a
    mv      a2, a1
    tail    field_mul_asm_riscv64
    .size field_square_asm_riscv64, .-field_square_asm_riscv64

/*
 * field_add_asm_riscv64
 * 
 * Add two field elements: r = a + b (mod p)
 */
    .globl field_add_asm_riscv64
    .type field_add_asm_riscv64, @function
field_add_asm_riscv64:
    # Load a
    ld      t0, 0(a1)
    ld      t1, 8(a1)
    ld      t2, 16(a1)
    ld      t3, 24(a1)
    
    # Load b
    ld      t4, 0(a2)
    ld      t5, 8(a2)
    ld      t6, 16(a2)
    ld      a3, 24(a2)
    
    # Add with carry propagation (clean add-with-carry chain)
    add     t0, t0, t4      # limb 0
    sltu    a4, t0, t4      # c0

    add     t1, t1, a4      # add carry from limb 0
    sltu    a5, t1, a4      # c1a
    add     t1, t1, t5      # add b[1]
    sltu    a6, t1, t5      # c1b
    or      a4, a5, a6      # c1

    add     t2, t2, a4      # add carry from limb 1
    sltu    a5, t2, a4
    add     t2, t2, t6      # add b[2]
    sltu    a6, t2, t6
    or      a4, a5, a6      # c2

    add     t3, t3, a4
    sltu    a5, t3, a4
    add     t3, t3, a3
    sltu    a6, t3, a3
    or      a4, a5, a6      # c3 (scalar overflow)

    # Result is t0, t1, t2, t3. Carry is a4.
    # If carry==1 OR result >= P, subtract P.
    # Logic: if (carry | (result+K overflows 256 bits)), then result = result+K (-2^256 + K = -P)

    # Calculate R + K
    li      t4, 0x3D1       # K low part
    li      a5, 1           # K high part
    slli    a5, a5, 32
    or      t4, t4, a5      # t4 = 0x1000003D1

    add     t5, t0, t4      # r0 + K
    sltu    a5, t5, t0      # k_c0 (FIXED: compare result with original!)

    add     t6, t1, a5      # t6 = t1 + carry
    sltu    a5, t6, t1      # k_c1 (compare with original!)

    add     a1, t2, a5      # a1 = t2 + carry
    sltu    a5, a1, t2      # k_c2 (compare with original!)

    add     a2, t3, a5      # a2 = t3 + carry
    sltu    a3, a2, t3      # k_c3 (final carry out, compare with original!)

    # Final condition: a4 (original carry) OR a3 (K-check carry)
    or      a4, a4, a3

    # If a4 is non-zero, use [t5, t6, a1, a2] (which is R+K)
    # Else use [t0, t1, t2, t3]

    # Branchless select
    neg     a4, a4          # mask = -condition

    xor     a5, t0, t5
    and     a5, a5, a4
    xor     t0, t0, a5      # t0 = mask ? t5 : t0 (Verified: m=0 -> x, m=-1 -> y)

    xor     a5, t1, t6
    and     a5, a5, a4
    xor     t1, t1, a5

    xor     a5, t2, a1
    and     a5, a5, a4
    xor     t2, t2, a5

    xor     a5, t3, a2
    and     a5, a5, a4
    xor     t3, t3, a5

    sd      t0, 0(a0)
    sd      t1, 8(a0)
    sd      t2, 16(a0)
    sd      t3, 24(a0)
    
    ret
    .size field_add_asm_riscv64, .-field_add_asm_riscv64
.Lfield_add_asm_riscv64_end:


#ifdef SECP256K1_HAS_RISCV_VECTOR
/*
 * RISC-V Vector Extension (RVV) Batch Operations
 * 
 * These functions process multiple field elements at once using SIMD.
 * Requires: RVV 1.0 (rv64gcv)
 * 
 * Expected speedup: 4-8× for batch operations (8-16 elements at once)
 */

/*
 * field_mul_batch_rvv
 * 
 * Batch multiply: r[i] = a[i] × b[i] (mod p) for i = 0..count-1
 * 
 * Parameters:
 *   a0 = result pointer (uint64_t r[][4])
 *   a1 = operand a pointer (uint64_t a[][4])
 *   a2 = operand b pointer (uint64_t b[][4])
 *   a3 = count (number of elements)
 * 
 * Note: This is a placeholder for future RVV implementation.
 * Full RVV code requires vector register management and loop unrolling.
 */
    .globl field_mul_batch_rvv
    .type field_mul_batch_rvv, @function
field_mul_batch_rvv:
    # Save ra and s0-s4
    addi    sp, sp, -48
    sd      ra, 0(sp)
    sd      s0, 8(sp)   # counter
    sd      s1, 16(sp)  # result ptr
    sd      s2, 24(sp)  # a ptr
    sd      s3, 32(sp)  # b ptr
    sd      s4, 40(sp)  # count

    # Move arguments to saved registers
    mv      s1, a0
    mv      s2, a1
    mv      s3, a2
    mv      s4, a3
    li      s0, 0       # counter = 0

    # For now, fall back to scalar loop
1:
    beq     s0, s4, 2f      # if counter == count, exit

    # Call scalar multiply
    mv      a0, s1
    mv      a1, s2
    mv      a2, s3
    call    field_mul_asm_riscv64
    
    # Advance pointers and counter
    addi    s1, s1, 32      # r += 4 limbs
    addi    s2, s2, 32      # a += 4 limbs
    addi    s3, s3, 32      # b += 4 limbs
    addi    s0, s0, 1       # counter++
    j       1b
    
2:
    # Restore registers
    ld      ra, 0(sp)
    ld      s0, 8(sp)
    ld      s1, 16(sp)
    ld      s2, 24(sp)
    ld      s3, 32(sp)
    ld      s4, 40(sp)
    addi    sp, sp, 48
    ret
    .size field_mul_batch_rvv, .-field_mul_batch_rvv

#endif // SECP256K1_HAS_RISCV_VECTOR

    .section .note.GNU-stack,"",@progbits
