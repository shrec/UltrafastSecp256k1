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
 */
    .align 3
    .globl field_mul_asm_riscv64
    .type field_mul_asm_riscv64, @function
field_mul_asm_riscv64:
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

    mv      s11, a0

    ld      s0, 0(a1)
    ld      s1, 8(a1)
    ld      s2, 16(a1)
    ld      s3, 24(a1)

    ld      s4, 0(a2)
    ld      s5, 8(a2)
    ld      s6, 16(a2)
    ld      s7, 24(a2)

    # Column 0
    mul     t0, s0, s4
    mulhu   s9, s0, s4
    li      s10, 0

    # Column 1
    mv      t1, s9
    mv      s9, s10
    li      s10, 0

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

    # Column 2
    mv      t2, s9
    mv      s9, s10
    li      s10, 0

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

    # Column 3
    mv      t3, s9
    mv      s9, s10
    li      s10, 0

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

    # Column 4
    mv      t4, s9
    mv      s9, s10
    li      s10, 0

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

    # Column 5
    mv      t5, s9
    mv      s9, s10
    li      s10, 0

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

    # Column 6
    mv      t6, s9
    mv      s9, s10
    li      s10, 0

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

    # Column 7
    mv      s8, s9

    # Fast reduction
    li      s10, 977
    li      s9, 0

    # Process r4
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

    # Process r5
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

    # Process r6
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

    # Process r7
    mul     a1, s8, s10
    mulhu   a2, s8, s10
    add     t3, t3, a1
    sltu    a3, t3, a1
    add     s9, s9, a2
    add     s9, s9, a3

    slli    a1, s8, 32
    srli    a2, s8, 32
    add     t3, t3, a1
    sltu    a3, t3, a1
    add     s9, s9, a2
    add     s9, s9, a3

.Lreduce_loop:
    beqz    s9, .Lfinal_check

    mv      t4, s9
    li      s9, 0

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
    li      a1, 0x3D1
    li      a2, 1
    slli    a2, a2, 32
    or      a1, a1, a2

    add     a2, t0, a1
    sltu    a3, a2, t0
    add     a4, t1, a3
    sltu    a3, a4, t1
    add     a5, t2, a3
    sltu    a3, a5, t2
    add     a6, t3, a3
    sltu    a7, a6, t3

    neg     a7, a7

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

    sd      t0, 0(s11)
    sd      t1, 8(s11)
    sd      t2, 16(s11)
    sd      t3, 24(s11)

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

/*
 * field_sub_asm_riscv64
 *
 * Subtract two field elements: r = a - b (mod p)
 */
    .align 3
    .globl field_sub_asm_riscv64
    .type field_sub_asm_riscv64, @function
field_sub_asm_riscv64:
    addi    sp, sp, -80
    sd      s0, 8(sp)
    sd      s1, 16(sp)
    sd      s2, 24(sp)
    sd      s3, 32(sp)
    sd      s4, 40(sp)
    sd      s5, 48(sp)
    sd      s6, 56(sp)
    sd      s7, 64(sp)

    ld      a3, 0(a1)
    ld      a4, 8(a1)
    ld      a5, 16(a1)
    ld      a6, 24(a1)

    ld      t0, 0(a2)
    ld      t1, 8(a2)
    ld      t2, 16(a2)
    ld      t3, 24(a2)

    sltu    t5, a3, t0
    sub     t4, a3, t0

    mv      t6, a4
    sub     t6, t6, t5
    sltu    s0, a4, t5
    sltu    s1, t6, t1
    sub     t6, t6, t1
    or      t5, s0, s1

    mv      s2, a5
    sub     s2, s2, t5
    sltu    s3, a5, t5
    sltu    s4, s2, t2
    sub     s2, s2, t2
    or      t5, s3, s4

    mv      s5, a6
    sub     s5, s5, t5
    sltu    s6, a6, t5
    sltu    s7, s5, t3
    sub     s5, s5, t3
    or      t5, s6, s7

    li      a3, 0x3D1
    li      s0, 1
    slli    s0, s0, 32
    or      a3, a3, s0

    sltu    a5, t4, a3
    sub     a4, t4, a3

    mv      a6, t6
    sub     a6, a6, a5
    sltu    a5, t6, a5

    mv      a7, s2
    sub     a7, a7, a5
    sltu    a5, s2, a5

    mv      s0, s5
    sub     s0, s0, a5

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

    sd      t4, 0(a0)
    sd      t6, 8(a0)
    sd      s2, 16(a0)
    sd      s5, 24(a0)

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

/*
 * field_negate_asm_riscv64
 *
 * Negate a field element: r = -a (mod p)
 */
    .align 3
    .globl field_negate_asm_riscv64
    .type field_negate_asm_riscv64, @function
field_negate_asm_riscv64:
    addi    sp, sp, -64
    sd      s0, 8(sp)
    sd      s1, 16(sp)
    sd      s2, 24(sp)
    sd      s3, 32(sp)
    sd      s4, 40(sp)
    sd      s5, 48(sp)

    ld      a3, 0(a1)
    ld      a4, 8(a1)
    ld      a5, 16(a1)
    ld      a6, 24(a1)

    li      t0, 0xFFFFFC2F
    li      t1, 0xFFFFFFFE
    slli    t1, t1, 32
    or      t0, t0, t1
    li      t1, -1

    sltu    t5, t0, a3
    sub     t4, t0, a3

    mv      t6, t1
    sub     t6, t6, t5
    sltu    s0, t1, t5
    sltu    s1, t6, a4
    sub     t6, t6, a4
    or      t5, s0, s1

    mv      s0, t1
    sub     s0, s0, t5
    sltu    s2, t1, t5
    sltu    s3, s0, a5
    sub     s0, s0, a5
    or      t5, s2, s3

    mv      s1, t1
    sub     s1, s1, t5
    sltu    s4, t1, t5
    sltu    s5, s1, a6
    sub     s1, s1, a6

    sd      t4, 0(a0)
    sd      t6, 8(a0)
    sd      s0, 16(a0)
    sd      s1, 24(a0)

    ld      s0, 8(sp)
    ld      s1, 16(sp)
    ld      s2, 24(sp)
    ld      s3, 32(sp)
    ld      s4, 40(sp)
    ld      s5, 48(sp)
    addi    sp, sp, 64
    ret
    .size field_negate_asm_riscv64, .-field_negate_asm_riscv64

/*
 * field_square_asm_riscv64
 *
 * Square a field element: r = a² (mod p)
 *
 * Optimized squaring exploits symmetry: a[i]*a[j] = a[j]*a[i]
 * Instead of 16 multiplications, we do 10 muls + shifts
 *
 * 512-bit product structure for a² where a = [a0, a1, a2, a3]:
 *   r0 = a0²
 *   r1 = 2·a0·a1
 *   r2 = 2·a0·a2 + a1²
 *   r3 = 2·a0·a3 + 2·a1·a2
 *   r4 = 2·a1·a3 + a2²
 *   r5 = 2·a2·a3
 *   r6 = a3²
 */
    .align 3
    .globl field_square_asm_riscv64
    .type field_square_asm_riscv64, @function
field_square_asm_riscv64:
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

    mv      s11, a0         # Save result pointer

    # Load input a[0..3]
    ld      s0, 0(a1)       # a0
    ld      s1, 8(a1)       # a1
    ld      s2, 16(a1)      # a2
    ld      s3, 24(a1)      # a3

    # =========================================================
    # r0 = a0² (diagonal)
    # =========================================================
    mul     t0, s0, s0      # r0_lo
    mulhu   s9, s0, s0      # r0_hi -> carry to r1

    # =========================================================
    # r1 = 2·a0·a1 (off-diagonal, doubled)
    # =========================================================
    mul     a1, s0, s1      # (a0·a1)_lo
    mulhu   a2, s0, s1      # (a0·a1)_hi

    # Double: 2·(a0·a1)
    slli    t1, a1, 1       # low << 1
    srli    a3, a1, 63      # high bit of low
    slli    a4, a2, 1       # high << 1
    or      a4, a4, a3      # high | carry_from_low
    srli    s10, a2, 63     # overflow bit

    # Add carry from r0
    add     t1, t1, s9
    sltu    a3, t1, s9
    add     s9, a4, a3      # s9 = carry to r2
    add     s10, s10, zero  # s10 = overflow (usually 0)

    # =========================================================
    # r2 = 2·a0·a2 + a1²
    # =========================================================
    # First: a1² (diagonal)
    mul     a5, s1, s1      # (a1²)_lo
    mulhu   a6, s1, s1      # (a1²)_hi

    # Second: 2·a0·a2 (off-diagonal)
    mul     a1, s0, s2
    mulhu   a2, s0, s2
    slli    a7, a1, 1
    srli    a3, a1, 63
    slli    t6, a2, 1
    or      t6, t6, a3
    srli    a4, a2, 63      # overflow

    # Sum: 2·a0·a2 + a1²
    add     t2, a7, a5
    sltu    a3, t2, a5
    add     a6, a6, t6
    sltu    t6, a6, t6
    add     a6, a6, a3
    sltu    a3, a6, a3
    add     t6, t6, a3
    add     t6, t6, a4      # carry chain

    # Add carry from r1
    add     t2, t2, s9
    sltu    a3, t2, s9
    add     s9, a6, a3
    sltu    a3, s9, a3
    add     s10, t6, a3

    # =========================================================
    # r3 = 2·a0·a3 + 2·a1·a2
    # =========================================================
    # a0·a3
    mul     a1, s0, s3
    mulhu   a2, s0, s3

    # a1·a2
    mul     a5, s1, s2
    mulhu   a6, s1, s2

    # Sum before doubling
    add     a1, a1, a5
    sltu    a3, a1, a5
    add     a2, a2, a6
    sltu    a4, a2, a6
    add     a2, a2, a3
    sltu    a3, a2, a3
    add     a4, a4, a3      # overflow

    # Double the sum
    slli    t3, a1, 1
    srli    a3, a1, 63
    slli    a5, a2, 1
    or      a5, a5, a3
    srli    a6, a2, 63
    slli    a4, a4, 1
    or      a4, a4, a6

    # Add carry from r2
    add     t3, t3, s9
    sltu    a3, t3, s9
    add     s9, a5, a3
    sltu    a3, s9, a3
    add     s10, a4, a3
    add     s10, s10, zero

    # =========================================================
    # r4 = 2·a1·a3 + a2²
    # =========================================================
    # a2² (diagonal)
    mul     a5, s2, s2
    mulhu   a6, s2, s2

    # 2·a1·a3 (off-diagonal)
    mul     a1, s1, s3
    mulhu   a2, s1, s3
    slli    a7, a1, 1
    srli    a3, a1, 63
    slli    t6, a2, 1
    or      t6, t6, a3
    srli    a4, a2, 63

    # Sum
    add     t4, a7, a5
    sltu    a3, t4, a5
    add     a6, a6, t6
    sltu    t6, a6, t6
    add     a6, a6, a3
    sltu    a3, a6, a3
    add     t6, t6, a3
    add     t6, t6, a4

    # Add carry from r3
    add     t4, t4, s9
    sltu    a3, t4, s9
    add     s9, a6, a3
    sltu    a3, s9, a3
    add     s10, t6, a3

    # =========================================================
    # r5 = 2·a2·a3
    # =========================================================
    mul     a1, s2, s3
    mulhu   a2, s2, s3
    slli    t5, a1, 1
    srli    a3, a1, 63
    slli    a4, a2, 1
    or      a4, a4, a3
    srli    a5, a2, 63

    # Add carry from r4
    add     t5, t5, s9
    sltu    a3, t5, s9
    add     s9, a4, a3
    sltu    a3, s9, a3
    add     s10, a5, a3

    # =========================================================
    # r6 = a3² (diagonal)
    # =========================================================
    mul     t6, s3, s3
    mulhu   s8, s3, s3

    # Add carry from r5
    add     t6, t6, s9
    sltu    a3, t6, s9
    add     s8, s8, a3

    # =========================================================
    # Reduction: 512-bit -> 256-bit mod p
    # p = 2^256 - 2^32 - 977
    # High limbs [r4,r5,r6,r7] * (2^32 + 977) added to [r0,r1,r2,r3]
    # =========================================================
    li      s10, 977
    li      s9, 0

    # --- Process r4 (t4) ---
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
    mul     a1, s8, s10
    mulhu   a2, s8, s10
    add     t3, t3, a1
    sltu    a3, t3, a1
    add     s9, s9, a2
    add     s9, s9, a3

    slli    a1, s8, 32
    srli    a2, s8, 32
    add     t3, t3, a1
    sltu    a3, t3, a1
    add     s9, s9, a2
    add     s9, s9, a3

    # =========================================================
    # Final reduction loop (handle overflow in s9)
    # =========================================================
.Lsquare_reduce_loop:
    beqz    s9, .Lsquare_final_check

    mv      t4, s9
    li      s9, 0

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

    j       .Lsquare_reduce_loop

.Lsquare_final_check:
    # Branchless final reduction: if result >= p, subtract p
    # K = 2^32 + 977 = 0x1000003D1
    li      a1, 0x3D1
    li      a2, 1
    slli    a2, a2, 32
    or      a1, a1, a2

    add     a2, t0, a1
    sltu    a3, a2, t0
    add     a4, t1, a3
    sltu    a3, a4, t1
    add     a5, t2, a3
    sltu    a3, a5, t2
    add     a6, t3, a3
    sltu    a7, a6, t3

    # Branchless select: if overflow, use reduced result
    neg     a7, a7

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

    # Store result
    sd      t0, 0(s11)
    sd      t1, 8(s11)
    sd      t2, 16(s11)
    sd      t3, 24(s11)

    # Restore callee-saved registers
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
    .size field_square_asm_riscv64, .-field_square_asm_riscv64

/*
 * field_add_asm_riscv64
 *
 * Add two field elements: r = a + b (mod p)
 */
    .align 3
    .globl field_add_asm_riscv64
    .type field_add_asm_riscv64, @function
field_add_asm_riscv64:
    ld      t0, 0(a1)
    ld      t1, 8(a1)
    ld      t2, 16(a1)
    ld      t3, 24(a1)

    ld      t4, 0(a2)
    ld      t5, 8(a2)
    ld      t6, 16(a2)
    ld      a3, 24(a2)

    add     t0, t0, t4
    sltu    a4, t0, t4

    add     t1, t1, a4
    sltu    a5, t1, a4
    add     t1, t1, t5
    sltu    a6, t1, t5
    or      a4, a5, a6

    add     t2, t2, a4
    sltu    a5, t2, a4
    add     t2, t2, t6
    sltu    a6, t2, t6
    or      a4, a5, a6

    add     t3, t3, a4
    sltu    a5, t3, a4
    add     t3, t3, a3
    sltu    a6, t3, a3
    or      a4, a5, a6

    li      t4, 0x3D1
    li      a5, 1
    slli    a5, a5, 32
    or      t4, t4, a5

    add     t5, t0, t4
    sltu    a5, t5, t0

    add     t6, t1, a5
    sltu    a5, t6, t1

    add     a1, t2, a5
    sltu    a5, a1, t2

    add     a2, t3, a5
    sltu    a3, a2, t3

    or      a4, a4, a3
    neg     a4, a4

    xor     a5, t0, t5
    and     a5, a5, a4
    xor     t0, t0, a5

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

#ifdef SECP256K1_HAS_RISCV_VECTOR
/*
 * field_mul_batch_rvv - RVV batch multiply (placeholder)
 */
    .globl field_mul_batch_rvv
    .type field_mul_batch_rvv, @function
field_mul_batch_rvv:
    addi    sp, sp, -48
    sd      ra, 0(sp)
    sd      s0, 8(sp)
    sd      s1, 16(sp)
    sd      s2, 24(sp)
    sd      s3, 32(sp)
    sd      s4, 40(sp)

    mv      s1, a0
    mv      s2, a1
    mv      s3, a2
    mv      s4, a3
    li      s0, 0

1:
    beq     s0, s4, 2f
    mv      a0, s1
    mv      a1, s2
    mv      a2, s3
    call    field_mul_asm_riscv64
    addi    s1, s1, 32
    addi    s2, s2, 32
    addi    s3, s3, 32
    addi    s0, s0, 1
    j       1b

2:
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

/*
 * scalar_add_asm_riscv64
 *
 * Add two 256-bit scalars mod N (secp256k1 curve order)
 * N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
 *
 * Parameters:
 *   a0 = result pointer (uint64_t r[4])
 *   a1 = operand a pointer (uint64_t a[4])
 *   a2 = operand b pointer (uint64_t b[4])
 */
    .align 3
    .globl scalar_add_asm_riscv64
    .type scalar_add_asm_riscv64, @function
scalar_add_asm_riscv64:
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

    # Add with carry propagation
    add     t0, t0, t4
    sltu    a4, t0, t4

    add     t1, t1, a4
    sltu    a5, t1, a4
    add     t1, t1, t5
    sltu    a6, t1, t5
    or      a4, a5, a6

    add     t2, t2, a4
    sltu    a5, t2, a4
    add     t2, t2, t6
    sltu    a6, t2, t6
    or      a4, a5, a6

    add     t3, t3, a4
    sltu    a5, t3, a4
    add     t3, t3, a3
    sltu    a6, t3, a3
    or      a4, a5, a6      # carry out

    # Load N (curve order) for comparison
    # N[0] = 0xBFD25E8CD0364141
    li      t4, 0xD0364141
    li      a5, 0xBFD25E8C
    slli    a5, a5, 32
    or      t4, t4, a5

    # N[1] = 0xBAAEDCE6AF48A03B
    li      t5, 0xAF48A03B
    li      a5, 0xBAAEDCE6
    slli    a5, a5, 32
    or      t5, t5, a5

    # N[2] = 0xFFFFFFFFFFFFFFFE
    li      t6, -2

    # N[3] = 0xFFFFFFFFFFFFFFFF
    li      a3, -1

    # Compute result - N
    sltu    a5, t0, t4
    sub     a1, t0, t4      # r0 - N0

    mv      a6, t1
    sub     a6, a6, a5
    sltu    a7, t1, a5
    sltu    a5, a6, t5
    sub     a6, a6, t5      # r1 - N1 - borrow
    or      a5, a7, a5

    mv      a7, t2
    sub     a7, a7, a5
    sltu    a2, t2, a5
    sltu    a5, a7, t6
    sub     a7, a7, t6      # r2 - N2 - borrow
    or      a5, a2, a5

    mv      a2, t3
    sub     a2, a2, a5
    sltu    a5, t3, a5
    sltu    t4, a2, a3
    sub     a2, a2, a3      # r3 - N3 - borrow
    or      a5, a5, t4      # final borrow

    # Select: if (carry || !borrow) use subtracted result
    # mask = -(carry | !borrow)
    seqz    a5, a5          # a5 = (borrow == 0) ? 1 : 0
    or      a4, a4, a5      # condition = carry | !borrow
    neg     a4, a4          # mask

    xor     t4, t0, a1
    and     t4, t4, a4
    xor     t0, t0, t4

    xor     t4, t1, a6
    and     t4, t4, a4
    xor     t1, t1, t4

    xor     t4, t2, a7
    and     t4, t4, a4
    xor     t2, t2, t4

    xor     t4, t3, a2
    and     t4, t4, a4
    xor     t3, t3, t4

    sd      t0, 0(a0)
    sd      t1, 8(a0)
    sd      t2, 16(a0)
    sd      t3, 24(a0)

    ret
    .size scalar_add_asm_riscv64, .-scalar_add_asm_riscv64

/*
 * scalar_sub_asm_riscv64
 *
 * Subtract two 256-bit scalars mod N: r = a - b (mod N)
 */
    .align 3
    .globl scalar_sub_asm_riscv64
    .type scalar_sub_asm_riscv64, @function
scalar_sub_asm_riscv64:
    addi    sp, sp, -64
    sd      s0, 8(sp)
    sd      s1, 16(sp)
    sd      s2, 24(sp)
    sd      s3, 32(sp)
    sd      s4, 40(sp)
    sd      s5, 48(sp)

    # Load a
    ld      a3, 0(a1)
    ld      a4, 8(a1)
    ld      a5, 16(a1)
    ld      a6, 24(a1)

    # Load b
    ld      t0, 0(a2)
    ld      t1, 8(a2)
    ld      t2, 16(a2)
    ld      t3, 24(a2)

    # Subtract with borrow chain
    sltu    t5, a3, t0
    sub     t4, a3, t0

    mv      t6, a4
    sub     t6, t6, t5
    sltu    s0, a4, t5
    sltu    s1, t6, t1
    sub     t6, t6, t1
    or      t5, s0, s1

    mv      s2, a5
    sub     s2, s2, t5
    sltu    s3, a5, t5
    sltu    s4, s2, t2
    sub     s2, s2, t2
    or      t5, s3, s4

    mv      s5, a6
    sub     s5, s5, t5
    sltu    s3, a6, t5
    sltu    s4, s5, t3
    sub     s5, s5, t3
    or      t5, s3, s4      # final borrow

    # If borrow, add N back
    # N[0] = 0xBFD25E8CD0364141
    li      a3, 0xD0364141
    li      s0, 0xBFD25E8C
    slli    s0, s0, 32
    or      a3, a3, s0

    # N[1] = 0xBAAEDCE6AF48A03B
    li      a4, 0xAF48A03B
    li      s0, 0xBAAEDCE6
    slli    s0, s0, 32
    or      a4, a4, s0

    # N[2] = 0xFFFFFFFFFFFFFFFE, N[3] = 0xFFFFFFFFFFFFFFFF
    li      a5, -2
    li      a6, -1

    # Add N to result
    add     a1, t4, a3
    sltu    s0, a1, t4

    add     a2, t6, s0
    sltu    s1, a2, t6
    add     a2, a2, a4
    sltu    s3, a2, a4
    or      s0, s1, s3

    add     a7, s2, s0
    sltu    s1, a7, s2
    add     a7, a7, a5
    sltu    s3, a7, a5
    or      s0, s1, s3

    add     s0, s5, s0
    add     s0, s0, a6

    # Branchless select: if borrow, use added result
    neg     t5, t5          # mask = -borrow

    xor     s1, t4, a1
    and     s1, s1, t5
    xor     t4, t4, s1

    xor     s1, t6, a2
    and     s1, s1, t5
    xor     t6, t6, s1

    xor     s1, s2, a7
    and     s1, s1, t5
    xor     s2, s2, s1

    xor     s1, s5, s0
    and     s1, s1, t5
    xor     s5, s5, s1

    sd      t4, 0(a0)
    sd      t6, 8(a0)
    sd      s2, 16(a0)
    sd      s5, 24(a0)

    ld      s0, 8(sp)
    ld      s1, 16(sp)
    ld      s2, 24(sp)
    ld      s3, 32(sp)
    ld      s4, 40(sp)
    ld      s5, 48(sp)
    addi    sp, sp, 64
    ret
    .size scalar_sub_asm_riscv64, .-scalar_sub_asm_riscv64

    .section .note.GNU-stack,"",@progbits

