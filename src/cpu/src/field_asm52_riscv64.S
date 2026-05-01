# ═══════════════════════════════════════════════════════════════════════════
# 5×52 Field Arithmetic — RISC-V 64 Assembly
# ═══════════════════════════════════════════════════════════════════════════
#
# Optimized field multiplication and squaring for secp256k1 using
# RISC-V MUL/MULHU instructions for 64×64→128-bit products.
#
# RISC-V has no flags register — carry detection uses SLTU.
#
# Functions:
#   fe52_mul_inner_riscv64(uint64_t *r, const uint64_t *a, const uint64_t *b)
#   fe52_sqr_inner_riscv64(uint64_t *r, const uint64_t *a)
#
# Calling convention: RISC-V LP64 (a0=r, a1=a, a2=b)
# Callee-saved: s0-s11, ra
# ═══════════════════════════════════════════════════════════════════════════

.text
.option arch, +zba, +zbb

# Reduction constants
.set R52,    0x1000003D10
.set R52_R4, 0x1000003D1
.set R52_L12,0x1000003D10000
.set M52,    0xFFFFFFFFFFFFF

# ═══════════════════════════════════════════════════════════════════════════
# Helper macro: 128-bit multiply-accumulate
# (d_hi:d_lo) += rs1 * rs2
# Uses t3, t4 as scratch (clobbers them)
# ═══════════════════════════════════════════════════════════════════════════
.macro MULACCUM d_lo, d_hi, rs1, rs2, tl, th
    mul   \tl, \rs1, \rs2       # tl = lo(rs1 * rs2)
    mulhu \th, \rs1, \rs2       # th = hi(rs1 * rs2)
    add   \d_lo, \d_lo, \tl     # d_lo += tl
    sltu  \tl, \d_lo, \tl       # carry = (d_lo < tl) ? 1 : 0
    add   \d_hi, \d_hi, \th     # d_hi += th
    add   \d_hi, \d_hi, \tl     # d_hi += carry
.endm

# (d_hi:d_lo) = rs1 * rs2  (initial, no accumulate)
.macro MULPROD d_lo, d_hi, rs1, rs2
    mul   \d_lo, \rs1, \rs2
    mulhu \d_hi, \rs1, \rs2
.endm

# 128-bit right shift by 52: (hi:lo) >>= 52
# result_lo = (lo >> 52) | (hi << 12)
# result_hi = hi >> 52
.macro RSHIFT52 lo, hi
    srli  \lo, \lo, 52
    slli  t6, \hi, 12
    or    \lo, \lo, t6
    srli  \hi, \hi, 52
.endm

# ═══════════════════════════════════════════════════════════════════════════
# fe52_mul_inner_riscv64
# a0 = r (output), a1 = a (input), a2 = b (input)
# ═══════════════════════════════════════════════════════════════════════════
.globl fe52_mul_inner_riscv64
.type  fe52_mul_inner_riscv64, @function
.p2align 2
fe52_mul_inner_riscv64:
    # Save callee-saved registers
    addi sp, sp, -112
    sd   s0,   0(sp)
    sd   s1,   8(sp)
    sd   s2,  16(sp)
    sd   s3,  24(sp)
    sd   s4,  32(sp)
    sd   s5,  40(sp)
    sd   s6,  48(sp)
    sd   s7,  56(sp)
    sd   s8,  64(sp)
    sd   s9,  72(sp)
    sd   s10, 80(sp)
    sd   s11, 88(sp)
    sd   ra,  96(sp)

    # Register assignments:
    # s0=a0, s1=a1, s2=a2, s3=a3, s4=a4 (loaded from a[])
    # s5=M52 mask
    # s6=r pointer (saved a0)
    # s7=b0, t1=b1, t2=b2, t5=b3, a7=b4 (preloaded from b[])
    # s8=d_lo, s9=d_hi
    # s10=c_lo, s11=c_hi
    # t0, t3, t4, t6 = scratch

    mv   s6, a0                 # save r pointer

    # Load a[0..4]
    ld   s0,  0(a1)
    ld   s1,  8(a1)
    ld   s2, 16(a1)
    ld   s3, 24(a1)
    ld   s4, 32(a1)

    # Preload b[0..4] into registers -- eliminates ~20 memory loads
    # (each ld is 3-4 cycle stall on SiFive U74 L1 cache)
    ld   s7,  0(a2)             # b[0]
    ld   t1,  8(a2)             # b[1]
    ld   t2, 16(a2)             # b[2]
    ld   t5, 24(a2)             # b[3]
    ld   a7, 32(a2)             # b[4]

    # M52 constant
    li   s5, 0xFFFFFFFFFFFFF

    # --- Step 1: d = a0*b3 + a1*b2 + a2*b1 + a3*b0 ------------------
    MULPROD s8, s9, s0, t5      # d = a0*b3

    MULACCUM s8, s9, s1, t2, t3, t4

    MULACCUM s8, s9, s2, t1, t3, t4

    MULACCUM s8, s9, s3, s7, t3, t4

    # c = a4*b4
    MULPROD s10, s11, s4, a7

    # d += R * c_lo
    li   t0, 0x1000003D10
    MULACCUM s8, s9, t0, s10, t3, t4

    # c >>= 64: c_lo = c_hi
    mv   s10, s11

    # t3 = d & M52;  d >>= 52
    and  a3, s8, s5             # a3 = t3
    RSHIFT52 s8, s9

    # --- Step 2: d += a0*b4 + a1*b3 + a2*b2 + a3_val*b1 + a4*b0 ------
    MULACCUM s8, s9, s0, a7, t3, t4
    MULACCUM s8, s9, s1, t5, t3, t4
    MULACCUM s8, s9, s2, t2, t3, t4
    MULACCUM s8, s9, s3, t1, t3, t4
    MULACCUM s8, s9, s4, s7, t3, t4

    # d += (R<<12) * c_remaining
    li   t0, 0x1000003D10000
    MULACCUM s8, s9, t0, s10, t3, t4

    # t4_val = d & M52;  d >>= 52
    and  a4, s8, s5             # a4 = t4
    RSHIFT52 s8, s9

    # tx = t4 >> 48;  t4 &= M48
    srli a5, a4, 48             # a5 = tx
    srli t0, s5, 4
    and  a4, a4, t0             # t4 with M48 mask

    # --- Step 3: col0 + col5 -------------------------------------------
    MULACCUM s8, s9, s1, a7, t3, t4
    MULACCUM s8, s9, s2, t5, t3, t4
    MULACCUM s8, s9, s3, t2, t3, t4
    MULACCUM s8, s9, s4, t1, t3, t4

    # u0 = d & M52;  d >>= 52
    and  a6, s8, s5             # a6 = u0
    RSHIFT52 s8, s9

    # u0 = (u0 << 4) | tx
    slli a6, a6, 4
    or   a6, a6, a5

    # c = a0 * b[0]
    MULPROD s10, s11, s0, s7

    # c += u0 * (R >> 4)
    li   t0, 0x1000003D1
    MULACCUM s10, s11, a6, t0, t3, t4

    # r[0] = c & M52;  c >>= 52
    and  t0, s10, s5
    sd   t0, 0(s6)
    RSHIFT52 s10, s11

    # --- Step 4: col1 + col6 -------------------------------------------
    MULACCUM s10, s11, s0, t1, t3, t4
    MULACCUM s10, s11, s1, s7, t3, t4

    MULACCUM s8, s9, s2, a7, t3, t4
    MULACCUM s8, s9, s3, t5, t3, t4
    MULACCUM s8, s9, s4, t2, t3, t4

    # c += (d & M52) * R
    and  a6, s8, s5
    li   t0, 0x1000003D10
    MULACCUM s10, s11, a6, t0, t3, t4

    # d >>= 52
    RSHIFT52 s8, s9

    # r[1] = c & M52;  c >>= 52
    and  t0, s10, s5
    sd   t0, 8(s6)
    RSHIFT52 s10, s11

    # --- Step 5: col2 + col7 -------------------------------------------
    MULACCUM s10, s11, s0, t2, t3, t4
    MULACCUM s10, s11, s1, t1, t3, t4
    MULACCUM s10, s11, s2, s7, t3, t4

    MULACCUM s8, s9, s3, a7, t3, t4
    MULACCUM s8, s9, s4, t5, t3, t4

    # c += R * d_lo (full 64-bit!)
    li   t0, 0x1000003D10
    MULACCUM s10, s11, t0, s8, t3, t4

    # d >>= 64
    mv   s8, s9
    li   s9, 0

    # r[2] = c & M52;  c >>= 52
    and  t0, s10, s5
    sd   t0, 16(s6)
    RSHIFT52 s10, s11

    # --- Step 6: Finalize -----------------------------------------------
    # c += (R<<12) * d
    li   t0, 0x1000003D10000
    MULACCUM s10, s11, t0, s8, t3, t4

    # c += t3
    add  s10, s10, a3
    sltu t0, s10, a3
    add  s11, s11, t0

    # r[3] = c & M52;  c >>= 52
    and  t0, s10, s5
    sd   t0, 24(s6)
    RSHIFT52 s10, s11

    # c += t4
    add  s10, s10, a4

    # r[4] = c
    sd   s10, 32(s6)

    # Restore registers
    ld   s0,   0(sp)
    ld   s1,   8(sp)
    ld   s2,  16(sp)
    ld   s3,  24(sp)
    ld   s4,  32(sp)
    ld   s5,  40(sp)
    ld   s6,  48(sp)
    ld   s7,  56(sp)
    ld   s8,  64(sp)
    ld   s9,  72(sp)
    ld   s10, 80(sp)
    ld   s11, 88(sp)
    ld   ra,  96(sp)
    addi sp, sp, 112
    ret
.size fe52_mul_inner_riscv64, .-fe52_mul_inner_riscv64


# ═══════════════════════════════════════════════════════════════════════════
# fe52_sqr_inner_riscv64
# a0 = r (output), a1 = a (input)
# ═══════════════════════════════════════════════════════════════════════════
.globl fe52_sqr_inner_riscv64
.type  fe52_sqr_inner_riscv64, @function
.p2align 2
fe52_sqr_inner_riscv64:
    addi sp, sp, -112
    sd   s0,   0(sp)
    sd   s1,   8(sp)
    sd   s2,  16(sp)
    sd   s3,  24(sp)
    sd   s4,  32(sp)
    sd   s5,  40(sp)
    sd   s6,  48(sp)
    sd   s7,  56(sp)
    sd   s8,  64(sp)
    sd   s9,  72(sp)
    sd   s10, 80(sp)
    sd   s11, 88(sp)
    sd   ra,  96(sp)

    mv   s6, a0                 # save r pointer

    # Load a[0..4]
    ld   s0,  0(a1)
    ld   s1,  8(a1)
    ld   s2, 16(a1)
    ld   s3, 24(a1)
    ld   s4, 32(a1)

    li   s5, 0xFFFFFFFFFFFFF    # M52

    # Pre-compute doubled values into s7, a2, a3_reg, a4_reg
    # s7 = a0*2, a2 = a1*2, a7 = a2*2, t5 = a3*2
    slli s7, s0, 1              # a0*2
    slli a2, s1, 1              # a1*2
    slli a7, s2, 1              # a2*2
    slli t5, s3, 1              # a3*2

    # ─── Step 1 ──────────────────────────────────────────────────────
    MULPROD s8, s9, s7, s3              # d = (a0*2)*a3
    MULACCUM s8, s9, a2, s2, t3, t4    # d += (a1*2)*a2
    MULPROD s10, s11, s4, s4            # c = a4²
    li   t0, 0x1000003D10
    MULACCUM s8, s9, t0, s10, t3, t4   # d += R*c_lo
    mv   s10, s11                       # c >>= 64
    and  a3, s8, s5                     # t3 = d & M
    RSHIFT52 s8, s9

    # ─── Step 2 ──────────────────────────────────────────────────────
    MULACCUM s8, s9, s7, s4, t3, t4    # (a0*2)*a4
    MULACCUM s8, s9, a2, s3, t3, t4    # (a1*2)*a3
    MULACCUM s8, s9, s2, s2, t3, t4    # a2²
    li   t0, 0x1000003D10000
    MULACCUM s8, s9, t0, s10, t3, t4   # (R<<12)*c_rem
    and  a4, s8, s5                     # t4
    RSHIFT52 s8, s9
    srli a5, a4, 48                     # tx
    srli t0, s5, 4
    and  a4, a4, t0                     # t4 &= M48

    # ─── Step 3 ──────────────────────────────────────────────────────
    MULACCUM s8, s9, a2, s4, t3, t4    # (a1*2)*a4
    MULACCUM s8, s9, a7, s3, t3, t4    # (a2*2)*a3
    and  a6, s8, s5                     # u0
    RSHIFT52 s8, s9
    slli a6, a6, 4
    or   a6, a6, a5                     # u0 = (u0<<4)|tx
    MULPROD s10, s11, s0, s0            # c = a0²
    li   t0, 0x1000003D1
    MULACCUM s10, s11, a6, t0, t3, t4  # c += u0*(R>>4)
    and  t0, s10, s5
    sd   t0, 0(s6)                      # r[0]
    RSHIFT52 s10, s11

    # ─── Step 4 ──────────────────────────────────────────────────────
    MULACCUM s10, s11, s7, s1, t3, t4  # (a0*2)*a1
    MULACCUM s8, s9, a7, s4, t3, t4    # (a2*2)*a4
    MULACCUM s8, s9, s3, s3, t3, t4    # a3²
    and  a6, s8, s5
    li   t0, 0x1000003D10
    MULACCUM s10, s11, a6, t0, t3, t4
    RSHIFT52 s8, s9
    and  t0, s10, s5
    sd   t0, 8(s6)                      # r[1]
    RSHIFT52 s10, s11

    # ─── Step 5 ──────────────────────────────────────────────────────
    MULACCUM s10, s11, s7, s2, t3, t4  # (a0*2)*a2
    MULACCUM s10, s11, s1, s1, t3, t4  # a1²
    MULACCUM s8, s9, t5, s4, t3, t4    # (a3*2)*a4
    li   t0, 0x1000003D10
    MULACCUM s10, s11, t0, s8, t3, t4  # c += R*d_lo
    mv   s8, s9
    li   s9, 0
    and  t0, s10, s5
    sd   t0, 16(s6)                     # r[2]
    RSHIFT52 s10, s11

    # ─── Step 6 ──────────────────────────────────────────────────────
    li   t0, 0x1000003D10000
    MULACCUM s10, s11, t0, s8, t3, t4
    add  s10, s10, a3
    sltu t0, s10, a3
    add  s11, s11, t0
    and  t0, s10, s5
    sd   t0, 24(s6)                     # r[3]
    RSHIFT52 s10, s11
    add  s10, s10, a4
    sd   s10, 32(s6)                    # r[4]

    # Restore
    ld   s0,   0(sp)
    ld   s1,   8(sp)
    ld   s2,  16(sp)
    ld   s3,  24(sp)
    ld   s4,  32(sp)
    ld   s5,  40(sp)
    ld   s6,  48(sp)
    ld   s7,  56(sp)
    ld   s8,  64(sp)
    ld   s9,  72(sp)
    ld   s10, 80(sp)
    ld   s11, 88(sp)
    ld   ra,  96(sp)
    addi sp, sp, 112
    ret
.size fe52_sqr_inner_riscv64, .-fe52_sqr_inner_riscv64
