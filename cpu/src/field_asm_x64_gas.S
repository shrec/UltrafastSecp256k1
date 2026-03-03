# ==============================================================================
# SECP256K1 Field Operations - x64 Assembly (GAS/AT&T Syntax)
# ==============================================================================
# Ultra-optimized field arithmetic using MULX + ADCX/ADOX dual carry chains
# Target: eliminate pipeline stalls from sequential adc-zero chains
#
# Platform: Linux x64, GAS (GNU Assembler)
# Required: BMI2 (MULX) + ADX (ADCX/ADOX) (Intel Broadwell 2015+, AMD Zen 2017+)
# ==============================================================================

#if defined(_WIN32) || defined(__WIN32__) || defined(WIN32)
    #define ELF_TYPE(name)
    #define ELF_SIZE(name)
    #define ELF_SECTION(...)

    /* Windows (Clang with sysv_abi) Compatibility Layer */
    /* C++ calls with __attribute__((sysv_abi)), so regs are RDI, RSI, RDX... */
    /* No shims needed for registers! Just hide ELF directives. */
    #define PROLOGUE_3ARG
    #define PROLOGUE_2ARG
    #define PROLOGUE_1ARG
    #define EPILOGUE
#else
    #define ELF_TYPE(name) .type name, @function
    #define ELF_SIZE(name) .size name, .-name
    /* Accept commas by taking varargs and emitting them verbatim. */
    #define ELF_SECTION(...) __VA_ARGS__

    #define PROLOGUE_3ARG
    #define PROLOGUE_2ARG
    #define PROLOGUE_1ARG
    #define EPILOGUE
#endif

.text
.intel_syntax noprefix

# ==============================================================================
# mul_4x4_asm: 256-bit × 256-bit → 512-bit multiplication
# ==============================================================================
# Ultra-fast bigint multiplication using BMI2 MULX + parallel carry chains
#
# Parameters (x64 System V calling convention - Linux):
#   RDI = const uint64_t* a  (pointer to 4 limbs)
#   RSI = const uint64_t* b  (pointer to 4 limbs)
#   RDX = uint64_t* result   (pointer to 8 limbs output)
#
# Note: Different from Windows! System V uses RDI, RSI, RDX, RCX, R8, R9
#
# Performance: Target 6-8 ns (vs 33-37 ns with BMI2 intrinsics)
# ==============================================================================

.globl mul_4x4_asm
ELF_TYPE(mul_4x4_asm)
mul_4x4_asm:
    PROLOGUE_3ARG
    push rbx
    push rbp
    push r12
    push r13
    push r14
    push r15
    push rdx              # save result pointer (rdx used by mulx)

    # Zero accumulators
    xor r8, r8
    xor r9, r9
    xor r10, r10
    xor r11, r11
    xor r12, r12
    xor r13, r13
    xor r14, r14
    xor r15, r15
    xor ebp, ebp          # overflow register

    # ---- Row 0: a[0] * b[0..3] (MULX + ADCX/ADOX dual carry) ----
    mov rdx, [rdi]
    xor ecx, ecx          # rcx=0, CF=0, OF=0

    mulx rbx, rax, [rsi]
    adcx r8, rax
    adox r9, rbx

    mulx rbx, rax, [rsi+8]
    adcx r9, rax
    adox r10, rbx

    mulx rbx, rax, [rsi+16]
    adcx r10, rax
    adox r11, rbx

    mulx rbx, rax, [rsi+24]
    adcx r11, rax
    adox r12, rbx

    adcx r12, rcx
    adox r13, rcx

    # ---- Row 1: a[1] * b[0..3] ----
    mov rdx, [rdi+8]
    xor ecx, ecx

    mulx rbx, rax, [rsi]
    adcx r9, rax
    adox r10, rbx

    mulx rbx, rax, [rsi+8]
    adcx r10, rax
    adox r11, rbx

    mulx rbx, rax, [rsi+16]
    adcx r11, rax
    adox r12, rbx

    mulx rbx, rax, [rsi+24]
    adcx r12, rax
    adox r13, rbx

    adcx r13, rcx
    adox r14, rcx

    # ---- Row 2: a[2] * b[0..3] ----
    mov rdx, [rdi+16]
    xor ecx, ecx

    mulx rbx, rax, [rsi]
    adcx r10, rax
    adox r11, rbx

    mulx rbx, rax, [rsi+8]
    adcx r11, rax
    adox r12, rbx

    mulx rbx, rax, [rsi+16]
    adcx r12, rax
    adox r13, rbx

    mulx rbx, rax, [rsi+24]
    adcx r13, rax
    adox r14, rbx

    adcx r14, rcx
    adox r15, rcx

    # ---- Row 3: a[3] * b[0..3] ----
    mov rdx, [rdi+24]
    xor ecx, ecx

    mulx rbx, rax, [rsi]
    adcx r11, rax
    adox r12, rbx

    mulx rbx, rax, [rsi+8]
    adcx r12, rax
    adox r13, rbx

    mulx rbx, rax, [rsi+16]
    adcx r13, rax
    adox r14, rbx

    mulx rbx, rax, [rsi+24]
    adcx r14, rax
    adox r15, rbx

    adcx r15, rcx
    adox rbp, rcx          # overflow (0 for 256x256)

    # Restore result pointer
    pop rcx

    # Store result (8 limbs)
    mov [rcx+0],  r8
    mov [rcx+8],  r9
    mov [rcx+16], r10
    mov [rcx+24], r11
    mov [rcx+32], r12
    mov [rcx+40], r13
    mov [rcx+48], r14
    mov [rcx+56], r15

    pop r15
    pop r14
    pop r13
    pop r12
    pop rbp
    pop rbx
    EPILOGUE
    ret
ELF_SIZE(mul_4x4_asm)

# ==============================================================================
# sqr_4x4_asm: 256-bit Squaring -> 512-bit result
# ==============================================================================
# Optimized squaring using cross-products (10 muls instead of 16)
#
# Parameters:
#   RDI = const uint64_t* a
#   RSI = uint64_t* result
# ==============================================================================

.globl sqr_4x4_asm
ELF_TYPE(sqr_4x4_asm)
sqr_4x4_asm:
    PROLOGUE_2ARG
    push rbx
    push rbp
    push r12
    push r13
    push r14
    push r15

    # rdi = a
    # rsi = result
    
    # Clear accumulators
    xor r8, r8
    xor r9, r9
    xor r10, r10
    xor r11, r11
    xor r12, r12
    xor r13, r13
    xor r14, r14
    xor r15, r15

    # ----------------------------------------------------------------
    # 1. Cross Products (ADCX/ADOX dual carry chain)
    # ----------------------------------------------------------------
    
    # Pass A: a[0] * (a[1], a[2], a[3])
    mov rdx, [rdi]
    xor ecx, ecx          # rcx=0, CF=0, OF=0

    mulx rbx, rax, [rdi+8]
    adcx r9, rax
    adox r10, rbx

    mulx rbx, rax, [rdi+16]
    adcx r10, rax
    adox r11, rbx

    mulx rbx, rax, [rdi+24]
    adcx r11, rax
    adox r12, rbx

    adcx r12, rcx
    adox r13, rcx

    # Pass B: a[1] * (a[2], a[3])
    mov rdx, [rdi+8]
    xor ecx, ecx

    mulx rbx, rax, [rdi+16]
    adcx r11, rax
    adox r12, rbx

    mulx rbx, rax, [rdi+24]
    adcx r12, rax
    adox r13, rbx

    adcx r13, rcx
    adox r14, rcx

    # Pass C: a[2] * a[3]
    mov rdx, [rdi+16]
    xor ecx, ecx

    mulx rbx, rax, [rdi+24]
    adcx r13, rax
    adox r14, rbx

    adcx r14, rcx
    adox r15, rcx

    # ----------------------------------------------------------------
    # 2. Double the cross products (r9..r15)
    # ----------------------------------------------------------------
    add r9, r9
    adc r10, r10
    adc r11, r11
    adc r12, r12
    adc r13, r13
    adc r14, r14
    adc r15, r15
    
    # ----------------------------------------------------------------
    # 3. Add Squares (single ADCX chain -- no adc-zero tails)
    # ----------------------------------------------------------------
    xor ecx, ecx          # CF=0 for ADCX chain

    mov rdx, [rdi]
    mulx rbx, rax, rdx
    adcx r8, rax
    adcx r9, rbx

    mov rdx, [rdi+8]
    mulx rbx, rax, rdx
    adcx r10, rax
    adcx r11, rbx

    mov rdx, [rdi+16]
    mulx rbx, rax, rdx
    adcx r12, rax
    adcx r13, rbx

    mov rdx, [rdi+24]
    mulx rbx, rax, rdx
    adcx r14, rax
    adcx r15, rbx
    
    # Store result
    mov [rsi+0],  r8
    mov [rsi+8],  r9
    mov [rsi+16], r10
    mov [rsi+24], r11
    mov [rsi+32], r12
    mov [rsi+40], r13
    mov [rsi+48], r14
    mov [rsi+56], r15

    pop r15
    pop r14
    pop r13
    pop r12
    pop rbp
    pop rbx
    EPILOGUE
    ret
ELF_SIZE(sqr_4x4_asm)

# ==============================================================================
# add_4_asm: 256-bit Modular Addition (a + b mod p)
# ==============================================================================
# p = 2^256 - 2^32 - 977
#
# Parameters:
#   RDI = const uint64_t* a
#   RSI = const uint64_t* b
#   RDX = uint64_t* result
# ==============================================================================

.globl add_4_asm
ELF_TYPE(add_4_asm)
add_4_asm:
    PROLOGUE_3ARG
    push rbx
    push rbp
    
    # Load a
    mov r8, [rdi]
    mov r9, [rdi+8]
    mov r10, [rdi+16]
    mov r11, [rdi+24]
    
    # Add b
    add r8, [rsi]
    adc r9, [rsi+8]
    adc r10, [rsi+16]
    adc r11, [rsi+24]
    
    # Capture carry of S
    setc cl  # cl = 1 if S overflowed
    
    # Calculate T = S + K
    # K = 0x1000003D1
    mov rax, 0x1000003D1
    
    mov rdi, r8
    add rdi, rax
    mov rsi, r9
    adc rsi, 0
    mov rbx, r10
    adc rbx, 0
    mov rbp, r11
    adc rbp, 0
    
    # Capture carry of T
    setc al # al = 1 if T overflowed
    
    # Condition: S_carry OR T_carry
    or al, cl
    
    # If true, use T (rdi, rsi, rbx, rbp)
    # If false, use S (r8, r9, r10, r11)
    
    test al, al
    cmovnz r8, rdi
    cmovnz r9, rsi
    cmovnz r10, rbx
    cmovnz r11, rbp
    
    # Store result
    mov [rdx], r8
    mov [rdx+8], r9
    mov [rdx+16], r10
    mov [rdx+24], r11
    
    pop rbp
    pop rbx
    EPILOGUE
    ret

ELF_SIZE(add_4_asm)

# ==============================================================================
# sub_4_asm: 256-bit Modular Subtraction (a - b mod p)
# ==============================================================================
# Parameters:
#   RDI = const uint64_t* a
#   RSI = const uint64_t* b
#   RDX = uint64_t* result
# ==============================================================================

.globl sub_4_asm
ELF_TYPE(sub_4_asm)
sub_4_asm:
    PROLOGUE_3ARG
    push rbx
    push rbp
    
    # Load a
    mov r8, [rdi]
    mov r9, [rdi+8]
    mov r10, [rdi+16]
    mov r11, [rdi+24]
    
    # Subtract b
    sub r8, [rsi]
    sbb r9, [rsi+8]
    sbb r10, [rsi+16]
    sbb r11, [rsi+24]
    
    # Capture borrow
    setc cl
    
    # Calculate T = S - K
    # K = 0x1000003D1
    mov rax, 0x1000003D1
    
    mov rdi, r8
    sub rdi, rax
    mov rsi, r9
    sbb rsi, 0
    mov rbx, r10
    sbb rbx, 0
    mov rbp, r11
    sbb rbp, 0
    
    # If borrow (cl=1), use T
    test cl, cl
    cmovnz r8, rdi
    cmovnz r9, rsi
    cmovnz r10, rbx
    cmovnz r11, rbp
    
    # Store
    mov [rdx], r8
    mov [rdx+8], r9
    mov [rdx+16], r10
    mov [rdx+24], r11
    
    pop rbp
    pop rbx
    EPILOGUE
    ret

ELF_SIZE(sub_4_asm)

.text

# ==============================================================================

# ==============================================================================
# reduce_4_asm: 512-bit -> 256-bit Modular Reduction (Strictly Normalized)
# ==============================================================================
# Reduces a 512-bit number modulo P = 2^256 - 2^32 - 977
# Ensures result < P
#
# Parameters:
#   RDI = uint64_t* data (pointer to 8 limbs, result stored in first 4)
# ==============================================================================

.globl reduce_4_asm
ELF_TYPE(reduce_4_asm)
reduce_4_asm:
    PROLOGUE_1ARG
    push rbx
    push r12
    push r13
    push r14
    push r15

    # Load low 256 bits
    mov r8, [rdi]
    mov r9, [rdi+8]
    mov r10, [rdi+16]
    mov r11, [rdi+24]
    xor r12, r12  # carry accumulator

    mov rdx, 0x1000003D1  # K = 2^32 + 977

    # === Limb 4: add data[4] * K to r8..r11 ===
    mov rax, [rdi+32]
    mulx rcx, rbx, rax
    add  r8,  rbx
    adc  r9,  rcx
    adc  r10, 0
    adc  r11, 0
    adc  r12, 0

    # === Limb 5: add data[5] * K to r9..r11 ===
    mov rax, [rdi+40]
    mulx rcx, rbx, rax
    add  r9,  rbx
    adc  r10, rcx
    adc  r11, 0
    adc  r12, 0

    # === Limb 6: add data[6] * K to r10..r11 ===
    mov rax, [rdi+48]
    mulx rcx, rbx, rax
    add  r10, rbx
    adc  r11, rcx
    adc  r12, 0

    # === Limb 7: add data[7] * K to r11 ===
    mov rax, [rdi+56]
    mulx rcx, rbx, rax
    add  r11, rbx
    adc  r12, rcx

    # --- Pass 1: reduce overflow (r12 is at most ~34 bits) ---
    mulx rcx, rbx, r12
    xor  r12, r12
    add  r8,  rbx
    adc  r9,  rcx
    adc  r10, 0
    adc  r11, 0
    adc  r12, 0

    # --- Pass 2: reduce final 1-bit overflow (branchless) ---
    mov  rax, 0x1000003D1
    and  rax, r12
    add  r8,  rax
    adc  r9,  0
    adc  r10, 0
    adc  r11, 0

    # Branchless mod-p: if result >= p, subtract p
    mov r13, 0x1000003D1
    
    mov rax, r8
    add rax, r13
    mov rbx, r9
    adc rbx, 0
    mov rcx, r10
    adc rcx, 0
    mov rdx, r11
    adc rdx, 0
    
    cmovc r8, rax
    cmovc r9, rbx
    cmovc r10, rcx
    cmovc r11, rdx

    mov [rdi], r8
    mov [rdi+8], r9
    mov [rdi+16], r10
    mov [rdi+24], r11

    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    ret
ELF_SIZE(reduce_4_asm)

# ==============================================================================
# field_mul_full_asm: 256x256 -> 512 -> 256-bit Modular Multiplication
# ==============================================================================
# Combines mul_4x4_asm and reduce_4_asm into a single function.
# Avoids memory round-trip for the 512-bit intermediate result.
#
# Parameters:
#   RDI = const uint64_t* a
#   RSI = const uint64_t* b
#   RDX = uint64_t* result
# ==============================================================================

.globl field_mul_full_asm
ELF_TYPE(field_mul_full_asm)
field_mul_full_asm:
    PROLOGUE_3ARG
    push rbx
    push rbp
    push r12
    push r13
    push r14
    push r15

    # Save result pointer (rdx will be used by mulx)
    push rdx

    # ----------------------------------------------------------------
    # 1. Multiplication (256x256 -> 512) -- ADX dual carry chain
    #    MULX + ADCX (CF) + ADOX (OF) eliminates adc-zero stalls
    #    Result in r8..r15
    # ----------------------------------------------------------------

    # Zero accumulators
    xor r8, r8
    xor r9, r9
    xor r10, r10
    xor r11, r11
    xor r12, r12
    xor r13, r13
    xor r14, r14
    xor r15, r15
    xor ebp, ebp          # overflow register

    # ---- Row 0: a[0] * b[0..3] ----
    mov rdx, [rdi]
    xor ecx, ecx          # rcx=0, CF=0, OF=0

    mulx rbx, rax, [rsi]
    adcx r8, rax
    adox r9, rbx

    mulx rbx, rax, [rsi+8]
    adcx r9, rax
    adox r10, rbx

    mulx rbx, rax, [rsi+16]
    adcx r10, rax
    adox r11, rbx

    mulx rbx, rax, [rsi+24]
    adcx r11, rax
    adox r12, rbx

    adcx r12, rcx
    adox r13, rcx

    # ---- Row 1: a[1] * b[0..3] ----
    mov rdx, [rdi+8]
    xor ecx, ecx

    mulx rbx, rax, [rsi]
    adcx r9, rax
    adox r10, rbx

    mulx rbx, rax, [rsi+8]
    adcx r10, rax
    adox r11, rbx

    mulx rbx, rax, [rsi+16]
    adcx r11, rax
    adox r12, rbx

    mulx rbx, rax, [rsi+24]
    adcx r12, rax
    adox r13, rbx

    adcx r13, rcx
    adox r14, rcx

    # ---- Row 2: a[2] * b[0..3] ----
    mov rdx, [rdi+16]
    xor ecx, ecx

    mulx rbx, rax, [rsi]
    adcx r10, rax
    adox r11, rbx

    mulx rbx, rax, [rsi+8]
    adcx r11, rax
    adox r12, rbx

    mulx rbx, rax, [rsi+16]
    adcx r12, rax
    adox r13, rbx

    mulx rbx, rax, [rsi+24]
    adcx r13, rax
    adox r14, rbx

    adcx r14, rcx
    adox r15, rcx

    # ---- Row 3: a[3] * b[0..3] ----
    mov rdx, [rdi+24]
    xor ecx, ecx

    mulx rbx, rax, [rsi]
    adcx r11, rax
    adox r12, rbx

    mulx rbx, rax, [rsi+8]
    adcx r12, rax
    adox r13, rbx

    mulx rbx, rax, [rsi+16]
    adcx r13, rax
    adox r14, rbx

    mulx rbx, rax, [rsi+24]
    adcx r14, rax
    adox r15, rbx

    adcx r15, rcx
    adox rbp, rcx          # overflow (0 for 256x256)

    # ----------------------------------------------------------------
    # 2. Reduction (512 -> 256)
    #    Input: r8..r15
    #    Output: r8..r11
    # ----------------------------------------------------------------
    #
    # secp256k1: p = 2^256 - K where K = 0x1000003D1
    # So 2^256 = K (mod p), meaning high[i] * 2^(256+64i) = high[i] * K * 2^(64i)
    # Single MULX per high limb: K * high[i] -> hi:lo, add to result[i..i+1]
    #
    
    # rdi is free (inputs already loaded). Use as carry accumulator.
    xor rdi, rdi
    mov rdx, 0x1000003D1  # K constant for full reduction

    # === Limb 4 (r12): add r12 * K to r8..r11 ===
    mulx rcx, rbx, r12
    add  r8,  rbx
    adc  r9,  rcx
    adc  r10, 0
    adc  r11, 0
    adc  rdi, 0

    # === Limb 5 (r13): add r13 * K to r9..r11 ===
    mulx rcx, rbx, r13
    add  r9,  rbx
    adc  r10, rcx
    adc  r11, 0
    adc  rdi, 0

    # === Limb 6 (r14): add r14 * K to r10..r11 ===
    mulx rcx, rbx, r14
    add  r10, rbx
    adc  r11, rcx
    adc  rdi, 0

    # === Limb 7 (r15): add r15 * K to r11 ===
    mulx rcx, rbx, r15
    add  r11, rbx
    adc  rdi, rcx

    # ----------------------------------------------------------------
    # 3. Overflow reduction (unrolled, no loop)
    #    After reducing 4 high limbs, carry (rdi) is at most ~34 bits.
    #    One pass: rdi * K is at most ~67 bits; new carry <= 1 bit.
    #    Second pass: 1 * K is at most 33 bits; no further carry.
    # ----------------------------------------------------------------

    # --- Pass 1: reduce rdi overflow ---
    mulx rcx, rbx, rdi      # rdi * K -> rcx:rbx
    xor  rdi, rdi            # clear carry for pass 2
    add  r8,  rbx
    adc  r9,  rcx
    adc  r10, 0
    adc  r11, 0
    adc  rdi, 0              # rdi = 0 or 1

    # --- Pass 2: reduce final 1-bit overflow (branchless) ---
    # K * rdi where rdi is 0 or 1: result is 0 or K.
    # Use AND mask instead of MULX (cheaper for 1-bit input).
    mov  rax, 0x1000003D1
    and  rax, rdi            # rax = K if rdi=1, else 0
    add  r8,  rax
    adc  r9,  0
    adc  r10, 0
    adc  r11, 0
    # No further carry possible: max(r8..r11) + K < 2^257 in this path,
    # and the branchless mod-p check below handles values in [p, 2p).

    # ----------------------------------------------------------------
    # 4. Branchless mod-p check: if result >= p, subtract p
    #    p = 2^256 - K, so result >= p iff result + K >= 2^256 (carry set)
    # ----------------------------------------------------------------
    mov r13, 0x1000003D1
    
    mov rax, r8
    add rax, r13
    mov rbx, r9
    adc rbx, 0
    mov rcx, r10
    adc rcx, 0
    mov rdx, r11
    adc rdx, 0
    
    cmovc r8,  rax
    cmovc r9,  rbx
    cmovc r10, rcx
    cmovc r11, rdx

    # Restore result pointer
    pop rdx

    # Store result
    mov [rdx], r8
    mov [rdx+8], r9
    mov [rdx+16], r10
    mov [rdx+24], r11

    pop r15
    pop r14
    pop r13
    pop r12
    pop rbp
    pop rbx
    EPILOGUE
    ret
ELF_SIZE(field_mul_full_asm)

# ==============================================================================
# field_sqr_full_asm: 256 -> 512 -> 256-bit Modular Squaring
# ==============================================================================
# Combines sqr_4x4_asm and reduce_4_asm.
#
# Parameters:
#   RDI = const uint64_t* a
#   RSI = uint64_t* result
# ==============================================================================

.globl field_sqr_full_asm
ELF_TYPE(field_sqr_full_asm)
field_sqr_full_asm:
    PROLOGUE_2ARG
    push rbx
    push rbp
    push r12
    push r13
    push r14
    push r15

    # Save result pointer
    push rsi

    # Clear accumulators
    xor r8, r8
    xor r9, r9
    xor r10, r10
    xor r11, r11
    xor r12, r12
    xor r13, r13
    xor r14, r14
    xor r15, r15

    # ----------------------------------------------------------------
    # 1. Cross Products (ADCX/ADOX dual carry chain)
    # ----------------------------------------------------------------
    
    # Pass A: a[0] * (a[1], a[2], a[3])
    mov rdx, [rdi]
    xor ecx, ecx          # rcx=0, CF=0, OF=0

    mulx rbx, rax, [rdi+8]
    adcx r9, rax
    adox r10, rbx

    mulx rbx, rax, [rdi+16]
    adcx r10, rax
    adox r11, rbx

    mulx rbx, rax, [rdi+24]
    adcx r11, rax
    adox r12, rbx

    adcx r12, rcx
    adox r13, rcx

    # Pass B: a[1] * (a[2], a[3])
    mov rdx, [rdi+8]
    xor ecx, ecx

    mulx rbx, rax, [rdi+16]
    adcx r11, rax
    adox r12, rbx

    mulx rbx, rax, [rdi+24]
    adcx r12, rax
    adox r13, rbx

    adcx r13, rcx
    adox r14, rcx

    # Pass C: a[2] * a[3]
    mov rdx, [rdi+16]
    xor ecx, ecx

    mulx rbx, rax, [rdi+24]
    adcx r13, rax
    adox r14, rbx

    adcx r14, rcx
    adox r15, rcx

    # ----------------------------------------------------------------
    # 2. Double the cross products
    # ----------------------------------------------------------------
    add r9, r9
    adc r10, r10
    adc r11, r11
    adc r12, r12
    adc r13, r13
    adc r14, r14
    adc r15, r15
    
    # ----------------------------------------------------------------
    # 3. Add Squares (single ADCX chain -- no adc-zero tails)
    # ----------------------------------------------------------------
    xor ecx, ecx          # CF=0 for ADCX chain

    mov rdx, [rdi]
    mulx rbx, rax, rdx
    adcx r8, rax
    adcx r9, rbx

    mov rdx, [rdi+8]
    mulx rbx, rax, rdx
    adcx r10, rax
    adcx r11, rbx

    mov rdx, [rdi+16]
    mulx rbx, rax, rdx
    adcx r12, rax
    adcx r13, rbx

    mov rdx, [rdi+24]
    mulx rbx, rax, rdx
    adcx r14, rax
    adcx r15, rbx

    # ----------------------------------------------------------------
    # 4. Reduction (512 -> 256)
    # ----------------------------------------------------------------
    
    # ----------------------------------------------------------------
    # Reduction (512 -> 256) using K = 0x1000003D1
    # Single MULX per high limb, no loop, fully unrolled overflow handling
    # ----------------------------------------------------------------
    
    xor rdi, rdi
    mov rdx, 0x1000003D1  # K constant

    # === Limb 4 (r12): add r12 * K to r8..r11 ===
    mulx rcx, rbx, r12
    add  r8,  rbx
    adc  r9,  rcx
    adc  r10, 0
    adc  r11, 0
    adc  rdi, 0

    # === Limb 5 (r13): add r13 * K to r9..r11 ===
    mulx rcx, rbx, r13
    add  r9,  rbx
    adc  r10, rcx
    adc  r11, 0
    adc  rdi, 0

    # === Limb 6 (r14): add r14 * K to r10..r11 ===
    mulx rcx, rbx, r14
    add  r10, rbx
    adc  r11, rcx
    adc  rdi, 0

    # === Limb 7 (r15): add r15 * K to r11 ===
    mulx rcx, rbx, r15
    add  r11, rbx
    adc  rdi, rcx

    # --- Pass 1: reduce overflow ---
    mulx rcx, rbx, rdi
    xor  rdi, rdi
    add  r8,  rbx
    adc  r9,  rcx
    adc  r10, 0
    adc  r11, 0
    adc  rdi, 0

    # --- Pass 2: reduce final 1-bit overflow (branchless) ---
    mov  rax, 0x1000003D1
    and  rax, rdi
    add  r8,  rax
    adc  r9,  0
    adc  r10, 0
    adc  r11, 0

    # Branchless mod-p: if result >= p, subtract p
    mov r13, 0x1000003D1
    
    mov rax, r8
    add rax, r13
    mov rbx, r9
    adc rbx, 0
    mov rcx, r10
    adc rcx, 0
    mov rdx, r11
    adc rdx, 0
    
    cmovc r8, rax
    cmovc r9, rbx
    cmovc r10, rcx
    cmovc r11, rdx

    # Restore result pointer
    pop rsi

    mov [rsi], r8
    mov [rsi+8], r9
    mov [rsi+16], r10
    mov [rsi+24], r11

    pop r15
    pop r14
    pop r13
    pop r12
    pop rbp
    pop rbx
    EPILOGUE
    ret
ELF_SIZE(field_sqr_full_asm)

/* Mark stack as non-executable (ELF). No-op on Windows. */
#if !defined(_WIN32)
.section .note.GNU-stack,"",@progbits
#endif

