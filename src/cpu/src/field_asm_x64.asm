; ==============================================================================
; SECP256K1 Field Operations - x64 Assembly (MASM Syntax for Windows)
; ==============================================================================
; Ported from GAS (Linux) version.
;
; Calling Convention (Windows x64):
;   RCX = 1st argument
;   RDX = 2nd argument
;   R8  = 3rd argument
;   R9  = 4th argument
;
; Non-volatile registers (must be preserved):
;   RBX, RBP, RDI, RSI, RSP, R12, R13, R14, R15
;
; ==============================================================================

.code

; ==============================================================================
; mul_4x4_asm(const uint64_t* a, const uint64_t* b, uint64_t* result)
; RCX = a, RDX = b, R8 = result
; ==============================================================================
mul_4x4_asm PROC
    push rbx
    push rbp
    push rdi
    push rsi
    push r12
    push r13
    push r14
    push r15

    mov rdi, rcx  ; a
    mov rsi, rdx  ; b
    mov rdx, r8   ; result

    mov rcx, rdx  ; rcx = result pointer
    
    xor r8, r8
    xor r9, r9
    xor r10, r10
    xor r11, r11
    xor r12, r12
    xor r13, r13
    xor r14, r14
    xor r15, r15

    ; i = 0
    mov rbp, [rdi+0]
    mov rdx, [rsi+0]
    mulx rbx, rax, rbp
    add r8, rax
    adc r9, rbx
    adc r10, 0

    mov rdx, [rsi+8]
    mulx rbx, rax, rbp
    add r9, rax
    adc r10, rbx
    adc r11, 0

    mov rdx, [rsi+16]
    mulx rbx, rax, rbp
    add r10, rax
    adc r11, rbx
    adc r12, 0

    mov rdx, [rsi+24]
    mulx rbx, rax, rbp
    add r11, rax
    adc r12, rbx
    adc r13, 0
    adc r14, 0
    adc r15, 0

    ; i = 1
    mov rbp, [rdi+8]
    mov rdx, [rsi+0]
    mulx rbx, rax, rbp
    add r9, rax
    adc r10, rbx
    adc r11, 0
    adc r12, 0
    adc r13, 0

    mov rdx, [rsi+8]
    mulx rbx, rax, rbp
    add r10, rax
    adc r11, rbx
    adc r12, 0
    adc r13, 0
    adc r14, 0

    mov rdx, [rsi+16]
    mulx rbx, rax, rbp
    add r11, rax
    adc r12, rbx
    adc r13, 0
    adc r14, 0

    mov rdx, [rsi+24]
    mulx rbx, rax, rbp
    add r12, rax
    adc r13, rbx
    adc r14, 0
    adc r15, 0

    ; i = 2
    mov rbp, [rdi+16]
    mov rdx, [rsi+0]
    mulx rbx, rax, rbp
    add r10, rax
    adc r11, rbx
    adc r12, 0
    adc r13, 0
    adc r14, 0
    adc r15, 0

    mov rdx, [rsi+8]
    mulx rbx, rax, rbp
    add r11, rax
    adc r12, rbx
    adc r13, 0
    adc r14, 0
    adc r15, 0

    mov rdx, [rsi+16]
    mulx rbx, rax, rbp
    add r12, rax
    adc r13, rbx
    adc r14, 0
    adc r15, 0

    mov rdx, [rsi+24]
    mulx rbx, rax, rbp
    add r13, rax
    adc r14, rbx
    adc r15, 0

    ; i = 3
    mov rbp, [rdi+24]
    mov rdx, [rsi+0]
    mulx rbx, rax, rbp
    add r11, rax
    adc r12, rbx
    adc r13, 0
    adc r14, 0
    adc r15, 0

    mov rdx, [rsi+8]
    mulx rbx, rax, rbp
    add r12, rax
    adc r13, rbx
    adc r14, 0
    adc r15, 0

    mov rdx, [rsi+16]
    mulx rbx, rax, rbp
    add r13, rax
    adc r14, rbx
    adc r15, 0

    mov rdx, [rsi+24]
    mulx rbx, rax, rbp
    add r14, rax
    adc r15, rbx

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
    pop rsi
    pop rdi
    pop rbp
    pop rbx
    ret
mul_4x4_asm ENDP

; ==============================================================================
; sqr_4x4_asm(const uint64_t* a, uint64_t* result)
; RCX = a, RDX = result
; ==============================================================================
sqr_4x4_asm PROC
    push rbx
    push rbp
    push rdi
    push rsi
    push r12
    push r13
    push r14
    push r15

    mov rdi, rcx
    mov rsi, rdx

    xor r8, r8
    xor r9, r9
    xor r10, r10
    xor r11, r11
    xor r12, r12
    xor r13, r13
    xor r14, r14
    xor r15, r15

    mov rdx, [rdi+8]
    mulx r10, r9, [rdi]
    
    mov rdx, [rdi+16]
    mulx rbx, rax, [rdi]
    add r10, rax
    adc r11, rbx
    
    mov rdx, [rdi+24]
    mulx rbx, rax, [rdi]
    add r11, rax
    adc r12, rbx
    
    mov rdx, [rdi+16]
    mulx rbx, rax, [rdi+8]
    add r11, rax
    adc r12, rbx
    adc r13, 0
    
    mov rdx, [rdi+24]
    mulx rbx, rax, [rdi+8]
    add r12, rax
    adc r13, rbx
    adc r14, 0
    
    mov rdx, [rdi+24]
    mulx rbx, rax, [rdi+16]
    add r13, rax
    adc r14, rbx
    adc r15, 0
    
    add r9, r9
    adc r10, r10
    adc r11, r11
    adc r12, r12
    adc r13, r13
    adc r14, r14
    adc r15, r15
    
    mov rdx, [rdi]
    mulx rbx, rax, rdx
    add r8, rax
    adc r9, rbx
    adc r10, 0
    adc r11, 0
    adc r12, 0
    adc r13, 0
    adc r14, 0
    adc r15, 0
    
    mov rdx, [rdi+8]
    mulx rbx, rax, rdx
    add r10, rax
    adc r11, rbx
    adc r12, 0
    adc r13, 0
    adc r14, 0
    adc r15, 0
    
    mov rdx, [rdi+16]
    mulx rbx, rax, rdx
    add r12, rax
    adc r13, rbx
    adc r14, 0
    adc r15, 0
    
    mov rdx, [rdi+24]
    mulx rbx, rax, rdx
    add r14, rax
    adc r15, rbx
    
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
    pop rsi
    pop rdi
    pop rbp
    pop rbx
    ret
sqr_4x4_asm ENDP

; ==============================================================================
; add_4_asm(const uint64_t* a, const uint64_t* b, uint64_t* result)
; RCX = a, RDX = b, R8 = result
; ==============================================================================
add_4_asm PROC
    push rbx
    push rbp
    push rdi
    push rsi
    
    mov rdi, rcx
    mov rsi, rdx
    mov rdx, r8
    
    mov r8, [rdi]
    mov r9, [rdi+8]
    mov r10, [rdi+16]
    mov r11, [rdi+24]
    
    add r8, [rsi]
    adc r9, [rsi+8]
    adc r10, [rsi+16]
    adc r11, [rsi+24]
    
    setc cl
    
    mov rax, 01000003D1h
    
    mov rdi, r8
    add rdi, rax
    mov rsi, r9
    adc rsi, 0
    mov rbx, r10
    adc rbx, 0
    mov rbp, r11
    adc rbp, 0
    
    setc al
    or al, cl
    
    test al, al
    cmovnz r8, rdi
    cmovnz r9, rsi
    cmovnz r10, rbx
    cmovnz r11, rbp
    
    mov [rdx], r8
    mov [rdx+8], r9
    mov [rdx+16], r10
    mov [rdx+24], r11
    
    pop rsi
    pop rdi
    pop rbp
    pop rbx
    ret
add_4_asm ENDP

; ==============================================================================
; sub_4_asm(const uint64_t* a, const uint64_t* b, uint64_t* result)
; RCX = a, RDX = b, R8 = result
; ==============================================================================
sub_4_asm PROC
    push rbx
    push rbp
    push rdi
    push rsi
    
    mov rdi, rcx
    mov rsi, rdx
    mov rdx, r8
    
    mov r8, [rdi]
    mov r9, [rdi+8]
    mov r10, [rdi+16]
    mov r11, [rdi+24]
    
    sub r8, [rsi]
    sbb r9, [rsi+8]
    sbb r10, [rsi+16]
    sbb r11, [rsi+24]
    
    setc cl
    
    mov rax, 01000003D1h
    
    mov rdi, r8
    sub rdi, rax
    mov rsi, r9
    sbb rsi, 0
    mov rbx, r10
    sbb rbx, 0
    mov rbp, r11
    sbb rbp, 0
    
    test cl, cl
    cmovnz r8, rdi
    cmovnz r9, rsi
    cmovnz r10, rbx
    cmovnz r11, rbp
    
    mov [rdx], r8
    mov [rdx+8], r9
    mov [rdx+16], r10
    mov [rdx+24], r11
    
    pop rsi
    pop rdi
    pop rbp
    pop rbx
    ret
sub_4_asm ENDP

; ==============================================================================
; reduce_4_asm(uint64_t* data)
; RCX = data
; ==============================================================================
reduce_4_asm PROC
    push rbx
    push rdi
    push r12
    push r13
    push r14
    push r15

    mov rdi, rcx

    mov r8, [rdi]
    mov r9, [rdi+8]
    mov r10, [rdi+16]
    mov r11, [rdi+24]
    xor r12, r12

    mov rdx, 977

    mov rax, [rdi+32]
    mulx rcx, rbx, rax
    add r8, rbx
    adc r9, rcx
    adc r10, 0
    adc r11, 0
    adc r12, 0

    mov rbx, rax
    shl rbx, 32
    shr rax, 32
    add r8, rbx
    adc r9, rax
    adc r10, 0
    adc r11, 0
    adc r12, 0

    mov rax, [rdi+40]
    mulx rcx, rbx, rax
    add r9, rbx
    adc r10, rcx
    adc r11, 0
    adc r12, 0

    mov rbx, rax
    shl rbx, 32
    shr rax, 32
    add r9, rbx
    adc r10, rax
    adc r11, 0
    adc r12, 0

    mov rax, [rdi+48]
    mulx rcx, rbx, rax
    add r10, rbx
    adc r11, rcx
    adc r12, 0

    mov rbx, rax
    shl rbx, 32
    shr rax, 32
    add r10, rbx
    adc r11, rax
    adc r12, 0

    mov rax, [rdi+56]
    mulx rcx, rbx, rax
    add r11, rbx
    adc r12, rcx

    mov rbx, rax
    shl rbx, 32
    shr rax, 32
    add r11, rbx
    adc r12, rax

reduce_loop:
    test r12, r12
    jz check_mod_p

    mov rax, r12
    xor r12, r12
    
    mulx rcx, rbx, rax
    add r8, rbx
    adc r9, rcx
    adc r10, 0
    adc r11, 0
    adc r12, 0

    mov rbx, rax
    shl rbx, 32
    shr rax, 32
    add r8, rbx
    adc r9, rax
    adc r10, 0
    adc r11, 0
    adc r12, 0
    
    jmp reduce_loop

check_mod_p:
    mov r13, 01000003D1h
    
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
    pop rdi
    pop rbx
    ret
reduce_4_asm ENDP

; ==============================================================================
; field_mul_full_asm(const uint64_t* a, const uint64_t* b, uint64_t* result)
; RCX = a, RDX = b, R8 = result
; ==============================================================================
field_mul_full_asm PROC
    push rbx
    push rbp
    push rdi
    push rsi
    push r12
    push r13
    push r14
    push r15

    mov rdi, rcx
    mov rsi, rdx
    mov rdx, r8
    push rdx ; Save result pointer

    xor r8, r8
    xor r9, r9
    xor r10, r10
    xor r11, r11
    xor r12, r12
    xor r13, r13
    xor r14, r14
    xor r15, r15

    ; i = 0
    mov rbp, [rdi+0]
    mov rdx, [rsi+0]
    mulx rbx, rax, rbp
    add r8, rax
    adc r9, rbx
    adc r10, 0

    mov rdx, [rsi+8]
    mulx rbx, rax, rbp
    add r9, rax
    adc r10, rbx
    adc r11, 0

    mov rdx, [rsi+16]
    mulx rbx, rax, rbp
    add r10, rax
    adc r11, rbx
    adc r12, 0

    mov rdx, [rsi+24]
    mulx rbx, rax, rbp
    add r11, rax
    adc r12, rbx
    adc r13, 0
    adc r14, 0
    adc r15, 0

    ; i = 1
    mov rbp, [rdi+8]
    mov rdx, [rsi+0]
    mulx rbx, rax, rbp
    add r9, rax
    adc r10, rbx
    adc r11, 0
    adc r12, 0
    adc r13, 0

    mov rdx, [rsi+8]
    mulx rbx, rax, rbp
    add r10, rax
    adc r11, rbx
    adc r12, 0
    adc r13, 0
    adc r14, 0

    mov rdx, [rsi+16]
    mulx rbx, rax, rbp
    add r11, rax
    adc r12, rbx
    adc r13, 0
    adc r14, 0

    mov rdx, [rsi+24]
    mulx rbx, rax, rbp
    add r12, rax
    adc r13, rbx
    adc r14, 0
    adc r15, 0

    ; i = 2
    mov rbp, [rdi+16]
    mov rdx, [rsi+0]
    mulx rbx, rax, rbp
    add r10, rax
    adc r11, rbx
    adc r12, 0
    adc r13, 0
    adc r14, 0
    adc r15, 0

    mov rdx, [rsi+8]
    mulx rbx, rax, rbp
    add r11, rax
    adc r12, rbx
    adc r13, 0
    adc r14, 0
    adc r15, 0

    mov rdx, [rsi+16]
    mulx rbx, rax, rbp
    add r12, rax
    adc r13, rbx
    adc r14, 0
    adc r15, 0

    mov rdx, [rsi+24]
    mulx rbx, rax, rbp
    add r13, rax
    adc r14, rbx
    adc r15, 0

    ; i = 3
    mov rbp, [rdi+24]
    mov rdx, [rsi+0]
    mulx rbx, rax, rbp
    add r11, rax
    adc r12, rbx
    adc r13, 0
    adc r14, 0
    adc r15, 0

    mov rdx, [rsi+8]
    mulx rbx, rax, rbp
    add r12, rax
    adc r13, rbx
    adc r14, 0
    adc r15, 0

    mov rdx, [rsi+16]
    mulx rbx, rax, rbp
    add r13, rax
    adc r14, rbx
    adc r15, 0

    mov rdx, [rsi+24]
    mulx rbx, rax, rbp
    add r14, rax
    adc r15, rbx

    ; Reduction
    xor rdi, rdi
    mov rdx, 977

    mov rax, r12
    mulx rcx, rbx, rax
    add r8, rbx
    adc r9, rcx
    adc r10, 0
    adc r11, 0
    adc rdi, 0

    mov rbx, rax
    shl rbx, 32
    shr rax, 32
    add r8, rbx
    adc r9, rax
    adc r10, 0
    adc r11, 0
    adc rdi, 0

    mov rax, r13
    mulx rcx, rbx, rax
    add r9, rbx
    adc r10, rcx
    adc r11, 0
    adc rdi, 0

    mov rbx, rax
    shl rbx, 32
    shr rax, 32
    add r9, rbx
    adc r10, rax
    adc r11, 0
    adc rdi, 0

    mov rax, r14
    mulx rcx, rbx, rax
    add r10, rbx
    adc r11, rcx
    adc rdi, 0

    mov rbx, rax
    shl rbx, 32
    shr rax, 32
    add r10, rbx
    adc r11, rax
    adc rdi, 0

    mov rax, r15
    mulx rcx, rbx, rax
    add r11, rbx
    adc rdi, rcx

    mov rbx, rax
    shl rbx, 32
    shr rax, 32
    add r11, rbx
    adc rdi, rax

full_reduce_loop:
    test rdi, rdi
    jz full_check_mod_p

    mov rax, rdi
    xor rdi, rdi

    mulx rcx, rbx, rax
    add r8, rbx
    adc r9, rcx
    adc r10, 0
    adc r11, 0
    adc rdi, 0

    mov rbx, rax
    shl rbx, 32
    shr rax, 32
    add r8, rbx
    adc r9, rax
    adc r10, 0
    adc r11, 0
    adc rdi, 0
    
    jmp full_reduce_loop

full_check_mod_p:
    mov r13, 01000003D1h
    
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

    pop rdx ; Restore result pointer

    mov [rdx], r8
    mov [rdx+8], r9
    mov [rdx+16], r10
    mov [rdx+24], r11

    pop r15
    pop r14
    pop r13
    pop r12
    pop rsi
    pop rdi
    pop rbp
    pop rbx
    ret
field_mul_full_asm ENDP

; ==============================================================================
; field_sqr_full_asm(const uint64_t* a, uint64_t* result)
; RCX = a, RDX = result
; ==============================================================================
field_sqr_full_asm PROC
    push rbx
    push rbp
    push rdi
    push rsi
    push r12
    push r13
    push r14
    push r15

    mov rdi, rcx
    mov rsi, rdx
    push rsi ; Save result pointer

    xor r8, r8
    xor r9, r9
    xor r10, r10
    xor r11, r11
    xor r12, r12
    xor r13, r13
    xor r14, r14
    xor r15, r15

    mov rdx, [rdi+8]
    mulx r10, r9, [rdi]
    
    mov rdx, [rdi+16]
    mulx rbx, rax, [rdi]
    add r10, rax
    adc r11, rbx
    
    mov rdx, [rdi+24]
    mulx rbx, rax, [rdi]
    add r11, rax
    adc r12, rbx
    
    mov rdx, [rdi+16]
    mulx rbx, rax, [rdi+8]
    add r11, rax
    adc r12, rbx
    adc r13, 0
    
    mov rdx, [rdi+24]
    mulx rbx, rax, [rdi+8]
    add r12, rax
    adc r13, rbx
    adc r14, 0
    
    mov rdx, [rdi+24]
    mulx rbx, rax, [rdi+16]
    add r13, rax
    adc r14, rbx
    adc r15, 0
    
    add r9, r9
    adc r10, r10
    adc r11, r11
    adc r12, r12
    adc r13, r13
    adc r14, r14
    adc r15, r15
    
    mov rdx, [rdi]
    mulx rbx, rax, rdx
    add r8, rax
    adc r9, rbx
    adc r10, 0
    adc r11, 0
    adc r12, 0
    adc r13, 0
    adc r14, 0
    adc r15, 0
    
    mov rdx, [rdi+8]
    mulx rbx, rax, rdx
    add r10, rax
    adc r11, rbx
    adc r12, 0
    adc r13, 0
    adc r14, 0
    adc r15, 0
    
    mov rdx, [rdi+16]
    mulx rbx, rax, rdx
    add r12, rax
    adc r13, rbx
    adc r14, 0
    adc r15, 0
    
    mov rdx, [rdi+24]
    mulx rbx, rax, rdx
    add r14, rax
    adc r15, rbx

    ; Reduction
    xor rdi, rdi
    mov rdx, 977

    mov rax, r12
    mulx rcx, rbx, rax
    add r8, rbx
    adc r9, rcx
    adc r10, 0
    adc r11, 0
    adc rdi, 0

    mov rbx, rax
    shl rbx, 32
    shr rax, 32
    add r8, rbx
    adc r9, rax
    adc r10, 0
    adc r11, 0
    adc rdi, 0

    mov rax, r13
    mulx rcx, rbx, rax
    add r9, rbx
    adc r10, rcx
    adc r11, 0
    adc rdi, 0

    mov rbx, rax
    shl rbx, 32
    shr rax, 32
    add r9, rbx
    adc r10, rax
    adc r11, 0
    adc rdi, 0

    mov rax, r14
    mulx rcx, rbx, rax
    add r10, rbx
    adc r11, rcx
    adc rdi, 0

    mov rbx, rax
    shl rbx, 32
    shr rax, 32
    add r10, rbx
    adc r11, rax
    adc rdi, 0

    mov rax, r15
    mulx rcx, rbx, rax
    add r11, rbx
    adc rdi, rcx

    mov rbx, rax
    shl rbx, 32
    shr rax, 32
    add r11, rbx
    adc rdi, rax

sqr_reduce_loop:
    test rdi, rdi
    jz sqr_check_mod_p

    mov rax, rdi
    xor rdi, rdi
    
    mulx rcx, rbx, rax
    add r8, rbx
    adc r9, rcx
    adc r10, 0
    adc r11, 0
    adc rdi, 0

    mov rbx, rax
    shl rbx, 32
    shr rax, 32
    add r8, rbx
    adc r9, rax
    adc r10, 0
    adc r11, 0
    adc rdi, 0
    
    jmp sqr_reduce_loop

sqr_check_mod_p:
    mov r13, 01000003D1h
    
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

    pop rsi ; Restore result pointer

    mov [rsi], r8
    mov [rsi+8], r9
    mov [rsi+16], r10
    mov [rsi+24], r11

    pop r15
    pop r14
    pop r13
    pop r12
    pop rsi
    pop rdi
    pop rbp
    pop rbx
    ret
field_sqr_full_asm ENDP

END
