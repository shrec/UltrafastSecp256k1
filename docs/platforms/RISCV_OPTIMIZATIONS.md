# RISC-V Optimizations Report

**Platform:** Milk-V Mars (StarFive JH7110, SiFive U74 dual-core, 1.5 GHz)  
**ISA:** RV64GC + RVV (Vector Extension)  
**Compiler:** Clang 21.1.8, `-Ofast`, LTO  
**Last Updated:** 2026-02-15  

---

## Optimization Timeline

| Phase | Scalar Mul | Field Mul | Key Change |
|-------|-----------|-----------|------------|
| Baseline (C++ only) | ~900 us | ~300 ns | Portable C++ |
| + Assembly mul/square | 694 us | 197 ns | Comba multiply + fast reduction |
| + Dedicated square asm | 672 us | 197 ns | 10 mul vs 16 (symmetry exploit) |
| + Branchless field ops | 624 us | 174 ns | ge/add/sub/normalize branchless |
| + Direct asm calls | 624 us | 174 ns | Bypass FieldElement wrapper |
| + Branchless asm reduce | **621 us** | **173 ns** | Remove beqz/j loop from reduce |

**Total improvement: ~31% scalar mul, ~42% field mul from baseline.**

---

## 1. Assembly Multiply & Square (field_asm_riscv64.S)

### Comba Multiplication (16 -> 16 mul)

Standard 4-limb x 4-limb Comba multiplication producing 8-limb (512-bit) intermediate.

**Columns:**
```
c0 = a0*b0
c1 = a0*b1 + a1*b0
c2 = a0*b2 + a1*b1 + a2*b0
c3 = a0*b3 + a1*b2 + a2*b1 + a3*b0
c4 = a1*b3 + a2*b2 + a3*b1
c5 = a2*b3 + a3*b2
c6 = a3*b3
```

Uses `mul` / `mulhu` pairs with `sltu`-based carry propagation throughout.

### Dedicated Square (10 mul)

Exploits $a^2 = \sum a_i^2 + 2\sum_{i<j} a_i \cdot a_j$ symmetry:
- **4 diagonal:** `a0^2, a1^2, a2^2, a3^2`
- **6 off-diagonal:** `a0*a1, a0*a2, a0*a3, a1*a2, a1*a3, a2*a3`
- Doubling via add-twice (no 128-bit shift carry complexity)

**Result:** Square 186 -> 177 ns (**5% improvement**)

### Fast Reduction (mod p = 2^2⁵⁶ - 2^3^2 - 977)

Reduces [c0..c7] -> [r0..r3] using $p = 2^{256} - C$ where $C = 2^{32} + 977$:

For each high limb $c_i$ ($i = 4..7$):
```
r[i-4] += c_i * 977       (via mul/mulhu)
r[i-4] += c_i << 32       (via slli/srli)
```

With full carry propagation through the chain.

### Branchless Overflow Reduce (v2, 2026-02-15)

After first-pass reduction, overflow `s9 < 2^34`. **Previous code** had a branch loop:
```asm
# OLD (branchy):
.Lreduce_loop:
    beqz    s9, .Lfinal_check   # <- branch
    ...reduce body...
    j       .Lreduce_loop       # <- back-branch
```

**New code** executes reduce body unconditionally once (s9 -> {0,1}), then merges residual into final check:
```asm
# NEW (branchless):
    mv      t4, s9              # always execute
    li      s9, 0
    ...reduce body...           # s9 now 0 or 1

    # Final: select reduced if overflow OR residual
    or      a7, a7, s9          # <- key line
    neg     a7, a7
    # branchless XOR/AND/XOR select follows
```

**Mathematical proof:** After first-pass, $s9 < 2^{34}$. One pass of $s9 \times C$ where $C \approx 2^{32}$ produces at most $\sim 2^{66}$ which distributed across 4 limbs yields $s9' \in \{0, 1\}$. The final conditional subtract handles $s9' = 1$ via `or a7, a7, s9`.

**Result:** Mul 174 -> 173 ns, Square 162 -> 160 ns (deterministic timing, no branch variance)

---

## 2. Branchless C++ Field Operations (field.cpp)

### ge() -- Greater-or-Equal Comparison

**Before:** Branchy for-loop with early return:
```cpp
for (int i = 3; i >= 0; --i) {
    if (v[i] > other.v[i]) return true;
    if (v[i] < other.v[i]) return false;
}
```

**After:** Full subtraction chain, check borrow:
```cpp
uint64_t borrow = 0;
for (int i = 0; i < 4; ++i) {
    uint64_t diff = v[i] - other.v[i] - borrow;
    borrow = (v[i] < other.v[i] + borrow) | ...;
}
return borrow == 0;  // no borrow = a >= b
```

### add_impl -- Field Addition

**Before:** While-loop carry propagation + while-loop conditional reduction.

**After:** Single-pass add with branchless conditional subtract:
```cpp
// Add
uint64_t carry = 0;
for (int i = 0; i < 4; ++i) {
    __uint128_t s = (__uint128_t)a[i] + b[i] + carry;
    out[i] = (uint64_t)s;
    carry = (uint64_t)(s >> 64);
}

// Branchless: try subtract p, select via mask
uint64_t borrow = 0;
uint64_t reduced[4];
for (int i = 0; i < 4; ++i) { /* sub PRIME */ }

uint64_t mask = -(uint64_t)(carry | (borrow == 0));
for (int i = 0; i < 4; ++i)
    out[i] ^= (out[i] ^ reduced[i]) & mask;
```

### sub_impl -- Field Subtraction

**Before:** if-branch calling `ge()` then subtract or reverse-subtract.

**After:** Always compute `a - b`, conditionally add `p` on borrow:
```cpp
uint64_t borrow = 0;
for (int i = 0; i < 4; ++i) { /* sub chain */ }

// Branchless: add PRIME masked by borrow
uint64_t mask = -borrow;
uint64_t addend[4] = { PRIME[i] & mask };
// add addend to result
```

### normalize()

**Before:** While-loop `while (ge(PRIME))`.

**After:** Single-pass branchless conditional subtract (same XOR/AND pattern).

---

## 3. Direct ASM Calls (No Wrapper Overhead)

**Before:** C++ wrapper created `FieldElement` temporaries:
```cpp
void mul_impl(const uint64_t* a, const uint64_t* b, uint64_t* out) {
    FieldElement fa = FieldElement::from_limbs(a);  // calls normalize()!
    FieldElement fb = FieldElement::from_limbs(b);
    FieldElement result = field_mul_riscv(fa, fb);
    std::memcpy(out, result.v, 32);
}
```

**After:** Direct `extern "C"` call on raw pointers:
```cpp
extern "C" void field_mul_asm_riscv64(uint64_t* out, const uint64_t* a, const uint64_t* b);

void mul_impl(const uint64_t* a, const uint64_t* b, uint64_t* out) {
    field_mul_asm_riscv64(out, a, b);
}
```

Eliminates 2x `normalize()` + 2x `memcpy` per mul/square call.

---

## 4. wNAF Window Width (w=4 -> w=5)

**File:** `cpu/src/point.cpp`

On RISC-V (not ESP32/STM32), scalar_mul uses wNAF with w=5:
- 16 precomputed points: [1P, 3P, 5P, ..., 31P]
- Fewer non-zero digits -> fewer point additions in main loop
- Trade-off: 8 extra precomputed points (8 doublings + 8 additions) vs ~10% fewer additions in 256-bit scan

**Result:** Scalar Mul 678 -> 672 us (**~1% improvement**)

---

## 5. Optimizations Attempted But Reverted

### Hand-Written Add/Sub Assembly

Wrote `field_add_asm_riscv64` and `field_sub_asm_riscv64` in assembly, wired via `#elif defined(SECP256K1_HAS_RISCV_ASM)` in field.cpp.

**Result:** **Regression.** Field Add: 34 -> 43 ns (+26%), Field Sub: 31 -> 51 ns (+64%).

**Root Cause:** Clang 21 generates better code for simple 256-bit add/sub on U74's in-order pipeline. The compiler:
- Optimally schedules instructions to fill pipeline bubbles
- Avoids unnecessary register spills (asm used callee-saved regs)
- Inline-expands without function call overhead

**Lesson:** On in-order cores with good compilers, only complex operations (mul/square with 16+ multiplications) benefit from hand-written assembly.

---

## Key Learnings

1. **Assembly wrapper overhead matters:** For ~30ns operations, converting between `limbs4` <-> `FieldElement` costs more than the operation itself.

2. **Branchless > branchy on in-order cores:** U74 has no speculative execution -- branch misprediction flushes the entire pipeline. Even well-predicted branches add 1-2 cycles of overhead.

3. **Compiler wins for simple ops:** Clang 21 with `-Ofast` generates near-optimal code for add/sub. Only complex mul/square with carry chains benefit from hand-tuned assembly.

4. **Single-pass reduction is sufficient:** After first-pass, overflow is bounded by $2^{34}$. One unconditional pass always reduces to {0,1}. No loop needed.

5. **Binary GCD beats Fermat:** `hybrid_eea` inverse (18 us) is 3x faster than addition chain methods (~60 us) on RISC-V.

---

## Current Best Results (2026-02-15)

| Operation | Time | Implementation |
|-----------|------|---------------|
| Field Mul | 173 ns | RISC-V asm (Comba + branchless reduce) |
| Field Square | 160 ns | RISC-V asm (10 mul + branchless reduce) |
| Field Add | 38 ns | C++ branchless (compiler-optimized) |
| Field Sub | 34 ns | C++ branchless (compiler-optimized) |
| Field Inverse | 17 us | Binary GCD (hybrid_eea) |
| Point Add | 3 us | Jacobian mixed addition (7M + 4S) |
| Point Double | 1 us | Jacobian doubling (4S + 4M, a=0) |
| **Scalar Mul** | **621 us** | **GLV + Shamir + wNAF(w=5)** |
| **Generator Mul** | **37 us** | **Precomputed fixed-base table** |
| Batch Inv (n=100) | 695 ns | Montgomery's trick |
| Batch Inv (n=1000) | 547 ns | Montgomery's trick |

All 29+ tests pass [OK]

---

## Files

| File | Role |
|------|------|
| `cpu/src/field_asm_riscv64.S` | Assembly: mul, square, add, sub, negate |
| `cpu/src/field_asm_riscv64.cpp` | C++ wrappers (mul/square wrappers unused now) |
| `cpu/src/field.cpp` | Branchless ge/add/sub/normalize, direct asm calls |
| `cpu/src/point.cpp` | GLV + Shamir + wNAF(w=5) scalar mul |
| `cpu/src/scalar.cpp` | Scalar arithmetic (pure C++) |

---

## Next Steps for Cross-Platform

### Priority 1: x86-64 Field Square
```cpp
// Portable C++ implementation using __int128
inline void field_square_opt(uint64_t* r, const uint64_t* a) {
    // Diagonal terms
    __uint128_t d0 = (__uint128_t)a[0] * a[0];
    __uint128_t d1 = (__uint128_t)a[1] * a[1];
    __uint128_t d2 = (__uint128_t)a[2] * a[2];
    __uint128_t d3 = (__uint128_t)a[3] * a[3];
    
    // Off-diagonal terms (computed once, added twice)
    __uint128_t c01 = (__uint128_t)a[0] * a[1];
    __uint128_t c02 = (__uint128_t)a[0] * a[2];
    __uint128_t c03 = (__uint128_t)a[0] * a[3];
    __uint128_t c12 = (__uint128_t)a[1] * a[2];
    __uint128_t c13 = (__uint128_t)a[1] * a[3];
    __uint128_t c23 = (__uint128_t)a[2] * a[3];
    
    // Build columns and reduce...
}
```

### Priority 2: CUDA Field Square
```cuda
__device__ void field_square_opt(uint64_t* r, const uint64_t* a) {
    // Same algorithm, use __umul64hi() for high bits
    uint64_t d0_lo = a[0] * a[0];
    uint64_t d0_hi = __umul64hi(a[0], a[0]);
    // ...
}
```

---

## Conclusion

The RISC-V optimizations provide:
- **5% improvement** in field squaring
- **1.2% improvement** in scalar multiplication  
- **4-16% improvement** from LTO

All optimizations are algorithmically portable and can be implemented on x86-64 and CUDA with similar or better results due to those platforms' more advanced hardware capabilities.

