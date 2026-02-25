# CPU Guide

Detailed guide for the CPU implementation of UltrafastSecp256k1.

---

## Supported Platforms

| Platform | Assembly | SIMD | Field Repr | Status |
|----------|----------|------|------------|--------|
| x86-64 Linux/Windows | BMI2/ADX | AVX2 | 5x52 | Production |
| ARM64 (Cortex-A55/A76) | MUL/UMULH | NEON | 10x26 | Production |
| RISC-V 64 (RV64GC) | Zba/Zbb | RVV 1.0 | 4x64 | Production |
| Android ARM64 (arm64-v8a) | MUL/UMULH | NEON + Crypto | 10x26 | Production |
| ESP32-S3 (Xtensa LX7) | -- | -- | 10x26 | Production |
| ESP32 (Xtensa LX6) | -- | -- | 10x26 | Production |
| STM32F103 (Cortex-M3) | UMULL/ADDS | -- | 10x26 | Production |

---

## x86-64 Optimizations

### BMI2/ADX Instructions

The x86-64 implementation uses specialized instructions for 256-bit arithmetic:

- **MULX**: Multiply without affecting flags
- **ADCX**: Add with carry (carry chain)
- **ADOX**: Add with overflow (parallel chain)

These enable efficient carry-chain multiplication with two parallel addition chains.

### 5x52 Field Representation

Since v3.10, x86-64 uses the **5x52** field representation with `__int128` lazy reduction. This provides ~2.8x speedup over the previous 4x64 representation:

| Operation | 4x64 | 5x52 | Speedup |
|-----------|------:|------:|--------:|
| Multiplication | 42 ns | 15 ns | **2.76x** |
| Squaring | 31 ns | 13 ns | **2.44x** |
| Addition | 4.3 ns | 1.6 ns | **2.69x** |

### Build for x86-64

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DSECP256K1_USE_ASM=ON

cmake --build build -j
```

### Performance (x86-64, Clang 21)

| Operation | Time |
|-----------|------|
| Field Mul | 17 ns |
| Field Square | 14 ns |
| Field Add | 1 ns |
| Field Inverse | 1 us |
| Point Add (effective-affine) | 159 ns |
| Point Double | 100 ns |
| Scalar Mul (k*P, GLV) | 25 us |
| Generator Mul (k*G) | 5 us |

### x86-64 Signature Operations

| Operation | Time | Throughput |
|-----------|------:|----------:|
| ECDSA Sign (RFC 6979) | 8.5 us | 118,000 op/s |
| ECDSA Verify | 23.6 us | 42,400 op/s |
| Schnorr Sign (BIP-340) | 6.8 us | 146,000 op/s |
| Schnorr Verify (BIP-340) | 24.0 us | 41,600 op/s |
| Key Generation (CT) | 9.5 us | 105,500 op/s |
| Key Generation (fast) | 5.5 us | 182,000 op/s |
| ECDH | 23.9 us | 41,800 op/s |

---

## ARM64 Optimizations

### Inline Assembly

The ARM64 implementation includes hand-optimized inline assembly (`cpu/src/field_asm_arm64.cpp`):

- **`field_mul_arm64`** -- 4x4 schoolbook MUL/UMULH + secp256k1 fast reduction
- **`field_sqr_arm64`** -- Optimized squaring (10 mul vs 16)
- **`field_add_arm64`** -- ADDS/ADCS + branchless normalization
- **`field_sub_arm64`** -- SUBS/SBCS + conditional add p
- **`field_neg_arm64`** -- Branchless `p - a` with CSEL

Additional hardware features:
- **NEON**: 128-bit SIMD (implicit in ARMv8-A)
- **Crypto extensions**: AES/SHA hardware acceleration
- **`__int128`**: 64x64->128 multiply for scalar/field operations

### 10x26 Field Representation

ARM64 uses the **10x26** field representation, which was verified as optimal for Cortex-A76 (74 ns mul vs 100 ns with 5x52). This is because the 10x26 representation avoids the `__int128` dependency for reduction and works better with the ARM64 MUL/UMULH pipeline.

### Build for ARM64

```bash
# Native build (on ARM64 host)
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# Android cross-build
cmake -S android -B build-android-arm64 \
    -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-24 \
    -DANDROID_STL=c++_static \
    -DCMAKE_BUILD_TYPE=Release \
    -G Ninja
cmake --build build-android-arm64 -j
```

### Performance (ARM64, RK3588 Cortex-A76)

| Operation | ARM64 ASM | Generic C++ | Speedup |
|-----------|----------:|------------:|--------:|
| Field Mul | 74 ns | ~350 ns | ~4.7x |
| Field Square | 50 ns | ~280 ns | ~5.6x |
| Field Add | 8 ns | ~30 ns | ~3.8x |
| Field Sub | 8 ns | ~28 ns | ~3.5x |
| Field Inverse | 2 us | ~11 us | ~5.5x |
| Scalar Mul (k*G) | 14 us | ~70 us | ~5x |
| Scalar Mul (k*P) | 131 us | ~400 us | ~3x |

---

## RISC-V Optimizations

### Supported Extensions

- **RV64GC**: Base 64-bit with compressed instructions
- **Zba**: Bit-manipulation address generation
- **Zbb**: Bit-manipulation basic operations
- **RVV 1.0**: Vector extension for batch operations (optional)

### v3.11.0 Improvements

Since v3.11.0, RISC-V benefits from:

1. **Auto-detect CPU** -- CMake reads `/proc/cpuinfo` uarch field to set `-mcpu=sifive-u74` automatically
2. **ThinLTO propagation** -- ARCH_FLAGS propagated via INTERFACE compile+link options
3. **Zba/Zbb extensions** -- Explicit `-march=rv64gc_zba_zbb` alongside `-mcpu`
4. **Effective-affine GLV** -- Batch-normalize P-multiples to affine in scalar_mul_glv52

These changes combine for a **28-34% speedup** on Milk-V Mars (Scalar Mul 235->154 us).

### Assembly Optimizations

The RISC-V implementation includes hand-optimized assembly for:

1. **Field Multiplication** -- Optimized carry chain
2. **Field Squaring** -- Dedicated routine (25% fewer muls)
3. **Field Add/Sub** -- Branchless implementation
4. **Modular Reduction** -- Fast reduction for secp256k1

> **Note:** Since v3.11.0, C++ `__int128` inline code is 26-33% faster than hand-written FE52 assembly on RISC-V, so FE52 asm is disabled by default.

### Build for RISC-V

**Native build:**
```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=clang-21 \
  -DCMAKE_CXX_COMPILER=clang++-21

cmake --build build -j
```

**Cross-compilation:**
```bash
cmake -S . -B build-riscv -G Ninja \
  -DCMAKE_SYSTEM_NAME=Linux \
  -DCMAKE_SYSTEM_PROCESSOR=riscv64 \
  -DCMAKE_C_COMPILER=riscv64-linux-gnu-gcc \
  -DCMAKE_CXX_COMPILER=riscv64-linux-gnu-g++

cmake --build build-riscv -j
```

### Performance (RISC-V, Milk-V Mars)

| Operation | Time |
|-----------|------|
| Field Mul | 95 ns |
| Field Square | 70 ns |
| Field Add | 11 ns |
| Field Inverse | 4 us |
| Point Add | 1 us |
| Scalar Mul (k*P) | 154 us |
| Generator Mul (k*G) | 33 us |

### RISC-V Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `SECP256K1_RISCV_FAST_REDUCTION` | ON | Fast modular reduction |
| `SECP256K1_RISCV_USE_VECTOR` | ON | RVV vectorization |
| `SECP256K1_RISCV_USE_PREFETCH` | ON | Memory prefetch hints |

---

## Algorithm Details

### Scalar Multiplication

Uses GLV (Gallant-Lambert-Vanstone) endomorphism with wNAF encoding:

1. **GLV Decomposition**: Split k into k1, k2 where k = k1 + lambda*k2
2. **wNAF Encoding**: Convert to signed digit form
3. **Shamir's Trick**: Compute k1*P + k2*phi(P) simultaneously
4. **Effective-Affine Table** (v3.11+): Batch-normalize P-multiples to affine, eliminating Z-coordinate arithmetic from the main loop

This reduces scalar multiplication from 256 to ~128 point operations.

### KPlan Optimization

For fixed-K * variable-Q pattern:

```cpp
// Precompute K-dependent work once
Scalar K = Scalar::from_hex("...");
KPlan plan = KPlan::from_scalar(K);

// Fast multiplication for each Q
for (auto& Q : points) {
    Point R = Q.scalar_mul_with_plan(plan);  // Uses cached decomposition
}
```

### Field Arithmetic

| Platform | Representation | Reduction | Notes |
|----------|---------------|-----------|-------|
| x86-64 | 5x52 | `__int128` lazy | Best for BMI2/ADX pipeline |
| ARM64 | 10x26 | No `__int128` needed | Best for MUL/UMULH pipeline |
| RISC-V | 4x64 | `__int128` | Standard layout |
| Embedded | 10x26 | Portable C++ | No `__int128` required |

---

## Memory Layout

### FieldElement (4x64)
```
limbs[0]: bits   0-63  (least significant)
limbs[1]: bits  64-127
limbs[2]: bits 128-191
limbs[3]: bits 192-255 (most significant)
```

### Endianness

- **Internal**: Little-endian (native x86/RISC-V/ARM64)
- **`from_limbs()`**: Little-endian input (for binary I/O)
- **`from_bytes()`**: Big-endian input (for hex/test vectors)

---

## Best Practices

### 1. Use In-Place Operations

```cpp
// Slower: creates temporary
point = point.add(other);

// Faster: no allocation (~12% speedup)
point.add_inplace(other);
```

### 2. Use KPlan for Fixed-K

```cpp
// If K is constant and Q varies
KPlan plan = KPlan::from_scalar(K);
for (auto& Q : points) {
    R = Q.scalar_mul_with_plan(plan);
}
```

### 3. Batch Inversions

```cpp
// Single inversion + 3(n-1) multiplications
// vs n inversions
std::vector<FieldElement> values = {...};
batch_invert(values);
```

### 4. Avoid Unnecessary Conversions

```cpp
// For binary files, use from_limbs (native)
FieldElement::from_limbs(limbs);

// Only use from_bytes for hex/test vectors
FieldElement::from_bytes(bytes);
```

---

## Constant-Time (CT) Layer

The CT layer provides **side-channel resistant** operations for use with secret data. It lives in `secp256k1::ct::` and is always compiled alongside `fast::` -- no build flags needed.

### Architecture

```
secp256k1::fast::  <-- Maximum throughput (variable-time)
    FieldElement, Scalar, Point  <-- Shared data types
secp256k1::ct::    <-- Side-channel resistant (constant-time)
```

- **Same data types**: `ct::` functions accept and return `fast::FieldElement`, `fast::Scalar`, `fast::Point`
- **Freely mixable**: Use `fast::` for public data, `ct::` for secret-dependent operations
- **No compile flags**: Both namespaces always available on all platforms

### CT Guarantees

| Property | Guarantee |
|----------|-----------|
| Branches | No secret-dependent branches |
| Memory access | No secret-dependent access patterns |
| Instruction count | Fixed regardless of input |
| Compiler safety | `value_barrier()` prevents branch conversion |

### Key Algorithms

#### Complete Addition Formula

Unlike `fast::Point::add()` which has separate codepaths for P+Q vs P+P, the CT `point_add_complete()` handles **all cases** in a single branchless codepath:

- **P + Q** (general addition)
- **P + P** (doubling -- detected via H==0 && R==0)
- **P + O** or **O + Q** (identity -- selected via cmov)
- **P + (-P) = O** (inverse -- detected via H==0 && R!=0)

Cost: ~16M + 6S (fixed, no branches on point values)

#### CT Scalar Multiplication

Fixed-window method (w=4) with CT table lookup:

1. Precompute table: T[0]=O, T[1]=P, ..., T[15]=15P
2. For each 4-bit window (64 windows, MSB to LSB):
   - 4 doublings
   - CT table lookup (always scans all 16 entries)
   - CT conditional add

#### CT Generator Multiplication

Uses precomputed 16-entry G-table with batch inversion for fast k*G:

- **x86-64**: 9.9 us (vs 5.3 us fast, 1.86x overhead)
- **ARM64**: CT available via JNI

### CT Overhead Summary

| Operation | Fast (x86-64) | CT (x86-64) | Overhead |
|-----------|------:|------:|--------:|
| Field Mul | 17 ns | 23 ns | 1.08x |
| Field Inverse | 0.8 us | 1.7 us | 2.05x |
| Scalar Mul (k*P) | 23.6 us | 26.6 us | 1.13x |
| Generator Mul (k*G) | 5.3 us | 9.9 us | 1.86x |

---

## See Also

- [[Benchmarks]] - Full benchmark results
- [[API Reference]] - Function documentation
- [[CUDA Guide]] - GPU implementation
- [[Android Guide]] - Android port details
