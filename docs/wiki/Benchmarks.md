# Benchmarks

Performance measurements for UltrafastSecp256k1 across all platforms.

All results from **v3.11.0** (2026-02-23). Benchmarks use IQR outlier removal and median reporting.

---

## Summary

### CPU

| Operation | x86-64 (Clang 21, AVX2) | ARM64 (Cortex-A76) | RISC-V (Milk-V Mars) |
|-----------|-------------------------:|--------------------:|---------------------:|
| Field Mul | 17 ns | 74 ns | 95 ns |
| Field Square | 14 ns | 50 ns | 70 ns |
| Field Add | 1 ns | 8 ns | 11 ns |
| Field Inverse | 1 us | 2 us | 4 us |
| Point Add | 159 ns | 992 ns | 1 us |
| Generator Mul (k*G) | 5 us | 14 us | 33 us |
| Scalar Mul (k*P) | 25 us | 131 us | 154 us |

### GPU

| Backend | kG/s | ECDSA Sign | ECDSA Verify | Schnorr Sign | Schnorr Verify |
|---------|------|------------|--------------|--------------|----------------|
| **CUDA** (RTX 5060 Ti) | 4.59 M/s | 4.88 M/s | 2.44 M/s | 3.66 M/s | 2.82 M/s |
| **OpenCL** (RTX 5060 Ti) | 3.39 M/s | -- | -- | -- | -- |
| **Metal** (M3 Pro) | 0.33 M/s | -- | -- | -- | -- |

### Embedded

| Operation | ESP32-S3 (240 MHz) | ESP32 (240 MHz) | STM32F103 (72 MHz) |
|-----------|-------------------:|-------------------:|-------------------:|
| Field Mul | 6,105 ns | 6,993 ns | 15,331 ns |
| Field Square | 5,020 ns | 6,247 ns | 12,083 ns |
| Field Add | 850 ns | 985 ns | 4,139 ns |
| Field Inv | 2,524 us | 609 us | 1,645 us |
| Fast Scalar*G | 5,226 us | 6,203 us | 37,982 us |

---

## x86-64 Results

**Hardware:** Intel Core i5 (AVX2, BMI2, ADX)
**OS:** Linux
**Compiler:** Clang 21
**Field Representation:** 5x52 with `__int128` lazy reduction

| Operation | Time | Throughput |
|-----------|------|------------|
| Field Mul | 17 ns | 58.8 M/s |
| Field Square | 14 ns | 71.4 M/s |
| Field Add | 1 ns | 1,000 M/s |
| Field Inverse | 1 us | 1 M/s |
| Point Add (effective-affine) | 159 ns | 6.3 M/s |
| Point Double | 100 ns | 10 M/s |
| Scalar Mul (k*P, GLV) | 25 us | 40 K/s |
| Generator Mul (k*G, precomputed) | 5 us | 200 K/s |

### Signature Operations (x86-64)

| Operation | Time | Throughput |
|-----------|------:|----------:|
| ECDSA Sign (RFC 6979) | 8.5 us | 118,000 op/s |
| ECDSA Verify | 23.6 us | 42,400 op/s |
| Schnorr Sign (BIP-340) | 6.8 us | 146,000 op/s |
| Schnorr Verify (BIP-340) | 24.0 us | 41,600 op/s |
| Key Generation (CT) | 9.5 us | 105,500 op/s |
| Key Generation (fast) | 5.5 us | 182,000 op/s |
| ECDH | 23.9 us | 41,800 op/s |

### CT Layer Overhead (x86-64)

| Operation | Fast | CT | Overhead |
|-----------|------:|------:|--------:|
| Field Mul | 17 ns | 23 ns | 1.08x |
| Field Inverse | 0.8 us | 1.7 us | 2.05x |
| Complete Addition | -- | 276 ns | -- |
| Scalar Mul (k*P) | 23.6 us | 26.6 us | 1.13x |
| Generator Mul (k*G) | 5.3 us | 9.9 us | 1.86x |

### Field Representation: 5x52 vs 4x64

| Operation | 4x64 | 5x52 | Speedup |
|-----------|------:|------:|--------:|
| Multiplication | 42 ns | 15 ns | **2.76x** |
| Squaring | 31 ns | 13 ns | **2.44x** |
| Addition | 4.3 ns | 1.6 ns | **2.69x** |
| Add chain (32 ops) | 286 ns | 57 ns | **5.01x** |

---

## ARM64 Results

**Hardware:** RK3588, Cortex-A55/A76 @ 2.256 GHz
**OS:** Linux
**Compiler:** GCC 13 / Clang (NDK r27c)
**Field Representation:** 10x26 (optimal for Cortex-A76)

| Operation | ARM64 ASM | Generic C++ | Speedup |
|-----------|----------:|------------:|--------:|
| Field Mul | 74 ns | ~350 ns | ~4.7x |
| Field Square | 50 ns | ~280 ns | ~5.6x |
| Field Add | 8 ns | ~30 ns | ~3.8x |
| Field Sub | 8 ns | ~28 ns | ~3.5x |
| Field Inverse | 2 us | ~11 us | ~5.5x |
| Scalar Mul (k*G) | 14 us | ~70 us | ~5x |
| Scalar Mul (k*P) | 131 us | ~400 us | ~3x |

### ARM64 Signature Operations

| Operation | Time |
|-----------|------|
| ECDSA Sign | 30 us |
| Scalar Mul (k*G, fast) | 14 us |
| Scalar Mul (k*P, fast) | 131 us |
| ECDH (CT) | 545 us |

---

## RISC-V Results

**Hardware:** Milk-V Mars (SiFive U74, RV64GC + Zba + Zbb)
**OS:** Linux
**Compiler:** Clang 21 with `-mcpu=sifive-u74`
**Optimizations:** ThinLTO, auto-detected CPU, effective-affine GLV

| Operation | Time | Throughput |
|-----------|------|------------|
| Field Mul | 95 ns | 10.5 M/s |
| Field Square | 70 ns | 14.3 M/s |
| Field Add | 11 ns | 90.9 M/s |
| Field Inverse | 4 us | 250 K/s |
| Point Add | 1 us | 1 M/s |
| Generator Mul (k*G) | 33 us | 30 K/s |
| Scalar Mul (k*P) | 154 us | 6.5 K/s |

### RISC-V Optimization History

| Version | Field Mul | Scalar Mul | Notes |
|---------|-----------|------------|-------|
| v3.6 | 307 ns | 954 us | Initial implementation |
| v3.8 | 205 ns | 676 us | Carry chain optimization |
| v3.10 | 198 ns | 672 us | Square optimization |
| v3.11 | 95 ns | 154 us | Auto-detect CPU, ThinLTO, Zba/Zbb, effective-affine GLV |

**Total Improvement:** 3.23x faster (Field Mul), 6.19x faster (Scalar Mul).

---

## CUDA Results

**Hardware:** NVIDIA RTX 5060 Ti (36 SMs, 2602 MHz, 16 GB GDDR7, 128-bit bus)
**CUDA:** 12.0, sm_86;sm_89
**Build:** Clang 19 + nvcc, Release, -O3 --use_fast_math

### Core ECC Operations

| Operation | Batch Size | Time/Op | Throughput |
|-----------|------------|---------|------------|
| Field Mul | 1M | 0.2 ns | 4,142 M/s |
| Field Add | 1M | 0.2 ns | 4,130 M/s |
| Field Inv | 64K | 10.2 ns | 98.35 M/s |
| Point Add | 256K | 1.6 ns | 619 M/s |
| Point Double | 256K | 0.8 ns | 1,282 M/s |
| Scalar Mul (P*k) | 64K | 225.8 ns | 4.43 M/s |
| Generator Mul (G*k) | 128K | 217.7 ns | 4.59 M/s |
| Affine Add | 256K | 0.4 ns | 2,532 M/s |
| Batch Inv | 64K | 2.9 ns | 340 M/s |
| Jac->Affine | 64K | 14.9 ns | 66.9 M/s |

### GPU Signature Operations

> **No other open-source GPU library provides secp256k1 ECDSA + Schnorr sign/verify.**

| Operation | Batch Size | Time/Op | Throughput |
|-----------|------------|---------|------------|
| ECDSA Sign (RFC 6979) | 16K | 204.8 ns | 4.88 M/s |
| ECDSA Verify (Shamir + GLV) | 16K | 410.1 ns | 2.44 M/s |
| ECDSA Sign + Recid | 16K | 311.5 ns | 3.21 M/s |
| Schnorr Sign (BIP-340) | 16K | 273.4 ns | 3.66 M/s |
| Schnorr Verify (BIP-340) | 16K | 354.6 ns | 2.82 M/s |

### CUDA vs OpenCL Comparison (RTX 5060 Ti)

| Operation | CUDA | OpenCL | Winner |
|-----------|------|--------|--------|
| Field Mul | 0.2 ns | 0.2 ns | Tie |
| Field Inv | 10.2 ns | 14.3 ns | **CUDA 1.40x** |
| Point Double | 0.8 ns | 0.9 ns | **CUDA 1.13x** |
| Point Add | 1.6 ns | 1.6 ns | Tie |
| kG (Generator Mul) | 217.7 ns | 295.1 ns | **CUDA 1.36x** |

---

## Apple Metal Results (M3 Pro)

**Hardware:** Apple M3 Pro (18 GPU cores, Unified Memory 18 GB)
**Metal:** 2.4, 8x32-bit Comba limbs

| Operation | Time/Op | Throughput |
|-----------|---------|------------|
| Field Mul | 1.9 ns | 527 M/s |
| Field Inv | 106.4 ns | 9.40 M/s |
| Point Add | 10.1 ns | 98.6 M/s |
| Point Double | 5.1 ns | 196 M/s |
| Scalar Mul (P*k) | 2.94 us | 0.34 M/s |
| Generator Mul (G*k) | 3.00 us | 0.33 M/s |

---

## Embedded Results

### ESP32-S3 (Xtensa LX7, 240 MHz)

| Operation | Time |
|-----------|------|
| Field Mul | 6,105 ns |
| Field Square | 5,020 ns |
| Field Add | 850 ns |
| Field Inv | 2,524 us |
| Fast Scalar*G | 5,226 us |
| CT Scalar*G | 15,527 us |
| CT Generator*k | 4,951 us |

### ESP32 (Xtensa LX6, 240 MHz)

| Operation | Time |
|-----------|------|
| Field Mul | 6,993 ns |
| Field Square | 6,247 ns |
| Field Add | 985 ns |
| Field Inv | 609 us |
| Fast Scalar*G | 6,203 us |

### STM32F103 (ARM Cortex-M3, 72 MHz)

| Operation | Time |
|-----------|------|
| Field Mul | 15,331 ns |
| Field Square | 12,083 ns |
| Field Add | 4,139 ns |
| Field Inv | 1,645 us |
| Fast Scalar*G | 37,982 us |

---

## Key Optimizations

### Algorithm Level

| Optimization | Speedup | Description |
|--------------|---------|-------------|
| GLV Endomorphism | 1.5x | Reduces scalar bits by half |
| Effective-Affine Tables | 3.3x | Batch-normalize P-multiples to skip Z-coord arithmetic |
| wNAF Encoding | 1.3x | 33% fewer non-zero digits |
| Precomputed G Table | 10x | Generator multiplication |
| 5x52 Field Representation | 2.8x | `__int128` lazy reduction on 64-bit platforms |
| SafeGCD30 Inverse | 25x | GCD-based inverse for embedded (no `__int128`) |

### Implementation Level

| Optimization | Speedup | Platform |
|--------------|---------|----------|
| BMI2/ADX Assembly | 3x | x86-64 |
| MUL/UMULH Inline ASM | 5x | ARM64 |
| ThinLTO + mcpu auto-detect | 1.3x | RISC-V |
| Dedicated Squaring | 1.25x | All |
| Branchless Add/Sub | 1.2x | RISC-V, ARM64 |
| 32-bit Hybrid Mul | 1.1x | CUDA |
| Batch Inversion | 500x | All (N=1024) |
| `noinline` I-cache opt | 1.6x | x86-64 CT |

---

## Running Benchmarks

### CPU Benchmark

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSECP256K1_BUILD_BENCH=ON
cmake --build build -j

./build/cpu/bench_unified
./build/cpu/bench_ct                   # Fast vs CT comparison
./build/cpu/bench_field_52              # Field arithmetic (5x52)
./build/cpu/bench_kP                    # Scalar multiplication k*P
```

### CUDA Benchmark

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DSECP256K1_BUILD_CUDA=ON \
  -DSECP256K1_BUILD_BENCH=ON

cmake --build build -j
./build/cuda/secp256k1_cuda_bench
```

---

## Methodology

### CPU

- **Measurement:** Median of multiple iterations with IQR outlier removal
- **Timer:** RDTSCP (when available) or `std::chrono::high_resolution_clock`
- **Compiler:** `-O3 -march=native` (or platform-specific flags)
- **Pinning:** Single-core

### CUDA

- **Warm-up:** 10 kernel launches (discarded)
- **Measurement:** 100 launches, average
- **Timer:** CUDA events with synchronization
- **Configuration:** Default thread/block counts from config

### Reproducibility

Results saved to timestamped files:
```
benchmark-x86_64-linux-20260223-143000.txt
benchmark-aarch64-linux-20260223-143000.txt
benchmark-risc-v-64-bit-linux-20260223-143000.txt
```

---

## Hardware Details

### x86-64

- **CPU:** Intel Core i5 (AVX2, BMI2, ADX)
- **Memory:** DDR4
- **OS:** Linux
- **Compiler:** Clang 21

### ARM64

- **SoC:** RK3588 (Cortex-A55 + Cortex-A76 @ 2.256 GHz)
- **Memory:** LPDDR4
- **OS:** Linux
- **Compiler:** GCC 13 / NDK r27c Clang

### RISC-V

- **Board:** Milk-V Mars
- **CPU:** SiFive U74 (RV64GC + Zba + Zbb)
- **Memory:** DDR4
- **OS:** Linux
- **Compiler:** Clang 21 with `-mcpu=sifive-u74`

### CUDA

- **GPU:** NVIDIA RTX 5060 Ti (36 SMs, 16 GB GDDR7)
- **Driver:** 580+
- **CUDA:** 12.0

---

## See Also

- [[CPU Guide]] - CPU optimization details
- [[CUDA Guide]] - GPU optimization details
- [[API Reference]] - Function documentation

