# Benchmark Comparison: UltrafastSecp256k1 vs libsecp256k1

> Head-to-head performance comparison using an **identical benchmark harness** -- same
> timer, warmup, statistical methodology, compiler flags, and data pool -- for both
> libraries. Both are compiled from source, linked into a single binary, and measured
> under the exact same conditions. **Fully reproducible.**

---

## Table of Contents

- [Methodology](#methodology)
- [Platform Summary](#platform-summary)
- [x86-64 Results (Intel i5-14400F)](#x86-64-results-intel-i5-14400f)
- [RISC-V 64 Results (SiFive U74)](#risc-v-64-results-sifive-u74)
- [ARM64 Results (RK3588 Cortex-A76)](#arm64-results-rk3588-cortex-a76)
- [GPU Results](#gpu-results)
- [Embedded Results](#embedded-results)
- [Cross-Platform Summary](#cross-platform-summary)
- [Bitcoin Block Validation](#bitcoin-block-validation)
- [Why UltrafastSecp256k1 Is Faster](#why-ultrafastsecp256k1-is-faster)
- [Batch Verification](#batch-verification)
- [How to Reproduce](#how-to-reproduce)

---

## Methodology

| Parameter | Value |
|-----------|-------|
| **Timer** | RDTSCP on x86-64 (serializing, sub-ns precision); `chrono::high_resolution_clock` on ARM64/RISC-V/ESP32 |
| **TSC calibration** | 10 ms spin-loop measuring chrono vs TSC to derive ns/tick ratio |
| **CPU ramp-up** | 3 seconds of real `k*G` crypto work to force CPU frequency to stable max |
| **TSC stabilization** | 10 x 200 ms windows; stable when two consecutive windows drift < 1% |
| **Warmup** | 500 iterations per operation (50 in `--quick` mode) |
| **Measurement passes** | 11 (odd count for clean median) |
| **Outlier removal** | IQR-based: discard samples outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR] |
| **Reported metric** | Median of filtered passes |
| **Core pinning** | `SetThreadAffinityMask(1)` (Windows), `sched_setaffinity` / `taskset -c 0` (Linux) |
| **Thread priority** | `HIGH_PRIORITY_CLASS` + `THREAD_PRIORITY_HIGHEST` (Windows); `SCHED_FIFO` (Linux) |
| **Data pool** | 64 independent key/msg/sig sets, round-robin indexed to prevent cache training |
| **Compiler barriers** | `DoNotOptimize()` (asm volatile / `_ReadWriteBarrier`) + `ClobberMemory()` |
| **Compiler parity** | Both libraries compiled with identical compiler, flags (`-O3 -march=native`), same link step |
| **Regression threshold** | >20% = warning, >50% = block merge, >100% = revert |
| **Source** | `bench_unified.cpp` -- open-source, fully reproducible |

### Statistical Rigor

The harness produces a **JSON report** with machine-readable metadata:

```json
{
  "metadata": {
    "cpu": "Intel Core i5-14400F",
    "compiler": "GCC 14.2.0",
    "arch": "x86-64",
    "timer": "RDTSCP",
    "tsc_ghz": 2.497,
    "passes": 11,
    "warmup": 500,
    "pool_size": 64
  }
}
```

---

## Platform Summary

| Platform | ECDSA Sign FAST | ECDSA Verify | Schnorr Sign FAST | Schnorr Verify | Generator k*G |
|----------|:----------:|:------------:|:------------:|:--------------:|:-------------:|
| **x86-64 (i7-11700, Clang 21)** | **2.94x** | 0.82x | **2.40x** | 0.85x | **3.44x** |
| **RISC-V 64 (U74, Clang 21)** | **1.84x** | 0.90x | **1.93x** | 0.90x | **2.37x** |
| **x86-64 CT-vs-CT** | 0.89x | 0.82x | **1.21x** | 0.85x | **1.26x** |
| **RISC-V CT-vs-CT** | **1.01x** | 0.90x | 0.94x | 0.90x | **1.04x** |

> CT-vs-CT is the **fairest comparison** since libsecp256k1 is always constant-time.
> UltrafastSecp256k1's FAST path provides additional speedup when secret protection is not needed.
> **Verify operations are currently slower** -- optimization in progress (root cause: `dual_scalar_mul` bottleneck).

---

## x86-64 Results (Intel i7-11700)

| Detail | Value |
|--------|-------|
| CPU | Intel Core i7-11700 @ 2.50 GHz (Rocket Lake) |
| OS | Windows, kernel 10.x |
| Compiler | Clang 21.1.0, `-O3 -march=native -fno-exceptions -fno-rtti` |
| ISA features | BMI2 (MULX), ADX, AVX2, SHA-NI |
| libsecp256k1 | v0.7.x (latest master, 5x52 + exhaustive GLV Strauss) |
| UltrafastSecp256k1 | v3.18.0 (dev), 5x52 limb layout, `__int128` field arithmetic |
| Timer | RDTSCP (serializing, sub-ns precision) |
| Harness | 3s CPU ramp-up, 500 warmup/op, 11 passes, IQR outlier removal, median |

### FAST Path (variable-time, non-secret inputs)

| Operation | Ultra FAST (ns) | libsecp256k1 (ns) | **Speedup** |
|-----------|----------------:|-------------------:|:-----------:|
| Generator x k (pubkey_create) | 4,160 | 14,312 | **3.44x** |
| ECDSA Sign | 7,159 | 21,067 | **2.94x** |
| ECDSA Verify | 29,967 | 24,523 | 0.82x |
| Schnorr Keypair Create | 4,660 | 14,312 | **3.07x** |
| Schnorr Sign (BIP-340) | 6,427 | 15,397 | **2.40x** |
| Schnorr Verify (BIP-340) | 29,454 | 24,927 | 0.85x |

### CT Path (constant-time -- true apples-to-apples)

Since libsecp256k1 is constant-time by design, **this is the fairest comparison**:

| Operation | Ultra CT (ns) | libsecp256k1 (ns) | **Speedup** |
|-----------|--------------:|-------------------:|:-----------:|
| ECDSA Sign | 23,542 | 21,067 | 0.89x |
| ECDSA Verify | 29,967 | 24,523 | 0.82x |
| Schnorr Sign (BIP-340) | 12,741 | 15,397 | **1.21x** |
| Schnorr Verify (BIP-340) | 29,454 | 24,927 | 0.85x |

### Throughput (single core)

| Operation | Ultra FAST | Ultra CT | libsecp256k1 |
|-----------|----------:|---------:|-------------:|
| ECDSA Sign | **139.7k** op/s | 42.5k op/s | 47.5k op/s |
| ECDSA Verify | 33.4k op/s | -- | **40.8k** op/s |
| Schnorr Sign | **155.6k** op/s | 78.5k op/s | 65.0k op/s |
| Schnorr Verify | 33.9k op/s | -- | **40.1k** op/s |
| pubkey_create (k*G) | **240.4k** op/s | -- | 69.9k op/s |

### Field Micro-ops

| Operation | Time (ns) | Notes |
|-----------|----------:|-------|
| FE52 mul | 15.3 | 5x52, `__int128` -> MULX (ASM52 OFF) |
| FE52 sqr | 14.4 | Dedicated squaring |
| FE52 add | 1.6 | |
| FE52 inverse (SafeGCD) | 666.8 | Bernstein-Yang, `__builtin_ctzll` |
| Scalar mul | 15.6 | 4x64 |
| Scalar inverse (SafeGCD) | 1,605.5 | |
| GLV decomposition | 75.1 | Lattice-based |

---

## RISC-V 64 Results (SiFive U74)

| Detail | Value |
|--------|-------|
| CPU | SiFive U74-MC (quad-core RV64GC) |
| Microarchitecture | Dual-issue in-order, 32 KB L1i, 32 KB L1d |
| ISA extensions | rv64gc + Zba (address), Zbb (bit-manipulation) |
| Clock | ~1.5 GHz (StarFive JH7110 SoC) |
| OS | Debian (StarFive kernel 6.6.20) |
| Compiler | Clang 21.1.8, `-O3 -march=rv64gcv_zba_zbb` |
| libsecp256k1 | v0.7.x (latest master) |
| UltrafastSecp256k1 | v3.18.0 (dev), 5x52, `__int128`, RISC-V FE52 ASM |
| Timer | chrono::high_resolution_clock |
| Harness | 3s CPU ramp-up, 500 warmup/op, 11 passes, IQR outlier removal, median |

### FAST Path

| Operation | Ultra FAST (ns) | libsecp256k1 (ns) | **Speedup** |
|-----------|----------------:|-------------------:|:-----------:|
| Generator x k (pubkey_create) | 40,031 | 94,767 | **2.37x** |
| ECDSA Sign | 75,217 | 138,337 | **1.84x** |
| ECDSA Verify | 208,016 | 187,088 | 0.90x |
| Schnorr Keypair Create | 45,891 | 95,123 | **2.07x** |
| Schnorr Sign (BIP-340) | 54,480 | 104,901 | **1.93x** |
| Schnorr Verify (BIP-340) | 213,416 | 191,644 | 0.90x |

### CT Path (constant-time, apples-to-apples)

| Operation | Ultra CT (ns) | libsecp256k1 (ns) | **Speedup** |
|-----------|--------------:|-------------------:|:-----------:|
| ECDSA Sign | 137,389 | 138,337 | **1.01x** |
| ECDSA Verify | 208,016 | 187,088 | 0.90x |
| Schnorr Sign (BIP-340) | 111,580 | 104,901 | 0.94x |
| Schnorr Verify (BIP-340) | 213,416 | 191,644 | 0.90x |

> CT Schnorr Sign on RISC-V is 0.94x due to auxiliary overhead (SHA-256 tagged hash,
> nonce derivation) not related to the core ECC operation. The core scalar_mul is faster.

### Throughput (single core)

| Operation | Ultra FAST | Ultra CT | libsecp256k1 |
|-----------|----------:|---------:|-------------:|
| ECDSA Sign | **13.3k** op/s | 7.3k op/s | 7.2k op/s |
| ECDSA Verify | 4.8k op/s | -- | **5.3k** op/s |
| Schnorr Sign | **18.4k** op/s | 9.0k op/s | 9.5k op/s |
| Schnorr Verify | 4.7k op/s | -- | **5.2k** op/s |
| pubkey_create (k*G) | **25.0k** op/s | -- | 10.6k op/s |

### Field Micro-ops

| Operation | Time (ns) | Notes |
|-----------|----------:|-------|
| FE52 mul | 176.4 | 5x52, `__int128` -> MUL/MULHU |
| FE52 sqr | 167.0 | Dedicated squaring |
| FE52 add | 42.1 | |
| FE52 inverse (SafeGCD) | 4,736.4 | Bernstein-Yang |
| Scalar mul | 149.6 | 4x64 |
| Scalar inverse (SafeGCD) | 3,610.1 | |
| GLV decomposition | 522.7 | Lattice-based |

### RISC-V Notes

- The U74 is a dual-issue in-order core -- no out-of-order execution, no speculative
  execution, no branch prediction beyond basic BTB.
- Despite this, the precomputed comb table gives a **2.4x** generator speedup, showing
  the optimization is **algorithmic** (fewer point additions) not microarchitecture-dependent.
- CT generator_mul uses an 11-block comb with a ~31 KB table that fits in the U74's
  32 KB L1D cache.

---

## ARM64 Results (RK3588 Cortex-A76)

| Detail | Value |
|--------|-------|
| CPU | RK3588 (Cortex-A76 @ 2.256 GHz, pinned to big cores) |
| OS | Android |
| Compiler | NDK r26, Clang 17.0.2 |
| Assembly | ARM64 inline (MUL/UMULH) |
| Field representation | 10x26 (optimal for ARM64) |

### Results

| Operation | Ultra FAST | Notes |
|-----------|----------:|-------|
| Field Mul | 74 ns | ARM64 MUL/UMULH, 10x26 |
| Field Square | 50 ns | |
| Field Inverse | 2 us | Fermat's theorem |
| Generator k*G | 14 us | Precomputed tables |
| Scalar k*P | 131 us | GLV + wNAF |
| ECDSA Sign | 30 us | RFC 6979 |
| ECDSA Verify | 153 us | Shamir + GLV |
| Schnorr Sign (BIP-340) | 38 us | |
| Schnorr Verify (BIP-340) | 173 us | |
| Batch Inverse (n=100) | 265 ns/elem | Montgomery's trick |
| Batch Inverse (n=1000) | 240 ns/elem | |

---

## GPU Results

> **No other open-source GPU library provides secp256k1 ECDSA + Schnorr sign/verify on GPU.**

### CUDA (NVIDIA RTX 5060 Ti)

| Detail | Value |
|--------|-------|
| GPU | RTX 5060 Ti (36 SMs, 2602 MHz, 15.8 GB GDDR7) |
| CUDA | 12.0, Compute SM 86/89 (Blackwell) |
| Build | Clang 19 + nvcc, Release, `-O3 --use_fast_math` |

| Operation | Time/Op | Throughput | Batch Size |
|-----------|--------:|----------:|-----------:|
| Generator k*G | 217.7 ns | **4.59 M/s** | 128K |
| Scalar k*P | 225.8 ns | **4.43 M/s** | 64K |
| ECDSA Sign | 204.8 ns | **4.88 M/s** | 16K |
| ECDSA Verify | 410.1 ns | **2.44 M/s** | 16K |
| ECDSA Sign + Recid | 311.5 ns | **3.21 M/s** | 16K |
| Schnorr Sign (BIP-340) | 273.4 ns | **3.66 M/s** | 16K |
| Schnorr Verify (BIP-340) | 354.6 ns | **2.82 M/s** | 16K |
| Field Mul | 0.2 ns | 4,142 M/s | 1M |
| Field Inv | 10.2 ns | 98.35 M/s | 64K |
| Point Add | 1.6 ns | 619 M/s | 256K |
| Point Double | 0.8 ns | 1,282 M/s | 256K |

### OpenCL (NVIDIA RTX 5060 Ti)

| Operation | Kernel-Only | End-to-End | Batch |
|-----------|------------:|-----------:|------:|
| Field Mul | 0.2 ns | 27.7 ns | 1M |
| Field Inv | 14.3 ns | 29.0 ns | 1M |
| Point Add | 1.6 ns | 111.9 ns | 256K |
| Point Double | 0.9 ns | 58.4 ns | 1M |
| Generator k*G | 295.1 ns | 307.7 ns | 65K |

### Apple Metal (M3 Pro)

| Detail | Value |
|--------|-------|
| GPU | Apple M3 Pro (18 GPU cores, Unified Memory 18 GB) |
| Metal | 2.4, MSL macos-metal2.4 |
| Limb Model | 8x32-bit Comba (no 64-bit int in MSL) |

| Operation | Time/Op | Throughput |
|-----------|--------:|----------:|
| Field Mul | 1.9 ns | 527 M/s |
| Field Add | 1.0 ns | 990 M/s |
| Field Inv | 106.4 ns | 9.40 M/s |
| Point Add | 10.1 ns | 98.6 M/s |
| Point Double | 5.1 ns | 196 M/s |
| Scalar k*P | 2.94 us | 0.34 M/s |
| Generator k*G | 3.00 us | 0.33 M/s |

### GPU Backend Comparison

| Operation | CUDA (RTX 5060 Ti) | OpenCL (RTX 5060 Ti) | Metal (M3 Pro) |
|-----------|-------------------:|---------------------:|---------------:|
| Field Mul | 0.2 ns | 0.2 ns | 1.9 ns |
| Field Inv | 10.2 ns | 14.3 ns | 106.4 ns |
| Point Add | 1.6 ns | 1.6 ns | 10.1 ns |
| Generator k*G | 217.7 ns | 295.1 ns | 3.00 us |
| ECDSA Sign | 204.8 ns | -- | -- |
| ECDSA Verify | 410.1 ns | -- | -- |
| Schnorr Sign | 273.4 ns | -- | -- |
| Schnorr Verify | 354.6 ns | -- | -- |

> RTX 5060 Ti has ~8x more compute throughput; M3 Pro's advantage is in unified memory
> zero-copy I/O.

---

## Embedded Results

| Operation | ESP32-S3 (LX7) | ESP32 (LX6) | STM32F103 (CM3) |
|-----------|:--------------:|:-----------:|:---------------:|
| | 240 MHz | 240 MHz | 72 MHz |
| Field Mul | 7,458 ns | 6,993 ns | 15,331 ns |
| Field Square | 7,592 ns | 6,247 ns | 12,083 ns |
| Field Add | 636 ns | 985 ns | 4,139 ns |
| Field Inverse | 844 us | 609 us | 1,645 us |
| Generator k*G | 2,483 us | 6,203 us | 37,982 us |
| CT Generator k*G | -- | 44,810 us | -- |
| Self-tests PASS | 35/35 | 35/35 + 8 CT | 35/35 |

---

## Cross-Platform Summary

### ECDSA Sign (lower is better)

| Platform | Ultra FAST | Ultra CT | libsecp256k1 | FAST Speedup |
|----------|----------:|---------:|-------------:|:------------:|
| x86-64 (i7-11700) | 7,159 ns | 23,542 ns | 21,067 ns | **2.94x** |
| RISC-V 64 (U74) | 75,217 ns | 137,389 ns | 138,337 ns | **1.84x** |
| ARM64 (A76) | 30,000 ns | -- | -- | -- |
| CUDA (RTX 5060 Ti) | 204.8 ns | -- | -- | -- |

### ECDSA Verify (lower is better)

| Platform | Ultra FAST | libsecp256k1 | Speedup |
|----------|----------:|-------------:|:-------:|
| x86-64 (i7-11700) | 29,967 ns | 24,523 ns | 0.82x |
| RISC-V 64 (U74) | 208,016 ns | 187,088 ns | 0.90x |
| ARM64 (A76) | 153,000 ns | -- | -- |
| CUDA (RTX 5060 Ti) | 410.1 ns | -- | -- |

### Generator k*G (lower is better)

| Platform | Ultra FAST | libsecp256k1 | Speedup |
|----------|----------:|-------------:|:-------:|
| x86-64 (i7-11700) | 4,160 ns | 14,312 ns | **3.44x** |
| RISC-V 64 (U74) | 40,031 ns | 94,767 ns | **2.37x** |
| ARM64 (A76) | 14,000 ns | -- | -- |
| CUDA (RTX 5060 Ti) | 217.7 ns | -- | -- |
| ESP32-S3 | 2,483,000 ns | -- | -- |
| STM32F103 | 37,982,000 ns | -- | -- |

---

## Bitcoin Block Validation

Estimated per-block validation time (single core):

### x86-64 (i7-11700)

| Block Type | Ultra FAST | libsecp256k1 | Speedup |
|------------|----------:|-------------:|:-------:|
| Pre-Taproot (~3000 ECDSA verify) | 89.9 ms | 73.6 ms | 0.82x |
| Taproot (~2000 Schnorr + ~1000 ECDSA) | 88.9 ms | 73.5 ms | 0.83x |

### RISC-V 64 (U74)

| Block Type | Ultra FAST | libsecp256k1 | Speedup |
|------------|----------:|-------------:|:-------:|
| Pre-Taproot (~3000 ECDSA verify) | 624.0 ms | 561.3 ms | 0.90x |
| Taproot (~2000 Schnorr + ~1000 ECDSA) | 634.8 ms | 565.4 ms | 0.90x |

### GPU Block Validation (theoretical, RTX 5060 Ti)

| Block Type | Time | vs CPU (Ultra) |
|------------|-----:|:--------------:|
| Pre-Taproot (~3000 ECDSA verify) | 1.23 ms | **52x** faster |
| Taproot full block | 1.06 ms | **60x** faster |

---

## Batch Verification

Batch verification amortizes per-signature overhead via multi-scalar multiplication:

| Batch Size | Schnorr Batch (ns/sig) | Speedup vs Single | ECDSA Batch (ns/sig) | Speedup vs Single |
|:----------:|----------------------:|:-----------------:|--------------------:|:-----------------:|
| 1 | 21,151 | 1.00x | 21,324 | 1.00x |
| 4 | ~12,000 | ~1.76x | ~13,000 | ~1.64x |
| 16 | ~7,000 | ~3.02x | ~8,000 | ~2.67x |
| 64 | ~4,500 | ~4.70x | ~5,500 | ~3.88x |

---

## Why UltrafastSecp256k1 Is Faster

### 7 Key Optimizations

| # | Optimization | Effect | vs libsecp256k1 |
|---|-------------|--------|-----------------|
| 1 | **Precomputed generator table** | 8192-entry W=15 comb table for k*G | Fewer point additions than Strauss (6.7 us vs 11.4 us on x86) |
| 2 | **Force-inlined doubling** | `jac52_double_inplace` with `always_inline` in hot loop | Eliminates call overhead in tight double-add loop |
| 3 | **Called (not inlined) additions** | ecmult function: 39 KB vs 124 KB | Fits hot loop in L1 I-cache (1.5 KB body vs 32 KB cache) |
| 4 | **Branchless conditional negate** | XOR-select in Strauss loop | Eliminates 50% unpredictable sign branches |
| 5 | **Single affine conversion** | Merged X-check + Y-parity in Schnorr verify | Saves 1 sqr + 1 mul + redundant parse |
| 6 | **SW prefetch** | Prefetch G/H table entries before doublings | Hides memory latency for table lookups |
| 7 | **2M+5S doubling formula** | Saves 1M per double | vs libsecp's 3M+4S formula |

### Architectural Advantage

UltrafastSecp256k1 provides a **dual-layer architecture**:

- **FAST path** (`secp256k1::fast::*`) -- Variable-time, maximum throughput. For public
  verification, batch operations, search workloads.
- **CT path** (`secp256k1::ct::*`) -- Constant-time, side-channel resistant. For signing,
  key derivation, secret-dependent operations.

The C ABI (`ufsecp_*`) automatically routes each function to the correct layer -- no opt-in
flag required. This means applications get the best of both worlds: CT security for secrets,
maximum speed for public operations.

---

## Field Representation Comparison

### 5x52 vs 4x64 (x86-64, Windows Clang 21)

| Operation | 4x64 (ns) | 5x52 (ns) | 5x52 Speedup |
|-----------|----------:|----------:|:------------:|
| Multiplication | 41.9 | 15.2 | **2.76x** |
| Squaring | 31.2 | 12.8 | **2.44x** |
| Addition | 4.3 | 1.6 | **2.69x** |
| Add chain (16 ops) | 137.7 | 30.3 | **4.55x** |
| Add chain (64 ops) | 566.8 | 117.1 | **4.84x** |
| 256 squarings | 9,039 | 4,055 | **2.23x** |

> 5x52 with `__int128` and lazy carry reduction is 2-5x faster across all operations.

---

## How to Reproduce

### Build

```bash
# Clone
git clone --recurse-submodules https://github.com/shrec/UltrafastSecp256k1.git
cd UltrafastSecp256k1

# Configure and build
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

### Run Benchmark

```bash
# Linux (pin to core 0 for stability)
taskset -c 0 ./build/cpu/bench_unified

# Windows (PowerShell)
Start-Process -FilePath .\build\cpu\Release\bench_unified.exe -NoNewWindow -Wait

# JSON output
taskset -c 0 ./build/cpu/bench_unified --json bench_results.json

# Quick mode (3 passes instead of 11)
taskset -c 0 ./build/cpu/bench_unified --quick
```

### Run with libsecp256k1 Comparison

The benchmark automatically includes libsecp256k1 comparison when the library is available
(FetchContent, downloaded during CMake configuration).

### Contributing Benchmarks

We welcome benchmark contributions from other platforms. To add your results:

1. Run `taskset -c 0 ./build/cpu/bench_unified` (or equivalent core pinning)
2. Copy the full terminal output or use `--json` for structured data
3. Open a PR adding a new platform section with your hardware details

Platforms we'd especially like to see: **AMD Zen 4/5**, **Apple M-series (ARM64)**,
**AWS Graviton**, **AMD EPYC**, **Intel Xeon Sapphire Rapids**, **Milk-V Pioneer (C920)**.

---

*Last updated: 2025-07-14*
*UltrafastSecp256k1 v4.0.0*
*Benchmark harness: bench_unified.cpp (11-pass median, IQR outlier removal)*
*Platforms measured: x86-64 (i7-11700, Clang 21), RISC-V (U74, Clang 21)*
