# Benchmarking Guide

How to build, run, and interpret benchmarks on all supported platforms.

---

## Overview

### Benchmark Targets

#### CPU Benchmarks

| Target | CI Canonical | Always Builds | Purpose |
|--------|:---:|:---:|---------|
| **`bench_unified`** | **YES** | No (needs libsecp256k1 src) | THE standard: full apple-to-apple vs libsecp256k1 + OpenSSL |
| `bench_ct` | No | YES | CT-layer benchmarks (standalone, no dependencies) |
| `bench_field_52` | No | YES (x86/ARM/RISC-V) | Field arithmetic micro-benchmarks (5x52 limbs) |
| `bench_field_26` | No | YES | Field arithmetic micro-benchmarks (10x26 limbs) |
| `bench_kP` | No | YES | Scalar multiplication (k*P) benchmarks |
| `bench_hornet` | No | No (Android only) | ARM64 Android benchmark (in android/test/) |

#### GPU Benchmarks

| Target | CI Canonical | Purpose |
|--------|:---:|---------|
| **`gpu_bench_unified`** | **YES** | GPU unified: FAST + CT ops, all categories, structured report |
| `secp256k1_cuda_bench` | No | Basic GPU search throughput measurement |

**`bench_unified`** is the canonical benchmark runner (see [.github/copilot-instructions.md](../.github/copilot-instructions.md) "Benchmark rules" section).
It runs ALL operation categories in a single binary and produces apple-to-apple
comparison ratios against both libsecp256k1 and OpenSSL.

**`bench_hornet`** is the ARM64 Android port at `android/test/bench_hornet_android.cpp`.
It measures Bitcoin consensus operations with an apple-to-apple comparison.

### What It Measures

| Section | Operations |
|---------|-----------|
| A. Core Ops (FAST) | Generator Mul, ECDSA Sign, ECDSA Verify, Schnorr Keypair, Schnorr Sign, Schnorr Verify |
| B. Constant-Time Ops | CT ECDSA Sign, CT Schnorr Sign |
| C. Batch Verification | ECDSA batch verify (100, 1000 sigs), Schnorr batch verify (100, 1000 sigs) |
| D. Block Validation | Pre-Taproot block (500 ECDSA sigs), Taproot block (500 Schnorr sigs) |
| E. Throughput | ECDSA tx/s, Schnorr tx/s (single-core) |
| F. Apple-to-Apple | Same 6 ops using bitcoin-core/libsecp256k1 directly, with speedup ratio |

### Methodology

- **x86-64**: RDTSC cycle counting converted to microseconds via measured frequency
- **ARM64 / RISC-V / ESP32**: `clock_gettime(CLOCK_MONOTONIC)` or `esp_timer_get_time()`
- **Outlier removal**: IQR (interquartile range) filtering
- **Passes**: Median of 11 passes (x86) or median of 5 passes (embedded)
- **Key pool**: 32 random keys pre-generated, messages vary per iteration

---

## Platform Build Instructions

### 1. x86-64 (Windows / Linux)

```bash
# Configure (from repo root)
cmake -S . -B build-bench -G Ninja -DCMAKE_BUILD_TYPE=Release

# Build bench_unified target
cmake --build build-bench --target bench_unified -j

# Run
./build-bench/cpu/bench_unified
```

On Windows with Clang:
```cmd
cmake -S . -B build-bench -G "NMake Makefiles" ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_CXX_COMPILER=clang-cl ^
  -DCMAKE_LINKER=lld-link
cmake --build build-bench --target bench_unified -j
build-bench\cpu\bench_unified.exe
```

### 2. ARM64 Android (Cross-compile via NDK)

Requires:
- Android NDK (tested with r27, Clang 18.0.1)
- Android device/emulator (arm64-v8a)
- ADB

```bash
# Configure with NDK toolchain
cmake -S . -B build-android -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-28
  
# Build
cmake --build build-android --target bench_hornet -j

# Deploy and run
adb push build-android/android/test/bench_hornet /data/local/tmp/
adb shell chmod +x /data/local/tmp/bench_hornet
adb shell /data/local/tmp/bench_hornet
```

### 3. RISC-V 64 (Cross-compile for Milk-V Mars / SiFive U74)

Requires:
- `riscv64-linux-gnu-gcc` 13+ (available in Ubuntu repos)
- Target board (Milk-V Mars) reachable over SSH

```bash
# Configure (using WSL or Linux host)
cmake -S . -B build-riscv -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE=cmake/riscv64-toolchain.cmake

# Build
cmake --build build-riscv --target bench_unified -j

# Deploy and run (example: Milk-V Mars at 192.168.1.31)
scp build-riscv/cpu/bench_unified user@192.168.1.31:/tmp/
ssh user@192.168.1.31 /tmp/bench_unified
```

### 4. ESP32-S3 (ESP-IDF)

Requires:
- ESP-IDF 5.5+ installed and sourced
- ESP32-S3 board connected via USB

```bash
cd examples/esp32_bench_hornet
idf.py set-target esp32s3
idf.py build
idf.py flash monitor
```

Results print to serial monitor. The ESP32 version uses `esp_timer_get_time()`
and a reduced key pool (16 keys, median of 5 passes) due to memory limits.

### 5. GPU (CUDA)

Requires:
- NVIDIA GPU with Compute Capability 7.5+ (Turing, Ampere, Ada Lovelace, Blackwell)
- CUDA Toolkit 12.0+
- CMake with `-DSECP256K1_BUILD_CUDA=ON`

```bash
# Configure (from repo root)
cmake -S . -B build-cuda -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DSECP256K1_BUILD_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="86;89"

# Build gpu_bench_unified
cmake --build build-cuda --target gpu_bench_unified -j

# Run
./build-cuda/cuda/gpu_bench_unified
```

For Blackwell GPUs (RTX 50 series), use PTX JIT:
```bash
cmake -S . -B build-cuda -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DSECP256K1_BUILD_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=90

cmake --build build-cuda --target gpu_bench_unified -j
./build-cuda/cuda/gpu_bench_unified
```

#### GPU Benchmark Sections

`gpu_bench_unified` measures all GPU operations in a single binary:

| Section | Operations |
|---------|-----------|
| 1. Field Arithmetic | field_mul, field_sqr, field_inv, field_add, field_sub |
| 2. Scalar Arithmetic | scalar_mul, scalar_inv, scalar_add, scalar_negate |
| 3. Point Arithmetic | k*G (generator), k*P (arbitrary), point_add, point_dbl |
| 4. ECDSA | sign (FAST), verify |
| 5. Schnorr / BIP-340 | keypair, sign (FAST), verify |
| 6. Constant-Time (CT) | ct::k*G, ct::k*P, ct::ecdsa_sign, ct::schnorr_sign |
| 7. Throughput | ECDSA sign/s, Schnorr sign/s |

Each section reports:
- **ns/op** (nanoseconds per operation, averaged over batched GPU launch)
- **ops/sec** (throughput)
- **CT/FAST ratio** (for CT section, overhead vs. FAST equivalent)

#### GPU Performance Expectations

| GPU | k*G | ECDSA Sign | CT ECDSA Sign | CT/FAST |
|-----|-----|-----------|---------------|---------|
| RTX 5060 Ti (SM 12.0) | 129.1 ns | 211.1 ns | 433.9 ns | 2.06x |
| RTX 4090 (SM 8.9) | ~90-120 ns | ~150-200 ns | ~300-400 ns | ~2x |

**Note**: GPU kernel timings include launch overhead. Batch size strongly
affects per-op cost -- larger batches amortize launch overhead better.

---

## Apple-to-Apple Comparison

Section F runs the same 6 operations using the official bitcoin-core
libsecp256k1 compiled as a single translation unit (`libsecp_provider.c`).
Both libraries execute on the same CPU, at the same optimization level,
in the same process -- eliminating all environmental variables.

The comparison reports:
- libsecp256k1 timing for each operation
- Speedup ratio: `libsecp_time / ultra_time`
- `> 1.0x` means UltrafastSecp256k1 is faster

### CT-vs-CT Fair Comparison

UltrafastSecp256k1 FAST operations are non-constant-time (variable-time
optimizations). libsecp256k1 is *always* constant-time. For a fair
comparison of signing operations, use the **CT-vs-CT** results which
compare `secp256k1::ct::*` operations against libsecp256k1's CT ops.

---

## Output Format

bench_unified prints a structured ASCII table suitable for capture:

```
=============================================================
 UltrafastSecp256k1 bench_unified
=============================================================
 Library     : UltrafastSecp256k1 v3.20.0 ...
 Platform    : x86-64 (AVX2, BMI2, ADX)
 CPU         : 13th Gen Intel Core i7-11700
 ...
-------------------------------------------------------------
 A. Core Operations (FAST, non-constant-time)
-------------------------------------------------------------
 Generator Mul (kxG)         :    4.95 us     [median of 11]
 ECDSA Sign                  :    8.30 us     [median of 11]
 ...
```

### Capturing Reports

To save reports for the audit campaign:
```bash
./bench_unified > bench_unified_report.txt 2>&1
```

JSON report files (for platform-reports/) are generated separately
by the benchmark infrastructure scripts -- see `audit/platform-reports/`.

---

## Interpreting Results

### Performance Expectations by Platform

| Platform | Gen Mul | ECDSA Sign | ECDSA Verify |
|----------|---------|------------|-------------|
| x86-64 (modern) | 4-6 us | 6-10 us | 25-35 us |
| ARM64 (A55) | 25-35 us | 50-70 us | 140-180 us |
| RISC-V (U74) | 40-50 us | 80-100 us | 230-270 us |
| ESP32-S3 (LX7) | 2000-2500 us | 2700-3000 us | 6000-7000 us |

### Key Insights

1. **Generator Mul** is highly optimized with precomputed tables -- always the fastest operation
2. **Verify** operations are dominated by double-point multiplication (Shamir trick with GLV)
3. **CT operations** are 50-100% slower than FAST due to constant-time requirements
4. **Batch verification** amortizes per-signature cost; 1000-sig batches approach ~80% of single-sig time per op
5. **Throughput** numbers (tx/s) reflect single-core Bitcoin consensus validation

---

## Files

| File | Purpose |
|------|---------|
| `cpu/bench/bench_unified.cpp` | THE standard: full apple-to-apple benchmark |
| `cpu/bench/bench_ct.cpp` | CT-layer benchmarks |
| `cpu/bench/bench_field_52.cpp` | 5x52 field arithmetic micro-benchmarks |
| `cpu/bench/bench_field_26.cpp` | 10x26 field arithmetic micro-benchmarks |
| `cpu/bench/libsecp_provider.c` | libsecp256k1 apple-to-apple provider |
| `cuda/src/gpu_bench_unified.cu` | GPU unified benchmark (FAST + CT) |
| `android/test/bench_hornet_android.cpp` | ARM64 Android port |
| `android/test/libsecp_bench.c` | libsecp256k1 apple-to-apple (ARM64) |
| `examples/esp32_bench_hornet/` | ESP32-S3 bench_hornet example |
| `audit/platform-reports/*-bench-hornet.*` | Generated reports (JSON + TXT) |
| `docs/BENCHMARKS.md` | Raw benchmark data tables |
