# Benchmarking Guide

How to build, run, and interpret the bench_hornet benchmark suite on all
supported platforms.

---

## Overview

**bench_hornet** is the canonical single-core CPU benchmark for
UltrafastSecp256k1. It measures every secp256k1 operation relevant to
Bitcoin block validation and IBD (Initial Block Download), including an
apple-to-apple comparison against bitcoin-core/libsecp256k1 v0.7.2.

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

# Build bench_hornet target
cmake --build build-bench --target bench_hornet -j

# Run
./build-bench/cpu/bench/bench_hornet
```

On Windows with Clang:
```cmd
cmake -S . -B build-bench -G "NMake Makefiles" ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_CXX_COMPILER=clang-cl ^
  -DCMAKE_LINKER=lld-link
cmake --build build-bench --target bench_hornet -j
build-bench\cpu\bench\bench_hornet.exe
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
cmake --build build-riscv --target bench_hornet -j

# Deploy and run (example: Milk-V Mars at 192.168.1.31)
scp build-riscv/cpu/bench/bench_hornet user@192.168.1.31:/tmp/
ssh user@192.168.1.31 /tmp/bench_hornet
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

---

## Apple-to-Apple Comparison

Section F runs the same 6 operations using the official bitcoin-core
libsecp256k1 compiled as a single translation unit (`libsecp_bench.c`).
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

bench_hornet prints a structured ASCII table suitable for capture:

```
=============================================================
 UltrafastSecp256k1 bench_hornet -- Bitcoin Consensus Benchmark
=============================================================
 Library     : UltrafastSecp256k1 v3.16.0 ...
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
./bench_hornet > bench_hornet_report.txt 2>&1
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
| `cpu/bench/bench_hornet.cpp` | x86-64 / Linux bench_hornet (RDTSC) |
| `cpu/bench/libsecp_bench.c` | libsecp256k1 apple-to-apple (x86 / RISC-V) |
| `android/test/bench_hornet_android.cpp` | ARM64 Android port |
| `android/test/libsecp_bench.c` | libsecp256k1 apple-to-apple (ARM64) |
| `examples/esp32_bench_hornet/` | ESP32-S3 bench_hornet example |
| `audit/platform-reports/*-bench-hornet.*` | Generated reports (JSON + TXT) |
| `docs/BENCHMARKS.md` | Raw benchmark data tables |
