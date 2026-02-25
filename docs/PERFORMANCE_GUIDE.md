# Performance Guide

Practical tuning recommendations for UltrafastSecp256k1 across platforms.

---

## Table of Contents

1. [Quick Summary](#quick-summary)
2. [Compiler Selection](#compiler-selection)
3. [Assembly Backend](#assembly-backend)
4. [Build Type & Flags](#build-type--flags)
5. [Batch Operations](#batch-operations)
6. [GPU Acceleration](#gpu-acceleration)
7. [Memory & Cache](#memory--cache)
8. [Platform-Specific Tuning](#platform-specific-tuning)
9. [Constant-Time Cost](#constant-time-cost)
10. [Profiling Guide](#profiling-guide)

---

## Quick Summary

| Tuning | Impact | Effort |
|--------|--------|--------|
| Use Clang 17+ (LTO) | 10-20% speedup | Low |
| Enable ASM (`SECP256K1_USE_ASM=ON`) | 2-5x on field ops | Low |
| Use batch inverse for bulk ops | 10-50x for N>100 | Medium |
| GPU batch for >10K operations | 100-1000x throughput | High |
| Precomputed tables (gen_mul) | 20x vs generic mul | Zero (default) |

---

## Compiler Selection

### Recommended

| Platform | Compiler | Notes |
|----------|----------|-------|
| Linux | Clang 17+ (with lld) | Best codegen for x86-64 + ARM64 |
| macOS | Apple Clang 15+ | Native ARM64 codegen |
| Windows | Clang-cl 17+ or MSVC 19.40+ | Clang preferred for vectorization |
| Embedded | GCC 13+ | Best for RISC-V, ARM Cortex-M |

### LTO (Link-Time Optimization)

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON
```

LTO gives 10-20% speedup through cross-module inlining. The library's hot path
functions (`field_mul`, `scalar_mul`, `point_add`) benefit significantly.

**Warning**: Do NOT use LTO with CUDA targets. The release build explicitly
disables IPO for `gpu_cuda_test` to avoid `nvcc -dlink` failures.

---

## Assembly Backend

### x86-64 (BMI2/ADX)

The assembly backend provides optimized `mulx`/`adcx`/`adox` sequences for field
multiplication and squaring. This is the default on x86-64.

```bash
# Explicitly enable (usually auto-detected)
-DSECP256K1_USE_ASM=ON
```

| Operation | C++ Generic | x86-64 ASM | Speedup |
|-----------|------------|------------|---------|
| Field Mul | 85 ns | 17 ns | 5.0x |
| Field Square | 80 ns | 16 ns | 5.0x |
| Field Inverse | 12 us | 5 us | 2.4x |

### ARM64 (NEON)

ARM64 uses `umulh`/`umaddl` instructions for field arithmetic.

### RISC-V 64

RISC-V uses custom `mulhu`/`mul` sequences. Enable with:

```bash
-DCMAKE_SYSTEM_PROCESSOR=riscv64 -DSECP256K1_USE_ASM=ON
```

---

## Build Type & Flags

### Production Build

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DSECP256K1_USE_ASM=ON \
  -DSECP256K1_BUILD_BENCH=OFF \
  -DSECP256K1_BUILD_TESTS=OFF \
  -DSECP256K1_BUILD_EXAMPLES=OFF
```

### Speed-First Build (Unsafe Optimizations)

```bash
-DSECP256K1_SPEED_FIRST=ON
```

This enables aggressive optimizations including `-ffast-math` for non-crypto code
paths. **Never use for cryptographic operations** -- only for search/batch workloads
where IEEE 754 compliance is not required.

---

## Batch Operations

### Batch Inverse (Montgomery's Trick)

For `N` field inversions, batch inverse computes all `N` results with only one
full inversion + `3(N-1)` multiplications:

| N | Per-Element Cost | vs Individual |
|---|-----------------|---------------|
| 1 | 5,000 ns | 1.0x |
| 10 | 500 ns | 10x |
| 100 | 140 ns | 36x |
| 1000 | 92 ns | 54x |
| 8192 | 85 ns | 59x |

**Usage**: All multi-point operations (batch verify, multi-scalar mul) use batch
inverse automatically.

### Multi-Scalar Multiplication

For computing `sum(k_i * P_i)`:

| Method | Time (10 points) | Time (100 points) |
|--------|-----------------|-------------------|
| Individual | 1,100 us | 11,000 us |
| Multi-scalar (Straus) | 250 us | 1,800 us |
| Multi-scalar (Pippenger) | -- | 900 us |

Pippenger is automatically selected when N > 64.

---

## GPU Acceleration

### When to Use GPU

GPU is beneficial for **embarrassingly parallel** workloads:

| Workload | CPU (1 core) | GPU (RTX 5060 Ti) | Speedup |
|----------|-------------|-------------------|---------|
| 1 scalar mul | 25 us | 225 ns + launch overhead | Slower |
| 1K scalar muls | 25 ms | 0.3 ms | 83x |
| 1M scalar muls | 25 s | 0.25 s | 100x |

**Rule of thumb**: GPU wins when batch size > 1,000 operations.

### GPU Configuration

```json
{
  "device_id": 0,
  "threads_per_batch": 131072,
  "batch_interval": 64,
  "max_matches": 786432
}
```

| Parameter | Recommended | Notes |
|-----------|-------------|-------|
| `threads_per_batch` | SM_count x 1024 | Fill all SMs |
| `batch_interval` | 32-128 | Higher = more work per kernel |
| `max_matches` | >= expected_matches x 2 | Pre-allocated result buffer |

### GPU Backend Selection

| Backend | Best For | Notes |
|---------|----------|-------|
| CUDA | NVIDIA GPUs | Fastest, most mature |
| ROCm/HIP | AMD GPUs | API-compatible with CUDA |
| OpenCL | Cross-vendor | Slightly slower than native |
| Metal | Apple Silicon | macOS/iOS only |

**Important**: GPU backends are **NOT constant-time**. Never process secret keys
on GPU. See [docs/CT_VERIFICATION.md](CT_VERIFICATION.md).

---

## Memory & Cache

### Hot Path Memory Model

The library's hot path is **zero-allocation**:

- No `malloc`/`new` in field/scalar/point operations
- Pre-allocated scratch buffers via arena pattern
- Thread-local scratch on CPU
- Fixed-size POD types (no hidden copies)

### Cache Optimization

- **Generator mul precomputed table**: ~64 KB, fits in L1 cache on most CPUs
- **Batch operations**: Sequential memory access pattern for cache friendliness
- **GLV decomposition**: Splits 256-bit scalar mul into two 128-bit muls, reducing
  table lookups by ~40%

### Stack Usage

| Operation | Stack (approx) |
|-----------|---------------|
| Field mul | 128 bytes |
| Scalar mul | 2 KB |
| ECDSA sign | 4 KB |
| FROST sign | 8 KB |
| Multi-scalar (N=100) | 16 KB |

Embedded targets (ESP32, STM32) should ensure sufficient stack allocation.
The CMake build sets `/STACK:4194304` on Windows for test binaries.

---

## Platform-Specific Tuning

### x86-64

```bash
# Maximum performance: Clang + LTO + native tuning
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON \
  -DSECP256K1_USE_ASM=ON
```

### ARM64 (Raspberry Pi 5, Apple Silicon)

```bash
# ARM64 auto-detects NEON; no special flags needed
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DSECP256K1_USE_ASM=ON
```

### ESP32-S3

```bash
# Use ESP-IDF CMake integration
idf.py set-target esp32s3
idf.py build
```

Key constraint: 520 KB SRAM total. The library uses ~64 KB for precomputed tables.
Disable unused protocol modules to save flash.

### WASM (Emscripten)

```bash
emcmake cmake -S . -B build-wasm \
  -DCMAKE_BUILD_TYPE=Release \
  -DSECP256K1_BUILD_WASM=ON
emmake cmake --build build-wasm
```

WASM performance is typically 3-5x slower than native due to 64-bit integer
emulation, but still competitive for client-side applications.

---

## Constant-Time Cost

The `ct::` namespace provides timing-safe operations at a performance cost:

| Operation | FAST path | CT path | Overhead |
|-----------|-----------|---------|----------|
| Scalar mul | 25 us | 150 us | 6.0x |
| ECDSA sign | 30 us | 180 us | 6.0x |
| Schnorr sign | 28 us | 170 us | 6.1x |
| Field inverse | 5 us | 35 us | 7.0x |

**When to use CT**: Always use `ct::` variants when processing private keys, nonces,
or any secret-dependent data. The FAST path is only safe for public inputs.

---

## Profiling Guide

### Quick Benchmark

```bash
cmake --build build --target bench_comprehensive
./build/cpu/bench_comprehensive
```

### Targeted Profiling (Linux)

```bash
# perf stat for operation counts
perf stat -e cycles,instructions,cache-misses ./build/cpu/bench_comprehensive

# perf record for flame graph
perf record -g ./build/cpu/bench_comprehensive
perf script | stackcollapse-perf.pl | flamegraph.pl > flame.svg
```

### Targeted Profiling (Windows)

```powershell
# Use Visual Studio Profiler or Intel VTune
# Build with debug info for symbol resolution
cmake -S . -B build-profile -DCMAKE_BUILD_TYPE=RelWithDebInfo
```

### Key Metrics to Watch

| Metric | Target | Red Flag |
|--------|--------|----------|
| Field mul | < 20 ns (x86-64 ASM) | > 50 ns |
| Generator mul | < 6 us | > 15 us |
| Scalar mul | < 30 us | > 80 us |
| ECDSA sign | < 35 us | > 100 us |
| Cache miss rate | < 2% | > 10% |
| Branch misprediction | < 1% | > 5% |

---

## See Also

- [docs/BENCHMARKS.md](BENCHMARKS.md) -- Full benchmark results
- [docs/BENCHMARK_METHODOLOGY.md](BENCHMARK_METHODOLOGY.md) -- How benchmarks are collected
- [docs/CT_VERIFICATION.md](CT_VERIFICATION.md) -- Constant-time verification details
- [PORTING.md](../PORTING.md) -- Platform porting guide
