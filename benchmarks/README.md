# UltrafastSecp256k1 Benchmarks

Performance benchmarks across different platforms and configurations.

## 📊 Benchmark Results

### Directory Structure

```
benchmarks/
+-- cpu/
|   +-- x86-64/
|   |   +-- windows/     # Windows x64 results
|   |   +-- linux/       # Linux x64 results
|   +-- riscv64/
|   |   +-- linux/       # RISC-V RV64GC (Milk-V Mars, etc.)
|   +-- arm64/
|   |   +-- linux/       # ARM64 Linux (RPi, etc.)
|   |   +-- macos/       # Apple Silicon (M1/M2/M3)
|   +-- esp32/
|       +-- embedded/    # ESP32 (limited, core only)
+-- gpu/
|   +-- cuda/
|   |   +-- rtx-40xx/    # RTX 4090, 4080, etc.
|   |   +-- rtx-30xx/    # RTX 3090, 3080, etc.
|   |   +-- rtx-20xx/    # RTX 2080 Ti, etc.
|   |   +-- datacenter/  # A100, H100, V100
|   +-- opencl/          # NVIDIA, AMD, Intel, etc.
+-- comparison/          # Cross-platform comparisons
```

## 🚀 Running Benchmarks

### CPU Benchmarks

```bash
# Build with benchmarks
cmake -B build -DSECP256K1_BUILD_BENCH=ON
cmake --build build -j

# Run all CPU benchmarks
./build/cpu/bench/benchmark_field
./build/cpu/bench/benchmark_point
./build/cpu/bench/benchmark_scalar

# RISC-V comprehensive benchmark
./build/libs/UltrafastSecp256k1/cpu/bench_comprehensive_riscv

# Save results
./build/cpu/bench/benchmark_field > benchmarks/cpu/x86-64/linux/field_$(date +%Y%m%d).txt
```

### GPU Benchmarks (CUDA)

```bash
# Build with CUDA
cmake -B build -DSECP256K1_BUILD_CUDA=ON -DSECP256K1_BUILD_BENCH=ON
cmake --build build -j

# Run GPU benchmarks
./build/cuda/bench/cuda_benchmark

# Save results
./build/cuda/bench/cuda_benchmark > benchmarks/gpu/cuda/rtx-4090/batch_$(date +%Y%m%d).txt
```

## 📈 Benchmark Format

Each benchmark file should include:

```
Platform: x86-64 / RISC-V / ARM64 / CUDA
CPU/GPU: Specific model
OS: Windows 11 / Linux 6.x / macOS 14
Compiler: GCC 13.2 / Clang 18 / MSVC 2022
Build: Release / -O3 / Assembly ON/OFF
Date: YYYY-MM-DD

=== Field Operations ===
Addition:        X ns/op
Multiplication:  X ns/op
Squaring:        X ns/op
Inversion:       X ns/op

=== Point Operations ===
Point Addition:      X us/op
Point Doubling:      X us/op
Point Multiply:      X us/op
Batch Multiply (n):  X ms for n ops

=== Throughput ===
Operations/second:   X M ops/s
```

## 🎯 Submitting Benchmarks

If you run benchmarks on your hardware, please submit them!

1. Run the benchmark suite
2. Save results to appropriate directory
3. Include system information
4. Submit via Pull Request

**Template:**
```bash
# System info
uname -a
lscpu  # or cat /proc/cpuinfo
gcc --version  # or clang --version

# Run benchmarks
./build/cpu/bench/benchmark_field > results.txt
```

## 📊 Current Results

See individual platform directories for detailed results:
- [x86-64 Windows](cpu/x86-64/windows/)
- [x86-64 Linux](cpu/x86-64/linux/)
- [**RISC-V Linux (Milk-V Mars)** OK](cpu/riscv64/linux/) - **Updated 2026-02-11**
- [**ESP32-S3 Embedded** OK](cpu/esp32/embedded/) - **Updated 2026-02-13**
- [ARM64 Linux](cpu/arm64/linux/)
- [CUDA RTX 4090](gpu/cuda/rtx-40xx/)

## 🏆 Platform Performance Comparison

### ESP32-S3 (Xtensa LX7 @ 240 MHz)
**Configuration:** Portable C++ (no assembly, no __int128)  
**Date:** 2026-02-13 | **Tests:** 28/28 OK

| Operation | Performance |
|-----------|-------------|
| Field Multiply | 7,458 ns |
| Field Square | 7,592 ns |
| Field Add | 636 ns |
| Scalar x G | 2,483 us |

### RISC-V (Milk-V Mars - StarFive JH7110 @ 1.5 GHz)
**Configuration:** Assembly + RVV + Fast Modular Reduction  
**Date:** 2026-02-11 | **Tests:** 29/29 OK

| Operation | Performance |
|-----------|-------------|
| Field Multiply | 200 ns |
| Field Square | 185 ns |
| Point Scalar Mul | 665 us |
| Generator Mul | 44 us |
| Batch Inverse (1000) | 611 ns/element |

### x86-64 (Typical Desktop/Server)
| Operation | Performance (est.) |
|-----------|-------------|
| Field Multiply | 8-12 ns |
| Point Scalar Mul | 60-80 us |
| Generator Mul | 4-6 us |

*Note: x86-64 performance varies by CPU model (Intel/AMD), clock speed (3-5 GHz typical), and assembly optimizations.*

### Performance Insights

- **ESP32-S3 vs x86-64:** ~230x difference in field multiply, primarily due to:
  - Clock speed (240 MHz vs 3.5+ GHz)
  - 32-bit portable arithmetic vs 64-bit with BMI2/ADX
  - No assembly optimizations on Xtensa (yet)
  
- **ESP32-S3 Achievement:** Library runs correctly on resource-constrained MCU!
  - All 28 tests pass
  - Suitable for IoT authentication, hardware wallets
  - ~2.5ms per signature verification

- **RISC-V vs x86-64:** ~8-10x difference, primarily due to:
  - Clock speed (1.5 GHz vs 3.5+ GHz)
  - ISA maturity and compiler optimizations
  - Memory subsystem performance
  
- **RISC-V Achievement:** Production-ready performance for embedded/IoT cryptographic applications

- **Assembly Impact:** 2-3x speedup vs portable C++ on x86-64 and RISC-V platforms

**Contribute your results to expand this comparison!**

---

For questions about benchmarking, open an issue or discussion on GitHub.
