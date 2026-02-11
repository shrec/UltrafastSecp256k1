# Benchmarks

Performance measurements for UltrafastSecp256k1 across different platforms.

---

## Summary

| Platform | Scalar Mul | Field Mul | Generator Mul |
|----------|-----------|-----------|---------------|
| x86-64 (i5, AVX2) | 110 μs | 33 ns | 5 μs |
| RISC-V 64 (RVV) | 672 μs | 198 ns | 40 μs |
| CUDA (RTX 4090) | TBD | TBD | TBD |

---

## x86-64 Results

**Hardware:** Intel Core i5 (AVX2, BMI2, ADX)  
**OS:** Linux  
**Compiler:** Clang 19.1.7  
**Assembly:** x86-64 with BMI2/ADX

| Operation | Time | Throughput |
|-----------|------|------------|
| Field Mul | 33 ns | 30.3 M/s |
| Field Square | 32 ns | 31.3 M/s |
| Field Add | 11 ns | 90.9 M/s |
| Field Sub | 12 ns | 83.3 M/s |
| Field Inverse | 5 μs | 200 K/s |
| Point Add | 521 ns | 1.9 M/s |
| Point Double | 278 ns | 3.6 M/s |
| Scalar Mul | 110 μs | 9.1 K/s |
| Generator Mul | 5 μs | 200 K/s |
| Batch Inverse (n=100) | 140 ns/elem | 7.1 M/s |
| Batch Inverse (n=1000) | 92 ns/elem | 10.9 M/s |

---

## RISC-V Results

**Hardware:** RISC-V 64-bit (RV64GC + V extension)  
**OS:** Linux  
**Compiler:** Clang 21.1.8  
**Assembly:** RISC-V native + RVV

| Operation | Time | Throughput |
|-----------|------|------------|
| Field Mul | 198 ns | 5.1 M/s |
| Field Square | 177 ns | 5.6 M/s |
| Field Add | 34 ns | 29.4 M/s |
| Field Sub | 31 ns | 32.3 M/s |
| Field Inverse | 18 μs | 56 K/s |
| Point Add | 3 μs | 333 K/s |
| Point Double | 1 μs | 1 M/s |
| Scalar Mul | 672 μs | 1.5 K/s |
| Generator Mul | 40 μs | 25 K/s |
| Batch Inverse (n=100) | 765 ns/elem | 1.3 M/s |
| Batch Inverse (n=1000) | 615 ns/elem | 1.6 M/s |

### RISC-V Optimization History

| Date | Field Mul | Scalar Mul | Notes |
|------|-----------|------------|-------|
| 2026-02-08 | 307 ns | 954 μs | Initial implementation |
| 2026-02-09 | 205 ns | 676 μs | Carry chain optimization |
| 2026-02-10 | 198 ns | 672 μs | Square optimization |

**Total Improvement:** 1.55× faster (Field Mul), 1.42× faster (Scalar Mul)

---

## CUDA Results

**Hardware:** NVIDIA RTX 5060 Ti  
**CUDA:** 12.0+  
**Architecture:** sm_89 (Ada Lovelace)

| Operation | Batch Size | Time | Throughput |
|-----------|------------|------|------------|
| Scalar Mul | 1M | TBD | TBD |
| Key Generation | 1M | TBD | TBD |
| Hash160 | 1M | TBD | TBD |

---

## Comparison with Other Libraries

### vs libsecp256k1 (Bitcoin Core)

| Operation | Ultrafast | libsecp256k1 | Speedup |
|-----------|----------|--------------|---------|
| Scalar Mul (x86-64) | 110 μs | ~150 μs | 1.36× |
| Generator Mul | 5 μs | ~8 μs | 1.6× |
| Batch Verify | TBD | TBD | TBD |

### vs tiny-ecdsa

| Operation | Ultrafast | tiny-ecdsa | Speedup |
|-----------|----------|------------|---------|
| Scalar Mul | 110 μs | ~500 μs | 4.5× |
| Field Mul | 33 ns | ~200 ns | 6× |

---

## Key Optimizations

### Algorithm Level

| Optimization | Speedup | Description |
|--------------|---------|-------------|
| GLV Endomorphism | 1.5× | Reduces scalar bits by half |
| wNAF Encoding | 1.3× | 33% fewer non-zero digits |
| Precomputed Tables | 10× | Generator multiplication |

### Implementation Level

| Optimization | Speedup | Platform |
|--------------|---------|----------|
| BMI2/ADX Assembly | 3× | x86-64 |
| Dedicated Squaring | 1.25× | All |
| Branchless Add/Sub | 1.2× | RISC-V |
| 32-bit Hybrid Mul | 1.1× | CUDA |
| Batch Inversion | 3× | All |

---

## Running Benchmarks

### CPU Benchmark

```bash
# Build
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSECP256K1_BUILD_BENCH=ON
cmake --build build -j

# Run
./build/cpu/bench_comprehensive
```

### CUDA Benchmark

```bash
# Build
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DSECP256K1_BUILD_CUDA=ON \
  -DSECP256K1_BUILD_BENCH=ON

cmake --build build -j

# Run
./build/cuda/secp256k1_cuda_bench
```

---

## Methodology

### CPU Benchmarks

- **Warm-up:** 1 iteration (discarded)
- **Measurement:** 3 iterations, take median
- **Timer:** `std::chrono::high_resolution_clock`
- **Compiler:** `-O3 -march=native`

### CUDA Benchmarks

- **Warm-up:** 10 kernel launches (discarded)
- **Measurement:** 100 launches, average
- **Timer:** CUDA events with synchronization
- **Configuration:** Default thread/block counts

### Reproducibility

Results are saved to timestamped files:
```
benchmark-x86_64-linux-20260211-143000.txt
benchmark-risc-v-64-bit-linux-20260211-143000.txt
```

---

## Hardware Details

### x86-64 Test System

- **CPU:** Intel Core i5 (specific model TBD)
- **Features:** AVX2, BMI2, ADX
- **Memory:** DDR4
- **OS:** Linux

### RISC-V Test System

- **CPU:** RISC-V 64-bit
- **Extensions:** RV64GC, V (Vector 1.0)
- **Memory:** TBD
- **OS:** Linux

### CUDA Test System

- **GPU:** NVIDIA RTX 5060 Ti
- **VRAM:** 16GB
- **Compute:** sm_89
- **Driver:** 560+

---

## See Also

- [[CPU Guide]] - CPU optimization details
- [[CUDA Guide]] - GPU optimization details
- [[API Reference]] - Function documentation

