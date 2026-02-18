# Performance Benchmarks

Benchmark results for UltrafastSecp256k1 across all supported platforms.

---

## Summary

| Platform | Field Mul | Generator Mul | Scalar Mul |
|----------|-----------|---------------|------------|
| x86-64 (i5, AVX2) | 33 ns | 5 μs | 110 μs |
| x86-64 (Clang 21, Win) | 44 ns (4×64) / 23 ns (5×52) | 8 μs | 42 μs |
| RISC-V 64 (RVV) | 173 ns | 37 μs | 621 μs |
| ARM64 (RK3588) | 85 ns | 7.6 μs | 77.6 μs |
| ESP32-S3 (LX7, 240 MHz) | 7,458 ns | 2,483 μs | — |
| ESP32 (LX6, 240 MHz) | 6,993 ns | 6,203 μs | — |
| STM32F103 (CM3, 72 MHz) | 15,331 ns | 37,982 μs | — |
| CUDA (RTX 5060 Ti) | 0.2 ns | 216.1 ns | 266.5 ns |
| OpenCL (RTX 5060 Ti) | 0.2 ns | 295.1 ns | — |
| Metal (Apple M3 Pro) | 1.9 ns | 3.00 μs | 2.94 μs |

---

## x86-64 Benchmarks

### x86-64 / Linux (i5, Clang 19.1.7, AVX2)

**Hardware:** Intel Core i5 (AVX2, BMI2, ADX)  
**OS:** Linux  
**Compiler:** Clang 19.1.7  
**Assembly:** x86-64 with BMI2/ADX intrinsics  
**SIMD:** AVX2

| Operation | Time | Notes |
|-----------|------|-------|
| Field Mul | 33 ns | Using mulx/adcx/adox |
| Field Square | 32 ns | Optimized squaring |
| Field Add | 11 ns | |
| Field Sub | 12 ns | |
| Field Inverse | 5 μs | Fermat's little theorem |
| Point Add | 521 ns | Jacobian coordinates |
| Point Double | 278 ns | |
| Point Scalar Mul | 110 μs | GLV + wNAF |
| Generator Mul | 5 μs | Precomputed tables |
| Batch Inverse (n=100) | 140 ns/elem | Montgomery's trick |
| Batch Inverse (n=1000) | 92 ns/elem | |

### x86-64 / Windows (Clang 21.1.0, AVX2)

**Hardware:** x86-64 (AVX2)  
**OS:** Windows  
**Compiler:** Clang 21.1.0  
**Assembly:** x86-64 ASM enabled  
**SIMD:** AVX2

| Operation | Time | Notes |
|-----------|------|-------|
| Field Mul (4×64) | 44 ns | Portable representation |
| Field Mul (5×52) | 23 ns | `__int128` lazy reduction |
| Field Square (4×64) | 33 ns | |
| Field Square (5×52) | 14 ns | |
| Field Add | 9 ns | |
| Field Sub | 10 ns | |
| Field Inverse | 5 μs | Fermat's little theorem |
| Point Add | 821 ns | Jacobian coordinates |
| Point Double | 380 ns | |
| Point Scalar Mul (k×P) | 42 μs | GLV + 5×52 + Shamir |
| Generator Mul (k×G) | 8 μs | Precomputed tables |
| Batch Inverse (n=100) | 166 ns/elem | Montgomery's trick |
| Batch Inverse (n=1000) | 120 ns/elem | |

---

## RISC-V 64 Benchmarks

**Hardware:** RISC-V 64-bit (RV64GC + V extension)  
**OS:** Linux  
**Compiler:** Clang 21.1.8  
**Assembly:** RISC-V native assembly  
**SIMD:** RVV 1.0

| Operation | Time | Notes |
|-----------|------|-------|
| Field Mul | 198 ns | Optimized carry chain |
| Field Square | 177 ns | Dedicated squaring |
| Field Add | 34 ns | Branchless |
| Field Sub | 31 ns | Branchless |
| Field Inverse | 18 μs | |
| Point Add | 3 μs | |
| Point Double | 1 μs | |
| Point Scalar Mul | 672 μs | GLV + wNAF |
| Generator Mul | 40 μs | Precomputed tables |
| Batch Inverse (n=100) | 765 ns/elem | |
| Batch Inverse (n=1000) | 615 ns/elem | |

### RISC-V Optimization Gains

| Optimization | Speedup | Applied To |
|--------------|---------|------------|
| Native assembly | 2-3× | Field mul/square |
| Branchless algorithms | 1.2× | Field add/sub |
| Fast modular reduction | 1.5× | All field ops |
| RVV vectorization | 1.1× | Batch operations |
| Carry chain optimization | 1.3× | Multiplication |

---

## CUDA Benchmarks

**Hardware:** NVIDIA RTX 5060 Ti (36 SMs, 2602 MHz, 15847 MB, 128-bit bus)  
**CUDA:** 12.0, Compute 12.0 (Blackwell)  
**Architecture:** sm_86;sm89  
**Build:** Clang 19 + nvcc, Release, -O3 --use_fast_math

| Operation | Time/Op | Throughput | Notes |
|-----------|---------|------------|-------|
| Field Mul | 0.2 ns | 4,139 M/s | Kernel-only, batch 1M |
| Field Add | 0.2 ns | 4,122 M/s | Kernel-only, batch 1M |
| Field Inv | 12.1 ns | 82.65 M/s | Kernel-only, batch 64K |
| Point Add | 1.1 ns | 916 M/s | Kernel-only, batch 256K |
| Point Double | 0.7 ns | 1,352 M/s | Kernel-only, batch 256K |
| Scalar Mul (P*k) | 266.5 ns | 3.75 M/s | Kernel-only, batch 64K |
| Generator Mul (G*k) | 216.1 ns | 4.63 M/s | Kernel-only, batch 128K |

---

## OpenCL Benchmarks

**Hardware:** NVIDIA RTX 5060 Ti (36 CUs, 2602 MHz)  
**OpenCL:** 3.0 CUDA, Driver 580.126.09  
**Build:** Clang 19, Release, -O3, PTX inline assembly  

### Kernel-Only Timing (no buffer alloc/copy overhead)

| Operation | Time/Op | Throughput | Notes |
|-----------|---------|------------|-------|
| Field Mul | 0.2 ns | 4,137 M/s | batch 1M |
| Field Add | 0.2 ns | 4,124 M/s | batch 1M |
| Field Sub | 0.2 ns | 4,119 M/s | batch 1M |
| Field Sqr | 0.2 ns | 5,985 M/s | batch 1M |
| Field Inv | 14.3 ns | 69.97 M/s | batch 1M |
| Point Double | 0.9 ns | 1,139 M/s | batch 256K |
| Point Add | 1.6 ns | 630.6 M/s | batch 256K |
| kG (kernel) | 295.1 ns | 3.39 M/s | batch 256K |

### End-to-End Timing (including buffer transfers)

| Operation | Time/Op | Throughput | Notes |
|-----------|---------|------------|-------|
| Field Add | 27.3 ns | 36.67 M/s | batch 1M |
| Field Mul | 27.7 ns | 36.07 M/s | batch 1M |
| Field Inv | 29.0 ns | 34.43 M/s | batch 1M |
| Point Double | 58.4 ns | 17.11 M/s | batch 1M |
| Point Add | 111.9 ns | 8.94 M/s | batch 1M |
| kG (batch=65K) | 307.7 ns | 3.25 M/s | |
| kG (batch=16K) | 311.6 ns | 3.21 M/s | |

### CUDA / OpenCL Configuration

```cpp
// Optimal settings for RTX 5060 Ti
#define SECP256K1_CUDA_USE_HYBRID_MUL 1  // 32-bit hybrid (~10% faster)
#define SECP256K1_CUDA_USE_MONTGOMERY 0  // Standard domain (faster for search)
```

### CUDA vs OpenCL Kernel-Only Comparison (RTX 5060 Ti)

| Operation | CUDA | OpenCL | Faster |
|-----------|------|--------|--------|
| Field Mul | 0.2 ns | 0.2 ns | Tie |
| Field Add | 0.2 ns | 0.2 ns | Tie |
| Field Inv | 12.1 ns | 14.3 ns | CUDA 1.18× |
| Point Double | 0.7 ns | 0.9 ns | **CUDA 1.29×** |
| Point Add | 1.1 ns | 1.6 ns | **CUDA 1.45×** |
| Scalar Mul (kG) | 216.1 ns | 295.1 ns | **CUDA 1.37×** |

---

## Apple Metal Benchmarks

**Hardware:** Apple M3 Pro (18 GPU cores, Unified Memory 18 GB)  
**OS:** macOS Sequoia  
**Metal:** Metal 2.4, MSL macos-metal2.4  
**Limb Model:** 8×32-bit Comba (no 64-bit int in MSL)  
**Build:** AppleClang, Release, -O3, ARC

| Operation | Time/Op | Throughput | Notes |
|-----------|---------|------------|-------|
| Field Mul | 1.9 ns | 527 M/s | Comba product scanning, batch 1M |
| Field Add | 1.0 ns | 990 M/s | Branchless, batch 1M |
| Field Sub | 1.1 ns | 892 M/s | Branchless, batch 1M |
| Field Sqr | 1.1 ns | 872 M/s | Comba + symmetry, batch 1M |
| Field Inv | 106.4 ns | 9.40 M/s | Fermat (a^(p-2)), batch 64K |
| Point Add | 10.1 ns | 98.6 M/s | Jacobian, batch 256K |
| Point Double | 5.1 ns | 196 M/s | dbl-2001-b, batch 256K |
| Scalar Mul (P×k) | 2.94 μs | 0.34 M/s | 4-bit windowed, batch 64K |
| Generator Mul (G×k) | 3.00 μs | 0.33 M/s | 4-bit windowed, batch 128K |

### Metal vs CUDA vs OpenCL — GPU Comparison

| Operation | CUDA (RTX 5060 Ti) | OpenCL (RTX 5060 Ti) | Metal (M3 Pro) |
|-----------|-------------------|---------------------|----------------|
| Field Mul | 0.2 ns | 0.2 ns | 1.9 ns |
| Field Add | 0.2 ns | 0.2 ns | 1.0 ns |
| Field Inv | 12.1 ns | 14.3 ns | 106.4 ns |
| Point Double | 0.7 ns | 0.9 ns | 5.1 ns |
| Point Add | 1.1 ns | 1.6 ns | 10.1 ns |
| Scalar Mul | 266.5 ns | 295.1 ns | 2.94 μs |
| Generator Mul | 216.1 ns | 295.1 ns | 3.00 μs |

> **შენიშვნა:** CUDA/OpenCL — RTX 5060 Ti (36 SMs, 2602 MHz, GDDR7 256 GB/s).  
> Metal — M3 Pro (18 GPU cores, ~150 GB/s unified memory bandwidth).  
> RTX 5060 Ti-ს აქვს ~8× მეტი compute throughput; Metal-ის უპირატესობა unified memory zero-copy I/O-ში.

---

## Android ARM64 Benchmarks

**Hardware:** RK3588 (Cortex-A55/A76 @ 2.4 GHz)  
**OS:** Android  
**Compiler:** NDK r27, Clang 18  
**Assembly:** ARM64 inline (MUL/UMULH)

| Operation | Time | Notes |
|-----------|------|-------|
| Field Mul | 85 ns | ARM64 MUL/UMULH |
| Field Square | 66 ns | |
| Field Add | 18 ns | |
| Field Sub | 16 ns | |
| Field Inverse | 2,621 ns | Fermat's theorem |
| Scalar Mul | 105 ns | |
| Point Add | 9,329 ns | |
| Point Double | 8,711 ns | |
| Fast Scalar × G | 7.6 μs | Precomputed tables |
| Fast Scalar × P | 77.6 μs | Non-generator |
| CT Scalar × G | 545 μs | Constant-time |
| CT ECDH | 545 μs | Full CT |

ARM64 inline assembly provides **~5× speedup** over portable C++.

---

## ESP32-S3 Benchmarks (Embedded)

**Hardware:** ESP32-S3 (Xtensa LX7 Dual Core @ 240 MHz)  
**OS:** ESP-IDF v5.5.1  
**Assembly:** None (portable C++, no `__int128`)

| Operation | Time | Notes |
|-----------|------|-------|
| Field Mul | 7,458 ns | |
| Field Square | 7,592 ns | |
| Field Add | 636 ns | |
| Field Inv | 844 μs | |
| Scalar × G | 2,483 μs | Generator mul |

All 35 library self-tests pass.

---

## ESP32-PICO-D4 Benchmarks (Embedded)

**Hardware:** ESP32-PICO-D4 (Xtensa LX6 Dual Core @ 240 MHz)  
**OS:** ESP-IDF v5.5.1  
**Assembly:** None (portable C++, no `__int128`)

| Operation | Time | Notes |
|-----------|------|-------|
| Field Mul | 6,993 ns | |
| Field Square | 6,247 ns | |
| Field Add | 985 ns | |
| Field Inv | 609 μs | |
| Scalar × G | 6,203 μs | Generator mul |
| CT Scalar × G | 44,810 μs | Constant-time |
| CT Add (complete) | 249,672 ns | |
| CT Dbl | 87,113 ns | |
| CT/Fast ratio | 6.5× | |

All 35 self-tests + 8 CT tests pass.

---

## STM32F103 Benchmarks (Embedded)

**Hardware:** STM32F103ZET6 (ARM Cortex-M3 @ 72 MHz)  
**Compiler:** ARM GCC 13.3.1, -O3  
**Assembly:** ARM Cortex-M3 inline (UMULL/ADDS/ADCS)

| Operation | Time | Notes |
|-----------|------|-------|
| Field Mul | 15,331 ns | ARM inline asm |
| Field Square | 12,083 ns | ARM inline asm |
| Field Add | 4,139 ns | Portable C++ |
| Field Inv | 1,645 μs | |
| Scalar × G | 37,982 μs | Generator mul |

All 35 library self-tests pass.

---

## Embedded Cross-Platform Comparison

| Operation | ESP32-S3 (LX7) | ESP32 (LX6) | STM32F103 (M3) |
|-----------|:--------------:|:-----------:|:-------------:|
| | 240 MHz | 240 MHz | 72 MHz |
| Field Mul | 7,458 ns | 6,993 ns | 15,331 ns |
| Field Square | 7,592 ns | 6,247 ns | 12,083 ns |
| Field Add | 636 ns | 985 ns | 4,139 ns |
| Field Inv | 844 μs | 609 μs | 1,645 μs |
| Scalar × G | 2,483 μs | 6,203 μs | 37,982 μs |

---

## Comparison with Other Libraries

### vs Bitcoin Core libsecp256k1 (Head-to-Head)

**Test Conditions:** Same machine, same compiler (Clang 21.1.0), same OS (Windows x64), Release builds.  
**libsecp256k1:** Latest master, built with all modules (recovery, schnorrsig, extrakeys, ecdh).

#### High-Level Operations

| Operation | UltrafastSecp256k1 | libsecp256k1 | Winner |
|-----------|-------------------:|-------------:|--------|
| ECDSA Sign (k×G dominated) | **8.5 μs** | 26.2 μs | ✅ **Ours 3.1×** |
| EC Keygen | **8.5 μs** | 17.9 μs | ✅ **Ours 2.1×** |
| Schnorr Sign | **8.5 μs** | 18.2 μs | ✅ **Ours 2.1×** |
| ECDSA Verify (k₁×G+k₂×Q) | 50 μs | **37.3 μs** | ⚠️ Theirs 1.3× |
| ECDSA Recover | — | 37.8 μs | — |
| ECDH | — | 36.3 μs | — |

*ECDSA Sign dominated by k×G generator multiplication, where our precomputed tables excel.*  
*ECDSA Verify requires k₁×G + k₂×Q; libsecp256k1 uses Strauss endomorphism-aware multi-scalar, we use separate computation.*  
*Since v3.4: GLV endomorphism + 5×52 lazy-reduction + Shamir's trick reduced K×Q from 132→42 μs (3.1× faster).*

#### Internal Field Operations

| Operation | UltrafastSecp256k1 | libsecp256k1 | Ratio |
|-----------|-------------------:|-------------:|------:|
| Field Mul (5×52) | 22 ns | **15.3 ns** | 1.44× |
| Field Square (5×52) | 17 ns | **13.6 ns** | 1.25× |
| Field Mul (4×64 portable) | 50 ns | 15.3 ns | 3.27× |
| Field Add | 10 ns | **2.6 ns** | 3.85× |
| Field Normalize | — | 7.7 ns | — |
| Field Inverse | 5 μs | **1.71 μs** | 2.92× |
| Field Sqrt | — | 3.73 μs | — |

#### Internal Point/Group Operations

| Operation | UltrafastSecp256k1 | libsecp256k1 | Ratio |
|-----------|-------------------:|-------------:|------:|
| Point Double | 380 ns | **103 ns** | 3.69× |
| Point Add (Jacobian) | 821 ns | **255 ns** | 3.22× |
| Point Add (Affine mixed) | 689 ns | **172 ns** | 4.01× |
| Generator Mul (k×G) | **8 μs** | 15.3 μs | **0.52×** |
| ecmult\_const (k×P) | 42 μs | **33.1 μs** | 1.27× |

*Since v3.4: k×P uses GLV decomposition + 5×52 field arithmetic + Shamir's trick.*  
*Previous k×P was 132 μs (wNAF only) → 42 μs with GLV+5×52 (3.1× improvement).*

#### Internal Scalar Operations

| Operation | UltrafastSecp256k1 | libsecp256k1 | Ratio |
|-----------|-------------------:|-------------:|------:|
| Scalar Add | 3 ns | **5.4 ns** | 0.56× ✅ |
| Scalar Mul | — | 33.9 ns | — |
| Scalar Inverse | — | 1.73 μs | — |
| Scalar Inverse (var) | — | 1.10 μs | — |

#### Analysis Summary

| Metric | Winner | Details |
|--------|--------|---------|
| **Generator Multiplication (k×G)** | ✅ UltrafastSecp256k1 | 1.8–3× faster (larger precomputed tables) |
| **ECDSA Signing** | ✅ UltrafastSecp256k1 | 3× faster (dominated by k×G) |
| **Field arithmetic (atomic)** | ⚠️ libsecp256k1 | 10+ years of hand-tuned x86-64 assembly |
| **Point operations (atomic)** | ⚠️ libsecp256k1 | Tighter formulas + field advantage compounds |
| **Arbitrary-Point Scalar Mul (k×P)** | ≈ parity | GLV+5×52+Shamir → 42 μs vs 33 μs (1.27×) |
| **ECDSA Verification** | ⚠️ libsecp256k1 | Strauss multi-scalar w/ endomorphism (1.3×) |
| **Platform breadth** | ✅ UltrafastSecp256k1 | x86, ARM64, RISC-V, Xtensa, Cortex-M, CUDA, OpenCL, Metal |

*libsecp256k1 is the gold standard for single-threaded x86 CPU secp256k1. UltrafastSecp256k1 trades some single-op latency for portability, GPU backends, and faster generator-based operations.*

### vs tiny-ecdsa (Estimated)

| Operation | UltrafastSecp256k1 | tiny-ecdsa | Speedup |
|-----------|-------------------|------------|---------|
| Scalar Mul | 42 μs | ~500 μs | ~12× |
| Field Mul | 44 ns | ~200 ns | ~4.5× |

---

## Specialized Benchmark Results (Windows x64, Clang 21.1.0)

### Field Representation Comparison (5×52 vs 4×64)

5×52 uses `__int128` with lazy carry reduction — fewer normalizations = faster chains.

| Operation | 4×64 (ns) | 5×52 (ns) | 5×52 Speedup |
|-----------|----------:|----------:|-------------:|
| Multiplication | 41.9 | 15.2 | **2.76×** |
| Squaring | 31.2 | 12.8 | **2.44×** |
| Addition | 4.3 | 1.6 | **2.69×** |
| Negation | 7.6 | 2.4 | **3.13×** |
| Add chain (4 ops) | 33.2 | 8.6 | **3.84×** |
| Add chain (8 ops) | 65.4 | 16.4 | **3.98×** |
| Add chain (16 ops) | 137.7 | 30.3 | **4.55×** |
| Add chain (32 ops) | 285.9 | 57.0 | **5.01×** |
| Add chain (64 ops) | 566.8 | 117.1 | **4.84×** |
| Point-Add simulation | 428.3 | 174.8 | **2.45×** |
| 256 squarings | 9,039 | 4,055 | **2.23×** |

*Conclusion: 5×52 is 2.0–5.0× faster across all operations. The advantage grows for addition-heavy chains (lazy reduction amortizes normalization cost).*

### Field Representation Comparison (10×26 vs 4×64)

10×26 is the 32-bit target representation — useful for embedded and GPU where 64-bit multiply is expensive.

| Operation | 4×64 (ns) | 10×26 (ns) | 10×26 Speedup |
|-----------|----------:|----------:|--------------:|
| Addition | 4.7 | 1.8 | **2.57×** |
| Multiplication | ~39 | ~39 | ~1× (tie) |
| Add chain (16 ops) | wide | 3.3× faster | — |

### Constant-Time (CT) Layer Performance

CT layer provides side-channel resistance at the cost of performance.

| Operation | Fast | CT | Overhead |
|-----------|------:|------:|--------:|
| Field Mul | 36 ns | 55 ns | 1.50× |
| Field Square | 34 ns | 43 ns | 1.28× |
| Field Inverse | 3.0 μs | 14.2 μs | 4.80× |
| Scalar Add | 3 ns | 10 ns | 3.02× |
| Scalar Sub | 2 ns | 10 ns | 6.33× |
| Point Add | 0.65 μs | 1.63 μs | 2.50× |
| Point Double | 0.36 μs | 0.67 μs | 1.88× |
| Scalar Mul (k×P) | 130 μs | 322 μs | 2.49× |
| Generator Mul (k×G) | 7.6 μs | 310 μs | 40.8× |

*Generator mul overhead (40×) is high because CT disables precomputed variable-time table lookups. For signing with side-channel requirements, CT scalar mul (2.49× overhead) is the relevant metric.*

### Multi-Scalar Multiplication (ECDSA Verify Path)

| Method | Time | Description |
|--------|------:|------------|
| Separate (prod-like) | 137.4 μs | k₁×G (precompute) + k₂×Q (variable-base) |
| Separate (variable) | 351.5 μs | Both via fixed-window variable-base |
| Shamir interleaved | 155.2 μs | Merged stream — fewer doublings |
| Windowed Shamir | 9.2 μs | Optimized multi-scalar |
| JSF (Joint Sparse Form) | 9.5 μs | Joint encoding of both scalars |

### Atomic ECC Building Blocks

| Operation | Time | Formula Cost |
|-----------|------:|-------------|
| Point Add (immutable) | 959 ns | 12M + 4S + alloc |
| Point Add (in-place) | 1,859 ns | 12M + 4S |
| Point Double (immutable) | 673 ns | 4M + 4S + alloc |
| Point Double (in-place) | 890 ns | 4M + 4S |
| Point Negation | 11 ns | Y := −Y |
| Point Triple | 1,585 ns | 2×P + P |
| To Affine conversion | 15,389 ns | 1 inverse + 2–3 mul |
| Field S/M ratio | 0.818 | (ideal: ~0.80) |
| Field I/M ratio | 78× | Inverse is expensive — use Jacobian! |

---

## Available Benchmark Targets

All targets registered in CMake. Build with `cmake --build build -j` then run from `build/cpu/`.

| Target | What It Measures |
|--------|-----------------|
| `bench_comprehensive` | Full field/point/batch/5×52/10×26 suite — primary benchmark |
| `bench_scalar_mul` | k×G and k×P with wNAF overhead analysis |
| `bench_ct` | Fast (`fast::`) vs Constant-Time (`ct::`) layer comparison |
| `bench_atomic_operations` | Individual ECC building block latencies (point add/dbl, field mul/sqr/inv) |
| `bench_field_52` | 4×64 vs 5×52 field representation: single ops + add chains + ECC simulation |
| `bench_field_26` | 4×64 vs 10×26 field representation comparison |
| `bench_field_mul_kernels` | BMI2 `mulx` kernel micro-benchmark |
| `bench_ecdsa_multiscalar` | k₁×G + k₂×Q: Shamir interleaved vs separate vs variable-base |
| `bench_jsf_vs_shamir` | JSF (Joint Sparse Form) vs Windowed Shamir multi-scalar |
| `bench_adaptive_glv` | GLV window size sweep (w=8 to w=20) |
| `bench_glv_decomp_profile` | GLV decomposition profiling |
| `bench_comprehensive_riscv` | RISC-V specific benchmark suite |

---

## Benchmark Methodology

### CPU Benchmarks

1. **Warm-up:** 1 iteration discarded
2. **Measurement:** 3 iterations, take median
3. **Timer:** `std::chrono::high_resolution_clock`
4. **Compiler flags:** `-O3 -march=native`

### CUDA Benchmarks

1. **Warm-up:** 10 kernel launches discarded
2. **Measurement:** 100 launches, average
3. **Timer:** CUDA events
4. **Sync:** Full device synchronization between measurements

### Reproducibility

```bash
# Run CPU benchmark
./build/cpu/bench_comprehensive

# Run CUDA benchmark
./build/cuda/secp256k1_cuda_bench

# Results saved to: benchmark-<platform>-<date>.txt
```

---

## Optimization History

### RISC-V Timeline

| Date | Field Mul | Scalar Mul | Change |
|------|-----------|------------|--------|
| 2026-02-08 | 307 ns | 954 μs | Initial |
| 2026-02-09 | 205 ns | 676 μs | Carry optimization |
| 2026-02-10 | 198 ns | 672 μs | Square optimization |
| 2026-02-10 | 198 ns | 672 μs | **Current** |

### Key Optimizations Applied

1. **Branchless field operations** - Eliminates unpredictable branches
2. **Optimized carry propagation** - Reduces instruction count
3. **Dedicated squaring routine** - 25% fewer multiplications than generic mul
4. **GLV decomposition** - ~50% reduction in scalar bits
5. **wNAF encoding** - ~33% fewer point additions
6. **Precomputed tables** - Generator multiplication 10× faster

---

## Future Optimizations

### Planned

- [ ] AVX-512 vectorization (x86-64)
- [ ] Multi-threaded batch operations
- [x] ARM64 NEON/MUL assembly (**DONE** — ~5× speedup)
- [x] OpenCL backend (**DONE** — 3.39M kG/s)
- [x] Apple Metal backend (**DONE** — 527M field_mul/s, M3 Pro)
- [x] Shared POD types across backends
- [x] ARM64 inline assembly (MUL/UMULH)

### Experimental

- [ ] AVX-512 vectorization (x86-64)
- [ ] Multi-threaded batch operations
- [x] Montgomery domain for CUDA (mixed results)
- [x] 8×32-bit hybrid limb representation (**DONE** — 1.10× faster mul)
- [x] Constant-time side-channel resistance (CT layer implemented)

---

## Version

UltrafastSecp256k1 v3.3.0  
Benchmarks updated: 2026-02-18

