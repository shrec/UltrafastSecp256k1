# Performance Benchmarks

Benchmark results for UltrafastSecp256k1 across all supported platforms.

---

## Summary

| Platform | Field Mul | Generator Mul | Scalar Mul |
|----------|-----------|---------------|------------|
| x86-64 (i5, AVX2) | 33 ns | 5 μs | 110 μs |
| x86-64 (Clang 21, Win) | 17 ns (5×52) | 5 μs | 25 μs |
| RISC-V 64 (SiFive U74, LTO) | 95 ns | 33 μs | 154 μs |
| ARM64 (RK3588, A76) | 74 ns | 14 μs | 131 μs |
| ESP32-S3 (LX7, 240 MHz) | 7,458 ns | 2,483 μs | — |
| ESP32 (LX6, 240 MHz) | 6,993 ns | 6,203 μs | — |
| STM32F103 (CM3, 72 MHz) | 15,331 ns | 37,982 μs | — |
| CUDA (RTX 5060 Ti) | 0.2 ns | 217.7 ns | 225.8 ns |
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
| Field Mul (5×52) | 17 ns | `__int128` lazy reduction |
| Field Square (5×52) | 14 ns | |
| Field Add | 1 ns | |
| Field Negate | 1 ns | |
| Field Inverse | 1 μs | Fermat's little theorem |
| Point Add | 159 ns | Jacobian coordinates |
| Point Double | 98 ns | |
| Point Scalar Mul (k×P) | 25 μs | GLV + 5×52 + Shamir |
| Generator Mul (k×G) | 5 μs | Precomputed tables |
| ECDSA Sign | 8 μs | RFC 6979 |
| ECDSA Verify | 31 μs | Shamir + GLV |
| Schnorr Sign (BIP-340) | 14 μs | |
| Schnorr Verify (BIP-340) | 33 μs | |
| Batch Inverse (n=100) | 84 ns/elem | Montgomery's trick |
| Batch Inverse (n=1000) | 88 ns/elem | |

---

## RISC-V 64 Benchmarks

**Hardware:** Milk-V Mars (SiFive U74, RV64GC + Zba + Zbb)  
**OS:** Linux  
**Compiler:** Clang 21.1.8, `-mcpu=sifive-u74 -march=rv64gc_zba_zbb`  
**Assembly:** RISC-V native assembly  
**LTO:** ThinLTO enabled (auto-detected)

| Operation | Time | Notes |
|-----------|------|-------|
| Field Mul | 95 ns | Optimized carry chain |
| Field Square | 70 ns | Dedicated squaring |
| Field Add | 11 ns | Branchless |
| Field Sub | 11 ns | Branchless |
| Field Negate | 8 ns | Branchless |
| Field Inverse | 4 μs | Fermat's little theorem |
| Point Add | 1 μs | Jacobian coordinates |
| Point Double | 595 ns | |
| Point Scalar Mul (k×P) | 154 μs | GLV + wNAF |
| Generator Mul (k×G) | 33 μs | Precomputed tables |
| ECDSA Sign | 67 μs | RFC 6979 |
| ECDSA Verify | 186 μs | Shamir + GLV |
| Schnorr Sign (BIP-340) | 86 μs | |
| Schnorr Verify (BIP-340) | 216 μs | |

### RISC-V Optimization Gains (vs generic RV64GC build)

| Optimization | Speedup | Applied To |
|--------------|---------|------------|
| `-mcpu=sifive-u74` targeting | 1.3× | All operations |
| ThinLTO (cross-TU inlining) | 1.1× | Point/scalar ops |
| Native assembly | 2-3× | Field mul/square |
| Branchless algorithms | 1.2× | Field add/sub |
| Fast modular reduction | 1.5× | All field ops |
| Carry chain optimization | 1.3× | Multiplication |

---

## CUDA Benchmarks

**Hardware:** NVIDIA RTX 5060 Ti (36 SMs, 2602 MHz, 15847 MB, 128-bit bus)  
**CUDA:** 12.0, Compute 12.0 (Blackwell)  
**Architecture:** sm_86;sm89  
**Build:** Clang 19 + nvcc, Release, -O3 --use_fast_math

### Core ECC Operations

| Operation | Time/Op | Throughput | Notes |
|-----------|---------|------------|-------|
| Field Mul | 0.2 ns | 4,142 M/s | Kernel-only, batch 1M |
| Field Add | 0.2 ns | 4,130 M/s | Kernel-only, batch 1M |
| Field Inv | 10.2 ns | 98.35 M/s | Kernel-only, batch 64K |
| Point Add | 1.6 ns | 619 M/s | Kernel-only, batch 256K |
| Point Double | 0.8 ns | 1,282 M/s | Kernel-only, batch 256K |
| Scalar Mul (P×k) | 225.8 ns | 4.43 M/s | Kernel-only, batch 64K |
| Generator Mul (G×k) | 217.7 ns | 4.59 M/s | Kernel-only, batch 128K |
| Affine Add | 0.4 ns | 2,532 M/s | Kernel-only, batch 256K |
| Affine Lambda | 0.6 ns | 1,654 M/s | Kernel-only, batch 256K |
| Affine X-Only | 0.4 ns | 2,328 M/s | Kernel-only, batch 256K |
| Batch Inv | 2.9 ns | 340 M/s | Kernel-only, batch 64K |
| Jac→Affine | 14.9 ns | 66.9 M/s | Kernel-only, batch 64K |

### GPU Signature Operations

> **No other open-source GPU library provides secp256k1 ECDSA + Schnorr sign/verify on GPU.**

| Operation | Time/Op | Throughput | Notes |
|-----------|---------|------------|-------|
| ECDSA Sign | 204.8 ns | 4.88 M/s | RFC 6979, low-S, batch 16K |
| ECDSA Verify | 410.1 ns | 2.44 M/s | Shamir + GLV, batch 16K |
| ECDSA Sign + Recid | 311.5 ns | 3.21 M/s | Recoverable, batch 16K |
| Schnorr Sign (BIP-340) | 273.4 ns | 3.66 M/s | Tagged hash midstates, batch 16K |
| Schnorr Verify (BIP-340) | 354.6 ns | 2.82 M/s | X-only pubkey, batch 16K |

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
| Field Inv | 10.2 ns | 14.3 ns | **CUDA 1.40×** |
| Point Double | 0.8 ns | 0.9 ns | CUDA 1.13× |
| Point Add | 1.6 ns | 1.6 ns | Tie |
| Scalar Mul (kG) | 217.7 ns | 295.1 ns | **CUDA 1.36×** |
| ECDSA Sign | 204.8 ns | — | CUDA only |
| ECDSA Verify | 410.1 ns | — | CUDA only |
| Schnorr Sign | 273.4 ns | — | CUDA only |
| Schnorr Verify | 354.6 ns | — | CUDA only |

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
| Field Inv | 10.2 ns | 14.3 ns | 106.4 ns |
| Point Double | 0.8 ns | 0.9 ns | 5.1 ns |
| Point Add | 1.6 ns | 1.6 ns | 10.1 ns |
| Scalar Mul | 225.8 ns | 295.1 ns | 2.94 μs |
| Generator Mul | 217.7 ns | 295.1 ns | 3.00 μs |
| ECDSA Sign | 204.8 ns | — | — |
| ECDSA Verify | 410.1 ns | — | — |
| Schnorr Sign | 273.4 ns | — | — |
| Schnorr Verify | 354.6 ns | — | — |

> **Note:** CUDA/OpenCL — RTX 5060 Ti (36 SMs, 2602 MHz, GDDR7 256 GB/s).  
> Metal — M3 Pro (18 GPU cores, ~150 GB/s unified memory bandwidth).  
> RTX 5060 Ti has ~8× more compute throughput; Metal's advantage is in unified memory zero-copy I/O.

---

## Android ARM64 Benchmarks

**Hardware:** RK3588 (Cortex-A76 @ 2.256 GHz, pinned to big cores)  
**OS:** Android  
**Compiler:** NDK r26, Clang 17.0.2  
**Assembly:** ARM64 inline (MUL/UMULH)  
**Field:** 10×26 (optimal for ARM64)

| Operation | Time | Notes |
|-----------|------|-------|
| Field Mul | 74 ns | ARM64 MUL/UMULH, 10×26 |
| Field Square | 50 ns | |
| Field Add | 8 ns | |
| Field Negate | 18 ns | |
| Field Inverse | 2 μs | Fermat's theorem |
| Point Add | 992 ns | Jacobian coordinates |
| Point Double | 548 ns | |
| Generator Mul (k×G) | 14 μs | Precomputed tables |
| Scalar Mul (k×P) | 131 μs | GLV + wNAF |
| ECDSA Sign | 30 μs | RFC 6979 |
| ECDSA Verify | 153 μs | Shamir + GLV |
| Schnorr Sign (BIP-340) | 38 μs | |
| Schnorr Verify (BIP-340) | 173 μs | |
| Batch Inverse (n=100) | 265 ns/elem | Montgomery's trick |
| Batch Inverse (n=1000) | 240 ns/elem | |

ARM64 10×26 representation with MUL/UMULH assembly provides optimal field arithmetic performance.

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

UltrafastSecp256k1 v3.6.0  
Benchmarks updated: 2026-02-20

