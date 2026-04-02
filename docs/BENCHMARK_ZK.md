# Zero-Knowledge Proof Benchmarks

Performance benchmarks for UltrafastSecp256k1 ZK proof operations across CPU and GPU.

---

## Summary

| Operation | CPU (i5-14400F) | GPU CUDA (RTX 5060 Ti) | GPU/CPU Speedup |
|-----------|----------------:|-----------------------:|----------------:|
| Knowledge Prove | 24.3 us | 252.3 ns | **96x** |
| Knowledge Verify | 23.8 us | 749.9 ns | **32x** |
| DLEQ Prove | 42.4 us | 668.3 ns | **63x** |
| DLEQ Verify | 60.6 us | 1,919 ns | **32x** |
| Pedersen Commit | 29.7 us | 66.0 ns | **450x** |
| Bulletproof Range Prove | 13,619 us | 3,712 us | **3.7x** |
| Bulletproof Range Verify | 2,670 us | 765 us | **3.5x** |

---

## CPU Benchmarks

**Hardware:** Intel Core i5-14400F (P-core, Raptor Lake, pinned to core 0)
**OS:** Linux
**Compiler:** Clang 19.1.7, `-O3 -march=native`
**Assembly:** x86-64 with BMI2/ADX intrinsics
**Methodology:** bench_unified --suite core --passes 11, IQR outlier removal, median, 64-key pool
**Timer:** RDTSCP (2.501 GHz TSC)

### ZK Proof Operations

| Operation | Time/Op | Throughput | Path | Notes |
|-----------|--------:|----------:|------|-------|
| Pedersen Commit | 29.7 us | 33,670 op/s | FAST | v*H + r*G (two scalar muls) |
| Knowledge Prove | 24.3 us | 41,152 op/s | CT | Schnorr sigma protocol, deterministic nonce |
| Knowledge Verify | 23.8 us | 42,017 op/s | FAST | s*G == R + e*P |
| DLEQ Prove | 42.4 us | 23,585 op/s | CT | log_G(P) == log_H(Q), two CT scalar muls |
| DLEQ Verify | 60.6 us | 16,502 op/s | FAST | Two-base Shamir verification |
| Range Prove (64-bit) | 13,619 us | 73 op/s | CT | Bulletproof prover |
| Range Verify (64-bit) | 2,670 us | 375 op/s | FAST | MSM-optimized verifier |

### CT Overhead Analysis

Prove operations use the constant-time (CT) path for secret protection.
Verify operations use the fast (variable-time) path since inputs are public.

| Operation | Fast Scalar Mul | CT Scalar Mul | CT Overhead |
|-----------|----------------:|--------------:|------------:|
| Knowledge Prove | 1x ct_scalar_mul(G,k) | ~10.5 us | ~2.5x vs fast kG |
| DLEQ Prove | 2x ct_scalar_mul | ~21.0 us | ~2.5x vs fast kP |

---

## GPU CUDA Benchmarks

**Hardware:** NVIDIA RTX 5060 Ti (36 SMs, 2602 MHz, 15847 MB GDDR7, 128-bit bus)
**CUDA:** 12.0, Compute 12.0 (Blackwell, JIT from sm_89)
**Driver:** 580.126.09
**Build:** Clang 19 + nvcc, Release, -O3 --use_fast_math
**Methodology:** bench_zk, batch=4096 (Knowledge/DLEQ/Pedersen), batch=256 (Bulletproof), warmup=5, passes=11, median
**Timer:** CUDA events (ns/op = elapsed_ms * 1e6 / batch_size)

### ZK Proof Operations

| Operation | Time/Op | Throughput | Path | Notes |
|-----------|--------:|----------:|------|-------|
| Knowledge Prove (G) | 252.3 ns | 3,964 k/s | CT | Precomputed G table, batch 4K |
| Knowledge Verify | 749.9 ns | 1,334 k/s | FAST | s*G + e*P multi-scalar, batch 4K |
| DLEQ Prove | 668.3 ns | 1,496 k/s | CT | Two CT scalar muls + batch inv, batch 4K |
| DLEQ Verify | 1,919.1 ns | 521 k/s | FAST | Two-base verification, batch 4K |
| Pedersen Commit | 66.0 ns | 15,160 k/s | FAST | v*H + r*G, batch 4K |
| Range Prove (64-bit) | 3,711,570 ns | 0.27 k/s | CT | Bulletproof prover, batch 256 |
| Range Verify (64-bit) | 764,649 ns | 1.3 k/s | FAST | Full IPA verification, batch 256 |

### Kernel Resource Usage

| Kernel | Registers | Stack (bytes) | Notes |
|--------|----------:|--------------:|-------|
| knowledge_prove | 255 | 1,328 | Max registers, minimal spill |
| knowledge_verify | 166 | 816 | Low pressure |
| dleq_prove | 255 | 8,608 | Max registers, significant spill |
| dleq_verify | 229 | 1,680 | Near max |

DLEQ prove has high stack usage due to two `ct_scalar_mul` calls inside
`ct_dleq_prove_device` (each allocating ~4.8 KB of precomputed tables).
Functions marked `__noinline__` to reduce register pressure from inlining.

### Correctness Validation

Before timing, bench_zk validates:
- 0/4096 verify failures for Knowledge and DLEQ proofs
- 0/256 verify failures for Bulletproof range proofs

The benchmark exits with error if any verification fails.

---

## GPU vs CPU Comparison

### Per-Operation Latency

| Operation | CPU | GPU | GPU/CPU |
|-----------|----:|----:|--------:|
| Knowledge Prove | 24,292 ns | 252.3 ns | **96x** |
| Knowledge Verify | 23,830 ns | 749.9 ns | **32x** |
| DLEQ Prove | 42,370 ns | 668.3 ns | **63x** |
| DLEQ Verify | 60,607 ns | 1,919.1 ns | **32x** |
| Pedersen Commit | 29,718 ns | 66.0 ns | **450x** |
| Range Prove (64-bit) | 13,618,693 ns | 3,711,570 ns | **3.7x** |
| Range Verify (64-bit) | 2,669,843 ns | 764,649 ns | **3.5x** |

### Batch Throughput

| Batch Size | Knowledge Prove CPU | Knowledge Prove GPU | Speedup |
|-----------:|--------------------:|--------------------:|--------:|
| 1 | 24.3 us | ~264 ns + launch | CPU wins |
| 1,000 | 24.3 ms | ~0.27 ms | ~90x |
| 4,096 | 99.6 ms | 1.08 ms | **92x** |
| 100,000 | 2.43 s | ~26.4 ms | **92x** |

### Why Prove is Faster Than Verify on GPU

On CPU, prove and verify take similar time. On GPU, prove is **2.8x faster** than
verify for Knowledge proofs (263.7 vs 744.5 ns). This is because:

1. **Prove** uses `ct_generator_mul` with precomputed `__constant__` memory tables
   (zero allocation, high cache hit rate across threads)
2. **Verify** uses two variable-base scalar multiplications (`s*G` and `e*P`) which
   require per-thread table construction in registers/local memory
3. The constant-time path's regular access pattern paradoxically helps GPU occupancy
   (no thread divergence from early exits)

---

## Protocol Details

### Knowledge Proof (Schnorr Sigma)

Proves knowledge of `x` such that `P = x*G` without revealing `x`.

- **Proof size:** 64 bytes (R.x[32] + s[32])
- **Prove:** CT path (secret `x` and nonce `k`)
- **Verify:** FAST path (all inputs public)
- **Nonce:** Deterministic (RFC 6979-style, hedged with aux_rand)

### DLEQ Proof (Discrete Log Equality)

Proves `log_G(P) == log_H(Q)` (same secret for two bases) without revealing the secret.

- **Proof size:** 64 bytes (e[32] + s[32])
- **Prove:** CT path (2x ct_scalar_mul for R1=k*G, R2=k*H)
- **Verify:** FAST path (2x two-base Shamir: s*G vs R1+e*P, s*H vs R2+e*Q)
- **Use cases:** VRFs, provable ECDH, adaptor signatures

### Bulletproof Range Proof

Proves that a Pedersen commitment hides a value in [0, 2^64) without revealing it.

- **Proof size:** ~672 bytes (for 64-bit range)
- **Prove:** CT path, 64-bit decomposition, inner product argument
- **Verify:** FAST path, polynomial commitment check + full IPA verification
- **GPU implementation:** try-and-increment generator derivation, batch scalar inversion
- **Optimization:** MSM + Montgomery batch inversion (CPU: 1.93x speedup in v3.4+)
- **GPU/CPU:** Prove 3.7x, Verify 3.5x (Bulletproof is compute-heavy per-thread, lower parallelism)

---

## Reproducing

```bash
# CPU ZK benchmarks (part of bench_unified Section 8)
cmake --build build -j
./build/cpu/bench_unified --suite core --passes 11

# GPU ZK benchmarks
cmake --build build --target bench_zk -j
./build/cuda/bench_zk
```
