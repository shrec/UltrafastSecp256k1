# CUDA vs OpenCL Comparison -- NVIDIA RTX 5060 Ti

**Date:** 2026-02-14 (updated with optimized OpenCL kernels)  
**Hardware:** NVIDIA GeForce RTX 5060 Ti (36 SMs, 2602 MHz, 16 GB, 128-bit bus)  
**OS:** Linux x86_64  
**CUDA:** 12.0, sm_89, batch=1M (field), 65K-131K (point/scalar)  
**OpenCL:** 3.0 CUDA, Driver 580.126.09, batch=65K  

---

## Optimizations Applied (OpenCL)

1. **field_mul**: Fully unrolled 4x4 schoolbook multiplication (no loops)
2. **field_sqr**: Fully unrolled with separate off-diagonal/diagonal phases
3. **field_inv**: Addition chain (Fermat chain) -- replaced naive 256-bit binary exponentiation
4. **scalar_mul**: wNAF window-5 with 8-entry precomputed table -- replaced simple double-and-add
5. **Benchmark**: Batch throughput measurement (amortized, same methodology as CUDA)

---

## Batch Throughput Comparison (Amortized Per-Element)

| Operation | CUDA ns/op | CUDA M/s | OpenCL ns/op | OpenCL M/s | Ratio |
|-----------|-----------|----------|-------------|-----------|-------|
| Field Add | 0.2 | 4,130 | 13.1 | 76 | 54x |
| Field Mul | 0.2 | 4,134 | 12.2 | 82 | 50x |
| Field Sqr | -- | -- | 8.3 | 121 | -- |
| Field Inv | 12.1 | 82.7 | 44.8 | 22.3 | 3.7x |
| Point Double | 1.6 | 642 | 49.7 | 20 | 32x |
| Point Add | 2.1 | 477 | 70.8 | 14 | 34x |
| Scalar Mul (Gxk) | 591 | 1.69 | 419 | 2.39 | 0.7x OK |

### Scalar Multiplication Scaling

| Batch Size | CUDA ns/op | OpenCL ns/op |
|-----------|-----------|-------------|
| 256 | -- | 13,000 |
| 1,024 | -- | 3,300 |
| 4,096 | -- | 838 |
| 16,384 | -- | 425 |
| 65,536 | ~591 | 419 |
| 131,072 | 591 | -- |

---

## Key Observations

1. **OpenCL scalar_mul matches CUDA** -- at batch=65K, OpenCL achieves 2.39 M/s vs CUDA's 1.69 M/s. The wNAF implementation and efficient kernel dispatch make this competitive. Both use window-5 wNAF with 8-entry precomputation tables.

2. **CUDA dominates field arithmetic** -- 50-54x faster for field add/mul. CUDA's native PTX `mad.lo/hi.u64` instructions and compiler register allocation give sub-nanosecond amortized times that OpenCL cannot match through `mul_hi()`.

3. **Field inversion gap narrows to 3.7x** -- the addition chain optimization reduced OpenCL field_inv from ~246us (single-op with overhead) to 44.8 ns/op (batch), closing most of the gap with CUDA's 12.1 ns.

4. **Point operations ~30x gap** -- these compose multiple field operations, so the field arithmetic gap propagates. Each point_double uses ~10 field ops, each point_add ~16 field ops.

5. **Cross-platform advantage** -- OpenCL runs on Intel, AMD, and NVIDIA GPUs without code changes. CUDA is NVIDIA-only but provides the best possible performance on NVIDIA hardware for field-level operations.

## When to Use Which

| Use Case | Recommendation |
|----------|---------------|
| Maximum field throughput on NVIDIA | CUDA |
| Batch scalar multiplication | Either (comparable) |
| Cross-platform GPU support | OpenCL |
| Intel/AMD GPU | OpenCL (only option) |
| Portable research/verification | OpenCL |
| Production search workload | CUDA (field ops dominate) |
