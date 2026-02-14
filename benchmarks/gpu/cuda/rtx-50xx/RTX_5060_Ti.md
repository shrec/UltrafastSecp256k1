# CUDA Benchmark — NVIDIA RTX 5060 Ti

**Date:** 2026-02-14 (updated after 32-bit hybrid optimization)  
**OS:** Linux x86_64 (Ubuntu)  
**Driver:** NVIDIA 580.126.09  
**CUDA:** 12.0, sm_89  
**Build:** GCC 14.2.0, Release, -O3 --use_fast_math  

## GPU Info

| Property | Value |
|----------|-------|
| Device | NVIDIA GeForce RTX 5060 Ti |
| Compute Capability | 12.0 |
| SM Count | 36 |
| Clock | 2602 MHz |
| Memory | 15847 MB |
| Memory Clock | 14001 MHz |
| Memory Bus | 128-bit |

## Results

| Operation | Time/Op | Throughput |
|-----------|---------|------------|
| Field Mul | 0.2 ns | 4,133.93 M/s |
| Field Add | 0.2 ns | 4,141.98 M/s |
| Field Inverse | 10.2 ns | 97.57 M/s |
| Point Add | 0.9 ns | 1,065.72 M/s |
| Point Double | 0.7 ns | 1,356.07 M/s |
| Scalar Mul (P×k) | 234.8 ns | 4.26 M/s |
| Generator Mul (G×k) | 221.7 ns | 4.51 M/s |

## Optimizations Applied

1. **32-bit Hybrid Multiplication** (`SECP256K1_CUDA_USE_HYBRID_MUL=1`):
   - Comba-style 32-bit multiplication (64 MAD32 via PTX) instead of 64-bit
   - Consumer GPUs have INT32 throughput 32× higher than INT64
2. **32-bit Reduction** (`reduce_512_to_256_32`):
   - T_hi × 977 in 32-bit MAD chain (16 PTX ops) + T_hi << 32 shift
   - Avoids INT64 multiplies in the hot-path reduction
3. **Single-pass K_MOD reduction** (64-bit path):
   - T_hi × K_MOD in one MAD chain instead of T_hi×977 + T_hi<<32 (two passes)

## Improvement vs Previous

| Operation | Before | After | Speedup |
|-----------|--------|-------|---------|
| Point Add | 2.1 ns (476 M/s) | 0.9 ns (1,066 M/s) | **2.24×** |
| Point Double | 1.6 ns (642 M/s) | 0.7 ns (1,356 M/s) | **2.11×** |
| Scalar Mul | 624.9 ns (1.60 M/s) | 234.8 ns (4.26 M/s) | **2.66×** |
| Generator Mul | 591.5 ns (1.69 M/s) | 221.7 ns (4.51 M/s) | **2.67×** |
| Field Inverse | 12.1 ns (82.66 M/s) | 10.2 ns (97.57 M/s) | **1.18×** |

## Notes

- Batch size: 1,048,576 (1M) for field ops, 65K-131K for point/scalar ops
- Amortized per-element time (includes kernel launch cost spread over batch)
- Results consistent across 5 measurement iterations with 3 warmup passes
- Field Mul/Add unchanged at 0.2 ns (memory bandwidth limited at this batch size)
- GPU search app: 1,131 → 1,223 M/s (+8.1%) end-to-end throughput
