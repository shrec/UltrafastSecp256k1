# CUDA Benchmark -- NVIDIA RTX 5060 Ti

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
| Scalar Mul (Pxk) | 234.8 ns | 4.26 M/s |
| Generator Mul (Gxk) | 221.7 ns | 4.51 M/s |

## Optimizations Applied

1. **32-bit Hybrid Multiplication** (`SECP256K1_CUDA_USE_HYBRID_MUL=1`):
   - Comba-style 32-bit multiplication (64 MAD32 via PTX) instead of 64-bit
   - Consumer GPUs have INT32 throughput 32x higher than INT64
2. **32-bit Reduction** (`reduce_512_to_256_32`):
   - T_hi x 977 in 32-bit MAD chain (16 PTX ops) + T_hi << 32 shift
   - Avoids INT64 multiplies in the hot-path reduction
3. **Single-pass K_MOD reduction** (64-bit path):
   - T_hi x K_MOD in one MAD chain instead of T_hix977 + T_hi<<32 (two passes)

## Improvement vs Previous

| Operation | Before | After | Speedup |
|-----------|--------|-------|---------|
| Point Add | 2.1 ns (476 M/s) | 0.9 ns (1,066 M/s) | **2.24x** |
| Point Double | 1.6 ns (642 M/s) | 0.7 ns (1,356 M/s) | **2.11x** |
| Scalar Mul | 624.9 ns (1.60 M/s) | 234.8 ns (4.26 M/s) | **2.66x** |
| Generator Mul | 591.5 ns (1.69 M/s) | 221.7 ns (4.51 M/s) | **2.67x** |
| Field Inverse | 12.1 ns (82.66 M/s) | 10.2 ns (97.57 M/s) | **1.18x** |

## Notes

- Batch size: 1,048,576 (1M) for field ops, 65K-131K for point/scalar ops
- Amortized per-element time (includes kernel launch cost spread over batch)
- Results consistent across 5 measurement iterations with 3 warmup passes
- Field Mul/Add unchanged at 0.2 ns (memory bandwidth limited at this batch size)
- GPU search app: 1,131 -> 1,223 M/s (+8.1%) end-to-end throughput

## BIP-352 Silent Payment Scan (`ufsecp_gpu_bip352_scan_batch`)

**Date:** 2026-04-04  
**N:** 500,000 tweak points, 11 passes median, 3 warmup  
**Source:** `bench_bip352` — internal kernel; C ABI dispatch overhead <2 ns

| Mode | Time/Op | Throughput | vs CPU |
|------|---------|------------|--------|
| CPU (UltrafastSecp256k1 KPlan) | 24,436.5 ns | 40.9 K/s | 1.00x |
| GPU (CUDA GLV, tpb=384) | 178.9 ns | 5.59 M/s | **136.6x** |
| GPU + LUT (16×64K table) | 90.5 ns | 11.05 M/s | **270.0x** |

### Per-Operation Breakdown (1000 ops, median)

| Step | CPU (ns) | GPU (ns) | GPU Speedup |
|------|----------|----------|-------------|
| k×P (scalar_mul) | 18,800.9 | 608.9 | 30.9x |
| to_compressed (1st) | 7.2 | 109.1 | — |
| tagged SHA-256 (cached) | 50.3 | 13.4 | 3.75x |
| k×G (w=4 GLV) | 5,991.3 | 975.4 | 6.1x |
| k×G (LUT, 1M-pt global) | 5,991.3 | 75.5 | **79.4x** |
| point_add | 1,324.4 | 9.1 | **146.2x** |
| to_compressed (2nd) | 7.0 | 108.8 | — |

### Time Breakdown (% of full pipeline)

| Step | CPU % | GPU % |
|------|-------|-------|
| k×P | 71.8% | 33.4% |
| Serialize ×2 | ~0% | 12.0% |
| SHA-256 | 0.2% | 0.7% |
| k×G | 22.9% | 53.5% |
| Point add | 5.1% | 0.5% |

Validation: `[OK] ALL MATCH` (CPU = GPU GLV = GPU+LUT prefix check)
