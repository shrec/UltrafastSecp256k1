# OpenCL Benchmark — NVIDIA RTX 5060 Ti

**Date:** 2026-02-14 (updated: optimized kernels)  
**OS:** Linux x86_64 (Ubuntu)  
**Driver:** NVIDIA 580.126.09  
**OpenCL:** 3.0 CUDA  
**Build:** GCC 14.2.0, Release, C++20  

## GPU Info

| Property | Value |
|----------|-------|
| Device | NVIDIA GeForce RTX 5060 Ti |
| Vendor | NVIDIA Corporation |
| OpenCL Version | OpenCL 3.0 CUDA |
| Driver | 580.126.09 |
| Global Memory | 15,847 MB |
| Local Memory | 48 KB |
| Compute Units | 36 |
| Max Clock | 2,602 MHz |

## Optimizations Applied

1. **field_mul**: Fully unrolled 4×4 schoolbook (no loops, 16 explicit mul64_full)
2. **field_sqr**: Fully unrolled off-diagonal + diagonal computation
3. **field_inv**: Fermat addition chain (~260 ops instead of ~448 naive)
4. **scalar_mul**: wNAF window-5 with 8-entry precomputed table
5. **Benchmark**: Batch throughput measurement (amortized per-element, batch=65536)

## Field Arithmetic (batch=65,536)

| Operation | Time/Op | Throughput |
|-----------|---------|------------|
| Field Add | 13.1 ns | 76.08 M/s |
| Field Sub | 12.4 ns | 80.71 M/s |
| Field Mul | 12.2 ns | 81.70 M/s |
| Field Sqr | 8.3 ns | 121.09 M/s |
| Field Inv | 44.8 ns | 22.33 M/s |

## Point Operations (batch=65,536)

| Operation | Time/Op | Throughput |
|-----------|---------|------------|
| Point Double | 49.7 ns | 20.12 M/s |
| Point Add | 70.8 ns | 14.13 M/s |

## Scalar Multiplication (G×k) Scaling

| Batch Size | Time/Op | Throughput |
|------------|---------|------------|
| 256 | 13.0 μs | 77 K/s |
| 1,024 | 3.3 μs | 306 K/s |
| 4,096 | 838 ns | 1.19 M/s |
| 16,384 | 425 ns | 2.35 M/s |
| 65,536 | 419 ns | 2.39 M/s |

## Batch Field Inversion Scaling

| Batch Size | Time/Op | Throughput |
|------------|---------|------------|
| 256 | 1.5 μs | 651 K/s |
| 1,024 | 370 ns | 2.70 M/s |
| 4,096 | 97.9 ns | 10.21 M/s |
| 16,384 | 49.9 ns | 20.04 M/s |

## Notes

- All times are amortized per-element from batch dispatch (same methodology as CUDA benchmark)
- Scalar multiplication at batch=65K achieves 2.39 M/s (CUDA now achieves 4.51 M/s after 32-bit hybrid optimization)
- Field arithmetic ~50× slower than CUDA due to OpenCL buffer transfer overhead vs in-register CUDA kernel
- 32/32 correctness tests pass
