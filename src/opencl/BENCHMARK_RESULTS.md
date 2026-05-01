# UltrafastSecp256k1 OpenCL Benchmark Results

**Date:** 2026-02-14  
**OS:** Linux (x86_64)  
**Build:** Clang 19.1.7, Release, C++20, -O3  
**OpenCL:** 3.0 CUDA, Driver 580.126.09  
**PTX:** Inline assembly enabled (`__NV_CL_C_VERSION`)

---

## Device

| Property        | NVIDIA GeForce RTX 5060 Ti      |
|-----------------|--------------------------------|
| Vendor          | NVIDIA Corporation             |
| OpenCL Version  | OpenCL 3.0 CUDA                |
| Driver          | 580.126.09                     |
| Global Memory   | 15,847 MB                      |
| Local Memory    | 48 KB                          |
| Compute Units   | 36                             |
| Max Clock       | 2,602 MHz                      |
| Memory Bus      | 128-bit                        |

---

## Kernel-Only Timing (no buffer alloc/copy overhead)

| Operation | Time/Op | Throughput | Batch |
|-----------|---------|------------|-------|
| Field Add | 0.2 ns | 4,124 M/s | 1M |
| Field Sub | 0.2 ns | 4,119 M/s | 1M |
| Field Mul | 0.2 ns | 4,137 M/s | 1M |
| Field Sqr | 0.2 ns | 5,985 M/s | 1M |
| Field Inv | 14.3 ns | 69.97 M/s | 1M |
| Point Double | 0.9 ns | 1,139 M/s | 256K |
| Point Add | 1.6 ns | 630.6 M/s | 256K |
| kG (kernel) | 295.1 ns | 3.39 M/s | 256K |

---

## End-to-End Timing (including buffer transfers)

### Field Arithmetic (batch=1,048,576)

| Operation | Time/Op | Throughput |
|-----------|---------|------------|
| Field Add | 27.3 ns | 36.67 M/s |
| Field Sub | 28.5 ns | 35.06 M/s |
| Field Mul | 27.7 ns | 36.07 M/s |
| Field Sqr | 15.7 ns | 63.62 M/s |
| Field Inv | 29.0 ns | 34.43 M/s |

### Point Operations (batch=1,048,576)

| Operation | Time/Op | Throughput |
|-----------|---------|------------|
| Point Double | 58.4 ns | 17.11 M/s |
| Point Add | 111.9 ns | 8.94 M/s |

### Batch Scalar Multiplication (kG)

| Batch Size | Time/Op | Throughput |
|------------|---------|------------|
| 256 | 9.5 us | 105 K/s |
| 1,024 | 2.4 us | 422 K/s |
| 4,096 | 610.6 ns | 1.64 M/s |
| 16,384 | 311.6 ns | 3.21 M/s |
| 65,536 | 307.7 ns | 3.25 M/s |

### Batch Field Inversion

| Batch Size | Time/Op | Throughput |
|------------|---------|------------|
| 256 | 737.9 ns | 1.36 M/s |
| 1,024 | 191.8 ns | 5.21 M/s |
| 4,096 | 53.0 ns | 18.87 M/s |
| 16,384 | 28.1 ns | 35.62 M/s |

---

## CUDA vs OpenCL Comparison (Kernel-Only, RTX 5060 Ti)

| Operation | CUDA | OpenCL | Winner |
|-----------|------|--------|--------|
| Field Mul | 0.2 ns | 0.2 ns | Tie |
| Field Inv | 10.2 ns | 14.3 ns | **CUDA 1.40x** |
| Point Double | 0.7 ns | 0.9 ns | **CUDA 1.29x** |
| Point Add | 0.9 ns | 1.6 ns | **CUDA 1.78x** |
| kG | 221.7 ns | 295.1 ns | **CUDA 1.33x** |

**CUDA wins** across all operations after the 32-bit hybrid Comba optimization
(PTX `mad.lo.cc.u32` / `madc.hi.u32` with hardware carry flags).
OpenCL uses portable `mul_hi(ulong)` which NVIDIA's compiler already decomposes
into optimal 32-bit PTX -- manual 32-bit Comba adds no benefit on OpenCL.
