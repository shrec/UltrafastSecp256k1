# UltrafastSecp256k1 OpenCL Benchmark Results

**Date:** 2026-02-14  
**OS:** Windows 10 (x86_64)  
**Build:** MSVC 19.44, Release, C++20  
**OpenCL SDK:** NVIDIA CUDA Toolkit 12.8  

---

## Devices

| Property        | Intel UHD Graphics 750         | NVIDIA GeForce GT 730         |
|-----------------|-------------------------------|-------------------------------|
| Vendor          | Intel(R) Corporation          | NVIDIA Corporation            |
| OpenCL Version  | OpenCL 3.0 NEO                | OpenCL 1.2 CUDA               |
| Driver          | 32.0.101.7084                 | 456.71                        |
| Global Memory   | 37,937 MB                     | 2,048 MB                      |
| Local Memory    | 64 KB                         | 48 KB                         |
| Compute Units   | 32                            | 2                             |
| Max Clock       | 1,300 MHz                     | 901 MHz                       |

---

## Field Arithmetic (ns/op, lower is better)

| Operation   | Intel UHD 750 | NVIDIA GT 730 | Intel Speedup |
|-------------|--------------|---------------|---------------|
| Field Add   | 189,023      | 829,316       | 4.39x         |
| Field Sub   | 218,417      | 721,139       | 3.30x         |
| Field Mul   | 195,478      | 724,748       | 3.71x         |
| Field Sqr   | 179,685      | 636,987       | 3.54x         |
| Field Inv   | 2,554,314    | 2,925,589     | 1.15x         |

---

## Point Operations (lower is better)

| Operation      | Intel UHD 750   | NVIDIA GT 730   | Intel Speedup |
|----------------|----------------|----------------|---------------|
| Point Double   | 221,327 ns     | 715,677 ns     | 3.23x         |
| Point Add      | 305,155 ns     | 805,494 ns     | 2.64x         |
| Scalar Mul     | 1,327.0 µs     | 1,744.2 µs     | 1.31x         |

---

## Batch Scalar Multiplication (higher ops/s is better)

| Batch Size | Intel UHD 750          | NVIDIA GT 730          | Intel Speedup |
|------------|------------------------|------------------------|---------------|
| 256        | 1.21 ms (211,064 op/s) | 1.78 ms (143,942 op/s) | 1.47x         |
| 1,024      | 1.70 ms (603,596 op/s) | 2.38 ms (430,252 op/s) | 1.40x         |
| 4,096      | 4.74 ms (863,516 op/s) | 6.05 ms (676,790 op/s) | 1.28x         |
| 16,384     | 17.08 ms (959,509 op/s)| 24.27 ms (675,142 op/s)| 1.42x         |

---

## Batch Field Inversion (higher ops/s is better)

| Batch Size | Intel UHD 750          | NVIDIA GT 730          | Intel Speedup |
|------------|------------------------|------------------------|---------------|
| 256        | 2.80 ms (91,396 op/s)  | 2.90 ms (88,389 op/s)  | 1.03x         |
| 1,024      | 3.89 ms (263,043 op/s) | 3.10 ms (330,717 op/s) | 0.80x (NVIDIA wins) |
| 4,096      | 8.35 ms (490,492 op/s) | 7.49 ms (546,592 op/s) | 0.90x (NVIDIA wins) |

---

## Summary

- **Intel UHD 750** dominates in nearly all categories: **3-4x faster** on single field ops, **1.3-1.5x faster** on batch scalar multiplication
- **Peak throughput:** Intel reaches **~960K scalar mul ops/s** at batch 16K vs NVIDIA's **~675K ops/s**
- **NVIDIA GT 730 wins** slightly on batch field inversion at larger sizes (1K-4K), possibly due to the Montgomery inversion algorithm favoring its memory hierarchy
- Field inversion is the only area competitive between the two GPUs (1.15x difference on single ops)
- The Intel advantage is primarily due to **16x more compute units** (32 vs 2) and higher clock (1300 vs 901 MHz)

---

## Command Reference

```bash
# NVIDIA benchmark
opencl_benchmark.exe --nvidia

# Intel benchmark  
opencl_benchmark.exe --intel

# Explicit platform/device selection
opencl_benchmark.exe --platform 0 --device 0   # NVIDIA (platform 0)
opencl_benchmark.exe --platform 1 --device 0   # Intel GPU (platform 1)
```
