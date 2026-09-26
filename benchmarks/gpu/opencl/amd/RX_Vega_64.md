# OpenCL Benchmark -- AMD Radeon RX Vega 64

**Date:** 2026-09-26
**OS:** Linux x86_64 (Ubuntu 26.04 LTS, kernel 7.0)
**Driver:** AMD ROCm 7.1 OpenCL (AMD-APP 3581.0, HSA1.1, LC)
**Build:** GCC 15.2.0, Release, `dev` 77ff44a
**Host:** AMD Ryzen 9 3900X (12C/24T), 30 GB. Another long-running process used about 1-2 threads during the runs (load average 0.6-2.7).

## GPU Info

| Property | Value |
|----------|-------|
| Device | AMD Radeon RX Vega 64 (`gfx900:xnack-`) |
| Compute Units | 64 |
| Clock | 1630 MHz |
| Memory | 8 GB (7.98 GiB global) |
| OpenCL | Device OpenCL 2.0, platform OpenCL 2.1 AMD-APP (3581.0) |

## Results (`opencl_benchmark`, batch = 1,048,576, including transfers)

| Operation | Time/Op | Throughput |
|-----------|---------|------------|
| Field Add | 13.3 ns | 75.26 M/s |
| Field Sub | 13.0 ns | 76.86 M/s |
| Field Mul | 13.0 ns | 76.82 M/s |
| Field Sqr | 8.8 ns | 113.91 M/s |
| Field Inverse | 417.2 ns | 2.40 M/s |
| Point Double | 35.9 ns | 27.85 M/s |
| Point Add | 72.7 ns | 13.76 M/s |

### Kernel-only timing (no buffer alloc/copy)

| Operation | Time/Op | Throughput |
|-----------|---------|------------|
| Field Add | 1.7 ns | 590.31 M/s |
| Field Sub | 1.4 ns | 714.98 M/s |
| Field Mul | 1.4 ns | 717.89 M/s |
| Field Sqr | 0.9 ns | 1,115.08 M/s |
| Field Inverse | 409.5 ns | 2.44 M/s |
| Affine Add (2M+1S+inv) | 417.0 ns | 2.40 M/s |
| Affine Lambda (2M+1S) | 3.4 ns | 296.72 M/s |
| Affine X-Only (1M+1S) | 2.7 ns | 364.26 M/s |
| Jac->Affine (per point) | 413.9 ns | 2.42 M/s |
| Point Double | 4.3 ns | 232.10 M/s |
| Point Add | 26.7 ns | 37.51 M/s |
| kG (kernel) | 1.8 us | 553 K/s |
| kP (kernel) | 1.7 us | 593 K/s |
| kP upload / readback | 12.1 / 15.3 ns | |

### Scalar multiplication scaling

| Batch Size | kG | kP |
|-----------|----|----|
| 256 | 50.7 us | 110.2 us |
| 1,024 | 12.7 us | 27.6 us |
| 4,096 | 3.2 us | 6.9 us |
| 16,384 | 1.8 us | 1.8 us |
| 65,536 | 1.8 us (547 K/s) | 1.7 us (580 K/s) |

### Batch field inversion

| Batch Size | Time/Op |
|-----------|---------|
| 256 | 28.1 us |
| 1,024 | 7.0 us |
| 4,096 | 1.8 us |
| 16,384 | 449.9 ns (2.22 M/s) |

## Comparison with RTX 5060 Ti OpenCL (batch = 65,536)

RTX 5060 Ti figures are from `benchmarks/comparison/cuda_vs_opencl_rtx5060ti.md`. The Vega figures are from `opencl_benchmark --batch 65536`. The RTX figures were measured on 2026-02-14 with an earlier library version, and the Vega figures on `dev` 77ff44a, so the comparison reflects both hardware and code changes.

| Operation | RX Vega 64 | RTX 5060 Ti (OpenCL) |
|-----------|-----------|----------------------|
| Field Add | 20.4 ns | 13.1 ns |
| Field Mul | 18.8 ns | 12.2 ns |
| Field Sqr | 13.1 ns | 8.3 ns |
| Field Inverse | 424.0 ns | 44.8 ns |
| Point Double | 42.8 ns | 49.7 ns |
| Point Add | 82.4 ns | 70.8 ns |
| Scalar Mul (Gxk) | 1.8 us (0.55 M/s) | 419 ns (2.39 M/s) |

## libbitcoin direct batch verify (`bench_lbtc_direct_batch 262144 3`)

Correctness: PASS (all-valid + corruption detected). The column path goes through the GPU hook when a device is available, so its throughput doesn't change with the thread count. Row and 24-thread figures are CPU (Ryzen 9 3900X).

| Path | Threads | ECDSA | Schnorr |
|------|---------|-------|---------|
| Row (CPU) | 1 | 0.02 M sig/s | 0.03 M sig/s |
| Row (CPU) | 24 | 0.31 M sig/s | 0.42 M sig/s |
| Columns (GPU) | 1 or 24 | 0.13 M sig/s | 0.14 M sig/s |

Without a usable GPU the columns path falls back to the CPU and matches the row path (0.30 / 0.40 M sig/s at 24 threads, measured with no OpenCL platform). The GPU hook needs `UFSECP_OPENCL_KERNEL_DIR` (or an installed kernel directory) to find `secp256k1_extended.cl`; otherwise the column path silently runs on the CPU.

On this host the Vega is 4.7-6.5x faster than one CPU thread and 2.4-3.0x slower than all 24.

## BIP-352 pipeline (`opencl_bip352_benchmark`, N = 500,000)

| | Time/Op | Throughput |
|-|---------|------------|
| OpenCL fused pipeline (local = 256, autotuned) | 2,651.0 ns | 0.38 M/s |
| CUDA reference printed by the tool (RTX 5060 Ti, GLV) | 179.1 ns | 5.58 M/s |

Validation: `[OK] MATCH`. On non-NVIDIA devices the benchmark needs `-cl-nv-opt-level=3` limited to NVIDIA, otherwise the build is rejected before compiling.

## First-call kernel compile time (ROCm, no compiler cache)

| Program | Compile time |
|---------|--------------|
| Embedded program (`Context::create`) | 29.0 s |
| `secp256k1_extended.cl` | 140.9 s |
| `secp256k1_bip352.cl` | 187.9 s |
| `secp256k1_zk.cl` | 473.0 s |

With ROCm's own compiler cache (default), repeat runs take 2.8 s for the embedded program and 25.5 s for `secp256k1_extended.cl`.

## Raw output

- `opencl_benchmark_rx_vega64_20260926.txt`
- `opencl_benchmark_batch65536_rx_vega64_20260926.txt`
- `bip352_rx_vega64_20260926.txt`
- `lbtc_batch_rx_vega64_20260926.json`
