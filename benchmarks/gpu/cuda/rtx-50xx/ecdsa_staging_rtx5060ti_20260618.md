# CUDA ECDSA Staging Benchmark — RTX 5060 Ti — 2026-06-18

Diagnostic benchmark for the CUDA ECDSA verify staging change that uploads
compact 64-byte `r||s` signatures and parses them in the verify kernel instead
of staging `ECDSASignatureGPU` structs on the host.

This artifact is diagnostic, not a release-wide performance claim. It separates
kernel-only timing from libbitcoin bridge end-to-end timing because the change
is expected to trade a tiny amount of GPU-side parse work for less host-side
conversion/allocation/copy work.

## Environment

| Field | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 5060 Ti |
| Compute | 12.0 |
| SM count | 36 |
| GPU clock | 2602 MHz |
| Memory | 15844 MB |
| Driver | 580.159.04 |
| CUDA compiler | nvcc 12.0.140 |
| Host compiler | GCC 14.2.0 |
| CUDA arch flag | `CMAKE_CUDA_ARCHITECTURES=89` |

## Compared Trees

| Label | Source |
|---|---|
| Baseline | `3a3d33f5` (`docs: document engine and c abi package split`) |
| Optimized | dev tree with CUDA compact-signature verify staging and benchmark declaration fix from this commit |

## Kernel-Only CUDA Verify

Command:

```bash
cmake --preset cuda-release-5060ti
cmake --build out/cuda-release-5060ti --target secp256k1_cuda_bench -j2
out/cuda-release-5060ti/src/cuda/secp256k1_cuda_bench --batch 1048576
```

| Metric | Baseline | Optimized | Delta |
|---|---:|---:|---:|
| ECDSA verify ns/op | 248.0 | 249.3 | +0.5% slower |
| ECDSA verify throughput | 4.03 M/s | 4.01 M/s | -0.5% |

Interpretation: kernel-only timing is intentionally not where this change wins.
The optimized kernel parses compact signatures on device, so a tiny kernel-only
cost is expected.

## Libbitcoin Bridge End-To-End GPU Path

Command:

```bash
cmake -S . -B out/cuda-lbtc-bench -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DCMAKE_CUDA_SEPARABLE_COMPILATION=ON \
  -DSECP256K1_BUILD_CPU=ON \
  -DSECP256K1_BUILD_CUDA=ON \
  -DSECP256K1_ENABLE_OPENMP=ON \
  -DSECP256K1_BUILD_LIBBITCOIN=ON \
  -DSECP256K1_BUILD_LIBBITCOIN_BENCH=ON
cmake --build out/cuda-lbtc-bench --target bench_lbtc_batch -j2
out/cuda-lbtc-bench/include/ufsecp/bench_lbtc_batch 100000 3 5000
```

The GPU correctness gate passed before timing on both trees:
all-valid ECDSA/Schnorr, single-corruption detection, row layout, column layout,
and collect layout. The CPU leg was interrupted after the GPU section because
this artifact is scoped to the GPU staging change.

| GPU path | Baseline | Optimized | Delta |
|---|---:|---:|---:|
| ECDSA row | 3.90 M sig/s | 3.97 M sig/s | +1.8% |
| ECDSA columns | 3.66 M sig/s | 3.83 M sig/s | +4.6% |
| ECDSA collect | 3.79 M sig/s | 3.86 M sig/s | +1.8% |
| ECDSA column collect | 3.64 M sig/s | 3.78 M sig/s | +3.8% |

Interpretation: the end-to-end bridge path improves despite the kernel-only
parse cost, because host-side signature staging overhead is reduced.

## Raw Entry Smoke

Command:

```bash
cmake --build out/cuda-release-5060ti --target bench_raw_entry -j2
out/cuda-release-5060ti/src/cuda/bench_raw_entry
```

| Path | Time | Throughput | Valid |
|---|---:|---:|---:|
| Pre-expanded pubkey + compact sig | 248.6 ns/op | 4.02 M/s | 16384/16384 |
| Raw compressed pubkey + opaque LE sig | 249.8 ns/op | 4.00 M/s | 16384/16384 |

Raw-entry overhead was 1.2 ns/op (+0.5%). This supports moving more
public-data prep to the GPU when it removes host memory movement or staging.

## OpenCL Check

OpenCL was configured and built successfully on the same machine:

```bash
cmake -S . -B out/opencl-bench -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DSECP256K1_BUILD_CPU=ON \
  -DSECP256K1_BUILD_OPENCL=ON \
  -DSECP256K1_BUILD_CUDA=OFF \
  -DSECP256K1_BUILD_BENCH=ON \
  -DSECP256K1_ENABLE_OPENMP=ON
cmake --build out/opencl-bench --target opencl_benchmark -j2
out/opencl-bench/src/opencl/opencl_benchmark --nvidia --batch 65536
```

Runtime used `NVIDIA GeForce RTX 5060 Ti`, OpenCL 3.0 CUDA, driver 580.159.04.
The current OpenCL benchmark skipped signature sections because it could not
find `secp256k1_extended.cl`, so no OpenCL ECDSA staging comparison is recorded
in this artifact.
