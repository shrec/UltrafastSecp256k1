# Community & Contributor Benchmarks

This page collects benchmark results submitted by community members running
UltrafastSecp256k1 on their own hardware. Each entry includes the exact build
command, hardware details, date, and test count so results are reproducible.

> **Want to contribute?** Run the benchmark suite on your hardware and share
> your results via a [GitHub Discussion](https://github.com/shrec/UltrafastSecp256k1/discussions)
> or open an issue with the output of `bench_unified --suite all` (or
> `gpu_bench_unified` for GPU).  We'll add your entry here and credit you.
>
> **Tip:** Include your GPU/CPU model, OS, driver/CUDA/compiler version, the
> exact cmake command you used, and the full benchmark table.

---

## Index

| # | Contributor | Hardware | Date | Tests |
|---|-------------|----------|------|------:|
| 1 | Community (GigaChad) | NVIDIA RTX 5070 Ti (Blackwell) | 2026-03-24 | 45/45 |
| 2 | [@craigraw](https://github.com/craigraw) | x86-64 CPU (libsecp baseline) | 2026-02-xx | — |

---

## Entry #1 — NVIDIA RTX 5070 Ti

**Contributor:** Community member (Discord: GigaChad) — thank you for running
the full test suite and for identifying the `CMAKE_CUDA_SEPARABLE_COMPILATION`
flag required for Blackwell devices! 🙏

| Field | Value |
|-------|-------|
| **GPU** | NVIDIA GeForce RTX 5070 Ti |
| **Architecture** | Blackwell (SM 12.0) |
| **OS** | Linux |
| **CUDA** | (native) |
| **Tests passed** | **45 / 45** |
| **Date** | 2026-03-24 |

**Build command:**

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DSECP256K1_BUILD_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=native \
  -DCMAKE_CUDA_SEPARABLE_COMPILATION=ON
```

> **Note:** `CMAKE_CUDA_SEPARABLE_COMPILATION=ON` is required on RTX 50xx
> (Blackwell) devices.  This flag is now set automatically inside
> `cuda/CMakeLists.txt` and baked into all CUDA CMake presets — you only need
> to pass it explicitly if you invoke CMake without a preset.

### Core ECC Operations

| Operation | Time/Op | Throughput |
|-----------|--------:|-----------:|
| Field Mul | 5.8 ns | 173.43 M/s |
| Field Add | 2.5 ns | 408.04 M/s |
| Field Inverse | 5.2 ns | 191.55 M/s |
| Point Add | 9.9 ns | 100.89 M/s |
| Point Double | 5.5 ns | 181.70 M/s |
| Scalar Mul (Pk) | 101.4 ns | 9.86 M/s |
| Generator Mul (Gk) | 92.1 ns | 10.86 M/s |
| Affine Add (2M+1S+inv) | 0.1 ns | 8,388.29 M/s |
| Affine Lambda (2M+1S) | 0.2 ns | 4,117.82 M/s |
| Affine X-Only (1M+1S) | 0.1 ns | 8,354.07 M/s |
| Batch Inv (Montgomery) | 5.8 ns | 173.21 M/s |
| Jac→Affine (per-pt) | 14.4 ns | 69.34 M/s |

### Signature Operations

| Operation | Time/Op | Throughput |
|-----------|--------:|-----------:|
| ECDSA Sign | 105.3 ns | 9.49 M/s |
| ECDSA Verify | 122.8 ns | 8.14 M/s |
| ECDSA Sign+Recid | 155.8 ns | 6.42 M/s |
| Schnorr Sign | 137.7 ns | 7.26 M/s |
| Schnorr Verify | 92.7 ns | 10.79 M/s |

---

## Entry #2 — CPU libsecp256k1 Comparison (BIP-352 Standalone)

**Contributor:** [@craigraw](https://github.com/craigraw) ([Sparrow Wallet](https://sparrowwallet.com)) — thank you for creating the standalone
[bench_bip352](https://github.com/craigraw/bench_bip352) benchmark and for the
independent reproducible comparison! 🙏

| Field | Value |
|-------|-------|
| **CPU** | x86-64 |
| **Compiler** | GCC 12.4, `-O3 -march=native`, `USE_ASM_X86_64=1` |
| **Benchmark tool** | [bench_bip352](https://github.com/craigraw/bench_bip352) |
| **Mode** | Single-threaded, 10K points, 11 passes, median |

### Full Pipeline

| Backend | Median | ns/op | Ratio |
|---------|-------:|------:|------:|
| libsecp256k1 | 545.2 ms | 54,519 ns | 1.00× |
| **UltrafastSecp256k1** | **456.1 ms** | **45,615 ns** | **1.20× faster** |

### Per-Operation Breakdown (1K points, 11 passes, median)

| Operation | libsecp256k1 | UltrafastSecp256k1 | Ratio |
|-----------|------------:|-----------------:|------:|
| k×P (scalar mul) | 37,975 ns | 26,460 ns | 1.44× faster |
| Serialize compressed (1st) | 36 ns | 15 ns | 2.4× faster |
| Tagged SHA-256 | 744 ns | 65 ns | 11.4× faster |
| k×G (generator mul) | 17,460 ns | 8,559 ns | 2.04× faster |
| Point addition | 2,250 ns | 2,457 ns | 0.92× |
| Serialize compressed (2nd) | 23 ns | 21 ns | 1.1× faster |

> Point addition is slightly slower because both inputs have Z=1 (affine), so
> UltrafastSecp256k1 uses direct affine addition with a field inversion to
> return an affine result — this eliminates the separate inversion step inside
> serialization.

---

## How to Submit Your Benchmark

1. Build the project for your target (CPU or GPU).
2. Run `bench_unified --suite all --passes 11` (CPU) or `gpu_bench_unified`
   (CUDA/OpenCL/Metal).
3. Post results in
   [GitHub Discussions → Benchmarks](https://github.com/shrec/UltrafastSecp256k1/discussions/categories/benchmarks)
   with:
   - Hardware model (CPU/GPU/embedded)
   - OS, compiler, driver/CUDA version
   - Exact cmake command
   - Full benchmark output (or paste the table)
4. We'll add your entry here and list you in the contributor acknowledgments in
   `README.md`.

---

*Last updated: 2026-03-24*
