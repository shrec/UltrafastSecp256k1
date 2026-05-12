# Backend Parity Evidence Matrix

**Last updated**: 2026-04-21 | **Version**: 4.0.0

This document tracks feature parity, audit coverage, and benchmark evidence across all GPU backends and the CPU reference implementation.

---

## Backend Summary

| Property | CPU | CUDA | OpenCL | Metal |
|----------|-----|------|--------|-------|
| **CMake option** | `SECP256K1_BUILD_CPU` (ON) | `SECP256K1_BUILD_CUDA` (OFF) | `SECP256K1_BUILD_OPENCL` (OFF) | `SECP256K1_BUILD_METAL` (OFF) |
| **Language** | C++20 | CUDA C++ | C99 (.cl kernels) | Metal Shading Language + ObjC++ |
| **Interface** | `fastsecp256k1` static lib | `secp256k1_cuda_lib` | `secp256k1_opencl` | `secp256k1_metal_lib` |
| **Audit runner** | `unified_audit_runner` (60 non-exploit + 257 exploit PoCs = 249 modules) | `gpu_audit_runner` (47 modules) | `opencl_audit_runner` (40 modules) | `metal_audit_runner` (30 modules) |
| **Benchmark** | `bench_unified` (8 categories) | `gpu_bench_unified` + 4 specialized | `opencl_benchmark` | `metal_secp256k1_bench_full` |
| **CTest targets** | 18 core + 27 audit | 3 (`cuda_selftest`, `gpu_audit`, `gpu_ct_smoke`) | 2 (`opencl_selftest`, `opencl_audit`) | 5 (`metal_host_test`, `secp256k1_metal_test`, `secp256k1_metal_bench`, `secp256k1_metal_bench_full`, `secp256k1_metal_audit`) |
| **Tested hardware** | x86-64, ARM64, RISC-V, ESP32-S3/P4/C6 | RTX 5060 Ti, RTX 4090, RTX 3090 | RTX 5060 Ti (via NVIDIA OpenCL) | Apple M3 Pro, M1 (CI) |

---

## Operation Parity Matrix

Y = implemented + tested in audit runner, K = kernel/shader exists (not in audit), - = not present

| Operation | CPU | CUDA | OpenCL | Metal | Notes |
|-----------|-----|------|--------|-------|-------|
| **Core Math** | | | | | |
| Field arithmetic (mul, sqr, inv, add, sub) | Y | Y | Y | Y | |
| Scalar arithmetic (mul, inv, add, negate) | Y | Y | Y | Y | |
| Point arithmetic (add, dbl, mixed add) | Y | Y | Y | Y | |
| Scalar multiplication (k*G, k*P) | Y | Y | Y | Y | |
| GLV endomorphism | Y | Y | Y | Y | |
| Generator table (w8 precomp) | Y | Y | Y | Y | Kernel: `gen_table_w8` |
| MSM / Pippenger | Y | Y | Y | Y | |
| **Batch Operations** | | | | | |
| Batch Montgomery inversion | Y | Y | Y | Y | |
| Batch Jacobian-to-affine | Y | Y | Y | Y | |
| Affine batch add | Y | Y | Y | Y | |
| **Signatures** | | | | | |
| ECDSA sign (RFC 6979) | Y | Y | Y | Y | GPU: device-side RFC 6979 nonce |
| ECDSA verify | Y | Y | Y | Y | |
| ECDSA recovery | Y | Y | Y | Y | |
| Schnorr sign (BIP-340) | Y | Y | Y | Y | |
| Schnorr verify (BIP-340) | Y | Y | Y | Y | |
| BIP-340 midstate optimization | Y | Y | Y | Y | Precomputed SHA-256 midstates |
| Batch ECDSA verify | Y | Y | Y | Y | Exposed through `ufsecp_gpu_ecdsa_verify_batch` |
| Batch Schnorr verify | Y | Y | Y | Y | Exposed through `ufsecp_gpu_schnorr_verify_batch` |
| Batch ECDSA recovery (`ecrecover_batch`) | Y (host loop) | Y | Y | Y | Stable GPU ABI parity closed across all 3 GPU backends |
| **Hashing** | | | | | |
| SHA-256 | Y | Y | Y | Y | |
| Hash160 (RIPEMD160(SHA256)) | Y | Y | Y | Y | |
| Keccak-256 (Ethereum) | Y | Y | Y | Y | |
| ETH address + EIP-55 | Y | Y | Y | Y | |
| **HD / Key Derivation** | | | | | |
| BIP-32 HD derivation | Y | Y | Y | Y | |
| ECDH (x-only + raw) | Y | Y | Y | Y | |
| **Advanced Crypto** | | | | | |
| Pedersen commitment | Y | Y | Y | Y | |
| ZK proofs (knowledge, DLEQ) | Y | Y | Y | Y | |
| Bulletproof range proof verify | Y | Y | Y | Y | |
| Bulletproof generator table | Y | Y | Y | Y | |
| **GPU Batch Ops (C ABI)** | | | | | |
| ZK knowledge verify batch | Y (host) | Y | - | - | `ufsecp_gpu_zk_knowledge_verify_batch` |
| ZK DLEQ verify batch | Y (host) | Y | - | - | `ufsecp_gpu_zk_dleq_verify_batch` |
| Bulletproof verify batch | Y (host) | Y | - | - | `ufsecp_gpu_bulletproof_verify_batch` |
| BIP-324 AEAD encrypt batch | Y (host) | Y | - | - | `ufsecp_gpu_bip324_aead_encrypt_batch` |
| BIP-324 AEAD decrypt batch | Y (host) | Y | - | - | `ufsecp_gpu_bip324_aead_decrypt_batch` |
| ECDSA SNARK witness batch | Y (host) | Y | Y | - | `ufsecp_gpu_zk_ecdsa_snark_witness_batch` (eprint 2025/695) |
| BIP-352 Silent Payment scan | Y (host) | Y | Y | stub | `ufsecp_gpu_bip352_scan_batch`; Metal: `TODO(parity):bip352-scan-metal` |
| **Constant-Time Layer** | | | | | |
| CT field ops | Y | Y | K | K | OCL/Metal: kernel files exist, not in audit |
| CT scalar ops | Y | Y | K | K | Same |
| CT point ops | Y | Y | K | K | Same |
| CT sign (ECDSA + Schnorr) | Y | Y | K | K | Same |
| CT ZK proofs | Y | Y | K | K | Same |
| **Utility** | | | | | |
| Bloom filter lookup | Y | Y | Y | Y | |

---

## Audit Module Comparison

### Sections present per runner

| Section | CPU (unified) | CUDA | OpenCL | Metal |
|---------|--------------|------|--------|-------|
| math_invariants | Y (12) | Y (12) | Y (12) | Y (12) |
| signatures | Y | Y (3) | Y (3) | Y (3) |
| batch_advanced | Y | Y (4) | Y (2) | Y (2) |
| differential | Y | Y (1-3) | Y (1) | Y (1) |
| memory_safety | - | Y (2) | - | - |
| ct_analysis | Y | Y (6) | - | - |
| standard_vectors | Y | Y (3) | Y (2) | Y (2) |
| protocol_security | Y | Y (6) | Y (2) | Y (2) |
| fuzzing | Y | Y (4) | Y (3) | Y (3) |
| performance | Y | Y (2) | Y (2) | Y (2) |

### CUDA-only modules (not in OpenCL/Metal)

| Module | Section | Why CUDA-only |
|--------|---------|---------------|
| `batch_ecdsa_ver` (16 sigs) | batch_advanced | Batch verify kernel not in OCL |
| `msm_consistency` | batch_advanced | MSM naive vs expected |
| `diff_field_mul` | differential | Requires `HAVE_CPU_LIB` link |
| `diff_ecdsa` | differential | Requires `HAVE_CPU_LIB` + 64-bit limbs |
| `mem_stress` | memory_safety | CUDA device memory specific |
| `error_state` | memory_safety | CUDA error state specific |
| `ct_field_ops` | ct_analysis | CT test infrastructure |
| `ct_scalar_ops` | ct_analysis | CT test infrastructure |
| `ct_point_ops` | ct_analysis | CT test infrastructure |
| `ct_ecdsa_rt` | ct_analysis | CT ECDSA sign + fast verify |
| `ct_schnorr_rt` | ct_analysis | CT Schnorr sign + fast verify |
| `ct_fast_parity` | ct_analysis | CT vs FAST bit-exact parity |
| `bip32_derivation` | standard_vectors | BIP-32 master+child |
| `ecdh_commutative` | protocol_security | ECDH shared secret commutativity |
| `ecdsa_recovery` | protocol_security | Recoverable sig -> pubkey |
| `bip32_chain` | protocol_security | BIP-32 derivation chain |
| `hash160_consist` | protocol_security | SHA256+RIPEMD160 consistency |
| `fuzz_serial_rt` | fuzzing | Point serialization roundtrip |

### CUDA conditional compilation

| Guard | Modules affected | Effect |
|-------|-----------------|--------|
| `HAVE_CPU_LIB` | `diff_field_mul`, `diff_ecdsa` | Omitted when CPU lib not linked |
| `SECP256K1_CUDA_LIMBS_32` | 13 sign/verify modules | Return 0 (skip) on 32-bit limb builds |

---

## Benchmark Evidence

| Backend | Binary | Categories | Output |
|---------|--------|-----------|--------|
| CPU | `bench_unified` | 8 (field, scalar, point, ECDSA, Schnorr, CT, libsecp256k1 comparison, OpenSSL comparison) | JSON + stdout |
| CUDA | `gpu_bench_unified` | 7 (mirrors CPU format) | JSON + stdout |
| CUDA | `bench_compare` | CPU vs GPU side-by-side | stdout |
| CUDA | `bench_bip352` | Silent Payments pipeline (GLV + LUT modes) | stdout |
| CUDA | `bench_zk` | ZK operations | stdout |
| OpenCL | `opencl_benchmark` | Core ops | stdout |
| OpenCL | `opencl_bip352_benchmark` | Silent Payments pipeline (`--batch`, `--lut` flags) | stdout |
| Metal | `metal_secp256k1_bench_full` | Core ops | stdout |

### BIP-352 Scan Benchmark Numbers (RTX 5060 Ti, 2026-04-04)

| Backend | Mode | Time/Op | Throughput | vs CPU |
|---------|------|---------|------------|--------|
| CUDA | GLV (tpb=384, N=500K) | 178.9 ns | 5.59 M/s | **136.6x** |
| CUDA | + LUT 16×64K (N=500K) | 90.5 ns | 11.05 M/s | **270.0x** |
| OpenCL | Fused (local=128, N=50K) | 197.9 ns | 5.05 M/s | 1.10x vs CUDA GLV |
| OpenCL | + LUT (local=128, N=50K) | 96.5 ns | 10.36 M/s | 1.07x vs CUDA+LUT |
| CPU | UltrafastSecp256k1 KPlan | 24,436.5 ns | 40.9 K/s | 1.00x |

### ECDSA SNARK Witness Benchmark Numbers (RTX 5060 Ti, eprint 2025/695, 2026-04-05)

**bench_snark_witness** — N=65536, 11 passes, median, tpb=128

| Backend | Time/Op | Throughput | vs CPU |
|---------|---------|------------|--------|
| CUDA | 224.8 ns | 4.45 M/s | **107x** |
| CPU (single-thread) | 24,118.5 ns | 41.5 K/s | 1.00x |

Witness size: 760 bytes. Correctness validated (GPU s_inv matches CPU, valid flag matches).

---

## Parity Gaps (remaining)

| # | Gap | Backends | Severity | Notes |
|---|-----|----------|----------|-------|
| 1 | CT audit modules | OpenCL, Metal | Low | Kernel files exist but audit runner doesn't exercise them |
| 2 | Differential CPU-GPU | OpenCL (1 of 3), Metal (1 of 3) | Low | Only scalar mul differential, not field/ECDSA |
| 3 | Memory safety tests | OpenCL, Metal | Informational | CUDA-specific (device alloc/error state) |

All remaining gaps are **non-blocking** for release. The stable GPU ABI now exposes full functional parity for generator mul, ECDSA verify, Schnorr verify, ECDH, Hash160, MSM, FROST partial verification, and `ecrecover_batch` across CUDA, OpenCL, and Metal.

**New in 3.4.0:** 5 additional GPU batch operations (`zk_knowledge_verify_batch`, `zk_dleq_verify_batch`, `bulletproof_verify_batch`, `bip324_aead_encrypt_batch`, `bip324_aead_decrypt_batch`) are fully implemented on CUDA; OpenCL and Metal have explicit stubs with `TODO(parity)` tracking.

---

## Verification Commands

```bash
# CPU unified audit
./build-linux/audit/unified_audit_runner --report-dir reports/

# CUDA audit
./build-cuda/cuda/gpu_audit_runner --report-dir reports/

# OpenCL audit (requires --kernel-dir)
./build-opencl/opencl/opencl_audit_runner --kernel-dir src/opencl/kernels/

# Metal audit (macOS only)
./build-metal/metal/metal_audit_runner --report-dir reports/

# CTest (all backends)
ctest --test-dir build-linux -C Release --output-on-failure
```
