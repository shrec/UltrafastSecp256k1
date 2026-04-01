# Backend Assurance Matrix

**UltrafastSecp256k1** -- Feature/correctness coverage by compute backend

> Performance scales across backends. Assurance does not — it must be measured.

## TL;DR

Not all backends have equal assurance. Each is evaluated independently against
audit coverage, CI enforcement, and benchmark validation.

Backend trust is measured, not assumed.

| Backend | Assurance Level | Notes |
|---------|----------------|-------|
| CPU (fast path) | **HIGH** | Full audit coverage, all invariants, CI enforced |
| CPU (CT path) | **HIGH** | Formal CT verification (LLVM + empirical + Valgrind) |
| CUDA | **HIGH** | Full GPU ABI audit, benchmark validated, CI enforced |
| OpenCL | **MEDIUM** | ABI-complete, partial differential coverage |
| Metal | **MEDIUM** | ABI-complete, CI validated, hardware-level CT unprotected |
| ROCm/HIP | **EXPERIMENTAL** | ABI partial, hardware-backed validation pending |

## Assurance Levels

- **HIGH** — full audit coverage, CI-enforced, reproducible locally
- **MEDIUM** — ABI-complete, partial coverage, evolving
- **EXPERIMENTAL** — limited validation, not recommended for critical paths

---

## Feature Matrix

The table below distinguishes between the **public GPU ABI** (functions exposed via
`ufsecp_gpu_*` in `ufsecp_gpu.h`) and **internal GPU kernel support** (primitives
compiled into the device code but not directly callable through the stable C ABI).
A kernel being present internally does not imply a public API exists for it.

### Public GPU ABI operations (13 functions, backend-neutral)

| Function | CPU (fast) | CPU (CT) | CUDA | OpenCL | Metal |
|---|---|---|---|---|---|
| `ufsecp_gpu_generator_mul_batch` (k·G) | Y | - | Y | Y | Y |
| `ufsecp_gpu_ecdsa_verify_batch` | Y | - | Y | Y | Y |
| `ufsecp_gpu_schnorr_verify_batch` | Y | - | Y | Y | Y |
| `ufsecp_gpu_ecdh_batch` ¹ | Y | Y | Y | Y | Y |
| `ufsecp_gpu_hash160_pubkey_batch` | Y | - | Y | Y | Y |
| `ufsecp_gpu_ecrecover_batch` | Y | - | Y | Y | Y |
| `ufsecp_gpu_msm` | Y | - | Y | Y | Y |
| `ufsecp_gpu_frost_verify_partial_batch` | Y | - | Y | Y | Y |
| `ufsecp_gpu_zk_knowledge_verify_batch` | Y | - | Y | Y | Y |
| `ufsecp_gpu_zk_dleq_verify_batch` | Y | - | Y | Y | Y |
| `ufsecp_gpu_bulletproof_verify_batch` | Y | - | Y | Y | Y |
| `ufsecp_gpu_bip324_aead_encrypt_batch` | Y | - | Y | Y | Y |
| `ufsecp_gpu_bip324_aead_decrypt_batch` | Y | - | Y | Y | Y |

¹ `ufsecp_gpu_ecdh_batch` is the only GPU public API function that accepts private
keys. This is an intentional exception for BIP-352 silent payment scanning workloads
where the ECDH step cannot be split from the GPU pipeline without losing throughput.
Callers must accept the implied security posture of sending private keys to the GPU
driver. See *Secret-Use Policy* below.

### CPU-only operations (no GPU public API)

| Feature | CPU (fast) | CPU (CT) | GPU note |
|---|---|---|---|
| ECDSA sign | Y | Y (CT) | GPU kernel exists for CT smoke testing only; never in production signing path |
| Schnorr sign (BIP-340) | Y | Y (CT) | Same as above |
| BIP-32 HD derivation | Y | Y | Internal GPU kernel (`bip32.cuh`, `secp256k1_bip32.h`) — no public GPU API |
| ECDSA sign batch | - | Y | CPU CT only; private keys never sent to GPU by design |
| Schnorr sign batch | - | Y | CPU CT only; private keys never sent to GPU by design |

### Internal GPU kernel support (not in public ABI)

These primitives are compiled into the CUDA/OpenCL/Metal device code and used
internally by the public batch operations above. They are not callable directly
through `ufsecp_gpu.h`.

| Primitive | CUDA | OpenCL | Metal | Used by |
|---|---|---|---|---|
| Field ops (mul, sqr, inv) | Y | Y | Y | all point operations |
| CT field ops | Y | Y | Y | CT smoke tests |
| CT scalar ops | Y | Y | Y | CT smoke tests |
| CT point ops (complete add, `jacobian_cmov`) | Y | Y | Y | CT smoke tests, ECDH |
| CT sign (generator mul, ECDSA, Schnorr) | Y | Y | Y | CT smoke tests only |
| CT ZK (range prove, inner product) | Y | Y | Y | range prove device path |
| Pedersen commitment | Y | Y | Y | `bulletproof_verify_batch` internals |
| Keccak-256 / `eth_address` | Y | Y | Y | internal Ethereum address derivation |
| BIP-32 derive child | Y | - | Y | internal HD key derivation (app use) |

---

## Permanent Architecture Exceptions

| Operation | Backend | Reason |
|---|---|---|
| `ecdsa_sign_batch` / `schnorr_sign_batch` | CUDA / OpenCL / Metal | Private keys never sent to GPU. Signing is CPU CT-only by design. |
| BIP-32 derivation batch | CUDA / OpenCL / Metal | No public GPU API. Internal kernel exists for app use only. |

---

## Parity Status

All 13 public GPU ABI operations are implemented on CUDA, OpenCL, and Metal.
No partial stubs remain. Last resolved: 2026-03-24.

ROCm/HIP: early-development compatibility path via the shared CUDA/HIP portability
layer. Not yet part of the hardware-validated matrix. Promotion requires archived
benchmark JSON, audit output, driver metadata, and a real AMD device record
per `docs/GPU_BACKEND_EVIDENCE.json`. CUDA source-sharing is not acceptable
evidence for ROCm/HIP promotion.

---

## Audit Coverage

| Audit Type | CPU | CUDA | OpenCL | Metal |
|---|---|---|---|---|
| Audit runner binary | `unified_audit_runner` | `gpu_audit_runner` | `opencl_audit_runner` | `metal_audit_runner` |
| Audit modules | 49 | 27+ | 27 | 27 |
| Selftest | Y | Y | Y | Y |
| CT equivalence | Y | Y (smoke) | Y | Y |
| Side-channel (dudect / device-cycle probe) | Y (600 s) | Y (Welch t-test, `clock64()`) | - | - |
| Differential | Y | - | - | - |
| Fault injection | Y | - | - | - |
| Wycheproof vectors | Y | - | - | - |
| Fuzz harnesses | Y | - | - | - |
| Adversarial protocol | Y | - | - | - |

---

## Benchmark Coverage

| Benchmark | CPU | CUDA | OpenCL | Metal |
|---|---|---|---|---|
| Benchmark binary | `bench_unified` | `gpu_bench_unified` | `opencl_benchmark` | `metal_secp256k1_bench_full` |
| Field ops | Y | Y | Y | Y |
| Scalar ops | Y | Y | - | - |
| Point ops (k·G, k·P) | Y | Y | Y | Y |
| ECDSA sign/verify | Y | Y | Y | Y |
| Schnorr sign/verify | Y | Y | Y | Y |
| CT overhead ratio | Y | - | - | - |
| Cross-library comparison | Y | - | - | - |

---

## Hardware & Platform

| Property | CPU | CUDA | OpenCL | Metal |
|---|---|---|---|---|
| Supported platforms | Linux, Windows, macOS, Android, RISC-V, ESP32 | Linux, Windows | Linux, Windows, macOS, Android | macOS, iOS |
| Minimum requirement | C++20 compiler | SM 5.0+ (Maxwell) | OpenCL 1.2+ | Metal 2.0+ (Apple Silicon) |
| Build option | (always on) | `-DSECP256K1_BUILD_CUDA=ON` | `-DSECP256K1_BUILD_OPENCL=ON` | `-DSECP256K1_BUILD_METAL=ON` |
| Default CUDA architectures | — | `CMAKE_CUDA_ARCHITECTURES=86;89` | — | — |

---

## Secret-Use Policy

| Backend | Accepts private keys? | Policy |
|---|---|---|
| CPU (fast) | No | Variable-time only — public data, batch verify, search |
| CPU (CT) | Yes | Constant-time mandatory for all secret-bearing operations |
| CUDA | ECDH only (exception ¹) | Search/batch verify workloads; no signing on GPU |
| OpenCL | ECDH only (exception ¹) | Search/batch verify workloads; no signing on GPU |
| Metal | ECDH only (exception ¹) | Search/batch verify workloads; no signing on GPU |

GPU CT kernels exist to verify device paths match the constant-time CPU reference
(CT smoke tests). They are not a recommendation to move private-key signing to GPU.

> **GPU is variable-time**: GPU kernels are NOT constant-time with respect to secret inputs.
> `ufsecp_gpu_ecdh_batch` accepts private keys for BIP-352 scanning workloads where the GPU
> operates in a trusted single-tenant environment. Sending secret keys to GPU in a shared/cloud
> GPU environment is a critical vulnerability. Production signing MUST use the CPU CT layer.

---

## CTest Targets by Backend

### CPU
`selftest`, `comprehensive`, `exhaustive`, `field_52`, `field_26`, `hash_accel`,
`batch_add_affine`, `bip340_vectors`, `bip340_strict`, `bip32_vectors`, `bip39`,
`rfc6979_vectors`, `ecc_properties`, `edge_cases`, `ethereum`, `wallet`,
`ct_sidechannel`, `ct_sidechannel_smoke`, `differential`, `ct_equivalence`,
`fault_injection`, `debug_invariants`, `fiat_crypto_vectors`, `carry_propagation`,
`wycheproof_ecdsa`, `wycheproof_ecdh`, `batch_randomness`, `cross_platform_kat`,
`abi_gate`, `ct_verif_formal`, `fiat_crypto_linkage`, `audit_fuzz`,
`adversarial_protocol`, `ecies_regression`, `diag_scalar_mul`, `unified_audit`

### CUDA
`cuda_selftest`, `gpu_audit`, `gpu_ct_smoke`, `gpu_ct_leakage_probe`

### OpenCL
`opencl_selftest`, `opencl_audit`

### Metal
`secp256k1_metal_test`, `secp256k1_metal_audit`, `secp256k1_metal_bench`,
`secp256k1_metal_bench_full`, `metal_host_test`
