# Backend Assurance Matrix

**UltrafastSecp256k1** -- Feature/correctness coverage by compute backend

---

## Feature Matrix

| Feature | CPU (fast) | CPU (CT) | CUDA | OpenCL | Metal |
|---------|-----------|----------|------|--------|-------|
| ECDSA sign | Y | Y (CT) | Y | Y | Y |
| ECDSA verify | Y | - | Y | Y | Y |
| Schnorr sign (BIP-340) | Y | Y (CT) | Y | Y | Y |
| Schnorr verify | Y | - | Y | Y | Y |
| Batch verify | Y | - | Y | - | Y |
| BIP-32 HD derivation | Y | Y | Y | Y | Y |
| ECDH | Y | Y | Y | Y | Y |
| Hash160 (RIPEMD160) | Y | - | Y | Y | Y |
| Keccak-256 | Y | - | Y | Y | Y |
| Pedersen commitment | Y | Y | Y | Y | Y |
| ZK proofs | Y | Y | Y | Y | Y |
| ZK knowledge verify batch | - | - | Y | Y | Y |
| ZK DLEQ verify batch | - | - | Y | Y | Y |
| Bulletproof verify batch | - | - | Y | Y | Y |
| BIP-324 AEAD encrypt batch | - | - | Y | Y | Y |
| BIP-324 AEAD decrypt batch | - | - | Y | Y | Y |
| Multi-scalar mul | Y | - | Y | Y | Y |
| CT field ops | - | Y | Y | Y | Y |
| CT scalar ops | - | Y | Y | Y | Y |
| CT point ops | - | Y | Y | Y | Y |
| CT sign | - | Y | Y | Y | Y |
| CT ZK | - | Y | Y | Y | Y |
| Bloom filter | Y | - | Y | Y | Y |
| Key recovery (single) | Y | - | Y | Y | Y |
| Key recovery batch (`ecrecover_batch`) | Y | - | Y | Y | Y |
| ECDSA sign batch (CPU CT) | - | Y | N/A | N/A | N/A |
| Schnorr sign batch (CPU CT) | - | Y | N/A | N/A | N/A |

---

## Parity Tracking

### Current temporary stubs (must be resolved before full OpenCL/Metal parity)

| Operation | Backend | Tracking note |
|-----------|---------|---------------|
| *(none — all parity gaps resolved)* | — | — |

> All ZK and BIP-324 batch operations, including `bulletproof_verify_batch`, are now
> **fully implemented on all three GPU backends** (CUDA, OpenCL, Metal).
> OpenCL kernel `#if 0` guard removed; address-space qualifier fix applied to
> `range_verify_full_impl` (added `__global` to `bp_G`/`bp_H`, local copy in loop).
> Metal host dispatch wired via `range_proof_poly_batch` kernel.
> Resolved 2026-03-24.

### Current permanent exceptions

| Operation | Backend | Reason |
|-----------|---------|--------|
| `ecdsa_sign_batch` / `schnorr_sign_batch` | CUDA / OpenCL / Metal | Architecture decision: private keys never sent to GPU. Signing is CPU CT-only by design. |

---

## Audit Coverage

| Audit Type | CPU | CUDA | OpenCL | Metal |
|-----------|-----|------|--------|-------|
| Audit runner binary | `unified_audit_runner` | `gpu_audit_runner` | `opencl_audit_runner` | `metal_audit_runner` |
| Audit modules | 49 | 27+ | 27 | 27 |
| Selftest | Y | Y | Y | Y |
| CT equivalence | Y | Y (smoke) | Y | Y |
| Side-channel (dudect) | Y (600s) | - | - | - |
| Differential | Y | - | - | - |
| Fault injection | Y | - | - | - |
| Wycheproof vectors | Y | - | - | - |
| Fuzz harnesses | Y | - | - | - |
| Adversarial protocol | Y | - | - | - |

---

## Benchmark Coverage

| Benchmark | CPU | CUDA | OpenCL | Metal |
|-----------|-----|------|--------|-------|
| Benchmark binary | `bench_unified` | `gpu_bench_unified` | `opencl_benchmark` | `metal_secp256k1_bench_full` |
| Field ops | Y | Y | Y | Y |
| Scalar ops | Y | Y | - | - |
| Point ops (kG, kP) | Y | Y | Y | Y |
| ECDSA sign/verify | Y | Y | Y | Y |
| Schnorr sign/verify | Y | Y | Y | Y |
| CT overhead ratio | Y | - | - | - |
| Cross-library comparison | Y | - | - | - |

---

## Hardware & Platform

| Property | CPU | CUDA | OpenCL | Metal |
|----------|-----|------|--------|-------|
| Supported platforms | Linux, Windows, macOS, Android, RISC-V, ESP32 | Linux, Windows | Linux, Windows, macOS, Android | macOS, iOS |
| Minimum requirement | C++20 compiler | SM 5.0+ (Maxwell) | OpenCL 1.2+ | Metal 2.0+ (Apple Silicon) |
| Build option | (always on) | `-DSECP256K1_BUILD_CUDA=ON` | `-DSECP256K1_BUILD_OPENCL=ON` | `-DSECP256K1_BUILD_METAL=ON` |
| Default architectures | native | `CMAKE_CUDA_ARCHITECTURES=86;89` | all available | Apple Silicon |

---

## Secret-Use Policy

| Backend | Signs with secrets? | Policy |
|---------|-------------------|--------|
| CPU (fast) | No | Variable-time only (public data, batch verify, search) |
| CPU (CT) | **Yes** | Constant-time mandatory (signing, key derivation) |
| CUDA | No | Search/batch workloads only; no secret keys on GPU |
| OpenCL | No | Search/batch workloads only; no secret keys on GPU |
| Metal | No | Search/batch workloads only; no secret keys on GPU |

**Note**: GPU backends have CT kernel implementations for correctness testing (CT smoke tests), but production signing MUST use CPU CT layer. GPU CT kernels exist for verification that the CT implementation produces equivalent results.

---

## CTest Targets by Backend

### CPU
`selftest`, `comprehensive`, `exhaustive`, `field_52`, `field_26`, `hash_accel`, `batch_add_affine`, `bip340_vectors`, `bip340_strict`, `bip32_vectors`, `bip39`, `rfc6979_vectors`, `ecc_properties`, `edge_cases`, `ethereum`, `wallet`, `ct_sidechannel`, `ct_sidechannel_smoke`, `differential`, `ct_equivalence`, `fault_injection`, `debug_invariants`, `fiat_crypto_vectors`, `carry_propagation`, `wycheproof_ecdsa`, `wycheproof_ecdh`, `batch_randomness`, `cross_platform_kat`, `abi_gate`, `ct_verif_formal`, `fiat_crypto_linkage`, `audit_fuzz`, `adversarial_protocol`, `ecies_regression`, `diag_scalar_mul`, `unified_audit`

### CUDA
`cuda_selftest`, `gpu_audit`, `gpu_ct_smoke`

### OpenCL
`opencl_selftest`, `opencl_audit`

### Metal
`secp256k1_metal_test`, `secp256k1_metal_audit`, `secp256k1_metal_bench`, `secp256k1_metal_bench_full`, `metal_host_test`
