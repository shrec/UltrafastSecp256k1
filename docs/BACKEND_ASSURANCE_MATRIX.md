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

### Non-GPU Product Profile Assurance (Added 2026-05-01)

Full taxonomy: [docs/PRODUCT_PROFILES.md](PRODUCT_PROFILES.md).

| Profile | Tier | CT Status | CAAS Gate |
|---|---|---|---|
| `bitcoin-core-backend` (CPU + libsecp256k1 shim) | `production` | Full CT via `secp256k1::ct::*` as of 2026-04-28/2026-05-01 | Hard (audit_gate + security_autonomy + bundle_verify) |
| `cpu-signing` (public C++ API) | `production` | `signing_generator_mul()` → `ct::generator_mul_blinded()` | Hard |
| `ffi-bindings` (legacy C API + bindings) | `beta` | CT signing as of 2026-05-01; bindings inherit from C API | Partial |
| `wasm` | `experimental` | Prebuilt artifact — WASM-specific CT evidence is incomplete | None — do not claim production-CT without CI rebuild + timing analysis |
| `bchn-compat` | `compat-only` | CT generator mul + strict key parsing as of 2026-05-01 | Advisory only — NOT Bitcoin Core, NOT BIP-340 |

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

### Public GPU ABI operations (16 functions, backend-neutral)

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
| `ufsecp_gpu_zk_ecdsa_snark_witness_batch` | Y | - | Y | Y | Y |
| `ufsecp_gpu_zk_schnorr_snark_witness_batch` | Y | - | Y | Y | Y |
| `ufsecp_gpu_bip352_scan_batch` | Y | - | Y | Y | Y |

¹ Several GPU public API functions accept private or secret key material:
`ufsecp_gpu_ecdh_batch`, `ufsecp_gpu_bip352_scan_batch`,
`ufsecp_gpu_bip324_aead_encrypt_batch`, and `ufsecp_gpu_bip324_aead_decrypt_batch`.
These are intentional for high-throughput workloads (BIP-352 scanning, BIP-324
transport encryption) where the secret-bearing step cannot be split from the GPU
pipeline without losing throughput. Callers must accept the implied security posture
of sending keys to the GPU driver and must ensure a trusted single-tenant
environment. See *Secret-Use Policy* below.

### CPU-only operations (no GPU public API)

| Feature | CPU (fast) | CPU (CT) | GPU note |
|---|---|---|---|
| ECDSA sign | Y | Y (CT) | GPU kernel exists for CT smoke testing; production signing uses CPU CT layer |
| Schnorr sign (BIP-340) | Y | Y (CT) | Same as above |
| BIP-32 HD derivation | Y | Y | Internal GPU kernel (`bip32.cuh`, `secp256k1_bip32.h`) — no public GPU API |
| ECDSA sign batch | - | Y | CPU CT only; no GPU public API for batch signing |
| Schnorr sign batch | - | Y | CPU CT only; no GPU public API for batch signing |

### Internal GPU kernel support (not in public ABI)

These primitives are compiled into the CUDA/OpenCL/Metal device code and used
internally by the public batch operations above. They are not callable directly
through `ufsecp_gpu.h`.

| Primitive | CUDA | OpenCL | Metal | Used by |
|---|---|---|---|---|
| Field ops (mul, sqr, inv) | Y | Y | Y | all point operations |
| CT field ops | Y | Y | Y | CT smoke tests (CUDA: `test_ct_smoke.cu`; OpenCL: `gpu_ct_smoke` audit section; Metal: `ct_smoke_kernel`) |
| CT scalar ops | Y | Y | Y | CT smoke tests — same coverage as CT field ops |
| CT point ops (complete add, `jacobian_cmov`) | Y | Y | Y | CT smoke tests, ECDH |
| CT sign (generator mul, ECDSA, Schnorr) | Y | Y | Y | CT smoke tests on all 3 backends (code-discipline CT; vendor JIT caveat — see AUDIT_PHILOSOPHY.md) |
| CT ZK (range prove, inner product) | Y | Y | Y | range prove device path |
| Pedersen commitment | Y | Y | Y | `bulletproof_verify_batch` internals |
| Keccak-256 / `eth_address` | Y | Y | Y | internal Ethereum address derivation |
| BIP-32 derive child | Y | - | Y | internal HD key derivation (app use) |

---

## Permanent Architecture Exceptions

| Operation | Backend | Reason |
|---|---|---|
| `ecdsa_sign_batch` / `schnorr_sign_batch` | CUDA / OpenCL / Metal | No GPU public API. Production signing uses CPU CT layer. |
| BIP-32 derivation batch | CUDA / OpenCL / Metal | No public GPU API. Internal kernel exists for app use only. |

---

## Parity Status

> **Parity tracking is machine-generated.** The source graph (`tools/source_graph_kit/source_graph.py`)
> cross-references the `GpuBackend` virtual interface against CUDA, OpenCL, and Metal
> implementations on every CI build. Any gap introduced by a commit is flagged
> immediately by the parity audit workflow. The numbers below reflect the current
> HEAD — they are not a manually maintained snapshot.

All 16 public GPU ABI operations are implemented natively on CUDA, OpenCL, and Metal.
No partial stubs or CPU fallbacks remain for any of them. Last resolved: 2026-04-24.

`ufsecp_gpu_zk_schnorr_snark_witness_batch` (added 2026-04-15; GPU-native kernels
added 2026-04-24): native device kernels now exist on all three backends
(CUDA: `schnorr_snark_witness_batch_kernel` in `src/cuda/src/secp256k1.cu`;
OpenCL: `schnorr_snark_witness_batch` in `src/opencl/kernels/secp256k1_extended.cl`;
Metal: `schnorr_snark_witness_batch` in `src/metal/shaders/secp256k1_kernels.metal`).
The CPU fallback in `gpu_backend_fallback.cpp` is retained for reference and as a
correctness baseline, but no backend dispatches through it any longer.
CPU-side `ufsecp_zk_schnorr_snark_witness()` is fully functional.

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
| CUDA | Yes (ECDH, BIP-352, BIP-324 AEAD) ¹ | Trusted single-tenant only; batch verify/search; no signing |
| OpenCL | Yes (ECDH, BIP-352, BIP-324 AEAD) ¹ | Trusted single-tenant only; batch verify/search; no signing |
| Metal | Yes (ECDH, BIP-352, BIP-324 AEAD) ¹ | Trusted single-tenant only; batch verify/search; no signing |

GPU CT kernels exist to verify device paths match the constant-time CPU reference
(CT smoke tests). They are not a recommendation to move private-key signing to GPU.

> **GPU is variable-time**: GPU kernels are NOT constant-time with respect to secret inputs.
> Several GPU API functions accept private keys for high-throughput workloads
> (`ecdh_batch`, `bip352_scan_batch`, `bip324_aead_*_batch`). These require a
> trusted single-tenant environment. Sending secret keys to GPU in a shared/cloud
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
