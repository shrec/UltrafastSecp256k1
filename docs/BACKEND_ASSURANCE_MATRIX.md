# Backend Assurance Matrix

**UltrafastSecp256k1** -- Feature/correctness coverage by compute backend

> Performance scales across backends. Assurance does not — it must be measured.

> **Evidence-status gated (Bastion B16).** The GPU/hardware claim surface is made
> explicit and freshness-gated by [`docs/GPU_HARDWARE_EVIDENCE_STATUS.json`](GPU_HARDWARE_EVIDENCE_STATUS.json)
> and [`ci/check_gpu_hardware_evidence.py`](../ci/check_gpu_hardware_evidence.py)
> (also `audit_gate.py --gpu-hardware-evidence`, principle **G-16**). Each row
> declares a `claim_type` (correctness / performance / **fallback_correctness** /
> hardware_ct / out_of_scope): committed host-side / CPU-fallback correctness
> evidence (no GPU needed) is tracked separately from native-device performance,
> real-device CUDA/OpenCL/Metal/ROCm evidence is **owner_gated** (no GitHub GPU
> runners — owner-run, never current on push), and hardware power/EM/fault and
> ROCm/HIP real-device are **documented_residual** rows that must resolve to a
> `docs/RESIDUAL_RISK_REGISTER.md` id (RR-003 / RR-005 / RR-006). A
> `fallback_correctness` row is never counted as native-performance evidence.
> Run: `python3 ci/check_gpu_hardware_evidence.py --json`.

## TL;DR

Not all backends have equal assurance. Each is evaluated independently against
audit coverage, CI enforcement, and benchmark validation.

Backend trust is measured, not assumed.

| Backend | Assurance Level | Notes |
|---------|----------------|-------|
| CPU (fast path) | **HIGH** | Full audit coverage, all invariants, CI enforced |
| CPU (CT path) | **HIGH** | Formal CT verification (LLVM + empirical + Valgrind) |
| CUDA | **HIGH** | Full GPU ABI audit, partially validated (diagnostic runs; no canonical JSON artifact per canonical_numbers.json), CI enforced¹ |
| OpenCL | **MEDIUM** | ABI-complete, partial differential coverage¹ |
| Metal | **MEDIUM** | ABI-complete, CI validated, hardware-level CT unprotected¹ |
| ROCm/HIP | **EXPERIMENTAL** | ABI partial, hardware-backed validation pending |

### libbitcoin bridge "collect" verify (Added 2026-06-02)

The in-place collect verify (`ufsecp_lbtc_verify_*_collect`) has a dedicated
on-device CUDA kernel (`ufsecp_gpu_*_verify_collect`); OpenCL and Metal inherit
the `GpuBackend` default (`GpuError::Unsupported`) and the bridge falls back to
the host-collapse path (the existing, audited `ufsecp_gpu_*_verify_batch` kernels
+ a host-side verdict write). All paths route through the identical verify cores,
so the rejected-set is consensus-identical across them.

| Backend | collect path | Assurance |
|---------|--------------|-----------|
| CPU     | reference (per-row verify) | **HIGH** — gated vs libsecp256k1 |
| CUDA    | dedicated on-device kernel | **HIGH** — `test_lbtc_consensus_diff` proves GPU==CPU==libsecp on the rejected-set (ECDSA+Schnorr, mixed corpus); kernel is a verbatim copy of the audited verify kernel with only the output store changed |
| OpenCL  | host-collapse fallback (`Unsupported` → `*_verify_batch` + host write) | **MEDIUM** — verify kernel ABI-complete; collect verdict applied host-side (no untested device kernel) |
| Metal   | host-collapse fallback | **MEDIUM** — same as OpenCL |

A native OpenCL/Metal collect kernel is a documented follow-up, gated on those
backends gaining local/CI hardware coverage (a consensus-bearing accept/reject
device kernel must never ship unverified).

### libbitcoin opaque ECDSA row verify (Added 2026-06-13)

`ufsecp_gpu_ecdsa_verify_opaque_rows` is the native GPU row-format entry point
for copied `secp256k1_ecdsa_signature` payloads, including libbitcoin's
existing ECDSA batch rows:
`32-byte sighash | 33-byte compressed pubkey | 64-byte opaque
secp256k1_ecdsa_signature | optional tail`. CUDA, OpenCL, and Metal read the
strided rows directly, parse the opaque scalar limbs on device, normalize high-S
signatures before verification, and return per-row verdict bytes. This avoids
bridge-side msg/pub/sig column staging for the packed-row API while preserving
libsecp-compatible verification semantics. `ufsecp_gpu_ecdsa_verify_lbtc_rows`
is retained as a compatibility alias.

### Non-GPU Product Profile Assurance (Added 2026-05-01)

Full taxonomy: [docs/PRODUCT_PROFILES.md](PRODUCT_PROFILES.md).

| Profile | Tier | CT Status | CAAS Gate |
|---|---|---|---|
| `bitcoin-core-backend` (CPU + libsecp256k1 shim) | `production` | Full CT via `secp256k1::ct::*` as of 2026-05-12 (CT-BLIND-01: nonce paths use `generator_mul_blinded`; prior CT arithmetic fixes: 2026-04-28/2026-05-01) | Hard (audit_gate + security_autonomy + bundle_verify) |
| `cpu-signing` (public C++ API) | `production` | `signing_generator_mul()` → `ct::generator_mul_blinded()` | Hard |
| `ffi-bindings` (legacy C API + bindings) | `beta` | CT signing as of 2026-05-01; bindings inherit from C API | Partial |
| `wasm` | `experimental` | Prebuilt artifact — WASM-specific CT evidence is incomplete | None — do not claim production-CT without CI rebuild + timing analysis |
| `bchn-compat` | `compat-only` | CT generator mul + strict key parsing as of 2026-05-01 | Advisory only — NOT Bitcoin Core, NOT BIP-340 |

## Assurance Levels

- **HIGH** — full audit coverage, CI-enforced, reproducible locally
- **MEDIUM** — ABI-complete, partial coverage, evolving
- **EXPERIMENTAL** — limited validation, not recommended for critical paths

¹ GPU CI enforcement is local-only (self-hosted GPU runner with RTX hardware).
  GitHub CI covers CPU audit only — GPU advisory modules return ADVISORY_SKIP_CODE (77)
  in the absence of GPU hardware. "CI enforced" above refers to local CI pipeline.

---

## Feature Matrix

The table below distinguishes between the **public GPU ABI** (functions exposed via
`ufsecp_gpu_*` in `ufsecp_gpu.h`) and **internal GPU kernel support** (primitives
compiled into the device code but not directly callable through the stable C ABI).
A kernel being present internally does not imply a public API exists for it.

### Public GPU ABI operations (17 functions, backend-neutral)

| Function | CPU (fast) | CPU (CT) | CUDA | OpenCL | Metal |
|---|---|---|---|---|---|
| `ufsecp_gpu_generator_mul_batch` (k·G) | Y | - | Y | Y | Y |
| `ufsecp_gpu_ecdsa_verify_batch` | Y | - | Y | Y | Y |
| `ufsecp_gpu_ecdsa_verify_opaque_rows` / `ufsecp_gpu_ecdsa_verify_lbtc_rows` alias | Y | - | Y | Y | Y |
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

As of 2026-06-06, all public GPU C ABI result-bearing operations clear their
outputs to zero/invalid defaults before processing and after backend non-OK
returns, except the in-place collect APIs whose marker buffer is intentionally
caller-owned input/output state. CUDA, OpenCL, and Metal BIP-352 scan paths now
strict-reject zero or order-or-larger scan keys, clear prefix/plan outputs on
failure, and erase host/shared/device scan-key material before releasing buffers.
Metal BIP-324 key buffers now use the same erase-before-release discipline as
CUDA/OpenCL.

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

All 17 public GPU ABI operations are implemented natively on CUDA, OpenCL, and Metal.
No partial stubs or CPU fallbacks remain for any of them. Last resolved:
2026-06-13 (`ufsecp_gpu_ecdsa_verify_opaque_rows`).

`ufsecp_gpu_zk_schnorr_snark_witness_batch` (added 2026-04-15; GPU-native kernels
added 2026-04-24): native device kernels now exist on all three backends
(CUDA: `schnorr_snark_witness_batch_kernel` in `src/cuda/src/secp256k1.cu`;
OpenCL: `schnorr_snark_witness_batch` in `src/opencl/kernels/secp256k1_extended.cl`;
Metal: `schnorr_snark_witness_batch` in `src/metal/shaders/secp256k1_kernels.metal`).
The CPU fallback in `gpu_backend_fallback.cpp` is retained for reference and as a
correctness baseline, but no backend dispatches through it any longer.
CPU-side `ufsecp_zk_schnorr_snark_witness()` is fully functional.

### GPU-side pubkey decompression (2026-06-17)

All three GPU backends now perform SEC1 33-byte compressed pubkey decompression
natively on the device, eliminating CPU-side sqrt+parity computation, host-side
buffer allocation, and 3.2× PCIe data transfer overhead.

| Operation | CUDA | OpenCL | Metal |
|-----------|------|--------|-------|
| ECDSA verify batch | ✅ `point_from_compressed` | ✅ `ecdsa_verify_compressed` | ✅ `ecdsa_verify_batch_compressed` |
| SNARK witness batch | ✅ `batch_compressed_to_jac_kernel` + snark kernel | ✅ `ecdsa_snark_witness_batch_compressed` | ✅ `ecdsa_snark_witness_batch_compressed` |
| ECDH | ✅ `point_from_compressed` | ✅ `ecdh_scalar_mul_compressed` | ✅ `scalar_mul_batch_compressed` |
| MSM | ✅ `point_from_compressed` | ✅ `ecdh_scalar_mul_compressed` | ✅ `scalar_mul_batch_compressed` |
| Schnorr verify | ✅ x-only (32B, no decompress) | ✅ x-only (32B) | ✅ x-only (32B) |
| ecrecover | ✅ (no input pubkey) | ✅ (no input) | ✅ (no input) |

Benchmark (RTX 5060 Ti, CUDA): GPU decompress overhead = +0.8 ns/op (+0.3%)
vs pre-decompressed JacobianPoint verify. Full raw-entry kernel
(decompress + sig parse + low-S normalize + verify) overhead = +1.8 ns (+0.7%).

Kernels added:
- OpenCL: `ecdsa_verify_compressed`, `ecdsa_snark_witness_batch_compressed`, `scalar_mul_compressed`
- Metal: `ecdsa_verify_batch_compressed`, `ecdsa_snark_witness_batch_compressed`, `scalar_mul_batch_compressed`

ROCm/HIP: early-development compatibility path via the shared CUDA/HIP portability
layer. Not yet part of the hardware-validated matrix. Promotion requires archived
benchmark JSON, audit output, driver metadata, and a real AMD device record
per `docs/GPU_BACKEND_EVIDENCE.json`. CUDA source-sharing is not acceptable
evidence for ROCm/HIP promotion.

### Default-stub parity exceptions (libbitcoin-bridge specializations)

Beyond the 16 public ABI ops, the `GpuBackend` interface exposes optional
**libbitcoin-bridge specialization** virtual methods (`ecdsa_verify_collect`,
`schnorr_verify_collect`, `xonly_validate`, `commitment_verify`, `tagged_hash`,
`tagged_hash_var`, `pubkey_validate`, `hash256`, and the ZK/AEAD/BIP-352 batch
variants). These are **CUDA-native** and intentionally default to
`GpuError::Unsupported` on OpenCL/Metal, where the caller transparently falls
back to the host/CPU path (e.g. `*_verify_batch` + a host collapse). All inputs
on these paths are **public data**, and the fallback is deterministic, so this is
a **performance residual, not a correctness parity gap**.

Each such return in `src/gpu/include/gpu_backend.hpp` carries an inline
`PARITY-EXCEPTION` marker, and the `audit_gate.py --gpu-parity` gate enforces that
**every `return ...Unsupported` in GPU backend source is either implemented or
carries a `TODO(parity)`/`PARITY-EXCEPTION` marker** — a backend may not silently
return Unsupported without a documented exception. The gate scans source files
only (build/generated trees are pruned) so the signal is precise.

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
