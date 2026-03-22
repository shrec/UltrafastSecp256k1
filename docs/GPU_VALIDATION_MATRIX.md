# GPU Validation Matrix

Unified view of GPU backend validation coverage in UltrafastSecp256k1.

This document answers four practical questions for each backend:

1. Do we have correctness tests?
2. Do we have a unified self-audit runner?
3. Do we have benchmark coverage?
4. Do we have host-side integration tests?

It is intended as an engineering checklist, not a marketing page.

---

## C ABI Ops -- Per-Backend Status

The C ABI layer (`ufsecp_gpu.h`) currently exposes 8 backend-neutral GPU batch
operations. Backend support varies, and operations returning
`UFSECP_ERR_GPU_UNSUPPORTED` (104) gracefully decline.

| Operation | CUDA | OpenCL | Metal | Data Class |
|-----------|------|--------|-------|------------|
| `generator_mul_batch` | implemented | implemented | implemented | PUBLIC |
| `ecdsa_verify_batch` | implemented | implemented | implemented | PUBLIC |
| `schnorr_verify_batch` | implemented | implemented | implemented | PUBLIC |
| `ecdh_batch` | implemented | implemented | implemented | SECRET |
| `hash160_pubkey_batch` | implemented | implemented | implemented | PUBLIC |
| `msm` | implemented | implemented | implemented | PUBLIC |
| `frost_verify_partial_batch` | implemented | implemented | implemented | PUBLIC |
| `ecrecover_batch` | implemented | temporary stub (`UFSECP_ERR_GPU_UNSUPPORTED`) | temporary stub (`UFSECP_ERR_GPU_UNSUPPORTED`) | PUBLIC |
| **Total (unified GPU C ABI)** | **8/8** | **7/8** | **7/8** | |

### Expansion Roadmap

- **OpenCL next in the unified GPU C ABI**: `ecrecover_batch` (the native OpenCL backend already has the recovery device function; the remaining work is wiring a batch kernel through the shared host ABI layer)
- **Metal next in the unified GPU C ABI**: `ecrecover_batch` shader/kernel parity

### C ABI Test Coverage

| Test | Scope | Guard |
|------|-------|-------|
| `gpu_abi_gate` | ABI surface, error codes, discovery, lifecycle, NULL handling | GPU host + ufsecp |
| `gpu_ops_equivalence` | GPU vs CPU reference for supported ops (skips `UFSECP_ERR_GPU_UNSUPPORTED`) | GPU host + ufsecp |
| `gpu_host_api_negative` | NULL ptrs, count=0, invalid backend/device, error strings | GPU host + ufsecp |
| `gpu_backend_matrix` | Backend enumeration, device info, per-backend op probing | GPU host + ufsecp |

### CI and Local Verification

| Environment | CUDA | OpenCL | Metal | Tests |
|-------------|------|--------|-------|-------|
| **Local (dev machine)** | [OK] RTX 5060 Ti | [OK] RTX 5060 Ti | N/A (Linux) | All 49 tests pass including gpu_abi_gate, gpu_ops_equivalence, gpu_host_api_negative, gpu_backend_matrix |
| **GitHub Actions CI** | N/A (no GPU runners) | N/A (no GPU runners) | [OK] macOS (lifecycle) | Metal discovery + lifecycle via macOS job |

> **Note**: GitHub Actions standard runners do not have NVIDIA GPUs or OpenCL devices. CUDA and OpenCL tests are validated locally on developer machines with GPU hardware. Self-hosted GPU runners are planned for future CI coverage.

---

## Summary

| Backend | Correctness Tests | Unified Audit | Unified Bench | Host / Integration | Notes |
|--------|-------------------|---------------|---------------|--------------------|-------|
| CUDA | [OK] | [OK] | [OK] | [OK] | Strongest GPU validation path today |
| ROCm/HIP | [!] Planned / Source-Shared | [!] Planned / Source-Shared | [!] Planned / Source-Shared | [!] Planned / Source-Shared | Shares CUDA/HIP code path, but not yet validated on real AMD hardware |
| OpenCL | [OK] | [OK] | [OK] | [OK] | Good coverage, entry points are more distributed |
| Metal | [OK] | [OK] | [OK] | [OK] | Good coverage on Apple platforms |

ROCm/HIP reuses the CUDA/HIP source tree and runners, but real AMD GPU validation is still pending.

---

## CUDA / ROCm

### Main Entry Points

- Benchmark: [gpu_bench_unified.cu](/home/shrek/Secp256K1/Secp256K1fast/libs/UltrafastSecp256k1/cuda/src/gpu_bench_unified.cu)
- Audit runner: [gpu_audit_runner.cu](/home/shrek/Secp256K1/Secp256K1fast/libs/UltrafastSecp256K1/cuda/src/gpu_audit_runner.cu)
- Full test suite: [test_suite.cu](/home/shrek/Secp256K1/Secp256K1fast/libs/UltrafastSecp256K1/cuda/src/test_suite.cu)
- CT smoke: [test_ct_smoke.cu](/home/shrek/Secp256K1/Secp256K1fast/libs/UltrafastSecp256K1/cuda/src/test_ct_smoke.cu)
- Specialized benches:
  - [bench_bip352.cu](/home/shrek/Secp256K1/Secp256K1fast/libs/UltrafastSecp256K1/cuda/src/bench_bip352.cu)
  - [bench_zk.cu](/home/shrek/Secp256K1/Secp256K1fast/libs/UltrafastSecp256K1/cuda/src/bench_zk.cu)
  - [bench_cuda.cu](/home/shrek/Secp256K1/Secp256K1fast/libs/UltrafastSecp256K1/cuda/src/bench_cuda.cu)

### Coverage

| Area | Status | Notes |
|------|--------|-------|
| Field arithmetic | [OK] | Included in selftest + audit runner + unified bench |
| Scalar arithmetic | [OK] | Included in unified bench and audit runner |
| Point operations | [OK] | Add/double/kG/kP covered |
| ECDSA | [OK] | Sign/verify in bench + audit |
| Schnorr | [OK] | Sign/verify in bench + audit |
| ECDH | [OK] | Present in audit runner |
| Recovery | [OK] | Present in audit runner |
| Batch verify | [OK] | Included in audit runner |
| BIP32 | [OK] | Present in audit runner |
| CT GPU path | [OK] | Bench + CT smoke present |
| Real workload benches | [OK] | BIP-352 and ZK present |

### Current Strength

CUDA is the most unified backend today. If someone asks, "Which GPU backend has the cleanest validation story?" the answer is CUDA.

### Remaining Engineering Gaps

- ROCm/HIP should not be treated as validated until tested on real AMD hardware.
- Keep cross-device reproducibility artifacts organized by GPU model and driver version.
- Keep backend-specific regression logs together with benchmark JSON/TXT artifacts.

---

## OpenCL

### Main Entry Points

- Audit runner: [opencl_audit_runner.cpp](/home/shrek/Secp256K1/Secp256K1fast/libs/UltrafastSecp256K1/opencl/src/opencl_audit_runner.cpp)
- Selftest: [opencl_selftest.cpp](/home/shrek/Secp256K1/Secp256K1fast/libs/UltrafastSecp256K1/opencl/src/opencl_selftest.cpp)
- Extended test + bench: [opencl_extended_test.cpp](/home/shrek/Secp256K1/Secp256K1fast/libs/UltrafastSecp256K1/opencl/tests/opencl_extended_test.cpp)
- Basic test harness: [test_opencl.cpp](/home/shrek/Secp256K1/Secp256K1fast/libs/UltrafastSecp256K1/opencl/tests/test_opencl.cpp)
- Benchmark app: [bench_opencl.cpp](/home/shrek/Secp256K1/Secp256K1fast/libs/UltrafastSecp256K1/opencl/benchmarks/bench_opencl.cpp)

### Coverage

| Area | Status | Notes |
|------|--------|-------|
| Field arithmetic | [OK] | Covered by selftest + extended test |
| Point operations | [OK] | Covered by selftest + extended test |
| Scalar / hash / ECDSA / Schnorr / ECDH / recovery / MSM | [OK] | Covered via extended kernel set + host test |
| Audit report generation | [OK] | `opencl_audit_runner` exists |
| Benchmark coverage | [OK] | `opencl_benchmark` + extended test bench mode |
| Host integration | [OK] | Dedicated host-side extended test |

### Current Strength

OpenCL has broad native validation coverage already and is stronger than it may first appear from the unified GPU C ABI table alone.

### Remaining Engineering Gaps

- Entry points are more fragmented than CUDA.
- A single "OpenCL unified benchmark" story should stay easy to discover in docs.
- Cross-vendor reports should be organized clearly: NVIDIA OpenCL, AMD OpenCL, Intel OpenCL.

---

## Metal

### Main Entry Points

- Audit runner: [metal_audit_runner.mm](/home/shrek/Secp256K1/Secp256K1fast/libs/UltrafastSecp256K1/metal/src/metal_audit_runner.mm)
- Extended test + bench: [metal_extended_test.mm](/home/shrek/Secp256K1/Secp256K1fast/libs/UltrafastSecp256K1/metal/tests/metal_extended_test.mm)
- Host test: [test_metal_host.cpp](/home/shrek/Secp256K1/Secp256K1fast/libs/UltrafastSecp256K1/metal/tests/test_metal_host.cpp)
- App bench/test: [metal_test.mm](/home/shrek/Secp256K1/Secp256K1fast/libs/UltrafastSecp256K1/metal/app/metal_test.mm)
- Metal bench app: [bench_metal.mm](/home/shrek/Secp256K1/Secp256K1fast/libs/UltrafastSecp256K1/metal/app/bench_metal.mm)

### Coverage

| Area | Status | Notes |
|------|--------|-------|
| Field arithmetic | [OK] | Covered in tests and app bench |
| Point operations | [OK] | Covered in tests and app bench |
| Extended crypto ops | [OK] | Covered by extended test |
| Audit report generation | [OK] | `metal_audit_runner` exists |
| Benchmark coverage | [OK] | Bench mode and app bench exist |
| Host integration | [OK] | Dedicated host test present |

### Current Strength

Metal has a reasonably complete validation stack and is already beyond "demo backend" level.

### Remaining Engineering Gaps

- Keep Apple GPU model coverage explicit in benchmark docs.
- Keep shader/library build steps easy to reproduce from CI and local machines.

---

## Recommended Backend Checklist

Use this checklist before calling a GPU backend "fully validated" for a release candidate:

- [ ] Backend selftest passes
- [ ] Backend audit runner passes
- [ ] Unified benchmark runs and emits report
- [ ] Host-side integration test passes
- [ ] One real-device benchmark artifact is saved
- [ ] One real-device audit artifact is saved
- [ ] Driver/toolkit version is recorded
- [ ] JSON + TXT reports are archived

---

## Practical Reading

If the goal is day-to-day engineering confidence:

- Start with CUDA as the reference GPU backend.
- Treat OpenCL and Metal as validated but separately operationalized backends.
- Treat ROCm/HIP as source-compatible with CUDA, but require AMD hardware evidence for each serious release.
