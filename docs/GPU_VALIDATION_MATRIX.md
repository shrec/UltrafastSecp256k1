# GPU Validation Matrix

Unified view of GPU backend validation coverage in UltrafastSecp256k1.

The machine-readable companion for backend status, publishability, and artifact requirements is `docs/GPU_BACKEND_EVIDENCE.json`. When this document and the JSON diverge, treat the JSON as the enforcement source.

This document answers four practical questions for each backend:

1. Do we have correctness tests?
2. Do we have a unified self-audit runner?
3. Do we have benchmark coverage?
4. Do we have host-side integration tests?

It is intended as an engineering checklist, not a marketing page.

---

## C ABI Ops -- Per-Backend Status

The C ABI layer (`ufsecp_gpu.h`) currently exposes 13 backend-neutral GPU batch
operations. CUDA, OpenCL, and Metal all implement that stable public surface.
`UFSECP_ERR_GPU_UNSUPPORTED` (104) remains part of the ABI for unsupported
backend selection, missing device/runtime capability, or invalid execution
context, not as a standing parity gap for the stable 19-op surface.

| Operation | CUDA | OpenCL | Metal | Data Class |
|-----------|------|--------|-------|------------|
| `generator_mul_batch` | implemented | implemented | implemented | PUBLIC |
| `ecdsa_verify_batch` | implemented | implemented | implemented | PUBLIC |
| `schnorr_verify_batch` | implemented | implemented | implemented | PUBLIC |
| `ecdh_batch` | implemented | implemented | implemented | SECRET |
| `hash160_pubkey_batch` | implemented | implemented | implemented | PUBLIC |
| `msm` | implemented | implemented | implemented | PUBLIC |
| `frost_verify_partial_batch` | implemented | implemented | implemented | PUBLIC |
| `ecrecover_batch` | implemented | implemented | implemented | PUBLIC |
| `zk_knowledge_verify_batch` | implemented | implemented | implemented | PUBLIC |
| `zk_dleq_verify_batch` | implemented | implemented | implemented | PUBLIC |
| `bulletproof_verify_batch` | implemented | implemented | implemented | PUBLIC |
| `bip324_aead_encrypt_batch` | implemented | implemented | implemented | PUBLIC |
| `bip324_aead_decrypt_batch` | implemented | implemented | implemented | SECRET |
| **Total (unified GPU C ABI)** | **13/13** | **13/13** | **13/13** | |

### Expansion Roadmap

- Unified 13/13 GPU C ABI parity is closed across CUDA, OpenCL, and Metal.
- The five ZK/BIP-324 batch ops are implemented on all three backends and exposed through the stable C ABI.
- Remaining GPU governance work is no longer backend parity; it is hardware-backed publishability, artifact retention, and cross-device reproducibility.

### C ABI Test Coverage

| Test | Scope | Guard |
|------|-------|-------|
| `gpu_abi_gate` | ABI surface, error codes, discovery, lifecycle, NULL handling | GPU host + ufsecp |
| `gpu_ops_equivalence` | GPU vs CPU reference for the stable public op surface; unsupported paths remain negative-test coverage | GPU host + ufsecp |
| `gpu_host_api_negative` | NULL ptrs, count=0, invalid backend/device, error strings | GPU host + ufsecp |
| `gpu_backend_matrix` | Backend enumeration, device info, per-backend op probing | GPU host + ufsecp |

### CI and Local Verification

| Environment | CUDA | OpenCL | Metal | Tests |
|-------------|------|--------|-------|-------|
| **Local (dev machine)** | [OK] RTX 5060 Ti | [OK] RTX 5060 Ti | N/A (Linux) | All 49 tests pass including gpu_abi_gate, gpu_ops_equivalence, gpu_host_api_negative, gpu_backend_matrix |
| **Local Docker parity** | [OK]* host GPU passthrough | [OK]* host OpenCL/runtime passthrough | N/A (Linux) | Same Linux CI toolchain via `docker-compose.ci.yml` / `Dockerfile.local-ci`; GPU slices remain host-hardware dependent |
| **GitHub Actions CI** | N/A (no GPU runners) | N/A (no GPU runners) | [OK] macOS (lifecycle) | Metal discovery + lifecycle via macOS job |

> **Note**: GitHub Actions standard runners do not have NVIDIA GPUs or OpenCL devices, but that is not the only reproducible Linux path. The repository ships a containerized local CI stack in `Dockerfile.local-ci`, `docker-compose.ci.yml`, and `docs/LOCAL_CI.md`, so contributors can reproduce the same Linux toolchain in Docker on their own machines. GPU validation still requires the host GPU driver/runtime stack and appropriate device passthrough into the container or local process.

---

## Summary

| Backend | Correctness Tests | Unified Audit | Unified Bench | Host / Integration | Notes |
|--------|-------------------|---------------|---------------|--------------------|-------|
| CUDA | [OK] | [OK] | [OK] | [OK] | Strongest GPU validation path today |
| ROCm/HIP | [!] Planned / Source-Shared | [!] Planned / Source-Shared | [!] Planned / Source-Shared | [!] Planned / Source-Shared | Optional future backend expansion; current absence of AMD hardware does not invalidate existing audited backends |
| OpenCL | [OK] | [OK] | [OK] | [OK] | Good coverage, entry points are more distributed |
| Metal | [OK] | [OK] | [OK] | [OK] | Good coverage on Apple platforms |

ROCm/HIP reuses the CUDA/HIP source tree and runners, but real AMD GPU validation is still pending.

That pending state is intentionally non-blocking for current audit validity.
CPU, CUDA, OpenCL, and Metal assurance claims remain grounded in their own
existing evidence paths; ROCm/HIP only becomes relevant once the project
chooses to publish real AMD-backed claims.

This status is intentionally fail-closed in `docs/GPU_BACKEND_EVIDENCE.json`: ROCm/HIP remains non-publishable until AMD hardware-backed benchmark and audit artifacts exist.

---

## CUDA / ROCm

### Main Entry Points

- Benchmark: [gpu_bench_unified.cu](../src/cuda/src/gpu_bench_unified.cu)
- Audit runner: [gpu_audit_runner.cu](../src/cuda/src/gpu_audit_runner.cu)
- Full test suite: [test_suite.cu](../src/cuda/src/test_suite.cu)
- CT smoke: [test_ct_smoke.cu](../src/cuda/src/test_ct_smoke.cu)
- Specialized benches:
  - [bench_bip352.cu](../src/cuda/src/bench_bip352.cu)
  - [bench_zk.cu](../src/cuda/src/bench_zk.cu)
  - [bench_cuda.cu](../src/cuda/src/bench_cuda.cu)

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

For Linux reproducibility, that story is not limited to the original developer workstation: the same local CI environment can be recreated through the repo's Docker-based local parity infrastructure, then paired with host GPU access for CUDA/OpenCL runs.

### Remaining Engineering Gaps

- ROCm/HIP should not be treated as validated until tested on real AMD hardware.
- Keep cross-device reproducibility artifacts organized by GPU model and driver version.
- Keep backend-specific regression logs together with benchmark JSON/TXT artifacts.

---

## OpenCL

### Main Entry Points

- Audit runner: [opencl_audit_runner.cpp](../src/opencl/src/opencl_audit_runner.cpp)
- Selftest: [opencl_selftest.cpp](../src/opencl/src/opencl_selftest.cpp)
- Extended test + bench: [opencl_extended_test.cpp](../src/opencl/tests/opencl_extended_test.cpp)
- Basic test harness: [test_opencl.cpp](../src/opencl/tests/test_opencl.cpp)
- Benchmark app: [bench_opencl.cpp](../src/opencl/benchmarks/bench_opencl.cpp)

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

OpenCL has broad native validation coverage already and now matches CUDA and Metal on the stable public GPU C ABI surface.

The repo's local Docker parity environment also helps standardize the Linux-side toolchain for OpenCL validation; the remaining variable is the host OpenCL ICD/device stack, not the surrounding build/test container.

### Remaining Engineering Gaps

- Entry points are more fragmented than CUDA.
- A single "OpenCL unified benchmark" story should stay easy to discover in docs.
- Cross-vendor reports should be organized clearly: NVIDIA OpenCL, AMD OpenCL, Intel OpenCL.

---

## Metal

### Main Entry Points

- Audit runner: [metal_audit_runner.mm](../src/metal/src/metal_audit_runner.mm)
- Extended test + bench: [metal_extended_test.mm](../src/metal/tests/metal_extended_test.mm)
- Host test: [test_metal_host.cpp](../src/metal/tests/test_metal_host.cpp)
- App bench/test: [metal_test.mm](../src/metal/app/metal_test.mm)
- Metal bench app: [bench_metal.mm](../src/metal/app/bench_metal.mm)

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

## ROCm/HIP Promotion Checklist

ROCm/HIP is source-shared with the CUDA/HIP portability layer, but it is not
promoted by source compatibility alone. Promotion from `planned` to
`validated` and `publishable: true` requires all of the following on real AMD
hardware:

1. `docs/GPU_BACKEND_EVIDENCE.json` is updated so `rocm-hip` becomes `validated`, `hardware_backed: true`, and `publishable: true` in the same change.
2. Benchmark artifacts exist in both JSON and text form for the recorded AMD device.
3. Audit runner output exists for the same hardware and is reproducible from the recorded command path.
4. Driver/runtime metadata is archived, including ROCm driver version and AMD device model.
5. Any published numbers identify the exact device class and do not reuse CUDA labels or NVIDIA-only evidence.
6. `python3 ci/preflight.py --gpu-evidence` and `python3 ci/validate_assurance.py` pass after the evidence update.

Until those conditions are met, ROCm/HIP remains deliberately fail-closed for
public benchmark and validation claims. This is a containment rule for future
AMD-specific claims, not a defect in the validity of the current audit and
validation surfaces.

This checklist is enforceable through:

```bash
python3 ci/preflight.py --gpu-evidence
python3 ci/validate_assurance.py
```

---

## Practical Reading

If the goal is day-to-day engineering confidence:

- Start with CUDA as the reference GPU backend.
- Treat OpenCL and Metal as validated but separately operationalized backends.
- Treat ROCm/HIP as source-compatible with CUDA, but not promotion-eligible until the AMD hardware-backed checklist above is complete.
