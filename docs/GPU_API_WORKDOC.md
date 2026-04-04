# GPU API Workdoc

This document is a direct implementation brief for the coder model.

Goal: turn the existing GPU engines into a consumable multi-language acceleration layer without weakening the current internal architecture.

Current state:
- The CUDA side is strong as an internal engine.
- OpenCL and Metal also exist as real backends with their own runtime layers.
- The public stable C ABI currently covers CPU-focused `ufsecp_*` APIs.
- GPU batch operations are explicitly documented as a separate library with no C ABI promise.
- Rust bindings currently expose CPU-side C FFI, not GPU operations.

Target state:
- Keep the existing GPU engines as the internal implementation layer.
- Add a host-side GPU operations API.
- Add a stable backend-neutral C ABI for selected GPU operations.
- Make future Rust/Python/Go bindings depend on that C ABI, not on backend internals.

## Non-Negotiable Rules

1. Do not break or redesign the existing GPU engines.
2. Do not expose raw CUDA/OpenCL/Metal kernels or backend-specific runtime types in the public C ABI.
3. Do not use STL types in the public C ABI.
4. Do not expose `cudaStream_t`, `CUdevice`, `cl_context`, `cl_command_queue`, Metal Objective-C runtime types, `std::vector`, `std::string`, or C++ exceptions across the ABI boundary.
5. Keep naming backend-neutral at the ABI boundary: use `gpu`, not `cuda`, in public function names.
6. Use opaque handles for contexts and plans.
7. Return explicit status codes and lengths for all public operations.
8. Secret-key GPU operations must not be positioned as production-safe unless the threat model already permits it.
9. Default first wave to public-data or clearly bounded workloads.
10. Preserve existing tests, benchmarks, and audit flows.

## Architecture To Build

Implement four layers. Only Layers 2 and 3 are new public-facing work.

### Layer 1: Internal Engines

Already exists. Keep as-is.

Examples:
- `cuda/include/secp256k1.cuh`
- `cuda/include/ecdsa.cuh`
- `cuda/include/schnorr.cuh`
- `cuda/include/ecdh.cuh`
- `cuda/include/msm.cuh`
- `cuda/src/secp256k1.cu`
- `opencl/kernels/*.cl`
- `opencl/src/*.cpp`
- `metal/shaders/*.h`
- `metal/src/*.mm`

Responsibilities:
- device math
- kernels
- low-level backend-specific optimization
- CUDA/HIP portability
- OpenCL runtime dispatch
- Metal runtime dispatch

Do not try to make this layer stable for external users.

### Layer 2: Host Ops API

Add a new host-side API layer for reusable GPU operations.

Suggested location:
- `gpu/include/`
- `gpu/src/`

This layer may internally call:
- CUDA host launchers
- OpenCL command queue dispatch
- Metal command encoder / pipeline dispatch

Suggested types:
- `GpuContext`
- `GpuDeviceInfo`
- `GpuBatchConfig`
- `GpuBufferView`
- `GpuResult`

Suggested responsibilities:
- backend enumeration
- device enumeration
- device selection
- context init/shutdown
- stream/queue/command-buffer ownership
- scratch/workspace management
- argument validation
- launch configuration
- kernel dispatch
- result collection
- error normalization

This layer may use C++ internally, plus Objective-C++ for Metal where required, but must be written so Layer 3 can wrap it cleanly.

### Layer 3: Stable C ABI

Expose a small, stable C ABI for selected GPU operations.

Suggested public header:
- `include/ufsecp/ufsecp_gpu.h`

Suggested opaque types:
- `ufsecp_gpu_ctx`

Suggested common support functions:
- `ufsecp_gpu_backend_count`
- `ufsecp_gpu_backend_info`
- `ufsecp_gpu_device_count`
- `ufsecp_gpu_device_info`
- `ufsecp_gpu_ctx_create`
- `ufsecp_gpu_ctx_destroy`
- `ufsecp_gpu_ctx_clone` only if truly useful
- `ufsecp_gpu_last_error`
- `ufsecp_gpu_last_error_msg`
- `ufsecp_gpu_backend_name`
- `ufsecp_gpu_is_available`

Do not expose backend-specific concepts in this header.

### Layer 4: Language Bindings

Do not fully implement all language bindings in this task.
Only prepare the ABI so these become straightforward:
- Rust
- Python
- Go
- C#

If time permits, add a minimal Rust raw FFI extension after the C ABI lands.

## First-Wave GPU Operations

Implement only the highest-value operations first.

Required first wave:
1. `ufsecp_gpu_generator_mul_batch`
2. `ufsecp_gpu_ecdsa_verify_batch`
3. `ufsecp_gpu_schnorr_verify_batch`
4. `ufsecp_gpu_ecdh_batch`
5. `ufsecp_gpu_hash160_pubkey_batch`
6. `ufsecp_gpu_msm`

These six are enough to turn the current engine into a serious interop layer.

Current implementation status after the stable ABI rollout:
- The stable GPU C ABI now also exposes `ufsecp_gpu_frost_verify_partial_batch`
    and `ufsecp_gpu_ecrecover_batch`.
- `ufsecp_gpu_ecrecover_batch` is fully implemented on CUDA and currently returns
    `UFSECP_ERR_GPU_UNSUPPORTED` on OpenCL and Metal via documented temporary stubs.
- CPU constant-time batch signing lives in `ufsecp.h` as
    `ufsecp_ecdsa_sign_batch` and `ufsecp_schnorr_sign_batch`; these are not GPU
    APIs because private keys do not leave the host CT layer.

## What Each GPU ABI Function Must Guarantee

Each function must:
- validate all pointers and sizes
- reject zero-length invalid batch requests unless explicitly supported
- avoid hidden allocation contracts
- document input layout exactly
- document output layout exactly
- return deterministic error codes
- be backend-neutral at the API level

Each function must explicitly define:
- host input format
- output format
- batch semantics
- memory ownership
- whether in-place aliasing is allowed
- whether partial success is possible
- whether operation is public-data only or secret-bearing

## Suggested C ABI Shape

Use plain C signatures like:

```c
UFSECP_API ufsecp_error_t ufsecp_gpu_ctx_create(
    ufsecp_gpu_ctx** ctx_out,
    uint32_t device_index
);

UFSECP_API void ufsecp_gpu_ctx_destroy(ufsecp_gpu_ctx* ctx);

UFSECP_API ufsecp_error_t ufsecp_gpu_generator_mul_batch(
    ufsecp_gpu_ctx* ctx,
    const uint8_t* scalars32,
    size_t count,
    uint8_t* out_pubkeys33
);
```

Batch layout rules should be simple:
- scalars: `count * 32`
- compressed pubkeys: `count * 33`
- x-only pubkeys: `count * 32`
- hashes: `count * 32`
- boolean results: `count`

Avoid variable per-item sizes in first-wave GPU APIs.

## Context Design

`ufsecp_gpu_ctx` should own:
- selected backend
- selected device
- stream or execution queue
- cached launch config if useful
- optional scratch allocator
- last error storage

It must not expose layout publicly.

It should be possible to create one context per thread or per pipeline.

Backends to support in design:
- `CUDA`
- `OpenCL`
- `Metal`

If a backend cannot support a specific op yet, return `UFSECP_ERR_GPU_UNSUPPORTED`.

## Error Model

Reuse existing `ufsecp_error_t` where reasonable.
Add GPU-specific errors only if necessary, for example:
- `UFSECP_ERR_GPU_UNAVAILABLE`
- `UFSECP_ERR_GPU_DEVICE`
- `UFSECP_ERR_GPU_LAUNCH`
- `UFSECP_ERR_GPU_MEMORY`
- `UFSECP_ERR_GPU_UNSUPPORTED`
- `UFSECP_ERR_GPU_BACKEND`
- `UFSECP_ERR_GPU_QUEUE`

Do not overload unrelated CPU errors for GPU runtime failures.

## Testing Requirements

Add tests for the new GPU host/API layer.

Required:
- backend enumeration
- unsupported backend/op behavior
- context create/destroy
- invalid device index
- null pointer rejection
- undersized output buffer rejection
- zero/empty batch edge cases
- malformed pubkey rejection where applicable
- CPU vs GPU equivalence for each new operation
- CUDA vs OpenCL equivalence where both support the op
- CUDA vs Metal equivalence where both support the op
- multi-batch repeated execution stability
- error propagation when GPU is unavailable

Suggested new test names:
- `gpu_abi_gate`
- `gpu_ops_equivalence`
- `gpu_host_api_negative`
- `gpu_backend_matrix`

If GPU CI coverage is not available everywhere, keep tests feature-gated but runnable where hardware exists.

## Documentation Requirements

Update or add:
- `docs/API_REFERENCE.md`
- `docs/FEATURE_MATURITY.md`
- `docs/TEST_MATRIX.md`
- `docs/AUDIT_TRACEABILITY.md`
- `include/ufsecp/SUPPORTED_GUARANTEES.md`

The docs must clearly state:
- GPU ops are now available through stable C ABI, if implemented
- which operations are exposed
- backend support status
- per-backend support matrix
- whether operations are public-data only or secret-bearing
- whether the API is stable or experimental

## Rust Preparation

Do not bind raw kernels.
Do not bind host C++ directly.

Prepare for Rust by making Layer 3 clean enough that a future raw crate can expose:
- `ufsecp_gpu_ctx_create`
- `ufsecp_gpu_ctx_destroy`
- `ufsecp_gpu_generator_mul_batch`
- `ufsecp_gpu_ecdsa_verify_batch`
- `ufsecp_gpu_schnorr_verify_batch`
- `ufsecp_gpu_ecdh_batch`
- `ufsecp_gpu_hash160_pubkey_batch`
- `ufsecp_gpu_msm`
- `ufsecp_gpu_frost_verify_partial_batch`
- `ufsecp_gpu_ecrecover_batch`

If you add Rust changes, keep them limited to raw FFI declarations only.

The same ABI should also be usable from:
- Python
- Go
- C#

## Explicitly Out Of Scope For This Task

Do not implement all possible GPU features.
Do not expose:
- full protocol layer
- experimental ZK flows
- dynamic per-item mixed-size envelopes
- complex planner systems
- app-specific pipelines
- backend-specific public headers for CUDA/OpenCL/Metal end users

Focus on stable reusable batch operations only.

## Concrete Deliverables

Required deliverables:
1. New public GPU C ABI header
2. New internal host-side GPU API implementation
3. First-wave GPU ops implementation
4. Tests for context, negative cases, and CPU/GPU equivalence
5. Documentation updates
6. Assurance/test matrix updates
7. Backend capability matrix for CUDA/OpenCL/Metal

Optional deliverables:
1. Minimal Rust raw FFI extension
2. Example C usage for one GPU batch op
3. One backend-neutral sample showing runtime backend selection

## Acceptance Criteria

This task is complete only when:
- the public GPU ABI is backend-neutral
- at least the six first-wave operations exist
- there is no raw CUDA/OpenCL/Metal type leakage into the public ABI
- tests exist for correctness and negative cases
- CPU vs GPU equivalence is demonstrated for the new ops
- backend support and fallback behavior are documented
- docs reflect the new API surface
- assurance docs and test matrix are updated

## Priority Order

1. Public GPU C ABI design
2. Opaque GPU context with backend selection
3. Generator mul / verify / ECDH / hash160 / MSM ops
4. Negative tests and equivalence tests
5. CUDA/OpenCL/Metal support matrix
6. Docs and assurance updates
7. Optional Rust raw FFI prep

## Final Guidance

Do not over-design this.
The correct move is to expose a narrow, high-value GPU operations layer above the existing engine.

The engines stay internal.
The operations API becomes reusable.
The C ABI becomes the interop boundary.
CUDA/OpenCL/Metal remain implementation backends, not public contracts.

---

## BIP-352 Silent Payment GPU Batch Scan — Design Notes

### Public C ABI

```c
/* Size of the precomputed scan plan (GLV wNAF encoded, 264 bytes). */
#define UFSECP_BIP352_SCAN_PLAN_BYTES 264

/* CPU utility: precompute scan plan from raw scan private key.
 * Outputs a 264-byte struct: two 130-byte wNAF arrays + 2 flags + 2 pad bytes.
 * Safe to cache.  Does NOT contain the raw scalar. */
ufsecp_error_t ufsecp_bip352_prepare_scan_plan(
    const uint8_t scan_privkey32[32],
    uint8_t plan264_out[264]);

/* GPU batch: SECRET-BEARING.
 * For each tweak_i, computes scan_privkey × tweak_i → shared_i,
 * SHA256_tagged("BIP0352/SharedSecret", ser37(shared_i)),
 * hash_i × G + spend_pubkey, and returns upper-64-bit prefix_i. */
ufsecp_error_t ufsecp_gpu_bip352_scan_batch(
    ufsecp_gpu_ctx*  ctx,
    const uint8_t    scan_privkey32[32],
    const uint8_t    spend_pubkey33[33],
    const uint8_t*   tweak_pubkeys33,
    size_t           n_tweaks,
    uint64_t*        prefix64_out);
```

### Pipeline (per tweak)

1. `scan_privkey × tweak_i` → `shared_i` (GLV wNAF on GPU)
2. `ser37 = [prefix_byte] || shared_i.x || [0, 0, 0, 0]`  (37 bytes)
3. `hash_i = SHA256_tagged("BIP0352/SharedSecret", ser37)`  — tagged SHA256 with precomputed midstate `{0x88831537, 0x5127079b, 0x69c2137b, 0xab0303e6, 0x98fa21fa, 0x4a888523, 0xbd99daab, 0xf25e5e0a}`
4. `output_i = hash_i × G + spend_pubkey`  (hash-to-scalar + generator mul + point add)
5. `prefix64_out[i] = upper 64 bits of output_i.x`

### BIP352ScanKeyGlv struct (264 bytes)

| Offset | Size | Field | Description |
|--------|------|-------|-------------|
| 0 | 130 | `wnaf1[130]` | wNAF digits for k1 (GLV half-scalar) |
| 130 | 130 | `wnaf2[130]` | wNAF digits for k2 (GLV half-scalar) |
| 260 | 1 | `k1_neg` | 1 if k1 was negative (negate base.y) |
| 261 | 1 | `flip_phi` | `k1_neg XOR k2_neg` — flip endomorphism table y |
| 262 | 2 | `pad[2]` | zero padding |

### Security posture

`scan_privkey32` is uploaded to the GPU driver.  Callers must accept this posture.  It is equivalent
to `ufsecp_gpu_ecdh_batch`, which is the only other SECRET-BEARING GPU operation.

### Backend status

| Backend | Status |
|---------|--------|
| CUDA | Full inline kernel — no bench dependency |
| OpenCL | Full implementation — lazy-loads `secp256k1_bip352.cl`, CPU wNAF plan |
| Metal | `TODO(parity):bip352-scan-metal` stub — returns `GpuError::Unsupported` |

### Audit coverage

`audit/test_gpu_bip352_scan.cpp` — tests SW-BIP352-1 through SW-BIP352-13:
- Macro size check
- CPU plan: null-arg rejection, non-zero output
- GPU: null ctx/args rejection, zero-count no-crash, single-tweak prefix non-zero,
  batch-16 all non-zero, determinism, distinct-tweaks produce distinct prefixes

### Benchmark results (RTX 5060 Ti, 2026-04-04)

Measured via `bench_bip352` (CUDA, N=500,000, 11 passes median) and
`opencl_bip352_benchmark --batch 50000` (OpenCL, 11 passes median).
The C ABI wrapper (`ufsecp_gpu_bip352_scan_batch`) adds <2 ns virtual dispatch
overhead over the kernel measurements below.

| Backend | Mode | Time/Op | Throughput | vs CPU |
|---------|------|---------|------------|--------|
| CUDA | GLV (tpb=384) | 178.9 ns | 5.59 M/s | **136.6x** |
| CUDA | + LUT 16×64K | 90.5 ns | 11.05 M/s | **270.0x** |
| OpenCL | Fused pipeline | 197.9 ns | 5.05 M/s | 123.5x |
| OpenCL | + LUT | 96.5 ns | 10.36 M/s | 253.2x |
| CPU | KPlan (single-thread) | 24,436.5 ns | 40.9 K/s | 1.00x |

Validation passed on all runs (CPU = CUDA = OpenCL prefix checksums).
