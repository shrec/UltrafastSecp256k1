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

The in-place collect verify (`ufsecp_lbtc_verify_*_collect`) now has a dedicated
on-device kernel on **all three backends** (`ecdsa_verify_collect` /
`schnorr_verify_collect`). Each collect kernel is a verbatim clone of the same
backend's audited `*_verify_lbtc_columns` verify kernel — the verdict is
bit-for-bit identical to `*_verify_batch`; ONLY the output store changes to the
collect convention: the caller pre-seeds the 1-byte-per-row `key_buffer` non-zero,
a VALID row writes 0, an INVALID row is left seeded (rejected id survives). The
override seeds the device verdict channel from `key_buffer`, reads it back
verbatim, and returns a non-OK `GpuError` on any operational fault (engine falls
back) — never zeroing a non-valid row and never emitting an all-zero buffer.

| Backend | collect path | Assurance |
|---------|--------------|-----------|
| CPU     | reference (per-row verify) | **HIGH** — gated vs libsecp256k1 |
| CUDA    | dedicated on-device kernel | **HIGH** — `test_lbtc_consensus_diff` proves GPU==CPU==libsecp on the rejected-set (ECDSA+Schnorr, mixed corpus); kernel is a verbatim copy of the audited verify kernel with only the output store changed |
| OpenCL  | dedicated on-device kernel (`ecdsa_verify_lbtc_collect` / `schnorr_verify_lbtc_collect` in `secp256k1_extended.cl`) | **HIGH** — verified on-device end-to-end (NVIDIA RTX 5060 Ti, OPENCL=ON/CUDA=OFF): the wired `test_gpu_collect_verify_parity` audit test runs through the C ABI → OpenCL override → kernel and asserts `collect == ufsecp_gpu_*_verify_batch` verdict per-row on an all-valid + tampered corpus → **pass=24 fail=0**. ECDSA collect parses the **compact (big-endian r‖s)** sig via `lbtc_parse_compact_signature` — the SAME format as `ecdsa_verify_batch` (`ecdsa_verify_compressed`), not the little-endian columns/opaque parse |
| Metal   | dedicated on-device kernel (`lbtc_ecdsa_verify_collect` / `lbtc_schnorr_verify_collect` in `secp256k1_kernels.metal`) | **MEDIUM — code-complete, runtime parity PENDING Apple-hardware validation** — ECDSA collect mirrors `ecdsa_verify_batch_compressed` (compact big-endian sig parse + `ecdsa_verify(Scalar256,AffinePoint,r,s)`); Schnorr collect clones the Metal `schnorr_verify_lbtc_columns` verify core. Only the output store changes to the collect convention. Not built/run here (Metal is Apple-only); owner validates on Apple GPU (same `test_gpu_collect_verify_parity`) before promotion to HIGH |

Cross-backend parity is proven by the wired advisory audit test
`test_gpu_collect_verify_parity` (`gpu_collect_verify_parity` module): it asserts
the native collect verdict equals the `*_verify_batch` verdict per-row on-device,
and self-skips the device portion when no GPU is present (its null-ctx contract
portion runs everywhere).

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

### libbitcoin ECDSA/Schnorr column verify (Added 2026-07-01)

The single public libbitcoin-direct verify surface for the column (Structure-of-
Arrays) layout is the pair of engine entrypoints
`secp256k1::ecdsa_batch_verify_opaque_columns(digests32, pubkeys33, sigs64, count,
out_results)` and `secp256k1::schnorr_batch_verify_bip340_columns(digests32,
xonly32, sigs64, count, out_results)` (the `ufsecp::lbtc::*` header calls exactly
these). There is **no caller-visible `_gpu` variant on this surface and no
recoverable GPU status** — the caller sees one API with a boolean-plus-per-row
result and never chunks, never selects CPU vs GPU, and never manages GPU buffers.

The columns consumed are libbitcoin's parallel spans:
`digests32[count]`, `pubkeys[count]` (33 compressed for ECDSA / 32 x-only for
Schnorr), `sigs64[count]` (ECDSA = opaque LE `secp256k1_ecdsa_signature`; Schnorr
= BIP-340 `R.x||s`). Verification is **variable-time over PUBLIC data** (pubkey,
signature, message) — correct by design, not a side-channel gap.

**GPU is an internal accelerator, selected inside the engine.** A GPU provider is
attached by installing a process-wide hook
`secp256k1::install_gpu_columns_verify_hook(...)` (thread-safe, returns the
previous hook; installed by the libbitcoin-GPU build). When a hook is installed,
each column entrypoint attempts the GPU column verify first; when no hook is
installed, or the hook **declines** (GPU not compiled / no device / Unsupported /
operational backend error), the engine transparently completes the batch on the
deterministic CPU column path with **identical boolean and per-row semantics**.
Chunking and reusable device staging are engine/backend-owned, never caller-owned.

**Failure policy — fatal, not invalid.** An operational GPU/backend error is
converted to a CPU fallback via the hook's decline; it is **never** represented as
a consensus-invalid row or an all-zero OK result. Only an unrecoverable inability
to complete the required math (with no fallback possible) fails hard — that is a
fatal engine failure, not consensus data. A malformed layout (null column
pointers or size overflow) remains fail-closed: the entrypoint returns `false` and
zeroes `out_results` — this is input validation, not a GPU error.

The backend column methods `ecdsa_verify_lbtc_columns` / `schnorr_verify_lbtc_columns`
exist **natively on CUDA, OpenCL, and Metal**. They parse the ECDSA opaque
little-endian r/s and Schnorr BIP-340 signatures **device-side** (no host
compact-signature staging) and decompress the 33-byte pubkeys on device; per-row
offsets are 64-bit and results are `uint8_t` 1=valid/0=invalid. A backend that
returns `Unsupported`/non-OK is completed by the engine CPU fallback — that is
correctness-preserving, not a success gap. The native column method and the
always-present CPU fallback are both true simultaneously: the kernels exist on all
three backends, and the engine always has a deterministic CPU path to fall back
to; there is no "kernels not yet ready" state on this surface.

| Backend | column path | Assurance | Notes |
|---|---|---|---|
| CPU | reference (deterministic fallback) | **HIGH** — gated vs libsecp256k1 | public-data variable-time; byte-identical fallback for every backend |
| CUDA | native on-device kernel | **HIGH** — `lbtc_gpu_columns_diff` proves GPU==CPU on valid/tampered/malformed + forced small chunks | public-data variable-time |
| OpenCL | native on-device kernel | **HIGH** — same kernel device-parse; differential-tested when an OpenCL device is present | public-data variable-time |
| Metal | native on-device kernel | **MEDIUM** — compiles/validates on macOS CI; Linux host cannot build Metal | public-data variable-time; fatal-not-invalid dispatch guard (below) |

**Engine-owned reusable staging (acceptance A6).** Chunk sizing is engine-owned and
memory-aware on every backend (`lbtc_columns_chunk`); the caller never chunks or sees
a chunk parameter. Device/host staging is grow-only and **reused across successive
engine calls**, so a hot libbitcoin verify loop performs no fresh per-call allocation:

- **CUDA** — a thread-local `CudaBatchScratch` keeps one packed device buffer
  (`dig | key | sig`) plus the results buffer and host readback vector, grown on
  demand and freed only at thread exit. Validated on RTX 5060 Ti including forced
  1-row chunks (`UFSECP_GPU_COLUMNS_CHUNK=1`, byte-identical to CPU).
- **OpenCL** — a thread-local `OclColumnsPool` (mirroring the existing MSM pool) keeps
  the four column buffers persistent across calls and chunks; each chunk re-uploads
  exactly its rows via `clEnqueueWriteBuffer` on the in-order queue, so no stale data
  from a larger prior batch is read. Validated on the NVIDIA OpenCL runtime including
  forced small chunks.
- **Metal** — the column methods allocate `MTLResourceStorageModeShared` buffers per
  call, consistent with every other Metal op (the Metal backend has no persistent
  device-buffer pool). On Apple Silicon these are cheap unified-memory allocations; a
  persistent Metal pool is a tracked future optimization, not a correctness gap.

**Fatal-not-invalid on Metal dispatch failure (acceptance A5/A9).**
`MetalRuntime::dispatch_sync` is `void` and only logs a command-buffer error, so a GPU
execution failure (watchdog / device-lost / mid-run fault) would leave the shared
result buffer unwritten and be misread as "every row invalid" — an operational failure
masquerading as a consensus verdict. The Metal column methods now seed every result
slot with a sentinel the kernel never writes (it writes only 0/1 per row) and **decline
with a non-OK `GpuError` if any sentinel survives the dispatch**, so the engine hook
falls back to CPU instead of emitting invalid rows. CUDA and OpenCL already detect this
via `cudaDeviceSynchronize` / `clFinish` return codes. The deeper fix — returning a
status from `dispatch_sync` in `src/metal/src/metal_runtime.mm` — is tracked for the
runtime owner, as that file is outside this card's write scope.

Caller contract: **no bridge, no caller-visible GPU API, engine-owned chunking.**
The default direct build is **pure CPU** — no compile-time macro gates the GPU
provider. The engine (`fastsecp256k1`) owns an installable `GpuColumnsVerifyHook`
(see `secp256k1/batch_verify.hpp`) and carries no `gpu::` dependency, because
`secp256k1_gpu_host` already PRIVATE-links the engine and a reverse link would be a
static-library cycle. The default provider that bridges that hook to a real backend
lives in the GPU-host layer (`src/gpu/src/gpu_engine_hook.cpp`) and **self-installs
via a static initializer whenever the GPU host is linked**. Enablement is therefore
the inspectable build fact "is the provider translation unit linked," not a macro:

- GPU host not built (default direct build): the provider is never linked, the hook
  stays null, and the engine surface runs the pure-CPU column path.
- GPU host built: the provider self-installs and the engine surface
  (`ecdsa_batch_verify_opaque_columns` / `schnorr_batch_verify_bip340_columns`)
  transparently accelerates on the GPU, declining to CPU on any operational error
  (fatal-not-invalid: an operational decline never yields invalid rows).

The audit build proves this end-to-end on a GPU-less runner: `lbtc_gpu_columns_diff`
asserts the provider self-installed (`install_gpu_columns_verify_hook(nullptr)`
returns non-null) before exercising the decline → CPU-fallback contract. Residual
finalization owned by the companion `compat/libbitcoin_direct` entrypoint card: that
consumer must link `secp256k1_gpu_host` and retain the self-registering provider
object (a targeted `--undefined` anchor is used, not a blanket `WHOLE_ARCHIVE`, to
pull only the provider) to reach the GPU from the bridge-free direct path. That
retention no longer breaks the ZK-less link: the Schnorr SNARK-witness host fallback
that once dragged an undefined `secp256k1::zk::` reference into the minimal engine is
now `#if SECP256K1_HAS_ZK`-gated in `gpu_backend.hpp` and its TU dropped from
`secp256k1_gpu_host` when `SECP256K1_BUILD_ZK=OFF` (LBTC-GPU-DIRECT-ZK-BLOCKER,
resolved). Verified on RTX 5060 Ti: the `LIBBITCOIN+LIBBITCOIN_GPU+CUDA` direct-GPU
build links and `nm` shows the provider anchor + installer + hook present with the
SNARK fallback and `zk::` symbols absent. Do not claim the default direct build is
GPU-accelerated; by default it is pure CPU. The C ABI
`ufsecp_gpu_ecdsa_verify_lbtc_columns` / `ufsecp_gpu_schnorr_verify_lbtc_columns`
exist only for C ABI completeness and are not the libbitcoin-direct surface.

### libbitcoin public-data batch ops: validate / commitment / hashing (Added 2026-07-04)

Six header-only `ufsecp::lbtc::*` batch primitives — `xonly_validate_batch`,
`pubkey_validate_batch`, `taproot_commitment_verify_batch`, `tagged_hash_batch`,
`tagged_hash_var_batch`, `hash256_batch` — share the identical one-surface
contract as the column-verify path above: internal GPU acceleration via the
**EXISTING** `GpuBackend` virtuals (`xonly_validate`, `pubkey_validate`,
`commitment_verify`, `tagged_hash`, `tagged_hash_var`, `hash256`) + a
deterministic CPU fallback, presented as ONE `bool`-returning inline call (no
CPU/GPU split, no GPU status code, no caller chunking, no bridge/C-ABI). All six
are PUBLIC-DATA / variable-time (no secret is touched). The offload is wired
through engine-owned `inline std::atomic<>` hooks
(`compat/libbitcoin_direct/include/ufsecp/lbtc_gpu_ops.hpp`) that
`src/gpu/src/gpu_engine_hook.cpp` self-installs (guarded by
`SECP256K1_LBTC_GPU_OPS`, retained by the same
`secp256k1_gpu_columns_provider_anchor`). No new virtual, no new C ABI, no new
anchor, no new CTest target.

| Backend | validate/commitment/hash path | Assurance | Notes |
|---|---|---|---|
| CPU | reference (deterministic fallback) | **HIGH** — `lbtc_direct_verify` cross-checks each op vs a serial reference (schnorr `lift_x`, `P+t*G`, SHA256, double-SHA256) + hostile-caller matrix | public-data variable-time; byte-identical fallback for every backend |
| CUDA | native on-device kernel | **HIGH** — the six virtuals are CUDA-native (`lbtc_*_kernel` in `gpu_backend_cuda.cu`); operational error declines → CPU | public-data variable-time; hash caps (msg_len≤256, stride≤256, input_len≤320) decline out-of-cap lengths → CPU covers all |
| OpenCL | native on-device kernel | **HIGH** — the six virtuals are OpenCL-native (`lbtc_*` kernels in `src/opencl/kernels/secp256k1_extended.cl` + overrides in `gpu_backend_opencl.cpp`); operational error declines → CPU. Verified on-device (NVIDIA RTX 5060 Ti, OpenCL via NVIDIA CUDA platform): all 6 kernels bit-exact vs a Python `hashlib`+EC reference, and `lbtc_direct_verify` passes end-to-end with OpenCL as the active backend (`-DSECP256K1_BUILD_OPENCL=ON -DSECP256K1_BUILD_CUDA=OFF`) | public-data variable-time; same hash caps as CUDA (msg_len≤256, stride≤256, input_len≤320) decline out-of-cap → CPU. `commitment_verify` uses `shamir_double_mul_glv_impl` (the proven ecdsa-verify path) for `tweak*G + P` |
| Metal | native on-device kernel | **MEDIUM — code-complete, runtime parity PENDING Apple-hardware validation** — the six virtuals are Metal-native (`lbtc_*` kernels in `src/metal/shaders/secp256k1_kernels.metal` + overrides in `gpu_backend_metal.mm`), mirroring the CUDA/OpenCL reference; `commitment_verify` uses the proven `generator_affine`+`scalar_mul_glv`+`jacobian_add` path (same as Metal `ecdsa_verify`). NOT built/run here: Metal compiles only on Apple (`src/metal/CMakeLists.txt` returns on non-Apple), so this Linux host cannot execute it | operational error declines → CPU. No measured Metal numbers; owner validates on Apple GPU before this row is promoted to HIGH |

Fail-closed / never-consensus-invalid: **validate ops** deterministically
overwrite every result row on the CPU path (operational GPU failure never yields
an all-zero buffer); **hash ops** never pre-zero `out32` (a zero row would be a
wrong/consensus-invalid hash) and reject bad input (`false`) without touching
`out32`. Every trampoline maps no-GPU / non-Ok `GpuError` / exception to a
decline (`-1`), so control always falls to the correct CPU computation. A
CPU-only libbitcoin build never links `secp256k1_gpu_host`, so the hooks stay
null and every call runs the deterministic CPU path — no GPU symbol is required.

### `hash256_var`: GPU batch variable-length double-SHA256 (Added 2026-07-06)

A new `GpuBackend::hash256_var` virtual (`src/gpu/include/gpu_backend.hpp`)
extends the libbitcoin public-data batch-op surface above with a
**variable-length** Bitcoin HASH256 primitive: row `i` is
`inputs[i*stride .. i*stride+input_lens[i])`, no tag prefix, no transaction
parsing on GPU — this is the primitive a future libbitcoin
`txid_batch`/`wtxid_batch` convenience wrapper will compose with CPU-side
BIP141 serialization (`legacy_serialize`/`witness_serialize` in
`src/cpu/src/bip144.cpp`). Exposed via C ABI `ufsecp_gpu_hash256_var`
(`include/ufsecp/ufsecp_gpu.h` / `src/cpu/src/ufsecp_gpu_impl.cpp`) and the
bridge-free `ufsecp::lbtc::hash256_var_batch` direct wrapper
(`compat/libbitcoin_direct/include/ufsecp/libbitcoin.hpp`) with its own
hook/installer in `lbtc_gpu_ops.hpp`. PUBLIC-DATA / variable-time — not
secret-bearing, no CT requirement.

| Backend | `hash256_var` path | Assurance | Notes |
|---|---|---|---|
| CPU | reference (deterministic fallback) | **HIGH** — `hash256_var_batch` per-row-validates `input_lens[i]` against `stride` before dispatch and falls back to `secp256k1::SHA256::hash256` per row; covered by `test_regression_hash256_var_batch` (KAT/boundary differential) | public-data variable-time; byte-identical fallback for every backend; `out32` never touched on a rejected call |
| CUDA | native on-device kernel | **HIGH** — `hash256_var` virtual is CUDA-native (`lbtc_hash256_var_kernel` in `gpu_backend_cuda.cu`); operational error declines → CPU | public-data variable-time; streams each row directly in 64-byte SHA-256 compression blocks (no full-row local copy), so rows up to `kMaxHash256VarStride` (4 MiB) are supported — unlike `tagged_hash_var`/`hash256`, which copy each row into a small fixed on-chip buffer (256–320 bytes) to prepend the BIP-340 tag and are capped accordingly |
| OpenCL | native on-device kernel | **HIGH** — `hash256_var` virtual is OpenCL-native (`lbtc_hash256_var` kernel in `src/opencl/kernels/secp256k1_extended.cl` + override in `gpu_backend_opencl.cpp`); operational error declines → CPU | public-data variable-time; same streaming design as CUDA (`sha256_update_global` reads each row directly from global memory, no full-row local copy); same 4 MiB `kMaxHash256VarStride` bound |
| Metal | native on-device kernel | **MEDIUM — code-complete, runtime parity PENDING Apple-hardware validation** — `hash256_var` virtual is Metal-native (`lbtc_hash256_var` kernel in `src/metal/shaders/secp256k1_kernels.metal` + override in `gpu_backend_metal.mm`), mirroring the CUDA/OpenCL streaming design (`sha256_update_device`); NOT built/run here: Metal compiles only on Apple, so this Linux host cannot execute it — same status already documented above for the sibling `xonly_validate`/`pubkey_validate`/`commitment_verify`/`tagged_hash`/`tagged_hash_var`/`hash256` ops | operational error declines → CPU. No measured Metal numbers; owner validates on Apple GPU before this row is promoted to HIGH |

Fail-closed: `ufsecp_gpu_hash256_var` / `hash256_var_batch` never pre-zero
`out32` and reject bad input without touching it — `ctx==nullptr` /
null `inputs`/`input_lens`/`out32` with `n>0` → `UFSECP_ERR_NULL_ARG`;
`n==0` → `UFSECP_OK` no-op; `n` over the existing `kMaxGpuBatchN` (64M) batch
cap, `stride==0`/`>kMaxHash256VarStride`, or any individual
`input_lens[i]==0`/`>stride` → `UFSECP_ERR_BAD_INPUT`. Unlike the older
`ufsecp_gpu_tagged_hash_var` ABI wrapper (which only bounds-checks the scalar
`stride`), the `hash256_var` wrapper validates every row's length
individually before GPU dispatch, since row lengths here are fully
caller-controlled. Test coverage: `audit/test_regression_hash256_var_batch.cpp`
(KAT/boundary), `audit/test_regression_hash256_var_parity.cpp` (cross-backend
byte-identical parity), `audit/test_exploit_hash256_var_bounds.cpp` (hostile
inputs).

### ECDSA compact signature staging (Updated 2026-06-18)

`ufsecp_gpu_ecdsa_verify_batch` accepts public compact `r||s` signatures. CUDA
and OpenCL now upload those 64-byte rows directly and parse them inside the
verify kernel, matching the existing Metal path. This keeps CPU-side memory
traffic in the caller/public format and moves the cheap byte-to-scalar transform
to backend registers. The public ABI and consensus-bearing result convention are
unchanged.

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

### Public GPU ABI operations (18 functions, backend-neutral)

| Function | CPU (fast) | CPU (CT) | CUDA | OpenCL | Metal |
|---|---|---|---|---|---|
| `ufsecp_gpu_generator_mul_batch` (k·G) | Y | - | Y | Y | Y |
| `ufsecp_gpu_ecdsa_verify_batch` | Y | - | Y | Y | Y |
| `ufsecp_gpu_ecdsa_verify_opaque_rows` / `ufsecp_gpu_ecdsa_verify_lbtc_rows` alias | Y | - | Y | Y | Y |
| `ufsecp_gpu_schnorr_verify_batch` | Y | - | Y | Y | Y |
| `ufsecp_gpu_ecdh_batch` ¹ | Y | Y | Y | Y | Y |
| `ufsecp_gpu_hash160_pubkey_batch` | Y | - | Y | Y | Y |
| `ufsecp_gpu_hash256_var` | Y | - | Y | Y | Y |
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

---

## Resolved Secondary-Invariant Constant Bugs

### GLV β (beta) endomorphism constant — Metal + OpenCL (fixed 2026-07-08)

**Bug:** Both Metal (`secp256k1_ct_point.h`) and OpenCL (`secp256k1_ct_point.cl`)
independently defined local duplicate copies of the secp256k1 GLV β constant
inside `ct_scalar_mul_point()` that diverged from the canonical value used by
their respective `apply_endomorphism_impl` implementations.

- **Metal:** Local `BETA_METAL[8]` (32-bit limbs) had a dropped hex nibble in
  word[0] during manual 64→32-bit transcription. Now references canonical
  `BETA_LIMBS[8]` from `secp256k1_point.h:412`.
- **OpenCL:** Local inline 64-bit literals matched canonical at limb[0] but
  diverged at limbs[1..3]. Now references canonical `GLV_BETA0..3` from
  `secp256k1_extended.cl:51-54`.
- **CUDA:** Not affected — single `BETA[4]` in `secp256k1.cuh` reused by both
  VT and CT paths.
- **Severity:** P0 correctness bug, fail-loud (broken proofs fail verification).
  Affected: CT ZK prove paths (knowledge-of-DL, DLEQ, bulletproof range proof)
  on Metal and OpenCL. Unaffected: ECDSA/Schnorr signing, ECDH, CUDA all paths.
- **Test:** `audit/test_regression_gpu_beta_constants.cpp` (math_invariants,
  advisory=false) — verifies canonical β value, pre-fix divergence, and scans
  actual shader source files for regression.
- **Metal runtime caveat:** Fixed code is code-complete but NOT runtime-verified
  on this Linux host (Metal requires Apple hardware). Owner validation on real
  Apple hardware is needed before promotion to HIGH assurance.

## Parity Status

> **Parity tracking is machine-generated.** The source graph (`tools/source_graph_kit/source_graph.py`)
> cross-references the `GpuBackend` virtual interface against CUDA, OpenCL, and Metal
> implementations on every CI build. Any gap introduced by a commit is flagged
> immediately by the parity audit workflow. The numbers below reflect the current
> HEAD — they are not a manually maintained snapshot.

18 of the 18 public GPU ABI operations are implemented natively on CUDA, OpenCL,
and Metal. Last resolved: 2026-07-07 (`ufsecp_gpu_hash160_pubkey_batch` — native
OpenCL kernel dispatch via `hash160_batch` in `secp256k1_hash160.cl`). No known
exceptions remain; see "GPU backend native-parity gate" immediately below.

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

### Default-stub parity exceptions (resolved for the six libbitcoin-bridge public-data ops)

The `xonly_validate`, `commitment_verify`, `tagged_hash`, `tagged_hash_var`,
`pubkey_validate`, and `hash256` **libbitcoin-bridge specialization** virtual
methods are **not** a parity exception: as of the libbitcoin public-data batch
ops work (2026-07-04), all three shipped backends (CUDA, OpenCL, Metal)
provide dedicated native overrides for all six — the base `GpuBackend` virtual
default (`return GpuError::Unsupported`) is unreachable for any currently
shipped backend and exists only as an abstract-safe fallback for a
hypothetical future backend, exactly like `ecdsa_verify_collect` /
`schnorr_verify_collect` below. See "libbitcoin public-data batch ops:
validate / commitment / hashing" above (~line 195) for the authoritative
per-backend assurance status.

`ecdsa_verify_collect` / `schnorr_verify_collect` are resolved the same way:
as of the libbitcoin "collect" verify work (2026-06-02), all three shipped
backends (CUDA, OpenCL, Metal) provide dedicated native overrides — the base
`GpuBackend` virtual is unreachable for any currently shipped backend and
exists only as an abstract-safe default for a hypothetical future backend.
See the "libbitcoin bridge 'collect' verify" section above (~line 37) for the
authoritative per-backend assurance status.

Any GPU-gated or ZK-gated virtual that still has a genuine, currently-true
default-stub gap carries an inline `PARITY-EXCEPTION` or `TODO(parity)` marker
in `src/gpu/include/gpu_backend.hpp`, and the `audit_gate.py --gpu-parity` gate
enforces that **every `return ...Unsupported` in GPU backend source is either
implemented or carries a `TODO(parity)`/`PARITY-EXCEPTION` marker, OR is one of
the abstract-safe-only base defaults documented above** — a backend may not
silently return Unsupported without a documented exception. The gate scans
source files only (build/generated trees are pruned) so the signal is
precise.

### GPU backend native-parity gate (added 2026-07-07)

`ci/check_gpu_backend_parity.py` mechanically enforces the owner rule that any
`GpuBackend` virtual operation implemented natively on one shipping backend
must be native on CUDA, OpenCL, AND Metal, plus have required C ABI exposure.

Unlike `audit_gate.py --gpu-parity` above — which only checks that a
`return ...Unsupported` site has a `PARITY-EXCEPTION`/`TODO(parity)` comment
somewhere nearby, trivially satisfiable by adding a comment, and blind to an
override that exists but merely delegates to a CPU loop instead of
dispatching a GPU kernel — this gate independently inspects each backend's
override body for real device-dispatch evidence (CUDA `<<<...>>>` kernel
launch, OpenCL `clEnqueueNDRangeKernel`/context-wrapper dispatch, Metal
`dispatch_sync`) and never trusts an inline source comment as proof. A gap
only passes the gate if the `(operation, backend)` pair is listed in the
"Permanent Architecture Exceptions" table above — a doc that goes through
normal PR review, not a code comment any single commit can add unilaterally.
It also cross-checks the Feature Matrix above so this document cannot claim
`Y` (native) for a cell the code does not actually provide.

As of 2026-07-07, the gate passes with zero violations across all 37 enumerated
`GpuBackend` operations. The one pre-existing gap the gate caught at
introduction time — `hash160_pubkey_batch` having no native OpenCL kernel
dispatch — is now resolved: `GpuBackendOpenCL::hash160_pubkey_batch` dispatches
the existing `hash160_batch` kernel in `secp256k1_hash160.cl` via
`clEnqueueNDRangeKernel`, lazily loaded/compiled through `ensure_hash160_kernel()`
the same way `frost_verify_partial_batch` loads `secp256k1_frost.cl`.

Run: `python3 ci/check_gpu_backend_parity.py` (add `--json` for machine
output or `--list` to see every operation's per-backend classification).

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
