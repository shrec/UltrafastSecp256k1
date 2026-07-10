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

### `txid_hash_batch` / `wtxid_hash_batch` / `merkle_pair_hash_batch`: libbitcoin txid, wtxid, and Merkle-pair hashing (Added 2026-07-08)

Three bridge-free `ufsecp::lbtc::*_batch` wrappers
(`compat/libbitcoin_direct/include/ufsecp/libbitcoin.hpp`) extend the
`hash256_var` primitive above with libbitcoin-facing semantics.
`txid_hash_batch` and `wtxid_hash_batch` are semantic aliases over
`hash256_var_batch` — zero new backend work, same GPU hook/kernel/bound
(`kMaxHash256VarStride`, 4 MiB) as `hash256_var`. `merkle_pair_hash_batch` is
a distinct primitive: a fixed 64-byte `left32 || right32` double-SHA256 over
two 32-byte column spans (Structure-of-Arrays), backed by its own
`GpuBackend::merkle_pair_hash` virtual, C ABI `ufsecp_gpu_merkle_pair_hash`
(`include/ufsecp/ufsecp_gpu.h` / `src/cpu/src/ufsecp_gpu_impl.cpp`), and its
own hook/installer in `lbtc_gpu_ops.hpp`. All three are PUBLIC-DATA /
variable-time — no secret material, no CT requirement.

| Backend | `txid_hash_batch` / `wtxid_hash_batch` | `merkle_pair_hash_batch` | Assurance | Notes |
|---|---|---|---|---|
| CPU | reference (deterministic fallback via `hash256_var_batch`) | reference (deterministic fallback: `SHA256::hash256(left32\|\|right32)` per row) | **HIGH** — covered by `test_regression_merkle_pair_hash.cpp` (differential vs oracle, `n==0`, `left==right` odd-leaf) and `compat/libbitcoin_direct/tests/test_direct_verify.cpp` (byte-identical vs `hash256_var_batch` / HASH256 oracle, count==0, null-each-arg, overflow rejection, hook-decline, hook-success sentinel, left/right non-commutativity) | public-data variable-time; byte-identical fallback for every backend; output never touched on a rejected call |
| CUDA | native (`hash256_var` kernel, no new work) | native on-device kernel — `lbtc_merkle_pair_kernel` in `gpu_backend_cuda.cu` | **HIGH** — operational error declines → CPU | public-data variable-time |
| OpenCL | native (`hash256_var` kernel, no new work) | native on-device kernel — `lbtc_merkle_pair` in `src/opencl/kernels/secp256k1_extended.cl` + override in `gpu_backend_opencl.cpp` | **HIGH** — operational error declines → CPU | public-data variable-time |
| Metal | native (`hash256_var` kernel, no new work) | native on-device kernel — `lbtc_merkle_pair` in `src/metal/shaders/secp256k1_kernels.metal` + override in `gpu_backend_metal.mm` | **MEDIUM — code-complete, runtime parity PENDING Apple-hardware validation** — NOT built/run here: Metal compiles only on Apple, so this Linux host cannot execute it — same status already documented above for the sibling `hash256_var`/public-data batch ops | operational error declines → CPU. No measured Metal numbers; owner validates on Apple GPU before this row is promoted to HIGH |

Fail-closed: none of the three pre-zero their output, and all reject bad
input without touching it. `txid_hash_batch` / `wtxid_hash_batch` inherit
`hash256_var_batch`'s validation (`count==0` no-op; null column, `stride==0`,
per-row length out of range, or layout overflow → `false`, output untouched).
`merkle_pair_hash_batch` (C++): `count==0` no-op; null `left32`/`right32`/
`out32` or `count*32` layout overflow → `false`, output untouched.
`ufsecp_gpu_merkle_pair_hash` (C ABI): `ctx==nullptr` → `UFSECP_ERR_NULL_ARG`;
`n==0` → `UFSECP_OK` no-op; `n` over `kMaxGpuBatchN` (64M) →
`UFSECP_ERR_BAD_INPUT`; null `left32`/`right32`/`out32` with `n>0` →
`UFSECP_ERR_NULL_ARG`; non-OK leaves `out32` cleared. Test coverage:
`audit/test_regression_merkle_pair_hash.cpp` (differential/structural KAT),
`audit/test_exploit_merkle_pair_bounds.cpp` (hostile-caller bounds),
`compat/libbitcoin_direct/bench/bench_public_ops.cpp` (`txid_hash`,
`wtxid_hash`, `merkle_pair_hash` rows — sanity-checked non-zero timing on this
machine; not yet a controlled ≥5-run benchmark per this repo's performance
protocol, so no ns/op numbers are quoted here).

### `merkle_level_reduce_batch` / `merkle_root_from_leaves`: direct C++ libbitcoin merkle workloads (Added 2026-07-08)

Two bridge-free `ufsecp::lbtc::*` functions (`compat/libbitcoin_direct/include/ufsecp/libbitcoin.hpp`)
that compose Bitcoin merkle-tree construction entirely over the already-shipped
`merkle_pair_hash_batch`.  **ZERO new GpuBackend virtuals, CUDA/OpenCL/Metal
kernels, C ABI functions, or production hooks.**  These are pure direct C++
libbitcoin workloads — not new backend primitives.

`merkle_level_reduce_batch` is a semantic alias (one-line delegate to
`merkle_pair_hash_batch`).  `merkle_root_from_leaves` uses caller-provided
scratch (≥ `leaf_count × 64` bytes, no heap allocation) to iteratively reduce
tree levels via `merkle_pair_hash_batch`, following Bitcoin merkle semantics
(odd-level last-hash duplication, HASH256 with `left32 ‖ right32` byte order).

| Backend | `merkle_level_reduce_batch` / `merkle_root_from_leaves` | Assurance | Notes |
|---|---|---|---|
| CPU | reference (deterministic — composes `merkle_pair_hash_batch` CPU fallback exclusively) | **HIGH** — covered by `compat/libbitcoin_direct/tests/test_direct_operations.cpp` (0/1/2/3/7 leaves, KAT vs independent HASH256 oracle, duplicate-last semantics, multi-level root, null args, scratch undersize, overflow guard, hook-decline inherited, hook-sentinel inherited) | public-data variable-time; zero allocation; caller-provided scratch; all size multiplications overflow-checked |
| CUDA | inherited from `merkle_pair_hash_batch` (no new kernel) | **HIGH** — operational error declines → CPU | no new CUDA code; the existing `lbtc_merkle_pair_kernel` in `gpu_backend_cuda.cu` is invoked indirectly through `merkle_pair_hash_batch` |
| OpenCL | inherited from `merkle_pair_hash_batch` (no new kernel) | **HIGH** — operational error declines → CPU | no new OpenCL code; the existing `lbtc_merkle_pair` in `src/opencl/kernels/secp256k1_extended.cl` is invoked indirectly through `merkle_pair_hash_batch` |
| Metal | inherited from `merkle_pair_hash_batch` (no new kernel) | **MEDIUM — code-complete, runtime parity PENDING Apple-hardware validation** (same status as `merkle_pair_hash_batch`) | no new Metal code; inherits the existing `lbtc_merkle_pair` kernel status |

Fail-closed: `merkle_root_from_leaves` zeroes non-null `out_root32` on every
failure path (count==0, null inputs, overflow, undersize scratch, internal hash
failure); `out_root32 == nullptr` returns `false` before any write.
`leaf_count==1` copies the single leaf as root (Bitcoin semantics).
`merkle_level_reduce_batch` inherits `merkle_pair_hash_batch`'s fail-closed
contract identically.  Test coverage:
`compat/libbitcoin_direct/tests/test_direct_operations.cpp` (structural KAT +
boundary + hook-inheritance tests described above).

### `sighash_descriptor_hash`: GPU descriptor-shaped Bitcoin sighash preimage hashing (Added 2026-07-10)

A new `GpuBackend::sighash_descriptor_hash` virtual (`src/gpu/include/gpu_backend.hpp`)
computes `HASH256` of a Bitcoin sighash preimage assembled per a compact
descriptor bytecode (see `compat/libbitcoin_direct/include/ufsecp/libbitcoin.hpp`,
`sighash_descriptor_hash_batch`), without ever materializing a full per-row
preimage buffer on host or device — each referenced field is streamed
directly into a running SHA-256 context. Landed CUDA-first (task
`lbtc-sighash-gpu-core-cuda-deepseek`); this entry adds native OpenCL and
Metal parity (task `lbtc-sighash-gpu-opencl-metal-evidence-claude`).

OpenCL 1.2 (and this codebase's Metal usage) cannot bind an array of
`__global`/`device` buffer pointers as a single kernel argument the way CUDA's
device pointer array does, so both new backends use a different, but
functionally equivalent, design: referenced field columns are gathered into
ONE packed device buffer (columns copied verbatim, never a row-assembled
preimage) plus four small per-field metadata arrays
(`col_offsets`/`strides`/`fixed_lens`/`varlen_offsets`); the kernel streams
each field from that packed buffer through one running SHA-256 context
(`sha256_update_global` on OpenCL, `sha256_update_device` on Metal — the
same O(1)-local-memory primitives already used by `hash256_var`).

**Repair round (2026-07-10, same day):** Codex review of the round above found
two real defects — (1) OpenCL combined `CL_MEM_COPY_HOST_PTR` with a null host
pointer whenever a descriptor had zero variable-length fields (e.g. an
all-fixed legacy sighash — the `test_kat_legacy_all_fixed` shape), which is
invalid per the OpenCL spec; (2) both backends allocated fresh
`clCreateBuffer`/`alloc_buffer_shared` device buffers on every call instead of
reusing them. Both were fixed: OpenCL now uses a thread-local `OclSighashPool`
(mirroring the pre-existing `OclColumnsPool` used by
`ecdsa_verify_lbtc_columns`) and Metal now uses a member `MetalSighashPool`
(mirroring the pre-existing `MetalMsmPool` used by `msm()`) — both grow-only,
reused across calls, with the four metadata buffers sized once at the
compile-time `MAX_FIELDS` (64) bound and the three bulk buffers (packed
columns, packed var-lens, output) growing independently by byte capacity.
OpenCL additionally stopped host-packing referenced columns into an
intermediate `std::vector` first, instead writing each column directly from
the caller's own `field_data`/`field_var_lens` pointers into its computed
offset in the persistent device buffer via `clEnqueueWriteBuffer`; every
OpenCL call's return code (`clEnqueueWriteBuffer`/`clSetKernelArg`/
`clEnqueueNDRangeKernel`/`clFinish`/`clEnqueueReadBuffer`) is now checked, and
explicit `uint64_t` overflow guards precede every allocation-feeding
multiplication (`count*stride`, the running packed-column-bytes accumulator,
`count*32` output size) since this backend is directly reachable from the C
ABI, not just the pre-validated C++ direct-API caller. Both backends also now
independently reject Taproot-only field IDs `0x0C`-`0x0F` (annex,
tapleaf_hash, key_version, codesep_pos) — this op computes legacy/BIP143-style
`HASH256` only, never BIP-341 TapSighash. Metal's fix is code-reviewed only
(no Apple hardware on this host); OpenCL's fix is verified on real hardware
(see the OpenCL row below).

The standalone audit regression/exploit binaries
(`test_regression_sighash_descriptor_gpu_standalone`,
`test_exploit_sighash_descriptor_malformed_standalone`) previously
self-skipped 100% of their on-device checks even on a GPU-enabled build with a
real device present, because they raw-compile `src/gpu/src/gpu_registry.cpp`
directly rather than linking the fully-configured `secp256k1_gpu_host`
library, and nothing defined `SECP256K1_HAVE_OPENCL`/`_CUDA`/`_METAL` or added
the backend source files for that raw compile — so `create_backend()` always
returned null and every "on-device" check silently reported "no GPU
available." This affected every GPU-dependent differential audit module built
this way, not just sighash (e.g. the pre-existing `merkle_pair_hash`/
`hash256_var` regression modules had the same gap). Fixed in
`audit/CMakeLists.txt` with a new `audit_wire_real_gpu_backends()` helper that
mirrors `src/gpu/CMakeLists.txt`'s own backend-source/define/link
accumulation for every affected standalone target plus `unified_audit_runner`
itself, gated the same way (`if(SECP256K1_BUILD_X AND TARGET secp256k1_x)`) so
CPU-only builds remain an unaffected no-op. Verified on real hardware (RTX
5060 Ti, isolated OpenCL-only build dir, `SECP256K1_BUILD_CUDA=OFF`): both
sighash standalone binaries now print `Backend: OpenCL` and exercise the real
on-device KAT/hostile-input coverage instead of self-skipping
(`test_regression_sighash_descriptor_gpu_standalone`: pass=18 fail=0;
`test_exploit_sighash_descriptor_malformed_standalone`: pass=37 fail=0), and
`unified_audit_runner` now shows real OpenCL backend activity for every
affected GPU differential module, not only sighash. The combined
CUDA+OpenCL `build-audit` profile could not be used for this proof: as of
2026-07-10, `src/gpu/src/gpu_backend_cuda.cu` (owned by the companion
`lbtc-sighash-gpu-core-cuda-deepseek` card, out of this task's `allowed_writes`)
does not compile (`nvcc` "transfer of control bypasses initialization of ..."
errors around `gpu_backend_cuda.cu:2174-2244`) — this is independent of the
audit-linkage fix above and blocks any target that needs
`secp256k1_gpu_host` with `SECP256K1_BUILD_CUDA=ON` until resolved on that
card.

Boundary KATs were added covering variable-length values of exactly 0, 1, 63,
64, and 65 bytes (every interesting SHA-256 block-boundary case around a
64-byte compression block), a 129-byte multi-block field, and adjacent short
fixed fields, to both `audit/test_regression_sighash_descriptor_gpu.cpp` and
`compat/libbitcoin_direct/tests/test_direct_operations.cpp` (the latter file
is owned by the companion CUDA card, not this task). Note: as of this writing
the `test_direct_operations.cpp` `var-len=0` KAT itself has a test-construction
bug outside this task's scope — `std::vector<uint8_t> raw_lit(0, 0)` for the
0-length case produces an empty vector whose `.data()` returns `nullptr`,
which correctly (not incorrectly) trips `sighash_descriptor_hash_batch`'s
"null field_data for a referenced field" guard; the two resulting test
failures are a fixture bug in that file's construction, not a backend defect
— confirmed by the equivalent `var_len=0` case passing cleanly in this task's
own `test_regression_sighash_descriptor_gpu.cpp` KAT, which backs the pointer
with a real (non-empty) buffer.

**Second repair round (2026-07-10, same day):** a follow-on review found that
neither new backend's descriptor-parse loop checked a FIXED-width field's
declared row stride (`field_lengths[fid]`) against that field's protocol-fixed
serialized length (`fixed_len` — e.g. txid/hashPrevouts=32, sequence=4,
amount=8), unlike the CPU direct parser
(`compat/libbitcoin_direct/include/ufsecp/libbitcoin.hpp:1500-1501`) and the
CUDA backend, which both already enforced this. Both kernels already clamp the
per-row read length to `min(fixed_len_or_varlen, stride)`, so this was never a
memory-safety out-of-bounds read — it was a silent-wrong-digest / spec-fidelity
bug: an undersized declared stride for a fixed field produced a HASH256
computed over truncated field bytes with no error, instead of the
deterministic `BadInput` rejection every other backend already gave. Fixed
identically on both backends (`gpu_backend_opencl.cpp` / `gpu_backend_metal.mm`,
same position in the parse loop, same `"stride < fixed_len"` error text as
CUDA): `if (!is_var && field_lengths[fid] < flen) return
set_error(GpuError::BadInput, "stride < fixed_len");`. All four
implementations (CPU direct, CUDA, OpenCL, Metal) now reject this case
identically. Two independent fixes landed alongside it, both host-side, no
kernel (`.cl`/`.metal`) change: (1) OpenCL's `sighash_dispatch_mtx_` previously
only wrapped the pool-alloc/upload/launch/readback span, not the lazy
`ensure_extended_kernels()` call preceding it, so two threads racing a cold
(never-yet-dispatched) backend instance could race the unsynchronized lazy
kernel-build path — the lock is now acquired before `ensure_extended_kernels()`
too, scoped to this op only; (2) two early-return paths in Metal's
`sighash_descriptor_hash` (`is_ready()` failure, and a NULL
`descriptor`/`field_data`/`field_lengths` with a valid non-NULL `out32`)
previously returned before the fail-closed `memset(out32, 0, out_bytes)` —
reordered so `count==0`/`out32==NULL` are checked first, `out_bytes` is
computed and `out32` is zeroed, and only then does `is_ready()` and the
remaining NULL checks run, plus explicit `uint64_t` vs `SIZE_MAX` bound checks
were added before the two `size_t` narrowing casts of
`packed_varlens_bytes64`/`total_col_bytes`. New test coverage:
`audit/test_exploit_sighash_descriptor_malformed.cpp`
(`test_fixed_field_stride_less_than_fixed_len`) and
`audit/test_regression_sighash_descriptor_gpu.cpp`
(`test_ocl_cold_concurrent_dispatch`, a 6-thread dispatch race against a fresh
ctx with no pre-spawn warm-up, unlike the pre-existing
`test_ocl_concurrent_dispatch`), both added to their existing `ALL_MODULES`
entries (`exploit_sighash_descriptor_malformed` / `sighash_descriptor_gpu`) —
no new modules registered. OpenCL/CUDA real-hardware validation for this
specific round: not yet run on this host — pending re-measurement, see
`workingdocs/libbitcoin_gpu_workloads/sighash_gpu_opencl_metal_evidence_claude.json`
round 4. Metal's fix is code-reviewed only, as with every other Metal op in
this table (no Apple/Metal toolchain on this Linux host).

| Backend | `sighash_descriptor_hash` path | Assurance | Notes |
|---|---|---|---|
| CPU | reference (deterministic fallback) | **HIGH** — `sighash_descriptor_hash_batch`'s CPU path performs the full descriptor parse, per-row `var_len`-vs-stride and 4 MiB preimage-size validation, then streams each field into `secp256k1::SHA256`; covered by `test_regression_sighash_descriptor_gpu.cpp` (KAT vs direct-concatenation oracle) and the pre-existing `compat/libbitcoin_direct/tests/test_direct_operations.cpp` sighash section | public-data variable-time; `out32` never touched on a rejected call |
| CUDA | native on-device kernel | **UNKNOWN — does not currently compile** — `sighash_descriptor_hash` virtual is CUDA-native (`lbtc_sighash_descriptor_hash_kernel` in `gpu_backend_cuda.cu`). As of 2026-07-10 this file fails to build (`nvcc` "transfer of control bypasses initialization of ..." errors, `gpu_backend_cuda.cu:2174-2244`), verified with `cmake --build build-audit --target secp256k1_gpu_host`. The prior noted gap (CUDA host not independently re-checking per-row `var_len > stride` / the 4 MiB cap before dispatch) was explicitly in scope for the companion card's repair round; whether it was fixed cannot be assessed until the file compiles again. `gpu_backend_cuda.cu` is out of this task's `allowed_writes` — owned by `lbtc-sighash-gpu-core-cuda-deepseek`. Do not promote this row until it both compiles and is hardware-verified | public-data variable-time |
| OpenCL | native on-device kernel | **HIGH** — `sighash_descriptor_hash` virtual is OpenCL-native (`lbtc_sighash_descriptor` kernel in `src/opencl/kernels/secp256k1_extended.cl` + override in `gpu_backend_opencl.cpp`); independently re-validates descriptor grammar, per-row `var_len > stride`, the 4 MiB preimage cap, AND (second repair round) that every FIXED-width field's stride is ≥ its protocol-fixed length (defense in depth, since this backend is also reachable directly via the C ABI); the lazy `ensure_extended_kernels()` build path is now covered by `sighash_dispatch_mtx_` (second repair round — closes a cold-start concurrency race, see above); operational error declines → CPU. Reviewer-verified on real hardware (RTX 5060 Ti, `build-review-lbtc-gpu` OpenCL profile, first repair round): `bench_lbtc_workloads`'s `sighash_batch` workload reached `backend=opencl`/`device=NVIDIA GeForce RTX 5060 Ti`/`evidence_class=gpu_acceleration` at both small (count=64) and medium batch classes, byte-identical to the independent oracle both times (small: 0.74 M rows/s / 1356.1 ns/row vs 3.31 M rows/s CPU-forced — GPU slower, per-call overhead not yet amortized at this size, consistent with the other 4 workloads at small; medium: 13.65 M rows/s / 73.3 ns/row vs 3.48 M rows/s CPU-forced — single-run functional evidence only, not a controlled ≥5-run speedup claim per this repo's perf protocol). The two standalone audit KATs also now run real on-device coverage instead of self-skipping (see the audit-linkage fix above): pass=18 fail=0 / pass=37 fail=0 (first-repair-round counts; re-run against the second repair round's new `stride < fixed_len` and concurrency test cases is pending — see the evidence JSON referenced above) | public-data variable-time; packed-buffer design (see above) — no array-of-pointers kernel argument; grow-only reusable `OclSighashPool` (first repair round) |
| Metal | native on-device kernel | **MEDIUM — code-complete, runtime parity PENDING Apple-hardware validation** — `sighash_descriptor_hash` virtual is Metal-native (`lbtc_sighash_descriptor` kernel in `src/metal/shaders/secp256k1_kernels.metal` + override in `gpu_backend_metal.mm`), mirroring the OpenCL packed-buffer design, its var_len/4-MiB defense-in-depth checks, and (first repair round) the same grow-only reusable `MetalSighashPool` design and Taproot field-id rejection; second repair round adds the same `stride < fixed_len` fixed-field check as OpenCL/CUDA/CPU-direct, plus reorders two early-return paths (`is_ready()` failure; NULL `descriptor`/`field_data`/`field_lengths` with non-NULL `out32`) to fire AFTER `out32` is zeroed, so both paths are now fail-closed instead of leaving `out32` untouched, and adds explicit `uint64_t`-vs-`SIZE_MAX` bound checks before two `size_t` narrowing casts; NOT built/run here: Metal compiles only on Apple, so this Linux host cannot execute it — same status already documented above for the sibling `hash256_var`/`merkle_pair_hash`/public-data batch ops | operational error declines → CPU. No measured Metal numbers; owner validates on Apple GPU before this row is promoted to HIGH |

Fail-closed: both new backends return `GpuError::BadInput` for every
malformed-descriptor case (grammar violations, `var_len > stride`, preimage
`> 4 MiB`) before touching `out32`; the C ABI wrapper's `clear_output_bytes`
(called before backend dispatch) means `out32` is zeroed on any reject that
fires after that point, matching the sibling ops' convention. Test coverage:
`audit/test_regression_sighash_descriptor_gpu.cpp` (KAT vs a from-scratch
per-row field-concatenation oracle, not the implementation's own parser),
`audit/test_exploit_sighash_descriptor_malformed.cpp` (hostile descriptor
grammar, `var_len > stride`, preimage `> 4 MiB`, NULL-pointer isolation,
positive control). Benchmark: `compat/libbitcoin_direct/bench/bench_workloads.cpp`
`sighash_batch` row (legacy BIP-143 sighash ALL descriptor; CPU-forced +
paired GPU row when a real backend is linked/ready).

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

### Public GPU ABI operations (20 functions, backend-neutral)

| Function | CPU (fast) | CPU (CT) | CUDA | OpenCL | Metal |
|---|---|---|---|---|---|
| `ufsecp_gpu_generator_mul_batch` (k·G) | Y | - | Y | Y | Y |
| `ufsecp_gpu_ecdsa_verify_batch` | Y | - | Y | Y | Y |
| `ufsecp_gpu_ecdsa_verify_opaque_rows` / `ufsecp_gpu_ecdsa_verify_lbtc_rows` alias | Y | - | Y | Y | Y |
| `ufsecp_gpu_schnorr_verify_batch` | Y | - | Y | Y | Y |
| `ufsecp_gpu_ecdh_batch` ¹ | Y | Y | Y | Y | Y |
| `ufsecp_gpu_hash160_pubkey_batch` | Y | - | Y | Y | Y |
| `ufsecp_gpu_hash256_var` | Y | - | Y | Y | Y |
| `ufsecp_gpu_merkle_pair_hash` | Y | - | Y | Y | Y |
| `ufsecp_gpu_sighash_descriptor_hash` | Y | - | Y | Y | Y |
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
