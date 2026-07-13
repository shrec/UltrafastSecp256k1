# libbitcoin direct public-data benchmarks and examples

This page is the shareable benchmark/example index for the bridge-free
`ufsecp::lbtc::*` libbitcoin integration surface.

## What is covered

| Target | Covers | Purpose |
|---|---|---|
| `bench_lbtc_direct_batch` | ECDSA/Schnorr row + column verify | Signature verification throughput, including libbitcoin's packed row and SoA column layouts |
| `bench_lbtc_public_ops` | `xonly_validate_batch`, `pubkey_validate_batch`, `taproot_commitment_verify_batch`, `tagged_hash_batch`, tag-string overload, `tagged_hash_var_batch`, `hash256_batch`, `hash256_var_batch` | Block-connect-scale public-data entrypoints |
| `bench_lbtc_hash256_var` | `hash256_var_batch` only, larger count | txid/wtxid-shaped variable-length HASH256 preimage throughput |
| `bench_lbtc_workloads` | `txid_batch`, `wtxid_batch`, `merkle_pair_batch`, `merkle_root_batch`, `sighash_batch` | libbitcoin block-processing-shaped workload evidence, schema `ufsecp-lbtc-gpu-workload-benchmark-v1`, one JSON artifact per workload |
| `example_lbtc_public_ops` | All public-data entrypoints | Runnable minimal integration example |

All targets link the direct C++ libbitcoin surface only:

- no `ufsecp` C ABI;
- no libsecp256k1 shim;
- no `ufsecp_lbtc` bridge;
- no caller-visible CPU/GPU split.

## Related GPU kernel storage optimization (not benchmarked in this doc)

Task `opencl-generator-w4-production-claude-v4` (2026-07-12) hoisted the
OpenCL window-4 generator table (`GENERATOR_TABLE_W4`,
`src/opencl/kernels/secp256k1_extended.cl`) from a per-call private
per-work-item array rebuild (an unqualified function-scope `AffinePoint
table[16]` — PRIVATE address-space, per-work-item storage in OpenCL C 1.2,
NOT `__local`) to a
single `__constant` declaration. **None of the targets tracked on this
page are affected** — `taproot_commitment_verify_batch` /
`lbtc_commitment_verify` computes `Q = tweak*G + P` via
`shamir_double_mul_glv_impl` (deliberately avoiding
`scalar_mul_generator_windowed_impl` + `point_add_mixed_unchecked`, per the
comment in that kernel), and none of the other public-ops/workload targets
in the table above call `generator_mul_windowed` or
`scalar_mul_generator_windowed_impl` either. The only OpenCL entry points
that exercise the optimized function are `__kernel generator_mul_windowed`
and the BIP-352 scan pipeline (`bip352_pipeline_kernel` /
`bip352_pipeline_kernel_compressed`, `src/opencl/kernels/secp256k1_bip352.cl`)
— neither is part of the bridge-free `ufsecp::lbtc::*` surface this document
covers. See `docs/BACKEND_ASSURANCE_MATRIX.md` (section "OpenCL generator w4
table storage optimization") and the tracked, repository-local
[`docs/benchmark_artifacts/opencl_generator_w4_production_claude_v4.json`](benchmark_artifacts/opencl_generator_w4_production_claude_v4.json)
for the actual measured evidence.

## Build

CPU/direct:

```bash
cmake -S libs/UltrafastSecp256k1 -B out/libbitcoin-public-ops-bench -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DSECP256K1_BUILD_LIBBITCOIN=ON \
  -DSECP256K1_BUILD_LIBBITCOIN_TESTS=ON \
  -DSECP256K1_BUILD_LIBBITCOIN_BENCH=ON \
  -DSECP256K1_BUILD_LIBBITCOIN_EXAMPLES=ON
cmake --build out/libbitcoin-public-ops-bench \
  --target bench_lbtc_direct_batch bench_lbtc_public_ops bench_lbtc_hash256_var \
           bench_lbtc_workloads example_lbtc_public_ops -j
```

GPU/direct uses the same executable names; add the internal GPU provider:

```bash
cmake -S libs/UltrafastSecp256k1 -B out/libbitcoin-public-ops-cuda -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DSECP256K1_BUILD_LIBBITCOIN=ON \
  -DSECP256K1_BUILD_LIBBITCOIN_GPU=ON \
  -DSECP256K1_BUILD_LIBBITCOIN_BENCH=ON \
  -DSECP256K1_BUILD_LIBBITCOIN_EXAMPLES=ON \
  -DSECP256K1_BUILD_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=86
```

### Kernel staging (no manual copy)

When `SECP256K1_BUILD_LIBBITCOIN_GPU=ON` and the OpenCL backend is linked,
`compat/libbitcoin_direct/CMakeLists.txt` copies `secp256k1_extended.cl` and
its transitive includes into a `kernels/` directory next to
`bench_lbtc_public_ops`/`bench_lbtc_workloads` on every build (`POST_BUILD`,
`copy_if_different`). Normal in-tree builds are also covered by the
executable-relative resolver in `OpenCLBackend::resolve_opencl_kernel()`;
the staged `kernels/` copy is a fallback for out-of-tree or packaged bench
layouts where the source tree is not reachable. Run via `cmake -E chdir`
into the binary directory for the staged-copy reproduce path; no manual
`cp`/`rsync` of `src/opencl/kernels/` is required.

Interpretation rule: a production benchmark row is GPU performance evidence only
when the direct-GPU profile links `secp256k1_gpu_host` and the relevant hook
accepts the batch. CPU-only runs are still valuable: they prove the direct API,
output parity, JSON format, and no-shim/no-bridge build.

## Evidence gate (schema + rejection rules)

`bench_lbtc_public_ops --json <path>` artifacts (schema
`ufsecp-lbtc-public-ops-benchmark-v1`) carry evidence-gate fields on top of the
original throughput columns, validated by
`ci/check_lbtc_gpu_workload_evidence.py`:

- `host_context` (top-level): `compiler`, `cpu_model`, `turbo_disabled`,
  `cpu_pinned`, `kernel` — gathered once per run (Linux: `/proc/cpuinfo`,
  `sched_getaffinity`, `/sys/devices/system/cpu/intel_pstate/no_turbo`,
  `uname`; other platforms report honest `"unknown"`/`false`, never a guess).
- `phase_timing_available: false` (top-level) — this harness measures one
  wall-clock span per call with no separate upload/kernel/download
  instrumentation. `kernel_seconds` mirrors that single measured span
  (`best_seconds`); `prep_seconds`/`upload_seconds`/`download_seconds` are
  `null` rather than a fabricated split.
- Per row: `provider_linked` (was a GPU hook already self-installed in this
  binary *before* the row forced it off — independent of `hook_installed`,
  which tracks whether the hook was active for that specific row), `backend`,
  `device`, `driver_version`, `count`, `validation_hash` (hex SHA-256 over the
  row's output buffer), `validation_status` (always `"matched_reference"` — a
  mismatch aborts the run before the row is recorded, see `bench_output`),
  `evidence_class` (always `"api_correctness"` for this schema, **never**
  `"gpu_acceleration"`, regardless of backend — see below).
- `backend`/`device`/`driver_version` identification: a `direct-cpu-forced`
  row is always `backend="cpu"`, `device="n/a"`, `driver_version=null`. A
  `direct-production` row is `backend="cpu"`/`device="n/a"` **unless** the
  hook was active for that row (`hook_installed=true`), a real GPU backend is
  linked, initialized, and ready on this host, and an independent direct-hook
  probe for that op accepts the same batch and matches the CPU-forced oracle.
  Only then are `backend`/`device` queried live via
  `ufsecp::lbtc::gpu_hook::g_lbtc_gpu_telemetry_hook`
  (`GpuTelemetry`, populated only from the already-existing
  `GpuBackend::backend_id()`/`backend_name()`/`device_info()` — see
  `compat/libbitcoin_direct/include/ufsecp/lbtc_gpu_ops.hpp` and
  `src/gpu/src/gpu_engine_hook.cpp`). `driver_version` stays `null` even on a
  GPU row: `DeviceInfo` (`src/gpu/include/gpu_backend.hpp`) carries no driver
  field, so this harness reports the honest absence of that data rather than
  fabricate a string — `ci/check_lbtc_gpu_workload_evidence.py` accepts
  `driver_version=null` on a non-cpu row for exactly this reason (an empty
  string `""` is still rejected as malformed).
- `evidence_class` stays `"api_correctness"` unconditionally under this
  schema (`ufsecp-lbtc-public-ops-benchmark-v1`) even on a row with a real
  GPU `backend` — this harness still measures one wall-clock span per call
  with no upload/kernel/download phase-split, so it does not claim
  `gpu_acceleration` evidence under schema v1 regardless of which backend
  served the call. The phase-aware `bench_lbtc_workloads` harness below
  (schema `ufsecp-lbtc-gpu-workload-benchmark-v1`) is the one that emits
  `gpu_acceleration` rows.

Rejection rules (full authoritative text in
`workingdocs/libbitcoin_gpu_workloads/evidence_matrix_claude.json`
`rejection_rules`, gate implementation docstring in
`ci/check_lbtc_gpu_workload_evidence.py`): zero/negative timing on a present
field, `ns_per_row`/`count`/`best_seconds` arithmetic inconsistency (>1%),
missing/mistyped required fields, backend/device/driver_version
contradictions, GPU claims without `provider_linked`+`hook_installed`,
CPU-only rows relabeled `gpu_acceleration`, `validation_status` other than
`matched_reference`, one-sided `speedup_vs_cpu_forced` claims, and (for
`evidence_class="gpu_acceleration"` rows specifically) an explicit combined
check that backend/device/provider_linked/hook_installed/validation/timing
are ALL simultaneously satisfied, not merely individually plausible.

Reproduce + validate (CPU-only host — no GPU backend compiled in, every
production row honestly stays `backend="cpu"`):

```bash
cmake -S libs/UltrafastSecp256k1 -B out/libbitcoin-cpu-only -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DSECP256K1_BUILD_LIBBITCOIN=ON \
  -DSECP256K1_BUILD_LIBBITCOIN_BENCH=ON  # SECP256K1_BUILD_LIBBITCOIN_GPU left OFF (default)
cmake --build out/libbitcoin-cpu-only --target bench_lbtc_public_ops -j
out/libbitcoin-cpu-only/compat/libbitcoin_direct/bench_lbtc_public_ops \
  128 1 80 128 64 128 --json /tmp/lbtc_public_ops_evidence_smoke.json
python3 ci/check_lbtc_gpu_workload_evidence.py /tmp/lbtc_public_ops_evidence_smoke.json
```

Reproduce + validate (GPU-linked host — `direct-production` rows carry a
real, telemetry-identified `backend`/`device` only when the direct hook
independently accepted and matched the batch; otherwise they remain
`backend="cpu"`/`device="n/a"`):

```bash
cmake -S libs/UltrafastSecp256k1 -B out/libbitcoin-gpu-linked -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DSECP256K1_BUILD_LIBBITCOIN=ON \
  -DSECP256K1_BUILD_LIBBITCOIN_GPU=ON -DSECP256K1_BUILD_LIBBITCOIN_BENCH=ON \
  -DSECP256K1_BUILD_OPENCL=ON   # or -DSECP256K1_BUILD_CUDA=ON / -DSECP256K1_BUILD_METAL=ON
cmake --build out/libbitcoin-gpu-linked --target bench_lbtc_public_ops -j
# No manual kernel copy needed: compat/libbitcoin_direct/CMakeLists.txt stages
# secp256k1_extended.cl + includes into a kernels/ dir next to the binary
# (see "Kernel staging" below). `cmake -E chdir` exercises that staged-copy
# fallback path explicitly.
cmake -E chdir out/libbitcoin-gpu-linked/compat/libbitcoin_direct \
  ./bench_lbtc_public_ops 128 1 80 128 64 128 \
  --json /tmp/lbtc_public_ops_evidence_gpu.json
python3 ci/check_lbtc_gpu_workload_evidence.py /tmp/lbtc_public_ops_evidence_gpu.json
```

Review smoke on this machine 2026-07-09 (`build-review-lbtc-gpu`, OpenCL
provider linked), first pass: every `direct-production` row had
`hook_installed=true`, but the independent direct-hook probes declined the
benchmark shapes, so every row honestly remained `backend="cpu"`,
`device="n/a"`, `evidence_class="api_correctness"`. The artifact passed
`check_lbtc_gpu_workload_evidence.py` (22/22 rows, 0 rejected). This was the
CWD-relative kernel-resolution blocker described under "Root cause" below.

Same-day follow-up (task `lbtc-direct-bench-kernel-staging-claude`): with the
new `kernels/` staging in `compat/libbitcoin_direct/CMakeLists.txt` and using
the `cmake -E chdir` invocation above, `pubkey_validate`,
`commitment_verify`, `tagged_hash*`, `hash256*`, `txid_hash`, `wtxid_hash`,
and `merkle_pair_hash` all reached `backend="opencl"`,
`device="NVIDIA GeForce RTX 5060 Ti"` (`xonly_validate` alone still showed
`hook=yes` at ~0.0003 M rows/s — the OpenCL program's one-time JIT compile
absorbed into that first call). The artifact still passed
`check_lbtc_gpu_workload_evidence.py` (22/22 rows, 0 rejected). The backend
resolver fix in commit `a9e0c25d` independently finds the kernel via an
executable-relative walk-up for in-tree build directories; the `kernels/`
staging added here is the build-layout-independent fallback for out-of-tree
builds or packaged binary directories with the source tree stripped.

The same gate also validates the phase-aware
`ufsecp-lbtc-gpu-workload-benchmark-v1` schema (txid/wtxid/merkle workloads,
see `workingdocs/libbitcoin_gpu_workloads/evidence_matrix_claude.json`),
enforcing the stricter always-measured `prep_seconds`/`kernel_seconds` rule
documented there — see "Workload benchmark harness" below for that schema's
CPU-only vs GPU-linked evidence.

## Local Results (2026-07-06, CPU/direct)

Host/build:

- Linux 6.8.0-134-generic, x86_64
- Intel Core i5-14400F, 16 logical CPUs
- GCC 14.2.0, Release
- `SECP256K1_BUILD_LIBBITCOIN_GPU=OFF`, so `hook=no` for every row

Command:

```bash
out/libbitcoin-public-ops-bench/compat/libbitcoin_direct/bench_lbtc_public_ops \
  8192 3 80 512 80 512 \
  --json out/libbitcoin-public-ops-bench/lbtc_public_ops.json
```

Shape: `count=8192`, `iters=3`, `fixed_len=80`, `stride=512`,
`var_lens=[80,512]`.

| Operation | Mode | Hook | M rows/s | MiB/s | ns/row |
|---|---|---:|---:|---:|---:|
| `xonly_validate` | direct-cpu-forced | no | 0.15 | 4.55 | 6701.8 |
| `xonly_validate` | direct-production | no | 0.15 | 4.60 | 6640.0 |
| `pubkey_validate` | direct-cpu-forced | no | 0.26 | 8.15 | 3862.9 |
| `pubkey_validate` | direct-production | no | 0.29 | 9.09 | 3461.6 |
| `commitment_verify` | direct-cpu-forced | no | 0.04 | 3.79 | 24411.9 |
| `commitment_verify` | direct-production | no | 0.04 | 3.78 | 24450.4 |
| `tagged_hash` | direct-cpu-forced | no | 9.35 | 713.34 | 107.0 |
| `tagged_hash` | direct-production | no | 9.39 | 716.65 | 106.5 |
| `tagged_hash_tag_overload` | direct-cpu-forced | no | 9.42 | 718.53 | 106.2 |
| `tagged_hash_tag_overload` | direct-production | no | 9.42 | 718.97 | 106.1 |
| `tagged_hash_var` | direct-cpu-forced | no | 4.45 | 1271.51 | 224.9 |
| `tagged_hash_var` | direct-production | no | 4.46 | 1274.10 | 224.4 |
| `hash256` | direct-cpu-forced | no | 8.53 | 650.87 | 117.2 |
| `hash256` | direct-production | no | 6.33 | 483.13 | 157.9 |
| `hash256_var` | direct-cpu-forced | no | 4.22 | 1206.42 | 237.0 |
| `hash256_var` | direct-production | no | 4.11 | 1175.83 | 243.2 |

The `hash256` production row is lower than the forced row in this CPU-only run;
with `hook=no` that is benchmark noise / CPU scheduling, not a semantic change.
Use best-of/repeated local artifacts for absolute CPU claims.

Specialized larger `hash256_var` run:

```bash
out/libbitcoin-public-ops-bench/compat/libbitcoin_direct/bench_lbtc_hash256_var \
  262144 5 512 80 512 \
  --json out/libbitcoin-public-ops-bench/lbtc_hash256_var.json
```

| Row | M rows/s | Payload MiB/s | Stride MiB/s | ns/row | Speedup vs serial |
|---|---:|---:|---:|---:|---:|
| `serial-reference` | 4.60 | 1296.68 | 2247.45 | 217.3 | 1.00x |
| `direct-cpu-forced` | 4.67 | 1316.40 | 2281.62 | 214.0 | 1.02x |
| `direct-production` | 4.64 | 1307.02 | 2265.36 | 215.5 | 1.01x |

## Workload benchmark harness (`bench_lbtc_workloads`, schema v2)

`bench_lbtc_workloads` benchmarks 5 libbitcoin block-processing-shaped
workloads over the same bridge-free `ufsecp::lbtc::*` surface:
`txid_batch`, `wtxid_batch`, `merkle_pair_batch`, `merkle_root_batch`,
`sighash_batch`.

`sighash_batch` uses the canonical BIP-143 legacy sighash ALL descriptor
(`nVersion`, `hashPrevouts`, `hashSequence`, `outpoint`, `scriptCode`
(variable), `value`, `nSequence`, `hashOutputs`, `nLocktime`, `nHashType`),
validated against a direct per-row field-concatenation +
`secp256k1::SHA256::hash256` oracle that never calls
`sighash_descriptor_hash_batch`'s own parser. On this reviewer's host
(OpenCL-enabled build profile, RTX 5060 Ti), the workload reached
`backend=opencl`/`device=NVIDIA GeForce RTX 5060 Ti`/`evidence_class=gpu_acceleration`,
matching the independent oracle byte-for-byte. See
`docs/BACKEND_ASSURANCE_MATRIX.md` for the per-backend
`sighash_descriptor_hash` parity status (CUDA landed separately, does not
currently compile as of 2026-07-10 — see that doc; Metal is
compile/static-parity only, pending Apple hardware).

**Repair round (2026-07-10):** after replacing OpenCL/Metal's per-call buffer
allocation with a grow-only reusable pool (see `BACKEND_ASSURANCE_MATRIX.md`),
`sighash_batch` was re-measured on both `small` and `medium` batch classes on
the same `build-review-lbtc-gpu` OpenCL profile (RTX 5060 Ti):

| batch_class | mode | rows/s | ns/row |
|---|---|---:|---:|
| small (count=64) | direct-cpu-forced | 3.31 M | 301.8 |
| small (count=64) | direct-production (opencl) | 0.74 M | 1356.1 |
| medium | direct-cpu-forced | 3.48 M | 287.6 |
| medium | direct-production (opencl) | 13.65 M | 73.3 |

Both rows validated byte-identical against the independent oracle at both
batch sizes (the harness aborts on any mismatch; it did not). At `small`,
per-call dispatch/upload/download overhead still dominates and GPU is slower
than CPU — consistent with the other 4 workloads in this harness at the same
batch size. At `medium`, GPU overtakes CPU now that the reusable pool removes
the per-call allocation cost. This is a **single 5-iteration run**, not a
controlled ≥5-run/CPU-pinned/turbo-disabled measurement per this repo's
Performance Optimization Protocol — it is functional/hardware-execution
evidence of correct GPU execution at both batch sizes, not a confirmed
speedup claim.

### Schema and evidence honesty

Artifacts use schema `ufsecp-lbtc-gpu-workload-benchmark-v1` (defined in
`workingdocs/libbitcoin_gpu_workloads/evidence_matrix_claude.json`, enforced
by `ci/check_lbtc_gpu_workload_evidence.py`). That schema's envelope carries
a single `workload` + `batch_class` value, so `bench_lbtc_workloads` writes
**one JSON artifact per workload** (via `--json-dir`), not one combined
file.

Every artifact's first row is always `mode="direct-cpu-forced"`,
`backend="cpu"`, `evidence_class="api_correctness"`: the GPU hook is
explicitly forced off before that timed call, on every host, GPU-linked or
not.

**A second, paired row is present only when a real GPU backend is linked,
initialized, ready, and the direct workload hook independently accepts and
handles the batch on this host.** Backend identity is queried with
`ufsecp::lbtc::gpu_hook::g_lbtc_gpu_telemetry_hook` (`GpuTelemetry`, see
`compat/libbitcoin_direct/include/ufsecp/lbtc_gpu_ops.hpp` and
`src/gpu/src/gpu_engine_hook.cpp` — populated only from the already-existing
`GpuBackend::backend_id()`/`backend_name()`/`device_info()`, no new backend
method, no `gpu_backend.hpp`/`*_cuda.cu`/`*_opencl.cpp`/`*_metal.mm` edit).
When available and the direct hook accepts the exact benchmark shape, that
second row is `mode="direct-production"`,
`hook_installed=true`, `backend`/`device` set to the telemetry-identified
values (e.g. `"opencl"` / `"NVIDIA GeForce RTX 5060 Ti"`), and
`evidence_class="gpu_acceleration"` — checked against the **same** independent
oracle as the CPU-forced row (see below), so a `gpu_acceleration` row is only
ever emitted when its own output independently verified correct. If a direct
GPU hook accepts the batch but mismatches the oracle, the run aborts (same
fail-fast convention as the CPU-forced pass). If no GPU backend is linked or
ready, or if the hook declines the exact benchmark shape, only the first row
is emitted — `provider_linked` records whether an op hook was self-installed
in the binary at all, but never by itself upgrades a row to
`gpu_acceleration` (that needs `hook_installed=true`, a successfully queried
real backend/device, and direct-hook acceptance for that specific workload).

`driver_version` is `null` on every row, including `gpu_acceleration` rows:
`GpuBackend::DeviceInfo` carries no driver-version field, so this is an
honest absence-of-data marker, not a fabricated value (see
`ci/check_lbtc_gpu_workload_evidence.py` "Honest gaps").
`upload_seconds`/`download_seconds` are `null` on every row for the same
reason — `GpuBackend` exposes each op as a single opaque call with no
upload/kernel/download phase-split instrumentation on either the CPU or GPU
path, so `kernel_seconds` mirrors `best_seconds` (the single measured
compute span, min-of-`iters`) rather than inventing a split.
`prep_seconds` is the one-time wall-clock cost of generating that workload's
input buffers (measured before the timed loop, shared by both rows since
they hash/combine the identical prepared input). A paired
`gpu_acceleration` row also carries `speedup_vs_cpu_forced` (both
`compute_only_ratio` and `end_to_end_ratio`, computed from the two real
measured rows in the same artifact — never a one-sided claim).

### Independent validation oracles

Every row is checked against an oracle that does **not** call the
`ufsecp::lbtc` batch function under test, avoiding a tautological
self-check:

| Workload | Library function under test | Independent oracle |
|---|---|---|
| `txid_batch` | `txid_hash_batch` (alias of `hash256_var_batch`) | `secp256k1::SHA256::hash256(span)` per row, called directly |
| `wtxid_batch` | `wtxid_hash_batch` (alias of `hash256_var_batch`) | `secp256k1::SHA256::hash256(span)` per row, called directly |
| `merkle_pair_batch` | `merkle_pair_hash_batch` | `secp256k1::SHA256::hash256(left32\|\|right32)` per row, called directly |
| `merkle_root_batch` | `merkle_root_from_leaves` | hand-written level-reduction loop in `bench_workloads.cpp` (`independent_merkle_root`) with Bitcoin odd-leaf duplication re-derived at every level — does not call `merkle_pair_hash_batch`, `merkle_level_reduce_batch`, or `merkle_root_from_leaves` |

A mismatch aborts the run before any row is written (same fail-fast
convention as `bench_lbtc_public_ops`); `validation_status` is therefore
always `"matched_reference"` in a successfully written artifact.

### Batch-size sizing

Batch sizes are workload-shape *design parameters* from
`workingdocs/libbitcoin_gpu_workloads/benchmark_plan_claude.md` (that
document is explicit these are parameters, not benchmark claims), selected
via `--batch-class {small,medium,block_scale,stress}`:

| Batch class | txid/wtxid rows | merkle_pair rows | merkle_root trees |
|---|---:|---:|---:|
| `small` | 64 | 128 | 8 |
| `medium` | 32768 | 65536 | 512 |
| `block_scale` | 4096 | 8192 | 64 |
| `stress` | 1048576 | 2097152 | 4096 |

`merkle_root_batch` simulates a fixed 2048 leaves per tree regardless of
`batch_class` (`batch_class` controls how many trees/blocks are
benchmarked, not the shape of one tree) — a synthetic block-shaped leaf
count, not tied to any specific historical Bitcoin block.

### Reproduce + validate

CPU-only host (no GPU backend compiled in — every artifact has exactly 1 row):

```bash
cmake -S libs/UltrafastSecp256k1 -B out/libbitcoin-cpu-only -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DSECP256K1_BUILD_LIBBITCOIN=ON \
  -DSECP256K1_BUILD_LIBBITCOIN_BENCH=ON  # SECP256K1_BUILD_LIBBITCOIN_GPU left OFF (default)
cmake --build out/libbitcoin-cpu-only --target bench_lbtc_workloads -j
mkdir -p /tmp/lbtc_workloads_smoke
out/libbitcoin-cpu-only/compat/libbitcoin_direct/bench_lbtc_workloads \
  --batch-class small --iters 5 --json-dir /tmp/lbtc_workloads_smoke
for f in /tmp/lbtc_workloads_smoke/*.json; do
  python3 ci/check_lbtc_gpu_workload_evidence.py "$f"
done
python3 ci/test_lbtc_gpu_workload_evidence.py
```

GPU-linked host (a real backend is linked/ready; artifacts have 2 paired rows
only for workloads whose direct hook accepts the benchmark shape):

```bash
cmake -S libs/UltrafastSecp256k1 -B out/libbitcoin-gpu-linked -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DSECP256K1_BUILD_LIBBITCOIN=ON \
  -DSECP256K1_BUILD_LIBBITCOIN_GPU=ON -DSECP256K1_BUILD_LIBBITCOIN_BENCH=ON \
  -DSECP256K1_BUILD_OPENCL=ON   # or -DSECP256K1_BUILD_CUDA=ON / -DSECP256K1_BUILD_METAL=ON
cmake --build out/libbitcoin-gpu-linked --target bench_lbtc_workloads -j
mkdir -p /tmp/lbtc_workloads_gpu
# No manual kernel copy needed — see "Kernel staging" note above.
cmake -E chdir out/libbitcoin-gpu-linked/compat/libbitcoin_direct \
  ./bench_lbtc_workloads --batch-class small --iters 5 \
  --json-dir /tmp/lbtc_workloads_gpu
for f in /tmp/lbtc_workloads_gpu/*.json; do
  python3 ci/check_lbtc_gpu_workload_evidence.py "$f"
done
python3 ci/test_lbtc_gpu_workload_evidence.py
```

### Review Smoke Results (2026-07-09, `batch-class=small`)

Host/build:

- Linux 6.8.0-134-generic, x86_64
- Intel(R) Core(TM) i5-14400F, GCC 14.2.0, Release
- `SECP256K1_BUILD_LIBBITCOIN_GPU=ON`, `SECP256K1_BUILD_OPENCL=ON`,
  `SECP256K1_BUILD_CUDA=OFF` in `build-review-lbtc-gpu`.
- The lbtc GPU hook provider was linked, but the direct hook probes declined
  the benchmark shapes for `txid_batch`, `wtxid_batch`, `merkle_pair_batch`,
  and `merkle_root_batch`; therefore every artifact correctly emitted only
  one CPU-forced `api_correctness` row. All 4 artifacts passed
  `check_lbtc_gpu_workload_evidence.py` (1/1 row each, 0 rejected).

Command:

```bash
<build-dir>/compat/libbitcoin_direct/bench_lbtc_workloads \
  --batch-class small --iters 5 --json-dir /tmp/lbtc_workload_gpu_evidence_review
```

Shape: `batch_class=small`, `iters=5` (txid/wtxid: 64 rows,
merkle_pair: 128 rows, merkle_root: 8 trees x 2048 leaves).

| Workload | Op | Mode | Backend | M rows/s | MiB/s | ns/row | prep (s) |
|---|---|---|---|---:|---:|---:|---:|
| `txid_batch` | `txid_hash` | direct-cpu-forced | cpu | 4.81 | 1365.13 | 207.7 | 0.000054 |
| `wtxid_batch` | `wtxid_hash` | direct-cpu-forced | cpu | 3.39 | 1345.72 | 294.7 | 0.000059 |
| `merkle_pair_batch` | `merkle_pair_hash` | direct-cpu-forced | cpu | 8.52 | 519.76 | 117.4 | 0.000013 |
| `merkle_root_batch` | `merkle_root_from_leaves` | direct-cpu-forced | cpu | 0.004 | 254.28 | 245791.4 | 0.000800 |

`merkle_root_batch`'s `ns_per_row` is per *tree* (512 trees of 2048 leaves
each, not per leaf) — the small `M rows/s` value reflects that each row is
a full merkle tree, not a single hash. This is a single local run, not a
formal >=5-run controlled comparison (see `CLAUDE.md` Performance
Optimization Protocol). This run is evidence of the current honest state of
the evidence path, not a GPU speedup claim: although a GPU provider was
linked, the direct hooks declined these benchmark shapes, so no
`gpu_acceleration` rows or `speedup_vs_cpu_forced` ratios were emitted.

**Same-day follow-up (task `lbtc-direct-bench-kernel-staging-claude`):**
with the `kernels/` staging described in "Kernel staging" above and the
separate resolver walk-up fix in commit `a9e0c25d`, all 4 workloads now emit
a paired `gpu_acceleration` row at `--batch-class small`, `--iters 3`,
reproduced with the `cmake -E chdir` invocation:

| Workload | Backend | M rows/s (cpu-forced → production) | `speedup_vs_cpu_forced` (compute_only) |
|---|---|---|---:|
| `txid_batch` | opencl / RTX 5060 Ti | 4.28 → 0.61 | 0.14x |
| `wtxid_batch` | opencl / RTX 5060 Ti | 3.39 → 0.61 | 0.18x |
| `merkle_pair_batch` | opencl / RTX 5060 Ti | 8.52 → 1.60 | 0.19x |
| `merkle_root_batch` | opencl / RTX 5060 Ti | 0.004 → 0.001 | 0.27x |

All 4 artifacts still passed `check_lbtc_gpu_workload_evidence.py` (2/2 rows
each, 0 rejected). Read the ratio honestly: at `batch-class small` (64-128
rows / 8 trees), per-call OpenCL dispatch overhead makes the GPU path
*slower* than CPU, not faster — this is real measured data, not a speedup
claim. Larger `--batch-class` values were not benchmarked in this task (out
of scope for a CMake-staging card); a throughput crossover point, if any, is
unverified.

#### Root cause of the 2026-07-09 declines (diagnosed, not fixed in this pass)

The decline above is **not** specific to `txid_batch`/`wtxid_batch`/
`merkle_pair_batch`/`merkle_root_batch`, and not a bug in any of those four
workloads' library functions, hooks, or backend kernels. Reproduced on this
host with the new decline-reason diagnostics
(see "Decline diagnostics" in `docs/BENCHMARK_POLICY.md`): **every** lbtc op
that routes through `OpenCLBackend::ensure_extended_kernels()`
(`src/gpu/src/gpu_backend_opencl.cpp:2937`) — including `xonly_validate`,
`tagged_hash`, `hash256`, etc., not just the 4 workload ops — declined in
the same process, all reporting `gpu_error_code=102` (`GpuError::Launch`).

A standalone one-shot probe (single fresh process, single
`merkle_pair_hash()` call, no prior calls to poison the state) isolated the
underlying message: `"secp256k1_extended.cl not found"`. Root cause:
`ensure_extended_kernels()`'s kernel-source search list
(`src/gpu/src/gpu_backend_opencl.cpp:2959-2966`) is six hardcoded paths
resolved relative to the **process's current working directory**
(`std::ifstream f(path, ...)` in `load_file_to_string()`, line 60 of the
same file) — not the executable's own directory, and not anchored to the
build or source tree. `bench_lbtc_workloads`/`bench_lbtc_public_ops` live at
`<build-dir>/compat/libbitcoin_direct/`, and none of the six candidates
resolve `secp256k1_extended.cl` from a typical invocation CWD (repo root or
the build directory root) to its actual location
(`<build-dir>/src/opencl/kernels/secp256k1_extended.cl`, staged there only
as a side effect of building the unrelated `opencl_audit_runner` target, or
`src/opencl/kernels/secp256k1_extended.cl` in the source tree). Because
`ensure_extended_kernels()` also latches `ext_init_attempted_ = true` on
this first failure and returns the same `Launch` error (now generic:
`"extended kernel init previously failed"`) on every later call for the
rest of the process, one missed file poisons every extended-kernel lbtc op
for that entire benchmark run — this is why all 4 workloads decline
together, and why `bench_lbtc_public_ops` shows all 10 ops (not only
`merkle_pair_hash`) declining in the same run.

This also explains the discrepancy with the earlier same-day
`workingdocs/libbitcoin_gpu_workloads/workload_gpu_evidence_claude.json`
worker report, which recorded successful `opencl`/RTX-5060-Ti
`gpu_acceleration` rows for all 4 workloads: that session's invocation CWD
happened to satisfy one of the six candidates; this review smoke's did not.
The behavior is CWD-dependent, not flaky hardware or a regression in the
newly-added merkle code.

Manually staging a full copy of `src/opencl/kernels/` next to the bench
binary (matching candidate path `kernels/secp256k1_extended.cl`, the same
workaround already on record for the unrelated
`unified_audit_runner`/`opencl_audit_runner` kernel-deployment gap) lets
`clBuildProgram` proceed — confirming the failure is purely file-resolution,
not a kernel-source defect.

**Follow-up (task `lbtc-direct-bench-kernel-staging-claude`, same day):** the
second half of this blocker — a `POST_BUILD` kernel-copy step for the two
bench targets, mirroring `opencl_audit_runner`'s existing pattern — is now
done: `compat/libbitcoin_direct/CMakeLists.txt` stages
`secp256k1_extended.cl` and its 6 transitive includes
(`secp256k1_point.cl`, `secp256k1_field.cl`, `secp256k1_ct_ops.cl`,
`secp256k1_ct_field.cl`, `secp256k1_ct_scalar.cl`, `secp256k1_ct_point.cl`)
into `$<TARGET_FILE_DIR:...>/kernels/` for both `bench_lbtc_public_ops` and
`bench_lbtc_workloads`, gated on the libbitcoin GPU bench linking an OpenCL
backend (`_lbtc_gpu_host AND SECP256K1_BUILD_OPENCL`). Combined with
`cmake -E chdir` into the binary's
directory (see reproduce commands above/below), this makes the legacy
CWD-relative `kernels/` search candidate in `resolve_opencl_kernel()`
succeed with no manual copy step. The OpenCL resolver implementation in
`src/gpu/src/gpu_backend_opencl.cpp` remains untouched by this task (backend
implementation, forbidden to edit under this card's contract). The separate
resolver card landed in commit `a9e0c25d` and independently finds the kernel
with no staging for in-tree build directories; the staged `kernels/` copy
remains useful for out-of-tree/package layouts. See
`workingdocs/libbitcoin_gpu_workloads/direct_bench_kernel_staging_claude.json`
for the measured before/after.

## Example

```bash
out/libbitcoin-public-ops-bench/compat/libbitcoin_direct/example_lbtc_public_ops
```

Expected output includes sample hash prefixes and:

```text
example_lbtc_public_ops: PASS (direct C++, no C ABI/shim/bridge)
```

Source: `compat/libbitcoin_direct/examples/public_ops_example.cpp`.

## Benefit Summary

- `xonly_validate_batch`: batch BIP-340 `lift_x` validation for x-only keys.
- `pubkey_validate_batch`: batch SEC1 compressed-pubkey validation.
- `taproot_commitment_verify_batch`: batch raw Taproot tweak commitment check
  (`Q = P + tweak*G`) without caller-side GPU staging.
- `tagged_hash_batch` / overload: BIP-340 tagged hash at block-connect scale.
- `tagged_hash_var_batch`: variable-length tagged hash without the old
  256-byte CPU/GPU behavior split.
- `hash256_batch`: fixed-length Bitcoin double-SHA256 batching.
- `hash256_var_batch`: variable-length Bitcoin double-SHA256 for txid/wtxid
  preimages after libbitcoin serializes transactions on CPU.
