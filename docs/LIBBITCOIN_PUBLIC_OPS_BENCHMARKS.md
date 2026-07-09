# libbitcoin direct public-data benchmarks and examples

This page is the shareable benchmark/example index for the bridge-free
`ufsecp::lbtc::*` libbitcoin integration surface.

## What is covered

| Target | Covers | Purpose |
|---|---|---|
| `bench_lbtc_direct_batch` | ECDSA/Schnorr row + column verify | Signature verification throughput, including libbitcoin's packed row and SoA column layouts |
| `bench_lbtc_public_ops` | `xonly_validate_batch`, `pubkey_validate_batch`, `taproot_commitment_verify_batch`, `tagged_hash_batch`, tag-string overload, `tagged_hash_var_batch`, `hash256_batch`, `hash256_var_batch` | Block-connect-scale public-data entrypoints |
| `bench_lbtc_hash256_var` | `hash256_var_batch` only, larger count | txid/wtxid-shaped variable-length HASH256 preimage throughput |
| `bench_lbtc_workloads` | `txid_batch`, `wtxid_batch`, `merkle_pair_batch`, `merkle_root_batch` (`sighash_batch` excluded, see below) | libbitcoin block-processing-shaped workload evidence, schema `ufsecp-lbtc-gpu-workload-benchmark-v1`, one JSON artifact per workload |
| `example_lbtc_public_ops` | All public-data entrypoints | Runnable minimal integration example |

All targets link the direct C++ libbitcoin surface only:

- no `ufsecp` C ABI;
- no libsecp256k1 shim;
- no `ufsecp_lbtc` bridge;
- no caller-visible CPU/GPU split.

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
  which tracks whether the hook was active for that specific row), `backend`
  (always `"cpu"` — this harness has no backend-identification API yet),
  `device` (`"n/a"`), `driver_version` (`null`), `count`, `validation_hash`
  (hex SHA-256 over the row's output buffer), `validation_status` (always
  `"matched_reference"` — a mismatch aborts the run before the row is
  recorded, see `bench_output`), `evidence_class` (always `"api_correctness"`
  for this schema, never `"gpu_acceleration"` — this harness cannot yet
  distinguish device-compute time from CPU orchestration time, so it does not
  claim GPU acceleration evidence regardless of `hook_installed`).

Rejection rules (full authoritative text in
`workingdocs/libbitcoin_gpu_workloads/evidence_matrix_claude.json`
`rejection_rules`, gate implementation docstring in
`ci/check_lbtc_gpu_workload_evidence.py`): zero/negative timing on a present
field, `ns_per_row`/`count`/`best_seconds` arithmetic inconsistency (>1%),
missing/mistyped required fields, backend/device/driver_version
contradictions, GPU claims without `provider_linked`+`hook_installed`,
CPU-only rows relabeled `gpu_acceleration`, `validation_status` other than
`matched_reference`, and one-sided `speedup_vs_cpu_forced` claims.

Reproduce + validate:

```bash
cmake --build <build-dir> --target bench_lbtc_public_ops -j
<build-dir>/compat/libbitcoin_direct/bench_lbtc_public_ops \
  128 1 80 128 64 128 --json /tmp/lbtc_public_ops_evidence_smoke.json
python3 ci/check_lbtc_gpu_workload_evidence.py /tmp/lbtc_public_ops_evidence_smoke.json
```

The same gate also validates the future phase-aware
`ufsecp-lbtc-gpu-workload-benchmark-v1` schema (txid/wtxid/sighash/merkle
workloads, see `workingdocs/libbitcoin_gpu_workloads/evidence_matrix_claude.json`),
enforcing the stricter always-measured `prep_seconds`/`kernel_seconds` rule
documented there.

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

`bench_lbtc_workloads` benchmarks 4 libbitcoin block-processing-shaped
workloads over the same bridge-free `ufsecp::lbtc::*` surface:
`txid_batch`, `wtxid_batch`, `merkle_pair_batch`, `merkle_root_batch`.

`sighash_batch` is intentionally **not** implemented here: no
`sighash_descriptor_hash_batch` exists anywhere in this codebase yet, and
`workingdocs/libbitcoin_gpu_workloads/api_plan_blocker_resolution_deepseek.md`
(blocker B1) explicitly gates any sighash descriptor kernel work behind an
external libbitcoin-developer review that has not happened. Do not add a
`sighash_batch` row to this harness until that descriptor contract is
accepted.

### Schema and evidence honesty

Artifacts use schema `ufsecp-lbtc-gpu-workload-benchmark-v1` (defined in
`workingdocs/libbitcoin_gpu_workloads/evidence_matrix_claude.json`, enforced
by `ci/check_lbtc_gpu_workload_evidence.py`). That schema's envelope carries
a single `workload` + `batch_class` value, so `bench_lbtc_workloads` writes
**one JSON artifact per workload** (via `--json-dir`), not one combined
file.

Every row in every artifact is `mode="direct-cpu-forced"`,
`backend="cpu"`, `evidence_class="api_correctness"`: the GPU hook is
explicitly forced off before each timed call. This harness has no
backend/device/driver identification API, so — per the project rule against
fabricated GPU evidence — it never claims `gpu_acceleration`, regardless of
whether a GPU provider happens to be linked into the binary
(`provider_linked` records that fact only as metadata; it never changes
`backend` or `evidence_class`). Unlike `bench_lbtc_public_ops`
(schema v1, where `prep_seconds`/`kernel_seconds` may be `null`), schema v2
requires both to be real positive measurements: `prep_seconds` is the
one-time wall-clock cost of generating that workload's input buffers
(measured before the timed loop, not fabricated), and `kernel_seconds`
mirrors `best_seconds` (the single measured compute span, min-of-`iters`).

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

```bash
cmake --build <build-dir> --target bench_lbtc_workloads -j
mkdir -p /tmp/lbtc_workloads_smoke
<build-dir>/compat/libbitcoin_direct/bench_lbtc_workloads \
  --batch-class small --iters 5 --json-dir /tmp/lbtc_workloads_smoke
for f in /tmp/lbtc_workloads_smoke/*.json; do
  python3 ci/check_lbtc_gpu_workload_evidence.py "$f"
done
python3 ci/test_lbtc_gpu_workload_evidence.py
```

### Local Results (2026-07-09, CPU/direct, `batch-class=medium`)

Host/build:

- Linux 6.8.0-134-generic, x86_64
- Intel(R) Core(TM) i5-14400F, GCC 14.2.0, Release
- `SECP256K1_BUILD_LIBBITCOIN_GPU=ON` (OpenCL provider linked,
  `provider_linked=true`), but every row's hook is explicitly forced off
  (`hook_installed=false`) — these are CPU-forced correctness/throughput
  rows, not GPU acceleration evidence.

Command:

```bash
<build-dir>/compat/libbitcoin_direct/bench_lbtc_workloads \
  --batch-class medium --iters 5 --json-dir out/lbtc_workloads_medium
```

Shape: `batch_class=medium`, `iters=5` (txid/wtxid: 32768 rows,
merkle_pair: 65536 rows, merkle_root: 512 trees x 2048 leaves).

| Workload | Op | M rows/s | MiB/s | ns/row | prep (s) |
|---|---|---:|---:|---:|---:|
| `txid_batch` | `txid_hash` | 4.28 | 1202.72 | 233.8 | 0.023752 |
| `wtxid_batch` | `wtxid_hash` | 3.76 | 1468.94 | 266.2 | 0.034137 |
| `merkle_pair_batch` | `merkle_pair_hash` | 8.12 | 495.40 | 123.2 | 0.006827 |
| `merkle_root_batch` | `merkle_root_from_leaves` | 0.00418 | 261.40 | 239101.7 | 0.059297 |

`merkle_root_batch`'s `ns_per_row` is per *tree* (512 trees of 2048 leaves
each, not per leaf) — the small `M rows/s` value reflects that each row is
a full merkle tree, not a single hash. This is a single local run, not a
formal >=5-run controlled comparison (see `CLAUDE.md` Performance
Optimization Protocol) — no speedup claim is made, and there is nothing
here to compare against (no GPU row exists in this harness).

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
