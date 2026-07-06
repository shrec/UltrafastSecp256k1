# libbitcoin hash256_var GPU/direct integration report

This report is the focused evidence page for the bridge-free
`ufsecp::lbtc::hash256_var_batch` entry point. For the benchmark covering every
libbitcoin public-data batch op, see
[`LIBBITCOIN_PUBLIC_OPS_BENCHMARKS.md`](LIBBITCOIN_PUBLIC_OPS_BENCHMARKS.md).

## What was added

- `ufsecp::lbtc::hash256_var_batch(inputs, input_lens, stride, count, out32)`
  accepts libbitcoin-owned serialized byte rows directly from C++.
- CUDA, OpenCL, and Metal all have native `GpuBackend::hash256_var`
  implementations; the direct libbitcoin surface reaches them through the
  engine-owned self-installing hook.
- The CPU fallback is deterministic and byte-identical. A GPU/backend operational
  failure declines to CPU; it is never represented as a consensus-invalid row or
  an all-zero hash.
- `bench_lbtc_hash256_var` measures the direct surface without using the C ABI,
  libsecp256k1 shim, or legacy `ufsecp_lbtc` bridge.

## Why it helps libbitcoin

`hash256_var_batch` targets the txid/wtxid-shaped workload without moving Bitcoin
transaction parsing into the GPU. libbitcoin keeps CPU ownership of BIP141/BIP144
serialization, then hands a batch of byte spans to the engine:

```cpp
ufsecp::lbtc::hash256_var_batch(
    serialized_rows, row_lengths, stride, count, out_hashes32);
```

Benefits:

- **No bridge overhead:** direct inline C++ call into the engine target
  `secp256k1::fastsecp256k1_libbitcoin`.
- **No caller chunking:** libbitcoin does not size GPU chunks, stage device
  buffers, or choose CPU vs GPU.
- **Same API on every build:** CPU-only and GPU-enabled builds use the same
  function and return type.
- **Real public-data acceleration:** HASH256 over serialized transactions is
  public data and can run variable-time on GPU without touching private keys or
  nonces.
- **Large-row path:** unlike fixed/tagged hash helpers that use small per-thread
  buffers, `hash256_var` streams rows in 64-byte SHA-256 blocks and supports the
  direct path policy cap of 4 MiB per row.
- **Fatal-not-invalid semantics:** no GPU error is collapsed into a consensus
  verdict.

## Benchmark commands

CPU/direct baseline:

```bash
cmake -S libs/UltrafastSecp256k1 -B out/libbitcoin-hash256-var-bench -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DSECP256K1_BUILD_LIBBITCOIN=ON \
  -DSECP256K1_BUILD_LIBBITCOIN_BENCH=ON
cmake --build out/libbitcoin-hash256-var-bench --target bench_lbtc_hash256_var -j
out/libbitcoin-hash256-var-bench/compat/libbitcoin_direct/bench_lbtc_hash256_var \
  262144 5 512 80 512 \
  --json out/libbitcoin-hash256-var-bench/lbtc_hash256_var.json
```

CUDA/direct evidence example:

```bash
cmake -S libs/UltrafastSecp256k1 -B out/libbitcoin-hash256-var-cuda -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DSECP256K1_BUILD_LIBBITCOIN=ON \
  -DSECP256K1_BUILD_LIBBITCOIN_GPU=ON \
  -DSECP256K1_BUILD_LIBBITCOIN_BENCH=ON \
  -DSECP256K1_BUILD_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build out/libbitcoin-hash256-var-cuda --target bench_lbtc_hash256_var -j
out/libbitcoin-hash256-var-cuda/compat/libbitcoin_direct/bench_lbtc_hash256_var \
  1048576 5 512 80 512 \
  --json out/libbitcoin-hash256-var-cuda/lbtc_hash256_var_cuda.json
```

Interpretation rule: `direct-production` is a GPU performance row only when the
JSON fields show `production_hook_installed: true` and
`production_hook_sample_status: 0`. If the hook is absent or declines, the row is
the deterministic CPU fallback and must not be cited as GPU throughput.

## Local evidence (2026-07-06, CPU/direct)

Command:

```bash
out/libbitcoin-hash256-var-bench/compat/libbitcoin_direct/bench_lbtc_hash256_var \
  262144 5 512 80 512 \
  --json out/libbitcoin-hash256-var-bench/lbtc_hash256_var.json
```

Host/build:

- Linux 6.8.0-134-generic, x86_64
- Intel Core i5-14400F, 16 logical CPUs
- GCC 14.2.0, Release, `SECP256K1_BUILD_LIBBITCOIN=ON`
- `SECP256K1_BUILD_LIBBITCOIN_GPU=OFF`, so `production_hook_installed=false`

Benchmark shape:

- `count=262144`, `iters=5`, `stride=512`, `input_lens=[80,512]`
- payload bytes per iteration: 77,437,923 bytes (73.85 MiB)
- stride bytes per iteration: 134,217,728 bytes (128.00 MiB)

| Row | M rows/s | Payload MiB/s | Stride MiB/s | ns/row | Speedup vs serial |
|---|---:|---:|---:|---:|---:|
| `serial-reference` | 4.60 | 1296.68 | 2247.45 | 217.3 | 1.00x |
| `direct-cpu-forced` | 4.67 | 1316.40 | 2281.62 | 214.0 | 1.02x |
| `direct-production` | 4.64 | 1307.02 | 2265.36 | 215.5 | 1.01x |

This local run is a CPU/direct integration number, not a GPU throughput claim:
the production hook was not linked in this build. It proves the direct benchmark
surface, reference parity, untouched-output bad-input behavior, and JSON evidence
format. GPU measurements should use the CUDA/OpenCL/Metal commands above and may
only cite GPU throughput when the JSON reports `production_hook_installed=true`
and `production_hook_sample_status=0`.

The exact local numbers depend on host CPU/GPU and build flags. Commit-time
validation for this report builds the benchmark and runs a CPU/direct smoke
artifact; GPU measurements require owner-run hardware because GitHub hosted
runners do not provide the CUDA/OpenCL/Metal device matrix.

## Links

- Integration guide: `docs/LIBBITCOIN_INTEGRATION.md`
- Benchmark source: `compat/libbitcoin_direct/bench/bench_hash256_var.cpp`
- All public ops benchmark: `docs/LIBBITCOIN_PUBLIC_OPS_BENCHMARKS.md`
- Direct C++ API: `compat/libbitcoin_direct/include/ufsecp/libbitcoin.hpp`
- GPU hook contract: `compat/libbitcoin_direct/include/ufsecp/lbtc_gpu_ops.hpp`
- Backend assurance: `docs/BACKEND_ASSURANCE_MATRIX.md`
