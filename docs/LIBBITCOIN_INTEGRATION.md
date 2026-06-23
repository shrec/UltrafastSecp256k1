# libbitcoin integration (libbitcoin table format first)

This guide is for libbitcoin-system integration work against the current
UltrafastSecp256k1 `dev` branch. The integration rule is simple:

**libbitcoin keeps its existing batch table standard. UltrafastSecp256k1 accepts
that layout directly.**

The engine must not require libbitcoin to reshape batches, translate opaque
ECDSA signatures to a different public format, or add a second side table just
to call the accelerator. The libbitcoin row/span layout is the integration
contract.

## Integration Surfaces

There are two independent surfaces:

| Surface | Use in libbitcoin | Library/API |
|---|---|---|
| Normal per-signature verify/sign | Drop-in replacement for libsecp256k1 paths | `secp256k1_*` shim |
| libbitcoin batch / mmap tables | Existing `std::span<Batch>` packed tables, per-row results, optional GPU | `ufsecp_lbtc_*` |

`libfastsecp256k1` is the native engine package. `libufsecp` is the optional C
ABI package. For the libbitcoin batch path, the `ufsecp_lbtc_*` symbols are
compiled into `libufsecp` by the libbitcoin profile; libbitcoin should not link
a second bridge library.

## Build Profile

The libbitcoin profile is the intended one-flag entry point:

```bash
cmake -S . -B out/libbitcoin-release -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DSECP256K1_BUILD_LIBBITCOIN=ON \
  -DSECP256K1_BUILD_CABI=ON \
  -DSECP256K1_INSTALL_CABI=ON

cmake --build out/libbitcoin-release -j
cmake --install out/libbitcoin-release --prefix <install-prefix>
```

For local integration validation, also build the libbitcoin tests:

```bash
cmake -S . -B out/libbitcoin-test -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DSECP256K1_BUILD_LIBBITCOIN=ON \
  -DSECP256K1_BUILD_CABI=ON \
  -DSECP256K1_BUILD_LIBBITCOIN_TESTS=ON

cmake --build out/libbitcoin-test -j
ctest --test-dir out/libbitcoin-test -R "lbtc|shim" --output-on-failure
```

CUDA is optional:

```bash
cmake -S . -B out/libbitcoin-cuda -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DSECP256K1_BUILD_LIBBITCOIN=ON \
  -DSECP256K1_BUILD_CABI=ON \
  -DSECP256K1_BUILD_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=native
```

CUDA builds link the CUDA runtime statically by default
(`CMAKE_CUDA_RUNTIME_LIBRARY=Static`). The NVIDIA driver remains a normal host
dependency. On machines without a usable GPU backend, `UFSECP_LBTC_AUTO` binds
the CPU reference path.

### Windows Notes

- Windows x64 and Windows ARM64 are supported.
- MSVC `cl` is supported. The default MSVC path keeps the portable x64 ABI
  baseline and enables measured-safe `/Ob3`; whole-program `/GL` is opt-in via
  `SECP256K1_MSVC_WPO=ON` because it can force downstream `/LTCG`.
- The current MSVC Windows profile has been owner-benchmarked as accelerated on
  the integration machine. Treat that as owner-side evidence until a
  `bench_lbtc_batch --json` artifact from the same hardware is attached.
- `clang-cl` is the recommended fast Windows compiler when the consumer accepts
  the MSVC ABI with Clang codegen:

```bash
cmake --preset windows-clang-cl
cmake --build --preset windows-clang-cl -j
```

For Windows ARM64:

```bash
cmake --preset windows-arm64-clang-cl
cmake --build --preset windows-arm64-clang-cl -j
```

For a reproducible MSVC libbitcoin benchmark artifact:

```powershell
cmake -S . -B out/libbitcoin-msvc -G Ninja `
  -DCMAKE_BUILD_TYPE=Release `
  -DSECP256K1_BUILD_LIBBITCOIN=ON `
  -DSECP256K1_BUILD_CABI=ON `
  -DSECP256K1_BUILD_LIBBITCOIN_BENCH=ON

cmake --build out/libbitcoin-msvc -j
out\libbitcoin-msvc\include\ufsecp\bench_lbtc_batch.exe 1000000 5 50000 --json out\libbitcoin-msvc\lbtc_msvc.json
```

## Batch Row Contract

The row pointer always points at a tightly packed array. The caller passes the
record count and the trailing opaque key size. There is no buffer-size argument:
the byte size is implied by `count * (record_size + key_size)`.

| Batch kind | Verified record bytes | Signature format |
|---|---:|---|
| ECDSA | `32-byte sighash | 33-byte compressed pubkey | 64-byte signature` | libbitcoin `ec_signature` / libsecp-compatible opaque scalar storage |
| Schnorr | `32-byte sighash | 32-byte x-only pubkey | 64-byte signature` | BIP-340 public signature |

The bytes after the verified record are libbitcoin-owned correlation/accounting
data. The engine carries or ignores them according to the API; it never
interprets them as cryptographic input.

ECDSA details:

- `ufsecp_lbtc_verify_ecdsa` is the default libbitcoin path and is the same as
  `ufsecp_lbtc_verify_ecdsa_opaque`.
- Opaque ECDSA means libbitcoin's existing `ec_signature` storage: the copied
  `secp256k1_ecdsa_signature` scalar layout used by the current secp-backed
  path, not DER and not public compact big-endian `r||s`.
- High-S signatures with `s < n` are accepted and normalized internally in
  scratch before verification, matching libsecp256k1 fallback semantics.
- Caller-owned rows are not rewritten.
- Public compact `r||s` is supported only through the explicit
  `ufsecp_lbtc_verify_ecdsa_compact` and `*_columns_compact` variants.

### Multithreaded CPU verify (`_mt`)

The default packed-row verify functions run the **CPU** path single-threaded (one
controller call = one core), so for IBD you should call the multi-threaded twins,
which fan the CPU verify across cores:

```c
/* ECDSA — opaque ec_signature rows; 0 = auto (all cores) */
ufsecp_lbtc_verify_ecdsa_opaque_mt(ctrl, rows, n, key_size,
    results, invalid_idx, invalid_cap, invalid_count, /*max_threads=*/0, cancel);

/* Schnorr — BIP-340 rows */
ufsecp_lbtc_verify_schnorr_mt(ctrl, rows, n, key_size,
    results, invalid_idx, invalid_cap, invalid_count, /*max_threads=*/0, cancel);
```

- `max_threads`: `0` = auto (`hardware_concurrency`, all cores), `1` = serial
  (byte-for-byte identical to the non-`mt` function), `N` = up to N.
- The per-row verdict / `invalid_idx` / `invalid_count` / return codes are
  **identical** to the serial functions for any thread count (verify is
  variable-time over public data — threading is a pure throughput change).
- The **GPU path is unaffected** — `max_threads` governs only the CPU fallback.
- **Cancellation still works** under `_mt` (the token is polled between chunks).
- If your node already shards block validation across its own thread pool, keep
  calling the **single-threaded** functions per shard (with `max_threads` you
  could otherwise nest pools / oversubscribe).
- `ufsecp_lbtc_verify_ecdsa_compact_mt` accepts `max_threads` for symmetry but its
  CPU fallback verifies per-row (serial); prefer the opaque form for MT.

## libbitcoin `batch_verify` Mapping

Keep libbitcoin's existing function shape. `count` is not a new libbitcoin
parameter; it is derived from the existing `std::span`.

```cpp
template <typename Batch>
data_chunk batch_verify(const std::span<Batch>& batch, bool) NOEXCEPT
{
    static thread_local ufsecp::lbtc::Controller context;
    if (!context.ok()) std::abort();

    const auto count = batch.size();
    data_chunk results(count);
    const auto out = results.data();
    const auto in = pointer_cast<const uint8_t>(batch.data());

    ufsecp_error_t status = UFSECP_OK;
    if constexpr (is_same_type<Batch, schnorr::batch>)
    {
        constexpr auto extra_size = sizeof(Batch) - (sizeof(hash_digest) +
            sizeof(ec_xonly) + sizeof(ec_signature));
        status = context.verify_schnorr(in, count, extra_size, out);
    }
    else
    {
        constexpr auto extra_size = sizeof(Batch) - (sizeof(hash_digest) +
            sizeof(ec_compressed) + sizeof(ec_signature));
        status = context.verify_ecdsa(in, count, extra_size, out);
    }

    if (status == UFSECP_ERR_CANCELLED)
        return {}; // caller requested stop; discard partial verdicts
    if (status != UFSECP_OK)
        std::abort();

    return results;
}
```

No `_ex` entry point is required. Existing libbitcoin functions are extended with
a trailing optional cancellation token: `NULL`/omitted preserves the old behavior.
C++ callers get the default parameter from `ufsecp_libbitcoin.h`; plain C callers
pass the final argument explicitly.

A caller-owned token is useful for node shutdown, validation chaser rotation, or
window replacement. It is polled between chunks; on `UFSECP_ERR_CANCELLED`, the
caller must discard `results`/collect cells because a prefix may already have
been processed.

```cpp
struct cancel_state { std::atomic_bool stop{false}; };

int is_cancelled(const void* user) noexcept
{
    return static_cast<const cancel_state*>(user)->stop.load(std::memory_order_relaxed)
        ? 1 : 0;
}

cancel_state state;
ufsecp_cancel_token token{ is_cancelled, &state, 8192 }; // 0 = engine chunk default

const auto status = context.verify_ecdsa(in, count, extra_size, out, &token);
if (status == UFSECP_ERR_CANCELLED)
    return {}; // do not consume partial results
if (status != UFSECP_OK)
    std::abort();
```

The C++20 typed-span overloads can remove the explicit `count` and `extra_size`
calculation at the call site:

```cpp
static thread_local ufsecp::lbtc::Controller context;
data_chunk results(batch.size());
context.verify_ecdsa(std::span<const ecdsa::batch>{ batch }, results.data());
```

Use this only when the `Batch` type is a tightly packed standard-layout row
whose first bytes are the verified record.

## Columnar / mmap Path

If libbitcoin later rotates from horizontal rows to vertical mmap columns, use
the column API instead of rebuilding rows:

```cpp
context.verify_ecdsa_columns(
    msg_hashes32, pubkeys33, sigs64, count, results);

context.verify_schnorr_columns(
    msg_hashes32, pubkeys_x32, sigs64, count, results);
```

For in-place collection, pass the mutable key-cell column:

```cpp
context.collect_ecdsa_columns(
    msg_hashes32, pubkeys33, sigs64, count, key_cells, key_size);
```

The column ECDSA signature format is opaque by default, exactly like the row
API. Use `*_columns_compact` only for public compact `r||s`.

The current no-intermediate-table GPU path is the packed opaque row API. The
column API preserves the opaque format at the public boundary and is tracked in
`docs/LIBBITCOIN_PERF_MATRIX_STATUS.json` so native no-copy GPU-column claims
cannot be made until the parity surface exists across CUDA/OpenCL/Metal.

## Backend Selection

Create one controller per worker thread or serialize access externally. The
C++ wrapper defaults to `UFSECP_LBTC_AUTO`, so the normal libbitcoin call site
does not need to spell the enum:

```cpp
static thread_local ufsecp::lbtc::Controller context;
if (!context.ok()) std::abort();
```

Backend modes:

| Mode | Behavior |
|---|---|
| `UFSECP_LBTC_AUTO` | GPU if usable, otherwise CPU reference path |
| `UFSECP_LBTC_GPU` | require GPU; controller creation fails without one |
| `UFSECP_LBTC_CPU` | force CPU reference path |

Once the controller exists, signature outcomes are per-row results. A bad
signature is not an API error; `results[i] == 0` marks the row invalid.

## Silent Payments (BIP-352) — scan + 8-byte prefix match

`ufsecp_lbtc_match_silent_prefixes` (issue #312) is a **direct CPU port of the per-row scan
pipeline that the DuckDB extension** (`duckdb-ufsecp-extension`, used by Sparrow/Frigate) runs
in `ProcessBatch`. The byte layouts match that extension exactly, so the same data flows
through unchanged (the additive bridge entry point does **not** touch the extension's own
`ufsecp_scan` table function — Sparrow's existing API is untouched).

```c
int ufsecp_lbtc_match_silent_prefixes(
    const uint8_t  scan_privkey32[32],  // 32-byte scalar, LITTLE-ENDIAN (Frigate sends reversed)
    const uint8_t  spend_pubkey64[64],  // uncompressed point: x(32 LE) || y(32 LE)
    const uint8_t* tweaks,              // count * 64 bytes, each x(32 LE) || y(32 LE)
    const uint64_t* prefixes,           // count target prefixes (BE top 8 bytes of output x)
    size_t          count,
    uint8_t*        matches);           // count bytes (caller-allocated), 0/1
```

Per row, given the tweak point:

```
shared = scan_privkey * tweak_point                              (k*P, GLV plan)
hash   = TaggedHash("BIP0352/SharedSecret", compress(shared) || counter_be32(0))
output = spend_pubkey + hash * G
prefix = big-endian top 8 bytes of output.x                      (== ExtractUpper64)
matches[i] = (prefix == prefixes[i]) ? 1 : 0
```

Return value is the number of matching rows (`>= 0`), or `-1` on a NULL / malformed-key error.
This is a **filter**: an 8-byte prefix can collide, so confirm survivors against the full
32-byte x-coordinate before treating a row as a real hit. No controller is needed, so it runs
on any build; the existing GPU-accelerated `ufsecp_lbtc_sp_scan` (33-byte SEC1 tweaks →
`prefix64_out`) covers the GPU path for callers that prefer it.

Correctness is pinned by a golden vector: `test_lbtc_bridge` replicates `bench_bip352`'s
deterministic tweak #9999 and asserts this function reproduces its validated prefix
`0xb63b4601066a6971` (cross-checked there against libsecp256k1 / CUDA / OpenCL).

## Validation Checklist

Before handing a build to libbitcoin maintainers:

```bash
python3 tools/source_graph_kit/source_graph.py build -i
python3 ci/check_source_graph_quality.py
bash ci/run_fast_gates.sh
cmake --build out/libbitcoin-test -j
ctest --test-dir out/libbitcoin-test -R "lbtc|shim" --output-on-failure
```

For GPU-capable local machines, also run:

```bash
ctest --test-dir out/libbitcoin-cuda -R "lbtc_consensus_diff|lbtc_multisig_threshold" --output-on-failure
```

Benchmark runs should emit a JSON artifact:

```bash
out/libbitcoin-cuda/include/ufsecp/bench_lbtc_batch 1000000 5 50000 --json out/libbitcoin-cuda/lbtc_batch.json
python3 ci/audit_gate.py --libbitcoin-perf-matrix
```

The consensus rule is CPU/GPU/libsecp equivalence: GPU verdicts must match the
CPU reference bit-for-bit, and the CPU reference is gated against
libsecp256k1-compatible behavior.
