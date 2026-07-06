# libbitcoin direct integration (`ufsecp/libbitcoin.hpp`)

The **single, minimal C++ integration surface** for libbitcoin. libbitcoin is a
static C++20 build that is making UltrafastSecp256k1 its default engine, so it
does **not** need any of the intermediary layers:

| layer | what it is | needed by libbitcoin? |
|-------|-----------|-----------------------|
| libsecp256k1 **shim** (`secp256k1.h`, `secp256k1_*`) | drop-in libsecp API for build-time swap | ❌ overhead once ufsecp is the default |
| ufsecp **C ABI** (`ufsecp.h`, `ufsecp_*`) | C interface for FFI / bindings | ❌ libbitcoin is C++, no FFI |
| **bridge** (`ufsecp_libbitcoin.h`, `ufsecp_lbtc_*`) | zero-copy C batch interface | ❌ still a C-ABI boundary |
| **direct** (`ufsecp/libbitcoin.hpp`, `ufsecp::lbtc::*`) | **inline C++ → engine** | ✅ this header |

`ufsecp::lbtc::*` are **stateless inline functions** that hand libbitcoin's exact
byte layouts straight to the engine (`secp256k1::*`) — zero marshalling, no
`thread_local` context, fully inline-able. Verify paths are variable-time (all
inputs public) — correct and fastest; no secret material is handled here.

## Byte layouts (identical to libbitcoin / libsecp256k1)

- pubkey: 33-byte compressed (`0x02/0x03 || X` big-endian)
- hash: 32-byte message hash
- ECDSA sig: 64-byte `secp256k1_ecdsa_signature` == raw scalar limbs **little-endian**
  (`r limbs || s limbs`). On LE x86 this is byte-identical to libbitcoin's
  `ec_signature` (which aliases `secp256k1_ecdsa_signature`).
- Schnorr sig: 64-byte BIP-340 (`R.x` big-endian `|| s` big-endian)
- x-only: 32-byte x-only public key

## Status

| surface | status |
|---------|--------|
| `ecdsa_verify` / `schnorr_verify` (single) | ✅ done + tested |
| `ecdsa_verify_batch` / `schnorr_verify_batch` (interleaved rows, MT) | ✅ done + tested |
| **`ecdsa_verify_columns` / `schnorr_verify_columns`** (Structure-of-Arrays, MT) | ✅ done + tested — **matches libbitcoin's column-span batch** |
| parallelism: fused parallel parse+verify, **7.6× / 16 cores** (no serial-parse Amdahl wall) | ✅ validated |
| public-data batch ops (`xonly_validate`, `pubkey_validate`, `taproot_commitment_verify`, `tagged_hash`, `tagged_hash_var`, `hash256`, `hash256_var`) | ✅ done + tested — one direct API, transparent GPU hook, deterministic CPU fallback |
| sign / recover / keys / math / serialize / context | ⏳ next increment |
| silent payments (BIP-352 scan) | ⏳ engine has `bip352_*`; awaiting evoskuil's `silent::batch` design |
| GPU verify / scan | ✅ direct libbitcoin GPU hook for columns + public-data batch ops; BIP-352 scan remains engine-level until libbitcoin exposes its batch design |

Verify is the IBD-critical path. Test: `tests/test_direct_verify.cpp` (ECDSA +
Schnorr, single + batch + columns, all-valid + fail-closed-on-tamper).

## How libbitcoin calls it (column-span / SoA batch)

libbitcoin's `ecdsa::batch` / `schnorr::batch` hold parallel spans. In
`batch_verify` (HAVE_ULTRAFAST), hand the span data straight to the engine:

```cpp
#include <ufsecp/libbitcoin.hpp>
// ecdsa::batch { span<hash_digest> digests; span<ec_compressed> points;
//                span<ec_signature> signatures; ... }
const auto count = batch.digests.size();
data_chunk results(count);
ufsecp::lbtc::ecdsa_verify_columns(
    pointer_cast<const uint8_t>(batch.digests.data()),     // [count][32]
    pointer_cast<const uint8_t>(batch.points.data()),      // [count][33] compressed
    pointer_cast<const uint8_t>(batch.signatures.data()),  // [count][64] opaque LE
    count, results.data(), /*max_threads=*/0 /*auto*/);
// schnorr: schnorr_verify_columns(digests, xonly_points[32], sigs64_bip340, ...)
```

`ec_signature` aliases `secp256k1_ecdsa_signature` (raw scalar limbs LE) == the
engine's opaque form — zero conversion. `max_threads`: 0=auto (all cores),
1=serial, N=cap. Returns true iff ALL valid; per-row 1/0 in `results`
(fail-closed). The fused MT path decompresses each chunk's pubkeys in parallel —
no serial prelude, so it scales on high-core boxes.

## Build (pure C++20 — no C ABI, no shim, no bridge, no FFI/NuGet)

libbitcoin compiles this header directly and links only the engine static lib.
No `libsecp256k1` shim, no `libufsecp` C ABI, no bridge `.a`.

One flag configures the canonical, bridge-free build (engine + this header's
interface target only — no shim, no C ABI, no `ufsecp_lbtc` bridge):

```sh
cmake -B <build> -DSECP256K1_BUILD_LIBBITCOIN=ON
```

There is a real CMake `INTERFACE` target the consumer links:
`secp256k1::fastsecp256k1_libbitcoin`. It carries the include dir for
`<ufsecp/libbitcoin.hpp>`, C++20, the `HAVE_ULTRAFAST` compile definition, and
the engine target `secp256k1::fastsecp256k1` (transitively). Link this **one**
target and you get all of it:

```cmake
# secp256k1-fast package (find_package or add_subdirectory) exposes the
# canonical libbitcoin integration target. Link it and you get
# <ufsecp/libbitcoin.hpp>, C++20, HAVE_ULTRAFAST, and the engine.
target_link_libraries(bitcoin-system PRIVATE secp256k1::fastsecp256k1_libbitcoin)
```

Engine target stable name: `secp256k1::fastsecp256k1` (the build-tree alias is
identical to the installed export name). The older `secp256k1::fast` alias still
works. The header is `inline` — it folds into libbitcoin's TUs, so the verify is
a direct call into engine code (no symbol boundary to cross).

**Tests.** Add `-DSECP256K1_BUILD_LIBBITCOIN_TESTS=ON` to build the CTest
`lbtc_direct_verify` / `lbtc_direct_operations`, then run:

```sh
ctest --test-dir <build> -R lbtc_direct --output-on-failure
```

**Benchmark/examples.** Add `-DSECP256K1_BUILD_LIBBITCOIN_BENCH=ON` to build
`bench_lbtc_direct_batch`, `bench_lbtc_public_ops`, and
`bench_lbtc_hash256_var`. Add `-DSECP256K1_BUILD_LIBBITCOIN_EXAMPLES=ON` for
`example_lbtc_public_ops`. All are direct C++ only: no C ABI, no shim, no bridge.

```sh
cmake --build <build> --target bench_lbtc_direct_batch bench_lbtc_public_ops \
  bench_lbtc_hash256_var example_lbtc_public_ops -j
<build>/compat/libbitcoin_direct/bench_lbtc_direct_batch 1000000 5 50000 \
  --json <build>/lbtc_direct_batch.json
<build>/compat/libbitcoin_direct/bench_lbtc_public_ops 8192 3 80 512 80 512 \
  --json <build>/lbtc_public_ops.json
<build>/compat/libbitcoin_direct/bench_lbtc_hash256_var 262144 5 512 80 512 \
  --json <build>/lbtc_hash256_var.json
<build>/compat/libbitcoin_direct/example_lbtc_public_ops
```

`bench_lbtc_public_ops` reports forced-CPU and production direct rows for every
public-data entrypoint. `bench_lbtc_hash256_var` reports serial reference,
forced CPU, and production direct rows for the larger variable-length HASH256
workload. If the hook is absent or declines, the production row is the
deterministic CPU fallback, not a GPU performance claim.

The shim, the C ABI, and the `ufsecp_lbtc_*` bridge are compatibility-only and
live behind a separate flag — `-DSECP256K1_BUILD_LIBBITCOIN_BRIDGE=ON`.

## Consumer rule

The libbitcoin consumer should link exactly
`secp256k1::fastsecp256k1_libbitcoin`. Do not link `libufsecp`, do not include
`ufsecp.h`, and do not call `ufsecp_lbtc_*` from the canonical integration. Those
surfaces remain only for compatibility tests and non-C++ FFI consumers.

## Minimal build (no shim/CABI/bridge/FFI/NuGet)

Consumer needs: this header's include dir + the engine C++ headers
(`secp256k1/*.hpp`) + the engine static lib (`libfastsecp256k1.a`). No pkg-config,
no NuGet, no shared object required for a static build.
