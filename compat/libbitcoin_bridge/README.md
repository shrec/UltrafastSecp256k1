# UltrafastSecp256k1 ⇄ libbitcoin acceleration bridge

A small **shim controller** that hands libbitcoin (and any other node) two
GPU-accelerated batch capabilities behind one tiny C ABI, with a **mandatory CPU
fallback**. Pure C / C++ — **no FFI, no language bindings**.

> Status: **CPU path built & tested; GPU dispatch compile-verified.**
> The header [`ufsecp_libbitcoin.h`](include/ufsecp_libbitcoin.h) is the
> entry-point contract; [`src/ufsecp_libbitcoin.cpp`](src/ufsecp_libbitcoin.cpp)
> implements it. The CPU consensus-reference path passes the correctness test
> ([`tests/test_lbtc_bridge.cpp`](tests/test_lbtc_bridge.cpp): all-valid,
> single-corruption identification, opaque-key stride, ECDSA + Schnorr, empty
> batch). GPU dispatch (`-DUFSECP_LBTC_WITH_GPU=ON`) compiles against the engine
> GPU ABI; runtime GPU verification + the differential consensus gate + the
> libbitcoin branch are the next steps. The contract is a proposal — corrections
> expected after libbitcoin review.

## Why this exists

From the design discussion with the libbitcoin maintainer (evoskuil):

> *"The batch processing for SP and for script validation are the two areas that
> interest us. Honestly it wouldn't be worth changing the dependency that we
> currently have on libsecp256k1 without these."*

So the bridge targets exactly those two:

1. **Script-signature batch verification** (the big win). Signatures are ~95% of
   script-validation cost. The node extracts `sig / key / sighash` triples from
   scripts (`CHECKSIG`, `CHECKMULTISIG`, Taproot `CHECKSIG` / `CHECKSIGADD`),
   packs them into one big array, and forwards the array in a single call —
   getting an array of per-row results back. Target: **IBD / historical block
   validation** (batches of tens of thousands of blocks), **not** mempool
   latency.

2. **BIP-352 Silent Payments scan** (bonus, for the Electrum/server side).
   GPU-accelerated ECDH scan reusing the existing engine SP pipeline.

ECDSA and Schnorr are **separate calls** — uniform-sized data, stacked
independently (one GPU stream each, or one card each).

## The entry-point contract

### One unified table per call

Each row is `[ signature record ][ optional opaque correlation key ]`:

| Kind    | Record (verified)                         | Bytes |
|---------|-------------------------------------------|-------|
| ECDSA   | `32 msg ‖ 33 pubkey ‖ 64 sig`             | 129   |
| Schnorr | `32 xonly ‖ 32 msg ‖ 64 sig`              | 128   |

> Note the deliberate field-order difference: the message is the **first** field
> for ECDSA and the **second** field for Schnorr. This matches the engine's
> existing `ufsecp_ecdsa_batch_verify` / `ufsecp_schnorr_batch_verify` packed
> formats.

The trailing **opaque key** is the caller's own tag (e.g. 3-byte block id,
4-byte tx id). The bridge **never interprets** it — it is carried only so an
invalid row can be mapped back to its block/tx **without a second side table**.

- `key_size` is passed on the call and is **variable**. `key_size == 0` disables
  the column entirely (fastest; row stride == record size).
- **Sizing contract (record count + key size, no buffer size).** The call passes
  exactly two of `{record count, key size, buffer size}` — the **record count**
  and the **key size**. The buffer size is *implied*, always
  `count * (RECORD + key_size)`. There is deliberately **no buffer-size argument**,
  so there is no buffer/stride mismatch and **no corresponding error condition**.
  **Keyed rows (the usual node case)** — each on-wire row is
  `[ record | key_size bytes ]`. In C++ you append your correlation key to the
  record as a packed struct and hand the controller a `std::span` of it; the
  wrapper **derives both sizes from the array** — the count from `span.size()` and
  the key size from `sizeof(Row) - sizeof(record)` — so you never pass a size by
  hand:

  ```cpp
  #pragma pack(push,1)
  struct Row { ufsecp::lbtc::EcdsaRecord rec; uint8_t txid[4]; };  // 129 + 4
  #pragma pack(pop)
  ctrl.verify_ecdsa(std::span<const Row>(rows, n), results);       // zero-copy
  ```

  (A `std::span<const EcdsaRecord>` is just the `key_size == 0` special case of
  the same template.) The low-level `verify_ecdsa(rows_ptr, count, key_size)` and
  the C ABI take the same two values explicitly.
- The caller builds **one unified table**; the controller internally splits each
  row into the signature payload (→ verify) and the opaque tail (→ carried).

### Results returned directly

- `results[i]` — `1` valid / `0` invalid, one byte per row (optional).
- `invalid_idx[]` + `invalid_count` — compact list of failing row indices
  (optional). Since failures are rare during honest IBD, returning just the few
  bad indices is cheap. The caller then reads its own key from
  `rows[idx]`’s opaque tail (or its own id table) to locate the block/tx.

This is the design space evoskuil and Vano converged on: *"make entry point
function as you wish with your data and return types … inside that function i
will link gpu side."* The controller is that middleware entry point.

## Consensus correctness

For block validation the GPU result is **consensus-bearing**. Correctness is
anchored on the CPU / libsecp256k1-equivalent reference:

- The CPU fallback (always present, never optional) is the reference.
- The GPU path must match the CPU path **bit-for-bit**, enforced by a
  differential test gate (GPU vs CPU vs libsecp256k1 shim).
- Structurally-invalid rows (`s >= n`, `R.x >= p`, off-curve pubkey) verify to
  `invalid`, never crash the batch.

## How it maps onto existing engine primitives

The controller is **dispatch + marshalling only** — it reuses primitives that
already exist and are tested:

| Capability        | GPU path                              | CPU fallback                          |
|-------------------|---------------------------------------|---------------------------------------|
| ECDSA batch       | `ufsecp_gpu_ecdsa_verify_batch`       | `ufsecp_ecdsa_batch_identify_invalid` |
| Schnorr batch     | `ufsecp_gpu_schnorr_verify_batch`     | `ufsecp_schnorr_batch_identify_invalid` |
| BIP-352 scan      | `ufsecp_gpu_bip352_scan_batch`        | engine CPU SP scan                    |

The GPU verify kernels already write **per-entry** results (one signature per
thread), so per-row pass/fail falls out for free.

## libbitcoin integration shape

libbitcoin-system wraps libsecp256k1 in
`src/crypto/secp256k1.cpp` (`<secp256k1.h>`, `<secp256k1_recovery.h>`,
`<secp256k1_schnorrsig.h>`) and has **no batch API today**. Two layers:

1. **Single-signature ops** — already covered by the drop-in
   `libsecp256k1_shim` (zero source changes; their existing wrapper just links
   our shim instead of stock libsecp256k1).
2. **Batch script-sig verify + SP scan** — *new* capability, added through this
   bridge controller. libbitcoin calls `ufsecp_lbtc_verify_ecdsa` /
   `_verify_schnorr` from its IBD signature-validation flow.

The integration is delivered to libbitcoin as a **separate branch / PR** (a
small plugin-style harness, mirroring the `bench_bip352` workflow used with
CraigRaw), so they can build, marshal their data into it, and we tune the GPU
side against their real pipeline.

## Build — libbitcoin profile (primary path)

The bridge is **compiled into the central C ABI library** (`libufsecp`), not shipped
as a separate library. A single flag activates the minimal node profile: drop-in
shim + libsecp contract (CT enforced, strict ABI) + the batch bridge, with tests,
benches, examples, Java and Ethereum turned off.

```bash
cmake -S . -B out/libbitcoin -G Ninja -DSECP256K1_BUILD_LIBBITCOIN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build out/libbitcoin --target ufsecp_static    # ufsecp_lbtc_* are in libufsecp
# add a GPU backend to enable GPU dispatch (UFSECP_LBTC_WITH_GPU is auto-defined):
#   -DSECP256K1_BUILD_CUDA=ON   (or -DSECP256K1_BUILD_OPENCL=ON)
```

A standalone `CMakeLists.txt` is also provided for out-of-tree dev/testing (builds a
small `ufsecp_lbtc_bridge` lib + the test + example against an installed/prebuilt
`libufsecp`); the profile path above is what libbitcoin ships.

## Files

```
compat/libbitcoin_bridge/
  include/ufsecp_libbitcoin.h   ← entry-point contract (public header)
  src/ufsecp_libbitcoin.cpp     ← controller (CPU fallback + GPU dispatch)
  tests/test_lbtc_bridge.cpp    ← correctness test (CPU path verified)
  example/lbtc_batch_demo.cpp   ← skeleton harness for libbitcoin
  CMakeLists.txt                ← standalone dev/test build
  README.md
```

Wired into the engine build via `SECP256K1_BUILD_LIBBITCOIN` in the top-level
`CMakeLists.txt` and `include/ufsecp/CMakeLists.txt`.
