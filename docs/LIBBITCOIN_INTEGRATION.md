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

## libbitcoin `batch_verify` Mapping

Keep libbitcoin's existing function shape. `count` is not a new libbitcoin
parameter; it is derived from the existing `std::span`.

```cpp
template <typename Batch>
data_chunk batch_verify(const std::span<Batch>& batch, bool) NOEXCEPT
{
    static thread_local ufsecp::lbtc::Controller context{ UFSECP_LBTC_AUTO };
    if (!context.ok()) std::abort();

    const auto count = batch.size();
    data_chunk results(count);
    const auto out = results.data();
    const auto in = pointer_cast<const uint8_t>(batch.data());

    if constexpr (is_same_type<Batch, schnorr::batch>)
    {
        constexpr auto extra_size = sizeof(Batch) - (sizeof(hash_digest) +
            sizeof(ec_xonly) + sizeof(ec_signature));
        context.verify_schnorr(in, count, extra_size, out);
    }
    else
    {
        constexpr auto extra_size = sizeof(Batch) - (sizeof(hash_digest) +
            sizeof(ec_compressed) + sizeof(ec_signature));
        context.verify_ecdsa(in, count, extra_size, out);
    }

    return results;
}
```

The C++20 typed-span overloads can remove the explicit `count` and `extra_size`
calculation at the call site:

```cpp
static thread_local ufsecp::lbtc::Controller context{ UFSECP_LBTC_AUTO };
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

## Backend Selection

Create one controller per worker thread or serialize access externally:

```cpp
static thread_local ufsecp::lbtc::Controller context{ UFSECP_LBTC_AUTO };
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

The consensus rule is CPU/GPU/libsecp equivalence: GPU verdicts must match the
CPU reference bit-for-bit, and the CPU reference is gated against
libsecp256k1-compatible behavior.
