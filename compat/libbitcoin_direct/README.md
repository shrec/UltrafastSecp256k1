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
| sign / recover / keys / math / serialize / context | ⏳ next increment |
| silent payments (BIP-352 scan) | ⏳ engine has `bip352_*`; awaiting evoskuil's `silent::batch` design |
| GPU verify / scan | ⏳ engine GPU backends |

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
No `libsecp256k1` shim, no `libufsecp` C ABI, no bridge `.a`:

```cmake
# secp256k1-fast package exposes the engine target + C++ headers.
target_compile_definitions(bitcoin-system PRIVATE HAVE_ULTRAFAST)
target_link_libraries(bitcoin-system PRIVATE secp256k1::fastsecp256k1)
# include dirs: <prefix>/include  (ufsecp/libbitcoin.hpp + secp256k1/*.hpp)
```

Engine flags (set by the secp256k1-fast build): `-O2 -std=c++20` + the field/CT
backend selectors; nothing libbitcoin-specific. The header is `inline` — it folds
into libbitcoin's TUs, so the verify is a direct call into engine code (no symbol
boundary to cross).

## Remaining: full secp256k1 surface libbitcoin calls (to fully drop the shim)

libbitcoin's `src/crypto/secp256k1/*.cpp` calls these `secp256k1_*` (shim) symbols.
Each maps to an engine C++ call; add an inline `ufsecp::lbtc::*` and a
`#if defined(HAVE_ULTRAFAST)` branch in the corresponding libbitcoin file:

| libbitcoin call (file) | engine C++ mapping |
|------------------------|--------------------|
| `secp256k1_ecdsa_sign` (ecdsa.cpp) | `ct::ecdsa_sign(msg, sk)` → `to_compact`/opaque |
| `secp256k1_ecdsa_signature_normalize` (ecdsa.cpp) | `ECDSASignature::normalize()` (low-S) |
| `secp256k1_ecdsa_signature_serialize_compact/der` (ecdsa.cpp) | `to_compact()` / DER encode |
| `secp256k1_ec_pubkey_create` (keys.cpp) | `ct::generator_mul(sk)` → compress |
| `secp256k1_ec_seckey_verify` (keys.cpp) | `Scalar::parse_strict_nonzero` |
| `secp256k1_ec_pubkey_combine/negate` (math.cpp) | `Point::operator+` / `negate` |
| `secp256k1_ec_pubkey_tweak_add/mul` (math.cpp) | `Point + G·t` / `Point·t` |
| `secp256k1_ec_seckey_negate/tweak_add/mul` (math.cpp) | `Scalar` neg/add/mul |
| `secp256k1_ecdsa_recover` + recoverable (recover.cpp) | `ecdsa_recover` (recovery.hpp) |
| `secp256k1_keypair_create` / `schnorrsig_sign32` (schnorr.cpp) | `schnorr_keypair_create` / `schnorr_sign` |
| `secp256k1_xonly_pubkey_parse` / `tweak_add_check` (schnorr.cpp) | `schnorr_xonly_pubkey_parse` / taproot tweak |
| `secp256k1_context_create/destroy` (ec_context.cpp) | no-op (engine is contextless) |

Until those land, libbitcoin still links `libufsecp` for the cold-path shim
symbols (mixed state); migrating them all lets the build link **only** the engine
(`libfastsecp256k1`) + this header — no shim, no C ABI, no bridge.

## Minimal build (no shim/CABI/bridge/FFI/NuGet)

Consumer needs: this header's include dir + the engine C++ headers
(`secp256k1/*.hpp`) + the engine static lib (`libfastsecp256k1.a`). No pkg-config,
no NuGet, no shared object required for a static build.
