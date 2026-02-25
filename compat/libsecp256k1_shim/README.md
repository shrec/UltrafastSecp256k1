# libsecp256k1 Compatibility Shim

A thin C API wrapper that maps the [bitcoin-core/secp256k1](https://github.com/bitcoin-core/secp256k1) API surface onto **UltrafastSecp256k1** internals.

## Purpose

Drop-in replacement for projects written against the libsecp256k1 C API. Link this shim instead of libsecp256k1, and existing code works unchanged.

## Supported API Surface

| Category | Functions | Status |
|---|---|---|
| Context | `create`, `destroy`, `randomize` | [OK] Stub (context is no-op) |
| Public Keys | `pubkey_create`, `pubkey_parse`, `pubkey_serialize`, `pubkey_negate`, `pubkey_tweak_add`, `pubkey_tweak_mul`, `pubkey_combine` | [OK] |
| ECDSA | `ecdsa_sign`, `ecdsa_verify`, `signature_parse_compact`, `signature_serialize_compact`, `signature_normalize` | [OK] |
| Schnorr (BIP-340) | `schnorrsig_sign32`, `schnorrsig_verify` | [OK] |
| Extra Keys | `xonly_pubkey_parse`, `xonly_pubkey_serialize`, `keypair_create` | [OK] |
| Secret Keys | `seckey_verify`, `seckey_negate`, `seckey_tweak_add`, `seckey_tweak_mul` | [OK] |
| DER Signatures | `signature_parse_der`, `signature_serialize_der` | [OK] |
| Tagged Hash | `tagged_sha256` | [OK] |

## Usage

```cmake
# In your CMakeLists.txt
add_subdirectory(path/to/UltrafastSecp256k1/compat/libsecp256k1_shim)
target_link_libraries(my_app PRIVATE secp256k1_shim)
```

Then in your code -- no changes needed:

```c
#include <secp256k1.h>
#include <secp256k1_schnorrsig.h>

secp256k1_context* ctx = secp256k1_context_create(SECP256K1_CONTEXT_NONE);
// ... all existing code works ...
secp256k1_context_destroy(ctx);
```

## Limitations

- Context randomization (`secp256k1_context_randomize`) is accepted but has no effect -- UltrafastSecp256k1 does not use blinding.
- `secp256k1_context_static` is provided but points to a dummy.
- `secp256k1_ecdh` and `secp256k1_ellswift` modules are not yet shimmed.
- Performance characteristics differ (typically faster).

## Building

```bash
cmake -S . -B build -G Ninja
cmake --build build
```
