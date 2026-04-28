# libsecp256k1 Compatibility Shim

A thin C API wrapper that maps the [bitcoin-core/secp256k1](https://github.com/bitcoin-core/secp256k1) API surface onto **UltrafastSecp256k1** internals.

## Purpose

Drop-in replacement for projects written against the libsecp256k1 C API. Link this shim instead of libsecp256k1, and existing code works unchanged.

## Supported API Surface

| Category | Functions | Status |
|---|---|---|
| Context | `create`, `clone`, `destroy`, `randomize`, `set_illegal_callback`, `set_error_callback`, `selftest` | [OK] Full — randomize installs thread-local scalar blinding; callbacks store a user-supplied handler (default: `abort()`); selftest verifies 1·G |
| Public Keys | `pubkey_create`, `pubkey_parse`, `pubkey_serialize`, `pubkey_negate`, `pubkey_tweak_add`, `pubkey_tweak_mul`, `pubkey_combine` | [OK] |
| ECDSA | `ecdsa_sign`, `ecdsa_verify`, `signature_parse_compact`, `signature_serialize_compact`, `signature_normalize` | [OK] |
| Schnorr (BIP-340) | `schnorrsig_sign32`, `schnorrsig_verify` | [OK] |
| Extra Keys | `xonly_pubkey_parse`, `xonly_pubkey_serialize`, `xonly_pubkey_cmp`, `xonly_pubkey_from_pubkey`, `xonly_pubkey_tweak_add`, `xonly_pubkey_tweak_add_check`, `keypair_create`, `keypair_sec`, `keypair_pub`, `keypair_xonly_pub`, `keypair_xonly_tweak_add` | [OK] |
| Secret Keys | `seckey_verify`, `seckey_negate`, `seckey_tweak_add`, `seckey_tweak_mul` | [OK] |
| DER Signatures | `signature_parse_der`, `signature_serialize_der` | [OK] |
| Tagged Hash | `tagged_sha256` | [OK] |
| ECDH | `secp256k1_ecdh` (custom + default SHA-256 hash) | [OK] |
| ElligatorSwift (BIP-324) | `ellswift_encode`, `ellswift_decode`, `ellswift_create`, `ellswift_xdh` | [OK] |

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

- Context randomization (`secp256k1_context_randomize`) is fully implemented — it installs additive thread-local scalar blinding (`ct::set_blinding(r, r_G)`) that protects all subsequent signing on that thread. Passing NULL clears blinding (`ct::clear_blinding()`), matching libsecp256k1 semantics exactly.
- Custom `noncefp` callbacks in `secp256k1_ecdsa_sign` are **not forwarded** — the shim uses RFC 6979 internally. Passing a custom non-NULL nonce function (that is not `secp256k1_nonce_function_rfc6979` or `secp256k1_nonce_function_default`) returns 0 (fail-closed). Bitcoin Core passes NULL noncefp exclusively, so this is not a compatibility issue for Bitcoin Core integration.
- `secp256k1_context_static` is provided but points to a dummy.
- Performance characteristics differ (typically faster — uses w=18 precomputed table for generator muls, GLV for variable-base).

## Building

```bash
cmake -S . -B build -G Ninja
cmake --build build
```
