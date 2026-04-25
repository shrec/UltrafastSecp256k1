# libsecp256k1 Bitcoin Cash Node (BCHN) Compatibility Shim

A thin C API wrapper that maps the [Bitcoin Cash Node secp256k1](https://github.com/bitcoin-cash-node/secp256k1) API onto **UltrafastSecp256k1** internals.

## Purpose

Drop-in replacement for projects written against the BCHN secp256k1 C API. Link this shim instead of the BCHN fork, and existing BCH code works unchanged.

## Supported API Surface

| Category | Functions | Status |
|---|---|---|
| Context | `create`, `destroy`, `randomize` | [OK] Stub (context is no-op) |
| Public Keys | `pubkey_create`, `pubkey_parse`, `pubkey_serialize`, `pubkey_negate`, `pubkey_tweak_add`, `pubkey_tweak_mul`, `pubkey_combine` | [OK] |
| ECDSA | `ecdsa_sign`, `ecdsa_verify`, `signature_parse_compact`, `signature_serialize_compact`, `signature_normalize` | [OK] |
| DER Signatures | `signature_parse_der`, `signature_serialize_der` | [OK] |
| Tagged Hash | `tagged_sha256` | [OK] |
| Recoverable Sig (ECDSA) | `ecdsa_sign_recoverable`, `ecdsa_recover`, `recoverable_signature_*` | [OK] |
| ECDH | `secp256k1_ecdh` (custom + default SHA-256 hash) | [OK] |
| Secret Keys | `seckey_verify`, `seckey_negate`, `seckey_tweak_add`, `seckey_tweak_mul` | [OK] |
| BCH Schnorr | `schnorr_sign`, `schnorr_verify` | [OK] |

**Not included** (not part of BCH):
- `secp256k1_extrakeys` (Taproot / BIP-340 x-only keys — BTC only)
- `secp256k1_schnorrsig` (BIP-340 Schnorr — BTC only)
- `secp256k1_ellswift` (BIP-324 encrypted transport — BTC only)

## BCH Schnorr Scheme

The BCH Schnorr is **not** BIP-340. It uses:

```
k  = RFC6979(seckey, msg32)
R  = k * G
e  = SHA256(R.x[32] || P_compressed[33] || msg32[32])
s  = k + e * seckey  (mod n)
sig[64] = R.x[32] || s[32]
```

Verify: `s*G - e*P` has x-coordinate == `sig.r`.

## Usage

```cmake
add_subdirectory(path/to/UltrafastSecp256k1/compat/libsecp256k1_bchn_shim)
target_link_libraries(my_bch_app PRIVATE secp256k1_bchn_shim)
```

```c
#include <secp256k1.h>
#include <secp256k1_schnorr.h>

secp256k1_context *ctx = secp256k1_context_create(SECP256K1_CONTEXT_NONE);
unsigned char sig[64];
secp256k1_schnorr_sign(ctx, sig, msg32, seckey, NULL, NULL);
secp256k1_context_destroy(ctx);
```

## Performance

All generator multiplications use the w=18 precomputed fixed-base table (~6.7 µs). Variable-base multiplications use fast GLV (~17 µs). Both are significantly faster than the reference implementation.
