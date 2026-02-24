# ultrafast_secp256k1 — C API

Standalone C header-only API for [UltrafastSecp256k1](https://github.com/shrec/UltrafastSecp256k1) — high-performance secp256k1 elliptic curve cryptography.

This is a **stateless** API with `secp256k1_*` naming (no context object). It differs from the main `ufsecp_*` context-based API.

## Features

- **ECDSA** — sign, verify, DER serialization (RFC 6979)
- **Schnorr** — BIP-340 sign/verify
- **ECDH** — shared secret
- **BIP-32** — HD key derivation
- **Taproot** — output key tweaking, commitment verification (BIP-341)
- **Addresses** — P2PKH, P2WPKH, P2TR
- **WIF** — encode/decode
- **Hashing** — SHA-256, HASH160

## Quick Start

```c
#include "ultrafast_secp256k1.h"

secp256k1_init();

uint8_t privkey[32] = {0};
privkey[31] = 1;

uint8_t pubkey[33];
secp256k1_ec_pubkey_create(privkey, pubkey);

uint8_t msg[32] = {0};
uint8_t sig[64];
secp256k1_ecdsa_sign(msg, privkey, sig);

int ok = secp256k1_ecdsa_verify(msg, sig, pubkey);
```

## API Differences

| Feature | `ufsecp_*` (main) | `secp256k1_*` (this) |
|---------|-------------------|----------------------|
| Context | Required | None (global init) |
| Init | `ufsecp_ctx_create()` | `secp256k1_init()` |
| Naming | `ufsecp_ecdsa_sign` | `secp256k1_ecdsa_sign` |
| Thread safety | Per-context | Global state |

## Architecture Note

Both APIs use the **fast** (variable-time) implementation for maximum throughput. A constant-time (CT) layer with identical mathematical operations is available via the C++ headers for applications requiring timing-attack resistance.

## License

MIT
