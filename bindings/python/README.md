# ufsecp — Python

Python ctypes binding for [UltrafastSecp256k1](https://github.com/shrec/UltrafastSecp256k1) — high-performance secp256k1 elliptic curve cryptography.

## Features

- **ECDSA** — sign, verify, recover, DER serialization (RFC 6979)
- **Schnorr** — BIP-340 sign/verify
- **ECDH** — compressed, x-only, raw shared secret
- **BIP-32** — HD key derivation (master/derive/path/privkey/pubkey)
- **Taproot** — output key tweaking, verification (BIP-341)
- **Addresses** — P2PKH, P2WPKH, P2TR
- **WIF** — encode/decode
- **Hashing** — SHA-256 (hardware-accelerated), HASH160, tagged hash
- **Key tweaking** — negate, add, multiply

## Install

```bash
pip install ufsecp
```

Requires the native `libufsecp.so` / `ufsecp.dll` / `libufsecp.dylib` alongside the package or set `UFSECP_LIB` env var.

## Quick Start

```python
from ufsecp import Ufsecp

with Ufsecp() as ctx:
    privkey = bytes(31) + b'\x01'
    pubkey = ctx.pubkey_create(privkey)
    msg_hash = ctx.sha256(b'hello')
    sig = ctx.ecdsa_sign(msg_hash, privkey)
    valid = ctx.ecdsa_verify(msg_hash, sig, pubkey)
```

## ECDSA Recovery

```python
rs = ctx.ecdsa_sign_recoverable(msg_hash, privkey)
recovered = ctx.ecdsa_recover(msg_hash, rs.signature, rs.recovery_id)
```

## BIP-32 HD Derivation

```python
master = ctx.bip32_master(seed)
child = ctx.bip32_derive_path(master, "m/44'/0'/0'/0/0")
child_priv = ctx.bip32_privkey(child)
child_pub = ctx.bip32_pubkey(child)
```

## Taproot (BIP-341)

```python
tok = ctx.taproot_output_key(xonly_pub)
tweaked = ctx.taproot_tweak_seckey(privkey)
valid = ctx.taproot_verify(tok.output_key_x, tok.parity, xonly_pub)
```

## Architecture Note

The C ABI layer uses the **fast** (variable-time) implementation for maximum throughput. A constant-time (CT) layer with identical mathematical operations is available via the C++ headers for applications requiring timing-attack resistance.

## License

MIT
