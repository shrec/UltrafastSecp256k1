# ufsecp — Rust

Safe Rust wrapper for [UltrafastSecp256k1](https://github.com/shrec/UltrafastSecp256k1) — high-performance secp256k1 elliptic curve cryptography.

Wraps the `ufsecp-sys` FFI crate with a safe, ergonomic API.

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

## Quick Start

```rust
use ufsecp::Context;

let ctx = Context::new()?;
let privkey = [0u8; 31].iter().chain(&[1u8]).copied().collect::<Vec<_>>();
let pubkey = ctx.pubkey_create(&privkey)?;
let msg_hash = Context::sha256(b"hello")?;
let sig = ctx.ecdsa_sign(&msg_hash, &privkey)?;
let valid = ctx.ecdsa_verify(&msg_hash, &sig, &pubkey)?;
```

## ECDSA Recovery

```rust
let (sig, recid) = ctx.ecdsa_sign_recoverable(&msg_hash, &privkey)?;
let recovered = ctx.ecdsa_recover(&msg_hash, &sig, recid)?;
```

## Taproot (BIP-341)

```rust
let (output_key, parity) = ctx.taproot_output_key(&xonly_pub, None)?;
let tweaked = ctx.taproot_tweak_seckey(&privkey, None)?;
let valid = ctx.taproot_verify(&output_key, parity, &xonly_pub, None)?;
```

## Architecture Note

The C ABI layer uses the **fast** (variable-time) implementation for maximum throughput. A constant-time (CT) layer with identical mathematical operations is available via the C++ headers for applications requiring timing-attack resistance.

## License

MIT
