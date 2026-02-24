# ufsecp — Ruby

Ruby FFI binding for [UltrafastSecp256k1](https://github.com/shrec/UltrafastSecp256k1) — high-performance secp256k1 elliptic curve cryptography.

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

```ruby
gem 'ufsecp'
```

Requires `libufsecp.so` / `ufsecp.dll` / `libufsecp.dylib` on the library path.

## Quick Start

```ruby
require 'ufsecp'

ctx = Ufsecp::Context.new

privkey = "\x00" * 31 + "\x01"
pubkey = ctx.pubkey_create(privkey)
msg_hash = Ufsecp.sha256("hello")
sig = ctx.ecdsa_sign(msg_hash, privkey)
valid = ctx.ecdsa_verify(msg_hash, sig, pubkey)

ctx.destroy
```

## ECDSA Recovery

```ruby
sig, recid = ctx.ecdsa_sign_recoverable(msg_hash, privkey)
recovered = ctx.ecdsa_recover(msg_hash, sig, recid)
```

## Taproot (BIP-341)

```ruby
output_key, parity = ctx.taproot_output_key(xonly_pub)
tweaked = ctx.taproot_tweak_seckey(privkey)
valid = ctx.taproot_verify(output_key, parity, xonly_pub)
```

## Architecture Note

The C ABI layer uses the **fast** (variable-time) implementation for maximum throughput. A constant-time (CT) layer with identical mathematical operations is available via the C++ headers for applications requiring timing-attack resistance.

## License

MIT
