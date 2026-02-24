# ufsecp — Go

Go (CGo) binding for [UltrafastSecp256k1](https://github.com/shrec/UltrafastSecp256k1) — high-performance secp256k1 elliptic curve cryptography.

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

```go
import "github.com/shrec/UltrafastSecp256k1/bindings/go"

ctx, err := ufsecp.NewContext()
if err != nil { panic(err) }
defer ctx.Destroy()

privkey := make([]byte, 32)
privkey[31] = 1

pubkey, err := ctx.PubkeyCreate(privkey)
msgHash, _ := ufsecp.Sha256([]byte("hello"))
sig, err := ctx.EcdsaSign(msgHash, privkey)
valid, err := ctx.EcdsaVerify(msgHash, sig, pubkey)
```

## ECDSA Recovery

```go
sig, recid, err := ctx.EcdsaSignRecoverable(msgHash, privkey)
recovered, err := ctx.EcdsaRecover(msgHash, sig, recid)
```

## Taproot (BIP-341)

```go
outputKey, parity, err := ctx.TaprootOutputKey(xonlyPub, nil)
tweaked, err := ctx.TaprootTweakSeckey(privkey, nil)
valid, err := ctx.TaprootVerify(outputKey, parity, xonlyPub, nil)
```

## Architecture Note

The C ABI layer uses the **fast** (variable-time) implementation for maximum throughput. A constant-time (CT) layer with identical mathematical operations is available via the C++ headers for applications requiring timing-attack resistance.

## License

MIT
