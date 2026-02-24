# Ufsecp — PHP

PHP FFI binding for [UltrafastSecp256k1](https://github.com/shrec/UltrafastSecp256k1) — high-performance secp256k1 elliptic curve cryptography.

This is the **reference binding** with 100% API coverage.

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
- **Context** — create, destroy, clone, last_error, ctx_size

## Requirements

- PHP 7.4+ with FFI extension enabled
- `libufsecp.so` / `ufsecp.dll` / `libufsecp.dylib`

## Quick Start

```php
use Ultrafast\Ufsecp;

$ctx = new Ufsecp();

$privkey = str_repeat("\x00", 31) . "\x01";
$pubkey = $ctx->pubkeyCreate($privkey);
$msgHash = Ufsecp::sha256("hello");
$sig = $ctx->ecdsaSign($msgHash, $privkey);
$valid = $ctx->ecdsaVerify($msgHash, $sig, $pubkey);

$ctx->destroy();
```

## ECDSA Recovery

```php
[$sig, $recid] = $ctx->ecdsaSignRecoverable($msgHash, $privkey);
$recovered = $ctx->ecdsaRecover($msgHash, $sig, $recid);
```

## Taproot (BIP-341)

```php
[$outputKey, $parity] = $ctx->taprootOutputKey($xonlyPub);
$tweaked = $ctx->taprootTweakSeckey($privkey);
$valid = $ctx->taprootVerify($outputKey, $parity, $xonlyPub);
```

## Architecture Note

The C ABI layer uses the **fast** (variable-time) implementation for maximum throughput. A constant-time (CT) layer with identical mathematical operations is available via the C++ headers for applications requiring timing-attack resistance.

## License

MIT
