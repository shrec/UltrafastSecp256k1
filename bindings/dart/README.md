# ufsecp — Dart

Dart FFI binding for [UltrafastSecp256k1](https://github.com/shrec/UltrafastSecp256k1) — high-performance secp256k1 elliptic curve cryptography.

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

```dart
import 'package:ufsecp/ufsecp.dart';

final ctx = UfsecpContext();

final privkey = Uint8List(32)..[31] = 1;
final pubkey = ctx.pubkeyCreate(privkey);
final msgHash = UfsecpContext.sha256(utf8.encode('hello'));
final sig = ctx.ecdsaSign(msgHash, privkey);
final valid = ctx.ecdsaVerify(msgHash, sig, pubkey);

ctx.destroy();
```

## ECDSA Recovery

```dart
final rs = ctx.ecdsaSignRecoverable(msgHash, privkey);
final recovered = ctx.ecdsaRecover(msgHash, rs.signature, rs.recoveryId);
```

## Taproot (BIP-341)

```dart
final tok = ctx.taprootOutputKey(xonlyPub);
final tweaked = ctx.taprootTweakSeckey(privkey);
final tapValid = ctx.taprootVerify(tok.outputKeyX, tok.parity, xonlyPub);
```

## Architecture Note

The C ABI layer uses the **fast** (variable-time) implementation for maximum throughput. A constant-time (CT) layer with identical mathematical operations is available via the C++ headers for applications requiring timing-attack resistance.

## License

MIT
