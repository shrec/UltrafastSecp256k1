# react-native-ultrafast-secp256k1

High-performance secp256k1 elliptic curve cryptography for React Native, powered by [UltrafastSecp256k1](https://github.com/shrec/UltrafastSecp256k1).

Uses native C/C++ through JSI (Android NDK + iOS) for maximum performance — no bridge overhead.

## Features

- **ECDSA** — sign, verify, recover (RFC 6979, low-S)
- **Schnorr** — BIP-340 sign/verify
- **ECDH** — shared secret derivation
- **BIP-32** — HD key derivation
- **Taproot** — BIP-341 output key tweaking
- **Addresses** — P2PKH, P2WPKH, P2TR
- **WIF** — encode/decode
- **Hashing** — SHA-256, HASH160, tagged hash

## Install

```bash
npm install react-native-ultrafast-secp256k1
cd ios && pod install
```

## Quick Start

```js
import { Secp256k1 } from 'react-native-ultrafast-secp256k1';

const secp = new Secp256k1();

// Generate public key from 32-byte private key
const privkey = Buffer.from('...', 'hex'); // use secure random source
const pubkey = secp.ecPubkeyCreate(privkey);
```

## ECDSA Sign & Verify

```js
const msgHash = secp.sha256(Buffer.from('hello world'));

// Sign
const sig = secp.ecdsaSign(msgHash, privkey);

// Verify
const valid = secp.ecdsaVerify(msgHash, sig, pubkey);
console.log('valid:', valid); // true
```

## Schnorr (BIP-340)

```js
const xOnlyPub = secp.schnorrPubkey(privkey);
const msg = secp.sha256(Buffer.from('schnorr message'));
const auxRand = Buffer.alloc(32); // use crypto.getRandomValues in production

const schnorrSig = secp.schnorrSign(msg, privkey, auxRand);
const ok = secp.schnorrVerify(msg, schnorrSig, xOnlyPub);
```

## Bitcoin Addresses

```js
import { NETWORK_MAINNET } from 'react-native-ultrafast-secp256k1';

const p2wpkh = secp.addressP2WPKH(pubkey, NETWORK_MAINNET); // bc1q...
const p2tr = secp.addressP2TR(xOnlyPub, NETWORK_MAINNET);   // bc1p...
```

## BIP-32 HD Derivation

```js
const seed = Buffer.alloc(64); // use proper entropy
const master = secp.bip32MasterKey(seed);
const child = secp.bip32DerivePath(master, "m/44'/0'/0'/0/0");
const childPriv = secp.bip32GetPrivkey(child);
```

## Platform Requirements

| Platform | Requirement |
|----------|-------------|
| Android | NDK, minSdkVersion 21+ |
| iOS | iOS 13+, CocoaPods |
| React Native | >= 0.71.0 |

## Architecture Note

The C ABI layer uses the **fast** (variable-time) implementation for maximum throughput. A constant-time (CT) layer with identical mathematical operations is available via the C++ headers for applications requiring timing-attack resistance.

## License

MIT

## Links

- [GitHub](https://github.com/shrec/UltrafastSecp256k1)
- [Benchmarks](https://github.com/shrec/UltrafastSecp256k1/blob/main/docs/BENCHMARKS.md)
- [Changelog](https://github.com/shrec/UltrafastSecp256k1/blob/main/CHANGELOG.md)
