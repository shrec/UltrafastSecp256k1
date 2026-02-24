# ultrafast-secp256k1

High-performance Node.js native addon for secp256k1 elliptic curve cryptography, powered by [UltrafastSecp256k1](https://github.com/shrec/UltrafastSecp256k1).

## Features

- **ECDSA** — sign, verify, recover, DER serialization (RFC 6979)
- **Schnorr** — BIP-340 sign/verify
- **ECDH** — compressed, x-only, raw shared secret
- **BIP-32** — HD key derivation
- **Taproot** — output key tweaking (BIP-341)
- **Addresses** — P2PKH, P2WPKH, P2TR
- **WIF** — encode/decode
- **Hashing** — SHA-256 (hardware-accelerated), HASH160, tagged hash

## Install

```bash
npm install ultrafast-secp256k1
```

Requires a C++ compiler and `node-gyp` (the native addon is built on install).

## Quick Start

```js
const { Secp256k1 } = require('ultrafast-secp256k1');
const crypto = require('crypto');

const secp = new Secp256k1();

// Generate a random private key
const privkey = crypto.randomBytes(32);

// Derive compressed public key (33 bytes)
const pubkey = secp.ecPubkeyCreate(privkey);
console.log('pubkey:', pubkey.toString('hex'));
```

## ECDSA Sign & Verify

```js
const msgHash = secp.sha256(Buffer.from('hello world'));

// Sign (RFC 6979 deterministic nonce, low-S)
const sig = secp.ecdsaSign(msgHash, privkey);

// Verify
const valid = secp.ecdsaVerify(msgHash, sig, pubkey);
console.log('ECDSA valid:', valid); // true

// DER-encode for transmission
const der = secp.ecdsaSerializeDer(sig);
```

## Schnorr (BIP-340)

```js
const xOnlyPub = secp.schnorrPubkey(privkey);
const auxRand = crypto.randomBytes(32);
const msg = secp.sha256(Buffer.from('schnorr message'));

const schnorrSig = secp.schnorrSign(msg, privkey, auxRand);
const ok = secp.schnorrVerify(msg, schnorrSig, xOnlyPub);
console.log('Schnorr valid:', ok); // true
```

## ECDH

```js
const otherPriv = crypto.randomBytes(32);
const otherPub = secp.ecPubkeyCreate(otherPriv);

const shared = secp.ecdh(privkey, otherPub);       // SHA-256 of compressed point
const xonly = secp.ecdhXonly(privkey, otherPub);    // SHA-256 of x-coordinate
const raw = secp.ecdhRaw(privkey, otherPub);        // raw 32-byte x-coordinate
```

## Bitcoin Addresses

```js
const { NETWORK_MAINNET, NETWORK_TESTNET } = require('ultrafast-secp256k1');

const p2pkh = secp.addressP2PKH(pubkey, NETWORK_MAINNET);   // 1...
const p2wpkh = secp.addressP2WPKH(pubkey, NETWORK_MAINNET); // bc1q...
const p2tr = secp.addressP2TR(xOnlyPub, NETWORK_MAINNET);   // bc1p...
```

## BIP-32 HD Derivation

```js
const seed = crypto.randomBytes(64);
const master = secp.bip32MasterKey(seed);
const child = secp.bip32DerivePath(master, "m/44'/0'/0'/0/0");
const childPriv = secp.bip32GetPrivkey(child);
const childPub = secp.bip32GetPubkey(child);
```

## WIF

```js
const wif = secp.wifEncode(privkey, true, NETWORK_MAINNET);
const { privkey: decoded, compressed, network } = secp.wifDecode(wif);
```

## Taproot

```js
const { outputKeyX, parity } = secp.taprootOutputKey(xOnlyPub);
const tweakedPriv = secp.taprootTweakPrivkey(privkey);
```

## Architecture Note

Built on hand-optimized C/C++ with platform-specific acceleration (AVX2, SHA-NI, BMI2 on x86; NEON on ARM). The C ABI layer uses the **fast** (variable-time) implementation for maximum throughput. A constant-time (CT) layer with identical mathematical operations is available via the C++ headers for applications requiring timing-attack resistance.

| Operation | x86-64 | ARM64 | RISC-V |
|-----------|--------|-------|--------|
| ECDSA Sign | 8 μs | 30 μs | — |
| kG (generator mul) | 5 μs | 14 μs | 33 μs |
| kP (arbitrary mul) | 25 μs | 131 μs | 154 μs |

## License

MIT

## Links

- [GitHub](https://github.com/shrec/UltrafastSecp256k1)
- [Benchmarks](https://github.com/shrec/UltrafastSecp256k1/blob/main/docs/BENCHMARKS.md)
- [Changelog](https://github.com/shrec/UltrafastSecp256k1/blob/main/CHANGELOG.md)
