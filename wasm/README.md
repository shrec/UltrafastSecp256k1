# @ultrafastsecp256k1/wasm

Ultra high-performance secp256k1 for the browser and Node.js via Emscripten/WebAssembly.

Powered by [UltrafastSecp256k1](https://github.com/shrec/UltrafastSecp256k1).

## Install

```bash
npm install @ultrafastsecp256k1/wasm
```

## Features

- **ECDSA** sign/verify (RFC 6979 deterministic nonce, low-S)
- **Schnorr BIP-340** sign/verify
- **SHA-256** hashing
- Point arithmetic (scalar mul, add)
- Public key derivation
- Self-test
- ES6 module output with TypeScript declarations

## Quick Start

### Build

```bash
# Install Emscripten SDK
git clone https://github.com/emscripten-core/emsdk.git
cd emsdk && ./emsdk install latest && ./emsdk activate latest
source emsdk_env.sh

# Build WASM module
cd UltrafastSecp256k1
./scripts/build_wasm.sh
```

Output in `build/wasm/dist/`:
| File | Description |
|------|-------------|
| `secp256k1_wasm.wasm` | WebAssembly binary |
| `secp256k1_wasm.js` | Emscripten loader (ES6) |
| `secp256k1.mjs` | High-level JS wrapper |
| `secp256k1.d.ts` | TypeScript declarations |
| `package.json` | npm package metadata |

### Usage (Node.js)

```javascript
import { Secp256k1 } from './dist/secp256k1.mjs';

const lib = await Secp256k1.create();
console.log('Version:', lib.version());
console.log('Self-test:', lib.selftest() ? 'PASS' : 'FAIL');

// Generate public key from private key
const privkey = new Uint8Array(32);
privkey[31] = 1;  // scalar = 1 (generator point)
const { x, y } = lib.pubkeyCreate(privkey);
console.log('G.x:', Buffer.from(x).toString('hex'));

// ECDSA sign
const msgHash = lib.sha256(new TextEncoder().encode('Hello secp256k1'));
const sig = lib.ecdsaSign(msgHash, privkey);
console.log('Signature:', Buffer.from(sig).toString('hex'));

// ECDSA verify
const valid = lib.ecdsaVerify(msgHash, x, y, sig);
console.log('Valid:', valid);  // true
```

### Usage (Browser)

```html
<script type="module">
  import { Secp256k1 } from './dist/secp256k1.mjs';

  const lib = await Secp256k1.create();
  console.log('secp256k1 WASM loaded, version:', lib.version());
  // ... use lib.ecdsaSign(), lib.schnorrSign(), etc.
</script>
```

## API Reference

### `Secp256k1.create(options?): Promise<Secp256k1>`
Initialize the WASM module.

### `selftest(): boolean`
Run built-in self-test.

### `version(): string`
Library version (e.g. "3.0.0").

### `pubkeyCreate(seckey: Uint8Array): { x, y }`
Derive public key from 32-byte private key.

### `pointMul(pointX, pointY, scalar): { x, y }`
Scalar × Point multiplication.

### `pointAdd(px, py, qx, qy): { x, y }`
Point addition.

### `ecdsaSign(msgHash, seckey): Uint8Array`
ECDSA sign (64-byte compact: r ‖ s).

### `ecdsaVerify(msgHash, pubX, pubY, sig): boolean`
ECDSA verify.

### `schnorrSign(seckey, msg, auxRand?): Uint8Array`
Schnorr BIP-340 sign (64-byte: R.x ‖ s).

### `schnorrVerify(pubkeyX, msg, sig): boolean`
Schnorr BIP-340 verify.

### `schnorrPubkey(seckey): Uint8Array`
Derive x-only public key for BIP-340.

### `sha256(data): Uint8Array`
SHA-256 hash.

## C API (Low-level)

For direct use from C/C++ or custom WASM bindings, see [`secp256k1_wasm.h`](secp256k1_wasm.h).

## Build Options

```bash
# Debug build (assertions, safe heap, stack overflow checks)
./scripts/build_wasm.sh debug

# Or manually:
emcmake cmake -S wasm -B build/wasm -DCMAKE_BUILD_TYPE=Debug
cmake --build build/wasm -j
```

## License

AGPL-3.0

## Links

- [GitHub](https://github.com/shrec/UltrafastSecp256k1)
- [Benchmarks](https://github.com/shrec/UltrafastSecp256k1/blob/main/docs/BENCHMARKS.md)
- [Changelog](https://github.com/shrec/UltrafastSecp256k1/blob/main/CHANGELOG.md)
