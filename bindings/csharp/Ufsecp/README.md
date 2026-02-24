# Ufsecp

C# P/Invoke bindings for [UltrafastSecp256k1](https://github.com/shrec/UltrafastSecp256k1) — high-performance secp256k1 elliptic curve cryptography.

Bundles native runtimes for Windows x64, Linux x64, Linux ARM64, and macOS ARM64. The native library is auto-copied to your build output — no manual setup required.

## Install

```bash
dotnet add package Ufsecp
```

## Quick Start

```csharp
using Ultrafast.Ufsecp;
using System.Security.Cryptography;

using var ctx = new Ufsecp();

// Generate key pair
byte[] privkey = RandomNumberGenerator.GetBytes(32);
byte[] pubkey = ctx.PubkeyCreate(privkey);

Console.WriteLine($"Version: {Ufsecp.VersionString}");
Console.WriteLine($"Pubkey:  {Convert.ToHexString(pubkey)}");
```

## ECDSA Sign & Verify

```csharp
byte[] msgHash = Ufsecp.Sha256("hello world"u8.ToArray());

// Sign (RFC 6979 deterministic nonce, low-S normalized)
byte[] sig = ctx.EcdsaSign(msgHash, privkey);

// Verify
bool valid = ctx.EcdsaVerify(msgHash, sig, pubkey);

// DER encode/decode
byte[] der = ctx.EcdsaSigToDer(sig);
byte[] compact = ctx.EcdsaSigFromDer(der);
```

## ECDSA Recovery

```csharp
var (recSig, recId) = ctx.EcdsaSignRecoverable(msgHash, privkey);
byte[] recovered = ctx.EcdsaRecover(msgHash, recSig, recId);
```

## Schnorr (BIP-340)

```csharp
byte[] xOnlyPub = ctx.PubkeyXonly(privkey);
byte[] auxRand = RandomNumberGenerator.GetBytes(32);

byte[] schnorrSig = ctx.SchnorrSign(msgHash, privkey, auxRand);
bool ok = ctx.SchnorrVerify(msgHash, schnorrSig, xOnlyPub);
```

## ECDH

```csharp
byte[] otherPriv = RandomNumberGenerator.GetBytes(32);
byte[] otherPub = ctx.PubkeyCreate(otherPriv);

byte[] shared = ctx.Ecdh(privkey, otherPub);        // SHA-256 of compressed point
byte[] xonly  = ctx.EcdhXonly(privkey, otherPub);    // SHA-256 of x-coordinate
byte[] raw    = ctx.EcdhRaw(privkey, otherPub);      // raw 32-byte x-coordinate
```

## Bitcoin Addresses

```csharp
string p2pkh  = ctx.AddrP2PKH(pubkey);                           // 1...
string p2wpkh = ctx.AddrP2WPKH(pubkey);                          // bc1q...
string p2tr   = ctx.AddrP2TR(xOnlyPub);                          // bc1p...
string test   = ctx.AddrP2WPKH(pubkey, Network.Testnet);         // tb1q...
```

## BIP-32 HD Derivation

```csharp
byte[] seed = RandomNumberGenerator.GetBytes(64);
byte[] master = ctx.Bip32Master(seed);
byte[] child = ctx.Bip32DerivePath(master, "m/44'/0'/0'/0/0");
byte[] childPriv = ctx.Bip32Privkey(child);
byte[] childPub = ctx.Bip32Pubkey(child);
```

## WIF

```csharp
string wif = ctx.WifEncode(privkey, compressed: true, Network.Mainnet);
var decoded = ctx.WifDecode(wif);
// decoded.Privkey, decoded.Compressed, decoded.Network
```

## Taproot (BIP-341)

```csharp
var (outputKeyX, parity) = ctx.TaprootOutputKey(xOnlyPub);
byte[] tweakedPriv = ctx.TaprootTweakSeckey(privkey);
bool tapValid = ctx.TaprootVerify(outputKeyX, parity, xOnlyPub);
```

## Hashing

```csharp
byte[] sha = Ufsecp.Sha256(data);                    // SHA-256 (SHA-NI accelerated)
byte[] h160 = Ufsecp.Hash160(data);                   // RIPEMD160(SHA256(data))
byte[] tagged = Ufsecp.TaggedHash("BIP0340/aux", data); // BIP-340 tagged hash
```

## API Coverage (45+ functions)

Keys, ECDSA (sign/verify/recover/DER), Schnorr BIP-340, ECDH (compressed/xonly/raw), SHA-256, HASH160, Tagged Hash, BIP-32 HD, Taproot (BIP-341), Bitcoin Addresses (P2PKH/P2WPKH/P2TR), WIF, Key Tweaking.

## Architecture Note

The C ABI layer uses the **fast** (variable-time) implementation for maximum throughput. A constant-time (CT) layer with identical mathematical operations is available via the C++ headers for applications requiring timing-attack resistance.

## Supported Platforms

| Platform | Runtime |
|----------|---------|
| Windows x64 | `ufsecp.dll` |
| Linux x64 | `libufsecp.so` |
| Linux ARM64 | `libufsecp.so` |
| macOS ARM64 | `libufsecp.dylib` |

## License

AGPL-3.0

## Links

- [GitHub](https://github.com/shrec/UltrafastSecp256k1)
- [Benchmarks](https://github.com/shrec/UltrafastSecp256k1/blob/main/docs/BENCHMARKS.md)
- [Changelog](https://github.com/shrec/UltrafastSecp256k1/blob/main/CHANGELOG.md)
