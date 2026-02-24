# UltrafastSecp256k1.Native

Native runtime package for [UltrafastSecp256k1](https://github.com/shrec/UltrafastSecp256k1) — providing the `ufsecp` stable C ABI for secp256k1 elliptic curve cryptography.

## What's included

| Platform | Library |
|----------|---------|
| Windows x64 | `ufsecp.dll` + `ufsecp_s.lib` (static) |
| Linux x64 | `libufsecp.so` |
| macOS ARM64 | `libufsecp.dylib` |

## Quick Start (.NET)

```csharp
using System.Runtime.InteropServices;
using System.Security.Cryptography;

// P/Invoke — the native library is auto-copied to output
[DllImport("ufsecp")] static extern int ufsecp_ctx_create(out IntPtr ctx);
[DllImport("ufsecp")] static extern void ufsecp_ctx_destroy(IntPtr ctx);
[DllImport("ufsecp")] static extern int ufsecp_selftest(IntPtr ctx);
[DllImport("ufsecp")] static extern int ufsecp_ec_pubkey_create(IntPtr ctx, byte[] pub33, byte[] seckey);
[DllImport("ufsecp")] static extern int ufsecp_ecdsa_sign(IntPtr ctx, byte[] sig64, byte[] msghash, byte[] seckey);
[DllImport("ufsecp")] static extern int ufsecp_ecdsa_verify(IntPtr ctx, byte[] msghash, byte[] sig64, byte[] pub33);

// Create context
ufsecp_ctx_create(out var ctx);

// Generate key pair
var seckey = RandomNumberGenerator.GetBytes(32);
var pubkey = new byte[33];
ufsecp_ec_pubkey_create(ctx, pubkey, seckey);

// ECDSA sign & verify
var msgHash = SHA256.HashData("hello world"u8);
var sig = new byte[64];
ufsecp_ecdsa_sign(ctx, sig, msgHash, seckey);
var ok = ufsecp_ecdsa_verify(ctx, msgHash, sig, pubkey);

ufsecp_ctx_destroy(ctx);
```

## API Coverage (45 functions)

Context, Key generation, ECDSA (sign/verify/recover/DER), Schnorr BIP-340, SHA-256 (SHA-NI accelerated), ECDH (compressed/xonly/raw), BIP-32 HD derivation, Bitcoin addresses (P2PKH/P2WPKH/P2TR), WIF encode/decode, public key tweaks.

## Constant-Time Architecture

All secret-key operations (signing, ECDH, key derivation) automatically use the constant-time layer — no flags, no opt-in. Both FAST and CT layers are always active simultaneously.

## Links

- [GitHub](https://github.com/shrec/UltrafastSecp256k1)
- [CHANGELOG](https://github.com/shrec/UltrafastSecp256k1/blob/main/CHANGELOG.md)
- [Stability Guarantees](https://github.com/shrec/UltrafastSecp256k1/blob/main/include/ufsecp/SUPPORTED_GUARANTEES.md)
