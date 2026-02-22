# UltrafastSecp256k1.Native

Native runtime package for [UltrafastSecp256k1](https://github.com/shrec/Secp256K1fast) — providing the `ufsecp` stable C ABI for secp256k1 elliptic curve cryptography.

## What's included

| Platform | Library |
|----------|---------|
| Windows x64 | `ufsecp.dll` + `ufsecp_s.lib` (static) |
| Linux x64 | `libufsecp.so` |
| macOS ARM64 | `libufsecp.dylib` |

## Quick Start (.NET)

```csharp
using System.Runtime.InteropServices;

// P/Invoke — the native library is auto-copied to output
[DllImport("ufsecp")] static extern int ufsecp_ctx_create(out IntPtr ctx);
[DllImport("ufsecp")] static extern void ufsecp_ctx_destroy(IntPtr ctx);
[DllImport("ufsecp")] static extern int ufsecp_selftest(IntPtr ctx);
```

## API Coverage (45 functions)

Context, Key generation, ECDSA (sign/verify/recover/DER), Schnorr BIP-340, SHA-256 (SHA-NI accelerated), ECDH (compressed/xonly/raw), BIP-32 HD derivation, Bitcoin addresses (P2PKH/P2WPKH/P2TR), WIF encode/decode, public key tweaks.

## Constant-Time Architecture

All secret-key operations (signing, ECDH, key derivation) automatically use the constant-time layer — no flags, no opt-in. Both FAST and CT layers are always active simultaneously.

## Links

- [GitHub](https://github.com/shrec/Secp256K1fast)
- [CHANGELOG](https://github.com/shrec/Secp256K1fast/blob/main/libs/UltrafastSecp256k1/CHANGELOG.md)
- [Stability Guarantees](https://github.com/shrec/Secp256K1fast/blob/main/libs/UltrafastSecp256k1/include/ufsecp/SUPPORTED_GUARANTEES.md)
