# Bindings Parity Matrix

**UltrafastSecp256k1 v3.14.0** — Cross-Language Coverage & Verification Status

---

## Overview

All bindings wrap the stable **ufsecp C ABI v1** (`include/ufsecp/ufsecp.h`).
Each binding ships two implementations:
- **Legacy** (`UltrafastSecp256k1` / `Secp256k1`): direct C++ API (deprecated)
- **Stable** (`Ufsecp` / `ufsecp`): context-based C ABI with error codes

This document tracks the **stable (ufsecp)** bindings only.

---

## API Function Coverage

| C API Function | C# | Java | Swift | RN | Node | Python | Rust | Go | Dart | PHP | Ruby |
|---------------|:--:|:----:|:-----:|:--:|:----:|:------:|:----:|:--:|:----:|:---:|:----:|
| **Context** | | | | | | | | | | | |
| `ctx_create` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `ctx_clone` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `ctx_destroy` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `last_error` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `last_error_msg` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `abi_version` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Private Key** | | | | | | | | | | | |
| `seckey_verify` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `seckey_negate` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `seckey_tweak_add` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `seckey_tweak_mul` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Public Key** | | | | | | | | | | | |
| `pubkey_create` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `pubkey_create_uncompressed` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `pubkey_parse` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `pubkey_xonly` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **ECDSA** | | | | | | | | | | | |
| `ecdsa_sign` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `ecdsa_verify` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `ecdsa_sig_to_der` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `ecdsa_sig_from_der` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `ecdsa_sign_recoverable` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `ecdsa_recover` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Schnorr** | | | | | | | | | | | |
| `schnorr_sign` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `schnorr_verify` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **ECDH** | | | | | | | | | | | |
| `ecdh` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `ecdh_xonly` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `ecdh_raw` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Hash** | | | | | | | | | | | |
| `sha256` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `hash160` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `tagged_hash` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Address** | | | | | | | | | | | |
| `addr_p2pkh` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `addr_p2wpkh` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `addr_p2tr` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **WIF** | | | | | | | | | | | |
| `wif_encode` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `wif_decode` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **BIP-32** | | | | | | | | | | | |
| `bip32_master` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `bip32_derive` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `bip32_derive_path` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `bip32_privkey` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `bip32_pubkey` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Taproot** | | | | | | | | | | | |
| `taproot_output_key` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `taproot_tweak_seckey` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `taproot_verify` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

**Total functions**: 42 &nbsp; | &nbsp; **Coverage**: 42/42 per binding (100%)

---

## Infrastructure Parity

| Feature | C# | Java | Swift | RN | Node | Python | Rust | Go | Dart | PHP | Ruby |
|---------|:--:|:----:|:-----:|:--:|:----:|:------:|:----:|:--:|:----:|:---:|:----:|
| ABI version check | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Error code propagation | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `last_error` / `last_error_msg` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Context-managed lifetime | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| README | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| CI compile check | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Smoke tests | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Golden vectors | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Sign/verify example | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Address example | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Error handling example | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

---

## Tier Classification

| Tier | Languages | SLA |
|------|-----------|-----|
| **Tier 1** | C#, Java, Swift, Python, Node.js, Rust | Full parity, CI smoke tests, golden vectors, examples |
| **Tier 2** | Go, Dart, React Native | Full parity, CI compile, examples |
| **Tier 3** | PHP, Ruby | Full parity, CI compile, best-effort tests |
| **Platforms** | Android (AAR/JNI), iOS (XCFramework/SPM), WASM | Platform-specific packaging + tests |

---

## Verification Status

### Differential Testing

| Dimension | Status |
|-----------|--------|
| Core library vs libsecp256k1 | 10 suites, 1.3M+ checks nightly |
| Binding output vs core library | All bindings call same C ABI → identical output |
| Golden vector verification | BIP-340 + RFC 6979 known-answer tests per binding |

> All bindings are thin wrappers over the same C shared library.
> Differential correctness is inherited from the core library's
> cross-library test suite against bitcoin-core/libsecp256k1.

### CI Pipeline

```
bindings.yml (every push to dev/main):
  build-capi          → Shared lib on Linux/macOS/Windows
  python              → py_compile + mypy
  nodejs              → tsc --noEmit
  csharp              → dotnet build
  java                → javac
  swift               → swiftc -typecheck
  go                  → go vet
  rust                → cargo check
  dart                → dart analyze
  php                 → php -l
  ruby                → ruby -c
  react-native        → tsc --noEmit
```

---

## Packaging Distribution

| Language | Package | Registry | Format |
|----------|---------|----------|--------|
| C# | UltrafastSecp256k1.Native | NuGet | Multi-RID .nupkg |
| Java | com.ultrafast:ufsecp | Maven Central / GitHub Packages | JAR + shaded JNI |
| Swift | UltrafastSecp256k1 | SPM (Package.swift) | XCFramework |
| React Native | react-native-ufsecp | npm | JS + .podspec + AAR |
| Node.js | ufsecp | npm | N-API addon |
| Python | ufsecp | PyPI | manylinux/macos/windows wheels |
| Rust | ufsecp | crates.io | -sys + safe wrapper |
| Go | github.com/.../ufsecp | Go modules | CGo binding |
| Dart | ufsecp | pub.dev | FFI plugin |
| PHP | ultrafast/ufsecp | Packagist | FFI extension |
| Ruby | ufsecp | RubyGems | FFI gem |
| iOS | UltrafastSecp256k1 | CocoaPods + SPM | .podspec + Package.swift |
| Android | UltrafastSecp256k1 | Maven / local AAR | prefab AAR |
| Linux | libufsecp | deb/rpm/PKGBUILD | System packages |

---

## Version Mapping

| Wrapper Version | Core Version | ABI Version | Notes |
|----------------|--------------|-------------|-------|
| 3.14.x | 3.14.x | 1 | Current stable |

Bindings MUST check `ufsecp_abi_version() >= EXPECTED_ABI` on context creation.
ABI version bumps only on binary-incompatible changes.

---

## Related Documents

| Document | Purpose |
|----------|---------|
| [BINDINGS_ERROR_MODEL.md](BINDINGS_ERROR_MODEL.md) | Error semantics across all languages |
| [BINDINGS_MEMORY_MODEL.md](BINDINGS_MEMORY_MODEL.md) | Secret handling at wrapper boundary |
| [ABI_VERSIONING.md](ABI_VERSIONING.md) | ABI compatibility policy |
| [CT_VERIFICATION.md](CT_VERIFICATION.md) | How CT safety is enforced in the C ABI |

---

*UltrafastSecp256k1 v3.14.0 — Bindings Parity Matrix*
