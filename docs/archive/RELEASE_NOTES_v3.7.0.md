## What's New in v3.7.0

### ufsecp Stable C ABI
- **45 exported C functions** with opaque `ufsecp_ctx` context
- Dual-layer constant-time protection (always-on)
- Single header: `ufsecp.h` -- covers ECDSA, Schnorr, ECDH, BIP-32, addresses, WIF, taproot
- Error codes 0-10 (`ufsecp_error_t`)

### 12 Language Bindings
All bindings use the new `ufsecp_*` context-based API:
- **Go** (CGo), **PHP** (FFI 8.1+), **Dart** (dart:ffi), **Python** (ctypes)
- **Node.js** (ffi-napi), **C#** (P/Invoke), **Rust** (sys + safe wrapper)
- **Ruby** (FFI), **Swift** (C module), **Java** (JNI)
- **React Native** (Android + iOS native modules)

### Package Manifests
Cargo.toml, composer.json, pubspec.yaml, pyproject.toml, package.json, gemspec, .csproj, Package.swift, podspec

### Release CI/CD
- **9 build platforms**: linux-x64, win-x64, macos-arm64, linux-arm64, android-arm64/armv7/x64, ios-xcframework, wasm
- **5 packaging jobs**: Python wheels, npm, NuGet, gem, Java JAR

### Core Library Improvements
- GLV + 5x52 + Shamir scalar_mul: K*Q 132->42us (3.1x faster)
- GPU signature operations (ECDSA + Schnorr) in CUDA/Metal/OpenCL
- Cross-platform CI fixes (arm64, armeabi-v7a, x86_64)
- field_52 `__int128` guard for 32-bit fallback
- libsecp256k1 compatibility shim

### Adoption Docs
- API_STABILITY.md, BACKENDS.md, INTEGRATION.md, PORTING.md
