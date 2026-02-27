# UltrafastSecp256k1 Wiki

Welcome to the **UltrafastSecp256k1** wiki -- an ultra high-performance secp256k1 elliptic curve cryptography library.

## Quick Navigation

| Page | Description |
|------|-------------|
| [[Getting Started]] | Installation and first steps |
| [[API Reference]] | Complete function documentation |
| [[CPU Guide]] | CPU implementation details (x86-64, ARM64, RISC-V) |
| [[Android Guide]] | Android port (ARM64, ARMv7, JNI) |
| [[CUDA Guide]] | GPU implementation and batch processing |
| [[Benchmarks]] | Performance measurements |
| [[Examples]] | Code examples and use cases |

## Key Features

- **Multi-Platform**: x86-64, ARM64, RISC-V, CUDA, OpenCL, Metal, ROCm, WASM, iOS, Android, ESP32, STM32
- **GPU Signatures**: Only open-source library with ECDSA + Schnorr sign/verify on GPU (CUDA, OpenCL, Metal)
- **Constant-Time (CT) Layer**: Side-channel resistant operations (`secp256k1::ct::`)
- **Full Protocol Suite**: ECDSA, Schnorr (BIP-340), ECDH, BIP-32, MuSig2, Taproot, FROST, Pedersen
- **Stable C ABI**: `ufsecp` FFI (45 exports) for C#, Python, Rust, Go, Java, Node.js
- **27 Blockchains**: Bitcoin, Ethereum, Litecoin, and more with auto-dispatch address generation
- **Zero Dependencies**: Pure C++20, no Boost, no OpenSSL
- **Easy Integration**: CMake, vcpkg, CocoaPods, Swift PM, pkg-config

## Dual API: Fast + Constant-Time

The library provides **two namespaces** -- always compiled, no flags needed:

| Namespace | Purpose | Use When |
|-----------|---------|----------|
| `secp256k1::fast::` | Maximum throughput | Public data, batch processing, verification |
| `secp256k1::ct::` | Side-channel resistance | Secret keys, signing, ECDH |

Both share the same data types (`FieldElement`, `Scalar`, `Point`) and are freely mixable:

```cpp
PT pub_point = PT::generator().scalar_mul(pub_k);  // fast:: for public data
PT result = secp256k1::ct::scalar_mul(pub_point, secret_k);  // ct:: for secret
```

See [[API Reference]] for the full CT API and [[Examples]] for usage patterns.

## Performance at a Glance

### CPU

| Platform | Scalar Mul (k*P) | Generator Mul (k*G) | Field Mul | Notes |
|----------|:----------------:|:-------------------:|:---------:|-------|
| x86-64 (Clang 21, AVX2) | 25 us | 5 us | 17 ns | BMI2/ADX assembly, 5x52 FE |
| ARM64 (Cortex-A76) | 131 us | 14 us | 74 ns | MUL/UMULH inline assembly |
| RISC-V (Milk-V Mars) | 154 us | 33 us | 95 ns | RV64GC + Zba/Zbb, ThinLTO |

### GPU

| Backend | Scalar Mul (k*G) | ECDSA Sign | ECDSA Verify | Schnorr Sign | Schnorr Verify |
|---------|:----------------:|:----------:|:------------:|:------------:|:--------------:|
| CUDA (RTX 5060 Ti) | 4.59 M/s | 4.88 M/s | 2.44 M/s | 3.66 M/s | 2.82 M/s |
| OpenCL (RTX 5060 Ti) | 3.39 M/s | -- | -- | -- | -- |
| Metal (M3 Pro) | 0.33 M/s | -- | -- | -- | -- |

### Embedded

| Platform | Scalar Mul (k*G) | Notes |
|----------|:----------------:|-------|
| ESP32-S3 (240 MHz) | 5.2 ms | Xtensa LX7, CT available |
| ESP32 (240 MHz) | 6.2 ms | Xtensa LX6 |
| STM32F103 (72 MHz) | 38 ms | ARM Cortex-M3 |

## Links

- [GitHub Repository](https://github.com/shrec/UltrafastSecp256k1)
- [Issue Tracker](https://github.com/shrec/UltrafastSecp256k1/issues)
- [Discord](https://discord.gg/sUmW7cc5)
- [Live Benchmark Dashboard](https://shrec.github.io/UltrafastSecp256k1/dev/bench/)

## License

MIT -- See [LICENSE](https://github.com/shrec/UltrafastSecp256k1/blob/main/LICENSE)

**Commercial support**: Contact [payysoon@gmail.com](mailto:payysoon@gmail.com) for integration consulting.

## Support the Project

[![Donate with Bitcoin Lightning](https://img.shields.io/badge/Donate%20with-Lightning%20%E2%9A%A1-yellow?style=for-the-badge&logo=bitcoin)](https://stacker.news/shrec)

**Lightning Address:** `shrec@stacker.news`

[![PayPal](https://img.shields.io/badge/PayPal-Donate-blue.svg?logo=paypal)](https://paypal.me/IChkheidze)

