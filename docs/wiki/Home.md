# UltrafastSecp256k1 Wiki

Welcome to the **UltrafastSecp256k1** wiki - an ultra high-performance secp256k1 elliptic curve cryptography library.

## 🚀 Quick Navigation

| Page | Description |
|------|-------------|
| [[Getting Started]] | Installation and first steps |
| [[API Reference]] | Complete function documentation |
| [[CPU Guide]] | CPU implementation details (x86-64, RISC-V) |
| [[Android Guide]] | Android port (ARM64, ARMv7, JNI) |
| [[CUDA Guide]] | GPU implementation and batch processing |
| [[Benchmarks]] | Performance measurements |
| [[Examples]] | Code examples and use cases |

## ✨ Key Features

- **Multi-Platform**: x86-64, RISC-V, Android (ARM64/ARMv7), CUDA GPU
- **High Performance**: 3-5× faster than standard libraries
- **Constant-Time (CT) Layer**: Side-channel resistant operations (`secp256k1::ct::`)
- **Production Ready**: Comprehensive test suite
- **Easy Integration**: CMake, pkg-config support

## 🔒 Dual API: Fast + Constant-Time

The library provides **two namespaces** — always compiled, no flags needed:

| Namespace | Purpose | Use When |
|-----------|---------|----------|
| `secp256k1::fast::` | Maximum throughput | Public data, batch processing |
| `secp256k1::ct::` | Side-channel resistance | Secret keys, signing, ECDH |

Both share the same data types (`FieldElement`, `Scalar`, `Point`) and are freely mixable:

```cpp
PT pub_point = PT::generator().scalar_mul(pub_k);  // fast:: for public data
PT result = secp256k1::ct::scalar_mul(pub_point, secret_k);  // ct:: for secret
```

See [[API Reference]] for the full CT API and [[Examples]] for usage patterns.

## 📊 Performance at a Glance

| Platform | Scalar Mul | Field Mul | Notes |
|----------|-----------|-----------|-------|
| x86-64 (AVX2) | 110 μs | 33 ns | BMI2/ADX assembly |
| RISC-V (RVV) | 672 μs | 198 ns | Vector optimized |
| CUDA (RTX 4090) | TBD | TBD | Batch parallel |

## 🔗 Links

- [GitHub Repository](https://github.com/shrec/Secp256K1fast)
- [Main Project](https://github.com/shrec/Secp256K1fast)
- [Issue Tracker](https://github.com/shrec/Secp256K1fast/issues)

## 📄 License

AGPL v3 - See [LICENSE](https://github.com/shrec/Secp256K1fast/blob/main/LICENSE)

## ☕ Support the Project

If you find this library useful, consider buying me a coffee!

[![PayPal](https://img.shields.io/badge/PayPal-Donate-blue.svg?logo=paypal)](https://paypal.me/IChkheidze)

**PayPal:** [paypal.me/IChkheidze](https://paypal.me/IChkheidze)

