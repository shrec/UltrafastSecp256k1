# UltrafastSecp256k1 Wiki

Welcome to the **UltrafastSecp256k1** wiki - an ultra high-performance secp256k1 elliptic curve cryptography library.

## 🚀 Quick Navigation

| Page | Description |
|------|-------------|
| [[Getting Started]] | Installation and first steps |
| [[API Reference]] | Complete function documentation |
| [[CPU Guide]] | CPU implementation details (x86-64, RISC-V) |
| [[CUDA Guide]] | GPU implementation and batch processing |
| [[Benchmarks]] | Performance measurements |
| [[Examples]] | Code examples and use cases |

## ✨ Key Features

- **Multi-Platform**: x86-64, RISC-V, CUDA GPU
- **High Performance**: 3-5× faster than standard libraries
- **Production Ready**: Comprehensive test suite
- **Easy Integration**: CMake, pkg-config support

## 📊 Performance at a Glance

| Platform | Scalar Mul | Field Mul | Notes |
|----------|-----------|-----------|-------|
| x86-64 (AVX2) | 110 μs | 33 ns | BMI2/ADX assembly |
| RISC-V (RVV) | 672 μs | 198 ns | Vector optimized |
| CUDA (RTX 4090) | TBD | TBD | Batch parallel |

## 🔗 Links

- [GitHub Repository](https://github.com/shrec/UltrafastSecp256k1)
- [Main Project](https://github.com/shrec/Secp256K1fast)
- [Issue Tracker](https://github.com/shrec/UltrafastSecp256k1/issues)

## 📄 License

AGPL v3 - See [LICENSE](https://github.com/shrec/UltrafastSecp256k1/blob/main/LICENSE)

## ☕ Support the Project

If you find this library useful, consider buying me a coffee!

[![PayPal](https://img.shields.io/badge/PayPal-Donate-blue.svg?logo=paypal)](https://paypal.me/IChkheidze)

**PayPal:** [paypal.me/IChkheidze](https://paypal.me/IChkheidze)

