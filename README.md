# UltrafastSecp256k1

Ultra high-performance secp256k1 elliptic curve cryptography library with multi-platform support.

[![GitHub stars](https://img.shields.io/github/stars/shrec/UltrafastSecp256k1?style=social)](https://github.com/shrec/UltrafastSecp256k1/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/shrec/UltrafastSecp256k1?style=social)](https://github.com/shrec/UltrafastSecp256k1/network/members)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](https://en.cppreference.com/w/cpp/20)
[![CUDA](https://img.shields.io/badge/CUDA-12.0+-green.svg)](https://developer.nvidia.com/cuda-toolkit)

## 🚀 Features

- **Multi-Platform Architecture**
  - CPU: Optimized for x86-64 (BMI2/ADX) and RISC-V (RV64GC)
  - GPU: CUDA acceleration for batch operations
  - Future: OpenCL support planned

- **Performance**
  - x86-64: 3-5× speedup with BMI2/ADX assembly
  - RISC-V: 2-3× speedup with native assembly
  - CUDA: Batch processing of thousands of operations in parallel
  - Memory-mapped database support for large-scale lookups

- **Features**
  - Complete secp256k1 field and scalar arithmetic
  - Point addition, doubling, and multiplication
  - GLV endomorphism optimization
  - Efficient batch operations
  - Signature verification (ECDSA)
  - Public key derivation

## 📦 Use Cases

- **Cryptocurrency Applications**
  - Bitcoin/Ethereum address generation
  - Transaction signing and verification
  - Hardware wallet integration
  - Bulk address validation

- **Cryptographic Research**
  - ECC algorithm testing
  - Performance benchmarking
  - Custom curve implementations

- **General Purpose**
  - Any application requiring secp256k1 operations
  - High-throughput cryptographic services
  - Embedded systems (RISC-V support)

## 🔐 Security Model

UltrafastSecp256k1 is primarily a performance-focused secp256k1 engine designed for high-throughput elliptic curve operations across multiple platforms.

By default, this library prioritizes speed and research flexibility.

⚠️ **Constant-time behavior is NOT guaranteed unless explicitly built in hardened mode.**

Two security profiles are planned and partially implemented:

### FAST Profile (Default)

* Optimized for maximum performance
* May use variable-time algorithms
* Intended for:
  * Public-key operations
  * Verification workloads
  * Research and benchmarking
  * Closed systems where side-channel exposure is not a concern

### CT / HARDENED Profile (Planned / Ongoing)

* Intended for secret scalar operations (private key handling, signing)
* Will enforce:
  * Constant-time arithmetic paths
  * Side-channel resistant table access
  * Optional Montgomery ladder implementations
  * Restricted compiler flags (no `-Ofast`, no unsafe math optimizations)

Users are responsible for selecting the appropriate build configuration for their use case.

This project does **not** encourage misuse and does **not** include brute-force or attack logic.
It is a mathematical cryptographic engine intended for legitimate cryptographic applications.

## 🛠️ Building

### Prerequisites

- CMake 3.18+
- C++20 compiler (GCC 11+, Clang/LLVM 15+)
  - MSVC 2022+ (optional, disabled by default - use `-DSECP256K1_ALLOW_MSVC=ON`)
- CUDA Toolkit 12.0+ (optional, for GPU support)
- Ninja (recommended)

### CPU-Only Build

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

### With CUDA Support

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DSECP256K1_BUILD_CUDA=ON
cmake --build build -j
```

### Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `SECP256K1_USE_ASM` | ON | Enable assembly optimizations (x64/RISC-V) |
| `SECP256K1_BUILD_CUDA` | OFF | Build CUDA GPU support |
| `SECP256K1_BUILD_OPENCL` | OFF | Build OpenCL support (future) |
| `SECP256K1_BUILD_TESTS` | ON | Build test suite |
| `SECP256K1_BUILD_BENCH` | ON | Build benchmarks |
| `SECP256K1_RISCV_FAST_REDUCTION` | ON | Fast modular reduction (RISC-V) |
| `SECP256K1_RISCV_USE_VECTOR` | ON | RVV vector extension (RISC-V) |

### Build Profiles

UltrafastSecp256k1 is designed with two conceptual build targets:

#### 1️⃣ FAST (Performance Research Mode)

* Maximum throughput
* Aggressive compiler optimizations allowed
* Suitable for:
  * Benchmarking
  * Public key generation
  * Batch verification
  * High-performance research environments

#### 2️⃣ CT (Constant-Time Hardened Mode)

* Secret-dependent branches avoided
* Deterministic execution paths
* Safer for:
  * Private key operations
  * Signing workflows
  * External-facing cryptographic services

CT mode is under continuous development and will be expanded with:

* Montgomery ladder options
* Constant-time table selection
* Optional blinding techniques
* Timing regression testing integration

## 🎯 Quick Start

### Basic CPU Usage

```cpp
#include <secp256k1/field.hpp>
#include <secp256k1/point.hpp>
#include <secp256k1/scalar.hpp>
#include <iostream>

using namespace secp256k1::fast;

int main() {
    // 1. Field arithmetic
    auto a = FieldElement::from_hex(
        "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141"
    );
    auto b = FieldElement::from_hex(
        "1234567890ABCDEF1234567890ABCDEF1234567890ABCDEF1234567890ABCDEF"
    );
    
    auto sum = a + b;
    auto product = a * b;
    auto inverse = a.inverse();
    
    std::cout << "Sum: " << sum.to_hex() << "\n";
    std::cout << "Product: " << product.to_hex() << "\n";
    
    // 2. Point operations (public key derivation)
    auto generator = Point::generator();
    auto private_key = Scalar::from_hex(
        "E9873D79C6D87DC0FB6A5778633389F4453213303DA61F20BD67FC233AA33262"
    );
    
    // Multiply generator by private key
    auto public_key = generator * private_key;
    
    std::cout << "Public Key X: " << public_key.x().to_hex() << "\n";
    std::cout << "Public Key Y: " << public_key.y().to_hex() << "\n";
    
    // 3. Point addition
    auto point1 = Point::from_coordinates(
        FieldElement::from_hex("..."),
        FieldElement::from_hex("...")
    );
    auto point2 = Point::from_coordinates(
        FieldElement::from_hex("..."),
        FieldElement::from_hex("...")
    );
    
    auto result = point1 + point2;
    
    return 0;
}
```

**Compile & Run:**
```bash
# Link with the library
g++ -std=c++20 example.cpp -lsecp256k1-fast-cpu -o example
./example
```

### Advanced: Batch Signature Verification

```cpp
#include <secp256k1/point.hpp>
#include <secp256k1/scalar.hpp>
#include <vector>

using namespace secp256k1::fast;

bool verify_signatures_batch(
    const std::vector<Point>& public_keys,
    const std::vector<std::array<uint8_t, 32>>& messages,
    const std::vector<Scalar>& r_values,
    const std::vector<Scalar>& s_values
) {
    auto generator = Point::generator();
    
    for (size_t i = 0; i < public_keys.size(); ++i) {
        // Hash message
        auto msg_hash = Scalar::from_bytes(messages[i]);
        
        // Verify: s*G = R + hash*PubKey
        auto s_inv = s_values[i].inverse();
        auto u1 = msg_hash * s_inv;
        auto u2 = r_values[i] * s_inv;
        
        auto point = generator * u1 + public_keys[i] * u2;
        
        if (point.x().to_scalar() != r_values[i]) {
            return false;
        }
    }
    
    return true;
}
```

### CUDA GPU Acceleration

```cpp
#include <secp256k1_cuda/batch_operations.hpp>
#include <secp256k1/point.hpp>
#include <vector>

using namespace secp256k1::fast;

int main() {
    // Prepare batch data (1 million operations)
    std::vector<Point> base_points(1'000'000);
    std::vector<Scalar> scalars(1'000'000);
    
    // Fill with data...
    for (size_t i = 0; i < base_points.size(); ++i) {
        base_points[i] = Point::generator();
        scalars[i] = Scalar::random();
    }
    
    // GPU batch multiplication
    cuda::BatchConfig config{
        .device_id = 0,
        .threads_per_block = 256,
        .streams = 4
    };
    
    auto results = cuda::batch_multiply(
        base_points, 
        scalars, 
        config
    );
    
    std::cout << "Processed " << results.size() 
              << " point multiplications on GPU\n";
    
    // Results are already on host memory
    for (const auto& result : results) {
        std::cout << "Result: " << result.x().to_hex() << "\n";
    }
    
    return 0;
}
```

**Compile with CUDA:**
```bash
nvcc -std=c++20 cuda_example.cpp \
     -lsecp256k1-fast-cpu -lsecp256k1-fast-cuda \
     -o cuda_example
./cuda_example
```

### CUDA: Batch Address Generation

```cpp
#include <secp256k1_cuda/batch_operations.hpp>
#include <secp256k1_cuda/address_generator.hpp>

int main() {
    // Generate 10 million Bitcoin addresses on GPU
    std::vector<Scalar> private_keys(10'000'000);
    
    // Fill with sequential or random keys
    for (size_t i = 0; i < private_keys.size(); ++i) {
        private_keys[i] = Scalar::from_int(i + 1);
    }
    
    // GPU batch generation
    auto addresses = cuda::generate_addresses(
        private_keys,
        cuda::AddressType::P2PKH // Bitcoin P2PKH format
    );
    
    std::cout << "Generated " << addresses.size() << " addresses\n";
    
    // First few addresses
    for (size_t i = 0; i < 10; ++i) {
        std::cout << "Address " << i << ": " 
                  << addresses[i] << "\n";
    }
    
    return 0;
}
```

### Performance Tuning Example

```cpp
#include <secp256k1/field.hpp>
#include <secp256k1/field_asm.hpp>
#include <chrono>

using namespace secp256k1::fast;

void benchmark_field_multiply() {
    auto a = FieldElement::random();
    auto b = FieldElement::random();
    
    const int iterations = 1'000'000;
    
    // Warm-up
    for (int i = 0; i < 1000; ++i) {
        volatile auto result = a * b;
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; ++i) {
        volatile auto result = a * b;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
        end - start
    ).count();
    
    std::cout << "Field multiply: " 
              << (duration / iterations) << " ns/op\n";
    
    // Check if using assembly
    if (has_bmi2_support()) {
        std::cout << "Using BMI2 intrinsics: YES\n";
    }
    
#ifdef SECP256K1_HAS_ASM
    std::cout << "Using assembly: YES\n";
#else
    std::cout << "Using portable C++\n";
#endif
}
```

## 📊 Performance

Benchmarks below are from `bench_comprehensive_riscv` (Release builds).
RISC-V results were collected on **Milk-V Mars** (RV64 + RVV).

### x86_64 / Windows (Clang 21.1.0, Release)

| Operation | Time |
|-----------|------:|
| Field Mul | 32 ns |
| Field Square | 28 ns |
| Field Add | 11 ns |
| Field Sub | 12 ns |
| Field Inverse | 5 us |
| Point Add | 644 ns |
| Point Double | 313 ns |
| Point Scalar Mul | 111 us |
| Generator Mul | 7 us |
| Batch Inverse (n=100) | 145 ns |
| Batch Inverse (n=1000) | 98 ns |

### x86_64 / Linux (i5, Clang 19.1.7, AVX2, Release)

| Operation | Time |
|-----------|------:|
| Field Mul | 33 ns |
| Field Square | 32 ns |
| Field Add | 11 ns |
| Field Sub | 12 ns |
| Field Inverse | 5 μs |
| Point Add | 521 ns |
| Point Double | 278 ns |
| Point Scalar Mul | 110 μs |
| Generator Mul | 5 μs |
| Batch Inverse (n=100) | 140 ns |
| Batch Inverse (n=1000) | 92 ns |

### RISC-V 64-bit / Linux (Milk-V Mars, RVV, Clang 21.1.8, Release)

| Operation | Time |
|-----------|------:|
| Field Mul | 197 ns |
| Field Square | 177 ns |
| Field Add | 35 ns |
| Field Sub | 31 ns |
| Field Inverse | 18 μs |
| Point Add | 3 μs |
| Point Double | 1 μs |
| Point Scalar Mul | 672 μs |
| Generator Mul | 40 μs |
| Batch Inverse (n=100) | 765 ns |
| Batch Inverse (n=1000) | 615 ns |

*See [RISCV_OPTIMIZATIONS.md](RISCV_OPTIMIZATIONS.md) for optimization details.*

### CUDA (NVIDIA RTX 5060 Ti)

| Operation | Throughput |
|-----------|------------|
| Scalar Mul (G×k batch) | ~1.86M ops/s |
| Field Multiplication | ~4.1 Gops/s |

*Note: CUDA performance depends heavily on batch size and GPU occupancy. Results from batch processing millions of operations.*

## 🏗️ Architecture

```
secp256k1-fast/
├── cpu/                 # CPU-optimized implementation
│   ├── include/         # Public headers
│   ├── src/            # Implementation
│   │   ├── field.cpp           # Field arithmetic
│   │   ├── scalar.cpp          # Scalar arithmetic
│   │   ├── point.cpp           # Point operations
│   │   ├── field_asm_x64.asm   # x64 assembly
│   │   ├── field_asm_x64_gas.S # x64 GAS syntax
│   │   └── field_asm_riscv64.S # RISC-V assembly
│   └── tests/          # Unit tests
├── cuda/               # CUDA GPU acceleration
│   ├── include/        # CUDA headers
│   ├── src/           # CUDA kernels
│   └── tests/         # CUDA tests
└── opencl/            # OpenCL support (future)
```

## 🔬 Research Statement

This project represents over a year of research and several months of implementation effort focused on:

* Efficient field arithmetic
* Optimized scalar multiplication
* Cross-architecture portability
* Clean dependency-free design

The goal is to explore the practical performance limits of secp256k1 implementations while maintaining responsible cryptographic engineering practices.

Community feedback, peer review, and constructive criticism are welcome.

## 📚 Variant Overview

The different internal variants represent experimental optimization stages and research branches.

### `secp256k1_32_fast`

Early performance-focused implementation.
Optimized for speed without strict constant-time guarantees.

### `secp256k1_32_hybrid_smart`

Introduces algorithmic balance between raw speed and structural clarity.
Used for experimentation with mixed optimization strategies.

### `secp256k1_32_hybrid_final`

Refined hybrid approach with improved arithmetic consistency and cleaner internal flow.

### `secp256k1_32_really_final`

Stable experimental branch consolidating performance refinements.
Represents the most mature 32-bit variant in the research series.

These variants are part of ongoing development and benchmarking.
Detailed performance comparisons will be added progressively.

## 🚫 Clarification on Usage

UltrafastSecp256k1 is a cryptographic arithmetic library.

It does not implement:

* Brute-force search engines
* Key cracking tools
* Wallet recovery systems
* Attack frameworks

It provides optimized elliptic curve math primitives.

Any higher-level usage is entirely external to this repository.

## 📚 Documentation

- [API Reference](docs/api.md)
- [Build Guide](docs/building.md)
- [Performance Tuning](docs/performance.md)
- [Platform Support](docs/platforms.md)
- [Contributing](CONTRIBUTING.md)

## 🧪 Testing

```bash
# Run all tests
ctest --test-dir build --output-on-failure

# Run benchmarks
./build/cpu/bench/benchmark_field
./build/cpu/bench/benchmark_point
./build/cuda/tests/cuda_benchmark
```

## 🤝 Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
git clone https://github.com/shrec/Secp256K1fast.git
cd UltrafastSecp256k1
cmake -S . -B build-dev -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build build-dev -j
```

## 📄 License

This project is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**.

### Open Source License

The library is free to use under AGPL-3.0 for open source projects. This means:
- ✅ You can use, modify, and distribute the code
- ✅ You must disclose your source code
- ✅ You must license your project under AGPL-3.0 or compatible license
- ✅ You must provide network access to your source code if you run it as a service

See [LICENSE](LICENSE) for full details.

### Commercial License

**For commercial/proprietary use without AGPL-3.0 obligations:**

If you want to use this library in a proprietary/closed-source product or service without disclosing your source code, please contact us for a commercial license.

📧 **Contact for commercial licensing:**
- Email: [payysoon@gmail.com](mailto:payysoon@gmail.com)
- GitHub: https://github.com/shrec/Secp256K1fast

We offer flexible licensing options for commercial applications.

## 🙏 Acknowledgments

- Based on optimized secp256k1 implementations
- Inspired by Bitcoin Core's libsecp256k1
- RISC-V assembly contributions
- CUDA kernel optimizations

## 📧 Contact

- Issues: [GitHub Issues](https://github.com/shrec/Secp256K1fast/issues)
- Discussions: [GitHub Discussions](https://github.com/shrec/Secp256K1fast/discussions)

## ☕ Support the Project

If you find this library useful, consider buying me a coffee!

[![PayPal](https://img.shields.io/badge/PayPal-Donate-blue.svg?logo=paypal)](https://paypal.me/IChkheidze)

**PayPal:** [paypal.me/IChkheidze](https://paypal.me/IChkheidze)

Your support helps maintain and improve this project. Thank you! 🙏

---

**UltrafastSecp256k1** - Ultra high-performance elliptic curve cryptography for modern hardware.
