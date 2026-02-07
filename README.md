# UltrafastSecp256k1

Ultra high-performance secp256k1 elliptic curve cryptography library with multi-platform support.

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](https://en.cppreference.com/w/cpp/20)
[![CUDA](https://img.shields.io/badge/CUDA-12.0+-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![GitHub](https://img.shields.io/badge/GitHub-shrec%2FUltrafastSecp256k1-blue)](https://github.com/shrec/UltrafastSecp256k1)

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

### x86-64 (AMD Zen 5, 5.7 GHz)

| Operation | Portable C++ | BMI2 Intrinsics | Assembly |
|-----------|--------------|-----------------|----------|
| Field Mul | 25 ns | 12 ns | 8 ns |
| Point Mul | 45 µs | 22 µs | 15 µs |

### RISC-V (StarFive JH7110, 1.5 GHz)

| Operation | Portable C++ | Assembly | With RVV |
|-----------|--------------|----------|----------|
| Field Mul | 180 ns | 75 ns | 22 ns |
| Point Mul | 420 µs | 165 µs | 55 µs |

### CUDA (RTX 4090)

| Batch Size | Time | Throughput |
|------------|------|------------|
| 1,000 | 0.8 ms | 1.25M ops/s |
| 1,000,000 | 125 ms | 8M ops/s |

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
git clone https://github.com/shrec/UltrafastSecp256k1.git
cd UltrafastSecp256k1
cmake -S . -B build-dev -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build build-dev -j
```

## 📄 License

This project is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**.

See [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- Based on optimized secp256k1 implementations
- Inspired by Bitcoin Core's libsecp256k1
- RISC-V assembly contributions
- CUDA kernel optimizations

## 📧 Contact

- Issues: [GitHub Issues](https://github.com/shrec/UltrafastSecp256k1/issues)
- Discussions: [GitHub Discussions](https://github.com/shrec/UltrafastSecp256k1/discussions)

## 🌟 Related Projects

- [Bitcoin Core libsecp256k1](https://github.com/bitcoin-core/secp256k1)
- [OpenSSL](https://www.openssl.org/)
- [GMP](https://gmplib.org/)

---

**UltrafastSecp256k1** - Ultra high-performance elliptic curve cryptography for modern hardware.
