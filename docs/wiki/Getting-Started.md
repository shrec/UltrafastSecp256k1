# Getting Started

This guide will help you build and use UltrafastSecp256k1 in your project.

## Prerequisites

- **CMake** 3.18+
- **C++20 compiler**:
  - GCC 11+ 
  - Clang/LLVM 15+ (recommended)
  - MSVC 2022+ (with `-DSECP256K1_ALLOW_MSVC=ON`)
- **Ninja** (recommended) or Make
- **CUDA 12.0+** (optional, for GPU support)

## Quick Build

### CPU Only

```bash
# Clone
git clone --recursive https://github.com/shrec/UltrafastSecp256k1.git
cd UltrafastSecp256k1

# Configure
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build build -j

# Test
ctest --test-dir build --output-on-failure
```

### With CUDA

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DSECP256K1_BUILD_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="86;89"

cmake --build build -j
```

## First Program

Create `example.cpp`:

```cpp
#include <secp256k1/point.hpp>
#include <secp256k1/scalar.hpp>
#include <iostream>
#include <iomanip>

using namespace secp256k1::fast;

int main() {
    // Run self-test first
    if (!Selftest(false)) {
        std::cerr << "Self-test failed!" << std::endl;
        return 1;
    }
    
    // Create private key from hex
    Scalar private_key = Scalar::from_hex(
        "E9873D79C6D87DC0FB6A5778633389F4453213303DA61F20BD67FC233AA33262"
    );
    
    // Generate public key: public_key = private_key × G
    Point G = Point::generator();
    Point public_key = G.scalar_mul(private_key);
    
    // Get compressed public key (33 bytes)
    auto compressed = public_key.to_compressed();
    
    // Print as hex
    std::cout << "Public Key: ";
    for (auto byte : compressed) {
        std::cout << std::hex << std::setfill('0') << std::setw(2) << (int)byte;
    }
    std::cout << std::endl;
    
    return 0;
}
```

Compile and run:

```bash
g++ -std=c++20 -O3 example.cpp -I build/cpu/include -L build/cpu -lfastsecp256k1 -o example
./example
```

## CMake Integration

### As Subdirectory

```cmake
add_subdirectory(UltrafastSecp256k1)
target_link_libraries(your_target PRIVATE secp256k1::fast)
```

### As Installed Package

```cmake
find_package(secp256k1-fast REQUIRED)
target_link_libraries(your_target PRIVATE secp256k1::fastsecp256k1)
```

### pkg-config

```bash
pkg-config --cflags --libs secp256k1-fast
```

## Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `SECP256K1_USE_ASM` | ON | Assembly optimizations |
| `SECP256K1_BUILD_CUDA` | OFF | CUDA GPU support |
| `SECP256K1_BUILD_OPENCL` | OFF | OpenCL GPU support |
| `SECP256K1_BUILD_METAL` | OFF | Metal GPU support (macOS) |
| `SECP256K1_BUILD_ROCM` | OFF | ROCm GPU support (AMD) |
| `SECP256K1_BUILD_TESTS` | ON | Build tests |
| `SECP256K1_BUILD_BENCH` | ON | Build benchmarks |
| `SECP256K1_USE_LTO` | OFF | Link-Time Optimization |
| `SECP256K1_SPEED_FIRST` | OFF | Aggressive optimizations |
| `SECP256K1_ALLOW_MSVC` | OFF | Allow MSVC 2022+ compiler |

## Next Steps

- [[API Reference]] - Learn the API
- [[Examples]] - More code examples
- [[Benchmarks]] - Performance data

