# UltrafastSecp256k1 OpenCL Implementation

Cross-platform GPU acceleration for secp256k1 cryptographic operations.

## Features

- **Cross-platform GPU support**: Intel, AMD, NVIDIA GPUs via OpenCL 1.2+
- **Zero external dependencies**: Only requires OpenCL runtime
- **Full field arithmetic**: Addition, subtraction, multiplication, squaring, inversion
- **Point operations**: Doubling, addition, scalar multiplication
- **Batch operations**: High-throughput batch scalar multiplication with generator
- **Same test vectors**: Identical test suite as CPU and CUDA implementations
- **Branchless operations**: Critical operations are branchless for security

## Requirements

- **Windows**: Intel OpenCL runtime (included with Intel GPU driver) or NVIDIA/AMD drivers
- **Linux**: `ocl-icd-opencl-dev` and vendor-specific drivers
- **CMake** 3.16+ with Ninja (recommended)

### Intel GPU Drivers (Windows)

Download from: https://www.intel.com/content/www/us/en/download/785597/intel-arc-iris-xe-graphics-windows.html

### Verify OpenCL Installation

```bash
# Windows (PowerShell)
clinfo | Select-String "Platform Name|Device Name"

# Linux
clinfo | grep -E "Platform Name|Device Name"
```

## Building

```bash
# Configure (from opencl directory)
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build build

# Run tests
./build/opencl_test

# Run benchmarks
./build/opencl_benchmark
```

## API Usage

```cpp
#include "secp256k1_opencl.hpp"

using namespace secp256k1::opencl;

// Create context (auto-selects best GPU)
DeviceConfig config;
config.prefer_intel = true;
auto ctx = Context::create(config);

// Single operations
FieldElement a = field_from_u64(7);
FieldElement b = field_from_u64(11);
FieldElement c = ctx->field_mul(a, b);  // c = 77

// Scalar multiplication with generator
Scalar k = scalar_from_u64(12345);
JacobianPoint P = ctx->scalar_mul_generator(k);  // P = 12345 * G

// Batch operations (high throughput)
std::vector<Scalar> scalars(1000);
std::vector<JacobianPoint> results(1000);
ctx->batch_scalar_mul_generator(scalars.data(), results.data(), 1000);

// Convert to affine coordinates
AffinePoint affine = jacobian_to_affine(P);
```

## Architecture

```
opencl/
+-- include/secp256k1_opencl.hpp   # Main API
+-- kernels/                        # OpenCL kernel sources
+-- src/                           # Implementation
+-- tests/                         # Test suite (32+ tests)
```

## Test Vectors

Uses identical test vectors as CPU implementation. All 32+ tests must pass.
