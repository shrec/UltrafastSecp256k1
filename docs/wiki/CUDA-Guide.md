# CUDA Guide

Guide for the CUDA GPU implementation of UltrafastSecp256k1.

---

## Requirements

- **CUDA Toolkit** 12.0 or later
- **GPU**: NVIDIA with Compute Capability 7.5+ (Turing or newer)
- **Driver**: 525.60+ (Linux) or 528.02+ (Windows)

### Supported GPUs

| Architecture | Compute Capability | GPUs |
|--------------|-------------------|------|
| Turing | 7.5 | RTX 2060-2080, T4 |
| Ampere | 8.0, 8.6 | A100, RTX 3060-3090 |
| Ada Lovelace | 8.9 | RTX 4060-4090, L4, L40 |
| Hopper | 9.0 | H100 |

---

## Building

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DSECP256K1_BUILD_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="86;89"

cmake --build build -j
```

### Architecture Selection

Set `CMAKE_CUDA_ARCHITECTURES` for your GPU:

```bash
# RTX 3090
-DCMAKE_CUDA_ARCHITECTURES=86

# RTX 4090
-DCMAKE_CUDA_ARCHITECTURES=89

# Multiple GPUs
-DCMAKE_CUDA_ARCHITECTURES="86;89"
```

---

## API Overview

### Header

```cpp
#include <secp256k1.cuh>

using namespace secp256k1::cuda;
```

### Data Structures

```cpp
// Field element (4 × 64-bit limbs)
struct FieldElement {
    uint64_t limbs[4];
};

// Scalar (4 × 64-bit limbs)
struct Scalar {
    uint64_t limbs[4];
};

// Jacobian point (X, Y, Z coordinates)
struct JacobianPoint {
    FieldElement x;
    FieldElement y;
    FieldElement z;
    bool infinity;
};

// Affine point (x, y coordinates)
struct AffinePoint {
    FieldElement x;
    FieldElement y;
};
```

---

## Device Functions

All functions are `__device__` and can only be called from GPU kernels.

### Field Operations

```cpp
// Initialization
__device__ void field_set_zero(FieldElement* r);
__device__ void field_set_one(FieldElement* r);

// Arithmetic
__device__ void field_add(const FieldElement* a, const FieldElement* b, FieldElement* r);
__device__ void field_sub(const FieldElement* a, const FieldElement* b, FieldElement* r);
__device__ void field_mul(const FieldElement* a, const FieldElement* b, FieldElement* r);
__device__ void field_sqr(const FieldElement* a, FieldElement* r);
__device__ void field_inv(const FieldElement* a, FieldElement* r);
__device__ void field_neg(const FieldElement* a, FieldElement* r);

// Comparison
__device__ bool field_is_zero(const FieldElement* a);
__device__ bool field_eq(const FieldElement* a, const FieldElement* b);
```

### Point Operations

```cpp
// Initialization
__device__ void jacobian_set_infinity(JacobianPoint* p);
__device__ void jacobian_set_generator(JacobianPoint* p);

// Arithmetic
__device__ void jacobian_add(const JacobianPoint* p, const JacobianPoint* q, JacobianPoint* r);
__device__ void jacobian_add_mixed(const JacobianPoint* p, const AffinePoint* q, JacobianPoint* r);
__device__ void jacobian_double(const JacobianPoint* p, JacobianPoint* r);

// Scalar multiplication
__device__ void scalar_mul(const JacobianPoint* p, const Scalar* k, JacobianPoint* r);
__device__ void scalar_mul_generator(const Scalar* k, JacobianPoint* r);

// Conversion
__device__ void jacobian_to_affine(const JacobianPoint* p, AffinePoint* r);
__device__ bool jacobian_is_infinity(const JacobianPoint* p);
```

### Hash Operations

```cpp
#include <hash160.cuh>

// HASH160 = RIPEMD160(SHA256(pubkey))
__device__ void hash160_compressed(const uint8_t pubkey[33], uint8_t hash[20]);
__device__ void hash160_uncompressed(const uint8_t pubkey[65], uint8_t hash[20]);
```

---

## Example: Batch Key Generation

```cpp
#include <secp256k1.cuh>
#include <cuda_runtime.h>

using namespace secp256k1::cuda;

__global__ void generate_keys_kernel(
    const Scalar* private_keys,
    AffinePoint* public_keys,
    int count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    
    JacobianPoint jac;
    scalar_mul_generator(&private_keys[idx], &jac);
    jacobian_to_affine(&jac, &public_keys[idx]);
}

void generate_keys(
    const Scalar* d_private_keys,
    AffinePoint* d_public_keys,
    int count
) {
    int threads = 256;
    int blocks = (count + threads - 1) / threads;
    
    generate_keys_kernel<<<blocks, threads>>>(
        d_private_keys,
        d_public_keys,
        count
    );
    cudaDeviceSynchronize();
}
```

---

## Configuration Macros

### Compile-Time Options

| Macro | Default | Description |
|-------|---------|-------------|
| `SECP256K1_CUDA_USE_HYBRID_MUL` | 1 | 32-bit hybrid multiplication (~10% faster) |
| `SECP256K1_CUDA_USE_MONTGOMERY` | 0 | Montgomery domain arithmetic |
| `SECP256K1_CUDA_LIMBS_32` | 0 | Use 8×32-bit limbs (experimental) |

### Setting Options

In CMakeLists.txt:
```cmake
add_compile_definitions(SECP256K1_CUDA_USE_HYBRID_MUL=1)
```

Or at compile time:
```bash
nvcc -DSECP256K1_CUDA_USE_HYBRID_MUL=1 ...
```

---

## Optimization Tips

### 1. Batch Operations

Process thousands of keys in parallel:

```cpp
// Good: Batch processing
generate_keys<<<blocks, 256>>>(keys, results, 100000);

// Bad: Sequential
for (int i = 0; i < 100000; i++) {
    generate_keys<<<1, 1>>>(&keys[i], &results[i], 1);
}
```

### 2. Memory Coalescing

Align data structures for efficient memory access:

```cpp
// Ensure 32-byte alignment
alignas(32) AffinePoint points[N];
```

### 3. Occupancy

Choose thread counts that maximize occupancy:

```cpp
int threads = 256;  // Good default for most GPUs
int blocks = (count + threads - 1) / threads;
```

### 4. Avoid Divergence

Use branchless code where possible:

```cpp
// Branchless selection
uint64_t mask = -(uint64_t)(condition);
result = (a & mask) | (b & ~mask);
```

### 5. Minimize Host-Device Transfers

```cpp
// Good: Single transfer
cudaMemcpy(d_keys, h_keys, count * sizeof(Scalar), cudaMemcpyHostToDevice);
process_kernel<<<blocks, threads>>>(d_keys, d_results, count);
cudaMemcpy(h_results, d_results, count * sizeof(AffinePoint), cudaMemcpyDeviceToHost);

// Bad: Multiple transfers
for (int i = 0; i < count; i++) {
    cudaMemcpy(&d_keys[i], &h_keys[i], sizeof(Scalar), cudaMemcpyHostToDevice);
    // ...
}
```

---

## Hybrid 32-bit Multiplication

The default implementation uses 32-bit hybrid multiplication which is ~10% faster than pure 64-bit on most GPUs.

**How it works:**
1. Split 64-bit limbs into 32-bit halves
2. Use native PTX `mad.lo.cc.u32` instructions
3. Combine results with proven 64-bit reduction

This is controlled by `SECP256K1_CUDA_USE_HYBRID_MUL` (default: ON).

---

## Troubleshooting

### "nvcc not found"

Add CUDA to PATH:
```bash
export PATH=/usr/local/cuda/bin:$PATH
```

### "unsupported gpu architecture"

Update CMAKE_CUDA_ARCHITECTURES:
```bash
cmake -DCMAKE_CUDA_ARCHITECTURES=89 ...
```

### Out of memory

Reduce batch size or use streaming:
```cpp
for (int i = 0; i < total; i += batch_size) {
    int n = min(batch_size, total - i);
    process_batch(d_data + i, n);
}
```

---

## See Also

- [[API Reference]] - Complete function list
- [[CPU Guide]] - CPU implementation
- [[Benchmarks]] - Performance data

