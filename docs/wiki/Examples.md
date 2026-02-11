# Examples

Code examples for common use cases with UltrafastSecp256k1.

---

## Table of Contents

1. [Basic Operations](#basic-operations)
2. [Key Generation](#key-generation)
3. [Batch Processing](#batch-processing)
4. [Database Lookups](#database-lookups)
5. [CUDA Examples](#cuda-examples)

---

## Basic Operations

### Field Arithmetic

```cpp
#include <secp256k1/field.hpp>
#include <iostream>

using namespace secp256k1::fast;

int main() {
    // Create field elements
    auto a = FieldElement::from_hex(
        "79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798"
    );
    auto b = FieldElement::from_hex(
        "483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8"
    );
    
    // Arithmetic
    auto sum = a + b;
    auto diff = a - b;
    auto prod = a * b;
    auto sq = a.square();
    auto inv = a.inverse();
    
    // Verify: a * a^(-1) == 1
    auto check = a * inv;
    std::cout << "a * a^(-1) = " << check.to_hex() << std::endl;
    // Should be: 0000...0001
    
    return 0;
}
```

### Scalar Operations

```cpp
#include <secp256k1/scalar.hpp>
#include <iostream>

using namespace secp256k1::fast;

int main() {
    auto k1 = Scalar::from_hex(
        "E9873D79C6D87DC0FB6A5778633389F4453213303DA61F20BD67FC233AA33262"
    );
    auto k2 = Scalar::from_hex(
        "0000000000000000000000000000000000000000000000000000000000000002"
    );
    
    // Arithmetic (mod n)
    auto sum = k1 + k2;
    auto prod = k1 * k2;  // 2 * k1
    
    // Get individual bits
    for (int i = 255; i >= 0; i--) {
        std::cout << (int)k1.bit(i);
    }
    std::cout << std::endl;
    
    return 0;
}
```

---

## Key Generation

### Generate Single Key

```cpp
#include <secp256k1/point.hpp>
#include <secp256k1/scalar.hpp>
#include <iostream>
#include <iomanip>

using namespace secp256k1::fast;

void print_hex(const uint8_t* data, size_t len) {
    for (size_t i = 0; i < len; i++) {
        std::cout << std::hex << std::setfill('0') << std::setw(2) << (int)data[i];
    }
    std::cout << std::dec << std::endl;
}

int main() {
    // Private key (256-bit)
    Scalar private_key = Scalar::from_hex(
        "E9873D79C6D87DC0FB6A5778633389F4453213303DA61F20BD67FC233AA33262"
    );
    
    // Public key = private_key × G
    Point G = Point::generator();
    Point public_key = G.scalar_mul(private_key);
    
    // Compressed format (33 bytes)
    auto compressed = public_key.to_compressed();
    std::cout << "Compressed: ";
    print_hex(compressed.data(), 33);
    
    // Uncompressed format (65 bytes)
    auto uncompressed = public_key.to_uncompressed();
    std::cout << "Uncompressed: ";
    print_hex(uncompressed.data(), 65);
    
    // Individual coordinates
    std::cout << "X: " << public_key.x().to_hex() << std::endl;
    std::cout << "Y: " << public_key.y().to_hex() << std::endl;
    
    return 0;
}
```

### Generate Key Range

```cpp
#include <secp256k1/point.hpp>
#include <secp256k1/scalar.hpp>
#include <vector>

using namespace secp256k1::fast;

std::vector<Point> generate_key_range(const Scalar& start, int count) {
    std::vector<Point> keys;
    keys.reserve(count);
    
    // Start point
    Point current = Point::generator().scalar_mul(start);
    keys.push_back(current);
    
    // Increment using next_inplace (faster than scalar_mul)
    for (int i = 1; i < count; i++) {
        current.next_inplace();  // current += G
        keys.push_back(current);
    }
    
    return keys;
}

int main() {
    auto start = Scalar::from_uint64(1000000);
    auto keys = generate_key_range(start, 1000);
    
    // keys[i] corresponds to private key (start + i)
    return 0;
}
```

---

## Batch Processing

### Fixed-K Multiplication

When multiplying many points by the same scalar K:

```cpp
#include <secp256k1/point.hpp>
#include <vector>

using namespace secp256k1::fast;

int main() {
    // Fixed scalar K (e.g., from ECDH shared secret)
    Scalar K = Scalar::from_hex(
        "4727DAF2986A9804B1117F8261ABA645C34537E4474E19BE58700792D501A591"
    );
    
    // Precompute K-dependent work ONCE
    KPlan plan = KPlan::from_scalar(K);
    
    // Variable points Q (e.g., multiple public keys)
    std::vector<Point> points = {
        Point::from_hex("...", "..."),
        Point::from_hex("...", "..."),
        // ... many more
    };
    
    // Fast multiplication for each Q
    std::vector<Point> results;
    results.reserve(points.size());
    
    for (const auto& Q : points) {
        // Uses cached GLV decomposition and wNAF
        Point R = Q.scalar_mul_with_plan(plan);
        results.push_back(R);
    }
    
    return 0;
}
```

### In-Place Point Chain

```cpp
#include <secp256k1/point.hpp>

using namespace secp256k1::fast;

int main() {
    Point p = Point::generator();
    
    // Build chain: G, 2G, 3G, 4G, ...
    // Using in-place operations for speed
    for (int i = 0; i < 1000; i++) {
        // Process current point...
        auto compressed = p.to_compressed();
        
        // Move to next
        p.next_inplace();  // p += G (no allocation)
    }
    
    return 0;
}
```

---

## Database Lookups

### X-Coordinate Split Keys

```cpp
#include <secp256k1/point.hpp>
#include <cstring>

using namespace secp256k1::fast;

struct SplitKey {
    uint8_t first_half[16];   // First 16 bytes of x
    uint8_t second_half[16];  // Last 16 bytes of x
};

SplitKey make_split_key(const Point& p) {
    SplitKey key;
    auto first = p.x_first_half();
    auto second = p.x_second_half();
    std::memcpy(key.first_half, first.data(), 16);
    std::memcpy(key.second_half, second.data(), 16);
    return key;
}

int main() {
    Point p = Point::generator().scalar_mul(Scalar::from_uint64(12345));
    
    SplitKey key = make_split_key(p);
    
    // Use for database lookup
    // lookup_database(key.first_half);
    
    return 0;
}
```

---

## CUDA Examples

### Batch Key Generation

```cpp
#include <secp256k1.cuh>
#include <cuda_runtime.h>
#include <vector>

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

std::vector<AffinePoint> generate_keys_gpu(const std::vector<Scalar>& private_keys) {
    int count = private_keys.size();
    
    // Allocate device memory
    Scalar* d_private;
    AffinePoint* d_public;
    cudaMalloc(&d_private, count * sizeof(Scalar));
    cudaMalloc(&d_public, count * sizeof(AffinePoint));
    
    // Copy to device
    cudaMemcpy(d_private, private_keys.data(), 
               count * sizeof(Scalar), cudaMemcpyHostToDevice);
    
    // Launch kernel
    int threads = 256;
    int blocks = (count + threads - 1) / threads;
    generate_keys_kernel<<<blocks, threads>>>(d_private, d_public, count);
    
    // Copy results back
    std::vector<AffinePoint> public_keys(count);
    cudaMemcpy(public_keys.data(), d_public,
               count * sizeof(AffinePoint), cudaMemcpyDeviceToHost);
    
    // Cleanup
    cudaFree(d_private);
    cudaFree(d_public);
    
    return public_keys;
}
```

### Hash160 Computation

```cpp
#include <secp256k1.cuh>
#include <hash160.cuh>

using namespace secp256k1::cuda;

__global__ void compute_hash160_kernel(
    const AffinePoint* public_keys,
    uint8_t* hashes,  // 20 bytes per hash
    int count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    
    // Serialize to compressed format
    uint8_t pubkey[33];
    // ... serialize public_keys[idx] to pubkey ...
    
    // Compute HASH160
    hash160_compressed(pubkey, &hashes[idx * 20]);
}
```

---

## Self-Test

Always run self-test after building:

```cpp
#include <secp256k1/point.hpp>
#include <iostream>

int main() {
    bool ok = secp256k1::fast::Selftest(true);  // verbose
    
    if (ok) {
        std::cout << "All tests passed!" << std::endl;
        return 0;
    } else {
        std::cerr << "TESTS FAILED!" << std::endl;
        return 1;
    }
}
```

---

## See Also

- [[API Reference]] - Complete function list
- [[CPU Guide]] - CPU optimization tips
- [[CUDA Guide]] - GPU programming guide

