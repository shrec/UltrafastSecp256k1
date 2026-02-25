# Examples

Code examples for common use cases with UltrafastSecp256k1.

---

## Table of Contents

1. [Basic Operations](#basic-operations)
2. [Key Generation](#key-generation)
3. [ECDSA Signatures](#ecdsa-signatures)
4. [Schnorr Signatures](#schnorr-signatures)
5. [Batch Processing](#batch-processing)
6. [Database Lookups](#database-lookups)
7. [CUDA Examples](#cuda-examples)

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
    
    // Public key = private_key x G
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

## ECDSA Signatures

### Sign and Verify

```cpp
#include <secp256k1/ecdsa.hpp>
#include <secp256k1/sha256.hpp>
#include <iostream>

using namespace secp256k1::fast;

int main() {
    // Private key
    Scalar seckey = Scalar::from_hex(
        "E9873D79C6D87DC0FB6A5778633389F4453213303DA61F20BD67FC233AA33262"
    );
    
    // Message hash (SHA-256 of the actual message)
    uint8_t msg[] = "Hello, UltrafastSecp256k1!";
    auto msg_hash = sha256(msg, sizeof(msg) - 1);
    
    // Sign (RFC 6979 deterministic nonces, low-S normalization)
    auto [sig_r, sig_s] = ecdsa_sign(msg_hash.data(), seckey);
    
    // Derive public key
    Point pubkey = Point::generator().scalar_mul(seckey);
    
    // Verify
    bool valid = ecdsa_verify(msg_hash.data(), pubkey, sig_r, sig_s);
    std::cout << "ECDSA verification: " << (valid ? "PASSED" : "FAILED") << std::endl;
    
    return 0;
}
```

### Recoverable Signatures (EIP-155 / Ethereum)

```cpp
#include <secp256k1/ecdsa.hpp>

using namespace secp256k1::fast;

int main() {
    Scalar seckey = Scalar::from_hex("...");
    uint8_t msg_hash[32] = { /* ... */ };
    
    // Sign with recovery ID
    auto [sig_r, sig_s, recid] = ecdsa_sign_recoverable(msg_hash, seckey);
    
    // Recover public key from signature (no private key needed)
    Point recovered = ecdsa_recover(msg_hash, sig_r, sig_s, recid);
    
    // Verify recovery
    Point original = Point::generator().scalar_mul(seckey);
    // recovered == original
    
    return 0;
}
```

---

## Schnorr Signatures

### BIP-340 Sign and Verify

```cpp
#include <secp256k1/schnorr.hpp>
#include <iostream>

using namespace secp256k1::fast;

int main() {
    Scalar seckey = Scalar::from_hex(
        "E9873D79C6D87DC0FB6A5778633389F4453213303DA61F20BD67FC233AA33262"
    );
    
    uint8_t msg[32] = { /* message hash */ };
    uint8_t aux[32] = { /* auxiliary randomness */ };
    
    // Sign (BIP-340 tagged hashing)
    auto sig = schnorr_sign(msg, seckey, aux);
    
    // Verify with x-only pubkey (32 bytes, not 33)
    auto pubkey_x = Point::generator().scalar_mul(seckey).x().to_bytes();
    bool valid = schnorr_verify(msg, pubkey_x.data(), sig);
    
    std::cout << "Schnorr verification: " << (valid ? "PASSED" : "FAILED") << std::endl;
    
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

## Constant-Time (CT) Operations

### ECDH with CT Protection

Use `ct::` for secret-dependent scalar multiplication (e.g., ECDH shared secret):

```cpp
#include <secp256k1/fast.hpp>
#include <secp256k1/ct/point.hpp>
#include <iostream>

using namespace secp256k1::fast;
namespace ct = secp256k1::ct;

int main() {
    // Alice's secret key
    Scalar alice_secret = Scalar::from_hex(
        "E9873D79C6D87DC0FB6A5778633389F4453213303DA61F20BD67FC233AA33262"
    );
    
    // Bob's public key (received over the network -- public data)
    Point bob_public = Point::from_hex(
        "D2E670A19C6D753D1A6D8B5F5D0C0E4C1A7E4F0B3E3D2A1C0B9A8E7D6C5B4A39",
        "4E7A1D5C3B2A0F9E8D7C6B5A4F3E2D1C0B9A8E7D6C5B4A3F2E1D0C9B8A7E6D5"
    );
    
    // ECDH: shared_secret = alice_secret x bob_public
    // Use CT to protect the secret scalar!
    Point shared_point = ct::scalar_mul(bob_public, alice_secret);
    
    // The x-coordinate of the shared point is the shared secret
    auto shared_secret = shared_point.x().to_bytes();
    
    std::cout << "Shared secret (x): " << shared_point.x().to_hex() << std::endl;
    return 0;
}
```

### CT Key Generation

Generate a public key from a secret key using constant-time operations:

```cpp
#include <secp256k1/fast.hpp>
#include <secp256k1/ct/point.hpp>

using namespace secp256k1::fast;
namespace ct = secp256k1::ct;

int main() {
    Scalar secret_key = Scalar::from_hex(
        "4727DAF2986A9804B1117F8261ABA645C34537E4474E19BE58700792D501A591"
    );
    
    // CT generator multiplication: public_key = secret_key x G
    Point public_key = ct::generator_mul(secret_key);
    
    // Verify the key is on the curve (also CT)
    uint64_t on_curve = ct::point_is_on_curve(public_key);
    if (on_curve) {
        auto compressed = public_key.to_compressed();
        // Use compressed public key...
    }
    
    return 0;
}
```

### Mixing fast:: and ct::

Use `fast::` for public data, `ct::` for secret-dependent operations:

```cpp
#include <secp256k1/fast.hpp>
#include <secp256k1/ct/field.hpp>
#include <secp256k1/ct/scalar.hpp>
#include <secp256k1/ct/point.hpp>

using namespace secp256k1::fast;
namespace ct = secp256k1::ct;

int main() {
    // -- Public computation (fast::) --
    // Base point is public -- use fast:: for maximum speed
    Scalar pub_k = Scalar::from_uint64(100);
    Point base_point = Point::generator().scalar_mul(pub_k);  // fast::
    
    // -- Secret computation (ct::) --
    // The scalar is secret -- switch to CT
    Scalar secret_k = Scalar::from_hex(
        "E9873D79C6D87DC0FB6A5778633389F4453213303DA61F20BD67FC233AA33262"
    );
    Point result = ct::scalar_mul(base_point, secret_k);  // ct::
    
    // -- Verification (ct::) --
    // Compare points without leaking which one matched
    Point expected = Point::generator().scalar_mul(
        Scalar::from_uint64(100) * secret_k
    );
    uint64_t eq = ct::point_eq(result, expected);
    // eq == 0xFFFFFFFFFFFFFFFF if equal, 0 otherwise
    
    return 0;
}
```

### CT Conditional Operations

Branchless conditional logic for secret-dependent control flow:

```cpp
#include <secp256k1/ct/ops.hpp>
#include <secp256k1/ct/field.hpp>
#include <secp256k1/ct/scalar.hpp>

using namespace secp256k1::fast;
namespace ct = secp256k1::ct;

void ct_conditional_example() {
    FieldElement a = FieldElement::from_uint64(42);
    FieldElement b = FieldElement::from_uint64(99);
    
    // CT select: choose a or b based on secret condition
    uint64_t secret_flag = 1;
    uint64_t mask = ct::bool_to_mask(secret_flag);  // all-ones or all-zeros
    FieldElement chosen = ct::field_select(a, b, mask);  // a if mask=1s, else b
    
    // CT conditional negate
    FieldElement maybe_neg = ct::field_cneg(a, mask);  // -a if mask=1s, else a
    
    // CT conditional swap
    ct::field_cswap(&a, &b, mask);  // swap if mask=1s
    
    // CT comparison (returns mask, not bool)
    uint64_t is_eq = ct::field_eq(a, b);  // all-ones if equal
    uint64_t is_z  = ct::field_is_zero(a);  // all-ones if zero
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

