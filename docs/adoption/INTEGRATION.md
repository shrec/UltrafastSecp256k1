# Integration Guide

Drop-in integration for **UltrafastSecp256k1** into any C++20 project.

---

## Table of Contents

1. [CMake FetchContent (Recommended)](#cmake-fetchcontent-recommended)
2. [add_subdirectory](#add_subdirectory)
3. [vcpkg](#vcpkg)
4. [System Install](#system-install)
5. [Minimal Examples](#minimal-examples)
6. [Migration from libsecp256k1](#migration-from-libsecp256k1)
7. [Build Options Reference](#build-options-reference)

---

## CMake FetchContent (Recommended)

The simplest integration path — no manual cloning required.

```cmake
cmake_minimum_required(VERSION 3.18)
project(my_project LANGUAGES CXX)

include(FetchContent)

FetchContent_Declare(
  secp256k1_fast
  GIT_REPOSITORY https://github.com/shrec/UltrafastSecp256k1.git
  GIT_TAG        v3.3.0
)

# Disable components you don't need
set(SECP256K1_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(SECP256K1_BUILD_BENCH OFF CACHE BOOL "" FORCE)
set(SECP256K1_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(secp256k1_fast)

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE secp256k1::fast)
```

---

## add_subdirectory

If you vendor the library or use git submodules:

```bash
git submodule add https://github.com/shrec/UltrafastSecp256k1.git third_party/secp256k1_fast
```

```cmake
set(SECP256K1_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(SECP256K1_BUILD_BENCH OFF CACHE BOOL "" FORCE)
set(SECP256K1_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

add_subdirectory(third_party/secp256k1_fast)

target_link_libraries(my_app PRIVATE secp256k1::fast)
```

---

## vcpkg

A `vcpkg.json` manifest is provided in the repository root:

```json
{
  "name": "ultrafastsecp256k1",
  "version": "3.3.0"
}
```

To use from a vcpkg overlay port or after it's published:

```cmake
find_package(secp256k1-fast CONFIG REQUIRED)
target_link_libraries(my_app PRIVATE secp256k1::fastsecp256k1)
```

---

## System Install

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local
cmake --build build -j
cmake --install build
```

Then in your project:

```cmake
find_package(secp256k1-fast 3.3 CONFIG REQUIRED)
target_link_libraries(my_app PRIVATE secp256k1::fastsecp256k1)
```

Or via pkg-config:

```bash
pkg-config --cflags --libs secp256k1-fast
```

---

## Minimal Examples

### 1. Derive Public Key

```cpp
#include <secp256k1/field.hpp>
#include <secp256k1/scalar.hpp>
#include <secp256k1/point.hpp>
#include <cstdio>

using namespace secp256k1::fast;

int main() {
    auto privkey = Scalar::from_hex(
        "e8f32e723decf4051aefac8e2c93c9c5b214313817cdb01a1494b917c8436b35");

    auto pubkey = Point::generator().scalar_mul(privkey);
    auto compressed = pubkey.to_compressed();

    printf("Public key: ");
    for (auto b : compressed) printf("%02x", b);
    printf("\n");
}
```

### 2. ECDSA Sign + Verify

```cpp
#include <secp256k1/ecdsa.hpp>
#include <secp256k1/scalar.hpp>
#include <secp256k1/point.hpp>
#include <cstdio>

using namespace secp256k1::fast;

int main() {
    // Private key and message hash
    auto privkey = Scalar::from_hex(
        "e8f32e723decf4051aefac8e2c93c9c5b214313817cdb01a1494b917c8436b35");
    auto msg_hash = Scalar::from_hex(
        "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9");

    // Derive public key
    auto pubkey = Point::generator().scalar_mul(privkey);

    // Sign (RFC 6979 deterministic nonce)
    auto [r, s] = ecdsa_sign(privkey, msg_hash);

    // Verify
    bool valid = ecdsa_verify(pubkey, msg_hash, r, s);
    printf("ECDSA verify: %s\n", valid ? "PASS" : "FAIL");
}
```

### 3. Schnorr Sign + Verify (BIP-340)

```cpp
#include <secp256k1/schnorr.hpp>
#include <secp256k1/scalar.hpp>
#include <secp256k1/point.hpp>
#include <cstdio>
#include <array>

using namespace secp256k1::fast;

int main() {
    auto privkey = Scalar::from_hex(
        "e8f32e723decf4051aefac8e2c93c9c5b214313817cdb01a1494b917c8436b35");

    // BIP-340 uses 32-byte x-only pubkey
    std::array<uint8_t, 32> msg{};
    msg.fill(0x42);  // example message

    auto [sig, ok] = schnorr_sign(privkey, msg);
    if (!ok) { printf("Sign failed\n"); return 1; }

    auto pubkey = Point::generator().scalar_mul(privkey);
    bool valid = schnorr_verify(pubkey, msg, sig);
    printf("Schnorr verify: %s\n", valid ? "PASS" : "FAIL");
}
```

### 4. Point Arithmetic

```cpp
#include <secp256k1/point.hpp>
#include <secp256k1/scalar.hpp>
#include <cstdio>

using namespace secp256k1::fast;

int main() {
    auto G = Point::generator();
    auto G2 = G.dbl();                   // 2G
    auto G3 = G2.add(G);                 // 3G
    auto G5 = G3.add(G2);               // 5G

    // Same via scalar multiplication
    auto G5_check = G.scalar_mul(Scalar::from_uint64(5));

    printf("5G match: %s\n", G5.equals(G5_check) ? "YES" : "NO");
}
```

### 5. ECDH Shared Secret

```cpp
#include <secp256k1/ecdh.hpp>
#include <secp256k1/scalar.hpp>
#include <secp256k1/point.hpp>
#include <cstdio>

using namespace secp256k1::fast;

int main() {
    // Alice and Bob generate keypairs
    auto alice_priv = Scalar::from_hex(
        "0000000000000000000000000000000000000000000000000000000000000001");
    auto bob_priv = Scalar::from_hex(
        "0000000000000000000000000000000000000000000000000000000000000002");

    auto alice_pub = Point::generator().scalar_mul(alice_priv);
    auto bob_pub = Point::generator().scalar_mul(bob_priv);

    // Shared secret: Alice's priv × Bob's pub == Bob's priv × Alice's pub
    auto shared_a = ecdh(alice_priv, bob_pub);
    auto shared_b = ecdh(bob_priv, alice_pub);

    printf("ECDH match: %s\n",
           (shared_a == shared_b) ? "YES" : "NO");
}
```

---

## Migration from libsecp256k1

### Conceptual Mapping

| libsecp256k1 (C) | UltrafastSecp256k1 (C++) |
|---|---|
| `secp256k1_context_create(SECP256K1_CONTEXT_NONE)` | No context needed — stateless API |
| `secp256k1_ec_pubkey_create(ctx, &pub, seckey)` | `Point::generator().scalar_mul(Scalar::from_bytes(seckey))` |
| `secp256k1_ecdsa_sign(ctx, &sig, msg, seckey, ...)` | `auto [r,s] = ecdsa_sign(privkey, msg_hash)` |
| `secp256k1_ecdsa_verify(ctx, &sig, msg, &pub)` | `ecdsa_verify(pubkey, msg_hash, r, s)` |
| `secp256k1_schnorrsig_sign32(ctx, sig, msg, &kp, aux)` | `auto [sig,ok] = schnorr_sign(privkey, msg)` |
| `secp256k1_schnorrsig_verify(ctx, sig, msg, len, &xpub)` | `schnorr_verify(pubkey, msg, sig)` |
| `secp256k1_ec_pubkey_serialize(ctx, out, &len, &pub, flags)` | `pubkey.to_compressed()` / `pubkey.to_uncompressed()` |
| `secp256k1_ec_pubkey_parse(ctx, &pub, in, len)` | `Point::from_compressed(bytes)` / `Point::from_uncompressed(bytes)` |

### Key Differences

1. **No context objects**: UltrafastSecp256k1 is entirely stateless. No `create`/`destroy` boilerplate.
2. **C++20 value types**: `FieldElement`, `Scalar`, `Point` are regular value types with copy/move semantics.
3. **Structured bindings**: Sign functions return `auto [r, s]` or `auto [sig, ok]` via `std::pair`/`std::tuple`.
4. **Hex I/O built-in**: `from_hex()` / `to_hex()` on all types — no manual byte array wrangling.
5. **No flags**: Compression is chosen by the serialization function, not a flag parameter.

### Drop-in Compatibility Shim

For projects that need a C API compatible with libsecp256k1, see [`compat/libsecp256k1_shim/`](../../compat/libsecp256k1_shim/) — a thin C wrapper that maps the libsecp256k1 API to UltrafastSecp256k1 internals.

---

## Build Options Reference

| Option | Default | Description |
|---|---|---|
| `SECP256K1_BUILD_CPU` | `ON` | Build CPU implementation |
| `SECP256K1_BUILD_CUDA` | `OFF` | Build CUDA GPU backend |
| `SECP256K1_BUILD_ROCM` | `OFF` | Build ROCm/HIP (AMD GPU) backend |
| `SECP256K1_BUILD_OPENCL` | `OFF` | Build OpenCL backend |
| `SECP256K1_BUILD_METAL` | `OFF` | Build Apple Metal backend |
| `SECP256K1_BUILD_TESTS` | `ON` | Build test suite |
| `SECP256K1_BUILD_BENCH` | `ON` | Build benchmarks |
| `SECP256K1_BUILD_EXAMPLES` | `ON` | Build example programs |
| `SECP256K1_USE_ASM` | `ON` | Enable assembly optimizations |
| `SECP256K1_SPEED_FIRST` | `OFF` | Prioritize speed (skip runtime safety checks) |
| `SECP256K1_BUILD_SHARED` | `OFF` | Build shared library instead of static |
| `SECP256K1_INSTALL` | `ON` | Generate install targets |
| `CMAKE_CUDA_ARCHITECTURES` | `86;89` | GPU compute capabilities (GPU builds) |

---

## Minimum Compiler Requirements

| Compiler | Minimum | Recommended |
|---|---|---|
| GCC | 11 | 13+ |
| Clang/LLVM | 15 | 19+ |
| MSVC | 2022 (17.0) | Latest |
| AppleClang | 15 | Latest |
| NVIDIA nvcc | 12.0 | 12.4+ |

All require C++20 support (`-std=c++20` / `/std:c++20`).
