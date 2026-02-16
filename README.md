# UltrafastSecp256k1

Ultra high-performance secp256k1 elliptic curve cryptography library with multi-platform support.

[![GitHub stars](https://img.shields.io/github/stars/shrec/UltrafastSecp256k1?style=social)](https://github.com/shrec/UltrafastSecp256k1/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/shrec/UltrafastSecp256k1?style=social)](https://github.com/shrec/UltrafastSecp256k1/network/members)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](https://en.cppreference.com/w/cpp/20)
[![CUDA](https://img.shields.io/badge/CUDA-12.0+-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![OpenCL](https://img.shields.io/badge/OpenCL-3.0-green.svg)](https://www.khronos.org/opencl/)
[![RISC-V](https://img.shields.io/badge/RISC--V-RV64GC-orange.svg)](https://riscv.org/)
[![ARM64](https://img.shields.io/badge/ARM64-Cortex--A55%2FA76-orange.svg)](https://developer.android.com/ndk)
[![ESP32-S3](https://img.shields.io/badge/ESP32--S3-Xtensa%20LX7-orange.svg)](https://www.espressif.com/en/products/socs/esp32-s3)
[![ESP32](https://img.shields.io/badge/ESP32-Xtensa%20LX6-orange.svg)](https://www.espressif.com/en/products/socs/esp32)
[![STM32](https://img.shields.io/badge/STM32-Cortex--M3-orange.svg)](https://www.st.com/en/microcontrollers-microprocessors/stm32f103ze.html)
[![WebAssembly](https://img.shields.io/badge/WebAssembly-Emscripten-purple.svg)](wasm/)
[![iOS](https://img.shields.io/badge/iOS-17%2B%20XCFramework-lightgrey.svg)](cmake/ios.toolchain.cmake)
[![ROCm](https://img.shields.io/badge/ROCm-6.3%20HIP-red.svg)](cuda/README.md)
[![Android](https://img.shields.io/badge/Android-NDK%20r27-brightgreen.svg)](android/)

## ⚠️ Security Notice

**Research & Development Project - Not Audited**

This library has **not undergone independent security audits**. It is provided for research, educational, and experimental purposes.

**Production Use:**
- ❌ Not recommended without independent cryptographic audit
- ❌ No formal security guarantees
- ✅ All self-tests pass (76/76 including all backends)
- ✅ Constant-time (CT) layer available for side-channel resistance

**Reporting Security Issues:**
- Email: [payysoon@gmail.com](mailto:payysoon@gmail.com)
- GitHub Issues: [UltrafastSecp256k1/issues](https://github.com/shrec/UltrafastSecp256k1/issues)

**Disclaimer:**
Users assume all risks. For production cryptographic systems, prefer audited libraries like [libsecp256k1](https://github.com/bitcoin-core/secp256k1).

---

## 🚀 Features

- **Multi-Platform Architecture**
  - CPU: Optimized for x86-64 (BMI2/ADX), RISC-V (RV64GC), and ARM64 (MUL/UMULH)
  - Mobile: Android ARM64 (NDK r27, Clang 18) + iOS 17+ (XCFramework, SPM, CocoaPods)
  - WebAssembly: Emscripten ES6 module with TypeScript declarations
  - Embedded: ESP32-S3 (Xtensa LX7) + ESP32-PICO-D4 (Xtensa LX6) + STM32F103 (ARM Cortex-M3)
  - GPU/CUDA: Batch operations with 4.63M kG/s throughput
  - GPU/ROCm (HIP): Portable PTX→__int128 fallbacks for AMD GPUs
  - GPU/OpenCL: PTX inline asm, 3.39M kG/s

- **Performance**
  - x86-64: 3-5× speedup with BMI2/ADX assembly
  - ARM64: ~5× speedup with MUL/UMULH inline assembly
  - RISC-V: 2-3× speedup with native assembly
  - CUDA: Batch processing of thousands of operations in parallel

- **Features**
  - Complete secp256k1 field and scalar arithmetic
  - Point addition, doubling, and multiplication
  - GLV endomorphism optimization
  - Efficient batch operations
  - ECDSA sign/verify (RFC 6979 deterministic nonce, low-S)
  - Schnorr BIP-340 sign/verify
  - SHA-256 hashing
  - Constant-time (CT) layer for side-channel resistance
  - Public key derivation

### Feature Coverage (v3.2.0)

| Category | Component | Status |
|----------|-----------|--------|
| **Core** | Field, Scalar, Point, GLV, Precompute | ✅ |
| **Assembly** | x64 MASM/GAS, BMI2/ADX, RISC-V | ✅ |
| **SIMD** | AVX2/AVX-512 batch ops, Montgomery batch inverse | ✅ |
| **CT** | Constant-time field/scalar/point | ✅ |
| **ECDSA** | Sign/Verify, RFC 6979, DER/Compact, low-S | ✅ |
| **Schnorr** | BIP-340 sign/verify | ✅ |
| **Recovery** | ECDSA pubkey recovery (recid) | ✅ |
| **ECDH** | Key exchange (raw, xonly, SHA-256) | ✅ |
| **Multi-scalar** | Strauss/Shamir | ✅ |
| **Batch verify** | ECDSA + Schnorr batch | ✅ |
| **BIP-32** | HD derivation, path parsing, xprv/xpub | ✅ |
| **MuSig2** | BIP-327, key aggregation, 2-round | ✅ |
| **Taproot** | BIP-341/342, tweak, Merkle | ✅ |
| **Pedersen** | Commitments, homomorphic, switch | ✅ |
| **FROST** | Threshold signatures, t-of-n | ✅ |
| **Adaptor** | Schnorr + ECDSA adaptor sigs | ✅ |
| **Address** | P2PKH, P2WPKH, P2TR, Base58, Bech32/m | ✅ |
| **Silent Pay** | BIP-352 | ✅ |
| **Hashing** | SHA-256, SHA-512, HMAC, Keccak-256 | ✅ |
| **Coins** | 27 coins, auto-dispatch, EIP-55 | ✅ |
| **Custom G** | CurveContext, custom generator/curve | ✅ |
| **BIP-44** | Coin-type HD, auto-purpose | ✅ |
| **Self-test** | Known vector verification | ✅ |
| **GPU** | CUDA kernels, occupancy | ✅ |
| **Platforms** | x64, ARM64, RISC-V, ESP32, WASM, iOS, Android, ROCm | ✅ |

## � Batch Modular Inverse (Montgomery Trick)

All backends include **batch modular inversion** — a critical building block for Jacobian→Affine conversion and high-throughput point operations:

| Backend | File | Function(s) |
|---------|------|-------------|
| **CPU** | `cpu/src/field.cpp` | `fe_batch_inverse(FieldElement*, size_t)` — Montgomery trick with scratch buffer |
| **CPU** | `cpu/src/precompute.cpp` | `batch_inverse(std::vector<FieldElement>&)` — vector variant |
| **CUDA** | `cuda/include/batch_inversion.cuh` | `batch_inverse_montgomery` — GPU Montgomery trick kernel |
| **CUDA** | `cuda/include/batch_inversion.cuh` | `batch_inverse_fermat` — Fermat's little theorem variant |
| **CUDA** | `cuda/include/batch_inversion.cuh` | `batch_inverse_kernel` — production kernel (`__launch_bounds__(256, 4)`) |
| **CUDA** | `cuda/src/test_suite.cu` | `fe_batch_inverse()` — host wrapper + unit tests |
| **Metal** | `metal/shaders/secp256k1_kernels.metal` | `batch_inverse` — chunked Montgomery inverse (parallel threadgroups) |

**Algorithm**: Montgomery batch inverse computes N field inversions using only **1 modular inversion + 3(N−1) multiplications**, amortizing the expensive inversion across the entire batch.
## ⚡ Mixed Addition (Jacobian + Affine)

The library provides **branchless mixed addition** (`add_mixed_inplace`) — the fastest way to add a point with known affine coordinates (Z=1) to a Jacobian point. Uses the **madd-2007-bl** formula (7M + 4S, vs 11M + 5S for full Jacobian add).

| Backend | File | Function |
|---------|------|----------|
| **CPU** | `cpu/src/point.cpp` | `jacobian_add_mixed(JacobianPoint&, AffinePoint&)` |
| **CPU** | `cpu/src/point.cpp` | `Point::add_mixed_inplace(FieldElement&, FieldElement&)` |
| **CPU** | `cpu/src/point.cpp` | `Point::sub_mixed_inplace(FieldElement&, FieldElement&)` |
| **CPU** | `cpu/src/precompute.cpp` | `jacobian_add_mixed_local(JacobianPoint&, AffinePointPacked&)` |
| **OpenCL** | `opencl/kernels/secp256k1_point.cl` | `point_add_mixed_impl(JacobianPoint*, AffinePoint*)` |
| **Metal** | `metal/shaders/secp256k1_point.h` | `jacobian_add_mixed(JacobianPoint&, AffinePoint&)` |

### Usage Example (CPU)

```cpp
#include <secp256k1/point.hpp>

using namespace secp256k1::fast;

// Start with generator point G
Point P = Point::generator();

// Get affine coordinates of G for mixed addition
FieldElement gx = P.x();
FieldElement gy = P.y();

// Compute 2G using mixed add (Jacobian + Affine, 7M + 4S)
Point Q = Point::generator();
Q.add_mixed_inplace(gx, gy);  // Q = G + G = 2G

// Subtraction variant: Q = Q - G
Q.sub_mixed_inplace(gx, gy);  // Q = 2G - G = G

// Batch walk: P, P+G, P+2G, ... using repeated mixed add
Point walker = P;
for (int i = 0; i < 1000; ++i) {
    walker.add_mixed_inplace(gx, gy);  // walker += G each step
    // ... process walker ...
}
```
### Mixed Add + Batch Inverse: Collecting Z Values for Cheap Jacobian→Affine

During serial mixed additions, each point accumulates a growing Z coordinate.
To extract affine X for comparison, you need Z⁻² — which requires an expensive modular inversion.
**Solution**: Collect Z values in a batch, then invert them all at once with Montgomery trick (1 inversion + 3N multiplications instead of N inversions).

```cpp
#include <secp256k1/point.hpp>
#include <secp256k1/field.hpp>

using namespace secp256k1::fast;

constexpr size_t BATCH_SIZE = 1024;

// Buffers (allocate once, reuse)
Point batch_points[BATCH_SIZE];
FieldElement batch_z[BATCH_SIZE];

// Start from some point P
Point walker = Point::generator();
FieldElement gx = walker.x();
FieldElement gy = walker.y();

size_t idx = 0;

for (uint64_t j = 0; j < total_count; ++j) {
    // Save point and its Z coordinate
    batch_points[idx] = walker;
    batch_z[idx] = walker.z();
    idx++;

    // Advance walker using mixed add (7M + 4S)
    walker.add_mixed_inplace(gx, gy);

    // When batch is full — do batch inversion
    if (idx == BATCH_SIZE) {
        // ONE modular inversion for 1024 points!
        fe_batch_inverse(batch_z.data(), idx);

        // Now batch_z[i] contains Z_i^(-1)
        for (size_t i = 0; i < idx; ++i) {
            FieldElement z_inv_sq = batch_z[i].square();         // Z^(-2)
            FieldElement x_affine = batch_points[i].X() * z_inv_sq;  // X_affine = X_jac * Z^(-2)
            // Use x_affine as needed
        }
        idx = 0;  // Reset batch
    }
}
```

**Performance**: For N=1024 batch, this is **~500× cheaper** than individual inversions. A single field inversion costs ~3.5μs (Fermat), while batch amortizes to ~7ns per element.

### GPU Pattern: H-Product Serial Inversion (`jacobian_add_mixed_h`)

Production GPU apps use a more memory-efficient variant: instead of storing full Z coordinates,
`jacobian_add_mixed_h` returns **H = U2 − X1** separately from each addition. Since Z_{k} = Z_0 · H_0 · H_1 · … · H_{k-1},
we can reconstruct and invert the entire Z chain from just the H values + initial Z_0.

**Step 1 — Collect H values during serial additions** (CUDA kernel):
```cuda
// jacobian_add_mixed_h: madd-2004-hmv (8M+3S), outputs H separately
// H = U2 - X1, and internally computes Z3 = Z1 * H
__device__ void jacobian_add_mixed_h(
    const JacobianPoint* p, const AffinePoint* q,
    JacobianPoint* r, FieldElement& h_out);

// --- Step kernel: add G repeatedly, save X and H at each slot ---
FieldElement h;
win_z0[tid] = P.z;                    // Save initial Z_0

for (int slot = 0; slot < batch_interval; ++slot) {
    win_x[tid + slot * stride] = P.x; // Save Jacobian X
    jacobian_add_mixed_h(&P, &G, &P, h);
    win_h[tid + slot * stride] = h;   // Save H (not Z!)
}
```

**Step 2 — Serial Z chain inversion** (1 Fermat inversion per thread):
```cuda
// Forward: reconstruct Z_final = Z_0 * H_0 * H_1 * ... * H_{N-1}
FieldElement z_current = z0_values[tid];
for (int slot = 0; slot < batch_interval; ++slot) {
    z_current = z_current * h_array[tid + slot * stride];
}

// ONE inversion of Z_final (Fermat: 255 sqr + 16 mul)
FieldElement z_inv = field_inverse(z_current);

// Backward: unwind to get Z_slot^{-2} at each position
for (int slot = batch_interval - 1; slot >= 0; --slot) {
    int idx = tid + slot * stride;
    z_inv = z_inv * h_array[idx];     // Z_{slot}^{-1}
    h_array[idx] = z_inv * z_inv;     // Z_{slot}^{-2} (overwrite H in-place!)
}
```

**Step 3 — Affine X extraction**:
```cuda
// h_array now contains Z^{-2} at each slot
for (int slot = 0; slot < batch_interval; ++slot) {
    int idx = tid + slot * stride;
    FieldElement x_affine = win_x[idx] * h_array[idx];  // X_jac * Z^{-2}
    // Use x_affine as needed
}
```

**Why H instead of Z?**
- **Memory**: H is a single field element; Z would also be a field element, but H is computed "for free" inside the addition — no extra multiply needed
- **Serial inversion**: Z_k = Z_0 · ∏H_i, so the backward sweep naturally yields Z_k^{-1} at each step using just the stored H values
- **In-place**: H array is overwritten with Z^{-2} — zero extra memory allocation
- **Cost**: 1 Fermat inversion + 2N multiplications per thread (vs N Fermat inversions naively)

> See production usage: `apps/secp256k1_search_gpu_only/gpu_only.cu` (step kernel) + `unified_split.cuh` (batch inversion kernel)

### Other Batch Inverse Use Cases

#### 1. Full Point Conversion: Jacobian → Affine (X + Y)

When you need both X and Y (precompute table, serialization, debugging):

```cpp
// N Jacobian points → N Affine points (1 inversion)
FieldElement z_values[N];
for (size_t i = 0; i < N; ++i)
    z_values[i] = points[i].z();

fe_batch_inverse(z_values.data(), N);  // z_values[i] = Z_i^(-1)

for (size_t i = 0; i < N; ++i) {
    FieldElement z_inv = z_values[i];
    FieldElement z2 = z_inv.square();          // Z^(-2)
    FieldElement z3 = z2 * z_inv;              // Z^(-3)
    affine_x[i] = points[i].X() * z2;         // X_affine = X_jac · Z^(-2)
    affine_y[i] = points[i].Y() * z3;         // Y_affine = Y_jac · Z^(-3)
}
```

#### 2. X-Only Coordinate Extraction

In most cases you don't need Y — only the affine X coordinate is required:

```cpp
// CPU pattern
constexpr size_t BATCH_SIZE = 1024;
Point batch_points[BATCH_SIZE];
FieldElement batch_z[BATCH_SIZE];
size_t batch_idx = 0;

for (uint64_t j = start; j < end; ++j) {
    batch_points[batch_idx] = p;
    batch_z[batch_idx] = p.z();
    batch_idx++;
    p.next_inplace();

    if (batch_idx == BATCH_SIZE || j == end - 1) {
        fe_batch_inverse(batch_z.data(), batch_idx);  // 1 inversion!

        for (size_t i = 0; i < batch_idx; ++i) {
            FieldElement z_inv_sq = batch_z[i].square();           // Z^(-2)
            FieldElement x_affine = batch_points[i].X() * z_inv_sq;  // X only!
            // Use x_affine as needed
        }
        batch_idx = 0;
    }
}
```

#### 3. CUDA: Z Extraction → batch_inverse_kernel → Affine X

On GPU where you have an array of `JacobianPoint` — Z coordinates are extracted separately, inversion uses shared memory:

```cuda
// Step 1: Extract Z coordinates (1 kernel)
__global__ void extract_z_kernel(const JacobianPoint* points,
                                 FieldElement* zs, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) zs[idx] = points[idx].z;
}

// Step 2: Montgomery batch inverse (shared memory prefix/suffix scan)
//         1 inversion per block, inner elements use multiplications only
batch_inverse_kernel<<<blocks, 256, shared_mem>>>(d_zs, d_inv_zs, N);

// Step 3: Affine X = X_jac * Z_inv²
__global__ void affine_extraction_kernel(const JacobianPoint* points,
                                         const FieldElement* inv_zs, ...) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    FieldElement z_inv = inv_zs[idx];
    FieldElement z2;
    field_sqr(&z_inv, &z2);           // Z^(-2)
    FieldElement x_aff;
    field_mul(&points[idx].x, &z2, &x_aff);  // X_affine
    // Use x_aff as needed
}
```

#### 4. Batch Modular Division: a[i] / b[i]

Arbitrary batch division for field elements:

```cpp
FieldElement denominators[] = {b0, b1, b2, b3};
fe_batch_inverse(denominators, 4);
// denominators[i] = b_i^(-1)
FieldElement r0 = a0 * denominators[0];  // a0 / b0
FieldElement r1 = a1 * denominators[1];  // a1 / b1
FieldElement r2 = a2 * denominators[2];  // a2 / b2
FieldElement r3 = a3 * denominators[3];  // a3 / b3
```

#### 5. Scratch Buffer Reuse

When processing multiple rounds, a single pre-allocated scratch buffer is reused across all rounds:

```cpp
std::vector<FieldElement> scratch;
scratch.reserve(BATCH_SIZE);  // Allocate once

for (int round = 0; round < total_rounds; ++round) {
    // ... fill batch_z[] ...
    fe_batch_inverse(batch_z.data(), N, scratch);  // Reuses scratch buffer
    // ... affine conversion ...
}
```

### Montgomery Trick — Full Algorithm Explanation

```
Input: [a₀, a₁, a₂, ..., aₙ₋₁]

1) Forward pass — cumulative products:
   prod[0] = a₀
   prod[1] = a₀ · a₁
   prod[2] = a₀ · a₁ · a₂
   ...
   prod[N-1] = a₀ · a₁ · ... · aₙ₋₁

2) Single inversion:
   inv = prod[N-1]⁻¹ = (a₀ · a₁ · ... · aₙ₋₁)⁻¹

3) Backward pass — extract individual inverses:
   aₙ₋₁⁻¹ = inv · prod[N-2]
   inv ← inv · aₙ₋₁(original)
   aₙ₋₂⁻¹ = inv · prod[N-3]
   inv ← inv · aₙ₋₂(original)
   ...
   a₀⁻¹ = inv

Cost: 1 inversion + 3(N-1) multiplications
N=1024: 1×3.5μs + 3069×5ns ≈ 18.8μs (vs 1024×3.5μs = 3584μs → 190× faster!)
```

## �📦 Use Cases

> ### ⚠️ Testers Wanted
> We need community testers for platforms we cannot fully validate in CI:
> - **iOS** — Build & run on real iPhone/iPad hardware with Xcode
> - **AMD GPU (ROCm/HIP)** — Test on AMD Radeon RX / Instinct GPUs
>
> If you can help, please [open an issue](https://github.com/shrec/UltrafastSecp256k1/issues) with your results!

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

Choose the appropriate profile for your use case.

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

### WebAssembly (Emscripten)

```bash
# Requires Emscripten SDK (emsdk)
./scripts/build_wasm.sh        # → build-wasm/dist/
```

Output: `secp256k1_wasm.wasm` + `secp256k1.mjs` (ES6 module with TypeScript types). See [wasm/README.md](wasm/README.md) for JS/TS usage.

### iOS (XCFramework)

```bash
./scripts/build_xcframework.sh  # → build-xcframework/output/
```

Produces a universal XCFramework (arm64 device + arm64 simulator). Also available via **Swift Package Manager** and **CocoaPods**.

### Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `SECP256K1_USE_ASM` | ON | Enable assembly optimizations (x64/RISC-V) |
| `SECP256K1_BUILD_CUDA` | OFF | Build CUDA GPU support |
| `SECP256K1_BUILD_OPENCL` | OFF | Build OpenCL GPU support |
| `SECP256K1_BUILD_ROCM` | OFF | Build ROCm/HIP GPU support (AMD) |
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
| Field Mul | 173 ns |
| Field Square | 160 ns |
| Field Add | 38 ns |
| Field Sub | 34 ns |
| Field Inverse | 17 μs |
| Point Add | 3 μs |
| Point Double | 1 μs |
| Point Scalar Mul | 621 μs |
| Generator Mul | 37 μs |
| Batch Inverse (n=100) | 695 ns |
| Batch Inverse (n=1000) | 547 ns |

*See [RISCV_OPTIMIZATIONS.md](RISCV_OPTIMIZATIONS.md) for optimization details.*

### ESP32-S3 / Embedded (Xtensa LX7 @ 240 MHz, ESP-IDF v5.5.1, -O3)

| Operation | Time |
|-----------|------:|
| Field Mul | 7,458 ns |
| Field Square | 7,592 ns |
| Field Add | 636 ns |
| Field Inv | 844 μs |
| Scalar × G (Generator Mul) | 2,483 μs |

*Portable C++ (no `__int128`, no assembly). All 35 library tests pass. See [examples/esp32_test/](examples/esp32_test/) for details.*

### ESP32-PICO-D4 / Embedded (Xtensa LX6 Dual Core @ 240 MHz, ESP-IDF v5.5.1, -O3)

| Operation | Time |
|-----------|------:|
| Field Mul | 6,993 ns |
| Field Square | 6,247 ns |
| Field Add | 985 ns |
| Field Inv | 609 μs |
| Scalar × G (Generator Mul) | 6,203 μs |
| CT Scalar × G | 44,810 μs |
| CT Add (complete) | 249,672 ns |
| CT Dbl | 87,113 ns |
| CT/Fast ratio | 6.5× |

*Portable C++ (no `__int128`, no assembly). All 35 self-tests + 8 CT tests pass. See [examples/esp32_test/](examples/esp32_test/) for details.*

### STM32F103ZET6 / Embedded (ARM Cortex-M3 @ 72 MHz, GCC 13.3.1, -O3)

| Operation | Time |
|-----------|------:|
| Field Mul | 15,331 ns |
| Field Square | 12,083 ns |
| Field Add | 4,139 ns |
| Field Inv | 1,645 μs |
| Scalar × G (Generator Mul) | 37,982 μs |

*ARM Cortex-M3 inline assembly (UMULL/ADDS/ADCS) for multiply/squaring/reduction. Portable C++ for field add/sub. All 35 library tests pass. See [examples/stm32_test/](examples/stm32_test/) for details.*

### Android ARM64 (RK3588, Cortex-A55/A76 @ 2.4 GHz, NDK r27 Clang 18, -O3)

| Operation | Time |
|-----------|------:|
| Field Mul | 85 ns |
| Field Square | 66 ns |
| Field Add | 18 ns |
| Field Sub | 16 ns |
| Field Inverse | 2,621 ns |
| Scalar Mul | 105 ns |
| Scalar Add | 12 ns |
| Point Add | 9,329 ns |
| Point Double | 8,711 ns |
| Fast Scalar × G (Generator Mul) | 7.6 μs |
| Fast Scalar × P (Non-Generator) | 77.6 μs |
| CT Scalar × G | 545 μs |
| CT ECDH | 545 μs |

*ARM64 inline assembly (MUL/UMULH) for field mul/sqr/add/sub/neg. ~5× faster than generic C++. All 12 Android tests pass. See [android/](android/) for details.*

### Embedded Cross-Platform Comparison

| Operation | ESP32-S3 LX7 (240 MHz) | ESP32 LX6 (240 MHz) | STM32F103 (72 MHz) |
|-----------|-------------------:|-------------------:|-------------------:|
| Field Mul | 7,458 ns | 6,993 ns | 15,331 ns |
| Field Square | 7,592 ns | 6,247 ns | 12,083 ns |
| Field Add | 636 ns | 985 ns | 4,139 ns |
| Field Inv | 844 μs | 609 μs | 1,645 μs |
| Scalar × G | 2,483 μs | 6,203 μs | 37,982 μs |

*Clock-Normalized = (STM32 time × 72) / (ESP32 time × 240). Values < 1.0 mean STM32 is faster per-clock.*

### CUDA (NVIDIA RTX 5060 Ti) — Kernel-Only

| Operation | Time/Op | Throughput |
|-----------|---------|------------|
| Field Mul | 0.2 ns | 4,139 M/s |
| Field Add | 0.2 ns | 4,122 M/s |
| Field Inv | 12.1 ns | 82.65 M/s |
| Point Add | 1.1 ns | 916 M/s |
| Point Double | 0.7 ns | 1,352 M/s |
| Scalar Mul (P×k) | 266.5 ns | 3.75 M/s |
| Generator Mul (G×k) | 216.1 ns | 4.63 M/s |

*CUDA 12.0, sm_86;sm_89, batch=1M, RTX 5060 Ti (36 SMs, 2602 MHz)*

### OpenCL (NVIDIA RTX 5060 Ti) — Kernel-Only

| Operation | Time/Op | Throughput |
|-----------|---------|------------|
| Field Mul | 0.2 ns | 4,137 M/s |
| Field Add | 0.2 ns | 4,124 M/s |
| Field Sqr | 0.2 ns | 5,985 M/s |
| Field Inv | 14.3 ns | 69.97 M/s |
| Point Add | 1.6 ns | 630.6 M/s |
| Point Double | 0.9 ns | 1,139 M/s |
| kG (Generator Mul) | 295.1 ns | 3.39 M/s |

*OpenCL 3.0 CUDA, Driver 580.126.09, PTX inline asm, batch=256K–1M*

### CUDA vs OpenCL — Kernel-Only Comparison (RTX 5060 Ti)

| Operation | CUDA | OpenCL | Faster |
|-----------|------|--------|--------|
| Field Mul | 0.2 ns | 0.2 ns | Tie |
| Field Add | 0.2 ns | 0.2 ns | Tie |
| Field Inv | 12.1 ns | 14.3 ns | CUDA 1.18× |
| Point Double | 0.7 ns | 0.9 ns | **CUDA 1.29×** |
| Point Add | 1.1 ns | 1.6 ns | **CUDA 1.45×** |
| kG (Generator Mul) | 216.1 ns | 295.1 ns | **CUDA 1.37×** |

> **Note:** Both measurements are kernel-only (no buffer allocation/copy overhead). CUDA uses local-variable optimization for zero pointer-aliasing overhead.

*Benchmarks: 2026-02-14, Linux x86_64, NVIDIA Driver 580.126.09*

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
├── opencl/            # OpenCL GPU acceleration
│   ├── kernels/       # OpenCL kernel sources (.cl)
│   ├── include/       # OpenCL headers
│   ├── src/           # Host-side OpenCL code
│   └── tests/         # OpenCL tests
└── examples/
    ├── esp32_test/    # ESP32-S3 Xtensa LX7 port
    └── stm32_test/    # STM32F103ZET6 ARM Cortex-M3 port
```

## 🔬 Research Statement

This library explores the performance ceiling of secp256k1 across CPU architectures (x64, ARM64, RISC-V, Cortex-M, Xtensa) and GPUs (CUDA, OpenCL, Metal, ROCm). Zero external dependencies.

## 📚 Variant Overview

Internal 32-bit arithmetic variants (historical optimization stages):

| Variant | Description |
|---------|-------------|
| `secp256k1_32_fast` | Speed-first, variable-time |
| `secp256k1_32_hybrid_smart` | Mixed strategy experiments |
| `secp256k1_32_hybrid_final` | Stabilized hybrid arithmetic |
| `secp256k1_32_really_final` | Most mature 32-bit variant |

## 🚫 Scope

This is an ECC arithmetic library. It provides field/scalar/point operations.
It does not include key cracking, wallet recovery, or attack tools.

## 📚 Documentation

- [Documentation Index](docs/README.md)
- [API Reference](docs/API_REFERENCE.md)
- [Build Guide](docs/BUILDING.md)
- [Benchmarks](docs/BENCHMARKS.md)
- [Contributing](CONTRIBUTING.md)
- [Security Policy](SECURITY.md)
- [Changelog](CHANGELOG.md)

## 🧪 Testing

### Built-in Selftest

The library includes a comprehensive self-test (`Selftest()`) that runs **deterministic KAT vectors** covering all arithmetic operations. Every test/bench executable runs this selftest on startup.

### Three Modes

| Mode | Time | When | What |
|------|------|------|------|
| **smoke** | ~1-2s | App startup, embedded | Core KAT (10 scalar mul, field/scalar identities, point ops, batch inverse, boundary vectors) |
| **ci** | ~30-90s | Every push (CI) | Smoke + cross-checks, bilinearity, NAF/wNAF, batch sweeps, fast-vs-generic, algebraic stress |
| **stress** | ~10-60min | Nightly / manual | CI + 1000 random scalar muls, 500 field triples, 100 bilinearity pairs, batch inverse up to 8192 |

```cpp
#include "secp256k1/selftest.hpp"
using namespace secp256k1::fast;

// Legacy (runs ci mode):
Selftest(true);

// Explicit mode + seed:
Selftest(true, SelftestMode::smoke);              // Fast startup check
Selftest(true, SelftestMode::ci);                  // Full CI suite
Selftest(true, SelftestMode::stress, 0xDEADBEEF); // Nightly with custom seed
```

### Repro Bundle

On verbose output, selftest prints everything needed to reproduce a failure:

```
  Mode:     ci
  Seed:     0x53454350324b3147
  Compiler: Clang 17.0.6
  Platform: Linux x64
  Build:    Release
  ASM:      enabled
  Repro:    Selftest(true, SelftestMode::ci, 0x53454350324b3147)
```

### Sanitizer Builds

```bash
# ASan + UBSan (catches UB, out-of-bounds, use-after-free)
cmake --preset cpu-asan
cmake --build build/cpu-asan -j
ctest --test-dir build/cpu-asan --output-on-failure

# TSan (catches data races in multi-threaded code)
cmake --preset cpu-tsan
cmake --build build/cpu-tsan -j
ctest --test-dir build/cpu-tsan --output-on-failure
```

### Running Tests

```bash
# Build and run all tests (ci mode)
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSECP256K1_BUILD_TESTS=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

### Platform Coverage Dashboard

| Platform | Backend | Compiler | Selftest CI | Stress | Notes |
|----------|---------|----------|-------------|--------|-------|
| Linux x64 | CPU | GCC 13 | ✅ CI | - | Debug + Release |
| Linux x64 | CPU | Clang 17 | ✅ CI | - | Debug + Release |
| Linux x64 | CPU | Clang 17 (ASan+UBSan) | ✅ CI | - | Sanitizer build |
| Linux x64 | CPU | Clang 17 (TSan) | ✅ CI | - | Thread sanitizer |
| Windows x64 | CPU | MSVC 2022 | ✅ CI | - | Release |
| macOS ARM64 | CPU + Metal | AppleClang | ✅ CI | - | Apple Silicon |
| macOS ARM64 | Metal GPU | AppleClang | ✅ CI | - | GPU shader tests |
| iOS ARM64 | CPU | Xcode | ✅ CI | - | Device + Simulator |
| Android ARM64 | CPU | NDK r27c | ✅ CI | - | arm64-v8a |
| WebAssembly | CPU | Emscripten | ✅ CI | - | Compile-only |
| ROCm/HIP | CPU + GPU | ROCm 6.3 | ✅ CI | - | Compile + CPU test |

> Community-tested platforms: if you run selftest on a new platform, submit the log via PR and we'll add a row.

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
- GitHub: https://github.com/shrec/UltrafastSecp256k1

We offer flexible licensing options for commercial applications.

## 🙏 Acknowledgments

- Based on optimized secp256k1 implementations
- Inspired by Bitcoin Core's libsecp256k1
- RISC-V assembly contributions
- CUDA kernel optimizations

## 📧 Contact

- Issues: [GitHub Issues](https://github.com/shrec/UltrafastSecp256k1/issues)
- Discussions: [GitHub Discussions](https://github.com/shrec/UltrafastSecp256k1/discussions)

## ☕ Support the Project

If you find this library useful, consider buying me a coffee!

[![PayPal](https://img.shields.io/badge/PayPal-Donate-blue.svg?logo=paypal)](https://paypal.me/IChkheidze)

**PayPal:** [paypal.me/IChkheidze](https://paypal.me/IChkheidze)

Your support helps maintain and improve this project. Thank you! 🙏

---

**UltrafastSecp256k1** - Ultra high-performance elliptic curve cryptography for modern hardware.
