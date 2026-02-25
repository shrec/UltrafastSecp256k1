# UltrafastSecp256k1 Documentation

> **Version 3.12.1** -- Cross-platform secp256k1 ECC library

---

## Quick Links

| Document | Description |
|----------|-------------|
| [API Reference](API_REFERENCE.md) | Complete CPU + CUDA + WASM function reference |
| [Building](BUILDING.md) | Build instructions for all 10+ platforms |
| [Benchmarks](BENCHMARKS.md) | Performance data: CPU, GPU, embedded, mobile |
| [ESP32 Setup](ESP32_SETUP.md) | ESP32-S3/PICO-D4 flashing & testing guide |
| [RISC-V Optimizations](../RISCV_OPTIMIZATIONS.md) | RISC-V assembly & RVV details |
| [Porting Guide](../PORTING.md) | Add new platforms, architectures, GPU backends |

## Security & Audit

| Document | Description |
|----------|-------------|
| [Audit Guide](../AUDIT_GUIDE.md) | **Start here** -- Auditor navigation, checklist, reproduction commands |
| [Architecture](ARCHITECTURE.md) | Technical architecture deep-dive for auditors |
| [CT Verification](CT_VERIFICATION.md) | Constant-time methodology, dudect, known limitations |
| [Test Matrix](TEST_MATRIX.md) | Function -> test coverage map with gap analysis |
| [Security Policy](../SECURITY.md) | Vulnerability reporting, audit status, production readiness |
| [Threat Model](../THREAT_MODEL.md) | Layer-by-layer risk + attack surface analysis |
| [Audit Report](../AUDIT_REPORT.md) | Internal audit: 641,194 checks, 8 suites, 0 failures |

## Adoption & Integration

| Document | Description |
|----------|-------------|
| [Integration Guide](adoption/INTEGRATION.md) | CMake FetchContent, add_subdirectory, vcpkg, migration from libsecp256k1 |
| [API Stability](adoption/API_STABILITY.md) | Header stability tiers: Stable / Provisional / Experimental |
| [Backend Guide](adoption/BACKENDS.md) | CPU, CUDA, ROCm, OpenCL, Metal, WASM, Android matrix |
| [libsecp256k1 Shim](../compat/libsecp256k1_shim/README.md) | Drop-in C API compatibility layer |

## External Docs

| Document | Description |
|----------|-------------|
| [CUDA / ROCm GPU](../cuda/README.md) | CUDA + HIP/ROCm architecture, kernels, benchmarks |
| [WebAssembly](../wasm/README.md) | WASM build, JS/TS API, npm package |
| [Contributing](../CONTRIBUTING.md) | Development workflow, coding standards, PR process |
| [Security](../SECURITY.md) | Vulnerability reporting, security model |
| [Changelog](../CHANGELOG.md) | Version history |

---

## Getting Started

### 1. Build

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

### 2. Self-Test

```bash
ctest --test-dir build --output-on-failure
```

### 3. Use in Your Code

```cpp
#include <secp256k1/field.hpp>
#include <secp256k1/point.hpp>
#include <secp256k1/scalar.hpp>
#include <secp256k1/ecdsa.hpp>
#include <secp256k1/schnorr.hpp>
#include <secp256k1/sha256.hpp>

using namespace secp256k1::fast;

int main() {
    // Key generation
    Scalar private_key = Scalar::from_hex(
        "E9873D79C6D87DC0FB6A5778633389F4453213303DA61F20BD67FC233AA33262"
    );
    Point public_key = Point::generator().scalar_mul(private_key);

    // ECDSA sign / verify
    std::array<uint8_t, 32> msg_hash = sha256::hash("hello", 5);
    auto [r, s] = ecdsa::sign(msg_hash, private_key);
    bool ok = ecdsa::verify(msg_hash, public_key, r, s);

    return ok ? 0 : 1;
}
```

---

## Architecture Overview

```
UltrafastSecp256k1/
+-- cpu/                         # CPU library (C++20, header-only + compiled)
|   +-- include/secp256k1/       # Public headers
|   |   +-- field.hpp            #   Field element (mod p)
|   |   +-- scalar.hpp           #   Scalar (mod n)
|   |   +-- point.hpp            #   EC point operations
|   |   +-- ecdsa.hpp            #   ECDSA sign/verify (RFC 6979)
|   |   +-- schnorr.hpp          #   Schnorr BIP-340 sign/verify
|   |   +-- sha256.hpp           #   SHA-256 hash
|   |   +-- glv.hpp              #   GLV endomorphism
|   |   +-- precompute.hpp       #   Generator table
|   |   +-- ct/                  #   Constant-time variants
|   |   +-- types.hpp            #   Cross-backend POD types
|   +-- src/                     # Implementation + platform ASM
|   |   +-- field.cpp
|   |   +-- field_asm_x64.asm    #   x86-64 BMI2/ADX
|   |   +-- field_asm_riscv64.S  #   RISC-V RV64GC + RVV
|   |   +-- field_asm_arm64.cpp  #   ARM64 MUL/UMULH
|   |   +-- ecdsa.cpp
|   |   +-- schnorr.cpp
|   |   +-- ...
|   +-- tests/                   # CTest unit tests
|   +-- bench/                   # Benchmarks
|   +-- fuzz/                    # libFuzzer harnesses
|
+-- cuda/                        # CUDA + ROCm/HIP GPU library
|   +-- include/
|   |   +-- secp256k1.cuh        #   All device functions (field/point/scalar)
|   |   +-- ptx_math.cuh         #   PTX inline asm (with __int128 fallback)
|   |   +-- gpu_compat.h         #   CUDA <-> HIP API mapping
|   |   +-- batch_inversion.cuh  #   Montgomery trick batch inverse
|   |   +-- bloom.cuh            #   Device-side Bloom filter
|   |   +-- hash160.cuh          #   SHA-256 + RIPEMD-160
|   +-- app/                     #   Experimental search kernels
|   +-- src/                     #   Kernel wrappers, tests, benchmarks
|
+-- opencl/                      # OpenCL GPU library
|   +-- kernels/                 #   .cl kernel sources
|   +-- ...
|
+-- wasm/                        # WebAssembly (Emscripten)
|   +-- secp256k1_wasm.h         #   C API (11 functions)
|   +-- secp256k1_wasm.cpp       #   Implementation
|   +-- secp256k1.mjs            #   JS wrapper
|   +-- secp256k1.d.ts           #   TypeScript declarations
|   +-- package.json             #   npm: @ultrafastsecp256k1/wasm
|
+-- examples/
|   +-- basic_usage/             #   Desktop C++ example
|   +-- esp32_test/              #   ESP32-S3 / ESP32-PICO-D4
|   +-- stm32_test/              #   STM32F103ZET6 ARM Cortex-M3
|
+-- cmake/
|   +-- version.hpp.in           #   Auto-generated version header
|   +-- ios.toolchain.cmake      #   iOS cross-compilation toolchain
|
+-- scripts/
|   +-- build_wasm.sh            #   Emscripten WASM build
|   +-- build_xcframework.sh     #   iOS XCFramework build
|
+-- .github/workflows/
|   +-- ci.yml                   #   CI: Linux/Win/macOS/iOS/WASM/Android/ROCm
|   +-- docs.yml                 #   Doxygen -> GitHub Pages
|
+-- Package.swift                # Swift Package Manager
+-- UltrafastSecp256k1.podspec   # CocoaPods
+-- Doxyfile                     # Doxygen config
+-- CMakeLists.txt               # Top-level CMake (v3.0.0)
```

## Supported Platforms

| Platform | Architecture | Assembly | Status |
|----------|-------------|----------|--------|
| Linux x86-64 | BMI2/ADX | x86-64 ASM | [OK] Production |
| Windows x86-64 | BMI2/ADX | x86-64 ASM | [OK] Production |
| macOS x86-64 / ARM64 | Native | ARM64 ASM | [OK] Production |
| RISC-V 64 | RV64GC + RVV | RISC-V ASM | [OK] Production |
| Android ARM64 | Cortex-A55/A76 | ARM64 ASM | [OK] Production |
| iOS 17+ | Apple Silicon | ARM64 ASM | [OK] CI (testers wanted) |
| CUDA (sm_75+) | PTX | PTX inline | [OK] Production |
| ROCm / HIP | GCN / RDNA | Portable | [OK] CI (testers wanted) |
| OpenCL 3.0 | PTX | PTX inline | [OK] Production |
| WebAssembly | Emscripten | Portable C++ | [OK] Production |
| ESP32-S3 | Xtensa LX7 | Portable C++ | [OK] Tested |
| ESP32-PICO-D4 | Xtensa LX6 | Portable C++ | [OK] Tested |
| STM32F103 | Cortex-M3 | ARM Thumb ASM | [OK] Tested |

---

## License

AGPL-3.0 -- See [LICENSE](../LICENSE)

Commercial licensing available -- contact [payysoon@gmail.com](mailto:payysoon@gmail.com)

