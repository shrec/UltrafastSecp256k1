# Backend Guide

UltrafastSecp256k1 supports multiple compute backends for different hardware targets.

---

## Backend Matrix

| Backend | Directory | Status | Platforms | Requirements |
|---|---|---|---|---|
| **CPU** | `cpu/` | **Production** | x86-64, ARM64, RISC-V, ESP32, STM32 | C++20 compiler |
| **CUDA** | `cuda/` | **Production** | NVIDIA GPUs (sm_50+) | CUDA Toolkit 12.0+ |
| **ROCm/HIP** | `cuda/` (shared) | **Beta** | AMD GPUs (gfx900+) | ROCm 5.0+, CMake 3.21+ |
| **OpenCL** | `opencl/` | **Beta** | Any OpenCL 1.2+ device | OpenCL SDK or driver |
| **Metal** | `metal/` | **Experimental** | Apple Silicon (M1+) | Xcode 15+, Metal 3 |
| **WASM** | `wasm/` | **Experimental** | Browsers, Node.js | Emscripten SDK |
| **Android** | `android/` | **Experimental** | ARM64 Android devices | Android NDK r27+ |

---

## CPU Backend (Production)

The default and most mature backend. Pure C++20 with optional platform-specific assembly.

### Architecture Support

| Architecture | ASM | Performance Tier | Notes |
|---|---|---|---|
| **x86-64** | Yes | Tier 1 | BMI2/ADX acceleration, `-march=native` recommended |
| **ARM64/AArch64** | Yes | Tier 1 | NEON intrinsics, Apple M-series optimized |
| **RISC-V 64** | Yes | Tier 2 | Zba/Zbb extensions, branchless carry chains |
| **ESP32-S3** | No | Tier 3 | 32-bit fallback (10x26 limbs), no `__int128` |
| **STM32 (Cortex-M)** | No | Tier 3 | Bare-metal, 32-bit fallback |
| **Generic** | No | Tier 3 | Any C++20 platform with 64-bit integers |

### Enabling

```cmake
cmake -S . -B build -DSECP256K1_BUILD_CPU=ON   # default
```

### Performance (x86-64, Clang 21, -O3 -march=native)

| Operation | Time |
|---|---|
| Generator Mul (kxG) | **7 us** |
| Scalar Mul (kxP) | **25 us** |
| ECDSA Sign | **16 us** |
| ECDSA Verify | **32 us** |
| Schnorr Sign | **19 us** |
| Schnorr Verify | **42 us** |
| Point Add | **163 ns** |
| Field Inverse | **1 us** |

---

## CUDA Backend (Production)

Batch elliptic curve operations on NVIDIA GPUs. Used for the GPU search application (`gpu_cuda_test`).

### Requirements

- NVIDIA GPU with Compute Capability 5.0+ (Maxwell or newer)
- CUDA Toolkit 12.0+
- Host compiler: GCC, Clang, or MSVC (nvcc host compiler)

### Enabling

```cmake
cmake -S . -B build \
  -DSECP256K1_BUILD_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="86;89"
```

### Architecture Targets

| GPU Generation | Architecture | Flag |
|---|---|---|
| Maxwell | sm_50 | `-DCMAKE_CUDA_ARCHITECTURES=50` |
| Pascal | sm_60 | 60 |
| Volta | sm_70 | 70 |
| Turing | sm_75 | 75 |
| Ampere | sm_86 | 86 |
| Ada Lovelace | sm_89 | 89 |
| Hopper | sm_90 | 90 |

### Key Features

- Batch point multiplication (hundreds of thousands of ops)
- Bloom filter lookups on GPU memory
- In-kernel ECC with no host/device sync in hot loops
- Config-driven launch parameters (`threads_per_batch`, `batch_interval`)

### Gotchas

- **No `-flto` for CUDA targets**: LTO can break `nvcc -dlink`. The build system handles this automatically.
- **Windows**: nvcc requires MSVC host compiler (`cl.exe`). If unavailable, CUDA build is skipped with a warning.

---

## ROCm/HIP Backend (Beta)

Reuses CUDA sources with HIP portability layer for AMD GPUs.

### Requirements

- AMD GPU (gfx900 or newer)
- ROCm 5.0+ / HIP SDK
- CMake 3.21+ (native HIP language support)

### Enabling

```cmake
cmake -S . -B build -DSECP256K1_BUILD_ROCM=ON
```

### Notes

- Shares source with `cuda/`, builds in separate directory `cuda_rocm/`
- Portable math fallbacks used where CUDA intrinsics are unavailable
- Performance within ~15% of equivalent NVIDIA hardware

---

## OpenCL Backend (Beta)

Platform-agnostic GPU compute -- works with NVIDIA, AMD, and Intel GPUs.

### Requirements

- OpenCL 1.2+ runtime (typically bundled with GPU drivers)
- OpenCL headers (or system `OpenCL.lib` on Windows)

### Enabling

```cmake
cmake -S . -B build -DSECP256K1_BUILD_OPENCL=ON
```

### Notes

- Windows: Falls back to `C:\Windows\System32\OpenCL.dll` if SDK not found
- Lower performance than native CUDA/ROCm but broader hardware support
- Kernel source compiled at runtime (JIT)

---

## Metal Backend (Experimental)

Apple GPU compute for macOS, iOS, and visionOS.

### Requirements

- Apple platform with Metal 3 support (M1+, A14+)
- Xcode 15+
- macOS 14+ / iOS 17+

### Enabling

```cmake
cmake -S . -B build -DSECP256K1_BUILD_METAL=ON
```

### Notes

- Builds on non-Apple platforms in **host-test mode** only (type tests, no GPU execution)
- Metal Shading Language kernels in `.metal` files
- Leverages Apple's unified memory architecture (no explicit host<->device copies)

---

## WASM Backend (Experimental)

WebAssembly compilation for browsers and Node.js.

### Requirements

- Emscripten SDK (emsdk)
- CMake with Emscripten toolchain file

### Enabling

```bash
emcmake cmake -S . -B build-wasm -DSECP256K1_BUILD_CPU=ON
cmake --build build-wasm
```

### Notes

- No assembly optimizations (pure C++ fallback)
- 32-bit arithmetic path (10x26 limb representation)
- Suitable for client-side transaction signing in web wallets

---

## Android Backend (Experimental)

ARM64 native library for Android applications.

### Requirements

- Android NDK r27+
- CMake (bundled with Android Studio)

### Enabling

```bash
cmake -S . -B build-android \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-26
```

### Notes

- Uses ARM64 assembly path (same as Linux ARM64)
- Integrates via CMake in Android Studio's `build.gradle.kts`:

```kotlin
externalNativeBuild {
    cmake {
        arguments("-DSECP256K1_BUILD_CPU=ON", "-DSECP256K1_BUILD_TESTS=OFF")
    }
}
```

---

## Backend Selection Guide

| Use Case | Recommended Backend |
|---|---|
| General-purpose signing/verification | CPU |
| Bitcoin Core / wallet integration | CPU |
| Batch key search / vanity addresses | CUDA or ROCm |
| Cross-platform GPU (multi-vendor) | OpenCL |
| macOS/iOS native app | CPU + Metal |
| Web wallet / browser extension | WASM |
| Android app | Android (ARM64 CPU) |
| Embedded IoT | CPU (ESP32/STM32 path) |

---

## Multi-Backend Build

All backends can be enabled simultaneously:

```cmake
cmake -S . -B build \
  -DSECP256K1_BUILD_CPU=ON \
  -DSECP256K1_BUILD_CUDA=ON \
  -DSECP256K1_BUILD_OPENCL=ON \
  -DSECP256K1_BUILD_METAL=ON \
  -DCMAKE_BUILD_TYPE=Release
```

Each backend produces its own library target that can be linked independently.
