# Building UltrafastSecp256k1

Complete build guide for all supported platforms.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Start](#quick-start)
3. [Build Options](#build-options)
4. [Platform-Specific Instructions](#platform-specific-instructions)
   - [Linux x86-64](#linux-x86-64)
   - [Windows x86-64](#windows-x86-64)
   - [macOS](#macos)
   - [RISC-V 64](#risc-v-64)
   - [CUDA (NVIDIA GPU)](#cuda)
   - [ROCm / HIP (AMD GPU)](#rocm--hip-amd-gpu)
   - [OpenCL](#opencl)
   - [WebAssembly (Emscripten)](#webassembly-emscripten)
   - [iOS (XCFramework)](#ios-xcframework)
   - [Android (NDK)](#android-ndk)
   - [ESP32 (ESP-IDF)](#esp32-esp-idf)
   - [STM32 (ARM Cortex-M)](#stm32-arm-cortex-m)
5. [Cross-Compilation](#cross-compilation)
6. [CMake Integration](#cmake-integration)
7. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required

- **CMake** 3.18 or later
- **C++20 compiler**:
  - GCC 11+ (recommended for Linux)
  - Clang/LLVM 15+ (recommended, best optimization)
  - MSVC 2022+ (Windows, requires `-DSECP256K1_ALLOW_MSVC=ON`)
- **Ninja** (recommended) or Make

### Optional

- **CUDA Toolkit 12.0+** for NVIDIA GPU support
- **ROCm 5.0+ / HIP SDK** for AMD GPU support (CMake 3.21+)
- **Emscripten SDK** for WebAssembly builds
- **Xcode 15+** for iOS builds
- **Android NDK r27+** for Android builds
- **ESP-IDF v5.2+** for ESP32 builds
- **ARM GCC** for STM32 builds
- **clang-19/21** for best RISC-V optimization

---

## Quick Start

### CPU Only (Default)

```bash
# Configure
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build build -j

# Test
ctest --test-dir build --output-on-failure
```

### With CUDA

```bash
# Configure
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DSECP256K1_BUILD_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="75;86;89"

# Build
cmake --build build -j
```

---

## Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `SECP256K1_USE_ASM` | ON | Assembly optimizations (x64/RISC-V) |
| `SECP256K1_BUILD_CUDA` | OFF | Build CUDA GPU library |
| `SECP256K1_BUILD_OPENCL` | OFF | Build OpenCL GPU support |
| `SECP256K1_BUILD_ROCM` | OFF | Build ROCm/HIP GPU support (AMD) |
| `SECP256K1_BUILD_TESTS` | ON | Build test suite |
| `SECP256K1_BUILD_BENCH` | ON | Build benchmarks |
| `SECP256K1_BUILD_EXAMPLES` | ON | Build example programs |
| `SECP256K1_USE_LTO` | OFF | Link-Time Optimization |
| `SECP256K1_SPEED_FIRST` | OFF | Aggressive speed optimizations |
| `SECP256K1_ALLOW_MSVC` | OFF | Allow MSVC compiler |
| `SECP256K1_INSTALL` | ON | Generate install target |

### RISC-V Specific

| Option | Default | Description |
|--------|---------|-------------|
| `SECP256K1_RISCV_FAST_REDUCTION` | ON | Fast modular reduction |
| `SECP256K1_RISCV_USE_VECTOR` | ON | RVV vector extension |
| `SECP256K1_RISCV_USE_PREFETCH` | ON | Memory prefetch hints |

### CUDA Specific

| Option | Default | Description |
|--------|---------|-------------|
| `SECP256K1_CUDA_USE_MONTGOMERY` | OFF | Montgomery domain arithmetic |
| `SECP256K1_CUDA_LIMBS_32` | OFF | 8×32-bit limbs (experimental) |
| `CMAKE_CUDA_ARCHITECTURES` | 89 | Target GPU architectures |

---

## Platform-Specific Instructions

### Linux x86-64

#### Using Clang (Recommended)

```bash
# Install dependencies
sudo apt install cmake ninja-build clang-19 lld-19

# Configure with Clang
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=clang-19 \
  -DCMAKE_CXX_COMPILER=clang++-19

# Build
cmake --build build -j$(nproc)
```

#### Using GCC

```bash
# Install dependencies
sudo apt install cmake ninja-build g++-11

# Configure with GCC
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=gcc-11 \
  -DCMAKE_CXX_COMPILER=g++-11

# Build
cmake --build build -j$(nproc)
```

---

### Windows x86-64

#### Using Clang/LLVM (Recommended)

```powershell
# Install LLVM from https://llvm.org/builds/
# Or via winget:
winget install LLVM.LLVM

# Configure
cmake -S . -B build -G Ninja `
  -DCMAKE_BUILD_TYPE=Release `
  -DCMAKE_C_COMPILER=clang `
  -DCMAKE_CXX_COMPILER=clang++

# Build
cmake --build build -j
```

#### Using MSVC (Not Recommended)

```powershell
# Open Visual Studio Developer Command Prompt
# Then:
cmake -S . -B build -G "Visual Studio 17 2022" `
  -DSECP256K1_ALLOW_MSVC=ON

cmake --build build --config Release
```

> ⚠️ **Warning**: MSVC produces slower code compared to Clang/GCC.

---

### macOS

```bash
# AppleClang (default)
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# Or with Homebrew LLVM (for latest optimizations)
brew install llvm ninja
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=$(brew --prefix llvm)/bin/clang \
  -DCMAKE_CXX_COMPILER=$(brew --prefix llvm)/bin/clang++
cmake --build build -j
```

ARM64 (Apple Silicon) inline assembly is automatically enabled on aarch64.

---

### RISC-V 64

#### Native Build

```bash
# On RISC-V machine with Clang 19+
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=clang-21 \
  -DCMAKE_CXX_COMPILER=clang++-21

cmake --build build -j$(nproc)
```

#### Cross-Compilation (from x86-64)

```bash
# Install toolchain
sudo apt install gcc-riscv64-linux-gnu g++-riscv64-linux-gnu

# Configure for cross-compilation
cmake -S . -B build-riscv -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_SYSTEM_NAME=Linux \
  -DCMAKE_SYSTEM_PROCESSOR=riscv64 \
  -DCMAKE_C_COMPILER=riscv64-linux-gnu-gcc \
  -DCMAKE_CXX_COMPILER=riscv64-linux-gnu-g++

cmake --build build-riscv -j$(nproc)
```

#### Expected Performance (RISC-V)

| Operation | Time |
|-----------|------|
| Field Mul | ~198 ns |
| Field Square | ~177 ns |
| Field Add | ~34 ns |
| Point Scalar Mul | ~672 μs |
| Generator Mul | ~40 μs |

---

### CUDA

#### Prerequisites

1. Install [CUDA Toolkit 12.0+](https://developer.nvidia.com/cuda-downloads)
2. Ensure `nvcc` is in PATH

#### Build

```bash
# Configure
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DSECP256K1_BUILD_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="75;86;89"

# Build
cmake --build build -j
```

#### GPU Architecture Reference

| Architecture | GPUs | Flag |
|--------------|------|------|
| sm_75 | RTX 2060-2080, T4 | 75 |
| sm_80 | A100 | 80 |
| sm_86 | RTX 3060-3090, A6000 | 86 |
| sm_89 | RTX 4060-4090, L4, L40 | 89 |
| sm_90 | H100 | 90 |

Example for RTX 4090:
```bash
cmake -DCMAKE_CUDA_ARCHITECTURES=89 ...
```

---

### ROCm / HIP (AMD GPU)

#### Prerequisites

1. Install [ROCm 5.0+](https://rocm.docs.amd.com/en/latest/deploy/linux/quick_start.html) or HIP SDK
2. CMake 3.21+ (native HIP language support)

#### Build

```bash
cmake -S . -B build-rocm -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DSECP256K1_BUILD_ROCM=ON \
  -DCMAKE_HIP_ARCHITECTURES="gfx1030;gfx1100"

cmake --build build-rocm -j
```

#### AMD GPU Architecture Reference

| Architecture | GPUs | Flag |
|--------------|------|------|
| gfx906 | Radeon VII, MI50 | gfx906 |
| gfx908 | MI100 | gfx908 |
| gfx90a | MI210, MI250 | gfx90a |
| gfx1030 | RX 6800/6900 | gfx1030 |
| gfx1100 | RX 7900 XTX | gfx1100 |

> **Note**: PTX inline asm is automatically replaced with portable `__int128` fallbacks on HIP. The hybrid 32-bit mul backend is disabled on HIP (PTX-dependent).

#### Docker Build

```bash
docker run --rm -v $(pwd):/src rocm/dev-ubuntu-22.04:6.3 \
  bash -c "cd /src && cmake -S . -B build -G Ninja \
    -DSECP256K1_BUILD_ROCM=ON && cmake --build build -j"
```

---

### OpenCL

#### Prerequisites

- OpenCL 3.0 runtime (GPU drivers typically provide this)
- OpenCL headers (`apt install opencl-headers` on Linux)

#### Build

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DSECP256K1_BUILD_OPENCL=ON

cmake --build build -j
```

On Windows, CMake auto-detects `OpenCL.lib` from the GPU driver in `C:\Windows\System32`.

---

### WebAssembly (Emscripten)

#### Prerequisites

```bash
# Install Emscripten SDK
git clone https://github.com/emscripten-core/emsdk.git
cd emsdk && ./emsdk install latest && ./emsdk activate latest
source emsdk_env.sh
```

#### Build

```bash
# Using the build script (recommended)
./scripts/build_wasm.sh

# Or manually:
emcmake cmake -S wasm -B build-wasm -DCMAKE_BUILD_TYPE=Release
cmake --build build-wasm -j
```

#### Output

| File | Description |
|------|-------------|
| `build-wasm/dist/secp256k1_wasm.wasm` | WebAssembly binary |
| `build-wasm/dist/secp256k1_wasm.js` | Emscripten ES6 loader |
| `build-wasm/dist/secp256k1.mjs` | High-level JS wrapper |
| `build-wasm/dist/secp256k1.d.ts` | TypeScript declarations |

See [wasm/README.md](../wasm/README.md) for JS/TS usage and npm publishing.

---

### iOS (XCFramework)

#### Prerequisites

- macOS with Xcode 15+
- CMake 3.18+

#### Build

```bash
# Build universal XCFramework (device + simulator)
./scripts/build_xcframework.sh
```

Output: `build-xcframework/output/UltrafastSecp256k1.xcframework`

#### Integration

**Swift Package Manager:**
```swift
// Package.swift
dependencies: [
    .package(url: "https://github.com/shrec/UltrafastSecp256k1.git", from: "3.0.0")
]
```

**CocoaPods:**
```ruby
# Podfile
pod 'UltrafastSecp256k1', '~> 3.0.0'
```

**Manual XCFramework:**
1. Drag `UltrafastSecp256k1.xcframework` into Xcode
2. Add to target's "Frameworks, Libraries, and Embedded Content"
3. Include headers:
   ```cpp
   #include <secp256k1/field.hpp>
   #include <secp256k1/ecdsa.hpp>
   ```

---

### Android (NDK)

#### Prerequisites

- [Android NDK r27+](https://developer.android.com/ndk/downloads)
- CMake 3.18+

#### Build

```bash
# Set NDK path
export ANDROID_NDK=$HOME/Android/Sdk/ndk/27.2.12479018

# Build for arm64-v8a
cmake -S . -B build-android -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-24 \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build-android -j
```

The library produces `libsecp256k1-fast-cpu.a` for linking into Android apps via JNI.

---

### ESP32 (ESP-IDF)

#### Prerequisites

- [ESP-IDF v5.2+](https://docs.espressif.com/projects/esp-idf/en/latest/esp32/get-started/)
- Supported targets: ESP32-S3 (Xtensa LX7), ESP32 (Xtensa LX6)

#### Build & Flash

```bash
cd examples/esp32_test

# Set target (esp32s3 or esp32)
idf.py set-target esp32s3

# Build, flash and monitor
idf.py -p /dev/ttyUSB0 build flash monitor
```

See [docs/ESP32_SETUP.md](ESP32_SETUP.md) for detailed CLion/IDE integration.

> **Note**: ESP32 uses portable C++ (no `__int128`, no platform assembly). All 35 library self-tests pass on both ESP32-S3 and ESP32-PICO-D4.

---

### STM32 (ARM Cortex-M)

#### Prerequisites

- ARM GCC toolchain (`arm-none-eabi-gcc` 13+)
- STM32CubeMX or raw Makefile

#### Build

```bash
cd examples/stm32_test
make -j
```

The STM32 port uses ARM Cortex-M3 inline assembly (`UMULL/ADDS/ADCS`) for field multiplication and squaring. Portable C++ for field add/sub.

See [examples/stm32_test/](../examples/stm32_test/) for the complete project.

---

---

## Cross-Compilation

### RISC-V from x86-64

See [RISC-V 64](#risc-v-64) section above.

### Windows from Linux

```bash
# Using MinGW-w64
sudo apt install mingw-w64

cmake -S . -B build-win -G Ninja \
  -DCMAKE_SYSTEM_NAME=Windows \
  -DCMAKE_C_COMPILER=x86_64-w64-mingw32-gcc \
  -DCMAKE_CXX_COMPILER=x86_64-w64-mingw32-g++

cmake --build build-win -j
```

---

## Troubleshooting

### LTO Not Working on RISC-V

LTO requires `LLVMgold.so` plugin. If missing:

```bash
# Install LLVM with gold plugin
sudo apt install llvm-19-dev

# Or disable LTO
cmake -DSECP256K1_USE_LTO=OFF ...
```

### CUDA Compilation Errors

1. **"nvcc not found"**: Add CUDA to PATH
   ```bash
   export PATH=/usr/local/cuda/bin:$PATH
   ```

2. **"unsupported architecture"**: Update `CMAKE_CUDA_ARCHITECTURES`:
   ```bash
   cmake -DCMAKE_CUDA_ARCHITECTURES=89 ...
   ```

### Assembly Errors on RISC-V

If you see "symbol already defined" errors:
```bash
# Rebuild clean
rm -rf build
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

### MSVC Link Errors

MSVC is not fully supported. Use Clang instead:
```powershell
# Install LLVM from https://llvm.org/builds/
cmake -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ ...
```

---

## CMake Integration

### As a Subdirectory

```cmake
add_subdirectory(path/to/UltrafastSecp256k1)
target_link_libraries(your_target PRIVATE secp256k1-fast-cpu)
```

### After Installation

```bash
# Install to system (Linux)
sudo cmake --install build --prefix /usr/local

# Install to custom location
cmake --install build --prefix /opt/secp256k1
```

```cmake
# In your CMakeLists.txt
find_package(secp256k1-fast REQUIRED)
target_link_libraries(your_target PRIVATE secp256k1::fast-cpu)
```

### pkg-config

```bash
pkg-config --cflags --libs secp256k1-fast
```

---

## Verification

After building, run tests to verify correctness:

```bash
# Run all tests
ctest --test-dir build --output-on-failure

# Run benchmarks
./build/cpu/bench_comprehensive
./build/cuda/secp256k1_cuda_bench  # If CUDA enabled
```

---

## Install from Linux Packages

Pre-built packages are available for each [GitHub Release](https://github.com/shrec/UltrafastSecp256k1/releases).

### Debian / Ubuntu (APT)

```bash
# Add the repository
curl -fsSL https://shrec.github.io/UltrafastSecp256k1/apt/KEY.gpg \
  | sudo gpg --dearmor -o /etc/apt/keyrings/ultrafastsecp256k1.gpg
echo "deb [signed-by=/etc/apt/keyrings/ultrafastsecp256k1.gpg] \
  https://shrec.github.io/UltrafastSecp256k1/apt stable main" \
  | sudo tee /etc/apt/sources.list.d/ultrafastsecp256k1.list

sudo apt update
sudo apt install libsecp256k1-fast-dev   # headers + static + shared
```

### Fedora / RHEL (RPM)

```bash
# Download the latest .rpm from GitHub Releases
curl -LO "https://github.com/shrec/UltrafastSecp256k1/releases/latest/download/\
UltrafastSecp256k1-$(rpm -E %{_arch}).rpm"
sudo dnf install ./UltrafastSecp256k1-*.rpm
```

### Arch Linux (AUR)

```bash
yay -S libsecp256k1-fast
```

### Docker (build from source)

```bash
docker build -t ultrafastsecp256k1 -f Dockerfile .
docker run --rm ultrafastsecp256k1 ctest --output-on-failure
```

---

## Version

Current version is read from `VERSION.txt` at configure time.

