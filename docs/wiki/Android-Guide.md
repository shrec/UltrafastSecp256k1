# Android Guide — UltrafastSecp256k1

Full CPU library port for Android — ARM64 (arm64-v8a), ARMv7 (armeabi-v7a), x86_64/x86 (emulator).

## Architecture

```
android/
├── CMakeLists.txt           # Android-specific CMake build
├── build_android.sh         # Linux/macOS build script
├── build_android.ps1        # Windows PowerShell build script
├── jni/
│   └── secp256k1_jni.cpp    # JNI bridge (C++ → Java/Kotlin)
├── kotlin/
│   └── com/secp256k1/native/
│       └── Secp256k1.kt     # Kotlin wrapper class
├── example/                 # Full Android application example
│   ├── build.gradle.kts
│   └── src/main/
│       ├── cpp/CMakeLists.txt
│       └── kotlin/.../MainActivity.kt
└── output/                  # Build output (jniLibs/)
```

## ABI Support

| ABI | Architecture | `__int128` | Assembly | Notes |
|-----|-------------|-----------|----------|---------|
| `arm64-v8a` | ARMv8-A + crypto + NEON | ✅ | ✅ ARM64 ASM (MUL/UMULH) | Primary target |
| `armeabi-v7a` | ARMv7-A + NEON | ❌ (32-bit) | ❌ | `SECP256K1_NO_INT128` fallback |
| `x86_64` | x86-64 + SSE4.2 | ✅ | ❌ (cross-compile) | For emulator |
| `x86` | i686 + SSE3 | ❌ (32-bit) | ❌ | For emulator |

> **Note**: ARM64 inline assembly optimization is now enabled — `MUL`/`UMULH` instructions for field arithmetic (mul, sqr, add, sub, neg). This provides **~5x speedup** compared to generic C++ code for scalar_mul operations.

## Quick Start

### Prerequisites

- Android NDK r25+ (r26c recommended)
- CMake 3.18+
- Ninja

### Build (Command Line)

```bash
# Linux/macOS
export ANDROID_NDK_HOME=/path/to/android-ndk-r26c
cd libs/UltrafastSecp256k1/android/
./build_android.sh arm64-v8a

# Windows PowerShell
$env:ANDROID_NDK_HOME = "C:\Users\user\AppData\Local\Android\Sdk\ndk\26.1.10909125"
cd libs\UltrafastSecp256k1\android\
.\build_android.ps1 -ABIs arm64-v8a
```

### Build (Manual CMake)

```bash
cmake -S android -B android/build-android-arm64 \
    -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-24 \
    -DANDROID_STL=c++_static \
    -DCMAKE_BUILD_TYPE=Release \
    -G Ninja

cmake --build android/build-android-arm64 -j
```

### Output

```
android/output/jniLibs/
├── arm64-v8a/
│   └── libsecp256k1_jni.so      # ~200-400 KB
├── armeabi-v7a/
│   └── libsecp256k1_jni.so
├── x86_64/
│   └── libsecp256k1_jni.so
└── x86/
    └── libsecp256k1_jni.so
```

## Integration in an Android Project

### Option 1: Pre-built JNI (simplest)

1. Copy `output/jniLibs/` to your Android project:
```
app/src/main/jniLibs/
├── arm64-v8a/libsecp256k1_jni.so
└── x86_64/libsecp256k1_jni.so
```

2. Copy `Secp256k1.kt` to your Kotlin source:
```
app/src/main/kotlin/com/secp256k1/native/Secp256k1.kt
```

3. Use it:
```kotlin
Secp256k1.init()
val pubkey = Secp256k1.ctScalarMulGenerator(privkey)
```

### Option 2: Gradle CMake Integration

In `app/build.gradle.kts`:
```kotlin
android {
    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
        }
    }
    defaultConfig {
        externalNativeBuild {
            cmake {
                abiFilters += listOf("arm64-v8a", "x86_64")
                arguments += "-DANDROID_STL=c++_static"
            }
        }
    }
}
```

`app/src/main/cpp/CMakeLists.txt`:
```cmake
cmake_minimum_required(VERSION 3.18)
project(MyApp LANGUAGES CXX)
add_subdirectory(/path/to/UltrafastSecp256k1/android ${CMAKE_BINARY_DIR}/secp256k1)
```

## API

### Fast API (Maximum Speed)

```kotlin
// Initialization
Secp256k1.init()

// Point operations
val g = Secp256k1.getGenerator()             // G (65 bytes)
val g2 = Secp256k1.pointDouble(g)            // 2G
val g3 = Secp256k1.pointAdd(g2, g)           // 3G
val neg = Secp256k1.pointNegate(g)           // -G
val compressed = Secp256k1.pointCompress(g)  // 33 bytes

// Scalar × Point (NOT side-channel safe!)
val result = Secp256k1.scalarMulGenerator(k)      // k*G
val result2 = Secp256k1.scalarMulPoint(k, point)  // k*P

// Scalar arithmetic
val sum = Secp256k1.scalarAdd(a, b)
val product = Secp256k1.scalarMul(a, b)
val diff = Secp256k1.scalarSub(a, b)
```

### CT API (side-channel resistant)

Use for **all** private key operations:

```kotlin
// Key generation (CT)
val pubkey = Secp256k1.ctScalarMulGenerator(privkey)

// k*P (CT)
val result = Secp256k1.ctScalarMulPoint(k, point)

// ECDH shared secret (CT)
val secret = Secp256k1.ctEcdh(myPrivkey, theirPubkey)
```

### When to Use CT vs Fast

| Operation | API | Reason |
|---------|-----|--------|
| Private key → Public key | **CT** | Key is secret |
| ECDH | **CT** | Private key is involved |
| Signing | **CT** | nonce/key leak = catastrophe |
| Signature verification | Fast | Public data only |
| Point aggregation | Fast | Public data only |
| Batch verification | Fast | Maximum speed |

## Platform Details

### ARM64 Optimizations

**Inline Assembly** (`cpu/src/field_asm_arm64.cpp`):
- **`field_mul_arm64`** — 4×4 schoolbook MUL/UMULH + secp256k1 fast reduction (85 ns/op)
- **`field_sqr_arm64`** — Optimized squaring (10 mul vs 16) (66 ns/op)
- **`field_add_arm64`** — ADDS/ADCS + branchless normalization (18 ns/op)
- **`field_sub_arm64`** — SUBS/SBCS + conditional add p (16 ns/op)
- **`field_neg_arm64`** — Branchless p - a with CSEL

NDK Clang additionally uses:
- **NEON**: 128-bit SIMD (implicit in ARMv8-A)
- **Crypto extensions**: AES/SHA hardware acceleration
- **`__int128`**: 64×64→128 multiplication (in scalar/field operations)
- **Auto-vectorization**: `-ftree-vectorize -funroll-loops`

### Benchmark Results (RK3588, Cortex-A55/A76)

| Operation | ARM64 ASM | Generic C++ | Speedup |
|---------|-----------|-------------|-----------|
| field_mul (a*b mod p) | **85 ns** | ~350 ns | ~4x |
| field_sqr (a² mod p) | **66 ns** | ~280 ns | ~4x |
| field_add (a+b mod p) | **18 ns** | ~30 ns | ~1.7x |
| field_sub (a-b mod p) | **16 ns** | ~28 ns | ~1.8x |
| field_inverse | **2,621 ns** | ~11,000 ns | ~4x |
| **fast scalar_mul (k*G)** | **7.6 μs** | ~40 μs | **~5.3x** |
| fast scalar_mul (k*P) | **77.6 μs** | ~400 μs | **~5.1x** |
| CT scalar_mul (k*G) | 545 μs | ~400 μs | 0.7x* |
| ECDH (full CT) | 545 μs | — | — |

\* CT mode uses generic C++ (for constant-time guarantees)

### ARMv7 (32-bit) Limitations

- No `__int128` → `SECP256K1_NO_INT128` fallback (portable 64×64→128)
- NEON VFPv4 available
- ~2-3x slower than ARM64

### Android-Specific CMake Changes

Automatically in CPU `CMakeLists.txt`:
- `-march=native` → `-march=armv8-a+crypto` (cross-compile)
- `-mbmi2 -madx` excluded on ARM
- `SECP256K1_NO_INT128=1` on 32-bit targets
- x86 assembly excluded (cannot compile on ARM)

## Troubleshooting

### NDK Not Found
```
export ANDROID_NDK_HOME=/full/path/to/ndk
```

### `c++_static` linkage error
In build.gradle.kts:
```kotlin
cmake { arguments += "-DANDROID_STL=c++_static" }
```

### UnsatisfiedLinkError at Runtime
Check that `libsecp256k1_jni.so` is in the correct ABI folder (`jniLibs/arm64-v8a/`).

### 32-bit build warnings
Normal on ARMv7/x86 builds — `SECP256K1_NO_INT128` is automatically enabled.
