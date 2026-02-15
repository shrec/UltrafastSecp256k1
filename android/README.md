# UltrafastSecp256k1 — Android Port

Full CPU port of UltrafastSecp256k1 for Android (ARM64, ARMv7, x86_64, x86).

## Quick Build

```bash
# Set NDK path
export ANDROID_NDK_HOME=/path/to/android-ndk-r26c

# Build ARM64 (primary target)
./build_android.sh arm64-v8a

# Build all ABIs
./build_android.sh
```

## Output

```
output/jniLibs/
├── arm64-v8a/libsecp256k1_jni.so
├── armeabi-v7a/libsecp256k1_jni.so
├── x86_64/libsecp256k1_jni.so
└── x86/libsecp256k1_jni.so
```

## Usage (Kotlin)

```kotlin
import com.secp256k1.native.Secp256k1

// Initialize once
Secp256k1.init()

// Key generation (constant-time — side-channel safe)
val pubkey = Secp256k1.ctScalarMulGenerator(privkeyBytes)

// ECDH (constant-time)
val secret = Secp256k1.ctEcdh(myPrivkey, theirPubkey)

// Fast operations (public data only)
val g = Secp256k1.getGenerator()
val sum = Secp256k1.pointAdd(p1, p2)
```

## Project Structure

```
android/
├── CMakeLists.txt          — Android CMake build
├── build_android.sh        — Linux/macOS build script
├── build_android.ps1       — Windows build script
├── jni/secp256k1_jni.cpp   — JNI bridge (C++ ↔ Java/Kotlin)
├── kotlin/.../Secp256k1.kt — Kotlin wrapper
└── example/                — Example Android app
```

See [Android Guide](../docs/wiki/Android-Guide.md) for full documentation.

## Benchmarks (RK3588, Cortex-A55/A76, ARM64 ASM)

| Operation | Time |
|-----------|------|
| field_mul (a*b mod p) | 85 ns |
| field_sqr (a² mod p) | 66 ns |
| field_add (a+b mod p) | 18 ns |
| field_sub (a-b mod p) | 16 ns |
| field_inverse | 2,621 ns |
| **fast scalar_mul (k*G)** | **7.6 μs** |
| fast scalar_mul (k*P) | 77.6 μs |
| CT scalar_mul (k*G) | 545 μs |
| ECDH (full CT) | 545 μs |

Backend: ARM64 inline assembly (MUL/UMULH). ~5x faster than generic C++.
