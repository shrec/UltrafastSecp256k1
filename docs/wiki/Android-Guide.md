# Android Guide — UltrafastSecp256k1

Android-ზე სრული CPU ბიბლიოთეკის პორტი — ARM64 (arm64-v8a), ARMv7 (armeabi-v7a), x86_64/x86 (ემულატორი).

## არქიტექტურა

```
android/
├── CMakeLists.txt           # Android-სპეციფიკური CMake build
├── build_android.sh         # Linux/macOS build script
├── build_android.ps1        # Windows PowerShell build script
├── jni/
│   └── secp256k1_jni.cpp    # JNI ხიდი (C++ → Java/Kotlin)
├── kotlin/
│   └── com/secp256k1/native/
│       └── Secp256k1.kt     # Kotlin wrapper კლასი
├── example/                 # სრული Android აპლიკაციის მაგალითი
│   ├── build.gradle.kts
│   └── src/main/
│       ├── cpp/CMakeLists.txt
│       └── kotlin/.../MainActivity.kt
└── output/                  # build-ის შედეგი (jniLibs/)
```

## ABI მხარდაჭერა

| ABI | არქიტექტურა | `__int128` | Assembly | შენიშვნა |
|-----|-------------|-----------|----------|---------|
| `arm64-v8a` | ARMv8-A + crypto + NEON | ✅ | ✅ ARM64 ASM (MUL/UMULH) | პირველადი target |
| `armeabi-v7a` | ARMv7-A + NEON | ❌ (32-bit) | ❌ | `SECP256K1_NO_INT128` fallback |
| `x86_64` | x86-64 + SSE4.2 | ✅ | ❌ (cross-compile) | ემულატორისთვის |
| `x86` | i686 + SSE3 | ❌ (32-bit) | ❌ | ემულატორისთვის |

> **შენიშვნა**: ARM64-ზე ახლა inline assembly ოპტიმიზაცია ჩართულია — `MUL`/`UMULH` ინსტრუქციები field arithmetic-ისთვის (mul, sqr, add, sub, neg). ეს უზრუნველყოფს **~5x დაჩქარებას** generic C++ კოდთან შედარებით scalar_mul ოპერაციებზე.

## სწრაფი დაწყება

### წინაპირობები

- Android NDK r25+ (რეკომენდებულია r26c)
- CMake 3.18+
- Ninja

### Build (ბრძანების ხაზი)

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

### Build (ხელით CMake)

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

### შედეგი

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

## Android პროექტში ინტეგრაცია

### ვარიანტი 1: Pre-built JNI (უმარტივესი)

1. დააკოპირეთ `output/jniLibs/` თქვენს Android პროექტში:
```
app/src/main/jniLibs/
├── arm64-v8a/libsecp256k1_jni.so
└── x86_64/libsecp256k1_jni.so
```

2. დააკოპირეთ `Secp256k1.kt` თქვენს Kotlin source-ში:
```
app/src/main/kotlin/com/secp256k1/native/Secp256k1.kt
```

3. გამოიყენეთ:
```kotlin
Secp256k1.init()
val pubkey = Secp256k1.ctScalarMulGenerator(privkey)
```

### ვარიანტი 2: Gradle CMake ინტეგრაცია

`app/build.gradle.kts`-ში:
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

### Fast API (მაქსიმალური სიჩქარე)

```kotlin
// ინიციალიზაცია
Secp256k1.init()

// Point ოპერაციები
val g = Secp256k1.getGenerator()             // G (65 bytes)
val g2 = Secp256k1.pointDouble(g)            // 2G
val g3 = Secp256k1.pointAdd(g2, g)           // 3G
val neg = Secp256k1.pointNegate(g)           // -G
val compressed = Secp256k1.pointCompress(g)  // 33 bytes

// Scalar × Point (არ არის side-channel safe!)
val result = Secp256k1.scalarMulGenerator(k)      // k*G
val result2 = Secp256k1.scalarMulPoint(k, point)  // k*P

// Scalar არითმეტიკა
val sum = Secp256k1.scalarAdd(a, b)
val product = Secp256k1.scalarMul(a, b)
val diff = Secp256k1.scalarSub(a, b)
```

### CT API (side-channel რეზისტენტული)

გამოიყენეთ **ყველა** პრივატული გასაღებით ოპერაციისთვის:

```kotlin
// Key generation (CT)
val pubkey = Secp256k1.ctScalarMulGenerator(privkey)

// k*P (CT)
val result = Secp256k1.ctScalarMulPoint(k, point)

// ECDH shared secret (CT)
val secret = Secp256k1.ctEcdh(myPrivkey, theirPubkey)
```

### როდის გამოვიყენოთ CT vs Fast

| ოპერაცია | API | მიზეზი |
|---------|-----|--------|
| Private key → Public key | **CT** | გასაღები საიდუმლოა |
| ECDH | **CT** | პრივატული გასაღები მონაწილეობს |
| ხელმოწერა (signing) | **CT** | nonce/key leak = კატასტროფა |
| ხელმოწერის ვერიფიკაცია | Fast | მხოლოდ საჯარო მონაცემები |
| Point-ების ბმა (aggregate) | Fast | მხოლოდ საჯარო მონაცემები |
| Batch ვერიფიკაცია | Fast | მაქსიმალური სიჩქარე |

## პლატფორმის დეტალები

### ARM64 ოპტიმიზაციები

**Inline Assembly** (`cpu/src/field_asm_arm64.cpp`):
- **`field_mul_arm64`** — 4×4 schoolbook MUL/UMULH + secp256k1 fast reduction (85 ns/op)
- **`field_sqr_arm64`** — ოპტიმიზებული squaring (10 mul vs 16) (66 ns/op)
- **`field_add_arm64`** — ADDS/ADCS + branchless normalization (18 ns/op)
- **`field_sub_arm64`** — SUBS/SBCS + conditional add p (16 ns/op)
- **`field_neg_arm64`** — Branchless p - a with CSEL

NDK Clang დამატებით იყენებს:
- **NEON**: 128-bit SIMD (იმპლიციტურია ARMv8-A-ში)
- **Crypto extensions**: AES/SHA hardware acceleration
- **`__int128`**: 64×64→128 გამრავლება (scalar/field ოპერაციებში)
- **Auto-vectorization**: `-ftree-vectorize -funroll-loops`

### Benchmark შედეგები (RK3588, Cortex-A55/A76)

| ოპერაცია | ARM64 ASM | Generic C++ | დაჩქარება |
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

\* CT რეჟიმი generic C++ იყენებს (constant-time გარანტიისთვის)

### ARMv7 (32-bit) შეზღუდვები

- `__int128` არ არის → `SECP256K1_NO_INT128` fallback (portable 64×64→128)
- NEON VFPv4 ხელმისაწვდომია
- ~2-3x ნელია ARM64-ზე ვიდრე

### Android-სპეციფიკური ცვლილებები CMake-ში

CPU `CMakeLists.txt`-ში ავტომატურად:
- `-march=native` → `-march=armv8-a+crypto` (cross-compile)
- `-mbmi2 -madx` გამოირიცხება ARM-ზე
- 32-bit target-ებზე `SECP256K1_NO_INT128=1`
- x86 assembly გამოირიცხება (ARM-ზე ვერ იკომპილირდება)

## Troubleshooting

### NDK ვერ მოიძებნა
```
export ANDROID_NDK_HOME=/full/path/to/ndk
```

### `c++_static` linkage error
build.gradle.kts-ში:
```kotlin
cmake { arguments += "-DANDROID_STL=c++_static" }
```

### UnsatisfiedLinkError runtime-ზე
შეამოწმეთ რომ `libsecp256k1_jni.so` სწორ ABI ფოლდერშია (`jniLibs/arm64-v8a/`).

### 32-bit build warnings
ARMv7/x86 ბილდებზე ნორმალურია — `SECP256K1_NO_INT128` ავტომატურად ჩაირთვება.
