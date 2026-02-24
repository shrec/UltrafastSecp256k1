# Bindings Packaging Guide
## UltrafastSecp256k1 Per-Ecosystem Distribution

> **Goal**: Each binding must be installable via the ecosystem's standard package manager in ≤3 commands.

---

## 1. Distribution Matrix

| Language | Package Manager | Package Name | Registry | Status |
|---|---|---|---|:---:|
| C/C++ | CMake / pkg-config | `ufsecp` | System install / vcpkg | ✅ |
| Python | pip | `ufsecp` | PyPI | ✅ |
| Node.js | npm | `@ultrafast/ufsecp` | npmjs.com | ✅ |
| C# | NuGet | `UltrafastSecp256k1` | nuget.org | ✅ |
| Java | Maven/Gradle | `com.ultrafast:ufsecp` | Maven Central | ✅ |
| Swift | SPM / CocoaPods | `Ufsecp` | GitHub releases | ✅ |
| Go | go modules | `github.com/nicenemo/ufsecp` | proxy.golang.org | ✅ |
| Rust | cargo | `ufsecp` | crates.io | ✅ |
| Dart | pub | `ufsecp` | pub.dev | ✅ |
| PHP | Composer | `ultrafast/ufsecp` | Packagist | ✅ |
| Ruby | gem | `ufsecp` | RubyGems | ✅ |
| React Native | npm | `react-native-ufsecp` | npmjs.com | ✅ |

---

## 2. Per-Ecosystem Details

### 2.1 C/C++ (CMake)

```cmake
find_package(ufsecp REQUIRED)
target_link_libraries(myapp PRIVATE ufsecp::ufsecp)
```

**Distribution**: Static/shared lib + headers installed to system prefix.

**Platform artifacts**:
| Platform | File |
|---|---|
| Linux | `libufsecp.so` / `libufsecp.a` |
| macOS | `libufsecp.dylib` / `libufsecp.a` |
| Windows | `ufsecp.dll` + `ufsecp.lib` |

---

### 2.2 Python (pip / PyPI)

```bash
pip install ufsecp
```

**Wheel matrix** (PEP 517, `manylinux2014`):

| Platform | Architectures |
|---|---|
| Linux | x86_64, aarch64 |
| macOS | x86_64, arm64 (universal2) |
| Windows | amd64 |

**Build backend**: `setuptools` with C extension build.  
**Includes**: Pre-compiled shared library + Python wrapper.  
**Source dist**: `sdist` includes C sources for building from source.

---

### 2.3 Node.js (npm)

```bash
npm install @ultrafast/ufsecp
```

**Native module strategy**: `prebuild-install` with N-API (ABI-stable).

| Platform | Prebuilt |
|---|---|
| Linux x64 | ✅ |
| macOS x64/arm64 | ✅ |
| Windows x64 | ✅ |

**Fallback**: `node-gyp` rebuild from source if prebuild unavailable.  
**N-API version**: 8 (Node.js 12.22+).

---

### 2.4 C# (NuGet)

```xml
<PackageReference Include="UltrafastSecp256k1" Version="3.14.0" />
```

```bash
dotnet add package UltrafastSecp256k1
```

**RID matrix** (runtime identifiers):

| RID | Library |
|---|---|
| `linux-x64` | `runtimes/linux-x64/native/libufsecp.so` |
| `linux-arm64` | `runtimes/linux-arm64/native/libufsecp.so` |
| `osx-x64` | `runtimes/osx-x64/native/libufsecp.dylib` |
| `osx-arm64` | `runtimes/osx-arm64/native/libufsecp.dylib` |
| `win-x64` | `runtimes/win-x64/native/ufsecp.dll` |

**Target frameworks**: `net6.0`, `net8.0`, `netstandard2.0`.

---

### 2.5 Java (Maven Central)

```xml
<dependency>
    <groupId>com.ultrafast</groupId>
    <artifactId>ufsecp</artifactId>
    <version>3.14.0</version>
</dependency>
```

```groovy
// Gradle
implementation 'com.ultrafast:ufsecp:3.14.0'
```

**JNI native library packaging**:

| OS | Arch | Path in JAR |
|---|---|---|
| Linux | x86_64 | `native/linux-x86_64/libufsecp_jni.so` |
| Linux | aarch64 | `native/linux-aarch64/libufsecp_jni.so` |
| macOS | x86_64 | `native/darwin-x86_64/libufsecp_jni.dylib` |
| macOS | aarch64 | `native/darwin-aarch64/libufsecp_jni.dylib` |
| Windows | x86_64 | `native/windows-x86_64/ufsecp_jni.dll` |

**Android**: AAR artifact with `jniLibs/` for `arm64-v8a`, `armeabi-v7a`, `x86`, `x86_64`.

**Java version**: 11+ (LTS).

---

### 2.6 Swift (SPM + CocoaPods)

**Swift Package Manager**:
```swift
// Package.swift
dependencies: [
    .package(url: "https://github.com/nicenemo/ufsecp-swift", from: "3.14.0"),
]
```

**CocoaPods**:
```ruby
pod 'Ufsecp', '~> 3.14.0'
```

**XCFramework**: Pre-built binary for iOS (arm64), iOS Simulator (arm64, x86_64), macOS (arm64, x86_64).

**Minimum deployment**: iOS 13+, macOS 10.15+.

---

### 2.7 Go (Go Modules)

```bash
go get github.com/nicenemo/ufsecp@v3.14.0
```

**CGo requirements**: Requires C compiler + `libufsecp` installed system-wide or via `CGO_LDFLAGS`/`CGO_CFLAGS`.

**Prebuilt strategy**: The go module wraps CGo. Users must have the C library installed:
```bash
# Linux
sudo apt install libufsecp-dev

# macOS
brew install ufsecp

# From source
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
sudo cmake --install build
```

---

### 2.8 Rust (crates.io)

```toml
[dependencies]
ufsecp = "3.14.0"
```

**Crate structure**:
- `ufsecp-sys` — raw FFI bindings (build.rs links native lib)
- `ufsecp` — safe Rust wrapper

**Linking strategy**:
- Default: dynamic linking (`-lufsecp`)
- Feature `static`: static linking from vendored sources
- Feature `vendored`: builds C library from bundled source

```toml
[dependencies]
ufsecp = { version = "3.14.0", features = ["vendored"] }
```

---

### 2.9 Dart (pub.dev)

```yaml
dependencies:
  ufsecp: ^3.14.0
```

```bash
dart pub add ufsecp
```

**Platform detection**: `dart:ffi` with runtime library name resolution:
- Linux: `libufsecp.so`
- macOS: `libufsecp.dylib`
- Windows: `ufsecp.dll`

**Flutter integration**: Native library bundled via Flutter plugin convention:
- Android: `jniLibs/`
- iOS: Framework in `ios/Frameworks/`

---

### 2.10 PHP (Packagist)

```bash
composer require ultrafast/ufsecp
```

**Requirements**: PHP 8.1+ with `ext-ffi` enabled.

**System dependency**: `libufsecp` must be installed system-wide:
```bash
sudo apt install libufsecp-dev   # Debian/Ubuntu
```

---

### 2.11 Ruby (RubyGems)

```bash
gem install ufsecp
```

**Requirements**: `ffi` gem + `libufsecp` installed.

**Native extension**: Optional C extension for improved loading:
```ruby
# Gemfile
gem 'ufsecp', '~> 3.14.0'
```

---

### 2.12 React Native (npm)

```bash
npm install react-native-ufsecp
cd ios && pod install
```

**Auto-linking**: Supports React Native 0.60+ auto-linking.

**Platform support**:
| Platform | Native Module |
|---|---|
| iOS | ObjC bridge → C library (XCFramework) |
| Android | Java JNI → C library (AAR with `jniLibs/`) |

**Android ABIs**: `arm64-v8a`, `armeabi-v7a`, `x86_64`.  
**iOS architectures**: `arm64` (device), `arm64` + `x86_64` (simulator).

---

## 3. CI Validation Matrix

The `bindings.yml` workflow validates packaging for all ecosystems:

| Language | CI Check | What It Tests |
|---|---|---|
| C API | CMake build (3 OSes) | Library compiles, headers parse |
| Python | `pip install -e .` + smoke test | Wrapper loads, vectors pass |
| Node.js | `npm test` + smoke test | N-API loads, vectors pass |
| C# | `dotnet build` + smoke test | P/Invoke resolution |
| Java | `javac` compile check | JNI header compatibility |
| Swift | `swift build` | SPM package resolution |
| Go | `go vet` + `go build` | CGo builds, types match |
| Rust | `cargo check` | FFI types, lifetimes |
| Dart | `dart analyze` | FFI type safety |
| PHP | `php -l` (lint) | Syntax check |
| Ruby | `ruby -c` (syntax) | Syntax check |
| React Native | `npx tsc --noEmit` | TypeScript types |

---

## 4. Release Workflow

```
1. Bump VERSION.txt → e.g., "3.15.0"
2. CMake configure → generates ufsecp_version.h from .in
3. Build + test on all CI platforms
4. Package each ecosystem:
   - pip sdist + wheel
   - npm pack
   - dotnet pack
   - mvn deploy
   - cargo publish
   - dart pub publish
   - pod trunk push
   - gem push
5. Create GitHub release with:
   - SBOM (generate_sbom.sh)
   - Checksums (SHA-256)
   - Signature (cosign)
6. Verify reproducible build across 2+ machines
```
