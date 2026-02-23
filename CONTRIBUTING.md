# Contributing to UltrafastSecp256k1

Thank you for your interest in contributing to UltrafastSecp256k1! This document provides guidelines for contributing to the project.

## ⚠️ Requirements for Acceptable Contributions

All contributions **MUST** comply with the following before they can be accepted:

1. **Coding Standards** — read and follow the [Coding Standards](https://github.com/shrec/UltrafastSecp256k1/blob/main/docs/CODING_STANDARDS.md) document in full
2. **All tests pass** — `ctest --test-dir build-dev --output-on-failure`
3. **Code formatted** — `clang-format -i <files>` (`.clang-format` config in repo root)
4. **No compiler warnings** — clean build with `-Wall -Wextra`
5. **License** — all contributions are licensed under [AGPL-3.0-or-later](https://github.com/shrec/UltrafastSecp256k1/blob/main/LICENSE)
6. **Security** — follow the [Security Policy](https://github.com/shrec/UltrafastSecp256k1/blob/main/SECURITY.md); never open public issues for vulnerabilities

Pull requests that do not meet these requirements will be rejected.

## 📋 Table of Contents

- [Requirements for Acceptable Contributions](#️-requirements-for-acceptable-contributions)
- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Process](#development-process)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Reporting Issues](#reporting-issues)
- [Areas for Contribution](#areas-for-contribution)

## 🤝 Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow
- Maintain professional communication

## 🚀 Getting Started

### Prerequisites

```bash
# Install dependencies
# Ubuntu/Debian
sudo apt install cmake ninja-build g++-13 clang-tidy

# Arch Linux
sudo pacman -S cmake ninja gcc clang

# macOS
brew install cmake ninja llvm
```

### Development Build

```bash
git clone https://github.com/shrec/UltrafastSecp256k1.git
cd UltrafastSecp256k1
cmake -S . -B build-dev -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DSECP256K1_BUILD_TESTS=ON
cmake --build build-dev -j
```

## 🔄 Development Process

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Commit** your changes: `git commit -m 'Add amazing feature'`
4. **Push** to the branch: `git push origin feature/amazing-feature`
5. **Open** a Pull Request

### Branch Naming

- `feature/` - New features
- `fix/` - Bug fixes
- `perf/` - Performance improvements
- `docs/` - Documentation updates
- `refactor/` - Code refactoring

## 📝 Coding Standards

> **Full reference:** [docs/CODING_STANDARDS.md](https://github.com/shrec/UltrafastSecp256k1/blob/main/docs/CODING_STANDARDS.md)

The complete coding standards document covers naming, formatting, hot-path contracts, memory model, cryptographic correctness, GPU rules, and commit standards. Below is a summary.

### C++ Style

- **Standard**: C++20
- **Formatting**: ClangFormat (`.clang-format` provided)
- **Naming Conventions**:
  - Classes: `PascalCase`
  - Functions: `snake_case`
  - Variables: `snake_case`
  - Constants: `UPPER_SNAKE_CASE`
  - Member variables: `m_snake_case` or `snake_case_`

### Code Example

```cpp
namespace secp256k1::fast {

class FieldElement {
public:
    FieldElement() = default;
    
    static FieldElement from_hex(std::string_view hex);
    
    FieldElement operator+(const FieldElement& other) const;
    
private:
    std::array<uint64_t, 4> m_limbs{};
};

} // namespace secp256k1::fast
```

### Documentation

- Use Doxygen-style comments for public APIs
- Document complex algorithms
- Include performance characteristics where relevant

```cpp
/**
 * @brief Multiply two field elements modulo p
 * 
 * @param a First operand
 * @param b Second operand
 * @return Product a * b (mod p)
 * 
 * @note This operation is constant-time
 * @performance ~8ns on x86-64 with assembly, ~25ns portable
 */
FieldElement field_mul(const FieldElement& a, const FieldElement& b);
```

## 🧪 Testing

### Running Tests

```bash
# All tests
ctest --test-dir build-dev --output-on-failure

# Specific test
./build-dev/cpu/tests/test_field

# With verbose output
ctest --test-dir build-dev -V
```

### Adding Tests

- Place tests in `cpu/tests/` or `cuda/tests/`
- Use descriptive test names
- Test edge cases and error conditions
- Include performance regression tests

```cpp
TEST(FieldElement, MultiplicationIsCommutative) {
    auto a = FieldElement::from_hex("1234...");
    auto b = FieldElement::from_hex("5678...");
    
    EXPECT_EQ(a * b, b * a);
}
```

## 📤 Pull Request Process

### Before Submitting

1. **Build** successfully: `cmake --build build-dev`
2. **Pass all tests**: `ctest --test-dir build-dev`
3. **Format code**: `clang-format -i <files>`
4. **Run clang-tidy**: `clang-tidy -p build-dev cpu/src/*.cpp`
5. **Update documentation** if needed
6. **Add tests** for new features

A PR checklist template is automatically applied — see [.github/PULL_REQUEST_TEMPLATE.md](https://github.com/shrec/UltrafastSecp256k1/blob/main/.github/PULL_REQUEST_TEMPLATE.md).

### Review Process

- Maintainers will review within 48-72 hours
- Address feedback in new commits (don't force push)
- Once approved, maintainers will merge

## 🎯 Areas for Contribution

### High Priority

- **Formal verification** of field/scalar arithmetic
- **Side-channel analysis** and hardening (cache-timing, power analysis)
- **Performance benchmarking** on new hardware (Apple M3/M4, Intel Raptor Lake, AMD Zen 5)
- **GPU kernel optimization** (occupancy, register pressure, warp-level primitives)
- **Additional signature schemes** (EdDSA/Ed25519, multi-sig)

### Good First Issues

- Documentation improvements and typo fixes
- Example programs (key derivation, address generation, HD wallets)
- Test coverage improvements (edge cases, error paths)
- Build system enhancements (new compilers, package managers)
- Localization of documentation

### Advanced Contributions

- **FPGA** acceleration port
- **New embedded targets** (nRF52, RP2040, AVR)
- **Multi-scalar multiplication** (Pippenger, Straus)
- **Batch verification** for ECDSA and Schnorr signatures
- **Zero-knowledge proof** integration
- **Threshold signatures** (FROST, GG20)

### Already Implemented ✅

The following were previously listed as desired contributions and are now part of v3.12:

- ✅ ARM64/AArch64 assembly optimizations (MUL/UMULH)
- ✅ OpenCL implementation (3.39M kG/s)
- ✅ WebAssembly port (Emscripten, npm package)
- ✅ Constant-time layer (ct:: namespace)
- ✅ ECDSA signatures (RFC 6979)
- ✅ Schnorr signatures (BIP-340)
- ✅ iOS support (XCFramework, SPM, CocoaPods)
- ✅ Android NDK support
- ✅ ROCm/HIP GPU support
- ✅ ESP32/STM32 embedded support
- ✅ Linux distribution packaging (DEB, RPM, Arch/AUR)
- ✅ Docker multi-stage build
- ✅ Clang-tidy CI integration
- ✅ GitHub Scorecard + OpenSSF Best Practices badge

## 🐛 Reporting Issues

### Bug Reports

Include:
- **Description**: What happened vs. what should happen
- **Steps to reproduce**: Minimal reproducible example
- **Environment**: OS, compiler, CMake version, CPU architecture
- **Build configuration**: CMake options used
- **Logs**: Relevant error messages or stack traces

### Feature Requests

Include:
- **Use case**: What problem does it solve?
- **Proposed API**: How would you like to use it?
- **Alternatives**: What workarounds exist?

## 📚 Resources

- [Documentation Index](docs/README.md)
- [API Reference](docs/API_REFERENCE.md)
- [Building Guide](docs/BUILDING.md)
- [Benchmarks](docs/BENCHMARKS.md)
- [Security Policy](SECURITY.md)
- [Changelog](CHANGELOG.md)

## 📧 Contact

- **Issues**: [GitHub Issues](https://github.com/shrec/UltrafastSecp256k1/issues)
- **Discussions**: [GitHub Discussions](https://github.com/shrec/UltrafastSecp256k1/discussions)

---

Thank you for contributing to UltrafastSecp256k1! 🎉
