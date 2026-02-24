# UltrafastSecp256k1 — Fastest Open-Source secp256k1 Library

**Zero-dependency, multi-backend secp256k1 elliptic curve cryptography library** — GPU-accelerated ECDSA & Schnorr signatures, constant-time side-channel protection, 12+ platform targets inc. CUDA, Metal, OpenCL, ROCm, WebAssembly, RISC-V, ESP32, and STM32.

> **4.88 M ECDSA signs/s** · **2.44 M ECDSA verifies/s** · **3.66 M Schnorr signs/s** · **2.82 M Schnorr verifies/s** — single GPU (RTX 5060 Ti)

### Why UltrafastSecp256k1?

- **Fastest open-source GPU signatures** — no other library provides secp256k1 ECDSA + Schnorr sign/verify on CUDA, OpenCL, and Metal ([reproducible benchmark suite and raw logs](docs/BENCHMARKS.md))
- **Zero dependencies** — pure C++20, no Boost, no OpenSSL, compiles anywhere with a conforming compiler
- **Dual-layer security** — variable-time FAST path for throughput, constant-time CT path for secret-key operations
- **12+ platforms** — x86-64, ARM64, RISC-V, WASM, iOS, Android, ESP32, STM32, CUDA, Metal, OpenCL, ROCm

> **Benchmark reproducibility:** All numbers come from pinned compiler/driver/toolkit versions with exact commands and raw logs. See [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) (methodology) and the [live dashboard](https://shrec.github.io/UltrafastSecp256k1/dev/bench/).

**Quick links:** [Discord](https://discord.gg/sUmW7cc5) · [Benchmarks](docs/BENCHMARKS.md) · [Build Guide](docs/BUILDING.md) · [API Reference](docs/API_REFERENCE.md) · [Security Policy](SECURITY.md) · [Threat Model](THREAT_MODEL.md) · [Porting Guide](PORTING.md)

---

[![GitHub stars](https://img.shields.io/github/stars/shrec/UltrafastSecp256k1?style=flat-square&logo=github&label=Stars)](https://github.com/shrec/UltrafastSecp256k1/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/shrec/UltrafastSecp256k1?style=flat-square&logo=github&label=Forks)](https://github.com/shrec/UltrafastSecp256k1/network/members)
[![CI](https://img.shields.io/github/actions/workflow/status/shrec/UltrafastSecp256k1/ci.yml?branch=main&label=CI)](https://github.com/shrec/UltrafastSecp256k1/actions/workflows/ci.yml)
[![Benchmark](https://img.shields.io/github/actions/workflow/status/shrec/UltrafastSecp256k1/benchmark.yml?branch=main&label=Bench)](https://shrec.github.io/UltrafastSecp256k1/dev/bench/)
[![Release](https://img.shields.io/github/v/release/shrec/UltrafastSecp256k1?label=Release)](https://github.com/shrec/UltrafastSecp256k1/releases/latest)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](https://en.cppreference.com/w/cpp/20)
[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/shrec/UltrafastSecp256k1/badge)](https://scorecard.dev/viewer/?uri=github.com/shrec/UltrafastSecp256k1)
[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/12011/badge)](https://www.bestpractices.dev/projects/12011)
[![CodeQL](https://github.com/shrec/UltrafastSecp256k1/actions/workflows/codeql.yml/badge.svg)](https://github.com/shrec/UltrafastSecp256k1/actions/workflows/codeql.yml)
[![Security Audit](https://github.com/shrec/UltrafastSecp256k1/actions/workflows/security-audit.yml/badge.svg)](https://github.com/shrec/UltrafastSecp256k1/actions/workflows/security-audit.yml)
[![Clang-Tidy](https://github.com/shrec/UltrafastSecp256k1/actions/workflows/clang-tidy.yml/badge.svg)](https://github.com/shrec/UltrafastSecp256k1/actions/workflows/clang-tidy.yml)
[![SonarCloud](https://sonarcloud.io/api/project_badges/measure?project=shrec_UltrafastSecp256k1&metric=security_rating)](https://sonarcloud.io/summary/overall?id=shrec_UltrafastSecp256k1)
[![codecov](https://codecov.io/gh/shrec/UltrafastSecp256k1/graph/badge.svg)](https://codecov.io/gh/shrec/UltrafastSecp256k1)
[![Discord](https://img.shields.io/badge/Discord-Join%20Us-5865F2?logo=discord&logoColor=white)](https://discord.gg/sUmW7cc5)

**Supported Blockchains (secp256k1-based):**

[![Bitcoin](https://img.shields.io/badge/Bitcoin-BTC-F7931A.svg?logo=bitcoin&logoColor=white)](https://bitcoin.org)
[![Ethereum](https://img.shields.io/badge/Ethereum-ETH-3C3C3D.svg?logo=ethereum&logoColor=white)](https://ethereum.org)
[![Litecoin](https://img.shields.io/badge/Litecoin-LTC-A6A9AA.svg?logo=litecoin&logoColor=white)](https://litecoin.org)
[![Dogecoin](https://img.shields.io/badge/Dogecoin-DOGE-C2A633.svg?logo=dogecoin&logoColor=white)](https://dogecoin.com)
[![Bitcoin Cash](https://img.shields.io/badge/Bitcoin%20Cash-BCH-8DC351.svg?logo=bitcoincash&logoColor=white)](https://bitcoincash.org)
[![Zcash](https://img.shields.io/badge/Zcash-ZEC-F4B728.svg)](https://z.cash)
[![Dash](https://img.shields.io/badge/Dash-DASH-008CE7.svg?logo=dash&logoColor=white)](https://dash.org)
[![BNB Chain](https://img.shields.io/badge/BNB%20Chain-BNB-F0B90B.svg?logo=binance&logoColor=white)](https://www.bnbchain.org)
[![Polygon](https://img.shields.io/badge/Polygon-MATIC-8247E5.svg?logo=polygon&logoColor=white)](https://polygon.technology)
[![Avalanche](https://img.shields.io/badge/Avalanche-AVAX-E84142.svg?logo=avalanche&logoColor=white)](https://avax.network)
[![Arbitrum](https://img.shields.io/badge/Arbitrum-ARB-28A0F0.svg)](https://arbitrum.io)
[![Optimism](https://img.shields.io/badge/Optimism-OP-FF0420.svg)](https://optimism.io)
[![+15 more](https://img.shields.io/badge/+15%20more-secp256k1%20coins-grey.svg)](#secp256k1-supported-coins-27-blockchains)

**GPU & Platform Support:**

[![CUDA](https://img.shields.io/badge/CUDA-12.0+-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![OpenCL](https://img.shields.io/badge/OpenCL-3.0-green.svg)](https://www.khronos.org/opencl/)
[![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-M1%2FM2%2FM3%2FM4-black.svg?logo=apple)](metal/)
[![Metal](https://img.shields.io/badge/Metal-GPU%20Compute-silver.svg?logo=apple)](metal/)
[![ROCm](https://img.shields.io/badge/ROCm-6.3%20HIP-red.svg)](cuda/README.md)
[![WebAssembly](https://img.shields.io/badge/WebAssembly-Emscripten-purple.svg)](wasm/)
[![ARM64](https://img.shields.io/badge/ARM64-Cortex--A55%2FA76-orange.svg)](https://developer.android.com/ndk)
[![RISC-V](https://img.shields.io/badge/RISC--V-RV64GC-orange.svg)](https://riscv.org/)
[![Android](https://img.shields.io/badge/Android-NDK%20r27-brightgreen.svg)](android/)
[![iOS](https://img.shields.io/badge/iOS-17%2B%20XCFramework-lightgrey.svg)](cmake/ios.toolchain.cmake)
[![ESP32-S3](https://img.shields.io/badge/ESP32--S3-Xtensa%20LX7-orange.svg)](https://www.espressif.com/en/products/socs/esp32-s3)
[![ESP32](https://img.shields.io/badge/ESP32-Xtensa%20LX6-orange.svg)](https://www.espressif.com/en/products/socs/esp32)
[![STM32](https://img.shields.io/badge/STM32-Cortex--M3-orange.svg)](https://www.st.com/en/microcontrollers-microprocessors/stm32f103ze.html)

---

## ⚠️ Security Notice

**Research & Development Project — Not Audited**

This library has **not undergone independent security audits**. It is provided for research, educational, and experimental purposes.

- ❌ Not recommended for production without independent cryptographic audit
- ✅ All self-tests pass (76/76 including all backends)
- ✅ Dual-layer constant-time architecture (FAST + CT always active)
- ✅ Stable C ABI (`ufsecp`) with 45 exported functions
- ✅ Fuzz-tested core arithmetic (libFuzzer + ASan)

**Report vulnerabilities** via [GitHub Security Advisories](https://github.com/shrec/UltrafastSecp256k1/security/advisories/new) or email [payysoon@gmail.com](mailto:payysoon@gmail.com).
For production cryptographic systems, prefer audited libraries like [libsecp256k1](https://github.com/bitcoin-core/secp256k1).

---

## secp256k1 Feature Overview

Features are organized into **maturity tiers** (see [SUPPORTED_GUARANTEES.md](include/ufsecp/SUPPORTED_GUARANTEES.md) for detailed guarantees):

| Tier | Category | Component | Status |
|------|----------|-----------|--------|
| **1 — Core** | Field / Scalar / Point | GLV, Precompute, Batch Inverse | ✅ |
| **1 — Core** | Assembly | x64 MASM/GAS, BMI2/ADX, ARM64, RISC-V RV64GC | ✅ |
| **1 — Core** | SIMD | AVX2/AVX-512 batch ops, Montgomery batch inverse | ✅ |
| **1 — Core** | Constant-Time | CT field/scalar/point — no secret-dependent branches | ✅ |
| **1 — Core** | ECDSA | Sign/Verify, RFC 6979, DER/Compact, low-S, Recovery | ✅ |
| **1 — Core** | Schnorr | BIP-340 sign/verify, tagged hashing, x-only pubkeys | ✅ |
| **1 — Core** | ECDH | Key exchange (raw, xonly, SHA-256) | ✅ |
| **1 — Core** | Multi-scalar | Strauss/Shamir dual-scalar multiplication | ✅ |
| **1 — Core** | Batch verify | ECDSA + Schnorr batch verification | ✅ |
| **1 — Core** | Hashing | SHA-256 (SHA-NI), SHA-512, HMAC, Keccak-256 | ✅ |
| **1 — Core** | C ABI | `ufsecp` stable FFI (45 exports) | ✅ |
| **2 — Protocol** | BIP-32/44 | HD derivation, path parsing, xprv/xpub, coin-type | ✅ |
| **2 — Protocol** | Taproot | BIP-341/342, tweak, Merkle tree | ✅ |
| **2 — Protocol** | MuSig2 | BIP-327, key aggregation, 2-round signing | ✅ |
| **2 — Protocol** | FROST | Threshold signatures, t-of-n | ✅ |
| **2 — Protocol** | Adaptor | Schnorr + ECDSA adaptor signatures | ✅ |
| **2 — Protocol** | Pedersen | Commitments, homomorphic, switch commitments | ✅ |
| **3 — Convenience** | Address | P2PKH, P2WPKH, P2TR, Base58, Bech32/m, EIP-55 | ✅ |
| **3 — Convenience** | Coins | 27 blockchains, auto-dispatch | ✅ |
| — | GPU | CUDA, Metal, OpenCL, ROCm kernels | ✅ |
| — | Platforms | x64, ARM64, RISC-V, ESP32, STM32, WASM, iOS, Android | ✅ |

> **Tier 1** = battle-tested core crypto with stable API. **Tier 2** = protocol-level features, API may evolve. **Tier 3** = convenience utilities.

---

## 60-Second Quickstart

Get a working selftest in under a minute:

**Option A — Linux (apt)**
```bash
sudo apt install libufsecp3
ufsecp_selftest          # Expected: "OK (version 3.x, backend CPU)"
```

**Option B — npm (any OS)**
```bash
npm i ufsecp
node -e "require('ufsecp').selftest()"   # Expected: "OK"
```

**Option C — Python (any OS)**
```bash
pip install ufsecp
python -c "import ufsecp; ufsecp.selftest()"  # Expected: "OK"
```

**Option D — Build from source**
```bash
git clone https://github.com/shrec/UltrafastSecp256k1.git && cd UltrafastSecp256k1
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
./build/selftest          # Expected: "ALL TESTS PASSED"
```

---

## Platform Support Matrix

| Target | Backend | Install / Entry Point | Status |
|--------|---------|----------------------|--------|
| **Linux x64** | CPU | `apt install libufsecp3` | ✅ Stable |
| **Windows x64** | CPU | NuGet `UltrafastSecp256k1` / [Release .zip](https://github.com/shrec/UltrafastSecp256k1/releases) | ✅ Stable |
| **macOS (x64/ARM64)** | CPU + Metal | `brew install ufsecp` / build from source | ✅ Stable |
| **Android ARM64** | CPU | `implementation 'io.github.shrec:ufsecp'` (Maven) | ✅ Stable |
| **iOS ARM64** | CPU | Swift Package / CocoaPods / XCFramework | ✅ Stable |
| **Browser / Node.js** | WASM | `npm i ufsecp` | ✅ Stable |
| **ESP32-S3 / ESP32** | CPU | PlatformIO / IDF component | ✅ Tested |
| **STM32 (Cortex-M)** | CPU | CMake cross-compile | ✅ Tested |
| **NVIDIA GPU** | CUDA 12+ | Build with `-DSECP256K1_BUILD_CUDA=ON` | ✅ Stable |
| **AMD GPU** | ROCm/HIP | Build with `-DSECP256K1_BUILD_ROCM=ON` | ⚠️ Beta |
| **Apple GPU** | Metal | Build with Metal backend | ✅ Stable |
| **Any GPU** | OpenCL | Build with `-DSECP256K1_BUILD_OPENCL=ON` | ⚠️ Beta |
| **RISC-V (RV64GC)** | CPU | Cross-compile | ✅ Tested |

---

## Installation

### Linux (APT — Debian / Ubuntu)

```bash
# Add repository
curl -fsSL https://shrec.github.io/UltrafastSecp256k1/apt/KEY.gpg | sudo gpg --dearmor -o /etc/apt/keyrings/ultrafastsecp256k1.gpg
echo "deb [signed-by=/etc/apt/keyrings/ultrafastsecp256k1.gpg] https://shrec.github.io/UltrafastSecp256k1/apt stable main" \
  | sudo tee /etc/apt/sources.list.d/ultrafastsecp256k1.list
sudo apt update

# Install (runtime only)
sudo apt install libufsecp3

# Install (development — headers, static lib, cmake/pkgconfig)
sudo apt install libufsecp-dev
```

### Linux (RPM — Fedora / RHEL)

```bash
# Download from GitHub Releases
curl -LO https://github.com/shrec/UltrafastSecp256k1/releases/latest/download/UltrafastSecp256k1-*.rpm
sudo dnf install ./UltrafastSecp256k1-*.rpm
```

### Arch Linux (AUR)

```bash
# Using yay
yay -S libufsecp

# Or manually
git clone https://aur.archlinux.org/libufsecp.git
cd libufsecp && makepkg -si
```

### From source (any platform)

```bash
cmake -S . -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr \
    -DSECP256K1_BUILD_SHARED=ON \
    -DSECP256K1_INSTALL=ON \
    -DSECP256K1_USE_ASM=ON
cmake --build build -j$(nproc)
sudo cmake --install build
sudo ldconfig
```

### Use in your CMake project

```cmake
find_package(ufsecp 3 REQUIRED)
target_link_libraries(myapp PRIVATE ufsecp::ufsecp)
```

### Use with pkg-config

```bash
g++ myapp.cpp $(pkg-config --cflags --libs ufsecp) -o myapp
```

---

## secp256k1 GPU Acceleration (CUDA / OpenCL / Metal / ROCm)

UltrafastSecp256k1 is the **only open-source library** that provides full secp256k1 ECDSA + Schnorr sign/verify on GPU across four backends (as of February 2026; if you know of another, [please let us know](https://github.com/shrec/UltrafastSecp256k1/issues)):

| Backend | Hardware | kG/s | ECDSA Sign | ECDSA Verify | Schnorr Sign | Schnorr Verify |
|---------|----------|------|------------|--------------|--------------|----------------|
| **CUDA** | RTX 5060 Ti | 4.59 M/s | 4.88 M/s | 2.44 M/s | 3.66 M/s | 2.82 M/s |
| **OpenCL** | RTX 5060 Ti | 3.39 M/s | — | — | — | — |
| **Metal** | Apple M3 Pro | 0.33 M/s | — | — | — | — |
| **ROCm (HIP)** | AMD GPUs | Portable | — | — | — | — |

*CUDA 12.0, sm_86;sm_89, batch=16K signatures. Metal 2.4, 8×32-bit Comba limbs, 18 GPU cores.*

### CUDA Core ECC Operations (Kernel-Only Throughput)

| Operation | Time/Op | Throughput |
|-----------|---------|------------|
| Field Mul | 0.2 ns | 4,142 M/s |
| Field Add | 0.2 ns | 4,130 M/s |
| Field Inv | 10.2 ns | 98.35 M/s |
| Point Add | 1.6 ns | 619 M/s |
| Point Double | 0.8 ns | 1,282 M/s |
| Scalar Mul (P×k) | 225.8 ns | 4.43 M/s |
| Generator Mul (G×k) | 217.7 ns | 4.59 M/s |
| Batch Inv (Montgomery) | 2.9 ns | 340 M/s |
| Jac→Affine (per-pt) | 14.9 ns | 66.9 M/s |

### GPU Signature Operations (ECDSA + Schnorr)

| Operation | Time/Op | Throughput | Protocol |
|-----------|---------|------------|----------|
| **ECDSA Sign** | **204.8 ns** | **4.88 M/s** | RFC 6979 + low-S |
| **ECDSA Verify** | **410.1 ns** | **2.44 M/s** | Shamir + GLV |
| **ECDSA Sign+Recid** | **311.5 ns** | **3.21 M/s** | Recoverable (EIP-155) |
| **Schnorr Sign** | **273.4 ns** | **3.66 M/s** | BIP-340 |
| **Schnorr Verify** | **354.6 ns** | **2.82 M/s** | BIP-340 + GLV |

### CUDA vs OpenCL Comparison (RTX 5060 Ti)

| Operation | CUDA | OpenCL | Winner |
|-----------|------|--------|--------|
| Field Mul | 0.2 ns | 0.2 ns | Tie |
| Field Inv | 10.2 ns | 14.3 ns | **CUDA 1.40×** |
| Point Double | 0.8 ns | 0.9 ns | **CUDA 1.13×** |
| Point Add | 1.6 ns | 1.6 ns | Tie |
| kG (Generator Mul) | 217.7 ns | 295.1 ns | **CUDA 1.36×** |

*Benchmarks: 2026-02-14, Linux x86_64, NVIDIA Driver 580.126.09. Both kernel-only (no buffer allocation/copy overhead).*

### Apple Metal (M3 Pro) — Kernel-Only

| Operation | Time/Op | Throughput |
|-----------|---------|------------|
| Field Mul | 1.9 ns | 527 M/s |
| Field Inv | 106.4 ns | 9.40 M/s |
| Point Add | 10.1 ns | 98.6 M/s |
| Point Double | 5.1 ns | 196 M/s |
| Scalar Mul (P×k) | 2.94 μs | 0.34 M/s |
| Generator Mul (G×k) | 3.00 μs | 0.33 M/s |

*Metal 2.4, 8×32-bit Comba limbs, Apple M3 Pro (18 GPU cores, Unified Memory 18 GB)*

---

## secp256k1 ECDSA & Schnorr Signatures (BIP-340, RFC 6979)

Full signature support across CPU and GPU:

- **ECDSA**: RFC 6979 deterministic nonces, low-S normalization, DER/Compact encoding, public key recovery (recid)
- **Schnorr**: BIP-340 compliant — tagged hashing, x-only public keys
- **Batch verification**: ECDSA and Schnorr batch verify
- **Multi-scalar**: Shamir's trick (k₁×G + k₂×Q) for fast verification

### CPU Signature Benchmarks (x86-64, Clang 19, AVX2, Release)

| Operation | Time | Throughput |
|-----------|------:|----------:|
| ECDSA Sign (RFC 6979) | 8.5 μs | 118,000 op/s |
| ECDSA Verify | 23.6 μs | 42,400 op/s |
| Schnorr Sign (BIP-340) | 6.8 μs | 146,000 op/s |
| Schnorr Verify (BIP-340) | 24.0 μs | 41,600 op/s |
| Key Generation (CT) | 9.5 μs | 105,500 op/s |
| Key Generation (fast) | 5.5 μs | 182,000 op/s |
| ECDH | 23.9 μs | 41,800 op/s |

*Schnorr sign is ~25% faster than ECDSA sign due to simpler nonce derivation (no modular inverse). Measured single-core, pinned, 2026-02-21.*

---

## Constant-Time secp256k1 (Side-Channel Resistance)

The `ct::` namespace provides constant-time operations for secret-key material — no secret-dependent branches or memory access patterns:

| Operation | Fast | CT | Overhead |
|-----------|------:|------:|--------:|
| Field Mul | 17 ns | 23 ns | 1.08× |
| Field Inverse | 0.8 μs | 1.7 μs | 2.05× |
| Complete Addition | — | 276 ns | — |
| Scalar Mul (k×P) | 23.6 μs | 26.6 μs | 1.13× |
| Generator Mul (k×G) | 5.3 μs | 9.9 μs | 1.86× |

**CT layer provides:** `ct::field_mul`, `ct::field_inv`, `ct::scalar_mul`, `ct::point_add_complete`, `ct::point_dbl`

**Use the CT layer for**: private key operations, signing, nonce generation, ECDH.
**Use the FAST layer for**: verification, public key derivation, batch processing, benchmarks.

See [THREAT_MODEL.md](THREAT_MODEL.md) for a full layer-by-layer risk assessment.

### CT Evidence & Methodology

| Evidence | Scope | Status |
|----------|-------|--------|
| **No secret-dependent branches** | All `ct::` functions | ✅ Enforced by design, verified via Clang-Tidy checks |
| **No secret-dependent memory access** | All `ct::` table lookups use constant-index cmov | ✅ |
| **ASan + UBSan CI** | Every push — catches undefined behavior in CT paths | ✅ CI |
| **Timing tests (dudect)** | CPU field/scalar ops | 🔜 Planned (see [roadmap](ROADMAP.md)) |
| **Formal CT verification** | Fiat-Crypto style | 🔜 Planned |

**Assumptions:** CT guarantees depend on compiler not introducing secret-dependent branches during optimization. Builds use `-O2` with Clang; MSVC may require additional flags. Micro-architectural side channels (Spectre, power analysis) are outside current scope — see [THREAT_MODEL.md](THREAT_MODEL.md).

---

## secp256k1 Benchmarks — Cross-Platform Comparison

### CPU: x86-64 vs ARM64 vs RISC-V

| Operation | x86-64 (Clang 21, AVX2) | ARM64 (Cortex-A76) | RISC-V (Milk-V Mars) |
|-----------|-------------------------:|--------------------:|---------------------:|
| Field Mul | 17 ns | 74 ns | 95 ns |
| Field Square | 14 ns | 50 ns | 70 ns |
| Field Add | 1 ns | 8 ns | 11 ns |
| Field Inverse | 1 μs | 2 μs | 4 μs |
| Point Add | 159 ns | 992 ns | 1 μs |
| Generator Mul (k×G) | 5 μs | 14 μs | 33 μs |
| Scalar Mul (k×P) | 25 μs | 131 μs | 154 μs |

### GPU: CUDA vs OpenCL vs Metal

| Operation | CUDA (RTX 5060 Ti) | OpenCL (RTX 5060 Ti) | Metal (M3 Pro) |
|-----------|--------------------:|---------------------:|---------------:|
| Field Mul | 0.2 ns | 0.2 ns | 1.9 ns |
| Field Inv | 10.2 ns | 14.3 ns | 106.4 ns |
| Point Add | 1.6 ns | 1.6 ns | 10.1 ns |
| Generator Mul (G×k) | 217.7 ns | 295.1 ns | 3.00 μs |

### Embedded: ESP32-S3 vs ESP32 vs STM32

| Operation | ESP32-S3 LX7 (240 MHz) | ESP32 LX6 (240 MHz) | STM32F103 (72 MHz) |
|-----------|-------------------:|-------------------:|-------------------:|
| Field Mul | 6,105 ns | 6,993 ns | 15,331 ns |
| Field Square | 5,020 ns | 6,247 ns | 12,083 ns |
| Field Add | 850 ns | 985 ns | 4,139 ns |
| Field Inv | 2,524 μs | 609 μs | 1,645 μs |
| **Fast** Scalar × G | 5,226 μs | 6,203 μs | 37,982 μs |
| **CT** Scalar × G | 15,527 μs | — | — |
| **CT** Generator × k | 4,951 μs | — | — |

### Field Representation: 5×52 vs 4×64

| Operation | 4×64 | 5×52 | Speedup |
|-----------|------:|------:|--------:|
| Multiplication | 42 ns | 15 ns | **2.76×** |
| Squaring | 31 ns | 13 ns | **2.44×** |
| Addition | 4.3 ns | 1.6 ns | **2.69×** |
| Add chain (32 ops) | 286 ns | 57 ns | **5.01×** |

*5×52 uses `__int128` lazy reduction — ideal for 64-bit platforms.*

For full benchmark results, see [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

---

## secp256k1 on Embedded (ESP32 / STM32 / ARM Cortex-M)

UltrafastSecp256k1 runs on resource-constrained microcontrollers with **portable C++ (no `__int128`, no assembly required)**:

- **ESP32-S3** (Xtensa LX7 @ 240 MHz): Fast scalar × G in 5.2 ms, **CT generator × k in 4.9 ms**
- **ESP32-PICO-D4** (Xtensa LX6 @ 240 MHz): Scalar × G in 6.2 ms, CT layer available (44.8 ms CT)
- **STM32F103** (ARM Cortex-M3 @ 72 MHz): Scalar × G in 38 ms with ARM inline assembly (UMULL/ADDS/ADCS)
- **Android ARM64** (RK3588, Cortex-A76 @ 2.256 GHz): Scalar × G in 14 μs, Scalar × P in 131 μs, ECDSA Sign 30 μs

All 37 library tests pass on every embedded target. See [examples/esp32_test/](examples/esp32_test/) and [examples/stm32_test/](examples/stm32_test/).

### Porting to New Platforms

See [PORTING.md](PORTING.md) for a step-by-step checklist to add new CPU architectures, embedded targets, or GPU backends.

---

## WASM secp256k1 (Browser & Node.js)

WebAssembly build via Emscripten — runs secp256k1 in any modern browser or Node.js:

```bash
./scripts/build_wasm.sh        # → build/wasm/dist/
```

Output: `secp256k1_wasm.wasm` + `secp256k1.mjs` (ES6 module with TypeScript declarations).
See [wasm/README.md](wasm/README.md) for JavaScript/TypeScript integration.

---

## secp256k1 Batch Modular Inverse (Montgomery Trick)

All backends include **batch modular inversion** — a critical building block for Jacobian→Affine conversion:

| Backend | Function | Notes |
|---------|----------|-------|
| **CPU** | `fe_batch_inverse(FieldElement*, size_t)` | Montgomery trick with scratch buffer |
| **CUDA** | `batch_inverse_montgomery` / `batch_inverse_kernel` | GPU Montgomery trick kernel |
| **Metal** | `batch_inverse` | Chunked parallel threadgroups |
| **OpenCL** | Inline PTX inverse | Batch via host orchestration |

**Algorithm**: Montgomery batch inverse computes N field inversions using only **1 modular inversion + 3(N−1) multiplications**, amortizing the expensive inversion across the entire batch.

For N=1024: ~500× cheaper than individual inversions. A single field inversion costs ~3.5 μs (Fermat), while batch amortizes to ~7 ns per element.

### Mixed Addition (Jacobian + Affine)

Branchless mixed addition (`add_mixed_inplace`) uses the **madd-2007-bl** formula: **7M + 4S** (vs 11M + 5S for full Jacobian add).

```cpp
#include <secp256k1/point.hpp>
using namespace secp256k1::fast;

Point P = Point::generator();
FieldElement gx = P.x(), gy = P.y();

// Compute 2G using mixed add (7M + 4S)
Point Q = Point::generator();
Q.add_mixed_inplace(gx, gy);  // Q = G + G = 2G

// Batch walk: P, P+G, P+2G, ...
Point walker = P;
for (int i = 0; i < 1000; ++i) {
    walker.add_mixed_inplace(gx, gy);  // walker += G each step
}
```

### GPU Pattern: H-Product Serial Inversion

Production GPU apps use a memory-efficient variant: instead of storing full Z coordinates, `jacobian_add_mixed_h` returns **H = U2 − X1** separately. Since Z_k = Z_0 · H_0 · H_1 · … · H_{k-1}, the entire Z chain is invertible from H values + initial Z_0.

**Cost**: 1 Fermat inversion + 2N multiplications per thread (vs N Fermat inversions naively).

> See `apps/secp256k1_search_gpu_only/gpu_only.cu` (step kernel) + `unified_split.cuh` (batch inversion kernel)

---

## secp256k1 Stable C ABI (`ufsecp`) — FFI Bindings

Starting with **v3.4.0**, UltrafastSecp256k1 ships a stable C ABI — `ufsecp` — designed for FFI bindings (C#, Python, Rust, Go, Java, Node.js, etc.):

```
┌──────────────────────────────────────────────────┐
│                  Your Application                │
│          (C, C#, Python, Go, Rust, …)            │
└──────────────────┬───────────────────────────────┘
                   │  ufsecp C ABI (45 functions)
┌──────────────────▼───────────────────────────────┐
│           ufsecp.dll / libufsecp.so              │
│  Opaque ctx  │  Error model  │  ABI versioning   │
├──────────────┴───────────────┴───────────────────┤
│   FAST layer (variable-time public ops)          │
├──────────────────────────────────────────────────┤
│   CT layer (constant-time secret-key ops)        │
└──────────────────────────────────────────────────┘
```

**Default behavior:**
- **C ABI (`ufsecp`)**: Defaults to safe behavior — all secret-key operations (sign, derive, ECDH) use CT internally. No configuration needed.
- **C++ API**: Exposes both `fast::` and `ct::` namespaces — the developer chooses explicitly per call site.

### Quick Start (C)

```c
#include "ufsecp.h"

ufsecp_ctx* ctx = NULL;
ufsecp_ctx_create(&ctx);

// Generate keypair
unsigned char seckey[32], pubkey[33];
ufsecp_keygen(ctx, seckey, pubkey);

// ECDSA sign
unsigned char msg[32] = { /* SHA-256 hash */ };
unsigned char sig[64];
ufsecp_ecdsa_sign(ctx, seckey, msg, sig);

// Verify
int valid = 0;
ufsecp_ecdsa_verify(ctx, pubkey, 33, msg, sig, &valid);

ufsecp_ctx_destroy(ctx);
```

### API Coverage

| Category | Functions |
|----------|-----------|
| **Context** | `ctx_create`, `ctx_destroy`, `selftest`, `last_error` |
| **Keys** | `keygen`, `seckey_verify`, `pubkey_create`, `pubkey_parse`, `pubkey_serialize` |
| **ECDSA** | `ecdsa_sign`, `ecdsa_verify`, `ecdsa_sign_der`, `ecdsa_verify_der`, `ecdsa_recover` |
| **Schnorr** | `schnorr_sign`, `schnorr_verify` |
| **SHA-256** | `sha256` (SHA-NI accelerated) |
| **ECDH** | `ecdh_compressed`, `ecdh_xonly`, `ecdh_raw` |
| **BIP-32** | `bip32_from_seed`, `bip32_derive_child`, `bip32_serialize` |
| **Address** | `address_p2pkh`, `address_p2wpkh`, `address_p2tr` |
| **WIF** | `wif_encode`, `wif_decode` |
| **Tweak** | `pubkey_tweak_add`, `pubkey_tweak_mul` |
| **Version** | `version`, `abi_version`, `version_string` |

See [SUPPORTED_GUARANTEES.md](include/ufsecp/SUPPORTED_GUARANTEES.md) for Tier 1/2/3 stability guarantees.

---

## secp256k1 Use Cases

- **Transaction Signing & Verification** — Bitcoin, Ethereum, and 25+ blockchain transaction signing at CPU or GPU scale
- **Batch Signature Verification** — verify thousands of ECDSA/Schnorr signatures per second for block validation
- **HD Wallet Key Derivation** — BIP-32/44 hierarchical deterministic derivation with 27-coin address generation
- **Embedded IoT Signing** — ESP32 and STM32 on-device key generation and transaction signing
- **High-Throughput Indexing** — GPU-accelerated public key derivation for address indexing services
- **Zero-Knowledge Proof Systems** — Pedersen commitments, adaptor signatures for ZK protocols
- **Multi-Party Computation** — MuSig2 (BIP-327) and FROST threshold signing
- **Cross-Platform Cryptographic Services** — single codebase across server (CUDA), desktop (OpenCL/Metal), mobile (ARM64), browser (WASM), and embedded (ESP32/STM32)
- **Cryptographic Research & Benchmarking** — field/group operation microbenchmarks, algorithm variant comparison

> ### Testers Wanted
> We need community testers for platforms we cannot fully validate in CI:
> - **iOS** — Build & run on real iPhone/iPad hardware with Xcode
> - **AMD GPU (ROCm/HIP)** — Test on AMD Radeon RX / Instinct GPUs
>
> [Open an issue](https://github.com/shrec/UltrafastSecp256k1/issues) with your results!

---

## Building secp256k1 from Source (CMake)

### Prerequisites

- CMake 3.18+
- C++20 compiler (GCC 11+, Clang/LLVM 15+, MSVC 2022+ with `-DSECP256K1_ALLOW_MSVC=ON`)
- CUDA Toolkit 12.0+ (optional, for GPU)
- Ninja (recommended)

### CPU-Only Build

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

### With CUDA GPU Support

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DSECP256K1_BUILD_CUDA=ON
cmake --build build -j
```

### WebAssembly (Emscripten)

```bash
./scripts/build_wasm.sh        # → build/wasm/dist/
```

### iOS (XCFramework)

```bash
./scripts/build_xcframework.sh  # → build/xcframework/output/
```

Universal XCFramework (arm64 device + arm64 simulator). Also available via **Swift Package Manager** and **CocoaPods**.

### Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `SECP256K1_USE_ASM` | ON | Assembly optimizations (x64/ARM64/RISC-V) |
| `SECP256K1_BUILD_CUDA` | OFF | CUDA GPU support |
| `SECP256K1_BUILD_OPENCL` | OFF | OpenCL GPU support |
| `SECP256K1_BUILD_ROCM` | OFF | ROCm/HIP GPU support (AMD) |
| `SECP256K1_BUILD_TESTS` | ON | Test suite |
| `SECP256K1_BUILD_BENCH` | ON | Benchmarks |
| `SECP256K1_RISCV_FAST_REDUCTION` | ON | Fast modular reduction (RISC-V) |
| `SECP256K1_RISCV_USE_VECTOR` | ON | RVV vector extension (RISC-V) |

For detailed build instructions, see [docs/BUILDING.md](docs/BUILDING.md).

---

## secp256k1 Quick Start (C++ Examples)

### Basic Point Operations

```cpp
#include <secp256k1/field.hpp>
#include <secp256k1/point.hpp>
#include <secp256k1/scalar.hpp>
#include <iostream>

using namespace secp256k1::fast;

int main() {
    // Public key derivation: private_key × G = public_key
    auto generator = Point::generator();
    auto private_key = Scalar::from_hex(
        "E9873D79C6D87DC0FB6A5778633389F4453213303DA61F20BD67FC233AA33262"
    );
    auto public_key = generator * private_key;

    std::cout << "Public Key X: " << public_key.x().to_hex() << "\n";
    std::cout << "Public Key Y: " << public_key.y().to_hex() << "\n";
    return 0;
}
```

```bash
g++ -std=c++20 example.cpp -lufsecp -o example && ./example
```

### GPU Batch Multiplication

```cpp
#include <secp256k1_cuda/batch_operations.hpp>
#include <secp256k1/point.hpp>
#include <vector>

using namespace secp256k1::fast;

int main() {
    std::vector<Point> base_points(1'000'000, Point::generator());
    std::vector<Scalar> scalars(1'000'000);
    for (auto& s : scalars) s = Scalar::random();

    cuda::BatchConfig config{.device_id = 0, .threads_per_block = 256, .streams = 4};
    auto results = cuda::batch_multiply(base_points, scalars, config);

    std::cout << "Processed " << results.size() << " point multiplications\n";
    return 0;
}
```

---

## secp256k1 Security Model (FAST vs CT)

Two security profiles are **always active** — no flag-based selection:

### FAST Profile (Default)

- Maximum throughput, variable-time algorithms
- Use for: verification, batch processing, public key derivation, benchmarking
- ⚠️ **Not safe for secret key operations** — timing side-channels possible

### CT / Hardened Profile (`ct::` namespace)

- Constant-time arithmetic — no secret-dependent branches or memory access
- ~5–7× performance penalty vs FAST
- Use for: signing, private key handling, nonce generation, ECDH

**Choose the appropriate profile for your use case.** Using FAST with secret data is a security vulnerability.
See [THREAT_MODEL.md](THREAT_MODEL.md) for full details.

---

## secp256k1 Supported Coins (27 Blockchains)

| # | Coin | Ticker | Address Types | BIP-44 |
|---|------|--------|---------------|--------|
| 1 | **Bitcoin** | BTC | P2PKH, P2WPKH (Bech32), P2TR (Bech32m) | m/86'/0' |
| 2 | **Ethereum** | ETH | EIP-55 Checksum | m/44'/60' |
| 3 | **Litecoin** | LTC | P2PKH, P2WPKH | m/84'/2' |
| 4 | **Dogecoin** | DOGE | P2PKH | m/44'/3' |
| 5 | **Bitcoin Cash** | BCH | P2PKH | m/44'/145' |
| 6 | **Bitcoin SV** | BSV | P2PKH | m/44'/236' |
| 7 | **Zcash** | ZEC | P2PKH (transparent) | m/44'/133' |
| 8 | **Dash** | DASH | P2PKH | m/44'/5' |
| 9 | **DigiByte** | DGB | P2PKH, P2WPKH | m/44'/20' |
| 10 | **Namecoin** | NMC | P2PKH | m/44'/7' |
| 11 | **Peercoin** | PPC | P2PKH | m/44'/6' |
| 12 | **Vertcoin** | VTC | P2PKH, P2WPKH | m/44'/28' |
| 13 | **Viacoin** | VIA | P2PKH | m/44'/14' |
| 14 | **Groestlcoin** | GRS | P2PKH, P2WPKH | m/44'/17' |
| 15 | **Syscoin** | SYS | P2PKH | m/44'/57' |
| 16 | **BNB Smart Chain** | BNB | EIP-55 | m/44'/60' |
| 17 | **Polygon** | MATIC | EIP-55 | m/44'/60' |
| 18 | **Avalanche** | AVAX | EIP-55 (C-Chain) | m/44'/60' |
| 19 | **Fantom** | FTM | EIP-55 | m/44'/60' |
| 20 | **Arbitrum** | ARB | EIP-55 | m/44'/60' |
| 21 | **Optimism** | OP | EIP-55 | m/44'/60' |
| 22 | **Ravencoin** | RVN | P2PKH | m/44'/175' |
| 23 | **Flux** | FLUX | P2PKH | m/44'/19167' |
| 24 | **Qtum** | QTUM | P2PKH | m/44'/2301' |
| 25 | **Horizen** | ZEN | P2PKH | m/44'/121' |
| 26 | **Bitcoin Gold** | BTG | P2PKH | m/44'/156' |
| 27 | **Komodo** | KMD | P2PKH | m/44'/141' |

All EVM chains (ETH, BNB, MATIC, AVAX, FTM, ARB, OP) share the same address format (EIP-55 checksummed hex).

---

## secp256k1 Architecture

```
UltrafastSecp256k1/
├── cpu/                 # CPU-optimized implementation
│   ├── include/         # Public headers (field.hpp, scalar.hpp, point.hpp, ecdsa.hpp, schnorr.hpp)
│   ├── src/             # Implementation (field_asm_x64.asm, field_asm_riscv64.S, ...)
│   ├── fuzz/            # libFuzzer harnesses
│   └── tests/           # Unit tests
├── cuda/                # CUDA GPU acceleration
├── opencl/              # OpenCL GPU acceleration
├── metal/               # Apple Metal GPU acceleration
├── wasm/                # WebAssembly (Emscripten)
├── android/             # Android NDK (ARM64)
├── include/ufsecp/      # Stable C ABI
├── examples/
│   ├── esp32_test/      # ESP32-S3 Xtensa LX7 port
│   └── stm32_test/      # STM32F103 ARM Cortex-M3 port
└── docs/                # Documentation
```

---

## secp256k1 Testing & Verification

### Built-in Selftest

Every executable runs a deterministic **Known Answer Test (KAT)** on startup, covering all arithmetic operations:

| Mode | Time | When | What |
|------|------|------|------|
| **smoke** | ~1-2s | App startup, embedded | Core KAT (10 scalar mul, field/scalar identities, boundary vectors) |
| **ci** | ~30-90s | Every push (CI) | Smoke + cross-checks, bilinearity, NAF/wNAF, batch sweeps, algebraic stress |
| **stress** | ~10-60min | Nightly / manual | CI + 1000 random scalar muls, 500 field triples, batch inverse up to 8192 |

```cpp
#include "secp256k1/selftest.hpp"
using namespace secp256k1::fast;

Selftest(true, SelftestMode::smoke);              // Fast startup check
Selftest(true, SelftestMode::ci);                  // Full CI suite
Selftest(true, SelftestMode::stress, 0xDEADBEEF); // Nightly with custom seed
```

### Sanitizer Builds

```bash
cmake --preset cpu-asan && cmake --build build/cpu-asan -j    # ASan + UBSan
cmake --preset cpu-tsan && cmake --build build/cpu-tsan -j    # TSan (data races)
ctest --test-dir build/cpu-asan --output-on-failure
```

### Fuzz Testing

libFuzzer harnesses cover core arithmetic (`cpu/fuzz/`):

| Target | What it tests |
|--------|---------------|
| `fuzz_field` | add/sub round-trip, mul identity, square, inverse |
| `fuzz_scalar` | add/sub, mul identity, distributive law |
| `fuzz_point` | on-curve check, negate, compress round-trip, dbl vs add |

### Platform CI Coverage

| Platform | Backend | Compiler | Status |
|----------|---------|----------|--------|
| Linux x64 | CPU | GCC 13 / Clang 17 | ✅ CI |
| Linux x64 | CPU | Clang 17 (ASan+UBSan) | ✅ CI |
| Linux x64 | CPU | Clang 17 (TSan) | ✅ CI |
| Windows x64 | CPU | MSVC 2022 | ✅ CI |
| macOS ARM64 | CPU + Metal | AppleClang | ✅ CI |
| iOS ARM64 | CPU | Xcode | ✅ CI |
| Android ARM64 | CPU | NDK r27c | ✅ CI |
| WebAssembly | CPU | Emscripten | ✅ CI |
| ROCm/HIP | CPU + GPU | ROCm 6.3 | ✅ CI |

---

## secp256k1 Benchmark Targets

| Target | Description |
|--------|-------------|
| `bench_comprehensive` | Full field/point/batch/signature suite |
| `bench_scalar_mul` | k×G and k×P with wNAF analysis |
| `bench_ct` | Fast-vs-CT overhead comparison |
| `bench_atomic_operations` | Individual ECC building block latencies |
| `bench_field_52` | 4×64 vs 5×52 field representation |
| `bench_ecdsa_multiscalar` | k₁×G + k₂×Q (Shamir vs separate) |
| `bench_jsf_vs_shamir` | JSF vs Windowed Shamir comparison |
| `bench_adaptive_glv` | GLV window size sweep (8–20) |
| `bench_comprehensive_riscv` | RISC-V optimized benchmark suite |

---

## Research Statement

This library explores the **performance ceiling of secp256k1** across CPU architectures (x64, ARM64, RISC-V, Cortex-M, Xtensa) and GPUs (CUDA, OpenCL, Metal, ROCm). Zero external dependencies. Pure C++20.

---

## API Stability

**C++ API**: Not yet stable. Breaking changes may occur before **v4.0**. Core layers (field, scalar, point, ECDSA, Schnorr) are mature. Experimental layers (MuSig2, FROST, Adaptor, Pedersen, Taproot, HD, Coins) may change.

**C ABI (`ufsecp`)**: Stable from v3.4.0. ABI version tracked separately. See [SUPPORTED_GUARANTEES.md](include/ufsecp/SUPPORTED_GUARANTEES.md).

---

## Release Verification

All release artifacts ship with integrity checksums:

```bash
# Download release + checksums
curl -LO https://github.com/shrec/UltrafastSecp256k1/releases/latest/download/SHA256SUMS.txt

# Verify
sha256sum -c SHA256SUMS.txt
```

| Supply Chain | Status |
|-------------|--------|
| SHA256SUMS for all artifacts | ✅ Every release |
| SLSA Build Provenance (GitHub Attestation) | ✅ Every release |
| Reproducible builds documentation | 🔜 Planned |
| Cosign / Sigstore signing | 🔜 Planned |

---

## FAQ

**Is UltrafastSecp256k1 a drop-in replacement for libsecp256k1?**
> No. It is an independent implementation with a different API. The C ABI (`ufsecp`) provides a stable FFI surface, but function signatures differ from libsecp256k1. Migration requires code changes.

**Is the API stable?**
> The C ABI (`ufsecp`) is stable from v3.4.0. The C++ API (namespaces `fast::`, `ct::`) is mature for Tier 1 features but may change before v4.0.

**What is the constant-time scope?**
> All functions in `ct::` namespace are constant-time: field arithmetic, scalar arithmetic, point multiplication, complete addition, signing, and ECDH. The C ABI uses CT internally for all secret-key operations. See [CT Evidence](#ct-evidence--methodology) above.

**Which parts are production-safe today?**
> This library has **not been independently audited**. Tier 1 features (core ECC, ECDSA, Schnorr, ECDH) are extensively tested (12,000+ test cases, ASan/UBSan/TSan CI, fuzz testing). For production systems handling real funds, prefer audited libraries until an independent audit is completed.

**How do I reproduce the benchmarks?**
> See [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) for exact commands, pinned compiler/driver versions, and raw logs. The [live dashboard](https://shrec.github.io/UltrafastSecp256k1/dev/bench/) tracks performance across commits.

---

## Documentation

| Document | Description |
|----------|-------------|
| [API Reference](docs/API_REFERENCE.md) | Full C++ and C ABI reference |
| [Build Guide](docs/BUILDING.md) | Detailed build instructions for all platforms |
| [Benchmarks](docs/BENCHMARKS.md) | Complete benchmark results and methodology |
| [Threat Model](THREAT_MODEL.md) | Layer-by-layer security risk assessment |
| [Security Policy](SECURITY.md) | Vulnerability reporting and audit status |
| [Porting Guide](PORTING.md) | Add new platforms, architectures, GPU backends |
| [RISC-V Optimizations](RISCV_OPTIMIZATIONS.md) | RISC-V assembly details |
| [ESP32 Setup](docs/ESP32_SETUP.md) | ESP32 embedded development guide |
| [Contributing](CONTRIBUTING.md) | Development guidelines |
| [Changelog](CHANGELOG.md) | Version history |

---

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md).

```bash
git clone https://github.com/shrec/UltrafastSecp256k1.git
cd UltrafastSecp256k1
cmake -S . -B build/dev -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build build/dev -j
ctest --test-dir build/dev --output-on-failure
```

---

## License

**GNU Affero General Public License v3.0 (AGPL-3.0)**

- ✅ Use, modify, and distribute under AGPL-3.0
- ✅ Must disclose source code
- ✅ Must provide network access to source if run as a service

**Commercial License**: For proprietary use without AGPL obligations, contact [payysoon@gmail.com](mailto:payysoon@gmail.com).

See [LICENSE](LICENSE) for full details.

---

## Contact & Community

| Channel | Link |
|---------|------|
| Issues | [GitHub Issues](https://github.com/shrec/UltrafastSecp256k1/issues) |
| Discussions | [GitHub Discussions](https://github.com/shrec/UltrafastSecp256k1/discussions) |
| Wiki | [Documentation Wiki](https://github.com/shrec/UltrafastSecp256k1/wiki) |
| Benchmarks | [Live Dashboard](https://shrec.github.io/UltrafastSecp256k1/dev/bench/) |
| Security | [Report Vulnerability](https://github.com/shrec/UltrafastSecp256k1/security/advisories/new) |
| Commercial | [payysoon@gmail.com](mailto:payysoon@gmail.com) |

---

## Acknowledgements

UltrafastSecp256k1 is an independent implementation — written from scratch with our own architecture, GPU pipeline, embedded ports, and optimization techniques. At the same time, no project exists in a vacuum. The published research, specifications, and open discussions from the wider cryptographic community helped us refine our own ideas and validate our results.

We want to acknowledge the teams whose public work informed parts of our journey:

- **[bitcoin-core/secp256k1](https://github.com/bitcoin-core/secp256k1)** — The reference C library whose published research on constant-time field arithmetic and endomorphism-based scalar multiplication (GLV, Strauss, Pippenger) helped us benchmark and verify our own independent implementations on GPU and embedded targets.
- **[Bitcoin Core](https://github.com/bitcoin/bitcoin)** contributors — For open specifications (BIP-340 Schnorr, BIP-341 Taproot, RFC 6979) and a correctness-first engineering culture that benefits everyone building in this space.
- **Pieter Wuille, Jonas Nick, Tim Ruffing** and the libsecp256k1 maintainers — For publicly sharing their research on side-channel resistance, exhaustive testing, and field representation trade-offs. Their published findings helped us make better decisions when designing our own architecture.

We share our optimizations, GPU kernels, embedded ports, and cross-platform techniques freely — because open-source cryptography grows stronger when knowledge flows in every direction.

Special thanks to the [Stacker News](https://stacker.news) and [Delving Bitcoin](https://delvingbitcoin.org) communities for their early support and technical feedback.

Extra gratitude to [@0xbitcoiner](https://stacker.news/0xbitcoiner) for the initial outreach and for helping bridge the project with the wider Bitcoin developer ecosystem.

---

## ⚡ Support the Project

If you find **UltrafastSecp256k1** useful, consider supporting its development!

[![Donate with Bitcoin Lightning](https://img.shields.io/badge/Donate%20with-Lightning%20%E2%9A%A1-yellow?style=for-the-badge&logo=bitcoin)](https://stacker.news/shrec)

**Lightning Address:** `shrec@stacker.news` — send sats via any Lightning wallet or [stacker.news/shrec](https://stacker.news/shrec)

[![Sponsor](https://img.shields.io/badge/Sponsor-GitHub%20Sponsors-ea4aaa.svg?logo=github)](https://github.com/sponsors/shrec)
[![PayPal](https://img.shields.io/badge/PayPal-Donate-blue.svg?logo=paypal)](https://paypal.me/IChkheidze)

---

**UltrafastSecp256k1** — The fastest open-source secp256k1 library. GPU-accelerated ECDSA & Schnorr signatures for Bitcoin, Ethereum, and 25+ blockchains. Zero dependencies. Constant-time layer. 12+ platforms.

<!-- SEO keywords (not rendered by GitHub) -->
<!-- secp256k1 library fastest GPU CUDA OpenCL Metal ROCm ECDSA sign verify Schnorr BIP-340 Bitcoin Ethereum signature acceleration elliptic curve cryptography C++ C++20 high performance zero dependency batch verification constant time side channel resistance embedded ESP32 STM32 ARM Cortex-M RISC-V ARM64 WebAssembly WASM cross-platform multi-coin address generation BIP-32 BIP-44 HD wallet derivation key recovery EIP-155 RFC-6979 transaction signing blockchain cryptocurrency libsecp256k1 alternative NVIDIA AMD Apple Silicon MuSig2 FROST threshold signatures Taproot BIP-341 BIP-342 Pedersen commitments adaptor signatures ECDH key exchange secp256k1 GPU acceleration secp256k1 on embedded secp256k1 benchmarks secp256k1 constant time secp256k1 WASM secp256k1 C ABI FFI bindings Python Go Rust Java Node.js fastest secp256k1 implementation constant-time ECC library for RISC-V bitcoin cryptography optimization high-throughput elliptic curve signing secp256k1 RISC-V constant-time branchless cryptography GLV endomorphism Hamburg signed-digit comb Renes-Costello-Bathalter complete addition formulas dudect side-channel testing ASan UBSan TSan fuzzing libFuzzer valgrind memcheck security audit vulnerability scanning SLSA provenance supply chain security OpenSSF Scorecard CodeQL SonarCloud clang-tidy static analysis Docker container reproducible build Debian APT RPM Arch AUR Linux packaging AGPL-3.0 open source cryptographic library secp256k1 formal verification Fiat-Crypto Montgomery multiplication Barrett reduction BIP-327 multi-party computation MPC digital signatures public key cryptography PKI key agreement protocol -->
