# UltrafastSecp256k1

[![CI](https://github.com/shrec/UltrafastSecp256k1/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/shrec/UltrafastSecp256k1/actions/workflows/ci.yml)
[![ct-verif](https://github.com/shrec/UltrafastSecp256k1/actions/workflows/ct-verif.yml/badge.svg?branch=main)](https://github.com/shrec/UltrafastSecp256k1/actions/workflows/ct-verif.yml)
[![CodeQL](https://github.com/shrec/UltrafastSecp256k1/actions/workflows/codeql.yml/badge.svg?branch=main)](https://github.com/shrec/UltrafastSecp256k1/actions/workflows/codeql.yml)
[![OSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/shrec/UltrafastSecp256k1/badge)](https://securityscorecards.dev/viewer/?uri=github.com/shrec/UltrafastSecp256k1)
[![GPU Self-Hosted](https://github.com/shrec/UltrafastSecp256k1/actions/workflows/gpu-selfhosted.yml/badge.svg)](https://github.com/shrec/UltrafastSecp256k1/actions/workflows/gpu-selfhosted.yml)

High-performance, multi-backend secp256k1 engine with continuous audit and adversarial verification.

---

## What Is This?

UltrafastSecp256k1 is:

- ⚡ **High-throughput ECC engine** — CPU + GPU + embedded (CUDA, OpenCL, Metal, ARM64, RISC-V, WASM, ESP32, STM32)
- 🔐 **Cryptographic stack** — ECDSA, Schnorr, FROST, MuSig2, Taproot, BIP-352 Silent Payments, ZK Proofs, ECDH
- 🧠 **Continuous audit system** — 1,000,000+ assertions per build, 58 modules, 0 failures — not a snapshot
- 🧪 **Adversarially tested** — 189 exploit PoC tests, 11 fuzzer harnesses, 39 formal Cryptol properties, 3 independent CT pipelines

> Security is not assumed — it is continuously verified on every commit.

**11.00 M BIP352 scans/s** · **4.88 M ECDSA signs/s** · **4.05 M ECDSA verifies/s** · **3.66 M Schnorr signs/s** · **5.38 M Schnorr verifies/s** · **1.34 M FROST partial verifies/s** · **97.2 M point compressions/s** — RTX 5060 Ti (CUDA 12)

Known production use: [Sparrow Wallet Frigate](https://github.com/sparrowwallet/frigate) — [Details →](docs/ADOPTION.md)

---

## Quick Start

**Option A — Linux (apt)**
```bash
sudo apt install libufsecp3
ufsecp_selftest          # Expected: "OK (version 3.x, backend CPU)"
```

**Option B — npm (any OS)**
```bash
npm i ufsecp
node -e "require('ufsecp').selftest()"
```

**Option C — Python (any OS)**
```bash
pip install ufsecp
python -c "import ufsecp; ufsecp.selftest()"
```

**Option D — Build from source**
```bash
git clone https://github.com/shrec/UltrafastSecp256k1.git && cd UltrafastSecp256k1
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
./build/selftest          # Expected: "ALL TESTS PASSED"
```

→ [Full installation guide](#installation) · [Build from source](#building-secp256k1-from-source-cmake)

---

## Where to Start

| If you want to… | Go here |
|---|---|
| 🔍 Run the audit | [docs/AUDIT_GUIDE.md](docs/AUDIT_GUIDE.md) |
| ⚔️ Try to break the system | [docs/ATTACK_GUIDE.md](docs/ATTACK_GUIDE.md) |
| 📊 Understand the guarantees | [docs/AUDIT_TRACEABILITY.md](docs/AUDIT_TRACEABILITY.md) |
| 🧠 Audit philosophy & design rationale | [docs/AUDIT_PHILOSOPHY.md](docs/AUDIT_PHILOSOPHY.md) |
| 📐 Audit methodology specification (CAAS) | [docs/AUDIT_STANDARD.md](docs/AUDIT_STANDARD.md) |
| 🚀 External auditor quick start | [docs/AUDITOR_QUICKSTART.md](docs/AUDITOR_QUICKSTART.md) |
| 📄 See latest audit results | [AUDIT_REPORT.md](AUDIT_REPORT.md) |
| 🛡️ Exploit PoC test catalog | [docs/EXPLOIT_TEST_CATALOG.md](docs/EXPLOIT_TEST_CATALOG.md) |
| 🔀 Backend assurance matrix | [docs/BACKEND_ASSURANCE_MATRIX.md](docs/BACKEND_ASSURANCE_MATRIX.md) |
| 🚦 CI gating policy | [docs/CI_GATING_POLICY.md](docs/CI_GATING_POLICY.md) |
| 🗺️ ABI layer routing matrix | [docs/LAYER_ROUTING_MATRIX.md](docs/LAYER_ROUTING_MATRIX.md) |
| 🏗️ Build guide | [docs/BUILDING.md](docs/BUILDING.md) |
| 🔗 C ABI / FFI reference | [docs/API_REFERENCE.md](docs/API_REFERENCE.md) |
| 📈 Community benchmarks | [docs/COMMUNITY_BENCHMARKS.md](docs/COMMUNITY_BENCHMARKS.md) |
| 🏛️ Architecture overview | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) |
| 🔐 Security claims & contracts | [docs/SECURITY_CLAIMS.md](docs/SECURITY_CLAIMS.md) |
| 🔑 Secret lifecycle (zeroization, CT) | [docs/SECRET_LIFECYCLE.md](docs/SECRET_LIFECYCLE.md) |
| 🧮 Cryptographic invariants | [docs/CRYPTO_INVARIANTS.md](docs/CRYPTO_INVARIANTS.md) |
| 🧵 Thread-safety guarantees | [docs/THREAD_SAFETY.md](docs/THREAD_SAFETY.md) |
| 🛡️ Safe defaults | [docs/SAFE_DEFAULTS.md](docs/SAFE_DEFAULTS.md) |
| 🔬 Differential testing | [docs/DIFFERENTIAL_TESTING.md](docs/DIFFERENTIAL_TESTING.md) |
| 📋 Reproducible builds | [docs/REPRODUCIBLE_BUILDS.md](docs/REPRODUCIBLE_BUILDS.md) |
| 🚨 Incident response | [docs/INCIDENT_RESPONSE.md](docs/INCIDENT_RESPONSE.md) |
| 📦 Install packages | [Installation](#installation) |
| 🤔 Why this library? | [WHY_ULTRAFASTSECP256K1.md](WHY_ULTRAFASTSECP256K1.md) |
| 🌐 Sponsor | [github.com/sponsors/shrec](https://github.com/sponsors/shrec) |

> **Claim map:** [docs/ASSURANCE_LEDGER.md](docs/ASSURANCE_LEDGER.md) · **Security policy:** [SECURITY.md](SECURITY.md) · **Discord:** [discord.gg/E4BK8SeMYU](https://discord.gg/E4BK8SeMYU)

---

## Why This Exists

Traditional model: `code → audit PDF → trust`

This project: `code → test → execution → evidence → continuous verification`

We do not rely on trust. We provide reproducible evidence.

- Every exploit attempt becomes a permanent regression test
- Every commit runs 1,000,000+ assertions across 58 audit modules
- Every claim maps to a test in [docs/AUDIT_TRACEABILITY.md](docs/AUDIT_TRACEABILITY.md)
- Every performance number has pinned compiler/driver/toolkit versions and raw logs

> If a claim cannot be traced to a test, it is not valid.

For the full breakdown of the audit culture, CI/CD pipeline, formal verification layers, and supply-chain hardening, see [WHY_ULTRAFASTSECP256K1.md](WHY_ULTRAFASTSECP256K1.md).

---

<!-- Keywords (machine-readable, SEO):
secp256k1 · secp256k1 python · secp256k1 nodejs · secp256k1 rust · secp256k1 go · secp256k1 c# · secp256k1 java · secp256k1 swift · ecdsa python · schnorr python · bitcoin python library · ethereum python library · ECDSA batch verify · Schnorr BIP-340 · FROST threshold signatures · MuSig2 · Bitcoin cryptography · CUDA secp256k1 · OpenCL ECC · Metal GPU crypto · BIP-352 Silent Payments · constant-time cryptography · embedded ECC · WebAssembly crypto · elliptic curve cryptography · bitcoin wallet library · taproot signatures · BIP-32 HD keys · BIP-39 mnemonic · adaptor signatures · Pedersen commitment · Bulletproof range proof · react native crypto · python ecdsa signing · node.js secp256k1 · GPU elliptic curve · wasm crypto · mobile crypto library · threshold signature scheme · multi-party computation · pip install secp256k1 · npm secp256k1 · cargo secp256k1 · nuget secp256k1 · ufsecp
-->

## The Audit Model

Most libraries ship fast code and trust it's correct.
This library ships fast code — then systematically tries to break it, on every commit, permanently.

- New CVE published → PoC written → CI gate added → runs forever
- New ePrint attack → evaluated within 1 day → permanent regression test
- Contributor finds exploit → pull request → built into the system

[How it works](#engineering-quality--self-audit-culture) · [The standard](docs/AUDIT_STANDARD.md)

---

## Recent Performance Milestones (March 2026)

All measurements: RTX 5060 Ti (SM 12.0, CUDA 12), batch=16 384, kernel-only throughput.

| Operation | Previous | **Now** | Δ |
|-----------|----------|---------|---|
| ECDSA Verify (GPU) | 410.1 ns / 2.44 M/s | **246.7 ns / 4.05 M/s** | **+66 % throughput** |
| Schnorr Verify (GPU) | 354.6 ns / 2.82 M/s | **185.9 ns / 5.38 M/s** | **+91 % throughput** |
| FROST Partial Verify (GPU) | — | **748.9 ns / 1.34 M/s** | ⭐ New — first open-source GPU FROST |
| Batch Jacobian → Compressed | — | **10.3 ns / 97.2 M/s** | ⭐ New kernel |
| BIP-352 Silent Payments (GPU LUT) | 179.2 ns / 5.58 M/s | **91.0 ns / 11.00 M/s** | **+97 % throughput** |

> The ECDSA and Schnorr verify speedups come from the Shamir+GLV double-scalar multiplication, INT32 field arithmetic, and warp-level reduction pipeline. FROST partial verify is now callable via the stable C ABI as [`ufsecp_gpu_frost_verify_partial_batch()`](#gpu-c-abi-ufsecp_gpu).

## Why UltrafastSecp256k1? — Detail

> TL;DR is above. This section covers what differentiates this library in depth.

- **Continuous adversarial audit system** -- every exploit attempt becomes a permanent regression test; 1,000,000+ assertions per build, 189 exploit-PoC test files (all 189 registered as runner modules, parity enforced by `scripts/check_exploit_wiring.py`) across 200+ attack vectors, 37 CI workflows, 3 formal CT verification pipelines (ct-verif + Valgrind CT + dudect), 1.3M+ nightly differential checks — security hardens on every commit, not just on release day ([→ how it works](#engineering-quality--self-audit-culture))
- **Differentiated GPU secp256k1 surface** -- CUDA, OpenCL, and Metal all implement the stable 19-op GPU C ABI, while CUDA also carries the highest-throughput signing and verification kernels plus **GPU FROST partial verification** ([reproducible benchmark suite and raw logs](docs/BENCHMARKS.md))
- **High-performance CPU secp256k1 engine** -- optimized generator multiply, scalar multiply, hashing, and serialization pipelines across x86-64, ARM64, RISC-V, and embedded targets ([see bench_unified ratio table](docs/BENCHMARKS.md))
- **BIP-352 Silent Payments at 11.00 M/s** -- the full 7-stage GPU pipeline (k×P → hash → k×G → add → match) runs at 91.0 ns/op on CUDA, **267× faster** than single-threaded CPU ([GPU bench](docs/BENCHMARKS.md), [standalone CPU benchmark by @craigraw](https://github.com/craigraw/bench_bip352))
- **Built for modern secp256k1 workloads** -- signing, verification, wallet derivation, threshold protocols, adaptor signatures, ZK primitives, address generation, and large-scale public-key pipelines in one engine
- **Known production adoption** -- publicly disclosed production use includes [SparrowWallet Frigate](https://github.com/sparrowwallet/frigate), with permission to publish the adoption note from Craig Raw
- **Field-tested GPU pipeline** -- the CUDA engine has been stress-tested in live high-throughput workflows over long-running sessions and very large point volumes, not only in short synthetic benchmarks
- **Zero dependencies** -- pure C++20, no Boost, no OpenSSL, compiles anywhere with a conforming compiler
- **Dual-layer security** -- variable-time FAST path for throughput, constant-time CT path for secret-key operations
- **12+ platforms** -- x86-64, ARM64, RISC-V, WASM, iOS, Android, ESP32, STM32, CUDA, Metal, OpenCL, plus an early-development ROCm/HIP compatibility path slated for hardware-backed validation

> **Benchmark reproducibility:** All numbers come from pinned compiler/driver/toolkit versions with exact commands and raw logs. See [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) (methodology) and the [live dashboard](https://shrec.github.io/UltrafastSecp256k1/dev/bench/).

> **Why this library, in depth?** See [WHY_ULTRAFASTSECP256K1.md](WHY_ULTRAFASTSECP256K1.md) for a full breakdown of the audit culture, 24-workflow CI/CD pipeline, graph-assisted review model, formal verification layers, and supply-chain hardening that back these claims.

> **External auditor prep:** Run `bash scripts/external_audit_prep.sh` to produce a reproducible auditor-facing bundle with preflight outputs, assurance export, traceability artifacts, and an optional full audit package.

> **Claim map:** Top-level trust claims are keyed in [docs/ASSURANCE_LEDGER.md](docs/ASSURANCE_LEDGER.md): CPU CT routing `A-001`, stable GPU ABI `A-002`, cross-backend GPU parity `A-003`, benchmark reproducibility `A-004`, exploit-audit surface `A-005`, graph-assisted review `A-006`, open self-audit transparency `A-007`, and ROCm/HIP status discipline `A-008`.

**Quick links:** [Discord](https://discord.gg/sUmW7cc5) * [Benchmarks](docs/BENCHMARKS.md) * [Community Benchmarks](docs/COMMUNITY_BENCHMARKS.md) * [Adopters](ADOPTERS.md) * [Build Guide](docs/BUILDING.md) * [API Reference](docs/API_REFERENCE.md) * [Binding Usage Standard](docs/BINDINGS_USAGE_STANDARD.md) * [Security Policy](SECURITY.md) * [Threat Model](THREAT_MODEL.md) * [Assurance Ledger](docs/ASSURANCE_LEDGER.md) * [AI Audit Protocol](docs/AI_AUDIT_PROTOCOL.md) * [Audit Standard (CAAS)](docs/AUDIT_STANDARD.md) * [**Why This Library?**](WHY_ULTRAFASTSECP256K1.md) * [Porting Guide](PORTING.md) * [**Sponsor**](https://github.com/sponsors/shrec)

### Real-world Adoption

UltrafastSecp256k1 is used by [Sparrow Wallet's Frigate](https://github.com/sparrowwallet/frigate).

Frigate 1.4.0 switched its DuckDB extension to `ufsecp.duckdb_extension` using UltrafastSecp256k1, and its README documents a custom DuckDB extension wrapping UltrafastSecp256k1 for `ufsecp_scan(...)`-based Silent Payments scanning with CUDA, OpenCL and Metal backend support.

See: [Frigate 1.4.0 release](https://github.com/sparrowwallet/frigate/releases/tag/1.4.0) · [Frigate README](https://github.com/sparrowwallet/frigate/blob/master/README.md) · [Details →](docs/ADOPTION.md)

Package traction: [`ufsecp`](https://www.npmjs.com/package/ufsecp) 1,192 npm downloads/30d · [`react-native-ufsecp`](https://www.npmjs.com/package/react-native-ufsecp) 1,295/30d · [`Ufsecp`](https://www.nuget.org/packages/Ufsecp) 1,491 NuGet total (as of 2026-03-29).

Full adopter list: [ADOPTERS.md](ADOPTERS.md)

---

[![GitHub stars](https://img.shields.io/github/stars/shrec/UltrafastSecp256k1?style=flat-square&logo=github&label=Stars)](https://github.com/shrec/UltrafastSecp256k1/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/shrec/UltrafastSecp256k1?style=flat-square&logo=github&label=Forks)](https://github.com/shrec/UltrafastSecp256k1/network/members)
[![CI](https://img.shields.io/github/actions/workflow/status/shrec/UltrafastSecp256k1/ci.yml?branch=main&label=CI)](https://github.com/shrec/UltrafastSecp256k1/actions/workflows/ci.yml)
[![Benchmark](https://img.shields.io/github/actions/workflow/status/shrec/UltrafastSecp256k1/benchmark.yml?branch=main&label=Bench)](https://shrec.github.io/UltrafastSecp256k1/dev/bench/)
[![Release](https://img.shields.io/github/v/release/shrec/UltrafastSecp256k1?label=Release)](https://github.com/shrec/UltrafastSecp256k1/releases/latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](https://en.cppreference.com/w/cpp/20)
[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/shrec/UltrafastSecp256k1/badge)](https://scorecard.dev/viewer/?uri=github.com/shrec/UltrafastSecp256k1)
[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/12011/badge)](https://www.bestpractices.dev/projects/12011)
[![CodeQL](https://github.com/shrec/UltrafastSecp256k1/actions/workflows/codeql.yml/badge.svg)](https://github.com/shrec/UltrafastSecp256k1/actions/workflows/codeql.yml)
[![Security Audit](https://github.com/shrec/UltrafastSecp256k1/actions/workflows/security-audit.yml/badge.svg)](https://github.com/shrec/UltrafastSecp256k1/actions/workflows/security-audit.yml)
[![CT ARM64](https://github.com/shrec/UltrafastSecp256k1/actions/workflows/ct-arm64.yml/badge.svg)](https://github.com/shrec/UltrafastSecp256k1/actions/workflows/ct-arm64.yml)
[![CT-Verif](https://github.com/shrec/UltrafastSecp256k1/actions/workflows/ct-verif.yml/badge.svg)](https://github.com/shrec/UltrafastSecp256k1/actions/workflows/ct-verif.yml)
[![Valgrind CT](https://github.com/shrec/UltrafastSecp256k1/actions/workflows/valgrind-ct.yml/badge.svg)](https://github.com/shrec/UltrafastSecp256k1/actions/workflows/valgrind-ct.yml)
[![Perf Gate](https://github.com/shrec/UltrafastSecp256k1/actions/workflows/bench-regression.yml/badge.svg)](https://github.com/shrec/UltrafastSecp256k1/actions/workflows/bench-regression.yml)
[![Clang-Tidy](https://github.com/shrec/UltrafastSecp256k1/actions/workflows/clang-tidy.yml/badge.svg)](https://github.com/shrec/UltrafastSecp256k1/actions/workflows/clang-tidy.yml)
[![SonarCloud](https://sonarcloud.io/api/project_badges/measure?project=shrec_UltrafastSecp256k1&metric=security_rating)](https://sonarcloud.io/summary/overall?id=shrec_UltrafastSecp256k1)
[![codecov](https://codecov.io/gh/shrec/UltrafastSecp256k1/graph/badge.svg)](https://codecov.io/gh/shrec/UltrafastSecp256k1)
[![Discord](https://img.shields.io/badge/Discord-Join%20Us-5865F2?logo=discord&logoColor=white)](https://discord.gg/E4BK8SeMYU)

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

## Highlights

- **BIP-352 GPU pipeline at 11.00 M/s** -- full silent payment scanning pipeline on CUDA (91.0 ns/op), 267× faster than CPU
- **GPU-accelerated secp256k1** -- high-throughput CUDA verification kernels, batch ECDH, BIP-352 scanning, and BIP-324 encryption on CUDA/OpenCL/Metal; CT-sensitive signing always routes through the CPU CT layer; GPU operations that handle secret material (ECDH, BIP-352, BIP-324) require a trusted single-tenant environment (see [GPU Security Model](docs/BACKEND_ASSURANCE_MATRIX.md))
- **GPU C ABI (`ufsecp_gpu`)** -- stable 19-op FFI for GPU batch ops across CUDA, OpenCL, and Metal, with full backend parity on the public surface
- **Zero-Knowledge cryptographic layer** -- Pedersen commitments, DLEQ proofs, Bulletproof range proofs, Ethereum-compatible Keccak-256
- **17–67× faster batch operations** -- all-affine Pippenger with touched-bucket optimization
- **Multi-language bindings** -- Python (`pip install ufsecp`), Node.js (`npm i ufsecp`), Rust, Go, C#/.NET, Java, Swift, PHP, Ruby, Dart, React Native — all via the stable C ABI
- **Embedded device support** -- ESP32-S3, ESP32-P4, ESP32-C6, STM32 Cortex-M
- **Zero-dependency C++20 core** -- no Boost, no OpenSSL, compiles anywhere
- **Massively parallel workloads** -- batch verification, key scanning, address generation at GPU scale

---

## Engineering Quality & Self-Audit Culture

> Most high-performance cryptographic libraries ship fast code and trust that it is correct.
> UltrafastSecp256k1 ships fast code **and then systematically tries to break it**.
> The internal self-audit system was designed in parallel with the cryptographic implementation as a first-class engineering artifact — not bolted on afterwards.

The governing idea is Bitcoin-style: **don't trust, verify**. The project does not treat assurance as a PDF milestone that must be waited on before the next improvement. Instead, it treats auditability as an always-on property of the repository: reproducible builds, rerunnable tests, structured artifacts, graph-backed code navigation, and continuous adversarial review that anyone can repeat.

This top-level narrative maps directly to the assurance ledger: CT secret-key routing (`A-001`), exploit-style audit coverage (`A-005`), graph-assisted review (`A-006`), and self-audit transparency (`A-007`).

### By the Numbers

| Metric | Value |
|--------|-------|
| Internal audit assertions per build | **~1,000,000+** |
| Audit modules (`unified_audit_runner`) | **55 modules, 8 sections, 0 failures** |
| Exploit PoC test files | **86 tests, 14 coverage areas, 0 failures** |
| CI/CD workflows | **37 GitHub Actions workflows** |
| Build matrix (arch × config × OS) | **7 × 17 × 5 = 595 combinations** |
| Nightly differential tests | **~1,300,000+ random checks / night** |
| Constant-time verification pipelines | **3 independent (ct-arm64, ct-verif, Valgrind CT)** |
| Fuzzing adversarial corpus | **530,000+ cases (libFuzzer + ClusterFuzz-Lite)** |
| Static analysis tools | **4 (CodeQL, Clang-Tidy, CPPCheck, SonarCloud)** |
| Self-audit documents in repo | **13 dedicated audit/quality documents** |
| Self-tests passing (all backends) | **76/76** |

### CI/CD Pipeline Highlights

| Workflow | Purpose | Trigger |
|----------|---------|---------|
| `security-audit.yml` | Runs full `unified_audit_runner` — 55 modules, ~1M+ assertions | Every push |
| `ct-arm64.yml` | Constant-time verification on native ARM64 hardware | Every push |
| `ct-verif.yml` | Formal constant-time verification pass | Every push |
| `valgrind-ct.yml` | Valgrind memcheck + CT analysis | Every push |
| `bench-regression.yml` | Performance regression gate — CI fails if throughput drops | Every push |
| `nightly.yml` | 1.3M+ differential checks + extended fuzz + full sanitizer run | Nightly |
| `cflite.yml` | ClusterFuzz-Lite continuous fuzzing integration | Every push |
| `mutation.yml` | Mutation testing — verifies test suite kills every injected fault | Scheduled |
| `codeql.yml` | GitHub CodeQL static analysis (C++) | Every push |
| `sonarcloud.yml` | SonarCloud code quality and security rating | Every push |
| `scorecard.yml` | OpenSSF Scorecard + Best Practices supply-chain scan | Weekly |
| `ci.yml` | Core build + test across 17 configs × 7 architectures × 5 OSes | Every push / PR |

### What "Self-Audit Culture" Means in Practice

- Every field arithmetic property is verified algebraically: commutativity, associativity, distributivity, carry propagation, canonical form
- Every constant-time path is verified under **formal CT analysis + Valgrind + hardware-native ARM64 CT pipeline** — three independent layers
- Every ECDSA/Schnorr implementation is cross-validated against **Wycheproof vectors, independent reference golden vectors, and BIP test vectors**
- Every commit that would regress throughput **fails CI automatically** via `bench-regression.yml`
- Audit results are logged as **structured artifacts** (JSON reports, per-platform logs), not just pass/fail signals
- **Nightly differential testing** runs ~1.3M random round-trips against reference implementations every night
- All 55 audit modules return `AUDIT-READY` status. Zero failures across all tested platforms.

### Exploit PoC Test Suite (187 Tests, 20+ Coverage Areas)

In addition to the 55-module `unified_audit_runner`, UltrafastSecp256k1 ships **189 dedicated exploit-style PoC tests** that actively try to break the library across its highest-risk surfaces. Each `audit/test_exploit_*.cpp` target builds and runs standalone so failures stay easy to attribute and reproduce.

| Coverage Area | Representative attack focus |
|---------------|-----------------------------|
| ECDSA / Signature | malleability, RFC 6979 KATs, recovery edge cases |
| Schnorr / BIP-340 / Batch | batch soundness, forged signatures, invalid identification paths |
| GLV / ECC Math | endomorphism invariants, multiscalar correctness, Pippenger behavior |
| BIP-32 / BIP-39 / HD Keys | path overflow, hardened isolation, mnemonic and derivation edge cases |
| MuSig2 / FROST | nonce reuse, transcript fork equivocation, stale commitment replay, rogue-key aggregation, Byzantine participants, DKG and Lagrange edge cases |
| Adaptor Signatures / ZK | adaptor parity attacks, Pedersen invariants, malformed ZK proofs |
| Crypto Primitives / AEAD | ChaCha20-Poly1305 integrity, HKDF, SHA/Keccak/RIPEMD KATs |
| ECIES | authentication forgery, encryption correctness, roundtrip safety |
| Bitcoin / Protocol BIPs | BIP-143, BIP-144, BIP-324, SegWit, Taproot protocol edge cases |
| Address / Wallet / Signing | address encoding, wallet API misuse, Ethereum and Bitcoin signing flows |
| Constant-Time / Security | CT divergence, key-recovery style probes, backend divergence detection |
| ElligatorSwift | encoding correctness and ECDH roundtrips |
| Self-Test / Recovery | self-test API behavior and recovery boundary cases |
| Batch Verify | aggregate verification math correctness |

> All 189 exploit tests live in `audit/test_exploit_*.cpp`. Build with `cmake -S . -B build-audit -G Ninja -DCMAKE_BUILD_TYPE=Release` and run them standalone or via `ctest`.

### Self-Audit Document Index

| Document | Contents |
|----------|---------|
| [WHY_ULTRAFASTSECP256K1.md](WHY_ULTRAFASTSECP256K1.md) | Full audit infrastructure, CI pipeline index, formal verification evidence |
| [docs/AUDIT_PHILOSOPHY.md](docs/AUDIT_PHILOSOPHY.md) | Audit philosophy, continuous evidence model, design rationale, common objections answered |
| [AUDIT_REPORT.md](AUDIT_REPORT.md) | Historical baseline audit (641,194 core checks). Current: 247 modules, 0 failures |
| [AUDIT_COVERAGE.md](AUDIT_COVERAGE.md) | Per-module coverage matrix |
| [THREAT_MODEL.md](THREAT_MODEL.md) | Layer-by-layer risk analysis |
| [SECURITY.md](SECURITY.md) | Vulnerability disclosure policy |
| [docs/AUDIT_GUIDE.md](docs/AUDIT_GUIDE.md) | Navigation guide for external auditors |
| [docs/CI_ENFORCEMENT.md](docs/CI_ENFORCEMENT.md) | Full CI enforcement policy |
| [docs/BACKEND_ASSURANCE_MATRIX.md](docs/BACKEND_ASSURANCE_MATRIX.md) | Per-backend assurance matrix |
| [docs/AUDIT_TRACEABILITY.md](docs/AUDIT_TRACEABILITY.md) | Requirement-to-test traceability map |

> **Note:** UltrafastSecp256k1 has not undergone a third-party cryptographic audit.
> The assurance model is open self-audit: reproducible tests, traceability, CI enforcement, and public review artifacts that anyone can rerun.
> The project hardens continuously through internal audit on every build and every commit.

---

## Performance

**RTX 5060 Ti (CUDA 12, kernel throughput)**

| Metric | Value | Notes |
|--------|-------|-------|
| ECC operations (field/point) | ~2.3 B ops/sec | kernel-only |
| ECDSA sign | **4.88 M sigs/sec** | RFC 6979, low-S |
| ECDSA verify | **4.05 M verifies/sec** | Shamir+GLV (+66% vs prev) |
| Schnorr sign (BIP-340) | **3.66 M sigs/sec** | BIP-340 tagged hash |
| Schnorr verify (BIP-340) | **5.38 M verifies/sec** | BIP-340+GLV (+91% vs prev) |
| FROST partial verify | **1.34 M verifies/sec** | ⭐ New — first open-source GPU FROST |
| Batch point compress (J→SEC1) | **97.2 M pts/sec** | New kernel |

## Architecture

```
+-------------------------------------------------------+
|              Language Bindings (FFI)                   |
|  Python | Node | Rust | Go | C# | Java | Swift | PHP |
+-------------------------------------------------------+
                         |
                  Bindings Layer
                 (ctypes / koffi / cgo
                  JNA / P/Invoke / FFI)
                         |
+-------------------------------------------------------+
|          UltrafastSecp256k1 Core (C++20)               |
|                                                       |
|  ECDSA | Schnorr | ECDH | MuSig2 | FROST | Pedersen  |
|  Taproot | BIP-32 HD | Adaptor Sigs | ZK Proofs       |
|  [FAST layer]              [CT layer]                 |
+-------------------------------------------------------+
                         |
+--------+---------+---------+---------+----------------+
|  CPU   |  CUDA   | OpenCL  |  Metal  |   Embedded     |
| x86_64 | NVIDIA  | AMD/NV  |  Apple  | ESP32 / STM32  |
| ARM64  | sm_50+  | any GPU | Silicon | RISC-V / WASM  |
| RISC-V |         |         |         | Cortex-M       |
+--------+---------+---------+---------+----------------+
```

## Examples

| Category | Description | Link |
|----------|-------------|------|
| **CPU** | Core ECC, ECDSA, Schnorr, BIP-32, Taproot, Pedersen | [examples/](examples/) |
| **CUDA** | GPU benchmark signing kernels, batch verify, FROST, device management (production secret-key signing uses CPU CT layer) | [examples/](examples/) |
| **OpenCL** | Cross-vendor GPU compute | [examples/](examples/) |
| **Metal** | Apple Silicon GPU acceleration | [examples/](examples/) |
| **Multi-language** | C, Python, Rust, Node.js, Go, Java binding examples | [examples/README.md](examples/README.md) |
| **Embedded** | ESP32-S3, STM32 platform ports | [examples/esp32_test/](examples/esp32_test/) |

## Use Cases

- **Blockchain infrastructure** -- high-throughput transaction validation and signing pipelines (secret-key signing runs on the CPU CT layer; batch verification scales on GPU)
- **Signature verification at scale** -- batch verify millions of signatures per second on GPU
- **Cryptographic research** -- independent secp256k1 implementation with full source access
- **Zero-knowledge pipelines** -- Pedersen commitments, Bulletproofs, DLEQ proofs
- **Embedded cryptographic systems** -- hardware wallets, IoT devices, microcontrollers
- **Key scanning & address generation** -- BIP-352 Silent Payments, vanity address mining

> Star the repository if you find it useful!

---

## Security & Vulnerability Reporting

**Report vulnerabilities** via [GitHub Security Advisories](https://github.com/shrec/UltrafastSecp256k1/security/advisories/new) or email [payysoon@gmail.com](mailto:payysoon@gmail.com).
For production cryptographic systems, perform your own risk review, review the current guarantees in [SUPPORTED_GUARANTEES.md](include/ufsecp/SUPPORTED_GUARANTEES.md), and apply the assurance level appropriate to your deployment.

For the full audit infrastructure breakdown (1M+ assertions, 37 CI workflows, formal CT verification pipelines, self-audit document index), see the [Engineering Quality & Self-Audit Culture](#engineering-quality--self-audit-culture) section above and [WHY_ULTRAFASTSECP256K1.md](WHY_ULTRAFASTSECP256K1.md).

---

## Seeking Sponsors -- Bug Bounty & Development

> **We are actively seeking sponsors and funding partners** to expand continuous verification, bug bounty coverage, and long-term maintenance.

[![Sponsor](https://img.shields.io/badge/Sponsor_This_Project-GitHub_Sponsors-ea4aaa.svg?style=for-the-badge&logo=github)](https://github.com/sponsors/shrec)
[![Donate with Bitcoin Lightning](https://img.shields.io/badge/Lightning_Sats-shrec@stacker.news-F7931A?style=for-the-badge&logo=bitcoin)](https://stacker.news/shrec)

UltrafastSecp256k1 is a **high-performance, zero-dependency secp256k1 library** with GPU acceleration, constant-time side-channel protection, and 12+ platform targets. The funding priorities are:

### 1. Bug Bounty Program

We want to establish a **funded bug bounty program** to incentivize security researchers:

- Critical vulnerabilities (signature forgery, key recovery, CT bypass) -- high bounty tier
- Correctness bugs (arithmetic errors, edge cases) -- medium bounty tier
- Memory safety / undefined behavior -- standard bounty tier
- All GPU backends (CUDA, OpenCL, Metal, ROCm) covered

### 2. Open Audit Infrastructure

We want to make outside review easier without turning assurance into a bureaucratic checkbox:

- One-command replay packs for self-audit, differential tests, and platform validation
- Public artifact bundles for benchmark and audit reruns
- More native-device automation for ARM64 and RISC-V
- More adversarial and concurrency stress harnesses that outside reviewers can reuse

Currently we accept vulnerability reports via [GitHub Security Advisories](https://github.com/shrec/UltrafastSecp256k1/security/advisories/new) but **cannot offer financial rewards without sponsor funding**.

### 3. Ongoing Development

Sponsorship helps sustain development of:

- **Zero-knowledge proofs** -- Pedersen commitments, Bulletproofs, Schnorr sigma protocols, DLEQ proofs
- **GPU compute** -- CUDA, OpenCL, Metal, ROCm batch signature verification, key scanning, and address generation
- **Platform ports** -- embedded (ESP32, STM32), mobile (iOS, Android), WASM
- **Protocol features** -- MuSig2, FROST threshold signatures, Taproot, BIP-352 Silent Payments
- **Multi-coin support** -- 27+ blockchain address formats and signing
- **Formal verification** -- Fiat-Crypto integration, Cryptol models (ct-verif and valgrind-ct already active in CI)
- **CI/CD infrastructure** -- cross-platform testing, performance regression gates, fuzzing

### How to Sponsor

| Method | Link |
|--------|------|
| **GitHub Sponsors** (preferred) | [github.com/sponsors/shrec](https://github.com/sponsors/shrec) |
| **Bitcoin Lightning** | `shrec@stacker.news` (any Lightning wallet) |
| **PayPal** | [paypal.me/IChkheidze](https://paypal.me/IChkheidze) |
| **Corporate / Foundation** | [payysoon@gmail.com](mailto:payysoon@gmail.com) |
| **Discord** | [Join our server](https://discord.gg/E4BK8SeMYU) |

All sponsors will be acknowledged in the README, release notes, and project documentation.
For corporate partnerships, audit co-funding, or grant applications -- please reach out via email.

---

## secp256k1 Feature Overview

Features are organized into **maturity tiers** (see [SUPPORTED_GUARANTEES.md](include/ufsecp/SUPPORTED_GUARANTEES.md) for detailed guarantees):

| Tier | Category | Component | Status |
|------|----------|-----------|--------|
| **1 -- Core** | Field / Scalar / Point | GLV, Precompute, Batch Inverse | [OK] |
| **1 -- Core** | Assembly | x64 MASM/GAS, BMI2/ADX, ARM64, RISC-V RV64GC | [OK] |
| **1 -- Core** | SIMD | AVX2/AVX-512 batch ops, Montgomery batch inverse | [OK] |
| **1 -- Core** | Constant-Time | CT field/scalar/point -- no secret-dependent branches | [OK] |
| **1 -- Core** | ECDSA | Sign/Verify, RFC 6979, DER/Compact, low-S, Recovery | [OK] |
| **1 -- Core** | Schnorr | BIP-340 sign/verify, tagged hashing, x-only pubkeys | [OK] |
| **1 -- Core** | ECDH | Key exchange (raw, xonly, SHA-256) | [OK] |
| **1 -- Core** | Multi-scalar | Strauss/Shamir dual-scalar multiplication | [OK] |
| **1 -- Core** | Batch verify | ECDSA + Schnorr batch verification | [OK] |
| **1 -- Core** | Hashing | SHA-256 (SHA-NI), SHA-512, HMAC, Keccak-256 | [OK] |
| **1 -- Core** | C ABI | `ufsecp` stable FFI (45 exports) | [OK] |
| **2 -- Protocol** | BIP-32/44 | HD derivation, path parsing, xprv/xpub, coin-type | [OK] |
| **2 -- Protocol** | Taproot | BIP-341/342, tweak, Merkle tree | [OK] |
| **2 -- Protocol** | MuSig2 | BIP-327, key aggregation, 2-round signing | [OK] |
| **2 -- Protocol** | FROST | Threshold signatures, t-of-n | [OK] |
| **2 -- Protocol** | Adaptor | Schnorr + ECDSA adaptor signatures | [OK] |
| **2 -- Protocol** | Pedersen | Commitments, homomorphic, switch commitments | [OK] |
| **2 -- Protocol** | ZK Proofs | Schnorr sigma, DLEQ, Bulletproof range proofs (64-bit) | [OK] |
| **3 -- Convenience** | Address | P2PKH, P2WPKH, P2TR, Base58, Bech32/m, EIP-55 | [OK] |
| **3 -- Convenience** | Coins | 27 blockchains, auto-dispatch | [OK] |
| **2 -- Protocol** | BIP-352 | Silent Payments scanning pipeline (CPU + GPU) | [OK] |
| **2 -- Protocol** | ECIES | Elliptic curve integrated encryption | [OK] |
| -- | GPU | CUDA, Metal, OpenCL, ROCm kernels | [OK] |
| -- | GPU C ABI | `ufsecp_gpu` -- 7 batch ops across 3 backends (26 functions, incl. FROST) | [OK] |
| -- | Platforms | x64, ARM64, RISC-V, ESP32, STM32, WASM, iOS, Android | [OK] |

> **Tier 1** = battle-tested core crypto with stable API. **Tier 2** = protocol-level features, API may evolve. **Tier 3** = convenience utilities.

### BIP-340 Strict Encoding

All public API functions enforce **canonical input encoding** as required by BIP-340 and Bitcoin consensus:
- Signatures with `r >= p` or `s >= n` are **rejected, not reduced**
- Public keys with `x >= p` are **rejected, not reduced**
- Private keys must satisfy `1 <= sk < n`

The C ABI (`ufsecp_*`) returns distinct error codes: `UFSECP_ERR_BAD_SIG` (non-canonical signature) vs `UFSECP_ERR_VERIFY_FAIL` (valid encoding, bad math). See [docs/COMPATIBILITY.md](docs/COMPATIBILITY.md) for details.

---

## BIP-352 Silent Payments Scanning Benchmark

### GPU Pipeline (CUDA, RTX 5060 Ti)

The full 7-stage BIP-352 scanning pipeline runs entirely on-GPU with zero CPU round-trips:

1. **k×P** -- scalar multiply tweak point by scan private key
2. **Serialize** -- compress shared secret to 33-byte SEC1
3. **Tagged SHA-256** -- `BIP0352/SharedSecret` tagged hash
4. **k×G** -- generator multiply by hash scalar
5. **Point add** -- `spend_pubkey + output_point`
6. **Serialize + prefix** -- compress candidate, extract upper 64 bits
7. **Prefix match** -- compare against output prefix list

| Mode | ns/op | Throughput | Notes |
|------|-------|------------|-------|
| GPU pipeline (GLV, w=4) | 179.2 ns | 5.58 M/s | GLV wNAF decomposition |
| **GPU pipeline (LUT)** | **91.0 ns** | **11.00 M/s** | 64 MB precomputed 16×64K generator table |
| GPU pipeline (LUT + pretbl) | 102.1 ns | ~9.79 M/s | Precomputed per-tweak tables |

*500K tweak points per batch, 11 passes, median. Near-optimal occupancy for RTX 5060 Ti (SM 12.0, 36 SMs). ~950 billion candidates/day.*

### GPU vs CPU Comparison

| Platform | Full Pipeline | vs GPU (LUT) |
|----------|--------------|-------|
| **CUDA GPU (RTX 5060 Ti)** | **91.0 ns/op** | **baseline** |
| x86-64 CPU (i5-14400F, GCC 14) | 24,285 ns/op | 267× slower |
| ARM64 CPU (Cortex-A55, Clang 18) | 153,385 ns/op | 1,644× slower |
| RISC-V 64 (SiFive U74, GCC 13) | 257,996 ns/op | 2,765× slower |

### Community & Contributor Benchmarks

See **[docs/COMMUNITY_BENCHMARKS.md](docs/COMMUNITY_BENCHMARKS.md)** for all hardware results submitted by community members — including RTX 5070 Ti (Blackwell) and a standalone BIP-352 CPU comparison vs libsecp256k1.  Want to add yours? Instructions are in that file.

### Real-world scanning performance (Frigate / Sparrow Wallet)

Independent benchmarks from [Sparrow Wallet's Frigate](https://github.com/sparrowwallet/frigate) — a DuckDB-based Silent Payments scanning pipeline using UltrafastSecp256k1 via [`ufsecp_scan(...)`](https://github.com/sparrowwallet/duckdb-ufsecp-extension). Results produced by Frigate's `benchmark.py` scanning mainnet to block 914,000.

**GPU scanning (full BIP-352 pipeline, 2-year scan, 133M tweaks):**

| Hardware | Backend | Time | Throughput |
|----------|---------|------|------------|
| 2× NVIDIA RTX 5090 | CUDA | 3.2 s | ~41.5 M/s |
| NVIDIA RTX 5080 | CUDA | 7.7 s | ~17.3 M/s |
| Apple M1 Pro | Metal | 3m 47s | ~584 K/s |

**CPU scanning (full BIP-352 pipeline, 2-year scan, 133M tweaks):**

| Hardware | CPUs | Time | Throughput |
|----------|------|------|------------|
| Intel Core Ultra 9 285K | 24 | 3m 50s | ~577 K/s |
| Apple M1 Pro | 10 | 7m 47s | ~284 K/s |

Source: [Frigate README — Performance](https://github.com/sparrowwallet/frigate/blob/master/README.md#performance)

### CPU vs libsecp256k1 (standalone external benchmark)

Standalone single-threaded benchmark by [@craigraw](https://github.com/craigraw) ([bench_bip352](https://github.com/craigraw/bench_bip352)) — full results in [docs/COMMUNITY_BENCHMARKS.md](docs/COMMUNITY_BENCHMARKS.md). Thank you for the contribution!

**Full pipeline** (10K points, 11 passes, median, GCC 12.4, `-O3 -march=native`, `USE_ASM_X86_64=1`):

| Backend | Median | ns/op | Ratio |
|---------|--------|-------|-------|
| libsecp256k1 | 545.2 ms | 54,519 ns | 1.00x |
| **UltrafastSecp256k1** | **456.1 ms** | **45,615 ns** | **1.20x faster** |

**Per-operation breakdown** (1K points, 11 passes, median):

| Operation | libsecp256k1 | UltrafastSecp256k1 | Ratio |
|-----------|-------------|-------------------|-------|
| k\*P (scalar mul) | 37,975 ns | 26,460 ns | 1.44x faster |
| Serialize compressed (1st) | 36 ns | 15 ns | 2.4x faster |
| Tagged SHA-256 | 744 ns | 65 ns | 11.4x faster |
| k\*G (generator mul) | 17,460 ns | 8,559 ns | 2.04x faster |
| Point addition | 2,250 ns | 2,457 ns | 0.92x |
| Serialize compressed (2nd) | 23 ns | 21 ns | 1.1x faster |

> **Note:** Point addition is slightly slower because both inputs have Z=1 (affine), so UltrafastSecp256k1 uses direct affine addition with a field inversion to return an affine result -- this eliminates the separate inversion in serialization.

---

## 60-Second Quickstart

Get a working selftest in under a minute:

**Option A -- Linux (apt)**
```bash
sudo apt install libufsecp3
ufsecp_selftest          # Expected: "OK (version 3.x, backend CPU)"
```

**Option B -- npm (any OS)**
```bash
npm i ufsecp
node -e "require('ufsecp').selftest()"   # Expected: "OK"
```

**Option C -- Python (any OS)**
```bash
pip install ufsecp
python -c "import ufsecp; ufsecp.selftest()"  # Expected: "OK"
```

**Option D -- Build from source**
```bash
git clone https://github.com/shrec/UltrafastSecp256k1.git && cd UltrafastSecp256k1
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
./build/selftest          # Expected: "ALL TESTS PASSED"
```

---

## Platform Support Matrix

| Target | Backend | Install / Entry Point | Status |
|--------|---------|----------------------|--------|
| **Linux x64** | CPU | `apt install libufsecp3` | [OK] Stable |
| **Windows x64** | CPU | NuGet `UltrafastSecp256k1` / [Release .zip](https://github.com/shrec/UltrafastSecp256k1/releases) | [OK] Stable |
| **macOS (x64/ARM64)** | CPU + Metal | `brew install ufsecp` / build from source | [OK] Stable |
| **Android ARM64** | CPU | `implementation 'io.github.shrec:ufsecp'` (Maven) | [OK] Stable |
| **iOS ARM64** | CPU | Swift Package / CocoaPods / XCFramework | [OK] Stable |
| **Browser / Node.js** | WASM | `npm i ufsecp` | [OK] Stable |
| **ESP32-S3 / ESP32** | CPU | PlatformIO / IDF component | [OK] Tested |
| **STM32 (Cortex-M)** | CPU | CMake cross-compile | [OK] Tested |
| **NVIDIA GPU** | CUDA 12+ | Build with `-DSECP256K1_BUILD_CUDA=ON` | [OK] Stable |
| **AMD GPU** | ROCm/HIP | Build with `-DSECP256K1_BUILD_ROCM=ON` | [!] Beta |
| **Apple GPU** | Metal | Build with Metal backend | [..] Experimental (discovery only) |
| **Any GPU** | OpenCL | Build with `-DSECP256K1_BUILD_OPENCL=ON` | [OK] Full (6/6 ops) |
| **RISC-V (RV64GC)** | CPU | Cross-compile | [OK] Tested |

---

## Installation

### Linux (APT -- Debian / Ubuntu)

```bash
# Add repository
curl -fsSL https://shrec.github.io/UltrafastSecp256k1/apt/KEY.gpg | sudo gpg --dearmor -o /etc/apt/keyrings/ultrafastsecp256k1.gpg
echo "deb [signed-by=/etc/apt/keyrings/ultrafastsecp256k1.gpg] https://shrec.github.io/UltrafastSecp256k1/apt stable main" \
  | sudo tee /etc/apt/sources.list.d/ultrafastsecp256k1.list
sudo apt update

# Install (runtime only)
sudo apt install libufsecp3

# Install (development -- headers, static lib, cmake/pkgconfig)
sudo apt install libufsecp-dev
```

### Linux (RPM -- Fedora / RHEL)

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

| Backend | Hardware | kG/s | ECDSA Sign | ECDSA Verify | Schnorr Sign | Schnorr Verify | FROST Verify |
|---------|----------|------|------------|--------------|--------------|----------------|-------------|
| **CUDA** | RTX 5060 Ti | 4.59 M/s | 4.88 M/s | **4.05 M/s** | 3.66 M/s | **5.38 M/s** | **1.34 M/s** |
| **OpenCL** | RTX 5060 Ti | 3.86 M/s | -- | 2.44 M/s* | -- | 2.82 M/s* | — |
| **Metal** | Apple M3 Pro | 0.33 M/s | -- | -- | -- | -- |
| **ROCm (HIP)** | AMD GPUs | Portable | -- | -- | -- | -- |

*CUDA 12.0, sm_86;sm_89, batch=16K signatures, measured on RTX 5060 Ti. The CUDA path uses our own hybrid GPU execution model, which improved end-to-end throughput by more than 10% during optimization. Metal 2.4, 8x32-bit Comba limbs, 18 GPU cores. (\*) OpenCL ECDSA/Schnorr verify uses extended kernel with lazy-loaded runtime compilation.*

### CUDA Core ECC Operations (Kernel-Only Throughput)

| Operation | Time/Op | Throughput |
|-----------|---------|------------|
| Field Mul | 0.2 ns | 4,142 M/s |
| Field Add | 0.2 ns | 4,130 M/s |
| Field Inv | 10.2 ns | 98.35 M/s |
| Point Add | 1.6 ns | 619 M/s |
| Point Double | 0.8 ns | 1,282 M/s |
| Scalar Mul (Pxk) | 225.8 ns | 4.43 M/s |
| Generator Mul (Gxk) | 217.7 ns | 4.59 M/s |
| Batch Inv (Montgomery) | 2.9 ns | 340 M/s |
| Jac->Affine (per-pt) | 14.9 ns | 66.9 M/s |

### GPU Signature Operations (ECDSA + Schnorr)

| Operation | Time/Op | Throughput | Protocol | Δ vs prev |
|-----------|---------|------------|----------|----------|
| **ECDSA Sign** | **204.8 ns** | **4.88 M/s** | RFC 6979 + low-S | — |
| **ECDSA Verify** | **246.7 ns** | **4.05 M/s** | Shamir + GLV | **+66%** |
| **ECDSA Sign+Recid** | **311.5 ns** | **3.21 M/s** | Recoverable (EIP-155) | — |
| **Schnorr Sign** | **273.4 ns** | **3.66 M/s** | BIP-340 | — |
| **Schnorr Verify** | **185.9 ns** | **5.38 M/s** | BIP-340 + GLV | **+91%** |
| **FROST Partial Verify** | **748.9 ns** | **1.34 M/s** | t-of-n threshold | ⭐ New |

### CUDA vs OpenCL Comparison (RTX 5060 Ti)

| Operation | CUDA | OpenCL | Winner |
|-----------|------|--------|--------|
| Field Mul | 0.2 ns | 0.2 ns | Tie |
| Field Inv | 10.2 ns | 14.3 ns | **CUDA 1.40x** |
| Point Double | 0.8 ns | 0.9 ns | **CUDA 1.13x** |
| Point Add | 1.6 ns | 1.6 ns | Tie |
| kG (Generator Mul) | 217.7 ns | 258.9 ns | **CUDA 1.19x** |
| BIP352 Pipeline | 91.0 ns | 93.6 ns | **CUDA 1.03x** |

*Benchmarks: 2026-02-14, Linux x86_64, NVIDIA Driver 580.126.09. Both kernel-only (no buffer allocation/copy overhead).*

### Apple Metal (M3 Pro) -- Kernel-Only

| Operation | Time/Op | Throughput |
|-----------|---------|------------|
| Field Mul | 1.9 ns | 527 M/s |
| Field Inv | 106.4 ns | 9.40 M/s |
| Point Add | 10.1 ns | 98.6 M/s |
| Point Double | 5.1 ns | 196 M/s |
| Scalar Mul (Pxk) | 2.94 us | 0.34 M/s |
| Generator Mul (Gxk) | 3.00 us | 0.33 M/s |

*Metal 2.4, 8x32-bit Comba limbs, Apple M3 Pro (18 GPU cores, Unified Memory 18 GB)*

---

## secp256k1 ECDSA & Schnorr Signatures (BIP-340, RFC 6979)

Full signature support across CPU and GPU:

- **ECDSA**: RFC 6979 deterministic nonces, low-S normalization, DER/Compact encoding, public key recovery (recid)
- **Schnorr**: BIP-340 compliant -- tagged hashing, x-only public keys
- **Batch verification**: ECDSA and Schnorr batch verify
- **Multi-scalar**: Shamir's trick (k_1xG + k_2xQ) for fast verification

### CPU Signature Benchmarks (x86-64, Clang 19, AVX2, Release)

| Operation | Time | Throughput |
|-----------|------:|----------:|
| ECDSA Sign (RFC 6979) | 8.5 us | 118,000 op/s |
| ECDSA Verify | 23.6 us | 42,400 op/s |
| Schnorr Sign (BIP-340) | 6.8 us | 146,000 op/s |
| Schnorr Verify (BIP-340) | 24.0 us | 41,600 op/s |
| Key Generation (CT) | 9.5 us | 105,500 op/s |
| Key Generation (fast) | 5.5 us | 182,000 op/s |
| ECDH | 23.9 us | 41,800 op/s |

*Schnorr sign is ~25% faster than ECDSA sign due to simpler nonce derivation (no modular inverse). Measured single-core, pinned, 2026-02-21.*

---

## Constant-Time secp256k1 (Side-Channel Resistance)

The `ct::` namespace provides constant-time operations for secret-key material -- no secret-dependent branches or memory access patterns:

| Operation | Fast | CT | Overhead |
|-----------|------:|------:|--------:|
| Field Mul | 17 ns | 23 ns | 1.08x |
| Field Inverse | 0.8 us | 1.7 us | 2.05x |
| Complete Addition | -- | 276 ns | -- |
| Scalar Mul (kxP) | 23.6 us | 26.6 us | 1.13x |
| Generator Mul (kxG) | 5.3 us | 9.9 us | 1.86x |

**CT layer provides:** `ct::field_mul`, `ct::field_inv`, `ct::scalar_mul`, `ct::point_add_complete`, `ct::point_dbl`

**Use the CT layer for**: private key operations, signing, nonce generation, ECDH.
**Use the FAST layer for**: verification, public key derivation, batch processing, benchmarks.

See [THREAT_MODEL.md](THREAT_MODEL.md) for a full layer-by-layer risk assessment.

### CT Evidence & Methodology

| Evidence | Scope | Status |
|----------|-------|--------|
| **No secret-dependent branches** | All `ct::` functions | [OK] Enforced by design, verified via Clang-Tidy checks |
| **No secret-dependent memory access** | All `ct::` table lookups use constant-index cmov | [OK] |
| **ASan + UBSan CI** | Every push -- catches undefined behavior in CT paths | [OK] CI |
| **Timing tests (dudect)** | CPU field/scalar ops | [OK] Implemented in CI + nightly + native ARM64 |
| **Deterministic CT verification** | `ct-verif` LLVM + Valgrind CT | [OK] Implemented |

**Assumptions:** CT guarantees depend on compiler not introducing secret-dependent branches during optimization. Builds use `-O2` with Clang; MSVC may require additional flags. Micro-architectural side channels (Spectre, power analysis) are outside current scope -- see [THREAT_MODEL.md](THREAT_MODEL.md).

---

## Zero-Knowledge Proofs (Schnorr Sigma, DLEQ, Bulletproofs)

UltrafastSecp256k1 provides ZK proof primitives over the secp256k1 curve:

| Proof Type | Prove | Verify | Proof Size | Use Cases |
|------------|-------|--------|------------|-----------|
| **Knowledge Proof** | 20.3 us | 21.8 us | 64 bytes | Prove knowledge of discrete log (x: P = x*G) |
| **DLEQ Proof** | 40.0 us | 56.4 us | 64 bytes | Prove log_G(P) == log_H(Q) -- VRFs, adaptor sigs, atomic swaps |
| **Bulletproof Range** | 13,467 us | 2,634 us | ~620 bytes | Prove committed value in [0, 2^64) -- Confidential Transactions |

**Security model:**
- All proving operations use the **CT layer** (constant-time, side-channel resistant)
- All verification uses the **FAST layer** (variable-time; public inputs only — no secret material)
- Non-interactive via **Fiat-Shamir** (tagged SHA-256)
- Nothing-up-my-sleeve generators for Bulletproofs (no trusted setup)

**API:** `#include <secp256k1/zk.hpp>` -- namespace `secp256k1::zk`

```cpp
// Knowledge proof: prove you know x such that P = x*G
auto proof = zk::knowledge_prove(secret, pubkey, msg, aux_rand);
bool ok = zk::knowledge_verify(proof, pubkey, msg);

// DLEQ: prove log_G(P) == log_H(Q)
auto dleq = zk::dleq_prove(secret, G, H, P, Q, aux_rand);
bool ok = zk::dleq_verify(dleq, G, H, P, Q);

// Bulletproof range proof: prove committed value in [0, 2^64)
auto rp = zk::range_prove(value, blinding, commitment, aux_rand);
bool ok = zk::range_verify(commitment, rp);
```

*Benchmarks: i7-14400F, 11 passes, pinned core, median. See [docs/BENCHMARKS.md](docs/BENCHMARKS.md).*

---

## secp256k1 Benchmarks -- Cross-Platform Comparison

### CPU: x86-64 vs ARM64 vs RISC-V

| Operation | x86-64 (Clang 21, AVX2) | ARM64 (Cortex-A76) | RISC-V (Milk-V Mars) |
|-----------|-------------------------:|--------------------:|---------------------:|
| Field Mul | 17 ns | 74 ns | 95 ns |
| Field Square | 14 ns | 50 ns | 70 ns |
| Field Add | 1 ns | 8 ns | 11 ns |
| Field Inverse | 1 us | 2 us | 4 us |
| Point Add | 159 ns | 992 ns | 1 us |
| Generator Mul (kxG) | 5 us | 14 us | 33 us |
| Scalar Mul (kxP) | 25 us | 131 us | 154 us |

### GPU: CUDA vs OpenCL vs Metal

| Operation | CUDA (RTX 5060 Ti) | OpenCL (RTX 5060 Ti) | Metal (M3 Pro) |
|-----------|--------------------:|---------------------:|---------------:|
| Field Mul | 0.2 ns | 0.2 ns | 1.9 ns |
| Field Inv | 10.2 ns | 14.3 ns | 106.4 ns |
| Point Add | 1.6 ns | 1.6 ns | 10.1 ns |
| Generator Mul (Gxk) | 217.7 ns | 295.1 ns | 3.00 us |

### Embedded: ESP32-S3 vs ESP32 vs STM32

| Operation | ESP32-S3 LX7 (240 MHz) | ESP32 LX6 (240 MHz) | STM32F103 (72 MHz) |
|-----------|-------------------:|-------------------:|-------------------:|
| Field Mul | 6,105 ns | 6,993 ns | 15,331 ns |
| Field Square | 5,020 ns | 6,247 ns | 12,083 ns |
| Field Add | 850 ns | 985 ns | 4,139 ns |
| Field Inv | 2,524 us | 609 us | 1,645 us |
| **Fast** Scalar x G | 5,226 us | 6,203 us | 37,982 us |
| **CT** Scalar x G | 15,527 us | -- | -- |
| **CT** Generator x k | 4,951 us | -- | -- |

### Field Representation: 5x52 vs 4x64

| Operation | 4x64 | 5x52 | Speedup |
|-----------|------:|------:|--------:|
| Multiplication | 42 ns | 15 ns | **2.76x** |
| Squaring | 31 ns | 13 ns | **2.44x** |
| Addition | 4.3 ns | 1.6 ns | **2.69x** |
| Add chain (32 ops) | 286 ns | 57 ns | **5.01x** |

*5x52 uses `__int128` lazy reduction -- ideal for 64-bit platforms.*

For full benchmark results, see [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

---

## secp256k1 on Embedded (ESP32 / STM32 / ARM Cortex-M)

UltrafastSecp256k1 runs on resource-constrained microcontrollers with **portable C++ (no `__int128`, no assembly required)**:

- **ESP32-S3** (Xtensa LX7 @ 240 MHz): Fast scalar x G in 5.2 ms, **CT generator x k in 4.9 ms**
- **ESP32-PICO-D4** (Xtensa LX6 @ 240 MHz): Scalar x G in 6.2 ms, CT layer available (44.8 ms CT)
- **STM32F103** (ARM Cortex-M3 @ 72 MHz): Scalar x G in 38 ms with ARM inline assembly (UMULL/ADDS/ADCS)
- **Android ARM64** (RK3588, Cortex-A76 @ 2.256 GHz): Scalar x G in 14 us, Scalar x P in 131 us, ECDSA Sign 30 us

All 37 library tests pass on every embedded target. See [examples/esp32_test/](examples/esp32_test/) and [examples/stm32_test/](examples/stm32_test/).

### Porting to New Platforms

See [PORTING.md](PORTING.md) for a step-by-step checklist to add new CPU architectures, embedded targets, or GPU backends.

---

## WASM secp256k1 (Browser & Node.js)

WebAssembly build via Emscripten -- runs secp256k1 in any modern browser or Node.js:

```bash
./scripts/build_wasm.sh        # -> build/wasm/dist/
```

Output: `secp256k1_wasm.wasm` + `secp256k1.mjs` (ES6 module with TypeScript declarations).
See [wasm/README.md](wasm/README.md) for JavaScript/TypeScript integration.

---

## secp256k1 Batch Modular Inverse (Montgomery Trick)

All backends include **batch modular inversion** -- a critical building block for Jacobian->Affine conversion:

| Backend | Function | Notes |
|---------|----------|-------|
| **CPU** | `fe_batch_inverse(FieldElement*, size_t)` | Montgomery trick with scratch buffer |
| **CUDA** | `batch_inverse_montgomery` / `batch_inverse_kernel` | GPU Montgomery trick kernel |
| **Metal** | `batch_inverse` | Chunked parallel threadgroups |
| **OpenCL** | Inline PTX inverse | Batch via host orchestration |

**Algorithm**: Montgomery batch inverse computes N field inversions using only **1 modular inversion + 3(N-1) multiplications**, amortizing the expensive inversion across the entire batch.

For N=1024: ~500x cheaper than individual inversions. A single field inversion costs ~3.5 us (Fermat), while batch amortizes to ~7 ns per element.

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

Production GPU apps use a memory-efficient variant: instead of storing full Z coordinates, `jacobian_add_mixed_h` returns **H = U2 - X1** separately. Since Z_k = Z_0 * H_0 * H_1 * … * H_{k-1}, the entire Z chain is invertible from H values + initial Z_0.

**Cost**: 1 Fermat inversion + 2N multiplications per thread (vs N Fermat inversions naively).

> See `apps/secp256k1_search_gpu_only/gpu_only.cu` (step kernel) + `unified_split.cuh` (batch inversion kernel)

---

## secp256k1 Stable C ABI (`ufsecp`) -- FFI Bindings

Starting with **v3.4.0**, UltrafastSecp256k1 ships a stable C ABI -- `ufsecp` -- designed for FFI bindings (C#, Python, Rust, Go, Java, Node.js, Dart, React Native, PHP, Ruby, etc.):

```
+--------------------------------------------------+
|                  Your Application                |
|          (C, C#, Python, Go, Rust, …)            |
+------------------+-------------------------------+
                   |  ufsecp C ABI (45 functions)
+------------------▼-------------------------------+
|           ufsecp.dll / libufsecp.so              |
|  Opaque ctx  |  Error model  |  ABI versioning   |
+--------------+---------------+-------------------+
|   FAST layer (variable-time public ops)          |
+--------------------------------------------------+
|   CT layer (constant-time secret-key ops)        |
+--------------------------------------------------+
```

**Default behavior:**
- **C ABI (`ufsecp`)**: Defaults to safe behavior -- all secret-key operations (sign, derive, ECDH) use CT internally. No configuration needed.
- **C++ API**: Exposes both `fast::` and `ct::` namespaces -- the developer chooses explicitly per call site.

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

### GPU C ABI (`ufsecp_gpu`)

Starting with **v3.3.0**, the GPU layer is fully accessible from any FFI language via `ufsecp_gpu.h`:

| Category | Functions |
|----------|-----------|
| **Discovery** | `gpu_backend_count`, `gpu_backend_name`, `gpu_is_available`, `gpu_device_count`, `gpu_device_info` |
| **Lifecycle** | `gpu_ctx_create`, `gpu_ctx_destroy`, `gpu_last_error`, `gpu_last_error_msg`, `gpu_error_str` |
| **Batch Ops** | `gpu_generator_mul_batch`, `gpu_ecdsa_verify_batch`, `gpu_schnorr_verify_batch`, `gpu_ecdh_batch`, `gpu_hash160_pubkey_batch`, `gpu_msm`, `gpu_frost_verify_partial_batch`, `gpu_ecrecover_batch` |

| Batch Operation | CUDA | OpenCL | Metal |
|----------------|------|--------|-------|
| `generator_mul_batch` | [OK] | [OK] | [OK] |
| `ecdsa_verify_batch` | [OK] | [OK] | [OK] |
| `schnorr_verify_batch` | [OK] | [OK] | [OK] |
| `ecdh_batch` | [OK] | [OK] | [OK] |
| `hash160_pubkey_batch` | [OK] | [OK] | [OK] |
| `msm` | [OK] | [OK] | [OK] |
| `frost_verify_partial_batch` | [OK] | [OK] | [OK] |
| `ecrecover_batch` | [OK] | [..] temporary stub | [..] temporary stub |

See [ufsecp_gpu.h](include/ufsecp/ufsecp_gpu.h) and [GPU Validation Matrix](docs/GPU_VALIDATION_MATRIX.md) for details.

### CPU C ABI Coverage

| Category | Functions |
|----------|-----------|
| **Context** | `ctx_create`, `ctx_destroy`, `selftest`, `last_error` |
| **Keys** | `keygen`, `seckey_verify`, `pubkey_create`, `pubkey_parse`, `pubkey_serialize` |
| **ECDSA** | `ecdsa_sign`, `ecdsa_sign_batch`, `ecdsa_verify`, `ecdsa_sign_der`, `ecdsa_verify_der`, `ecdsa_recover` |
| **Schnorr** | `schnorr_sign`, `schnorr_sign_batch`, `schnorr_verify` |
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

- **Transaction Signing & Verification** -- CPU constant-time signing + GPU-accelerated batch verification across Bitcoin, Ethereum, and 25+ blockchains
- **Batch Signature Verification** -- verify thousands of ECDSA/Schnorr signatures per second for block validation
- **HD Wallet Key Derivation** -- BIP-32/44 hierarchical deterministic derivation with 27-coin address generation
- **Embedded IoT Signing** -- ESP32 and STM32 on-device key generation and transaction signing
- **High-Throughput Indexing** -- GPU-accelerated public key derivation for address indexing services
- **Zero-Knowledge Proof Systems** -- Pedersen commitments, adaptor signatures for ZK protocols
- **Multi-Party Computation** -- MuSig2 (BIP-327) and FROST threshold signing
- **Cross-Platform Cryptographic Services** -- single codebase across server (CUDA), desktop (OpenCL/Metal), mobile (ARM64), browser (WASM), and embedded (ESP32/STM32)
- **Cryptographic Research & Benchmarking** -- field/group operation microbenchmarks, algorithm variant comparison

> ### Testers Wanted
> We need community testers for platforms we cannot fully validate in CI:
> - **iOS** -- Build & run on real iPhone/iPad hardware with Xcode
> - **AMD GPU (ROCm/HIP)** -- Test on AMD Radeon RX / Instinct GPUs
>
> [Open an issue](https://github.com/shrec/UltrafastSecp256k1/issues) with your results!

---

## Building secp256k1 from Source (CMake)

### Prerequisites

- CMake 3.18+
- C++20 compiler (GCC 11+, Clang/LLVM 15+, MSVC 2022+)
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
./scripts/build_wasm.sh        # -> build/wasm/dist/
```

### iOS (XCFramework)

```bash
./scripts/build_xcframework.sh  # -> build/xcframework/output/
```

Universal XCFramework (arm64 device + arm64 simulator). Also available via **Swift Package Manager** and **CocoaPods**.

### Local ARM64 / RISC-V QEMU Smoke

```bash
# ARM64 cross-build + QEMU smoke
bash ./scripts/run-qemu-smoke.sh arm64

# RISC-V cross-build + QEMU smoke
bash ./scripts/run-qemu-smoke.sh riscv64

# Both architectures
bash ./scripts/run-qemu-smoke.sh all
```

This local helper runs the same cross-arch smoke surface now used in CI:
`run_selftest smoke`, `test_bip324_standalone`, `bench_kP`, and `bench_bip324`.
Install the corresponding cross toolchain, libc sysroot, `qemu-user-static`, and `ninja-build` first.

If you prefer the existing local CI entry point, the same coverage is also available as:

```bash
bash ./scripts/local-ci.sh --job qemu-smoke

# Optional: limit to one architecture
SECP256K1_QEMU_SMOKE_TARGET=arm64 bash ./scripts/local-ci.sh --job qemu-smoke
SECP256K1_QEMU_SMOKE_TARGET=riscv64 bash ./scripts/local-ci.sh --job qemu-smoke
```

### Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `SECP256K1_USE_ASM` | ON | Assembly optimizations (x64/ARM64/RISC-V) |
| `SECP256K1_BUILD_CUDA` | OFF | CUDA GPU support |
| `SECP256K1_BUILD_OPENCL` | OFF | OpenCL GPU support |
| `SECP256K1_BUILD_ROCM` | OFF | ROCm/HIP GPU support (AMD) |
| `SECP256K1_BUILD_TESTS` | ON | Test suite |
| `SECP256K1_BUILD_BENCH` | ON | Benchmarks |
| `SECP256K1_GLV_WINDOW_WIDTH` | platform | GLV window width (4-7); default 5 on x86/ARM/RISC-V, 4 on ESP32/WASM |
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
    // Public key derivation: private_key x G = public_key
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

Two security profiles are **always active** -- no flag-based selection:

### FAST Profile (Default)

- Maximum throughput, variable-time algorithms
- Use for: verification, batch processing, public key derivation, benchmarking
- [!] **Not safe for secret key operations** -- timing side-channels possible

### CT / Hardened Profile (`ct::` namespace)

- Constant-time arithmetic -- no secret-dependent branches or memory access
- ~5-7x performance penalty vs FAST
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

### Library Stack

```
+----------------------------------------------------------+
|           Language Bindings (FFI / C ABI)                 |
|  Python | Node.js | Rust | Go | C# | Java | Swift | PHP |
+----------------------------------------------------------+
                          |
                   Bindings Layer
                  (ctypes / koffi / cgo
                   JNA / P/Invoke / FFI)
                          |
+----------------------------------------------------------+
|            UltrafastSecp256k1 Core (C++20)                |
|                                                          |
|  Field Arithmetic | Scalar Ops | Point Ops | GLV/Endomo  |
|  ECDSA | Schnorr BIP-340 | ECDH | MuSig2 | FROST       |
|  Pedersen | Taproot | BIP-32 HD | Adaptor Sigs | ZK      |
|                                                          |
|  [FAST layer]              [CT layer]                    |
|  Variable-time             Constant-time                 |
|  Max throughput            Side-channel safe              |
+----------------------------------------------------------+
                          |
+----------+----------+----------+----------+--------------+
|   CPU    |   CUDA   |  OpenCL  |  Metal   |  Embedded    |
|          |          |          |          |              |
| x86_64   | NVIDIA   | AMD/NVIDIA| Apple   | ESP32-S3     |
| ARM64    | sm_50+   | any GPU  | Silicon | ESP32-C6     |
| RISC-V   |          |          |          | STM32        |
| WASM     |          |          |          | Cortex-M     |
+----------+----------+----------+----------+--------------+
```

### Hardware Compatibility

| Platform | Architecture | Backend | Status |
|----------|-------------|---------|--------|
| **Desktop CPU** | x86_64 (Intel / AMD) | CPU | [OK] Stable |
| **Desktop CPU** | ARM64 (Apple Silicon, Ampere) | CPU | [OK] Stable |
| **Desktop CPU** | RISC-V RV64GC | CPU | [OK] Stable |
| **Raspberry Pi** | ARM64 (BCM2710, Zero 2 W) | CPU | [..] Testing |
| **NVIDIA GPU** | RTX / GTX / Tesla (sm_50+) | CUDA 12+ | [OK] Stable (8/8 GPU C ABI ops) |
| **AMD GPU** | RDNA / CDNA | OpenCL | [OK] Broad (7/8 GPU C ABI ops; `ecrecover_batch` pending) |
| **AMD GPU** | RDNA / CDNA | ROCm/HIP | [!] Beta |
| **Apple GPU** | Apple Silicon (M1/M2/M3/M4) | Metal | [..] Experimental (7/8 GPU C ABI ops; `ecrecover_batch` pending) |
| **Any GPU** | OpenCL 1.2+ compatible | OpenCL | [OK] Broad (7/8 GPU C ABI ops; `ecrecover_batch` pending) |
| **ESP32-S3** | Xtensa LX7 @ 240 MHz | CPU | [OK] Tested |
| **ESP32-P4** | RISC-V @ 400 MHz | CPU | [OK] Supported |
| **ESP32-C6** | RISC-V (single-core) | CPU | [OK] Supported |
| **STM32** | ARM Cortex-M3/M4 | CPU | [..] Experimental |
| **WebAssembly** | WASM (Emscripten) | CPU | [OK] Stable |
| **Android** | ARM64 (NDK r27c) | CPU | [OK] Stable |
| **iOS** | ARM64 (Xcode) | CPU | [OK] Stable |

> **GPU C ABI ops**: generator_mul_batch, ecdsa_verify_batch, schnorr_verify_batch, ecdh_batch, hash160_pubkey_batch, msm, frost_verify_partial_batch, ecrecover_batch. See [GPU Validation Matrix](docs/GPU_VALIDATION_MATRIX.md) for per-backend details.

### Embedded Targets

| Target | MCU | Clock | Scalar x G | Flash | RAM |
|--------|-----|-------|-----------|-------|-----|
| ESP32-S3 | Xtensa LX7 (dual) | 240 MHz | 5.2 ms | ~120 KB | ~8 KB |
| ESP32-PICO-D4 | Xtensa LX6 (dual) | 240 MHz | 6.2 ms | ~120 KB | ~8 KB |
| ESP32-P4 | RISC-V | 400 MHz | ~3 ms | ~120 KB | ~8 KB |
| ESP32-C6 | RISC-V (single) | 160 MHz | ~12 ms | ~120 KB | ~8 KB |
| STM32F103 | Cortex-M3 | 72 MHz | 38 ms | ~100 KB | ~6 KB |

### Source Directory

```
UltrafastSecp256k1/
+-- cpu/                 # CPU-optimized implementation
|   +-- include/         # Public headers (field.hpp, scalar.hpp, point.hpp, ecdsa.hpp, schnorr.hpp)
|   +-- src/             # Implementation (field_asm_x64.asm, field_asm_riscv64.S, ...)
|   +-- fuzz/            # libFuzzer harnesses
|   +-- tests/           # Unit tests
+-- cuda/                # CUDA GPU acceleration
+-- opencl/              # OpenCL GPU acceleration
+-- metal/               # Apple Metal GPU acceleration
+-- wasm/                # WebAssembly (Emscripten)
+-- android/             # Android NDK (ARM64)
+-- include/ufsecp/      # Stable C ABI
+-- bindings/            # Language bindings (Rust, Python, Node.js, Go, C#, Java, ...)
+-- examples/
|   +-- c_example/       # C API usage
|   +-- rust_example/    # Rust FFI example
|   +-- python_example/  # Python ctypes example
|   +-- nodejs_example/  # Node.js koffi example
|   +-- go_example/      # Go cgo example
|   +-- java_example/    # Java JNA example
|   +-- esp32_test/      # ESP32-S3 Xtensa LX7 port
|   +-- stm32_test/      # STM32F103 ARM Cortex-M3 port
+-- docs/                # Documentation
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
| Linux x64 | CPU | GCC 13 / Clang 17 | [OK] CI |
| Linux x64 | CPU | Clang 17 (ASan+UBSan) | [OK] CI |
| Linux x64 | CPU | Clang 17 (TSan) | [OK] CI |
| Windows x64 | CPU | MSVC 2022 | [OK] CI |
| macOS ARM64 | CPU + Metal | AppleClang | [OK] CI |
| iOS ARM64 | CPU | Xcode | [OK] CI |
| Android ARM64 | CPU | NDK r27c | [OK] CI |
| WebAssembly | CPU | Emscripten | [OK] CI |
| ROCm/HIP | CPU + GPU | ROCm 6.3 | [OK] CI |

### Cross-Platform Audit Results

The `unified_audit_runner` executes **54 audit modules** across 8 sections
(mathematical invariants, constant-time analysis, differential testing, standard
vectors, fuzzing, protocol security, ABI safety, performance validation).

| Platform | OS | Compiler | Modules | Verdict | Time |
|----------|----|----------|---------|---------|------|
| Windows (local) | Windows x86-64 | Clang 21.1.0 | 54/55 | AUDIT-READY | 42 s |
| Linux Docker | Linux x86-64 | GCC 13.3.0 | 54/55 | AUDIT-READY | 51 s |
| Linux CI | Linux x86-64 | Clang 17.0.6 | 55/55 | AUDIT-READY | 48 s |
| Linux CI | Linux x86-64 | GCC 13.3.0 | 55/55 | AUDIT-READY | 52 s |
| Windows CI | Windows x86-64 | MSVC 1944 | 55/55 | AUDIT-READY | 143 s |

> 54/55 = 1 advisory warning (dudect timing smoke -- probabilistic, flakes under hypervisor noise).
> Full reports: [audit/platform-reports/](audit/platform-reports/PLATFORM_AUDIT.md)

---

## secp256k1 Benchmark Targets

| Target | Description |
|--------|-------------|
| `bench_unified` | THE standard: full apple-to-apple vs libsecp256k1 + OpenSSL |
| `bench_ct` | Fast-vs-CT overhead comparison |
| `bench_field_52` | 5x52 field arithmetic micro-benchmarks |
| `bench_field_26` | 10x26 field arithmetic micro-benchmarks |
| `bench_kP` | Scalar multiplication (k*P) benchmarks |

---

## Research Statement

This library explores the **performance ceiling of secp256k1** across CPU architectures (x64, ARM64, RISC-V, Cortex-M, Xtensa) and GPUs (CUDA, OpenCL, Metal, ROCm). Zero external dependencies. Pure C++20.

---

## API Stability

**C++ API**: Not yet stable. Breaking changes may occur before **v4.0**. Core layers (field, scalar, point, ECDSA, Schnorr) are production-ready with full audit coverage. Extended layers (MuSig2, FROST, Adaptor, Pedersen, ZK, Taproot, HD, Coins) are **Experimental** — implemented and covered by PoC exploit tests and CT verification; APIs may change. C++ API signatures may still evolve before v4.0.

**C ABI (`ufsecp`)**: Stable from v3.4.0. ABI version tracked separately. See [SUPPORTED_GUARANTEES.md](include/ufsecp/SUPPORTED_GUARANTEES.md).

---

## Release Signing & Verification

All releases starting from **v3.15.0** are cryptographically signed using
[Sigstore cosign](https://docs.sigstore.dev/) (keyless, GitHub OIDC identity).
Older historical releases remain unsigned but are preserved unchanged.

Every release includes:

| Artifact | Purpose |
|----------|---------|
| `SHA256SUMS` | Checksums for all release archives |
| `SHA256SUMS.sig` | Cosign signature of the manifest |
| `SHA256SUMS.pem` | Signing certificate (Sigstore OIDC) |
| `sbom.cdx.json` | CycloneDX Software Bill of Materials |
| Per-archive `.sig` + `.pem` | Individual artifact signatures |

### Verify checksums

**Linux:**

```bash
curl -LO https://github.com/shrec/UltrafastSecp256k1/releases/latest/download/SHA256SUMS
sha256sum -c SHA256SUMS
```

**macOS:**

```bash
shasum -a 256 -c SHA256SUMS
```

**Windows (PowerShell):**

```powershell
Get-Content SHA256SUMS | ForEach-Object {
  $parts = $_ -split '  '
  $expected = $parts[0]; $file = $parts[1]
  $actual = (Get-FileHash $file -Algorithm SHA256).Hash.ToLower()
  if ($actual -eq $expected) { "[OK] $file" } else { "[FAIL] $file" }
}
```

### Verify signature (cosign)

```bash
cosign verify-blob SHA256SUMS \
  --signature SHA256SUMS.sig \
  --certificate SHA256SUMS.pem \
  --certificate-identity-regexp "github.com/shrec/UltrafastSecp256k1" \
  --certificate-oidc-issuer https://token.actions.githubusercontent.com
```

| Supply Chain | Status |
|-------------|--------|
| SHA256SUMS for all artifacts | [OK] Every release |
| Cosign / Sigstore manifest signing | [OK] v3.15.0+ |
| Per-artifact Cosign signatures | [OK] v3.15.0+ |
| SLSA Build Provenance (GitHub Attestation) | [OK] Every release |
| CycloneDX SBOM | [OK] Every release |
| Reproducible builds documentation | [OK] Dockerfile.reproducible |

---

## FAQ

**Is UltrafastSecp256k1 a drop-in replacement for libsecp256k1?**
> No. It is an independent implementation with a different API. The C ABI (`ufsecp`) provides a stable FFI surface, but function signatures differ from libsecp256k1. Migration requires code changes.

**Is the API stable?**
> The C ABI (`ufsecp`) is stable from v3.4.0. The C++ API (namespaces `fast::`, `ct::`) is mature for Tier 1 features but may change before v4.0.

**What is the constant-time scope?**
> All functions in `ct::` namespace are constant-time: field arithmetic, scalar arithmetic, point multiplication, complete addition, signing, and ECDH. The C ABI uses CT internally for all secret-key operations. See [CT Evidence](#ct-evidence--methodology) above.

**Which parts are production-safe today?**
> This library has not undergone a third-party audit. Tier 1 features (core ECC, ECDSA, Schnorr, ECDH, stable C ABI) are extensively tested, fuzzed, regression-gated, and run through sanitizer-backed CI with a strong self-audit trail and reproducible evidence.

**How do I reproduce the benchmarks?**
> See [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) for exact commands, pinned compiler/driver versions, and raw logs. The [live dashboard](https://shrec.github.io/UltrafastSecp256k1/dev/bench/) tracks performance across commits.

---

## Documentation

| Document | Description |
|----------|-------------|
| [API Reference](docs/API_REFERENCE.md) | Full C++ and C ABI reference |
| [Build Guide](docs/BUILDING.md) | Detailed build instructions for all platforms |
| [Benchmarks](docs/BENCHMARKS.md) | Complete benchmark results and methodology |
| [GPU API](include/ufsecp/ufsecp_gpu.h) | GPU C ABI header (27 functions, 8 ops, 3 backends) |
| [GPU Validation Matrix](docs/GPU_VALIDATION_MATRIX.md) | Per-backend op coverage and validation status |
| [Feature Maturity](docs/FEATURE_MATURITY.md) | Per-feature GPU/CT/fuzz/tier status table |
| [Supported Guarantees](include/ufsecp/SUPPORTED_GUARANTEES.md) | ABI stability tiers and commitment levels |
| [Audit Coverage](AUDIT_COVERAGE.md) | Full audit report with 55 modules and platform verdicts |
| [Audit Guide](docs/AUDIT_GUIDE.md) | How to run and interpret audit suite |
| [Test Matrix](docs/TEST_MATRIX.md) | Comprehensive test coverage map for auditors |
| [ARM64 Audit & Benchmark](docs/ARM64_AUDIT_BENCHMARK.md) | ARM64 platform certification and performance analysis |
| [Threat Model](THREAT_MODEL.md) | Layer-by-layer security risk assessment |
| [Security Policy](SECURITY.md) | Vulnerability reporting and audit status |
| [Porting Guide](PORTING.md) | Add new platforms, architectures, GPU backends |
| [RISC-V Optimizations](RISCV_OPTIMIZATIONS.md) | RISC-V assembly details |
| [ESP32 Setup](docs/ESP32_SETUP.md) | ESP32 embedded development guide |
| [Examples](examples/README.md) | Multi-language binding examples (C, Python, Rust, Node.js, Go, Java) |
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

**MIT License**

This project is licensed under the MIT License.
Previously released versions (up to v3.14.x) were under AGPL-3.0.
As of v3.15.0 the license is MIT -- to align with the broader Bitcoin ecosystem
and remove adoption friction.

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

UltrafastSecp256k1 is an independent implementation -- written from scratch with our own architecture, hybrid GPU execution model, embedded ports, and optimization techniques. The library's core structure and most performance gains came from direct experimentation, profiling, and iteration. At the same time, no project exists in a vacuum. Studying public research and implementation notes from the wider cryptographic community later helped us validate decisions, avoid weaker paths, and uncover additional optimization opportunities.

We want to acknowledge the teams whose public work informed parts of our journey:

- **[bitcoin-core/secp256k1](https://github.com/bitcoin-core/secp256k1)** -- A major reference point for the ecosystem. UltrafastSecp256k1 was built independently from scratch, but studying their published research later helped us benchmark our own implementations, validate design choices, and extract additional optimization ideas for CPU, GPU, and embedded targets.
- **[Bitcoin Core](https://github.com/bitcoin/bitcoin)** contributors -- For open specifications (BIP-340 Schnorr, BIP-341 Taproot, RFC 6979) and a correctness-first engineering culture that benefits everyone building in this space.
- **Pieter Wuille, Jonas Nick, Tim Ruffing** and the libsecp256k1 maintainers -- For publicly sharing research and implementation insights on side-channel resistance, exhaustive testing, field representation trade-offs, and practical optimization techniques. Their published work was valuable to study in the later optimization phase and helped us push our independently built engine further.
- **[@craigraw](https://github.com/craigraw)** ([Sparrow Wallet](https://sparrowwallet.com)) -- For creating the [bench_bip352](https://github.com/craigraw/bench_bip352) standalone BIP-352 Silent Payments scanning benchmark, which provided an independent, reproducible pipeline comparison between secp256k1 implementations.
- **Community / GigaChad** -- For running the full CUDA test suite on RTX 5070 Ti (Blackwell), confirming 45/45 tests pass, and identifying the `CMAKE_CUDA_SEPARABLE_COMPILATION` flag required for Blackwell devices. Results in [docs/COMMUNITY_BENCHMARKS.md](docs/COMMUNITY_BENCHMARKS.md).

We share our optimizations, GPU kernels, embedded ports, and cross-platform techniques freely -- because open-source cryptography grows stronger when knowledge flows in every direction.

Special thanks to the [Stacker News](https://stacker.news) and [Delving Bitcoin](https://delvingbitcoin.org) communities for their early support and technical feedback.

Extra gratitude to [@0xbitcoiner](https://stacker.news/0xbitcoiner) for the initial outreach and for helping bridge the project with the wider Bitcoin developer ecosystem.

---

## Support the Project

If you find **UltrafastSecp256k1** useful, consider supporting its development!

> **We are actively seeking sponsors for a funded bug bounty program, stronger open audit infrastructure, and ongoing development.**
> See the [Seeking Sponsors](#seeking-sponsors----bug-bounty--development) section above for details.

[![Sponsor](https://img.shields.io/badge/Sponsor_This_Project-GitHub_Sponsors-ea4aaa.svg?style=for-the-badge&logo=github)](https://github.com/sponsors/shrec)
[![Donate with Bitcoin Lightning](https://img.shields.io/badge/Lightning_Sats-shrec@stacker.news-F7931A?style=for-the-badge&logo=bitcoin)](https://stacker.news/shrec)
[![PayPal](https://img.shields.io/badge/PayPal-Donate-blue.svg?style=for-the-badge&logo=paypal)](https://paypal.me/IChkheidze)

| Method | Link |
|--------|------|
| **GitHub Sponsors** (preferred) | [github.com/sponsors/shrec](https://github.com/sponsors/shrec) |
| **Bitcoin Lightning** | `shrec@stacker.news` via any Lightning wallet |
| **PayPal** | [paypal.me/IChkheidze](https://paypal.me/IChkheidze) |
| **Corporate / Foundation grants** | [payysoon@gmail.com](mailto:payysoon@gmail.com) |

### What Your Sponsorship Funds

- **Open Audit Infrastructure** -- reproducible audit packs, more validation automation, and reviewer-ready evidence bundles
- **Bug Bounty** -- Financial rewards for security researchers who find vulnerabilities
- **Development** -- GPU acceleration, ZK proofs, formal verification, embedded platform support
- **Infrastructure** -- CI/CD, cross-platform testing, fuzzing, performance regression gates

All sponsors are acknowledged in the README and release notes.

---

**UltrafastSecp256k1** -- High-performance secp256k1 cryptography for CPU, CUDA, OpenCL, mobile, embedded, and WebAssembly. GPU-accelerated ECDSA and Schnorr on CUDA, zero dependencies, constant-time secret-key paths, and broad multi-platform coverage.

<!-- SEO keywords (not rendered by GitHub) -->
<!-- secp256k1 library fastest GPU CUDA OpenCL Metal ROCm ECDSA sign verify Schnorr BIP-340 Bitcoin Ethereum signature acceleration elliptic curve cryptography C++ C++20 high performance zero dependency batch verification constant time side channel resistance embedded ESP32 STM32 ARM Cortex-M RISC-V ARM64 WebAssembly WASM cross-platform multi-coin address generation BIP-32 BIP-44 HD wallet derivation key recovery EIP-155 RFC-6979 transaction signing blockchain cryptocurrency libsecp256k1 alternative NVIDIA AMD Apple Silicon MuSig2 FROST threshold signatures Taproot BIP-341 BIP-342 Pedersen commitments adaptor signatures ECDH key exchange secp256k1 GPU acceleration secp256k1 on embedded secp256k1 benchmarks secp256k1 constant time secp256k1 WASM secp256k1 C ABI FFI bindings Python Go Rust Java Node.js fastest secp256k1 implementation constant-time ECC library for RISC-V bitcoin cryptography optimization high-throughput elliptic curve signing secp256k1 RISC-V constant-time branchless cryptography GLV endomorphism Hamburg signed-digit comb Renes-Costello-Bathalter complete addition formulas dudect side-channel testing ASan UBSan TSan fuzzing libFuzzer valgrind memcheck security audit vulnerability scanning SLSA provenance supply chain security OpenSSF Scorecard CodeQL SonarCloud clang-tidy static analysis Docker container reproducible build Debian APT RPM Arch AUR Linux packaging MIT open source cryptographic library secp256k1 formal verification Fiat-Crypto Montgomery multiplication Barrett reduction BIP-327 multi-party computation MPC digital signatures public key cryptography PKI key agreement protocol -->
