# UltrafastSecp256k1 v3.6.0 — GPU Signature Operations

## 🎯 Highlights

**World-first:** The only open-source GPU library with secp256k1 ECDSA + Schnorr sign/verify on CUDA.

### GPU Signature Performance (RTX 5060 Ti)

| Operation | Time/Op | Throughput |
|-----------|---------|------------|
| **ECDSA Sign** | 204.8 ns | **4.88 M/s** |
| **ECDSA Verify** | 410.1 ns | **2.44 M/s** |
| ECDSA Sign + Recid | 311.5 ns | 3.21 M/s |
| **Schnorr Sign (BIP-340)** | 273.4 ns | **3.66 M/s** |
| **Schnorr Verify (BIP-340)** | 354.6 ns | **2.82 M/s** |

### Core ECC Performance (RTX 5060 Ti)

| Operation | Time/Op | Throughput |
|-----------|---------|------------|
| Field Mul | 0.2 ns | 4,142 M/s |
| Scalar Mul (P×k) | 225.8 ns | 4.43 M/s |
| Generator Mul (G×k) | 217.7 ns | 4.59 M/s |

## What's New

### GPU Signature Operations
- 6 new CUDA batch kernel wrappers (`__launch_bounds__(128, 2)`):
  - `ecdsa_sign_batch_kernel` — RFC 6979 deterministic nonces, low-S normalization
  - `ecdsa_verify_batch_kernel` — Shamir's trick + GLV endomorphism
  - `ecdsa_sign_recoverable_batch_kernel` — with recovery ID
  - `ecdsa_recover_batch_kernel` — public key recovery
  - `schnorr_sign_batch_kernel` — BIP-340 with tagged hash midstates
  - `schnorr_verify_batch_kernel` — x-only pubkey verification

### Benchmarks
- 5 new GPU signature benchmarks in `bench_cuda.cu`
- `prepare_ecdsa_test_data()` helper for verify benchmark correctness
- Fixed CUDA benchmark thread mismatch (256 vs `__launch_bounds__(128)`)

### Documentation
- README: blockchain coin badges, GPU signature benchmark tables, 27-coin table, SEO footer
- BENCHMARKS.md: split CUDA section into Core ECC + Signatures
- API_REFERENCE.md: full CUDA Signature Operations section
- CHANGELOG.md: v3.6.0 entry
- Wiki: updated Benchmarks.md and CUDA-Guide.md

## Supported Blockchains (27 coins)

Bitcoin, Ethereum, Litecoin, Dogecoin, Bitcoin Cash, Bitcoin SV, Zcash, Dash, DigiByte, Namecoin, Peercoin, Vertcoin, Viacoin, Groestlcoin, Syscoin, BNB Smart Chain, Polygon, Avalanche, Fantom, Arbitrum, Optimism, Ravencoin, Flux, Qtum, Horizen, Bitcoin Gold, Komodo

## Platforms

| Backend | Status |
|---------|--------|
| CUDA (NVIDIA) | ✅ Full signatures |
| OpenCL (NVIDIA/AMD) | ✅ Core ECC |
| Metal (Apple Silicon) | ✅ Core ECC |
| CPU (x86-64/ARM64/RISC-V) | ✅ Full signatures |
| WASM | ✅ Full signatures |

## Build

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/cuda/secp256k1_cuda_bench
```
