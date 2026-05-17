# UltrafastSecp256k1 v4.0 — Optional Secondary secp256k1 Backend for Evaluation

## Overview

UltrafastSecp256k1 v4.0 is a high-performance secp256k1 engine built for evaluation as an **optional secondary backend** for Bitcoin Core. The goal is not to replace libsecp256k1, but to make it possible to measure, compare, and selectively enable an alternative implementation under controlled conditions.

This post presents the current state of the project, its integration model, and the evidence gathered through continuous audit infrastructure.

Repository: https://github.com/shrec/UltrafastSecp256k1
Release v4.0.0: https://github.com/shrec/UltrafastSecp256k1/releases/tag/v4.0.0

---

## Integration Model

Integration uses a shim layer that exposes the identical `secp256k1.h` API surface. Bitcoin Core can be built with the alternative backend using a single CMake flag:

```bash
cmake -B build -DSECP256K1_BACKEND=ultrafast
cmake --build build
```

The default backend remains libsecp256k1 unchanged. All existing Bitcoin Core C++ source files remain unmodified — only the CMake build system references a different library.

Fork demonstrating this integration: https://github.com/shrec/bitcoin/tree/feature/ultrafast-secp256k1-backend

Shim API coverage:
- `secp256k1.h` — context, pubkey, seckey
- `secp256k1_extrakeys.h` — keypair, x-only pubkey (BIP-340/341)
- `secp256k1_schnorrsig.h` — Schnorr sign/verify (BIP-340)
- `secp256k1_ecdh.h` — ECDH
- `secp256k1_recovery.h` — ECDSA recovery
- `secp256k1_ellswift.h` — ElligatorSwift (BIP-324)
- `secp256k1_musig.h` — MuSig2 (BIP-327, all 14 functions)

---

## Performance — Bitcoin Core Integration Paths

All numbers from `bench_bitcoin` (Bitcoin Core's native benchmark harness) on Intel i5-14400F, GCC 14.2.0, Release+LTO, `intel_pstate/no_turbo=1`, `taskset -c 0`, `nice -20`, 5 runs.

Canonical artifact: [`docs/BITCOIN_CORE_BENCH_RESULTS.json`](https://github.com/shrec/UltrafastSecp256k1/blob/main/docs/BITCOIN_CORE_BENCH_RESULTS.json)

**Transaction signing:**

| Benchmark | Ultra | libsecp256k1 | Delta |
|---|---|---|---|
| `SignSchnorrWithMerkleRoot` | 83.9 µs | 113.4 µs | **+35% faster** |
| `SignSchnorrWithNullMerkleRoot` | 84.0 µs | 113.0 µs | **+35% faster** |
| `SignTransactionECDSA` | 149.5 µs | 165.1 µs | **+10% faster** |
| `SignTransactionSchnorr` | 125.4 µs | 137.5 µs | **+10% faster** |

**Script verification:**

| Benchmark | Ultra | libsecp256k1 | Delta |
|---|---|---|---|
| `VerifyScriptP2TR_KeyPath` | 45.4 µs | 46.3 µs | +2.0% faster |
| `VerifyScriptP2TR_ScriptPath` | 76.5 µs | 83.8 µs | **+10% faster** |
| `VerifyScriptP2WPKH` | 46.0 µs | 45.8 µs | parity (within noise) |

**Block validation aggregate (ConnectBlock, 2000 unique signatures):**

| Scenario | Ultra | libsecp256k1 | Delta |
|---|---|---|---|
| All ECDSA | 254.3 ms | 257.4 ms | **+1.2% faster** |
| All Schnorr | 253.0 ms | 255.3 ms | **+0.9% faster** |
| Mixed (2k Schnorr + 1k ECDSA) | 253.9 ms | 257.7 ms | **+1.5% faster** |

> Without LTO: ConnectBlock is ~0.5–1.0% **slower** than libsecp256k1 due to i-cache pressure from a larger code footprint. LTO is required for Ultra to win the aggregate. This tradeoff is documented in [`docs/SHIM_KNOWN_DIVERGENCES.md`](https://github.com/shrec/UltrafastSecp256k1/blob/main/docs/SHIM_KNOWN_DIVERGENCES.md).

**Bitcoin Core test suite: 749/749 passing** with Ultra backend (GCC 14.2.0, May 2026).

---

## Performance — Constant-Time Signing Primitives

From [`docs/bench_unified_2026-05-16_gcc14_x86-64.json`](https://github.com/shrec/UltrafastSecp256k1/blob/main/docs/bench_unified_2026-05-16_gcc14_x86-64.json). CT-vs-CT co-measured in the same run (ratios are TSC-independent):

| Operation | Ultra CT | libsecp256k1 | Ratio |
|---|---|---|---|
| CT ECDSA sign | 21.6 µs | 59.7 µs | **1.30× faster** |
| CT Schnorr sign (BIP-340) | 18.1 µs | 46.5 µs | **1.28× faster** |
| Schnorr verify | 84.3 µs | 84.3 µs | equal |

**Field arithmetic primitives:**

| Primitive | Ultra | libsecp256k1 | Ratio |
|---|---|---|---|
| `field_mul` | 20.1 ns | 26.5 ns | 1.32× |
| `field_sqr` | 18.7 ns | 22.3 ns | 1.19× |
| `field_inv` | 1253.9 ns | 1506.0 ns | 1.20× |
| `field_from_bytes` | 4.8 ns | 13.0 ns | 2.71× |

---

## Continuous Audit and Assurance (CAAS)

The repository includes a continuous audit infrastructure tracking security regressions across every commit.

Current state at v4.0.0:
- **262 exploit PoC** modules covering 20+ CVE/attack classes — all pass
- **369 total audit modules** (262 exploit PoC + 107 non-exploit)
- **CAAS autonomy score: 100/100** (8/8 gates)

Source: [`docs/SECURITY_AUTONOMY_KPI.json`](https://github.com/shrec/UltrafastSecp256k1/blob/main/docs/SECURITY_AUTONOMY_KPI.json)

Audit surfaces include: nonce reuse, side-channel timing (dudect), CT boundary verification, batch verify soundness, MuSig2/FROST protocol attacks, adaptor signatures, DER BIP-66 strict parsing, BIP-340/RFC-6979 known-answer tests, Wycheproof vectors, structure-aware fuzzing, differential testing vs libsecp256k1, and Python-based algebraic property testing.

Security properties enforced:
- Constant-time signing: ECDSA, Schnorr, MuSig2, FROST, BIP-324 XDH
- Per-context DPA blinding (`secp256k1_context_randomize` fully implemented)
- Strict scalar parsing for private key inputs (`parse_bytes_strict_nonzero`)
- Fail-closed batch signing APIs
- BIP-66 strict DER enforcement in all shim paths

---

## Benchmarking Methodology

Benchmarks distinguish between:
- **CT vs CT** — constant-time Ultra signing vs constant-time libsecp signing (production-equivalent, fair comparison)
- **Bitcoin Core integration** — `bench_bitcoin` binary, real validation pipeline
- **LTO vs no-LTO** — both measured and documented
- **Warm-cache vs cold-cache** — noted where applicable

All claimed improvements include error percentage from nanobench's internal statistics. Inconclusive results (overlapping ranges) are reported as such. Results are reproducible from the canonical JSON artifacts in the repository.

---

## Reproducibility

Builds are deterministic:
- `-ffile-prefix-map` strips source paths from debug info
- `SOURCE_DATE_EPOCH` awareness
- Fixed `-march=x86-64-v3` (no host-native variation)
- SLSA provenance attached to v4.0.0 release artifacts (via Sigstore/Cosign)

---

## Platform Coverage

CI passing on all platforms as of v4.0.0:

| Platform | Architecture | Compiler |
|---|---|---|
| Linux | x86-64-v3 | GCC 14 / Clang 17 |
| Linux | ARM64 | GCC 14 |
| Linux | RISC-V 64 | GCC 14 |
| macOS | ARM64 (Apple Silicon) | Clang 15 |
| Windows | x86-64 | MSVC / GCC |

Additional: Android ARM64 (NDK), WASM, ESP32, STM32.

---

## Known Limitations

- ConnectBlock improvement requires Release+LTO. Without LTO: ~0.5–1.0% slower than libsecp256k1.
- GPU backends (CUDA, OpenCL, Metal) are present but not part of the Bitcoin Core evaluation profile.
- This covers the CPU backend only.

---

## Current Objective

No mandatory integration path is proposed. The current objective is to make an alternative backend available for evaluation on technical grounds, with reproducible evidence, minimal integration surface, and an easy rollback path.

The default backend remains libsecp256k1. All existing behavior is preserved unless the new backend is explicitly enabled at build time.

---

## Reviewer Entry Path

```bash
git clone https://github.com/shrec/UltrafastSecp256k1
cd UltrafastSecp256k1
python3 ci/verify_external_audit_bundle.py --allow-commit-mismatch
```

Key documents:
- [`docs/BITCOIN_CORE_BACKEND_EVIDENCE.md`](https://github.com/shrec/UltrafastSecp256k1/blob/main/docs/BITCOIN_CORE_BACKEND_EVIDENCE.md) — full reviewer package
- [`docs/BENCHMARKS.md`](https://github.com/shrec/UltrafastSecp256k1/blob/main/docs/BENCHMARKS.md) — benchmark methodology and raw data
- [`docs/AUDIT_CHANGELOG.md`](https://github.com/shrec/UltrafastSecp256k1/blob/main/docs/AUDIT_CHANGELOG.md) — security audit history
- [`docs/SHIM_KNOWN_DIVERGENCES.md`](https://github.com/shrec/UltrafastSecp256k1/blob/main/docs/SHIM_KNOWN_DIVERGENCES.md) — documented behavioral differences from libsecp256k1

---

*All performance numbers are from controlled benchmark runs with hard turbo lock. Raw data available in the repository. Canonical benchmark artifact: `docs/BITCOIN_CORE_BENCH_RESULTS.json`, `docs/bench_unified_2026-05-16_gcc14_x86-64.json`.*
