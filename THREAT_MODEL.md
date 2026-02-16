# Threat Model

UltrafastSecp256k1 v3.2.0 — Layer-by-Layer Risk Assessment

---

## Architecture Overview

```
┌──────────────────────────────────────────────────┐
│                  Application                     │
├──────────────┬───────────────┬───────────────────┤
│  Coins (27)  │  HD (BIP-32)  │  Taproot/MuSig2  │
├──────────────┴───────────────┴───────────────────┤
│            ECDSA / Schnorr / Adaptor             │
├──────────────────────────────────────────────────┤
│  FAST (variable-time)  │  CT (constant-time)     │
├────────────────────────┴─────────────────────────┤
│         Field / Scalar / Point core              │
├──────────────────────────────────────────────────┤
│  CPU (x64, ARM64, RISC-V, M3, Xtensa)           │
│  GPU (CUDA, ROCm/HIP, OpenCL, Metal)            │
└──────────────────────────────────────────────────┘
```

---

## Layer Definitions

### 1. FAST Layer (Default)

**Purpose**: Maximum throughput for public-key operations, verification, and research.

| Property | Value |
|----------|-------|
| Constant-time | No |
| Secret key handling | Not safe |
| Intended use | Verification, batch processing, benchmarking |
| Threat | Side-channel attacks if used with secrets |
| Mitigation | Use `ct::` namespace for any secret-dependent operation |

Variable-time algorithms may leak information about operands through timing, cache access patterns, or branch prediction state. This is acceptable only when all operands are public (e.g., signature verification, public key derivation from known scalars).

### 2. CT Layer (`ct::` namespace)

**Purpose**: Side-channel resistant operations for secret key material.

| Property | Value |
|----------|-------|
| Constant-time | Yes (no secret-dependent branches or memory access) |
| Secret key handling | Designed for this |
| Performance penalty | ~5–7× vs FAST |
| Threat | Compiler optimization may break CT guarantees |
| Mitigation | Sanitizer builds (ASan, TSan), manual inspection, `-O2` only |

The CT layer provides complete addition formulas, constant-time field inversion, and timing-safe scalar multiplication. Callers must still zero sensitive buffers after use — the library does not manage key lifetimes.

**Known limitation**: No formal verification (e.g., ct-verif, Vale) has been applied. CT guarantees rely on code review and compiler discipline.

### 3. GPU Backends (CUDA, ROCm, OpenCL, Metal)

**Purpose**: High-throughput batch operations on GPU hardware.

| Property | Value |
|----------|-------|
| Constant-time | No |
| Secret key handling | Not safe |
| Intended use | Batch verification, public key generation, search |
| Threat | GPU memory may be observable; no CT guarantees on device |
| Mitigation | Use GPU only for public-data workloads |

GPU kernels are variable-time by design. Device memory is not zeroed. Do not pass secret keys to GPU kernels.

### 4. Signature Schemes (ECDSA, Schnorr, MuSig2, FROST, Adaptor)

**Purpose**: Digital signatures and multi-party signing protocols.

| Property | Value |
|----------|-------|
| Nonce generation | Deterministic (RFC 6979 for ECDSA) |
| Input validation | Point-on-curve, scalar range checks |
| Threat | Nonce reuse → private key recovery |
| Mitigation | RFC 6979 eliminates random nonce dependency |

MuSig2, FROST, and Adaptor Signatures are **experimental**. Their APIs may change and they have not been independently reviewed.

### 5. HD Derivation & Coin Dispatch

**Purpose**: BIP-32/44 key derivation and 27-coin address generation.

| Property | Value |
|----------|-------|
| Key derivation | BIP-32 (hardened + normal) |
| Address generation | Coin-specific encoding (Base58, Bech32, etc.) |
| Secret handling | Derived keys are secret; use CT layer for signing |
| Threat | Incorrect derivation path → wrong keys |
| Mitigation | Test vectors from BIP-32/44 specifications |

The coin dispatch layer generates addresses only. It does **not** store keys, manage UTXOs, or broadcast transactions.

### 6. Batch Operations & Multi-Scalar

**Purpose**: Batch inverse (Montgomery trick), multi-scalar multiplication.

| Property | Value |
|----------|-------|
| Allocation | Zero heap allocation (scratchpad model) |
| Threat | Incorrect batch inverse → silent wrong results |
| Mitigation | Sweep-tested up to 8192; boundary KAT vectors; fuzz harness |

---

## Trust Boundaries

```
TRUSTED (this library controls):
  ├─ Arithmetic correctness (field, scalar, point)
  ├─ CT layer timing properties
  ├─ Deterministic nonce generation
  └─ Input validation (on-curve, range)

NOT TRUSTED (caller responsibility):
  ├─ Key storage and lifecycle
  ├─ Buffer zeroing after use
  ├─ Choosing FAST vs CT appropriately
  ├─ Network security / transport
  └─ Entropy source (if any randomness needed)
```

---

## What This Library Does NOT Protect Against

- **Physical attacks** (power analysis, EM emanation, fault injection)
- **Compromised compilers** or build environments
- **OS-level memory disclosure** (cold boot, swap file, core dumps)
- **Application-level misuse** (wrong profile selection, key reuse)
- **Quantum adversaries** (secp256k1 is not post-quantum)

---

## Recommendations for Integrators

1. **Always use `ct::` for secret scalar operations** (signing, key derivation)
2. **Zero sensitive buffers** after use (`memset_s` or platform equivalent)
3. **Build with sanitizers** regularly (`cpu-asan`, `cpu-tsan` presets)
4. **Run selftest on startup** (`Selftest(false, SelftestMode::smoke)`)
5. **Do not expose GPU memory** to untrusted contexts
6. **Pin your dependency version** — API may change before v4.0

---

*UltrafastSecp256k1 v3.2.0 — Threat Model*
