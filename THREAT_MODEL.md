# Threat Model

UltrafastSecp256k1 v3.12.1 -- Layer-by-Layer Risk Assessment

---

## Architecture Overview

```
+-----------------------------------------------------------------+
|                     Application Layer                           |
|  (Wallet, Signer, Verifier, Key Manager, Address Generator)     |
+--------------+---------------+-------------------+--------------+
|  Coins (27)  |  HD (BIP-32)  |  Taproot/MuSig2   | FROST/Adaptor|
+--------------+---------------+-------------------+--------------+
|      ECDSA (RFC 6979)  |  Schnorr (BIP-340)  |  Pedersen       |
+-----------------------------------------------------------------+
|  FAST (variable-time)  |  CT (constant-time)                    |
|  secp256k1::fast::     |  secp256k1::ct::                       |
+-----------------------------------------------------------------+
|         Field / Scalar / Point core (4x64 limbs)                |
+-----------------------------------------------------------------+
|  CPU (x64 BMI2/ADX, ARM64, RISC-V, Xtensa, Cortex-M3)         |
|  GPU (CUDA PTX, ROCm/HIP, OpenCL 3.0, Metal)                   |
|  WASM (Emscripten)                                              |
+-----------------------------------------------------------------+
```

> See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed technical architecture.
> See [AUDIT_GUIDE.md](AUDIT_GUIDE.md) for the auditor navigation guide.

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
| Performance penalty | ~5-7x vs FAST |
| Threat | Compiler optimization may break CT guarantees |
| Mitigation | Sanitizer builds (ASan, TSan), manual inspection, `-O2` only |

The CT layer provides complete addition formulas, constant-time field inversion, and timing-safe scalar multiplication. Callers must still zero sensitive buffers after use -- the library does not manage key lifetimes.

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
| Threat | Nonce reuse -> private key recovery |
| Mitigation | RFC 6979 eliminates random nonce dependency |

MuSig2, FROST, and Adaptor Signatures are **experimental**. Their APIs may change and they have not been independently reviewed.

### 5. HD Derivation & Coin Dispatch

**Purpose**: BIP-32/44 key derivation and 27-coin address generation.

| Property | Value |
|----------|-------|
| Key derivation | BIP-32 (hardened + normal) |
| Address generation | Coin-specific encoding (Base58, Bech32, etc.) |
| Secret handling | Derived keys are secret; use CT layer for signing |
| Threat | Incorrect derivation path -> wrong keys |
| Mitigation | Test vectors from BIP-32/44 specifications |

The coin dispatch layer generates addresses only. It does **not** store keys, manage UTXOs, or broadcast transactions.

### 6. Batch Operations & Multi-Scalar

**Purpose**: Batch inverse (Montgomery trick), multi-scalar multiplication.

| Property | Value |
|----------|-------|
| Allocation | Zero heap allocation (scratchpad model) |
| Threat | Incorrect batch inverse -> silent wrong results |
| Mitigation | Sweep-tested up to 8192; boundary KAT vectors; fuzz harness |

---

## Trust Boundaries

```
TRUSTED (this library controls):
  +- Arithmetic correctness (field, scalar, point)
  +- CT layer timing properties
  +- Deterministic nonce generation
  +- Input validation (on-curve, range)

NOT TRUSTED (caller responsibility):
  +- Key storage and lifecycle
  +- Buffer zeroing after use
  +- Choosing FAST vs CT appropriately
  +- Network security / transport
  +- Entropy source (if any randomness needed)
```

---

## What This Library Does NOT Protect Against

- **Physical attacks** (power analysis, EM emanation, fault injection)
- **Compromised compilers** or build environments
- **OS-level memory disclosure** (cold boot, swap file, core dumps)
- **Application-level misuse** (wrong profile selection, key reuse)
- **Quantum adversaries** (secp256k1 is not post-quantum)

---

## Attack Surface Analysis

### A1: Timing Side Channels

| Vector | Risk | Mitigation |
|--------|------|------------|
| Variable-time scalar mul leaking secret bits | HIGH (if `fast::` used with secrets) | Always use `ct::` for secret-dependent ops |
| Variable-time field inversion | MEDIUM | CT inversion uses fixed-length chain |
| Cache-timing on table lookups | MEDIUM | CT uses linear scan + cmov, not indexed |
| Compiler-introduced branches | MEDIUM | `asm volatile` barriers, `-O2` recommended |
| Microarchitecture-specific timing | LOW | dudect testing on x86-64, ARM64 |

**Testing**: `tests/test_ct_sidechannel.cpp` -- dudect Welch t-test, |t| < 4.5

### A2: Nonce Attacks

| Vector | Risk | Mitigation |
|--------|------|------------|
| ECDSA random nonce reuse -> key recovery | CRITICAL | RFC 6979 deterministic nonces (no randomness needed) |
| Biased nonces -> lattice attack | HIGH | RFC 6979 provides uniform distribution |
| Schnorr nonce bias | HIGH | BIP-340 tagged hash nonce derivation |
| FROST nonce mishandling | MEDIUM | Experimental -- under review |

### A3: Arithmetic Errors

| Vector | Risk | Mitigation |
|--------|------|------------|
| Incorrect field reduction | CRITICAL | 641,194 audit checks, fuzz testing |
| Point addition edge cases (P+P, P+O, P+(-P)) | CRITICAL | Complete addition formulas in CT, sweep tests |
| GLV decomposition error | HIGH | Reconstruction test: k1+k2*lambda == k for random k |
| SafeGCD inverse error | HIGH | Cross-checked against Fermat chain |
| Batch inverse corrupting elements | MEDIUM | Sweep-tested up to 8192 elements |

### A4: Memory Safety

| Vector | Risk | Mitigation |
|--------|------|------------|
| Buffer overflow in field/scalar ops | CRITICAL | Fixed-size POD types, no dynamic allocation |
| Use-after-free | HIGH | ASan in CI, no heap pointers in hot path |
| Uninitialized memory reads | MEDIUM | Valgrind memcheck (weekly CI) |
| Stack-based secret leakage | MEDIUM | Caller must zero sensitive buffers |

### A5: Supply Chain

| Vector | Risk | Mitigation |
|--------|------|------------|
| Compromised dependency | HIGH | Dependabot, Dependency Review, SLSA attestation |
| Malicious PR injection | HIGH | Branch protection, CODEOWNERS, required reviews |
| Build reproducibility | MEDIUM | Docker SHA-pinned, deterministic builds |
| Typosquatting (npm/vcpkg) | LOW | Official package names documented |

### A6: GPU-Specific

| Vector | Risk | Mitigation |
|--------|------|------------|
| GPU shared memory observable | HIGH | GPU is for public data ONLY |
| Branch divergence leaking data | HIGH | No CT guarantees on GPU |
| Device memory not zeroed | MEDIUM | Do not pass secrets to GPU |
| PCIe bus snooping | LOW | Physical access required |

---

## Recommendations for Integrators

1. **Always use `ct::` for secret scalar operations** (signing, key derivation)
2. **Zero sensitive buffers** after use (`memset_s` or platform equivalent)
3. **Build with sanitizers** regularly (`cpu-asan`, `cpu-tsan` presets)
4. **Run selftest on startup** (`Selftest(false, SelftestMode::smoke)`)
5. **Do not expose GPU memory** to untrusted contexts
6. **Pin your dependency version** -- API may change before v4.0
7. **Review CT_VERIFICATION.md** for known constant-time limitations
8. **Use `-O2` for production CT builds** -- higher levels may break CT properties
9. **Run dudect test** on your target hardware before deployment

---

## Automated Security Measures (v3.12.1)

| Measure | Frequency | What It Catches |
|---------|-----------|-----------------|
| CodeQL | Every push/PR | Static security bugs, injection, overflow |
| OpenSSF Scorecard | Weekly | Supply-chain weaknesses |
| Security Audit CI | Push/PR + weekly | Compiler warnings (-Werror), memory errors, UB |
| Clang-Tidy (30+ checks) | Every push/PR | Bugprone patterns, cert violations |
| SonarCloud | Every push/PR | Code quality, security hotspots |
| ASan + UBSan | CI | Address errors, undefined behavior |
| TSan | CI | Data races, thread safety |
| Valgrind | Weekly | Memory leaks, invalid access |
| Dependabot | Daily | Vulnerable dependency updates |
| Dependency Review | Every PR | New vulnerable dependencies |
| SLSA Attestation | Every release | Build provenance verification |
| libFuzzer | Continuous | Random input crashes |

---

## Related Documents

| Document | Purpose |
|----------|---------|
| [AUDIT_GUIDE.md](AUDIT_GUIDE.md) | Auditor navigation and checklist |
| [AUDIT_REPORT.md](AUDIT_REPORT.md) | Internal audit: 641,194 checks |
| [SECURITY.md](SECURITY.md) | Vulnerability reporting, production readiness |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Technical architecture deep-dive |
| [docs/CT_VERIFICATION.md](docs/CT_VERIFICATION.md) | Constant-time methodology |
| [docs/TEST_MATRIX.md](docs/TEST_MATRIX.md) | Test coverage matrix |

---

*UltrafastSecp256k1 v3.12.1 -- Threat Model*
