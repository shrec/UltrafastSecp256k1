# API Stability Guide

Classification of every public header by stability tier.

---

## Stability Tiers

| Tier | Meaning | SemVer Promise |
|---|---|---|
| **Stable** | Battle-tested, will not break in minor versions | Breaking changes require major version bump |
| **Provisional** | API may change in minor versions with deprecation warnings | At least 1 minor version deprecation cycle |
| **Experimental** | Can change or be removed at any time | No backward compatibility guarantee |
| **Internal** | Implementation detail -- do not depend on | May change without notice |

---

## Header Classification

### Stable (safe to depend on)

| Header | Description | Since |
|---|---|---|
| `field.hpp` | 256-bit field element (mod p) -- 5x52 limb representation | v1.0 |
| `scalar.hpp` | 256-bit scalar (mod n) -- arithmetic, inverse, serialization | v1.0 |
| `point.hpp` | Affine/Jacobian point arithmetic on secp256k1 | v1.0 |
| `ecdsa.hpp` | ECDSA sign/verify (RFC 6979 deterministic nonce) | v1.0 |
| `schnorr.hpp` | Schnorr sign/verify (BIP-340) | v2.0 |
| `sha256.hpp` | SHA-256 (used internally; stable public API) | v1.0 |
| `sha512.hpp` | SHA-512 (HMAC-SHA512 for RFC 6979) | v2.0 |
| `ecdh.hpp` | ECDH shared secret derivation | v2.0 |
| `recovery.hpp` | ECDSA public key recovery from signature | v2.0 |
| `selftest.hpp` | Runtime self-test for correctness verification | v1.0 |
| `types.hpp` | Core type aliases and constants | v1.0 |
| `config.hpp` | Compile-time feature detection macros | v2.0 |

### Provisional (approaching stable)

| Header | Description | Notes |
|---|---|---|
| `bip32.hpp` | BIP-32 HD key derivation | API shape under review |
| `batch_verify.hpp` | Batch ECDSA/Schnorr verification | Signature may change |
| `multiscalar.hpp` | Multi-scalar multiplication (Shamir, Pippenger) | Return type may change |
| `glv.hpp` | GLV endomorphism decomposition | May be made internal |
| `pedersen.hpp` | Pedersen commitments | Range proof integration TBD |
| `taproot.hpp` | BIP-341 Taproot key tweaking | Depends on Schnorr stability |
| `address.hpp` | Address encoding utilities | Format support may expand |
| `init.hpp` | Explicit initialization (rarely needed) | May merge into selftest |

### Experimental (active development)

| Header | Description | Notes |
|---|---|---|
| `musig2.hpp` | MuSig2 (BIP-327) multi-signature | Protocol still being refined |
| `frost.hpp` | FROST threshold signatures | Early implementation |
| `adaptor.hpp` | Adaptor signatures (atomic swaps) | Research-grade |
| `pippenger.hpp` | Pippenger multi-scalar multiplication | Optimization layer -- API unstable |
| `ecmult_gen_comb.hpp` | Precomputed generator comb tables | Internal optimization detail |
| `precompute.hpp` | Wbits/comb precomputation tables | May become internal |
| `hash_accel.hpp` | SHA-256 hardware acceleration (SHA-NI, ARM CE) | Platform-specific |
| `batch_add_affine.hpp` | Batch affine addition for multi-point ops | Internal batch primitive |
| `context.hpp` | Optional context for configuration | Shape unclear |

### Internal (do not depend on)

| Header | Description |
|---|---|
| `fast.hpp` | Umbrella includes -- for convenience only |
| `field_26.hpp` | 10x26 field representation (32-bit fallback) |
| `field_52.hpp` | 5x52 field representation (primary) |
| `field_52_impl.hpp` | 5x52 implementation details |
| `field_asm.hpp` | Platform-specific ASM for field operations |
| `field_branchless.hpp` | Branchless field primitives |
| `field_h_based.hpp` | H-based field operations |
| `field_optimal.hpp` | Optimal field dispatching |
| `field_simd.hpp` | SIMD field operations |
| `ct_utils.hpp` | Constant-time utility functions |
| `ct/` (directory) | Constant-time implementation internals |
| `coins/` (directory) | Coin-specific parameter sets |
| `test_framework.hpp` | Internal test utilities |

---

## Deprecation Policy

1. Deprecated APIs are annotated with `[[deprecated("message")]]`
2. Deprecated APIs remain functional for at least **one minor version**
3. Removal happens in the next **major version** after deprecation
4. All deprecations are listed in `CHANGELOG.md`

### Currently Deprecated

| Symbol | Deprecated In | Removal Target | Replacement |
|---|---|---|---|
| -- | -- | -- | No active deprecations in v3.3 |

---

## ABI Stability

- **Static library**: Header-level compatibility only (recompile when upgrading)
- **Shared library** (`-DSECP256K1_BUILD_SHARED=ON`): Symbol-level ABI stability within the same major version for Stable-tier APIs
- **Internal symbols**: May change or disappear in any release
- **Layout**: `FieldElement`, `Scalar`, `Point` layouts are **not** guaranteed stable across major versions

---

## Version Macros

```cpp
#include <secp256k1/version.hpp>

// Compile-time version checks
static_assert(SECP256K1_FAST_VERSION_MAJOR >= 3);
static_assert(SECP256K1_FAST_VERSION_MINOR >= 3);

// Runtime version string
const char* v = secp256k1::fast::version_string();
```

---

## Recommended Include Pattern

For maximum forward compatibility, include only what you need:

```cpp
// Good -- stable, minimal
#include <secp256k1/scalar.hpp>
#include <secp256k1/point.hpp>
#include <secp256k1/ecdsa.hpp>

// Acceptable -- stable umbrella
#include <secp256k1/fast.hpp>

// Risky -- experimental, may break
#include <secp256k1/frost.hpp>
#include <secp256k1/musig2.hpp>
```
