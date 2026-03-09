# Security Claims & API Contract

**UltrafastSecp256k1 v3.21.0** -- FAST / CT Dual-Layer Architecture (CPU + GPU)

---

## 1. Semantic Equivalence Contract

> **FAST and CT functions return identical results for all valid inputs.**

Both layers implement the same secp256k1 elliptic curve operations with the same
mathematical semantics. They differ **only** in execution profile:

| Property | FAST (`secp256k1::fast::`, `secp256k1::`) | CT (`secp256k1::ct::`) |
|----------|-------------------------------------------|------------------------|
| **Throughput** | Maximum | ~1.8-3.2x slower |
| **Timing** | Data-dependent (variable-time) | Data-independent (constant-time) |
| **Branching** | May short-circuit on identity/zero | Never branches on secret data |
| **Table Lookup** | Direct index | Scans all entries via cmov |
| **Nonce Erasure** | Not erased | Intermediate nonces erased (volatile fn-ptr) |
| **Side-Channel** | Not resistant | Resistant (CPU backend) |

### CT Overhead by Platform (v3.21.0)

Measured with `bench_unified` / `gpu_bench_unified` (signing operations; verify uses public inputs -- CT not needed):

| Platform | ECDSA Sign CT/FAST | Schnorr Sign CT/FAST |
|---|---|---|
| x86-64 (i5-14400F, GCC 14.2) | **1.93x** | **2.13x** |
| ARM64 Cortex-A55 (Clang 18) | 2.57x | 3.18x |
| RISC-V U74 @ 1.5 GHz (GCC 13) | 1.96x | 2.37x |
| ESP32-S3 Xtensa LX7 @ 240 MHz | 1.05x | 1.06x |
| **GPU RTX 5060 Ti (CUDA 12.0)** | **2.06x** | **2.51x** |

ESP32 has near-zero CT overhead: in-order core, no speculative execution. x86 overhead
improved in v3.16.0 (was 1.94x ECDSA) following the GLV decomposition correctness fix.

### Where Results May Differ

Both layers are tested for bit-exact equivalence. Possible divergences:

- **Error handling**: Both return zero/infinity for invalid inputs, but CT may
  take longer to return on error (it completes the full execution trace).
- **Timing**: By design — FAST is faster, CT is constant-time.
- **Input validation**: Identical. Both reject zero scalars, out-of-range values.

### Verified by CI

FAST == CT equivalence is verified in every CI run:
- `test_ct` — arithmetic, scalar mul, generator mul, ECDSA sign, Schnorr sign
- `test_ct_equivalence` — property-based (random + edge vectors)

---

## 2. Developer Guidance: When to Use FAST vs CT

### CT Is REQUIRED For:

| Operation | Why | Function |
|-----------|-----|----------|
| **ECDSA signing** | Private key enters scalar multiplication | `ct::ecdsa_sign()` |
| **Schnorr signing** | Private key + nonce in scalar mul | `ct::schnorr_sign()` |
| **Key generation / derivation** | Secret scalar x G | `ct::generator_mul()` |
| **Keypair creation** | Private key enters point mul | `ct::schnorr_keypair_create()` |
| **X-only pubkey from privkey** | Secret scalar x G | `ct::schnorr_pubkey()` |
| **Any scalar mul with secret scalar** | Timing leaks scalar bits | `ct::scalar_mul()` |
| **Nonce generation** | k must remain secret | RFC 6979 (used internally) |
| **Secret-dependent selection** | Branch on secret data | `ct::scalar_cmov/cswap/select` |

### FAST Is OK For:

| Operation | Why | Function |
|-----------|-----|----------|
| **ECDSA verification** | All inputs are public | `ecdsa_verify()` |
| **Schnorr verification** | All inputs are public | `schnorr_verify()` |
| **Batch verification** | Public signatures + public keys | `schnorr_verify()` in loop |
| **Public key arithmetic** | No secret data involved | `Point::scalar_mul()` on public key |
| **Parsing / serialization** | No secret data | `from_bytes()`, `to_bytes()` |
| **Hash operations** | BIP-340 tagged hash on public data | `tagged_hash()` |
| **Address generation from public key** | No secret data | All coin-dispatch functions |

### If You Are Unsure: Use CT

When in doubt about whether an input is secret, **always use the CT variant**.
The performance cost is bounded (1.8-3.2x depending on platform) and eliminates
timing side-channel risk.

```cpp
// [OK] CORRECT: CT for signing (private key is secret)
#include <secp256k1/ct/sign.hpp>
auto sig = secp256k1::ct::ecdsa_sign(msg_hash, private_key);

// [OK] CORRECT: FAST for verification (all inputs public)
#include <secp256k1/ecdsa.hpp>
bool ok = secp256k1::ecdsa_verify(msg_hash, pubkey, sig);

// [FAIL] WRONG: FAST for signing (leaks private key timing)
auto sig = secp256k1::ecdsa_sign(msg_hash, private_key);
```

### Compile-Time Guardrail

Define `SECP256K1_REQUIRE_CT=1` to get deprecation warnings on non-CT sign
functions. This helps catch accidental use of the FAST path for secret operations:

```bash
cmake -DCMAKE_CXX_FLAGS="-DSECP256K1_REQUIRE_CT=1" ...
```

---

## 3. BIP-340 Strict Parsing (v3.16.0)

> **All cryptographic parsing now enforces strict encoding by default.**

v3.16.0 adds strict parsing APIs that reject all malformed inputs at parse time,
preventing degenerate or out-of-range values from entering the cryptographic pipeline.

### Strict APIs

| API | Rejects |
|-----|---------|
| `Scalar::parse_bytes_strict(bytes)` | zero scalar, value >= group order n |
| `FieldElement::parse_bytes_strict(bytes)` | zero element, value >= field prime p |
| `SchnorrSignature::parse_strict(bytes)` | r >= p, s >= n |

### C ABI Strict Enforcement

The following C ABI functions use strict parsing internally (v3.16.0):
- `ufsecp_schnorr_verify` — rejects malformed signatures before any computation
- `ufsecp_schnorr_sign` — validates keypair before signing
- `ufsecp_xonly_pubkey_parse` — rejects x-coordinate >= p

### CMake Option

```cmake
# Enforce strict parsing library-wide (replaces all lenient parse_bytes calls)
-DUFSECP_BITCOIN_STRICT=ON
```

### Test Coverage

31-test BIP-340 strict suite (`test_bip340_strict_parsing`):
- reject-zero scalar, reject-zero field element
- reject overflow (r == n, s == p, r == p+1)
- accept all valid boundary values (r == 1, r == n-1)

---

## 4. CT Nonce Erasure (v3.16.0)

> **Intermediate nonces are erased from the stack after signing.**

`ct::schnorr_sign` and `ct::ecdsa_sign` now erase intermediate RFC 6979 nonces
immediately after use via the **volatile function-pointer trick**, matching the
approach used in bitcoin-core/libsecp256k1:

```cpp
// Pattern used internally in ct::ecdsa_sign and ct::schnorr_sign:
static void (*volatile wipe_fn)(void*, size_t) = memset;
wipe_fn(&nonce_k, 0, sizeof(nonce_k));
```

This is a best-effort mitigation. Complete nonce erasure cannot be guaranteed
due to compiler stack reuse and register allocation — this is true for all
cryptographic implementations, including libsecp256k1.

---

## 5. FROST / MuSig2 Protocol CT Status (v3.16.0)

### MuSig2 (BIP-327)

- **Scalar multiplications in signing**: use `ct::` namespace — CT-protected
- **Nonce generation**: RFC 6979-based — CT-protected
- **Protocol-level timing**: added to dudect in v3.16.0
- **Status**: Early implementation. API may change. Not externally audited.

### FROST (RFC 9591)

- **DKG scalar operations**: use `ct::` namespace
- **Signing round scalar mul**: CT-protected
- **Protocol-level timing**: added to dudect in v3.16.0 (sample counts lower)
- **Status**: Early implementation. secp256k1 ciphersuite not in RFC 9591.

> **Explicit claim**: Neither MuSig2 nor FROST have been subjected to a
> protocol-level side-channel analysis by a third party. Use in production
> at your own risk.

---

## 6. API Mapping: FAST <-> CT

### CPU API

| Operation | FAST (public data) | CT (secret data) |
|-----------|--------------------|-------------------|
| Scalar x G | `Point::generator().scalar_mul(k)` | `ct::generator_mul(k)` |
| Scalar x P | `P.scalar_mul(k)` | `ct::scalar_mul(P, k)` |
| Point add | `Point::add(P, Q)` | `ct::point_add_complete(P, Q)` |
| Point double | `Point::double_point(P)` | `ct::point_dbl(P)` |
| ECDSA sign | `secp256k1::ecdsa_sign(...)` | `ct::ecdsa_sign(...)` |
| Schnorr sign | `secp256k1::schnorr_sign(...)` | `ct::schnorr_sign(...)` |
| Schnorr pubkey | `secp256k1::schnorr_pubkey(k)` | `ct::schnorr_pubkey(k)` |
| Keypair create | `schnorr_keypair_create(k)` | `ct::schnorr_keypair_create(k)` |
| Scalar cond. move | N/A (use if/else) | `ct::scalar_cmov(r, a, mask)` |
| Scalar cond. swap | N/A (use std::swap) | `ct::scalar_cswap(a, b, mask)` |
| Scalar cond. negate | `s.negate()` with if | `ct::scalar_cneg(a, mask)` |

### GPU (CUDA) API

All GPU CT functions are in the `secp256k1::cuda::ct::` namespace.
Headers: `cuda/include/ct/{ct_ops.cuh, ct_field.cuh, ct_scalar.cuh, ct_point.cuh, ct_sign.cuh}`

| Operation | FAST (`secp256k1::cuda::`) | CT (`secp256k1::cuda::ct::`) |
|-----------|---------------------------|------------------------------|
| Scalar x G | `scalar_mul_generator_const(k, &r)` | `ct_generator_mul(k, &r)` |
| Scalar x P | `scalar_mul(&P, k, &r)` | `ct_scalar_mul(&P, k, &r)` |
| Point add | `jacobian_add(&P, &Q, &r)` | `ct_point_add(&P, &Q, &r)` |
| Point double | `jacobian_double(&P, &r)` | `ct_point_dbl(&P, &r)` |
| Mixed add | N/A | `ct_point_add_mixed(&P, &Q, &r)` |
| ECDSA sign | `ecdsa_sign(msg, key, &sig)` | `ct_ecdsa_sign(msg, key, &sig)` |
| Schnorr sign | `schnorr_sign(key, msg, aux, &sig)` | `ct_schnorr_sign(key, msg, aux, &sig)` |
| Keypair create | N/A | `ct_schnorr_keypair_create(key, &kp)` |
| Field cmov | N/A | `field_cmov(&r, &a, mask)` |
| Scalar cmov | N/A | `scalar_cmov(&r, &a, mask)` |
| Scalar inverse | `scalar_inverse(a, &r)` | `scalar_inverse(a, &r)` (CT Fermat) |

#### GPU CT Throughput (RTX 5060 Ti)

| Operation | ns/op | Throughput | CT/FAST |
|-----------|-------|------------|---------|
| ct::k*G | 341.9 | 2.92 M/s | 2.65x |
| ct::k*P | 347.2 | 2.88 M/s | -- |
| ct::ecdsa_sign | 433.9 | **2.30 M/s** | 2.06x |
| ct::schnorr_sign | 715.8 | **1.40 M/s** | 2.51x |

---

## 7. CT Timing Verification

CT claims are verified empirically using the **dudect** methodology
(Reparaz, Balasch, Verbauwhede, 2017):

- **Per-PR**: smoke test (`|t| < 25.0`, ~30s) in `security-audit.yml`
- **Nightly**: full statistical analysis (`|t| < 4.5`, ~30 min) in `nightly.yml`
- **Native ARM64**: Apple Silicon M1 (macos-14): smoke per-PR + full nightly in `ct-arm64.yml`
- **Valgrind taint**: `MAKE_MEM_UNDEFINED` on all secret inputs, every CI run
- **ct-verif LLVM pass**: compile-time CT verification (no secret-dependent branches at IR level)
- **MuSig2/FROST**: protocol-level timing tests added in v3.16.0

### Functions Under dudect Coverage

`ct::field_mul`, `ct::field_inv`, `ct::field_square`, `ct::scalar_mul`,
`ct::generator_mul`, `ct::point_add_complete`, `field_select`, ECDSA sign,
Schnorr sign, MuSig2 sign (protocol-level), FROST sign (protocol-level).

See [docs/CT_EMPIRICAL_REPORT.md](CT_EMPIRICAL_REPORT.md) for full methodology.

### CT Claim Scope

> The CT guarantee applies to:
> - **CPU**: `secp256k1::ct::` under `g++-13` / `clang-17+` at `-O2`, on **x86-64** and **ARM64**
> - **GPU**: `secp256k1::cuda::ct::` under CUDA 12.0+ / nvcc, on **SM 7.5+** (Turing through Blackwell)

The GPU CT layer provides **algorithmic** constant-time guarantees (no secret-dependent
branches or memory access patterns). Hardware-level side-channel resistance on GPUs
is limited by the SIMT execution model.

**Explicitly NOT covered:**
- Protocol internals of FROST and MuSig2 -- partial coverage only
- Compilers or optimization levels not tested in CI
- Microarchitectures not in the CI matrix
- Hardware-level electromagnetic/power analysis on any platform

---

## 8. Release CT Scope Tracking

Every release must answer: **"Did the CT scope change?"**

| Release | CT Scope Changed? | Details |
|---------|-------------------|---------|
| v3.21.0 | **Yes** | GPU CT layer (5 headers); GPU CT audit modules in gpu_audit_runner; GPU CT benchmarks in gpu_bench_unified |
| v3.16.0 | **Yes** | CT nonce erasure (volatile fn-ptr trick); MuSig2/FROST dudect added; ct-arm64 ARM64 native CI |
| v3.15.0 | **Yes** | Branchless `scalar_window` on RISC-V; `value_barrier` after mask; RISC-V `is_zero_mask` asm |
| v3.13.1 | **Yes (fix)** | GLV decomposition correctness fix; CT scalar_mul overhead reduced to 1.05x |
| v3.13.0 | **Yes** | Added `ct::ecdsa_sign`, `ct::schnorr_sign`, `ct::schnorr_pubkey`, `ct::schnorr_keypair_create` |
| v3.12.x | No | CT layer existed (scalar/field/point), no high-level sign API |

---

## 9. Equivalence Test Coverage

### Automated in CI (`test_ct` + `test_ct_equivalence`)

| Category | Tests | Edge Vectors |
|----------|-------|--------------|
| Field arithmetic | add, sub, mul, sqr, neg, inv, normalize | 0, 1, p-1 |
| Scalar arithmetic | add, sub, neg, half | 0, 1, n-1 |
| Conditional ops | cmov, cswap, select, cneg, is_zero, eq | all-zero, all-ones |
| Point addition | general, doubling, identity, inverse | O+O, P+O, O+P, P+(-P) |
| Scalar mul | k=0,1,2, known vectors, large k, random | 0, 1, 2, n-1, n-2, random |
| Generator mul | fast vs CT equivalence | 1, 2, random 256-bit |
| ECDSA sign | CT vs FAST identical output | Key=1, key=3, random keys |
| Schnorr sign | CT vs FAST identical output | Key=1, key=3, random keys |
| Schnorr pubkey | CT vs FAST identical output | Key=1, random keys |

### Property-Based (`test_ct_equivalence`)

- 64 random 256-bit scalars → `ct::generator_mul(k) == fast::scalar_mul(G, k)`
- 64 random scalars → `ct::scalar_mul(P, k) == fast::scalar_mul(P, k)`
- 32 random key+msg pairs → `ct::ecdsa_sign == fast::ecdsa_sign` + verify
- 32 random key+msg pairs → `ct::schnorr_sign == fast::schnorr_sign` + verify
- Boundary scalars: 0, 1, 2, n-1, n-2, (n+1)/2

---

## References

- [SECURITY.md](../SECURITY.md) — Vulnerability reporting
- [THREAT_MODEL.md](../THREAT_MODEL.md) — Attack surface analysis
- [docs/CT_VERIFICATION.md](CT_VERIFICATION.md) — Technical CT methodology, dudect details
- [docs/CT_EMPIRICAL_REPORT.md](CT_EMPIRICAL_REPORT.md) — Full empirical proof report
- [AUDIT_GUIDE.md](../AUDIT_GUIDE.md) — Auditor navigation
- [dudect paper](https://eprint.iacr.org/2016/1123) — Reparaz et al., 2017

---

*UltrafastSecp256k1 v3.21.0 -- Security Claims*
