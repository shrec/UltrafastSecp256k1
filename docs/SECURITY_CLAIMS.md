# Security Claims & API Contract

**UltrafastSecp256k1 v3.13.0** -- FAST / CT Dual-Layer Architecture

---

## 1. Semantic Equivalence Contract

> **FAST and CT functions return identical results for all valid inputs.**

Both layers implement the same secp256k1 elliptic curve operations with the same
mathematical semantics. They differ **only** in execution profile:

| Property | FAST (`secp256k1::fast::`, `secp256k1::`) | CT (`secp256k1::ct::`) |
|----------|-------------------------------------------|------------------------|
| **Throughput** | Maximum | ~2-3x slower |
| **Timing** | Data-dependent (variable-time) | Data-independent (constant-time) |
| **Branching** | May short-circuit on identity/zero | Never branches on secret data |
| **Table Lookup** | Direct index | Scans all entries via cmov |
| **Side-Channel** | Not resistant | Resistant (CPU backend) |

### Where Results May Differ

Both layers are tested for bit-exact equivalence. Possible divergences:

- **Error handling**: Both return zero/infinity for invalid inputs, but CT may
  take longer to return on error (it completes the full execution trace).
- **Timing**: By design -- FAST is faster, CT is constant-time.
- **Input validation**: Identical. Both reject zero scalars, out-of-range values.

### Verified by CI

FAST == CT equivalence is verified in every CI run:
- `test_ct` -- arithmetic, scalar mul, generator mul, ECDSA sign, Schnorr sign
- `test_ct_equivalence` -- property-based (random + edge vectors) 

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
The performance cost is bounded (2-3x) and eliminates timing side-channel risk.

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

## 3. API Mapping: FAST <-> CT

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

---

## 4. CT Timing Verification

CT claims are verified empirically using the **dudect** methodology
(Reparaz, Balasch, Verbauwhede, 2017):

- **Per-PR**: 5-minute smoke test in `security-audit.yml` (every push to main)
- **Nightly**: 30-minute full statistical analysis in `nightly.yml`
- **Threshold**: Welch's t-test, |t| < 4.5 -> PASS

### Functions Under dudect Coverage

`ct::field_mul`, `ct::field_inv`, `ct::field_square`, `ct::scalar_mul`,
`ct::generator_mul`, `ct::point_add`, `field_select`, ECDSA sign, Schnorr sign.

See [docs/CT_VERIFICATION.md](CT_VERIFICATION.md) for full methodology.

### CT Claim Scope

> The CT guarantee applies to the **CPU backend** (`secp256k1::ct::`) under
> the specified compiler (`g++-13` / `clang-17+`) at `-O2`, on **x86-64** and
> **ARM64** architectures.

**Explicitly NOT covered:**
- GPU backends (CUDA, ROCm, OpenCL, Metal) -- SIMT model leaks by design
- Experimental protocols (FROST, MuSig2) -- not CT-audited
- Compilers or optimization levels not tested in CI
- Microarchitectures not in the CI matrix

---

## 5. Release CT Scope Tracking

Every release must answer: **"Did the CT scope change?"**

| Release | CT Scope Changed? | Details |
|---------|-------------------|---------|
| v3.13.0 | **Yes** | Added `ct::ecdsa_sign`, `ct::schnorr_sign`, `ct::schnorr_pubkey`, `ct::schnorr_keypair_create` |
| v3.12.x | No | CT layer existed (scalar/field/point), no high-level sign API |

Future releases will include this in the CHANGELOG:
```
### CT Scope
- Changed: [list affected functions]
- No change (default)
```

---

## 6. Equivalence Test Coverage

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

- 64 random 256-bit scalars -> `ct::generator_mul(k) == fast::scalar_mul(G, k)`
- 64 random scalars -> `ct::scalar_mul(P, k) == fast::scalar_mul(P, k)`
- 32 random key+msg pairs -> `ct::ecdsa_sign == fast::ecdsa_sign` + verify
- 32 random key+msg pairs -> `ct::schnorr_sign == fast::schnorr_sign` + verify
- Boundary scalars: 0, 1, 2, n-1, n-2, (n+1)/2

---

## References

- [SECURITY.md](../SECURITY.md) -- Vulnerability reporting
- [THREAT_MODEL.md](../THREAT_MODEL.md) -- Attack surface analysis
- [docs/CT_VERIFICATION.md](CT_VERIFICATION.md) -- Technical CT methodology, dudect details
- [AUDIT_GUIDE.md](../AUDIT_GUIDE.md) -- Auditor navigation
- [dudect paper](https://eprint.iacr.org/2016/1123) -- Reparaz et al., 2017

---

*UltrafastSecp256k1 v3.13.0 -- Security Claims*
