# Security Claims & API Contract

**UltrafastSecp256k1 v3.68.0** -- FAST / CT Dual-Layer Architecture (CPU + GPU)

## Operating Assumption

This document is written for owner-grade deployment standards, not just
outside evaluation.

The working question is not "will outsiders trust this?" The working question
is "if this engine had to satisfy the same standard we would demand for
protecting large assets, what claims would we require before accepting that
risk?"

That means:

1. Claims should be conservative.
2. Residual risks should be explicit.
3. Secret-bearing paths should be judged more harshly than public-data paths.
4. Experimental features should be treated as opt-in risk, not silently upgraded to production trust.
5. Secret-bearing code changes must be paired with updates to the matching evidence docs, enforced by `ci/check_secret_path_changes.py`.

## Fail-Closed Assurance Perimeter

The default `ci/audit_gate.py` path now treats the following as first-class
enforcement surfaces rather than optional tooling:

1. Failure-class matrix execution
2. ABI hostile-caller quartet coverage
3. Structured invalid-input grammar rejection
4. Stateful multi-call sequence integrity
5. Audit test-quality scanning

Two explicit non-claims remain important:

1. The library does **not** claim total hostile-caller quartet closure while the live ABI blocker set is non-zero.
2. Mutation kill-rate evidence is part of the assurance perimeter, but remains the heavier batch / owner-grade lane rather than the default per-commit runtime path.

Additional enforcement surfaces added in the 2026-04-07 CI hardening:

6. Crash-risk analysis: division-by-zero and other crash vectors in CT-sensitive paths
7. MemorySanitizer (MSan): detects use-of-uninitialized-memory, complements zeroization
8. Coverage upload failure gating: `fail_ci_if_error: true`
9. CT scalar_inverse(0) zero guard: both SafeGCD and Fermat fallback CT paths now return
   `Scalar::zero()` for zero input, matching the FAST path behavior (defense-in-depth)
10. Boundary sentinel test suite: 18 exploit-class checks covering inverse(0), empty batch
    verify, half-order low-S boundary, MuSig2 duplicate keys, aux_rand edge values,
    has_even_y(infinity), and CT inverse round-trips (`test_exploit_boundary_sentinels`)

---

## 1. Semantic Equivalence Contract

> **FAST and CT functions return identical results for all valid inputs.**

Both layers implement the same secp256k1 elliptic curve operations with the same
mathematical semantics. They differ **only** in execution profile:

| Property | FAST (`secp256k1::fast::`) | Public-namespace signing (`secp256k1::`) | CT (`secp256k1::ct::`) |
|----------|-----------------------------|------------------------------------------|------------------------|
| **Throughput** | Maximum | ~same as CT for signing | ~1.8-3.2x slower than fast:: |
| **Signing timing** | Data-dependent (variable-time) | **CT** — routes through `ct::generator_mul_blinded` + `ct::scalar_inverse` | Data-independent (constant-time) |
| **Branching** | May short-circuit on identity/zero | No secret-dependent branches in signing | Never branches on secret data |
| **Table Lookup** | Direct index | Blinded comb (same as ct::) for sign | Scans all entries via cmov |
| **Nonce Erasure** | Not erased | Stack nonces erased (k, k_inv, z) | Intermediate nonces erased (volatile fn-ptr) |
| **Side-Channel** | Not resistant | Resistant for signing paths | Resistant (CPU backend) |
| **Audit/ABI backing** | No | Use `ct::` or ABI for explicit audit trail | Canonical CT-validated path |

> **Note (2026-05-01):** `secp256k1::ecdsa_sign`, `secp256k1::schnorr_sign`, and
> `secp256k1::ecdsa_sign_recoverable` were found to internally call
> `ct::generator_mul_blinded` and `ct::scalar_inverse`, making them CT-equivalent
> in practice. The previous documentation describing them as "variable-time" was
> incorrect. See `src/cpu/src/ecdsa.cpp:signing_generator_mul` and
> `src/cpu/src/recovery.cpp:signing_generator_mul` (both alias `ct::generator_mul_blinded`),
> and `src/cpu/src/schnorr.cpp` (calls `ct::generator_mul_blinded` directly).
> The canonical explicit CT path (`secp256k1::ct::ecdsa_sign`) remains preferred
> for audits and new code because it carries explicit CT intent.

### CT Overhead by Platform (v3.4.0)

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

**2026-05-04 (add_affine_fast_ct magnitude fix):** Corrected FE52 negate()
magnitude parameters in the incomplete mixed-add used by `ct::generator_mul`
and `ct::scalar_mul_prebuilt_fast`. The 5x52 lazy-reduction scheme requires
`negate(m)` to be called with m ≥ actual element magnitude to avoid uint64
underflow. Affected parameters: negate(8)→negate(18) for X1 (mag≤18 after
point_dbl_n_core), negate(4)→negate(10) for Y1 (mag≤10), negate(1)→negate(7)
for X3 (mag=7 after two-add accumulation). This was the root cause of all
signing correctness failures introduced with the incomplete-add optimization.
CT timing invariant unaffected.

**2026-05-04 (AVX2 comb_lookup):** `ct::generator_mul` reduced from 13951 ns to ~8700 ns
(-37.7%) on x86-64-v3 (AVX2) via vectorized 32-entry comb table scan in `comb_lookup`.
The AVX2 path uses the same XOR/AND CT blend idiom as `table_lookup_core` (already used
by `ct::scalar_mul`), processing each 80-byte CTAffinePoint as 2.5 ymm registers.
CT invariant preserved: no data-dependent branches; all 32 entries touched per lookup.
Post-AVX2 CT-vs-libsecp comparison (x86-64-v3, bench_unified):
  ct::generator_mul:  8700 ns  (was 13951 ns)
  CT Schnorr Sign:   10864 ns  vs libsecp 13635 ns  → 1.26x  (was 0.87x)
  CT ECDSA Sign:     12897 ns  vs libsecp 17809 ns  → 1.38x  (was 0.91x)

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

Preflight coverage reporting is graph-driven. The project graph builder records
coverage from direct CTest targets and from selected unified-audit modules for
public-data code paths such as batch verification, recovery, taproot, address,
and multiscalar logic.

---

## 2. Developer Guidance: When to Use FAST vs CT

### CT Is REQUIRED For:

| Operation | Why | Function |
|-----------|-----|----------|
| **ECDSA signing** | Private key enters scalar multiplication | `ct::ecdsa_sign()` (canonical); `secp256k1::ecdsa_sign()` also CT as of 2026-05-01 |
| **Schnorr signing** | Private key + nonce in scalar mul | `ct::schnorr_sign()` (canonical); `secp256k1::schnorr_sign()` also CT as of 2026-05-01 |
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
// [OK] CORRECT: explicit CT namespace for signing — canonical, auditable path
#include <secp256k1/ct/sign.hpp>
auto sig = secp256k1::ct::ecdsa_sign(msg_hash, private_key);

// [OK] ALSO CORRECT: secp256k1::ecdsa_sign routes through ct::generator_mul_blinded
// and ct::scalar_inverse (see src/cpu/src/ecdsa.cpp:signing_generator_mul).
// Use this only when you have confirmed CT coverage via the source graph.
// For new code, prefer the explicit ct:: namespace for audit clarity.
#include <secp256k1/ecdsa.hpp>
auto sig2 = secp256k1::ecdsa_sign(msg_hash, private_key);

// [OK] CORRECT: FAST for verification (all inputs public)
#include <secp256k1/ecdsa.hpp>
bool ok = secp256k1::ecdsa_verify(msg_hash, pubkey, sig);

// [FAIL] WRONG: direct fast:: scalar_mul on generator with secret scalar —
// variable-time, leaks private key timing.
// Point::generator().scalar_mul(private_key)  <-- never do this for secrets
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
- `ufsecp_pubkey_xonly` — rejects x-coordinate >= p when lifting to curve point

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
- **Status**: Early implementation. API may change while CAAS protocol evidence matures.

### FROST (RFC 9591)

- **DKG scalar operations**: use `ct::` namespace
- **Signing round scalar mul**: CT-protected
- **Protocol-level timing**: added to dudect in v3.16.0 (sample counts lower)
- **Secret scratch reduction**: 2026-04-14 refactor removed temporary signer-ID and
  binding-factor heap vectors from signing/verification/aggregation paths; this lowers
  transient secret-adjacent allocation pressure without changing the public API.
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
| Knowledge prove | N/A | `zk::knowledge_prove()` (uses CT internally) |
| DLEQ prove | N/A | `zk::dleq_prove()` (uses CT internally) |
| Range prove | N/A | `zk::range_prove()` (uses CT internally) |
| Knowledge verify | `zk::knowledge_verify()` | N/A (public data) |
| DLEQ verify | `zk::dleq_verify()` | N/A (public data) |
| Range verify | `zk::range_verify()` | N/A (public data) |
| Scalar cond. move | N/A (use if/else) | `ct::scalar_cmov(r, a, mask)` |
| Scalar cond. swap | N/A (use std::swap) | `ct::scalar_cswap(a, b, mask)` |
| Scalar cond. negate | `s.negate()` with if | `ct::scalar_cneg(a, mask)` |

### GPU (CUDA/OpenCL/Metal) API

All GPU CT functions are in the `secp256k1::cuda::ct::` namespace (CUDA),
with equivalent kernels in OpenCL (`secp256k1_ct_sign.cl`, `secp256k1_ct_zk.cl`)
and Metal (`secp256k1_ct_sign.metal`, `secp256k1_ct_zk.metal`).
All three backends implement identical CT algorithms.

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
| Knowledge prove | N/A | `ct_knowledge_prove_device(sec, pk, base, msg, aux, &pf)` |
| DLEQ prove | N/A | `ct_dleq_prove_device(sec, G, H, P, Q, aux, &pf)` |
| Knowledge verify | `knowledge_verify_device(...)` | N/A (public data) |
| DLEQ verify | `dleq_verify_device(...)` | N/A (public data) |
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

- **Per-PR**: dudect smoke test (`|t| < 25.0`, ~60s) in the Extended Tests workflow
- **Per-push** (dev/main): full statistical analysis (`|t| < 4.5`, ~30 min) in the Extended Tests workflow
- **Native ARM64**: Apple Silicon M1 (macos-14): smoke per-PR + full per-push in `ct-arm64.yml`
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
> - **CUDA GPU**: `secp256k1::cuda::ct::` under CUDA 12.0+ / nvcc, on **SM 7.5+** (Turing through Blackwell)
> - **OpenCL GPU**: CT kernels in `secp256k1_ct_sign.cl` / `secp256k1_ct_zk.cl`
> - **Metal GPU**: CT shaders in `secp256k1_ct_sign.metal` / `secp256k1_ct_zk.metal`

All GPU CT layers provide **algorithmic** constant-time guarantees (no secret-dependent
branches or memory access patterns). Hardware-level side-channel resistance on GPUs
is limited by the SIMT/SIMD execution model.

**Explicitly NOT covered:**
- Protocol internals of FROST and MuSig2 -- partial coverage only
- Compilers or optimization levels not tested in CI
- Microarchitectures not in the CI matrix
- Hardware-level electromagnetic/power analysis on any platform

---

## 8. ZK Proof Security Properties

### Schnorr Knowledge Proof

- **Soundness**: Prover cannot forge proof without knowing discrete log (Fiat-Shamir in ROM)
- **Zero-Knowledge**: Proof reveals no information about secret beyond the public key
- **Binding**: Challenge derived via tagged SHA-256 ("ZK/knowledge"), bound to R, P, and msg
- **CT**: Proving uses `ct::generator_mul` for nonce commitment; nonce erased after use

### DLEQ Proof (Discrete Log Equality)

- **Soundness**: Both discrete logs must be identical or attack succeeds with negligible probability
- **Binding**: Challenge bound to full tuple (G, H, P, Q, R1, R2) via tagged SHA-256
- **Zero-Knowledge**: Proof reveals no information about the shared secret
- **CT**: Proving uses CT scalar multiplications for both bases

### Bulletproof Range Proof

- **Completeness**: Valid proofs always verify
- **Soundness**: Prover cannot create proof for value outside [0, 2^64) except with negligible probability
- **Zero-Knowledge**: Proof leaks no information about value or blinding factor
- **Logarithmic Size**: O(log n) group elements for n-bit range (12 group elements for 64-bit)
- **No Trusted Setup**: Nothing-up-my-sleeve generators derived from tagged hashes
- **CT**: Prover uses CT layer for all secret-dependent operations (blinding, nonce generation)
- **Verification**: Uses FAST layer with MSM optimization (public data only)

---

## 9. Release CT Scope Tracking

Every release must answer: **"Did the CT scope change?"**

| Release | CT Scope Changed? | Details |
|---------|-------------------|---------|
| dev (2026-05-04) | **Perf, no scope change** | `ct::generator_mul` comb inner loop: complete unified add (12M+2S) → incomplete mixed Jacobian+affine add (7M+3S, `add_affine_fast_ct`). CT invariants (fixed iteration count, branchless cmov lookup) unchanged. Safety: all table entries are fixed G multiples; degenerate probability ~2^-128. Both 52-bit and 4x64 paths updated. |
| v3.22.0 | **Yes** | OpenCL CT layer (secp256k1_ct_sign.cl, secp256k1_ct_zk.cl); Metal CT layer (secp256k1_ct_sign.metal, secp256k1_ct_zk.metal); full C ABI with 80+ functions; BIP-39, Ethereum, Pedersen, ZK, Adaptor, MuSig2, FROST |
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

<!-- 2026-04-28: ufsecp_gpu.h docstring corrected — ufsecp_gpu_context_create → ufsecp_gpu_ctx_create (phantom export removal, misuse_resistance gate fix). No behavioral change; GPU ABI secret-bearing claims unchanged. -->
<!-- 2026-05-04: ct_point.cpp — generator_mul comb inner loop switched to incomplete mixed add (add_affine_fast_ct). CT scope unchanged; fixed iteration count and branchless cmov table lookup preserved. Performance improvement only. -->

*UltrafastSecp256k1 v3.68.0 -- Security Claims*
