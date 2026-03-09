# Constant-Time Verification

**UltrafastSecp256k1 v3.21.0** -- CT Layer Methodology & Audit Status

---

## Overview

The constant-time (CT) layer provides side-channel resistant operations for secret key material. It is available on **both CPU and GPU backends**:

- **CPU**: `secp256k1::ct::` namespace (headers in `cpu/include/secp256k1/ct/`)
- **GPU**: `secp256k1::cuda::ct::` namespace (headers in `cuda/include/ct/`)

The FAST layer (`secp256k1::fast::` on CPU, `secp256k1::cuda::` on GPU) is explicitly variable-time for maximum throughput on public data.

**Principle**: Any operation that touches secret data (private keys, nonces, intermediate scalars) MUST use `ct::` functions. The default `fast::` namespace is allowed only when all inputs are public.

---

## CT Layer Architecture

### CPU CT Layer

```
secp256k1::ct::
+-- ops.hpp          -- Low-level CT primitives (cmov, select, cswap)
+-- field.hpp        -- CT field multiplication, inversion, square
+-- scalar.hpp       -- CT scalar multiplication, addition
+-- point.hpp        -- CT point operations (scalar_mul, generator_mul)
+-- ct_utils.hpp     -- Utility: timing barriers, constant-time comparison

secp256k1::fast::
+-- field_branchless.hpp  -- Branchless field_select (bitwise cmov)
+-- ...                   -- Variable-time (NOT for secrets)
```

### GPU CT Layer

```
secp256k1::cuda::ct::
+-- ct_ops.cuh       -- CT primitives: value_barrier (PTX asm), masks, cmov, cswap
+-- ct_field.cuh     -- CT field: add, sub, neg, mul, sqr, inv, half, cmov, cswap
+-- ct_scalar.cuh    -- CT scalar: add, sub, neg, half, mul, inverse (Fermat), GLV
+-- ct_point.cuh     -- CT point: dbl, add_mixed (Brier-Joye 7M+5S), add (11M+6S),
|                       scalar_mul (GLV + bit-by-bit), generator_mul
+-- ct_sign.cuh      -- CT signing: ct_ecdsa_sign, ct_schnorr_sign, ct_schnorr_keypair
```

The GPU CT layer mirrors the CPU CT layer with identical algorithms adapted for CUDA:
- `value_barrier()` uses PTX `asm volatile` to prevent compiler optimization
- All mask operations are 64-bit (matching GPU's native word size)
- No branch divergence on secret data (critical for SIMT warp execution)
- Field/scalar heavy arithmetic delegates to fast-path (same cost) with CT
  control flow wrapping

#### GPU CT Usage

```cuda
#include "ct/ct_sign.cuh"

__global__ void sign_kernel(const uint8_t* msg, const Scalar* privkey,
                            ECDSASignatureGPU* sig, bool* ok) {
    // CT ECDSA sign -- constant-time k*G, k^-1, scalar ops
    *ok = secp256k1::cuda::ct::ct_ecdsa_sign(msg, privkey, sig);
}

__global__ void schnorr_kernel(const Scalar* privkey, const uint8_t* msg,
                               const uint8_t* aux, SchnorrSignatureGPU* sig, bool* ok) {
    // CT Schnorr sign -- constant-time nonce generation + signing
    *ok = secp256k1::cuda::ct::ct_schnorr_sign(privkey, msg, aux, sig);
}
```

#### GPU CT Benchmark Results (RTX 5060 Ti, SM 12.0)

| Operation | FAST | CT | CT/FAST Overhead |
|-----------|------|-----|------------------|
| k*G (generator) | 129.1 ns | 341.9 ns | 2.65x |
| k*P (scalar mul) | -- | 347.2 ns | -- |
| ECDSA sign | 211.1 ns | 433.9 ns | 2.06x |
| Schnorr sign | 284.9 ns | 715.8 ns | 2.51x |

GPU CT throughput: **2.30M ECDSA sign/sec**, **1.40M Schnorr sign/sec**.

---

## CT Guarantees

### What IS Constant-Time

| Operation | Implementation | Guarantee Level |
|-----------|---------------|-----------------|
| `ct::scalar_mul(P, k)` | GLV + signed-digit, fixed iteration count | Strong |
| `ct::generator_mul(k)` | Hamburg comb, precomputed table | Strong |
| `ct::field_mul` | Same arithmetic as FAST, no early-exit | Strong |
| `ct::field_inv` | Fixed iteration SafeGCD or exponentiation chain | Strong |
| `ct::point_add_complete` | Complete addition formula (handles all cases) | Strong |
| `ct::point_dbl` | No identity check branching | Strong |
| `field_select(a, b, flag)` | Bitwise masking: `(a & mask) \| (b & ~mask)` | Strong |
| ECDSA nonce (RFC 6979) | Deterministic, CT HMAC-DRBG | Strong |

### What Is NOT Constant-Time

| Operation | Why | Risk |
|-----------|-----|------|
| `fast::scalar_mul` | Window-NAF with variable-length representation | Timing leak on scalar bits |
| `fast::field_inverse` | Variable-time SafeGCD (divsteps exit early) | Leak on field element value |
| `fast::point_add` | Short-circuits on infinity | Leak on point identity |
| GPU kernels (all) | SIMT execution model, shared memory | Observable via GPU profiling |
| FROST / MuSig2 | Experimental, not CT-audited | Unknown |

---

## CT Primitive: Constant-Time Select

The fundamental building block of CT operations:

```cpp
// cpu/include/secp256k1/field_branchless.hpp

inline FieldElement field_select(const FieldElement& a,
                                  const FieldElement& b,
                                  bool flag) noexcept {
    // Convert bool to all-1s or all-0s mask (branchless)
    std::uint64_t mask = -static_cast<std::uint64_t>(flag);

    const auto& a_limbs = a.limbs();
    const auto& b_limbs = b.limbs();

    return FieldElement::from_limbs({
        (a_limbs[0] & mask) | (b_limbs[0] & ~mask),
        (a_limbs[1] & mask) | (b_limbs[1] & ~mask),
        (a_limbs[2] & mask) | (b_limbs[2] & ~mask),
        (a_limbs[3] & mask) | (b_limbs[3] & ~mask)
    });
}
```

**Audit points**:
1. `bool -> uint64_t mask` must not be compiled to a branch
2. Both paths of `from_limbs` must execute (no short-circuit)
3. Compiler must not optimize away the unused path

---

## CT Scalar Multiplication Details

### `ct::scalar_mul(P, k)` -- Arbitrary Point

```
Algorithm: GLV + 5-bit signed encoding

1. Transform: s = (k + K) / 2  (K = group order bias)
2. GLV split: s -> v1, v2 (each ~129 bits)
3. Recode v1, v2 into 26 groups of 5-bit signed odd digits
   -> every digit is guaranteed non-zero and odd
4. Precompute table: 16 odd multiples of P and lambdaP
   T = [1P, 3P, 5P, ..., 31P, 1lambdaP, 3lambdaP, ..., 31lambdaP]
5. Fixed iteration: for i = 25 downto 0:
   a. 5 x point_double (CT)
   b. lookup T[|v1[i]|] with CT table scan (touch all entries)
   c. conditional negate based on sign bit (CT)
   d. unified_add (CT complete formula)
   e. repeat for v2[i]

Cost: 125 dbl + 52 unified_add + 52 signed_lookups(16)
All iterations execute regardless of scalar value.
```

### `ct::generator_mul(k)` -- Generator Point

```
Algorithm: Hamburg signed-digit comb

1. v = (k + 2^256 - 1) / 2 mod n
2. Every 4-bit window yields guaranteed odd digit
3. Precomputed table: 8 entries per window (generated at init)
4. 64 iterations:
   a. CT table lookup(8) -- scan all entries
   b. conditional negate based on sign bit (CT)
   c. unified_add (CT)
5. No doublings needed (comb structure)

Cost: 64 unified_add + 64 signed_lookups(8)
~3x faster than ct::scalar_mul(G, k)
```

---

## Timing Verification: dudect Methodology

### Implementation

File: `tests/test_ct_sidechannel.cpp` (1300+ lines)

Uses the dudect approach (Reparaz, Balasch, Verbauwhede, 2017):

```
1. Two classes of inputs:
   - Class 0: Edge-case values (zero, one, identity, max)
   - Class 1: Random pre-generated values

2. For each function under test:
   a. Pre-generate N input pairs (class 0 and class 1)
   b. Random class assignment per measurement
   c. Array-based class selection (constant-cost lookup)
   d. rdtsc/cntvct timing of the operation
   e. Collect timing distributions

3. Statistical test: Welch's t-test
   - |t| < 4.5 -> no detectable timing difference (PASS)
   - |t| >= 4.5 -> timing leak detected (FAIL, 99.999% confidence)

4. Timing barriers: asm volatile prevents reordering
```

### Functions Tested

| Function | Class 0 (edge) | Class 1 (random) |
|----------|----------------|-------------------|
| `ct::field_mul` | Zero, One | Random field elements |
| `ct::field_inv` | One | Random field elements |
| `ct::field_square` | Zero | Random field elements |
| `ct::scalar_mul` | Small scalars | Random 256-bit scalars |
| `ct::generator_mul` | One, Two | Random 256-bit scalars |
| `ct::point_add` | Identity + P | Random points |
| `field_select` | flag=0, flag=1 | Random flags |
| ECDSA sign | Known keys | Random keys |
| Schnorr sign | Known keys | Random keys |

### Running the Test

```bash
# Direct execution (recommended)
./build/tests/test_ct_sidechannel

# Under Valgrind (checks memory access patterns)
valgrind ./build/tests/test_ct_sidechannel_vg

# Interpretation:
# |t| < 4.5 for all operations -> PASS
# Current result: timing variance ratio 1.035 (well below 1.2 concern threshold)
```

---

## Known Limitations

### 1. Formal Verification (Partial)

The CT layer is verified using:
- **ct-verif LLVM pass** -- deterministic compile-time CT check of `ct_field.cpp`, `ct_scalar.cpp`, `ct_sign.cpp` (`.github/workflows/ct-verif.yml`). If the LLVM pass is unavailable, a fallback IR branch analysis runs.

Not yet integrated:
- **Vale** (F\* verified assembly)
- **Fiat-Crypto** (formally verified field arithmetic)
- **Cryptol/SAW** (symbolic analysis)

Additional CT guarantees come from:
- Manual code review
- Compiler discipline (`-O2` specifically)
- dudect empirical testing (x86-64 + ARM64 native)
- ASan/UBSan runtime checks

### 2. Compiler Risk

Compilers may break CT properties by:
- Converting bitwise cmov to branches for "optimization"
- Eliminating "dead" computation paths
- Auto-vectorizing with data-dependent masking
- Different behavior at `-O3` vs `-O2`

**Mitigation**: The project uses `asm volatile` barriers and recommends `-O2` for production CT builds. Higher optimization levels should be validated with dudect.

### 3. Microarchitecture Variability

CT properties verified on one CPU may not hold on another:
- Intel vs AMD vs ARM have different timing behaviors
- Variable-latency multipliers on some uarch
- Cache hierarchy differences

**Status**: Tested on x86-64 (Intel/AMD) and ARM64 (Apple M1 native). Multi-uarch dudect coverage:
- x86-64: CI runners (ubuntu-24.04) -- every push/PR
- ARM64: Apple Silicon M1 (macos-14) -- smoke per-PR, full nightly (`.github/workflows/ct-arm64.yml`)
- ARM64: cross-compiled via aarch64-linux-gnu-g++-13 (compile check only)

### 4. GPU CT Guarantees

The GPU CT layer (`secp256k1::cuda::ct::`) provides **algorithmic** constant-time
guarantees: no secret-dependent branches, no secret-dependent memory access patterns,
fixed iteration counts.

**What GPU CT protects against:**
- Software-level timing attacks from co-located GPU workloads
- Branch divergence leaking scalar bits within a warp
- Memory access pattern analysis via GPU profiling tools

**What GPU CT does NOT protect against:**
- Hardware-level electromagnetic or power analysis
- GPU shared memory bank conflict timing (microarchitectural)
- Driver-level scheduling observation
- Physical side-channels requiring oscilloscope-level measurements

The GPU CT layer is tested via `test_ct_smoke` (7 functional tests) and integrated
into the GPU audit runner (Section S6: CT Analysis). See the GPU audit section below.

### 5. Experimental Protocols

FROST and MuSig2 have NOT been CT-audited:
- Multi-party protocol simulation needed
- Nonce handling under review
- API instability prevents thorough CT analysis

---

## CT Audit Checklist for Reviewers

- [ ] **field_select**: Verify `-static_cast<uint64_t>(flag)` produces all-1s/all-0s
- [ ] **field_select**: Confirm compiler emits no branch (inspect assembly)
- [ ] **ct::scalar_mul**: Fixed iteration count (26 groups x 5 doublings + 52 adds)
- [ ] **ct::scalar_mul**: Table lookup scans ALL entries (no early-exit)
- [ ] **ct::generator_mul**: Fixed 64 iterations, no conditional skip
- [ ] **ct::point_add_complete**: Handles P+P, P+O, O+P, P+(-P) without branching
- [ ] **ct::field_inv**: Fixed exponentiation chain length (no variable-time SafeGCD)
- [ ] **ECDSA nonce**: RFC 6979 HMAC-DRBG is CT (no secret-dependent branches)
- [ ] **Schnorr nonce**: BIP-340 tagged hash is CT
- [ ] **No early return**: grep for `if (is_zero())` or `if (is_infinity())` in CT path
- [ ] **No array indexing by secret**: all lookups use linear scan + cmov
- [ ] **asm volatile barriers**: present around timing-sensitive sections
- [ ] **dudect passes**: |t| < 4.5 for all tested functions

---

## Planned Improvements

- [ ] **Formal verification** with Fiat-Crypto for field arithmetic
- [x] **ct-verif** LLVM pass integration for CT verification (`.github/workflows/ct-verif.yml`)
- [x] **Multi-uarch dudect** -- x86-64 CI + ARM64 Apple M1 native (`.github/workflows/ct-arm64.yml`)
- [x] **dudect expansion** to cover FROST/MuSig2 -- `musig2_partial_sign`, `frost_sign`, `frost_lagrange_coefficient`
- [x] **Valgrind CT taint** in CI -- MAKE_MEM_UNDEFINED + --track-origins (`.github/workflows/valgrind-ct.yml`)
- [ ] **Hardware timing analysis** with oscilloscope-level measurements
- [ ] **Compiler output audit** for every release at `-O2` and `-O3`

---

## References

- [dudect: dude, is my code constant time?](https://eprint.iacr.org/2016/1123) -- Reparaz et al., 2017
- [Timing-safe code: A guide for the rest of us](https://www.chosenplaintext.ca/open-source/dudect/) -- Aumasson
- [ct-verif: A Tool for Constant-Time Verification](https://github.com/imdea-software/verifying-constant-time) -- IMDEA
- [Fiat-Crypto: Proofs of Correctness of ECC](https://github.com/mit-plv/fiat-crypto) -- MIT
- [bitcoin-core/secp256k1](https://github.com/bitcoin-core/secp256k1) -- Reference CT implementation

---

*UltrafastSecp256k1 v3.16.0 -- CT Verification*
