# Constant-Time Verification

**UltrafastSecp256k1 v3.68.0** -- CT Layer Methodology & Audit Status

---

## Overview

The constant-time (CT) layer provides side-channel resistant operations for secret key material. It is available on **all backends**:

- **CPU**: `secp256k1::ct::` namespace (headers in `cpu/include/secp256k1/ct/`)
- **CUDA GPU**: `secp256k1::cuda::ct::` namespace (headers in `cuda/include/ct/`)
- **OpenCL GPU**: CT kernels in `opencl/kernels/` (`secp256k1_ct_sign.cl`, `secp256k1_ct_zk.cl`)
- **Metal GPU**: CT shaders in `metal/shaders/` (`secp256k1_ct_sign.metal`, `secp256k1_ct_zk.metal`)

The FAST layer (`secp256k1::fast::` on CPU, `secp256k1::cuda::` on GPU) is explicitly variable-time for maximum throughput.

**Principle**: Any operation that touches secret data (private keys, nonces, intermediate scalars) MUST use `ct::` functions on CPU. GPU operations that accept secret keys (`ecdh_batch`, `bip352_scan_batch`, `bip324_aead_*_batch`) require a trusted single-tenant environment. The default `fast::` namespace is allowed only when all inputs are public.

The repository preflight also consumes graph-linked coverage metadata from
`scripts/build_project_graph.py`. That metadata records both standalone CTest
coverage and selected unified-audit module coverage for core files; it is used
for coverage-gap reporting and does not replace the executable CT tests.

CT secret-bearing implementation changes are under stricter change control:
`scripts/check_secret_path_changes.py` requires paired updates to this document
and `docs/SECURITY_CLAIMS.md` whenever CT-layer secret surfaces change.

---

## CT-Routed C ABI Surface

The audit gate treats the graph's `abi_routing` table as the routing source of
truth for secret-bearing C ABI entry points. The functions below are the
currently documented CT-routed ABI surface together with the routed internal
operation recorded by the graph.

| C ABI Function | Routed Internal Operation |
|----------------|---------------------------|
| `ufsecp_bip32_derive` | `CKD_priv or CKD_pub` |
| `ufsecp_bip32_derive_path` | `multi-level CKD` |
| `ufsecp_bip32_master` | `HMAC-SHA512(seed)` |
| `ufsecp_bip32_privkey` | `ExtendedKey::privkey()` |
| `ufsecp_bip39_generate` | `bip39_generate(strength)` |
| `ufsecp_bip39_to_seed` | `PBKDF2-SHA512(mnemonic, passphrase)` |
| `ufsecp_btc_message_sign` | `btc_message_sign(msg, sk)` |
| `ufsecp_coin_hd_derive` | `coin_hd_derive(coin, xprv, path)` |
| `ufsecp_ecdh` | `ct::scalar_mul(pubkey, sk)` |
| `ufsecp_ecdh_raw` | `ct::scalar_mul + raw output` |
| `ufsecp_ecdh_xonly` | `ct::scalar_mul + x-only output` |
| `ufsecp_ecdsa_adaptor_adapt` | `ecdsa_adaptor_adapt` |
| `ufsecp_ecdsa_adaptor_sign` | `ct::ecdsa_adaptor_sign(sk)` |
| `ufsecp_ecdsa_sign` | `ct::ecdsa_sign(msg, sk)` |
| `ufsecp_ecdsa_sign_recoverable` | `ct::ecdsa_sign + recovery_id` |
| `ufsecp_ecdsa_sign_verified` | `ct::ecdsa_sign + ecdsa_verify` |
| `ufsecp_ecies_decrypt` | `ecies_decrypt(sk, ciphertext)` |
| `ufsecp_ecies_encrypt` | `ecies_encrypt(pubkey, msg)` |
| `ufsecp_eth_sign` | `ct::ecdsa_sign(keccak(msg), sk) + v` |
| `ufsecp_frost_keygen_begin` | `frost_keygen_begin` |
| `ufsecp_frost_sign` | `ct::frost_sign(sk, nonce)` |
| `ufsecp_frost_sign_nonce_gen` | `frost_sign_nonce_gen` |
| `ufsecp_musig2_nonce_gen` | `musig2_nonce_gen(sk)` |
| `ufsecp_musig2_partial_sign` | `ct::musig2_partial_sign(sk)` |
| `ufsecp_musig2_start_sign_session` | `musig2_session_init` |
| `ufsecp_pedersen_blind_sum` | `blind factor sum` |
| `ufsecp_pubkey_create` | `ct::generator_mul(sk)` |
| `ufsecp_pubkey_create_uncompressed` | `ct::generator_mul(sk)` |
| `ufsecp_pubkey_xonly` | `schnorr_pubkey(sk)` |
| `ufsecp_schnorr_adaptor_adapt` | `adaptor_adapt(pre_sig, secret)` |
| `ufsecp_schnorr_adaptor_sign` | `ct::adaptor_sign(sk)` |
| `ufsecp_schnorr_keypair` | `generate_schnorr_keypair(sk)` |
| `ufsecp_schnorr_sign` | `ct::schnorr_sign(msg, sk)` |
| `ufsecp_schnorr_sign_verified` | `ct::schnorr_sign + schnorr_verify` |
| `ufsecp_seckey_negate` | `Scalar::negate` |
| `ufsecp_seckey_tweak_add` | `ct scalar add + validate` |
| `ufsecp_seckey_tweak_mul` | `ct scalar mul + validate` |
| `ufsecp_seckey_verify` | `Scalar::parse_bytes_strict_nonzero` |
| `ufsecp_silent_payment_create_output` | `silent_payment_create_output` |
| `ufsecp_silent_payment_scan` | `silent_payment_scan` |
| `ufsecp_taproot_tweak_seckey` | `taproot_tweak_seckey(sk, merkle)` |
| `ufsecp_zk_dleq_prove` | `dleq_prove(sk)` |
| `ufsecp_zk_knowledge_prove` | `prove_knowledge(sk)` |
| `ufsecp_zk_range_proof_create` | `create_range_proof` |

These entries are routing and review anchors, not stand-alone proof claims.
They document which exported secret-bearing APIs are expected to stay on the CT
path and which internal CT-sensitive primitive or wrapper currently implements
that route.

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

#### GPU CT ZK Layer

```
secp256k1::cuda::ct::
+-- ct_zk.cuh        -- CT ZK proving: knowledge proof (Schnorr sigma), DLEQ proof
                        Uses ct_scalar_mul for secret nonce operations, ct_jacobian_to_affine,
                        scalar_cneg for BIP-340 Y-parity normalization.
                        Deterministic nonce: SHA-256 tagged hash with XOR hedging.
```

The GPU CT ZK layer ensures that all proving operations (which handle secret keys
and nonces) use constant-time scalar multiplication and arithmetic. Verification
operations use the fast path since all inputs are public.

| CT ZK Operation | Approach | Secret Data Protected |
|-----------------|----------|----------------------|
| `ct_knowledge_prove_device` | CT `ct_scalar_mul` for k*B | Nonce k, secret key |
| `ct_knowledge_prove_generator_device` | CT `ct_scalar_mul` for k*G | Nonce k, secret key |
| `ct_dleq_prove_device` | 2x CT `ct_scalar_mul` for k*G, k*H | Nonce k, secret key |
| `knowledge_verify_device` | Fast-path `scalar_mul` | N/A (public data) |
| `dleq_verify_device` | Fast-path `scalar_mul` | N/A (public data) |

**Test coverage:** `test_ct_smoke.cu` tests 8-9 verify CT knowledge prove + verify and
CT DLEQ prove + verify round-trips on GPU. All 9/9 tests pass.

### OpenCL CT Layer

```
opencl/kernels/
+-- secp256k1_ct_sign.cl    -- CT ECDSA sign, CT Schnorr sign, CT keypair create
+-- secp256k1_ct_zk.cl      -- CT ZK proving: knowledge proof, DLEQ proof
```

The OpenCL CT layer mirrors the CUDA CT implementation with OpenCL-native barriers:
- `value_barrier()` via inline OpenCL `asm volatile` or volatile loads
- Branchless masks and conditional moves on all secret-dependent paths
- CT scalar multiplication with fixed iteration count (GLV + signed-digit)
- Audited via `opencl_audit_runner` (27 modules including CT sections)

### Metal CT Layer

```
metal/shaders/
+-- secp256k1_ct_sign.metal -- CT ECDSA sign, CT Schnorr sign, CT keypair create
+-- secp256k1_ct_zk.metal   -- CT ZK proving: knowledge proof, DLEQ proof
```

The Metal CT layer uses Metal Shading Language (MSL) with:
- `value_barrier()` via threadgroup memory fence pattern
- Identical algorithms to CUDA/OpenCL CT layers
- Audited via `metal_audit_runner` (27 modules including CT sections)

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
| Some FROST / MuSig2 helpers | Session setup, public-index math, and broader protocol orchestration are not all elevated to full secret-path CT claims | Experimental surface; secret-bearing `musig2_partial_sign` and `frost_sign` do have protocol-level dudect coverage |

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

File: `audit/test_ct_sidechannel.cpp` (1300+ lines)

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
| `musig2_partial_sign` | Fixed secret key | Random secret keys |
| `frost_sign` | Low-Hamming-weight signing share | High-Hamming-weight signing share |
| `frost_lagrange_coefficient` | Signer set `{1,2}` | Signer set `{1,3}` (advisory, public indices) |

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

### 1. Formal Verification (Expanded, Still Not End-to-End)

The CT layer is verified using:
- **ct-verif LLVM pass** -- deterministic compile-time CT check of `ct_field.cpp`, `ct_scalar.cpp`, `ct_point.cpp`, and `ct_sign.cpp` (`.github/workflows/ct-verif.yml`). If the LLVM pass is unavailable, a fallback IR branch analysis runs.
- **Valgrind CT taint analysis** -- `scripts/valgrind_ct_check.sh` marks private-key bytes as secret via `MAKE_MEM_UNDEFINED` / `--track-origins=yes` and runs signing + ECDH operations, failing on any secret-derived branch or memory access. Integrated in `.github/workflows/valgrind-ct.yml`.

The normalized evidence collector (`scripts/collect_ct_evidence.py --strict`) now treats the expected ct-verif module set as owner-grade blocking: `ct_field`, `ct_scalar`, `ct_point`, and `ct_sign`. A deterministic artifact that omits one of those modules is downgraded from usable proof material to a configured-only gap.

Not yet integrated:
- **Vale** (F\* verified assembly)
- **Fiat-Crypto** (formally verified field arithmetic)
- **Cryptol/SAW** (symbolic analysis)

Additional CT guarantees come from:
- Manual code review
- Compiler discipline (`-O2` specifically)
- dudect empirical testing (x86-64 + ARM64 native)
- ASan/UBSan runtime checks

Protocol-level nuance:
- `musig2_partial_sign` and `frost_sign` are exercised by the dudect suite in `audit/test_ct_sidechannel.cpp`
- `frost_lagrange_coefficient` is tracked as advisory timing only because it operates on public participant indices
- This timing evidence improves confidence for secret-bearing signing steps, but does not by itself promote the whole MuSig2/FROST protocol stack out of experimental status

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

The GPU CT layers (CUDA `secp256k1::cuda::ct::`, OpenCL `secp256k1_ct_sign.cl`/`secp256k1_ct_zk.cl`,
Metal `secp256k1_ct_sign.metal`/`secp256k1_ct_zk.metal`) provide **algorithmic** constant-time
guarantees: no secret-dependent branches, no secret-dependent memory access patterns,
fixed iteration counts. All three GPU backends implement identical CT algorithms.

**What GPU CT protects against:**
- Software-level timing attacks from co-located GPU workloads
- Branch divergence leaking scalar bits within a warp/wavefront/threadgroup
- Memory access pattern analysis via GPU profiling tools

**What GPU CT does NOT protect against:**
- Hardware-level electromagnetic or power analysis
- GPU shared memory bank conflict timing (microarchitectural)
- Driver-level scheduling observation
- Physical side-channels requiring oscilloscope-level measurements

The GPU CT layers are tested via:
- **CUDA**: `test_ct_smoke` (9 functional tests) + GPU audit runner (Section S6: CT Analysis)
- **OpenCL**: `opencl_audit_runner` (27 modules including CT signing + CT ZK sections)
- **Metal**: `metal_audit_runner` (27 modules including CT signing + CT ZK sections)

### 5. Experimental Protocols

FROST and MuSig2 remain broader experimental protocol surfaces, but the repo no longer treats them as completely unaudited:
- Secret-bearing `musig2_partial_sign` and `frost_sign` have protocol-level dudect coverage in `audit/test_ct_sidechannel.cpp`
- `frost_lagrange_coefficient` is tracked as advisory timing-only because it operates on public participant indices
- Multi-party orchestration, session setup, and misuse-boundary analysis still need deeper protocol-grade review before the full stacks can be promoted to strong CT claims

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
- [ ] **ct::scalar_inverse(0) guard**: both SafeGCD and Fermat CT inverse paths return `Scalar::zero()` for zero input (defense-in-depth, not a timing exit — the zero check is on the input, not on secret-derived data; verified by `test_exploit_boundary_sentinels` BS-1, BS-10)
- [ ] **No array indexing by secret**: all lookups use linear scan + cmov
- [ ] **asm volatile barriers**: present around timing-sensitive sections
- [ ] **dudect passes**: |t| < 4.5 for all tested functions

---

## Planned Improvements

- [ ] **Formal verification** with Fiat-Crypto for field arithmetic
- [x] **ct-verif** LLVM pass integration for CT verification (`.github/workflows/ct-verif.yml`)
- [x] **ct-verif: ct_point.cpp** -- point operations are included in `.github/workflows/ct-verif.yml`, and `scripts/collect_ct_evidence.py --strict` treats missing point-module evidence as owner-grade blocking
- [x] **Multi-uarch dudect** -- x86-64 CI + ARM64 Apple M1 native (`.github/workflows/ct-arm64.yml`)
- [x] **dudect expansion** to cover FROST/MuSig2 -- `musig2_partial_sign`, `frost_sign`, `frost_lagrange_coefficient`
- [x] **Valgrind CT taint** in CI -- MAKE_MEM_UNDEFINED + --track-origins (`scripts/valgrind_ct_check.sh`, `.github/workflows/valgrind-ct.yml`)
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

*UltrafastSecp256k1 v3.68.0 -- CT Verification*
