# Audit Traceability Matrix

**UltrafastSecp256k1 v3.14.0** -- Evidence-Based Correctness & Security Mapping

> This document maps every mathematical invariant to its implementation code,
> validation method, and specific test location. It is the primary artifact for
> auditors to verify that all claimed guarantees have corresponding evidence.

---

## Methodology

Each row in this matrix links:
1. **Invariant ID** -- from [INVARIANTS.md](INVARIANTS.md) (108 total)
2. **Mathematical Claim** -- the exact property guaranteed
3. **Implementation** -- source file(s) implementing the primitive
4. **Validation Method** -- how it is verified (deterministic, statistical, differential)
5. **Test Location** -- exact file and function/line where evidence is produced
6. **Status** -- [OK] Verified | [!] Partial | [FAIL] Gap

---

## 1. Field Arithmetic ($\mathbb{F}_p$, $p = 2^{256} - 2^{32} - 977$)

| ID | Invariant | Implementation | Validation | Test Location | Status |
|----|-----------|---------------|------------|---------------|--------|
| **F1** | $\text{normalize}(a) \in [0, p)$ | `cpu/field.hpp` | Canonical serialization check (10K random) | `audit_field.cpp` -> `test_canonical()` | [OK] |
| **F2** | $a + b \equiv (a + b) \bmod p$ | `cpu/field.hpp` | Commutativity + associativity + overflow (3K random) | `audit_field.cpp` -> `test_addition_overflow()` | [OK] |
| **F3** | $a - b \equiv (a - b + p) \bmod p$ | `cpu/field.hpp` | Borrow-chain, $0 - a = -a$ (3K random) | `audit_field.cpp` -> `test_subtraction_borrow()` | [OK] |
| **F4** | $a \cdot b \equiv (a \cdot b) \bmod p$ | `cpu/field.hpp` | Commutativity + associativity + distributivity (5K random) | `audit_field.cpp` -> `test_mul_carry()` | [OK] |
| **F5** | $a^2 = a \cdot a$ | `cpu/field.hpp` | Square vs mul equivalence (10K random) | `audit_field.cpp` -> `test_square_vs_mul()` | [OK] |
| **F6** | $a \cdot a^{-1} \equiv 1 \bmod p$ for $a \neq 0$ | `cpu/field.hpp` | Inverse correctness + double inverse (11K random) | `audit_field.cpp` -> `test_inverse()` | [OK] |
| **F7** | $\text{inv}(0)$ is undefined / returns zero | `cpu/field.hpp` | Exception/zero-return check | `audit_security.cpp` -> `test_zero_key_handling()` | [OK] |
| **F8** | $\sqrt{a}^2 = a$ when $a$ is QR | `cpu/field.hpp` | Square root correctness (10K random, ~50.72% QR) | `audit_field.cpp` -> `test_sqrt()` | [OK] |
| **F9** | $\sqrt{a}$ returns nullopt for QNR | `cpu/field.hpp` | Implicit (non-QR returns +-x mismatch) | `audit_field.cpp` -> `test_sqrt()` | [OK] |
| **F10** | $-a + a \equiv 0 \bmod p$ | `cpu/field.hpp` | Negate + add to zero (1K random) | `audit_field.cpp` -> `test_addition_overflow()` | [OK] |
| **F11** | `from_bytes(to_bytes(a)) == a` | `cpu/field.hpp` | Serialization round-trip (1K random) | `audit_field.cpp` -> `test_reduction()` | [OK] |
| **F12** | `from_limbs` = little-endian uint64[4] | `cpu/field.hpp` | Endianness conformance | `audit_field.cpp` -> `test_limb_boundary()` | [OK] |
| **F13** | `from_bytes` = big-endian 32 bytes | `cpu/field.hpp` | Known vector: $\text{from\_bytes}(p) = 0$ | `audit_field.cpp` -> `test_reduction()` | [OK] |
| **F14** | Commutativity: $a+b = b+a$, $a \cdot b = b \cdot a$ | `cpu/field.hpp` | Random stress (2K) | `audit_field.cpp` -> `test_addition_overflow()`, `test_mul_carry()` | [OK] |
| **F15** | Associativity: $(a+b)+c = a+(b+c)$ | `cpu/field.hpp` | Random stress (1K) | `audit_field.cpp` -> `test_addition_overflow()` | [OK] |
| **F16** | Distributivity: $a(b+c) = ab + ac$ | `cpu/field.hpp` | Random stress (1K) | `audit_field.cpp` -> `test_mul_carry()` | [OK] |
| **F17** | `field_select` branchless: $\text{sel}(0,a,b)=a$, $\text{sel}(1,a,b)=b$ | `cpu/ct/ops.hpp` | Functional correctness | `audit_ct.cpp` -> `test_ct_cmov_cswap()` | [OK] |

**Field Subtotal: 17/17 [OK]**

---

## 2. Scalar Arithmetic ($\mathbb{Z}_n$, $n = $ order of secp256k1)

| ID | Invariant | Implementation | Validation | Test Location | Status |
|----|-----------|---------------|------------|---------------|--------|
| **S1** | $a + b \equiv (a + b) \bmod n$ | `cpu/scalar.hpp` | Commutativity + associativity (10K random) | `audit_scalar.cpp` -> `test_scalar_laws()` | [OK] |
| **S2** | $a - b \equiv (a - b + n) \bmod n$ | `cpu/scalar.hpp` | Edge cases + random | `audit_scalar.cpp` -> `test_edge_scalars()` | [OK] |
| **S3** | $a \cdot b \equiv (a \cdot b) \bmod n$ | `cpu/scalar.hpp` | Commutativity + associativity + distributivity (10K) | `audit_scalar.cpp` -> `test_scalar_laws()` | [OK] |
| **S4** | $a \cdot a^{-1} \equiv 1 \bmod n$ for $a \neq 0$ | `cpu/scalar.hpp` | Inverse + double inverse (11K random) | `audit_scalar.cpp` -> `test_scalar_inverse()` | [OK] |
| **S5** | $-a + a \equiv 0 \bmod n$ | `cpu/scalar.hpp` | Negate self-consistency (10K) | `audit_scalar.cpp` -> `test_negate()` | [OK] |
| **S6** | `is_zero(0) == true` | `cpu/scalar.hpp` | Direct check | `audit_scalar.cpp` -> `test_edge_scalars()` | [OK] |
| **S7** | `is_zero(1) == false` | `cpu/scalar.hpp` | Direct check | `audit_scalar.cpp` -> `test_edge_scalars()` | [OK] |
| **S8** | `normalize(a)` yields $0 \leq a < n$ | `cpu/scalar.hpp` | Overflow normalization (10K random) | `audit_scalar.cpp` -> `test_overflow_normalization()` | [OK] |
| **S9** | Low-S: if $s > n/2$, replace with $n - s$ | `cpu/ecdsa.hpp` | High-S detection + normalization (1K) | `audit_security.cpp` -> `test_high_s_rejection()` | [OK] |

**Scalar Subtotal: 9/9 [OK]**

---

## 3. Point / Group Invariants (secp256k1 curve)

| ID | Invariant | Implementation | Validation | Test Location | Status |
|----|-----------|---------------|------------|---------------|--------|
| **P1** | $G$ on curve: $G_y^2 = G_x^3 + 7 \bmod p$ | `cpu/point.hpp` | On-curve check (100K random points) | `audit_point.cpp` -> `test_stress_random()` | [OK] |
| **P2** | $n \cdot G = \mathcal{O}$ | `cpu/point.hpp` | Direct computation | `audit_point.cpp` -> `test_infinity()` | [OK] |
| **P3** | $P + \mathcal{O} = P$ | `cpu/point.hpp` | Identity element | `audit_point.cpp` -> `test_infinity()` | [OK] |
| **P4** | $P + (-P) = \mathcal{O}$ | `cpu/point.hpp` | Inverse cancellation (1K random) | `audit_point.cpp` -> `test_point_negation()` | [OK] |
| **P5** | $(P+Q)+R = P+(Q+R)$ | `cpu/point.hpp` | Associativity (500 random triples) | `audit_point.cpp` -> `test_jacobian_add()` | [OK] |
| **P6** | $P + Q = Q + P$ | `cpu/point.hpp` | Commutativity (1K random) | `audit_point.cpp` -> `test_jacobian_add()` | [OK] |
| **P7** | $k(P+Q) = kP + kQ$ | `cpu/point.hpp` | Distributivity | `test_ecc_properties.cpp` -> `test_distributivity()` | [OK] |
| **P8** | $(a+b) \cdot G = aG + bG$ | `cpu/point.hpp` | Scalar addition homomorphism (1K) | `audit_point.cpp` -> `test_scalar_mul_identities()` | [OK] |
| **P9** | $(ab) \cdot G = a(bG)$ | `cpu/point.hpp` | Scalar multiplication (500) | `audit_point.cpp` -> `test_scalar_mul_identities()` | [OK] |
| **P10** | `to_affine(to_jacobian(P)) == P` | `cpu/point.hpp` | Round-trip (1K) | `test_ecc_properties.cpp` -> `test_jacobian_affine_roundtrip()` | [OK] |
| **P11** | Jacobian add == Affine add | `cpu/point.hpp` | Consistency | `test_ecc_properties.cpp` | [OK] |
| **P12** | $\text{dbl}(P) = P + P$ | `cpu/point.hpp` | Double vs add (chain of 10 dbls = 1024*G) | `audit_point.cpp` -> `test_jacobian_dbl()` | [OK] |
| **P13** | $\forall P: P_y^2 = P_x^3 + 7$ | `cpu/point.hpp` | On-curve stress (100K) | `audit_point.cpp` -> `test_stress_random()` | [OK] |
| **P14** | `deserialize(serialize(P)) == P` | `cpu/point.hpp` | Compressed + uncompressed (1K) | `audit_point.cpp` -> `test_affine_conversion()` | [OK] |

**Point Subtotal: 14/14 [OK]**

---

## 4. GLV Endomorphism

| ID | Invariant | Implementation | Validation | Test Location | Status |
|----|-----------|---------------|------------|---------------|--------|
| **G1** | $\phi(P) = \lambda \cdot P$, $\lambda^3 \equiv 1 \bmod n$ | `cpu/glv.hpp` | Algebraic point verification | `audit_scalar.cpp` -> `test_glv_split()` | [OK] |
| **G2** | $\phi(\phi(P)) + \phi(P) + P = \mathcal{O}$ | `cpu/glv.hpp` | Endomorphism relation | Comprehensive test #22 | [OK] |
| **G3** | $k \equiv k_1 + k_2 \lambda \bmod n$ | `cpu/glv.hpp` | Decomposition algebraic check | `audit_scalar.cpp` -> `test_glv_split()` | [OK] |
| **G4** | $|k_1|, |k_2| < \sqrt{n}$ | `cpu/glv.hpp` | Balanced split | Comprehensive test #22 | [OK] |

**GLV Subtotal: 4/4 [OK]**

---

## 5. ECDSA (RFC 6979)

| ID | Invariant | Implementation | Validation | Test Location | Status |
|----|-----------|---------------|------------|---------------|--------|
| **E1** | `verify(msg, sign(msg, sk), pk) == true` | `cpu/ecdsa.hpp` | Sign+verify round-trip (1K random) + official vectors | `audit_point.cpp` -> `test_ecdsa_roundtrip()`, `test_rfc6979_vectors.cpp` | [OK] |
| **E2** | Deterministic nonce (same msg+sk -> same sig) | `cpu/ecdsa.hpp` | 6 official RFC 6979 nonce vectors | `test_rfc6979_vectors.cpp` | [OK] |
| **E3** | $r \in [1, n-1]$, $s \in [1, n-1]$ | `cpu/ecdsa.hpp` | Non-zero sig check (1K) | `audit_point.cpp` -> `test_ecdsa_roundtrip()` | [OK] |
| **E4** | Low-S enforced: $s \leq n/2$ | `cpu/ecdsa.hpp` | `is_low_s()` check + high-S rejection | `audit_security.cpp` -> `test_high_s_rejection()` | [OK] |
| **E5** | DER encoding round-trip | `cpu/ecdsa.hpp` | Parse -> serialize -> parse | `test_fuzz_parsers.cpp` suites 1-3 | [OK] |
| **E6** | Sign with $sk = 0$ or $sk \geq n$ -> failure | `cpu/ecdsa.hpp` | Zero/overflow key rejection | `audit_security.cpp` -> `test_zero_key_handling()` | [OK] |
| **E7** | Verify with wrong message -> false | `cpu/ecdsa.hpp` | Message bit-flip (1K) | `audit_point.cpp` -> `test_ecdsa_roundtrip()` | [OK] |
| **E8** | Verify with wrong pubkey -> false | `cpu/ecdsa.hpp` | Wrong-key rejection (1K) | `audit_point.cpp` -> `test_ecdsa_roundtrip()` | [OK] |

**ECDSA Subtotal: 8/8 [OK]**

---

## 6. Schnorr / BIP-340

| ID | Invariant | Implementation | Validation | Test Location | Status |
|----|-----------|---------------|------------|---------------|--------|
| **B1** | BIP-340 sign+verify round-trip | `cpu/schnorr.hpp` | 1K random round-trips | `audit_point.cpp` -> `test_schnorr_roundtrip()` | [OK] |
| **B2** | All 15 official test vectors | `cpu/schnorr.hpp` | v0-v3 sign + v4-v14 verify | `test_bip340_vectors.cpp` | [OK] |
| **B3** | Signature = 64 bytes $(R_x \| s)$ | `cpu/schnorr.hpp` | Format validation | `test_bip340_vectors.cpp` | [OK] |
| **B4** | $R$ has even y-coordinate | `cpu/schnorr.hpp` | Parity check in vectors | `test_bip340_vectors.cpp` | [OK] |
| **B5** | Public key is x-only (32 bytes) | `cpu/schnorr.hpp` | X-only format | `test_bip340_vectors.cpp` | [OK] |
| **B6** | Sign with $sk = 0$ -> failure | `cpu/schnorr.hpp` | Edge case | `test_fuzz_address_bip32_ffi.cpp` | [OK] |

**Schnorr Subtotal: 6/6 [OK]**

---

## 7. MuSig2 (BIP-327)

| ID | Invariant | Implementation | Validation | Test Location | Status |
|----|-----------|---------------|------------|---------------|--------|
| **M1** | Aggregated sig verifies as BIP-340 | `cpu/musig2.hpp` | Multi-party simulation | `test_musig2_frost.cpp` suites 1-6 | [OK] |
| **M2** | Key aggregation deterministic | `cpu/musig2.hpp` | Same-input reproducibility | `test_musig2_frost.cpp` | [OK] |
| **M3** | Nonce aggregation deterministic | `cpu/musig2.hpp` | Same-input reproducibility | `test_musig2_frost.cpp` | [OK] |
| **M4** | 2/3/5-of-N signing | `cpu/musig2.hpp` | Multi-threshold simulation | `test_musig2_frost.cpp` suites 4-6 | [OK] |
| **M5** | Invalid partial sig detected | `cpu/musig2.hpp` | Fault injection | `test_musig2_frost_advanced.cpp` suite 5 | [OK] |
| **M6** | Rogue-key attack detected | `cpu/musig2.hpp` | Wagner-style simulation | `test_musig2_frost_advanced.cpp` suites 1-2 | [OK] |
| **M7** | Nonce reuse detected | `cpu/musig2.hpp` | Cross-message detection | `test_musig2_frost_advanced.cpp` suites 3-4 | [OK] |

**MuSig2 Subtotal: 7/7 [OK]**

---

## 8. FROST Threshold Signatures

| ID | Invariant | Implementation | Validation | Test Location | Status |
|----|-----------|---------------|------------|---------------|--------|
| **FR1** | t-of-n DKG consistent group pubkey | `cpu/frost.hpp` | 2-of-3, 3-of-5 DKG | `test_musig2_frost.cpp` suites 7, 9 | [OK] |
| **FR2** | Shamir reconstruction: $\sum \lambda_i s_i = s$ | `cpu/frost.hpp` | Lagrange reconstruction | `test_musig2_frost.cpp` | [OK] |
| **FR3** | Aggregated sig verifies as BIP-340 | `cpu/frost.hpp` | Signing round-trip | `test_musig2_frost.cpp` suites 8, 10-11 | [OK] |
| **FR4** | 2-of-3 with any 2 signers | `cpu/frost.hpp` | Combinatorial test | `test_musig2_frost.cpp` | [OK] |
| **FR5** | 3-of-5 with any 3 signers | `cpu/frost.hpp` | Combinatorial test | `test_musig2_frost.cpp` | [OK] |
| **FR6** | Lagrange coefficients correct | `cpu/frost.hpp` | Secret reconstruction | `test_musig2_frost.cpp` | [OK] |
| **FR7** | Malicious DKG share detected | `cpu/frost.hpp` | Commitment verification | `test_musig2_frost_advanced.cpp` suites 6-7 | [OK] |
| **FR8** | Invalid partial sig detected | `cpu/frost.hpp` | Rejection test | `test_musig2_frost_advanced.cpp` | [OK] |
| **FR9** | Below-threshold subset fails | `cpu/frost.hpp` | 1-of-3 attempt -> fail | `test_musig2_frost_advanced.cpp` | [OK] |

**FROST Subtotal: 9/9 [OK]**

---

## 9. BIP-32 HD Derivation

| ID | Invariant | Implementation | Validation | Test Location | Status |
|----|-----------|---------------|------------|---------------|--------|
| **H1** | TV1-TV5 official vectors (90 checks) | `cpu/bip32.hpp` | Byte-exact comparison | `test_bip32_vectors.cpp` | [OK] |
| **H2** | `derive(master, "m") == master` | `cpu/bip32.hpp` | Identity derivation | `test_bip32_vectors.cpp` | [OK] |
| **H3** | Hardened derivation formula correct | `cpu/bip32.hpp` | Official vector conformance | `test_bip32_vectors.cpp` | [OK] |
| **H4** | Normal derivation formula correct | `cpu/bip32.hpp` | Official vector conformance | `test_bip32_vectors.cpp` | [OK] |
| **H5** | Path parser: valid/invalid paths | `cpu/bip32.hpp` | Fuzz testing | `test_fuzz_address_bip32_ffi.cpp` suites 5-7 | [OK] |
| **H6** | Seed length 16-64 bytes enforced | `cpu/bip32.hpp` | Boundary test | `test_fuzz_address_bip32_ffi.cpp` | [OK] |
| **H7** | Deterministic for same seed+path | `cpu/bip32.hpp` | Reproducibility | `test_bip32_vectors.cpp` | [OK] |

**BIP-32 Subtotal: 7/7 [OK]**

---

## 10. Address Generation

| ID | Invariant | Implementation | Validation | Test Location | Status |
|----|-----------|---------------|------------|---------------|--------|
| **A1** | P2PKH: `1...` prefix (mainnet) | `cpu/address.hpp` | Prefix check | `test_fuzz_address_bip32_ffi.cpp` suites 1-4 | [OK] |
| **A2** | P2WPKH: `bc1q...` prefix (mainnet) | `cpu/address.hpp` | Prefix check | `test_fuzz_address_bip32_ffi.cpp` | [OK] |
| **A3** | P2TR: `bc1p...` prefix (mainnet) | `cpu/address.hpp` | Prefix check | `test_fuzz_address_bip32_ffi.cpp` | [OK] |
| **A4** | WIF round-trip | `cpu/address.hpp` | Encode->decode identity | `test_fuzz_address_bip32_ffi.cpp` | [OK] |
| **A5** | NULL/invalid -> error (no crash) | `cpu/address.hpp` | Fuzz 10K random blobs | `test_fuzz_address_bip32_ffi.cpp` | [OK] |
| **A6** | Zero pubkey -> graceful failure | `cpu/address.hpp` | Edge case | `test_fuzz_address_bip32_ffi.cpp` | [OK] |

**Address Subtotal: 6/6 [OK]**

---

## 11. C ABI (`ufsecp` shim)

| ID | Invariant | Implementation | Validation | Test Location | Status |
|----|-----------|---------------|------------|---------------|--------|
| **C1** | `context_create()` -> non-NULL | `compat/ufsecp.h` | Direct check | `test_fuzz_address_bip32_ffi.cpp` suites 8-13 | [OK] |
| **C2** | `context_destroy(NULL)` = safe no-op | `compat/ufsecp.h` | NULL safety | `test_fuzz_address_bip32_ffi.cpp` | [OK] |
| **C3** | NULL args -> `UFSECP_ERROR_NULL_ARGUMENT` | `compat/ufsecp.h` | All functions | `test_fuzz_address_bip32_ffi.cpp` | [OK] |
| **C4** | `last_error()` reflects last code | `compat/ufsecp.h` | Sequence check | `test_fuzz_address_bip32_ffi.cpp` | [OK] |
| **C5** | `error_string()` -> non-NULL for all codes | `compat/ufsecp.h` | Exhaustive | `test_fuzz_address_bip32_ffi.cpp` | [OK] |
| **C6** | `abi_version()` -> non-zero | `compat/ufsecp.h` | Version check | `test_fuzz_address_bip32_ffi.cpp` | [OK] |
| **C7** | Thread-safety: separate contexts safe | `compat/ufsecp.h` | TSan CI | CI `tsan.yml` | [!] |

**C ABI Subtotal: 6/7 (1 partial -- C7 requires full TSan harness)**

---

## 12. Constant-Time (Side-Channel Resistance)

| ID | Invariant | Implementation | Validation | Test Location | Status |
|----|-----------|---------------|------------|---------------|--------|
| **CT1** | `ct::scalar_mul` timing-independent of scalar | `cpu/ct/point.hpp` | dudect Welch t-test ($|t| < 4.5$) | `test_ct_sidechannel.cpp` -- sections 4a-4b | [OK] |
| **CT2** | `ct::ecdsa_sign` timing-independent of privkey | `cpu/ct/point.hpp` | dudect Welch t-test | `test_ct_sidechannel.cpp` -- section 4c | [OK] |
| **CT3** | `ct::schnorr_sign` timing-independent of privkey | `cpu/ct/point.hpp` | dudect Welch t-test | `test_ct_sidechannel.cpp` -- section 4d | [OK] |
| **CT4** | `ct::field_inv` timing-independent of input | `cpu/ct/field.hpp` | dudect Welch t-test | `test_ct_sidechannel.cpp` -- section 2e | [OK] |
| **CT5** | No secret-dependent branches in CT paths | `cpu/ct/*.hpp` | Code review + compiler disassembly | Manual + `objdump` verification | [!] |
| **CT6** | No secret-dependent memory access in CT paths | `cpu/ct/*.hpp` | Code review + Valgrind (planned) | Manual review | [!] |

**CT Subtotal: 4/6 (2 partial -- CT5/CT6 require formal verification tooling)**

---

## 13. Batch / Performance

| ID | Invariant | Implementation | Validation | Test Location | Status |
|----|-----------|---------------|------------|---------------|--------|
| **BP1** | `batch_inverse(a[]) * a[i] == 1` | `cpu/field.hpp` | Batch vs single inverse (256 elements) | `audit_field.cpp` -> `test_batch_inverse()` | [OK] |
| **BP2** | Batch verify == sequential verify | `cpu/batch_verify.hpp` | Cross-library differential | `test_cross_libsecp256k1.cpp` suites 8-9 | [OK] |
| **BP3** | Hamburg comb == double-and-add | `cpu/ct/point.hpp` | CT generator mul vs naive | `audit_ct.cpp` -> `test_ct_generator_mul()` | [OK] |

**Batch Subtotal: 3/3 [OK]**

---

## 14. Serialization / Parsing

| ID | Invariant | Implementation | Validation | Test Location | Status |
|----|-----------|---------------|------------|---------------|--------|
| **SP1** | DER parse->serialize round-trip | `cpu/ecdsa.hpp` | Fuzz 10K random | `test_fuzz_parsers.cpp` suites 1-3 | [OK] |
| **SP2** | Compressed pubkey round-trip (33 bytes) | `cpu/point.hpp` | Fuzz | `test_fuzz_parsers.cpp` suites 6-8 | [OK] |
| **SP3** | Uncompressed pubkey round-trip (65 bytes) | `cpu/point.hpp` | Fuzz | `test_fuzz_parsers.cpp` suites 6-8 | [OK] |
| **SP4** | Invalid DER -> error (no crash) | `cpu/ecdsa.hpp` | Truncated/bad-tag/bad-length | `test_fuzz_parsers.cpp` suites 1-3 | [OK] |
| **SP5** | 10K random blobs -> no crash | `cpu/ecdsa.hpp` | Fuzz robustness | `test_fuzz_parsers.cpp` | [OK] |

**Parsing Subtotal: 5/5 [OK]**

---

## Cross-Cutting Evidence

### Differential Testing (Gold Standard)

| Evidence | Method | Scale | Location |
|----------|--------|-------|----------|
| UltrafastSecp256k1 == libsecp256k1 v0.6.0 | Bit-exact output comparison | 7,860 checks/CI, 1.3M/nightly | `test_cross_libsecp256k1.cpp` (10 suites) |
| ECDSA cross-sign/verify | UF signs -> Ref verifies, Ref signs -> UF verifies | 500xM each direction | Suites [2], [3] |
| Schnorr cross-sign/verify | Bidirectional BIP-340 | 500xM | Suite [4] |
| RFC 6979 byte-exact nonce | Compact sig byte comparison | 200xM | Suite [5] |

### Boundary Value Coverage

All core arithmetic operations are tested on boundary values:

| Boundary | Field ($\mathbb{F}_p$) | Scalar ($\mathbb{Z}_n$) | Point |
|----------|------------------------|-------------------------|-------|
| $0$ | [OK] `audit_field.cpp` | [OK] `audit_scalar.cpp` | [OK] $\mathcal{O}$ in `audit_point.cpp` |
| $1$ | [OK] | [OK] | [OK] $G$ |
| $p-1$ / $n-1$ | [OK] `test_limb_boundary` | [OK] `test_edge_scalars` | [OK] $(n-1) \cdot G$ |
| $p$ / $n$ | [OK] reduces to 0 | [OK] reduces to 0 | [OK] $n \cdot G = \mathcal{O}$ |
| $p+1$ / $n+1$ | [OK] reduces to 1 | [OK] reduces to 1 | -- |
| $2^{255}$ | [OK] limb stress | [OK] `test_high_bits` | -- |
| $2^{256}-1$ | [OK] `0xFF..FF` stress | -- | -- |

### Fuzzing Coverage

| Harness | Target | Iterations (Nightly) | Location |
|---------|--------|---------------------|----------|
| `fuzz_field` | Field arithmetic | 100K+ | `tests/fuzz/fuzz_field.cpp` |
| `fuzz_scalar` | Scalar arithmetic | 100K+ | `tests/fuzz/fuzz_scalar.cpp` |
| `fuzz_point` | Point operations | 100K+ | `tests/fuzz/fuzz_point.cpp` |
| DER parser fuzz | `test_fuzz_parsers.cpp` | 10K per suite | Suites 1-3 |
| Schnorr parser fuzz | `test_fuzz_parsers.cpp` | 10K per suite | Suites 4-5 |
| Pubkey parse fuzz | `test_fuzz_parsers.cpp` | 10K per suite | Suites 6-8 |
| Address encoder fuzz | `test_fuzz_address_bip32_ffi.cpp` | 10K per suite | Suites 1-4 |
| BIP32 path fuzz | `test_fuzz_address_bip32_ffi.cpp` | 10K per suite | Suites 5-7 |
| FFI boundary fuzz | `test_fuzz_address_bip32_ffi.cpp` | 10K per suite | Suites 8-13 |

### Negative Testing (Adversarial Inputs)

| Category | Description | Test Location |
|----------|-------------|---------------|
| Zero key ECDSA | `sign(msg, 0)` -> zero sig; `verify` rejects | `audit_security.cpp` -> `test_zero_key_handling()` |
| Zero key Schnorr | `schnorr_sign(0, msg, aux)` -> fails gracefully | `audit_fuzz.cpp` -> `test_malformed_pubkeys()` |
| Off-curve point | Verify with infinity -> false | `audit_fuzz.cpp` -> `test_malformed_pubkeys()` |
| $r = 0$ signature | `verify(msg, pk, {r=0, s=1})` -> false | `audit_fuzz.cpp` -> `test_invalid_ecdsa_sigs()` |
| $s = 0$ signature | `verify(msg, pk, {r=1, s=0})` -> false | `audit_fuzz.cpp` -> `test_invalid_ecdsa_sigs()` |
| Bit-flip resilience | 1-bit change in sig -> verify fails | `audit_security.cpp` -> `test_bitflip_resilience()` |
| Message bit-flip | 1-bit change in msg -> verify fails | `audit_security.cpp` -> `test_message_bitflip()` |
| Nonce determinism | Same (msg, sk) -> same nonce | `audit_security.cpp` -> `test_nonce_determinism()` |
| Zeroization | Secret memory zeroed after use | `audit_security.cpp` -> `test_zeroization()` |

---

## Aggregate Summary

| Category | Total | [OK] Verified | [!] Partial | [FAIL] Gap |
|----------|-------|------------|-----------|-------|
| Field (F) | 17 | 17 | 0 | 0 |
| Scalar (S) | 9 | 9 | 0 | 0 |
| Point (P) | 14 | 14 | 0 | 0 |
| GLV (G) | 4 | 4 | 0 | 0 |
| ECDSA (E) | 8 | 8 | 0 | 0 |
| Schnorr (B) | 6 | 6 | 0 | 0 |
| MuSig2 (M) | 7 | 7 | 0 | 0 |
| FROST (FR) | 9 | 9 | 0 | 0 |
| BIP-32 (H) | 7 | 7 | 0 | 0 |
| Address (A) | 6 | 6 | 0 | 0 |
| C ABI (C) | 7 | 6 | 1 | 0 |
| CT (CT) | 6 | 4 | 2 | 0 |
| Batch (BP) | 3 | 3 | 0 | 0 |
| Parsing (SP) | 5 | 5 | 0 | 0 |
| **Total** | **108** | **105** | **3** | **0** |

**Partial items** (3):
- **C7**: Thread-safety (TSan in CI, but no dedicated multi-threaded stress test)
- **CT5**: No secret-dependent branches (code review only, no CTGRIND/formal tool)
- **CT6**: No secret-dependent memory access (code review only)

---

## How to Reproduce

```bash
# Full audit suite (from build directory)
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure

# Specific audit targets
./build/cpu/audit_field          # 641K+ field checks
./build/cpu/audit_scalar         # scalar checks
./build/cpu/audit_point          # point + signature checks
./build/cpu/audit_ct             # CT correctness
./build/cpu/audit_security       # security hardening
./build/cpu/audit_fuzz           # adversarial inputs
./build/cpu/audit_integration    # end-to-end flows

# Differential testing (requires libsecp256k1)
./build/cpu/test_cross_libsecp256k1    # 7,860 baseline checks
DIFFERENTIAL_MULTIPLIER=100 ./build/cpu/test_cross_libsecp256k1  # 1.3M checks

# dudect side-channel (statistical)
./build/cpu/test_ct_sidechannel        # full mode (~30 min)
./build/cpu/test_ct_sidechannel_smoke  # smoke mode (~2 min)
```

---

*Generated: 2026-02-25*
*Invariant source: [INVARIANTS.md](INVARIANTS.md)*
*This document is auto-updatable via `scripts/generate_traceability.sh`*
