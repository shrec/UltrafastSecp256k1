# Audit Traceability Matrix

**UltrafastSecp256k1 v3.22.0** -- Evidence-Based Correctness & Security Mapping

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
| **M6** | Rogue-key attack detected | `cpu/musig2.hpp` | Wagner-style simulation | `test_musig2_frost_advanced.cpp` suites 1-2, `test_adversarial_protocol.cpp` A.4 | [OK] |
| **M7** | Nonce reuse detected | `cpu/musig2.hpp` | Cross-message detection | `test_musig2_frost_advanced.cpp` suites 3-4, `test_adversarial_protocol.cpp` A.1 | [OK] |
| **M8** | Transcript mutation detected | `cpu/musig2.hpp` | Corrupt keyagg blob between steps | `test_adversarial_protocol.cpp` A.5 | [OK] |
| **M9** | Signer ordering mismatch detected | `cpu/musig2.hpp` | Sign with wrong index | `test_adversarial_protocol.cpp` A.6 | [OK] |
| **M10** | Malicious aggregator detected | `cpu/musig2.hpp` | Tampered aggnonce | `test_adversarial_protocol.cpp` A.7 | [OK] |

**MuSig2 Subtotal: 10/10 [OK]**

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
| **FR9** | Below-threshold subset fails | `cpu/frost.hpp` | 1-of-3 attempt -> fail | `test_musig2_frost_advanced.cpp`, `test_adversarial_protocol.cpp` B.1 | [OK] |
| **FR10** | Malicious coordinator detected | `cpu/frost.hpp` | Inconsistent commit sets | `test_adversarial_protocol.cpp` B.4 | [OK] |
| **FR11** | Duplicate nonce commitments handled | `cpu/frost.hpp` | Submit same nonce twice | `test_adversarial_protocol.cpp` B.5 | [OK] |

**FROST Subtotal: 11/11 [OK]**

---

## 9. BIP-324 Encrypted Transport

| ID | Invariant | Implementation | Validation | Test Location | Status |
|----|-----------|---------------|------------|---------------|--------|
| **T3241** | Initiator/responder derive matching session state from the same handshake transcript | `cpu/include/secp256k1/bip324.hpp`, `cpu/src/bip324.cpp` | Deterministic handshake round-trip | `cpu/tests/test_bip324.cpp` -> `test_bip324_session()` | [OK] |
| **T3242** | Post-handshake encrypt/decrypt are inverse operations across message sequences | `cpu/include/secp256k1/bip324.hpp`, `cpu/src/bip324.cpp` | Multi-message round-trip | `cpu/tests/test_bip324.cpp` -> `test_bip324_sequence()` | [OK] |
| **T3243** | Fixed-key sessions remain deterministic across repeated runs | `cpu/include/secp256k1/bip324.hpp`, `cpu/src/bip324.cpp` | Repeatability check | `cpu/tests/test_bip324.cpp` -> `test_bip324_determinism()` | [OK] |
| **T3244** | Empty, short, and 4 KiB payloads survive transport round-trip | `cpu/include/secp256k1/bip324.hpp`, `cpu/src/bip324.cpp` | Size sweep | `cpu/tests/test_bip324.cpp` -> `test_bip324_sizes()` | [OK] |
| **T3245** | Tampered ciphertext, tag, or transcript inputs are rejected | `cpu/include/secp256k1/bip324.hpp`, `cpu/src/bip324.cpp` | Tamper matrix + wrong-peer checks | `cpu/tests/test_bip324.cpp` -> `test_bip324_tamper()` | [OK] |
| **T3246** | Public C ABI create/handshake/encrypt/decrypt surface matches C++ transport behavior | `include/ufsecp/ufsecp.h`, `include/ufsecp/ufsecp_impl.cpp` | FFI round-trip coverage | `cpu/tests/test_ffi_coverage.cpp` -> `test_bip324_session()` | [OK] |

**BIP-324 Subtotal: 6/6 [OK]**

---

## 10. BIP-32 HD Derivation

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

## 11. Address Generation

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

## 12. C ABI (`ufsecp` shim)

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

## 13. Constant-Time (Side-Channel Resistance)

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

## 14. Batch / Performance

| ID | Invariant | Implementation | Validation | Test Location | Status |
|----|-----------|---------------|------------|---------------|--------|
| **BP1** | `batch_inverse(a[]) * a[i] == 1` | `cpu/field.hpp` | Batch vs single inverse (256 elements) | `audit_field.cpp` -> `test_batch_inverse()` | [OK] |
| **BP2** | Batch verify == sequential verify | `cpu/batch_verify.hpp` | Cross-library differential | `test_cross_libsecp256k1.cpp` suites 8-9 | [OK] |
| **BP3** | Hamburg comb == double-and-add | `cpu/ct/point.hpp` | CT generator mul vs naive | `audit_ct.cpp` -> `test_ct_generator_mul()` | [OK] |

**Batch Subtotal: 3/3 [OK]**

---

## 15. Serialization / Parsing

| ID | Invariant | Implementation | Validation | Test Location | Status |
|----|-----------|---------------|------------|---------------|--------|
| **SP1** | DER parse->serialize round-trip | `cpu/ecdsa.hpp` | Fuzz 10K random | `test_fuzz_parsers.cpp` suites 1-3 | [OK] |
| **SP2** | Compressed pubkey round-trip (33 bytes) | `cpu/point.hpp` | Fuzz | `test_fuzz_parsers.cpp` suites 6-8 | [OK] |
| **SP3** | Uncompressed pubkey round-trip (65 bytes) | `cpu/point.hpp` | Fuzz | `test_fuzz_parsers.cpp` suites 6-8 | [OK] |
| **SP4** | Invalid DER -> error (no crash) | `cpu/ecdsa.hpp` | Truncated/bad-tag/bad-length | `test_fuzz_parsers.cpp` suites 1-3 | [OK] |
| **SP5** | 10K random blobs -> no crash | `cpu/ecdsa.hpp` | Fuzz robustness | `test_fuzz_parsers.cpp` | [OK] |

**Parsing Subtotal: 5/5 [OK]**

---

## 16. ECIES Hardening

| ID | Invariant | Implementation | Validation | Test Location | Status |
|----|-----------|---------------|------------|---------------|--------|
| **EC1** | Encrypt->decrypt round-trip (1, 13, 32 byte plaintexts) | `cpu/src/ecies.cpp` | KAT with 3 sizes, wrong-key rejection | `test_ecies_regression.cpp` -> `test_ecies_roundtrip_kat()` | [OK] |
| **EC2** | Parity tamper: flip 0x02/0x03 on ephemeral pubkey -> decrypt fails | `cpu/src/ecies.cpp` | Deterministic bit-flip | `test_ecies_regression.cpp` -> `test_ecies_parity_tamper()` | [OK] |
| **EC3** | Invalid prefix (0x00, 0x04, 0xFF) -> clean error | `cpu/src/ecies.cpp` | 3 bad prefix checks | `test_ecies_regression.cpp` -> `test_ecies_invalid_prefix()` | [OK] |
| **EC4** | Truncated envelope (0-81 bytes) -> clean error, no crash | `cpu/src/ecies.cpp` | 6 truncated sizes | `test_ecies_regression.cpp` -> `test_ecies_truncated_envelope()` | [OK] |
| **EC5** | Single-bit tamper in any field (pubkey/IV/ct/HMAC) -> decrypt fails | `cpu/src/ecies.cpp` | Tamper matrix: 4 fields x bit-flip | `test_ecies_regression.cpp` -> `test_ecies_tamper_matrix()` | [OK] |
| **EC6** | ABI prefix rejection: 6 bad prefixes x 5 endpoints -> consistent ERR | `include/ufsecp/ufsecp_impl.cpp` | 30 ABI boundary checks | `test_ecies_regression.cpp` -> `test_abi_prefix_rejection()` | [OK] |
| **EC7** | Pubkey parser consistency: malformed x-coords -> same error across all parsers | `include/ufsecp/ufsecp_impl.cpp` | 3 malformed coords x 3 functions | `test_ecies_regression.cpp` -> `test_pubkey_parser_consistency()` | [OK] |
| **EC8** | RNG fail-closed: blocked `getrandom` -> process SIGABRT (no silent fallback) | `cpu/src/random.cpp` | fork + seccomp filter (Linux x86-64) | `test_ecies_regression.cpp` -> `test_rng_fail_closed()` | [OK] |

**ECIES Subtotal: 8/8 [OK]**

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
| BIP-324 transport checks | `cpu/tests/test_bip324.cpp` | 62 checks | handshake, sequence, determinism, sizes, tamper, random-key paths |
| ECIES regression | `test_ecies_regression.cpp` | 85 tests | Categories A-H |

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
| MuSig2 rogue-key | 0xFF / zero / duplicate xonly keys | `test_adversarial_protocol.cpp` A.4 |
| MuSig2 transcript mutation | Corrupt keyagg blob between steps | `test_adversarial_protocol.cpp` A.5 |
| MuSig2 signer ordering | Wrong signer index | `test_adversarial_protocol.cpp` A.6 |
| MuSig2 malicious aggregator | Tampered aggnonce | `test_adversarial_protocol.cpp` A.7 |
| FROST malicious coordinator | Inconsistent commit sets to signers | `test_adversarial_protocol.cpp` B.4 |
| FROST duplicate nonce | Same commitment submitted twice | `test_adversarial_protocol.cpp` B.5 |
| BIP-324 tampered packet | Ciphertext/tag/transcript mutation must fail | `cpu/tests/test_bip324.cpp` -> `test_bip324_tamper()` |
| Adaptor transcript mismatch | Sign msg1, verify msg2 -> reject | `test_adversarial_protocol.cpp` D.5 |
| Adaptor extraction misuse | Extract from unrelated sig pair | `test_adversarial_protocol.cpp` D.6 |
| DLEQ malformed proof | 6 corruption strategies + zero proof | `test_adversarial_protocol.cpp` E.4 |
| DLEQ wrong generators | Swap G/H, swap P/Q, different G'/H' | `test_adversarial_protocol.cpp` E.5 |
| FFI undersized buffers | DER, WIF, BIP-39 with tiny output buffers | `test_adversarial_protocol.cpp` G.18 |
| FFI overlapping buffers | Input==output aliasing | `test_adversarial_protocol.cpp` G.19 |
| FFI malformed counts | n=0 for combine, batch, multi_scalar_mul | `test_adversarial_protocol.cpp` G.20 |

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
| MuSig2 (M) | 10 | 10 | 0 | 0 |
| FROST (FR) | 11 | 11 | 0 | 0 |
| BIP-324 (T324) | 6 | 6 | 0 | 0 |
| BIP-32 (H) | 7 | 7 | 0 | 0 |
| Address (A) | 6 | 6 | 0 | 0 |
| C ABI (C) | 7 | 6 | 1 | 0 |
| CT (CT) | 6 | 4 | 2 | 0 |
| Batch (BP) | 3 | 3 | 0 | 0 |
| Parsing (SP) | 5 | 5 | 0 | 0 |
| ECIES (EC) | 8 | 8 | 0 | 0 |
| **Total** | **128** | **125** | **3** | **0** |

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

## GPU C ABI Audit Coverage

| Test | Scope | Source |
|------|-------|--------|
| `gpu_abi_gate` | ABI surface, error codes, discovery, lifecycle, NULL safety | `audit/test_gpu_abi_gate.cpp` |
| `gpu_ops_equivalence` | GPU vs CPU reference: all 6 ops (gen_mul, ecdsa, schnorr, ecdh, hash160, msm) | `audit/test_gpu_ops_equivalence.cpp` |
| `gpu_host_api_negative` | NULL ptrs, count=0, invalid backend/device, error strings | `audit/test_gpu_host_api_negative.cpp` |
| `gpu_backend_matrix` | Backend enumeration, device info, per-backend op probing | `audit/test_gpu_backend_matrix.cpp` |

Backend-specific internal audit runners:
- CUDA: `cuda/src/gpu_audit_runner.cu` (27 modules)
- OpenCL: `opencl/src/opencl_audit_runner.cpp` (27 modules)
- Metal: `metal/src/metal_audit_runner.mm` (27 modules)

---

## New ABI Surface Edge Cases (§H / §N, v3.22+)

> Gap analysis (v3.22) found 26 `ufsecp_*` functions with no dedicated
> edge-case coverage. All gaps are closed by the test functions below, wired
> into `test_adversarial_protocol_run()`.
> Source: `audit/test_adversarial_protocol.cpp`

| ID | Claim | ABI functions | Validation method | Test location | Status |
|----|-------|---------------|-------------------|---------------|--------|
| **H1** | `ufsecp_ctx_size()` returns > 0 | `ufsecp_ctx_size` | Direct check | `test_h1_ctx_size()` | [OK] |
| **H2a** | AEAD encrypt/decrypt NULL args → `ERR_NULL_ARG` | `ufsecp_aead_chacha20_encrypt`, `ufsecp_aead_chacha20_decrypt` | NULL injection | `test_h2_aead()` | [OK] |
| **H2b** | AEAD bad authentication tag rejected | `ufsecp_aead_chacha20_decrypt` | Tampered ciphertext | `test_h2_aead()` | [OK] |
| **H2c** | AEAD wrong nonce rejected | `ufsecp_aead_chacha20_decrypt` | Modified nonce | `test_h2_aead()` | [OK] |
| **H2d** | AEAD zero-length plaintext roundtrip succeeds | `ufsecp_aead_chacha20_encrypt`, `ufsecp_aead_chapha20_decrypt` | Smoke | `test_h2_aead()` | [OK] |
| **H3a** | ECIES NULL args → `ERR_NULL_ARG` | `ufsecp_ecies_encrypt`, `ufsecp_ecies_decrypt` | NULL injection | `test_h3_ecies()` | [OK] |
| **H3b** | ECIES off-curve pubkey rejected | `ufsecp_ecies_encrypt` | Invalid point | `test_h3_ecies()` | [OK] |
| **H3c** | ECIES tampered envelope rejected | `ufsecp_ecies_decrypt` | Byte flip | `test_h3_ecies()` | [OK] |
| **H4a** | EllSwift NULL args → `ERR_NULL_ARG` | `ufsecp_ellswift_create`, `ufsecp_ellswift_xdh` | NULL injection | `test_h4_ellswift()` | [OK] |
| **H4b** | EllSwift zero privkey rejected | `ufsecp_ellswift_create` | Zero scalar | `test_h4_ellswift()` | [OK] |
| **H4c** | EllSwift symmetric shared secret | `ufsecp_ellswift_xdh` | A-to-B == B-to-A | `test_h4_ellswift()` | [OK] |
| **H5a** | ETH checksummed NULL args → error | `ufsecp_eth_address_checksummed` | NULL injection | `test_h5_eth_edge()` | [OK] |
| **H5b** | ETH checksummed undersized buffer → `ERR_BUF_TOO_SMALL` | `ufsecp_eth_address_checksummed` | Short buffer | `test_h5_eth_edge()` | [OK] |
| **H5c** | ETH personal_hash NULL args → error | `ufsecp_eth_personal_hash` | NULL injection | `test_h5_eth_edge()` | [OK] |
| **H6a** | Pedersen switch_commit NULL args → `ERR_NULL_ARG` | `ufsecp_pedersen_switch_commit` | NULL injection | `test_h6_pedersen_switch()` | [OK] |
| **H6b** | Pedersen switch_commit output is valid point (prefix 0x02/0x03) | `ufsecp_pedersen_switch_commit` | Prefix byte check | `test_h6_pedersen_switch()` | [OK] |
| **H7a** | Schnorr adaptor extract NULL args → `ERR_NULL_ARG` | `ufsecp_schnorr_adaptor_extract` | NULL injection | `test_h7_schnorr_adaptor_extract()` | [OK] |
| **H7b** | Schnorr adaptor extract zero inputs rejected | `ufsecp_schnorr_adaptor_extract` | Zero bytes | `test_h7_schnorr_adaptor_extract()` | [OK] |
| **H8a** | `ecdsa_sign_batch` NULL args → error | `ufsecp_ecdsa_sign_batch` | NULL injection | `test_h8_batch_sign()` | [OK] |
| **H8b** | `schnorr_sign_batch` NULL args → error | `ufsecp_schnorr_sign_batch` | NULL injection | `test_h8_batch_sign()` | [OK] |
| **H8c** | Batch sign count=0 → error | `ufsecp_ecdsa_sign_batch`, `ufsecp_schnorr_sign_batch` | Zero count | `test_h8_batch_sign()` | [OK] |
| **H9a** | BIP-143 sighash NULL args → `ERR_NULL_ARG` | `ufsecp_bip143_sighash` | NULL injection | `test_h9_bip143()` | [OK] |
| **H9b** | P2WPKH script_code has OP_DUP OP_HASH160 PUSH20 format | `ufsecp_bip143_p2wpkh_script_code` | Byte pattern check | `test_h9_bip143()` | [OK] |
| **H10a** | BIP-144 txid/wtxid NULL args → `ERR_NULL_ARG` | `ufsecp_bip144_txid`, `ufsecp_bip144_wtxid` | NULL injection | `test_h10_bip144()` | [OK] |
| **H10b** | BIP-144 witness_commitment is deterministic | `ufsecp_bip144_witness_commitment` | Two-call equality | `test_h10_bip144()` | [OK] |
| **H11a** | is_witness_program: short/non-witness → 0 | `ufsecp_is_witness_program` | Short + P2PKH input | `test_h11_segwit()` | [OK] |
| **H11b** | parse_witness_program: non-witness → error | `ufsecp_parse_witness_program` | P2PKH script | `test_h11_segwit()` | [OK] |
| **H11c** | `ufsecp_p2wpkh_spk` output: OP_0 + PUSH20 (22 bytes) | `ufsecp_p2wpkh_spk` | Length + opcode check | `test_h11_segwit()` | [OK] |
| **H11d** | `ufsecp_p2wsh_spk` output: OP_0 + PUSH32 (34 bytes) | `ufsecp_p2wsh_spk` | Length + opcode check | `test_h11_segwit()` | [OK] |
| **H11e** | `ufsecp_p2tr_spk` output: OP_1 + PUSH32 (34 bytes) | `ufsecp_p2tr_spk` | Length + opcode check | `test_h11_segwit()` | [OK] |
| **H12a** | Taproot keypath sighash NULL ctx/prevouts → `ERR_NULL_ARG` | `ufsecp_taproot_keypath_sighash` | NULL injection | `test_h12_taproot_sighash()` | [OK] |
| **H12b** | Taproot keypath sighash count=0 → error | `ufsecp_taproot_keypath_sighash` | Zero count | `test_h12_taproot_sighash()` | [OK] |
| **H12c** | Taproot keypath sighash OOB input_index → error | `ufsecp_taproot_keypath_sighash` | index >= count | `test_h12_taproot_sighash()` | [OK] |
| **H12d** | Tapscript sighash NULL tapleaf_hash → `ERR_NULL_ARG` | `ufsecp_tapscript_sighash` | NULL injection | `test_h12_taproot_sighash()` | [OK] |
| **H12e** | Taproot sighash is deterministic | `ufsecp_taproot_keypath_sighash` | Two-call equality | `test_h12_taproot_sighash()` | [OK] |

**H Subtotal: 35/35 [OK]**

---

## Remaining ABI Surface (§I / §O, v3.23+)

Eight functions with zero prior coverage: `ctx_clone`, `last_error_msg`, `last_error`,
`pubkey_parse`, `pubkey_create_uncompressed`, `ecdsa_sign_recoverable`, `ecdsa_recover`,
`ecdsa_sign_verified`, `schnorr_sign_verified`, plus deep batch-verify coverage for
`schnorr_batch_verify`, `ecdsa_batch_verify`, `schnorr_batch_identify_invalid`,
`ecdsa_batch_identify_invalid`.

| ID | Invariant | Function(s) | Class | Test | Status |
|----|-----------|-------------|-------|------|--------|
| **I1a** | `ctx_clone(nullptr, &out)` → `ERR_NULL_ARG` | `ufsecp_ctx_clone` | NULL injection | `test_i1_ctx_clone_and_last_error_msg()` | [OK] |
| **I1b** | `ctx_clone(ctx, nullptr)` → `ERR_NULL_ARG` | `ufsecp_ctx_clone` | NULL injection | `test_i1_ctx_clone_and_last_error_msg()` | [OK] |
| **I1c** | `ctx_clone(valid, &out)` produces independent non-null ctx | `ufsecp_ctx_clone` | Valid call | `test_i1_ctx_clone_and_last_error_msg()` | [OK] |
| **I1d** | `last_error_msg` on fresh ctx is non-null | `ufsecp_last_error_msg` | State probe | `test_i1_ctx_clone_and_last_error_msg()` | [OK] |
| **I1e** | `last_error_msg` + `last_error` non-zero after forced error | `ufsecp_last_error_msg`, `ufsecp_last_error` | Error path | `test_i1_ctx_clone_and_last_error_msg()` | [OK] |
| **I1f** | Cloned ctx produces identical output to original | `ufsecp_ctx_clone` | Functional | `test_i1_ctx_clone_and_last_error_msg()` | [OK] |
| **I2a** | `pubkey_create_uncompressed` NULL ctx/privkey/output → error | `ufsecp_pubkey_create_uncompressed` | NULL injection | `test_i2_pubkey_parse_and_uncompressed()` | [OK] |
| **I2b** | `pubkey_create_uncompressed` zero privkey rejected | `ufsecp_pubkey_create_uncompressed` | Zero key | `test_i2_pubkey_parse_and_uncompressed()` | [OK] |
| **I2c** | `pubkey_create_uncompressed` output starts with 0x04 | `ufsecp_pubkey_create_uncompressed` | Format check | `test_i2_pubkey_parse_and_uncompressed()` | [OK] |
| **I2d** | `pubkey_parse` NULL ctx/input/output → error | `ufsecp_pubkey_parse` | NULL injection | `test_i2_pubkey_parse_and_uncompressed()` | [OK] |
| **I2e** | `pubkey_parse` wrong length (32) → error | `ufsecp_pubkey_parse` | Bad length | `test_i2_pubkey_parse_and_uncompressed()` | [OK] |
| **I2f** | `pubkey_parse` 0x00 prefix → error | `ufsecp_pubkey_parse` | Bad prefix | `test_i2_pubkey_parse_and_uncompressed()` | [OK] |
| **I2g** | `pubkey_parse` uncompressed (65B) → normalised to compressed | `ufsecp_pubkey_parse` | Normalisation | `test_i2_pubkey_parse_and_uncompressed()` | [OK] |
| **I2h** | `pubkey_parse` compressed round-trip identical | `ufsecp_pubkey_parse` | Round-trip | `test_i2_pubkey_parse_and_uncompressed()` | [OK] |
| **I3a** | `sign_recoverable` NULL ctx/msg/privkey/sig → error | `ufsecp_ecdsa_sign_recoverable` | NULL injection | `test_i3_ecdsa_recoverable_roundtrip()` | [OK] |
| **I3b** | `sign_recoverable` NULL recid_out rejected | `ufsecp_ecdsa_sign_recoverable` | NULL injection | `test_i3_ecdsa_recoverable_roundtrip()` | [OK] |
| **I3c** | `sign_recoverable` zero privkey rejected | `ufsecp_ecdsa_sign_recoverable` | Zero key | `test_i3_ecdsa_recoverable_roundtrip()` | [OK] |
| **I3d** | recid output is in [0, 3] | `ufsecp_ecdsa_sign_recoverable` | Range check | `test_i3_ecdsa_recoverable_roundtrip()` | [OK] |
| **I3e** | `ecdsa_recover` NULL ctx/msg/sig/output → error | `ufsecp_ecdsa_recover` | NULL injection | `test_i3_ecdsa_recoverable_roundtrip()` | [OK] |
| **I3f** | `ecdsa_recover` invalid recid (-1, 4) rejected | `ufsecp_ecdsa_recover` | Bad recid | `test_i3_ecdsa_recoverable_roundtrip()` | [OK] |
| **I3g** | Recovered pubkey matches original on correct recid | `ufsecp_ecdsa_recover` | Round-trip | `test_i3_ecdsa_recoverable_roundtrip()` | [OK] |
| **I3h** | Wrong recid produces different pubkey (or error) | `ufsecp_ecdsa_recover` | Non-malleability | `test_i3_ecdsa_recoverable_roundtrip()` | [OK] |
| **I4a** | `ecdsa_sign_verified` NULL all args → error | `ufsecp_ecdsa_sign_verified` | NULL injection | `test_i4_sign_verified()` | [OK] |
| **I4b** | `ecdsa_sign_verified` zero privkey rejected | `ufsecp_ecdsa_sign_verified` | Zero key | `test_i4_sign_verified()` | [OK] |
| **I4c** | `ecdsa_sign_verified` output verifies with `ecdsa_verify` | `ufsecp_ecdsa_sign_verified` | Round-trip | `test_i4_sign_verified()` | [OK] |
| **I4d** | `schnorr_sign_verified` NULL all args → error | `ufsecp_schnorr_sign_verified` | NULL injection | `test_i4_sign_verified()` | [OK] |
| **I4e** | `schnorr_sign_verified` zero privkey rejected | `ufsecp_schnorr_sign_verified` | Zero key | `test_i4_sign_verified()` | [OK] |
| **I4f** | `schnorr_sign_verified` output verifies with `schnorr_verify` | `ufsecp_schnorr_sign_verified` | Round-trip | `test_i4_sign_verified()` | [OK] |
| **I5a** | `schnorr_batch_verify` 1 valid entry → OK | `ufsecp_schnorr_batch_verify` | Valid path | `test_i5_batch_verify_deep()` | [OK] |
| **I5b** | `schnorr_batch_verify` tampered sig → error | `ufsecp_schnorr_batch_verify` | Tampered sig | `test_i5_batch_verify_deep()` | [OK] |
| **I5c** | `schnorr_batch_identify_invalid` finds index 0 for tampered entry | `ufsecp_schnorr_batch_identify_invalid` | Index tracking | `test_i5_batch_verify_deep()` | [OK] |
| **I5d** | `ecdsa_batch_verify` 1 valid entry → OK | `ufsecp_ecdsa_batch_verify` | Valid path | `test_i5_batch_verify_deep()` | [OK] |
| **I5e** | `ecdsa_batch_verify` tampered sig → error | `ufsecp_ecdsa_batch_verify` | Tampered sig | `test_i5_batch_verify_deep()` | [OK] |
| **I5f** | `ecdsa_batch_identify_invalid` finds index 0 for tampered entry | `ufsecp_ecdsa_batch_identify_invalid` | Index tracking | `test_i5_batch_verify_deep()` | [OK] |
| **I5g** | Batch verify count=0 is vacuously OK | `ufsecp_schnorr_batch_verify`, `ufsecp_ecdsa_batch_verify` | Empty input | `test_i5_batch_verify_deep()` | [OK] |

**I Subtotal: 35/35 [OK]**

---

*Generated: 2026-03-23*
*Invariant source: [INVARIANTS.md](INVARIANTS.md)*
*This document is auto-updatable via `scripts/generate_traceability.sh`*
