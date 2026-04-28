# Upstream Test Porting Analysis — Bitcoin Core & libsecp256k1

**Date:** 2026-04-28  
**CAAS Audit Rounds:** All 11 original CAAS gaps resolved  
**Current exploit PoC count:** 207 `test_exploit_*.cpp` files  
**ALL_MODULES count:** ~180+ registered in `unified_audit_runner.cpp`  
**Upstream libsecp256k1 `tests.c`:** 56 `run_*` test functions  
**Bitcoin Core `test_bitcoin`:** 693 tests, all passing via shim

---

## 1. Executive Summary

This analysis compares the **UltrafastSecp256k1 CAAS audit suite** against two upstream
test sources:

1. **Upstream libsecp256k1 `tests.c`** (56 `run_*` functions + 6 module `tests_impl.h`)
2. **Bitcoin Core `test_bitcoin`** (693 tests accessed via shim compatibility)

**Overall finding:** CAAS coverage is **significantly broader** than upstream in most
categories (exploit PoCs, Wycheproof vectors, GPU, side-channel, formal verification).
However, there are **8 specific gaps** where upstream tests exercise unique code paths
that our suite does not explicitly cover.

---

## 2. libsecp256k1 tests.c — Full Comparison

### 2.1. Fields & Scalars — Covered ✅

| Upstream function | Our equivalent | Status |
|---|---|---|
| `run_field_convert` | `audit_field_run`, `field_26`, `field_52` | ✅ Covered |
| `run_field_be32_overflow` | `exploit_field_boundary_exhaustive` | ✅ Covered |
| `run_field_half` | `audit_field_run` | ✅ Covered |
| `run_field_misc` | `audit_field_run`, `audit_scalar_run` | ✅ Covered |
| `run_fe_mul` | `audit_field_run` | ✅ Covered |
| `run_sqr` | `audit_field_run` | ✅ Covered |
| `run_sqrt` | `audit_field_run` | ✅ Covered |
| `run_scalar_tests` | `audit_scalar_run`, `exploit_scalar_systematic` | ✅ Covered |
| `run_scalar_set_b32_seckey_tests` | `exploit_private_key` | ✅ Covered |
| `run_inverse_tests` / `run_modinv_tests` | `exploit_safegcd_divsteps` | ✅ Covered |
| `run_int128_tests` | `carry_propagation`, `field_52` | ✅ Covered |

### 2.2. Group Operations & ECC — Covered ✅

| Upstream function | Our equivalent | Status |
|---|---|---|
| `run_ge` | `audit_point_run`, `point_group_law` | ✅ Covered |
| `run_gej` | `audit_point_run` | ✅ Covered |
| `run_ec_combine` | `audit_point_run` | ✅ Covered |
| `run_group_decompress` | `point_serialization` | ✅ Covered |
| `run_point_times_order` | `audit_point_run` | ✅ Covered |
| `run_ecmult_chain` | `multiscalar`, `exploit_multiscalar` | ✅ Covered |
| `run_ecmult_const_tests` | `ct` (CT scalar_mul), `diag_scalar_mul` | ✅ Covered |
| `run_ecmult_multi_tests` | `multiscalar_batch`, `exploit_pippenger_msm` | ✅ Covered |
| `run_ecmult_constants` | `audit_point_run` | ✅ Covered |
| `run_ecmult_gen_blind` | `exploit_ct_systematic`, `c_abi_negative` | ✅ Covered |
| `run_endomorphism_tests` | `glv_endomorphism`, `glv_kat` | ✅ Covered |
| `run_ecmult_near_split_bound` | `audit_scalar_run` | ✅ Covered |
| `run_ecmult_pre_g` | (gen_fb_table in selftest) | ✅ Covered |
| `run_ec_pubkey_parse_test` | `c_abi_negative`, `parse_strictness` | ✅ Covered |

### 2.3. ECDSA, Signatures — Covered ✅

| Upstream function | Our equivalent | Status |
|---|---|---|
| `run_eckey_edge_case_test` | `exploit_libsecp_eckey_api` (ECKEY-1..17) | ✅ **Ported directly** |
| `run_eckey_negate_test` | `exploit_libsecp_eckey_api` (ECKEY-11) | ✅ Covered |
| `run_ecdsa_sign_verify` | `ecdsa_schnorr`, `exploit_ecdsa_edge_cases` | ✅ Covered |
| `run_ecdsa_end_to_end` | `exploit_ecdsa_rfc6979_kat` | ✅ Covered |
| `run_ecdsa_der_parse` | `exploit_ecdsa_der_confusion`, `c_abi_negative` | ✅ Covered |
| `run_ecdsa_edge_cases` | `exploit_ecdsa_edge_cases`, `exploit_ecdsa_r_overflow` | ✅ Covered |
| `run_ecdsa_wycheproof` | 11 Wycheproof suites (4× ECDSA, 2× ECDH, etc.) | ✅ **Exceeds upstream** |

### 2.4. Hash Functions & HMAC — Covered ✅

| Upstream function | Our equivalent | Status |
|---|---|---|
| `run_sha256_known_output_tests` | `sha256_kat`, `sha_kat` | ✅ Covered |
| `run_sha256_counter_tests` | `sha256_kat` | ✅ Covered |
| `run_hmac_sha256_tests` | `wycheproof_hmac_sha256` | ✅ Covered |
| `run_rfc6979_hmac_sha256_tests` | `rfc6979_vectors` | ✅ Covered |
| `run_tagged_sha256_tests` | `sha256_kat` (BIP-340 tagged) | ✅ Covered |

### 2.5. Memory Safety & Byte Ordering — Covered ✅

| Upstream function | Our equivalent | Status |
|---|---|---|
| `run_secp256k1_memczero_test` | `audit_secure_erase`, `regression_bip324_session` | ✅ Covered |
| `run_secp256k1_is_zero_array_test` | `audit_secure_erase` | ✅ Covered |
| `run_secp256k1_byteorder_tests` | `exploit_scalar_invariants` | ✅ Covered |
| `run_cmov_tests` | `ct_run`, `ct_equivalence` | ✅ Covered |

### 2.6. Context & Lifecycle — Covered ✅

| Upstream function | Our equivalent | Status |
|---|---|---|
| `run_deprecated_context_flags_test` | `c_abi_negative` | ✅ Covered |
| `run_ec_illegal_argument_tests` | `c_abi_negative` (null guards) | ✅ Covered |
| `run_static_context_tests` | `c_abi_negative` | ✅ Covered |
| `run_proper_context_tests` | `c_abi_negative`, `c_abi_thread_stress` | ✅ Covered |
| `run_selftest_tests` | `exploit_selftest_api` | ✅ Covered |

---

## 3. ⚠️ GAP: Upstream Tests Without CAAS Equivalents

These 7 upstream test categories have **no direct CAAS exploit PoC equivalent**.
Each represents a code path coverage gap in our audit suite.

### GAP-1: `run_scratch_tests` — Scratch Space Allocation

**What it tests:** `secp256k1_scratch_space_create`, `_destroy`, `_max_pages`,
`_alloc` edge cases including OOM, zero-size, and alignment.

**Why it matters:** The libsecp256k1 scratch space allocator is used by batch
verification and multiscalar operations. If our equivalent allocator has bugs
(alignment, size overflow, use-after-free), they won't show up in protocol-level
tests that don't drive allocation boundaries.

**Coverage in CAAS:** ❌ None. No exploit test targets scratch allocation directly.

**Porting effort:** Low (~half day). A ~150-line test exercising scratch allocation
boundaries (null, zero, max, alignment, double-free, OOM recovery).

---

### GAP-2: `run_hsort_tests` — Heap Sort Stability

**What it tests:** `secp256k1_heapsort` — the internal O(n log n) sorting
implementation used by batch verification. Tests correct ordering on various
input patterns (already sorted, reversed, duplicates, random).

**Why it matters:** Batch verification depends on correct sorting of signatures
by public key hash. A sorting bug could produce false batch verification results
without detection.

**Coverage in CAAS:** ❌ None.

**Porting effort:** Low (~2 hours). A ~100-line test exercising the heap sort
implementation on boundary inputs.

---

### GAP-3: `run_pubkey_comparison` — Public Key Comparison

**What it tests:** `secp256k1_ec_pubkey_cmp` — ordering of compressed and
uncompressed public keys (lexicographic by x-coordinate, then y parity).

**Why it matters:** Used by batch verification for deterministic sorting and by
MuSig2 key aggregation. Incorrect comparison could break MuSig2 signing or
batch verification correctness.

**Coverage in CAAS:** ❌ None. No exploit test exercises pubkey comparison directly.

**Porting effort:** Low (~2 hours). ~50 lines of test vectors with known comparison
results.

---

### GAP-4: `run_pubkey_sort` — Public Key Sorting

**What it tests:** Sorting an array of public keys using `secp256k1_ec_pubkey_sort`
(a convenience wrapper around heapsort + pubkey_cmp).

**Why it matters:** Used by MuSig2 key aggregation ordering. Incorrect sort
→ wrong MuSig2 aggregate key → verification failure (or worse: silent fork).

**Coverage in CAAS:** ❌ None. We test MuSig2 key aggregation with correctly
ordered inputs (test_exploit_musig2_key_agg, test_exploit_musig2_ordering) but
not the sort correctness itself.

**Porting effort:** Low (~2 hours). ~80 lines of sorted/unsorted key arrays.

---

### GAP-5: `run_wnaf` — wNAF Window Method

**What it tests:** Fixed-window NAF representation correctness for scalar
multiplication. Exercises window widths 2–6, checks digit bounds,
non-adjacency property, and round-trip (wNAF → scalar → wNAF).

**Why it matters:** While our library uses GLV + Strauss (not vanilla wNAF),
the wNAF codepath may be used as a fallback or in specific configurations.
A wNAF bug could produce correct-looking-but-wrong results.

**Coverage in CAAS:** ❌ None. We test multiscalar (Strauss/Pippenger) but not
the wNAF window decomposition explicitly.

**Porting effort:** Low (~3 hours). ~150 lines of wNAF decomposition +
reconstruction tests with edge window widths.

---

### GAP-6: `run_xoshiro256pp_tests` — RNG Self-Test

**What it tests:** The xoshiro256** PRNG implementation: stateless operation,
output distribution, seed expansion, and determinism.

**Why it matters:** Libsecp256k1 uses this RNG for context randomization,
test-vector generation, and nonce verification. Our library uses RFC 6979 for
deterministic nonces, but the RNG path exists for hedged signing.

**Coverage in CAAS:** ❌ None specific. Our `test_rfc6979_vectors` covers
deterministic nonces, but not the xoshiro256** PRNG itself.

**Porting effort:** Medium (~half day). ~200 lines of chi-squared/frequency tests
for PRNG output.

---

### GAP-7 (Partial): `run_int128_test_case` — 128-bit Arithmetic Helpers

**What it tests:** `secp256k1_int128_mul`, `_add`, `_sub`, `_cond_negate`,
`_from_i64`, etc. Edge cases: overflow, underflow, sign extension, zero.

**Why it matters:** Our `field_52` implementation uses `__uint128_t` directly
(compiler built-in). The upstream tests the lib's own int128 abstraction layer.
If our code uses similar helper functions, they need similar tests.

**Coverage in CAAS:** ⚠️ Partial. We have `carry_propagation` and `field_52`
which exercise 128-bit arithmetic implicitly.

**Porting effort:** Low (~2 hours). ~80 lines of explicit int128 boundary tests.

---

## 4. Module tests_impl.h — Full Coverage ✅ (No Gaps)

| Upstream module `tests_impl.h` | Our coverage | Status |
|---|---|---|
| `run_nonce_function_bip340_tests` | `bip340_vectors`, `bip340_strict` | ✅ |
| `run_schnorrsig_tests` | `ecdsa_schnorr`, `schnorr_edge_cases`, `schnorr_bip340_kat` | ✅ |
| `run_recovery_tests` | `exploit_ecdsa_recovery`, `exploit_recovery_extended`, `exploit_ct_recov`, `exploit_recoverable_sign_ct` | ✅ |
| `run_extrakeys_tests` | `c_abi_negative`, `exploit_seckey_arith`, `exploit_seckey_tweak_cancel` | ✅ |
| `run_ecdh_tests` | `ecdh_recovery_taproot`, `exploit_ecdh`, `exploit_ecdh_degenerate`, `exploit_ecdh_variants` | ✅ |
| `run_musig_tests` | `musig2`, `exploit_musig2_*` (8 files), `musig2_bip327_vectors` | ✅ |

**Note:** Our MuSig2 coverage significantly exceeds upstream (8 exploit PoCs +
BIP-327 vector tests vs. upstream's single `run_musig_tests`).

---

## 5. Bitcoin Core test_bitcoin — What We Don't Port

The 693 `test_bitcoin` tests pass through the shim, which means **all code paths
exercised by those tests are covered** during our integration validation. However,
these test categories exist **inside Bitcoin Core** and are **out of scope for
CAAS** (they test Bitcoin Core business logic, not library primitives):

| test_bitcoin category | Why NOT to port to CAAS |
|---|---|
| `script_tests` | Bitcoin Script interpreter — tests Core logic, not library crypto |
| `tx_validation_tests` | Transaction validation rules — Core logic |
| `tx_mempool_tests` | Mempool policy — Core logic |
| `wallet_tests` | Wallet RPC/db — Core logic |
| `wallet_crypto_tests` | Wallet key encryption — uses AES, not secp256k1 |
| `net_tests` | P2P networking — Core logic |
| `rpc_tests` | RPC interface — Core logic |
| `miner_tests` | Mining — Core logic |
| `policy_fee_tests` | Fee estimation — Core logic |
| `consensus_tests` | Consensus rules — Core logic |
| `blockfilter_index_tests` | BIP-158 index — Core logic |
| `descriptor_tests` | Descriptor parsing — partially relevant |
| `psbt_tests` | PSBT — partially relevant (we have `exploit_psbt_input_confusion`) |
| `bip324_tests` | Transport protocol — our crypto layer is tested separately |

**Tests that ARE relevant to CAAS but live in Bitcoin Core:**

| Category | Current CAAS coverage | Action |
|---|---|---|
| DER encoding round-trip | `exploit_ecdsa_der_confusion` — good | Already covered |
| BIP-341/342 Taproot signing | `exploit_taproot_*` — 4 files | Already covered |
| ElligatorSwift BIP-324 | `exploit_ellswift*` — 4 files | Already covered |
| BIP-352 Silent Payments | `exploit_bip352_*` — 3 files | **Exceeds Core** (Core doesn't have BIP-352 yet) |

---

## 6. Recommended Porting Plan (Priority Order)

### Tier 1 — Port Now (Before Bitcoin Core PR)

| # | Test | Priority | Effort | Owner |
|---|---|---|---|---|
| 1 | `run_scratch_tests` | **HIGH** — allocator bugs are silent | 0.5d | port from tests.c |
| 2 | `run_pubkey_comparison` | **HIGH** — affects MuSig2 + batch verify | 2h | port from tests.c |
| 3 | `run_pubkey_sort` | **HIGH** — affects MuSig2 key agg | 2h | port from tests.c |

### Tier 2 — Port Soon (Before CAAS v3.0)

| # | Test | Priority | Effort | Owner |
|---|---|---|---|---|
| 4 | `run_hsort_tests` | MEDIUM — sorting correctness | 2h | port from tests.c |
| 5 | `run_wnaf` | MEDIUM — fallback codepath | 3h | port from tests.c |
| 6 | `run_int128_tests` explicit | MEDIUM — boundary coverage | 2h | augment carry_propagation |

### Tier 3 — Port Eventually

| # | Test | Priority | Effort | Owner |
|---|---|---|---|---|
| 7 | `run_xoshiro256pp_tests` | LOW — RNG for hedged sigs | 0.5d | new test file |
| 8 | `run_ecmult_const_tests` (detailed) | LOW — already covered implicitly | — | skip (redundant) |

---

## 7. Estimation Summary

| Metric | Value |
|---|---|
| Upstream test categories compared | 56 `run_*` + 6 module tests |
| Fully covered by CAAS | 55 of 62 (89%) |
| Gap tests (explicit coverage missing) | **7** (see §3) |
| Porting effort (Tier 1) | **~1 day** |
| Porting effort (Tier 1 + Tier 2) | **~1.5 days** |
| Porting effort (all tiers) | **~2.5 days** |
| New exploit test files to create | **6** |
| ALL_MODULES entries to add | **6** |
| Upstream Wycheproof KAT suites matched | **11 vs. upstream's 1** (we exceed) |
| Exploit PoCs vs. upstream standard tests | **207 vs. ~50** unique test functions |

---

## 8. Bottom Line

**CAAS already exceeds upstream libsecp256k1's test coverage in breadth and depth**
for security-critical paths: side-channel analysis, exploit PoCs, Wycheproof vectors,
GPU backends, formal verification, and protocol-level adversarial testing.

However, there are **7 focused gaps** in upstream `tests.c` that exercise
utility functions (scratch allocator, heapsort, pubkey comparison/sort, wNAF,
PRNG, int128 helper) which our suite does not directly test. These are
**low-effort to port** (~1 day for Tier 1) and would close the last remaining
coverage gaps relative to the upstream test suite.

**No Bitcoin Core `test_bitcoin` tests need porting to CAAS** — those test
Bitcoin Core business logic and protocol rules, not library primitives. The
693 passing shim tests provide sufficient coverage validation at the integration
level.
