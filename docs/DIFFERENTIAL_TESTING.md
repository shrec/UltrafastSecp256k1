# Differential Testing Methodology

**UltrafastSecp256k1 v3.14.0** -- Cross-Library Verification Protocol

---

## Overview

Differential testing is the primary correctness guarantee for this library.
We link **both** UltrafastSecp256k1 and Bitcoin Core's libsecp256k1 (v0.6.0)
in the same process and compare outputs for identical inputs, bit-for-bit.

If both libraries agree on all operations, they implement the same elliptic
curve math. This is the gold-standard correctness check.

---

## Test Matrix

| Suite | Operation | Rounds (xM) | Comparison Method |
|-------|-----------|-------------|-------------------|
| [1] Pubkey Derivation | k -> k*G | 500xM | Compressed + uncompressed byte-exact |
| [2] ECDSA UF->Ref | Sign(UF), Verify(Ref) | 500xM | Ref library accepts UF's signature |
| [3] ECDSA Ref->UF | Sign(Ref), Verify(UF) | 500xM | UF library accepts Ref's signature |
| [4] Schnorr BIP-340 | Bidirectional sign/verify | 500xM | Both accept other's signatures |
| [5] RFC 6979 | ECDSA compact byte match | 200xM | Byte-exact r‖s comparison |
| [6] Edge Cases | k=1, k=2, k=n-1, 2^i | 256+3 | Reference-checked |
| [7] Point Addition | a*G + b*G | 200xM | Compressed byte-exact |
| [8] Schnorr Batch | Batch verify 16-sig batches | 50xM | Valid batch + corrupted rejection |
| [9] ECDSA Batch | Batch verify 16-sig batches | 50xM | Valid batch + corrupted rejection |
| [10] Extended Edges | n-2, P+P, mutation, negation | 550+xM | See subsections below |

**M** = multiplier (default: 1 for CI, 100 for nightly = **1.3M+ checks**).

---

## Multiplier System

```
Default (CI):    M=1    ->  ~7,860 checks per push
Nightly:         M=100  ->  ~1,310,000 checks (3 AM UTC daily)
Manual trigger:  M=N    ->  custom (workflow_dispatch)
```

### CI (every push)
```yaml
# .github/workflows/ci.yml
- name: Run cross-library test
  run: ./build/cpu/test_cross_libsecp256k1
  # M=1, ~8K checks in <10s
```

### Nightly (extended)
```yaml
# .github/workflows/nightly.yml -- differential job
env:
  DIFFERENTIAL_MULTIPLIER: 100  # ~=1.3M checks
run: ./build/cpu/test_differential_standalone "${DIFFERENTIAL_MULTIPLIER}"
```

### Manual
```bash
# Run with arbitrary multiplier
./build/cpu/test_cross_libsecp256k1 200  # 200x = ~2.6M checks
```

---

## Edge-Case Corpus

### Built-in (Suite [6] + [10])

| Input | Description | Why It Matters |
|-------|-------------|---------------|
| k = 1 | Generator point | GLV decomposition base case |
| k = 2 | Simplest doubling | P+P formula correctness |
| k = n-1 | Maximum valid scalar | -G, near-overflow |
| k = n-2 | Near max | Second-to-last valid |
| k = (n-1)/2 | Half-order | Middle of scalar range |
| k = 2^i (i=0..255) | Powers of two | Single-bit scalars, window alignment |
| P + P | Point doubling vs 2xP | Complete addition |
| k*G + (-k)*G | Negation to infinity | Infinity handling |
| (k+1)*G == k*G + G | Consecutive scalars | Additive structure |
| Signature mutation | Bit-flip in r[0] | Rejection correctness |

### Fuzz Corpus (31 pinned inputs)

```
tests/corpus/MANIFEST.txt -- 31 pinned regression inputs:
  scalar/     -- edge scalars (near-n, all-0xFF, near-zero)
  schnorr/    -- zero signatures, malformed sigs
  pubkey/     -- prefix variations, zero coordinates
  address/    -- encoding edge cases
  bip32/      -- overflow index, deep paths
  ffi/        -- zero privkey, null inputs
```

---

## Batch Verification Cross-Check

### Purpose

Batch verify is an optimization (multi-scalar multiplication).
If it silently drops invalid signatures or rejects valid ones,
the consequences are critical. Cross-checking catches:

1. Valid batch accepted by both individual and batch verify
2. Corrupted batch rejected by both individual and batch verify
3. Batch consistency: batch accepts ⟺ all individuals accept

### Method

```
For each batch of 16 signatures:
  1. Sign all 16 with UF
  2. Verify each individually with libsecp256k1 <- cross-library
  3. Batch verify all 16 with UF <- batch correctness
  4. Corrupt one signature, batch verify again <- rejection check
```

---

## Statistical Confidence

At **M=100** (nightly):

| Suite | Checks | Failure probability if bug exists |
|-------|--------|-----------------------------------|
| Pubkey | 50,000 | < 2^{-50000} |
| ECDSA sign+verify | 100,000 | < 2^{-100000} |
| Schnorr | 50,000 | < 2^{-50000} |
| Batch (16x50x100) | 80,000 | < 2^{-80000} |
| Edge cases | ~26,000 | deterministic |
| **Total** | **~1,310,000** | **~=0** |

After 1M+ random inputs with identical outputs, the probability of a
latent arithmetic bug is astronomically small.

---

## Determinism

- **Fixed seed**: 42 (all runs produce identical random values)
- **No external entropy**: `std::mt19937_64(42)` only
- **Bit-exact reproducibility**: same binary -> same checks -> same pass/fail

This means:
- Failures are always reproducible
- CI and local results are identical
- No flaky tests possible

---

## Reference Library

| Property | Value |
|----------|-------|
| Library | bitcoin-core/secp256k1 |
| Version | v0.6.0 |
| Build | Static, linked into same binary |
| API | C (secp256k1.h, secp256k1_schnorrsig.h, secp256k1_extrakeys.h) |

---

## How to Run

```bash
# Build with cross tests enabled
cmake -S . -B build -DSECP256K1_BUILD_CROSS_TESTS=ON
cmake --build build --target test_cross_libsecp256k1 -j$(nproc)

# Default run (M=1, ~8K checks)
./build/cpu/test_cross_libsecp256k1

# Extended run (M=100, ~1.3M checks)
./build/cpu/test_cross_libsecp256k1 100

# Via CTest
ctest --test-dir build -R cross_libsecp
```

---

## Verification Checklist

- [x] All 10 suites pass at M=1 (CI)
- [x] All 10 suites pass at M=100 (nightly)
- [x] RFC 6979 byte-exact match for ECDSA
- [x] Schnorr BIP-340 bidirectional verify
- [x] Batch verify cross-checked against individual verify
- [x] Edge cases: k=1, k=2, k=n-1, k=n-2, k=(n-1)/2, 2^i
- [x] Point identity: k*G + (-k)*G = O
- [x] Signature mutation always rejected
- [x] Deterministic seed for reproducibility
- [x] libsecp256k1 v0.6.0 pinned

---

## Related Documents

| Document | Purpose |
|----------|---------|
| [AUDIT_READINESS_REPORT_v1.md](AUDIT_READINESS_REPORT_v1.md) | Verification transparency report |
| [INTERNAL_AUDIT.md](INTERNAL_AUDIT.md) | Full internal audit results |
| [CT_VERIFICATION.md](CT_VERIFICATION.md) | Constant-time methodology |
| [TEST_MATRIX.md](TEST_MATRIX.md) | Full test coverage matrix |

---

*UltrafastSecp256k1 v3.14.0 -- Differential Testing Methodology*
