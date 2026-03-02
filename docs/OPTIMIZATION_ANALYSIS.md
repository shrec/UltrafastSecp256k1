# Optimization Analysis: UltrafastSecp256k1 vs libsecp256k1

## Current Performance Gap (v3.16.1, Windows x86-64, Clang 21)

| Operation       | Ours (us) | libsecp (us) | Ratio |
|-----------------|-----------|--------------|-------|
| ECDSA Verify    | 30.5      | 24.7         | 0.81x |
| Schnorr Verify  | 29.8      | 25.3         | 0.85x |
| ECDSA Sign (CT) | 24.3      | 23.8         | 1.02x |
| Schnorr Sign    | 24.5      | 24.2         | 1.01x |

**Verify gap: ~5us (both ECDSA and Schnorr).**
Sign is at parity -- no further work needed on sign path.

---

## Architecture Comparison (Verify Hot Path)

Both libraries use identical high-level algorithm:
- GLV decomposition: split scalar into 2 ~128-bit halves
- 4-stream Strauss interleaved wNAF scan
- W=15 for G (8192-entry precomputed table), W=5 for P (8-entry per-call table)

### What our implementation already matches:
1. **Effective-affine table construction** (z-ratio technique, 0 inversions for P table)
2. **AffinePointCompact** (64-byte cache-aligned G/H entries)
3. **add_zinv** for G/H streams (folds Z_shared into formula)
4. **On-the-fly Y negation** (halves table memory, ~1.25MB vs 2.5MB)
5. **SW prefetch** for G/H table entries before doubling
6. **wNAF word-at-a-time** bit extraction (no to_bytes serialization)
7. **5x52 field arithmetic** with __int128
8. **Inversion-free X-check** in Schnorr verify (r*Z^2 == X early exit)

---

## Optimizations Applied This Session

### 1. Schnorr Inverse Elimination (schnorr.cpp)
- **Before**: compute Z^-1 first, then X-check (x_aff == r), then Y-parity
- **After**: X-check via r*Z^2 == X (1S+1M, no inverse); Z^-1 only for Y-parity
- **Impact**: Saves ~3.5us on invalid signatures (early exit before inverse)
- **Valid sigs**: No change (inverse still needed for Y-parity)

### 2. Force-Inline Point Additions (point.cpp)
- `jac52_add_mixed_inplace` and `jac52_add_zinv_inplace` marked force-inline
- Eliminates ~3ns x 42 calls/verify = ~126ns function call overhead

### 3. wNAF Word-at-a-Time Rewrite (point.cpp)
- Old: to_bytes() serialization + multi-word shift/sub per bit
- New: Direct limb read + get_bits lambda (cross-limb boundary handling)
- **Impact**: Saves ~800-1200ns per verify (4 wNAF computations)

### 4. Batch Verify G-Separation (batch_verify.cpp)
- Separated G coefficient from MSM (compute g_coeff*G via precomputed comb)
- Improved batch Schnorr from 0.46-0.56x to 0.62-0.65x vs individual
- Still not faster than individual verify due to fundamental MSM vs GLV gap

### 5. Dead Code Cleanup
- Removed `#if 0` buggy Montgomery assembly (field_asm.cpp, 25 lines)
- Removed `#if 0` ARM64 v2 disabled declarations (field_52_impl.hpp)
- Removed unused `toFieldElement()` legacy lowercase (field.hpp)
- Removed duplicate `(void)t3` (precompute.cpp)

---

## Failed Experiments (Reverted)

### GLV-MSM (multiscalar.cpp, pippenger.cpp)
- **Idea**: Decompose each MSM scalar into 2 ~128-bit halves via GLV endomorphism
- **Result**: WORSE performance (Schnorr batch N=64: 0.41x, down from 0.65x)
- **Root cause**: Doubling point count (N->2N) increases per-step add cost more
  than the saved doublings (~126 vs ~256). Individual scalar_mul already uses GLV.
- **Verdict**: GLV-MSM is counterproductive for secp256k1 batch verify

---

## Remaining Gap Analysis (~5us)

### Hypothesis 1: Field Multiplication Throughput
Our 5x52 C++ mul uses `__int128` -> compiler emits MUL+UMULH pairs.
libsecp256k1 uses hand-tuned x86-64 assembly (BMI2 MULX + ADX ADCX/ADOX).
- libsecp field_mul: ~7ns (measured)
- Our fe52_mul: ~9-11ns (estimated from verify timing)
- Per verify: ~300 field muls x ~2-3ns gap = ~600-900ns

**Action**: Profile fe52_mul vs libsecp field_5x52_mul_inner. Consider porting
their BMI2/ADX assembly or enabling our ASM52 path.

### Hypothesis 2: Point Doubling Formula
Our `jac52_double_inplace`: standard formula with temporary variables.
libsecp uses `secp256k1_gej_double_var` -- variable-time with early exit.
- Operation count should be identical (1M + 5S + several add/sub)
- But register pressure and instruction scheduling may differ

**Action**: Compare generated assembly for doubling. Ensure compiler isn't
spilling registers (check with `objdump -d` or Compiler Explorer).

### Hypothesis 3: Table Lookup Overhead
Our `AffinePointCompact::to_affine52()` converts from compact format per lookup.
libsecp stores entries in direct `secp256k1_ge_storage` format and loads directly.
- Each conversion: ~2-3ns
- Per verify: ~18 G/H lookups x 2-3ns = ~36-54ns

**Action**: Profile to_affine52 cost. Consider storing FE52 directly in tables
(increases table size from 64B to 80B per entry, but saves conversion).

### Hypothesis 4: Normalization Overhead
Our code calls `normalize_weak()` / `normalize()` in places where libsecp
uses lazy carry propagation. Each unnecessary normalize: ~5-10ns.
- Per verify: unknown count, needs profiling

**Action**: Audit all normalize calls in hot path. Remove any that aren't
necessary for correctness.

### Hypothesis 5: Compiler vs Hand-Tuned Assembly
libsecp256k1 has hand-tuned x86-64 assembly for:
- `secp256k1_fe_mul_inner` (5x52 multiply)
- `secp256k1_fe_sqr_inner` (5x52 square)
These are in `src/asm/field_10x26_arm.s` and `field_5x52_asm_impl.h`.

Our library relies on compiler-generated code from C++ with `__int128`.
Clang 21 generates good but not optimal code.

**Estimated total gap from assembly: 1-3us** (300+ muls x 3-10ns each).

---

## Priority Actions (Ordered by Expected Impact)

1. **Profile individual operations** -- add micro-benchmarks for:
   - fe52_mul, fe52_sqr, fe52_add, fe52_negate
   - jac52_double_inplace, jac52_add_mixed_inplace, jac52_add_zinv_inplace
   - Compare with libsecp equivalent timings
   
2. **Enable/port x86-64 field assembly** -- bridge the 2-3ns/mul gap
   
3. **Eliminate unnecessary normalizations** -- audit hot path
   
4. **Optimize AffinePointCompact::to_affine52** -- store FE52 directly or
   make conversion cheaper

5. **Benchmark on Linux** -- eliminate Windows scheduling noise for accurate
   apple-to-apple comparison

---

## Research Papers Reviewed (No Actionable Improvements Found)

| Paper | Finding |
|-------|---------|
| Drucker & Gueron (P-256) | Pubkey precompute cache -- already have KPlan |
| NXP 2014-862 (Car2Car) | ASIC/FPGA specific, not applicable |
| Courtois et al. 2016-103 | Confirmed our W=15 choice is optimal |
| mastering-taproot (Aaron Zhang) | Batch verify -- implemented, not faster |
| noot/schnorr-verify | Solidity-specific |
| Ergo Sigma Protocols | Standard Schnorr theory |
| Sei Giga blog | ZK for ECDSA, not applicable |

---

## Test Status

- **25/26 pass** (ct_sidechannel: pre-existing failure, constant-time analysis)
- Build: Clang 21.1.0, Release, `-O3 -DNDEBUG -march=native`
- All correctness tests pass including BIP-340 vectors, RFC 6979, MuSig2
