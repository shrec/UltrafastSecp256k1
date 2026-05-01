# BIP-352 CPU Pipeline Optimization — Working Document
**Status: SCRATCH / DELETE BEFORE RELEASE**

---

## 0. Baseline (bench_bip352_cpu, N=100K, 16T)

```
[A] Naive  1T:   24917 ns/op   0.04 M/s
[D] Full  16T:    3762 ns/op   0.27 M/s

Per-step (single thread):
  Stage 1: b_scan × A_i   16480 ns   68.5%   ← bottleneck #1
  Stage 1b: batch compress   183 ns    0.8%
  Stage 2a: SHA256           190 ns    0.8%   ← negligible
  Stage 2b: t×G + add + compress  7199 ns  29.9%  ← bottleneck #2
```

16T efficiency: 3762 vs theoretical 24917/16=1557 → **49% efficiency**
Cause: cache thrashing (generator table + tweaks × 16 threads competing for L2/L3)

---

## 1. Mathematical Foundation

### 1.1 Pipeline per tx_i

```
Input:  A_i = Σ(input pubkeys)          ← EC point (affine, from blockchain)
        b_scan                           ← scalar, CONSTANT per wallet
        B_spend = (x_B, y_B)            ← EC point, CONSTANT, affine

Step 1: S_i = b_scan × A_i             ← scalar mul, variable base
Step 2: ser_i = compress(S_i)          ← field inversion + 33 bytes
Step 3: t_i = SHA256("BIP0352/SharedSecret" || ser_i || k)  ← scalar (mod n)
Step 4: T_i = t_i × G                  ← scalar mul, fixed base G
Step 5: P_i = B_spend + T_i            ← point addition
Step 6: match: x(P_i) == output.x ?    ← comparison
```

### 1.2 Jacobian representation issue

Jacobian point (X:Y:Z) represents affine point (X/Z², Y/Z³).

**Same affine point, different Jacobian representations:**
```
(X₁:Y₁:Z₁) ≡ (X₂:Y₂:Z₂)  iff  X₁·Z₂² = X₂·Z₁²
```

Therefore: `P.X_jacobian ≠ output.x` in general.

**Correct comparison (option A — Jacobian trick):**
```
output.x × Z² == P.X_jacobian   (1 field_mul, ~10ns)
```
Saves field_inv (~1000ns) for non-matching txs.

**Alternative (option B — affine pipeline):**
Normalize T_i = t_i×G in batch, then compute x(B_spend + T_i) using
x-only affine addition — always in affine space, no Z tracking.

### 1.3 x-only affine addition formula

Given affine points P=(x_P, y_P) and Q=(x_Q, y_Q), P≠Q, P≠-Q:
```
λ   = (y_Q - y_P) / (x_Q - x_P)
x_R = λ² - x_P - x_Q
y_R = λ(x_P - x_R) - y_P          ← skip if only x_R needed
```

**Cost per add (individual):** 1 field_inv + 3 field_muls + 2 squares + 4 adds
**Cost per add (batch N, Montgomery trick):** 3 field_muls + 1/N field_inv ≈ 3×10ns

**For B_spend + T_i where B_spend=(x_B, y_B) is CONSTANT:**
```
denom_i = x_Ti - x_B              ← 1 field subtraction (free)
λ_i     = (y_Ti - y_B) / denom_i  ← need inv(denom_i)
x_Ri    = λ_i² - x_B - x_Ti
```

Batch: invert all denom_i together → 3N muls + 1 inv.
Then: λ_i, x_Ri each need ~3 field_muls.
**Total: ~6N field_muls + 1 inv ≈ 6×10×N + 1000 ns**

For N=1000: 61,000 ns → 61 ns/op for Step 5.
vs current to_compressed: ~1000 ns/op.
**~16× speedup on Step 5.**

### 1.4 SIMD 4× scalar multiplication

b_scan wNAF digits [d_0..d_127] are CONSTANT.
Four points A_1, A_2, A_3, A_4 share the same digit sequence:

```
for j = 127 downto 0:
    d = b_scan_wnaf[j]       ← read once
    acc1 = double(acc1); if d!=0: acc1 += window1[|d|]   ← FE52 field ops
    acc2 = double(acc2); if d!=0: acc2 += window2[|d|]   ← FE52 field ops
    acc3 = double(acc3); if d!=0: acc3 += window3[|d|]   ← parallel
    acc4 = double(acc4); if d!=0: acc4 += window4[|d|]   ← parallel
```

With AVX2: 4 FE52 field multiplications fit in one 256-bit SIMD op.
Expected: ~4× throughput for Stage 1.

For t_i×G (variable scalar t_i, fixed base G):
- 4 different scalars → 4 different wNAF digit sequences
- NOT same loop → cannot use KPlan SIMD trick directly
- BUT: can vectorize the field arithmetic inside each scalar mul
- With AVX2: process 4 point doublings/additions simultaneously
  (interleave their field operations into SIMD lanes)

---

## 2. Optimization Plan

### Priority 1: x-only affine Step 5 + Jacobian trick for comparison

**Current Stage 2 (per tx, sequential):**
```cpp
CpuPoint tG = CpuPoint::generator().scalar_mul(hs);  // Jacobian result
tG.add_mixed_inplace(bs_x, bs_y);                    // Still Jacobian
auto cc = tG.to_compressed();                         // field_inv ~1000ns
prefixes[i] = extract_prefix(cc.data() + 1);         // x comparison
```

**Option A — Jacobian comparison (simpler):**
```cpp
CpuPoint tG = CpuPoint::generator().scalar_mul(hs);  // Jacobian (X:Y:Z)
// Check: output.x × Z² == tG.X  (field_mul only)
FieldElement Z2 = tG.Z52() * tG.Z52();              // Z²
for each output_x:
    FieldElement lhs = output_x * Z2;               // 1 field_mul
    if (lhs == tG.X52() + B_spend_contribution):    // compare
        → match: normalize fully
```
NOTE: B_spend + tG is not yet done — need to add B_spend first, then check.
Correct sequence:
```
P = tG + B_spend   (Jacobian add: no inv, ~12 field_muls = 120ns)
Check: output.x × P.Z² == P.X  (1 field_mul + compare)
Only normalize P if match
```
Savings: 1000ns - 120ns - 10ns = ~870ns per tx.

**Option B — Batch x-only affine (faster for large N):**
```
Phase 1: compute all T_i = t_i×G (Jacobian, no normalize) for block
Phase 2: batch_normalize all T_i → affine (x_Ti, y_Ti)
         cost: 3N field_muls + 1 field_inv
Phase 3: batch x-only add: x(B_spend + T_i) for all i
         - batch_invert all (x_Ti - x_B): 3N field_muls + 1 field_inv
         - compute λ_i, x_Ri: 3N field_muls
         - total: 9N field_muls + 2 field_inv ≈ 91ns/tx (N=1000)
Phase 4: compare x_Ri with output.x
```
Savings vs current: 7199ns → ~91ns + (t×G scalar mul time).

**Decision: Option B is better for N≥100.**

### Priority 2: LUT window size for L1 cache

Current generator table window width: check library default (likely w=7, ~150KB).

For L1 fitting (target: ≤32KB per thread):
- Each window table entry: affine point = 2 × 32 bytes = 64 bytes
- Window entries: 2^(w-1)
- Windows needed: ceil(256/w) for full scalar
- Total: 2^(w-1) × ceil(256/w) × 64 bytes

| w  | entries/window | windows | total bytes | fits in |
|----|---------------|---------|-------------|---------|
| 4  | 8             | 64      | 32,768 B    | L1 ✓   |
| 5  | 16            | 52      | 53,248 B    | L1/L2  |
| 6  | 32            | 43      | 88,064 B    | L2 ✓   |
| 7  | 64            | 37      | 151,552 B   | L2     |
| 8  | 128           | 32      | 262,144 B   | L2/L3  |

**For 16T concurrent access: effective cache per thread = L2/num_threads**
With 16T sharing 1MB L2: 64KB per thread → w=6 is max safe.
With L3 (16MB shared): ~1MB per thread → w=8 feasible but with contention.

**Action: benchmark generator_mul with w=4,5,6,7 under 16T load.**
Expected: w=4 or w=5 wins under 16T despite more iterations.

### Priority 3: SIMD 4× fixed-K scalar mul (Stage 1)

**Requires:** 4-way parallel wNAF evaluation using FE52 field ops.

**Design:**
```cpp
// Process 4 A_i simultaneously with same b_scan wNAF
void batch_scalar_mul_fixed_k_simd4(
    const KPlan& plan,
    const Point* A,    // 4 input points
    Point* out         // 4 output points
);
```

Inner loop (conceptual):
```cpp
for (int j = max_len-1; j >= 0; --j) {
    // Double all 4 accumulators (8 FE52 field_muls each = 32 total)
    dbl_inplace_simd4(acc);  // AVX2: 4× field_mul in parallel
    
    int32_t d1 = plan.wnaf1[j], d2 = plan.wnaf2[j];
    if (d1 != 0) {
        // Add precomputed window entry to all 4 accumulators
        // Each acc has different window table (built from A_i)
        // But doubling pattern is identical → branch predictor friendly
        add_window_simd4(acc, windows, d1);
    }
    if (d2 != 0) {
        add_window_simd4(acc, phi_windows, d2);
    }
}
```

**Field multiply vectorization with FE52 (AVX2):**
Each FE52 field_mul uses 5×52-bit limbs.
4 independent field_muls → 20 independent 52-bit multiplications.
AVX2 can do 4× uint64 multiply per instruction → ~5× instructions for 20 muls.
Expected: ~3-4× throughput vs scalar FE52.

### Priority 4: Multi-block batching

**When does batching K blocks together help?**

Inversion savings: K blocks → 2 inversions total vs 2K.
But batch_normalize cost is O(N) regardless → minimal savings from larger N.

**Real benefit: amortize SHA256 midstate + thread launch overhead.**
Thread launch (16T): ~100 μs.
1 block (~2.5ms compute): overhead = 4% → noticeable.
5 blocks (~12.5ms): overhead = 0.8% → negligible.

**Recommendation: batch 2-3 blocks minimum for 16T.**

---

## 3. Implementation Skeleton

### 3.1 Modified stage2_worker (Option A — Jacobian comparison)

```cpp
static void stage2_worker_jac(
    int begin, int end,
    const std::array<uint8_t, 33>* compressed,  // ser(S_i) from Stage 1
    const secp256k1::SHA256& tag_mid,
    const CpuField& bs_x, const CpuField& bs_y,  // B_spend affine
    uint64_t* out_prefixes)
{
    for (int i = begin; i < end; ++i) {
        // SHA256
        uint8_t ser[37];
        memcpy(ser, compressed[i].data(), 33);
        memset(ser + 33, 0, 4);
        auto hash = secp256k1::detail::cached_tagged_hash(tag_mid, ser, 37);
        CpuScalar t = CpuScalar::from_bytes(hash.data());

        // t×G (Jacobian, no normalize)
        CpuPoint tG = CpuPoint::generator().scalar_mul(t);  // returns Jacobian

        // Add B_spend (Jacobian + affine mixed add: ~120ns, no inv)
        tG.add_mixed_inplace(bs_x, bs_y);  // P_i = t×G + B_spend, Jacobian

        // x-only comparison via Jacobian trick
        // affine x = X/Z² → output.x × Z² == P_i.X  iff match
        // NOTE: to_compressed avoided entirely for non-matches
        // For prefix matching (probabilistic filter first):
        auto cc = tG.to_compressed();       // TODO: replace with Jac trick
        out_prefixes[i] = extract_prefix(cc.data() + 1);
    }
}
```

**TODO:** expose Jacobian X, Z fields from Point for the comparison.
Check if `X52()`, `Z52()` are accessible (they are in point.hpp).
Implement: `Z2 = Z52()*Z52(); lhs = output_x_fe * Z2; match = (lhs == X52())`.
Problem: output.x must be a FieldElement, not just bytes → parse once per output.

### 3.2 Batch x-only Stage 2 (Option B — affine pipeline)

```cpp
// Phase 2a: collect all t_i scalars
std::vector<CpuScalar> t_scalars(N);
std::vector<CpuPoint>  tG_jac(N);

for (int i = 0; i < N; ++i) {
    uint8_t ser[37];
    memcpy(ser, compressed[i].data(), 33);
    memset(ser + 33, 0, 4);
    auto hash = secp256k1::detail::cached_tagged_hash(tag_mid, ser, 37);
    t_scalars[i] = CpuScalar::from_bytes(hash.data());
}

// Phase 2b: batch generator_mul (all t_i × G → Jacobian)
for (int i = 0; i < N; ++i)
    tG_jac[i] = CpuPoint::generator().scalar_mul(t_scalars[i]);

// Phase 2c: batch normalize T_i → affine (x_Ti, y_Ti)
std::vector<CpuField> tG_x(N), tG_y(N);
CpuPoint::batch_normalize(tG_jac.data(), N, tG_x.data(), tG_y.data());

// Phase 2d: batch x-only affine add: x(B_spend + T_i)
// denom[i] = x_Ti - x_B
std::vector<CpuField> denom(N), denom_inv(N);
for (int i = 0; i < N; ++i)
    denom[i] = tG_x[i] - bs_x;  // subtraction (free: no inv)

// TODO: batch_field_invert(denom.data(), N, denom_inv.data())
// batch inverse: 3N muls + 1 inv (Montgomery trick)

// Then per-i:
// lambda_i = (tG_y[i] - bs_y) * denom_inv[i]
// x_R[i]   = lambda_i * lambda_i - bs_x - tG_x[i]
// compare x_R[i] with output.x
```

**MISSING API:** `batch_field_invert` — need to check if this exists
or if `batch_normalize` can be repurposed / a new helper written.

### 3.3 SIMD 4× KPlan evaluation (Stage 1)

```cpp
// Process 4 tweak points per iteration
void kplan_eval_simd4(
    const secp256k1::fast::KPlan& plan,
    const CpuPoint* tweaks,  // 4 points: tweaks[0..3]
    CpuPoint* results)       // 4 outputs
{
    // Build window tables for all 4 points (phi too)
    // Uses GLV: T_k = k1*A + k2*phi(A) where k1,k2 from plan
    // phi(A) = (beta*A.x, A.y) — field_mul by constant beta (cheap)

    // Build 4 × 2^(w-1) table entries each for A and phi(A)
    // Window entries: {A, 3A, 5A, ...} for wNAF
    // Can batch-normalize all 4×8 table points together (1 batch_inv)

    // Loop over wNAF digits (same for all 4):
    for (int j = plan.wnaf1_len-1; j >= 0; --j) {
        // dbl all 4 acc1, acc2, acc3, acc4 (k1 half)
        // add window entry if digit != 0
        // dbl all 4 phi_acc1...phi_acc4 (k2 half)
        // add phi window entry if digit != 0
    }
    // results[k] = acc_k + phi_acc_k for k=0..3
}
```

**Key question:** Can existing FE52 field ops be called 4× in SIMD lanes?
Check: does the library expose AVX2 field_mul primitives directly?
Alt: write thin wrapper that interleaves 4 FE52 field_muls → AVX2.

---

## 4. Expected Performance After Each Step

| After | Stage1 ns | Stage2 ns | Total 1T | Total 16T | M/s 16T |
|-------|-----------|-----------|----------|-----------|---------|
| Baseline | 16480 | 7199 | 24917 | 3762 | 0.27 |
| + Jac comparison (A) | 16480 | ~6300 | ~23000 | ~3500 | 0.29 |
| + batch x-only (B) | 16480 | ~200 | ~17000 | ~1100 | 0.91 |
| + LUT w=4 (Stage2) | 16480 | ~100 | ~16800 | ~1050 | 0.95 |
| + SIMD 4× Stage1 | ~4500 | ~100 | ~5000 | ~320 | 3.1 |
| + LUT w=4 Stage1 too | ~4000 | ~100 | ~4300 | ~270 | 3.7 |

**Target: ~3-4 M/s on 16T CPU-only**

Ruben use case:
- Daily 144K txs: 144K/3.7M = 0.039s — instant
- Full 300M txs: 300M/3.7M = 81 seconds ≈ 1.4 minutes — fully local

---

## 5. Open Questions / Decisions Needed

1. **Option A vs B for Step 5:**
   - A (Jacobian trick): simpler, moderate gain (~870ns/tx)
   - B (batch x-only affine): bigger gain (~7000ns/tx → ~90ns/tx), more complex
   - **Recommend B** if `batch_field_invert` or equivalent exists

2. **Does x-only affine add library operation (2.8B/s) apply here?**
   - User's existing `x_only_afine_add` at 2.8B/s — check its API
   - If it takes (x_P, y_P, x_Q, y_Q) → x_R: use in batch for Step 5
   - Or if it requires specific input format: adapt

3. **batch_field_invert — does it exist?**
   - Need for Option B Phase 2d
   - Check: `batch_add_affine.hpp` or `field.hpp` for batch inverse API
   - If missing: implement Montgomery batch inverse inline

4. **SIMD 4× scalar mul — library support?**
   - Check if AVX2 field_mul primitives are exposed
   - Check `field_52.hpp` / `field_asm.hpp` for vectorized ops
   - If available: 4× KPlan eval is ~2 days of work
   - If not: may need assembly/intrinsics

5. **Generator table window tuning:**
   - Need to benchmark w=4..7 explicitly under 16T load
   - `point.cpp` has `gen_fixed_mul` — find window width constant
   - Make it runtime-configurable for benchmarking

---

## 6. Next Action

**Step 1 (today):** Implement and benchmark Option B (batch x-only affine Step 5)
- Check batch_field_invert existence
- Implement batch_xonly_add helper
- Add [E] mode to bench_bip352_cpu.cpp
- Expected: Stage 2b drops from 7199ns to ~200ns

**Step 2:** Generator window benchmark (w=4,5,6 under 16T)

**Step 3:** SIMD 4× KPlan (if field primitive access feasible)

---

## 7. [E]/[F] Mode Results — batch_add_affine_x implemented

**Date: 2026-04-25**

### 7.1 [E] Mode (batch affine Stage 2, single-thread Stage 1)

Uses `batch_add_affine.hpp` which already has `batch_add_affine_x`:
- `base = B_spend` (constant affine)
- `offsets = T_i = t_i×G` (normalized from batch_normalize)
- Output: `x(B_spend + T_i)` for all i — x-coordinates only, no inversion per point

```
[E] Stage 2 breakdown (N=100K, 16T t×G):
  2b: t_i×G Jacobian (16T, no normalize):   572.7 ns
  2c: batch_normalize (1 inversion/N):       156.8 ns
  2d: pack AffinePointCompact:                 2.9 ns
  2e: batch_add_affine_x (B_spend+T_i):     151.3 ns
  2f: x to_bytes + prefix:                    1.6 ns
  Stage 2 total (excl. SHA256):             885.3 ns   ← vs 7121 ns before = 8.0× speedup

[E] total: 17983 ns/op  — same as [C] because Stage 1 is still single-thread (16573 ns/pt)
```

**Key finding: batch affine cuts Stage 2 from 7121 → 885 ns (8× speedup on Stage 2 alone).**

### 7.2 [F] Mode (parallel Stage 1 + batch affine Stage 2)

Combines [D] parallelism with [E] batch approach:
- Stage 1: 16T parallel `scalar_mul_with_plan` (Jacobian output)
- Stage 1b: `batch_to_compressed` (serial, 1 inversion amortized)
- Stage 2a: SHA256 (16T parallel)
- Stage 2b: t_i×G Jacobian (16T parallel)
- Stage 2c: `batch_normalize` (serial)
- Stage 2d: `batch_add_affine_x` (serial)

```
[F] 4123 ns/op  (0.24 M/s)  — nearly identical to [D] 4137 ns/op
```

**Key finding: [F] ≈ [D]. Batch affine does NOT help when Stage 1 is also parallelized.**

### 7.3 Why [F] ≈ [D]: Fundamental insight

Batch normalization (Montgomery trick) is beneficial when inversions are SERIAL.
When 16T are running, each thread already amortizes its per-point normalize cost
by running in parallel. The batch normalize adds a SERIAL BARRIER that costs more
(~157 ns/pt serial) than it saves (~500 ns ÷ 16T = 31 ns wall time per point).

**Rule:** Batch inversion wins ↔ operations are serial (1T or forced sequential).
         Batch inversion loses ↔ operations are already parallel (16T distributed).

### 7.4 Updated benchmark summary

```
[A] Naive  1T:    25070 ns/op     0.04 M/s
[B] Batch  1T:    24159 ns/op     0.04 M/s   (1.04×)
[C] Batch 16T:    17650 ns/op     0.06 M/s   (1.42×)   ← Stage 1 single-thread bottleneck
[D] Full  16T:     4137 ns/op     0.24 M/s   (6.06×)   ← current best
[E] BatchAff16T:  17983 ns/op     0.06 M/s   (1.39×)   ← Stage 2 fast but Stage 1 bottleneck
[F] BestCPU 16T:   4123 ns/op     0.24 M/s   (6.08×)   ← [D]≈[F], batch doesn't help here
GPU+LUT:              91 ns/op    10.96 M/s   (ref 1 GPU)
```

Validation: all modes produce identical prefix `0xeccf8f305e69b840`.

### 7.5 Revised performance table

| Optimization | Stage1 ns | Stage2 ns | Total 1T | Total 16T | M/s 16T |
|---|---|---|---|---|---|
| Baseline [A] | 16480 | 7199 | 24917 | — | — |
| Batch Stage1 [B] | 16650 | 7121 | 24159 | — | — |
| Full 16T [D] | — | — | — | 4137 | 0.24 |
| Batch affine Stage2 only [E] | 16650 | 885 | ~17900 | 17983 | 0.06 |
| Both parallel [F] | — | — | — | 4123 | 0.24 |
| **Next: SIMD 4× Stage1** | ~4000 | 885 | ~5200 | **~325** | **~3.1** |

### 7.6 Revised next actions

Batch affine ([E]/[F]) experiment is complete. Key insight: the remaining bottleneck
is Stage 1 throughput per thread (scalar_mul_with_plan cost), not inversion overhead.

**Next optimization target: Stage 1 scalar mul itself**
Options:
1. **SIMD 4× wNAF eval**: process 4 A_i per digit loop iteration with same b_scan wNAF
   Expected: 4× Stage 1 throughput → ~1035 ns/pt → total 16T ~1500 ns → 0.67 M/s
2. **Generator table window size for t×G**: reduce 16T cache pressure
   Each thread reads same large generator LUT → smaller w=4 table fits L1 (32KB)
   Expected: t×G cost reduction from cache miss savings (~30-50%)
3. **Check if `generator().scalar_mul` uses fast fixed-base path**
   If not: implement `scalar_mul_gen_fast(scalar, w=4)` with small cache-friendly table
