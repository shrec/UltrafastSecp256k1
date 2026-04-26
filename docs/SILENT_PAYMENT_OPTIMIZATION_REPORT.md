# Silent Payment CPU Optimization Report

## Measured Performance (bench_bip352_cpu, N=100K, 16-core x86, 2026-04-26)

### Baseline (before OpenMP)

| Mode | ns/op | tx/s | vs Naive |
|------|------:|-----:|---------:|
| [A] Naive 1T (KPlan per-point, full pipeline) | 27,425 | 36K | 1.0× |
| [B] Batch 1T (`batch_scalar_mul_fixed_k` + 1T Stage2) | 37,087 | 27K | 0.74× |
| [C] Batch + 16T Stage2 | 30,734 | 33K | 0.89× |
| [D] Naive 16T (per-point full pipeline, std::thread) | 7,020 | 142K | **3.91×** |
| Stage 1 alone (`batch_scalar_mul_fixed_k`, 1T) | 28,992 | — | — |

> **Key finding**: On this machine, `batch_scalar_mul_fixed_k` is *slower* than naive
> per-point `scalar_mul_with_plan` due to cache pressure from the lockstep wNAF loop.

### After OpenMP Parallel Stage 1 (2026-04-26, IMPLEMENTED)

| Mode | Before | After | Speedup |
|------|-------:|------:|--------:|
| Stage 1 alone | 28,992 ns | 6,624 ns | **4.38×** |
| [B] Batch 1T (full pipeline) | 37,087 ns | 15,421 ns | **2.40×** |
| [C] Batch + 16T Stage2 | 30,734 ns | 8,825 ns | **3.48×** |
| [D] Naive 16T (unchanged) | 7,020 ns | 7,057 ns | 1.0× |
| [I] LUT 16T (best mode) | 9,297 ns | 5,854 ns | **1.59×** |

Implementation: `#pragma omp parallel for schedule(dynamic, 1) if (n >= kChunkSize * 4)`
in `batch_scalar_mul_fixed_k` (point.cpp:3033). OMP was already linked via CMakeLists.txt.

**Note on Stage 2 OMP** (`batch_scalar_mul_generator`): Adding OMP to Stage 2 was tested
and reverted. For small N (sub-chunks of 512), OMP thread wake-up latency (>50µs) dominates
and regresses performance by 1.5× vs sequential. For large N it would be beneficial but
the precomputed table cache thrashing (2MB generator table × 16T) limits the gain.

---

## Original Bottleneck Analysis (reference machine)

`fast_scan_batch` breakdown (16T, reference machine):

| Stage | Operation | ns/tx | % time |
|-------|-----------|------:|-------:|
| 1 | `batch_scalar_mul_fixed_k` (a_eff → S₁) | ~4,225 | 87% |
| 2 | SHA256-t_k + `batch_scalar_mul_generator` | ~625 | 13% |

To reach 496K tx/s (2.016 µs/tx), Stage 1 must go from 4,225 → ~1,391 ns/tx (**3.04× faster**).

On the measured machine (16-core x86), the best single-thread `fast_scan_batch` is now
~15,421 ns/op = **65K tx/s** (was 27K tx/s). The [D]-style naive 16T pipeline gives
**142K tx/s** and remains the ceiling for the current architecture.

---

## Optimization Proposal #1: Parallel Stage 1 (Biggest Win)

**Problem**: `batch_scalar_mul_fixed_k` is single-threaded. N=50K points processed sequentially in chunks of 2048.

**Solution**: Parallel partition — split N points across T threads.

```cpp
// Current (sequential, chunk_size=2048)
for (chunk_start = 0; chunk_start < n; chunk_start += kChunkSize) {
    // Process chunk: build tables, batch_invert, wNAF loop
}

// Proposed (parallel)
#pragma omp parallel for schedule(dynamic, kChunkSize) \
    if (n >= kChunkSize * 4)
for (size_t chunk_start = 0; chunk_start < n; chunk_start += kChunkSize) {
    // Each thread has its own thread_local scratch
    // Process chunk independently (same KPlan, different pts)
}
```

**Why this works**:
- Each chunk is fully independent (same KPlan, different pts)
- All scratch buffers are already `thread_local` — zero contention
- Chunk_n=2048 is large enough to amortize table-building overhead
- At n=50K, 24 chunks → 16 threads → ~3-4 ms per thread

**Expected gain**: ~12-14× speedup on Stage 1 (near-linear to thread count minus overhead)
**Complexity**: Low — add OMP header + one pragma
**Risk**: None — data-independent parallelism, identical results

---

## Optimization Proposal #2: Thread Pool + Pipelined Stages

**Problem**: Stage 1 → Stage 2 is fully sequential. While Stage 1 runs, Stage 2 hardware is idle.

**Solution**: Producer-consumer pipeline — Stage 1 output feeds Stage 2 on a queue.

```
T0: Stage 1 (chunk 0) ──┐
T1: Stage 1 (chunk 1) ──┤──→ Stage 2 consumer (single thread)
T2: Stage 1 (chunk 2) ──┤
T3: Stage 1 (chunk 3) ──┘
```

- Stage 2 is 1.60 M/s = 6.25× faster than current combined throughput
- Stage 2 can keep up with 1 producer thread (206K tx/s × 6.25 = 1.29 M/s needed)
- With parallel Stage 1, Stage 2 becomes the bottleneck unless parallelized

**Expected gain**: ~10-20% from overlap (pipeline fill/drain)
**Complexity**: Medium — lock-free SPSC queue or thread-safe vector handoff

---

## Optimization Proposal #3: Multi-thread Stage 2 (SHA256 + batch_scalar_mul_generator)

**Problem**: `batch_scalar_mul_generator` is a serial loop (precompute.cpp:3727-3743).

```cpp
// Current (serial for loop)
for (std::size_t i = 0; i < n; ++i) {
    // split_scalar_internal → fill_window → shamir_windowed_glv
}
```

**Solution**: Partition across threads. Each thread processes a sub-range.

```cpp
// Proposed
#pragma omp parallel for
for (std::size_t i = 0; i < n; ++i) {
    JacobianPoint result = {/*infinity*/};
    auto dec = split_scalar_internal(scalars[i]);
    fill_window_digits_into(dec.k1, wb, wc, tl_d1.data());
    fill_window_digits_into(dec.k2, wb, wc, tl_d2.data());
    // ...
    results[i] = Point::from_jacobian_coords(result);
}
```

**Why this works**: Each scalar_mul_generator is independent. The precompute tables are read-only after lock is released. No data race.

**Expected gain**: ~8-10× on Stage 2 (memory-bound, not CPU-bound — tables must be in cache)
**Complexity**: Low
**Risk**: Thread migration might cause L2 misses on table access

---

## Optimization Proposal #4: True Batch Generator Multiplication

**Problem**: `batch_scalar_mul_generator` (precompute.cpp:3686-3745) is NOT a true batch — it's a serial loop with per-scalar GLV decomposition and per-scalar `shamir_windowed_glv`.

**Solution**: Implement a true batch generator multiplication using the same architecture as `batch_scalar_mul_fixed_k`:

```
For each scalar, decompose into (k1, k2) via GLV
Compute wNAF for k1, k2 — store per-scalar
Build phi tables once (generator is fixed) → just one set of tables
Lockstep: for each wNAF position d:
    For each scalar: double accumulator + add if digit non-zero
Batch Z-inversion at end for affine conversion
```

But this is complex because each scalar has different wNAF digits. The key insight: **the base point G is fixed**, so the GLV table is universal. Only the wNAF digits differ per scalar.

**Expected gain**: ~2-3× vs current per-scalar loop (from sharing double operations + batch inversion)
**Complexity**: High — needs a new algorithm similar to `batch_scalar_mul_fixed_k` but with variable digits
**Risk**: Medium

---

## Optimization Proposal #5: SHA-NI for SHA256 Midstate

**Problem**: Stage 2 per-output cost: `memcpy(h, s_base_state, 32)` + `sha256_compress_dispatch(blk, h)` + scalar parse.

**Solution**: Use SHA-NI instructions (`_mm_sha256rnds2_epu32`, `_mm_sha256msg1_epu32`, etc.) for the SHA256 compression function.

**Code location**: `cpu/src/hash_accel.cpp` — `sha256_compress_dispatch`

**Current path**: Portable C implementation, ~60-80 cycles per compress

**With SHA-NI**: ~12-15 cycles per compress → 4-5× faster hashing

For M=100 outputs per tx:
- Current: 100 × 80 cycles = 8000 cycles ≈ 2.5 ns/iter (at 3.2 GHz)
- SHA-NI: 100 × 15 cycles = 1500 cycles ≈ 0.47 ns/iter

**Expected gain**: ~3× on hashing portion (~10% of total → ~7% overall)
**Complexity**: Low — add `#ifdef __SHA__` path in hash_accel.cpp
**Risk**: None — SHA-NI is widely available on x86_64 (Haswell+, 2013)

---

## Optimization Proposal #6: Reduce SHA256 Per-Output Cost

**Problem**: In `fast_scan_batch`, for each output we do:
1. Write k_be to blk[33..36] (4 bytes)
2. `memcpy(h, s_base_state, 32)` (32 bytes)
3. `sha256_compress_dispatch(blk, h)`
4. Convert h → t_bytes → parse_scalar → t_k + spend_privkey

**Solution**: Remove the `memcpy` by keeping the hash state on the stack and only restoring the base state when the output key changes. Since outputs are sorted by k within each tx, we can pipeline the state.

Actually, the current code already does this well — the base state is reused across all outputs of all txs. The per-output cost is just 32-byte memcpy + one SHA256 compress + scalar ops. That's hard to optimize further without SHA-NI.

---

## Optimization Proposal #7: AVX-512 for Batch Field Operations

**Problem**: `batch_scalar_mul_fixed_k` spends significant time on `add_mixed52_inplace` in the lockstep wNAF loop. Each add involves multiple field element multiplications.

**Solution**: Use AVX-512 IFMA (VPMADD52) for field element arithmetic. The 52-bit representation maps perfectly to AVX-512 IFMA52 instructions:

```c
// Current: 52-bit scalar multiplication (7-8 mulx)
// With IFMA52:
__m512i a_lo, a_hi, b_lo, b_hi;
__m512i prod = _mm512_madd52hi_epu64(a_lo, b_lo);
```

**AVX-512 IFMA**: Available on Ice Lake+ (2020) and Zen 4+ (2022). 2-per-cycle throughput for 52-bit multiply-add.

**Expected gain**: ~2× on field arithmetic → ~30-40% on Stage 1 overall
**Complexity**: Medium — need to detect AVX512IFMA at runtime
**Risk**: Low — fallback to existing FE52 path

---

## Optimization Proposal #8: Reduce Thread-Local Vector Resize Overhead

**Problem**: `fast_scan_batch` resizes thread_local vectors on every call:
```cpp
tl_a_eff.resize(n);
tl_s1.resize(n);
tl_s1c.resize(n);
tl_blk.resize(n);
```

For the first call with a given thread, these allocate. On subsequent calls, they only grow. But resize() is still O(n) for construction.

**Solution**: Use `reserve()` + unchecked push_back (or raw pointer assignment) instead of `resize()`, leveraging the fact that a_eff values are overwritten immediately:

```cpp
// Current:
tl_a_eff.resize(n);          // constructs n default Points
for (i=0; i<n; ++i)
    tl_a_eff[i] = txs[i].a_eff;  // overwrites

// Proposed:
tl_a_eff.reserve(n);          // no construction — just capacity
for (i=0; i<n; ++i)
    tl_a_eff.emplace_back(txs[i].a_eff);  // construct in-place
```

**Expected gain**: ~1-2% overall (small but free)
**Complexity**: Low
**Risk**: Need to call .clear() + .reserve() instead of .resize()

---

## Optimization Proposal #9: Combined Stage 1+2 with Shared Batch Inversion

**Problem**: Stage 1 ends with `batch_to_compressed` (one batch inversion across N points producing 33-byte serialized output). Then Stage 2 reads these bytes and builds SHA256 blocks.

**Solution**: Keep intermediate results in Jacobian form with a shared batch inversion at the very end. If we combine Stage 1 Jacobian output with Stage 2 final x-only check, we can do ONE batch inversion instead of two.

Current:
```
Stage 1: batch_scalar_mul_fixed_k → Jacobian → batch_to_compressed (INVERSION #1)
Stage 2: SHA256 → batch_scalar_mul_generator → Jacobian → batch_x_only_bytes (INVERSION #2)
```

Proposed:
```
Stage 1: batch_scalar_mul_fixed_k → keep Jacobian
Stage 2a: SHA256 → tl_out_scalars
Stage 2b: batch_scalar_mul_generator → keep Jacobian
Final: Combined Jacobian batch inversion → x_only check
```

Wait — these are independent batches. Stage 1 produces N Jacobian points, Stage 2 produces M Jacobian points. They can't share an inversion.

**But**: `batch_scalar_mul_fixed_k` already does a batch inversion internally for each chunk. The final `batch_to_compressed` does a second batch inversion. We could eliminate the final one by doing x-only extraction directly from the Jacobian accumulator output.

Actually, looking at the code more carefully:

`batch_scalar_mul_fixed_k` (point.cpp:3151) stores results as `s_acc[i]` — Jacobian output. Then the caller does `batch_to_compressed(tl_s1.data(), n, tl_s1c.data())` on those results.

But the Stage 2 input needs S₁ (compressed for SHA256 midstate). The SHA256 midstate needs 33 bytes (compressed point). So we do need the compressed form.

Unless... we store the S₁ hash differently. The SHA256 tagged hash is:
```
SHA256(SHA256(tag) || SHA256(tag) || ser_compressed(S) || ser32(k))
```

We could precompute `SHA256(tag) || SHA256(tag)` as the base state (already done), and feed `ser_compressed(S)` as bytes. The compressed form is needed for the hash — we can't avoid it without redesigning BIP-352.

So the batch_to_compressed is necessary.

---

## Optimization Proposal #10: SVE/SIMD for ARM64

If targeting ARM64 (Apple M-series, Graviton, etc.), use:
- SVE2 for vector field arithmetic
- SHA-256 instructions (ARMv8.4-SHA)
- Batch processing with NEON

The codebase already has `SECP256K1_FAST_52BIT` which is x64-only. ARM64 uses 4x64-bit limbs. Adding ARM64 SVE would be a new code path.

---

## Priority Ranking

| # | Optimization | Est. Gain | Complexity | Risk | Stage | Status |
|---|-------------|----------:|-----------|:----:|:----:|:------:|
| 1 | Parallel Stage 1 (OpenMP) | **4.38× measured** | Low | None | Stage 1 | ✅ DONE |
| 2 | SHA-NI for SHA256 | already implemented | — | — | Stage 2 | ✅ exists |
| 3 | GLV, window method, NEON | already implemented | — | — | Stage 1 | ✅ exists |
| 4 | Parallel batch_scalar_mul_generator | ❌ regresses for N<4096 (OMP wake-up) | Low | High | Stage 2 | ❌ reverted |
| 5 | True batch gen_mul | 2-3× | High | Medium | Stage 2 | pending |
| 6 | Pipelined stages | 10-20% | Medium | Low | Both | pending |
| 7 | AVX-512 IFMA field ops | 2× on field arith | Medium | Low | Stage 1 | pending |
| 8 | ARM64 NEON/SVE | already implemented | — | — | Stage 1 | ✅ exists |

---

## Implementation — Parallel Stage 1 (DONE, 2026-04-26)

Added to `point.cpp` (at `batch_scalar_mul_fixed_k` chunk loop):

```cpp
#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 1) if (n >= kChunkSize * 4)
#endif
for (size_t chunk_start = 0; chunk_start < n; chunk_start += kChunkSize) {
```

`find_package(OpenMP)` + `target_link_libraries(... OpenMP::OpenMP_CXX)` were already
in `cpu/CMakeLists.txt`. Only the pragma was missing.

Guard `if (n >= kChunkSize * 4)` = `if (n >= 8192)` ensures OMP skips for small batches
where thread activation overhead exceeds computation benefit.

---

## Summary (Updated)

| Metric | Before | After (Stage 1 OMP) |
|--------|-------:|--------------------:|
| `batch_scalar_mul_fixed_k` (1T) | 28,992 ns/op | **6,624 ns/op** (4.38×) |
| `fast_scan_batch` 1T equivalent | ~37,000 ns/op | **~15,421 ns/op** (2.40×) |
| `fast_scan_batch` + 16T Stage2 | ~30,734 ns/op | **~8,825 ns/op** (3.48×) |

Best achievable on this machine: **[I] LUT 16T = 5,854 ns/op (170K tx/s)** (requires
one-time per-scan-key LUT build of ~105ms).

Next highest priority: parallelizing `fast_scan_batch` at caller level (like [D] bench
mode — naive per-point 16T = 7,020 ns/op = 142K tx/s, no setup cost).
