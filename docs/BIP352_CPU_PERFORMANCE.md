# BIP-352 CPU performance — GitHub issue #336

Status: **candidate implementation; physical M5 closure pending**

This document supersedes the earlier x86 contention interpretation. It does not
claim that an Apple result has passed.

## Reporter evidence and root cause

Craig Raw reproduced the regression on an Apple M5 Max. His first clean
comparison at development revision `4556fa05` reported:

| path | real | user | sys |
|---|---:|---:|---:|
| v3.68 old/per-row | 12.8 s | 194 s | 1.0 s |
| dev old/per-row | 15.6–15.7 s | 214–216 s | 8.6–9.7 s |
| dev batch | 17.8–18.6 s | 218–229 s | about 43 s |

Craig then captured this exact two-query matrix on the same revision and
10,356,829-row/18-caller workload. Run 1 is the warm-up and Run 2 is the steady
query. The resource counters are the clean, unsampled invocation's
`/usr/bin/time -l` process envelope, not per-query deltas:

| measurement | old/per-row | batch |
|---|---:|---:|
| Run 1 warm-up real / user / sys | 16.33 / 225.8 / 8.89 s | 18.73 / 216.0 / 44.02 s |
| Run 2 steady real / user / sys | 16.41 / 233.9 / 8.26 s | 18.94 / 220.3 / 44.16 s |
| peak RSS | 4.40 GB | 5.17 GB |
| voluntary / involuntary context switches | 5,582 / 1,608,450 | 0 / 3,076,715 |
| page reclaims / hard faults | 396,708 / 31 | 515,685 / 25 |
| instructions retired | 7.52 T | 7.17 T |

With `sample` attached, the observed query took 18.8 s per-row and 21.2 s
batch, so those observer-affected durations are not timing baselines. The
profile showed no per-operation thread creation and negligible `mmap`/`munmap`;
the differentiating signal was publication-related `__psynch_mutexwait` and
`__psynch_mutexdrop`.

The later profile measured `__psynch_mutexwait` 8,630 times in the per-row
capture and 26,358 times in the batch capture. Stacks reached libc++
`std::__sp_mut::lock` from the deprecated free-function atomic load of
`shared_ptr<PrecomputeContext>`. Apple's implementation hashes these operations
onto a mutex pool. The former statement that this shared-pointer operation was
lock-free was therefore incorrect.

Cardinality instrumentation also exposed why the batch capture contradicted the
intended source branch. `SECP256K1_ESP32_BUILD` is always defined to either
zero or one, but the batch function tested it with `#if defined(...)`.
Consequently desktop builds selected the ESP fallback and called
`Point::generator().scalar_mul()` once per element. The fix tests the macro's
value, so desktop `n > 1` now reaches the one-acquisition context-taking path.

## Implemented publication protocol

Desktop fixed-base context state now has three distinct roles:

- `g_context_owner`: the only global `shared_ptr`, accessed under `g_mutex`;
- `g_published_context`: an acquire/release atomic raw identity token;
- `tl_context_owner`: one TLS `shared_ptr<const PrecomputeContext>` lifetime
  owner per calling thread.

The raw identity is never dereferenced as an unowned pointer. A steady reader
compares it with `tl_context_owner.get()` and dereferences the TLS owner only
when they match. On a mismatch or null publication, the reader locks, builds or
copies the current mutex-owned owner into TLS, and then resumes.

Invalidation release-stores null before dropping the global owner. Publication
fully constructs and validates a context, installs global ownership, and then
release-stores its identity. An in-flight reader can therefore finish on its old
TLS owner during reset/reconfigure; its next acquisition refreshes to the new
owner. A stale TLS owner pins its allocation, preventing pointer-address ABA.
Diagnostic epochs used by tests do not participate in lifetime correctness.

`scalar_mul_generator()` acquires once per call. For `n > 1`,
`batch_scalar_mul_generator()` acquires once before its element loop and invokes
a context-taking internal multiply helper for every element. Public cache
save/load wrappers now take `g_mutex` before calling their `_locked` helpers.
Cache load, cache rebuild, configure, cache-directory change and static-table
load all use the centralized publication/invalidation helpers.

The replacement pointer atomic has a compile-time
`is_always_lock_free` gate. The benchmark also prints both that value and the
exact target runtime `is_lock_free()` result. This is capability evidence, not
an Apple pass claim.

## Correctness and local validation contract

Focused tests use this independent oracle:

```cpp
Point const& generator = Point::generator();
Point plain = Point::from_affine(generator.x(), generator.y());
Point expected = plain.scalar_mul(k);
```

`Point::from_affine` clears the generator dispatch identity, so the oracle
cannot recurse into the fixed-base path under test. Coverage includes scalar,
`n=0`, `n=1`, odd batches, GLV/non-GLV, one and 18 readers, null/new
publication, stale-TLS refresh, cache-directory changes, cache load and
reconfigure. Test-only counters prove one acquisition per scalar or batch call,
not per batch element.

Required local gates are:

- focused Release correctness/lifecycle tests;
- GCC 14 C++20 `-Werror`, with no deprecated shared-pointer atomic diagnostic;
- focused ASan/UBSan and supported TSan runs;
- `git diff --check`;
- build/test output only below `out/.tasks/GH336_LIFETIME_FIX_016/`.

Candidate validation on the Linux review runner (2026-07-23):

| gate | result |
|---|---|
| focused Release tests | 77/77 CPU checks and 14/14 audit checks passed |
| acquisition counter | scalar delta 1; `n=67` batch delta 1 |
| ASan + UBSan focused tests | passed |
| TSan focused tests | direct CPU test passed; direct audit launch failed before `main` with GCC TSan `unexpected memory mapping`; with ASLR disabled via `setarch x86_64 -R`, both passed |
| GCC 14 C++20 `-Werror` library build | passed |
| tiny replay-driver smoke | passed; printed compile-time/runtime identity capability as 1/1 on this x86_64 host |

These results validate the candidate locally but do not replace the M5 gates
below.

## Exact M5 Max replay

Use a clean non-LTO Release build from the candidate SHA. The issue replay
defaults intentionally match Craig's application:

```text
rows=10,356,829
callers=18
window_bits=12
thread_count=1
enable_glv=false
```

Build timing against the normal, uninstrumented `fastsecp256k1` target.
Acquisition cardinality comes from the separate private
`fastsecp256k1_gh336_test_hooks` test object and must never be enabled in the
timing binary.

Run both `--mode=legacy` and `--mode=batch`, with at least two warm-ups and five
recorded steady second-query passes. Each `measured pass` line reports its own
wall time and `getrusage` deltas for user/sys, voluntary/involuntary context
switches, page reclaims and hard faults. `peak_rss_native` is the process
high-water mark at the end of that pass (bytes on Darwin, KiB on Linux), so
compare it only across separate clean mode invocations. The driver folds every
candidate prefix after every run, requires stability across passes, and runs the
other API mode after timing to require an equal full-prefix checksum.

Keep `/usr/bin/time -l` as a supporting process-envelope artifact; do not
attribute its aggregate counters to one pass when an invocation contains cold,
warm-up and multiple measured passes. Also record a separate 15-second `sample`
capture, correctness database totals, the benchmark's two atomic capability
lines, and optimized disassembly for
`scalar_mul_generator`/`batch_scalar_mul_generator`.

The disassembly must show an inline raw identity load and no call from either
entry point to a shared-pointer atomic, `__get_sp_mut`, pthread mutex helper, or
generic non-lock-free atomic helper. The sample stacks must contain no
publication-related `std::__sp_mut`/`__psynch_mutexwait` path.

Issue #336 closes only when all of these hold on Craig's M5 Max:

1. CPU correctness is 100% for the 10,356,829-row replay and the Metal
   cross-check remains unchanged.
2. Runtime raw-pointer `is_lock_free()` is true; optimized disassembly and
   sample stacks corroborate it.
3. Median per-row wall time is at most 1.05× the same-run v3.68 baseline and
   median sys time is at most 2.0× that baseline.
4. Batch median sys time is no worse than per-row by more than 20% or 1.0
   second, whichever allowance is larger, and batch wall time is no worse by
   more than 5%.
5. Candidate per-row median total context switches are no worse than v3.68 by
   more than 20% or 50,000, whichever allowance is larger. Batch is no worse
   than candidate per-row by the same allowance. Record voluntary and
   involuntary columns separately even though the threshold uses their sum.
6. Local correctness, sanitizer, race and GCC 14 gates pass.

If lock stacks disappear but a timing threshold misses, keep the issue open and
profile the remaining staging/RSS cost separately.
