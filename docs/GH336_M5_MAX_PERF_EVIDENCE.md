# Issue #336 Apple M5 Max closure evidence

Status: **pending owner rerun**

No Linux result, source inspection or synthetic multiplier can close this
physical-M5 regression. Fill this record with the exact candidate SHA and raw
owner-run artifacts before closing the issue.

The candidate repairs both proven library defects: Apple's mutex-backed
free-function `shared_ptr` atomic was removed from the per-row path, and the
desktop batch path no longer enters the ESP fallback because of an always-true
`#if defined(SECP256K1_ESP32_BUILD)` condition.

## Reporter matrix before this candidate

Craig's clean M5 Max capture on `dev` at `4556fa05` used the exact fixed replay
below. It is historical evidence for the defective implementation, not a
candidate result. Run 1 was warm-up and Run 2 the steady query; the resource
rows are the unsampled `/usr/bin/time -l` envelope over both queries.

| measurement | old/per-row | batch |
|---|---:|---:|
| Run 1 warm-up real / user / sys | 16.33 / 225.8 / 8.89 s | 18.73 / 216.0 / 44.02 s |
| Run 2 steady real / user / sys | 16.41 / 233.9 / 8.26 s | 18.94 / 220.3 / 44.16 s |
| peak RSS | 4.40 GB | 5.17 GB |
| voluntary / involuntary context switches | 5,582 / 1,608,450 | 0 / 3,076,715 |
| page reclaims / hard faults | 396,708 / 31 | 515,685 / 25 |
| instructions retired | 7.52 T | 7.17 T |

The attached-sampler queries took 18.8 s and 21.2 s respectively and are not
timing baselines. The useful attribution was 8,630 versus 26,358
`__psynch_mutexwait` leaf samples and 1,276 versus 8,717
`__psynch_mutexdrop` leaf samples. No per-operation thread creation or material
`mmap`/`munmap` signal appeared.

## Fixed replay configuration

| field | required value |
|---|---|
| total rows | 10,356,829 |
| callers | 18 |
| fixed-base window | 12 |
| table threads | 1 |
| GLV | false |
| build | clean Release, non-LTO |
| modes | per-row (`legacy`) and `batch` |
| sampling | 2+ warm-ups, 5+ recorded steady second-query runs |

The replay driver prints:

```text
context_identity_is_always_lock_free=<0|1>
runtime_is_lock_free=<0|1>
```

The exact M5 Release binary must print `runtime_is_lock_free=1`. Preserve the
complete benchmark stdout rather than copying only this line.

## Environment and build receipt

- [ ] Candidate commit SHA and v3.68 baseline SHA.
- [ ] Clean worktree status for each build.
- [ ] `sw_vers`, `uname -a`, `uname -m`.
- [ ] `sysctl -n machdep.cpu.brand_string`, `hw.physicalcpu`,
      `hw.logicalcpu`.
- [ ] Compiler, linker, CMake and Ninja versions.
- [ ] Full configure/build commands and proof that LTO is disabled.
- [ ] Power mode, thermal state and meaningful concurrent workload notes.

## Correctness receipt

- [ ] CPU database replay: 10,356,829 / 10,356,829 correct in both modes.
- [ ] Existing Metal correctness cross-check remains 100%.
- [ ] No crash, exception, stale-context mismatch or output divergence.

## Timing receipt

Use the normal uninstrumented `fastsecp256k1` library for every timing run.
Build the private `fastsecp256k1_gh336_test_hooks` object only for focused
cardinality/lifecycle tests; test hooks must not be present in the timed binary.

The replay driver prints one `measured pass` record containing wall time,
user/sys deltas, full-prefix checksum, process peak RSS, context-switch deltas
and page-fault deltas. Keep at least five such passes per mode. Its final
cross-mode validation must show identical legacy/batch full-prefix checksums.

Wrap each clean mode invocation with `/usr/bin/time -l` and attach raw output,
but treat it as a supporting whole-process envelope: it includes input
generation, cold run, warm-ups, measured passes and cross-mode validation.
Never assign those aggregate values to a single measured pass.

| revision/mode | warm-ups | measured | median wall | median user | median sys | peak RSS |
|---|---:|---:|---:|---:|---:|---:|
| v3.68 per-row | pending | pending | pending | pending | pending | pending |
| candidate per-row | pending | pending | pending | pending | pending | pending |
| candidate batch | pending | pending | pending | pending | pending | pending |

| revision/mode | median voluntary ctx | median involuntary ctx | median reclaims | median hard faults | instructions retired |
|---|---:|---:|---:|---:|---:|
| v3.68 per-row | pending | pending | pending | pending | pending |
| candidate per-row | pending | pending | pending | pending | pending |
| candidate batch | pending | pending | pending | pending | pending |

Pass thresholds:

- candidate per-row median wall ≤ 1.05× v3.68 per-row;
- candidate per-row median sys ≤ 2.0× v3.68 per-row;
- batch median sys ≤ per-row median sys plus the larger of 20% or 1.0 s;
- batch median wall ≤ 1.05× candidate per-row.
- candidate per-row median total context switches ≤ v3.68 per-row plus the
  larger of 20% or 50,000;
- batch median total context switches ≤ candidate per-row plus the larger of
  20% or 50,000.

## Publication-lock closure

For both candidate modes:

- [ ] Save a 15-second `sample` capture.
- [ ] Confirm no `std::__sp_mut`, `__get_sp_mut`,
      publication-related `__psynch_mutexwait` or `__psynch_mutexdrop` stack
      occurs below the fixed-base entry points.
- [ ] Save optimized Release disassembly for
      `scalar_mul_generator` and `batch_scalar_mul_generator`.
- [ ] Confirm the raw identity acquire is inline.
- [ ] Confirm neither symbol calls a shared-pointer atomic, pthread mutex
      helper or generic non-lock-free atomic helper.
- [ ] Confirm scalar acquisition cardinality is O(rows) and batch cardinality
      is O(batch calls), not O(elements).

Example Darwin inspection commands (adjust the binary path, keep full output):

```sh
nm -nm <bench-binary> | c++filt | grep -E \
  'scalar_mul_generator|batch_scalar_mul_generator'
llvm-objdump --disassemble --demangle <bench-binary> > issue336.disassembly.txt
sample <pid> 15 -file issue336.sample.txt
```

## Local candidate gates

Attach the developer-side results for:

- focused scalar/batch independent-oracle correctness;
- deterministic old-owner completion and same-thread stale-TLS refresh;
- one- and 18-reader races against configure, cache-directory change and
  public cache load;
- acquisition-cardinality diagnostics;
- GCC 14 C++20 `-Werror`;
- ASan/UBSan and supported TSan;
- `git diff --check`.

Linux review-runner receipt (2026-07-23): focused Release tests passed
77/77 CPU checks and 14/14 audit checks; scalar and `n=67` batch counters were
1 and 1; ASan/UBSan passed. The direct GCC TSan CPU test passed; the direct
audit launch failed before `main` with this kernel's `unexpected memory
mapping` runtime limitation. With ASLR disabled via `setarch x86_64 -R`, both
focused TSan binaries passed. GCC 14 C++20 `-Werror` built the library;
`git diff --check` passed. This receipt is not M5 performance evidence.

## Closure decision

Keep issue #336 open if any cell, raw artifact or threshold above is missing.
If publication lock stacks are gone but a performance threshold still misses,
record the new hotspot attribution and continue with a focused follow-up; do
not convert absence of the original lock stack into a performance pass.
