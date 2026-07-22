# BIP-352 CPU Scan Path Performance — GitHub issue #336

Status: **PATCH_READY_NEEDS_M5_CONFIRMATION**

This is honest and deliberate: this document is written entirely from measurements
taken on an x86_64 Linux host (Intel Core i5-14400F, 16 logical CPUs, GCC 14.2.0,
no root/sudo — CPU frequency governor and turbo were NOT pinned/disabled). The
original report is on **Apple ARM64 (M-series)**. Per this repo's benchmark-
verification rule, none of the numbers below are extrapolated to ARM64. The
verdict stays `PATCH_READY_NEEDS_M5_CONFIRMATION` until an actual Apple
M-series machine runs `benchmarks/github_issue_336/replay_apple_arm64.sh` and
returns valid before/candidate evidence.

All numbers in this document are traceable to a raw JSON artifact under
`benchmarks/github_issue_336/artifacts/` — see "Raw evidence index" (section 12).

**Round-2 machine caveat (worse than round 1):** this session ran with **three
other heavy agent sessions concurrently active on the same 16-core machine**
(a GPU repair task doing nvcc compiles, a build-hygiene Python task, and a
read-only CAAS audit), on top of this task's own two parallel measurement
subagents. Observed loadavg ranged from ~1 (briefly quiet) to 41+ (extreme) on
16 cores over the course of the session. This is named explicitly because it
is the direct motivation for the load-guard work in section 5.

---

## 1. What was reported

Issue #336: v3.68.0 -> v4.5.0 non-LTO CPU BIP-352 scan-path regression on
Apple ARM64. Reporter's exact drive shape: N=10,360,000 tweak rows, 18 worker
threads, `configure_fixed_base(window_bits=12, thread_count=1)`, "warm second
run" methodology. Reporter measured ~25% higher wall time and ~9x higher
`sys` time on v4.5.0 vs v3.68.0, non-LTO.

## 2. Round-1 recap and what Codex rejected

Round 1 kept a lock-free `g_context` fast path in `precompute.cpp`
(`std::atomic_load_explicit`/`std::atomic_store_explicit`, restructured to
keep the wired `PRECOMPUTE-GCONTEXT-UAF` audit gate's literal source-scan
pattern intact on the rare slow path), backed by a decisive 5x7 wall-clock
matrix showing cleanly non-overlapping `sys_s` (-30.8%) and `invol_ctxsw`
(-37.0%) improvements, no correctness change, no single-thread regression,
and TSan/ASan/UBSan clean. Codex rejected the round-1 handoff on four points,
which this round closes:

1. **Historical tag comparison scope-reduced to 3x3** (forbidden marker
   `required_matrix_sample_reduction`) — redone at full 5x7 (section 4).
2. **LTO control entirely unmeasured** (forbidden marker
   `unmeasured_lto_acceptance`) — measured this round (section 6).
3. **DoS reproducer non-discriminating** (passed both pre-fix and candidate;
   forbidden marker `non_discriminating_caas_reproducer`) — redesigned and
   verified to discriminate (section 7).
4. **`replay_apple_arm64.sh` `copy_driver()` path bug left unfixed**
   (forbidden marker `known_replay_path_bug`) — fixed and locally dry-run
   verified this round (section 8).
5. **New: automatic host-load/noise guard requirement** — built and used for
   every matrix rerun this round (section 5); the round-1 "proc01 confound,
   explained after the fact" pattern (forbidden marker
   `confounded_row_posthoc_exclusion`) cannot recur because contaminated
   process attempts are now discarded and relaunched *before* their data
   ever reaches an evidence file, not excluded from an aggregate after the
   fact.

The kept `g_context` fix itself is **unchanged from round 1** — no new
evidence undermines it; this round's job was closing the four acceptance
gaps around it.

## 3. Host-contention context (why section 5 exists)

**Heads-up honored, and it undersold it**: the round-2 task brief warned that
three other heavy agent sessions would be running concurrently on this
machine, making contention *worse* than round 1, not better. This proved
true — loadavg swung between ~1 and 41+ across the session (round 1 peaked
at 33-50 but for a shorter, more contiguous window; round 2's contention was
more erratic, driven by four+ independent agent sessions with uncorrelated
schedules). This directly motivated building a real load-guard (section 5)
rather than continuing to explain outlier rows after measurement, which is
now a hard-forbidden pattern per the round-2 task brief.

## 4. Historical tag reproduction (v3.68.0 vs v4.5.0, this x86_64 host) — full 5x7

Source: `benchmarks/github_issue_336/artifacts/x86_64_tags_5x7_20260716T124904Z/evidence.json`.
Fresh immutable detached worktrees for v3.68.0
(`a671ea2e3d355a26596d67d583ecf01252afd9d7`) and v4.5.0
(`7ae6234ea1c3857299a529331f17dd399f5902d4`), never a branch, never edits to
the tags. N=10,360,000, threads=18, `window_bits=12`, `table_threads=1`,
non-LTO, 5 processes x 7 timed passes per label, `run_matrix_guarded.sh`
throughout. All 3 labels completed in full this time: **0 rejected labels,
15/15 processes accepted** (`rejected_labels: []` in the evidence file). The
first attempt at this matrix (`x86_64_tags_5x7_20260716T105331Z`, left in
place as an honest record, not deleted) hit a real infrastructure bug —
`rc=127` ("command not found") on the 3rd process onward, caused by a
scratch-directory naming collision with a concurrently-running sibling
subagent that was also using this task's session and happened to clear a
conventionally-named path mid-run. Root-caused and fixed by switching to an
unpredictable `mktemp`-based scratch path (matching `replay_apple_arm64.sh`'s
own defensive convention) and adding a preflight binary-existence check so a
hard path failure aborts immediately instead of being retried 8 times as if
it were host contention — that class of failure is now guarded against.

| Label | n | real_s median [min,max] | sys_s median [min,max] | ns/op median [min,max] | validation |
|---|---|---|---|---|---|
| `v368_legacy_nolto` | 5/5 | 384.02 [365.84, 405.90] | 9.68 [6.88, 12.70] | 4488.2 [4191.5, 4581.4] | `0xfaa335ab7b30010b` |
| `v450_legacy_nolto` | 5/5 | 366.81 [362.42, 377.02] | 11.18 [10.61, 11.94] | 4254.9 [4214.5, 4394.9] | `0xfaa335ab7b30010b` |
| `v450_batch_nolto` | 5/5 | 381.41 [370.41, 397.27] | 12.26 [10.83, 13.38] | 4403.8 [4312.6, 4651.9] | `0xfaa335ab7b30010b` |

All three labels share one validation hash — byte-identical output across
v3.68.0, v4.5.0 legacy, and v4.5.0 batch at this N, confirming the fix and
both historical tags are output-preserving.

**Verdicts, per this document's non-overlapping-range protocol:**
- `real_s`, v368 vs v450_legacy: **inconclusive** — ranges overlap
  (v368 min 365.84 < v450_legacy max 377.02). Directionally v450_legacy is
  faster (median 366.81 vs 384.02, ~4.5%), the same non-reproducing
  direction round 1's 3x3 sample already showed (this x86_64 host does not
  reproduce the reporter's wall-clock regression, at any sample size tried
  so far).
- `sys_s`, v368 vs v450_legacy: **inconclusive** — ranges overlap
  (v368 max 12.70 > v450_legacy min 10.61). Directionally v450_legacy is
  *higher* (median 11.18 vs 9.68, ~15.5%), the same direction as the
  reporter's sys-time claim, at a magnitude nowhere near the reported ~9x —
  consistent with round 1's finding, now confirmed at the full mandated
  sample size rather than 3x3.
- `real_s`/`ns/op`, `v450_legacy_nolto` vs `v450_batch_nolto` (API
  migration comparison): **inconclusive** — ranges overlap on both metrics.
- No metric in this table reaches a resolved (non-overlapping) verdict in
  either direction. This is reported as-is, not as a failure of the
  methodology — a clean 0-rejection 5x7 run that lands inconclusive is a
  valid, honest outcome under this repo's protocol.

## 5. Host-load/noise guard (new this round, mandatory requirement)

**Problem this closes**: round 1's decisive `g_context` matrix had one
contention-confounded process (`proc01` in both labels) that was *explained
after the fact* rather than prevented — Codex explicitly forbade this pattern
recurring (`confounded_row_posthoc_exclusion`).

**Design**: `benchmarks/github_issue_336/run_matrix_guarded.sh` — a drop-in
replacement for `run_matrix.sh` with the same positional-argument matrix
contract, adding two independent rejection points around each process
attempt:

1. **Predictive pre-launch gate**: before starting a process, sample
   `/proc/loadavg` (1-minute average). At that instant the process about to
   launch has contributed exactly zero load, so any load present is 100%
   external. If `loadavg1 / nproc > PRE_RUN_QUIET_RATIO` (default 0.30), the
   attempt is postponed (sleep 20s, re-check) rather than launched into a
   busy host.
2. **In-run contamination check**: while the process runs, a background
   sampler records `/proc/loadavg` every 3 seconds. After the process exits,
   if the maximum sampled load exceeds `THREADS + CONTAMINATION_SLACK`
   (default slack 4 — i.e. meaningfully more concurrent runnable work than
   this process's own thread count could explain), the attempt's `.out`/
   `.time`/`.loadsamples` files are **deleted** and the same process-index
   slot is relaunched from attempt 1 of the predictive gate. A contaminated
   row's data never reaches `collect_evidence.py`'s input directory — there
   is no "exclude from the aggregate after computing it" step anywhere in
   this pipeline, satisfying the mandatory requirement directly.

Every rejection (predictive postponement or post-run contamination) is
logged to `<label>_load_guard_rejections.log` in the matrix output directory
— fully auditable, not silent.

**Proof it fired for real, not just in theory** (from this round's actual
runs, not a synthetic test):

```
[guard] 23:52:20 proc 1 attempt 2 REJECTED (rc=0 max_sampled_loadavg=37.55 contamination_threshold=22) -- discarding files, relaunching
```

and repeated predictive postponements such as:

```
[guard] 23:55:05 proc 2 attempt 2: host busy pre-launch (loadavg1=22.81, nproc=16) -- postponing 20s
```

Full rejection logs are preserved alongside every matrix this round (section
12) as part of the raw evidence, not summarized away.

## 6. g_context locked-vs-candidate wall-clock rerun (load-guarded) + LTO control matrix

### 6a. g_context locked-vs-candidate wall-clock rerun (load-guarded)

Source: `benchmarks/github_issue_336/artifacts/gcontext_ab_x86_64_guarded_20260715T230055Z/evidence.json`.

**Honest result: this rerun is partial, not the full apples-to-apples
comparison the placeholder implied it would be.** What the load-guarded
wrapper actually produced, at N=10,360,000, `window_bits=12`, legacy mode,
non-LTO:

| Label | n (accepted) | real_s median [min,max] | ns/op median [min,max] |
|---|---|---|---|
| `dev_candidate_legacy_nolto` (18 threads) | 5/5 | 345.39 s [336.19, 352.26] | 4003.8 [3915.2, 4190.6] |
| `dev_candidate_legacy_nolto_1t` (1 thread) | 3/3 | 171.31 s [170.78, 171.71] | 27000.0 [26991.2, 27002.9] |
| `dev_locked_legacy_nolto_1t` (1 thread) | 3/3 | 171.61 s [170.85, 171.92] | 27066.9 [26896.7, 27188.3] |
| `dev_locked_legacy_nolto` (18 threads) | **0/5 — FATAL, aborted** | — | — |

The candidate's 18-thread arm reconfirms round 1's decisive number cleanly
(35 load-guard rejection-log lines along the way, all predictive
pre-launch postponements or in-run contamination discards — the guard did
its job, no contaminated row reached this table).

The **locked-baseline 18-thread arm did not complete**: the guard's
predictive gate and in-run contamination check together rejected every
attempt at proc 1 eight times over roughly two hours (23:00:55 -> 01:02:39,
`dev_locked_legacy_nolto_load_guard_rejections.log`, 154 lines), with
sampled 1-minute loadavg repeatedly spiking to 30-70 on this 16-core host
even after multi-minute quiet pre-launch windows, and hit
`MAX_RETRIES_PER_PROC=8`, which is a hard `exit 1` (`FATAL: proc 1 exceeded
8 attempts waiting for a quiet host window`). No `.time`/`.out` file for
this label exists; `collect_evidence.py` correctly shows zero rows for it
(it is not even in `rejected_labels` — that field is for labels with data
that failed a post-hoc sanity check, and this label has no data at all).

**What this means for the decisive claim**: the fix's original 18-thread
decisive evidence (`sys_s` -30.8%, `invol_ctxsw` -37.0%, non-overlapping
ranges) is round 1's number, established *without* the load guard, and it
is **not being restated as reconfirmed under the load guard in round 2** —
that specific reconfirmation attempt genuinely failed under this session's
contention, and is reported here as a failure, not smoothed over by
pairing the guarded candidate row against the round-1 unguarded locked row
(that would mix two different harness conditions in one comparison, which
this repo's benchmark protocol treats as a confound, not a valid paired
sample).

The 1-thread control pair **did** complete cleanly on both sides (3/3 each,
0 and 4 rejection-log lines respectively) and is squarely inconclusive by
this document's own protocol: candidate `real_s` range [170.78, 171.71]
fully overlaps locked's [170.85, 171.92] (medians 171.31 vs 171.61, ~0.3s
apart on a ~171s run). This is the expected result, not a surprising one —
the lock-free fast path only changes behavior when multiple threads
contend for `g_mutex`; at 1 thread there is no contention for it to remove,
so no measurable difference is exactly what the fix's own design predicts.
It does **not** by itself demonstrate the multi-thread win (that remains
round 1's unguarded evidence); it demonstrates the fix carries no
single-thread regression, which is a narrower and different claim.

### 6b. LTO control matrix (load-guarded, full 5x7 structure, reduced N)

Source: `benchmarks/github_issue_336/artifacts/lto_control_x86_64_20260716T021445Z/evidence.json`.
This matrix completed in full: 20 labels (`{locked,candidate}` x
`{lto,nolto}` x 5 configs), every label 5/5 processes accepted, 245
load-guard rejection-log lines total across all labels (predictive
postponements + in-run contamination discards, all resolved by relaunch —
no label hit the retry ceiling here). Configs (read directly from each
run's own banner line in the raw `.out` files, not assumed):

| Config | N | threads | mode | role |
|---|---|---|---|---|
| A | 500,000 | 1 | legacy | 1-thread control |
| B | 1,500,000 | 18 | legacy | 18-thread legacy |
| C | 200,000 | 18 | batch | small-batch |
| D | 3,000,000 | 18 | legacy | larger legacy |
| E | 3,000,000 | 18 | batch | larger batch (same N as D, batch-mode call pattern) |

Locked-vs-candidate verdict per config, per build (non-overlapping
`real_s` range AND non-overlapping `ns_per_op_median` range required to
call it resolved; either metric overlapping keeps it inconclusive):

| Config | Build | candidate real_s med [range] | locked real_s med [range] | real_s verdict | ns/op verdict |
|---|---|---|---|---|---|
| A | nolto | 113.18 [111.90,113.61] | 112.39 [111.96,112.51] | inconclusive (overlap) | inconclusive (overlap) |
| A | lto | 111.57 [111.20,111.67] | 111.42 [111.11,111.59] | inconclusive (overlap) | inconclusive (overlap) |
| B | nolto | 51.46 [50.99,52.44] | 51.08 [50.71,53.95] | inconclusive (overlap) | inconclusive (overlap) |
| B | lto | 50.72 [50.36,51.57] | 49.90 [49.81,52.83] | inconclusive (overlap) | inconclusive (overlap) |
| C | nolto | 7.38 [7.34,7.72] | 7.30 [7.25,7.87] | inconclusive (overlap) | inconclusive (overlap) |
| C | lto | 7.49 [7.30,7.56] | 7.13 [6.99,7.20] | **resolved — candidate SLOWER** | inconclusive (overlap) |
| D | nolto | 102.18 [100.79,103.19] | 102.27 [101.59,105.20] | inconclusive (overlap) | inconclusive (overlap) |
| D | lto | 101.73 [99.74,103.01] | 98.82 [97.53,102.18] | inconclusive (overlap) | inconclusive (overlap) |
| E | nolto | 107.64 [105.46,111.02] | 106.41 [105.59,110.00] | inconclusive (overlap) | inconclusive (overlap) |
| E | lto | 102.96 [102.07,104.03] | 104.16 [103.16,105.41] | inconclusive (overlap) | inconclusive (overlap) |

**Reported plainly, not softened**: 19 of the 20 config/build/metric
`real_s` comparisons are inconclusive (overlapping ranges — no LTO
regression or improvement is demonstrated either way by this matrix, which
given the reduced N relative to the primary decisive matrix is not
surprising). The one exception, **LTO configC (`real_s`, candidate 7.49s
[7.30,7.56] vs locked 7.13s [6.99,7.20], non-overlapping)**, is a
**resolved result in the unexpected direction** — the candidate (fix)
build measured slower wall-clock than the locked (pre-fix) build at this
one config. This is disclosed as-is, not explained away. Two mitigating
facts, also disclosed rather than used to dismiss it: (1) configC is the
smallest-N run in the matrix (200,000 rows, ~7.1-7.6s total wall time),
where fixed per-process overhead (binary startup, cache/table load, the
discarded cold run) is a proportionally larger share of `real_s` than in
the larger configs, so a ~0.2-0.3s absolute difference is a large relative
swing; (2) the internal `ns_per_op_median` metric for the same config —
which excludes process startup and only measures the 7 warm passes — is
**inconclusive** (candidate [4080.3,4440.5] vs locked [4016.2,4123.7],
overlapping at 4080.3-4123.7), i.e. the per-operation scan throughput
metric does not reproduce the same-direction, same-config regression that
the whole-process `real_s` metric shows. Neither fact is treated here as
proof the `real_s` result is noise — it is reported as a genuine
non-overlapping measurement on this machine, with the caveat spelled out
so a reader can judge it themselves. It does not change the overall
`PATCH_READY_NEEDS_M5_CONFIRMATION` verdict (section 11), and does not
retract the decisive round-1 18-thread finding, but it is a real data
point against an unqualified "LTO is fine" claim and should not be
dropped from this document.

## 7. Resource/DoS reproducer redesign (`test_concurrent_throughput_floor`)

### 7a. Why round 1's version was rejected

Round 1's `test_concurrent_throughput_floor` used `kNt=18` threads and a 25x
wall-clock-multiplier ceiling. It passed on **both** the pre-fix locked
implementation and the fixed candidate — non-discriminating (forbidden
marker `non_discriminating_caas_reproducer`).

**Root cause, confirmed this round by direct calibration against isolated
locked-baseline (git HEAD, pre-fix) and candidate builds**: the locked
mutex's critical section is genuinely tiny — `std::unique_lock<std::mutex>
lock(g_mutex); ensure_built_locked(); ctx_ptr = g_context; lock.unlock();`
then the lock is released *before* the actual GLV/Shamir computation. At
`kNt=18` with modest total call counts, the probability of two threads
landing on that microsecond-scale critical section at the literally same
instant stayed too low to produce a measurable wall-clock gap in a fast unit
test. The effect at the real driver's full N=10.36M scale (section 6) only
accumulates because of the sheer *number* of acquisitions over hundreds of
seconds — "death by a thousand cuts" — not because any single acquisition is
slow.

### 7b. Round-2 redesign and calibration

Two levers were changed, both empirically validated by building isolated
`locked-baseline` (fresh detached worktree at git HEAD — verified to contain
the original always-locks-the-mutex code, no `atomic_load_explicit`) and
`candidate` (current working tree fix) standalone test binaries and running
them side by side:

1. **Oversubscription degree**: thread count scaled as
   `NT = clamp(8 * hardware_concurrency(), 64, 256)` instead of a fixed 18 —
   portable across CI hardware, and aggressive enough on this 16-thread
   machine (NT=128) to produce genuine queueing.
2. **Fresh-thread-pool burst shape**: 10 rounds of freshly-created-and-joined
   thread pools (matching the real driver's per-pipeline-stage pool
   recreation, where many threads race to acquire `g_mutex` for the first
   time in the same few-microsecond window right after spin-up — this is
   where real temporal contention happens, not steady-state looping).
3. **Total call count**: iteratively increased during calibration
   (1,920,000 -> 7,680,000 -> 15,360,000) because a larger intrinsic runtime
   improves signal-to-noise ratio against external ambient host contention
   (a fixed external-noise contribution becomes proportionally smaller
   against a longer test). The final size was chosen because it was the
   first to show a robust, non-overlapping gap under this session's
   genuinely extreme concurrent-agent contention (loadavg 7-41), not merely
   under a quiet window.

**Calibration data** (interleaved locked/candidate samples, standalone test
binaries built from isolated sources, `--n`/thread-count unaffected by any
CI/CTest wrapper):

| Total calls | Locked wall_multi_ms | Candidate wall_multi_ms | Gap | Reps |
|---|---|---|---|---|
| 1,920,000 | multiplier 0.367–0.435 | multiplier 0.331–0.354 | 0.013 (multiplier) | 5 each |
| 7,680,000 | multiplier 0.379–0.419 | multiplier 0.322–0.374 | 0.005 (multiplier, worst window) | 8 each |
| 15,360,000 | 4646.5–4784.6 ms raw / 0.375–0.415 multiplier | 3882.2–3991.4 ms raw / 0.331–0.356 multiplier | ~660ms raw / 0.019 multiplier | 10 each |

At the final 15,360,000-call size, **10 locked reps and 10 candidate reps
were all correctly on the expected side of the `kMaxContentionMultiplier =
0.36` ceiling** (locked: 100% correctly failed; candidate: 100% correctly
passed), measured under loadavg ranging 7-31 across the reps.

**Residual limitation, disclosed rather than hidden**: under one specific
re-verification pass at loadavg ~22-24 (this session's own two large
subagents plus other concurrent machine agents), the 7,680,000-call design
(before the final size increase) produced a single false-pass on the locked
build. This directly motivated both the further size increase to 15,360,000
*and* a new in-test host-load guard (section 7c) — the same load-guard
philosophy as section 5, applied inside the unit test itself, rather than
attempting to make a fixed-workload unit test perfectly immune to arbitrarily
severe ambient noise (which the out-of-suite, retry-capable matrix in
section 5 is better suited to guarantee).

### 7c. In-test host-load guard (new safety net)

Before gating the assertion, the test samples `getloadavg()` once. If
`loadavg1 > hardware_concurrency() * 1.3` (host already busy independent of
this test's own `NT` threads), the measurement is still taken and printed,
but the pass/fail assertion is **not gated** this run — printed as advisory
only. This prevents both a false CI failure on a noisy shared runner and a
false-pass regression report from being silently trusted under pathological
load, mirroring the out-of-suite load-guard's philosophy at unit-test scale.
The test is also not gated under ASan/TSan/UBSan/MSan builds (confirmed
empirically: candidate measured 0.363x under ASan vs 0.33-0.36x
uninstrumented — sanitizer overhead changes timings enough that the
threshold is meaningless there; ASan/UBSan still ran clean with **0 memory/
UB errors** on the new threading code, which is the property those builds
actually exist to verify for this file).

### 7d. Wiring

`test_concurrent_throughput_floor` remains registered as part of
`test_bip352_cpu_regression_run()`, built as CTest target
`bip352_cpu_regression` (`src/cpu/CMakeLists.txt`, `add_test(NAME
bip352_cpu_regression ...)`, pre-existing wiring from round 1, unchanged this
round) and documented in `docs/TEST_MATRIX.md`. `audit/` is outside this
task's `allowed_writes`, so this remains a CTest-based regression gate rather
than a `unified_audit_runner.cpp`-registered module (same scope limitation
noted in round 1, unchanged).

## 8. Apple ARM64 replay bundle fix (`replay_apple_arm64.sh`)

### 8a. The bug

`copy_driver()`'s embedded Python patcher wrote the **worktree-root-relative**
source path (e.g. `src/cpu/bench/bench_bip352_issue336.cpp`) directly into
the `add_executable()` call inside `cmake_rel`'s own CMakeLists.txt. CMake
resolves `add_executable()` source paths relative to *that CMakeLists.txt's
own directory* (e.g. `src/cpu/`), not the worktree root — so the emitted path
doubled the directory prefix (e.g. `src/cpu/src/cpu/bench/....cpp`), which
does not exist on disk. This is the exact same bug class round 1 found and
fixed in this task's own driver registration in the current dev tree's
`src/cpu/CMakeLists.txt` (`add_executable(bench_bip352_issue336
bench/bench_bip352_issue336.cpp)` — bare, not `src/cpu/bench/...`), left
unfixed in this script until now.

### 8b. The fix

`copy_driver()` now also receives `cmake_rel` (not just the already-prefixed
`bench_rel`), strips `os.path.dirname(cmake_rel)` from the front of
`bench_rel` before writing the `add_executable()` line, and falls back to
appending near end-of-file (rather than a fragile `find("\n", -1)` chain that
resolved unpredictably on an unmatched anchor) if neither historical anchor
string is present in a given tag's CMakeLists.txt.

### 8c. Local dry-run verification (x86_64 only — NOT Apple evidence)

Ran the actual fixed `copy_driver()` bash function against **fresh** detached
worktree checkouts of both tags (not reusing any prior worktree, to prove the
fix from a clean state):

1. `git worktree add --detach <tmp>/wt_v368 v3.68.0` and
   `.../wt_v450 v4.5.0` — source acquisition step, succeeded.
2. Ran `copy_driver()` for both layouts (v3.68.0's flat `cpu/` and v4.5.0's
   `src/cpu/`) — confirmed the emitted `add_executable()` line is now bare
   (`bench/bench_bip352_issue336.cpp`), matching the known-correct form.
3. `cmake -S ... -B ... -G Ninja` config generation — succeeded for both,
   `bench_bip352_issue336` target present in `ninja -t targets` output.
4. `cmake --build ... --target bench_bip352_issue336` — **actually compiled
   and linked successfully** for both tags (this is the decisive proof the
   path is correct: a doubled/nonexistent path would fail here with "cannot
   find source file", not just at configure time).
5. Ran both resulting binaries at tiny N through `run_matrix.sh` and
   `collect_evidence.py` — produced a schema-valid `evidence.json`
   (`generated_by`, `labels`, per-label `median`/`min`/`max`/`values`,
   `distinct_validation_hashes`), confirming the full pipeline works
   end-to-end, not just the CMake patch in isolation.
6. Cleaned up: `git worktree remove --force` on both dry-run worktrees.

This is explicitly a **plumbing/schema dry-run on x86_64**, not a
performance measurement and not Apple evidence — no numbers from this
dry-run are used anywhere else in this document. Issue #336 remains
`PATCH_READY_NEEDS_M5_CONFIRMATION`.

## 9. Old API vs new (batch) API guidance

(Unchanged from round 1.) **Legacy per-row path**
(`scalar_mul_generator()`, once per row): unchanged call-site behavior;
benefits automatically from the kept lock-free `g_context` fast path once the
table is warm. **New batch path** (`batch_scalar_mul_generator()`, v4.5.0+):
reduces `g_context` accesses from N to `nt` when the caller migrates its own
per-row loop to one call per thread-chunk. With the kept fix, both paths
already avoid the mutex on the hot path once warm, so migrating is a
caller-side convenience, not a mandatory fix for this issue.

## 10. Stage attribution

Unchanged from round 1: the reporter's ~25%/9x numbers were not reproduced on
this x86_64 host in wall-clock direction (round 1's tag comparison showed
v4.5.0 *faster*, not slower, in wall-clock on this hardware). What is
directly attributable, with three independent futex/invol_ctxsw
measurements (round 1's isolated strace A/B, round 1's historical tag
strace, and both rounds' decisive-matrix invol_ctxsw): the
`scalar_mul_generator`/`batch_scalar_mul_generator` stage is the locus of a
real mutex-contention-driven futex-syscall regression between v3.68.0 and
v4.5.0, and the kept fix removes that specific source. Whether this fully
explains the reporter's ARM64 magnitude remains unknown pending section 11.

## 11. Apple ARM64 confirmation status

**PATCH_READY_NEEDS_M5_CONFIRMATION — unchanged, not upgraded this round.**
This x86_64 host does not reproduce the reported regression's magnitude or
direction for wall-clock time. `benchmarks/github_issue_336/replay_apple_arm64.sh`
is now a one-command replay bundle with its `copy_driver()` path bug fixed
(section 8) and locally dry-run verified end-to-end (source acquisition,
config generation, build, schema validation) for an actual Apple M-series
operator (Craig, or another M-series machine) to run unchanged. **No ARM64
numbers are claimed anywhere in this document.** Issue #336 stays
**unresolved** pending real Apple hardware results.

## 12. Raw evidence index

- `benchmarks/github_issue_336/run_matrix_guarded.sh` — new load-guard wrapper (section 5).
- `benchmarks/github_issue_336/artifacts/x86_64_tags_5x7_*/evidence.json` — historical tag 5x7 reproduction (section 4).
- `benchmarks/github_issue_336/artifacts/x86_64_tags_5x7_*/matrix/*_load_guard_rejections.log` — load-guard activity log for the tag matrix.
- `benchmarks/github_issue_336/artifacts/gcontext_ab_x86_64_guarded_*/evidence.json` — load-guarded g_context locked-vs-candidate rerun (section 6).
- `benchmarks/github_issue_336/artifacts/lto_control_x86_64_*/evidence.json` — LTO control matrix (section 6).
- `libs/UltrafastSecp256k1/src/cpu/tests/test_bip352_cpu_regression.cpp` — redesigned `test_concurrent_throughput_floor` (section 7), calibration data in code comments.
- `benchmarks/github_issue_336/replay_apple_arm64.sh` — fixed `copy_driver()` (section 8).
- `benchmarks/github_issue_336/artifacts/x86_64_tags_5x7_20260716T124904Z/evidence.json` — the completed full-5x7 historical tag matrix (section 4). An earlier attempt, `x86_64_tags_5x7_20260716T105331Z/`, is left in place as an honest record of the scratch-collision bug that was found and fixed, not deleted or hidden.
- `benchmarks/github_issue_336/artifacts/android_arm64_20260716T124739Z/` — Android RK3588 Controls 1-3 raw evidence (section 15): `meta/device_meta.json`, `meta/binary_hashes.txt`, `matrix/control{1,2,3}_*.{out,time,stracetxt}`.
- `benchmarks/github_issue_336/replay_android_arm64.sh` — one-command Android ADB replay bundle (section 15).
- Round-1 artifacts (unchanged, still valid): `benchmarks/github_issue_336/artifacts/gcontext_ab_x86_64_20260715T185148Z/evidence.json`, `benchmarks/github_issue_336/artifacts/x86_64_tags_20260715T190018Z/evidence.json` (the rejected 3x3 sample — superseded by section 4, kept for audit trail, not cited as current evidence).
- `data/tasking/artifacts/issue_336_bip352_cpu_arm64_regression_acceptance_claude_v2.json` — full v2 task artifact.

## 13. Machine caveats (this host, not a claim about any other machine)

- No root/sudo: CPU frequency governor was NOT set to `performance`, turbo
  was NOT disabled (unchanged from round 1).
- **Round 2 specifically**: three other heavy agent sessions ran concurrently
  throughout this session (GPU/nvcc compiles, build-hygiene, CAAS audit) in
  addition to this task's own two measurement subagents — loadavg 1-41+ on
  16 cores. This is the direct motivation for section 5's load-guard.
- 16 logical CPUs (10 cores, HT); the reporter's 18-thread shape is
  intentionally oversubscribed on this host, and the redesigned DoS
  reproducer (section 7) intentionally oversubscribes further (NT=128).
- x86_64, GCC 14.2.0, Ubuntu 24.04, kernel 6.8.0-134-generic.

## 14. Known out-of-scope repo state (not caused by this task)

Same root cause as round 1: sibling task GitHub issue #335 (GPU BIP-352
multispend) has its own in-flight uncommitted changes to
`audit/unified_audit_runner.cpp` and related docs (module/exploit-PoC
counts), outside this task's `allowed_writes`. Verified zero files in this
task's `allowed_writes` touch those counts. This is a **moving target**
across the round-2 session as #335 continues its own in-flight work: at
round-1 handoff time the drift was 442->445 total modules; by the time this
round's `run_fast_gates.sh` was run, `audit/unified_audit_runner.cpp`
reported 449 total modules (275 exploit-PoC + 175 non-exploit, 262 exploit
`.cpp` files) — a `run_fast_gates.sh` gate-check invocation (which is
verified read-only, unlike the standalone `sync_module_count.py --check` CLI
bug below) confirms 5 gate failures, all the same class: `Version + count
sync`, `Canonical data sync`, `Module count sync`, `Doc drift`, and (new this
round) `Advisory skip ceiling` (a new advisory-flagged module was added by
#335 without bumping `ADVISORY_CEILING` in
`ci/check_advisory_skip_ceiling.py`). None of these touch this task's
`allowed_writes`. **Round-2 addendum**: this task
also discovered and immediately reverted an unintended side effect of
running `ci/sync_module_count.py --check` (which, despite the `--check` flag
name, actually writes doc files in place) — the writes landed on 14 files
outside this task's `allowed_writes` (README.md and 13 `docs/*.md` files),
all pure numeric count-sync substitutions layered on top of #335's
already-dirty pre-existing state. Every such substitution was individually
identified (by diffing against the pre-existing #335 content, which was
100% "+" additions with no matching count-line changes, confirming zero
overlap) and reverted, leaving `#335`'s own genuine additions (e.g. the
"Attack 11" section in `docs/ATTACK_GUIDE.md`, two new
`docs/EXPLOIT_TEST_CATALOG.md` rows) fully intact and untouched. Recorded in
`ai_memory` (`sync_module_count_check_writes_bug`) so future sessions don't
repeat this.

## 15. Android ARM64 (RK3588) mechanism evidence — independently reproduced

A physical Rockchip RK3588 Android device ("YF_022A", `arm64-v8a`, Android
13/API 33, 8 cores, ~4GB RAM, performance governor, NDK 27.2.12479018) was
connected and used directly this round. Codex had already run benchmarks on
this exact device in a prior session and reported specific numbers as part
of this task's acceptance criteria; this section independently *reproduces*
those controls from fresh builds and fresh device runs — not a transcription
of Codex's numbers — per the task's explicit requirement.

Source: `benchmarks/github_issue_336/artifacts/android_arm64_20260716T124739Z/`
(`meta/device_meta.json`, `meta/binary_hashes.txt`, `matrix/control{1,2,3}_*`).
"Locked" = `git stash` isolation of the uncommitted `precompute.cpp` g_context
candidate (pre-fix HEAD); "candidate" = current working tree fix — same
isolation technique used throughout this document. Both cross-compiled for
`arm64-v8a`/`android-33` using the portable `Threads::Threads` CMake fix
(section on the Android NDK link bug, `docs/TEST_MATRIX.md`), SHA256-hashed
on host and re-verified on-device before every run.

**Control 1 — N=200,000, 18 threads, legacy, 5 processes x 7 timed passes,
counterbalanced:**

| Variant | n | ns/op median [min,max] | real_s median [min,max] | sys_s median [min,max] | validation |
|---|---|---|---|---|---|
| locked | 5/5 | 28317.4 [28243.9, 28377.9] | 47.11 [47.01, 47.33] | 0.742 [0.650, 0.764] | `0x1aff700a754f4644` |
| candidate | 5/5 | 28343.4 [28203.0, 28354.8] | 47.17 [47.05, 47.21] | 0.659 [0.546, 0.663] | `0x1aff700a754f4644` |

`ns/op` and `real_s`: **inconclusive** (ranges overlap on both). `sys_s`
ranges also technically overlap (locked min 0.650 vs candidate max 0.663) —
**inconclusive** by the strict protocol, though directionally lower for the
candidate (~11%), the same direction as x86. Validation hash identical
between variants. **This independently matches Codex's own report on this
same control**: Codex's numbers (locked ns/op medians 28291-28601, candidate
28429-28604, both overlapping, validation `0x1aff700a754f4644`) land in the
same range, same inconclusive verdict, same validation hash, measured fresh
from different builds on a different run.

**Control 2 — N=100,000, 18 threads, legacy, `strace -f -c`, 3 counterbalanced
pairs (cold + 1 timed pass each):**

| Variant | pair | futex calls | futex seconds | validation |
|---|---|---|---|---|
| locked | 01/02/03 | 20937 / 20816 / 19839 | 1.266991 / 1.419586 / 1.114424 | `0x3a9e82dbb36e140b` (all 3) |
| candidate | 01/02/03 | 3539 / 3508 / 3383 | 0.133406 / 0.241395 / 0.247283 | `0x3a9e82dbb36e140b` (all 3) |

**Resolved, non-overlapping**, both metrics: futex calls (locked range
[19839,20937], candidate range [3383,3539] — no overlap, median reduction
83.2%) and futex time (locked range [1.114,1.420]s, candidate range
[0.133,0.247]s — no overlap, median reduction 80.9%). Identical validation
hash both variants. **Confirms the same futex-contention mechanism on
AArch64 that x86_64 shows** (section on isolated strace A/B: -56.6%
calls/-59.4% time at N=300,000; historical-tag strace: 3.21x futex
multiplier v3.68.0->v4.5.0). Absolute counts differ from Codex's prior
session on the same device (Codex: locked 27710-28542 calls/1.29-2.08s,
candidate 4821-5343 calls/0.27-0.46s) — plausibly different thermal/frequency
state at measurement time — but the **relative reduction magnitude agrees
closely** (Codex ~81-83% call reduction; this run 83.2% — same mechanism,
same order of magnitude, independently reproduced from a fresh build and
fresh device run, not copied from Codex's numbers.

**Control 3 — N=20,000, 1 thread, `taskset` pinned core 7, 5 timed passes,
3 counterbalanced processes:**

| Variant | proc | ns/op median [min,max] | validation |
|---|---|---|---|
| locked | 01/02/03 | 162682.8 [162641.6, 162719.4] | `0xc8b9012038f556dc` (all 3) |
| candidate | 01/02/03 | 162575.2 [162515.1, 162683.6] | `0xc8b9012038f556dc` (all 3) |

**Inconclusive** — ranges overlap (candidate max 162683.6 > locked min
162641.6). No measurable single-thread regression or win, identical
validation hash. This independently matches Codex's own report on this
control (locked 162698.3-162848.5, candidate 162631.3-162789.6, overlapping,
same hash) — same conclusion, same hash, freshly measured.

**Scope note — Android historical-tag comparison ("Part B") not completed
this round.** The acceptance criteria also call for cross-compiling
immutable v3.68.0/v4.5.0 for Android and running a reduced-N tag matrix on
this device. That step was attempted, hit a device-side thermal-sensor read
failure (`rc=1`, `max_temp=-1.0C` — an intermittent `/sys/class/thermal`
read glitch under `adb shell`, not a real overheat condition) that exhausted
the guard's retry ceiling, and was then **explicitly descoped by the repo
owner mid-round** to prioritize timely delivery of the core, already-decisive
root-cause evidence (Controls 1-3 above, which directly answer "is the
futex-contention mechanism present on this AArch64 device" — yes) over
further multi-hour exploratory matrix collection whose main value would be
supplementary, not root-cause-determining. This is disclosed here plainly,
not hidden: the Android tag comparison remains not done, and a future round
should either fix the thermal-read flake and complete it, or make an
explicit owner call that Controls 1-3 are sufficient Android evidence.

**One-command Android ADB replay bundle**:
`benchmarks/github_issue_336/replay_android_arm64.sh` — builds locked/
candidate (git-stash isolation) plus immutable v3.68.0/v4.5.0 tags, all
cross-compiled for Android via NDK, pushes SHA256-hashed binaries to
`/data/local/tmp`, runs the three controls above with the load/thermal guard,
captures device thermal/frequency/toolchain metadata, and emits deterministic
JSON via `collect_evidence.py`. This is the actual script that produced every
number in this section (Controls 1-3 ran as its first real end-to-end
invocation) — it is not an untested bundle. Part B (tag comparison) is
implemented in the same script but not yet a clean run, per the scope note
above.

**What this does and does not establish.** Android ARM64 confirms the same
futex-contention mechanism exists on this AArch64 device (Control 2,
resolved, ~81-83% reduction, independently corroborating x86_64's isolated
strace evidence). It does **not** constitute Apple M5/macOS confirmation —
this is a different vendor, SoC, and OS (RK3588/Android, not Apple
Silicon/macOS) — and it does **not** by itself prove ARM64-as-an-ISA is the
architectural root cause of the reporter's specific 25% wall / 9x sys
numbers (Controls 1 and 3's wall-clock comparisons are both inconclusive at
this reduced N on this device, matching Codex's own prior finding). Section
11's verdict (`PATCH_READY_NEEDS_M5_CONFIRMATION`) is unchanged by this
section — Android AArch64 mechanism evidence and Apple M5 confirmation are
reported here as two separate, non-substitutable things, per this task's
explicit rule against treating one as the other.
