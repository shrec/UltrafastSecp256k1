# F2: bounded delayed normalization and a size-dependent portfolio

This prospective native C++ experiment retains the F1 final-only contract:

x[i+1] = (x[i] + rhs[i]) mod n; return only canonical x[N].

x0 and all RHS operands are canonical in [0,n), each input is processed at
runtime and x0 is included once. Intermediate prefixes and side effects are
not observable. The owner permits different algorithms for different scenarios;
there is no presumption that one representation should win every input size.

## Four complete jobs

| Variant | Included representation work |
|---|---|
| serial | Frozen pa_f1::serial, with N Original full modular additions |
| Wide5 | Exact unreduced five-limb accumulation; four frozen add64 calls plus a high-word addition per RHS; one final general 320-bit reduction |
| Columns128 | Four independent 128-bit column sums of the same stream; checked column normalization and one final general 320-bit reduction |
| cold multicore | Frozen F1 contiguous-chunk reduction, explicit worker pinning, allocation/create/join/merge/destruction inside every job |

Columns are parts of one number, not four unrelated final recurrences. Both F2
representations initialize from x0 and preserve its full unreduced sum with all
RHS values. They remove per-input modular reduction; Columns additionally
defers cross-column carry propagation until the final boundary.

For N <= 1,000,000,000, the exact sum is less than
(N+1)*2^256 < 2^286 and fits five 64-bit limbs. Each independent column is at
most (N+1)*(2^64-1) < 2^94 and fits unsigned __int128. Normalization validates
column bounds before propagating carries. The reducer accepts any 320-bit
integer, retains the first fold's high carry, folds it as needed, and performs
final canonicalization. It is not the canonical-input-only modular adder
misapplied to a wide value. Read and test the complete header's range argument.

Wide5 and Columns perform no heap allocation or thread creation. Their fixed
source accumulator types are 40 and 64 bytes, respectively, not measurements
of actual register occupancy or spill traffic. State initialization, structural
bounds checks, accumulation, normalization where required, all reduction folds,
canonicalization and 32-byte result materialization are inside the timed job.
No reused intermediate or precomputed modular summary is supplied.

Cold multicore costs N+workers source modular additions for nonempty input.
It includes three vector storage requests, platform thread-start resources,
creation, explicit affinity, private chunk reductions, joining, ordered merge
and destruction. Cold means a create/pin/join lifecycle per job, not cold caches,
fresh physical pages, a new allocator or a persistent worker pool. It is a
comparison point, not an automatic preferred implementation for large N.

F2 requires GCC/Clang-compatible unsigned __int128. SECP256K1_NO_INT128 selects
the frozen portable add64 path used in references/Wide5 and the optional
reducer fold; it does **not** remove the new 128-bit columns or products.
No constant-time guarantee is implied. Callers using ct::scalar_add, secret
contributions, extra validation or all-prefix observables are not drop-in
targets for these variable-time final-only jobs.

## Inputs, boundaries and observability

Registered N values are 1, 16, 256, 4096, 65536 and 1048576: 32 bytes,
512 bytes, 8 KiB, 128 KiB, 2 MiB and 32 MiB of RHS storage. The smallest cases
expose finalization and complete-call overhead, not just a bulk asymptote.
These are input sizes, not PMU-confirmed cache residency or DRAM bytes.

Seeded SplitMix64 rejection sampling generates canonical x0 and RHS before
timing. Default seed is 20260905; coordinator CPU defaults to 4. The unchanged
public Scalar recurrence computes a full reference outside timing. Input
generation, public-object construction and reference computation are excluded
uniformly from every candidate. Full setup/finalization of each representation
is included inside its job.

A timed region repeats independent complete jobs, each starting from the same
x0 and consuming the same runtime RHS. It does not concatenate recurrences or
replace repeated work with a periodic-stream summary. Every job has a common
noinline boundary; GCC also applies noipa. Empty compiler memory clobbers at
job inputs and the materialized output make the full call observable. There
are no per-addition barriers or volatile arithmetic. At N=1 especially, inspect
the actual binary to verify every full-job call and output store survives.

Every job materializes all 32 result bytes. Only the last identical-job result
in each timed region is compared byte-for-byte against the public reference.
Every individual preconditioning job is compared. Earlier timed-job results
are not individually validated, and no intermediate-prefix validation is
claimed. Checksums are evidence fields, never substitutes for output equality.
Before/after input checksums provide additional corruption detection; exact
input-preservation gates belong to the separate correctness executable.

## Prospectively fixed stabilization and calibration

This procedure was approved **before observing F2 timing**:

1. Perform exactly four **untimed full jobs per variant**, in four rotating
   rounds of the same seeded order. Each variant occupies each position once.
   Check every returned byte and retain all 16 untimed records.
2. Calibrate each variant in the seeded initial order, starting at one job.
   Require **two consecutive elapsed times >= the target at the same job
   count** to qualify and freeze that variant's job count.
3. After any below-target trial, reset the qualifying streak to zero and grow
   monotonically to ceil(1.5 * target_ns * current_jobs / elapsed_ns), clamped
   to the applicable resource cap. Never decrease jobs.
4. Permit at most 12 timed attempts per variant. Retain every attempt with
   elapsed time, jobs, cap, proposed next jobs, qualifying streak and decision.
5. If qualification is impossible at a cap or the attempt limit, emit a
   structured calibration_status="unqualified" JSON result and exit 2
   **before** warmup or measured rounds. Preserve all accumulated records,
   pending/qualified/unqualified variant states and the failure reason.
6. Once all variants qualify, run two warmup rounds followed by eight measured
   rounds. Never recalibrate after looking at measured results.

Elapsed values must be positive; division and growth are checked for finite
results. Floating-point growth is clamped before integer conversion. Self-tests
exercise repeat qualification, streak reset, growth, cap exhaustion and invalid
calibration states. Calibration stabilizes observed duration, not hardware
frequency, scheduler conditions or an individual-job latency distribution.
Later measured regions can still be short; retain and label them unchanged.

Resource caps are 100,000,000 jobs, 1,000,000,000 total RHS inputs and 1,000,000
cold thread launches per timed region. Products are preceded by division
checks. Actual cold workers is min(N,selected CPU count): N=1 launches one
worker, even if the selected pool contains eight CPU representatives. All
launch counts, work caps and source operation accounting use the actual count.

A bounded --max-jobs diagnostic override may lower, never raise, the job cap.
Registered performance invocations omit this override. Its use is explicit in
JSON; it provides a fast, real execution path to test retained unqualified
evidence without inventing benchmark results:

~~~sh
f2_reduce_compare --measure --cpu 4 --count 1 --target-ms 200 --max-jobs 1
~~~

In normal test conditions this emits unqualified evidence and exit 2 after
tiny conditioning/calibration, without sustained warmup or measurement.

## Statistics and registered matrix

Each variant calibrates separately, so job counts and total work volumes
differ. Primary timing is elapsed_region_ns/jobs, reported as
region_mean_ns_per_full_job. A secondary metric divides this by N to describe
time per RHS input. Neither is called modular-adder latency.

Median, min/max, MAD and inclusive IQR summarize eight **region means**, not
eight individual-job latencies. Same-round ratios compare serial's normalized
region mean to the candidate's normalized region mean. Above one favors the
candidate in that recorded pair. Normalization does not erase different work
volumes, allocator, scheduler, cache or thermal conditions. All eight ratios
are emitted and remain descriptive, not confidence intervals.

A seeded four-variant permutation rotates once per warmup/measured round.
Eight measured rounds put every variant at every position twice. Preconditioning
uses the same rotation for its four rounds. There is no separate per-sample
prewarm, post-result retuning, sample removal or best-series selection.

The prospective matrix is two separately retained series across six N values:
12 invocations and 384 measured records. The manager registers invocation order,
source/binary hashes and machine observations before sustained measurement.
Results may support a size-dependent portfolio; no production dispatcher or
threshold is installed by this experiment.

## CPU selection and environment

Inherited affinity is read before coordinator pinning. Complete package/core
topology selects one logical representative per allowed physical core,
excludes the coordinator's entire physical core and reserves one additional
core when possible. Remaining representatives form the selected pool. If
topology is incomplete, the explicitly labeled fallback uses distinct allowed
logical CPUs and analogous headroom. A single available unit uses one
colocated worker and reports that limitation.

The selected pool and actual first min(N,pool_size) worker CPU list are emitted
separately. The kernel explicitly pins every newly created worker; it does
not silently inherit CPU4-only affinity. Failures propagate after joining
started workers. Core types may be heterogeneous; separate physical cores
do not imply identical performance.

Only one benchmark region runs at once. No competing local builds, tests or
benchmarks run during authorized measurement. Frequency/governor endpoints
cover the coordinator and actual worker CPUs; no_turbo is read if available.
Missing values are null. These snapshots do not establish fixed frequency
inside regions. External load, sibling activity and thermals remain uncontrolled.
Only this program's process/thread affinity changes; no sudo/global settings.

## Validation, CLI and custody

Before timing, require native GCC, portable-helper GCC, native Clang and UBSan
correctness gates with independent Boost, unchanged Scalar byte equality,
empty/small/random sequences, wide integer equality, maximum-column bounds,
arbitrary 320-bit reduction, deliberate high-carry/fold boundaries and negative
controls. Reused fixtures are not independent corpora. Driver self-tests cover
decimal parsing, statistics, resource limits, calibration transitions, all
256 single-bit output corruptions and balanced order.

~~~sh
f2_reduce_compare (--smoke|--measure) \
  [--count 1|16|256|4096|65536|1048576] [--cpu N] [--seed N] \
  [--target-ms 200..1000] [--max-jobs 1..100000000]
~~~

Unknown, duplicate, missing, signed or overflowing options return exit 2.
Smoke rejects target and job-limit overrides, performs the same initial
conditioning, uses two jobs per region and skips qualification. Its two/eight
round output has timing_claim=false and is not performance evidence.

~~~sh
g++ -std=c++17 -O3 -march=native -DNDEBUG -Wall -Wextra -pthread \
  -Isrc/cpu/include experiments/parseatlas_secp256k1/probes/f2_reduce_compare.cpp \
  src/cpu/src/scalar.cpp -o /tmp/REVIEWED_UNIQUE_DIRECTORY/f2_reduce_compare
/tmp/REVIEWED_UNIQUE_DIRECTORY/f2_reduce_compare --smoke --cpu 4 --count 1
/tmp/REVIEWED_UNIQUE_DIRECTORY/f2_reduce_compare --measure --cpu 4 \
  --count 256 --seed 20260905 --target-ms 200
~~~

Create the unique output directory with mktemp -d. No LTO. The manager runs
mechanical gates, then reads source and actual full-job/repetition-loop/reducer/
worker assembly before authorizing sustained timing. Output schema is
f2_reduce_compare_v1. Persist raw JSON timestamps without rounding 64-bit
decimal values. Source counts are semantic calls/additions/state sizes, not
instructions, physical traffic or allocator system calls. Manifest source,
reference, binary, protocol, test, exact command and environment identities.

Task MCP lifecycle remains owner-suspended pending its repair. Worker-scoped
tools are absent; bounded manager-authorized exact reads were used, while
manager Source Graph verification remains active. Only this driver and protocol
are this worker's writes. Frozen F1 files and production remain unchanged.

All arithmetic and timing are C++; no Python. Finite tests are not a universal
proof. No novelty, constant-time safety, automatic production eligibility,
all-prefix acceleration or whole-engine gain is claimed.
