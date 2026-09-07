# F1: one recurrence, alternative final-state representations

This prospective native C++ experiment tests representation-induced parallelism
of the **same final-only recurrence**, not a batch of unrelated chains:

x[i+1] = (x[i] + rhs[i]) mod n, with canonical x0 and every rhs[i] in [0,n).

The sole observable output is canonical x[N], serialized as exactly 32 bytes.
Intermediate prefixes, input-dependent future operands and intermediate side
effects are outside this contract. Associativity permits contiguous block
summaries and tree composition without changing that output. Every arithmetic
addition uses the unchanged Original full variable-time modular adder; this
stage does not combine that change with a new carry algorithm.

## Four complete-job implementations

| Variant | Representation and included work | Source modular additions, N > 0 |
|---|---|---:|
| serial | One dependency chain starting at x0 | N |
| chunk4 | Four balanced **contiguous** chunks reduced from zero, interleaved on one core; ordered pairwise merge and incorporate x0 once | N + 4 |
| tree | Allocate and zero-initialize N scalar elements, copy every RHS, compact ordered adjacent pairs, incorporate x0, destroy/free workspace | N |
| cold multicore | Allocate partial/error/thread vectors, create and explicitly pin workers, reduce contiguous chunks from zero, join, ordered merge, incorporate x0, destroy resources | N + workers |

Four in chunk4 is a representation width, not a hardcoded machine worker count.
The four partial sums reconstruct **one** recurrence's final result, unlike the
previous benchmark's eight independent final accumulators.

The tree kernel accepts caller-owned scratch for independent tests. This driver
deliberately allocates std::vector<Limbs>(N) inside **every** full-job wrapper,
then calls the kernel and destroys the vector before returning. Allocation,
source zero-initialization, copy, reduction and deallocation are all timed.
There is no reused-workspace fast path in this experiment. Source zero/copy
counts are semantic accounting; the compiler may optimize stores. They are not
measured machine instructions or physical memory-traffic counters.

Cold multicore means a fresh thread create/pin/join lifecycle **per job**, not
cold caches, untouched allocator pages, fresh OS resources or a persistent-pool
comparison. Its kernel allocates three vectors (partials, failure slots and
reserved thread handles) plus platform thread-start resources. Worker failures
are propagated after joining all started threads; partial startup failure also
joins already-started threads. The coordinator is not an extra arithmetic
worker. Both tree and multicore retain their allocation costs inside timing.

The header permits empty input as an identity and validates count, required
pointers, scratch capacity and CPU list syntax. Actual pointer extents,
canonical inputs and scratch disjointness remain documented preconditions.
The driver selects only positive registered sizes and owns disjoint allocations.
Finite-corpus correctness is not a full-domain or constant-time proof.

## Runtime inputs and repeated complete jobs

Registered counts are 256, 4096, 65536 and 1048576 scalar operands: 8 KiB,
128 KiB, 2 MiB and 32 MiB of RHS input. An additional canonical x0 contributes
exactly once. These sizes are locality dimensions, not PMU-confirmed cache
residency or DRAM traffic. The host may have heterogeneous core types; selecting
separate physical cores does not establish homogeneous worker performance.

Inputs are generated before timing using seeded SplitMix64 and rejection of
noncanonical values. Default seed is 20260905. All variants and all repetitions
within an invocation receive the same x0 and RHS. The public Scalar reference
is computed once, outside timing, by the original sequential recurrence.
Input generation and reference conversion are not included in any variant's
time. No candidate receives a precomputed modular summary.

A benchmark region repeats **independent complete jobs** to obtain enough clock
duration. Each job starts from x0, consumes every runtime RHS value and returns
the same final 32 bytes. It does not continue the previous job's accumulator.
It may not turn repeated jobs into one periodic-stream summary or multiply a
precomputed total by the number of jobs.

Every variant enters a common noinline full-job wrapper. GCC additionally uses
noipa; Clang uses noinline. Empty GNU compiler memory barriers at the job input
and materialized 32-byte output boundaries prevent repeated calls from being
treated as an effect-free reusable result. There are no per-addition barriers,
volatile arithmetic or hand-forced arithmetic instructions. These complete-job
boundary costs are included equally; source is not sufficient evidence that
work survived optimization. Review the actual measured binary's four job
routines, four timed repetition loops and reached kernel/worker bodies.

Every job materializes its complete output before return. The timed region
also retains the last output. Outside timing, that final result is compared by
all 32 little-endian bytes with the public reference for **every** preflight,
calibration, warmup and measured record. Earlier identical-job results are
materialized but not individually compared; no claim of checking every job or
every intermediate prefix is made. The checksum is evidence, not the result
acceptance predicate. Input checksums before/after additionally detect ordinary
corruption; exact input-preservation testing belongs to the separate tests.

## Calibration and statistics: differing work volumes

A single common job count would make small cold-thread workloads excessively
long, so each variant is calibrated **separately** to at least the target,
default 200 ms. There are at most eight trials per variant. The selected count
is then frozen for that variant's warmup and measured regions.

Resource gates require at most 1,000,000 jobs, at most 1,000,000,000 RHS inputs
per region, and at most 1,000,000 cold worker launches per region. Division
checks precede count products. The fixed N choices fit both size_t allocation
bounds and these work limits. Impossible calibration, invalid resource bounds,
thread/affinity errors or correctness failures terminate with exit 2; no failed
invocation produces accepted partial result JSON.

Per-variant calibration means candidates execute different job counts and
different aggregate work volumes. The primary per-record quantity is:

region_mean_ns_per_full_job = elapsed_region_ns / jobs

The secondary quantity divides that region mean by N, giving normalized time
per RHS input. Neither is a single modular-adder latency. Reported median,
min/max, MAD and inclusive IQR summarize eight **region means**, not a
distribution of individual-job latencies.

Same-round ratios are:

(serial elapsed / serial jobs) / (candidate elapsed / candidate jobs)

These compare normalized region means at differing work volumes, not
matched-work latency measurements. Above one favors the candidate in that
round's recorded conditions. All eight ratios remain visible; they are
descriptive, not confidence intervals or causal isolation of a single hardware
factor. Distinct total work volumes can change cache, allocator, scheduler and
thermal conditions; the normalization does not erase this limitation.

Preflight runs one full job per variant. Calibration follows the seeded initial
variant order. Two full external warmup rounds precede eight measured rounds.
The four-variant permutation rotates once each round, putting each variant at
every position exactly twice in the eight measured rounds. There is no separate
per-sample prewarm. Every preflight, calibration, warmup and measured record is
retained. A later or faster sample may fall below the requested target; it is
labeled and counted, not discarded or silently lengthened.

The prospective matrix is two separately retained series of all four sizes,
eight invocations total, four variants times eight measured rounds each:
256 measured records. The manager records invocation order and freezes all
source/binary identities before any sustained measurement. There is no pooled
series, best-run selection or deletion of negative results.

## CPU selection and observed environment

The driver reads the inherited allowed CPU mask **before** pinning the
coordinator (default CPU 4). If package/core topology is readable for every
allowed CPU, it groups by physical package and core and selects one logical
representative per physical core. It excludes the coordinator's whole physical
core, then reserves one additional core when possible. The remaining
representatives are worker CPUs, so worker count is derived from observation.

If complete topology cannot be read, the fallback uses unique allowed logical
CPUs, excludes the coordinator and reserves one extra CPU when possible.
It explicitly labels this logical fallback; physical separation is not claimed.
A one-unit environment uses one colocated worker and reports that no separate
worker/headroom layout was possible. The actual kernel uses min(N,selected CPUs)
workers. Counts, inherited mask, topology records, selected CPU list and reserved
representatives are emitted.

The coordinator pins only itself. Each new worker explicitly sets its own
selected affinity; it must not merely inherit the coordinator's CPU-4-only
mask and claim parallel execution. Worker affinity errors are not ignored.
The coordinator's CPU is checked before and after every record.

Only one benchmark region is active at a time; parallelism is internal to
cold multicore jobs. No competing local builds/tests/benchmarks run during
manager-authorized measurement. External processes, sibling activity and
thermal state remain uncontrolled.

Before and after the invocation, scaling_cur_freq and scaling_governor are
read for the coordinator and every selected worker CPU, plus intel_pstate
no_turbo when available. Missing values are null. Endpoints are observations,
not proof of fixed frequency throughout the region. No governor/turbo/system
setting is changed and no sudo is used.

## Correctness, CLI and evidence gates

Before timing, the separate native tests must pass independent Boost arithmetic,
unchanged public Scalar byte equality, canonical outputs, empty/singleton,
uneven contiguous partitions, tree odd tails, deliberate reduction boundaries,
input preservation, CPU argument rejection and comparator-negative controls.
Native GCC, portable-carry GCC, native Clang and UBSan are correctness
configurations, not four independent random corpora.

The driver itself self-tests decimal overflow/rejection, statistics, resource
caps, all 256 single-bit output corruptions and round position balance.
Its own complete-job preflight covers the actual selected dataset and CPUs;
it is not a substitute for the independent oracle tests.

~~~sh
f1_reduce_compare (--smoke|--measure) \
  [--count 256|4096|65536|1048576] [--cpu N] [--seed N] \
  [--target-ms 200..1000]
~~~

Unknown, duplicate, missing, negative and overflowing options are rejected.
Smoke uses two jobs per variant, the same two/eight round schedule, and
timing_claim=false. Smoke timing is functional evidence only; target options
are invalid in smoke mode.

~~~sh
g++ -std=c++17 -O3 -march=native -DNDEBUG -Wall -Wextra -pthread \
  -Isrc/cpu/include experiments/parseatlas_secp256k1/probes/f1_reduce_compare.cpp \
  src/cpu/src/scalar.cpp -o /tmp/REVIEWED_UNIQUE_DIRECTORY/f1_reduce_compare
/tmp/REVIEWED_UNIQUE_DIRECTORY/f1_reduce_compare --smoke --cpu 4 --count 256
/tmp/REVIEWED_UNIQUE_DIRECTORY/f1_reduce_compare --measure --cpu 4 \
  --count 256 --seed 20260905 --target-ms 200
~~~

Use mktemp -d for the unique exact output directory; the placeholder is not
literal. No LTO. The manager compiles, runs mechanical gates, reads source and
actual assembly, then authorizes sustained measurements. The driver prints one
JSON document and writes no evidence files. Its compiler version and selected
backend macros do not replace the external build-command/source/binary manifest.
Preserve raw timestamps without rounding their 64-bit decimal values.

Source counts include N or N+merge additions, tree scalar-element initialization
and copy, allocation requests, materialized output bytes and worker launches.
They are not dynamic instruction counts, allocator system-call counts or PMU
traffic. Serialize all records, exact job counts/order and full summaries.

Task MCP lifecycle remains temporarily suspended by the owner while native
preflight is repaired. Worker-scoped tools were not exposed; this worker used
only manager-authorized exact file targets. Manager Source Graph verification
remains in use. Only this driver and protocol were this worker's allowed writes.

All arithmetic and timing are native C++; no Python. A final-only sum may
resemble aggregation callers, but callers requiring constant-time arithmetic
or additional validation are not drop-in targets for this variable-time F1
experiment. No constant-time proof, novelty claim, production replacement,
all-prefix acceleration or whole-engine speedup follows from these results.

