# Native C++ crossed state-form control

This prospective control asks whether the earlier approximately 1.5x conditioned
recurrence result survives a fair change in how **all four** arithmetic methods
express their live state. It does not assume that the result transfers to a
production modular operation. All arithmetic, correctness gates, timing,
orchestration and summary statistics are native C++; no Python is used.

## Fixed boundary and eight variants

The experiment crosses the frozen `original`, `gp_materialized`, `gp_recomputed`
and `blocked2` equations with two source forms:

- `aggregate`: a local `Input` state and the frozen `compute<M>` result aggregate.
- `scalar_local`: nine named scalar locals, with inlined scalar equations and no
  per-step construction of an `Input` or `Output` aggregate in the source.

Each step computes the complete four-limb integer sum `A+B+carry_in`, retains the
low 256 bits and the outgoing bit, then conditions every second-operand limb as
`b[j] = rotl64(old_b[j],13) XOR new_a[j]`. All new `a` limbs, all conditioned `b`
limbs and the outgoing carry feed the next step. Carry is always in `{0,1}`.
This is not addition modulo the field prime or scalar order. The conditioning
recurrence is an experimental workload, not a claim about a production call site.

`advance<M,F>(state,count,passes)` unpacks once, runs `count` steps per pass with
state continuing across pass boundaries, and writes back the full final state
once. Both forms include unpack and writeback inside the measured region.

The region-local Aggregate copy with one final writeback is itself a new control:
the historical harness mutated its referenced state directly. Its arithmetic
equations are frozen, but its generated code is not presumed identical to that
historical baseline. Only this executable's same-round controls determine the
crossed comparison; historical timing is context, not an interchangeable control.

A zero count or zero passes is the identity within the individually bounded domain:
`count <= 10^9`, `passes <= 10^6`, and `count*passes <= 10^9`. The checked external
gate uses division to avoid overflow. The inner API is unchecked and requires
that gate's carry and count/pass preconditions; timed samples must be nonempty.
Scalar-step references point to distinct local values, not arbitrary overlapping
buffers. There is no pointer-overlap compatibility claim.

Scalar-local syntax does not guarantee scalar machine instructions, fewer
registers, no spills or different code. Ordinary compiler CSE/vectorization may
merge source distinctions. No `volatile` arithmetic, forced reloads, hardware
fences, assembly arithmetic or per-step optimization barriers create artificial
work. A noinline timed wrapper exists per method/form; method/form dispatch is
outside the timer. Actual binary disassembly is required before interpretation.

## Correctness before measurement

Build and run `tests/test_l0_state_control.cpp` first. It tests all eight variants
against the frozen Original and a separate Boost.Multiprecision arithmetic
oracle, comparing explicitly serialized little-endian `a`, `b` and carry, not
struct padding or a digest alone. It covers boundary/random inputs, every step
of seeded trajectories, full multi-pass regions, flattened and per-pass
partitions, zero identities, invalid carry/cap gates and deliberate comparator
corruptions. Checks stay live under `-DNDEBUG`. This finite corpus is not a
full-domain 256-bit proof or a constant-time proof. GCC native, portable-carry,
Clang native and UBSan builds are correctness gates, not performance series.

The driver additionally compares every variant's first 4096 recurrence states
(64 in smoke mode) against Original, including all state words. After calibrating,
it executes a complete untimed Original replay at the final count/pass length.
Every full warmup and measurement result is compared **exactly** to that replay's
complete state outside timing; the emitted checksum is evidence serialization,
not the acceptance test. Intermediate calibration trials are retained but only
the final calibration trial is compared to that final-length replay. The prefix
and full replay use Original, not an independent mathematical oracle; the
separate correctness executable supplies that independent gate.

Every invocation self-tests unsigned parsing, odd/eight-value/singleton
statistics and the eight-round position balance. Unknown, duplicate, missing,
negative or overflowing CLI values, invalid affinity, failed gates or resource
limits return exit 2; there is no partial accepted JSON. `--smoke` is functional
validation only and its JSON explicitly says it is not performance evidence.

## Timing and registered schedule

Default seed is 20260905 and default CPU is 4. The process first verifies the
CPU is in its inherited allowed mask and pins **only itself**, checking CPU
before and after each sample. No system setting, other process, sudo credential,
governor or turbo value is changed. One serial measurement worker deliberately
avoids benchmark self-interference; this is not a multicore throughput test.
External load, sibling activity, frequency and thermal state are uncontrolled.

Measurement uses `count=4096`. Original Aggregate calibrates a common pass count
to at least 200 ms (`--target-ms` accepts 200..1000), at most eight trials and
within the stated caps. All eight variants receive the same count and passes.
Faster candidates or changed conditions can yield below-target samples; those
samples remain visible, counted and retained. No sample is silently lengthened,
dropped or selected as the preferred run.

Every sample starts from the same generated state and executes a same-variant
warmup of `min(count,4096)` steps outside timing. That warmup leaves the same
conditioned initial state for the timed region. The clock includes state unpack,
all arithmetic, rotate/XOR conditioning, loop control, any spills and complete
final writeback. It excludes generation, dispatch, warmup, correctness checking,
checksumming and summaries. Empty GNU compiler memory barriers appear only at
the region boundaries and are not hardware fences.

Two full external warmup rounds precede eight measured rounds. A seeded initial
permutation of all eight variants rotates by one position every round. The eight
measured rounds therefore put every variant in every position exactly once.
Warmup and measurement phases each start from that same permutation. Smoke mode
uses count 16, passes 2, one external warmup round and eight sample rounds; it has
no target and must not be interpreted as a performance comparison.

The manager preregisters **two separate same-CPU measurement invocations** after
source, correctness and disassembly review. No competing test/build/benchmark
starts during that measurement window. Both invocations, all raw samples and
outliers are retained. There is no best-series selection or pooling claim.

JSON reports count, passes, times, CPU endpoints, checksum, actual variant order,
all calibration/warmup/sample records and below-target status. Per-variant native
summaries report median, min/max, MAD and inclusive IQR. Each round supplies two
ratios: `Original Aggregate / variant`, and `Original of the same form / variant`.
Both are elapsed ns/addition ratios; above one favors that variant in that pair.
The raw per-round ratios are retained. These are descriptive ratios, not
confidence intervals, causal proof or automatic production acceptance.

## Build and evidence

```sh
g++ -std=c++17 -O3 -march=native -DNDEBUG -Wall -Wextra \
  -Isrc/cpu/include experiments/parseatlas_secp256k1/tests/test_l0_state_control.cpp \
  -o /tmp/REVIEWED_UNIQUE_DIRECTORY/test_l0_state_control
g++ -std=c++17 -O3 -march=native -DNDEBUG -Wall -Wextra \
  -Isrc/cpu/include experiments/parseatlas_secp256k1/probes/l0_state_compare.cpp \
  -o /tmp/REVIEWED_UNIQUE_DIRECTORY/l0_state_compare
/tmp/REVIEWED_UNIQUE_DIRECTORY/test_l0_state_control
/tmp/REVIEWED_UNIQUE_DIRECTORY/l0_state_compare --smoke --cpu 4
/tmp/REVIEWED_UNIQUE_DIRECTORY/l0_state_compare --measure --cpu 4 --target-ms 200
```

Create the unique output directory with `mktemp -d`; the placeholder is not a
literal directory. The executable prints one JSON document, writes no result
files and does not self-attest source identities. The manager persists stdout
with source/header/protocol/test, reference commit and binary SHA-256 identities,
compiler flags and machine conditions in a separate evidence envelope. The
frozen original header remains unchanged. No PMU counters are collected. A null
or negative crossed-control result remains valuable: it limits the applicability
of the earlier gain and prevents confusing source packing with a general new
arithmetic result. There is no novelty, generic 25-lens superiority, modular
arithmetic speedup or whole-engine speedup claim here.

This experiment is diagnostic rather than a permanent production restriction.
The owner authorizes replacement when a candidate wins appropriate production
correctness, security and realistic-workload performance gates; this synthetic
recurrence alone does not satisfy those acceptance gates.
