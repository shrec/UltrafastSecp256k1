# Full scalar-order native addition protocol

This prospective C++ experiment asks whether a changed four-limb carry method
wins after the **complete existing variable-time addition modulo scalar order
n**, including the production early return and conditional reduction. It does
not inherit the historical synthetic recurrence's approximately 1.5x result.

## Contract and controls

The research header accepts two canonical integers in [0,n), returns their
canonical sum modulo n, and permits exact x += x aliasing. It retains the frozen
production scaffold from scalar.cpp at commit
fef231d4e4173bd016fb2a3a1eff67087396a203. Only the full 256-bit sum plus its high
carry changes across Original, Blocked2 and GP recomputed. These are three
matched inline methods in the same translation unit and measured region.

A fourth, separately labeled diagnostic invokes the **unchanged public
fast::Scalar::operator+=**, linked from the real scalar.cpp translation unit.
All Scalar initial values and RHS objects are constructed before timing, not
once per addition. This variant intentionally retains the API call boundary.
Its timing is reported, but no Original/API ratio is emitted as evidence for a
carry-algebra speedup. Build without LTO and inspect the actual linked binary:
source-level matching alone does not establish machine-code equivalence.

No candidate from this variable-time experiment may silently replace
ct::scalar_add. Correctness and constant-time eligibility are separate gates.
The owner authorizes a production replacement after appropriate correctness,
security and realistic-call-site performance gates; this research executable
does not change production, default builds, public interfaces or CI.

## Workload and locality dimensions

One invocation selects exactly one shape and one count:

- dependent: a single accumulator takes every RHS value in order;
- throughput: eight separately seeded accumulators take consecutive RHS values
  in an explicitly unrolled eight-way stripe, with no cross-lane dependency.

All eight initial accumulators are separately generated, even in dependent
mode; inactive lanes must remain unchanged. Every final lane's four limbs is
retained and compared. Eight 256-bit accumulators exceed the scalar general-
purpose register capacity, so throughput is an ILP-plus-state-traffic workload,
not isolated ALU throughput. Compiler allocation, spills, instruction scheduling
and vectorization remain free to differ between methods.

Count is **total modular additions per pass**, not additions per lane. Count
256 uses an 8 KiB RHS stream; count 4096 uses 128 KiB. Throughput divides this
same total work equally among eight lanes. These are prospective locality
controls, not PMU-confirmed cache-residency or DRAM-bandwidth measurements.
Separate Limbs and Scalar representations coexist, each with a 32-byte element
stride; only the selected representation is accessed inside each timed region.
Warmup touches the first 256 operands, so the 4096-element stream is not fully
prewarmed by the per-sample warmup alone.

Canonical RHS values and all initial states come from seeded SplitMix64 with
rejection of values >=n. There is no biased high-bit masking or timed random
generation. Both shapes repeat their RHS stream across passes, with state
continuing across pass boundaries. Since addition is associative, this workload
is mathematically collapsible: review the actual measured binary to establish
that the intended modular inner and outer loops remain. No volatile arithmetic,
per-operation compiler barriers or forced reloads artificially prevent legal
optimization.

Default seed is 20260905 and CPU is 4. Each sample resets identically and applies
the first 256 operands once using the same variant outside timing. The measured
region includes local state copy/unpacking, every modular addition, loop
control, any spills and complete final state writeback. Initialization, method
dispatch, warmup, comparisons, checksum and statistics are outside timing.
There are empty GNU compiler memory barriers at region boundaries only, not
hardware fences or arithmetic assembly.

## Correctness before timing

The separate native correctness executable must pass independent
Boost.Multiprecision and actual public Scalar comparisons, including reduction
boundaries that random input almost never reaches, aliasing and comparator
negative controls. Frozen Original remains an additional control. Native GCC,
portable-carry GCC, native Clang and UBSan are correctness configurations, not
four independent random corpora or four performance series.

The driver first compares all eight serialized little-endian lane values after
**each** operation of one full RHS pass for each of the three matched methods
against the actual public Scalar API. It checks that resulting active values
remain canonical. Emitted preflight path counts classify the Original prefix
into no reduction, high-carry reduction, and threshold reduction without carry.
These counts describe only that untimed initial prefix, not measured-trajectory
branch coverage. Random data will generally exercise high-carry reduction and
no reduction; the no-wrap threshold interval is extremely rare and belongs in
the separate deliberate correctness corpus.

After calibration, a complete untimed public Scalar replay uses the final
count/pass length plus the identical warmup. All eight lanes of the final
calibration, every external warmup and every measured sample are compared to
this replay by explicit byte equality, not digest equality. Intermediate
calibration states are retained but are **not** separately replay-checked.
The public API replay is not an independent mathematical oracle; the separate
Boost gate supplies that independence. A checksum over all 256 serialized bytes
is emitted only as durable evidence.

Every invocation self-tests unsigned parsing, odd/even/singleton statistics,
work overflow bounds, full-state comparator corruption and position balance.
Canonical generation and conversion checks occur before timing. Count must be
256 or 4096, passes is positive and at most 1,000,000, and total additions is at
most 1,000,000,000, checked by division before multiplication. The throughput
count is consequently a multiple of eight; the last i+7 is in bounds. The inner
region is deliberately unchecked and is called only after these outer gates.

## Registered timing schedule

The strict CLI is:

~~~sh
modn_add_compare (--smoke|--measure) \
  [--shape dependent|throughput] [--count 256|4096] \
  [--cpu N] [--seed N] [--target-ms 200..1000]
~~~

Shape defaults to dependent and count to 256. Unknown, duplicate, missing,
negative and overflowing options are rejected with exit 2. Smoke rejects a
target option, uses two common passes, and emits timing_claim=false. Its clock
values and summaries are functional diagnostics, never performance evidence.

Measurement calibrates Original to at least the target (default 200 ms), using
at most eight trials and the work caps above. The resulting count and passes
are shared by all four variants. Each invocation retains every calibration
trial, two complete external warmup rounds and eight measured rounds. A seeded
four-variant permutation rotates once per round, putting each variant in each
position exactly twice over the eight measured rounds. Smoke retains the same
round schedule, so order and serialization machinery are exercised too.

Calibration does not guarantee that later samples, especially faster
candidates, exceed the requested target. Below-target samples are explicitly
labeled and counted; they are not discarded, lengthened or selected away.
Raw elapsed times, start/stop times, CPU endpoints, exact order, count, passes,
work count and checksums remain visible. Per-variant native summaries report
median, min/max, MAD and inclusive IQR. Only the matched three methods receive
same-round Original/candidate ratios and all eight individual ratios. Above
one favors that candidate in that pair; ratios are descriptive, not confidence
intervals or a production acceptance decision.

Before any measurement, the manager freezes and hashes the source, headers,
tests, protocol and binary, runs mechanical gates, reads source and actual
assembly, and records an exact run plan. The planned matrix is two separate
series of dependent/throughput crossed with count 256/4096: eight invocations,
four variants times eight measured rounds per invocation, 256 measured samples
in total. No pooling or best-series selection is permitted; all results,
including negative results, remain evidence.

Only this process's affinity changes. It first checks inherited affinity,
pins to the selected CPU, and checks the CPU before/after every sample. One
serial measurement worker is intentional to avoid self-interference, not a
multicore scaling result. No build, competing test or benchmark should run
during the registered measurement window. External activity, sibling activity,
frequency and thermal variation remain possible; the manager records observed
conditions without changing global settings or using sudo.

## Reproduction and evidence custody

~~~sh
g++ -std=c++17 -O3 -march=native -DNDEBUG -Wall -Wextra \
  -Isrc/cpu/include experiments/parseatlas_secp256k1/probes/modn_add_compare.cpp \
  src/cpu/src/scalar.cpp -o /tmp/REVIEWED_UNIQUE_DIRECTORY/modn_add_compare
/tmp/REVIEWED_UNIQUE_DIRECTORY/modn_add_compare --smoke --cpu 4 \
  --shape dependent --count 256
/tmp/REVIEWED_UNIQUE_DIRECTORY/modn_add_compare --measure --cpu 4 \
  --shape dependent --count 256 --target-ms 200
~~~

Create the exact unique output directory with mktemp -d; the placeholder is not
literal. Link the unchanged scalar.cpp separately, without LTO or alternate
production definitions. Record the complete build command and flags externally;
__VERSION__, __cplusplus and selected int128 macros are emitted by the driver.
The executable writes one JSON document to stdout and does not self-attest its
source hash. The manager preserves raw stdout and binds source, reference,
binary, protocol, test, command and machine identities in an evidence manifest.

The owner temporarily suspended Task MCP lifecycle use while its native
preflight blocker is repaired. This worker's tool discovery found no exposed
worker-scoped Source Graph/session tools; only manager-provided exact targets
were read. Manager Source Graph verification remains in use; this is not a
claim that Source Graph is unavailable globally. Only this protocol and its
driver are this worker's allowed writes.

All arithmetic, tests and timing are C++; no Python is used. This experiment
does not establish novelty, full-domain proof, constant-time safety, PMU-derived
memory costs, all 25 lenses, or whole-engine acceleration. Its question is
narrow: does the raw carry candidate survive full variable-time modular
reduction in these two explicitly defined dependency/locality contexts?

