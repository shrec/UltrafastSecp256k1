# Native C++ L0 comparison protocol

This preregistration fixes the first candidate comparison, not a winner. All new
correctness tests, benchmark arithmetic, orchestration, clocks and statistical
summaries run in C++. No Python participates in this experiment. Historical
Python structural models remain mathematical exploration, not CPU timings.

## Frozen boundary and candidates

`probes/l0_native_kernels.hpp` defines a complete four-limb unsigned addition:
`a + b + carry_in = low + carry_out * 2^256`, with both carries in `{0,1}` and
little-endian 64-bit limbs. This is not Fp/Fn modular reduction, an engine-wide
benchmark, arbitrary aliasing support or a constant-time proof.

The original method directly composes the engine's `detail::add64` from
`src/cpu/include/secp256k1/detail/arith64.hpp`. The comparison includes three
source-level carry representations: `gp_materialized`, `gp_recomputed` and
`blocked2`. Their equations and independent Boost.Multiprecision correctness
gate live with the kernel/test artifacts. All methods preserve the same complete
input/output boundary. This turn tests neither deferred normalization across
multiple additions nor an alternative bulk data layout.

The GP materialization distinction describes source-level local values. It does
not promise array writes, register residency, differing machine code or fewer
actual memory transactions. Compiler optimizations can make candidates
identical. The actual compiled binary must therefore be disassembled before
interpreting a difference or lack of difference. The identical `noinline`
timed-region wrapper policy preserves symbols without adding method dispatch to
the per-addition loop; the arithmetic remains inline inside those wrappers.

## Native correctness gates

Build and run the separate native kernel correctness executable before sustained
measurement. It compares complete results against an independent unbounded
integer oracle and the existing implementation. Its finite observations are
not promoted into an exhaustive 256-bit proof.

The measurement driver additionally validates **every generated bulk input**
against Original for every method before any timing. All four low words and the
full carry word must agree; checks remain enabled under `-DNDEBUG`. Bulk checks
use the original implementation, not an independent arithmetic oracle. The
driver compares every final timed output checksum with the original dataset's
checksum; it never regards a checksum alone as pre-timing correctness evidence.

For conditioned latency, every method's first 4096 recurrence states (64 in
smoke mode) are compared with Original, including all low words, carry and
conditioned `b`. After calibration, an untimed Original replay covers the full
final recurrence length. Every warmup and measured sample must match that final
checksum. This full-run digest test is weaker than comparing every internal
state and is not claimed to be an independent oracle.

Every invocation self-tests inclusive-quantile statistics with odd, even and
singleton fixtures and validates the unsigned-decimal parser. `--smoke` runs all
four methods across all three tiny regimes, one external warmup round and four
sample rounds; its JSON is explicitly **not performance evidence**. Invalid
CLI, unsupported CPU, bad carry, mismatches or resource failures return code 2,
not a partial accepted result.

## Workloads and timed region

| Regime | Measurement configuration (research-only) | Timed work |
|---|---|---|
| Conditioned latency | 4096 dependent additions per pass | All limbs and carry feed the next addition; each `b[j] = rotl64(b[j],13) XOR new_a[j]` |
| Hot bulk | 219 AoS records; 24,528 logical bytes | Independent complete four-limb additions |
| Large bulk | 748,983 AoS records; 83,886,096 logical bytes | Exactly the same bulk loop over a larger dataset |

These fixed sizes inherit the recorded CPU4 baseline (48 KiB L1d and 20 MiB LLC).
They do not dynamically prove hot-cache or LLC-exceeding status on a different
machine; the manager records actual cache topology and labels any mismatch.
Smoke sizes are 16 dependent steps, 7 hot records and 257 large records, with two
passes, and must never be interpreted as cache-regime measurements.

An input record is 72 bytes and an output record 40 bytes; a compile-time check
enforces those layouts. Logical bytes per bulk addition are 112. Such accounting
does not measure cache-line transfers, write allocation, prefetching or DRAM
bandwidth. Latency's logical state is 72 bytes and logical streamed bytes are
zero; this does not assert zero machine loads/stores or spills.

Allocation and deterministic SplitMix64 initialization happen outside the timed
region after CPU pinning. The seed defaults to 20260905 and is shared by all
methods. Every sample gets one internal warmup pass (or up to 4096 latency steps)
outside its clock interval. Latency is reset to identical generated state before
that warmup on every invocation. Loop control, conditioning and any compiler
spills remain inside the measured region. Bulk uses an opaque compiler memory
barrier after each pass to preserve all output stores; it emits no hardware
fence. Checksumming all output words and carry is outside the clock.

## Calibration, ordering and interpretation

Original calibrates each regime to at least 200 ms by default, with at most eight
trials, one million passes and one billion additions per sample. `--target-ms`
accepts 200 through 1000. All methods use the same final count and pass count for
that regime. A faster candidate or a changing environment can produce samples
below the requested target; raw JSON retains and counts them. They are not
silently extended, dropped or relabeled. An unreachable target is an explicit
failure.

Two full external warmup rounds and seven measured rounds follow. Every round
contains Original and all three candidates. A seeded initial permutation rotates
by one position each round, covering every position once per four rounds; seven
rounds are not perfectly position-balanced. Warmup and measured phases each
start from that same preregistered permutation. The regime order is fixed:
conditioned latency, hot bulk, large bulk. Only methods within each regime are
interleaved. No data-dependent method ordering or best-run selection occurs.

JSON preserves every calibration, warmup and measured elapsed interval, count,
pass count, CPU endpoints, checksum, actual order and below-target status.
Summaries contain median, minimum, maximum, median absolute deviation and
inclusive interquartile range. The paired ratio for a round is
`Original ns/addition / method ns/addition`: values above one favor the method
for that observed pair. Pair samples and their descriptive summaries are
retained. Ratios are not confidence intervals, proof of causation or an automatic
acceptance decision. Do not compare candidates only to a historical best sample.

Measurement is deliberately one serial worker: parallel benchmark execution
would interfere with the cache/bandwidth/frequency conditions being measured.
Linux affinity is checked against the caller's allowed mask, then only this
process is pinned before dataset first-touch. Default CPU is 4, overridable with
`--cpu`. No shell/other-process affinity, governor, turbo, sysctl, security
setting or other global state is modified. External load, sibling activity and
thermal/frequency variation are not controlled and must remain reported limits.
The aggregate live dataset allocation is below 256 MiB; workloads are released
between regimes. This is a one-core experiment, not a multicore throughput test.

## Build, evidence and reproduction

Compile with GCC using the same reference flags as the original baseline:

```sh
g++ -std=c++17 -O3 -march=native -DNDEBUG -Wall -Wextra \
  -Isrc/cpu/include experiments/parseatlas_secp256k1/probes/l0_native_compare.cpp \
  -o /tmp/REVIEWED_UNIQUE_DIRECTORY/l0_native_compare
/tmp/REVIEWED_UNIQUE_DIRECTORY/l0_native_compare --smoke --cpu 4
/tmp/REVIEWED_UNIQUE_DIRECTORY/l0_native_compare --measure --cpu 4 --target-ms 200
```

Create the unique output directory with `mktemp -d`; the placeholder is not a
literal pre-existing directory. The executable prints one JSON document to
stdout and writes no result file. The manager persists that output with an
external envelope containing the source, frozen engine/reference and binary
SHA-256 identities, exact compiler arguments, compiler target and machine
metadata. The JSON embeds compiler version, language version, backend, seed,
configuration and arithmetic record sizes, but does not independently attest
that external envelope. Build and independent test/disassembly review precede
the declared quiet measurement window. No competing build/test/benchmark should
be launched during that window.

PMU counters are not collected here. Any separately authorized whole-process
`perf` experiment must keep its setup/checksum/process scope distinct from this
inner timer. There is no sudo or credential handling in this driver. No result
alone establishes cryptographic constant-time behavior, novelty, generic
25-world superiority or a production acceptance.
