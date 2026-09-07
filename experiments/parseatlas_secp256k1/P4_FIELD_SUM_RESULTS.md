# P4: native 4x64 final-only field-sum controls — 2026-09-05

Status: two native C++ series completed; arithmetic, duration and source/codegen
gates passed. Retain experimental candidates; production integration is deferred.
Field research is not complete. Independent numeric audit and final review are
linked below; neither a whole-engine gain nor CT certification is claimed.

## Result and observation contract

For canonical field-p values, return (x0 + sum(N RHS)) mod p as canonical raw
FE64 and all32big-endian bytes. N counts RHS; x0 occurs once globally. Only the
final result is observable. These are complete vector-sum timings, not standalone
field-add latencies or prefix-output timings.

- At every measured N>=16, all seven candidates beat the actual corrected FE64
  API sum in all same-round pairs: **560/560** across two series.
- Inline eager beats that baseline at N=1 by1.127/1.137x, winning16/16pairs.
  All other six candidates have losing N=1 paired medians (10isolated wins/96).
- Native Wide4094/lane1 gives7.663/7.866x atN16,15.232/14.825x atN256,
  16.782/16.728x atN4096 and14.495/14.605x atN65536.
- AtN1048576, native Wide16/lane1 gives8.086/7.894x and has the best median
  baseline ratio among the eight measured routes in both series.
- Four banks have losing same-chunk paired medians against one bank at EVERY
  measured N in both series. Their isolated wins total only8/192 comparisons.

The selected schedules above summarize the observed matrix after measurement;
they do not establish unmeasured crossover thresholds or a production dispatcher.
All comparisons below were preregistered, and losing observations are retained.
The primitive's representation need not change to benefit from deferred carry
and canonicalization. That conclusion does not isolate a single instruction cost.

## All primary paired comparisons

Median of eight SAME-ROUND FE64-API/candidate ratios, S1 / S2; above one favors
the candidate. Jobs can differ between regions; ratios use actual elapsed/jobs.
A median of ratios is not a ratio of independently computed medians.

| N RHS | Inline eager | Wide16 / 1 | Wide4094 / 1 | Wide16 / 4 | Wide4094 / 4 | FE52 Full16 | FE52 Full4094 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.127 / 1.137 | 0.974 / 0.960 | 0.954 / 0.917 | 0.753 / 0.725 | 0.750 / 0.759 | 0.889 / 0.877 | 0.869 / 0.871 |
| 16 | 2.031 / 2.087 | 7.446 / 7.593 | 7.663 / 7.866 | 4.547 / 4.702 | 4.190 / 4.295 | 5.416 / 5.362 | 5.447 / 5.474 |
| 256 | 2.314 / 2.302 | 10.296 / 10.240 | 15.232 / 14.825 | 6.544 / 6.515 | 9.626 / 9.798 | 9.670 / 9.362 | 12.052 / 11.930 |
| 4096 | 2.286 / 2.287 | 11.416 / 11.314 | 16.782 / 16.728 | 6.578 / 6.584 | 9.617 / 9.623 | 9.222 / 9.360 | 12.494 / 12.561 |
| 65536 | 2.331 / 2.328 | 11.314 / 11.198 | 14.495 / 14.605 | 6.535 / 6.548 | 9.862 / 9.598 | 9.297 / 9.467 | 12.557 / 12.502 |
| 1048576 | 2.265 / 2.261 | 8.086 / 7.894 | 5.943 / 6.005 | 5.976 / 5.722 | 5.720 / 5.822 | 7.718 / 7.484 | 6.376 / 6.145 |

Every route enters with the same FE64 RHS array. Native wide uses four u128
columns per bank in radix2^64; FE52 controls pack at use inside the timed sum.
No preconverted resident FE52 array is allocated in P4. Allocation and corpus
generation are excluded for every route; seed handling, loads, boundary work,
whole-job calls and final32-byte materialization are included.

## Absolute normalized region means

Median ns/RHS, S1 / S2. Multiply by N for whole-sum ns. These normalized means
include amortized whole-job work and are not single-add latency measurements.

| N RHS | FE64 API | Inline eager | Wide16 / 1 | Wide4094 / 1 | Wide16 / 4 | Wide4094 / 4 | FE52 Full16 | FE52 Full4094 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 11.312 / 11.056 | 10.176 / 9.824 | 11.625 / 11.675 | 11.853 / 12.405 | 15.308 / 15.313 | 15.010 / 15.028 | 12.947 / 12.816 | 13.361 / 12.883 |
| 16 | 8.820 / 9.093 | 4.387 / 4.443 | 1.177 / 1.219 | 1.182 / 1.169 | 1.926 / 1.944 | 2.129 / 2.109 | 1.643 / 1.720 | 1.627 / 1.676 |
| 256 | 10.389 / 9.967 | 4.444 / 4.373 | 0.998 / 0.986 | 0.678 / 0.668 | 1.551 / 1.567 | 1.068 / 1.014 | 1.076 / 1.033 | 0.845 / 0.832 |
| 4096 | 10.052 / 9.940 | 4.393 / 4.415 | 0.882 / 0.876 | 0.603 / 0.598 | 1.513 / 1.543 | 1.027 / 1.034 | 1.106 / 1.087 | 0.822 / 0.802 |
| 65536 | 10.236 / 10.024 | 4.379 / 4.340 | 0.919 / 0.902 | 0.680 / 0.705 | 1.572 / 1.532 | 1.033 / 1.059 | 1.106 / 1.070 | 0.810 / 0.805 |
| 1048576 | 10.340 / 10.111 | 4.640 / 4.485 | 1.294 / 1.316 | 1.742 / 1.720 | 1.732 / 1.730 | 1.805 / 1.727 | 1.309 / 1.366 | 1.650 / 1.620 |

ForN1048576, FE64-API sum medians are10.842/10.603ms, versus1.356/1.380ms
for native Wide16/lane1. These separate medians illustrate elapsed cost;
the ratio table still uses the paired estimator.

## What the controlled comparisons show

### API implementation is not the whole explanation

Inline eager removes production per-RHS arithmetic calls and keeps raw limbs
until final FE construction. It improves the complete sum by about2.03–2.33x
atN>=16. This also changes instruction selection and temporary placement, so it
is NOT a pure identical-code call-overhead ablation.

Native deferred Wide4094/lane1 still beats inline eager atN4096 by
7.278/7.352x in same-round ratios (16/16wins). Thus the final-only advantage
survives a native inlined eager control; it is not solely the old API boundary.

### Four banks are a measured negative result here

Equal-chunk one-bank/four-bank comparisons have all24series medians below one.
The four-bank implementation has extra bank state/merge work; static inspection
also finds explicit stack-memory accumulators in the large four-bank loop.
It remains scalar add/adc, not packed SIMD arithmetic. Those observations do
not measure the cost of spills, memory bandwidth or any isolated instruction.
They reject this particular four-bank scheduling candidate as the preferred
schedule on this profile, not mathematical parallelizability in general.

### Equal chunk sizes do not produce a universal representation winner

AtN4096, native Wide4094/lane1 versus FE52 Full4094 has paired native/FE52
ratios0.743/0.743: native wins16/16pairs. AtN1048576, the direction reverses:
the same native/FE52 ratio is1.065/1.041 and FE52 wins14/16pairs.

AtN1048576 and chunk16, native lane1 has paired native/FE52 ratios0.953/0.953
and wins12/16pairs (5/8then7/8). This modest, mixed-pair advantage is distinct
from the much larger baseline-API gains; do not call it a categorical winner.

Changing chunk4094 to chunk16 atN1048576 benefits native lane1 in16/16pairs.
More normalization boundaries can coexist with less total elapsed time.
When N does not reach either bound, different template chunks can still select
different compiled paths; equal actual boundary counts do not guarantee equal
machine code.

All secondary comparisons are fixed in the [protocol](P4_FIELD_SUM_PROTOCOL.md):
API/inline, inline/native-deferred, bank1/bank4, chunk16/chunk4094 and matched
native/FE52 pairs. The raw JSON preserves all13pairs at each of six sizes.
No P3/P2/P1/F2 timing is pooled into P4.

## Machine code, not guessed costs

The [codegen report](P4_FIELD_SUM_CODEGEN.md) and its
[20 retained disassemblies](data/p4_codegen_20260905.json) distinguish:

- FE64 API per-RHS operator+/add_impl calls and accumulator-copy work.
- Inline eager scalar correction without per-RHS calls, but with stack operands.
- Native Wide4094/lane1: four scalar add/adc column pairs, one32-byte RHS per
  iteration, no calls or explicit rsp/rbp memory operands INSIDE that loop.
- Native Wide4094/lane4: four RHS/128bytes per iteration; scalar add/adc includes
  stack-memory bank accumulators. Chunk16 native paths are unrolled scalar code.
- FE52 Full4094: four FE64 inputs/128bytes per vector iteration, packing plus
  five YMM packed additions; Full16 also retains packed arithmetic.

Surrounding merges/reduction/output can still access the stack even where a
selected inner loop does not. Small outlined wrappers are not whole-kernel
sizes. The manager replayed all20disassembly ranges and20symbol extents against
the exact GCC binary. Repeated jobs, continuous-region extensions and complete
32-byte output stores remain present.

No dynamic cycles/instructions, cache misses, bandwidth, port pressure, energy,
critical-path depth or spill-cost attribution follows. PMU access remains
unavailable under the previously measured policy; no settings were changed.

## Correctness and native bounds

The [independent written bound argument](P4_FIELD_SUM_CONTRACT.md) admits
native chunk1..4095RHS and exactly1or4banks. Every chunk adds one canonical
prefix seed to its zero-based RHS banks. AtM<=4096totalterms, merged columns
and carry intermediates are below2^76, propagated high<=4095, and high*K<2^45.
Two folds produce a value below2^256; one p correction canonicalizes it.
The original x0 is not added again at later chunks or once per bank.

These bounds apply only to canonical sums. They are not FE52 direct-normalizer
bounds, arbitrary u128 reducer inputs, subtraction/negation or product bounds.
N0/null returns x0; positive N needs valid canonical ranges; inputs are read-only
and same-type x0/RHS aliasing is supported. There is no positive-null rejection
promise or returned output-buffer overlap contract.

The same finite C++ corpus passed three manager-run configurations:

1. Native GCC14 with the corrected frozen P1 library.
2. Clang18 independently compiling field sources with the retained GAS object.
3. GCC NO_ASM, ASan+UBSan, leak checking and halt-on-error.

Each:148fixtures,136aliases,72nullidentities,284inline calls and2272wide calls;
**21,130,305 assertions, zero mismatches**. Independent Boost ordinary-integer
prefix sums modulo p plus corrected FE64 compare exact canonical raw limbs and
all32bytes. Input checksum9bf04085e07140c8; output checksum b1c845bb4263e586.
The147ordinary-prefix fixtures plus one targeted witness share nine corpora.
Repeated compiler runs are not additional disjoint fixtures.

Bounded actual-helper replay covers907162chunks, including9071620bank-u128
and3628648merged-column checks. It exercises first-fold carry six times using
the valid three-term sum2B-1 and canonical subtraction60098times. Helpers run
separately, not as instrumented timed kernels; the three large fixtures check
complete outputs/preserved inputs without replaying every checkpoint.
Counts overlap. Chunk0,chunk4096 andlanes2fail at the intended compile assertions.
See [arithmetic validation](data/p4_validation_20260905.json).

## Duration qualification and raw evidence

Both runs have48cells,96warmup and384measurement regions. Calibration attempts,
short prefixes, probe timestamps, actual job counts and outputs are retained.

- S1:652regions =172calibration +96warmup +384measurement.
  74short calibration samples;21later regions each append one extra batch.
  Measured region durations200.853844–583.101070ms.
- S2:649regions =169calibration +96warmup +384measurement.
  73short calibration samples; no later extensions.
  Measured region durations206.998300–448.504119ms.
- Combined:1301regions;960warmup/measurement regions ALL reach200ms.
  No region clock was restarted, no short prefix discarded and no selective rerun.

Every later region uses one continuous clock and appends work if needed.
Actual total jobs determine ns/sum and ns/RHS. Intermediate clock checks,
bookkeeping and continuation work inside the continuous interval are timed;
final JSON formatting and probe conversion are outside it.

A separate [duration diagnostic](data/p4_duration_validation_20260905.json)
passed3163assertions in GCC and Clang. It interposes fabricated monotonic ticks
ONLY in a separate test executable to exercise extension success, extension-limit
failure, invalid time and preflight work-cap rejection. The actual renamed driver
main correctly exits2and preserves97regions after a synthetic first-warmup failure.
These synthetic clock values are NOT performance evidence and are never pooled.
Within-region work-cap/capped-final-batch paths were statically reviewed, not
dynamically exercised by that small diagnostic.

## Environment, reproducibility and remaining gates

Full series: Intel i5-14400F, CPU4; GCC14.2, C++20, native ISA, LTO OFF.
Native assembly/fast reduction are enabled in the frozen corrected profile.
Power governor remains powersave; turbo remains enabled. S2 endpoint frequency
changes from4100000to4600046kHz, not an average-frequency measurement.
All region CPU endpoints are4. No owned build/correctness/competing benchmark
overlapped either full series. External services, indexing, thermals and SMT
sibling activity were not controlled; core endpoints do not prove constant clocks.

The FE64 corpus recipe/seed20260905/full-byte checksum a228d8a510693ab4 matches
P3, but the performance records and binary are new. Actual N-element prefixes
reach32MiB; there is no small cyclic RHS table. Inputs are reconstructed from
the source-bound recipe and regenerated for post-validation/post-region checks.

Raw [S1](data/p4_run1_20260905.json) / [S2](data/p4_run2_20260905.json);
[independent numeric audit](data/p4_numeric_audit_20260905.json);
[build/source manifest](data/p4_build_20260905.json);
[manager final review](data/p4_final_review_20260905.json).
The independent numeric audit passed97,816checks with zero errors, covering
all1301rawregions,1322cumulativeprobes and240preregistered comparison summaries.
Source Graph new-file/stale intermediate generations and missing worker-role
tools are recorded in [obstacles](data/p4_environment_obstacles_20260905.json);
final source receipts match frozen files. No stale body is used as final evidence.

The [25-lens outcome map](P4_FIELD_SUM_LENS_RESULTS.md) and
[central frontier](RESEARCH_FRONTIER.md) collect this with earlier P3/P2 field
and F2 scalar evidence without replacing or merging their contracts.
Next field layer: multiply/square sparse-prime reduction with its own accumulator
bounds and chain/independent measurements, then inversion/batch-inversion.
Compiler/platform, setup-inclusive and security qualification remain open.

No production optimization, scalar/upper-layer experiment, default build/CI,
commit, push or tag change was made in P4. Owner Task MCP suspension remains;
this is local independent manager review, not canonical task acceptance.
