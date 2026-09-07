# P3 field sum: results through 25 lenses

Status: P3 evidence collected; **field research is not finished**. No production
optimization, scalar change or upper-layer integration follows from these results.
The preregistered [contract](P3_FIELD_SUM_CONTRACT.md) and
[protocol](P3_FIELD_SUM_PROTOCOL.md) remain unchanged. This document records
25 observations, not 25 discoveries or wins; see the
[main result report](P3_FIELD_SUM_RESULTS.md) for the full measurement account.

Labels: **MEASURED** denotes recorded C++ outputs/timings; **DERIVED** denotes
the stated mathematical or source-level argument; **STATIC** denotes inspection
of the frozen binary, not dynamic hardware counters. Unresolved numerical
quantities remain **PENDING IMPORT**. Proposed explanations are hypotheses.

## What was measured

The contract is the final canonical value `(x0 + sum(rhs[0..N))) mod p`, with
N counting RHS operands and x0 included once. It is not one field addition,
an observable-prefix API, or an engine benchmark. Seven routes use the same
actual corpus prefixes at N = 1, 16, 256, 4096, 65536 and 1048576.

Both full series use GCC 14.2, C++20, native assembly, LTO off and the corrected
FE64 library on CPU4. Ratios below are medians of eight **same-round normalized
whole-region means**, FE64/candidate; they are not ratios from unrelated runs
or individual-call latency percentiles. Seed packing and final 32-byte output
are timed. E2E routes also pack each RHS inside the sum; resident excludes RHS
preconversion. Allocation and corpus construction are excluded for all routes.

| Selected route and size | Series 1 | Series 2 | Interpretation |
|---|---:|---:|---|
| E2E Full256, N = 16 | 5.81x | 5.64x | Full conversion-inclusive sum |
| E2E Full256, N = 256 | 11.92x | 11.66x | Full conversion-inclusive sum |
| E2E Full4094, N = 4096 | 12.50x | 12.56x | Full conversion-inclusive sum |
| E2E Full4094, N = 65536 | 12.58x | 12.56x | Full conversion-inclusive sum |
| E2E Full16, N = 1048576 | 7.38x | 6.97x | Best median among the timed E2E routes at this size |
| Resident Weak4095, N = 256 | 29.47x | 30.13x | RHS preconversion excluded; not E2E |
| Resident Weak4095, N = 1048576 | 5.81x | 5.63x | RHS preconversion excluded; not E2E |

Across both series, all six candidates beat FE64 in **480/480 same-round pairs
at N >= 16**. At N = 1, every candidate's per-series median ratio is below one;
only 4 of 96 individual pairs favor a candidate. Small-input losses are retained.
Raw evidence: [series 1](data/p3_run1_20260905.json),
[series 2](data/p3_run2_20260905.json).

Exploratory, post-hoc candidate comparisons at N = 1048576: Full16 beats
Full4094 in 15/16 same-round pairs and E2E Weak4095 in 15/16; E2E Weak4095
beats resident Weak4095 in 15/16. These are measured ordering observations,
not a preselected universal policy or proof of cache/bandwidth causality.

### Retained timing deviation and finite validation

Series 1 retains 569 regions: 149 calibration, 84 warmup and 336 measured.
Series 2 retains 573: 153 calibration, 84 warmup and 336 measured. Initial
short calibration records remain in both series. Every cell qualified with
two consecutive regions at least 200 ms at the same job count.

**Series 2 subsequently has one short warmup and eight short measured regions**,
all E2E Full16 at N = 1. Its qualifying durations were 279.320339 and
283.242527 ms; its measured durations were 173.934008–195.343627 ms.
Nothing was deleted or selectively rerun. All N >= 16 warmup and measured
regions meet the 200 ms floor. A completed run does not mean every region
met the floor. CPU endpoints are all CPU4, not proof of uninterrupted residence;
power snapshots, thermal state and external load do not establish constant clocks.

The [validation artifact](data/p3_validation_20260905.json) records zero
mismatches in GCC native, independently compiled Clang arithmetic, and GCC
NO_ASM ASan+UBSan configurations. Each repeats the **same** finite corpus:
147 fixtures, 3 large fixtures, 43 negative controls and 3,703,557 assertions.
Counters overlap and must not be added as distinct cases. Large fixtures check
complete outputs and input preservation but do not replay every checkpoint;
smaller checkpoint replay uses exposed helpers, not instrumented timed kernels.

## Exactly 25 observations and remaining gates

| Lens | P3 observation | Remaining gate / limit |
|---|---|---|
| V01 Reachability | **STATIC:** all seven routes exist in the frozen binary. FE64 calls its returning add API per RHS; larger FE52 E2E loops include input packing. | These call boundaries differ. A direct inline/deferred 4x64 control remains **PENDING IMPORT** before isolating representation benefit from API cost. |
| V02 Corpus | **MEASURED:** both timing series share 1,048,576 canonical RHS values, one x0, real prefixes and identical full-input checksums. Correctness covers chunk edges, identities and large vectors. | One deterministic timing distribution and finite validation do not characterize every workload. |
| V03 Equivalence | **MEASURED:** independent Boost full-integer sums and corrected FE64 agree with candidate raw canonical limbs and all 32 bytes in three configurations. | Finite agreement is not a universal proof. The separate integer-bound argument supplies the admitted-domain reasoning. |
| V04 Definedness | **MEASURED / DERIVED:** N = 0 accepts null RHS and returns x0; positive N requires a valid readable canonical range. | No positive-null rejection promise or deliberately invalid-memory test. Contracts outside canonical field sums remain unexamined. |
| V05 Range / overflow | **DERIVED:** direct full normalization admits at most 4095 local canonical terms; weak-then-full admits 4096. The seed consumes one, leaving 4094/4095 RHS. Invalid template schedules fail compilation. | These are canonical-sum bounds, not arbitrary magnitude, subtraction, product or reducer bounds. |
| V06 Aliasing | **MEASURED:** supported x0/RHS aliases and input preservation pass; timed-corpus checks regenerate and compare all raw FE64/FE52 inputs after validation and regions. | No cross-type aliasing or caller-output-buffer overlap contract is introduced. |
| V07 Observable output | **MEASURED / STATIC:** each timed job materializes all 32 bytes; the last identical job per region is checked outside timing. Raw canonical outputs are checked before serialization in correctness tests. | Not every repeated timed job or every large-fixture checkpoint is individually checked; only the final sum is externally observable. |
| V08 Invariants / carries | **DERIVED:** component sums preserve an ordinary integer inside admitted chunks; folding uses `2^256 = K mod p`, and canonical checkpoints preserve the global prefix with x0 once. | Lost word overflow cannot be recovered by later normalization. Keep the unsafe P2 diagnostic separate from valid P3 schedules. |
| V09 Constant time | **DERIVED:** public N and fixed chunk sizes avoid secret-value dispatch in the experimental schedule. | No new CT certification follows from timings or selected branchless-looking code. Security qualification: **PENDING IMPORT**. |
| V10 Resources | **DERIVED / STATIC:** largest active RHS payload is 32 MiB for FE64 or 40 MiB resident FE52; stack operands occur in some setup, boundary and chunk paths. | Allocation is excluded. Total live memory, peak stack and dynamic scratch cost remain **PENDING IMPORT**. |
| V11 Property visibility | **MEASURED:** final-only observation permits deferred canonicalization, and the admitted candidate sums win all 480 pairs at N >= 16. | This establishes the combined schedule/representation/codegen result, not attribution to one mechanism or a faster single add. |
| V12 Dependency depth | **STATIC:** FE64 has scalar per-RHS carry work; Full1 repeats scalar packing/normalization; larger E2E loops accumulate packed components between boundaries. | Critical-path cycles and exact dependency depth remain **PENDING IMPORT**; source boundary count is not measured machine latency. |
| V13 Live state | **STATIC:** Full16's main chunk uses explicit YMM stack stores/reads. Selected larger E2E and resident inner loops have no explicit rsp/rbp operands. | This is range-local evidence, not a universal no-spill claim. Peak live registers and spill cost remain **PENDING IMPORT**. |
| V14 Operations | **DERIVED:** for N > 0 and chunk limit H, C = 1 + floor((N-1)/H); FE52 executes N RHS additions and C full normalizations, plus C weak calls for Weak4095. | Final `to_fe()` supplies the last full normalization once. Dynamic instruction counts and normalization cost attribution remain **PENDING IMPORT**. |
| V15 Conversion | **MEASURED:** E2E Full256 reaches 11.92x/11.66x at N = 256 with packing timed; resident reaches 29.47x/30.13x with RHS preconversion excluded. | The contracts are different. Setup/allocation-inclusive break-even remains **PENDING IMPORT**. |
| V16 Latency | **MEASURED:** complete-sum region means show the selected gains above and median losses at N = 1; all samples, including the short N = 1 deviation, are retained. | Not isolated add latency or individual-sum tail latency. Broader workload/frequency reproducibility remains open. |
| V17 Throughput | **MEASURED / STATIC:** ns/RHS improves in the one-core final-sum workload, with packed addition loops in the emitted candidate code. | No multicore optimum or upper-layer throughput is measured. Threads were deliberately excluded from this dependency experiment. |
| V18 Instructions / PMU | **STATIC:** 12 retained ranges expose packed loop strides, actual calls and full-output stores; manager replay passed. The repeated-job loop remains between clock calls. | PMU was unavailable under the recorded policy. Retired instructions, cycles, uops and energy remain **PENDING IMPORT**, not zero. |
| V19 Cache / bandwidth | **MEASURED, exploratory:** at N = 1048576, E2E Weak4095 beats resident in 15/16 pairs despite doing input packing. **STATIC:** E2E and resident use different packed loop shapes. | Footprint and timings do not identify the cause. Cache misses, memory traffic and bandwidth remain **PENDING IMPORT**. |
| V20 Code size | **STATIC:** job symbols range from 115 to 3056 bytes. Weak4095's 115-byte wrapper calls a separate 2913-byte sum body; their subtotal is 3028 bytes. | Individual symbols/subtotals exclude other callees, cold clones and constants. Full hot-path size and instruction-cache effects remain **PENDING IMPORT**. |
| V21 Portability | **MEASURED:** GCC, Clang arithmetic and sanitizer builds repeat the same correctness corpus successfully; Clang driver smoke also passes. | Full performance series are GCC only. Cross-compiler, other-CPU/architecture and default-LTO performance remain **PENDING IMPORT**. |
| V22 Compiler | **STATIC:** Full16 has an unrolled/vectorized chunk; larger E2E loops consume four FE64 objects per packed iteration; resident consumes two FE52 objects. Source/library/binary provenance is frozen. | Compiler policy can affect ranking. No claim that another compiler emits the same loop or achieves the same gain. |
| V23 Policy / ties | **MEASURED:** N = 1 has median losses. At N = 1048576, Full16 is best by median among timed E2E routes and beats Full4094 and Weak4095 in 15/16 pairs each, post hoc. | No universal winner, production dispatcher or threshold accepted. Untested sizes and setup regimes remain gates. |
| V24 Transfer | **DERIVED:** these gains concern a final-only field vector sum with an existing returning-API control. | Production integration and upper layers stay deferred. Scalar F2 evidence is retained separately; no proportional transfer is asserted. |
| V25 Prior art / novelty | **DERIVED:** existing FE52 encoding and sparse-prime folding are reused; this wave combines a bounded final-observation contract with local measurements. | No new-mathematics, world-record or external novelty claim. Novelty qualification remains **PENDING IMPORT**. |

Static evidence: [codegen report](P3_FIELD_SUM_CODEGEN.md) and
[retained ranges, symbols and commands](data/p3_codegen_20260905.json).
Static observations do not become hardware measurements merely because their
associated route was timed. The codegen artifact retains its original inspection
status; this results document does not rewrite that evidence.

## What remains before field research can be summarized as complete

Next gates include a direct 4x64 sum control, matched compiler/LTO comparisons,
setup-inclusive resident break-even, and hardware-counter evidence if available.
Field multiplication, symmetric squaring/reduction, subtraction/negation bounds,
and inverse/batch-inverse contracts remain separate research fronts. P3 does not
close them or authorize production integration.

Only this new lens-results document was written by this task. No benchmark,
production change, scalar change or rewrite of preregistered/raw evidence was
performed. Task MCP remains owner-suspended; the manager retains independent
review and the combined report/manifest handoff.
