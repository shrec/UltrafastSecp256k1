# P4 field sum: results through 25 observation lenses

Status: evidence synthesis submitted for manager review. Field-only collection
continues; no production integration or completion of field research is claimed.
These are 25 observations and unresolved gates, **not 25 discoveries**.
Lens identities are unchanged from the [contract](P4_FIELD_SUM_CONTRACT.md).

Evidence: [main report](P4_FIELD_SUM_RESULTS.md), [protocol](P4_FIELD_SUM_PROTOCOL.md),
[series 1](data/p4_run1_20260905.json), [reversed series 2](data/p4_run2_20260905.json),
[arithmetic qualification](data/p4_validation_20260905.json),
[synthetic duration qualification](data/p4_duration_validation_20260905.json),
[static emitted code](P4_FIELD_SUM_CODEGEN.md), and
[environment limitations](data/p4_environment_obstacles_20260905.json).

Numerical observations below were independently recomputed from raw measured
`elapsed_ns / jobs`, paired by size and round, not copied from stored summaries.
S1/S2 always denotes the two complete P4 series, not two compilers. Reported
speedups are medians of eight same-round ratios, not ratios of medians. For
an explicitly named A/B comparison, greater than one favors B. Timing is of
whole final-only sums; no P3/P2/P1 timing is pooled into these comparisons.

S1 retained 652 regions: 172 calibration, 96 warmup, 384 measured. S2 retained
649: 169 calibration, 96 warmup, 384 measured. All 960 later-phase regions
met the 200 ms floor; minima were 200.853844 ms and 206.998300 ms respectively.
S1 required 21 continuous-region extensions, one each in 4 warmup and 17
measured regions; S2 required none. All 147 short calibration records remain.
Every recorded output matched its cell's expected full bytes; recorded elapsed
times, actual RHS work and final cumulative probes were consistent. CPU
endpoints were all CPU4, which does not prove constant frequency or no migration
between endpoints. Input-preservation checks passed before and after both series.

**MEASURED** means finite arithmetic checks or this machine's timed observations;
**DERIVED** means reasoning under the stated domain; **STATIC** means emitted-code
inspection. Unperformed causal or portability measurements remain **PENDING IMPORT**.

| Lens | New evidence | Unresolved gate / limit |
|---|---|---|
| V01 Reachability | **MEASURED / STATIC:** all 48 cells exercised eight frozen routes: actual FE64 API, inline eager, native wide16/wide4094 with one/four banks, and unchanged FE52 E2E Full16/Full4094. Actual job/callee and complete-output paths are recorded. | Experiment-local entry points only; no production dispatcher or caller integration. |
| V02 Corpus | **MEASURED:** both performance series used seed 20260905 and complete FE64 checksum `a228d8a510693ab4`, with the same six actual RHS prefixes. Arithmetic qualification used 148 fixtures over nine corpora, including three large fixtures and 46 negative controls. | Repeated compiler runs reuse this finite corpus. Assertions and helper comparisons are not distinct random inputs; large fixtures do not replay every intermediate checkpoint. |
| V03 Equivalence | **MEASURED / DERIVED:** GCC, independently compiled Clang field sources, and NO_ASM ASan+UBSan each passed 21,130,305 assertions with zero mismatches against Boost ordinary integers plus the corrected FE64 reference. Raw canonical limbs and all 32 final bytes are checked. | Finite evidence plus the written bound argument, not a machine-checked universal proof. Passing the existing implementation alone would not establish the mathematical result. |
| V04 Definedness | **DERIVED / MEASURED:** canonical FE64 inputs, a valid positive-length range and genuine unsigned128 remain preconditions. Tests include 72 N=0/null identity calls per configuration. | No positive-null rejection API; no admission of noncanonical inputs, unsupported integer types or unrelated moduli. Sanitizer qualification is finite. |
| V05 Range / overflow | **DERIVED / MEASURED:** native chunk admission gives M≤4096, column/carry intermediates <2^76, h≤4095 and hK<2^45; the second fold's high is zero. Tests include chunks 1, 16, 4094 and 4095; chunk0, chunk4096 and lanes2 were rejected at compilation. | These are sufficient admitted bounds, not maximal unsigned128 capacity. FE52 direct-full admission remains 4094 RHS, not the native maximum4095. |
| V06 Aliasing | **MEASURED:** arithmetic qualification includes 136 alias cases and 2,701 input-preservation checks per configuration. Both full series regenerated and compared all 1,048,576 RHS objects and x0 after validation and after measurement. | Same-type seed/RHS alias is supported with both contributions counted; cross-type alias and caller-owned output-buffer overlap are not added contracts. |
| V07 Observable output | **DERIVED / MEASURED / STATIC:** only final canonical FE64 and 32 BE bytes are external observations. Tests check raw limbs before serialization; emitted jobs retain full output stores, and the last identical job of every region matched all expected bytes. | This is not a prefix-sum API or a per-job byte comparison inside the timed loop. Adding observable intermediate outputs changes the optimization freedom. |
| V08 Invariants / carries | **MEASURED / DERIVED:** actual narrow helpers were replayed separately across 907,162 chunks per configuration, checking banks, merge, propagated high, both folds and correction. Six rare first-fold overflow cases were exercised. Seed-at-merge occurs once per chunk; global x0 occurs once. | Helper replay is separate from timed kernels, not instrumentation of every full call. Three large fixtures have complete-call checks but no full phase replay. Carry bits must not be mistaken for full-word masks. |
| V09 Constant time | **DERIVED:** public chunk sizes and bank assignment do not themselves establish secret-independent machine execution. | CT qualification is **PENDING IMPORT**. No security substitution, secret-dependent dispatch or timing-based CT certification. |
| V10 Resources | **DERIVED / STATIC:** every route reads a 32N-byte FE64 RHS array, reaching 32 MiB. Native mathematical bank payload is 64 bytes for one bank or 256 for four, before other state. Codegen records both register and stack operands. | Payload is not peak live state, physical traffic or heap working set. Resource occupancy and dynamic cost remain unmeasured. |
| V11 Property visibility | **MEASURED:** all seven candidates beat the actual API in all 560 same-round pairs at N≥16. At N=1, inline eager won 16/16 pairs with 1.127× / 1.137× median speedups; the other six candidates lost every series median, despite 10 isolated wins among their 96 pairs. | Native deferred sums show that the large API-relative advantage is not exclusive to FE52 packing. This does not isolate one mathematical or machine cause, or imply isolated binary-add speedups of the same magnitude. |
| V12 Dependency depth | **MEASURED / STATIC:** four-bank variants lost every equal-chunk, per-series median comparison against one bank; they won only 8/192 individual pairs across both chunks and all six sizes. Native accumulation is scalar add/adc in this build. | More independent source accumulators did not yield a winning complete implementation here. No measured critical-path depth, cycles, pure ILP gain or general impossibility result. |
| V13 Live state | **STATIC / MEASURED:** native wide4094/one-bank has four low/high register pairs and no explicit stack operands in its displayed accumulation loop. Four-bank4094 uses register pairs plus stack-relative read-modify-write columns. Its equal-chunk median timings are worse. | Coexistence of stack accesses and slower results is not measured spill causality. Merge overhead, instruction choice and other live state also differ; dynamic attribution is **PENDING IMPORT**. |
| V14 Operations | **DERIVED / MEASURED:** native chunks reduce cross-column/canonical boundaries from N to C=ceil(N/H). At N=4096, H16 has 256 boundaries versus two for H4094; the larger one-bank chunk wins 16/16 pairs. At N=1,048,576, H16 beats H4094 in 16/16 pairs despite 65,536 versus 257 boundaries. | Fewer source-level reductions do not guarantee a faster complete sum. No universal optimum or intermediate-size dispatch threshold follows from six sampled sizes. |
| V15 Conversion | **MEASURED / STATIC:** all FE52 packing is inside the timed sum; there is no resident FE52 array or excluded conversion setup. At N=4096, native one-bank4094 / FE52 Full4094 ratios are 0.7431 / 0.7429, favoring native. At N=1,048,576 they reverse to 1.0647 / 1.0408; FE52 wins 14/16 pairs. | Same input layout and equal chunks still change radix, packing, accumulator scheduling and codegen together. Neither representation is universally superior, and packing cost is not isolated. |
| V16 Latency | **MEASURED:** at N=4096, native one-bank4094 median whole-job times are 2,469.87 / 2,449.76 ns, versus API 41,170.99 / 40,713.88 ns. Same-round median speedups are 16.782× / 16.728×. | These are duration-region averages per complete sum, not individual-call latency percentiles or a 16× field-add instruction improvement. |
| V17 Throughput | **MEASURED:** at N=1,048,576, native one-bank16 takes median 1.2935 / 1.3157 ns per RHS versus API 10.3398 / 10.1114, with 8.086× / 7.894× paired speedups. Native one-bank16 / FE52 Full16 ratios are 0.95263 / 0.95264, but native wins only 12/16 pairs. | Amortized ns/RHS includes whole-job output and boundary costs. No SIMD, multicore, isolated-operation or upper-layer throughput follows automatically; the smaller equal-chunk gap needs broader replication. |
| V18 Instructions / PMU | **STATIC:** retained code distinguishes per-RHS API calls, call-free inline eager arithmetic, scalar unsigned128 accumulation and packed FE52 addition. Prior cycles/instructions access was denied; P4 did not change policy or repeat the denied request. | Dynamic instructions, cycles, ports, cache events and energy are unavailable, not zero. Static excerpts do not supply those measurements. |
| V19 Cache / bandwidth | **MEASURED / DERIVED:** the one-bank chunk preference reverses between the middle sizes and the 32 MiB largest input. All routes share the same FE64 input layout; none receives a preconverted array. | Equal layouts do not establish equal traffic or residency. The reversal is not evidence of bandwidth, cache, prefetch or spill causality without further controlled measurements. |
| V20 Code size | **STATIC:** actual job symbols differ substantially: API 209 bytes, inline eager 547, native one-bank16 1,557 and one-bank4094 875. Four-bank jobs have 145-byte wrappers calling 3,193/2,462-byte implementations; FE52 wrappers call separate bodies too. | Individual symbol extents are not transitive hot-path or instruction-cache footprints. Small wrappers must not be reported as whole kernels; no cache-cost attribution. |
| V21 Portability | **MEASURED:** arithmetic passed the same corpus on GCC14, independently compiled Clang18 field sources, and NO_ASM ASan+UBSan. The separate duration test passed 48 synthetic cases / 3,163 assertions on exactly two configurations, GCC and Clang. | Full performance series are GCC-only. Genuine unsigned128 is required; other architectures, compilers, whole-Clang performance and production build variants remain **PENDING IMPORT**. Synthetic clocks are never performance evidence. |
| V22 Compiler | **STATIC:** GCC emits scalar native u128 column updates, unrolled short native chunks, and vectorized FE52 packing/accumulation with five YMM vpaddq streams in its long-chunk loops. Inline eager removes per-RHS library calls in this binary. | These are compiler-specific complete implementations, not identical-code call-removal or representation-only ablations. Freeze flags and binaries before further compiler comparisons. |
| V23 Policy / ties | **MEASURED:** every warmup/measured region passed its duration floor using actual jobs; all 21 S1 extensions and every short calibration attempt remain. At N=16, one-bank16 / one-bank4094 ratios are only 1.0101 / 1.0286, with larger-chunk wins 4/8 and 6/8: a mixed near comparison, not an exact tie. | Preserve losses and sample distributions; no selected reruns. Synthetic extension-limit failure retains 97 regions and non-success status. Within-region work-cap/capped-final-batch handling was statically reviewed, not dynamically exercised by that diagnostic. |
| V24 Transfer | **DERIVED:** P4 identifies useful native final-only field-sum candidates and negative four-bank controls. It does not replace any existing production routine. | Field research remains unfinished: product/symmetric-square reducers, inversion, additional arithmetic contracts and integration are separate gates. Scalar, point, signature and multicore gains are not established. |
| V25 Prior art / novelty | **DERIVED:** the existing pseudo-Mersenne identity is reused while observation frequency, native columns, bank count and compiler lowering are compared. | No novel-mathematics, novel-algorithm, world-record or security claim. External novelty classification remains **PENDING IMPORT**. |

## Resulting research direction

Keep more than one candidate. Inline eager is the favorable N=1 control; native
one-bank deferred sums are strong middle-size candidates; native one-bank16
and FE52 Full16 remain close competitors at the largest sampled size. Larger
chunks and additional banks are not unconditional improvements. Replicate
whole-program comparisons across compilers and sizes before selecting any
policy, and investigate resource causes without treating timings as PMU data.

This document changes no source, protocol, earlier evidence or production code.
Worker Source Graph tools remain unavailable; exact manager-supplied targets
were read under the recorded bounded handoff, without manager-role impersonation.
Owner Task MCP suspension remains active. Stop at manager review.
