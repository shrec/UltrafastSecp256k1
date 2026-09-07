# P6 field boundaries: measured results through 25 lenses

Status: measured field-only synthesis submitted for manager review. The
[fixed protocol](P6_FIELD_BOUNDARY_PROTOCOL.md),
[contract](P6_FIELD_BOUNDARY_CONTRACT.md) and
[preregistered lens plan](P6_FIELD_BOUNDARY_LENSES.md) remain unchanged.
These are 25 observations and limitations, not 25 discoveries or production
acceptances. P6 changes boundaries around existing arithmetic, not the arithmetic.

Evidence: [main report](P6_FIELD_BOUNDARY_RESULTS.md),
[series 1](data/p6_run1_20260905.json),
[reversed series 2](data/p6_run2_20260905.json),
[arithmetic qualification](data/p6_validation_20260905.json),
[duration qualification](data/p6_duration_validation_20260905.json),
[pre-timing source/build freeze](data/p6_performance_freeze_20260905.json),
[static codegen](P6_FIELD_BOUNDARY_CODEGEN.md),
[raw codegen](data/p6_codegen_20260905.json),
[manager codegen replay](data/p6_codegen_replay_20260905.json),
[numeric audit](data/p6_numeric_audit_20260905.json), and
[manager numeric replay](data/p6_numeric_replay_20260905.json).

Timing observations below were independently recomputed from measured raw
elapsed_ns / operations, with operations = actual_jobs * 1024 * active_lanes.
Each ratio is the median of eight same-round A/B ratios, not a ratio of medians;
greater than one favors B. S1/S2 denotes the two GCC series. No historical,
Clang-smoke or synthetic-clock samples enter these ratios. The 64 pair summaries
across both series reuse records; they are not 512 independent experiments.
The independent numeric audit passed 42,411 checks with zero errors; the manager's
full-object replay matched the final audit bound by SHA256
9fd6a75a665abd175e0482a5cce6a64bef5941109ee46387e52e4228a6625d97.

S1 retained 214 regions: 54 calibration, 32 warmup and 128 measured. S2 retained
208: 48 calibration, 32 warmup and 128 measured. All 320 later-phase regions
met 200 ms; minima were 200.387601 ms / 200.716604 ms. Nine continuous-region
extensions remain: S1 one warmup/five measured, S2 one warmup/two measured.
All 35 short calibration records remain. Full outputs, actual work, endpoint
arithmetic and final cumulative probes were consistent; input checks and
lane0 agreement passed. All CPU endpoints were CPU4, not proof of constant
frequency or no migration between snapshots.

The [retained environment note](data/p6_measurement_environment_note_20260905.json)
records that a final lightweight metadata check has no exact finish timestamp:
brief overlap with full timing cannot be excluded. Owned builds, tests and
substantive disassembly had finished; background services, frequency, thermals
and SMT activity remain uncontrolled. No sample was removed or selectively
rerun. This reinforces conservative interpretation of approximately 1% gaps.

**MEASURED** denotes finite tests or this machine's timings; **DERIVED** denotes
contract reasoning; **STATIC** denotes emitted-code evidence, independently
replayed by the manager. Unknown causal/security/platform properties are
identified as unmeasured, not silently inferred.

| Lens | New observation | Limit / next gate |
|---|---|---|
| V01 Reachability | **STATIC / MEASURED:** all 16 cells reached the intended API, existing ASM, inline row or separate-TU row path. Eight direct-ASM/outlined-row caller pairs across GCC/Clang match after address normalization and replacing only the arithmetic-leaf target. | This strengthens the boundary control, not equality of arithmetic leaves or raw code bytes. The source/build freeze and manager replay passed; no production route was wired. |
| V02 Corpus | **MEASURED:** the exact P5 corpus identity was preserved: four nonzero seeds, 256 canonical nonzero RHS values and 256 unused wide control values, FNV64 7c2e57392893ac80. Qualification used 13,102 pair entries / 13,074 distinct canonical pairs, plus derived chains. | Unused wide values are not a P6 reducer workload. Pair entries and repeated compiler checks are not distinct independent fixtures. |
| V03 Equivalence | **MEASURED / DERIVED:** native GCC and independently compiled Clang field sources each passed 406,239 assertions with zero mismatches. NO_ASM ASan+UBSan passed 358,744; shared checksum a9076830a8ab714b. Boost checked canonical raw words and all bytes, with FE64 as an additional reference. | Portable qualification explicitly skipped 47,498 direct-ASM outputs and rejected two compile-disabled wrappers. It is not a fourth-route pass or sanitizer instrumentation of ASM; all evidence remains finite. |
| V04 Definedness | **DERIVED / MEASURED:** native capability admission precedes every direct job/trace; admitted wrappers add no per-operation feature check. Canonical four-word inputs and disjoint four-word outputs remain preconditions. Compile-disabled direct wrappers fail closed. | Unsupported-native execution was not emulated. No null, output/input overlap, silent fallback or weakened production admission contract is introduced. |
| V05 Range / overflow | **DERIVED / MEASURED:** the frozen P5 row product and serial reducer are unchanged. Existing unsigned128 cell bounds, full-width q*K and final canonical correction remain applicable; boundary qualification passed. | This wave establishes no new reducer, width reduction or lazy-sum headroom. ABI changes do not substitute for an arithmetic proof. |
| V06 Aliasing | **MEASURED:** qualification retained same-object inputs and 81,894 preservation checks per configuration. The new row C ABI passed 47,498 guarded-output checks with 32-byte canaries before/after its 32-byte result. | Canaries instrument the row C leaves, not ASM. Read-only a/b alias is admitted; output must remain disjoint. No restrict, cross-type or enlarged in-place contract. |
| V07 Observable output | **MEASURED / STATIC:** each series made 40,960 canonical-state/byte trace checks, with no zero/one state observed. Complete compiled jobs and lane0 cross-mode outputs matched. All jobs store 128 bytes, including 96 zero inactive bytes in chain1. | Helper traces are untimed and separate from complete loops; each region checks its last identical job outside the clock. Every primitive step remains canonical, not just the final serialized value. |
| V08 Invariants / carries | **MEASURED / DERIVED:** three named carry KATs and eight 1,024-step qualification chains crossed the boundary routes, alongside carry-neighbor and ordinary-integer checks. Input/result handling preserved the inherited residue invariant. | The 8,192 derived qualification steps and performance traces are different checks. No new arithmetic identity follows from carrying existing regression witnesses through another ABI. |
| V09 Constant time | **STATIC NEGATIVE:** Clang's shared reducer has a result-dependent final-correction branch: its condition derives from subtraction borrow. A source-level mask did not guarantee branch-free emitted selection. | No CT certification for this generic row path, no inference that GCC is certified, and no VT/CT substitution. Secret-bearing use requires separate security qualification; timing gains do not resolve this finding. |
| V10 Resources | **STATIC / DERIVED:** direct and outlined callers match except their leaf target; their callees differ. C++ arithmetic/caller state includes stack-relative operands; ASM saves/restores registers but has no explicit RSP/RBP-relative arithmetic operands. | Equal pointer ABI and 32-byte values do not equalize leaf prologues, live state or physical occupancy. Stack syntax is not measured spill cost. |
| V11 Property visibility | **MEASURED:** direct ASM beat API in all 64/64 pairs. API/direct ratios were 1.11728 / 1.12283 for multiply chain, 1.23919 / 1.23644 for multiply ILP4, 1.06453 / 1.07018 for square chain and 1.29902 / 1.28879 for square ILP4. | This is an observed boundary opportunity around unchanged ASM, not a new arithmetic algorithm, pure call-cycle saving or authorized production replacement. |
| V12 Dependency depth | **STATIC / MEASURED:** the direct route reaches the same ASM leaf as the API without the intermediate public dispatcher. Direct ASM also beat the matched-caller outlined row in all 64/64 pairs. | API/direct still changes caller types, return handling and liveness; direct/outlined changes leaf instructions and prologues. No wall-clock subtraction identifies a critical path or universal call cost. |
| V13 Live state | **MEASURED / STATIC:** outlining did not have one universal effect. Multiply ILP4 outlined row beat inline row 16/16, with inline/outlined ratios 1.08968 / 1.09173, although both lost to API/direct ASM. Square ILP4 outlined won only 3/16, ratios 0.97023 / 0.97907. | Optimizer visibility and caller/callee state change together. These opposite outcomes do not isolate register pressure, spills or one beneficial instruction. |
| V14 Operations | **STATIC:** both compilers reuse symmetric row-square products, leaving ten product multiply sites rather than sixteen source self-products. GCC row leaves include six reducer multiply sites; Clang may call a separately compiled reducer. | Mathematical operation counts are unchanged. Static sites, loop execution counts and retired instructions remain different quantities; no new six-product saving is claimed. |
| V15 Conversion | **DERIVED / STATIC:** every route stays native 4×64. There is no FE52 packing or excluded resident preprocessing; active seed setup and full output materialization are inside each job. | Equal logical 128-byte output need not use identical generated instructions. Inline/outlined comparison changes optimization scope, not only a call instruction. |
| V16 Latency | **MEASURED:** inline/API chain wins were 15/16 for multiply and 16/16 for square. API/inline ratios were 1.12909 / 1.13414 and 1.13022 / 1.16447. Against direct ASM, inline square retained 16/16 wins with direct/inline ratios 1.07447 / 1.08815. | These are 1,024-step region-average ns/operation comparisons, not individual-call latency percentiles. The positive square-chain result remains conditional on this GCC build and boundary. |
| V17 Throughput | **MEASURED:** both row ILP4 candidates lost every API comparison, 64/64 combined, and every direct-ASM comparison. Direct chain/ILP normalized ratios were 2.18187 / 2.15222 for multiply and 2.35444 / 2.39083 for square, each favoring ILP4 in 16/16 pairs. | Four independent scalar states are not four threads, packed field multiplication or measured pure port throughput. Chain gains do not transfer automatically to ILP4. |
| V18 Instructions / PMU | **STATIC:** manager replay passed 10,521 checks over 66 slices, 32 GCC/Clang jobs, eight matched pairs and 64 windows. Repeated job calls, real outlined calls and full output stores survive. | Dynamic PMU events remain unavailable under the recorded policy. Static replay does not supply cycles, bandwidth, cache misses, energy or execution-port costs. |
| V19 Cache / bandwidth | **DERIVED / MEASURED:** multiply cycles an 8 KiB RHS payload; square has no RHS recurrence reads. The retained 16 KiB wide array is untimed corpus-identity data, never a P6 arithmetic stream. | Layout sizes and timings do not establish residency, physical traffic or bandwidth causality. No comparison silently includes a raw-reducer workload. |
| V20 Code size | **STATIC:** GCC direct/outlined multiply chain callers are each 349 bytes; their ASM/row leaves are 610 / 1,244 bytes. Square callers are each 326 bytes with 520 / 1,300-byte leaves. Inline chain jobs are 1,681 / 1,363 bytes for multiply/square. | Individual symbols are not transitive hot-path or instruction-cache footprints. Outlining moves arithmetic; small wrappers cannot be treated as whole kernels. |
| V21 Portability | **MEASURED / STATIC:** native GCC/Clang and explicitly scoped portable sanitizer qualification passed. Separate synthetic duration diagnostics passed 96 cases / 5,282 assertions per GCC/Clang configuration. | Full performance is GCC-only. Clang driver/row codegen links the frozen GCC library; independent Clang arithmetic qualification is a different build. No ASM sanitizer, other-platform timing or CT inference. |
| V22 Compiler | **STATIC:** GCC embeds inline-route evaluators in jobs; Clang outlines all four row-inline evaluators. Clang chain evaluators call the shared reducer, while ILP4 unrolls lanes and inlines reductions with some packed ADD/SUB, not packed field multiply. | Source inline is not guaranteed machine inlining. Actual reachable callees, the result-dependent correction branch and compiler-specific allocation must remain in interpretation; whole-Clang timing is unmeasured. |
| V23 Policy / ties | **MEASURED:** multiply-chain direct/inline ratios were only 1.01058 / 1.01099, with inline wins 12/16: a near comparison, not an exact tie or robust new-method gain. All nine extensions and 35 short calibrations remain. | The environment note precludes treating approximately 1% as isolated causal evidence. Synthetic failure retained 33 regions/13 probes; within-region cap/tail exhaustion remains static-reviewed only. No selective rerun. |
| V24 Transfer | **MEASURED / DERIVED:** outlined row lost every API pair, 0/64 wins, and all matched direct-ASM pairs. Inline chain candidates and direct-ASM boundary gains remain useful conditional findings; the fully opaque row boundary does not preserve the chain win. | No new arithmetic, production/scalar/point/signature integration, universal dispatch rule or whole-engine gain. Field research remains unfinished; boundary evidence must guide, not be bypassed by, future arithmetic experiments. |
| V25 Prior art / novelty | **DERIVED:** P6 reuses existing ASM and the unchanged P5 row/reduction identities, with explicit call-boundary controls and measured negative results. | No novel mathematics/algorithm, world record or security claim. Novelty is not established by a faster invocation of existing arithmetic. |

## Disposition

Retain direct ASM as the strong native boundary control, inline square as a
conditional chain candidate, and the outlined-row losses as part of the result.
Treat multiply-chain inline/direct as unresolved at approximately 1%, not a new
winner to integrate. The Clang correction branch is a concrete security limit,
not an unavailable measurement. No code, timing or policy was changed here.

Only this measured map was written. The owner’s Task MCP suspension and recorded
worker Source Graph exposure limitation remain; exact manager-supplied targets
were used without manager-role impersonation. Stop at manager review.
