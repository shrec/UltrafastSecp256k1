# P6 field boundaries: 25 preregistered observation lenses

Status: premeasurement hypothesis/verification plan, submitted for manager review.
No P6 timing, compiler result or qualification pass is asserted here. These are
25 observation lenses, not 25 discoveries. Arithmetic and integration scope stay
fixed by the [contract](P6_FIELD_BOUNDARY_CONTRACT.md) and
[protocol](P6_FIELD_BOUNDARY_PROTOCOL.md).

The matrix has 16 cells: actual FE64 API, direct existing ASM, unchanged P5 row
inline and the same row composition behind a separate-TU C pointer ABI, each for
multiply/square in chain1/ILP4. Each active lane performs 1024 canonical steps;
every job materializes128 bytes, including96 zero inactive bytes for chain1.
Only experimental boundaries change; no new product/reduction algorithm enters.

**DERIVED** denotes a contract argument, **HYPOTHESIS** an unmeasured possibility,
and **MEASURED/STATIC (P5)** explicitly named historical motivation. New P6
qualification, emitted-code and performance cells are **PENDING IMPORT** until
their evidence exists. Historical and smoke/synthetic-clock timings are not P6
samples. Lens identities are unchanged from the
[P5 measured map](P5_FIELD_PRODUCT_LENS_RESULTS.md), SHA256
`8fc5555cad93581c9a82aafb5b4f8a077084e3779cdfd1e7b9a51211e63e33a0`.

| Lens | P6 hypothesis or derived observation | Required evidence / boundary |
|---|---|---|
| V01 Reachability | **DERIVED:** four route families must reach actual API, existing ASM, inline row and genuinely outlined row leaves in all 16 cells. | Bind the timed bodies and their transitive callees to frozen sources/objects/binary. No new ASM or production dispatcher. **STATIC PENDING IMPORT**. |
| V02 Corpus | **DERIVED:** P6 admits canonical operands and reuses four seeds/256 RHS plus the unused256 raw512 entries for exact P5 corpus identity, FNV64 `7c2e57392893ac80`. | Check recipe and every input word before/after; do not count unused raw entries as an executed reducer domain. Count unique qualification fixtures separately from assertions/configurations. |
| V03 Equivalence | **DERIVED:** the unchanged P5 row-product/serial-reducer proof applies to canonical multiply and self-product square; ABI boundaries must preserve that value. | Independent Boost checks raw canonical words and all bytes across every admitted route. Existing FE64 is an additional reference, not the only oracle. New outcomes **PENDING IMPORT**. |
| V04 Definedness | **DERIVED:** C leaves require valid four-word inputs, canonical values and disjoint writable output. Native direct ASM requires x86-64 Linux GCC/Clang ASM-build BMI2+ADX admission outside all jobs/traces. | No null/output-overlap promise, per-step guard, unsupported execution or silent direct-ASM fallback. Portable NO_ASM explicitly skips ASM arithmetic and tests only its compile-disabled wrapper failure. |
| V05 Range / overflow | **DERIVED:** row unsigned128-cell and serial arbitrary512-reduction bounds are inherited unchanged from the exact P5 contract. | Retain full-width intermediate arithmetic and canonical raw output. This boundary wave adds no lazy-sum headroom, reduced-width product or new reducer proof. |
| V06 Aliasing | **DERIVED:** a/b may alias read-only, especially a==b; leaf output is disjoint and returning wrappers use a private result. | Test raw input preservation, guarded four-word output bounds and same-input square/multiply identity. No restrict, cross-type or in-place output widening; const declarations alone do not prove external ASM preservation. |
| V07 Observable output | **DERIVED:** each step is canonical; job output is128 bytes with all active BE32 lanes and zero inactive bytes. | Check raw words before normalizing serializers, bounded every-step traces, complete compiled jobs and lane0 agreement. No per-step benchmark barrier or claim that every timed step is instrumented. |
| V08 Invariants / carries | **DERIVED:** changing a call boundary must not change the row/reducer invariant or lose any result word before canonicalization. | Carry-boundary and exact earlier regression fixtures must cross the new wrapper/ABI paths. Do not recast ABI equivalence as a new arithmetic discovery. |
| V09 Constant time | **DERIVED:** capability admission is public, but fixed schedules and canonical mathematical outputs do not certify compiled secret-independent execution. | No CT certificate, VT/CT substitution or secret-value-dependent dispatch from these timings. Separate security qualification remains **PENDING IMPORT**. |
| V10 Resources | **DERIVED:** canonical inputs/results are32 bytes; pointer leaves expose explicit four-word input/output ranges. | Inspect actual caller/callee registers, stack, return handling and temporaries. Equal ABI or logical state size is not equal physical occupancy or measured resource cost. **STATIC PENDING IMPORT**. |
| V11 Property visibility | **MEASURED (P5):** row chain won31/32 combined multiply/square pairs while actual API won96/96 ILP4 candidate pairs. | This motivates operation/mode-specific controls, not a P6 prediction. New 16-cell results must retain all losses and their exact workload boundaries. |
| V12 Dependency depth | **HYPOTHESIS:** direct ASM may remove intermediate boundaries while outlining row may change compiler-visible scheduling across a call. | Inspect actual data/call dependencies in complete jobs. Do not turn a wall-clock difference into a critical-path or fixed call-cycle measurement. |
| V13 Live state | **HYPOTHESIS:** equal pointer ABI can still alter save/restore traffic and state live across calls; inline expansion may trade caller freedom for register pressure. | Record emitted caller/callee boundaries and stack syntax. Timing plus stack operands does not isolate spills, cache cost or one cause. **STATIC PENDING IMPORT**. |
| V14 Operations | **STATIC (P5):** GCC already emitted10 row-square product multiply sites despite16 source self-products. P6 changes no mathematical arithmetic count. | Reinspect each new inline/outlined body and distinguish static sites, looped work and dynamic instructions. No promised six-product saving or new operation-count speedup. |
| V15 Conversion | **DERIVED:** all families use native4x64; there is no FE52 packing or excluded resident preprocessing. | Include each route's actual initialization, wrapper/return handling and final materialization. Equal128-byte observation does not require or imply identical machine instructions. |
| V16 Latency | **HYPOTHESIS:** a direct-ASM or row boundary may change complete dependent-chain normalized time. | Measure1024-step chain jobs and eight same-round ratios per comparison/series. Region-mean ns/operation is not an individual-call percentile or instruction latency. |
| V17 Throughput | **HYPOTHESIS:** four independent states may change the winner because boundaries constrain optimizer visibility differently. | Count actual_jobs*1024*lanes; compare chain/ILP normalized cost without claiming four threads, packed SIMD, pure port throughput or the multicore optimum. |
| V18 Instructions / PMU | **MEASURED prior diagnostic:** PMU access remains unavailable under recorded policy; no settings change is authorized. | Inspect calls, arithmetic, output stores and support dispatch in the actual binary. Instruction presence is not retired counts, cycles, bandwidth, energy or CT certification. |
| V19 Cache / bandwidth | **DERIVED:** multiply cycles an8 KiB RHS payload; square reads no RHS during recurrence. The retained16 KiB raw512 corpus is unused arithmetic input in P6. | Distinguish identity/setup data from timed streams and transitive code. No residency, physical traffic or bandwidth cause is inferred from source sizes or ratios. |
| V20 Code size | **HYPOTHESIS:** outlining can shrink a job while moving arithmetic into a shared leaf; direct ASM retains external arithmetic. | Publish job and reachable leaf symbol extents/calls separately. A small wrapper is not a small transitive kernel and does not prove instruction-cache benefit. **STATIC PENDING IMPORT**. |
| V21 Portability | **DERIVED plan:** GCC and independent Clang qualify four routes; portable NO_ASM ASan+UBSan qualifies three and reports direct ASM skipped. | Do not call a skip a pass, imply ASM sanitizer instrumentation, pool repeated corpora or infer other-platform performance. Full timing is two frozen native GCC series only. |
| V22 Compiler | **HYPOTHESIS:** separate-TU/no-full-LTO leaves can change constant propagation, row-square reuse, allocation and prologues while preserving the same arithmetic source. | Freeze compiler/flags, P5 dependencies, corrected ASM/library, separate object and binary hashes. Verify calls survive and no invented per-step barriers equalize the routes. |
| V23 Policy / ties | **DERIVED plan:** reuse the continuously extended P5 duration contract with two same-count calibration successes, two warmups, eight rounds and reversed second-series order. | Retain all probes/short calibration samples, actual work, cap failures and losses/ties. Later regions need200 ms; synthetic/smoke data are validation-only, not pooled performance. |
| V24 Transfer | **DERIVED:** primary API/candidate, secondary ASM/row and inline/outlined, and mode comparisons answer different questions about complete implementations. | No causal pure-call attribution or unmeasured dispatcher threshold. Keep conditional winners; production/scalar/point/signature integration and remaining field work stay deferred. |
| V25 Prior art / novelty | **DERIVED from scope:** P6 reuses existing ASM and the established P5 row/reduction implementation without new arithmetic. | No novel mathematics/algorithm, world record, whole-engine gain or security claim. External novelty classification is outside this boundary-control wave. |

## Fixed decision criteria

Arithmetic, input-preservation and raw-canonicality failures block the affected
route before timing. Native capability failure blocks native direct-ASM
measurement; it is not repaired by relabeling a portable fallback. Portable
qualification reports the direct route skip explicitly. The manager reviews
mechanical qualification first, then the actual boundary code and written rules.

Before timing interpretation, emitted-code evidence must bind the intended API
and direct ASM call chains, row inline/outlined distinction, separate translation
unit/no-full-LTO build, complete whole-job recurrence and128-byte output. The
same source arithmetic does not require identical inline/outlined instructions;
that compiler difference is part of the experiment, not an invalid comparison.
Whole-job anti-hoisting must survive without per-step benchmark barriers.

Per series, preregister12 primary API/three-candidate comparisons and12 secondary
comparisons: ASM/row-inline, ASM/row-outlined and row-inline/row-outlined for each
operation/mode. Add eight chain/ILP normalized mode comparisons. A/B above one
favors B. Retain all eight paired samples, wins/ties/losses and distributions;
never substitute a ratio of independently computed medians. These comparisons
share observations and must not be presented as disjoint independent trials.

Accept measured observations at their exact operation/mode/compiler boundary,
including a split winner or no gain. P5 chain results are motivation only and
are not pooled; no candidate is accepted on fewer calls or source operations
alone. API/direct ASM also changes liveness/materialization; matching the pointer
ABI does not equalize algorithms, prologues or temporary placement. Pure call
cost, memory causality, security and universal transfer remain unestablished.

## Provenance and handoff

Historical P5 findings come from its [results](P5_FIELD_PRODUCT_RESULTS.md),
[measured lens map](P5_FIELD_PRODUCT_LENS_RESULTS.md) and
[emitted-code report](P5_FIELD_PRODUCT_CODEGEN.md), not new P6 measurements.
The inherited P5 contract SHA256 is
`1886cc57ff5ad813546c649ebf25af57c0fe02270a673fc1acbca8689bc4607a`.
The original P5 preregistered lens plan remains unchanged, SHA256
`2551d9bb2bed2612ba234828dae27c739dff9f1f37930ef31c0472b9d314f983`.

Verified repo_id `repo_666797171f0141c58bf05f579b2ee16e`; manager session
`01a06be6-2904-7c62-9e7d-1245c34a5312`. Worker-role Source Graph/session/memory/KB
tools remain unexposed under the recorded NeedFix; exact manager-supplied reads
were used without manager impersonation or context-database writes. Owner Task
MCP suspension continues. Only this plan and its companion contract are written;
no code, test, build, benchmark, preregistered P5 document or policy is changed.
Aligned P6 protocol SHA256:
`68ee687cf8d4b8b4dccacea9d82eb4a4c35d7b315f814c723bb8d5d1dff16113`.
Stop at manager review; all new P6 outcome cells remain **PENDING IMPORT**.
