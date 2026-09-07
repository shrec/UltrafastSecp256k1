# P5 field products: 25 preregistered observation lenses

Status: premeasurement hypothesis/verification plan, submitted for manager review.
There are no P5 speed measurements or candidate acceptances in this document.
These are25observation lenses, not25discoveries. All integration remains deferred.

The [fixed protocol](P5_FIELD_PRODUCT_PROTOCOL.md) and independently authored
[arithmetic contract](P5_FIELD_PRODUCT_CONTRACT.md) govern the20cells: multiply
and square each have an actual-API reference plus three candidates; raw reduction
has two candidates; every route is measured in chain1 and ILP4 on one core.
Each job performs1024steps per active lane and materializes128output bytes.

Labels: **DERIVED** means an argument under the exact contract; **HYPOTHESIS**
means an unmeasured scheduling possibility; **MEASURED (P2/P4)** imports only
named historical evidence; **STATIC PENDING IMPORT** requires new emitted-code
inspection. All new qualification or timing outcomes remain **PENDING IMPORT**.
Historical timings and smoke/synthetic-clock data are never P5 samples.

The25identities below match the unchanged
[P4 measured lens map](P4_FIELD_SUM_LENS_RESULTS.md) and
[P4 contract](P4_FIELD_SUM_CONTRACT.md). Their application changes because this
wave canonicalizes each multiplication/square/reduction, not only a vector sum.

| Lens | P5 hypothesis or derived observation | Required evidence / boundary |
|---|---|---|
| V01 Reachability | **DERIVED:** twenty cells route through actual FE64 multiply/square, row/Comba/symmetric products and serial/precomputed raw reducers. | Bind each compiled job, helper/callee and macro profile to frozen sources/binary; no production dispatcher or default-build integration. **STATIC PENDING IMPORT**. |
| V02 Corpus | **DERIVED:** full products admit arbitrary256-bit operands, reducers arbitrary512-bit values, complete field pipelines canonical operands. These are three distinct test domains. | Independent C++ corpora must include all-max, p/p^2 boundaries, high/low-only values, both exact P1 KATs, deterministic random values and bounded recurrences. Count unique fixtures separately from repeated checks/configurations. |
| V03 Equivalence | **DERIVED:** exact row/Comba products followed by a proven reducer equal ordinary integer multiplication modulo p; symmetric expansion equals self-product. | Boost checks all eight raw product words, four canonical residue words and every final byte, with corrected FE64 only an additional reference. New outcomes: **PENDING IMPORT**. |
| V04 Definedness | **DERIVED:** raw fixed-width domains differ from canonical field inputs; unsigned128 operations require actual compiler type support and defined shifts. | Same-object operands/read-only inputs are valid. Do not normalize away an arbitrary raw input, add unrelated pointer/null APIs or invoke inadmissible C++ memory behavior. Sanitizer qualification: **PENDING IMPORT**. |
| V05 Range / overflow | **DERIVED:** row cells fit128bits; Comba columns fit130bits in a192-bit state; doubled cross products may need129bits. First reducer fold cells fit97bits, q<=K and qK<2^65. | Preserve the full column/high parts, exercise q=K and qK>2^64, and prove third-fold high zero before the final correction. P3/P4 sum caps are not imported. |
| V06 Aliasing | **DERIVED:** independent private results permit same-object multiplication and square/self-product comparisons without input mutation. | Check exact raw input preservation and same-type aliases; no restrict, cross-type or caller-owned output-buffer overlap promise is added. |
| V07 Observable output | **DERIVED:** each primitive returns canonical raw limbs; jobs expose all active lane bytes and zero inactive bytes in a fixed128-byte output. | Check raw limbs before serialization, all128job bytes and chain1/ILP4 lane0 agreement. Final-only job bytes do not authorize noncanonical intermediate primitive results. |
| V08 Invariants / carries | **DERIVED:** row carries preserve each product cell; Comba carries can occupy66bits; reducer identities successively replace HB, qB and eB by HK, qK and eK. | Target the all-max512witness: q=K, a possible first-word carry2, overall second-fold overflow1 and final K^2-1. Do not confuse an internal word carry, overall high bit or full-word selection mask. |
| V09 Constant time | **DERIVED:** fixed schedules and public corpus indices do not certify secret-independent compiled execution. | Preserve distinct security contracts; no timing-based CT certification, secret-value-dependent dispatch or VT/CT substitution. CT qualification remains **PENDING IMPORT**. |
| V10 Resources | **DERIVED:** input/output widths are known: raw products64bytes, canonical field values32bytes, Comba accumulator24bytes before other state. | Inspect actual live registers, stack, temporary arrays and helper ABI; source payload is not peak physical occupancy or measured resource cost. **STATIC PENDING IMPORT**. |
| V11 Property visibility | **MEASURED (P2):** canonical resident FE52 chain multiply won13/16pairs and square16/16; FE64 won ILP4 mul/square32/32. | This is historical motivation for separate dependency modes, not P5 evidence. P5 changes native product/reducer schedules and must establish its own whole-primitive gains/losses. |
| V12 Dependency depth | **HYPOTHESIS:** diagonal product scheduling or four precomputed high*K products may expose work before carry reconciliation. | Inspect actual emitted dependency placement and compare complete jobs in both modes. Reordering independent source products does not remove mathematical carries or establish critical-path cycles. |
| V13 Live state | **HYPOTHESIS:** Comba's three-word state may trade output-array activity for carry pressure; precomputed reducer products or ILP4 may increase concurrent temporaries. | Record actual registers/stack accesses and outlined calls. Any timing coexistence is not measured spill causality. **STATIC PENDING IMPORT**. |
| V14 Operations | **DERIVED:** row and general Comba use16mathematical64x64products; symmetric square uses10distinct products but cross terms are added twice. Both first-fold reducer schedules use four high-word products. | Count actual emitted multiply/carry instructions separately. Constant-K lowering, doubling, masking and setup can change total work;10versus16does not promise1.6x speed. |
| V15 Conversion | **DERIVED:** P5 keeps native4x64 input/output representation; no FE52 packing or resident-preconversion route participates. | Include complete API/helper boundary, construction, canonicalization and final serialization costs. Equal representation does not make row/Comba a perfectly isolated causal ablation. |
| V16 Latency | **HYPOTHESIS:** a product or reduction schedule may improve dependent-chain normalized time. | Measure complete1024-step chain jobs with all output bytes; report region-mean ns/operation, not individual-call latency percentiles or guessed instruction latency. |
| V17 Throughput | **HYPOTHESIS:** four independent states may overlap work differently from chain1. | Count actual_jobs*1024*active_lanes; shared RHS index is step&255, square has no RHS. Compare amortized chain/ILP costs without claiming pure port throughput, hardware SIMD or multicore speed. |
| V18 Instructions / PMU | **MEASURED prior manager diagnostic:** dynamic cycles/instructions access was denied under perf_event_paranoid=4; no policy change is authorized. | Use new emitted-code evidence for instruction presence and calls, not dynamic retired counts, cycles, energy or port pressure. PMU unavailable means unavailable, not zero. |
| V19 Cache / bandwidth | **DERIVED:** the256-element canonical RHS payload is8KiB and arbitrary512payload16KiB; square reads neither RHS stream during its recurrence. | These are data-layout counts, not residency or traffic measurements. Do not infer bandwidth/cache causes or compare reducer/multiply timing as interchangeable work. |
| V20 Code size | **HYPOTHESIS:** row unrolling, Comba/symmetric expansion and precomputation may alter code size and outlining. | Retain actual job/helper symbol extents and call graph boundaries; small wrappers are not whole kernels, and size alone does not establish instruction-cache cost. **STATIC PENDING IMPORT**. |
| V21 Portability | **DERIVED plan:** qualify the same finite arithmetic corpus under native GCC, independently compiled Clang field sources and NO_ASM ASan+UBSan. | Do not count three runs as three distinct corpora or arithmetic success as Clang/new-platform performance. Genuine unsigned128 and separate security/platform qualification remain prerequisites. |
| V22 Compiler | **HYPOTHESIS:** BMI2/ADX selection, constant multiplication lowering, inlining and register allocation can reverse source-level expectations. | Freeze flags/macros/source/library/binary hashes before timing; inspect actual products, carries, output stores and noinline/noipa boundaries. No inferred instruction counts. |
| V23 Policy / ties | **MEASURED (P4):** continuous-region extensions retained21short prefixes while all later regions met their floor. | Independently qualify P5 duration bookkeeping: same-count calibration pair, retained cumulative probes/actual work, fail-closed caps, reverse ordering, all losses/ties. Smoke/synthetic data are not timing evidence. |
| V24 Transfer | **DERIVED:** a raw reducer gain, canonical multiply gain and square gain are separate claims; a faster isolated reducer need not improve a composed field operation. | Keep primary actual-API and secondary schedule comparisons distinct. Production/scalar/point/signature integration remains deferred; field research is not complete. |
| V25 Prior art / novelty | **DERIVED from local scope:** this wave reuses the pseudo-Mersenne identity and compares declared row/Comba/symmetric scheduling candidates. | No new-mathematics, novel-algorithm, world-record or whole-engine claim. External novelty classification is **PENDING IMPORT**, not implied by a positive experiment. |

## Decision criteria before any timing conclusion

Arithmetic acceptance requires exact ordinary-integer equality and canonical raw
outputs throughout the admitted domains, then the manager's complete code and
contract review. Tests must specifically retain the130-bit Comba-column witness,
129-bit doubled-cross case and65-bit qK/second-fold-overflow case; random testing
alone is unlikely to establish those boundary paths. A failed width or byte check
blocks the affected candidate before timing, with its evidence preserved.

For each mode and series, the primary multiply/square comparison is actual
API/candidate, with three candidates each. Raw reduction compares
serial/precomputed under its own XOR-fed arbitrary512recurrence. Secondary
row+serial/Comba+serial and Comba+serial/Comba+precomputed comparisons concern
complete generated implementations. An A/B ratio above one favors B. Report all
eight same-round ratios and wins/ties/losses, along with per-operation samples;
never substitute a ratio of independently computed medians for the paired
estimator. Chain1 and ILP4 are distinct workload modes, not threads.

Do not reject a candidate merely because it has more source operations, or
accept it merely because it has fewer multiplications. Retain measured losses
and split winners by operation/mode if supported. A favorable profile does not
authorize an unmeasured universal threshold, compiler-independent winner or
production dispatch. All P5 outcome cells remain **PENDING IMPORT** until the
independent arithmetic, duration, emitted-code and full raw-data gates complete.

## Provenance and handoff

Historical P2 observations are from [P2 results](P2_FIELD_RESULTS.md), with their
own canonical resident/API and setup contracts; none of those timing samples
enters P5. P4 provides lens identities and duration-method lessons only.
P4 measured lens-map SHA256:
`0f462a3be6502274b3016e492f1f60f5eb65a523dc2b1a4d6dbeb5260831c382`.
P4 contract SHA256:
`76a057fc47803132eb09828f3b4b0161b0cf0114f96dd7e29175e453feb54dba`.
Aligned P5 protocol SHA256:
`10c8e13c09bddd103689e0ba10d50676672cc37b659df53def7c9c6e7fc82092`.

Verified repo_id: `repo_666797171f0141c58bf05f579b2ee16e`; manager session:
`01a06be6-2904-7c62-9e7d-1245c34a5312`. Worker-role Source Graph tools remain
unexposed under the recorded NeedFix. Exact manager-supplied document targets
were used without manager-role impersonation or context-database writes.
Owner Task MCP suspension remains active. This subtask writes only this plan
and its companion contract; no code, test, build, benchmark or policy is changed.
Stop at manager review.
