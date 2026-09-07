# Collected research frontier — field first, integrate later

Current owner decision, 2026-09-05: retain all measured findings and negative
results; finish field-p candidates before returning to scalar-n candidates;
integrate qualified changes together later. No universal algorithm is required.

| Finding | Evidence / observed scope | Disposition |
|---|---|---|
| F2 Columns scalar sum | [F2 results](F2_REDUCTION_RESULTS.md): final modular-n sum, 2.54–2.56x at 16 RHS operands, 7.31–7.38x at 65536; old path wins at 1 RHS | Retained experimental candidate. Initial x0 is included once. Not one scalar-add or field-p gain. CT adaptation and caller integration deferred |
| P1 arithmetic reference | [P1 results](P1_PRIMITIVE_RESULTS.md): field/Scalar baseline; independent oracle exposed two field reduction defects, minimal repairs tested in four configurations | Corrected frozen reference, not a new performance algorithm. Original evidence retained |
| P2 canonical resident FE52 | [P2 results](P2_FIELD_RESULTS.md): add/sub win 64/64 pairs; chain square 16/16; chain multiply 13/16. FE64 wins ILP4 mul/square 32/32 | Retain conditional region candidates, not drop-in replacements. Resident setup boundary explicit |
| P2 FE52 per-op bridge | Full canonical conversion/normalization; loses 128/128 add/sub/mul/square pairs; inverse has no stable advantage | Retained negative result for this exact wrapper/build/workload |
| P2 normalization headroom | [NeedFix](data/p2_lazy_needfix_20260905.json): exact 4096-term sum can lose 2^64 during direct full normalization; weak-then-full is correct | Record input-contract/ordering constraint. No production caller reachability or timing gain asserted |
| P3 bounded field-p sum | [P3 results](P3_FIELD_SUM_RESULTS.md): final-only x0 + N canonical RHS; E2E candidates about 5.6–5.8x at N=16, 11.7–12.6x at medium sizes, Full16 6.97–7.38x at N=1048576 | Retain experimental vector primitive; not a single field-add replacement. All N>=16 candidates win 480/480 paired comparisons; N=1 medians favor the old path |
| P3 resident and scheduling observations | Resident Weak4095 peaks 29.47–30.13x at N=256 with preprocessing excluded. At N=1048576, Full16 beats larger E2E chunks and E2E Weak4095 beats resident, each in 15/16 exploratory pairs | Keep setup contracts separate; no universal chunk size, memory-bandwidth causality or production dispatch threshold inferred |
| P3 validation / duration boundary | Same 3,703,557 C++ assertions pass in GCC, independent Clang ASM and NO_ASM sanitizers. S2 N=1 Full16 has 1 short warmup and 8 short measured regions despite qualifying calibration | Preserve [duration deviation](data/p3_duration_floor_deviation_20260905.json); no samples removed. All N>=16 warmup/measurement regions meet 200 ms |
| P4 native field-p sum controls | [P4 results](P4_FIELD_SUM_RESULTS.md): native Wide4094/lane1 gives 7.66–7.87x at N16, 14.82–15.23x at N256, 16.73–16.78x at N4096, 14.50–14.60x at N65536; Wide16/lane1 7.89–8.09x at N1048576 | Retain final-only FE64-input candidates; same-profile comparison, not standalone add or whole-engine gain. All seven candidates beat API in560/560 N>=16pairs |
| P4 eager and bank controls | Inline eager beats API 1.13–1.14x at N1 and about2.03–2.33x at N>=16. Four-bank controls lose every matched one-bank series median (8/192 isolated wins) | Keep implementation-boundary and scheduling findings separate; no pure call-overhead or spill-cost attribution |
| P4 representation/size crossover | At N4096 native4094/lane1 beats FE52 Full4094 16/16pairs; at N1048576 the FE52 control wins14/16. Native16/lane1 beats FE52 Full16 12/16 at N1048576 | No universal representation or chunk size; no production dispatcher threshold inferred |
| P4 qualification | Three C++ configurations each pass21,130,305assertions; all960 full-series warmup/measurement regions reach200ms, with21continuous extensions retained | [Arithmetic gates](data/p4_validation_20260905.json); synthetic duration diagnostics kept separate from real performance. Security/platform gates remain open |
| P5 row-product chain candidate | [P5 results](P5_FIELD_PRODUCT_RESULTS.md): inline row+serial multiply wins15/16 chain pairs at1.119/1.107x; row self-product square wins16/16 at1.079/1.183x | Retain conditional canonical field candidates. Complete implementation/API-boundary comparison, not pure call-overhead or multiplication-count gain; no integration |
| P5 independent-state and Comba negatives | Actual FE64 API wins96/96 ILP4 candidate pairs and128/128 Comba/symmetric pairs across both modes; counts overlap. Row+serial beats Comba+serial64/64 | No universal primitive replacement. Public column loops and scalar lane scheduling remain in generated C++; no causal timing attribution |
| P5 compiler-equivalent reducer controls | Five of six serial/precomputed full-job pairs have identical normalized instructions; raw chain differs only in two XOR-staging instructions. Row square already compiles to10product multiply sites | Retain compiler/representation evidence; small ratio differences do not establish a distinct reducer algorithm gain |
| P5 qualification | Three C++ configurations each pass714,283assertions,0mismatches; all400later regions reach200ms,25continuous extensions and42short calibration records retained | [Arithmetic gates](data/p5_validation_20260905.json), [numeric audit](data/p5_numeric_audit_20260905.json); arbitrary512reducer and130/129/65bit bounds separate from sum contracts |

| P6 existing-ASM boundary candidate | [P6 results](P6_FIELD_BOUNDARY_RESULTS.md): direct unchanged ASM beats actual API64/64 pairs; multiply chain1 1.117/1.123x, ILP4 1.239/1.236x; square chain1 1.065/1.070x, ILP4 1.299/1.289x | Retain implementation-boundary candidate, not new arithmetic, pure call-cost attribution or whole-engine gain |
| P6 inline-row chain refinement | Inline/API multiply wins15/16 and square16/16. Against direct ASM, multiply is near-tied at1.01058/1.01099x (12/16); square wins16/16 at1.07447/1.08815x | Preserve conditional inline-square candidate; do not promote near1percent multiply difference as robust |
| P6 opaque-row and independent-state controls | Outlined row loses to API64/64 and direct ASM64/64 across all groups. All row ILP4 candidates lose to API64/64. Outlined multiply nevertheless beats inline multiply in ILP4 16/16 at1.090/1.092x | No universal inline/outlined winner or drop-in row replacement; operation/mode boundary matters |
| P6 emitted-code / security negative | [Codegen](P6_FIELD_BOUNDARY_CODEGEN.md): eight direct/outlined caller pairs match after named leaf-target normalization; actual ASM bytes unchanged. Clang shared reducer has a result-dependent correction branch | Boundary control is established, not CT safety or dynamic bandwidth/spill causality. Compiler-specific security qualification required |
| P6 qualification / duration | GCC and independent Clang each406239 assertions; NO_ASM sanitizers358744 with47498 explicit ASM skips. All320later regions meet200ms;9extensions/35shortcal retained | [Arithmetic gates](data/p6_validation_20260905.json), [numeric audit](data/p6_numeric_audit_20260905.json). Possible brief metadata-check overlap recorded; no reruns or integration |

## Remaining field-only sequence

Owner clarification: unconventional GPU views motivate the broad
[representation-search plan](FIELD_REPRESENTATION_SEARCH_PLAN.md), not a search
limited to pointers. The bounded [P7 GPU side probe](P7_CUDA_VIEW_RESULTS.md)
found no useful shift/memcpy view-only gain despite different PTX; existing
hybrid beats its all64 contrast about1.16x/1.51–1.52x in two workloads.
This is existing GPU algorithm evidence, not a new CPU or whole-engine gain.
All78,521oracle assertions pass; GPU memory sanitizer remains unqualified.

1. P3/P4 final-only sums now include native 4x64 controls and continuous-region
   duration qualification. Remaining sum gates: setup-inclusive crossover,
   additional compiler/platform profiles and security; no integration yet.
2. P5/P6 multiply/square now have bounds, native measurements and matched
   API/direct-ASM/inline/opaque-row controls. Next test final-fold selection,
   unrolled Comba and fused product/reduction in separate bounded waves; replicate
   compiler/platform profiles and security before transferring retained candidates.
3. Inversion within separate VT/CT contracts; public batch inversion as a distinct
   vector primitive with zero, scratch, alias and setup costs made explicit.
4. Working-set, compiler/platform and security qualification of retained winners.

Detailed observations: [P2 lenses](P2_FIELD_25_LENSES.md) and
[P3 measured 25-lens outcomes](P3_FIELD_SUM_LENS_RESULTS.md), and
[P4 measured 25-lens outcomes](P4_FIELD_SUM_LENS_RESULTS.md), and
[P5 measured 25-lens outcomes](P5_FIELD_PRODUCT_LENS_RESULTS.md), and
[P6 measured 25-lens outcomes](P6_FIELD_BOUNDARY_LENS_RESULTS.md).
Protocols: [P2](P2_FIELD_PROTOCOL.md), [P3](P3_FIELD_SUM_PROTOCOL.md),
[P4](P4_FIELD_SUM_PROTOCOL.md), [P5](P5_FIELD_PRODUCT_PROTOCOL.md),
[P6](P6_FIELD_BOUNDARY_PROTOCOL.md).
Unmeasured quantities remain `PENDING IMPORT`; no promised speedup or discovery.

This central ledger is mutable. Its exact pre-P3 bytes are retained in
[the P2 frontier snapshot](data/p2_frontier_snapshot_20260905.md), SHA256
`54616857756547e61416bc64a9961b8159df7bb84404e29c02e88e3f7f20c20a`,
so the older P2 manifest's historical ledger hash is not silently rewritten.
Its exact pre-P4 bytes are likewise retained in
[the P3 frontier snapshot](data/p3_frontier_snapshot_20260905.md), SHA256
`df2995a5f27cfe527e24c0557fd26957641c41f39107dd682bf4a95f612472ae`.
Its exact pre-P5 bytes are retained in
[the P4 frontier snapshot](data/p4_frontier_snapshot_20260905.md), SHA256
`dd09f228bf6b9cf9c9af6b57862eb4f288e1a40398ad0170ed3a00a925253522`.
Its exact pre-P6 bytes are retained in
[the P5 frontier snapshot](data/p5_frontier_snapshot_20260905.md), SHA256
`25370020b8c7fcada0675f86418667d63ae6673cfe3accd1ab625c73a4bd84be`.
Its exact pre-P7 bytes are retained in
[the P6 frontier snapshot](data/p6_frontier_snapshot_20260905.md), SHA256
`2105984a9f7b8c1cbd0e5e9bfce964039c7e4afcf4b73946ae59660bf0566a4f`.
Original P2/P3/P4/P5/P6 manifests remain unchanged. Relative links inside byte snapshots
use the original ledger directory.

After the field collection gate, resume scalar primitives, then evaluate combined
integration and actual upper-layer transfer. Each candidate keeps its exact
contract, original reference bytes, independent oracle, positive/negative data,
public selection condition and unresolved security/portability requirements.
