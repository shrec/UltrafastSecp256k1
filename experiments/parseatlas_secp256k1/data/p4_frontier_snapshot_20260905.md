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

## Remaining field-only sequence

1. P3/P4 final-only sums now include native 4x64 controls and continuous-region
   duration qualification. Remaining sum gates: setup-inclusive crossover,
   additional compiler/platform profiles and security; no integration yet.
2. Multiply/square and sparse-prime reducer scheduling: full accumulator bounds,
   emitted dependencies, register/stack evidence, both chain and independent modes.
3. Inversion within separate VT/CT contracts; public batch inversion as a distinct
   vector primitive with zero, scratch, alias and setup costs made explicit.
4. Working-set, compiler/platform and security qualification of retained winners.

Detailed observations: [P2 lenses](P2_FIELD_25_LENSES.md) and
[P3 measured 25-lens outcomes](P3_FIELD_SUM_LENS_RESULTS.md), and
[P4 measured 25-lens outcomes](P4_FIELD_SUM_LENS_RESULTS.md).
Protocols: [P2](P2_FIELD_PROTOCOL.md), [P3](P3_FIELD_SUM_PROTOCOL.md),
[P4](P4_FIELD_SUM_PROTOCOL.md).
Unmeasured quantities remain `PENDING IMPORT`; no promised speedup or discovery.

This central ledger is mutable. Its exact pre-P3 bytes are retained in
[the P2 frontier snapshot](data/p2_frontier_snapshot_20260905.md), SHA256
`54616857756547e61416bc64a9961b8159df7bb84404e29c02e88e3f7f20c20a`,
so the older P2 manifest's historical ledger hash is not silently rewritten.
Its exact pre-P4 bytes are likewise retained in
[the P3 frontier snapshot](data/p3_frontier_snapshot_20260905.md), SHA256
`df2995a5f27cfe527e24c0557fd26957641c41f39107dd682bf4a95f612472ae`.
Original P2/P3 manifests remain unchanged. Relative links inside byte snapshots
use the original ledger directory.

After the field collection gate, resume scalar primitives, then evaluate combined
integration and actual upper-layer transfer. Each candidate keeps its exact
contract, original reference bytes, independent oracle, positive/negative data,
public selection condition and unresolved security/portability requirements.
