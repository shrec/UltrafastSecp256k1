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

## Remaining field-only sequence

1. Bounded deferred-normalization regions: explicit decode, permitted observations,
   canonical-summand count, normalization schedule and whole-region timing.
2. Multiply/square and sparse-prime reducer scheduling: full accumulator bounds,
   emitted dependencies, register/stack evidence, both chain and independent modes.
3. Inversion within separate VT/CT contracts; public batch inversion as a distinct
   vector primitive with zero, scratch, alias and setup costs made explicit.
4. Working-set, compiler/platform and security qualification of retained winners.

Detailed observations: [25 lenses](P2_FIELD_25_LENSES.md).
Current protocol: [P2 field protocol](P2_FIELD_PROTOCOL.md).
Unmeasured quantities remain `PENDING IMPORT`; no promised speedup or discovery.

After the field collection gate, resume scalar primitives, then evaluate combined
integration and actual upper-layer transfer. Each candidate keeps its exact
contract, original reference bytes, independent oracle, positive/negative data,
public selection condition and unresolved security/portability requirements.
