# Normative methodology (English)

This document and `METHODOLOGY.ka.md` have equal normative force. The JSON Schema
is the sole definition site for IDs and executable record obligations.

## Normative parity block

`CONTRACT_VERSION=2; WORLD_COUNT=25; VIEW_COUNT=25; MATRIX_CELLS=625; PRIMITIVE_CLASSES=4; RANKING=search_reachable>boundary_equivalent>invariants_eligible>resource_feasible>cost_pareto_within_feasible_slice; CLAIM_CLASSES=OBSERVED,PROVEN,VERIFIED,HYPOTHESIS,MEASURED,CORRECTION; HISTORICAL_REVISION=d71b406c95141d81749671431a4c0f4605e0c4e4; GUIDANCE_REVISION=13fac0a300cf07eac9424434ec1e2c6bd5f75564; PARALLEL_WORLDS_REVISION=f428fb53948bc15c4668c303366231817a947708; INPUT_EXPERIMENTS=0037,0038,0039,0040,0041; E0039=46:1_MODEL_DEFECT+8_OPERATOR_LANGUAGE+18_TOKEN_WORD+19_RULE_LIMIT; E0041=58_ENTRIES,25_WORLDS,1450_CELLS,52_FORCED,6_MOVABLE,TAX_FLOOR_21,VARIABLE_1..4,TIE_CHANGES_52,MAX_DEPTH_RATIO_BOUND_2.333,MUTANTS_19_OF_22`

## Fixed problem, objects, and W01–W25

The founding question transferred here is: if mathematical traditions chose
different foundational representation conventions, how would each solve the
same secp256k1 problem with its natural primitives? Fixed objects are field
elements modulo `p`, scalars modulo the distinct modulus `n`, group points
(including infinity), and signature verification inputs/results. Modulo `p` and
modulo `n` evidence is never interchangeable.

W01–W25 are 25 reusable, internally consistent representation policies, not 25
precedence grammars, unrelated knobs, algorithms, or claimed discoveries. Each
policy governs permitted derivations, carried state, normal forms, encode/decode,
invariants, and evaluation. It is applied separately to every fixed primitive or
region. Every world/primitive cell must say `APPLICABLE` and give its concrete
output representation, or `N/A` with a reason. Unknown is not N/A. Collapsed or
duplicate worlds are retained honestly. The 25-policy design is neither complete
nor a theorem. W01 is the unchanged production control.

For every applicable candidate the obligation is
`decode_out(candidate(encode_in(x))) = baseline(x)` over declared input/output
domains. Record ranges, definedness/preconditions, all carried state,
caller-visible behavior, overflow and aliasing. A boundary witness establishes
reachability, not equivalence. Rational cancellation cannot certify limb-machine
arithmetic without its range, overflow, and finite-field obligations.

The original current engine at source commit
`fef231d4e4173bd016fb2a3a1eff67087396a203` is the frozen differential reference.
Every actual experiment records its source and build hashes, compiler, flags, and
target, and compares canonical output bytes plus defined caller-visible status/state
on identical deterministic inputs. It also checks modulo `p`/`n` independently with
unbounded integers. Different internal representations are allowed, but independently
valid decode/normalization must not conceal carry, range, or alias violations.
`FieldElement::to_bytes` merely serializes stored limbs (`field.cpp:2517-2526`); it is
not normalization. Byte-corpus agreement is OBSERVED, not full-domain proof. Actual
C++ reference integration belongs to `ORACLE_011`, outside this contract card.

## V01–V25 and evidence

## Canonical world catalog (IDs reference the schema)

Each row states policy/state; conversion boundary; invariants; applicability. Except W01's
declared all-primitive control, support is decided per primitive by an APPLICABLE record
with a concrete output representation or a reasoned N/A record.

|ID|Policy and carried state|Boundary; invariants|Support|
|--|--|--|--|
|W01|production control/state|identity; baseline behavior|all four primitives|
|W02|canonical least residue|canonical integer; congruence/range|record-time|
|W03|centered signed residue|canonical mapping; congruence/signed range|record-time|
|W04|bounded redundant value+magnitude|normalize/decode; congruence/magnitude|record-time|
|W05|Montgomery `xR` state|R/R-inverse; Montgomery congruence/range|record-time|
|W06|Barrett residue+reciprocal|canonical; congruence/correction bounds|record-time|
|W07|pseudo-Mersenne coefficients|pack/unpack; modulus relation/bounds|record-time|
|W08|saturated limb vector|limb pack; radix value/carry bounds|record-time|
|W09|spare-bit limbs+magnitude|pack/normalize; radix value/limb bounds|record-time|
|W10|mixed-radix digits|weighted decompose/sum; value/digit bounds|record-time|
|W11|carry-save sum/carry channels|combine; combined value/channel bounds|record-time|
|W12|CRT residue tuple|split/reconstruct; congruence/product range|record-time|
|W13|bounded signed digits|radix reconstruct; value/digit bounds|record-time|
|W14|factored-product DAG|evaluate; definedness/product equivalence|record-time|
|W15|numerator/denominator pair|proved inversion; nonzero/equivalence|record-time|
|W16|shared-value DAG|evaluate; node definitions/output equivalence|record-time|
|W17|balanced expression DAG|evaluate; associativity domain/operand multiset|record-time|
|W18|serial accumulator|evaluate fold; associativity domain/order|record-time|
|W19|unreduced accumulator+bound|final reduce; growth bound/congruence|record-time|
|W20|eager canonical residues|canonical boundary; range/congruence|record-time|
|W21|two carry chains+merge|merge; partition identity/carry bounds|record-time|
|W22|SIMD lane tuple|pack/unpack; lane independence/equivalence|record-time|
|W23|bitsliced planes|transpose; reconstruction/lane independence|record-time|
|W24|base+authenticated table|build/validate; derivation/selection behavior|record-time|
|W25|typed region composition|encode/decode; component/caller-visible invariants|record-time|

## Canonical view catalog (IDs reference the schema)

|ID|Observable|Measurement or assessment method|Applicability|
|--|--|--|--|
|V01|search reachability|execute derivation witness|all primitives|
|V02|sampled agreement|seeded differential corpus; OBSERVED only|applicable cells|
|V03|full-domain equivalence|proof or mechanical full-domain artifact|applicable cells|
|V04|definedness/preconditions|proof and invalid-input fixtures|applicable cells|
|V05|range/overflow|range proof and boundary fixtures|arithmetic cells|
|V06|aliasing|alias matrix tests|implementation cells|
|V07|caller-visible behavior|API differential tests|group/signature; exposed field/scalar|
|V08|invariant eligibility|artifact per invariant|applicable cells|
|V09|constant-time eligibility|static trace review and leakage experiment|secret-bearing cells|
|V10|resource feasibility|declared hard ceilings|implementation cells|
|V11|property manifestness|preregistered syntactic extractor|declared properties|
|V12|dependency depth|DAG longest-path static model|operation regions|
|V13|peak live values|liveness analysis|operation regions|
|V14|operation multiset|typed static counters|operation regions|
|V15|conversion cost|typed static operation counts or measured conversion timing|non-identity boundaries|
|V16|region latency|warmed repeated benchmark|executable regions|
|V17|region throughput|warmed batched benchmark|executable regions|
|V18|instruction count|versioned hardware counter|supported targets|
|V19|cache behavior|versioned hardware counters|supported targets|
|V20|code size|linked-symbol accounting|compiled implementations|
|V21|portability|declared build/test matrix|implementation cells|
|V22|compiler sensitivity|fixed version/flag matrix|compiled implementations|
|V23|policy/tie sensitivity|preregistered optimistic/pessimistic reruns|multi-derivation cells|
|V24|rediscovery/transfer|blind generator then unchanged held-out axis|generative policies|
|V25|prior-art status|reproduction/gain/novelty-review classification|positive claims|

V01–V25 independently define observable, method, and applicability in the schema.
Finite seeded differential agreement is OBSERVED evidence (V02), never
full-domain equivalence (V03). PROVEN requires a reviewable derivation and
assumptions; VERIFIED requires a mechanical full-domain artifact; HYPOTHESIS is
unconfirmed; MEASURED requires environment, software versions, corpus, warm-up,
at least two repetitions, statistic, uncertainty, unit, value, and artifacts;
CORRECTION names the retained predecessor, reason, author, and timestamp. No
class silently upgrades another. Unknown measurements are exactly `PENDING IMPORT`.

Negative results retain stable claim identity, seed, artifacts, provenance and
limits: unreachable, non-equivalent, invariant failure, resource infeasibility,
null effect, measurement failure, and collapsed duplicate. A correction appends
history; it does not rewrite it. Explicit invalid-state and rejection-branch
fixtures are mandatory.

## Reachability-first T31 ranking

Ranking is lexicographic: search reachability, proved boundary equivalence,
invariant/constant-time eligibility, then resource feasibility. Only inside a
declared feasible slice may measured costs compare eligible worlds. Dependency
depth, peak live values, operation multiset, conversion cost, region latency and
throughput remain separate Pareto coordinates. Static depth and unit-cost models
are not measurements. Property manifestness is separate from equivalence and cost.

Generator scope must declare permitted token-word/state changes and preserve a
derivation witness. Rediscovery controls do not embed the target identity; an
unchanged axis is then tested on a held-out object. Reports distinguish known-
method reproduction, observed implementation gain, and possible novelty requiring
prior-art review.

## Imported claims and limits

The immutable historical input is ParseAtlas Direction B at
`d71b406c95141d81749671431a4c0f4605e0c4e4`, especially experiments 0037–0040.
Guidance is attributed to `13fac0a300cf07eac9424434ec1e2c6bd5f75564` and 0041
to `f428fb53948bc15c4668c303366231817a947708`; later claims do not rewrite the
historical baseline. Experiment 0033 manually supplies one rational deferred-
division axis; it is not autonomous generation or a finite-field proof procedure.
General synthesis, finite-field discharge, and ranking remain HYPOTHESIS/planned
until implemented and measured in this engine.

In 0039 the 46 missing pairs divide into exactly 1 model defect, 8 operator-
language limits, 18 requiring another token word, and 19 declared rule limits.
Thus neither all 37 residue pairs nor “no world shows everything” is a universal
shape-rule result; both are corpus/rule-relative.

At pinned 0041, 58 entries × 25 worlds = 1450 cells split into 52 forced and 6
movable entries; the tax floor is 21 and variable remainder 1..4. Optimistic vs
pessimistic ties change 52/1450 cells. The maximum depth ratio 2.333 is a BOUND in
a fixed-word unit-cost model, not secp runtime. A fixed tree has invariant depth;
only equivalent writings change it. Constant-floor rank preservation is an
arithmetic identity, not validation, and bracket tax does not transfer to limb
performance. The report killed 19/22 mutants; two rejection-branch fixtures were
missing and one probe was redundant, so it provides no full-mutation-coverage claim.

All conclusions are limited to declared object, policy, view, corpus, rules,
platform, compiler and versions. No result alone establishes cryptographic
security, portability, constant time, general performance, or novelty.
