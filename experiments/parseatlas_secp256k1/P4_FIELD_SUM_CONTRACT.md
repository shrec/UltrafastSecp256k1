# P4 field sum: native 4x64 bounds and 25 observation lenses

Status: independently derived mathematical/design contract, submitted for
manager review. P4 implementation qualification, timings and acceptance are
**PENDING IMPORT**. No production integration is authorized by this document.
The owner's field-only collection phase and Task MCP suspension remain in force.

The fixed [P4 protocol](P4_FIELD_SUM_PROTOCOL.md) asks whether P3's final-only
sum gains survive native radix-2^64 accumulation without FE52 input packing.
The [P3 contract](P3_FIELD_SUM_CONTRACT.md) remains unchanged. P3 measurements
motivate the controls; they are not P4 measurements or proof of a P4 speedup.

## Exact primitive, domain and observation

Return `S = (x0 + sum(rhs[i], i=0..N-1)) mod p`, where
`p = 2^256 - 2^32 - 977`. **N counts RHS operands**, not the seed.
Global x0 contributes exactly once. Every input is a canonical FE64 object:
its four raw unsigned 64-bit limbs decode to an integer in `[0,p)`.
An object that merely serializes to such a value after normalization does not
satisfy this raw-input precondition.

- For `N>0`, RHS denotes a valid readable range of N canonical FE64 objects.
  This is a precondition, not a new runtime validation/rejection API. A null
  positive-length pointer is outside the contract; no throw or rejection is
  promised and invalid-memory dereferences are not arithmetic test cases.
- For `N=0`, a null RHS pointer is allowed and must not be dereferenced.
  Return canonical x0 unchanged, without an arithmetic chunk.
- All input objects and bytes remain unchanged. x0 may alias any RHS element:
  its seed contribution and its later indexed RHS contribution both count.
  Private state must not alter either one. No cross-type alias or caller-owned
  output-buffer overlap contract is introduced; the result is returned by value.
- Only the final canonical FE64 value and all 32 big-endian output bytes are
  externally observable. Tests must inspect the four raw result limbs before
  serialization as well as every byte. A serializer normalizing a copy cannot
  be allowed to conceal a noncanonical or otherwise wrong raw result.

Internal raw columns, carry states and canonical chunk checkpoints are proof
and test observations, not a returned prefix-sum API. Requiring every prefix to
be materialized would change this contract and remove its deferred-observation
freedom. Fixed chunks and bank assignments depend on public sizes/indices, not
on whether an input happens to be small or zero. No constant-time certification
follows from that design statement.

## Fixed routes and chunk accounting

| Route ID | Route | RHS per chunk H | RHS accumulation banks |
|---:|---|---:|---:|
| 0 | Existing corrected FE64 returning-add API | 1 | 1 |
| 1 | Inline eager native 4x64 | 1 | 1 |
| 2 | Native wide16 lane1 | 16 | 1 |
| 3 | Native wide4094 lane1 | 4094 | 1 |
| 4 | Native wide16 lane4 | 16 | 4 |
| 5 | Native wide4094 lane4 | 4094 | 4 |
| 6 | Unchanged P3 FE52 E2E Full16 | 16 | Compiler-selected |
| 7 | Unchanged P3 FE52 E2E Full4094 | 4094 | Compiler-selected |

The native wide template admits `1 <= ChunkRhs <= 4095` and only
`Lanes in {1,4}`. The measured native chunks are 16 and 4094; tests additionally
cover 1 and 4095. Invalid chunk0, chunk4096 and lanes2 must fail compilation.
The maximum is a deliberately sufficient admitted bound, not the largest
possible use of unsigned 128-bit storage.

Every nonempty chunk consumes k new RHS operands with `1 <= k <= H`, plus
one canonical running-prefix seed. Its local total is `M=k+1 <= 4096`.
All RHS banks start at zero. A disjoint partition assigns each of these k RHS
operands to exactly one bank; merge then adds the prefix seed exactly once.
It must not seed every bank, nor add the seed both before and after merging.
After canonicalization, the current prefix becomes the next chunk's seed.
That is not another addition of the original x0.

For `N>0`, there are `C=1+floor((N-1)/H)` chunks, N RHS contributions and C
seed-at-merge contributions to their respective local sums. There are C native
carry-propagation/reduction boundaries, including the last one. The already
canonical native result can be constructed directly as FE64; no FE52 conversion
is involved. Final byte materialization remains part of every timed job.
These are source-level algorithm counts, not emitted instruction counts.
N=0 has no chunk and no seed-at-merge operation.

The FE52 controls retain P3's own normalization/packing contract unchanged:
their full chunks admit at most 4094 RHS plus one canonical seed, and final
`to_fe()` supplies the last full normalization. P4's native4095 admission must
not be imported into the direct-full FE52 route. The timed equal-chunk pairs
use 16 and 4094, not mismatched maximal capacities.

## Native wide proof: exact accumulation and reduction

Let `W=2^64`, `B=W^4=2^256` and `K=2^32+977`, so `p=B-K`.
Limb0 is least significant. Equations in this proof are ordinary nonnegative
integer equations; every stored unsigned 128-bit intermediate must satisfy the
bounds below before any narrowing conversion. Compiler support for a genuine
unsigned 128-bit integer type is a prerequisite of this candidate.

### Zero-based banks and one seed at merge

For bank b, let m_b be its RHS count and let
`A[b][i] = sum(rhs[j].limb[i] assigned to bank b)` for `i=0..3`.
The m_b sum to k; initially all A are zero. Each limb is at most W-1, hence

`0 <= A[b][i] <= m_b*(W-1)`.

Merge computes `Q_i = seed.limb[i] + sum_b A[b][i]`, once per limb. Thus

`0 <= Q_i <= M*(W-1) < 2^76 < 2^128`, since `M <= 4096`.

Every accumulation prefix and every partial merge has the same or a smaller
bound, because all terms are nonnegative. Neither one nor four banks can
overflow unsigned128 during accumulation or merge. Four banks do not multiply
the total term budget: their RHS sets partition the same k operands.
The seed-at-merge phase makes its single contribution directly observable in
tests; physically seeding one bank would be mathematically equivalent but is
not the specified phase layout.

### Carry propagation across all four columns

Set `c_0=0`. For each `i=0..3`, compute

`t_i=Q_i+c_i`, `L_i=t_i mod W`, `c_(i+1)=floor(t_i/W)`.

Inductively, `c_i <= M-1` implies

`t_i <= M*(W-1)+(M-1) = M*W-1 < 2^76`,

and therefore `c_(i+1) <= M-1`. The base c0 satisfies the bound. No addition
wraps unsigned128; narrowing to each L_i discards only the explicitly retained
carry. With `h=c_4`, `0 <= h <= M-1 <= 4095`, and the exact local integer sum is

`S_chunk = L + h*B`, where `L=sum(L_i*W^i, i=0..3)` and `0 <= L < B`.

The carry out of limb3 is h, not an ignorable overflow. Dropping it before
modular reduction would change the residue by h*K modulo p.

### First fold and its possible overflow bit

Because `B = K mod p`, first replace h*B by `f=h*K`. The product satisfies

`0 <= f <= (M-1)*K < 2^45 < W`.

Compute `T=L+f` with carry propagation through all four 64-bit limbs, not only
limb0. It obeys `T < B+(M-1)*K < 2*B`. Thus its outgoing high value is a single
bit e, and `T=R0+e*B` with `0 <= R0 < B`. Residue equality has been preserved.

### Second fold cannot overflow bit256

The second fold computes `R=R0+e*K`. If e=0, R=R0<B. If e=1, the first fold
crossed B, and L<=B-1 yields `R0=L+h*K-B <= h*K-1`. Consequently

`R <= (h+1)*K-1 <= M*K-1 < 2^45 < B`.

Therefore the second fold's high output is **provably zero** throughout the
admitted domain. A second generic `fold_high` invocation is sufficient; an
unbounded repeat-until-small reduction is unnecessary. Each fold's low-word
addition fits unsigned128, and its carry must still propagate across all limbs.

The fold coefficient is a number, not a mask: implement `high*K`. For the
second fold's 0/1 high, `K & high` is not equivalent. A mask-based expression
would require a full-width zero/all-ones mask, such as `K & (0-high)` with
appropriately unsigned types. Direct multiplication avoids that ambiguity.

### One final canonical correction

After two folds, `0 <= R < B < 2*p`. Therefore either R is already below p or
exactly one subtraction of p makes it canonical. Equivalently compute `R+K`
across all four limbs: its carry q is one exactly when R>=p. Select its low
256-bit value when q=1, otherwise retain R. The selected result is in `[0,p)`
and equals S_chunk modulo p.

For every chunk, the bank sums, propagation and folds preserve the local
integer or its residue as stated. The resulting canonical value is the next
seed, so induction over chunks proves `(x0+sum N RHS) mod p`, with x0 once
globally. This is a written integer-bound proof, not machine-checked proof or
evidence that an unreviewed implementation follows it.

## Inline eager: specialized two-canonical-input reduction

The eager route need not use the generic wide reducer. For canonical a and b,
let `S=a+b<2*p`, and compute its raw four-limb sum plus high bit:
`S=L+c*B`, with c in {0,1}. Compute `U=L+K=v+q*B`, with `0<=v<B` and q a bit.

- If c=0, S=L<B and q is one exactly when S>=p. Returning v for q=1 and L
  otherwise performs the necessary single canonical correction.
- If c=1, `L=S-B < B-2*K`, so `U=L+K < B-K=p`. Thus q=0 and v is canonical;
  moreover `v=L+K=S-p`.

Consequently selecting v when `c|q` is one, and L otherwise, returns the
canonical sum. The carries c and q cannot both be one under the canonical
two-input precondition. The boolean selection must use the implementation's
correct unsigned-mask or selection semantics, not mistake a carry bit for a
full-word mask. Repeating this addition for each RHS proves the eager route.

This specialization relies on `a,b<p`; it is not a reducer for arbitrary raw
limb arrays or lazy accumulators. Likewise, the wide proof and its admitted
chunk budget are not a contract for subtraction/negation, multiply/square
columns, noncanonical inputs, unrelated moduli or silently larger schedules.

## Correctness and measurement gates

Independent native C++ Boost ordinary-integer arithmetic is the mathematical
oracle; the corrected existing FE64 route is an additional reference, not the
sole oracle. Check exact raw output limbs, `[0,p)` canonicality, every final
byte and input preservation for all routes. Cover canonical boundaries, carry
cascades, zeros/cancellation, nonzero seeds, aliases, null identities, random
inputs, the retained P2 witness, chunk/lane tails and multiple chunks.

On a bounded replay corpus, inspect actual phase-helper outputs before later
normalization: each bank and merged u128 column, propagated four limbs and h,
first-fold low value and e, second-fold zero high, and final canonical limbs.
Compare ordinary integers at each stage. A final byte-only equality cannot
establish these phase invariants. Large whole-call cases not replayed at every
checkpoint must be labelled accordingly. Repeat compiler/sanitizer runs on the
same corpus are qualifications, not additional independent input corpora.

All eight timed routes read the same canonical FE64 array: RHS layout is32N
bytes, or32MiB at N=1048576. FE52 E2E packs each input at use inside the timed
sum; it does not use P3's preconverted resident40N-byte array. Native routes
need no such representation conversion. Wide bank payload is mathematically
four u128 columns per bank (64 or256 bytes); actual registers, stack allocation,
merging temporaries and peak live state depend on generated code. These layout
counts establish neither memory-traffic reductions nor cache/bandwidth causes.

Allocation, corpus generation and untimed validation stay outside jobs. Every
job includes full32-byte output materialization. One core deliberately measures
carry dependencies and compiler scheduling without inter-thread dispatch. Four
banks are one-core independent accumulator chains, not four threads or a claim
of SIMD execution. This exception does not prescribe a sequential production
batch design or claim its multicore optimum.

The protocol fixes48cells (eight routes, six N values), two complete series,
two warmups and eight measured rounds per cell. Primary ratios compare each
candidate with route0 in the same round. Secondary comparisons below are also
preregistered, not discoveries selected after seeing timings. Region extension
keeps one continuous clock and every short prefix; actual total jobs determines
ns/sum and ns/RHS. Preserve cumulative probes and all calibration attempts;
explicitly fail rather than declare an unfinished/short region qualified. The
extra clock-check overhead belongs to the timed region and must be disclosed.

| Fixed comparison | What it can distinguish | What it cannot isolate by itself |
|---|---|---|
| Route0 / route1 | Existing API versus inline eager complete implementation | Pure call overhead: reduction code, visibility and materialization also differ. |
| Route1 / routes2..5 | Eager versus deferred native carry/canonicalization schedules | One instruction's cost or normalization alone, independent of types/codegen. |
| Route2 / route4; route3 / route5 | One versus four banks at equal chunks | Pure ILP gain without bank merge, register or stack overhead. |
| Route2 / route3; route4 / route5 | Chunk16 versus chunk4094 for the same bank count | A universal optimal chunk or unmeasured dispatch threshold. |
| Route2 / route6; route3 / route7 | Native lane1 versus FE52 E2E at equal chunks | Pure algebraic attribution: radix, packing and compiler choices all differ. |
| Route4 / route6; route5 / route7 | Native lane4 versus FE52 E2E at equal chunks | Representation alone, independently of native bank partitioning. |

For A/B above, a ratio greater than one favors B. No historical P3/P2/P1 timing
is pooled into P4 ratios. More operations may still produce a faster complete
sum, but that remains an empirical question. Static code inspection may show
actual instructions, calls, vectorization and stack accesses; it does not
provide dynamic cycles, traffic, port pressure or cache-miss attribution.

## Exactly 25 P4 observation lenses

**DERIVED** denotes reasoning under this contract. **MEASURED (P3)** imports
earlier evidence only; P4 outcomes remain **PENDING IMPORT**. **HYPOTHESIS**
denotes an unmeasured P4 expectation. Lens identities match the P3 contract.

| Lens | Current evidence / P4 application | Next gate or limitation |
|---|---|---|
| V01 Reachability | **DERIVED:** eight fixed routes include existing FE64, inline eager, four native-wide controls and two unchanged P3 FE52 E2E controls. | Check actual compiled route and source/binary binding; no production dispatcher integration. |
| V02 Corpus | **MEASURED (P3):** finite canonical boundaries, seeded aliases and chunk checkpoints exposed the observation contract. | Preserve the shared FE64 recipe/seed/checksum; add native lane/chunk tails, maximum4095 and phase evidence; P4 outcomes: **PENDING IMPORT**. |
| V03 Equivalence | **DERIVED:** column/merge equality, carry decomposition and two folds preserve each chunk's residue; prefix induction counts x0 once. | Independent C++ ordinary-integer oracle plus corrected reference, exact raw limbs and all bytes; P4 gate: **PENDING IMPORT**. |
| V04 Definedness | **DERIVED:** canonical inputs and valid N-element ranges are preconditions; N0/null returns identity. | No positive-null rejection promise; require genuine unsigned128 and defined shifts/narrowing. |
| V05 Range / overflow | **DERIVED:** M<=4096 gives columns/carry intermediates<2^76, h<=4095, hK<2^45 and zero high after the second fold. | Reject unsupported templates and test maximal admitted chunks; not a universal u128 capacity theorem. |
| V06 Aliasing | **DERIVED:** read-only inputs and private state allow same-type x0/RHS alias while retaining both mathematical contributions. | Check all input bytes before/after; do not invent cross-type or output-buffer aliasing contracts. |
| V07 Observable output | **DERIVED:** final-only canonical FE64/32BE permits deferred boundaries. | Raw limbs must be checked before serialization; prefix observations remain internal test invariants. |
| V08 Invariants / carries | **DERIVED:** four raw columns per bank retain independent chains; seed-at-merge is once; hB becomes hK, then eB becomes eK. | Replay actual columns, all four propagated limbs, both highs and canonical correction; never treat a carry bit as an all-ones mask. |
| V09 Constant time | **DERIVED:** public N/chunks/bank partition do not establish secret-independent machine execution. | No CT certification, security substitution or secret-value-dependent scheduling; qualification: **PENDING IMPORT**. |
| V10 Resources | **DERIVED:** all timed input layouts are32N bytes; native bank payload is64 or256 bytes before other state. | Actual live registers, stack and temporary storage require emitted-code evidence; no heap/working-set inference. |
| V11 Property visibility | **MEASURED (P3):** the six final-only candidates won480/480 baseline pairs for N>=16, but native-wide controls were absent. | P4 asks whether the advantage depends on packing/representation or deferred observation; those P3 wins are not P4 results. |
| V12 Dependency depth | **HYPOTHESIS:** deferred native carries and four RHS banks may shorten repeated dependencies. | Equal-chunk lane1/lane4 comparisons include merge overhead; operation counts alone do not measure critical-path cycles. |
| V13 Live state | **HYPOTHESIS:** four banks may offer scheduling freedom or cause register/stack pressure. | Inspect actual live-state consequences and spills; no measured gain or spill cost is assumed. |
| V14 Operations | **DERIVED:** N RHS contributions, C once-per-chunk seeds, C propagation/two-fold/correction boundaries in the wide algorithm. | Eager uses specialized c|q selection each RHS; source-level operations need not survive unchanged in optimized code. |
| V15 Conversion | **DERIVED:** native routes read FE64 directly; P3 E2E controls pack at use; no resident FE52 array is timed. | Equal input/storage boundary does not isolate conversion cost from radix/arithmetic/compiler choices. |
| V16 Latency | **HYPOTHESIS:** complete native sums may improve ns/sum against the actual API and FE52 controls. | Report all whole-sum samples and same-round ratios; not isolated add latency or individual-call percentiles. |
| V17 Throughput | **HYPOTHESIS:** bank independence may overlap work on one core. | ns/RHS includes amortized final output; no automatic SIMD, multicore or upper-layer speed claim. |
| V18 Instructions / PMU | **RECORDED prior manager diagnostic:** perf cycles/instructions access failed exit255 with perf_event_paranoid=4; policy unchanged. | Dynamic events are unavailable, not zero; emitted-code inspection is not dynamic PMU measurement. |
| V19 Cache / bandwidth | **DERIVED:** all eight routes read the same32N-byte FE64 input prefixes, reaching32MiB. | Identical layout does not imply identical traffic/residency; timing alone cannot establish memory causality. |
| V20 Code size | **HYPOTHESIS:** templates, four banks and inlining may alter emitted size and control flow. | Record actual symbols/call boundaries and stack accesses; byte size alone does not establish instruction-cache cost. |
| V21 Portability | **MEASURED (P3):** three compiler/sanitizer configurations passed the same finite arithmetic corpus. | Repeat native GCC, independent Clang and NO_ASM ASan+UBSan for P4; no unsupported-u128 or new-architecture qualification is inherited. |
| V22 Compiler | **HYPOTHESIS:** u128 lowering, bank unrolling, FE52 packing and vectorization may change schedule outcomes. | Freeze compiler/flags/macros/no-LTO boundaries and source/binary hashes; inspect actual generated code. |
| V23 Policy / ties | **MEASURED (P3):** later duration drift left one S2 N1 Full16 cell short despite qualified calibration. | P4 retains continuous-region extensions/probes and actual jobs, every loss/tie and preregistered comparison; no selected reruns or inferred thresholds. |
| V24 Transfer | **DERIVED:** this is a final-only field-sum control, not a scalar, point or signature result. | Production integration and upper-layer transfer remain deferred; no claim field research is complete. |
| V25 Prior art / novelty | **DERIVED from local provenance:** the pseudo-Mersenne identity is reused while accumulator representation and observation frequency are tested. | No novel-mathematics, novel-algorithm, world-record or security claim; external novelty classification: **PENDING IMPORT**. |

## Review, provenance and limits

This subtask writes only this document. No arithmetic source, production file,
test, build setting or earlier evidence is modified, and no benchmark/build
is run here. Implementation details above reflect the bounded kernel-design
handoff, not an assertion that the final code has already been inspected.
Correctness gates must run before manager implementation/contract review.

The manager supplied exact targets and the verified repository/session route.
Worker-specific Source Graph tools are not exposed; discovery returned no
worker source-graph entry. This worker used only exact supplied document reads
and did not impersonate manager tools. Owner Task MCP suspension remains active.
Repository identity: `repo_666797171f0141c58bf05f579b2ee16e`; manager session:
`01a06be6-2904-7c62-9e7d-1245c34a5312`.

Aligned protocol SHA256:
`1b408b7d4a69c74843c36b9b6014fa432d6ea903934efde1613b1b1d64a81aaa`.
Unchanged P3 contract SHA256:
`957a0b51b71f4a9412728bb21928ad9bfa03538c38ab6dbb58cefe4184037ac3`.
Imported P3 measured claims are retained in
[P3 results](P3_FIELD_SUM_RESULTS.md) and the
[P3 independent numeric audit](data/p3_numeric_audit_20260905.json); they are
not repeated P4 experiments. Stop at manager review before treating any P4
implementation or performance claim as accepted.
