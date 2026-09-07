# P5 field products: exact domains, carry bounds and observations

Status: independent written bound/design argument, submitted for manager review.
P5 arithmetic implementation qualification and performance are **PENDING IMPORT**.
This document authorizes no production integration, scalar or upper-layer change.
It accompanies the fixed [P5 protocol](P5_FIELD_PRODUCT_PROTOCOL.md) and the
separate [25-lens verification plan](P5_FIELD_PRODUCT_LENSES.md).

## Three different arithmetic contracts

Let `W=2^64`, `B=W^4=2^256`, `K=2^32+977` and `p=B-K`.
Limbs are unsigned64, least significant first. Mathematical equations below
use ordinary integers unless explicitly reduced modulo p.

1. Full products: `pa_p5::mul_row(a,b)` and `mul_comba(a,b)` accept arbitrary
   four-limb integers `0<=a,b<B` and return all eight limbs of the exact product.
   `square_comba(a)` returns the exact eight-limb square on that same domain.
   These helpers must preserve all 512 result bits; they are not modular field
   operations and must not first reduce their inputs modulo p.
2. Raw reducers: `reduce_serial(t)` and `reduce_parallel(t)` accept every
   eight-limb unsigned integer `0<=t<B^2` and return its canonical residue in
   `[0,p)`. Inputs need not be products or satisfy any carry/magnitude restriction
   beyond their eight-word width. The second name describes precomputed high-word
   products, not threads or an assertion that every carry becomes independent.
3. Complete field pipelines: canonical four-limb inputs `a,b<p` produce
   `(a*b) mod p` or `a^2 mod p` as canonical raw FE64 and all32big-endian bytes.
   Each operation returns a canonical state, including every dependency step.
   Existing corrected FE64 multiplication and square are differential references;
   an independent ordinary-integer oracle remains required.

All input objects/limbs remain unchanged. Same-object operands, including
`mul_row(a,a)` and the complete `a*a` case, are permitted: there is no restrict
promise. Values are returned by value; this contract adds no caller-owned output
buffer overlap behavior or cross-type aliasing. These are fixed-width primitives,
not pointer/length sum APIs; P3/P4's N=0/null rules do not become new product APIs.

Raw result limbs must be inspected before serialization. A `to_bytes()` that
normalizes a copy cannot conceal a noncanonical field result, and a constructor
that reduces arbitrary input would destroy the full-product test domain.
Eight-limb product equality checks all512bits; reducer/pipeline qualification
checks all four raw output limbs, their `[0,p)` range, and every final32byte.
Invalid pointers or improperly formed C++ objects are not admitted inputs.

## Exact full-product accumulation

### Row schedule: each u128 cell fits

For raw limbs, one product satisfies `a_i*b_j <= (W-1)^2`.
In the standard zero-initialized row schedule, an existing output word and an
incoming row carry are each at most W-1. Their combined cell therefore obeys

`a_i*b_j + out_(i+j) + carry <= (W-1)^2 + 2*(W-1) = W^2-1`.

Thus a genuine unsigned128 temporary retains the exact cell sum. Store its low
word and preserve its high word as the next row carry. After four products in
row i, the carry belongs at word i+4. In this particular row ordering that word
has not been touched by earlier rows: earlier final positions are at most i+3.
Assignment is valid there; an alternative schedule that adds into an already
occupied word needs its own carry propagation and must not silently discard it.

The row invariant accounts for every `a_i*b_j*W^(i+j)` exactly once. There are
16 mathematical 64x64 products. The final integer is below B^2 and fits exactly
eight words. This is a bound for the stated cell schedule, not permission to
sum several full products in one unsigned128 variable.

### Comba schedule: a three-word accumulator is necessary and sufficient

For diagonal k, let
`D_k = sum(a_i*b_j where i+j=k, 0<=i,j<4)`.
Diagonal lengths are `1,2,3,4,3,2,1`. With incoming carry c_k, emit

`s_k=D_k+c_k`, `out_k=s_k mod W`, `c_(k+1)=floor(s_k/W)`, starting `c_0=0`.

Inductively `c_k<=4*(W-1)` gives

`s_k <= 4*(W-1)^2 + 4*(W-1) = 4*W*(W-1) < 4*W^2 = 2^130`,

and `c_(k+1)<=4*(W-1)` again. Therefore each complete column needs at most
130 bits, and its carried remainder can need66bits. An accumulator of three
unsigned64 words has192bits and ample headroom. Two words are not sufficient.
Each added product and any partial column sum obey the same bound, because
all contributions are nonnegative.

When adding a 128-bit product to the three-word state, retain both the low-word
carry and any carry beyond the middle word. After emitting the low word, shift
the remaining two words down; do not narrow the carry to one uint64. The exact
decode invariant is the weighted sum of already emitted words plus the active
carry state and remaining products. It proves the final eight words equal a*b.
After the last product diagonal, the final word also fits: the complete product
is below B^2. This final bound does not justify discarding an earlier high word.

The all-max input `a=b=B-1` demonstrates the width requirement. At the central
diagonal, the incoming carry is `3*W-4`, giving

`s_3 = 4*(W-1)^2 + (3*W-4) = 4*W^2-5*W`.

It exceeds a128-bit accumulator. Its outgoing carry is `4*W-5`, which itself
does not fit uint64. The exact eight-limb product is

`{1,0,0,0,W-2,W-1,W-1,W-1}`.

### Symmetric square: ten products, but 129-bit cross contributions

The identity

`a^2 = sum(a_i^2*W^(2i)) + 2*sum(a_i*a_j*W^(i+j), i<j)`

uses four diagonal products and six distinct cross-products: ten mathematical
64x64 multiplications, compared with16for a generic self-product. But a doubled
cross term can reach `2*(W-1)^2 < 2^129`, exceeding unsigned128.

It is safe to add the original 128-bit product twice into the three-word
accumulator, preserving carries each time. Alternatively, an explicit doubling
must retain the bit at position128 in the third word. Merely computing
`u128_product << 1` and forgetting that bit is incorrect. The expanded diagonal
sum is exactly the multiplication diagonal above, so the same `<2^130` bound
proves three words sufficient. Ten products do not imply fewer total additions,
instructions, spills or cycles; square speed is **PENDING IMPORT**.

## Arbitrary512 reduction: three folds and one correction

Write the input uniquely as `T=L+H*B`, where `0<=L,H<B`.
Because `B=K mod p`, replacing a high coefficient times B with that coefficient
times K preserves the residue. This proof covers the entire eight-word domain,
not only canonical field products.

### First fold: four high-word products and q<=K

The exact first fold is `U=L+H*K`. Its maximum satisfies

`U <= (B-1)+(B-1)*K = (K+1)*B-(K+1) < (K+1)*B`.

Hence `U=u+q*B`, with `0<=u<B` and `0<=q<=K`. The high value q is not generally
a bit and is much larger than P4's sum-only bound4095.

Serial word processing can compute, for i=0..3,

`v_i=H_i*K + L_i + c_i`, `u_i=v_i mod W`, `c_(i+1)=floor(v_i/W)`, `c_0=0`.

The induction `c_i<=K` gives

`v_i <= (W-1)*K + (W-1) + K = (K+1)*W-1 < 2^97`.

Each cell fits unsigned128, and q=c4 fits unsigned64. In the precomputed route,
first form the four independent values `P_i=H_i*K<2^97`, then combine them with
L and carries under the same inequality. These four products can be exposed
independently; the carry reconciliation still exists. Both schedules represent
the same exact U, with four source-level high-word products. Compiler lowering
of multiplication by constant K may differ from this source-level count.

### Second fold: q*K requires up to65bits

Compute `V=u+q*K`. Here

`q*K <= K^2 = W + 1954*2^32 + 977^2 < 2*W = 2^65`.

This coefficient does not universally fit uint64. Retain its low and high parts
in unsigned128 while propagating across all four result words. If implementation
starts the low-word carry with the entire q*K, then `u_0+q*K<3*W`: that first
word's outgoing carry may be2, not merely0/1. The following word's outgoing
carry is at most1. A two-word add schedule is also valid if it preserves the
same full65-bit coefficient and every subsequent carry.

Since `V < B+K^2 < 2*B`, the overall bit256 overflow e is0or1.
Write `V=v+e*B`, with `0<=v<B`. This overall single-bit bound must not be
misapplied to the internal first-word carry just described.

### Third fold: its outgoing high is provably zero

Compute `R=v+e*K`. If e=0, R=v<B. If e=1, the second fold crossed B, so

`v=u+q*K-B <= q*K-1 <= K^2-1`.

Therefore `R <= K^2+K-1 < 2^65 < B`. In both cases the third fold has no
bit256 overflow. This is a finite three-fold reduction, not an unbounded
repeat-until-small algorithm. Three is the total here: first H*K, second q*K,
third e*K; it is not P4's two-fold sum-specific scheme.

### Canonical correction

Now `0<=R<B<2*p`, so at most one subtraction of p is required. Equivalently
form R+K across all four words and retain its final carry bit: it is one exactly
when R>=p. Select the overflowing low result R-p in that case, otherwise R.
Unsigned mask selection must expand a bit to an all-ones mask when needed;
`K & carry_bit` is not multiplication by K or conditional addition of K.

The final raw limbs are the unique `[0,p)` residue of T. Composing any proven
exact full-product helper with either reducer therefore implements the complete
canonical multiplication/square contract. No lazy sum headroom, subtraction,
negation or inversion invariant is imported into this argument.

## Concrete edge fixtures and qualification requirements

These are **DERIVED** expected outcomes for independent native C++ tests,
not claims that any new implementation already passed them.

| Input or phase | Required exact outcome / exercised boundary |
|---|---|
| Raw reducer: L=H=B-1, all eight words max | First U=(K+1)B-(K+1), q=K, u=p-1; qK=K^2>W. Second-fold e=1; final residue K^2-1. The full-qK first-word addition carries2. |
| Raw reducer: H=0, L=p-1 / p / p+1 | Canonical outputs p-1 /0/1; detects missing or excessive final correction. |
| Raw reducer: H=1, L=0 | B reduces to K; a lost high half cannot pass. |
| Raw reducer: H=0, L=B-1 | B-1 reduces to K-1. |
| Raw reducer: T=p^2-1 / p^2 / p^2+1 | Canonical outputs p-1 /0/1, within the arbitrary512 domain. |
| Full product/square: a=b=B-1 | Exact eight limbs {1,0,0,0,W-2,W-1,W-1,W-1}; stresses130-bit central accumulation and129-bit doubled cross products. |
| Full products: selected a=W^i, b=W^j | Single one in output limb i+j; use all i,j in0..3 to expose dropped/permuted diagonals. |
| Canonical complete pipeline: a=b=p-1 | Final output1; the raw product must still be checked independently. |
| Canonical zero/one and same-object operands | Zero products, multiplication identity, square/self-product equality, and unchanged inputs. |

Retain the two exact P1 regression KATs from the manager-supplied corrected
reference qualification as additional fixtures. Do not reconstruct a guessed
KAT from a description or substitute a newly chosen vector under its old name.
Include deterministic random arbitrary raw operands, canonical operands,
high-only/low-only512values, limb-boundary neighbors and carry cascades.

Compare products as unbounded integer multiplication, reducers against ordinary
integer modulo p, and composed pipelines independently at raw limbs and bytes.
Bounded phase witnesses should expose actual carry/high states where helpers
make them observable; final equality alone does not show which internal width
was exercised. Negative controls must show that corrupted high/low words and
output bytes are detected. Count fixtures separately from repeated compiler
runs, assertions and overlapping intermediate checks. The proof is a written
integer argument, not machine-checked verification of an implementation.

## Benchmark observations and comparisons

The matrix has20cells: four multiply routes, four square routes and two raw
reducer routes, each in chain1 and ILP4. Routes are those fixed in the protocol:
actual API; row+serial; Comba+serial; Comba+precomputed for multiplication;
actual API; row self-product+serial; symmetric Comba+serial; symmetric
Comba+precomputed for square; and the two reducers alone for raw reduction.

Each job initializes four deterministic canonical nonzero seeds. There are
1024steps per active lane and either one or four active lanes. For step t,
multiplication uses the same shared canonical nonzero RHS at `t & 255` in each
active lane; square evolves `x -> x^2` without an RHS. Raw reduction uses the
shared arbitrary512corpus entry at the same index: its low words are the current
canonical x XOR corpus-low, and its high words are corpus-high. This input can
be any512-bit value; it is not asserted to be a product. Its result becomes x.

Lane states do not feed one another. Lane0 must reproduce the chain1 result
when seeds and recurrence match. The complete job output is128bytes: each active
lane contributes canonical32BE, and chain1 includes96zero inactive bytes.
Every output byte stays observable. Those output/reset costs are inside all
jobs, while allocation, corpus generation and prevalidation are outside.
No per-step benchmark barrier is added. Every region's operation count is

`actual_jobs * 1024 * active_lanes`.

Raw-reducer steps have their own meaning: low-half XOR construction plus
reduction, not field multiplication. Square performs no corpus RHS loads.
Only like-contract comparisons support the corresponding primitive conclusion.
End-of-region checks inspect the last identical job; separate independent C++
traces must check bounded recurrences at every step and preserve all inputs.

Every field step is canonical; benchmark final-byte materialization only once
per job does not authorize a lazy sequence changing the per-operation contract.
Returning full products or reducing arbitrary512inputs is also distinct from
P3/P4 final-only vector summation. Existing sum ratios are not predictors here.

Primary field comparisons use actual API/candidate, three candidates for each
of multiplication and square in each mode. Raw reduction uses serial/precomputed.
Secondary comparisons are row+serial/Comba+serial and Comba+serial/Comba+precomputed
for each operation/mode. All A/B ratios above one favor B. Chain/ILP normalized
costs describe complete jobs; they do not by themselves isolate port throughput,
one instruction's latency or a purely causal effect of one schedule change.

Row versus Comba can change carry placement, temporary arrays, optimizer
visibility, emitted calls and register/stack usage. The square comparison also
changes distinct multiplication count and doubled-cross handling. Serial versus
precomputed reducers can change scheduling and live state even with equal
mathematical high-word product counts. Complete program comparisons must retain
these qualifications; a faster raw reducer does not prove a faster multiplication.

Two native GCC series use the protocol's same corpus and reversed second
order, two warmups/eight measured rounds and at least200ms per later region.
Two consecutive same-count calibration regions qualify the initial batch.
Short later prefixes remain inside one continuously timed region as more jobs
are appended; cumulative probes and actual total operations are retained.
Caps must fail explicitly, not discard samples or call a short region qualified.
Smoke/synthetic-clock tests and historical P2/P3/P4 timings are never pooled.

One core deliberately measures dependency chains and independent-state
scheduling without thread-dispatch costs. Four arithmetic states are not four
threads, not an automatic SIMD claim and not the multicore optimum. Canonical
mathematics and public schedules do not certify constant-time compiled behavior.
Production acceptance, security qualification, novelty and whole-engine gains
are outside this document.

## Source and authority boundary

The manager supplied exact targets under verified repository
`repo_666797171f0141c58bf05f579b2ee16e` and session
`01a06be6-2904-7c62-9e7d-1245c34a5312`. Worker-role Source Graph tools remain
unexposed under the recorded NeedFix; no manager-role tool was impersonated.
Owner Task MCP suspension remains active. Only this contract and its companion
lens-plan document are written by this subtask; no code/build/timing is changed.

Aligned P5 protocol SHA256:
`10c8e13c09bddd103689e0ba10d50676672cc37b659df53def7c9c6e7fc82092`.
The manager's explicit handoff additionally fixes shared `step&255` indexing and
the128-byte whole-job output with inactive chain bytes zeroed. The P4 contract
and measured lens identities are retained unchanged; none of their sum bounds
is a substitute for the product/reducer arguments above. Stop at manager review.
