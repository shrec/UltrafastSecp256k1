# P6 field boundaries: unchanged arithmetic, explicit call contracts

Status: independent premeasurement contract, submitted for manager review.
P6 qualification, emitted-code observations and timings are **PENDING IMPORT**.
This wave changes experimental API/call boundaries only. It introduces no new
product, square, reducer, production route, scalar or upper-layer algorithm.
The [P6 protocol](P6_FIELD_BOUNDARY_PROTOCOL.md) fixes execution policy; the
[25-lens plan](P6_FIELD_BOUNDARY_LENSES.md) fixes observation criteria.

## Value contract and reused proof

Let `W=2^64`, `B=W^4=2^256`, `K=2^32+977` and `p=B-K`.
A canonical field value consists of four unsigned64 words, least significant
first, decoding to an integer in `[0,p)`. Multiplication accepts canonical a,b
and returns the canonical residue `a*b mod p`; square accepts canonical a and
returns `a^2 mod p`. Every primitive step, not just a job's final serialization,
must satisfy this raw four-word canonicality contract.

The row routes reuse the unchanged P5 exact row product and serial reducer.
The [P5 bound contract](P5_FIELD_PRODUCT_CONTRACT.md), SHA256
`1886cc57ff5ad813546c649ebf25af57c0fe02270a673fc1acbca8689bc4607a`,
already proves the relevant composition:

- Each row cell `a_i*b_j + existing_word + carry <= W^2-1` fits unsigned128;
  the eight result words decode to the complete ordinary-integer product.
- Arbitrary512 reduction preserves the residue through the `H*K`, `q*K`
  and `e*K` folds. Its full-width `q*K < 2^65` is retained; the third fold
  has no bit256 overflow, and one final p correction makes the result canonical.
- Restricting that composition to canonical a,b gives the field contract above;
  using the same input twice gives square without a new arithmetic identity.

These are inherited written integer arguments, not newly measured results or
machine-checked implementation proofs. P6 must bind the actual P5 kernel bytes
and independently qualify its new boundaries. The existing ASM/API paths also
require ordinary-integer checks; the row proof does not prove unrelated ASM
instructions correct. No P3/P4 lazy-sum bound, Comba change, raw-reducer timing
route, inversion or security assumption is imported.

## Four experimental route families

Each family has multiplication and square, measured in chain1 and ILP4:

| Route family | Required implementation boundary | What is not assumed equal |
|---|---|---|
| Actual FE64 API | Existing corrected `FieldElement` multiplication/square, including its real dispatcher and selected arithmetic path. | Object construction, return handling, calls and optimizer visibility need not match raw-word routes. |
| Direct existing ASM | Call the existing `field_mul_full_asm(a,b,out)` / `field_sqr_full_asm(a,out)` symbols after native capability admission. No new ASM arithmetic. | Removing the public API/dispatcher changes caller liveness and materialization too; it is not a measured constant call-cycle subtraction. |
| Row inline | Inline the unchanged P5 `mul_row` plus `reduce_serial`; square uses the row self-product. | The compiler may expose, reuse or schedule arithmetic differently in the complete caller. |
| Row outlined | Call a separately compiled C-linkage pointer-ABI leaf implementing that same P5 row composition, with full LTO disabled. | Outlining can change register allocation, constants, caller/callee state and optimization, not only add a call instruction. |

The new declaration/definition targets are
`probes/p6_field_boundary.hpp` and `probes/p6_field_boundary.cpp`.
The public experiment wrappers in namespace `pa_p6` return `Limbs` by value:
`row_mul_inline`, `row_mul_outlined`, `asm_mul_direct`, and their square
counterparts. The actual API route is retained separately by the driver.

The row C leaf signatures are:

```cpp
extern "C" void pa_p6_row_mul(
    const uint64_t* a, const uint64_t* b, uint64_t* out) noexcept;
extern "C" void pa_p6_row_square(
    const uint64_t* a, uint64_t* out) noexcept;
```

Only the separate `.cpp` defines these leaves. Existing ASM uses the same
input-first/output-last four-word pointer layout, with square's single input.
The selected platform is native x86-64 Linux with its C ABI; matching signatures
do not promise identical prologues, stack use, symbol sizes or instruction flow.
The compilation/link manifest must demonstrate a real separate translation
unit, `-fno-lto`, and noinline/noipa on GCC or noinline on Clang as specified
by the protocol. Emitted-code review must demonstrate that the intended
outlined calls survive and the direct route reaches the existing ASM symbols.

## Memory and support preconditions

Each leaf input pointer denotes a valid readable range of four suitably aligned
`uint64_t` objects holding a canonical value. The output denotes a valid writable
four-word range disjoint from every input range. All inputs remain byte- and
word-identical after the call. Multiplication's read-only inputs may alias;
in particular `a==b` is admitted. There is no `restrict` promise. Cross-type
aliasing, invalid C++ objects, null pointers and output/input overlap are outside
this contract. This is not a pointer/length sum API: there is no N=0/null identity
case, runtime null rejection promise or newly enlarged in-place ASM guarantee.

The boundary preservation argument is separate from integer reduction: a leaf
reads canonical words, performs the proven composition in private state, and
writes only its disjoint output range. An input alias does not change its value
because both aliases are read-only. Returning wrappers allocate a private result
and satisfy the disjoint-output precondition before replacing the caller's lane
state. Thus in the recurrence a new canonical state replaces the old state only
after its use as input has finished. Actual code and tests must establish these
read/write facts; a const declaration alone cannot certify an external ASM body.

Direct ASM admission requires the compiled native x86-64 Linux GCC/Clang ASM
path and both BMI2 and ADX support. The caller must successfully use
`native_asm_available()` before every possible native direct job or trace,
outside all such jobs/traces. There is deliberately no per-operation
support check in the admitted native wrapper. Unsupported machines must not run
or label an alternative path as this direct-ASM benchmark. Compile-disabled /
NO_ASM direct wrappers must fail explicitly without referencing ASM symbols;
they must not silently fall back to row or portable FE64 under the same name.
Test the fail-closed wrapper only when direct ASM is compile-disabled; do not
invoke an unguarded native wrapper on an unsupported host. No malformed-pointer
call is needed to test these preconditions.

## Fixed recurrence and observation contract

There are 16 cells: four route families x two operations x two dependency modes.
Every active lane starts from the same respective P5 canonical nonzero seed.
Each job performs exactly 1024 steps per active lane, using one lane for chain1
and four independent states for ILP4. For multiplication, step t uses the same
canonical nonzero RHS at `t & 255` in every active lane. Square is `x -> x^2`
without RHS mixing or RHS reads in the arithmetic recurrence. Lane states never
feed one another; lane0 must agree between chain1 and ILP4.

The corpus recipe is the unchanged P5 seed20260905 recipe: four seeds,
256 canonical RHS values and 256 raw512 entries, FNV64 `7c2e57392893ac80`.
The raw512 entries are retained and checked for source-corpus identity but are
unused by every P6 arithmetic route. They do not become a timed input stream or
an additional raw-reducer cell. Generation, allocation and prevalidation are
outside measured regions; their exclusion must remain explicit.

Each job initializes its active states and materializes exactly 128 output
bytes. Each active lane contributes all32 canonical big-endian bytes; chain1's
remaining96 bytes are zero. These reset/output costs are timed for all families,
although equal logical work need not compile to identical machine instructions.
No per-step benchmark barrier or deliberately added equalization work is allowed.
Whole-job noinline/noipa and a complete-output barrier keep invocation and all
output bytes observable; emitted repeated calls must be verified. The exact count is

`operations = actual_jobs * 1024 * active_lanes`.

Raw output words must be checked before any serializer that could normalize a
copy. Separate untimed C++ traces check each canonical recurrence state and all
bytes, then compare the complete compiled evaluator/job outputs. A region checks
its last identical job's entire128-byte output outside the clock; it does not
claim per-operation instrumentation inside the timed loop. The complete corpus
must be regenerated and compared word-for-word, including unused raw512 entries,
before and after measurement; hashes supplement rather than replace those checks.
No zero/one absorbing trace state should silently turn the intended workload
into trivial arithmetic. Any such observation is retained and resolved before
claiming the intended nontrivial recurrence was measured.

One core deliberately measures dependency scheduling and independent-state
overlap without introducing thread-launch or multicore coordination. ILP4 means
four scalar states, not four threads or necessarily packed SIMD. It is not a
production multicore design decision or a pure hardware-throughput measurement.

## Qualification and measurement gates

Native GCC and independently compiled Clang configurations must qualify all
four routes against ordinary Boost integer multiplication/modulo, with corrected
FE64 an additional reference. A portable NO_ASM ASan+UBSan configuration qualifies
the three applicable families and explicitly reports direct ASM as skipped.
Do not count the portable skip as a fourth-path pass, sanitizer instrumentation
of ASM, or repeated compiler runs as independent corpora. Test canonical zero,
one, p-1, limb boundaries/carry cascades, deterministic random operands, same-input
aliases, unchanged input words, raw canonicality, all32 bytes, full job output and
lane0 consistency. Retain both exact P1 carry KATs and the older large-square
KAT by identity. Guarded output buffers must detect writes outside four words.
Negative observer controls must detect corrupted limbs/bytes; malformed output
overlap and unsupported-machine arithmetic are not admitted test inputs.

Freeze source, corrected library/ASM object, compiler/flags, separate-TU object
and executable identities before the two GCC full series. P5 or other historical
timings, smokes, sanitizer runs and synthetic clocks are never pooled into P6.
Use the P5 continuous-region duration mechanism: two consecutive calibration
regions at the same initial job count, two warmups and eight measured rounds,
with reversed second-series scheduling and a 200ms floor for later regions.
Retain every calibration sample and every cumulative extension probe. A short
later prefix is extended under the same start clock; report actual total jobs
and operations, not only the initial batch. A cap or clock failure must remain
visible and fail closed; no short region is dropped or mislabeled qualified.

The fixed comparisons per series are:

- 12 primary: actual API / each of three candidates, for both operations/modes.
- 12 secondary: direct ASM / row inline, direct ASM / row outlined, and
  row inline / row outlined, for both operations/modes.
- Eight mode comparisons: chain1 / ILP4 normalized ns/operation for every
  operation/route family.

For A/B, a ratio above one favors B. Pair samples by the same round; publish all
eight ratios and wins/ties/losses, plus their min/median/max and the underlying
region-mean ns/operation values. The median of paired ratios is not generally
the ratio of the independently computed medians. The two series have 64 total
comparison summaries and 512 ratios; these overlap in cells and are not 512
independent experiments. Keep primary, secondary and mode interpretations apart.

## Attribution and acceptance boundary

API/direct ASM controls the public boundary around the existing arithmetic but
also affects visible types, materialization and liveness. Inline/outlined row
holds the source arithmetic composition fixed but changes optimization scope
and calling convention effects together. Direct ASM/outlined row matches the
pointer ABI, not arithmetic instructions, prologues or caller/callee allocation.
Direct ASM/inline row compares complete implementations with both differences.
None provides a universal number of cycles attributable solely to one call.

P5's conditional row-chain win motivates P6; it does not predict a P6 winner.
The compiler already recognized row-square symmetry in P5, and normalized
serial/precomputed reducers largely coincided. P6 introduces neither a fresh
ten-versus-sixteen operation-count claim nor a parallel-reducer discovery.
Accept observations only within their measured operation/mode/build boundary;
retain all negative results and do not infer an unmeasured dispatch threshold.
CT certification, memory-bandwidth/spill causality, novelty, a world record,
whole-engine speedup and production integration remain outside this wave.

## Provenance

Verified repo_id `repo_666797171f0141c58bf05f579b2ee16e`; manager session
`01a06be6-2904-7c62-9e7d-1245c34a5312`. The manager supplied the exact P5
documents and existing `field.cpp:1117-1215` ABI/dispatcher range. Worker-role
Source Graph, session, AI Memory and KB tools remain unexposed under the recorded
NeedFix. Exact-target handoff reads do not impersonate manager-role tools or
write context databases. Owner Task MCP suspension remains active.
This subtask writes only this contract and its companion lens plan.
Aligned P6 protocol SHA256:
`68ee687cf8d4b8b4dccacea9d82eb4a4c35d7b315f814c723bb8d5d1dff16113`.
Stop at manager review before any outcome or integration claim.
