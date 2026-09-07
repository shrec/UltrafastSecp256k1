# P2: field-p research through 25 observation lenses

Status: first canonical P2 comparison measured and reviewed; see
[P2 results](P2_FIELD_RESULTS.md). Remaining field candidates are not complete.
The owner's current instruction defers performance integration: finish the
field-p investigation and collect its positive and negative results first.
No production optimization, scalar change or upper-layer change is authorized
by this document. The already reviewed P1 correctness repairs remain the baseline.

Evidence labels used below:

- **MEASURED (P1):** imported from the named local P1 evidence, not a new P2 run.
- **DERIVED:** source inspection or an explicit mathematical consequence under
  stated preconditions; not a measured performance result.
- **HYPOTHESIS:** a candidate explanation or experiment, not an established gain.
- Unknown quantitative evidence is written exactly **PENDING IMPORT**.

This is not an admitted ParseAtlas case-study record. The local
[Atlas guidance](../../research/ParseAtlas/case_studies/modular_arithmetic_guidance.md)
requires an explicit decode invariant and whole-region costs, with dependency
depth, live state and feasibility kept separate. Its 25 notation-precedence
conventions do not literally transfer to C++ or assembly. These are 25
engineering observation lenses, not 25 asserted discoveries or wins.

## Evidence already available

[P1 results](P1_PRIMITIVE_RESULTS.md) and
[P1 roadmap](P1_PRIMITIVE_ROADMAP.md) establish the corrected fast FE64 baseline.
The roadmap's eventual integration step is deferred by the newer owner decision.
The independent integer oracle found two old reducer defects before timing;
the production implementation alone therefore is not a sufficient oracle.

**MEASURED (P1):** median normalized region means, ns/op, series 1 / series 2:

| Field operation | Dependent chain1 | Independent ILP4 on one core |
|---|---:|---:|
| Add | 9.54 / 9.47 | 5.77 / 5.82 |
| Subtract | 13.40 / 13.74 | 5.92 / 5.82 |
| Multiply | 23.15 / 22.80 | 11.76 / 11.82 |
| Square | 20.77 / 20.75 | 11.06 / 11.27 |
| Inverse plus dependent lookup | 932.92 / 918.16 | 896.65 / 900.46 |

These are corrected-library, GCC 14.2.0, CMake Release, native ASM, LTO-OFF
workloads on CPU4 of the i5-14400F. They include API/dispatch and harness costs;
inverse includes lookup. The machine used powersave with turbo enabled.
They are not instruction latencies, default-LTO numbers, FE52/CT measurements,
or a basis for comparing an unpaired future run. P2 must remeasure its FE64
control in the same process and profile as the other routes.

P1 finite correctness evidence: the same deterministic corpus passed GCC ASM,
Clang ASM, NO_ASM sanitizers and NO_ASM/NO_INT128 configurations. Each checked
258,125 field outputs, including canonical raw limbs and all 32 big-endian bytes.
This is neither a universal proof nor a side-channel certification. Primary
evidence is linked in [P1 validation](data/p1_validation_20260905.json) and the
P1 result report; do not pool P1 timings into P2 statistics.

## Contracts before costs

Let `p = 2^256 - K`, `K = 2^32 + 977`, `b = 2^52` and `q = 2^48`.

**DERIVED:** the FE64 decode is `sum(L[i]*2^(64*i)) mod p`. The FE52 decode
is `sum(n[i]*2^(52*i)) mod p`. Equivalence means equal decoded field values,
not equal intermediate arrays. At a canonical observation boundary, additionally
require `0 <= value < p`, FE64's four exact limbs and all 32 BE bytes. For a
canonical FE52 state require `n[0..3] < b`, `n[4] < q`, and its decoded integer
before modular reduction to be below p. Limb-width checks alone are insufficient.

| Route to investigate | Arithmetic and observation contract | Cost boundary |
|---|---|---|
| Fast FE64 | Existing canonical public add/sub/mul/square; nonzero variable-time inverse | Existing ABI, native dispatch, assignments and final bytes |
| Per-operation FE52 bridge | Canonical FE64 inputs -> FE52 operation -> full canonical FE64 result | Both input conversions where applicable and full output normalization/packing on every operation |
| Canonical resident FE52 | State remains FE52; every step explicitly full-normalizes; equal decoded operation sequence | Per-job seed conversion and final decode; separately disclose preconverted RHS setup and resident storage |
| Existing CT API | Exact `ct::field_*` APIs and their security contract | Keep their conversion, compiler-barrier and fixed-schedule costs; not an interchangeable VT candidate |

For the planned 256-value RHS corpus, FE64 storage is 8,192 bytes and FE52
storage is 10,240 bytes (**DERIVED**, 256 times 32/40 bytes). This is not measured
cache traffic or total working set. Preconverting resident RHS values outside
jobs is a distinct setup contract: its gain cannot be presented as a per-operation
drop-in gain. Conversion/setup break-even: **PENDING IMPORT**.

## Actual FE52 implementation and safe first-wave bounds

Source authority is the implementation, not optimistic or stale header comments:
[FE52 API](../../src/cpu/include/secp256k1/field_52.hpp),
[inline kernels and normalization](../../src/cpu/include/secp256k1/field_52_impl.hpp),
[FE52 out-of-line code](../../src/cpu/src/field_52.cpp),
[FE64 and variable-time SafeGCD](../../src/cpu/src/field.cpp),
[CT field implementation](../../src/cpu/src/ct_field.cpp).

**DERIVED from the inspected code:**

1. `from_fe` (inline header 2744-2754) rearranges bits; it does not normalize.
   Its first-wave input is a canonical FE64 value.
2. `operator+` (2487 onward) performs five independent word additions without
   carry propagation. With two canonical FE52 inputs, each lower limb is at
   most `2*(b-1)` and the top at most `2*(q-1)`.
3. There is no direct FE52 subtraction operator in the API. For canonical
   inputs use `a + b.negate(1)` followed by full normalization. `negate(1)`
   computes each component as `2*P[i]-b[i]` (2517 onward); each result is
   nonnegative for this bounded input. The sum is bounded componentwise by
   `3*(b-1)` / `3*(q-1)`, comfortably within 64 bits. Its decoded value is
   `a-b` modulo p. This bound does not license arbitrary magnitude arguments.
4. `fe52_mul_inner` and `fe52_sqr_inner` write masked lower limbs but finish
   with `r[4] = carry + t4`; the public wrappers do not subsequently call
   full normalization. A header's "normalized output" wording is therefore
   not a sufficient canonical-output guarantee. First-wave inputs are fully
   canonical, and the resident result must explicitly call `normalize()`
   before the next step. Inputs with larger magnitude are outside this wave.
5. For canonical inputs, each doubled squaring operand is below `2^53`, so
   its word doubling does not overflow. The product and reduction accumulator
   bounds must be checked for the selected compiled kernel, including the
   folded high columns; a symbolic product count alone is not that proof.
   Maximum admitted noncanonical kernel-input magnitude: **PENDING IMPORT**.
6. `normalize_weak` (2410-2430) propagates, folds and propagates again, but does
   not select a representative below p; the top limb can retain bit 48.
   It is not the full canonical boundary used in this wave.
7. Full `normalize` (2636-2675) performs a fold plus conditional final
   reduction. For canonical add/sub intermediates above, lower limbs are below
   `3*b`, the top below `3*q`, and the initial top overflow coefficient is at
   most 2. The additions of this coefficient times K and subsequent carries
   stay within 64-bit words. Arbitrary 64-bit limb arrays are not admitted.
8. `to_fe` (2758-2769) full-normalizes a **copy** and then packs it. It does
   not canonicalize the resident source object. Tests must inspect resident
   raw limbs before decoding, or an omitted step normalization can be hidden.
9. Raw kernels have a documented restrict/no-overlap contract (245-268);
   the public in-place wrappers compute a temporary before assignment
   (2438-2485). Preserve the raw destination/input separation and test the
   supported public alias contracts separately.

Do not promote the header's approximate "4096 additions" headroom statement
into a proven normalization-input limit. Full normalization adds
`(n[4] >> 48)*K` to the 64-bit low limb before propagation, so headroom for
that fold matters in addition to headroom for preceding plain additions.
A universal maximum safe lazy-region bound and its boundary tests are
**PENDING IMPORT**. No lazy-region experiment is admitted by this first wave.

### Separate canonical-sum boundary finding

The statement above concerns arbitrary lazy regions/magnitudes. A narrower
sum-only boundary has now been derived and reproduced in native C++ with GCC
and Clang; it does not change the per-step-canonical performance contract.
Full evidence: [P2 lazy NeedFix](data/p2_lazy_needfix_20260905.json).

For `N` fully canonical FE52 summands, including any arbitrary canonical initial
accumulator, direct full normalization is universally safe for `N <= 4095`.
The existing `normalize_weak(); normalize();` ordering admits `N <= 4096`.
These are sum-only bounds, not bounds for arbitrary noncanonical magnitudes,
subtraction/negation outputs or multiplication inputs.

**MEASURED (P2 diagnostic):** use the canonical term
`{2^52-1, 0, 0, 0, 2^48-1}`, repeatedly added to zero:

| Terms | Raw sum still exact? | Direct full normalize | Weak then full |
|---:|---|---|---|
| 4095 | Yes | Correct | Correct |
| 4096 | Yes | Incorrect, loses exactly `2^64` | Correct |
| 4097 | No: addition already wraps | Incorrect | Incorrect |

At 4096, `n[0]=2^64-4096` and the top-fold coefficient is 4095. Adding
`4095*K` to that 64-bit low word wraps before its 52-bit carry is extracted.
The correct next limb is 4096; the observed next limb is zero. This is loss of
`2^64`, not loss of `2^256`. Carry-first normalization avoids this particular
overflow. Both compilers match independent Boost residues and the unchanged
FE64 reference on the expected result, and reproduce the same failure.

**DERIVED sufficient-bound argument:** with `b=2^52`, `q=2^48`, `W=2^64`,
`N<=4095` gives initial `x<=N-1` and
`t0<=N*(b-1)+(N-1)*K<N*b+b<=W`. Its carry is at most N; higher additions are
at most `N*b<W`. The first-pass value is below `2^256+N*b^4<2p`, so the existing
final correction is sufficient. Weak-first instead propagates before folding;
its initial sums are at most `N*b-1<=W-1` even at N=4096, and the fold then
starts from a 52-bit low word. An arbitrary nonzero `x0` counts toward N.

Disposition: retain as **NeedFix** for the documented headroom/normalization
contract and as a candidate ordering constraint for later experiments. No
production modification, caller-reachability/exploit claim or timing win is
inferred from this diagnostic.

### Inversion route and security distinctions

**DERIVED:** `FieldElement::inverse` uses variable-time signed-62 SafeGCD and
throws on zero on this host. `FieldElement52::inverse_safegcd` (inline header
2914-2927) returns zero on zero and otherwise normalizes then calls
`fe52_inverse_safegcd_var`. The native helper (`field.cpp` 3183-3232) converts
**directly** between 5x52 and signed-62; the old FE64-round-trip heading and
API comment are stale. Do not charge a conversion that the code no longer does.

`FieldElement52::inverse()` is a different Fermat-chain method. With
`SECP256K1_HYBRID_4X64_ACTIVE`, `field_52.cpp` packs once, runs the chain through
direct FE64 ASM and unpacks once. Calling it "resident FE52 inversion" would
misidentify the executed representation. Fixed source operation count alone
does not certify constant-time behavior of the whole compiled path.

`ct::field_inv` (`ct_field.cpp` 661-732) instead uses a fixed 10-by-59-divstep
schedule on the native-int128 route, and 25-by-30 on its fallback. Keep it
separate from both VT inverses. CT add/sub delegate to FE64; native CT mul/sqr
bridge through FE52 with compiler barriers (`ct_field.cpp` 155-209). A custom
bridge without those barriers is not the same CT API measurement.

## All 25 lenses: current evidence and next gates

| Lens | Current evidence and status | Required next observation / gate |
|---|---|---|
| V01 Reachability | **DERIVED:** FE64, resident FE52, per-op bridge and CT are different compiled routes; two FE52 inline-ASM alternatives are disabled by `#elif 0`. | P2 records actual macros: native ASM/fast reduction and inline FE52 enabled, LTO OFF. Emitted dependency graph remains **PENDING IMPORT**. |
| V02 Corpus | **MEASURED (P1/P2):** modulus/bit/carry boundaries, zeros and aliases; P2 has16200 binary cases and3030 nonzero inverse inputs per route. | Timing uses the same256 RHS values plus4 seeds across routes and series. Broader distributions and working-set sweep remain **PENDING IMPORT**. |
| V03 Equivalence | **MEASURED (P2):** three configurations pass the same2769460-assertion corpus, zero mismatches; raw bridge residues and all32 output bytes checked. | Preserve independent Boost/Euclidean oracle, not just normalizing equality; no universal proof claim. |
| V04 Definedness | **DERIVED:** FE64 inverse throws on zero; FE52 SafeGCD returns zero; raw limb bounds and restrict matter. | Nonzero common timing domain; separate zero/error tests; sanitizer builds and no illegal raw aliases. |
| V05 Range / overflow | **DERIVED:** canonical-input add/sub bounds above; P1 five-limb reducer bound retained. | Trace every mul/square folded accumulator and normalization precondition. General lazy magnitude ceiling: **PENDING IMPORT**. |
| V06 Aliasing | **MEASURED (P2):** supported public self-alias/in-place and input-preservation checks pass across three configurations. | Raw destination/input separation is preserved; no arbitrary raw-kernel alias contract is added. |
| V07 Observable output | **DERIVED:** width-bounded FE52 is not necessarily canonical; to_fe normalizes a copy. | Raw resident canonical check before conversion, then all 32 BE bytes. Do not use normalizing equality alone. |
| V08 Invariants | **DERIVED:** `2^256 = K (mod p)` and `2^260 = 16K (mod p)` explain folds. | Write down decode preservation at every changed boundary; a lost carry cannot be repaired by later canonicalization. |
| V09 Constant time | **DERIVED:** VT and fixed-schedule inverse contracts differ; CT wrappers add barriers. | Separate security classes, inspect compiled branches/addresses and run appropriate leakage checks before any security claim. P2 CT certification: **PENDING IMPORT**. |
| V10 Resources | **DERIVED:** FE64/FE52 objects occupy 32/40 bytes; resident RHS storage differs. | Record scratch, stack, allocation, resident corpus and setup. Peak live memory: **PENDING IMPORT**. |
| V11 Property visibility | **MEASURED (P2):** canonical resident FE52 add/sub wins64/64 pairs; per-op bridge add/sub/mul/square loses128/128. | Whole representation/ABI boundaries matter; isolating the cause of each gain remains **PENDING IMPORT**. |
| V12 Dependency depth | **MEASURED (P1):** chain1 and ILP4 workloads differ in observed time. | Build an actual machine dependency graph for product, fold, normalize and conversion; do not call chain length an observed critical path. Depth: **PENDING IMPORT**. |
| V13 Live state | **HYPOTHESIS:** extra limbs or independent states may trade latency for register pressure/spills. | Inspect simultaneous live values, stack accesses and spills for each inlining/call shape. Counts: **PENDING IMPORT**. |
| V14 Operations | **DERIVED:** a symmetric 5-limb square has 15 unique mathematical limb products versus 25 in general multiplication, before reduction. | Count emitted multiplies, carries, shifts, normalization and bridge work; fewer products is not a speed result. Instruction counts: **PENDING IMPORT**. |
| V15 Conversion | **MEASURED (P2):** full bridge and amortized resident region are separately priced in the result tables; no stable bridge advantage. | Isolated conversion cost and setup-inclusive break-even remain **PENDING IMPORT**; a single setup timing is descriptive only. |
| V16 Latency | **MEASURED (P2):** resident chain add1.17–1.19x, sub1.43–1.45x, multiply about1.08x (13/16 wins), square about1.06x; inverse no stable gain. | Same-round ratios, not individual instruction latency. Preserve modest/mixed signals and inverse-lookup caveat. |
| V17 Throughput | **MEASURED (P2):** resident ILP4 add1.38–1.40x and sub1.32–1.34x; FE64 wins all32 ILP4 multiply/square pairs. | Four independent states share one core; not a multicore or upper-layer gain. |
| V18 Instructions | **MEASURED (P1):** corrected native disassembly has three added NEG instructions. | Attribute P2 claims to actual emitted instructions; PMU/cycles, if unavailable, stay **PENDING IMPORT**. |
| V19 Cache / bandwidth | **DERIVED:** resident FE52 RHS bytes exceed FE64 RHS bytes; this alone says nothing about bandwidth pressure. | Vary public working-set sizes in a later controlled experiment and measure misses/traffic before causal claims. P2 counters: **PENDING IMPORT**. |
| V20 Code size | **MEASURED (P2):** linked symbol sizes retained in data/p2_symbol_sizes_20260905.txt; multiply bridge1345B, resident multiply jobs1653/2519B, FE64 jobs198/365B plus callees. | These are symbol sizes, not complete hot paths. Total dynamic hot-path footprint and instruction-cache impact remain **PENDING IMPORT**. |
| V21 Portability | **MEASURED (P2):** GCC actual library, Clang-compiled field/FE52/CT, and GCC NO_ASM ASan/UBSan pass the same corpus. | Other architectures, default-LTO behavior and compiler performance replication remain **PENDING IMPORT**. |
| V22 Compiler | **MEASURED/RECORDED (P2):** GCC14 native LTO-OFF performance; GCC/Clang smoke; exact commands, macros and hashes in data/p2_build_20260905.json. | Compiler-policy A/B and CT leakage qualification remain separate work, not implied by correctness or a macro. |
| V23 Policy / ties | **HYPOTHESIS:** different public workload shapes may favor different representations. | No secret-dependent dispatch, hidden reruns or forced universal winner; retain ties/losses. Any dispatch threshold: **PENDING IMPORT**. |
| V24 Transfer | **DERIVED:** a primitive gain cannot by itself establish an upper-layer gain. | Upper layers and performance integration remain deferred; collect the complete field evidence first. Transfer gain: **PENDING IMPORT**. |
| V25 Prior art | **DERIVED from repository provenance:** existing FE52 kernels identify their upstream adaptation; lazy reduction and alternative encodings are not newly invented here. | Verify primary prior-art sources before any novelty claim. Novelty classification: **PENDING IMPORT**; no world-record claim. |

## Candidate queue after the first canonical gate

Each row is a **HYPOTHESIS**, not an implementation approval or speed claim.

| Candidate family | Why examine it | Admission gate |
|---|---|---|
| FE64 versus full FE52 bridge | Different limb width, scheduling and reduction shape for the same canonical primitive | Same input/output contract and per-operation conversion paid; paired correctness and timing |
| Canonical resident FE52 | Removing repeated representation crossings may amortize setup | Label region/setup contract; count full per-step normalization and final decode |
| Deferred-normalization field region | Extra local additions may remove carry/normalization dependencies | Explicit permitted observations, magnitude invariant, fold headroom, normalization schedule and full-region oracle; currently **PENDING IMPORT** |
| Product / symmetric-square / reducer scheduling | Independent columns and sparse-p folds may shift the depth/live-state trade | Bound full accumulators first; inspect emitted code; count reduction and canonicalization, not only products |
| CT versus VT inversion scheduling | Fixed schedule, early termination and direct encoding conversions solve distinct contracts | Compare within each security class; preserve zero semantics; do not retry unchanged historically rejected guards |
| Public batch inversion | One inversion plus products may amortize inverse cost across independent field inputs | A separate vector contract: size, zeros, aliasing, scratch, input preservation, canonical per-output bytes and setup-inclusive break-even. No implementation in this wave |

The public batch row is not a replacement for one inverse and is not a point
or signature experiment. Its multiplication count, scratch bound, zero policy,
timing and crossover are **PENDING IMPORT** until the exact algorithm/contract
is selected. Do not reuse scalar aggregate F1/F2 results as field-inversion evidence.

## Handoff and provenance

Allowed write for this work: this new document only. No production, P1/F2,
scalar, build or CI edits. No benchmark or expensive command was run by this
inventory worker. Bounds/routing findings were sent to the manager and separate
driver/test workers before their implementation gate.

Repository: `repo_666797171f0141c58bf05f579b2ee16e`; verified manager session
provided for this wave: `01a06be6-2904-7c62-9e7d-1245c34a5312`.
Task MCP remains suspended by the owner. Worker-specific Source Graph tools
are absent: this worker used manager-verified exact files/ranges, never manager
impersonation. The manager performs independent mechanical and source review.
Atlas guidance content hash: `03d70a369e4df8094091e235fce1bad2635121cd0976795dc9ff22639fb07ca1`.
Manager post-worker update: mechanical and full source reviews passed;
the independent numerical audit reports13831 checks and zero errors. Two full
series retain1042 regions, including81 short calibration regions and no short
warmup/measurement region. See [results](P2_FIELD_RESULTS.md),
[numeric audit](data/p2_numeric_audit_20260905.json),
[build manifest](data/p2_build_20260905.json) and
[collected frontier](RESEARCH_FRONTIER.md). Integration remains deferred.
