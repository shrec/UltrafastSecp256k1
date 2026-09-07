# P3 field sum: contract, bounds and 25 observation lenses

Status: independently derived design contract, pending manager code/test review.
P3 performance, compiler qualification and candidate acceptance are **PENDING IMPORT**.
This document authorizes no production integration. The owner's field-first
collection phase and Task MCP suspension remain in force.

## Exact primitive and input contract

Return the canonical field value

`S = (x0 + sum(rhs[i], i=0..N-1)) mod p`, where `p = 2^256 - 2^32 - 977`.

**N always counts RHS operands**, never the initial accumulator. Global `x0`
is included exactly once. Inputs are fully canonical field representatives:
FE64 inputs have four 64-bit limbs encoding an integer in `[0,p)`; resident
FE52 inputs have four limbs below `2^52`, a top limb below `2^48`, and their
unreduced decoded integer is below p. Width bounds alone are insufficient.
These are preconditions, not new per-input checks inside the arithmetic kernel.

- For `N>0`, the pointer denotes a valid readable range of N canonical inputs.
  A null positive-length range is outside the contract; the kernel does not
  promise to detect, reject or throw for it. Do not deliberately dereference
  invalid ranges in an arithmetic test.
- For `N=0`, a null RHS pointer is allowed and must not be dereferenced. Return
  canonical `x0` unchanged. No normalization is mathematically necessary.
- All input objects and input bytes remain unchanged. In FE64-input routes,
  `x0` may alias any RHS element: copying the seed into private state must not
  mutate that element or change its later contribution as an RHS operand.
  Do not manufacture cross-type FE64/FE52 aliasing to extend this contract.
- Return a canonical FE64 value: exact four raw limbs and all 32 big-endian
  bytes must match the independent integer oracle. The output is by value;
  no caller-owned output-buffer overlap contract is introduced here.

Only the final result is externally observable. Internal chunk checkpoints
are mathematical invariants, not returned prefix outputs. Consequently this
is a vector-sum primitive, not a replacement for one field addition or an API
that must materialize every prefix. See the preregistered
[measurement protocol](P3_FIELD_SUM_PROTOCOL.md).

## Chunk boundaries and exact normalization costs

A nonempty chunk starts with a canonical running-prefix accumulator. It adds
k new RHS operands, so its local bound counts **M=k+1 canonical summands**.
After a nonfinal chunk is canonicalized, that one canonical prefix becomes
the next chunk's seed. This is not another addition of the original `x0`.

| Schedule | Maximum RHS per chunk H | Maximum local total M | Nonfinal boundary | Final boundary |
|---|---:|---:|---|---|
| Full1 | 1 | 2 | Full normalize | `to_fe()` |
| Full16 | 16 | 17 | Full normalize | `to_fe()` |
| Full256 | 256 | 257 | Full normalize | `to_fe()` |
| Full4094 | 4094 | 4095 | Full normalize | `to_fe()` |
| Weak4095 | 4095 | 4096 | Weak, then full normalize | Weak, then `to_fe()` |

The selected `to_fe()` implementation full-normalizes a copy exactly once,
packs four limbs, and calls `from_limbs_raw`; it does not perform a second
normalization through that constructor. Do not explicitly full-normalize the
last chunk immediately before `to_fe()`. Weak4095 performs weak normalization
unconditionally at **every** boundary, including a short final chunk. Its final
full normalization is still supplied by `to_fe()`.

For `N>0`, let `C=1+floor((N-1)/H)`. The FE52 route therefore has C full
normalizations: C-1 explicit interior calls and one inside final `to_fe()`.
Weak4095 additionally has C weak normalizations. There are exactly N RHS
additions and one initial seed conversion. These are source-level algorithm
counts, not emitted instruction counts or measured costs. N=0 has no chunks.

All five schedules above are FE64-input end-to-end candidates. The first
resident-input candidate is Weak4095 only, as specified by the protocol.
Canonical FE64 serial addition is the reference route. Fixed public chunk
sizes must not be silently enlarged when a secret-dependent value happens
to be small or zero. A different public-input specialization needs a separate
contract and measurement.

## Decode invariant and independent correctness argument

Write `B=2^256`, `b=2^52`, `q=2^48`, `W=2^64=4096*b`,
and `K=2^32+977`, so `p=B-K` and `B=q*b^4`.
For FE52 limbs define the ordinary nonnegative integer
`D(n)=sum(n[i]*b^i, i=0..4)`; its field decode is `D(n) mod p`.

The invariant inside a chunk is
`D(acc) = D(seed) + sum(D(new RHS terms))` **as an ordinary integer**, until
the chosen normalization boundary. Plain component additions preserve this
identity only while no 64-bit limb wraps. Normalization preserves the residue
because replacing a contribution `x*B` by `x*K` subtracts exactly `x*p`.
The canonical checkpoint therefore equals the complete prefix modulo p.
Induction across chunks proves the stated final sum with global `x0` once.

The proofs below assume M fully canonical summands. Their component sums obey
`r[0..3] <= M*(b-1)` and `r[4] <= M*(q-1)`. Canonical inputs may be zero;
the conservative schedule still budgets its seed as one summand.

### Direct full normalization: M <= 4095

The implementation folds the original top overflow **before** propagating
the lower limbs. Its initial coefficient is
`x=floor(r[4]/q) <= M-1`. Since `(M-1)*K < b`, the first low-word addition obeys

`t0 <= M*(b-1) + (M-1)*K < M*b+b <= W`.

Thus the fold cannot wrap a uint64 word. Its carry in radix b is at most M.
At each successive lower limb, the largest addition is
`M*(b-1)+M = M*b < W`; the outgoing carry is again at most M.
All raw accumulation and propagation are exact. After masking the original
top limb and propagating, `t4 <= q-1+M < 2*q`, hence its overflow is at most
one bit. The first-pass decoded integer U satisfies

`0 <= U < B+M*b^4 < 2*p`.

The implementation's final predicate distinguishes `U>=B` from the canonical
limb comparison for `p<=U<B`. In either case, adding K and dropping the bit at
256 is subtraction of p. Otherwise it leaves U unchanged. One such correction
suffices because U is below 2p. The final carry pass starts with lower limbs
below b and adds at most K<b, so every additional carry is at most one and no
64-bit intermediate wraps. The result is the unique representative in `[0,p)`.

Therefore Full4094 is safe for an arbitrary canonical prefix seed plus4094
RHS values. Full1, Full16 and Full256 satisfy the same sufficient bound.
The 4095-total bound cannot be raised universally for this direct algorithm:
the canonical witness below fails at4096 total summands.

### Weak then full: M <= 4096

Weak normalization propagates bottom-to-top **before** folding. The initial
low-limb carry is at most M-1. Inductively every subsequent lower-word sum is

`M*(b-1)+(M-1) = M*b-1 <= W-1`.

Hence all four lower-word propagations remain exact even at M=4096. The top
sum is at most `M*q-1`, so its fold coefficient x is at most M-1. At that point
the low limb is already below b. Adding `x*K` is below `b+(M-1)*K < 2*b`,
and subsequent carries are at most one. The returned weak state has lower
limbs below b and top limb at most q; its decoded integer is below
`B+(M-1)*K < B+b < 2*p`.

That weak output is an admitted input to full normalization even though it
need not itself be canonical. Its initial full-normalizer coefficient is only
zero or one; `t0+K<2*b`, each lower-word carry addition is at most b, and the
single final correction has the same argument as above. Thus weak followed
by full normalization is safe and canonical. The full call may be the one
inside final `to_fe()`; an extra explicit full call is unnecessary.

This proves Weak4095 for a canonical prefix seed plus4095 RHS operands. It
does not license4096 RHS on top of an arbitrary canonical seed: that would
be4097 total summands, where raw addition can already lose information.

### Boundary witness and domain separation

**MEASURED (P2 diagnostic), not a P3 timing:** the canonical FE52 value
`a={b-1,0,0,0,q-1}` was summed from zero with4095,4096 and4097 copies.
At4096 copies, the raw accumulation still fits, but direct full normalization
starts with `r0=W-4096` and `x=4095`; `r0+x*K` wraps before radix-b carry
extraction. With `d=4095*K-4096=0x00000fff003cfc2f`:

`correct={d,4096,0,0,q-4096}`, whereas `direct={d,0,0,0,q-4096}`.

The decoded loss is exactly `4096*b=2^64`, not `2^256`. Weak then full succeeds.
At4097 copies, raw summation itself wraps and neither later ordering restores
the missing value. The proof and GCC/Clang diagnostic are retained in
[P2 NeedFix](data/p2_lazy_needfix_20260905.json); that file's N denotes total
copies from zero, unlike P3's RHS-count N. Do not interchange these counts.

For a nonzero initial seed equal to a, the same dangerous totals occur at
P3 RHS counts4095 and4096. The admitted full and weak schedules split earlier,
so tests must exercise these vectors without deliberately exceeding a chunk's
contract. Retain the unsafe unsplit P2 diagnostic separately as a negative
control; do not silently repair its evidence or call its failure a P3 failure.

These are canonical-sum theorems, not universal magnitude limits. They do not
extend to subtraction/negation, arbitrary 64-bit arrays, multiply/square
accumulators or outputs known only to have bounded widths. Those require their
own invariants. No production caller reachability or exploit claim is made.

## Conversion, storage and timing boundary

End-to-end routes accept FE64 inputs, convert each RHS where it is consumed
inside the timed sum, and include seed packing and final FE64 decoding. The
resident route accepts already canonical FE52 inputs; its RHS preconversion
is outside repeated jobs and must remain separately labelled. All routes
include final canonical32-byte materialization per timed job. Allocation and
corpus construction are outside the arithmetic kernels and repeated timing.

Input storage is32N bytes for FE64 and40N bytes for resident FE52. At the
protocol's largest N=1048576 these are32MiB and40MiB, respectively. The resident
representation therefore uses more RHS bytes; no reduction in memory traffic,
cache misses, bandwidth pressure, live registers or total working set follows
from this layout calculation. Actual compiler temporaries and spills require
emitted-code inspection; setup-inclusive break-even requires a new measurement.

The first experiment is deliberately one-core: it measures dependent prefix
updates and compiler scheduling of the independent limb chains without thread
launch or scheduler costs. Final sums are mathematically amenable to other
partitions, but this is not a claim that a production batch implementation
should be sequential, nor a measurement of its multicore optimum.

No P1/P2 or scalar-F2 timings are pooled into P3 ratios. P2's canonical-per-step
gains are motivation, not evidence of a deferred-normalization sum speedup.
P3 full-region measurements and independent output checks are required.

## Exactly 25 P3 observation lenses

Labels: **MEASURED (P2)** imports named earlier evidence only; **DERIVED** is
source/mathematical reasoning under the stated assumptions; **HYPOTHESIS** is
an unmeasured candidate. Unavailable P3 quantities remain **PENDING IMPORT**.

| Lens | Current evidence / P3 application | Next gate or limitation |
|---|---|---|
| V01 Reachability | **DERIVED:** seven preregistered routes call existing FE64/FE52 primitives; no production dispatcher changes. | Verify compiled route, macros and actual boundary calls; P3 emitted route evidence: **PENDING IMPORT**. |
| V02 Corpus | **MEASURED (P2):** canonical boundaries and the4096-term witness distinguish safe domains. | P3 needs real N-element prefixes, nonzero/zero seeds, all chunk edges and multiple chunks; finite tests are not a universal proof. |
| V03 Equivalence | **DERIVED:** chunk induction preserves `(x0+sum RHS) mod p`; x0 appears once globally. | Independent C++ Boost oracle plus corrected FE64 reference, raw canonical limbs and every output byte; P3 outcomes: **PENDING IMPORT**. |
| V04 Definedness | **DERIVED:** canonical valid ranges are preconditions; null is allowed only when N=0, with identity output. | No positive-null rejection promise or undefined-memory tests; preserve valid-range and size contracts. |
| V05 Range / overflow | **DERIVED:** full chunks admit4095 total terms; weak-first chunks4096; subtract one for the canonical seed. | Compile-time rejection of excessive schedules; test maximal admitted chunks and the separately retained unsafe witness. |
| V06 Aliasing | **DERIVED:** private accumulator plus read-only RHS permits x0 to alias an FE64 RHS element. | Check unchanged inputs and supported same-type aliases; never fabricate cross-type/raw-kernel aliases. |
| V07 Observable output | **DERIVED:** only final canonical FE64/32BE output is externally observable. | Prefix output would change the contract and invalidate deferred observation; inspect raw checkpoint invariants in tests only. |
| V08 Invariants / carries | **DERIVED:** radix52 additions retain independent limb chains until normalization; `B=K mod p` explains folds. | Audit every carry boundary and the weak-output domain; missing uint64 overflow cannot be fixed by later reduction. |
| V09 Constant time | **DERIVED:** public N and fixed chunk schedules do not by themselves prove secret-independent compiled execution. | No CT certification, secret-dependent dispatch or VT/CT substitution; security qualification: **PENDING IMPORT**. |
| V10 Resources | **DERIVED:** RHS storage is32N versus40N bytes; kernel owns a private running accumulator. | Allocation/setup exclusions explicit; actual scratch/stack and total live memory: **PENDING IMPORT**. |
| V11 Property visibility | **MEASURED (P2):** canonical resident add/sub won64/64 pairs while canonical bridges lost their add/sub/mul/square128/128. | P3 changes observation frequency as well as representation; do not import those ratios as P3 results. |
| V12 Dependency depth | **HYPOTHESIS:** fewer carry-normalization boundaries may shorten the sum's repeated dependency work. | Count whole-loop emitted dependencies and compare matched sums; reduced mathematical boundary count is not measured critical-path depth. |
| V13 Live state | **HYPOTHESIS:** five limb chains and conversion temporaries may expose scheduling freedom or create spills. | Inspect actual registers and stack accesses for each compiled route; peak live state: **PENDING IMPORT**. |
| V14 Operations | **DERIVED:** N RHS additions, C full normalizations, plus C weak calls for Weak4095; C includes final decode's full call. | Validate no duplicate final full normalization; emitted instruction counts and their performance effect: **PENDING IMPORT**. |
| V15 Conversion | **DERIVED:** E2E packs N RHS inside the sum; resident preprocessing is excluded and pays seed/final conversion only per job. | Report distinct contracts; preconversion/allocation-inclusive crossover: **PENDING IMPORT**. |
| V16 Latency | **HYPOTHESIS:** deferred normalization may improve whole-sum elapsed time. | Measure ns/sum including final bytes; not an isolated add latency or individual-call percentile. |
| V17 Throughput | **HYPOTHESIS:** one core may overlap independent limb operations without extra threads. | Normalize ns/RHS but retain whole-sum costs; neither a multicore nor an upper-layer result. |
| V18 Instructions / PMU | **RECORDED manager diagnostic:** `perf stat -e cycles,instructions -- true` failed exit255 with `perf_event_paranoid=4`; policy unchanged. | PMU events are unavailable, not zero; use emitted-code inspection only until authorized capability changes. |
| V19 Cache / bandwidth | **DERIVED:** real N-element prefixes and the32MiB/40MiB largest layouts expose distinct footprints. | Do not infer cache residency or bandwidth causality from sizes/timing; traffic counters: **PENDING IMPORT**. |
| V20 Code size | **MEASURED (P2):** symbol sizes differed materially between wrapper and resident jobs, but were not full hot-path sizes. | Collect P3 symbols and call boundaries separately; spills and instruction-cache impact cannot be inferred from byte counts. |
| V21 Portability | **MEASURED (P2):** GCC/Clang/sanitizers repeated one arithmetic corpus successfully. | Repeat P3 gates; no new architecture, default-LTO or compiler-speed qualification is inherited. |
| V22 Compiler | **HYPOTHESIS:** packing, unrolling or vectorization can change the benefit of the same proven schedule. | Freeze flags, macros, source/binary hashes and no-LTO job boundaries; compare emitted code, not guessed instruction costs. |
| V23 Policy / ties | **DERIVED:** chunk sizes and N are public workload parameters, not data-value dispatch inputs. | Retain all calibration and measured records, ties and losses; no universal winner or invented threshold. |
| V24 Transfer | **DERIVED:** a final-only field sum is neither per-prefix arithmetic nor a point/signature benchmark. | Production integration, scalar work and upper-layer transfer stay deferred; transfer gain: **PENDING IMPORT**. |
| V25 Prior art / novelty | **DERIVED from local provenance:** FE52 and fold identities already exist in the repository; P2 exposed a contract boundary. | No new-mathematics, novel-algorithm or world-record claim; external novelty classification: **PENDING IMPORT**. |

## Review and provenance

The manager supplies verified source ranges; no worker-specific Source Graph
tool is exposed, and this worker did not impersonate manager tools. Relevant
implementation is [field_52_impl.hpp](../../src/cpu/include/secp256k1/field_52_impl.hpp):
weak helper2410-2430, add_assign2506, full helper2636-2668, public wrappers2562/
2673, from_fe2744 and to_fe2758. Retained implementation SHA256:
`2c3f77d7855e9881ebeaf2b7f53de61e683de5201c77d23caaa288df8bf2afb3`.
The existing [P2 lens map](P2_FIELD_25_LENSES.md) and
[P2 NeedFix](data/p2_lazy_needfix_20260905.json) remain unchanged.

Repository identity: `repo_666797171f0141c58bf05f579b2ee16e`; manager session:
`01a06be6-2904-7c62-9e7d-1245c34a5312`. Only this contract document is written
by this subtask. No arithmetic implementation, production file, scalar file,
build setting or earlier evidence is changed, and no benchmark is run here.
The theorem is a written integer-bound argument, not machine-checked proof.
Stop at manager review before treating implementation or timings as accepted.
