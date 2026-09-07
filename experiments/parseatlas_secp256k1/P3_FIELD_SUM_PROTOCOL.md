# P3 field sum — preregistered measurement protocol

Status: design/implementation, not a measured result. Owner requests continued
field-first collection and defers production integration.

## Question and exact observation contract

For canonical field-p values, return `(x0 + sum(rhs[0..N))) mod p` as canonical
FE64 and all 32 big-endian bytes. N counts RHS operands, not total summands;
x0 is included once. N=0 returns x0 and permits a null RHS pointer.
Only the final result is externally observable. This is a vector sum primitive,
not a single field-add replacement and not a prefix-output benchmark.

The first comparison prices representation, carry scheduling and conversion
together. It deliberately does not extend canonical-sum bounds to subtraction,
negation or products. No secret-dependent dispatch or CT certification is added.

## Candidate matrix

| Route | RHS representation at function entry | Maximum RHS per chunk | Boundary |
|---|---|---:|---|
| fe64_serial | FE64 | 1 | Existing returning field addition per RHS |
| fe52_e2e_full1 | FE64, packed at each use | 1 | Full normalization |
| fe52_e2e_full16 | FE64, packed at each use | 16 | Full normalization |
| fe52_e2e_full256 | FE64, packed at each use | 256 | Full normalization |
| fe52_e2e_full4094 | FE64, packed at each use | 4094 | Full normalization |
| fe52_e2e_weak4095 | FE64, packed at each use | 4095 | Weak then full normalization |
| fe52_resident_weak4095 | Preconverted canonical FE52 | 4095 | Weak then full normalization |

Every nonempty chunk starts with a canonical accumulator, which consumes one
summand of the P2 bound. The maximum TOTAL canonical summands are 4095 for
direct full normalization and 4096 for weak-then-full. Compile-time assertions
must reject configurations outside those bounds. After each nonfinal chunk,
the state is canonical again. At the final chunk, `to_fe()` supplies the one
full normalization; no duplicate full normalization immediately before it.
Weak-first routes still call weak normalization before final `to_fe()`.

All FE52 routes include seed packing and final canonical output conversion.
Only the resident route excludes RHS preconversion from repeated timed jobs.
The resident route requires 40 bytes per input versus 32 for FE64. Allocation
and corpus construction are excluded for all routes; no claim about allocation
or preconversion break-even follows. No resident result may be labelled E2E.

## Workloads and timing

- N = 1, 16, 256, 4096, 65536, 1048576 RHS; correctness adds zero and chunk edges.
- One runtime-generated deterministic canonical corpus, seed 20260905, shared
  across all routes and both series. Use actual N-element prefixes of the same
  corpus, not a small cyclic RHS table disguised as a larger working set.
- Keep the exact corpus generation recipe, seed, checksums covering every input
  byte, x0, expected outputs and input-layout bytes in the evidence. The source
  hash binds the recipe. Largest FE64 input is 32 MiB, FE52 40 MiB.
- Seven routes x six sizes = 42 cells per series.
- Same core and corrected P1 library as P2: CPU4, native GCC14, assembly and
  fast reduction enabled, C++20, LTO off. Clang validates correctness/smoke;
  full-series compiler qualification is not implied.
- One core by deliberate benchmark contract: measure carry dependencies and
  compiler instruction-level scheduling without worker-thread or task-scheduling
  costs. No conclusion about optimal multicore batch implementation follows.
- Two full series, second with reversed ordering; seeded rotations/reversals of
  size groups and routes across rounds. Eight measured rounds and two warmups
  per cell. Compare SAME-ROUND FE64/candidate ratios within each size.
- Calibrate each cell until two consecutive regions at the SAME job count reach
  at least 200 ms. Retain every calibration, warmup and measured record, including
  short or failed qualification attempts. No selective reruns or sample deletion.
- Report whole-region ns/sum and ns/RHS. Per-job route call and final 32-byte
  materialization are timed. No per-RHS compiler barrier. Use noinline/noipa
  job boundaries and an observable full output so repeated identical jobs cannot
  be hoisted or erased. The last identical job per region is byte-checked outside
  timing; separate untimed validation checks all sizes/routes.
- Input packing is inside E2E jobs; resident preconversion is outside and labelled.
  Do not pool P1, P2 or scalar F2 times into P3 ratios.
- No owned build, correctness test or other performance run overlaps a full
  series. Record CPU endpoints and power state without changing machine settings.
  Background services, indexing, thermal conditions and shared SMT sibling load
  are not controlled; short-lived clock readings are not average frequency.

## Correctness gate before performance acceptance

A separate C++ test uses Boost integer arithmetic modulo p, not only the
production reference. Check FE64 raw canonical limbs and every output byte;
validate resident inputs are canonical before use, and preserve all input bytes.
The unchanged corrected FE64 reference is a second comparison, not sole oracle.

Required corpus: zero/one/p-1, limb-boundary patterns, P2 normalization witness,
random canonical inputs/seeds, sizes immediately around every chunk boundary,
multiple chunks and a large vector. Explicitly exercise nonzero x0, x0 aliasing
an RHS element, N=0 with null, all-zero sums, zero residues and negative controls.
Canonical-input preconditions are not permission to test invalid overflow and
then normalize away lost information. Keep P2's invalid direct-4096 diagnostic
separate from admitted chunk schedules.

Run native GCC against the frozen library, independently compiled Clang ASM,
and GCC NO_ASM ASan+UBSan with halt-on-error. The same finite corpus across
configurations is repetition, not additional distinct test coverage.
Compiler-rejection probes verify impossible template schedules cannot compile.

## Evidence and acceptance boundary

Manager executes mechanical gates, then reads new code and its contract.
Independently replay numerical evidence and keep positive/negative results.
Inspect emitted hot-loop shape and symbol sizes; distinguish direct inspection
from measured PMU events. An attempted unavailable hardware counter is reported
as unavailable, not inferred from speed or source operation count.

Use the [P3 contract and 25-lens application](P3_FIELD_SUM_CONTRACT.md) for bounds
and hypothesis accounting. The [P2 map](P2_FIELD_25_LENSES.md) remains unchanged.
P3 may update the central collection ledger after measurement; it must not
silently revise earlier evidence or promote a candidate into production.

No production, scalar, upper-layer, default build/CI, commit, push or tag changes.
Task MCP remains owner-suspended; verified manager tools and disjoint bounded
worker scopes are used. This is not a canonical task acceptance.
