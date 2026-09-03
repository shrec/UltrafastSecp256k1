# CPU representation search

Research tooling for the question: **for each primitive in this engine, how many
different representations compute the same answer, and what does each one cost?**

Nothing here is production code. Nothing here builds by default. The purpose is
to generate representation variants, *prove* they are exactly equivalent, model
their cost, and hand the survivors to a real benchmark.

## The principle

The only constraints on a candidate are:

1. it must produce the **same answer**,
2. **correctly**, on every input the contract admits,
3. and it must **pass the tests**.

Anything satisfying those three is a legitimate candidate, however unconventional.
Operator precedence, association order, sign placement, coordinate system, limb
layout and normalization schedule are all free variables — they are *choices*,
not laws.

## Where it comes from: the 24-grammar experiment

A bare token sequence is not yet a mathematical expression. It becomes one only
after a grammar turns it into an AST. With four binary operators there are
`4! = 24` strict precedence orders, so one token sequence denotes up to 24
different functions.

`repsearch/grammar.py` enumerates all 24 and groups the results by **exact**
equality of rational functions (cross-multiplication over ℚ — no floating point,
no probabilistic identity test). Reproducing the two originating experiments:

```
$ python3 tools/toy_grammar.py "2*t+1-t/2"
24 grammars -> 8 distinct forms
```

| class | canonical form | grammars |
|---|---|---:|
| 1 | `(3t+2)/2` | 6 |
| 2 | `1` — **the variable vanishes** | 4 |
| 3 | `(3t+1)/2` | 3 |
| 4 | `(3t+4)/2` | 3 |
| 5 | `(t+1)/2` | 3 |
| 6 | `t+2` | 3 |
| 7 | `(t+2)/2` | 1 |
| 8 | `t+1` | 1 |

```
$ python3 tools/toy_grammar.py "2*x*x+1-x/2"
24 grammars -> 8 distinct forms
```

including the total collapse `y = x` (4 grammars) — a nominally quadratic token
sequence that loses its square entirely.

Both match the originating results exactly. That is the validation test for the
tool: it is locked in `tests/test_grammar_toy.py`.

## Translating it to secp256k1

For cryptography the *function* must be preserved bit-exactly, so a grammar
permutation is used as a **generator**, not as the answer. The pipeline is:

```
operation graph
  -> generate alternative representations   (all axes below)
  -> prove exact equivalence                (reference oracle, mod p / group law)
  -> model cost                             (ops, depth, live values, magnitude)
  -> compile and benchmark the survivors    (the only thing that decides)
  -> keep the Pareto frontier, retain the negatives
```

Search axes, all of which preserve the answer:

`ast_precedence_permutation` · `algebraic_rewrite` · `sign_placement` ·
`common_subexpression` · `mul_square_trade` · `coordinate_change` ·
`limb_representation` · `normalization_schedule` · `operation_fusion` ·
`table_layout` · `algorithm_substitution` · `loop_structure`

## Components

| file | role |
|---|---|
| `repsearch/poly.py` | exact multivariate polynomial / rational function over ℚ; equality by cross-multiplication |
| `repsearch/expr.py` | tokenizer + **precedence-parametrised** parser; the grammar is an argument, not a convention |
| `repsearch/grammar.py` | 24 precedence orders, equivalence-class collapse |
| `repsearch/field.py` | secp256k1 reference oracle: F_p, textbook group law, projective equivalence, deterministic corpora |
| `repsearch/slp.py` | straight-line program IR: exact evaluation, cost model, **FE52 magnitude propagation**, C++ emission |
| `repsearch/pointforms.py` | production formulas transcribed from this tree + literature + derived variants |
| `repsearch/equiv.py` | prove-or-reject against the oracle; degenerate-input probes |
| `tools/run_point_equiv.py` | the runner; writes `out/representation-search-cpu/point_equiv.json` |

## Correctness model

Two point formulas are "the same" when they are **projectively equivalent**,

    (X : Y : Z) ~ (L²X : L³Y : LZ),  L ≠ 0

tested without computing `L`, by cross-multiplication: `X'Z² = XZ'²` and
`Y'Z³ = YZ'³`. Raw coordinate identity is *not* required — different formulas
legitimately land on different representatives of the same projective class.
The runner reports the scaling factor `L`; a uniform `L = 2` across all inputs
means the candidate is the same point on a doubled representative.

Every candidate is checked against a textbook affine group law on:
deterministic pseudorandom curve points, several **randomized nonzero Z**
representatives per point (so no formula can pass by accidentally ignoring Z),
and explicit degenerate probes (`P = Q`, `P = -Q`, `Y = 0`, `Z = 0`).

## The magnitude model — what it does and does not decide

FE52 carries explicit magnitude bookkeeping. `repsearch/slp.py` propagates it:

```
negate(m)        -> m + 1        (computes (m+1)·p − a)
a − b            -> m_a + m_b + 1   (FE52 has no native subtract)
a + b            -> m_a + m_b
mul_int(k)       -> m·k          requires k ≤ 32 and m·k < 4096
half             -> (m >> 1) + 1
mul / sqr        -> 1            requires 5·m_a·m_b < ~3.3e6
```

**It does not decide cost.** `tools/magnitude_fixpoint.py` feeds each formula's
outputs back into its own inputs and iterates. *Every* candidate reaches a
magnitude fixed point within 3 iterations, because `mul`/`sqr` reset the
magnitude to 1. None of them ever requires an inserted `normalize`. FE52's
12 bits of headroom are simply large enough that these formulas never approach
the bound. An earlier draft of this document claimed the opposite; that claim
was wrong and is withdrawn.

**It does decide interoperability, and getting it wrong is silent.**
`point.cpp` hardcodes the declared magnitude at each `negate()` call site —
`p.x.negate(8)` and `p.y.negate(4)`. `negate(m)` computes `(m+1)·p − a`
limbwise; if the true magnitude of `a` exceeds `m`, the subtraction
**underflows** and yields a valid-looking field element with the wrong value.
No assertion, no crash. `tools/interop_check.py` reports which formulas are
drop-in safe. Z is never negated in these formulas — only multiplied — so it is
bounded by the accumulator, not by a declared magnitude.

## Results so far — point layer

All 12 candidates are **proven exactly equivalent** to the reference group law
(96 cases each: 24 curve points × 4 randomized-Z representatives).

Weights are a ranking filter only (`M = 1.00`, `S = 0.93`, add/negate/mul_int/half
≈ 0.05). **They are not measurements and must never be quoted as such.**

### Doubling, general Z

| formula | ops | weighted | depth | live | L | out mags | closed |
|---|---|---:|---:|---:|---:|---|---|
| `dbl_production` *(in tree)* | 3M+4S | 7.140 | 3.250 | 6 | 1 | X3/Y3/Z1 | ✅ |
| `dbl_prod_alt_sign` *(derived)* | 3M+4S | **7.060** | **3.170** | 7 | 1 | X4/Y3/Z1 | ✅ |
| `dbl_prod_mul_by_3_as_add` *(derived)* | 3M+4S | 7.210 | 3.310 | 7 | 1 | X3/Y3/Z1 | ✅ |
| `dbl_2009_l` *(in tree, legacy 4x64)* | 2M+5S | 7.290 | 3.340 | 8 | 2 | X22/Y10/Z2 | ❌ |
| `dbl_2007_bl` *(EFD)* | 1M+7S | 8.270 | 3.340 | 10 | 2 | X22/Y10/Z5 | ❌ |

### Doubling, Z == 1

| formula | ops | weighted | depth | live | L | out mags |
|---|---|---:|---:|---:|---:|---|
| `dbl_z1_production` *(in tree)* | 2M+4S | 6.140 | 3.250 | 6 | 1 | X3/Y3/Z4 |
| `mdbl_2007_bl` *(EFD)* | 1M+5S | 6.290 | 3.340 | 7 | 2 | X22/Y10/Z8 |

### Mixed addition, Jacobian + affine

| formula | ops | weighted | depth | live | L | out mags | closed |
|---|---|---:|---:|---:|---:|---|---|
| `madd_production` *(in tree)* | 8M+3S | 11.330 | 5.260 | 8 | 1 | X4/Y2/Z1 | ✅ |
| `madd_prod_no_signfold` *(derived)* | 8M+3S | **11.210** | **5.160** | 8 | 1 | X6/Y3/Z1 | ✅ |
| `madd_prod_s2_reassoc` *(derived)* | 8M+3S | 11.330 | 5.260 | 8 | 1 | X4/Y2/Z1 | ✅ |
| `madd_2007_bl_zmul` *(derived)* | 8M+3S | 11.440 | 5.210 | 9 | 2 | X6/Y4/Z2 | ❌ |
| `madd_2007_bl` *(in tree, legacy 4x64)* | 7M+4S | 11.490 | 5.210 | 11 | 2 | X6/Y4/Z5 | ❌ |

### What this already establishes

**1. The M/S trade is the wrong axis on this engine.** `S/M ≈ 0.93`, so swapping
a multiply for a square buys ~7% of one operation out of ~7 — under a tenth of a
percent of the primitive. Meanwhile the magnitude column moves by a factor of 7.

**2. Magnitude closure explains production's choices, and operation counts do not.**
`madd_production` spends one *extra* multiply relative to `madd_2007_bl` (8M+3S
vs 7M+4S) and is *more* expensive by raw weighted count. It wins because the
h2-negation turns every subtraction into an addition, and in FE52 a subtraction
costs `+1` magnitude while an addition does not. The result: output magnitudes
`X4/Y2/Z1`, inside budget, iterable with no normalize. `madd_2007_bl` exits the
budget on `Z` (`Z5` vs `Z ≤ 1`). That is a representation win that was already
harvested in this tree — and it is invisible to every published M/S table.

**3. The originating hypothesis (2M+5S beats 3M+4S) is rejected by the model.**
`dbl_2009_l` is worse on weighted cost (7.290 vs 7.140), worse on depth, needs
2 more live values, and is *not* magnitude-closed (peak 33, outputs `X22/Y10`).
It is also already in this tree as the superseded 4x64 path. Measurement can
still overturn the model, but the model now says what to expect and why.

**4. Two derived candidates beat production on every modelled axis.**
`dbl_prod_alt_sign` (7.060 vs 7.140, depth 3.170 vs 3.250, still closed) and
`madd_prod_no_signfold` (11.210 vs 11.330, depth 5.160 vs 5.260, still closed).
Both are pure sign-placement/association changes with identical M/S counts.
Both must now be **measured** — the model is a filter, never a verdict.

**5. The oracle earns its keep.** A hand-derived sign-placement variant of the
mixed add produced the correct X and the negated Y. It looked right, it had the
right operation count, and it was wrong. The corpus caught it deterministically
on the first case.

## MEASURED results

Machine: x86-64, GCC 14.2.0, `-O3 -march=native -DNDEBUG`, `powersave` governor,
shared/loaded machine. Estimator: minimum over passes (contention only adds
time), reported as best-of-5 independent runs. Raw output in
`out/representation-search-cpu/`.

### FE52 operation costs (measured directly, not fitted)

An earlier attempt fitted these by least squares over the formula timings. The
fit was excellent -- max error 1.65% -- and the coefficients were nonsense
(`sqr` at 0.10x `mul`, `half` at 1.37x `mul`), because the design matrix is
collinear: M and S counts move together across the formulas. A good fit does not
make a collinear coefficient identifiable. These are measured one operation at a
time instead.

| operation | latency ns | rel. mul | note |
|---|---:|---:|---|
| `mul` | 21.52 | 1.000 | dependent chain |
| `sqr` | 18.81 | **0.874** | dependent chain |
| `half` | 2.09 | 0.097 | |
| `neg` | 0.71 | 0.033 | |
| `sub` (= `add` + `negate`) | 0.72 | 0.033 | FE52 exposes **no** `operator-` |
| `mul_int(3)` | 0.48 | 0.022 | upper bound, includes the reset copy |
| `add` | 0.36 | 0.017 | |
| `add` + `normalize_weak` | 4.69 | 0.218 | |

### Instruction-level parallelism — the number that governs formula cost

| independent chains | mul ns/op | sqr ns/op | speedup |
|---:|---:|---:|---:|
| 1 | 21.62 | 19.10 | 1.00x |
| 2 | 12.50 | 10.75 | **1.73x** |
| 3 | 12.45 | 10.15 | 1.74x |
| 4 | 12.45 | 10.11 | 1.74x |
| 8 | 12.36 | 10.05 | 1.75x |

**ILP saturates at two field multiplies in flight and goes no further.** This is
the single most important constant for formula design on this engine, and no
operation count contains it. A formula's cost sits between a latency bound
(21.5 ns per heavy op on the critical path) and a throughput bound (12.4 ns per
heavy op when two are independent) -- a factor of 1.74 that representation
changes move you along.

Sanity check on `dbl_production` (3M+4S): throughput bound
`3 x 12.43 + 4 x 10.11 = 77.7 ns`; measured **78.77 ns**. The production
doubling is essentially throughput-bound, i.e. already scheduled well enough to
keep the multiplier busy.

### Point formulas, measured side by side in ONE binary

All variants in one translation unit, one input set, rotating order, warm-up
before timing, checksum printed so nothing can be optimised away. Every variant
is checked projectively against the production body at runtime before any timing
happens. Compared against `*_generated`, so the code-generation style is held
constant and only the formula varies.

| doubling | ns | vs production formula |
|---|---:|---:|
| `dbl_production_generated` | **78.77** | — |
| `dbl_prod_mul_by_3_as_add` | 79.20 | +0.54% |
| `dbl_production_search2` (auto) | 79.22 | +0.57% |
| `dbl_production_search0` (auto) | 79.44 | +0.85% |
| `dbl_prod_alt_sign` | 79.70 | +1.17% |
| `dbl_handwritten` (in-place, production style) | 83.94 | **+6.56%** |
| `dbl_2009_l` (2M+5S) | 84.30 | **+7.02%** |
| `dbl_2007_bl` (1M+7S) | 88.44 | **+12.27%** |

| mixed addition | ns | vs production formula |
|---|---:|---:|
| `madd_production_search0` (auto) | 122.95 | -0.34% |
| `madd_prod_s2_reassoc` | 123.02 | -0.28% |
| `madd_production_generated` | **123.37** | — |
| `madd_handwritten` (in-place, production style) | 126.37 | **+2.43%** |
| `madd_prod_no_signfold` | 126.94 | +2.89% |
| `madd_2007_bl` (7M+4S) | 146.34 | **+18.62%** |
| `madd_2007_bl_zmul` (8M+3S) | 148.17 | **+20.10%** |

### What the measurements establish

**1. No representation beats the production formulas.** The best the automatic
search found is `madd_production_search0` at -0.34%, which is inside the noise.
Across ~5900 generated programs, the production point formulas are at a local
optimum.

**2. The published M-for-S alternatives are 7% to 20% slower, and the operation
count predicted ~1%.** The op-count model was wrong by up to 13x on
`madd_2007_bl` (predicted +1.4%, measured +18.6%). Trading a multiply for a
square buys 12.6% of one operation on this engine (S/M = 0.874) and costs far
more than that in dependency structure and cheap-op traffic.

**3. The hand-written in-place style is SLOWER than plain SSA/by-value code**:
+6.56% on doubling, +2.43% on mixed addition, for the identical formula. The
in-place variants exist to avoid "the 128-byte return value copy"; on GCC 14.2
with `-O3 -march=native` that trade is currently negative. This is the only
result here that points at a real engine speedup, and it needs validating at the
actual call sites before any claim is made -- this harness passes results through
out-parameters, which is not how `jac52_double_inplace` is invoked.

**4. Negative results are the bulk of the yield, and that was the expected
outcome.** The value delivered is a calibrated cost model, a magnitude checker,
an equivalence oracle, and a measured refutation of the formula alternatives --
not a speedup.

## x86-64: CLOSED

Every axis has a verdict. Two wins; everything else refuted or below what this
machine can resolve. Recorded in full as knowledge-base entry
`X86-REPRESENTATION-SEARCH-EXHAUSTED`.

### Confirmed wins

| change | measured | where |
|---|---|---|
| **co-Z coordinate change**, all 4 live sites | `dual_scalar_mul_gen_point` **−7.19%**, `ecdsa_verify` **−1.96%**, 0 regressions | verify + `ct::scalar_mul` + BIP-324 |
| **CT SafeGCD inverse**, both call sites | `ElligatorSwift XDH` **−8.23%**, `Session handshake` **−4.35%** | ECDH, BIP-352 scan |

Both are macro-guarded; the default build is byte-identical.

The inverse result is worth stating precisely because it is the cleanest
prediction the cost model made: the primitive measured 3847.5 ns (Fermat, CT)
against 1555.0 ns (`ct::field_inv`, CT SafeGCD). 2292.5 ns saved on a 29 122.7 ns
XDH predicts **7.87%**; measured **8.23%**. And `ecdh_compute` did *not* move
(+0.18%) — correctly, because it does not go through `ecmult_const_xonly`.

### One real bug

`WINDOW_G` was a duplicated literal that had to match `kDualMulWindowG`.
Changing the table width alone **segfaulted at window 12** and returned **five
silently wrong `dual_mul` results at window 13**. Fixed, `static_assert`ed,
behaviour pinned by the new audit module `regression_table_build_invariants`,
filed as #399.

### Refuted or below the noise floor — do not re-chase

| axis | verdict |
|---|---|
| Point formulas at fixed coordinates | **local optimum** — 5894 generated programs, best −0.34% |
| M-for-S trades | worthless, S/M = 0.874 |
| co-Z Montgomery ladder | 4110 heavy ops vs 1690 — **2.43× more work** |
| Fermat addition chain | **proven minimal**, `l(223) = 11` by exhaustive search |
| Limb-level Karatsuba | multiplies are 13.6% of the kernel; 3 fewer ≈ 1.3%, against ~20 added ALU ops |
| Limb schedule, more columns | `wide_mul` measured **+9.45% slower** |
| `optimize("O2")` on the kernels | byte-identical to `O3` |
| Removing `noinline` | microbenchmark says −7.20%; it is a single-call-site artefact. Prior controlled A/B: 0.2% slower under LTO. `noinline` is an I-cache win worth **+12%** end-to-end |
| Scalar reduction mod n (int128) | already the Solinas `N_C` fold |
| `add2_assign` / duplicated-negate CSE | GCC already emits identical assembly |
| wNAF digit sign folding | 0–0.16% of a verify |
| **AVX2 vectorisation** | **refuted at the multiply port**: 25 limb-products per field multiply in *both* 5×52-scalar and 10×26-AVX2 — the lane count is exactly cancelled by the limb-count growth |
| Strauss vs Pippenger | **already optimal** at every `n` the engine reaches |
| GLV decomposition | 84.92 ns = **0.34%** of a verify even if deleted outright |
| `sqrt` / `jacobi` | one jacobi per *cold* `lift_x` only; the 1024-slot cache makes it **0.0%** of `schnorr_verify` |
| Redundant `normalize_weak` | 0.60% of a BIP-352 scan, **exactly 0.00%** of verify and ConnectBlock |

### Still open on x86

The dual-mul G-table window width — 15 (default, **1280 KB of a 2048 KB L2**)
against 13 and 12. All three builds exist and pass `test_ecc_properties`. The
decision needs **ConnectBlock on a quiet frequency-locked machine**, not
`bench_unified`: an isolated benchmark calls the function repeatedly with nothing
competing, keeps the whole table resident, and therefore systematically flatters
the large window.

### The transferable result

Searching **within** a fixed representation found nothing — 5894 rewrites moved
0.34%. **Changing** the representation found 7–8%. And the operation-count model
predicted direction correctly only *after* it was calibrated against measured
per-operation costs; before that it was wrong by up to 13× (it put
`madd-2007-bl` at +1.4%; measured +18.6%).

## Running it

```bash
cd experiments/representation_search

# reproduce the originating grammar experiments
python3 tools/toy_grammar.py "2*t+1-t/2"
python3 tools/toy_grammar.py "2*x*x+1-x/2"

# prove or reject every point-formula candidate
python3 tools/run_point_equiv.py --points 24 \
    --json ../../out/representation-search-cpu/point_equiv.json

# emit a candidate as C++ FE52 code (with correct negate() magnitudes)
python3 tools/emit_cpp.py dbl_prod_alt_sign

# lock-in tests
python3 -m pytest tests/ -q
```

No dependencies beyond the standard library. Raw results go to
`out/representation-search-cpu/`, which is outside the source tree.

## Status

| stage | state |
|---|---|
| exact equivalence oracle (ℚ and F_p) | done, self-tested |
| 24-grammar generator + collapse | done, reproduces both originating experiments |
| secp256k1 reference group law | done, self-tested |
| SLP IR, cost model, magnitude model | done |
| point layer: doubling / Z=1 / mixed add | 12 candidates proven equivalent |
| point layer: full Jacobian addition | not yet |
| field layer (mul, sqr, normalize, inverse) | not yet |
| scalar / GLV / recoding layer | not yet |
| end-to-end workloads | not yet |
| C++ A/B benchmark harness | not yet — **nothing here is a measurement** |

No number in this document came from running code on hardware. The weighted
costs are model outputs. Until the benchmark harness exists and a controlled
A/B run is recorded, every entry above is a *prediction*.

## Cross-formula equivalence on a declared slice

`repsearch/crossform.py`, driven by `tools/run_crossform.py`.

The beam search in `rewrite.py` rewrites one straight-line program in place. Every
candidate it builds uses the reference's own input list, its gate demands the
candidate's output key set equal the reference's exactly, and equivalence is raw
value equality on every coordinate. Those three commitments are right for a
peephole substitution -- a caller reading `.z` must keep getting the same `.z` --
and they are exactly why that search can never propose a **different formula**.

Meloni's co-Z addition is the standing example:

```text
madd_production   inputs (X1, Y1, Z1, X2, Y2)   8M+3S+3neg+7add   weighted 11.33  depth 5.26
zaddu             inputs (X1, Y1, X2, Y2, Z)    5M+2S+7sub        weighted  7.28  depth 3.23
```

Same arity, four shared names, but the fifth position means different things: `Z1`
belongs to P alone with Q implicitly affine, while `Z` is shared by P and Q. No
local rewrite bridges that.

This module does not weaken the equivalence test. The raw-value comparison stays
exactly as strong. What it adds is the ability to state the **precondition** under
which two different formulas agree:

```text
madd_production(X1, Y1, 1, X2, Y2) == zaddu_sum_only(X1, Y1, X2, Y2, 1)
```

Both sides are affine there, which is one situation described twice.

Running it against the whole registry:

```text
formula                  slice               weighted     depth  extra outputs
zaddu                    Z=1, Z1=1             -35.7%    -38.6%  Xp, Yp, Zp
zaddu_sum_only           Z=1, Z1=1             -35.7%    -38.6%  -
madd_prod_no_signfold    everywhere             -1.1%     -1.9%  -
madd_prod_s2_reassoc     everywhere             +0.0%     +0.0%  -
```

The −35.7% is the same transformation the engine measured at 34.7% on hardware.

**The control is the part that matters.** An unpinned input takes a random field
value, so the empty slice is a real test rather than an error, and co-Z must be
seen to fail it:

```text
off the slice (Z1 free): agreed  0/64   holds=False
on the slice  (Z1 = 1) : agreed 64/64   holds=True
```

Zero of sixty-four. The two `madd_prod_*` variants, which are genuine rewrites,
hold with no precondition at all. A tool that could not tell those two situations
apart would licence substituting a formula whose precondition the caller does not
meet, which is worse than having no tool.

Extra outputs are reported, never dropped: `zaddu` hands back P on the new Z
alongside P+Q, and those three values are the reason the formula exists -- a chain
of them needs no normalisation between steps.

Slices are **declared**, never inferred from whether they happen to pass.
