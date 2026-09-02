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
