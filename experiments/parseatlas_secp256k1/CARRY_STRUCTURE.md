# L0 carry structure: ordered summaries and state-loss controls

Status: bounded mathematical diagnostic, 2026-09-05. This supplements the
[typed region kernel](REGION_MODEL.md) and follows the existing
[methodology](METHODOLOGY.en.md); it does not extend the schema, introduce world
or view IDs, complete the 25-world search, or implement a performance candidate.
The script is independent of `region_model.py` and the production engine.

The important separation is **carry response versus the complete arithmetic
state**. Three transfer states suffice for an unknown incoming bit. They do not
retain result digits and do not compress the one current carry bit needed by a
serial computation whose incoming carry is already known.

## Mathematical derivation, separate from finite observations

The following is a reviewable mathematical argument under the stated assumptions,
not a result inferred from testing and not a machine-checked proof artifact.
Let the radix be an integer `B >= 2`, digits `0 <= a,b < B`, and incoming carry
`c` be exactly `0` or `1`. Put `s = a+b`. The outgoing carry is
`C(c) = floor((s+c)/B)`.

| State | Condition | Truth table `(C(0), C(1))` |
|---|---|---|
| K: kill | `s <= B-2` | `(0,0)` |
| P: propagate | `s = B-1` | `(0,1)` |
| G: generate | `s >= B` | `(1,1)` |

These cases partition `0 <= s <= 2B-2`. All are attainable, for example by
`(a,b)=(0,0)`, `(0,B-1)`, and `(1,B-1)`. The missing truth table `(1,0)` is
impossible: increasing `c` cannot decrease the quotient. Across all `B^2`
ordered digit pairs, K and G each have `B(B-1)/2` representatives, while P has B.
This count follows by summing the triangular number of pairs below `B-1`, then
using the reflection `(a,b) -> (B-1-a,B-1-b)` for G.

Write `s = B*g+r`, `0 <= r < B`, and choose canonical propagate
`p = [r = B-1]`. Then `C(c) = g OR (p AND c)`. Actual two-digit sums never give
canonical `g=p=1`: if `g=1`, then `r <= B-2`. Nevertheless, the general Boolean
encoding `(g,p)=(1,1)` denotes the same constant-one function as `(1,0)`.
There are **four Boolean encodings, three functions**, not four distinct carry
behaviors. This agrees with, but is tested independently of, the region kernel.

For a lower-significance segment L followed by a higher-significance segment H,
composition is H **after** L:

```text
C_HL(c) = C_H(C_L(c))
g_HL = g_H OR (p_H AND g_L)
p_HL = p_H AND p_L
```

Function composition proves associativity. P is the two-sided identity. It is
not commutative: `K after G = K`, whereas `G after K = G`. Regrouping ordered
segments is permitted; permuting their significance order is not.

The three functions are pairwise distinguishable: K versus P at `c=1`, K versus
G at `c=0`, and P versus G at `c=0`. Thus a lossless summary that must answer for
**both possible incoming bits** needs at least three distinguishable states.
This is not a minimality claim for arithmetic representations, storage bytes,
register allocation, or a known-carry serial execution. In particular, K and P
have the same response when the incoming carry is fixed to zero.

### Why reblocking preserves the complete output

The carry summary alone is insufficient. At radix 4, `(a,b)=(0,0)` and `(0,1)`
both have summary K, but give different low digits with `c=0`.

Retain or recompute each raw residual `r_i^(0)=(a_i+b_i) mod B`. For its actual
incoming carry `c_i`, compute

```text
r_i = (r_i^(0) + c_i) mod B
c_(i+1) = C_i(c_i)
a_i + b_i + c_i = r_i + B*c_(i+1)
```

Let X and Y be the whole input operands. Multiplying by `B^i` and summing over n
digits telescopes the internal carries: `X+Y+c_0 = R + B^n*c_n`.
Associative composition gives the same carry entering each block under any
ordered contiguous reblocking; the retained residuals then give the same result
digits inside each block. Both low and final carry survive. Block lengths and
significance order are retained throughout.

This is not an argument that summaries alone encode an entire block, nor that
building summaries is faster than a serial addition. The diagnostic computes
summaries and local residuals explicitly; their implementation costs are unmeasured.

### Distinct negative controls for repeated addition

Define an intentionally lossy combine on `(low, carry)` pairs: add only the two
low components, split the sum, and discard both operands' stored carry values.
At radix 4:

- Inputs `(1,1,3)` give `(low,carry)=(1,1)` with left grouping and `(1,0)` with
  right grouping. The tuple operation is not associative. The low result is 1
  in both cases: ordinary addition modulo 4 remains associative.
- Inputs `(3,3,3)` total 9, so the exact state is `(low,high)=(1,2)`. Both lossy
  groupings return `(1,1)`. This demonstrates insufficient high-state width,
  **not** nonassociativity. A single output carry bit cannot represent this total.

No associative full-high or carry-save accumulator is implemented here. An
expanded representation must specify and retain the high-state growth bound;
silently dropping, OR-merging, or normalizing it away does not preserve the
declared full-integer boundary. Carry-transfer composition and lossy repeated
addition are different operations and must not share an unqualified
"associative" label.

## Finite executable observations

`carry_structure.py --report` enumerates actual integer digit additions before
comparing their truth tables with the Boolean g/p form. It also checks all 16
g/p encoding pairs, all 27 triples of transfer states for associativity, all
three two-sided identity cases, and explicit distinguishability/direction cases.

The default report uses digit widths 1..4, 1..4 limbs, and at most six total bits
per whole input operand for reblocking. The digit-only enumeration is separate
and covers all requested digit widths. Across the ten admitted layouts:

- **340 ordered digit pairs**, each checked at both incoming carries.
- **18,248 input triples across layouts**, including repeated mathematical
  triples represented in different layouts; not 18,248 unique integer triples.
- **55,528 full-output executions** across every contiguous partition.
- **Zero mismatches** in every low limb, final carry, reconstructed low integer,
  fixed-width little-endian bytes, and reconstructed full sum.
- At radix 4, all **64 three-operand inputs** retain modulo-low associativity;
  **20** show lossy-tuple nonassociativity and **4** need an exact high value of 2.

The oracle uses whole-integer `divmod(X+Y+c_in, radix^limbs)`, independently
of the transfer composition. Mutation fixtures corrupt low output and final
carry separately, ensuring the report cannot succeed by checking only one.
Reversed composition and dropped state have explicit tests. Invalid bools,
out-of-range digits/carries, mutable vectors, bad partitions, nonmonotone truth
tables, unsupported widths, and excessive exhaustive bounds are rejected.

The ordinary tests also include selected 64-bit digit boundaries with 1..4
limbs; these are fixtures, not exhaustive full-width coverage. There is no
production differential integration in this diagnostic. The prior native
reference baselines remain a separate artifact.

Worker mechanical gate: **89 tests passed**. The observation records and source
identity are reproducible with:

```sh
.venv/bin/python -P -m pytest -q experiments/parseatlas_secp256k1/tests/test_carry_structure.py
.venv/bin/python -P experiments/parseatlas_secp256k1/carry_structure.py --report
```

The CLI emits deterministic JSON to stdout, with source SHA-256, explicit
bounds, counts, negative controls, and a digest of the complete ordered output
records. It writes no report file and uses no random seed. The JSON format is a
standalone diagnostic format, not a candidate record satisfying schema
eligibility gates. Maximum exhaustive width is deliberately capped at six bits.

Recorded default output-record SHA-256:
`9b29692c6093e9885b0903cfabe537dc5d1d1eee4abee4544ac48b923e45045a`.
The report computes its source hash at execution; pin that hash when saving an
observation, because changing diagnostic code requires renewed review.

These finite observations have claim class **OBSERVED**. Runtime is exactly
`PENDING IMPORT`. No timing, resource, constant-time, full-domain production
equivalence, portability, prior-art novelty, candidate ranking, or optimization
gain is claimed. This is an explicit carry-algebra control, not a blind
rediscovery experiment or an automatically synthesized new arithmetic method.
