# M0 CPU primitive inventory

This is a source-observed, frozen inventory for ParseAtlas-style bottom-up
research, not a new arithmetic implementation, runtime trace, performance result,
constant-time certification or full-domain proof.

Reference revision: `fef231d4e4173bd016fb2a3a1eff67087396a203`.
The [machine-readable graph](data/primitive_graph.json) contains 10 primitive
families, 18 typed edges, 51 bounded witnesses and 16 source-file SHA256 values.
Every witness is checked against both the current exact file and its blob at the
reference revision. Missing sources fail validation; they cannot become “pending”
successes. The 5 existing C++ test anchors are located but **not executed by this
inventory**.

## Dependency structure

Arrows mean “consumed by”, not “calls downward”. Conversion edges are separate
from the acyclic consumption graph; `consumers` lists must exactly match outgoing
consumption edges.

```text
L0  word / carry / borrow / full product
      ├── L1 Fp 4x64 ⇄ Fp 5x52     (explicit conversion boundaries)
      └── L1 Fn 4x64                (different modulus and zero policies)
                  │
L2             point operations ← Fp
                  │
L3       scalar_mul / dual_scalar_mul / general MSM ← Fn
                    │                    │
L4       single ECDSA verify       large Schnorr batch verify
               dual path          generator term + general MSM
```

This is a bounded graph of source families. Compile-time macros and runtime CPU
dispatch determine which implementations execute. In particular,
`SECP256K1_FE52_COMPUTE` and `SECP256K1_FAST_52BIT` are different capabilities;
FE52 compute does not prove FE52 point storage. No selected build configuration
or executed branch is inferred from an indexed function name.

## Primitive contract inventory

Each JSON node explicitly records its domain/range, input and output state,
invariants, CT classification, consumers, test evidence, benchmark evidence and
open research obligations.

| Family | Preserved object / boundary | First research obligations |
|---|---|---|
| word64 | Unsigned word **and** carry/borrow or high product word | Full-width overflow, signedness, shifts, backend identity |
| Fp 4x64 | Residue modulo p; raw storage versus canonical boundary | Reduction schedule, exception policy, alias/range contract |
| Fp 5x52 | Same Fp residue plus lazy magnitude state | Weak/full normalization, negate bounds, conversion cost |
| Fn 4x64 | Residue modulo n, not p | Order-complement reduction, strict/reducing parsing, zero inverse |
| point | Group point plus caller-visible coordinate/flag state | Infinity, equal/opposite inputs, randomized nonzero Z |
| scalar_mul | kP with fixed/variable base and selected backend | Recoding, tables, normalization and public/secret route |
| dual_scalar_mul | aG+bP, distinct platform families | G-table consistency, conversion, actual ECDSA consumer |
| MSM | Sum of scalar-point terms | Signed carry, affine guards, size thresholds, scratch |
| ECDSA verify | Boolean protocol acceptance | Fn products, dual multiplication, field-coordinate check |
| Schnorr batch | Boolean result with randomness as an input | Small/large branch distinction and randomized MSM soundness |

The entry descriptions are obligations to preserve, not evidence that every
branch already satisfies them. Most arithmetic and point families are explicitly
backend-conditional or public-data variable-time; no unreviewed fast path is
licensed for secret-key use.

## Primary source observations

- Generic Fp addition calls `add64` with two carry chains in
  `field.cpp:423–448`; Fn addition independently uses `ORDER`, `add64` and
  `sub64` in `scalar.cpp:83–103`. The domains do not share a reduction rule.
- `FieldElement::to_bytes` at `field.cpp:2517–2526` serializes stored limbs.
  It does **not** normalize. Raw construction/mutation exists, so an oracle must
  state where canonicalization occurs instead of assuming the serializer does it.
- `FieldElement52::from_fe` at `field_52_impl.hpp:2744–2754` only repacks.
  `to_fe` at `2758–2769` first normalizes a copy and then packs with
  `from_limbs_raw`. Lazy and canonical states are not interchangeable labels.
- `FieldElement::inverse(0)` at `field.cpp:3541–3554` throws on ordinary hosts
  but returns zero under the listed embedded macros. The witnessed native-128
  `Scalar::inverse(0)` at `scalar.cpp:893–897` returns zero.
- `ecdsa_verify` at `ecdsa.cpp:890–913` directly calls
  `Point::dual_scalar_mul_gen_point`, **not** general `msm`. Its coordinate
  check crosses the scalar-value/field-representative boundary; that is not an
  isomorphism between Fn and Fp.
- `schnorr_batch_verify_impl` at `batch_verify.cpp:361–463` uses individual
  verification for small batches. Only the large branch builds `2*n` terms,
  calls `msm` at line 460, adds a generator term and checks infinity. CSPRNG
  randomness is an explicit input; deterministic replay must not weaken
  production entropy.

The manager's separate [L0 reference diagnostic](L0_REFERENCE_CHECK.md) reports
a forced portable `mulhi64` fallback disagreement for two maximum words:
expected high word `fffffffffffffffe`, observed `fffffffefffffffe`.
That result is not reproduced by these inventory tests and does not establish
that the native production build selects the fallback. It is a reason to retain
an independent integer oracle and never treat a frozen implementation as its own
proof. No arithmetic fix is included here.

## Prior results: retain wins, negatives and corrections

The graph includes 11 individually identified historical records with exact
README spans, affected nodes and reported-evidence type. All are
`SECONDARY_NOT_IMPORTED`: a pinned README is not the primary benchmark artifact
or proof certificate.

| Relevant layer | Pinned historical README span | What is retained, without promotion |
|---|---|---|
| Fp / FE52 | 216–251 | Reported direct costs and two-chain ILP plateau; collinear fitted coefficients were rejected |
| point formulas | 287–311 | Reported fixed-coordinate alternatives lost or stayed inside noise |
| point / dual / ECDSA | 319–332 | Reported macro-guarded co-Z gains and CT-inverse call-site gains |
| word / limb kernels | 342–361 | Wider-column slowdown, limited Karatsuba upside and AVX2 objection in the measured setting |
| Fn / MSM | 342–361 | Existing N_C fold and then-current routing claims, not domain exhaustion |
| dual / workload | 363–370 | G-table window question remains open pending workload measurement |
| evidence consistency | 403–420 | “No hardware measurement yet” status conflicts with earlier measured sections |
| boundary / equivalence | 440–484, 526–551 | Declared slices, extra outputs, input-name seeding bug, opt-in projective equality |
| FE52 cost model | 493–524 | Later correction prices subtraction as negate+add and reverses early modeled wins |

The README's finite-case use of “proven” and its “CLOSED” heading are not imported
as full-domain proof or global search exhaustion. Its later corrections outrank
the earlier uncorrected model conclusions. Old percentages do not calibrate a
new backend or workload without the primary artifacts and a controlled rerun.

## Validation and limitations

Run offline from the repository root:

```bash
.venv/bin/python -P -m pytest -q experiments/parseatlas_secp256k1/tests/test_primitive_graph.py
git diff --check
```

The real Draft 2020-12 validator enforces the strict schema. Additional tests
verify source hashes against the frozen Git objects, bounded spans and literal
witnesses, closed reviewed role bindings, node/edge references, consumption acyclicity,
consumer agreement, and test/history cross-links. Negative cases include valid
but unrelated function substitution, missing mandatory evidence, source
self-repinning, path escape, cycles, mislabeled test execution and promoted
historical claims. The fixed v1 inventory accepts only the reviewed node,
witness, edge, test-anchor and historical-record vocabulary; an extension requires
an explicit fixture update and review. Valid source text from an unrelated
function or README section does not authenticate a different role. Historical
evidence kinds are bound as well, so a model statement cannot be relabeled as a
measurement. These are curated evidence bindings, not a C++ semantic parser.
The DAG helper is tested independently, including self and multi-node cycles,
so closed-vocabulary rejection cannot masquerade as a successful cycle test.

No C++ arithmetic test, timing benchmark or byte-differential corpus has run as a
consequence of this inventory. Explicit FE52 conversion, full word-state, complete
backend and secret-path coverage remain later work. GLV/Strauß internals,
precomputation internals, non-CPU backends and other protocols are not exhaustively
mapped.

The next layer starts with modular primitives: coherent representation worlds
and observation views, then an admitted-input corpus, frozen-engine comparison
and independent integer mod-p/mod-n oracle. Exported canonical bytes and all
contractually exposed state must match on identical inputs. Internal
representatives may differ only where the caller contract permits. Finite corpus
agreement remains `OBSERVED`; it cannot establish full-domain correctness, CT
behavior, speed or novelty.
