# M0 typed region kernel: bounded L0 addition

Status: executable research infrastructure and known-method controls, 2026-09-05.
The normative [methodology](METHODOLOGY.en.md) remains authoritative, and the
[roadmap](ROADMAP.md) tracks progress. This is not the complete representation engine, a 25-world search,
a new arithmetic discovery, a production change or a performance result.

## Fixed object and boundary

`region_model.py` implements only unsigned L0 addition
`a + b + carry_in`, for `1 <= word_bits <= 64`, `1 <= limbs <= 4`.
Let `B = 2**word_bits` and `M = B**limbs`. Inputs satisfy
`0 <= a,b < M`, and `carry_in` is exactly integer `0` or `1` (not a Python bool).
Both `low` and `carry_out` are outputs:

`a + b + carry_in = low + M * carry_out`, with `0 <= low < M`.

`L0Domain` is immutable and explicitly has no modulus. Domain names for Fp or Fn,
a supplied modulus, unsupported widths, and equal-total-width but different
limb layouts are rejected at the relevant boundary. This is not modular
arithmetic modulo either secp256k1 `p` or `n`; those need separate domain contracts.

The encoder checks the whole-integer ranges before `divmod` decomposition. Input
state has exactly `a0..aN`, `b0..bN`, `carry_in`, in canonical order; output state
has exactly `low0..lowN`, `carry_out`. Here `N = limbs - 1`. Both states use frozen
tuples of canonical little-endian limbs. Missing, extra, duplicated, reordered,
out-of-range or mutable state is rejected. No hidden normalization or modulo
operation repairs invalid state.

The decoder reconstructs the low integer without reduction, emits exactly
`ceil(word_bits * limbs / 8)` little-endian bytes, and retains carry separately.
For non-byte-aligned tiny domains, the unused high bits of the final byte are
zero by the limb bounds. Equal low bytes with a different carry is a mismatch.
There is no arbitrary byte decoder or mutable-buffer API in this slice.
Aliasing of immutable Python values is harmless; this does not test production
C++ in-place/overlapping-buffer alias contracts.

`bigint_baseline` uses whole-integer `divmod(a+b+carry_in, M)` independently of the
limb encoder, registered DAG operations and decoder. Its output bytes and carry
are compared together with the candidate output. `Observation.claim_class` is
always `OBSERVED`, including a negative observation. A passing finite corpus
does not set full-domain equivalence, CT eligibility or production acceptance.

The frozen original engine remains
`fef231d4e4173bd016fb2a3a1eff67087396a203`, but **neither Python control is that
engine or the W01 production reference**. C++ reference integration is still
pending. The separately retained [mulhi64 disagreement](L0_REFERENCE_CHECK.md)
is neither fixed nor concealed by this model.

## Typed DAG and explicit derivation

`Region` declares a domain, input bindings, an ordered tuple of `Node`s, and
output bindings. Each operation has a closed signature over `WORD` and `BIT`.
Node IDs are unique, all uses refer to earlier definitions, arity/types are
checked, and every canonical boundary output is bound to an existing value of
the correct type. Cycles, forward references, unregistered operations and free
callables cannot enter the evaluator. The bounded region allows 1..128 nodes.

Input boundary labels are fixed, but their graph aliases and all node IDs can be
consistently renamed. Evaluation, operation counts, depth and schedule-equivalent
liveness are identifier-neutral. Hashes intentionally change under renaming.

The closed multi-output operation vocabulary is:

| Operation | Inputs | Outputs and semantics |
|---|---|---|
| `ADC` | word, word, bit | Explicit `divmod` gives low word and carry bit. |
| `SPLIT_GP` | word, word | Split the two-word sum into low word, generate and propagate. |
| `COMPOSE_GP` | high g/p, low g/p | Ordered composition `(gH OR (pH AND gL), pH AND pL)`. |
| `APPLY_GP` | g, p, carry-in | Transfer function `g OR (p AND carry_in)`. |
| `FINISH` | raw low, incoming carry, raw generate | Split the increment; **add** raw generate to increment carry, range-check both outputs. |

For one limb, write `a_i+b_i = B*g_i + r_i`, and set
`p_i = (r_i == B-1)`. Its carry transfer is
`C_i(c) = g_i OR (p_i AND c)`. The actual split guarantees that g and p are not
both 1: when g is 1, `r_i <= B-2`. Transfer-function composition is associative
because it is function composition; its direction is high segment after low
segment. The tests retain a directional carry case, not just symmetric inputs.

Arbitrary BIT channels may contain `(g,p)=(1,1)`, a redundant encoding of the
constant-one transfer function. General graph typing does not assert canonical
GP exclusivity or prove correlations between channels. Those properties follow
from the registered split/template construction, not merely BIT types.
Similarly, an arbitrary typed graph can feed `FINISH` an incoherent raw state.
If its arithmetic carry becomes 2, evaluation rejects it; it is never OR-folded
silently into carry 1. Structural validity therefore does not imply definedness
or correct addition over the whole input domain.

Two deterministic constructors are available:

- `serial_region`: one carry-dependent `ADC` per limb.
- `prefix_region`: independent split nodes, an inclusive generate/propagate scan,
  explicit incoming carries and finishing nodes. The unused highest prefix is
  not built; the highest `FINISH` exposes the final carry.

These are supplied known-method templates, not blind rediscovery or autonomous
synthesis. `derive_prefix` only accepts the exact registered serial template.
It produces a witness binding the domain, rule/version and both graph digests.
`replay_derivation` reconstructs the expected target and compares the complete
graph and witness. A forged target does not become legal by recomputing its hash.
Swapped low outputs, replaced final carry, dropped carry-in, reversed GP operands
and role-swapped same-typed operands have concrete rejection fixtures.

Only `serial_adc_to_inclusive_gp_v1` is registered. A renamed graph remains
executable but is not accepted by this exact-template derivation; no rename rule
is claimed. Replay establishes reachability under this trusted rule, separately
from finite differential observations. The operation evaluator and template
constructors are part of the trusted model implementation, not a separately
verified theorem prover. No full-domain production proof is claimed here.

## Static observations, not cost ranking

`static_diagnostics` returns unranked model diagnostics, not a
`research_contract.schema.json` candidate `cost_vector`. The latter requires
eligibility gates that this L0 infrastructure does not establish; that schema
also intentionally has no standalone word primitive. No new world/view IDs or
false applicability/eligibility records are introduced.

One registered operation is one **atomic, multi-output, unit-cost** DAG node.
Depth is the longest dependency path, not a CPU instruction or cycle count.
`region_depth` includes every listed operation, even if a user-built graph has
dead outputs; `output_depth` considers the declared outputs. Operation counts
remain separate because the five different macro-operations do not have equal
real implementation costs.

Liveness uses an explicitly supplied topological schedule, or the declared node
order. All input values are initially resident. All outputs of a node are
allocated before its last-use inputs are freed; unused results are then freed,
and boundary outputs remain resident. Both value count and logical bit count
are returned. This is not register allocation or host memory measurement. An
invalid, incomplete, duplicate or dependency-violating schedule is rejected.

For four 64-bit limbs in the default schedules:

| Control | Registered operations | Region/output depth | Peak live values | Peak logical live bits |
|---|---|---|---|---|
| serial | 4 ADC | 4 / 4 | 11 | 578 |
| prefix | 4 split + 3 compose + 3 apply + 4 finish = 14 | 5 / 5 | 16 | 579 |

This limited macro model gives **no depth or operation-count improvement** for
the prefix control at four limbs. That negative/null control is retained; it is
not a claim that compiled prefix arithmetic is slower or universally inferior.
Different instruction decompositions, wider domains or schedules require
separate preregistered models and measurements, not favorable post-hoc weights.

Boundary costs are separately counted: encoding uses `2*limbs` limb `divmod`s;
decoding uses `limbs` limb shifts, `limbs` sum terms and one fixed-width byte
serialization. These counters describe the explicit conversion algorithm.
Validation, Python allocation, dictionary access, oracle comparison and the
optional `DecodedOutput.total` carry recombination are excluded. There are no
conversion timings, latency/throughput estimates, CT guarantees or Pareto ranks.

## Reproduction and limits

From the repository root:

```sh
.venv/bin/python -P -m pytest -q experiments/parseatlas_secp256k1/tests/test_region_model.py
```

Worker gate after the carry2 correction: **78 passed**. The independent manager
also ran the combined methodology, inventory and kernel suite: **199 passed**. The ordinary test suite
exhausts five tiny domains up to six bits per operand, plus carry-in, checks 1..4 limbs of
64-bit boundary values, and exercises malformed state/graphs/witnesses and
null controls. It is a deterministic correctness gate, not an arithmetic runtime
benchmark. The review correction is retained here: an initial `FINISH` used OR
for two carry channels; explicit additive carry plus rejection now prevents an
invalid user-built graph from hiding carry2. Registered templates were unchanged.

Optional larger exhaustive replay checks all 131,072 `(a,b,carry_in)` inputs for
each of two distinct eight-bit layouts and both candidate templates, directly
against independent whole-integer addition. No random seed is needed:

```sh
.venv/bin/python -P - <<'PY'
import hashlib
import importlib.util
from pathlib import Path
import sys

path = Path('experiments/parseatlas_secp256k1/region_model.py')
spec = importlib.util.spec_from_file_location('l0_replay', path)
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)
print('model_sha256', hashlib.sha256(path.read_bytes()).hexdigest())
for bits, limbs in ((4, 2), (2, 4)):
    domain = m.L0Domain(bits, limbs)
    source = m.serial_region(domain)
    target, witness = m.derive_prefix(source)
    m.replay_derivation(source, target, witness)
    cases = 0
    for a in range(256):
        for b in range(256):
            for cin in (0, 1):
                carry, low = divmod(a + b + cin, 256)
                state = m.encode_inputs(domain, a, b, cin)
                for region in (source, target):
                    out = m.decode_output(m.evaluate(region, state))
                    assert (out.low, out.carry_out, out.low_bytes) == (
                        low, carry, low.to_bytes(1, 'little'))
                cases += 1
    print((bits, limbs), 'inputs', cases, 'evaluations', 2 * cases)
PY
```

Independent replay on the final model hash below found zero mismatches for both
layouts: 131,072 input triples and 262,144 candidate executions per layout.
Together that is **524,288 candidate executions**: the same 131,072 triples
across two layouts and two controls, not 262,144 distinct triples. The manager
ran `(4,2)` and an independent reviewer ran `(2,4)`; both checked low, carry and
bytes, and the latter also explicitly checked the reconstructed total.

Graph digests bind declarations, not the Python evaluator implementation. Pin the
model source hash as well when replaying a saved witness; the optional command
prints it. Changing the trusted evaluator requires renewed review and evidence.

This exhaustive tiny-domain observation does not prove 64/256-bit arithmetic,
cryptographic constant time, C++ equivalence, security, target portability or
performance. Still pending are redundant/carry-save states, generic region
composition, world-policy synthesis, Fp/Fn reductions, production oracles,
held-out transfer design, CPU benchmarks and the actual 25-by-25 observations.

Code independently reviewed by the manager after mechanical gates, with the
finite limits above preserved:

- `region_model.py` SHA-256:
  `cafa2fd9db3c865c880c98e0ce3c843e4ef64d80915e795e2dbe5b41ed29a9b5`.
- `tests/test_region_model.py` SHA-256:
  `909f83fac898452617f02be10c71c52835c1eefde6a6a73be9682773be924341`.
