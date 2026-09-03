"""Cross-formula equivalence on a declared input slice.

The beam search in `rewrite.py` rewrites ONE straight-line program in place. Every
candidate it produces is built from the reference's own input list, and its gate
demands that the candidate's output key set equal the reference's exactly and that
every coordinate match by raw value. Those three commitments are correct for a
peephole substitution -- a caller that reads `.z` must keep getting the same `.z` --
and they are precisely why that search can never propose a *different formula*.

Meloni's co-Z addition is the standing example. Against the engine's mixed add:

    madd_production   inputs (X1, Y1, Z1, X2, Y2)   8M+3S+3neg+7add
    zaddu             inputs (X1, Y1, X2, Y2, Z)    5M+2S+7sub, and six outputs

The input lists are the same length and share four names, but the fifth position
means different things: `Z1` belongs to P alone and Q is implicitly affine, while
`Z` is shared by both. No local rewrite bridges that, so the search is not merely
failing to find co-Z -- it is structurally unable to express it.

What this module adds is not a weaker equivalence. The raw-value test stays exactly
as strong. What it adds is the ability to state the PRECONDITION under which two
different formulas compute the same thing:

    madd_production(X1, Y1, 1, X2, Y2) == zaddu_sum_only(X1, Y1, X2, Y2, 1)

Both sides are affine there, which is the same situation described twice. On that
slice they agree on every coordinate by raw value, and the cheaper one is 36% less
work by the engine's own weighting. A search that cannot say "on this slice" cannot
see that, and no amount of rewriting will make it see it.

The slice is DECLARED, never inferred. A pair that agrees only because the sample
happened to avoid the disagreement is a false result, so agreement is reported with
the sample count that produced it and a disagreeing point is returned rather than
summarised.
"""

from __future__ import annotations

from typing import Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple

from .field import P
from .slp import SLP


class Disagreement(NamedTuple):
    """One point where the two programs differ, kept whole for rechecking."""
    point: Dict[str, int]
    reference: Dict[str, int]
    candidate: Dict[str, int]
    keys: Tuple[str, ...]          # the outputs that differ


class Agreement(NamedTuple):
    reference_name: str
    candidate_name: str
    pins: Dict[str, int]           # the declared slice, reference-side names
    var_map: Dict[str, str]        # candidate input -> reference input
    shared_outputs: Tuple[str, ...]
    extra_outputs: Tuple[str, ...] # candidate outputs the reference does not have
    samples: int
    agreed: int
    disagreement: Optional[Disagreement]
    weighted_before: float
    weighted_after: float
    depth_before: float
    depth_after: float

    @property
    def holds(self) -> bool:
        return self.samples > 0 and self.agreed == self.samples

    @property
    def weighted_delta(self) -> float:
        if not self.weighted_before:
            return 0.0
        return (self.weighted_after - self.weighted_before) / self.weighted_before

    @property
    def depth_delta(self) -> float:
        if not self.depth_before:
            return 0.0
        return (self.depth_after - self.depth_before) / self.depth_before


def _on_curve_points(count: int, seed: int) -> List[Tuple[int, int]]:
    """Points that satisfy y^2 = x^3 + 7, because a point formula is only claimed
    to be correct on the curve. Feeding it off-curve tuples would compare two
    polynomial maps outside the domain either one is defined on."""
    import random
    rng = random.Random(seed)
    out: List[Tuple[int, int]] = []
    while len(out) < count:
        x = rng.randrange(1, P)
        y2 = ((x * x % P) * x + 7) % P
        y = pow(y2, (P + 1) // 4, P)
        if y * y % P == y2:
            out.append((x, y))
    return out


def slice_agreement(reference: SLP,
                    candidate: SLP,
                    var_map: Dict[str, str],
                    pins: Dict[str, int],
                    samples: int = 256,
                    seed: int = 0x5EC0256B1,
                    point_inputs: Sequence[Tuple[str, str]] = (("X1", "Y1"), ("X2", "Y2"))) -> Agreement:
    """Do `reference` and `candidate` compute the same outputs on the declared slice?

    `var_map` maps each candidate input to a reference input; a candidate input
    that is pinned may map to itself and take its value from `pins`.
    `pins` fixes reference-side inputs to constants -- that IS the slice, and it
    is why the comparison is honest rather than a coincidence.

    Only outputs the two programs share are compared. Extra candidate outputs are
    reported, not ignored: for co-Z they are the whole point of the formula, since
    they carry P forward on the new Z and remove the next step's normalisation.
    """
    pairs = _on_curve_points(2 * samples, seed)
    shared = tuple(sorted(set(reference.outputs) & set(candidate.outputs)))
    extra = tuple(sorted(set(candidate.outputs) - set(reference.outputs)))

    import random as _random
    _rng = _random.Random(seed ^ 0x21D)
    free = [_rng.randrange(2, P) for _ in range(max(samples, 8))]
    free_of: Dict[str, List[int]] = {}

    agreed = 0
    used = 0
    first_bad: Optional[Disagreement] = None

    for i in range(samples):
        (x1, y1), (x2, y2) = pairs[2 * i], pairs[2 * i + 1]
        if x1 == x2:
            continue                      # doubling, not addition: a different formula
        supply = {"X1": x1, "Y1": y1, "X2": x2, "Y2": y2}
        # An input that is neither a point coordinate nor pinned gets a random
        # field value. That is what makes the EMPTY slice a real test rather than
        # an error: with Z1 free, a formula that only agrees when Z1 = 1 must be
        # seen to disagree. A precondition nobody can fail is not a precondition.
        for name in set(reference.inputs) | {var_map.get(n, n) for n in candidate.inputs}:
            if name not in supply and name not in pins:
                supply[name] = free[i % len(free)] if name not in free_of else free_of[name][i % samples]
        supply.update(pins)

        ref_env = {name: supply[name] for name in reference.inputs}
        cand_env = {name: supply[var_map.get(name, name)] for name in candidate.inputs}

        try:
            got_ref = reference.evaluate(ref_env)
            got_cand = candidate.evaluate(cand_env)
        except ZeroDivisionError:
            continue

        used += 1
        bad = tuple(k for k in shared if got_ref[k] % P != got_cand[k] % P)
        if bad:
            if first_bad is None:
                first_bad = Disagreement(dict(ref_env), got_ref, got_cand, bad)
        else:
            agreed += 1

    cb, cc = reference.cost(), candidate.cost()
    return Agreement(
        reference_name=reference.name, candidate_name=candidate.name,
        pins=dict(pins), var_map=dict(var_map),
        shared_outputs=shared, extra_outputs=extra,
        samples=used, agreed=agreed, disagreement=first_bad,
        weighted_before=cb.weighted, weighted_after=cc.weighted,
        depth_before=cb.depth, depth_after=cc.depth,
    )


def sweep(reference: SLP,
          candidates: Sequence[SLP],
          slices: Sequence[Dict[str, int]],
          var_maps: Dict[str, Dict[str, str]] = None,
          samples: int = 256,
          progress: Callable[[str], None] = None) -> List[Agreement]:
    """Every candidate against the reference on every declared slice.

    A pair is reported once per slice it holds on, cheapest slice first, because
    "these agree when Z = 1" and "these agree everywhere" are different claims and
    collapsing them would overstate the second.
    """
    say = progress or (lambda _m: None)
    var_maps = var_maps or {}
    out: List[Agreement] = []
    for cand in candidates:
        if cand.name == reference.name:
            continue
        for pins in slices:
            try:
                a = slice_agreement(reference, cand, var_maps.get(cand.name, {}),
                                    pins, samples=samples)
            except ValueError as exc:
                say("  skip %-24s %s" % (cand.name, exc))
                continue
            if a.holds:
                out.append(a)
                say("  hold %-24s on %-16s  weighted %+.1f%%  depth %+.1f%%"
                    % (cand.name, pins or "everything",
                       100 * a.weighted_delta, 100 * a.depth_delta))
    return out
