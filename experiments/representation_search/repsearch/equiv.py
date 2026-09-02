"""Equivalence checking of candidate formulas against the reference oracle."""

from __future__ import annotations

from typing import Callable, Dict, List, NamedTuple, Optional, Sequence

from .field import (Affine, Jacobian, P, Rng, affine_add, affine_double,
                    affine_neg, curve_points, field_boundary_values,
                    is_on_curve, jacobian_randomizations)
from .slp import SLP


class Mismatch(NamedTuple):
    case: str
    inputs: Dict[str, int]
    got: str
    expected: str


class EquivResult(NamedTuple):
    name: str
    cases: int
    mismatches: List[Mismatch]
    scaling_factors: List[Optional[int]]

    @property
    def ok(self) -> bool:
        return not self.mismatches

    @property
    def uniform_scaling(self) -> Optional[int]:
        """If every case scaled by the same constant L, return it.

        A constant L across all inputs means the candidate is the SAME formula
        family expressed on a different projective representative -- exactly the
        (X,Y,Z) ~ (L^2 X, L^3 Y, L Z) relation.  A varying L is still correct,
        it just means the representative depends on the input.
        """
        vals = [s for s in self.scaling_factors if s is not None]
        if not vals:
            return None
        return vals[0] if all(v == vals[0] for v in vals) else None


def _jac_from_outputs(out: Dict[str, int]) -> Jacobian:
    return Jacobian(out["X"], out["Y"], out["Z"])


def check_doubling(slp: SLP, points: Sequence[Affine], z_variants: int = 4,
                   reference: SLP = None, z_one_only: bool = False) -> EquivResult:
    """Verify slp(X,Y,Z) doubles the point, for many representatives of Z."""
    mismatches: List[Mismatch] = []
    scalings: List[Optional[int]] = []
    cases = 0
    for pt in points:
        if pt.infinity:
            continue
        jacs = ([Jacobian(pt.x, pt.y, 1)] if z_one_only
                else jacobian_randomizations(pt, z_variants))
        want_affine = affine_double(pt)
        for j in jacs:
            env = {"X": j.X, "Y": j.Y, "Z": j.Z}
            env = {k: v for k, v in env.items() if k in slp.inputs}
            try:
                out = slp.evaluate(env)
            except Exception as exc:
                mismatches.append(Mismatch("eval", dict(env), "%s" % exc, str(want_affine)))
                continue
            got = _jac_from_outputs(out)
            cases += 1
            if got.to_affine() != want_affine:
                mismatches.append(Mismatch("double", dict(env), str(got.to_affine()),
                                           str(want_affine)))
            if reference is not None:
                renv = {k: v for k, v in {"X": j.X, "Y": j.Y, "Z": j.Z}.items()
                        if k in reference.inputs}
                ref = _jac_from_outputs(reference.evaluate(renv))
                scalings.append(ref.scaling_factor(got))
    return EquivResult(slp.name, cases, mismatches, scalings)


def check_mixed_add(slp: SLP, pairs: Sequence[tuple], z_variants: int = 3,
                    reference: SLP = None) -> EquivResult:
    """Verify slp(X1,Y1,Z1,X2,Y2) = P + Q with Q affine (Z2 == 1).

    Only generic inputs are exercised here: P != +-Q and neither is infinity.
    The exceptional cases are handled by the CALLER in the real engine (they are
    branches outside the formula), so they are tested separately.
    """
    mismatches: List[Mismatch] = []
    scalings: List[Optional[int]] = []
    cases = 0
    for p, q in pairs:
        if p.infinity or q.infinity or p.x == q.x:
            continue
        want = affine_add(p, q)
        for j in jacobian_randomizations(p, z_variants):
            env = {"X1": j.X, "Y1": j.Y, "Z1": j.Z, "X2": q.x, "Y2": q.y}
            try:
                out = slp.evaluate(env)
            except Exception as exc:
                mismatches.append(Mismatch("eval", dict(env), "%s" % exc, str(want)))
                continue
            got = _jac_from_outputs(out)
            cases += 1
            if got.to_affine() != want:
                mismatches.append(Mismatch("madd", dict(env), str(got.to_affine()), str(want)))
            if reference is not None:
                ref = _jac_from_outputs(reference.evaluate(env))
                scalings.append(ref.scaling_factor(got))
    return EquivResult(slp.name, cases, mismatches, scalings)


def degenerate_probe(slp: SLP, kind: str) -> List[str]:
    """Report what a formula DOES on inputs its contract excludes.

    This is a security question, not a correctness one: a formula that silently
    returns a wrong finite point on P == Q is more dangerous than one that
    returns infinity, because a caller that forgets the guard gets no signal.
    """
    notes: List[str] = []
    pts = curve_points(3, seed=0xBADC0DE)
    p = pts[0]
    if kind == "madd":
        env = {"X1": p.x, "Y1": p.y, "Z1": 1, "X2": p.x, "Y2": p.y}
        out = _jac_from_outputs(slp.evaluate(env))
        notes.append("P == Q  -> %s" % ("infinity" if out.infinity else "FINITE %s" % out.to_affine()))
        env = {"X1": p.x, "Y1": p.y, "Z1": 1, "X2": p.x, "Y2": (-p.y) % P}
        out = _jac_from_outputs(slp.evaluate(env))
        notes.append("P == -Q -> %s" % ("infinity" if out.infinity else "FINITE %s" % out.to_affine()))
    elif kind == "dbl":
        if "Z" in slp.inputs:
            env = {"X": p.x, "Y": p.y, "Z": 0}
            out = _jac_from_outputs(slp.evaluate(env))
            notes.append("Z == 0  -> %s" % ("infinity" if out.infinity else "FINITE"))
        else:
            notes.append("Z == 0  -> n/a (Z is not an input; Z==1 specialisation)")
        env = {"X": 0, "Y": 0, "Z": 1}
        env = {k: v for k, v in env.items() if k in slp.inputs}
        out = _jac_from_outputs(slp.evaluate(env))
        notes.append("Y == 0  -> %s" % ("infinity" if out.infinity else "FINITE"))
    return notes
