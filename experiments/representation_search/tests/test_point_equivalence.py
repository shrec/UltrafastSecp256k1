"""Every candidate point formula must be exactly equivalent to the group law.

A failure here means a candidate is WRONG, not slow.  These run against the
independent textbook reference oracle, never against another candidate.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from repsearch import pointforms
from repsearch.equiv import check_doubling, check_mixed_add
from repsearch.field import (P, affine_add, affine_double, affine_mul,
                             affine_neg, curve_points, is_on_curve, G, N)

JAC_BUDGET = {"X": 7, "Y": 4, "Z": 1, "X1": 7, "Y1": 4, "Z1": 1, "X2": 1, "Y2": 1}

POINTS = curve_points(12)
PAIRS = list(zip(POINTS, POINTS[1:] + POINTS[:1]))


def test_oracle_self_consistency():
    p, q = POINTS[0], POINTS[1]
    assert all(is_on_curve(pt.x, pt.y) for pt in POINTS)
    assert affine_add(p, q) == affine_add(q, p)
    assert affine_double(p) == affine_add(p, p)
    assert affine_add(p, affine_neg(p)).infinity
    assert affine_mul(N, G).infinity
    assert affine_mul(3, p) == affine_add(affine_double(p), p)


@pytest.mark.parametrize("name", sorted(pointforms.DOUBLING))
def test_doubling_equivalent(name):
    res = check_doubling(pointforms.DOUBLING[name](), POINTS, z_variants=4)
    assert res.ok, res.mismatches[:1]
    assert res.cases > 0


@pytest.mark.parametrize("name", sorted(pointforms.DOUBLING_Z1))
def test_doubling_z1_equivalent(name):
    res = check_doubling(pointforms.DOUBLING_Z1[name](), POINTS, z_one_only=True)
    assert res.ok, res.mismatches[:1]
    assert res.cases > 0


@pytest.mark.parametrize("name", sorted(pointforms.MIXED_ADD))
def test_mixed_add_equivalent(name):
    res = check_mixed_add(pointforms.MIXED_ADD[name](), PAIRS, z_variants=3)
    assert res.ok, res.mismatches[:1]
    assert res.cases > 0


@pytest.mark.parametrize("name", sorted(pointforms.DOUBLING))
def test_no_magnitude_overflow(name):
    """A magnitude violation is a silent wrong answer, not a slowdown."""
    _mags, violations, _peak = pointforms.DOUBLING[name]().magnitudes(JAC_BUDGET)
    hard = [v for v in violations if "overflow" in v.detail or ">=" in v.detail]
    assert not hard, hard


def test_production_formulas_stay_inside_declared_magnitudes():
    """Production outputs must stay inside the magnitudes point.cpp DECLARES.

    point.cpp hardcodes p.x.negate(8) and p.y.negate(4).  negate(m) computes
    (m+1)*p - a limbwise; if the true magnitude of `a` exceeds m the subtraction
    underflows and returns a valid-looking field element with the wrong value --
    silently.  This is an interoperability constraint, NOT a normalization cost:
    every formula here reaches a magnitude fixed point and none ever requires an
    inserted normalize (see tools/magnitude_fixpoint.py).
    """
    for fn in (pointforms.dbl_production, pointforms.madd_production):
        slp = fn()
        outs = slp.output_magnitudes(JAC_BUDGET)
        assert outs["X"] <= JAC_BUDGET["X"], (slp.name, outs)
        assert outs["Y"] <= JAC_BUDGET["Y"], (slp.name, outs)
        assert outs["Z"] <= JAC_BUDGET["Z"], (slp.name, outs)


def test_m_for_s_trade_formulas_exceed_declared_magnitudes():
    """The M-for-S trade formulas are NOT drop-in safe.

    dbl_2009_l / dbl_2007_bl / mdbl_2007_bl settle at X=22, Y=10, far above the
    X<=8 / Y<=4 that point.cpp declares at its negate() call sites.  Swapping
    one in without also updating those arguments is a silent wrong answer.
    """
    outs = pointforms.dbl_2009_l().output_magnitudes(JAC_BUDGET)
    assert outs["X"] > JAC_BUDGET["X"], outs


def test_projective_scaling_is_detected():
    """dbl_2009_l lands on the L=2 representative of the same projective class."""
    res = check_doubling(pointforms.dbl_2009_l(), POINTS, z_variants=4,
                         reference=pointforms.dbl_production())
    assert res.ok
    assert res.uniform_scaling == 2
