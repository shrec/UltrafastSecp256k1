"""Cross-formula equivalence on a declared slice, and its negative control.

The point of this module is a distinction the beam search cannot make: between
"these two programs compute the same thing" and "these two programs compute the
same thing WHEN Z = 1". Getting the first right is easy; the test that matters is
that the second is not silently promoted to the first.

So the central case here is a negative one. Meloni's co-Z addition agrees with the
engine's mixed add on the affine slice and disagrees off it, and a tool that
reported it as equivalent everywhere would be worse than no tool -- it would licence
substituting a formula whose precondition the caller does not meet.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from repsearch.crossform import slice_agreement, sweep
from repsearch.pointforms import (madd_production, madd_prod_no_signfold,
                                  madd_prod_s2_reassoc, zaddu, zaddu_sum_only)

AFFINE = {"Z1": 1, "Z": 1}
FREE = {}
ZMAP = {"Z": "Z"}


def test_co_z_agrees_on_the_affine_slice():
    a = slice_agreement(madd_production(), zaddu_sum_only(), ZMAP, AFFINE, samples=128)
    assert a.samples > 100, "the sample was discarded, so the result means nothing"
    assert a.holds, a.disagreement
    assert a.agreed == a.samples


def test_co_z_does_not_agree_off_the_slice():
    """The control. With Z1 free the two are different maps, and the tool has to
    say so -- otherwise the precondition it reports is decoration."""
    a = slice_agreement(madd_production(), zaddu_sum_only(), ZMAP, FREE, samples=128)
    assert a.samples > 100
    assert not a.holds, "co-Z must NOT agree with the mixed add for arbitrary Z1"
    assert a.disagreement is not None
    assert a.disagreement.keys, "a disagreement must name the outputs that differ"
    # and the disagreeing point must be re-checkable by hand
    assert set(a.disagreement.point) == set(madd_production().inputs)


def test_co_z_is_cheaper_on_both_metrics():
    a = slice_agreement(madd_production(), zaddu_sum_only(), ZMAP, AFFINE, samples=64)
    assert a.weighted_delta < -0.30, a.weighted_delta
    assert a.depth_delta < -0.30, a.depth_delta


def test_full_co_z_reports_its_extra_outputs():
    """zaddu returns P on the new Z alongside P+Q. Those three extra outputs are
    the reason the formula exists, so they must be reported rather than dropped."""
    a = slice_agreement(madd_production(), zaddu(), ZMAP, AFFINE, samples=64)
    assert a.holds
    assert set(a.extra_outputs) == {"Xp", "Yp", "Zp"}
    assert set(a.shared_outputs) == {"X", "Y", "Z"}


def test_a_true_rewrite_agrees_everywhere():
    """The two madd variants are genuine rewrites of one program, so they must hold
    with no precondition at all. If they only held on a slice, the slice machinery
    would be hiding a real difference."""
    for cand in (madd_prod_no_signfold(), madd_prod_s2_reassoc()):
        a = slice_agreement(madd_production(), cand, {}, FREE, samples=128)
        assert a.holds, (cand.name, a.disagreement)


def test_self_comparison_is_exact():
    a = slice_agreement(madd_production(), madd_production(), {}, FREE, samples=32)
    assert a.holds
    assert a.weighted_delta == 0.0
    assert a.depth_delta == 0.0
    assert a.extra_outputs == ()


def test_sweep_separates_conditional_from_unconditional():
    ref = madd_production()
    cands = [madd_prod_no_signfold(), madd_prod_s2_reassoc(), zaddu_sum_only()]
    held = sweep(ref, cands, [FREE, AFFINE], {"zaddu_sum_only": ZMAP}, samples=64)
    by = {}
    for a in held:
        by.setdefault(a.candidate_name, []).append(bool(a.pins))
    # the rewrites hold both with and without a precondition
    assert by["madd_prod_no_signfold"] == [False, True]
    assert by["madd_prod_s2_reassoc"] == [False, True]
    # co-Z holds ONLY with one
    assert by["zaddu_sum_only"] == [True]


def test_sample_count_is_reported_not_assumed():
    """A pair that agrees on zero admissible points is not an agreement."""
    a = slice_agreement(madd_production(), zaddu_sum_only(), ZMAP, AFFINE, samples=4)
    assert a.samples <= 4
    assert a.holds == (a.samples > 0 and a.agreed == a.samples)
