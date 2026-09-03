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


def test_one_point_and_two_point_names_denote_the_same_point():
    """A doubling calls its input (X, Y); an addition calls the first point
    (X1, Y1). Both names have to be seeded from the SAME point, or a doubling
    compared against a doubling is evaluated at two different inputs and the
    disagreement that comes back means nothing.

    That is not hypothetical: it is what an all-pairs registry sweep did before
    this was fixed. Every doubling pair was reported as non-agreeing, which read
    as "the doubling family is exhausted" when the truth was that the family had
    never been compared at all.
    """
    from repsearch.pointforms import (dbl_production, dbl_prod_alt_sign,
                                      dbl_prod_mul_by_3_as_add)

    ref = dbl_production()
    for cand in (dbl_prod_alt_sign(), dbl_prod_mul_by_3_as_add()):
        a = slice_agreement(ref, cand, {}, FREE, samples=64)
        assert a.samples > 40, "the sample was discarded"
        assert a.holds, (cand.name, a.disagreement)


def test_the_two_z_conventions_are_kept_apart_by_default():
    """The registry holds two Jacobian conventions. libsecp's doubling ends
    Z3 = Y*Z; the EFD formulas end Z3 = 2*Y*Z. Those denote the same affine
    point through different representatives, and by default the tool must call
    them different -- a call site that reuses Z, or tests it against 1, is not
    free to swap one for the other.

    Under the explicit projective gate they agree, and that is the ONLY way the
    two families can be compared at all. Both halves matter: the first keeps the
    default sound, the second keeps a whole third of the registry reachable.
    """
    from repsearch.pointforms import dbl_production, dbl_2009_l

    ref, cand = dbl_production(), dbl_2009_l()

    raw = slice_agreement(ref, cand, {}, FREE, samples=64)
    assert not raw.holds, "a change of representative must not pass the raw gate"
    assert raw.equivalence == "raw"
    assert set(raw.disagreement.keys) == {"X", "Y", "Z"}

    proj = slice_agreement(ref, cand, {}, FREE, samples=64, projective=True)
    assert proj.samples > 40
    assert proj.holds, proj.disagreement
    assert proj.equivalence == "projective"


def test_projective_gate_still_rejects_a_genuinely_different_map():
    """The weaker gate must stay a gate. co-Z off its slice is a different map,
    not a rescaled one, so loosening raw equality to projective equality must
    not let it through."""
    a = slice_agreement(madd_production(), zaddu_sum_only(), ZMAP, FREE,
                        samples=128, projective=True)
    assert a.samples > 100
    assert not a.holds, "projective equality is not an excuse to drop a precondition"


def test_the_shipped_formulas_are_the_registry_minimum():
    """The point of the sweep is to find something cheaper than what is shipped.
    Pinning the current answer -- nothing is -- means a future formula that beats
    a production one shows up as a test failure rather than going unnoticed.

    Ties count as held: madd_prod_s2_reassoc reaches the same weight as the
    shipped mixed add by reassociating one product, which is a draw and not a
    win. Only a strictly cheaper formula should break this test.

    This also guards the cost model: it was the corrected sub weight that put
    the in-tree doubling and mixed add back on top, and a regression there would
    silently re-rank them.
    """
    from repsearch import pointforms

    by_role = {}
    for name in dir(pointforms):
        fn = getattr(pointforms, name)
        if name.startswith("_") or not callable(fn):
            continue
        if not name.startswith(("madd", "dbl", "mdbl", "zaddu")):
            continue
        try:
            f = fn()
        except Exception:
            continue
        ins = set(f.inputs)
        role = "dbl" if ins <= {"X", "Y", "Z"} else ("madd" if "Z1" in ins else "cozadd")
        by_role.setdefault(role, []).append((f.cost().weighted, f.name, f.note or ""))

    for role in ("dbl", "madd"):
        entries = sorted(by_role[role])
        best = entries[0][0]
        tied = [e for e in entries if e[0] <= best + 1e-9]
        assert any("in_tree" in e[2] for e in tied), (role, tied)
