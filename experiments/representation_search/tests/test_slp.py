"""The cost model's own invariants.

Ranking candidates is the only thing the weights are for, so a weight that is
wrong does not produce a wrong number -- it produces a wrong winner, and a wrong
winner reads exactly like a result.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))




def test_sub_costs_exactly_neg_plus_add():
    """FE52 has no fused subtract -- only add_assign and negate_assign. So the
    two ways of spelling a subtraction must cost the same, on both metrics.

    They did not, and the gap silently manufactured "wins": a registry sweep
    ranked four formulas above the shipped ones on nothing but this discount,
    with identical multiply and square counts. A cost model that prefers one
    spelling of the same instruction sequence is worse than no cost model,
    because its output looks like a measurement.
    """
    from repsearch.slp import DEFAULT_WEIGHTS, Builder

    assert DEFAULT_WEIGHTS["sub"] == DEFAULT_WEIGHTS["neg"] + DEFAULT_WEIGHTS["add"]

    a = Builder("X", "Y")
    a_out = a.sub("X", "Y")
    prog_sub = a.build("sub_form", {"R": a_out}, "spelled with sub")

    b = Builder("X", "Y")
    b_out = b.add("X", b.neg("Y"))
    prog_neg = b.build("neg_form", {"R": b_out}, "spelled with neg + add")

    cs, cn = prog_sub.cost(), prog_neg.cost()
    assert cs.weighted == cn.weighted, (cs.weighted, cn.weighted)
    assert cs.depth == cn.depth, (cs.depth, cn.depth)
