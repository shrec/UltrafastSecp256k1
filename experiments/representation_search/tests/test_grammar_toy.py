"""Lock-in tests: the tool must reproduce the originating grammar experiments.

If these ever fail, the equivalence engine has regressed and NO result produced
by this experiment can be trusted.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from repsearch.grammar import (PRECEDENCE_ORDERS, collapse, conventional,
                               enumerate_grammars)
from repsearch.expr import parse
from repsearch.poly import RatFunc


def canonical_forms(source):
    return sorted(str(c.value) for c in collapse(enumerate_grammars(source)))


def test_twenty_four_orders():
    assert len(PRECEDENCE_ORDERS) == 24
    assert len(set(PRECEDENCE_ORDERS)) == 24


def test_sine_argument_collapses_to_eight():
    forms = canonical_forms("2*t+1-t/2")
    assert len(forms) == 8, forms
    # The eight affine arguments from the originating experiment.
    assert set(forms) == {
        "1",                    # the variable vanishes entirely
        "(t + 2) / (2)",        # t/2 + 1
        "(3*t + 4) / (2)",      # (3/2)t + 2
        "t + 2",
        "(t + 1) / (2)",        # t/2 + 1/2
        "(3*t + 1) / (2)",      # (3/2)t + 1/2
        "t + 1",
        "(3*t + 2) / (2)",      # (3/2)t + 1
    }


def test_sine_constant_class_has_four_grammars():
    classes = {str(c.value): c for c in collapse(enumerate_grammars("2*t+1-t/2"))}
    assert len(classes["1"].members) == 4


def test_parabola_collapses_to_eight_including_identity():
    forms = canonical_forms("2*x*x+1-x/2")
    assert len(forms) == 8, forms
    # A nominally quadratic token sequence that degenerates to y = x.
    assert "x" in forms
    assert "x^2 + x" in forms          # x(x+1)
    assert "x^2 + 2*x" in forms        # x(x+2)
    assert "(2*x^2 + x) / (2)" in forms       # x(x+1/2)
    assert "(4*x^2 + 3*x) / (2)" in forms     # x(4x+3)/2
    assert "(4*x^2 - x + 2) / (2)" in forms   # the conventional reading


def test_conventional_grammar_is_the_expected_reading():
    ast = parse("2*t+1-t/2", conventional())
    assert ast.to_ratfunc().equals(
        RatFunc.const(3) * RatFunc.var("t") / RatFunc.const(2) + RatFunc.const(1))


def test_parenthesised_source_is_grammar_invariant():
    # Full parenthesisation removes the grammar degree of freedom entirely.
    forms = canonical_forms("((2*t)+1)-(t/2)")
    assert len(forms) == 1, forms
