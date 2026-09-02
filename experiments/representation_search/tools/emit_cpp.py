#!/usr/bin/env python3
"""Emit a candidate formula as C++ FieldElement52 code.

Magnitudes are propagated through the program so that every negate() gets the
argument the FE52 contract requires -- a hand-written candidate almost always
gets these wrong, and a wrong negate() magnitude is a silent wrong answer.

    python3 tools/emit_cpp.py dbl_prod_alt_sign
    python3 tools/emit_cpp.py --list
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from repsearch import pointforms

JAC_BUDGET = {"X": 7, "Y": 4, "Z": 1, "X1": 7, "Y1": 4, "Z1": 1, "X2": 1, "Y2": 1}

ALL = {}
ALL.update(pointforms.DOUBLING)
ALL.update(pointforms.DOUBLING_Z1)
ALL.update(pointforms.MIXED_ADD)

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] == "--list":
        for k in ALL:
            print(k)
        raise SystemExit(0)
    name = sys.argv[1]
    if name not in ALL:
        print("unknown formula %r; use --list" % name)
        raise SystemExit(2)
    slp = ALL[name]()
    _m, violations, peak = slp.magnitudes(JAC_BUDGET)
    print("// peak magnitude %d, outputs %s" % (peak, slp.output_magnitudes(JAC_BUDGET)))
    for v in violations:
        print("// MAGNITUDE WARNING %s: %s" % (v.kind, v.detail))
    print(slp.to_cpp(input_mags=JAC_BUDGET))
