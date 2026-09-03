#!/usr/bin/env python3
"""How many times can a formula be iterated before a normalize is REQUIRED?

Operation counts say nothing about this.  A formula whose output magnitudes
exceed its input magnitudes grows every iteration until it hits the FE52
multiply-accumulator bound, at which point the answer becomes silently wrong
unless a normalize is inserted.  This tool finds either the fixed point (the
formula is self-sustaining and never needs one) or the iteration count at which
the first normalize becomes mandatory.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from repsearch import pointforms
from repsearch.slp import MUL_ACC_LIMIT, MUL_INT_LIMIT

START = {"X": 1, "Y": 1, "Z": 1, "X1": 1, "Y1": 1, "Z1": 1, "X2": 1, "Y2": 1}
MAX_ITERS = 64


def trace(slp, start=None, affine_inputs=("X2", "Y2")):
    """Feed the outputs back into the inputs and watch the magnitudes."""
    mags = dict(start or START)
    history = []
    for i in range(1, MAX_ITERS + 1):
        try:
            _m, violations, peak = slp.magnitudes(mags)
        except Exception as exc:
            return history, "error: %s" % exc
        hard = [v for v in violations if "overflow" in v.detail or "MAG" in v.kind]
        outs = slp.output_magnitudes(mags)
        history.append((i, dict(mags), outs, peak, len(violations)))
        if any("overflow" in v.detail for v in violations):
            return history, "NORMALIZE REQUIRED at iteration %d (accumulator overflow)" % i
        nxt = dict(mags)
        for key, val in outs.items():
            for cand in (key, key + "1"):
                if cand in nxt and cand not in affine_inputs:
                    nxt[cand] = val
        if nxt == mags:
            return history, "FIXED POINT at iteration %d: %s -- never needs a normalize" % (
                i, {k: v for k, v in outs.items()})
        mags = nxt
    return history, "did not settle within %d iterations" % MAX_ITERS


def main():
    groups = [("doubling", pointforms.DOUBLING),
              ("doubling_z1", pointforms.DOUBLING_Z1),
              ("mixed_add", pointforms.MIXED_ADD)]
    print("FE52 magnitude fixed-point analysis")
    print("mul accumulator bound: 5*ma*mb < %d ; mul_int bound: m*k < %d\n"
          % (MUL_ACC_LIMIT, MUL_INT_LIMIT))
    for label, group in groups:
        print("== %s ==" % label)
        for name, fn in group.items():
            slp = fn()
            history, verdict = trace(slp)
            path = " -> ".join("%d/%d/%d" % (h[2].get("X", 0), h[2].get("Y", 0), h[2].get("Z", 0))
                               for h in history[:6])
            print("  %-28s %s" % (name, verdict))
            print("  %-28s out X/Y/Z per iter: %s%s" % ("", path,
                                                        " ..." if len(history) > 6 else ""))
        print()


if __name__ == "__main__":
    main()
