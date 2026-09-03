#!/usr/bin/env python3
"""Cross-formula magnitude compatibility.

Individually, every candidate here reaches a magnitude fixed point and never
needs an inserted normalize -- FE52 has 12 bits of headroom and mul/sqr resets
the magnitude to 1.  So magnitude does NOT decide between these formulas on
cost grounds.

What it DOES decide is interoperability.  point.cpp hardcodes the declared
magnitude at every negate() call site:

    p.x.negate(8)   // asserts magnitude(X) <= 8
    p.y.negate(4)   // asserts magnitude(Y) <= 4   (GEJ_Y_MAG_MAX)

negate(m) computes (m+1)*p - a limbwise.  If the true magnitude of `a` exceeds
m, the subtraction UNDERFLOWS and the result is silently wrong -- no assertion,
no crash, a valid-looking field element with the wrong value.

So a doubling formula cannot be swapped in on its own: if its outputs exceed
the magnitudes the *consuming* formula declares, the consumer must be updated
in the same change.  This tool reports exactly which pairings are safe.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from repsearch import pointforms

# Declared magnitudes at the negate() call sites in src/cpu/src/point.cpp.
# Only X and Y are ever passed to negate(); Z is only ever multiplied, so it is
# bounded by the multiply accumulator, not by a declared negate magnitude.
# Listing a constraint that does not exist would manufacture a false finding.
DECLARED = {"X": 8, "Y": 4}
START = {"X": 1, "Y": 1, "Z": 1, "X1": 1, "Y1": 1, "Z1": 1, "X2": 1, "Y2": 1}


def steady_state(slp, affine=("X2", "Y2")):
    mags = dict(START)
    for _ in range(32):
        outs = slp.output_magnitudes(mags)
        nxt = dict(mags)
        for k, v in outs.items():
            for cand in (k, k + "1"):
                if cand in nxt and cand not in affine:
                    nxt[cand] = v
        if nxt == mags:
            return outs
        mags = nxt
    return slp.output_magnitudes(mags)


def main():
    print("Cross-formula magnitude compatibility against point.cpp's declared")
    print("negate() magnitudes: X<=%d, Y<=%d  (Z is never negated, only multiplied)"
          % (DECLARED["X"], DECLARED["Y"]))
    print("Exceeding a declared magnitude is a SILENT WRONG ANSWER, not a crash.\n")
    groups = [("doubling", pointforms.DOUBLING),
              ("doubling_z1", pointforms.DOUBLING_Z1),
              ("mixed_add", pointforms.MIXED_ADD)]
    unsafe = 0
    for label, group in groups:
        print("== %s ==" % label)
        for name, fn in group.items():
            slp = fn()
            outs = steady_state(slp)
            bad = {k: (outs[k], DECLARED[k]) for k in DECLARED
                   if k in outs and outs[k] > DECLARED[k]}
            if bad:
                unsafe += 1
                detail = ", ".join("%s=%d > declared %d" % (k, v[0], v[1]) for k, v in bad.items())
                print("  UNSAFE  %-28s steady X%d/Y%d/Z%d  -- %s"
                      % (name, outs["X"], outs["Y"], outs["Z"], detail))
                print("          %-28s drop-in would underflow negate(); the consuming"
                      % "")
                print("          %-28s formula's negate() arguments must change too." % "")
            else:
                print("  safe    %-28s steady X%d/Y%d/Z%d"
                      % (name, outs["X"], outs["Y"], outs["Z"]))
        print()
    print("%d formula(s) are not drop-in safe at the currently declared magnitudes." % unsafe)


if __name__ == "__main__":
    main()
