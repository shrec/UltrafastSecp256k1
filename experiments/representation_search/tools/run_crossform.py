#!/usr/bin/env python3
"""Every point formula against every other, on declared input slices.

The beam search rewrites one program and can only ever produce programs over that
program's own inputs. This driver asks the question it cannot: does some OTHER
formula in the registry compute the same thing, and under what precondition?
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from repsearch import pointforms
from repsearch.crossform import sweep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", default="madd_production")
    ap.add_argument("--samples", type=int, default=256)
    args = ap.parse_args()

    builders = {n: getattr(pointforms, n) for n in dir(pointforms)
                if not n.startswith("_") and callable(getattr(pointforms, n))
                and n.startswith(("madd", "dbl", "mdbl", "zaddu"))}
    forms = {}
    for n, b in sorted(builders.items()):
        try:
            forms[n] = b()
        except Exception:
            pass
    if args.reference not in forms:
        print("unknown reference %r; have: %s" % (args.reference, ", ".join(sorted(forms))))
        return 2
    ref = forms[args.reference]

    # The slices are declared here, in the open, and each is a claim about when
    # the comparison is legitimate -- never inferred from whether it happens to pass.
    slices = [
        {},                 # everywhere: no precondition at all
        {"Z1": 1, "Z": 1},  # both points affine -- the table-building situation
    ]
    # Position five means different things in the two input lists, which is the
    # whole barrier: Z1 belongs to P alone, Z is shared by P and Q.
    var_maps = {"zaddu": {"Z": "Z"}, "zaddu_sum_only": {"Z": "Z"}}

    print("reference: %s  %s" % (ref.name, ref.inputs))
    print("           weighted %.2f  depth %.2f" % (ref.cost().weighted, ref.cost().depth))
    print()
    held = sweep(ref, list(forms.values()), slices, var_maps,
                 samples=args.samples, progress=print)
    print()
    if not held:
        print("no other formula in the registry agrees with the reference on any declared slice")
        return 0
    held.sort(key=lambda a: a.weighted_delta)
    print("%-24s %-18s %9s %9s  %s" % ("formula", "slice", "weighted", "depth", "extra outputs"))
    for a in held:
        sl = ", ".join("%s=%d" % kv for kv in sorted(a.pins.items())) or "everywhere"
        print("%-24s %-18s %+8.1f%% %+8.1f%%  %s"
              % (a.candidate_name, sl, 100 * a.weighted_delta, 100 * a.depth_delta,
                 ", ".join(a.extra_outputs) or "-"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
