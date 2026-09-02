#!/usr/bin/env python3
"""Prove-or-reject every candidate point formula against the reference oracle.

Usage:  python3 tools/run_point_equiv.py [--points N] [--json OUT]

Exit code is non-zero if any candidate that claims equivalence fails.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from repsearch import pointforms
from repsearch.equiv import check_doubling, check_mixed_add, degenerate_probe
from repsearch.field import curve_points

# The engine's Jacobian magnitude budget, read from point.cpp's own annotations:
#   GEJ_Y_MAG_MAX = 4, "jac52_add x max: 7", Z is kept normalized (mag 1).
# A formula is "magnitude-closed" when its outputs fit back inside this budget,
# i.e. it can be iterated in a scalar-multiplication loop with no inserted
# normalize.  A formula that is NOT closed needs extra normalization work that
# no operation count reveals -- this is the single most under-modelled cost in
# published M/S formula comparisons.
JAC_BUDGET = {"X": 7, "Y": 4, "Z": 1, "X1": 7, "Y1": 4, "Z1": 1, "X2": 1, "Y2": 1}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", type=int, default=24)
    ap.add_argument("--z-variants", type=int, default=4)
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    pts = curve_points(args.points)
    pairs = list(zip(pts, pts[1:] + pts[:1]))
    report = {"doubling": [], "doubling_z1": [], "mixed_add": []}
    failures = 0

    def emit(section, slp, res, probes):
        nonlocal failures
        cost = slp.cost()
        _mags, violations, peak = slp.magnitudes(JAC_BUDGET)
        out_mags = slp.output_magnitudes(JAC_BUDGET)
        closed = all(out_mags[k] <= JAC_BUDGET[k] for k in out_mags if k in JAC_BUDGET)
        lam = res.uniform_scaling
        status = "OK " if res.ok else "FAIL"
        if not res.ok:
            failures += 1
        lam_s = "L=%s" % lam if lam is not None else ("L=varies" if res.scaling_factors else "-")
        print("  %-4s %-26s %s %-8s peak=%-3d out=%-12s %s" %
              (status, slp.name, cost.summary(), lam_s, peak,
               "/".join("%s%d" % (k, out_mags[k]) for k in ("X", "Y", "Z")),
               "CLOSED" if closed else "NEEDS-NORMALIZE"))
        for v in violations:
            print("         MAGNITUDE %s: %s" % (v.kind, v.detail))
        for note in probes:
            print("         degenerate: %s" % note)
        if not res.ok:
            for m in res.mismatches[:2]:
                print("         MISMATCH %s got=%s want=%s" % (m.case, m.got, m.expected))
        report[section].append({
            "name": slp.name, "note": slp.note, "ok": res.ok, "cases": res.cases,
            "op_counts": cost.counts, "weighted_cost": cost.weighted,
            "critical_depth": cost.depth, "max_live": cost.max_live,
            "uniform_scaling": str(lam) if lam is not None else None,
            "peak_magnitude": peak,
            "output_magnitudes": out_mags,
            "magnitude_closed": closed,
            "magnitude_violations": [v._asdict() for v in violations],
            "degenerate_behaviour": probes,
            "mismatches": len(res.mismatches),
        })

    ref_dbl = pointforms.dbl_production()
    print("\nDOUBLING (general Z)   reference = %s" % ref_dbl.name)
    for name, fn in pointforms.DOUBLING.items():
        slp = fn()
        res = check_doubling(slp, pts, args.z_variants, reference=ref_dbl)
        emit("doubling", slp, res, degenerate_probe(slp, "dbl"))

    ref_z1 = pointforms.dbl_z1_production()
    print("\nDOUBLING (Z == 1)      reference = %s" % ref_z1.name)
    for name, fn in pointforms.DOUBLING_Z1.items():
        slp = fn()
        res = check_doubling(slp, pts, reference=ref_z1, z_one_only=True)
        emit("doubling_z1", slp, res, degenerate_probe(slp, "dbl"))

    ref_madd = pointforms.madd_production()
    print("\nMIXED ADDITION         reference = %s" % ref_madd.name)
    for name, fn in pointforms.MIXED_ADD.items():
        slp = fn()
        res = check_mixed_add(slp, pairs, args.z_variants, reference=ref_madd)
        emit("mixed_add", slp, res, degenerate_probe(slp, "madd"))

    print("\n%d candidate(s) failed equivalence." % failures if failures
          else "\nAll candidates are exactly equivalent to the reference group law.")

    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(report, f, indent=2)
        print("wrote %s" % args.json)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
