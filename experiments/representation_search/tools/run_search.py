#!/usr/bin/env python3
"""Run the automatic representation search over the engine's point formulas.

This is the core experiment: generate every equivalence-preserving program the
rewrite rules can reach, verify each by exact evaluation in F_p, reject anything
that would overflow a magnitude, then rank what survives.

    python3 tools/run_search.py --depth 3 --beam 40
    python3 tools/run_search.py --formula dbl_production --emit-top 3
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from repsearch import pointforms
from repsearch.rewrite import search

# Jacobian input magnitudes and the magnitudes point.cpp DECLARES at its
# negate() call sites.  A candidate that exceeds a declared magnitude is
# rejected by the search, not merely flagged -- it would corrupt silently.
INPUT_MAGS = {"X": 7, "Y": 4, "Z": 1, "X1": 7, "Y1": 4, "Z1": 1, "X2": 1, "Y2": 1}
DECLARED = {"X": 8, "Y": 4}

ALL = {}
ALL.update(pointforms.DOUBLING)
ALL.update(pointforms.DOUBLING_Z1)
ALL.update(pointforms.MIXED_ADD)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--formula", default="", help="only this formula")
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--beam", type=int, default=40)
    ap.add_argument("--corpus", type=int, default=16)
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--emit-top", type=int, default=0, help="emit N best as C++")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    targets = {args.formula: ALL[args.formula]} if args.formula else ALL
    report = {"depth": args.depth, "beam": args.beam, "families": []}

    for name, fn in targets.items():
        ref = fn()
        print("\n=== %s ===" % name)
        print("  reference: %s" % ref.cost().summary())
        t0 = time.time()
        res = search(ref, INPUT_MAGS, DECLARED, depth=args.depth, beam=args.beam,
                     corpus_size=args.corpus, progress=print)
        elapsed = time.time() - t0

        base = res.reference
        better = [c for c in res.candidates
                  if (c.weighted, c.depth, c.live) < (base.weighted, base.depth, base.live)]
        equal_cost = [c for c in res.candidates if abs(c.weighted - base.weighted) < 1e-9]

        print("  %d rewrites tried, %d distinct equivalent programs kept, "
              "%d unsound rejected, %d magnitude-rejected  (%.1fs)"
              % (res.explored, len(res.candidates), res.rejected_unsound,
                 res.rejected_magnitude, elapsed))
        print("  %d strictly better than the reference on (weighted, depth, live)"
              % len(better))
        print("  %d distinct programs compute the SAME function at the SAME weighted cost"
              % len(equal_cost))

        axes = collections.Counter()
        for c in res.candidates:
            for k in c.kinds:
                axes[k] += 1
        if axes:
            print("  rewrite axes used: %s"
                  % ", ".join("%s=%d" % kv for kv in axes.most_common()))

        print("\n  %-4s %-26s %8s %8s %6s  %s"
              % ("#", "ops", "weighted", "depth", "live", "path"))
        for i, c in enumerate(res.candidates[:args.top]):
            counts = c.counts
            ops = "+".join("%d%s" % (counts[k], lbl) for k, lbl in
                           (("mul", "M"), ("sqr", "S"), ("add", "A"), ("sub", "A-"),
                            ("neg", "N"), ("mulint", "I"), ("half", "H"))
                           if counts.get(k))
            mark = "  <-- reference" if c.path == () else ""
            print("  %-4d %-26s %8.3f %8.3f %6d  %s%s"
                  % (i, ops, c.weighted, c.depth, c.live,
                     " > ".join(c.path) if c.path else "(unchanged)", mark))

        report["families"].append({
            "formula": name,
            "reference": {"weighted": base.weighted, "depth": base.depth,
                          "live": base.live, "counts": base.counts},
            "explored": res.explored,
            "kept": len(res.candidates),
            "rejected_unsound": res.rejected_unsound,
            "rejected_magnitude": res.rejected_magnitude,
            "strictly_better": len(better),
            "same_cost_distinct_programs": len(equal_cost),
            "elapsed_s": round(elapsed, 2),
            "top": [{"weighted": c.weighted, "depth": c.depth, "live": c.live,
                     "counts": c.counts, "path": list(c.path), "kinds": list(c.kinds)}
                    for c in res.candidates[:args.top]],
        })

        for c in res.candidates[:args.emit_top]:
            if c.path == ():
                continue
            print("\n  // %s" % " > ".join(c.path))
            print(c.slp.to_cpp(fn_name="%s_cand" % name, input_mags=INPUT_MAGS))

    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(report, f, indent=2)
        print("\nwrote %s" % args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
