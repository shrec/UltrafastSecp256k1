#!/usr/bin/env python3
"""Compare every build variant from the final benchmark run.

Verdict rule is range separation over the rounds, the same rule used throughout
this search: overlapping ranges are inconclusive and inconclusive is never a win.
Controls (unmodified libsecp entries in bench_unified) are reported separately --
if a control separates, the run is not trustworthy and no verdict below it means
anything.
"""
import glob
import json
import os
import statistics
import sys


def flatten(doc):
    out = {}

    def walk(node, section=""):
        if isinstance(node, dict):
            if "name" in node and "ns" in node:
                try:
                    out["%s | %s" % (node.get("section", section), node["name"])] = float(node["ns"])
                except (TypeError, ValueError):
                    pass
            for _k, v in node.items():
                walk(v, node.get("section", section))
        elif isinstance(node, list):
            for v in node:
                walk(v, section)

    walk(doc)
    return out


def load(outdir):
    runs = {}
    for path in sorted(glob.glob(os.path.join(outdir, "r*-*.json"))):
        base = os.path.basename(path)[:-5]           # rN-variant
        variant = base.split("-", 1)[1]
        try:
            runs.setdefault(variant, []).append(flatten(json.load(open(path))))
        except Exception:
            pass
    return runs


def verdict(a, b):
    if len(a) < 3 or len(b) < 3:
        return "too-few-samples"
    if max(b) < min(a):
        return "IMPROVEMENT"
    if min(b) > max(a):
        return "REGRESSION"
    return "inconclusive"


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "out/representation-search-cpu/final"
    runs = load(outdir)
    if "baseline" not in runs:
        print("no baseline runs found in %s" % outdir)
        return 1
    base = runs["baseline"]
    print("variants: %s" % ", ".join("%s(%d runs)" % (k, len(v)) for k, v in sorted(runs.items())))
    print()

    CONTROL = ("libsecp", "OpenSSL", "SHA-256", "HKDF", "AEAD", "tx_weight")

    for name in sorted(k for k in runs if k != "baseline"):
        cand = runs[name]
        names = set().union(*base) & set().union(*cand)
        imp, reg, ctrl = [], [], []
        for metric in sorted(names):
            a = [r[metric] for r in base if metric in r]
            b = [r[metric] for r in cand if metric in r]
            if len(a) < 3 or len(b) < 3:
                continue
            am, bm = statistics.median(a), statistics.median(b)
            if am <= 0:
                continue
            v = verdict(a, b)
            row = (100.0 * (bm - am) / am, metric, am, bm, v)
            if any(c in metric for c in CONTROL):
                if v != "inconclusive":
                    ctrl.append(row)
            elif v == "IMPROVEMENT":
                imp.append(row)
            elif v == "REGRESSION":
                reg.append(row)

        print("=== %s vs baseline ===" % name)
        if ctrl:
            print("  !! %d CONTROL metric(s) separated -- this run is not trustworthy:" % len(ctrl))
            for d, m, am, bm, v in sorted(ctrl)[:5]:
                print("       %-46s %+7.2f%%  %s" % (m[-46:], d, v))
        imp.sort()
        reg.sort(key=lambda r: -r[0])
        print("  IMPROVEMENT: %d   REGRESSION: %d" % (len(imp), len(reg)))
        for d, m, am, bm, _v in imp[:12]:
            print("    %-46s %9.1f -> %9.1f  %+7.2f%%" % (m[-46:], am, bm, d))
        for d, m, am, bm, _v in reg[:8]:
            print("    REG %-42s %9.1f -> %9.1f  %+7.2f%%" % (m[-42:], am, bm, d))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
