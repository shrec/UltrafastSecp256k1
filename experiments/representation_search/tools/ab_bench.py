#!/usr/bin/env python3
"""Interleaved A/B benchmark driver for representation-search candidates.

Follows the repository's performance protocol:

  * baseline and candidate are built from the SAME source tree, differing only
    in a -D flag, so the source tree itself is not a confounder;
  * runs are INTERLEAVED A/B/B/A so thermal drift and ordering bias cancel
    rather than accumulating into one arm;
  * bench_unified already performs 11 internal passes with IQR trimming per
    invocation, so each invocation is one stabilised sample, not a raw timing;
  * the verdict rule is range separation, not median comparison:
        improvement  iff  candidate_max < baseline_min
        regression   iff  candidate_min > baseline_max
        otherwise    inconclusive -- and inconclusive is NOT a green light.

Usage:
    python3 tools/ab_bench.py \
        --baseline out/repsearch-baseline/src/cpu/bench_unified \
        --candidate out/repsearch-cand1/src/cpu/bench_unified \
        --rounds 3 --core 0 \
        --out out/representation-search-cpu/ab_dbl_alt_sign.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time

# bench_unified prints lines like:
#   Point::dbl (jac52_double)                          57.55 ns
# Tolerant to column drift: take the last float on a line that has a label.
LINE_RE = re.compile(r"^\s*(?P<name>\S.*?\S)\s{2,}(?P<num>[0-9]+\.[0-9]+)\s*(?:ns|ns/op)?\s*$")


def _flatten_json(doc) -> dict:
    """bench_unified --json: pull every {name, ns} record, keyed by section+name."""
    out = {}

    def walk(node, section=""):
        if isinstance(node, dict):
            if "name" in node and "ns" in node:
                try:
                    key = node["name"]
                    sec = node.get("section", section)
                    out["%s | %s" % (sec, key) if sec else key] = float(node["ns"])
                except (TypeError, ValueError):
                    pass
            for k, v in node.items():
                walk(v, node.get("section", section))
        elif isinstance(node, list):
            for v in node:
                walk(v, section)

    walk(doc)
    return out


def run_once(binary: str, core: int | None, timeout: int, scratch: str, tag: str) -> dict:
    """One invocation.  bench_unified already averages 11 internal passes with
    IQR trimming, so this returns one stabilised sample per metric."""
    json_path = os.path.join(scratch, "bench_%s.json" % tag)
    cmd = []
    if core is not None and shutil.which("taskset"):
        cmd += ["taskset", "-c", str(core)]
    if shutil.which("nice"):
        cmd += ["nice", "-n", "-20"]
    cmd += [binary, "--json", json_path]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

    if os.path.exists(json_path):
        try:
            with open(json_path) as f:
                parsed = _flatten_json(json.load(f))
            if parsed:
                return parsed
        except Exception as exc:
            sys.stderr.write("json parse failed (%s), falling back to stdout\n" % exc)

    out = {}
    for line in (proc.stdout or "").splitlines():
        m = LINE_RE.match(line)
        if m:
            try:
                out[m.group("name")] = float(m.group("num"))
            except ValueError:
                pass
    if not out:
        sys.stderr.write("no parsable timings from %s\nstdout head:\n%s\n"
                         % (binary, "\n".join((proc.stdout or "").splitlines()[:25])))
    return out


def summarise(samples: list) -> dict:
    if not samples:
        return {}
    return {
        "n": len(samples),
        "median": statistics.median(samples),
        "min": min(samples),
        "max": max(samples),
        "spread_pct": (max(samples) - min(samples)) / statistics.median(samples) * 100.0
        if statistics.median(samples) else 0.0,
    }


def verdict(base: dict, cand: dict) -> str:
    """Range separation. Overlapping ranges are inconclusive, never a win."""
    if not base or not cand:
        return "no-data"
    if cand["max"] < base["min"]:
        return "IMPROVEMENT"
    if cand["min"] > base["max"]:
        return "REGRESSION"
    return "inconclusive"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--rounds", type=int, default=3,
                    help="A/B/B/A rounds; each round is 2 baseline + 2 candidate runs")
    ap.add_argument("--core", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--out", default="")
    ap.add_argument("--only", default="",
                    help="comma-separated substrings; report only matching rows")
    args = ap.parse_args()

    for path in (args.baseline, args.candidate):
        if not os.access(path, os.X_OK):
            print("not executable: %s" % path)
            return 2

    scratch = os.path.join(os.path.dirname(os.path.abspath(args.out or ".")), "bench_raw")
    os.makedirs(scratch, exist_ok=True)

    base_runs, cand_runs = [], []
    order = []
    for r in range(args.rounds):
        # A B B A -- ordering bias cancels within each round.
        for slot, tag in enumerate(("A", "B", "B", "A")):
            t0 = time.time()
            label = "r%d_%s%d" % (r, tag, slot)
            if tag == "A":
                base_runs.append(run_once(args.baseline, args.core, args.timeout, scratch, label))
            else:
                cand_runs.append(run_once(args.candidate, args.core, args.timeout, scratch, label))
            order.append({"round": r, "arm": tag, "seconds": round(time.time() - t0, 2)})
            print("  round %d  arm %s  %.1fs" % (r, tag, order[-1]["seconds"]), flush=True)

    names = set()
    for d in base_runs + cand_runs:
        names |= set(d)
    filters = [f.strip() for f in args.only.split(",") if f.strip()]
    if filters:
        names = {n for n in names if any(f in n for f in filters)}

    rows = []
    for name in sorted(names):
        b = summarise([d[name] for d in base_runs if name in d])
        c = summarise([d[name] for d in cand_runs if name in d])
        if not b or not c:
            continue
        v = verdict(b, c)
        delta = (c["median"] - b["median"]) / b["median"] * 100.0 if b["median"] else 0.0
        rows.append({"name": name, "baseline": b, "candidate": c,
                     "delta_pct": delta, "verdict": v})

    rows.sort(key=lambda r: (r["verdict"] != "IMPROVEMENT", r["verdict"] != "REGRESSION",
                             abs(r["delta_pct"]) * -1))

    print("\n%-46s %10s %10s %8s  %s" % ("metric", "base med", "cand med", "delta%", "verdict"))
    print("-" * 92)
    for r in rows:
        print("%-46s %10.2f %10.2f %+7.2f%%  %-13s base[%.2f..%.2f] cand[%.2f..%.2f]" %
              (r["name"][:46], r["baseline"]["median"], r["candidate"]["median"],
               r["delta_pct"], r["verdict"],
               r["baseline"]["min"], r["baseline"]["max"],
               r["candidate"]["min"], r["candidate"]["max"]))

    noisy = [r for r in rows if r["baseline"]["spread_pct"] > 3.0]
    if noisy:
        print("\n%d metric(s) had baseline run-to-run spread > 3%%; their verdicts are"
              " not trustworthy on this machine:" % len(noisy))
        for r in noisy[:10]:
            print("   %-46s spread %.1f%%" % (r["name"][:46], r["baseline"]["spread_pct"]))

    improved = [r["name"] for r in rows if r["verdict"] == "IMPROVEMENT"]
    regressed = [r["name"] for r in rows if r["verdict"] == "REGRESSION"]
    print("\nIMPROVEMENT: %d   REGRESSION: %d   inconclusive: %d"
          % (len(improved), len(regressed),
             sum(1 for r in rows if r["verdict"] == "inconclusive")))

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({
                "baseline_binary": os.path.abspath(args.baseline),
                "candidate_binary": os.path.abspath(args.candidate),
                "rounds": args.rounds, "core": args.core,
                "run_order": order, "rows": rows,
                "protocol": "interleaved A/B/B/A; bench_unified does 11 internal "
                            "passes with IQR trimming per invocation; verdict by "
                            "range separation, overlapping ranges are inconclusive",
            }, f, indent=2)
        print("wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
