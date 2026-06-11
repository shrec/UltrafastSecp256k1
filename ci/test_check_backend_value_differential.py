#!/usr/bin/env python3
"""
Self-test for ci/check_backend_value_differential.py — DON'T TRUST, VERIFY.

The live CPU<->GPU run needs a GPU, but the two decision layers are deterministic and are
proven here: (1) the byte comparator MUST flag a one-element divergence and a length
mismatch, and MUST pass identical vectors; (2) the coverage check MUST block a 'covered'
op whose differential probe is unwired, and pass when the real ledger is wired.
"""
import importlib.util
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load():
    spec = importlib.util.spec_from_file_location(
        "bvd", os.path.join(ROOT, "ci", "check_backend_value_differential.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main() -> int:
    m = _load()
    fails = []

    # 0. The real repo ledger must pass (coverage not spuriously red).
    if m.run() != 0:
        fails.append("real BACKEND_DIFFERENTIAL_OPS.json coverage should pass (exit 0)")

    # 1. Comparator: identical vectors -> match.
    if m.classify(m.diff_outputs(["aa", "bb", "cc"], ["aa", "bb", "cc"])) != "match":
        fails.append("identical CPU/GPU outputs must classify as match")

    # 2. Comparator: a single differing element -> divergence (and pinpoints the index).
    mm = m.diff_outputs(["aa", "bb", "cc"], ["aa", "bX", "cc"])
    if m.classify(mm) != "divergence" or not any(i == 1 for (i, _, _) in mm):
        fails.append("a one-element CPU/GPU value divergence must be flagged at its index")

    # 3. Comparator: length mismatch (GPU dropped a slot) -> divergence.
    if m.classify(m.diff_outputs(["aa", "bb"], ["aa"])) != "divergence":
        fails.append("a length mismatch (dropped GPU output) must be a divergence")

    # 4. Coverage: a 'covered' op whose probe is unwired MUST block.
    bad = {"ops": [{"id": "x", "op": "x", "claim": "x",
                    "probe_module": "DOES_NOT_EXIST", "probe_run_symbol": "test_DOES_NOT_EXIST_run",
                    "status": "covered"}]}
    blk, _, _ = m.check_coverage(bad, runner_src="")
    if not blk:
        fails.append("a 'covered' op with an unwired differential probe MUST block")

    # 5. Coverage: a 'covered' op with no probe declared MUST block.
    bad2 = {"ops": [{"id": "x", "op": "x", "claim": "x",
                     "probe_module": None, "probe_run_symbol": None, "status": "covered"}]}
    blk2, _, _ = m.check_coverage(bad2, runner_src="")
    if not blk2:
        fails.append("a 'covered' op with no probe MUST block")

    # 6. Coverage: a wired covered op passes; a roadmap op never blocks.
    good = {"ops": [
        {"id": "ok", "op": "x", "claim": "x", "probe_module": "mod_x",
         "probe_run_symbol": "test_mod_x_run", "status": "covered"},
        {"id": "later", "op": "x", "claim": "x", "probe_module": None,
         "probe_run_symbol": None, "status": "roadmap"}]}
    blk3, cov, road = m.check_coverage(good, runner_src='"mod_x" test_mod_x_run')
    if blk3 or len(cov) != 1 or len(road) != 1:
        fails.append("a wired covered op + a roadmap op must pass coverage")

    print("=" * 60)
    if fails:
        print("  check_backend_value_differential SELF-TEST: FAILED")
        for f in fails:
            print("   - " + f)
        print("=" * 60)
        return 1
    print("  check_backend_value_differential SELF-TEST PASSED")
    print("  (byte comparator proven to flag value/length divergence; coverage blocks unwired probes)")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
