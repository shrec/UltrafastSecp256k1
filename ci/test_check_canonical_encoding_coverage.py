#!/usr/bin/env python3
"""
Self-test for ci/check_canonical_encoding_coverage.py — DON'T TRUST, VERIFY.

Prove the gate blocks a 'covered' encoding whose non-canonical-rejection probe is unwired,
and passes a wired covered entry + roadmap. The real ledger must pass.
"""
import importlib.util
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load():
    spec = importlib.util.spec_from_file_location(
        "cec", os.path.join(ROOT, "ci", "check_canonical_encoding_coverage.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main() -> int:
    m = _load()
    fails = []

    if m.run() != 0:
        fails.append("real CANONICAL_ENCODING_LEDGER.json should pass (exit 0)")

    bad = {"encodings": [{"id": "x", "encoding": "x", "claim": "x",
                          "probe_module": "DOES_NOT_EXIST",
                          "probe_run_symbol": "test_DOES_NOT_EXIST_run", "status": "covered"}]}
    blk, _, _ = m.check_coverage(bad, runner_src="")
    if not blk:
        fails.append("a 'covered' encoding with an unwired probe MUST block")

    bad2 = {"encodings": [{"id": "x", "encoding": "x", "claim": "x",
                           "probe_module": None, "probe_run_symbol": None, "status": "covered"}]}
    blk2, _, _ = m.check_coverage(bad2, runner_src="")
    if not blk2:
        fails.append("a 'covered' encoding with no probe MUST block")

    good = {"encodings": [
        {"id": "ok", "encoding": "x", "claim": "x", "probe_module": "mod_x",
         "probe_run_symbol": "test_mod_x_run", "status": "covered"},
        {"id": "later", "encoding": "x", "claim": "x", "probe_module": None,
         "probe_run_symbol": None, "status": "roadmap"}]}
    blk3, cov, road = m.check_coverage(good, runner_src='"mod_x" test_mod_x_run')
    if blk3 or len(cov) != 1 or len(road) != 1:
        fails.append("a wired covered encoding + roadmap must pass")

    print("=" * 60)
    if fails:
        print("  check_canonical_encoding_coverage SELF-TEST: FAILED")
        for f in fails:
            print("   - " + f)
        print("=" * 60)
        return 1
    print("  check_canonical_encoding_coverage SELF-TEST PASSED")
    print("  (gate blocks unwired covered probes; passes wired + roadmap)")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
