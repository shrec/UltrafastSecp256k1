#!/usr/bin/env python3
"""
Self-test for ci/check_metamorphic_coverage.py — DON'T TRUST, VERIFY.

We do not trust that the metamorphic-coverage gate works; we PROVE it blocks by
injecting violations and asserting it returns non-zero, and that the real ledger
returns zero. A gate that cannot be shown to block is not a gate.
"""
import importlib.util
import json
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load():
    spec = importlib.util.spec_from_file_location(
        "cmc", os.path.join(ROOT, "ci", "check_metamorphic_coverage.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _run_with_ledger(m, ledger_obj):
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    json.dump(ledger_obj, open(path, "w"))
    saved = m.LEDGER
    try:
        m.LEDGER = path
        return m.main()
    finally:
        m.LEDGER = saved
        os.remove(path)


def main() -> int:
    m = _load()
    fails = []

    # 1. The real ledger must currently PASS (the gate is not spuriously red).
    if m.main() != 0:
        fails.append("real METAMORPHIC_RELATIONS.json should pass (exit 0)")

    # 2. A 'covered' relation whose probe is NOT wired must BLOCK (exit 1).
    bad = {"relations": [{
        "id": "selftest-unwired", "protocol": "x", "transform": "x", "relation": "x",
        "probe_module": "metamorphic_DOES_NOT_EXIST",
        "probe_run_symbol": "test_metamorphic_DOES_NOT_EXIST_run", "status": "covered"}]}
    if _run_with_ledger(m, bad) == 0:
        fails.append("a 'covered' relation with an unwired probe MUST block (expected exit 1)")

    # 3. A 'covered' relation with NO probe declared must BLOCK.
    bad2 = {"relations": [{
        "id": "selftest-noprobe", "protocol": "x", "transform": "x", "relation": "x",
        "probe_module": None, "probe_run_symbol": None, "status": "covered"}]}
    if _run_with_ledger(m, bad2) == 0:
        fails.append("a 'covered' relation with no probe MUST block (expected exit 1)")

    # 4. A clean 'roadmap' relation (no probe yet) must NOT block (declared, being built).
    ok = {"relations": [{
        "id": "selftest-roadmap", "protocol": "x", "transform": "x", "relation": "x",
        "probe_module": None, "probe_run_symbol": None, "status": "roadmap"}]}
    if _run_with_ledger(m, ok) != 0:
        fails.append("a declared 'roadmap' relation must not block (exit 0)")

    print("=" * 60)
    if fails:
        print("  check_metamorphic_coverage SELF-TEST: FAILED")
        for f in fails:
            print("   - " + f)
        print("=" * 60)
        return 1
    print("  check_metamorphic_coverage SELF-TEST PASSED")
    print("  (gate proven to block unwired/undeclared probes; clean ledger passes)")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
