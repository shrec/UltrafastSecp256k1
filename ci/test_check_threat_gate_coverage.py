#!/usr/bin/env python3
"""
Self-test for ci/check_threat_gate_coverage.py — DON'T TRUST, VERIFY.

The meta-gate enforces that we don't trust gates without proof. By the same rule,
we don't trust the meta-gate either: this proves it blocks false coverage claims.
"""
import importlib.util
import json
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load():
    spec = importlib.util.spec_from_file_location(
        "ctg", os.path.join(ROOT, "ci", "check_threat_gate_coverage.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _run(m, obj):
    fd, p = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    json.dump(obj, open(p, "w"))
    try:
        return m.run(p)
    finally:
        os.remove(p)


def main() -> int:
    m = _load()
    fails = []

    # 1. The real matrix must pass.
    if m.run(m.DEFAULT_MATRIX) != 0:
        fails.append("real THREAT_GATE_MATRIX.json should pass (exit 0)")

    # 2. 'verified' with a missing self-test file MUST block.
    bad = {"threats": [{"id": "x", "class": "c", "gate": "ci/check_ct_branches.py",
                        "self_test": "ci/test_DOES_NOT_EXIST.py", "status": "verified"}]}
    if _run(m, bad) == 0:
        fails.append("a 'verified' threat with a missing self-test MUST block")

    # 3. 'verified' with a missing GATE file MUST block.
    bad2 = {"threats": [{"id": "x", "class": "c", "gate": "ci/check_DOES_NOT_EXIST.py",
                         "self_test": "ci/test_check_ct_branches.py", "status": "verified"}]}
    if _run(m, bad2) == 0:
        fails.append("a 'verified' threat with a missing gate file MUST block")

    # 4. 'trusted' with a missing gate file MUST block.
    bad3 = {"threats": [{"id": "x", "class": "c", "gate": "ci/check_DOES_NOT_EXIST.py",
                         "self_test": None, "status": "trusted"}]}
    if _run(m, bad3) == 0:
        fails.append("a 'trusted' threat claiming a missing gate file MUST block")

    # 5. A clean 'gap' (no gate) must NOT block (declared hole, being built).
    ok = {"threats": [{"id": "g", "class": "c", "gate": None,
                       "self_test": None, "status": "gap", "note": "n"}]}
    if _run(m, ok) != 0:
        fails.append("a declared 'gap' must not block (exit 0)")

    print("=" * 60)
    if fails:
        print("  check_threat_gate_coverage SELF-TEST: FAILED")
        for f in fails:
            print("   - " + f)
        print("=" * 60)
        return 1
    print("  check_threat_gate_coverage SELF-TEST PASSED")
    print("  (meta-gate proven to block false 'verified'/'trusted' claims)")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
