#!/usr/bin/env python3
"""
Self-test for ci/check_fuzz_harness_wiring.py — DON'T TRUST, VERIFY.

We prove the fuzz-liveness gate blocks a dead harness (one wired into no build) and a
dangling build reference, and that a fully-wired set passes. The gate's core is the pure
analyze() function, so we feed it crafted ref/disk sets — no disk scaffolding required.
"""
import importlib.util
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load():
    spec = importlib.util.spec_from_file_location(
        "cfw", os.path.join(ROOT, "ci", "check_fuzz_harness_wiring.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main() -> int:
    m = _load()
    fails = []

    # 0. The REAL repo state must pass (the gate is not spuriously red).
    if m.run() != 0:
        fails.append("real fuzz-harness wiring should pass (exit 0)")

    # 1. A discovered CFL harness wired into NO build MUST block.
    blk, _ = m.analyze(
        {"cfl": ["fuzz_field.cpp", "fuzz_DEAD.cpp"], "audit": []},
        cfl_refs={"fuzz_field"},                 # fuzz_DEAD not built
        audit_refs=set(),
        disk_cfl={"fuzz_field", "fuzz_DEAD"}, disk_audit=set())
    if not blk:
        fails.append("a CFL libFuzzer harness wired into no build MUST block")

    # 2. An audit harness not in _FUZZ_HARNESSES MUST block.
    blk2, _ = m.analyze(
        {"cfl": [], "audit": ["fuzz_der_parse.cpp", "fuzz_ORPHAN.cpp"]},
        cfl_refs=set(),
        audit_refs={"fuzz_der_parse"},           # fuzz_ORPHAN not listed
        disk_cfl=set(), disk_audit={"fuzz_der_parse", "fuzz_ORPHAN"})
    if not blk2:
        fails.append("an audit harness not in _FUZZ_HARNESSES MUST block")

    # 3. A dangling reference (built/listed stem absent on disk) MUST block.
    blk3, _ = m.analyze(
        {"cfl": ["fuzz_field.cpp"], "audit": []},
        cfl_refs={"fuzz_field", "fuzz_GHOST"},   # fuzz_GHOST not on disk
        audit_refs=set(),
        disk_cfl={"fuzz_field"}, disk_audit=set())
    if not blk3:
        fails.append("a dangling build reference to a missing harness MUST block")

    # 4. A fully-wired set must NOT block.
    ok, _ = m.analyze(
        {"cfl": ["fuzz_field.cpp"], "audit": ["fuzz_der_parse.cpp"]},
        cfl_refs={"fuzz_field"},
        audit_refs={"fuzz_der_parse"},
        disk_cfl={"fuzz_field"}, disk_audit={"fuzz_der_parse"})
    if ok:
        fails.append("a fully-wired harness set must NOT block")

    print("=" * 60)
    if fails:
        print("  check_fuzz_harness_wiring SELF-TEST: FAILED")
        for f in fails:
            print("   - " + f)
        print("=" * 60)
        return 1
    print("  check_fuzz_harness_wiring SELF-TEST PASSED")
    print("  (gate proven to block dead harnesses + dangling references)")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
