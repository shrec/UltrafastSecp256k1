#!/usr/bin/env python3
"""
check_metamorphic_coverage.py — the metamorphic-relation coverage gate.

WHY (the bug class this prevents):
A roundtrip test (honest sign -> verify==OK) proves a single happy path. It does
NOT pin the STRUCTURE of a transform across the input family. A protocol bug that
breaks the adapt/extract correspondence (the atomicity guarantee behind DLCs and
atomic swaps), or the MuSig2 aggregate==single-key equivalence, or Pedersen
homomorphism, can leave every honest roundtrip passing while the algebraic relation
that makes the protocol *safe* is silently broken. This is the POSITIVE complement
to check_soundness_coverage.py: soundness forges a violating input and asserts
rejection; metamorphic asserts an identity that must hold for ALL valid inputs.

docs/METAMORPHIC_RELATIONS.json is the ledger. Every custom-protocol transform
declares its metamorphic relation and (when covered) a probe wired into the audit
runner. The gate:

  1. BLOCKS if a 'covered' relation's probe is not actually wired into the runner
     (a regression of existing metamorphic coverage).
  2. REPORTS the 'roadmap' relations loudly (declared-but-not-yet-probed) so the
     gap is visible and closed incrementally.

Exit 0 = clean, exit 1 = a covered probe is unwired/undeclared.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEDGER = os.path.join(ROOT, "docs", "METAMORPHIC_RELATIONS.json")
RUNNER = os.path.join(ROOT, "audit", "unified_audit_runner.cpp")


def fail(msg, out):
    out.append("  \033[91mFAIL\033[0m  " + msg)


def main() -> int:
    out = []
    blocking = 0

    if not os.path.exists(LEDGER):
        print(f"FAIL: metamorphic ledger not found: {LEDGER}")
        return 1
    ledger = json.load(open(LEDGER, encoding="utf-8"))
    rels = ledger.get("relations", [])

    runner_src = open(RUNNER, encoding="utf-8").read() if os.path.exists(RUNNER) else ""

    covered = [r for r in rels if r.get("status") == "covered"]
    roadmap = [r for r in rels if r.get("status") == "roadmap"]

    # (1) Every covered relation must have its probe wired into the runner.
    for r in covered:
        sym = r.get("probe_run_symbol")
        mod = r.get("probe_module")
        if not sym or not mod:
            fail(f"relation '{r['id']}' is 'covered' but declares no probe_run_symbol/probe_module", out)
            blocking += 1
            continue
        if sym not in runner_src:
            fail(f"covered probe '{sym}' not forward-declared/referenced in unified_audit_runner.cpp", out)
            blocking += 1
        if f'"{mod}"' not in runner_src:
            fail(f"covered probe module \"{mod}\" not registered in ALL_MODULES (unified_audit_runner.cpp)", out)
            blocking += 1

    # Report
    print("=" * 70)
    print("  Metamorphic-Relation Coverage Gate (positive-invariant ledger)")
    print("=" * 70)
    print(f"  relations: {len(rels)}  |  covered (probed): {len(covered)}  |  roadmap: {len(roadmap)}")
    for r in covered:
        print(f"    \033[92m[COVERED]\033[0m {r['id']:42} <- {r.get('probe_module')}")
    for r in roadmap:
        print(f"    \033[93m[ROADMAP]\033[0m {r['id']:42} (probe required)")
    if out:
        print()
        print("\n".join(out))

    if blocking:
        print()
        print(f"\033[91m\033[1m  METAMORPHIC-COVERAGE: {blocking} blocking issue(s)\033[0m")
        print("  A metamorphic probe asserts an algebraic identity preserved across a")
        print("  protocol transform. Without it, a structural break passes every roundtrip.")
        return 1

    print()
    if roadmap:
        print(f"  OK: all {len(covered)} covered probe(s) wired; {len(roadmap)} roadmap relation(s) "
              f"declared (probes pending — close incrementally).")
    else:
        print(f"  OK: all {len(rels)} metamorphic relations covered by wired probes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
