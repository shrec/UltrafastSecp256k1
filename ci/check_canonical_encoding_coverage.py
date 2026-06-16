#!/usr/bin/env python3
"""
check_canonical_encoding_coverage.py — canonical-encoding / malleability coverage gate.

WHY (the bug class this prevents):
A consensus-relevant decoder (signature, pubkey, point, field, key) must REJECT non-canonical
encodings. A second valid encoding of the same value is transaction/txid malleability, and for
signatures a consensus split (BIP-66 strict DER, BIP-146 low-S, BIP-340 r<p/s<n, libsecp PR
#1839 fe_set_b32 overflow). The threat matrix is closed-world, so an undeclared canonical-
encoding regression is invisible — it cannot even be marked a 'gap'. This ledger makes the
class explicit: every consensus-relevant encoding declares its non-canonical-rejection probe.

docs/CANONICAL_ENCODING_LEDGER.json is the ledger. The gate BLOCKS if a 'covered' encoding's
probe is not wired into the audit runner (a regression of existing malleability coverage), and
REPORTS the 'roadmap' encodings (declared-but-not-yet-probed) so the gap is visible.

Exit 0 = clean, exit 1 = a covered probe is unwired/undeclared.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEDGER = os.path.join(ROOT, "docs", "CANONICAL_ENCODING_LEDGER.json")
RUNNER = os.path.join(ROOT, "audit", "unified_audit_runner.cpp")


def check_coverage(ledger, runner_src):
    """Return (blocking[list[str]], covered[list], roadmap[list])."""
    encs = ledger.get("encodings", [])
    covered = [e for e in encs if e.get("status") == "covered"]
    roadmap = [e for e in encs if e.get("status") == "roadmap"]
    blocking = []
    for e in covered:
        sym = e.get("probe_run_symbol")
        mod = e.get("probe_module")
        if not sym or not mod:
            blocking.append(f"encoding '{e['id']}' is 'covered' but declares no probe_run_symbol/probe_module")
            continue
        if sym not in runner_src:
            blocking.append(f"covered probe '{sym}' not referenced in unified_audit_runner.cpp")
        if f'"{mod}"' not in runner_src:
            blocking.append(f"covered probe module \"{mod}\" not registered in ALL_MODULES")
    return blocking, covered, roadmap


def run() -> int:
    if not os.path.exists(LEDGER):
        print(f"FAIL: canonical-encoding ledger not found: {LEDGER}")
        return 1
    ledger = json.load(open(LEDGER, encoding="utf-8"))
    runner_src = open(RUNNER, encoding="utf-8").read() if os.path.exists(RUNNER) else ""
    blocking, covered, roadmap = check_coverage(ledger, runner_src)

    print("=" * 70)
    print("  Canonical-Encoding / Malleability Coverage Gate")
    print("=" * 70)
    print(f"  encodings: {len(covered) + len(roadmap)}  |  covered (probed): {len(covered)}  |  roadmap: {len(roadmap)}")
    for e in covered:
        print(f"    \033[92m[COVERED]\033[0m {e['id']:28} <- {e.get('probe_module')}")
    for e in roadmap:
        print(f"    \033[93m[ROADMAP]\033[0m {e['id']:28} (non-canonical-rejection probe required)")
    if blocking:
        print()
        for b in blocking:
            print(f"  \033[91mFAIL\033[0m  {b}")
        print(f"\n\033[91m\033[1m  CANONICAL-ENCODING COVERAGE: {len(blocking)} blocking issue(s)\033[0m")
        return 1
    print()
    if roadmap:
        print(f"  OK: all {len(covered)} covered probe(s) wired; {len(roadmap)} roadmap encoding(s) declared.")
    else:
        print(f"  OK: all {len(covered)} encodings covered by wired non-canonical-rejection probes.")
    return 0


def main() -> int:
    return run()


if __name__ == "__main__":
    sys.exit(main())
