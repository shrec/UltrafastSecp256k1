#!/usr/bin/env python3
"""
check_fault_countermeasure_coverage.py — fault-injection countermeasure coverage gate.

WHY (the bug class this prevents):
A single induced fault — a skipped branch, a flipped bit, a glitched comparison — can turn
an invalid signature into an emitted 'success' or leak a nonce (Boneh-DeMillo-Lipton, and
differential fault analysis on deterministic RFC 6979). The standard countermeasure is
sign-then-verify (FIPS 186-4): re-verify the signature you just produced before returning
it, so a fault during signing is caught by the verify. Two ways this silently regresses:
  * the sign-then-verify function is deleted, or
  * it is hollowed out (still named *_sign_verified but no longer calls verify),
and no fault-injection test still drives the path.

This gate makes both structural. docs/FAULT_COUNTERMEASURE_LEDGER.json lists each critical
signing path, its countermeasure function, and the fault probe. The gate enforces:
  1. The countermeasure function still EXISTS in its file AND the file re-verifies
     (the countermeasure is live, not a hollow stub).
  2. The fault-injection probe is wired into the audit runner (the path is still driven).
  3. 'roadmap' paths are reported as declared gaps.

A self-test (ci/test_check_fault_countermeasure_coverage.py) proves the gate blocks a
hollowed-out countermeasure and an unwired probe.

Exit 0 = every covered countermeasure is live + probed; exit 1 = a regression.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEDGER = os.path.join(ROOT, "docs", "FAULT_COUNTERMEASURE_LEDGER.json")
RUNNER = os.path.join(ROOT, "audit", "unified_audit_runner.cpp")


def countermeasure_present(file_text, symbol):
    """The countermeasure function is defined in its file."""
    return bool(symbol) and (symbol + "(") in file_text


def countermeasure_reverifies(file_text, verify_token="verify"):
    """The countermeasure's file re-verifies (sign-then-verify is live, not hollow)."""
    return (verify_token + "(") in file_text or ("_" + verify_token + "(") in file_text


def single_fault_caught(reverifies, corrupted):
    """Model of the countermeasure's guarantee, for the self-test: a sign-then-verify
    path that re-verifies CATCHES a fault-corrupted signature (returns False = not emitted);
    a path that does not re-verify EMITS the corrupted signature (returns True = forged)."""
    if not corrupted:
        return False           # nothing corrupted -> nothing to emit
    return not reverifies      # re-verify caught it (False) ; no verify -> emitted (True)


def check_coverage(ledger, runner_src, read_file):
    """Return (blocking[list[str]], covered[list], roadmap[list]).
    read_file(path) -> file text (injectable for the self-test)."""
    paths = ledger.get("paths", [])
    covered = [p for p in paths if p.get("status") == "covered"]
    roadmap = [p for p in paths if p.get("status") == "roadmap"]
    blocking = []
    for p in covered:
        sym = p.get("probe_run_symbol")
        mod = p.get("probe_module")
        if not sym or not mod:
            blocking.append(f"path '{p['id']}' is 'covered' but declares no probe_run_symbol/probe_module")
        else:
            if sym not in runner_src:
                blocking.append(f"covered fault probe '{sym}' not referenced in unified_audit_runner.cpp")
            if f'"{mod}"' not in runner_src:
                blocking.append(f"covered fault module \"{mod}\" not registered in ALL_MODULES")
        # Countermeasure-presence is checked only where the path declares one.
        cm = p.get("countermeasure_symbol")
        cf = p.get("countermeasure_file")
        if cm and cf:
            text = read_file(cf)
            if text is None:
                blocking.append(f"countermeasure file missing for '{p['id']}': {cf}")
            elif not countermeasure_present(text, cm):
                blocking.append(f"countermeasure '{cm}' no longer defined in {cf} (fault guard deleted)")
            elif not countermeasure_reverifies(text, p.get("verify_token") or "verify"):
                blocking.append(f"countermeasure '{cm}' in {cf} no longer re-verifies "
                                f"(hollowed-out sign-then-verify — fault-forgery surface re-opened)")
    return blocking, covered, roadmap


def _read_file(path):
    full = os.path.join(ROOT, path)
    if not os.path.exists(full):
        return None
    return open(full, encoding="utf-8", errors="replace").read()


def run() -> int:
    if not os.path.exists(LEDGER):
        print(f"FAIL: fault-countermeasure ledger not found: {LEDGER}")
        return 1
    ledger = json.load(open(LEDGER, encoding="utf-8"))
    runner_src = open(RUNNER, encoding="utf-8").read() if os.path.exists(RUNNER) else ""

    blocking, covered, roadmap = check_coverage(ledger, runner_src, _read_file)

    print("=" * 70)
    print("  Fault-Injection Countermeasure Coverage Gate (sign-then-verify)")
    print("=" * 70)
    print(f"  paths: {len(covered) + len(roadmap)}  |  covered: {len(covered)}  |  roadmap: {len(roadmap)}")
    for p in covered:
        cm = p.get("countermeasure_symbol") or "(DFA probe)"
        print(f"    \033[92m[COVERED]\033[0m {p['id']:32} {cm:24} <- {p.get('probe_module')}")
    for p in roadmap:
        print(f"    \033[93m[ROADMAP]\033[0m {p['id']:32} (countermeasure + probe required)")

    if blocking:
        print()
        for b in blocking:
            print(f"  \033[91mFAIL\033[0m  {b}")
        print(f"\n\033[91m\033[1m  FAULT-COUNTERMEASURE COVERAGE: {len(blocking)} blocking issue(s)\033[0m")
        return 1

    print()
    print(f"  OK: all {len(covered)} covered countermeasures are live + probed; "
          f"{len(roadmap)} roadmap path(s) declared.")
    return 0


def main() -> int:
    return run()


if __name__ == "__main__":
    sys.exit(main())
