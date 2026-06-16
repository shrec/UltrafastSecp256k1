#!/usr/bin/env python3
"""
check_threat_gate_coverage.py — the META gate: "what gate are we missing?"

DON'T TRUST, VERIFY. This is the apex of the CAAS bastion: it enforces that every
threat class has a gate, and that every gate is itself VERIFIED by a self-test that
proves it blocks (inject a violation -> the gate goes red). A gate we merely TRUST
(no proof-it-blocks self-test) is not the same as a gate we have VERIFIED.

Backed by docs/THREAT_GATE_MATRIX.json. Status per threat:
  verified -> gate file exists AND a self-test file exists      (proven to block)
  trusted  -> gate file exists, no self-test                    (believed, not proven)
  gap      -> no gate exists for this class                     (a hole)

Enforcement:
  * BLOCK if a 'verified' threat's gate OR self-test file is missing (a false claim).
  * BLOCK if a 'trusted' threat's gate file is missing (a claimed gate that is absent).
  * REPORT the 'trusted' set loudly — these are the don't-trust-verify backlog
    (gates we have never confirmed actually block).
  * REPORT the 'gap' set loudly — threat classes with no gate at all.

A clean self-test path (ci/test_check_threat_gate_coverage.py) proves THIS gate also
blocks — we don't trust the meta-gate either.

Exit 0 = clean (no false claims), exit 1 = a verified/trusted gate file is missing.
"""
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MATRIX = os.path.join(ROOT, "docs", "THREAT_GATE_MATRIX.json")

# A gate field may be a file path, a `file::function` ref, or a prose description
# of a workflow set. Only the first form is checkable as a file.
_PATHISH = re.compile(r"^(ci/|fuzz/|tools/|scripts/)[\w./-]+\.(py|sh|cc|cpp)\b")


def _gate_file(gate: str):
    if not gate:
        return None
    head = gate.split("::", 1)[0].split()[0]
    if _PATHISH.match(head):
        return head
    return None  # prose / workflow-glob description — not a single checkable file


def run(matrix_path: str) -> int:
    if not os.path.exists(matrix_path):
        print(f"FAIL: threat-gate matrix not found: {matrix_path}")
        return 1
    matrix = json.load(open(matrix_path, encoding="utf-8"))
    threats = matrix.get("threats", [])

    blocking = []
    verified, trusted, gaps = [], [], []

    for t in threats:
        st = t.get("status")
        gate = t.get("gate")
        stest = t.get("self_test")
        tid = t.get("id", "?")
        if st == "verified":
            gf = _gate_file(gate)
            if gf and not os.path.exists(os.path.join(ROOT, gf)):
                blocking.append(f"'{tid}' is 'verified' but its gate file is missing: {gf}")
            if not stest or not os.path.exists(os.path.join(ROOT, stest)):
                blocking.append(f"'{tid}' is 'verified' but its self-test is missing: {stest} "
                                f"(a 'verified' claim REQUIRES a proof-it-blocks self-test)")
            verified.append(t)
        elif st == "trusted":
            gf = _gate_file(gate)
            if gf and not os.path.exists(os.path.join(ROOT, gf)):
                blocking.append(f"'{tid}' claims gate '{gf}' but the file is missing")
            trusted.append(t)
        elif st == "gap":
            gaps.append(t)
        else:
            blocking.append(f"'{tid}' has unknown status '{st}'")

    print("=" * 70)
    print("  Threat-Gate Coverage Matrix  (DON'T TRUST, VERIFY)")
    print("=" * 70)
    print(f"  threats: {len(threats)}  |  \033[92mverified: {len(verified)}\033[0m  "
          f"|  \033[93mtrusted (unverified): {len(trusted)}\033[0m  "
          f"|  \033[91mgap (no gate): {len(gaps)}\033[0m")
    for t in verified:
        print(f"    \033[92m[VERIFIED]\033[0m {t['id']:34} gate+self-test present")
    if trusted:
        print("  --- TRUSTED-NOT-VERIFIED (gate exists, no proof-it-blocks self-test) ---")
        for t in trusted:
            print(f"    \033[93m[TRUSTED ]\033[0m {t['id']:34} {t.get('gate','')[:46]}")
    if gaps:
        print("  --- GAPS (no gate for this threat class) ---")
        for t in gaps:
            print(f"    \033[91m[GAP]\033[0m {t['id']:34} {t.get('note','')[:60]}")

    if blocking:
        print()
        for b in blocking:
            print(f"  \033[91mFAIL\033[0m  {b}")
        print(f"\n\033[91m\033[1m  THREAT-GATE COVERAGE: {len(blocking)} blocking issue(s)\033[0m")
        return 1

    print()
    print(f"  OK: no false claims. {len(verified)} verified, {len(trusted)} trusted "
          f"(self-tests pending), {len(gaps)} declared gaps (being built).")
    return 0


def main() -> int:
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--matrix", default=DEFAULT_MATRIX)
    args = p.parse_args()
    return run(args.matrix)


if __name__ == "__main__":
    sys.exit(main())
