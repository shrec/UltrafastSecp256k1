#!/usr/bin/env python3
"""Research signal-matrix attack-class taxonomy + evidence-routing gate (Bastion B18, closes RR-BAS-04).

Validates docs/RESEARCH_SIGNAL_MATRIX.json so research intake is routed to audit
work, not just labelled covered/candidate. Every in-scope signal class must carry
a stable `attack_class` (from `attack_class_enum`) and route to the expected
evidence surface + gate:

  - attack_class present and in the enum (every class);
  - covered / original_analysis classes: every `expected_evidence` path exists and
    `expected_gate` resolves (an audit_gate CHECK_MAP flag or a ci/*.py script);
  - candidate classes: a `missing_evidence_action` is declared;
  - out_of_scope classes: a `rationale` (or `reason`) is present;
  - `expected_evidence` paths must exist unless the class is out_of_scope.

Exit code:
  0  the matrix is fully routed
  1  a missing/invalid attack_class, missing evidence, unresolved gate, or missing action
  2  the matrix is missing or malformed

Usage:
  python3 ci/check_research_signal_matrix.py [--json] [-o report.json]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
LIB_ROOT = SCRIPT_DIR.parent
MATRIX_PATH = LIB_ROOT / "docs" / "RESEARCH_SIGNAL_MATRIX.json"
AUDIT_GATE = SCRIPT_DIR / "audit_gate.py"

COVERED_STATUSES = {"covered", "original_analysis"}


def _audit_gate_flags() -> set:
    """Set of --flags registered in audit_gate.py CHECK_MAP."""
    if not AUDIT_GATE.exists():
        return set()
    src = AUDIT_GATE.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"CHECK_MAP\s*=\s*\{(.*?)\n\}", src, re.S)
    if not m:
        return set()
    return set(re.findall(r"'(--[a-z0-9-]+)'", m.group(1)))


def _gate_resolves(gate: str, flags: set) -> bool:
    if not gate:
        return False
    gate = gate.strip()
    if gate.startswith("audit_gate.py"):
        flag = next((tok for tok in gate.split() if tok.startswith("--")), None)
        return flag in flags
    # otherwise treat as a ci/*.py (or repo-relative) script path
    return (LIB_ROOT / gate).exists()


def evaluate(classes: list, enum, flags) -> dict:
    """Evaluate signal classes against the taxonomy (pure / testable)."""
    enum_set = set(enum or [])
    rows = []
    for c in classes:
        cid = c.get("id", "<no-id>")
        status = c.get("status", "")
        ac = c.get("attack_class")
        problems = []

        if not ac:
            problems.append("missing_attack_class")
        elif ac not in enum_set:
            problems.append(f"invalid_attack_class:{ac}")

        # expected_evidence paths must exist unless out_of_scope.
        if status != "out_of_scope":
            for ev in (c.get("expected_evidence") or []):
                if not (LIB_ROOT / ev).exists():
                    problems.append(f"missing_evidence:{ev}")

        if status in COVERED_STATUSES:
            if not _gate_resolves(c.get("expected_gate", ""), flags):
                problems.append(f"unresolved_gate:{c.get('expected_gate')!r}")
        elif status == "candidate":
            if not (c.get("missing_evidence_action") or "").strip():
                problems.append("missing_action:candidate without missing_evidence_action")
        elif status == "out_of_scope":
            if not (c.get("rationale") or c.get("reason") or "").strip():
                problems.append("missing_action:out_of_scope without rationale/residual text")

        rows.append({"id": cid, "status": status, "attack_class": ac,
                     "affected_surface": c.get("affected_surface"),
                     "expected_gate": c.get("expected_gate"), "problems": problems})

    def collect(prefix):
        return [r["id"] for r in rows if any(p.split(":")[0] == prefix for p in r["problems"])]

    missing_ac = collect("missing_attack_class")
    invalid_ac = collect("invalid_attack_class")
    missing_ev = collect("missing_evidence")
    unresolved = collect("unresolved_gate")
    missing_action = collect("missing_action")
    overall = not (missing_ac or invalid_ac or missing_ev or unresolved or missing_action)

    return {
        "overall_pass": overall,
        "classes_total": len(rows),
        "missing_attack_class": missing_ac,
        "invalid_attack_class": invalid_ac,
        "missing_evidence": missing_ev,
        "unresolved_gate": unresolved,
        "missing_action": missing_action,
        "rows": rows,
    }


def load_and_evaluate() -> tuple[dict, int]:
    if not MATRIX_PATH.exists():
        return ({"overall_pass": False, "error": f"matrix missing: {MATRIX_PATH}"}, 2)
    try:
        matrix = json.loads(MATRIX_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        return ({"overall_pass": False, "error": f"matrix malformed JSON: {exc}"}, 2)
    classes = matrix.get("classes")
    enum = matrix.get("attack_class_enum")
    if not isinstance(classes, list) or not classes:
        return ({"overall_pass": False, "error": "matrix has no classes[]"}, 2)
    if not enum:
        return ({"overall_pass": False, "error": "matrix has no attack_class_enum"}, 2)
    report = evaluate(classes, enum, _audit_gate_flags())
    return (report, 0 if report["overall_pass"] else 1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("-o", dest="out_file", help="Write report to file")
    args = parser.parse_args()

    report, code = load_and_evaluate()
    rendered = json.dumps(report, indent=2)
    if args.out_file:
        Path(args.out_file).write_text(rendered, encoding="utf-8")

    if args.json:
        print(rendered)
    else:
        if "error" in report:
            print(f"FAIL research signal matrix: {report['error']}")
            return code
        for r in report["rows"]:
            if r["problems"]:
                print(f"  [FAIL] {r['id']}: {r['problems']}")
        print()
        if report["overall_pass"]:
            print(f"PASS research signal matrix ({report['classes_total']} classes fully routed)")
        else:
            print(f"FAIL research signal matrix: missing_attack_class={report['missing_attack_class']} "
                  f"invalid={report['invalid_attack_class']} missing_evidence={report['missing_evidence']} "
                  f"unresolved_gate={report['unresolved_gate']} missing_action={report['missing_action']}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
