#!/usr/bin/env python3
"""Security Autonomy Check — master orchestrator for all security gates.

Runs ALL security gates and produces a unified autonomy verdict:
  1. Formal invariant spec completeness
  2. Risk-surface coverage matrix
  3. Audit SLA/SLO compliance
  4. Supply-chain gate
  5. Performance-security co-gate
  6. Misuse-resistance gate
  7. Evidence governance chain validation
  8. Incident drills

Produces an autonomy_score (0-100) and autonomy_ready boolean.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
LIB_ROOT = SCRIPT_DIR.parent

# Gates with their weight in the autonomy score
GATES: list[dict] = [
    {"name": "formal_invariants", "script": "check_formal_invariants.py", "weight": 15},
    {"name": "risk_surface_coverage", "script": "risk_surface_coverage.py", "weight": 15},
    {"name": "audit_sla", "script": "audit_sla_check.py", "weight": 10},
    {"name": "supply_chain", "script": "supply_chain_gate.py", "weight": 15},
    {"name": "perf_security_cogate", "script": "perf_security_cogate.py", "weight": 10},
    {"name": "misuse_resistance", "script": "check_misuse_resistance.py", "weight": 10},
    {"name": "evidence_governance", "script": "evidence_governance.py", "weight": 15, "args": ["validate"]},
    {"name": "incident_drills", "script": "incident_drills.py", "weight": 10},
]


def _run_gate(gate: dict, timeout: int = 300) -> dict:
    """Run a single gate and return result."""
    script = SCRIPT_DIR / gate["script"]
    if not script.exists():
        return {
            "gate": gate["name"],
            "weight": gate["weight"],
            "status": "missing",
            "passing": False,
            "score": 0,
            "detail": f"Script not found: {script}",
        }

    extra_args = gate.get("args", [])
    cmd = [sys.executable, str(script)] + extra_args + ["--json"]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                                timeout=timeout, cwd=str(LIB_ROOT))
        try:
            gate_report = json.loads(result.stdout)
            passing = gate_report.get("overall_pass", False)
        except (json.JSONDecodeError, ValueError):
            passing = result.returncode == 0
            gate_report = {"raw_output": result.stdout[:500]}

        score = gate["weight"] if passing else 0

        return {
            "gate": gate["name"],
            "weight": gate["weight"],
            "status": "ran",
            "passing": passing,
            "score": score,
            "returncode": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {
            "gate": gate["name"],
            "weight": gate["weight"],
            "status": "timeout",
            "passing": False,
            "score": 0,
            "detail": f"Timed out after {timeout}s",
        }
    except Exception as exc:
        return {
            "gate": gate["name"],
            "weight": gate["weight"],
            "status": "error",
            "passing": False,
            "score": 0,
            "detail": str(exc),
        }


def run(json_mode: bool, out_file: str | None, timeout: int = 300) -> int:
    results = []
    for gate in GATES:
        results.append(_run_gate(gate, timeout))

    total_weight = sum(g["weight"] for g in GATES)
    earned_score = sum(r["score"] for r in results)
    autonomy_score = round(earned_score / total_weight * 100) if total_weight > 0 else 0

    gates_total = len(results)
    gates_passing = sum(1 for r in results if r["passing"])
    autonomy_ready = autonomy_score >= 90

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "autonomy_score": autonomy_score,
        "autonomy_ready": autonomy_ready,
        "gates_total": gates_total,
        "gates_passing": gates_passing,
        "overall_pass": autonomy_ready,
        "gates": results,
    }

    rendered = json.dumps(report, indent=2)
    if out_file:
        Path(out_file).write_text(rendered, encoding="utf-8")

    # Also save to KPI file
    kpi_file = LIB_ROOT / "docs" / "SECURITY_AUTONOMY_KPI.json"
    try:
        kpi_file.write_text(rendered, encoding="utf-8")
    except Exception:
        pass

    if json_mode:
        print(rendered)
    else:
        print("  Security Autonomy Check")
        print("  " + "=" * 50)
        for r in results:
            status = "PASS" if r["passing"] else "FAIL"
            print(f"  {status} {r['gate']} (weight={r['weight']}, score={r['score']})")
        print()
        print(f"  Autonomy Score: {autonomy_score}/100")
        print(f"  Gates Passing:  {gates_passing}/{gates_total}")
        print(f"  Autonomy Ready: {'YES' if autonomy_ready else 'NO'}")
        print()
        if autonomy_ready:
            print("PASS security autonomy check")
        else:
            failing = [r["gate"] for r in results if not r["passing"]]
            print(f"FAIL autonomy score {autonomy_score}/100 (need ≥90)")
            print(f"  Failing gates: {', '.join(failing)}")

    return 0 if autonomy_ready else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("-o", dest="out_file", help="Write report to file")
    parser.add_argument("--timeout", type=int, default=300, help="Per-gate timeout (seconds)")
    args = parser.parse_args()
    return run(args.json, args.out_file, args.timeout)


if __name__ == "__main__":
    raise SystemExit(main())
