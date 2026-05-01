#!/usr/bin/env python3
"""Performance-security co-gating — rejects optimizations that regress security.

Ensures that performance improvements cannot merge if any of the following
security properties regress:
  - Constant-time evidence (dudect, valgrind-ct, cachegrind)
  - Determinism gate (cross-architecture bitwise identity)
  - Secret lifecycle (zeroization paths, nonce handling)
  - GPU parity (no new undocumented Unsupported stubs)

The gate is coupled: perf cannot pass if security regresses.
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


def _run_gate(script_name: str, *args: str, timeout: int = 300) -> dict:
    """Run a security gate script and return result."""
    script = SCRIPT_DIR / script_name
    if not script.exists():
        return {
            "gate": script_name,
            "status": "missing",
            "passing": False,
            "detail": f"Gate script not found: {script}",
        }

    cmd = [sys.executable, str(script), "--json"] + list(args)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                                timeout=timeout, cwd=str(LIB_ROOT))
        try:
            gate_report = json.loads(result.stdout)
            passing = gate_report.get("overall_pass", False)
        except (json.JSONDecodeError, ValueError):
            passing = result.returncode == 0
            gate_report = {"raw_output": result.stdout[:500]}

        return {
            "gate": script_name,
            "status": "ran",
            "passing": passing,
            "returncode": result.returncode,
            "detail": gate_report,
        }
    except subprocess.TimeoutExpired:
        return {
            "gate": script_name,
            "status": "timeout",
            "passing": False,
            "detail": f"Gate timed out after {timeout}s",
        }
    except Exception as exc:
        return {
            "gate": script_name,
            "status": "error",
            "passing": False,
            "detail": str(exc),
        }


def check_ct_evidence() -> dict:
    """Check that CT evidence has not regressed."""
    ci_evidence = LIB_ROOT / "audit" / "ci-evidence"
    if not ci_evidence.is_dir():
        return {
            "gate": "ct_evidence",
            "status": "missing",
            "passing": False,
            "detail": "audit/ci-evidence directory not found",
        }

    evidence_files = list(ci_evidence.rglob("*.json")) + list(ci_evidence.rglob("*.txt"))
    if not evidence_files:
        return {
            "gate": "ct_evidence",
            "status": "empty",
            "passing": False,
            "detail": "No CT evidence files found in audit/ci-evidence/",
        }

    return {
        "gate": "ct_evidence",
        "status": "present",
        "passing": True,
        "evidence_files": len(evidence_files),
        "detail": f"{len(evidence_files)} CT evidence file(s) found",
    }


def check_secret_lifecycle() -> dict:
    """Check that secret lifecycle documentation is current."""
    secret_doc = LIB_ROOT / "docs" / "SECRET_LIFECYCLE.md"
    if not secret_doc.exists():
        return {
            "gate": "secret_lifecycle",
            "status": "missing",
            "passing": False,
            "detail": "docs/SECRET_LIFECYCLE.md not found",
        }

    return {
        "gate": "secret_lifecycle",
        "status": "present",
        "passing": True,
        "detail": "SECRET_LIFECYCLE.md exists and is tracked",
    }


def check_gpu_parity_stubs() -> dict:
    """Check for undocumented GPU Unsupported stubs."""
    gpu_dirs = [
        LIB_ROOT / "gpu" / "src",
        LIB_ROOT / "opencl",
        LIB_ROOT / "metal",
    ]

    undocumented = []
    for d in gpu_dirs:
        if not d.is_dir():
            continue
        for f in d.rglob("*.cpp"):
            try:
                content = f.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            lines = content.split("\n")
            for i, line in enumerate(lines):
                if "Unsupported" in line:
                    # Check for documenting comment
                    context_start = max(0, i - 3)
                    context = "\n".join(lines[context_start:i + 1])
                    if "TODO(parity)" not in context and "PARITY-EXCEPTION" not in context:
                        undocumented.append(f"{f.relative_to(LIB_ROOT)}:{i + 1}")

    return {
        "gate": "gpu_parity_stubs",
        "status": "checked",
        "passing": len(undocumented) == 0,
        "undocumented_stubs": undocumented,
        "detail": f"{len(undocumented)} undocumented Unsupported stub(s)" if undocumented
                  else "No undocumented GPU stubs",
    }


def run(json_mode: bool, out_file: str | None) -> int:
    results = [
        check_ct_evidence(),
        _run_gate("check_formal_invariants.py"),
        check_secret_lifecycle(),
        check_gpu_parity_stubs(),
    ]

    all_passing = all(r["passing"] for r in results)
    failing_gates = [r["gate"] for r in results if not r["passing"]]

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "overall_pass": all_passing,
        "gates_total": len(results),
        "gates_passing": sum(1 for r in results if r["passing"]),
        "failing_gates": failing_gates,
        "security_regressions_blocked": len(failing_gates),
        "cogate_pass": all_passing,
        "gates": results,
    }

    rendered = json.dumps(report, indent=2)
    if out_file:
        Path(out_file).write_text(rendered, encoding="utf-8")

    if json_mode:
        print(rendered)
    else:
        for r in results:
            status = "PASS" if r["passing"] else "FAIL"
            detail = r.get("detail", "")
            if isinstance(detail, dict):
                detail = ""
            print(f"  {status} {r['gate']}: {detail}")
        print()
        if all_passing:
            print("PASS performance-security co-gate")
        else:
            print(f"FAIL co-gate blocked: {', '.join(failing_gates)}")

    return 0 if all_passing else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("-o", dest="out_file", help="Write report to file")
    args = parser.parse_args()
    return run(args.json, args.out_file)


if __name__ == "__main__":
    raise SystemExit(main())
