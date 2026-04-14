#!/usr/bin/env python3
"""
caas_runner.py  --  Continuous Audit as a Service (CAAS) local runner

Runs the full audit pipeline in a single fail-fast pass:

  Stage 1  —  Static analysis (audit_test_quality_scanner)
  Stage 2  —  Audit gate        (audit_gate.py)
  Stage 3  —  Security autonomy (security_autonomy_check.py)
  Stage 4  —  Bundle produce    (external_audit_bundle.py)
  Stage 5  —  Bundle verify     (verify_external_audit_bundle.py)

Exit codes:
    0   all stages PASS
    1   one or more stages FAIL (fail-fast: stops at first failure)
    2   invocation/setup error

Usage:
    python3 scripts/caas_runner.py                   # fail-fast by default
    python3 scripts/caas_runner.py --no-fail-fast    # run all stages, report all
    python3 scripts/caas_runner.py --json            # JSON summary to stdout
    python3 scripts/caas_runner.py --json -o caas_report.json
    python3 scripts/caas_runner.py --skip-bundle     # skip bundle gen/verify (faster)
    python3 scripts/caas_runner.py --stage scanner   # run only one stage

Environment overrides:
    CAAS_SCANNER_DIR   — override audit dir for scanner (default: audit/)
    CAAS_TIMEOUT       — per-stage timeout in seconds (default: 300)
    CAAS_FAIL_FAST     — set to "0" to disable fail-fast
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
LIB_ROOT = SCRIPT_DIR.parent

RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"

CAAS_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Stage definitions
# ---------------------------------------------------------------------------

STAGES = [
    {
        "id": "scanner",
        "name": "Static Analysis (audit_test_quality_scanner)",
        "script": "audit_test_quality_scanner.py",
        "args": ["--json"],
        "pass_fn": "_scanner_pass",
        "description": "Zero-findings requirement on exploit PoC suite",
        "blocking": True,
    },
    {
        "id": "audit_gate",
        "name": "Audit Gate (audit_gate.py)",
        "script": "audit_gate.py",
        "args": ["--json"],
        "pass_fn": "_generic_pass",
        "description": "P0–P18 audit principles gate",
        "blocking": True,
    },
    {
        "id": "security_autonomy",
        "name": "Security Autonomy (security_autonomy_check.py)",
        "script": "security_autonomy_check.py",
        "args": ["--json"],
        "pass_fn": "_autonomy_pass",
        "description": "8-gate autonomy check — must score 100/100",
        "blocking": True,
    },
    {
        "id": "bundle_produce",
        "name": "Bundle Produce (external_audit_bundle.py)",
        "script": "external_audit_bundle.py",
        "args": [],
        "pass_fn": "_generic_pass",
        "description": "Hash-pin audit evidence bundle",
        "blocking": True,
        "skippable": True,
    },
    {
        "id": "bundle_verify",
        "name": "Bundle Verify (verify_external_audit_bundle.py)",
        "script": "verify_external_audit_bundle.py",
        "args": ["--json"],
        "pass_fn": "_generic_pass",
        "description": "Cryptographic integrity check of evidence bundle",
        "blocking": True,
        "skippable": True,
    },
]


# ---------------------------------------------------------------------------
# Pass functions (evaluated after subprocess returns)
# ---------------------------------------------------------------------------

def _generic_pass(result: subprocess.CompletedProcess, _stdout_json: dict | None) -> tuple[bool, str]:
    """Passes if exit code is 0."""
    passed = result.returncode == 0
    detail = "exit 0" if passed else f"exit {result.returncode}"
    return passed, detail


def _scanner_pass(result: subprocess.CompletedProcess, stdout_json: dict | None) -> tuple[bool, str]:
    """Passes only when total_findings == 0."""
    if result.returncode != 0:
        return False, f"scanner exited {result.returncode}"
    if stdout_json is None:
        return False, "scanner produced no JSON output"
    total = stdout_json.get("total_findings", -1)
    if total == 0:
        return True, "0 findings"
    counts = {sev: 0 for sev in ("critical", "high", "medium", "low", "info")}
    for f in stdout_json.get("findings", []):
        sev = f.get("severity", "info")
        counts[sev] = counts.get(sev, 0) + 1
    summary = ", ".join(f"{v} {k}" for k, v in counts.items() if v)
    return False, f"{total} finding(s): {summary}"


def _autonomy_pass(result: subprocess.CompletedProcess, stdout_json: dict | None) -> tuple[bool, str]:
    """Passes only when autonomy_score == 100."""
    if result.returncode != 0 and stdout_json is None:
        return False, f"autonomy check exited {result.returncode}"
    if stdout_json is not None:
        score = stdout_json.get("autonomy_score", 0)
        ready = stdout_json.get("autonomy_ready", False)
        if ready and score >= 100:
            return True, f"score={score}/100"
        return False, f"score={score}/100, ready={ready}"
    return result.returncode == 0, f"exit {result.returncode}"


# Patch forward-declared functions into STAGES table
def _resolve_stage_fns():
    fn_map = {
        "_scanner_pass": _scanner_pass,
        "_generic_pass": _generic_pass,
        "_autonomy_pass": _autonomy_pass,
    }
    for stage in STAGES:
        fn_name = stage.get("pass_fn")
        if isinstance(fn_name, str):
            stage["pass_fn"] = fn_map[fn_name]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_stage(stage: dict, timeout: int) -> dict:
    """Run a single CAAS stage and return a result dict."""
    script_path = SCRIPT_DIR / stage["script"]
    if not script_path.exists():
        return {
            "id": stage["id"],
            "name": stage["name"],
            "status": "missing",
            "passed": False,
            "detail": f"Script not found: {script_path}",
            "duration_s": 0.0,
        }

    import os
    env_overrides = {}
    if stage["id"] == "scanner":
        scanner_dir = os.environ.get("CAAS_SCANNER_DIR")
        if scanner_dir:
            env_overrides["CAAS_SCANNER_DIR"] = scanner_dir

    cmd = [sys.executable, str(script_path)] + stage["args"]
    t0 = time.monotonic()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(LIB_ROOT),
            env={**__import__("os").environ, **env_overrides},
        )
    except subprocess.TimeoutExpired:
        elapsed = time.monotonic() - t0
        return {
            "id": stage["id"],
            "name": stage["name"],
            "status": "timeout",
            "passed": False,
            "detail": f"Timed out after {timeout}s",
            "duration_s": round(elapsed, 2),
        }
    elapsed = round(time.monotonic() - t0, 2)

    stdout_json = None
    try:
        stdout_json = json.loads(result.stdout)
    except (json.JSONDecodeError, ValueError):
        pass

    passed, detail = stage["pass_fn"](result, stdout_json)

    return {
        "id": stage["id"],
        "name": stage["name"],
        "status": "passed" if passed else "failed",
        "passed": passed,
        "returncode": result.returncode,
        "detail": detail,
        "duration_s": elapsed,
        "stdout_tail": result.stdout[-2000:] if not passed else "",
        "stderr_tail": result.stderr[-500:] if not passed else "",
    }


def print_banner(text: str, color: str = CYAN) -> None:
    bar = "─" * 60
    print(f"\n{color}{BOLD}{bar}{RESET}")
    print(f"{color}{BOLD}  {text}{RESET}")
    print(f"{color}{BOLD}{bar}{RESET}")


def print_stage_header(idx: int, total: int, name: str) -> None:
    print(f"\n{CYAN}[{idx}/{total}] {name}{RESET}")


def print_result(result: dict) -> None:
    if result["passed"]:
        icon = f"{GREEN}✓{RESET}"
        label = f"{GREEN}PASS{RESET}"
    elif result["status"] == "missing":
        icon = f"{YELLOW}?{RESET}"
        label = f"{YELLOW}MISSING{RESET}"
    elif result["status"] == "timeout":
        icon = f"{YELLOW}⏱{RESET}"
        label = f"{YELLOW}TIMEOUT{RESET}"
    else:
        icon = f"{RED}✗{RESET}"
        label = f"{RED}FAIL{RESET}"

    print(f"  {icon} {label}  — {result['detail']}  ({result['duration_s']}s)")
    if not result["passed"]:
        tail = result.get("stdout_tail", "")
        if tail:
            print(f"\n{YELLOW}  --- output tail ---{RESET}")
            for line in tail.splitlines()[-30:]:
                print(f"  {line}")
        etail = result.get("stderr_tail", "")
        if etail:
            print(f"\n{YELLOW}  --- stderr tail ---{RESET}")
            for line in etail.splitlines()[-10:]:
                print(f"  {line}")


def print_summary(results: list[dict], total_s: float) -> None:
    print_banner("CAAS Summary", BOLD)
    max_name = max((len(r["name"]) for r in results), default=10)
    for r in results:
        if r["passed"]:
            status = f"{GREEN}PASS{RESET}"
        elif r["status"] == "missing":
            status = f"{YELLOW}MISSING{RESET}"
        else:
            status = f"{RED}FAIL{RESET}"
        pad = " " * (max_name - len(r["name"]) + 2)
        print(f"  {status}  {r['name']}{pad}{r['detail']}  ({r['duration_s']}s)")
    print()
    overall = all(r["passed"] for r in results)
    overall_color = GREEN if overall else RED
    overall_text = "ALL STAGES PASSED" if overall else "AUDIT GATE VIOLATION — SEE ABOVE"
    print(f"  {overall_color}{BOLD}{overall_text}{RESET}  (total: {round(total_s, 1)}s)")


def build_json_report(results: list[dict], total_s: float) -> dict:
    overall_pass = all(r["passed"] for r in results)
    return {
        "caas_version": CAAS_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "overall_pass": overall_pass,
        "total_duration_s": round(total_s, 2),
        "stages": results,
    }


def main(argv: list[str] | None = None) -> int:
    import os

    _resolve_stage_fns()

    parser = argparse.ArgumentParser(
        description="CAAS runner — continuous audit gate for UltrafastSecp256k1"
    )
    parser.add_argument("--json", action="store_true", help="Output JSON report to stdout")
    parser.add_argument("-o", "--output", metavar="FILE", help="Write JSON report to file")
    parser.add_argument(
        "--no-fail-fast",
        action="store_true",
        help="Run all stages even after a failure (default: stop at first failure)",
    )
    parser.add_argument(
        "--skip-bundle",
        action="store_true",
        help="Skip bundle produce/verify stages (faster local iteration)",
    )
    parser.add_argument(
        "--stage",
        metavar="ID",
        help="Run only the stage with this ID (scanner|audit_gate|security_autonomy|bundle_produce|bundle_verify)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=int(os.environ.get("CAAS_TIMEOUT", "300")),
        metavar="SECONDS",
        help="Per-stage timeout in seconds (default: 300)",
    )
    args = parser.parse_args(argv)

    fail_fast = not args.no_fail_fast
    env_ff = os.environ.get("CAAS_FAIL_FAST", "1")
    if env_ff == "0":
        fail_fast = False

    # Select stages
    stages = list(STAGES)
    if args.stage:
        stages = [s for s in stages if s["id"] == args.stage]
        if not stages:
            print(f"{RED}ERROR: Unknown stage '{args.stage}'{RESET}", file=sys.stderr)
            valid = ", ".join(s["id"] for s in STAGES)
            print(f"Valid stages: {valid}", file=sys.stderr)
            return 2
    if args.skip_bundle:
        stages = [s for s in stages if not s.get("skippable")]

    if not args.json:
        print_banner(f"CAAS  v{CAAS_VERSION}  —  Continuous Audit as a Service")

    results: list[dict] = []
    t_global = time.monotonic()
    aborted_at = None

    for idx, stage in enumerate(stages, 1):
        if not args.json:
            print_stage_header(idx, len(stages), stage["name"])

        result = run_stage(stage, args.timeout)
        results.append(result)

        if not args.json:
            print_result(result)

        if not result["passed"] and stage.get("blocking", True) and fail_fast:
            aborted_at = stage["id"]
            # Mark remaining stages as skipped
            for remaining in stages[idx:]:
                results.append({
                    "id": remaining["id"],
                    "name": remaining["name"],
                    "status": "skipped",
                    "passed": False,
                    "detail": f"skipped — fail-fast triggered by '{aborted_at}'",
                    "duration_s": 0.0,
                })
            break

    total_s = time.monotonic() - t_global
    overall_pass = all(r["passed"] for r in results if r.get("status") != "skipped")

    if not args.json:
        print_summary(results, total_s)
        if aborted_at:
            print(
                f"\n{YELLOW}  Fail-fast triggered at stage '{aborted_at}'."
                f" Use --no-fail-fast to run all stages.{RESET}\n"
            )

    report = build_json_report(results, total_s)

    if args.json:
        print(json.dumps(report, indent=2))

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        if not args.json:
            print(f"\n  Report written to: {out_path}")

    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
