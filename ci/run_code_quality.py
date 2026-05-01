#!/usr/bin/env python3
"""
run_code_quality.py  —  Unified code-quality gate for UltrafastSecp256k1.

Orchestrates all static-analysis scanners in one invocation:
  1. dev_bug_scanner.py        (27 classic development-bug patterns)
  2. hot_path_alloc_scanner.py (heap allocations on hot paths)
  3. audit_test_quality_scanner.py (audit test quality checks)

Features:
  • Combined JSON report with per-scanner sections
  • Baseline comparison: fail CI only on regressions (count exceeds baseline)
  • GitHub Actions step-summary output (--github-summary)
  • Configurable severity threshold (--fail-on-severity)
  • Machine-readable exit codes

Exit codes:
  0  All clear (no regressions, no new HIGH findings)
  1  Regression detected or HIGH threshold exceeded
  2  Scanner execution error

Usage:
  python3 ci/run_code_quality.py                      # human-readable
  python3 ci/run_code_quality.py --json -o report.json # JSON report
  python3 ci/run_code_quality.py --fail-on-regression  # CI gate mode
  python3 ci/run_code_quality.py --update-baseline     # refresh baseline
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
LIB_ROOT = SCRIPT_DIR.parent
BASELINE_PATH = SCRIPT_DIR / "code_quality_baseline.json"

# Scanner registry: (script_name, label, severity_key_normalizer)
SCANNERS = [
    ("dev_bug_scanner.py", "dev_bug_scanner"),
    ("hot_path_alloc_scanner.py", "hot_path_alloc_scanner"),
    ("audit_test_quality_scanner.py", "audit_test_quality_scanner"),
]

# Color codes
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
RESET = "\033[0m"
NO_COLOR = os.environ.get("NO_COLOR", "") != ""


def c(code: str, text: str) -> str:
    if NO_COLOR:
        return text
    return f"{code}{text}{RESET}"


def load_baseline() -> Dict[str, Dict[str, int]]:
    if BASELINE_PATH.exists():
        with open(BASELINE_PATH) as f:
            data = json.load(f)
        # Strip metadata keys
        return {k: v for k, v in data.items() if not k.startswith("_")}
    return {}


def save_baseline(results: Dict[str, Dict[str, int]]) -> None:
    data: Dict[str, Any] = {
        "_comment": "Known-good baseline: CI fails only if counts INCREASE above "
                    "these values. Updated automatically by run_code_quality.py "
                    "--update-baseline.",
        "_updated": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    }
    data.update(results)
    with open(BASELINE_PATH, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def normalize_severity(sev: str) -> str:
    """Normalize severity to uppercase (audit_test_quality uses lowercase)."""
    return sev.upper() if sev else "LOW"


def run_scanner(script_name: str) -> Dict[str, Any]:
    """Run a single scanner and return parsed results."""
    script_path = SCRIPT_DIR / script_name
    if not script_path.exists():
        return {"error": f"{script_name} not found", "findings": []}

    try:
        result = subprocess.run(
            [sys.executable, str(script_path), "--json"],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(LIB_ROOT),
        )
    except subprocess.TimeoutExpired:
        return {"error": f"{script_name} timed out (300s)", "findings": []}
    except Exception as exc:
        return {"error": f"{script_name} failed: {exc}", "findings": []}

    if result.stdout.strip():
        try:
            data = json.loads(result.stdout)
            findings = data if isinstance(data, list) else data.get("findings", [])
            return {"findings": findings}
        except json.JSONDecodeError:
            return {"error": f"{script_name} produced invalid JSON", "findings": []}
    return {"findings": []}


def count_by_severity(findings: List[Dict]) -> Dict[str, int]:
    """Count findings by normalized severity, always including all levels."""
    counts: Counter = Counter()
    for f in findings:
        sev = normalize_severity(f.get("severity", "LOW"))
        counts[sev] += 1
    return {
        "HIGH": counts.get("HIGH", 0),
        "MEDIUM": counts.get("MEDIUM", 0),
        "LOW": counts.get("LOW", 0),
        "total": len(findings),
    }


def check_regressions(
    current: Dict[str, Dict[str, int]],
    baseline: Dict[str, Dict[str, int]],
) -> List[str]:
    """Compare current counts against baseline, return list of regression messages."""
    regressions: List[str] = []
    for scanner_name, cur_counts in current.items():
        base = baseline.get(scanner_name, {})
        for sev in ("HIGH", "MEDIUM", "LOW"):
            cur_val = cur_counts.get(sev, 0)
            base_val = base.get(sev, 0)
            if cur_val > base_val:
                regressions.append(
                    f"{scanner_name}: {sev} increased {base_val} → {cur_val} "
                    f"(+{cur_val - base_val})"
                )
    return regressions


def format_table_text(
    results: Dict[str, Dict[str, Any]],
    severity_counts: Dict[str, Dict[str, int]],
    baseline: Dict[str, Dict[str, int]],
) -> str:
    """Format human-readable results table."""
    lines: List[str] = []
    lines.append(f"\n{c(BOLD, '=' * 64)}")
    lines.append(f"{c(BOLD, '  UltrafastSecp256k1 Code Quality Report')}")
    lines.append(f"{c(BOLD, '=' * 64)}\n")

    grand_high = 0
    grand_medium = 0
    grand_low = 0
    grand_total = 0

    for script_name, label in SCANNERS:
        data = results.get(label, {})
        counts = severity_counts.get(label, {})
        base = baseline.get(label, {})

        lines.append(f"{c(BOLD, f'  [{label}]')}")

        if "error" in data:
            lines.append(f"    {c(RED, 'ERROR')}: {data['error']}")
            lines.append("")
            continue

        h, m, lo, t = counts.get("HIGH", 0), counts.get("MEDIUM", 0), counts.get("LOW", 0), counts.get("total", 0)
        bh, bm, blo = base.get("HIGH", 0), base.get("MEDIUM", 0), base.get("LOW", 0)

        def delta_str(cur: int, base_val: int) -> str:
            diff = cur - base_val
            if diff > 0:
                return c(RED, f" (+{diff} REGRESSION)")
            elif diff < 0:
                return c(GREEN, f" ({diff} improved)")
            return ""

        lines.append(f"    HIGH:   {h}{delta_str(h, bh)}")
        lines.append(f"    MEDIUM: {m}{delta_str(m, bm)}")
        lines.append(f"    LOW:    {lo}{delta_str(lo, blo)}")
        lines.append(f"    Total:  {t}")
        lines.append("")

        grand_high += h
        grand_medium += m
        grand_low += lo
        grand_total += t

    lines.append(f"{c(BOLD, '  Grand Total')}")
    lines.append(f"    HIGH: {grand_high}  MEDIUM: {grand_medium}  LOW: {grand_low}  Total: {grand_total}")
    lines.append("")

    return "\n".join(lines)


def format_github_summary(
    severity_counts: Dict[str, Dict[str, int]],
    baseline: Dict[str, Dict[str, int]],
    regressions: List[str],
) -> str:
    """Format GitHub Actions step summary in Markdown."""
    lines: List[str] = []
    lines.append("## Code Quality Report")
    lines.append("")
    lines.append("| Scanner | HIGH | MEDIUM | LOW | Total |")
    lines.append("|---------|------|--------|-----|-------|")

    for _, label in SCANNERS:
        counts = severity_counts.get(label, {})
        base = baseline.get(label, {})

        def cell(sev: str) -> str:
            cur = counts.get(sev, 0)
            bv = base.get(sev, 0)
            diff = cur - bv
            if diff > 0:
                return f"**{cur}** (+{diff} :red_circle:)"
            elif diff < 0:
                return f"{cur} ({diff} :green_circle:)"
            return str(cur)

        lines.append(
            f"| {label} | {cell('HIGH')} | {cell('MEDIUM')} | {cell('LOW')} | {counts.get('total', 0)} |"
        )

    lines.append("")
    if regressions:
        lines.append("### :red_circle: Regressions Detected")
        for r in regressions:
            lines.append(f"- {r}")
    else:
        lines.append(":green_circle: **No regressions against baseline**")
    lines.append("")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Unified code-quality gate for UltrafastSecp256k1"
    )
    parser.add_argument("--json", action="store_true", help="Emit combined JSON report")
    parser.add_argument("-o", "--output", metavar="FILE", help="Write report to FILE")
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit 1 if any severity count exceeds baseline",
    )
    parser.add_argument(
        "--fail-on-severity",
        choices=["HIGH", "MEDIUM", "LOW"],
        default=None,
        help="Exit 1 if any finding at or above this severity exists",
    )
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Update baseline file with current counts (run after fixing)",
    )
    parser.add_argument(
        "--github-summary",
        action="store_true",
        help="Write Markdown summary to $GITHUB_STEP_SUMMARY",
    )
    parser.add_argument(
        "--scanner",
        action="append",
        choices=[label for _, label in SCANNERS],
        help="Run only this scanner (repeatable; default: all)",
    )
    args = parser.parse_args()

    baseline = load_baseline()

    # Determine which scanners to run
    active_scanners = SCANNERS
    if args.scanner:
        active_scanners = [(s, l) for s, l in SCANNERS if l in args.scanner]

    # Run all scanners
    results: Dict[str, Dict[str, Any]] = {}
    severity_counts: Dict[str, Dict[str, int]] = {}
    all_findings: List[Dict] = []
    errors: List[str] = []

    t0 = time.monotonic()
    for script_name, label in active_scanners:
        print(f"Running {label}...", file=sys.stderr) if not args.json else None
        data = run_scanner(script_name)
        results[label] = data

        if "error" in data:
            errors.append(data["error"])

        findings = data.get("findings", [])
        # Tag findings with scanner source
        for f in findings:
            f["_scanner"] = label
        all_findings.extend(findings)
        severity_counts[label] = count_by_severity(findings)

    elapsed = time.monotonic() - t0

    # Check regressions
    regressions = check_regressions(severity_counts, baseline)

    # Update baseline if requested
    if args.update_baseline:
        save_baseline(severity_counts)
        print(f"Baseline updated: {BASELINE_PATH}", file=sys.stderr)

    # Output
    exit_code = 0

    if args.json:
        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": round(elapsed, 2),
            "scanners": {
                label: {
                    "counts": severity_counts.get(label, {}),
                    "baseline": baseline.get(label, {}),
                    "error": results.get(label, {}).get("error"),
                }
                for _, label in active_scanners
            },
            "regressions": regressions,
            "total_findings": len(all_findings),
            "findings": all_findings,
        }
        output = json.dumps(report, indent=2)
    else:
        output = format_table_text(results, severity_counts, baseline)
        output += f"  Scan time: {elapsed:.1f}s\n"

    if args.output:
        Path(args.output).write_text(output)
        if not args.json:
            print(output)
            print(f"Report written to {args.output}")
    else:
        print(output)

    # GitHub step summary
    if args.github_summary:
        summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
        if summary_path:
            md = format_github_summary(severity_counts, baseline, regressions)
            with open(summary_path, "a") as f:
                f.write(md)

    # Exit code logic
    if errors:
        exit_code = 2

    if args.fail_on_regression and regressions:
        if not args.json:
            for r in regressions:
                print(f"  {c(RED, 'REGRESSION')}: {r}")
        exit_code = 1

    if args.fail_on_severity:
        threshold = args.fail_on_severity
        rank = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
        threshold_rank = rank[threshold]
        for _, label in active_scanners:
            counts = severity_counts.get(label, {})
            for sev in ("HIGH", "MEDIUM", "LOW"):
                if rank.get(sev, 0) >= threshold_rank and counts.get(sev, 0) > 0:
                    exit_code = 1
                    break

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
