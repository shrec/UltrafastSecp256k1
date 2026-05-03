#!/usr/bin/env python3
"""
audit_verdict.py -- aggregate cross-platform audit reports for CI.

Rules:
  - PASS/AUDIT-READY reports are accepted.
  - Missing reports are fatal only when the producing job completed and should
    have uploaded an artifact.
  - Cancelled/skipped jobs are reported in the summary but do not fail the
    aggregate verdict.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path


PASS_VERDICTS = {"PASS", "pass", "AUDIT-READY"}
NON_FATAL_MISSING_RESULTS = {"cancelled", "skipped"}


@dataclass(frozen=True)
class PlatformSpec:
    artifact: str
    job_result: str

    @property
    def platform(self) -> str:
        return self.artifact.removeprefix("audit-report-")


def parse_platform_spec(raw: str) -> PlatformSpec:
    if "=" not in raw:
        raise argparse.ArgumentTypeError(
            f"platform spec must be artifact=result, got: {raw!r}"
        )
    artifact, result = raw.split("=", 1)
    artifact = artifact.strip()
    result = result.strip().lower()
    if not artifact.startswith("audit-report-"):
        raise argparse.ArgumentTypeError(
            f"artifact name must start with 'audit-report-': {artifact!r}"
        )
    if not result:
        raise argparse.ArgumentTypeError(f"missing job result in: {raw!r}")
    return PlatformSpec(artifact=artifact, job_result=result)


def load_verdict(json_path: Path) -> str | None:
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    # "audit_verdict" is the canonical field; "verdict" is a backward-compat alias
    # emitted by audit_gate.py for consumers that read only the older field name.
    verdict = data.get("audit_verdict") or data.get("verdict")
    return verdict if isinstance(verdict, str) else None


def append_summary(summary_file: Path | None, lines: list[str]) -> None:
    if summary_file is None:
        return
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines) + "\n"
    with summary_file.open("a", encoding="utf-8") as handle:
        handle.write(text)


def evaluate(
    platforms: list[PlatformSpec],
    artifact_root: Path,
    required_platforms: set[str],
) -> tuple[int, list[str]]:
    summary_lines = [
        "## Cross-Platform Audit Verdict",
        "",
        "| Platform | Verdict |",
        "|----------|---------|",
    ]
    all_pass = True
    observed_reports = 0

    for spec in platforms:
        report_json = artifact_root / spec.artifact / "audit_report.json"
        platform = spec.platform
        is_required = platform in required_platforms
        if report_json.is_file():
            observed_reports += 1
            verdict = load_verdict(report_json)
            verdict_display = verdict or "INVALID REPORT"
            label = f"{platform} (required)" if is_required else platform
            summary_lines.append(f"| {label} | {verdict_display} |")
            if verdict not in PASS_VERDICTS:
                print(
                    f"::error::Audit verdict FAILED on platform: {platform} "
                    f"(verdict: {verdict_display})"
                )
                all_pass = False
            continue

        status = spec.job_result
        label = f"{platform} (required)" if is_required else platform
        summary_lines.append(f"| {label} | NO REPORT ({status}) |")
        # A required platform must produce a report. skipped/cancelled is fatal
        # for required platforms (we cannot accept "all greenest job vanished"
        # as a pass), but remains non-fatal for optional platforms.
        if is_required:
            print(
                f"::error::Required platform {platform} produced no report "
                f"(job result: {status}) — fail-closed"
            )
            all_pass = False
            continue
        if status in NON_FATAL_MISSING_RESULTS:
            print(
                f"::warning::No audit report for platform: {platform} "
                f"because job result was {status}"
            )
            continue

        print(
            f"::error::No audit report generated for platform: {platform} "
            f"(job result: {status})"
        )
        all_pass = False

    downloaded = sorted(
        path.name for path in artifact_root.glob("audit-report-*") if path.is_dir()
    )
    if downloaded:
        summary_lines.extend(["", f"Downloaded artifacts: {' '.join(downloaded)}"])

    if observed_reports == 0:
        summary_lines.extend([
            "",
            "No usable audit_report.json artifact was produced on any platform.",
        ])
        print("::error::Audit verdict check failed -- no audit evidence was produced on any platform")
        all_pass = False

    summary_lines.append("")
    if all_pass:
        summary_lines.append("**Overall: PASS**")
        return 0, summary_lines

    summary_lines.append("**Overall: FAIL**")
    print("::error::Audit verdict check failed -- one or more required platforms did not PASS")
    return 1, summary_lines


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate multi-platform audit verdicts")
    parser.add_argument(
        "--artifact-root",
        type=Path,
        required=True,
        help="Directory containing downloaded audit-report-* artifact folders",
    )
    parser.add_argument(
        "--platform",
        action="append",
        default=[],
        type=parse_platform_spec,
        help="Platform spec in the form audit-report-<name>=<job-result>",
    )
    parser.add_argument(
        "--summary-file",
        type=Path,
        default=None,
        help="Optional GitHub step summary file to append markdown output to",
    )
    parser.add_argument(
        "--required-platform",
        action="append",
        default=[],
        help=(
            "Platform name (without 'audit-report-' prefix) that MUST produce "
            "a passing report; skipped/cancelled is fatal for these. "
            "May be passed multiple times. Default: linux-gcc13."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.platform:
        parser.error("at least one --platform must be provided")

    required = set(args.required_platform) or {"linux-gcc13"}
    exit_code, summary_lines = evaluate(args.platform, args.artifact_root, required)
    append_summary(args.summary_file, summary_lines)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())