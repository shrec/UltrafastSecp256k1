#!/usr/bin/env python3
"""check_readme_ci_badges.py -- README branch-aware CI badge regression gate.

Context (branch-aware-ci-badges-claude-v1):
README.md used to hardcode every GitHub Actions badge (Gate/CI/Security
Audit/CAAS/CodeQL) to `?branch=dev`, so a viewer reading the README on the
`main` branch on GitHub was shown `dev`'s CI results with no indication of
the mismatch. The fix replaced the single badge row with two explicitly
labeled rows (`main` / `dev`), each carrying its own branch-qualified badge
image AND a branch-filtered workflow-run link.

This script is the regression gate that keeps that fix honest over time. It
parses the `<!-- CI-STATUS-TABLE-START -->` / `<!-- CI-STATUS-TABLE-END -->`
block in README.md and fails (non-zero exit) if any of the following regress:

  1. Either the `main` row or the `dev` row is missing.
  2. A row is not clearly labeled with its branch name (and, for extra
     clarity, the release/development qualifier).
  3. Any of the 5 required workflows (gate, ci, security-audit, caas,
     codeql) is missing from either row.
  4. A badge nested under the `main` row actually points at `branch=dev`
     (or vice versa) -- image src OR link href.
  5. The exact set of workflows shown differs between the two rows (parity
     -- a badge added to one row must be added to the other).

Usage:
    python3 ci/check_readme_ci_badges.py
    python3 ci/check_readme_ci_badges.py --readme /path/to/README.md
    python3 ci/check_readme_ci_badges.py --json -o report.json

Exit codes:
    0 = all checks passed
    1 = a regression was found (see stderr / findings for detail)
    2 = README.md or the CI-STATUS-TABLE markers could not be located
        (structural failure -- the table was removed/renamed)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
LIB_ROOT = SCRIPT_DIR.parent

START_MARKER = "<!-- CI-STATUS-TABLE-START -->"
END_MARKER = "<!-- CI-STATUS-TABLE-END -->"

# The 5 workflows every branch-status row must carry a badge for. Keep this
# list in sync with .github/workflows/*.yml -- if a new mandatory workflow
# is added to the branch-status table, add it here too.
REQUIRED_WORKFLOWS = ["gate.yml", "ci.yml", "security-audit.yml", "caas.yml", "codeql.yml"]

BRANCHES = {
    "main": {"label_terms": ["main"], "qualifier_terms": ["release"]},
    "dev": {"label_terms": ["dev"], "qualifier_terms": ["development"]},
}

# Matches markdown "linked image" badges: [![alt](img_url)](link_url)
BADGE_RE = re.compile(r"\[!\[([^\]]*)\]\(([^)\s]+)\)\]\(([^)\s]+)\)")
WORKFLOW_RE = re.compile(r"workflows/([A-Za-z0-9_.\-]+\.yml)")
BRANCH_PARAM_RE = re.compile(r"[?&]branch=([A-Za-z0-9_./\-]+)")
QUERY_BRANCH_RE = re.compile(r"query=branch%3A([A-Za-z0-9_./\-]+)")


def find_table_block(text: str) -> tuple[str, list[str]]:
    """Return (block_text, findings). block_text is '' if markers absent."""
    findings: list[str] = []
    start = text.find(START_MARKER)
    end = text.find(END_MARKER)
    if start == -1 or end == -1 or end < start:
        findings.append(
            f"CI-STATUS-TABLE markers not found or malformed in README.md "
            f"(expected '{START_MARKER}' ... '{END_MARKER}')"
        )
        return "", findings
    return text[start + len(START_MARKER):end], findings


def split_rows(block: str) -> list[str]:
    """Return the markdown table data rows (skip header + separator lines)."""
    rows = []
    for line in block.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        # Separator row looks like: |---|---|---|...
        cells = [c.strip() for c in line.strip("|").split("|")]
        if all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
            continue
        # Header row: first cell literally "Branch" (case-insensitive)
        if cells and cells[0].lower() == "branch":
            continue
        rows.append(line)
    return rows


def classify_row(row: str) -> str | None:
    """Return 'main', 'dev', or None if the row's first cell doesn't clearly
    identify a single branch."""
    cells = [c.strip() for c in row.strip("|").split("|")]
    if not cells:
        return None
    label = cells[0].lower()
    hit_main = "main" in label
    hit_dev = "dev" in label
    if hit_main and not hit_dev:
        return "main"
    if hit_dev and not hit_main:
        return "dev"
    return None


def extract_badges(row: str) -> list[dict]:
    badges = []
    for alt, img_url, link_url in BADGE_RE.findall(row):
        wf_match = WORKFLOW_RE.search(img_url) or WORKFLOW_RE.search(link_url)
        img_branch = BRANCH_PARAM_RE.search(img_url)
        link_branch = BRANCH_PARAM_RE.search(link_url) or QUERY_BRANCH_RE.search(link_url)
        badges.append({
            "alt": alt,
            "img_url": img_url,
            "link_url": link_url,
            "workflow": wf_match.group(1) if wf_match else None,
            "img_branch": img_branch.group(1) if img_branch else None,
            "link_branch": link_branch.group(1) if link_branch else None,
        })
    return badges


def check_readme_badges(readme_path: Path) -> dict:
    findings: list[str] = []
    if not readme_path.exists():
        return {
            "passed": False,
            "findings": [f"README not found: {readme_path}"],
            "structural_failure": True,
        }

    text = readme_path.read_text(encoding="utf-8")
    block, marker_findings = find_table_block(text)
    if marker_findings:
        return {"passed": False, "findings": marker_findings, "structural_failure": True}

    rows = split_rows(block)
    row_by_branch: dict[str, str] = {}
    for row in rows:
        branch = classify_row(row)
        if branch is None:
            findings.append(
                f"Row does not unambiguously label a single branch "
                f"(expected 'main' xor 'dev' in the first cell): {row[:120]!r}"
            )
            continue
        if branch in row_by_branch:
            findings.append(f"Duplicate '{branch}' row found in CI status table")
            continue
        row_by_branch[branch] = row

    for branch in ("main", "dev"):
        if branch not in row_by_branch:
            findings.append(f"'{branch}' row is missing from the CI status table")

    if findings:
        return {"passed": False, "findings": findings, "structural_failure": True}

    per_branch_workflows: dict[str, set] = {}

    for branch, row in row_by_branch.items():
        spec = BRANCHES[branch]
        cells = [c.strip() for c in row.strip("|").split("|")]
        label_cell = cells[0].lower()

        # Labeling check: branch name required; qualifier (release/development)
        # required for clarity per acceptance criteria #2/#5.
        for term in spec["label_terms"]:
            if term not in label_cell:
                findings.append(f"'{branch}' row label missing required term '{term}': {cells[0]!r}")
        for term in spec["qualifier_terms"]:
            if term not in label_cell:
                findings.append(
                    f"'{branch}' row label missing clarifying qualifier '{term}' "
                    f"(e.g. 'main (release)' / 'dev (development)'): {cells[0]!r}"
                )

        badges = extract_badges(row)
        found_workflows = set()
        for b in badges:
            wf = b["workflow"]
            if wf is None:
                continue
            found_workflows.add(wf)

            if b["img_branch"] is None:
                findings.append(f"'{branch}' row badge for {wf}: image URL has no branch= param ({b['img_url']})")
            elif b["img_branch"] != branch:
                findings.append(
                    f"'{branch}' row badge for {wf}: image URL points at "
                    f"branch={b['img_branch']!r} instead of {branch!r} -- {b['img_url']}"
                )

            if b["link_branch"] is None:
                findings.append(
                    f"'{branch}' row badge for {wf}: link target is not branch-filtered "
                    f"(expected a 'branch=' or 'query=branch%3A' param) -- {b['link_url']}"
                )
            elif b["link_branch"] != branch:
                findings.append(
                    f"'{branch}' row badge for {wf}: link target points at "
                    f"branch={b['link_branch']!r} instead of {branch!r} -- {b['link_url']}"
                )

        missing = [wf for wf in REQUIRED_WORKFLOWS if wf not in found_workflows]
        if missing:
            findings.append(f"'{branch}' row is missing required workflow badge(s): {missing}")

        extra = found_workflows - set(REQUIRED_WORKFLOWS)
        # Extra workflow badges are not an error by themselves (future workflows
        # may be added deliberately), but they must still be recorded so a
        # silent asymmetry between rows (see below) is caught.
        per_branch_workflows[branch] = found_workflows

    if not findings and per_branch_workflows.get("main") != per_branch_workflows.get("dev"):
        findings.append(
            "'main' and 'dev' rows show a different set of workflow badges -- "
            f"main={sorted(per_branch_workflows.get('main', set()))} "
            f"dev={sorted(per_branch_workflows.get('dev', set()))}"
        )

    return {
        "passed": not findings,
        "findings": findings,
        "structural_failure": False,
        "workflows_checked": REQUIRED_WORKFLOWS,
        "main_workflows_found": sorted(per_branch_workflows.get("main", set())),
        "dev_workflows_found": sorted(per_branch_workflows.get("dev", set())),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--readme", default=str(LIB_ROOT / "README.md"), help="Path to README.md")
    parser.add_argument("--json", action="store_true", help="Emit JSON report to stdout")
    parser.add_argument("-o", "--output", help="Write JSON report to this path (implies --json content)")
    args = parser.parse_args()

    readme_path = Path(args.readme)
    result = check_readme_badges(readme_path)

    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        if result["passed"]:
            print(
                "check_readme_ci_badges: PASS -- main/dev rows present, "
                f"{len(REQUIRED_WORKFLOWS)} required workflow badges verified in each, "
                "no cross-branch leakage"
            )
        else:
            print("check_readme_ci_badges: FAIL", file=sys.stderr)
            for f in result["findings"]:
                print(f"  - {f}", file=sys.stderr)

    if result.get("structural_failure"):
        return 2
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
