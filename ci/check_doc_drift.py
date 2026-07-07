#!/usr/bin/env python3
"""
check_doc_drift.py — Doc Drift Gate
=====================================
Detects stale references in README / docs that are not caught by other gates:

  1. Codecov badge in README but no .github/workflows/codecov.yml
  2. References to removed workflow files (pipeline.yml, gcc-analyzer.yml)

Exit codes:
  0 — no drift found
  1 — one or more drift findings
"""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"
README        = REPO_ROOT / "README.md"
DOCS_DIR      = REPO_ROOT / "docs"

FINDINGS: list[str] = []


def check_codecov_badge() -> None:
    """Fail if README references a codecov badge but codecov.yml is absent."""
    if not README.exists():
        return
    text = README.read_text(encoding="utf-8")
    has_badge = "codecov.io" in text
    has_workflow = (WORKFLOWS_DIR / "codecov.yml").exists()
    if has_badge and not has_workflow:
        FINDINGS.append(
            "README contains a codecov badge but .github/workflows/codecov.yml is missing. "
            "Either restore codecov.yml or remove the badge."
        )


def check_removed_workflows() -> None:
    """Fail if any doc references a workflow file that no longer exists.

    Exemption: lines that explicitly note the file was removed/deleted are
    changelog tombstones and are intentional — they are not flagged.
    """
    REMOVED = ["pipeline.yml", "gcc-analyzer.yml"]
    EXEMPT_PHRASES = ("was subsequently removed", "was removed", "has been removed", "no longer exists")
    docs_to_scan = [README] + list(DOCS_DIR.glob("*.md"))
    for removed in REMOVED:
        if (WORKFLOWS_DIR / removed).exists():
            continue
        for doc in docs_to_scan:
            try:
                text = doc.read_text(encoding="utf-8")
            except OSError:
                continue
            for line in text.splitlines():
                if removed not in line:
                    continue
                if any(phrase in line for phrase in EXEMPT_PHRASES):
                    continue
                FINDINGS.append(
                    f"{doc.relative_to(REPO_ROOT)} references '{removed}' "
                    f"but that workflow file no longer exists."
                )
                break


def check_module_count_drift() -> None:
    """Fail if canonical module/exploit counts would rewrite docs."""
    script = REPO_ROOT / "ci" / "sync_module_count.py"
    if not script.exists():
        FINDINGS.append("ci/sync_module_count.py is missing; cannot verify module-count doc drift.")
        return

    result = subprocess.run(
        [sys.executable, str(script), "--dry-run"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode == 0:
        return

    output = (result.stdout + result.stderr).strip()
    tail = "\n".join(output.splitlines()[-8:])
    FINDINGS.append(
        "Canonical module/exploit count drift detected. "
        "Run `python3 ci/sync_module_count.py` or `python3 ci/sync_all_docs.py`.\n"
        f"{tail}"
    )


def main() -> int:
    check_codecov_badge()
    check_removed_workflows()
    check_module_count_drift()

    if FINDINGS:
        print(f"Doc drift gate: {len(FINDINGS)} finding(s)")
        for f in FINDINGS:
            print(f"  FAIL  {f}")
        return 1

    print("Doc drift gate: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
