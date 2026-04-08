#!/usr/bin/env python3
"""
sync_audit_report_version.py  --  Sync AUDIT_REPORT.md version with VERSION.txt
"""

from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parent.parent
VERSION_PATH = ROOT / "VERSION.txt"
REPORT_PATH = ROOT / "AUDIT_REPORT.md"

BEGIN_MARKER = "<!-- BEGIN CURRENT RELEASE VERSION -->"
END_MARKER = "<!-- END CURRENT RELEASE VERSION -->"


def main() -> int:
    version = VERSION_PATH.read_text(encoding="utf-8").strip()
    report = REPORT_PATH.read_text(encoding="utf-8")

    block_pattern = re.compile(
        rf"{re.escape(BEGIN_MARKER)}.*?{re.escape(END_MARKER)}",
        re.DOTALL,
    )

    replacement = (
        f"{BEGIN_MARKER}\n"
        "<!-- This line is maintained by scripts/sync_audit_report_version.py -->\n"
        f"**Current Release Version:** {version}\n"
        f"{END_MARKER}"
    )

    if not block_pattern.search(report):
        print("sync_audit_report_version.py: version marker block not found", file=sys.stderr)
        return 1

    updated = block_pattern.sub(replacement, report, count=1)

    # Keep the visible summary line in sync too.
    updated = re.sub(
        r"^\*\*Current Release Version:\*\* .*?$",
        f"**Current Release Version:** {version}  ",
        updated,
        count=1,
        flags=re.MULTILINE,
    )

    REPORT_PATH.write_text(updated, encoding="utf-8")
    print(f"Updated {REPORT_PATH.name} to current release version {version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())