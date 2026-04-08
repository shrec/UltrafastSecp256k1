#!/usr/bin/env python3
"""
stamp_changelog.py — Replace the [Unreleased] section header in CHANGELOG.md
with a versioned header: [X.Y.Z] — YYYY-MM-DD.

Usage:
    python3 scripts/stamp_changelog.py --version 3.62.0
    python3 scripts/stamp_changelog.py              # reads VERSION.txt

If no [Unreleased] section exists, the script exits successfully (no-op).
After stamping, a fresh empty [Unreleased] section is inserted above the
newly versioned section so the changelog is ready for future development.

Called automatically by the release CI workflow (release.yml) in the
sync-docs job.
"""

import argparse
import re
import sys
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Stamp CHANGELOG.md [Unreleased] → version")
    parser.add_argument("--version", help="Version string (e.g. 3.62.0). Default: read VERSION.txt")
    parser.add_argument("--changelog", default="CHANGELOG.md", help="Path to CHANGELOG.md")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    changelog = root / args.changelog
    if not changelog.exists():
        print(f"stamp_changelog: {changelog} not found — skipping")
        return 0

    version = args.version
    if not version:
        vtxt = root / "VERSION.txt"
        if vtxt.exists():
            version = vtxt.read_text().strip()
        else:
            print("stamp_changelog: no --version and no VERSION.txt — skipping")
            return 0

    text = changelog.read_text(encoding="utf-8")

    # Match ## [Unreleased] (case-insensitive, optional trailing text)
    pattern = re.compile(r"^(## \[)[Uu]nreleased\](.*)$", re.MULTILINE)
    m = pattern.search(text)
    if not m:
        print(f"stamp_changelog: no [Unreleased] section found — nothing to stamp")
        return 0

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    versioned_header = f"## [{version}] — {today}"

    # Replace [Unreleased] header with versioned header
    new_text = text[:m.start()] + versioned_header + text[m.end():]

    # Insert a fresh empty [Unreleased] section above the versioned one
    fresh_unreleased = "## [Unreleased]\n\n"
    insert_pos = new_text.find(versioned_header)
    new_text = new_text[:insert_pos] + fresh_unreleased + new_text[insert_pos:]

    changelog.write_text(new_text, encoding="utf-8")
    print(f"stamp_changelog: stamped [{version}] — {today}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
