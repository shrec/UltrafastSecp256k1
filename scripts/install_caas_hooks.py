#!/usr/bin/env python3
"""
install_caas_hooks.py  --  Install CAAS git hooks locally

Installs a pre-push hook that runs caas_runner.py (Stages 1–3, fast mode)
before every `git push`. If any stage fails, the push is blocked.

The hook skips the bundle stages (--skip-bundle) to keep the pre-push
latency low. Full bundle checks run in CI via caas.yml.

Usage:
    python3 scripts/install_caas_hooks.py          # install
    python3 scripts/install_caas_hooks.py --remove # uninstall
    python3 scripts/install_caas_hooks.py --status # check if installed

The hook is installed at .git/hooks/pre-push.
If a pre-push hook already exists it is backed up to pre-push.caas-backup.
"""

from __future__ import annotations

import argparse
import os
import stat
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
LIB_ROOT = SCRIPT_DIR.parent

HOOK_TEMPLATE = """\
#!/usr/bin/env bash
# CAAS pre-push hook — installed by install_caas_hooks.py
# Runs audit_test_quality_scanner, audit_gate, and security_autonomy_check
# before every push.  Any failure blocks the push.
#
# To bypass (emergency only):
#   git push --no-verify
# To uninstall:
#   python3 scripts/install_caas_hooks.py --remove

REPO_ROOT="$(git rev-parse --show-toplevel)"
RUNNER="$REPO_ROOT/scripts/caas_runner.py"

if [ ! -f "$RUNNER" ]; then
    echo "[CAAS pre-push] WARNING: caas_runner.py not found — hook skipped"
    exit 0
fi

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║  CAAS pre-push gate — running audit pipeline    ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

# Skip bundle stages — full bundle runs in CI
python3 "$RUNNER" --skip-bundle
EXIT=$?

if [ $EXIT -ne 0 ]; then
    echo ""
    echo "╔══════════════════════════════════════════════════╗"
    echo "║  PUSH BLOCKED — CAAS gate failed               ║"
    echo "║  Fix all findings before pushing.               ║"
    echo "║  Emergency bypass: git push --no-verify         ║"
    echo "╚══════════════════════════════════════════════════╝"
    echo ""
    exit 1
fi

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║  CAAS gate PASSED — push proceeding            ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""
exit 0
"""

CAAS_MARKER = "CAAS pre-push hook"


def git_dir() -> Path:
    """Find the .git directory (or module git dir) for the library repo.

    For a submodule, `git rev-parse --git-dir` returns the module's actual
    git directory rather than the parent workspace .git directory.
    """
    import subprocess as _sp
    try:
        result = _sp.run(
            ["git", "rev-parse", "--git-dir"],
            capture_output=True,
            text=True,
            cwd=str(LIB_ROOT),
            timeout=5,
        )
        if result.returncode == 0:
            gd = result.stdout.strip()
            gd_path = Path(gd)
            # git may return a relative or absolute path
            if not gd_path.is_absolute():
                gd_path = (LIB_ROOT / gd_path).resolve()
            if gd_path.is_dir():
                return gd_path
    except Exception:
        pass

    # Fallback: walk up looking for .git directory
    for candidate_root in [LIB_ROOT] + list(LIB_ROOT.parents):
        candidate = candidate_root / ".git"
        if candidate.is_dir():
            return candidate
    raise RuntimeError(f"No .git directory found starting from {LIB_ROOT}")


def hook_path() -> Path:
    return git_dir() / "hooks" / "pre-push"


def is_installed() -> bool:
    h = hook_path()
    if not h.exists():
        return False
    return CAAS_MARKER in h.read_text(encoding="utf-8", errors="ignore")


def install() -> None:
    h = hook_path()
    hooks_dir = h.parent
    hooks_dir.mkdir(parents=True, exist_ok=True)

    if h.exists():
        if CAAS_MARKER in h.read_text(encoding="utf-8", errors="ignore"):
            print("CAAS pre-push hook is already installed.")
            return
        # Back up existing hook
        backup = h.with_name("pre-push.caas-backup")
        backup.write_bytes(h.read_bytes())
        print(f"Existing pre-push hook backed up to: {backup}")

    h.write_text(HOOK_TEMPLATE, encoding="utf-8")
    h.chmod(h.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    print(f"CAAS pre-push hook installed at: {h}")
    print("  Stages: scanner + audit_gate + security_autonomy (bundle skipped for speed)")
    print("  Bypass: git push --no-verify")


def remove() -> None:
    h = hook_path()
    if not h.exists():
        print("No pre-push hook found — nothing to remove.")
        return
    if CAAS_MARKER not in h.read_text(encoding="utf-8", errors="ignore"):
        print("Existing pre-push hook is NOT the CAAS hook — not removing.")
        return

    # Restore backup if it exists
    backup = h.with_name("pre-push.caas-backup")
    if backup.exists():
        h.write_bytes(backup.read_bytes())
        # Restore executable bit
        h.chmod(h.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        backup.unlink()
        print(f"Restored previous pre-push hook from backup.")
    else:
        h.unlink()
        print("CAAS pre-push hook removed.")


def status() -> None:
    h = hook_path()
    if not h.exists():
        print(f"pre-push hook: NOT PRESENT ({h})")
    elif is_installed():
        print(f"pre-push hook: CAAS INSTALLED ({h})")
    else:
        print(f"pre-push hook: EXISTS (not CAAS) ({h})")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Install/remove/check the CAAS git pre-push hook"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--remove", action="store_true", help="Uninstall the CAAS hook")
    group.add_argument("--status", action="store_true", help="Check installation status")
    args = parser.parse_args(argv)

    try:
        if args.remove:
            remove()
        elif args.status:
            status()
        else:
            install()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
