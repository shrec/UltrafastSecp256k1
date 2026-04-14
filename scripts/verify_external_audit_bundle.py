#!/usr/bin/env python3
"""Verify an external-audit bundle independently.

Validates:
  1. Detached bundle digest (SHA-256)
  2. Evidence file existence + SHA-256 hashes
  3. Optional replay of bundled gate commands by hash comparison

Exit code:
  0 when verification passes
  1 when any required verification check fails
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
LIB_ROOT = SCRIPT_DIR.parent
DEFAULT_BUNDLE = LIB_ROOT / "docs" / "EXTERNAL_AUDIT_BUNDLE.json"
DEFAULT_DIGEST = LIB_ROOT / "docs" / "EXTERNAL_AUDIT_BUNDLE.sha256"


def _sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_head() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(LIB_ROOT),
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        if result.returncode != 0:
            return ""
        return result.stdout.strip()
    except Exception:
        return ""


def _parse_digest(path: Path) -> tuple[str, str]:
    line = path.read_text(encoding="utf-8").strip()
    if not line:
        return "", ""
    parts = line.split()
    if len(parts) < 2:
        return "", ""
    return parts[0], parts[-1]


def _cmd_for_name(name: str) -> list[str] | None:
    mapping = {
        "audit_gate": [sys.executable, "scripts/audit_gate.py", "--json"],
        "audit_gap_report_strict": [sys.executable, "scripts/audit_gap_report.py", "--json", "--strict"],
        "security_autonomy_check": [sys.executable, "scripts/security_autonomy_check.py", "--json"],
        "supply_chain_gate": [sys.executable, "scripts/supply_chain_gate.py", "--json"],
    }
    return mapping.get(name)


def verify(
    bundle_path: Path,
    digest_path: Path,
    replay_commands: bool,
    allow_commit_mismatch: bool,
    json_mode: bool,
) -> int:
    checks: list[dict[str, object]] = []

    if not bundle_path.exists():
        checks.append({"name": "bundle_exists", "passing": False, "detail": f"missing: {bundle_path}"})
        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "overall_pass": False,
            "checks": checks,
        }
        print(json.dumps(report, indent=2) if json_mode else f"FAIL missing bundle: {bundle_path}")
        return 1

    raw_bundle = bundle_path.read_bytes()
    bundle_hash = _sha256_bytes(raw_bundle)
    bundle = json.loads(raw_bundle.decode("utf-8"))

    if not digest_path.exists():
        checks.append({"name": "bundle_digest_exists", "passing": False, "detail": f"missing: {digest_path}"})
    else:
        expected_hash, expected_name = _parse_digest(digest_path)
        digest_ok = bool(expected_hash) and expected_name == bundle_path.name and expected_hash == bundle_hash
        checks.append(
            {
                "name": "bundle_digest",
                "passing": digest_ok,
                "expected_hash": expected_hash,
                "actual_hash": bundle_hash,
                "expected_name": expected_name,
            }
        )

    bundle_commit = str(bundle.get("git", {}).get("commit", ""))
    current_commit = _git_head()
    commit_ok = allow_commit_mismatch or (bundle_commit != "" and bundle_commit == current_commit)
    checks.append(
        {
            "name": "commit_match",
            "passing": commit_ok,
            "bundle_commit": bundle_commit,
            "current_commit": current_commit,
            "allow_commit_mismatch": allow_commit_mismatch,
        }
    )

    for ev in bundle.get("evidence", []):
        rel = str(ev.get("path", ""))
        expected_hash = str(ev.get("sha256", ""))
        full = LIB_ROOT / rel
        if not full.exists():
            checks.append({"name": f"evidence:{rel}", "passing": False, "detail": "missing file"})
            continue
        actual_hash = _sha256_file(full)
        checks.append(
            {
                "name": f"evidence:{rel}",
                "passing": actual_hash == expected_hash,
                "expected_hash": expected_hash,
                "actual_hash": actual_hash,
            }
        )

    if replay_commands:
        for gate in bundle.get("gate_results", []):
            name = str(gate.get("name", ""))
            cmd = _cmd_for_name(name)
            if not cmd:
                checks.append({"name": f"replay:{name}", "passing": False, "detail": "unknown gate command"})
                continue
            try:
                result = subprocess.run(
                    cmd,
                    cwd=str(LIB_ROOT),
                    capture_output=True,
                    text=True,
                    timeout=900,
                    check=False,
                )
                stdout_hash = _sha256_bytes(result.stdout.encode("utf-8", errors="replace"))
                stderr_hash = _sha256_bytes(result.stderr.encode("utf-8", errors="replace"))
                checks.append(
                    {
                        "name": f"replay:{name}",
                        "passing": (
                            result.returncode == int(gate.get("returncode", -999))
                            and stdout_hash == str(gate.get("stdout_sha256", ""))
                            and stderr_hash == str(gate.get("stderr_sha256", ""))
                        ),
                        "expected_returncode": gate.get("returncode"),
                        "actual_returncode": result.returncode,
                        "expected_stdout_sha256": gate.get("stdout_sha256"),
                        "actual_stdout_sha256": stdout_hash,
                        "expected_stderr_sha256": gate.get("stderr_sha256"),
                        "actual_stderr_sha256": stderr_hash,
                    }
                )
            except Exception as exc:
                checks.append({"name": f"replay:{name}", "passing": False, "detail": str(exc)})

    overall_pass = all(bool(c.get("passing", False)) for c in checks)
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "bundle": str(bundle_path),
        "digest": str(digest_path),
        "overall_pass": overall_pass,
        "checks_total": len(checks),
        "checks_passing": sum(1 for c in checks if c.get("passing", False)),
        "checks": checks,
    }

    if json_mode:
        print(json.dumps(report, indent=2))
    else:
        print(f"overall_pass: {overall_pass}")
        print(f"checks: {report['checks_passing']}/{report['checks_total']}")
        for c in checks:
            if not c.get("passing", False):
                print(f"  FAIL {c.get('name')} :: {c.get('detail', '')}")

    return 0 if overall_pass else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", default=str(DEFAULT_BUNDLE), help="Path to EXTERNAL_AUDIT_BUNDLE.json")
    parser.add_argument("--digest", default=str(DEFAULT_DIGEST), help="Path to EXTERNAL_AUDIT_BUNDLE.sha256")
    parser.add_argument("--replay-commands", action="store_true", help="Re-run bundled gate commands and verify output hashes")
    parser.add_argument("--allow-commit-mismatch", action="store_true", help="Do not fail when current HEAD differs from bundle commit")
    parser.add_argument("--json", action="store_true", help="JSON output")
    args = parser.parse_args()

    return verify(
        bundle_path=Path(args.bundle),
        digest_path=Path(args.digest),
        replay_commands=args.replay_commands,
        allow_commit_mismatch=args.allow_commit_mismatch,
        json_mode=args.json,
    )


if __name__ == "__main__":
    raise SystemExit(main())
