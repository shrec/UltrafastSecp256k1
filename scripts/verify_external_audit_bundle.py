#!/usr/bin/env python3
"""Verify an external-audit bundle independently.

Validates:
  1. Detached bundle digest (SHA-256)
  2. Evidence file existence + SHA-256 hashes
  3. Optional replay of bundled gate commands by hash comparison
  4. Optional deep-replay of key pipeline scripts (--deep-replay)

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

# Default per-command timeout for deep-replay (seconds)
DEEP_REPLAY_TIMEOUT = 120


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


def _run_command(
    cmd: list[str],
    cwd: str,
    timeout: int,
) -> tuple[int, str, str, str | None]:
    """Run a command and return (returncode, stdout, stderr, error_msg).

    error_msg is None on success, a string description on exception.
    """
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return result.returncode, result.stdout, result.stderr, None
    except subprocess.TimeoutExpired:
        return -1, "", "", f"timed out after {timeout}s"
    except Exception as exc:
        return -1, "", "", str(exc)


def run_deep_replay(
    bundle: dict,
    timeout: int = DEEP_REPLAY_TIMEOUT,
) -> dict:
    """Re-run key pipeline scripts and compare to bundle's recorded results.

    Returns a dict with keys:
      commands_run, commands_passed, results (list of per-command dicts)
    """
    # Commands that are always attempted
    always_run: list[dict] = [
        {
            "label": "audit_gate",
            "cmd": [sys.executable, "scripts/audit_gate.py", "--json",
                    "-o", "/tmp/replay_audit_gate.json"],
            "required": True,
        },
    ]

    # Scripts run only if they exist
    optional_scripts = [
        "scripts/check_libsecp_shim_parity.py",
        "scripts/check_core_build_mode.py",
    ]

    commands = list(always_run)
    for rel in optional_scripts:
        p = LIB_ROOT / rel
        if p.exists():
            commands.append({
                "label": rel,
                "cmd": [sys.executable, rel],
                "required": False,
            })

    # Build a quick lookup from bundle gate_results by name
    bundle_gates: dict[str, dict] = {}
    for g in bundle.get("gate_results", []):
        name = str(g.get("name", ""))
        if name:
            bundle_gates[name] = g

    results: list[dict] = []
    passed = 0

    for entry in commands:
        label: str = entry["label"]
        cmd: list[str] = entry["cmd"]

        rc, stdout, stderr, err_msg = _run_command(cmd, str(LIB_ROOT), timeout)

        if err_msg is not None:
            results.append({
                "label": label,
                "pass": False,
                "returncode": rc,
                "error": err_msg,
                "comparison": None,
            })
            continue

        # Compare to bundle if there is a recorded result
        comparison: dict | None = None
        gate_key = label if "/" not in label else label.split("/")[-1].replace(".py", "")
        recorded = bundle_gates.get(label) or bundle_gates.get(gate_key)
        if recorded is not None:
            exp_rc = recorded.get("returncode")
            exp_stdout_sha = recorded.get("stdout_sha256", "")
            exp_stderr_sha = recorded.get("stderr_sha256", "")
            actual_stdout_sha = _sha256_bytes(stdout.encode("utf-8", errors="replace"))
            actual_stderr_sha = _sha256_bytes(stderr.encode("utf-8", errors="replace"))
            comparison = {
                "expected_returncode": exp_rc,
                "actual_returncode": rc,
                "returncode_match": rc == exp_rc,
                "expected_stdout_sha256": exp_stdout_sha,
                "actual_stdout_sha256": actual_stdout_sha,
                "stdout_match": actual_stdout_sha == exp_stdout_sha,
                "expected_stderr_sha256": exp_stderr_sha,
                "actual_stderr_sha256": actual_stderr_sha,
                "stderr_match": actual_stderr_sha == exp_stderr_sha,
            }

        # A command "passes" if it exits 0
        cmd_pass = rc == 0
        if cmd_pass:
            passed += 1

        results.append({
            "label": label,
            "pass": cmd_pass,
            "returncode": rc,
            "error": None,
            "comparison": comparison,
        })

    return {
        "commands_run": len(results),
        "commands_passed": passed,
        "results": results,
    }


def verify(
    bundle_path: Path,
    digest_path: Path,
    replay_commands: bool,
    allow_commit_mismatch: bool,
    json_mode: bool,
    deep_replay: bool = False,
    strict: bool = False,
    deep_replay_timeout: int = DEEP_REPLAY_TIMEOUT,
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

    # --- Deep replay ---
    deep_replay_result: dict | None = None
    deep_replay_warn = False

    if deep_replay:
        deep_replay_result = run_deep_replay(bundle, timeout=deep_replay_timeout)
        n_run = deep_replay_result["commands_run"]
        n_pass = deep_replay_result["commands_passed"]
        n_fail = n_run - n_pass
        if n_fail > 0:
            deep_replay_warn = True

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "bundle": str(bundle_path),
        "digest": str(digest_path),
        "overall_pass": overall_pass,
        "checks_total": len(checks),
        "checks_passing": sum(1 for c in checks if c.get("passing", False)),
        "checks": checks,
    }

    if deep_replay_result is not None:
        report["deep_replay"] = deep_replay_result

    # Determine final exit code
    exit_code = 0 if overall_pass else 1
    if deep_replay_warn and strict:
        exit_code = 1

    if json_mode:
        print(json.dumps(report, indent=2))
    else:
        print(f"overall_pass: {overall_pass}")
        print(f"checks: {report['checks_passing']}/{report['checks_total']}")
        for c in checks:
            if not c.get("passing", False):
                print(f"  FAIL {c.get('name')} :: {c.get('detail', '')}")
        if deep_replay_result is not None:
            n_run = deep_replay_result["commands_run"]
            n_pass = deep_replay_result["commands_passed"]
            status_prefix = "WARN" if deep_replay_warn and not strict else (
                "FAIL" if deep_replay_warn and strict else "OK"
            )
            print(f"Deep replay: {n_pass}/{n_run} commands passed  [{status_prefix}]")
            for r in deep_replay_result["results"]:
                icon = "PASS" if r["pass"] else "FAIL"
                err = f" ({r['error']})" if r.get("error") else ""
                print(f"  {icon} {r['label']}{err}")

    return exit_code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", default=str(DEFAULT_BUNDLE), help="Path to EXTERNAL_AUDIT_BUNDLE.json")
    parser.add_argument("--digest", default=str(DEFAULT_DIGEST), help="Path to EXTERNAL_AUDIT_BUNDLE.sha256")
    parser.add_argument("--replay-commands", action="store_true", help="Re-run bundled gate commands and verify output hashes")
    parser.add_argument("--allow-commit-mismatch", action="store_true", help="Do not fail when current HEAD differs from bundle commit")
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument(
        "--deep-replay",
        action="store_true",
        help=(
            "After regular verification, re-run key pipeline scripts "
            "(audit_gate, check_libsecp_shim_parity, check_core_build_mode) "
            "and compare to bundle results. Failures are WARNs unless --strict."
        ),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat any WARN (including deep-replay failures) as FAIL for exit-code purposes.",
    )
    parser.add_argument(
        "--deep-replay-timeout",
        type=int,
        default=DEEP_REPLAY_TIMEOUT,
        metavar="SECONDS",
        help=f"Per-command timeout for deep-replay (default: {DEEP_REPLAY_TIMEOUT}s).",
    )
    args = parser.parse_args()

    return verify(
        bundle_path=Path(args.bundle),
        digest_path=Path(args.digest),
        replay_commands=args.replay_commands,
        allow_commit_mismatch=args.allow_commit_mismatch,
        json_mode=args.json,
        deep_replay=args.deep_replay,
        strict=args.strict,
        deep_replay_timeout=args.deep_replay_timeout,
    )


if __name__ == "__main__":
    raise SystemExit(main())
