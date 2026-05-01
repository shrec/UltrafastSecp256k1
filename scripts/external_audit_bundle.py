#!/usr/bin/env python3
"""Build a cryptographically verifiable external-audit bundle.

This script is intended for independent external auditors.
It captures:
  - current commit and dirty-state metadata
  - outputs of critical audit gates
  - SHA-256 hashes for all referenced evidence files
  - a detached SHA-256 digest for the bundle itself

Exit code is fail-closed:
  0 on success
  1 on any missing evidence or failed gate command
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
LIB_ROOT = SCRIPT_DIR.parent

DEFAULT_BUNDLE = LIB_ROOT / "docs" / "EXTERNAL_AUDIT_BUNDLE.json"
DEFAULT_DIGEST = LIB_ROOT / "docs" / "EXTERNAL_AUDIT_BUNDLE.sha256"

GATE_COMMANDS: list[dict[str, object]] = [
    {
        "name": "audit_gate",
        # Use -o file output (not stdout capture) so nested harness subprocesses
        # behave identically to the direct CAAS Stage 2 invocation. Stdout capture
        # of a nested subprocess tree can cause empty stderr/stdout in harnesses.
        "cmd": [sys.executable, "scripts/audit_gate.py", "--json"],
        "required": True,
        "use_file_output": True,
    },
    {
        "name": "audit_gap_report_strict",
        "cmd": [sys.executable, "scripts/audit_gap_report.py", "--json", "--strict"],
        "required": True,
    },
    {
        "name": "security_autonomy_check",
        "cmd": [sys.executable, "scripts/security_autonomy_check.py", "--json"],
        "required": True,
    },
    {
        "name": "supply_chain_gate",
        "cmd": [sys.executable, "scripts/supply_chain_gate.py", "--json"],
        "required": True,
    },
]

EVIDENCE_FILES: list[str] = [
    "docs/AUDIT_MANIFEST.md",
    "docs/SELF_AUDIT_FAILURE_MATRIX.md",
    "docs/AUDIT_SLA.json",
    "docs/FORMAL_INVARIANTS_SPEC.json",
    "docs/FEATURE_ASSURANCE_LEDGER.md",
    "docs/TEST_MATRIX.md",
    "docs/CT_VERIFICATION.md",
    "docs/FFI_HOSTILE_CALLER.md",
    "docs/BACKEND_ASSURANCE_MATRIX.md",
    # NOTE: SECURITY_AUTONOMY_KPI.json is intentionally excluded from EVIDENCE_FILES.
    # Its content (autonomy score) is captured in gate_results via the
    # security_autonomy_check gate command above, which is content-hashed there.
    # Including it here causes a hash race: caas_runner.py re-runs the same gate
    # (updating the file timestamp/nonce) immediately after bundle generation,
    # making every subsequent verify fail on this single file. The gate_results
    # entry already provides the tamper-evidence for this artifact.
    "mutation_kill_report.json",
]


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


def _git(cmd: list[str]) -> str:
    try:
        result = subprocess.run(
            ["git"] + cmd,
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


def _run_gate(name: str, cmd: list[str], required: bool,
              use_file_output: bool = False) -> dict[str, object]:
    try:
        if use_file_output:
            # Write gate JSON output to a temp file (same as direct CAAS Stage 2
            # invocation). This prevents nested subprocess stdout capture from
            # silently swallowing harness output in deep subprocess trees.
            with tempfile.NamedTemporaryFile(
                prefix=f"bundle_gate_{name}_",
                suffix=".json",
                mode="w",
                delete=False,
            ) as tmp:
                out_path = Path(tmp.name)
            actual_cmd = cmd + ["-o", str(out_path)]
            result = subprocess.run(
                actual_cmd,
                cwd=str(LIB_ROOT),
                capture_output=True,
                text=True,
                timeout=900,
                check=False,
            )
            stdout_content = ""
            parsed: dict[str, object] | None = None
            if out_path.exists() and out_path.stat().st_size > 0:
                try:
                    parsed = json.loads(out_path.read_text(encoding="utf-8", errors="replace"))
                    stdout_content = json.dumps(parsed)
                except Exception:
                    parsed = None
            try:
                out_path.unlink(missing_ok=True)
            except Exception:
                pass
        else:
            result = subprocess.run(
                cmd,
                cwd=str(LIB_ROOT),
                capture_output=True,
                text=True,
                timeout=900,
                check=False,
            )
            stdout_content = result.stdout
            parsed = None
            try:
                parsed = json.loads(result.stdout)
            except Exception:
                parsed = None
    except Exception as exc:
        return {
            "name": name,
            "required": required,
            "passing": False,
            "returncode": -1,
            "error": str(exc),
            "stdout_sha256": "",
            "stderr_sha256": "",
        }

    passing = result.returncode == 0
    if isinstance(parsed, dict) and "overall_pass" in parsed:
        passing = passing and bool(parsed.get("overall_pass", False))

    return {
        "name": name,
        "required": required,
        "passing": passing,
        "returncode": result.returncode,
        "stdout_sha256": _sha256_bytes(stdout_content.encode("utf-8", errors="replace")),
        "stderr_sha256": _sha256_bytes(result.stderr.encode("utf-8", errors="replace")),
        "parsed_json": parsed,
    }


def _collect_evidence(paths: list[str]) -> tuple[list[dict[str, object]], list[str]]:
    rows: list[dict[str, object]] = []
    missing: list[str] = []
    for rel in paths:
        full = LIB_ROOT / rel
        if not full.exists():
            missing.append(rel)
            rows.append(
                {
                    "path": rel,
                    "exists": False,
                    "size": 0,
                    "sha256": "",
                }
            )
            continue
        rows.append(
            {
                "path": rel,
                "exists": True,
                "size": full.stat().st_size,
                "sha256": _sha256_file(full),
            }
        )
    return rows, missing


def run(bundle_path: Path, digest_path: Path, json_mode: bool) -> int:
    gate_results: list[dict[str, object]] = []
    for gate in GATE_COMMANDS:
        gate_results.append(
            _run_gate(
                name=str(gate["name"]),
                cmd=list(gate["cmd"]),
                required=bool(gate["required"]),
                use_file_output=bool(gate.get("use_file_output", False)),
            )
        )

    evidence_rows, missing_evidence = _collect_evidence(EVIDENCE_FILES)

    required_gate_failures = [
        r["name"]
        for r in gate_results
        if bool(r.get("required", False)) and not bool(r.get("passing", False))
    ]

    commit_sha = _git(["rev-parse", "HEAD"])
    dirty = bool(_git(["status", "--porcelain"]))
    branch = _git(["rev-parse", "--abbrev-ref", "HEAD"])

    bundle = {
        "schema_version": "1.0.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(LIB_ROOT),
        "git": {
            "commit": commit_sha,
            "branch": branch,
            "dirty": dirty,
        },
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "gate_results": gate_results,
        "evidence": evidence_rows,
        "summary": {
            "required_gate_failures": required_gate_failures,
            "missing_evidence": missing_evidence,
            "overall_pass": len(required_gate_failures) == 0 and len(missing_evidence) == 0,
        },
    }

    rendered = json.dumps(bundle, indent=2, sort_keys=True)
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    bundle_path.write_text(rendered + "\n", encoding="utf-8")

    bundle_sha = _sha256_bytes((rendered + "\n").encode("utf-8"))
    digest_path.write_text(f"{bundle_sha}  {bundle_path.name}\n", encoding="utf-8")

    if json_mode:
        print(rendered)
    else:
        print(f"bundle: {bundle_path}")
        print(f"digest: {digest_path}")
        print(f"required gate failures: {len(required_gate_failures)}")
        print(f"missing evidence: {len(missing_evidence)}")
        print(f"overall_pass: {bundle['summary']['overall_pass']}")

    return 0 if bundle["summary"]["overall_pass"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Print JSON bundle to stdout")
    parser.add_argument(
        "-o",
        "--output",
        default=str(DEFAULT_BUNDLE),
        help="Bundle output path (default: docs/EXTERNAL_AUDIT_BUNDLE.json)",
    )
    parser.add_argument(
        "--digest-output",
        default=str(DEFAULT_DIGEST),
        help="Digest output path (default: docs/EXTERNAL_AUDIT_BUNDLE.sha256)",
    )
    args = parser.parse_args()

    return run(
        bundle_path=Path(args.output),
        digest_path=Path(args.digest_output),
        json_mode=args.json,
    )


if __name__ == "__main__":
    raise SystemExit(main())
