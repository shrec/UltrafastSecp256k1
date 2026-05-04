#!/usr/bin/env python3
"""Evidence governance — tamper-resistant evidence chain for audit verdicts.

Every critical verdict records:
  who   — script/tool that produced it
  what  — exact check performed
  when  — ISO 8601 timestamp
  commit — git SHA at evidence time
  binary_hash — SHA-256 of tested artifact (if applicable)
  verdict — pass/fail/skip with reason
  signature — HMAC of above fields (tamper detection)

The evidence chain must be complete and internally consistent.
Any gap or tamper = fail.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
LIB_ROOT = SCRIPT_DIR.parent
EVIDENCE_CHAIN_FILE = LIB_ROOT / "docs" / "EVIDENCE_CHAIN.json"

# HMAC key for evidence chain integrity.
# In CI (GITHUB_ACTIONS=true) the CAAS_HMAC_KEY secret MUST be set — using the
# public in-repo fallback key in a CI environment means any repo reader can forge
# evidence records, which makes the chain forensically worthless.
# Locally the fallback is accepted; CI hard-fails if the secret is absent.
#
# NOTE: GitHub Actions sets secrets to an empty string (not absent) when the
# secret is not configured. Both "not in os.environ" and "== ''" are treated as
# "no key configured" to prevent the bypass.
_HMAC_KEY_DEFAULT = "ufsecp-evidence-chain-v1"
_caas_hmac_env = os.environ.get("CAAS_HMAC_KEY", "").strip()
_HMAC_KEY_IS_DEFAULT = not _caas_hmac_env
_HMAC_KEY = _caas_hmac_env.encode() if _caas_hmac_env else _HMAC_KEY_DEFAULT.encode()
if _HMAC_KEY_IS_DEFAULT and os.environ.get("GITHUB_ACTIONS") != "true":
    print(
        "WARNING: Using public in-repo HMAC key. Tamper detection only — not "
        "cryptographic auth. Set CAAS_HMAC_KEY env var for production.",
        file=sys.stderr,
    )


def _enforce_ci_hmac_key() -> None:
    """Fail-fast when CI runs without a real HMAC key.

    F-06 fix: moved from module-level sys.exit(1) to an explicit function so
    that importing evidence_governance as a library doesn't kill the caller.
    Called from main() only — subprocess invocations still fail correctly.
    """
    if _HMAC_KEY_IS_DEFAULT and os.environ.get("GITHUB_ACTIONS") == "true":
        print(
            "::error::CAAS_HMAC_KEY secret is not set or is empty. "
            "Evidence chain HMAC uses the public in-repo key in CI, which allows "
            "anyone with read access to forge records. "
            "Add CAAS_HMAC_KEY as a repository secret and pass it via env: in the workflow.",
            file=sys.stderr,
        )
        sys.exit(1)


def _git_sha() -> str:
    """Get current git HEAD SHA."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
            cwd=str(LIB_ROOT),
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _file_sha256(path: Path) -> str:
    """Compute SHA-256 of a file."""
    if not path.exists():
        return "missing"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _compute_hmac(record: dict) -> str:
    """Compute HMAC for tamper detection (covers all evidence fields including reason).

    run_id and signed_by_ci are intentionally excluded from the HMAC payload to
    preserve backward compatibility with existing chain records written before
    those fields were added. They are metadata-only fields.
    """
    payload = json.dumps({
        "who": record.get("who", ""),
        "what": record.get("what", ""),
        "when": record.get("when", ""),
        "commit": record.get("commit", ""),
        "binary_hash": record.get("binary_hash", ""),
        "verdict": record.get("verdict", ""),
        "reason": record.get("reason", ""),
    }, sort_keys=True)
    return hmac.new(_HMAC_KEY, payload.encode(), hashlib.sha256).hexdigest()


def create_evidence_record(
    who: str,
    what: str,
    verdict: str,
    binary_path: Path | None = None,
    reason: str = "",
    commit: str = "",
) -> dict:
    """Create a new evidence record with full provenance."""
    # Prefer caller-supplied commit (the audited commit) over HEAD so that
    # nightly refresh records are traceable to the commit that was audited,
    # not the HEAD at refresh time (FINDING-13 fix).
    resolved_commit = commit.strip() if commit.strip() else _git_sha()
    # L-2 fix: fail fast if the commit SHA cannot be determined. Writing an
    # "unknown" commit record permanently marks the evidence chain as having
    # an orphaned entry, which causes all subsequent chain-validation steps to
    # fail (hard error in caas.yml). Better to refuse the write than to corrupt
    # the chain.
    if not resolved_commit or resolved_commit in ("unknown", "n/a", ""):
        raise ValueError(
            "create_evidence_record: cannot determine commit SHA — "
            "git rev-parse HEAD failed or returned empty. "
            "Set the --commit argument explicitly or fix the git environment."
        )
    record = {
        "who": who,
        "what": what,
        "when": datetime.now(timezone.utc).isoformat(),
        "commit": resolved_commit,
        # run_id is metadata-only — intentionally excluded from the HMAC payload
        # (see _compute_hmac) for backward compatibility with older chain records.
        # It does not contribute to tamper-detection; the HMAC covers who/what/when/
        # commit/binary_hash/verdict/reason instead.
        "run_id": os.environ.get("GITHUB_RUN_ID", "local"),
        "binary_hash": _file_sha256(binary_path) if binary_path else "n/a",
        "verdict": verdict,
        "reason": reason,
        # PERSIST-1 fix: distinguish CI-signed from locally-signed records so
        # auditors can identify which evidence was produced in the trusted CI
        # environment vs on a developer's local machine.
        "signed_by_ci": bool(os.environ.get("GITHUB_ACTIONS") == "true"),
    }
    record["signature"] = _compute_hmac(record)
    return record


def load_chain() -> list[dict]:
    """Load existing evidence chain."""
    if not EVIDENCE_CHAIN_FILE.exists():
        return []
    try:
        data = json.loads(EVIDENCE_CHAIN_FILE.read_text(encoding="utf-8"))
        return data.get("records", [])
    except (json.JSONDecodeError, KeyError):
        return []


def save_chain(records: list[dict]) -> None:
    """Save evidence chain."""
    data = {
        "version": 1,
        "description": "Tamper-resistant evidence chain for audit verdicts",
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "records_count": len(records),
        "records": records,
    }
    EVIDENCE_CHAIN_FILE.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def append_record(record: dict) -> None:
    """Append a record to the evidence chain."""
    chain = load_chain()
    chain.append(record)
    save_chain(chain)


def validate_chain() -> dict:
    """Validate the entire evidence chain for completeness and integrity."""
    chain = load_chain()
    issues: list[str] = []
    tampered: list[int] = []
    orphaned: list[int] = []

    for i, record in enumerate(chain):
        # Check required fields (run_id is metadata-only — not required for backward compat)
        required = ["who", "what", "when", "commit", "verdict", "signature"]
        for field in required:
            if field not in record or not record[field]:
                issues.append(f"record[{i}]: missing required field '{field}'")

        # Verify HMAC
        if "signature" in record:
            expected = _compute_hmac(record)
            if record["signature"] != expected:
                tampered.append(i)
                issues.append(f"record[{i}]: HMAC mismatch — possible tamper")

        # Check commit is valid hex; records with no traceable commit are orphaned.
        # Orphaned records (commit == "unknown") are counted as issues — a forensic
        # chain record without a traceable git commit is useless for retrospective audits.
        commit = record.get("commit", "")
        if not commit or commit in ("unknown", "n/a", ""):
            orphaned.append(i)
            issues.append(f"record[{i}]: orphaned record — commit is '{commit}' (not traceable)")
        elif len(commit) < 7 or not all(c in "0123456789abcdef" for c in commit):
            issues.append(f"record[{i}]: invalid commit SHA format")

    return {
        "total_records": len(chain),
        "valid_records": len(chain) - len(tampered),
        "tampered_records": tampered,
        "orphaned_records": orphaned,
        "issues": issues,
        "chain_valid": len(issues) == 0,
    }


def run(mode: str, json_mode: bool, out_file: str | None,
        who: str = "", what: str = "", verdict: str = "",
        binary: str = "", reason: str = "", commit: str = "") -> int:
    if mode == "validate":
        if _HMAC_KEY_IS_DEFAULT and not json_mode:
            print("WARNING: Using hardcoded HMAC key. Set CAAS_HMAC_KEY env var for production.",
                  file=sys.stderr)
        result = validate_chain()
        overall_pass = result["chain_valid"]

        report = {
            "overall_pass": overall_pass,
            "evidence_records_total": result["total_records"],
            "evidence_chain_valid": result["chain_valid"],
            "orphaned_verdicts": len(result["orphaned_records"]),
            "tampered_count": len(result["tampered_records"]),
            "issues": result["issues"],
            "hmac_key_is_default": _HMAC_KEY_IS_DEFAULT,
        }

        rendered = json.dumps(report, indent=2)
        if out_file:
            Path(out_file).write_text(rendered, encoding="utf-8")

        if json_mode:
            print(rendered)
        else:
            if result["issues"]:
                for issue in result["issues"]:
                    print(f"  ISSUE: {issue}")
            print()
            if overall_pass:
                print(f"PASS evidence chain ({result['total_records']} records, all valid)")
            else:
                print(f"FAIL evidence chain ({len(result['issues'])} issue(s))")

        return 0 if overall_pass else 1

    elif mode == "record":
        if not who or not what or not verdict:
            print("ERROR: --who, --what, and --verdict are required for record mode")
            return 1

        binary_path = Path(binary) if binary else None
        record = create_evidence_record(who, what, verdict, binary_path, reason, commit)
        append_record(record)

        if json_mode:
            print(json.dumps(record, indent=2))
        else:
            print(f"Recorded: {who} / {what} / {verdict}")
        return 0

    elif mode == "show":
        chain = load_chain()
        if json_mode:
            print(json.dumps({"records": chain}, indent=2))
        else:
            for i, r in enumerate(chain):
                print(f"  [{i}] {r.get('when', '?')} | {r.get('who', '?')} | "
                      f"{r.get('what', '?')} | {r.get('verdict', '?')}")
        return 0

    else:
        print(f"Unknown mode: {mode}")
        return 1


def main() -> int:
    _enforce_ci_hmac_key()
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["validate", "record", "show"],
                        help="validate: check chain integrity; record: add entry; show: list entries")
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("-o", dest="out_file", help="Write report to file")
    parser.add_argument("--who", default="", help="Who produced the evidence (for record mode)")
    parser.add_argument("--what", default="", help="What was checked (for record mode)")
    parser.add_argument("--verdict", default="", help="pass/fail/skip (for record mode)")
    parser.add_argument("--binary", default="", help="Path to tested binary (for record mode)")
    parser.add_argument("--reason", default="", help="Reason for verdict (for record mode)")
    parser.add_argument("--commit", default="",
                        help="Audited commit SHA (overrides git HEAD; use ${{ github.sha }} in CI)")
    args = parser.parse_args()

    return run(args.mode, args.json, args.out_file,
               args.who, args.what, args.verdict, args.binary, args.reason, args.commit)


if __name__ == "__main__":
    raise SystemExit(main())
