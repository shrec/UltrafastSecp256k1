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

# HMAC key for evidence chain integrity. This key is public (in-repo) and provides
# content-hash tamper detection only — NOT cryptographic authentication.
# An adversary with repo read access can compute valid HMACs for forged records.
# For production forensic evidence, use a secret key via CAAS_HMAC_KEY env var.
_HMAC_KEY_DEFAULT = "ufsecp-evidence-chain-v1"
_HMAC_KEY = os.environ.get("CAAS_HMAC_KEY", _HMAC_KEY_DEFAULT).encode()
_HMAC_KEY_IS_DEFAULT = "CAAS_HMAC_KEY" not in os.environ
if _HMAC_KEY_IS_DEFAULT:
    import sys as _sys
    print(
        "WARNING: Using public in-repo HMAC key. Tamper detection only — not cryptographic auth. "
        "Set CAAS_HMAC_KEY env var for production.",
        file=_sys.stderr,
    )


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
    """Compute HMAC for tamper detection (covers all evidence fields including reason)."""
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
) -> dict:
    """Create a new evidence record with full provenance."""
    record = {
        "who": who,
        "what": what,
        "when": datetime.now(timezone.utc).isoformat(),
        "commit": _git_sha(),
        "binary_hash": _file_sha256(binary_path) if binary_path else "n/a",
        "verdict": verdict,
        "reason": reason,
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
        # Check required fields
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

        # Check commit is valid hex; records with no traceable commit are orphaned
        commit = record.get("commit", "")
        if not commit or commit in ("unknown", "n/a", ""):
            orphaned.append(i)
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
        binary: str = "", reason: str = "") -> int:
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
        record = create_evidence_record(who, what, verdict, binary_path, reason)
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
    args = parser.parse_args()

    return run(args.mode, args.json, args.out_file,
               args.who, args.what, args.verdict, args.binary, args.reason)


if __name__ == "__main__":
    raise SystemExit(main())
