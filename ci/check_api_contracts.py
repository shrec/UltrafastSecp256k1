#!/usr/bin/env python3
"""Validate machine-readable API security contracts and enforce fail-closed updates."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
LIB_ROOT = SCRIPT_DIR.parent
CONTRACT_FILE = LIB_ROOT / "docs" / "API_SECURITY_CONTRACTS.json"
TEST_MATRIX_FILE = LIB_ROOT / "docs" / "TEST_MATRIX.md"

REQUIRED_TOP_LEVEL = (
    "version",
    "entries",
)

REQUIRED_ENTRY_FIELDS = (
    "api",
    "criticality",
    "ct_class",
    "secrets_touched",
    "preconditions",
    "postconditions",
    "failure_modes",
    "evidence_tests",
    "evidence_docs",
)

ALLOWED_CRITICALITY = {"critical", "high", "medium", "low"}
ALLOWED_CT_CLASS = {"ct-required", "ct-adjacent", "public-only"}

SENSITIVE_PREFIXES = (
    "include/ufsecp/",
    "cpu/src/ecdsa.cpp",
    "cpu/src/schnorr.cpp",
    "cpu/src/musig2.cpp",
    "cpu/src/frost.cpp",
    "cpu/src/ecdh.cpp",
    "cpu/src/bip32.cpp",
    "cpu/src/batch_verify.cpp",
)


def _get_changed_files() -> list[str]:
    changed = set()
    commands = (
        ["git", "diff", "--name-only", "HEAD"],
        ["git", "diff", "--cached", "--name-only"],
        ["git", "ls-files", "--others", "--exclude-standard"],
    )
    for command in commands:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            cwd=str(LIB_ROOT),
            check=False,
        )
        for line in result.stdout.splitlines():
            line = line.strip()
            if line:
                changed.add(line)
    return sorted(changed)


def _looks_sensitive(path: str) -> bool:
    return any(path.startswith(prefix) for prefix in SENSITIVE_PREFIXES)


def _ensure_list_of_strings(value: object) -> bool:
    return isinstance(value, list) and all(isinstance(v, str) and v.strip() for v in value)


def validate_contracts() -> tuple[list[str], dict]:
    issues: list[str] = []
    payload: dict = {}

    if not CONTRACT_FILE.exists():
        return [f"MISSING-CONTRACT-FILE {CONTRACT_FILE.relative_to(LIB_ROOT)}"], payload

    try:
        payload = json.loads(CONTRACT_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"INVALID-JSON {CONTRACT_FILE.name}: {exc}"], payload

    if not isinstance(payload, dict):
        return ["INVALID-TOPLEVEL expected JSON object"], payload

    for key in REQUIRED_TOP_LEVEL:
        if key not in payload:
            issues.append(f"MISSING-TOPLEVEL-FIELD {key}")

    entries = payload.get("entries")
    if not isinstance(entries, list):
        issues.append("INVALID-ENTRIES expected list")
        return issues, payload

    if not entries:
        issues.append("EMPTY-ENTRIES at least one API contract entry is required")
        return issues, payload

    seen_apis = set()
    test_matrix = TEST_MATRIX_FILE.read_text(encoding="utf-8", errors="replace") if TEST_MATRIX_FILE.exists() else ""

    for idx, entry in enumerate(entries):
        where = f"entries[{idx}]"
        if not isinstance(entry, dict):
            issues.append(f"{where} INVALID-ENTRY expected object")
            continue

        for field in REQUIRED_ENTRY_FIELDS:
            if field not in entry:
                issues.append(f"{where} MISSING-FIELD {field}")

        api = entry.get("api")
        if not isinstance(api, str) or not api.startswith("ufsecp_"):
            issues.append(f"{where} INVALID-API expected ufsecp_* symbol")
        else:
            if api in seen_apis:
                issues.append(f"{where} DUPLICATE-API {api}")
            seen_apis.add(api)

        criticality = entry.get("criticality")
        if criticality not in ALLOWED_CRITICALITY:
            issues.append(f"{where} INVALID-CRITICALITY {criticality}")

        ct_class = entry.get("ct_class")
        if ct_class not in ALLOWED_CT_CLASS:
            issues.append(f"{where} INVALID-CT-CLASS {ct_class}")

        if not _ensure_list_of_strings(entry.get("secrets_touched")):
            issues.append(f"{where} INVALID-secrets_touched expected non-empty string list")

        if not _ensure_list_of_strings(entry.get("preconditions")):
            issues.append(f"{where} INVALID-preconditions expected non-empty string list")

        if not _ensure_list_of_strings(entry.get("postconditions")):
            issues.append(f"{where} INVALID-postconditions expected non-empty string list")

        if not _ensure_list_of_strings(entry.get("failure_modes")):
            issues.append(f"{where} INVALID-failure_modes expected non-empty string list")

        tests = entry.get("evidence_tests")
        if not _ensure_list_of_strings(tests):
            issues.append(f"{where} INVALID-evidence_tests expected non-empty string list")
        else:
            for test_name in tests:
                if test_matrix and test_name not in test_matrix:
                    issues.append(f"{where} UNKNOWN-TEST {test_name} not found in docs/TEST_MATRIX.md")

        docs = entry.get("evidence_docs")
        if not _ensure_list_of_strings(docs):
            issues.append(f"{where} INVALID-evidence_docs expected non-empty string list")
        else:
            for doc_path in docs:
                rel = Path(doc_path)
                if rel.is_absolute() or ".." in rel.parts:
                    issues.append(f"{where} INVALID-DOC-PATH {doc_path}")
                    continue
                if not (LIB_ROOT / rel).exists():
                    issues.append(f"{where} MISSING-DOC {doc_path}")

    return issues, payload


def check_changed_file_policy(changed_files: list[str]) -> list[str]:
    issues: list[str] = []
    sensitive_changed = [f for f in changed_files if _looks_sensitive(f)]
    if sensitive_changed and str(CONTRACT_FILE.relative_to(LIB_ROOT)) not in set(changed_files):
        issues.append(
            "CONTRACT-UPDATE-REQUIRED sensitive API/security files changed without docs/API_SECURITY_CONTRACTS.json update"
        )
    return issues


def main(argv: list[str]) -> int:
    json_mode = "--json" in argv
    changed = _get_changed_files()

    issues, payload = validate_contracts()
    issues.extend(check_changed_file_policy(changed))

    if json_mode:
        print(
            json.dumps(
                {
                    "contract_file": str(CONTRACT_FILE.relative_to(LIB_ROOT)),
                    "changed_files": changed,
                    "issues": issues,
                    "entries": len(payload.get("entries", [])) if isinstance(payload, dict) else 0,
                },
                indent=2,
            )
        )
    else:
        if issues:
            for issue in issues:
                print(f"FAIL {issue}")
            print(f"FAIL {len(issues)} API contract issue(s)")
        else:
            print("PASS API contracts are valid and up-to-date")

    return 1 if issues else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
