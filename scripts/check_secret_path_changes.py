#!/usr/bin/env python3
"""Enforce stricter doc pairing for secret-bearing code changes."""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
LIB_ROOT = SCRIPT_DIR.parent
DB_PATH = LIB_ROOT / "tools" / "source_graph_kit" / "source_graph.db"

CT_DOCS = [
    "docs/CT_VERIFICATION.md",
    "docs/SECURITY_CLAIMS.md",
]
LIFECYCLE_DOCS = [
    "docs/SECRET_LIFECYCLE.md",
    "docs/SECURITY_CLAIMS.md",
]
ABI_DOCS = [
    "docs/SECURITY_CLAIMS.md",
    "docs/FFI_HOSTILE_CALLER.md",
]

FALLBACK_CT_PREFIXES = (
    "cpu/src/ct_",
    "cpu/include/secp256k1/ct_",
    "include/ct/",
)
FALLBACK_SECURITY_FILES = {
    "cpu/include/secp256k1/detail/secure_erase.hpp",
    "cpu/include/secp256k1/private_key.hpp",
    "cpu/src/bip32.cpp",
    "cpu/src/bip39.cpp",
    "cpu/src/ecdh.cpp",
    "cpu/src/ecdsa.cpp",
    "cpu/src/ecies.cpp",
    "cpu/src/frost.cpp",
    "cpu/src/musig2.cpp",
    "cpu/src/silent_payments.cpp",
    "include/ufsecp/ufsecp_impl.cpp",
}
SECRET_ABI_FILES = {
    "include/ufsecp/ufsecp.h",
    "include/ufsecp/ufsecp_gpu.h",
    "include/ufsecp/ufsecp_impl.cpp",
}


def get_changed_files() -> list[str]:
    changed = set()
    commands = [
        ["git", "diff", "--name-only", "HEAD"],
        ["git", "diff", "--cached", "--name-only"],
    ]
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


def _load_secret_surface_sets() -> tuple[set[str], set[str]]:
    ct_files = set()
    security_files = set(FALLBACK_SECURITY_FILES)

    if not DB_PATH.exists():
        return ct_files, security_files

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        # source_graph.db schema: symbol_metadata has ct_sensitive flag,
        # semantic_tags has tag='security' per file.
        ct_files = {
            row["file_path"]
            for row in conn.execute(
                "SELECT DISTINCT file_path FROM symbol_metadata WHERE ct_sensitive = 1"
            ).fetchall()
        }
        security_files |= {
            row["file"]
            for row in conn.execute(
                "SELECT DISTINCT file FROM semantic_tags WHERE tag = 'security'"
            ).fetchall()
            if row["file"]
        }
    except Exception:
        # Graceful fallback if schema differs
        pass
    finally:
        conn.close()

    return ct_files, security_files


def _matches_prefixes(path: str, prefixes: tuple[str, ...]) -> bool:
    return any(path.startswith(prefix) for prefix in prefixes)


def build_report(changed_files: list[str] | None = None) -> tuple[dict, bool]:
    changed = sorted(set(changed_files or get_changed_files()))
    changed_set = set(changed)
    ct_files, security_files = _load_secret_surface_sets()

    rules = [
        {
            "name": "CT secret-bearing surfaces",
            "required_docs": CT_DOCS,
            "reason": "Constant-time secret-bearing changes must refresh CT guarantees and owner-grade claims.",
            "matches": sorted(
                path
                for path in changed
                if path in ct_files or _matches_prefixes(path, FALLBACK_CT_PREFIXES)
            ),
        },
        {
            "name": "Secret lifecycle and zeroization surfaces",
            "required_docs": LIFECYCLE_DOCS,
            "reason": "Secret lifecycle changes must refresh zeroization and cleanup evidence.",
            "matches": sorted(path for path in changed if path in security_files),
        },
        {
            "name": "Secret-bearing ABI boundary",
            "required_docs": ABI_DOCS,
            "reason": "Secret-bearing ABI changes must refresh caller-contract and claim documentation.",
            "matches": sorted(path for path in changed if path in SECRET_ABI_FILES),
        },
    ]

    triggered_rules = []
    blocking_findings = []
    for rule in rules:
        if not rule["matches"]:
            continue
        missing_docs = [doc for doc in rule["required_docs"] if doc not in changed_set]
        entry = {
            "name": rule["name"],
            "matches": rule["matches"],
            "required_docs": list(rule["required_docs"]),
            "missing_docs": missing_docs,
            "reason": rule["reason"],
        }
        triggered_rules.append(entry)
        if missing_docs:
            preview = ", ".join(rule["matches"][:3])
            blocking_findings.append(
                {
                    "rule": rule["name"],
                    "message": (
                        f"{rule['name']}: {preview} changed without updates to "
                        f"{', '.join(missing_docs)}"
                    ),
                }
            )

    report = {
        "changed_files": changed,
        "triggered_rules": triggered_rules,
        "blocking_findings": blocking_findings,
    }
    return report, bool(blocking_findings)


def _print_text_report(report: dict, has_fail: bool) -> None:
    if not report["changed_files"]:
        print("PASS No changed files")
        return
    if not report["triggered_rules"]:
        print("PASS No changed secret-bearing paths")
        return

    for rule in report["triggered_rules"]:
        status = "FAIL" if rule["missing_docs"] else "PASS"
        print(f"{status} {rule['name']}")
        print(f"  Matches: {', '.join(rule['matches'])}")
        print(f"  Required docs: {', '.join(rule['required_docs'])}")
        if rule["missing_docs"]:
            print(f"  Missing docs: {', '.join(rule['missing_docs'])}")
        print(f"  Reason: {rule['reason']}")

    if has_fail:
        print(f"FAIL {len(report['blocking_findings'])} blocking secret-path change finding(s)")
    else:
        print("PASS Secret-bearing changes have paired documentation updates")


def main(argv: list[str]) -> int:
    files = None
    json_mode = "--json" in argv
    if "--files" in argv:
        index = argv.index("--files")
        files = [arg for arg in argv[index + 1:] if not arg.startswith("--")]

    report, has_fail = build_report(files)
    if json_mode:
        print(json.dumps(report, indent=2))
    else:
        _print_text_report(report, has_fail)
    return 1 if has_fail else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))