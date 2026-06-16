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
    "src/cpu/src/ct_",
    "src/cpu/include/secp256k1/ct_",
    "include/ct/",
)
FALLBACK_SECURITY_FILES = {
    "src/cpu/include/secp256k1/detail/secure_erase.hpp",
    "src/cpu/include/secp256k1/private_key.hpp",
    "src/cpu/src/bip32.cpp",
    "src/cpu/src/bip39.cpp",
    "src/cpu/src/ecdh.cpp",
    "src/cpu/src/ecdsa.cpp",
    "src/cpu/src/ecies.cpp",
    "src/cpu/src/frost.cpp",
    "src/cpu/src/musig2.cpp",
    "src/cpu/src/address.cpp",
    "src/cpu/src/ufsecp_impl.cpp",
}
SECRET_ABI_FILES = {
    "include/ufsecp/ufsecp.h",
    "include/ufsecp/ufsecp_gpu.h",
    "src/cpu/src/ufsecp_impl.cpp",
}


def _git_diff_names(ref: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "diff", "--name-only", f"{ref}..HEAD"],
        capture_output=True,
        text=True,
        cwd=str(LIB_ROOT),
        check=False,
    )


def _fetch_ref(ref: str) -> subprocess.CompletedProcess[str]:
    if ref.startswith("origin/"):
        branch = ref.split("/", 1)[1]
        fetch_ref = f"{branch}:refs/remotes/origin/{branch}"
    else:
        fetch_ref = ref
    return subprocess.run(
        ["git", "fetch", "--no-tags", "--depth=1", "origin", fetch_ref],
        capture_output=True,
        text=True,
        cwd=str(LIB_ROOT),
        check=False,
    )


def _changed_files_from_ref(ref: str) -> tuple[list[str] | None, str | None]:
    result = _git_diff_names(ref)
    if result.returncode != 0:
        fetch = _fetch_ref(ref)
        if fetch.returncode == 0:
            result = _git_diff_names(ref)
    if result.returncode != 0:
        return None, (
            f"'git diff {ref}..HEAD' failed (rc={result.returncode}): "
            f"{result.stderr.strip()}"
        )
    changed = sorted(
        line.strip()
        for line in result.stdout.splitlines()
        if line.strip()
    )
    return changed, None


def get_changed_files(base: str | None = None, before_sha: str | None = None) -> list[str]:
    changed = set()

    # On GitHub Actions the working tree is clean — git diff HEAD / --cached always
    # returns empty.  Use the base ref or before-sha to compare commits instead.
    if base or before_sha:
        # Prefer before_sha on push events: it points at the commit just before
        # the push, so diff is non-empty even when origin/dev already equals HEAD.
        # Fall back to base (branch ref) for PRs and local runs.
        refs: list[tuple[str, str]] = []
        if before_sha and len(before_sha) == 40 and not all(c == "0" for c in before_sha):
            refs.append(("before-sha", before_sha))
        if base:
            refs.append(("base", base))

        errors = []
        for label, ref in refs:
            files, error = _changed_files_from_ref(ref)
            if files is not None:
                if errors:
                    sys.stderr.write(
                        "::warning::check_secret_path_changes: "
                        f"falling back to {label} ref {ref}; "
                        f"previous ref failed: {errors[-1]}\n"
                    )
                return files
            errors.append(error or f"{label} ref {ref} failed")

        # CAAS-06 fix: previously the returncode was ignored. A failed `git diff`
        # (unresolvable ref, shallow clone, missing before-sha) produced empty
        # stdout, so the gate saw "no files changed" and PASSED — fail-open, even
        # if a secret-bearing file was modified. Fail closed only after every
        # available ref fails; force-push before-sha expiry may safely fall back to
        # the base ref, which is conservative and still non-empty for protected CI.
        sys.stderr.write(
            "::error::check_secret_path_changes: could not determine changed files.\n"
            + "\n".join(f"  - {err}" for err in errors)
            + "\nEnsure the base / before-sha ref is fetched or reachable.\n"
        )
        raise SystemExit(2)

    # Local fallback: staged + unstaged changes relative to HEAD.
    for command in [
        ["git", "diff", "--name-only", "HEAD"],
        ["git", "diff", "--cached", "--name-only"],
    ]:
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


def build_report(
    changed_files: list[str] | None = None,
    base: str | None = None,
    before_sha: str | None = None,
) -> tuple[dict, bool]:
    changed = sorted(set(changed_files or get_changed_files(base=base, before_sha=before_sha)))
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
    base = None
    before_sha = None
    json_mode = "--json" in argv

    if "--files" in argv:
        index = argv.index("--files")
        files = [arg for arg in argv[index + 1:] if not arg.startswith("--")]
    if "--base" in argv:
        index = argv.index("--base")
        base = argv[index + 1]
    if "--before-sha" in argv:
        index = argv.index("--before-sha")
        before_sha = argv[index + 1]

    report, has_fail = build_report(files, base=base, before_sha=before_sha)
    if json_mode:
        print(json.dumps(report, indent=2))
    else:
        _print_text_report(report, has_fail)
    return 1 if has_fail else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
