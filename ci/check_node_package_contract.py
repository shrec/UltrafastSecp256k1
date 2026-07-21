#!/usr/bin/env python3
"""Validate the source Node.js package as an FFI-only npm package.

The package loads the project shared library through ffi-napi. It must not
advertise nonexistent native-addon inputs or run node-gyp for this package,
and every declared runtime entry must be included by the npm files allowlist.

Exit 0 = valid contract, 1 = malformed or contradictory package manifest.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

LIB_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = LIB_ROOT / "bindings" / "nodejs"
MANIFEST = PACKAGE_ROOT / "package.json"
LOCKFILE = PACKAGE_ROOT / "package-lock.json"
NATIVE_BUILD_DEPENDENCIES = {"node-gyp", "node-addon-api"}


def _normalized(path: str) -> str:
    return path.replace("\\", "/").rstrip("/")


def _is_packed(path: str, allowlist: list[str]) -> bool:
    target = _normalized(path)
    for entry in allowlist:
        allowed = _normalized(entry)
        if target == allowed or target.startswith(allowed + "/"):
            return True
    return False


def evaluate_manifest(manifest: dict, existing_paths: set[str]) -> list[dict]:
    """Return deterministic contract violations for a manifest/filesystem view."""
    problems = []
    allowlist = manifest.get("files")
    if not isinstance(allowlist, list) or not all(isinstance(v, str) for v in allowlist):
        return [{"kind": "invalid_files_allowlist", "detail": "files must be a string array"}]

    normalized_existing = {_normalized(p) for p in existing_paths}
    for entry in allowlist:
        if _normalized(entry) not in normalized_existing:
            problems.append({"kind": "listed_path_missing", "path": entry})

    scripts = manifest.get("scripts", {})
    if isinstance(scripts, dict):
        for name in ("install", "build", "preinstall", "postinstall"):
            command = scripts.get(name)
            if isinstance(command, str) and "node-gyp" in command:
                problems.append({"kind": "node_gyp_script", "script": name})

    dependencies = manifest.get("dependencies", {})
    if isinstance(dependencies, dict):
        for name in sorted(NATIVE_BUILD_DEPENDENCIES & set(dependencies)):
            problems.append({"kind": "native_build_dependency", "dependency": name})

    for field in ("main", "types"):
        entry = manifest.get(field)
        if not isinstance(entry, str):
            problems.append({"kind": "runtime_entry_missing", "field": field})
        elif _normalized(entry) not in normalized_existing:
            problems.append({"kind": "runtime_entry_missing_on_disk", "field": field, "path": entry})
        elif not _is_packed(entry, allowlist):
            problems.append({"kind": "runtime_entry_not_packed", "field": field, "path": entry})

    return problems


def evaluate_lockfile(manifest: dict, lockfile: dict) -> list[dict]:
    """Require the lockfile root dependency sets to match package.json."""
    root = lockfile.get("packages", {}).get("")
    if not isinstance(root, dict):
        return [{"kind": "lockfile_root_missing"}]
    problems = []
    for field in ("dependencies", "devDependencies"):
        expected = manifest.get(field, {})
        actual = root.get(field, {})
        if actual != expected:
            problems.append({
                "kind": "lockfile_dependency_drift",
                "field": field,
                "expected": expected,
                "actual": actual,
            })
    return problems


def _existing_paths() -> set[str]:
    paths = set()
    for path in PACKAGE_ROOT.rglob("*"):
        paths.add(path.relative_to(PACKAGE_ROOT).as_posix())
    return paths


def evaluate() -> dict:
    try:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"overall_pass": False, "problems": [{"kind": "manifest_unreadable", "detail": str(exc)}]}
    problems = evaluate_manifest(manifest, _existing_paths())
    try:
        lockfile = json.loads(LOCKFILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        problems.append({"kind": "lockfile_unreadable", "detail": str(exc)})
    else:
        problems.extend(evaluate_lockfile(manifest, lockfile))
    return {"overall_pass": not problems, "problems": problems}


def main() -> int:
    report = evaluate()
    if report["overall_pass"]:
        print("check_node_package_contract: FFI manifest and npm allowlist [PASS]")
        return 0
    for problem in report["problems"]:
        print(f"::error::check_node_package_contract: {problem}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
