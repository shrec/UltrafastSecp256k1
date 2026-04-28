#!/usr/bin/env python3
"""create_replay_capsule.py  --  Generate a CAAS replay capsule for UltrafastSecp256k1

Creates docs/REPLAY_CAPSULE.json (and its SHA-256 digest at
docs/REPLAY_CAPSULE.sha256) so that a third party can reproduce the exact
CAAS pipeline state represented by this commit.

The capsule records:
  - commit hash + dirty-tree flag
  - tool versions (Python, CMake, Git)
  - build flags
  - SHA-256 hashes of the source graph DB and external audit bundle
  - the sequence of commands needed to replay the full pipeline

Usage
-----
    python3 scripts/create_replay_capsule.py
    python3 scripts/create_replay_capsule.py --profile release --json
    python3 scripts/create_replay_capsule.py --output /tmp/capsule.json

Exit codes
----------
  0  capsule written successfully
  1  a required step failed (details on stderr)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
LIB_ROOT   = SCRIPT_DIR.parent

SCHEMA_VERSION = "1.0"
CAAS_VERSION   = "1.0.0"
CXX_STANDARD   = "20"
CMAKE_BUILD_TYPE = "Release"

# Residuals that are documented as known / accepted
KNOWN_RESIDUALS = ["RR-001", "RR-002", "RR-003", "RR-006", "RR-007", "RR-008"]


# ---------------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------------

def _run(cmd: list[str], cwd: Path | None = None, timeout: int = 30) -> str:
    """Run a command and return stripped stdout, or '' on any failure."""
    try:
        result = subprocess.run(
            cmd,
            cwd=str(cwd or LIB_ROOT),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return result.stdout.strip()
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Field collectors
# ---------------------------------------------------------------------------

def _collect_commit_hash() -> str:
    out = _run(["git", "rev-parse", "HEAD"])
    return out if out else "unknown"


def _collect_dirty() -> bool:
    out = _run(["git", "status", "--porcelain"])
    return bool(out)


def _collect_compiler() -> str:
    """Return compiler id from CC/CXX env, or detect from PATH."""
    cc  = os.environ.get("CC", "")
    cxx = os.environ.get("CXX", "")

    if cxx:
        ver = _run([cxx, "--version"])
        if ver:
            return ver.splitlines()[0]
        return cxx

    if cc:
        ver = _run([cc, "--version"])
        if ver:
            return ver.splitlines()[0]
        return cc

    # Auto-detect: try g++, clang++ in that order
    for candidate in ("g++", "clang++", "c++"):
        ver = _run([candidate, "--version"])
        if ver:
            return ver.splitlines()[0]

    return "unknown"


def _collect_python_version() -> str:
    return sys.version.replace("\n", " ").strip()


def _collect_cmake_version() -> str:
    out = _run(["cmake", "--version"])
    if out:
        return out.splitlines()[0]
    return "cmake not found"


def _collect_git_version() -> str:
    out = _run(["git", "--version"])
    return out if out else "git not found"


# ---------------------------------------------------------------------------
# Hash helpers
# ---------------------------------------------------------------------------

def _sha256_file(path: Path) -> str | None:
    """Return hex SHA-256 of a file, or None if it does not exist."""
    if not path.exists():
        return None
    h = hashlib.sha256()
    try:
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 16), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# Capsule builder
# ---------------------------------------------------------------------------

def build_capsule(profile: str) -> dict:
    """Collect all fields and return the capsule dict."""
    db_path     = LIB_ROOT / ".project_graph.db"
    bundle_path = LIB_ROOT / "docs" / "EXTERNAL_AUDIT_BUNDLE.json"

    capsule = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "commit_hash": _collect_commit_hash(),
        "dirty": _collect_dirty(),
        "profile": profile,
        "build_flags": {
            "cmake_build_type": CMAKE_BUILD_TYPE,
            "compiler": _collect_compiler(),
            "cxx_standard": CXX_STANDARD,
        },
        "tool_versions": {
            "python": _collect_python_version(),
            "cmake":  _collect_cmake_version(),
            "git":    _collect_git_version(),
        },
        "graph_hash": _sha256_file(db_path),
        "evidence_bundle_hash": _sha256_file(bundle_path),
        "replay_commands": [
            "cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release",
            "ninja -C build",
            f"python3 scripts/caas_runner.py --profile {profile} --json -o caas_report.json",
            "python3 scripts/verify_external_audit_bundle.py --json",
        ],
        "known_residuals": KNOWN_RESIDUALS,
        "caas_version": CAAS_VERSION,
    }
    return capsule


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _write_capsule(capsule: dict, out_path: Path) -> str:
    """Serialise the capsule to JSON, write it, and return the raw bytes."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(capsule, indent=2, ensure_ascii=False)
    out_path.write_text(raw + "\n", encoding="utf-8")
    return raw


def _write_digest(raw: str, digest_path: Path, capsule_path: Path) -> str:
    """Write '<hex>  <filename>' to digest_path and return the hex digest."""
    digest = _sha256_bytes(raw.encode("utf-8") + b"\n")
    digest_line = f"{digest}  {capsule_path.name}\n"
    digest_path.parent.mkdir(parents=True, exist_ok=True)
    digest_path.write_text(digest_line, encoding="utf-8")
    return digest


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create a CAAS replay capsule for UltrafastSecp256k1."
    )
    parser.add_argument(
        "--profile",
        metavar="NAME",
        default="default",
        help="CAAS profile name to embed in the capsule (default: default)",
    )
    parser.add_argument(
        "--output",
        metavar="PATH",
        default=str(LIB_ROOT / "docs" / "REPLAY_CAPSULE.json"),
        help="Output path for the capsule JSON (default: docs/REPLAY_CAPSULE.json)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print a summary JSON object to stdout after writing",
    )
    args = parser.parse_args(argv)

    out_path    = Path(args.output).resolve()
    digest_path = out_path.with_suffix(".sha256")

    # --- Build capsule ---
    try:
        capsule = build_capsule(args.profile)
    except Exception as exc:
        print(f"ERROR: failed to collect capsule fields: {exc}", file=sys.stderr)
        return 1

    # --- Write capsule JSON ---
    try:
        raw = _write_capsule(capsule, out_path)
    except Exception as exc:
        print(f"ERROR: cannot write capsule to {out_path}: {exc}", file=sys.stderr)
        return 1

    # --- Write digest ---
    try:
        digest = _write_digest(raw, digest_path, out_path)
    except Exception as exc:
        print(f"ERROR: cannot write digest to {digest_path}: {exc}", file=sys.stderr)
        return 1

    # --- Summary output ---
    summary = {
        "status": "ok",
        "capsule": str(out_path),
        "digest":  str(digest_path),
        "sha256":  digest,
        "commit_hash": capsule["commit_hash"],
        "dirty": capsule["dirty"],
        "profile": capsule["profile"],
        "graph_hash": capsule["graph_hash"],
        "evidence_bundle_hash": capsule["evidence_bundle_hash"],
        "generated_at": capsule["generated_at"],
    }

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        w = 64
        print("=" * w)
        print("Replay Capsule Created")
        print("=" * w)
        print(f"  Capsule : {out_path}")
        print(f"  Digest  : {digest_path}")
        print(f"  SHA-256 : {digest}")
        print(f"  Commit  : {capsule['commit_hash']}")
        print(f"  Dirty   : {capsule['dirty']}")
        print(f"  Profile : {capsule['profile']}")
        gh = capsule["graph_hash"] or "(not found)"
        eb = capsule["evidence_bundle_hash"] or "(not found)"
        print(f"  Graph   : {gh}")
        print(f"  Bundle  : {eb}")
        print()
        print("  Replay commands:")
        for cmd in capsule["replay_commands"]:
            print(f"    $ {cmd}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
