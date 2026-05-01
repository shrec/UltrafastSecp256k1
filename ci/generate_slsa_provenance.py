#!/usr/bin/env python3
"""
generate_slsa_provenance.py — Local SLSA v1.0 provenance generation
====================================================================

Generates a SLSA v1.0 provenance document (intoto Statement) for a given
UltrafastSecp256k1 build artifact.  In CI this is handled by the official
slsa-github-generator action (see .github/workflows/slsa-provenance.yml).
This local script is for developers who build locally and want to produce
a verifiable provenance document for their artifact.

The document is written to docs/slsa_provenance.json (or --output path).

Usage
-----
  # After building locally:
  python3 ci/generate_slsa_provenance.py \\
      --artifact  build/ufsecp_shared.so \\
      --output    docs/slsa_provenance.json

  # Generate metadata-only (no artifact — records build environment):
  python3 ci/generate_slsa_provenance.py --metadata-only

SLSA references
---------------
  https://slsa.dev/spec/v1.0/provenance
  https://github.com/slsa-framework/slsa-github-generator
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
LIB_ROOT = SCRIPT_DIR.parent

BUILDER_ID = "https://github.com/slsa-framework/slsa-github-generator/.github/workflows/generator_generic_slsa3.yml"
BUILD_TYPE  = "https://slsa.dev/provenance/v1"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=LIB_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def _git_repo() -> str:
    try:
        url = subprocess.check_output(
            ["git", "remote", "get-url", "origin"], cwd=LIB_ROOT, text=True
        ).strip()
        # Normalise SSH → HTTPS form
        if url.startswith("git@github.com:"):
            url = "https://github.com/" + url[len("git@github.com:"):]
        return url.rstrip(".git")
    except Exception:
        return "https://github.com/shrec/UltrafastSecp256k1"


def generate(artifact: Path | None, metadata_only: bool, output: Path) -> dict:
    commit = _git_commit()
    repo   = _git_repo()
    now    = datetime.datetime.now(datetime.timezone.utc).isoformat()

    subjects: list[dict] = []
    if artifact and artifact.exists() and not metadata_only:
        digest = _sha256(artifact)
        subjects.append({
            "name": artifact.name,
            "digest": {"sha256": digest},
        })
    else:
        subjects.append({
            "name": "UltrafastSecp256k1-local-build",
            "digest": {"sha256": "metadata-only-no-artifact"},
        })

    provenance = {
        "_type": "https://in-toto.io/Statement/v0.1",
        "subject": subjects,
        "predicateType": BUILD_TYPE,
        "predicate": {
            "buildDefinition": {
                "buildType": BUILD_TYPE,
                "externalParameters": {
                    "repository": repo,
                    "ref": f"refs/heads/dev",
                    "commit": commit,
                },
                "resolvedDependencies": [
                    {
                        "uri": repo,
                        "digest": {"gitCommit": commit},
                    }
                ],
            },
            "runDetails": {
                "builder": {
                    "id": BUILDER_ID,
                    "version": {"slsa-github-generator": "v2.1.0"},
                },
                "metadata": {
                    "invocationId": f"local-build-{now}",
                    "startedOn": now,
                    "finishedOn": now,
                },
                "byproducts": [
                    {
                        "name": "build-environment",
                        "content": {
                            "platform": platform.platform(),
                            "python": sys.version,
                            "os_uname": list(os.uname()),
                        },
                    }
                ],
            },
        },
    }
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=None,
                        help="Path to the artifact to record (optional)")
    parser.add_argument("--metadata-only", action="store_true",
                        help="Generate metadata-only provenance (no artifact hash)")
    parser.add_argument("--output", type=Path,
                        default=LIB_ROOT / "docs" / "slsa_provenance.json",
                        help="Output path for provenance JSON")
    args = parser.parse_args()

    doc = generate(args.artifact, args.metadata_only, args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(f"SLSA provenance written to {args.output}")
    print(f"  subject: {doc['subject'][0]['name']}")
    print(f"  commit:  {doc['predicate']['buildDefinition']['externalParameters']['commit']}")


if __name__ == "__main__":
    main()
