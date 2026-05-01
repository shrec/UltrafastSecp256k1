#!/usr/bin/env python3
"""
G-7: Multi-CI Reproducible Build Cross-Provider Check.

Compares SHA-256 hash files produced by two independent CI builds
and asserts bit-identical output. Designed to be called after a
matrix job where provider A and provider B both upload a hash file.

Usage:
    python3 ci/multi_ci_repro_check.py <hash_file_a> <hash_file_b> [--json]

Exit codes:
    0  hashes match — artifact is reproducible across providers
    1  hashes differ — artifact is NOT reproducible
    2  input file missing or empty

Hash file format (one line per artifact):
    <sha256hex>  <artifact_path>

This is the standard output of `sha256sum`, which the CI workflow
writes with:
    find build -name 'libufsecp*.a' -exec sha256sum {} \\;
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_hash_file(path: Path) -> dict[str, str]:
    """Return {filename: sha256} from a sha256sum-format file."""
    if not path.exists():
        print(f"ERROR: hash file not found: {path}", file=sys.stderr)
        sys.exit(2)

    entries: dict[str, str] = {}
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        print(f"ERROR: hash file is empty: {path}", file=sys.stderr)
        sys.exit(2)

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            print(f"WARN: unexpected line format: {line!r}", file=sys.stderr)
            continue
        sha, artifact = parts
        # Normalise artifact path to basename for cross-provider comparison
        entries[Path(artifact).name] = sha.lower()
    return entries


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="G-7: Compare SHA-256 hashes from two CI provider builds.",
    )
    parser.add_argument("file_a", help="Hash file from provider A (sha256sum format)")
    parser.add_argument("file_b", help="Hash file from provider B (sha256sum format)")
    parser.add_argument("--json", action="store_true", help="Emit JSON result to stdout")
    args = parser.parse_args(argv[1:])

    hashes_a = parse_hash_file(Path(args.file_a))
    hashes_b = parse_hash_file(Path(args.file_b))

    if not hashes_a:
        print("ERROR: provider A produced no hashes", file=sys.stderr)
        return 2
    if not hashes_b:
        print("ERROR: provider B produced no hashes", file=sys.stderr)
        return 2

    common = sorted(set(hashes_a) & set(hashes_b))
    only_a = sorted(set(hashes_a) - set(hashes_b))
    only_b = sorted(set(hashes_b) - set(hashes_a))
    mismatches = [(name, hashes_a[name], hashes_b[name])
                  for name in common if hashes_a[name] != hashes_b[name]]

    overall_pass = not mismatches and common

    if args.json:
        result = {
            "overall_pass": overall_pass,
            "compared": len(common),
            "mismatches": [
                {"artifact": name, "hash_a": ha, "hash_b": hb}
                for name, ha, hb in mismatches
            ],
            "only_in_a": only_a,
            "only_in_b": only_b,
            "artifacts": [
                {"name": name, "sha256": hashes_a[name], "match": hashes_a[name] == hashes_b.get(name)}
                for name in common
            ],
        }
        print(json.dumps(result, indent=2))
        return 0 if overall_pass else 1

    if not common:
        print("ERROR: no common artifacts to compare — providers built different things?")
        return 2

    print(f"Compared {len(common)} artifact(s) across two CI providers:")
    for name in common:
        ha = hashes_a[name]
        hb = hashes_b[name]
        status = "OK" if ha == hb else "DIFFER"
        print(f"  [{status}] {name}")
        print(f"         A: {ha}")
        if ha != hb:
            print(f"         B: {hb}")

    if only_a:
        print(f"Only in provider A: {only_a}")
    if only_b:
        print(f"Only in provider B: {only_b}")

    if mismatches:
        print(f"\nFAIL: {len(mismatches)} artifact(s) differ between CI providers — NOT reproducible across providers")
        return 1

    print(f"\nPASS: {len(common)} artifact(s) are bit-identical across both CI providers")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
