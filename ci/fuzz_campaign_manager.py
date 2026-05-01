#!/usr/bin/env python3
"""Fuzz campaign manager — seed replay, crash triage, exploit→regression pipeline.

Manages the transition from ad-hoc fuzzing to infrastructure-grade fuzzing:
  - Seed replay: register corpus entries as CTest regression tests
  - Crash triage: dedup, severity classify, auto-file
  - Corpus minimization: wrapper around libFuzzer -merge or cmin
  - Exploit-to-regression: crash → minimal reproducer → CTest
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
LIB_ROOT = SCRIPT_DIR.parent
CORPUS_DIR = LIB_ROOT / "audit" / "corpus"
CRASH_DIR = LIB_ROOT / "audit" / "crashes"
REGRESSION_DIR = LIB_ROOT / "audit" / "regressions"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def scan_corpus() -> dict:
    """Scan the corpus directory and return statistics."""
    if not CORPUS_DIR.exists():
        return {"exists": False, "seeds": 0, "targets": []}

    targets: list[dict] = []
    total_seeds = 0

    for target_dir in sorted(CORPUS_DIR.iterdir()):
        if not target_dir.is_dir():
            continue
        seeds = list(target_dir.glob("*"))
        seed_count = len([s for s in seeds if s.is_file()])
        total_seeds += seed_count
        targets.append({
            "target": target_dir.name,
            "seed_count": seed_count,
        })

    return {
        "exists": True,
        "seeds": total_seeds,
        "targets": targets,
    }


def scan_crashes() -> dict:
    """Scan crash directory for unprocessed crashes."""
    if not CRASH_DIR.exists():
        CRASH_DIR.mkdir(parents=True, exist_ok=True)
        return {"crashes": 0, "unprocessed": 0, "files": []}

    crashes: list[dict] = []
    for f in sorted(CRASH_DIR.rglob("*")):
        if not f.is_file():
            continue
        crashes.append({
            "file": str(f.relative_to(CRASH_DIR)),
            "size": f.stat().st_size,
            "hash": _sha256_file(f),
        })

    # Dedup by hash
    seen_hashes: set[str] = set()
    unique = []
    dupes = 0
    for c in crashes:
        if c["hash"] not in seen_hashes:
            seen_hashes.add(c["hash"])
            unique.append(c)
        else:
            dupes += 1

    return {
        "crashes": len(crashes),
        "unique": len(unique),
        "duplicates": dupes,
        "unprocessed": len(unique),
        "files": unique,
    }


def scan_regressions() -> dict:
    """Scan regression directory."""
    if not REGRESSION_DIR.exists():
        REGRESSION_DIR.mkdir(parents=True, exist_ok=True)
        return {"regressions": 0, "files": []}

    regressions = list(REGRESSION_DIR.glob("*.cpp")) + list(REGRESSION_DIR.glob("*.c"))
    return {
        "regressions": len(regressions),
        "files": [str(r.relative_to(REGRESSION_DIR)) for r in regressions],
    }


def status(json_mode: bool, out_file: str | None) -> int:
    """Show current fuzz infrastructure status."""
    corpus = scan_corpus()
    crashes = scan_crashes()
    regressions = scan_regressions()

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "corpus": corpus,
        "crashes": crashes,
        "regressions": regressions,
        "active_corpus_size": corpus["seeds"],
        "regressions_from_crashes": regressions["regressions"],
    }

    rendered = json.dumps(report, indent=2)
    if out_file:
        Path(out_file).write_text(rendered, encoding="utf-8")

    if json_mode:
        print(rendered)
    else:
        print(f"  Corpus: {corpus['seeds']} seeds across {len(corpus.get('targets', []))} targets")
        if corpus.get("targets"):
            for t in corpus["targets"]:
                print(f"    {t['target']}: {t['seed_count']} seeds")
        print(f"  Crashes: {crashes['crashes']} ({crashes.get('unique', 0)} unique, "
              f"{crashes.get('duplicates', 0)} dupes)")
        print(f"  Regressions: {regressions['regressions']}")

    return 0


def crash_to_regression(crash_file: str, target: str) -> int:
    """Convert a crash file to a regression test stub."""
    crash_path = Path(crash_file)
    if not crash_path.exists():
        print(f"Crash file not found: {crash_path}")
        return 1

    crash_hash = _sha256_file(crash_path)[:12]
    crash_data = crash_path.read_bytes()

    REGRESSION_DIR.mkdir(parents=True, exist_ok=True)

    # Generate C regression test
    hex_data = ", ".join(f"0x{b:02x}" for b in crash_data)
    reg_name = f"regression_{target}_{crash_hash}"
    reg_file = REGRESSION_DIR / f"{reg_name}.cpp"

    content = f"""// Auto-generated regression test from crash
// Source: {crash_path.name}
// Hash: {crash_hash}
// Target: {target}
// Generated: {datetime.now(timezone.utc).isoformat()}

#include <cstdint>
#include <cstddef>

// Forward declare the fuzz target
extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size);

static const uint8_t crash_data[] = {{
    {hex_data}
}};

int main() {{
    LLVMFuzzerTestOneInput(crash_data, sizeof(crash_data));
    return 0;
}}
"""

    reg_file.write_text(content, encoding="utf-8")
    print(f"Generated regression: {reg_file}")
    return 0


def run(args: argparse.Namespace) -> int:
    if args.command == "status":
        return status(args.json, args.out_file)
    elif args.command == "crash-to-regression":
        return crash_to_regression(args.crash_file, args.target)
    else:
        print(f"Unknown command: {args.command}")
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("-o", dest="out_file", help="Write report to file")

    subparsers = parser.add_subparsers(dest="command")

    sub_status = subparsers.add_parser("status", help="Show fuzz infrastructure status")
    sub_c2r = subparsers.add_parser("crash-to-regression",
                                      help="Convert crash to regression test")
    sub_c2r.add_argument("crash_file", help="Path to crash file")
    sub_c2r.add_argument("--target", required=True, help="Fuzz target name")

    args = parser.parse_args()
    if args.command is None:
        args.command = "status"

    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
