#!/usr/bin/env python3
"""Self-test for canonical audit/module count documentation replacement."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
MODULE_PATH = SCRIPT_DIR / "sync_module_count.py"


def load_sync_module_count():
    spec = importlib.util.spec_from_file_location("sync_module_count_selftest", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    module = load_sync_module_count()

    original = "\n".join([
        "# Summary",
        "GitHub cached README: 262 exploit PoCs / 367 modules.",
        "Standalone short form: 262 exploit PoCs.",
        "",
    ])
    updated, changes = module.make_replacements(
        original,
        total=431,
        exploit_mods=269,
        non_exploit=162,
        exploit_files=257,
        n_sections=10,
    )

    required = [
        "269 exploit PoCs / 431 modules",
        "Standalone short form: 269 exploit PoCs",
    ]
    missing = [needle for needle in required if needle not in updated]
    stale = ["262 exploit PoCs", "367 modules"]
    leftovers = [needle for needle in stale if needle in updated]

    if changes < 2 or missing or leftovers:
        print("sync_module_count paired-count regression failed", file=sys.stderr)
        print(f"changes={changes}", file=sys.stderr)
        print(f"missing={missing}", file=sys.stderr)
        print(f"leftovers={leftovers}", file=sys.stderr)
        print(updated, file=sys.stderr)
        return 1

    print("sync_module_count paired-count regression: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
