#!/usr/bin/env python3
"""
Self-test for ci/check_entropy_source_integrity.py — DON'T TRUST, VERIFY.

Prove Rule 1 blocks a banned local csprng_fill and a fail-open (bool, no abort) canonical,
and passes a single fail-closed source. The real repo (post-fix) must pass.
"""
import importlib.util
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load():
    spec = importlib.util.spec_from_file_location(
        "esi", os.path.join(ROOT, "ci", "check_entropy_source_integrity.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


FAILCLOSED = "inline void csprng_fill(unsigned char* b, std::size_t n) noexcept { if(!ok) std::abort(); }"
FAILOPEN = "inline bool csprng_fill(unsigned char* b, std::size_t n) { return read(b,n); }"


def main() -> int:
    m = _load()
    fails = []

    # 0. The real repo must pass (single fail-closed source; bip39 dup removed).
    if m.run() != 0:
        fails.append("real repo should pass (single fail-closed csprng_fill)")

    # 1. A SECOND csprng_fill definition (banned local copy) MUST block.
    files = {m.CANONICAL: FAILCLOSED, "src/cpu/src/bip39.cpp": "static bool csprng_fill(uint8_t* b, size_t n){ return false; }"}
    blk, _ = m.analyze(files)
    if not blk:
        fails.append("a banned local csprng_fill definition MUST block")

    # 2. A fail-OPEN canonical (bool, no abort) MUST block.
    blk2, _ = m.analyze({m.CANONICAL: FAILOPEN})
    if not blk2:
        fails.append("a fail-open canonical csprng_fill (bool, no abort) MUST block")

    # 3. A single fail-closed source must NOT block.
    blk3, _ = m.analyze({m.CANONICAL: FAILCLOSED})
    if blk3:
        fails.append("a single fail-closed csprng_fill must NOT block")

    # 4. fail-closed detector sanity.
    if not m.canonical_is_fail_closed(FAILCLOSED):
        fails.append("canonical_is_fail_closed must accept void+abort")
    if m.canonical_is_fail_closed(FAILOPEN):
        fails.append("canonical_is_fail_closed must reject a bool/no-abort signature")

    print("=" * 60)
    if fails:
        print("  check_entropy_source_integrity SELF-TEST: FAILED")
        for f in fails:
            print("   - " + f)
        print("=" * 60)
        return 1
    print("  check_entropy_source_integrity SELF-TEST PASSED")
    print("  (single-source + fail-closed proven; banned local copy + fail-open block)")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
