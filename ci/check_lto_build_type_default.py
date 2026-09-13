#!/usr/bin/env python3
"""check_lto_build_type_default.py — LTO-DEBUG-001 regression gate.

`SECP256K1_USE_LTO` must default OFF for a Debug build and ON for an optimized
one, and an explicit `-DSECP256K1_USE_LTO=ON` must still win in Debug.

Why this is worth a gate
------------------------
The library exports ThinLTO to consumers as an INTERFACE link option:

    target_link_options(${SECP256K1_LIB_NAME} INTERFACE -flto=thin -fuse-ld=lld)

so every executable that links it redoes whole-program codegen at link time.
This tree builds ~434 standalone test executables, which turns that one line
into the dominant cost of any build that produces them. In a Debug or sanitizer
build it buys nothing: the binary is run once, under 10-20x instrumentation, to
find a bug.

Measured on one machine, same 85 targets in both configurations, per-edge times
taken from ninja's own .ninja_log (clang-18, Debug, SECP256K1_USE_ASM=OFF):

           link CPU (85 targets)   median link   max link
  LTO ON           2110.65 s          25.11 s     83.11 s
  LTO OFF            18.54 s           0.21 s      0.41 s
                  ----------------------------------------
                       114x              117x        201x

The CI MSan job showed the same shape before the fix: 2 h 03 m in Build, 91.5%
of it linking (112 min over 434 links, median 25 s), 8.5% compiling.

Restoring the unconditional `option(... ON)` would silently put those hours
back, and nothing else in the tree would notice -- the build would still be
correct, just hours slower. Hence a gate rather than a comment.

What is asserted
----------------
  LTO-1  Debug defaults to OFF                     (the fix)
  LTO-2  Release defaults to ON                    (NEGATIVE CONTROL: without
         this, LTO-1 would also pass against "LTO deleted entirely")
  LTO-3  Debug + explicit -DSECP256K1_USE_LTO=ON stays ON
         (the escape hatch the comment in src/cpu/CMakeLists.txt promises)

Each case is a real `cmake` configure into a throwaway directory; the verdict is
read back out of that directory's CMakeCache.txt, so this tests the build system
as CI actually invokes it rather than pattern-matching the CMakeLists text.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Configure-only, with everything optional switched off: this gate cares about
# one cache variable, and a lean configure keeps it inside the fast-gate budget.
COMMON_ARGS = [
    "-DSECP256K1_BUILD_TESTS=OFF",
    "-DSECP256K1_BUILD_EXAMPLES=OFF",
    "-DSECP256K1_BUILD_BENCH=OFF",
    "-DSECP256K1_BUILD_JAVA=OFF",
]


def read_cache_bool(build_dir: Path, name: str) -> str | None:
    """Return the raw value of a BOOL cache entry, or None if it is absent."""
    cache = build_dir / "CMakeCache.txt"
    if not cache.exists():
        return None
    prefix = f"{name}:BOOL="
    for line in cache.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith(prefix):
            return line[len(prefix):].strip()
    return None


def configure(label: str, extra: list[str]) -> tuple[str | None, str]:
    """Configure into a temp dir and return (SECP256K1_USE_LTO value, output)."""
    with tempfile.TemporaryDirectory(prefix="ufsecp-lto-gate-") as tmp:
        build_dir = Path(tmp) / "b"
        cmd = ["cmake", "-S", str(REPO_ROOT), "-B", str(build_dir)] + COMMON_ARGS + extra
        proc = subprocess.run(cmd, capture_output=True, text=True)
        value = read_cache_bool(build_dir, "SECP256K1_USE_LTO")
        if value is None:
            tail = (proc.stdout + proc.stderr).strip().splitlines()[-25:]
            return None, f"{label}: configure produced no SECP256K1_USE_LTO entry\n" + "\n".join(
                "    " + t for t in tail
            )
        return value, ""


def truthy(value: str) -> bool:
    return value.upper() in {"ON", "1", "TRUE", "YES", "Y"}


def main() -> int:
    if shutil.which("cmake") is None:
        print("FAIL: cmake not found. This gate configures the real project; a CI")
        print("      runner without cmake cannot validate the build system, and a")
        print("      silent skip here would hide a multi-hour regression.")
        return 1

    failures: list[str] = []

    cases = [
        (
            "LTO-1",
            "Debug defaults to LTO OFF",
            ["-DCMAKE_BUILD_TYPE=Debug"],
            False,
            "ThinLTO is INTERFACE-propagated, so a Debug build would redo "
            "whole-program codegen for every one of ~434 test executables "
            "(measured 114x more link CPU) to produce binaries that only ever "
            "run once under a sanitizer.",
        ),
        (
            "LTO-2",
            "Release defaults to LTO ON",
            ["-DCMAKE_BUILD_TYPE=Release"],
            True,
            "NEGATIVE CONTROL. Release is where LTO earns its cost and where "
            "every benchmark and release artifact is built. If this flips, LTO-1 "
            "would still pass while LTO had simply been deleted.",
        ),
        (
            "LTO-3",
            "Debug with an explicit -DSECP256K1_USE_LTO=ON stays ON",
            ["-DCMAKE_BUILD_TYPE=Debug", "-DSECP256K1_USE_LTO=ON"],
            True,
            "The build-type default must not become a prohibition: option() "
            "leaves a user-set cache entry alone, and src/cpu/CMakeLists.txt "
            "documents that escape hatch.",
        ),
    ]

    for tag, title, extra, want_on, why in cases:
        value, err = configure(tag, extra)
        if value is None:
            failures.append(err)
            print(f"  [FAIL] {tag}: {title} -- configure failed")
            continue
        got_on = truthy(value)
        if got_on == want_on:
            print(f"  [ OK ] {tag}: {title} (SECP256K1_USE_LTO={value})")
        else:
            failures.append(
                f"{tag}: {title}\n"
                f"    expected SECP256K1_USE_LTO={'ON' if want_on else 'OFF'}, got {value}\n"
                f"    {why}"
            )
            print(f"  [FAIL] {tag}: {title} (SECP256K1_USE_LTO={value})")

    print()
    if failures:
        print(f"FAIL: {len(failures)}/{len(cases)} LTO default checks failed\n")
        for f in failures:
            print("  " + f.replace("\n", "\n  "))
        print()
        print("  Fix: src/cpu/CMakeLists.txt sets _secp256k1_lto_default from")
        print("       CMAKE_BUILD_TYPE before option(SECP256K1_USE_LTO ...).")
        return 1

    print(f"PASS: {len(cases)}/{len(cases)} — LTO default follows the build type")
    return 0


if __name__ == "__main__":
    sys.exit(main())
