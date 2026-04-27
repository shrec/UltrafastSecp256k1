#!/usr/bin/env python3
"""
check_core_build_mode.py
========================
Verifies that the UltrafastSecp256k1 library can be built in
"Bitcoin Core compatible mode":
  - No C++20 leaked PUBLIC to consumers
  - No FetchContent in the core library build
  - No config.ini required for the build
  - Adequate CMake minimum version
  - Shim uses PRIVATE C++20 and PRIVATE link to fastsecp256k1

Exit codes:
  0 — all checks pass (warnings do not change exit code)
  1 — one or more FAIL findings

Usage:
  python3 check_core_build_mode.py [--json]
"""

import sys
import re
import json
import argparse
from pathlib import Path

# ---------------------------------------------------------------------------
# Path resolution — library root is parent of scripts/
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
LIB_ROOT   = SCRIPT_DIR.parent

# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------

def read_file(path: Path) -> str | None:
    """Return file contents as a string, or None if the file does not exist."""
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def glob_cmake_files(root: Path) -> list[Path]:
    """Return all CMakeLists.txt files under root, excluding build dirs."""
    results = []
    for p in root.rglob("CMakeLists.txt"):
        # Skip build directories (conventional names)
        parts = set(p.parts)
        if any(
            part.startswith("build") or part in {"_build", ".build", "out", "dist"}
            for part in p.relative_to(root).parts[:-1]
        ):
            continue
        results.append(p)
    return sorted(results)


# ---------------------------------------------------------------------------
# Individual check implementations
# Each returns a result dict with keys:
#   id, name, status ("PASS" | "WARN" | "FAIL"), detail
# ---------------------------------------------------------------------------

def check_no_public_cxx_std_20() -> dict:
    """
    No target_compile_features(...PUBLIC...cxx_std_20) in any CMakeLists.txt.
    Only PRIVATE cxx_std_20 is acceptable.
    """
    check_id   = "no_public_cxx_std_20"
    check_name = "No PUBLIC cxx_std_20 (C++20 must not leak to consumers)"

    cmake_files = glob_cmake_files(LIB_ROOT)
    # Pattern: target_compile_features( <target> PUBLIC ... cxx_std_20 ...)
    # We accept whitespace and possible other features between PUBLIC and cxx_std_20.
    pat = re.compile(
        r'target_compile_features\s*\([^)]*\bPUBLIC\b[^)]*\bcxx_std_20\b',
        re.DOTALL | re.IGNORECASE,
    )

    violations: list[str] = []
    for cmake_path in cmake_files:
        text = read_file(cmake_path)
        if text is None:
            continue
        m = pat.search(text)
        if m:
            rel = cmake_path.relative_to(LIB_ROOT)
            # Find line number
            line_no = text[: m.start()].count("\n") + 1
            violations.append(f"{rel}:{line_no}: PUBLIC cxx_std_20 found")

    if violations:
        return {
            "id":     check_id,
            "name":   check_name,
            "status": "FAIL",
            "detail": violations,
        }
    return {
        "id":     check_id,
        "name":   check_name,
        "status": "PASS",
        "detail": ["All cxx_std_20 usages are PRIVATE (or not present)"],
    }


def check_no_fetchcontent_in_core() -> dict:
    """
    FetchContent must not be used unconditionally in the core library CMake.
    Optional/test-only sections that are guarded by option() flags are WARN.
    Pure comment references are ignored.
    """
    check_id   = "no_fetchcontent_core"
    check_name = "No FetchContent in core library CMake (no network deps at config time)"

    # Core library files (not test/optional/tool files)
    core_files = [
        LIB_ROOT / "CMakeLists.txt",
        LIB_ROOT / "cpu" / "CMakeLists.txt",
        LIB_ROOT / "compat" / "libsecp256k1_shim" / "CMakeLists.txt",
    ]

    # Pattern for actual FetchContent usage (not in comments)
    fetch_pat = re.compile(r'^\s*(?!#).*\bFetchContent_', re.MULTILINE)
    # Pattern for a comment-only reference
    comment_pat = re.compile(r'^\s*#.*FetchContent', re.MULTILINE)

    fails:  list[str] = []
    warns:  list[str] = []

    for cmake_path in core_files:
        text = read_file(cmake_path)
        if text is None:
            continue
        rel = cmake_path.relative_to(LIB_ROOT)

        for m in fetch_pat.finditer(text):
            snippet = m.group(0).strip()
            line_no = text[: m.start()].count("\n") + 1

            # Check if this line is inside an option-guarded block.
            # Heuristic: look backwards for an 'if(' that mentions an option
            # variable before finding the matching closing 'endif()'.
            before = text[: m.start()]
            # Count open if() vs endif() -- very rough nesting depth estimate
            open_count   = len(re.findall(r'\bif\s*\(', before, re.IGNORECASE))
            close_count  = len(re.findall(r'\bendif\s*\(', before, re.IGNORECASE))
            # If we're inside at least one conditional, treat as WARN
            if open_count > close_count:
                warns.append(
                    f"{rel}:{line_no}: FetchContent inside conditional block "
                    f"(likely optional/test-only)"
                )
            else:
                fails.append(
                    f"{rel}:{line_no}: Unconditional FetchContent usage — "
                    f"would run at configure time for all consumers"
                )

    if fails:
        return {
            "id":     check_id,
            "name":   check_name,
            "status": "FAIL",
            "detail": fails + warns,
        }
    if warns:
        return {
            "id":     check_id,
            "name":   check_name,
            "status": "WARN",
            "detail": warns + ["Note: FetchContent only in guarded optional sections — acceptable"],
        }
    return {
        "id":     check_id,
        "name":   check_name,
        "status": "PASS",
        "detail": ["No FetchContent usage found in core library CMake files"],
    }


def check_no_config_ini_required() -> dict:
    """
    config.ini must not be required by core library CMake or source files.
    """
    check_id   = "no_config_ini_required"
    check_name = "No config.ini required (core library builds with standard CMake options)"

    # Files to check: all CMakeLists in lib root + cpu/, shim/
    cmake_files = [
        LIB_ROOT / "CMakeLists.txt",
        LIB_ROOT / "cpu" / "CMakeLists.txt",
        LIB_ROOT / "compat" / "libsecp256k1_shim" / "CMakeLists.txt",
    ]

    pat = re.compile(r'\bconfig\.ini\b', re.IGNORECASE)
    violations: list[str] = []

    for cmake_path in cmake_files:
        text = read_file(cmake_path)
        if text is None:
            continue
        for m in pat.finditer(text):
            # Skip pure comment lines
            line_start = text.rfind("\n", 0, m.start()) + 1
            line_text  = text[line_start : text.find("\n", m.start())]
            if re.match(r'\s*#', line_text):
                continue
            line_no = text[: m.start()].count("\n") + 1
            rel     = cmake_path.relative_to(LIB_ROOT)
            violations.append(f"{rel}:{line_no}: config.ini reference: {line_text.strip()}")

    # Also scan core C++ source headers/files for config.ini reads
    for src_dir in [LIB_ROOT / "cpu" / "src", LIB_ROOT / "include"]:
        if not src_dir.exists():
            continue
        for src_file in src_dir.rglob("*"):
            if src_file.suffix not in (".cpp", ".hpp", ".h", ".cxx", ".c"):
                continue
            text = read_file(src_file)
            if text is None or not pat.search(text):
                continue
            # Only flag if it looks like an actual open/parse
            if re.search(r'open\s*\(\s*["\'].*config\.ini|config\.ini.*open', text, re.IGNORECASE):
                violations.append(
                    f"{src_file.relative_to(LIB_ROOT)}: source file reads config.ini"
                )

    if violations:
        return {
            "id":     check_id,
            "name":   check_name,
            "status": "FAIL",
            "detail": violations,
        }
    return {
        "id":     check_id,
        "name":   check_name,
        "status": "PASS",
        "detail": ["No config.ini dependency found in core library CMake or source files"],
    }


def check_cmake_minimum_version() -> dict:
    """
    cmake_minimum_required must be 3.18 or higher in the root CMakeLists.txt.
    (Bitcoin Core uses 3.22+, so 3.18 is acceptable as the library minimum.)
    """
    check_id   = "cmake_minimum_version"
    check_name = "CMake minimum version >= 3.18 (Bitcoin Core uses 3.22+)"

    root_cmake = LIB_ROOT / "CMakeLists.txt"
    text = read_file(root_cmake)
    if text is None:
        return {
            "id":     check_id,
            "name":   check_name,
            "status": "FAIL",
            "detail": [f"Root CMakeLists.txt not found at {root_cmake}"],
        }

    pat = re.compile(r'cmake_minimum_required\s*\(\s*VERSION\s+([\d.]+)', re.IGNORECASE)
    m   = pat.search(text)
    if not m:
        return {
            "id":     check_id,
            "name":   check_name,
            "status": "FAIL",
            "detail": ["cmake_minimum_required not found in root CMakeLists.txt"],
        }

    version_str = m.group(1)
    parts = [int(x) for x in version_str.split(".")]
    # Pad to at least [major, minor]
    while len(parts) < 2:
        parts.append(0)

    major, minor = parts[0], parts[1]
    if (major, minor) >= (3, 18):
        return {
            "id":     check_id,
            "name":   check_name,
            "status": "PASS",
            "detail": [f"cmake_minimum_required(VERSION {version_str}) — satisfies >= 3.18"],
        }
    return {
        "id":     check_id,
        "name":   check_name,
        "status": "FAIL",
        "detail": [
            f"cmake_minimum_required(VERSION {version_str}) is too old "
            f"(need >= 3.18, Bitcoin Core uses 3.22+)"
        ],
    }


def check_no_global_cxx_standard_in_shim() -> dict:
    """
    The shim CMakeLists.txt must not set CMAKE_CXX_STANDARD globally
    (which would leak into the parent scope / consuming project).
    It should use target_compile_features(... PRIVATE cxx_std_20) instead.
    """
    check_id   = "shim_no_global_cxx_standard"
    check_name = (
        "Shim CMakeLists.txt: no global CMAKE_CXX_STANDARD "
        "(must use target_compile_features PRIVATE)"
    )

    shim_cmake = LIB_ROOT / "compat" / "libsecp256k1_shim" / "CMakeLists.txt"
    text = read_file(shim_cmake)
    if text is None:
        return {
            "id":     check_id,
            "name":   check_name,
            "status": "FAIL",
            "detail": [f"Shim CMakeLists.txt not found at {shim_cmake}"],
        }

    # Global set of CMAKE_CXX_STANDARD (not inside target_compile_features)
    global_pat = re.compile(r'^\s*set\s*\(\s*CMAKE_CXX_STANDARD\b', re.MULTILINE)
    # PRIVATE target_compile_features with cxx_std_20
    private_pat = re.compile(
        r'target_compile_features\s*\([^)]*\bPRIVATE\b[^)]*\bcxx_std_20\b',
        re.DOTALL | re.IGNORECASE,
    )

    violations: list[str] = []
    passes:     list[str] = []

    for m in global_pat.finditer(text):
        line_no = text[: m.start()].count("\n") + 1
        violations.append(
            f"compat/libsecp256k1_shim/CMakeLists.txt:{line_no}: "
            f"global set(CMAKE_CXX_STANDARD ...) leaks C++ standard to parent scope"
        )

    if private_pat.search(text):
        passes.append(
            "target_compile_features(secp256k1_shim PRIVATE cxx_std_20) found — correct"
        )
    else:
        violations.append(
            "target_compile_features(...PRIVATE cxx_std_20) not found in shim CMakeLists.txt"
        )

    if violations:
        return {
            "id":     check_id,
            "name":   check_name,
            "status": "FAIL",
            "detail": violations + passes,
        }
    return {
        "id":     check_id,
        "name":   check_name,
        "status": "PASS",
        "detail": passes,
    }


def check_shim_private_link() -> dict:
    """
    The shim must link to fastsecp256k1 (or its aliases) with PRIVATE visibility,
    so that fastsecp256k1's compile requirements do not propagate to consumers.
    """
    check_id   = "shim_private_link"
    check_name = (
        "Shim links fastsecp256k1 as PRIVATE "
        "(prevents propagation of fastsecp256k1 requirements to consumers)"
    )

    shim_cmake = LIB_ROOT / "compat" / "libsecp256k1_shim" / "CMakeLists.txt"
    text = read_file(shim_cmake)
    if text is None:
        return {
            "id":     check_id,
            "name":   check_name,
            "status": "FAIL",
            "detail": [f"Shim CMakeLists.txt not found at {shim_cmake}"],
        }

    # Match: target_link_libraries(secp256k1_shim PRIVATE fastsecp256k1|secp256k1_fast|secp256k1::fast)
    private_pat = re.compile(
        r'target_link_libraries\s*\(\s*secp256k1_shim\s+PRIVATE\s+'
        r'(?:fastsecp256k1|secp256k1_fast|secp256k1::fast)\b',
        re.DOTALL | re.IGNORECASE,
    )
    # Public linkage is a violation
    public_pat = re.compile(
        r'target_link_libraries\s*\(\s*secp256k1_shim\s+PUBLIC\s+'
        r'(?:fastsecp256k1|secp256k1_fast|secp256k1::fast)\b',
        re.DOTALL | re.IGNORECASE,
    )

    violations: list[str] = []
    passes:     list[str] = []

    if public_pat.search(text):
        violations.append(
            "PUBLIC linkage to fastsecp256k1 found — all fastsecp256k1 "
            "compile requirements (C++20, -march=native, etc.) would propagate to consumers"
        )

    if private_pat.search(text):
        passes.append(
            "target_link_libraries(secp256k1_shim PRIVATE fastsecp256k1) found — correct"
        )
    else:
        # Maybe there's no link at all
        any_link_pat = re.compile(
            r'target_link_libraries\s*\(\s*secp256k1_shim\b',
            re.DOTALL | re.IGNORECASE,
        )
        if not any_link_pat.search(text):
            violations.append(
                "No target_link_libraries for secp256k1_shim found — shim must link fastsecp256k1"
            )
        else:
            violations.append(
                "PRIVATE link to fastsecp256k1/secp256k1_fast/secp256k1::fast not found"
            )

    if violations:
        return {
            "id":     check_id,
            "name":   check_name,
            "status": "FAIL",
            "detail": violations + passes,
        }
    return {
        "id":     check_id,
        "name":   check_name,
        "status": "PASS",
        "detail": passes,
    }


# ---------------------------------------------------------------------------
# All checks in presentation order
# ---------------------------------------------------------------------------

def check_core_backend_mode_option() -> dict:
    """SECP256K1_CORE_BACKEND_MODE option must exist in root CMakeLists.txt."""
    check_id   = "core_backend_mode_option"
    check_name = "SECP256K1_CORE_BACKEND_MODE CMake option (deterministic Core build)"
    root_cmake = LIB_ROOT / "CMakeLists.txt"
    if not root_cmake.exists():
        return {"id": check_id, "name": check_name, "status": "SKIP",
                "detail": ["CMakeLists.txt not found"]}
    text = root_cmake.read_text(errors="replace")
    pat = re.compile(r'option\s*\(\s*SECP256K1_CORE_BACKEND_MODE\b', re.IGNORECASE)
    if pat.search(text):
        return {"id": check_id, "name": check_name, "status": "PASS",
                "detail": ["SECP256K1_CORE_BACKEND_MODE option found"]}
    return {"id": check_id, "name": check_name, "status": "FAIL",
            "detail": ["SECP256K1_CORE_BACKEND_MODE option missing — needed for Core deterministic mode"]}


ALL_CHECKS = [
    check_no_public_cxx_std_20,
    check_no_fetchcontent_in_core,
    check_no_config_ini_required,
    check_cmake_minimum_version,
    check_no_global_cxx_standard_in_shim,
    check_shim_private_link,
    check_core_backend_mode_option,
]

# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

COL_RESET  = "\033[0m"
COL_GREEN  = "\033[32m"
COL_RED    = "\033[31m"
COL_YELLOW = "\033[33m"
COL_CYAN   = "\033[36m"
COL_BOLD   = "\033[1m"

STATUS_WIDTH = 5
NAME_WIDTH   = 70


def _colorize(text: str, color: str) -> str:
    if not sys.stdout.isatty():
        return text
    return f"{color}{text}{COL_RESET}"


def print_table(results: list[dict]) -> None:
    header    = f"{'STATUS':<{STATUS_WIDTH}}  {'CHECK'}"
    separator = "-" * (STATUS_WIDTH + 2 + NAME_WIDTH)

    print()
    print(_colorize(header, COL_BOLD))
    print(separator)

    for r in results:
        status = r["status"]
        if status == "PASS":
            colored_status = _colorize(f"{'PASS':<{STATUS_WIDTH}}", COL_GREEN)
        elif status == "WARN":
            colored_status = _colorize(f"{'WARN':<{STATUS_WIDTH}}", COL_YELLOW)
        else:
            colored_status = _colorize(f"{'FAIL':<{STATUS_WIDTH}}", COL_RED)

        print(f"{colored_status}  {r['name']}")
        for line in r["detail"]:
            indent_color = COL_CYAN if status == "PASS" else COL_YELLOW if status == "WARN" else COL_RED
            print(f"         {_colorize('>', indent_color)} {line}")

    print(separator)

    passes = sum(1 for r in results if r["status"] == "PASS")
    warns  = sum(1 for r in results if r["status"] == "WARN")
    fails  = sum(1 for r in results if r["status"] == "FAIL")
    total  = len(results)

    parts = [f"{total} checks"]
    if passes:
        parts.append(_colorize(f"{passes} passed", COL_GREEN))
    if warns:
        parts.append(_colorize(f"{warns} warnings", COL_YELLOW))
    if fails:
        parts.append(_colorize(f"{fails} failed", COL_RED))

    print("  " + ", ".join(parts))
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Verify UltrafastSecp256k1 can be built in Bitcoin Core compatible mode: "
            "no leaked C++20, no FetchContent, no config.ini dependency."
        )
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON instead of a table",
    )
    args = parser.parse_args()

    results = [fn() for fn in ALL_CHECKS]

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print(f"\nBitcoin Core build mode check — library root: {LIB_ROOT}")
        print_table(results)

    # Exit 1 only on FAIL; WARN is acceptable
    any_fail = any(r["status"] == "FAIL" for r in results)
    return 1 if any_fail else 0


if __name__ == "__main__":
    sys.exit(main())
