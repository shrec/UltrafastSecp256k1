#!/usr/bin/env python3
"""check_abi_version_sync.py — REL-ABI-VERSION-MISMATCH-001 regression gate.

The library reports its ABI integer at runtime via `ufsecp_abi_version()`, which
returns `UFSECP_ABI_VERSION`. That macro is templated in
`include/ufsecp/ufsecp_version.h.in` as `@PROJECT_VERSION_MAJOR@`, i.e. ABI == the
MAJOR field of VERSION.txt. Every language binding hard-fails context creation
when `ufsecp_abi_version() != EXPECTED_ABI`. Therefore every binding's
`EXPECTED_ABI` constant MUST equal VERSION.txt's MAJOR — otherwise EVERY binding
constructor throws "ABI mismatch" against the real library.

This gate caught REL-ABI-VERSION-MISMATCH-001: library shipped ABI 4 while all six
bindings hardcoded EXPECTED_ABI = 1, so every binding was dead-on-arrival.

Checks (run from the submodule root):
  1. VERSION.txt MAJOR is parseable.
  2. ufsecp_version.h.in derives UFSECP_ABI_VERSION from @PROJECT_VERSION_MAJOR@
     (not a drifting hardcode).
  3. The committed generated header (if present) hardcodes the same MAJOR.
  4. Every binding's EXPECTED_ABI constant == MAJOR.
  5. docs/BINDINGS_ABI_COMPAT.md "Current ABI version: N" == MAJOR.
  6. include/ufsecp/SUPPORTED_GUARANTEES.md version banner ABI == MAJOR.

Exit 0 = in sync, 1 = mismatch.
"""
import re
import sys
from pathlib import Path

VERSION_TXT = Path("VERSION.txt")
ABI_IN = Path("include/ufsecp/ufsecp_version.h.in")
ABI_HDR = Path("include/ufsecp/ufsecp_version.h")
COMPAT_DOC = Path("docs/BINDINGS_ABI_COMPAT.md")
GUARANTEES_DOC = Path("include/ufsecp/SUPPORTED_GUARANTEES.md")

# (label, path, regex capturing the integer)
BINDINGS = [
    ("rust", "bindings/rust/ufsecp/src/lib.rs",
     r"const\s+EXPECTED_ABI\s*:\s*u32\s*=\s*(\d+)\s*;"),
    ("go", "bindings/go/ufsecp.go",
     r"\bExpectedABI\s*=\s*(\d+)\b"),
    ("java", "bindings/java/src/com/ultrafast/ufsecp/Ufsecp.java",
     r"\bEXPECTED_ABI\s*=\s*(\d+)\s*;"),
    ("php", "bindings/php/src/Ufsecp.php",
     r"\bEXPECTED_ABI\s*=\s*(\d+)\s*;"),
    ("react-native", "bindings/react-native/lib/ufsecp.js",
     r"\bEXPECTED_ABI\s*=\s*(\d+)\s*;"),
    ("ruby", "bindings/ruby/lib/ufsecp.rb",
     r"\bEXPECTED_ABI\s*=\s*(\d+)\b"),
]


def fail(msg: str) -> None:
    print(f"::error::check_abi_version_sync: {msg}")


def check_guarantees_doc(text: str, major: int) -> list[str]:
    """Validate the current-version banner without changing the ABI>=1 floor."""
    match = re.search(
        r"^>\s*\*\*Version\*\*:\s*[^\n]*\(ABI\s+(\d+)\)\s*$",
        text,
        re.MULTILINE,
    )
    if not match:
        return ["version banner does not contain '(ABI N)'"]
    documented = int(match.group(1))
    if documented != major:
        return [f"version banner ABI {documented} != MAJOR {major}"]
    return []


def main() -> int:
    if not VERSION_TXT.exists():
        print(f"check_abi_version_sync: {VERSION_TXT} not found — run from the "
              "UltrafastSecp256k1 submodule root")
        return 1

    raw = VERSION_TXT.read_text().strip()
    m = re.match(r"^(\d+)\.\d+\.\d+", raw)
    if not m:
        fail(f"VERSION.txt '{raw}' is not MAJOR.MINOR.PATCH")
        return 1
    major = int(m.group(1))
    print(f"check_abi_version_sync: VERSION.txt MAJOR = {major} (ABI must equal this)")

    errors = 0

    # 2. template derives ABI from MAJOR, not a hardcode
    if ABI_IN.exists():
        intxt = ABI_IN.read_text()
        if not re.search(r"#define\s+UFSECP_ABI_VERSION\s+@PROJECT_VERSION_MAJOR@", intxt):
            fail(f"{ABI_IN}: UFSECP_ABI_VERSION must be templated as "
                 "@PROJECT_VERSION_MAJOR@ (do not hardcode the ABI in the template)")
            errors += 1
    else:
        fail(f"{ABI_IN} not found")
        errors += 1

    # 3. committed generated header (if present) matches MAJOR
    if ABI_HDR.exists():
        htxt = ABI_HDR.read_text()
        hm = re.search(r"#define\s+UFSECP_ABI_VERSION\s+(\d+)", htxt)
        if hm and int(hm.group(1)) != major:
            fail(f"{ABI_HDR}: hardcoded UFSECP_ABI_VERSION {hm.group(1)} != MAJOR {major}")
            errors += 1

    # 4. every binding EXPECTED_ABI == MAJOR
    for label, path, rx in BINDINGS:
        p = Path(path)
        if not p.exists():
            fail(f"{label}: binding file {path} not found")
            errors += 1
            continue
        bm = re.search(rx, p.read_text())
        if not bm:
            fail(f"{label}: could not find EXPECTED_ABI constant in {path}")
            errors += 1
            continue
        val = int(bm.group(1))
        if val != major:
            fail(f"{label}: EXPECTED_ABI = {val} != library ABI {major} ({path}) — "
                 "every context constructor would throw 'ABI mismatch'")
            errors += 1
        else:
            print(f"  [ok] {label:13} EXPECTED_ABI = {val}")

    # 5. docs/BINDINGS_ABI_COMPAT.md "Current ABI version: N"
    if COMPAT_DOC.exists():
        dtxt = COMPAT_DOC.read_text()
        dm = re.search(r"\*\*Current ABI version\*\*\s*:\s*(\d+)", dtxt)
        if dm and int(dm.group(1)) != major:
            fail(f"{COMPAT_DOC}: 'Current ABI version: {dm.group(1)}' != MAJOR {major}")
            errors += 1

    # 6. public guarantees version banner matches the current ABI. The
    # "Tier 1 -- Stable (ABI >= 1)" heading is a compatibility floor, not the
    # current ABI number, and intentionally remains unchanged.
    if GUARANTEES_DOC.exists():
        for problem in check_guarantees_doc(GUARANTEES_DOC.read_text(), major):
            fail(f"{GUARANTEES_DOC}: {problem}")
            errors += 1
    else:
        fail(f"{GUARANTEES_DOC} not found")
        errors += 1

    if errors:
        print(f"check_abi_version_sync: {errors} mismatch(es) — bindings/docs out of "
              f"sync with library ABI {major}")
        return 1
    print(f"check_abi_version_sync: all bindings + docs in sync with ABI {major} [PASS]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
