#!/usr/bin/env python3
"""Release package content guard.

Desktop release archives must ship only product libraries in lib/static and
lib/shared. Build-tree test, audit, exploit, fuzz, benchmark, or standalone
target libraries are valid CI artifacts, but they must never be copied into a
consumer package.

Usage:
  python3 ci/check_release_package_contents.py <package-dir-or-archive>...
  python3 ci/check_release_package_contents.py --json <package-dir-or-archive>...
"""

from __future__ import annotations

import argparse
import json
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

STATIC_ALLOWED = {
    "ultrafast_secp256k1.lib",
    "libultrafast_secp256k1.a",
    "ufsecp_s.lib",
    "libufsecp.a",
    "libufsecp_s.a",
}

SHARED_ALLOWED = {
    "ultrafast_secp256k1.dll",
    "ultrafast_secp256k1.lib",
    "libultrafast_secp256k1.so",
    "libultrafast_secp256k1.dylib",
    "ufsecp.dll",
    "ufsecp.lib",
    "libufsecp.so",
    "libufsecp.dylib",
}

SHARED_ALLOWED_PREFIXES = (
    "libultrafast_secp256k1.so.",
    "libufsecp.so.",
)

FORBIDDEN_TOKENS = (
    "_standalone",
    "test_",
    "libtest_",
    "exploit",
    "audit",
    "fuzz",
    "mutation",
    "benchmark",
    "bench_",
)


def is_library_name(name: str) -> bool:
    """Return true for archive/shared-library filenames we police."""
    lower = name.lower()
    return (
        lower.endswith(".a")
        or lower.endswith(".lib")
        or lower.endswith(".dll")
        or lower.endswith(".dylib")
        or lower.endswith(".so")
        or ".so." in lower
    )


def lib_area(path: str) -> tuple[str | None, str]:
    """Return (static/shared/misplaced/None, basename) for a package member."""
    normalized = path.replace("\\", "/")
    parts = PurePosixPath(normalized).parts
    if not parts:
        return None, ""
    basename = parts[-1]
    if not is_library_name(basename):
        return None, basename
    for idx, part in enumerate(parts[:-1]):
        if part == "lib" and idx + 1 < len(parts):
            area = parts[idx + 1]
            if area in ("static", "shared"):
                return area, basename
            return "misplaced", basename
    return "misplaced", basename


def is_allowed(area: str, basename: str) -> bool:
    """Return true when basename is a product library for this package area."""
    if area == "static":
        return basename in STATIC_ALLOWED
    if area == "shared":
        return basename in SHARED_ALLOWED or any(
            basename.startswith(prefix) for prefix in SHARED_ALLOWED_PREFIXES
        )
    return False


def classify_member(path: str) -> dict | None:
    """Classify one package member. Return a violation row or None."""
    area, basename = lib_area(path)
    if area is None:
        return None
    lower = basename.lower()
    if area == "misplaced":
        return {"path": path, "library": basename, "problem": "misplaced_library"}
    forbidden = [token for token in FORBIDDEN_TOKENS if token in lower]
    if forbidden:
        return {
            "path": path,
            "library": basename,
            "problem": "forbidden_test_or_audit_library",
            "tokens": forbidden,
        }
    if not is_allowed(area, basename):
        return {
            "path": path,
            "library": basename,
            "problem": "unexpected_library",
            "area": area,
        }
    return None


def directory_members(path: Path) -> list[str]:
    return [
        str(p.relative_to(path)).replace("\\", "/")
        for p in path.rglob("*")
        if p.is_file() or p.is_symlink()
    ]


def archive_members(path: Path) -> list[str]:
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as zf:
            return [info.filename for info in zf.infolist() if not info.is_dir()]
    if tarfile.is_tarfile(path):
        with tarfile.open(path) as tf:
            return [member.name for member in tf.getmembers() if member.isfile() or member.issym()]
    raise ValueError("unsupported package format")


def scan_package(path: Path) -> dict:
    if not path.exists():
        return {
            "package": str(path),
            "overall_pass": False,
            "error": "package_missing",
            "library_count": 0,
            "violations": [],
        }

    try:
        members = directory_members(path) if path.is_dir() else archive_members(path)
    except (OSError, tarfile.TarError, zipfile.BadZipFile, ValueError) as exc:
        return {
            "package": str(path),
            "overall_pass": False,
            "error": f"package_unreadable:{exc}",
            "library_count": 0,
            "violations": [],
        }

    library_members = [m for m in members if lib_area(m)[0] is not None]
    violations = [v for m in members if (v := classify_member(m)) is not None]
    if not library_members:
        violations.append({
            "path": str(path),
            "library": "",
            "problem": "missing_product_library",
        })

    return {
        "package": str(path),
        "overall_pass": not violations,
        "error": None,
        "library_count": len(library_members),
        "violations": violations,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("packages", nargs="+", help="Package directories or archives to validate")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    args = parser.parse_args()

    reports = [scan_package(Path(p)) for p in args.packages]
    overall = all(r["overall_pass"] for r in reports)
    payload = {"overall_pass": overall, "packages": reports}

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        for report in reports:
            if report["overall_pass"]:
                print(f"PASS release package contents: {report['package']} "
                      f"({report['library_count']} product libraries)")
                continue
            print(f"FAIL release package contents: {report['package']}")
            if report.get("error"):
                print(f"  error: {report['error']}")
            for v in report["violations"]:
                print(f"  [{v['problem']}] {v['path']}")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
