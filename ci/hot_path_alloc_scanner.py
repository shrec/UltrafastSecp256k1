#!/usr/bin/env python3
"""
hot_path_alloc_scanner.py  —  Hot-path heap allocation detector for UltrafastSecp256k1.

Detects heap allocations inside performance-critical functions (sign, verify, ECDH,
scalar_mul, pubkey derivation, BIP-32 derive, etc.).  Heap allocations on hot paths
violate the library's zero-allocation-per-op rule and degrade throughput.

Findings
--------
  HEAP_NEW     raw `new T` / `new T[N]` inside a hot-path function  [HIGH]
  HEAP_MALLOC  malloc / realloc / calloc call                        [HIGH]
  HEAP_VEC     local std::vector<T> construction (heap alloc)        [HIGH]
  HEAP_PUSH    push_back / emplace_back (potential vector realloc)    [MEDIUM]
  HEAP_STRING  local std::string construction                         [MEDIUM]
  HEAP_MAP     local std::map / unordered_map / set construction      [MEDIUM]
  HEAP_RET     function returns std::vector<T> by value (per-call)    [HIGH]

Hot-path classification (two complementary criteria)
-----------------------------------------------------
  A. File is in the hot-path file set (src/cpu/src/ecdsa.cpp, schnorr.cpp, ecdh.cpp,
     point.cpp, scalar.cpp, ct_sign.cpp, bip32.cpp, address.cpp, musig2.cpp,
     frost.cpp, and GPU equivalents).
  B. Function name contains a hot-path keyword: sign, verify, ecdh, scalar_mul,
     pubkey, compress, decompress, derive, keygen, recover, aggregate, partial.

Exempt functions (allocations are expected / one-time)
------------------------------------------------------
  Matched by name containing: precompute, build_table, build_gen, from_hex,
  to_hex, to_string, to_json, encrypt, decrypt, ecies, fuzz_, test_, benchmark_,
  from_bytes_variable, format_, print_, display_, demo_.

  Also exempt: static local variable initializers (`static std::vector`),
  function parameters (`const std::vector<...>&`), comment-only lines,
  and batch-output functions where the output IS the heap buffer.

Usage examples
--------------
  # Scan everything (auto-detect source dirs), human-readable:
  python3 ci/hot_path_alloc_scanner.py

  # Scan just src/cpu/src:
  python3 ci/hot_path_alloc_scanner.py --src-dir src/cpu/src

  # JSON output to file:
  python3 ci/hot_path_alloc_scanner.py --json -o report.json

  # Non-zero exit when any HIGH finding exists:
  python3 ci/hot_path_alloc_scanner.py --fail-on-high
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

# ──────────────────────────────────────────────────────────────────────────────
# Data model
# ──────────────────────────────────────────────────────────────────────────────

SEVERITY_RANK = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}


@dataclass
class Finding:
    file: str
    line: int
    function: str          # enclosing function name ('' if top-level)
    severity: str          # HIGH | MEDIUM | LOW
    category: str          # HEAP_NEW | HEAP_VEC | …
    message: str
    snippet: str           # raw source line (stripped)
    fix_hint: str


# ──────────────────────────────────────────────────────────────────────────────
# Classification tables
# ──────────────────────────────────────────────────────────────────────────────

# Source files whose *entire contents* are considered hot-path (relative suffix match)
HOT_PATH_FILES: frozenset[str] = frozenset({
    "src/cpu/src/ecdsa.cpp",
    "src/cpu/src/schnorr.cpp",
    "src/cpu/src/ecdh.cpp",
    "src/cpu/src/point.cpp",
    "src/cpu/src/scalar.cpp",
    "src/cpu/src/field.cpp",
    "src/cpu/src/ct_sign.cpp",
    "src/cpu/src/bip32.cpp",
    "src/cpu/src/address.cpp",
    "src/cpu/src/musig2.cpp",
    "src/cpu/src/frost.cpp",
    "src/cpu/src/adaptor.cpp",
    "src/cpu/src/ecmult.cpp",
    "src/cpu/src/group.cpp",
    # GPU backends are NOT hot-path for allocation purposes: vectors there are
    # marshalling buffers created *before* kernel launch.  The kernel launch
    # cost (hundreds of µs) dwarfs any host-side vector allocation.
    # Keeping them here only inflates false positives.
})

# Function name must contain one of these (case-insensitive) to be a hot path by name
HOT_PATH_KEYWORDS: Tuple[str, ...] = (
    "sign", "verify", "ecdh", "scalar_mul", "pubkey", "compress",
    "decompress", "derive", "keygen", "recover", "aggregate", "partial_sig",
    "point_mul", "ecmult", "schnorr", "tweak", "xonly",
)

# Function name contains any of these → NOT a hot path (setup / debug / test)
EXEMPT_KEYWORDS: Tuple[str, ...] = (
    "precompute", "build_table", "build_gen", "build_", "from_hex", "to_hex",
    "to_string", "to_json", "to_pem", "encrypt", "decrypt", "ecies",
    "fuzz_", "test_", "benchmark_", "bench_", "from_bytes_variable",
    "format_", "print_", "display_", "demo_", "dump_", "debug_",
    "selftest", "to_wif", "from_wif", "from_base58", "to_base58",
    "generate_sbom", "generate_report",
    "init_", "_init", "setup_", "_setup",  # one-time initialization
    "init_gen", "init_table", "init_ctx",  # explicit init patterns
    "identify_invalid",  # diagnostic path after batch failure, not steady-state hot path
    "keygen_begin",      # DKG/setup path, not steady-state signing/verification hot path
    "keygen_finalize",   # DKG/setup path, not steady-state signing/verification hot path
    "ufsecp_",  # C ABI wrappers — allocation is marshalling, not inner loop
    "segwit_",  # Address encoding helpers — not hot-path crypto
    "base58",   # Base58 encode/decode — 1x per address, not inner loop
    "bech32",   # Bech32 encode/decode — 1x per address, not inner loop
    "cashaddr",  # CashAddr helpers — not inner-loop crypto
    "wif_",     # WIF encode/decode — not inner-loop crypto
    "address_",  # Address generation — format/serialize, not tight crypto
    "convert_bits",  # Bech32 bit conversion helper
)


def _is_hot_path_file(rel_path: str) -> bool:
    """True if the file's relative path suffix matches a known hot-path file."""
    p = rel_path.replace("\\", "/")
    return any(p.endswith(h) or p == h for h in HOT_PATH_FILES)


def _is_hot_path_function(func_name: str) -> bool:
    """True if the function name contains a hot-path keyword."""
    lo = func_name.lower()
    return any(kw in lo for kw in HOT_PATH_KEYWORDS)


def _is_exempt_function(func_name: str) -> bool:
    """True if the function is setup/debug/test — allocations are expected."""
    lo = func_name.lower()
    return any(kw in lo for kw in EXEMPT_KEYWORDS)


# ──────────────────────────────────────────────────────────────────────────────
# Source helpers
# ──────────────────────────────────────────────────────────────────────────────

def _strip_line_comment(line: str) -> str:
    """Remove trailing // comment (simple heuristic)."""
    in_str = False
    i = 0
    while i < len(line) - 1:
        c = line[i]
        if c == '"' and (i == 0 or line[i - 1] != '\\'):
            in_str = not in_str
        if not in_str and line[i] == '/' and line[i + 1] == '/':
            return line[:i]
        i += 1
    return line


_FUNC_SIG = re.compile(
    r'^(?:(?:static|inline|virtual|explicit|constexpr|__attribute__\s*\(\([^)]*\)\))\s+)*'
    r'(?:[\w:<>*&,\s]+?\s+)?'          # return type (loose)
    r'([\w:~<>]+)\s*'                   # function name (group 1)
    r'\('                               # opening paren
)


def _extract_functions(lines: List[str]) -> Iterator[Tuple[str, int, int]]:
    """
    Yield (func_name, body_start_line, body_end_line) for each top-level or
    class-member function body found in `lines` (1-based line numbers).

    Uses a brace-depth parser.  Not a perfect C++ parser — good enough for
    pattern matching inside known library source files.
    """
    i = 0
    n = len(lines)

    while i < n:
        raw = lines[i]
        stripped = raw.strip()

        # Skip preprocessor, comments, blank lines for signature detection
        if stripped.startswith('#') or stripped.startswith('//') or not stripped:
            i += 1
            continue

        # Try to match a function signature ending with '{'
        # We look for lines that open a brace at depth 0
        # First: find a candidate name on this or recent lines
        func_name = ""
        m = _FUNC_SIG.match(stripped)
        if m:
            func_name = m.group(1).split("::")[-1]  # strip namespace

        # Find the opening brace (may be on the same line or next few)
        brace_line = i
        found_open = False
        look_ahead = min(i + 6, n)
        for j in range(i, look_ahead):
            if '{' in lines[j]:
                brace_line = j
                found_open = True
                break

        if not found_open:
            i += 1
            continue

        # Check we are at depth 0 by counting braces up to this point
        # (We skip over struct/class bodies by tracking depth properly)
        # Simple heuristic: only parse if the signature line looks function-like
        if not func_name:
            i += 1
            continue

        # Walk the body: count braces starting from brace_line
        depth = 0
        body_start = brace_line + 1  # 1-based
        body_end = brace_line + 1    # 1-based
        for k in range(brace_line, n):
            ln = lines[k]
            opens = ln.count('{')
            closes = ln.count('}')
            depth += opens - closes
            if depth > 0 and body_start == brace_line + 1:
                body_start = k + 1
            if depth == 0 and k >= brace_line:
                body_end = k + 1
                i = k + 1
                break
        else:
            i = n
            body_end = n

        if body_end > body_start:
            yield func_name, body_start, body_end


# ──────────────────────────────────────────────────────────────────────────────
# Per-line allocation pattern detectors
# ──────────────────────────────────────────────────────────────────────────────

# raw `new` allocation: `new SomeType` or `new SomeType[N]`
_NEW_ALLOC = re.compile(r'\bnew\s+[A-Za-z_]')

# C-style allocators
_MALLOC = re.compile(r'\b(malloc|realloc|calloc)\s*\(')

# Local std::vector construction (not a reference parameter, not static)
# Matches: `std::vector<T> varname(` or `std::vector<T> varname{` or `std::vector<T> varname;`
_VEC_LOCAL = re.compile(r'\bstd::vector\s*<[^>]+>\s+\w+\s*(?:\(|\{|;|=)')

# std::vector returned by value from a function
_VEC_RETURN = re.compile(r'\bstd::vector\s*<[^>]+>\s+\w[\w:~<>]*\s*\(')  # function return type

# push_back / emplace_back / emplace
_PUSH = re.compile(r'\.\s*(push_back|emplace_back|emplace)\s*\(')

# Local std::string construction
_STR_LOCAL = re.compile(r'\bstd::string\s+\w+\s*(?:\(|\{|;|=|[+])')

# Local map/set
_MAP_LOCAL = re.compile(r'\bstd::(unordered_map|unordered_set|map|set|multimap|multiset)\s*<')


def _is_static_local(line: str) -> bool:
    """True if the line declares a static local variable — initialized only once."""
    return bool(re.search(r'\bstatic\b', line))


def _is_one_time_init_context(lines: List[str], line_idx: int) -> bool:
    """True if the allocation is inside a static lambda/one-time initializer.

    Detects patterns like:
      static const GenTables* const t = []() { ... }();
      static const auto& x = *[]() { return new X; }();
    by scanning backwards for a 'static' declaration on a recent line.
    """
    start = max(0, line_idx - 32)
    for j in range(line_idx, start - 1, -1):
        ln = lines[j].strip()
        if re.search(r'\bstatic\b.*\[]', ln) or re.search(r'\bstatic\s+(const\s+)?[\w:<>*&]+.*=', ln):
            return True
        # Stop scanning at function boundary markers
        if ln == '{' and j < line_idx - 1:
            break
    return False


def _is_gpu_marshalling_file(rel_path: str) -> bool:
    """True if the file is a GPU backend host file where vectors are marshalling buffers."""
    p = rel_path.replace("\\", "/").lower()
    return any(x in p for x in (
        'gpu_backend_cuda', 'gpu_backend_opencl', 'gpu_backend_metal',
        'gpu_backend_cuda_host',
        'ecdsa_cuda', 'schnorr_cuda', 'ecdh_cuda', 'point_cuda',
        'ecdsa_ocl', 'schnorr_ocl', 'ecdsa_metal', 'schnorr_metal',
    ))


def _is_benchmark_helper_file(rel_path: str) -> bool:
    """True if the file is a benchmark/helper surface where vector returns are intentional."""
    p = rel_path.replace("\\", "/").lower()
    return "/benchmarks/" in p or p.startswith("benchmarks/") or "/examples/" in p or p.startswith("examples/")


def _is_parameter_line(line: str) -> bool:
    """
    True if this looks like a function parameter list line (contains `&` patterns
    typical of pass-by-reference vector params).
    """
    # const std::vector<...>& — passing by const-ref, no allocation
    return bool(re.search(r'const\s+std::vector\s*<[^>]+>\s*&', line))


def _is_batch_output(func_name: str) -> bool:
    """Batch functions necessarily collect results into a vector — allocation at batch level is tolerated."""
    lo = func_name.lower()
    return "batch" in lo or lo.endswith("_all") or lo.endswith("_many")


def check_line(
    rel_path: str,
    lines: List[str],
    func_name: str,
    func_start: int,
    func_end: int,
    is_hot: bool,
) -> List[Finding]:
    """Scan the body of one function for hot-path allocation patterns."""
    findings: List[Finding] = []

    if not is_hot:
        return findings
    if _is_exempt_function(func_name):
        return findings
    # GPU marshalling files: host-side vector allocations are expected before kernel launch
    if _is_gpu_marshalling_file(rel_path):
        return findings
    # Batch output: push_back warnings suppressed, but new/malloc still flagged
    batch = _is_batch_output(func_name)

    for i in range(func_start - 1, func_end - 1):
        if i >= len(lines):
            break
        raw = lines[i]
        line = _strip_line_comment(raw)
        stripped = line.strip()

        # Skip blank lines and pure comments
        if not stripped or stripped.startswith("//") or stripped.startswith("*"):
            continue

        rel_lineno = i + 1  # 1-based

        # HEAP_NEW: raw new expression
        if _NEW_ALLOC.search(line):
            # Exception: placement new, static one-time init
            if not re.search(r'\bnew\s*\(', line) and not _is_one_time_init_context(lines, i):
                findings.append(Finding(
                    file=rel_path, line=rel_lineno, function=func_name,
                    severity="HIGH", category="HEAP_NEW",
                    message=f"`new` heap allocation in hot-path function `{func_name}`",
                    snippet=raw.strip(),
                    fix_hint="Use stack allocation (std::array / fixed buffer) or pre-allocated scratch space",
                ))

        # HEAP_MALLOC: C-style allocator
        m = _MALLOC.search(line)
        if m:
            findings.append(Finding(
                file=rel_path, line=rel_lineno, function=func_name,
                severity="HIGH", category="HEAP_MALLOC",
                message=f"`{m.group(1)}()` in hot-path function `{func_name}`",
                snippet=raw.strip(),
                fix_hint="Replace with stack buffer or caller-supplied scratch allocation",
            ))

        # HEAP_VEC: local vector construction
        if _VEC_LOCAL.search(line) and not _is_static_local(line) and not _is_parameter_line(line):
            # Distinguish: `std::vector<T> name(count)` construction vs `std::vector<T>& ref`
            if '&' not in line.split('=')[0] and not _is_one_time_init_context(lines, i):
                findings.append(Finding(
                    file=rel_path, line=rel_lineno, function=func_name,
                    severity="HIGH", category="HEAP_VEC",
                    message=f"Local `std::vector` construction in hot-path function `{func_name}` — heap alloc per call",
                    snippet=raw.strip(),
                    fix_hint="Pre-allocate into a span/array passed by caller, or use a fixed-size stack buffer",
                ))

        # HEAP_PUSH: push_back / emplace_back
        if not batch and _PUSH.search(line):
            findings.append(Finding(
                file=rel_path, line=rel_lineno, function=func_name,
                severity="MEDIUM", category="HEAP_PUSH",
                message=f"`push_back` / `emplace_back` in hot-path function `{func_name}` — may trigger reallocation",
                snippet=raw.strip(),
                fix_hint="Pre-reserve capacity or replace with a fixed-size stack array + manual index",
            ))

        # HEAP_STRING: local std::string construction
        if _STR_LOCAL.search(line) and not _is_static_local(line) and '&' not in line.split('=')[0]:
            findings.append(Finding(
                file=rel_path, line=rel_lineno, function=func_name,
                severity="MEDIUM", category="HEAP_STRING",
                message=f"Local `std::string` in hot-path function `{func_name}` — heap alloc per call",
                snippet=raw.strip(),
                fix_hint="Use a stack char buffer or std::string_view; avoid string construction on hot paths",
            ))

        # HEAP_MAP: local map/set construction
        if _MAP_LOCAL.search(line) and not _is_static_local(line) and '&' not in line:
            findings.append(Finding(
                file=rel_path, line=rel_lineno, function=func_name,
                severity="MEDIUM", category="HEAP_MAP",
                message=f"Local `std::map`/`std::set` in hot-path function `{func_name}` — heap alloc per call",
                snippet=raw.strip(),
                fix_hint="Replace with a fixed-size flat array + linear search for small N, or move off hot path",
            ))

    return findings


# ──────────────────────────────────────────────────────────────────────────────
# File-level: check for functions returning std::vector by value
# ──────────────────────────────────────────────────────────────────────────────

_VEC_RETURN_FUNC = re.compile(r'^\s*std::vector\s*<[^>]+>\s+([\w:~<>]+)\s*\(')


def check_heap_return(rel_path: str, lines: List[str]) -> List[Finding]:
    """Flag functions that return std::vector<T> by value — forces heap alloc on every call."""
    findings: List[Finding] = []
    if _is_gpu_marshalling_file(rel_path) or _is_benchmark_helper_file(rel_path):
        return findings
    for i, raw in enumerate(lines):
        m = _VEC_RETURN_FUNC.match(raw)
        if not m:
            continue
        func_name = m.group(1).split("::")[-1]
        if _is_exempt_function(func_name):
            continue
        if not _is_hot_path_function(func_name):
            continue
        findings.append(Finding(
            file=rel_path, line=i + 1, function=func_name,
            severity="HIGH", category="HEAP_RET",
            message=f"`{func_name}` returns `std::vector<T>` by value — forces heap allocation on every call site",
            snippet=raw.strip(),
            fix_hint="Change signature to accept an output span/array; or accept std::vector<T>& out parameter",
        ))
    return findings


# ──────────────────────────────────────────────────────────────────────────────
# File scanner
# ──────────────────────────────────────────────────────────────────────────────

def scan_file(path: Path, base_dir: Path) -> List[Finding]:
    try:
        text = path.read_text(encoding='utf-8', errors='replace')
    except OSError:
        return []

    lines = text.splitlines()
    rel = str(path.relative_to(base_dir)) if path.is_relative_to(base_dir) else str(path)
    rel = rel.replace("\\", "/")

    file_is_hot = _is_hot_path_file(rel)
    all_findings: List[Finding] = []

    # Check return types at file level
    all_findings.extend(check_heap_return(rel, lines))

    # Extract and scan function bodies
    try:
        for func_name, body_start, body_end in _extract_functions(lines):
            func_is_hot = file_is_hot or _is_hot_path_function(func_name)
            all_findings.extend(
                check_line(rel, lines, func_name, body_start, body_end, func_is_hot)
            )
    except Exception as exc:
        all_findings.append(Finding(
            file=rel, line=0, function="", severity="LOW", category="INTERNAL",
            message=f"Scanner error: {exc}",
            snippet="", fix_hint="",
        ))

    return all_findings


# ──────────────────────────────────────────────────────────────────────────────
# Output
# ──────────────────────────────────────────────────────────────────────────────

def _sev_color(sev: str) -> str:
    return {"HIGH": "\033[31m", "MEDIUM": "\033[33m", "LOW": "\033[36m"}.get(sev, "")


RESET = "\033[0m"


def format_human(findings: List[Finding]) -> str:
    out: List[str] = []
    by_file: dict[str, List[Finding]] = {}
    for f in findings:
        by_file.setdefault(f.file, []).append(f)

    for file_path, flist in sorted(by_file.items()):
        out.append(f"\n{'─'*72}")
        out.append(f"  {file_path}  ({len(flist)} finding{'s' if len(flist) != 1 else ''})")
        out.append(f"{'─'*72}")
        for f in sorted(flist, key=lambda x: x.line):
            col = _sev_color(f.severity)
            out.append(f"  Line {f.line:5d}  [{col}{f.severity:6s}{RESET}]  [{f.category}]  fn: {f.function}")
            out.append(f"             {f.message}")
            if f.snippet:
                out.append(f"             ▶  {f.snippet[:120]}")
            out.append(f"             ✦  {f.fix_hint}")
    return "\n".join(out)


# ──────────────────────────────────────────────────────────────────────────────
# Source directory defaults
# ──────────────────────────────────────────────────────────────────────────────

_THIRD_PARTY_DIRS = frozenset({
    "node_modules", "_deps", "vendor", "third_party", "thirdparty",
    "external", ".git", "__pycache__",
})


def _collect_sources(src_dirs: List[Path]) -> Iterator[Path]:
    exts = {".cpp", ".cc", ".cxx", ".c", ".mm", ".cu", ".cl"}
    for d in src_dirs:
        if d.is_file():
            yield d
            continue
        for root, dirs, files in os.walk(d):
            dirs[:] = [dd for dd in dirs if dd not in _THIRD_PARTY_DIRS]
            for f in files:
                if any(f.endswith(e) for e in exts):
                    yield Path(root) / f


def _default_src_dirs(base: Path) -> List[Path]:
    candidates = [
        base / "cpu" / "src",
        base / "gpu" / "src",
        base / "opencl",
        base / "metal",
        base / "bindings",
        base / "include",
    ]
    return [p for p in candidates if p.exists()]


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Hot-path heap allocation detector for UltrafastSecp256k1",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "--src-dir", dest="src_dirs", action="append", metavar="DIR",
        help="Source directory or file to scan (repeatable; default: auto-detect)",
    )
    ap.add_argument(
        "--json", dest="output_json", action="store_true",
        help="Emit JSON output instead of human-readable text",
    )
    ap.add_argument(
        "-o", "--output", dest="output_file", metavar="FILE",
        help="Write output to FILE instead of stdout",
    )
    ap.add_argument(
        "--min-severity", dest="min_sev", default="LOW",
        choices=["LOW", "MEDIUM", "HIGH"],
        help="Minimum severity to include in output (default: LOW)",
    )
    ap.add_argument(
        "--fail-on-high", dest="fail_on_high", action="store_true",
        help="Exit with code 1 if any HIGH finding is present",
    )
    ap.add_argument(
        "--fail-on-findings", dest="fail_on_any", action="store_true",
        help="Exit with code 1 if any finding is present",
    )
    ap.add_argument(
        "--category", dest="categories", action="append", metavar="CAT",
        help="Only show this category (repeatable)",
    )

    args = ap.parse_args(argv)

    # Resolve base dir (repo root: two levels up from this script)
    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir.parent
    assert base_dir.exists(), f"Base dir not found: {base_dir}"

    # Collect source directories
    if args.src_dirs:
        src_dirs = [Path(d) for d in args.src_dirs]
    else:
        src_dirs = _default_src_dirs(base_dir)

    if not src_dirs:
        print("[hot_path_alloc_scanner] No source directories found; exiting.", file=sys.stderr)
        return 0

    # Scan
    all_findings: List[Finding] = []
    for src_path in _collect_sources(src_dirs):
        all_findings.extend(scan_file(src_path, base_dir))

    # Filter
    min_rank = SEVERITY_RANK[args.min_sev]
    all_findings = [f for f in all_findings if SEVERITY_RANK.get(f.severity, 0) >= min_rank]
    if args.categories:
        cats = {c.upper() for c in args.categories}
        all_findings = [f for f in all_findings if f.category in cats]

    # Sort: file, line
    all_findings.sort(key=lambda f: (f.file, f.line))

    # Summarise to stderr
    high = sum(1 for f in all_findings if f.severity == "HIGH")
    med  = sum(1 for f in all_findings if f.severity == "MEDIUM")
    low  = sum(1 for f in all_findings if f.severity == "LOW")
    cats: dict[str, int] = {}
    for f in all_findings:
        cats[f.category] = cats.get(f.category, 0) + 1
    cat_summary = " | ".join(f"{k} {v}" for k, v in sorted(cats.items()))

    print(
        f"[hot_path_alloc_scanner] {len(src_dirs)} source dirs, "
        f"{len(all_findings)} findings "
        f"({high} HIGH, {med} MEDIUM, {low} LOW)"
        + (f"  [{cat_summary}]" if cat_summary else ""),
        file=sys.stderr,
    )

    # Format
    if args.output_json:
        output_str = json.dumps(
            {"findings": [asdict(f) for f in all_findings]},
            indent=2,
        )
    else:
        output_str = format_human(all_findings)

    # Write
    if args.output_file:
        Path(args.output_file).write_text(output_str, encoding="utf-8")
        print(f"[hot_path_alloc_scanner] Report written to {args.output_file}", file=sys.stderr)
    else:
        print(output_str)

    # Exit code
    if args.fail_on_high and high > 0:
        return 1
    if args.fail_on_any and all_findings:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
