#!/usr/bin/env python3
"""
gen_build_options.py — generate docs/BUILD_OPTIONS.md from the CMake option() declarations.

SINGLE SOURCE OF TRUTH = the `option(...)` / `cmake_dependent_option(...)` calls in the
project's CMakeLists.txt files. Each carries its own description string and default, so the
reference is always complete and never drifts. Do NOT hand-edit docs/BUILD_OPTIONS.md.

Usage:
  python3 ci/gen_build_options.py            # (re)write docs/BUILD_OPTIONS.md
  python3 ci/gen_build_options.py --check    # exit 1 if the doc is stale (CI drift gate)
  python3 ci/gen_build_options.py --stdout   # print to stdout, write nothing

Deterministic output (no timestamps) so --check is stable.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "docs" / "BUILD_OPTIONS.md"

# Directories never scanned (vendored upstreams, build trees, agent scratch).
SKIP_PARTS = {
    "out", "_research_repos", "_libsecp256k1", "third_party", "build", ".git",
    "bitcoin-core-dev", "litecoin-core-dev", "dogecoin-core-dev", "libbitcoin-system",
    ".claude", "node_modules", "cmake-build-debug", "cmake-build-release",
}

# Local/scratch build-output directory NAME PATTERNS, not just the exact
# literals above. Agents and developers routinely create ad-hoc build trees
# such as "build-audit", "build_bench_run", "build-review-lbtc-gpu",
# "out-review" — these contain CMake-*generated* probe files (e.g.
# `CMakeFiles/CheckCUDA/CMakeLists.txt`) that must never feed the option scan
# or the "Generated from:" footer. Without this, docs/BUILD_OPTIONS.md becomes
# non-deterministic: --check passes or fails purely depending on which
# transient build directories happen to exist on the machine that last ran
# this generator, even when zero option() declarations actually changed.
SKIP_PART_RE = re.compile(r"^(?:build|out|cmake-build)(?:[-_].*)?$")


def _is_skipped_part(part: str) -> bool:
    return part in SKIP_PARTS or bool(SKIP_PART_RE.match(part))

# Map a CMakeLists' directory (relative to ROOT) to a human scope label + sort order.
SCOPE_ORDER = [
    (".",          "Global / top-level (backends, GPU op selection, install)"),
    ("src/cpu",    "CPU implementation (crypto modules, optimization, integration)"),
    ("src/gpu",    "GPU (common host layer)"),
    ("src/cuda",   "CUDA backend"),
    ("src/opencl", "OpenCL backend"),
    ("src/metal",  "Apple Metal backend"),
    ("src/rocm",   "ROCm / HIP backend"),
    ("include/ufsecp",              "C ABI library (ufsecp_* shared/static)"),
    ("compat/libsecp256k1_shim",    "libsecp256k1 compatibility shim"),
    ("compat/libbitcoin_bridge",    "libbitcoin bridge (script-sig batch verify + scan)"),
    ("audit",                       "Audit / test harness (differential, fuzz, protocol tests)"),
]

OPT_RE = re.compile(r"\b(option|cmake_dependent_option)\s*\(")


def _scope_for(rel_dir: str) -> str:
    for d, label in SCOPE_ORDER:
        if rel_dir == d:
            return label
    return f"Other ({rel_dir})"


def parse_options(text: str):
    """Yield (name, description, default, kind) for each option call in `text`."""
    out = []
    i = 0
    n = len(text)
    while True:
        m = OPT_RE.search(text, i)
        if not m:
            break
        kind = m.group(1)
        j = m.end()  # just past '('
        # NAME
        while j < n and text[j].isspace():
            j += 1
        nm = re.match(r"[A-Za-z0-9_]+", text[j:])
        if not nm:
            i = m.end()
            continue
        name = nm.group(0)
        j += nm.end()
        # advance to the opening quote of the description (bail if ')' comes first)
        while j < n and text[j] != '"':
            if text[j] == ")":
                break
            j += 1
        if j >= n or text[j] != '"':
            i = m.end()
            continue
        # description runs to the next quote (CMake descriptions don't escape quotes)
        close = text.find('"', j + 1)
        if close < 0:
            i = m.end()
            continue
        desc = re.sub(r"\s+", " ", text[j + 1:close]).strip()
        j = close + 1
        # default = next bare token (ON/OFF/variable) up to whitespace or ')'
        while j < n and text[j].isspace():
            j += 1
        dt = re.match(r"[^\s)]+", text[j:])
        default = dt.group(0) if dt else "?"
        out.append((name, desc, default, kind))
        i = max(j, m.end())
    return out


def collect():
    """Return {scope_label: {name: (desc, default, kind)}} deduped by richest description."""
    best: dict[str, tuple[str, str, str, str]] = {}  # name -> (desc, default, kind, scope)
    files = []
    for p in sorted(ROOT.rglob("CMakeLists.txt")):
        rel = p.relative_to(ROOT)
        if any(_is_skipped_part(part) for part in rel.parts):
            continue
        files.append(rel)
        rel_dir = str(rel.parent) if str(rel.parent) != "." else "."
        scope = _scope_for(rel_dir)
        for name, desc, default, kind in parse_options(p.read_text(errors="replace")):
            prev = best.get(name)
            # keep the declaration with the longest (richest) description
            if prev is None or len(desc) > len(prev[0]):
                best[name] = (desc, default, kind, scope)
    grouped: dict[str, dict] = {}
    for name, (desc, default, kind, scope) in best.items():
        grouped.setdefault(scope, {})[name] = (desc, default, kind)
    return grouped, len(best), files


def render() -> str:
    grouped, total, files = collect()
    lines = []
    lines.append("# Build Options Reference")
    lines.append("")
    lines.append("> **Auto-generated — do not edit by hand.** Regenerate with "
                 "`python3 ci/gen_build_options.py`. The source of truth is the "
                 "`option()` / `cmake_dependent_option()` declarations in the project "
                 "`CMakeLists.txt` files (each carries its own description + default).")
    lines.append(">")
    lines.append("> Defaults below are the **CMake declaration defaults**. Named build "
                 "profiles (see [CMakePresets.json](../CMakePresets.json) and "
                 "[BUILDING.md](BUILDING.md)) override many of them for a minimal footprint "
                 "per coin / use case. A `cmake_dependent_option` is only honoured when its "
                 "guard condition holds (otherwise it is forced off).")
    lines.append("")
    lines.append(f"**{total} options** across {len([s for s in grouped])} scope(s). "
                 "Set any flag at configure time with `-D<FLAG>=ON|OFF`.")
    lines.append("")
    lines.append("```bash")
    lines.append("# Example: CPU build with the shim + MuSig2, no ZK/FROST")
    lines.append("cmake -S . -B out/mybuild -G Ninja -DCMAKE_BUILD_TYPE=Release \\")
    lines.append("  -DSECP256K1_BUILD_SHIM=ON -DSECP256K1_BUILD_MUSIG2=ON \\")
    lines.append("  -DSECP256K1_BUILD_ZK=OFF -DSECP256K1_BUILD_FROST=OFF")
    lines.append("```")
    lines.append("")

    # Stable scope ordering: known scopes first (declared order), then any extras alpha.
    known = [label for _d, label in SCOPE_ORDER]
    ordered = [s for s in known if s in grouped] + \
              sorted(s for s in grouped if s not in known)

    for scope in ordered:
        opts = grouped[scope]
        lines.append(f"## {scope}")
        lines.append("")
        lines.append("| Flag | Default | Description |")
        lines.append("|------|---------|-------------|")
        for name in sorted(opts):
            desc, default, kind = opts[name]
            dep = " _(conditional)_" if kind == "cmake_dependent_option" else ""
            # escape pipes in the description for markdown table safety
            d = desc.replace("|", "\\|")
            lines.append(f"| `{name}` | `{default}` | {d}{dep} |")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("_Generated from:_ " + ", ".join(f"`{f}`" for f in files))
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    content = render()
    if "--stdout" in sys.argv:
        sys.stdout.write(content)
        return 0
    if "--check" in sys.argv:
        current = DOC.read_text() if DOC.exists() else ""
        if current != content:
            print("::error::docs/BUILD_OPTIONS.md is stale — run "
                  "`python3 ci/gen_build_options.py` and commit.")
            return 1
        print("gen_build_options: docs/BUILD_OPTIONS.md is in sync.")
        return 0
    DOC.write_text(content)
    n = content.count("\n| `")
    print(f"gen_build_options: wrote {DOC.relative_to(ROOT)} ({n} options).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
