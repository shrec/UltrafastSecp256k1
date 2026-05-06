#!/usr/bin/env python3
"""
ci_gate_detect.py  --  Impact-based CI gate level detection

Analyzes changed files to determine whether the full "hard gate" (Tier 1 + Tier 2)
or only the "light gate" (Tier 1) is needed.

Usage:
    python3 scripts/ci_gate_detect.py                  # auto-detect from git
    python3 scripts/ci_gate_detect.py --base origin/dev
    python3 scripts/ci_gate_detect.py --files f1.cpp f2.cpp

Exit codes:
    0 = light gate (docs/bindings/tooling only)
    1 = hard gate (crypto/CT/ABI change)

Output (for CI):
    JSON to stdout with gate level and reasoning.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
LIB_ROOT = SCRIPT_DIR.parent

# Paths that trigger the hard gate (crypto/CT/ABI/GPU)
HARD_GATE_PATTERNS = [
    'src/cpu/src/ct_',
    'src/cpu/src/ecdsa',
    'src/cpu/src/schnorr',
    'src/cpu/src/field_',
    'src/cpu/src/scalar_',
    'src/cpu/src/bip32',
    'src/cpu/src/musig',
    'src/cpu/src/frost',
    'src/cpu/src/ecies',
    'src/cpu/src/adaptor',
    'src/cpu/include/secp256k1/ct/',
    'src/cpu/include/secp256k1/detail/secure_erase',
    'src/cpu/include/secp256k1/detail/value_barrier',
    'src/opencl/kernels/',
    'src/cuda/include/',
    'src/metal/shaders/',
    'gpu/',
    'include/ufsecp/ufsecp.h',
    'src/cpu/src/ufsecp_impl.cpp',
    'include/ufsecp/ufsecp_gpu.h',
    'src/cpu/src/ufsecp_gpu_impl.cpp',
    'include/ufsecp/ufsecp_version.h',
]

# Paths that are always light gate (no crypto impact)
LIGHT_GATE_PATTERNS = [
    'docs/',
    'bindings/',
    'scripts/',
    'benchmarks/',
    'examples/',
    '.github/',
    'schemas/',
    'tools/ai_memory/',
    'tools/source_graph_kit/',
    'README.md',
    'CHANGELOG.md',
    'LICENSE',
    '.gitignore',
    '.clang-format',
]


def get_changed_files(base: str = 'origin/dev') -> list[str]:
    """Get list of changed files relative to base."""
    try:
        merge_base = subprocess.run(
            ['git', 'merge-base', 'HEAD', base],
            capture_output=True, text=True, cwd=str(LIB_ROOT),
        )
        if merge_base.returncode != 0:
            # Fallback: use HEAD~1
            merge_base_sha = 'HEAD~1'
        else:
            merge_base_sha = merge_base.stdout.strip()

        result = subprocess.run(
            ['git', 'diff', '--name-only', merge_base_sha, 'HEAD'],
            capture_output=True, text=True, cwd=str(LIB_ROOT),
        )
        if result.returncode == 0:
            return [f.strip() for f in result.stdout.strip().splitlines() if f.strip()]
    except Exception:
        pass
    return []


def classify_file(path: str) -> str:
    """Classify a file path as 'hard' or 'light'."""
    for pattern in HARD_GATE_PATTERNS:
        if pattern in path:
            return 'hard'
    return 'light'


_DOCS_ONLY_EXTENSIONS = {'.md', '.txt', '.rst', '.adoc'}
_DOCS_ONLY_PREFIXES = ('docs/', 'CHANGELOG', 'LICENSE', 'README')


def is_docs_only_file(path: str) -> bool:
    """Conservative check: True only if the file is purely a documentation asset.

    A mixed PR with even ONE .cpp/.hpp/.py file outside docs/ must return False.
    This prevents gate.yml from treating 'code + doc' PRs as docs-only.
    """
    p = Path(path)
    if p.suffix in _DOCS_ONLY_EXTENSIONS:
        return True
    for prefix in _DOCS_ONLY_PREFIXES:
        if path.startswith(prefix):
            return True
    return False


def detect_gate_level(files: list[str]) -> dict:
    """Determine the gate level from a list of changed files."""
    hard_triggers = []
    light_files = []
    unknown_files = []

    for f in files:
        classification = classify_file(f)
        if classification == 'hard':
            hard_triggers.append(f)
        else:
            # Check if it's a known light pattern
            is_light = any(p in f for p in LIGHT_GATE_PATTERNS)
            if is_light:
                light_files.append(f)
            else:
                # Unknown files default to hard gate for safety
                unknown_files.append(f)

    gate = 'hard' if (hard_triggers or unknown_files) else 'light'

    # docs_only: True only if EVERY changed file is a pure documentation asset.
    # A single .cpp alongside a .md must produce docs_only=false.
    docs_only = bool(files) and all(is_docs_only_file(f) for f in files)

    return {
        'gate': gate,
        'docs_only': docs_only,
        'changed_files': len(files),
        'hard_triggers': hard_triggers,
        'light_files': light_files,
        'unknown_files': unknown_files,
        'reason': (
            f'{len(hard_triggers)} crypto/CT/ABI files changed'
            if hard_triggers
            else f'{len(unknown_files)} unclassified files (defaulting to hard)'
            if unknown_files
            else f'only {len(light_files)} docs/tooling files changed'
        ),
    }


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description='CI gate level detection')
    parser.add_argument('--base', default='origin/dev', help='Git base ref')
    parser.add_argument('--files', nargs='*', help='Explicit file list (skip git)')
    args = parser.parse_args()

    if args.files:
        files = args.files
    else:
        files = get_changed_files(args.base)

    if not files:
        result = {'gate': 'light', 'changed_files': 0, 'reason': 'no changed files detected'}
    else:
        result = detect_gate_level(files)

    print(json.dumps(result, indent=2))
    return 1 if result['gate'] == 'hard' else 0


if __name__ == '__main__':
    sys.exit(main())
