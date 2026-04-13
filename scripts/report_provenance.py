#!/usr/bin/env python3
"""
report_provenance.py  --  Build provenance for audit/assurance reports

Collects reproducible provenance metadata for any report:
  - commit SHA (git rev-parse HEAD)
  - dirty tree flag (git diff --stat)
  - submodule SHAs
  - toolchain fingerprint (compiler + version)
  - key build flags (or their hash)
  - platform info (OS, arch)

Usage as module:
    from report_provenance import collect_provenance
    prov = collect_provenance()

Usage standalone:
    python3 scripts/report_provenance.py              # JSON to stdout
    python3 scripts/report_provenance.py -o prov.json  # write to file
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
LIB_ROOT = SCRIPT_DIR.parent


def _run(cmd: list[str], cwd: Optional[Path] = None, timeout: int = 10) -> Optional[str]:
    """Run a command, return stripped stdout or None on any failure."""
    try:
        r = subprocess.run(
            cmd, capture_output=True, text=True,
            cwd=str(cwd or LIB_ROOT), timeout=timeout,
        )
        if r.returncode == 0:
            return r.stdout.strip()
    except Exception:
        pass
    return None


def get_git_commit() -> dict[str, Any]:
    """Return commit SHA, short SHA, and dirty flag."""
    sha = _run(['git', 'rev-parse', 'HEAD'])
    short = _run(['git', 'rev-parse', '--short=10', 'HEAD'])
    dirty_out = _run(['git', 'status', '--porcelain', '--untracked-files=no'])
    dirty = bool(dirty_out) if dirty_out is not None else None

    return {
        'sha': sha,
        'short': short,
        'dirty': dirty,
        'ref': _run(['git', 'rev-parse', '--abbrev-ref', 'HEAD']),
    }


def get_submodule_shas() -> list[dict[str, str]]:
    """Return submodule paths and their SHAs."""
    out = _run(['git', 'submodule', 'status', '--recursive'])
    if not out:
        return []
    result = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        # Format: " <sha> <path> (<desc>)" or "+<sha> <path> (<desc>)"
        parts = line.lstrip('+-U ').split()
        if len(parts) >= 2:
            result.append({'path': parts[1], 'sha': parts[0]})
    return result


def get_toolchain() -> dict[str, Any]:
    """Detect C/C++ compiler and version."""
    compilers = ['cc', 'gcc', 'clang', 'c++', 'g++', 'clang++']
    # Also check environment variables
    for env_var in ('CC', 'CXX', 'CMAKE_C_COMPILER', 'CMAKE_CXX_COMPILER'):
        val = os.environ.get(env_var)
        if val:
            compilers.insert(0, val)

    detected = {}
    seen = set()
    for comp in compilers:
        version_out = _run([comp, '--version'])
        if version_out and comp not in seen:
            seen.add(comp)
            first_line = version_out.splitlines()[0] if version_out else ''
            detected[comp] = first_line

    cmake_version = _run(['cmake', '--version'])
    if cmake_version:
        first_line = cmake_version.splitlines()[0] if cmake_version else ''
        detected['cmake'] = first_line

    ninja_version = _run(['ninja', '--version'])
    if ninja_version:
        detected['ninja'] = ninja_version

    return detected


def get_build_flags(build_dir: Optional[Path] = None) -> dict[str, Any]:
    """Extract key build flags from CMakeCache.txt."""
    if build_dir is None:
        # Try common build directories
        for name in ('build', 'build_opencl', 'build_rel', 'build-cuda'):
            candidate = LIB_ROOT / name
            if (candidate / 'CMakeCache.txt').exists():
                build_dir = candidate
                break
            # Check parent (suite repo wraps library)
            candidate = LIB_ROOT.parent.parent / name
            if (candidate / 'CMakeCache.txt').exists():
                build_dir = candidate
                break

    if build_dir is None or not (build_dir / 'CMakeCache.txt').exists():
        return {'available': False, 'reason': 'no CMakeCache.txt found'}

    cache_path = build_dir / 'CMakeCache.txt'
    try:
        cache_text = cache_path.read_text(errors='replace')
    except Exception:
        return {'available': False, 'reason': 'CMakeCache.txt unreadable'}

    # Extract key flags
    key_vars = [
        'CMAKE_BUILD_TYPE',
        'CMAKE_C_COMPILER',
        'CMAKE_CXX_COMPILER',
        'CMAKE_C_FLAGS',
        'CMAKE_CXX_FLAGS',
        'CMAKE_C_FLAGS_RELEASE',
        'CMAKE_CXX_FLAGS_RELEASE',
        'SECP256K1_ENABLE_CT',
        'SECP256K1_ENABLE_GPU',
        'SECP256K1_ENABLE_CUDA',
        'SECP256K1_ENABLE_OPENCL',
        'SECP256K1_ENABLE_METAL',
        'UFSECP_ENABLE_ETHEREUM',
    ]

    flags = {}
    for var in key_vars:
        pattern = re.compile(rf'^{re.escape(var)}(?::\w+)?=(.*)$', re.MULTILINE)
        m = pattern.search(cache_text)
        if m:
            flags[var] = m.group(1)

    # Compute a hash of all collected flags for quick comparison
    flag_str = json.dumps(flags, sort_keys=True)
    flags_hash = hashlib.sha256(flag_str.encode()).hexdigest()[:16]

    return {
        'available': True,
        'build_dir': str(build_dir),
        'flags': flags,
        'flags_hash': flags_hash,
    }


def get_platform_info() -> dict[str, str]:
    """Return platform identification."""
    return {
        'system': platform.system(),
        'release': platform.release(),
        'machine': platform.machine(),
        'python': platform.python_version(),
    }


def get_version() -> Optional[str]:
    """Read VERSION.txt."""
    vfile = LIB_ROOT / 'VERSION.txt'
    if vfile.exists():
        return vfile.read_text().strip()
    return None


def collect_provenance(build_dir: Optional[Path] = None) -> dict[str, Any]:
    """Collect full provenance metadata.

    Returns a dict suitable for embedding in any report JSON.
    """
    git = get_git_commit()
    submodules = get_submodule_shas()
    toolchain = get_toolchain()
    build_flags = get_build_flags(build_dir)
    plat = get_platform_info()
    version = get_version()

    return {
        'collected_at': datetime.now(timezone.utc).isoformat(),
        'version': version,
        'git': git,
        'submodules': submodules if submodules else None,
        'toolchain': toolchain,
        'build_flags': build_flags,
        'platform': plat,
    }


def main():
    out_file = None
    if '-o' in sys.argv:
        idx = sys.argv.index('-o')
        if idx + 1 < len(sys.argv):
            out_file = sys.argv[idx + 1]

    prov = collect_provenance()
    output = json.dumps(prov, indent=2, ensure_ascii=False)

    if out_file:
        Path(out_file).write_text(output + '\n')
        print(f'Provenance written to {out_file}', file=sys.stderr)
    else:
        print(output)


if __name__ == '__main__':
    main()
