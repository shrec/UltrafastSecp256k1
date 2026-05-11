#!/usr/bin/env python3
"""Impact-based CI profile detector.

The detector is the first block in the CAAS CI/CD flow.  It maps changed files
to product profiles so the PR/push gate can run the smallest safe set of
blocking checks, while release mode can still force the full suite.

Usage:
    python3 ci/ci_gate_detect.py
    python3 ci/ci_gate_detect.py --base origin/dev
    python3 ci/ci_gate_detect.py --files f1.cpp f2.cpp
    python3 ci/ci_gate_detect.py --github-output "$GITHUB_OUTPUT"

Exit codes preserve the historical contract:
    0 = light gate
    1 = hard gate
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
LIB_ROOT = SCRIPT_DIR.parent

PROFILE_PATTERNS: dict[str, list[str]] = {
    "core-engine": [
        "src/cpu/src/**",
        "src/cpu/include/**",
        "src/cpu/tests/**",
        "src/cpu/fuzz/**",
        "src/cpu/bench/**",
        "include/ufsecp/**",
        "CMakeLists.txt",
        "cmake/**",
    ],
    "bitcoin-core-backend": [
        "compat/libsecp256k1_shim/**",
        "docs/BITCOIN_CORE_*.md",
        "docs/*BITCOIN_CORE*.md",
    ],
    "node-compat-shims": [
        "compat/**",
    ],
    "ffi-bindings": [
        "bindings/**",
    ],
    "wasm": [
        "bindings/wasm/**",
        "wasm/**",
    ],
    "gpu-public-data": [
        "src/gpu/**",
        "src/cuda/**",
        "src/opencl/**",
        "src/metal/**",
    ],
    "audit": [
        "audit/**",
        "tools/source_graph_kit/**",
    ],
    "infra": [
        ".github/**",
        "ci/**",
        ".clusterfuzzlite/**",
        "codecov.yml",
    ],
    # Security-evidence files: audit catalog, changelog, evidence chain, and
    # KPI/benchmark JSON files whose mutation could inflate security scores.
    # These are NOT docs-only — changes here must trigger CAAS even with no
    # source changes (CI-003 fix: a commit deleting an exploit test from disk AND
    # from EXPLOIT_TEST_CATALOG.md previously bypassed all security gates).
    "security-evidence": [
        "docs/EXPLOIT_TEST_CATALOG.md",
        "docs/AUDIT_CHANGELOG.md",
        "docs/EVIDENCE_CHAIN.json",
        "docs/EXTERNAL_AUDIT_BUNDLE.json",
        "docs/SECURITY_AUTONOMY_KPI.json",
        "docs/BITCOIN_CORE_TEST_RESULTS.json",
        "docs/BITCOIN_CORE_BENCH_RESULTS.json",
        # CI-004: benchmark evidence artifacts — mutation of these could inflate
        # performance claims without triggering CAAS gates.
        "docs/canonical_numbers.json",
        "docs/bench_unified_*.json",
    ],
    "docs-only": [
        "docs/**",
        "README.md",
        "CHANGELOG.md",
        "SECURITY.md",
        "CONTRIBUTING.md",
        "LICENSE",
    ],
    "packaging": [
        "packaging/**",
        "conanfile.py",
        "Package.swift",
    ],
    "examples": [
        "examples/**",
    ],
}

HARD_PROFILES = {
    "core-engine",
    "bitcoin-core-backend",
    "node-compat-shims",
    "gpu-public-data",
    "audit",
    "infra",
    "security-evidence",
}

CT_SENSITIVE_PATTERNS = [
    "src/cpu/src/ct_*",
    "src/cpu/src/ecdsa*",
    "src/cpu/src/schnorr*",
    "src/cpu/src/recovery*",
    "src/cpu/src/scalar*",
    "src/cpu/src/field*",
    "src/cpu/src/musig*",
    "src/cpu/src/frost*",
    "src/cpu/src/ecdh*",
    "src/cpu/src/ecies*",
    "src/cpu/include/secp256k1/ct/**",
    "src/cpu/include/secp256k1/detail/secure_erase*",
    "src/cpu/include/secp256k1/detail/value_barrier*",
    "include/ufsecp/**",
    "compat/libsecp256k1_shim/**",
]


def get_changed_files_from_sha(before_sha: str) -> list[str]:
    """Get files changed since before_sha (push event with known before-SHA)."""
    try:
        result = subprocess.run(
            ['git', 'diff', '--name-only', before_sha, 'HEAD'],
            capture_output=True, text=True, cwd=str(LIB_ROOT),
        )
        if result.returncode == 0:
            return [f.strip() for f in result.stdout.strip().splitlines() if f.strip()]
    except Exception:
        pass
    return []


def get_changed_files(base: str = 'origin/dev') -> list[str]:
    """Get list of changed files relative to base."""
    try:
        merge_base = subprocess.run(
            ['git', 'merge-base', 'HEAD', base],
            capture_output=True, text=True, cwd=str(LIB_ROOT),
        )
        if merge_base.returncode != 0:
            merge_base_sha = 'HEAD~1'
        else:
            merge_base_sha = merge_base.stdout.strip()
            # On push events, origin/dev is updated before CI runs, so
            # merge-base HEAD origin/dev == HEAD → git diff HEAD HEAD is empty.
            # Detect this and fall back to HEAD~1 so the last commit's files
            # are classified correctly. For multi-commit pushes, callers should
            # use get_changed_files_from_sha() with github.event.before instead.
            head = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True, text=True, cwd=str(LIB_ROOT),
            )
            if head.returncode == 0 and merge_base_sha == head.stdout.strip():
                merge_base_sha = 'HEAD~1'

        result = subprocess.run(
            ['git', 'diff', '--name-only', merge_base_sha, 'HEAD'],
            capture_output=True, text=True, cwd=str(LIB_ROOT),
        )
        if result.returncode == 0:
            return [f.strip() for f in result.stdout.strip().splitlines() if f.strip()]
    except Exception:
        pass
    return []


def matches(path: str, patterns: list[str]) -> bool:
    """Return whether a path matches any profile glob."""
    return any(fnmatch.fnmatch(path, pattern) for pattern in patterns)


def classify_file(path: str) -> set[str]:
    """Classify a file path into one or more CI profiles."""
    profiles = {
        profile
        for profile, patterns in PROFILE_PATTERNS.items()
        if matches(path, patterns)
    }
    if "compat/libsecp256k1_shim/" in path:
        profiles.add("bitcoin-core-backend")
        profiles.discard("node-compat-shims")
    if "bindings/wasm/" in path or path.startswith("wasm/"):
        profiles.add("wasm")
        profiles.discard("ffi-bindings")
    return profiles or {"unknown"}


def detect_gate_level(files: list[str], force_release: bool = False) -> dict:
    """Determine gate level and profile set from a list of changed files."""
    profile_hits: dict[str, list[str]] = {}
    ct_sensitive_files: list[str] = []

    for f in files:
        profiles = classify_file(f)
        for profile in profiles:
            profile_hits.setdefault(profile, []).append(f)
        if matches(f, CT_SENSITIVE_PATTERNS):
            ct_sensitive_files.append(f)

    profiles = sorted(profile_hits)
    unknown_files = profile_hits.get("unknown", [])
    hard_profiles = sorted((set(profiles) & HARD_PROFILES) | ({"unknown"} if unknown_files else set()))
    docs_only = bool(profiles) and set(profiles) <= {"docs-only", "examples"}
    light_only = bool(profiles) and not hard_profiles and not ct_sensitive_files
    gate = "release" if force_release else ("light" if light_only or docs_only or not files else "hard")

    run_core = gate in {"hard", "release"} or any(p in profiles for p in ("core-engine", "bitcoin-core-backend"))
    # "audit" profile is explicitly listed here so that commits touching only
    # audit/**  (new exploit PoC, unified_audit_runner.cpp changes, etc.) always
    # trigger the caas-security job.  Without this, an unwired or broken exploit
    # test could slip through on a docs_only-adjacent gate classification.
    run_caas = (
        gate in {"hard", "release"}
        or any(p in profiles for p in ("audit", "infra", "bitcoin-core-backend", "security-evidence"))
        or any(f.startswith("audit/") for f in files)  # belt-and-suspenders guard
    )
    run_bindings = gate == "release" or any(p in profiles for p in ("ffi-bindings", "wasm"))
    run_wasm = gate == "release" or "wasm" in profiles
    run_gpu = gate == "release" or "gpu-public-data" in profiles
    run_compat = gate == "release" or any(p in profiles for p in ("bitcoin-core-backend", "node-compat-shims"))
    run_deep = gate == "release" or bool(ct_sensitive_files)

    if force_release:
        reason = "release gate requested; all release profiles selected"
    elif ct_sensitive_files:
        reason = f"{len(ct_sensitive_files)} CT/crypto/ABI-sensitive file(s) changed"
    elif hard_profiles:
        reason = f"hard profile(s) changed: {', '.join(hard_profiles)}"
    elif docs_only:
        reason = "documentation/example-only change"
    elif files:
        reason = f"light profile(s) changed: {', '.join(profiles)}"
    else:
        reason = "no changed files detected"

    return {
        "gate": gate,
        "profiles": profiles,
        "profile_hits": profile_hits,
        "changed_files": len(files),
        "ct_sensitive_files": ct_sensitive_files,
        "unknown_files": unknown_files,
        "hard_profiles": hard_profiles,
        "docs_only": docs_only,
        "run_core": run_core,
        "run_caas": run_caas,
        "run_bindings": run_bindings,
        "run_wasm": run_wasm,
        "run_gpu": run_gpu,
        "run_compat": run_compat,
        "run_deep": run_deep,
        "reason": reason,
    }


def write_github_outputs(path: str, result: dict) -> None:
    """Write detector outputs for GitHub Actions jobs."""
    def as_bool(value: object) -> str:
        return "true" if bool(value) else "false"

    outputs = {
        "gate": result["gate"],
        "profiles": ",".join(result["profiles"]),
        "reason": result["reason"],
        "changed_files": str(result["changed_files"]),
        "docs_only": as_bool(result["docs_only"]),
        "run_core": as_bool(result["run_core"]),
        "run_caas": as_bool(result["run_caas"]),
        "run_bindings": as_bool(result["run_bindings"]),
        "run_wasm": as_bool(result["run_wasm"]),
        "run_gpu": as_bool(result["run_gpu"]),
        "run_compat": as_bool(result["run_compat"]),
        "run_deep": as_bool(result["run_deep"]),
    }
    with open(path, "a", encoding="utf-8") as fh:
        for key, value in outputs.items():
            fh.write(f"{key}={value}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description='CI gate level detection')
    parser.add_argument('--base', default='origin/dev', help='Git base ref')
    parser.add_argument('--files', nargs='*', help='Explicit file list (skip git)')
    parser.add_argument('--release', action='store_true', help='Force release profile selection')
    parser.add_argument('--github-output', help='Write GitHub Actions outputs to this file')
    parser.add_argument(
        '--before-sha', default='',
        help='SHA before the push (github.event.before). When provided and non-zero, '
             'used as the diff base instead of merge-base detection, correctly handling '
             'multi-commit pushes.',
    )
    args = parser.parse_args()

    if args.files:
        files = args.files
    elif args.before_sha and len(args.before_sha) == 40 and not all(c == '0' for c in args.before_sha):
        # Push event with a known before-SHA: diff from that SHA to HEAD captures
        # all commits in the push, not just the last one (which HEAD~1 would give).
        files = get_changed_files_from_sha(args.before_sha)
        if not files:
            # before-sha diff empty → new branch or shallow clone, fall back
            files = get_changed_files(args.base)
    else:
        files = get_changed_files(args.base)

    result = detect_gate_level(files, force_release=args.release)

    print(json.dumps(result, indent=2))
    if args.github_output:
        write_github_outputs(args.github_output, result)
    return 1 if result['gate'] in {'hard', 'release'} else 0


if __name__ == '__main__':
    sys.exit(main())
