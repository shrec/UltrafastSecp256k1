#!/usr/bin/env python3
"""Canonical build configurator — all local builds go under out/<profile>.

Usage:
  python3 scripts/configure_build.py <profile> [extra cmake args...]
  python3 scripts/configure_build.py --list
  python3 scripts/configure_build.py --configure-only <profile>
  python3 scripts/configure_build.py --build <profile>

Profiles:
  debug      out/debug      — Debug with sanitizer info (-g -O0)
  release    out/release    — Optimized release (-O3, LTO)
  audit      out/audit      — Audit/ASAN build used by CI gates
  asan       out/asan       — AddressSanitizer
  tsan       out/tsan       — ThreadSanitizer
  msan       out/msan       — MemorySanitizer (clang only)
  wasm       out/wasm       — WebAssembly via Emscripten
  cuda       out/gpu-cuda   — CUDA GPU backend
  opencl     out/gpu-opencl — OpenCL GPU backend
  bench      out/bench      — Benchmark build (release + profiling)
  shim       out/shim       — libsecp256k1 shim only (Bitcoin Core profile)
  core       out/core       — Alias for shim (Bitcoin Core backend)

Migration note: old build dirs (build/, build-*, build_*) are safe to delete
after migrating to out/<profile>. Run scripts/clean_local_artifacts.sh to remove them.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

PROFILES: dict[str, dict] = {
    "debug": {
        "dir": "out/debug",
        "cmake_args": [
            "-DCMAKE_BUILD_TYPE=Debug",
            "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
        ],
        "description": "Debug build (-g -O0)",
    },
    "release": {
        "dir": "out/release",
        "cmake_args": [
            "-DCMAKE_BUILD_TYPE=Release",
            "-DENABLE_LTO=ON",
        ],
        "description": "Optimized release (-O3, LTO)",
    },
    "audit": {
        "dir": "out/audit",
        "cmake_args": [
            "-DCMAKE_BUILD_TYPE=RelWithDebInfo",
            "-DENABLE_AUDIT=ON",
            "-DENABLE_ASAN=ON",
            "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
        ],
        "description": "Audit/ASAN build (used by CI gates)",
    },
    "asan": {
        "dir": "out/asan",
        "cmake_args": [
            "-DCMAKE_BUILD_TYPE=Debug",
            "-DENABLE_ASAN=ON",
        ],
        "description": "AddressSanitizer build",
    },
    "tsan": {
        "dir": "out/tsan",
        "cmake_args": [
            "-DCMAKE_BUILD_TYPE=Debug",
            "-DENABLE_TSAN=ON",
        ],
        "description": "ThreadSanitizer build",
    },
    "msan": {
        "dir": "out/msan",
        "cmake_args": [
            "-DCMAKE_BUILD_TYPE=Debug",
            "-DENABLE_MSAN=ON",
            "-DCMAKE_C_COMPILER=clang",
            "-DCMAKE_CXX_COMPILER=clang++",
        ],
        "description": "MemorySanitizer build (clang only)",
    },
    "wasm": {
        "dir": "out/wasm",
        "cmake_args": [
            "-DCMAKE_BUILD_TYPE=Release",
            "-DENABLE_WASM=ON",
        ],
        "toolchain_hint": "emcmake cmake",
        "description": "WebAssembly via Emscripten",
    },
    "cuda": {
        "dir": "out/gpu-cuda",
        "cmake_args": [
            "-DCMAKE_BUILD_TYPE=Release",
            "-DENABLE_CUDA=ON",
        ],
        "description": "CUDA GPU backend",
    },
    "opencl": {
        "dir": "out/gpu-opencl",
        "cmake_args": [
            "-DCMAKE_BUILD_TYPE=Release",
            "-DENABLE_OPENCL=ON",
        ],
        "description": "OpenCL GPU backend",
    },
    "bench": {
        "dir": "out/bench",
        "cmake_args": [
            "-DCMAKE_BUILD_TYPE=Release",
            "-DENABLE_BENCHMARKS=ON",
            "-DENABLE_LTO=ON",
        ],
        "description": "Benchmark build (release + profiling)",
    },
    "shim": {
        "dir": "out/shim",
        "cmake_args": [
            "-DCMAKE_BUILD_TYPE=Release",
            "-DENABLE_LIBSECP256K1_SHIM=ON",
            "-DENABLE_AUDIT=ON",
        ],
        "description": "libsecp256k1 shim only (Bitcoin Core backend profile)",
    },
    "core": {
        "dir": "out/core",
        "cmake_args": [
            "-DCMAKE_BUILD_TYPE=Release",
            "-DENABLE_LIBSECP256K1_SHIM=ON",
            "-DENABLE_AUDIT=ON",
        ],
        "description": "Alias for 'shim' — Bitcoin Core backend profile",
    },
}


def list_profiles() -> None:
    print("Available build profiles:\n")
    for name, info in PROFILES.items():
        print(f"  {name:<12}  {info['dir']:<20}  {info['description']}")
    print()
    print("Usage: python3 scripts/configure_build.py <profile> [extra cmake args...]")


def configure(profile_name: str, extra_args: list[str], build: bool) -> int:
    if profile_name not in PROFILES:
        print(f"ERROR: unknown profile '{profile_name}'")
        print(f"Known profiles: {', '.join(PROFILES)}")
        return 1

    profile = PROFILES[profile_name]
    build_dir = REPO_ROOT / profile["dir"]
    build_dir.mkdir(parents=True, exist_ok=True)

    cmake_args = [
        "cmake",
        "-S", str(REPO_ROOT),
        "-B", str(build_dir),
    ] + profile["cmake_args"] + extra_args

    print(f"Profile  : {profile_name}")
    print(f"Build dir: {build_dir}")
    print(f"Command  : {' '.join(cmake_args)}\n")

    ret = subprocess.run(cmake_args, cwd=str(REPO_ROOT)).returncode
    if ret != 0:
        print(f"\nConfigure FAILED (exit {ret})")
        return ret

    print(f"\nConfigure OK — build dir: {build_dir}")

    if build:
        print("\nBuilding...")
        build_cmd = ["cmake", "--build", str(build_dir), "--parallel"]
        ret = subprocess.run(build_cmd, cwd=str(REPO_ROOT)).returncode
        if ret != 0:
            print(f"\nBuild FAILED (exit {ret})")
            return ret
        print("\nBuild OK")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("profile", nargs="?", help="Build profile name")
    parser.add_argument("extra", nargs="*", help="Extra CMake arguments")
    parser.add_argument("--list", action="store_true", help="List all profiles")
    parser.add_argument("--configure-only", metavar="PROFILE",
                        help="Configure only (no build)")
    parser.add_argument("--build", metavar="PROFILE",
                        help="Configure and build")

    args = parser.parse_args()

    if args.list:
        list_profiles()
        return 0

    if args.configure_only:
        return configure(args.configure_only, args.extra, build=False)

    if args.build:
        return configure(args.build, args.extra, build=True)

    if args.profile:
        return configure(args.profile, args.extra, build=False)

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
