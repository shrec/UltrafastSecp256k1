#!/usr/bin/env python3
"""Release package content guard.

Desktop release archives must ship only product libraries in lib/static and
lib/shared. Build-tree test, audit, exploit, fuzz, benchmark, or standalone
target libraries are valid CI artifacts, but they must never be copied into a
consumer package.

Usage:
  python3 ci/check_release_package_contents.py <package-dir-or-archive>...
  python3 ci/check_release_package_contents.py --json <package-dir-or-archive>...

Second, independent mode -- GPU CMake export-set closure probe (issue #335
acceptance repair, round 10; see the "GPU CMake export-set closure probe"
section below for the full design rationale):

  python3 ci/check_release_package_contents.py --gpu-export-closure
  python3 ci/check_release_package_contents.py --gpu-export-closure --json
  python3 ci/check_release_package_contents.py --gpu-export-closure --combo opencl

This second mode shares this file (not a separate script) because the round-10
task card that requested it named this exact filename, not realizing an
unrelated, already-wired, already-documented gate (the package-content scan
above -- live in `.github/workflows/release.yml`, `docs/RELEASE_PROCESS.md`,
`docs/PACKAGE_PROVENANCE.md`, `docs/AUDIT_CHANGELOG.md` since commit
55cbc745, "ci: guard release package contents") already owned this name. The
two checks are unrelated (library-content scanning of a finished release
archive vs. a live `cmake` configure probe) and are kept behind fully
separate CLI surfaces (`--gpu-export-closure` opts into the second mode; the
positional `packages` legacy scan is completely unaffected and remains
byte-for-byte backward compatible with every existing caller) rather than
merged into one code path.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import tarfile
import tempfile
import time
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


# ============================================================================
# GPU CMake export-set closure probe (issue #335 acceptance repair, round 10)
# ============================================================================
#
# History (issue #335): `SECP256K1_INSTALL_CABI=ON` combined with any GPU
# backend (`SECP256K1_BUILD_OPENCL`/`CUDA`/`METAL`) failed to configure from
# round 9 through round 10 -- `ufsecp_static` PRIVATE-links
# `secp256k1_gpu_host` whenever a GPU backend is on, and CMake's
# `install(EXPORT ufsecpTargets ...)` refused to generate because
# `secp256k1_gpu_host` (and, transitively, `secp256k1_opencl`/
# `secp256k1_cuda_lib`) was not itself in any export set -- a hard
# configure-time error, not a build or link error. Round 10 disclosed this
# but, after mistakenly treating the required CMakeLists.txt files as out of
# scope, never fixed it; that round was rejected for exactly this reason.
# Round 11 fixed it for real: `src/gpu|opencl|cuda/CMakeLists.txt` now join
# `secp256k1_gpu_host`/`secp256k1_opencl`/`secp256k1_cuda_lib` to the real
# `ufsecpTargets` export set, and `src/metal/CMakeLists.txt` got the same
# structural treatment for parity (Metal itself remains unverifiable on a
# non-Apple host). Round 12 (this round) removed the waiver below that had
# let this exact failure signature pass the gate as a "known, disclosed,
# non-blocking gap" -- that classification was a deliberate fail-before/
# pass-after fixture while the bug was still open; now that it is fixed,
# leaving the waiver in place would silently hide a REGRESSION back to the
# old broken state. There is no more "expected" failure class for this gate:
# any configure/build/install failure for an available backend is a
# genuine, blocking regression.
#
# This gate runs a REAL `cmake -S -B` configure (root CMakeLists.txt, out-of-
# tree scratch dir, never inside the repo) for five backend combinations and
# classifies the real, captured result. It is a fail-closed fixture (this
# repo's Bug-to-CAAS convention), not a rubber stamp: the pass criterion for
# a GPU combo is "configure succeeds AND `cmake --build --target install`
# succeeds AND the installed tree actually contains the GPU C ABI header
# (`ufsecp_gpu.h`, only installed by root CMakeLists.txt when `TARGET
# secp256k1_gpu_host` genuinely exists)" -- not merely "configure did not
# error". That distinction is not academic: this gate's own probing found
# that `-DSECP256K1_BUILD_METAL=ON` on a non-Apple host configures with exit
# 0 (`GPU API: no backends available -- skipping gpu/ build`, verified live
# on this Linux host) while never building any GPU code at all -- a naive
# "configure succeeded" check would have reported a fabricated PASS for a
# combo that tested nothing. Toolchain availability is therefore checked per
# backend BEFORE attempting a real configure, and unavailable backends (no
# CUDA toolkit, non-Apple host for Metal) are reported as an explicit,
# labeled ADVISORY_SKIP -- never silently folded into PASS or FAIL.
#
# Overall gate exit status is 0 only if every combo is a genuine PASS or an
# honest toolchain-unavailable ADVISORY_SKIP -- there is no other passing
# status. What fails the gate (all of these are blocking, unconditionally): the
# CPU-only positive control breaking, ANY GPU combo failing to configure for
# any reason (including a reintroduction of the old export-closure
# signature above -- there is deliberately no special-cased exemption for
# it any more), a build/install failure after a successful configure, or a
# GPU combo reporting configure/build success while the installed tree is
# missing the artifacts it should contain (the fabricated-pass class
# described above).

SCRIPT_DIR = Path(__file__).resolve().parent
LIB_ROOT = SCRIPT_DIR.parent

GPU_COMBOS = [
    {
        "key": "cpu_only",
        "label": "CPU-only (positive control: SECP256K1_INSTALL_CABI=ON, no GPU backend)",
        "flags": {},
        "toolchain": None,
        "expect_gpu_header": False,
    },
    {
        "key": "opencl",
        "label": "OpenCL-only",
        "flags": {"SECP256K1_BUILD_OPENCL": "ON"},
        "toolchain": "opencl",
        "expect_gpu_header": True,
    },
    {
        "key": "cuda",
        "label": "CUDA-only",
        "flags": {"SECP256K1_BUILD_CUDA": "ON", "SECP256K1_CUDA_ARCH_PROFILE": "local-native"},
        "toolchain": "cuda",
        "expect_gpu_header": True,
    },
    {
        "key": "metal",
        "label": "Metal-only",
        "flags": {"SECP256K1_BUILD_METAL": "ON"},
        "toolchain": "metal",
        "expect_gpu_header": True,
    },
    {
        "key": "opencl_cuda",
        "label": "OpenCL+CUDA combined",
        "flags": {
            "SECP256K1_BUILD_OPENCL": "ON",
            "SECP256K1_BUILD_CUDA": "ON",
            "SECP256K1_CUDA_ARCH_PROFILE": "local-native",
        },
        "toolchain": "opencl+cuda",
        "expect_gpu_header": True,
    },
]


def _detect_local_gpu_compute_capability() -> str | None:
    """Query the actual local GPU's compute capability LIVE via nvidia-smi
    (e.g. '12.0'). Returns None if nvidia-smi is unavailable, fails, or no
    device is present -- never guessed or hardcoded. This is what lets
    detect_cuda_toolchain() verify a candidate nvcc against the REAL local
    device instead of trusting a version-number table."""
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return None
    try:
        proc = subprocess.run(
            [nvidia_smi, "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    return proc.stdout.strip().splitlines()[0].strip()


def _nvcc_candidates() -> list[Path]:
    """Every nvcc this machine might plausibly have, without hardcoding one
    specific version or path (issue #335 acceptance repair, round 11
    coordinator follow-up): an explicit override env var (mirrors
    ci/check_cuda_arch_policy.py's UFSECP_CI_NVCC12/UFSECP_CI_NVCC13
    convention -- lets a CI preflight step feed this gate the exact
    compiler it already resolved and validated), whatever `nvcc` resolves
    on PATH, and every `/usr/local/cuda-*/bin/nvcc` /
    `/opt/cuda-*/bin/nvcc` glob match (sorted so a numerically newer
    toolkit directory name is tried first -- a newer CUDA toolkit
    generally supports newer GPU architectures; this machine, for
    instance, has both /usr/bin/nvcc from a `cuda-12.0` package on PATH
    and /usr/local/cuda-13.2/bin/nvcc, per
    cmake/CUDA_ARCHITECTURE_POLICY.md's "Implementation-session
    verification" section). Deduplicated by resolved real path so the same
    physical binary is never probed twice."""
    candidates: list[Path] = []
    override = os.environ.get("UFSECP_GPU_EXPORT_CLOSURE_NVCC")
    if override:
        candidates.append(Path(override))
    on_path = shutil.which("nvcc")
    if on_path:
        candidates.append(Path(on_path))
    for root in ("/usr/local", "/opt"):
        try:
            matches = sorted(Path(root).glob("cuda-*/bin/nvcc"), reverse=True)
        except OSError:
            matches = []
        candidates.extend(matches)
    seen: set[str] = set()
    deduped: list[Path] = []
    for c in candidates:
        try:
            real = str(c.resolve(strict=False))
        except OSError:
            real = str(c)
        if real in seen:
            continue
        seen.add(real)
        deduped.append(c)
    return deduped


def _nvcc_can_target_arch(nvcc: Path, arch_digits: str, tmp_dir: Path) -> bool:
    """Real, direct probe -- not a version-table guess: actually invoke
    THIS exact nvcc binary to compile a trivial __global__ kernel for
    -arch=compute_<arch_digits>. Mirrors this repo's benchmark policy
    ("never write a number/verdict you have not actually measured") applied
    to toolchain capability instead of performance."""
    if not (nvcc.is_file() and os.access(nvcc, os.X_OK)):
        return False
    probe_src = tmp_dir / f"nvcc_arch_probe_{arch_digits}.cu"
    probe_obj = tmp_dir / f"nvcc_arch_probe_{arch_digits}.o"
    try:
        probe_src.write_text("__global__ void ufsecp_nvcc_arch_probe_kernel() {}\n")
        proc = subprocess.run(
            [str(nvcc), f"-arch=compute_{arch_digits}", "-c", str(probe_src), "-o", str(probe_obj)],
            capture_output=True, text=True, timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0


def detect_cuda_toolchain() -> dict:
    """Real check -- does NOT stop at "is any nvcc on PATH" (issue #335
    acceptance repair, round 11 coordinator follow-up). This machine has
    MULTIPLE installed CUDA toolkits: the default `nvcc` on PATH (12.0,
    predates this GPU's Blackwell/compute_120 architecture -- confirmed
    live by directly invoking it with `-arch=compute_120` and observing
    `nvcc fatal: Unsupported gpu architecture`) and
    `/usr/local/cuda-13.2/bin/nvcc` (13.2, already verified elsewhere in
    this repo -- cmake/CUDA_ARCHITECTURE_POLICY.md's
    "Implementation-session verification" section -- to target this exact
    GPU/architecture). The OLD PATH-only check silently resolved the wrong
    (too-old) one: toolchain detection reported "available", the combo's
    `cmake` CONFIGURE step succeeded (SECP256K1_CUDA_ARCH_PROFILE=local-native
    just sets CMAKE_CUDA_ARCHITECTURES=native, a configure-time no-op that
    defers the real device/arch match to the first actual nvcc invocation),
    and only the real `cmake --build` compile step failed, much later and
    far more confusingly, with a raw nvcc diagnostic.

    This instead probes every nvcc candidate (_nvcc_candidates(), general --
    not hardcoded to this machine's specific paths/versions) and picks the
    first one verified, via a REAL trivial-kernel compile
    (_nvcc_can_target_arch(), not a version-number table), to actually
    target the LOCAL GPU's real compute capability (queried live via
    nvidia-smi in _detect_local_gpu_compute_capability() -- never assumed).
    If nvidia-smi is unavailable (no NVIDIA GPU present, e.g. a headless or
    non-NVIDIA host), this falls back to the old "first nvcc found"
    behavior, since there is then no local device to verify a match
    against -- honestly labeled as such, not silently claimed as verified.
    Absence of any nvcc that can target the detected device is reported as
    an honest, labeled ADVISORY_SKIP -- never silently treated as pass or
    fail, and never worked around by picking an arch/profile combination
    that avoids exercising the actual local hardware."""
    with tempfile.TemporaryDirectory(prefix="ufsecp_nvcc_probe_") as tmp:
        tmp_path = Path(tmp)
        candidates = _nvcc_candidates()
        if not candidates:
            return {
                "available": False,
                "reason": "no nvcc found (checked PATH, $UFSECP_GPU_EXPORT_CLOSURE_NVCC, "
                          "/usr/local/cuda-*/bin/nvcc, /opt/cuda-*/bin/nvcc) -- no CUDA "
                          "toolkit installed",
                "nvcc_path": None,
            }
        compute_cap = _detect_local_gpu_compute_capability()
        if compute_cap is None:
            nvcc = candidates[0]
            return {
                "available": True,
                "reason": f"nvcc found at {nvcc} (nvidia-smi unavailable or reported no "
                          f"device -- could not verify this candidate against a specific "
                          f"local GPU architecture; falling back to the first resolved "
                          f"candidate, matching this check's pre-round-11 behavior)",
                "nvcc_path": str(nvcc),
            }
        arch_digits = compute_cap.replace(".", "")  # e.g. "12.0" -> "120"
        tried: list[str] = []
        for nvcc in candidates:
            if _nvcc_can_target_arch(nvcc, arch_digits, tmp_path):
                return {
                    "available": True,
                    "reason": f"nvcc at {nvcc} verified live (real -arch=compute_{arch_digits} "
                              f"trivial-kernel compile, not a version-number guess) to target "
                              f"this machine's actual GPU (compute capability {compute_cap}, "
                              f"queried via nvidia-smi)",
                    "nvcc_path": str(nvcc),
                }
            tried.append(str(nvcc))
        return {
            "available": False,
            "reason": f"this machine's local GPU has compute capability {compute_cap} "
                      f"(queried via nvidia-smi) but none of the {len(tried)} nvcc "
                      f"candidate(s) tried ({tried}) can compile -arch=compute_{arch_digits} "
                      f"(each was actually invoked on a trivial kernel, not guessed from a "
                      f"version table) -- no usable CUDA toolchain for this GPU on this host",
            "nvcc_path": None,
        }


def detect_metal_toolchain() -> dict:
    """Metal is an Apple-only GPU framework. On any non-Darwin host this is a
    genuine, permanent cross-compile-unavailable case (not a transient
    missing-tool situation) -- reported as an explicit skip so a Linux CI
    runner can never be mistaken for having exercised Metal."""
    system = platform.system()
    if system != "Darwin":
        return {
            "available": False,
            "reason": (
                f"not running on macOS/Darwin (platform.system()={system!r}) -- Metal "
                f"is an Apple-only GPU framework; verified live that "
                f"-DSECP256K1_BUILD_METAL=ON on this host configures with exit 0 but "
                f"never builds any GPU code (src/metal/CMakeLists.txt returns early "
                f"with 'GPU backend skipped (not Apple)'), so attempting a configure "
                f"here would prove nothing and risks a fabricated pass"
            ),
        }
    return {"available": True, "reason": "running on Darwin (macOS) -- Metal framework probed via cmake"}


def detect_opencl_toolchain() -> dict:
    """OpenCL needs a loadable ICD library for CMake's find_package(OpenCL)
    to succeed. Checked via ctypes.util.find_library with a small fallback
    list of common Linux install paths (find_library can miss versioned-only
    .so files on some distros)."""
    import ctypes.util

    lib = ctypes.util.find_library("OpenCL")
    if lib:
        return {"available": True, "reason": f"OpenCL ICD library found via find_library ({lib})"}
    known_paths = (
        "/usr/lib/x86_64-linux-gnu/libOpenCL.so",
        "/usr/lib/x86_64-linux-gnu/libOpenCL.so.1",
        "/usr/lib/libOpenCL.so",
        "/usr/local/cuda/targets/x86_64-linux/lib/libOpenCL.so.1",
    )
    for p in known_paths:
        if Path(p).exists():
            return {"available": True, "reason": f"OpenCL ICD library found at {p}"}
    return {
        "available": False,
        "reason": "no libOpenCL found via ctypes.util.find_library or common install paths",
    }


def detect_toolchain(name: str) -> dict:
    if name == "opencl+cuda":
        o, c = detect_opencl_toolchain(), detect_cuda_toolchain()
        if o["available"] and c["available"]:
            return {
                "available": True,
                "reason": f"OpenCL ({o['reason']}) and CUDA ({c['reason']}) both available",
                # Propagate the specific nvcc detect_cuda_toolchain() verified against this
                # machine's real GPU, so evaluate_combo() can pin it for the combined combo
                # too (not just the CUDA-only one).
                "nvcc_path": c.get("nvcc_path"),
            }
        missing = []
        if not o["available"]:
            missing.append(f"OpenCL unavailable: {o['reason']}")
        if not c["available"]:
            missing.append(f"CUDA unavailable: {c['reason']}")
        return {"available": False, "reason": "; ".join(missing), "nvcc_path": None}
    detectors = {"cuda": detect_cuda_toolchain, "metal": detect_metal_toolchain, "opencl": detect_opencl_toolchain}
    fn = detectors.get(name)
    if fn is None:
        return {"available": True, "reason": "no toolchain gate registered for this combo", "nvcc_path": None}
    return fn()


def classify_configure_failure(combined_output: str) -> tuple[str, str]:
    """Pure classifier over a failed configure's captured stdout+stderr.
    Returns (kind, evidence) where evidence is ALWAYS a literal substring of
    the real captured output -- never a synthesized description. Exercised
    directly (no subprocess) by the ci/test_audit_scripts.py self-test.

    Round 12 (issue #335): this used to special-case the exact round-9/10
    GPU export-closure error signature as a non-blocking "known gap" so the
    gate could pass before the underlying CMake bug was fixed. That bug is
    fixed (round 11); this classifier no longer has, or needs, an "expected"
    failure kind -- ANY configure failure is unexpected and blocking,
    including a regression back to the exact old signature."""
    tail = combined_output.strip()
    return "unexpected_configure_failure", tail[-1500:]


def run_cmake_configure(build_dir: Path, install_prefix: Path, extra_flags: dict,
                         timeout_s: int = 180) -> dict:
    """Real `cmake -S <repo-root> -B <scratch build_dir>` invocation. Never
    touches the repo tree (build_dir/install_prefix are always caller-owned
    scratch paths outside it). UFSECP_BUILD_SHARED=OFF keeps the probe to the
    static CABI library only -- verified live to reproduce the identical
    known error text as the shared-enabled default, and roughly halves the
    later build+install time on the combos that reach it."""
    cmd = [
        "cmake", "-S", str(LIB_ROOT), "-B", str(build_dir),
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DCMAKE_INSTALL_PREFIX={install_prefix}",
        "-DSECP256K1_INSTALL_CABI=ON",
        "-DUFSECP_BUILD_SHARED=OFF",
        "-DSECP256K1_BUILD_TESTS=OFF",
        "-DSECP256K1_BUILD_BENCH=OFF",
        "-DSECP256K1_BUILD_EXAMPLES=OFF",
        "-DSECP256K1_BUILD_JAVA=OFF",
    ]
    for k, v in sorted(extra_flags.items()):
        cmd.append(f"-D{k}={v}")
    t0 = time.monotonic()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"returncode": None, "combined": f"cmake invocation failed to run: {exc}",
                "elapsed_s": time.monotonic() - t0, "cmd": cmd}
    return {
        "returncode": proc.returncode,
        "combined": (proc.stdout or "") + "\n" + (proc.stderr or ""),
        "elapsed_s": time.monotonic() - t0,
        "cmd": cmd,
    }


def run_build_install(build_dir: Path, timeout_s: int = 600) -> dict:
    """Real `cmake --build --target install`. Only ever invoked after a
    successful configure -- this is what actually produces the installed
    package tree that check_install_tree_contents() then inspects."""
    cmd = ["cmake", "--build", str(build_dir), "--target", "install", "-j", str(os.cpu_count() or 2)]
    t0 = time.monotonic()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"returncode": None, "combined": f"build+install invocation failed: {exc}",
                "elapsed_s": time.monotonic() - t0}
    return {
        "returncode": proc.returncode,
        "combined": (proc.stdout or "") + "\n" + (proc.stderr or ""),
        "elapsed_s": time.monotonic() - t0,
    }


def check_install_tree_contents(install_prefix: Path, expect_gpu_header: bool) -> dict:
    """Genuine content-correctness check, not a configure-succeeded rubber
    stamp: the installed tree must contain the CABI static library, the base
    ufsecp.h header, and the exported ufsecpTargets.cmake package config --
    and, for a GPU-backend combo, ufsecp_gpu.h too (root CMakeLists.txt only
    installs it `if(TARGET secp256k1_gpu_host)`, i.e. only when the GPU host
    layer genuinely got built). This is what catches the Metal-class
    fabricated-pass scenario documented above: a GPU flag can be ON and
    configure can still exit 0 while the GPU host layer was silently never
    built."""
    required = {
        "static_lib": install_prefix / "lib" / "libufsecp.a",
        "cabi_header": install_prefix / "include" / "ufsecp" / "ufsecp.h",
        "cmake_export": install_prefix / "lib" / "cmake" / "ufsecp" / "ufsecpTargets.cmake",
    }
    missing = [str(p) for p in required.values() if not p.is_file()]
    gpu_header = install_prefix / "include" / "ufsecp" / "ufsecp_gpu.h"
    gpu_header_present = gpu_header.is_file()
    if expect_gpu_header and not gpu_header_present:
        missing.append(
            f"{gpu_header} (GPU C ABI header expected for this backend combo -- its "
            f"absence despite a successful configure+build+install means the GPU host "
            f"layer was not genuinely built; success alone is not sufficient evidence)"
        )
    return {
        "ok": not missing,
        "missing": missing,
        "gpu_header_present": gpu_header_present,
        "checked_paths": {k: str(v) for k, v in required.items()},
    }


def evaluate_combo(combo: dict, scratch_base: Path) -> dict:
    """Run one backend combo end-to-end and classify the real result. Never
    raises on ordinary cmake/build failures -- those are captured outcomes,
    not script errors."""
    result: dict = {"key": combo["key"], "label": combo["label"]}

    # Copy, never mutate GPU_COMBOS itself -- a resolved compiler pin
    # (below) is specific to this run's real hardware, not a permanent
    # property of the combo definition.
    combo_flags = dict(combo["flags"])

    if combo["toolchain"] is not None:
        detection = detect_toolchain(combo["toolchain"])
        result["toolchain_detection"] = detection
        if not detection["available"]:
            result["status"] = "ADVISORY_SKIP"
            result["reason"] = detection["reason"]
            return result
        # Issue #335 acceptance repair (round 11 coordinator follow-up): pin
        # the exact nvcc detect_cuda_toolchain() verified live against this
        # machine's real GPU (see that function's docstring) -- never let
        # CMake's own unpinned PATH resolution silently pick a different,
        # possibly-too-old nvcc than the one this gate already confirmed
        # can actually target the local device.
        nvcc_path = detection.get("nvcc_path")
        if nvcc_path:
            combo_flags["CMAKE_CUDA_COMPILER"] = nvcc_path

    combo_dir = scratch_base / combo["key"]
    build_dir = combo_dir / "build"
    install_prefix = combo_dir / "install"

    configure = run_cmake_configure(build_dir, install_prefix, combo_flags)
    result["configure_returncode"] = configure["returncode"]
    result["configure_elapsed_s"] = round(configure["elapsed_s"], 2)

    if configure["returncode"] != 0:
        kind, evidence = classify_configure_failure(configure["combined"])
        result["configure_failure_kind"] = kind
        result["evidence"] = evidence.strip()
        # Round 12 (issue #335): every configure failure is blocking now,
        # with no non-blocking "known gap" exemption for any combo --
        # including a regression back to the old export-closure signature
        # (round 9/10) that this gate used to wave through deliberately
        # while that bug was still open. It is fixed (round 11); this gate
        # must catch it coming back, not shrug it off again.
        result["status"] = "UNEXPECTED_FAIL"
        if combo["key"] == "cpu_only":
            result["message"] = (
                "CPU-only SECP256K1_INSTALL_CABI=ON configure FAILED. This is the "
                "positive control and is expected to always succeed."
            )
        else:
            result["message"] = (
                f"{combo['label']} + SECP256K1_INSTALL_CABI=ON failed at configure "
                f"time. The GPU CMake export-set closure was fixed in round 11 "
                f"(src/gpu|opencl|cuda|metal/CMakeLists.txt) -- this combo is expected "
                f"to configure cleanly; any failure here, including the old "
                f"'ufsecpTargets ... secp256k1_gpu_host ... not in any export set' "
                f"signature, is a regression requiring investigation, not a known gap."
            )
        return result

    # Configure succeeded -- prove it with a real build+install, then inspect
    # the installed tree. "Configure succeeded" alone is not the pass bar.
    build = run_build_install(build_dir)
    result["build_install_returncode"] = build["returncode"]
    result["build_install_elapsed_s"] = round(build["elapsed_s"], 2)
    if build["returncode"] != 0:
        result["status"] = "UNEXPECTED_FAIL"
        result["evidence"] = build["combined"].strip()[-1500:]
        result["message"] = (
            f"{combo['label']}: configure succeeded but `cmake --build --target "
            f"install` FAILED."
        )
        return result

    content = check_install_tree_contents(install_prefix, combo["expect_gpu_header"])
    result["content_check"] = content
    if content["ok"]:
        result["status"] = "PASS"
        result["message"] = (
            f"{combo['label']}: configure + build + install all succeeded and the "
            f"installed package tree contains the expected artifacts"
            + (" including the GPU C ABI header (ufsecp_gpu.h)." if combo["expect_gpu_header"] else ".")
        )
    else:
        result["status"] = "UNEXPECTED_FAIL"
        result["message"] = (
            f"{combo['label']}: configure + build + install all reported success, "
            f"but the installed tree is missing expected artifact(s) -- a bare "
            f"'configure succeeded' check would have wrongly reported PASS here: "
            f"{content['missing']}"
        )
    return result


def evaluate_gpu_export_closure(only: str | None = None, scratch_dir: Path | None = None) -> dict:
    """Orchestrate the full sweep. `only` restricts to one combo key (for
    faster iteration / debugging); default None/"all" runs every combo.
    `scratch_dir`, if given, is used instead of an auto-cleaned temp dir
    (caller owns cleanup) -- used by tests and manual debugging."""
    combos = GPU_COMBOS if only in (None, "all") else [c for c in GPU_COMBOS if c["key"] == only]
    if not combos:
        return {"overall_pass": False, "error": f"unknown combo key {only!r}", "combos": []}

    owns_scratch = scratch_dir is None
    scratch_ctx = tempfile.TemporaryDirectory(prefix="ufsecp_gpu_export_closure_") if owns_scratch else None
    scratch_base = Path(scratch_ctx.name) if owns_scratch else scratch_dir
    try:
        results = [evaluate_combo(c, scratch_base) for c in combos]
    finally:
        if scratch_ctx is not None:
            scratch_ctx.cleanup()

    blocking = [r["key"] for r in results if r["status"] == "UNEXPECTED_FAIL"]
    return {
        "overall_pass": not blocking,
        "blocking_combos": blocking,
        "combos": results,
        "total_elapsed_s": round(
            sum(r.get("configure_elapsed_s", 0) + r.get("build_install_elapsed_s", 0) for r in results), 2
        ),
    }


def print_gpu_export_closure_report(report: dict) -> None:
    if "error" in report:
        print(f"ERROR: {report['error']}")
        return
    print("GPU CMake export-set closure probe (issue #335 acceptance repair)")
    print("=" * 72)
    status_tag = {
        "PASS": "PASS ",
        "ADVISORY_SKIP": "SKIP ",
        "UNEXPECTED_FAIL": "FAIL!",
    }
    for r in report["combos"]:
        tag = status_tag.get(r["status"], r["status"])
        print(f"[{tag}] {r['key']:12s} {r['label']}")
        if r["status"] == "ADVISORY_SKIP":
            print(f"          reason: {r['reason']}")
        elif "message" in r:
            print(f"          {r['message']}")
            if r.get("evidence"):
                snippet = r["evidence"][:400].replace("\n", "\n          ")
                print(f"          evidence: {snippet}")
    print("-" * 72)
    print(f"Total real cmake wall time  : {report.get('total_elapsed_s', 0)}s")
    print("Legend: PASS=genuine pass  SKIP=toolchain unavailable (advisory)  "
          "FAIL!=blocks the gate")
    if report["overall_pass"]:
        print(
            "PASS gpu-export-closure gate (every combo is a genuine PASS or an "
            "honest toolchain-unavailable skip -- zero failures)"
        )
    else:
        print(f"FAIL gpu-export-closure gate -- unexpected failure(s) in: {report['blocking_combos']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("packages", nargs="*",
                         help="Package directories or archives to validate (legacy library-content scan)")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    parser.add_argument("--gpu-export-closure", action="store_true",
                         help="Run the GPU CMake export-set closure probe instead of the legacy "
                              "package-content scan (see module docstring)")
    parser.add_argument("--combo", default="all",
                         choices=[c["key"] for c in GPU_COMBOS] + ["all"],
                         help="With --gpu-export-closure: restrict to one backend combo (default: all)")
    parser.add_argument("--scratch-dir", type=Path, default=None,
                         help="With --gpu-export-closure: use this directory for cmake scratch "
                              "build/install trees instead of an auto-cleaned temp dir (caller must "
                              "clean up; useful for debugging a specific combo's build tree)")
    args = parser.parse_args()

    if args.gpu_export_closure:
        report = evaluate_gpu_export_closure(
            only=None if args.combo == "all" else args.combo,
            scratch_dir=args.scratch_dir,
        )
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            print_gpu_export_closure_report(report)
        return 0 if report["overall_pass"] else 1

    if not args.packages:
        parser.error("at least one package directory or archive is required (or pass --gpu-export-closure)")

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
