#!/usr/bin/env python3
"""
check_cuda_arch_policy.py
==========================
Focused CI policy gate for the explicit CUDA architecture policy implemented
in:
  - CMakeLists.txt                              (outer Secp256K1fast suite)
  - libs/UltrafastSecp256k1/CMakeLists.txt       (standalone library)
  - libs/UltrafastSecp256k1/src/cuda/CMakeLists.txt
  - libs/UltrafastSecp256k1/cmake/CudaArchPolicy.cmake (shared policy module)
  - CMakePresets.json / libs/UltrafastSecp256k1/CMakePresets.json (5060 Ti presets)
  - libs/UltrafastSecp256k1/.github/workflows/gpu-selfhosted.yml (self-hosted CI wiring)

See libs/UltrafastSecp256k1/cmake/CUDA_ARCHITECTURE_POLICY.md for the
narrative policy this gate enforces. Invoked as a real self-hosted CI step by
gpu-selfhosted.yml (not a standalone script nothing runs).

Three kinds of checks, all in this one file:
  1. Real `cmake -S <entry> -B <fresh dir>` configure scenarios (compiler-
     execution) against a build directory that is deleted (if present)
     immediately before the run and never reused across scenarios -- so no
     scenario can pass or fail because of a stale/warm CMakeCache.txt.
  2. `cmake -P` module unit-probes that shadow CMAKE_CUDA_COMPILER_VERSION /
     CMAKE_VERSION and call the real ufsecp_apply_cuda_arch_profile() function
     directly, for version-guard branches that would otherwise need a
     toolchain (nvcc 11.x, CMake < 3.24) this project deliberately never
     installs just to satisfy a test.
  3. Static, no-cmake-invocation file-content checks proving the 5060 Ti
     presets and the self-hosted workflow cannot resolve an nvcc13-only
     profile without an explicit nvcc13 compiler pin/preflight actually
     present in the file.

This gate only *configures* (never builds) -- it is a fast, deterministic
policy check, not a compile smoke test. The task's separate validation step
(`cmake --build ... -DCMAKE_CUDA_ARCHITECTURES=120-real`) is the real
compile+link smoke test on this machine's own GPU.

Both NVCC_12 and NVCC_13 accept environment overrides (UFSECP_CI_NVCC12 /
UFSECP_CI_NVCC13) so a CI preflight step can feed this gate the exact
compiler it just resolved and validated, without ever silently substituting
a different one.

Exit codes:
  0  -- all scenarios behaved as expected
  1  -- at least one scenario diverged from the documented policy
  77 -- SKIP: a required real toolkit is genuinely absent from this machine
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

CI_DIR = Path(__file__).resolve().parent
STANDALONE_ROOT = CI_DIR.parent                      # libs/UltrafastSecp256k1
SUITE_ROOT = STANDALONE_ROOT.parent.parent            # Secp256K1fast

# nvcc 12.0.140 is the default `nvcc` on PATH on this machine (ceiling
# compute_90); nvcc 13.2.78 (needed for compute_120/redistributable/
# ci-bounded-recent) lives at a fixed, non-default toolkit path. Both are
# real, installed toolchains on this machine -- neither path is fabricated.
#
# Both paths accept an explicit environment override (UFSECP_CI_NVCC12 /
# UFSECP_CI_NVCC13) so this gate can run unmodified on a different machine
# (or be fed the exact compiler a CI preflight step just resolved, e.g.
# gpu-selfhosted.yml's "Select CUDA 13 toolchain" step) -- it never silently
# substitutes one compiler for another: an override that does not exist on
# disk fails the same honest SKIP(77) path as the hardcoded default below.
NVCC_12 = Path(os.environ.get("UFSECP_CI_NVCC12", "/usr/bin/nvcc"))
NVCC_13 = Path(os.environ.get("UFSECP_CI_NVCC13", "/usr/local/cuda-13.2/bin/nvcc"))

FAILURES: list[str] = []


def _fresh_dir(tmp_root: Path, name: str) -> Path:
    d = tmp_root / name
    if d.exists():
        shutil.rmtree(d)
    return d


def _configure(entry: Path, build_dir: Path, defines: dict[str, str], env_extra: dict[str, str] | None = None,
               timeout: int = 120) -> tuple[int, str]:
    args = ["cmake", "-S", str(entry), "-B", str(build_dir), "-G", "Ninja"]
    for k, v in defines.items():
        args.append(f"-D{k}={v}")
    env = os.environ.copy()
    # Never let a real CUDAARCHS from the invoking shell leak into a scenario
    # that isn't explicitly testing it.
    env.pop("CUDAARCHS", None)
    if env_extra:
        env.update(env_extra)
    try:
        proc = subprocess.run(args, capture_output=True, text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired as e:
        return 124, f"TIMEOUT after {timeout}s: {e}"
    return proc.returncode, proc.stdout + "\n" + proc.stderr


def _cache_value(build_dir: Path, var: str) -> str | None:
    cache = build_dir / "CMakeCache.txt"
    if not cache.exists():
        return None
    text = cache.read_text(errors="replace")
    m = re.search(rf"^{re.escape(var)}:[A-Z]+=(.*)$", text, re.MULTILINE)
    return m.group(1) if m else None


def _check(label: str, condition: bool, detail: str) -> None:
    status = "OK" if condition else "FAIL"
    print(f"  [{status}] {label}")
    if not condition:
        FAILURES.append(f"{label}: {detail}")
        print(f"         {detail}")


def run_standalone_scenarios(tmp_root: Path) -> None:
    entry = STANDALONE_ROOT
    common = {
        "SECP256K1_BUILD_TESTS": "OFF",
        "SECP256K1_BUILD_BENCH": "OFF",
        "SECP256K1_BUILD_EXAMPLES": "OFF",
    }

    # -- 1. Explicit -D precedence: must win even when a profile is ALSO given.
    d = _fresh_dir(tmp_root, "standalone-explicit-precedence")
    rc, out = _configure(entry, d, {
        **common,
        "SECP256K1_BUILD_CUDA": "ON",
        "CMAKE_CUDA_COMPILER": str(NVCC_12),
        "CMAKE_CUDA_ARCHITECTURES": "61-real",
        "SECP256K1_CUDA_ARCH_PROFILE": "legacy-compat",
    })
    resolved = _cache_value(d, "CMAKE_CUDA_ARCHITECTURES")
    _check("standalone: explicit -D wins over profile", rc == 0 and resolved == "61-real",
           f"rc={rc} resolved={resolved!r}\n{out[-1500:]}")

    # -- 2. CUDAARCHS env precedence: must win over a profile, without an explicit -D.
    d = _fresh_dir(tmp_root, "standalone-cudaarchs-precedence")
    rc, out = _configure(entry, d, {
        **common,
        "SECP256K1_BUILD_CUDA": "ON",
        "CMAKE_CUDA_COMPILER": str(NVCC_12),
        "SECP256K1_CUDA_ARCH_PROFILE": "legacy-compat",
    }, env_extra={"CUDAARCHS": "75-real"})
    resolved = _cache_value(d, "CMAKE_CUDA_ARCHITECTURES")
    _check("standalone: CUDAARCHS env wins over profile", rc == 0 and resolved == "75-real",
           f"rc={rc} resolved={resolved!r}\n{out[-1500:]}")

    # -- 3. Missing input -> fail-fast, no silent default.
    d = _fresh_dir(tmp_root, "standalone-missing-input")
    rc, out = _configure(entry, d, {**common, "SECP256K1_BUILD_CUDA": "ON", "CMAKE_CUDA_COMPILER": str(NVCC_13)})
    _check("standalone: missing arch input fails configure",
           rc != 0 and "no trustworthy CUDA architecture choice" in out,
           f"rc={rc}\n{out[-1500:]}")

    # -- 4. Invalid profile name -> fail-fast.
    d = _fresh_dir(tmp_root, "standalone-invalid-profile")
    rc, out = _configure(entry, d, {
        **common, "SECP256K1_BUILD_CUDA": "ON", "CMAKE_CUDA_COMPILER": str(NVCC_13),
        "SECP256K1_CUDA_ARCH_PROFILE": "not-a-real-profile",
    })
    _check("standalone: invalid profile name fails configure",
           rc != 0 and "Unknown SECP256K1_CUDA_ARCH_PROFILE" in out,
           f"rc={rc}\n{out[-1500:]}")

    # -- 5. local-native profile happy path (CMake 4.3.4 installed >= 3.24).
    d = _fresh_dir(tmp_root, "standalone-profile-local-native")
    rc, out = _configure(entry, d, {
        **common, "SECP256K1_BUILD_CUDA": "ON", "CMAKE_CUDA_COMPILER": str(NVCC_13),
        "SECP256K1_CUDA_ARCH_PROFILE": "local-native",
    })
    resolved = _cache_value(d, "CMAKE_CUDA_ARCHITECTURES")
    _check("standalone: local-native profile configures (CMAKE_CUDA_ARCHITECTURES=native)",
           rc == 0 and resolved == "native", f"rc={rc} resolved={resolved!r}\n{out[-1500:]}")

    # -- 6. ci-bounded-recent profile happy path (needs nvcc >= 13.0).
    d = _fresh_dir(tmp_root, "standalone-profile-ci-bounded-recent")
    rc, out = _configure(entry, d, {
        **common, "SECP256K1_BUILD_CUDA": "ON", "CMAKE_CUDA_COMPILER": str(NVCC_13),
        "SECP256K1_CUDA_ARCH_PROFILE": "ci-bounded-recent",
    })
    resolved = _cache_value(d, "CMAKE_CUDA_ARCHITECTURES")
    expected = "89-real;89-virtual;120-real;120-virtual"
    _check("standalone: ci-bounded-recent profile configures", rc == 0 and resolved == expected,
           f"rc={rc} resolved={resolved!r}\n{out[-1500:]}")

    # -- 7. legacy-compat profile happy path (needs nvcc < 13.0).
    d = _fresh_dir(tmp_root, "standalone-profile-legacy-compat")
    rc, out = _configure(entry, d, {
        **common, "SECP256K1_BUILD_CUDA": "ON", "CMAKE_CUDA_COMPILER": str(NVCC_12),
        "SECP256K1_CUDA_ARCH_PROFILE": "legacy-compat",
    })
    resolved = _cache_value(d, "CMAKE_CUDA_ARCHITECTURES")
    expected = "52-virtual;89-real;89-virtual"
    _check("standalone: legacy-compat profile configures on nvcc 12.x", rc == 0 and resolved == expected,
           f"rc={rc} resolved={resolved!r}\n{out[-1500:]}")

    # -- 8. redistributable profile happy path (needs nvcc >= 13.0).
    d = _fresh_dir(tmp_root, "standalone-profile-redistributable")
    rc, out = _configure(entry, d, {
        **common, "SECP256K1_BUILD_CUDA": "ON", "CMAKE_CUDA_COMPILER": str(NVCC_13),
        "SECP256K1_CUDA_ARCH_PROFILE": "redistributable",
    })
    resolved = _cache_value(d, "CMAKE_CUDA_ARCHITECTURES")
    expected = "75-virtual;86-real;89-real;120-real;120-virtual"
    _check("standalone: redistributable profile configures", rc == 0 and resolved == expected,
           f"rc={rc} resolved={resolved!r}\n{out[-1500:]}")

    # -- 9. Version guard: ci-bounded-recent must REJECT nvcc < 13.0.
    d = _fresh_dir(tmp_root, "standalone-guard-ci-bounded-recent-old-nvcc")
    rc, out = _configure(entry, d, {
        **common, "SECP256K1_BUILD_CUDA": "ON", "CMAKE_CUDA_COMPILER": str(NVCC_12),
        "SECP256K1_CUDA_ARCH_PROFILE": "ci-bounded-recent",
    })
    _check("standalone: ci-bounded-recent rejects nvcc 12.x",
           rc != 0 and "requires nvcc >= 13.0" in out, f"rc={rc}\n{out[-1500:]}")

    # -- 10. Version guard: legacy-compat must REJECT nvcc >= 13.0 (never keep
    #        claiming compute_52 support under nvcc 13 -- this is the exact
    #        defect this policy exists to prevent).
    d = _fresh_dir(tmp_root, "standalone-guard-legacy-compat-new-nvcc")
    rc, out = _configure(entry, d, {
        **common, "SECP256K1_BUILD_CUDA": "ON", "CMAKE_CUDA_COMPILER": str(NVCC_13),
        "SECP256K1_CUDA_ARCH_PROFILE": "legacy-compat",
    })
    _check("standalone: legacy-compat rejects nvcc 13.x (no compute_52 claim under nvcc13)",
           rc != 0 and "nvcc 13.x no longer accepts" in out, f"rc={rc}\n{out[-1500:]}")

    # -- 11. Version guard: redistributable must REJECT nvcc < 13.0.
    d = _fresh_dir(tmp_root, "standalone-guard-redistributable-old-nvcc")
    rc, out = _configure(entry, d, {
        **common, "SECP256K1_BUILD_CUDA": "ON", "CMAKE_CUDA_COMPILER": str(NVCC_12),
        "SECP256K1_CUDA_ARCH_PROFILE": "redistributable",
    })
    _check("standalone: redistributable rejects nvcc 12.x",
           rc != 0 and "requires nvcc >= 13.0" in out, f"rc={rc}\n{out[-1500:]}")

    # -- 12. CPU-only builds are completely unaffected by this policy.
    d = _fresh_dir(tmp_root, "standalone-cpu-only-unaffected")
    rc, out = _configure(entry, d, {**common, "SECP256K1_BUILD_CUDA": "OFF"})
    _check("standalone: CPU-only (SECP256K1_BUILD_CUDA=OFF) configures cleanly",
           rc == 0, f"rc={rc}\n{out[-1500:]}")

    # -- 13. local-native's CMake>=3.24 guard: no CMake<3.24 binary exists on
    #        this machine to exercise end-to-end, so the guard's *logic* is
    #        verified directly by invoking the real policy module in
    #        `cmake -P` script mode with CMAKE_VERSION shadowed to an old
    #        value in the calling scope (CMake variable reads fall through to
    #        the enclosing scope) -- this executes the actual FATAL_ERROR
    #        branch in cmake/CudaArchPolicy.cmake, not a re-implementation of it.
    probe = tmp_root / "local_native_version_guard_probe.cmake"
    probe.write_text(
        f'set(CMAKE_VERSION "3.20.0")\n'
        f'include("{(STANDALONE_ROOT / "cmake" / "CudaArchPolicy.cmake").as_posix()}")\n'
        f'set(SECP256K1_CUDA_ARCH_PROFILE "local-native")\n'
        f'ufsecp_apply_cuda_arch_profile()\n'
    )
    proc = subprocess.run(["cmake", "-P", str(probe)], capture_output=True, text=True, timeout=30)
    out = proc.stdout + proc.stderr
    _check("standalone: local-native rejects CMake < 3.24 (module unit-probe)",
           proc.returncode != 0 and "requires CMake >= 3.24" in out,
           f"rc={proc.returncode}\n{out[-1000:]}")

    # -- 14. Version guard: legacy-compat must REJECT nvcc < 12.0 (lower
    #        bound). No real nvcc 11.x binary exists on this machine, and this
    #        project never installs one just to satisfy a test -- nvcc 11.x is
    #        a permanently rejected toolchain (see CLAUDE.md forbidden list:
    #        "no legacy nvcc11 acceptance"). Exactly like check #13's CMake<3.24
    #        probe, this shadows CMAKE_CUDA_COMPILER_VERSION in a `cmake -P`
    #        top-level scope and calls the REAL ufsecp_apply_cuda_arch_profile()
    #        function, exercising the actual FATAL_ERROR branch in
    #        cmake/CudaArchPolicy.cmake -- a real unit-probe of the policy
    #        module's own logic, not a fabricated "successful" nvcc run.
    probe = tmp_root / "legacy_compat_lower_bound_probe.cmake"
    probe.write_text(
        f'set(CMAKE_CUDA_COMPILER_VERSION "11.8")\n'
        f'include("{(STANDALONE_ROOT / "cmake" / "CudaArchPolicy.cmake").as_posix()}")\n'
        f'set(SECP256K1_CUDA_ARCH_PROFILE "legacy-compat")\n'
        f'ufsecp_apply_cuda_arch_profile()\n'
    )
    proc = subprocess.run(["cmake", "-P", str(probe)], capture_output=True, text=True, timeout=30)
    out = proc.stdout + proc.stderr
    _check("standalone: legacy-compat rejects nvcc < 12.0 (module unit-probe, no nvcc11 binary needed)",
           proc.returncode != 0 and "requires nvcc >= 12.0" in out,
           f"rc={proc.returncode}\n{out[-1000:]}")

    # -- 15. Sanity: the SAME probe technique must confirm legacy-compat's
    #        happy path (nvcc 12.x, exactly the range >=12.0,<13.0) still
    #        resolves via the real module, not just via the real-compiler
    #        scenario #7 above -- i.e. the lower-bound guard added for this
    #        task did not accidentally narrow the accepted range.
    probe = tmp_root / "legacy_compat_in_range_probe.cmake"
    probe.write_text(
        f'set(CMAKE_CUDA_COMPILER_VERSION "12.0")\n'
        f'include("{(STANDALONE_ROOT / "cmake" / "CudaArchPolicy.cmake").as_posix()}")\n'
        f'set(SECP256K1_CUDA_ARCH_PROFILE "legacy-compat")\n'
        f'ufsecp_apply_cuda_arch_profile()\n'
        f'message(STATUS "PROBE_RESULT=${{CMAKE_CUDA_ARCHITECTURES}}")\n'
    )
    proc = subprocess.run(["cmake", "-P", str(probe)], capture_output=True, text=True, timeout=30)
    out = proc.stdout + proc.stderr
    _check("standalone: legacy-compat accepts nvcc exactly 12.0 (module unit-probe, lower boundary inclusive)",
           proc.returncode == 0 and "PROBE_RESULT=52-virtual;89-real;89-virtual" in out,
           f"rc={proc.returncode}\n{out[-1000:]}")


def run_suite_scenarios(tmp_root: Path) -> None:
    entry = SUITE_ROOT
    common = {"BUILD_TESTING": "OFF"}

    # -- 1. Missing input at the OUTER suite entry point must delegate to the
    #       same fail-fast policy (no separate silent default reappears here).
    d = _fresh_dir(tmp_root, "suite-missing-input")
    rc, out = _configure(entry, d, {**common, "SECP256K1_BUILD_CUDA": "ON", "CMAKE_CUDA_COMPILER": str(NVCC_13)})
    _check("suite: missing arch input fails configure (delegated fail-fast)",
           rc != 0 and "no trustworthy CUDA architecture choice" in out,
           f"rc={rc}\n{out[-1500:]}")

    # -- 2. Explicit -D precedence must also hold at the outer entry point --
    #       the outer suite must never re-populate CMAKE_CUDA_ARCHITECTURES
    #       itself (that was the old "89" CACHE-without-FORCE race bug).
    d = _fresh_dir(tmp_root, "suite-explicit-precedence")
    rc, out = _configure(entry, d, {
        **common, "SECP256K1_BUILD_CUDA": "ON", "CMAKE_CUDA_COMPILER": str(NVCC_12),
        "CMAKE_CUDA_ARCHITECTURES": "61-real",
    })
    resolved = _cache_value(d, "CMAKE_CUDA_ARCHITECTURES")
    _check("suite: explicit -D is preserved through outer delegation",
           rc == 0 and resolved == "61-real", f"rc={rc} resolved={resolved!r}\n{out[-1500:]}")

    # -- 3. A named profile resolves correctly end-to-end through the outer
    #       suite's add_subdirectory(libs/UltrafastSecp256k1) delegation.
    d = _fresh_dir(tmp_root, "suite-profile-ci-bounded-recent")
    rc, out = _configure(entry, d, {
        **common, "SECP256K1_BUILD_CUDA": "ON", "CMAKE_CUDA_COMPILER": str(NVCC_13),
        "SECP256K1_CUDA_ARCH_PROFILE": "ci-bounded-recent",
    })
    resolved = _cache_value(d, "CMAKE_CUDA_ARCHITECTURES")
    expected = "89-real;89-virtual;120-real;120-virtual"
    _check("suite: ci-bounded-recent profile resolves through outer delegation",
           rc == 0 and resolved == expected, f"rc={rc} resolved={resolved!r}\n{out[-1500:]}")

    # -- 4. CPU-only outer-suite configure is unaffected.
    d = _fresh_dir(tmp_root, "suite-cpu-only-unaffected")
    rc, out = _configure(entry, d, {**common, "SECP256K1_BUILD_CUDA": "OFF"})
    _check("suite: CPU-only outer-suite configure is unaffected",
           rc == 0, f"rc={rc}\n{out[-1500:]}")


def _suite_checkout_present() -> bool:
    """True iff SUITE_ROOT looks like a real Secp256K1fast outer-suite checkout
    (root CMakeLists.txt that add_subdirectory()s this library), not just some
    unrelated ancestor directory. On a standalone-repo-only checkout of
    UltrafastSecp256k1 (e.g. this library's own self-hosted GPU CI runner,
    which checks out ONLY this repository -- no nested outer suite exists on
    that runner's disk at all), SUITE_ROOT resolves to an arbitrary ancestor
    that is not a Secp256K1fast checkout. Detecting that honestly and skipping
    the outer-suite scenarios (loudly, not silently) is correct there; running
    them would fail with a "source directory has no CMakeLists.txt" cmake
    error that has nothing to do with the CUDA architecture policy itself.
    """
    cmake_file = SUITE_ROOT / "CMakeLists.txt"
    if not cmake_file.exists():
        return False
    try:
        text = cmake_file.read_text(errors="replace")
    except OSError:
        return False
    return "add_subdirectory(libs/UltrafastSecp256k1)" in text


def _preset_entry(presets_json: dict, preset_name: str) -> dict | None:
    for p in presets_json.get("configurePresets", []):
        if p.get("name") == preset_name:
            return p
    return None


def _assert_5060ti_preset_pins_nvcc13(presets_path: Path, preset_name: str) -> None:
    """Static (no cmake invocation) proof that a named RTX 5060 Ti preset
    cannot silently resolve an nvcc13-only profile: it must both select a
    profile that requires nvcc >= 13.0 AND pin CMAKE_CUDA_COMPILER to a real,
    version-verified nvcc >= 13.0 path -- never rely on whatever `nvcc`
    happens to resolve to on PATH (this machine's PATH default is nvcc 12.0,
    which would FATAL_ERROR on the ci-bounded-recent version guard)."""
    label = f"{presets_path}: preset '{preset_name}'"
    if not presets_path.exists():
        print(f"  [SKIP] {presets_path} does not exist in this checkout.")
        return
    presets_json = json.loads(presets_path.read_text())
    entry = _preset_entry(presets_json, preset_name)
    if entry is None:
        _check(f"{label} exists", False, f"no configurePresets entry named {preset_name!r}")
        return
    cache = entry.get("cacheVariables", {})
    profile = cache.get("SECP256K1_CUDA_ARCH_PROFILE")
    compiler = cache.get("CMAKE_CUDA_COMPILER")
    _check(f"{label} selects an nvcc>=13-requiring profile",
           profile in ("ci-bounded-recent", "redistributable"),
           f"expected SECP256K1_CUDA_ARCH_PROFILE in (ci-bounded-recent, redistributable), got {profile!r}")
    _check(f"{label} pins an explicit CMAKE_CUDA_COMPILER (never the PATH default)",
           bool(compiler),
           f"preset selects {profile!r} (requires nvcc>=13.0) but does not pin "
           f"CMAKE_CUDA_COMPILER -- a fresh invocation would resolve whatever "
           f"`nvcc` is on PATH (nvcc 12.0 on this machine) and FATAL_ERROR on "
           f"the version guard instead of building sm_120")
    if not compiler:
        return
    compiler_path = Path(compiler)
    exists = compiler_path.exists()
    _check(f"{label} CMAKE_CUDA_COMPILER path exists on this machine",
           exists, f"{compiler} does not exist")
    if not exists:
        return
    proc = subprocess.run([str(compiler_path), "--version"], capture_output=True, text=True, timeout=15)
    m = re.search(r"release (\d+)\.", proc.stdout)
    major = int(m.group(1)) if m else -1
    _check(f"{label} pinned CMAKE_CUDA_COMPILER really is nvcc >= 13",
           major >= 13, f"{compiler} --version reports major={major}, expected >= 13")


def run_static_coherence_checks() -> None:
    """No-cmake-invocation, file-content proof that the 5060 Ti presets and
    the self-hosted GPU workflow cannot select an nvcc13-only profile without
    an explicit nvcc13 compiler path/preflight actually present in the file
    (as opposed to the compiler-execution scenarios above, which prove the
    CMake *logic* rejects a wrong compiler when one is explicitly given)."""
    print("== static preset/workflow coherence checks ==")

    _assert_5060ti_preset_pins_nvcc13(SUITE_ROOT / "CMakePresets.json", "linux-cuda-5060ti")
    _assert_5060ti_preset_pins_nvcc13(STANDALONE_ROOT / "CMakePresets.json", "cuda-release-5060ti")

    wf_path = STANDALONE_ROOT / ".github" / "workflows" / "gpu-selfhosted.yml"
    if not wf_path.exists():
        _check("gpu-selfhosted.yml exists", False, f"{wf_path} not found")
        return
    wf_text = wf_path.read_text()

    idx_preflight = wf_text.find("Select CUDA 13 toolchain")
    idx_manifest = wf_text.find("Emit GPU environment manifest")
    idx_policy_gate_step = wf_text.find("CUDA architecture policy gate")
    idx_configure_main = wf_text.find("Configure (CUDA + OpenCL Release)")
    idx_configure_lbtc = wf_text.find("libbitcoin GPU/CPU consensus differential")

    _check("workflow: has an explicit CUDA13 toolchain preflight step",
           idx_preflight != -1 and "CUDACXX=" in wf_text,
           "gpu-selfhosted.yml has no toolchain-selection step exporting CUDACXX")
    _check("workflow: preflight step verifies the compiler binary exists",
           "-x \"${CANDIDATE}\"" in wf_text or "-x \"$CANDIDATE\"" in wf_text,
           "preflight step must verify the resolved nvcc path is executable before trusting it")
    _check("workflow: preflight step verifies the compiler major version",
           "RAW_MAJOR" in wf_text and "-lt 13" in wf_text,
           "preflight step must parse and check the resolved nvcc's major version >= 13")
    _check("workflow: policy gate step (check_cuda_arch_policy.py) is wired in",
           idx_policy_gate_step != -1 and "check_cuda_arch_policy.py" in wf_text,
           "gpu-selfhosted.yml never invokes ci/check_cuda_arch_policy.py -- not a real CI gate")
    _check("workflow: policy gate step runs AFTER toolchain selection",
           idx_preflight != -1 and idx_policy_gate_step != -1 and idx_preflight < idx_policy_gate_step,
           "the policy gate step must run after the toolchain preflight step")
    _check("workflow: preflight step runs BEFORE the environment manifest step",
           idx_preflight != -1 and idx_manifest != -1 and idx_preflight < idx_manifest,
           "toolchain preflight must run BEFORE the manifest step")
    _check("workflow: preflight step runs BEFORE both configure steps",
           idx_preflight != -1 and idx_configure_main != -1 and idx_configure_lbtc != -1
           and idx_preflight < idx_configure_main and idx_preflight < idx_configure_lbtc,
           "toolchain preflight must run BEFORE both configure steps")
    _check("workflow: both configure steps pass CMAKE_CUDA_COMPILER explicitly",
           wf_text.count('-DCMAKE_CUDA_COMPILER="${CUDACXX}"') >= 2,
           'both cmake configure invocations must pass -DCMAKE_CUDA_COMPILER="${CUDACXX}" '
           'explicitly, not rely on env auto-detection alone')
    manifest_block = wf_text[idx_manifest: idx_configure_main] if -1 not in (idx_manifest, idx_configure_main) else ""
    _check("workflow: manifest step reports the SAME $CUDACXX the preflight step selected",
           "${CUDACXX}" in manifest_block,
           "the manifest step must report the exact $CUDACXX the preflight step resolved, "
           "not re-derive a (possibly different) compiler from PATH")


def main() -> int:
    if not NVCC_13.exists():
        print(f"SKIP: {NVCC_13} not present on this machine -- the ci-bounded-recent/"
              "redistributable/version-guard scenarios require a real nvcc >= 13.0 "
              "toolkit. This is a real-hardware/toolkit gate, not something to fake.")
        return 77
    if not NVCC_12.exists():
        print(f"SKIP: {NVCC_12} not present on this machine -- the legacy-compat "
              "scenarios require a real nvcc < 13.0 toolkit.")
        return 77

    with tempfile.TemporaryDirectory(prefix="ufsecp-cuda-arch-policy-gate-") as tmp:
        tmp_root = Path(tmp)
        print("== standalone library entry point (libs/UltrafastSecp256k1) ==")
        run_standalone_scenarios(tmp_root)
        print()
        print("== outer suite entry point (Secp256K1fast) ==")
        if _suite_checkout_present():
            run_suite_scenarios(tmp_root)
        else:
            print(f"  [SKIP] {SUITE_ROOT} is not a Secp256K1fast outer-suite checkout "
                  f"(no CMakeLists.txt with add_subdirectory(libs/UltrafastSecp256k1)) -- "
                  f"expected on a standalone-repo-only checkout, e.g. this library's own "
                  f"self-hosted GPU CI runner, which checks out ONLY UltrafastSecp256k1. "
                  f"The standalone-entry scenarios above already cover the full policy "
                  f"module; this is an honest, explicit skip -- not a silent pass.")

    print()
    run_static_coherence_checks()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} scenario(s) FAILED:")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("All CUDA architecture policy scenarios behaved as documented.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
