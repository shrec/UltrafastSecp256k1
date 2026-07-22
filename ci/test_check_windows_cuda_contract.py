#!/usr/bin/env python3
"""
test_check_windows_cuda_contract.py — unit test for ci/check_windows_cuda_contract.py.

Fail-before/pass-after fixtures for all 5 rejection classes, plus a
comment-cannot-fool-the-gate adversarial pair, plus a sanity check that the
gate passes clean against the real (post-fix) repo tree.

Self-contained. Exit 0 = pass, 1 = fail.
"""
import importlib.util
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GATE = ROOT / "ci" / "check_windows_cuda_contract.py"

failures = []


def check(cond, msg):
    print(("  ok  : " if cond else "  FAIL: ") + msg)
    if not cond:
        failures.append(msg)


def load_gate():
    spec = importlib.util.spec_from_file_location("chk_windows_cuda_contract", str(GATE))
    chk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(chk)
    return chk


def write(tmpdir: Path, rel: str, content: str) -> Path:
    p = tmpdir / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


CUH_SIG = (
    "__device__ inline void field_mul_small(const FieldElement* a, "
    "uint32_t {param}, FieldElement* r) {{\n"
    "    uint64_t x = static_cast<uint64_t>({param});\n"
    "}}\n"
)


def test_reserved_param_name(chk):
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        # fail-before: reserved name present
        bad = write(tmp, "bad.cuh", CUH_SIG.format(param="small"))
        chk.CUH_PATH = bad
        issues = []
        chk.check_field_mul_small_param_name(issues)
        check(any(i["rule"] == "reserved-small-parameter" for i in issues),
              "fail-before: reserved `small` parameter is rejected")

        # pass-after: renamed
        good = write(tmp, "good.cuh", CUH_SIG.format(param="factor"))
        chk.CUH_PATH = good
        issues = []
        chk.check_field_mul_small_param_name(issues)
        check(issues == [], "pass-after: `factor` parameter name is accepted")

        # comment cannot cause a false PASS: signature still says `small`,
        # but a misleading comment claims otherwise
        trick_pass = write(tmp, "trick_pass.cuh",
                            "// FIXED: renamed small -> factor\n"
                            + CUH_SIG.format(param="small"))
        chk.CUH_PATH = trick_pass
        issues = []
        chk.check_field_mul_small_param_name(issues)
        check(any(i["rule"] == "reserved-small-parameter" for i in issues),
              "comment claiming the fix happened does not cause a false PASS")

        # comment cannot cause a false REJECT: signature says `factor`, but a
        # stale comment still mentions `small`
        trick_reject = write(tmp, "trick_reject.cuh",
                              "// old param was called small\n"
                              + CUH_SIG.format(param="factor"))
        chk.CUH_PATH = trick_reject
        issues = []
        chk.check_field_mul_small_param_name(issues)
        check(issues == [], "a stale comment mentioning `small` does not cause a false REJECT")


CMAKE_MISSING = '''
set(_lbtc_gpu_retain "LINKER:--undefined=anchor_x")
'''

CMAKE_GOOD = '''
if(MSVC)
    set(_lbtc_gpu_retain "LINKER:/INCLUDE:anchor_x")
else()
    set(_lbtc_gpu_retain "LINKER:--undefined=anchor_x")
endif()
'''

CMAKE_SWAPPED = '''
if(MSVC)
    set(_lbtc_gpu_retain "LINKER:--undefined=anchor_x")
else()
    set(_lbtc_gpu_retain "LINKER:/INCLUDE:anchor_x")
endif()
'''

HOOK_GOOD = 'extern "C" int anchor_x = 1;\n'
HOOK_MISMATCH = 'extern "C" int anchor_y = 1;\n'


def test_retention_options(chk):
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)

        # fail-before: PR#353 pre-fix shape (no MSVC branch at all)
        chk.CMAKE_PATH = write(tmp, "missing.cmake", CMAKE_MISSING)
        issues = []
        chk.check_lbtc_retention_options(issues)
        check(any(i["rule"] == "msvc-retention-missing" for i in issues),
              "fail-before: missing if(MSVC) branch is rejected")

        # swapped
        chk.CMAKE_PATH = write(tmp, "swapped.cmake", CMAKE_SWAPPED)
        issues = []
        chk.check_lbtc_retention_options(issues)
        check(any(i["rule"] in ("msvc-retention-swapped-or-unscoped",
                                 "gnu-retention-swapped-or-unscoped") for i in issues),
              "swapped MSVC/non-MSVC options are rejected")

        # pass-after: correct shape
        chk.CMAKE_PATH = write(tmp, "good.cmake", CMAKE_GOOD)
        issues = []
        chk.check_lbtc_retention_options(issues)
        check(issues == [], "pass-after: correctly-scoped if(MSVC)/else() is accepted")


def test_anchor_name_consistency(chk):
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        chk.CMAKE_PATH = write(tmp, "good.cmake", CMAKE_GOOD)

        # mismatch
        chk.HOOK_PATH = write(tmp, "mismatch.cpp", HOOK_MISMATCH)
        issues = []
        chk.check_anchor_name_consistency(issues)
        check(any(i["rule"] == "anchor-name-mismatch" for i in issues),
              "anchor name mismatch between CMake and gpu_engine_hook.cpp is rejected")

        # match
        chk.HOOK_PATH = write(tmp, "good.cpp", HOOK_GOOD)
        issues = []
        chk.check_anchor_name_consistency(issues)
        check(issues == [], "matching anchor names across CMake and C++ are accepted")


MSVC_REL = "ci/fixtures/pr353_msvc_link_retention"
CUDA_REL = "ci/fixtures/pr353_windows_small_macro_smoke.cu"


def test_windows_fixture_reachability(chk):
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        (tmp / MSVC_REL).mkdir(parents=True, exist_ok=True)
        chk.LIB_ROOT = tmp
        chk.MSVC_FIXTURE_DIR = MSVC_REL

        # fail-before: workflow_dispatch-only (manual-only is not "reachable")
        chk.CI_YML = write(tmp, ".github/workflows/ci.yml", f'''
on:
  workflow_dispatch: {{}}
jobs:
  windows:
    runs-on: windows-2022
    steps:
      - run: cmake -S {MSVC_REL} -B build
''')
        issues = []
        chk.check_windows_fixture_reachable(issues)
        check(any(i["rule"] == "msvc-fixture-unreachable" for i in issues),
              "fail-before: workflow_dispatch-only Windows job is rejected as unreachable")

        # pass-after: push/pull_request-triggered, windows-* job references the fixture
        chk.CI_YML = write(tmp, ".github/workflows/ci.yml", f'''
on:
  push:
    branches: [main, dev]
  pull_request:
    branches: [main, dev]
jobs:
  windows:
    runs-on: windows-2022
    steps:
      - run: cmake -S {MSVC_REL} -B build
''')
        issues = []
        chk.check_windows_fixture_reachable(issues)
        check(issues == [], "pass-after: mandatory windows-* job referencing the fixture is accepted")


def test_cuda_macro_fixture_reachability(chk):
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        write(tmp, CUDA_REL, "// fixture placeholder\n")
        chk.LIB_ROOT = tmp
        chk.CUDA_MACRO_FIXTURE = CUDA_REL
        chk.GPU_YML = tmp / ".github/workflows/gpu-selfhosted.yml"  # absent -> ignored

        # fail-before: self-hosted-only job (even with nvcc + mandatory trigger)
        chk.CI_YML = write(tmp, ".github/workflows/ci.yml", f'''
on:
  push:
    branches: [main, dev]
jobs:
  gpu:
    runs-on: [self-hosted, linux, cuda]
    steps:
      - run: nvcc -c {CUDA_REL}
''')
        issues = []
        chk.check_cuda_macro_fixture_reachable(issues)
        check(any(i["rule"] == "cuda-macro-fixture-unreachable" for i in issues),
              "fail-before: self-hosted-only nvcc job is rejected (needs no-GPU-hardware path)")

        # pass-after: ordinary runner, mandatory trigger, nvcc present
        chk.CI_YML = write(tmp, ".github/workflows/ci.yml", f'''
on:
  push:
    branches: [main, dev]
  pull_request:
    branches: [main, dev]
jobs:
  cuda-compile-check:
    runs-on: ubuntu-24.04
    steps:
      - run: sudo apt-get install -y nvidia-cuda-toolkit
      - run: nvcc -c {CUDA_REL} -o /tmp/smoke.o -arch=sm_75
''')
        issues = []
        chk.check_cuda_macro_fixture_reachable(issues)
        check(issues == [], "pass-after: ordinary-runner, mandatory-trigger, nvcc job is accepted")


def test_real_repo_sanity(chk):
    """Once the header rename, CMake if(MSVC) branch, both fixtures, and the
    CI workflow wiring are all applied, the gate must pass clean against the
    REAL repo tree (not a fixture)."""
    chk2 = load_gate()  # fresh module, real module-level paths (no monkeypatch)
    report = chk2.evaluate()
    check(report["overall_pass"],
          f"sanity: evaluate() passes against the real repo tree "
          f"(issues: {report['issues']})")


def main() -> int:
    chk = load_gate()
    test_reserved_param_name(chk)
    chk = load_gate()  # reset module-level constants between test groups
    test_retention_options(chk)
    chk = load_gate()
    test_anchor_name_consistency(chk)
    chk = load_gate()
    test_windows_fixture_reachability(chk)
    chk = load_gate()
    test_cuda_macro_fixture_reachability(chk)
    test_real_repo_sanity(chk)

    print("\n" + ("ALL PASS" if not failures else f"FAILURES: {len(failures)}"))
    return 1 if failures else 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
