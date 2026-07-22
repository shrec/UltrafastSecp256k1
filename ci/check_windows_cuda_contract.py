#!/usr/bin/env python3
"""
check_windows_cuda_contract.py — CI gate: PR #353 Windows/MSVC CUDA
build-portability contract (Eric Voskuil / evoskuil, reproduced by hand into
dev — see docs/WINDOWS_CUDA_BUILD_CONTRACT.md; the PR itself is never merged
or cherry-picked, only its two intended fixes are reproduced).

Enforces, via STRUCTURAL parsing (never a bare substring/comment match):

  (a) field_mul_small's ACTUAL parameter list in src/cuda/include/secp256k1.cuh
      must not contain a parameter literally named `small` (collides with
      Windows <rpcndr.h>'s `#define small char`). Comments are stripped before
      parsing, so a comment cannot cause a false PASS (claiming the rename
      happened) or a false REJECT (an old comment mentioning `small`).

  (b) compat/libbitcoin_direct/CMakeLists.txt must set _lbtc_gpu_retain to
      LINKER:/INCLUDE:<anchor> strictly inside an if(MSVC) branch, and
      LINKER:--undefined=<anchor> strictly inside that construct's matching
      else() branch -- detected by a real if/elseif/else/endif nesting walk,
      not proximity. Missing, swapped, or unscoped options are rejected.

  (c) The anchor name in both CMake retention options must equal the anchor
      name in src/gpu/src/gpu_engine_hook.cpp's
      `extern "C" int <anchor> = 1;` definition.

  (d) ci/fixtures/pr353_msvc_link_retention/ must be reachable: built and
      tested by a job that (i) runs on a windows-* runner and (ii) lives in
      a workflow whose `on:` trigger includes push or pull_request (a
      workflow_dispatch-only workflow does not count).

  (e) ci/fixtures/pr353_windows_small_macro_smoke.cu must be reachable: an
      nvcc-invoking step in a job that (i) does NOT run on a self-hosted
      runner and (ii) lives in a push/pull_request-triggered workflow.

Exit 0 = contract holds. Exit 1 = one or more violations. No ADVISORY_SKIP
sentinel: every check here is pure static analysis over files that always
exist in a checked-out tree, so this gate is never "inapplicable".

Usage:
    python3 ci/check_windows_cuda_contract.py            # text report
    python3 ci/check_windows_cuda_contract.py --json      # machine-readable

The same gate also enforces WIN-CUDA-001 for the real Windows CUDA workflow:
the compiler/runtime development headers must be installed, configuration must
explicitly enable CUDA, and both the GPU host and kernel targets must be built.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

try:
    import yaml
except ImportError as exc:  # Mandatory gate: an unavailable parser is a failure.
    print(f"::error::check_windows_cuda_contract: PyYAML unavailable: {exc}")
    sys.exit(1)


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "windows-cuda.yml"
REQUIRED_SUBPACKAGES = {
    "nvcc",
    "crt",
    "cudart",
    "thrust",
    "visual_studio_integration",
}
INVALID_WINDOWS_SUBPACKAGES = {"cudart_dev"}
REQUIRED_TARGETS = {"secp256k1_gpu_host", "secp256k1_cuda_lib"}


def _problem(kind: str, detail: str) -> dict:
    return {"kind": kind, "detail": detail}


def evaluate_document(document: dict) -> list[dict]:
    """Return deterministic contract violations for a parsed workflow."""
    jobs = document.get("jobs") if isinstance(document, dict) else None
    job = jobs.get("windows-cuda") if isinstance(jobs, dict) else None
    steps = job.get("steps") if isinstance(job, dict) else None
    if not isinstance(steps, list):
        return [_problem("windows_cuda_job_missing", "jobs.windows-cuda.steps is absent")]

    action_steps = [
        step for step in steps
        if isinstance(step, dict)
        and str(step.get("uses", "")).startswith("Jimver/cuda-toolkit@")
    ]
    problems = []
    if len(action_steps) != 1:
        problems.append(_problem(
            "cuda_toolkit_action_count",
            f"expected exactly one pinned Jimver/cuda-toolkit action, found {len(action_steps)}",
        ))
    else:
        action = action_steps[0]
        uses = str(action.get("uses", ""))
        revision = uses.rsplit("@", 1)[-1]
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            problems.append(_problem("cuda_toolkit_action_unpinned", uses))
        options = action.get("with")
        options = options if isinstance(options, dict) else {}
        if options.get("method") != "network":
            problems.append(_problem("cuda_install_method", "method must be network"))
        raw_packages = options.get("sub-packages")
        try:
            packages = json.loads(raw_packages) if isinstance(raw_packages, str) else raw_packages
        except json.JSONDecodeError as exc:
            packages = None
            problems.append(_problem("cuda_subpackages_malformed", str(exc)))
        if not isinstance(packages, list) or not all(isinstance(v, str) for v in packages):
            if not any(p["kind"] == "cuda_subpackages_malformed" for p in problems):
                problems.append(_problem("cuda_subpackages_malformed", "must be a JSON string array"))
        else:
            missing = sorted(REQUIRED_SUBPACKAGES - set(packages))
            if missing:
                problems.append(_problem("cuda_subpackages_missing", ", ".join(missing)))
            invalid = sorted(INVALID_WINDOWS_SUBPACKAGES & set(packages))
            if invalid:
                problems.append(_problem(
                    "cuda_subpackages_invalid_windows",
                    ", ".join(invalid),
                ))

    configure_runs = [
        str(step.get("run", ""))
        for step in steps
        if isinstance(step, dict) and "configure" in str(step.get("name", "")).lower()
    ]
    configure = "\n".join(configure_runs)
    if "CMAKE_CUDA_COMPILER" not in configure:
        problems.append(_problem("cuda_compiler_not_explicit", "configure must pin CMAKE_CUDA_COMPILER"))
    if "SECP256K1_BUILD_CUDA=ON" not in configure:
        problems.append(_problem("cuda_build_not_enabled", "configure must enable SECP256K1_BUILD_CUDA"))

    build_runs = [
        str(step.get("run", ""))
        for step in steps
        if isinstance(step, dict) and "build" in str(step.get("name", "")).lower()
    ]
    build = "\n".join(build_runs)
    missing_targets = sorted(target for target in REQUIRED_TARGETS if target not in build)
    if missing_targets:
        problems.append(_problem("cuda_targets_not_built", ", ".join(missing_targets)))

    return problems


def evaluate_workflow() -> dict:
    try:
        document = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        problems = [_problem("workflow_unreadable", str(exc))]
    else:
        problems = evaluate_document(document)
    return {"overall_pass": not problems, "problems": problems}


SCRIPT_DIR = Path(__file__).resolve().parent
LIB_ROOT = SCRIPT_DIR.parent

CUH_PATH = LIB_ROOT / "src/cuda/include/secp256k1.cuh"
CMAKE_PATH = LIB_ROOT / "compat/libbitcoin_direct/CMakeLists.txt"
HOOK_PATH = LIB_ROOT / "src/gpu/src/gpu_engine_hook.cpp"
CI_YML = LIB_ROOT / ".github/workflows/ci.yml"
GPU_YML = LIB_ROOT / ".github/workflows/gpu-selfhosted.yml"

CUDA_MACRO_FIXTURE = "ci/fixtures/pr353_windows_small_macro_smoke.cu"
MSVC_FIXTURE_DIR = "ci/fixtures/pr353_msvc_link_retention"

RESERVED_PARAM_NAME = "small"

FIELD_MUL_SMALL_SIG_RE = re.compile(
    r'__device__\s+inline\s+void\s+field_mul_small\s*\(\s*([^)]*)\)',
    re.DOTALL,
)

IF_RE = re.compile(r'^\s*if\s*\(\s*(.*?)\s*\)\s*$', re.IGNORECASE)
ELSEIF_RE = re.compile(r'^\s*elseif\s*\(\s*(.*?)\s*\)\s*$', re.IGNORECASE)
ELSE_RE = re.compile(r'^\s*else\s*\(\s*\)\s*$', re.IGNORECASE)
ENDIF_RE = re.compile(r'^\s*endif\s*\(', re.IGNORECASE)
SET_RETAIN_RE = re.compile(r'^\s*set\s*\(\s*_lbtc_gpu_retain\s+"([^"]*)"', re.IGNORECASE)

ANCHOR_RE_CMAKE_MSVC = re.compile(r'LINKER:/INCLUDE:([A-Za-z_]\w*)')
ANCHOR_RE_CMAKE_GNU = re.compile(r'LINKER:--undefined=([A-Za-z_]\w*)')
ANCHOR_RE_CPP = re.compile(r'extern\s+"C"\s+int\s+([A-Za-z_]\w*)\s*=\s*1\s*;')


# ---------------------------------------------------------------------------
# Comment stripping (never let a comment influence a PASS or a REJECT)
# ---------------------------------------------------------------------------
def strip_c_comments(text: str) -> str:
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
    text = re.sub(r'//[^\n]*', '', text)
    return text


def strip_cmake_comments(text: str) -> str:
    return re.sub(r'#[^\n]*', '', text)


# ---------------------------------------------------------------------------
# (a) field_mul_small reserved parameter name
# ---------------------------------------------------------------------------
def _extract_param_names(param_list: str) -> list:
    names = []
    for raw in param_list.split(','):
        raw = raw.split('=')[0].strip()
        if not raw:
            continue
        tokens = raw.replace('*', ' * ').replace('&', ' & ').split()
        if not tokens:
            continue
        name = tokens[-1].lstrip('*&').rstrip('[]')
        names.append(name)
    return names


def check_field_mul_small_param_name(issues: list):
    if not CUH_PATH.exists():
        issues.append({"rule": "field-mul-small-missing-file",
                        "file": "src/cuda/include/secp256k1.cuh",
                        "message": "File not found."})
        return
    text = strip_c_comments(CUH_PATH.read_text(encoding="utf-8", errors="replace"))
    m = FIELD_MUL_SMALL_SIG_RE.search(text)
    if not m:
        issues.append({"rule": "field-mul-small-not-found",
                        "file": "src/cuda/include/secp256k1.cuh",
                        "message": "field_mul_small(...) definition not found."})
        return
    params = _extract_param_names(m.group(1))
    if RESERVED_PARAM_NAME in params:
        issues.append({
            "rule": "reserved-small-parameter",
            "file": "src/cuda/include/secp256k1.cuh",
            "message": (
                "field_mul_small() declares a parameter literally named 'small', "
                "which collides with Windows <rpcndr.h>'s `#define small char` "
                "on any MSVC/nvcc-on-Windows build (PR #353). Rename to `factor`."
            ),
        })


# ---------------------------------------------------------------------------
# (b) + (c) CMake retention options: presence, scoping, anchor consistency
# ---------------------------------------------------------------------------
def _build_line_branch_map(text: str):
    """Forward pass: for each line, the FULL stack of enclosing
    if/elseif/else branches active at that line (innermost last). An `else`
    entry PRESERVES the paired if's original condition text (for pairing
    checks), tagged with kind='else'."""
    lines = text.splitlines()
    stack: list = []
    stack_at_line = []
    for line in lines:
        s = line.strip()
        if IF_RE.match(s):
            stack.append(['if', IF_RE.match(s).group(1)])
        elif ELSEIF_RE.match(s):
            if stack:
                stack[-1] = ['elseif', ELSEIF_RE.match(s).group(1)]
        elif ELSE_RE.match(s):
            if stack:
                stack[-1] = ['else', stack[-1][1]]
        elif ENDIF_RE.match(s):
            if stack:
                stack.pop()
        stack_at_line.append([tuple(e) for e in stack])
    return lines, stack_at_line


def check_lbtc_retention_options(issues: list):
    if not CMAKE_PATH.exists():
        issues.append({"rule": "cmake-missing-file",
                        "file": "compat/libbitcoin_direct/CMakeLists.txt",
                        "message": "File not found."})
        return
    raw = CMAKE_PATH.read_text(encoding="utf-8", errors="replace")
    text = strip_cmake_comments(raw)
    lines, stack_at_line = _build_line_branch_map(text)

    msvc_hits, gnu_hits = [], []
    for i, line in enumerate(lines):
        m = SET_RETAIN_RE.match(line)
        if not m:
            continue
        value = m.group(1)
        innermost = stack_at_line[i][-1] if stack_at_line[i] else None
        if value.startswith("LINKER:/INCLUDE:"):
            msvc_hits.append((i, value, innermost))
        elif value.startswith("LINKER:--undefined="):
            gnu_hits.append((i, value, innermost))

    rel = "compat/libbitcoin_direct/CMakeLists.txt"
    if not msvc_hits:
        issues.append({"rule": "msvc-retention-missing", "file": rel,
                        "message": "No `set(_lbtc_gpu_retain \"LINKER:/INCLUDE:<anchor>\")` "
                                   "found -- MSVC retention option is missing."})
    if not gnu_hits:
        issues.append({"rule": "gnu-retention-missing", "file": rel,
                        "message": "No `set(_lbtc_gpu_retain \"LINKER:--undefined=<anchor>\")` "
                                   "found -- non-MSVC retention option is missing/regressed."})
    if not msvc_hits or not gnu_hits:
        return

    for (i, _value, innermost) in msvc_hits:
        ok = (innermost is not None and innermost[0] in ('if', 'elseif')
              and re.search(r'\bMSVC\b', innermost[1] or '', re.IGNORECASE))
        if not ok:
            issues.append({"rule": "msvc-retention-swapped-or-unscoped", "file": rel,
                            "message": f"line {i + 1}: LINKER:/INCLUDE: is not inside an "
                                       f"if(MSVC) branch (innermost: {innermost}) -- looks "
                                       f"swapped or unguarded."})

    for (i, _value, innermost) in gnu_hits:
        if innermost is None or innermost[0] != 'else':
            issues.append({"rule": "gnu-retention-swapped-or-unscoped", "file": rel,
                            "message": f"line {i + 1}: LINKER:--undefined= is not inside the "
                                       f"else() of an if(MSVC) construct (innermost: "
                                       f"{innermost}) -- looks swapped, or the MSVC branch "
                                       f"is missing."})
        elif not re.search(r'\bMSVC\b', innermost[1] or '', re.IGNORECASE):
            issues.append({"rule": "gnu-retention-else-not-msvc-paired", "file": rel,
                            "message": f"line {i + 1}: else()'s paired if() condition is "
                                       f"'{innermost[1]}', not MSVC -- LINKER:--undefined= is "
                                       f"not the designated non-MSVC fallback."})


def check_anchor_name_consistency(issues: list):
    if not HOOK_PATH.exists() or not CMAKE_PATH.exists():
        return  # already reported by the checks above
    cpp_text = strip_c_comments(HOOK_PATH.read_text(encoding="utf-8", errors="replace"))
    cmake_text = strip_cmake_comments(CMAKE_PATH.read_text(encoding="utf-8", errors="replace"))

    cpp_m = ANCHOR_RE_CPP.search(cpp_text)
    if not cpp_m:
        issues.append({"rule": "anchor-definition-not-found",
                        "file": "src/gpu/src/gpu_engine_hook.cpp",
                        "message": "Could not find `extern \"C\" int <name> = 1;` anchor "
                                   "definition."})
        return
    cpp_name = cpp_m.group(1)

    names = {}
    m = ANCHOR_RE_CMAKE_MSVC.search(cmake_text)
    if m:
        names['msvc'] = m.group(1)
    m = ANCHOR_RE_CMAKE_GNU.search(cmake_text)
    if m:
        names['gnu (non-MSVC)'] = m.group(1)

    for label, name in names.items():
        if name != cpp_name:
            issues.append({
                "rule": "anchor-name-mismatch",
                "file": "compat/libbitcoin_direct/CMakeLists.txt",
                "message": f"{label} retention option references anchor '{name}', but "
                           f"src/gpu/src/gpu_engine_hook.cpp defines '{cpp_name}' -- the "
                           f"linker would retain the wrong (or a nonexistent) symbol.",
            })


# ---------------------------------------------------------------------------
# (d) + (e) CI fixture reachability (real YAML parse, not text grep)
# ---------------------------------------------------------------------------
def _load_workflow(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _workflow_triggers(wf: dict) -> set:
    # PyYAML's YAML-1.1 resolver parses the bare `on:` key as boolean True in
    # every workflow in this repo (verified) -- must check both.
    on = wf.get(True, wf.get('on'))
    if isinstance(on, str):
        return {on}
    if isinstance(on, list):
        return set(on)
    if isinstance(on, dict):
        return set(on.keys())
    return set()


def _workflow_is_mandatory(wf: dict) -> bool:
    return bool(_workflow_triggers(wf) & {'push', 'pull_request'})


def _runs_on_list(job: dict) -> list:
    runs_on = job.get('runs-on', '')
    return [str(x) for x in runs_on] if isinstance(runs_on, list) else [str(runs_on)]


def _job_run_blob(job: dict) -> str:
    return "\n".join(
        s.get('run', '') for s in (job.get('steps') or []) if isinstance(s.get('run'), str)
    )


def check_windows_fixture_reachable(issues: list):
    if not (LIB_ROOT / MSVC_FIXTURE_DIR).exists():
        issues.append({"rule": "msvc-fixture-missing", "file": MSVC_FIXTURE_DIR,
                        "message": f"{MSVC_FIXTURE_DIR}/ does not exist."})
        return
    if not CI_YML.exists():
        issues.append({"rule": "ci-yml-missing", "file": ".github/workflows/ci.yml",
                        "message": "ci.yml not found."})
        return
    wf = _load_workflow(CI_YML)
    mandatory = _workflow_is_mandatory(wf)
    found = False
    for _job_id, job in (wf.get('jobs') or {}).items():
        if MSVC_FIXTURE_DIR not in _job_run_blob(job):
            continue
        if any('windows' in r.lower() for r in _runs_on_list(job)):
            found = True
    if not (mandatory and found):
        issues.append({
            "rule": "msvc-fixture-unreachable",
            "file": ".github/workflows/ci.yml",
            "message": f"{MSVC_FIXTURE_DIR}/ is not built+verified by any windows-* job "
                       f"in a push/pull_request-triggered workflow.",
        })


def check_cuda_macro_fixture_reachable(issues: list):
    fixture = LIB_ROOT / CUDA_MACRO_FIXTURE
    if not fixture.exists():
        issues.append({"rule": "cuda-macro-fixture-missing", "file": CUDA_MACRO_FIXTURE,
                        "message": f"{CUDA_MACRO_FIXTURE} does not exist."})
        return
    reachable = False
    for wf_path in (CI_YML, GPU_YML):
        if not wf_path.exists():
            continue
        wf = _load_workflow(wf_path)
        mandatory = _workflow_is_mandatory(wf)
        for _job_id, job in (wf.get('jobs') or {}).items():
            blob = _job_run_blob(job)
            if CUDA_MACRO_FIXTURE not in blob:
                continue
            is_self_hosted = any('self-hosted' in r.lower() for r in _runs_on_list(job))
            has_nvcc = 'nvcc' in blob
            if mandatory and not is_self_hosted and has_nvcc:
                reachable = True
    if not reachable:
        issues.append({
            "rule": "cuda-macro-fixture-unreachable",
            "file": ".github/workflows/ci.yml",
            "message": f"{CUDA_MACRO_FIXTURE} is not compiled by any nvcc-invoking job "
                       f"in a push/pull_request-triggered, non-self-hosted workflow.",
        })


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def evaluate() -> dict:
    issues: list = []
    check_field_mul_small_param_name(issues)
    check_lbtc_retention_options(issues)
    check_anchor_name_consistency(issues)
    check_windows_fixture_reachable(issues)
    check_cuda_macro_fixture_reachable(issues)

    workflow_report = evaluate_workflow()
    for problem in workflow_report["problems"]:
        issues.append({
            "rule": f"workflow-{problem['kind']}",
            "file": str(WORKFLOW.relative_to(LIB_ROOT)),
            "message": problem["detail"],
        })

    return {
        "overall_pass": len(issues) == 0,
        "issues": issues,
        "issue_count": len(issues),
        "problems": workflow_report["problems"],
    }


def run(json_mode: bool) -> int:
    report = evaluate()
    if json_mode:
        print(json.dumps(report, indent=2))
    else:
        if report["issues"]:
            print(f"Windows/MSVC CUDA build contract violations ({report['issue_count']}):")
            for iss in report["issues"]:
                print(f"  [{iss['rule']}] {iss['file']}: {iss['message']}")
        else:
            print("PASS: PR #353 Windows/MSVC CUDA build contract holds.")
    return 0 if report["overall_pass"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--json', action='store_true', help='JSON output')
    args = parser.parse_args()
    return run(args.json)


if __name__ == '__main__':
    sys.exit(main())
