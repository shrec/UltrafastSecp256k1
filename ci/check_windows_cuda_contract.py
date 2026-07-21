#!/usr/bin/env python3
"""Validate that the Windows workflow compiles the real CUDA backend.

WIN-CUDA-001 protects three independent requirements: the Windows CUDA compiler
and CRT headers are installed, CMake cannot silently configure a CPU-only build,
and the workflow actually builds both GPU host and kernel targets.
"""

from __future__ import annotations

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


def evaluate() -> dict:
    try:
        document = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        problems = [_problem("workflow_unreadable", str(exc))]
    else:
        problems = evaluate_document(document)
    return {"overall_pass": not problems, "problems": problems}


def main() -> int:
    report = evaluate()
    if report["overall_pass"]:
        print("check_windows_cuda_contract: real CUDA compile workflow [PASS]")
        return 0
    for problem in report["problems"]:
        print(f"::error::check_windows_cuda_contract: {problem}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
