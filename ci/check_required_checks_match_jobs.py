#!/usr/bin/env python3
"""check_required_checks_match_jobs.py — CAAS-CI-001 regression gate.

Branch-protection required status checks (ci/update_required_checks.sh) must
reference REAL job display names that actually run on pull_request → main.
A required context that matches no job is either ignored (the security gate is
then NOT required at the merge boundary) or, under strict mode, blocks every
merge forever.

This gate caught CAAS-CI-001: the contexts list pinned "linux (gcc-13, …)"
(ci.yml's matrix is gcc-14) and contained ZERO gate.yml CAAS jobs, so the CAAS
security pipeline was not required to merge.

For every context in update_required_checks.sh this gate asserts:
  1. it resolves to a job display name in some .github/workflows/*.yml
     (matrix display names are expanded), AND
  2. that workflow triggers on `pull_request` (else strict mode blocks forever).

Exit 0 = every context resolves to a PR-triggered job, 1 = mismatch.
"""
import itertools
import re
import sys
from pathlib import Path

import yaml

WORKFLOWS = Path(".github/workflows")
REQUIRED_SH = Path("ci/update_required_checks.sh")


def fail(msg: str) -> None:
    print(f"::error::check_required_checks_match_jobs: {msg}")


def parse_contexts(text: str):
    """Extract the JSON-ish contexts array from the heredoc in the shell script."""
    m = re.search(r'"contexts"\s*:\s*\[(.*?)\]', text, re.S)
    if not m:
        return None
    return re.findall(r'"([^"]+)"', m.group(1))


def matrix_combos(matrix: dict):
    """Yield dicts of matrix var→value, expanding the cartesian product of
    list-valued axes and merging `include` entries (mirrors GitHub Actions)."""
    if not isinstance(matrix, dict):
        return []
    cart_keys = [k for k, v in matrix.items()
                 if k not in ("include", "exclude") and isinstance(v, list)]
    base = [dict(zip(cart_keys, vals))
            for vals in itertools.product(*[matrix[k] for k in cart_keys])] or [{}]
    for inc in matrix.get("include", []) or []:
        if not isinstance(inc, dict):
            continue
        overlap = [k for k in cart_keys if k in inc]
        if overlap:
            for combo in base:
                if all(combo.get(k) == inc[k] for k in overlap):
                    combo.update(inc)
        else:
            base.append(dict(inc))
    return base, cart_keys


def job_display_names(job_id: str, job: dict):
    """All possible status-check display names for one job (matrix-expanded)."""
    name = job.get("name")
    matrix = (job.get("strategy") or {}).get("matrix")
    if not matrix:
        return {str(name) if name else job_id}
    combos, cart_keys = matrix_combos(matrix)
    out = set()
    for combo in combos:
        if name and "${{" in str(name):
            disp = str(name)
            for k, v in combo.items():
                disp = disp.replace("${{ matrix.%s }}" % k, str(v))
                disp = disp.replace("${{matrix.%s}}" % k, str(v))
            out.add(disp)
        elif name:
            out.add(str(name))
        else:
            suffix = ", ".join(str(combo[k]) for k in cart_keys if k in combo)
            out.add(f"{job_id} ({suffix})")
    return out


def triggers_on_pr(doc: dict) -> bool:
    # YAML parses bare `on:` as the boolean key True.
    on = doc.get("on", doc.get(True))
    if on is None:
        return False
    if isinstance(on, str):
        return on == "pull_request"
    if isinstance(on, list):
        return "pull_request" in on
    if isinstance(on, dict):
        return "pull_request" in on
    return False


def main() -> int:
    if not REQUIRED_SH.exists():
        print(f"check_required_checks_match_jobs: {REQUIRED_SH} not found — run from "
              "the UltrafastSecp256k1 submodule root")
        return 1

    contexts = parse_contexts(REQUIRED_SH.read_text())
    if contexts is None:
        fail(f"could not parse a contexts array from {REQUIRED_SH}")
        return 1

    # name -> set of workflow files that emit it on a PR-triggered workflow
    pr_names = {}
    all_names = {}
    for wf in sorted(WORKFLOWS.glob("*.yml")):
        try:
            doc = yaml.safe_load(wf.read_text())
        except yaml.YAMLError as e:
            print(f"  (warn) could not parse {wf.name}: {e}")
            continue
        if not isinstance(doc, dict):
            continue
        on_pr = triggers_on_pr(doc)
        for job_id, job in (doc.get("jobs") or {}).items():
            if not isinstance(job, dict):
                continue
            for disp in job_display_names(job_id, job):
                all_names.setdefault(disp, set()).add(wf.name)
                if on_pr:
                    pr_names.setdefault(disp, set()).add(wf.name)

    errors = 0
    for ctx in contexts:
        if ctx in pr_names:
            print(f"  [ok] {ctx}")
        elif ctx in all_names:
            fail(f"context '{ctx}' matches a job in {sorted(all_names[ctx])} but that "
                 "workflow does NOT trigger on pull_request — strict mode would block "
                 "every merge forever")
            errors += 1
        else:
            fail(f"context '{ctx}' matches NO job display name in any workflow "
                 "(stale/phantom required check)")
            errors += 1

    if errors:
        print(f"check_required_checks_match_jobs: {errors} unresolved context(s)")
        return 1
    print(f"check_required_checks_match_jobs: all {len(contexts)} required contexts "
          "resolve to PR-triggered jobs [PASS]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
