#!/usr/bin/env python3
"""check_workflow_trigger_claims.py — CLAIM-07 regression gate.

Several CT/audit GitHub Actions workflows are `workflow_dispatch:`-only (manual /
release-only), NOT push/PR triggered. Reviewer docs must not claim they run on
every commit. This gate caught CLAIM-07: docs/AUDIT_COVERAGE.md labelled the CT
verification workflows "push/PR" and "(all CI-enforced)", contradicting the
already-corrected README/WHY wording.

Method (run from repo root):
  1. Parse each .github/workflows/*.yml `on:` block. A workflow is "dispatch-only"
     if its on: triggers contain workflow_dispatch but NO push/pull_request/schedule.
  2. In a fixed set of reviewer docs, flag any line that names a dispatch-only
     workflow's stem (e.g. "ct-verif") AND makes a positive push/CI-enforced claim
     ("push/PR", "on push", "CI-enforced", "every commit"), UNLESS the same line is
     explicitly qualified ("workflow_dispatch", "manual", "not on", "release tag").

Exit 0 = OK, 1 = a doc mislabels a dispatch-only workflow.
"""
import re
import sys
from pathlib import Path

WF_DIR = Path(".github/workflows")
DOCS = [
    "README.md",
    "docs/AUDIT_COVERAGE.md",
    "docs/WHY_ULTRAFASTSECP256K1.md",
    "docs/BITCOIN_CORE_BACKEND_EVIDENCE.md",
    # P7-DOC-001: per-workflow assurance doc — CT_INDEPENDENCE.md §5 previously
    # claimed push/PR/weekly triggers while ct-independence.yml is dispatch-only.
    "docs/CT_INDEPENDENCE.md",
]
POSITIVE_CLAIM = re.compile(
    r"push/PR|push/pr|on push\b|CI-enforced|CI enforced|every commit|on every commit",
    re.IGNORECASE,
)
QUALIFIER = re.compile(
    r"workflow_dispatch|manual|not on|release[ -]?tag|dispatch", re.IGNORECASE
)


def dispatch_only_workflows():
    out = []
    for wf in sorted(WF_DIR.glob("*.yml")):
        txt = wf.read_text()
        i = txt.find("\non:")
        if i < 0 and txt.startswith("on:"):
            i = 0
        if i < 0:
            continue
        j = txt.find("\njobs:", i)
        on_block = txt[i: j if j > i else len(txt)]
        has_dispatch = "workflow_dispatch" in on_block
        has_auto = bool(re.search(r"\bpush:|\bpull_request:|\bpull_request_target:|\bschedule:",
                                  on_block))
        if has_dispatch and not has_auto:
            out.append(wf.stem)
    return out


def main() -> int:
    if not WF_DIR.is_dir():
        print("check_workflow_trigger_claims: no .github/workflows — run from repo root")
        return 1

    dispatch_only = dispatch_only_workflows()
    print(f"check_workflow_trigger_claims: dispatch-only workflows: "
          f"{', '.join(dispatch_only) or '(none)'}")

    errors = 0
    for doc in DOCS:
        p = Path(doc)
        if not p.exists():
            continue
        for lineno, line in enumerate(p.read_text().splitlines(), 1):
            if not POSITIVE_CLAIM.search(line):
                continue
            if QUALIFIER.search(line):
                continue
            for stem in dispatch_only:
                # Match only an UNAMBIGUOUS reference: the workflow filename
                # (`<stem>.yml`), or a hyphenated stem as a whole word. Single-word
                # generic stems (docs, bindings, mutation, tsan, ...) collide with
                # prose, so they count only via the `.yml` form.
                hit = re.search(rf"\b{re.escape(stem)}\.yml\b", line)
                if not hit and "-" in stem:
                    hit = re.search(rf"(?<![\w-]){re.escape(stem)}(?![\w-])", line)
                if hit:
                    print(f"::error::{doc}:{lineno} labels dispatch-only workflow "
                          f"'{stem}' with a push/CI-enforced claim: {line.strip()[:140]}")
                    errors += 1
                    break

    if errors:
        print(f"check_workflow_trigger_claims: {errors} mislabel(s) — a "
              f"workflow_dispatch-only workflow is documented as push/PR/CI-enforced. "
              f"Use 'manual (workflow_dispatch)' / 'not on every commit'.")
        return 1
    print("check_workflow_trigger_claims: OK — no doc mislabels a dispatch-only workflow.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
