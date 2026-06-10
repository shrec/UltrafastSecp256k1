#!/usr/bin/env python3
"""check_sanitizer_result_assertions.py — CAAS6-01 false-green gate.

A Valgrind memcheck CI step that runs `ctest ... -T MemCheck || true` and then
decides its verdict ONLY by grepping the produced ``MemoryChecker.*.log`` files is
fail-open: the ``|| true`` tolerates valgrind timeout exits (intended), but it also
masks the case where memcheck never produced any log at all (early crash, empty
``-E`` selection, runner error). The grep verdicts are silenced with ``2>/dev/null``,
so "memcheck silently did not run" exits 0 — indistinguishable from "memcheck ran
clean" — on a branch-protection-*required* gate.

Rule enforced here: every ``run:`` block that runs ``ctest ... -T MemCheck`` and
greps ``MemoryChecker`` logs for its verdict MUST also contain a *log-presence*
fail-closed guard, i.e. it must exit non-zero when zero MemoryChecker logs exist.

Accepted guard shapes (any one):
  * ``shopt -s nullglob`` + an array of ``MemoryChecker.*.log`` + ``... -eq 0`` +
    ``exit 1`` (the canonical fix), or
  * ``compgen -G '...MemoryChecker...'`` guarded by ``|| { ...; exit 1; }``.

Scope is deliberately narrow (``-T MemCheck`` blocks only). The TSan/ASan/MSan jobs
are already fail-closed via ``--error-exitcode`` / ctest return codes and are not
flagged.

Exit codes: 0 = all memcheck blocks are fail-closed (or none present); 1 = at least
one fail-open memcheck block found.
"""
from __future__ import annotations

import glob
import os
import re
import sys

WORKFLOW_GLOBS = (".github/workflows/*.yml", ".github/workflows/*.yaml")

_RUN_KEY = re.compile(r"^(?P<indent>\s*)(?:-\s+)?run:\s*([|>][+-]?\s*)?(?P<inline>\S.*)?$")
_MEMCHECK = re.compile(r"-T\s+MemCheck")
_GREP_VERDICT = re.compile(r"grep\b.*MemoryChecker")
# Canonical guard: nullglob array + zero-count + exit non-zero.
_GUARD_NULLGLOB = re.compile(r"nullglob")
_GUARD_ZERO = re.compile(r"-eq\s*\"?0\"?|\[\s*-z\b|compgen\s+-G")
_GUARD_EXIT = re.compile(r"exit\s+[1-9]")
_GUARD_MEMLOG = re.compile(r"MemoryChecker")


def _leading_ws(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def extract_run_blocks(text: str):
    """Yield the text of every YAML ``run:`` block (literal/folded or inline)."""
    lines = text.splitlines()
    i = 0
    n = len(lines)
    while i < n:
        m = _RUN_KEY.match(lines[i])
        if not m:
            i += 1
            continue
        inline = (m.group("inline") or "").strip()
        if inline:
            # Inline `run: ctest ...` — single logical line.
            yield inline
            i += 1
            continue
        # Block scalar: capture following lines more-indented than the `run:` key.
        key_indent = len(m.group("indent"))
        block = []
        j = i + 1
        while j < n:
            ln = lines[j]
            if ln.strip() == "":
                block.append(ln)
                j += 1
                continue
            if _leading_ws(ln) <= key_indent:
                break
            block.append(ln)
            j += 1
        yield "\n".join(block)
        i = j


def block_is_fail_closed(block: str) -> bool:
    """True if the memcheck block exits non-zero when no MemoryChecker logs exist."""
    has_nullglob = bool(_GUARD_NULLGLOB.search(block))
    has_zero = bool(_GUARD_ZERO.search(block))
    has_exit = bool(_GUARD_EXIT.search(block))
    has_memlog = bool(_GUARD_MEMLOG.search(block))
    # nullglob-array + zero-count + exit, OR compgen -G guard + exit, both over the
    # MemoryChecker logs.
    return has_memlog and has_exit and (has_nullglob or "compgen -G" in block) and has_zero


def scan_workflow_text(text: str):
    """Return a list of problem strings for one workflow file's text."""
    problems = []
    for block in extract_run_blocks(text):
        if not _MEMCHECK.search(block):
            continue
        if not _GREP_VERDICT.search(block):
            # No grep-based verdict — not the fail-open pattern this gate targets.
            continue
        if not block_is_fail_closed(block):
            problems.append(
                "memcheck (`-T MemCheck`) block decides its verdict by grepping "
                "MemoryChecker logs but has no log-presence fail-closed guard "
                "(expected `shopt -s nullglob` + `... -eq 0` + `exit 1`). "
                '"memcheck did not run" would pass green.'
            )
    return problems


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    root = argv[0] if argv else "."
    files = []
    for pat in WORKFLOW_GLOBS:
        files.extend(sorted(glob.glob(os.path.join(root, pat))))
    failed = 0
    checked = 0
    for path in files:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                text = fh.read()
        except OSError as exc:  # pragma: no cover
            print(f"  WARN: cannot read {path}: {exc}")
            continue
        if not _MEMCHECK.search(text):
            continue
        checked += 1
        for problem in scan_workflow_text(text):
            failed += 1
            print(f"  FAIL: {os.path.relpath(path, root)}: {problem}")
    if failed:
        print(
            f"\ncheck_sanitizer_result_assertions: {failed} fail-open memcheck "
            f"block(s) found. Add a log-presence guard so the gate fails closed "
            f"when memcheck produces no logs (CAAS6-01)."
        )
        return 1
    print(
        f"check_sanitizer_result_assertions: OK "
        f"({checked} workflow file(s) with `-T MemCheck` are fail-closed)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
