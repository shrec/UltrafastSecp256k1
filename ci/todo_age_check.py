#!/usr/bin/env python3
"""
todo_age_check.py — TODO/FIXME age SLA gate.

For each TODO / FIXME / XXX / HACK marker in tracked source files compute the
introduction age via `git blame` and fail if any marker exceeds the SLA
configured in `docs/AUDIT_SLA.json` under
`audit_sla.todo_max_open_days` without an explicit deferral marker
(the literal token `DEFERRED:` or a `(tracked: ...)` annotation on the same
line).

Implements CAAS hardening item H-8 (see `docs/CAAS_HARDENING_TODO.md`).

Usage:
    python3 ci/todo_age_check.py [--json] [-o REPORT.json] [--threshold N]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent

# Match TODO/FIXME/XXX/HACK markers as whole words.
_TODO_RE = re.compile(r'\b(TODO|FIXME|XXX|HACK)\b', re.IGNORECASE)

# Source extensions to scan. Conservative — skip binary, build, generated dirs.
_EXTS = {'.c', '.cc', '.cpp', '.cxx', '.h', '.hh', '.hpp', '.hxx',
         '.cu', '.cuh', '.cl', '.metal', '.py', '.sh', '.yml', '.yaml',
         '.toml', '.md'}

_SKIP_DIRS = {'build', 'build_rel', 'build-arm64', 'build-cuda',
              'build-riscv-rel', 'build_opencl', 'build_ufsecp_swift',
              'build-ct-arm64', '.git', '.cache', 'node_modules',
              '_research_repos', 'tools/source_graph_kit'}


def _git(*args: str) -> str:
    return subprocess.check_output(
        ['git', '-C', str(REPO_ROOT), *args],
        text=True, stderr=subprocess.DEVNULL,
    )


def _tracked_files() -> list[Path]:
    out = _git('ls-files')
    files: list[Path] = []
    for line in out.splitlines():
        if not line:
            continue
        skip = False
        for d in _SKIP_DIRS:
            if line == d or line.startswith(d + '/'):
                skip = True
                break
        if skip:
            continue
        p = REPO_ROOT / line
        if p.suffix.lower() in _EXTS:
            files.append(p)
    return files


def _blame_age_days(path: Path, line_no: int) -> int | None:
    """Return age in days for the line as of HEAD, or None on error."""
    try:
        rel = path.relative_to(REPO_ROOT)
        out = _git('blame', '-p', '-L', f'{line_no},{line_no}', '--', str(rel))
    except subprocess.CalledProcessError:
        return None
    for line in out.splitlines():
        if line.startswith('author-time '):
            ts = int(line.split()[1])
            return max(0, int((time.time() - ts) // 86400))
    return None


def _is_deferred(snippet: str) -> bool:
    s = snippet.lower()
    return ('deferred:' in s) or ('(tracked:' in s) or ('(tracking:' in s)


def _scan() -> Iterable[dict[str, Any]]:
    for path in _tracked_files():
        try:
            text = path.read_text(encoding='utf-8', errors='replace')
        except OSError:
            continue
        for i, line in enumerate(text.splitlines(), start=1):
            if not _TODO_RE.search(line):
                continue
            snippet = line.strip()
            if _is_deferred(snippet):
                continue
            age = _blame_age_days(path, i)
            yield {
                'file': str(path.relative_to(REPO_ROOT)),
                'line': i,
                'age_days': age,
                'snippet': snippet[:200],
                'deferred': False,
            }


def _load_threshold() -> int:
    sla_path = REPO_ROOT / 'docs' / 'AUDIT_SLA.json'
    try:
        sla = json.loads(sla_path.read_text())
        slo = sla.get('audit_sla', {}).get('todo_max_open_days', {})
        v = slo.get('threshold')
        if isinstance(v, int) and v > 0:
            return v
    except (OSError, json.JSONDecodeError, KeyError):
        pass
    return 180


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--json', action='store_true', help='emit JSON to stdout')
    p.add_argument('-o', '--output', help='write JSON to file')
    p.add_argument('--threshold', type=int,
                   help='override SLA threshold in days')
    args = p.parse_args()

    threshold = args.threshold if args.threshold is not None else _load_threshold()

    findings = list(_scan())
    over = [f for f in findings
            if f['age_days'] is not None and f['age_days'] > threshold]

    report = {
        'tool': 'todo_age_check',
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'threshold_days': threshold,
        'total_markers': len(findings),
        'over_sla': len(over),
        'findings_over_sla': over,
    }

    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2))
        print(f'todo_age_check: report written to {args.output}', file=sys.stderr)
    if args.json:
        json.dump(report, sys.stdout, indent=2)
        print()
    else:
        print(f'todo_age_check: {len(findings)} TODO/FIXME markers, '
              f'{len(over)} over SLA ({threshold} days)')
        for f in over[:20]:
            print(f"  [{f['age_days']:>4}d] {f['file']}:{f['line']}  {f['snippet'][:80]}")

    return 1 if over else 0


if __name__ == '__main__':
    sys.exit(main())
