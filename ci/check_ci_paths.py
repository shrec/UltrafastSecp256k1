#!/usr/bin/env python3
"""
check_ci_paths.py — Validate all paths referenced in CI workflows exist.

Scans all .github/workflows/*.yml and ci/*.sh for references to scripts,
files, and directories. Reports any that don't exist in the repo.

Prevents the #1 cause of CI breakage: "No such file or directory" in GitHub
Actions after a local refactor that moved a script.

Usage:
    python3 ci/check_ci_paths.py          # report, exit 1 if missing
    python3 ci/check_ci_paths.py --fix    # show fix hints
"""

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"
CI_DIR = REPO_ROOT / "ci"

# Patterns that reference file paths in workflow YAML
YAML_PATH_PATTERNS = [
    # bash script calls: ./ci/foo.sh, bash ci/foo.sh, python3 ci/foo.py
    re.compile(r'(?:bash|sh|python3?|perl|node)\s+([./]?(?:ci|scripts|tools)/[\w./\-]+\.(?:sh|py|js))'),
    # chmod +x path
    re.compile(r'chmod\s+\+x\s+([./]?(?:ci|scripts|tools)/[\w./\-]+\.sh)'),
    # run: ./path style
    re.compile(r'run:\s*\|\s*\n\s+\./([ci|scripts|tools]/[\w./\-]+\.sh)'),
]

# Patterns in shell scripts
SHELL_PATH_PATTERNS = [
    re.compile(r'python3\s+"?\$\{?[A-Z_]*\}?([./]?(?:ci|scripts|tools)/[\w./\-]+\.py)"?'),
    re.compile(r'bash\s+"?([./]?(?:ci|scripts|tools)/[\w./\-]+\.sh)"?'),
]

# Known false positives (generated at runtime, not in repo)
SKIP_PATTERNS = {
    'ci-impact.json',
    'build/',
    'out/',
}

def check_path(path_str: str, source_file: Path) -> tuple[str, bool]:
    """Check if a referenced path exists. Returns (canonical_path, exists)."""
    if not path_str or any(s in path_str for s in SKIP_PATTERNS):
        return path_str, True

    # Strip leading ./
    clean = path_str.lstrip('./')

    # Try relative to repo root
    candidate = REPO_ROOT / clean
    if candidate.exists():
        return clean, True

    return clean, False

def scan_workflows() -> list[dict]:
    missing = []
    for yml in sorted(WORKFLOWS_DIR.glob("*.yml")):
        text = yml.read_text(errors='replace')
        for pattern in YAML_PATH_PATTERNS:
            for match in pattern.finditer(text):
                path_ref = match.group(1)
                canonical, exists = check_path(path_ref, yml)
                if not exists:
                    # Find line number
                    lines = text[:match.start()].count('\n') + 1
                    missing.append({
                        'source': str(yml.relative_to(REPO_ROOT)),
                        'line': lines,
                        'path': canonical,
                        'raw': path_ref,
                    })
    return missing

def scan_ci_scripts() -> list[dict]:
    missing = []
    for sh in sorted(CI_DIR.glob("*.sh")):
        text = sh.read_text(errors='replace')
        for pattern in SHELL_PATH_PATTERNS:
            for match in pattern.finditer(text):
                path_ref = match.group(1)
                canonical, exists = check_path(path_ref, sh)
                if not exists:
                    lines = text[:match.start()].count('\n') + 1
                    missing.append({
                        'source': str(sh.relative_to(REPO_ROOT)),
                        'line': lines,
                        'path': canonical,
                        'raw': path_ref,
                    })
    return missing

def main() -> int:
    fix_mode = '--fix' in sys.argv

    missing = scan_workflows() + scan_ci_scripts()

    if not missing:
        print("✓ All CI path references exist")
        return 0

    print(f"✗ {len(missing)} missing CI path reference(s):\n")
    for m in missing:
        print(f"  {m['source']}:{m['line']}")
        print(f"    referenced: {m['path']}")
        if fix_mode:
            # Try to find the file somewhere in the repo
            name = Path(m['path']).name
            candidates = list(REPO_ROOT.rglob(name))
            if candidates:
                rel = [str(c.relative_to(REPO_ROOT)) for c in candidates[:3]]
                print(f"    hint: file exists at: {', '.join(rel)}")
        print()

    return 1

if __name__ == '__main__':
    sys.exit(main())
