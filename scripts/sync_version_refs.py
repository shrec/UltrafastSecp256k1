#!/usr/bin/env python3
"""
sync_version_refs.py — Update hardcoded version strings in docs to match VERSION.txt.

Usage:
    python3 scripts/sync_version_refs.py [--version X.Y.Z] [--dry-run] [--root DIR]

If --version is omitted, reads from VERSION.txt in the repo root.
Run after bumping VERSION.txt (release CI does this automatically).
"""

import re
import sys
import os
import fnmatch
import argparse
from pathlib import Path

# ---------------------------------------------------------------------------
# Patterns: (regex, replacement_template)
#
# Replacement templates use Python format strings where {v} = new version,
# {vv} = "v" + new version (e.g. "v3.50.0").
#
# Rules are tried in order on every line of every .md file.
# ---------------------------------------------------------------------------
RULES = [
    # > **Version**: 3.4.0
    (r'(>\s*\*\*Version\*\*:\s*)[\d]+\.[\d]+\.[\d]+', r'\g<1>{v}'),

    # **Last updated**: ... | **Version**: 3.4.0
    (r'(\*\*Version\*\*:\s*)[\d]+\.[\d]+\.[\d]+', r'\g<1>{v}'),

    # **UltrafastSecp256k1 v3.4.0** -- ...
    (r'(\*\*UltrafastSecp256k1\s+v)[\d]+\.[\d]+\.[\d]+(\*\*)', r'\g<1>{v}\g<2>'),

    # *UltrafastSecp256k1 v3.4.0 -- ...  (footer lines)
    (r'(\*UltrafastSecp256k1\s+v)[\d]+\.[\d]+\.[\d]+(\s+--)', r'\g<1>{v}\g<2>'),

    # | **Library Version** | 3.4.0 |
    (r'(\|\s*\*\*Library Version\*\*\s*\|\s*)[\d]+\.[\d]+\.[\d]+(\s*\|)', r'\g<1>{v}\g<2>'),

    # | Version | 3.4.0 |   (only when row starts with "| Version |")
    (r'(\|\s*Version\s*\|\s*)[\d]+\.[\d]+\.[\d]+(\s*\|)', r'\g<1>{v}\g<2>'),

    # UltrafastSecp256k1 v3.4.0  (standalone line / sentence)
    (r'(UltrafastSecp256k1\s+v)[\d]+\.[\d]+\.[\d]+', r'\g<1>{v}'),

    # conan install ultrafastsecp256k1/3.4.0@
    (r'(ultrafastsecp256k1/)[\d]+\.[\d]+\.[\d]+(@)', r'\g<1>{v}\g<2>'),

    # GIT_TAG v3.4.0
    (r'(GIT_TAG\s+v)[\d]+\.[\d]+\.[\d]+', r'\g<1>{v}'),

    # go get github.com/.../ufsecp@v3.4.0
    (r'(ufsecp@v)[\d]+\.[\d]+\.[\d]+', r'\g<1>{v}'),

    # <PackageReference Include="UltrafastSecp256k1" Version="3.4.0" />
    (r'(Include="UltrafastSecp256k1"\s+Version=")[\d]+\.[\d]+\.[\d]+(")', r'\g<1>{v}\g<2>'),

    # <version>3.4.0</version>  (Maven / NuGet .nuspec)
    (r'(<version>)[\d]+\.[\d]+\.[\d]+(</version>)', r'\g<1>{v}\g<2>'),

    # ufsecp = { version = "3.4.0"  (Cargo.toml example)
    (r'(ufsecp\s*=\s*\{\s*version\s*=\s*")[\d]+\.[\d]+\.[\d]+(")', r'\g<1>{v}\g<2>'),

    # pkg-config --modversion ufsecp   # -> 3.4.0
    (r'(pkg-config --modversion ufsecp\s+#\s*->\s*)[\d]+\.[\d]+\.[\d]+', r'\g<1>{v}'),

    # "library_version": "3.4.0"  (JSON examples in docs)
    (r'("library_version":\s*")[\d]+\.[\d]+\.[\d]+(")', r'\g<1>{v}\g<2>'),

    # Version: 3.4.0  (plain key: value in tables without bold)
    (r'^(Version:\s*)[\d]+\.[\d]+\.[\d]+\s*$', r'\g<1>{v}'),
]

# Files to skip entirely (binary, build artifacts, historical records)
SKIP_DIRS = {
    'build', 'build-', '_CPack', '.git', 'node_modules',
    'audit-output', 'benchmarks/comparison',
    'archive',           # Historical release records — intentional
}

# Files to explicitly skip
SKIP_FILES = {
    'CHANGELOG.md',      # Historical record — intentional
    'ROADMAP.md',        # Future versions — intentional
    'AUDIT_REPORT.md',   # Audit baseline — intentional
}

# File name patterns to skip (fnmatch)
SKIP_PATTERNS = [
    '_release_notes_*.md',   # Historical release notes at repo root
    'RELEASE_NOTES_*.md',
    'RELEASE_v*.md',
]


def should_skip_path(path: Path, root: Path) -> bool:
    rel = str(path.relative_to(root))
    for skip in SKIP_DIRS:
        if skip in rel:
            return True
    if path.name in SKIP_FILES:
        return True
    for pat in SKIP_PATTERNS:
        if fnmatch.fnmatch(path.name, pat):
            return True
    return False


def apply_rules(content: str, version: str) -> tuple[str, int]:
    """Apply all version-replacement rules. Returns (new_content, change_count)."""
    changes = 0
    lines = content.split('\n')
    new_lines = []
    for line in lines:
        new_line = line
        for pattern, repl in RULES:
            repl_filled = repl.replace('{v}', version).replace('{vv}', 'v' + version)
            result = re.sub(pattern, repl_filled, new_line)
            if result != new_line:
                changes += 1
                new_line = result
        new_lines.append(new_line)
    return '\n'.join(new_lines), changes


def sync_version(root: Path, version: str, dry_run: bool) -> int:
    total_changed = 0
    total_files = 0

    # Scan .md files in docs/, include/, root
    scan_dirs = [root / 'docs', root / 'include', root]
    target_suffixes = {'.md'}

    visited = set()
    for scan_dir in scan_dirs:
        if not scan_dir.exists():
            continue
        pattern = '*.md' if scan_dir == root else '**/*.md'
        for path in sorted(scan_dir.glob(pattern)):
            if path in visited:
                continue
            visited.add(path)
            if should_skip_path(path, root):
                continue
            try:
                original = path.read_text(encoding='utf-8')
            except Exception:
                continue
            updated, changes = apply_rules(original, version)
            if changes:
                total_changed += changes
                total_files += 1
                if dry_run:
                    print(f'  [dry-run] {path.relative_to(root)}: {changes} replacement(s)')
                else:
                    path.write_text(updated, encoding='utf-8')
                    print(f'  {path.relative_to(root)}: {changes} replacement(s)')

    return total_files


def main():
    parser = argparse.ArgumentParser(description='Sync version strings in docs to VERSION.txt')
    parser.add_argument('--version', help='Version string (default: read from VERSION.txt)')
    parser.add_argument('--dry-run', action='store_true', help='Show changes without writing')
    parser.add_argument('--root', default=None, help='Repo root (default: parent of scripts/)')
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    root = Path(args.root) if args.root else script_dir.parent

    if args.version:
        version = args.version.lstrip('v')
    else:
        version_file = root / 'VERSION.txt'
        if not version_file.exists():
            print(f'ERROR: VERSION.txt not found at {version_file}', file=sys.stderr)
            sys.exit(1)
        version = version_file.read_text().strip().lstrip('v')

    if not re.match(r'^\d+\.\d+\.\d+$', version):
        print(f'ERROR: invalid version format: {version!r}', file=sys.stderr)
        sys.exit(1)

    print(f'Syncing version: {version}  (dry_run={args.dry_run})')
    n = sync_version(root, version, args.dry_run)
    print(f'Done: {n} file(s) updated.')


if __name__ == '__main__':
    main()
