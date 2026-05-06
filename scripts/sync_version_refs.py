#!/usr/bin/env python3
"""
sync_version_refs.py — Update hardcoded version strings to match VERSION.txt.

Covers: docs (.md/.yml/.yaml), CITATION.cff, C ABI header (ufsecp_version.h),
        CocoaPods podspec, RPM spec, Arch PKGBUILD.

Usage:
    python3 scripts/sync_version_refs.py [--version X.Y.Z] [--dry-run] [--root DIR]
                                         [--date YYYY-MM-DD]

If --version is omitted, reads from VERSION.txt in the repo root.
--date updates date-released in CITATION.cff (default: today).
Run after bumping VERSION.txt (release CI does this automatically).
"""

import re
import sys
import os
import fnmatch
import argparse
import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# RULES_DOCS: applied to .md / .yml / .yaml / .cff files
# Replacement templates: {v} = version, {vv} = "v"+version,
#                        {major}, {minor}, {patch}
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

    # | 3.50.x  | [OK] Active |  (SECURITY.md supported-version table)
    (r'(\|\s*)[\d]+\.[\d]+\.x(\s*\|\s*\[OK\]\s*Active)', r'\g<1>{major}.{minor}.x\g<2>'),

    # *Generated from unified_audit_runner v3.4.0 output ...
    (r'(unified_audit_runner\s+v)[\d]+\.[\d]+\.[\d]+', r'\g<1>{v}'),

    # **Library Version:** UltrafastSecp256k1 v3.4.0
    (r'(\*\*Library Version:\*\*\s*UltrafastSecp256k1\s+v)[\d]+\.[\d]+\.[\d]+', r'\g<1>{v}'),

    # description: 'Existing tag to backfill (e.g. v3.4.0)'
    (r'(e\.g\.\s+v)[\d]+\.[\d]+\.[\d]+', r'\g<1>{v}'),

    # Library:    UltrafastSecp256k1 v3.4.0  (bench/audit output headers)
    (r'(Library:\s+UltrafastSecp256k1\s+v)[\d]+\.[\d]+\.[\d]+', r'\g<1>{v}'),

    # CITATION.cff: version: "3.12.2"
    (r'^(version:\s*")\d+\.\d+\.\d+(")\s*$', r'\g<1>{v}\g<2>'),
]

# ---------------------------------------------------------------------------
# RULES_PACKAGING: applied to packaging files (.podspec, .spec, PKGBUILD)
# ---------------------------------------------------------------------------
RULES_PACKAGING = [
    # CocoaPods podspec:  s.version = "3.0.0"
    (r'(s\.version\s*=\s*")\d+\.\d+\.\d+(")', r'\g<1>{v}\g<2>'),

    # RPM spec:  Version: 3.12.1  (line-start, not inside macro blocks)
    (r'^(Version:\s+)\d+\.\d+\.\d+', r'\g<1>{v}'),

    # RPM spec:  %global soversion 3  → major version
    (r'^(%global\s+soversion\s+)\d+', r'\g<1>{major}'),

    # Arch PKGBUILD:  pkgver=3.12.1
    (r'^(pkgver=)\d+\.\d+\.\d+', r'\g<1>{v}'),
]

# ---------------------------------------------------------------------------
# RULES_CHEADER: applied to include/ufsecp/ufsecp_version.h only
# ---------------------------------------------------------------------------
RULES_CHEADER = [
    (r'^(#define\s+UFSECP_VERSION_MAJOR\s+)\d+', r'\g<1>{major}'),
    (r'^(#define\s+UFSECP_VERSION_MINOR\s+)\d+', r'\g<1>{minor}'),
    (r'^(#define\s+UFSECP_VERSION_PATCH\s+)\d+', r'\g<1>{patch}'),
    (r'^(#define\s+UFSECP_VERSION_STRING\s+")\d+\.\d+\.\d+(")', r'\g<1>{v}\g<2>'),
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
        # Use path-component matching to avoid '.git' matching '.github'
        if any(part == skip or part.startswith(skip) and skip.endswith('-')
               for part in Path(rel).parts):
            return True
    if path.name in SKIP_FILES:
        return True
    for pat in SKIP_PATTERNS:
        if fnmatch.fnmatch(path.name, pat):
            return True
    return False


def apply_rules(content: str, version: str, rules=None) -> tuple[str, int]:
    """Apply version-replacement rules. Returns (new_content, change_count)."""
    if rules is None:
        rules = RULES
    parts = version.split('.')
    major = parts[0] if len(parts) > 0 else '0'
    minor = parts[1] if len(parts) > 1 else '0'
    patch = parts[2] if len(parts) > 2 else '0'
    changes = 0
    lines = content.split('\n')
    new_lines = []
    for line in lines:
        new_line = line
        for pattern, repl in rules:
            repl_filled = (repl
                           .replace('{v}', version)
                           .replace('{vv}', 'v' + version)
                           .replace('{major}', major)
                           .replace('{minor}', minor)
                           .replace('{patch}', patch))
            result = re.sub(pattern, repl_filled, new_line, flags=re.MULTILINE)
            if result != new_line:
                changes += 1
                new_line = result
        new_lines.append(new_line)
    return '\n'.join(new_lines), changes


def _update_file(path: Path, version: str, rules, dry_run: bool) -> int:
    """Apply rules to a single file. Returns number of replacements made."""
    try:
        original = path.read_text(encoding='utf-8')
    except Exception:
        return 0
    updated, changes = apply_rules(original, version, rules)
    if changes:
        if dry_run:
            print(f'  [dry-run] {path}: {changes} replacement(s)')
        else:
            path.write_text(updated, encoding='utf-8')
            print(f'  {path}: {changes} replacement(s)')
    return changes


def sync_version(root: Path, version: str, dry_run: bool, release_date: str | None = None) -> int:
    total_files = 0

    # ── 1. Docs: .md / .yml / .yaml / .cff in docs/, include/, .github/, root ──
    scan_dirs = [root / 'docs', root / 'include', root / '.github', root]
    target_suffixes = {'.md', '.yml', '.yaml', '.cff'}

    visited: set[Path] = set()
    for scan_dir in scan_dirs:
        if not scan_dir.exists():
            continue
        for suffix in target_suffixes:
            glob_pat = f'*{suffix}' if scan_dir == root else f'**/*{suffix}'
            for path in sorted(scan_dir.glob(glob_pat)):
                if path in visited:
                    continue
                visited.add(path)
                if should_skip_path(path, root):
                    continue
                if _update_file(path, version, RULES, dry_run):
                    total_files += 1

    # ── 2. CITATION.cff date-released (only when --date supplied) ─────────────
    cff = root / 'CITATION.cff'
    if cff.exists() and release_date:
        try:
            text = cff.read_text(encoding='utf-8')
            updated = re.sub(
                r'^(date-released:\s*")\d{4}-\d{2}-\d{2}(")',
                rf'\g<1>{release_date}\g<2>',
                text, flags=re.MULTILINE,
            )
            if updated != text:
                if dry_run:
                    print(f'  [dry-run] CITATION.cff: date-released → {release_date}')
                else:
                    cff.write_text(updated, encoding='utf-8')
                    print(f'  CITATION.cff: date-released → {release_date}')
        except Exception:
            pass

    # ── 3. C ABI version header (static copy, non-CMake consumers) ────────────
    header = root / 'include' / 'ufsecp' / 'ufsecp_version.h'
    if header.exists():
        if _update_file(header, version, RULES_CHEADER, dry_run):
            total_files += 1

    # ── 4. Packaging: CocoaPods, RPM, Arch ────────────────────────────────────
    packaging_targets = [
        # (glob_pattern_relative_to_packaging, rules)
        ('cocoapods/*.podspec', RULES_PACKAGING),
        ('rpm/*.spec',          RULES_PACKAGING),
        ('arch/PKGBUILD',       RULES_PACKAGING),
    ]
    packaging_dir = root / 'packaging'
    if packaging_dir.exists():
        for glob_pat, rules in packaging_targets:
            for path in sorted(packaging_dir.glob(glob_pat)):
                if _update_file(path, version, rules, dry_run):
                    total_files += 1

    return total_files


def main():
    parser = argparse.ArgumentParser(
        description='Sync version strings in docs, headers, and packaging to VERSION.txt')
    parser.add_argument('--version', help='Version string (default: read from VERSION.txt)')
    parser.add_argument('--date',
                        help='Release date for CITATION.cff date-released (YYYY-MM-DD). '
                             'Default: today. Pass --no-date to skip.')
    parser.add_argument('--no-date', action='store_true',
                        help='Do not update date-released in CITATION.cff')
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

    if not re.match(r'^\d+\.\d+(\.\d+)?$', version):
        print(f'ERROR: invalid version format: {version!r}', file=sys.stderr)
        sys.exit(1)

    if args.no_date:
        release_date = None
    elif args.date:
        release_date = args.date
    else:
        release_date = datetime.date.today().isoformat()

    print(f'Syncing version: {version}  date: {release_date or "(skip)"}  '
          f'dry_run={args.dry_run}')
    n = sync_version(root, version, args.dry_run, release_date)
    print(f'Done: {n} file(s) updated.')


if __name__ == '__main__':
    main()
