#!/usr/bin/env python3
"""
generate_abi_negative_tests.py -- Build a hostile-caller coverage manifest for
the public C ABI.

This script scans public headers, existing hostile-caller documentation, and
project-graph test mappings to build a machine-readable manifest showing which
exports satisfy the four-part hostile-caller rule:

  1. NULL rejection for relevant pointer parameters
  2. Zero-count / zero-length / zero-key rejection where the contract implies it
  3. Invalid-content rejection for structured inputs and selectors
  4. Success-smoke coverage with valid inputs

Outputs:
  - docs/ABI_NEGATIVE_TEST_MANIFEST.json
  - docs/ABI_NEGATIVE_TEST_MANIFEST.md
"""

from __future__ import annotations

import json
import re
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
LIB_ROOT = SCRIPT_DIR.parent
DB_PATH = LIB_ROOT / ".project_graph.db"
DOC_PATH = LIB_ROOT / "docs" / "FFI_HOSTILE_CALLER.md"
JSON_OUT = LIB_ROOT / "docs" / "ABI_NEGATIVE_TEST_MANIFEST.json"
MD_OUT = LIB_ROOT / "docs" / "ABI_NEGATIVE_TEST_MANIFEST.md"

HEADERS = [
    LIB_ROOT / "include" / "ufsecp" / "ufsecp.h",
    LIB_ROOT / "include" / "ufsecp" / "ufsecp_gpu.h",
]

CHECK_NULL = "null_rejection"
CHECK_ZERO = "zero_edge"
CHECK_INVALID = "invalid_content"
CHECK_SMOKE = "success_smoke"
ALL_CHECKS = [CHECK_NULL, CHECK_ZERO, CHECK_INVALID, CHECK_SMOKE]
TEST_EVIDENCE_GLOBS = [
    'audit/test_*.cpp',
    'audit/*.cpp',
    'tests/test_*.cpp',
    'tests/*.cpp',
    'src/cpu/tests/test_*.cpp',
    'src/cpu/tests/*.cpp',
]


@dataclass
class ExportedFunction:
    name: str
    header: str
    signature: str
    params: list[str]
    category: str


def _connect() -> sqlite3.Connection:
    con = sqlite3.connect(str(DB_PATH))
    con.row_factory = sqlite3.Row
    return con


def _normalize_signature(sig: str) -> str:
    return " ".join(sig.replace("\n", " ").split())


def _split_params(param_blob: str) -> list[str]:
    params = []
    depth = 0
    current = []
    for char in param_blob:
        if char == ',' and depth == 0:
            token = ''.join(current).strip()
            if token and token != 'void':
                params.append(token)
            current = []
            continue
        if char in '([':
            depth += 1
        elif char in ')]' and depth > 0:
            depth -= 1
        current.append(char)
    token = ''.join(current).strip()
    if token and token != 'void':
        params.append(token)
    return params


def scan_public_exports() -> list[ExportedFunction]:
    decl_re = re.compile(r'UFSECP_API\s+([^;{]+?\))\s*;', re.S)
    fn_re = re.compile(r'(ufsecp_\w+)\s*\((.*)\)')
    exports = []
    for header in HEADERS:
        text = header.read_text(encoding='utf-8', errors='replace')
        for match in decl_re.finditer(text):
            signature = _normalize_signature(match.group(1))
            fn_match = fn_re.search(signature)
            if not fn_match:
                continue
            name = fn_match.group(1)
            params = _split_params(fn_match.group(2))
            category = 'gpu' if '_gpu_' in name else 'cpu'
            exports.append(ExportedFunction(
                name=name,
                header=str(header.relative_to(LIB_ROOT)).replace('\\', '/'),
                signature=signature,
                params=params,
                category=category,
            ))
    return sorted(exports, key=lambda item: item.name)


def _doc_mentions() -> dict[str, dict[str, list[str]]]:
    text = DOC_PATH.read_text(encoding='utf-8', errors='replace')
    sections: dict[str, dict[str, list[str]]] = {}
    current = 'general'
    section_re = re.compile(r'^##\s+(.*)$')
    fn_re = re.compile(r'`(ufsecp_[^`]+)`')
    for line in text.splitlines():
        section_match = section_re.match(line.strip())
        if section_match:
            current = section_match.group(1).strip()
            sections.setdefault(current, {})
            continue
        if '|' not in line:
            continue
        functions = fn_re.findall(line)
        if not functions:
            continue
        coverage_text = line.lower()
        for name in functions:
            evidence = sections.setdefault(current, {}).setdefault(name, [])
            evidence.append(coverage_text)
    return sections


def _targets_from_graph(con: sqlite3.Connection) -> dict[str, list[str]]:
    rows = con.execute(
        "SELECT function_name, test_target FROM function_test_map WHERE function_name LIKE 'ufsecp_%'"
    ).fetchall()
    mapped: dict[str, set[str]] = {}
    for row in rows:
        mapped.setdefault(row['function_name'], set()).add(row['test_target'])
    return {name: sorted(targets) for name, targets in mapped.items()}


def _requires_zero_edge(fn: ExportedFunction) -> bool:
    if fn.name in {
        'ufsecp_gpu_backend_count',
        'ufsecp_gpu_device_info',
        'ufsecp_gpu_ctx_create',
        'ufsecp_bip324_create',
        'ufsecp_btc_message_hash',
        'ufsecp_btc_message_sign',
        'ufsecp_btc_message_verify',
        'ufsecp_pubkey_tweak_add',
    }:
        return False

    zero_tokens = ('count', 'len', 'size', 'entropy_bytes')
    key_tokens = ('privkey', 'tweak', 'seckey', 'scalar', 'nonce', 'seed', 'entropy')
    lowered_params = ' '.join(fn.params).lower()
    if any(token in lowered_params for token in zero_tokens):
        return True
    if any(token in lowered_params for token in key_tokens):
        return True
    if fn.name.endswith(('_batch', '_create', '_derive_path')):
        return True
    return False


def _requires_invalid_content(fn: ExportedFunction) -> bool:
    content_tokens = (
        'pubkey', 'sig', 'path', 'wif', 'mnemonic', 'address', 'script',
        'cipher', 'envelope', 'nonce', 'proof', 'session', 'message', 'msg32',
        'backend_id', 'network', 'recid', 'input'
    )
    lowered_params = ' '.join(fn.params).lower()
    if any(token in lowered_params for token in content_tokens):
        return True
    if any(token in fn.name for token in ('parse', 'verify', 'decode', 'decrypt', 'derive', 'validate')):
        return True
    return False


def _requires_null(fn: ExportedFunction) -> bool:
    return any('*' in param or '[' in param for param in fn.params)


def _required_checks(fn: ExportedFunction) -> list[str]:
    required = [CHECK_SMOKE]
    if _requires_null(fn):
        required.append(CHECK_NULL)
    if _requires_zero_edge(fn):
        required.append(CHECK_ZERO)
    if _requires_invalid_content(fn):
        required.append(CHECK_INVALID)
    return required


def _mark(checks: dict[str, list[str]], check_name: str, evidence: str) -> None:
    checks.setdefault(check_name, [])
    if evidence not in checks[check_name]:
        checks[check_name].append(evidence)


def _check_keywords(text: str) -> set[str]:
    lowered = text.lower()
    hits = set()
    if 'null' in lowered:
        hits.add(CHECK_NULL)
    if any(token in lowered for token in (
        'zero', 'count=0', 'n=0', 'zero-length', 'undersized', 'empty',
        'minimum-size', 'short_seed', 'order_n'
    )):
        hits.add(CHECK_ZERO)
    if any(token in lowered for token in ('invalid', 'bad', 'wrong', 'tampered', 'truncated', 'off-curve', 'oob', 'reject', 'malformed', 'hostile', 'neg-', 'run_neg', 'negative')):
        hits.add(CHECK_INVALID)
    if '0xffffffff' in lowered:
        hits.add(CHECK_INVALID)
    if any(token in lowered for token in ('backend_name(', 'is_available(', 'device_count(', 'ctx_create(', 'device_info(')) and any(token in lowered for token in ('99', '255', 'oob')):
        hits.add(CHECK_INVALID)
    if any(token in lowered for token in ('smoke', 'round-trip', 'roundtrip', 'determinism', 'valid', 'succeeds', 'accepts', 'lifecycle', 'check_ok(', '== ufsecp_ok')):
        hits.add(CHECK_SMOKE)
    return hits


def _scan_direct_test_evidence() -> dict[str, dict[str, list[str]]]:
    fn_def_re = re.compile(r'^\s*(?:static\s+)?(?:inline\s+)?(?:void|bool|int|size_t|ufsecp_error_t)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\([^;]*\)\s*\{?')
    call_re = re.compile(r'\b(ufsecp_\w+)\s*\(')
    evidence: dict[str, dict[str, list[str]]] = {}

    files = []
    for pattern in TEST_EVIDENCE_GLOBS:
        files.extend(LIB_ROOT.glob(pattern))

    for path in sorted({item.resolve() for item in files if item.is_file()}):
        rel_path = str(path.relative_to(LIB_ROOT)).replace('\\', '/')
        lines = path.read_text(encoding='utf-8', errors='replace').splitlines()
        current_fn = None
        brace_depth = 0
        recent_comments: list[str] = []
        for line_no, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith('//'):
                recent_comments.append(stripped)
                recent_comments = recent_comments[-3:]
            elif stripped:
                recent_comments = recent_comments[-3:]

            fn_match = fn_def_re.match(line)
            if fn_match:
                current_fn = fn_match.group(1)
                brace_depth = line.count('{') - line.count('}')
                if brace_depth <= 0:
                    brace_depth = 1
            elif current_fn is not None:
                brace_depth += line.count('{') - line.count('}')
                if brace_depth <= 0:
                    current_fn = None
                    brace_depth = 0

            calls = call_re.findall(line)
            if not calls:
                continue

            window_start = max(0, line_no - 3)
            window_end = min(len(lines), line_no + 2)
            context_parts = lines[window_start:window_end]
            if current_fn:
                context_parts.append(current_fn)
            context_parts.extend(recent_comments)
            context_text = ' '.join(context_parts)
            hits = _check_keywords(context_text)
            if current_fn and current_fn.lower().startswith(('test_', 'run_', 'suite_', 'kat_', 'demo_')):
                hits.add(CHECK_SMOKE)

            for fn_name in calls:
                fn_evidence = evidence.setdefault(fn_name, {})
                for hit in hits:
                    label = f'{rel_path}:{current_fn or line_no}'
                    fn_evidence.setdefault(hit, [])
                    if label not in fn_evidence[hit]:
                        fn_evidence[hit].append(label)

    return evidence


def build_manifest() -> tuple[dict, bool]:
    exports = scan_public_exports()
    doc_sections = _doc_mentions()
    direct_test_evidence = _scan_direct_test_evidence()
    con = _connect()
    try:
        graph_targets = _targets_from_graph(con)
    finally:
        con.close()

    items = []
    blocker_count = 0
    counts = {check_name: 0 for check_name in ALL_CHECKS}

    for fn in exports:
        covered: dict[str, list[str]] = {}
        required = _required_checks(fn)

        if fn.name in graph_targets:
            for target in graph_targets[fn.name]:
                _mark(covered, CHECK_SMOKE, f'graph:function_test_map:{target}')

        for check_name, labels in direct_test_evidence.get(fn.name, {}).items():
            for label in labels:
                _mark(covered, check_name, f'test-call:{label}')

        if fn.category == 'cpu':
            _mark(covered, CHECK_NULL, 'docs/FFI_HOSTILE_CALLER.md:G.1 all CPU ABI functions')

        for section_name, section_map in doc_sections.items():
            for text in section_map.get(fn.name, []):
                for hit in _check_keywords(text):
                    _mark(covered, hit, f'docs/FFI_HOSTILE_CALLER.md:{section_name}')

        if fn.category == 'gpu' and fn.name.startswith('ufsecp_gpu_'):
            _mark(covered, CHECK_NULL, 'docs/FFI_HOSTILE_CALLER.md:Section J GPU NULL guards')

        missing = [check for check in required if check not in covered]
        for check_name in ALL_CHECKS:
            if check_name in covered:
                counts[check_name] += 1
        if missing:
            blocker_count += 1

        items.append({
            'function': fn.name,
            'category': fn.category,
            'header': fn.header,
            'signature': fn.signature,
            'required_checks': required,
            'covered_checks': covered,
            'missing_checks': missing,
            'blocking': bool(missing),
        })

    report = {
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'header_count': len(exports),
        'blocking_function_count': blocker_count,
        'coverage_counts': counts,
        'functions': items,
    }
    return report, blocker_count > 0


def _write_markdown(report: dict) -> str:
    lines = []
    lines.append('# ABI Negative-Test Manifest')
    lines.append('')
    lines.append(f"Generated: {report['generated_at']}")
    lines.append('')
    lines.append('Machine-generated hostile-caller coverage manifest for the public `ufsecp_*` ABI.')
    lines.append('')
    lines.append('## Summary')
    lines.append('')
    lines.append(f"- Exported functions scanned: {report['header_count']}")
    lines.append(f"- Blocking functions: {report['blocking_function_count']}")
    lines.append(f"- Null rejection evidence: {report['coverage_counts'][CHECK_NULL]}")
    lines.append(f"- Zero-edge evidence: {report['coverage_counts'][CHECK_ZERO]}")
    lines.append(f"- Invalid-content evidence: {report['coverage_counts'][CHECK_INVALID]}")
    lines.append(f"- Success-smoke evidence: {report['coverage_counts'][CHECK_SMOKE]}")
    lines.append('')
    lines.append('## Blocking Functions')
    lines.append('')
    lines.append('| Function | Missing Checks | Header |')
    lines.append('|----------|----------------|--------|')
    for item in report['functions']:
        if not item['blocking']:
            continue
        missing = ', '.join(item['missing_checks'])
        lines.append(f"| `{item['function']}` | `{missing}` | `{item['header']}` |")
    if report['blocking_function_count'] == 0:
        lines.append('| *(none)* | | |')
    lines.append('')
    lines.append('## Rule')
    lines.append('')
    lines.append('Every exported `ufsecp_*` function should satisfy the hostile-caller quartet when the contract implies it:')
    lines.append('')
    lines.append('1. `null_rejection`')
    lines.append('2. `zero_edge`')
    lines.append('3. `invalid_content`')
    lines.append('4. `success_smoke`')
    lines.append('')
    return '\n'.join(lines) + '\n'


def main() -> int:
    json_mode = '--json' in sys.argv[1:]
    write_files = '--write' in sys.argv[1:] or not json_mode
    report, has_fail = build_manifest()
    output = json.dumps(report, indent=2)
    if write_files:
        JSON_OUT.write_text(output, encoding='utf-8')
        MD_OUT.write_text(_write_markdown(report), encoding='utf-8')
    if json_mode:
        print(output)
    else:
        print(f"ABI negative-test manifest: {report['header_count']} exports, {report['blocking_function_count']} blocking")
        for item in report['functions']:
            if item['blocking']:
                print(f"FAIL {item['function']}: missing {', '.join(item['missing_checks'])}")
    return 1 if has_fail else 0


if __name__ == '__main__':
    sys.exit(main())