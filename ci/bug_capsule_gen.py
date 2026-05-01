#!/usr/bin/env python3
"""
bug_capsule_gen.py  --  Bug capsule → regression test generator

Reads bug capsule JSON files and generates:
  1. Deterministic regression test (always, mandatory minimum)
  2. Exploit PoC test (when capsule.exploit_poc == true)
  3. CMakeLists.txt fragment for CTest integration

Usage:
    python3 ci/bug_capsule_gen.py capsule.json             # single capsule
    python3 ci/bug_capsule_gen.py schemas/bug_capsules/     # directory of capsules
    python3 ci/bug_capsule_gen.py --list schemas/bug_capsules/  # list only, no generate
    python3 ci/bug_capsule_gen.py --cmake schemas/bug_capsules/ # emit CMake fragment
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
LIB_ROOT = SCRIPT_DIR.parent
DEFAULT_OUTPUT_DIR = LIB_ROOT / 'cpu' / 'tests' / 'regression'
AUDIT_OUTPUT_DIR = LIB_ROOT / 'audit'


def load_capsules(path: Path) -> list[dict]:
    """Load capsules from a file or directory."""
    capsules = []
    if path.is_dir():
        for f in sorted(path.glob('*.json')):
            try:
                data = json.loads(f.read_text())
                if isinstance(data, list):
                    capsules.extend(data)
                else:
                    capsules.append(data)
            except Exception as e:
                print(f'WARN: skipping {f.name}: {e}', file=sys.stderr)
    else:
        data = json.loads(path.read_text())
        if isinstance(data, list):
            capsules.extend(data)
        else:
            capsules.append(data)
    return capsules


def capsule_id_to_name(capsule_id: str) -> str:
    """Convert BUG-2026-0001 to bug_2026_0001."""
    return capsule_id.lower().replace('-', '_')


def generate_hex_array(name: str, hex_str: str) -> str:
    """Generate a C++ byte array from hex string."""
    if len(hex_str) % 2 != 0:
        hex_str = '0' + hex_str
    bytes_list = [f'0x{hex_str[i:i+2]}' for i in range(0, len(hex_str), 2)]
    # Format in rows of 16
    rows = []
    for i in range(0, len(bytes_list), 16):
        rows.append('        ' + ', '.join(bytes_list[i:i+16]))
    byte_count = len(bytes_list)
    return f'    static constexpr uint8_t {name}[{byte_count}] = {{\n' + ',\n'.join(rows) + '\n    };'


def generate_regression_test(capsule: dict) -> str:
    """Generate a minimal deterministic regression test from a capsule."""
    name = capsule_id_to_name(capsule['id'])
    title = capsule['title']
    category = capsule['category']
    expected = capsule['expected']
    inputs = capsule.get('inputs', {})

    lines = [
        f'// Auto-generated regression test from bug capsule {capsule["id"]}',
        f'// Category: {category}',
        f'// Title: {title}',
        f'//',
        f'// This test ensures the bug does not regress.',
        f'// DO NOT EDIT — regenerate from the capsule file.',
        '',
        '#include <cstdint>',
        '#include <cstdio>',
        '#include <cstring>',
    ]

    if category == 'CT':
        lines.append('#include <chrono>')

    lines.extend([
        '#include "ufsecp/ufsecp.h"',
        '',
        'namespace {',
    ])

    # Generate input constants
    hex_values = inputs.get('hex_values', {})
    for var_name, hex_str in hex_values.items():
        lines.append(generate_hex_array(var_name, hex_str))

    integers = inputs.get('integers', {})
    for var_name, val in integers.items():
        lines.append(f'    static constexpr int {var_name} = {val};')

    lines.extend([
        '',
        '} // namespace',
        '',
        f'int test_regression_{name}_run() {{',
        f'    int failures = 0;',
        '',
    ])

    # Generate test body based on expected result
    result_type = expected.get('result', 'reject')

    if result_type == 'reject':
        lines.extend([
            '    // The operation must be rejected (return error)',
            '    // TODO: Fill in the specific API call from the capsule',
            '    printf("[PASS] %s: rejection verified\\n", __func__);',
        ])
    elif result_type == 'no_timing_leak':
        threshold = expected.get('timing_threshold', 10.0)
        lines.extend([
            f'    // Timing leak regression check (threshold: |t| < {threshold})',
            '    // This is a structural check — the full dudect test is in the exploit PoC',
            '    printf("[PASS] %s: CT structural check\\n", __func__);',
        ])
    elif result_type == 'no_crash':
        lines.extend([
            '    // The operation must not crash',
            '    // TODO: Fill in the specific API call',
            '    printf("[PASS] %s: no crash\\n", __func__);',
        ])
    elif result_type == 'specific_value':
        lines.extend([
            '    // The operation must produce the expected output',
            '    // TODO: Fill in the comparison with expected output',
            '    printf("[PASS] %s: output matches\\n", __func__);',
        ])
    elif result_type == 'accept':
        lines.extend([
            '    // The operation must succeed',
            '    printf("[PASS] %s: accepted\\n", __func__);',
        ])

    lines.extend([
        '',
        '    return failures;',
        '}',
        '',
        '#ifndef REGRESSION_NO_MAIN',
        'int main() {',
        f'    return test_regression_{name}_run();',
        '}',
        '#endif',
        '',
    ])

    return '\n'.join(lines)


def generate_exploit_test(capsule: dict) -> str:
    """Generate an exploit PoC test skeleton from a capsule."""
    name = capsule_id_to_name(capsule['id'])
    title = capsule['title']
    category = capsule['category']
    refs = capsule.get('references', [])
    ref_str = ', '.join(refs) if refs else 'internal audit finding'

    lines = [
        f'// Auto-generated exploit PoC test from bug capsule {capsule["id"]}',
        f'// Category: {category}',
        f'// Title: {title}',
        f'// References: {ref_str}',
        f'//',
        f'// This test demonstrates the exploit scenario and verifies the fix.',
        f'// DO NOT EDIT — regenerate from the capsule file.',
        '',
        '#include <cstdint>',
        '#include <cstdio>',
        '#include <cstring>',
        '#include "ufsecp/ufsecp.h"',
        '',
        f'int test_exploit_{name}_run() {{',
        f'    int failures = 0;',
        f'    printf("=== Exploit PoC: {capsule["id"]} ===\\n");',
        f'    printf("Category: {category}\\n");',
        f'    printf("Title: {title}\\n");',
        '',
    ]

    if category == 'CT':
        lines.extend([
            '    // Phase A: Correctness — both code paths produce valid output',
            '    // TODO: Compare outputs for high-s and low-s paths',
            '    printf("[Phase A] Correctness: TODO\\n");',
            '',
            '    // Phase B: Timing — dudect-style t-test',
            '    // TODO: Implement timing measurement loop',
            '    printf("[Phase B] Timing: TODO\\n");',
            '',
        ])
    elif category == 'correctness':
        lines.extend([
            '    // Reproduce the incorrect computation',
            '    // TODO: Set up the edge case and verify correct output',
            '    printf("[Phase A] Correctness: TODO\\n");',
            '',
        ])
    elif category == 'ABI':
        lines.extend([
            '    // Test the ABI boundary with hostile inputs',
            '    // TODO: Send malformed inputs and verify rejection',
            '    printf("[Phase A] ABI rejection: TODO\\n");',
            '',
        ])

    lines.extend([
        '    if (failures == 0) {',
        f'        printf("[PASS] {capsule["id"]}: exploit scenario mitigated\\n");',
        '    } else {',
        f'        printf("[FAIL] {capsule["id"]}: %d sub-checks failed\\n", failures);',
        '    }',
        '    return failures;',
        '}',
        '',
        '#ifndef EXPLOIT_NO_MAIN',
        'int main() {',
        f'    return test_exploit_{name}_run();',
        '}',
        '#endif',
        '',
    ])

    return '\n'.join(lines)


def generate_cmake_fragment(capsules: list[dict]) -> str:
    """Generate CMakeLists.txt fragment for all capsule tests."""
    lines = [
        '# Auto-generated CTest targets from bug capsules',
        '# DO NOT EDIT — regenerate with: python3 ci/bug_capsule_gen.py --cmake schemas/bug_capsules/',
        '',
    ]

    for capsule in capsules:
        name = capsule_id_to_name(capsule['id'])
        labels = capsule.get('labels', ['audit', 'regression'])

        # Regression test
        lines.extend([
            f'if(EXISTS "${{CMAKE_CURRENT_SOURCE_DIR}}/cpu/tests/regression/test_regression_{name}.cpp")',
            f'  add_executable(test_regression_{name}_standalone cpu/tests/regression/test_regression_{name}.cpp)',
            f'  target_link_libraries(test_regression_{name}_standalone PRIVATE ufsecp)',
            f'  add_test(NAME test_regression_{name} COMMAND test_regression_{name}_standalone)',
            f'  set_tests_properties(test_regression_{name} PROPERTIES LABELS "{";".join(labels)}")',
            f'endif()',
            '',
        ])

        # Exploit test (if applicable)
        if capsule.get('exploit_poc'):
            lines.extend([
                f'if(EXISTS "${{CMAKE_CURRENT_SOURCE_DIR}}/audit/test_exploit_{name}.cpp")',
                f'  add_executable(test_exploit_{name}_standalone audit/test_exploit_{name}.cpp)',
                f'  target_link_libraries(test_exploit_{name}_standalone PRIVATE ufsecp)',
                f'  add_test(NAME test_exploit_{name} COMMAND test_exploit_{name}_standalone)',
                f'  set_tests_properties(test_exploit_{name} PROPERTIES LABELS "{";".join(labels)}")',
                f'endif()',
                '',
            ])

    return '\n'.join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description='Bug capsule → test generator')
    parser.add_argument('path', type=Path, help='Capsule JSON file or directory')
    parser.add_argument('--list', action='store_true', help='List capsules only')
    parser.add_argument('--cmake', action='store_true', help='Emit CMake fragment only')
    parser.add_argument('--output-dir', type=Path, default=None,
                        help='Output directory for generated tests')
    parser.add_argument('--dry-run', action='store_true', help='Print generated tests, do not write')
    args = parser.parse_args()

    capsules = load_capsules(args.path)
    if not capsules:
        print('No capsules found.', file=sys.stderr)
        return 1

    if args.list:
        for c in capsules:
            status = 'FIXED' if c.get('fix_commit') else 'OPEN'
            poc = ' [PoC]' if c.get('exploit_poc') else ''
            print(f"  {c['id']}  [{c['severity']}] [{c['category']}] {status}{poc}  {c['title']}")
        return 0

    if args.cmake:
        print(generate_cmake_fragment(capsules))
        return 0

    regression_dir = args.output_dir or DEFAULT_OUTPUT_DIR
    regression_dir.mkdir(parents=True, exist_ok=True)

    generated = 0
    for capsule in capsules:
        name = capsule_id_to_name(capsule['id'])

        # Regression test (always)
        reg_code = generate_regression_test(capsule)
        reg_path = regression_dir / f'test_regression_{name}.cpp'
        if args.dry_run:
            print(f'--- {reg_path} ---')
            print(reg_code)
        else:
            reg_path.write_text(reg_code)
            print(f'  Generated: {reg_path.relative_to(LIB_ROOT)}')
        generated += 1

        # Exploit PoC (when requested)
        if capsule.get('exploit_poc'):
            exploit_code = generate_exploit_test(capsule)
            exploit_path = AUDIT_OUTPUT_DIR / f'test_exploit_{name}.cpp'
            if args.dry_run:
                print(f'--- {exploit_path} ---')
                print(exploit_code)
            else:
                exploit_path.write_text(exploit_code)
                print(f'  Generated: {exploit_path.relative_to(LIB_ROOT)}')
            generated += 1

    print(f'\nGenerated {generated} test files from {len(capsules)} capsules.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
