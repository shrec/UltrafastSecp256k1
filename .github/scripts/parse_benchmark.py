#!/usr/bin/env python3
"""
Parse benchmark output from bench_unified (or legacy bench_comprehensive)
into github-action-benchmark compatible JSON (customSmallerIsBetter format).

Input:  Raw text output from bench_unified or bench_comprehensive
Output: JSON file with benchmark entries

Supported input formats:

  bench_unified table (ns/op, always nanoseconds):
    | field_mul                                      |        8.2 |
    | Generator * k (FAST)                           |     5841.0 |

  Legacy bench_comprehensive (name: value unit):
    scalar_mul (K*G):          18.45 us    (54,201 ops/sec)
    field_mul:                  8.23 ns    (121,506,682 ops/sec)
"""

import json
import re
import sys
from pathlib import Path


def parse_benchmark_output(text: str) -> list[dict]:
    """Parse benchmark text output into benchmark entries."""
    entries = []
    seen = set()

    # Sections whose entries should be excluded from regression comparison.
    # MICRO-DIAGNOSTICS are sub-operation benchmarks that may legitimately
    # diverge from end-to-end performance (e.g. per-op speed traded for
    # better pipeline ILP via inlining).
    excluded_sections = {'MICRO-DIAGNOSTICS'}

    # Pattern 1 (bench_unified): table rows "| name | value |"
    # Matches numeric ns/op values only; skips headers (ns/op) and ratios (1.23x).
    table_pattern = re.compile(
        r'^\|\s+(.+?)\s+\|\s+([\d,]+(?:\.[\d]+)?)\s+\|$',
        re.MULTILINE
    )
    # Section header: "| SECTION NAME (extra) | ns/op |"
    section_pattern = re.compile(
        r'^\|\s+(.+?)\s+\|\s+ns/op\s+\|$',
        re.MULTILINE
    )

    # Build a map of position -> section name for section-aware filtering
    section_starts = []
    for m in section_pattern.finditer(text):
        section_starts.append((m.start(), m.group(1).strip()))

    def get_section(pos: int) -> str:
        """Return the section name for a given text position."""
        current = ''
        for start, name in section_starts:
            if start > pos:
                break
            current = name
        return current

    for match in table_pattern.finditer(text):
        name = match.group(1).strip()
        value_str = match.group(2).replace(',', '')
        # Skip header rows
        if name == '' or value_str == '' or 'ns/op' in name.lower():
            continue
        try:
            value_ns = float(value_str)
        except ValueError:
            continue
        # Skip entries from excluded sections
        section = get_section(match.start())
        if any(excl in section for excl in excluded_sections):
            continue
        if name not in seen:
            seen.add(name)
            entries.append({
                'name': name,
                'unit': 'ns',
                'value': round(value_ns, 2),
            })

    # Pattern 2 (legacy): "name: value unit (ops/sec)"
    # Examples:
    #   scalar_mul:     18.45 us
    #   field_mul:       8.23 ns
    # Name class includes = for entries like "Batch Inverse (n=100)".
    legacy_pattern = re.compile(
        r'^\s*([a-zA-Z0-9_\s\(\)/\*\+\-=]+?):\s+'
        r'([\d,\.]+)\s*(ns|us|ms|s)\b',
        re.MULTILINE
    )

    for match in legacy_pattern.finditer(text):
        name = match.group(1).strip()
        value_str = match.group(2).replace(',', '')
        unit = match.group(3)

        if name in seen:
            continue

        # In bench_unified output, all standalone benchmarks use table-format
        # rows (Pattern 1 above). Legacy-format printf lines that appear
        # within bench_unified sections are diagnostic/derived metrics (cost
        # decompositions, UNEXPLAINED gap, RFC6979 overhead, etc.), not
        # independent benchmarks. Skip them to avoid false regression alerts
        # on noisy computed values.
        section = get_section(match.start())
        if section:
            continue

        try:
            value = float(value_str)
        except ValueError:
            continue

        # Normalize to nanoseconds for consistent comparison
        multiplier = {
            'ns': 1.0,
            'us': 1000.0,
            'ms': 1_000_000.0,
            's': 1_000_000_000.0,
        }.get(unit, 1.0)

        value_ns = value * multiplier
        seen.add(name)

        entries.append({
            'name': name,
            'unit': 'ns',
            'value': round(value_ns, 2),
        })

    return entries


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <input.txt> <output.json>")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    if not input_path.exists():
        print(f"Error: {input_path} not found")
        sys.exit(1)

    text = input_path.read_text(encoding='utf-8', errors='replace')
    entries = parse_benchmark_output(text)

    if not entries:
        # No benchmarks parsed -- this is a HARD FAILURE.
        # A dummy entry would hide real regressions by making future comparisons
        # succeed against a meaningless baseline.
        print("Error: No benchmark entries parsed from output -- failing")
        print("  Input file:", input_path)
        print("  Input size:", input_path.stat().st_size, "bytes")
        print("  First 500 chars of input:")
        print(text[:500])
        sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(entries, indent=2))
    print(f"Wrote {len(entries)} benchmark entries to {output_path}")


if __name__ == '__main__':
    main()
