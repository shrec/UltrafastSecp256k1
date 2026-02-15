#!/usr/bin/env python3
"""
Parse benchmark output from bench_comprehensive into
github-action-benchmark compatible JSON (customSmallerIsBetter format).

Input:  Raw text output from bench_comprehensive
Output: JSON file with benchmark entries

Expected input format (lines like):
  scalar_mul (K*G):          18.45 µs    (54,201 ops/sec)
  point_add:                  0.87 µs    (1,149,425 ops/sec)
  field_mul:                  8.23 ns    (121,506,682 ops/sec)
"""

import json
import re
import sys
from pathlib import Path


def parse_benchmark_output(text: str) -> list[dict]:
    """Parse benchmark text output into benchmark entries."""
    entries = []

    # Pattern: "name: value unit (ops/sec)"
    # Examples:
    #   scalar_mul:     18.45 µs
    #   field_mul:       8.23 ns
    pattern = re.compile(
        r'^\s*([a-zA-Z0-9_\s\(\)/\*\+\-]+?):\s+'
        r'([\d,\.]+)\s*(ns|µs|us|ms|s)\b',
        re.MULTILINE
    )

    for match in pattern.finditer(text):
        name = match.group(1).strip()
        value_str = match.group(2).replace(',', '')
        unit = match.group(3)

        try:
            value = float(value_str)
        except ValueError:
            continue

        # Normalize to nanoseconds for consistent comparison
        multiplier = {
            'ns': 1.0,
            'µs': 1000.0,
            'us': 1000.0,
            'ms': 1_000_000.0,
            's': 1_000_000_000.0,
        }.get(unit, 1.0)

        value_ns = value * multiplier

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
        # Create a dummy entry if no benchmarks were parsed
        entries = [{
            'name': 'benchmark_parse_warning',
            'unit': 'ns',
            'value': 0,
        }]
        print("Warning: No benchmark entries parsed from output")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(entries, indent=2))
    print(f"Wrote {len(entries)} benchmark entries to {output_path}")


if __name__ == '__main__':
    main()
