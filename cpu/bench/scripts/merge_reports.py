#!/usr/bin/env python3
"""
merge_reports.py -- Merge multiple bench_unified JSON reports into a comparison table.

Usage:
    python3 merge_reports.py report1.json report2.json [report3.json ...]
    python3 merge_reports.py --dir /path/to/reports/
    python3 merge_reports.py --dir . --output comparison.md

Output: Markdown table comparing ns/op and ratios across platforms.
"""
import argparse
import json
import os
import sys
from pathlib import Path


def load_report(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def platform_tag(meta: dict) -> str:
    """Short platform label from metadata."""
    arch = meta.get("arch", "?")
    cpu = meta.get("cpu", "?")
    # Shorten CPU brand
    for remove in ["(R)", "(TM)", "Intel ", "AMD ", "Core "]:
        cpu = cpu.replace(remove, "")
    cpu = cpu.strip()
    if len(cpu) > 24:
        cpu = cpu[:24] + ".."
    return f"{arch} / {cpu}"


def main():
    parser = argparse.ArgumentParser(description="Merge bench_unified JSON reports")
    parser.add_argument("files", nargs="*", help="JSON report files")
    parser.add_argument("--dir", help="Directory containing JSON reports")
    parser.add_argument("--output", "-o", default="-", help="Output file (default: stdout)")
    parser.add_argument("--filter", default="", help="Only show entries matching this substring")
    args = parser.parse_args()

    # Collect files
    paths = list(args.files)
    if args.dir:
        for f in sorted(Path(args.dir).glob("bench_*.json")):
            paths.append(str(f))
    if not paths:
        print("No JSON files specified. Use --dir or pass file paths.", file=sys.stderr)
        sys.exit(1)

    # Load all reports
    reports = []
    for p in paths:
        try:
            reports.append((p, load_report(p)))
        except Exception as e:
            print(f"[!] Skipping {p}: {e}", file=sys.stderr)

    if not reports:
        print("No valid reports found.", file=sys.stderr)
        sys.exit(1)

    # Build platform labels
    labels = []
    for path, rpt in reports:
        meta = rpt.get("metadata", {})
        labels.append(platform_tag(meta))

    # Collect all unique (section, name) keys preserving order
    seen = set()
    all_keys = []
    for _, rpt in reports:
        for entry in rpt.get("results", []):
            key = (entry["section"], entry["name"])
            if key not in seen:
                seen.add(key)
                all_keys.append(key)

    # Build lookup: (section, name) -> value for each report
    lookups = []
    for _, rpt in reports:
        lut = {}
        for entry in rpt.get("results", []):
            key = (entry["section"], entry["name"])
            if "ratio" in entry:
                lut[key] = ("ratio", entry["ratio"])
            else:
                lut[key] = ("ns", entry["ns"])
        lookups.append(lut)

    # Filter
    if args.filter:
        filt = args.filter.lower()
        all_keys = [(s, n) for s, n in all_keys
                    if filt in s.lower() or filt in n.lower()]

    # Generate markdown
    out = sys.stdout if args.output == "-" else open(args.output, "w")

    n_platforms = len(reports)
    col_width = max(14, max(len(l) for l in labels) + 2)

    # Header
    out.write("# Benchmark Comparison\n\n")
    out.write(f"| {'Operation':<44} |")
    for label in labels:
        out.write(f" {label:>{col_width}} |")
    out.write("\n")

    out.write(f"|{'-'*46}|")
    for _ in labels:
        out.write(f"{'-'*(col_width+2)}|")
    out.write("\n")

    # Rows
    prev_section = ""
    for section, name in all_keys:
        if section != prev_section:
            # Section separator
            out.write(f"| **{section:<42}** |")
            for _ in labels:
                out.write(f" {'':>{col_width}} |")
            out.write("\n")
            prev_section = section

        out.write(f"| {name:<44} |")
        for lut in lookups:
            val = lut.get((section, name))
            if val is None:
                out.write(f" {'--':>{col_width}} |")
            elif val[0] == "ratio":
                out.write(f" {f'{val[1]:.2f}x':>{col_width}} |")
            else:
                ns = val[1]
                if ns >= 1e6:
                    out.write(f" {f'{ns/1e6:.2f} ms':>{col_width}} |")
                elif ns >= 1e3:
                    out.write(f" {f'{ns/1e3:.2f} us':>{col_width}} |")
                else:
                    out.write(f" {f'{ns:.1f} ns':>{col_width}} |")
        out.write("\n")

    if args.output != "-":
        out.close()
        print(f"Written to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
