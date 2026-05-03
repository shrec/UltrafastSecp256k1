#!/usr/bin/env python3
"""parse_gate_report.py -- Print a human-readable summary of a CAAS gate JSON report.

Usage:
    python3 ci/parse_gate_report.py <report.json>            # audit gate summary
    python3 ci/parse_gate_report.py <report.json> --autonomy # autonomy score summary
"""
import json
import sys
from pathlib import Path


def main():
    args = sys.argv[1:]
    autonomy_mode = "--autonomy" in args
    paths = [a for a in args if not a.startswith("--")]

    if not paths:
        print("(no report file specified)")
        sys.exit(0)

    report_path = Path(paths[0])
    try:
        d = json.loads(report_path.read_text(errors="replace"))
    except Exception as e:
        print(f"(could not parse {report_path.name}: {e})")
        sys.exit(0)

    if autonomy_mode:
        score = d.get("autonomy_score", 0)
        ready = d.get("autonomy_ready", False)
        print(f"score={score}/100  ready={ready}")
        return

    # Audit gate summary — accept both 'verdict' (new schema) and
    # 'audit_verdict' (legacy) for resilience to schema evolution.
    verdict = d.get("verdict") or d.get("audit_verdict") or "unknown"
    summary = d.get("summary", {})
    blocking = summary.get("blocking", summary.get("failing", summary.get("fail", 0)))
    advisory = summary.get("advisory", 0)
    print(f"verdict={verdict}  blocking={blocking}  advisory={advisory}")
    for c in d.get("checks", d.get("gates", [])):
        if c.get("status") == "FAIL":
            print(f"  FAIL: {c.get('name', c.get('gate', '?'))}")
            for f in c.get("findings", []):
                if f.get("severity") == "FAIL":
                    print(f"    - {f['message']}")


if __name__ == "__main__":
    main()
