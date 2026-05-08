#!/usr/bin/env python3
"""Validate docs/AUDIT_TREND_TIMELINE.json. Remove if empty/invalid.

Used by caas-evidence-refresh.yml — was previously inline `python3 -c "..."`
which broke YAML literal-block parsing because Python module-level code
cannot be indented to match the YAML block scalar.
"""
import json
import os
import sys

PATH = "docs/AUDIT_TREND_TIMELINE.json"

if not os.path.exists(PATH):
    sys.exit(0)

try:
    d = json.load(open(PATH))
except Exception as e:  # noqa: BLE001
    print(f"::warning::{PATH} is invalid JSON: {e}")
    sys.exit(0)

# Accept both list-of-events and {events:[...]} envelope formats.
events = d if isinstance(d, list) else d.get("events", d.get("runs", []))
if not isinstance(events, list) or len(events) == 0:
    print(f"::warning::{PATH} has no events — removing to avoid committing empty timeline")
    os.remove(PATH)
