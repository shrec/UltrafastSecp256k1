#!/usr/bin/env python3
"""
artifact_analyzer.py  --  Multi-report analysis and regression tracking

MVP artifact analyzer that:
  1. Ingests reports from different runners (unified, sanitizer, ct-verif, GPU, bench)
  2. Normalizes into a unified event model
  3. Detects regressions (commit-to-commit FAIL/WARN transitions)
  4. Detects platform divergence (ARM vs x86 vs RISC-V vs ESP32)
  5. Detects flakes (dudect/perf statistical tests)
  6. Exports: aggregated SARIF, Markdown summary, timeline JSON

Usage:
    python3 ci/artifact_analyzer.py ingest report1.json [report2.json ...]
    python3 ci/artifact_analyzer.py diff --before a.json --after b.json
    python3 ci/artifact_analyzer.py timeline --db analyzer.db
    python3 ci/artifact_analyzer.py export --format sarif|markdown|timeline
    python3 ci/artifact_analyzer.py divergence r1.json r2.json [r3.json ...]
    python3 ci/artifact_analyzer.py flakes --db analyzer.db --window 10
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
LIB_ROOT = SCRIPT_DIR.parent
DEFAULT_DB = LIB_ROOT / 'build' / 'artifact_analyzer.db'


# ---------------------------------------------------------------------------
# Event model: normalized representation of any finding
# ---------------------------------------------------------------------------

def normalize_event(report: dict, finding: dict, section: Optional[str] = None) -> dict:
    """Convert any finding from any runner into a normalized event."""
    # Detect schema version
    schema_ver = report.get('schema_version', '0.0.0')

    # Extract commit
    if schema_ver >= '1.0.0':
        commit_obj = report.get('commit', {})
        sha = commit_obj.get('value', {}).get('sha') if isinstance(commit_obj.get('value'), dict) else None
    else:
        sha = report.get('provenance', {}).get('git', {}).get('sha') if report.get('provenance') else None

    # Extract platform
    platform = report.get('platform')
    if isinstance(platform, dict):
        platform = f"{platform.get('system', '')}-{platform.get('machine', '')}"

    # Normalize severity
    raw_sev = finding.get('severity', finding.get('level', 'info'))
    if ':' in str(raw_sev):
        # schema v1.0.0: "blocking:critical", "advisory:medium"
        sev_class, sev_level = raw_sev.split(':', 1)
    elif raw_sev in ('FAIL', 'ERROR', 'error'):
        sev_class, sev_level = 'blocking', 'high'
    elif raw_sev in ('WARN', 'WARNING', 'warning'):
        sev_class, sev_level = 'advisory', 'medium'
    elif raw_sev in ('PASS', 'pass'):
        sev_class, sev_level = 'pass', 'pass'
    else:
        sev_class, sev_level = 'info', 'info'

    return {
        'run_id': report.get('run_id', _synthetic_run_id(report)),
        'runner': report.get('runner', 'unknown_runner'),
        'generated_at': report.get('generated_at', ''),
        'commit_sha': sha,
        'platform': platform,
        'section': section,
        'check_id': finding.get('check_id', finding.get('name', '')),
        'severity_class': sev_class,
        'severity_level': sev_level,
        'title': finding.get('title', finding.get('message', '')),
        'detail': finding.get('detail'),
        'file': finding.get('file'),
        'line': finding.get('line'),
    }


def _synthetic_run_id(report: dict) -> str:
    ts = report.get('generated_at', datetime.now(timezone.utc).isoformat())
    runner = report.get('runner', 'anon')
    h = hashlib.sha256(f'{runner}-{ts}'.encode()).hexdigest()[:10]
    return f'{runner}-{h}'


def extract_events(report: dict) -> list[dict]:
    """Extract all normalized events from a report, any schema version."""
    events = []

    # Schema v1.0.0 with sections[]
    if 'sections' in report:
        for sec in report['sections']:
            sec_name = sec.get('name', '')
            for finding in sec.get('findings', []):
                events.append(normalize_event(report, finding, sec_name))
    # Legacy audit_gate format with checks[]
    elif 'checks' in report:
        for check in report['checks']:
            sec_name = check.get('name', '')
            for finding in check.get('findings', []):
                events.append(normalize_event(report, finding, sec_name))
    # Flat findings[]
    elif 'findings' in report:
        for finding in report['findings']:
            events.append(normalize_event(report, finding))
    # Coverage-style (auditor_mode)
    elif 'coverage' in report:
        for item in report.get('coverage', []):
            events.append(normalize_event(report, {
                'check_id': item.get('key', ''),
                'severity': 'FAIL' if not item.get('present') else 'PASS',
                'title': f"{item.get('key', '')} {'present' if item.get('present') else 'missing'}",
            }))

    return events


# ---------------------------------------------------------------------------
# SQLite store for timeline / history tracking
# ---------------------------------------------------------------------------

def init_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            runner TEXT NOT NULL,
            generated_at TEXT,
            commit_sha TEXT,
            platform TEXT,
            section TEXT,
            check_id TEXT,
            severity_class TEXT,
            severity_level TEXT,
            title TEXT,
            detail TEXT,
            file TEXT,
            line INTEGER,
            ingested_at TEXT DEFAULT (datetime('now'))
        );
        CREATE INDEX IF NOT EXISTS idx_events_commit ON events(commit_sha);
        CREATE INDEX IF NOT EXISTS idx_events_runner ON events(runner);
        CREATE INDEX IF NOT EXISTS idx_events_check ON events(check_id);

        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT UNIQUE NOT NULL,
            runner TEXT,
            generated_at TEXT,
            commit_sha TEXT,
            platform TEXT,
            verdict TEXT,
            raw_json TEXT,
            ingested_at TEXT DEFAULT (datetime('now'))
        );
    """)
    return conn


def ingest_report(conn: sqlite3.Connection, report: dict) -> int:
    """Ingest a report into the database. Returns number of events stored."""
    run_id = report.get('run_id', _synthetic_run_id(report))

    commit_obj = report.get('commit', {})
    if isinstance(commit_obj, dict) and isinstance(commit_obj.get('value'), dict):
        sha = commit_obj['value'].get('sha')
    elif report.get('provenance', {}).get('git', {}).get('sha'):
        sha = report['provenance']['git']['sha']
    else:
        sha = None

    # Store report metadata
    conn.execute("""
        INSERT OR REPLACE INTO reports (run_id, runner, generated_at, commit_sha, platform, verdict, raw_json)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        run_id,
        report.get('runner'),
        report.get('generated_at'),
        sha,
        report.get('platform') if isinstance(report.get('platform'), str) else json.dumps(report.get('platform')),
        report.get('verdict', report.get('status')),
        json.dumps(report),
    ))

    # Store events
    events = extract_events(report)
    for ev in events:
        conn.execute("""
            INSERT INTO events (run_id, runner, generated_at, commit_sha, platform,
                                section, check_id, severity_class, severity_level,
                                title, detail, file, line)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            ev['run_id'], ev['runner'], ev['generated_at'], ev['commit_sha'],
            ev['platform'], ev['section'], ev['check_id'], ev['severity_class'],
            ev['severity_level'], ev['title'], ev['detail'], ev['file'], ev['line'],
        ))

    conn.commit()
    return len(events)


# ---------------------------------------------------------------------------
# Regression diff
# ---------------------------------------------------------------------------

def regression_diff(before: dict, after: dict) -> dict:
    """Compare two reports and identify regressions and fixes."""
    before_events = {(e['check_id'], e['section']): e for e in extract_events(before)}
    after_events = {(e['check_id'], e['section']): e for e in extract_events(after)}

    regressions = []
    fixes = []
    new_findings = []
    removed = []

    all_keys = set(before_events.keys()) | set(after_events.keys())
    for key in sorted(all_keys):
        b = before_events.get(key)
        a = after_events.get(key)

        if b and a:
            b_blocking = b['severity_class'] == 'blocking'
            a_blocking = a['severity_class'] == 'blocking'
            if not b_blocking and a_blocking:
                regressions.append({
                    'check_id': key[0], 'section': key[1],
                    'before': b['severity_class'] + ':' + b['severity_level'],
                    'after': a['severity_class'] + ':' + a['severity_level'],
                    'title': a['title'],
                })
            elif b_blocking and not a_blocking:
                fixes.append({
                    'check_id': key[0], 'section': key[1],
                    'before': b['severity_class'] + ':' + b['severity_level'],
                    'after': a['severity_class'] + ':' + a['severity_level'],
                    'title': a['title'],
                })
        elif a and not b:
            new_findings.append({
                'check_id': key[0], 'section': key[1],
                'severity': a['severity_class'] + ':' + a['severity_level'],
                'title': a['title'],
            })
        elif b and not a:
            removed.append({
                'check_id': key[0], 'section': key[1],
                'severity': b['severity_class'] + ':' + b['severity_level'],
                'title': b['title'],
            })

    before_sha = before.get('provenance', {}).get('git', {}).get('short', '?')
    after_sha = after.get('provenance', {}).get('git', {}).get('short', '?')

    return {
        'before_commit': before_sha,
        'after_commit': after_sha,
        'regressions': regressions,
        'fixes': fixes,
        'new_findings': new_findings,
        'removed': removed,
        'summary': {
            'regressions': len(regressions),
            'fixes': len(fixes),
            'new': len(new_findings),
            'removed': len(removed),
        },
    }


# ---------------------------------------------------------------------------
# Platform divergence detection
# ---------------------------------------------------------------------------

def detect_divergence(reports: list[dict]) -> dict:
    """Compare reports from different platforms and find divergent results."""
    platform_events: dict[str, dict[tuple, dict]] = {}

    for report in reports:
        plat = report.get('platform', 'unknown')
        if isinstance(plat, dict):
            plat = f"{plat.get('system', '')}-{plat.get('machine', '')}"
        events = extract_events(report)
        platform_events[plat] = {(e['check_id'], e['section']): e for e in events}

    platforms = list(platform_events.keys())
    divergences = []

    if len(platforms) < 2:
        return {'platforms': platforms, 'divergences': [], 'note': 'need >=2 platforms to compare'}

    # Find checks that differ across platforms
    all_keys: set[tuple] = set()
    for evts in platform_events.values():
        all_keys.update(evts.keys())

    for key in sorted(all_keys):
        results_by_platform = {}
        for plat in platforms:
            ev = platform_events[plat].get(key)
            if ev:
                results_by_platform[plat] = ev['severity_class']

        # Check if all platforms agree
        unique_severities = set(results_by_platform.values())
        if len(unique_severities) > 1:
            divergences.append({
                'check_id': key[0],
                'section': key[1],
                'results': results_by_platform,
            })

    return {
        'platforms': platforms,
        'divergences': divergences,
        'divergent_count': len(divergences),
    }


# ---------------------------------------------------------------------------
# Flake detection
# ---------------------------------------------------------------------------

def detect_flakes(conn: sqlite3.Connection, window: int = 10) -> list[dict]:
    """Find check_ids that flip between pass and fail across recent runs."""
    rows = conn.execute("""
        SELECT check_id, section, severity_class,
               GROUP_CONCAT(severity_class) AS history,
               COUNT(DISTINCT severity_class) AS unique_severities
        FROM (
            SELECT check_id, section, severity_class, generated_at
            FROM events
            WHERE severity_class IN ('blocking', 'pass', 'advisory')
            ORDER BY generated_at DESC
            LIMIT ?
        )
        GROUP BY check_id, section
        HAVING unique_severities > 1
        ORDER BY unique_severities DESC
    """, (window * 50,)).fetchall()

    flakes = []
    for r in rows:
        history = r['history'].split(',') if r['history'] else []
        flip_count = sum(1 for i in range(1, len(history)) if history[i] != history[i-1])
        if flip_count >= 2:
            flakes.append({
                'check_id': r['check_id'],
                'section': r['section'],
                'flip_count': flip_count,
                'latest': history[0] if history else None,
                'history_sample': history[:window],
            })

    return sorted(flakes, key=lambda x: x['flip_count'], reverse=True)


# ---------------------------------------------------------------------------
# Export: SARIF
# ---------------------------------------------------------------------------

_SARIF_LEVEL_MAP = {
    'blocking': 'error',
    'advisory': 'warning',
    'info': 'note',
    'pass': 'none',
}


def export_sarif(events: list[dict], runner: str = 'artifact_analyzer') -> dict:
    """Convert events into SARIF 2.1.0 for GitHub Code Scanning."""
    rules: dict[str, dict] = {}
    results = []

    for ev in events:
        if ev['severity_class'] == 'pass':
            continue

        rule_id = ev['check_id'] or 'UNKNOWN'
        if rule_id not in rules:
            rules[rule_id] = {
                'id': rule_id,
                'shortDescription': {'text': ev['title'] or rule_id},
                'defaultConfiguration': {
                    'level': _SARIF_LEVEL_MAP.get(ev['severity_class'], 'warning'),
                },
            }

        result: dict[str, Any] = {
            'ruleId': rule_id,
            'level': _SARIF_LEVEL_MAP.get(ev['severity_class'], 'warning'),
            'message': {'text': ev['title'] or '(no title)'},
        }
        if ev.get('file'):
            result['locations'] = [{
                'physicalLocation': {
                    'artifactLocation': {'uri': ev['file'], 'uriBaseId': '%SRCROOT%'},
                    'region': {'startLine': ev.get('line', 1) or 1},
                },
            }]
        results.append(result)

    return {
        '$schema': 'https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json',
        'version': '2.1.0',
        'runs': [{
            'tool': {
                'driver': {
                    'name': f'UltrafastSecp256k1 {runner}',
                    'version': '1.0.0',
                    'rules': list(rules.values()),
                },
            },
            'results': results,
        }],
    }


# ---------------------------------------------------------------------------
# Export: Markdown summary (PR comment style)
# ---------------------------------------------------------------------------

def export_markdown(diff: Optional[dict] = None, events: Optional[list[dict]] = None) -> str:
    """Generate a Markdown summary suitable for PR comments."""
    lines = ['## Audit Artifact Analysis', '']

    if diff:
        lines.append(f'**Comparing** `{diff["before_commit"]}` → `{diff["after_commit"]}`')
        lines.append('')
        s = diff['summary']
        if s['regressions']:
            lines.append(f'### :x: Regressions ({s["regressions"]})')
            for r in diff['regressions']:
                lines.append(f'- **{r["check_id"]}** ({r["section"]}): {r["before"]} → {r["after"]} — {r["title"]}')
            lines.append('')
        if s['fixes']:
            lines.append(f'### :white_check_mark: Fixes ({s["fixes"]})')
            for r in diff['fixes']:
                lines.append(f'- **{r["check_id"]}** ({r["section"]}): {r["before"]} → {r["after"]} — {r["title"]}')
            lines.append('')
        if s['new']:
            lines.append(f'### :new: New findings ({s["new"]})')
            for r in diff['new_findings']:
                lines.append(f'- **{r["check_id"]}** ({r["section"]}): {r["severity"]} — {r["title"]}')
            lines.append('')
        if not (s['regressions'] or s['fixes'] or s['new']):
            lines.append('No changes detected.')
            lines.append('')

    if events:
        blocking = [e for e in events if e['severity_class'] == 'blocking']
        advisory = [e for e in events if e['severity_class'] == 'advisory']
        lines.append(f'| Metric | Count |')
        lines.append(f'|--------|-------|')
        lines.append(f'| Blocking | {len(blocking)} |')
        lines.append(f'| Advisory | {len(advisory)} |')
        lines.append(f'| Total | {len(events)} |')
        lines.append('')

    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Export: Timeline JSON
# ---------------------------------------------------------------------------

def export_timeline(conn: sqlite3.Connection, limit: int = 100) -> list[dict]:
    """Export a timeline of recent events for dashboard consumption."""
    rows = conn.execute("""
        SELECT r.run_id, r.runner, r.generated_at, r.commit_sha, r.platform, r.verdict,
               COUNT(e.id) AS event_count,
               SUM(CASE WHEN e.severity_class = 'blocking' THEN 1 ELSE 0 END) AS blocking,
               SUM(CASE WHEN e.severity_class = 'advisory' THEN 1 ELSE 0 END) AS advisory
        FROM reports r
        LEFT JOIN events e ON e.run_id = r.run_id
        GROUP BY r.run_id
        ORDER BY r.generated_at DESC
        LIMIT ?
    """, (limit,)).fetchall()

    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cmd_ingest(args: argparse.Namespace) -> int:
    conn = init_db(Path(args.db))
    total = 0
    for path in args.reports:
        p = Path(path)
        if not p.exists():
            print(f'WARN: {p} not found, skipping', file=sys.stderr)
            continue
        report = json.loads(p.read_text())
        n = ingest_report(conn, report)
        total += n
        print(f'Ingested {p.name}: {n} events')
    conn.close()
    print(f'Total: {total} events ingested')
    return 0


def cmd_diff(args: argparse.Namespace) -> int:
    before = json.loads(Path(args.before).read_text())
    after = json.loads(Path(args.after).read_text())
    diff = regression_diff(before, after)

    if args.format == 'markdown':
        print(export_markdown(diff=diff))
    else:
        print(json.dumps(diff, indent=2))
    return 1 if diff['summary']['regressions'] else 0


def cmd_divergence(args: argparse.Namespace) -> int:
    reports = [json.loads(Path(p).read_text()) for p in args.reports]
    result = detect_divergence(reports)
    print(json.dumps(result, indent=2))
    return 1 if result['divergent_count'] else 0


def cmd_flakes(args: argparse.Namespace) -> int:
    conn = init_db(Path(args.db))
    flakes = detect_flakes(conn, window=args.window)
    conn.close()
    print(json.dumps(flakes, indent=2))
    return 1 if flakes else 0


def cmd_timeline(args: argparse.Namespace) -> int:
    conn = init_db(Path(args.db))
    tl = export_timeline(conn, limit=args.limit)
    conn.close()
    print(json.dumps(tl, indent=2))
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    conn = init_db(Path(args.db))
    if args.format == 'sarif':
        events = [dict(r) for r in conn.execute("SELECT * FROM events ORDER BY generated_at DESC LIMIT 1000").fetchall()]
        sarif = export_sarif(events)
        output = json.dumps(sarif, indent=2)
    elif args.format == 'markdown':
        events = [dict(r) for r in conn.execute("SELECT * FROM events ORDER BY generated_at DESC LIMIT 500").fetchall()]
        output = export_markdown(events=events)
    elif args.format == 'timeline':
        tl = export_timeline(conn)
        output = json.dumps(tl, indent=2)
    else:
        print(f'Unknown format: {args.format}', file=sys.stderr)
        return 1

    conn.close()
    if args.output:
        Path(args.output).write_text(output + '\n')
        print(f'Written to {args.output}', file=sys.stderr)
    else:
        print(output)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description='Artifact Analyzer for UltrafastSecp256k1')
    parser.add_argument('--db', default=str(DEFAULT_DB), help='SQLite database path')
    sub = parser.add_subparsers(dest='command')

    p_ingest = sub.add_parser('ingest', help='Ingest report files')
    p_ingest.add_argument('reports', nargs='+', help='JSON report files to ingest')

    p_diff = sub.add_parser('diff', help='Regression diff between two reports')
    p_diff.add_argument('--before', required=True, help='Before report JSON')
    p_diff.add_argument('--after', required=True, help='After report JSON')
    p_diff.add_argument('--format', choices=['json', 'markdown'], default='json')

    p_div = sub.add_parser('divergence', help='Detect platform divergence')
    p_div.add_argument('reports', nargs='+', help='Reports from different platforms')

    p_flakes = sub.add_parser('flakes', help='Detect flaky checks')
    p_flakes.add_argument('--window', type=int, default=10, help='History window')

    p_tl = sub.add_parser('timeline', help='Export event timeline')
    p_tl.add_argument('--limit', type=int, default=100)

    p_export = sub.add_parser('export', help='Export aggregated data')
    p_export.add_argument('--format', choices=['sarif', 'markdown', 'timeline'], required=True)
    p_export.add_argument('-o', '--output', help='Output file')

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return 0

    handlers = {
        'ingest': cmd_ingest,
        'diff': cmd_diff,
        'divergence': cmd_divergence,
        'flakes': cmd_flakes,
        'timeline': cmd_timeline,
        'export': cmd_export,
    }
    return handlers[args.command](args)


if __name__ == '__main__':
    sys.exit(main())
