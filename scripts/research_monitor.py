#!/usr/bin/env python3
"""Fetch recent secp256k1-related external signals and compare them to repo evidence."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import textwrap
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from html import unescape
from pathlib import Path
from typing import Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
LIB_ROOT = SCRIPT_DIR.parent
DEFAULT_OUTPUT_DIR = LIB_ROOT / 'build' / 'research_monitor'
DEFAULT_MATRIX = LIB_ROOT / 'docs' / 'RESEARCH_SIGNAL_MATRIX.json'
USER_AGENT = 'UltrafastSecp256k1ResearchMonitor/1.0 (+https://github.com/shrec/UltrafastSecp256k1)'
ARXIV_NS = {'atom': 'http://www.w3.org/2005/Atom'}
FOCUS_TERMS = (
    'attack',
    'break',
    'breaking',
    'exploit',
    'side-channel',
    'side channel',
    'timing',
    'nonce',
    'ecdsa',
    'schnorr',
    'frost',
    'musig',
    'ecdh',
    'glv',
    'zvp',
    'safegcd',
    'divsteps',
    'formal verification',
    'verifiable c',
    'coq',
    'fault',
    'wallet',
    'batch verification',
    'precomputation',
    'precompute',
    'optimization',
    'accelerat',
    'performance',
    'benchmark',
    'cve',
)

STATUS_RANK = {
    'gap': 5,
    'candidate': 4,
    'partial': 3,
    'covered': 2,
    'out_of_scope': 1,
}

ACTIONABLE_STATUSES = {'gap', 'candidate', 'partial', 'unmapped'}


@dataclass(frozen=True)
class SignalClass:
    signal_id: str
    status: str
    priority: str
    action: str
    keywords: tuple[str, ...]
    repo_evidence: tuple[str, ...]
    reason: str


@dataclass
class SourceItem:
    source: str
    item_id: str
    title: str
    summary: str
    published: datetime
    updated: datetime
    url: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--query', default='secp256k1', help='Search term for external sources.')
    parser.add_argument('--lookback-days', type=int, default=14, help='Only include items published or updated within this many days.')
    parser.add_argument('--max-results', type=int, default=12, help='Maximum items per source before filtering.')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--matrix', type=Path, default=DEFAULT_MATRIX)
    parser.add_argument('--github-output', type=Path, help='Optional GitHub output file for workflow outputs.')
    parser.add_argument('--fail-on-actionable', action='store_true', help='Return exit code 3 when actionable items are found.')
    parser.add_argument('--open-issue', action='store_true',
                        help='Open a GitHub issue via `gh` when actionable items are found. '
                             'Requires `gh` CLI authenticated to the repo.')
    return parser.parse_args()


def http_get_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={'User-Agent': USER_AGENT, 'Accept': 'application/json'})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def http_get_text(url: str, accept: str = 'application/atom+xml,text/xml,text/plain') -> str:
    request = urllib.request.Request(url, headers={'User-Agent': USER_AGENT, 'Accept': accept})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read().decode('utf-8', errors='replace')


def parse_timestamp(raw: str) -> datetime:
    text = raw.strip()
    if text.endswith('Z'):
        text = text[:-1] + '+00:00'
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def strip_markup(text: str) -> str:
    no_tags = re.sub(r'<[^>]+>', ' ', text)
    compact = re.sub(r'\s+', ' ', unescape(no_tags))
    return compact.strip()


def normalize_title(title: str) -> str:
    return re.sub(r'\s+', ' ', title.strip()).lower()


def short_summary(text: str, limit: int = 280) -> str:
    compact = re.sub(r'\s+', ' ', text.strip())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + '...'


def load_signal_matrix(path: Path) -> list[SignalClass]:
    payload = json.loads(path.read_text(encoding='utf-8'))
    classes = []
    for entry in payload.get('classes', []):
        classes.append(
            SignalClass(
                signal_id=entry['id'],
                status=entry['status'],
                priority=entry['priority'],
                action=entry['action'],
                keywords=tuple(keyword.lower() for keyword in entry.get('keywords', [])),
                repo_evidence=tuple(entry.get('repo_evidence', [])),
                reason=entry['reason'],
            )
        )
    return classes


def validate_matrix_evidence(classes: Iterable[SignalClass]) -> None:
    missing: list[str] = []
    for signal in classes:
        for rel_path in signal.repo_evidence:
            if not (LIB_ROOT / rel_path).exists():
                missing.append(f'{signal.signal_id}: {rel_path}')
    if missing:
        joined = '\n'.join(f'  - {item}' for item in missing)
        raise FileNotFoundError('research signal matrix references missing repo evidence paths:\n' + joined)


def fetch_arxiv(query: str, max_results: int) -> list[SourceItem]:
    params = urllib.parse.urlencode(
        {
            'search_query': f'ti:"{query}" OR abs:"{query}"',
            'start': 0,
            'max_results': max_results,
            'sortBy': 'lastUpdatedDate',
            'sortOrder': 'descending',
        }
    )
    xml_text = http_get_text(f'https://export.arxiv.org/api/query?{params}')
    root = ET.fromstring(xml_text)
    items: list[SourceItem] = []
    for entry in root.findall('atom:entry', ARXIV_NS):
        title = strip_markup(entry.findtext('atom:title', default='', namespaces=ARXIV_NS))
        summary = strip_markup(entry.findtext('atom:summary', default='', namespaces=ARXIV_NS))
        item_id = entry.findtext('atom:id', default='', namespaces=ARXIV_NS).strip()
        published = parse_timestamp(entry.findtext('atom:published', default='', namespaces=ARXIV_NS))
        updated = parse_timestamp(entry.findtext('atom:updated', default='', namespaces=ARXIV_NS))
        url = item_id
        items.append(
            SourceItem(
                source='arXiv',
                item_id=item_id,
                title=title,
                summary=summary,
                published=published,
                updated=updated,
                url=url,
            )
        )
    return items


def fetch_crossref(query: str, max_results: int, cutoff: datetime) -> list[SourceItem]:
    params = urllib.parse.urlencode(
        {
            'query.bibliographic': query,
            'rows': max_results,
            'sort': 'updated',
            'order': 'desc',
            'filter': f'from-pub-date:{cutoff.date().isoformat()}',
            'mailto': 'research-monitor@example.invalid',
        }
    )
    payload = http_get_json(f'https://api.crossref.org/works?{params}')
    items: list[SourceItem] = []
    for item in payload.get('message', {}).get('items', []):
        title_parts = item.get('title') or []
        title = strip_markup(title_parts[0] if title_parts else '')
        abstract = strip_markup(item.get('abstract', ''))
        if not title:
            continue
        doi = item.get('DOI', title)
        published_parts = item.get('published-print') or item.get('published-online') or item.get('issued') or {}
        date_parts = published_parts.get('date-parts', [[1970, 1, 1]])[0]
        year = int(date_parts[0])
        month = int(date_parts[1]) if len(date_parts) > 1 else 1
        day = int(date_parts[2]) if len(date_parts) > 2 else 1
        published = datetime(year, month, day, tzinfo=timezone.utc)
        updated_raw = item.get('indexed', {}).get('date-time') or item.get('created', {}).get('date-time') or published.isoformat()
        updated = parse_timestamp(updated_raw)
        url = item.get('URL') or f'https://doi.org/{doi}'
        items.append(
            SourceItem(
                source='Crossref',
                item_id=doi,
                title=title,
                summary=abstract,
                published=published,
                updated=updated,
                url=url,
            )
        )
    return items


def fetch_nvd(query: str, max_results: int, cutoff: datetime) -> list[SourceItem]:
    params = urllib.parse.urlencode(
        {
            'keywordSearch': query,
            'resultsPerPage': max_results,
        }
    )
    payload = http_get_json(f'https://services.nvd.nist.gov/rest/json/cves/2.0?{params}')
    items: list[SourceItem] = []
    for wrapper in payload.get('vulnerabilities', []):
        cve = wrapper.get('cve', {})
        cve_id = cve.get('id', 'CVE-UNKNOWN')
        descriptions = cve.get('descriptions', [])
        english = next((d.get('value', '') for d in descriptions if d.get('lang') == 'en'), '')
        title = cve.get('sourceIdentifier', 'NVD') + ': ' + cve_id
        published = parse_timestamp(cve.get('published', '1970-01-01T00:00:00Z'))
        updated = parse_timestamp(cve.get('lastModified', cve.get('published', '1970-01-01T00:00:00Z')))
        url = f'https://nvd.nist.gov/vuln/detail/{cve_id}'
        items.append(
            SourceItem(
                source='NVD',
                item_id=cve_id,
                title=title,
                summary=strip_markup(english),
                published=published,
                updated=updated,
                url=url,
            )
        )
    return items


def collect_items(query: str, max_results: int, cutoff: datetime) -> tuple[list[SourceItem], list[dict], list[dict]]:
    all_items: list[SourceItem] = []
    source_stats: list[dict] = []
    source_errors: list[dict] = []
    fetchers = [
        ('arXiv', lambda: fetch_arxiv(query, max_results)),
        ('Crossref', lambda: fetch_crossref(query, max_results, cutoff)),
        ('NVD', lambda: fetch_nvd(query, max_results, cutoff)),
    ]
    for name, fetcher in fetchers:
        try:
            items = fetcher()
            all_items.extend(items)
            source_stats.append({'source': name, 'count': len(items), 'status': 'ok'})
        except Exception as exc:
            source_errors.append({'source': name, 'error': str(exc)})
            source_stats.append({'source': name, 'count': 0, 'status': 'error'})
    return all_items, source_stats, source_errors


def dedupe_and_filter(items: Iterable[SourceItem], cutoff: datetime) -> list[SourceItem]:
    deduped: dict[str, SourceItem] = {}
    for item in items:
        if max(item.published, item.updated) < cutoff:
            continue
        key = item.url.strip() or normalize_title(item.title)
        existing = deduped.get(key)
        if existing is None or item.updated > existing.updated:
            deduped[key] = item
    return sorted(deduped.values(), key=lambda item: max(item.updated, item.published), reverse=True)


def classify_item(item: SourceItem, classes: Iterable[SignalClass]) -> dict:
    haystack = f'{item.title} {item.summary}'.lower()
    title_haystack = item.title.lower()
    matched: list[SignalClass] = []
    for signal in classes:
        if any(keyword in haystack for keyword in signal.keywords):
            matched.append(signal)
    if not matched:
        if item.source != 'NVD' and not any(term in title_haystack for term in FOCUS_TERMS):
            return {
                'status': 'out_of_scope',
                'actionable': False,
                'action': 'watch-only',
                'reason': 'The item mentions secp256k1 but does not look like a direct security, correctness, protocol-misuse, or optimization signal for this repository.',
                'matches': [],
            }
        return {
            'status': 'unmapped',
            'actionable': True,
            'action': 'review',
            'reason': 'No checked-in signal class matched this item. Review whether the taxonomy needs expansion or the repository needs a new deterministic audit surface.',
            'matches': [],
        }

    strongest = max(matched, key=lambda entry: STATUS_RANK.get(entry.status, 0))
    return {
        'status': strongest.status,
        'actionable': strongest.status in ACTIONABLE_STATUSES,
        'action': strongest.action,
        'reason': strongest.reason,
        'matches': [
            {
                'id': signal.signal_id,
                'status': signal.status,
                'priority': signal.priority,
                'action': signal.action,
                'repo_evidence': list(signal.repo_evidence),
                'reason': signal.reason,
            }
            for signal in matched
        ],
    }


def build_report(items: list[SourceItem], classes: list[SignalClass], query: str, lookback_days: int, source_stats: list[dict], source_errors: list[dict]) -> dict:
    classified_items = []
    counts = {
        'total_items': 0,
        'actionable_items': 0,
        'covered_items': 0,
        'candidate_items': 0,
        'gap_items': 0,
        'out_of_scope_items': 0,
        'unmapped_items': 0,
    }

    for item in items:
        classification = classify_item(item, classes)
        status = classification['status']
        counts['total_items'] += 1
        counts[f'{status}_items'] = counts.get(f'{status}_items', 0) + 1
        if classification['actionable']:
            counts['actionable_items'] += 1
        classified_items.append(
            {
                'source': item.source,
                'item_id': item.item_id,
                'title': item.title,
                'summary': short_summary(item.summary),
                'published': item.published.isoformat(),
                'updated': item.updated.isoformat(),
                'url': item.url,
                **classification,
            }
        )

    generated_at = datetime.now(timezone.utc)
    return {
        'generated_at': generated_at.isoformat(),
        'generated_rfc2822': format_datetime(generated_at),
        'query': query,
        'lookback_days': lookback_days,
        'sources': source_stats,
        'source_errors': source_errors,
        'counts': counts,
        'items': classified_items,
    }


def render_markdown(report: dict) -> str:
    counts = report['counts']
    lines = [
        '# External Research Monitor Report',
        '',
        f"- Generated: {report['generated_at']}",
        f"- Query: {report['query']}",
        f"- Lookback: {report['lookback_days']} days",
        f"- Actionable items: {counts['actionable_items']}",
        '',
        '## Source Status',
        '',
    ]
    for source in report['sources']:
        lines.append(f"- {source['source']}: {source['status']} ({source['count']} raw items)")

    if report['source_errors']:
        lines.extend(['', '## Source Errors', ''])
        for error in report['source_errors']:
            lines.append(f"- {error['source']}: {error['error']}")

    actionable = [item for item in report['items'] if item['actionable']]
    informational = [item for item in report['items'] if not item['actionable']]

    lines.extend(['', '## Actionable Findings', ''])
    if not actionable:
        lines.append('- No actionable items found in this window.')
    for item in actionable:
        lines.append(f"### {item['title']}")
        lines.append(f"- Source: {item['source']}")
        lines.append(f"- Status: {item['status']}")
        lines.append(f"- Recommended action: {item['action']}")
        lines.append(f"- Published: {item['published']}")
        lines.append(f"- URL: {item['url']}")
        lines.append(f"- Why flagged: {item['reason']}")
        if item['matches']:
            lines.append('- Repo matches:')
            for match in item['matches']:
                evidence = ', '.join(match['repo_evidence']) or 'none'
                lines.append(f"  - {match['id']} [{match['status']}] -> {evidence}")
        lines.append(f"- Summary: {item['summary']}")
        lines.append('')

    lines.extend(['## Informational Findings', ''])
    if not informational:
        lines.append('- No informational items in this window.')
    for item in informational:
        lines.append(f"- [{item['source']}] {item['title']} ({item['status']})")

    return '\n'.join(lines).rstrip() + '\n'


def render_text(report: dict) -> str:
    counts = report['counts']
    lines = [
        'UltrafastSecp256k1 External Research Monitor',
        '===========================================',
        f"Generated: {report['generated_at']}",
        f"Query: {report['query']}",
        f"Lookback: {report['lookback_days']} days",
        '',
        'Counts:',
        f"  total:       {counts['total_items']}",
        f"  actionable:  {counts['actionable_items']}",
        f"  covered:     {counts['covered_items']}",
        f"  candidate:   {counts['candidate_items']}",
        f"  gap:         {counts['gap_items']}",
        f"  out_of_scope:{counts['out_of_scope_items']}",
        f"  unmapped:    {counts['unmapped_items']}",
        '',
    ]
    actionable = [item for item in report['items'] if item['actionable']]
    if actionable:
        lines.append('Actionable findings:')
        for item in actionable:
            lines.append(f"- {item['title']} [{item['status']}] ({item['source']})")
            lines.append(f"  action: {item['action']}")
            lines.append(f"  url: {item['url']}")
            lines.append(f"  why: {item['reason']}")
    else:
        lines.append('Actionable findings: none')
    if report['source_errors']:
        lines.extend(['', 'Source errors:'])
        for error in report['source_errors']:
            lines.append(f"- {error['source']}: {error['error']}")
    return '\n'.join(lines).rstrip() + '\n'


def render_mail_subject(report: dict) -> str:
    counts = report['counts']
    return f"[Research Monitor] {counts['actionable_items']} actionable / {counts['total_items']} total secp256k1 signals"


def render_mail_body(report: dict) -> str:
    counts = report['counts']
    actionable = [item for item in report['items'] if item['actionable']]
    lines = [
        'UltrafastSecp256k1 external research monitor',
        '',
        f"Generated: {report['generated_at']}",
        f"Query: {report['query']}",
        f"Lookback: {report['lookback_days']} days",
        '',
        f"Actionable items: {counts['actionable_items']}",
        f"Covered items: {counts['covered_items']}",
        f"Candidate items: {counts['candidate_items']}",
        f"Gap items: {counts['gap_items']}",
        f"Unmapped items: {counts['unmapped_items']}",
        '',
    ]
    if not actionable:
        lines.append('No actionable items were detected in this window.')
    for index, item in enumerate(actionable, start=1):
        lines.extend(
            [
                f"{index}. {item['title']}",
                f"   Source: {item['source']}",
                f"   Status: {item['status']}",
                f"   Action: {item['action']}",
                f"   Published: {item['published']}",
                f"   URL: {item['url']}",
                f"   Why: {item['reason']}",
                f"   Summary: {item['summary']}",
                '',
            ]
        )
    if report['source_errors']:
        lines.append('Source errors:')
        for error in report['source_errors']:
            lines.append(f"- {error['source']}: {error['error']}")
    return '\n'.join(lines).rstrip() + '\n'


def open_github_issue(report: dict) -> None:
    """Open a GitHub issue with the research monitor findings using the `gh` CLI."""
    counts = report['counts']
    actionable = counts['actionable_items']
    date_str = report['generated_at'][:10]  # YYYY-MM-DD
    title = f"[Research Monitor] {actionable} actionable signal(s) — {date_str}"

    body_lines = [
        '> Auto-opened by `scripts/research_monitor.py --open-issue`.',
        '> Review each finding below, add a PoC / CI gate, then close this issue.',
        '',
        render_markdown(report),
    ]
    body = '\n'.join(body_lines)

    cmd = [
        'gh', 'issue', 'create',
        '--title', title,
        '--body', body,
        '--label', 'research-signal,security',
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        url = result.stdout.strip()
        print(f'GitHub issue opened: {url}')
    except FileNotFoundError:
        print('warning: `gh` CLI not found — skipping issue creation', file=sys.stderr)
    except subprocess.CalledProcessError as exc:
        # Label may not exist yet; retry without labels
        if 'Label' in (exc.stderr or ''):
            cmd_no_label = [a for a in cmd if a not in ('--label', 'research-signal,security')]
            try:
                result = subprocess.run(cmd_no_label, capture_output=True, text=True, check=True)
                print(f'GitHub issue opened (no labels): {result.stdout.strip()}')
            except subprocess.CalledProcessError as exc2:
                print(f'warning: gh issue create failed: {exc2.stderr}', file=sys.stderr)
        else:
            print(f'warning: gh issue create failed: {exc.stderr}', file=sys.stderr)


def write_outputs(report: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'research_report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    (output_dir / 'research_report.md').write_text(render_markdown(report), encoding='utf-8')
    (output_dir / 'research_report.txt').write_text(render_text(report), encoding='utf-8')
    (output_dir / 'mail_subject.txt').write_text(render_mail_subject(report) + '\n', encoding='utf-8')
    (output_dir / 'mail_body.txt').write_text(render_mail_body(report), encoding='utf-8')


def write_github_outputs(path: Path, report: dict) -> None:
    counts = report['counts']
    lines = [
        f"actionable_count={counts['actionable_items']}",
        f"total_count={counts['total_items']}",
        f"source_error_count={len(report['source_errors'])}",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as handle:
        handle.write('\n'.join(lines) + '\n')


def print_console_summary(report: dict) -> None:
    counts = report['counts']
    print(
        textwrap.dedent(
            f"""
            Research monitor completed.
              total items:      {counts['total_items']}
              actionable items: {counts['actionable_items']}
              covered items:    {counts['covered_items']}
              candidate items:  {counts['candidate_items']}
              gap items:        {counts['gap_items']}
              unmapped items:   {counts['unmapped_items']}
            """
        ).strip()
    )
    if report['source_errors']:
        print('Source errors:')
        for error in report['source_errors']:
            print(f"  - {error['source']}: {error['error']}")


def main() -> int:
    args = parse_args()
    classes = load_signal_matrix(args.matrix)
    validate_matrix_evidence(classes)

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=args.lookback_days)
    raw_items, source_stats, source_errors = collect_items(args.query, args.max_results, cutoff)
    if all(source['status'] == 'error' for source in source_stats):
        raise RuntimeError('all external sources failed; refusing to emit a misleading empty report')

    items = dedupe_and_filter(raw_items, cutoff)
    report = build_report(items, classes, args.query, args.lookback_days, source_stats, source_errors)
    write_outputs(report, args.output_dir)
    if args.github_output:
        write_github_outputs(args.github_output, report)
    print_console_summary(report)

    if args.open_issue and report['counts']['actionable_items'] > 0:
        open_github_issue(report)
    if args.fail_on_actionable and report['counts']['actionable_items'] > 0:
        return 3
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f'error: {exc}', file=sys.stderr)
        raise