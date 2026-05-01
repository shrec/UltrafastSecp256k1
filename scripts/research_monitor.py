#!/usr/bin/env python3
"""Fetch recent secp256k1-related external signals and compare them to repo evidence.

Three-tier triage:
  high_confidence  — score >= THRESHOLD_HIGH and passes crypto relevance gate
  needs_review     — score >= THRESHOLD_REVIEW (log only, no issue)
  discarded        — score < THRESHOLD_REVIEW or dominated by negative terms

GitHub issues are opened ONLY for high_confidence items.
"""

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

# Minimum score thresholds for triage buckets.
# Score is computed by relevance_score() over title + abstract.
THRESHOLD_HIGH   = 10   # high_confidence: score >= this AND mapped or hard focus term
THRESHOLD_REVIEW = 4    # needs_review:    score >= this (logged but no issue)
# Below THRESHOLD_REVIEW → discarded (not shown in issue, only in JSON artifact).

# ---------------------------------------------------------------------------
# Relevance scoring — crypto-domain positive terms
# ---------------------------------------------------------------------------
# Only terms that are unambiguously relevant to secp256k1 / ECC cryptography
# receive high weight. Generic terms that also appear in unrelated fields
# (microarchitecture, forgery, wallet, cache) are removed to prevent
# medical/food/deepfake papers from scoring above zero.
POSITIVE_TERMS: dict[str, int] = {
    # Core ECC primitives
    'secp256k1':              15,
    'elliptic curve':         8,
    'ecdsa':                  10,
    'schnorr':                10,
    'ecdh':                   8,
    'ecies':                  8,
    'bip-340':                10,
    'bip340':                 10,
    # Multi-party / threshold
    'musig':                  9,
    'frost':                  9,
    'threshold signature':    8,
    'distributed key':        7,
    # Field / scalar arithmetic
    'scalar multiplication':  9,
    'field arithmetic':       8,
    'modular inverse':        8,
    'modular arithmetic':     7,
    'glv endomorphism':       10,
    'glv':                    7,
    'safegcd':                10,
    'divsteps':               10,
    'montgomery multiplication': 8,
    'barrett reduction':      7,
    # Security / side-channel
    'side-channel':           9,
    'timing attack':          9,
    'power analysis':         8,
    'differential power':     9,
    'fault injection':        8,
    'nonce reuse':            10,
    'nonce bias':             9,
    'lattice attack':         9,
    'hertzbleed':             10,
    'constant-time':          8,
    'constant time':          8,
    # Zero-knowledge / proofs
    'fiat-shamir':            8,
    'fiat shamir':            8,
    'zero-knowledge proof':   8,
    'bulletproof':            8,
    'pedersen commitment':    8,
    'range proof':            7,
    'dleq':                   8,
    'sigma protocol':         7,
    # GPU / acceleration
    'cuda secp':              9,
    'gpu elliptic':           8,
    'simd elliptic':          8,
    'batch verification':     8,
    # Formal verification
    'formal verification':    7,
    'verified cryptography':  8,
    'coq proof':              7,
    # CVE/NVD (any CVE about crypto libs)
    'cve-':                   6,
    'libsecp256k1':           12,
    'bitcoin core':           6,
    'secp256r1':              5,
    'curve25519':             5,
    'ed25519':                5,
}

# Negative terms that strongly indicate the item is NOT about ECC/secp256k1.
# A large negative score pushes even keyword-matching items into discard.
NEGATIVE_TERMS: dict[str, int] = {
    # Medical / biological
    'depression':     -10,
    'alzheimer':      -10,
    'sleep':          -9,
    'vitamin d':      -9,
    'bone':           -8,
    'dental':         -8,
    'biofilm':        -8,
    'pharmacolog':    -8,
    'neuroblastoma':  -9,
    'stem cell':      -8,
    'aerogel':        -8,
    'tissue':         -7,
    'pregnancy':      -8,
    'asthma':         -8,
    'surgery':        -8,
    'clinical':       -6,
    'hospital':       -6,
    # Food / materials
    'food matrix':    -9,
    'solute transport':-8,
    'emulsion':       -8,
    'porous':         -7,
    'waterjet':       -8,
    # Generic image processing / deepfake (not ECC)
    'deepfake':       -8,
    'face forgery':   -9,
    'image forgery':  -9,
    'facial':         -7,
    'diffusion model':-6,   # image diffusion, not crypto
    'watermark':      -5,
    # E-wallet behavior (not crypto implementation)
    'e-wallet adoption': -9,
    'e-wallet perilaku': -10,
    'konsumtif':      -10,
    # Brake / mechanical
    'wedge brake':    -10,
    'brake-by-wire':  -10,
    # Generic ML/NLP (not ECC)
    'knowledge graph': -5,
    'student attention': -8,
    'language model': -5,
    'reinforcement learning': -4,
    # Unrelated microarchitecture (CPU but not crypto-specific)
    'string search':  -7,
    'hydropower':     -8,
    'traffic volume': -8,
    # Blockchain application-level analysis (uses secp256k1 but doesn't study it)
    'fraudulent activity pattern': -12,
    'suspicious wallet behavior':  -12,
    'transaction graph':           -8,
    'fraud detection':             -7,
    'anomaly detection':           -5,
}

# Hard focus terms: an UNMAPPED item with one of these in the title is
# still worth reviewing even without a signal-matrix match.
# Kept narrow — only terms that are unambiguous in crypto context.
HARD_FOCUS_TERMS = frozenset({
    'secp256k1',
    'ecdsa',
    'schnorr',
    'musig',
    'frost threshold signature',
    'side-channel elliptic',
    'nonce reuse',
    'lattice attack ecdsa',
    'libsecp256k1',
    'bip-340',
    'bip340',
    'glv endomorphism',
    'safegcd',
    'divsteps',
    'hertzbleed',
    'scalar multiplication attack',
})

STATUS_RANK = {
    'gap':         5,
    'candidate':   4,
    'partial':     3,
    'covered':     2,
    'out_of_scope':1,
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
    parser.add_argument('--fail-on-actionable', action='store_true', help='Return exit code 3 when high-confidence actionable items are found.')
    parser.add_argument('--open-issue', action='store_true',
                        help='Open a GitHub issue via `gh` when high-confidence items are found. '
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


def relevance_score(item: SourceItem) -> int:
    """Compute a relevance score for item based on positive/negative term weighting.

    Score >= THRESHOLD_HIGH   → high_confidence bucket (issue-worthy)
    Score >= THRESHOLD_REVIEW → needs_review bucket (log only)
    Score <  THRESHOLD_REVIEW → discard
    """
    haystack = (item.title + ' ' + item.summary).lower()
    score = 0
    for term, weight in POSITIVE_TERMS.items():
        if term in haystack:
            score += weight
    for term, weight in NEGATIVE_TERMS.items():
        if term in haystack:
            score += weight  # weight is already negative
    # NVD items (CVEs) get a baseline boost — any matching CVE is worth reviewing.
    if item.source == 'NVD':
        score += 5
    return score


def has_hard_focus(item: SourceItem) -> bool:
    """Return True if the item title/summary contains a hard-focus crypto term."""
    haystack = (item.title + ' ' + item.summary).lower()
    return any(term in haystack for term in HARD_FOCUS_TERMS)


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
        items.append(
            SourceItem(
                source='arXiv',
                item_id=item_id,
                title=title,
                summary=summary,
                published=published,
                updated=updated,
                url=item_id,
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
    """Classify an item and assign it to a triage bucket.

    Bucket logic:
      high_confidence  score >= THRESHOLD_HIGH  AND (mapped to signal OR hard focus term)
      needs_review     score >= THRESHOLD_REVIEW (not high_confidence)
      discard          score <  THRESHOLD_REVIEW
    """
    haystack = f'{item.title} {item.summary}'.lower()
    score = relevance_score(item)
    hard_focus = has_hard_focus(item)

    # Match against signal matrix
    matched: list[SignalClass] = []
    for signal in classes:
        if any(keyword in haystack for keyword in signal.keywords):
            matched.append(signal)

    # Determine signal-matrix status
    if not matched:
        if score < THRESHOLD_REVIEW and not hard_focus:
            bucket = 'discard'
            status = 'out_of_scope'
            action = 'watch-only'
            reason = f'Score {score} below threshold — unrelated to secp256k1 ECC cryptography.'
        elif score >= THRESHOLD_HIGH and hard_focus:
            bucket = 'high_confidence'
            status = 'unmapped'
            action = 'review'
            reason = (f'Score {score}: hard crypto focus term matched but no signal-matrix class. '
                      'Review whether the taxonomy needs expansion or a new audit surface is required.')
        elif score >= THRESHOLD_REVIEW:
            bucket = 'needs_review'
            status = 'unmapped'
            action = 'review'
            reason = (f'Score {score}: passes relevance threshold but no signal-matrix class. '
                      'May need taxonomy expansion.')
        else:
            bucket = 'discard'
            status = 'out_of_scope'
            action = 'watch-only'
            reason = f'Score {score}: not relevant enough without a hard focus term.'
    else:
        strongest = max(matched, key=lambda e: STATUS_RANK.get(e.status, 0))
        status = strongest.status
        action = strongest.action
        reason = strongest.reason

        # High-confidence requires score above THRESHOLD_HIGH
        if score >= THRESHOLD_HIGH and status in ACTIONABLE_STATUSES:
            bucket = 'high_confidence'
        elif score >= THRESHOLD_REVIEW and status in ACTIONABLE_STATUSES:
            bucket = 'needs_review'
        elif status in ('covered', 'out_of_scope') and score < THRESHOLD_REVIEW:
            bucket = 'discard'
        elif status in ('covered', 'out_of_scope'):
            bucket = 'needs_review'
        else:
            # mapped but low score — demote to needs_review max
            bucket = 'needs_review' if score >= THRESHOLD_REVIEW else 'discard'

    return {
        'status': status,
        'bucket': bucket,
        'score': score,
        'actionable': bucket == 'high_confidence',
        'action': action,
        'reason': reason,
        'matches': [
            {
                'id': s.signal_id,
                'status': s.status,
                'priority': s.priority,
                'action': s.action,
                'repo_evidence': list(s.repo_evidence),
                'reason': s.reason,
            }
            for s in matched
        ],
    }


def build_report(
    items: list[SourceItem],
    classes: list[SignalClass],
    query: str,
    lookback_days: int,
    source_stats: list[dict],
    source_errors: list[dict],
) -> dict:
    classified_items = []
    counts: dict[str, int] = {
        'total_fetched': 0,
        'high_confidence': 0,
        'needs_review': 0,
        'discarded': 0,
        # Legacy keys kept for backward compat with downstream parsers
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
        bucket = classification['bucket']
        status = classification['status']
        counts['total_fetched'] += 1
        counts['total_items'] += 1
        counts[f'{bucket}s' if bucket != 'discard' else 'discarded'] = counts.get(
            f'{bucket}s' if bucket != 'discard' else 'discarded', 0
        ) + 1
        # Update legacy bucket counters
        if bucket == 'high_confidence':
            counts['actionable_items'] += 1
        counts[f'{status}_items'] = counts.get(f'{status}_items', 0) + 1

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

    # Normalise bucket counter names
    counts['high_confidence'] = sum(1 for c in classified_items if c['bucket'] == 'high_confidence')
    counts['needs_review']    = sum(1 for c in classified_items if c['bucket'] == 'needs_review')
    counts['discarded']       = sum(1 for c in classified_items if c['bucket'] == 'discard')

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
        '',
        '## Triage Summary',
        '',
        f"| Bucket | Count |",
        f"|--------|-------|",
        f"| High confidence (issue-worthy) | {counts['high_confidence']} |",
        f"| Needs review (log only) | {counts['needs_review']} |",
        f"| Discarded (noise) | {counts['discarded']} |",
        f"| **Total fetched** | **{counts['total_fetched']}** |",
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

    high_conf = [item for item in report['items'] if item['bucket'] == 'high_confidence']
    review    = [item for item in report['items'] if item['bucket'] == 'needs_review']
    discarded = [item for item in report['items'] if item['bucket'] == 'discard']

    lines.extend(['', '## High-Confidence Actionable Findings', ''])
    if not high_conf:
        lines.append('_No high-confidence findings in this window._')
    for item in high_conf:
        lines.append(f"### {item['title']}")
        lines.append(f"- **Source:** {item['source']}  **Score:** {item['score']}  **Status:** {item['status']}")
        lines.append(f"- **Recommended action:** {item['action']}")
        lines.append(f"- **Published:** {item['published']}")
        lines.append(f"- **URL:** {item['url']}")
        lines.append(f"- **Why flagged:** {item['reason']}")
        if item['matches']:
            lines.append('- **Repo matches:**')
            for match in item['matches']:
                evidence = ', '.join(match['repo_evidence']) or 'none'
                lines.append(f"  - `{match['id']}` [{match['status']}] → {evidence}")
        lines.append(f"- **Summary:** {item['summary']}")
        lines.append('')

    lines.extend(['## Needs Review (log only — no issue)', ''])
    if not review:
        lines.append('_None._')
    for item in review:
        lines.append(f"- [{item['source']}] **{item['title']}** (score={item['score']}, status={item['status']})")
        lines.append(f"  {item['url']}")
    lines.append('')

    lines.extend(['## Discarded as Noise', ''])
    if not discarded:
        lines.append('_None._')
    for item in discarded:
        lines.append(f"- [{item['source']}] {item['title']} (score={item['score']})")

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
        'Triage:',
        f"  high_confidence (issue-worthy): {counts['high_confidence']}",
        f"  needs_review    (log only):     {counts['needs_review']}",
        f"  discarded       (noise):        {counts['discarded']}",
        f"  total fetched:                  {counts['total_fetched']}",
        '',
    ]
    high_conf = [item for item in report['items'] if item['bucket'] == 'high_confidence']
    if high_conf:
        lines.append('High-confidence findings:')
        for item in high_conf:
            lines.append(f"- [{item['score']:+d}] {item['title']} ({item['source']})")
            lines.append(f"  action: {item['action']}")
            lines.append(f"  url: {item['url']}")
            lines.append(f"  why: {item['reason']}")
    else:
        lines.append('High-confidence findings: none')
    if report['source_errors']:
        lines.extend(['', 'Source errors:'])
        for error in report['source_errors']:
            lines.append(f"- {error['source']}: {error['error']}")
    return '\n'.join(lines).rstrip() + '\n'


def render_mail_subject(report: dict) -> str:
    counts = report['counts']
    hc = counts['high_confidence']
    nr = counts['needs_review']
    return f"[Research Monitor] {hc} high-confidence / {nr} needs-review secp256k1 signals"


def render_mail_body(report: dict) -> str:
    counts = report['counts']
    high_conf = [item for item in report['items'] if item['bucket'] == 'high_confidence']
    lines = [
        'UltrafastSecp256k1 external research monitor',
        '',
        f"Generated: {report['generated_at']}",
        f"Query: {report['query']}",
        f"Lookback: {report['lookback_days']} days",
        '',
        f"High-confidence (issue-worthy): {counts['high_confidence']}",
        f"Needs review (log only):        {counts['needs_review']}",
        f"Discarded as noise:             {counts['discarded']}",
        '',
    ]
    if not high_conf:
        lines.append('No high-confidence items were detected in this window.')
    for index, item in enumerate(high_conf, start=1):
        lines.extend(
            [
                f"{index}. {item['title']}",
                f"   Source: {item['source']}  Score: {item['score']}",
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
    """Open a GitHub issue ONLY for high-confidence findings."""
    counts = report['counts']
    hc = counts['high_confidence']
    if hc == 0:
        return
    date_str = report['generated_at'][:10]
    title = f"[Research Monitor] {hc} high-confidence signal(s) — {date_str}"

    body_lines = [
        '> Auto-opened by `scripts/research_monitor.py --open-issue`.',
        '> Only **high-confidence** items appear here (score ≥ 10 + crypto focus term).',
        '> Needs-review and discarded items are in the build artifact only.',
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
        f"high_confidence_count={counts['high_confidence']}",
        f"needs_review_count={counts['needs_review']}",
        f"discarded_count={counts['discarded']}",
        # Legacy key for any downstream parsers
        f"actionable_count={counts['high_confidence']}",
        f"total_count={counts['total_fetched']}",
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
              high-confidence (issue-worthy): {counts['high_confidence']}
              needs-review    (log only):     {counts['needs_review']}
              discarded       (noise):        {counts['discarded']}
              total fetched:                  {counts['total_fetched']}
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

    # Query expansion: only crypto-specific phrases to avoid broad noise.
    # Each additional query targets a distinct attack class or optimization area
    # that may not appear with "secp256k1" verbatim.
    queries = [args.query]
    if args.query == 'secp256k1':
        queries.extend([
            'ecdsa nonce bias lattice',
            'schnorr signature forgery',
            'elliptic curve side channel',
            'scalar multiplication timing',
            'libsecp256k1',
        ])

    raw_items: list[SourceItem] = []
    source_stats: list[dict] = []
    source_errors: list[dict] = []

    for q in queries:
        q_items, q_stats, q_errors = collect_items(q, args.max_results, cutoff)
        raw_items.extend(q_items)
        source_stats.extend(q_stats)
        source_errors.extend(q_errors)

    if all(source['status'] == 'error' for source in source_stats):
        raise RuntimeError('all external sources failed; refusing to emit a misleading empty report')

    items = dedupe_and_filter(raw_items, cutoff)
    report = build_report(items, classes, args.query, args.lookback_days, source_stats, source_errors)
    write_outputs(report, args.output_dir)
    if args.github_output:
        write_github_outputs(args.github_output, report)
    print_console_summary(report)

    if args.open_issue and report['counts']['high_confidence'] > 0:
        open_github_issue(report)
    if args.fail_on_actionable and report['counts']['high_confidence'] > 0:
        return 3
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f'error: {exc}', file=sys.stderr)
        raise
