#!/usr/bin/env python3
"""Fetch recent secp256k1-related external signals and compare them to repo evidence.

Three-tier triage:
  high_confidence  — score >= THRESHOLD_HIGH and passes crypto relevance gate
  needs_review     — score >= THRESHOLD_REVIEW (review queue)
  discarded        — score < THRESHOLD_REVIEW or dominated by negative terms

GitHub issues are opened for high_confidence items by default. The optional
review-escalation mode also opens an issue when only needs_review items exist.
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
from email.utils import format_datetime, parsedate_to_datetime
from html import unescape
from pathlib import Path
from typing import Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
LIB_ROOT = SCRIPT_DIR.parent
DEFAULT_OUTPUT_DIR = LIB_ROOT / 'out' / 'research_monitor'
DEFAULT_MATRIX = LIB_ROOT / 'docs' / 'RESEARCH_SIGNAL_MATRIX.json'
USER_AGENT = 'UltrafastSecp256k1ResearchMonitor/1.0 (+https://github.com/shrec/UltrafastSecp256k1)'
EPRINT_RSS_URL = 'https://eprint.iacr.org/rss/rss.xml'
RESEARCH_ISSUE_LABELS = 'research-monitor,security,triage'
ARXIV_NS = {'atom': 'http://www.w3.org/2005/Atom'}
RSS_DC_NS = {'dc': 'http://purl.org/dc/elements/1.1/'}
QUERY_STOP_WORDS = {
    'a', 'an', 'and', 'for', 'from', 'in', 'of', 'on', 'or', 'the', 'to', 'with',
}
PREFIX_TERMS = {
    'cve-',
    'pharmacolog',
}

# Minimum score thresholds for triage buckets.
# Score is computed by relevance_score() over title + abstract.
THRESHOLD_HIGH   = 10   # high_confidence: score >= this AND mapped or hard focus term
THRESHOLD_REVIEW = 4    # needs_review:    score >= this (review queue)
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
    # Post-quantum schemes — NOT secp256k1/ECC. A "side-channel" paper on ML-KEM /
    # Kyber / Dilithium scores ~33 on the generic side-channel term weights, but it
    # is out of scope; a strong penalty keeps it out of the needs-review queue.
    # NOTE: 'lattice-based' (a PQ construction) != the positive 'lattice attack'
    # (lattice cryptanalysis of ECDSA nonces) — different phrases, both word-bounded.
    'ml-kem':                 -30,
    'mlkem':                  -30,
    'ml-dsa':                 -30,
    'mldsa':                  -30,
    'kyber':                  -25,
    'dilithium':              -25,
    'sphincs':                -20,
    'lattice-based':          -10,
    'module-lwe':             -15,
    'module-lattice':         -15,
    'learning with errors':   -15,
    'post-quantum':           -8,
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
    # Bastion B18: attack-class taxonomy + evidence routing.
    attack_class: str = ''
    affected_primitive: str = ''
    affected_surface: str = ''
    expected_gate: str = ''
    missing_evidence_action: str = ''


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
    parser.add_argument('--open-review-issue', action='store_true',
                        help='With --open-issue, also open an issue when needs-review items exist.')
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


def parse_rss_timestamp(raw: str) -> datetime:
    parsed = parsedate_to_datetime(raw.strip())
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _bounded_date_part(values: object, index: int, default: int, low: int, high: int) -> int:
    if not isinstance(values, (list, tuple)) or index >= len(values):
        return default
    try:
        value = int(values[index])
    except (TypeError, ValueError):
        return default
    if value < low or value > high:
        return default
    return value


def parse_crossref_date_parts(date_parts: object) -> datetime:
    """Parse Crossref date-parts defensively.

    Crossref occasionally returns partial or malformed arrays. A single bad
    date should not make the whole source fail.
    """
    values = date_parts
    if isinstance(date_parts, (list, tuple)) and date_parts:
        first = date_parts[0]
        if isinstance(first, (list, tuple)):
            values = first

    year = _bounded_date_part(values, 0, 1970, 1, 9999)
    month = _bounded_date_part(values, 1, 1, 1, 12)
    day = _bounded_date_part(values, 2, 1, 1, 31)
    try:
        return datetime(year, month, day, tzinfo=timezone.utc)
    except ValueError:
        return datetime(year, month, 1, tzinfo=timezone.utc)


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


def compact_report_error(error: object, limit: int = 240) -> str:
    compact = re.sub(r'\s+', ' ', str(error)).strip()
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + '...'


def source_report_label(record: dict) -> str:
    source = str(record.get('source', 'unknown'))
    query = str(record.get('query', '')).strip()
    if query:
        return f'{source} [{query}]'
    return source


def term_matches(text: str, term: str) -> bool:
    haystack = normalize_title(text)
    needle = normalize_title(term)
    if not needle:
        return False
    if needle in PREFIX_TERMS or needle.endswith('-'):
        pattern = rf'(?<![a-z0-9]){re.escape(needle)}'
    else:
        pattern = rf'(?<![a-z0-9]){re.escape(needle)}(?![a-z0-9])'
    return re.search(pattern, haystack) is not None


def source_query_match(query: str, text: str) -> bool:
    query_text = normalize_title(query)
    haystack = normalize_title(text)
    if not query_text:
        return True
    if term_matches(haystack, query_text):
        return True

    terms = [
        term for term in re.split(r'[^a-z0-9]+', query_text)
        if len(term) >= 3 and term not in QUERY_STOP_WORDS
    ]
    if not terms:
        return True
    hits = sum(1 for term in terms if term_matches(haystack, term))
    required = 1 if len(terms) <= 2 else 2
    return hits >= required


def relevance_score(item: SourceItem) -> int:
    """Compute a relevance score for item based on positive/negative term weighting.

    Score >= THRESHOLD_HIGH   → high_confidence bucket (issue-worthy)
    Score >= THRESHOLD_REVIEW → needs_review bucket (log only)
    Score <  THRESHOLD_REVIEW → discard
    """
    haystack = (item.title + ' ' + item.summary).lower()
    score = 0
    for term, weight in POSITIVE_TERMS.items():
        if term_matches(haystack, term):
            score += weight
    for term, weight in NEGATIVE_TERMS.items():
        if term_matches(haystack, term):
            score += weight  # weight is already negative
    # NVD items (CVEs) get a baseline boost — any matching CVE is worth reviewing.
    if item.source == 'NVD':
        score += 5
    return score


def has_hard_focus(item: SourceItem) -> bool:
    """Return True if the item title/summary contains a hard-focus crypto term."""
    haystack = (item.title + ' ' + item.summary).lower()
    return any(term_matches(haystack, term) for term in HARD_FOCUS_TERMS)


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
                attack_class=entry.get('attack_class', ''),
                affected_primitive=entry.get('affected_primitive', ''),
                affected_surface=entry.get('affected_surface', ''),
                expected_gate=entry.get('expected_gate', ''),
                missing_evidence_action=entry.get('missing_evidence_action', ''),
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


def fetch_eprint(query: str, max_results: int, cutoff: datetime) -> list[SourceItem]:
    xml_text = http_get_text(EPRINT_RSS_URL, accept='application/rss+xml,application/xml,text/xml,text/plain')
    root = ET.fromstring(xml_text)
    items: list[SourceItem] = []
    for entry in root.findall('./channel/item'):
        title = strip_markup(entry.findtext('title', default=''))
        summary = strip_markup(entry.findtext('description', default=''))
        link = entry.findtext('link', default='').strip()
        guid = entry.findtext('guid', default=link).strip() or link or title
        categories = [
            strip_markup(category.text or '')
            for category in entry.findall('category')
            if (category.text or '').strip()
        ]
        creators = [
            strip_markup(creator.text or '')
            for creator in entry.findall('dc:creator', RSS_DC_NS)
            if (creator.text or '').strip()
        ]
        pub_raw = entry.findtext('pubDate', default='Thu, 01 Jan 1970 00:00:00 +0000')
        published = parse_rss_timestamp(pub_raw)
        if published < cutoff:
            continue

        searchable = ' '.join([title, summary, *categories, *creators])
        if not source_query_match(query, searchable):
            continue

        details = []
        if categories:
            details.append('Categories: ' + ', '.join(categories))
        if creators:
            details.append('Authors: ' + ', '.join(creators))
        details.append(summary)

        items.append(
            SourceItem(
                source='IACR ePrint',
                item_id=guid,
                title=title,
                summary=' '.join(part for part in details if part).strip(),
                published=published,
                updated=published,
                url=link or guid,
            )
        )
        if len(items) >= max_results:
            break
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
        date_parts = published_parts.get('date-parts') if isinstance(published_parts, dict) else None
        published = parse_crossref_date_parts(date_parts)
        indexed = item.get('indexed') or {}
        created = item.get('created') or {}
        indexed_time = indexed.get('date-time') if isinstance(indexed, dict) else None
        created_time = created.get('date-time') if isinstance(created, dict) else None
        updated_raw = indexed_time or created_time or published.isoformat()
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
        ('IACR ePrint', lambda: fetch_eprint(query, max_results, cutoff)),
        ('arXiv', lambda: fetch_arxiv(query, max_results)),
        ('Crossref', lambda: fetch_crossref(query, max_results, cutoff)),
        ('NVD', lambda: fetch_nvd(query, max_results, cutoff)),
    ]
    for name, fetcher in fetchers:
        try:
            items = fetcher()
            all_items.extend(items)
            source_stats.append({'source': name, 'query': query, 'count': len(items), 'status': 'ok'})
        except Exception as exc:
            source_errors.append({'source': name, 'query': query, 'error': compact_report_error(exc)})
            source_stats.append({'source': name, 'query': query, 'count': 0, 'status': 'error'})
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
        if any(term_matches(haystack, keyword) for keyword in signal.keywords):
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
                'attack_class': s.attack_class,
                'affected_primitive': s.affected_primitive,
                'affected_surface': s.affected_surface,
                'expected_gate': s.expected_gate,
                'missing_evidence_action': s.missing_evidence_action,
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
        f"| Needs review (review queue) | {counts['needs_review']} |",
        f"| Discarded (noise) | {counts['discarded']} |",
        f"| **Total fetched** | **{counts['total_fetched']}** |",
        '',
        '## Source Status',
        '',
    ]
    for source in report['sources']:
        lines.append(f"- {source_report_label(source)}: {source['status']} ({source['count']} raw items)")

    if report['source_errors']:
        lines.extend(['', '## Source Errors', ''])
        for error in report['source_errors']:
            lines.append(f"- {source_report_label(error)}: {error['error']}")

    high_conf = [item for item in report['items'] if item['bucket'] == 'high_confidence']
    review    = [item for item in report['items'] if item['bucket'] == 'needs_review']
    discarded = [item for item in report['items'] if item['bucket'] == 'discard']

    lines.extend(['', '## High-Confidence Actionable Findings', ''])
    if not high_conf:
        lines.append('_No high-confidence findings in this window._')
    for item in high_conf:
        matches = item.get('matches') or []
        # Bastion B6 + B18: turn a signal into routed audit work — derive the
        # attack class, affected primitive/surface, expected gate, and the
        # missing-evidence action from the matched signal classes.
        coverage = sorted({m['status'] for m in matches})
        evidence_paths = sorted({e for m in matches for e in m.get('repo_evidence', [])})
        attack_classes = sorted({m.get('attack_class') for m in matches if m.get('attack_class')})
        primitives = sorted({m.get('affected_primitive') for m in matches if m.get('affected_primitive')})
        surfaces = sorted({m.get('affected_surface') for m in matches if m.get('affected_surface')})
        gates = sorted({m.get('expected_gate') for m in matches if m.get('expected_gate')})
        actions = [m.get('missing_evidence_action') for m in matches if m.get('missing_evidence_action')]
        is_gap = any(m['status'] in ('gap', 'candidate', 'unmapped') for m in matches) or not matches

        lines.append(f"### {item['title']}")
        lines.append(f"- **Source:** {item['source']}  **Score:** {item['score']}  **Status:** {item['status']}")
        lines.append(f"- **Attack class:** "
                     + (', '.join(attack_classes) if attack_classes else 'unmapped — taxonomy expansion candidate'))
        lines.append(f"- **Affected primitive:** " + (', '.join(primitives) if primitives else 'unmapped'))
        lines.append(f"- **Affected surface:** " + (', '.join(surfaces) if surfaces else 'unmapped'))
        lines.append(f"- **Expected gate:** "
                     + (', '.join(f"`{g}`" for g in gates) if gates else 'route to a CAAS gate'))
        lines.append(f"- **Coverage status:** " + (', '.join(coverage) if coverage else 'unmapped'))
        lines.append(f"- **Recommended action:** {item['action']}")
        if evidence_paths:
            lines.append("- **Existing evidence:** "
                         + ', '.join(f"`{p}`" for p in evidence_paths[:6])
                         + (' …' if len(evidence_paths) > 6 else ''))
        lines.append(f"- **Published:** {item['published']}")
        lines.append(f"- **URL:** {item['url']}")
        lines.append(f"- **Why flagged:** {item['reason']}")
        if matches:
            lines.append('- **Repo matches:**')
            for match in matches:
                evidence = ', '.join(match.get('repo_evidence', [])) or 'none'
                ac = match.get('attack_class') or '?'
                lines.append(f"  - `{match['id']}` [{match['status']}] ({ac}) → {evidence}")
        lines.append(f"- **Summary:** {item['summary']}")
        # Bastion B18: patch plan — the matched signal's missing_evidence_action is
        # the FIRST step, then verification + the routed gate.
        lines.append('- **Patch plan:**')
        if actions:
            lines.append(f"  - {actions[0]}")
        lines.append(f"  - First verification: `python3 ci/research_monitor.py "
                     f"--lookback-days {report['lookback_days']} --max-results 10`")
        lines.append(f"  - Inspect the source: `{item['url']}`")
        if gates:
            lines.append(f"  - Route to gate: re-run `{gates[0]}` after adding/refreshing "
                         "evidence for the affected surface.")
        elif is_gap:
            lines.append("  - Missing test/doc: add a regression test or threat-model row "
                         "for the affected surface, then re-run the relevant `audit_gate.py` sub-check.")
        else:
            lines.append("  - Confirm existing evidence above exercises this finding; "
                         "if not, file a permanent regression test.")
        lines.append('')

    lines.extend(['## Needs Review (review queue)', ''])
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
        f"  needs_review    (review queue): {counts['needs_review']}",
        f"  discarded       (noise):        {counts['discarded']}",
        f"  total fetched:                  {counts['total_fetched']}",
        '',
    ]
    high_conf = [item for item in report['items'] if item['bucket'] == 'high_confidence']
    review = [item for item in report['items'] if item['bucket'] == 'needs_review']
    if high_conf:
        lines.append('High-confidence findings:')
        for item in high_conf:
            lines.append(f"- [{item['score']:+d}] {item['title']} ({item['source']})")
            lines.append(f"  action: {item['action']}")
            lines.append(f"  url: {item['url']}")
            lines.append(f"  why: {item['reason']}")
    else:
        lines.append('High-confidence findings: none')
    if review:
        lines.extend(['', 'Needs-review findings:'])
        for item in review:
            lines.append(f"- [{item['score']:+d}] {item['title']} ({item['source']})")
            lines.append(f"  action: {item['action']}")
            lines.append(f"  url: {item['url']}")
            lines.append(f"  why: {item['reason']}")
    else:
        lines.extend(['', 'Needs-review findings: none'])
    if report['source_errors']:
        lines.extend(['', 'Source errors:'])
        for error in report['source_errors']:
            lines.append(f"- {source_report_label(error)}: {error['error']}")
    return '\n'.join(lines).rstrip() + '\n'


def render_mail_subject(report: dict) -> str:
    counts = report['counts']
    hc = counts['high_confidence']
    nr = counts['needs_review']
    return f"[Research Monitor] {hc} high-confidence / {nr} needs-review secp256k1 signals"


def render_mail_body(report: dict) -> str:
    counts = report['counts']
    high_conf = [item for item in report['items'] if item['bucket'] == 'high_confidence']
    review = [item for item in report['items'] if item['bucket'] == 'needs_review']
    lines = [
        'UltrafastSecp256k1 external research monitor',
        '',
        f"Generated: {report['generated_at']}",
        f"Query: {report['query']}",
        f"Lookback: {report['lookback_days']} days",
        '',
        f"High-confidence (issue-worthy): {counts['high_confidence']}",
        f"Needs review (review queue):    {counts['needs_review']}",
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
    if review:
        lines.extend(['Needs-review findings:', ''])
        for index, item in enumerate(review, start=1):
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
    else:
        lines.append('No needs-review items were detected in this window.')
    if report['source_errors']:
        lines.append('Source errors:')
        for error in report['source_errors']:
            lines.append(f"- {source_report_label(error)}: {error['error']}")
    return '\n'.join(lines).rstrip() + '\n'


def open_github_issue(report: dict, include_review: bool = False) -> None:
    """Open a GitHub issue for high-confidence, optionally needs-review, findings."""
    counts = report['counts']
    hc = counts['high_confidence']
    nr = counts['needs_review']
    if hc == 0 and (not include_review or nr == 0):
        return
    date_str = report['generated_at'][:10]
    if include_review:
        title = f"[Research Monitor] {hc} high-confidence / {nr} needs-review secp256k1 signal(s) - {date_str}"
    else:
        title = f"[Research Monitor] {hc} high-confidence secp256k1 signal(s) - {date_str}"

    body_lines = [
        '> Auto-opened by `ci/research_monitor.py --open-issue`.',
        '> High-confidence items are immediate action signals.',
    ]
    if include_review:
        body_lines.append('> Needs-review items are included because review escalation is enabled.')
    else:
        body_lines.append('> Needs-review and discarded items are in the build artifact only.')
    body_lines.extend(['', render_markdown(report)])
    body = '\n'.join(body_lines)

    search_cmd = [
        'gh', 'issue', 'list',
        '--label', 'research-monitor',
        '--state', 'open',
        '--search', f'Research Monitor secp256k1 {date_str}',
        '--json', 'number',
        '--jq', '.[0].number // empty',
    ]
    cmd = [
        'gh', 'issue', 'create',
        '--title', title,
        '--body', body,
        '--label', RESEARCH_ISSUE_LABELS,
    ]
    try:
        existing = subprocess.run(search_cmd, capture_output=True, text=True, check=True).stdout.strip()
        if existing:
            print(f'GitHub issue #{existing} already open — skipping duplicate.')
            return
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        url = result.stdout.strip()
        print(f'GitHub issue opened: {url}')
    except FileNotFoundError:
        print('warning: `gh` CLI not found — skipping issue creation', file=sys.stderr)
    except subprocess.CalledProcessError as exc:
        if 'Label' in (exc.stderr or ''):
            cmd_no_label = [a for a in cmd if a not in ('--label', RESEARCH_ISSUE_LABELS)]
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
        f"research_signal_count={counts['high_confidence'] + counts['needs_review']}",
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
              needs-review    (review queue): {counts['needs_review']}
              discarded       (noise):        {counts['discarded']}
              total fetched:                  {counts['total_fetched']}
            """
        ).strip()
    )
    if report['source_errors']:
        print('Source errors:')
        for error in report['source_errors']:
            print(f"  - {source_report_label(error)}: {error['error']}")


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

    if args.open_issue:
        open_github_issue(report, include_review=args.open_review_issue)
    if args.fail_on_actionable and report['counts']['high_confidence'] > 0:
        return 3
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f'error: {exc}', file=sys.stderr)
        raise
