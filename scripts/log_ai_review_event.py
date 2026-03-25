#!/usr/bin/env python3
"""Append validated AI review events to docs/AI_REVIEW_EVENTS.json."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
LIB_ROOT = SCRIPT_DIR.parent
EVENTS_PATH = LIB_ROOT / 'docs' / 'AI_REVIEW_EVENTS.json'

ALLOWED_MODES = {
    'auditor', 'attacker', 'bug-bounty', 'performance-skeptic', 'documentation-skeptic'
}
ALLOWED_CLASSES = {
    'bug', 'security', 'performance', 'docs-drift', 'coverage-gap', 'graph-gap', 'false-positive'
}
ALLOWED_STATUS = {'accepted', 'rejected', 'unconfirmed'}


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')


def _load_events():
    if not EVENTS_PATH.exists():
        return {'schema_version': 1, 'updated_at': _utc_now()[:10], 'events': []}
    return json.loads(EVENTS_PATH.read_text(encoding='utf-8'))


def _validate_event(event: dict):
    required = {
        'event_id', 'reviewed_at', 'review_mode', 'finding_class', 'status',
        'target', 'summary', 'reproduced', 'repository_evidence', 'resulting_changes'
    }
    missing = sorted(required - set(event.keys()))
    if missing:
        raise ValueError(f"missing fields: {', '.join(missing)}")
    if event['review_mode'] not in ALLOWED_MODES:
        raise ValueError(f"invalid review_mode: {event['review_mode']}")
    if event['finding_class'] not in ALLOWED_CLASSES:
        raise ValueError(f"invalid finding_class: {event['finding_class']}")
    if event['status'] not in ALLOWED_STATUS:
        raise ValueError(f"invalid status: {event['status']}")
    if not isinstance(event['reproduced'], bool):
        raise ValueError('reproduced must be boolean')
    for field in ('repository_evidence', 'resulting_changes'):
        if not isinstance(event[field], list):
            raise ValueError(f'{field} must be a list')
    if event['status'] == 'accepted':
        if not event['repository_evidence']:
            raise ValueError('accepted events must include repository_evidence')
        if not event['resulting_changes']:
            raise ValueError('accepted events must include resulting_changes')


def main():
    parser = argparse.ArgumentParser(description='Append an AI review event')
    parser.add_argument('event_id')
    parser.add_argument('review_mode', choices=sorted(ALLOWED_MODES))
    parser.add_argument('finding_class', choices=sorted(ALLOWED_CLASSES))
    parser.add_argument('status', choices=sorted(ALLOWED_STATUS))
    parser.add_argument('target')
    parser.add_argument('summary')
    parser.add_argument('--reviewed-at', default=_utc_now())
    parser.add_argument('--reproduced', action='store_true')
    parser.add_argument('--evidence', action='append', default=[])
    parser.add_argument('--change', action='append', default=[])
    parser.add_argument('--notes', default='')
    args = parser.parse_args()

    payload = _load_events()
    events = payload.get('events', [])

    event = {
        'event_id': args.event_id,
        'reviewed_at': args.reviewed_at,
        'review_mode': args.review_mode,
        'finding_class': args.finding_class,
        'status': args.status,
        'target': args.target,
        'summary': args.summary,
        'reproduced': bool(args.reproduced),
        'repository_evidence': args.evidence,
        'resulting_changes': args.change,
        'notes': args.notes,
    }
    _validate_event(event)

    events = [existing for existing in events if existing.get('event_id') != event['event_id']]
    events.append(event)
    payload['schema_version'] = 1
    payload['updated_at'] = _utc_now()[:10]
    payload['events'] = sorted(events, key=lambda item: item['reviewed_at'])
    EVENTS_PATH.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'ok': True, 'event_id': args.event_id}))


if __name__ == '__main__':
    main()