#!/usr/bin/env python3
"""
report_schema.py  --  Unified report schema for UltrafastSecp256k1

Defines the canonical report envelope that ALL report producers must use.
Enforces:
  - Provenance (never "unknown")
  - null + status + reason for unavailable data (not "unknown" strings)
  - SKIP(reason) standard for platform/feature unavailability
  - Advisory vs Blocking severity classification
  - Uniform field names across all runners

Usage as module:
    from report_schema import ReportBuilder, Finding, SkipReason, Severity

Usage standalone (schema dump):
    python3 ci/report_schema.py                 # print JSON schema
    python3 ci/report_schema.py --validate f.json  # validate a report
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Severity: Advisory vs Blocking
# ---------------------------------------------------------------------------

class Severity(str, Enum):
    """Finding severity with explicit advisory/blocking semantics."""
    BLOCKING_CRITICAL = 'blocking:critical'   # Must fix before release
    BLOCKING_HIGH     = 'blocking:high'       # Must fix before release
    ADVISORY_MEDIUM   = 'advisory:medium'     # Should fix, not a release gate
    ADVISORY_LOW      = 'advisory:low'        # Informational, improve later
    INFO              = 'info'                # Pure informational note
    PASS              = 'pass'                # Check passed

    @property
    def is_blocking(self) -> bool:
        return self.value.startswith('blocking:')

    @property
    def is_advisory(self) -> bool:
        return self.value.startswith('advisory:')

    @property
    def display(self) -> str:
        """Human-readable display: 'PASS', 'PASS with advisory', 'FAIL'."""
        if self == Severity.PASS:
            return 'PASS'
        if self == Severity.INFO:
            return 'INFO'
        if self.is_advisory:
            return 'PASS with advisory'
        return 'FAIL'


# ---------------------------------------------------------------------------
# SKIP reason
# ---------------------------------------------------------------------------

@dataclass
class SkipReason:
    """Structured skip record for tests/checks that cannot run."""
    platform: Optional[str] = None     # e.g. "MSVC", "ESP32", "RISC-V"
    constraint: Optional[str] = None   # e.g. "no __int128", "no CUDA"
    feature_flag: Optional[str] = None # e.g. "UFSECP_ENABLE_ETHEREUM=OFF"
    detail: Optional[str] = None       # free-form explanation

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}

    def __str__(self) -> str:
        parts = []
        if self.platform:
            parts.append(self.platform)
        if self.constraint:
            parts.append(self.constraint)
        if self.feature_flag:
            parts.append(self.feature_flag)
        if self.detail:
            parts.append(self.detail)
        return f"SKIP({'; '.join(parts)})"


# ---------------------------------------------------------------------------
# NullableField: replaces "unknown" strings
# ---------------------------------------------------------------------------

@dataclass
class NullableField:
    """Wraps a value that may be unavailable. Never use 'unknown' as a string.

    Policy: value=null means the data is not available. Callers must check
    status and reason to understand why.
    """
    value: Any = None
    status: str = 'available'  # 'available' | 'unavailable' | 'error' | 'skipped'
    reason: Optional[str] = None

    @classmethod
    def available(cls, value: Any) -> NullableField:
        return cls(value=value, status='available')

    @classmethod
    def unavailable(cls, reason: str) -> NullableField:
        return cls(value=None, status='unavailable', reason=reason)

    @classmethod
    def error(cls, reason: str) -> NullableField:
        return cls(value=None, status='error', reason=reason)

    @classmethod
    def skipped(cls, reason: str) -> NullableField:
        return cls(value=None, status='skipped', reason=reason)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {'value': self.value, 'status': self.status}
        if self.reason is not None:
            d['reason'] = self.reason
        return d


# ---------------------------------------------------------------------------
# Finding
# ---------------------------------------------------------------------------

@dataclass
class Finding:
    """A single check result / finding."""
    check_id: str                       # e.g. "P1_abi_completeness"
    severity: Severity
    title: str
    detail: Optional[str] = None
    file: Optional[str] = None
    line: Optional[int] = None
    skip: Optional[SkipReason] = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            'check_id': self.check_id,
            'severity': self.severity.value,
            'severity_display': self.severity.display,
            'title': self.title,
        }
        if self.detail:
            d['detail'] = self.detail
        if self.file:
            d['file'] = self.file
        if self.line is not None:
            d['line'] = self.line
        if self.skip:
            d['skip'] = self.skip.to_dict()
        return d


# ---------------------------------------------------------------------------
# Section
# ---------------------------------------------------------------------------

@dataclass
class Section:
    """A logical group of findings (e.g. 'CT Verification', 'ABI Completeness')."""
    name: str
    findings: list[Finding] = field(default_factory=list)
    skipped: Optional[SkipReason] = None

    @property
    def verdict(self) -> str:
        if self.skipped:
            return f'SKIP({self.skipped})'
        blocking = [f for f in self.findings if f.severity.is_blocking]
        advisory = [f for f in self.findings if f.severity.is_advisory]
        if blocking:
            return 'FAIL'
        if advisory:
            return 'PASS with advisory'
        return 'PASS'

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            'name': self.name,
            'verdict': self.verdict,
            'findings': [f.to_dict() for f in self.findings],
        }
        if self.skipped:
            d['skipped'] = self.skipped.to_dict()
        return d


# ---------------------------------------------------------------------------
# ReportBuilder
# ---------------------------------------------------------------------------

class ReportBuilder:
    """Builds a unified report with provenance, sections, and verdicts.

    Usage:
        from report_provenance import collect_provenance
        rb = ReportBuilder(
            runner='audit_gate',
            provenance=collect_provenance(),
        )
        sec = rb.add_section('ABI Completeness')
        sec.findings.append(Finding(
            check_id='P1_abi',
            severity=Severity.PASS,
            title='All 181 ABI functions accounted for',
        ))
        report = rb.build()
    """

    def __init__(
        self,
        runner: str,
        provenance: Optional[dict] = None,
        platform: Optional[str] = None,
    ):
        self.runner = runner
        self.provenance = provenance
        self.platform = platform
        self.sections: list[Section] = []
        self.metadata: dict[str, Any] = {}

    def add_section(self, name: str, skip: Optional[SkipReason] = None) -> Section:
        sec = Section(name=name, skipped=skip)
        self.sections.append(sec)
        return sec

    def set_metadata(self, key: str, value: Any) -> None:
        self.metadata[key] = value

    @property
    def verdict(self) -> str:
        """Overall report verdict."""
        verdicts = [s.verdict for s in self.sections]
        if any(v == 'FAIL' for v in verdicts):
            return 'FAIL'
        if any('advisory' in v.lower() for v in verdicts):
            return 'PASS with advisory'
        if any(v.startswith('SKIP') for v in verdicts):
            return 'PASS with skips'
        return 'PASS'

    @property
    def all_findings(self) -> list[Finding]:
        result = []
        for sec in self.sections:
            result.extend(sec.findings)
        return result

    @property
    def run_id(self) -> str:
        """Deterministic run_id from provenance."""
        if self.provenance and self.provenance.get('git', {}).get('sha'):
            sha = self.provenance['git']['sha'][:10]
            ts = self.provenance.get('collected_at', '')[:19].replace(':', '').replace('-', '')
            return f'{self.runner}-{sha}-{ts}'
        ts = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
        return f'{self.runner}-noprov-{ts}'

    def build(self) -> dict[str, Any]:
        """Produce the final report dict."""
        commit_info = NullableField.unavailable('git not available')
        if self.provenance:
            git = self.provenance.get('git', {})
            if git.get('sha'):
                commit_info = NullableField.available({
                    'sha': git['sha'],
                    'short': git.get('short'),
                    'dirty': git.get('dirty'),
                    'ref': git.get('ref'),
                })

        blocking_count = sum(1 for f in self.all_findings if f.severity.is_blocking)
        advisory_count = sum(1 for f in self.all_findings if f.severity.is_advisory)
        skip_count = sum(1 for s in self.sections if s.skipped)

        report: dict[str, Any] = {
            'schema_version': '1.0.0',
            'run_id': self.run_id,
            'runner': self.runner,
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'commit': commit_info.to_dict(),
            'platform': self.platform or (
                self.provenance.get('platform', {}).get('system', 'unknown')
                if self.provenance else None
            ),
            'provenance': self.provenance,
            'verdict': self.verdict,
            'summary': {
                'total_findings': len(self.all_findings),
                'blocking': blocking_count,
                'advisory': advisory_count,
                'skipped_sections': skip_count,
                'sections': len(self.sections),
            },
            'sections': [s.to_dict() for s in self.sections],
        }

        if self.metadata:
            report['metadata'] = self.metadata

        return report


# ---------------------------------------------------------------------------
# Schema definition (JSON Schema draft-07)
# ---------------------------------------------------------------------------

REPORT_SCHEMA = {
    '$schema': 'http://json-schema.org/draft-07/schema#',
    'title': 'UltrafastSecp256k1 Unified Report',
    'type': 'object',
    'required': ['schema_version', 'run_id', 'runner', 'generated_at',
                 'commit', 'verdict', 'sections'],
    'properties': {
        'schema_version': {'type': 'string', 'pattern': r'^\d+\.\d+\.\d+$'},
        'run_id': {'type': 'string'},
        'runner': {'type': 'string'},
        'generated_at': {'type': 'string', 'format': 'date-time'},
        'commit': {
            'type': 'object',
            'required': ['value', 'status'],
            'properties': {
                'value': {},
                'status': {'type': 'string', 'enum': ['available', 'unavailable', 'error', 'skipped']},
                'reason': {'type': ['string', 'null']},
            },
        },
        'platform': {'type': ['string', 'null']},
        'provenance': {'type': ['object', 'null']},
        'verdict': {
            'type': 'string',
            'enum': ['PASS', 'PASS with advisory', 'PASS with skips', 'FAIL'],
        },
        'summary': {
            'type': 'object',
            'properties': {
                'total_findings': {'type': 'integer'},
                'blocking': {'type': 'integer'},
                'advisory': {'type': 'integer'},
                'skipped_sections': {'type': 'integer'},
                'sections': {'type': 'integer'},
            },
        },
        'sections': {
            'type': 'array',
            'items': {
                'type': 'object',
                'required': ['name', 'verdict', 'findings'],
                'properties': {
                    'name': {'type': 'string'},
                    'verdict': {'type': 'string'},
                    'findings': {
                        'type': 'array',
                        'items': {
                            'type': 'object',
                            'required': ['check_id', 'severity', 'title'],
                            'properties': {
                                'check_id': {'type': 'string'},
                                'severity': {'type': 'string'},
                                'severity_display': {'type': 'string'},
                                'title': {'type': 'string'},
                                'detail': {'type': ['string', 'null']},
                                'file': {'type': ['string', 'null']},
                                'line': {'type': ['integer', 'null']},
                                'skip': {'type': ['object', 'null']},
                            },
                        },
                    },
                    'skipped': {'type': ['object', 'null']},
                },
            },
        },
        'metadata': {'type': 'object'},
    },
}


def validate_report(report: dict) -> list[str]:
    """Basic structural validation without jsonschema dependency."""
    errors = []

    for req in ['schema_version', 'run_id', 'runner', 'generated_at', 'commit', 'verdict', 'sections']:
        if req not in report:
            errors.append(f'missing required field: {req}')

    commit = report.get('commit', {})
    if isinstance(commit, dict):
        if 'status' not in commit:
            errors.append('commit: missing status field')
        if commit.get('status') == 'available' and not commit.get('value'):
            errors.append('commit: status=available but no value')
    else:
        errors.append('commit: must be an object with value+status+reason, not a string')

    # Check for "unknown" strings anywhere in the report
    def check_unknown(obj: Any, path: str = '') -> None:
        if isinstance(obj, str) and obj.lower() == 'unknown':
            errors.append(f'{path}: found "unknown" string — use null+status+reason instead')
        elif isinstance(obj, dict):
            for k, v in obj.items():
                check_unknown(v, f'{path}.{k}')
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                check_unknown(v, f'{path}[{i}]')

    check_unknown(report, 'report')

    verdict = report.get('verdict', '')
    valid_verdicts = {'PASS', 'PASS with advisory', 'PASS with skips', 'FAIL'}
    if verdict not in valid_verdicts:
        errors.append(f'verdict: {verdict!r} not in {valid_verdicts}')

    for i, sec in enumerate(report.get('sections', [])):
        if not isinstance(sec, dict):
            errors.append(f'sections[{i}]: must be an object')
            continue
        for req in ['name', 'verdict', 'findings']:
            if req not in sec:
                errors.append(f'sections[{i}]: missing {req}')

    return errors


def main():
    parser = argparse.ArgumentParser(description='Unified report schema')
    parser.add_argument('--validate', type=str, help='Validate a report JSON file')
    parser.add_argument('--schema', action='store_true', help='Print JSON schema')
    args = parser.parse_args()

    if args.validate:
        path = Path(args.validate)
        if not path.exists():
            print(f'File not found: {path}', file=sys.stderr)
            sys.exit(1)
        report = json.loads(path.read_text())
        errors = validate_report(report)
        if errors:
            for e in errors:
                print(f'  ERROR: {e}')
            sys.exit(1)
        else:
            print('Report is valid.')
    else:
        print(json.dumps(REPORT_SCHEMA, indent=2))


if __name__ == '__main__':
    main()
