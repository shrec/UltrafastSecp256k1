#!/usr/bin/env python3
"""Auditor-mode automation: emulate private audit intake and gap reporting.

This script does not replace manual audits. It automates the same first-pass
surface mapping a private auditor would run:
  1) Extract registered exploit probes from unified_audit_runner
  2) Compare them against a curated high-risk vector baseline
  3) Emit machine-readable + human-readable gap reports
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
LIB_ROOT = SCRIPT_DIR.parent
DEFAULT_OUTPUT_DIR = LIB_ROOT / 'out' / 'auditor_mode'
RUNNER_CPP = LIB_ROOT / 'audit' / 'unified_audit_runner.cpp'


@dataclass(frozen=True)
class RequiredProbe:
    key: str
    severity: str
    rationale: str
    expected_symbol: str
    expected_file: str


# Baseline + forward-looking probes from public research and practical audit playbooks.
REQUIRED_PROBES = [
    RequiredProbe(
        key='invalid_curve_twist',
        severity='high',
        rationale='Reject off-curve/twist points across parse+ECDH paths.',
        expected_symbol='test_exploit_invalid_curve_twist_run',
        expected_file='audit/test_exploit_invalid_curve_twist.cpp',
    ),
    RequiredProbe(
        key='ecdsa_nonce_reuse',
        severity='critical',
        rationale='Prevent private-key recovery under nonce reuse.',
        expected_symbol='test_exploit_ecdsa_nonce_reuse_run',
        expected_file='audit/test_exploit_ecdsa_nonce_reuse.cpp',
    ),
    RequiredProbe(
        key='frost_ct_nonce',
        severity='high',
        rationale='Detect nonce handling leaks in threshold signing flows.',
        expected_symbol='test_exploit_frost_ct_nonce_run',
        expected_file='audit/test_exploit_frost_ct_nonce.cpp',
    ),
    RequiredProbe(
        key='musig2_parallel_sessions',
        severity='high',
        rationale='Check concurrent session transcript separation.',
        expected_symbol='test_exploit_musig2_parallel_session_cross_run',
        expected_file='audit/test_exploit_musig2_parallel_session_cross.cpp',
    ),
    RequiredProbe(
        key='ecdh_zvp_glv_static',
        severity='critical',
        rationale='ZVP/GLV static-ECDH leakage class from recent literature.',
        expected_symbol='test_exploit_ecdh_zvp_glv_static_run',
        expected_file='audit/test_exploit_ecdh_zvp_glv_static.cpp',
    ),
    RequiredProbe(
        key='frost_adaptive_corruption',
        severity='critical',
        rationale='Adaptive-corruption threshold Schnorr/FROST state leakage.',
        expected_symbol='test_exploit_frost_adaptive_corruption_run',
        expected_file='audit/test_exploit_frost_adaptive_corruption.cpp',
    ),
    RequiredProbe(
        key='ecdsa_fault_injection_deterministic',
        severity='critical',
        rationale='Fault-injection resistance for deterministic ECDSA flows.',
        expected_symbol='test_exploit_ecdsa_fault_injection_run',
        expected_file='audit/test_exploit_ecdsa_fault_injection.cpp',
    ),
    RequiredProbe(
        key='cache_sidechannel_amplification',
        severity='high',
        rationale='Microarchitectural leakage amplification regression checks.',
        expected_symbol='test_exploit_cache_sidechannel_amplification_run',
        expected_file='audit/test_exploit_cache_sidechannel_amplification.cpp',
    ),
]


# Non-blocking, forward-looking probes from recent public research.
# Missing items here do not mark the audit as blocked, but are surfaced so
# hardening work stays visible in every auditor-mode report.
RECOMMENDED_PROBES = [
    RequiredProbe(
        key='ladderleak_subbit_nonce',
        severity='high',
        rationale='Sub-bit nonce leakage resilience (LadderLeak class).',
        expected_symbol='test_exploit_ladderleak_subbit_nonce_run',
        expected_file='audit/test_exploit_ladderleak_subbit_nonce.cpp',
    ),
    RequiredProbe(
        key='minerva_noisy_hnp',
        severity='high',
        rationale='Noisy timing-to-HNP robustness (Minerva class).',
        expected_symbol='test_exploit_minerva_noisy_hnp_run',
        expected_file='audit/test_exploit_minerva_noisy_hnp.cpp',
    ),
    RequiredProbe(
        key='hertzbleed_dvfs_timing',
        severity='high',
        rationale='DVFS frequency-leakage regression surface (Hertzbleed class).',
        expected_symbol='test_exploit_hertzbleed_dvfs_timing_run',
        expected_file='audit/test_exploit_hertzbleed_dvfs_timing.cpp',
    ),
    RequiredProbe(
        key='biased_nonce_chain_scan',
        severity='medium',
        rationale='Nonce-bias drift detection inspired by chain-scale incidents.',
        expected_symbol='test_exploit_biased_nonce_chain_scan_run',
        expected_file='audit/test_exploit_biased_nonce_chain_scan.cpp',
    ),
    RequiredProbe(
        key='kr_ecdsa_buff_binding',
        severity='medium',
        rationale='KR-ECDSA address/key-binding misuse resistance (BUFF-oriented).',
        expected_symbol='test_exploit_kr_ecdsa_buff_binding_run',
        expected_file='audit/test_exploit_kr_ecdsa_buff_binding.cpp',
    ),
]


def _parse_registered_symbols(runner_text: str) -> set[str]:
    # Declaration lines: int test_exploit_..._run();
    decls = set(re.findall(r'\bint\s+(test_exploit_[a-zA-Z0-9_]+_run)\s*\(', runner_text))
    # Registration lines inside module table often contain the same symbol.
    uses = set(re.findall(r'\b(test_exploit_[a-zA-Z0-9_]+_run)\b', runner_text))
    return decls.union(uses)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding='utf-8')


# ---------------------------------------------------------------------------
# SARIF 2.1.0 export — lets external reviewers import findings into GitHub
# Code Scanning / any SARIF-aware tool.
# Spec: https://docs.oasis-open.org/sarif/sarif/v2.1.0/
# ---------------------------------------------------------------------------

_SARIF_LEVEL = {
    'critical': 'error',
    'high':     'error',
    'medium':   'warning',
    'low':      'note',
}


def build_sarif(report: dict, lib_root: Path) -> dict:
    """Convert an auditor_mode report dict into a SARIF 2.1.0 document."""
    rules: list[dict] = []
    results: list[dict] = []

    # Collect rules from all probes (required + recommended)
    all_items = report.get('coverage', []) + report.get('recommended_coverage', [])

    rule_ids_seen: set[str] = set()
    for item in all_items:
        rule_id = f"UFSECP-{item['key'].upper().replace('-', '_')}"
        if rule_id not in rule_ids_seen:
            rule_ids_seen.add(rule_id)
            rules.append({
                'id': rule_id,
                'name': item['key'].replace('-', ' ').title().replace(' ', ''),
                'shortDescription': {'text': item['rationale']},
                'fullDescription': {'text': f"Expected symbol: {item['expected_symbol']}  Expected file: {item['expected_file']}"},
                'help': {'text': item['rationale']},
                'defaultConfiguration': {
                    'level': _SARIF_LEVEL.get(item.get('severity', 'medium'), 'warning'),
                },
                'properties': {
                    'severity': item.get('severity', 'medium'),
                    'tags': ['cryptography', 'exploit-probe', 'audit'],
                },
            })

        if not item['present']:
            # Only emit SARIF result for probes that are MISSING
            rel_path = item['expected_file']
            artifact_uri = rel_path.replace('\\', '/')
            results.append({
                'ruleId': rule_id,
                'level': _SARIF_LEVEL.get(item.get('severity', 'medium'), 'warning'),
                'message': {
                    'text': (
                        f"Missing exploit-probe coverage for '{item['key']}'. "
                        f"Expected: {item['expected_symbol']} in {item['expected_file']}. "
                        f"Rationale: {item['rationale']}"
                    )
                },
                'locations': [{
                    'physicalLocation': {
                        'artifactLocation': {
                            'uri': artifact_uri,
                            'uriBaseId': '%SRCROOT%',
                        },
                        'region': {'startLine': 1},
                    },
                }],
                'properties': {
                    'present': False,
                    'expected_symbol': item['expected_symbol'],
                },
            })

    sarif_doc = {
        '$schema': 'https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json',
        'version': '2.1.0',
        'runs': [{
            'tool': {
                'driver': {
                    'name': 'UltrafastSecp256k1 Auditor Mode',
                    'version': '1.0.0',
                    'informationUri': 'https://github.com/shrec/UltrafastSecp256k1',
                    'rules': rules,
                },
            },
            'results': results,
            'properties': {
                'auditor_mode_status': report.get('status', 'unknown'),
                'generated_at': report.get('generated_at', ''),
                'registered_symbol_count': report.get('registered_symbol_count', 0),
            },
        }],
    }
    return sarif_doc


def _build_text_summary(report: dict) -> str:
    lines = [
        '============================================================',
        ' Auditor-Mode Gap Report',
        '============================================================',
        '',
        f"Status: {report['status']}",
        f"Registered exploit symbols: {report['registered_symbol_count']}",
        f"Required vectors: {report['required_count']}",
        f"Covered: {report['covered_count']}",
        f"Missing: {report['missing_count']}",
        f"Recommended vectors: {report['recommended_count']}",
        f"Recommended covered: {report['recommended_covered_count']}",
        f"Recommended missing: {report['recommended_missing_count']}",
        '',
        'Missing high-priority vectors:',
    ]
    if report['missing']:
        for item in report['missing']:
            lines.append(
                f"  - [{item['severity']}] {item['key']} :: {item['expected_symbol']} ({item['expected_file']})"
            )
    else:
        lines.append('  - none')

    lines.extend(['', 'Coverage details:'])
    for item in report['coverage']:
        tag = 'OK' if item['present'] else 'MISSING'
        lines.append(f"  - [{tag}] {item['key']} -> {item['expected_symbol']}")

    lines.extend(['', 'Recommended hardening vectors:'])
    if report['recommended_coverage']:
        for item in report['recommended_coverage']:
            tag = 'OK' if item['present'] else 'TODO'
            lines.append(f"  - [{tag}] {item['key']} -> {item['expected_symbol']}")

    lines.append('')
    return '\n'.join(lines)


def build_report(strict: bool = False) -> tuple[dict, int]:
    if not RUNNER_CPP.exists():
        payload = {
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'status': 'error',
            'error': f'missing runner source: {RUNNER_CPP}',
        }
        return payload, 1

    runner_text = RUNNER_CPP.read_text(encoding='utf-8', errors='replace')
    symbols = _parse_registered_symbols(runner_text)

    coverage = []
    missing = []
    for probe in REQUIRED_PROBES:
        present = probe.expected_symbol in symbols
        item = {
            'key': probe.key,
            'severity': probe.severity,
            'rationale': probe.rationale,
            'expected_symbol': probe.expected_symbol,
            'expected_file': probe.expected_file,
            'present': present,
        }
        coverage.append(item)
        if not present:
            missing.append(item)

    missing_critical = [item for item in missing if item['severity'] == 'critical']

    recommended_coverage = []
    recommended_missing = []
    for probe in RECOMMENDED_PROBES:
        present = probe.expected_symbol in symbols
        item = {
            'key': probe.key,
            'severity': probe.severity,
            'rationale': probe.rationale,
            'expected_symbol': probe.expected_symbol,
            'expected_file': probe.expected_file,
            'present': present,
        }
        recommended_coverage.append(item)
        if not present:
            recommended_missing.append(item)

    status = 'ready' if not missing else 'partial'
    if missing_critical:
        status = 'high-risk-gaps'

    report = {
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'status': status,
        'strict_mode': strict,
        'runner_source': str(RUNNER_CPP.relative_to(LIB_ROOT)).replace('\\', '/'),
        'registered_symbol_count': len(symbols),
        'required_count': len(REQUIRED_PROBES),
        'covered_count': len(REQUIRED_PROBES) - len(missing),
        'missing_count': len(missing),
        'missing_critical_count': len(missing_critical),
        'recommended_count': len(RECOMMENDED_PROBES),
        'recommended_covered_count': len(RECOMMENDED_PROBES) - len(recommended_missing),
        'recommended_missing_count': len(recommended_missing),
        'coverage': coverage,
        'missing': missing,
        'recommended_coverage': recommended_coverage,
        'recommended_missing': recommended_missing,
    }

    exit_code = 0
    if strict and (missing_critical or missing):
        exit_code = 1
    return report, exit_code


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate auditor-mode automation report.')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR, help='Directory for generated reports')
    parser.add_argument('--json', action='store_true', help='Print JSON report to stdout')
    parser.add_argument('--sarif', action='store_true', help='Also write SARIF 2.1.0 report for GitHub Code Scanning / external tooling')
    parser.add_argument('--strict', action='store_true', help='Fail when required probes are missing')
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else (LIB_ROOT / args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report, code = build_report(strict=args.strict)
    json_path = output_dir / 'auditor_mode_report.json'
    txt_path = output_dir / 'auditor_mode_report.txt'
    _write_json(json_path, report)
    _write_text(txt_path, _build_text_summary(report))

    if args.sarif:
        sarif_doc = build_sarif(report, LIB_ROOT)
        sarif_path = output_dir / 'auditor_mode_report.sarif'
        _write_json(sarif_path, sarif_doc)
        missing_count = report.get('missing_count', 0) + report.get('recommended_missing_count', 0)
        try:
            display = sarif_path.relative_to(LIB_ROOT)
        except ValueError:
            display = sarif_path
        print(f'SARIF written: {display}  ({missing_count} findings)')

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(_build_text_summary(report))

    return code


if __name__ == '__main__':
    sys.exit(main())