#!/usr/bin/env python3
"""Build a compact owner-grade audit bundle from existing assurance surfaces."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
LIB_ROOT = SCRIPT_DIR.parent
DEFAULT_OUTPUT_DIR = LIB_ROOT / 'build' / 'owner_audit'


def _read_text(path: Path) -> str:
    return path.read_text(encoding='utf-8', errors='replace')


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding='utf-8')


def _run_json_command(command: list[str]) -> dict:
    result = subprocess.run(
        command,
        cwd=str(LIB_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    payload = None
    error = None
    stdout = (result.stdout or '').strip()
    stderr = (result.stderr or '').strip()
    if stdout:
        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError as exc:
            error = f'failed to parse JSON output: {exc}'
    elif stderr:
        error = stderr
    return {
        'command': command,
        'exit_code': result.returncode,
        'payload': payload,
        'stdout': stdout,
        'stderr': stderr,
        'error': error,
    }


def _discover_latest_audit_summary() -> dict:
    candidates = []
    # Search canonical location first (out/audit-output/<compiler>/),
    # then fall back to legacy root-level dirs for backward compat.
    search_roots = [LIB_ROOT / 'out' / 'audit-output', LIB_ROOT]
    search_patterns = ('*', 'audit-output-*', 'audit-evidence-*')
    for root in search_roots:
        if not root.is_dir():
            continue
        pattern = '*' if root.name == 'audit-output' else None
        for pat in (search_patterns if pattern is None else (pattern,)):
            for directory in root.glob(pat):
                if not directory.is_dir():
                    continue
                report_json = directory / 'audit_report.json'
                report_txt = directory / 'audit_report.txt'
                if report_json.exists():
                    candidates.append((report_json.stat().st_mtime, directory, report_json, report_txt if report_txt.exists() else None))
    if not candidates:
        return {
            'status': 'missing',
            'issues': ['no audit_report.json found under out/audit-output/ or audit-output-*/audit-evidence-*'],
        }

    _, directory, report_json, report_txt = max(candidates, key=lambda item: item[0])
    payload = None
    issues = []
    try:
        payload = json.loads(_read_text(report_json))
    except Exception as exc:
        issues.append(f'could not parse {report_json.name}: {exc}')

    section_count = None
    if isinstance(payload, dict):
        if isinstance(payload.get('sections'), list):
            section_count = len(payload['sections'])
        elif isinstance(payload.get('results'), list):
            section_count = len(payload['results'])

    return {
        'status': 'present' if not issues else 'invalid',
        'directory': str(directory.relative_to(LIB_ROOT)).replace('\\', '/'),
        'audit_report_json': str(report_json.relative_to(LIB_ROOT)).replace('\\', '/'),
        'audit_report_txt': str(report_txt.relative_to(LIB_ROOT)).replace('\\', '/') if report_txt else None,
        'section_count': section_count,
        'issues': issues,
    }


def _discover_runner_binary() -> str | None:
    patterns = [
        'build/audit/unified_audit_runner',
        'build/*/audit/unified_audit_runner',
        'build*/audit/unified_audit_runner',
    ]
    candidates = []
    for pattern in patterns:
        for path in LIB_ROOT.glob(pattern):
            if path.is_file() and path.exists():
                candidates.append(path)
    if not candidates:
        return None
    newest = max(candidates, key=lambda item: item.stat().st_mtime)
    return str(newest.relative_to(LIB_ROOT)).replace('\\', '/')


def _parse_workflow_threshold(text: str) -> int | None:
    match = re.search(r"alert-threshold:\s*'?(\d+)%'?", text)
    return int(match.group(1)) if match else None


def _parse_workflow_fail_on_alert(text: str) -> bool | None:
    match = re.search(r'fail-on-alert:\s*(true|false)', text)
    if not match:
        return None
    return match.group(1) == 'true'


def _parse_policy_threshold(text: str) -> int | None:
    match = re.search(r'Threshold\*\*: (\d+)%', text)
    if match:
        return int(match.group(1))
    match = re.search(r'Threshold:\s*(\d+)%', text)
    return int(match.group(1)) if match else None


def _build_benchmark_publishability() -> dict:
    policy_path = LIB_ROOT / 'docs' / 'BENCHMARK_POLICY.md'
    dashboard_path = LIB_ROOT / '.github' / 'workflows' / 'benchmark.yml'
    gate_path = LIB_ROOT / '.github' / 'workflows' / 'bench-regression.yml'
    claims_path = LIB_ROOT / 'docs' / 'ASSURANCE_CLAIMS.json'
    gpu_path = LIB_ROOT / 'docs' / 'GPU_BACKEND_EVIDENCE.json'

    missing = [
        str(path.relative_to(LIB_ROOT)).replace('\\', '/')
        for path in (policy_path, dashboard_path, gate_path, claims_path, gpu_path)
        if not path.exists()
    ]
    if missing:
        return {
            'status': 'missing',
            'missing': missing,
            'issues': [f'missing benchmark publishability surface: {item}' for item in missing],
        }

    policy_text = _read_text(policy_path)
    dashboard_text = _read_text(dashboard_path)
    gate_text = _read_text(gate_path)
    claims = json.loads(_read_text(claims_path))
    gpu = json.loads(_read_text(gpu_path))

    policy_threshold = _parse_policy_threshold(policy_text)
    dashboard_threshold = _parse_workflow_threshold(dashboard_text)
    gate_threshold = _parse_workflow_threshold(gate_text)
    dashboard_fail = _parse_workflow_fail_on_alert(dashboard_text)
    gate_fail = _parse_workflow_fail_on_alert(gate_text)

    issues = []
    if policy_threshold is not None and gate_threshold is not None and policy_threshold != gate_threshold:
        issues.append(
            f'policy threshold {policy_threshold}% does not match bench-regression.yml threshold {gate_threshold}%'
        )
    if dashboard_fail is not False:
        issues.append('benchmark.yml should remain non-blocking (fail-on-alert: false) for dashboard publishing')
    if gate_fail is not True:
        issues.append('bench-regression.yml should remain blocking (fail-on-alert: true) for regression gating')

    claim = None
    for item in claims.get('claims', []):
        if item.get('claim_id') == 'A-004':
            claim = item
            break

    publishable_backends = [item['backend'] for item in gpu.get('backends', []) if item.get('publishable')]
    blocked_backends = [item['backend'] for item in gpu.get('backends', []) if not item.get('publishable')]

    return {
        'status': 'drift' if issues else 'governed',
        'policy_threshold_percent': policy_threshold,
        'dashboard_threshold_percent': dashboard_threshold,
        'dashboard_fail_on_alert': dashboard_fail,
        'gate_threshold_percent': gate_threshold,
        'gate_fail_on_alert': gate_fail,
        'claim_a004_present': claim is not None,
        'publishable_backends': publishable_backends,
        'blocked_backends': blocked_backends,
        'issues': issues,
    }


def _count_failures(audit_payload: dict | None) -> int:
    if not isinstance(audit_payload, dict):
        return 0
    total = 0
    for check in audit_payload.get('checks', []):
        for finding in check.get('findings', []):
            if finding.get('severity') == 'FAIL':
                total += 1
    return total


def _build_summary(report: dict) -> str:
    lines = [
        '============================================================',
        '  Owner-Grade Audit Bundle',
        '============================================================',
        '',
        f"Generated at: {report['generated_at']}",
        f"Overall status: {report['overall_status']}",
        '',
        'Blocking findings:',
    ]
    if report['blocking_findings']:
        lines.extend([f"  - {item}" for item in report['blocking_findings']])
    else:
        lines.append('  - none')

    lines.extend(['', 'Mandatory missing artifacts:'])
    if report['mandatory_missing']:
        lines.extend([f"  - {item}" for item in report['mandatory_missing']])
    else:
        lines.append('  - none')

    lines.extend(['', 'Residual gaps:'])
    if report['residual_gaps']:
        lines.extend([f"  - {item}" for item in report['residual_gaps']])
    else:
        lines.append('  - none')

    lines.extend([
        '',
        f"Audit gate status: {report['audit_gate'].get('status', 'unknown')}",
        f"Failure matrix strict residuals: {report['failure_matrix_strict'].get('counts', {}).get('partial', 0)} partial, {report['failure_matrix_strict'].get('counts', {}).get('unknown', 0)} unknown",
        f"CT evidence status: {report['ct_evidence'].get('overall_status', 'unknown')}",
        f"Auditor-mode status: {report['auditor_mode'].get('status', 'unknown')}",
        f"Auditor-mode missing critical: {report['auditor_mode'].get('missing_critical_count', 'unknown')}",
        f"Benchmark publishability: {report['benchmark_publishability'].get('status', 'unknown')}",
        f"Latest audit summary: {report['audit_summary'].get('directory', 'missing')}",
        '',
    ])
    return '\n'.join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build a compact owner-grade audit bundle from existing assurance surfaces.')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR, help='Destination directory for the bundle summary files')
    parser.add_argument('--strict', action='store_true', help='Return non-zero when blocking findings remain, not only when mandatory artifacts are missing')
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else (LIB_ROOT / args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    audit_gate = _run_json_command(['python3', 'ci/audit_gate.py', '--json'])
    validate_assurance = _run_json_command(['python3', 'ci/validate_assurance.py', '--json'])
    failure_matrix = _run_json_command(['python3', 'ci/audit_gap_report.py', '--json'])
    failure_matrix_strict = _run_json_command(['python3', 'ci/audit_gap_report.py', '--json', '--strict'])
    auditor_mode = _run_json_command(['python3', 'ci/auditor_mode.py', '--json'])
    runner_binary = _discover_runner_binary()

    ct_dir = output_dir / 'ct_evidence'
    ct_command = ['python3', 'ci/collect_ct_evidence.py', '--output-dir', str(ct_dir)]
    if runner_binary:
        ct_command.extend(['--runner-binary', runner_binary])
    ct_collection = subprocess.run(
        ct_command,
        cwd=str(LIB_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    ct_summary_path = ct_dir / 'ct_evidence_summary.json'
    ct_evidence = json.loads(_read_text(ct_summary_path)) if ct_summary_path.exists() else {
        'overall_status': 'missing',
        'owner_grade_gaps': ['ct evidence summary missing'],
    }

    benchmark_publishability = _build_benchmark_publishability()
    audit_summary = _discover_latest_audit_summary()

    mandatory_missing = []
    for label, payload in (
        ('audit_gate_json', audit_gate),
        ('validate_assurance_json', validate_assurance),
        ('failure_matrix_json', failure_matrix),
        ('failure_matrix_strict_json', failure_matrix_strict),
        ('auditor_mode_json', auditor_mode),
    ):
        if payload.get('payload') is None:
            mandatory_missing.append(label)
    if not ct_summary_path.exists():
        mandatory_missing.append('ct_evidence_summary.json')
    if benchmark_publishability.get('status') == 'missing':
        mandatory_missing.extend(benchmark_publishability.get('missing', []))
    if audit_summary.get('status') != 'present':
        mandatory_missing.append('audit_report.json')

    blocking_findings = []
    if isinstance(audit_gate.get('payload'), dict) and audit_gate['payload'].get('status') == 'FAIL':
        blocking_findings.append(f"audit gate has {_count_failures(audit_gate['payload'])} FAIL findings")
    strict_payload = failure_matrix_strict.get('payload') or {}
    counts = strict_payload.get('counts', {}) if isinstance(strict_payload, dict) else {}
    partial = counts.get('partial', 0)
    unknown = counts.get('unknown', 0)
    if partial or unknown:
        blocking_findings.append(f'failure matrix strict residuals remain: partial={partial}, unknown={unknown}')
    auditor_payload = auditor_mode.get('payload') or {}
    if isinstance(auditor_payload, dict):
        missing_critical = auditor_payload.get('missing_critical_count', 0)
        if missing_critical:
            blocking_findings.append(f'auditor-mode missing critical vectors: {missing_critical}')
    owner_grade_gaps = ct_evidence.get('owner_grade_gaps', []) if isinstance(ct_evidence, dict) else []
    for gap in owner_grade_gaps:
        blocking_findings.append(f'ct evidence owner-grade gap: {gap}')

    residual_gaps = []
    audit_payload = audit_gate.get('payload') or {}
    if isinstance(audit_payload, dict):
        for check in audit_payload.get('checks', []):
            if check.get('name') == 'P0: ABI Hostile-Caller Manifest':
                for finding in check.get('findings', []):
                    if finding.get('severity') == 'FAIL':
                        residual_gaps.append(finding['message'])
            if check.get('name') == 'P4: CT Integrity':
                for finding in check.get('findings', []):
                    if finding.get('severity') == 'WARN':
                        residual_gaps.append(finding['message'])
    residual_gaps.extend(benchmark_publishability.get('issues', []))
    residual_gaps.extend(audit_summary.get('issues', []))
    if isinstance(auditor_payload, dict):
        for item in auditor_payload.get('missing', []):
            residual_gaps.append(f"auditor-mode missing [{item.get('severity', 'unknown')}]: {item.get('key', 'unknown')}")

    overall_status = 'ready'
    if mandatory_missing:
        overall_status = 'missing-artifacts'
    elif blocking_findings:
        overall_status = 'partial'
    elif benchmark_publishability.get('issues'):
        overall_status = 'ready-with-drift'

    report = {
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'output_dir': str(output_dir.relative_to(LIB_ROOT)).replace('\\', '/'),
        'overall_status': overall_status,
        'mandatory_missing': sorted(dict.fromkeys(mandatory_missing)),
        'blocking_findings': sorted(dict.fromkeys(blocking_findings)),
        'residual_gaps': sorted(dict.fromkeys(residual_gaps)),
        'audit_gate': audit_gate.get('payload') or {'error': audit_gate.get('error'), 'exit_code': audit_gate.get('exit_code')},
        'validate_assurance': validate_assurance.get('payload') or {'error': validate_assurance.get('error'), 'exit_code': validate_assurance.get('exit_code')},
        'failure_matrix': failure_matrix.get('payload') or {'error': failure_matrix.get('error'), 'exit_code': failure_matrix.get('exit_code')},
        'failure_matrix_strict': failure_matrix_strict.get('payload') or {'error': failure_matrix_strict.get('error'), 'exit_code': failure_matrix_strict.get('exit_code')},
        'auditor_mode': auditor_mode.get('payload') or {'error': auditor_mode.get('error'), 'exit_code': auditor_mode.get('exit_code')},
        'ct_evidence': ct_evidence,
        'benchmark_publishability': benchmark_publishability,
        'audit_summary': audit_summary,
        'inputs': {
            'runner_binary': runner_binary,
            'ct_collection_exit_code': ct_collection.returncode,
            'ct_collection_stdout': (ct_collection.stdout or '').strip(),
            'ct_collection_stderr': (ct_collection.stderr or '').strip(),
        },
    }

    json_path = output_dir / 'owner_audit_bundle.json'
    text_path = output_dir / 'owner_audit_bundle.txt'
    _write_json(json_path, report)
    _write_text(text_path, _build_summary(report))

    print(_build_summary(report))
    if mandatory_missing:
        return 1
    if args.strict and blocking_findings:
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())