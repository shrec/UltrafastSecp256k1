#!/usr/bin/env python3
"""
render_audit_dashboard.py — emit `docs/AUDIT_DASHBOARD.md`.

Single-page Markdown summary of the assurance posture pulled from the
JSON evidence already produced by other tools. Designed to be regenerated
by the nightly `caas-evidence-refresh.yml` workflow (CAAS hardening H-1)
and committed alongside the refreshed evidence.

Implements CAAS hardening item H-9 (see `docs/CAAS_HARDENING_TODO.md`).

Usage:
    python3 ci/render_audit_dashboard.py [-o docs/AUDIT_DASHBOARD.md]
    python3 ci/render_audit_dashboard.py --profile bitcoin-core-backend
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
SUITE_ROOT = REPO_ROOT.parent.parent  # …/Secp256K1fast

PROFILES = ("default", "bitcoin-core-backend", "cpu-signing", "release")


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _row(label: str, value: Any) -> str:
    return f'| {label} | {value} |'


def _section(title: str) -> str:
    return f'\n## {title}\n'


def _format_autonomy(kpi: dict[str, Any] | None) -> str:
    if not kpi:
        return '_No SECURITY_AUTONOMY_KPI.json available._'
    score = kpi.get('autonomy_score', kpi.get('total_score', kpi.get('score', '?')))
    out = [f'**Score:** {score} / 100', '']
    out.append('| Component | Score |')
    out.append('|-----------|-------|')
    for k, v in (kpi.get('breakdown') or kpi.get('components') or {}).items():
        out.append(_row(k, v))
    return '\n'.join(out)


def _format_assurance(rep: dict[str, Any] | None) -> str:
    if not rep:
        return '_No out/reports/assurance_report.json available._'
    out = ['| Field | Value |', '|-------|-------|']
    for k in ('generated_at', 'overall_pass', 'release_ready',
              'failed_modules', 'advisory_failed', 'advisory_skipped'):
        if k in rep:
            v = rep[k]
            if isinstance(v, list):
                v = ', '.join(map(str, v)) or '(none)'
            out.append(_row(k, v))
    return '\n'.join(out)


def _format_residual(reg_path: Path) -> str:
    try:
        text = reg_path.read_text()
    except OSError:
        return '_No RESIDUAL_RISK_REGISTER.md available._'
    lines = text.splitlines()
    table = [ln for ln in lines if ln.startswith('| RR-')]
    if not table:
        return '_Register table empty._'
    return '\n'.join(['| ID | Status (one-line) |', '|----|-------------------|']
                     + [f"| {ln.split('|')[1].strip()} | "
                        f"{ln.split('|')[3].strip()} |"
                        for ln in table])


def _format_caas_hardening(todo_path: Path) -> str:
    try:
        text = todo_path.read_text()
    except OSError:
        return '_No CAAS_HARDENING_TODO.md available._'
    done = text.count('✓ Done')
    items = sum(1 for ln in text.splitlines() if ln.startswith('### H-'))
    pct = (100 * done // items) if items else 0
    return (f'**Hardening progress:** {done} / {items} items closed ({pct} %).\n'
            f'See [`docs/CAAS_HARDENING_TODO.md`](CAAS_HARDENING_TODO.md).')


def _format_profile(profile: str) -> str:
    if profile == "bitcoin-core-backend":
        scope = "Bitcoin Core secondary secp256k1 backend (compile-time opt-in)"
        out_of_scope = "ECDH, signing key material not surfaced to Core, GPU paths"
        return '\n'.join([
            f'**Profile:** `{profile}`',
            '',
            f'**Scope:** {scope}',
            '',
            f'**Out of scope:** {out_of_scope}',
        ])
    if profile == "cpu-signing":
        scope = "CPU CT signing paths (ECDSA, Schnorr, recovery)"
        out_of_scope = "GPU backends, batch GPU, OpenCL/CUDA paths"
        return '\n'.join([
            f'**Profile:** `{profile}`',
            '',
            f'**Scope:** {scope}',
            '',
            f'**Out of scope:** {out_of_scope}',
        ])
    if profile == "release":
        scope = "Full library — all backends, all signing paths, ABI surface"
        out_of_scope = "Experimental / preview features behind compile-time guards"
        return '\n'.join([
            f'**Profile:** `{profile}`',
            '',
            f'**Scope:** {scope}',
            '',
            f'**Out of scope:** {out_of_scope}',
        ])
    # generic fallback for any unknown profile
    return '\n'.join([
        f'**Profile:** `{profile}`',
        '',
        '_No specific scope note for this profile._',
    ])


def _format_caas_pipeline(kpi: dict[str, Any] | None) -> str:
    if kpi is None:
        return (
            '_`docs/SECURITY_AUTONOMY_KPI.json` not found._\n\n'
            'Run `python3 ci/caas_runner.py` to generate.'
        )
    score = kpi.get('autonomy_score', kpi.get('total_score', kpi.get('score', '?')))
    gates_pass = kpi.get('gates_pass', kpi.get('overall_pass', None))
    gates_str = 'PASS' if gates_pass else ('FAIL' if gates_pass is False else '?')
    out = [
        f'**Overall score:** {score} / 100',
        f'**Gates:** {gates_str}',
    ]
    # Show individual gate statuses if present
    gates = kpi.get('gates') or kpi.get('gate_results') or []
    if gates:
        out.append('')
        out.append('| Gate | Pass |')
        out.append('|------|------|')
        for g in gates:
            name = g.get('name', '?')
            passing = g.get('passing', g.get('pass', g.get('overall_pass', '?')))
            out.append(_row(name, 'yes' if passing is True else ('no' if passing is False else str(passing))))
    return '\n'.join(out)


def _format_evidence_bundle(
    bundle_path: Path,
    digest_path: Path,
) -> str:
    bundle_exists = bundle_path.exists()
    digest_exists = digest_path.exists()

    lines: list[str] = []
    lines.append(f'**Bundle file:** `{bundle_path.name}` — '
                 + ('exists' if bundle_exists else 'MISSING'))
    lines.append(f'**Digest file:** `{digest_path.name}` — '
                 + ('exists' if digest_exists else 'MISSING'))

    if digest_exists:
        try:
            raw = digest_path.read_text(encoding='utf-8').strip()
            sha_token = raw.split()[0] if raw else ''
            display = (sha_token[:16] + '...') if len(sha_token) >= 16 else sha_token
            lines.append(f'**SHA-256 (first 16):** `{display}`')
        except OSError:
            lines.append('**SHA-256:** _(unreadable)_')

    if bundle_exists:
        bundle = _load_json(bundle_path)
        if bundle:
            gen_at = bundle.get('generated_at', bundle.get('timestamp', ''))
            if gen_at:
                lines.append(f'**Generated at:** {gen_at}')
        else:
            lines.append('**Bundle JSON:** _(parse error)_')

    return '\n'.join(lines)


def _format_open_risks(reg_path: Path) -> str:
    try:
        text = reg_path.read_text()
    except OSError:
        return '_No RESIDUAL_RISK_REGISTER.md available._'
    lines = text.splitlines()
    table_rows = [ln for ln in lines if ln.startswith('| RR-')]
    open_count = 0
    closed_count = 0
    accepted_count = 0
    for ln in table_rows:
        parts = ln.split('|')
        if len(parts) < 5:
            continue
        status = parts[3].strip().upper()
        if 'OPEN' in status:
            open_count += 1
        elif 'CLOSED' in status or 'FIXED' in status or 'RESOLVED' in status:
            closed_count += 1
        elif 'ACCEPTED' in status or 'MITIGATED' in status or 'WONTFIX' in status:
            accepted_count += 1
        else:
            open_count += 1  # treat unknown as open (conservative)
    total = open_count + closed_count + accepted_count
    if total == 0:
        return '_Register table empty._'
    return (
        f'**{open_count} open, {closed_count} closed, {accepted_count} accepted** '
        f'({total} total)\n\n'
        f'See [`docs/RESIDUAL_RISK_REGISTER.md`](RESIDUAL_RISK_REGISTER.md).'
    )


def _format_btc_core_section() -> str:
    return '\n'.join([
        '**BTC Core scope:**',
        '',
        '- Field arithmetic (`secp256k1_field_*`)',
        '- Group operations (`secp256k1_ge_*`, `secp256k1_gej_*`)',
        '- Scalar operations (`secp256k1_scalar_*`)',
        '- ECDSA verify (non-signing, public-key path only)',
        '- CT signing if compile guard `UFSECP_ENABLE_SIGNING` is set',
        '',
        '**Key gates that must pass for Core integration:**',
        '',
        '| Gate | Description |',
        '|------|-------------|',
        '| `audit_gate` | Overall audit gate — zero failures |',
        '| `audit_gap_report_strict` | No uncovered public symbols |',
        '| `security_autonomy_check` | Autonomy KPI ≥ threshold |',
        '| `supply_chain_gate` | No unexpected dependencies |',
    ])


def render(profile: str = "default") -> str:
    out: list[str] = []
    out.append('# Audit Dashboard')
    out.append('')
    out.append(f'_Generated {datetime.now(timezone.utc).isoformat()}._')
    out.append('')
    out.append('> Auto-generated by `ci/render_audit_dashboard.py`. ')
    out.append('> Refreshed nightly by the `caas-evidence-refresh` workflow.')
    out.append('')

    # Profile section — only when non-default
    if profile != "default":
        out.append(_section('Profile'))
        out.append(_format_profile(profile))
        if profile == "bitcoin-core-backend":
            out.append('')
            out.append(_format_btc_core_section())

    out.append(_section('Security Autonomy KPI'))
    kpi = _load_json(REPO_ROOT / 'docs' / 'SECURITY_AUTONOMY_KPI.json')
    out.append(_format_autonomy(kpi))

    out.append(_section('CAAS Pipeline Status'))
    out.append(_format_caas_pipeline(kpi))

    out.append(_section('Assurance Report'))
    # Search the locations where assurance_report.json may live, in priority order:
    #   1. parent suite workspace (SUITE_ROOT/out/reports/) — local development
    #   2. submodule out/reports/                          — produced by preflight.yml
    #   3. submodule root                                  — produced by export_assurance.py
    #      via caas-evidence-refresh.yml (default output path)
    # Ordering reflects "most-curated → least-curated"; fall back if absent.
    rep = (
        _load_json(SUITE_ROOT / 'out/reports/assurance_report.json')
        or _load_json(REPO_ROOT / 'out/reports/assurance_report.json')
        or _load_json(REPO_ROOT / 'assurance_report.json')
    )
    out.append(_format_assurance(rep))

    out.append(_section('Evidence Bundle'))
    bundle_path = REPO_ROOT / 'docs' / 'EXTERNAL_AUDIT_BUNDLE.json'
    digest_path = REPO_ROOT / 'docs' / 'EXTERNAL_AUDIT_BUNDLE.sha256'
    out.append(_format_evidence_bundle(bundle_path, digest_path))

    out.append(_section('Residual Risk Register (snapshot)'))
    out.append(_format_residual(REPO_ROOT / 'docs' / 'RESIDUAL_RISK_REGISTER.md'))

    out.append(_section('Open Residual Risks'))
    out.append(_format_open_risks(REPO_ROOT / 'docs' / 'RESIDUAL_RISK_REGISTER.md'))

    out.append(_section('CAAS Hardening Progress'))
    out.append(_format_caas_hardening(REPO_ROOT / 'docs' / 'CAAS_HARDENING_TODO.md'))

    out.append(_section('Replay Command'))
    out.append(
        f'```\npython3 ci/caas_runner.py --profile {profile} --auditor-mode\n'
        f'python3 ci/audit_viewer.py\n```'
    )

    out.append('')
    return '\n'.join(out) + '\n'


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('-o', '--output',
                   default=str(REPO_ROOT / 'docs' / 'AUDIT_DASHBOARD.md'),
                   help='output Markdown path')
    p.add_argument('--profile',
                   choices=PROFILES,
                   default='default',
                   help='audit profile scope (default: default)')
    args = p.parse_args()
    Path(args.output).write_text(render(profile=args.profile))
    print(f'render_audit_dashboard: wrote {args.output}', file=sys.stderr)
    return 0


if __name__ == '__main__':
    sys.exit(main())
