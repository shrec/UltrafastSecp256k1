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
import subprocess
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
# In a full monorepo workspace SUITE_ROOT is two levels up; in a clean CI
# checkout (only the submodule is checked out) that path does not exist.
# Fall back to REPO_ROOT so the dashboard still renders with whatever data
# is available locally rather than silently showing empty sections.
_suite_candidate = REPO_ROOT.parent.parent
_SUITE_ROOT_FALLBACK = False
if _suite_candidate.is_dir():
    SUITE_ROOT = _suite_candidate
else:
    import os as _os
    _SUITE_ROOT_FALLBACK = True
    if _os.environ.get("GITHUB_ACTIONS"):
        print(
            "::warning::render_audit_dashboard: SUITE_ROOT not found — "
            "dashboard uses REPO_ROOT fallback; suite-level evidence will be missing",
            flush=True,
        )
    SUITE_ROOT = REPO_ROOT

PROFILES = ("default", "bitcoin-core-backend", "cpu-signing", "release")


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


_KPI_STALE_HOURS = 25  # treat KPI file as stale if older than this


_ADVISORY_SKIP_CODE = 77


def _try_live_autonomy_run() -> dict[str, Any] | None:
    """Run security_autonomy_check.py live; return parsed JSON or None.

    Accepts exit 0 (pass) and exit 77 (advisory skip — no GPU/infrastructure).
    Exit 77 is a clean outcome, not a failure; the JSON payload is still valid.
    """
    try:
        r = subprocess.run(
            ["python3", "ci/security_autonomy_check.py", "--json"],
            capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=60,
        )
        if r.returncode in (0, _ADVISORY_SKIP_CODE) and r.stdout.strip():
            d = json.loads(r.stdout)
            d["_source"] = "live"
            d["_advisory_skip"] = (r.returncode == _ADVISORY_SKIP_CODE)
            d["_generated_at"] = datetime.now(timezone.utc).isoformat()
            return d
    except Exception:
        pass
    return None


def _staleness_banner(kpi: dict[str, Any] | None, source: str) -> str:
    """Return a Markdown warning block if the KPI data is stale (>25h old)."""
    if kpi is None or source == "live":
        return ""
    ts_raw = kpi.get("generated_at") or kpi.get("timestamp") or kpi.get("_generated_at")
    if not ts_raw:
        return "> **⚠ WARNING:** KPI source has no timestamp — age cannot be verified.\n"
    try:
        ts = datetime.fromisoformat(str(ts_raw).replace("Z", "+00:00"))
        age_h = (datetime.now(timezone.utc) - ts).total_seconds() / 3600
        if age_h > _KPI_STALE_HOURS:
            return (
                f"> **⚠ STALE DATA** — KPI last updated {age_h:.0f}h ago "
                f"(>{_KPI_STALE_HOURS}h threshold). "
                f"Gate status may not reflect current codebase. "
                f"Run `python3 ci/security_autonomy_check.py` to refresh.\n"
            )
    except (ValueError, TypeError):
        pass
    return ""


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
    # F-20 fix: use the actual top-level keys emitted by export_assurance.py.
    # The previous list included 'overall_pass', 'release_ready', 'failed_modules',
    # 'advisory_failed', 'advisory_skipped' — none of which exist in the export.
    # The real keys are: schema_version, generated_at, provenance, graph_meta,
    # subsystem_summary, api_coverage, test_targets, security_density, etc.
    display_keys = (
        'schema_version', 'generated_at',
        'api_coverage', 'security_density', 'protocol_status',
        'subsystem_summary', 'graph_meta',
    )
    out = ['| Field | Value |', '|-------|-------|']
    for k in display_keys:
        if k in rep:
            v = rep[k]
            if isinstance(v, dict):
                v = ', '.join(f'{sk}={sv}' for sk, sv in list(v.items())[:4])
            elif isinstance(v, list):
                v = ', '.join(map(str, v)) or '(none)'
            out.append(_row(k, str(v)[:120]))
    if len(out) == 2:
        out.append('| (no recognised keys present) | — |')
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
    # Item headings look like "### H-1 — title ✅ CLOSED 2026-04-21" (closed)
    # or "### H-1 — title" (open). Count any closed/done marker on the heading.
    item_lines = [ln for ln in text.splitlines() if ln.startswith('### H-')]
    items = len(item_lines)
    done = sum(
        1 for ln in item_lines
        if '✅ CLOSED' in ln or '✓ Done' in ln or '✅ DONE' in ln
    )
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
            # security_autonomy_check.py emits {"gate": "..."}; older bundles
            # used {"name": "..."}. Accept both so the dashboard renders for
            # any producer.
            name = g.get('gate', g.get('name', '?'))
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
            # Bastion B4: state plainly whether the committed bundle reflects the
            # current commit. The committed bundle is a historical baseline; the
            # strict current-run evidence is regenerated in CI. Reviewers must be
            # able to tell at a glance whether the on-disk bundle matches HEAD.
            bundle_commit = str((bundle.get('git') or {}).get('commit', ''))
            head = ''
            try:
                res = subprocess.run(['git', 'rev-parse', 'HEAD'],
                                     cwd=str(REPO_ROOT), capture_output=True,
                                     text=True, timeout=15, check=False)
                if res.returncode == 0:
                    head = res.stdout.strip()
            except (OSError, subprocess.SubprocessError):
                head = ''
            if bundle_commit and head:
                if bundle_commit == head:
                    status = f'CURRENT (matches HEAD `{head[:12]}`)'
                else:
                    status = (f'HISTORICAL BASELINE — bundle commit `{bundle_commit[:12]}` '
                              f'≠ HEAD `{head[:12]}`; regenerate for strict current-run evidence '
                              f'(`ci/external_audit_bundle.py`)')
                lines.append(f'**Commit vs HEAD:** {status}')
            elif bundle_commit:
                lines.append(f'**Bundle commit:** `{bundle_commit[:12]}` (HEAD unavailable)')
        else:
            lines.append('**Bundle JSON:** _(parse error)_')

    return '\n'.join(lines)


def _format_ct_independence(workflow_path: Path) -> str:
    """Bastion B11: summarise the constant-time independence posture — how many
    independent CT tools are configured, the ≥N-tools fail-closed rule, the gate,
    and where the live verdicts live."""
    import re
    lines: list[str] = []
    if not workflow_path.exists():
        return '_`.github/workflows/ct-independence.yml` not present._'
    text = workflow_path.read_text(encoding='utf-8', errors='replace')
    tools = sorted(set(re.findall(r'ct-verdict-([a-z0-9_-]+)\.json', text)))
    m = re.search(r'--min-tools\s+(\d+)', text)
    min_tools = m.group(1) if m else '2'

    lines.append(f'**Independent CT tools configured:** {len(tools)}'
                 + (f' — {", ".join(tools)}' if tools else ''))
    lines.append(f'**Independence rule:** ≥ {min_tools} distinct PASS required. A single PASS with '
                 f'the other tool(s) SKIP is reported **INCONCLUSIVE (exit 2), never PASS** — '
                 f'fail-closed (negative-fixture proven in `ci/test_audit_scripts.py`).')
    lines.append('**Gate:** `ci/ct_independence_check.py` consumes per-tool verdict JSON artifacts; '
                 'the deterministic CT authority is ct-verif / valgrind-ct (blocking).')
    lines.append('**Live verdicts:** `ct-verdict-*.json` are produced by `ct-independence.yml` and '
                 'retained as CI artifacts (30-day retention); they are not committed to the tree.')

    # Bastion B14: CT evidence freshness manifest summary.
    status_path = REPO_ROOT / 'docs' / 'CT_EVIDENCE_STATUS.json'
    manifest = _load_json(status_path)
    if manifest and isinstance(manifest.get('rows'), list):
        rows = manifest['rows']
        by_sev = {'blocking': 0, 'warning': 0, 'owner_gated': 0}
        for r in rows:
            by_sev[r.get('severity', '')] = by_sev.get(r.get('severity', ''), 0) + 1
        owner = [r['id'] for r in rows if r.get('severity') == 'owner_gated']
        lines.append(f'**CT evidence surfaces (G-14):** {len(rows)} tracked — '
                     f"{by_sev.get('blocking', 0)} blocking, {by_sev.get('warning', 0)} warning, "
                     f"{by_sev.get('owner_gated', 0)} owner-gated "
                     f'(`docs/CT_EVIDENCE_STATUS.json`, gated by `ci/check_ct_evidence_status.py`).'
                     + (f' Owner-gated (not current evidence): {", ".join(owner)}.' if owner else ''))
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
    # H-4 fix: try a live security_autonomy_check.py run first so the
    # dashboard reflects the current codebase state rather than up-to-24h-old
    # nightly KPI data. Fall back to the KPI file if the live run fails, but
    # annotate stale data clearly so auditors are not misled.
    kpi = _try_live_autonomy_run()
    kpi_source = "live"
    if kpi is None:
        kpi = _load_json(REPO_ROOT / 'docs' / 'SECURITY_AUTONOMY_KPI.json')
        kpi_source = "cached"
    staleness_note = _staleness_banner(kpi, kpi_source)
    if staleness_note:
        out.append(staleness_note)
    out.append(_format_autonomy(kpi))

    out.append(_section('CAAS Pipeline Status'))
    out.append(_format_caas_pipeline(kpi))

    out.append(_section('Assurance Report'))
    # VIZ-4 fix: when SUITE_ROOT fell back to REPO_ROOT, add a visible
    # warning so auditors know suite-level evidence is unavailable.
    if _SUITE_ROOT_FALLBACK:
        out.append(
            '> **⚠ Suite evidence unavailable** — showing repo root fallback. '
            'Suite-level evidence files from the parent workspace are missing.\n'
        )
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

    out.append(_section('Constant-Time Independence'))
    out.append(_format_ct_independence(REPO_ROOT / '.github' / 'workflows' / 'ct-independence.yml'))

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
