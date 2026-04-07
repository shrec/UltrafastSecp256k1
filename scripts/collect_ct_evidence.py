#!/usr/bin/env python3
"""
collect_ct_evidence.py -- Discover and normalize constant-time evidence.

This tool turns today's scattered CT evidence surfaces into a canonical
artifact directory. It copies or synthesizes the standard artifact slots under
an output directory and emits one machine-readable and one text summary.

Usage:
  python3 scripts/collect_ct_evidence.py
  python3 scripts/collect_ct_evidence.py --build-dir build/copilot-ci
  python3 scripts/collect_ct_evidence.py --output-dir /tmp/ufsecp-ct
  python3 scripts/collect_ct_evidence.py --runner-binary build/audit/unified_audit_runner
  python3 scripts/collect_ct_evidence.py --strict
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
LIB_ROOT = SCRIPT_DIR.parent
DEFAULT_OUTPUT_DIR = LIB_ROOT / 'artifacts' / 'ct'
CT_VERIF_EXPECTED_MODULES = ['ct_field', 'ct_scalar', 'ct_point', 'ct_sign']


@dataclass
class LayerSpec:
    layer_id: str
    evidence_class: str
    summary_label: str
    canonical_files: list[str]
    configured_surfaces: list[str]
    binary_names: list[str] = field(default_factory=list)
    required_for_owner_grade: bool = False


@dataclass
class LayerResult:
    layer_id: str
    evidence_class: str
    summary_label: str
    status: str
    canonical_artifacts: list[str]
    source_paths: list[str]
    configured_surfaces: list[str]
    binary_paths: list[str]
    notes: list[str]
    metadata: dict
    owner_grade_gap: bool


LAYER_SPECS = [
    LayerSpec(
        layer_id='ct_verif',
        evidence_class='deterministic',
        summary_label='ct-verif',
        canonical_files=['ct_verif.log', 'ct_verif_summary.json'],
        configured_surfaces=['.github/workflows/ct-verif.yml', 'docs/CT_VERIFICATION.md', 'audit/test_ct_verif_formal.cpp'],
        binary_names=['test_ct_verif_formal_standalone'],
        required_for_owner_grade=True,
    ),
    LayerSpec(
        layer_id='valgrind_ct',
        evidence_class='deterministic',
        summary_label='valgrind-ct',
        canonical_files=['valgrind_ct.log', 'valgrind_ct_report.json'],
        configured_surfaces=['.github/workflows/valgrind-ct.yml', 'docs/CT_VERIFICATION.md', 'scripts/valgrind_ct_check.sh', 'audit/test_ct_sidechannel.cpp'],
        binary_names=['test_ct_sidechannel_standalone', 'test_ct_sidechannel_smoke'],
        required_for_owner_grade=True,
    ),
    LayerSpec(
        layer_id='disasm_scan',
        evidence_class='deterministic',
        summary_label='disassembly branch scan',
        canonical_files=['disasm_branch_scan.json'],
        configured_surfaces=['docs/CT_VERIFICATION.md', 'scripts/verify_ct_disasm.sh'],
        required_for_owner_grade=True,
    ),
    LayerSpec(
        layer_id='dudect_smoke',
        evidence_class='statistical',
        summary_label='dudect smoke',
        canonical_files=['dudect_smoke.log'],
        configured_surfaces=['docs/CT_VERIFICATION.md', 'audit/test_ct_sidechannel.cpp'],
        binary_names=['test_ct_sidechannel_smoke'],
    ),
    LayerSpec(
        layer_id='dudect_full',
        evidence_class='statistical',
        summary_label='dudect full',
        canonical_files=['dudect_full.log'],
        configured_surfaces=['docs/CT_VERIFICATION.md', 'audit/test_ct_sidechannel.cpp'],
        binary_names=['test_ct_sidechannel_standalone'],
    ),
    LayerSpec(
        layer_id='manual_ct_docs',
        evidence_class='manual-only',
        summary_label='manual CT claims',
        canonical_files=[],
        configured_surfaces=['docs/CT_VERIFICATION.md', 'docs/SECURITY_CLAIMS.md'],
    ),
]


STATUS_ORDER = {
    'artifact-present': 0,
    'manual-only': 1,
    'configured-only': 2,
    'missing': 3,
}


def _rel(path: Path) -> str:
    return str(path.resolve().relative_to(LIB_ROOT.resolve())).replace('\\', '/')


def _safe_rel(path: Path) -> str:
    try:
        return _rel(path)
    except ValueError:
        return str(path)


def _read_text(path: Path) -> str:
    return path.read_text(encoding='utf-8', errors='replace')


def _existing_paths(paths: list[Path]) -> list[Path]:
    return sorted({path.resolve() for path in paths if path.exists()})


def _candidate_dirs(repo_root: Path, requested_dirs: list[Path]) -> list[Path]:
    candidates: list[Path] = []
    for path in requested_dirs:
        resolved = path if path.is_absolute() else (repo_root / path)
        if resolved.exists() and resolved.is_dir():
            candidates.append(resolved.resolve())

    for child in sorted(repo_root.iterdir()):
        if not child.is_dir():
            continue
        name = child.name
        if name.startswith('build') or name.startswith('audit-output') or name == 'local-ci-output':
            candidates.append(child.resolve())

    seen = set()
    ordered = []
    for path in candidates:
        if path not in seen:
            seen.add(path)
            ordered.append(path)
    return ordered


def _discover_binary_paths(search_dirs: list[Path], binary_names: list[str]) -> list[Path]:
    found: list[Path] = []
    for directory in search_dirs:
        for binary_name in binary_names:
            found.extend(directory.rglob(binary_name))
            found.extend(directory.rglob(f'{binary_name}.exe'))
    return _existing_paths(found)


def _discover_configured_surfaces(repo_root: Path, rel_paths: list[str]) -> list[Path]:
    return _existing_paths([(repo_root / rel_path) for rel_path in rel_paths])


def _discover_named_files(repo_root: Path, output_dir: Path, search_dirs: list[Path], file_name: str) -> list[Path]:
    matches = [output_dir / file_name, repo_root / 'artifacts' / 'ct' / file_name, repo_root / file_name]
    for directory in search_dirs:
        matches.extend(directory.rglob(file_name))
    return _existing_paths(matches)


def _discover_ct_verif_reports(repo_root: Path, search_dirs: list[Path]) -> list[Path]:
    matches = []
    for directory in [repo_root, *search_dirs]:
        matches.extend(directory.rglob('*_report.txt'))
    return _existing_paths([path for path in matches if path.parent.name == 'ct-ir'])


def _copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() == destination.resolve():
        return
    shutil.copy2(source, destination)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding='utf-8')


def _parse_ct_verif_verdict(text: str) -> str:
    lowered = text.lower()
    passive_tokens = ('ct-verif backend: passive', 'mode: passive')
    fail_tokens = ('ct violation', 'blocking merge', '::error::', 'reported ct violation')
    if any(token in lowered for token in passive_tokens):
        return 'PASSIVE'
    pass_tokens = ('all ct modules verified constant-time', 'no ct violations found')
    if any(token in lowered for token in fail_tokens):
        return 'FAIL'
    if any(token in lowered for token in pass_tokens):
        return 'PASS'
    return 'UNKNOWN'


def _extract_ct_verif_modules(log_path: Path, report_files: list[Path]) -> list[str]:
    modules: list[str] = []
    for report in report_files:
        name = report.stem
        modules.append(name[:-7] if name.endswith('_report') else name)

    if modules:
        return sorted(dict.fromkeys(modules))

    if not log_path.exists():
        return []

    for raw_line in _read_text(log_path).splitlines():
        line = raw_line.strip()
        if line.startswith('=====') and '_report.txt' in line:
            report_name = Path(line.strip('=').strip()).stem
            modules.append(report_name[:-7] if report_name.endswith('_report') else report_name)
            continue
        if line.startswith('--- Analyzing:'):
            module = line.split(':', 1)[1].strip().strip('- ').split()[0]
            if module:
                modules.append(module)

    return sorted(dict.fromkeys(modules))


def _parse_valgrind_log(path: Path) -> dict:
    text = _read_text(path)
    ct_branch_errors = text.count('Conditional jump or move depends on uninitialised')
    uninit_value_errors = text.count('Use of uninitialised value')
    verdict = 'PASS' if ct_branch_errors == 0 else 'FAIL'
    return {
        'tool': 'valgrind_ct_check',
        'generated_by': 'scripts/collect_ct_evidence.py',
        'source_log': _safe_rel(path),
        'ct_branch_errors': ct_branch_errors,
        'uninit_value_errors': uninit_value_errors,
        'verdict': verdict,
    }


def _synthesize_ct_verif_log(report_files: list[Path], destination: Path) -> None:
    chunks = []
    for report in report_files:
        chunks.append(f'===== {_safe_rel(report)} =====\n')
        chunks.append(_read_text(report).rstrip())
        chunks.append('\n\n')
    _write_text(destination, ''.join(chunks))


def _synthesize_ct_verif_summary(log_path: Path, report_files: list[Path], destination: Path) -> dict:
    report_payloads = []
    overall_verdict = 'UNKNOWN'
    backend_mode = 'unknown'
    analyzed_modules = _extract_ct_verif_modules(log_path, report_files)
    missing_modules = [module for module in CT_VERIF_EXPECTED_MODULES if module not in analyzed_modules]
    if report_files:
        verdicts = []
        for report in report_files:
            verdict = _parse_ct_verif_verdict(_read_text(report))
            verdicts.append(verdict)
            report_payloads.append({'path': _safe_rel(report), 'verdict': verdict})
        if verdicts and all(verdict == 'PASS' for verdict in verdicts):
            overall_verdict = 'PASS'
            backend_mode = 'active'
        elif any(verdict == 'FAIL' for verdict in verdicts):
            overall_verdict = 'FAIL'
            backend_mode = 'active'
    elif log_path.exists():
        overall_verdict = _parse_ct_verif_verdict(_read_text(log_path))
        if overall_verdict in ('PASS', 'FAIL'):
            backend_mode = 'active'
        elif overall_verdict == 'PASSIVE':
            backend_mode = 'passive'

    payload = {
        'tool': 'ct_verif',
        'generated_by': 'scripts/collect_ct_evidence.py',
        'source_log': _safe_rel(log_path) if log_path.exists() else None,
        'report_count': len(report_files),
        'reports': report_payloads,
        'expected_modules': CT_VERIF_EXPECTED_MODULES,
        'analyzed_modules': analyzed_modules,
        'missing_modules': missing_modules,
        'module_coverage_complete': not missing_modules,
        'verdict': overall_verdict,
        'backend_mode': backend_mode,
        'deterministic_evidence': backend_mode == 'active' and overall_verdict in ('PASS', 'FAIL'),
    }
    _write_json(destination, payload)
    return payload


def _run_disasm_if_missing(repo_root: Path, runner_binary: Path | None, output_dir: Path) -> list[str]:
    notes: list[str] = []
    if runner_binary is None:
        return notes

    disasm_script = repo_root / 'scripts' / 'verify_ct_disasm.sh'
    if not disasm_script.exists():
        return notes

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / 'disasm_branch_scan.json'
    txt_path = output_dir / 'disasm_branch_scan.txt'

    result = subprocess.run(
        ['bash', str(disasm_script), str(runner_binary), '--json', str(json_path)],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    _write_text(txt_path, (result.stdout or '') + (result.stderr or ''))
    if json_path.exists():
        notes.append('generated disasm_branch_scan.json via scripts/verify_ct_disasm.sh')
    elif result.returncode != 0:
        notes.append(f'disasm scan generation failed with exit code {result.returncode}')
    return notes


def _build_layer_result(
    spec: LayerSpec,
    repo_root: Path,
    output_dir: Path,
    search_dirs: list[Path],
    runner_binary: Path | None,
) -> LayerResult:
    configured_surfaces = _discover_configured_surfaces(repo_root, spec.configured_surfaces)
    binary_paths = _discover_binary_paths(search_dirs, spec.binary_names)
    canonical_artifacts: list[str] = []
    source_paths: list[str] = []
    notes: list[str] = []
    metadata: dict = {}

    if spec.layer_id == 'manual_ct_docs':
        status = 'manual-only' if configured_surfaces else 'missing'
        return LayerResult(
            layer_id=spec.layer_id,
            evidence_class=spec.evidence_class,
            summary_label=spec.summary_label,
            status=status,
            canonical_artifacts=canonical_artifacts,
            source_paths=source_paths,
            configured_surfaces=[_safe_rel(path) for path in configured_surfaces],
            binary_paths=[_safe_rel(path) for path in binary_paths],
            notes=notes,
            metadata=metadata,
            owner_grade_gap=False,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    discovered: dict[str, list[Path]] = {
        file_name: _discover_named_files(repo_root, output_dir, search_dirs, file_name)
        for file_name in spec.canonical_files
    }

    if spec.layer_id == 'ct_verif':
        report_files = _discover_ct_verif_reports(repo_root, search_dirs)
        if report_files and not discovered['ct_verif.log']:
            synthesized_log = output_dir / 'ct_verif.log'
            _synthesize_ct_verif_log(report_files, synthesized_log)
            discovered['ct_verif.log'] = [synthesized_log]
            notes.append('synthesized ct_verif.log from ct-ir/*_report.txt files')

        if discovered['ct_verif.log']:
            synthesized_summary = output_dir / 'ct_verif_summary.json'
            summary_payload = _synthesize_ct_verif_summary(discovered['ct_verif.log'][0], report_files, synthesized_summary)
            discovered['ct_verif_summary.json'] = [synthesized_summary]
            notes.append('generated ct_verif_summary.json from available ct-verif outputs')
            metadata['summary'] = summary_payload

    if spec.layer_id == 'valgrind_ct' and discovered['valgrind_ct.log'] and not discovered['valgrind_ct_report.json']:
        synthesized_report = output_dir / 'valgrind_ct_report.json'
        report_payload = _parse_valgrind_log(discovered['valgrind_ct.log'][0])
        _write_json(synthesized_report, report_payload)
        discovered['valgrind_ct_report.json'] = [synthesized_report]
        notes.append('generated valgrind_ct_report.json from valgrind_ct.log')
        metadata['summary'] = report_payload

    if spec.layer_id == 'disasm_scan' and not discovered['disasm_branch_scan.json']:
        notes.extend(_run_disasm_if_missing(repo_root, runner_binary, output_dir))
        discovered['disasm_branch_scan.json'] = _discover_named_files(repo_root, output_dir, search_dirs, 'disasm_branch_scan.json')

    for file_name in spec.canonical_files:
        candidates = discovered[file_name]
        if not candidates:
            continue
        source = candidates[0]
        destination = output_dir / file_name
        _copy_file(source, destination)
        canonical_artifacts.append(_safe_rel(destination))
        source_paths.append(_safe_rel(source))

    status = 'artifact-present' if canonical_artifacts else ('configured-only' if configured_surfaces or binary_paths else 'missing')

    if spec.layer_id == 'ct_verif':
        summary = metadata.get('summary')
        if summary is None and (output_dir / 'ct_verif_summary.json').exists():
            summary = json.loads(_read_text(output_dir / 'ct_verif_summary.json'))
            metadata['summary'] = summary
        if summary and not summary.get('deterministic_evidence', False):
            status = 'configured-only'
            notes.append('ct-verif artifacts are passive-only in this environment; not counted as deterministic evidence')
        if summary and not summary.get('module_coverage_complete', False):
            status = 'configured-only'
            missing_modules = ', '.join(summary.get('missing_modules', [])) or 'unknown'
            notes.append(f'ct-verif missing expected modules: {missing_modules}')

    owner_grade_gap = spec.required_for_owner_grade and status != 'artifact-present'

    if spec.layer_id == 'ct_verif' and 'summary' not in metadata and (output_dir / 'ct_verif_summary.json').exists():
        metadata['summary'] = json.loads(_read_text(output_dir / 'ct_verif_summary.json'))
    if spec.layer_id == 'valgrind_ct' and 'summary' not in metadata and (output_dir / 'valgrind_ct_report.json').exists():
        metadata['summary'] = json.loads(_read_text(output_dir / 'valgrind_ct_report.json'))

    return LayerResult(
        layer_id=spec.layer_id,
        evidence_class=spec.evidence_class,
        summary_label=spec.summary_label,
        status=status,
        canonical_artifacts=sorted(dict.fromkeys(canonical_artifacts)),
        source_paths=sorted(dict.fromkeys(source_paths)),
        configured_surfaces=[_safe_rel(path) for path in configured_surfaces],
        binary_paths=[_safe_rel(path) for path in binary_paths],
        notes=notes,
        metadata=metadata,
        owner_grade_gap=owner_grade_gap,
    )


def _render_text_summary(report: dict) -> str:
    lines = [
        '============================================================',
        '  Constant-Time Evidence Summary',
        '============================================================',
        '',
        f"Output directory: {report['output_dir']}",
        f"Overall status:   {report['overall_status']}",
        '',
        'Layers:',
    ]
    for layer in report['layers']:
        lines.append(f"  - {layer['summary_label']} [{layer['evidence_class']}] -> {layer['status']}")
        if layer['canonical_artifacts']:
            lines.append(f"      artifacts: {', '.join(layer['canonical_artifacts'])}")
        if layer['source_paths']:
            lines.append(f"      sources:   {', '.join(layer['source_paths'])}")
        summary = layer.get('metadata', {}).get('summary', {})
        if summary.get('analyzed_modules'):
            lines.append(f"      modules:   {', '.join(summary['analyzed_modules'])}")
        if layer['notes']:
            lines.append(f"      notes:     {'; '.join(layer['notes'])}")
    lines.extend(['', 'Owner-grade gaps:'])
    if report['owner_grade_gaps']:
        for gap in report['owner_grade_gaps']:
            lines.append(f'  - {gap}')
    else:
        lines.append('  - none')
    lines.append('')
    return '\n'.join(lines)


def build_report(repo_root: Path, output_dir: Path, build_dirs: list[Path], runner_binary: Path | None) -> dict:
    search_dirs = _candidate_dirs(repo_root, build_dirs)
    layer_results = [
        _build_layer_result(spec, repo_root, output_dir, search_dirs, runner_binary)
        for spec in LAYER_SPECS
    ]
    layer_results.sort(key=lambda item: STATUS_ORDER.get(item.status, 99))

    owner_grade_gaps = [
        f"{layer.summary_label}: {layer.status}"
        for layer in layer_results
        if layer.owner_grade_gap
    ]
    advisory_gaps = [
        f"{layer.summary_label}: {layer.status}"
        for layer in layer_results
        if not layer.owner_grade_gap and layer.evidence_class == 'statistical' and layer.status != 'artifact-present'
    ]

    if owner_grade_gaps:
        overall_status = 'partial'
    elif any(layer.status == 'artifact-present' for layer in layer_results if layer.evidence_class != 'manual-only'):
        overall_status = 'ready-with-advisories' if advisory_gaps else 'ready'
    else:
        overall_status = 'configured-only'

    report = {
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'repo_root': _safe_rel(repo_root),
        'output_dir': _safe_rel(output_dir),
        'overall_status': overall_status,
        'owner_grade_gaps': owner_grade_gaps,
        'advisory_gaps': advisory_gaps,
        'searched_directories': [_safe_rel(path) for path in search_dirs],
        'layers': [
            {
                'layer_id': layer.layer_id,
                'summary_label': layer.summary_label,
                'evidence_class': layer.evidence_class,
                'status': layer.status,
                'canonical_artifacts': layer.canonical_artifacts,
                'source_paths': layer.source_paths,
                'configured_surfaces': layer.configured_surfaces,
                'binary_paths': layer.binary_paths,
                'notes': layer.notes,
                'metadata': layer.metadata,
                'owner_grade_gap': layer.owner_grade_gap,
            }
            for layer in layer_results
        ],
    }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Collect and normalize CT evidence into canonical artifact slots.')
    parser.add_argument('--repo-root', type=Path, default=LIB_ROOT, help='Repository root (default: script parent)')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR, help='Destination directory for normalized CT artifacts')
    parser.add_argument('--build-dir', action='append', type=Path, default=[], help='Additional build/output directory to inspect')
    parser.add_argument('--runner-binary', type=Path, default=None, help='Optional unified_audit_runner path for disasm generation when missing')
    parser.add_argument('--strict', action='store_true', help='Return non-zero if deterministic owner-grade CT artifacts are missing')
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir if args.output_dir.is_absolute() else (repo_root / args.output_dir)
    runner_binary = None
    if args.runner_binary is not None:
        runner_binary = args.runner_binary if args.runner_binary.is_absolute() else (repo_root / args.runner_binary)
        runner_binary = runner_binary.resolve()

    report = build_report(repo_root, output_dir.resolve(), args.build_dir, runner_binary)
    summary_json = output_dir / 'ct_evidence_summary.json'
    summary_txt = output_dir / 'ct_evidence_summary.txt'
    _write_json(summary_json, report)
    _write_text(summary_txt, _render_text_summary(report))

    print(_render_text_summary(report))
    if args.strict and report['owner_grade_gaps']:
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())