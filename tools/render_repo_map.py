#!/usr/bin/env python3
"""Render the README repository map SVG from a JSON manifest.

The SVG is intentionally static: GitHub renders SVG safely, but strips script
execution. Treat this as a reproducible, generated visual index.
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "docs" / "repo_map_manifest.json"
DEFAULT_OUTPUT = ROOT / "docs" / "assets" / "repo-map.svg"

TIER_COLORS = {
    "production": ("#0f766e", "#ecfeff"),
    "bitcoin-core-scope": ("#1d4ed8", "#eff6ff"),
    "public-data-only": ("#7c3aed", "#f5f3ff"),
    "needs-hardening": ("#b45309", "#fffbeb"),
    "compat-only": ("#64748b", "#f8fafc"),
    "evidence-system": ("#be123c", "#fff1f2"),
    "claim-control": ("#047857", "#ecfdf5"),
    "experimental": ("#9333ea", "#faf5ff"),
    "deprecated": ("#991b1b", "#fef2f2"),
}


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def wrap_text(text: str, max_chars: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    for word in words:
        candidate = " ".join([*current, word])
        if current and len(candidate) > max_chars:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))
    return lines


def text_block(lines: list[str], x: int, y: int, *, size: int = 13, weight: str = "400",
               fill: str = "#334155", line_height: int = 18) -> str:
    parts = []
    for idx, line in enumerate(lines):
        parts.append(
            f'<text x="{x}" y="{y + idx * line_height}" font-size="{size}" '
            f'font-weight="{weight}" fill="{fill}">{esc(line)}</text>'
        )
    return "\n".join(parts)


def render_card(profile: dict, x: int, y: int, width: int, height: int) -> str:
    tier = str(profile.get("tier", "experimental"))
    stroke, fill = TIER_COLORS.get(tier, TIER_COLORS["experimental"])
    title = str(profile.get("title", profile.get("id", "Untitled")))
    summary = str(profile.get("summary", ""))
    paths = [str(p) for p in profile.get("paths", [])]
    signals = [str(s) for s in profile.get("signals", [])]

    title_lines = wrap_text(title, 28)[:2]
    summary_lines = wrap_text(summary, 42)[:3]
    path_lines = paths[:4]
    signal_lines = signals[:3]

    body = [
        f'<g id="profile-{esc(profile.get("id", title))}">',
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="18" fill="{fill}" stroke="{stroke}" stroke-width="2"/>',
        f'<rect x="{x + 18}" y="{y + 18}" width="{min(210, width - 36)}" height="28" rx="14" fill="{stroke}"/>',
        f'<text x="{x + 32}" y="{y + 37}" font-size="13" font-weight="700" fill="#ffffff">{esc(tier)}</text>',
        text_block(title_lines, x + 22, y + 72, size=18, weight="800", fill="#0f172a", line_height=22),
        text_block(summary_lines, x + 22, y + 116, size=13, fill="#334155", line_height=17),
        f'<text x="{x + 22}" y="{y + 184}" font-size="12" font-weight="700" fill="#475569">Primary paths</text>',
    ]

    py = y + 206
    for path in path_lines:
        body.append(f'<text x="{x + 34}" y="{py}" font-size="12" fill="#0f172a">- {esc(path)}</text>')
        py += 16

    sy = y + 278
    body.append(f'<text x="{x + 22}" y="{sy}" font-size="12" font-weight="700" fill="#475569">Signals</text>')
    sy += 21
    for signal in signal_lines:
        body.append(f'<text x="{x + 34}" y="{sy}" font-size="12" fill="#0f172a">- {esc(signal)}</text>')
        sy += 16

    body.append("</g>")
    return "\n".join(body)


def render_svg(manifest: dict) -> str:
    profiles = list(manifest.get("profiles", []))
    width = 1400
    card_w = 310
    card_h = 360
    gap_x = 28
    gap_y = 32
    left = 44
    top = 178
    cols = 4
    rows = (len(profiles) + cols - 1) // cols
    height = top + rows * card_h + (rows - 1) * gap_y + 80

    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        "<title id=\"title\">UltrafastSecp256k1 Repository Map</title>",
        "<desc id=\"desc\">Generated visual map of repository product profiles, tiers, and primary paths.</desc>",
        "<defs>",
        "<linearGradient id=\"bg\" x1=\"0\" y1=\"0\" x2=\"1\" y2=\"1\">",
        "<stop offset=\"0%\" stop-color=\"#f8fafc\"/>",
        "<stop offset=\"100%\" stop-color=\"#e0f2fe\"/>",
        "</linearGradient>",
        "<filter id=\"shadow\" x=\"-10%\" y=\"-10%\" width=\"120%\" height=\"120%\">",
        "<feDropShadow dx=\"0\" dy=\"8\" stdDeviation=\"10\" flood-color=\"#0f172a\" flood-opacity=\"0.10\"/>",
        "</filter>",
        "</defs>",
        '<rect width="100%" height="100%" fill="url(#bg)"/>',
        '<g font-family="Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif">',
        f'<text x="44" y="62" font-size="34" font-weight="900" fill="#0f172a">{esc(manifest.get("title", "Repository Map"))}</text>',
        f'<text x="44" y="96" font-size="16" fill="#334155">{esc(manifest.get("subtitle", ""))}</text>',
        f'<text x="44" y="126" font-size="13" fill="#64748b">{esc(manifest.get("generated_note", ""))}</text>',
        '<text x="1110" y="62" font-size="13" fill="#475569">Scope legend</text>',
    ]

    legend = [
        ("production", "Production"),
        ("bitcoin-core-scope", "Core scope"),
        ("public-data-only", "Public data"),
        ("needs-hardening", "Needs hardening"),
        ("compat-only", "Compat only"),
    ]
    lx, ly = 1110, 84
    for idx, (tier, label) in enumerate(legend):
        stroke, fill = TIER_COLORS[tier]
        y = ly + idx * 24
        parts.append(f'<rect x="{lx}" y="{y - 12}" width="14" height="14" rx="4" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
        parts.append(f'<text x="{lx + 22}" y="{y}" font-size="12" fill="#334155">{esc(label)}</text>')

    parts.append('<g filter="url(#shadow)">')
    for idx, profile in enumerate(profiles):
        col = idx % cols
        row = idx // cols
        x = left + col * (card_w + gap_x)
        y = top + row * (card_h + gap_y)
        parts.append(render_card(profile, x, y, card_w, card_h))
    parts.append("</g>")
    parts.append("</g>")
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true", help="fail if output is stale")
    args = parser.parse_args()

    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"error: failed to read manifest {args.manifest}: {exc}", file=sys.stderr)
        return 2

    svg = render_svg(manifest)
    if args.check:
        existing = args.output.read_text(encoding="utf-8") if args.output.exists() else ""
        if existing != svg:
            print(f"error: {args.output} is stale; run tools/render_repo_map.py", file=sys.stderr)
            return 1
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(svg, encoding="utf-8")
    print(f"wrote {args.output.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
