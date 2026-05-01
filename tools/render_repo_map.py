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


def bullet_list(items: list[str], x: int, y: int, *, size: int = 14,
                fill: str = "#0f172a", line_height: int = 19) -> str:
    parts = []
    for idx, item in enumerate(items):
        parts.append(f'<circle cx="{x}" cy="{y + idx * line_height - 4}" r="3.2" fill="#64748b"/>')
        parts.append(
            f'<text x="{x + 12}" y="{y + idx * line_height}" font-size="{size}" '
            f'fill="{fill}">{esc(item)}</text>'
        )
    return "\n".join(parts)


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

    title_lines = wrap_text(title, 36)[:2]
    summary_lines = wrap_text(summary, 62)[:3]
    path_lines = paths[:4]
    signal_lines = signals[:3]
    column_w = (width - 64) // 2

    body = [
        f'<g id="profile-{esc(profile.get("id", title))}">',
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="22" fill="{fill}" stroke="{stroke}" stroke-width="3"/>',
        f'<rect x="{x + 24}" y="{y + 22}" width="{min(250, width - 48)}" height="34" rx="17" fill="{stroke}"/>',
        f'<text x="{x + 42}" y="{y + 45}" font-size="15" font-weight="800" fill="#ffffff">{esc(tier)}</text>',
        text_block(title_lines, x + 26, y + 92, size=24, weight="900", fill="#0f172a", line_height=29),
        text_block(summary_lines, x + 26, y + 148, size=16, fill="#334155", line_height=22),
        f'<text x="{x + 26}" y="{y + 238}" font-size="15" font-weight="800" fill="#334155">Primary paths</text>',
        f'<text x="{x + 42 + column_w}" y="{y + 238}" font-size="15" font-weight="800" fill="#334155">Signals</text>',
    ]

    body.append(bullet_list(path_lines, x + 34, y + 268, size=14, line_height=21))
    body.append(bullet_list(signal_lines, x + 50 + column_w, y + 268, size=14, line_height=21))

    body.append("</g>")
    return "\n".join(body)


def render_svg(manifest: dict) -> str:
    profiles = list(manifest.get("profiles", []))
    width = 1400
    card_w = 640
    card_h = 350
    gap_x = 36
    gap_y = 34
    left = 42
    top = 176
    cols = 2
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
        f'<text x="44" y="60" font-size="38" font-weight="950" fill="#0f172a">{esc(manifest.get("title", "Repository Map"))}</text>',
        f'<text x="44" y="98" font-size="19" font-weight="500" fill="#334155">{esc(manifest.get("subtitle", ""))}</text>',
        f'<text x="44" y="130" font-size="14" fill="#64748b">{esc(manifest.get("generated_note", ""))}</text>',
        '<text x="980" y="54" font-size="15" font-weight="800" fill="#334155">Scope legend</text>',
    ]

    legend = [
        ("production", "Production"),
        ("bitcoin-core-scope", "Core scope"),
        ("public-data-only", "Public data"),
        ("needs-hardening", "Needs hardening"),
        ("compat-only", "Compat only"),
    ]
    lx, ly = 980, 82
    for idx, (tier, label) in enumerate(legend):
        stroke, fill = TIER_COLORS[tier]
        col = idx % 2
        row = idx // 2
        x = lx + col * 190
        y = ly + row * 28
        parts.append(f'<rect x="{x}" y="{y - 15}" width="18" height="18" rx="5" fill="{fill}" stroke="{stroke}" stroke-width="2.5"/>')
        parts.append(f'<text x="{x + 28}" y="{y}" font-size="14" fill="#334155">{esc(label)}</text>')

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
