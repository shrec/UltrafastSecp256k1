#!/usr/bin/env python3
"""
caas_serve.py — Web panel for CAAS dashboard + audit artifact browsing.

Default bind is 127.0.0.1 (local-only). For LAN access use --lan or
explicitly pass --bind 0.0.0.0.

Routes:
  /                    Rendered CAAS dashboard (ci/caas_dashboard.py output)
  /refresh             Regenerate the dashboard, then 302 to /
  /artifacts/          Searchable listing of audit-relevant files
  /artifacts/<path>    Single file rendered as HTML:
                         .json  — structured tree view with raw toggle
                         .md    — rendered Markdown (headings, code, links)
                         .sarif — results table (rule | level | file:line | message)
                         .html  — served as-is
                         other  — <pre> with raw text

Usage:
  python3 ci/caas_serve.py [--port 8080] [--bind 127.0.0.1] [--lan]
"""

from __future__ import annotations

import argparse
import html
import json
import mimetypes
import re
import socket
import subprocess
import sys
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

LIB_ROOT = Path(__file__).resolve().parent.parent
DASHBOARD_OUT = LIB_ROOT / "out" / "caas_dashboard.html"
DASHBOARD_GENERATOR = LIB_ROOT / "ci" / "caas_dashboard.py"

ARTIFACT_ROOTS = [
    LIB_ROOT / "docs",
    LIB_ROOT / "out" / "audit-output",
    LIB_ROOT / "out" / "owner_audit",
    LIB_ROOT / "out" / "research_monitor",
    LIB_ROOT / "out" / "reports",
    LIB_ROOT / "out" / "artifacts",
    LIB_ROOT / "out" / "auditor_mode",
]

MAX_FILE_BYTES = 50_000_000  # 50 MB safety cap


# ---------------------------------------------------------------------------
# Dashboard generation
# ---------------------------------------------------------------------------

def regenerate_dashboard() -> bytes:
    DASHBOARD_OUT.parent.mkdir(parents=True, exist_ok=True)
    if not DASHBOARD_GENERATOR.exists():
        return _placeholder_dashboard().encode("utf-8")
    subprocess.run(
        [sys.executable, str(DASHBOARD_GENERATOR), "-o", str(DASHBOARD_OUT)],
        cwd=str(LIB_ROOT),
        check=True,
    )
    return DASHBOARD_OUT.read_bytes()


def _placeholder_dashboard() -> str:
    return (
        '<!DOCTYPE html><html><head><meta charset="utf-8">'
        '<title>CAAS Web Panel</title></head><body>'
        '<h1>CAAS Web Panel</h1>'
        '<p>ci/caas_dashboard.py not found. The artifact browser is still available:</p>'
        '<p><a href="/artifacts/">Browse artifacts</a></p>'
        '</body></html>'
    )


def inject_navbar(dashboard_html: bytes) -> bytes:
    banner = (
        b'<div style="background:#fffae5;padding:0.7em 1em;'
        b'border-bottom:1px solid #ddd;font-family:monospace;font-size:14px;">'
        b'<a href="/" style="margin-right:1em;color:#0066cc;text-decoration:none;">Dashboard</a>'
        b'<a href="/artifacts/" style="margin-right:1em;color:#0066cc;text-decoration:none;">Browse Artifacts</a>'
        b'<a href="/refresh" style="color:#0066cc;text-decoration:none;">Refresh</a>'
        b'</div>'
    )
    if b"<body" in dashboard_html:
        idx = dashboard_html.find(b">", dashboard_html.find(b"<body")) + 1
        return dashboard_html[:idx] + banner + dashboard_html[idx:]
    return banner + dashboard_html


# ---------------------------------------------------------------------------
# Freshness badge
# ---------------------------------------------------------------------------

def _freshness_badge(mtime: float) -> str:
    age_hours = (datetime.now(timezone.utc).timestamp() - mtime) / 3600
    if age_hours < 24:
        return '<span style="background:#1a3a22;color:#3fb950;border:1px solid #2ea043;padding:1px 6px;border-radius:4px;font-size:.7em;font-weight:700;">FRESH</span>'
    if age_hours < 24 * 7:
        return '<span style="background:#3a2a0a;color:#d29922;border:1px solid #9e6a03;padding:1px 6px;border-radius:4px;font-size:.7em;font-weight:700;">RECENT</span>'
    return '<span style="background:#3a1a1a;color:#f85149;border:1px solid #da3633;padding:1px 6px;border-radius:4px;font-size:.7em;font-weight:700;">STALE</span>'


# ---------------------------------------------------------------------------
# Shared dark theme
# ---------------------------------------------------------------------------

_DARK_BASE = """
:root{--bg:#0d1117;--bg2:#161b22;--bg3:#21262d;--border:#30363d;
--text:#e6edf3;--text2:#8b949e;--blue:#58a6ff;--cyan:#39d353;
--green:#3fb950;--yellow:#d29922;--red:#f85149;}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',system-ui,sans-serif;background:var(--bg);
color:var(--text);font-size:14px;line-height:1.6}
a{color:var(--blue);text-decoration:none}a:hover{text-decoration:underline}
.nav{background:var(--bg2);border-bottom:1px solid var(--border);
padding:.5rem 1.2rem;display:flex;gap:1rem;align-items:center;font-size:.82rem;
position:sticky;top:0;z-index:100}
.nav a{color:var(--text2)}
.nav a:hover{color:var(--text)}
.wrap{max-width:1400px;margin:0 auto;padding:1rem 1.2rem}
h1{font-size:1.2rem;font-weight:600;margin:.8rem 0 .6rem;color:var(--blue)}
table{width:100%;border-collapse:collapse;font-size:.82rem}
th{background:var(--bg3);color:var(--text2);font-weight:600;text-align:left;
padding:.4rem .7rem;border-bottom:1px solid var(--border);font-size:.72rem;letter-spacing:.04em}
td{padding:.35rem .7rem;border-bottom:1px solid #21262d;vertical-align:top}
tr:hover td{background:#1c2128}
code{background:var(--bg3);border:1px solid var(--border);border-radius:3px;
padding:.1em .35em;font-size:.82em;font-family:monospace}
"""

_NAV_HTML = (
    '<nav class="nav">'
    '<a href="/">Dashboard</a>'
    '<a href="/artifacts/">Artifacts</a>'
    '<a href="/refresh">Refresh</a>'
    '</nav>'
)

# ---------------------------------------------------------------------------
# Artifact index with search + freshness
# ---------------------------------------------------------------------------

def render_artifacts_index() -> str:
    rows: list[str] = []
    for root in ARTIFACT_ROOTS:
        if not root.exists():
            continue
        rel_root = root.relative_to(LIB_ROOT)
        rows.append(
            f'<tr class="group-header"><td colspan="4"'
            f' style="background:var(--bg2);color:var(--text2);font-weight:600;padding:.6rem .7rem">'
            f'{html.escape(str(rel_root))}/</td></tr>'
        )
        for p in sorted(root.rglob("*")):
            if not p.is_file():
                continue
            try:
                st = p.stat()
            except OSError:
                continue
            if st.st_size > MAX_FILE_BYTES:
                continue
            rel = p.relative_to(LIB_ROOT)
            mtime_str = datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds")
            badge = _freshness_badge(st.st_mtime)
            rows.append(
                '<tr class="artifact-row">'
                f'<td><a href="/artifacts/{html.escape(str(rel))}">{html.escape(str(rel))}</a></td>'
                f'<td style="text-align:right;color:var(--text2)">{st.st_size:,}</td>'
                f'<td style="color:var(--text2)">{mtime_str}</td>'
                f'<td>{badge}</td>'
                '</tr>'
            )
    if not any(root.exists() for root in ARTIFACT_ROOTS):
        rows.append('<tr><td colspan="4" style="color:var(--text2)"><i>No artifact directories present yet.</i></td></tr>')

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>CAAS Artifacts</title>
<style>
{_DARK_BASE}
#search-box{{width:100%;padding:.45em .7em;margin:.6rem 0;font-size:.9em;
background:var(--bg3);border:1px solid var(--border);border-radius:5px;
color:var(--text);outline:none}}
#search-box:focus{{border-color:var(--blue)}}
.artifact-row.hidden{{display:none}}
.group-header{{user-select:none}}
</style>
</head><body>
{_NAV_HTML}
<div class="wrap">
<h1>Artifacts</h1>
<input id="search-box" type="search" placeholder="Filter by path, type, or keyword…" oninput="filterRows(this.value)">
<table id="artifact-table">
<tr><th>Path</th><th style="text-align:right">Size (B)</th><th>Modified</th><th>Freshness</th></tr>
{"".join(rows)}
</table>
</div>
<script>
function filterRows(q) {{
  q = q.toLowerCase();
  document.querySelectorAll('#artifact-table .artifact-row').forEach(function(row) {{
    var text = row.textContent.toLowerCase();
    row.classList.toggle('hidden', q.length > 0 && text.indexOf(q) === -1);
  }});
  document.querySelectorAll('#artifact-table .group-header').forEach(function(hdr) {{
    var sib = hdr.nextElementSibling;
    var hasVisible = false;
    while (sib && sib.classList.contains('artifact-row')) {{
      if (!sib.classList.contains('hidden')) {{ hasVisible = true; break; }}
      sib = sib.nextElementSibling;
    }}
    hdr.classList.toggle('hidden', !hasVisible && q.length > 0);
  }});
}}
</script>
</body></html>"""


# ---------------------------------------------------------------------------
# Markdown renderer (minimal, zero dependencies)
# ---------------------------------------------------------------------------

def render_markdown(text: str) -> str:
    """Render a subset of Markdown to safe HTML."""
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    i = 0
    in_para = False
    in_list = False

    def close_para():
        nonlocal in_para
        if in_para:
            out.append("</p>\n")
            in_para = False

    def close_list():
        nonlocal in_list
        if in_list:
            out.append("</ul>\n")
            in_list = False

    def inline(s: str) -> str:
        # Fenced inline code: `code`
        s = re.sub(r'`([^`]+)`', lambda m: f'<code>{html.escape(m.group(1))}</code>', s)
        # Bold: **text**
        s = re.sub(r'\*\*([^*]+)\*\*', lambda m: f'<strong>{html.escape(m.group(1))}</strong>', s)
        # Italic: *text*
        s = re.sub(r'\*([^*]+)\*', lambda m: f'<em>{html.escape(m.group(1))}</em>', s)
        # Links: [text](url)
        s = re.sub(r'\[([^\]]+)\]\(([^)]+)\)',
                   lambda m: f'<a href="{html.escape(m.group(2))}">{html.escape(m.group(1))}</a>', s)
        return s

    while i < len(lines):
        line = lines[i].rstrip('\n')

        # Fenced code block
        if line.startswith('```'):
            close_para()
            close_list()
            lang = html.escape(line[3:].strip())
            lang_attr = f' class="language-{lang}"' if lang else ''
            code_lines: list[str] = []
            i += 1
            while i < len(lines):
                l2 = lines[i].rstrip('\n')
                if l2.startswith('```'):
                    i += 1
                    break
                code_lines.append(html.escape(l2))
                i += 1
            out.append(f'<pre><code{lang_attr}>{chr(10).join(code_lines)}</code></pre>\n')
            continue

        # ATX headings
        m = re.match(r'^(#{1,4})\s+(.*)', line)
        if m:
            close_para()
            close_list()
            level = len(m.group(1))
            text_h = inline(html.escape(m.group(2)))
            out.append(f'<h{level}>{text_h}</h{level}>\n')
            i += 1
            continue

        # Horizontal rule
        if re.match(r'^[-*_]{3,}\s*$', line):
            close_para()
            close_list()
            out.append('<hr>\n')
            i += 1
            continue

        # Unordered list item
        m = re.match(r'^[-*]\s+(.*)', line)
        if m:
            close_para()
            if not in_list:
                out.append('<ul>\n')
                in_list = True
            out.append(f'<li>{inline(html.escape(m.group(1)))}</li>\n')
            i += 1
            continue

        # Blank line → end paragraph/list
        if not line.strip():
            close_para()
            close_list()
            i += 1
            continue

        # Regular paragraph text
        close_list()
        if not in_para:
            out.append('<p>')
            in_para = True
        else:
            out.append(' ')
        out.append(inline(html.escape(line)))
        i += 1

    close_para()
    close_list()
    return ''.join(out)


# ---------------------------------------------------------------------------
# SARIF table renderer
# ---------------------------------------------------------------------------

def render_sarif_html(data: dict) -> str:
    rows: list[str] = []
    level_cls = {"error": "color:#d32f2f;font-weight:700",
                 "warning": "color:#f57c00;font-weight:700",
                 "note": "color:#1976d2"}
    for run in data.get("runs", []):
        tool_name = run.get("tool", {}).get("driver", {}).get("name", "")
        for result in run.get("results", []):
            rule_id = html.escape(result.get("ruleId", ""))
            level = result.get("level", "warning")
            msg = html.escape(result.get("message", {}).get("text", "")[:200])
            locs = result.get("locations", [])
            if locs:
                phys = locs[0].get("physicalLocation", {})
                uri = html.escape(phys.get("artifactLocation", {}).get("uri", ""))
                line_no = phys.get("region", {}).get("startLine", "")
                loc_str = f"{uri}:{line_no}" if line_no else uri
            else:
                loc_str = ""
            cls = level_cls.get(level, "")
            rows.append(
                f'<tr>'
                f'<td style="font-family:monospace;font-size:.8em">{rule_id}</td>'
                f'<td style="{cls}">{html.escape(level)}</td>'
                f'<td style="font-family:monospace;font-size:.8em;color:#555">{loc_str}</td>'
                f'<td>{msg}</td>'
                f'</tr>'
            )
    total = len(rows)
    tool_html = f'<p style="color:#555;margin-bottom:.5em">Tool: <b>{html.escape(tool_name)}</b> &nbsp;·&nbsp; {total} result{"s" if total != 1 else ""}</p>' if tool_name else f'<p style="color:#555;margin-bottom:.5em">{total} result{"s" if total != 1 else ""}</p>'
    if not rows:
        return tool_html + '<p style="color:green;font-weight:700">✓ No findings</p>'
    return (
        tool_html
        + '<table style="border-collapse:collapse;width:100%;font-size:.85em">'
        '<thead><tr>'
        '<th style="text-align:left;padding:.3em .6em;border-bottom:2px solid #ddd;background:#f5f5f5">Rule</th>'
        '<th style="text-align:left;padding:.3em .6em;border-bottom:2px solid #ddd;background:#f5f5f5">Level</th>'
        '<th style="text-align:left;padding:.3em .6em;border-bottom:2px solid #ddd;background:#f5f5f5">Location</th>'
        '<th style="text-align:left;padding:.3em .6em;border-bottom:2px solid #ddd;background:#f5f5f5">Message</th>'
        '</tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table>'
    )


# ---------------------------------------------------------------------------
# Human-readable JSON renderer
# ---------------------------------------------------------------------------

_PRIM_CSS_CLASS = {bool: "v-bool", int: "v-num", float: "v-num", type(None): "v-null"}


def _fmt_primitive(value) -> str:
    if value is None:
        return '<span class="v-null">null</span>'
    if isinstance(value, bool):
        return f'<span class="v-bool">{str(value).lower()}</span>'
    if isinstance(value, (int, float)):
        return f'<span class="v-num">{html.escape(str(value))}</span>'
    if isinstance(value, str):
        return f'<span class="v-str">{html.escape(value)}</span>'
    return f'<span>{html.escape(repr(value))}</span>'


def _is_homogeneous_list_of_dicts(value) -> bool:
    if not isinstance(value, list) or not value:
        return False
    if not all(isinstance(x, dict) for x in value):
        return False
    keys = set(value[0].keys())
    return all(set(x.keys()) == keys for x in value)


def _render_value(value, depth: int = 0) -> str:
    if isinstance(value, dict):
        return _render_dict(value, depth)
    if isinstance(value, list):
        return _render_list(value, depth)
    return _fmt_primitive(value)


def _render_dict(obj: dict, depth: int) -> str:
    if not obj:
        return '<span class="v-empty">{}</span>'
    rows = []
    for k, v in obj.items():
        rows.append(
            f'<tr><th class="kcol">{html.escape(str(k))}</th>'
            f'<td>{_render_value(v, depth + 1)}</td></tr>'
        )
    table = f'<table class="kv">{"".join(rows)}</table>'
    if depth == 0:
        return table
    summary = f'{{ {len(obj)} key{"s" if len(obj) != 1 else ""} }}'
    return f'<details><summary>{html.escape(summary)}</summary>{table}</details>'


def _render_list(arr: list, depth: int) -> str:
    if not arr:
        return '<span class="v-empty">[]</span>'
    if _is_homogeneous_list_of_dicts(arr):
        cols = list(arr[0].keys())
        head = "".join(f"<th>{html.escape(str(c))}</th>" for c in cols)
        body_rows = []
        for item in arr:
            cells = "".join(
                f"<td>{_render_value(item.get(c), depth + 1)}</td>" for c in cols
            )
            body_rows.append(f"<tr>{cells}</tr>")
        return (
            f'<table class="rows"><thead><tr>{head}</tr></thead>'
            f'<tbody>{"".join(body_rows)}</tbody></table>'
        )
    items = "".join(
        f"<li>{_render_value(v, depth + 1)}</li>" for v in arr
    )
    summary = f"[ {len(arr)} item{'s' if len(arr) != 1 else ''} ]"
    if depth == 0:
        return f'<ol class="arr">{items}</ol>'
    return f'<details><summary>{html.escape(summary)}</summary><ol class="arr">{items}</ol></details>'


def render_json_html(data) -> str:
    return _render_value(data, depth=0)


# ---------------------------------------------------------------------------
# Per-file rendering
# ---------------------------------------------------------------------------

_PAGE_CSS = _DARK_BASE + """
  .wrap { max-width: 1400px; margin: 0 auto; padding: 1rem 1.2rem; }
  h2.file-path { font-family: monospace; font-size: 1em; word-break: break-all;
    color: var(--cyan); margin: .6rem 0 .3rem; font-weight: 600; }
  .meta { color: var(--text2); margin-bottom: .6rem; font-size: .82em; }
  pre { font-family: 'Cascadia Code','Fira Code',monospace; background: var(--bg2);
    color: var(--text); padding: 1em; overflow: auto; line-height: 1.5;
    border: 1px solid var(--border); border-radius: 5px; font-size: .82em; }
  pre code { background: none; border: none; padding: 0; font-size: inherit; }
  .json-toolbar { margin-bottom: .5rem; }
  .rawlink { color: var(--blue); font-size: .85em; }
  .md-body { max-width: 860px; line-height: 1.7; }
  .md-body h1,.md-body h2,.md-body h3,.md-body h4 { margin: 1em 0 .4em; color: var(--blue); }
  .md-body p { margin: .6em 0; color: var(--text); }
  .md-body ul { margin: .4em 0 .4em 1.5em; color: var(--text); }
  .md-body hr { border: none; border-top: 1px solid var(--border); margin: 1em 0; }
  table.kv { border-collapse: collapse; width: 100%; margin: .2em 0; }
  table.kv > tr > th.kcol { text-align: left; padding: .25em .7em .25em 0;
    font-weight: 600; color: var(--blue); vertical-align: top;
    white-space: nowrap; min-width: 11em; border-bottom: 1px solid var(--border); }
  table.kv > tr > td { padding: .25em 0; vertical-align: top; border-bottom: 1px solid var(--border); }
  table.rows { border-collapse: collapse; margin: .4em 0; min-width: 60%; }
  table.rows th, table.rows td { padding: .3em .7em; border: 1px solid var(--border);
    text-align: left; vertical-align: top; }
  table.rows th { background: var(--bg3); font-weight: 600; color: var(--text2); font-size:.72rem; }
  table.rows tr:nth-child(even) td { background: var(--bg2); }
  ol.arr { margin: .2em 0 .2em 1.5em; padding: 0; }
  ol.arr > li { margin: .2em 0; }
  details { margin: .2em 0; }
  details > summary { cursor: pointer; color: var(--text2); font-style: italic; }
  .json-tree { font-size: 13px; }
  .v-str  { color: #7ee787; }
  .v-num  { color: #ffa657; font-variant-numeric: tabular-nums; }
  .v-bool { color: #bc8cff; font-weight: 600; }
  .v-null { color: var(--text2); font-style: italic; }
  .v-empty { color: var(--text2); font-style: italic; }
"""

_NAV = _NAV_HTML


def render_artifact_page(path: Path, raw: bool = False) -> tuple[bytes, str]:
    suffix = path.suffix.lower()
    rel = path.relative_to(LIB_ROOT)

    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        ctype = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        return path.read_bytes(), ctype

    if suffix in (".html", ".htm"):
        return text.encode("utf-8"), "text/html; charset=utf-8"

    st = path.stat()
    meta = (
        f'<div class="meta">{st.st_size:,} bytes'
        f' &middot; modified {datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds")}'
        f' &middot; {_freshness_badge(st.st_mtime)}</div>'
    )

    body_html: str

    if suffix == ".json" and not raw:
        try:
            data = json.loads(text)
            # Detect SARIF by schema key
            if isinstance(data, dict) and data.get("version") and "runs" in data and suffix == ".json":
                body_html = (
                    '<div class="json-toolbar">'
                    '<a href="?raw=1" class="rawlink">View raw JSON</a>'
                    ' &middot; <a href="?structured=1" class="rawlink">View tree</a>'
                    '</div>'
                    + render_sarif_html(data)
                )
            else:
                body_html = (
                    '<div class="json-toolbar">'
                    '<a href="?raw=1" class="rawlink">View raw JSON</a>'
                    '</div>'
                    f'<div class="json-tree">{render_json_html(data)}</div>'
                )
        except json.JSONDecodeError:
            body_html = f"<pre>{html.escape(text)}</pre>"

    elif suffix == ".json" and raw:
        try:
            pretty = json.dumps(json.loads(text), indent=2, ensure_ascii=False)
        except json.JSONDecodeError:
            pretty = text
        body_html = (
            '<div class="json-toolbar">'
            '<a href="?" class="rawlink">View structured</a>'
            '</div>'
            f'<pre>{html.escape(pretty)}</pre>'
        )

    elif suffix == ".sarif":
        try:
            data = json.loads(text)
            body_html = (
                '<div class="json-toolbar">'
                '<a href="?raw=1" class="rawlink">View raw JSON</a>'
                '</div>'
                + render_sarif_html(data)
            )
            if raw:
                try:
                    pretty = json.dumps(data, indent=2, ensure_ascii=False)
                except Exception:
                    pretty = text
                body_html = (
                    '<div class="json-toolbar">'
                    '<a href="?" class="rawlink">View as table</a>'
                    '</div>'
                    f'<pre>{html.escape(pretty)}</pre>'
                )
        except json.JSONDecodeError:
            body_html = f"<pre>{html.escape(text)}</pre>"

    elif suffix == ".md":
        body_html = f'<div class="md-body">{render_markdown(text)}</div>'

    else:
        body_html = f"<pre>{html.escape(text)}</pre>"

    page = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{html.escape(str(rel))}</title>
<style>{_PAGE_CSS}</style></head><body>
{_NAV}
<div class="wrap">
<h2 class="file-path">{html.escape(str(rel))}</h2>
{meta}
{body_html}
</div>
</body></html>"""
    return page.encode("utf-8"), "text/html; charset=utf-8"


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

# VIZ-6 fix: use ThreadingMixIn so the /refresh endpoint (which calls
# subprocess.run synchronously) does not block the single-threaded server.
class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in separate threads to prevent /refresh deadlock."""
    daemon_threads = True


def make_handler(cache: list[bytes]):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *a):
            sys.stderr.write(f"[{self.address_string()}] {fmt % a}\n")

        def do_GET(self):
            try:
                parsed = urlparse(self.path)
                path = unquote(parsed.path)
                if path in ("/", "/index.html"):
                    self._send(200, "text/html; charset=utf-8", cache[0])
                    return
                if path == "/refresh":
                    cache[0] = inject_navbar(regenerate_dashboard())
                    self.send_response(302)
                    self.send_header("Location", "/")
                    self.end_headers()
                    return
                if path in ("/artifacts", "/artifacts/"):
                    body = render_artifacts_index().encode("utf-8")
                    self._send(200, "text/html; charset=utf-8", body)
                    return
                if path.startswith("/artifacts/"):
                    rel = path[len("/artifacts/"):]
                    candidate = (LIB_ROOT / rel).resolve()
                    try:
                        candidate.relative_to(LIB_ROOT)
                    except ValueError:
                        self._send(403, "text/plain", b"forbidden")
                        return
                    if not candidate.is_file():
                        self._send(404, "text/plain", b"not found")
                        return
                    raw = "raw" in parse_qs(parsed.query)
                    body, ctype = render_artifact_page(candidate, raw=raw)
                    self._send(200, ctype, body)
                    return
                self._send(404, "text/plain", b"not found")
            except Exception as exc:
                msg = f"500: {exc}".encode("utf-8", "replace")
                self._send(500, "text/plain; charset=utf-8", msg)

        def _send(self, code: int, ctype: str, body: bytes) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def lan_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--bind", default="127.0.0.1",
                    help="Interface to bind (default: 127.0.0.1 — local only).")
    ap.add_argument("--lan", action="store_true",
                    help="Bind to 0.0.0.0 so the panel is reachable from the LAN.")
    args = ap.parse_args()

    bind = "0.0.0.0" if args.lan else args.bind

    print("Generating dashboard...", flush=True)
    cache = [inject_navbar(regenerate_dashboard())]

    handler = make_handler(cache)
    srv = ThreadedHTTPServer((bind, args.port), handler)
    ip = lan_ip()
    print()
    print("CAAS Web Panel up:")
    print(f"  Local:     http://localhost:{args.port}/")
    if bind == "0.0.0.0":
        print(f"  LAN:       http://{ip}:{args.port}/")
        print(f"  Artifacts: http://{ip}:{args.port}/artifacts/")
    else:
        print(f"  (local-only — use --lan for LAN access)")
    print()
    print("Ctrl-C to stop.", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
