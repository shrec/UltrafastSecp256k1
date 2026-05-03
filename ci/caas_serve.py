#!/usr/bin/env python3
"""
caas_serve.py — Web panel for CAAS dashboard + audit artifact browsing.

Default bind is 0.0.0.0 so the panel is reachable from other machines on the
LAN (e.g. a Windows workstation pointing at a Linux dev box). For local-only
use pass --bind 127.0.0.1.

Routes:
  /                    Rendered CAAS dashboard (scripts/caas_dashboard.py output)
  /refresh             Regenerate the dashboard, then 302 to /
  /artifacts/          Listing of audit-relevant files
  /artifacts/<path>    Single file rendered as HTML (JSON pretty-printed,
                       Markdown / SARIF / log shown as <pre>, .html served raw)

Usage:
  python3 ci/caas_serve.py [--port 8080] [--bind 0.0.0.0]
"""

from __future__ import annotations

import argparse
import html
import json
import mimetypes
import socket
import subprocess
import sys
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

LIB_ROOT = Path(__file__).resolve().parent.parent
DASHBOARD_OUT = LIB_ROOT / "build" / "caas_dashboard.html"
DASHBOARD_GENERATOR = LIB_ROOT / "ci" / "caas_dashboard.py"

# Directories whose contents should be browsable. Order matters for display.
ARTIFACT_ROOTS = [
    LIB_ROOT / "docs",
    LIB_ROOT / "audit-output",
    LIB_ROOT / "build" / "owner_audit",
    LIB_ROOT / "build" / "research_monitor",
    LIB_ROOT / "out" / "reports",
]

MAX_FILE_BYTES = 50_000_000  # 50 MB safety cap on file rendering


# ---------------------------------------------------------------------------
# Dashboard generation
# ---------------------------------------------------------------------------

def regenerate_dashboard() -> bytes:
    DASHBOARD_OUT.parent.mkdir(parents=True, exist_ok=True)
    if not DASHBOARD_GENERATOR.exists():
        # Fall back to a minimal placeholder so the panel still serves.
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
        '<p>scripts/caas_dashboard.py not found. The artifact browser is still available:</p>'
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
# Artifact rendering
# ---------------------------------------------------------------------------

def render_artifacts_index() -> str:
    rows: list[str] = []
    for root in ARTIFACT_ROOTS:
        if not root.exists():
            continue
        rel_root = root.relative_to(LIB_ROOT)
        rows.append(
            f'<tr><td colspan="3" style="background:#f0f0f0;font-weight:bold;">'
            f'{html.escape(str(rel_root))}/</td></tr>'
        )
        for p in sorted(root.rglob("*")):
            if not p.is_file():
                continue
            try:
                size = p.stat().st_size
            except OSError:
                continue
            if size > MAX_FILE_BYTES:
                continue
            rel = p.relative_to(LIB_ROOT)
            mtime = datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds")
            rows.append(
                '<tr>'
                f'<td><a href="/artifacts/{html.escape(str(rel))}">{html.escape(str(rel))}</a></td>'
                f'<td style="text-align:right;">{size:,}</td>'
                f'<td>{mtime}</td>'
                '</tr>'
            )
    if not any(root.exists() for root in ARTIFACT_ROOTS):
        rows.append('<tr><td colspan="3"><i>No artifact directories present yet.</i></td></tr>')
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>CAAS Artifacts</title>
<style>
  body {{ font-family: monospace; max-width: 1200px; margin: 1em auto; padding: 0 1em; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ padding: 0.35em 0.8em; border-bottom: 1px solid #eee; text-align: left; }}
  th {{ background: #f0f0f0; }}
  a {{ color: #0066cc; text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
  .nav {{ margin-bottom: 1em; padding-bottom: 0.6em; border-bottom: 1px solid #ddd; }}
  .nav a {{ margin-right: 1em; color: #0066cc; text-decoration: none; }}
</style></head><body>
<div class="nav">
  <a href="/">Dashboard</a>
  <a href="/artifacts/">All Artifacts</a>
  <a href="/refresh">Refresh</a>
</div>
<h1>CAAS Artifacts</h1>
<table>
<tr><th>Path</th><th style="text-align:right;">Size (B)</th><th>Modified (UTC)</th></tr>
{"".join(rows)}
</table>
</body></html>"""


# ---------------------------------------------------------------------------
# Human-readable JSON renderer
# ---------------------------------------------------------------------------

# Tuned for the audit JSON shapes in this repo (EXTERNAL_AUDIT_BUNDLE,
# SECURITY_AUTONOMY_KPI, EVIDENCE_CHAIN, owner_audit_bundle, etc.). Renders
# nested objects as collapsible <details> blocks and arrays of homogeneous
# objects as proper HTML tables instead of raw JSON.

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
    # Wrap deeper objects in collapsible details for compact display.
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

def render_artifact_page(path: Path, raw: bool = False) -> tuple[bytes, str]:
    suffix = path.suffix.lower()
    rel = path.relative_to(LIB_ROOT)

    # Binary or unreadable — stream raw with best-guess content type.
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        ctype = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        return path.read_bytes(), ctype

    if suffix in (".html", ".htm"):
        return text.encode("utf-8"), "text/html; charset=utf-8"

    body_html: str
    if suffix == ".json" and not raw:
        try:
            data = json.loads(text)
            body_html = (
                '<div class="json-toolbar">'
                f'<a href="?raw=1" class="rawlink">View raw JSON</a>'
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
            f'<a href="?" class="rawlink">View structured</a>'
            '</div>'
            f'<pre>{html.escape(pretty)}</pre>'
        )
    else:
        body_html = f"<pre>{html.escape(text)}</pre>"

    page = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{html.escape(str(rel))}</title>
<style>
  body {{ font-family: -apple-system, "Segoe UI", system-ui, sans-serif; max-width: 1400px; margin: 1em auto; padding: 0 1em; color: #222; }}
  .nav {{ margin-bottom: 1em; padding-bottom: 0.6em; border-bottom: 1px solid #ddd; }}
  .nav a {{ margin-right: 1em; color: #0066cc; text-decoration: none; }}
  h2 {{ font-family: monospace; font-size: 1.15em; word-break: break-all; }}
  .meta {{ color: #666; margin-bottom: 0.6em; font-size: 0.9em; }}
  pre {{ font-family: monospace; background: #f7f7f7; padding: 1em; overflow: auto; line-height: 1.4; border: 1px solid #eee; border-radius: 3px; }}
  .json-toolbar {{ margin-bottom: 0.6em; }}
  .rawlink {{ color: #0066cc; font-size: 0.9em; }}

  /* Human-readable JSON */
  .json-tree {{ font-size: 14px; }}
  table.kv {{ border-collapse: collapse; width: 100%; margin: 0.2em 0; }}
  table.kv > tr > th.kcol {{
    text-align: left; padding: 0.25em 0.7em 0.25em 0;
    font-weight: 600; color: #333; vertical-align: top;
    white-space: nowrap; min-width: 11em;
    border-bottom: 1px dotted #eee;
  }}
  table.kv > tr > td {{ padding: 0.25em 0; vertical-align: top; border-bottom: 1px dotted #eee; }}
  table.rows {{ border-collapse: collapse; margin: 0.4em 0; min-width: 60%; }}
  table.rows th, table.rows td {{ padding: 0.3em 0.7em; border: 1px solid #e2e2e2; text-align: left; vertical-align: top; }}
  table.rows th {{ background: #f3f3f3; font-weight: 600; }}
  table.rows tr:nth-child(even) td {{ background: #fafafa; }}
  ol.arr {{ margin: 0.2em 0 0.2em 1.5em; padding: 0; }}
  ol.arr > li {{ margin: 0.2em 0; }}
  details {{ margin: 0.2em 0; }}
  details > summary {{ cursor: pointer; color: #555; font-style: italic; }}
  .v-str  {{ color: #1a7f1a; }}
  .v-num  {{ color: #b35900; font-variant-numeric: tabular-nums; }}
  .v-bool {{ color: #5a3fc0; font-weight: 600; }}
  .v-null {{ color: #999; font-style: italic; }}
  .v-empty {{ color: #999; font-style: italic; }}
</style></head><body>
<div class="nav">
  <a href="/">Dashboard</a>
  <a href="/artifacts/">All Artifacts</a>
  <a href="/refresh">Refresh</a>
</div>
<h2>{html.escape(str(rel))}</h2>
<div class="meta">{path.stat().st_size:,} bytes &middot; modified {datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec='seconds')}</div>
{body_html}
</body></html>"""
    return page.encode("utf-8"), "text/html; charset=utf-8"


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

def make_handler(cache: list[bytes]):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *a):  # noqa: D401 — stdlib override
            sys.stderr.write(f"[{self.address_string()}] {fmt % a}\n")

        def do_GET(self):  # noqa: N802 — stdlib override
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
                    # Path traversal guard: candidate must live under LIB_ROOT.
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
            except Exception as exc:  # surface server errors to the browser
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
    ap.add_argument("--bind", default="0.0.0.0",
                    help="Interface to bind. 0.0.0.0 = all interfaces (LAN-reachable).")
    args = ap.parse_args()

    print("Generating dashboard...", flush=True)
    cache = [inject_navbar(regenerate_dashboard())]

    handler = make_handler(cache)
    srv = HTTPServer((args.bind, args.port), handler)
    ip = lan_ip()
    print()
    print("CAAS Web Panel up:")
    print(f"  Local:     http://localhost:{args.port}/")
    if args.bind == "0.0.0.0":
        print(f"  LAN:       http://{ip}:{args.port}/")
        print(f"  Artifacts: http://{ip}:{args.port}/artifacts/")
    print()
    print("Ctrl-C to stop.", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
