#!/usr/bin/env python3
"""
UltrafastSecp256k1 — Audit Viewer
==================================
Local-only mini web server for browsing all audit artifacts in a browser.
No external dependencies — pure Python stdlib only.

Usage:
    python3 ci/audit_viewer.py [--port 8765] [--host 127.0.0.1]
"""

import argparse
import hashlib
import json
import os
import pathlib
import re
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

LIB_ROOT = pathlib.Path(__file__).resolve().parent.parent
DOCS = LIB_ROOT / "docs"
# audit_viewer fix: use the active source_graph.db (tools/source_graph_kit/)
# not the legacy .project_graph.db at the repo root. The legacy DB is absent
# in most CI clones and on developer machines, causing /graph to always show
# empty data even when the source graph is fully populated.
GRAPH_DB = LIB_ROOT / "tools" / "source_graph_kit" / "source_graph.db"
_LEGACY_GRAPH_DB = LIB_ROOT / ".project_graph.db"  # kept for fallback only

# ---------------------------------------------------------------------------
# CSS / HTML Template
# ---------------------------------------------------------------------------

NAV_LINKS = [
    ("/",            "Home"),
    ("/claims",      "Claims"),
    ("/exploits",    "Exploits"),
    ("/residuals",   "Residuals"),
    ("/ct",          "CT Evidence"),
    ("/parity",      "Parity"),
    ("/bundle",      "Bundle"),
    ("/graph",       "Graph"),
    ("/traceability","Traceability"),
    ("/faq",         "FAQ"),
]

STYLES = """
:root {
  --bg:      #1a1a2e;
  --bg2:     #16213e;
  --bg3:     #0f3460;
  --text:    #e0e0e0;
  --text2:   #a0a0b0;
  --accent:  #4fc3f7;
  --green:   #66bb6a;
  --red:     #ef5350;
  --yellow:  #ffa726;
  --gray:    #757575;
  --orange:  #fb8c00;
  --blue:    #42a5f5;
  --border:  #2a2a4a;
  --code-bg: #0d1117;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  background: var(--bg);
  color: var(--text);
  font-family: 'Segoe UI', system-ui, sans-serif;
  font-size: 15px;
  line-height: 1.6;
}
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }

nav {
  background: var(--bg3);
  border-bottom: 1px solid var(--border);
  padding: 0 24px;
  display: flex;
  align-items: center;
  gap: 4px;
  flex-wrap: wrap;
  position: sticky;
  top: 0;
  z-index: 100;
}
nav .brand {
  font-weight: 700;
  color: var(--accent);
  font-size: 13px;
  padding: 10px 12px 10px 0;
  border-right: 1px solid var(--border);
  margin-right: 8px;
  white-space: nowrap;
}
nav a {
  color: var(--text2);
  padding: 10px 10px;
  font-size: 13px;
  border-radius: 4px;
  transition: background 0.15s, color 0.15s;
  white-space: nowrap;
}
nav a:hover, nav a.active {
  background: rgba(79,195,247,0.12);
  color: var(--accent);
  text-decoration: none;
}

main {
  max-width: 1200px;
  margin: 0 auto;
  padding: 32px 24px 64px;
}

h1 { color: var(--accent); font-size: 1.8rem; margin-bottom: 8px; }
h2 { color: var(--accent); font-size: 1.35rem; margin: 28px 0 10px; border-bottom: 1px solid var(--border); padding-bottom: 6px; }
h3 { color: var(--text); font-size: 1.1rem; margin: 20px 0 8px; }
h4 { color: var(--text2); font-size: 1rem; margin: 14px 0 6px; }
p  { margin-bottom: 12px; }
ul, ol { margin: 8px 0 12px 24px; }
li { margin-bottom: 4px; }

code {
  background: var(--code-bg);
  color: #a8d8ea;
  padding: 2px 6px;
  border-radius: 3px;
  font-family: 'JetBrains Mono', 'Fira Code', 'Consolas', monospace;
  font-size: 0.88em;
}
pre {
  background: var(--code-bg);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 16px;
  overflow-x: auto;
  margin: 12px 0;
}
pre code {
  background: none;
  padding: 0;
  color: #e0e0e0;
  font-size: 0.85em;
}

table {
  width: 100%;
  border-collapse: collapse;
  margin: 12px 0 20px;
  font-size: 0.9em;
}
th {
  background: var(--bg3);
  color: var(--accent);
  text-align: left;
  padding: 10px 12px;
  font-weight: 600;
  border-bottom: 2px solid var(--border);
}
td {
  padding: 8px 12px;
  border-bottom: 1px solid var(--border);
  vertical-align: top;
}
tr:hover td { background: rgba(255,255,255,0.03); }
tr.fail td  { background: rgba(239,83,80,0.08); }
tr.pass td  { background: rgba(102,187,106,0.06); }

.badge {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 3px;
  font-size: 0.75em;
  font-weight: 700;
  letter-spacing: 0.03em;
  white-space: nowrap;
}
.badge-pass     { background: var(--green);  color: #fff; }
.badge-fail     { background: var(--red);    color: #fff; }
.badge-warn     { background: var(--yellow); color: #1a1a1a; }
.badge-closed   { background: var(--gray);   color: #fff; }
.badge-open     { background: var(--orange); color: #fff; }
.badge-accepted { background: var(--blue);   color: #fff; }
.badge-info     { background: var(--bg3);    color: var(--accent); border: 1px solid var(--accent); }

.card {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 20px 24px;
  margin-bottom: 20px;
}
.card-title {
  font-size: 1.1rem;
  font-weight: 600;
  color: var(--accent);
  margin-bottom: 8px;
}

.stats-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 16px;
  margin: 20px 0;
}
.stat-card {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 20px;
  text-align: center;
}
.stat-num  { font-size: 2.2rem; font-weight: 700; color: var(--accent); line-height: 1; }
.stat-label{ font-size: 0.8rem; color: var(--text2); margin-top: 6px; text-transform: uppercase; letter-spacing: 0.05em; }

.bar-chart { margin: 20px 0; }
.bar-row {
  display: flex;
  align-items: center;
  margin-bottom: 7px;
  gap: 10px;
  font-size: 0.85em;
}
.bar-label { width: 220px; color: var(--text2); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; flex-shrink: 0; }
.bar-outer { flex: 1; background: var(--bg3); border-radius: 3px; height: 16px; overflow: hidden; }
.bar-inner { height: 100%; background: var(--accent); border-radius: 3px; min-width: 2px; }
.bar-count { width: 36px; text-align: right; color: var(--text2); flex-shrink: 0; }

.alert {
  border-left: 4px solid var(--yellow);
  background: rgba(255,167,38,0.08);
  padding: 12px 16px;
  border-radius: 0 6px 6px 0;
  margin: 12px 0;
  font-size: 0.9em;
}
.alert.info { border-color: var(--accent); background: rgba(79,195,247,0.07); }
.alert.ok   { border-color: var(--green); background: rgba(102,187,106,0.07); }
.alert.err  { border-color: var(--red); background: rgba(239,83,80,0.07); }

.meta-row { display: flex; gap: 24px; flex-wrap: wrap; margin: 8px 0 20px; }
.meta-item { font-size: 0.82em; color: var(--text2); }
.meta-item strong { color: var(--text); }

.missing { color: var(--text2); font-style: italic; }

@media (max-width: 700px) {
  nav { padding: 0 12px; }
  main { padding: 20px 14px 48px; }
  .stats-grid { grid-template-columns: repeat(2, 1fr); }
  .bar-label { width: 130px; }
}
"""

def page(title: str, body: str, active_path: str = "") -> bytes:
    nav_items = ""
    for path, label in NAV_LINKS:
        active = ' class="active"' if path == active_path else ""
        nav_items += f'<a href="{path}"{active}>{label}</a>'

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} — UltrafastSecp256k1 Audit</title>
<style>{STYLES}</style>
</head>
<body>
<nav>
  <span class="brand">&#x1F512; UltrafastSecp256k1</span>
  {nav_items}
</nav>
<main>
{body}
</main>
</body>
</html>"""
    return html.encode("utf-8")


# ---------------------------------------------------------------------------
# Markdown → HTML (server-side, no JS)
# ---------------------------------------------------------------------------

def md_to_html(text: str) -> str:
    """Convert a reasonable subset of Markdown to HTML."""
    lines = text.split("\n")
    out = []
    in_pre = False
    pre_lang = ""
    pre_buf: list[str] = []
    in_ul = False
    in_ol = False
    in_table = False
    table_buf: list[str] = []
    para_buf: list[str] = []

    def flush_para():
        nonlocal para_buf
        if para_buf:
            content = " ".join(para_buf).strip()
            if content:
                out.append(f"<p>{content}</p>")
            para_buf = []

    def flush_table():
        nonlocal table_buf, in_table
        if not table_buf:
            in_table = False
            return
        rows = table_buf
        table_buf = []
        in_table = False
        # First row = header, second row = separator, rest = data
        html = '<table>\n'
        for i, row in enumerate(rows):
            cells = [c.strip() for c in row.strip("|").split("|")]
            if i == 0:
                html += "<thead><tr>"
                for c in cells:
                    html += f"<th>{inline_md(c)}</th>"
                html += "</tr></thead>\n<tbody>\n"
            elif i == 1:
                # separator row (---|---)
                continue
            else:
                html += "<tr>"
                for c in cells:
                    html += f"<td>{inline_md(c)}</td>"
                html += "</tr>\n"
        html += "</tbody></table>"
        out.append(html)

    def flush_ul():
        nonlocal in_ul
        if in_ul:
            out.append("</ul>")
            in_ul = False

    def flush_ol():
        nonlocal in_ol
        if in_ol:
            out.append("</ol>")
            in_ol = False

    def inline_md(s: str) -> str:
        """Apply inline formatting."""
        # code spans (must come before bold/italic)
        s = re.sub(r'`([^`]+)`', r'<code>\1</code>', s)
        # bold
        s = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', s)
        # italic
        s = re.sub(r'\*(.+?)\*', r'<em>\1</em>', s)
        # links
        s = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', s)
        # status badges (inline shorthand like **CLOSED**, **OPEN**)
        s = re.sub(
            r'<strong>(PASS|FAIL|WARN|CLOSED|OPEN|ACCEPTED|OK|ERROR)</strong>',
            lambda m: badge(m.group(1)),
            s
        )
        return s

    i = 0
    while i < len(lines):
        line = lines[i]

        # Fenced code block open
        if re.match(r'^```', line) and not in_pre:
            flush_para()
            flush_ul()
            if in_table:
                flush_table()
            in_pre = True
            pre_lang = line[3:].strip()
            pre_buf = []
            i += 1
            continue

        # Fenced code block close
        if in_pre and re.match(r'^```', line):
            code_content = "\n".join(pre_buf)
            # escape HTML in code
            code_content = (code_content
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;"))
            lang_cls = f' class="language-{pre_lang}"' if pre_lang else ""
            out.append(f'<pre><code{lang_cls}>{code_content}</code></pre>')
            in_pre = False
            pre_buf = []
            i += 1
            continue

        if in_pre:
            pre_buf.append(line)
            i += 1
            continue

        # Blank line
        if line.strip() == "":
            flush_para()
            flush_ul()
            flush_ol()
            if in_table:
                flush_table()
            i += 1
            continue

        # Table rows
        if line.strip().startswith("|"):
            flush_para()
            flush_ul()
            flush_ol()
            in_table = True
            table_buf.append(line)
            i += 1
            continue
        elif in_table:
            flush_table()

        # Headings
        m = re.match(r'^(#{1,4})\s+(.*)', line)
        if m:
            flush_para()
            flush_ul()
            flush_ol()
            level = len(m.group(1))
            content = inline_md(m.group(2))
            out.append(f'<h{level}>{content}</h{level}>')
            i += 1
            continue

        # List items
        m = re.match(r'^[\-\*]\s+(.*)', line)
        if m:
            flush_para()
            flush_ol()
            if not in_ul:
                out.append("<ul>")
                in_ul = True
            out.append(f'<li>{inline_md(m.group(1))}</li>')
            i += 1
            continue

        # Ordered list items
        m = re.match(r'^\d+\.\s+(.*)', line)
        if m:
            flush_para()
            flush_ul()
            if not in_ol:
                out.append("<ol>")
                in_ol = True
            out.append(f'<li>{inline_md(m.group(1))}</li>')
            i += 1
            continue

        # Horizontal rule
        if re.match(r'^---+\s*$', line):
            flush_para()
            flush_ul()
            flush_ol()
            out.append("<hr>")
            i += 1
            continue

        # Regular text → accumulate into paragraph
        flush_ul()
        flush_ol()
        para_buf.append(inline_md(line))
        i += 1

    # Flush remaining
    flush_para()
    flush_ul()
    flush_ol()
    if in_table:
        flush_table()

    return "\n".join(out)


def badge(status: str) -> str:
    s = status.upper()
    cls_map = {
        "PASS": "pass", "OK": "pass",
        "FAIL": "fail", "ERROR": "fail",
        "WARN": "warn",
        "CLOSED": "closed",
        "OPEN": "open",
        "ACCEPTED": "accepted",
    }
    cls = cls_map.get(s, "info")
    return f'<span class="badge badge-{cls}">{s}</span>'


def read_doc(name: str) -> str | None:
    """Read a doc file, return None if missing."""
    p = DOCS / name
    if p.exists():
        return p.read_text(encoding="utf-8", errors="replace")
    return None


def doc_or_missing(name: str) -> str:
    """Return doc content HTML or a styled alert."""
    content = read_doc(name)
    if content is None:
        return (
            f'<div class="alert">'
            f'<strong>File not found:</strong> <code>docs/{name}</code>. '
            f'Run the relevant generator script first.'
            f'</div>'
        )
    return md_to_html(content)


def git_info() -> dict:
    """Return dict with commit, branch, dirty."""
    def run(cmd):
        try:
            return subprocess.check_output(
                cmd, cwd=str(LIB_ROOT), stderr=subprocess.DEVNULL
            ).decode().strip()
        except Exception:
            return ""

    commit  = run(["git", "rev-parse", "--short", "HEAD"])
    branch  = run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    status  = run(["git", "status", "--porcelain"])
    dirty   = bool(status)
    return {"commit": commit, "branch": branch, "dirty": dirty, "status": status}


# ---------------------------------------------------------------------------
# Route handlers
# ---------------------------------------------------------------------------

def render_home() -> bytes:
    # Git info
    gi = git_info()
    commit_str = gi["commit"] or "unknown"
    branch_str = gi["branch"] or "unknown"
    dirty_badge = badge("WARN") + " dirty" if gi["dirty"] else badge("PASS") + " clean"

    # CAAS KPI
    kpi_path = DOCS / "SECURITY_AUTONOMY_KPI.json"
    if kpi_path.exists():
        try:
            kpi = json.loads(kpi_path.read_text())
            score       = kpi.get("autonomy_score", "?")
            ready       = kpi.get("autonomy_ready", False)
            gates_total = kpi.get("gates_total", "?")
            gates_pass  = kpi.get("gates_passed", kpi.get("gates_total", "?"))
            kpi_ts      = kpi.get("timestamp", "")[:19].replace("T", " ")
            kpi_html = f"""
            <div class="card">
              <div class="card-title">CAAS Pipeline Status</div>
              <div class="meta-row">
                <span class="meta-item"><strong>Score:</strong> {score}/100</span>
                <span class="meta-item"><strong>Ready:</strong> {badge("PASS") if ready else badge("FAIL")}</span>
                <span class="meta-item"><strong>Gates:</strong> {gates_pass} / {gates_total} passed</span>
                <span class="meta-item"><strong>As of:</strong> {kpi_ts} UTC</span>
              </div>
              <p style="font-size:0.88em;color:var(--text2)">
                Source: <code>docs/SECURITY_AUTONOMY_KPI.json</code>
              </p>
            </div>"""
        except Exception as exc:
            kpi_html = f'<div class="alert err">Could not parse SECURITY_AUTONOMY_KPI.json: {exc}</div>'
    else:
        kpi_html = '<div class="alert">CAAS KPI not found — run <code>ci/caas_runner.py</code> first.</div>'

    # Evidence bundle
    bundle_path = DOCS / "EXTERNAL_AUDIT_BUNDLE.json"
    sha_path    = DOCS / "EXTERNAL_AUDIT_BUNDLE.sha256"
    if bundle_path.exists():
        try:
            bundle = json.loads(bundle_path.read_text())
            evidence = bundle.get("evidence", [])
            total_e = len(evidence)
            missing_e = sum(1 for e in evidence if not e.get("exists", True))
            refresh_ts = bundle.get("generated_at", bundle.get("timestamp", ""))[:19].replace("T", " ")
            sha_digest = sha_path.read_text().strip() if sha_path.exists() else "—"
            bundle_status = badge("PASS") if missing_e == 0 else badge("WARN")
            bundle_html = f"""
            <div class="card">
              <div class="card-title">Evidence Bundle {bundle_status}</div>
              <div class="meta-row">
                <span class="meta-item"><strong>Artifacts:</strong> {total_e}</span>
                <span class="meta-item"><strong>Missing:</strong> {missing_e}</span>
                <span class="meta-item"><strong>Last refresh:</strong> {refresh_ts} UTC</span>
              </div>
              <p style="font-size:0.82em;color:var(--text2)">
                SHA-256: <code>{sha_digest[:64]}</code>
                &mdash; <a href="/bundle">View bundle details</a>
              </p>
            </div>"""
        except Exception as exc:
            bundle_html = f'<div class="alert err">Could not parse bundle: {exc}</div>'
    else:
        bundle_html = '<div class="alert">Evidence bundle not found — run <code>ci/build_owner_audit_bundle.py</code>.</div>'

    # Quick stats
    exploit_count = len(list((LIB_ROOT / "audit").glob("test_exploit_*.cpp"))) if (LIB_ROOT / "audit").exists() else 0
    ci_count = len(list((LIB_ROOT / ".github" / "workflows").glob("*.yml"))) if (LIB_ROOT / ".github" / "workflows").exists() else 0
    # Count residual risks from the register
    rr_content = read_doc("RESIDUAL_RISK_REGISTER.md") or ""
    rr_count = len(re.findall(r'\bRR-\d{3}\b', rr_content))

    stats_html = f"""
    <div class="stats-grid">
      <div class="stat-card">
        <div class="stat-num">{exploit_count}</div>
        <div class="stat-label">Exploit PoCs</div>
      </div>
      <div class="stat-card">
        <div class="stat-num">{ci_count}</div>
        <div class="stat-label">CI Workflows</div>
      </div>
      <div class="stat-card">
        <div class="stat-num">{rr_count}</div>
        <div class="stat-label">Residual Risks</div>
      </div>
    </div>"""

    # Navigation cards
    nav_cards = ""
    page_descs = [
        ("/claims",       "Security Claims",      "Formal security claims made by the library."),
        ("/exploits",     "Exploit PoC Catalog",  "All exploit PoC tests by attack category."),
        ("/residuals",    "Residual Risks",        "Known open risks and their acceptance status."),
        ("/ct",           "CT Evidence",           "Constant-time verification evidence."),
        ("/parity",       "Parity Status",         "CPU vs GPU backend parity matrix."),
        ("/bundle",       "Evidence Bundle",       "External audit bundle artifact listing."),
        ("/graph",        "Source Graph",          "Symbol and file statistics from project graph."),
        ("/traceability", "Traceability",          "Claim → test → CI linkage matrix."),
        ("/faq",          "FAQ",                   "Frequently asked questions about the audit."),
    ]
    nav_cards += '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:14px;margin-top:24px;">'
    for href, title, desc in page_descs:
        nav_cards += f"""
        <a href="{href}" style="text-decoration:none;">
          <div class="card" style="margin:0;cursor:pointer;transition:border-color 0.15s;"
               onmouseover="this.style.borderColor='#4fc3f7'" onmouseout="this.style.borderColor=''">
            <div class="card-title" style="margin-bottom:6px;">{title}</div>
            <div style="font-size:0.85em;color:var(--text2);">{desc}</div>
          </div>
        </a>"""
    nav_cards += '</div>'

    body = f"""
    <h1>Audit Dashboard</h1>
    <div class="meta-row">
      <span class="meta-item"><strong>Commit:</strong> <code>{commit_str}</code></span>
      <span class="meta-item"><strong>Branch:</strong> <code>{branch_str}</code></span>
      <span class="meta-item"><strong>Working tree:</strong> {dirty_badge}</span>
      <span class="meta-item"><strong>Library root:</strong> <code>{LIB_ROOT}</code></span>
      <span class="meta-item"><strong>Viewed:</strong> {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")} UTC</span>
    </div>
    {kpi_html}
    {bundle_html}
    <h2>Quick Stats</h2>
    {stats_html}
    <h2>Browse Audit Artifacts</h2>
    {nav_cards}
    """
    return page("Dashboard", body, "/")


def render_claims() -> bytes:
    body = "<h1>Security Claims</h1>\n" + doc_or_missing("SECURITY_CLAIMS.md")
    return page("Security Claims", body, "/claims")


def render_exploits() -> bytes:
    content = read_doc("EXPLOIT_TEST_CATALOG.md")
    if content is None:
        body = "<h1>Exploit PoC Catalog</h1>" + doc_or_missing("EXPLOIT_TEST_CATALOG.md")
        return page("Exploits", body, "/exploits")

    # Parse category distribution from the Summary table
    categories: dict[str, int] = {}
    in_table = False
    for line in content.split("\n"):
        if "| Category |" in line or "| Category|" in line:
            in_table = True
            continue
        if in_table:
            if line.strip().startswith("|---"):
                continue
            if not line.strip().startswith("|"):
                break
            cols = [c.strip() for c in line.strip("|").split("|")]
            if len(cols) >= 2:
                cat = cols[0]
                # file count is second column
                try:
                    count = int(cols[1])
                except ValueError:
                    count = 1
                if cat and cat != "Category":
                    categories[cat] = count

    max_count = max(categories.values(), default=1)

    chart_html = ""
    if categories:
        chart_html = '<h2>Attack Class Distribution</h2><div class="bar-chart">'
        for cat, cnt in sorted(categories.items(), key=lambda x: -x[1]):
            pct = int(cnt / max_count * 100)
            chart_html += f"""
            <div class="bar-row">
              <span class="bar-label" title="{cat}">{cat}</span>
              <div class="bar-outer"><div class="bar-inner" style="width:{pct}%"></div></div>
              <span class="bar-count">{cnt}</span>
            </div>"""
        chart_html += "</div>"

    body = "<h1>Exploit PoC Catalog</h1>\n" + chart_html + md_to_html(content)
    return page("Exploits", body, "/exploits")


def render_residuals() -> bytes:
    content = read_doc("RESIDUAL_RISK_REGISTER.md")
    if content is None:
        body = "<h1>Residual Risk Register</h1>" + doc_or_missing("RESIDUAL_RISK_REGISTER.md")
        return page("Residuals", body, "/residuals")

    # Post-process: colorize status cells in the rendered table
    html = md_to_html(content)

    # Replace status words in table cells with badges
    def badge_replace(m):
        full = m.group(0)
        text = m.group(1)
        upper = text.upper()
        if "CLOSED" in upper:
            return f'<td>{badge("CLOSED")}</td>'
        if "OPEN" in upper:
            return f'<td>{badge("OPEN")}</td>'
        if "ACCEPTED" in upper or "INTENTIONALLY DEFERRED" in upper or "INTENTIONAL" in upper:
            return f'<td>{badge("ACCEPTED")} {text}</td>'
        if "OUT-OF-SCOPE" in upper:
            return f'<td>{badge("INFO")} {text}</td>'
        return full

    html = re.sub(r'<td>((?:Accepted|Intentionally deferred|Out-of-scope|CLOSED|OPEN)[^<]*)</td>',
                  badge_replace, html, flags=re.IGNORECASE)

    body = "<h1>Residual Risk Register</h1>\n" + html
    return page("Residuals", body, "/residuals")


def render_ct() -> bytes:
    body = "<h1>CT Verification Evidence</h1>\n" + doc_or_missing("CT_VERIFICATION.md")
    return page("CT Evidence", body, "/ct")


def render_parity() -> bytes:
    content = read_doc("BACKEND_PARITY.md")
    if content is None:
        body = "<h1>Shim Parity Status</h1>" + doc_or_missing("BACKEND_PARITY.md")
        return page("Parity", body, "/parity")

    html = md_to_html(content)

    # Highlight PASS/FAIL in table cells with colored badges
    html = re.sub(r'\bPASS\b', badge("PASS"), html)
    html = re.sub(r'\bFAIL\b', badge("FAIL"), html)
    html = re.sub(r'\bWARN\b', badge("WARN"), html)
    html = re.sub(r'\bN/A\b',  '<span class="badge badge-info">N/A</span>', html)

    # Add row-level coloring via JS-free approach: wrap FAIL rows
    # (We already emit badge markup; the table cell background is set via CSS .fail class)
    # Inject row classes by post-processing <tr>
    lines = html.split("\n")
    new_lines = []
    for line in lines:
        if "<tr>" in line or line.strip() == "<tr>":
            if 'badge-fail' in line:
                line = line.replace("<tr>", '<tr class="fail">', 1)
            elif 'badge-pass' in line:
                line = line.replace("<tr>", '<tr class="pass">', 1)
        new_lines.append(line)
    html = "\n".join(new_lines)

    body = "<h1>Shim Parity Status</h1>\n" + html
    return page("Parity", body, "/parity")


def render_bundle() -> bytes:
    bundle_path = DOCS / "EXTERNAL_AUDIT_BUNDLE.json"
    sha_path    = DOCS / "EXTERNAL_AUDIT_BUNDLE.sha256"

    if not bundle_path.exists():
        body = """<h1>Evidence Bundle</h1>
        <div class="alert">
          <strong>Bundle not found.</strong> Run
          <code>python3 ci/build_owner_audit_bundle.py</code> to generate it.
        </div>"""
        return page("Bundle", body, "/bundle")

    try:
        bundle = json.loads(bundle_path.read_text())
    except Exception as exc:
        body = f"<h1>Evidence Bundle</h1><div class='alert err'>Parse error: {exc}</div>"
        return page("Bundle", body, "/bundle")

    sha_digest = sha_path.read_text().strip() if sha_path.exists() else "—"
    refresh_ts = bundle.get("generated_at", bundle.get("timestamp", "unknown"))[:19].replace("T", " ")

    evidence = bundle.get("evidence", [])

    # VIZ-2 fix: an empty evidence list means the bundle is malformed or
    # was generated with no artifacts — treat as a FAIL rather than PASS.
    if len(evidence) == 0:
        body = """<h1>Evidence Bundle</h1>
        <div class="alert err">
          <strong>Bundle contains zero evidence items</strong> — malformed or empty bundle.
          <br>Status: <span style="color:var(--red);font-weight:bold">FAIL</span>
          <br>Regenerate with <code>python3 ci/build_owner_audit_bundle.py</code>.
        </div>"""
        return page("Bundle", body, "/bundle")

    # Verify each artifact
    rows_html = ""
    pass_count = fail_count = warn_count = 0
    for art in evidence:
        rel_path = art.get("path", "")
        expected_hash = art.get("sha256", "")
        recorded_exists = art.get("exists", True)
        recorded_size   = art.get("size", 0)

        abs_path = LIB_ROOT / rel_path
        actual_exists = abs_path.exists()

        if actual_exists and expected_hash:
            try:
                data = abs_path.read_bytes()
                actual_hash = hashlib.sha256(data).hexdigest()
                hash_ok = actual_hash == expected_hash
            except Exception:
                actual_hash = ""
                hash_ok = False
        else:
            actual_hash = ""
            hash_ok = False

        if actual_exists and hash_ok:
            status_b = badge("PASS")
            pass_count += 1
            row_cls = "pass"
        elif actual_exists and not hash_ok:
            status_b = badge("WARN") + " hash mismatch"
            warn_count += 1
            row_cls = ""
        else:
            status_b = badge("FAIL") + " missing"
            fail_count += 1
            row_cls = "fail"

        size_kb = f"{recorded_size/1024:.1f} KB" if recorded_size else "—"
        rows_html += f"""
        <tr class="{row_cls}">
          <td><code>{rel_path}</code></td>
          <td>{status_b}</td>
          <td><code style="font-size:0.78em">{expected_hash[:16]}…</code></td>
          <td>{size_kb}</td>
        </tr>"""

    overall = badge("PASS") if fail_count == 0 and warn_count == 0 else (
              badge("WARN") if fail_count == 0 else badge("FAIL"))

    body = f"""
    <h1>Evidence Bundle</h1>
    <div class="card">
      <div class="meta-row">
        <span class="meta-item"><strong>Overall:</strong> {overall}</span>
        <span class="meta-item"><strong>Artifacts:</strong> {len(evidence)}</span>
        <span class="meta-item">{badge("PASS")} {pass_count} pass</span>
        <span class="meta-item">{badge("WARN")} {warn_count} warn</span>
        <span class="meta-item">{badge("FAIL")} {fail_count} fail</span>
        <span class="meta-item"><strong>Generated:</strong> {refresh_ts} UTC</span>
      </div>
      <p style="font-size:0.82em;color:var(--text2)">
        Bundle SHA-256: <code>{sha_digest}</code>
      </p>
    </div>
    <table>
      <thead><tr><th>Path</th><th>Status</th><th>Expected SHA-256</th><th>Size</th></tr></thead>
      <tbody>{rows_html}</tbody>
    </table>
    """
    return page("Bundle", body, "/bundle")


def render_graph() -> bytes:
    # Fallback to legacy .project_graph.db if the primary source_graph.db is absent.
    _db = GRAPH_DB if GRAPH_DB.exists() else (_LEGACY_GRAPH_DB if _LEGACY_GRAPH_DB.exists() else None)
    if _db is None:
        body = f"""
        <h1>Source Graph Stats</h1>
        <div class="alert">
          <strong>Graph DB not found</strong> at <code>{GRAPH_DB}</code>.
          Run <code>python3 tools/source_graph_kit/source_graph.py build -i</code> to build it.
        </div>"""
        return page("Graph", body, "/graph")

    try:
        con = sqlite3.connect(str(_db))
        cur = con.cursor()

        def count(table, where=""):
            try:
                q = f"SELECT COUNT(*) FROM {table}"
                if where:
                    q += f" WHERE {where}"
                cur.execute(q)
                return cur.fetchone()[0]
            except Exception:
                return "—"

        stats = {
            "Source Files":       count("source_files"),
            "C ABI Functions":    count("c_abi_functions"),
            "C++ Methods":        count("cpp_methods"),
            "Function Index":     count("function_index"),
            "Audit Modules":      count("audit_modules"),
            "CI Workflows":       count("ci_workflows"),
            "Security Patterns":  count("security_patterns"),
            "Call Edges":         count("call_edges"),
            "Symbols w/ Security": count("symbol_security"),
            "Symbols w/ Perf":    count("symbol_performance"),
        }

        stat_cards = ""
        for label, val in stats.items():
            stat_cards += f"""
            <div class="stat-card">
              <div class="stat-num">{val}</div>
              <div class="stat-label">{label}</div>
            </div>"""

        # Top functions by hotspot score
        top_funcs_html = ""
        try:
            cur.execute("""
                SELECT fi.name, fi.file_path, hs.score
                FROM function_index fi
                JOIN hotspot_scores hs ON fi.id = hs.function_id
                ORDER BY hs.score DESC LIMIT 20
            """)
            rows = cur.fetchall()
            if rows:
                top_funcs_html = """
                <h2>Top 20 Hot Functions</h2>
                <table>
                  <thead><tr><th>Function</th><th>File</th><th>Hotspot Score</th></tr></thead>
                  <tbody>"""
                for name, fpath, score in rows:
                    short_path = fpath.split("/")[-2] + "/" + fpath.split("/")[-1] if fpath else ""
                    top_funcs_html += f"""
                    <tr>
                      <td><code>{name}</code></td>
                      <td><code style="font-size:0.82em">{short_path}</code></td>
                      <td>{score:.2f}</td>
                    </tr>"""
                top_funcs_html += "</tbody></table>"
        except Exception:
            pass

        # Audit module list
        audit_html = ""
        try:
            cur.execute("SELECT name, section, advisory FROM audit_modules ORDER BY section, name LIMIT 60")
            rows = cur.fetchall()
            if rows:
                audit_html = """
                <h2>Registered Audit Modules (first 60)</h2>
                <table>
                  <thead><tr><th>Module</th><th>Section</th><th>Advisory</th></tr></thead>
                  <tbody>"""
                for name, section, advisory in rows:
                    adv_b = badge("WARN") if advisory else badge("PASS")
                    audit_html += f"<tr><td><code>{name}</code></td><td>{section}</td><td>{adv_b}</td></tr>"
                audit_html += "</tbody></table>"
        except Exception:
            pass

        # Meta info
        meta_html = ""
        try:
            cur.execute("SELECT key, value FROM meta ORDER BY key")
            meta_rows = cur.fetchall()
            if meta_rows:
                meta_html = '<h2>Graph Metadata</h2><table><thead><tr><th>Key</th><th>Value</th></tr></thead><tbody>'
                for k, v in meta_rows:
                    meta_html += f"<tr><td><code>{k}</code></td><td>{v}</td></tr>"
                meta_html += "</tbody></table>"
        except Exception:
            pass

        con.close()

        body = f"""
        <h1>Source Graph Stats</h1>
        <div class="alert info">
          DB: <code>{_db}</code>
        </div>
        <div class="stats-grid">{stat_cards}</div>
        {top_funcs_html}
        {audit_html}
        {meta_html}
        """

    except Exception as exc:
        body = f"<h1>Source Graph Stats</h1><div class='alert err'>DB error: {exc}</div>"

    return page("Graph", body, "/graph")


def render_traceability() -> bytes:
    content = read_doc("AUDIT_TRACEABILITY.md")
    if content is None:
        body = "<h1>Claim → Test → CI Traceability</h1>" + doc_or_missing("AUDIT_TRACEABILITY.md")
        return page("Traceability", body, "/traceability")

    html = md_to_html(content)
    html = re.sub(r'\bPASS\b', badge("PASS"), html)
    html = re.sub(r'\bFAIL\b', badge("FAIL"), html)
    html = re.sub(r'\bWARN\b', badge("WARN"), html)

    body = "<h1>Claim → Test → CI Traceability</h1>\n" + html
    return page("Traceability", body, "/traceability")


def render_faq() -> bytes:
    body = "<h1>CAAS FAQ</h1>\n" + doc_or_missing("CAAS_FAQ.md")
    return page("FAQ", body, "/faq")


def render_404(path: str) -> bytes:
    body = f"""
    <h1>404 — Not Found</h1>
    <p>No route for <code>{path}</code>.</p>
    <p><a href="/">Back to Dashboard</a></p>
    """
    return page("404", body)


# ---------------------------------------------------------------------------
# HTTP Handler
# ---------------------------------------------------------------------------

ROUTES: dict[str, "function"] = {
    "/":             render_home,
    "/claims":       render_claims,
    "/exploits":     render_exploits,
    "/residuals":    render_residuals,
    "/ct":           render_ct,
    "/parity":       render_parity,
    "/bundle":       render_bundle,
    "/graph":        render_graph,
    "/traceability": render_traceability,
    "/faq":          render_faq,
}


class AuditHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        # Use a compact format with timestamp
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"  [{ts}] {fmt % args}", file=sys.stderr)

    def do_GET(self):
        path = self.path.split("?")[0].rstrip("/") or "/"

        # Serve .md / .json / .txt files directly from docs/ or lib root
        if path != "/" and not ROUTES.get(path):
            # Strip leading slash, prevent path traversal
            rel = path.lstrip("/")
            if ".." not in rel:
                for base in (DOCS, LIB_ROOT):
                    candidate = base / rel
                    if candidate.is_file() and candidate.suffix in (".md", ".json", ".txt", ".sha256", ".log"):
                        raw = candidate.read_text(errors="replace")
                        if candidate.suffix == ".md":
                            body = f"<article style='max-width:900px;margin:0 auto'>{md_to_html(raw)}</article>"
                            content = page(candidate.name, body)
                            mime = "text/html; charset=utf-8"
                        else:
                            raw_bytes = raw.encode()
                            self.send_response(200)
                            self.send_header("Content-Type", "text/plain; charset=utf-8")
                            self.send_header("Content-Length", str(len(raw_bytes)))
                            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                            self.end_headers()
                            self.wfile.write(raw_bytes)
                            return
                        content_bytes = content if isinstance(content, bytes) else content.encode()
                        self.send_response(200)
                        self.send_header("Content-Type", mime)
                        self.send_header("Content-Length", str(len(content_bytes)))
                        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                        self.end_headers()
                        self.wfile.write(content_bytes)
                        return

        handler = ROUTES.get(path)

        if handler is None:
            content = render_404(path)
            self.send_response(404)
        else:
            try:
                content = handler()
                self.send_response(200)
            except Exception as exc:
                import traceback
                tb = traceback.format_exc()
                err_body = f"""
                <h1>Internal Error</h1>
                <p>Route <code>{path}</code> raised an exception:</p>
                <pre><code>{tb}</code></pre>
                <p><a href="/">Back to Dashboard</a></p>
                """
                content = page("Error", err_body)
                self.send_response(500)

        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(content)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="UltrafastSecp256k1 Audit Viewer — local read-only web UI"
    )
    parser.add_argument("--port", type=int, default=8765, help="TCP port (default: 8765)")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    args = parser.parse_args()

    print(f"Audit Viewer  →  http://{args.host}:{args.port}")
    print(f"Library root  →  {LIB_ROOT}")
    print("Press Ctrl+C to stop.")
    print()

    server = HTTPServer((args.host, args.port), AuditHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
