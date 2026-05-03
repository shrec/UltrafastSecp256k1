#!/usr/bin/env python3
"""
caas_dashboard.py
=================
CAAS Audit Dashboard — gate status + artifact navigator.

Usage:
    python3 ci/caas_dashboard.py [-o output.html] [--open]
"""

from __future__ import annotations

import argparse
import datetime
import html
import json
import os
import re
import subprocess
import sys
import webbrowser
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
LIB_ROOT   = SCRIPT_DIR.parent

# ---------------------------------------------------------------------------
# Data collection — reads existing on-disk evidence, live fallback
# ---------------------------------------------------------------------------

def _run(cmd: list[str]) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=LIB_ROOT, timeout=15)
        return r.stdout.strip()
    except Exception:
        return ""


def _load_json(path: Path) -> dict | list | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def collect_git() -> dict:
    return {
        "short":  _run(["git", "rev-parse", "--short=8", "HEAD"]),
        "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty":  bool(_run(["git", "status", "--short"])),
        "date":   _run(["git", "log", "-1", "--format=%ci"]),
        "message":_run(["git", "log", "-1", "--format=%s"]),
    }


# Evidence files in priority order (CI artifacts land here)
_GATE_EVIDENCE = {
    "scanner":  ["caas_scanner.json", "stage1a_scanner.json"],
    "gate":     ["caas_audit_gate.json", "stage2_audit_gate.json", "audit_report.json"],
    "autonomy": ["caas_autonomy.json", "stage3_autonomy.json", "docs/SECURITY_AUTONOMY_KPI.json"],
    "bundle":   ["docs/EXTERNAL_AUDIT_BUNDLE.json"],
}


def _find_evidence(candidates: list[str]) -> tuple[dict | None, str]:
    for name in candidates:
        p = LIB_ROOT / name
        if p.exists():
            d = _load_json(p)
            if d is not None:
                return d, name
    return None, ""


def collect_gates() -> list[dict]:
    gates = []

    # Stage 1 — scanner
    d, src = _find_evidence(_GATE_EVIDENCE["scanner"])
    if d:
        n = d.get("total_findings", 0)
        gates.append({"name": "Static Analysis", "verdict": "PASS" if n == 0 else "FAIL",
                      "detail": f"{n} finding(s)", "source": src})
    else:
        # Live
        try:
            r = subprocess.run(["python3", "ci/audit_test_quality_scanner.py", "--json"],
                               capture_output=True, text=True, cwd=LIB_ROOT, timeout=30)
            d2 = json.loads(r.stdout) if r.stdout.strip() else {}
            n = d2.get("total_findings", 0)
            gates.append({"name": "Static Analysis", "verdict": "PASS" if n == 0 else "FAIL",
                          "detail": f"{n} finding(s)", "source": "live"})
        except Exception:
            gates.append({"name": "Static Analysis", "verdict": "UNKNOWN", "detail": "no evidence", "source": ""})

    # Stage 2 — audit gate
    d, src = _find_evidence(_GATE_EVIDENCE["gate"])
    if d:
        verdict = d.get("audit_verdict") or d.get("verdict") or ("PASS" if d.get("all_passed") else "?")
        summary = d.get("summary", {})
        failed  = summary.get("failed", summary.get("failing", 0))
        gates.append({"name": "Audit Gate", "verdict": str(verdict).upper(),
                      "detail": f"{failed} failing check(s)", "source": src})
    else:
        gates.append({"name": "Audit Gate", "verdict": "UNKNOWN", "detail": "no evidence", "source": ""})

    # Stage 3 — autonomy
    d, src = _find_evidence(_GATE_EVIDENCE["autonomy"])
    if d:
        score = d.get("autonomy_score", "?")
        ready = d.get("autonomy_ready", False)
        v = "PASS" if ready else "FAIL"
        gates.append({"name": "Security Autonomy", "verdict": v,
                      "detail": f"{score}/100", "source": src})
    else:
        gates.append({"name": "Security Autonomy", "verdict": "UNKNOWN", "detail": "no evidence", "source": ""})

    # Stage 4 — bundle
    d, src = _find_evidence(_GATE_EVIDENCE["bundle"])
    if d:
        commit = d.get("git", {}).get("commit", "?")[:8]
        gates.append({"name": "Audit Bundle", "verdict": "PASS",
                      "detail": f"commit {commit}", "source": src})
    else:
        gates.append({"name": "Audit Bundle", "verdict": "UNKNOWN", "detail": "no bundle", "source": ""})

    return gates


def collect_artifacts() -> list[dict]:
    now = datetime.datetime.now(datetime.timezone.utc).timestamp()
    items = []
    for root in [LIB_ROOT / "docs",
                 LIB_ROOT / "audit-output",
                 LIB_ROOT / "build" / "owner_audit"]:
        if not root.exists():
            continue
        for p in sorted(root.rglob("*")):
            if not p.is_file():
                continue
            if p.suffix not in (".json", ".md", ".txt", ".sarif"):
                continue
            try:
                st = p.stat()
            except OSError:
                continue
            age_h = (now - st.st_mtime) / 3600
            items.append({
                "name":      p.name,
                "path":      str(p.relative_to(LIB_ROOT)),
                "type":      p.suffix.lstrip(".").upper(),
                "size":      st.st_size,
                "freshness": "FRESH" if age_h < 24 else "RECENT" if age_h < 168 else "STALE",
                "age_h":     round(age_h, 1),
            })
    return items


def collect_limitations() -> list[str]:
    rr = LIB_ROOT / "docs" / "RESIDUAL_RISK_REGISTER.md"
    if not rr.exists():
        return []
    out = []
    for ln in rr.read_text(errors="replace").splitlines():
        if not ln.startswith("| RR-"):
            continue
        cols = [c.strip() for c in ln.split("|")]
        if len(cols) < 5:
            continue
        rr_id, klass, status = cols[1], cols[2], cols[3]
        if any(t in status.upper() for t in ("CLOSED", "RESOLVED", "FIXED")):
            continue
        out.append(f"{rr_id} — {klass} ({status})")
    return out


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

CSS = """
:root{--bg:#0d1117;--bg2:#161b22;--bg3:#21262d;--border:#30363d;--text:#e6edf3;
--text2:#8b949e;--green:#3fb950;--red:#f85149;--yellow:#d29922;--blue:#58a6ff;
--cyan:#39d353;}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',system-ui,sans-serif;background:var(--bg);color:var(--text);
font-size:14px;line-height:1.6}
a{color:var(--blue);text-decoration:none}a:hover{text-decoration:underline}
h2{font-size:1.05rem;font-weight:600;margin:1.4rem 0 .6rem;color:var(--blue);
border-bottom:1px solid var(--border);padding-bottom:.3rem}
.header{background:var(--bg2);border-bottom:1px solid var(--border);
padding:1rem 2rem;display:flex;align-items:center;gap:1rem;flex-wrap:wrap}
.header-title{flex:1}
.header-meta{font-size:.78rem;color:var(--text2);margin-top:.2rem}
.badge{display:inline-flex;align-items:center;padding:.15rem .5rem;
border-radius:5px;font-size:.7rem;font-weight:700;letter-spacing:.04em;text-transform:uppercase}
.pass{background:#1a3a22;color:var(--green);border:1px solid #2ea043}
.fail{background:#3a1a1a;color:var(--red);border:1px solid #da3633}
.warn{background:#3a2a0a;color:var(--yellow);border:1px solid #9e6a03}
.unknown{background:#21262d;color:var(--text2);border:1px solid var(--border)}
.dirty{background:#3a2a0a;color:var(--yellow);border:1px solid #9e6a03}
.clean{background:#1a3a22;color:var(--green);border:1px solid #2ea043}
.main{max-width:1100px;margin:0 auto;padding:1.2rem 2rem}
.card{background:var(--bg2);border:1px solid var(--border);border-radius:7px;padding:.9rem 1.1rem;margin-bottom:1rem}
nav{position:sticky;top:0;z-index:100;background:var(--bg2);
border-bottom:1px solid var(--border);padding:.4rem 2rem;
display:flex;gap:.4rem;flex-wrap:wrap;font-size:.78rem}
nav a{color:var(--text2);padding:.2rem .45rem;border-radius:4px}
nav a:hover{background:var(--bg3);color:var(--text)}
table{width:100%;border-collapse:collapse;font-size:.82rem}
th{background:var(--bg3);color:var(--text2);font-weight:600;text-align:left;
padding:.4rem .65rem;border-bottom:1px solid var(--border);font-size:.72rem;letter-spacing:.04em}
td{padding:.35rem .65rem;border-bottom:1px solid #21262d;vertical-align:top}
tr:hover td{background:#1c2128}
code{background:var(--bg3);border:1px solid var(--border);border-radius:3px;
padding:.1em .35em;font-size:.82em;font-family:monospace}
#search-box{width:100%;padding:.4em .6em;margin-bottom:.6em;font-size:.9em;
background:var(--bg3);border:1px solid var(--border);border-radius:5px;
color:var(--text);outline:none}
#search-box:focus{border-color:var(--blue)}
.art-fresh{padding:1px 5px;border-radius:3px;font-size:.68em;font-weight:700}
.art-FRESH{background:#1a3a22;color:var(--green);border:1px solid #2ea043}
.art-RECENT{background:#3a2a0a;color:var(--yellow);border:1px solid #9e6a03}
.art-STALE{background:#3a1a1a;color:var(--red);border:1px solid #da3633}
.lim-list{list-style:none;padding:0}
.lim-list li{padding:.3rem 0;border-bottom:1px solid var(--border);color:var(--text2);font-size:.82rem}
footer{text-align:center;padding:1.5rem;color:var(--text2);font-size:.72rem;
border-top:1px solid var(--border);margin-top:1.5rem}
"""


def _badge(verdict: str) -> str:
    v = verdict.upper()
    cls = {"PASS": "pass", "FAIL": "fail", "WARN": "warn"}.get(v, "unknown")
    icon = {"pass": "✓", "fail": "✗", "warn": "⚠"}.get(cls, "·")
    return f'<span class="badge {cls}">{icon} {v}</span>'


def render_html(git: dict, gates: list[dict], artifacts: list[dict], lims: list[str]) -> str:
    generated = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    dirty_badge = '<span class="badge dirty">⚠ DIRTY</span>' if git.get("dirty") else '<span class="badge clean">✓ CLEAN</span>'

    overall_fail = any(g["verdict"] == "FAIL" for g in gates)
    overall_unknown = any(g["verdict"] == "UNKNOWN" for g in gates)
    if overall_fail:
        overall = '<span class="badge fail" style="font-size:.9rem;padding:.3rem .8rem">✗ BLOCKED</span>'
    elif overall_unknown:
        overall = '<span class="badge unknown" style="font-size:.9rem;padding:.3rem .8rem">· UNKNOWN</span>'
    else:
        overall = '<span class="badge pass" style="font-size:.9rem;padding:.3rem .8rem">✓ AUDIT-READY</span>'

    # Gates table
    gate_rows = ""
    for g in gates:
        src_html = f'<code style="font-size:.72em;color:var(--text2)">{html.escape(g["source"])}</code>' if g["source"] else ""
        gate_rows += (
            f'<tr><td>{html.escape(g["name"])}</td>'
            f'<td>{_badge(g["verdict"])}</td>'
            f'<td style="color:var(--text2)">{html.escape(g["detail"])}</td>'
            f'<td>{src_html}</td></tr>'
        )

    # Artifacts table
    art_rows = ""
    for a in artifacts:
        f = a["freshness"]
        age = f'{a["age_h"]}h' if isinstance(a["age_h"], (int, float)) else "?"
        art_rows += (
            f'<tr class="art-row">'
            f'<td><a href="http://localhost:8080/artifacts/{html.escape(a["path"])}">'
            f'{html.escape(a["name"])}</a></td>'
            f'<td style="color:var(--text2);font-size:.75em">{html.escape(a["path"])}</td>'
            f'<td style="color:var(--text2)">{a["type"]}</td>'
            f'<td style="text-align:right;color:var(--text2)">{a["size"]:,}</td>'
            f'<td><span class="art-fresh art-{f}">{f}</span> '
            f'<span style="color:var(--text2);font-size:.75em">{age}</span></td>'
            f'</tr>'
        )
    if not art_rows:
        art_rows = '<tr><td colspan="5" style="color:var(--text2)">No artifacts found. Run the CAAS pipeline to generate evidence.</td></tr>'

    # Limitations
    lim_items = "".join(f"<li>{html.escape(l)}</li>" for l in lims) if lims else '<li style="color:var(--text2)">None recorded.</li>'

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CAAS · {html.escape(git.get('short','?'))} · {'DIRTY' if git.get('dirty') else 'CLEAN'}</title>
<style>{CSS}</style>
</head>
<body>
<header class="header">
  <div class="header-title">
    <strong>CAAS Audit Dashboard</strong> &nbsp;·&nbsp;
    <code>{html.escape(git.get('short','?'))}</code> &nbsp;·&nbsp;
    <code>{html.escape(git.get('branch','?'))}</code> &nbsp;
    {dirty_badge}
    <div class="header-meta">{html.escape(git.get('message','')[:100])} &nbsp;·&nbsp; {html.escape(git.get('date','')[:19])}</div>
  </div>
  <div>{overall}</div>
</header>
<nav>
  <a href="#gates">Gates</a>
  <a href="#artifacts">Artifacts</a>
  <a href="#limitations">Limitations</a>
  <a href="http://localhost:8080/artifacts/" target="_blank">Browse →</a>
</nav>
<main class="main">

<section id="gates">
<h2>Gate Status</h2>
<div class="card">
<table>
<thead><tr><th>Stage</th><th>Verdict</th><th>Detail</th><th>Evidence file</th></tr></thead>
<tbody>{gate_rows}</tbody>
</table>
</div>
</section>

<section id="artifacts">
<h2>Evidence Artifacts</h2>
<input id="search-box" type="search" placeholder="Filter artifacts…" oninput="filterArts(this.value)">
<div class="card" style="padding:0;overflow:hidden">
<table>
<thead><tr><th>File</th><th>Path</th><th>Type</th><th style="text-align:right">Bytes</th><th>Freshness</th></tr></thead>
<tbody id="art-tbody">{art_rows}</tbody>
</table>
</div>
<p style="font-size:.75em;color:var(--text2);margin-top:.4rem">
  Links open in <code>caas_serve.py</code> panel (localhost:8080). Run <code>python3 ci/caas_serve.py --lan</code> to start.
</p>
</section>

<section id="limitations">
<h2>Residual Risks</h2>
<div class="card">
<ul class="lim-list">{lim_items}</ul>
</div>
</section>

</main>
<footer>Generated {generated}</footer>
<script>
function filterArts(q) {{
  q = q.toLowerCase();
  document.querySelectorAll('#art-tbody .art-row').forEach(function(r) {{
    r.style.display = (!q || r.textContent.toLowerCase().includes(q)) ? '' : 'none';
  }});
}}
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Generate CAAS audit dashboard")
    ap.add_argument("-o", "--output", default="caas_dashboard.html")
    ap.add_argument("--open", action="store_true")
    ap.add_argument("--serve", type=int, metavar="PORT")
    args = ap.parse_args()

    os.chdir(LIB_ROOT)

    print("Collecting audit data...", flush=True)
    git       = collect_git()
    gates     = collect_gates()
    artifacts = collect_artifacts()
    lims      = collect_limitations()

    print("Rendering...", flush=True)
    page = render_html(git, gates, artifacts, lims)

    if args.serve:
        import http.server
        body = page.encode("utf-8")
        class H(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", len(body))
                self.end_headers()
                self.wfile.write(body)
            def log_message(self, *a): pass
        srv = http.server.HTTPServer(("localhost", args.serve), H)
        print(f"\nDashboard: http://localhost:{args.serve}/  (Ctrl-C to stop)\n")
        webbrowser.open(f"http://localhost:{args.serve}/")
        srv.serve_forever()
    else:
        out = Path(args.output)
        out.write_text(page, encoding="utf-8")
        print(f"\nDashboard: {out}  ({out.stat().st_size // 1024} KB)\n")
        if args.open:
            webbrowser.open(out.resolve().as_uri())
    return 0


if __name__ == "__main__":
    sys.exit(main())
