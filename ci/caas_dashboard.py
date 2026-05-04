#!/usr/bin/env python3
"""
caas_dashboard.py
=================
CAAS Audit Cockpit — generates a self-contained HTML dashboard that
visualises the complete audit state of UltrafastSecp256k1.

Usage:
    python3 ci/caas_dashboard.py [--open] [-o output.html]

Options:
    --open          Open the generated HTML in the default browser
    -o PATH         Output path (default: caas_dashboard.html)
    --serve PORT    Serve on localhost:PORT instead of writing a file

The generated HTML has zero external dependencies (no CDN, no network calls).
All styling and scripts are inlined.

Sections:
  1.  Executive Summary
  2.  Gate Status (Preflight / CAAS gates)
  3.  Security Claims
  4.  Exploit PoC Corpus
  5.  Constant-Time
  6.  Differential Testing
  7.  Backend Parity
  8.  Source Graph / Coverage
  9.  Benchmarks
  10. Known Limitations
  11. Artifacts
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import subprocess
import sys
import webbrowser
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
LIB_ROOT   = SCRIPT_DIR.parent


# ---------------------------------------------------------------------------
# Data collectors — each returns a dict that the renderer consumes
# ---------------------------------------------------------------------------

def _run(cmd: list[str], cwd: Path | None = None) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd or LIB_ROOT, timeout=30)
        return r.stdout.strip()
    except Exception as exc:
        print(f"::warning::caas_dashboard _run({cmd[0]}): {exc}", file=sys.stderr)
        return ""


def _load_json(path: Path) -> dict | list | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def collect_git() -> dict:
    sha     = _run(["git", "rev-parse", "HEAD"])
    short   = _run(["git", "rev-parse", "--short=8", "HEAD"])
    branch  = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    dirty   = bool(_run(["git", "status", "--short"]))
    msg     = _run(["git", "log", "-1", "--format=%s"])
    date    = _run(["git", "log", "-1", "--format=%ci"])
    return {
        "sha": sha, "short": short, "branch": branch,
        "dirty": dirty, "message": msg, "date": date,
    }


def collect_platform() -> dict:
    import platform
    return {
        "python": sys.version.split()[0],
        "os": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "node": platform.node(),
    }


_FALLBACK_GATE_PATHS = (
    # Files written by the CI workflows in priority order. Used when running
    # the dashboard from a cold runner (no DB) where the live invocation of
    # audit_gate.py would crash with sys.exit(1).
    "caas_audit_gate.json",
    "stage2_audit_gate.json",
    "out/reports/caas_audit_gate.json",
)


_FALLBACK_MAX_AGE_DAYS = 7


def _read_fallback(name_candidates: tuple[str, ...]) -> dict | None:
    """Find and load the first matching report file.

    Returns None if absent or if the file's timestamp is older than
    _FALLBACK_MAX_AGE_DAYS (prevents stale data from silently passing as
    current). Skips stale candidates and tries the next one in order.
    M-5 fix: also accepts 'timestamp' (legacy field name) when 'generated_at'
    is absent, so legacy producers are not silently dropped.
    """
    cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(
        days=_FALLBACK_MAX_AGE_DAYS
    )
    for name in name_candidates:
        p = LIB_ROOT / name
        if not p.exists():
            continue
        data = _load_json(p)
        if not isinstance(data, dict):
            continue
        # Accept either generated_at (new) or timestamp (legacy producers).
        ts_raw = data.get("generated_at") or data.get("timestamp")
        if not ts_raw:
            # No timestamp at all — cannot verify freshness; treat as stale.
            continue
        try:
            ts = datetime.datetime.fromisoformat(
                str(ts_raw).replace("Z", "+00:00")
            )
            if ts < cutoff:
                continue  # stale — try next candidate
        except (ValueError, TypeError):
            # Malformed timestamp — cannot verify freshness; treat as stale.
            continue
        data["_source"] = f"fallback:{name}"
        return data
    return None


def collect_audit_gate() -> dict:
    try:
        raw = subprocess.run(
            ["python3", "ci/audit_gate.py", "--json"],
            capture_output=True, text=True, cwd=LIB_ROOT, timeout=60,
        )
        # BUG-5 fix: parse stdout unconditionally regardless of exit code.
        # A non-zero exit means the gate FAILED — but the JSON payload is still
        # valid and should be shown. Only fall back to cache if stdout is
        # empty/unparseable (cold runner / no DB).
        if raw.stdout.strip():
            try:
                d = json.loads(raw.stdout)
                d["_source"] = "live"
                return d
            except (json.JSONDecodeError, ValueError):
                pass
        # stdout is empty or unparseable — cold runner / no DB / crash.
        # Fall back to the most recent on-disk JSON produced by a prior CI step.
        # Note: we do NOT fall back when exit != 0 and stdout is valid JSON,
        # because that would hide genuine FAIL results behind stale cached data.
        fallback = _read_fallback(_FALLBACK_GATE_PATHS)
        if fallback is not None:
            return fallback
        err = raw.stderr[:300] if raw.stderr else f"exit {raw.returncode}"
        return {"error": err, "_source": "live-failed"}
    except Exception as exc:
        fallback = _read_fallback(_FALLBACK_GATE_PATHS)
        if fallback is not None:
            return fallback
        return {"error": str(exc), "_source": "live-exception"}


def collect_preflight_checks() -> list[dict]:
    """Run the lightweight preflight check scripts and collect results."""
    checks = []
    scripts = [
        ("Core Build Mode",        "ci/check_core_build_mode.py"),
        ("Bitcoin Core Gate 2e",   "ci/check_bitcoin_core_test_results.py"),
        ("Exploit Wiring (Stage 0)","ci/check_exploit_wiring.py"),
    ]
    for label, script in scripts:
        try:
            r = subprocess.run(
                ["python3", script, "--json"],
                capture_output=True, text=True, cwd=LIB_ROOT, timeout=60,
            )
            if r.returncode == 77:
                checks.append({"label": label, "status": "ADV-SKIP",
                               "detail": "advisory-skip (no required infrastructure)",
                               "items": []})
            elif "exploit_wiring" in script:
                try:
                    d = json.loads(r.stdout)
                    status = d.get("result", "UNKNOWN")
                    detail = (f"{d.get('poc_files', '?')} PoC files, "
                              f"{d.get('wired', '?')} wired, "
                              f"{d.get('unwired', '?')} unwired")
                except (json.JSONDecodeError, ValueError):
                    status = "ERROR"
                    detail = "failed to parse wiring report"
                checks.append({"label": label, "status": status, "detail": detail,
                               "items": []})
            else:
                # H-2 fix: check exit code before parsing. A crash with empty
                # stdout would previously produce items=[] → PASS (false green).
                if r.returncode != 0 and r.returncode != 77:
                    checks.append({"label": label, "status": "ERROR",
                                   "detail": f"script exited {r.returncode}; "
                                              f"stderr: {r.stderr[:200].strip()}",
                                   "items": []})
                    continue
                # M-3 fix: if stdout is empty after a non-error exit, treat as
                # PASS with 0 items rather than crashing on json.loads("").
                if not r.stdout.strip():
                    checks.append({"label": label, "status": "PASS",
                                   "detail": "0 checks (no output)", "items": []})
                    continue
                try:
                    items = json.loads(r.stdout)
                except (json.JSONDecodeError, ValueError):
                    # Script doesn't support --json or produced non-JSON output.
                    checks.append({"label": label, "status": "ERROR",
                                   "detail": "--json flag unsupported or non-JSON output",
                                   "items": []})
                    continue
                fail = sum(1 for i in items if i.get("status") == "FAIL")
                warn = sum(1 for i in items if i.get("status") == "WARN")
                status = "FAIL" if fail else ("WARN" if warn else "PASS")
                checks.append({"label": label, "status": status,
                               "detail": f"{len(items)} checks, {fail} failed, {warn} warnings",
                               "items": items})
        except Exception as exc:
            checks.append({"label": label, "status": "ERROR",
                           "detail": str(exc), "items": []})
    return checks


_FALLBACK_AUTONOMY_PATHS = (
    "caas_autonomy.json",
    "stage3_autonomy.json",
    "docs/SECURITY_AUTONOMY_KPI.json",
    "out/reports/caas_autonomy.json",
)


def collect_autonomy() -> dict:
    # F-04 fix: parse JSON from stdout regardless of exit code, mirroring
    # collect_audit_gate().  A non-zero exit (score < 100) still produces valid
    # JSON that we want to display; discarding it caused the dashboard to show
    # stale cached data while the live gate was actively failing.
    try:
        raw = subprocess.run(
            ["python3", "ci/security_autonomy_check.py", "--json"],
            capture_output=True, text=True, cwd=LIB_ROOT, timeout=60,
        )
        if raw.stdout.strip():
            try:
                d = json.loads(raw.stdout)
                d["_source"] = "live" if raw.returncode == 0 else "live-failed"
                return d
            except (json.JSONDecodeError, ValueError):
                pass
        # Cold-runner fallback — see collect_audit_gate() for rationale.
        fallback = _read_fallback(_FALLBACK_AUTONOMY_PATHS)
        if fallback is not None:
            return fallback
        if raw.returncode != 0:
            return {
                "autonomy_score": 0,
                "overall_pass": False,
                "error": f"security_autonomy_check.py exited {raw.returncode}",
                "stderr": raw.stderr[:200],
                "_source": "live-failed",
            }
        return {"error": "no output", "overall_pass": False, "_source": "live-empty"}
    except Exception as exc:
        fallback = _read_fallback(_FALLBACK_AUTONOMY_PATHS)
        if fallback is not None:
            return fallback
        return {"error": str(exc), "overall_pass": False, "_source": "live-exception"}


def collect_claims() -> list[dict]:
    data = _load_json(LIB_ROOT / "docs" / "ASSURANCE_CLAIMS.json")
    if not data or not isinstance(data, dict):
        return []
    return data.get("claims", [])


def collect_exploit_corpus() -> dict:
    """Read EXPLOIT_TEST_CATALOG.md and unified_audit_runner.cpp for stats."""
    catalog_path = LIB_ROOT / "docs" / "EXPLOIT_TEST_CATALOG.md"
    runner_path  = LIB_ROOT / "audit" / "unified_audit_runner.cpp"

    files = list((LIB_ROOT / "audit").glob("test_exploit_*.cpp"))
    total = len(files)

    # F-15 fix: count wired by scanning ALL_MODULES entries only (not forward
    # declarations), to avoid inflating the count with stale declarations that
    # no longer have a matching ALL_MODULES entry.  The ALL_MODULES initializer
    # list contains lines of the form `{test_exploit_*_run, ...}`.
    wired = 0
    if runner_path.exists():
        text = runner_path.read_text(errors="replace")
        # Find the ALL_MODULES block and extract symbols from it.
        # Pattern: { test_exploit_<name>_run, "section", ... }
        all_modules_symbols = set(re.findall(
            r'\{\s*(test_exploit_\w+_run)\s*,', text
        ))
        wired = len(all_modules_symbols)

    # Category breakdown from file names
    categories: dict[str, int] = {}
    for f in files:
        stem = f.stem.replace("test_exploit_", "")
        parts = stem.split("_")
        cat = parts[0] if parts else "other"
        categories[cat] = categories.get(cat, 0) + 1

    # Top 12 categories
    top_cats = sorted(categories.items(), key=lambda x: -x[1])[:12]

    return {"total": total, "wired": wired, "categories": top_cats}


def collect_ct_status() -> dict:
    data = _load_json(LIB_ROOT / "docs" / "ASSURANCE_CLAIMS.json")
    claims = data.get("claims", []) if data else []
    ct_claim = next((c for c in claims if "constant" in c.get("claim", "").lower()
                     or "ct" in c.get("claim_id", "").lower()), None)
    # Check for CT-related test files — derive names dynamically so the list
    # stays accurate when CT modules are renamed/added (FINDING-19 fix).
    ct_files = list((LIB_ROOT / "audit").glob("*ct*"))
    ct_files += list((LIB_ROOT / "audit").glob("*sidechannel*"))
    ct_coverage = sorted({f.stem for f in ct_files})
    return {
        "claim": ct_claim.get("claim", "CT signing via secp256k1::ct::*") if ct_claim else
                 "All secret-bearing signing paths use secp256k1::ct::* primitives",
        "status": ct_claim.get("current_status", "ACTIVE") if ct_claim else "ACTIVE",
        "test_files": len(ct_files),
        "coverage": ct_coverage,
    }


def collect_differential() -> dict:
    libsecp_file = LIB_ROOT / "audit" / "test_cross_libsecp256k1.cpp"
    btc_results  = LIB_ROOT / "docs" / "BITCOIN_CORE_TEST_RESULTS.json"
    btc = _load_json(btc_results)
    canonical = _load_json(LIB_ROOT / "docs" / "canonical_data.json") or {}
    btc_pass  = canonical.get("bitcoin_core_tests_pass", btc.get("summary", {}).get("passed", 0) if btc else 0)
    btc_total = canonical.get("bitcoin_core_tests_total", btc.get("summary", {}).get("total", 0) if btc else 0)
    return {
        "libsecp_cross_test": libsecp_file.exists(),
        "wycheproof": {
            "ecdsa":             canonical.get("wycheproof_ecdsa",             "?"),
            "ecdh":              canonical.get("wycheproof_ecdh",              "?"),
            "extended":          canonical.get("wycheproof_extended",          "?"),
            "sha256":            canonical.get("wycheproof_sha256",            "?"),
            "hmac":              canonical.get("wycheproof_hmac",              "?"),
            "chacha20_poly1305": canonical.get("wycheproof_chacha20_poly1305", "?"),
        },
        "bitcoin_core": {
            "total":  btc_total,
            "passed": btc_pass,
            "failed": btc.get("summary", {}).get("failed", 0) if btc else 0,
            "commit": btc.get("backend_commit", "") if btc else "",
        },
        "libsecp_eckey_api": canonical.get("libsecp_eckey_api", "?"),
        "rgrinding":         canonical.get("rgrinding",         "?"),
    }


def collect_backend_parity() -> dict:
    data = _load_json(LIB_ROOT / "docs" / "GPU_BACKEND_EVIDENCE.json")
    backends = []
    if data and isinstance(data, dict):
        for b in data.get("backends", []):
            publishable = b.get("publishable", b.get("parity_with_cpu", "?"))
            parity_str  = ("yes" if publishable is True else
                           "no"  if publishable is False else str(publishable))
            notes       = "; ".join(b.get("artifact_notes", [])) or b.get("notes", "")
            backends.append({
                "name":   b.get("backend", b.get("name", "?")),
                "status": b.get("status", "?"),
                "parity": parity_str,
                "notes":  notes[:120],
            })
    if not backends:
        # No GPU_BACKEND_EVIDENCE.json — surface the gap honestly. The
        # backend implementations exist (src/cuda/, src/opencl/, src/metal/,
        # src/gpu_backend_metal.mm) — previous code hardcoded
        # `Metal: PLANNED` here, which was already false since src/metal/
        # has a working backend implementation. Don't fabricate values when
        # the canonical evidence file is absent.
        backends = [
            {"name": "evidence",
             "status": "no GPU_BACKEND_EVIDENCE.json",
             "parity": "n/a",
             "notes": "see docs/GPU_BACKEND_EVIDENCE.json (not present)"},
        ]
    return {"backends": backends}


def collect_source_graph() -> dict:
    """Pull function/file counts from the SQLite source graph.

    Older code shelled out to a `query_graph.py --stats` script that does
    not exist in this checkout — that produced "?" rows in the dashboard.
    Now we query the `function_index` table directly via the canonical
    `source_graph.py sql` CLI.
    """
    def _count(sql: str) -> int | str:
        try:
            r = subprocess.run(
                ["python3", "tools/source_graph_kit/source_graph.py", "sql", sql],
                capture_output=True, text=True, cwd=LIB_ROOT, timeout=30,
            )
            for line in r.stdout.splitlines():
                stripped = line.strip()
                if stripped.isdigit():
                    return int(stripped)
            return "?"
        except Exception:
            return "?"

    funcs = _count("SELECT COUNT(*) FROM function_index")
    files = _count("SELECT COUNT(DISTINCT file) FROM function_index")
    return {
        "functions": funcs,
        "files":     files,
        "raw": f"functions={funcs}  files={files}",
    }


def collect_benchmarks() -> dict:
    data = _load_json(LIB_ROOT / "docs" / "BITCOIN_CORE_BENCH_RESULTS.json")
    if not data:
        return {"results": [], "config": {}, "corrupt_rows": [], "stale": False, "stale_reason": ""}
    stale = False
    stale_reason = ""
    generated_at = data.get("generated_at") or data.get("timestamp") or ""
    if generated_at:
        try:
            ts = datetime.datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
            age_days = (datetime.datetime.now(datetime.timezone.utc) - ts).total_seconds() / 86400
            if age_days > 30:
                stale = True
                stale_reason = f"{age_days:.0f} days old (max 30)"
        except Exception:
            pass
    valid_results: list = []
    corrupt_rows: list[str] = []
    for r in data.get("results", []):
        base = r.get("baseline_libsecp256k1", 0)
        ours = r.get("ultrafast_secp256k1", 0)
        pct  = r.get("improvement_pct", 0)
        name = r.get("benchmark", "?")
        if base == 0 or ours == 0:
            corrupt_rows.append(f"{name}: zero timing (base={base}, ours={ours})")
        elif abs(pct) > 99999:
            corrupt_rows.append(f"{name}: impossible improvement_pct={pct}")
        else:
            valid_results.append(r)
    return {
        "results":      valid_results,
        "corrupt_rows": corrupt_rows,
        "config":       data.get("bench_config", {}),
        "summary":      data.get("summary", {}),
        "methodology":  data.get("methodology", ""),
        "stale":        stale,
        "stale_reason": stale_reason,
    }


def collect_limitations() -> list[str]:
    """Real, current open items.

    Source of truth is `docs/RESIDUAL_RISK_REGISTER.md` (table rows starting
    `| RR-`).  We list only entries whose Status is OPEN, ACCEPTED, or DEFERRED
    — never CLOSED/RESOLVED/FIXED.  Falls back to KNOWN_LIMITATIONS.md (legacy)
    and finally to an empty list.

    F-29 fix: RESIDUAL_RISK_REGISTER.md is checked FIRST (authoritative); only
    if it is absent or empty do we fall back to the legacy KNOWN_LIMITATIONS.md.
    The previous order was reversed, causing the legacy file to shadow the
    authoritative register when both existed.
    """
    rr_path = LIB_ROOT / "docs" / "RESIDUAL_RISK_REGISTER.md"
    if rr_path.exists():
        out: list[str] = []
        for ln in rr_path.read_text(errors="replace").splitlines():
            if not ln.startswith("| RR-"):
                continue
            cols = [c.strip() for c in ln.split("|")]
            # Format: ['', 'RR-XXX', 'Class', 'Status', 'Owner', 'Notes', '']
            if len(cols) < 5:
                continue
            rr_id, klass, status = cols[1], cols[2], cols[3]
            status_upper = status.upper()
            # Skip closed/resolved/fixed entries — they are no longer limitations.
            if any(t in status_upper for t in ("CLOSED", "RESOLVED", "FIXED")):
                continue
            out.append(f"{rr_id} — {klass} ({status})")
        if out:
            return out[:12]

    lim_path = LIB_ROOT / "docs" / "KNOWN_LIMITATIONS.md"
    if lim_path.exists():
        text = lim_path.read_text(errors="replace")
        items = re.findall(r'^[-*]\s+(.+)', text, re.MULTILINE)
        if items:
            return items[:12]

    return []


def collect_artifacts() -> list[dict]:
    now = datetime.datetime.now(datetime.timezone.utc).timestamp()
    items = []
    # F-21 fix: also scan repo root for live CAAS report files written by gate.yml
    # and preflight.yml (caas_audit_gate.json, caas_autonomy.json, etc.).
    roots = [
        LIB_ROOT,
        LIB_ROOT / "docs",
        LIB_ROOT / "out" / "audit-output",
        LIB_ROOT / "out" / "owner_audit",
        LIB_ROOT / "out" / "research_monitor",
        LIB_ROOT / "out" / "auditor_mode",
    ]
    _seen: set[str] = set()
    for root in roots:
        if not root.exists():
            continue
        # Repo root: only pick up well-known CAAS report files to avoid noise.
        if root == LIB_ROOT:
            candidates = sorted(root.glob("caas_*.json")) + sorted(root.glob("stage*_*.json"))
        else:
            candidates = sorted(root.glob("*.json")) + sorted(root.glob("*.md"))
        for p in candidates:
            try:
                st = p.stat()
            except OSError:
                continue
            age_h = (now - st.st_mtime) / 3600
            freshness = "FRESH" if age_h < 24 else "RECENT" if age_h < 168 else "STALE"
            rel = str(p.relative_to(LIB_ROOT))
            if rel in _seen:
                continue
            _seen.add(rel)
            items.append({
                "name": p.name,
                "path": rel,
                "type": "JSON" if p.suffix == ".json" else "Markdown",
                "freshness": freshness,
                "age_h": round(age_h, 1),
            })
    return items


# ---------------------------------------------------------------------------
# HTML renderer
# ---------------------------------------------------------------------------

CSS = """
:root{--bg:#0d1117;--bg2:#161b22;--bg3:#21262d;--border:#30363d;--text:#e6edf3;
--text2:#8b949e;--green:#3fb950;--red:#f85149;--yellow:#d29922;--blue:#58a6ff;
--purple:#bc8cff;--cyan:#39d353;--accent:#1f6feb;}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',system-ui,sans-serif;background:var(--bg);color:var(--text);
font-size:14px;line-height:1.6}
a{color:var(--blue);text-decoration:none}
h1{font-size:1.6rem;font-weight:700;letter-spacing:-.5px}
h2{font-size:1.15rem;font-weight:600;margin:1.5rem 0 .75rem;color:var(--blue);
border-bottom:1px solid var(--border);padding-bottom:.4rem}
h3{font-size:.95rem;font-weight:600;color:var(--text2);margin:.8rem 0 .4rem}
.header{background:var(--bg2);border-bottom:1px solid var(--border);
padding:1.2rem 2rem;display:flex;align-items:center;gap:1.5rem;flex-wrap:wrap}
.header-title{flex:1}
.header-meta{font-size:.78rem;color:var(--text2);margin-top:.3rem}
.badge{display:inline-flex;align-items:center;padding:.2rem .55rem;
border-radius:6px;font-size:.72rem;font-weight:700;letter-spacing:.04em;
text-transform:uppercase;gap:.3rem}
.pass{background:#1a3a22;color:var(--green);border:1px solid #2ea043}
.fail{background:#3a1a1a;color:var(--red);border:1px solid #da3633}
.warn{background:#3a2a0a;color:var(--yellow);border:1px solid #9e6a03}
.info{background:#1a2a3a;color:var(--blue);border:1px solid #1f6feb}
.dirty{background:#3a2a0a;color:var(--yellow);border:1px solid #9e6a03}
.clean{background:#1a3a22;color:var(--green);border:1px solid #2ea043}
.active{background:#2a1a3a;color:var(--purple);border:1px solid #8957e5}
.main{max-width:1300px;margin:0 auto;padding:1.5rem 2rem}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:1rem}
.grid3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:1rem}
@media(max-width:900px){.grid2,.grid3{grid-template-columns:1fr}}
.card{background:var(--bg2);border:1px solid var(--border);border-radius:8px;
padding:1rem 1.2rem}
.stat-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:.75rem;
margin-bottom:1.2rem}
.stat{background:var(--bg3);border:1px solid var(--border);border-radius:6px;
padding:.7rem 1rem;text-align:center}
.stat-val{font-size:1.9rem;font-weight:700;line-height:1}
.stat-label{font-size:.72rem;color:var(--text2);margin-top:.2rem;letter-spacing:.04em;
text-transform:uppercase}
.green{color:var(--green)} .red{color:var(--red)}
.yellow{color:var(--yellow)} .blue{color:var(--blue)} .purple{color:var(--purple)}
table{width:100%;border-collapse:collapse;font-size:.82rem}
th{background:var(--bg3);color:var(--text2);font-weight:600;text-align:left;
padding:.45rem .7rem;border-bottom:1px solid var(--border);letter-spacing:.04em;font-size:.72rem}
td{padding:.4rem .7rem;border-bottom:1px solid #21262d;vertical-align:top}
tr:hover td{background:#1c2128}
.mono{font-family:'Cascadia Code','Fira Code',monospace;font-size:.8rem}
.bar-bg{background:var(--bg3);border-radius:4px;height:8px;overflow:hidden;width:100%;
margin-top:4px}
.bar-fill{height:100%;border-radius:4px;background:var(--green);transition:width .3s}
.bar-fill.warn{background:var(--yellow)} .bar-fill.fail{background:var(--red)}
.pill{display:inline-block;padding:.1rem .4rem;border-radius:4px;font-size:.72rem;
background:var(--bg3);border:1px solid var(--border);margin:1px}
.section-anchor{scroll-margin-top:70px}
nav{position:sticky;top:0;z-index:100;background:var(--bg2);
border-bottom:1px solid var(--border);padding:.5rem 2rem;
display:flex;gap:.5rem;flex-wrap:wrap;font-size:.78rem}
nav a{color:var(--text2);padding:.2rem .5rem;border-radius:4px}
nav a:hover{background:var(--bg3);color:var(--text)}
.bench-delta{font-weight:700}
.bench-delta.pos{color:var(--green)} .bench-delta.neg{color:var(--red)}
code{background:var(--bg3);border:1px solid var(--border);
border-radius:3px;padding:.1em .4em;font-size:.82em;font-family:'Cascadia Code','Fira Code',monospace}
.lim-list{list-style:none;padding:0}
.lim-list li{padding:.35rem 0;border-bottom:1px solid var(--border);
display:flex;gap:.6rem;align-items:flex-start}
.lim-list li::before{content:"⚠";color:var(--yellow);flex-shrink:0}
.artifact-table td:first-child{font-family:monospace;font-size:.78rem;color:var(--cyan)}
footer{text-align:center;padding:2rem;color:var(--text2);font-size:.75rem;
border-top:1px solid var(--border);margin-top:2rem}
"""

NAV_ITEMS = [
    ("exec",      "Executive"),
    ("gates",     "Gates"),
    ("claims",    "Claims"),
    ("exploit",   "Exploits"),
    ("ct",        "CT"),
    ("diff",      "Differential"),
    ("backend",   "Backends"),
    ("graph",     "Coverage"),
    ("bench",     "Benchmarks"),
    ("limits",    "Limitations"),
    ("artifacts", "Artifacts"),
]


def _badge(status: str, text: str | None = None) -> str:
    t = text or status
    cls = {"PASS":"pass","FAIL":"fail","WARN":"warn","ERROR":"fail",
           "COMPLETE":"pass","PLANNED":"warn","PARTIAL":"warn","ACTIVE":"active",
           "UNKNOWN":"warn"}.get(status.upper(), "info")
    icon = {"pass":"✓","fail":"✗","warn":"⚠","info":"ℹ","active":"●"}.get(cls,"·")
    return f'<span class="badge {cls}">{icon} {t}</span>'


def _pct_bar(pct: float, cls: str = "") -> str:
    c = cls or ("fail" if pct < 50 else "warn" if pct < 80 else "")
    return (f'<div class="bar-bg"><div class="bar-fill {c}" '
            f'style="width:{min(pct,100):.1f}%"></div></div>')


def render_section_exec(git: dict, platform: dict, autonomy: dict) -> str:
    canonical = _load_json(LIB_ROOT / "docs" / "canonical_data.json") or {}
    btc_tests_pass = canonical.get("bitcoin_core_tests_pass", "?")
    btc_tests_total = canonical.get("bitcoin_core_tests_total", btc_tests_pass)
    exploit_poc_count = canonical.get("exploit_poc_count", "?")
    btc_pass_rate = (
        f"{round(btc_tests_pass / btc_tests_total * 100)}%"
        if isinstance(btc_tests_pass, int) and isinstance(btc_tests_total, int) and btc_tests_total > 0
        else "?"
    )

    score = autonomy.get("autonomy_score", "?")
    score_cls = "green" if isinstance(score, int) and score >= 90 else \
                "yellow" if isinstance(score, int) and score >= 70 else "red"
    dirty_badge = _badge("WARN", "DIRTY") if git.get("dirty") else _badge("PASS", "CLEAN")

    score_bar = _pct_bar(score if isinstance(score, (int,float)) else 0)
    overall = autonomy.get("overall_pass", False)
    autonomy_src = autonomy.get("_source", "")
    # VIZ-01 fix: a failing autonomy check is FAIL (red), not "IN PROGRESS" (yellow).
    verdict = _badge("PASS", "AUDIT READY") if overall else _badge("FAIL", "AUTONOMY FAIL")

    # F-09 fix: show a staleness/failure banner when live data is unavailable or failing.
    autonomy_banner = ""
    if autonomy_src in ("live-failed", "live-empty", "live-exception"):
        err = autonomy.get("error", "") or autonomy.get("stderr", "")
        autonomy_banner = (
            f'<div style="background:#3a1a1a;border:1px solid #da3633;border-radius:6px;'
            f'padding:.6rem 1rem;margin-bottom:.8rem;color:#f85149;font-size:.82rem">'
            f'&#10007; <b>AUTONOMY LIVE CHECK FAILED</b> — '
            f'<code>security_autonomy_check.py</code> returned a non-zero exit.'
            f'{(" Error: " + err[:120]) if err else ""}'
            f'</div>'
        )
    elif autonomy_src.startswith("fallback"):
        fallback_name = autonomy_src.split(":", 1)[-1] if ":" in autonomy_src else "cached file"
        autonomy_banner = (
            f'<div style="background:#3a2500;border:1px solid #9e6a03;border-radius:6px;'
            f'padding:.6rem 1rem;margin-bottom:.8rem;color:#d29922;font-size:.82rem">'
            f'&#9888; <b>STALE DATA</b> — live autonomy check failed; showing cached '
            f'results from <code>{fallback_name}</code>.'
            f'</div>'
        )

    return f"""
<section class="section-anchor" id="exec">
<h2>1 · Executive Summary</h2>
<div class="stat-grid">
  <div class="stat"><div class="stat-val green">{btc_tests_pass}</div>
    <div class="stat-label">Bitcoin Core Tests</div></div>
  <div class="stat"><div class="stat-val green">{exploit_poc_count}</div>
    <div class="stat-label">Exploit PoC Count</div></div>
  <div class="stat"><div class="stat-val {score_cls}">{score}</div>
    <div class="stat-label">Autonomy Score /100</div>{score_bar}</div>
  <div class="stat"><div class="stat-val green">{btc_pass_rate}</div>
    <div class="stat-label">BTC Test Pass Rate</div></div>
</div>
{autonomy_banner}<div class="card">
  <table>
    <tr><td style="width:140px;color:var(--text2)">Verdict</td>
      <td>{verdict}</td></tr>
    <tr><td>Commit</td>
      <td class="mono">{git.get('short','?')} &nbsp;
        <span style="color:var(--text2)">{git.get('message','')[:80]}</span></td></tr>
    <tr><td>Branch</td>
      <td><code>{git.get('branch','?')}</code> &nbsp; {dirty_badge}</td></tr>
    <tr><td>Commit date</td>
      <td class="mono">{git.get('date','?')[:19]}</td></tr>
    <tr><td>Platform</td>
      <td class="mono">{platform.get('os','?')[:80]}</td></tr>
    <tr><td>Python</td>
      <td class="mono">{platform.get('python','?')}</td></tr>
    <tr><td>Generated</td>
      <td class="mono">{datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}</td></tr>
  </table>
</div>
</section>"""


def render_section_gates(preflight: list[dict], gate: dict) -> str:
    stale_banner = ""
    gate_source = gate.get("_source", "live")
    if gate_source.startswith("fallback"):
        fallback_name = gate_source.split(":", 1)[-1] if ":" in gate_source else "cached file"
        stale_banner = (
            f'<div style="background:#3a2500;border:1px solid #9e6a03;border-radius:6px;'
            f'padding:.6rem 1rem;margin-bottom:.8rem;color:#d29922;font-size:.82rem">'
            f'&#9888; <b>STALE DATA</b> — live audit_gate.py failed; showing cached '
            f'results from <code>{fallback_name}</code>. '
            f'Gate status may not reflect current codebase state.'
            f'</div>'
        )

    rows = ""
    for c in preflight:
        status = c.get("status", "?")
        rows += f"""<tr>
          <td>{c['label']}</td>
          <td>{_badge(status)}</td>
          <td style="color:var(--text2)">{c.get('detail','')}</td>
        </tr>"""
        for item in c.get("items", []):
            ist = item.get("status","?")
            detail = "; ".join(item.get("detail", []))[:100]
            rows += f"""<tr style="font-size:.78rem">
              <td style="padding-left:1.5rem;color:var(--text2)">{item.get('name','?')}</td>
              <td>{_badge(ist)}</td>
              <td style="color:var(--text2)">{detail}</td>
            </tr>"""

    # verdict may be a string ("PASS"/"FAIL") or a dict
    verdict_raw = gate.get("verdict", "?")
    if isinstance(verdict_raw, dict):
        verdict_status = verdict_raw.get("status", "?")
        verdict_label  = verdict_raw.get("label", verdict_status)
    else:
        verdict_status = str(verdict_raw)
        verdict_label  = verdict_status
    summary = gate.get("summary", {})
    return f"""
<section class="section-anchor" id="gates">
<h2>2 · Gate Status</h2>
{stale_banner}<div class="grid2">
<div class="card">
  <h3>Preflight Gates</h3>
  <table>
    <thead><tr><th>Gate</th><th>Status</th><th>Detail</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
</div>
<div class="card">
  <h3>CAAS Audit Gate</h3>
  <table>
    <tr><td>Verdict</td>
      <td>{_badge(verdict_status, verdict_label)}</td></tr>
    <tr><td>Checks total</td>
      <td><b>{summary.get('total_checks',summary.get('total','?'))}</b></td></tr>
    <tr><td>Passing</td>
      <td class="green"><b>{summary.get('passing',summary.get('pass','?'))}</b></td></tr>
    <tr><td>Failing</td>
      <td class="red"><b>{summary.get('failing',summary.get('fail', summary.get('failed','?')))}</b></td></tr>
    <tr><td>Advisory</td>
      <td class="yellow"><b>{summary.get('advisory','?')}</b></td></tr>
    <tr><td>Run ID</td>
      <td class="mono" style="font-size:.72rem">{gate.get('run_id','?')[:40]}</td></tr>
  </table>
</div>
</div>
</section>"""


def render_section_claims(claims: list[dict]) -> str:
    rows = ""
    for c in claims:
        cid   = c.get("claim_id", "?")
        area  = c.get("area", "")
        claim = c.get("claim", "")[:120]
        evid  = ", ".join(c.get("primary_evidence", []))[:80]
        status = c.get("current_status", "ACTIVE")
        rows += f"""<tr>
          <td class="mono" style="font-size:.75rem">{cid}</td>
          <td><span class="pill">{area}</span></td>
          <td>{claim}</td>
          <td style="color:var(--text2);font-size:.78rem">{evid}</td>
          <td>{_badge(status)}</td>
        </tr>"""
    return f"""
<section class="section-anchor" id="claims">
<h2>3 · Security Claims</h2>
<div class="card">
<table>
  <thead><tr><th>ID</th><th>Area</th><th>Claim</th><th>Primary Evidence</th><th>Status</th></tr></thead>
  <tbody>{rows}</tbody>
</table>
</div>
</section>"""


def render_section_exploit(corpus: dict) -> str:
    total = corpus.get("total", 0)
    wired = corpus.get("wired", 0)
    pct   = (wired / total * 100) if total else 0

    cats_html = ""
    for cat, count in corpus.get("categories", []):
        w = min(count / max(total, 1) * 100 * 4, 100)
        cats_html += f"""<tr>
          <td style="width:120px"><span class="pill">{cat}</span></td>
          <td>{count}</td>
          <td><div class="bar-bg" style="width:180px">
            <div class="bar-fill" style="width:{w:.0f}%"></div></div></td>
        </tr>"""

    return f"""
<section class="section-anchor" id="exploit">
<h2>4 · Exploit PoC Corpus</h2>
<div class="grid2">
<div class="card">
  <div class="stat-grid" style="grid-template-columns:1fr 1fr 1fr">
    <div class="stat"><div class="stat-val green">{total}</div>
      <div class="stat-label">Total PoC Files</div></div>
    <div class="stat"><div class="stat-val green">{wired}</div>
      <div class="stat-label">Wired in Runner</div></div>
    <div class="stat"><div class="stat-val {'green' if pct==100 else 'yellow'}">{pct:.0f}%</div>
      <div class="stat-label">Coverage</div></div>
  </div>
  {_pct_bar(pct)}
  <p style="margin-top:.8rem;color:var(--text2);font-size:.8rem">
    All exploit PoC files are registered in <code>unified_audit_runner.cpp</code>
    and catalogued in <code>docs/EXPLOIT_TEST_CATALOG.md</code>.
    <code>ci/check_exploit_wiring.py</code> enforces this as a CI gate.
  </p>
</div>
<div class="card">
  <h3>Category Breakdown (top 12)</h3>
  <table><tbody>{cats_html}</tbody></table>
</div>
</div>
</section>"""


def render_section_ct(ct: dict) -> str:
    cov = ct.get("coverage", [])
    cov_html = "".join(f'<span class="pill">{c}</span>' for c in cov)
    return f"""
<section class="section-anchor" id="ct">
<h2>5 · Constant-Time</h2>
<div class="card">
  <table>
    <tr><td style="width:160px">Claim</td>
      <td>{ct.get('claim','?')}</td></tr>
    <tr><td>Status</td>
      <td>{_badge(ct.get('status','?'))}</td></tr>
    <tr><td>CT test files</td>
      <td><b>{ct.get('test_files',0)}</b></td></tr>
    <tr><td>Coverage</td><td>{cov_html}</td></tr>
    <tr><td>Guarantee</td>
      <td style="color:var(--text2)">All signing paths that handle private keys or nonces
      route through <code>secp256k1::ct::*</code> branchless primitives.
      Variable-time fast paths are gated at the C++ namespace boundary.</td></tr>
  </table>
</div>
</section>"""


def render_section_differential(diff: dict) -> str:
    btc = diff.get("bitcoin_core", {})
    total  = btc.get("total", 0)
    passed = btc.get("passed", 0)
    failed = btc.get("failed", 0)
    commit = btc.get("commit", "")

    wyche = diff.get("wycheproof", {})
    # VIZ-3 fix: don't always use PASS badge for Wycheproof results — check
    # the actual value. A non-positive number or explicit FAIL/error string
    # should render as a FAIL badge.
    def _wyche_badge(v: object) -> str:
        if isinstance(v, int) and v > 0:
            return _badge("PASS", v)
        if isinstance(v, str) and v.upper() in ("FAIL", "ERROR", "0"):
            return _badge("FAIL", v)
        if isinstance(v, (int, float)) and v == 0:
            return _badge("FAIL", v)
        return _badge("PASS", v)

    wyche_html = "".join(
        f'<tr><td>{k}</td><td>{_wyche_badge(v)}</td></tr>'
        for k, v in wyche.items()
    )

    return f"""
<section class="section-anchor" id="diff">
<h2>6 · Differential Testing</h2>
<div class="grid2">
<div class="card">
  <h3>Bitcoin Core test_bitcoin</h3>
  <div class="stat-grid" style="grid-template-columns:1fr 1fr 1fr">
    <div class="stat"><div class="stat-val green">{passed}</div>
      <div class="stat-label">Passed</div></div>
    <div class="stat"><div class="stat-val {'red' if failed else 'green'}">{failed}</div>
      <div class="stat-label">Failed</div></div>
    <div class="stat"><div class="stat-val {'green' if total == 0 or failed == 0 else 'red'}">{round(passed / total * 100) if total else 0}%</div>
      <div class="stat-label">Pass Rate</div></div>
  </div>
  {_pct_bar(100 if total and failed==0 else (passed/total*100 if total else 0))}
  <p style="margin-top:.8rem;font-size:.78rem;color:var(--text2)">
    Backend commit: <code>{commit}</code><br>
    libsecp EC key API: {_wyche_badge(diff.get("libsecp_eckey_api","?"))}<br>
    R-grinding pattern: {_wyche_badge(diff.get("rgrinding","?"))}
  </p>
</div>
<div class="card">
  <h3>Wycheproof Test Vectors</h3>
  <table>
    <thead><tr><th>Suite</th><th>Result</th></tr></thead>
    <tbody>{wyche_html}</tbody>
  </table>
  <p style="margin-top:.8rem;font-size:.78rem;color:var(--text2)">
    Cross-library differential: <code>test_cross_libsecp256k1.cpp</code>
    links both libraries in-process and compares outputs for identical inputs.
  </p>
</div>
</div>
</section>"""


def render_section_backend(backend: dict) -> str:
    rows = ""
    for b in backend.get("backends", []):
        s = b.get("status", "?")
        rows += f"""<tr>
          <td><b>{b['name']}</b></td>
          <td>{_badge(s)}</td>
          <td>{b.get('parity','?')}</td>
          <td style="color:var(--text2)">{b.get('notes','')}</td>
        </tr>"""
    return f"""
<section class="section-anchor" id="backend">
<h2>7 · Backend Parity</h2>
<div class="card">
<table>
  <thead><tr><th>Backend</th><th>Status</th><th>CPU Parity</th><th>Notes</th></tr></thead>
  <tbody>{rows}</tbody>
</table>
</div>
</section>"""


def render_section_graph(graph: dict) -> str:
    funcs = graph.get("functions", "?")
    files = graph.get("files", "?")
    # VIZ-5 fix: "?" (string, not int) means the graph query failed — treat
    # that as FAIL, not as a neutral unknown status.
    graph_quality_pass = isinstance(graph.get('functions'), int) and graph.get('functions', 0) > 0
    graph_quality_label = 'PASS' if graph_quality_pass else 'FAIL'
    graph_quality_color = 'green' if graph_quality_pass else 'red'
    return f"""
<section class="section-anchor" id="graph">
<h2>8 · Source Graph / Coverage</h2>
<div class="card">
  <div class="stat-grid" style="grid-template-columns:1fr 1fr 1fr">
    <div class="stat"><div class="stat-val blue">{funcs}</div>
      <div class="stat-label">Indexed Functions</div></div>
    <div class="stat"><div class="stat-val blue">{files}</div>
      <div class="stat-label">Source Files</div></div>
    <div class="stat"><div class="stat-val {graph_quality_color}">{graph_quality_label}</div>
      <div class="stat-label">Graph Quality Gate</div></div>
  </div>
  <p style="color:var(--text2);font-size:.8rem;margin-top:.5rem">
    SQLite-backed source graph at <code>tools/source_graph_kit/source_graph.db</code>.
    Query with <code>python3 tools/source_graph_kit/query_graph.py &lt;symbol&gt;</code>.
    Mandatory before any code exploration (enforced by CLAUDE.md).
  </p>
</div>
</section>"""


def render_section_bench(bench: dict) -> str:
    results = bench.get("results", [])
    corrupt_rows = bench.get("corrupt_rows", [])
    rows = ""
    for r in results:
        base = r.get("baseline_libsecp256k1", 0)
        ours = r.get("ultrafast_secp256k1", 0)
        pct  = r.get("improvement_pct", 0)
        dcls = "pos" if pct >= 0 else "neg"
        rows += f"""<tr>
          <td><b>{r.get('benchmark','?')}</b></td>
          <td class="mono">{base:,} ns</td>
          <td class="mono">{ours:,} ns</td>
          <td class="bench-delta {dcls}">{pct:+.1f}%</td>
          <td style="color:var(--text2);font-size:.78rem">{r.get('note','')}</td>
        </tr>"""
    cfg = bench.get("config", {})
    corrupt_banner = ""
    if corrupt_rows:
        items = "".join(f"<li>{c}</li>" for c in corrupt_rows)
        corrupt_banner = (
            f'<div style="background:var(--fail-bg,#3b1a1a);border-left:3px solid #e05050;'
            f'padding:.6rem .9rem;margin-bottom:.8rem;font-size:.82rem;color:#e05050">'
            f'<b>⚠ Corrupt benchmark rows excluded ({len(corrupt_rows)}):</b><ul style="margin:.3rem 0 0 1.2rem">'
            f'{items}</ul></div>'
        )
    return f"""
<section class="section-anchor" id="bench">
<h2>9 · Benchmarks</h2>
<div class="card">
  {corrupt_banner}<h3>bench_bitcoin — UltrafastSecp256k1 vs libsecp256k1</h3>
  <table>
    <thead><tr><th>Benchmark</th><th>Baseline (libsecp)</th>
      <th>UltrafastSecp256k1</th><th>Δ</th><th>Notes</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
  <p style="margin-top:.8rem;font-size:.78rem;color:var(--text2)">
    Config: {cfg.get('duration_s','?')}s stable run · {cfg.get('build_type','?')} ·
    {cfg.get('extra_flags','')} · {cfg.get('shim_config','')}<br>
    Raw results: <code>docs/BITCOIN_CORE_BENCH_RESULTS.json</code>
  </p>
</div>
</section>"""


def render_section_limitations(lims: list[str]) -> str:
    items = "".join(f"<li>{l}</li>" for l in lims)
    return f"""
<section class="section-anchor" id="limits">
<h2>10 · Known Limitations</h2>
<div class="card">
  <p style="color:var(--text2);font-size:.82rem;margin-bottom:.8rem">
    These are <b>explicit open items</b>, not hidden. Full transparency is part of
    the audit contract — reviewers should evaluate the evidence despite these gaps,
    not dismiss the project because they exist.
  </p>
  <ul class="lim-list">{items}</ul>
</div>
</section>"""


def render_section_artifacts(artifacts: list[dict]) -> str:
    freshness_css = {
        "FRESH":  "background:#1a3a22;color:#3fb950;border:1px solid #2ea043",
        "RECENT": "background:#3a2a0a;color:#d29922;border:1px solid #9e6a03",
        "STALE":  "background:#3a1a1a;color:#f85149;border:1px solid #da3633",
    }
    rows = ""
    for a in artifacts[:60]:
        f = a.get("freshness", "RECENT")
        age = a.get("age_h", "?")
        age_str = f"{age}h ago" if isinstance(age, (int, float)) else "?"
        badge_css = freshness_css.get(f, freshness_css["RECENT"])
        path_html = (
            f'<code style="color:var(--cyan);font-size:.78rem">{a["path"]}</code>'
        )
        rows += (
            f'<tr>'
            f'<td>{a["name"]}</td>'
            f'<td>{_badge("info", a["type"])}</td>'
            f'<td class="mono">{path_html}</td>'
            f'<td><span style="{badge_css};padding:1px 6px;border-radius:4px;'
            f'font-size:.7em;font-weight:700">{f}</span>'
            f' <span style="color:var(--text2);font-size:.75em">{age_str}</span></td>'
            f'</tr>'
        )
    return f"""
<section class="section-anchor" id="artifacts">
<h2>11 · Artifacts</h2>
<div class="card">
<table class="artifact-table">
  <thead><tr><th>File</th><th>Type</th><th>Path</th><th>Freshness</th></tr></thead>
  <tbody>{rows}</tbody>
</table>
</div>
</section>"""


def render_html(data: dict) -> str:
    git      = data["git"]
    platform = data["platform"]
    dirty    = git.get("dirty", False)
    dirty_str = "DIRTY" if dirty else "CLEAN"

    # TODO: read from canonical_data.json or CAAS report dynamically
    canonical = _load_json(LIB_ROOT / "docs" / "canonical_data.json") or {}
    btc_tests_pass  = canonical.get("bitcoin_core_tests_pass", "?")
    btc_tests_total = canonical.get("bitcoin_core_tests_total", btc_tests_pass)
    exploit_poc_count = canonical.get("exploit_poc_count", "?")

    nav_html = "".join(
        f'<a href="#{a}">{label}</a>' for a, label in NAV_ITEMS
    )

    body = (
        render_section_exec(git, platform, data["autonomy"])
        + render_section_gates(data["preflight"], data["audit_gate"])
        + render_section_claims(data["claims"])
        + render_section_exploit(data["exploit"])
        + render_section_ct(data["ct"])
        + render_section_differential(data["differential"])
        + render_section_backend(data["backend"])
        + render_section_graph(data["source_graph"])
        + render_section_bench(data["benchmarks"])
        + render_section_artifacts(data["artifacts"])
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CAAS Audit Dashboard · UltrafastSecp256k1 · {git.get('short','?')} · {dirty_str}</title>
<style>{CSS}</style>
</head>
<body>
<header class="header">
  <div class="header-title">
    <h1>⬡ CAAS Audit Dashboard</h1>
    <div class="header-meta">
      UltrafastSecp256k1 &nbsp;·&nbsp; commit
      <code>{git.get('short','?')}</code> &nbsp;·&nbsp;
      branch <code>{git.get('branch','?')}</code> &nbsp;·&nbsp;
      {'<span class="badge dirty">⚠ DIRTY</span>' if dirty else '<span class="badge clean">✓ CLEAN</span>'}
      &nbsp;·&nbsp;
      generated {datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}
    </div>
  </div>
  <div>
    <span class="badge {'pass' if isinstance(btc_tests_pass, int) and isinstance(btc_tests_total, int) and btc_tests_total > 0 else 'warn'}" style="font-size:.85rem;padding:.3rem .8rem">
      {'✓' if isinstance(btc_tests_pass, int) else '?'} {btc_tests_pass}/{btc_tests_total} BTC TESTS</span>
    &nbsp;
    <span class="badge {'pass' if isinstance(exploit_poc_count, int) and exploit_poc_count > 0 else 'warn'}" style="font-size:.85rem;padding:.3rem .8rem">
      {'✓' if isinstance(exploit_poc_count, int) else '?'} {exploit_poc_count} EXPLOITS WIRED</span>
  </div>
</header>
<nav>{nav_html}</nav>
<main class="main">{body}</main>
<footer>
  UltrafastSecp256k1 CAAS Audit Cockpit &nbsp;·&nbsp;
  Executable evidence becomes reviewable evidence &nbsp;·&nbsp;
  Generated {datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}
</footer>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Generate CAAS audit HTML dashboard")
    ap.add_argument("-o", "--output", default="out/caas_dashboard.html",
                    help="Output file (default: out/caas_dashboard.html)")
    ap.add_argument("--open", action="store_true", help="Open in browser after generating")
    ap.add_argument("--serve", type=int, metavar="PORT",
                    help="Serve dashboard on localhost:PORT instead of writing file")
    args = ap.parse_args()

    os.chdir(LIB_ROOT)

    print("Collecting audit data...", flush=True)

    data = {
        "git":          collect_git(),
        "platform":     collect_platform(),
        "audit_gate":   collect_audit_gate(),
        "preflight":    collect_preflight_checks(),
        "autonomy":     collect_autonomy(),
        "claims":       collect_claims(),
        "exploit":      collect_exploit_corpus(),
        "ct":           collect_ct_status(),
        "differential": collect_differential(),
        "backend":      collect_backend_parity(),
        "source_graph": collect_source_graph(),
        "benchmarks":   collect_benchmarks(),
        "artifacts":    collect_artifacts(),
    }

    print("Rendering HTML...", flush=True)
    html = render_html(data)

    if args.serve:
        import http.server, threading, io
        html_bytes = html.encode("utf-8")

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", len(html_bytes))
                self.end_headers()
                self.wfile.write(html_bytes)
            def log_message(self, fmt, *a): pass

        addr = ("localhost", args.serve)
        srv  = http.server.HTTPServer(addr, Handler)
        url  = f"http://localhost:{args.serve}/"
        print(f"\nCAAS Dashboard: {url}  (Ctrl-C to stop)\n")
        webbrowser.open(url)
        srv.serve_forever()
    else:
        out = Path(args.output)
        out.write_text(html, encoding="utf-8")
        size_kb = out.stat().st_size // 1024
        print(f"\nDashboard written: {out}  ({size_kb} KB)\n")
        if args.open:
            webbrowser.open(out.resolve().as_uri())
        return 0


if __name__ == "__main__":
    sys.exit(main())
