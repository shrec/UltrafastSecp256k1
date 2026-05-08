#!/usr/bin/env python3
"""
CI Orchestrator — dispatches and polls GitHub Actions workflows in order.

Flow (push events):
  1. discord-commits.yml          (notify Discord)
  2. parallel: static analysis phase
     - code-quality.yml
     - preflight.yml
  3. parallel: platform builds (if static phase passed)
     - ci.yml (inline platform matrix via workflow_dispatch)
  4. parallel: deep security (concurrent with platforms)
     - security-audit.yml
  5. Final verdict

Workflow files must exist on the ref being dispatched.
Exit 0 = all passed. Exit 1 = any failure.
"""

import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

# Unbuffered stdout so progress shows in real-time in CI logs
sys.stdout.reconfigure(line_buffering=True)

# ── GitHub context ────────────────────────────────────────────────────────────
API = "https://api.github.com"
TOKEN = os.environ["GITHUB_TOKEN"]
REPO = os.environ.get("GITHUB_REPOSITORY", "shrec/UltrafastSecp256k1")
REF = os.environ.get("GITHUB_REF_NAME", "dev")
SHA = os.environ.get("GITHUB_SHA", "")
EVENT = os.environ.get("GITHUB_EVENT_NAME", "push")
RUN_ID = os.environ.get("GITHUB_RUN_ID", "local")

HEADERS = {
    "Accept": "application/vnd.github+json",
    "Authorization": f"Bearer {TOKEN}",
    "X-GitHub-Api-Version": "2022-11-28",
}

POLL_SEC = 20
FIND_TIMEOUT = 120  # seconds to wait for a dispatched run to appear


# ── HTTP helper ────────────────────────────────────────────────────────────────
def _request(method: str, path: str, data=None):
    body = json.dumps(data).encode() if data is not None else None
    req = urllib.request.Request(API + path, data=body, method=method, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read()
            return json.loads(raw.decode()) if raw else None
    except urllib.error.HTTPError as exc:
        msg = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {path} → HTTP {exc.code}: {msg}") from exc


# ── Dispatch + find + wait ─────────────────────────────────────────────────────
def _dispatch(workflow_file: str, inputs: dict | None = None) -> float:
    """Dispatch workflow_dispatch and return the timestamp just before dispatch.

    Only passes inputs that are explicitly given. Each child workflow declares
    its own workflow_dispatch inputs, so we cannot pass orchestrator_sha
    universally — most workflows don't accept it and reject with HTTP 422.
    """
    t0 = time.time()
    payload = {"ref": REF}
    if inputs:
        payload["inputs"] = inputs
    _request("POST", f"/repos/{REPO}/actions/workflows/{workflow_file}/dispatches", payload)
    print(f"  dispatched {workflow_file}")
    return t0


def _find_run(workflow_file: str, after_ts: float) -> int:
    """Poll until a run created after after_ts appears; return its run_id."""
    deadline = time.time() + FIND_TIMEOUT
    while time.time() < deadline:
        data = _request(
            "GET",
            f"/repos/{REPO}/actions/workflows/{workflow_file}/runs"
            f"?branch={REF}&event=workflow_dispatch&per_page=10",
        )
        for run in data.get("workflow_runs", []):
            created = run.get("created_at", "")
            # ISO 8601 → epoch (simple parse without datetime import)
            try:
                from datetime import datetime, timezone
                ts = datetime.fromisoformat(created.replace("Z", "+00:00")).timestamp()
            except Exception:
                ts = after_ts + 1  # fallback: assume it's new
            if ts >= after_ts - 5:  # 5-second grace for clock skew
                return run["id"]
        time.sleep(5)
    raise RuntimeError(f"Timed out waiting for {workflow_file} run to appear")


def _wait(run_id: int, label: str, timeout_sec: int = 7200) -> dict:
    """Poll run until completed; return run dict. Raises on failure."""
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        run = _request("GET", f"/repos/{REPO}/actions/runs/{run_id}")
        status = run.get("status")
        conclusion = run.get("conclusion")
        print(f"  {label}: {status}/{conclusion}  {run.get('html_url','')}")
        if status == "completed":
            if conclusion == "success":
                return run
            raise RuntimeError(
                f"{label} concluded={conclusion}  url={run.get('html_url')}"
            )
        time.sleep(POLL_SEC)
    raise RuntimeError(f"{label} timed out after {timeout_sec}s (run_id={run_id})")


def _run_workflow(workflow_file: str, inputs=None, timeout_sec: int = 7200) -> int:
    t0 = _dispatch(workflow_file, inputs)
    time.sleep(3)  # give GitHub a moment
    run_id = _find_run(workflow_file, t0)
    print(f"  {workflow_file} run_id={run_id}")
    _wait(run_id, workflow_file, timeout_sec=timeout_sec)
    return run_id


# ── Parallel runner ────────────────────────────────────────────────────────────
def _run_parallel(phase_name: str, tasks: list[tuple]) -> None:
    """
    tasks: list of (workflow_file, inputs_dict, timeout_sec)
    All run concurrently; if any fails, raises after collecting all results.
    """
    print(f"\n::group::{phase_name}")
    errors = []

    with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
        futures = {
            pool.submit(_run_workflow, wf, inp, tmo): wf
            for wf, inp, tmo in tasks
        }
        for fut in as_completed(futures):
            wf = futures[fut]
            try:
                run_id = fut.result()
                print(f"  ✓ {wf} passed  run_id={run_id}")
            except Exception as exc:
                errors.append(f"{wf}: {exc}")
                print(f"  ✗ {wf} FAILED: {exc}")

    print("::endgroup::")
    if errors:
        raise RuntimeError(f"{phase_name} failed:\n" + "\n".join(errors))


# ── Main ───────────────────────────────────────────────────────────────────────
def _notify_discord() -> None:
    """Post commit notification directly to Discord webhook.

    We don't dispatch discord-commits.yml because workflow_dispatch loses
    github.event.head_commit (message/author show as '(no message)'/'unknown').
    Instead, fetch commit details via API and curl Discord directly.
    """
    webhook = os.environ.get("DISCORD_WEBHOOK_COMMITS", "").strip()
    if not webhook:
        print("  DISCORD_WEBHOOK_COMMITS not set — skipping Discord notify")
        return

    commit = _request("GET", f"/repos/{REPO}/commits/{SHA}")
    msg = commit.get("commit", {}).get("message", "(no message)").splitlines()[0][:200]
    author = commit.get("commit", {}).get("author", {}).get("name", "unknown")
    short = SHA[:7]
    url = f"https://github.com/{REPO}/commit/{SHA}"

    payload = {
        "embeds": [{
            "title": f"New commit on {REF}",
            "url": url,
            "color": 7506394,
            "description": f"[`{short}`]({url}) {msg}",
            "fields": [
                {"name": "Branch", "value": REF, "inline": True},
                {"name": "Author", "value": author, "inline": True},
            ],
            "footer": {"text": "UltrafastSecp256k1"},
        }]
    }
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        webhook, data=body, method="POST",
        headers={
            "Content-Type": "application/json",
            # Discord rejects requests without a User-Agent (returns 403)
            "User-Agent": "UltrafastSecp256k1-CI-Orchestrator/1.0",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            resp.read()
        print(f"  Discord notified: {short} '{msg[:60]}...' by {author}")
    except urllib.error.HTTPError as exc:
        # Discord notify is non-blocking — log and continue
        print(f"  ::warning::Discord webhook returned HTTP {exc.code} (non-blocking)")


def main() -> None:
    print(f"CI Orchestrator  repo={REPO}  ref={REF}  sha={SHA[:8]}  event={EVENT}")

    # ── Phase 0: Discord notification (push only) ───────────────────────────
    if EVENT == "push":
        print("\n=== Phase 0: Discord notify ===")
        _notify_discord()

    # ── Phase 1: Static analysis (parallel) ───────────────────────────────
    print("\n=== Phase 1: Static analysis ===")
    _run_parallel("Static analysis", [
        ("code-quality.yml",  {}, 600),
        ("preflight.yml",     {}, 1800),
    ])

    # ── Phase 2: Platforms + security audit (parallel) ─────────────────────
    print("\n=== Phase 2: Platforms + security ===")
    _run_parallel("Platforms + security", [
        ("ci.yml",             {"phase": "platforms", "orchestrator_sha": SHA, "orchestrator_run": RUN_ID}, 7200),
        ("security-audit.yml", None, 3600),
    ])

    print("\n::notice::CI orchestration complete — all phases passed")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n::error::{exc}")
        sys.exit(1)
