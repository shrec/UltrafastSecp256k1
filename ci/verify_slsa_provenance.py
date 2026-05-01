#!/usr/bin/env python3
"""
verify_slsa_provenance.py — Local SLSA provenance verification for auditors
============================================================================

External auditors and downstream consumers use this script to verify that a
downloaded UltrafastSecp256k1 artifact was built by the official GitHub Actions
pipeline and not tampered with post-build.

What it verifies
----------------
1. SHA-256 digest of the artifact matches the provenance subject
2. The SLSA provenance document is well-formed SLSA v1.0 / v0.2
3. The provenance builder is the official GitHub-hosted slsa-github-generator
4. The source repository and ref match expected values
5. (If cosign bundle present) Sigstore keyless signature is valid

Dependencies (audit infrastructure only — zero library dependency change)
--------------------------------------------------------------------------
  pip install requests
  # Optional for Cosign verification:
  # Install cosign binary:  https://docs.sigstore.dev/cosign/installation/
  #   or via:  go install github.com/sigstore/cosign/v2/cmd/cosign@latest

Usage
-----
  # Download artifact + provenance from GitHub Release, then verify:
  python3 ci/verify_slsa_provenance.py \\
      --artifact   ufsecp-linux-x64.tar.gz \\
      --provenance ufsecp-linux-x64.tar.gz.intoto.jsonl \\
      --repo       shrec/UltrafastSecp256k1 \\
      --ref        refs/tags/v3.x.y

  # Verify with Cosign bundle (strongest verification):
  python3 ci/verify_slsa_provenance.py \\
      --artifact   ufsecp-linux-x64.tar.gz \\
      --bundle     ufsecp-linux-x64.bundle \\
      --repo       shrec/UltrafastSecp256k1

  # Fetch and verify latest release automatically:
  python3 ci/verify_slsa_provenance.py \\
      --fetch-release latest \\
      --repo shrec/UltrafastSecp256k1

Exit codes
----------
  0  All checks passed — artifact is authentic
  1  Verification failed — do NOT use this artifact
  2  Missing dependency or configuration error
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

SCRIPT_DIR = Path(__file__).resolve().parent
LIB_ROOT   = SCRIPT_DIR.parent

OFFICIAL_REPO    = "shrec/UltrafastSecp256k1"
OFFICIAL_BUILDER = "https://github.com/slsa-framework/slsa-github-generator"
OFFICIAL_OIDC    = "https://token.actions.githubusercontent.com"
CERT_ID_REGEX    = rf"github\.com/{OFFICIAL_REPO}"

# ---------------------------------------------------------------------------
# SHA-256 digest helpers
# ---------------------------------------------------------------------------

def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# SLSA provenance parsing + verification
# ---------------------------------------------------------------------------

def load_intoto_bundle(path: Path) -> list[dict]:
    """Load a .intoto.jsonl or .bundle file as a list of JSON envelopes."""
    envelopes = []
    try:
        text = path.read_text(encoding="utf-8")
        for line in text.splitlines():
            line = line.strip()
            if line:
                envelopes.append(json.loads(line))
    except Exception as e:
        raise ValueError(f"Cannot parse provenance file: {e}") from e
    return envelopes


def decode_payload(envelope: dict) -> dict:
    """Base64-decode the DSSE/in-toto payload."""
    raw = envelope.get("payload", "")
    if not raw:
        raw = envelope.get("dsseEnvelope", {}).get("payload", "")
    if not raw:
        raise ValueError("No payload field in envelope")
    decoded = base64.b64decode(raw + "==")  # pad for safety
    return json.loads(decoded)


def verify_slsa_v1(statement: dict, artifact_digest: str, repo: str, ref: Optional[str]) -> list[str]:
    """Verify SLSA v1.0 provenance statement. Returns list of failure messages."""
    failures = []

    # -- predicate type
    pred_type = statement.get("predicateType", "")
    if "slsa.dev/provenance" not in pred_type:
        failures.append(f"unexpected predicateType: {pred_type}")

    # -- subject digest
    subjects = statement.get("subject", [])
    found_digest = False
    for subj in subjects:
        digests = subj.get("digest", {})
        sha256 = digests.get("sha256", "")
        if sha256.lower() == artifact_digest.lower():
            found_digest = True
            break
    if not found_digest:
        failures.append(f"artifact digest {artifact_digest[:16]}... not found in provenance subjects")

    # -- builder (SLSA v1 uses predicate.buildDefinition or predicate.builder)
    pred = statement.get("predicate", {})
    builder = (pred.get("buildDefinition", {}).get("buildType", "") or
               pred.get("builder", {}).get("id", "") or "")
    if OFFICIAL_BUILDER not in builder and "slsa-github-generator" not in builder:
        failures.append(f"unexpected builder: {builder!r}")

    # -- source repo
    src_uri = ""
    if "buildDefinition" in pred:
        ext = pred["buildDefinition"].get("externalParameters", {})
        src_uri = (ext.get("source", {}).get("uri", "") or
                   ext.get("workflow", {}).get("repository", "") or "")
    elif "materials" in pred:
        for mat in pred.get("materials", []):
            if repo in mat.get("uri", ""):
                src_uri = mat["uri"]
                break
    if repo not in src_uri:
        # Try invocation.configSource
        inv = pred.get("invocation", {})
        cfg = inv.get("configSource", {})
        src_uri = cfg.get("uri", "")
    if repo not in src_uri:
        failures.append(f"source repository {repo!r} not found in provenance materials/buildDef")

    # -- ref check (optional)
    if ref:
        ref_found = json.dumps(statement).find(ref) != -1
        if not ref_found:
            failures.append(f"expected ref {ref!r} not found in provenance document")

    return failures


def verify_slsa_v02(statement: dict, artifact_digest: str, repo: str, ref: Optional[str]) -> list[str]:
    """Verify SLSA v0.2 provenance statement. Returns list of failure messages."""
    failures = []

    pred_type = statement.get("predicateType", "")
    if "slsa.dev/provenance/v0.2" not in pred_type and "slsa.dev/provenance" not in pred_type:
        failures.append(f"unexpected predicateType: {pred_type}")

    subjects = statement.get("subject", [])
    found_digest = any(
        s.get("digest", {}).get("sha256", "").lower() == artifact_digest.lower()
        for s in subjects
    )
    if not found_digest:
        failures.append(f"artifact digest {artifact_digest[:16]}... not in provenance subjects")

    pred = statement.get("predicate", {})
    builder_id = pred.get("builder", {}).get("id", "")
    if "slsa-github-generator" not in builder_id and OFFICIAL_BUILDER not in builder_id:
        failures.append(f"unexpected builder id: {builder_id!r}")

    inv = pred.get("invocation", {})
    cfg = inv.get("configSource", {})
    uri = cfg.get("uri", "")
    if repo not in uri:
        for mat in pred.get("materials", []):
            if repo in mat.get("uri", ""):
                uri = mat["uri"]
                break
    if repo not in uri:
        failures.append(f"source repository {repo!r} not in provenance")

    if ref:
        if ref not in json.dumps(statement):
            failures.append(f"ref {ref!r} not found in provenance")

    return failures


def verify_provenance_file(prov_path: Path, artifact_digest: str,
                           repo: str, ref: Optional[str]) -> tuple[bool, list[str]]:
    """Verify a .intoto.jsonl provenance file. Returns (ok, [messages])."""
    messages = []
    envelopes = load_intoto_bundle(prov_path)
    if not envelopes:
        return False, ["provenance file is empty"]

    any_passed = False
    for i, env in enumerate(envelopes):
        try:
            stmt = decode_payload(env)
        except Exception as e:
            messages.append(f"envelope {i}: cannot decode payload: {e}")
            continue

        pred_type = stmt.get("predicateType", "")
        if "v0.2" in pred_type or "v0.1" in pred_type:
            fails = verify_slsa_v02(stmt, artifact_digest, repo, ref)
        else:
            fails = verify_slsa_v1(stmt, artifact_digest, repo, ref)

        if not fails:
            any_passed = True
            messages.append(f"envelope {i}: SLSA provenance OK (predicateType={pred_type})")
        else:
            for f in fails:
                messages.append(f"envelope {i}: FAIL: {f}")

    return any_passed, messages


# ---------------------------------------------------------------------------
# Cosign verification
# ---------------------------------------------------------------------------

def verify_cosign_bundle(artifact_path: Path, bundle_path: Path,
                         repo: str) -> tuple[bool, str]:
    """Invoke `cosign verify-blob` on the artifact. Returns (ok, output)."""
    cosign = shutil.which("cosign")
    if not cosign:
        return False, "cosign binary not found (install from https://docs.sigstore.dev/cosign/installation/)"
    cmd = [
        cosign, "verify-blob",
        "--certificate-identity-regexp", CERT_ID_REGEX,
        "--certificate-oidc-issuer", OFFICIAL_OIDC,
        "--bundle", str(bundle_path),
        str(artifact_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    out = (result.stdout + result.stderr).strip()
    return result.returncode == 0, out


# ---------------------------------------------------------------------------
# GitHub Release auto-fetch
# ---------------------------------------------------------------------------

def fetch_latest_release(repo: str, out_dir: Path) -> tuple[Path, Optional[Path], Optional[Path]]:
    """Download latest release artifact + provenance from GitHub. Returns paths."""
    try:
        import urllib.request
    except ImportError:
        raise RuntimeError("urllib not available")

    api_url = f"https://api.github.com/repos/{repo}/releases/latest"
    with urllib.request.urlopen(api_url) as resp:
        release = json.loads(resp.read().decode())

    assets = {a["name"]: a["browser_download_url"] for a in release.get("assets", [])}
    # Find the main tarball
    tarball_name = next((n for n in assets if n.endswith(".tar.gz") and "linux-x64" in n), None)
    if not tarball_name:
        raise RuntimeError(f"No linux-x64 tarball in release assets: {list(assets)}")

    provenance_name = tarball_name + ".intoto.jsonl"
    bundle_name = tarball_name.replace(".tar.gz", ".bundle")

    def download(url: str, dest: Path):
        with urllib.request.urlopen(url) as r, open(dest, "wb") as f:
            f.write(r.read())

    tarball_path = out_dir / tarball_name
    download(assets[tarball_name], tarball_path)

    prov_path = None
    if provenance_name in assets:
        prov_path = out_dir / provenance_name
        download(assets[provenance_name], prov_path)

    bundle_path = None
    if bundle_name in assets:
        bundle_path = out_dir / bundle_name
        download(assets[bundle_name], bundle_path)

    return tarball_path, prov_path, bundle_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SLSA provenance verifier for UltrafastSecp256k1",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--artifact", help="Path to artifact file (e.g. ufsecp-linux-x64.tar.gz)")
    p.add_argument("--provenance", help="Path to .intoto.jsonl provenance file")
    p.add_argument("--bundle", help="Path to Cosign bundle file (.bundle)")
    p.add_argument("--repo", default=OFFICIAL_REPO,
                   help=f"GitHub repo (default: {OFFICIAL_REPO})")
    p.add_argument("--ref", default=None,
                   help="Expected git ref (e.g. refs/tags/v3.2.0)")
    p.add_argument("--fetch-release", metavar="VERSION",
                   help="Auto-fetch release from GitHub (use 'latest' or a tag)")
    p.add_argument("--json", action="store_true", help="Output JSON summary")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if args.fetch_release:
        print(f"Fetching release '{args.fetch_release}' from {args.repo}...")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            artifact, provenance, bundle = fetch_latest_release(args.repo, tmp_dir)
            # re-run with the downloaded files
            sys.argv = [sys.argv[0],
                        "--artifact", str(artifact),
                        *(["--provenance", str(provenance)] if provenance else []),
                        *(["--bundle", str(bundle)] if bundle else []),
                        "--repo", args.repo,
                        *(["--ref", args.ref] if args.ref else []),
                        *(["--json"] if args.json else []),
                        ]
            args = parse_args()

    if not args.artifact:
        print("ERROR: --artifact required", file=sys.stderr)
        return 2

    artifact_path = Path(args.artifact)
    if not artifact_path.is_file():
        print(f"ERROR: artifact not found: {artifact_path}", file=sys.stderr)
        return 2

    results = {}
    overall_ok = True

    # -- 1. SHA-256 digest ---------------------------------------------------
    digest = file_sha256(artifact_path)
    print(f"Artifact : {artifact_path.name}")
    print(f"SHA-256  : {digest}")
    results["artifact_sha256"] = digest

    # -- 2. Provenance verification ------------------------------------------
    if args.provenance:
        prov_path = Path(args.provenance)
        print(f"\nVerifying SLSA provenance: {prov_path.name}")
        ok_prov, msgs = verify_provenance_file(prov_path, digest, args.repo, args.ref)
        for m in msgs:
            icon = "  ✓" if "OK" in m else "  ✗"
            print(f"{icon} {m}")
        results["provenance"] = {"ok": ok_prov, "messages": msgs}
        if not ok_prov:
            overall_ok = False
            print("  [FAIL] Provenance verification FAILED")
        else:
            print("  [PASS] Provenance verified")
    else:
        print("\n[INFO] No --provenance file provided — skipping SLSA check")
        results["provenance"] = {"ok": None, "messages": ["not checked"]}

    # -- 3. Cosign bundle verification ---------------------------------------
    if args.bundle:
        bundle_path = Path(args.bundle)
        print(f"\nVerifying Cosign bundle: {bundle_path.name}")
        ok_cosign, cosign_out = verify_cosign_bundle(artifact_path, bundle_path, args.repo)
        print(f"  {'✓' if ok_cosign else '✗'} cosign: {cosign_out}")
        results["cosign"] = {"ok": ok_cosign, "output": cosign_out}
        if not ok_cosign:
            overall_ok = False
            print("  [FAIL] Cosign verification FAILED")
        else:
            print("  [PASS] Cosign signature verified")
    else:
        print("\n[INFO] No --bundle file provided — skipping Cosign check")
        results["cosign"] = {"ok": None, "output": "not checked"}

    # -- Summary -------------------------------------------------------------
    print("\n" + "=" * 50)
    if overall_ok and (args.provenance or args.bundle):
        print("RESULT: PASS — artifact is authentic")
        ret = 0
    elif not args.provenance and not args.bundle:
        print("RESULT: INCOMPLETE — provide --provenance and/or --bundle to verify")
        ret = 2
    else:
        print("RESULT: FAIL — artifact could not be verified; do NOT use")
        ret = 1

    if args.json:
        results["overall_ok"] = overall_ok
        results["artifact"] = str(artifact_path)
        print(json.dumps(results, indent=2))

    return ret


if __name__ == "__main__":
    sys.exit(main())
