#!/usr/bin/env python3
"""
check_tag_conformance.py — systemic guard for EVERY production tagged-hash tag string.

WHY (the bug class this prevents):
A BIP-340-style tagged hash derives a domain-separated value from a fixed tag string.
If the tag string diverges from the canonical spec value, the implementation stays
SELF-CONSISTENT (its own sign and verify agree, so roundtrip / self tests pass) but
silently DIVERGES from the standard — every external/cross-implementation interaction
fails. This exact failure shipped twice and was only caught by external differential
testing:
  * MuSig2 binding factor used "MuSig/nonceblinding" instead of BIP-327 "MuSig/noncecoef"
  * GPU Bulletproof range-prove used "BP/y|z|x|ip" instead of "Bulletproof/y|z|x|ip"

This gate makes the failure mode impossible to merge: it scans every production
tagged-hash tag literal across ALL backends (CPU + CUDA + OpenCL + Metal) and requires
each to be in the CANONICAL registry below. A misspelled or unknown tag — exactly what
"MuSig/nonceblinding" was — fails the gate at commit time, before any external test runs.

Adding a NEW legitimate tag requires adding it here (with its spec source), which forces
a deliberate review against the authoritative specification. That is the point.

Scope: production source only (src/cpu, src/cuda, src/opencl, src/metal), excluding
bench_*/test_* files and tests/ dirs. Exit 0 = clean, exit 1 = violation.
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCAN_DIRS = ("src/cpu", "src/cuda", "src/opencl", "src/metal")
EXTS = (".cpp", ".hpp", ".h", ".cu", ".cuh", ".cl", ".metal", ".inc")

# ── Canonical tag registry ───────────────────────────────────────────────────
# Every production tagged-hash tag, with its authoritative source. Membership here
# IS the conformance assertion: a tag not listed is rejected (unknown/typo).
CANONICAL_TAGS = {
    # BIP-340 Schnorr (bitcoin/bips bip-0340 + libsecp256k1 schnorrsig)
    "BIP0340/aux": "BIP-340",
    "BIP0340/nonce": "BIP-340",
    "BIP0340/challenge": "BIP-340",
    # Engine-specific message-prehash helper (NOT a raw BIP-340 tag — used only by the
    # *_msg convenience APIs; never on the raw BIP-340 interop path). Intentional.
    "BIP0340/msg": "engine (ufsecp_schnorr_*_msg prehash)",
    # BIP-341 Taproot (bitcoin/bips bip-0341)
    "TapTweak": "BIP-341",
    "TapLeaf": "BIP-341",
    "TapBranch": "BIP-341",
    "TapSighash": "BIP-341",
    # BIP-327 MuSig2 (bitcoin/bips bip-0327 reference.py + libsecp256k1 musig)
    "KeyAgg list": "BIP-327",
    "KeyAgg coefficient": "BIP-327",
    "MuSig/aux": "BIP-327",
    "MuSig/nonce": "BIP-327",
    "MuSig/noncecoef": "BIP-327",  # NOT "MuSig/nonceblinding" — see module docstring
    # BIP-322 generic signed message (bitcoin/bips bip-0322)
    "BIP0322-signed-message": "BIP-322",
    # BIP-352 Silent Payments (bitcoin/bips bip-0352 reference.py)
    "BIP0352/SharedSecret": "BIP-352",
    "BIP0352/Inputs": "BIP-352",
    "BIP0352/Label": "BIP-352",  # registered ahead of implementation (additive roadmap)
    # BIP-324 v2 transport (bitcoin/bips bip-0324 + libsecp256k1 ellswift)
    "bip324_ellswift_xonly_ecdh": "BIP-324",
    # Engine ZK (Bulletproofs / DLEQ — engine-internal Fiat-Shamir domain tags;
    # canonical = these exact strings, shared by prover and verifier on every backend)
    "Bulletproof/A": "engine ZK (Bulletproof)",
    "Bulletproof/S": "engine ZK (Bulletproof)",
    "Bulletproof/y": "engine ZK (Bulletproof)",
    "Bulletproof/z": "engine ZK (Bulletproof)",
    "Bulletproof/x": "engine ZK (Bulletproof)",
    "Bulletproof/ip": "engine ZK (Bulletproof)",
    "Bulletproof/gen": "engine ZK (Bulletproof)",
    "ZK/nonce": "engine ZK (DLEQ)",
    "ZK/knowledge": "engine ZK (DLEQ)",
    "ZK/dleq": "engine ZK (DLEQ)",
    "ZK/dleq/H": "engine ZK (DLEQ over H)",
    # Engine FROST binding (engine-internal)
    "FROST_binding": "engine FROST",
    # Engine adaptor signatures (ECDSA/Schnorr adaptor — engine-internal Fiat-Shamir
    # domain tags; secret-nonce derivation + the GHSA-c7q2 DLEQ challenge/nonce, all
    # distinct). Declared as `constexpr char domain[] = "..."` and passed by variable
    # to SHA256::hash, so they are surfaced by DECL_PATTERN below, not the call-site
    # PATTERNS.
    "adaptor_nonce_v1": "engine adaptor (Schnorr/ECDSA adaptor nonce)",
    "ufsecp/ecdsa_adaptor_dleq_v1": "engine adaptor (ECDSA adaptor DLEQ challenge, GHSA-c7q2)",
    "ufsecp/ecdsa_adaptor_dleq_nonce_v1": "engine adaptor (ECDSA adaptor DLEQ nonce, GHSA-c7q2)",
    # Coin-specific silent-payment variants (engine LTC/BCH silent payments)
    "LTCSP/SharedSecret": "engine LTC silent payments",
    "BCH/SharedSecret": "engine BCH silent payments",
}

# Known-wrong variants → the correct value. These have actually shipped; name them
# explicitly so the gate prints an actionable message rather than a generic "unknown".
BANNED_TAGS = {
    "MuSig/nonceblinding": "MuSig/noncecoef",   # BIP-327 binding-factor tag (shipped wrong, fixed ac8c7411)
    "BP/y": "Bulletproof/y",                      # GPU range-prove (shipped wrong, fixed 8f5915c5)
    "BP/z": "Bulletproof/z",
    "BP/x": "Bulletproof/x",
    "BP/ip": "Bulletproof/ip",
    "MuSig/noncecoeff": "MuSig/noncecoef",        # plausible double-f typo
    "TapSigHash": "TapSighash",                   # plausible casing typo
}

# Tag-literal call sites + domain-tag byte arrays.
PATTERNS = [
    re.compile(r'SECP256K1_MIDSTATE\s*\([^,]+,\s*"([^"]+)"'),
    re.compile(r'make_tag_midstate\s*\(\s*"([^"]+)"'),
    re.compile(r'\btagged_hash\w*\s*\(\s*"([^"]+)"'),
    re.compile(r'\bsha256_tagged\w*\s*\(\s*"([^"]+)"'),
    re.compile(r'\bzk_tagged_hash\w*\s*\(\s*"([^"]+)"'),
    re.compile(r'SHA256::hash\s*\(\s*reinterpret_cast<[^>]+>\s*\(\s*"([^"]+)"'),
]
BYTEARRAY = re.compile(r"\{\s*('(?:[^']|\\.)'(?:\s*,\s*'(?:[^']|\\.)'){2,})\s*\}")
# `constexpr char domain[] = "tag"` / `static const char tag[] = "tag"` declarations
# whose array is later passed by variable to SHA256::hash (the adaptor tags do this,
# so the call-site PATTERNS above never see the literal). Filtered through
# DOMAIN_PREFIX so unrelated char-array string literals are not flagged.
DECL_PATTERN = re.compile(r'\bchar\s+\w+\s*\[\s*\]\s*=\s*"([^"]+)"')
DOMAIN_PREFIX = re.compile(r"^(BIP0|MuSig|KeyAgg|Tap|Bulletproof|BP/|ZK/|FROST|bip324|LTCSP|BCH/|adaptor|ufsecp/ecdsa_adaptor)")


def decode_bytearray(seg: str) -> str:
    out = []
    for c in re.findall(r"'([^']|\\.)'", seg):
        if c.startswith("\\"):
            try:
                out.append(c.encode().decode("unicode_escape"))
            except Exception:
                out.append(c)
        else:
            out.append(c)
    return "".join(out)


def is_excluded(path: str, fname: str) -> bool:
    if any(x in path for x in (os.sep + "bench", os.sep + "tests", os.sep + "test")):
        return True
    return fname.startswith(("bench_", "test_"))


def main() -> int:
    found = {}  # tag -> list of "rel:line"
    for d in SCAN_DIRS:
        base = os.path.join(ROOT, d)
        if not os.path.isdir(base):
            continue
        for dp, _, files in os.walk(base):
            for f in files:
                if not f.endswith(EXTS) or is_excluded(dp, f):
                    continue
                p = os.path.join(dp, f)
                try:
                    text = open(p, errors="replace").read()
                except OSError:
                    continue
                for n, line in enumerate(text.split("\n"), 1):
                    cands = []
                    for pat in PATTERNS:
                        cands += pat.findall(line)
                    for m in BYTEARRAY.finditer(line):
                        s = decode_bytearray(m.group(1))
                        if DOMAIN_PREFIX.match(s):
                            cands.append(s)
                    for m in DECL_PATTERN.finditer(line):
                        s = m.group(1)
                        if DOMAIN_PREFIX.match(s):
                            cands.append(s)
                    for tag in cands:
                        found.setdefault(tag, []).append(f"{os.path.relpath(p, ROOT)}:{n}")

    banned_hits, unknown_hits = [], []
    for tag, locs in sorted(found.items()):
        if tag in BANNED_TAGS:
            banned_hits.append((tag, BANNED_TAGS[tag], locs))
        elif tag not in CANONICAL_TAGS:
            unknown_hits.append((tag, locs))

    if banned_hits or unknown_hits:
        print("FAIL: tagged-hash tag-conformance violation(s).")
        for tag, correct, locs in banned_hits:
            print(f'  BANNED tag "{tag}" -> use "{correct}"')
            for l in locs[:6]:
                print(f"      {l}")
        for tag, locs in unknown_hits:
            print(f'  UNKNOWN tag "{tag}" — not in the canonical registry.')
            print("      If this is a new, spec-verified tag, add it to CANONICAL_TAGS in")
            print("      ci/check_tag_conformance.py with its authoritative source. If it is a")
            print("      typo/divergence (e.g. the MuSig/nonceblinding class), fix it.")
            for l in locs[:6]:
                print(f"      {l}")
        return 1

    print(f"OK: {len(found)} production tagged-hash tag(s) all canonical "
          f"({len(CANONICAL_TAGS)} registered).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
