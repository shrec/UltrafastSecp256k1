# Evidence Chain Key Policy

> Status: 2026-04-21 — current state plus honest threat-model and the
> rotation/escrow procedure for any future move to a secret-grade HMAC key.

## Current State (Honest Description)

The evidence chain in
[docs/EVIDENCE_CHAIN.json](EVIDENCE_CHAIN.json) is signed using a constant
HMAC key embedded in `ci/evidence_governance.py`:

```python
# HMAC key derived from repository identity (not a secret — tamper detection only)
_HMAC_KEY = b"ufsecp-evidence-chain-v1"
```

**This key is intentionally public.** It exists to detect *accidental*
tampering of evidence records (truncation, partial corruption, replay across
incompatible schemas). It is **not** a defence against an attacker with write
access to the repository. Anyone with the source tree can re-sign any record.

This is by design and consistent with `ci/evidence_governance.py` line
33's own comment. Past audit-manifest claims that the chain is
"HMAC-verified tamper-resistant" must be read with this scope in mind:
the chain is tamper-*evident* against accidental drift, not tamper-*resistant*
against a hostile committer.

## Threat Model

| Threat | Currently mitigated? | By what |
|--------|---------------------|---------|
| Accidental record corruption | Yes | constant-key HMAC verify |
| Schema-drift across versions | Yes | record version + check |
| Hostile rewrite by repo committer | **No** | requires secret key, see below |
| Hostile rewrite by external attacker | Indirect | repo branch protection + signed commits + Scorecard policy |
| Replay of an old chain segment | Partial | timestamps + git SHA pinning per record |

## When To Promote To A Secret Key

Promote the HMAC key from public-constant → repository secret only when one of
the following is true:

1. The chain is consumed by an external auditor who needs cryptographic proof
   that the maintainer did not rewrite the chain after the fact.
2. The repo moves to a multi-party governance model where some committers
   should not be able to forge chain records.
3. A specific compliance frame (e.g. SOC2 evidence retention) requires a
   secret-bound integrity proof.

Until then, raising the bar from public-constant to secret would create the
illusion of a stronger control without changing who can actually rewrite
the chain (which is governed by repo write access).

## Rotation Procedure (For The Day The Key Becomes Secret)

1. **Generate** a new 32-byte random key:

   ```bash
   python3 -c 'import secrets; print(secrets.token_hex(32))'
   ```

2. **Escrow** the new key:

   - Primary copy: GitHub Actions repository secret
     `EVIDENCE_HMAC_KEY` (encrypted at rest, accessible only to workflows
     listed in `.github/workflows/caas*.yml`).
   - Backup copy: offline owner-controlled medium (printed paper, hardware
     token, or 1Password vault item `ufsecp/evidence_hmac_key`).
   - Never commit the key to the repository.

3. **Re-sign** the existing chain in place:

   ```bash
   EVIDENCE_HMAC_KEY=<new-key> \
   EVIDENCE_HMAC_KEY_OLD=<old-key> \
   python3 ci/rotate_evidence_key.py
   ```

   The script reads each record, verifies it under the old key, re-signs it
   under the new key, appends a `key_rotation` event to the chain, and
   atomically swaps the file via `os.replace`.

4. **Activate** the new key:

   - Update the workflow secret.
   - Push a new repo tag `evidence-key-v<N+1>` pointing at the rotation
     commit so external auditors can pin the new key generation.

5. **Verify** post-rotation:

   ```bash
   EVIDENCE_HMAC_KEY=<new-key> \
   python3 ci/evidence_governance.py validate --json
   ```

   Expected: `overall_pass=True`, last record is the `key_rotation` event.

6. **Notify**: write a dated entry into
   [AUDIT_CHANGELOG.md](AUDIT_CHANGELOG.md) of the form
   "Evidence HMAC key rotated to v<N+1> — old key retained for verifier
   convenience for 30 days, then destroyed".

## Compromise Playbook

If the secret key is suspected compromised:

1. Treat all chain records signed under the compromised key as
   **untrusted** until re-verified by hash against external evidence.
2. Run rotation immediately (steps 1–5 above).
3. File an incident drill record via
   `python3 ci/incident_drills.py --drill evidence_key_compromise --record`.
4. Review the previous 30 days of chain records against any external mirror
   of the bundle (`docs/EXTERNAL_AUDIT_BUNDLE.json` is timestamp-pinned and
   externally verifiable independent of the chain key).

## Implementation Status

- [ ] `ci/rotate_evidence_key.py` — to be added when promotion is needed.
- [ ] `incident_drills.py` `evidence_key_compromise` drill — to be added when
      promotion is needed.
- [x] This policy document exists and accurately describes the current state.

The pieces above are intentionally *not* shipped today, because today the key
is public-by-design. Shipping a "rotation" of a public key would be theatre.

## Audit Manifest Cross-Reference

This policy supports `AUDIT_MANIFEST.md` P16 (Evidence Governance) by making
the actual scope of the integrity guarantee explicit, instead of leaving the
"HMAC-verified" wording to imply secret-grade protection.

See also:
- [docs/CAAS_HARDENING_TODO.md](CAAS_HARDENING_TODO.md) item H-2.
- `ci/evidence_governance.py`.
