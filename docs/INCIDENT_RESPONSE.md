# Incident Response Runbook -- UltrafastSecp256k1

This document defines the process for handling security vulnerabilities
and critical bugs reported against UltrafastSecp256k1.

---

## 1. Triage (T+0 to T+24h)

1. **Acknowledge** receipt to reporter within 24 hours via the channel
   described in `SECURITY.md` (security@... or GitHub Security Advisory).
2. **Classify** severity using CVSS 3.1:
   - **Critical (9.0-10.0):** Secret key leak, signature forgery, RCE.
   - **High (7.0-8.9):** Side-channel with practical exploit, memory corruption.
   - **Medium (4.0-6.9):** Denial-of-service, non-exploitable UB.
   - **Low (0.1-3.9):** Informational, documentation-only issues.
3. **Assign** an owner from the `@shrec/crypto-core` CODEOWNERS team.
4. **Open** a private GitHub Security Advisory (GHSA) draft with:
   - Affected versions, components, and platforms.
   - Reproduction steps (if available).
   - Preliminary CVSS score.

## 2. Fix Development (T+24h to T+7d for Critical/High)

| Severity | Target Fix Time | Target Disclosure |
|----------|----------------|-------------------|
| Critical | 48 hours       | 7 days            |
| High     | 7 days         | 14 days           |
| Medium   | 30 days        | 60 days           |
| Low      | Next release   | Next release      |

1. **Branch** from `main` into a private fork or security advisory branch.
2. **Implement** the fix following project coding rules:
   - No math changes without explicit approval.
   - All CT rules apply if the fix touches secret-dependent paths.
3. **Add regression test** proving the vulnerability is fixed:
   - Wycheproof vector if applicable.
   - Fuzz corpus entry if applicable.
4. **Verify** full test suite passes (31+ tests):
   ```bash
   ctest --test-dir build -C Release --output-on-failure
   ```
5. **Run CT verification** if side-channel related:
   ```bash
   cmake -S . -B build-ct -DSECP256K1_CT_VALGRIND=1
   cmake --build build-ct
   ctest --test-dir build-ct -R ct_verif_formal
   ```

## 3. Advisory & Disclosure

1. **Write CVE description** with:
   - Vulnerability summary (1-2 sentences).
   - Affected versions (semver range).
   - Root cause analysis (which function, which layer).
   - Mitigation for users who cannot upgrade immediately.
2. **Request CVE** through GitHub Security Advisory (auto-assigns CNA).
3. **Coordinate disclosure date** with reporter (respect their timeline).
4. **Prepare release notes** with security section:
   ```
   ## Security
   - **CVE-YYYY-NNNNN** (Severity): <description>. Fixed in v<X.Y.Z>.
   ```

## 4. Release & Backport

1. **Tag release** with fix:
   ```bash
   git tag -s vX.Y.Z -m "security: fix CVE-YYYY-NNNNN"
   git push origin vX.Y.Z
   ```
2. **Backport** to supported branches if the fix applies:
   - Cherry-pick the fix commit.
   - Run full test suite on backport branch.
   - Tag a patch release (e.g., vX.Y-1.Z+1).
3. **Publish** GitHub Security Advisory (transitions from draft to published).
4. **Update** `SECURITY.md` supported-versions table if needed.
5. **Notify** downstream users via:
   - GitHub Advisory notification (automatic for dependents).
   - Release notes on GitHub Releases page.

## 5. Post-Incident Review (T+30d)

1. **Root cause** document: how did the bug enter the codebase?
2. **Detection gap**: why didn't tests/CI/audit catch it?
3. **Process improvement**: add missing test category, CI gate, or review rule.
4. **Update** `THREAT_MODEL.md` if the threat model needs revision.
5. **Update** `AUDIT_REPORT.md` with post-fix audit results.

---

## 6. Periodic Drills

Automated incident response drills run as preflight step [20/20] and validate
triage readiness, fix-time targets, and disclosure process completeness.

```bash
python3 ci/incident_drills.py --json
```

Three drill scenarios are exercised:

| Drill | What it validates |
|-------|------------------|
| `key_compromise` | Triage latency, key-revocation path via `_ufsecp.py` |
| `ci_poisoning` | CI workflow hardening, pinned action verification |
| `dependency_compromise` | Lock-file presence, dependency-review gate |

Drill results feed into the Security Autonomy orchestrator
(`ci/security_autonomy_check.py`) and are tracked in
`docs/SECURITY_AUTONOMY_KPI.json`.

---

## Contacts

| Role              | Handle/Email                    |
|-------------------|---------------------------------|
| Primary Maintainer | @shrec                         |
| Crypto Core Team  | @shrec/crypto-core (CODEOWNERS) |
| Security Reports  | See `SECURITY.md`               |

## Version History

| Date       | Change                          |
|------------|---------------------------------|
| 2025-01-XX | Initial runbook created         |
| 2026-04-14 | Added Section 6: Periodic Drills (`ci/incident_drills.py`) |
