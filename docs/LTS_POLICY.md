# Long-Term Support (LTS) Policy

**UltrafastSecp256k1** -- Version Lifecycle & Support Guarantees

---

## 1. Version Scheme

UltrafastSecp256k1 uses **Semantic Versioning 2.0.0** (`MAJOR.MINOR.PATCH`):

| Component | When Bumped |
|-----------|-------------|
| `MAJOR` | Breaking API/ABI changes |
| `MINOR` | New features, backward-compatible |
| `PATCH` | Bug fixes, security patches |

---

## 2. Release Cadence

| Release Type | Frequency | Notes |
|-------------|-----------|-------|
| **Minor** | Every 2-4 months | New features, performance improvements |
| **Patch** | As needed | Bug fixes, security patches |
| **Security** | Within 30 days of disclosure | Critical vulnerability fixes |
| **Major** | Rare (12+ months apart) | Only for breaking changes |

---

## 3. Support Tiers

### 3.1 Active Support

- Latest stable release (current `MINOR`)
- Receives: all bug fixes, security patches, performance improvements
- Duration: Until next `MINOR` release + 30-day grace period

### 3.2 LTS Support

- Designated LTS releases (announced at release time)
- Receives: **security patches and critical bug fixes only**
- Duration: **12 months** from release date
- No new features backported

### 3.3 Critical-Only Support

- Penultimate stable release (one `MINOR` behind current)
- Receives: **critical security patches only** (CVSS >= 9.0)
- Duration: Until superseded by two `MINOR` releases

### 3.4 End of Life (EOL)

- No longer receives any updates
- Users must migrate to a supported version
- EOL announced at least 30 days in advance

---

## 4. LTS Designation

Not all releases are LTS. A release is designated LTS when:

1. It represents a stable, well-tested state of the library
2. Significant production adoption has occurred
3. Maintainers commit to the 12-month support window
4. Announced in release notes with `[LTS]` tag

### 4.1 LTS Versioning

LTS patches follow `MAJOR.MINOR.PATCH` where only `PATCH` increments:
- `v4.0.0 [LTS]` -> `v4.0.1` -> `v4.0.2` -> ... (security/critical fixes)

### 4.2 Current LTS Schedule

| Version | Release Date | LTS Status | EOL Date |
|---------|-------------|------------|----------|
| v3.14.x | 2025-02 | Active (current) | Until v3.15.0 + 30 days |
| v4.0.0 | Planned | Candidate for first LTS | TBD |

---

## 5. Support Matrix

| Version | Status | Receives |
|---------|--------|----------|
| Latest minor (e.g., v3.15.x) | Active | All fixes |
| Previous minor (e.g., v3.14.x) | Critical-only | CVSS >= 9.0 only |
| Designated LTS (e.g., v4.0.x) | LTS | Security + critical fixes for 12 months |
| Older versions | EOL | No updates |

---

## 6. ABI Stability Within Support Window

### 6.1 Guarantees

- **Within a MINOR series** (e.g., v3.14.0 -> v3.14.5): Full ABI compatibility. No function signatures change. `UFSECP_ABI_VERSION` does not change.
- **Between MINOR versions** (e.g., v3.14 -> v3.15): ABI-compatible additions only. New functions may be added. Existing signatures preserved.
- **Between MAJOR versions** (e.g., v3.x -> v4.0): ABI may break. Migration guide provided.

### 6.2 LTS ABI Lock

LTS versions have a **frozen ABI**:
- No functions added or removed
- No signature changes
- No struct layout changes
- Only implementation bug fixes

---

## 7. Migration Path

### 7.1 Minor Version Migration

```
v3.14.x -> v3.15.x
- Relink with new library (ABI compatible)
- Check CHANGELOG for deprecated APIs
- Test suite should pass unchanged
```

### 7.2 Major Version Migration

```
v3.x -> v4.0
- Read MIGRATION_GUIDE.md
- Update deprecated function calls (removed in MAJOR)
- Recompile all code linking UltrafastSecp256k1
- Run full test suite
```

---

## 8. Deprecation Integration

Deprecation follows the policy in [DEPRECATION_POLICY.md](DEPRECATION_POLICY.md):

1. Feature deprecated in version `vX.Y.0` with compile-time warning
2. Remains functional for at least 2 minor releases
3. Removed no earlier than `vX.(Y+2).0`
4. LTS versions never have features removed (warnings only)

---

## 9. Platform Support Lifecycle

Platforms follow the same lifecycle as features:

| Platform tier | Minimum support duration |
|--------------|-------------------------|
| Tier 1 (x86-64, ARM64) | Full lifecycle of release |
| Tier 2 (RISC-V, WASM, CUDA) | Best-effort; may drop in MINOR with deprecation notice |
| Tier 3 (ESP32, STM32, Metal) | Community-maintained; no SLA |

---

## 10. Communication

- **Release notes**: Published on GitHub Releases for every version
- **Security advisories**: GitHub Security Advisories + SECURITY.md
- **EOL notices**: Posted 30 days before EOL in GitHub Discussions
- **LTS designations**: Announced in release notes with `[LTS]` tag
- **Migration guides**: Published in `docs/` for MAJOR version bumps

---

*Policy version: 1.0*  
*Effective date: 2026-02-24*
