# Pre-Release Checklist

**UltrafastSecp256k1** -- Mandatory Steps Before Any Release

---

## Instructions

Copy this checklist into the release PR description. All items must be checked before merge to `main`.

---

## Checklist

### 1. Version & Metadata

- [ ] `VERSION.txt` updated to new version
- [ ] `CHANGELOG.md` updated with all changes since last release
- [ ] API version in `ufsecp.h` (`UFSECP_ABI_VERSION`) bumped if ABI changed
- [ ] Copyright year current in `LICENSE`
- [ ] `SECURITY.md` supported versions table updated

### 2. Build Verification

- [ ] Builds clean on Linux (Clang/GCC): `cmake --build build -j`
- [ ] Builds clean on Windows (MSVC/Clang-cl)
- [ ] Builds clean on macOS (Apple Clang)
- [ ] No new compiler warnings with `-Wall -Wextra -Wpedantic -Wconversion`
- [ ] No new Clang-Tidy findings
- [ ] WASM build succeeds (if applicable)
- [ ] CUDA build succeeds with target architectures (if applicable)

### 3. Test Suite

- [ ] `ctest --output-on-failure` -- ALL tests pass
- [ ] `test_field_audit` -- 641K+ checks, 0 failures
- [ ] `test_bip340_vectors` -- all 15 vectors pass
- [ ] `test_rfc6979_vectors` -- all 6 nonce/sign vectors pass
- [ ] `test_bip32_vectors` -- 90 checks, 0 failures
- [ ] `test_cross_libsecp256k1` -- 7860 differential checks pass
- [ ] `test_ecc_properties` -- group law properties pass
- [ ] `test_musig2_frost` -- 975 checks pass
- [ ] `test_musig2_frost_advanced` -- 316 checks pass
- [ ] `test_fuzz_parsers` -- 580K+ checks, 0 failures
- [ ] `test_fuzz_address_bip32_ffi` -- 73K+ checks, 0 failures, 0 crashes
- [ ] `ct_sidechannel_smoke` -- dudect pass (t < threshold)

### 4. Security Checks

- [ ] CodeQL -- no new critical/high findings
- [ ] SonarCloud -- no new bugs, vulnerabilities, or code smells
- [ ] Dependency review -- no known vulnerable dependencies
- [ ] ASan build + test: no memory errors
- [ ] UBSan build + test: no undefined behavior
- [ ] TSan build + test: no data races
- [ ] Valgrind memcheck: no leaks or invalid reads/writes

### 5. ABI Compatibility (if applicable)

- [ ] No functions removed from `ufsecp.h` public API
- [ ] No function signature changes to existing public API
- [ ] New functions added with ABI version guard
- [ ] Bindings (Python/Rust/Go/C#) updated for new functions
- [ ] `ufsecp_abi_version()` returns correct value

### 6. Documentation

- [ ] `docs/API_REFERENCE.md` updated for new/changed functions
- [ ] `docs/USER_GUIDE.md` updated for new features
- [ ] `CHANGELOG.md` entry includes: what changed, why, migration steps
- [ ] Breaking changes explicitly documented

### 7. Release Artifacts

- [ ] Git tag format: `vX.Y.Z` (e.g., `v3.15.0`)
- [ ] Tag is annotated: `git tag -a vX.Y.Z -m "Release vX.Y.Z"`
- [ ] `SHA256SUMS.txt` generated for release artifacts
- [ ] SLSA attestation generated (CI)
- [ ] GitHub Release created with changelog and artifacts

### 8. Post-Release

- [ ] `dev` branch rebased on `main`
- [ ] Next `VERSION.txt` set to development version
- [ ] Release announced (if applicable)
- [ ] Package registries updated (npm, PyPI, crates.io, NuGet -- if applicable)
- [ ] Verify published packages install and pass smoke test

---

## Severity Gate

| Finding | Action |
|---------|--------|
| Any test failure | **BLOCK** release |
| New ASan/UBSan finding | **BLOCK** release |
| New CodeQL critical/high | **BLOCK** release |
| New compiler warnings | Review; block if in core arithmetic |
| dudect threshold exceeded | Review; block if in CT sign/mul paths |
| Documentation gap | May release with follow-up issue |

---

## Emergency / Hotfix Release

For security hotfixes, the following subset is mandatory:

- [ ] Fix addresses the specific vulnerability
- [ ] Regression test for the vulnerability included
- [ ] ASan + UBSan pass
- [ ] CodeQL pass
- [ ] Core test suites pass (field_audit, bip340, rfc6979, cross_libsecp256k1)
- [ ] CHANGELOG updated
- [ ] Tag + release created

Full checklist can be completed in a follow-up patch release within 7 days.

---

*Template version: 1.0*  
*Last updated: 2026-02-24*
