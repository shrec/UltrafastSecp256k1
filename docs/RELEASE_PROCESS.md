# Release Process

> **Applies to:** UltrafastSecp256k1 (`libufsecp`) -- all platforms and binding packages.

---

## 1. Release Cadence

| Release Type | Frequency | Branch | Trigger |
|-------------|-----------|--------|---------|
| **Patch** (3.14.*x*) | As needed | `main` | Bug/security fix |
| **Minor** (3.*x*.0) | ~4-8 weeks | `main` <- `dev` | New features, non-breaking changes |
| **Major** (*x*.0.0) | When required | `main` <- `dev` | ABI-breaking changes |

> **Unscheduled security releases** bypass the cadence and ship ASAP.

---

## 2. Pre-Release Checklist

### 2.1 Code Freeze

1. **All CI green** on `dev` -- every platform (Linux/macOS/Windows/WASM).
2. **No open P0 issues** tagged for this milestone.
3. Cross-library differential test passes (`test_cross_libsecp256k1`).
4. Parser fuzz tests pass (`test_fuzz_parsers`, `test_fuzz_address_bip32_ffi`).
5. Protocol tests pass (`test_musig2_frost`, `test_musig2_frost_advanced`).

### 2.2 Version Bump

| File | Action |
|------|--------|
| `VERSION.txt` | Set `MAJOR.MINOR.PATCH` |
| `include/ufsecp/ufsecp_version.h.in` | Verify `UFSECP_ABI_VERSION` (bump if ABI changed) |
| `CHANGELOG.md` | Add release section with date, summary, breaking changes |
| Binding manifests** | Update version where needed (Cargo.toml, package.json, etc.) |

**Single commit**: `release: vX.Y.Z` on `dev`.

### 2.3 Testing Gate

```bash
# Full build + test suite
cmake -S . -B build-rel -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DSECP256K1_BUILD_CROSS_TESTS=ON \
      -DSECP256K1_BUILD_FUZZ_TESTS=ON \
      -DSECP256K1_BUILD_PROTOCOL_TESTS=ON
cmake --build build-rel -j
ctest --test-dir build-rel --output-on-failure
```

All tests **must** pass. No exceptions for release builds.

### 2.4 ABI Compatibility Check (Minor/Patch only)

For **minor** and **patch** releases, verify backward ABI compatibility:

1. Build the previous release's shared library.
2. Link a test binary against the *new* headers + *old* library.
3. Confirm `ufsecp_abi_version()` returns the expected value.
4. Run the full test suite against the old library.

See [ABI_VERSIONING.md](ABI_VERSIONING.md) for the complete ABI contract.

---

## 3. Release Steps

### 3.1 Merge to `main`

```bash
git checkout main
git merge --no-ff dev -m "release: vX.Y.Z"
```

### 3.2 Tag

```bash
git tag -a vX.Y.Z -m "UltrafastSecp256k1 vX.Y.Z"
```

> **Future**: Signed tags with GPG or cosign (see roadmap task 2.5.2).

### 3.3 Push

```bash
git push origin main --tags
git checkout dev
git merge main  # keep dev in sync
git push origin dev
```

### 3.4 Build Release Artifacts

Build platform binaries:

| Platform | Toolchain | Output |
|----------|-----------|--------|
| Linux x86_64 | Clang 17+ / GCC 13+ | `libufsecp.so.X`, `libufsecp_s.a` |
| macOS arm64/x86_64 | Apple Clang 15+ | `libufsecp.X.dylib`, `libufsecp_s.a` |
| Windows x86_64 | Clang-cl 17+ / MSVC 2022 | `ufsecp.dll`, `ufsecp_s.lib` |
| WASM | Emscripten 3.1+ | `ufsecp.wasm`, `ufsecp.js` |

> **Future**: Deterministic Docker builds (task 2.5.1), signed binaries (task 2.5.3).

### 3.5 GitHub Release

1. Create GitHub Release from the tag.
2. Upload platform artifacts as release assets.
3. Copy the CHANGELOG section into the release description.
4. Mark as **pre-release** if `MAJOR` is 0 or release is `rc/beta`.

### 3.6 Package Registry Publish

| Registry | Package | Command |
|----------|---------|---------|
| crates.io | `ultrafastsecp256k1` | `cargo publish` |
| npm | `@ultrafastsecp256k1/wasm` | `npm publish` |
| PyPI | `ultrafastsecp256k1` | `twine upload dist/*` |
| NuGet | `UltrafastSecp256k1` | `dotnet nuget push` |
| vcpkg | port PR | `vcpkg x-add-version` |
| Conan | `ufsecp/X.Y.Z` | `conan create . --version X.Y.Z` |

---

## 4. Post-Release

1. **Announce**: GitHub Discussions / project channels.
2. **Monitor**: Watch issue tracker for 48h post-release for critical regressions.
3. **Bump dev version**: On `dev`, set `VERSION.txt` to `X.Y.(Z+1)-dev` for development builds.
4. **Update docs site**: If applicable, rebuild and deploy API reference.

---

## 5. Hotfix Process

For critical security fixes on a released version:

```
main  ----○--- vX.Y.Z ---○-- vX.Y.(Z+1)
           \                /
            hotfix/issue-N -
```

1. Branch `hotfix/issue-N` from `main` at the release tag.
2. Apply the minimal fix + test.
3. Bump `PATCH` in `VERSION.txt`.
4. Merge to `main`, tag, release.
5. Cherry-pick to `dev` (or merge `main` into `dev`).

---

## 6. Security Disclosure

| Severity | Response Time | Release Time |
|----------|---------------|--------------|
| Critical (RCE, key leak) | < 24h acknowledge | < 72h patch |
| High (crash, DoS) | < 48h acknowledge | < 1 week patch |
| Medium (non-default config) | < 1 week acknowledge | Next scheduled release |
| Low (cosmetic, docs) | Next triage | Next scheduled release |

Report security issues to: **security@[project-domain]** (or via GitHub Security Advisories).

---

## 7. Release Branch Naming

| Branch | Purpose |
|--------|---------|
| `main` | Stable releases only |
| `dev` | Active development |
| `release/vX.Y` | Long-term support branches (if needed) |
| `hotfix/issue-N` | Emergency fixes |

---

## 8. Rollback

If a release has a critical defect:

1. Immediately publish a hotfix (preferred) or yank the release.
2. For package registries, use the yank mechanism (`cargo yank`, `npm deprecate`).
3. Notify users via GitHub Advisory and project channels.
4. Do **not** force-push tags -- create a new patch version instead.
