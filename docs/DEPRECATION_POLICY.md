# Deprecation Policy

> **Applies to:** UltrafastSecp256k1 (`libufsecp`) C ABI and C++ public interfaces.

---

## 1. Scope

This policy covers:

- All `ufsecp_*` functions exported via `UFSECP_API` in `ufsecp.h`
- Public C++ headers in `cpu/include/secp256k1/*.hpp`
- Configuration keys in `config.json` schema
- Command-line interfaces of shipped tools
- Package names and import paths in language bindings

It does **not** cover:

- Internal implementation details (files not in `include/`)
- Build system internals (CMake variables prefixed with `_`)
- Experimental features explicitly marked `[EXPERIMENTAL]`

---

## 2. Deprecation Lifecycle

```
+----------+     +--------------+     +-----------+     +---------+
|  Active   |----▶|  Deprecated  |----▶|  Removed  |----▶|  Gone   |
| (current) |     | (warnings)   |     | (next ABI |     |         |
|           |     | + migration  |     |  major)   |     |         |
+----------+     +--------------+     +-----------+     +---------+
     ▲                  |
     |    minimum 2     |
     |  minor releases  |
     +------------------+
```

### Timeline Guarantee

| Change Type | Minimum Deprecation Period | Removal Allowed |
|-------------|---------------------------|-----------------|
| Function removal | 2 minor releases | Next MAJOR |
| Signature change | 2 minor releases | Next MAJOR |
| Config key rename | 2 minor releases | Next MAJOR |
| Behavior change (semantic) | 1 minor release | Next MINOR+1 |
| Binding package rename | 2 minor releases | Next MAJOR |

**Exception**: Security vulnerabilities may force immediate removal without a deprecation period.

---

## 3. How to Deprecate

### 3.1 C API (`ufsecp.h`)

Use compiler attributes to emit deprecation warnings:

```c
/* Deprecated in v3.14.0 -- use ufsecp_new_function() instead.
 * Will be removed in v4.0.0. */
UFSECP_API UFSECP_DEPRECATED("use ufsecp_new_function()")
ufsecp_error_t ufsecp_old_function(ufsecp_ctx* ctx, ...);
```

Where `UFSECP_DEPRECATED` is defined as:

```c
#if defined(__GNUC__) || defined(__clang__)
  #define UFSECP_DEPRECATED(msg) __attribute__((deprecated(msg)))
#elif defined(_MSC_VER)
  #define UFSECP_DEPRECATED(msg) __declspec(deprecated(msg))
#else
  #define UFSECP_DEPRECATED(msg)
#endif
```

### 3.2 C++ API

Use the C++14 `[[deprecated]]` attribute:

```cpp
// Deprecated in v3.14.0 -- use new_method() instead.
// Will be removed in v4.0.0.
[[deprecated("use new_method() instead")]]
Scalar old_method() const;
```

### 3.3 Configuration Keys

When renaming config keys, support **both** for the deprecation period:

```cpp
// Parse both old and new key
if (json.contains("new_key")) {
    value = json["new_key"];
} else if (json.contains("old_key")) {
    value = json["old_key"];
    warn_once("config: 'old_key' is deprecated, use 'new_key'");
}
```

### 3.4 Documentation

Every deprecation must be documented in:

1. **CHANGELOG.md** -- under a "Deprecated" section for the release
2. **API reference** -- inline doc comment on the deprecated item
3. **Migration guide** -- in `docs/MIGRATION.md` with before/after examples

---

## 4. Migration Guides

For every deprecation, provide a migration path:

```markdown
## Migrating from `ufsecp_old_function` to `ufsecp_new_function`

### Before (deprecated)
ufsecp_old_function(ctx, input, output);

### After
ufsecp_new_function(ctx, input, input_len, output, &output_len);

### Notes
- `ufsecp_new_function` adds explicit length parameters for safety.
- The old function will be removed in v4.0.0.
```

---

## 5. Removal Process

When the deprecation period expires and a MAJOR version bump is planned:

1. Remove the deprecated symbol from `ufsecp.h`.
2. Remove the implementation from `ufsecp_impl.cpp`.
3. Bump `UFSECP_ABI_VERSION`.
4. Update all binding wrappers to remove the deprecated function.
5. Update `CHANGELOG.md` with "Removed" section.
6. Run full test suite to verify no internal dependencies remain.

---

## 6. ABI Stability Matrix

| API Surface | Stability | Deprecation Required? |
|-------------|-----------|----------------------|
| `ufsecp_*` functions | **Stable** | Yes |
| `UFSECP_*` constants | **Stable** | Yes |
| `ufsecp_error_t` values | **Stable** (additive only) | Yes (for removal) |
| C++ `secp256k1::*` classes | **Semi-stable** | Best effort |
| Internal headers | **Unstable** | No |
| Build variables | **Unstable** | No |

---

## 7. Versioning Interaction

Deprecations interact with versioning as follows:

```
v3.14.0  -- Function A marked deprecated (warning emitted)
v3.15.0  -- Function A still works, warning continues
v3.16.0  -- Function A still works (minimum 2 minor releases)
v4.0.0   -- Function A removed (MAJOR bump, ABI_VERSION bumped)
```

### The "Two Minor Releases" Rule

A deprecated symbol must survive for at least **two** minor releases after the release where it was first marked deprecated. This gives downstream consumers at least two upgrade cycles to migrate.

---

## 8. Communication Channels

| Channel | When | What to communicate |
|---------|------|-------------------|
| CHANGELOG.md | Every release | Full list of deprecations |
| Compiler warnings | Build time | Per-symbol deprecation messages |
| GitHub Release Notes | Release | Summary of deprecations |
| GitHub Discussions | Deprecation announcement | Migration guidance, Q&A |
| docs/MIGRATION.md | Per-deprecation | Step-by-step migration |

---

## 9. Exceptions

The following may bypass the standard deprecation period:

1. **Security vulnerabilities** -- immediate removal if continued availability poses a risk.
2. **Legal requirements** -- compliance-driven changes.
3. **Experimental features** -- marked `[EXPERIMENTAL]` have no stability guarantee.

Any exception must be documented in the CHANGELOG with justification.

---

## 10. Current Deprecations

| Symbol/Key | Deprecated In | Replacement | Removal Target |
|-----------|---------------|-------------|----------------|
| Flat config keys (`"database.path"`) | v3.12.0 | Nested `{"database":{"path":...}}` | v4.0.0 |

> This table is updated with each release.
