# Release Notes Source — Bindings Validation Closure (2026-03-23)

## Purpose

This file is the release-time source note for the completed stable binding validation closure.

Use it later to assemble:
- release notes
- changelog summary text
- GitHub release body
- announcement/discussion post text

## User-Visible Summary

The stable `ufsecp` binding layer is now validated and documented as one coherent matrix instead of a partially verified set of wrappers.

Validated bindings:
- C#
- Java
- Swift
- Python
- Go
- Rust
- Node.js
- PHP
- Ruby
- Dart
- React Native contract surface

## Release Candidate Highlights

### Binding validation

- Unified the shared binding validator at `ci/validate_bindings.sh` across the stable bindings.
- Default validation now covers smoke suites for C#, Java, Swift, Python, Go, Rust, Node.js, PHP, Ruby, and Dart.
- React Native now has a default mock-bridge contract smoke path in plain Node.js.
- Full native React Native smoke remains available as an opt-in lane via `UFSECP_VALIDATE_REACT_NATIVE=1`.

### Wrapper fixes included in this closure

- Fixed wrapper/test drift across Go, Node.js, Rust, PHP, Ruby, Dart, and React Native.
- Hardened zero-length hashing input handling for wrappers that previously passed null pointers for empty buffers.
- Fixed Dart `NativeFinalizer` usage by making `UfsecpContext` implement `ffi.Finalizable`.
- Replaced the unstable Dart package:test path in the shared validator with a deterministic standalone smoke runner.

### Documentation alignment

- Canonical binding docs, examples, packaging notes, and README surfaces now match the validated state.
- Dart package naming is aligned to `ultrafast_secp256k1`.
- React Native docs are aligned to `react-native-ultrafast-secp256k1` and the context-based `UfsecpContext` API.
- The bindings matrix no longer describes Dart or React Native with stale compile-only or optional wording.

## Verified Numbers

- Shared validator default lanes green: 11
- Dart smoke runner: 12/12 checks passing
- React Native contract smoke: 12/12 checks passing
- Stable C ABI functions documented as covered per binding: 42/42

## Suggested Release Note Wording

### Short version

- Completed stable binding validation across C#, Java, Swift, Python, Go, Rust, Node.js, PHP, Ruby, and Dart, with React Native contract smoke added as the default validation path.
- Fixed multiple wrapper drift and FFI edge cases uncovered during the pass, including zero-length hashing buffers, Rust linker/search-path issues, and Dart finalizer/runtime issues.
- Synchronized canonical binding docs and packaging notes to the current validated package names and context-based APIs.

### Longer version

- The shared binding validator now runs a coherent multi-language smoke matrix for the stable `ufsecp` bindings instead of relying on a smaller subset of wrappers.
- Dart has been brought into the validated matrix with a standalone smoke runner, and React Native now has a practical default contract-validation lane that works without full mobile bootstrapping.
- The docs surface was cleaned up so examples, package names, and validation framing now match the validated implementation state.

## Boundaries / Caveats

- This closure does not claim that full native React Native runtime smoke runs by default; that lane remains opt-in.
- This closure does not itself publish packages or cut a release tag.
- Local Dart SDK execution required repairing execute bits on extracted runtime helpers during local validation; that was an environment issue, not a library ABI issue.

## Supporting Files

- `docs/reports/bindings_validation_closure_2026-03-23.md`
- `docs/reports/announcement_bindings_validation_2026-03-23.md`
- `CHANGELOG.md` (`[Unreleased]` section)
- `docs/BINDINGS.md`
- `docs/BINDINGS_USAGE_STANDARD.md`
- `docs/BINDINGS_EXAMPLES.md`
- `docs/BINDINGS_PACKAGING.md`
