# Build output consolidation audit 012

Status: implementation-ready architecture contract (read-only audit).  
Scope: tracked entry points named by `BUILD_TOPOLOGY_AUDIT_012`; evidence is
`path:line` in the audited revision.

## 1. Finding and invariant

Persistent build products are split among repository-root `build*` and
`audit-output-*` directories, `${TMPDIR:-/tmp}`, `local-ci-output`, `out`, the
user ccache directory, and a Docker volume. This makes ownership and safe
cleanup unknowable.

The implementation MUST resolve exactly one physical root:

```text
CANONICAL_OUT=<primary-repo>/out
```

`<primary-repo>` means the task registry's canonical/primary checkout, not the
current secondary worktree. It must be resolved to an absolute, symlink-free
path and verified as belonging to the same repository identity. A secondary
worktree MUST NOT use its own `<worktree>/out`; all persistent products point
to the one `CANONICAL_OUT`. No persistent topology in this contract uses an
`out/presets`, `out/cache`, or `<worktree-id>` component.

## 2. Complete tracked-entry-point inventory

The following table inventories every raw CMake build-tree invocation in the
named tracked entry points and every repository-local persistent output that
the entry point can create or select.

| Entry point | Exact invocation evidence | Current binary directory / repository-local outputs |
|---|---|---|
| `audit/run_full_audit.sh` | configure/build `231-246`; sanitizer configure/build `425-433`; tests `507` | defaults `build-audit` and `audit-output-<timestamp>` (`38-39`); artifacts under output (`48`, `66`), report (`707`); sanitizer tree `build-audit-asan` (`422`); nested `BUILD_DIR/valgrind-ct` and reports (`164`, `176`) |
| `audit/run_full_audit.ps1` | configure/build `338-355`; sanitizer branches `627-636` and `653-666`; tests `722` | defaults `build-audit` and `audit-output-<timestamp>` (`62-65`); sanitizer tree `build-audit-asan` (`609`); report `OutputDir/audit_report.md` (`999`) |
| `ci/generate_audit_package.sh` | configure/build `84-97` | default tree `out/auditor` (`17`, resolved at `77`); evidence directory `out/audit-evidence-<timestamp>` (`65-67`), including `build_info.json` (`148`), tool/CT evidence (`171-195`), README (`205`), and manifest/package contents (`284-305`) |
| `ci/generate_audit_package.ps1` | configure/build `91-108` | default root tree `build-audit` (`26`, resolved at `86`); root `audit-evidence-<timestamp>` and evidence subdirectories (`73-76`), including build info (`167`), tool/CT evidence (`180-217`), and README (`324`) |
| `ci/local-ci.sh` | configure/build pairs `141/147`, `163/175`, `196/208`, `227/232`, `274/279`, `305/318`, `390/399`, `450/460`, `488/495`, `535`, `575/581`, `709/717`; external reference configure/build/install `738-747` | base `${TMPDIR:-/tmp}/build-local-ci-$$` (`38`, `60-62`) with suffixes `-werror`, `-asan`, `-tsan`, `-valgrind`, `-dudect`, `-coverage`, `-tidy`, `-ci-<label>-<type>`, `-audit`, `-cppcheck`, `-bench`, `-valgrind-ct`, and `-cryptofuzz` (`139`, `161`, `194`, `225`, `272`, `303`, `388`, `448`, `486`, `531`, `573`, `662`, `707`); root `local-ci-output` coverage/tidy/audit/cppcheck/benchmark/trust/valgrind products (`371-374`, `417-420`, `497-512`, `532-561`, `583-596`, `609-644`, `667-668`, `966`); external dependency source/build/install under `/tmp` (`731-747`) |
| `ci/run-local-ci.ps1` | Docker image build `73`; delegates CMake work to local CI via container `81-109` | named persistent cache volume `uf-ccache` (`48`, `100-106`) and root `local-ci-output` (`120-130`) |
| `CMakePresets.json` | preset configure/build/test dispatch is declarative | every configure preset inherits `binaryDir=${sourceDir}/out/${presetName}` and exports compile commands (`10-16`); this already matches primary layout |

The remaining named tracked files do not launch CMake but affect topology and
therefore are mandatory migration surfaces:

- `ci/_ufsecp.py` searches `out` plus legacy root/suite trees
  `build_opencl`, `build_rel`, `build-cuda`, `build`, `build-audit`,
  `build_test`, `build-shim-v3`, `build-packaging-repro`, and
  `bindings/c_api/build` (`129-143`). It must search only authorized canonical
  configuration directories (or an explicit `--lib`), never revive legacy
  roots.
- `ci/audit_gate.py` prunes `build`, `build_bench`, `build-audit`, and
  `CMakeFiles` during scans (`695`) and defaults its mutation build to
  `build_opencl` (`862`). The default must become a task configuration below
  `CANONICAL_OUT/.tasks`.
- `.gitignore` masks `build*/`, `out/`, CMake products, logs, packages, and
  `local-ci-output` (`1-4`, `25-39`, `74-77`, `138`, `167-168`). Ignore rules
  are not a hygiene gate; the gate in section 5 must inspect ignored paths.

No raw CMake invocation occurs in `ci/_ufsecp.py`, `ci/audit_gate.py`,
`ci/run-local-ci.ps1`, `.gitignore`, or `CMakePresets.json`; their effects are
consumer, orchestration, default-path, ignore, and declarative-preset behavior
respectively.

## 3. Canonical topology

```text
CANONICAL_OUT/
  <preset>/                         # primary checkout preset binary dir
  .tasks/<task-id>/build/<configuration-id>/
  logs/<producer>/<run-id>/
  packages/<producer>/<run-id>/
  .cache/ccache/
  .cache/sccache/
  .cache/dependencies/
  .cache/vscode-cpptools/<workspace-key>/
  .quarantine/<quarantine-id>/
```

- Primary preset configure/build/test uses
  `CANONICAL_OUT/<preset>` exactly.
- Every secondary/task worktree, including audit, local CI, sanitizer, coverage,
  benchmark, mutation, cryptofuzz, and ad-hoc task configurations, uses
  `CANONICAL_OUT/.tasks/<task-id>/build/<configuration-id>` exactly.
  `configuration-id` is a stable digest/slug of generator, toolchain, compiler,
  build type, architecture, and relevant cache options; it is never guessed.
- Logs/evidence go to `CANONICAL_OUT/logs/<producer>/<run-id>`. Final archives,
  manifests, SBOMs, and distributable evidence go to
  `CANONICAL_OUT/packages/<producer>/<run-id>`. A run manifest records repository
  identity, source worktree, commit, task ID, configuration ID, creator, and
  timestamps.
- `CCACHE_DIR=CANONICAL_OUT/.cache/ccache` and
  `SCCACHE_DIR=CANONICAL_OUT/.cache/sccache`; use one selected compiler launcher,
  not both. Downloaded/configured dependency data uses
  `CANONICAL_OUT/.cache/dependencies`. Cache tools provide concurrency and
  content validation; scripts must not manually merge cache files.

### Why binary directories remain isolated

A CMake binary directory is stateful: `CMakeCache.txt`, generated Ninja files,
absolute source paths, compiler/toolchain probes, dependency state, and
generator metadata bind it to the source directory and configuration used at
configure time. Sharing one binary directory between distinct source
worktrees can build the wrong revision, reuse stale dependency decisions, race
regeneration, and make cleanup affect another task. Therefore two distinct
source worktrees MUST never share a CMake binary directory, even at the same
commit.

Compilation is nevertheless deduplicated safely at the artifact-content layer:
ccache or sccache keys compiler identity, flags, preprocessed input, and relevant
environment, so equivalent translation units can reuse object results across
isolated CMake trees. Dependency downloads can likewise share the bounded
`.cache/dependencies` store while each binary tree retains its own configured
dependency/build state.

## 4. Migration of the six observed repository-root directories

These exact observed names require registry/manifest resolution before any
move. A name is not proof of owner or configuration.

| Observed repository-root directory | Required destination if ownership/configuration is proven |
|---|---|
| `build_bench_run` | `CANONICAL_OUT/.tasks/<task-id>/build/<configuration-id>` |
| `build-audit` | primary preset: `CANONICAL_OUT/audit`; task-owned run: `CANONICAL_OUT/.tasks/<task-id>/build/<configuration-id>` |
| `build-batchmt` | `CANONICAL_OUT/.tasks/<task-id>/build/<configuration-id>` |
| `build-review-lbtc-gpu` | `CANONICAL_OUT/.tasks/<task-id>/build/<configuration-id>` |
| `build-sighash-gpu-proof` | `CANONICAL_OUT/.tasks/<task-id>/build/<configuration-id>` |
| `local-ci-output` | logs/evidence to `CANONICAL_OUT/logs/local-ci/<run-id>`; package-ready artifacts to `CANONICAL_OUT/packages/local-ci/<run-id>` |

For any directory whose task owner, source worktree, commit, or configuration
cannot be proven from its cache plus the task registry, migration MUST refuse
normal mapping and deletion. Move it only to
`CANONICAL_OUT/.quarantine/<quarantine-id>` with a manifest containing original
absolute path, device/inode (or Windows file identity), size, timestamps,
discovered CMake source path, hashes of identity files, reason for ambiguity,
and operator/time. Never invent a `<task-id>`, configuration, or owner.

## 5. Enforced hygiene gate

Run after configure/build workflows and in review CI. Resolve
`CANONICAL_OUT`, then scan each registered repository checkout/worktree without
pruning ignored files. Fail if any matching object is outside the physical
`CANONICAL_OUT`:

```text
CMakeCache.txt
build.ninja
compile_commands.json
build-*            (directory)
build*/            (directory; includes build and build_*)
local-ci-output    (directory)
```

The comparison must use canonical absolute paths after symlink resolution;
symlinks that enter or escape `CANONICAL_OUT` fail. Report exact offending
paths and owning worktree. Also fail if a configured CMake cache's
`CMAKE_HOME_DIRECTORY` disagrees with its manifest or if a task tree lacks a
registry-valid active/review/terminal owner. Do not exempt an offender because
`.gitignore` ignores it.

## 6. Safe terminal cleanup

Cleanup is task-registry-aware and is permitted only for a task in a terminal,
non-review state. Under a registry lock:

1. Resolve repository identity, `CANONICAL_OUT`, task ID, task state, worktree
   path, and manifest; recheck that the manifest source matches the registered
   worktree and that the target is exactly below
   `CANONICAL_OUT/.tasks/<task-id>`.
2. Refuse on missing/ambiguous identity, active process/lease, active task,
   review task, linked child task, or path/symlink mismatch.
3. Atomically rename the exact task directory to a same-filesystem
   `CANONICAL_OUT/.quarantine/<id>`, write the deletion manifest, release the
   registry lock, then delete only after the configured retention period.
4. Logs/packages follow explicit retention policy and registry references;
   shared `.cache` data is evicted only by cache-tool size/age policy, never by
   task cleanup.

Blind globbing or recursive deletion of repository roots, other repositories,
other task directories, other worktrees, active artifacts, or review artifacts
is forbidden. Cleanup must never derive a deletion target merely from a
directory name, PID, current working directory, or an untrusted environment
variable.

## 7. VS Code contract

Workspace settings must resolve the same primary repository and define:

```json
{
  "cmake.buildDirectory": "${primaryRepo}/out/.tasks/${taskId}/build/${configurationId}",
  "C_Cpp.intelliSenseCachePath": "${primaryRepo}/out/.cache/vscode-cpptools/${workspaceKey}",
  "C_Cpp.intelliSenseCacheSize": 512,
  "C_Cpp.default.compileCommands": "${primaryRepo}/out/.tasks/${taskId}/build/${configurationId}/compile_commands.json"
}
```

For the primary checkout/preset, both `cmake.buildDirectory` and compile
commands instead use `${primaryRepo}/out/${preset}`. `workspaceKey` is a stable
bounded key; the 512 MiB limit is mandatory and eviction must remain within
`.cache/vscode-cpptools`. CMake remains the sole producer of
`compile_commands.json` (`CMakePresets.json:15`); VS Code consumes the selected
active configuration's file and must not copy or symlink it into the source
tree. Switching preset/task updates both paths atomically. Missing or stale
compile commands is an error prompting configure, not permission to fall back
to a repository-local database.

## 8. Implementation acceptance sequence

1. Add a single canonical-out resolver backed by repository/task identity.
2. Migrate preset, audit, package, local-CI, mutation, consumer-search, Docker,
   compiler cache, dependency cache, and VS Code paths to sections 3 and 7.
3. Quarantine the six observed roots using section 4; refuse ambiguous cases.
4. Enable the section 5 gate before removing legacy ignore patterns.
5. Exercise two concurrent worktrees at the same configuration: binary
   directories must differ, compiler-cache hits should occur, and cleanup of
   one terminal task must leave the other task and all review artifacts intact.
