# BUILD_REGISTRY_CONTRACT_028 — Verified Consolidated Build Registry Contract

Decision: **specification GO for Codex review; every implementation, migration,
quarantine, and deletion action remains BLOCKED** until the AIWorkHub owner
supplies authoritative source for a public atomic build-output
claim/heartbeat/revalidate/safe-delete interface. This document authorizes no
cleanup.

This is a specification. “Observed” means verified in the canonical repository,
the Task MCP card/context, Git plumbing, `BUILD_TOPOLOGY_AUDIT_012`, or the
installed AIWorkHub source. “Required” means proposed behavior that does not yet
exist. A proposed operation name is a capability label, not a claim that an API
with that name exists.

## 0. Task and project-context receipt

The canonical Task MCP card was read as:

- task: `BUILD_REGISTRY_CONTRACT_028`
- runner: `build_contract_author_v2`
- topic: `build_registry_contract_final`
- state at authoring: `processing` / `claimed`
- sole allowed and required output:
  `docs/BUILD_REGISTRY_CONTRACT_028.md`
- dependency: `BUILD_TOPOLOGY_AUDIT_012`

The injected project-context evidence preserved by the card is:

| Evidence | Exact delivered value |
|---|---|
| Bundle schema | `aiworkhub.task_mcp.project_context_bundle.v1` |
| Delivered bundle sections | `section_count=4` |
| Worker acknowledgement | `section_count=2` |
| Delivered Source Graph | `hit_count=96`, `truncated=true`, `bytes=12221` |
| Delivered Source Graph SHA-256 | `426c3d621b19451376d737b8812521bc649bf42d3aac8b192f10baf737bba7c1` |
| Card-delivered isolated baseline | `7e620d1db48066a0a7456629c3133f01f40c9036` |
| Review-time primary `dev` | `bf33cfb2b52e2e274a6b8dca08c1d2c3698c6a8a` |

The delivered Source Graph shard is immutable receipt evidence. A later manager
re-query returned the same 96 hits but a different serialization size and
`truncated=false`; that cache projection does not replace or rewrite the
delivered receipt above.

The current session-state query returned `state=unknown` with zero evidence.
The bounded task-specific AI Memory and KB queries both returned zero results
and were not repeated. Source Graph names the canonical authority repository as
`/home/shrek/Secp256k1/Secp256K1fast/libs/UltrafastSecp256k1`.

Implementation must re-read repository identity, the primary `dev` revision,
the task card, and public owner API source at implementation time. Neither the
delivered baseline nor this review-time revision may be treated as permanently
current.

## 1. Canonical repository authority

### 1.1 The submodule owns its state

The sole repository identity for this contract is:

```text
root    = /home/shrek/Secp256k1/Secp256K1fast/libs/UltrafastSecp256k1
repo_id = repo_666797171f0141c58bf05f579b2ee16e
```

Its own authoritative files are:

```text
.aiworkhub/project.json
.aiworkhub/config/storage.json
.aiworkhub/tasking/task_queue.sqlite
```

The project manifest has schema `aiworkhub.project_manifest.v1`. The storage
manifest has schema `aiworkhub.storage_registry.v1`, the same `repo_id`,
`durable_root=.aiworkhub`, and a canonical-active `task_queue` entry whose
relative durable path is `tasking/task_queue.sqlite`.

The parent superproject has a different repository identity. Its registry,
manifest, task rows, and paths must never be substituted for this submodule’s
authority.

### 1.2 Git common-dir and worktree mapping are distinct facts

For the submodule:

```text
git common-dir =
/home/shrek/Secp256k1/Secp256K1fast/.git/modules/libs/UltrafastSecp256k1

core.worktree =
../../../../libs/UltrafastSecp256k1
```

`core.worktree` is resolved relative to the common Git directory. The result is
the primary submodule root above. The absolute common-dir must not be replaced
with the relative `core.worktree` value.

A retained worktree maps to the primary checkout by:

1. reading its Git common-dir;
2. reading `core.worktree` from that common-dir;
3. resolving both paths without trusting the current working directory;
4. verifying the submodule project manifest and exact `repo_id`;
5. comparing filesystem identity, not only path spelling; and
6. revalidating the primary revision before any implementation action.

This mapping must account for symlinks, case-folding filesystems, Windows
junctions/reparse points, and `\\?\` path spellings.

## 2. Installed AIWorkHub runtime: observed facts only

The active installed extension inspected for this contract is:

```text
extension ID: shrec.aiworkhub
package name: aiworkhub
publisher: shrec
version: 0.6.30
source root:
/home/shrek/.vscode-server/extensions/shrec.aiworkhub-0.6.30/runtime/aiworkhub/
```

Those identity fields come from the installed extension's `package.json`.
Its SHA-256 is
`05fa6d91ec084b04c28ab38ef6a7e59fde4b71568306d16d93d2cfdfdd7c475a`.
For a reproducible fingerprint of every installed file cited in this section,
the SHA-256 manifest was produced in this exact order: `package.json`,
`repository_state.py`, `storage_registry.py`, `task_store.py`, `core.py`,
`callback_store.py`, `process_launcher.py`, and `worker_supervisor.py`; hashing
that `sha256sum` manifest again yields
`e6c772a7ff0934f539a3853a35138fee499b740aaf2de07e1fada810c7b27618`.
The symbols in the table and the artifact lifecycle in section 2.2 were
reverified against that exact 0.6.30 source set.

Source Graph indexes the canonical project rather than this installed extension,
so bounded exact-file reads were used for the external installed source.

| Source | Exact observed symbol | Observed responsibility; explicit non-responsibility |
|---|---|---|
| `repository_state.py` | `PROJECT_MANIFEST_REL`; `inspect_repository` | Reads `.aiworkhub/project.json`, verifies optional expected `repo_id`, and returns repository/durable/runtime paths. It does **not** resolve build ownership or allocate build paths. |
| `storage_registry.py` | `STORAGE_REGISTRY_REL`; `load_storage_registry` | Reads `.aiworkhub/config/storage.json` and validates canonical database records. It does **not** map build configuration digests to build directories. |
| `task_store.py` | `storage_readiness`; `_connect`; `canonical_status` | Verifies manifest/storage/task-DB readiness; `_connect(readonly=True)` uses `mode=ro` and `PRAGMA query_only=ON`; `canonical_status(row)` normalizes a row. It provides no build-output claim. |
| `core.py` | `WRITE_COMMANDS`; `_canonical_connect`; `create_task`; `mark_review`; `mark_done`; `reject_review` | `WRITE_COMMANDS` is a set of task command names, not event vocabulary and not a build lock/API. The functions implement task lifecycle through the canonical task DB. |
| `callback_store.py` | `claim_pending_callback_batch`; `rebind_pending_callbacks` | Both use `BEGIN IMMEDIATE` for callback-batch transactions. Callback locking is not build-output locking. |
| `process_launcher.py` | `_registry_lock`; `GC_CANDIDATE_PROCESS_STATES`; `GC_ELIGIBLE_CANONICAL_STATUSES`; `_gc_finalized_workspace` | `_registry_lock` serializes process duplicate-check/spawn using the process-event lock. `_gc_finalized_workspace` removes retained isolated worker workspaces after exact checks. Neither is public build-output authority. |

### 2.1 Exact task and GC semantics

Observed `task_store.canonical_status` outputs are:

```text
pending
processing
review
blocked
finished
archived
```

Observed process GC constants are:

```text
GC_CANDIDATE_PROCESS_STATES = TERMINAL_PROCESS_STATES - {"blocked"}
GC_ELIGIBLE_CANONICAL_STATUSES = {"finished", "archived"}
```

`core.mark_done` moves an eligible reviewed task to `worker_status=done` and
`status=finished`. `core.reject_review` is **not terminal**: it returns the task
to `worker_status=unclaimed` and `status=pending`. Therefore rejection must
never make a build tree eligible for disposition or GC.

### 2.2 Exact process artifacts

The installed defaults are:

```text
.aiworkhub/runtime/process_logs/process_events.jsonl
.aiworkhub/runtime/process_logs/process_events.jsonl.lock
.aiworkhub/runtime/process_logs/processes/
```

The lock is derived exactly as `Path(f"{process_log_path}.lock")`. For each
isolated request, the `processes/` directory has these four mandatory,
persistent request artifacts (subject to later authorized lifecycle cleanup),
all keyed by the opaque request ID:

```text
<request>.request.json
<request>.supervisor.json
<request>.stdout.log
<request>.stderr.log
```

`<request>.supervisor-spec.json` is transient launch input:
`process_launcher.py` writes it before spawning the supervisor, then
`worker_supervisor.py:main` loads it and unlinks it in a `finally` block via
`_unlink_if_regular`. It must not be treated as a persistent post-spawn
artifact. `<request>.cancel.json` is conditional: `ProcessManager.cancel`
creates it only when cancellation is requested, and `worker_supervisor.py`
removes it during supervisor teardown. Therefore the runtime does not promise
that six files always exist for a request.

Process events and supervisor records carry PID and process start ticks.
Supervisor heartbeat evidence plus exact PID/start-tick identity informs
process liveness. These artifacts do not claim a build tree.

No `runtime/locks` directory, socket, PATH command, CLI adapter, build
configuration registry, build claim, build heartbeat, or build safe-delete API
was observed. None is invented here.

## 3. SQLite WAL read contract

Normal authoritative read behavior is:

1. Open the canonical task DB with URI `mode=ro`.
2. Set `PRAGMA query_only=ON`.
3. When the database is live in WAL mode and directory/sidecar permissions
   permit, the read-only connection participates in the live WAL snapshot and
   accesses the existing `-wal` and `-shm` state.
4. A sandbox may fail because it cannot create, open, map, or share the `-shm`
   sidecar or cannot access the `-wal` file. That failure is an **unknown /
   fail-closed** result, never permission to infer task state from files.

`immutable=1` is not a live-WAL fallback for a decision. It treats the database
as unchanging and may expose only stale main-file state while committed frames
remain in WAL. An immutable read may be used for explicitly labelled forensic
inspection of a genuinely quiescent copy, but its result is never ownership,
lease, reuse, terminal-state, quarantine, or deletion evidence.

Only a supplied public owner interface may make a build ownership decision. A
consumer may use the existing read APIs as supporting evidence, but must fail
closed on unreadable authority, schema mismatch, sidecar failure, identity
mismatch, or conflicting snapshots.

## 4. Required physical topology

`CANONICAL_OUT` is the physical `out` directory below the verified primary
submodule root:

```text
out/
  <preset>/
  .tasks/<task-id>/build/<configuration-id>/
  .cache/
    ccache/
    sccache/
    dependencies/
    vscode/
  logs/<producer>/<run-id>/
  packages/<producer>/<run-id>/
  .quarantine/<disposition-id>/
  .tmp/<operation-id>/
```

The three identity-bearing roots are exactly:

```text
out/<preset>
out/.tasks/<task-id>/build/<configuration-id>
out/.cache
```

Rules:

- A primary-checkout preset uses `out/<preset>`.
- Every isolated/task build uses the primary checkout’s
  `out/.tasks/<registered-task-id>/build/<configuration-id>`.
- A secondary worktree must not retain its own `out`, root `build*`,
  `local-ci-output`, or persistent `/tmp` build root.
- Logs, manifests, evidence, packages, compiler caches, resolver state, and
  operation temporary files are under the primary `out`.
- OS temporary space may hold only an unopened, non-authoritative transfer
  buffer that is removed before the producing process exits. It may never be a
  configured build root, manifest authority, cache, or cleanup source.
- `out/.cache` is shared; task binary trees are never shared.
- `ccache` and `sccache` are mutually exclusive for one build.
- A directory name is never evidence of task ownership.

## 5. Three separate identities

### 5.1 Owner tuple

The proposed owner tuple is operational and deliberately transient:

```text
repo_id
task_id
request_id
source_worktree_realpath
git_common_dir
process_pid
process_start_ticks
lease_nonce
lease_generation
lease_boot_or_session_id
heartbeat_wall_utc
heartbeat_monotonic
lease_deadline_wall_utc
```

The task ID must already exist in the submodule task DB. Local CI may not create
or substitute names such as `local-ci-<run-id>` in the task-ID slot. Run IDs are
separate manifest fields.

### 5.2 Build configuration ID

`configuration-id` is a deterministic digest over canonical serialization of:

- verified `repo_id`;
- clean Git tree object ID;
- a dirty-content digest that includes sorted relative path, file type/mode,
  staged content, unstaged content, and relevant untracked content;
- recursive submodule identities and exact revisions;
- recursive vendored/fetched dependency identity, source origin, revision or
  content digest, and patch-set digest;
- enabled/disabled feature definitions and preprocessor definitions;
- generator and generator version;
- toolchain file content digest;
- compiler/linker identity, version, target triple, ABI, and relevant flags;
- build type, sanitizer/LTO/debug/optimization mode;
- host/target platform and architecture, including GPU backend/target where
  applicable; and
- build-affecting cache variables/options.

The canonical serialization is versioned, ordered, length-delimited, and
rejects missing required fields. Merely recording `dirty=true` is insufficient:
the dirty contents themselves are bound.

The configuration ID excludes task ID, request ID, worktree path, PID, lease,
timestamps, run ID, and absolute output paths. Those belong to the owner/run
manifest. Equivalent content and configuration in two worktrees therefore has
the same configuration ID but still receives distinct task binary directories.

### 5.3 Stable compiler-cache namespace

The compiler-cache namespace is not the build configuration ID and not the
owner tuple. Its versioned key binds compiler family/version/content identity,
target triple, ABI, language mode, sanitizer mode, and other flags that change
object compatibility. It excludes task ID, request ID, worktree path, PID,
lease, run ID, and timestamps.

The cache tool’s content key must additionally bind the actual translation-unit
input/dependency content and effective command. Equivalent translation units
may then reuse cached objects across task trees, while incompatible toolchains
cannot collide. Native cache locking and bounded size/age eviction govern
`out/.cache/ccache` or `out/.cache/sccache`; task GC never recursively deletes
shared compiler-cache entries.

## 6. Required public build-owner authority — proposed and BLOCKED

The installed runtime supplies no public source/API for the following build
capabilities. The capability labels below are normative requirements, **not
existing function or command names**:

1. atomic build-output claim;
2. lease heartbeat;
3. ownership revalidation for reuse;
4. terminal disposition/quarantine; and
5. safe bounded deletion.

The AIWorkHub owner must supply the authoritative source root, commit, exact
source files, exported symbols, transaction/lock scope, persistent schema,
error model, and tests. Editing the installed VSIX/runtime copy is forbidden.
Until that source handoff is independently verified, every node in §12 after
the authority gate is BLOCKED.

Existing `process_launcher._registry_lock` cannot be reused: its documented
scope is process duplicate-check and spawn. Existing
`_gc_finalized_workspace` cannot be reused as build cleanup: its target is a
retained isolated worker workspace. `storage_registry` cannot be used as a
configuration-path registry. `WRITE_COMMANDS` cannot be treated as build
operations.

### 6.1 Required atomic claim semantics

The eventual owner implementation must:

1. bind the canonical repo, task row, owner tuple, configuration ID, and target
   below the already-opened primary `out` root;
2. acquire one owner-defined lock/transaction whose documented scope covers
   claim, heartbeat, revalidation, disposition, and deletion;
3. allocate a cryptographically unpredictable lease nonce and monotonically
   increasing generation for that task/configuration owner;
4. create a same-filesystem staging directory below `out/.tmp` using
   exclusive/no-follow operations;
5. write a versioned ownership manifest, `fsync` the file, atomically rename
   it into place, and `fsync` the containing directory;
6. publish the directory by same-filesystem atomic rename or an equivalent
   owner-proven atomic primitive;
7. reject an occupied owner path unless the old generation is proven
   terminal, process-dead, and safely disposed; and
8. return an opaque owner proof consumed by all producers and consumers.

A crash between allocation and publication must leave either no claim or a
recoverable, unambiguously uncommitted staging object. It must never leave a
normal-looking owner directory without a durable manifest.

Two different registered tasks using the same configuration ID have distinct
task paths and may build concurrently. They share only the compatible compiler
cache. A test that rejects the second task merely because the configuration ID
matches is invalid.

### 6.2 Restart-safe lease semantics

Each claim records wall-clock UTC, monotonic time, and a boot/session identity.
Monotonic values are compared only within the same boot/session. After restart
or boot/session mismatch, an old monotonic deadline is unknown rather than
expired. Recovery requires:

- exact PID plus process start-tick mismatch or proven process death;
- canonical task status re-read;
- owner generation/nonce match;
- no active child/dependent owner reference;
- a bounded recovery grace period; and
- revalidation under the same owner transaction before takeover.

Heartbeat updates wall and monotonic evidence under that transaction and must
present the current nonce/generation. A stale heartbeat, old PID, or old
generation cannot revive or mutate a newer claim.

### 6.3 Reuse and terminal disposition

Before reuse, quarantine, or delete, the owner must re-read:

- repository identity and physical root;
- task row and normalized canonical status;
- source/dependency/configuration digests;
- owner manifest and nonce/generation;
- supervisor PID/start ticks and lease evidence; and
- child/dependent references.

Only canonical `finished` or `archived` is terminal for build disposition.
`pending`, `processing`, `review`, and `blocked` are not terminal.
`reject_review` returns to pending and is never a GC trigger.

No consumer or producer may mutate the owner manifest, extend a lease, take
over a claim, move a tree, or delete anything directly.

## 7. Containment and race safety

String prefix checks and a one-time `realpath` are insufficient. The eventual
owner implementation must:

- open the verified physical `out` root once and retain its directory handle;
- traverse relative components with directory-handle-relative operations;
- reject `..`, absolute components, empty/ambiguous encodings, and alternate
  path separators;
- use no-follow semantics (`O_NOFOLLOW`/`openat` or platform equivalent) for
  every component;
- verify device/inode or Windows file identity before and after each rename or
  destructive phase;
- reject symlinks, mount crossings, bind-mount surprises, Windows junctions,
  and reparse points unless an owner-reviewed platform primitive proves
  containment;
- never follow a symlink found inside a quarantined tree during recursive
  deletion;
- rename to quarantine only on the same filesystem;
- operate on already-opened handles during deletion so a path substitution
  cannot retarget the operation; and
- fail closed if any identity changes between scan, lock, revalidation,
  disposition, and delete.

Required adversarial tests include:

1. symlink escape at every path component;
2. symlink or junction swap after validation but before open;
3. directory rename/substitution after open but before disposition;
4. target replacement between quarantine and deletion;
5. case-folding alias and Windows reparse/junction alias;
6. cross-device quarantine refusal;
7. owner generation ABA race;
8. stale PID reuse with mismatched start ticks;
9. stale heartbeat after a newer generation;
10. task transition from `finished` back to a nonterminal review episode before
    deletion;
11. active child/dependent appearing before final revalidation; and
12. crash after every durability boundary, followed by idempotent recovery.

## 8. Finite dead-letter/quarantine policy

Proposed `dead_letter_retention_seconds` accepts integer seconds only:

```text
default = 604800
minimum = 3600
maximum = 2592000
infinite = forbidden
```

Booleans, floats, strings, null, negative values, zero, values below 3600,
values above 2592000, overflow, and “infinite” sentinels are rejected. Values
are not silently clamped.

Disposition uses an audit record independent of the disposable artifact:

1. Under the owner transaction, create a unique disposition ID.
2. Atomically publish and `fsync` a `PREPARED` record below
   `out/logs/build-registry/dispositions/`, including owner proof,
   nonce/generation, source and target file identities, reason, policy value,
   size, and hashes.
3. Atomically rename the exact task tree to
   `out/.quarantine/<disposition-id>` on the same filesystem and `fsync` both
   parent directories.
4. Atomically replace the external record with `COMMITTED` and `fsync` it.
5. After finite retention, revalidate all authority and identities, delete
   using the no-follow directory-handle protocol, and atomically publish a
   terminal `DELETED` record. The record survives artifact deletion.

Recovery reconciles `PREPARED` and `COMMITTED` records using recorded file
identity and current owner state. It never guesses whether an interrupted
rename or delete completed.

Mandatory tests cover exact default/min/max acceptance; noninteger,
below-minimum, above-maximum, overflow and infinite rejection; record
persistence after artifact deletion; crash before/after each atomic rename and
`fsync`; concurrent disposition; retention-boundary races; restore versus
delete; task-state and lease changes; and delete-time path substitution.

## 9. Audited producers and consumers

The six build-producing entry points from
`docs/BUILD_OUTPUT_CONSOLIDATION_AUDIT_012.md` are:

| Producer | Observed legacy behavior | Required owner-backed result |
|---|---|---|
| `audit/run_full_audit.sh` | root `build-audit`, `build-audit-asan`, timestamped audit output | Resolve a real registered task claim; task build under `.tasks`; logs/packages under `out`. |
| `audit/run_full_audit.ps1` | same root patterns on PowerShell | Same contract and owner proof. |
| `ci/generate_audit_package.sh` | `out/auditor`, timestamped evidence | Registered task claim for task runs; evidence/package roots under `out`. |
| `ci/generate_audit_package.ps1` | root `build-audit`, root evidence | Registered task claim and canonical evidence/package roots. |
| `ci/local-ci.sh` | many `${TMPDIR}/build-local-ci-$$*` roots and root `local-ci-output`; local cryptofuzz and KLEE jobs | Every configuration uses the caller’s actual registered task ID and owner proof; KLEE/cryptofuzz temporary/build/results move below that task/log root. |
| `ci/run-local-ci.ps1` | Docker image orchestration, root `local-ci-output`, persistent `uf-ccache` volume | Pass the registered claim into the container; bind the canonical cache/output roots; no independent named-volume authority. |

No producer may generate a pseudo task ID from a run ID, PID, timestamp, branch,
or directory name. If no registered task/owner proof is supplied, the producer
must refuse the task-owned build mode.

Mandatory topology consumers/orchestrators are:

- `ci/_ufsecp.py`: remove legacy library search roots and consume only an
  owner-resolved primary preset or task configuration, unless the user supplies
  an explicit verified library path;
- `ci/audit_gate.py`: stop pruning legacy names directly and stop defaulting
  mutation builds to `build_opencl`;
- `.github/workflows/klee.yml` and `.github/workflows/cryptofuzz.yml`: consume
  the same registered task/root rules for CI scratch, logs, and results;
- `CMakePresets.json`: already uses `out/${presetName}` for primary presets and
  remains unchanged unless independent evidence requires a change; and
- `.gitignore`: remains a masking mechanism, never a hygiene or ownership
  authority.

The hygiene gate scans every registered checkout/worktree, including ignored
paths, for `CMakeCache.txt`, `build.ninja`, `compile_commands.json`, `build*`
directories, `cmake-build-*`, and `local-ci-output`. It reports but does not
delete. It verifies each CMake cache’s `CMAKE_HOME_DIRECTORY` against the owner
manifest and fails on any persistent build artifact outside physical
`CANONICAL_OUT`.

## 10. VS Code contract

The only repository VS Code files in scope are:

```text
.vscode/settings.json
.vscode/tasks.json
.vscode/launch.json
```

One proposed, versioned resolver publishes these two runtime artifacts:

```text
out/.cache/vscode/active-build.v1.json
out/.cache/vscode/compile_commands.json
```

`active-build.v1.json` binds the verified repo ID, primary root, registered task
ID or primary preset, configuration ID, owner nonce/generation, selected build
root, source digest, and compile-database digest. The resolver must obtain the
selection through the supplied public owner API; it must not infer a directory
from the workspace name or environment.

Publication protocol for both artifacts is: write a unique temporary file in
the same `out/.cache/vscode` directory, validate complete JSON, flush and
`fsync` the file, atomic rename/replace, then `fsync` the directory. A failed
configure never replaces the last complete compile database. The source tree
gets no copied or symlinked `compile_commands.json`.

All three VS Code files consume the one resolver selection:

- `settings.json` points CMake/C++ tooling at the stable resolver and
  `out/.cache/vscode/compile_commands.json`, never a root build folder;
- `tasks.json` defines the single resolve task and makes configure, build, test,
  and debug preparation depend on it; each task passes the resolver artifact’s
  version/digest back to the owner client for revalidation; and
- `launch.json` uses the exact build task as `preLaunchTask` and refuses launch
  if the resolver version/digest, executable containment, owner generation, or
  configuration ID is stale.

Changing preset/task is one owner-mediated resolver transaction. Readers either
observe the previous complete artifact pair or the new complete pair, never a
partially written file or mixed generation. If pairwise publication cannot be
made atomic on the platform, a small atomic generation pointer is published
last and both versioned files are addressed through that generation.

## 11. Hygiene, manifest, and final validation

Every owner/run manifest is versioned and contains the three distinct identity
groups from §5. Manifest publication uses temporary file, file `fsync`, atomic
rename, and directory `fsync`. Unknown fields are preserved for forward
compatibility; missing mandatory fields fail closed.

Before any implementation is accepted, tests must demonstrate:

- two tasks with the same configuration build concurrently in distinct task
  trees and share only compiler-cache hits;
- dirty-content and recursive dependency changes alter configuration ID;
- owner tuple changes do not alter configuration ID or cache namespace;
- incompatible compiler/ABI flags change the cache namespace;
- producer and consumer paths resolve to the same owner proof;
- no producer creates persistent output in a secondary worktree or `/tmp`;
- task rejection does not trigger disposition;
- only `finished`/`archived` can reach finite dead-letter disposition;
- all §7 and §8 adversarial cases pass on Linux and applicable Windows/macOS
  path primitives; and
- the hygiene scan finds the six known repository-root legacy names without
  deleting them.

## 12. Exact sequential downstream DAG

Every card below has an implicit `forbidden = all repository paths except the
listed allowed_writes`. The lists are exact and pairwise non-overlapping. Cards
run strictly in the listed dependency order; none may be started early.

### A0 — `BUILD_OWNER_AUTHORITY_HANDOFF_029` (sole root gate)

- `depends_on`: `[]`
- `allowed_writes`:
  - `docs/BUILD_OWNER_AUTHORITY_HANDOFF_029.json`
- required content: owner source root, immutable source commit, per-file hashes,
  actual exported symbol names for the five §6 capabilities, lock/transaction
  scope, persistent schema, error model, supported platforms, and authoritative
  test commands.
- validation:
  - JSON parses and has a versioned schema;
  - every path is beneath the supplied source root;
  - every hash matches an independently readable source file;
  - every supplied symbol and test exists at that commit;
  - no path points into an installed VSIX/extension copy; and
  - independent Codex review confirms the source provides atomic claim,
    nonce/generation, heartbeat, revalidate, disposition, and safe delete.
- gate: until accepted, **A1–A14 are BLOCKED**. This evidence card does not
  implement an API and authorizes no cleanup.

### A1 — `BUILD_OWNER_CLIENT_HELPER_030`

- `depends_on`: `BUILD_OWNER_AUTHORITY_HANDOFF_029`
- `allowed_writes`:
  - `ci/build_owner_client.py`
  - `ci/build_root_resolver.py`
  - `ci/test_build_owner_client.py`
  - `ci/test_build_root_resolver.py`
- validation:
  - `python3 -m unittest ci.test_build_owner_client ci.test_build_root_resolver`
  - tests use only the real public symbols recorded by A0;
  - fail-closed WAL, identity, nonce/generation, atomic manifest, resolver-pair,
    and concurrent-task tests pass.

### A2 — `BUILD_HYGIENE_GATE_031`

- `depends_on`: `BUILD_OWNER_CLIENT_HELPER_030`
- `allowed_writes`:
  - `ci/build_hygiene_gate.py`
  - `ci/test_build_hygiene_gate.py`
- validation:
  - `python3 -m unittest ci.test_build_hygiene_gate`
  - `python3 ci/build_hygiene_gate.py --check --no-delete`
  - containment, ignored-path, CMake-home mismatch, secondary-out, symlink, and
    outside-root fixtures pass.

### A3 — `BUILD_MIGRATE_AUDIT_SH_032`

- `depends_on`: `BUILD_HYGIENE_GATE_031`
- `allowed_writes`: `audit/run_full_audit.sh`
- validation:
  - `bash -n audit/run_full_audit.sh`
  - owner-client integration test proves both normal and sanitizer trees use
    the registered task root and emit no root `build-audit*`.

### A4 — `BUILD_MIGRATE_AUDIT_PS1_033`

- `depends_on`: `BUILD_MIGRATE_AUDIT_SH_032`
- `allowed_writes`: `audit/run_full_audit.ps1`
- validation:
  - PowerShell parser validation succeeds;
  - owner-client integration test proves normal/sanitizer path parity with A3.

### A5 — `BUILD_MIGRATE_PACKAGE_SH_034`

- `depends_on`: `BUILD_MIGRATE_AUDIT_PS1_033`
- `allowed_writes`: `ci/generate_audit_package.sh`
- validation:
  - `bash -n ci/generate_audit_package.sh`
  - build, evidence, manifest, and archive paths pass the hygiene gate.

### A6 — `BUILD_MIGRATE_PACKAGE_PS1_035`

- `depends_on`: `BUILD_MIGRATE_PACKAGE_SH_034`
- `allowed_writes`: `ci/generate_audit_package.ps1`
- validation:
  - PowerShell parser validation succeeds;
  - build/evidence/package path parity with A5 passes.

### A7 — `BUILD_MIGRATE_LOCAL_CI_SH_036`

- `depends_on`: `BUILD_MIGRATE_PACKAGE_PS1_035`
- `allowed_writes`: `ci/local-ci.sh`
- validation:
  - `bash -n ci/local-ci.sh`
  - all local CI variants, including KLEE and cryptofuzz, use the caller’s
    existing registered task ID;
  - no pseudo task ID, `${TMPDIR}/build-local-ci-*`, KLEE/cryptofuzz persistent
    `/tmp` tree, or root `local-ci-output` is created.

### A8 — `BUILD_MIGRATE_LOCAL_CI_PS1_037`

- `depends_on`: `BUILD_MIGRATE_LOCAL_CI_SH_036`
- `allowed_writes`: `ci/run-local-ci.ps1`
- validation:
  - PowerShell parser validation succeeds;
  - container receives the exact owner proof;
  - canonical `out` and cache bindings replace root output and `uf-ccache`.

### A9 — `BUILD_CONSUMER_PATHS_038`

- `depends_on`: `BUILD_MIGRATE_LOCAL_CI_PS1_037`
- `allowed_writes`:
  - `ci/_ufsecp.py`
  - `ci/audit_gate.py`
- validation:
  - `python3 -m py_compile ci/_ufsecp.py ci/audit_gate.py`
  - fixtures prove no legacy search/prune/default build path remains;
  - nonmatching or stale owner proof is refused.

### A10 — `BUILD_KLEE_CRYPTOFUZZ_WORKFLOWS_039`

- `depends_on`: `BUILD_CONSUMER_PATHS_038`
- `allowed_writes`:
  - `.github/workflows/klee.yml`
  - `.github/workflows/cryptofuzz.yml`
- validation:
  - workflow syntax validation succeeds;
  - every build/result/cache path is owner-resolved below canonical `out`;
  - artifact upload consumes canonical logs/packages only.

### A11 — `BUILD_VSCODE_ATOMIC_SELECTION_040`

- `depends_on`: `BUILD_KLEE_CRYPTOFUZZ_WORKFLOWS_039`
- `allowed_writes`:
  - `.vscode/settings.json`
  - `.vscode/tasks.json`
  - `.vscode/launch.json`
- validation:
  - all three files parse as JSON;
  - all three reference the same resolver schema/version and resolve task;
  - resolver/compile database atomic-publication, stale-generation, interrupted
    configure, and task-switch tests pass;
  - no fourth `.vscode` file or source-root compile database is created.

### A12 — `BUILD_REVIEW_CI_HOOKS_041`

- `depends_on`: `BUILD_VSCODE_ATOMIC_SELECTION_040`
- `allowed_writes`:
  - `.github/workflows/ci.yml`
  - `.github/workflows/preflight.yml`
- validation:
  - workflow syntax validation succeeds;
  - both run `ci/build_hygiene_gate.py --check --no-delete`;
  - a seeded outside-`out` artifact fails review CI;
  - the gate never deletes or quarantines.

### A13 — `BUILD_SAFE_CLEANUP_042`

- `depends_on`: `BUILD_REVIEW_CI_HOOKS_041`
- `allowed_writes`:
  - `ci/clean_local_artifacts.sh`
- validation:
  - `bash -n ci/clean_local_artifacts.sh`
  - default and `--dry-run` modes are report-only;
  - mutation mode can execute only through the A0 public owner proof;
  - all §7 containment and §8 dead-letter tests pass;
  - active/review/blocked/pending/rejected tasks and shared caches are retained.

### A14 — `BUILD_REGISTRY_FINAL_REVIEW_043`

- `depends_on`: `BUILD_SAFE_CLEANUP_042`
- `allowed_writes`:
  - `docs/BUILD_REGISTRY_IMPLEMENTATION_REVIEW_043.md`
- validation:
  - records exact commits/hashes and validation results for A0–A13;
  - reruns full tests and `python3 ci/build_hygiene_gate.py --check --no-delete`;
  - proves the six legacy producer paths are no longer generated;
  - stops at Codex review. Promotion and any first cleanup run require separate
    coordinator authorization.

## 13. Acceptance self-check

- [x] Canonical submodule, repo ID, its own manifests/task DB, absolute
  common-dir, and relative `core.worktree` are distinct; parent registry is
  excluded.
- [x] Bundle 4 / acknowledgement 2, exact delivered Source Graph facts, and
  `7e620d1` delivered baseline versus `bf33cfb` review-time primary are recorded.
- [x] Exact installed symbols, process lock/artifacts, GC sets, and real
  terminal semantics are mapped without build behavior invention.
- [x] Live `mode=ro` WAL/sidecar behavior, sandbox SHM failure, and immutable
  stale-data prohibition are explicit.
- [x] Private process lock/workspace GC are excluded from build authority;
  required public capability semantics are proposed and BLOCKED on source.
- [x] Exact physical topology and separate owner/config/cache identities include
  dirty content, recursive dependencies, features, and toolchain inputs.
- [x] Restart-safe lease, PID/start ticks, nonce/generation, durability,
  no-follow containment, races, and bounded terminal GC are specified.
- [x] The exact six producers and mandatory consumers require registered task
  ownership; no pseudo local-CI task ID is allowed.
- [x] Finite 604800/3600/2592000 integer-only dead-letter policy includes
  invalid-input, persistence, atomicity, race, recovery, and substitution tests.
- [x] The exact three VS Code files consume one versioned resolver selection;
  resolver and compile database publication are durable and atomic.
- [x] A0–A14 form a strict sequential DAG with exact pairwise-disjoint write
  sets, dependencies, and validations.
- [x] Specification is GO for review; implementation and cleanup remain BLOCKED
  solely on A0’s missing public build-owner source authority. No deletion is
  authorized.
