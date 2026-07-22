# UFCI Git-environment hardening specification

## Status and implementation gate

This document records the reproduced failure and the required tracked fix. It does
not authorize a CI change. The canonical Source Graph query `mode=focus`,
`bundle_type=explore`, `query=ci`, `budget=48` returned `hit_count=0` from the sole
canonical authority. This is the accepted **UFSG_RESEARCH_005** zero-hit blocker.
Because the mandatory code-task context is empty, no raw filesystem search may be
used to bypass it and no CI implementation may begin. Restore and acknowledge a
non-empty canonical Source Graph bundle for the targets below before creating a
separate implementation card.

## Reproduced failure

The caller can enter the local pre-push path with repository-scoped Git environment
variables inherited from a hook. A nested Git command that appears to address a
different checkout with `git -C <checkout> ...` still obeys inherited variables such
as `GIT_DIR` and `GIT_WORK_TREE`; `-C` does not sanitize them.

The fast gate therefore asks Git about the inherited repository instead of the
checkout named by `-C`. Its work-tree probe succeeds falsely, so setup/validation is
skipped. Later configuration operates on the inherited canonical Git directory. In
the reproduced incident this unintentionally wrote `core.bare=true` into the
canonical checkout or submodule config, making that working checkout unusable.

The required invariant is precise: the actual canonical checkout/submodule starts
with `core.bare=false`, remains `core.bare=false`, and remains usable as a work tree
after a run with a hostile inherited Git environment. A separate foreign bare
repository may start with `core.bare=true`; it is only a sentinel and its config and
contents must remain untouched. It must never be confused with the canonical
checkout/submodule.

## Sanitization contract

At the local-hook boundary, before any repository selection, fast gate, submodule
operation, configuration, or delegated CI command, remove every name reported by
`git rev-parse --local-env-vars`. Do this by iterating over that command's one-name-
per-line output and calling `unset` for each name; do not hard-code only the variables
present in one developer's environment. The Git-local variables that the
implementation and regression test must cover are:

- `GIT_ALTERNATE_OBJECT_DIRECTORIES`
- `GIT_COMMON_DIR`
- `GIT_CONFIG`
- `GIT_CONFIG_COUNT`
- `GIT_CONFIG_PARAMETERS`
- `GIT_DIR`
- `GIT_GRAFT_FILE`
- `GIT_IMPLICIT_WORK_TREE`
- `GIT_INDEX_FILE`
- `GIT_NO_REPLACE_OBJECTS`
- `GIT_OBJECT_DIRECTORY`
- `GIT_PREFIX`
- `GIT_REPLACE_REF_BASE`
- `GIT_SHALLOW_FILE`
- `GIT_WORK_TREE`

Sanitization must occur in a subshell or otherwise preserve the caller's environment.
After sanitization, resolve the intended repository from an explicit path and keep
using explicit `git -C <resolved-path>` operands. Configuration-injection variables
(`GIT_CONFIG_KEY_<n>`/`GIT_CONFIG_VALUE_<n>` members associated with a nonzero
`GIT_CONFIG_COUNT`) must also be rejected or cleared before invoking Git. Clear
repository-routing variables that Git versions add to `--local-env-vars` even when
they are absent from this explicit compatibility list.

## Tracked implementation and test scope

Once the Source Graph gate is restored, the implementation card is limited to the
tracked `ci_local` entry point that performs the pre-push fast gate and its tracked
regression test. The restored bundle must supply and confirm their exact repository
paths; this blocked research card deliberately does not guess paths or create files.
No global hook installation, developer `.git/config` edit, generated artifact, or
unrelated CI behavior belongs in scope.

The regression test must create three isolated repositories: the canonical checkout,
its real submodule/worktree target, and an unrelated foreign bare sentinel. It must:

1. Explicitly set and assert `core.bare=false` in the canonical checkout and
   submodule before invoking `ci_local`.
2. Set and assert `core.bare=true` in the foreign bare sentinel, then export hostile
   Git-local variables pointing at it (including at least `GIT_DIR`, `GIT_WORK_TREE`,
   `GIT_COMMON_DIR`, `GIT_OBJECT_DIRECTORY`, `GIT_INDEX_FILE`, and replace-ref/
   alternate-object-directory cases).
3. Run the real tracked pre-push/`ci_local` path, exercising the fast gate rather than
   a copied helper.
4. Assert the canonical checkout and submodule still report `core.bare=false`, pass
   `git rev-parse --is-inside-work-tree`, and remain usable for ordinary status/index
   operations.
5. Assert the foreign sentinel still reports `core.bare=true` and that its config,
   refs, index/object state, and contents are byte-for-byte or hash-identical to the
   pre-run snapshot.
6. Exercise every variable listed in the sanitization contract and the config-
   injection family, both individually where routing differs and together as the
   reproduced hostile environment.

## Validation

For this specification-only card, run exactly:

```sh
git diff --check
```

The future implementation card must run the exact tracked regression-test command
identified by the restored canonical Source Graph bundle, followed by
`git diff --check`. Until that bundle exists, inventing a test path or command would
bypass the mandatory code-task gate.
