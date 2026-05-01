# AGENTS.md -- UltrafastSecp256k1

Use the SQLite project graph before broad file search.

## Canonical Graph

- DB: `.project_graph.db`
- Rebuild:

```bash
python3 ci/build_project_graph.py --rebuild
```

## Preferred Workflow

1. Query graph first.
2. Read only the files or line ranges the graph points to.
3. After structural changes, rebuild the graph.
4. Before finishing, rerun `preflight.py` if the change is substantial.

## Session State Recovery

For non-trivial work, restore session state before making assumptions:

```bash
python3 ../../tools/session_state/session_state.py bootstrap --format text
```

Record durable handoff checkpoints after meaningful progress:

```bash
python3 ../../tools/session_state/session_state.py checkpoint \
  --summary "<what changed>" \
  --next "<next steps>"
```

Mandatory update triggers:

- new substantial task starts
- before planned file edits
- after file edits
- blocker found or resolved
- important decision made
- tests/validation run
- source graph or memory repaired
- before handoff or final response on non-trivial work

Use `decision`, `blocker`, and `file` subcommands for important context. Do not
checkpoint every tiny read or trivial reply. Checkpoint when losing the chat
would force the next model to rediscover context or repeat work. Do not store
secrets, private keys, credentials, or tokens.

## Most Useful Commands

```bash
python3 ci/query_graph.py context cpu/src/ct_sign.cpp
python3 ci/query_graph.py impact cpu/src/ecdh.cpp
python3 ci/query_graph.py routing ecdsa_sign
python3 ci/query_graph.py tags
python3 ci/query_graph.py tag constant_time
python3 ci/query_graph.py symbol ecdsa_sign
python3 ci/query_graph.py optimize 15
python3 ci/query_graph.py risk 15
python3 ci/query_graph.py gpuwork 15
python3 ci/query_graph.py fragile 15
python3 ci/query_graph.py hotspots 20
python3 ci/query_graph.py coverage ecdsa_sign
```

## Reasoning Layers

The graph includes more than structure. It also includes:

- semantic classification
- secret/CT metadata
- parser-boundary sensitivity
- performance/gpu-candidate scoring
- audit coverage
- change history
- risk/gain/optimization priority

Important tables/views:

- `semantic_tags`
- `entity_tags`
- `symbol_semantics`
- `symbol_security`
- `symbol_performance`
- `symbol_audit_coverage`
- `symbol_history`
- `symbol_scores`
- `v_symbol_reasoning`

## Rules

- Do not claim CT guarantees without checking the graph and the relevant tests.
- Do not claim audit coverage without checking `function_test_map` or `symbol_audit_coverage`.
- Do not change ABI-visible or secret-bearing code blindly; query `routing`, `bindings`, and `fragile` first.
- If you add new graph-worthy entities, update the graph builder.

## Security Guardrails (Strict)

These rules apply to every model and every agent working in this library.

1. **CT signing is mandatory for secret-bearing signing paths.**
   Public C++ APIs, C ABI wrappers, bindings, wallet integrations, recovery signing,
   ECDSA, Schnorr, Ethereum, Bitcoin message signing, ECIES, and batch signing must
   route private-key and nonce operations through `secp256k1::ct::*` primitives.
   Do not add or recommend default signing paths that call variable-time
   `Point::generator().scalar_mul()`, `Scalar::inverse()`, wNAF, GLV fast paths,
   or branchy recovery-ID logic on secrets.

2. **Non-CT public C++ signing is a security finding.**
   If `cpu/src/ecdsa.cpp`, `cpu/src/recovery.cpp`, or public headers expose a
   normal-looking non-CT signing API, either make CT the default or add an explicit
   opt-in guard, warning, deprecation, or compile-time restriction. Never claim
   production-safe signing while default public C++ signing is non-CT.

3. **Batch signing failure must be fail-closed.**
   All batch sign APIs must clear output buffers before processing and must not
   leave partial valid signatures visible after an error. Keep Schnorr and ECDSA
   batch semantics identical for null args, zero count, invalid key, overflow,
   and partial-failure behavior.

4. **Zero signatures from signing internals are ABI errors.**
   If a CT signing primitive returns `r == 0`, `s == 0`, or an all-zero Schnorr
   signature, ABI wrappers must return a non-OK error such as `UFSECP_ERR_INTERNAL`
   and must not serialize the zero signature as success.

5. **Graph evidence must match security claims.**
   Before claiming CT coverage, query source graph `symbol`, `coverage`, `risk`,
   and relevant `function_test_map` / `symbol_audit_coverage` entries. ABI wrappers
   that dispatch to CT implementations must have CT evidence mapped in the graph.
   If metadata is stale or wrong, repair the graph/config and rebuild it.

6. **Corrupt benchmark artifacts are not evidence.**
   Reject benchmark files with zero timings, impossible throughput, malformed
   columns, or concatenated numeric fields. Regenerate benchmarks with sanity
   checks before using them in performance, release, audit, or marketing claims.

7. **Documentation and tests must change with security fixes.**
   Any change to CT signing, ABI failure semantics, batch APIs, parser behavior,
   or benchmark evidence must update the matching docs and audit tests in the
   same commit.
