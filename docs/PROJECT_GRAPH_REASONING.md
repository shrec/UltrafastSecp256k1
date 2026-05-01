# Project Graph Reasoning

UltrafastSecp256k1 ships with a SQLite-backed project graph at:

- `.project_graph.db`

This is not only a code index. It is a cryptographic engineering knowledge base
used by humans and AI agents to reason about:

- subsystem boundaries
- constant-time and secret-bearing paths
- parser boundaries
- audit coverage
- optimization opportunity vs. audit risk
- CPU-to-GPU candidate selection
- change-history-sensitive review targets

## Rebuild

```bash
python3 ci/build_project_graph.py --rebuild
```

## What The Graph Contains

Core structural layers:

- source files
- include dependencies
- C ABI functions
- ABI routing
- C++ methods
- test targets
- audit modules
- docs
- call graph
- function-to-test map

Reasoning layers:

- `semantic_tags`
- `entity_tags`
- `symbol_semantics`
- `symbol_security`
- `symbol_performance`
- `symbol_audit_coverage`
- `symbol_history`
- `symbol_scores`
- `v_symbol_reasoning`

## Semantic Classification

Symbols and files are tagged with higher-level meaning, for example:

- `field_arithmetic`
- `scalar_arithmetic`
- `point_arithmetic`
- `modinv`
- `ecdsa`
- `schnorr`
- `ecdh`
- `bip352`
- `hashing`
- `ffi_abi`
- `wallet_flow`
- `gpu_acceleration`
- `constant_time`
- `parser_boundary`
- `audit_evidence`

This allows graph queries to operate on intent, not only filenames.

## Security Metadata

Each symbol may carry reasoning fields such as:

- `uses_secret_input`
- `must_be_constant_time`
- `public_data_only`
- `device_secret_upload`
- `requires_zeroization`
- `invalid_input_sensitive`

This makes it possible to separate:

- public verification code
- secret-bearing signing / derivation code
- parser-boundary code
- GPU/offload-sensitive flows

## Performance Metadata

Each symbol also gets estimated engineering metadata:

- `hotness_score`
- `estimated_cost`
- `batchable`
- `vectorizable`
- `gpu_candidate`
- `memory_bound`
- `compute_bound`
- `duplicated_backends`

This is heuristic, not a benchmark replacement. It is meant to guide review and
optimization triage.

## Audit Coverage Metadata

The graph records whether a symbol is covered by:

- unit tests
- fuzzing
- invalid-vector style tests
- CT tests
- cross-implementation differential checks
- GPU-equivalence style checks
- regression-style tests

It also stores:

- `last_audit_result`
- `times_failed_historically`
- `known_fragile`

## History Layer

The graph uses git history to derive:

- `times_modified`
- `recently_modified`
- `bug_fix_count`
- `performance_tuning_count`
- `audit_related_changes`

This helps prioritize review on recently changing, fragile, or heavily tuned
paths.

## Risk And Gain Scoring

Every symbol gets:

- `risk_score`
- `gain_score`
- `optimization_priority`

The current scoring is heuristic and intended for triage, not for automated
proof of correctness. It is useful for answering:

- which optimization candidates are high-gain and relatively lower-risk
- which secret-bearing or parser-sensitive symbols deserve review first
- which CPU paths look like strong GPU-offload candidates

## Query Commands

### Structural

```bash
python3 ci/query_graph.py context src/cpu/src/ct_sign.cpp
python3 ci/query_graph.py impact src/cpu/src/ecdh.cpp
python3 ci/query_graph.py callgraph pippenger_msm
python3 ci/query_graph.py coverage ecdsa_sign
```

### Semantic

```bash
python3 ci/query_graph.py tags
python3 ci/query_graph.py tag constant_time
python3 ci/query_graph.py symbol ecdsa_sign
```

### Optimization / Audit Triage

```bash
python3 ci/query_graph.py optimize 20
python3 ci/query_graph.py risk 20
python3 ci/query_graph.py gpuwork 20
python3 ci/query_graph.py fragile 20
```

## Recommended Workflow

### Before editing a file

```bash
python3 ci/query_graph.py context <file>
python3 ci/query_graph.py impact <file>
```

### Before touching secret-bearing code

```bash
python3 ci/query_graph.py security <file>
python3 ci/query_graph.py fragile 20
python3 ci/query_graph.py tag constant_time
```

### Before changing the C ABI

```bash
python3 ci/query_graph.py routing <name>
python3 ci/query_graph.py bindings
python3 ci/query_graph.py tag ffi_surface
```

### Before proposing an optimization

```bash
python3 ci/query_graph.py optimize 20
python3 ci/query_graph.py gpuwork 20
python3 ci/query_graph.py risk 20
```

## Machine-Readable Export

The reasoning graph is exported through:

```bash
python3 ci/export_assurance.py -o assurance_report.json
```

The exported JSON includes:

- semantic tag inventory
- reasoning summary by category/backend
- optimization candidates
- risk hotspots

## Notes

- The reasoning layer is intentionally heuristic.
- It should guide review and AI assistance, not replace cryptographic judgment.
- If graph builder logic changes, rebuild the DB before relying on query output.
