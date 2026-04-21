# Reviewer Prompts (CAAS H-10)

Standardised, graph-aware prompts for AI-assisted review. Each prompt
assumes the reviewer has access to:

- The diff under review.
- `python3 tools/source_graph_kit/source_graph.py` (focus, slice, impact,
  symbols, calls, coverage, hotspots, bodygrep).
- The pre-built artifacts under `build_rel/` and `build-cuda/`.

Prompts are deliberately short and load-bearing. Do **not** edit them to
soften the tone — empirically the strongest findings come from the
grumpiest prompt.

| File | Lens | Best for |
|------|------|----------|
| [`auditor.md`](auditor.md) | Find a real exploitable bug | Every PR touching crypto, parsers, or ABI |
| [`attacker.md`](attacker.md) | Build a PoC for the worst case | Diffs touching parsers, error paths, key handling |
| [`perf_skeptic.md`](perf_skeptic.md) | Challenge every perf claim with a measurement | Diffs whose commit message claims a speedup |
| [`docs_skeptic.md`](docs_skeptic.md) | Find a doc claim that the code now contradicts | Diffs that change behaviour but skip docs |

## Usage

Pick **one** prompt per review pass. Combining lenses dilutes signal —
the auditor finds different things than the docs-skeptic. Run them as
separate passes if budget allows.

## Source-graph reminder

All four prompts assume the graph-first rule from `AGENTS.md` and
`.github/copilot-instructions.md`. Do not let a reviewer regress to
`grep` / `rg` / `find` while the graph is available — start with
`focus <term> 24 --core` and escalate from there.

## Provenance

These prompts are part of CAAS hardening item **H-10** (see
[`docs/CAAS_HARDENING_TODO.md`](../CAAS_HARDENING_TODO.md)). Edits to the
prompts must be reflected in the dated `docs/AUDIT_CHANGELOG.md` entry
the same day so reviewers can verify they're using the latest revision.
