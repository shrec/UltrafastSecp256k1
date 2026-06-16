# Research Monitor

Scheduled early-warning intake for secp256k1-related papers, advisories, attack
surfaces, implementation bugs, and security-relevant updates.

The goal is not to outsource assurance. The goal is to continuously ingest fresh
external signals, compare them against the repository's existing audit surfaces,
and immediately escalate anything that looks new, unmapped, or worth upgrading
into deterministic evidence.

## What it does

The monitor runs `ci/research_monitor.py` and:

1. fetches recent secp256k1-related items from external sources
   - IACR Cryptology ePrint RSS
   - arXiv
   - Crossref
   - NVD CVEs
2. classifies them against `docs/RESEARCH_SIGNAL_MATRIX.json`
3. marks each item as one of:
   - `covered`
   - `candidate`
   - `gap`
   - `out_of_scope`
   - `unmapped`
4. emits:
   - machine-readable JSON
   - Markdown and text reports
   - mail subject/body files
5. opens a GitHub issue for escalated findings
6. optionally sends an SMTP email when escalated findings are found

### Actionable finding body (Bastion B6, routed in B18)

Each high-confidence finding renders not just a citation but a concrete next
step, so an escalated issue is opened ready to act on:

- **Attack class** — the matched signal class's `attack_class` (one of the 16-value
  `attack_class_enum`), so every finding carries a stable taxonomy label rather than
  a free-text guess.
- **Affected primitive / surface** — derived from the repo signal-class matches
  (`gap` / `candidate` / `covered` / …); `unmapped` when no class matched.
- **Expected gate** — the `audit_gate.py` sub-check (or `ci/*.py` script) that should
  catch this attack class, so the finding routes itself to the gate that owns it.
- **Existing evidence** — the repo evidence paths the matched signal classes point
  to, so a reviewer sees current coverage at a glance.
- **Patch plan** — the matched class's `missing_evidence_action` first (the exact
  evidence to add), then a "route to gate: re-run `audit_gate.py --…`" step, plus a
  `First verification` command (`python3 ci/research_monitor.py --lookback-days N …`)
  and a source-inspect command.

Source status and source errors are recorded per expanded query, for example
`Crossref [secp256k1]` versus `Crossref [libsecp256k1]`. This keeps failures
or empty result windows attributable to the exact source/query pair instead of
collapsing all expanded searches into repeated source names.

Keyword, query, and signal-matrix matches use term or phrase boundaries. This
prevents short crypto terms from matching inside unrelated words, such as `ROS`
inside ordinary prose or `ECIES` inside `species`.

Crossref metadata is parsed defensively: partial or malformed `date-parts`
arrays are normalized to a valid UTC date before filtering, and source error
messages are whitespace-compacted and bounded before they are written into
reports.

Escalation behavior:

- `high_confidence` always opens a GitHub issue when issue creation is enabled
- `needs_review` opens a GitHub issue when review escalation is enabled
- `discarded` stays in the artifact only

Issue creation checks for an existing same-day `Research Monitor` issue before
creating a new one. If repository labels are missing or temporarily unavailable,
the workflow retries issue creation without labels instead of losing the
escalated signal.

This keeps the workflow focused on items that may require new tests, new proof
artifacts, taxonomy expansion, or explicit owner review. The monitor opens an
issue rather than a pull request because research signals do not yet contain a
code diff, and repository branch policy keeps implementation work on `dev`.

## Workflow

See `.github/workflows/research-monitor.yml`.

Default behavior:

- scheduled daily
- manual dispatch supported
- uploads the generated report as an artifact
- writes a summary into the GitHub Actions job summary
- opens an issue when high-confidence findings exist
- opens an issue for needs-review findings when `open_review_issue` is enabled
- sends email for the same escalated finding set when SMTP secrets are configured

## Signal Matrix

`docs/RESEARCH_SIGNAL_MATRIX.json` is the canonical external-signal taxonomy.

The top level declares `attack_class_enum` — the 16 stable attack classes a signal
may belong to (`nonce_bias_or_reuse`, `signature_malleability`, `parser_boundary`,
`invalid_curve_or_pubkey`, `scalar_domain`, `batch_verification`, `side_channel_ct`,
`gpu_backend_parity`, `protocol_state_machine`, `threshold_multisig`, `supply_chain`,
`fuzz_crash`, `integration_consensus`, `benchmark_claim`, `hardware_fault_or_em`,
`out_of_scope`).

Each class records:

- keyword match rules
- `attack_class` (must be one of `attack_class_enum`)
- `affected_primitive` and `affected_surface`
- `expected_evidence` — the repo evidence paths that cover the class
- `expected_gate` — the `audit_gate.py` sub-check or `ci/*.py` script that catches it
- current stance (`covered`, `candidate`, `gap`, `out_of_scope`, `original_analysis`)
- `missing_evidence_action` (for `candidate`) / rationale (for `out_of_scope`)

### Taxonomy gate (Bastion B18, G-18)

`ci/check_research_signal_matrix.py` (`audit_gate.py --research-signal-matrix`)
enforces the taxonomy so research intake becomes routed audit work, not a bare
coverage label:

- every in-scope class carries an `attack_class` drawn from `attack_class_enum`;
- `covered` / `original_analysis` classes must have existing `expected_evidence`
  and an `expected_gate` that resolves (an `audit_gate.py` CHECK_MAP flag or a
  `ci/*.py` script that exists);
- `candidate` classes must declare a `missing_evidence_action`;
- `out_of_scope` classes must carry a rationale.

If a new failure or optimization class becomes relevant, update this matrix
first. The gate validates that every referenced repo evidence path exists and that
every class routes to a real gate.

## SMTP Secrets

To enable email notifications, configure these repository secrets:

- `RESEARCH_SMTP_HOST`
- `RESEARCH_SMTP_PORT`
- `RESEARCH_SMTP_USERNAME`
- `RESEARCH_SMTP_PASSWORD`
- `RESEARCH_REPORT_FROM`
- `RESEARCH_REPORT_TO`

Optional variables or secrets:

- `RESEARCH_SMTP_USE_SSL` (`true` or `false`, default `false`)
- `RESEARCH_REPORT_REPLY_TO`

If the SMTP secrets are missing, the workflow still generates the artifact and
job summary but skips mail delivery.

## Local Run

```bash
python3 ci/research_monitor.py \
  --lookback-days 14 \
  --max-results 10 \
  --output-dir build/research_monitor
```

Dry-run the mailer locally:

```bash
python3 ci/send_smtp_report.py \
  --subject-file build/research_monitor/mail_subject.txt \
  --body-file build/research_monitor/mail_body.txt \
  --dry-run
```

## Philosophy Fit

This workflow is intentionally compatible with the repository's self-audit
doctrine:

- outside research is treated as continuous input, not as primary trust
- new external signals must be mapped to deterministic repo evidence
- unmapped signals stay visible instead of being silently ignored
- report generation is reproducible and checked into the repository
