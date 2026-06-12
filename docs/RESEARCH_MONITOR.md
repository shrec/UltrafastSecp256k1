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

Each class records:

- keyword match rules
- expected repo evidence
- current stance (`covered`, `candidate`, `gap`, `out_of_scope`)
- expected handling action

If a new failure or optimization class becomes relevant, update this matrix
first. The monitor validates that every referenced repo evidence path exists.

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
