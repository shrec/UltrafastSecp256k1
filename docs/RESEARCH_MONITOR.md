# Research Monitor

Scheduled external-signal intake for secp256k1-related papers, advisories, and
security-relevant updates.

The goal is not to outsource assurance.
The goal is to continuously ingest fresh external signals, compare them against
the repository's existing audit surfaces, and surface anything that looks new,
unmapped, or worth upgrading into deterministic evidence.

## What it does

The monitor runs `ci/research_monitor.py` and:

1. fetches recent secp256k1-related items from external sources
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
5. optionally sends an SMTP email when actionable items are found

"Actionable" currently means:

- matched to a `gap`
- matched to a `candidate`
- matched to no known class at all (`unmapped`)

This keeps the workflow focused on items that may require new tests, new proof
artifacts, or explicit review.

## Workflow

See `.github/workflows/research-monitor.yml`.

Default behavior:

- scheduled daily
- manual dispatch supported
- uploads the generated report as an artifact
- writes a summary into the GitHub Actions job summary
- sends email only when actionable findings exist and SMTP secrets are configured

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