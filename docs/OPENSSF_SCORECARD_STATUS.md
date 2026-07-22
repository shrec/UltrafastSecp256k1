# OpenSSF Scorecard status

## Authoritative result

- API source: <https://api.securityscorecards.dev/projects/github.com/shrec/UltrafastSecp256k1>
- Retrieved at (UTC): `2026-07-22T23:28:05Z`
- Scan timestamp (API `date`): `2026-07-20T02:17:16Z`
- Repository: `github.com/shrec/UltrafastSecp256k1`
- Scored repository commit: `bd39cfbc0802ecd6eb21b7ee3b582a89617d20f0`
- Scorecard version/commit: `v5.3.0` / `c22063e786c11f9dd714d777a687ff7c4599b600`
- Authoritative total score: **8.4 / 10**
- Roadmap target (9.5): **NOT PROVEN**. The authoritative score is below 9.5; this status must not be upgraded until a later result from the same API reports at least 9.5 for the intended repository commit.

## Non-perfect checks

Every API check below 10 is listed. Scores of `-1` mean the check could not be evaluated, not that it passed.

| Check | Score | API finding | Owner | Remediation / exact blocker |
|---|---:|---|---|---|
| CI-Tests | -1 | No pull request found. | Repository maintainers | Merge a representative pull request only through the existing required CI gates, then wait for a subsequent Scorecard scan. Blocker: the scanner needs eligible pull-request history; documentation alone cannot prove this check. |
| Code-Review | 0 | 0 of 30 changesets were approved. | Repository maintainers and GitHub organization administrators | Require reviewed pull requests for changes, record approving human reviews, and prevent direct/bypass merges. Re-scan after sufficient reviewed changesets exist. Blocker: historical review evidence and repository rules are controlled on GitHub, outside this documentation-only task. |
| CII-Best-Practices | 7 | OpenSSF Best Practices Silver badge detected. | Project lead / OpenSSF Best Practices badge owner | Complete the remaining Gold-level badge criteria and publish the resulting badge state. Blocker: Gold requires truthful project/process evidence and approval in the external OpenSSF Best Practices service. |
| Signed-Releases | 8 | All 5 recent releases had signed artifacts, but releases `v4.2.1`, `v4.3.0`, `v4.4.0`, and `v4.5.0` lacked provenance. | Release engineering maintainers | Preserve artifact signing and attach verifiable build provenance for every artifact in each new release; verify the provenance is discoverable by Scorecard. Do not remove signing or provenance gates. |
| Branch-Protection | 5 | Main protects PRs/status checks, but administrator enforcement is off, one approval is required, and last-push approval is off. | GitHub organization/repository administrators | Enforce rules for administrators, require two approving reviews, and require approval of the last push while retaining all current PR, status-check, code-owner, stale-review, force-push, and deletion protections. Blocker: these settings require GitHub repository administration authority. |
| Contributors | 0 | No contributing companies or organizations detected. | Project maintainers / community stewardship owner | Invite and support sustained contributions from independent organizations and ensure affiliations are accurately public. Blocker: contributor diversity is an external community outcome and must not be fabricated or gamed. |

## Control-preservation statement

This report changes no security, branch-protection, release, or CI control. Remediation must add or strengthen evidence and controls; no existing gate may be disabled to improve the score.

## Revalidation rule

After owners complete remediation, query the API source again and record its retrieval timestamp, scan timestamp, scored commit, per-check results, and total. Mark the 9.5 target proven only when that authoritative total is at least 9.5.
