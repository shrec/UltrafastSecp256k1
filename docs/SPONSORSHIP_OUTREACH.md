# Sponsorship Outreach Plan — UltrafastSecp256k1

> Companion to [`SPONSORSHIP_CASE.md`](SPONSORSHIP_CASE.md).
> This document is **internal/operational**. It contains the verified
> contact emails for potential sponsors and a ready-to-send email
> template. Update it as outreach progresses.

> **Strict policy reminder.** Contact each target *once* using the
> public email below. Wait at least 4 weeks before any follow-up.
> No mass mail, no BCC blasts. Each target is a serious foundation
> or company; treat them like one.

---

## 1. Verified contact emails (April 2026)

All emails below were verified by visiting the official site. None
were guessed or scraped from third-party lists.

### Foundations & grant programs (PRIMARY targets)

| # | Organisation | Verified email | Apply form / page | Notes |
|---|---|---|---|---|
| 1 | **NLnet Foundation** (NL — NGI Zero) | `--- (use form)` | <https://nlnet.nl/propose/> | Use the proposal form, not email. Round closes **June 1, 2026**. Best fit for CAAS pipeline + open infra. |
| 2 | **Sovereign Tech Agency** (DE — STF, STR, STC) | `info@sovereign.tech` | <https://www.sovereign.tech/programs/fund> | Best fit for "critical open digital infrastructure" framing. Mention CAAS as resilience tooling. |
| 3 | **OpenSats** (US — Bitcoin/Nostr 501c3) | `support@opensats.org`, `partnerships@opensats.org` | <https://opensats.org/apply> | Best fit for Bitcoin engine angle. Apply form preferred; email partnerships for follow-up. |
| 4 | **OSTIF** (US — Open Source Technology Improvement Fund) | `ContactUs@ostif.org` | <https://ostif.org/get-an-audit/> | Specialises in security audits of OSS — they could *fund* an external audit of ufsecp, not just CAAS. |
| 5 | **Brink** (UK/US — Bitcoin protocol R&D) | `info@brink.dev` | <https://brink.dev/programs#grants> | Best fit for "improves Bitcoin protocol stack" angle. Their grants go to Bitcoin Core/LDK/BDK contributors. |
| 6 | **Spiral** (US — Block-funded Bitcoin grants) | `--- (use form)` | <https://spiral.xyz/#grants> ("APPLY NOW") | Funds FOSS Bitcoin work. Apply form only — no public email. |
| 7 | **HRF Bitcoin Development Fund** (US — Human Rights Foundation) | `donations@hrf.org` (general) | <http://hrf.org/bdfapply> | Year-round applications. Best fit for "uncensorable money infrastructure" angle. |
| 8 | **Sovereign Tech Resilience** (DE — bug-prevention program) | `info@sovereign.tech` | <https://www.sovereign.tech/programs/bug-resilience> | Distinct STA program — explicitly for "preventing and eliminating vulnerabilities in critical digital infra". CAAS fits exactly. |

### Companies (SECONDARY targets — single, polite, tailored mail)

| # | Company | Verified email | Notes |
|---|---|---|---|
| 9 | **Blockstream** | `inquiries@blockstream.com` (general), `Enterprise_team@blockstream.com` (business), `security@blockstream.com` (sec) | Liquid + Greenlight + Jade — natural users of an alternative secp256k1. |
| 10 | **Block, Inc.** (Spiral parent) | (no public sales email — go via Spiral) | If Spiral funds, Block visibility follows. |
| 11 | **Chaincode Labs** | (no public email; LinkedIn/Twitter for outreach) | Bitcoin Core review centre. Possible mentor-style support, not direct sponsorship. |
| 12 | **Casa, Unchained Capital, River, Swan** | (use website contact forms) | Custody firms — natural users of batch verify and CAAS evidence bundles. Approach only after foundations respond. |

### Bitcoin Policy Institute (US — policy think tank, not a grants org)

| # | Organisation | Email |
|---|---|---|
| 13 | Bitcoin Policy Institute | (use contact page <https://www.btcpolicy.org/contact>) — relevant only as awareness-raising, not direct funding. |

---

## 2. Recommended order of outreach

Send in this order, **one per day at most**, so each gets your full
attention and so you can iterate the wording based on early replies.

1. **NLnet** — highest probability, well-defined process, June 1 deadline.
2. **Sovereign Tech Fund** + **Sovereign Tech Resilience** — same email,
   one combined pitch (ask which programme is the right fit).
3. **OpenSats** — apply form first; email `partnerships@opensats.org`
   only after the form is in.
4. **Brink** — short personal email to `info@brink.dev`.
5. **OSTIF** — `ContactUs@ostif.org` with an audit-funding angle (they
   can connect ufsecp to a paid auditor rather than fund development
   directly).
6. **HRF Bitcoin Development Fund** — apply form.
7. **Spiral** — apply form.
8. **Blockstream Enterprise** — only after at least one foundation has
   replied; pitch as "we are open and would value your evaluation +
   sponsorship for AMD/Apple parity, both of which Liquid users care
   about".

---

## 3. Master email template (English)

Replace `<>` placeholders. Keep it short — under 250 words.

```
Subject: UltrafastSecp256k1 — open-source secp256k1 + CAAS pipeline,
         seeking sponsorship to close GPU vendor parity

Hello <Name or Team>,

I maintain UltrafastSecp256k1 [1], a BSD-2-Clause secp256k1 engine
with full CUDA support, OpenCL, Metal scaffolding, cross-platform
build matrix (RISC-V, ARM64, WASM, ESP32, STM32, iOS, Android), and
a fully open-source Continuous Audit-as-a-Service (CAAS) pipeline
that runs on every commit [2].

I'm reaching out because <organisation> funds exactly the kind of
work I am trying to finish: hardening open critical infrastructure
and closing platform-parity gaps without VC dependency.

There are three concrete gaps I cannot close alone — they need
hardware access and focused engineering time, not new ideas:

  • AMD ROCm OpenCL tuning to within 80 % of CUDA throughput
  • Intel Arc / Xe OpenCL tuning to the same target
  • Metal kernel completion + parity tests on Apple Silicon
  • Self-hosted GPU CI runners and a long-running fuzz/soak lane
    for the CAAS pipeline

The full sponsorship case — honest hardware shopping list
(~$6k–$11k one-time), engineering effort estimates ($60k–$100k
for everything above), tiered sponsorship structure, and the
deliverables sponsors get back — is at:

  https://github.com/shrec/UltrafastSecp256k1/blob/main/docs/SPONSORSHIP_CASE.md

The CAAS protocol document, which describes the methodology in a
deliberately project-agnostic way, is at:

  https://github.com/shrec/UltrafastSecp256k1/blob/main/docs/CAAS_PROTOCOL.md

I'd be grateful for any of:

  1. A pointer to the right application path inside <organisation>,
  2. A short call to discuss whether this fits one of your programmes,
  3. A direct hardware donation (the shopping list is short).

Happy to provide any further evidence, benchmarks, or references.

Thank you for your time.

— <Your Name>
   Maintainer, UltrafastSecp256k1
   <your-email>
   GPG: <fingerprint or link, optional>

[1] https://github.com/shrec/UltrafastSecp256k1
[2] https://github.com/shrec/UltrafastSecp256k1/blob/main/docs/CAAS_PROTOCOL.md
```

### Per-target subject-line variants

| Target | Suggested subject |
|---|---|
| NLnet | (Use the form — write the same body in the "Project description" field.) |
| Sovereign Tech | `UltrafastSecp256k1 — critical open Bitcoin infra seeking STF/STR sponsorship` |
| OpenSats | `UltrafastSecp256k1 + CAAS — Bitcoin secp256k1 engine seeking OpenSats partnership` |
| Brink | `UltrafastSecp256k1 — open secp256k1 engine + CAAS pipeline, seeking Brink sponsorship` |
| OSTIF | `UltrafastSecp256k1 — open secp256k1 engine, seeking OSTIF audit sponsorship` |
| HRF BDF | (Use the apply form — same body.) |
| Spiral | (Use the apply form — same body.) |
| Blockstream | `UltrafastSecp256k1 — open secp256k1 engine for Liquid/Greenlight workloads, seeking sponsorship` |

### Per-target one-line hook (paste before the closing)

| Target | One-line hook |
|---|---|
| NLnet | "CAAS is a portable, project-agnostic methodology — the secp256k1 instantiation is the reference implementation, and the protocol document is written so any other open-source project can adopt the same pattern." |
| Sovereign Tech | "Both the engine and the audit pipeline are open under BSD-2-Clause; sponsorship buys acceleration, not exclusivity." |
| OpenSats | "Drop-in C-ABI-compatible with libsecp256k1; complementary scope (batch / GPU / embedded) priorities for Bitcoin tooling that libsecp does not cover." |
| Brink | "Improves the broader Bitcoin developer toolchain by giving wallet, indexer, and validator builders a permissively-licensed alternative engine with batch-verify and GPU paths." |
| OSTIF | "I am not asking OSTIF to fund engine development — I am asking whether OSTIF can sponsor an external audit engagement of ufsecp by one of your audit partners. The CAAS evidence bundle is ready to hand to an auditor." |
| HRF BDF | "Faster on-chain verification and key generation help wallets and recovery services that human rights defenders rely on, especially on low-end hardware (RISC-V / ESP32 / STM32 / Android)." |
| Spiral | "Permissively-licensed alternative secp256k1 engine usable by BDK, LDK, and any downstream Bitcoin tool — complementary to libsecp, not competing." |
| Blockstream | "Liquid and Greenlight workloads benefit directly from batch verify; AMD and Apple Silicon parity (currently unfunded) is the remaining blocker for some of your enterprise customers." |

---

## 4. Tracking checklist

Update the date column as you send.

| Date sent | Target | Channel | Status |
|---|---|---|---|
|  | NLnet | form | ☐ |
|  | Sovereign Tech | `info@sovereign.tech` | ☐ |
|  | OpenSats | apply form, then `partnerships@opensats.org` | ☐ |
|  | Brink | `info@brink.dev` | ☐ |
|  | OSTIF | `ContactUs@ostif.org` | ☐ |
|  | HRF BDF | apply form | ☐ |
|  | Spiral | apply form | ☐ |
|  | Blockstream | `Enterprise_team@blockstream.com` | ☐ |

Replies, follow-ups, and meeting notes — keep in a private file
outside the repo. Do not commit reply contents to the public repo.

---

## 5. What NOT to do (hard rules)

- Do **not** mass-BCC. One target, one mail, one tailored hook.
- Do **not** chase replies before 4 weeks have passed.
- Do **not** post about pending applications publicly (no Twitter,
  no GitHub Discussions announcements, no Mastodon thread). Wait
  for outcomes.
- Do **not** quote prices or sponsorship tiers higher than what is
  in `SPONSORSHIP_CASE.md`. Inflated numbers kill credibility.
- Do **not** make promises (delivery dates, exclusivity, custom
  features). Sponsorship buys acceleration of the existing
  roadmap, not bespoke work.
- Do **not** mix the engine ask with the policy/think-tank ask.
  Bitcoin Policy Institute is awareness, not money.

---

*Last updated: 2026-04-22.*
