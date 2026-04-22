# Sponsorship Case — UltrafastSecp256k1 + CAAS

> **Goal of this document.** Give potential sponsors (NLnet NGI Zero,
> Sovereign Tech Fund, OpenSats, OSTIF, individual companies and
> philanthropists) a clear, honest, *non-inflated* picture of what the
> project is, what is missing, and exactly what funding would unlock.
>
> No marketing language, no inflated valuation. Numbers are based on
> real hardware prices (April 2026) and conservative effort estimates.

---

## 1. What the project already delivers

UltrafastSecp256k1 is a permissively-licensed (BSD-2-Clause)
production-grade secp256k1 engine plus an open-source, fully
reproducible Continuous Audit-as-a-Service (CAAS) pipeline.

Today, with one solo maintainer and zero external funding, the
repository ships:

- A C++20 secp256k1 engine with a stable C ABI (`ufsecp_*`)
- CPU backend with constant-time pipelines for all secret-bearing ops
- CUDA backend with batch verify, megabatch search, and full ABI parity
- OpenCL backend (vendor-agnostic — AMD / Intel / NVIDIA / Apple)
- Metal backend skeleton (Apple Silicon)
- Cross-compile build matrix: x86_64, ARM64, RISC-V, WASM, ESP32, STM32,
  iOS, Android
- BIP-340 (Schnorr), BIP-352 (Silent Payments), BIP-324 (P2P encryption),
  ECDSA, ECDH, FROST building blocks
- Bindings: Swift, Python, JS-ready, Rust-ready
- A wired audit suite of 60+ modules + exploit PoCs run on every commit
- The CAAS pipeline itself (this is the unique part — see
  [`CAAS_PROTOCOL.md`](CAAS_PROTOCOL.md))

This is real, working, in-tree, and verifiable today by cloning the
repo and running `caas_runner.py`.

## 2. What is missing — the gap funding would close

The honest picture: the project has unfinished surfaces that need
*hardware access* and *focused engineering time*, not new ideas.

### 2.1 GPU backend completion (AMD / Intel / Apple)

| Backend | Status today | Gap |
|---|---|---|
| CUDA (NVIDIA) | Production-grade, full ABI parity | None |
| OpenCL (NVIDIA) | Tested on dev box | Performance not yet on par with CUDA |
| OpenCL (AMD ROCm) | Compiles, untested at scale | **No AMD GPU available for tuning** |
| OpenCL (Intel Arc / Xe) | Compiles, untested | **No Intel GPU available** |
| Metal (Apple Silicon) | Skeleton kernels exist | **No Apple Silicon dev machine** |

Without physical hardware access, these backends cannot reach the same
tuning level as CUDA. They will keep working, but they will not deliver
the throughput numbers users expect on those vendors' devices.

### 2.2 CAAS pipeline hardening

The current CAAS pipeline runs on:

- 1 dev workstation (CPU + 1 NVIDIA GPU)
- GitHub Actions hosted runners (no GPU)
- A handful of community-contributed CI lanes

Gaps:

- No GPU CI runners — GPU regressions are caught only locally
- No AMD / Intel / Apple Silicon CI lanes
- No long-running soak/fuzz lanes (current GitHub Actions caps prevent
  multi-hour fuzz campaigns)
- Reproducible-build attestation infra exists but lacks a second
  independent rebuild farm

### 2.3 Specific engineering targets

These are the items that funding would directly accelerate:

1. **AMD ROCm OpenCL tuning** — bring AMD performance to within 80% of
   CUDA throughput on equivalent-class GPUs.
2. **Intel Arc / Xe tuning** — same target on Intel.
3. **Metal kernel completion** — finish Apple Silicon backend, ship
   parity tests.
4. **Self-hosted GPU CI** — one always-on runner per vendor, wired into
   GitHub Actions.
5. **CAAS pipeline v2** — add multi-vendor GPU lanes, add a long-running
   fuzz/soak lane, add a second reproducible-build verifier.

## 3. What sponsorship would actually pay for

This section is intentionally specific and modest.

### 3.1 Hardware (one-time)

Real April-2026 retail prices, no markup, no fluff:

| Item | Use | Estimated cost |
|---|---|---|
| NVIDIA RTX 5090 (24–32 GB) | CUDA tuning, megabatch perf work | $2,000–$2,500 |
| AMD Radeon RX 7900 XTX or Radeon Pro W7900 | ROCm OpenCL tuning | $1,000–$3,500 |
| Intel Arc A770 (16 GB) or Battlemage successor | Intel OpenCL tuning | $400–$700 |
| Apple Mac mini M4 Pro (or Studio) | Metal backend completion | $1,400–$2,500 |
| 1 small server chassis + PSU + cooling for the GPUs | Self-hosted CI runner | $1,000–$1,500 |

**Hardware total: ~$6k–$11k one-time.** This is the price of one
mid-tier laptop. It unlocks three GPU vendors and a permanent CI lane
for each.

### 3.2 Recurring infrastructure (annual)

| Item | Cost / year |
|---|---|
| Electricity for the self-hosted GPU runner (~250W avg) | ~$300–$500 |
| Backup off-site cold storage for evidence bundles | ~$100 |
| Domain + Zenodo (already free) + status page | ~$50 |

**Recurring total: ~$500–$700/year.** Negligible.

### 3.3 Engineering time (the actual cost)

This is what real sponsorship money would go toward — not the hardware,
the *time to use it well*.

| Workstream | Estimated effort | Honest cost band |
|---|---|---|
| AMD ROCm tuning to 80% of CUDA | 2–3 months focused | $15k–$25k |
| Intel Arc tuning | 1–2 months | $8k–$15k |
| Metal kernel completion + parity | 2–3 months | $15k–$25k |
| Self-hosted CI integration + monitoring | 1 month | $5k–$10k |
| CAAS v2 (long-fuzz lane, multi-vendor GPU lanes, second reproducible-build verifier) | 2–3 months | $15k–$25k |

**Engineering total: ~$60k–$100k** to close every named gap above.
That is the equivalent of one junior-level engineer at market rate for
6 months — except the work is being done by the existing maintainer
who already has full context.

### 3.4 Realistic sponsorship tiers

| Tier | Yearly amount | What it buys |
|---|---|---|
| **Individual** | $5–$50/month (Patreon / GitHub Sponsors) | Keeps the lights on. CI credits, electricity, small tooling. |
| **Small company / startup** | $1k–$5k/year | One hardware item per year; one CAAS gap closed per year. |
| **Mid-size company / exchange** | $10k–$25k/year | One full backend (AMD, Intel, or Metal) brought to parity per year. Priority bug-channel access. |
| **Foundation / grant (NLnet, STF, OpenSats, OSTIF)** | $50k–$120k one-shot | Closes *all* of section 2.3 in one funded cycle. |

These are deliberate, modest numbers. The aim is *sustainable
multi-year support*, not a single windfall.

## 4. What sponsors get back

Concrete, measurable deliverables — not vague gratitude:

1. **Public acknowledgement** in `README.md`, `SPONSORS.md`, and the
   release notes of every release that benefits from their support.
2. **Per-deliverable evidence**: every funded gap closes with a public
   commit, a CAAS evidence bundle, and a benchmark report attributing
   the new capability to the sponsor.
3. **Priority bug-fix channel** for paying sponsors (private security
   reports, agreed SLA on `unsafe-input` class issues).
4. **Influence on the roadmap** at the level of "vote for which gap
   gets closed next" — not "rewrite the engine for our use case".
5. **The infrastructure stays open** — sponsorship does not buy
   exclusivity, it buys acceleration. Everything funded becomes
   permanent open-source value for the entire ecosystem.

## 5. Why this is a good use of sponsorship money

A short, honest argument:

- **The work is concrete, not speculative.** Every item in section 2.3
  has a clear definition of done.
- **The maintainer has a track record.** The repository already ships
  at the level described in section 1, with no funding to date.
- **The marginal cost is tiny.** A few thousand dollars of hardware and
  a few months of focused work close gaps that would cost a commercial
  team an order of magnitude more.
- **The benefit is durable.** Funded improvements land in a permanent
  open-source codebase under BSD-2-Clause. Anyone — including the
  sponsor's competitors — benefits, which is the only honest
  open-source posture.
- **CAAS is reusable beyond this project.** The methodology (see
  [`CAAS_PROTOCOL.md`](CAAS_PROTOCOL.md)) can be adopted by any
  open-source project. Funding CAAS v2 in this repo also funds
  proof-of-concept work that other projects can copy.

## 6. How to sponsor

- **GitHub Sponsors**: <https://github.com/sponsors/shrec> (when
  enabled)
- **Patreon / OpenCollective**: TBD — links will be added here as they
  go live
- **Direct grant** (NLnet, STF, OpenSats, OSTIF, individual
  philanthropist): contact the maintainer through the email in
  `CITATION.cff`
- **Hardware donations**: a list of specific items in section 3.1 that
  can be shipped directly. Contact the maintainer for a shipping
  address.

## 7. What this document is *not*

To stay honest:

- This is not a valuation of the company or the codebase.
- This is not a fundraising round or an equity solicitation.
- This is not a guarantee of any specific delivery date — the
  maintainer is one person and life happens.
- This is not a replacement for paid security audits where regulation
  or contractual obligation requires them.

It is a transparent ask: *here is what is missing, here is what it
would cost to close, here is what you get back*. Decide accordingly.

---

*Last updated: 2026-04-22.*
*Document owner: project maintainer (see `CITATION.cff`).*
