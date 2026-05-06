# Funding Targets & Sponsor Outreach

Concrete grant programs, sponsorship channels, and outreach targets for UltrafastSecp256k1.

This document is a working playbook. It lists where the project can plausibly secure funding, what each programme looks for, and what the project already has on the table.

---

## 1. Active funding channels

| Channel | URL | Status |
|--------|-----|--------|
| GitHub Sponsors (`shrec`) | https://github.com/sponsors/shrec | Active |
| PayPal | https://paypal.me/IChkheidze | Active |
| Stacker News | https://stacker.news/shrec | Active |

A single one-line "Sponsor this project" CTA is in `README.md`. The project is also listed in `.github/FUNDING.yml`, which renders a sidebar Sponsor button on every GitHub page.

---

## 2. Grant programmes worth applying to

These are programmes that fit the project's actual scope (open-source secp256k1 / cryptography / Bitcoin / Ethereum infrastructure / EU public goods). Each row lists what the programme rewards and what UltrafastSecp256k1 can credibly claim against it.

### Bitcoin ecosystem

| Programme | Fit | What we already meet |
|-----------|-----|----------------------|
| **HRF Bitcoin Development Fund** (`hrf.org/programs/financial-freedom`) | Open-source Bitcoin tools, privacy, scalability | BIP-352 Silent Payments scanning at 11.00 M tweaks/s; production use in [Sparrow Wallet Frigate](https://github.com/sparrowwallet/frigate) |
| **OpenSats** (`opensats.org`) | Free and open-source Bitcoin contributors and projects | 253 exploit PoCs, MIT license, public CAAS evidence bundle, adopter case study |
| **Brink** (`brink.dev`) | Bitcoin protocol engineering grants | secp256k1 engine fits Bitcoin Core / wallet stack; Schnorr / Taproot / MuSig2 coverage |
| **Spiral** (`spiral.xyz`) | Bitcoin open-source contributors | GPU-accelerated Schnorr / Taproot useful for Lightning, LDK, BDK consumers |
| **Strike Catalyst** | Bitcoin builders | Layered library + adopter validation by an independent wallet |
| **MIT DCI** | Bitcoin / digital currency research | Reproducible-build attestation across 3 CI providers, formal invariants, Cryptol properties |

### Ethereum ecosystem

| Programme | Fit | What we already meet |
|-----------|-----|----------------------|
| **Ethereum Foundation Ecosystem Support Programme (ESP)** | Ethereum protocol, infra, security tools | secp256k1 used by EVM / Geth / Reth / clients; ECDSA recover at scale; cross-implementation differential matrix |
| **Protocol Guild** | Long-term Ethereum core maintainers | Continuous-audit infra is reusable for any signature-heavy stack |
| **EF Academic Grants Round** | Crypto research, formal verification, MEV, ZK | Formal invariant specs + Cryptol properties + GPU constant-time analysis |
| **Optimism RetroPGF** / **Arbitrum Foundation grants** / **Base / Coinbase Developer Grants** | L2 ecosystem infra | secp256k1 is a hot path for every rollup batcher / sequencer |

### Cross-cutting / EU public goods

| Programme | Fit | What we already meet |
|-----------|-----|----------------------|
| **NLnet NGI Zero PET / Commons / Entrust** (`nlnet.nl`) | Privacy, security, trust, internet commons | Constant-time engine, supply-chain hardening, Silent Payments privacy use-case, full reproducible builds |
| **Sovereign Tech Fund** (`sovereigntechfund.de`) | Critical OSS infrastructure (German federal funding) | Foundational crypto primitive; fits the "if it breaks, the chain breaks" criterion |
| **Open Source Collective / Open Collective Foundation** | Fiscal hosting + grant collection | Useful as receiving entity for non-individual grants |
| **OSI / OSTIF / Linux Foundation OSPO grants** | Critical infrastructure, audit funding | Audit-grade security stance is the literal value proposition |
| **GitHub Accelerator** | High-impact OSS maintainers | One-person-led, third-party adopter, demonstrable performance gains |
| **a16z crypto Open Source Grants** | Crypto open-source primitives | Engine-level work usable by every Bitcoin and Ethereum stack |

### Bug bounty / security-paid surfaces (indirect funding signal)

- **Immunefi** — even just being listed signals seriousness to integrators.
- **HackerOne** — same.
- The project's `SECURITY.md` and continuous exploit-PoC pipeline are a direct selling point for bounty hosting.

---

## 3. The 30-second pitch

> UltrafastSecp256k1 is a GPU-accelerated, MIT-licensed secp256k1 engine with a self-running audit pipeline (253 exploit PoCs, formal invariants, multi-CI reproducible-build attestations). It is in production use by Sparrow Wallet Frigate. It runs on CUDA, Metal, OpenCL, ARM64, RISC-V, WASM, ESP32, and STM32 with the same C ABI.
>
> The project is one-maintainer-led and currently unfunded apart from individual sponsorships. Foundation funding would underwrite (a) the GPU hardware needed for ROCm/HIP and Apple Silicon attestation evidence, (b) sustained INTEROP differential testing against BoringSSL, k256 (Rust), and btcd (Go), and (c) a security retainer for an external code review of the constant-time engine.

---

## 4. The 5-minute pitch (talking points)

1. **What the project is**: a secp256k1 engine that beats every other open-source implementation on GPU and matches them on CPU, with a continuous-audit pipeline that publishes the evidence rather than asking for trust.
2. **Why it exists**: traditional crypto libraries publish a one-time audit PDF and ask for trust. This project publishes the *audit infrastructure itself* — every claim maps to a runnable test (see `docs/AUDIT_TRACEABILITY.md`).
3. **Production proof**: `docs/ADOPTION.md` documents the Sparrow Wallet Frigate integration with independent benchmark numbers and a public release pointer.
4. **Performance proof**: 11.00 M BIP-352 scans/s, 4.88 M ECDSA signs/s, 4.05 M ECDSA verifies/s, 5.38 M Schnorr verifies/s on a single RTX 5060 Ti — all reproducible from `apps/cpu_megabatch/` and the GPU benchmark harness.
5. **Audit proof**: 253 exploit PoCs (catalog in `docs/EXPLOIT_TEST_CATALOG.md`), 60 non-exploit audit modules, 12/12 CAAS hardening items closed (`docs/AUDIT_DASHBOARD.md`), reproducible-build attestation across GitHub Actions / GitLab CI / Woodpecker (Codeberg).
6. **Security stance**: 3 independent constant-time pipelines, Cryptol formal properties, supply-chain gating, evidence-chain HMAC keys with documented rotation policy.
7. **Why funding is the bottleneck right now**:
   - Lack of access to AMD Instinct hardware blocks ROCm/HIP attestation evidence (currently scaffold-only, see RR-003).
   - Apple Silicon attestation evidence requires hosted Mac runners.
   - INTEROP §3 closure (BoringSSL / k256 / btcd / MuSig2-wire / FROST-wire) is a sustained engineering cost, not a one-shot.
   - External constant-time review by an independent firm has not been bought.

---

## 5. Outreach materials checklist

- [x] `CITATION.cff` (academic discoverability)
- [x] `.zenodo.json` (Zenodo archival; DOI on next release)
- [x] `docs/ADOPTION.md` (production proof)
- [x] `docs/AUDIT_DASHBOARD.md` (live audit evidence)
- [x] `docs/INTEROP_MATRIX.md` (cross-implementation discipline)
- [x] `docs/THREAT_MODEL.md`
- [x] `docs/CAAS_PROTOCOL.md`
- [x] `docs/SECURITY_AUTONOMY_PLAN.md`
- [x] `WHY_ULTRAFASTSECP256K1.md` (long-form rationale)
- [ ] One-page PDF pitch (for grant submissions; can be generated from this file)
- [ ] Demo video (≤90 s screen-recording of CAAS pipeline)
- [ ] Public bug-bounty listing
- [ ] Two-line "ask" tailored per programme (BTC, ETH, EU, security firms)

---

## 6. What this document is *not*

It is not an announcement to be posted publicly to GitHub Discussions, Reddit, or Twitter without owner authorisation. Public outreach for grants is owner-driven and per-programme.

It is also not a public commitment to apply to every programme listed. The list exists so that when the owner is ready to apply to one, the talking points and evidence pointers are already assembled.
