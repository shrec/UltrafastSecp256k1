# Bug Bounty Program

**UltrafastSecp256k1** -- Vulnerability Disclosure & Rewards

---

## 1. Scope

### 1.1 In-Scope Components

| Component | Priority | Description |
|-----------|----------|-------------|
| Core arithmetic (field/scalar/point) | Critical | Incorrect computation -> key recovery, forgery |
| ECDSA signing/verification | Critical | RFC 6979 nonce, signature correctness |
| Schnorr signing/verification | Critical | BIP-340 compliance |
| Constant-time layer (`ct::`) | Critical | Timing side-channel leaks |
| MuSig2 protocol | High | Key aggregation, rogue-key, nonce handling |
| FROST protocol | High | DKG, threshold signing, malicious participants |
| BIP-32 HD derivation | High | Child key derivation correctness |
| Address generation (27-coin) | High | Wrong addresses -> fund loss |
| C ABI (`ufsecp`) | Medium | NULL crashes, buffer overflows, UB |
| SHA-256 / RIPEMD-160 | Medium | Hash correctness |
| Serialization (DER, pubkey) | Medium | Parse confusion, malformed input crashes |

### 1.2 Out-of-Scope

- GPU backends (CUDA/OpenCL/Metal/ROCm) -- separate program planned
- Language bindings (Python/Rust/Go/C#) -- report upstream to binding repos
- Documentation errors (report as regular issues)
- Denial-of-service via large input (not a security library concern at this layer)
- Issues in dependencies (report to upstream maintainers)
- Social engineering, phishing
- Bugs in test code only (no production impact)

---

## 2. Reward Tiers

| Severity | Description | Reward Range |
|----------|-------------|-------------|
| **Critical** | Private key recovery, signature forgery, nonce leak, CT bypass enabling key extraction | $2,000 - $10,000 |
| **High** | Incorrect arithmetic producing wrong but non-exploitable results, FROST/MuSig2 protocol break, BIP-32 derivation error | $500 - $2,000 |
| **Medium** | Crash from crafted input, memory safety issue (OOB read/write), UB in production code path | $100 - $500 |
| **Low** | Non-security correctness bug (e.g., address checksum), minor API contract violation | $50 - $100 |
| **Informational** | Documentation gaps, hardening suggestions, code quality improvements | Public acknowledgment |

### 2.1 Bonus Multipliers

| Condition | Multiplier |
|-----------|-----------|
| Includes working exploit / proof-of-concept | 2x |
| Includes regression test case | 1.5x |
| Affects CT layer with demonstrated timing measurement | 2x |
| Found via formal methods / proof | 1.5x |

### 2.2 Reward Conditions

- First reporter of a unique vulnerability receives the reward
- Duplicate reports: first valid submission wins
- Partial findings that lead to a full vulnerability are eligible for partial reward
- Rewards are at maintainer discretion within the published ranges
- Payment via cryptocurrency (BTC, ETH) or other agreed method

---

## 3. Reporting Process

### 3.1 How to Report

**Do NOT open a public issue.**

1. **GitHub Security Advisories** (preferred):  
   [Create advisory](https://github.com/shrec/UltrafastSecp256k1/security/advisories/new)

2. **Email**: payysoon@gmail.com  
   Encrypt with PGP if available (key published in SECURITY.md)

### 3.2 Report Requirements

A valid report must include:

- **Summary**: One-paragraph description of the vulnerability
- **Affected component**: File path(s) and function name(s)
- **Severity assessment**: Your classification with justification
- **Reproduction steps**: Minimal code or commands to trigger
- **Impact analysis**: What an attacker could achieve
- **Environment**: OS, compiler, version, build flags used

### 3.3 Response Timeline

| Event | SLA |
|-------|-----|
| Acknowledgment | <= 72 hours |
| Severity triage | <= 7 days |
| Fix timeline communicated | <= 14 days |
| Critical fix released | <= 30 days |
| High fix released | <= 60 days |
| Medium/Low fix released | Next minor release |
| Public disclosure | 90 days after fix, or coordinated with reporter |

---

## 4. Disclosure Policy

### 4.1 Coordinated Disclosure

- Reporter agrees to keep findings confidential until public disclosure date
- Maintainer agrees to:
  - Credit the reporter (unless anonymity requested)
  - Publish fix before or simultaneously with disclosure
  - Include reporter in advisory drafting (optional)

### 4.2 Safe Harbor

- Good-faith security research is welcome
- We will not pursue legal action against researchers who:
  - Follow this disclosure process
  - Do not access, modify, or delete user data
  - Do not disrupt services
  - Provide sufficient time for fix before disclosure

### 4.3 CVE Assignment

- Critical and High findings will receive CVE identifiers
- CVE will be requested through GitHub Security Advisories
- Advisory will be published on GitHub and linked from SECURITY.md

---

## 5. Hall of Fame

Security researchers who responsibly disclose vulnerabilities will be credited in:

- `SECURITY.md` Hall of Fame section
- GitHub Security Advisory acknowledgments
- Release notes for the fix version

To opt out of public credit, state so in your report.

---

## 6. Exclusions

The following are NOT eligible for rewards:

- Issues already known and documented (check existing advisories)
- Issues in pre-release/experimental code explicitly marked as such
- Self-XSS, clickjacking, or UI-level issues (not applicable)
- Rate limiting or brute force (not applicable at library level)
- Findings that require physical access to the machine
- Compiler bugs (report to compiler vendor)
- Issues that only manifest with undefined build flags or non-supported configurations

---

## 7. Program Updates

This program may be updated at any time. Changes will be:
- Committed to this file in the repository
- Announced in release notes if reward tiers change
- Applied prospectively (existing reports under previous terms)

---

*Program effective date: 2026-02-24*  
*Last updated: 2026-02-24*
