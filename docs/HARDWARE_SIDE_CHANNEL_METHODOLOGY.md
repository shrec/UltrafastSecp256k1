# HARDWARE_SIDE_CHANNEL_METHODOLOGY.md — UltrafastSecp256k1

> Version: 1.0 — 2026-04-21
> Closes CAAS gap **G-3**.
>
> This document states what hardware side-channel claims the library
> makes and does not make, and the methodology that backs each claim.

## 1. Position

UltrafastSecp256k1 makes **software-side-channel** claims only.

| Channel class | Claim | Methodology |
|---------------|-------|-------------|
| Timing (instruction count, branch) | Constant-time on tested platforms | dudect, Valgrind CT, ct-verif (3-tool agreement, P11) |
| Cache (L1/L2 access patterns) | Constant address pattern in CT layer | dudect (statistical), source review, exploit PoCs `test_exploit_timing_*` |
| Branch predictor (Spectre v2 BTI) | Best-effort hardening; not certified | `compute-sanitizer.yml`; documented in RR-001 |
| Frequency scaling (Hertzbleed) | Out of scope | Documented in RR-002 |
| **Power analysis (SPA/DPA)** | **No claim** | See §3 |
| **EM emanation** | **No claim** | See §3 |
| **Fault injection (clock/voltage glitching, laser)** | **No claim** | See §3 |
| **Acoustic** | **No claim** | See §3 |

## 2. What is verified

The constant-time claim is verified by three independent tools that
must agree:

1. **dudect** — statistical timing test on real hardware
   (`audit/test_dudect_*`).
2. **Valgrind CT** — bit-precise data-flow tracking of secret bytes
   through every operation (`audit/test_valgrind_ct_*`).
3. **ct-verif** — formal symbolic execution proving no secret-dependent
   control flow or memory address (`tools/ct_verif/`).

A CT property is only marked verified when **all three** tools agree.
This is enforced by the P11 sub-gate in `scripts/audit_gate.py`.

## 3. What is NOT verified — and why

Hardware side channels (power, EM, fault) require:

- A controlled physical environment (Faraday cage, regulated supply).
- Specialised equipment (oscilloscope, EM probes, glitcher).
- A specific device under test (the actual chip in the actual product).

These are properties of the *operating environment*, not of source
code. A library cannot claim power-analysis resistance because the
relevant attack happens at the silicon level, not the C-source level.

Projects that do make such claims (e.g. smart-card crypto libraries
under Common Criteria EAL4+) ship with a specific physical platform,
a tested package, and lab evaluation reports. UltrafastSecp256k1 ships
as portable software and intentionally does not.

## 4. What a downstream user should do

If you deploy UltrafastSecp256k1 in a setting where hardware side
channels matter (HSM, smart card, embedded device with physical
attacker access):

1. Run dudect / Valgrind CT / ct-verif on **your** target device, not
   ours. CI agreement on x86-64 / ARM64 / RISC-V is a baseline, not a
   transferable proof.
2. Combine with hardware-level countermeasures appropriate to your
   threat model (shielding, tamper response, masking at gate level).
3. Treat secp256k1 keys as ephemeral or sealed in tamper-resistant
   storage; do not assume userland CT prevents physical extraction.

## 5. Test surface for software side channels

| Test | Coverage |
|------|----------|
| `audit/test_ct_*` (4 modules in `ct_analysis` section) | dudect / Valgrind / ct-verif / CT IR diff |
| `audit/test_exploit_timing_*` series | timing-channel exploit PoCs |
| `gpu_ct_leakage_report.json` | GPU CT survey (public-data-only confirms no secret crosses GPU boundary) |
| `audit/test_constant_time_field_ops.cpp` | per-primitive CT proof |

## 6. Change discipline

If a future release adds hardware-side-channel claims (e.g. via
masking, threshold implementations, or co-published lab evaluation),
this document and `docs/COMPLIANCE_STANCE.md` must be updated in the
same commit, with a sub-gate added to `scripts/audit_gate.py`.

Until then, the position is unchanged: software side channels yes,
hardware side channels no.
