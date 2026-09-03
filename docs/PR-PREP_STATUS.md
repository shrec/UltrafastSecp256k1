# PR-Prep Status — Bitcoin Core Backend PR

**Date:** 2026-05-22
**Branch:** `dev` (ready)
**Scope:** Bitcoin Core integration only (downstream forks handled separately)

This is the close-out report for the six final pre-PR items identified in the
2026-05-22 review. Items 1–5 are technical closes that landed in this branch;
item 6 is an owner-side scope decision documented for the PR body author.

---

## Item 1 — dev/main sync drift  ✅ CLOSED

Verified `dev` is 134 commits ahead of `main`, 0 commits behind. The owner-controlled
`dev → main` release merge is the only outstanding sync action and is explicitly
gated by CLAUDE.md "Release Authorization" — no automation will perform it.

```bash
git rev-list --left-right --count origin/main...origin/dev
# 0  134
```

No drift action required pre-PR. The PR will target a downstream Bitcoin Core fork,
not `main` of this repo.

---

## Item 2 — CMake profile smoke (6 profiles + audit)  ✅ CLOSED

`ci/check_profile_smoke.sh` now exercises **6 deployment profiles** at the
configure stage plus a full `bitcoin-core` build/link smoke:

| Profile | Configure | Link | Notes |
|---------|:---------:|:----:|-------|
| `bitcoin-core` | ✅ | ✅ (2,310 KB lib) | PR target |
| `litecoin` | ✅ | — | Downstream fork |
| `dogecoin` | ✅ | — | Downstream fork (ECDSA + Schnorr) |
| `bch-wallet` | ✅ | — | RPA + HD wallet |
| `wallet` | ✅ | — | Generic HD + ECIES |
| `audit` | ✅ | — | Audit harness, FROST/ZK enabled |

Runtime: ~5s per configure, ~60s for the bitcoin-core link smoke. Exit code 1 if
**any** profile fails configure or link. Wired into `ci/run_fast_gates.sh` so the
pre-push hook catches profile-flag drift before a CI cycle is spent.

The 5-second configure stage catches the class of bug where a base preset disables
an optional module (e.g. BIP-352) but leaves a dependent module (e.g. LTC-SP)
implicitly ON — these used to surface only after multi-minute CI builds.

---

## Item 3 — no-LTO benchmark wording refresh  ✅ CLOSED

The previous "i-cache pressure" narrative gave footprint as the sole root cause
of the no-LTO ~1.1% deficit. After PERF-002 and the DER parser fast-path,
~0.5–1.0% remains — small enough that the simpler narrative was overclaiming
the i-cache story. Three docs were rewritten to reflect the actual sequence:

1. **README.md §ConnectBlock bullet** — now lists both root-cause fixes (PERF-002
   commit `40697447` and the DER parser fast-path) and presents the residual
   ~0.5–1.0% as consistent with the **measured** 1.83× hot-path size delta.
2. **docs/BITCOIN_CORE_BACKEND_EVIDENCE.md §2.1** — full rewrite under a new
   subsection title "Without-LTO gap — root causes and residual size delta"
   that names both fixes explicitly with commit references and explains why
   the residual gap exists.
3. **docs/BENCHMARKS.md** — both the "Build mode matters" note and the
   "Earlier Results (RelWithDebInfo, archived)" footnote updated to match.

The size-delta numbers used in all three are now **measured**, not approximated
(see Item 4).

---

## Item 4 — shim-only binary size artifact  ✅ CLOSED

`docs/SHIM_FOOTPRINT_COMPARISON.md` (new) provides an apples-to-apples `.text`
section measurement across three builds, with a documented reproduction recipe.

Headline measurements (no-LTO, GCC 14.2.0, Intel i5-14400F):

| Build | `.text` | Multiplier |
|-------|--------:|-----------:|
| libsecp256k1 (Bitcoin Core bundled) | 1,261 KB | 1.00× |
| Ultra `bitcoin-core` profile (shim inlined) | 2,310 KB | **1.83×** |
| Ultra full-feature profile | 2,669 KB | 2.12× |

The shim wrapper layer itself is **188 KB** of object code across 12 translation
units. In the `bitcoin-core` profile (`SECP256K1_BUILD_SHIM=ON`), the shim is
compiled directly into `libfastsecp256k1.a` for LTO cross-TU inlining.

This **replaces** the previous "~1.3 MB Ultra vs ~400 KB libsecp" approximation
in the README and evidence doc — that figure pre-dated the bitcoin-core
deployment profile and used a stripped-binary comparison that excluded the shim
layer on both sides.

---

## Item 5 — audit-ordering comment language  ✅ CLOSED

`CMakeLists.txt` lines 404–434 already document the audit/ → compat/libsecp256k1_shim
ordering as **intentional architecture**, not "API drift". The comment explicitly says:

> "This split is the intended architecture, not a transient workaround: it lets
> the unified runner remain a self-contained artifact that consumers can ship
> without pulling the shim, while the standalone targets keep providing strict
> regression coverage of the shim ABI surface."

The two surfaces are described as complementary:

* `unified_audit_runner` — shim-independent by design, uses `ADVISORY_SKIP_CODE(77)`
  stubs from `audit/shim_run_stubs_unified.cpp`.
* Standalone CTest targets — authoritative shim-dependent gate, gated on
  `if(TARGET secp256k1_shim)` after the shim subdirectory is processed.

No edit needed. Verified clean.

---

## Item 6 — PR scope narrow (Bitcoin Core only)  ✅ DEFERRED to owner

PR body content is owner-controlled per the CLAUDE.md "GitHub Announcements
Prohibition" rule. This file records the **technical scope recommendation**
for the PR body author:

* **In scope:** `bitcoin-core` deployment profile only. Reference
  `docs/SHIM_FOOTPRINT_COMPARISON.md`, `docs/BITCOIN_CORE_BACKEND_EVIDENCE.md`,
  `docs/bench_unified_2026-09-03_gcc14_x86-64.json`, and
  `docs/BITCOIN_CORE_BENCH_RESULTS.json`.
* **Out of scope (do not mention in PR body):** Litecoin / Dogecoin / BCH /
  Wallet profiles, GPU backends (CUDA/OpenCL/Metal), CAAS framework, the
  ai_memory / knowledge_base / source_graph internal tooling.
* **Open follow-ups (post-merge, not blocking):** TASK-001/002 owner-controlled
  bench regen (playbook: `docs/BENCH_REGENERATION_PLAN.md`); `dev → main`
  release merge (explicit owner instruction required).

Reviewer-facing entry-point: `docs/CAAS_REVIEWER_QUICKSTART.md`.

---

## Summary

All 6 items closed. The branch is in a state where:

* All deployment profiles configure cleanly + `bitcoin-core` links cleanly.
* No-LTO performance discussion uses measured numbers and names both
  root-cause fixes (PERF-002 + DER parser fast-path) instead of a single
  "i-cache pressure" hand-wave.
* Hot-path size delta is documented with a reproduction recipe.
* No "API drift" language remains in build-system comments.
* PR scope is documented for the owner to apply when writing the PR body.

The only remaining gate is owner authorisation to open the PR.
