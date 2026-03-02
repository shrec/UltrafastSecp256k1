# Dead / Junk Code Inventory

**Date:** 2026-03-03 (updated 2026-03-03)
**Scope:** Primary (`libs/UltrafastSecp256k1/`), Root workspace (`Secp256K1/`)
**Purpose:** Identify orphaned, deprecated, and dead code for cleanup.
**Status:** Batch 1 executed -- 16 orphaned source files removed, bench_comprehensive_riscv alias removed.

---

## 1. Orphaned Source Files (NOT in CMake build graph)

| File | Evidence | Action |
|------|----------|--------|
| ~~`cpu/src/decomposition_optimized.hpp`~~ | Not compiled or #included. Dead GLV code. | **DELETED** |
| ~~`cpu/src/field_asm_x64.cpp`~~ | Not in CMakeLists. MSVC uses `.asm`, Clang uses `.S`. | **DELETED** |
| ~~`cpu/src/modinv_shim.h`~~ | Not #included anywhere | **DELETED** |
| ~~`cpu/bench/bench_field_operations.cpp`~~ | Not in CMakeLists | **DELETED** |
| ~~`cpu/bench/bench_glv_cache_analysis.cpp`~~ | Not in CMakeLists | **DELETED** |
| ~~`cpu/bench/bench_glv_cache_test.cpp`~~ | Not in CMakeLists | **DELETED** |
| ~~`cpu/bench/bench_h_based_inversion.cpp`~~ | Not in CMakeLists | **DELETED** |
| ~~`cpu/bench/bench_kg_glv_noglv.cpp`~~ | Not in CMakeLists | **DELETED** |
| ~~`cpu/bench/bench_lazy_reduction.cpp`~~ | Not in CMakeLists | **DELETED** |
| ~~`cpu/bench/bench_montgomery.cpp`~~ | Not in CMakeLists | **DELETED** |
| ~~`cpu/bench/bench_mutable_vs_immutable.cpp`~~ | Not in CMakeLists | **DELETED** |
| ~~`cpu/bench/bench_point_serialization.cpp`~~ | Not in CMakeLists | **DELETED** |
| ~~`cpu/bench/bench_turbo_intensive.cpp`~~ | Not in CMakeLists | **DELETED** |
| ~~`cpu/fuzz/fuzz_field.cpp`~~ | Not in CMakeLists (audit/ has own fuzz) | **DELETED** |
| ~~`cpu/fuzz/fuzz_point.cpp`~~ | Not in CMakeLists | **DELETED** |
| ~~`cpu/fuzz/fuzz_scalar.cpp`~~ | Not in CMakeLists | **DELETED** |

~~**Duplicate CMake target:** `bench_comprehensive_riscv` is an alias for `bench_comprehensive` (same source). Remove alias.~~ **DONE**

---

## 2. Vendored Repo (LARGE)

| Path | Evidence | Action |
|------|----------|--------|
| `cpu/secp256k1/` | Full bitcoin-core/secp256k1 clone with .git/. NOT referenced by CMake. Contains build artifacts. Redundant with `_research_repos/secp256k1`. | Delete |

**Estimated size: ~50 MB**

---

## 3. Build Artifacts in Source Tree

### UltrafastSecp256k1 Root

Category                | Count | Examples               | Action
------------------------|-------|------------------------|--------
Compiled binaries       | ~15   | `bench_asm_inline.exe`, `_test_cfl_repro.exe` | Delete
Object files            | 2     | `ct_scalar.obj`, `field.obj` | Delete
DLLs                    | 1     | `secp256k1-6.dll` (1.4 MB) | Delete
Git bundles             | 4     | `ct_riscv_fix.bundle` etc (~144 MB total) | Delete
Benchmark outputs       | 74    | `benchmark-x86_64-windows-*.txt` | Delete
Audit/debug logs        | ~20   | `audit_full_dump.txt`, `ci_log.txt` | Delete
Temp debug files        | ~10   | `_test_cfl_repro.cpp`, `_fix_alerts.py` | Delete
Build dirs              | 41    | `build-*/` | Delete

### cpu/ Root

Category                | Count | Examples               | Action
------------------------|-------|------------------------|--------
Object files            | 44    | `*.o` files | Delete
Compiled binaries       | 2     | `bench_vs_libsecp.exe` | Delete
Cache files             | 1     | `cache_w18.bin` (244 MB) | Delete
Bench outputs           | 2     | `bench_out.txt` | Delete
Build dirs              | 9     | `build-*/` | Delete

---

## 4. Workspace Root Junk

Category                | Count | Examples               | Action
------------------------|-------|------------------------|--------
Benchmark outputs       | 91    | `benchmark-x86_64-windows-*.txt` | Delete
Debug text files        | ~50   | `bench_*.txt`, `debug_*.txt` | Delete
Test binaries           | 3     | `test_btc_modinv.exe` | Delete
Cache binaries          | 2     | `cache_w15_glv.bin` (73 MB), `cache_w18.bin` (244 MB) | Delete
Git bundles             | 3     | `ct_fix.bundle` (44 MB) | Delete
Build dirs              | 41    | `build-*/` | Delete
RISCV fix notes         | ~15   | `RISCV_FIX_*.md`, `RISCV_DEBUG_*.md` | Archive or delete
One-off scripts         | ~10   | `run_arm64_tests.ps1` etc | Move to `scripts/` or delete

---

## 5. Stale Documentation

| File | Evidence | Action |
|------|----------|--------|
| `RELEASE_NOTES_v3.6.0.md` | Current version 3.16.0 | Archive |
| `RELEASE_NOTES_v3.7.0.md` | Very old | Archive |
| `RELEASE_v3.14.0.md` | Old, inconsistent naming | Archive |
| `_release_notes_v3.16.0.md` | Working draft (underscore prefix) | Finalize or delete |
| `ANNOUNCEMENT_DRAFT.md` | Marketing draft | Delete |
| `INDUSTRIAL_ROADMAP_WORKING.md` | Working suffix = draft | Delete or finalize |
| `STACKER_NEWS_POST.md` | Social media draft | Delete |

---

## 6. Summary Statistics

| Category | Count | Est. Size |
|----------|-------|-----------|
| Orphaned source files | 16 files | ~75 KB |
| Vendored repo | 1 dir | ~50 MB |
| Build artifacts (all levels) | ~150 files | ~400+ MB |
| Build directories | **91 dirs** | **Multiple GB** |
| Benchmark/debug text | ~235 files | ~3 MB |
| Stale docs | 7 files | ~100 KB |
| **Total reclaimable** | | **~3-5+ GB** |

---

## 7. Cleanup Safety Rules

1. Every deletion requires green full test suite before AND after.
2. No behavior change -- only unreferenced/dead code removed.
3. Cleanup PR batches should be small and focused (max 20 files per PR).
4. Build directories are local-only artifacts -- never committed.
5. `.gitignore` should be updated to prevent re-accumulation.

---

## 8. Priority Order

1. **Build artifacts** -- immediate, zero risk (not tracked by git)
2. **Build directories** -- immediate, zero risk (not tracked by git)
3. **Orphaned bench/fuzz files** -- low risk, easy to verify
4. **Orphaned source files** -- verify with `grep -r` before deleting
5. **Vendored repo** -- verify no external references
6. **Stale docs** -- lowest priority, archive preferred over delete
