# Dead / Junk Code Inventory

**Date:** 2026-03-03 (updated 2026-03-03)
**Scope:** Primary (`libs/UltrafastSecp256k1/`), Root workspace (`Secp256K1/`)
**Purpose:** Identify orphaned, deprecated, and dead code for cleanup.
**Status:** **ALL DONE** -- Full cleanup executed. All sections completed.

---

## 1. Orphaned Source Files (NOT in CMake build graph)

| File | Evidence | Action |
|------|----------|--------|
| ~~`src/cpu/src/decomposition_optimized.hpp`~~ | Not compiled or #included. Dead GLV code. | **DELETED** |
| ~~`src/cpu/src/field_asm_x64.cpp`~~ | Not in CMakeLists. MSVC uses `.asm`, Clang uses `.S`. | **DELETED** |
| ~~`src/cpu/src/modinv_shim.h`~~ | Not #included anywhere | **DELETED** |
| ~~`src/cpu/bench/bench_field_operations.cpp`~~ | Not in CMakeLists | **DELETED** |
| ~~`src/cpu/bench/bench_glv_cache_analysis.cpp`~~ | Not in CMakeLists | **DELETED** |
| ~~`src/cpu/bench/bench_glv_cache_test.cpp`~~ | Not in CMakeLists | **DELETED** |
| ~~`src/cpu/bench/bench_h_based_inversion.cpp`~~ | Not in CMakeLists | **DELETED** |
| ~~`src/cpu/bench/bench_kg_glv_noglv.cpp`~~ | Not in CMakeLists | **DELETED** |
| ~~`src/cpu/bench/bench_lazy_reduction.cpp`~~ | Not in CMakeLists | **DELETED** |
| ~~`src/cpu/bench/bench_montgomery.cpp`~~ | Not in CMakeLists | **DELETED** |
| ~~`src/cpu/bench/bench_mutable_vs_immutable.cpp`~~ | Not in CMakeLists | **DELETED** |
| ~~`src/cpu/bench/bench_point_serialization.cpp`~~ | Not in CMakeLists | **DELETED** |
| ~~`src/cpu/bench/bench_turbo_intensive.cpp`~~ | Not in CMakeLists | **DELETED** |
| ~~`cpu/fuzz/fuzz_field.cpp`~~ | Not in CMakeLists (audit/ has own fuzz) | **DELETED** |
| ~~`cpu/fuzz/fuzz_point.cpp`~~ | Not in CMakeLists | **DELETED** |
| ~~`cpu/fuzz/fuzz_scalar.cpp`~~ | Not in CMakeLists | **DELETED** |

~~**Duplicate CMake target:** `bench_comprehensive_riscv` is an alias for `bench_comprehensive` (same source). Remove alias.~~ **DONE**

---

## 2. Vendored Repo (LARGE)

| Path | Evidence | Action |
|------|----------|--------|
| ~~`cpu/secp256k1/`~~ | Full bitcoin-core/secp256k1 clone with .git/. NOT referenced by CMake. 37.4 MB. | **DELETED (local)** |

~~**Estimated size: ~50 MB**~~ **DONE -- 37.4 MB reclaimed**

---

## 3. Build Artifacts in Source Tree -- **ALL DELETED (local)**

### UltrafastSecp256k1 Root

- ~~Compiled binaries (.exe, .obj, .dll, .pdb)~~ **DELETED**
- ~~Git bundles (3x, ~96 MB)~~ **DELETED**
- ~~Benchmark outputs (74 files)~~ **DELETED**
- ~~Audit/debug logs (~20 files)~~ **DELETED**
- ~~Temp debug files (_*.cpp, _*.py, _*.txt)~~ **DELETED**
- ~~Build dirs (38 dirs)~~ **DELETED** (kept build-linux)

### cpu/ Root

- ~~Object files (40 .o files)~~ **DELETED**
- ~~Compiled binaries (bench_vs_libsecp.exe)~~ **DELETED**
- ~~Build dirs (8 dirs)~~ **DELETED**

---

## 4. Workspace Root Junk -- **ALL DELETED (local)**

- ~~Benchmark outputs (91 files)~~ **DELETED**
- ~~Debug text files (~75 files)~~ **DELETED**
- ~~Test binaries (test_btc_modinv.exe)~~ **DELETED**
- ~~Cache binaries (cache_w15_glv.bin)~~ **DELETED**
- ~~Git bundles (3x, ~130 MB)~~ **DELETED**
- ~~Build dirs (51 dirs)~~ **DELETED** (kept build-linux)
- ~~One-off scripts (run_arm64_*.ps1/.sh)~~ **DELETED**

---

## 5. Stale Documentation -- **ALL DONE**

| File | Action |
|------|--------|
| ~~`RELEASE_NOTES_v3.6.0.md`~~ | **ARCHIVED** to `docs/archive/` |
| ~~`RELEASE_NOTES_v3.7.0.md`~~ | **ARCHIVED** to `docs/archive/` |
| ~~`RELEASE_v3.14.0.md`~~ | **ARCHIVED** to `docs/archive/` |
| ~~`_release_notes_v3.16.0.md`~~ | **DELETED** (already gone) |
| ~~`ANNOUNCEMENT_DRAFT.md`~~ | **DELETED** (already gone) |
| ~~`INDUSTRIAL_ROADMAP_WORKING.md`~~ | **DELETED** (untracked) |
| ~~`STACKER_NEWS_POST.md`~~ | **DELETED** (gitignored) |

---

## 6. Summary Statistics

| Category | Items Cleaned | Est. Reclaimed |
|----------|---------------|----------------|
| Orphaned source files | 16 files | ~75 KB |
| Vendored repo | 1 dir | ~37 MB |
| Build artifacts (all levels) | ~150 files | ~400+ MB |
| Build directories | **89 dirs** | **Multiple GB** |
| Benchmark/debug text | ~235 files | ~3 MB |
| Stale docs | 7 files (3 archived, 4 deleted) | ~100 KB |
| **Total reclaimed** | | **~3-5+ GB** |

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
