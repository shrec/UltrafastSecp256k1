# Local CI Failure Playbook -- Linux

**Last updated**: 2026-03-15

When a local Docker CI job fails, use this playbook to diagnose and resolve the issue before pushing.

---

## Quick Reference

| Failure | First Check | Common Fix |
|---------|------------|------------|
| Build error | `cmake --build` output | Fix code, check includes |
| Test failure | `ctest --rerun-failed --output-on-failure` | Read assertion message |
| ASan/UBSan | Stack trace in output | Fix undefined behavior |
| TSan data race | Thread sanitizer report | Add synchronization |
| Valgrind leak | `memcheck.*.log` | Add `secure_erase`/cleanup |
| clang-tidy | Warning list in report | Fix or suppress with `NOLINT` |
| Audit FAIL | `audit_report.json` verdict | Check which module failed |
| OOM killed | Docker memory limit | Increase Docker memory |

---

## Job-Specific Playbooks

### 1. GCC/Clang Build Failure

```bash
# Reproduce
docker compose -f docker-compose.ci.yml run linux-gcc
# or
docker compose -f docker-compose.ci.yml run linux-clang
```

**Common causes:**
- Missing include: check `src/cpu/include/secp256k1/` headers
- C++20 feature gap between GCC/Clang: use `#if __has_include` or feature macros
- LTO failures (GCC): check for ODR violations, symbol visibility

**Fix loop:**
1. Read the first error (ignore cascading errors)
2. Fix locally, rebuild with `cmake --build build-linux`
3. Re-run the failing Docker job

### 2. ASan/UBSan (Sanitizer) Failure

```bash
docker compose -f docker-compose.ci.yml run asan
```

**Reading the report:**
- `heap-buffer-overflow`: array index out of bounds
- `stack-buffer-overflow`: local array overrun
- `use-after-free`: dangling pointer
- `signed-integer-overflow`: UB in arithmetic
- `shift-exponent`: shift amount >= type width
- `null-pointer-dereference`: null deref

**Common hotspots:** field arithmetic carry propagation, scalar parsing edge cases, batch operations with zero-size input.

**Fix:** Address the root cause. Never suppress sanitizer findings without documented justification.

### 3. TSan (Thread Sanitizer) Failure

```bash
docker compose -f docker-compose.ci.yml run tsan
```

**Reading the report:** Look for `WARNING: ThreadSanitizer: data race`. The report shows two concurrent accesses to the same memory location.

**Common causes:**
- Shared mutable state without lock
- Non-atomic counter updates
- Static local initialization races

**Fix:** Library is designed single-threaded per context (`ufsecp_ctx*` is thread-local). If TSan fires, it's a real bug.

### 4. Valgrind Memcheck Failure

```bash
docker compose -f docker-compose.ci.yml run valgrind
```

**Reading the report:** Check `local-ci-output/valgrind/memcheck.*.log`.

- `Invalid read/write`: buffer overrun or use-after-free
- `Conditional jump depends on uninitialised value`: used before init
- `definitely lost`: memory leak (uncommon -- no heap allocation in hot path)

**Note:** `still reachable` at exit is benign (static precompute tables).

### 5. Audit Failure

```bash
docker compose -f docker-compose.ci.yml run audit
```

**Diagnosis:**
```bash
# Check which module failed
jq '.sections[].modules[] | select(.status != "PASS")' audit_report.json
```

**Common module failures:**
- `ct_equivalence`: CT vs FAST output mismatch -- regression in one layer
- `differential`: External lib comparison mismatch -- check test vector update
- `carry_propagation`: Field arithmetic carry bug -- critical, stop and fix
- `fiat_crypto_vectors`: Known-answer test failure -- check field/scalar changes
- `wycheproof_ecdsa/ecdh`: Standard vector failure -- signature/ECDH regression

### 6. clang-tidy / cppcheck

```bash
docker compose -f docker-compose.ci.yml run clang-tidy
docker compose -f docker-compose.ci.yml run cppcheck
```

**Policy:** These are informational in local CI. Review warnings but don't block on false positives.

**Suppress individual findings:** Use `// NOLINT(check-name)` with justification comment.

### 7. Benchmark Regression

```bash
docker compose -f docker-compose.ci.yml run bench-regression
```

**Note:** Local benchmark results are noisy (no core pinning, background processes). A regression >50% on local hardware is likely real. Between 10-50%, re-run on quiet system.

**If real regression detected:**
1. Identify which operation regressed from benchmark output
2. `git diff` the relevant source file
3. Check if an optimization was accidentally removed
4. Profile with `perf stat` or `perf record` if unclear

---

## Environment Issues

### Docker Not Running
```bash
sudo systemctl start docker
```

### Out of Disk Space
```bash
docker system prune -f
docker volume prune -f
```

### Build Cache Stale
```bash
docker compose -f docker-compose.ci.yml build --no-cache
```

### Permission Denied
```bash
sudo usermod -aG docker $USER
# Log out and back in
```

---

## Escalation Policy

| Severity | Action |
|----------|--------|
| Build fails on both GCC + Clang | Fix before push -- compiler-agnostic bug |
| ASan/UBSan finding | Fix before push -- undefined behavior |
| TSan data race | Fix before push -- concurrency bug |
| Valgrind invalid read/write | Fix before push -- memory safety |
| Audit module FAIL | Fix before push -- correctness regression |
| clang-tidy warning | Review, fix if trivial, else document |
| Benchmark regression >50% | Investigate, fix or document trade-off |
| Benchmark regression 10-50% | Re-run on quiet system, document if real |
