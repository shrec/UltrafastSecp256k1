# Coding Standards

> SPDX-License-Identifier: MIT

This document defines the **required coding standards** for all contributions to UltrafastSecp256k1.
Every pull request MUST comply with these rules. Reviewers WILL reject PRs that violate them.

---

## 1. Language Standard

- **C++20** minimum (`-std=c++20`)
- No compiler extensions unless gated behind `#ifdef` with a portable fallback

## 2. Naming Conventions

| Element          | Convention         | Example                       |
|------------------|--------------------|-------------------------------|
| Namespaces       | `snake_case`       | `secp256k1::fast`             |
| Classes / Structs| `PascalCase`       | `FieldElement`, `Point`       |
| Functions        | `snake_case`       | `scalar_mul`, `field_add`     |
| Variables        | `snake_case`       | `result`, `carry_bit`         |
| Constants        | `UPPER_SNAKE_CASE` | `SECP256K1_P`, `CURVE_ORDER`  |
| Template params  | `PascalCase`       | `Limbs`, `Backend`            |
| Member variables | `m_snake_case`     | `m_limbs`, `m_magnitude`      |
| Macros           | `UPPER_SNAKE_CASE` | `SECP256K1_USE_ASM`           |

## 3. Formatting

- **Tool**: ClangFormat (`.clang-format` in repo root)
- Run `clang-format -i <files>` before committing
- Max line length: **120 characters**
- Braces: Allman style for functions, K&R for control flow
- Indent: 4 spaces (no tabs)

## 4. Hot-Path Contract (CRITICAL)

All performance-critical code paths MUST follow these rules:

### MUST

- Zero heap allocations
- Explicit buffer parameters (`out*`, `in*`, `scratch*`)
- Fixed-size POD types only
- In-place mutation
- Deterministic memory layout
- `alignas(32)` or `alignas(64)` where applicable
- Branchless algorithms where possible

### NEVER

- `new` / `delete` / `malloc` / `free`
- `std::vector::push_back` / `resize` in loops
- Exceptions, RTTI, virtual calls
- `std::string`, `std::iostream`, formatting
- Hidden temporaries or implicit conversions
- `%` or `/` when Montgomery/Barrett reduction is available

## 5. Memory Model

- **Single allocation -> full reuse** (arena/scratchpad pattern)
- Thread-local scratch buffers on CPU
- Pointer-based reset (no `memset` in loops)
- Caller owns all buffers -- clear ownership semantics

## 6. Cryptographic Correctness

- **No math changes** without explicit maintainer approval
- **No candidate dropping** -- every candidate must be evaluated
- **No probabilistic correctness** -- deterministic results required
- **No weakening of search coverage**
- Correctness **always** wins over performance

## 7. Endianness

- **Project standard**: Little-Endian (native x86/64)
- `FieldElement::from_limbs()` -- primary function for binary I/O (little-endian `uint64_t[4]`)
- `FieldElement::from_bytes()` -- **only** for standard crypto test vectors or hex strings (big-endian)

## 8. Documentation

- Doxygen-style comments for all public APIs
- Document complex algorithms with references (papers, BIPs)
- Include performance characteristics (`@performance`)
- Mark constant-time functions explicitly (`@note constant-time`)

```cpp
/**
 * @brief Multiply two field elements modulo p
 *
 * @param a First operand
 * @param b Second operand
 * @return Product a * b (mod p)
 *
 * @note This operation is constant-time
 * @performance ~16ns on x86-64 with assembly, ~25ns portable
 */
FieldElement field_mul(const FieldElement& a, const FieldElement& b);
```

## 9. GPU / CUDA Rules

- No dynamic allocation in device hot loops
- No per-iteration host/device sync
- Launch parameters derived from config, printed once at startup
- Use `CMAKE_CUDA_ARCHITECTURES` -- never hardcode `-arch=sm_XX`

## 10. Testing Requirements

- Every runtime-affecting change MUST include a test or deterministic repro command
- Place tests in `cpu/tests/` or `cuda/tests/`
- Use descriptive test names
- Test edge cases (zero, identity, order boundary, max limb values)
- Include performance regression tests for hot-path changes

## 11. Build Rules

- **Out-of-source builds only** -- never edit generated files
- Never commit build artifacts or anything under `build-*`
- All GitHub Actions pinned by SHA (no mutable tags)

## 12. Git Commit Standards

Commits MUST include:
- **What** changed
- **Why** it was changed
- **How to verify** (test command or repro steps)

Follow [Conventional Commits](https://www.conventionalcommits.org/):
- `feat:` -- new features
- `fix:` -- bug fixes
- `perf:` -- performance improvements
- `docs:` -- documentation
- `ci:` -- CI/CD changes
- `refactor:` -- code restructuring
- `test:` -- test additions/changes

## 13. Self-Check Checklist

Before submitting, verify:

- [ ] Hot paths listed and allocation-free
- [ ] No unnecessary copies in critical sections
- [ ] Scratch buffer reuse explained (if applicable)
- [ ] Performance impact documented (benchmarks if applicable)
- [ ] All existing tests pass
- [ ] New tests added for new functionality
- [ ] Code formatted with ClangFormat
- [ ] No compiler warnings

---

## References

- [CONTRIBUTING.md](../CONTRIBUTING.md) -- full contribution workflow
- [API Reference](API_REFERENCE.md) -- public API documentation
- [Building Guide](BUILDING.md) -- build instructions
- [Security Policy](../SECURITY.md) -- vulnerability reporting

---

> Performance comes from removing things, not adding them.
> Correctness comes from discipline, not hope.
