# P2: field representations, before integration

Owner decision, 2026-09-05: collect findings first and integrate qualified
changes together later. Finish the field-arithmetic investigation before
resuming scalar candidates or upper-layer work. P1's minimal correctness
repairs remain the frozen reference; this wave makes no production edits.

F2 is retained, not rejected or superseded: the measured Columns final-scalar
sum gain remains in `F2_REDUCTION_RESULTS.md`. Its CT adaptation and caller
integration are deferred. P1's lack of an accepted *P1 optimization* does not
negate the earlier F2 result.

## Question and boundaries

For add, subtract, multiply, square and nonzero inverse modulo
`p = 2^256 - 2^32 - 977`, what changes when we price the actual representation,
canonicalization, conversion and dependency structure together?

| Route | State / observation boundary | Security class in this experiment |
|---|---|---|
| FE64 | Actual corrected returning APIs, canonical 4x64 after every operation | Fast / variable-time permitted |
| FE52 bridge | FE64 input -> FE52 arithmetic -> canonical FE64 return for every operation | No new CT assurance |
| FE52 resident | Preconverted RHS corpus; seed packing and final decoding per timed job; fully normalize after each arithmetic step | Different amortized region contract, no new CT assurance |
| CT API | Actual `ct::field_add/sub/mul/sqr/inv` returning FE64 | Existing CT-designated API, arithmetic tests are not a CT certification |

Subtraction in FE52 is `a + b.negate(1)` for canonical inputs, followed by
canonicalization. Raw FE52 multiply/square output is not assumed to be `< p`.
All resident outputs are normalized in this first comparison; lazy scheduling
will be a separate experiment with an explicit magnitude budget.

FE52 inverse uses the explicit `inverse_safegcd()` route. Current native code
packs directly between 5x52 and signed62; the header's old FE64-bridge comment
is stale. It returns zero for zero, unlike FE64's throwing inverse. Timing
therefore uses nonzero inputs only. The CT inverse is the current fixed-divstep
implementation, not the Fermat algorithm described in its stale header.

## Gates before any speed conclusion

1. Independent C++ Boost-integer oracle, all 32 BE bytes, raw canonical FE64
   range and normalized FE52 bounds. Preserve both P1 failing KATs, boundaries,
   random cases, defined zero behavior, inputs and supported public aliases.
   Never violate raw kernel restrict contracts merely to test an alias.
2. Compile and smoke-test the new driver; independently review code and timing
   boundaries. No Python arithmetic or timing implementation.
3. Same-process, same-corpus comparison: four routes, five operations, chain1
   and ILP4 on ONE core. ILP4 is not multicore and is not an alternative formula.
4. Two series, eight measured rounds and two warmups per cell; alternate order
   and reverse the second series. Qualify with two consecutive regions at least
   200 ms. Keep all calibration, warmup and measured regions, including short
   regions. No selective deletion or undeclared reruns.
5. Compare complete output bytes outside timing. Disclose repeated identical
   jobs, final materialization and inverse's result-dependent public benchmark
   lookup, which contributes load/index overhead and is not a secret-safe API.

Resident RHS preprocessing is outside timed repeated jobs and must be reported
separately. A 256-entry FE64 corpus occupies 8192 bytes; the corresponding FE52
corpus occupies 10240 bytes. These are layout counts, not cache-miss evidence.
Bridge results include per-operation conversion; resident results must never
be presented as drop-in FE64 API speedups.

Review qualification: the first draft redundantly normalized a bridge output
before `to_fe()` normalized it again. The worker removed that extra pass before
full timing, and the independent test now checks the raw bridge residue before
its one canonical conversion. Pre-review smoke/binaries are retained but are
not performance evidence. The unary bridge helper still receives an unused
second reference; its possible argument-address/ABI cost belongs to this exact
wrapper measurement, not a claim about an optimal unary bridge.

## Frozen environment

- Branch `experiment/representation-search`, HEAD
  `fef231d4e4173bd016fb2a3a1eff67087396a203` plus recorded P1 repairs.
- Actual corrected CMake Release library from P1, explicitly LTO OFF:
  `/tmp/parseatlas-p1-primitives.1hlZNv/build-fixed/libfastsecp256k1.a`, SHA256
  `238294f2959e1b161428bca9a1baa990b803ee933f7f876b5b6e6cdc9b4fd577`.
- Native C++20, GCC14.2, `-O3 -march=native -DNDEBUG -fno-lto`, matching
  assembly/fast-reduction macros. Record exact commands and hashes separately.
- i5-14400F, observed 16 logical CPUs / 10 physical cores, CPU4 and sibling CPU5.
  Pin timing to CPU4; leave system governor/turbo unchanged and disclose them.
  Finish owned builds/tests before timing. External services remain uncontrolled.

## Evidence and next candidates

Use all 25 observation lenses in `P2_FIELD_25_LENSES.md`. Unknown quantities are
`PENDING IMPORT`, never invented zeros or qualitative guesses stated as data.
Atlas's decode relation and whole-region/depth/live-state reasoning transfer;
its 25 notation precedences are not 25 machine-arithmetic algorithms.

After this baseline: bounded deferred-normalization regions; carry scheduling;
product/reduction scheduling and symmetric square; inverse scheduling and
separately contracted batch inversion. Each needs its own before/after,
intermediate bounds, equal-work measurement and retained negative results.
No operation-count reduction, existing FE52 implementation, or changed notation
is by itself a new discovery or a measured engine speedup.

Workflow: owner's temporary Task MCP suspension remains in force; no canonical
card is claimed, launched or accepted. Manager bootstrap and task-health verified
the same repository (`repo_666797171f0141c58bf05f579b2ee16e`), manager session
`01a06be6-2904-7c62-9e7d-1245c34a5312`. Live manager Source Graph supplies exact
worker targets because worker-scoped tools are not exposed. Session/Memory/KB
queries returned no task-specific context. No plugin repair is attempted.
