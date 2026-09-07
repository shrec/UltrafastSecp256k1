# P6 field boundary: emitted-code inspection

The intended call boundaries survive in both final binaries. For every multiply/square and chain1/ILP4 pair, the direct-ASM and outlined-row callers have identical normalized instruction sequences after changing only the arithmetic-leaf destination. Their arithmetic leaves are different. This is a verified boundary control, not a measured gain or a pure call-cost decomposition.

## Frozen inputs and build identity

- Inspected GCC binary: `/tmp/parseatlas-p6-boundary.imLUFT/p6_compare_gcc`; execution alias `p6_compare_gcc_frozen` is byte-identical.
- GCC SHA256: `c4f9fafc0430916b8b74f0e148f7d790d96cffd26ac7101fe2996b1e3f492a95`.
- Clang binary: `/tmp/parseatlas-p6-boundary.imLUFT/p6_compare_clang`; SHA256 `8cb5e31a6d1b06c21809677d2658fdd1ed4e98cbc600457f60cdb60e860be38f`.
- Driver SHA256: `563098f0e3e89b9fbc9d4e9d110ccf2be735419a7c5071923fbfb3dbccb72217`.
- Boundary header/TU SHA256: `d82b6043d609178c75762e8cfe2f89bb10af713db4eab00241779280118eefc9` / `61db99c4f102830861f4b17305faeca9838fc9122d3eafb569cc03c31d98d8ba`.

[JSON evidence](data/p6_codegen_20260905.json) retains all exact commands, mangled names/aliases, nm lines, 66 complete raw-byte disassembly slices, byte and text hashes, 64 aligned call/output windows, and eight normalized caller comparisons. Every retained symbol's raw bytes cover its nm extent contiguously. GCC has 31 slices; the bounded Clang boundary crosscheck has 35, including five actually called evaluators. No performance measurements were run by this worker.

The manager built the driver and outlined row source as separate translation units with GCC14/Clang18, C++20, -O3 -march=native -fno-lto -DNDEBUG, the recorded ASM defines and include path. Both links use the same frozen corrected P1 library; the Clang crosscheck is of the driver/row compilation, not a recompilation of the entire library. Exact commands are in `build_manifest.commands`.

Retained GCC objects: `p6_driver_gcc.o` SHA256 `92ef9fcefab0682e6a4eda13904e2124f4a3fc02ebbf24907ea416048fdd61dc`; `p6_boundary_gcc.o` SHA256 `b9c2ce21db9a4ef9b538c8e8f7da336e0335ab5f8300562ac86d347c145de774`. Relinking these objects reproduced the inspected executable byte-for-byte. The worker verified these identities without building.

## All sixteen GCC hot jobs

Ranges are start-inclusive/end-exclusive; sizes are individual symbols, not transitive hot-path sizes. Clang's corresponding sizes are included only as a boundary crosscheck; its exact ranges and callees are in JSON.

| Cell | Route | Lanes | GCC range | GCC bytes hex / decimal | Clang bytes hex / decimal |
| --- | --- | ---: | --- | ---: | ---: |
| 0 | mul_api | 1 | 0x10d30..0x10e71 | 0x141 / 321 | 0x12d / 301 |
| 1 | mul_asm_direct | 1 | 0x16140..0x1629d | 0x15d / 349 | 0x122 / 290 |
| 2 | mul_row_inline | 1 | 0x12030..0x126c1 | 0x691 / 1681 | 0xa5 / 165 |
| 3 | mul_row_outlined | 1 | 0x15fe0..0x1613d | 0x15d / 349 | 0x122 / 290 |
| 4 | mul_api | 4 | 0x11640..0x1182a | 0x1ea / 490 | 0x160 / 352 |
| 5 | mul_asm_direct | 4 | 0x10aa0..0x10bf5 | 0x155 / 341 | 0x268 / 616 |
| 6 | mul_row_inline | 4 | 0x126d0..0x12ed7 | 0x807 / 2055 | 0x160 / 352 |
| 7 | mul_row_outlined | 4 | 0x10940..0x10a95 | 0x155 / 341 | 0x268 / 616 |
| 8 | square_api | 1 | 0x10c00..0x10d2a | 0x12a / 298 | 0x11b / 283 |
| 9 | square_asm_direct | 1 | 0x15e90..0x15fd6 | 0x146 / 326 | 0x100 / 256 |
| 10 | square_row_inline | 1 | 0x110e0..0x11633 | 0x553 / 1363 | 0xa5 / 165 |
| 11 | square_row_outlined | 1 | 0x15d40..0x15e86 | 0x146 / 326 | 0x100 / 256 |
| 12 | square_api | 4 | 0x10f10..0x110d4 | 0x1c4 / 452 | 0x2ba / 698 |
| 13 | square_asm_direct | 4 | 0x16430..0x165b9 | 0x189 / 393 | 0x231 / 561 |
| 14 | square_row_inline | 4 | 0x11890..0x12030 | 0x7a0 / 1952 | 0x160 / 352 |
| 15 | square_row_outlined | 4 | 0x162a0..0x16429 | 0x189 / 393 | 0x231 / 561 |

## Matched callers and real leaf destinations

After address/target normalization, all four GCC pairs and all four Clang pairs match exactly. GCC instruction counts are 73/75 for multiply chain1/ILP4 and 66/84 for square; Clang counts are 63/131 and 55/120. These are caller instruction counts, not whole arithmetic costs or dynamic execution counts.

Normalization strips instruction addresses and symbol annotations, converts local branch destinations to self-relative offsets, and preserves resolved external/RIP targets. It then replaces only the two specifically named arithmetic-leaf destinations in each pair with `ARITH_LEAF`. Registers, constants, instruction order and every other call target remain unchanged. The JSON retains the parser, normalization function, pair-specific leaf addresses and normalized hashes so the manager can recompute rather than trust a boolean. Raw code bytes naturally differ in relocations and leaf destinations.

GCC direct multiply calls `field_mul_full_asm` at `0x161cb` (chain) and `0x10b5c` (ILP4); outlined multiply calls `pa_p6_row_mul` at `0x1606b` and `0x109fc`. Direct square calls `field_sqr_full_asm` at `0x15f09` / `0x164c0`; outlined square calls `pa_p6_row_square` at `0x15db9` / `0x16330`. The exact pointer-argument setup and following state stores are retained in call windows.

These direct routes reach the very same ASM symbols used by the existing API dispatchers, without an extra forwarding function. GCC ASM ranges are `[0x3c4f3,0x3c755)` for multiply and `[0x3c755,0x3c95d)` for square. Clang link ranges are `[0x356c3,0x35925)` and `[0x35925,0x35b2d)`. The raw ASM bodies are identical between links: multiply is 610 bytes, SHA256 `a52d88ff51ddb4d842f134bede3e3d9158e4aa6107c1c9415c81e031e9673d7a`; square is 520 bytes, SHA256 `f9af4c245bcab5d164db04c5e26f7a30d9b8ee421d1df6b77131c0880c13d40a`.

No CPUID instruction or new native-admission reference occurs in the retained jobs and selected normal arithmetic/serialization callees. Existing FE64 `mul_impl`/`square_impl` still contain their cached ASM-availability guards, first-use feature-detection calls and portable fallback branches. The direct and outlined routes bypass those dispatchers. Caller admission before any direct operation remains a required experiment-local precondition; this audit does not execute that gate.

## Compiler visibility changes the arithmetic shape

| Arithmetic boundary | GCC | Clang |
| --- | --- | --- |
| Outlined row multiply | 1244-byte leaf, 16 product + 6 reducer multiply sites | 445-byte leaf, 16 product sites, calls 330-byte reducer |
| Outlined row square | 1300-byte leaf, 10 product + 6 reducer multiply sites | 379-byte leaf, 10 product sites, calls 330-byte reducer |
| Row-inline chain | Evaluator and reduction inside full_job | Outlined evaluator; reduction call on each step |
| Row-inline ILP4 | Scalar lane loop; reduction inside full_job | Outlined evaluator, four lanes unrolled, reduction inlined |

GCC's outlined leaves are at `0x206d0` and `0x20bb0`. Clang's are at `0x19f60` and `0x1a120`, with calls to `pa_p5::reduce_serial` at `0x1a0f4` / `0x1a277`. Small Clang leaf extents therefore do not imply proportionally small complete arithmetic paths. GCC arithmetic also includes explicit stack-relative operands; neither register syntax nor total extent is a cost measurement.

Clang actually outlines all four row-inline evaluators: chain multiply/square at `0x11e30` / `0x13530`, ILP4 at `0x12310` / `0x13980`. API multiply ILP4 also calls an evaluator at `0x12080`. These five complete bodies are retained because they are real hot callees; this differs from P5 GCC's untimed evaluator-only symbols.

The Clang chain evaluators contain 16/10 product multiply sites and call the shared reducer. Clang ILP4 evaluators have 84/60 static MULX sites in their four-lane bodies, corresponding to four sets of 16+5 or 10+5 products/reduction multiplications. Their bodies also contain packed integer ADD/SUB during reduction/canonicalization, e.g. `vpaddq` at `0x12dee` and `vpsubq` at `0x12eba`; this is not packed field multiplication. The GCC row-inline bodies instead preserve scalar lane loops and six reducer multiply sites. No amount of hardware overlap or throughput is inferred.

Both compilers reuse symmetric products in row self-multiplication, leaving ten product multiply sites rather than the generic source's sixteen. Source operation counts, static instruction sites and dynamic execution counts are different quantities.

Clang's shared reducer at `[0x18bc0,0x18d0a)` has five MULX sites and uses conditional selection for the final overflow-times-K fold. Its final canonical subtraction has a result-dependent branch: `test $0x1,%r14b` at `0x18cc3`, then `je 0x18cd7` at `0x18cc7`. The condition derives from the reduction result's subtraction borrow. The full branch and both paths are retained. Source masking is therefore not a CT guarantee; P6 makes no such claim.

The unchanged ASM arithmetic uses MULX with ADCX/ADOX and no arithmetic-body RSP/RBP-relative operands, while saving/restoring registers with PUSH/POP. C++ caller state and row arithmetic contain explicit stack-relative loads/stores. These observations do not establish compiler-spill cost, cache behavior, bandwidth or execution-port pressure.

## Repeated jobs, timing boundaries and full outputs

GCC loads the job pointer at `0x1d906`, takes the initial clock at `0x1d90a`, and repeatedly calls `*%r14` at `0x1d95f` with backedge `0x1d969 -> 0x1d950`. The checkpoint is at `0x1d975`; extension jumps `0x1d9ef -> 0x1d940` without restarting the initial clock.

Clang emits separate calibration-only and extensible-region loops: calls `0xc27a` / `0xc31a`, backedges `0xc280 -> 0xc270` / `0xc320 -> 0xc310`, checkpoint clocks `0xc282` / `0xc32a`. Both use the original clock taken at `0xc245`. Repeated full-job calls have not been hoisted away.

All sixteen GCC and sixteen inspected Clang jobs perform four final 32-byte stores at output offsets 0, 32, 64 and 96. GCC uses VMOVDQU, Clang VMOVUPS. Each job's four instructions and aligned final serialization/output window are retained. Chain1 initializes the inactive 96 bytes to zero; ILP4 serializes four active BE32 outputs. For example GCC direct multiply writes at `0x1623e`, `0x1624c`, `0x1625b`, `0x1626a`; Clang direct multiply writes at `0x14d44`, `0x14d49`, `0x14d4e`, `0x14d53`. Complete output observability survives compilation; numerical correctness is established by the separate manager/oracle gates.

## Replay and disposition

~~~bash
nm -S -n /tmp/parseatlas-p6-boundary.imLUFT/p6_compare_gcc
objdump -d -C --insn-width=16 --start-address=0x16140 --stop-address=0x1629d /tmp/parseatlas-p6-boundary.imLUFT/p6_compare_gcc
objdump -d -C --insn-width=16 --start-address=0x206d0 --stop-address=0x20bac /tmp/parseatlas-p6-boundary.imLUFT/p6_compare_gcc
~~~

Use each JSON `command_args` array for complete replay. `disassembly_sha256` hashes exact stdout; `raw_bytes_sha256` hashes decoded instruction bytes, not their hexadecimal text. `raw_byte_count` must equal the symbol extent. Matched-caller comparisons should be independently recomputed from the retained raw instructions.

Worker role, repository/session and Source Graph exact-target fallback are recorded in JSON; Task MCP remains owner-suspended. Only this report and its JSON were written. No builds, tests, benchmarks, CPU-policy changes or production edits occurred in this codegen task. The static inspection verifies the intended boundaries and documents compiler differences; it does not certify constant time or explain future speed ratios. The raw JSON evidence was independently replayed by the manager before full timing; the report prose was crosschecked afterward, following both full series.
