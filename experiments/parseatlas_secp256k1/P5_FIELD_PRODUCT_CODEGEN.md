# P5 field product: emitted-code inspection

Static evidence for the frozen GCC benchmark, not a performance or constant-time result. The two source reducer schedules mostly become the same machine instruction sequence. The compiler also reduces row self-multiplication to ten product multiply instructions, so source operation counts are not sufficient to interpret the comparison.

## Frozen inputs and replay

- Binary: `/tmp/parseatlas-p5-field-product.3Z4xU1/p5_compare_gcc`
- Binary SHA256: `71de9cf785e47f338c480a90757f115d2be2237b6f7eb856264768b6abd50a22`
- Driver SHA256: `9adcabba6473c6d865a62d8980345bf790840e05ec318bc5832701f5f2b9867d`
- Kernel SHA256: `4c9bb9f0865a44c4da15480b9ad2adab45db0efceeb2581e1db572ee0ff04d26`

[Raw evidence](data/p5_codegen_20260905.json) retains 65 exact-address disassembly slices, each with its command, complete stdout, SHA256, symbol extent and decimal/hex size. These comprise all 20 timed `full_job` symbols, all 20 separately emitted `evaluate` symbols, 10 `primitive` symbols, and 15 helper/serialization/measurement/baseline symbols. GNU nm/objdump version: Ubuntu Binutils 2.42.

~~~bash
nm -S -n -C /tmp/parseatlas-p5-field-product.3Z4xU1/p5_compare_gcc
objdump -d -C --no-show-raw-insn --start-address=0x16460 --stop-address=0x16af1 /tmp/parseatlas-p5-field-product.3Z4xU1/p5_compare_gcc
objdump -d -C --no-show-raw-insn --start-address=0x3fcf3 --stop-address=0x3ff55 /tmp/parseatlas-p5-field-product.3Z4xU1/p5_compare_gcc
~~~

Every other exact command is stored in its JSON slice. Slice hashes cover the exact stdout, including the binary-path heading and final newline. JSON also retains the instruction parser and normalization function used for pair comparisons.

## All timed symbol extents

Ranges are start-inclusive/end-exclusive. These are individual symbol sizes, not transitive hot-path sizes: API wrappers call arithmetic outside their own extent, and all jobs call serialization. Chain1 has one active state; ILP4 has four scalar states on one core.

| Cell | Route | Lanes | Start | End exclusive | Bytes hex / decimal |
| --- | --- | ---: | --- | --- | ---: |
| 0 | mul_api | 1 | 0x12720 | 0x12861 | 0x141 / 321 |
| 1 | mul_row_serial | 1 | 0x16460 | 0x16af1 | 0x691 / 1681 |
| 2 | mul_comba_serial | 1 | 0x16fa0 | 0x175d1 | 0x631 / 1585 |
| 3 | mul_comba_parallel | 1 | 0x14de0 | 0x15411 | 0x631 / 1585 |
| 4 | mul_api | 4 | 0x14370 | 0x1455a | 0x1ea / 490 |
| 5 | mul_row_serial | 4 | 0x18e90 | 0x19697 | 0x807 / 2055 |
| 6 | mul_comba_serial | 4 | 0x19dc0 | 0x1a483 | 0x6c3 / 1731 |
| 7 | mul_comba_parallel | 4 | 0x17900 | 0x17fc3 | 0x6c3 / 1731 |
| 8 | square_api | 1 | 0x125f0 | 0x1271a | 0x12a / 298 |
| 9 | square_row_serial | 1 | 0x15a60 | 0x15fb3 | 0x553 / 1363 |
| 10 | square_comba_serial | 1 | 0x12e60 | 0x13499 | 0x639 / 1593 |
| 11 | square_comba_parallel | 1 | 0x15420 | 0x15a59 | 0x639 / 1593 |
| 12 | square_api | 4 | 0x141a0 | 0x14364 | 0x1c4 / 452 |
| 13 | square_row_serial | 4 | 0x17fd0 | 0x18770 | 0x7a0 / 1952 |
| 14 | square_comba_serial | 4 | 0x196a0 | 0x19dbc | 0x71c / 1820 |
| 15 | square_comba_parallel | 4 | 0x18770 | 0x18e8c | 0x71c / 1820 |
| 16 | reduce_serial | 1 | 0x14560 | 0x14964 | 0x404 / 1028 |
| 17 | reduce_parallel | 1 | 0x149d0 | 0x14dd4 | 0x404 / 1028 |
| 18 | reduce_serial | 4 | 0x15fc0 | 0x16453 | 0x493 / 1171 |
| 19 | reduce_parallel | 4 | 0x16b00 | 0x16f93 | 0x493 / 1171 |

`evaluate` is inlined into every timed `full_job`; none of the 20 jobs calls a separate evaluator or experimental product/reducer helper. The separately emitted evaluator/stepper tables are used by untimed validation. This distinction matters: `primitive<7>` calls outlined `square_comba` at `0x23300` (0x2e9 / 745 bytes), while primitives 1, 2, 5, 6 and 8 call outlined `reduce_serial` at `0x235f0` (0x2b8 / 696 bytes). Those particular outlined calls do not occur in the timed jobs. Their full bodies and calling primitives are retained, not counted as extra timed work.

## Serial versus parallel: what was actually emitted

| Pair of route IDs | Mode | Normalized instructions | Result |
| --- | --- | ---: | --- |
| 2 / 3, Comba multiply | chain1 | 391 | Identical |
| 2 / 3, Comba multiply | ILP4 | 415 | Identical |
| 6 / 7, symmetric square | chain1 | 404 | Identical |
| 6 / 7, symmetric square | ILP4 | 458 | Identical |
| 8 / 9, raw reduction | chain1 | 247 | Two staging-register differences only |
| 8 / 9, raw reduction | ILP4 | 276 | Identical |

Normalization removes instruction addresses, symbol annotations and comments; local branch targets become offsets within the same symbol, external direct targets retain absolute addresses, and RIP-relative references retain their resolved absolute targets. Mnemonics, registers, constants, addressing modes and order are preserved. This is textual instruction equality after relocation normalization, not raw-byte equality or linker coalescing.

The raw chain1 exception is explicit: normalized indexes 42 and 47 use `mov %rcx,%rbx` / `xor %rbx,%r12` in serial versus `%rdx` in parallel (`0x1463d` / `0x1465e` versus corresponding route-9 locations). All remaining normalized instructions match. The arithmetic reducer schedule is therefore not a distinct serial/parallel experiment in this executable. Distinct addresses and placement remain; static equality does not guarantee identical timing.

## Products, carries and loops

Counts below are static multiplication instruction sites in the named timed body, split before/within its reducer. Looped Comba sites must not be confused with dynamic operation counts.

| Timed C++ route, either mode | Product sites | Reducer sites | Product code shape |
| --- | ---: | ---: | --- |
| `mul_row_serial` | 16 | 6 | Straight-line product |
| `square_row_serial` | 10 | 6 | Straight-line product; repeated symmetric products reused |
| Comba multiply, either reducer | 4 | 6 | Public column control flow remains |
| Symmetric Comba square, either reducer | 4 | 6 | Public column/diagonal control flow remains |
| Raw reducer, either reducer | 0 | 6 | Straight-line reducer within state loop |

The row-multiply chain body has 16 product MULX sites from `0x16551` through `0x167d3`, followed by six reducer multiply sites. Row self-multiply has only ten product MULX sites from `0x15b1d` through `0x15ca7`, although its generic source call contains 16 products. The ILP4 bodies show the same 16/10 split. Consequently, a claimed ten-versus-sixteen advantage of explicit symmetric square over this compiled row-square control would be incorrect.

General Comba retains indexed public-column branches, e.g. route 3 at `0x14ec0` onward, with ADD/ADC sequences maintaining its multiword column accumulator. Symmetric Comba likewise retains branches on public column/diagonal positions (`0x12f13`, `0x12f2a`, `0x12f34`) and explicit carry additions around product sites. Four multiply sites in these loops do not mean four executed multiplications per field operation. No instruction-count-to-latency conversion is made.

Representative outer backedges: row chain `0x16a4a -> 0x16520`; row-square chain `0x15f07 -> 0x15b10`; symmetric-square chain `0x1341b -> 0x12ee3`; raw-reducer chain `0x148c6 -> 0x14610`. ILP4 retains a scalar lane loop with 32-byte state stride, e.g. row `0x19604 -> 0x18f8a` with `add $0x20,%r14` at `0x195f1`, then the outer 1024-step backedge `0x1961a -> 0x18f18`. Four independent source states are not four vectorized arithmetic lanes, nor proof of a particular amount of hardware overlap.

No packed integer ADD/SUB/MUL or ADCX/ADOX instructions occur in the 20 `full_job` bodies. The arithmetic uses scalar MULX/MUL and ADD/ADC/SBB. Vector moves and zeroing are present for state/output handling; those are not vector field multiplication.

## Actual baseline and stack evidence

The baseline calls are `full_job -> FieldElement::operator* / square -> mul_impl / square_impl -> field_*_full_asm` on the cached BMI2/ADX-supported path. The public operators occupy 0x55 / 85 bytes each (`0x31ba0`, `0x31c00`); dispatchers occupy 0xea / 234 and 0xda / 218 bytes (`0xa980`, `0xaa70`). The support guard and portable fallback branch remain visible. Static inspection does not independently sample which branch ran; the baseline configuration/runtime gates belong to the manager's separate validation.

- `field_mul_full_asm`: `[0x3fcf3,0x3ff55)`, 0x262 / 610 bytes; 16 product plus five reducer MULX sites, 20 ADCX and 20 ADOX sites, no internal branches/calls.
- `field_sqr_full_asm`: `[0x3ff55,0x4015d)`, 0x208 / 520 bytes; ten product plus five reducer MULX sites, 17 ADCX and nine ADOX sites, no internal branches/calls.

The ASM final overflow fold uses NEG/AND (`0x3fef0` / `0x3fef3` for multiply); C++ retains a sixth reducer multiplication for the overflow bit times K (`0x14779` in raw chain serial). This is concrete instruction evidence, not a speed explanation.

Both ASM symbols save/restore registers with seven pushes/pops and store four output words; their arithmetic bodies have no explicit RSP/RBP-relative memory operands. In contrast, experimental bodies have explicit stack-relative operands inside arithmetic, not only prologues: row chain stores at `0x1655a` / `0x1656a` and reloads at `0x16562`; ILP row includes `mulx 0xa0(%rsp)` at `0x18fd2`; Comba chain has stack-relative column additions at `0x14f1c`. The JSON phase windows retain these exact instructions. Such operands are not automatically compiler spills; register-held pointers can also address stack objects. No spill cost, bandwidth, cache-miss or port-pressure conclusion follows.

## Repeated work and complete outputs

The measurement loop loads the selected job pointer at `0x22415`, takes the initial clock at `0x22419`, and calls `*%r14` at `0x2246f` inside the counted backedge `0x22479 -> 0x22460`. The checkpoint clock is at `0x22485`. Duration extension returns from `0x224ff` to `0x22450`, retaining the original start clock. The full-job call has not been hoisted out of repeated measurement.

All 20 jobs call their serialization clone after arithmetic and perform four 32-byte YMM stores to the caller's output, totaling 128 bytes. Each store address/instruction is retained in `full_jobs[].output_store_instructions`. Chain1 serialization zeroes the inactive 96 bytes (`0xdf0e`, `0xdf13`, `0xdf18`) and copies the active 32 bytes at `0xdf4a`; ILP4 serializes four 32-byte states via the loop `0xdea9 -> 0xde6c`. `to_bytes` at `[0x2f7b0,0x2f7de)` uses four MOVBE loads and four eight-byte stores. These checks concern materialized outputs and call boundaries, not independent mathematical correctness.

## Scope and disposition

Worker exact-target handoff: repository `repo_666797171f0141c58bf05f579b2ee16e`, manager session `01a06be6-2904-7c62-9e7d-1245c34a5312`. Task MCP remains owner-suspended; unavailable worker Source Graph and new-file indexing lag were recorded by the manager. No manager tools were impersonated. Only this document and its JSON were written.

All disassembly/source reads completed before the manager's first full series; documentation continued from retained text. No builds, benchmarks, CPU settings or production files were changed. PMU access remains unavailable according to existing manager evidence (`perf_event_paranoid=4`); it was not retried. No dynamic performance, CT certification or transitive code-size claim is made. Submitted for manager replay/review.
