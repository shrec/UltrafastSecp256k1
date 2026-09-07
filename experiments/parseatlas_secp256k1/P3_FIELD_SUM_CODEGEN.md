# P3 field sum: frozen GCC emitted-code observations

Status: static inspection complete; pending manager review. This is not a
performance result, CT assessment, PMU measurement, or production acceptance.
No source implementation, build, test, benchmark, or machine-policy change was
performed by this inspection subtask.

Inspected binary:

    /tmp/parseatlas-p3-field-sum.ebhSnl/p3_field_sum_compare_gcc

SHA256:

    32702b980349a2f5e6cbf8f29fc882f5bebb109e1a2c7f009b7fccd2fdfcbb91

The [retained JSON](data/p3_codegen_20260905.json) contains the hash observation,
selected nm -S -C lines, all exact commands, twelve bounded disassemblies,
twelve named loop excerpts, call sites, and output-store addresses. Addresses
below are hexadecimal virtual addresses in this exact ELF, not source lines.
The disassembly format is AT&T. End addresses recorded in JSON are exclusive.

## Individual symbol sizes, not whole hot-path sizes

| Route / symbol | Start | Size hex | Size bytes |
|---|---:|---:|---:|
| 0: FE64 serial job | 0xd5f0 | 0xd1 | 209 |
| 1: FE52 E2E Full1 job | 0x14250 | 0x5a3 | 1443 |
| 2: FE52 E2E Full16 job | 0x14800 | 0xbf0 | 3056 |
| 3: FE52 E2E Full256 job | 0xd6d0 | 0x93b | 2363 |
| 4: FE52 E2E Full4094 job | 0x153f0 | 0x9aa | 2474 |
| 5: FE52 E2E Weak4095 job wrapper | 0x1cda0 | 0x73 | 115 |
| 5: separate Weak4095 sum implementation | 0x1e190 | 0xb61 | 2913 |
| 6: FE52 resident Weak4095 job | 0x113b0 | 0x644 | 1604 |
| FE64 returning operator+ | 0x2bf10 | 0x55 | 85 |
| FE64 add_impl | 0x1eee0 | 0x137 | 311 |
| FE64 to_bytes | 0x29be0 | 0x2e | 46 |
| measure, non-cold symbol | 0x191e0 | 0x9d7 | 2519 |

These are individual ELF symbol extents. They exclude other callees, cold
clones, PLT entries, read-only constants and inter-symbol alignment. In
particular, route 5 calls its 2913-byte sum implementation at 0x1cdd6. Its
wrapper-plus-named-sum subtotal is 3028 bytes, not 115 bytes; even 3028 is not a
transitive hot-path total. No cross-route instruction-cache cost follows from
the table. Constructor calls and stack-check failure paths are recorded but
their implementation sizes are not included or claimed inspected.

## Arithmetic and loop shapes

### FE64 serial: actual returning API in the RHS loop

The loop 0xd650–0xd678 advances the FE64 input pointer by 0x20 bytes, calls
FieldElement::operator+ at 0xd65c, and copies the returned 32-byte temporary
into the accumulator. The returning API contains prefixed direct calls:

~~~text
2bf2e: addr32 call 1eee0 <add_impl>
2bf3f: addr32 call 1f7e0 <FieldElement(limbs,bool)>
~~~

The add_impl symbol contains scalar add/adc carry propagation, the reduction
constant 0x1000003d1, and cmove selection. Examples include 0x1ef21–0x1ef64
and 0x1efc0–0x1efdc. Vector instructions in its result packing do not make
its carry arithmetic a packed-vector addition algorithm.

The RHS loop explicitly reads 0x20(%rsp) and writes 0x50(%rsp) when copying
the returned value. The add_impl body also reads/writes stack-relative slots.
These are observed memory operands, not a claim that all are compiler spills
or a measurement of their cost.

### Full1: inline scalar packing and normalization

The nonfinal-RHS loop 0x143a0–0x144ea advances by 0x20 bytes. It contains
scalar shifts/masks for 4x64-to-5x52 packing and radix52 carry/fold work in
the same loop. No vpaddq occurs in this loop. Its explicit stack operand
at 0x144e5 is a loop-limit comparison against 0x20(%rsp), not an accumulator
spill classification. Final-chunk decoding is outside this nonfinal loop.

### Full16: a larger unrolled/vectorized chunk body

The main chunk loop 0x149b0–0x14e92 consumes 0x200 bytes, or sixteen FE64
inputs, per iteration. Its body contains YMM loads, vpunpcklqdq,
vpunpckhqdq, vpermq, packed shifts/masks, vpaddq, horizontal reductions,
and scalar normalization. Thus fewer source-level normalization boundaries
do not mean that the compiled chunk has no packing, reduction, or temporary
storage work.

Inside this main chunk body, YMM stores occur at 0x14a58/0x14ad5/0x14b45
to 0xa0/0xc0/0xe0(%rsp), with later reads of those slots. Other scalar and
constant stack slots are also used. The separate final-chunk vector loop
0x14f50–0x1501d has no explicit rsp/rbp memory operands.

### Larger E2E chunks: four FE64 inputs per packed loop iteration

| Route | Nonfinal vector backedge | Final vector backedge |
|---|---|---|
| Full256 | 0xd93f → 0xd869 | 0xdc1e → 0xdb54 |
| Full4094 | 0x15636 → 0x15570 | 0x159a1 → 0x158df |
| Weak4095, separate implementation | 0x1e392 → 0x1e2d0 | 0x1e89c → 0x1e7da |

Each listed vector loop advances its input pointer by 128 bytes, loads four
32-byte FE64 objects, and uses shuffles/unpacks plus shifts/masks to form
packed 52-bit components before five YMM vpaddq accumulation instructions.
These are compiler-emitted operations in the actual E2E path: RHS packing
has not been excluded as resident preprocessing.

No listed inner vector range contains a function call or an explicit
rsp/rbp memory operand. This statement is limited to those ranges. Horizontal
combination, scalar remainders, normalization, setup, and final serialization
are outside them. For example, the weak implementation uses stack slots at
0x1e565/0x1e577 and 0x1e6fc/0x1e70a in its surrounding chunk-boundary work.
Absence of explicit stack operands in one loop is not absence of stack work
throughout the job.

### Resident Weak4095: two 40-byte objects per packed iteration

The nonfinal loop is 0x11540–0x1155f; the final vector loop is
0x11769–0x11788. The nonfinal loop is retained verbatim here:

~~~text
11540: vpaddq (%rax),%xmm4,%xmm4
11544: vpaddq 0x10(%rax),%xmm3,%xmm3
11549: add    $0x50,%rax
1154d: vpaddq -0x30(%rax),%xmm2,%xmm2
11552: vpaddq -0x20(%rax),%xmm0,%xmm0
11557: vpaddq -0x10(%rax),%xmm1,%xmm1
1155c: cmp    %r10,%rax
1155f: jne    11540
~~~

The input object is 40 bytes, but the emitted loop stride is 80 bytes:
two objects per iteration, using five XMM packed 64-bit additions. There are
no packing shuffles, calls, or explicit rsp/rbp operands within these two
inner ranges. Register extraction/combination, any remaining RHS, weak/full
normalization, seed setup and final decoding remain outside these ranges.
For example, 0x11710 and 0x11715 read stack-relative setup values before the
final-region path. This is not a memory-bandwidth or cache-locality finding.

## Repeated jobs and full 32-byte output remain observable

The measured repetition is still in the binary:

~~~text
1927e: call steady_clock::now
19298: mov  0x10(%r13),%rsi
1929c: mov  0x50(%rsp),%rdx
192a1: mov  0x58(%rsp),%rdi
192a6: call *%r14
192a9: add  $0x1,%r12
192ad: cmp  %r12,%r15
192b0: jne  19298
192b2: call steady_clock::now
~~~

The output pointer passed through rdx points to the 32-byte region at
0x60(%rsp). Each route calls to_bytes() and copies one full YMM value
into its caller-provided output: stores 0xd693, 0x142b6, 0x14880, 0xd72d,
0x1544d, 0x1cdeb and 0x1140a for routes 0–6 respectively. The to_bytes body
uses four 64-bit big-endian loads and writes all offsets 0,8,16,24.
Consequently the retained binary does not replace repeated jobs with a
single hoisted result or a checksum-only partial output.

Power-observation calls 0x19261 and 0x192d3 are outside the clock endpoints.
This does not quantify their conditioning effect. The static stores also do
not claim that every repeated identical job is byte-checked: the protocol
checks the last job per region outside timing; the numerical/method reviewer
checks that harness contract separately.

## Limits and review disposition

Static code establishes that canonical full-only, larger deferred E2E, and
resident paths have materially different loop forms and call boundaries.
It does not establish which wins, the size of a gain, dynamic instructions
or micro-operations, critical-path cycles, register-pressure cost, cache
misses, memory traffic, bandwidth, energy, or end-to-end engine improvement.
No CT certification follows from selected branchless-looking excerpts.

The manager previously recorded PMU denial with perf_event_paranoid=4.
Counters remain unavailable, not zero. This subtask did not retry counters
or change policy. Full P3 performance is **PENDING IMPORT** here.

No emitted-code blocker found for the preregistered seven routes. This
inspection was performed by the kernel author as bounded supporting work;
it is not independent acceptance of that author's kernel. The manager
retains role-independent acceptance, and separate workers provide oracle
and numerical review. Repository identity:
repo_666797171f0141c58bf05f579b2ee16e; manager session:
01a06be6-2904-7c62-9e7d-1245c34a5312. Task MCP remains owner-suspended.
