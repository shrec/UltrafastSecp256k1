# P4 field sum: frozen GCC emitted-code observations

Status: static inspection complete, pending manager review. No speed, PMU,
constant-time, production-integration or upper-layer claim is made here.
This subtask performed no build, correctness run, benchmark or policy change.

Frozen benchmark:

    /tmp/parseatlas-p4-field-sum.EFROsY/p4_field_sum_compare_gcc
    SHA256 4e5d4be103774c8fe3ee34f6672a3232798b03751d42ce3f48b26103a4254f6e

Driver SHA256:

    29c9fa86318a35b58a440245779f51b66e8327e68f83e4bec0f00bdd6cd1013a

P4 kernel SHA256:

    e3fa770e084afe647e9214ca309dbc6caf52cf2684986b43c3ed48141d7da75b

The [retained JSON](data/p4_codegen_20260905.json) contains exact hash
observations, route bindings from the driver, selected nm lines, all commands,
twenty bounded disassemblies, eleven named loop/body excerpts, actual call
edges and full-output stores. Addresses below are hexadecimal virtual
addresses in this binary; JSON end addresses are exclusive. Assembly uses
AT&T syntax. Full-series numerical evidence is **PENDING IMPORT** here.

## Individual symbols and actual outlined callees

| Route or actual job callee | Start | Size hex | Size bytes |
|---|---:|---:|---:|
| 0: FE64 returning API serial job | 0xdd30 | 0xd1 | 209 |
| 1: FE64 inline eager job | 0xde20 | 0x223 | 547 |
| 2: FE64 wide16 / one bank job | 0x11620 | 0x615 | 1557 |
| 3: FE64 wide4094 / one bank job | 0x112b0 | 0x36b | 875 |
| 4: FE64 wide16 / four-bank job wrapper | 0xf7f0 | 0x91 | 145 |
| 4: outlined wide16 / four-bank implementation | 0xeb00 | 0xc79 | 3193 |
| 5: FE64 wide4094 / four-bank job wrapper | 0xea60 | 0x91 | 145 |
| 5: outlined wide4094 / four-bank implementation | 0xe050 | 0x99e | 2462 |
| 6: FE52 E2E Full16 job wrapper | 0x1aea0 | 0x73 | 115 |
| 6: outlined FE52 Full16 implementation | 0x1cc30 | 0xb87 | 2951 |
| 7: FE52 E2E Full4094 job wrapper | 0x1af70 | 0x73 | 115 |
| 7: outlined FE52 Full4094 implementation | 0x1d7c0 | 0x956 | 2390 |

Actual calls are 0xf872→0xeb00, 0xeae2→0xe050, 0x1aed6→0x1cc30 and
0x1afa6→0x1d7c0. The small wrapper sizes must not be reported as whole kernels.

Additional common inspected symbols:

| Symbol | Start | Size hex | Size bytes |
|---|---:|---:|---:|
| FE64 returning operator+ | 0x2b330 | 0x55 | 85 |
| FE64 add_impl | 0x1e300 | 0x137 | 311 |
| FE64 to_bytes | 0x29000 | 0x2e | 46 |
| FE64 default constructor | 0x1ebf0 | 0x10 | 16 |
| FE64 limbs/bool constructor | 0x1ec00 | 0x21 | 33 |
| measure, non-cold .isra.0 clone | 0x16900 | 0x148f | 5263 |

Standalone one-bank template copies also exist at 0x1c080 (0x594,1428 bytes)
and 0x1c640 (0x2f7, 759 bytes). Routes 2 and 3 contain their arithmetic inline
and do not call these copies. They are retained for audit, not added to the
timed jobs' sizes.

Every size is an individual ELF symbol extent, not a transitive hot-path
footprint. Other callees, cold clones, read-only constants, PLT and alignment
are excluded. The limbs/bool constructor has a false-normalized-flag tail
jump to another normalizer; that target was not inspected. The returning
addition calls this constructor with the normalized flag set. No whole
instruction-cache footprint or cache cost is inferred.

## Eager API versus inline eager

The FE64 API loop 0xdd90–0xddb8 consumes one 32-byte RHS per iteration and
calls operator+ at 0xdd9c. That operator calls add_impl at 0x2b34e and the
limbs/bool constructor at 0x2b35f. Its scalar add/adc arithmetic, conditional
canonical correction, return temporary and accumulator copy remain visible.
The job reads the temporary from 0x20(%rsp) and writes its accumulator at
0x50(%rsp) on each loop iteration.

The inline eager loop 0xded8–0xdfea also has a 32-byte input stride but
contains no function calls. Scalar add/adc chains compute the raw sum and
add-K correction; mask and/or instructions select the canonical result.
The default FE64 constructor is called once after this loop, at 0xe00e,
not per RHS. Thus the intended absence of per-RHS production API calls
survives this compilation.

Inline eager is not stack-free: examples inside its arithmetic loop include
stores at 0xdefd and 0xdf64 and reads at 0xdf89 and 0xdfc8. These are explicit
stack operands, not a classification of every access as a compiler spill.
Nor is this a pure identical-machine-code call-removal ablation: the two
routes also differ in instruction selection and temporary placement. Any
later measured difference cannot be attributed exclusively to call overhead.

## Native wide columns remain scalar in this build

The native eager and wide arithmetic paths contain no packed integer
add/sub instructions. They use scalar add/adc for unsigned128-bit arithmetic.
Vector moves and zeroing appear elsewhere, but are not packed arithmetic.

### One bank, chunk4094

The accumulation loop 0x11400–0x11426 is:

~~~text
11400: add    (%rcx),%rax
11403: adc    $0x0,%rdx
11407: add    0x8(%rcx),%r14
1140b: adc    $0x0,%r15
1140f: add    0x10(%rcx),%r8
11413: adc    $0x0,%r9
11417: add    0x18(%rcx),%r10
1141b: adc    $0x0,%r11
1141f: add    $0x20,%rcx
11423: cmp    %rcx,%rdi
11426: jne    11400
~~~

Four independent low/high register pairs accumulate the four input words:
each adc belongs to its own 128-bit column, not an inter-limb field carry.
The loop consumes one FE64 RHS per iteration and has no calls or explicit
rsp/rbp memory operands in the displayed range. Seed merging and cross-column
carry propagation start after it; for example, 0x11428 reads a prefix word
from 0x30(%rsp). Surrounding normalization and final output still use stack
slots.

### Four banks, chunk4094

The actual loop is in the outlined implementation at 0xe278–0xe376.
It consumes 128 bytes/four FE64 objects per iteration. Arithmetic remains
scalar; some bank columns use register pairs and others use stack-relative
read-modify-write operands:

~~~text
e278: mov    (%r14),%r15
e27b: add    %r15,0x90(%rsp)
e283: adcq   $0x0,0x98(%rsp)
...
e309: add    0x40(%r14),%r10
e30d: adc    $0x0,%r11
...
e36d: sub    $0xffffffffffffff80,%r14
e371: cmp    %r14,0x58(%rsp)
e376: jne    e278
~~~

The subtraction of -128 advances the pointer by 128. The complete retained
range records all sixteen column additions and their adc high-word updates.
The code is genuinely four-bank accumulation, not a packed SIMD add loop.
Its explicit stack accesses are evidence of storage placement, not measured
spill cost, port pressure, latency or bandwidth.

### Chunk16 and boundary work

The one-bank chunk16 job unrolls up to sixteen RHS updates with public-count
exits; examples are 0x11731–0x11a42. There is no per-RHS backedge in that
unrolled body. Chunk merge/reduction follows at 0x11a48, and the outer
backedge is 0x11bcc→0x116f0. Full chunks cover sixteen inputs/512 bytes;
the selected public short chunk may be smaller.

The four-bank chunk16 callee similarly uses unrolled scalar bank updates,
with count-dependent paths and many explicit stack slots. The retained
0xeb78–0xf070 range includes initial bank loading and the full-chunk update
sequence; it is not a single short uniform per-RHS loop.

Native chunk boundaries retain carry merging, an h*K multiplication,
overflow folding and canonical correction. Examples are 0x11a48–0x11bcc
for one-bank16 and 0xe811–0xe98c for four-bank4094. Fewer normalization
boundaries do not imply cost-free merging or reduction.

## FE52 controls in this same P4 binary

Both FE52 control jobs call outlined implementations. They must be assessed
using these P4 addresses and call boundaries, not older P3 symbol sizes.

Full16 has a vectorized/unrolled sixteen-input chunk body with scalar
normalization at its boundary. Its backedge is 0x1d250→0x1cd80. YMM temporary
stores at 0x1ce25/0x1cea2/0x1cf12 and subsequent reads are visible inside
this body. Its final-chunk vector backedge is 0x1d3dd→0x1d310.

Full4094 has vector loops 0x1d8f0–0x1d9b3 and 0x1dc67–0x1dd29.
Each advances 128 bytes and loads four FE64 objects. Unpacks, permutations,
shifts and masks perform 4x64-to5x52 packing; five YMM vpaddq instructions
accumulate the resulting component streams. These inner ranges contain no
calls or explicit rsp/rbp memory operands. Horizontal combination, scalar
remainders and normalization lie outside them.

All eight routes enter with the same FE64 RHS layout in this experiment.
The FE52 controls pack at use; there is no resident 40-byte input route here.
Vectorization is an observed implementation difference, not evidence that
one representation has lower bandwidth cost or is faster.

## Continuous-region repeated jobs and complete output

The original start-clock call remains at 0x16a40. The repeated job loop is:

~~~text
16a80: mov  0x10(%r12),%rsi
16a85: mov  0x70(%rsp),%rdx
16a8a: mov  0x78(%rsp),%rdi
16a8f: call *%r14
16a92: add  $0x1,%r13
16a96: cmp  %rbx,%r13
16a99: jne  16a80
16aa5: call steady_clock::now
~~~

The checkpoint time is compared with the original start saved at 0x50(%rsp).
Extension control returns at 0x16b1f to 0x16a70, after the original clock call;
it does not reset the start clock. Checkpoint bookkeeping and the extension
decision therefore remain in a continuing region. Duration-accounting and
qualification correctness belong to the separate numerical/method audit.

The output pointer refers to the 32-byte region at 0x80(%rsp). Each job calls
to_bytes() and emits a full YMM store to the caller's output:

| Route | Serialization call | Final 32-byte store |
|---|---:|---:|
| 0 | 0xddc3 | 0xddd3 |
| 1 | 0xde6c | 0xde77 |
| 2 | 0x11675 | 0x11680 |
| 3 | 0x11307 | 0x11312 |
| 4 | 0xf831 | 0xf83b |
| 5 | 0xeaa1 | 0xeaab |
| 6 | 0x1aee1 | 0x1aeeb |
| 7 | 0x1afb1 | 0x1afbb |

The to_bytes body writes all four 64-bit big-endian output words.
Repeated jobs and complete output stores have not been hoisted away or
replaced by a partial checksum. This does not claim every identical timed
job is byte-checked: the protocol checks the last job per region outside
timing. Power calls at 0x16a22 and 0x16bbf lie outside the measured interval;
their conditioning effect is not quantified here.

## Limits and disposition

No emitted-code blocker found. Static evidence distinguishes API calls,
inline eager correction, scalar 128-bit bank accumulators and packed FE52
arithmetic. It does not establish dynamic instruction counts, cycles,
throughput, critical-path depth, cache traffic, spill cost, energy, port
pressure or which candidate wins. Equal input layouts do not establish
equal memory traffic. No CT guarantee follows from selected excerpts.

PMU remains manager-reported unavailable under perf_event_paranoid=4;
this subtask performed no counter retry or policy change.

This is supporting inspection by the kernel author, not independent
acceptance of the author's kernel. The manager retains role-independent
review and separate workers supply oracle/numerical evidence. Repository:
repo_666797171f0141c58bf05f579b2ee16e; manager session:
01a06be6-2904-7c62-9e7d-1245c34a5312. Task MCP remains owner-suspended.
