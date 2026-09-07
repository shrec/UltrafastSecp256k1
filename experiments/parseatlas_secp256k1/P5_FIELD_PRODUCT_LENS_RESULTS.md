# P5 field products: measured results through 25 lenses

Status: evidence synthesis submitted for manager review. The unchanged
[preregistered lens plan](P5_FIELD_PRODUCT_LENSES.md),
[protocol](P5_FIELD_PRODUCT_PROTOCOL.md) and
[bound contract](P5_FIELD_PRODUCT_CONTRACT.md) govern these observations.
This is a field-only collection result, not production integration or
25 asserted discoveries. Field research remains unfinished.

Evidence: [main report](P5_FIELD_PRODUCT_RESULTS.md),
[series 1](data/p5_run1_20260905.json),
[reversed series 2](data/p5_run2_20260905.json),
[arithmetic qualification](data/p5_validation_20260905.json),
[synthetic duration qualification](data/p5_duration_validation_20260905.json),
[emitted-code report](P5_FIELD_PRODUCT_CODEGEN.md) and its
[raw static evidence](data/p5_codegen_20260905.json).

All timing ratios below were independently recomputed from measured raw
`elapsed_ns / operations`, where `operations = actual_jobs * 1024 * lanes`.
Samples are paired by round and matching contract; each series ratio is the
median of eight paired ratios, not a ratio of medians. S1/S2 denotes the two
GCC series, not two compilers. For a named A/B comparison, greater than one
favors B. Historical, smoke and synthetic-clock samples are not pooled.

S1 retained 262 regions: 62 calibration, 40 warmup and 160 measured. S2 retained
261: 61 calibration, 40 warmup and 160 measured. All 400 later-phase regions
met 200 ms; minimum durations were 206.721981 ms / 200.452804 ms. S1 retained
20 extensions, one each in 4 warmup and 16 measured regions; S2 retained five,
one each in 1 warmup and 4 measured regions. All 42 short calibration records
remain. Every recorded output, actual-work count, elapsed endpoint difference
and final cumulative probe was consistent. Input checks passed before and
after both series; all CPU endpoints were CPU4, without proving constant
frequency or absence of intervening migration.

**MEASURED** denotes finite tests or this machine's timings; **DERIVED** denotes
reasoning under the stated domain; **STATIC** denotes frozen emitted-code
inspection. Unperformed causal, security and portability gates remain
**PENDING IMPORT**.

| Lens | New observation | Limitation / next gate |
|---|---|---|
| V01 Reachability | **MEASURED / STATIC:** all 20 cells ran: four multiply routes, four square routes and two arbitrary-512-bit reducer routes, each in chain1/ILP4. All timed experimental arithmetic is inlined into its job; separately emitted evaluator/primitive helper calls belong to untimed validation. | Bind the actual timed bodies, not unrelated outlined helpers. No production dispatcher or default-build integration occurred. |
| V02 Corpus | **MEASURED:** qualification distinguishes 14,100 distinct raw pairs, 13,073 canonical pairs and 12,058 direct wide inputs. Performance uses four nonzero seeds, 256 canonical nonzero RHS values and 256 arbitrary-512-bit inputs, checksum `7c2e57392893ac80`. | These are different domains and overlapping checks, not one additive count of independent fixtures. Repeated compiler runs use the same finite corpus. |
| V03 Equivalence | **MEASURED / DERIVED:** native GCC, independently compiled Clang field sources and NO_ASM ASan+UBSan each passed 714,283 assertions, zero mismatches, checksum `75711b730275b15a`. Boost checks exact products and arbitrary-512-bit residues; corrected FE64 is an additional canonical reference. | Products require all eight raw words/64 bytes; residues require canonical four words/32 bytes. Finite tests and a written bound argument are not machine-checked universal verification. |
| V04 Definedness | **DERIVED / MEASURED:** arbitrary256 product inputs, arbitrary-512-bit reducer inputs and canonical field operands remain separate contracts. Native unsigned128 support is required; sanitizer qualification passed. | Do not normalize arbitrary product inputs away, import sum pointer/null rules, or infer unsupported compiler/platform behavior from this finite run. |
| V05 Range / overflow | **DERIVED / MEASURED:** row cells fit 128 bits; a Comba column can require 130 and its carry 66; doubled cross-products can require 129. Reduction retains q≤K and qK<2^65 before the final overflow fold. Targeted width checks passed. | P4 sum-only headroom and two-fold bounds do not apply. The arbitrary-512-bit reducer needs H*K, q*K and e*K folds plus canonical correction. |
| V06 Aliasing | **MEASURED:** each qualification configuration performed 54,468 same-object raw products and 201,672 preservation checks. Both timed series regenerated and compared every raw word of all four seeds, 256 RHS values and 256 wide values before/after execution. | Counts include repeated checks. Same-type read-only aliasing is covered; no cross-type, restrict or caller-owned output-buffer overlap contract is added. |
| V07 Observable output | **MEASURED / STATIC:** each series performed 51,200 per-step canonical-state/byte checks, observed no zero/one state, then compared complete compiled jobs. Every job materializes 128 bytes: four BE32 lanes for ILP4, or one BE32 lane plus 96 zero bytes for chain1. All region outputs matched. | Helper traces are separate from complete timed loops; only each region's last identical job is checked outside its clock. Every primitive step remains canonical despite final-only job serialization. |
| V08 Invariants / carries | **MEASURED / DERIVED:** qualification recorded 487 q=K cases, 504 qK-above-u64 cases, 103 second-fold overflows and eight final-correction cases, plus three named carry KATs. The all-max wide witness reduces to K²−1. | These phase counts overlap and are not distinct corpora. Internal word carry2, bit256 overflow1 and a full-word selection mask are different quantities; all must retain their specified width. |
| V09 Constant time | **DERIVED:** public recurrence indices and fixed job lengths do not establish secret-independent machine behavior. | CT certification, security substitution and secret-dependent dispatch remain outside scope. No timing result authorizes a VT/CT replacement. |
| V10 Resources | **DERIVED / STATIC:** canonical states are 32 bytes, raw products 64 and a Comba accumulator 24 before other live state. Experimental arithmetic contains explicit stack operands; ASM baseline arithmetic bodies have no explicit RSP/RBP-relative operands. | These are storage observations, not measured spills or peak occupancy. Register-held aliases can also reference stack data; no memory-cost attribution follows. |
| V11 Property visibility | **MEASURED:** the row+serial candidate won 15/16 multiply-chain and 16/16 square-chain pairs against the API. Conversely, the API won all 96/96 ILP4 primary comparisons against the three candidates for both operations. | There is no universal replacement. Dependency mode changes the measured winner; native row gains in chain1 do not transfer automatically to independent states. |
| V12 Dependency depth | **STATIC / MEASURED:** five of six serial/precomputed job pairs have identical normalized instruction sequences; raw chain differs only at two input-XOR staging-register operands. These builds do not realize distinct reducer arithmetic schedules. | Timing separation must be treated as a placement/measurement-control result, not evidence of a new parallel reducer algorithm. Normalized textual equality is not raw-byte equality or proof of identical runtime; causal depth remains unmeasured. |
| V13 Live state | **STATIC / MEASURED:** row and Comba arithmetic use stack-relative operands, and Comba retains public column control flow. Row+serial beat Comba+serial in all 64/64 secondary pairs across both operations/modes. | The complete generated row implementation wins this comparison; stack syntax, control flow and timing together do not isolate spill cost, carry latency or one source-schedule effect. |
| V14 Operations | **STATIC:** row multiplication has 16 straight-line product multiply sites; row self-multiplication has only 10 in both modes because GCC reuses symmetric products. Comba/symmetric bodies retain four looped product sites, not four dynamic products. | A ten-versus-sixteen machine-work advantage for explicit symmetric square over compiled row square is absent here. Source counts and static loop sites cannot be converted into cycles or a promised 1.6× gain. |
| V15 Conversion | **DERIVED / STATIC:** P5 keeps native 4×64 throughout; there is no FE52 packing or excluded resident conversion. The actual API retains its operator/dispatcher/ASM call chain; experimental products/reducers are inlined into timed jobs. | API boundaries, optimizer visibility, temporary placement and arithmetic differ together. Chain gains are complete implementation comparisons, not a pure carry-schedule or call-overhead ablation. |
| V16 Latency | **MEASURED:** chain API/row+serial median paired ratios are 1.11945 / 1.10725 for multiplication and 1.07853 / 1.18285 for square. Row median ns/op is 21.704 / 20.893 for multiply and 19.654 / 19.404 for square. | These are 1024-step region-average normalized costs, not individual-call percentiles or instruction latency. The varying square gain warrants further replication before any integration decision. |
| V17 Throughput | **MEASURED:** API chain/ILP4 ns/op ratios are 1.97305 / 1.97942 for multiply and 1.84166 / 2.00542 for square, each favoring ILP4 in 16/16 pairs. Symmetric-Comba/precomputed square instead has ratios 0.97528 / 0.93328; ILP4 wins only 4/16. | Four independent states do not guarantee a throughput gain. Generated ILP4 is a scalar lane loop on one core, not four threads or packed field arithmetic; no pure hardware-overlap quantity is measured. |
| V18 Instructions / PMU | **STATIC:** baseline ASM has ADCX/ADOX; experimental job bodies use scalar MULX/MUL and ADD/ADC/SBB, with no packed integer arithmetic. C++ reducers retain six multiply sites versus five in ASM, whose final overflow uses NEG/AND. | This is instruction-presence evidence, not a speed explanation or retired-instruction count. PMU remains unavailable under the recorded policy; no cycles, ports or energy claim. |
| V19 Cache / bandwidth | **DERIVED / MEASURED:** multiplication cycles an 8 KiB canonical RHS payload; raw reduction cycles 16 KiB of wide data; square reads neither RHS stream in its recurrence. Both series used the same complete corpus. | Payload is not residency, physical traffic or bandwidth. Raw reduction includes low-half XOR preparation and is not interchangeable with multiply/square work. |
| V20 Code size | **STATIC:** chain job symbols are 321 bytes for API multiply, 1,681 for row multiply and 1,585 for either Comba multiply. Square chain symbols are 298 / 1,363 / 1,593 respectively. Baseline arithmetic lives in additional callees; all jobs call serialization. | These are individual symbol extents, not transitive instruction-cache footprints. Smaller Comba symbols did not establish faster complete operations or lower cache cost. |
| V21 Portability | **MEASURED:** the same arithmetic corpus passed three configurations; duration qualification passed 120 synthetic cases / 6,550 assertions on exactly two, GCC and Clang. Full performance series use frozen GCC14 code. | Arithmetic success is not whole-Clang performance, another architecture's speed or unsupported-u128 qualification. Synthetic clocks are validation-only, never performance samples. |
| V22 Compiler | **STATIC / MEASURED:** GCC eliminated the intended arithmetic distinction between serial/precomputed reducer schedules and recognized row-square symmetry. Raw serial/precomputed ratios are 1.00271 / 1.04095 in chain1 and 1.02006 / 0.95682 in ILP4; precomputed wins 12/16 and 8/16 respectively. | No robust distinct reducer winner is established. Further schedule experiments must first demonstrate different emitted arithmetic, or explicitly be treated as code-placement/measurement controls. |
| V23 Policy / ties | **MEASURED:** all 400 later regions met their floor with actual operations and retained cumulative prefixes; all 25 extensions and 42 short calibration records remain. Synthetic failure exited 2, preserving 41 regions and 13 failure probes in each compiler test. | No selective rerun or best-sample choice. Within-region work-cap exhaustion/capped-final-batch handling is statically reviewed but not dynamically exercised by the tiny synthetic diagnostic. |
| V24 Transfer | **MEASURED / DERIVED:** API beats every Comba field candidate across both modes, 128/128 pairs; the positive result is specifically row+serial chain multiply/square. Raw reduction has its own recurrence and comparison. | Retain row chains as candidates and the API's ILP4 result as a control. No production, scalar, point or signature integration; field work, inversion and cross-compiler replication remain unfinished. |
| V25 Prior art / novelty | **DERIVED:** P5 reuses pseudo-Mersenne reduction and established product identities while examining source schedules and their compiled behavior. | No new mathematics, novel algorithm, world record, CT guarantee or whole-engine speedup. External novelty classification remains **PENDING IMPORT**. |

## Research disposition

Keep the row-chain candidate, the existing API for the observed ILP4 workloads,
and the negative Comba results. Do not promote the serial/precomputed timing
differences as an arithmetic discovery: the compiler made their reducer
arithmetic equivalent in this binary. Next useful gates are whole-compiler
replication, controlled API-boundary comparisons and genuinely distinct emitted
schedules, with all costs and losses retained. None is implemented here.

Only this new measured lens map was written. Preregistered documents, raw
evidence and production code are unchanged. Worker Source Graph remains
unexposed under the recorded NeedFix; exact manager-supplied targets were used,
without manager-role impersonation. Owner Task MCP suspension continues.
Stop at manager review.
