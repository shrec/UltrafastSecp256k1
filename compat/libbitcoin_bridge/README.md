# UltrafastSecp256k1 ⇄ libbitcoin acceleration bridge

A small **shim controller** that hands libbitcoin (and any other node) two
GPU-accelerated batch capabilities behind one tiny C ABI, with a **mandatory CPU
fallback**. Pure C / C++ — **no FFI, no language bindings**.

> Status: **CPU path built & tested; GPU dispatch compile-verified.**
> The header [`ufsecp_libbitcoin.h`](include/ufsecp_libbitcoin.h) is the
> entry-point contract; [`src/ufsecp_libbitcoin.cpp`](src/ufsecp_libbitcoin.cpp)
> implements it. The CPU consensus-reference path passes the correctness test
> ([`tests/test_lbtc_bridge.cpp`](tests/test_lbtc_bridge.cpp): all-valid,
> single-corruption identification, opaque-key stride, ECDSA + Schnorr, empty
> batch) plus the in-place collect contract
> ([`tests/test_lbtc_collect.cpp`](tests/test_lbtc_collect.cpp), incl. a
> multi-chunk variant). GPU dispatch (`-DUFSECP_LBTC_WITH_GPU=ON`) compiles against the engine
> GPU ABI; runtime GPU verification + the differential consensus gate + the
> libbitcoin branch are the next steps. The contract is a proposal — corrections
> expected after libbitcoin review.

## Why this exists

From the design discussion with the libbitcoin maintainer (evoskuil):

> *"The batch processing for SP and for script validation are the two areas that
> interest us. Honestly it wouldn't be worth changing the dependency that we
> currently have on libsecp256k1 without these."*

So the bridge targets exactly those two:

1. **Script-signature batch verification** (the big win). Signatures are ~95% of
   script-validation cost. The node extracts `sig / key / sighash` triples from
   scripts (`CHECKSIG`, `CHECKMULTISIG`, Taproot `CHECKSIG` / `CHECKSIGADD`),
   packs them into one big array, and forwards the array in a single call —
   getting an array of per-row results back. Target: **IBD / historical block
   validation** (batches of tens of thousands of blocks), **not** mempool
   latency.

2. **BIP-352 Silent Payments scan** (bonus, for the Electrum/server side).
   GPU-accelerated ECDH scan reusing the existing engine SP pipeline.

ECDSA and Schnorr are **separate calls** — uniform-sized data, stacked
independently (one GPU stream each, or one card each).

## The entry-point contract

### One unified table per call

Each row is `[ signature record ][ optional opaque correlation key ]`:

| Kind    | Record (verified)                         | Bytes |
|---------|-------------------------------------------|-------|
| ECDSA   | `32 msg ‖ 33 pubkey ‖ 64 sig`             | 129   |
| Schnorr | `32 msg ‖ 32 xonly ‖ 64 sig`              | 128   |

> Note the deliberate field-order difference: the message is the **first** field
> for ECDSA and the **second** field for Schnorr. This matches the engine's
> existing `ufsecp_ecdsa_batch_verify` / `ufsecp_schnorr_batch_verify` packed
> formats.

The trailing **opaque key** is the caller's own tag (e.g. 3-byte block id,
4-byte tx id). The bridge **never interprets** it — it is carried only so an
invalid row can be mapped back to its block/tx **without a second side table**.

- Each on-wire row is `[ record | key_size bytes ]` (`key_size` may be `0`).
- **Sizing contract — pass the record COUNT and the KEY SIZE, nothing else.**
  Two of `{count, key_size, buffer size}` are sufficient; the call passes the
  **count** and the **key size**, and the buffer size is *implied* —
  `count * (RECORD + key_size)`. There is deliberately **no buffer-size argument**:
  as evoskuil put it, the size is redundant with `count + key_size`, and providing
  it would only add an unnecessary error condition (a `byte*, size` API can
  mismatch; `count + key_size` cannot).

  ```c
  // C ABI — rows = contiguous [record | key] buffer; n = records; ks = key bytes/row;
  // results = n bytes (1=valid/0=invalid). Returns void: no error code, no buffer size,
  // no invalid-index outputs — the caller maps failures from results[].
  void ufsecp_lbtc_verify_ecdsa(ufsecp_lbtc_ctrl* ctrl,
                                const uint8_t* rows, size_t n,
                                size_t key_size, uint8_t* results);
  ```

  This is exactly the shape evoskuil's libbitcoin integration calls. With a packed
  `secp256k1::ecdsa::triple` (`#pragma pack(1)` of `{ hash_digest, ec_compressed,
  ec_signature, token }`, where `token = data_array<3>`), the key size is the
  compile-time `sizeof(token)`:

  ```cpp
  static thread_local ufsecp::lbtc::Controller context{ UFSECP_LBTC_AUTO };
  if (!context.ok()) std::abort();                   // only recoverable failure (no backend)
  const auto count   = batch.size();
  std::vector<uint8_t> results(count);               // zero-init → fail-closed on a no-op call
  const auto in      = pointer_cast<uint8_t>(batch.data());
  const auto out     = results.data();
  constexpr auto id_size = array_count<decltype(triple::identifier)>;   // = 3
  ufsecp_lbtc_verify_ecdsa(context.get(), in, count, id_size, out);     // void; 5 args
  // map failures (results[i] == 0) back to your tokens:
  for (size_t row = 0; row < results.size(); ++row)
      if (!to_bool(results[row])) failed.push_back(batch[row].identifier);
  ```

  **Typed-span convenience overload (C++20)** — if you prefer not to compute
  `id_size` yourself, pass the span and the bridge recovers both `count =
  batch.size()` and `key_size = sizeof(Row) - RECORD` from the element type:

  ```cpp
  std::span<const ecdsa::triple> batch = ...;
  std::vector<uint8_t> results(batch.size());
  ctrl.verify_ecdsa(batch, results.data());          // also void, results-only
  ```

  **Integration notes for the consumer:**
  - `count` + `key_size` fully determine the layout — the buffer is
    `count*(RECORD+key_size)` bytes by construction, so there is no size argument,
    no size mismatch, and no size error path.
  - `RECORD` is `129` for ECDSA, `128` for Schnorr; the message/sighash is the
    **first** field for both kinds (Schnorr is uniform with ECDSA, not xonly-first).
  - The correlation key is opaque to the bridge: it is carried, never interpreted,
    so `results[i]` maps straight back to your row `i`.
- The caller builds **one unified table**; the controller internally splits each
  row into the signature payload (→ verify) and the opaque tail (→ carried).

### Results returned directly

- The verify call returns **`void`**. `results[i]` — `1` valid / `0` invalid, one
  byte per row — is the only output. The caller maps a failing row back to its
  block/tx by reading the opaque tag carried in `rows[i]` (no second side table).
  Since honest IBD failures are rare, scanning `results[]` for the few zeros is cheap.
- There is **no error return** and **no invalid-index list**: the only recoverable
  failure (no usable backend) is reported at `ufsecp_lbtc_ctrl_create` /
  `Controller::ok()` time; a degenerate call (NULL/`n==0`) is a no-op that leaves
  `results` at the caller's zero-init (fail-closed = all invalid).

This is the design space evoskuil and Vano converged on: *"make entry point
function as you wish with your data and return types … inside that function i
will link gpu side."* The controller is that middleware entry point — and evoskuil
confirmed the final `void verify(ctrl, rows, count, key_size, results)` shape ideal.

## Consensus correctness

For block validation the GPU result is **consensus-bearing**. Correctness is
anchored on the CPU / libsecp256k1-equivalent reference:

- The CPU fallback (always present, never optional) is the reference.
- The GPU path must match the CPU path **bit-for-bit**, enforced by a
  differential test gate (GPU vs CPU vs libsecp256k1 shim).
- Structurally-invalid rows (`s >= n`, `R.x >= p`, off-curve pubkey) verify to
  `invalid`, never crash the batch.

## How it maps onto existing engine primitives

The controller is **dispatch + marshalling only** — it reuses primitives that
already exist and are tested:

| Capability        | GPU path                              | CPU fallback                          |
|-------------------|---------------------------------------|---------------------------------------|
| ECDSA batch       | `ufsecp_gpu_ecdsa_verify_batch`       | `ufsecp_ecdsa_batch_verify` (whole chunk; `n==1` per row to locate invalids) |
| Schnorr batch     | `ufsecp_gpu_schnorr_verify_batch`     | `ufsecp_schnorr_batch_verify` (reorder to engine record, then as above) |
| BIP-352 scan      | `ufsecp_gpu_bip352_scan_batch`        | engine CPU SP scan                    |

The GPU verify kernels already write **per-entry** results (one signature per
thread), so per-row pass/fail falls out for free. The CPU fallback verifies the
whole chunk in one `*_batch_verify` call; if that reports the chunk has an
invalid row, it re-runs each row as a single-entry verify to locate the exact
failures (honest IBD failures are rare, so the per-row pass is seldom taken).

## 4-table batching model (single + m-of-n)

libbitcoin accumulates signature tuples into **four** homogeneous tables so each
stacks independently at a uniform row size:

| table       | record (verified)        | tail (opaque)              | row size |
|-------------|--------------------------|----------------------------|---------:|
| `ecdsa`     | `EcdsaRecord` (129)      | `block-fk` (3)             | 132 |
| `schnorr`   | `SchnorrRecord` (128)    | `block-fk` (3)             | 131 |
| `multisig`  | `EcdsaRecord` (129)      | `m\|n`(1)+`group`(2)+`block-fk`(3) | 135 |
| `threshold` | `SchnorrRecord` (128)    | `m\|n`(1)+`group`(2)+`block-fk`(3) | 134 |

At the verify boundary the four tables collapse to **two verify kinds**: a
`multisig` row is one ECDSA signature, a `threshold` (tapscript CHECKSIG /
CHECKSIGADD) row is one Schnorr signature. The `m|n` + `group` bytes are
libbitcoin accounting metadata, carried in the opaque correlation tail and
**never interpreted** by the bridge — m-of-n satisfaction is the node's job,
downstream of the per-row verdict. So `multisig` verifies through the ECDSA path
and `threshold` through the Schnorr path, both with `key_size == 6`.

The canonical packed rows `ufsecp::lbtc::MultisigRow` (135) and `ThresholdRow`
(134) are the on-wire source of truth so a node's table forwards zero-copy:

```cpp
std::span<const MultisigRow> batch = ...;     // EcdsaRecord + m|n + group + block-fk
std::vector<uint8_t> results(batch.size());
ctrl.verify_multisig(batch, results.data());  // == verify_ecdsa, key_size=6
// or the rejected-id-list shape:
ctrl.collect_multisig(std::span<MultisigRow>(rows, n));  // valid tail zeroed
```

`verify_multisig` / `verify_threshold` (and the `collect_*` twins) are
intent-revealing aliases of the ECDSA / Schnorr calls — identical, gated
verification. Because the verify core reads only the leading record and walks
`stride = record + key_size`, the verdict is **independent of the tail width**:
the single (3-byte) and m-of-n (6-byte) tables verify the same signatures
identically (`tests/test_lbtc_multisig_threshold.cpp`).

## BIP-341 Taproot commitment batch — `verify_commitment`

Batches the key-path commitment check (`secp256k1_xonly_pubkey_tweak_add_check`):
for each item, accept iff `Q = lift_x(internal_x, even-y) + tweak*G` has
`x(Q) == tweaked_x` and `y-parity(Q) == parity`. Columns / SoA input:

```c
void ufsecp_lbtc_verify_commitment(ufsecp_lbtc_ctrl* ctrl,
    const uint8_t* internal_x32,  // n*32 x-only internal keys
    const uint8_t* tweak32,       // n*32 BIP-341 tweaks (each in [1, n))
    const uint8_t* tweaked_x32,   // n*32 claimed tweaked output keys
    const uint8_t* parity,        // n    claimed y-parity (0 even / 1 odd)
    size_t n, uint8_t* results);  // OUT n bytes, 1 valid / 0 invalid
```

All inputs are PUBLIC → variable-time. Each row is computed engine-native as one
2-term MSM (`Q = 1*P + tweak*G`, the compressed-point parse does the even-y lift),
so it is **per-row and EXACT** — no random-linear-combination, hence consensus-safe
with no aggregate randomness. The CPU path fans the independent rows across a
thread pool (≈ core-count speedup over the per-call path; ~7× on a 6P+4E box).
`tests/test_lbtc_commitment.cpp` cross-checks every verdict against the shim's
native `secp256k1_xonly_pubkey_tweak_add_check`.

**GPU version — `ufsecp_lbtc_commitment_batch_ok` (shipped).** Collapses the batch
to two device MSMs (`ufsecp_gpu_msm`): `Σrᵢ·Pᵢ + (Σrᵢ·tᵢ)·G == Σrᵢ·Qᵢ`. Returns a
single aggregate verdict — `1` all valid / `0` some invalid / `-1` no GPU — so the
rare failure falls back to the per-row `verify_commitment` to locate it. Measured
~2.5 M checks/s (RTX-class). The weights `rᵢ` are **Fiat-Shamir-derived from a
SHA-256 over the whole batch** (`e = H(rows)`, `rᵢ = H(e‖i) mod n`), so a crafted
block cannot force a false cancellation (a constant `r` would be forgeable — the
`test_lbtc_commitment.cpp` GPU cases assert corrupted batches return `0`).

```c
int ok = ufsecp_lbtc_commitment_batch_ok(ctrl, internal_x, tweak, tweaked_x, parity, n);
if (ok == 1)      accept_all();              // GPU fast-path: whole batch valid
else if (ok == 0) verify_commitment(...);    // locate the failure(s) per-row
else /* -1 */     verify_commitment(...);     // no GPU -> CPU per-row
```

**Single-buffer (AoS) — one mmap'd table, one pointer, zero repack.** The 4 columns
collapse to **3** by storing the tweaked key COMPRESSED (the `0x02/0x03` prefix folds
in the parity — no separate parity column), laid out as one packed record:

```
CommitmentRow (97B):  internal_x[32] | tweak[32] | tweaked[33, compressed]
   buffer = n × (record + optional tail)   ← your mmap'd table, verbatim
```
```c
// CPU: each record read IN PLACE at rows + i*stride (zero copy), threaded.
ufsecp_lbtc_verify_commitment_rows(ctrl, rows, n, /*stride=*/sizeof(Row), results);
// GPU RLC over the same buffer (Q_i read straight from the compressed field):
int ok = ufsecp_lbtc_commitment_batch_ok_rows(ctrl, rows, n, stride);  // 1/0/-1
```
The CPU path is genuinely copy-free (strided in-place reads); the GPU path's H2D
marshal is the same one every GPU dispatch pays (independent of AoS vs SoA). `stride`
may exceed 97 to carry a correlation tail the bridge ignores. The columns (SoA)
variants stay available for one-stream-per-field callers.

Remaining follow-up: a **per-item GPU kernel** (one thread per check) reading the
strided 97-byte records on-device (no host de-interleave) for per-row GPU verdicts
at a higher ceiling (~Schnorr-class, ~5 M/s).

## Collect (in-place) verify — `*_collect`

A second output shape, requested by evoskuil for the rejected-id-list use case.
Same verification, but **no `results[]` array**: the per-row verdict is collapsed
**into that row's own key cell** —

- **valid**   → the key cell is zeroed,
- **invalid** → the key cell is left exactly as supplied (the id survives).

The caller then walks the buffer once and **collects every row whose key cell is
still non-zero** — that set *is* the rejected list. No second side table, no
`results[]`.

```c
// rows is IN/OUT and MUST be writable; key_size MUST be > 0 (it is the result
// channel). n may be arbitrarily large — it is walked in internal chunks, so
// there is no size limit and no size error. Returns void.
void ufsecp_lbtc_verify_ecdsa_collect (ufsecp_lbtc_ctrl* ctrl,
                                       uint8_t* rows, size_t n, size_t key_size);
void ufsecp_lbtc_verify_schnorr_collect(ufsecp_lbtc_ctrl* ctrl,
                                       uint8_t* rows, size_t n, size_t key_size);
```

```cpp
// C++20 typed-span overload — note the span is over MUTABLE Row (the type system
// enforces that rows are writable, since collect writes the key cell in place):
std::span<ecdsa::triple> batch = ...;          // MUTABLE
ctrl.collect_ecdsa(batch);                      // key_size = sizeof(Row) - RECORD
for (auto& row : batch)
    if (key_cell_nonzero(row)) rejected.push_back(row.identifier);
```

This is the inverse of the results-based variant, which **never** touches the key
cell. They are mutually exclusive views of the same verification — pick one per
call site. `key_size == 0` is a no-op (no cell to write the verdict into →
fail-closed: every id survives = all rejected). The CPU and GPU paths reuse the
identical verify cores, so the collect rejected-set is consensus-identical to the
`results[]` channel (`tests/test_lbtc_consensus_diff.cpp` proves GPU==CPU==libsecp
on the rejected-set; `tests/test_lbtc_collect.cpp` proves the in-place contract,
including a `-DUFSECP_LBTC_CHUNK_OVERRIDE=8` multi-chunk variant).

**GPU collect kernel.** On CUDA the collect path uses a *dedicated on-device
kernel* (`ufsecp_gpu_ecdsa_verify_collect` / `_schnorr_verify_collect`) — a
verbatim copy of the verify kernel whose only change is the output store (it
writes a 1-byte verdict into a device buffer instead of a `bool` result array, so
the host skips the result-scatter pass; the variable-width key-cell zeroing stays
host-side). OpenCL and Metal do not implement it (the `GpuBackend` default returns
`Unsupported`); the bridge then falls back automatically to the host-collapse path
(the existing `*_verify_batch` kernels + a host verdict write) and finally to the
CPU reference — all consensus-identical. The **results-based** verify ABI's GPU
path is left byte-for-byte unchanged (only the collect path uses the new kernel,
gated by `if constexpr`). A test/bench seam `-DUFSECP_LBTC_DISABLE_DEDICATED`
forces the host-collapse arm for A/B comparison.

## libbitcoin integration shape

libbitcoin-system wraps libsecp256k1 in
`src/crypto/secp256k1.cpp` (`<secp256k1.h>`, `<secp256k1_recovery.h>`,
`<secp256k1_schnorrsig.h>`) and has **no batch API today**. Two layers:

1. **Single-signature ops** — already covered by the drop-in
   `libsecp256k1_shim` (zero source changes; their existing wrapper just links
   our shim instead of stock libsecp256k1).
2. **Batch script-sig verify + SP scan** — *new* capability, added through this
   bridge controller. libbitcoin calls `ufsecp_lbtc_verify_ecdsa` /
   `_verify_schnorr` from its IBD signature-validation flow.

The integration is delivered to libbitcoin as a **separate branch / PR** (a
small plugin-style harness, mirroring the `bench_bip352` workflow used with
CraigRaw), so they can build, marshal their data into it, and we tune the GPU
side against their real pipeline.

## Build — libbitcoin profile (primary path)

The bridge is **compiled into the central C ABI library** (`libufsecp`), not shipped
as a separate library. A single flag activates the minimal node profile: drop-in
shim + libsecp contract (CT enforced, strict ABI) + the batch bridge, with tests,
benches, examples, Java and Ethereum turned off.

```bash
cmake -S . -B out/libbitcoin -G Ninja -DSECP256K1_BUILD_LIBBITCOIN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build out/libbitcoin --target ufsecp_static    # ufsecp_lbtc_* are in libufsecp
# add a GPU backend to enable GPU dispatch (UFSECP_LBTC_WITH_GPU is auto-defined):
#   -DSECP256K1_BUILD_CUDA=ON   (or -DSECP256K1_BUILD_OPENCL=ON)
```

A standalone `CMakeLists.txt` is also provided for out-of-tree dev/testing (builds a
small `ufsecp_lbtc_bridge` lib + the test + example against an installed/prebuilt
`libufsecp`); the profile path above is what libbitcoin ships.

## Files

```
compat/libbitcoin_bridge/
  include/ufsecp_libbitcoin.h   ← entry-point contract (public header)
  src/ufsecp_libbitcoin.cpp     ← controller (CPU fallback + GPU dispatch)
  tests/test_lbtc_bridge.cpp    ← correctness test (CPU path verified)
  tests/test_lbtc_collect.cpp   ← in-place collect contract (+ smallchunk variant)
  tests/test_lbtc_multisig_threshold.cpp ← 4-table model: multisig/threshold verify+collect
  tests/test_lbtc_commitment.cpp ← BIP-341 Taproot commitment batch (vs shim tweak_add_check)
  tests/test_lbtc_consensus_diff.cpp ← GPU==CPU==libsecp differential (results + collect)
  example/lbtc_batch_demo.cpp   ← skeleton harness for libbitcoin
  CMakeLists.txt                ← standalone dev/test build
  README.md
```

Wired into the engine build via `SECP256K1_BUILD_LIBBITCOIN` in the top-level
`CMakeLists.txt` and `include/ufsecp/CMakeLists.txt`.
