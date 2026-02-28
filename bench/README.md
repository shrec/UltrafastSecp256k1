# bench_compare -- Apples-to-Apples: UltrafastSecp256k1 vs libsecp256k1

Reproducible, single-command benchmark comparing **UltrafastSecp256k1** against
**bitcoin-core/libsecp256k1 v0.6.0** under identical conditions.

Both libraries run **in the same process, on the same thread, with the same
dataset, compiler, flags, and CPU core**.

---

## Quick Start

```bash
# Linux (GCC / Clang)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSECP256K1_BUILD_BENCH_COMPARE=ON
cmake --build build -j$(nproc)
./build/bench/bench_compare --json=report.json

# Windows (MSVC / clang-cl)
cmake -S . -B build -DSECP256K1_BUILD_BENCH_COMPARE=ON
cmake --build build --config Release
.\build\bench\Release\bench_compare.exe --json=report.json
```

Or use the helper scripts:

```bash
# Linux
bash bench/scripts/build.sh          # configure + build
bash bench/scripts/run.sh            # run with sane defaults + JSON output
bash bench/scripts/run.sh --pin 2    # pin to core 2

# Windows (PowerShell)
.\bench\scripts\build.ps1
.\bench\scripts\run.ps1
.\bench\scripts\run.ps1 -PinCore 2
```

---

## What Is Measured

| Workload | Description |
|---|---|
| `ecdsa_verify_bytes` | Full ECDSA verify: parse pubkey + parse DER/compact sig + verify, per call |
| `ecdsa_verify_preparsed` | ECDSA verify with pre-parsed pubkey & signature (measures core math only) |
| `schnorr_verify_bytes` | Full BIP-340 Schnorr verify: parse x-only pubkey + verify |
| `schnorr_verify_preparsed` | Schnorr verify with pre-parsed (lifted) pubkey (skips sqrt) |
| `pubkey_create` | Scalar multiplication k*G  (fixed-base, precomputed tables) |
| `ecdh` | ECDH shared secret computation |

---

## How Fairness Is Ensured

1. **Same process, same thread.** Both providers share one `main()`. No separate
   processes that might land on different cores or NUMA nodes.

2. **CPU pinning.** `--pin-core=N` binds the thread to a single logical core
   (`SetThreadAffinityMask` on Windows, `sched_setaffinity` on Linux).
   Scripts default to core 2 to avoid scheduler noise on core 0.

3. **Same dataset.** A deterministic xoshiro256** PRNG (seeded with `--seed`)
   generates all secret keys, messages, and signatures. Both providers verify
   the exact same byte arrays.

4. **Same compiler + flags.** Both link into one executable; the same `-O3
   -DNDEBUG` (or MSVC `/O2`) apply to the harness. libsecp256k1 is fetched via
   CMake FetchContent and compiled alongside UF with matching build type.

5. **Warmup + measurement window.** Each workload gets a warmup phase
   (`--warmup`, default 250 ms) followed by a timed measurement phase
   (`--measure`, default 2000 ms). Only measurement-phase samples count.

6. **Per-operation timing.** Each individual verify/create call is timed
   separately. The report shows **median**, **P10**, **P90**, and **ops/sec**.

7. **Correctness gate.** Before any timing, 100 dataset items are verified for
   correctness (must return true). If the gate fails, the case is marked FAIL
   and skipped.

8. **No context tricks.** libsecp256k1 context is randomized with a fixed seed
   (deterministic). UF is stateless (no context).

---

## How Dataset Is Generated

Datasets are generated **at runtime** by the benchmark itself (no external files
needed). The deterministic PRNG guarantees identical data across runs.

For each ECDSA item:
- Random 32-byte message hash
- Random secret key -> compressed pubkey (33 bytes)
- ECDSA sign with UF -> low-S normalized -> compact (64 bytes) + DER

For each Schnorr item:
- Random 32-byte message
- Random secret key -> BIP-340 keypair -> x-only pubkey (32 bytes)
- Schnorr sign with UF -> 64-byte signature

For each pubkey_create item:
- Random secret key + expected compressed pubkey (cross-checked)

Default dataset size: 100,000. Override with `--n=<count>`.

---

## CLI Reference

```
bench_compare [OPTIONS]

Provider selection:
  --uf-only               Run UltrafastSecp256k1 only
  --libsecp-only          Run libsecp256k1 only

Case selection (default: ECDSA + Schnorr + pubkey_create):
  --case=ecdsa            ECDSA verify only
  --case=schnorr          Schnorr verify only
  --case=pubkey           pubkey_create only
  --case=ecdh             ECDH only
  --case=all              All cases

Dataset:
  --n=<count>             Dataset size (default: 100000)
  --seed=<uint64>         PRNG seed (default: 42)

Timing:
  --warmup=<ms>           Warmup duration (default: 250)
  --measure=<ms>          Measurement duration (default: 2000)
  --pin-core=<id>         Pin to CPU core (-1 = disabled)

Encoding:
  --ecdsa-sig=der         DER-encoded ECDSA signatures (default)
  --ecdsa-sig=compact     64-byte compact signatures

Output:
  --json=<path>           Write JSON report to file
  --help                  Show help
```

---

## Output Format

### Console (Markdown table)

```
=== Benchmark Report ===

CPU        : AMD Ryzen 9 7950X
OS         : Linux
Compiler   : clang 17.0.6
TSC freq   : 4501.2 MHz
Pinned core: 2

| Case                           | Provider     | Median(ns) | P10(ns)    | P90(ns)    | ops/sec      | OK?   |
|--------------------------------|--------------|------------|------------|------------|--------------|-------|
| ecdsa_verify_bytes             | UF           |     12345  |     11800  |     13200  |       81037  | OK    |
| ecdsa_verify_bytes             | libsecp256k1 |     18200  |     17500  |     19100  |       54945  | OK    |
| ...                            | ...          |       ...  |       ...  |       ...  |          ... | ...   |
```

### JSON (`report.json`)

```json
{
  "environment": {
    "cpu_model": "AMD Ryzen 9 7950X",
    "os": "Linux",
    "compiler": "clang 17.0.6",
    "tsc_mhz": 4501.2,
    "pinned_core": 2,
    "uf_commit": "f4cbe52",
    "libsecp_version": "v0.6.0"
  },
  "results": [
    {
      "case": "ecdsa_verify_bytes",
      "provider": "UltrafastSecp256k1",
      "median_ns": 12345.0,
      "p10_ns": 11800.0,
      "p90_ns": 13200.0,
      "ops_per_sec": 81037,
      "correctness": true
    }
  ]
}
```

---

## Expected Variance

On a **quiet** machine (no background load, CPU pinned, turbo stable):
- **P10/P90 spread**: < 15% of median is normal
- **Run-to-run median variance**: < 3% is excellent, < 8% is acceptable
- **Noisy CI runners** (GitHub Actions): expect 15--30% variance

Tips to reduce noise:
- Pin to a performance core (not E-core on Alder Lake+)
- Set CPU governor to `performance` on Linux: `cpupower frequency-set -g performance`
- Close background tasks, disable hyperthreading for cleanest numbers
- Use `--measure=5000` for longer measurement window

---

## Submit Your Results

We welcome benchmark results from different hardware/OS/compiler combinations.

1. Fork the repository
2. Run `bench_compare --json=report.json` on your machine
3. Upload `report.json` as a GitHub Gist
4. Open an issue using the **Benchmark Results** template
5. Paste the gist URL + console summary table

Or submit a PR adding your `report.json` to `bench/results/` (create the folder).

---

## Architecture

```
bench/
  CMakeLists.txt              # FetchContent for libsecp256k1 v0.6.0
  README.md                   # This file
  include/
    bench_affinity.h           # CPU pinning (Windows + Linux)
    bench_api.h                # IProvider interface (unified wrapper)
    bench_config.h             # CLI config struct
    bench_report.h             # JSON + Markdown output
    bench_rng.h                # Deterministic xoshiro256** PRNG
    bench_timer.h              # High-res clock + rdtsc + statistics
  src/
    main.cpp                   # Entry point, dataset gen, workload runner
    bench_config.cpp           # CLI parser
    providers/
      provider_uf.cpp          # UltrafastSecp256k1 IProvider (C++ API)
      provider_libsecp.cpp     # libsecp256k1 IProvider (C API)
  scripts/
    build.sh                   # Linux build helper
    build.ps1                  # Windows build helper
    run.sh                     # Linux run helper
    run.ps1                    # Windows run helper
```

---

## For Core Devs / Reviewers

> "I published a reproducible apples-to-apples harness comparing against
> libsecp256k1 pinned at v0.6.0 on identical hardware. Here are ECDSA
> verify/sec and the exact run command. If you can run it inside Hornet IBD
> / Bitcoin Core validation path, I'd love to see trace-level hotspots."

The harness is self-contained. One command, one binary, deterministic data,
JSON output. No external dependencies beyond CMake and a C++20 compiler.
