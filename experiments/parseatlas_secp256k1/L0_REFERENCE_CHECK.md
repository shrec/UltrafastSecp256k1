# L0 reference check: `mulhi64` fallback counterexample

Classification: **OBSERVED**, one fixed input across three actual probe builds.
This is a standalone diagnostic, not a 25-world candidate or a record validated
by `schemas/research_contract.schema.json`. No production fix was made.

At pinned source commit `fef231d4e4173bd016fb2a3a1eff67087396a203`, the existing
schoolbook fallback in `src/cpu/include/secp256k1/detail/arith64.hpp` returns an
incorrect upper word for `a = b = UINT64_MAX`. These are valid operands for the
header's documented unsigned 64-by-64 multiplication contract.

| Diagnostic configuration | Actual high word | Bigint comparison |
| --- | --- | --- |
| Native GCC, `__SIZEOF_INT128__=16` | `fffffffffffffffe` | Match |
| Only `SECP256K1_NO_INT128` defined; `__SIZEOF_INT128__=16` | `fffffffffffffffe` | Match |
| Forced fallback, `-U__SIZEOF_INT128__ -DSECP256K1_NO_INT128` | `fffffffefffffffe` | Mismatch |

The independent Python bigint product is
`fffffffffffffffe0000000000000001`; its upper word is `fffffffffffffffe`.
For this input, exact `mid = 0x1fffffffd00000000` needs 65 bits. Its `uint64_t`
storage loses `2^64` before the right shift, so the returned high word is too
small by `2^32`. This is an exact explanation of the counterexample, not a
generalized performance result or a proof that another branch is correct.

Deployment impact: **NOT_ESTABLISHED**. The controls show the branch selected by
these actual GCC translation units. They do not establish which branch any
production, MSVC, or embedded build selects. `SECP256K1_NO_INT128` alone did not
select the `mulhi64` fallback on this compiler. No novelty, security-proof,
constant-time, speed, or new-optimization claim is made.

The divergent fallback cannot be the sole correctness oracle. Keep frozen-engine
byte comparisons **and** an independent bigint oracle. If they disagree, record
the baseline defect explicitly; matching its incorrect output is not evidence
of mathematical correctness. This diagnostic neither silently changes the
research contract nor establishes a replacement baseline.

## Evidence and reproduction

Measured on 2026-09-05 with GCC
`g++ (Ubuntu 14.2.0-4ubuntu2~24.04.1) 14.2.0` and Python `3.12.3`.
The [raw record](data/l0_mulhi64_reference.json) contains commands, macros,
outputs, and exit codes. SHA-256 values:

- Header (both working copy and pinned commit):
  `cc95a42137df602eaee3a560993b8c59c091e017364a0345be486ba012de205d`.
- [Exact-header probe](probes/mulhi64_reference.cpp), byte-identical to the initial
  temporary diagnostic:
  `d9965d96b6817aa8ca688feb2d6aecacd50565e34b5a86c4bba121cc5e4bb995`.

Run from the repository root. The probe includes the real header without
changing it or copying its arithmetic. Macros affect only these test translation
units. Inputs are fixed, so no random seed is needed. The default execution
sandbox initially failed before compiler startup with `bwrap: loopback: Failed
RTM_NEWADDR: Operation not permitted`; the recorded builds and runs then
succeeded through approved escalation, with writes only to diagnostic artifacts.

```sh
git rev-parse HEAD
git show fef231d4e4173bd016fb2a3a1eff67087396a203:src/cpu/include/secp256k1/detail/arith64.hpp | sha256sum
sha256sum src/cpu/include/secp256k1/detail/arith64.hpp experiments/parseatlas_secp256k1/probes/mulhi64_reference.cpp
g++ --version
python3 --version

probe_dir=$(mktemp -d /tmp/parseatlas-mulhi64.XXXXXX)
g++ -std=c++17 -O2 -Wall -Wextra -Isrc/cpu/include \
  experiments/parseatlas_secp256k1/probes/mulhi64_reference.cpp \
  -o "$probe_dir/native"
g++ -std=c++17 -O2 -Wall -Wextra -DSECP256K1_NO_INT128 -Isrc/cpu/include \
  experiments/parseatlas_secp256k1/probes/mulhi64_reference.cpp \
  -o "$probe_dir/no_int128_define_only"
g++ -std=c++17 -O2 -Wall -Wextra -U__SIZEOF_INT128__ -DSECP256K1_NO_INT128 \
  -Isrc/cpu/include experiments/parseatlas_secp256k1/probes/mulhi64_reference.cpp \
  -o "$probe_dir/fallback"

"$probe_dir/native" ffffffffffffffff ffffffffffffffff
"$probe_dir/no_int128_define_only" ffffffffffffffff ffffffffffffffff
"$probe_dir/fallback" ffffffffffffffff ffffffffffffffff
python3 -c 'a = b = (1 << 64) - 1; print(f"{(a*b)>>64:016x}")'

python3 - "$probe_dir" <<'PY'
import subprocess
import sys

binary_dir = sys.argv[1]
a = b = (1 << 64) - 1
expected = (a * b) >> 64
mismatch = False
for variant in ("native", "no_int128_define_only", "fallback"):
    output = subprocess.check_output(
        [f"{binary_dir}/{variant}", f"{a:x}", f"{b:x}"], text=True
    )
    actual = int(output.rsplit("high=", 1)[1].strip(), 16)
    matches = actual == expected
    print(f"{variant}: expected={expected:016x} actual={actual:016x} matches={matches}")
    mismatch |= not matches
sys.exit(1 if mismatch else 0)
PY
```

The final comparison **intentionally exits 1 at the recorded header hash**,
because the forced fallback disagrees with bigint. Do not turn this into a
passing arithmetic test with `|| true`. With `set -e`, the nonzero result stops
the shell as usual; capture it explicitly if continuing a diagnostic session.
The standalone C++ probe exits 0 after printing a result, even if the arithmetic
is wrong; only the independent comparison judges agreement. The probe is not
registered as an expected-green suite test.
