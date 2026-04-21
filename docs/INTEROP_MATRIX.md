# INTEROP_MATRIX.md — UltrafastSecp256k1 Cross-Implementation Interop

> Version: 1.0 — 2026-04-21
> Closes CAAS gap **G-4**.
>
> This matrix records which cross-implementation interop surfaces
> UltrafastSecp256k1 is exercised against, what kind of cross-check
> happens, and where the evidence lives. An external auditor can
> verify a claim of "matches reference X" by running the row's test
> command without source archeology.

## 1. Scope

Interop here means: **identical-input → identical-output** between
UltrafastSecp256k1 and an independently-implemented reference, on
publicly-published vectors or live-generated random inputs.

Three flavours are tracked:

| Flavour | Definition |
|---------|------------|
| **Vector interop** | Both sides ingest a published test-vector file and produce identical output |
| **Live differential** | Both sides ingest random or fuzzed inputs and the outputs are byte-compared at runtime |
| **Wire interop** | Two implementations exchange messages over a real protocol (BIP-324, MuSig2 round protocol, FROST DKG) and the session completes |

## 2. Interop surfaces

### 2.1 Vector interop (published test vectors)

| Source | Algorithm | Vectors | Local test | Status |
|--------|-----------|---------|------------|--------|
| Wycheproof | ECDSA secp256k1 SHA-256 | `ecdsa_secp256k1_sha256_test.json` | `audit/test_wycheproof_ecdsa_secp256k1_sha256.cpp` | OK |
| Wycheproof | ECDSA secp256k1 SHA-256 P1363 | `ecdsa_secp256k1_sha256_p1363_test.json` | `audit/test_wycheproof_ecdsa_secp256k1_sha256_p1363.cpp` | OK |
| Wycheproof | ECDSA secp256k1 SHA-512 | `ecdsa_secp256k1_sha512_test.json` | `audit/test_wycheproof_ecdsa_secp256k1_sha512.cpp` | OK |
| Wycheproof | ECDSA secp256k1 SHA-512 P1363 | `ecdsa_secp256k1_sha512_p1363_test.json` | `audit/test_wycheproof_ecdsa_secp256k1_sha512_p1363.cpp` | OK |
| Wycheproof | ECDH secp256k1 | `ecdh_secp256k1_test.json` | `audit/test_wycheproof_ecdh.cpp` | OK |
| Wycheproof | HMAC-SHA256 | `hmac_sha256_test.json` | `audit/test_wycheproof_hmac_sha256.cpp` | OK |
| Wycheproof | HKDF-SHA256 | `hkdf_sha256_test.json` | `audit/test_wycheproof_hkdf_sha256.cpp` | OK |
| Wycheproof | ChaCha20-Poly1305 | `chacha20_poly1305_test.json` | `audit/test_wycheproof_chacha20_poly1305.cpp` | OK |
| Bitcoin Core | BIP-340 Schnorr | `bip-0340/test-vectors.csv` | `audit/test_bip340_vectors.cpp` | OK |
| BIP-32 spec | HD derivation | Spec vectors §3 | `audit/test_bip32_vectors.cpp` | OK |
| BIP-324 spec | Handshake/AEAD | Reference vectors | `audit/test_bip324_*` | OK |
| BIP-352 spec | Silent payments | Reference vectors | `audit/test_bip352_vectors.cpp` | OK |
| BIP-327 (MuSig2) | Key agg / sign / verify | Spec vectors | `audit/test_musig2_vectors.cpp` | OK |
| RFC 9591 | FROST | Appendix B vectors | `audit/test_frost_vectors.cpp` | OK |
| RFC 6979 | Deterministic ECDSA `k` | Appendix A.2 vectors | `audit/test_rfc6979_vectors.cpp` | OK |

### 2.2 Live differential (random/fuzz → byte-compare)

| Reference | Surface | Local test | Status |
|-----------|---------|------------|--------|
| `libsecp256k1` | ECDSA sign/verify (random keys + msgs) | `audit/test_exploit_differential_libsecp.cpp`, `audit/differential_test.cpp` | OK |
| `libsecp256k1` | Schnorr sign/verify (BIP-340) | `audit/differential_test.cpp` | OK |
| `libsecp256k1` | ECDH | `audit/differential_test.cpp` | OK |
| `libsecp256k1` | EC pubkey parse / serialize | `audit/differential_test.cpp` | OK |
| `libsecp256k1` (DER parser) | DER edge cases | `audit/test_exploit_der_parsing_differential.cpp` | OK |
| `libtomcrypt` | SHA-256, HMAC-SHA-256 (advisory) | `audit/test_differential_libtomcrypt.cpp` | OK (advisory) |
| `go-ethereum` (offline vectors) | Ethereum signature recovery (`ecrecover`) | `audit/test_exploit_ethereum_differential.cpp` | OK |
| Internal CT-vs-non-CT path | Same input, two code paths, identical output | `audit/test_ct_vs_classic_diff.cpp` | OK |

The differential suite executes ~1.3M assertions per nightly run
(`differential` audit section). Output is captured in
`build*/py_differential_report.json` per CAAS Stage 2.

### 2.3 Wire interop (real-protocol exchanges)

| Counterparty | Protocol | Test | Status |
|--------------|----------|------|--------|
| Self-against-self with two independent contexts | BIP-324 v2 transport handshake + AEAD | `audit/test_bip324_handshake.cpp` + `cpu/tests/test_bip324_standalone.cpp` | OK |
| Self-against-self | MuSig2 (BIP-327) full round protocol | `audit/test_musig2_round.cpp` | OK |
| Self-against-self | FROST (RFC 9591) DKG + sign | `audit/test_frost_dkg.cpp`, `audit/test_frost_round.cpp` | OK |
| Bitcoin Core test vectors | BIP-324 handshake transcripts | `audit/test_bip324_kdf_vectors.cpp` | OK |
| OpenSSL libcrypto (3.x) | ECDSA secp256k1 sign+verify cross-check | `audit/test_exploit_differential_openssl.cpp` (advisory) | Active when OpenSSL present, advisory skip otherwise |

## 3. References that we do NOT (yet) interop against

Listed for transparency. Adding any of these requires a new row in §2
and a wired test in `unified_audit_runner.cpp`.

| Reference | Algorithm | Why not yet | Tracking |
|-----------|-----------|-------------|----------|
| BoringSSL `EC_KEY` | Random differential | Build wiring (no system package on most CI) | Future G-4 extension |
| WolfSSL `wc_ecc_*` | Random differential | Same | Future G-4 extension |
| NSS `SECKEY_*` | Random differential | Same | Future G-4 extension |
| Rust `k256` / `secp256k1` | Random differential | Cross-ecosystem; needs FFI bridge | Future G-4 extension |
| Go `btcd/btcec` | Random differential | Cross-ecosystem | Future G-4 extension |
| `bitcoin-core/secp256k1` MuSig2 module | Wire interop | Counterparty harness needs build wiring | Future G-4 extension |
| `frost-dalek` (RFC 9591) | Wire interop | Cross-ecosystem | Future G-4 extension |

These are intentionally not in §2 because we will not list a row we
cannot back with a runnable test. Each future addition lands as a
separate `feat(interop): add <reference>` commit with an
`audit/test_interop_<ref>.cpp` and a `wycheproof_*` or live-diff row.

## 4. Verification commands

For an external auditor to reproduce every §2 row:

```bash
# Build with interop deps
cmake -B build_interop -S . -G Ninja \
  -DSECP256K1_BUILD_INTEROP=ON \
  -DSECP256K1_FETCH_LIBSECP256K1=ON \
  -DSECP256K1_FETCH_LIBTOMCRYPT=ON
cmake --build build_interop --target unified_audit_runner

# Run all interop-bearing audit modules in one shot
./build_interop/audit/unified_audit_runner --section differential
./build_interop/audit/unified_audit_runner --section standard_vectors
./build_interop/audit/unified_audit_runner --section exploit_poc \
  --filter "differential|wycheproof|interop"

# Or run each test individually
ctest --test-dir build_interop -R 'wycheproof|differential|interop' -V
```

The CAAS pipeline runs the same set in `caas-audit.yml` Stage 2 with
JSON output captured to `build_interop/py_differential_report.json`
and `build_interop/py_wycheproof_report.json`.

## 5. Failure semantics

A single mismatch in any §2 row is a hard CAAS failure. The pipeline
does not allow:

- Skipping a Wycheproof tcId (the runner refuses unknown skip flags).
- Marking a `libsecp256k1` differential mismatch as advisory.
- Down-grading a wire-interop fail to warning.

Historical mismatches are recorded in `docs/AUDIT_CHANGELOG.md` with
a closing commit hash. The Stark Bank `r∈[n,p−1]` class (RR-004) is
the most recent example, closed 2026-04-03 (`ea8cfb3c`).

## 6. Change discipline

Adding an interop reference requires, in the same commit:

1. Row in §2 (or §3 promotion to §2).
2. Wired audit module under `audit/test_*.cpp` registered in
   `unified_audit_runner.cpp`.
3. CMake option to fetch / link the reference (default OFF unless the
   reference is already a build dep).
4. JSON output captured by CAAS Stage 2.
5. Update of this document's last-updated date.
