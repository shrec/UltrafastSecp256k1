# Announcement Draft -- Verification Transparency Snapshot v3.14

> Target: DelvingBitcoin / Stacker News
> Tone: Technical, measured, no hype

---

## Post Title

**UltrafastSecp256k1 v3.14 -- Verification Transparency Snapshot**

## Post Body

---

UltrafastSecp256k1 is a C++20 secp256k1 implementation with ECDSA (RFC 6979),
Schnorr (BIP-340), MuSig2, and FROST threshold signatures.

This is not an audit announcement. This is a verification data drop.

### What was verified

- **641,194 deterministic internal checks** (field, scalar, point, CT, security, integration) -- 0 failures
- **Differential tested** against bitcoin-core/libsecp256k1 v0.6.0: 7,860 cross-library checks, 0 mismatches. Nightly run: ~1.3M checks.
- **Standard vectors**: BIP-340 (15/15), RFC 6979 (6/6), BIP-32 TV1-TV5 (90/90)
- **Sanitizers**: ASan, UBSan, TSan, Valgrind -- 0 findings
- **Constant-time**: dudect Welch t-test on `ct::scalar_mul`, `ct::ecdsa_sign`, `ct::schnorr_sign`, `ct::field_inv` -- all pass (t < 4.5)
- **Fuzzing**: ~580K+ structured fuzz iterations (DER, Schnorr, pubkey, address, BIP-32, FFI) -- 0 crashes
- **14 CI workflows** enforcing the above on every commit

### Machine-verifiable artifacts in every release

- `SHA256SUMS.txt` -- binary checksums
- Cosign signatures (Sigstore keyless) -- `.sig` + `.pem`
- SLSA provenance attestation
- `sbom.cdx.json` -- CycloneDX 1.6 SBOM
- `selftest_report.json` -- structured selftest output (JSON, parseable)
- `verification_report.md` -- full transparency report

### What we do NOT claim

- Not externally audited
- Not formally verified (no ct-verif, no Vale)
- CT tested on x86-64 only; other uarch may differ
- MuSig2 and FROST are experimental (API may change)
- GPU backends are variable-time by design

### Reproduce everything

```
git clone https://github.com/shrec/UltrafastSecp256k1.git
cd UltrafastSecp256k1 && git checkout v3.14.0
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DSECP256K1_BUILD_CROSS_TESTS=ON \
  -DSECP256K1_BUILD_FUZZ_TESTS=ON \
  -DSECP256K1_BUILD_PROTOCOL_TESTS=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Full verification transparency report:  
https://github.com/shrec/UltrafastSecp256k1/blob/dev/docs/AUDIT_READINESS_REPORT_v1.md

Internal audit results (718 lines, per-check detail):  
https://github.com/shrec/UltrafastSecp256k1/blob/dev/docs/INTERNAL_AUDIT.md

---

*Verification artifacts published for independent review.*
