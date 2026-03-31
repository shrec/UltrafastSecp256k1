# Feature Maturity Table

**Last updated**: 2026-03-15 | **Version**: 3.50.0

Each feature is rated by: implementation status, threat model coverage, test vector sources, fuzz coverage, GPU support, secret-safety classification, and release tier.

## Tier Definitions

| Tier | Meaning |
|------|---------|
| **Production** | Fully audited, fuzzed, differential-tested, CT-verified (where applicable), standard test vectors pass |
| **Hardened** | Production + adversarial protocol tests + Wycheproof + fault injection |
| **Experimental** | Implemented and tested but limited external vectors or fewer audit passes |

## Status Legend

| Symbol | Meaning |
|--------|---------|
| Y | Present and verified |
| CT | Uses constant-time layer (secret-safe) |
| FAST | Uses variable-time layer (public data only) |
| P | Partial (some aspects covered) |
| - | Not applicable or not present |

---

## Core Cryptographic Primitives

| Feature | Status | Threat Model | Test Vectors | Fuzzed | GPU | Secret-Safe | Tier |
|---------|--------|-------------|-------------|--------|-----|-------------|------|
| Field arithmetic (5x52 / 4x64) | Y | Carry propagation, overflow | Fiat-Crypto, KAT | Y | Y (all 3) | N/A (internal) | Production |
| Field arithmetic (10x26) | Y | Carry propagation | KAT | Y | - | N/A (internal) | Production |
| Scalar arithmetic | Y | Overflow, reduction | RFC 6979, KAT | Y | Y (all 3) | N/A (internal) | Production |
| Point arithmetic | Y | Identity, doubling edge | Exhaustive (small curve) | Y | Y (all 3) | N/A (internal) | Production |
| GLV endomorphism | Y | Lambda/beta consistency | Differential (vs non-GLV) | Y | Y (all 3) | FAST | Production |
| Generator precompute (w8) | Y | Table correctness | selftest on init | - | Y (all 3) | N/A | Production |
| MSM / Pippenger | Y | Bucket overflow, edge scalar | Differential | Y | Y (all 3) | FAST | Production |

## Signature Schemes

| Feature | Status | Threat Model | Test Vectors | Fuzzed | GPU | Secret-Safe | Tier |
|---------|--------|-------------|-------------|--------|-----|-------------|------|
| ECDSA sign (RFC 6979) | Y | Nonce bias, k=0/n-1 | RFC 6979, Wycheproof | Y | Y (all 3) | CT | Hardened |
| ECDSA verify | Y | Malleable sigs, r/s bounds | Wycheproof (500+ vectors) | Y | Y (CUDA) | FAST | Hardened |
| ECDSA DER encode/decode | Y | Truncation, padding, length | Wycheproof, fuzz_parsers | Y | - | FAST | Hardened |
| ECDSA recovery | Y | Invalid recid | Wycheproof | Y | Y (CUDA) | FAST | Hardened |
| ECDSA batch verify | Y | Batch vs individual consistency | Randomized | Y | Y (CUDA) | FAST | Production |
| Schnorr sign (BIP-340) | Y | Nonce bias, aux randomness | BIP-340 official (15 vectors) | Y | Y (all 3) | CT | Hardened |
| Schnorr verify (BIP-340) | Y | Invalid R, e=0 | BIP-340 official | Y | Y (CUDA) | FAST | Hardened |
| Schnorr batch verify | Y | Batch vs individual consistency | Randomized | Y | Y (CUDA) | FAST | Production |

## Threshold / Multi-Party

| Feature | Status | Threat Model | Test Vectors | Fuzzed | GPU | Secret-Safe | Tier |
|---------|--------|-------------|-------------|--------|-----|-------------|------|
| MuSig2 (BIP-327) key agg | Y | Rogue-key attack | BIP-327 official | Y | - | FAST | Hardened |
| MuSig2 nonce gen/agg | Y | Nonce reuse, replay | Adversarial protocol | Y | - | CT | Hardened |
| MuSig2 partial sign/verify/agg | Y | Transcript mutation, signer ordering, malicious aggregator, abort/restart | Adversarial protocol (A.1-A.8) | Y | - | CT | Hardened |
| FROST keygen | Y | Below-threshold attack | Adversarial protocol | Y | - | CT | Hardened |
| FROST sign/verify/aggregate | Y | Malformed commitment, malicious coordinator, duplicate nonce, identity mismatch | Adversarial protocol (B.1-B.6) | Y | - | CT | Hardened |
| ECDSA adaptor signatures | Y | Invalid adaptor point, transcript mismatch, extraction misuse | Adversarial protocol (D.1-D.4) | Y | - | CT | Hardened |
| Schnorr adaptor signatures | Y | Round-trip, invalid point | Adversarial protocol (E.1-E.2) | Y | - | CT | Hardened |

## Key Derivation & Wallet

| Feature | Status | Threat Model | Test Vectors | Fuzzed | GPU | Secret-Safe | Tier |
|---------|--------|-------------|-------------|--------|-----|-------------|------|
| BIP-32 HD derivation | Y | Hardened vs normal, invalid child | BIP-32 official, adversarial | Y | - | CT | Hardened |
| BIP-39 mnemonic | Y | Invalid checksum, bad entropy | BIP-39 official | Y | - | CT | Production |
| ECDH (x-only + raw) | Y | Infinity point, twist | Wycheproof ECDH | Y | Y (all 3) | CT | Hardened |
| WIF encode/decode | Y | Invalid prefix, checksum | FFI fuzz | Y | - | CT | Production |
| Bitcoin addresses (P2PKH, P2WPKH, P2TR) | Y | Bad pubkey | FFI fuzz | Y | - | FAST | Production |
| Multi-coin wallet | Y | Coin-specific derivation | FFI fuzz | Y | - | CT | Production |
| BIP-352 Silent Payments | Y | Malicious scan key, wrong output | Adversarial protocol (C.1) | Y | - | CT | Hardened |

## Advanced Crypto

| Feature | Status | Threat Model | Test Vectors | Fuzzed | GPU | Secret-Safe | Tier |
|---------|--------|-------------|-------------|--------|-----|-------------|------|
| Pedersen commitments | Y | Blinding factor overflow | Randomized | Y | Y (all 3) | CT | Production |
| ZK knowledge proofs | Y | Malformed proof | Randomized | Y | Y (all 3) | CT | Production |
| ZK DLEQ proofs | Y | Wrong generators, malformed | Adversarial protocol (F.1-F.2) | Y | Y (all 3) | CT | Hardened |
| ZK range proofs (Bulletproofs) | Y | Out-of-range, verify sum | Randomized | Y | Y (all 3) | CT | Production |
| Taproot (BIP-341) | Y | Tweak overflow | Randomized | Y | - | CT | Production |
| ECIES encrypt/decrypt | Y | Truncated CT, wrong key, empty msg, 1MB payload, overlapping buffers | 85 regression vectors | Y | - | CT | Hardened |

## Hashing

| Feature | Status | Threat Model | Test Vectors | Fuzzed | GPU | Secret-Safe | Tier |
|---------|--------|-------------|-------------|--------|-----|-------------|------|
| SHA-256 | Y | - | NIST | - | Y (all 3) | N/A | Production |
| SHA-512 | Y | - | NIST | - | - | N/A | Production |
| Hash160 (RIPEMD160(SHA256)) | Y | - | Bitcoin-compatible | - | Y (all 3) | N/A | Production |
| Keccak-256 | Y | - | Ethereum-compatible | - | Y (all 3) | N/A | Production |
| Tagged hash (BIP-340) | Y | Midstate precomp | BIP-340 | - | Y (all 3) | N/A | Production |

## Ethereum (conditional: SECP256K1_BUILD_ETHEREUM)

| Feature | Status | Threat Model | Test Vectors | Fuzzed | GPU | Secret-Safe | Tier |
|---------|--------|-------------|-------------|--------|-----|-------------|------|
| ETH address (Keccak) | Y | EIP-55 checksum | Ethereum test suites | - | Y (all 3) | FAST | Production |
| ETH personal sign/verify | Y | EIP-191 prefix | Randomized | - | - | CT | Production |
| ETH ecrecover | Y | Invalid v/r/s | Randomized | - | - | FAST | Production |

## Constant-Time Infrastructure

| Feature | Status | Threat Model | Test Vectors | Fuzzed | GPU | Secret-Safe | Tier |
|---------|--------|-------------|-------------|--------|-----|-------------|------|
| CT field (mul, sqr, inv) | Y | Timing sidechannel | dudect (>4.5 t-test threshold) | - | Y (all 3) | CT | Hardened |
| CT scalar (mul, inv, cneg) | Y | Timing sidechannel | dudect + Valgrind taint | - | Y (all 3) | CT | Hardened |
| CT point (add, dbl, gen_mul) | Y | Timing sidechannel | dudect + differential | - | Y (all 3) | CT | Hardened |
| CT ECDSA sign | Y | Timing sidechannel | dudect + Cachegrind | - | Y (CUDA) | CT | Hardened |
| CT Schnorr sign | Y | Timing sidechannel | dudect + Cachegrind | - | Y (CUDA) | CT | Hardened |
| Lim-Lee comb (CT gen mul) | Y | Timing sidechannel | Differential vs fast | - | - | CT | Hardened |
| value_barrier() | Y | Compiler optimization | ASM inspection | - | - | CT | Production |

## C ABI / FFI

| Feature | Status | Threat Model | Test Vectors | Fuzzed | GPU | Secret-Safe | Tier |
|---------|--------|-------------|-------------|--------|-----|-------------|------|
| 97 ufsecp_* functions | Y | Null args, bad sizes, overlapping buffers | 286 FFI round-trip calls | Y | - | Auto-dispatch | Hardened |
| ABI gate (version + struct size) | Y | ABI break detection | abi_gate test | - | - | N/A | Production |
| Error codes (10 variants) | Y | All paths return correct code | FFI round-trip | - | - | N/A | Production |

## GPU C ABI Layer

| Feature | Status | Threat Model | Test Vectors | Fuzzed | GPU | Secret-Safe | Tier |
|---------|--------|-------------|-------------|--------|-----|-------------|------|
| 23 ufsecp_gpu_* functions | Y | NULL ctx/args, invalid backend, bad device, unsupported op | gpu_abi_gate + gpu_backend_matrix | - | CUDA/OpenCL/Metal | ECDH + BIP-324 decrypt SECRET-BEARING | Hardened |
| GPU error codes (100-106) | Y | Unknown code mapping | error_str coverage | - | N/A | N/A | Production |
| Backend discovery (3 backends) | Y | No device present | Enumerate + probe | - | CUDA/OpenCL/Metal | N/A | Production |
| generator_mul_batch | Y | NULL buffers, count=0 | 1*G == G equivalence | - | OpenCL+CUDA | N/A (public) | Hardened |
| Batch verify (ECDSA/Schnorr) | Y | - | - | - | CUDA only | N/A (public) | Stable |
| ECDH/Hash160/MSM batch | Y | - | - | - | Partial | ECDH secret-bearing | Stable |
| FROST partial verify GPU ABI | Y | Malformed commitments, unsupported backend | Protocol vectors + backend probing | - | Y (all 3) | N/A (public) | Production |
| ecrecover_batch GPU ABI | Y | Invalid recid, malformed compact sig | Backend probing + GPU/CPU equivalence | - | CUDA/OpenCL/Metal | N/A (public) | Stable |

---

## Cross-Reference

- Detailed audit runner modules: [AUDIT_TRACEABILITY.md](AUDIT_TRACEABILITY.md)
- GPU backend parity: [BACKEND_PARITY.md](BACKEND_PARITY.md)
- Test methodology matrix: [FEATURE_ASSURANCE_LEDGER.md](FEATURE_ASSURANCE_LEDGER.md)
- CT verification methodology: [CT_VERIFICATION.md](CT_VERIFICATION.md)
- Platform support: [COMPATIBILITY.md](COMPATIBILITY.md)
