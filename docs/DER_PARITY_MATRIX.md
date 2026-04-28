# DER Parsing Parity Matrix

**Last updated:** 2026-04-28 | Scope: `ufsecp_ecdsa_sig_from_der` / `ufsecp_ecdsa_sig_to_der`

This document maps DER/ASN.1 edge cases for ECDSA signature parsing to:
- Bitcoin consensus rule (BIP-66 strict DER, BIP-62 low-S)
- libsecp256k1 reference behavior
- UltrafastSecp256k1 behavior
- Test that covers the case

All tests are in `audit/test_fuzz_parsers.cpp`, `audit/test_wycheproof_ecdsa.cpp`,
and `audit/test_exploit_der_parsing_differential.cpp`.

---

## Structural Validity (SEQUENCE framing)

| Input | BIP-66 | libsecp256k1 | UF | Test |
|-------|--------|--------------|-----|------|
| Empty buffer (len=0) | ❌ reject | reject | reject | `test_fuzz_parsers` — zero-length rejected |
| Wrong outer tag (`0x31` instead of `0x30`) | ❌ reject | reject | reject | `test_fuzz_parsers` — wrong SEQUENCE tag rejected |
| SEQUENCE length > remaining bytes | ❌ reject | reject | reject | `test_fuzz_parsers` — length overflow rejected |
| Trailing bytes after valid SEQUENCE | ❌ reject | reject | reject | `test_fuzz_parsers` — trailing bytes rejected |
| BER long-form length (`0x81 len`) | ❌ reject | reject | reject | `test_fuzz_parsers` — adversarial inputs |
| Empty SEQUENCE (`0x30 0x00`) | ❌ reject | reject | reject | Structurally no r/s |

## INTEGER Encoding (R and S components)

| Input | BIP-66 | libsecp256k1 | UF | Test |
|-------|--------|--------------|-----|------|
| R tag wrong (`0x03` instead of `0x02`) | ❌ reject | reject | reject | `test_fuzz_parsers` — bad R tag rejected |
| S tag wrong (`0x03` instead of `0x02`) | ❌ reject | reject | reject | `test_fuzz_parsers` — bad S tag rejected |
| R length = 0 | ❌ reject | reject | reject | `test_fuzz_parsers` — zero R length rejected |
| R or S > 32 bytes of scalar data | ❌ reject | reject | reject | `test_fuzz_parsers` — oversized R rejected |
| R or S with unnecessary leading `0x00` pad for non-negative value | ❌ reject | reject | reject | G.10 / `test_wycheproof_ecdsa` |
| R or S missing required `0x00` pad (high bit set, would be negative) | ❌ reject | reject | reject | Wycheproof vectors |
| R = 0 (scalar == 0) | ❌ reject | reject | reject | `test_exploit_der_parsing_differential` test 1 |
| S = 0 (scalar == 0) | ❌ reject | reject | reject | `test_exploit_der_parsing_differential` test 2 |

## Scalar Bounds (r, s vs. group order n)

| Input | Bitcoin | libsecp256k1 | UF | Test |
|-------|---------|--------------|-----|------|
| r ≥ n (group order) | ❌ reject | reject | reject | `test_exploit_der_parsing_differential` test 3 |
| s ≥ n (group order) | ❌ reject | reject | reject | `test_exploit_der_parsing_differential` test 4 |
| r = n−1 (maximum valid r) | ✅ allow (structurally) | allow | allow | `test_exploit_der_parsing_differential` test 5 |
| High-S (s > n/2) — parse | ✅ allow | allow | allow | `test_exploit_der_parsing_differential` test 6 |
| High-S (s > n/2) — production | ❌ sign produces low-S | always low-S | always low-S | test 12: sign always produces low-S |
| r = 0xFF×32 (all bits set) | ❌ reject (r ≥ n) | reject | reject | `test_exploit_der_parsing_differential` test 7 |
| s = 0xFF×32 (all bits set) | ❌ reject (s ≥ n) | reject | reject | `test_exploit_der_parsing_differential` test 8 |

## Bit-Flip / Corruption Resistance

| Input | Expected | UF | Test |
|-------|----------|----|------|
| Single bit flip in r | fail verify | fail verify | `test_exploit_der_parsing_differential` test 9 |
| Single bit flip in s | fail verify | fail verify | `test_exploit_der_parsing_differential` test 10 |
| 580K random malformed DER blobs | no crash, reject | no crash, reject | `test_fuzz_parsers` random fuzz |
| 500+ Wycheproof ECDSA DER vectors | match reference | match reference | `test_wycheproof_ecdsa` |

## Round-Trip Properties

| Property | Expected | UF | Test |
|----------|----------|----|------|
| `sig_to_der(sig_from_der(x)) == x` | yes for valid | yes | `test_fuzz_parsers` round-trip |
| `sign()` → `sig_to_der()` → `sig_from_der()` → `verify()` passes | yes | yes | `test_exploit_der_parsing_differential` test 11 |
| `sign()` always produces DER-encodeable output | yes | yes | 50-message roundtrip suite |
| `sig_to_der()` output ≤ 72 bytes | yes | yes | `UFSECP_SIG_DER_MAX_LEN = 72` |

---

## Interpretation: Parity Status

**Full parity with libsecp256k1 strict-DER parser** for all tested cases.

Every input that libsecp256k1 rejects is also rejected by UltrafastSecp256k1,
and vice versa. The rejection is via `UFSECP_ERR_BAD_SIG` or `UFSECP_ERR_BAD_INPUT`
with output buffers left untouched (no partial writes).

**References:**
- BIP-66: <https://github.com/bitcoin/bips/blob/master/bip-0066.mediawiki>
- BIP-62: <https://github.com/bitcoin/bips/blob/master/bip-0062.mediawiki>
- Wycheproof ECDSA: `audit/test_wycheproof_ecdsa.cpp`
- Differential evidence: `docs/BITCOIN_CORE_BACKEND_EVIDENCE.md §2`
