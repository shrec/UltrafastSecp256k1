#pragma once
// ============================================================================
// unicode_nfkd.hpp — UTF-8 NFKD normalization for BIP-39 compliance
// ============================================================================
// BIP-39 spec: PBKDF2(password=NFKD(mnemonic), salt="mnemonic"+NFKD(passphrase))
//
// Platform dispatch:
//   Windows  → NormalizeString(NormalizationKD)
//   macOS    → CFStringNormalize(kCFStringNormalizationFormKD)
//   Other    → table-based decomposition (Unicode 15.0, no external deps)
// ============================================================================

#include <string>

namespace secp256k1 {

/// Returns the UTF-8 NFKD-normalized form of the input.
/// For ASCII-only strings (all bytes 0x00-0x7F): returns unchanged (fast path).
/// Covers Latin-1/Extended, Latin Extended-A/B, Greek Extended, compatibility
/// forms (fi/fl/ff/ffi/ffl), superscripts, fractions, and letterlike symbols.
std::string nfkd_normalize(const std::string& utf8);

} // namespace secp256k1
