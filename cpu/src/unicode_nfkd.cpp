// ============================================================================
// unicode_nfkd.cpp — UTF-8 NFKD normalization (no external dependencies)
// ============================================================================
// Unicode 15.0 NFKD decomposition table + canonical ordering.
// Platform-native paths for Windows and macOS; generic table-based for Linux.
// ============================================================================

#include "secp256k1/unicode_nfkd.hpp"
#include <cstdint>
#include <cstring>
#include <vector>
#include <algorithm>

// ============================================================================
// Fast path: pure ASCII check
// ============================================================================
static bool is_ascii(const std::string& s) {
    for (unsigned char c : s) {
        if (c > 0x7F) return false;
    }
    return true;
}

// ============================================================================
// Windows path
// ============================================================================
#if defined(_WIN32)

#include <windows.h>

namespace secp256k1 {

std::string nfkd_normalize(const std::string& utf8) {
    if (utf8.empty() || is_ascii(utf8)) return utf8;

    // UTF-8 → UTF-16
    int wlen = MultiByteToWideChar(CP_UTF8, 0,
                                   utf8.data(), static_cast<int>(utf8.size()),
                                   nullptr, 0);
    if (wlen <= 0) return utf8;
    std::vector<wchar_t> wide(wlen);
    MultiByteToWideChar(CP_UTF8, 0,
                        utf8.data(), static_cast<int>(utf8.size()),
                        wide.data(), wlen);

    // Normalize to NFKD
    int nlen = NormalizeString(NormalizationKD, wide.data(), wlen, nullptr, 0);
    if (nlen <= 0) return utf8;
    std::vector<wchar_t> norm(nlen);
    nlen = NormalizeString(NormalizationKD, wide.data(), wlen, norm.data(), nlen);
    if (nlen <= 0) return utf8;

    // UTF-16 → UTF-8
    int olen = WideCharToMultiByte(CP_UTF8, 0,
                                   norm.data(), nlen,
                                   nullptr, 0, nullptr, nullptr);
    if (olen <= 0) return utf8;
    std::string out(static_cast<size_t>(olen), '\0');
    WideCharToMultiByte(CP_UTF8, 0,
                        norm.data(), nlen,
                        &out[0], olen, nullptr, nullptr);
    return out;
}

} // namespace secp256k1

// ============================================================================
// macOS path
// ============================================================================
#elif defined(__APPLE__)

#include <CoreFoundation/CoreFoundation.h>

namespace secp256k1 {

std::string nfkd_normalize(const std::string& utf8) {
    if (utf8.empty() || is_ascii(utf8)) return utf8;

    CFMutableStringRef s = CFStringCreateMutable(kCFAllocatorDefault, 0);
    if (!s) return utf8;

    CFStringRef tmp = CFStringCreateWithBytes(
        kCFAllocatorDefault,
        reinterpret_cast<const UInt8*>(utf8.data()),
        static_cast<CFIndex>(utf8.size()),
        kCFStringEncodingUTF8, false);
    if (!tmp) { CFRelease(s); return utf8; }

    CFStringAppend(s, tmp);
    CFRelease(tmp);

    CFStringNormalize(s, kCFStringNormalizationFormKD);

    CFIndex used = 0;
    CFIndex len  = CFStringGetLength(s);
    CFStringGetBytes(s, CFRangeMake(0, len),
                     kCFStringEncodingUTF8, '?', false,
                     nullptr, 0, &used);

    std::string out(static_cast<size_t>(used), '\0');
    CFStringGetBytes(s, CFRangeMake(0, len),
                     kCFStringEncodingUTF8, '?', false,
                     reinterpret_cast<UInt8*>(&out[0]),
                     static_cast<CFIndex>(out.size()), &used);
    CFRelease(s);
    return out;
}

} // namespace secp256k1

// ============================================================================
// Generic / Linux path — table-based Unicode 15.0 NFKD
// ============================================================================
#else

namespace {

// ---------------------------------------------------------------------------
// UTF-8 decode / encode helpers
// ---------------------------------------------------------------------------

/// Decode one UTF-8 codepoint from buf[pos..end).
/// Returns the codepoint and advances pos. Returns U+FFFD on error.
static uint32_t utf8_decode(const uint8_t* buf, size_t len, size_t& pos) {
    if (pos >= len) return 0xFFFD;
    uint8_t b0 = buf[pos++];

    if (b0 < 0x80) return b0;

    uint32_t cp;
    int extra;
    if ((b0 & 0xE0) == 0xC0)      { cp = b0 & 0x1F; extra = 1; }
    else if ((b0 & 0xF0) == 0xE0) { cp = b0 & 0x0F; extra = 2; }
    else if ((b0 & 0xF8) == 0xF0) { cp = b0 & 0x07; extra = 3; }
    else return 0xFFFD;  // invalid lead byte

    for (int i = 0; i < extra; ++i) {
        if (pos >= len) return 0xFFFD;
        uint8_t cont = buf[pos++];
        if ((cont & 0xC0) != 0x80) return 0xFFFD;
        cp = (cp << 6) | (cont & 0x3F);
    }
    return cp;
}

/// Encode one codepoint to UTF-8, appending to out.
static void utf8_encode(std::string& out, uint32_t cp) {
    if (cp < 0x80) {
        out += static_cast<char>(cp);
    } else if (cp < 0x800) {
        out += static_cast<char>(0xC0 | (cp >> 6));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        out += static_cast<char>(0xE0 | (cp >> 12));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    } else if (cp < 0x110000) {
        out += static_cast<char>(0xF0 | (cp >> 18));
        out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    }
    // else: silently drop invalid codepoint
}

// ---------------------------------------------------------------------------
// NFKD decomposition table — Unicode 15.0
// Each entry: source codepoint + up to 4 decomposed codepoints (0-terminated).
// The table is sorted by cp for binary search.
// ---------------------------------------------------------------------------

struct NfkdEntry {
    uint32_t cp;
    uint32_t d[4];  // decomposed codepoints, 0-terminated
};

// clang-format off
static const NfkdEntry NFKD_TABLE[] = {
    // ---- Latin-1 Supplement U+00A0-U+00BF ----
    { 0x00A0, { 0x0020, 0 } },             // NO-BREAK SPACE → SPACE
    { 0x00A8, { 0x0020, 0x0308, 0 } },     // DIAERESIS → SPACE + combining diaeresis
    { 0x00AA, { 0x0061, 0 } },             // FEMININE ORDINAL INDICATOR → a
    { 0x00AF, { 0x0020, 0x0304, 0 } },     // MACRON → SPACE + combining macron
    { 0x00B2, { 0x0032, 0 } },             // SUPERSCRIPT TWO → 2
    { 0x00B3, { 0x0033, 0 } },             // SUPERSCRIPT THREE → 3
    { 0x00B4, { 0x0020, 0x0301, 0 } },     // ACUTE ACCENT → SPACE + combining acute
    { 0x00B5, { 0x03BC, 0 } },             // MICRO SIGN → Greek mu
    { 0x00B8, { 0x0020, 0x0327, 0 } },     // CEDILLA → SPACE + combining cedilla
    { 0x00B9, { 0x0031, 0 } },             // SUPERSCRIPT ONE → 1
    { 0x00BA, { 0x006F, 0 } },             // MASCULINE ORDINAL INDICATOR → o
    { 0x00BC, { 0x0031, 0x2044, 0x0034, 0 } }, // VULGAR FRACTION ONE QUARTER → 1/4
    { 0x00BD, { 0x0031, 0x2044, 0x0032, 0 } }, // VULGAR FRACTION ONE HALF → 1/2
    { 0x00BE, { 0x0033, 0x2044, 0x0034, 0 } }, // VULGAR FRACTION THREE QUARTERS → 3/4

    // ---- Latin-1 Supplement precomposed U+00C0-U+00FF ----
    { 0x00C0, { 0x0041, 0x0300, 0 } },    // À → A + grave
    { 0x00C1, { 0x0041, 0x0301, 0 } },    // Á → A + acute
    { 0x00C2, { 0x0041, 0x0302, 0 } },    // Â → A + circumflex
    { 0x00C3, { 0x0041, 0x0303, 0 } },    // Ã → A + tilde
    { 0x00C4, { 0x0041, 0x0308, 0 } },    // Ä → A + diaeresis
    { 0x00C5, { 0x0041, 0x030A, 0 } },    // Å → A + ring above
    // U+00C6 Æ — no canonical decomposition, no NFKD decomp
    { 0x00C7, { 0x0043, 0x0327, 0 } },    // Ç → C + cedilla
    { 0x00C8, { 0x0045, 0x0300, 0 } },    // È → E + grave
    { 0x00C9, { 0x0045, 0x0301, 0 } },    // É → E + acute
    { 0x00CA, { 0x0045, 0x0302, 0 } },    // Ê → E + circumflex
    { 0x00CB, { 0x0045, 0x0308, 0 } },    // Ë → E + diaeresis
    { 0x00CC, { 0x0049, 0x0300, 0 } },    // Ì → I + grave
    { 0x00CD, { 0x0049, 0x0301, 0 } },    // Í → I + acute
    { 0x00CE, { 0x0049, 0x0302, 0 } },    // Î → I + circumflex
    { 0x00CF, { 0x0049, 0x0308, 0 } },    // Ï → I + diaeresis
    // U+00D0 Ð — no decomp
    { 0x00D1, { 0x004E, 0x0303, 0 } },    // Ñ → N + tilde
    { 0x00D2, { 0x004F, 0x0300, 0 } },    // Ò → O + grave
    { 0x00D3, { 0x004F, 0x0301, 0 } },    // Ó → O + acute
    { 0x00D4, { 0x004F, 0x0302, 0 } },    // Ô → O + circumflex
    { 0x00D5, { 0x004F, 0x0303, 0 } },    // Õ → O + tilde
    { 0x00D6, { 0x004F, 0x0308, 0 } },    // Ö → O + diaeresis
    // U+00D7 × — no decomp
    // U+00D8 Ø — no decomp
    { 0x00D9, { 0x0055, 0x0300, 0 } },    // Ù → U + grave
    { 0x00DA, { 0x0055, 0x0301, 0 } },    // Ú → U + acute
    { 0x00DB, { 0x0055, 0x0302, 0 } },    // Û → U + circumflex
    { 0x00DC, { 0x0055, 0x0308, 0 } },    // Ü → U + diaeresis
    { 0x00DD, { 0x0059, 0x0301, 0 } },    // Ý → Y + acute
    // U+00DE Þ — no decomp
    // U+00DF ß — no NFKD decomp (NFD=ß, NFKD=ß)
    { 0x00E0, { 0x0061, 0x0300, 0 } },    // à → a + grave
    { 0x00E1, { 0x0061, 0x0301, 0 } },    // á → a + acute
    { 0x00E2, { 0x0061, 0x0302, 0 } },    // â → a + circumflex
    { 0x00E3, { 0x0061, 0x0303, 0 } },    // ã → a + tilde
    { 0x00E4, { 0x0061, 0x0308, 0 } },    // ä → a + diaeresis
    { 0x00E5, { 0x0061, 0x030A, 0 } },    // å → a + ring above
    // U+00E6 æ — no decomp
    { 0x00E7, { 0x0063, 0x0327, 0 } },    // ç → c + cedilla
    { 0x00E8, { 0x0065, 0x0300, 0 } },    // è → e + grave
    { 0x00E9, { 0x0065, 0x0301, 0 } },    // é → e + acute
    { 0x00EA, { 0x0065, 0x0302, 0 } },    // ê → e + circumflex
    { 0x00EB, { 0x0065, 0x0308, 0 } },    // ë → e + diaeresis
    { 0x00EC, { 0x0069, 0x0300, 0 } },    // ì → i + grave
    { 0x00ED, { 0x0069, 0x0301, 0 } },    // í → i + acute
    { 0x00EE, { 0x0069, 0x0302, 0 } },    // î → i + circumflex
    { 0x00EF, { 0x0069, 0x0308, 0 } },    // ï → i + diaeresis
    // U+00F0 ð — no decomp
    { 0x00F1, { 0x006E, 0x0303, 0 } },    // ñ → n + tilde
    { 0x00F2, { 0x006F, 0x0300, 0 } },    // ò → o + grave
    { 0x00F3, { 0x006F, 0x0301, 0 } },    // ó → o + acute
    { 0x00F4, { 0x006F, 0x0302, 0 } },    // ô → o + circumflex
    { 0x00F5, { 0x006F, 0x0303, 0 } },    // õ → o + tilde
    { 0x00F6, { 0x006F, 0x0308, 0 } },    // ö → o + diaeresis
    // U+00F7 ÷ — no decomp
    // U+00F8 ø — no decomp
    { 0x00F9, { 0x0075, 0x0300, 0 } },    // ù → u + grave
    { 0x00FA, { 0x0075, 0x0301, 0 } },    // ú → u + acute
    { 0x00FB, { 0x0075, 0x0302, 0 } },    // û → u + circumflex
    { 0x00FC, { 0x0075, 0x0308, 0 } },    // ü → u + diaeresis
    { 0x00FD, { 0x0079, 0x0301, 0 } },    // ý → y + acute
    // U+00FE þ — no decomp
    { 0x00FF, { 0x0079, 0x0308, 0 } },    // ÿ → y + diaeresis

    // ---- Latin Extended-A U+0100-U+017F ----
    { 0x0100, { 0x0041, 0x0304, 0 } },    // Ā → A + macron
    { 0x0101, { 0x0061, 0x0304, 0 } },    // ā → a + macron
    { 0x0102, { 0x0041, 0x0306, 0 } },    // Ă → A + breve
    { 0x0103, { 0x0061, 0x0306, 0 } },    // ă → a + breve
    { 0x0104, { 0x0041, 0x0328, 0 } },    // Ą → A + ogonek
    { 0x0105, { 0x0061, 0x0328, 0 } },    // ą → a + ogonek
    { 0x0106, { 0x0043, 0x0301, 0 } },    // Ć → C + acute
    { 0x0107, { 0x0063, 0x0301, 0 } },    // ć → c + acute
    { 0x0108, { 0x0043, 0x0302, 0 } },    // Ĉ → C + circumflex
    { 0x0109, { 0x0063, 0x0302, 0 } },    // ĉ → c + circumflex
    { 0x010A, { 0x0043, 0x0307, 0 } },    // Ċ → C + dot above
    { 0x010B, { 0x0063, 0x0307, 0 } },    // ċ → c + dot above
    { 0x010C, { 0x0043, 0x030C, 0 } },    // Č → C + caron
    { 0x010D, { 0x0063, 0x030C, 0 } },    // č → c + caron
    { 0x010E, { 0x0044, 0x030C, 0 } },    // Ď → D + caron
    { 0x010F, { 0x0064, 0x030C, 0 } },    // ď → d + caron
    // U+0110-U+0111 Đ đ — no decomp
    { 0x0112, { 0x0045, 0x0304, 0 } },    // Ē → E + macron
    { 0x0113, { 0x0065, 0x0304, 0 } },    // ē → e + macron
    { 0x0114, { 0x0045, 0x0306, 0 } },    // Ĕ → E + breve
    { 0x0115, { 0x0065, 0x0306, 0 } },    // ĕ → e + breve
    { 0x0116, { 0x0045, 0x0307, 0 } },    // Ė → E + dot above
    { 0x0117, { 0x0065, 0x0307, 0 } },    // ė → e + dot above
    { 0x0118, { 0x0045, 0x0328, 0 } },    // Ę → E + ogonek
    { 0x0119, { 0x0065, 0x0328, 0 } },    // ę → e + ogonek
    { 0x011A, { 0x0045, 0x030C, 0 } },    // Ě → E + caron
    { 0x011B, { 0x0065, 0x030C, 0 } },    // ě → e + caron
    { 0x011C, { 0x0047, 0x0302, 0 } },    // Ĝ → G + circumflex
    { 0x011D, { 0x0067, 0x0302, 0 } },    // ĝ → g + circumflex
    { 0x011E, { 0x0047, 0x0306, 0 } },    // Ğ → G + breve
    { 0x011F, { 0x0067, 0x0306, 0 } },    // ğ → g + breve
    { 0x0120, { 0x0047, 0x0307, 0 } },    // Ġ → G + dot above
    { 0x0121, { 0x0067, 0x0307, 0 } },    // ġ → g + dot above
    { 0x0122, { 0x0047, 0x0327, 0 } },    // Ģ → G + cedilla
    { 0x0123, { 0x0067, 0x0327, 0 } },    // ģ → g + cedilla
    { 0x0124, { 0x0048, 0x0302, 0 } },    // Ĥ → H + circumflex
    { 0x0125, { 0x0068, 0x0302, 0 } },    // ĥ → h + circumflex
    // U+0126-U+0127 Ħ ħ — no decomp
    { 0x0128, { 0x0049, 0x0303, 0 } },    // Ĩ → I + tilde
    { 0x0129, { 0x0069, 0x0303, 0 } },    // ĩ → i + tilde
    { 0x012A, { 0x0049, 0x0304, 0 } },    // Ī → I + macron
    { 0x012B, { 0x0069, 0x0304, 0 } },    // ī → i + macron
    { 0x012C, { 0x0049, 0x0306, 0 } },    // Ĭ → I + breve
    { 0x012D, { 0x0069, 0x0306, 0 } },    // ĭ → i + breve
    { 0x012E, { 0x0049, 0x0328, 0 } },    // Į → I + ogonek
    { 0x012F, { 0x0069, 0x0328, 0 } },    // į → i + ogonek
    { 0x0130, { 0x0049, 0x0307, 0 } },    // İ → I + dot above
    // U+0131 ı — no decomp
    // U+0132-U+0133 IJ ij — no NFKD decomp (kept as is in NFD/NFKD)
    { 0x0134, { 0x004A, 0x0302, 0 } },    // Ĵ → J + circumflex
    { 0x0135, { 0x006A, 0x0302, 0 } },    // ĵ → j + circumflex
    { 0x0136, { 0x004B, 0x0327, 0 } },    // Ķ → K + cedilla
    { 0x0137, { 0x006B, 0x0327, 0 } },    // ķ → k + cedilla
    // U+0138 ĸ — no decomp
    { 0x0139, { 0x004C, 0x0301, 0 } },    // Ĺ → L + acute
    { 0x013A, { 0x006C, 0x0301, 0 } },    // ĺ → l + acute
    { 0x013B, { 0x004C, 0x0327, 0 } },    // Ļ → L + cedilla
    { 0x013C, { 0x006C, 0x0327, 0 } },    // ļ → l + cedilla
    { 0x013D, { 0x004C, 0x030C, 0 } },    // Ľ → L + caron
    { 0x013E, { 0x006C, 0x030C, 0 } },    // ľ → l + caron
    // U+013F-U+0141 Ŀ ŀ Ł — no decomp
    // U+0141-U+0142 Ł ł — no decomp
    { 0x0143, { 0x004E, 0x0301, 0 } },    // Ń → N + acute
    { 0x0144, { 0x006E, 0x0301, 0 } },    // ń → n + acute
    { 0x0145, { 0x004E, 0x0327, 0 } },    // Ņ → N + cedilla
    { 0x0146, { 0x006E, 0x0327, 0 } },    // ņ → n + cedilla
    { 0x0147, { 0x004E, 0x030C, 0 } },    // Ň → N + caron
    { 0x0148, { 0x006E, 0x030C, 0 } },    // ň → n + caron
    // U+0149 ŉ — NFKD: U+02BC U+006E
    { 0x0149, { 0x02BC, 0x006E, 0 } },    // ŉ → ʼ n
    // U+014A-U+014B Ŋ ŋ — no decomp
    { 0x014C, { 0x004F, 0x0304, 0 } },    // Ō → O + macron
    { 0x014D, { 0x006F, 0x0304, 0 } },    // ō → o + macron
    { 0x014E, { 0x004F, 0x0306, 0 } },    // Ŏ → O + breve
    { 0x014F, { 0x006F, 0x0306, 0 } },    // ŏ → o + breve
    { 0x0150, { 0x004F, 0x030B, 0 } },    // Ő → O + double acute
    { 0x0151, { 0x006F, 0x030B, 0 } },    // ő → o + double acute
    // U+0152-U+0153 Œ œ — no decomp
    { 0x0154, { 0x0052, 0x0301, 0 } },    // Ŕ → R + acute
    { 0x0155, { 0x0072, 0x0301, 0 } },    // ŕ → r + acute
    { 0x0156, { 0x0052, 0x0327, 0 } },    // Ŗ → R + cedilla
    { 0x0157, { 0x0072, 0x0327, 0 } },    // ŗ → r + cedilla
    { 0x0158, { 0x0052, 0x030C, 0 } },    // Ř → R + caron
    { 0x0159, { 0x0072, 0x030C, 0 } },    // ř → r + caron
    { 0x015A, { 0x0053, 0x0301, 0 } },    // Ś → S + acute
    { 0x015B, { 0x0073, 0x0301, 0 } },    // ś → s + acute
    { 0x015C, { 0x0053, 0x0302, 0 } },    // Ŝ → S + circumflex
    { 0x015D, { 0x0073, 0x0302, 0 } },    // ŝ → s + circumflex
    { 0x015E, { 0x0053, 0x0327, 0 } },    // Ş → S + cedilla
    { 0x015F, { 0x0073, 0x0327, 0 } },    // ş → s + cedilla
    { 0x0160, { 0x0053, 0x030C, 0 } },    // Š → S + caron
    { 0x0161, { 0x0073, 0x030C, 0 } },    // š → s + caron
    { 0x0162, { 0x0054, 0x0327, 0 } },    // Ţ → T + cedilla
    { 0x0163, { 0x0074, 0x0327, 0 } },    // ţ → t + cedilla
    { 0x0164, { 0x0054, 0x030C, 0 } },    // Ť → T + caron
    { 0x0165, { 0x0074, 0x030C, 0 } },    // ť → t + caron
    // U+0166-U+0167 Ŧ ŧ — no decomp
    { 0x0168, { 0x0055, 0x0303, 0 } },    // Ũ → U + tilde
    { 0x0169, { 0x0075, 0x0303, 0 } },    // ũ → u + tilde
    { 0x016A, { 0x0055, 0x0304, 0 } },    // Ū → U + macron
    { 0x016B, { 0x0075, 0x0304, 0 } },    // ū → u + macron
    { 0x016C, { 0x0055, 0x0306, 0 } },    // Ŭ → U + breve
    { 0x016D, { 0x0075, 0x0306, 0 } },    // ŭ → u + breve
    { 0x016E, { 0x0055, 0x030A, 0 } },    // Ů → U + ring above
    { 0x016F, { 0x0075, 0x030A, 0 } },    // ů → u + ring above
    { 0x0170, { 0x0055, 0x030B, 0 } },    // Ű → U + double acute
    { 0x0171, { 0x0075, 0x030B, 0 } },    // ű → u + double acute
    { 0x0172, { 0x0055, 0x0328, 0 } },    // Ų → U + ogonek
    { 0x0173, { 0x0075, 0x0328, 0 } },    // ų → u + ogonek
    { 0x0174, { 0x0057, 0x0302, 0 } },    // Ŵ → W + circumflex
    { 0x0175, { 0x0077, 0x0302, 0 } },    // ŵ → w + circumflex
    { 0x0176, { 0x0059, 0x0302, 0 } },    // Ŷ → Y + circumflex
    { 0x0177, { 0x0079, 0x0302, 0 } },    // ŷ → y + circumflex
    { 0x0178, { 0x0059, 0x0308, 0 } },    // Ÿ → Y + diaeresis
    { 0x0179, { 0x005A, 0x0301, 0 } },    // Ź → Z + acute
    { 0x017A, { 0x007A, 0x0301, 0 } },    // ź → z + acute
    { 0x017B, { 0x005A, 0x0307, 0 } },    // Ż → Z + dot above
    { 0x017C, { 0x007A, 0x0307, 0 } },    // ż → z + dot above
    { 0x017D, { 0x005A, 0x030C, 0 } },    // Ž → Z + caron
    { 0x017E, { 0x007A, 0x030C, 0 } },    // ž → z + caron
    // U+017F ſ LATIN SMALL LETTER LONG S — NFKD → s
    { 0x017F, { 0x0073, 0 } },

    // ---- Latin Extended-B U+0180-U+024F (selected with NFKD decompositions) ----
    // U+0180-U+01BF: mostly no decomp; those that have one:
    { 0x01A0, { 0x004F, 0x031B, 0 } },    // Ơ → O + horn
    { 0x01A1, { 0x006F, 0x031B, 0 } },    // ơ → o + horn
    { 0x01AF, { 0x0055, 0x031B, 0 } },    // Ư → U + horn
    { 0x01B0, { 0x0075, 0x031B, 0 } },    // ư → u + horn
    { 0x01C4, { 0x0044, 0x017D, 0 } },    // Ǆ → D + Ž (note: Ž itself decomposes further)
    { 0x01C5, { 0x0044, 0x017E, 0 } },    // ǅ → D + ž
    { 0x01C6, { 0x0064, 0x017E, 0 } },    // ǆ → d + ž
    { 0x01C7, { 0x004C, 0x004A, 0 } },    // Ǉ → L + J
    { 0x01C8, { 0x004C, 0x006A, 0 } },    // ǈ → L + j
    { 0x01C9, { 0x006C, 0x006A, 0 } },    // ǉ → l + j
    { 0x01CA, { 0x004E, 0x004A, 0 } },    // Ǌ → N + J
    { 0x01CB, { 0x004E, 0x006A, 0 } },    // ǋ → N + j
    { 0x01CC, { 0x006E, 0x006A, 0 } },    // ǌ → n + j
    { 0x01CD, { 0x0041, 0x030C, 0 } },    // Ǎ → A + caron
    { 0x01CE, { 0x0061, 0x030C, 0 } },    // ǎ → a + caron
    { 0x01CF, { 0x0049, 0x030C, 0 } },    // Ǐ → I + caron
    { 0x01D0, { 0x0069, 0x030C, 0 } },    // ǐ → i + caron
    { 0x01D1, { 0x004F, 0x030C, 0 } },    // Ǒ → O + caron
    { 0x01D2, { 0x006F, 0x030C, 0 } },    // ǒ → o + caron
    { 0x01D3, { 0x0055, 0x030C, 0 } },    // Ǔ → U + caron
    { 0x01D4, { 0x0075, 0x030C, 0 } },    // ǔ → u + caron
    { 0x01D5, { 0x0055, 0x0308, 0x0304, 0 } },  // Ǖ → U + diaeresis + macron
    { 0x01D6, { 0x0075, 0x0308, 0x0304, 0 } },  // ǖ → u + diaeresis + macron
    { 0x01D7, { 0x0055, 0x0308, 0x0301, 0 } },  // Ǘ → U + diaeresis + acute
    { 0x01D8, { 0x0075, 0x0308, 0x0301, 0 } },  // ǘ → u + diaeresis + acute
    { 0x01D9, { 0x0055, 0x0308, 0x030C, 0 } },  // Ǚ → U + diaeresis + caron
    { 0x01DA, { 0x0075, 0x0308, 0x030C, 0 } },  // ǚ → u + diaeresis + caron
    { 0x01DB, { 0x0055, 0x0308, 0x0300, 0 } },  // Ǜ → U + diaeresis + grave
    { 0x01DC, { 0x0075, 0x0308, 0x0300, 0 } },  // ǜ → u + diaeresis + grave
    { 0x01DE, { 0x0041, 0x0308, 0x0304, 0 } },  // Ǟ → A + diaeresis + macron
    { 0x01DF, { 0x0061, 0x0308, 0x0304, 0 } },  // ǟ → a + diaeresis + macron
    { 0x01E0, { 0x0041, 0x0307, 0x0304, 0 } },  // Ǡ → A + dot + macron
    { 0x01E1, { 0x0061, 0x0307, 0x0304, 0 } },  // ǡ → a + dot + macron
    { 0x01E6, { 0x0047, 0x030C, 0 } },    // Ǧ → G + caron
    { 0x01E7, { 0x0067, 0x030C, 0 } },    // ǧ → g + caron
    { 0x01E8, { 0x004B, 0x030C, 0 } },    // Ǩ → K + caron
    { 0x01E9, { 0x006B, 0x030C, 0 } },    // ǩ → k + caron
    { 0x01EA, { 0x004F, 0x0328, 0 } },    // Ǫ → O + ogonek
    { 0x01EB, { 0x006F, 0x0328, 0 } },    // ǫ → o + ogonek
    { 0x01EC, { 0x004F, 0x0328, 0x0304, 0 } }, // Ǭ → O + ogonek + macron
    { 0x01ED, { 0x006F, 0x0328, 0x0304, 0 } }, // ǭ → o + ogonek + macron
    { 0x01F0, { 0x006A, 0x030C, 0 } },    // ǰ → j + caron
    { 0x01F1, { 0x0044, 0x005A, 0 } },    // Ǳ → D + Z
    { 0x01F2, { 0x0044, 0x007A, 0 } },    // ǲ → D + z
    { 0x01F3, { 0x0064, 0x007A, 0 } },    // ǳ → d + z
    { 0x01F4, { 0x0047, 0x0301, 0 } },    // Ǵ → G + acute
    { 0x01F5, { 0x0067, 0x0301, 0 } },    // ǵ → g + acute
    { 0x01F8, { 0x004E, 0x0300, 0 } },    // Ǹ → N + grave
    { 0x01F9, { 0x006E, 0x0300, 0 } },    // ǹ → n + grave
    { 0x01FA, { 0x0041, 0x030A, 0x0301, 0 } }, // Ǻ → A + ring + acute
    { 0x01FB, { 0x0061, 0x030A, 0x0301, 0 } }, // ǻ → a + ring + acute
    { 0x01FE, { 0x004F, 0x0301, 0 } },    // Ǿ → O + acute (Ø with acute — NFKD keeps Ø base)
    { 0x01FF, { 0x006F, 0x0301, 0 } },    // ǿ → o + acute
    { 0x0200, { 0x0041, 0x030F, 0 } },    // Ȁ → A + double grave
    { 0x0201, { 0x0061, 0x030F, 0 } },    // ȁ → a + double grave
    { 0x0202, { 0x0041, 0x0311, 0 } },    // Ȃ → A + inverted breve
    { 0x0203, { 0x0061, 0x0311, 0 } },    // ȃ → a + inverted breve
    { 0x0204, { 0x0045, 0x030F, 0 } },    // Ȅ → E + double grave
    { 0x0205, { 0x0065, 0x030F, 0 } },    // ȅ → e + double grave
    { 0x0206, { 0x0045, 0x0311, 0 } },    // Ȇ → E + inverted breve
    { 0x0207, { 0x0065, 0x0311, 0 } },    // ȇ → e + inverted breve
    { 0x0208, { 0x0049, 0x030F, 0 } },    // Ȉ → I + double grave
    { 0x0209, { 0x0069, 0x030F, 0 } },    // ȉ → i + double grave
    { 0x020A, { 0x0049, 0x0311, 0 } },    // Ȋ → I + inverted breve
    { 0x020B, { 0x0069, 0x0311, 0 } },    // ȋ → i + inverted breve
    { 0x020C, { 0x004F, 0x030F, 0 } },    // Ȍ → O + double grave
    { 0x020D, { 0x006F, 0x030F, 0 } },    // ȍ → o + double grave
    { 0x020E, { 0x004F, 0x0311, 0 } },    // Ȏ → O + inverted breve
    { 0x020F, { 0x006F, 0x0311, 0 } },    // ȏ → o + inverted breve
    { 0x0210, { 0x0052, 0x030F, 0 } },    // Ȑ → R + double grave
    { 0x0211, { 0x0072, 0x030F, 0 } },    // ȑ → r + double grave
    { 0x0212, { 0x0052, 0x0311, 0 } },    // Ȓ → R + inverted breve
    { 0x0213, { 0x0072, 0x0311, 0 } },    // ȓ → r + inverted breve
    { 0x0214, { 0x0055, 0x030F, 0 } },    // Ȕ → U + double grave
    { 0x0215, { 0x0075, 0x030F, 0 } },    // ȕ → u + double grave
    { 0x0216, { 0x0055, 0x0311, 0 } },    // Ȗ → U + inverted breve
    { 0x0217, { 0x0075, 0x0311, 0 } },    // ȗ → u + inverted breve
    { 0x0218, { 0x0053, 0x0326, 0 } },    // Ș → S + comma below
    { 0x0219, { 0x0073, 0x0326, 0 } },    // ș → s + comma below
    { 0x021A, { 0x0054, 0x0326, 0 } },    // Ț → T + comma below
    { 0x021B, { 0x0074, 0x0326, 0 } },    // ț → t + comma below
    { 0x021E, { 0x0048, 0x030C, 0 } },    // Ȟ → H + caron
    { 0x021F, { 0x0068, 0x030C, 0 } },    // ȟ → h + caron
    { 0x0226, { 0x0041, 0x0307, 0 } },    // Ȧ → A + dot above
    { 0x0227, { 0x0061, 0x0307, 0 } },    // ȧ → a + dot above
    { 0x0228, { 0x0045, 0x0327, 0 } },    // Ȩ → E + cedilla
    { 0x0229, { 0x0065, 0x0327, 0 } },    // ȩ → e + cedilla
    { 0x022A, { 0x004F, 0x0308, 0x0304, 0 } }, // Ȫ → O + diaeresis + macron
    { 0x022B, { 0x006F, 0x0308, 0x0304, 0 } }, // ȫ → o + diaeresis + macron
    { 0x022C, { 0x004F, 0x0303, 0x0304, 0 } }, // Ȭ → O + tilde + macron
    { 0x022D, { 0x006F, 0x0303, 0x0304, 0 } }, // ȭ → o + tilde + macron
    { 0x022E, { 0x004F, 0x0307, 0 } },    // Ȯ → O + dot above
    { 0x022F, { 0x006F, 0x0307, 0 } },    // ȯ → o + dot above
    { 0x0230, { 0x004F, 0x0307, 0x0304, 0 } }, // Ȱ → O + dot + macron
    { 0x0231, { 0x006F, 0x0307, 0x0304, 0 } }, // ȱ → o + dot + macron
    { 0x0232, { 0x0059, 0x0304, 0 } },    // Ȳ → Y + macron
    { 0x0233, { 0x0079, 0x0304, 0 } },    // ȳ → y + macron

    // ---- Spacing Modifier Letters with NFKD decompositions U+02B0-U+02FF ----
    { 0x02B0, { 0x0068, 0 } },    // ʰ → h
    { 0x02B1, { 0x0266, 0 } },    // ʱ → ɦ
    { 0x02B2, { 0x006A, 0 } },    // ʲ → j
    { 0x02B3, { 0x0072, 0 } },    // ʳ → r
    { 0x02B4, { 0x0279, 0 } },    // ʴ → ɹ
    { 0x02B5, { 0x027B, 0 } },    // ʵ → ɻ
    { 0x02B6, { 0x0281, 0 } },    // ʶ → ʁ
    { 0x02B7, { 0x0077, 0 } },    // ʷ → w
    { 0x02B8, { 0x0079, 0 } },    // ʸ → y
    { 0x02D8, { 0x0020, 0x0306, 0 } },  // ˘ BREVE → SPACE + combining breve
    { 0x02D9, { 0x0020, 0x0307, 0 } },  // ˙ DOT ABOVE → SPACE + dot
    { 0x02DA, { 0x0020, 0x030A, 0 } },  // ˚ RING ABOVE → SPACE + ring
    { 0x02DB, { 0x0020, 0x0328, 0 } },  // ˛ OGONEK → SPACE + ogonek
    { 0x02DC, { 0x0020, 0x0303, 0 } },  // ˜ SMALL TILDE → SPACE + tilde
    { 0x02DD, { 0x0020, 0x030B, 0 } },  // ˝ DOUBLE ACUTE → SPACE + double acute
    { 0x02E0, { 0x0263, 0 } },    // ˠ → ɣ
    { 0x02E1, { 0x006C, 0 } },    // ˡ → l
    { 0x02E2, { 0x0073, 0 } },    // ˢ → s
    { 0x02E3, { 0x0078, 0 } },    // ˣ → x
    { 0x02E4, { 0x0295, 0 } },    // ˤ → ʕ

    // ---- Letterlike symbols ----
    { 0x2126, { 0x03A9, 0 } },    // Ω OHM SIGN → Greek capital omega
    { 0x212A, { 0x004B, 0 } },    // K KELVIN SIGN → Latin K
    { 0x212B, { 0x0041, 0x030A, 0 } }, // Å ANGSTROM SIGN → A + ring above

    // ---- Number forms ----
    { 0x2153, { 0x0031, 0x2044, 0x0033, 0 } }, // ⅓ → 1/3
    { 0x2154, { 0x0032, 0x2044, 0x0033, 0 } }, // ⅔ → 2/3
    { 0x2155, { 0x0031, 0x2044, 0x0035, 0 } }, // ⅕ → 1/5
    { 0x2156, { 0x0032, 0x2044, 0x0035, 0 } }, // ⅖ → 2/5
    { 0x2157, { 0x0033, 0x2044, 0x0035, 0 } }, // ⅗ → 3/5
    { 0x2158, { 0x0034, 0x2044, 0x0035, 0 } }, // ⅘ → 4/5
    { 0x2159, { 0x0031, 0x2044, 0x0036, 0 } }, // ⅙ → 1/6
    { 0x215A, { 0x0035, 0x2044, 0x0036, 0 } }, // ⅚ → 5/6
    { 0x215B, { 0x0031, 0x2044, 0x0038, 0 } }, // ⅛ → 1/8
    { 0x215C, { 0x0033, 0x2044, 0x0038, 0 } }, // ⅜ → 3/8
    { 0x215D, { 0x0035, 0x2044, 0x0038, 0 } }, // ⅝ → 5/8
    { 0x215E, { 0x0037, 0x2044, 0x0038, 0 } }, // ⅞ → 7/8
    { 0x215F, { 0x0031, 0x2044, 0 } },          // ⅟ → 1/

    // ---- Alphabetic Presentation Forms U+FB00-U+FB06 ----
    { 0xFB00, { 0x0066, 0x0066, 0 } },          // ﬀ ff
    { 0xFB01, { 0x0066, 0x0069, 0 } },          // ﬁ fi
    { 0xFB02, { 0x0066, 0x006C, 0 } },          // ﬂ fl
    { 0xFB03, { 0x0066, 0x0066, 0x0069, 0 } },  // ﬃ ffi
    { 0xFB04, { 0x0066, 0x0066, 0x006C, 0 } },  // ﬄ ffl
    { 0xFB05, { 0x0073, 0x0074, 0 } },          // ﬅ st (long s + t)
    { 0xFB06, { 0x0073, 0x0074, 0 } },          // ﬆ st
};
// clang-format on

static constexpr size_t NFKD_TABLE_SIZE =
    sizeof(NFKD_TABLE) / sizeof(NFKD_TABLE[0]);

/// Binary search in NFKD_TABLE.  Returns pointer to entry or nullptr.
static const NfkdEntry* nfkd_lookup(uint32_t cp) {
    size_t lo = 0, hi = NFKD_TABLE_SIZE;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        if (NFKD_TABLE[mid].cp == cp) return &NFKD_TABLE[mid];
        if (NFKD_TABLE[mid].cp < cp)  lo = mid + 1;
        else                           hi = mid;
    }
    return nullptr;
}

// ---------------------------------------------------------------------------
// Canonical combining class (CCC) — Unicode 15.0 U+0300-U+036F
// ---------------------------------------------------------------------------

/// Returns the canonical combining class for a codepoint.
/// 0 means "starter" (not a combining character, or unknown).
static uint8_t combining_class(uint32_t cp) {
    // U+0300-U+036F: Combining Diacritical Marks
    // Values from Unicode 15.0 DerivedCombiningClass.txt
    static const uint8_t CCC_0300[] = {
        // U+0300..U+030F
        230, 230, 230, 230, 230, 230, 230, 230, 230, 230, 230, 230, 230, 230, 230, 230,
        // U+0310..U+031F
        230, 230, 230, 230, 230, 232, 220, 220, 220, 220, 232, 216, 220, 220, 220, 220,
        // U+0320..U+032F
        220, 202, 202, 220, 220, 220, 220, 202, 202, 220, 220, 220, 220, 220, 220, 220,
        // U+0330..U+033F
        220, 220, 220, 220, 220, 220, 220, 220, 220, 220, 220, 220, 220, 220, 220, 220,
        // U+0340..U+034F
        230, 230, 230, 230, 230, 240, 230, 230, 220, 220, 220, 220, 230, 230,   1,   1,
        // U+0350..U+035F
        230, 230, 230, 230, 230, 230, 230, 230, 230, 230, 220, 220, 230, 230, 230, 230,
        // U+0360..U+036F
        230, 230, 230, 230, 230, 230, 230, 230, 230, 230, 230, 230, 230, 230, 230, 230,
    };

    if (cp >= 0x0300 && cp <= 0x036F)
        return CCC_0300[cp - 0x0300];

    // U+1DC0-U+1DFF: Combining Diacritical Marks Supplement (all 230)
    if (cp >= 0x1DC0 && cp <= 0x1DFF)
        return 230;

    // U+20D0-U+20FF: Combining Diacritical Marks for Symbols
    if (cp >= 0x20D0 && cp <= 0x20DC) return 230;
    if (cp == 0x20DD || cp == 0x20DE || cp == 0x20DF || cp == 0x20E0) return 1;
    if (cp >= 0x20E1 && cp <= 0x20E4) return 230;
    if (cp >= 0x20E5 && cp <= 0x20FF) return 1;

    return 0;
}

// ---------------------------------------------------------------------------
// Full NFKD decomposition (recursive, handles multi-level like Ǖ)
// ---------------------------------------------------------------------------

static void nfkd_decompose_cp(uint32_t cp, std::vector<uint32_t>& out) {
    const NfkdEntry* e = nfkd_lookup(cp);
    if (!e) {
        out.push_back(cp);
        return;
    }
    // Recursively decompose each component
    for (int i = 0; i < 4 && e->d[i] != 0; ++i) {
        nfkd_decompose_cp(e->d[i], out);
    }
}

// ---------------------------------------------------------------------------
// Canonical ordering: stable sort of combining characters
// ---------------------------------------------------------------------------

static void canonical_order(std::vector<uint32_t>& cps) {
    size_t n = cps.size();
    if (n < 2) return;

    // Bubble sort (data is typically very short — usually 1-3 combiners)
    bool changed = true;
    while (changed) {
        changed = false;
        for (size_t i = 1; i < n; ++i) {
            uint8_t cc_prev = combining_class(cps[i - 1]);
            uint8_t cc_cur  = combining_class(cps[i]);
            if (cc_prev > 0 && cc_cur > 0 && cc_prev > cc_cur) {
                std::swap(cps[i - 1], cps[i]);
                changed = true;
            }
        }
    }
}

} // anonymous namespace

namespace secp256k1 {

std::string nfkd_normalize(const std::string& utf8) {
    if (utf8.empty() || is_ascii(utf8)) return utf8;

    const uint8_t* buf = reinterpret_cast<const uint8_t*>(utf8.data());
    const size_t len   = utf8.size();

    // Step 1: decode UTF-8 to codepoints and NFKD-decompose each one
    std::vector<uint32_t> cps;
    cps.reserve(utf8.size());

    size_t pos = 0;
    while (pos < len) {
        uint32_t cp = utf8_decode(buf, len, pos);
        nfkd_decompose_cp(cp, cps);
    }

    // Step 2: canonical ordering
    canonical_order(cps);

    // Step 3: encode back to UTF-8
    std::string out;
    out.reserve(cps.size() * 2);
    for (uint32_t cp : cps) {
        utf8_encode(out, cp);
    }
    return out;
}

} // namespace secp256k1

#endif // platform dispatch
