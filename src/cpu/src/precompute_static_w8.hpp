// AUTO-GENERATED — see tools/gen_precompute_w8.py
// Header for the compiled-in w=8 precomputed table.
#pragma once
#include <cstdint>
namespace secp256k1 { namespace fast { namespace static_w8 {
    struct StaticPoint { unsigned char inf; unsigned char x[32]; unsigned char y[32]; };
    static constexpr unsigned kWindowBits  = 8;
    static constexpr unsigned kWindowCount = 32;
    static constexpr unsigned kDigitCount  = 256;
    extern const StaticPoint kBaseTable[32][256];
    extern const StaticPoint kPsiTable[32][256];
}}} // namespace
