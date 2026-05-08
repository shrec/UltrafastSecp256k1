// AUTO-GENERATED — see tools/gen_precompute_w10.py
#pragma once
#include <cstdint>
namespace secp256k1{namespace fast{namespace static_w10{
    struct StaticPoint{unsigned char inf;unsigned char x[32];unsigned char y[32];};
    static constexpr unsigned kWindowBits=10,kWindowCount=26,kDigitCount=1024;
    extern const StaticPoint kBaseTable[26][1024];
    extern const StaticPoint kPsiTable[26][1024];
}} }
