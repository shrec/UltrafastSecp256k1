/**
 * test_bip327_keyagg_vectors.cpp — official BIP-327 MuSig2 key-aggregation vectors.
 *
 * From bitcoin/bips bip-0327/vectors/key_agg_vectors.json. Exercises the
 * libsecp256k1 drop-in shim's MuSig2 module (secp256k1_musig_pubkey_agg):
 *   - valid cases: agg of the indexed pubkeys equals the canonical aggregate x-only
 *     key (order-dependent; duplicates handled per spec),
 *   - negative: the deliberately-malformed pubkeys (x>=field, 0x04 prefix in 33B)
 *     are rejected by secp256k1_ec_pubkey_parse.
 *
 * Standalone:
 *   g++ -std=c++17 -I ../compat/libsecp256k1_shim/include -I ../src/cpu/include \
 *       test_bip327_keyagg_vectors.cpp -lufsecp -o t && ./t
 */
#include "secp256k1.h"
#include "secp256k1_extrakeys.h"
#include "secp256k1_musig.h"
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

static int g_fail = 0;
#define CHECK(cond, msg) do { if(!(cond)){ std::printf("  FAIL: %s\n", msg); ++g_fail; } \
                              else { std::printf("  ok  : %s\n", msg); } } while(0)

static std::vector<uint8_t> hx(const char* s){
    std::vector<uint8_t> v; if(!s) return v; size_t n=std::strlen(s);
    for(size_t i=0;i+1<n;i+=2){ auto d=[&](char c)->int{ return c<='9'?c-'0':(c|32)-'a'+10; };
        v.push_back((uint8_t)((d(s[i])<<4)|d(s[i+1]))); }
    return v;
}

// bitcoin/bips bip-0327/vectors/key_agg_vectors.json  ("pubkeys" array)
static const char* PUBKEYS[] = {
    "02F9308A019258C31049344F85F89D5229B531C845836F99B08601F113BCE036F9",  // 0 valid
    "03DFF1D77F2A671C5F36183726DB2341BE58FEAE1DA2DECED843240F7B502BA659",  // 1 valid
    "023590A94E768F8E1815C2F24B4D80A8E3149316C3518CE7B7AD338368D038CA66",  // 2 valid
    "020000000000000000000000000000000000000000000000000000000000000005",  // 3 (not on curve)
    "02FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC30",  // 4 x >= field
    "04F9308A019258C31049344F85F89D5229B531C845836F99B08601F113BCE036F9",  // 5 bad 0x04 prefix
    "03935F972DA013F80AE011890FA89B67A27B7BE6CCB24D3274D18B2D4067F261A9",  // 6 valid
};

struct Case { std::vector<int> idx; const char* expected; };
static const Case VALID[] = {
    {{0,1,2},   "90539EEDE565F5D054F32CC0C220126889ED1E5D193BAF15AEF344FE59D4610C"},
    {{2,1,0},   "6204DE8B083426DC6EAF9502D27024D53FC826BF7D2012148A0575435DF54B2B"},
    {{0,0,0},   "B436E3BAD62B8CD409969A224731C193D051162D8C5AE8B109306127DA3AA935"},
    {{0,0,1,1}, "69BC22BFA5D106306E48A20679DE1D7389386124D07571D0D872686028C26A3E"},
};

int main(){
    secp256k1_context* ctx = secp256k1_context_create(SECP256K1_CONTEXT_SIGN | SECP256K1_CONTEXT_VERIFY);
    if(!ctx){ std::printf("ctx fail\n"); return 1; }
    std::printf("== BIP-327 MuSig2 key-agg vectors via libsecp shim ==\n");

    const int NPK = (int)(sizeof(PUBKEYS)/sizeof(PUBKEYS[0]));
    secp256k1_pubkey pk[7]; bool pkok[7];
    for(int i=0;i<NPK;++i){ auto b=hx(PUBKEYS[i]); pkok[i] = b.size()==33 &&
        secp256k1_ec_pubkey_parse(ctx,&pk[i],b.data(),b.size())==1; }

    // negative: the malformed pubkeys must be rejected at parse
    CHECK(!pkok[4], "pubkey 4 (x >= field size) rejected by ec_pubkey_parse");
    CHECK(!pkok[5], "pubkey 5 (0x04 prefix in 33-byte input) rejected by ec_pubkey_parse");

    // valid aggregation cases
    int ci=0;
    for(const auto& c : VALID){
        char nm[128];
        bool inputs_ok=true;
        std::vector<const secp256k1_pubkey*> ptrs;
        for(int j : c.idx){ if(j<0||j>=NPK||!pkok[j]){ inputs_ok=false; break; } ptrs.push_back(&pk[j]); }
        std::snprintf(nm,sizeof(nm),"case %d inputs parse", ci);
        CHECK(inputs_ok, nm);
        if(!inputs_ok){ ++ci; continue; }

        secp256k1_xonly_pubkey agg; secp256k1_musig_keyagg_cache cache;
        int r = secp256k1_musig_pubkey_agg(ctx, &agg, &cache, ptrs.data(), ptrs.size());
        uint8_t got[32]; if(r) secp256k1_xonly_pubkey_serialize(ctx, got, &agg);
        auto exp = hx(c.expected);
        bool ok = r==1 && exp.size()==32 && std::memcmp(got, exp.data(), 32)==0;
        std::snprintf(nm,sizeof(nm),"case %d musig_pubkey_agg == canonical aggregate", ci);
        CHECK(ok, nm);
        ++ci;
    }

    secp256k1_context_destroy(ctx);
    std::printf("\n%s\n", g_fail==0?"ALL PASS":"FAILURES PRESENT");
    return g_fail==0?0:1;
}
