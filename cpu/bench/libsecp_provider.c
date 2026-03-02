/**
 * libsecp_provider.c -- bitcoin-core libsecp256k1 symbol provider
 *
 * Compiles the official bitcoin-core libsecp256k1 as a single translation unit.
 * Provides public API symbols (secp256k1_context_create, secp256k1_ecdsa_sign,
 * secp256k1_schnorrsig_sign32, etc.) for bench_unified.cpp to call.
 *
 * No benchmark code -- just the library.
 * Build: compiled as C (not C++), linked with bench_unified target.
 */

/* libsecp256k1 module configuration -- match bitcoin-core defaults */
#define ENABLE_MODULE_ECDH 0
#define ENABLE_MODULE_RECOVERY 0
#define ENABLE_MODULE_EXTRAKEYS 1
#define ENABLE_MODULE_SCHNORRSIG 1
#define ENABLE_MODULE_MUSIG 0
#define ENABLE_MODULE_ELLSWIFT 0

/* Include the entire libsecp256k1 as a single compilation unit */
#include "../../../../_research_repos/secp256k1/src/secp256k1.c"

/* ---- Thin wrappers exposing internal field ops for benchmarking ---- */

#include "../../../../_research_repos/secp256k1/src/field.h"

void libsecp_fe_inv_var(unsigned char out32[32], const unsigned char in32[32]) {
    secp256k1_fe a, r;
    secp256k1_fe_set_b32_mod(&a, in32);
    secp256k1_fe_inv_var(&r, &a);
    secp256k1_fe_normalize_var(&r);
    secp256k1_fe_get_b32(out32, &r);
}

void libsecp_fe_inv_var_raw(void *r, const void *a) {
    secp256k1_fe_inv_var((secp256k1_fe *)r, (const secp256k1_fe *)a);
}
