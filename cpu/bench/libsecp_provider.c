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

/* Include the entire libsecp256k1 as a single compilation unit.
 * Path resolved via CMake target_include_directories (LIBSECP_SRC_DIR).
 * CI passes -DLIBSECP_SRC_DIR=<cloned>/src; local dev uses _research_repos default. */
#include "secp256k1.c"

/* ---- Thin wrappers exposing internal ops for benchmarking ---- */

#include "field.h"
#include "scalar.h"
#include "group.h"
#include "ecmult.h"
#include "ecmult_gen.h"

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

/* Field arithmetic wrappers (operate on raw secp256k1_fe) */
void libsecp_fe_mul(void *r, const void *a, const void *b) {
    secp256k1_fe_mul((secp256k1_fe *)r, (const secp256k1_fe *)a, (const secp256k1_fe *)b);
}

void libsecp_fe_sqr(void *r, const void *a) {
    secp256k1_fe_sqr((secp256k1_fe *)r, (const secp256k1_fe *)a);
}

void libsecp_fe_add(void *r, const void *a) {
    secp256k1_fe_add((secp256k1_fe *)r, (const secp256k1_fe *)a);
}

void libsecp_fe_negate(void *r, const void *a, int m) {
    secp256k1_fe_negate_unchecked((secp256k1_fe *)r, (const secp256k1_fe *)a, m);
}

void libsecp_fe_normalize(void *r) {
    secp256k1_fe_normalize((secp256k1_fe *)r);
}

void libsecp_fe_set_b32(void *r, const unsigned char *b32) {
    secp256k1_fe_set_b32_mod((secp256k1_fe *)r, b32);
}

/* Scalar arithmetic wrappers (operate on raw secp256k1_scalar) */
void libsecp_scalar_mul(void *r, const void *a, const void *b) {
    secp256k1_scalar_mul((secp256k1_scalar *)r, (const secp256k1_scalar *)a,
                         (const secp256k1_scalar *)b);
}

void libsecp_scalar_inverse(void *r, const void *a) {
    secp256k1_scalar_inverse((secp256k1_scalar *)r, (const secp256k1_scalar *)a);
}

void libsecp_scalar_inverse_var(void *r, const void *a) {
    secp256k1_scalar_inverse_var((secp256k1_scalar *)r, (const secp256k1_scalar *)a);
}

void libsecp_scalar_add(void *r, const void *a, const void *b) {
    secp256k1_scalar_add((secp256k1_scalar *)r, (const secp256k1_scalar *)a,
                         (const secp256k1_scalar *)b);
}

void libsecp_scalar_negate(void *r, const void *a) {
    secp256k1_scalar_negate((secp256k1_scalar *)r, (const secp256k1_scalar *)a);
}

void libsecp_scalar_set_b32(void *r, const unsigned char *b32, int *overflow) {
    secp256k1_scalar_set_b32((secp256k1_scalar *)r, b32, overflow);
}

/* Point arithmetic wrappers */
void libsecp_gej_double_var(void *r, const void *a) {
    secp256k1_gej_double_var((secp256k1_gej *)r, (const secp256k1_gej *)a, NULL);
}

void libsecp_gej_add_ge_var(void *r, const void *a, const void *b) {
    secp256k1_gej_add_ge_var((secp256k1_gej *)r, (const secp256k1_gej *)a,
                             (const secp256k1_ge *)b, NULL);
}

/* ecmult: a*P + b*G (Strauss dual mul - used in verify) */
void libsecp_ecmult(void *r, const void *a, const void *na, const void *ng) {
    secp256k1_ecmult((secp256k1_gej *)r, (const secp256k1_gej *)a,
                     (const secp256k1_scalar *)na, (const secp256k1_scalar *)ng);
}

/* ecmult_gen: k*G (generator mul using Comb tables) */
void libsecp_ecmult_gen(const void *ctx_ecmult_gen, void *r, const void *k) {
    secp256k1_ecmult_gen((const secp256k1_ecmult_gen_context *)ctx_ecmult_gen,
                         (secp256k1_gej *)r, (const secp256k1_scalar *)k);
}

/* Helper: get the ecmult_gen_context from a secp256k1_context */
const void* libsecp_get_ecmult_gen_ctx(const secp256k1_context *ctx) {
    return &ctx->ecmult_gen_ctx;
}

/* Helper: set up a Jacobian point from a pubkey (for ecmult input) */
void libsecp_gej_set_ge(void *r, const void *a) {
    secp256k1_gej_set_ge((secp256k1_gej *)r, (const secp256k1_ge *)a);
}

/* Helper: parse a pubkey into a ge (affine) point */
int libsecp_pubkey_load(const secp256k1_context *ctx, void *ge, const secp256k1_pubkey *pubkey) {
    return secp256k1_pubkey_load(ctx, (secp256k1_ge *)ge, pubkey);
}
