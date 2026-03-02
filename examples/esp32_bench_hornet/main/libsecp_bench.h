/* libsecp_bench.h -- result struct for apple-to-apple ratio computation */
#ifndef LIBSECP_BENCH_H
#define LIBSECP_BENCH_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    double generator_mul_ns;
    double ecdsa_sign_ns;
    double ecdsa_verify_ns;
    double schnorr_keypair_ns;
    double schnorr_sign_ns;
    double schnorr_verify_ns;
} libsecp_results_t;

void libsecp_benchmark(libsecp_results_t* out);

#ifdef __cplusplus
}
#endif

#endif /* LIBSECP_BENCH_H */
