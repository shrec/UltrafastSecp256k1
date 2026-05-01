// ============================================================================
// GPU ZK Benchmark -- batch timing for all ZK operations
// ============================================================================

#include "ct/ct_sign.cuh"
#include "ct/ct_zk.cuh"
#include "msm.cuh"
#include <cstdio>
#include <cstring>
#include <vector>

using namespace secp256k1::cuda;

// -- Config -------------------------------------------------------------------
static constexpr int BATCH       = 4096;
static constexpr int BP_BATCH    = 256;   // Bulletproof: smaller batch (heavy per-thread)
static constexpr int WARMUP      = 5;
static constexpr int PASSES      = 11;
static constexpr int WARPS_PER_BLOCK = 4; // Warp-cooperative: 128 threads per block

// -- Test data ----------------------------------------------------------------

__device__ static const uint8_t BENCH_MSG[32] = {
    0x9f, 0x86, 0xd0, 0x81, 0x88, 0x4c, 0x7d, 0x65,
    0x9a, 0x2f, 0xea, 0xa0, 0xc5, 0x5a, 0xd0, 0x15,
    0xa3, 0xbf, 0x4f, 0x1b, 0x2b, 0x0b, 0x82, 0x2c,
    0xd1, 0x5d, 0x6c, 0x15, 0xb0, 0xf0, 0x0a, 0x08
};
__device__ static const uint8_t BENCH_AUX[32] = {0};

__device__ static void make_key(int idx, Scalar* key) {
    key->limbs[0] = (uint64_t)idx + 1;
    key->limbs[1] = 0;
    key->limbs[2] = 0;
    key->limbs[3] = 0;
}

// ===== Kernels ===============================================================

__global__ void bench_knowledge_prove_kernel(
    const JacobianPoint* pubkeys, KnowledgeProofGPU* proofs, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    Scalar key; make_key(idx, &key);
    ct::ct_knowledge_prove_generator_device(&key, &pubkeys[idx], BENCH_MSG, BENCH_AUX, &proofs[idx]);
}

__global__ void bench_knowledge_verify_kernel(
    const KnowledgeProofGPU* proofs, const AffinePoint* pubs, int n, int* fail_count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    bool v = knowledge_verify_device(&proofs[idx], &pubs[idx], BENCH_MSG);
    if (!v) atomicAdd(fail_count, 1);
}

__global__ void bench_dleq_prove_kernel(
    const JacobianPoint* G, const JacobianPoint* H,
    const JacobianPoint* Pj, const JacobianPoint* Qj,
    DLEQProofGPU* proofs, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    Scalar key; make_key(idx, &key);
    ct::ct_dleq_prove_device(&key, G, H, &Pj[idx], &Qj[idx], BENCH_AUX, &proofs[idx]);
}

__global__ void bench_dleq_verify_kernel(
    const DLEQProofGPU* proofs,
    const AffinePoint* Ga, const AffinePoint* Ha,
    const AffinePoint* Pa, const AffinePoint* Qa,
    int n, int* fail_count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    bool v = dleq_verify_device(&proofs[idx], Ga, Ha, &Pa[idx], &Qa[idx]);
    if (!v) atomicAdd(fail_count, 1);
}

// ===== Setup kernels =========================================================

__global__ void setup_jac_pubkeys_kernel(JacobianPoint* pubs_jac, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    Scalar key; make_key(idx, &key);
    ct::ct_generator_mul(&key, &pubs_jac[idx]);
}

__global__ void setup_dleq_jac_points_kernel(
    const JacobianPoint* G, const JacobianPoint* H,
    JacobianPoint* Pj, JacobianPoint* Qj, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    Scalar key; make_key(idx, &key);
    ct::ct_scalar_mul(G, &key, &Pj[idx]);
    ct::ct_scalar_mul(H, &key, &Qj[idx]);
}

__global__ void setup_pubkeys_kernel(AffinePoint* pubs, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    Scalar key; make_key(idx, &key);
    JacobianPoint P;
    ct::ct_generator_mul(&key, &P);
    FieldElement ax, ay;
    jacobian_to_affine(&P, &ax, &ay);
    pubs[idx].x = ax;
    pubs[idx].y = ay;
}

__global__ void setup_bases_kernel(
    JacobianPoint* Gj, JacobianPoint* Hj,
    AffinePoint* Ga, AffinePoint* Ha)
{
    JacobianPoint G;
    for (int i = 0; i < 4; ++i) {
        G.x.limbs[i] = GENERATOR_X[i];
        G.y.limbs[i] = GENERATOR_Y[i];
    }
    field_set_one(&G.z);
    G.infinity = false;
    *Gj = G;

    JacobianPoint H;
    jacobian_double(&G, &H);
    *Hj = H;

    // Affine versions for verify
    Ga->x = G.x;
    Ga->y = G.y;  // G has z=1, already affine

    FieldElement hx, hy;
    jacobian_to_affine(&H, &hx, &hy);
    Ha->x = hx;
    Ha->y = hy;
}

__global__ void setup_dleq_points_kernel(
    const JacobianPoint* G, const JacobianPoint* H,
    AffinePoint* Pa, AffinePoint* Qa, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    Scalar key; make_key(idx, &key);
    JacobianPoint P, Q;
    ct::ct_scalar_mul(G, &key, &P);
    ct::ct_scalar_mul(H, &key, &Q);
    FieldElement tx, ty;
    jacobian_to_affine(&P, &tx, &ty);
    Pa[idx].x = tx; Pa[idx].y = ty;
    jacobian_to_affine(&Q, &tx, &ty);
    Qa[idx].x = tx; Qa[idx].y = ty;
}

// ===== Pedersen + Bulletproof kernels =========================================

__global__ void bench_pedersen_commit_kernel(
    const Scalar* values, const Scalar* blindings,
    const AffinePoint* H_gen, AffinePoint* commitments, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    JacobianPoint result;
    pedersen_commit_device(&values[idx], &blindings[idx], H_gen, &result);
    FieldElement z_inv, z_inv2, z_inv3;
    field_inv(&result.z, &z_inv);
    field_sqr(&z_inv, &z_inv2);
    field_mul(&z_inv2, &z_inv, &z_inv3);
    field_mul(&result.x, &z_inv2, &commitments[idx].x);
    field_mul(&result.y, &z_inv3, &commitments[idx].y);
}

__global__ void bench_bp_prove_kernel(
    const uint64_t* values, const Scalar* blindings,
    const AffinePoint* commitments, const AffinePoint* H_gen,
    RangeProofGPU* proofs, bool* results, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    uint8_t aux[32] = {0};
    results[idx] = ct::ct_range_prove_device(
        values[idx], &blindings[idx], &commitments[idx], H_gen, aux, &proofs[idx]);
}

// ===== Warp-cooperative Bulletproof prove kernel =============================
__global__ void bench_bp_prove_warp_kernel(
    const uint64_t* values, const Scalar* blindings,
    const AffinePoint* commitments, const AffinePoint* H_gen,
    RangeProofGPU* proofs, bool* results, int n)
{
    __shared__ ct::BPProveWarpShared prove_smem[WARPS_PER_BLOCK];

    const int warp_id_in_block = threadIdx.x >> 5;
    const int global_warp_id   = blockIdx.x * WARPS_PER_BLOCK + warp_id_in_block;

    if (global_warp_id >= n) return;

    uint8_t aux[32] = {0};
    bool ok = ct::ct_range_prove_warp_device(
        values[global_warp_id],
        &blindings[global_warp_id],
        &commitments[global_warp_id],
        H_gen,
        aux,
        &proofs[global_warp_id],
        &prove_smem[warp_id_in_block]);

    if ((threadIdx.x & 31) == 0)
        results[global_warp_id] = ok;
}

// ===== Warp-cooperative Bulletproof prove LUT4 kernel =========================
__global__ void bench_bp_prove_warp_lut4_kernel(
    const uint64_t* values, const Scalar* blindings,
    const AffinePoint* commitments, const AffinePoint* H_gen,
    RangeProofGPU* proofs, bool* results, int n)
{
    __shared__ ct::BPProveWarpShared prove_smem[WARPS_PER_BLOCK];

    const int warp_id_in_block = threadIdx.x >> 5;
    const int global_warp_id   = blockIdx.x * WARPS_PER_BLOCK + warp_id_in_block;

    if (global_warp_id >= n) return;

    uint8_t aux[32] = {0};
    bool ok = ct::ct_range_prove_warp_lut4_device(
        values[global_warp_id],
        &blindings[global_warp_id],
        &commitments[global_warp_id],
        H_gen,
        aux,
        &proofs[global_warp_id],
        &prove_smem[warp_id_in_block]);

    if ((threadIdx.x & 31) == 0)
        results[global_warp_id] = ok;
}

__global__ void bench_bp_verify_kernel(
    const RangeProofGPU* proofs, const AffinePoint* commitments,
    const AffinePoint* H_gen, int* fail_count, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    bool v = range_verify_full_device(&proofs[idx], &commitments[idx], H_gen);
    if (!v) atomicAdd(fail_count, 1);
}

// ===== Warp-cooperative Bulletproof verify kernel ============================
// 32 threads (1 warp) per proof. Each block has WARPS_PER_BLOCK warps.

__global__ void bench_bp_verify_warp_kernel(
    const RangeProofGPU* proofs, const AffinePoint* commitments,
    const AffinePoint* H_gen, int* fail_count, int n)
{
    // Shared memory: one BPWarpShared per warp in the block
    __shared__ BPWarpShared warp_smem[WARPS_PER_BLOCK];

    const int warp_id_in_block = threadIdx.x >> 5;         // which warp within this block
    const int global_warp_id   = blockIdx.x * WARPS_PER_BLOCK + warp_id_in_block;

    if (global_warp_id >= n) return;

    bool v = range_verify_warp_device(
        &proofs[global_warp_id],
        &commitments[global_warp_id],
        H_gen,
        &warp_smem[warp_id_in_block]);

    // Only lane 0 reports failure
    if ((threadIdx.x & 31) == 0 && !v)
        atomicAdd(fail_count, 1);
}

// ===== Precomputed-table warp verify kernels =================================

__global__ void bench_bp_verify_warp_precomp_w4_kernel(
    const RangeProofGPU* proofs, const AffinePoint* commitments,
    const AffinePoint* H_gen, int* fail_count, int n)
{
    __shared__ BPWarpShared warp_smem[WARPS_PER_BLOCK];
    const int warp_id_in_block = threadIdx.x >> 5;
    const int global_warp_id   = blockIdx.x * WARPS_PER_BLOCK + warp_id_in_block;
    if (global_warp_id >= n) return;

    bool v = range_verify_warp_precomp_w4_device(
        &proofs[global_warp_id], &commitments[global_warp_id],
        H_gen, &warp_smem[warp_id_in_block]);
    if ((threadIdx.x & 31) == 0 && !v)
        atomicAdd(fail_count, 1);
}

__global__ void bench_bp_verify_warp_precomp_w8_kernel(
    const RangeProofGPU* proofs, const AffinePoint* commitments,
    const AffinePoint* H_gen, int* fail_count, int n)
{
    __shared__ BPWarpShared warp_smem[WARPS_PER_BLOCK];
    const int warp_id_in_block = threadIdx.x >> 5;
    const int global_warp_id   = blockIdx.x * WARPS_PER_BLOCK + warp_id_in_block;
    if (global_warp_id >= n) return;

    bool v = range_verify_warp_precomp_w8_device(
        &proofs[global_warp_id], &commitments[global_warp_id],
        H_gen, &warp_smem[warp_id_in_block]);
    if ((threadIdx.x & 31) == 0 && !v)
        atomicAdd(fail_count, 1);
}

// ===== Phase profiling kernel: runs 1 proof, records per-phase cycle counts ===
__global__ void bench_bp_profile_p1par_kernel(
    const RangeProofGPU* proofs, const AffinePoint* commitments,
    const AffinePoint* H_gen, long long* d_phase_out)
{
    __shared__ BPWarpShared warp_smem[1];
    range_verify_warp_lut4_p1par_device(
        &proofs[0], &commitments[0], H_gen, &warp_smem[0]);
    if ((threadIdx.x & 31) == 0) {
        for (int i = 0; i < 6; ++i)
            d_phase_out[i] = warp_smem[0].phase_ts[i];
    }
}

// ===== Positional LUT4 warp verify kernel (zero doublings) ==================

__global__ void bench_bp_verify_warp_lut4_p1par_kernel(
    const RangeProofGPU* proofs, const AffinePoint* commitments,
    const AffinePoint* H_gen, int* fail_count, int n)
{
    __shared__ BPWarpShared warp_smem[WARPS_PER_BLOCK];
    const int warp_id_in_block = threadIdx.x >> 5;
    const int global_warp_id   = blockIdx.x * WARPS_PER_BLOCK + warp_id_in_block;
    if (global_warp_id >= n) return;

    bool v = range_verify_warp_lut4_p1par_device(
        &proofs[global_warp_id], &commitments[global_warp_id],
        H_gen, &warp_smem[warp_id_in_block]);
    if ((threadIdx.x & 31) == 0 && !v)
        atomicAdd(fail_count, 1);
}

__global__ void bench_bp_verify_warp_lut4_kernel(
    const RangeProofGPU* proofs, const AffinePoint* commitments,
    const AffinePoint* H_gen, int* fail_count, int n)
{
    __shared__ BPWarpShared warp_smem[WARPS_PER_BLOCK];
    const int warp_id_in_block = threadIdx.x >> 5;
    const int global_warp_id   = blockIdx.x * WARPS_PER_BLOCK + warp_id_in_block;
    if (global_warp_id >= n) return;

    bool v = range_verify_warp_lut4_device(
        &proofs[global_warp_id], &commitments[global_warp_id],
        H_gen, &warp_smem[warp_id_in_block]);
    if ((threadIdx.x & 31) == 0 && !v)
        atomicAdd(fail_count, 1);
}

// Setup: generate Pedersen H, commitments, and values for Bulletproof bench
__global__ void setup_pedersen_H_kernel(AffinePoint* H_out) {
    if (threadIdx.x != 0) return;
    // H = lift_x(SHA256("Pedersen_generator_H")) with try-and-increment
    const uint8_t tag[] = { 'P','e','d','e','r','s','e','n',
                            '_','g','e','n','e','r','a','t',
                            'o','r','_','H' };
    uint8_t hash[32];
    sha256_hash(tag, 20, hash);
    FieldElement x;
    field_from_bytes(hash, &x);
    hash_to_point_increment(&x, H_out);
}

__global__ void setup_bp_data_kernel(
    const AffinePoint* H_gen,
    uint64_t* values_out, Scalar* blindings_out, AffinePoint* commitments_out, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    // Value = idx + 1, blinding = make_key(idx)
    values_out[idx] = (uint64_t)(idx + 1);
    Scalar blind;
    blind.limbs[0] = (uint64_t)idx + 100;
    blind.limbs[1] = 0; blind.limbs[2] = 0; blind.limbs[3] = 0;
    blindings_out[idx] = blind;

    // Compute Pedersen commitment: C = value*H + blinding*G
    Scalar val_s;
    val_s.limbs[0] = values_out[idx]; val_s.limbs[1] = 0;
    val_s.limbs[2] = 0; val_s.limbs[3] = 0;
    JacobianPoint result;
    pedersen_commit_device(&val_s, &blind, H_gen, &result);
    FieldElement z_inv, z_inv2, z_inv3;
    field_inv(&result.z, &z_inv);
    field_sqr(&z_inv, &z_inv2);
    field_mul(&z_inv2, &z_inv, &z_inv3);
    field_mul(&result.x, &z_inv2, &commitments_out[idx].x);
    field_mul(&result.y, &z_inv3, &commitments_out[idx].y);
}

// ===== Timing helpers ========================================================

static double median_of(double* arr, int n) {
    for (int i = 1; i < n; i++) {
        double v = arr[i];
        int j = i - 1;
        while (j >= 0 && arr[j] > v) { arr[j+1] = arr[j]; j--; }
        arr[j+1] = v;
    }
    return (n & 1) ? arr[n/2] : (arr[n/2-1] + arr[n/2]) * 0.5;
}

template<typename F>
static double bench_kernel(F launch_fn, int warmup, int passes, int batch_size = BATCH) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < warmup; i++) {
        launch_fn();
        cudaDeviceSynchronize();
    }

    double times[32];
    for (int i = 0; i < passes; i++) {
        cudaEventRecord(start);
        launch_fn();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        times[i] = (double)ms * 1e6 / batch_size;  // ns per op
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return median_of(times, passes);
}

// ===== Main ==================================================================

int main() {
    printf("=== GPU ZK Benchmark (batch=%d, passes=%d) ===\n\n", BATCH, PASSES);
    fflush(stdout);

    int threads = 256;
    int blocks  = (BATCH + threads - 1) / threads;

    // -- Allocate --
    KnowledgeProofGPU* d_kproofs;
    DLEQProofGPU*      d_dproofs;
    AffinePoint*       d_pubs;
    AffinePoint*       d_Pa;
    AffinePoint*       d_Qa;
    AffinePoint*       d_Ga;
    AffinePoint*       d_Ha;
    JacobianPoint*     d_Gj;
    JacobianPoint*     d_Hj;
    int*               d_fail;

    cudaMalloc(&d_kproofs, BATCH * sizeof(KnowledgeProofGPU));
    cudaMalloc(&d_dproofs, BATCH * sizeof(DLEQProofGPU));
    cudaMalloc(&d_pubs,    BATCH * sizeof(AffinePoint));
    cudaMalloc(&d_Pa,      BATCH * sizeof(AffinePoint));
    cudaMalloc(&d_Qa,      BATCH * sizeof(AffinePoint));
    cudaMalloc(&d_Ga,      sizeof(AffinePoint));
    cudaMalloc(&d_Ha,      sizeof(AffinePoint));
    cudaMalloc(&d_Gj,      sizeof(JacobianPoint));
    cudaMalloc(&d_Hj,      sizeof(JacobianPoint));
    cudaMalloc(&d_fail,    sizeof(int));

    JacobianPoint*     d_pubs_jac;
    JacobianPoint*     d_Pj;
    JacobianPoint*     d_Qj;
    cudaMalloc(&d_pubs_jac, BATCH * sizeof(JacobianPoint));
    cudaMalloc(&d_Pj,       BATCH * sizeof(JacobianPoint));
    cudaMalloc(&d_Qj,       BATCH * sizeof(JacobianPoint));

    // -- Setup bases (Jacobian + Affine) --
    setup_bases_kernel<<<1,1>>>(d_Gj, d_Hj, d_Ga, d_Ha);
    cudaDeviceSynchronize();

    // -- Setup pubkeys --
    setup_pubkeys_kernel<<<blocks, threads>>>(d_pubs, BATCH);
    cudaDeviceSynchronize();

    // -- Setup DLEQ points (affine for verify) --
    setup_dleq_points_kernel<<<blocks, threads>>>(d_Gj, d_Hj, d_Pa, d_Qa, BATCH);
    cudaDeviceSynchronize();

    // -- Setup Jacobian pubkeys for prove benchmarks --
    setup_jac_pubkeys_kernel<<<blocks, threads>>>(d_pubs_jac, BATCH);
    cudaDeviceSynchronize();
    setup_dleq_jac_points_kernel<<<blocks, threads>>>(d_Gj, d_Hj, d_Pj, d_Qj, BATCH);
    cudaDeviceSynchronize();

    // -- Pre-generate proofs for verify benchmarks --
    bench_knowledge_prove_kernel<<<blocks, threads>>>(d_pubs_jac, d_kproofs, BATCH);
    cudaDeviceSynchronize();
    bench_dleq_prove_kernel<<<blocks, threads>>>(d_Gj, d_Hj, d_Pj, d_Qj, d_dproofs, BATCH);
    cudaDeviceSynchronize();

    // -- Correctness check --
    cudaMemset(d_fail, 0, sizeof(int));
    bench_knowledge_verify_kernel<<<blocks, threads>>>(d_kproofs, d_pubs, BATCH, d_fail);
    cudaDeviceSynchronize();
    int h_fail = 0;
    cudaMemcpy(&h_fail, d_fail, sizeof(int), cudaMemcpyDeviceToHost);
    if (h_fail != 0) {
        printf("FAIL: %d/%d knowledge_verify failures\n", h_fail, BATCH);
        return 1;
    }
    cudaMemset(d_fail, 0, sizeof(int));
    bench_dleq_verify_kernel<<<blocks, threads>>>(d_dproofs, d_Ga, d_Ha, d_Pa, d_Qa, BATCH, d_fail);
    cudaDeviceSynchronize();
    cudaMemcpy(&h_fail, d_fail, sizeof(int), cudaMemcpyDeviceToHost);
    if (h_fail != 0) {
        printf("FAIL: %d/%d dleq_verify failures\n", h_fail, BATCH);
        return 1;
    }

    printf("Correctness OK. Running benchmarks...\n\n");
    fflush(stdout);

    // ===== Benchmark =====

    double kp_ns = bench_kernel([&]() {
        bench_knowledge_prove_kernel<<<blocks, threads>>>(d_pubs_jac, d_kproofs, BATCH);
    }, WARMUP, PASSES);
    printf("  knowledge_prove(G)  : %8.1f ns/op  (%7.0f k/s)\n", kp_ns, 1e6/kp_ns);

    double kv_ns = bench_kernel([&]() {
        cudaMemset(d_fail, 0, sizeof(int));
        bench_knowledge_verify_kernel<<<blocks, threads>>>(d_kproofs, d_pubs, BATCH, d_fail);
    }, WARMUP, PASSES);
    printf("  knowledge_verify    : %8.1f ns/op  (%7.0f k/s)\n", kv_ns, 1e6/kv_ns);

    double dp_ns = bench_kernel([&]() {
        bench_dleq_prove_kernel<<<blocks, threads>>>(d_Gj, d_Hj, d_Pj, d_Qj, d_dproofs, BATCH);
    }, WARMUP, PASSES);
    printf("  dleq_prove          : %8.1f ns/op  (%7.0f k/s)\n", dp_ns, 1e6/dp_ns);

    double dv_ns = bench_kernel([&]() {
        cudaMemset(d_fail, 0, sizeof(int));
        bench_dleq_verify_kernel<<<blocks, threads>>>(d_dproofs, d_Ga, d_Ha, d_Pa, d_Qa, BATCH, d_fail);
    }, WARMUP, PASSES);
    printf("  dleq_verify         : %8.1f ns/op  (%7.0f k/s)\n", dv_ns, 1e6/dv_ns);

    // ===== Pedersen Commit =====
    printf("\n  -- Pedersen --\n");
    fflush(stdout);

    AffinePoint* d_H_ped;
    Scalar* d_ped_vals;
    Scalar* d_ped_blinds;
    AffinePoint* d_ped_commits;
    cudaMalloc(&d_H_ped, sizeof(AffinePoint));
    cudaMalloc(&d_ped_vals, BATCH * sizeof(Scalar));
    cudaMalloc(&d_ped_blinds, BATCH * sizeof(Scalar));
    cudaMalloc(&d_ped_commits, BATCH * sizeof(AffinePoint));

    setup_pedersen_H_kernel<<<1,1>>>(d_H_ped);
    cudaDeviceSynchronize();

    // Generate test values and blindings for Pedersen commit
    {
        std::vector<Scalar> h_vals(BATCH);
        std::vector<Scalar> h_blinds(BATCH);
        for (int i = 0; i < BATCH; ++i) {
            h_vals[i].limbs[0] = (uint64_t)(i + 1);
            h_vals[i].limbs[1] = 0; h_vals[i].limbs[2] = 0; h_vals[i].limbs[3] = 0;
            h_blinds[i].limbs[0] = (uint64_t)(i + 100);
            h_blinds[i].limbs[1] = 0; h_blinds[i].limbs[2] = 0; h_blinds[i].limbs[3] = 0;
        }
        cudaMemcpy(d_ped_vals, h_vals.data(), BATCH * sizeof(Scalar), cudaMemcpyHostToDevice);
        cudaMemcpy(d_ped_blinds, h_blinds.data(), BATCH * sizeof(Scalar), cudaMemcpyHostToDevice);
    }

    double ped_ns = bench_kernel([&]() {
        bench_pedersen_commit_kernel<<<blocks, threads>>>(
            d_ped_vals, d_ped_blinds, d_H_ped, d_ped_commits, BATCH);
    }, WARMUP, PASSES);
    printf("  pedersen_commit     : %8.1f ns/op  (%7.0f k/s)\n", ped_ns, 1e6/ped_ns);

    // ===== Bulletproof Range Proof =====
    printf("\n  -- Bulletproof (batch=%d) --\n", BP_BATCH);
    fflush(stdout);

    // Initialize Bulletproof generators
    bulletproof_init_kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
    printf("  Bulletproof generators initialized.\n");

    // Initialize precomputed tables for fixed-base acceleration
    bp_gen_table_init_w4_kernel<<<1, 128>>>();
    bp_gen_table_init_w8_kernel<<<1, 128>>>();
    bp_lut4_init_kernel<<<1, 128>>>();
    bp_h_lut4_init_kernel<<<1, 1>>>(d_H_ped);
    bp_g_lut4_init_kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
    printf("  Precomputed tables initialized (w4: 128 KB, w8: 2 MB, lut4: 8 MB, H-lut4: 64 KB, G-lut4: 64 KB).\n");
    fflush(stdout);

    int bp_blocks = (BP_BATCH + 64 - 1) / 64;  // fewer threads per block for heavy kernel
    int bp_threads = 64;

    uint64_t* d_bp_values;
    Scalar* d_bp_blindings;
    AffinePoint* d_bp_commits;
    RangeProofGPU* d_bp_proofs;
    bool* d_bp_results;
    cudaMalloc(&d_bp_values, BP_BATCH * sizeof(uint64_t));
    cudaMalloc(&d_bp_blindings, BP_BATCH * sizeof(Scalar));
    cudaMalloc(&d_bp_commits, BP_BATCH * sizeof(AffinePoint));
    cudaMalloc(&d_bp_proofs, BP_BATCH * sizeof(RangeProofGPU));
    cudaMalloc(&d_bp_results, BP_BATCH * sizeof(bool));

    setup_bp_data_kernel<<<bp_blocks, bp_threads>>>(
        d_H_ped, d_bp_values, d_bp_blindings, d_bp_commits, BP_BATCH);
    cudaDeviceSynchronize();

    // Prove (generate proofs)
    bench_bp_prove_kernel<<<bp_blocks, bp_threads>>>(
        d_bp_values, d_bp_blindings, d_bp_commits, d_H_ped,
        d_bp_proofs, d_bp_results, BP_BATCH);
    cudaDeviceSynchronize();

    // Check prove results
    {
        std::vector<unsigned char> h_results(BP_BATCH);
        cudaMemcpy(h_results.data(), d_bp_results, BP_BATCH * sizeof(bool), cudaMemcpyDeviceToHost);
        int prove_fails = 0;
        for (int i = 0; i < BP_BATCH; ++i) if (!h_results[i]) ++prove_fails;
        if (prove_fails > 0) {
            printf("  FAIL: %d/%d range_prove failures\n", prove_fails, BP_BATCH);
        }
    }

    // Verify correctness
    cudaMemset(d_fail, 0, sizeof(int));
    bench_bp_verify_kernel<<<bp_blocks, bp_threads>>>(
        d_bp_proofs, d_bp_commits, d_H_ped, d_fail, BP_BATCH);
    cudaDeviceSynchronize();
    cudaMemcpy(&h_fail, d_fail, sizeof(int), cudaMemcpyDeviceToHost);
    if (h_fail != 0) {
        printf("  FAIL: %d/%d range_verify failures\n", h_fail, BP_BATCH);
    } else {
        printf("  Bulletproof correctness OK.\n");
    }
    fflush(stdout);

    // Benchmark prove
    double bp_prove_ns = bench_kernel([&]() {
        bench_bp_prove_kernel<<<bp_blocks, bp_threads>>>(
            d_bp_values, d_bp_blindings, d_bp_commits, d_H_ped,
            d_bp_proofs, d_bp_results, BP_BATCH);
    }, WARMUP, PASSES, BP_BATCH);
    printf("  range_prove         : %8.1f ns/op  (%7.1f k/s)\n", bp_prove_ns, 1e6/bp_prove_ns);

    // Benchmark verify
    double bp_verify_ns = bench_kernel([&]() {
        cudaMemset(d_fail, 0, sizeof(int));
        bench_bp_verify_kernel<<<bp_blocks, bp_threads>>>(
            d_bp_proofs, d_bp_commits, d_H_ped, d_fail, BP_BATCH);
    }, WARMUP, PASSES, BP_BATCH);
    printf("  range_verify        : %8.1f ns/op  (%7.1f k/s)\n", bp_verify_ns, 1e6/bp_verify_ns);

    // ===== Warp-Cooperative Bulletproof Verify =====
    printf("\n  -- Bulletproof Warp-Cooperative (batch=%d) --\n", BP_BATCH);
    fflush(stdout);

    // Warp verify: 32 threads per proof, WARPS_PER_BLOCK warps per block
    int warp_threads_per_block = WARPS_PER_BLOCK * 32;  // 128
    int warp_blocks = (BP_BATCH + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

    // Correctness check: warp verify must match original
    cudaMemset(d_fail, 0, sizeof(int));
    bench_bp_verify_warp_kernel<<<warp_blocks, warp_threads_per_block>>>(
        d_bp_proofs, d_bp_commits, d_H_ped, d_fail, BP_BATCH);
    cudaDeviceSynchronize();
    cudaMemcpy(&h_fail, d_fail, sizeof(int), cudaMemcpyDeviceToHost);
    if (h_fail != 0) {
        printf("  FAIL: %d/%d warp_verify failures\n", h_fail, BP_BATCH);
    } else {
        printf("  Warp verify correctness OK.\n");
    }
    fflush(stdout);

    // Benchmark warp verify
    double bp_warp_verify_ns = bench_kernel([&]() {
        cudaMemset(d_fail, 0, sizeof(int));
        bench_bp_verify_warp_kernel<<<warp_blocks, warp_threads_per_block>>>(
            d_bp_proofs, d_bp_commits, d_H_ped, d_fail, BP_BATCH);
    }, WARMUP, PASSES, BP_BATCH);
    printf("  range_verify_warp   : %8.1f ns/op  (%7.1f k/s)\n", bp_warp_verify_ns, 1e6/bp_warp_verify_ns);
    printf("  warp vs single      :  %.1fx speedup\n", bp_verify_ns / bp_warp_verify_ns);

    // ===== Warp-Cooperative Bulletproof Prove =====
    printf("\n  -- Bulletproof Warp-Cooperative Prove (batch=%d) --\n", BP_BATCH);
    fflush(stdout);

    // Generate separate proofs for warp prover to avoid overwriting
    RangeProofGPU* d_bp_warp_proofs;
    bool* d_bp_warp_results;
    cudaMalloc(&d_bp_warp_proofs, BP_BATCH * sizeof(RangeProofGPU));
    cudaMalloc(&d_bp_warp_results, BP_BATCH * sizeof(bool));

    // Prove with warp kernel
    bench_bp_prove_warp_kernel<<<warp_blocks, warp_threads_per_block>>>(
        d_bp_values, d_bp_blindings, d_bp_commits, d_H_ped,
        d_bp_warp_proofs, d_bp_warp_results, BP_BATCH);
    cudaDeviceSynchronize();

    // Check warp prove results
    {
        std::vector<unsigned char> h_results(BP_BATCH);
        cudaMemcpy(h_results.data(), d_bp_warp_results, BP_BATCH * sizeof(bool), cudaMemcpyDeviceToHost);
        int prove_fails = 0;
        for (int i = 0; i < BP_BATCH; ++i) if (!h_results[i]) ++prove_fails;
        if (prove_fails > 0) {
            printf("  FAIL: %d/%d warp_prove failures\n", prove_fails, BP_BATCH);
        }
    }

    // Verify warp proofs with original verifier (cross-check)
    cudaMemset(d_fail, 0, sizeof(int));
    bench_bp_verify_kernel<<<bp_blocks, bp_threads>>>(
        d_bp_warp_proofs, d_bp_commits, d_H_ped, d_fail, BP_BATCH);
    cudaDeviceSynchronize();
    cudaMemcpy(&h_fail, d_fail, sizeof(int), cudaMemcpyDeviceToHost);
    if (h_fail != 0) {
        printf("  FAIL: %d/%d warp_prove->verify failures\n", h_fail, BP_BATCH);
    } else {
        printf("  Warp prove correctness OK (verified by original verifier).\n");
    }
    fflush(stdout);

    // Benchmark warp prove
    double bp_warp_prove_ns = bench_kernel([&]() {
        bench_bp_prove_warp_kernel<<<warp_blocks, warp_threads_per_block>>>(
            d_bp_values, d_bp_blindings, d_bp_commits, d_H_ped,
            d_bp_warp_proofs, d_bp_warp_results, BP_BATCH);
    }, WARMUP, PASSES, BP_BATCH);
    printf("  range_prove_warp    : %8.1f ns/op  (%7.1f k/s)\n", bp_warp_prove_ns, 1e6/bp_warp_prove_ns);
    printf("  warp vs single      :  %.1fx speedup\n", bp_prove_ns / bp_warp_prove_ns);

    // ===== Prove Warp + LUT4 (zero doublings for generators) =====
    printf("\n  -- Bulletproof Warp-Cooperative Prove LUT4 (batch=%d) --\n", BP_BATCH);
    fflush(stdout);

    // Prove with LUT4 kernel
    bench_bp_prove_warp_lut4_kernel<<<warp_blocks, warp_threads_per_block>>>(
        d_bp_values, d_bp_blindings, d_bp_commits, d_H_ped,
        d_bp_warp_proofs, d_bp_warp_results, BP_BATCH);
    cudaDeviceSynchronize();

    // Check prove results
    {
        std::vector<unsigned char> h_results(BP_BATCH);
        cudaMemcpy(h_results.data(), d_bp_warp_results, BP_BATCH * sizeof(bool), cudaMemcpyDeviceToHost);
        int prove_fails = 0;
        for (int i = 0; i < BP_BATCH; ++i) if (!h_results[i]) ++prove_fails;
        if (prove_fails > 0) {
            printf("  FAIL: %d/%d prove_lut4 failures\n", prove_fails, BP_BATCH);
        }
    }

    // Verify LUT4 proofs with original verifier (cross-check)
    cudaMemset(d_fail, 0, sizeof(int));
    bench_bp_verify_kernel<<<bp_blocks, bp_threads>>>(
        d_bp_warp_proofs, d_bp_commits, d_H_ped, d_fail, BP_BATCH);
    cudaDeviceSynchronize();
    cudaMemcpy(&h_fail, d_fail, sizeof(int), cudaMemcpyDeviceToHost);
    if (h_fail != 0) {
        printf("  FAIL: %d/%d prove_lut4->verify failures\n", h_fail, BP_BATCH);
    } else {
        printf("  Prove LUT4 correctness OK (verified by original verifier).\n");
    }
    fflush(stdout);

    // Benchmark prove LUT4
    double bp_prove_lut4_ns = bench_kernel([&]() {
        bench_bp_prove_warp_lut4_kernel<<<warp_blocks, warp_threads_per_block>>>(
            d_bp_values, d_bp_blindings, d_bp_commits, d_H_ped,
            d_bp_warp_proofs, d_bp_warp_results, BP_BATCH);
    }, WARMUP, PASSES, BP_BATCH);
    printf("  prove_warp_lut4     : %8.1f ns/op  (%7.1f k/s)\n", bp_prove_lut4_ns, 1e6/bp_prove_lut4_ns);
    printf("    vs warp_orig      :  %.2fx speedup\n", bp_warp_prove_ns / bp_prove_lut4_ns);
    printf("    vs single_orig    :  %.1fx total speedup\n", bp_prove_ns / bp_prove_lut4_ns);

    // ===== Precomputed Table Warp Verify (w=4 and w=8) =====
    printf("\n  -- Bulletproof Precomputed Table Verify (batch=%d) --\n", BP_BATCH);
    fflush(stdout);

    // Correctness: w=4 precomp must match original verifier
    cudaMemset(d_fail, 0, sizeof(int));
    bench_bp_verify_warp_precomp_w4_kernel<<<warp_blocks, warp_threads_per_block>>>(
        d_bp_proofs, d_bp_commits, d_H_ped, d_fail, BP_BATCH);
    cudaDeviceSynchronize();
    cudaMemcpy(&h_fail, d_fail, sizeof(int), cudaMemcpyDeviceToHost);
    if (h_fail != 0) {
        printf("  FAIL: %d/%d warp_precomp_w4 failures\n", h_fail, BP_BATCH);
    } else {
        printf("  Precomp w=4 correctness OK.\n");
    }
    fflush(stdout);

    // Correctness: w=8 precomp must match original verifier
    cudaMemset(d_fail, 0, sizeof(int));
    bench_bp_verify_warp_precomp_w8_kernel<<<warp_blocks, warp_threads_per_block>>>(
        d_bp_proofs, d_bp_commits, d_H_ped, d_fail, BP_BATCH);
    cudaDeviceSynchronize();
    cudaMemcpy(&h_fail, d_fail, sizeof(int), cudaMemcpyDeviceToHost);
    if (h_fail != 0) {
        printf("  FAIL: %d/%d warp_precomp_w8 failures\n", h_fail, BP_BATCH);
    } else {
        printf("  Precomp w=8 correctness OK.\n");
    }
    fflush(stdout);

    // Benchmark w=4 precomp
    double bp_precomp_w4_ns = bench_kernel([&]() {
        cudaMemset(d_fail, 0, sizeof(int));
        bench_bp_verify_warp_precomp_w4_kernel<<<warp_blocks, warp_threads_per_block>>>(
            d_bp_proofs, d_bp_commits, d_H_ped, d_fail, BP_BATCH);
    }, WARMUP, PASSES, BP_BATCH);
    printf("  verify_warp_w4      : %8.1f ns/op  (%7.1f k/s)\n", bp_precomp_w4_ns, 1e6/bp_precomp_w4_ns);
    printf("    vs warp_orig      :  %.2fx speedup\n", bp_warp_verify_ns / bp_precomp_w4_ns);

    // Benchmark w=8 precomp
    double bp_precomp_w8_ns = bench_kernel([&]() {
        cudaMemset(d_fail, 0, sizeof(int));
        bench_bp_verify_warp_precomp_w8_kernel<<<warp_blocks, warp_threads_per_block>>>(
            d_bp_proofs, d_bp_commits, d_H_ped, d_fail, BP_BATCH);
    }, WARMUP, PASSES, BP_BATCH);
    printf("  verify_warp_w8      : %8.1f ns/op  (%7.1f k/s)\n", bp_precomp_w8_ns, 1e6/bp_precomp_w8_ns);
    printf("    vs warp_orig      :  %.2fx speedup\n", bp_warp_verify_ns / bp_precomp_w8_ns);
    printf("    vs single_orig    :  %.1fx total speedup\n", bp_verify_ns / bp_precomp_w8_ns);

    // ===== Positional LUT4 Warp Verify (ZERO doublings) =====
    printf("\n  -- Bulletproof Positional LUT4 Verify (batch=%d) --\n", BP_BATCH);
    fflush(stdout);

    // Correctness: lut4 must match original verifier
    cudaMemset(d_fail, 0, sizeof(int));
    bench_bp_verify_warp_lut4_kernel<<<warp_blocks, warp_threads_per_block>>>(
        d_bp_proofs, d_bp_commits, d_H_ped, d_fail, BP_BATCH);
    cudaDeviceSynchronize();
    cudaMemcpy(&h_fail, d_fail, sizeof(int), cudaMemcpyDeviceToHost);
    if (h_fail != 0) {
        printf("  FAIL: %d/%d warp_lut4 failures\n", h_fail, BP_BATCH);
    } else {
        printf("  LUT4 correctness OK.\n");
    }
    fflush(stdout);

    // Benchmark lut4
    double bp_lut4_ns = bench_kernel([&]() {
        cudaMemset(d_fail, 0, sizeof(int));
        bench_bp_verify_warp_lut4_kernel<<<warp_blocks, warp_threads_per_block>>>(
            d_bp_proofs, d_bp_commits, d_H_ped, d_fail, BP_BATCH);
    }, WARMUP, PASSES, BP_BATCH);
    printf("  verify_warp_lut4    : %8.1f ns/op  (%7.1f k/s)\n", bp_lut4_ns, 1e6/bp_lut4_ns);
    printf("    vs warp_orig      :  %.2fx speedup\n", bp_warp_verify_ns / bp_lut4_ns);
    printf("    vs single_orig    :  %.1fx total speedup\n", bp_verify_ns / bp_lut4_ns);
    printf("    vs CPU            :  %.1fx\n", 2669843.0 / bp_lut4_ns);

    // ===== Phase 1 Parallel + LUT4 Verify =====
    printf("\n  -- Bulletproof Phase1-Parallel LUT4 Verify (batch=%d) --\n", BP_BATCH);
    fflush(stdout);

    // Correctness
    cudaMemset(d_fail, 0, sizeof(int));
    bench_bp_verify_warp_lut4_p1par_kernel<<<warp_blocks, warp_threads_per_block>>>(
        d_bp_proofs, d_bp_commits, d_H_ped, d_fail, BP_BATCH);
    cudaDeviceSynchronize();
    cudaMemcpy(&h_fail, d_fail, sizeof(int), cudaMemcpyDeviceToHost);
    if (h_fail != 0) {
        printf("  FAIL: %d/%d p1par_lut4 failures\n", h_fail, BP_BATCH);
    } else {
        printf("  P1par LUT4 correctness OK.\n");
    }
    fflush(stdout);

    // Benchmark
    double bp_p1par_ns = bench_kernel([&]() {
        cudaMemset(d_fail, 0, sizeof(int));
        bench_bp_verify_warp_lut4_p1par_kernel<<<warp_blocks, warp_threads_per_block>>>(
            d_bp_proofs, d_bp_commits, d_H_ped, d_fail, BP_BATCH);
    }, WARMUP, PASSES, BP_BATCH);
    printf("  verify_p1par_lut4   : %8.1f ns/op  (%7.1f k/s)\n", bp_p1par_ns, 1e6/bp_p1par_ns);
    printf("    vs lut4           :  %.2fx speedup\n", bp_lut4_ns / bp_p1par_ns);
    printf("    vs warp_orig      :  %.2fx speedup\n", bp_warp_verify_ns / bp_p1par_ns);
    printf("    vs single_orig    :  %.1fx total speedup\n", bp_verify_ns / bp_p1par_ns);
    printf("    vs CPU            :  %.1fx\n", 2669843.0 / bp_p1par_ns);

    // ===== Phase Profiling: P1a / P1b / P1c / P2 / P3 breakdown =====
    {
        printf("\n  -- Phase Profiling (p1par_lut4, single proof, 32 runs) --\n");
        fflush(stdout);
        long long* d_phase;
        cudaMalloc(&d_phase, 6 * sizeof(long long));
        constexpr int PROF_RUNS = 32;
        long long totals[5] = {0,0,0,0,0};
        for (int run = 0; run < PROF_RUNS + 4; ++run) {
            cudaMemset(d_phase, 0, 6 * sizeof(long long));
            bench_bp_profile_p1par_kernel<<<1, 32>>>(
                d_bp_proofs, d_bp_commits, d_H_ped, d_phase);
            cudaDeviceSynchronize();
            if (run < 4) continue; // warmup
            long long h_ts[6];
            cudaMemcpy(h_ts, d_phase, 6 * sizeof(long long), cudaMemcpyDeviceToHost);
            for (int p = 0; p < 5; ++p)
                totals[p] += (h_ts[p+1] - h_ts[p]);
        }
        long long total_all = 0;
        for (int p = 0; p < 5; ++p) total_all += totals[p];
        // Get GPU clock rate for ns conversion
        int dev; cudaGetDevice(&dev);
        int clock_khz;
        cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, dev);
        double clk_mhz = clock_khz / 1000.0;
        const char* phase_names[5] = {"P1a (Fiat-Shamir+IPA)", "P1b (s_coeff+inv+pow)", "P1c (22-lane scalar*P)", "P2  (LUT4 MSM 128gen)", "P3  (reduction+check)"};
        printf("  GPU clock: %.0f MHz\n", clk_mhz);
        printf("  %-24s %10s %8s %12s\n", "Phase", "Cycles", "%", "~ns (avg)");
        printf("  %-24s %10s %8s %12s\n", "------------------------", "----------", "--------", "------------");
        for (int p = 0; p < 5; ++p) {
            long long avg_cyc = totals[p] / PROF_RUNS;
            double pct = 100.0 * totals[p] / total_all;
            double ns = avg_cyc / clk_mhz * 1000.0;
            printf("  %-24s %10lld %7.1f%% %10.0f ns\n", phase_names[p], avg_cyc, pct, ns);
        }
        long long total_avg = total_all / PROF_RUNS;
        double total_ns = total_avg / clk_mhz * 1000.0;
        printf("  %-24s %10lld %7.1f%% %10.0f ns\n", "TOTAL", total_avg, 100.0, total_ns);
        cudaFree(d_phase);
        fflush(stdout);
    }

    // -- Summary with baseline comparison --
    printf("\n  %-22s %8s  %8s  %6s\n", "Operation", "Now", "Baseline", "Speedup");
    printf("  %-22s %8s  %8s  %6s\n", "----------------------", "--------", "--------", "------");
    printf("  %-22s %7.1f   %7.1f   %5.2fx\n", "knowledge_prove(G)", kp_ns, 384.3, 384.3/kp_ns);
    printf("  %-22s %7.1f   %7.1f   %5.2fx\n", "knowledge_verify",   kv_ns, 894.3, 894.3/kv_ns);
    printf("  %-22s %7.1f   %7.1f   %5.2fx\n", "dleq_prove",         dp_ns, 833.1, 833.1/dp_ns);
    printf("  %-22s %7.1f   %7.1f   %5.2fx\n", "dleq_verify",        dv_ns, 2148.2, 2148.2/dv_ns);
    printf("  %-22s %7.1f\n", "pedersen_commit", ped_ns);
    printf("  %-22s %7.1f\n", "range_prove", bp_prove_ns);
    printf("  %-22s %7.1f   (%.1fx vs single)\n", "range_prove_warp", bp_warp_prove_ns, bp_prove_ns / bp_warp_prove_ns);
    printf("  %-22s %7.1f   (%.2fx vs warp)\n", "prove_warp_lut4", bp_prove_lut4_ns, bp_warp_prove_ns / bp_prove_lut4_ns);
    printf("  %-22s %7.1f\n", "range_verify", bp_verify_ns);
    printf("  %-22s %7.1f   (%.1fx vs single)\n", "range_verify_warp", bp_warp_verify_ns, bp_verify_ns / bp_warp_verify_ns);
    printf("  %-22s %7.1f   (%.2fx vs warp)\n", "verify_warp_w4", bp_precomp_w4_ns, bp_warp_verify_ns / bp_precomp_w4_ns);
    printf("  %-22s %7.1f   (%.2fx vs warp)\n", "verify_warp_w8", bp_precomp_w8_ns, bp_warp_verify_ns / bp_precomp_w8_ns);
    printf("  %-22s %7.1f   (%.2fx vs warp)\n", "verify_warp_lut4", bp_lut4_ns, bp_warp_verify_ns / bp_lut4_ns);
    printf("  %-22s %7.1f   (%.2fx vs lut4)\n", "verify_p1par_lut4", bp_p1par_ns, bp_lut4_ns / bp_p1par_ns);

    printf("\n  -- GPU vs CPU (Bulletproof) --\n");
    printf("  %-22s %10s  %10s  %6s\n", "Operation", "GPU ns/op", "CPU ns/op", "Ratio");
    printf("  %-22s %10.1f  %10.0f  %5.1fx\n", "range_prove",  bp_prove_ns, 13618693.0, 13618693.0/bp_prove_ns);
    printf("  %-22s %10.1f  %10.0f  %5.1fx\n", "range_prove_warp",  bp_warp_prove_ns, 13618693.0, 13618693.0/bp_warp_prove_ns);
    printf("  %-22s %10.1f  %10.0f  %5.1fx\n", "prove_warp_lut4", bp_prove_lut4_ns, 13618693.0, 13618693.0/bp_prove_lut4_ns);
    printf("  %-22s %10.1f  %10.0f  %5.1fx\n", "range_verify", bp_verify_ns, 2669843.0, 2669843.0/bp_verify_ns);
    printf("  %-22s %10.1f  %10.0f  %5.1fx\n", "range_verify_warp", bp_warp_verify_ns, 2669843.0, 2669843.0/bp_warp_verify_ns);
    printf("  %-22s %10.1f  %10.0f  %5.1fx\n", "verify_warp_w4", bp_precomp_w4_ns, 2669843.0, 2669843.0/bp_precomp_w4_ns);
    printf("  %-22s %10.1f  %10.0f  %5.1fx\n", "verify_warp_w8", bp_precomp_w8_ns, 2669843.0, 2669843.0/bp_precomp_w8_ns);
    printf("  %-22s %10.1f  %10.0f  %5.1fx\n", "verify_warp_lut4", bp_lut4_ns, 2669843.0, 2669843.0/bp_lut4_ns);
    printf("  %-22s %10.1f  %10.0f  %5.1fx\n", "verify_p1par_lut4", bp_p1par_ns, 2669843.0, 2669843.0/bp_p1par_ns);
    printf("  %-22s %10.1f  %10.0f  %5.1fx\n", "pedersen_commit", ped_ns, 29718.0, 29718.0/ped_ns);

    printf("\n=== Done ===\n");

    cudaFree(d_kproofs);
    cudaFree(d_dproofs);
    cudaFree(d_pubs);
    cudaFree(d_Pa);
    cudaFree(d_Qa);
    cudaFree(d_Ga);
    cudaFree(d_Ha);
    cudaFree(d_Gj);
    cudaFree(d_Hj);
    cudaFree(d_fail);
    cudaFree(d_pubs_jac);
    cudaFree(d_Pj);
    cudaFree(d_Qj);
    cudaFree(d_H_ped);
    cudaFree(d_ped_vals);
    cudaFree(d_ped_blinds);
    cudaFree(d_ped_commits);
    cudaFree(d_bp_values);
    cudaFree(d_bp_blindings);
    cudaFree(d_bp_commits);
    cudaFree(d_bp_proofs);
    cudaFree(d_bp_results);
    cudaFree(d_bp_warp_proofs);
    cudaFree(d_bp_warp_results);
    return 0;
}
