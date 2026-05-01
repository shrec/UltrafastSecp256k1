// =============================================================================
// UltrafastSecp256k1 OpenCL - Benchmark (Batch Throughput + Kernel-Only)
// =============================================================================
// Includes both batch dispatch (with buffer overhead) and kernel-only timing
// (matching CUDA cudaEvent methodology) for fair cross-platform comparison.
// =============================================================================

#include "secp256k1_opencl.hpp"
#include <CL/cl.h>
#include <iostream>
#include <chrono>
#include <vector>
#include <iomanip>
#include <string>
#include <cstdlib>
#include <random>
#include <fstream>
#include <sstream>
#include <filesystem>

using namespace secp256k1::opencl;

// Load a text file into a string
static std::string load_cl_source(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) return {};
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

struct BenchResult {
    std::string name;
    double ns_per_op;
    double ops_per_sec;
};

// Batch benchmark helper: warmup, then measure over multiple iterations
template<typename F>
BenchResult bench_batch(const std::string& name, F&& func, std::size_t batch_size, int warmup_iters = 3, int measure_iters = 10) {
    // Warmup
    for (int i = 0; i < warmup_iters; ++i) func();

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < measure_iters; ++i) {
        func();
    }
    auto end = std::chrono::high_resolution_clock::now();

    double total_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    double total_ops = static_cast<double>(batch_size) * measure_iters;
    double ns_per_op = total_ns / total_ops;
    double ops_per_sec = total_ops / (total_ns * 1e-9);

    return {name, ns_per_op, ops_per_sec};
}

void print_result(const BenchResult& r) {
    std::cout << "  " << std::left << std::setw(20) << r.name;
    if (r.ns_per_op < 1000.0) {
        std::cout << std::right << std::setw(10) << std::fixed << std::setprecision(1) << r.ns_per_op << " ns/op";
    } else if (r.ns_per_op < 1000000.0) {
        std::cout << std::right << std::setw(10) << std::fixed << std::setprecision(1) << (r.ns_per_op / 1000.0) << " us/op";
    } else {
        std::cout << std::right << std::setw(10) << std::fixed << std::setprecision(2) << (r.ns_per_op / 1000000.0) << " ms/op";
    }
    // Throughput
    if (r.ops_per_sec >= 1e9) {
        std::cout << "  (" << std::setprecision(2) << (r.ops_per_sec / 1e6) << " M/s)";
    } else if (r.ops_per_sec >= 1e6) {
        std::cout << "  (" << std::setprecision(2) << (r.ops_per_sec / 1e6) << " M/s)";
    } else if (r.ops_per_sec >= 1e3) {
        std::cout << "  (" << std::setprecision(0) << (r.ops_per_sec / 1e3) << " K/s)";
    } else {
        std::cout << "  (" << std::setprecision(0) << r.ops_per_sec << " /s)";
    }
    std::cout << "\n";
}

int main(int argc, char* argv[]) {
    std::cout << "UltrafastSecp256k1 OpenCL Benchmark (Batch Throughput)\n";
    std::cout << "======================================================\n\n";

    int platform_id = -1;
    int device_id = 0;
    bool prefer_intel = false; // Default to NVIDIA for benchmark
    std::size_t batch_size = 1048576; // 1M default (matches CUDA benchmark)

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--platform" && i + 1 < argc) {
            platform_id = std::atoi(argv[++i]);
        } else if (arg == "--device" && i + 1 < argc) {
            device_id = std::atoi(argv[++i]);
        } else if (arg == "--intel") {
            prefer_intel = true;
        } else if (arg == "--nvidia") {
            prefer_intel = false;
        } else if (arg == "--batch" && i + 1 < argc) {
            batch_size = std::atoi(argv[++i]);
        }
    }

    // Create context
    DeviceConfig config;
    config.verbose = true;
    config.prefer_intel = prefer_intel;
    if (platform_id >= 0) {
        config.platform_id = platform_id;
        config.device_id = device_id;
    } else {
        config.platform_id = -1;
    }

    auto ctx = Context::create(config);
    if (!ctx) {
        std::cout << "Failed to create OpenCL context\n";
        return 1;
    }

    const auto& info = ctx->device_info();
    std::cout << "\nDevice: " << info.name << " (" << info.vendor << ")\n";
    std::cout << "Compute Units: " << info.compute_units << "\n";
    std::cout << "Global Memory: " << (info.global_mem_size / (1024*1024)) << " MB\n";
    std::cout << "Batch Size: " << batch_size << "\n\n";

    std::vector<BenchResult> results;

    // ==========================================================================
    // Prepare random test data
    // ==========================================================================
    std::mt19937_64 rng(42);

    std::vector<FieldElement> fe_a(batch_size);
    std::vector<FieldElement> fe_b(batch_size);
    std::vector<FieldElement> fe_r(batch_size);
    for (std::size_t i = 0; i < batch_size; ++i) {
        fe_a[i] = {{rng(), rng(), rng(), rng()}};
        fe_b[i] = {{rng(), rng(), rng(), rng()}};
    }

    // ==========================================================================
    // Field Arithmetic Benchmarks (Batch)
    // ==========================================================================
    std::cout << "Field Arithmetic (batch=" << batch_size << "):\n";
    std::cout << std::string(50, '-') << "\n";

    {
        auto r = bench_batch("Field Add", [&]() {
            ctx->batch_field_add(fe_a.data(), fe_b.data(), fe_r.data(), batch_size);
        }, batch_size);
        print_result(r);
        results.push_back(r);
    }

    {
        auto r = bench_batch("Field Sub", [&]() {
            ctx->batch_field_sub(fe_a.data(), fe_b.data(), fe_r.data(), batch_size);
        }, batch_size);
        print_result(r);
        results.push_back(r);
    }

    {
        auto r = bench_batch("Field Mul", [&]() {
            ctx->batch_field_mul(fe_a.data(), fe_b.data(), fe_r.data(), batch_size);
        }, batch_size);
        print_result(r);
        results.push_back(r);
    }

    {
        auto r = bench_batch("Field Sqr", [&]() {
            ctx->batch_field_sqr(fe_a.data(), fe_r.data(), batch_size);
        }, batch_size);
        print_result(r);
        results.push_back(r);
    }

    {
        auto r = bench_batch("Field Inv", [&]() {
            ctx->batch_field_inv(fe_a.data(), fe_r.data(), batch_size);
        }, batch_size);
        print_result(r);
        results.push_back(r);
    }

    // ==========================================================================
    // Point Operation Benchmarks (Batch)
    // ==========================================================================
    std::cout << "\nPoint Operations (batch=" << batch_size << "):\n";
    std::cout << std::string(50, '-') << "\n";

    // Generate random points by scalar-mul with G
    std::vector<Scalar> scalars(batch_size);
    std::vector<JacobianPoint> jac_points(batch_size);
    std::vector<JacobianPoint> jac_points2(batch_size);
    std::vector<JacobianPoint> jac_results(batch_size);

    for (std::size_t i = 0; i < batch_size; ++i) {
        scalars[i] = {{rng(), rng(), rng(), rng()}};
    }

    // Pre-generate points for point_double and point_add benchmarks
    // Match CUDA benchmark batch size for fair comparison
    std::size_t point_batch = std::min(batch_size, static_cast<std::size_t>(262144));
    std::vector<Scalar> point_scalars(point_batch);
    std::vector<JacobianPoint> pd_in(point_batch), pd_out(point_batch);
    std::vector<JacobianPoint> pa_in1(point_batch), pa_in2(point_batch), pa_out(point_batch);
    std::vector<AffinePoint> sm_points(point_batch);

    for (std::size_t i = 0; i < point_batch; ++i) {
        point_scalars[i] = {{rng(), rng(), rng(), rng()}};
    }
    ctx->batch_scalar_mul_generator(point_scalars.data(), pd_in.data(), point_batch);

    for (std::size_t i = 0; i < point_batch; ++i) {
        point_scalars[i] = {{rng(), rng(), rng(), rng()}};
    }
    ctx->batch_scalar_mul_generator(point_scalars.data(), pa_in2.data(), point_batch);
    pa_in1 = pd_in;
    ctx->batch_jacobian_to_affine(pd_in.data(), sm_points.data(), point_batch);

    {
        auto r = bench_batch("Point Double", [&]() {
            ctx->batch_point_double(pd_in.data(), pd_out.data(), point_batch);
        }, point_batch);
        print_result(r);
        results.push_back(r);
    }

    {
        auto r = bench_batch("Point Add", [&]() {
            ctx->batch_point_add(pa_in1.data(), pa_in2.data(), pa_out.data(), point_batch);
        }, point_batch);
        print_result(r);
        results.push_back(r);
    }

    // ==========================================================================
    // Scalar Multiplication Benchmarks (various batch sizes)
    // ==========================================================================
    std::cout << "\nScalar Multiplication:\n";
    std::cout << std::string(50, '-') << "\n";

    for (std::size_t bs : {256UL, 1024UL, 4096UL, 16384UL, 65536UL}) {
        if (bs > batch_size) break;

        std::vector<Scalar> sm_scalars(bs);
        std::vector<JacobianPoint> sm_results(bs);
        for (std::size_t i = 0; i < bs; ++i) {
            sm_scalars[i] = {{rng(), rng(), rng(), rng()}};
        }

        std::string name = "kG (batch=" + std::to_string(bs) + ")";
        auto r = bench_batch(name, [&]() {
            ctx->batch_scalar_mul_generator(sm_scalars.data(), sm_results.data(), bs);
        }, bs, 1, 3);
        print_result(r);
        results.push_back(r);

        std::string kp_name = "kP (batch=" + std::to_string(bs) + ")";
        auto kp = bench_batch(kp_name, [&]() {
            ctx->batch_scalar_mul(sm_scalars.data(), sm_points.data(), sm_results.data(), bs);
        }, bs, 1, 3);
        print_result(kp);
        results.push_back(kp);
    }

    // ==========================================================================
    // Batch Inversion Benchmark (various batch sizes)
    // ==========================================================================
    std::cout << "\nBatch Field Inversion:\n";
    std::cout << std::string(50, '-') << "\n";

    for (std::size_t bs : {256UL, 1024UL, 4096UL, 16384UL}) {
        if (bs > batch_size) break;

        std::vector<FieldElement> inv_in(bs);
        std::vector<FieldElement> inv_out(bs);
        for (std::size_t i = 0; i < bs; ++i) {
            inv_in[i] = {{rng(), rng(), rng(), rng()}};
        }

        std::string name = "Inv (batch=" + std::to_string(bs) + ")";
        auto r = bench_batch(name, [&]() {
            ctx->batch_field_inv(inv_in.data(), inv_out.data(), bs);
        }, bs, 1, 3);
        print_result(r);
        results.push_back(r);
    }

    // ==========================================================================
    // KERNEL-ONLY Benchmarks (matching CUDA cudaEvent methodology)
    // Pre-allocate buffers, upload once, time only kernel launches
    // ==========================================================================
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "KERNEL-ONLY Timing (no buffer alloc/copy overhead):\n";
    std::cout << std::string(60, '=') << "\n";

    {
        cl_context cl_ctx = (cl_context)ctx->native_context();
        cl_command_queue cl_q = (cl_command_queue)ctx->native_queue();
        cl_int err;

        std::size_t ksz = batch_size;
        cl_uint kcnt = static_cast<cl_uint>(ksz);

        // Pre-allocate persistent buffers
        cl_mem buf_a = clCreateBuffer(cl_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       ksz * sizeof(FieldElement), (void*)fe_a.data(), &err);
        cl_mem buf_b = clCreateBuffer(cl_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       ksz * sizeof(FieldElement), (void*)fe_b.data(), &err);
        cl_mem buf_r = clCreateBuffer(cl_ctx, CL_MEM_WRITE_ONLY,
                                       ksz * sizeof(FieldElement), nullptr, &err);
        clFinish(cl_q);

        std::size_t local_sz = 256;
        std::size_t global_sz = ((ksz + local_sz - 1) / local_sz) * local_sz;

        int k_warmup = 5;
        int k_iters = 20;

        // Lambda for kernel-only bench
        auto kernel_bench = [&](const char* name, cl_kernel kern, bool two_inputs) -> BenchResult {
            clSetKernelArg(kern, 0, sizeof(cl_mem), &buf_a);
            if (two_inputs) {
                clSetKernelArg(kern, 1, sizeof(cl_mem), &buf_b);
                clSetKernelArg(kern, 2, sizeof(cl_mem), &buf_r);
                clSetKernelArg(kern, 3, sizeof(cl_uint), &kcnt);
            } else {
                clSetKernelArg(kern, 1, sizeof(cl_mem), &buf_r);
                clSetKernelArg(kern, 2, sizeof(cl_uint), &kcnt);
            }

            // Warmup
            for (int i = 0; i < k_warmup; ++i)
                clEnqueueNDRangeKernel(cl_q, kern, 1, nullptr, &global_sz, &local_sz, 0, nullptr, nullptr);
            clFinish(cl_q);

            // Measure
            auto t0 = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < k_iters; ++i)
                clEnqueueNDRangeKernel(cl_q, kern, 1, nullptr, &global_sz, &local_sz, 0, nullptr, nullptr);
            clFinish(cl_q);
            auto t1 = std::chrono::high_resolution_clock::now();

            double ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
            double total_ops = static_cast<double>(ksz) * k_iters;
            return {name, ns / total_ops, total_ops / (ns * 1e-9)};
        };

        std::cout << "\nField Arithmetic (kernel-only, batch=" << ksz << "):\n";
        std::cout << std::string(50, '-') << "\n";

        auto r_add = kernel_bench("Field Add", (cl_kernel)ctx->native_kernel("field_add"), true);
        print_result(r_add); results.push_back(r_add);

        auto r_sub = kernel_bench("Field Sub", (cl_kernel)ctx->native_kernel("field_sub"), true);
        print_result(r_sub); results.push_back(r_sub);

        auto r_mul = kernel_bench("Field Mul", (cl_kernel)ctx->native_kernel("field_mul"), true);
        print_result(r_mul); results.push_back(r_mul);

        auto r_sqr = kernel_bench("Field Sqr", (cl_kernel)ctx->native_kernel("field_sqr"), false);
        print_result(r_sqr); results.push_back(r_sqr);

        // Field Inv (kernel-only) -- single-input kernel like field_sqr
        auto r_inv = kernel_bench("Field Inv", (cl_kernel)ctx->native_kernel("field_inv"), false);
        print_result(r_inv); results.push_back(r_inv);

        // ==================================================================
        // Affine Point Addition kernel-only benchmarks
        // ==================================================================
        std::cout << "\nAffine Point Addition (kernel-only, batch=" << point_batch << "):\n";
        std::cout << std::string(50, '-') << "\n";

        {
            cl_uint aff_cnt = static_cast<cl_uint>(point_batch);
            std::size_t aff_global = ((point_batch + local_sz - 1) / local_sz) * local_sz;

            // Generate random affine point data (separate x,y arrays)
            std::vector<FieldElement> aff_px(point_batch), aff_py(point_batch);
            std::vector<FieldElement> aff_qx(point_batch), aff_qy(point_batch);
            std::vector<FieldElement> aff_hinv(point_batch);
            for (std::size_t i = 0; i < point_batch; ++i) {
                aff_px[i] = {{rng(), rng(), rng(), rng()}};
                aff_py[i] = {{rng(), rng(), rng(), rng()}};
                aff_qx[i] = {{rng(), rng(), rng(), rng()}};
                aff_qy[i] = {{rng(), rng(), rng(), rng()}};
                aff_hinv[i] = {{rng(), rng(), rng(), rng()}};
                aff_px[i].limbs[3] &= 0x7FFFFFFFFFFFFFFFULL;
                aff_py[i].limbs[3] &= 0x7FFFFFFFFFFFFFFFULL;
                aff_qx[i].limbs[3] &= 0x7FFFFFFFFFFFFFFFULL;
                aff_qy[i].limbs[3] &= 0x7FFFFFFFFFFFFFFFULL;
                aff_hinv[i].limbs[3] &= 0x7FFFFFFFFFFFFFFFULL;
            }

            std::size_t fe_sz = point_batch * sizeof(FieldElement);
            cl_mem buf_px = clCreateBuffer(cl_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                            fe_sz, (void*)aff_px.data(), &err);
            cl_mem buf_py = clCreateBuffer(cl_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                            fe_sz, (void*)aff_py.data(), &err);
            cl_mem buf_qx = clCreateBuffer(cl_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                            fe_sz, (void*)aff_qx.data(), &err);
            cl_mem buf_qy = clCreateBuffer(cl_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                            fe_sz, (void*)aff_qy.data(), &err);
            cl_mem buf_hinv = clCreateBuffer(cl_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                              fe_sz, (void*)aff_hinv.data(), &err);
            cl_mem buf_rx = clCreateBuffer(cl_ctx, CL_MEM_WRITE_ONLY, fe_sz, nullptr, &err);
            cl_mem buf_ry = clCreateBuffer(cl_ctx, CL_MEM_WRITE_ONLY, fe_sz, nullptr, &err);
            clFinish(cl_q);

            // Affine Add (2M + 1S + inv)
            {
                cl_kernel kern = (cl_kernel)ctx->native_kernel("affine_add");
                if (kern) {
                    clSetKernelArg(kern, 0, sizeof(cl_mem), &buf_px);
                    clSetKernelArg(kern, 1, sizeof(cl_mem), &buf_py);
                    clSetKernelArg(kern, 2, sizeof(cl_mem), &buf_qx);
                    clSetKernelArg(kern, 3, sizeof(cl_mem), &buf_qy);
                    clSetKernelArg(kern, 4, sizeof(cl_mem), &buf_rx);
                    clSetKernelArg(kern, 5, sizeof(cl_mem), &buf_ry);
                    clSetKernelArg(kern, 6, sizeof(cl_uint), &aff_cnt);

                    for (int i = 0; i < k_warmup; ++i)
                        clEnqueueNDRangeKernel(cl_q, kern, 1, nullptr, &aff_global, &local_sz, 0, nullptr, nullptr);
                    clFinish(cl_q);

                    auto t0 = std::chrono::high_resolution_clock::now();
                    for (int i = 0; i < k_iters; ++i)
                        clEnqueueNDRangeKernel(cl_q, kern, 1, nullptr, &aff_global, &local_sz, 0, nullptr, nullptr);
                    clFinish(cl_q);
                    auto t1 = std::chrono::high_resolution_clock::now();

                    double ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
                    double total_ops = static_cast<double>(point_batch) * k_iters;
                    BenchResult r = {"Affine Add (2M+1S+inv)", ns / total_ops, total_ops / (ns * 1e-9)};
                    print_result(r); results.push_back(r);
                } else {
                    std::cout << "  [SKIP] affine_add kernel not found\n";
                }
            }

            // Affine Lambda (2M + 1S, pre-inverted H)
            {
                cl_kernel kern = (cl_kernel)ctx->native_kernel("affine_add_lambda");
                if (kern) {
                    clSetKernelArg(kern, 0, sizeof(cl_mem), &buf_px);
                    clSetKernelArg(kern, 1, sizeof(cl_mem), &buf_py);
                    clSetKernelArg(kern, 2, sizeof(cl_mem), &buf_qx);
                    clSetKernelArg(kern, 3, sizeof(cl_mem), &buf_qy);
                    clSetKernelArg(kern, 4, sizeof(cl_mem), &buf_hinv);
                    clSetKernelArg(kern, 5, sizeof(cl_mem), &buf_rx);
                    clSetKernelArg(kern, 6, sizeof(cl_mem), &buf_ry);
                    clSetKernelArg(kern, 7, sizeof(cl_uint), &aff_cnt);

                    for (int i = 0; i < k_warmup; ++i)
                        clEnqueueNDRangeKernel(cl_q, kern, 1, nullptr, &aff_global, &local_sz, 0, nullptr, nullptr);
                    clFinish(cl_q);

                    auto t0 = std::chrono::high_resolution_clock::now();
                    for (int i = 0; i < k_iters; ++i)
                        clEnqueueNDRangeKernel(cl_q, kern, 1, nullptr, &aff_global, &local_sz, 0, nullptr, nullptr);
                    clFinish(cl_q);
                    auto t1 = std::chrono::high_resolution_clock::now();

                    double ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
                    double total_ops = static_cast<double>(point_batch) * k_iters;
                    BenchResult r = {"Affine Lambda (2M+1S)", ns / total_ops, total_ops / (ns * 1e-9)};
                    print_result(r); results.push_back(r);
                } else {
                    std::cout << "  [SKIP] affine_add_lambda kernel not found\n";
                }
            }

            // Affine X-Only (1M + 1S, pre-inverted H)
            {
                cl_kernel kern = (cl_kernel)ctx->native_kernel("affine_add_x_only");
                if (kern) {
                    clSetKernelArg(kern, 0, sizeof(cl_mem), &buf_px);
                    clSetKernelArg(kern, 1, sizeof(cl_mem), &buf_py);
                    clSetKernelArg(kern, 2, sizeof(cl_mem), &buf_qx);
                    clSetKernelArg(kern, 3, sizeof(cl_mem), &buf_qy);
                    clSetKernelArg(kern, 4, sizeof(cl_mem), &buf_hinv);
                    clSetKernelArg(kern, 5, sizeof(cl_mem), &buf_rx);
                    clSetKernelArg(kern, 6, sizeof(cl_uint), &aff_cnt);

                    for (int i = 0; i < k_warmup; ++i)
                        clEnqueueNDRangeKernel(cl_q, kern, 1, nullptr, &aff_global, &local_sz, 0, nullptr, nullptr);
                    clFinish(cl_q);

                    auto t0 = std::chrono::high_resolution_clock::now();
                    for (int i = 0; i < k_iters; ++i)
                        clEnqueueNDRangeKernel(cl_q, kern, 1, nullptr, &aff_global, &local_sz, 0, nullptr, nullptr);
                    clFinish(cl_q);
                    auto t1 = std::chrono::high_resolution_clock::now();

                    double ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
                    double total_ops = static_cast<double>(point_batch) * k_iters;
                    BenchResult r = {"Affine X-Only (1M+1S)", ns / total_ops, total_ops / (ns * 1e-9)};
                    print_result(r); results.push_back(r);
                } else {
                    std::cout << "  [SKIP] affine_add_x_only kernel not found\n";
                }
            }

            // Jacobian -> Affine conversion
            {
                cl_kernel kern = (cl_kernel)ctx->native_kernel("jacobian_to_affine");
                if (kern) {
                    // Reuse buf_px, buf_py as J.x, J.y; buf_qx as J.z
                    clSetKernelArg(kern, 0, sizeof(cl_mem), &buf_px);
                    clSetKernelArg(kern, 1, sizeof(cl_mem), &buf_py);
                    clSetKernelArg(kern, 2, sizeof(cl_mem), &buf_qx);
                    clSetKernelArg(kern, 3, sizeof(cl_mem), &buf_rx);
                    clSetKernelArg(kern, 4, sizeof(cl_mem), &buf_ry);
                    clSetKernelArg(kern, 5, sizeof(cl_uint), &aff_cnt);

                    for (int i = 0; i < k_warmup; ++i)
                        clEnqueueNDRangeKernel(cl_q, kern, 1, nullptr, &aff_global, &local_sz, 0, nullptr, nullptr);
                    clFinish(cl_q);

                    auto t0 = std::chrono::high_resolution_clock::now();
                    for (int i = 0; i < k_iters; ++i)
                        clEnqueueNDRangeKernel(cl_q, kern, 1, nullptr, &aff_global, &local_sz, 0, nullptr, nullptr);
                    clFinish(cl_q);
                    auto t1 = std::chrono::high_resolution_clock::now();

                    double ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
                    double total_ops = static_cast<double>(point_batch) * k_iters;
                    BenchResult r = {"Jac->Affine (per-pt)", ns / total_ops, total_ops / (ns * 1e-9)};
                    print_result(r); results.push_back(r);
                } else {
                    std::cout << "  [SKIP] jacobian_to_affine kernel not found\n";
                }
            }

            clReleaseMemObject(buf_px);
            clReleaseMemObject(buf_py);
            clReleaseMemObject(buf_qx);
            clReleaseMemObject(buf_qy);
            clReleaseMemObject(buf_hinv);
            clReleaseMemObject(buf_rx);
            clReleaseMemObject(buf_ry);
        }

        clReleaseMemObject(buf_a);
        clReleaseMemObject(buf_b);
        clReleaseMemObject(buf_r);

        // ==================================================================
        // Point / Scalar kernel-only benchmarks
        // ==================================================================
        std::cout << "\nPoint & Scalar (kernel-only, batch=" << point_batch << "):\n";
        std::cout << std::string(50, '-') << "\n";

        cl_uint pcnt = static_cast<cl_uint>(point_batch);
        std::size_t p_local_sz = 256;
        std::size_t p_global_sz = ((point_batch + p_local_sz - 1) / p_local_sz) * p_local_sz;

        // Upload pre-generated Jacobian points (pd_in, pa_in1, pa_in2 from batch section)
        cl_mem buf_jp1 = clCreateBuffer(cl_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                         point_batch * sizeof(JacobianPoint), (void*)pd_in.data(), &err);
        cl_mem buf_jp2 = clCreateBuffer(cl_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                         point_batch * sizeof(JacobianPoint), (void*)pa_in2.data(), &err);
        cl_mem buf_jpr = clCreateBuffer(cl_ctx, CL_MEM_WRITE_ONLY,
                                         point_batch * sizeof(JacobianPoint), nullptr, &err);
        clFinish(cl_q);

        // Point Double (kernel-only)
        {
            cl_kernel kern = (cl_kernel)ctx->native_kernel("point_double");
            clSetKernelArg(kern, 0, sizeof(cl_mem), &buf_jp1);
            clSetKernelArg(kern, 1, sizeof(cl_mem), &buf_jpr);
            clSetKernelArg(kern, 2, sizeof(cl_uint), &pcnt);

            for (int i = 0; i < k_warmup; ++i)
                clEnqueueNDRangeKernel(cl_q, kern, 1, nullptr, &p_global_sz, &p_local_sz, 0, nullptr, nullptr);
            clFinish(cl_q);

            auto t0 = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < k_iters; ++i)
                clEnqueueNDRangeKernel(cl_q, kern, 1, nullptr, &p_global_sz, &p_local_sz, 0, nullptr, nullptr);
            clFinish(cl_q);
            auto t1 = std::chrono::high_resolution_clock::now();

            double ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
            double total_ops = static_cast<double>(point_batch) * k_iters;
            BenchResult r = {"Point Double", ns / total_ops, total_ops / (ns * 1e-9)};
            print_result(r); results.push_back(r);
        }

        // Point Add (kernel-only)
        {
            cl_kernel kern = (cl_kernel)ctx->native_kernel("point_add");
            clSetKernelArg(kern, 0, sizeof(cl_mem), &buf_jp1);
            clSetKernelArg(kern, 1, sizeof(cl_mem), &buf_jp2);
            clSetKernelArg(kern, 2, sizeof(cl_mem), &buf_jpr);
            clSetKernelArg(kern, 3, sizeof(cl_uint), &pcnt);

            for (int i = 0; i < k_warmup; ++i)
                clEnqueueNDRangeKernel(cl_q, kern, 1, nullptr, &p_global_sz, &p_local_sz, 0, nullptr, nullptr);
            clFinish(cl_q);

            auto t0 = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < k_iters; ++i)
                clEnqueueNDRangeKernel(cl_q, kern, 1, nullptr, &p_global_sz, &p_local_sz, 0, nullptr, nullptr);
            clFinish(cl_q);
            auto t1 = std::chrono::high_resolution_clock::now();

            double ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
            double total_ops = static_cast<double>(point_batch) * k_iters;
            BenchResult r = {"Point Add", ns / total_ops, total_ops / (ns * 1e-9)};
            print_result(r); results.push_back(r);
        }

        // Scalar Mul Generator (kernel-only) -- smaller batch due to cost
        {
            std::size_t smk_batch = std::min(point_batch, static_cast<std::size_t>(65536));
            cl_uint smk_cnt = static_cast<cl_uint>(smk_batch);
            std::size_t smk_local_sz = std::min<std::size_t>(128, p_local_sz);
            std::size_t smk_global = ((smk_batch + smk_local_sz - 1) / smk_local_sz) * smk_local_sz;

            // Use existing point_scalars for scalar data
            cl_mem buf_sc = clCreateBuffer(cl_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                            smk_batch * sizeof(Scalar), (void*)point_scalars.data(), &err);
            cl_mem buf_smr = clCreateBuffer(cl_ctx, CL_MEM_WRITE_ONLY,
                                             smk_batch * sizeof(JacobianPoint), nullptr, &err);
            clFinish(cl_q);

            cl_kernel kern = (cl_kernel)ctx->native_kernel("scalar_mul_generator");
            clSetKernelArg(kern, 0, sizeof(cl_mem), &buf_sc);
            clSetKernelArg(kern, 1, sizeof(cl_mem), &buf_smr);
            clSetKernelArg(kern, 2, sizeof(cl_uint), &smk_cnt);

            int smk_warmup = 2;
            int smk_iters = 5;

            for (int i = 0; i < smk_warmup; ++i)
                clEnqueueNDRangeKernel(cl_q, kern, 1, nullptr, &smk_global, &smk_local_sz, 0, nullptr, nullptr);
            clFinish(cl_q);

            auto t0 = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < smk_iters; ++i)
                clEnqueueNDRangeKernel(cl_q, kern, 1, nullptr, &smk_global, &smk_local_sz, 0, nullptr, nullptr);
            clFinish(cl_q);
            auto t1 = std::chrono::high_resolution_clock::now();

            double ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
            double total_ops = static_cast<double>(smk_batch) * smk_iters;
            BenchResult r = {"kG (kernel)", ns / total_ops, total_ops / (ns * 1e-9)};
            print_result(r); results.push_back(r);

            clReleaseMemObject(buf_sc);
            clReleaseMemObject(buf_smr);
        }

        // Scalar Mul Arbitrary Point (kernel-only) -- same batch cap as kG
        {
            std::size_t smk_batch = std::min(point_batch, static_cast<std::size_t>(65536));
            cl_uint smk_cnt = static_cast<cl_uint>(smk_batch);
            std::size_t smk_local_sz = std::min<std::size_t>(128, p_local_sz);
            std::size_t smk_global = ((smk_batch + smk_local_sz - 1) / smk_local_sz) * smk_local_sz;

            std::vector<AffinePoint> sm_points(smk_batch);
            ctx->batch_jacobian_to_affine(pd_in.data(), sm_points.data(), smk_batch);

            cl_mem buf_sc = clCreateBuffer(cl_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                            smk_batch * sizeof(Scalar), (void*)point_scalars.data(), &err);
            cl_mem buf_pts = clCreateBuffer(cl_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                             smk_batch * sizeof(AffinePoint), (void*)sm_points.data(), &err);
            cl_mem buf_smr = clCreateBuffer(cl_ctx, CL_MEM_WRITE_ONLY,
                                             smk_batch * sizeof(JacobianPoint), nullptr, &err);
            clFinish(cl_q);

            cl_kernel kern = (cl_kernel)ctx->native_kernel("scalar_mul");
            clSetKernelArg(kern, 0, sizeof(cl_mem), &buf_sc);
            clSetKernelArg(kern, 1, sizeof(cl_mem), &buf_pts);
            clSetKernelArg(kern, 2, sizeof(cl_mem), &buf_smr);
            clSetKernelArg(kern, 3, sizeof(cl_uint), &smk_cnt);

            int smk_warmup = 2;
            int smk_iters = 5;

            for (int i = 0; i < smk_warmup; ++i)
                clEnqueueNDRangeKernel(cl_q, kern, 1, nullptr, &smk_global, &smk_local_sz, 0, nullptr, nullptr);
            clFinish(cl_q);

            auto t0 = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < smk_iters; ++i)
                clEnqueueNDRangeKernel(cl_q, kern, 1, nullptr, &smk_global, &smk_local_sz, 0, nullptr, nullptr);
            clFinish(cl_q);
            auto t1 = std::chrono::high_resolution_clock::now();

            double ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
            double total_ops = static_cast<double>(smk_batch) * smk_iters;
            BenchResult r = {"kP (kernel)", ns / total_ops, total_ops / (ns * 1e-9)};
            print_result(r); results.push_back(r);

            for (int i = 0; i < smk_warmup; ++i) {
                clEnqueueWriteBuffer(cl_q, buf_pts, CL_FALSE, 0,
                                     smk_batch * sizeof(AffinePoint), (void*)sm_points.data(), 0, nullptr, nullptr);
                clEnqueueWriteBuffer(cl_q, buf_sc, CL_FALSE, 0,
                                     smk_batch * sizeof(Scalar), (void*)point_scalars.data(), 0, nullptr, nullptr);
            }
            clFinish(cl_q);

            t0 = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < smk_iters; ++i) {
                clEnqueueWriteBuffer(cl_q, buf_pts, CL_FALSE, 0,
                                     smk_batch * sizeof(AffinePoint), (void*)sm_points.data(), 0, nullptr, nullptr);
                clEnqueueWriteBuffer(cl_q, buf_sc, CL_FALSE, 0,
                                     smk_batch * sizeof(Scalar), (void*)point_scalars.data(), 0, nullptr, nullptr);
            }
            clFinish(cl_q);
            t1 = std::chrono::high_resolution_clock::now();

            ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
            BenchResult upload = {"kP (upload)", ns / total_ops, total_ops / (ns * 1e-9)};
            print_result(upload); results.push_back(upload);

            clEnqueueNDRangeKernel(cl_q, kern, 1, nullptr, &smk_global, &smk_local_sz, 0, nullptr, nullptr);
            clFinish(cl_q);

            std::vector<JacobianPoint> sm_readback(smk_batch);
            for (int i = 0; i < smk_warmup; ++i)
                clEnqueueReadBuffer(cl_q, buf_smr, CL_FALSE, 0,
                                    smk_batch * sizeof(JacobianPoint), sm_readback.data(), 0, nullptr, nullptr);
            clFinish(cl_q);

            t0 = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < smk_iters; ++i)
                clEnqueueReadBuffer(cl_q, buf_smr, CL_FALSE, 0,
                                    smk_batch * sizeof(JacobianPoint), sm_readback.data(), 0, nullptr, nullptr);
            clFinish(cl_q);
            t1 = std::chrono::high_resolution_clock::now();

            ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
            BenchResult readback = {"kP (readback)", ns / total_ops, total_ops / (ns * 1e-9)};
            print_result(readback); results.push_back(readback);

            clReleaseMemObject(buf_sc);
            clReleaseMemObject(buf_pts);
            clReleaseMemObject(buf_smr);
        }

        clReleaseMemObject(buf_jp1);
        clReleaseMemObject(buf_jp2);
        clReleaseMemObject(buf_jpr);
    }

    // ==========================================================================
    // Signature Benchmarks (kernel-only: sign + verify with valid pre-computed data)
    // Loads secp256k1_extended.cl, compiles it, then benchmarks ECDSA+Schnorr.
    // ==========================================================================
    {
        cl_context  sig_ctx = (cl_context)ctx->native_context();
        cl_command_queue sig_q = (cl_command_queue)ctx->native_queue();
        cl_int sig_err;

        // Locate secp256k1_extended.cl (same search strategy as audit runner)
        namespace fs = std::filesystem;
        std::string ext_src;
        std::string found_kernel_dir;
        {
            auto exe_dir = fs::path(argv[0]).parent_path();
            std::vector<std::string> candidates = {
                (exe_dir / "kernels").string(),
                (exe_dir / "../kernels").string(),
                (exe_dir / "../../opencl/kernels").string(),
                "kernels",
                "../kernels",
                "../../opencl/kernels",
            };
            for (auto& c : candidates) {
                std::string p = c + "/secp256k1_extended.cl";
                ext_src = load_cl_source(p);
                if (!ext_src.empty()) { found_kernel_dir = c; break; }
            }
        }

        if (ext_src.empty()) {
            std::cout << "\n[SKIP] Cannot find secp256k1_extended.cl — signature benchmarks skipped\n";
        } else {
            // Compile the extended program
            cl_device_id sig_dev = nullptr;
            sig_err = clGetCommandQueueInfo(sig_q, CL_QUEUE_DEVICE, sizeof(cl_device_id), &sig_dev, nullptr);

            const char* src_ptr = ext_src.c_str();
            size_t src_len = ext_src.size();
            cl_program ext_prog = clCreateProgramWithSource(sig_ctx, 1, &src_ptr, &src_len, &sig_err);
            std::string build_opts = "-cl-std=CL1.2 -cl-fast-relaxed-math -cl-mad-enable -I " + found_kernel_dir;
            sig_err = clBuildProgram(ext_prog, 1, &sig_dev, build_opts.c_str(), nullptr, nullptr);
            if (sig_err != CL_SUCCESS) {
                char log[8192] = {};
                clGetProgramBuildInfo(ext_prog, sig_dev, CL_PROGRAM_BUILD_LOG, sizeof(log), log, nullptr);
                std::cout << "\n[SKIP] Extended CL build failed:\n" << log << "\n";
                clReleaseProgram(ext_prog);
            } else {
                cl_kernel k_ecdsa_sign   = clCreateKernel(ext_prog, "ecdsa_sign",   &sig_err);
                cl_kernel k_ecdsa_verify = clCreateKernel(ext_prog, "ecdsa_verify", &sig_err);
                cl_kernel k_schnorr_sign = clCreateKernel(ext_prog, "schnorr_sign", &sig_err);
                cl_kernel k_schnorr_verify = clCreateKernel(ext_prog, "schnorr_verify", &sig_err);

                if (!k_ecdsa_sign || !k_ecdsa_verify || !k_schnorr_sign || !k_schnorr_verify) {
                    std::cout << "\n[SKIP] Signature kernel creation failed\n";
                    if (k_ecdsa_sign)     clReleaseKernel(k_ecdsa_sign);
                    if (k_ecdsa_verify)   clReleaseKernel(k_ecdsa_verify);
                    if (k_schnorr_sign)   clReleaseKernel(k_schnorr_sign);
                    if (k_schnorr_verify) clReleaseKernel(k_schnorr_verify);
                    clReleaseProgram(ext_prog);
                } else {

                std::size_t sig_batch = std::min(point_batch, static_cast<std::size_t>(65536));
                cl_uint     sig_cnt   = static_cast<cl_uint>(sig_batch);
                std::size_t sig_lsz   = 128;
                std::size_t sig_gsz   = ((sig_batch + sig_lsz - 1) / sig_lsz) * sig_lsz;

                std::cout << "\n" << std::string(60, '=') << "\n";
                std::cout << "Signature Operations (kernel-only, batch=" << sig_batch << "):\n";
                std::cout << std::string(60, '-') << "\n";

                // Host struct layouts matching OpenCL kernel struct definitions exactly
                struct ECDSASigHost   { uint64_t r[4]; uint64_t s[4]; };   // 64 bytes
                struct SchnorrSigHost { uint8_t  r[32]; uint64_t s[4]; };  // 64 bytes

                // Random private keys, messages, and aux randoms
                std::vector<Scalar>  sv_priv(sig_batch);
                std::vector<uint8_t> sv_msg(sig_batch * 32);
                std::vector<uint8_t> sv_aux(sig_batch * 32);

                for (std::size_t i = 0; i < sig_batch; ++i) {
                    sv_priv[i] = {{rng(), rng(), rng(), rng()}};
                    sv_priv[i].limbs[3] &= 0x0FFFFFFFFFFFFFFFULL; // keep < 2^252
                    if (!sv_priv[i].limbs[0] && !sv_priv[i].limbs[1] &&
                        !sv_priv[i].limbs[2] && !sv_priv[i].limbs[3])
                        sv_priv[i].limbs[0] = 1;
                }
                for (auto& v : sv_msg) v = static_cast<uint8_t>(rng());
                for (auto& v : sv_aux) v = static_cast<uint8_t>(rng());

                // Compute pubkeys via batch kG
                std::vector<JacobianPoint> sv_pub_jac(sig_batch);
                ctx->batch_scalar_mul_generator(sv_priv.data(), sv_pub_jac.data(), sig_batch);
                std::vector<AffinePoint> sv_pub_aff(sig_batch);
                ctx->batch_jacobian_to_affine(sv_pub_jac.data(), sv_pub_aff.data(), sig_batch);

                // x-only pubkeys (big-endian) for Schnorr verify (BIP340)
                std::vector<uint8_t> sv_pub_x(sig_batch * 32);
                for (std::size_t i = 0; i < sig_batch; ++i) {
                    const auto& fe = sv_pub_aff[i].x;
                    for (int j = 0; j < 4; j++) {
                        uint64_t limb = fe.limbs[3 - j];
                        for (int k = 0; k < 8; k++)
                            sv_pub_x[i*32 + j*8 + k] = static_cast<uint8_t>(limb >> (56 - k*8));
                    }
                }

                // Device buffers
                cl_mem d_priv   = clCreateBuffer(sig_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                                  sig_batch * sizeof(Scalar), sv_priv.data(), &sig_err);
                cl_mem d_msg    = clCreateBuffer(sig_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                                  sig_batch * 32, sv_msg.data(), &sig_err);
                cl_mem d_aux    = clCreateBuffer(sig_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                                  sig_batch * 32, sv_aux.data(), &sig_err);
                cl_mem d_pub_jac = clCreateBuffer(sig_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                                   sig_batch * sizeof(JacobianPoint), sv_pub_jac.data(), &sig_err);
                cl_mem d_pub_x   = clCreateBuffer(sig_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                                   sig_batch * 32, sv_pub_x.data(), &sig_err);
                cl_mem d_ecdsa_sig   = clCreateBuffer(sig_ctx, CL_MEM_READ_WRITE,
                                                       sig_batch * sizeof(ECDSASigHost), nullptr, &sig_err);
                cl_mem d_schnorr_sig = clCreateBuffer(sig_ctx, CL_MEM_READ_WRITE,
                                                       sig_batch * sizeof(SchnorrSigHost), nullptr, &sig_err);
                cl_mem d_ok  = clCreateBuffer(sig_ctx, CL_MEM_WRITE_ONLY,
                                               sig_batch * sizeof(int), nullptr, &sig_err);
                cl_mem d_res = clCreateBuffer(sig_ctx, CL_MEM_WRITE_ONLY,
                                               sig_batch * sizeof(int), nullptr, &sig_err);
                clFinish(sig_q);

                // Pre-sign once so verify kernels have valid signatures
                clSetKernelArg(k_ecdsa_sign, 0, sizeof(cl_mem), &d_msg);
                clSetKernelArg(k_ecdsa_sign, 1, sizeof(cl_mem), &d_priv);
                clSetKernelArg(k_ecdsa_sign, 2, sizeof(cl_mem), &d_ecdsa_sig);
                clSetKernelArg(k_ecdsa_sign, 3, sizeof(cl_mem), &d_ok);
                clSetKernelArg(k_ecdsa_sign, 4, sizeof(cl_uint), &sig_cnt);
                clEnqueueNDRangeKernel(sig_q, k_ecdsa_sign, 1, nullptr, &sig_gsz, &sig_lsz, 0, nullptr, nullptr);

                clSetKernelArg(k_schnorr_sign, 0, sizeof(cl_mem), &d_msg);
                clSetKernelArg(k_schnorr_sign, 1, sizeof(cl_mem), &d_priv);
                clSetKernelArg(k_schnorr_sign, 2, sizeof(cl_mem), &d_aux);
                clSetKernelArg(k_schnorr_sign, 3, sizeof(cl_mem), &d_schnorr_sig);
                clSetKernelArg(k_schnorr_sign, 4, sizeof(cl_mem), &d_ok);
                clSetKernelArg(k_schnorr_sign, 5, sizeof(cl_uint), &sig_cnt);
                clEnqueueNDRangeKernel(sig_q, k_schnorr_sign, 1, nullptr, &sig_gsz, &sig_lsz, 0, nullptr, nullptr);
                clFinish(sig_q);

                int sw = 3, si = 8;

                // ECDSA Sign
                {
                    for (int i = 0; i < sw; ++i)
                        clEnqueueNDRangeKernel(sig_q, k_ecdsa_sign, 1, nullptr, &sig_gsz, &sig_lsz, 0, nullptr, nullptr);
                    clFinish(sig_q);
                    auto t0 = std::chrono::high_resolution_clock::now();
                    for (int i = 0; i < si; ++i)
                        clEnqueueNDRangeKernel(sig_q, k_ecdsa_sign, 1, nullptr, &sig_gsz, &sig_lsz, 0, nullptr, nullptr);
                    clFinish(sig_q);
                    auto t1 = std::chrono::high_resolution_clock::now();
                    double ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1-t0).count();
                    double ops = static_cast<double>(sig_batch) * si;
                    BenchResult r = {"ECDSA Sign", ns/ops, ops/(ns*1e-9)};
                    print_result(r); results.push_back(r);
                }

                // ECDSA Verify
                clSetKernelArg(k_ecdsa_verify, 0, sizeof(cl_mem), &d_msg);
                clSetKernelArg(k_ecdsa_verify, 1, sizeof(cl_mem), &d_pub_jac);
                clSetKernelArg(k_ecdsa_verify, 2, sizeof(cl_mem), &d_ecdsa_sig);
                clSetKernelArg(k_ecdsa_verify, 3, sizeof(cl_mem), &d_res);
                clSetKernelArg(k_ecdsa_verify, 4, sizeof(cl_uint), &sig_cnt);
                {
                    for (int i = 0; i < sw; ++i)
                        clEnqueueNDRangeKernel(sig_q, k_ecdsa_verify, 1, nullptr, &sig_gsz, &sig_lsz, 0, nullptr, nullptr);
                    clFinish(sig_q);
                    auto t0 = std::chrono::high_resolution_clock::now();
                    for (int i = 0; i < si; ++i)
                        clEnqueueNDRangeKernel(sig_q, k_ecdsa_verify, 1, nullptr, &sig_gsz, &sig_lsz, 0, nullptr, nullptr);
                    clFinish(sig_q);
                    auto t1 = std::chrono::high_resolution_clock::now();
                    double ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1-t0).count();
                    double ops = static_cast<double>(sig_batch) * si;
                    BenchResult r = {"ECDSA Verify", ns/ops, ops/(ns*1e-9)};
                    print_result(r); results.push_back(r);
                }

                // Schnorr Sign
                {
                    for (int i = 0; i < sw; ++i)
                        clEnqueueNDRangeKernel(sig_q, k_schnorr_sign, 1, nullptr, &sig_gsz, &sig_lsz, 0, nullptr, nullptr);
                    clFinish(sig_q);
                    auto t0 = std::chrono::high_resolution_clock::now();
                    for (int i = 0; i < si; ++i)
                        clEnqueueNDRangeKernel(sig_q, k_schnorr_sign, 1, nullptr, &sig_gsz, &sig_lsz, 0, nullptr, nullptr);
                    clFinish(sig_q);
                    auto t1 = std::chrono::high_resolution_clock::now();
                    double ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1-t0).count();
                    double ops = static_cast<double>(sig_batch) * si;
                    BenchResult r = {"Schnorr Sign", ns/ops, ops/(ns*1e-9)};
                    print_result(r); results.push_back(r);
                }

                // Schnorr Verify
                clSetKernelArg(k_schnorr_verify, 0, sizeof(cl_mem), &d_pub_x);
                clSetKernelArg(k_schnorr_verify, 1, sizeof(cl_mem), &d_msg);
                clSetKernelArg(k_schnorr_verify, 2, sizeof(cl_mem), &d_schnorr_sig);
                clSetKernelArg(k_schnorr_verify, 3, sizeof(cl_mem), &d_res);
                clSetKernelArg(k_schnorr_verify, 4, sizeof(cl_uint), &sig_cnt);
                {
                    for (int i = 0; i < sw; ++i)
                        clEnqueueNDRangeKernel(sig_q, k_schnorr_verify, 1, nullptr, &sig_gsz, &sig_lsz, 0, nullptr, nullptr);
                    clFinish(sig_q);
                    auto t0 = std::chrono::high_resolution_clock::now();
                    for (int i = 0; i < si; ++i)
                        clEnqueueNDRangeKernel(sig_q, k_schnorr_verify, 1, nullptr, &sig_gsz, &sig_lsz, 0, nullptr, nullptr);
                    clFinish(sig_q);
                    auto t1 = std::chrono::high_resolution_clock::now();
                    double ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1-t0).count();
                    double ops = static_cast<double>(sig_batch) * si;
                    BenchResult r = {"Schnorr Verify", ns/ops, ops/(ns*1e-9)};
                    print_result(r); results.push_back(r);
                }

                // Cleanup device buffers
                clReleaseMemObject(d_priv);
                clReleaseMemObject(d_msg);
                clReleaseMemObject(d_aux);
                clReleaseMemObject(d_pub_jac);
                clReleaseMemObject(d_pub_x);
                clReleaseMemObject(d_ecdsa_sig);
                clReleaseMemObject(d_schnorr_sig);
                clReleaseMemObject(d_ok);
                clReleaseMemObject(d_res);

                // Cleanup kernels + program
                clReleaseKernel(k_ecdsa_sign);
                clReleaseKernel(k_ecdsa_verify);
                clReleaseKernel(k_schnorr_sign);
                clReleaseKernel(k_schnorr_verify);
                clReleaseProgram(ext_prog);
                } // end kernel-check else
            }
        }
    }

    // ==========================================================================
    // ZK Proof Benchmarks (kernel-only: knowledge prove/verify + DLEQ prove/verify)
    // Loads secp256k1_zk.cl, runs batch with valid prove→verify data.
    // ==========================================================================
    {
        cl_context       zk_ctx = (cl_context)ctx->native_context();
        cl_command_queue zk_q   = (cl_command_queue)ctx->native_queue();
        cl_int zk_err;

        // Locate secp256k1_zk.cl (same search as signature section above)
        std::string zk_src;
        std::string zk_kdir;
        {
            auto exe_dir = std::filesystem::path(argv[0]).parent_path();
            std::vector<std::string> cands = {
                (exe_dir / "kernels").string(),
                (exe_dir / "../kernels").string(),
                (exe_dir / "../../opencl/kernels").string(),
                "kernels", "../kernels", "../../opencl/kernels",
            };
            for (auto& c : cands) {
                std::string p = c + "/secp256k1_zk.cl";
                zk_src = load_cl_source(p);
                if (!zk_src.empty()) { zk_kdir = c; break; }
            }
        }

        if (zk_src.empty()) {
            std::cout << "\n[SKIP] Cannot find secp256k1_zk.cl — ZK benchmarks skipped\n";
        } else {
            cl_device_id zk_dev = nullptr;
            zk_err = clGetCommandQueueInfo(zk_q, CL_QUEUE_DEVICE, sizeof(cl_device_id), &zk_dev, nullptr);

            const char* zk_ptr = zk_src.c_str();
            size_t zk_len = zk_src.size();
            cl_program zk_prog = clCreateProgramWithSource(zk_ctx, 1, &zk_ptr, &zk_len, &zk_err);
            std::string zk_opts = "-cl-std=CL1.2 -cl-fast-relaxed-math -cl-mad-enable -I " + zk_kdir;
            zk_err = clBuildProgram(zk_prog, 1, &zk_dev, zk_opts.c_str(), nullptr, nullptr);
            if (zk_err != CL_SUCCESS) {
                char log[16384] = {};
                clGetProgramBuildInfo(zk_prog, zk_dev, CL_PROGRAM_BUILD_LOG, sizeof(log), log, nullptr);
                std::cout << "\n[SKIP] ZK CL build failed:\n" << log << "\n";
                clReleaseProgram(zk_prog);
            } else {
                cl_kernel k_kp = clCreateKernel(zk_prog, "zk_knowledge_prove_batch",  &zk_err);
                cl_kernel k_kv = clCreateKernel(zk_prog, "zk_knowledge_verify_batch", &zk_err);
                cl_kernel k_dp = clCreateKernel(zk_prog, "zk_dleq_prove_batch",       &zk_err);
                cl_kernel k_dv = clCreateKernel(zk_prog, "zk_dleq_verify_batch",      &zk_err);

                if (!k_kp || !k_kv || !k_dp || !k_dv) {
                    std::cout << "\n[SKIP] ZK kernel creation failed\n";
                    if (k_kp) clReleaseKernel(k_kp);
                    if (k_kv) clReleaseKernel(k_kv);
                    if (k_dp) clReleaseKernel(k_dp);
                    if (k_dv) clReleaseKernel(k_dv);
                } else {
                    // Proof struct layouts (must match secp256k1_zk.cl)
                    struct ZKKnProofH  { uint8_t  rx[32]; uint64_t s[4]; };  // 64 bytes
                    struct ZKDLEQProfH { uint64_t e[4];   uint64_t s[4]; };  // 64 bytes

                    std::size_t zk_batch = std::min(point_batch, static_cast<std::size_t>(8192));
                    cl_uint     zk_cnt   = static_cast<cl_uint>(zk_batch);
                    std::size_t zk_lsz   = 128;
                    std::size_t zk_gsz   = ((zk_batch + zk_lsz - 1) / zk_lsz) * zk_lsz;

                    // --- Build valid prove inputs ---------------------------------
                    // secrets: small random scalars (non-zero, < 2^128 to stay below order)
                    std::vector<Scalar> zk_secrets(zk_batch);
                    for (std::size_t i = 0; i < zk_batch; ++i) {
                        uint64_t lo = (rng() & 0xFFFFFFFFFFFFFFFFULL);
                        if (!lo) lo = 1;
                        zk_secrets[i] = {{lo, rng() & 0xFFFFFFFFULL, 0, 0}};
                    }

                    // pubkeys[i] = secrets[i] * G  (knowledge: pubkey = secret * base, base=G)
                    std::vector<JacobianPoint> zk_pubkeys(zk_batch);
                    ctx->batch_scalar_mul_generator(zk_secrets.data(), zk_pubkeys.data(), zk_batch);

                    // base = G for all (1*G)
                    Scalar one_sc = {{1, 0, 0, 0}};
                    JacobianPoint g_jac;
                    ctx->batch_scalar_mul_generator(&one_sc, &g_jac, 1);
                    std::vector<JacobianPoint> zk_bases(zk_batch, g_jac);

                    // DLEQ: H = 5*G; P[i] = secrets[i]*G; Q[i] = secrets[i]*H
                    Scalar five_sc = {{5, 0, 0, 0}};
                    JacobianPoint h_jac;
                    ctx->batch_scalar_mul_generator(&five_sc, &h_jac, 1);
                    AffinePoint h_aff;
                    ctx->batch_jacobian_to_affine(&h_jac, &h_aff, 1);
                    std::vector<AffinePoint> h_aff_arr(zk_batch, h_aff);
                    std::vector<JacobianPoint> zk_Q(zk_batch);
                    ctx->batch_scalar_mul(zk_secrets.data(), h_aff_arr.data(), zk_Q.data(), zk_batch);

                    std::vector<JacobianPoint> zk_G(zk_batch, g_jac);
                    std::vector<JacobianPoint> zk_H(zk_batch, h_jac);

                    // Messages + aux randoms
                    std::vector<uint8_t> zk_msgs(zk_batch * 32);
                    std::vector<uint8_t> zk_aux(zk_batch * 32);
                    for (auto& v : zk_msgs) v = static_cast<uint8_t>(rng());
                    for (auto& v : zk_aux)  v = static_cast<uint8_t>(rng());

                    // Device buffers
                    cl_mem d_sec    = clCreateBuffer(zk_ctx, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
                                                      zk_batch*sizeof(Scalar), zk_secrets.data(), &zk_err);
                    cl_mem d_pub    = clCreateBuffer(zk_ctx, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
                                                      zk_batch*sizeof(JacobianPoint), zk_pubkeys.data(), &zk_err);
                    cl_mem d_base   = clCreateBuffer(zk_ctx, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
                                                      zk_batch*sizeof(JacobianPoint), zk_bases.data(), &zk_err);
                    cl_mem d_zmsg   = clCreateBuffer(zk_ctx, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
                                                      zk_batch*32, zk_msgs.data(), &zk_err);
                    cl_mem d_zaux   = clCreateBuffer(zk_ctx, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
                                                      zk_batch*32, zk_aux.data(), &zk_err);
                    cl_mem d_kproof = clCreateBuffer(zk_ctx, CL_MEM_READ_WRITE,
                                                      zk_batch*sizeof(ZKKnProofH), nullptr, &zk_err);
                    cl_mem d_dproof = clCreateBuffer(zk_ctx, CL_MEM_READ_WRITE,
                                                      zk_batch*sizeof(ZKDLEQProfH), nullptr, &zk_err);
                    cl_mem d_zok    = clCreateBuffer(zk_ctx, CL_MEM_WRITE_ONLY,
                                                      zk_batch*sizeof(int), nullptr, &zk_err);
                    cl_mem d_G      = clCreateBuffer(zk_ctx, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
                                                      zk_batch*sizeof(JacobianPoint), zk_G.data(), &zk_err);
                    cl_mem d_H      = clCreateBuffer(zk_ctx, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
                                                      zk_batch*sizeof(JacobianPoint), zk_H.data(), &zk_err);
                    cl_mem d_P      = clCreateBuffer(zk_ctx, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
                                                      zk_batch*sizeof(JacobianPoint), zk_pubkeys.data(), &zk_err);
                    cl_mem d_Q      = clCreateBuffer(zk_ctx, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
                                                      zk_batch*sizeof(JacobianPoint), zk_Q.data(), &zk_err);
                    clFinish(zk_q);

                    // Pre-run prove once so verify kernels have valid proofs
                    clSetKernelArg(k_kp, 0, sizeof(cl_mem), &d_sec);
                    clSetKernelArg(k_kp, 1, sizeof(cl_mem), &d_pub);
                    clSetKernelArg(k_kp, 2, sizeof(cl_mem), &d_base);
                    clSetKernelArg(k_kp, 3, sizeof(cl_mem), &d_zmsg);
                    clSetKernelArg(k_kp, 4, sizeof(cl_mem), &d_zaux);
                    clSetKernelArg(k_kp, 5, sizeof(cl_mem), &d_kproof);
                    clSetKernelArg(k_kp, 6, sizeof(cl_mem), &d_zok);
                    clSetKernelArg(k_kp, 7, sizeof(cl_uint), &zk_cnt);
                    clEnqueueNDRangeKernel(zk_q, k_kp, 1, nullptr, &zk_gsz, &zk_lsz, 0, nullptr, nullptr);

                    clSetKernelArg(k_dp, 0, sizeof(cl_mem), &d_sec);
                    clSetKernelArg(k_dp, 1, sizeof(cl_mem), &d_G);
                    clSetKernelArg(k_dp, 2, sizeof(cl_mem), &d_H);
                    clSetKernelArg(k_dp, 3, sizeof(cl_mem), &d_P);
                    clSetKernelArg(k_dp, 4, sizeof(cl_mem), &d_Q);
                    clSetKernelArg(k_dp, 5, sizeof(cl_mem), &d_zaux);
                    clSetKernelArg(k_dp, 6, sizeof(cl_mem), &d_dproof);
                    clSetKernelArg(k_dp, 7, sizeof(cl_mem), &d_zok);
                    clSetKernelArg(k_dp, 8, sizeof(cl_uint), &zk_cnt);
                    clEnqueueNDRangeKernel(zk_q, k_dp, 1, nullptr, &zk_gsz, &zk_lsz, 0, nullptr, nullptr);
                    clFinish(zk_q);

                    int zw = 3, zi = 8;
                    std::cout << "\n" << std::string(60, '=') << "\n";
                    std::cout << "ZK Proofs (kernel-only, batch=" << zk_batch << "):\n";
                    std::cout << std::string(60, '-') << "\n";

                    // Knowledge Prove
                    {
                        for (int i = 0; i < zw; ++i)
                            clEnqueueNDRangeKernel(zk_q, k_kp, 1, nullptr, &zk_gsz, &zk_lsz, 0, nullptr, nullptr);
                        clFinish(zk_q);
                        auto t0 = std::chrono::high_resolution_clock::now();
                        for (int i = 0; i < zi; ++i)
                            clEnqueueNDRangeKernel(zk_q, k_kp, 1, nullptr, &zk_gsz, &zk_lsz, 0, nullptr, nullptr);
                        clFinish(zk_q);
                        auto t1 = std::chrono::high_resolution_clock::now();
                        double ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1-t0).count();
                        double ops = static_cast<double>(zk_batch) * zi;
                        BenchResult r = {"ZK Kn Prove", ns/ops, ops/(ns*1e-9)};
                        print_result(r); results.push_back(r);
                    }

                    // Knowledge Verify
                    clSetKernelArg(k_kv, 0, sizeof(cl_mem), &d_kproof);
                    clSetKernelArg(k_kv, 1, sizeof(cl_mem), &d_pub);
                    clSetKernelArg(k_kv, 2, sizeof(cl_mem), &d_base);
                    clSetKernelArg(k_kv, 3, sizeof(cl_mem), &d_zmsg);
                    clSetKernelArg(k_kv, 4, sizeof(cl_mem), &d_zok);
                    clSetKernelArg(k_kv, 5, sizeof(cl_uint), &zk_cnt);
                    {
                        for (int i = 0; i < zw; ++i)
                            clEnqueueNDRangeKernel(zk_q, k_kv, 1, nullptr, &zk_gsz, &zk_lsz, 0, nullptr, nullptr);
                        clFinish(zk_q);
                        auto t0 = std::chrono::high_resolution_clock::now();
                        for (int i = 0; i < zi; ++i)
                            clEnqueueNDRangeKernel(zk_q, k_kv, 1, nullptr, &zk_gsz, &zk_lsz, 0, nullptr, nullptr);
                        clFinish(zk_q);
                        auto t1 = std::chrono::high_resolution_clock::now();
                        double ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1-t0).count();
                        double ops = static_cast<double>(zk_batch) * zi;
                        BenchResult r = {"ZK Kn Verify", ns/ops, ops/(ns*1e-9)};
                        print_result(r); results.push_back(r);
                    }

                    // DLEQ Prove
                    {
                        for (int i = 0; i < zw; ++i)
                            clEnqueueNDRangeKernel(zk_q, k_dp, 1, nullptr, &zk_gsz, &zk_lsz, 0, nullptr, nullptr);
                        clFinish(zk_q);
                        auto t0 = std::chrono::high_resolution_clock::now();
                        for (int i = 0; i < zi; ++i)
                            clEnqueueNDRangeKernel(zk_q, k_dp, 1, nullptr, &zk_gsz, &zk_lsz, 0, nullptr, nullptr);
                        clFinish(zk_q);
                        auto t1 = std::chrono::high_resolution_clock::now();
                        double ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1-t0).count();
                        double ops = static_cast<double>(zk_batch) * zi;
                        BenchResult r = {"ZK DLEQ Prove", ns/ops, ops/(ns*1e-9)};
                        print_result(r); results.push_back(r);
                    }

                    // DLEQ Verify
                    clSetKernelArg(k_dv, 0, sizeof(cl_mem), &d_dproof);
                    clSetKernelArg(k_dv, 1, sizeof(cl_mem), &d_G);
                    clSetKernelArg(k_dv, 2, sizeof(cl_mem), &d_H);
                    clSetKernelArg(k_dv, 3, sizeof(cl_mem), &d_P);
                    clSetKernelArg(k_dv, 4, sizeof(cl_mem), &d_Q);
                    clSetKernelArg(k_dv, 5, sizeof(cl_mem), &d_zok);
                    clSetKernelArg(k_dv, 6, sizeof(cl_uint), &zk_cnt);
                    {
                        for (int i = 0; i < zw; ++i)
                            clEnqueueNDRangeKernel(zk_q, k_dv, 1, nullptr, &zk_gsz, &zk_lsz, 0, nullptr, nullptr);
                        clFinish(zk_q);
                        auto t0 = std::chrono::high_resolution_clock::now();
                        for (int i = 0; i < zi; ++i)
                            clEnqueueNDRangeKernel(zk_q, k_dv, 1, nullptr, &zk_gsz, &zk_lsz, 0, nullptr, nullptr);
                        clFinish(zk_q);
                        auto t1 = std::chrono::high_resolution_clock::now();
                        double ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1-t0).count();
                        double ops = static_cast<double>(zk_batch) * zi;
                        BenchResult r = {"ZK DLEQ Verify", ns/ops, ops/(ns*1e-9)};
                        print_result(r); results.push_back(r);
                    }

                    // Cleanup
                    clReleaseMemObject(d_sec);    clReleaseMemObject(d_pub);
                    clReleaseMemObject(d_base);   clReleaseMemObject(d_zmsg);
                    clReleaseMemObject(d_zaux);   clReleaseMemObject(d_kproof);
                    clReleaseMemObject(d_dproof); clReleaseMemObject(d_zok);
                    clReleaseMemObject(d_G);      clReleaseMemObject(d_H);
                    clReleaseMemObject(d_P);      clReleaseMemObject(d_Q);
                    clReleaseKernel(k_kp); clReleaseKernel(k_kv);
                    clReleaseKernel(k_dp); clReleaseKernel(k_dv);
                }
                clReleaseProgram(zk_prog);
            }
        }
    }

    // ==========================================================================
    // Summary Table
    // ==========================================================================
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "Summary:\n";
    std::cout << std::string(60, '-') << "\n";
    for (const auto& r : results) {
        print_result(r);
    }
    std::cout << std::string(60, '=') << "\n";
    std::cout << "\nBenchmark complete!\n";
    return 0;
}
