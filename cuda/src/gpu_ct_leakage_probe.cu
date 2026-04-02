// ============================================================================
// GPU CT Leakage Probe -- fixed-vs-random device-cycle Welch t-test
// ============================================================================

#include "ct/ct_sign.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace secp256k1::cuda;

namespace {

constexpr int kDefaultSamples = 8192;
constexpr int kDefaultRepetitions = 8;
constexpr int kThreadsPerBlock = 128;
constexpr double kThreshold = 10.0;

__device__ static const uint8_t TEST_MSG[32] = {
    0x9f, 0x86, 0xd0, 0x81, 0x88, 0x4c, 0x7d, 0x65,
    0x9a, 0x2f, 0xea, 0xa0, 0xc5, 0x5a, 0xd0, 0x15,
    0xa3, 0xbf, 0x4f, 0x1b, 0x2b, 0x0b, 0x82, 0x2c,
    0xd1, 0x5d, 0x6c, 0x15, 0xb0, 0xf0, 0x0a, 0x08
};

__device__ static const uint8_t ZERO_AUX[32] = {0};

struct WelchState {
    double n[2] = {};
    double mean[2] = {};
    double m2[2] = {};

    void push(int cls, double x) {
        n[cls] += 1.0;
        const double delta = x - mean[cls];
        mean[cls] += delta / n[cls];
        const double delta2 = x - mean[cls];
        m2[cls] += delta * delta2;
    }

    double variance(int cls) const {
        return n[cls] > 1.0 ? (m2[cls] / (n[cls] - 1.0)) : 0.0;
    }

    double stddev(int cls) const {
        return std::sqrt(variance(cls));
    }

    double t_value() const {
        if (n[0] < 2.0 || n[1] < 2.0) return 0.0;
        const double se = std::sqrt(variance(0) / n[0] + variance(1) / n[1]);
        if (se < 1e-15) return 0.0;
        return (mean[0] - mean[1]) / se;
    }
};

struct ProbeResult {
    std::string name;
    WelchState stats;
    std::size_t failures = 0;
    bool pass = false;
};

static Scalar make_scalar(uint64_t limb0) {
    Scalar s{};
    s.limbs[0] = limb0;
    return s;
}

static Scalar fixed_scalar() {
    return make_scalar(1);
}

static Scalar random_scalar(std::mt19937_64& rng) {
    uint64_t v = 0;
    while (v == 0) {
        v = rng() & 0x7FFFFFFFFFFFFFFFULL;
    }
    return make_scalar(v);
}

static bool has_flag(char** begin, char** end, const char* flag) {
    for (char** it = begin; it != end; ++it) {
        if (std::strcmp(*it, flag) == 0) return true;
    }
    return false;
}

static int parse_int_arg(char** begin, char** end, const char* flag, int fallback) {
    for (char** it = begin; it != end; ++it) {
        if (std::strcmp(*it, flag) == 0 && (it + 1) != end) {
            return std::atoi(*(it + 1));
        }
    }
    return fallback;
}

static std::string parse_string_arg(char** begin, char** end, const char* flag, const char* fallback) {
    for (char** it = begin; it != end; ++it) {
        if (std::strcmp(*it, flag) == 0 && (it + 1) != end) {
            return *(it + 1);
        }
    }
    return fallback;
}

__global__ void warmup_ct_kernel(const Scalar* key) {
    JacobianPoint out{};
    ct::ct_generator_mul(key, &out);
}

__global__ void probe_ct_generator_mul_kernel(const Scalar* keys, uint64_t* cycles,
                                              uint8_t* status, int reps, int count) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    Scalar key = keys[idx];
    JacobianPoint out{};
    const uint64_t start = clock64();
    for (int i = 0; i < reps; ++i) {
        ct::ct_generator_mul(&key, &out);
    }
    const uint64_t end = clock64();

    cycles[idx] = end - start;
    status[idx] = (!out.infinity && ((out.x.limbs[0] | out.y.limbs[0] | out.z.limbs[0]) != 0)) ? 1 : 0;
}

__global__ void probe_ct_ecdsa_sign_kernel(const Scalar* keys, uint64_t* cycles,
                                           uint8_t* status, int reps, int count) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    Scalar key = keys[idx];
    ECDSASignatureGPU sig{};
    bool ok = true;
    const uint64_t start = clock64();
    for (int i = 0; i < reps; ++i) {
        ok = ok && ct::ct_ecdsa_sign(TEST_MSG, &key, &sig);
    }
    const uint64_t end = clock64();

    cycles[idx] = end - start;
    status[idx] = (ok && ((sig.r.limbs[0] | sig.s.limbs[0]) != 0)) ? 1 : 0;
}

__global__ void probe_ct_schnorr_sign_kernel(const Scalar* keys, uint64_t* cycles,
                                             uint8_t* status, int reps, int count) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    Scalar key = keys[idx];
    SchnorrSignatureGPU sig{};
    bool ok = true;
    const uint64_t start = clock64();
    for (int i = 0; i < reps; ++i) {
        ok = ok && ct::ct_schnorr_sign(&key, TEST_MSG, ZERO_AUX, &sig);
    }
    const uint64_t end = clock64();

    cycles[idx] = end - start;
    status[idx] = (ok && ((sig.r[0] | sig.s.limbs[0]) != 0)) ? 1 : 0;
}

template <typename Kernel>
static ProbeResult run_probe(const char* name, Kernel kernel, int samples, int repetitions,
                             const std::vector<Scalar>& host_keys, const std::vector<uint8_t>& host_classes) {
    ProbeResult result{};
    result.name = name;

    Scalar* d_keys = nullptr;
    uint64_t* d_cycles = nullptr;
    uint8_t* d_status = nullptr;
    cudaMalloc(&d_keys, sizeof(Scalar) * host_keys.size());
    cudaMalloc(&d_cycles, sizeof(uint64_t) * host_keys.size());
    cudaMalloc(&d_status, sizeof(uint8_t) * host_keys.size());

    cudaMemcpy(d_keys, host_keys.data(), sizeof(Scalar) * host_keys.size(), cudaMemcpyHostToDevice);
    warmup_ct_kernel<<<1, 1>>>(d_keys);
    cudaDeviceSynchronize();

    const int blocks = (samples + kThreadsPerBlock - 1) / kThreadsPerBlock;
    kernel<<<blocks, kThreadsPerBlock>>>(d_keys, d_cycles, d_status, repetitions, samples);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "%s launch error: %s\n", name, cudaGetErrorString(err));
        result.failures = host_keys.size();
        cudaFree(d_keys);
        cudaFree(d_cycles);
        cudaFree(d_status);
        return result;
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "%s sync error: %s\n", name, cudaGetErrorString(err));
        result.failures = host_keys.size();
        cudaFree(d_keys);
        cudaFree(d_cycles);
        cudaFree(d_status);
        return result;
    }

    std::vector<uint64_t> host_cycles(host_keys.size());
    std::vector<uint8_t> host_status(host_keys.size());
    cudaMemcpy(host_cycles.data(), d_cycles, sizeof(uint64_t) * host_cycles.size(), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_status.data(), d_status, sizeof(uint8_t) * host_status.size(), cudaMemcpyDeviceToHost);

    cudaFree(d_keys);
    cudaFree(d_cycles);
    cudaFree(d_status);

    for (std::size_t i = 0; i < host_cycles.size(); ++i) {
        if (!host_status[i]) {
            ++result.failures;
            continue;
        }
        result.stats.push(host_classes[i] ? 1 : 0,
                          static_cast<double>(host_cycles[i]) / static_cast<double>(repetitions));
    }

    result.pass = (result.failures == 0) && (std::abs(result.stats.t_value()) < kThreshold);
    return result;
}

static void write_json(FILE* fp, const cudaDeviceProp& props, int device_id,
                       int samples, int repetitions, const std::vector<ProbeResult>& results) {
    std::fprintf(fp,
                 "{\n"
                 "  \"device_id\": %d,\n"
                 "  \"device_name\": \"%s\",\n"
                 "  \"samples\": %d,\n"
                 "  \"repetitions\": %d,\n"
                 "  \"threshold_abs_t\": %.1f,\n"
                 "  \"results\": [\n",
                 device_id, props.name, samples, repetitions, kThreshold);

    for (std::size_t i = 0; i < results.size(); ++i) {
        const auto& r = results[i];
        std::fprintf(fp,
                     "    {\n"
                     "      \"name\": \"%s\",\n"
                     "      \"class0_samples\": %.0f,\n"
                     "      \"class1_samples\": %.0f,\n"
                     "      \"class0_mean_cycles\": %.6f,\n"
                     "      \"class1_mean_cycles\": %.6f,\n"
                     "      \"class0_stddev_cycles\": %.6f,\n"
                     "      \"class1_stddev_cycles\": %.6f,\n"
                     "      \"t_value\": %.6f,\n"
                     "      \"abs_t\": %.6f,\n"
                     "      \"failures\": %zu,\n"
                     "      \"pass\": %s\n"
                     "    }%s\n",
                     r.name.c_str(),
                     r.stats.n[0], r.stats.n[1],
                     r.stats.mean[0], r.stats.mean[1],
                     r.stats.stddev(0), r.stats.stddev(1),
                     r.stats.t_value(), std::abs(r.stats.t_value()),
                     r.failures, r.pass ? "true" : "false",
                     (i + 1 == results.size()) ? "" : ",");
    }

    std::fprintf(fp, "  ]\n}\n");
}

}  // namespace

int main(int argc, char** argv) {
    const bool json_only = has_flag(argv + 1, argv + argc, "--json-only");
    const int samples = parse_int_arg(argv + 1, argv + argc, "--samples", kDefaultSamples);
    const int repetitions = parse_int_arg(argv + 1, argv + argc, "--repetitions", kDefaultRepetitions);
    const int device_id = parse_int_arg(argv + 1, argv + argc, "--device", 0);
    const std::string report_dir = parse_string_arg(argv + 1, argv + argc, "--report-dir", ".");

    if (samples < 128 || repetitions < 1) {
        std::fprintf(stderr, "invalid samples/repetitions\n");
        return 2;
    }

    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "failed to select device %d: %s\n", device_id, cudaGetErrorString(err));
        return 2;
    }

    cudaDeviceProp props{};
    cudaGetDeviceProperties(&props, device_id);

    std::vector<Scalar> host_keys(samples);
    std::vector<uint8_t> host_classes(samples);
    std::mt19937_64 rng(0xC7A11A6E5EEDULL);
    for (int i = 0; i < samples; ++i) {
        const uint8_t cls = static_cast<uint8_t>(i & 1);
        host_classes[i] = cls;
        host_keys[i] = cls ? random_scalar(rng) : fixed_scalar();
    }

    std::vector<ProbeResult> results;
    results.push_back(run_probe("ct_generator_mul", probe_ct_generator_mul_kernel, samples, repetitions, host_keys, host_classes));
    results.push_back(run_probe("ct_ecdsa_sign", probe_ct_ecdsa_sign_kernel, samples, repetitions, host_keys, host_classes));
    results.push_back(run_probe("ct_schnorr_sign", probe_ct_schnorr_sign_kernel, samples, repetitions, host_keys, host_classes));

    const std::string report_path = report_dir + "/gpu_ct_leakage_report.json";
    FILE* report = std::fopen(report_path.c_str(), "w");
    if (report) {
        write_json(report, props, device_id, samples, repetitions, results);
        std::fclose(report);
    }

    if (json_only) {
        write_json(stdout, props, device_id, samples, repetitions, results);
    } else {
        std::printf("=== GPU CT Leakage Probe ===\n");
        std::printf("device: %s (id=%d)\n", props.name, device_id);
        std::printf("samples=%d repetitions=%d threshold=|t|<%.1f\n\n", samples, repetitions, kThreshold);
        for (const auto& r : results) {
            std::printf("%-18s |t|=%8.3f  mean0=%10.3f  mean1=%10.3f  fail=%zu  %s\n",
                        r.name.c_str(),
                        std::abs(r.stats.t_value()),
                        r.stats.mean[0],
                        r.stats.mean[1],
                        r.failures,
                        r.pass ? "[OK]" : "[LEAK?]");
        }
        std::printf("\nreport: %s\n", report_path.c_str());
    }

    for (const auto& r : results) {
        if (!r.pass) return 1;
    }
    return 0;
}
