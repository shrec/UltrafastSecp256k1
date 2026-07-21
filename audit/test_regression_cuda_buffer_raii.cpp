// CUDA batch allocation lifetime regression (GitHub #345).
// Source-coupled checks ensure CUDA_TRY early returns cannot leak device buffers.

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

namespace {

int g_pass = 0;
int g_fail = 0;

void check(bool condition, const char* id, const char* description) {
    if (condition) {
        std::printf("[cuda-buffer-raii] PASS %s: %s\n", id, description);
        ++g_pass;
    } else {
        std::printf("[cuda-buffer-raii] FAIL %s: %s\n", id, description);
        ++g_fail;
    }
}

std::string read_file(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) return {};
    std::ostringstream contents;
    contents << input.rdbuf();
    return contents.str();
}

std::string read_cuda_backend() {
    const std::string compiled_path = __FILE__;
    const std::string marker = "audit/test_regression_cuda_buffer_raii.cpp";
    const size_t marker_pos = compiled_path.rfind(marker);
    if (marker_pos != std::string::npos) {
        const std::string source = read_file(
            compiled_path.substr(0, marker_pos) + "src/gpu/src/gpu_backend_cuda.cu");
        if (!source.empty()) return source;
    }

    const char* candidates[] = {
        "src/gpu/src/gpu_backend_cuda.cu",
        "../src/gpu/src/gpu_backend_cuda.cu",
        "../../src/gpu/src/gpu_backend_cuda.cu",
        "libs/UltrafastSecp256k1/src/gpu/src/gpu_backend_cuda.cu",
        "../libs/UltrafastSecp256k1/src/gpu/src/gpu_backend_cuda.cu",
    };
    for (const char* path : candidates) {
        const std::string source = read_file(path);
        if (!source.empty()) return source;
    }
    return {};
}

std::string extract_method(const std::string& source,
                           const char* signature,
                           const char* next_signature) {
    const size_t begin = source.find(signature);
    if (begin == std::string::npos) return {};
    const size_t end = source.find(next_signature, begin + 1);
    if (end == std::string::npos) return {};
    return source.substr(begin, end - begin);
}

std::string trim(const std::string& value) {
    const size_t begin = value.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) return {};
    const size_t end = value.find_last_not_of(" \t\r\n");
    return value.substr(begin, end - begin + 1);
}

bool allocations_are_guarded(const std::string& method) {
    static const std::string allocation = "CUDA_TRY(cudaMalloc(&";
    size_t cursor = 0;
    int allocation_count = 0;

    while ((cursor = method.find(allocation, cursor)) != std::string::npos) {
        const size_t name_begin = cursor + allocation.size();
        const size_t comma = method.find(',', name_begin);
        const size_t statement_end = method.find(';', comma);
        if (comma == std::string::npos || statement_end == std::string::npos)
            return false;

        const std::string name = trim(method.substr(name_begin, comma - name_begin));
        const std::string guard =
            "CudaBufferGuard " + name + "_guard(" + name + ");";
        const size_t guard_pos = method.find(guard, statement_end + 1);
        const size_t next_cuda_try = method.find("CUDA_TRY(", statement_end + 1);
        if (name.empty() || guard_pos == std::string::npos ||
            (next_cuda_try != std::string::npos && guard_pos > next_cuda_try)) {
            return false;
        }

        ++allocation_count;
        cursor = statement_end + 1;
    }

    return allocation_count > 0 && method.find("cudaFree(") == std::string::npos;
}

bool guard_owns_cleanup(const std::string& source) {
    const size_t begin = source.find("struct CudaBufferGuard");
    const size_t end = source.find("namespace secp256k1", begin);
    if (begin == std::string::npos || end == std::string::npos) return false;
    const std::string guard = source.substr(begin, end - begin);
    return guard.find("~CudaBufferGuard()") != std::string::npos &&
           guard.find("if (ptr) cudaFree(ptr);") != std::string::npos &&
           guard.find("CudaBufferGuard(const CudaBufferGuard&) = delete") !=
               std::string::npos;
}

} // namespace

int test_regression_cuda_buffer_raii_run() {
    g_pass = 0;
    g_fail = 0;
    std::printf("\n=== CUDA Buffer RAII Regression ===\n");

    const std::string pre_fix =
        "CUDA_TRY(cudaMalloc(&d_a, 32));\n"
        "CUDA_TRY(cudaMalloc(&d_b, 32));\n"
        "CUDA_TRY(cudaMemcpy(d_a, in, 32, cudaMemcpyHostToDevice));\n"
        "cudaFree(d_b); cudaFree(d_a);";
    check(!allocations_are_guarded(pre_fix), "CBR-0",
          "the checker rejects the original success-only cleanup pattern");

    const std::string fixed_fixture =
        "CUDA_TRY(cudaMalloc(&d_a, 32));\n"
        "CudaBufferGuard d_a_guard(d_a);\n"
        "CUDA_TRY(cudaMalloc(&d_b, 32));\n"
        "CudaBufferGuard d_b_guard(d_b);\n"
        "CUDA_TRY(cudaMemcpy(d_a, in, 32, cudaMemcpyHostToDevice));";
    check(allocations_are_guarded(fixed_fixture), "CBR-1",
          "the checker accepts immediate RAII ownership after each allocation");

    const std::string source = read_cuda_backend();
    check(!source.empty(), "CBR-2", "CUDA backend source is available to the audit");
    if (source.empty()) return 1;

    check(guard_owns_cleanup(source), "CBR-3",
          "CudaBufferGuard is non-copyable and frees its owned allocation");

    const std::string ecdsa = extract_method(
        source, "GpuError ecdsa_verify_batch(", "GpuError ecdsa_verify_lbtc_rows(");
    const std::string schnorr = extract_method(
        source, "GpuError schnorr_verify_batch(", "static size_t lbtc_columns_chunk(");
    const std::string frost = extract_method(
        source, "GpuError frost_verify_partial_batch(", "GpuError ecrecover_batch(");
    const std::string ecdsa_snark = extract_method(
        source, "GpuError snark_witness_batch(", "GpuError schnorr_snark_witness_batch(");
    const std::string schnorr_snark = extract_method(
        source, "GpuError schnorr_snark_witness_batch(", "GpuError bip352_scan_batch(");

    check(allocations_are_guarded(ecdsa), "CBR-4",
          "ECDSA verify allocations are owned across CUDA_TRY early returns");
    check(allocations_are_guarded(schnorr), "CBR-5",
          "Schnorr verify allocations are owned across CUDA_TRY early returns");
    check(allocations_are_guarded(frost), "CBR-6",
          "FROST partial-verify allocations are owned across CUDA_TRY early returns");
    check(allocations_are_guarded(ecdsa_snark), "CBR-7",
          "ECDSA SNARK witness allocations are owned across CUDA_TRY early returns");
    check(allocations_are_guarded(schnorr_snark), "CBR-8",
          "Schnorr SNARK witness allocations are owned across CUDA_TRY early returns");

    std::printf("\n  %d passed  %d failed  (total %d)\n",
                g_pass, g_fail, g_pass + g_fail);
    return g_fail == 0 ? 0 : 1;
}

#ifdef STANDALONE_TEST
int main() {
    return test_regression_cuda_buffer_raii_run();
}
#endif
