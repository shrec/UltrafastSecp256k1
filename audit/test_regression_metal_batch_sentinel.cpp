// Metal generic batch-verification sentinel regression (GitHub #347).
// A command-buffer failure must become GpuError::Launch, never a verdict.

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

namespace {

int g_pass = 0;
int g_fail = 0;

void check(bool condition, const char* id, const char* description) {
    if (condition) {
        std::printf("[metal-batch-sentinel] PASS %s: %s\n", id, description);
        ++g_pass;
    } else {
        std::printf("[metal-batch-sentinel] FAIL %s: %s\n", id, description);
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

std::string read_metal_backend() {
    const std::string compiled_path = __FILE__;
    const std::string marker = "audit/test_regression_metal_batch_sentinel.cpp";
    const size_t marker_pos = compiled_path.rfind(marker);
    if (marker_pos != std::string::npos) {
        const std::string source = read_file(
            compiled_path.substr(0, marker_pos) + "src/gpu/src/gpu_backend_metal.mm");
        if (!source.empty()) return source;
    }

    const char* candidates[] = {
        "src/gpu/src/gpu_backend_metal.mm",
        "../src/gpu/src/gpu_backend_metal.mm",
        "../../src/gpu/src/gpu_backend_metal.mm",
        "libs/UltrafastSecp256k1/src/gpu/src/gpu_backend_metal.mm",
        "../libs/UltrafastSecp256k1/src/gpu/src/gpu_backend_metal.mm",
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

bool has_fatal_not_invalid_contract(const std::string& method) {
    const size_t clear = method.find("std::memset(out_results, 0, count);");
    const size_t memory_error = method.find("GpuError::Memory", clear);
    const size_t pipeline_check = method.find("if (!pipe.valid())", memory_error);
    const size_t seed = method.find("constexpr uint32_t kUnwritten = 0xFFFFFFFFu", pipeline_check);
    const size_t dispatch = method.find("runtime_->dispatch_sync", seed);
    const size_t detect = method.find("res[i] == kUnwritten", dispatch);
    const size_t launch_error = method.find("GpuError::Launch", detect);
    const size_t fallback = method.find("declining to CPU", launch_error);
    const size_t verdict = method.find("out_results[i] = res[i] ? 1 : 0", fallback);

    return clear != std::string::npos &&
           memory_error != std::string::npos &&
           pipeline_check != std::string::npos &&
           seed != std::string::npos &&
           dispatch != std::string::npos &&
           detect != std::string::npos &&
           launch_error != std::string::npos &&
           fallback != std::string::npos &&
           verdict != std::string::npos &&
           clear < memory_error && memory_error < pipeline_check &&
           pipeline_check < seed && seed < dispatch && dispatch < detect &&
           detect < launch_error && launch_error < fallback && fallback < verdict;
}

} // namespace

int test_regression_metal_batch_sentinel_run() {
    g_pass = 0;
    g_fail = 0;
    std::printf("\n=== Metal Batch Sentinel Regression ===\n");

    const std::string pre_fix =
        "auto buf_res = runtime_->alloc_buffer_shared(count * 4);\n"
        "runtime_->dispatch_sync(pipe, count, 64, {&buf_res});\n"
        "const auto* res = static_cast<const uint32_t*>(buf_res.contents());\n"
        "out_results[i] = res[i] ? 1 : 0;";
    check(!has_fatal_not_invalid_contract(pre_fix), "MBS-0",
          "the checker rejects the original unseeded result-buffer path");

    const std::string fixed_fixture =
        "std::memset(out_results, 0, count);\n"
        "return set_error(GpuError::Memory, \"alloc\");\n"
        "if (!pipe.valid()) return set_error(GpuError::Launch, \"pipe\");\n"
        "constexpr uint32_t kUnwritten = 0xFFFFFFFFu;\n"
        "runtime_->dispatch_sync(pipe, count, 64, {&buf_res});\n"
        "if (res[i] == kUnwritten) return set_error(GpuError::Launch, "
        "\"declining to CPU\");\n"
        "out_results[i] = res[i] ? 1 : 0;";
    check(has_fatal_not_invalid_contract(fixed_fixture), "MBS-1",
          "the checker accepts sentinel detection that returns an operational error");

    const std::string source = read_metal_backend();
    check(!source.empty(), "MBS-2", "Metal backend source is available to the audit");
    if (source.empty()) return 1;

    const std::string ecdsa = extract_method(
        source, "GpuError ecdsa_verify_batch(", "GpuError ecdsa_verify_lbtc_rows(");
    const std::string schnorr = extract_method(
        source, "GpuError schnorr_verify_batch(", "GpuError ecdsa_verify_lbtc_columns(");

    check(has_fatal_not_invalid_contract(ecdsa), "MBS-3",
          "Metal ECDSA batch converts unwritten output into GpuError::Launch");
    check(has_fatal_not_invalid_contract(schnorr), "MBS-4",
          "Metal Schnorr batch converts unwritten output into GpuError::Launch");

    std::printf("\n  %d passed  %d failed  (total %d)\n",
                g_pass, g_fail, g_pass + g_fail);
    return g_fail == 0 ? 0 : 1;
}

#ifdef STANDALONE_TEST
int main() {
    return test_regression_metal_batch_sentinel_run();
}
#endif
