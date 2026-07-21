// Metal SNARK witness readiness regression (GitHub #344).
// The source-coupled checks keep an uninitialised Metal runtime from being
// dereferenced before the backend can return GpuError::Device.

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

namespace {

int g_pass = 0;
int g_fail = 0;

void check(bool condition, const char* id, const char* description) {
    if (condition) {
        std::printf("[metal-snark-readiness] PASS %s: %s\n", id, description);
        ++g_pass;
    } else {
        std::printf("[metal-snark-readiness] FAIL %s: %s\n", id, description);
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
    const std::string marker = "audit/test_regression_metal_snark_readiness.cpp";
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
    const size_t end = source.find(next_signature, begin + std::char_traits<char>::length(signature));
    if (end == std::string::npos) return {};
    return source.substr(begin, end - begin);
}

bool readiness_guard_precedes_dispatch(const std::string& method) {
    const size_t ready = method.find("if (!is_ready())");
    const size_t device = method.find("GpuError::Device");
    const size_t zero_count = method.find("if (!count)");
    const size_t null_arg = method.find("GpuError::NullArg");
    const size_t runtime = method.find("runtime_->");

    return ready != std::string::npos &&
           device != std::string::npos &&
           zero_count != std::string::npos &&
           null_arg != std::string::npos &&
           runtime != std::string::npos &&
           ready < device && device < zero_count &&
           ready < null_arg && ready < runtime;
}

} // namespace

int test_regression_metal_snark_readiness_run() {
    g_pass = 0;
    g_fail = 0;
    std::printf("\n=== Metal SNARK Witness Readiness Regression ===\n");

    const std::string pre_fix =
        "GpuError snark_witness_batch() { if (!count) return GpuError::Ok; "
        "auto buffer = runtime_->alloc_buffer_shared(32); }";
    check(!readiness_guard_precedes_dispatch(pre_fix), "MSR-0",
          "the checker rejects the pre-fix unguarded runtime dereference");

    const std::string source = read_metal_backend();
    check(!source.empty(), "MSR-1", "Metal backend source is available to the audit");
    if (source.empty()) return 1;

    const std::string ecdsa = extract_method(
        source, "GpuError snark_witness_batch(",
        "GpuError schnorr_snark_witness_batch(");
    const std::string schnorr = extract_method(
        source, "GpuError schnorr_snark_witness_batch(",
        "GpuError bip352_scan_batch(");

    check(readiness_guard_precedes_dispatch(ecdsa), "MSR-2",
          "ECDSA witness generation rejects an uninitialised runtime before dispatch");
    check(readiness_guard_precedes_dispatch(schnorr), "MSR-3",
          "Schnorr witness generation rejects an uninitialised runtime before dispatch");

    std::printf("\n  %d passed  %d failed  (total %d)\n",
                g_pass, g_fail, g_pass + g_fail);
    return g_fail == 0 ? 0 : 1;
}

#ifdef STANDALONE_TEST
int main() {
    return test_regression_metal_snark_readiness_run();
}
#endif
