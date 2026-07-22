// OpenCL collect dispatch/synchronisation regression (GitHub #346).
// Padded launches must remain bounds-safe and queue failures must not become verdicts.

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

namespace {

int g_pass = 0;
int g_fail = 0;

void check(bool condition, const char* id, const char* description) {
    if (condition) {
        std::printf("[opencl-collect-dispatch] PASS %s: %s\n", id, description);
        ++g_pass;
    } else {
        std::printf("[opencl-collect-dispatch] FAIL %s: %s\n", id, description);
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

std::string source_root() {
    const std::string compiled_path = __FILE__;
    const std::string marker = "audit/test_regression_opencl_collect_dispatch.cpp";
    const size_t marker_pos = compiled_path.rfind(marker);
    return marker_pos == std::string::npos ? std::string{} : compiled_path.substr(0, marker_pos);
}

std::string read_repo_file(const char* relative_path) {
    const std::string root = source_root();
    if (!root.empty()) {
        const std::string source = read_file(root + relative_path);
        if (!source.empty()) return source;
    }

    const std::string relative(relative_path);
    const std::string candidates[] = {
        relative,
        "../" + relative,
        "../../" + relative,
        "libs/UltrafastSecp256k1/" + relative,
        "../libs/UltrafastSecp256k1/" + relative,
    };
    for (const std::string& path : candidates) {
        const std::string source = read_file(path);
        if (!source.empty()) return source;
    }
    return {};
}

std::string extract_region(const std::string& source,
                           const char* begin_marker,
                           const char* end_marker) {
    const size_t begin = source.find(begin_marker);
    if (begin == std::string::npos) return {};
    const size_t end = source.find(end_marker, begin + 1);
    if (end == std::string::npos) return {};
    return source.substr(begin, end - begin);
}

bool has_checked_collect_contract(const std::string& method,
                                  const char* kernel_name) {
    const std::string local_pattern =
        std::string("const size_t local = lbtc_columns_local_size(") +
        kernel_name + ", queue);";
    const size_t local = method.find(local_pattern);
    const size_t arg_init = method.find("cl_int argerr = CL_SUCCESS;", local);
    const size_t arg_bind = method.find("argerr |= clSetKernelArg", arg_init);
    const size_t arg_check = method.find("if (argerr != CL_SUCCESS)", arg_bind);
    const size_t padded = method.find(
        "size_t global = lbtc_columns_padded_global(n, local);", arg_check);
    const size_t explicit_local = method.find("&global, &local", padded);
    const size_t finish = method.find("clerr = clFinish(queue);", explicit_local);
    const size_t finish_check = method.find("if (clerr != CL_SUCCESS)", finish);
    const size_t queue_error = method.find("GpuError::Queue", finish_check);
    const size_t readback = method.find("clEnqueueReadBuffer", queue_error);

    return local != std::string::npos &&
           arg_init != std::string::npos &&
           arg_bind != std::string::npos &&
           arg_check != std::string::npos &&
           padded != std::string::npos &&
           explicit_local != std::string::npos &&
           finish != std::string::npos &&
           finish_check != std::string::npos &&
           queue_error != std::string::npos &&
           readback != std::string::npos &&
           local < arg_init && arg_init < arg_bind && arg_bind < arg_check &&
           arg_check < padded && padded < explicit_local &&
           explicit_local < finish && finish < finish_check &&
           finish_check < queue_error && queue_error < readback;
}

bool has_count_guard(const std::string& kernel) {
    const size_t gid = kernel.find("uint gid = get_global_id(0);");
    const size_t guard = kernel.find("if (gid >= count) return;", gid);
    return gid != std::string::npos && guard != std::string::npos && gid < guard;
}

} // namespace

int test_regression_opencl_collect_dispatch_run() {
    g_pass = 0;
    g_fail = 0;
    std::printf("\n=== OpenCL Collect Dispatch Regression ===\n");

    const std::string pre_fix =
        "clSetKernelArg(ext_collect_, 0, sizeof(cl_mem), &d);\n"
        "size_t global = n;\n"
        "clEnqueueNDRangeKernel(queue, ext_collect_, 1, nullptr, &global, nullptr, 0, nullptr, nullptr);\n"
        "clFinish(queue);\n"
        "clEnqueueReadBuffer(queue, d, CL_TRUE, 0, n, out, 0, nullptr, nullptr);";
    check(!has_checked_collect_contract(pre_fix, "ext_collect_"), "OCD-0",
          "the checker rejects unchecked, driver-sized collect dispatch");

    const std::string fixed_fixture =
        "const size_t local = lbtc_columns_local_size(ext_collect_, queue);\n"
        "cl_int argerr = CL_SUCCESS;\n"
        "argerr |= clSetKernelArg(ext_collect_, 0, sizeof(cl_mem), &d);\n"
        "if (argerr != CL_SUCCESS) return GpuError::Launch;\n"
        "size_t global = lbtc_columns_padded_global(n, local);\n"
        "clEnqueueNDRangeKernel(queue, ext_collect_, 1, nullptr, &global, &local, 0, nullptr, nullptr);\n"
        "clerr = clFinish(queue);\n"
        "if (clerr != CL_SUCCESS) return GpuError::Queue;\n"
        "clEnqueueReadBuffer(queue, d, CL_TRUE, 0, n, out, 0, nullptr, nullptr);";
    check(has_checked_collect_contract(fixed_fixture, "ext_collect_"), "OCD-1",
          "the checker accepts padded dispatch with checked arg binding and queue sync");

    const std::string backend = read_repo_file("src/gpu/src/gpu_backend_opencl.cpp");
    const std::string kernels = read_repo_file("src/opencl/kernels/secp256k1_extended.cl");
    check(!backend.empty(), "OCD-2", "OpenCL backend source is available to the audit");
    check(!kernels.empty(), "OCD-3", "OpenCL collect kernel source is available to the audit");
    if (backend.empty() || kernels.empty()) return 1;

    const std::string ecdsa_method = extract_region(
        backend, "GpuError ecdsa_verify_collect(", "GpuError schnorr_verify_collect(");
    const std::string schnorr_method = extract_region(
        backend, "GpuError schnorr_verify_collect(", "libbitcoin-bridge PUBLIC-DATA ops");
    check(has_checked_collect_contract(ecdsa_method, "ext_ecdsa_lbtc_collect_"), "OCD-4",
          "ECDSA collect pads explicit-local dispatch and propagates queue failure");
    check(has_checked_collect_contract(schnorr_method, "ext_schnorr_lbtc_collect_"), "OCD-5",
          "Schnorr collect pads explicit-local dispatch and propagates queue failure");

    const std::string ecdsa_kernel = extract_region(
        kernels, "__kernel void ecdsa_verify_lbtc_collect(",
        "__kernel void schnorr_verify_lbtc_collect(");
    const std::string schnorr_kernel = extract_region(
        kernels, "__kernel void schnorr_verify_lbtc_collect(",
        "libbitcoin-bridge PUBLIC-DATA GpuBackend ops");
    check(has_count_guard(ecdsa_kernel), "OCD-6",
          "ECDSA collect rejects padded work-items beyond count");
    check(has_count_guard(schnorr_kernel), "OCD-7",
          "Schnorr collect rejects padded work-items beyond count");

    std::printf("\n  %d passed  %d failed  (total %d)\n",
                g_pass, g_fail, g_pass + g_fail);
    return g_fail == 0 ? 0 : 1;
}

#ifdef STANDALONE_TEST
int main() {
    return test_regression_opencl_collect_dispatch_run();
}
#endif
