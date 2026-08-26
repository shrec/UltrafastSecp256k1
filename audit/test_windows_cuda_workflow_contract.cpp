// Standalone contract regression for .github/workflows/windows-cuda.yml.
//
// This is a text-level contract check, not a full YAML grammar parser: it
// asserts the specific invariants that keep the mandatory Windows/MSVC CUDA
// job buildable (pinned toolkit version, Windows-valid sub-packages, fail-
// fast toolchain diagnostics, outputs confined to out/windows-cuda, and the
// libbitcoin-direct hook retention job kept real and unmasked). Each
// invariant is also proven to have teeth: a battery of single-point
// mutations is applied to a copy of the live file text and the corresponding
// check is asserted to flip from pass to fail.
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct Check {
    std::string name;
    bool passed;
    std::string detail;
};

bool locate_workflow_file(std::string& out_path) {
    namespace fs = std::filesystem;
    fs::path dir = fs::current_path();
    for (int i = 0; i < 10; ++i) {
        fs::path candidate = dir / ".github" / "workflows" / "windows-cuda.yml";
        std::error_code ec;
        if (fs::exists(candidate, ec)) {
            out_path = candidate.string();
            return true;
        }
        fs::path parent = dir.parent_path();
        if (parent.empty() || parent == dir) break;
        dir = parent;
    }
    return false;
}

bool read_file(const std::string& path, std::string& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    std::ostringstream ss;
    ss << f.rdbuf();
    out = ss.str();
    return true;
}

std::string replace_all(std::string s, const std::string& from, const std::string& to) {
    if (from.empty()) return s;
    size_t pos = 0;
    while ((pos = s.find(from, pos)) != std::string::npos) {
        s.replace(pos, from.size(), to);
        pos += to.size();
    }
    return s;
}

size_t count_occurrences(const std::string& s, const std::string& needle) {
    size_t count = 0, pos = 0;
    while ((pos = s.find(needle, pos)) != std::string::npos) {
        ++count;
        pos += needle.size();
    }
    return count;
}

std::string extract_diagnostics_block(const std::string& text) {
    size_t diag_pos = text.find("Fail-fast");
    if (diag_pos == std::string::npos) return std::string();
    size_t next_step = text.find("\n      - name:", diag_pos + 1);
    size_t diag_end = (next_step == std::string::npos) ? text.size() : next_step;
    return text.substr(diag_pos, diag_end - diag_pos);
}

std::string extract_cuda_version_line(const std::string& text) {
    size_t pos = text.find("\n          cuda: ");
    if (pos == std::string::npos) return std::string();
    size_t eol = text.find('\n', pos + 1);
    return text.substr(pos, (eol == std::string::npos ? text.size() : eol) - pos);
}

std::string extract_subpackages_line(const std::string& text) {
    size_t pos = text.find("sub-packages:");
    if (pos == std::string::npos) return std::string();
    size_t eol = text.find('\n', pos);
    return text.substr(pos, (eol == std::string::npos ? text.size() : eol) - pos);
}

std::string extract_job_header(const std::string& text) {
    size_t job_pos = text.find("\n  windows-cuda:");
    if (job_pos == std::string::npos) return std::string();
    size_t steps_pos = text.find("steps:", job_pos);
    if (steps_pos == std::string::npos) return text.substr(job_pos);
    return text.substr(job_pos, steps_pos - job_pos);
}

std::string extract_step_block(const std::string& text, const std::string& step_name_needle) {
    size_t name_pos = text.find(step_name_needle);
    if (name_pos == std::string::npos) return std::string();
    size_t step_start = text.rfind("\n      - name:", name_pos);
    if (step_start == std::string::npos) return std::string();
    size_t next_step = text.find("\n      - name:", step_start + 1);
    size_t step_end = (next_step == std::string::npos) ? text.size() : next_step;
    return text.substr(step_start, step_end - step_start);
}

// Isolates the single line that actually invokes the sm_89 compile probe
// (the "-arch=sm_89" nvcc invocation), as opposed to the whole diagnostics
// step, which also contains unrelated nvcc.exe existence/version checks.
std::string extract_sm89_probe_invocation(const std::string& text) {
    size_t pos = text.find("-arch=sm_89");
    if (pos == std::string::npos) return std::string();
    size_t line_start = text.rfind('\n', pos);
    line_start = (line_start == std::string::npos) ? 0 : line_start + 1;
    size_t line_end = text.find('\n', pos);
    return text.substr(line_start, (line_end == std::string::npos ? text.size() : line_end) - line_start);
}

// Isolates the single line containing an exact needle (used to pin down the
// real configure/build/test/probe command lines rather than scanning the
// whole file text for loose substrings).
std::string extract_line_containing(const std::string& text, const std::string& needle) {
    size_t pos = text.find(needle);
    if (pos == std::string::npos) return std::string();
    size_t line_start = text.rfind('\n', pos);
    line_start = (line_start == std::string::npos) ? 0 : line_start + 1;
    size_t line_end = text.find('\n', pos);
    return text.substr(line_start, (line_end == std::string::npos ? text.size() : line_end) - line_start);
}

// Reads the path token immediately following `flag` on `line` (e.g. "-B ",
// "--build ", "--test-dir ", "-o ", "/Fo:", "/Fe:"), stopping at whitespace
// or a batch-file line-continuation caret.
std::string extract_arg_after_flag(const std::string& line, const std::string& flag) {
    size_t pos = line.find(flag);
    if (pos == std::string::npos) return std::string();
    size_t start = pos + flag.size();
    size_t end = start;
    while (end < line.size() && line[end] != ' ' && line[end] != '\t' &&
           line[end] != '\r' && line[end] != '^') {
        ++end;
    }
    return line.substr(start, end - start);
}

std::string normalize_slashes(const std::string& path) {
    std::string out = path;
    for (auto& c : out) {
        if (c == '\\') c = '/';
    }
    return out;
}

bool contains_parent_segment(const std::string& normalized_path) {
    if (normalized_path == "..") return true;
    if (normalized_path.rfind("../", 0) == 0) return true;
    if (normalized_path.find("/../") != std::string::npos) return true;
    if (normalized_path.size() >= 3 &&
        normalized_path.compare(normalized_path.size() - 3, 3, "/..") == 0) {
        return true;
    }
    return false;
}

// A path is "rooted under out/windows-cuda" only if, after normalizing
// separators, it starts with that exact directory segment (not merely the
// substring, e.g. "out/windows-cuda-fake" must NOT match) and never escapes
// upward via a ".." segment anywhere in the path.
bool is_rooted_under_out_windows_cuda(const std::string& raw_path) {
    if (raw_path.empty()) return false;
    std::string p = normalize_slashes(raw_path);
    if (contains_parent_segment(p)) return false;
    static const std::string prefix = "out/windows-cuda";
    if (p.size() < prefix.size() || p.compare(0, prefix.size(), prefix) != 0) return false;
    if (p.size() > prefix.size() && p[prefix.size()] != '/') return false;
    return true;
}

std::vector<Check> validate_workflow_contract(const std::string& text) {
    std::vector<Check> checks;

    checks.push_back({
        "immutable_action_revision",
        text.find("Jimver/cuda-toolkit@3d45d157f327c09c04b50ee6ccdea2d9d017ec76") != std::string::npos,
        "Jimver/cuda-toolkit must stay pinned to its immutable commit SHA"
    });

    std::string cuda_line = extract_cuda_version_line(text);
    checks.push_back({
        "cuda_version_pin",
        cuda_line.find("12.8.1") != std::string::npos,
        "cuda toolkit input must pin the supported 12.8.1 version"
    });
    checks.push_back({
        "no_unsupported_cuda_version_drift",
        !cuda_line.empty() && cuda_line.find("13.2") == std::string::npos,
        "cuda toolkit input line must not drift to the unsupported 13.2 default"
    });

    std::string sub_line = extract_subpackages_line(text);
    bool has_nvcc = sub_line.find("\"nvcc\"") != std::string::npos;
    bool has_cudart = sub_line.find("\"cudart\"") != std::string::npos;
    bool has_thrust = sub_line.find("\"thrust\"") != std::string::npos;
    bool has_vsi = sub_line.find("\"visual_studio_integration\"") != std::string::npos;
    bool has_crt = sub_line.find("\"crt\"") != std::string::npos;
    checks.push_back({
        "subpackages_valid_no_crt",
        has_nvcc && has_cudart && has_thrust && has_vsi && !has_crt,
        "sub-packages must list only Windows-valid packages and must never include 'crt'"
    });

    std::string diag = extract_diagnostics_block(text);
    checks.push_back({"diagnostics_cuda_path", diag.find("CUDA_PATH") != std::string::npos,
                       "fail-fast diagnostics must check CUDA_PATH"});
    checks.push_back({"diagnostics_nvcc", diag.find("nvcc.exe") != std::string::npos,
                       "fail-fast diagnostics must check nvcc.exe"});
    checks.push_back({"diagnostics_cicc", diag.find("cicc.exe") != std::string::npos,
                       "fail-fast diagnostics must check cicc.exe"});
    checks.push_back({"diagnostics_ptxas", diag.find("ptxas.exe") != std::string::npos,
                       "fail-fast diagnostics must check ptxas.exe"});
    checks.push_back({"diagnostics_nvcc_profile", diag.find("nvcc.profile") != std::string::npos,
                       "fail-fast diagnostics must check nvcc.profile"});
    checks.push_back({"diagnostics_cl_exe", diag.find("cl.exe") != std::string::npos,
                       "fail-fast diagnostics must check cl.exe"});
    std::string sm89_probe_line = extract_sm89_probe_invocation(text);
    checks.push_back({
        "diagnostics_sm89_probe",
        diag.find("sm_89") != std::string::npos && diag.find(".cu") != std::string::npos &&
            !sm89_probe_line.empty() && sm89_probe_line.find("nvcc.exe") != std::string::npos,
        "fail-fast diagnostics must run a real sm_89 CUDA compile probe invoking nvcc.exe on that exact line"
    });

    std::string configure_line = extract_line_containing(text, "cmake -S . -B ");
    std::string build_line = extract_line_containing(text, "cmake --build ");
    std::string test_line = extract_line_containing(text, "--test-dir ");
    std::string probe_line = extract_sm89_probe_invocation(text);
    std::string cl_output_line = extract_line_containing(text, "/Fo:");

    std::string configure_path = extract_arg_after_flag(configure_line, "-B ");
    std::string build_path = extract_arg_after_flag(build_line, "--build ");
    std::string test_path = extract_arg_after_flag(test_line, "--test-dir ");
    std::string probe_output_path = extract_arg_after_flag(probe_line, "-o ");
    std::string fo_path = extract_arg_after_flag(cl_output_line, "/Fo:");
    std::string fe_path = extract_arg_after_flag(cl_output_line, "/Fe:");

    bool all_paths_present = !configure_path.empty() && !build_path.empty() && !test_path.empty() &&
                              !probe_output_path.empty() && !fo_path.empty() && !fe_path.empty();
    bool all_paths_rooted = all_paths_present &&
        is_rooted_under_out_windows_cuda(configure_path) &&
        is_rooted_under_out_windows_cuda(build_path) &&
        is_rooted_under_out_windows_cuda(test_path) &&
        is_rooted_under_out_windows_cuda(probe_output_path) &&
        is_rooted_under_out_windows_cuda(fo_path) &&
        is_rooted_under_out_windows_cuda(fe_path);
    checks.push_back({
        "outputs_under_out_windows_cuda",
        all_paths_rooted,
        "the actual configure (-B), build (--build), ctest (--test-dir), sm_89 probe (-o) and "
        "cl.exe (/Fo:, /Fe:) command-line paths must each be rooted exactly under out/windows-cuda, "
        "never a sibling out-level dir or a parent-relative escape"
    });

    checks.push_back({
        "no_continue_on_error_masking",
        text.find("continue-on-error") == std::string::npos,
        "no step may mask failures with continue-on-error"
    });

    std::string job_header = extract_job_header(text);
    bool job_not_gated = job_header.find("\n    if:") == std::string::npos;
    checks.push_back({
        "hook_job_mandatory",
        text.find("test_lbtc_direct_gpu_columns_hook") != std::string::npos && job_not_gated,
        "the libbitcoin-direct hook retention build/test must stay wired and the job must not be conditionally skipped"
    });

    std::string build_step = extract_step_block(text, "Build (engine, GPU host, kernels, macro fixture, hook test)");
    std::string test_step = extract_step_block(text, "Test (macro fixture + hook self-install retention)");
    bool build_not_masked = !build_step.empty() && build_step.find("\n        if:") == std::string::npos;
    bool test_not_masked = !test_step.empty() && test_step.find("\n        if:") == std::string::npos;
    checks.push_back({
        "hook_steps_not_step_level_masked",
        build_not_masked && test_not_masked,
        "the build and test steps that run the libbitcoin-direct hook retention check must not carry a step-level if: condition"
    });

    return checks;
}

bool check_passed(const std::vector<Check>& checks, const std::string& name) {
    for (const auto& c : checks) {
        if (c.name == name) return c.passed;
    }
    return false;
}

}  // namespace

int test_windows_cuda_workflow_contract_run() {
    std::string path;
    if (!locate_workflow_file(path)) {
        std::cerr << "[windows-cuda-workflow-contract] FAIL: could not locate "
                     ".github/workflows/windows-cuda.yml from cwd\n";
        return 1;
    }

    std::string text;
    if (!read_file(path, text)) {
        std::cerr << "[windows-cuda-workflow-contract] FAIL: could not read " << path << "\n";
        return 1;
    }

    int failures = 0;

    std::vector<Check> live_checks = validate_workflow_contract(text);
    for (const auto& c : live_checks) {
        if (!c.passed) {
            std::cerr << "[windows-cuda-workflow-contract] FAIL(live): " << c.name << " -- " << c.detail << "\n";
            ++failures;
        }
    }

    // Mutation battery: each entry proves the corresponding check has teeth
    // by flipping one specific invariant and asserting the checker notices.
    struct Mutation {
        std::string label;
        std::string mutated_text;
        std::string must_fail_check;
        bool setup_ok;
    };

    std::vector<Mutation> mutations;

    {
        std::string from = "cuda: '12.8.1'";
        size_t pos = text.find(from);
        bool ok = pos != std::string::npos;
        std::string mutated = text;
        if (ok) mutated.replace(pos, from.size(), "cuda: '13.2.0'");
        mutations.push_back({"cuda_version_drift_to_13_2", mutated, "no_unsupported_cuda_version_drift", ok});
    }
    {
        std::string from = "cuda: '12.8.1'";
        size_t pos = text.find(from);
        bool ok = pos != std::string::npos;
        std::string mutated = text;
        if (ok) mutated.replace(pos, from.size(), "cuda: '12.9.9'");
        mutations.push_back({"cuda_version_unpinned", mutated, "cuda_version_pin", ok});
    }
    {
        std::string from = "'[\"nvcc\", \"cudart\", \"thrust\", \"visual_studio_integration\"]'";
        size_t pos = text.find(from);
        bool ok = pos != std::string::npos;
        std::string mutated = text;
        if (ok) mutated.replace(pos, from.size(), "'[\"nvcc\", \"crt\", \"cudart\", \"thrust\", \"visual_studio_integration\"]'");
        mutations.push_back({"crt_subpackage_reintroduced", mutated, "subpackages_valid_no_crt", ok});
    }
    {
        std::string from = "-B out\\windows-cuda\\build";
        size_t pos = text.find(from);
        bool ok = pos != std::string::npos;
        std::string mutated = text;
        if (ok) mutated.replace(pos, from.size(), "-B build");
        mutations.push_back({"configure_output_path_drift", mutated, "outputs_under_out_windows_cuda", ok});
    }
    {
        std::string mutated = replace_all(replace_all(text, "out\\windows-cuda", "cuda_out_dir"),
                                           "out/windows-cuda", "cuda_out_dir");
        mutations.push_back({"out_windows_cuda_path_count_collapsed", mutated, "outputs_under_out_windows_cuda", true});
    }
    {
        std::string from = "--test-dir out\\windows-cuda\\build";
        size_t pos = text.find(from);
        bool ok = pos != std::string::npos;
        std::string mutated = text;
        if (ok) mutated.replace(pos, from.size(), "--test-dir build");
        mutations.push_back({"test_dir_bare_build_drift", mutated, "outputs_under_out_windows_cuda", ok});
    }
    {
        std::string from = "-B out\\windows-cuda\\build";
        size_t pos = text.find(from);
        bool ok = pos != std::string::npos;
        std::string mutated = text;
        if (ok) mutated.replace(pos, from.size(), "-B out\\build");
        mutations.push_back({"configure_path_drift_to_sibling_out_dir", mutated, "outputs_under_out_windows_cuda", ok});
    }
    {
        std::string from = "-B out\\windows-cuda\\build";
        size_t pos = text.find(from);
        bool ok = pos != std::string::npos;
        std::string mutated = text;
        if (ok) mutated.replace(pos, from.size(), "-B out\\windows-cuda\\build\\..\\..\\evil-build");
        mutations.push_back({"configure_path_parent_relative_escape", mutated, "outputs_under_out_windows_cuda", ok});
    }
    {
        std::string mutated = text + "\n      - name: mask\n        continue-on-error: true\n";
        mutations.push_back({"continue_on_error_masking_injected", mutated, "no_continue_on_error_masking", true});
    }
    {
        std::string mutated = replace_all(text, "cicc.exe", "");
        mutations.push_back({"cicc_diagnostic_removed", mutated, "diagnostics_cicc", true});
    }
    {
        std::string mutated = replace_all(text, "CUDA_PATH", "");
        mutations.push_back({"cuda_path_diagnostic_removed", mutated, "diagnostics_cuda_path", true});
    }
    {
        std::string mutated = replace_all(text, "nvcc.exe", "");
        mutations.push_back({"nvcc_diagnostic_removed", mutated, "diagnostics_nvcc", true});
    }
    {
        std::string mutated = replace_all(text, "ptxas.exe", "");
        mutations.push_back({"ptxas_diagnostic_removed", mutated, "diagnostics_ptxas", true});
    }
    {
        std::string mutated = replace_all(text, "nvcc.profile", "");
        mutations.push_back({"nvcc_profile_diagnostic_removed", mutated, "diagnostics_nvcc_profile", true});
    }
    {
        std::string mutated = replace_all(text, "cl.exe", "");
        mutations.push_back({"cl_exe_diagnostic_removed", mutated, "diagnostics_cl_exe", true});
    }
    {
        std::string mutated = replace_all(text, "sm_89", "");
        mutations.push_back({"sm89_probe_arch_removed", mutated, "diagnostics_sm89_probe", true});
    }
    {
        std::string mutated = replace_all(text, ".cu", "");
        mutations.push_back({"sm89_probe_cu_extension_removed", mutated, "diagnostics_sm89_probe", true});
    }
    {
        std::string probe_line = extract_sm89_probe_invocation(text);
        bool setup_ok = !probe_line.empty() && count_occurrences(text, "-arch=sm_89") == 1 &&
                        count_occurrences(probe_line, "nvcc.exe") == 1;
        std::string mutated = text;
        if (setup_ok) {
            size_t probe_pos = text.find(probe_line);
            size_t nvcc_in_line = probe_line.find("nvcc.exe");
            setup_ok = probe_pos != std::string::npos && nvcc_in_line != std::string::npos;
            if (setup_ok) {
                mutated.erase(probe_pos + nvcc_in_line, std::string("nvcc.exe").size());
            }
        }
        if (setup_ok) {
            for (const auto& other : mutations) {
                if (other.mutated_text == mutated) {
                    std::cerr << "[windows-cuda-workflow-contract] FAIL(mutation-distinctness): "
                                 "'sm89_probe_nvcc_reference_removed' produced text identical to '"
                              << other.label << "'\n";
                    ++failures;
                    break;
                }
            }
        }
        mutations.push_back({"sm89_probe_nvcc_reference_removed", mutated, "diagnostics_sm89_probe", setup_ok});
    }
    {
        std::string from = "Jimver/cuda-toolkit@3d45d157f327c09c04b50ee6ccdea2d9d017ec76";
        size_t pos = text.find(from);
        bool ok = pos != std::string::npos;
        std::string mutated = text;
        if (ok) mutated.replace(pos, from.size(), "Jimver/cuda-toolkit@1111111111111111111111111111111111111111");
        mutations.push_back({"action_revision_tampered", mutated, "immutable_action_revision", ok});
    }
    {
        std::string mutated = replace_all(text, "lbtc_direct_gpu_columns_hook", "");
        mutations.push_back({"hook_test_removed", mutated, "hook_job_mandatory", true});
    }
    {
        std::string from = "\n  windows-cuda:\n    name:";
        size_t pos = text.find(from);
        bool ok = pos != std::string::npos;
        std::string mutated = text;
        if (ok) mutated.replace(pos, from.size(), "\n  windows-cuda:\n    if: false\n    name:");
        mutations.push_back({"hook_job_conditionally_gated", mutated, "hook_job_mandatory", ok});
    }
    {
        std::string from = "      - name: Build (engine, GPU host, kernels, macro fixture, hook test)\n        shell: cmd";
        size_t pos = text.find(from);
        bool ok = pos != std::string::npos;
        std::string mutated = text;
        if (ok) {
            mutated.replace(pos, from.size(),
                "      - name: Build (engine, GPU host, kernels, macro fixture, hook test)\n        if: false\n        shell: cmd");
        }
        mutations.push_back({"build_step_conditionally_masked", mutated, "hook_steps_not_step_level_masked", ok});
    }
    {
        std::string from = "      - name: Test (macro fixture + hook self-install retention)\n        shell: cmd";
        size_t pos = text.find(from);
        bool ok = pos != std::string::npos;
        std::string mutated = text;
        if (ok) {
            mutated.replace(pos, from.size(),
                "      - name: Test (macro fixture + hook self-install retention)\n        if: false\n        shell: cmd");
        }
        mutations.push_back({"test_step_conditionally_masked", mutated, "hook_steps_not_step_level_masked", ok});
    }

    for (const auto& m : mutations) {
        if (!m.setup_ok) {
            std::cerr << "[windows-cuda-workflow-contract] FAIL(mutation-setup): pattern not found for '"
                       << m.label << "'\n";
            ++failures;
            continue;
        }
        std::vector<Check> mutated_checks = validate_workflow_contract(m.mutated_text);
        if (check_passed(mutated_checks, m.must_fail_check)) {
            std::cerr << "[windows-cuda-workflow-contract] FAIL(mutation): '" << m.label
                       << "' did not trip check '" << m.must_fail_check << "'\n";
            ++failures;
        }
    }

    if (failures == 0) {
        std::cout << "[windows-cuda-workflow-contract] PASS: " << live_checks.size()
                   << " live checks ok, " << mutations.size() << " mutation regressions all correctly rejected.\n";
        return 0;
    }

    std::cerr << "[windows-cuda-workflow-contract] " << failures << " failure(s).\n";
    return 1;
}

#ifdef STANDALONE_TEST
int main() {
    return test_windows_cuda_workflow_contract_run();
}
#endif
